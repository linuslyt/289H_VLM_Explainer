"""
Standalone script for testing concept importance estimation pipeline.
API server version is located in the acronim_server module.
"""

import argparse
import os
from datetime import datetime
import torch
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Tuple
import copy

from datasets import get_dataset_loader
from helpers.arguments import get_arguments
from helpers.logger import log_args, setup_logger
from helpers.utils import (clear_forward_hooks, clear_hooks_variables, get_most_free_gpu,
                           get_vocab_idx_of_target_token, get_first_pos_of_token_of_interest,
                           set_seed, setup_hooks, update_dict_of_list)
from models import get_model_class, get_module_by_path
from models.image_text_model import ImageTextModel

from save_features import inference
from analysis import analyse_features
from analysis.feature_decomposition import get_feature_matrix, project_representations
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME="llava-hf/llava-1.5-7b-hf"
TARGET_FEATURE_MODULES=[["language_model.model.layers.31"]]
# This hook extracts the hidden state from target module, 
# starting from the first position in the predicted caption where the target token is generated.
# This hook is used for concept dictionary learning, thus we must use this hook/hidden state extraction
# method so our projection onto the concept dictionary for importance estimation is valid.
HOOK_NAMES=["save_hidden_states_for_token_of_interest"]
DATASET_NAME="coco"
DATA_DIR="/media/data/ytllam/coco"
ANNOTATION_FILE="karpathy/dataset_coco.json"

INFERENCE_DATA_SPLIT="test"
INFERENCE_SUBSET_SIZE=5000
DICTIONARY_LEARNING_DATA_SPLIT="train"
DICTIONARY_LEARNING_SUBSET_SIZE=82783 # full set is 82783
NUM_CONCEPTS=20
TARGET_IMAGE=""
TARGET_TOKEN="motorcycle" # NOTE: some tokens will cause dictionary learning to fail, e.g. "dogs" for some reason. not entirely sure why.
SEED=28
OUT_DIR="/home/ytllam/xai/xl-vlms/out/gradient_concept"
LOG_FILE=os.path.join(os.path.join(OUT_DIR, f"importance_estimation_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}_{TARGET_TOKEN}.log"))
DICT_ANALYSIS_NAME="decompose_activations_text_grounding_image_grounding"
device = get_most_free_gpu() if torch.cuda.is_available() else torch.device("cpu")

DEFAULT_ARGS = {
    "model_name_or_path": MODEL_NAME,
    "dataset_name": DATASET_NAME,
    "dataset_size": "-1", # should be overridden
    "data_dir": DATA_DIR,
    "annotation_file": ANNOTATION_FILE,
    "split": "test", # should be overridden
    "hook_names": HOOK_NAMES,
    "modules_to_hook": TARGET_FEATURE_MODULES,
    # used to filter dataset to images where the token of interest exists in caption.
    # we can set this to True so we only sample from those images for which we have a concept dictionary precomputed for.
    "select_token_of_interest_samples": True,
    "token_of_interest": TARGET_TOKEN,
    "save_dir": OUT_DIR,
    "save_filename": "",
    "seed": SEED,
    "processor_name": MODEL_NAME,
    "generation_mode": True,
    "exact_match_modules_to_hook": True,
    "save_only_generated_tokens": True,
    "batch_size": 1,
}

FORCE_RECOMPUTE=True # TODO: make script argument

@torch.no_grad()
# This runs inference with a target token and module, and registers a forward hook so that the hidden state
# at the target module at the position after that token is first generated is saved.
# TODO: create single inference function for initial user input that doesn't save anything, but uses the same seed, just so
#       the user can select a token from the caption.
def single_inference(
    model_class: ImageTextModel,
    device: torch.device,
    logger: Callable = None,
    args: argparse.Namespace = None,
) -> Dict[str, Any]: # TODO: get proper type of input
    log_args(args, logger)
    # hook_return_functions: run here manually after model inference
    # hook_postprocessing_functions: run outside of inference loop, e.g. to save hidden states
    hook_return_functions, hook_postprocessing_functions = setup_hooks(
        model=model_class.model_,
        modules_to_hook=args.modules_to_hook,
        hook_names=args.hook_names,
        tokenizer=model_class.get_tokenizer(),
        logger=logger,
        args=args,
    )

    loader = get_dataset_loader(
        dataset_name=args.dataset_name, logger=logger, args=args,
    )

    logger.info(f"Generating caption for image {TARGET_IMAGE}")

    # print(f"Dataset type: {loader.dataset}")
    coco_test_set = loader.dataset
    test_item = coco_test_set[0]
    # e.g. for index 4: 
    # {'img_id': 'COCO_val2014_000000173081',
    #  'instruction': '\nProvide a one-sentence caption for the provided image.', 
    #  'response': 'Multiple beds stand on hardwood floors in a simple room.', 
    #  'image': '/media/data/ytllam/coco/val2014/COCO_val2014_000000173081.jpg', 
    #  'targets': 'Multiple beds stand on hardwood floors in a simple room.$$Four empty beds are shown made up inside a room.$$there are four small beds all in the same room$$Several neatly  made beds in an empty room.$$A room with four beds placed in various positions.', 
    #  'text': '\nProvide a one-sentence caption for the provided image.'}
    # `targets` is used for object detection
    # print(test_item)
    input_instance = model_class.preprocessor(
        instruction=test_item["text"],
        image_file=test_item["image"],
        response="", # not provided bc we're doing generation
        generation_mode=args.generation_mode
    )
    preprocessed_input = copy.deepcopy(input_instance)
    test_item["preprocessed_input"] = preprocessed_input

    model = model_class.get_model()
    out = model.generate(
        **input_instance, max_new_tokens=args.max_new_tokens, do_sample=False
    )
    
    test_item["model_output"] = out
    
    input_len = (
        input_instance["input_ids"].shape[1]
        if input_instance["input_ids"].ndim > 1
        else input_instance["input_ids"].shape[0]
    )

    test_item["model_generated_output"] = out[:, input_len:]
    test_item["model_predictions"] = model_class.get_tokenizer().batch_decode(
        out[:, input_len:], skip_special_tokens=True
    )

    if hook_return_functions is not None:
        for func in hook_return_functions:
            if func is not None:
                hook_output = func(**test_item)
                if hook_output:
                    test_item.update(hook_output)

    hook_data = update_dict_of_list(test_item, data={}, is_batched=False)
    clear_hooks_variables()

    clear_forward_hooks(llava_model.model_) # reset hooks so model can be reused
    # Save features to file
    if hook_postprocessing_functions is not None:
        for func in hook_postprocessing_functions:
            if func is not None:
                func(data=hook_data, 
                     data_keys=['img_id', 'instruction', 'response', 'image',
                                'targets', 'text', 'preprocessed_input', 
                                'model_output', 'model_generated_output', 'model_predictions',
                                'token_of_interest_mask', 'hidden_states'], # save all fields, not just hidden_states
                     args=args, logger=logger)
    
    return hook_data

@torch.no_grad()
def get_hidden_states_of_relevant_samples(
    model_class: ImageTextModel,
    device: torch.device,
    logger: Callable = None,
    args: argparse.Namespace = None,
):
    hook_return_functions, hook_postprocessing_functions = setup_hooks(
        model=model_class.model_,
        modules_to_hook=args.modules_to_hook,
        hook_names=args.hook_names,
        tokenizer=model_class.get_tokenizer(),
        logger=logger,
        args=args,
    )
    loader = get_dataset_loader(
        dataset_name=args.dataset_name, logger=logger, args=args,
    )

    hook_data = inference(
        loader=loader,
        model_class=model_class,
        hook_return_functions=hook_return_functions,
        device=device,
        logger=logger,
        args=args,
    )

    clear_forward_hooks(llava_model.model_) # reset hooks so model can be reused

    # Save features to file
    if hook_postprocessing_functions is not None:
        for func in hook_postprocessing_functions:
            if func is not None:
                func(data=hook_data, args=args, logger=logger)
    
    return hook_data


def get_concept_grad_at_target_token_step(
    model_class,
    test_item, # saved from single_inference()
    token_index, # token_index is the index of the token in the generated tokens (including prompt tokens).
                 # this is the index returned by get_first_pos_of_token_of_interest().
    h_recon_torch, # hidden state of test item for target token, projected onto global concept dictionary for target token
    target_id, # vocabulary index of target token. should be from get_vocab_idx_of_target_token()
    concept_matrix, # shape: [N_concepts, Hidden_Dim] (e.g., [20, 4096])
):
    model = model_class.get_model()
    layer_name = "language_model.model.layers.31"
    target_layer = get_module_by_path(model, layer_name)
    
    # Load full token sequence predicted previously. 
    full_sequence = test_item["model_output"][0] # shape: [1, seq_len]
    # print(f"full_sequence.shape={full_sequence.shape}")

    # Context up to, i.e. excluding, target token
    # If "motorcycle" is at index K, we input tokens [0 ... K-1]
    # The model will then try to predict token K ("motorcycle")
    input_up_to_target_token = full_sequence[:, :token_index]

    # Call model() for last forward pass to generate target token.
    # Patch hidden state with differentiable state reconstructed from projection onto concept dictionary.
    def replace_token_hidden_state(module, input, output):
        # output is a tuple: (hidden_states, self_attn, present_kv)
        hidden_states = output[0] # tensor [batch, seq, dim]
        rest_output = output[1:]

        # clone so autograd works as expected
        new_hidden = hidden_states.clone()

        # Patch the hidden state generated in the last timestep before the token of interest
        # If the target word translates to multiple subtokens, patch with the hidden state before the target's first subtoken
        new_hidden[0, -1, :] = h_recon_torch

        # return same structure transformer expects
        return (new_hidden, *rest_output)


    hook_handle = target_layer.register_forward_hook(replace_token_hidden_state)

    with torch.enable_grad():
        outputs = model(
            input_ids=input_up_to_target_token,
            output_hidden_states=False,
            use_cache=False,
        )
    
    # Extract logits for target token
    logits = outputs.logits[:, -1, :]  # shape [1, vocab]

    # If target_id is a tensor/list, i.e. the target word translates to multiple subtokens, use the position of the first one
    if hasattr(target_id, '__len__') and len(target_id) > 1:
        scalar_target_id = target_id[0]
    else:
        scalar_target_id = target_id
        
    target_logit = logits[0, scalar_target_id]

    print("target_logit:", target_logit, "requires_grad:", getattr(target_logit, "requires_grad", None))
    print("h_recon_torch.requires_grad:", h_recon_torch.requires_grad)

    # Clean up previous gradients
    model.zero_grad()
    if h_recon_torch.grad is not None:
        h_recon_torch.grad.zero_()

    # Backprop to calculate gradient for target token logit
    target_logit.backward()
    grad_wrt_input = h_recon_torch.grad

    # Compute Gradient w.r.t Concept Activations
    concept_matrix = concept_matrix.to(device=model.device, dtype=grad_wrt_input.dtype) # ensure it's on the same device/dtype=fp16
    grad_wrt_concepts = grad_wrt_input @ concept_matrix.T  # TODO: check if need transpose
    # grads shape: [1, 4096]
    # concept_matrix.T shape: [4096, 20]
    # result shape: [1, 20]
    print(f"Concept Gradients shape: {grad_wrt_concepts.shape}")
    print(grad_wrt_concepts)

    hook_handle.remove()
    return grad_wrt_concepts

if __name__ == "__main__":
    logger = setup_logger(LOG_FILE)
    set_seed(SEED)

    logger.info(f"Loading model={MODEL_NAME} on device={device}, gpu={torch.cuda.is_available()}...")
    model_setup_args = get_arguments({
        **DEFAULT_ARGS,
    })
    llava_model = get_model_class(
        model_name_or_path=MODEL_NAME,
        processor_name=MODEL_NAME,
        device=device,
        logger=logger,
        args=model_setup_args, # larger arg dict not needed for model setup
    )

    if FORCE_RECOMPUTE:
        logger.info(f"FORCE_RECOMPUTE={FORCE_RECOMPUTE}, recomputing all files")

    # TODO: replace 'INFERENCE_DATA_SPLIT' with img filename
    # TODO: add call for single inference without hooks. save preprocessed_test_input from that run along with output caption.
    #       user will select target token from that first inference
    test_item_filename = f"llava_input_inference_{DATASET_NAME}_{INFERENCE_DATA_SPLIT}_{TARGET_TOKEN}"
    inference_args = get_arguments({
        **DEFAULT_ARGS,
        "save_filename": test_item_filename,
        "dataset_size": INFERENCE_SUBSET_SIZE,
        "split": INFERENCE_DATA_SPLIT,
        "token_of_interest": TARGET_TOKEN,
    })
    test_item_final_path = os.path.join(OUT_DIR, "features", f"{HOOK_NAMES[0]}_{test_item_filename}.pth")
    if os.path.exists(test_item_final_path) and not FORCE_RECOMPUTE:
        logger.info(f"Loaded test item from {test_item_final_path}")
        test_item = torch.load(test_item_final_path)
    else:
        logger.info(f"Running inference for single input...")
        test_item = single_inference(
            model_class=llava_model,
            device=device,
            logger=logger,
            args=inference_args,
        )
    preprocessed_test_input = test_item["preprocessed_input"]
    # print(test_item)
    # print(f"preprocessed test input (type={type(preprocessed_test_input)})")
    # preprocessed test input (type=<class 'transformers.feature_extraction_utils.BatchFeature'>)
    # print(preprocessed_test_input)

    # TODO: select token in item's caption
    sampled_hidden_states_filename = f"llava_sampled_hidden_states_{DATASET_NAME}_{DICTIONARY_LEARNING_DATA_SPLIT}_{TARGET_TOKEN}"
    sampled_hidden_states_final_path = os.path.join(OUT_DIR, "features", f"{HOOK_NAMES[0]}_{sampled_hidden_states_filename}.pth")
    if os.path.exists(sampled_hidden_states_final_path) and not FORCE_RECOMPUTE:
        logger.info(f"Loaded sample hidden states from {sampled_hidden_states_final_path}")
        sampled_hidden_states = torch.load(sampled_hidden_states_final_path)
    else:
        logger.info(f"Extracting hidden states for selected token samples...")
        hidden_state_sampling_args = get_arguments({ # Should match save_features.sh
            **DEFAULT_ARGS,
            "dataset_size": DICTIONARY_LEARNING_SUBSET_SIZE,
            "split": DICTIONARY_LEARNING_DATA_SPLIT,
            "save_dir": OUT_DIR,
            "save_filename": sampled_hidden_states_filename,
            "batch_size": 26,
            "programmatic": True,
        })
        sampled_hidden_states = get_hidden_states_of_relevant_samples(
            model_class=llava_model,
            device=device,
            logger=logger,
            args=hidden_state_sampling_args,
        )
    # print(concept_dict_for_token)

    concept_dict_filename = f"llava_concept_dict_{DATASET_NAME}_{DICTIONARY_LEARNING_DATA_SPLIT}_{TARGET_TOKEN}"
    concept_dict_final_path = os.path.join(OUT_DIR, "concept_dicts", f"{DICT_ANALYSIS_NAME}_{concept_dict_filename}.pth")
    dict_learning_device = torch.device("cpu")
    dict_learning_model = get_model_class(
        model_name_or_path=MODEL_NAME,
        processor_name=MODEL_NAME,
        device=dict_learning_device,
        logger=logger,
        args=model_setup_args,
    )
    if os.path.exists(concept_dict_final_path) and not FORCE_RECOMPUTE:
        logger.info(f"Loaded concept dict from {concept_dict_final_path}")
        global_concept_dict_for_token = torch.load(concept_dict_final_path)
    else:
        # /home/ytllam/xai/xl-vlms/out/gradient_concept/concept_dicts/decompose_activations_text_grounding_image_grounding_concept_dict_filename
        logger.info(f"No concept dict found at {concept_dict_final_path}")
        logger.info(f"Learning concept dictionary for extracted features for selected token...")
        concept_dict_mining_args = get_arguments({ # Should match feature_decomposition.sh
            **DEFAULT_ARGS,
            "dataset_size": DICTIONARY_LEARNING_SUBSET_SIZE,
            "split": DICTIONARY_LEARNING_DATA_SPLIT,
            "features_path": sampled_hidden_states_final_path, # see utils.py save_hidden_states_to_file()
            "save_dir": os.path.join(OUT_DIR, "concept_dicts"),
            "save_filename": concept_dict_filename, #saves to os.path.join(OUT_DIR, "concept_dicts"), f"{DICT_ANALYSIS_NAME}_{save_filename}.pth")
            "batch_size": 26,
            "programmatic": True,
            "analysis_name": DICT_ANALYSIS_NAME, # generates concepts for the selected token across relevant samples in whole dataset, 
                                                # then grounds each concept textually and visually
            "module_to_decompose": TARGET_FEATURE_MODULES[0][0],
            "num_concepts": [NUM_CONCEPTS], # why the hell is nargs="+"??
            "decomposition_method": "snmf",
        })
        log_args(concept_dict_mining_args, logger)
        global_concept_dict_for_token = analyse_features(
            analysis_name=concept_dict_mining_args.analysis_name,
            logger=logger,
            model_class=dict_learning_model,
            device=dict_learning_device,
            args=concept_dict_mining_args,
        )
    print(global_concept_dict_for_token.keys())
    # analyse_features returns a dict with dict_keys(['concepts', 'activations',
    #   'decomposition_method', 'text_grounding', 'image_grounding_paths', 'analysis_model'])
    # d["analysis_model"] = decomposition_model used for smnf learning. this can be used to project any new features
    # onto the concept dictionary.

    logger.info(f"Projecting test instance hidden representation w/r/t selected token onto concept dict...")
    # From concept_grounding_visualization.ipynb example
    data = torch.load(test_item_final_path, map_location="cpu")
    feat = get_feature_matrix(data["hidden_states"], module_name="language_model.model.layers.31", token_idx=None)
    projections = project_representations(
        sample=feat,
        analysis_model=global_concept_dict_for_token["analysis_model"],
        decomposition_type=global_concept_dict_for_token["decomposition_method"],
    )
    # With 1 input sample, we should get (1, n_concepts=20), (1, feature_dims), and 1 respectively
    print(projections.shape, feat.shape, len(data["hidden_states"]))
    print(projections)

    # TODO: refactor into reconstruct_differentiable_hidden_state()
    target_token_vocab_idx = get_vocab_idx_of_target_token(TARGET_TOKEN, # TODO: args 
                                                           args_token_of_interest_idx=None,
                                                           tokenizer=llava_model.get_tokenizer())
    # print(test_item)
    # print(type(test_item))
    print(f"model_predictions: '{test_item.get('model_predictions')[0]}'")
    target_token_first_idx_in_entire_output, no_token_found_mask = get_first_pos_of_token_of_interest(
        tokens=test_item.get("hidden_states")[0].get("language_model.model.layers.31")[0],
        pred_tokens=test_item.get("model_output")[0],
        target_token_vocab_idx=target_token_vocab_idx, # index of the target token in the model vocabulary
    )
    n_caption_tokens = test_item.get("model_generated_output")[0].shape[1]
    n_total_tokens = test_item.get("model_output")[0].shape[1]
    n_prompt_tokens = n_total_tokens - n_caption_tokens
    target_token_first_idx_in_caption_output = target_token_first_idx_in_entire_output - n_prompt_tokens

    # TODO: won't need this after dynamic token selection, but need to handle cases where target token is not in the generated output
    print(f"vocab_idx={target_token_vocab_idx}")
    print(f"target_token_first_idx_in_caption_output={target_token_first_idx_in_caption_output}")
    # exit()

    v_X = torch.tensor(projections)
    h_recon = v_X @ global_concept_dict_for_token["concepts"]
    h_recon_torch = h_recon.to(dtype=llava_model.get_model().dtype, device=llava_model.get_model().device).clone().detach().requires_grad_(True)
    torch.save(h_recon_torch, "motorcycle_recon.pth")
    print(f"h_recon_torch.shape={h_recon_torch.shape}")


    # target_token_first_idx_in_generated_output: first index of token in generated output, calculated from first generated token (includes prompt)
    # need to modify token_index by start of caption token offset
    # get offset from -test_item.get("model_output")[0]
    concept_grads = get_concept_grad_at_target_token_step(
        model_class=llava_model,
        test_item=test_item,
        token_index=target_token_first_idx_in_entire_output,
        h_recon_torch=h_recon_torch,
        target_id=target_token_vocab_idx,
        concept_matrix=global_concept_dict_for_token["concepts"]
    )
    print(concept_grads)
    print(concept_grads.shape)
    # If these are zero gradient chain is broken
    print(f"Non-zero elements: {torch.count_nonzero(concept_grads).item()}")
    print(f"Max gradient: {concept_grads.abs().max().item()}")
    print(f"Non-zero elements: {torch.count_nonzero(concept_grads).item()}")
    
    torch.save(concept_grads, "motorcycle_grad.pth")
    concept_activations = v_X.to(device=llava_model.get_model().device, dtype=llava_model.get_model().dtype) # v_X shape: [1, N_concepts]
    concept_importance_scores = concept_activations * concept_grads
    top_scores, top_indices = torch.topk(concept_importance_scores, k=NUM_CONCEPTS) # if we match k=#concepts we should get ranking of all concepts

    # TODO: split into rankings for both positive and negative contributors.
    for rank, (score, concept_idx) in enumerate(zip(top_scores[0], top_indices[0])):
        print(f"#{rank+1}: Concept {concept_idx}."
              + f"\nImage groundings ={global_concept_dict_for_token['image_grounding_paths'][concept_idx][:10]}" 
              + f"\nText groundings ={global_concept_dict_for_token['text_grounding'][concept_idx][:10]}..."
              + f"\n(Importance {score.item():.4f}; Activation: {concept_activations[0, concept_idx]:.2f}; Gradient: {concept_grads[0, concept_idx]:.4f})")