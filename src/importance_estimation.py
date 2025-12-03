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
DICTIONARY_LEARNING_SUBSET_SIZE=5000 # full set is 82783
TARGET_IMAGE=""
TARGET_TOKEN="dog"
SEED=28
OUT_DIR="/home/ytllam/xai/xl-vlms/out/gradient_concept"
LOG_FILE=os.path.join(os.path.join(OUT_DIR, f"importance_estimation_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}_{TARGET_TOKEN}.log"))
DICT_ANALYSIS_NAME="decompose_activations_text_grounding_image_grounding"

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

@torch.no_grad()
def single_inference(
    model_class: ImageTextModel,
    device: torch.device,
    logger: Callable = None,
    args: argparse.Namespace = None,
) -> Tuple[Dict[str, Any], Any]: # TODO: get proper type of input
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
    test_item = coco_test_set[4]
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
                func(data=hook_data, args=args, logger=logger)
    
    return hook_data, preprocessed_input

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


if __name__ == "__main__":
    logger = setup_logger(LOG_FILE)
    set_seed(SEED)
    device = get_most_free_gpu() if torch.cuda.is_available() else torch.device("cpu")

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

    logger.info(f"Running inference for single input...")
    # TODO: replace 'test' with img filename
    inference_out_file = f"llava_input_inference_{DATASET_NAME}_{INFERENCE_DATA_SPLIT}_{'test'}"
    inference_args = get_arguments({
        **DEFAULT_ARGS,
        "save_filename": inference_out_file,
        "dataset_size": INFERENCE_SUBSET_SIZE,
        "split": INFERENCE_DATA_SPLIT,
    })
    test_item, preprocessed_test_input = single_inference(
        model_class=llava_model,
        device=device,
        logger=logger,
        args=inference_args,
    )
    # print(test_item)
    # preprocessed test input (type=<class 'transformers.feature_extraction_utils.BatchFeature'>)
    print(f"preprocessed test input (type={type(preprocessed_test_input)})")
    print(preprocessed_test_input)

    # TODO: select token in item's caption
    # TODO: skip if file already found
    logger.info(f"Extracting hidden states for selected token samples...")
    feature_out_file = f"llava_sampled_hidden_states_{DATASET_NAME}_{DICTIONARY_LEARNING_DATA_SPLIT}_{TARGET_TOKEN}"
    concept_dict_mining_args = get_arguments({ # Should match save_features.sh
        **DEFAULT_ARGS,
        "dataset_size": DICTIONARY_LEARNING_SUBSET_SIZE,
        "split": DICTIONARY_LEARNING_DATA_SPLIT,
        "save_dir": OUT_DIR,
        "save_filename": feature_out_file,
        "batch_size": 26,
        "programmatic": True,
    })
    global_hidden_states_for_token = get_hidden_states_of_relevant_samples(
        model_class=llava_model,
        device=device,
        logger=logger,
        args=concept_dict_mining_args,
    )
    # print(concept_dict_for_token)

    # TODO: skip if file already found
    logger.info(f"Learning concept dictionary for extracted features for selected token...")
    concept_dict_out_file = f"llava_concept_dict_{DATASET_NAME}_{DICTIONARY_LEARNING_DATA_SPLIT}_{TARGET_TOKEN}"
    concept_dict_mining_args = get_arguments({ # Should match feature_decomposition.sh
        **DEFAULT_ARGS,
        "dataset_size": DICTIONARY_LEARNING_SUBSET_SIZE,
        "split": DICTIONARY_LEARNING_DATA_SPLIT,
        "features_path": os.path.join(OUT_DIR, "features", f"{HOOK_NAMES[0]}_{feature_out_file}.pth"), # see utils.py save_hidden_states_to_file()
        "save_dir": os.path.join(OUT_DIR, "concept_dicts"),
        "save_filename": concept_dict_out_file, #saves to os.path.join(OUT_DIR, "concept_dicts"), f"{DICT_ANALYSIS_NAME}_{save_filename}.pth")
        "batch_size": 26,
        "programmatic": True,
        "analysis_name": DICT_ANALYSIS_NAME, # generates concepts for the selected token across relevant samples in whole dataset, 
                                             # then grounds each concept textually and visually
        "module_to_decompose": TARGET_FEATURE_MODULES[0][0],
        "num_concepts": [20], # why the hell is nargs="+"??
        "decomposition_method": "snmf",
    })
    log_args(concept_dict_mining_args, logger)
    dict_learning_device = torch.device("cpu")
    dict_learning_model = get_model_class(
        model_name_or_path=MODEL_NAME,
        processor_name=MODEL_NAME,
        device=dict_learning_device,
        logger=logger,
        args=model_setup_args,
    )
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
    data = torch.load(os.path.join(OUT_DIR, "features", f"{HOOK_NAMES[0]}_{inference_out_file}.pth"), map_location="cpu")
    feat = get_feature_matrix(data["hidden_states"], module_name="language_model.model.layers.31", token_idx=None)
    projections = project_representations(
        sample=feat,
        analysis_model=global_concept_dict_for_token["analysis_model"],
        decomposition_type=global_concept_dict_for_token["decomposition_method"],
    )
    # With 1 input sample, we should get (1, n_concepts=20), (1, feature_dims), and 1 respectively
    print(projections.shape, feat.shape, len(data["hidden_states"]))
    print(projections)

    v_X = projections[0]
    # TypeError: unsupported operand type(s) for @: 'numpy.ndarray' and 'Tensor'
    h_recon = v_X @ global_concept_dict_for_token["concepts"]
    print(h_recon)
    h_recon_torch = torch.tensor(h_recon, requires_grad=True)

    # TODO: refactor into reconstruct_differentiable_hidden_state()

    target_token_vocab_idx = get_vocab_idx_of_target_token()
    target_token_first_idx_in_generated_output = get_first_pos_of_token_of_interest() # TODO
    # get_concept_grad_at_target_token_step(
    #     model=llava_model,
    #     test_input=preprocessed_test_input,
    #     token_index=target_token_first_idx_in_generated_output
    #     h_recon_torch=h_recon_torch,
    #     target_id=target_token_vocab_idx,
    # )

exit()

# UNTESTED
def get_concept_grad_at_target_token_step(
    model,
    test_input, # preprocessed_test_input
    token_index, # token_index is the index of the token in the generated tokens (i.e. not including prompt tokens).
                 # this SHOULD be the same index returned by get_first_pos_of_token_of_interest().
    h_recon_torch, # hidden state of test item for target token, projected onto global concept dictionary for target token
    target_id, # vocabulary index of target token. should be from get_vocab_idx_of_target_token()
):

    layer_name = "language_model.model.layers.31"
    target_layer = get_module_by_path(model, layer_name)

    def replace_token_hidden_state(module, input, output):
        # TODO: only patch during the step where the target token is generated
        out = output.clone()
        out[0, :, :] = output[0]   # keep unchanged
        out[0, token_index, :] = h_recon_torch   # patch
        return out

    hook = target_layer.register_forward_hook(replace_token_hidden_state)

    with torch.enable_grad():
        gen_out = model.generate(
            **test_input,
            max_new_tokens=50, # TODO: args.max_new_tokens
            do_sample=False,
            # To get output logits for every token, and at every step
            output_scores=True,
            return_dict_in_generate=True,
            # To get gradients
            use_cache=False,     # needed for gradients
        )

    # extract logit for the target token at the step where it was generated
    target_logit = gen_out.scores[token_index][0, target_id]

    # compute gradient of logit wrt hidden state
    model.zero_grad()
    target_logit.backward()

    grad_h = h_recon_torch.grad.detach().cpu()

    # TODO: compute gradient of logit wrt concept vectors (mult by Z?)
    # TODO: compute gradient * concept activation --> concept importance
    hook.remove()
    return grad_h

# TODO: return groundings with 1) indexes sorted by activations and 2) indexes sorted by importances
# rank_and_ground_concepts()
    # see concept_grounding_visualization.ipynb 
    

# TODO: cut down number of concepts
# TODO (stretch goal): get concept dict for every word in caption
# TODO: maybe add plural version of token(?) but it maaaay change the "concept"
