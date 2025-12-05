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
from models import get_module_by_path
from models.image_text_model import ImageTextModel

from save_features import inference
from analysis import analyse_features
from analysis.feature_decomposition import get_feature_matrix, project_representations
from acronim_server import (get_output_hidden_state_paths, get_uploaded_img_saved_path,
                            get_uploaded_img_dir, get_saved_hidden_states_dir,
                            get_output_concept_dictionary_path,
                            get_saved_concept_dicts_dir,
                            get_llava_model_class, get_dict_model_class,
                            new_log_event, new_event, CAPTIONING_PROMPT)
import time

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
COCO_TRAIN_FULL_SIZE=82783 # full set is 82783
DICTIONARY_LEARNING_MIN_SAMPLE_SIZE=5000
DEFAULT_NUM_CONCEPTS=20
TARGET_IMAGE=""
SEED=28
OUT_DIR="/home/ytllam/xai/xl-vlms/out/gradient_concept"
DICT_ANALYSIS_NAME="decompose_activations_text_grounding_image_grounding"

set_seed(SEED)
device = get_most_free_gpu() if torch.cuda.is_available() else torch.device("cpu")

LOG_FILE=os.path.join(os.path.join(OUT_DIR, f"acronim_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.log"))
logger = setup_logger(LOG_FILE)

llava_model_class = get_llava_model_class()
model = llava_model_class.get_model()

FORCE_RECOMPUTE=False

def get_concept_grad_at_target_token_step(
    model_class,
    test_item, # saved from single_inference()
    token_index, # token_index is the index of the token in the generated tokens (including prompt tokens).
                 # this is the index returned by get_first_pos_of_token_of_interest().
    h_recon_torch, # hidden state of test item for target token, projected onto global concept dictionary for target token
    target_id, # vocabulary index of target token. should be from get_vocab_idx_of_target_token()
    concept_matrix, # shape: [N_concepts, Hidden_Dim] (e.g., [20, 4096])
):
    # free up concept dict model for extra vram
    # del get_dict_model_class().model_

    # model = model_class.get_model()
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
    grad_wrt_concepts = grad_wrt_input @ concept_matrix.T
    # grads shape: [1, 4096]
    # concept_matrix.T shape: [4096, 20]
    # result shape: [1, 20]
    print(f"Concept Gradients shape: {grad_wrt_concepts.shape}")
    print(grad_wrt_concepts)

    hook_handle.remove()
    return grad_wrt_concepts

async def calculate_concept_importance(token_of_interest, uploaded_img_hidden_state_path, uploaded_img_hidden_state, concept_dict,
                                       n_concepts: int=DEFAULT_NUM_CONCEPTS, force_recompute: bool=FORCE_RECOMPUTE):
    try:
        start_time = time.perf_counter()
        yield new_log_event(logger, f"Calculating concept importance...")

        yield new_log_event(logger, f"Projecting input image hidden representation w/r/t selected token onto concept dict...")
        # # From concept_grounding_visualization.ipynb example
        data = torch.load(uploaded_img_hidden_state_path, map_location="cpu")
        test_item = uploaded_img_hidden_state
        feat = get_feature_matrix(data["hidden_states"], module_name="language_model.model.layers.31", token_idx=None)

        projections = project_representations(
            sample=feat,
            analysis_model=concept_dict["analysis_model"],
            decomposition_type=concept_dict["decomposition_method"],
        )

        yield new_log_event(logger, f"Calculating gradients...")
        target_token_vocab_idx = get_vocab_idx_of_target_token(token_of_interest,
                                                            args_token_of_interest_idx=None,
                                                            tokenizer=llava_model_class.get_tokenizer())
        # print(f"model_predictions: '{test_item.get('model_predictions')[0]}'")

        target_token_first_idx_in_entire_output, no_token_found_mask = get_first_pos_of_token_of_interest(
            tokens=test_item.get("hidden_states")[0].get("language_model.model.layers.31")[0],
            pred_tokens=test_item.get("model_output")[0],
            target_token_vocab_idx=target_token_vocab_idx, # index of the target token in the model vocabulary
        )
        n_caption_tokens = test_item.get("model_generated_output")[0].shape[1]
        n_total_tokens = test_item.get("model_output")[0].shape[1]
        n_prompt_tokens = n_total_tokens - n_caption_tokens
        target_token_first_idx_in_caption_output = target_token_first_idx_in_entire_output - n_prompt_tokens

        # print(f"vocab_idx={target_token_vocab_idx}")
        # print(f"target_token_first_idx_in_caption_output={target_token_first_idx_in_caption_output}")

        target_dtype = concept_dict["concepts"].dtype
        v_X = torch.tensor(projections).to(dtype=target_dtype)
        h_recon = v_X @ concept_dict["concepts"]
        h_recon_torch = h_recon.to(dtype=model.dtype, device=model.device).clone().detach().requires_grad_(True)
        # print(f"h_recon_torch.shape={h_recon_torch.shape}")

        # target_token_first_idx_in_generated_output: first index of token in generated output, calculated from first generated token (includes prompt)
        # need to modify token_index by start of caption token offset
        # get offset from -test_item.get("model_output")[0]
        concept_grads = get_concept_grad_at_target_token_step(
            model_class=llava_model_class,
            test_item=test_item,
            token_index=target_token_first_idx_in_entire_output,
            h_recon_torch=h_recon_torch,
            target_id=target_token_vocab_idx,
            concept_matrix=concept_dict["concepts"]
        )
        # If these are zero gradient chain is broken
        # print(f"Non-zero elements: {torch.count_nonzero(concept_grads).item()}")
        # print(f"Max gradient: {concept_grads.abs().max().item()}")
        # print(f"Non-zero elements: {torch.count_nonzero(concept_grads).item()}")

        yield new_log_event(logger, f"Calculating importance scores...")
        concept_activations = v_X.to(device=model.device, dtype=model.dtype) # v_X shape: [1, N_concepts]
        concept_importance_scores = concept_activations * concept_grads
        logger.info(f"Importance scores: {concept_importance_scores}")
        logger.info(f"Activations: {concept_activations}")

        sorted_activations, indices_by_activations = torch.topk(concept_activations, k=n_concepts)
        sorted_importance_scores, indices_by_importance = torch.topk(concept_importance_scores, k=n_concepts)
        
        logger.info(f"Sorted importance scores: {sorted_importance_scores}, indices={indices_by_importance}")
        logger.info(f"Sorted activations: {sorted_activations}, indices={indices_by_activations}")

        for rank, (score, concept_idx) in enumerate(zip(sorted_importance_scores[0], indices_by_importance[0])):
            print(f"#{rank+1}: Concept {concept_idx}."
                + f"\nImage groundings ={concept_dict['image_grounding_paths'][concept_idx][:10]}" 
                + f"\nText groundings ={concept_dict['text_grounding'][concept_idx][:10]}..."
                + f"\n(Importance {score.item():.4f}; Activation: {concept_activations[0, concept_idx]:.2f}; Gradient: {concept_grads[0, concept_idx]:.4f})")

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        yield new_log_event(logger, f"Calculated concept importance scores for token={token_of_interest} in time={elapsed_time:.6f}s'")
        yield new_event(event_type="return", data={
            "activations": concept_activations.tolist()[0],
            "importance_scores": concept_importance_scores.tolist()[0],
            "indices_by_importance": indices_by_importance.tolist()[0],
            "indices_by_activations": indices_by_activations.tolist()[0],
            "text_groundings": concept_dict['text_grounding'],
            "image_grounding_paths": concept_dict['image_grounding_paths'],
        })
    except Exception as e:
        if "CUDA out of memory" in str(e):
            yield new_event(event_type="error", data="CUDA ran out of memory. Try using a smaller sampling inference batch size.")
        else:
            yield new_event(event_type="error", data=f"Ran into an error when calculating concept importance wrt token={token_of_interest}. Try again or try a different token.\n{str(e)}")
        return