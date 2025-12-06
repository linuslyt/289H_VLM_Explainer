import argparse
import os
from datetime import datetime
import torch
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Tuple, Union
import copy

from datasets import get_dataset_loader
from helpers.arguments import get_arguments
from helpers.logger import log_args, setup_logger
from helpers.utils import (clear_forward_hooks, clear_hooks_variables, get_most_free_gpu,
                           get_vocab_idx_of_target_token, get_first_pos_of_token_of_interest,
                           set_seed, setup_hooks, update_dict_of_list)
from models import get_module_by_path
from models.image_text_model import ImageTextModel
import numpy as np

from save_features import inference
from analysis import analyse_features
from analysis.feature_decomposition import get_feature_matrix, project_representations
from acronim_server import (get_output_hidden_state_paths, get_uploaded_img_saved_path,
                            get_uploaded_img_dir, get_saved_hidden_states_dir,
                            get_output_concept_dictionary_path,
                            get_saved_concept_dicts_dir,
                            get_llava_model_class, get_dict_model_class,
                            new_log_event, new_event, CAPTIONING_PROMPT)
from sklearn.metrics import auc  # Renamed to avoid confusion
import time

OUT_DIR="/home/ytllam/xai/xl-vlms/out/gradient_concept"
LOG_FILE=os.path.join(os.path.join(OUT_DIR, f"acronim_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.log"))
logger = setup_logger(LOG_FILE)

TARGET_LAYER_PATH = "language_model.model.layers.31"
llava_model_class = get_llava_model_class()
model = llava_model_class.get_model()

@torch.no_grad()
def get_metric_with_patched_state(
    test_item: Dict[str, Any],
    token_index: int,
    h_recon: torch.Tensor,
    target_id: int,
    metric_mode: str = "logit" # Options: "logit" or "probability"
):
    """
    Runs a forward pass patching the hidden state at the last step 
    and returns the metric (logit or probability) of the target token.
    """
    target_layer = get_module_by_path(model, TARGET_LAYER_PATH)
    
    # 1. Prepare Inputs
    # Load full token sequence predicted previously. 
    full_sequence = test_item["model_output"][0] # shape: [1, seq_len]
    
    # Context up to, i.e. excluding, target token. 
    # The model predicts the target token based on this input.
    input_ids = full_sequence[:, :token_index]

    # Patch hidden state with differentiable state reconstructed from projection onto concept dictionary.
    def replace_token_hidden_state(module, input, output):
        # output is a tuple: (hidden_states, self_attn, present_kv)
        hidden_states = output[0] # tensor [batch, seq, dim]
        rest_output = output[1:]

        # clone so autograd works as expected
        new_hidden = hidden_states.clone()

        # Patch the hidden state generated in the last timestep before the token of interest
        # If the target word translates to multiple subtokens, patch with the hidden state before the target's first subtoken
        new_hidden[0, -1, :] = h_recon

        # return same structure transformer expects
        return (new_hidden, *rest_output)

    # 3. Forward Pass
    hook_handle = target_layer.register_forward_hook(replace_token_hidden_state)
    
    try:
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=False,
            use_cache=False,
        )
        
        # 4. Get Metric
        final_token_logits = outputs.logits[:, -1, :] 
        
        # Handle multi-token ids (take first subtoken)
        if hasattr(target_id, '__len__') and len(target_id) > 1:
            target_logit_id = target_id[0]
        else:
            target_logit_id = target_id
        
        if metric_mode == "logit":
            # Return raw logit (Evidence)
            return final_token_logits[0, target_logit_id].item()
        else:
            # Return Softmax Probability (Confidence)
            probs = torch.softmax(final_token_logits, dim=-1)
            return probs[0, target_logit_id].item()
            
    finally:
        hook_handle.remove()

def compute_c_deletion_metrics(
    test_item: Dict[str, Any],
    token_index: int,
    target_id: int,
    concept_matrix: torch.Tensor,
    activations: torch.Tensor,
    ranking_scores: torch.Tensor,
    metric_name: str = "Method",
    metric_mode: str = "logit" 
) -> Tuple[List[float], float]:
    """
    Computes the trajectory and AUC for C-Deletion.
    
    Args:
        ranking_scores: The scores used to rank concepts (e.g., Importance Scores or Random).
        metric_mode: "logit" (recommended) or "probability".
    """
    device = model.device
    dtype = model.dtype
    
    # Ensure inputs are on correct device
    concept_matrix = concept_matrix.to(device=device, dtype=dtype)
    activations = activations.to(device=device, dtype=dtype)
    ranking_scores = ranking_scores.to(device=device, dtype=dtype)

    # 1. Sort concepts by score (Descending: Delete most important first)
    sorted_indices = torch.argsort(ranking_scores[0], descending=True)
    
    # 2. Setup
    current_activations = activations.clone()
    raw_trajectory = []
    
    # 3. Baseline (No deletion)
    h_baseline = current_activations @ concept_matrix
    val_base = get_metric_with_patched_state(test_item, token_index, h_baseline, target_id, metric_mode)
    raw_trajectory.append(val_base)

    # 4. Deletion Loop
    for idx_to_remove in sorted_indices:
        # Zero out activation
        current_activations[0, idx_to_remove] = 0.0
        
        # Reconstruct
        h_modified = current_activations @ concept_matrix
        
        # Measure
        val = get_metric_with_patched_state(test_item, token_index, h_modified, target_id, metric_mode)
        raw_trajectory.append(val)

    # 5. Normalize Curve 
    if metric_mode == "probability":
        # Relative drop from baseline
        if val_base > 0:
            norm_trajectory = [p / val_base for p in raw_trajectory]
        else:
            norm_trajectory = [0.0 for _ in raw_trajectory]
    else:
        # Logit Mode: Return RAW values. 
        # AUC calculated on raw logits represents "Integral of Evidence".
        # Comparing Ours vs Random is still valid: Ours should drop deeper (more negative) than Random.
        norm_trajectory = raw_trajectory

    # 6. Calculate AUC using sklearn
    n_steps = len(norm_trajectory)
    # x goes from 0/N to N/N (e.g., 0.0, 0.1, 0.2 ... 1.0)
    x_axis = np.linspace(0, 1, n_steps)
    
    # Compute AUC
    auc_val = auc(x_axis, norm_trajectory)
    
    return norm_trajectory, auc_val

async def run_c_deletion_pipeline(
    token_of_interest: str,
    uploaded_img_hidden_state: Dict[str, Any], # The 'test_item'
    concept_dict: Dict[str, Any],
    concept_activations: Union[torch.Tensor, List[float]], 
    concept_importance_scores: Union[torch.Tensor, List[float]] 
):
    """
    Async generator that runs C-Deletion analysis for a single sample.
    Compares the calculated importance scores against a random baseline.
    """
    try:
        start_time = time.perf_counter()
        yield new_log_event(logger, f"Starting C-Deletion faithfulness evaluation for token='{token_of_interest}'...")

        # Robust Tensor Conversion
        # Handles cases where input is List (needs wrapping) or Tensor (needs .to(device))
        if isinstance(concept_activations, list):
            concept_activations = torch.tensor([concept_activations], device=model.device, dtype=model.dtype)
        else:
            concept_activations = concept_activations.to(device=model.device, dtype=model.dtype)

        if isinstance(concept_importance_scores, list):
            concept_importance_scores = torch.tensor([concept_importance_scores], device=model.device, dtype=model.dtype)
        else:
            concept_importance_scores = concept_importance_scores.to(device=model.device, dtype=model.dtype)

        # 1. Setup indices and target ID
        target_token_vocab_idx = get_vocab_idx_of_target_token(
            token_of_interest,
            args_token_of_interest_idx=None,
            tokenizer=llava_model_class.get_tokenizer()
        )

        # Recalculate token position
        target_token_first_idx_in_entire_output, _ = get_first_pos_of_token_of_interest(
            tokens=uploaded_img_hidden_state.get("hidden_states")[0].get(TARGET_LAYER_PATH)[0],
            pred_tokens=uploaded_img_hidden_state.get("model_output")[0],
            target_token_vocab_idx=target_token_vocab_idx, 
        )
        
        # 2. Run C-Deletion (Our Method)
        yield new_log_event(logger, f"Evaluating deletion curve for Ours (Gradient * Activation)...")
        traj_ours, auc_ours = compute_c_deletion_metrics(
            test_item=uploaded_img_hidden_state,
            token_index=target_token_first_idx_in_entire_output,
            target_id=target_token_vocab_idx,
            concept_matrix=concept_dict["concepts"],
            activations=concept_activations,
            ranking_scores=concept_importance_scores,
            metric_name="Ours",
            metric_mode="logit" # Use Logits to avoid Softmax competition artifacts
        )
        yield new_log_event(logger, f"Ours AUC: {auc_ours:.4f}")

        # 3. Run C-Deletion (Random Baseline)
        yield new_log_event(logger, f"Evaluating deletion curve for Random Baseline...")
        # Generate random scores on same device
        random_scores = torch.rand_like(concept_importance_scores)
        
        traj_rand, auc_rand = compute_c_deletion_metrics(
            test_item=uploaded_img_hidden_state,
            token_index=target_token_first_idx_in_entire_output,
            target_id=target_token_vocab_idx,
            concept_matrix=concept_dict["concepts"],
            activations=concept_activations,
            ranking_scores=random_scores,
            metric_name="Random",
            metric_mode="logit"
        )
        yield new_log_event(logger, f"Random AUC: {auc_rand:.4f}")

        # 4. Compare
        is_faithful = auc_ours < auc_rand
        verdict = "FAITHFUL (Drop > Random)" if is_faithful else "UNFAITHFUL (Drop <= Random)"
        yield new_log_event(logger, f"Evaluation Result: {verdict}")

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        
        # 5. Return Results
        results = {
            "token": token_of_interest,
            "auc_ours": auc_ours,
            "auc_random": auc_rand,
            "trajectory_ours": traj_ours,
            "trajectory_random": traj_rand,
            "is_faithful": bool(is_faithful),
            "elapsed_time": elapsed_time
        }
        
        yield new_event(event_type="return", data=results)

    except Exception as e:
        logger.info(f"Error occured: {str(e)}")
        yield new_event(event_type="error", data=f"Error in C-Deletion pipeline: {str(e)}")
        return