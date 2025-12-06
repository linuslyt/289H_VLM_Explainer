# import argparse
# import os
# from datetime import datetime
# import torch
# from tqdm import tqdm
# from typing import Any, Callable, Dict, List, Tuple, Union
# import copy

# from datasets import get_dataset_loader
# from helpers.arguments import get_arguments
# from helpers.logger import log_args, setup_logger
# from helpers.utils import (clear_forward_hooks, clear_hooks_variables, get_most_free_gpu,
#                            get_vocab_idx_of_target_token, get_first_pos_of_token_of_interest,
#                            set_seed, setup_hooks, update_dict_of_list)
# from models import get_module_by_path
# from models.image_text_model import ImageTextModel
# import numpy as np

# from save_features import inference
# from analysis import analyse_features
# from analysis.feature_decomposition import get_feature_matrix, project_representations
# from acronim_server import (get_output_hidden_state_paths, get_uploaded_img_saved_path,
#                             get_uploaded_img_dir, get_saved_hidden_states_dir,
#                             get_output_concept_dictionary_path,
#                             get_saved_concept_dicts_dir,
#                             get_llava_model_class, get_dict_model_class,
#                             new_log_event, new_event, CAPTIONING_PROMPT)
# from sklearn.metrics import auc  # Renamed to avoid confusion
# import time

# OUT_DIR="/home/ytllam/xai/xl-vlms/out/gradient_concept"
# LOG_FILE=os.path.join(os.path.join(OUT_DIR, f"acronim_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.log"))
# logger = setup_logger(LOG_FILE)

# TARGET_LAYER_PATH = "language_model.model.layers.31"
# llava_model_class = get_llava_model_class()
# model = llava_model_class.get_model()


import argparse
import os
from datetime import datetime
import torch
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Tuple, Union
import copy
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import auc
import time

from helpers.logger import setup_logger
from helpers.utils import (get_vocab_idx_of_target_token, get_first_pos_of_token_of_interest)
from models import get_module_by_path
from acronim_server import (get_llava_model_class, new_log_event, new_event)

# Setup Logger
OUT_DIR = "/home/ytllam/xai/xl-vlms/out/gradient_concept"
LOG_FILE = os.path.join(os.path.join(OUT_DIR, f"acronim_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.log"))
logger = setup_logger(LOG_FILE)

# Model Setup
TARGET_LAYER_PATH = "language_model.model.layers.31"
llava_model_class = get_llava_model_class()
model = llava_model_class.get_model()

# -----------------------------------------------------------------------------
# HELPER: Forward Pass with Patching
# -----------------------------------------------------------------------------

@torch.no_grad()
def get_metric_with_patched_state(
    test_item: Dict[str, Any],
    token_index: int,
    h_recon: torch.Tensor,
    target_id: int,
    metric_mode: str = "logit" # Options: "logit" (Recommended) or "probability"
):
    """
    Runs a forward pass patching the hidden state at the last step 
    and returns the metric (logit or probability) of the target token.
    """
    target_layer = get_module_by_path(model, TARGET_LAYER_PATH)
    
    # Prepare Inputs
    full_sequence = test_item["model_output"][0] # shape: [1, seq_len]
    input_ids = full_sequence[:, :token_index]

    # Hook definition
    def replace_token_hidden_state(module, input, output):
        hidden_states = output[0] 
        rest_output = output[1:]
        new_hidden = hidden_states.clone()
        # Patch the hidden state generated in the last timestep (index -1)
        new_hidden[0, -1, :] = h_recon
        return (new_hidden, *rest_output)

    hook_handle = target_layer.register_forward_hook(replace_token_hidden_state)
    
    try:
        outputs = model(
            input_ids=input_ids,
            output_hidden_states=False,
            use_cache=False,
        )
        
        final_token_logits = outputs.logits[:, -1, :] 
        
        # Handle multi-token ids
        if hasattr(target_id, '__len__') and len(target_id) > 1:
            target_logit_id = target_id[0]
        else:
            target_logit_id = target_id
        
        if metric_mode == "logit":
            return final_token_logits[0, target_logit_id].item()
        else:
            probs = torch.softmax(final_token_logits, dim=-1)
            return probs[0, target_logit_id].item()
            
    finally:
        hook_handle.remove()


# -----------------------------------------------------------------------------
# METRIC 1: C-DELETION
# -----------------------------------------------------------------------------

def compute_c_deletion_metrics(
    test_item: Dict[str, Any],
    token_index: int,
    target_id: int,
    concept_matrix: torch.Tensor,
    activations: torch.Tensor,
    ranking_scores: torch.Tensor,
    metric_mode: str = "logit" 
) -> Tuple[List[float], float]:
    """
    Start with full concepts, remove from Most Important -> Least.
    Goal: AUC should be LOW (rapid drop).
    """
    device = model.device
    dtype = model.dtype
    
    concept_matrix = concept_matrix.to(device=device, dtype=dtype)
    activations = activations.to(device=device, dtype=dtype)
    ranking_scores = ranking_scores.to(device=device, dtype=dtype)

    # Sort descending (Remove most important first)
    sorted_indices = torch.argsort(ranking_scores[0], descending=True)
    
    current_activations = activations.clone()
    raw_trajectory = []
    
    # Baseline (No deletion)
    h_baseline = current_activations @ concept_matrix
    val_base = get_metric_with_patched_state(test_item, token_index, h_baseline, target_id, metric_mode)
    raw_trajectory.append(val_base)

    # Deletion Loop
    for idx_to_remove in sorted_indices:
        current_activations[0, idx_to_remove] = 0.0
        h_modified = current_activations @ concept_matrix
        val = get_metric_with_patched_state(test_item, token_index, h_modified, target_id, metric_mode)
        raw_trajectory.append(val)

    # Normalize/Prepare for AUC
    if metric_mode == "probability":
        if val_base > 0:
            norm_trajectory = [p / val_base for p in raw_trajectory]
        else:
            norm_trajectory = [0.0 for _ in raw_trajectory]
    else:
        norm_trajectory = raw_trajectory

    # Calculate AUC
    n_steps = len(norm_trajectory)
    x_axis = np.linspace(0, 1, n_steps)
    auc_val = auc(x_axis, norm_trajectory)
    
    return norm_trajectory, auc_val


# -----------------------------------------------------------------------------
# METRIC 2: C-INSERTION
# -----------------------------------------------------------------------------

def compute_c_insertion_metrics(
    test_item: Dict[str, Any],
    token_index: int,
    target_id: int,
    concept_matrix: torch.Tensor,
    activations: torch.Tensor,
    ranking_scores: torch.Tensor,
    metric_mode: str = "logit" 
) -> Tuple[List[float], float]:
    """
    Start with 0 concepts, add from Most Important -> Least.
    Goal: AUC should be HIGH (rapid recovery).
    """
    device = model.device
    dtype = model.dtype
    
    concept_matrix = concept_matrix.to(device=device, dtype=dtype)
    activations = activations.to(device=device, dtype=dtype)
    ranking_scores = ranking_scores.to(device=device, dtype=dtype)

    # Sort descending (Insert most important first)
    sorted_indices = torch.argsort(ranking_scores[0], descending=True)
    
    # Start with Zero Vector
    current_activations = torch.zeros_like(activations)
    raw_trajectory = []
    
    # Baseline (All Zero)
    h_baseline = current_activations @ concept_matrix
    val_base = get_metric_with_patched_state(test_item, token_index, h_baseline, target_id, metric_mode)
    raw_trajectory.append(val_base)

    # Insertion Loop
    for idx_to_insert in sorted_indices:
        # Restore activation
        original_val = activations[0, idx_to_insert]
        current_activations[0, idx_to_insert] = original_val
        
        # Reconstruct
        h_modified = current_activations @ concept_matrix
        val = get_metric_with_patched_state(test_item, token_index, h_modified, target_id, metric_mode)
        raw_trajectory.append(val)

    # Normalize/Prepare
    if metric_mode == "probability":
        # Usually for insertion we care about reaching the max probability
        max_val = max(raw_trajectory) if max(raw_trajectory) > 0 else 1.0
        norm_trajectory = [p / max_val for p in raw_trajectory]
    else:
        norm_trajectory = raw_trajectory

    # Calculate AUC
    n_steps = len(norm_trajectory)
    x_axis = np.linspace(0, 1, n_steps)
    auc_val = auc(x_axis, norm_trajectory)
    
    return norm_trajectory, auc_val


# -----------------------------------------------------------------------------
# METRIC 3: C-µFIDELITY
# -----------------------------------------------------------------------------

def compute_c_fidelity_metrics(
    test_item: Dict[str, Any],
    token_index: int,
    target_id: int,
    concept_matrix: torch.Tensor,
    activations: torch.Tensor,
    importance_scores: torch.Tensor,
    metric_mode: str = "logit",
    num_samples: int = 50 
) -> Tuple[float, float, List[float], List[float]]:
    """
    Computes Pearson correlation between Expected Drop (based on importance)
    and Actual Drop (based on model output) over random subsets.
    """
    device = model.device
    dtype = model.dtype
    
    concept_matrix = concept_matrix.to(device=device, dtype=dtype)
    activations = activations.to(device=device, dtype=dtype)
    importance_scores = importance_scores.to(device=device, dtype=dtype)
    n_concepts = activations.shape[1]
    
    # Original Output (Baseline)
    h_original = activations @ concept_matrix
    original_val = get_metric_with_patched_state(test_item, token_index, h_original, target_id, metric_mode)
    
    predicted_deltas = [] 
    actual_deltas = []   
    
    # Random Sampling Loop
    for _ in range(num_samples):
        # Random binary mask
        mask = torch.randint(0, 2, (1, n_concepts), device=device, dtype=dtype)
        
        # Expected Drop: Sum of importance of REMOVED concepts
        removed_mask = 1.0 - mask
        expected_drop = torch.sum(importance_scores * removed_mask).item()
        predicted_deltas.append(expected_drop)
        
        # Actual Drop
        masked_activations = activations * mask
        h_masked = masked_activations @ concept_matrix
        new_val = get_metric_with_patched_state(test_item, token_index, h_masked, target_id, metric_mode)
        
        actual_drop = original_val - new_val
        actual_deltas.append(actual_drop)
        
    # Calculate Correlation
    if len(predicted_deltas) > 1:
        # Check if variance exists (avoid divide by zero)
        if np.std(predicted_deltas) > 0 and np.std(actual_deltas) > 0:
            correlation, p_value = pearsonr(predicted_deltas, actual_deltas)
        else:
            correlation, p_value = 0.0, 1.0
    else:
        correlation, p_value = 0.0, 1.0

    return correlation, p_value, predicted_deltas, actual_deltas


# -----------------------------------------------------------------------------
# CONSOLIDATED PIPELINE
# -----------------------------------------------------------------------------

async def run_faithfulness_evaluation_pipeline(
    token_of_interest: str,
    uploaded_img_hidden_state: Dict[str, Any], # The 'test_item'
    concept_dict: Dict[str, Any],
    concept_activations: Union[torch.Tensor, List[float]], 
    concept_importance_scores: Union[torch.Tensor, List[float]] 
):
    """
    Runs C-Deletion, C-Insertion, and C-µFidelity sequentially.
    Yields progress events and returns a consolidated results dictionary.
    """
    results = {}
    
    try:
        start_time_all = time.perf_counter()
        yield new_log_event(logger, f"Starting Full Faithfulness Evaluation for '{token_of_interest}'...")

        # 1. SETUP & TENSOR CONVERSION
        # -------------------------------------------------
        if isinstance(concept_activations, list):
            concept_activations = torch.tensor([concept_activations], device=model.device, dtype=model.dtype)
        else:
            concept_activations = concept_activations.to(device=model.device, dtype=model.dtype)

        if isinstance(concept_importance_scores, list):
            concept_importance_scores = torch.tensor([concept_importance_scores], device=model.device, dtype=model.dtype)
        else:
            concept_importance_scores = concept_importance_scores.to(device=model.device, dtype=model.dtype)

        # Setup indices
        target_token_vocab_idx = get_vocab_idx_of_target_token(
            token_of_interest, None, llava_model_class.get_tokenizer()
        )
        token_index, _ = get_first_pos_of_token_of_interest(
            tokens=uploaded_img_hidden_state.get("hidden_states")[0].get(TARGET_LAYER_PATH)[0],
            pred_tokens=uploaded_img_hidden_state.get("model_output")[0],
            target_token_vocab_idx=target_token_vocab_idx, 
        )
        
        concept_matrix = concept_dict["concepts"]

        # 2. RUN C-DELETION
        # -------------------------------------------------
        yield new_log_event(logger, "Running C-Deletion (Goal: Ours AUC < Random AUC)...")
        
        # Ours
        traj_del_ours, auc_del_ours = compute_c_deletion_metrics(
            uploaded_img_hidden_state, token_index, target_token_vocab_idx,
            concept_matrix, concept_activations, concept_importance_scores
        )
        # Random
        rand_scores = torch.rand_like(concept_importance_scores)
        traj_del_rand, auc_del_rand = compute_c_deletion_metrics(
            uploaded_img_hidden_state, token_index, target_token_vocab_idx,
            concept_matrix, concept_activations, rand_scores
        )
        
        del_faithful = auc_del_ours < auc_del_rand
        yield new_log_event(logger, f" -> Deletion: {'FAITHFUL' if del_faithful else 'UNFAITHFUL'} (Ours: {auc_del_ours:.2f}, Rand: {auc_del_rand:.2f})")
        
        results["c_deletion"] = {
            "auc_ours": auc_del_ours, "auc_random": auc_del_rand,
            "trajectory_ours": traj_del_ours, "trajectory_random": traj_del_rand,
            "is_faithful": bool(del_faithful)
        }

        # 3. RUN C-INSERTION
        # -------------------------------------------------
        yield new_log_event(logger, "Running C-Insertion (Goal: Ours AUC > Random AUC)...")
        
        # Ours
        traj_ins_ours, auc_ins_ours = compute_c_insertion_metrics(
            uploaded_img_hidden_state, token_index, target_token_vocab_idx,
            concept_matrix, concept_activations, concept_importance_scores
        )
        # Random
        traj_ins_rand, auc_ins_rand = compute_c_insertion_metrics(
            uploaded_img_hidden_state, token_index, target_token_vocab_idx,
            concept_matrix, concept_activations, rand_scores
        )
        
        ins_faithful = auc_ins_ours > auc_ins_rand
        yield new_log_event(logger, f" -> Insertion: {'FAITHFUL' if ins_faithful else 'UNFAITHFUL'} (Ours: {auc_ins_ours:.2f}, Rand: {auc_ins_rand:.2f})")
        
        results["c_insertion"] = {
            "auc_ours": auc_ins_ours, "auc_random": auc_ins_rand,
            "trajectory_ours": traj_ins_ours, "trajectory_random": traj_ins_rand,
            "is_faithful": bool(ins_faithful)
        }

        # 4. RUN C-FIDELITY
        # -------------------------------------------------
        yield new_log_event(logger, "Running C-µFidelity (Goal: High Correlation)...")

        corr, p_val, pred_deltas, act_deltas = compute_c_fidelity_metrics(
            uploaded_img_hidden_state, token_index, target_token_vocab_idx,
            concept_matrix, concept_activations, concept_importance_scores
        )
        
        # Fidelity Verdict (Threshold > 0.1 is arbitrary but standard for checking existence of signal)
        fid_faithful = corr > 0.1 
        yield new_log_event(logger, f" -> Fidelity: {'FAITHFUL' if fid_faithful else 'LOW SIGNAL'} (Corr: {corr:.4f})")
        
        results["c_fidelity"] = {
            "correlation": corr,
            "p_value": p_val,
            "is_faithful": bool(fid_faithful),
            "predicted_deltas": pred_deltas, # for plotting
            "actual_deltas": act_deltas      # for plotting
        }

        # 5. FINALIZE
        # -------------------------------------------------
        elapsed = time.perf_counter() - start_time_all
        yield new_log_event(logger, f"Faithfulness Evaluation Complete in {elapsed:.2f}s")
        
        # Return consolidated dictionary
        yield new_event(event_type="return", data=results)

    except Exception as e:
        logger.error(f"Error in Faithfulness Pipeline: {str(e)}")
        yield new_event(event_type="error", data=f"Error in Faithfulness Pipeline: {str(e)}")
        return