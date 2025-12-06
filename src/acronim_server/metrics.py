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
from acronim_server import (get_llava_model_class, new_log_event, new_event, get_uploaded_img_dir)

# Import reusable logic
from acronim_server.concept_importance import get_concept_grad_at_target_token_step
from analysis.feature_decomposition import get_feature_matrix, project_representations
from acronim_server.inference import get_hidden_states_for_training_samples
from acronim_server.plotting import plot_dataset_benchmark

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
# HELPER: Calculate Importance on the Fly
# -----------------------------------------------------------------------------
def compute_sample_importance(
    test_item: Dict[str, Any],
    concept_dict: Dict[str, Any],
    token_of_interest: str
):
    """
    Calculates activations and importance scores for a single sample on the fly.
    Includes explicit check for 'token_of_interest_mask'.
    """
    # 0. SAFETY CHECK: Check the validity mask first
    # This prevents processing samples where the token wasn't generated.
    mask_val = False
    if "token_of_interest_mask" in test_item:
        mask_data = test_item["token_of_interest_mask"]
        
        # Handle various data structures (List of Tensors, single Tensor, or raw list)
        if isinstance(mask_data, list):
            mask_data = mask_data[0] # Unwrap list
        
        if isinstance(mask_data, torch.Tensor):
            mask_val = mask_data.item() # Unwrap tensor to bool
        else:
            mask_val = bool(mask_data)
            
    if not mask_val:
        raise ValueError(f"Skipping sample: Token '{token_of_interest}' was not found in the generated output (mask=False).")

    # 1. Project to get Activations
    # Note: inference.py returns hidden states in a list structure
    feat = get_feature_matrix(test_item["hidden_states"], module_name=TARGET_LAYER_PATH, token_idx=None)
    projections = project_representations(
        sample=feat,
        analysis_model=concept_dict["analysis_model"],
        decomposition_type=concept_dict["decomposition_method"],
    )
    
    activations = torch.tensor(projections, device=model.device, dtype=model.dtype)
    
    # 2. Reconstruct Hidden State (differentiable)
    concept_matrix = concept_dict["concepts"].to(device=model.device, dtype=model.dtype)
    h_recon = activations @ concept_matrix
    h_recon_torch = h_recon.clone().detach().requires_grad_(True)
    
    # 3. Get Token Indices
    target_token_vocab_idx = get_vocab_idx_of_target_token(token_of_interest, None, llava_model_class.get_tokenizer())
    
    # We use the safe accessor here, knowing the key exists because mask was True
    layer_tokens = test_item["hidden_states"][0][TARGET_LAYER_PATH][0]
    
    token_index, _ = get_first_pos_of_token_of_interest(
        tokens=layer_tokens,
        pred_tokens=test_item.get("model_output")[0],
        target_token_vocab_idx=target_token_vocab_idx, 
    )
    
    # 4. Calculate Gradient (Reusing imported function)
    grad_wrt_concepts = get_concept_grad_at_target_token_step(
        model_class=llava_model_class,
        test_item=test_item,
        token_index=token_index,
        h_recon_torch=h_recon_torch,
        target_id=target_token_vocab_idx,
        concept_matrix=concept_matrix
    )
        
    # 5. Compute Importance
    importance_scores = activations * grad_wrt_concepts
    
    return activations, importance_scores, token_index, target_token_vocab_idx

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
        if np.std(predicted_deltas) > 0 and np.std(actual_deltas) > 0:
            correlation, p_value = pearsonr(predicted_deltas, actual_deltas)
        else:
            correlation, p_value = 0.0, 1.0
    else:
        correlation, p_value = 0.0, 1.0

    return correlation, p_value, predicted_deltas, actual_deltas


# -----------------------------------------------------------------------------
# SINGLE SAMPLE PIPELINE
# -----------------------------------------------------------------------------

async def run_faithfulness_evaluation_pipeline(
    token_of_interest: str,
    uploaded_img_hidden_state: Dict[str, Any], # The 'test_item'
    concept_dict: Dict[str, Any],
    concept_activations: Union[torch.Tensor, List[float]], 
    concept_importance_scores: Union[torch.Tensor, List[float]] 
):
    """
    Runs metrics for a SINGLE sample.
    """
    results = {}
    
    try:
        start_time_all = time.perf_counter()
        yield new_log_event(logger, f"Starting Full Faithfulness Evaluation for '{token_of_interest}'...")

        # 1. SETUP & TENSOR CONVERSION
        if isinstance(concept_activations, list):
            concept_activations = torch.tensor([concept_activations], device=model.device, dtype=model.dtype)
        else:
            concept_activations = concept_activations.to(device=model.device, dtype=model.dtype)

        if isinstance(concept_importance_scores, list):
            concept_importance_scores = torch.tensor([concept_importance_scores], device=model.device, dtype=model.dtype)
        else:
            concept_importance_scores = concept_importance_scores.to(device=model.device, dtype=model.dtype)

        target_token_vocab_idx = get_vocab_idx_of_target_token(
            token_of_interest, None, llava_model_class.get_tokenizer()
        )
        token_index, _ = get_first_pos_of_token_of_interest(
            tokens=uploaded_img_hidden_state.get("hidden_states")[0].get(TARGET_LAYER_PATH)[0],
            pred_tokens=uploaded_img_hidden_state.get("model_output")[0],
            target_token_vocab_idx=target_token_vocab_idx, 
        )
        
        concept_matrix = concept_dict["concepts"]

        # 2. RUN METRICS
        yield new_log_event(logger, "Running C-Deletion...")
        traj_del_ours, auc_del_ours = compute_c_deletion_metrics(
            uploaded_img_hidden_state, token_index, target_token_vocab_idx,
            concept_matrix, concept_activations, concept_importance_scores, metric_mode="logit"
        )
        rand_scores = torch.rand_like(concept_importance_scores)
        traj_del_rand, auc_del_rand = compute_c_deletion_metrics(
            uploaded_img_hidden_state, token_index, target_token_vocab_idx,
            concept_matrix, concept_activations, rand_scores, metric_mode="logit"
        )
        results["c_deletion"] = {
            "auc_ours": auc_del_ours, "auc_random": auc_del_rand,
            "trajectory_ours": traj_del_ours, "trajectory_random": traj_del_rand,
            "is_faithful": bool(auc_del_ours < auc_del_rand)
        }

        yield new_log_event(logger, "Running C-Insertion...")
        traj_ins_ours, auc_ins_ours = compute_c_insertion_metrics(
            uploaded_img_hidden_state, token_index, target_token_vocab_idx,
            concept_matrix, concept_activations, concept_importance_scores, metric_mode="logit"
        )
        traj_ins_rand, auc_ins_rand = compute_c_insertion_metrics(
            uploaded_img_hidden_state, token_index, target_token_vocab_idx,
            concept_matrix, concept_activations, rand_scores, metric_mode="logit"
        )
        results["c_insertion"] = {
            "auc_ours": auc_ins_ours, "auc_random": auc_ins_rand,
            "trajectory_ours": traj_ins_ours, "trajectory_random": traj_ins_rand,
            "is_faithful": bool(auc_ins_ours > auc_ins_rand)
        }

        yield new_log_event(logger, "Running C-µFidelity...")
        corr, p_val, pred_deltas, act_deltas = compute_c_fidelity_metrics(
            uploaded_img_hidden_state, token_index, target_token_vocab_idx,
            concept_matrix, concept_activations, concept_importance_scores, metric_mode="logit"
        )
        results["c_fidelity"] = {
            "correlation": corr, "p_value": p_val,
            "is_faithful": bool(corr > 0.1),
            "predicted_deltas": pred_deltas, "actual_deltas": act_deltas
        }

        elapsed = time.perf_counter() - start_time_all
        yield new_log_event(logger, f"Faithfulness Evaluation Complete in {elapsed:.2f}s")
        yield new_event(event_type="return", data=results)

    except Exception as e:
        logger.error(f"Error in Single-Sample Pipeline: {str(e)}")
        yield new_event(event_type="error", data=f"Error: {str(e)}")
        return


# -----------------------------------------------------------------------------
# DATASET LEVEL PIPELINE
# -----------------------------------------------------------------------------

async def run_dataset_faithfulness_pipeline(
    token_of_interest: str,
    sampled_subset_size: int,
    num_eval_samples: int,
    concept_dict: Dict[str, Any],
    force_recompute: bool = False
):
    """
    Evaluates faithfulness over a batch of samples and generates aggregated plots.
    """
    
    # Aggregators
    results_agg = {
        "deletion_auc_ours": [], "deletion_auc_rand": [],
        "insertion_auc_ours": [], "insertion_auc_rand": [],
        "fidelity_corr": [],
        "samples_processed": 0
    }
    
    # Curve storage for averaging
    deletion_curves_ours = []
    deletion_curves_rand = []
    insertion_curves_ours = []
    insertion_curves_rand = []

    try:
        yield new_log_event(logger, f"Starting Dataset-Level Faithfulness on {num_eval_samples} samples...")

        # 1. Fetch Samples
        batch_data = None
        async for event_type, data in get_hidden_states_for_training_samples(
            token_of_interest=token_of_interest,
            sampled_subset_size=sampled_subset_size,
            force_recompute=force_recompute,
            batch_size=num_eval_samples 
        ):
            if event_type == "return":
                batch_data = data
            elif event_type == "error":
                yield new_event("error", data)
                return
        # print(batch_data)

        if not batch_data:
            yield new_event("error", "Failed to retrieve training samples.")
            return

        # 2. Iterate Samples
        total_available = len(batch_data['hidden_states'])
        limit = min(total_available, num_eval_samples)
        
        yield new_log_event(logger, f"Evaluating faithfulness metrics for token={token_of_interest} on {limit} samples...")

        for i in range(limit):
            # Slice batch into single sample dict
            test_item = {
                k: [v[i]] if isinstance(v, list) and len(v) > i else v 
                for k, v in batch_data.items()
            }
            
            try:
                # Calculate Importance on the fly
                activations, scores, t_idx, t_id = compute_sample_importance(
                    test_item, concept_dict, token_of_interest
                )
                
                # --- Deletion ---
                traj_del, auc_del = compute_c_deletion_metrics(
                    test_item, t_idx, t_id, concept_dict["concepts"], activations, scores, metric_mode="logit"
                )
                # Compute Random Baseline for Deletion
                rand_scores = torch.rand_like(scores)
                traj_del_rand, auc_del_rand = compute_c_deletion_metrics(
                    test_item, t_idx, t_id, concept_dict["concepts"], activations, rand_scores, metric_mode="logit"
                )
                
                # --- Insertion ---
                traj_ins, auc_ins = compute_c_insertion_metrics(
                    test_item, t_idx, t_id, concept_dict["concepts"], activations, scores, metric_mode="logit"
                )
                # Compute Random Baseline for Insertion
                traj_ins_rand, auc_ins_rand = compute_c_insertion_metrics(
                    test_item, t_idx, t_id, concept_dict["concepts"], activations, rand_scores, metric_mode="logit"
                )
                
                # --- Fidelity ---
                corr, _, _, _ = compute_c_fidelity_metrics(
                    test_item, t_idx, t_id, concept_dict["concepts"], activations, scores, metric_mode="logit"
                )

                # Store Stats
                results_agg["deletion_auc_ours"].append(auc_del)
                results_agg["deletion_auc_rand"].append(auc_del_rand)
                results_agg["insertion_auc_ours"].append(auc_ins)
                results_agg["insertion_auc_rand"].append(auc_ins_rand)
                if not np.isnan(corr): results_agg["fidelity_corr"].append(corr)
                
                # Store Curves for Averaging
                deletion_curves_ours.append(traj_del)
                deletion_curves_rand.append(traj_del_rand)
                insertion_curves_ours.append(traj_ins)
                insertion_curves_rand.append(traj_ins_rand)
                
                results_agg["samples_processed"] += 1
                
                if (i+1) % 5 == 0:
                    yield new_log_event(logger, f"Processed {i+1}/{limit} samples...")

            except Exception as e:
                logger.error(f"Failed sample {i}: {e}")
                continue

        # 3. Aggregate
        n = results_agg["samples_processed"]
        if n == 0:
            yield new_event("error", "No samples were successfully processed.")
            return

        final_stats = {
            "token": token_of_interest,
            "samples_n": n,
            "deletion": {
                "auc_ours_mean": float(np.mean(results_agg["deletion_auc_ours"])),
                "auc_ours_std": float(np.std(results_agg["deletion_auc_ours"])),
                "auc_rand_mean": float(np.mean(results_agg["deletion_auc_rand"])),
                "curve_ours_mean": np.mean(deletion_curves_ours, axis=0).tolist(),
                "curve_rand_mean": np.mean(deletion_curves_rand, axis=0).tolist(),
            },
            "insertion": {
                "auc_ours_mean": float(np.mean(results_agg["insertion_auc_ours"])),
                "auc_rand_mean": float(np.mean(results_agg["insertion_auc_rand"])),
                "curve_ours_mean": np.mean(insertion_curves_ours, axis=0).tolist(),
                "curve_rand_mean": np.mean(insertion_curves_rand, axis=0).tolist(),
            },
            "fidelity": {
                "correlation_mean": float(np.mean(results_agg["fidelity_corr"])),
                "correlation_std": float(np.std(results_agg["fidelity_corr"])),
                "raw_correlations": results_agg["fidelity_corr"]
            }
        }
        
        # 4. Generate Plots
        plots_dir = os.path.join(get_uploaded_img_dir(), "plots")
        plot_files = plot_dataset_benchmark(final_stats, plots_dir)
        final_stats["plots"] = plot_files
        
        verdict_msg = f"Dataset Eval ({n} samples) Complete. Plots generated."
        yield new_log_event(logger, verdict_msg)
        
        yield new_event("return", final_stats)

    except Exception as e:
        logger.error(f"Dataset Pipeline Error: {e}")
        yield new_event("error", str(e))