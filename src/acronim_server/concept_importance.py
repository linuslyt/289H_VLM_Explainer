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
from acronim_server import (get_output_hidden_state_paths, get_uploaded_img_saved_path,
                            get_uploaded_img_dir, get_saved_hidden_states_dir,
                            get_output_concept_dictionary_path,
                            get_saved_concept_dicts_dir,
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
NUM_CONCEPTS=20
TARGET_IMAGE=""
SEED=28
OUT_DIR="/home/ytllam/xai/xl-vlms/out/gradient_concept"
DICT_ANALYSIS_NAME="decompose_activations_text_grounding_image_grounding"

DEFAULT_CAPTIONING_ARGS = {
    "model_name_or_path": MODEL_NAME,
    "dataset_name": DATASET_NAME,
    "dataset_size": DICTIONARY_LEARNING_MIN_SAMPLE_SIZE,
    "data_dir": DATA_DIR,
    "annotation_file": ANNOTATION_FILE,
    "split": INFERENCE_DATA_SPLIT,
    "hook_names": HOOK_NAMES,
    "modules_to_hook": TARGET_FEATURE_MODULES,
    # used to filter dataset to images where the token of interest exists in caption.
    # we can set this to True so we only sample from those images for which we have a concept dictionary precomputed for.
    "select_token_of_interest_samples": True,
    "token_of_interest": "", # override
    "save_dir": get_uploaded_img_dir(),
    "save_filename": "", # should be overridden
    "seed": SEED,
    "processor_name": MODEL_NAME,
    "generation_mode": True,
    "exact_match_modules_to_hook": True,
    "save_only_generated_tokens": True,
    "batch_size": 1,
}

set_seed(SEED)
device = get_most_free_gpu() if torch.cuda.is_available() else torch.device("cpu")

LOG_FILE=os.path.join(os.path.join(OUT_DIR, f"acronim_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.log"))
logger = setup_logger(LOG_FILE)

default_model_args = get_arguments(DEFAULT_CAPTIONING_ARGS)
llava_model_class = get_model_class(
    model_name_or_path=MODEL_NAME,
    processor_name=MODEL_NAME,
    device=device,
    logger=logger,
    args=default_model_args, # larger arg dict not needed for model setup
)
model = llava_model_class.get_model()

async def calculate_concept_importance(token_of_interest, uploaded_img_hidden_state_path, concept_dict, force_recompute: bool=True):
    start_time = time.perf_counter()
    yield new_log_event(logger, f"Calculating concept importance...")
    print("tok:", token_of_interest)

    # logger.info(f"Projecting test instance hidden representation w/r/t selected token onto concept dict...")
    # # From concept_grounding_visualization.ipynb example
    data = torch.load(uploaded_img_hidden_state_path, map_location="cpu")
    print(data)
    test_item = data
    feat = get_feature_matrix(data["hidden_states"], module_name="language_model.model.layers.31", token_idx=None)
    print(feat)
    projections = project_representations(
        sample=feat,
        analysis_model=concept_dict["analysis_model"],
        decomposition_type=concept_dict["decomposition_method"],
    )
    # # With 1 input sample, we should get (1, n_concepts=20), (1, feature_dims), and 1 respectively
    print(projections.shape, feat.shape, len(data["hidden_states"]))
    print(projections)

    # # TODO: refactor into reconstruct_differentiable_hidden_state()
    target_token_vocab_idx = get_vocab_idx_of_target_token(token_of_interest,
                                                           args_token_of_interest_idx=None, # TODO: check what this is
                                                           tokenizer=llava_model_class.get_tokenizer())
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

    print(f"vocab_idx={target_token_vocab_idx}")
    print(f"target_token_first_idx_in_caption_output={target_token_first_idx_in_caption_output}")
    # exit()

    v_X = torch.tensor(projections)
    h_recon = v_X @ concept_dict["concepts"]
    print(f"h_recon_type:{type(h_recon)}, test_input_type:{type(preprocessed_test_input[0])}")
    h_recon_torch = h_recon.to(dtype=model.dtype, device=model.device).clone().detach().requires_grad_(True)
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
        concept_matrix=concept_dict["concepts"]
    )
    print(concept_grads)
    print(concept_grads.shape)
    # If these are zero gradient chain is broken
    print(f"Non-zero elements: {torch.count_nonzero(concept_grads).item()}")
    print(f"Max gradient: {concept_grads.abs().max().item()}")
    print(f"Non-zero elements: {torch.count_nonzero(concept_grads).item()}")

    torch.save(concept_grads, "motorcycle_grad.pth")
    concept_activations = v_X.to(device=model.device, dtype=model.dtype) # v_X shape: [1, N_concepts]
    concept_importance_scores = concept_activations * concept_grads
    top_scores, top_indices = torch.topk(concept_importance_scores, k=NUM_CONCEPTS) # if we match k=#concepts we should get ranking of all concepts

    # TODO: split into rankings for both positive and negative contributors.
    for rank, (score, concept_idx) in enumerate(zip(top_scores[0], top_indices[0])):
        print(f"#{rank+1}: Concept {concept_idx}."
            + f"\nImage groundings ={concept_dict['image_grounding_paths'][concept_idx][:10]}" 
            + f"\nText groundings ={concept_dict['text_grounding'][concept_idx][:10]}..."
            + f"\n(Importance {score.item():.4f}; Activation: {concept_activations[0, concept_idx]:.2f}; Gradient: {concept_grads[0, concept_idx]:.4f})")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    yield new_log_event(logger, f"Calculated concept importance scores for token={token_of_interest} in time={elapsed_time:.6f}s'")
    yield new_event(event_type="return", data={
        "scores": concept_importance_scores,
    })
