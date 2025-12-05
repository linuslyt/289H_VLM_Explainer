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
                            get_dict_model_class,
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

DEFAULT_LEARNING_ARGS = {
    "model_name_or_path": MODEL_NAME,
    "dataset_name": DATASET_NAME,
    "dataset_size": DICTIONARY_LEARNING_MIN_SAMPLE_SIZE,
    "data_dir": DATA_DIR,
    "annotation_file": ANNOTATION_FILE,
    "split": DICTIONARY_LEARNING_DATA_SPLIT,
    "hook_names": HOOK_NAMES,
    "modules_to_hook": TARGET_FEATURE_MODULES,
    # used to filter dataset to images where the token of interest exists in caption.
    # we can set this to True so we only sample from those images for which we have a concept dictionary precomputed for.
    "select_token_of_interest_samples": True,
    "token_of_interest": "", # override
    "save_dir": get_saved_concept_dicts_dir(),
    "save_filename": "", # should be overridden
    "seed": SEED,
    "processor_name": MODEL_NAME,
    "generation_mode": True,
    "exact_match_modules_to_hook": True,
    "save_only_generated_tokens": True,
    "batch_size": 26,
    "features_path": "", # override with path to hidden states for sampled training images
    "programmatic": True,
    "analysis_name": DICT_ANALYSIS_NAME, # generates concepts for the selected token across relevant samples in whole dataset, 
                                        # then grounds each concept textually and visually
    "module_to_decompose": TARGET_FEATURE_MODULES[0][0],
    "num_concepts": [NUM_CONCEPTS], # why the hell is nargs="+"??
    "decomposition_method": "snmf",
}

FORCE_RECOMPUTE=True

set_seed(SEED)

LOG_FILE=os.path.join(os.path.join(OUT_DIR, f"acronim_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.log"))
logger = setup_logger(LOG_FILE)

default_learning_args = get_arguments(DEFAULT_LEARNING_ARGS)
dict_learning_model_class = get_dict_model_class()

async def learn_concept_dictionary_for_token(token_of_interest: str, sampled_subset_size: int, force_recompute: bool=FORCE_RECOMPUTE):
    concept_dict_filename, concept_dict_full_saved_path = get_output_concept_dictionary_path(token_of_interest)
    if os.path.exists(concept_dict_full_saved_path) and not force_recompute:
        cached_concept_dict = torch.load(concept_dict_full_saved_path)
        yield new_log_event(logger, f"Loaded concept dict for token={token_of_interest} from path={concept_dict_full_saved_path}'")
        yield new_event(event_type="return", data=cached_concept_dict)
        return

    _, sampled_hidden_states_full_saved_path = get_output_hidden_state_paths(token_of_interest)

    start_time = time.perf_counter()
    yield new_log_event(logger, f"Learning concept dictionary for token={token_of_interest}...")
    
    learning_args = get_arguments({
        **DEFAULT_LEARNING_ARGS,
        "save_dir": get_saved_concept_dicts_dir(),
        "save_filename": concept_dict_filename,
        "features_path": sampled_hidden_states_full_saved_path,
        "dataset_size": sampled_subset_size,
    })

    log_args(learning_args, logger)
    global_concept_dict_for_token = analyse_features(
        analysis_name=learning_args.analysis_name,
        logger=logger,
        token_of_interest=token_of_interest,
        model_class=dict_learning_model_class,
        device=torch.device("cpu"),
        args=learning_args,
    )

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    yield new_log_event(logger, f"Learned concept dictionary for token={token_of_interest} in time={elapsed_time:.6f}s'")
    yield new_event(event_type="return", data=global_concept_dict_for_token)
