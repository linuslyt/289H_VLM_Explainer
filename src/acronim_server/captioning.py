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
from acronim_server import (get_output_hidden_state_path, get_uploaded_img_path, 
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
DICTIONARY_LEARNING_SUBSET_SIZE=82783 # full set is 82783
NUM_CONCEPTS=20
TARGET_IMAGE=""
TARGET_TOKEN="motorcycle" # NOTE: some tokens will cause dictionary learning to fail, e.g. "dogs" for some reason. not entirely sure why.
SEED=28
OUT_DIR="/home/ytllam/xai/xl-vlms/out/gradient_concept"
LOG_FILE=os.path.join(os.path.join(OUT_DIR, f"importance_estimation_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}_{TARGET_TOKEN}.log"))
DICT_ANALYSIS_NAME="decompose_activations_text_grounding_image_grounding"

uploaded_img_hidden_state_filename = f"llava_input_inference_{DATASET_NAME}_{INFERENCE_DATA_SPLIT}_{TARGET_TOKEN}"

CAPTIONING_ARGS = {
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
    "save_filename": uploaded_img_hidden_state_filename,
    "seed": SEED,
    "processor_name": MODEL_NAME,
    "generation_mode": True,
    "exact_match_modules_to_hook": True,
    "save_only_generated_tokens": True,
    "batch_size": 1,
}

FORCE_RECOMPUTE=False # TODO: make script argument

@torch.no_grad()
def caption_uploaded_img(
    uploaded_img_filename: str
):
    start_time = time.perf_counter()
    # if preprocessed_input exists, load...
    logger = setup_logger(LOG_FILE)
    set_seed(SEED)
    device = get_most_free_gpu() if torch.cuda.is_available() else torch.device("cpu")
    # output_filename, output_saved_path = get_output_hidden_state_path(uploaded_img_filename)
    args = get_arguments({
        **CAPTIONING_ARGS,
        # "save_filename": output_filename,
        # "dataset_size": INFERENCE_SUBSET_SIZE,
        # "split": INFERENCE_DATA_SPLIT,
        # "token_of_interest": TARGET_TOKEN,
    })
    log_args(args, logger)
    yield new_log_event(logger, f"Loading model={MODEL_NAME} on device={device}, gpu={torch.cuda.is_available()}...")
    model_class = get_model_class(
        model_name_or_path=MODEL_NAME,
        processor_name=MODEL_NAME,
        device=device,
        logger=logger,
        args=args, # larger arg dict not needed for model setup
    )

    yield new_log_event(logger, f"Generating caption for image={uploaded_img_filename}...")
    
    # Run without hooks for first inference
    img_path = get_uploaded_img_path(uploaded_img_filename)
    logger.info(f"Generating caption for image {img_path}")
    preprocessed_input = model_class.preprocessor(
        instruction=CAPTIONING_PROMPT,
        image_file=img_path,
        response="", # not provided bc we're doing generation
        generation_mode=args.generation_mode
    )
    # TODO: persist preprocessed_input to file

    model = model_class.get_model()
    out = model.generate(
        **preprocessed_input, max_new_tokens=args.max_new_tokens, do_sample=False
    )

    # Output tokens after prompt tokens, i.e. generated caption tokens
    prompt_token_len = (
        preprocessed_input["input_ids"].shape[1]
        if preprocessed_input["input_ids"].ndim > 1
        else preprocessed_input["input_ids"].shape[0]
    )
    output_caption = model_class.get_tokenizer().batch_decode(
        out[:, prompt_token_len:], skip_special_tokens=True
    )

    # Record the end time
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    # save preprocessed_input..., caption
    yield new_log_event(logger, f"Generated caption for img={uploaded_img_filename} in time={elapsed_time:.6f}s: '{output_caption}'")
    yield new_event(event_type="caption", data=output_caption[0])