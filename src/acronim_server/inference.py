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
                            get_preprocessed_img_saved_path,
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

FORCE_RECOMPUTE=False # TODO: make script argument

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

@torch.no_grad()
def caption_uploaded_img(
    uploaded_img_path: str,
    force_recompute: bool = True,
):
    start_time = time.perf_counter()
    # if preprocessed_input exists, load...
    log_args(default_model_args, logger)
    yield new_log_event(logger, f"Loading model={MODEL_NAME} on device={device}, gpu={torch.cuda.is_available()}...", passthrough=False)
    # model_class = get_model_class(
    #     model_name_or_path=MODEL_NAME,
    #     processor_name=MODEL_NAME,
    #     device=device,
    #     logger=logger,
    #     args=args, # larger arg dict not needed for model setup
    # )

    yield new_log_event(logger, f"Generating caption for image={uploaded_img_path}...", passthrough=False)
    
    # Run without hooks for first inference
    img_path = get_uploaded_img_saved_path(uploaded_img_path)
    preprocessed_img_path = get_preprocessed_img_saved_path(uploaded_img_path)
    if os.path.exists(preprocessed_img_path) and not force_recompute:
        yield new_log_event(logger, f"Loading preprocessed image from path={preprocessed_img_path}", passthrough=False)
        preprocessed_input = torch.load(preprocessed_img_path)
    else:
        preprocessed_input = llava_model_class.preprocessor(
            instruction=CAPTIONING_PROMPT,
            image_file=img_path,
            response="", # not provided bc we're doing generation
            generation_mode=default_model_args.generation_mode
        )
        torch.save(preprocessed_input, preprocessed_img_path)
        yield new_log_event(logger, f"Saved preprocessed image to path={preprocessed_img_path}", passthrough=False)

    out = model.generate(
        **preprocessed_input, max_new_tokens=default_model_args.max_new_tokens, do_sample=False
    )

    # Output tokens after prompt tokens, i.e. generated caption tokens
    prompt_token_len = (
        preprocessed_input["input_ids"].shape[1]
        if preprocessed_input["input_ids"].ndim > 1
        else preprocessed_input["input_ids"].shape[0]
    )
    output_caption = llava_model_class.get_tokenizer().batch_decode(
        out[:, prompt_token_len:], skip_special_tokens=True
    )


    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    yield new_log_event(logger, f"Generated caption for img={uploaded_img_path} in time={elapsed_time:.6f}s: '{output_caption}'", passthrough=False)
    yield new_event(event_type="caption", data=output_caption[0], passthrough=False)

@torch.no_grad()
async def get_hidden_state_for_input(
    uploaded_img_path: str,
    token_of_interest: str,
    force_recompute: bool = True,
):
    start_time = time.perf_counter()
    # if preprocessed_input exists, load...
    # filename and full path it's saved to by hooks
    hidden_state_filename, hidden_state_full_saved_path = get_output_hidden_state_paths(uploaded_img_path=uploaded_img_path, token_of_interest=token_of_interest)
    
    # Load from cached if exists
    if os.path.exists(hidden_state_full_saved_path) and not force_recompute:
        yield new_log_event(logger, f"Loaded hidden state for img={uploaded_img_path} wrt token={token_of_interest} from path={hidden_state_full_saved_path}'")
        cached_hidden_state = torch.load(hidden_state_full_saved_path)
        yield new_event(event_type="return", data=cached_hidden_state)
        return

    args = get_arguments({
        **DEFAULT_CAPTIONING_ARGS,
        "save_filename": hidden_state_filename,
        "dataset_size": INFERENCE_SUBSET_SIZE,
        "split": INFERENCE_DATA_SPLIT,
        "token_of_interest": token_of_interest,
    })
    log_args(args, logger)

    yield new_log_event(logger, f"Extracting hidden state w.r.t. token={token_of_interest} for image={uploaded_img_path}...")

    img_path = get_uploaded_img_saved_path(uploaded_img_path)
    preprocessed_img_path = get_preprocessed_img_saved_path(uploaded_img_path)
    if os.path.exists(preprocessed_img_path):
        yield new_log_event(logger, f"Loading preprocessed image from path={preprocessed_img_path}")
        preprocessed_input = torch.load(preprocessed_img_path)
    else:
        preprocessed_input = llava_model_class.preprocessor(
            instruction=CAPTIONING_PROMPT,
            image_file=img_path,
            response="", # not provided bc we're doing generation
            generation_mode=args.generation_mode
        )
        torch.save(preprocessed_input, preprocessed_img_path)
        yield new_log_event(logger, f"Saved preprocessed image to path={preprocessed_img_path}")

    # Set up forward hook - we need to save hidden state at token of interest this time
    hook_return_functions, hook_postprocessing_functions = setup_hooks(
        model=model,
        modules_to_hook=args.modules_to_hook,
        hook_names=args.hook_names,
        tokenizer=llava_model_class.get_tokenizer(),
        logger=logger,
        args=args,
    )

    out = model.generate(
        **preprocessed_input, max_new_tokens=args.max_new_tokens, do_sample=False
    )

    # Construct test_item dict to be saved
    img_name, img_extension = os.path.splitext(uploaded_img_path)
    test_item = {
        "img_id": img_name,
        "text": CAPTIONING_PROMPT,
        "image": uploaded_img_path,
        "model_output": out,
    }
    test_item["preprocessed_input"] = preprocessed_input

    # Get index where prompt tokens end and generated caption tokens begin
    prompt_token_len = (
        preprocessed_input["input_ids"].shape[1]
        if preprocessed_input["input_ids"].ndim > 1
        else preprocessed_input["input_ids"].shape[0]
    )

    test_item["model_generated_output"] = out[:, prompt_token_len:]
    test_item["model_predictions"] = llava_model_class.get_tokenizer().batch_decode(
        out[:, prompt_token_len:], skip_special_tokens=True
    )
    
    if hook_return_functions is not None:
            for func in hook_return_functions:
                if func is not None:
                    hook_output = func(**test_item)
                    if hook_output:
                        test_item.update(hook_output)

    hook_data = update_dict_of_list(test_item, data={}, is_batched=False)
    clear_hooks_variables()
    clear_forward_hooks(model) # reset hooks
    
    # Save hidden state to file and also return it
    if hook_postprocessing_functions is not None:
        for func in hook_postprocessing_functions:
            if func is not None:
                func(data=hook_data, 
                     data_keys=['img_id', 'instruction', 'response', 'image',
                                'targets', 'text', 'preprocessed_input', 
                                'model_output', 'model_generated_output', 'model_predictions',
                                'token_of_interest_mask', 'hidden_states'], # save all fields, not just hidden_states
                     args=args, logger=logger)
    
    yield new_log_event(logger, f"Saved hidden state for img={uploaded_img_path} to {hidden_state_full_saved_path}")

    # Record the end time
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    # save preprocessed_input..., caption
    yield new_log_event(logger, f"Extracted hidden state for img={uploaded_img_path} wrt token={token_of_interest} in time={elapsed_time:.6f}s'")
    yield new_event(event_type="return", data=hook_data)


@torch.no_grad()
async def get_hidden_states_for_training_samples(
    token_of_interest: str,
    sampled_subset_size: int,
    force_recompute: bool = True,
):
    sampled_hidden_states_filename, sampled_hidden_states_full_saved_path = get_output_hidden_state_paths(token_of_interest=token_of_interest)

    # Load from cached if exists
    if os.path.exists(sampled_hidden_states_full_saved_path) and not force_recompute:
        cached_hidden_states = torch.load(sampled_hidden_states_full_saved_path)
        yield new_log_event(logger, f"Loaded {len(cached_hidden_states['hidden_states'])} sampled hidden states for token={token_of_interest} from path={sampled_hidden_states_full_saved_path}'")
        yield new_event(event_type="return", data=cached_hidden_states)
        return

    start_time = time.perf_counter()
    yield new_log_event(logger, f"Extracting hidden states for relevant training data samples...")
    
    # Argument setup
    sampled_subset_size = max(DICTIONARY_LEARNING_MIN_SAMPLE_SIZE, min(sampled_subset_size, COCO_TRAIN_FULL_SIZE))

    batch_inference_args = get_arguments({ # Should match save_features.sh
        **DEFAULT_CAPTIONING_ARGS,
        "dataset_size": sampled_subset_size,
        "split": DICTIONARY_LEARNING_DATA_SPLIT,
        "save_filename": sampled_hidden_states_filename,
        "batch_size": 26,
        "programmatic": True,
        "token_of_interest": token_of_interest,
    })

    # Inference
    hook_return_functions, hook_postprocessing_functions = setup_hooks(
        model=llava_model_class.model_,
        modules_to_hook=batch_inference_args.modules_to_hook,
        hook_names=batch_inference_args.hook_names,
        tokenizer=llava_model_class.get_tokenizer(),
        logger=logger,
        args=batch_inference_args,
    )
    
    loader = get_dataset_loader(
        dataset_name=batch_inference_args.dataset_name, logger=logger, args=batch_inference_args,
    )

    hook_data = inference(
        loader=loader,
        model_class=llava_model_class,
        hook_return_functions=hook_return_functions,
        device=device,
        logger=logger,
        args=batch_inference_args,
    )

    clear_forward_hooks(model) # reset hooks so model can be reused

    # Save features to file
    if hook_postprocessing_functions is not None:
        for func in hook_postprocessing_functions:
            if func is not None:
                func(data=hook_data, args=batch_inference_args, logger=logger)

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    yield new_log_event(logger, f"Extracted hidden states for relevant samples wrt token={token_of_interest} in time={elapsed_time:.6f}s'")
    yield new_log_event(logger, f"Saved hidden state for relevant images to {sampled_hidden_states_full_saved_path}")
    yield new_event(event_type="return", data=hook_data)
