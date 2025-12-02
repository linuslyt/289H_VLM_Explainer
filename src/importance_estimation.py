import argparse
import os
from datetime import datetime
import torch
from tqdm import tqdm
from typing import Any, Callable, Dict, List, Tuple

from datasets import get_dataset_loader
from helpers.arguments import get_arguments
from helpers.logger import log_args, setup_logger
from helpers.utils import (clear_forward_hooks, clear_hooks_variables, get_most_free_gpu,
                           set_seed, setup_hooks)
from models import get_model_class
from models.image_text_model import ImageTextModel

from save_features import inference

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
OUT_FILE="gradient_concept_test"
LOG_FILE=os.path.join(os.path.join(OUT_DIR, f"importance_estimation_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}_{TARGET_TOKEN}.log"))

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
    # "save_dir": OUT_DIR,
    # "save_filename": OUT_FILE,
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
) -> Dict[str, Any]:
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
    clear_hooks_variables()
    return test_item

def save_concept_dict_for_token(
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

    return hook_data


if __name__ == "__main__":
    logger = setup_logger(LOG_FILE)
    set_seed(SEED)
    device = get_most_free_gpu() if torch.cuda.is_available() else torch.device("cpu")

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
    logger.info(f"Loaded model={MODEL_NAME} on device={device} gpu={torch.cuda.is_available()}")

    logger.info(f"Running inference for single input...")
    inference_args = get_arguments({
        **DEFAULT_ARGS,
        "dataset_size": INFERENCE_SUBSET_SIZE,
        "split": INFERENCE_DATA_SPLIT,
    })
    test_item = single_inference(
        model_class=llava_model,
        device=device,
        logger=logger,
        args=inference_args,
    )
    # reset hooks so model can be reused
    clear_forward_hooks(llava_model.model_)

    # print(test_item)
    # Should match save_features.sh
    logger.info(f"Extracting training features for selected token...")
    concept_dict_mining_args = get_arguments({
        **DEFAULT_ARGS,
        "dataset_size": DICTIONARY_LEARNING_SUBSET_SIZE,
        "split": DICTIONARY_LEARNING_DATA_SPLIT,
        "save_dir": OUT_DIR,
        "save_filename": OUT_FILE,
        "batch_size": 26,
        "programmatic": True,
    })

    # TODO: select token in item's caption
    concept_dict_for_token = save_concept_dict_for_token(
        model_class=llava_model,
        device=device,
        logger=logger,
        args=concept_dict_mining_args,
    )
    print(concept_dict_for_token)

# create_concept_dict_for_token()
# TODO: generate concept dict for token in output. use save_features then feature_decomposition. 
#       need separate arg namespace to change dataloader split to train, and batch size.

# create_concept_dicts_for_caption()
# TODO (stretch goal): get concept dict for every word in caption

# project_on_concept_dict()
# Use analysis_decomposition.project_representations(). see ipynb
# projections = analysis_decomposition.project_representations(
#     sample=feat,
#     analysis_model=concepts_dict["analysis_model"],
#     decomposition_type=concepts_dict["decomposition_method"],
# )
# their 143 input samples: projections.shape (143, 20) feat.shape torch.Size([143, 3584]) no. hidden states 143
# we should have 1 input sample, so we should get (1, n_concepts=20), (1, feature_dims), and 1 respectively
# where feat = get_feature_matrix(data["hidden_states"], module_name="model.norm", token_idx=None)
# hidden_states expects list of dicts [{}]. just wrap our instance in list

# reconstruct_differentiable_hidden_state()

# calculate_gradient_concept()

# create_groundings()
# see concept_grounding_visualization.ipynb 