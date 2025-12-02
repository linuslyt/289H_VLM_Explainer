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

DATA_SPLIT="test"
SUBSET_SIZE=5000
TARGET_IMAGE=""
TARGET_TOKEN="dog"
SEED=28
OUT_DIR="/home/ytllam/xai/xl-vlms/out/gradient_concept"
OUT_FILE="gradient_concept_test"
LOG_FILE=os.path.join(os.path.join(OUT_DIR, f"importance_estimation_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}_{TARGET_TOKEN}.log"))

@torch.no_grad()
def single_inference(
    device: torch.device,
    logger: Callable = None,
    args: argparse.Namespace = None,
) -> Dict[str, Any]:
    model_class = get_model_class(
        args.model_name_or_path,
        args.processor_name,
        device=device,
        logger=logger,
        args=args,
    )

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
    
    return test_item

if __name__ == "__main__":
    logger = setup_logger(LOG_FILE)
    set_seed(SEED)
    device = get_most_free_gpu() if torch.cuda.is_available() else torch.device("cpu")
    logger.info(f"Torch: using device={device} gpu={torch.cuda.is_available()}")

    inference_args = get_arguments({
        "model_name_or_path": MODEL_NAME,
        "dataset_name": DATASET_NAME,
        "dataset_size": SUBSET_SIZE,
        "data_dir": DATA_DIR,
        "annotation_file": ANNOTATION_FILE,
        "split": DATA_SPLIT,
        "hook_names": HOOK_NAMES,
        "modules_to_hook": TARGET_FEATURE_MODULES,
        # used to filter dataset to images where the token of interest exists in caption.
        # we can set this to True so we only sample from those images for which we have a concept dictionary precomputed for.
        "select_token_of_interest_samples": True,
        "token_of_interest": TARGET_TOKEN,
        "save_dir": OUT_DIR,
        "save_filename": OUT_FILE,
        "seed": SEED,
        "processor_name": MODEL_NAME,
        "generation_mode": True,
        "exact_match_modules_to_hook": True,
        "save_only_generated_tokens": True,
    })

    test_item = single_inference(
        device=device,
        logger=logger,
        args=inference_args,
    )

    print(test_item)

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