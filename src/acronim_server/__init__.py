import json
import os
from typing import Union, Tuple
import torch
from models import get_model_class, get_module_by_path
from helpers.utils import get_most_free_gpu
from datetime import datetime
from helpers.arguments import get_arguments
from helpers.logger import log_args, setup_logger


CAPTIONING_PROMPT = '\nProvide a one-sentence caption for the provided image.'
UPLOADED_IMG_DIR = "./uploads"
PREPROCESSED_INPUTS_DIR = "preprocessed_inputs"
SAVED_HIDDEN_STATES_DIR = "features"
SAVED_CONCEPT_DICTS_DIR = "concept_dicts"
DATASET_NAME="coco"
DICTIONARY_LEARNING_DATA_SPLIT="train"
DICT_ANALYSIS_NAME="decompose_activations_text_grounding_image_grounding"
GROUNDING_IMG_DIR="/media/data/ytllam/coco/train2014"

os.makedirs(UPLOADED_IMG_DIR, exist_ok=True)
os.makedirs(os.path.join(UPLOADED_IMG_DIR, PREPROCESSED_INPUTS_DIR), exist_ok=True)
os.makedirs(os.path.join(UPLOADED_IMG_DIR, SAVED_HIDDEN_STATES_DIR), exist_ok=True)
os.makedirs(os.path.join(UPLOADED_IMG_DIR, SAVED_CONCEPT_DICTS_DIR), exist_ok=True)

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
DICT_ANALYSIS_NAME="decompose_activations_text_grounding_image_grounding"
device = get_most_free_gpu() if torch.cuda.is_available() else torch.device("cpu")

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
    "save_dir": OUT_DIR,
    "save_filename": "",
    "seed": SEED,
    "processor_name": MODEL_NAME,
    "generation_mode": True,
    "exact_match_modules_to_hook": True,
    "save_only_generated_tokens": True,
    "batch_size": 1,
}

LOG_FILE=os.path.join(os.path.join(OUT_DIR, f"acronim_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.log"))
logger = setup_logger(LOG_FILE)

default_model_args = get_arguments(DEFAULT_ARGS)
llava_model_class = get_model_class(
    model_name_or_path=MODEL_NAME,
    processor_name=MODEL_NAME,
    device=device,
    logger=logger,
    args=default_model_args, # larger arg dict not needed for model setup
)
model = llava_model_class.get_model()
logger.info(f"Loading model={MODEL_NAME} on device={device}, gpu={torch.cuda.is_available()}...")

dict_learning_device = torch.device("cpu")
dict_learning_model_class = get_model_class(
    model_name_or_path=MODEL_NAME,
    processor_name=MODEL_NAME,
    device=dict_learning_device,
    logger=logger,
    args=default_model_args,
)

def get_llava_model_class():
    return llava_model_class

def get_dict_model_class():
    return dict_learning_model_class

# Use passthrough=False to return tuple for yield() directly
def new_event(event_type: str, data: Union[str, dict], passthrough=True):
    # https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format
    if passthrough:
        return (event_type, data)
    return f"event: {event_type}\ndata: {data if type(data) == str else json.dumps(data)}\n\n"

def new_log_event(logger, msg: str, passthrough=True):
    logger.info(msg)
    return new_event(event_type="log", data=msg, passthrough=passthrough)

def get_grounding_image_dir():
    return GROUNDING_IMG_DIR

def get_uploaded_img_dir():
    return UPLOADED_IMG_DIR

def get_preprocessed_img_dir():
    return os.path.join(UPLOADED_IMG_DIR, PREPROCESSED_INPUTS_DIR)

def get_saved_hidden_states_dir():
    return os.path.join(get_uploaded_img_dir(), SAVED_HIDDEN_STATES_DIR)

def get_saved_concept_dicts_dir():
    return os.path.join(get_uploaded_img_dir(), SAVED_CONCEPT_DICTS_DIR)

def get_uploaded_img_saved_path(uploaded_img_filename: str):
    return os.path.join(get_uploaded_img_dir(), uploaded_img_filename)

def get_preprocessed_img_saved_path(uploaded_img_filename):
    img_name, _ = os.path.splitext(uploaded_img_filename)
    return os.path.join(get_preprocessed_img_dir(), f"{img_name}.pth")

def get_output_hidden_state_paths(token_of_interest: str, uploaded_img_path: Union[str, None]=None, hook_name: str = "save_hidden_states_for_token_of_interest") -> Tuple[str, str]:
    if uploaded_img_path == None:
        img_name = "training_samples"
    else:
        img_name, _ = os.path.splitext(uploaded_img_path)
    
    hidden_state_filename = f"_{img_name}__{token_of_interest}"
    hidden_state_full_saved_path = os.path.join(get_uploaded_img_dir(), "features", f"{hook_name}_{hidden_state_filename}.pth")
    return hidden_state_filename, hidden_state_full_saved_path

def get_output_concept_dictionary_path(token_of_interest):
    concept_dict_filename = f"{DATASET_NAME}_{DICTIONARY_LEARNING_DATA_SPLIT}_{token_of_interest}_dict"
    concept_dict_full_saved_path = os.path.join(get_saved_concept_dicts_dir(), f"{DICT_ANALYSIS_NAME}_{concept_dict_filename}.pth")
    return concept_dict_filename, concept_dict_full_saved_path