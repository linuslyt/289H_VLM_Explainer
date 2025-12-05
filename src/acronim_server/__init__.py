import json
import os
from typing import Union, Tuple

CAPTIONING_PROMPT = '\nProvide a one-sentence caption for the provided image.'
UPLOADED_IMG_DIR = "./uploads"
PREPROCESSED_INPUTS_DIR = "preprocessed_inputs"
SAVED_HIDDEN_STATES_DIR = "features"
os.makedirs(UPLOADED_IMG_DIR, exist_ok=True)
os.makedirs(os.path.join(UPLOADED_IMG_DIR, PREPROCESSED_INPUTS_DIR), exist_ok=True)
os.makedirs(os.path.join(UPLOADED_IMG_DIR, SAVED_HIDDEN_STATES_DIR), exist_ok=True)


# Use passthrough=False to return tuple for yield() directly
def new_event(event_type: str, data: Union[str, dict], passthrough=True):
    # https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format
    if passthrough:
        return (event_type, data)
    return f"event: {event_type}\ndata: {data if type(data) == str else json.dumps(data)}\n\n"

def new_log_event(logger, msg: str, passthrough=True):
    logger.info(msg)
    return new_event(event_type="log", data=msg, passthrough=passthrough)

def get_uploaded_img_dir():
    return UPLOADED_IMG_DIR

def get_saved_hidden_states_dir():
    return SAVED_HIDDEN_STATES_DIR

def get_uploaded_img_full_path(uploaded_img_filename: str):
    return os.path.join(get_uploaded_img_dir(), uploaded_img_filename)

def get_output_hidden_state_paths(uploaded_img_path: Union[str, None], hook_name: str, token_of_interest: str) -> Tuple[str, str]:
    if uploaded_img_path == None:
        img_name = "training_samples"
    else:
        img_name, _ = os.path.splitext(uploaded_img_path)
    
    hidden_state_filename = f"_{img_name}__{token_of_interest}"
    hidden_state_full_saved_path = os.path.join(get_uploaded_img_dir(), "features", f"{hook_name}_{hidden_state_filename}.pth")
    return hidden_state_filename, hidden_state_full_saved_path

def get_preprocessed_img_stored_path(uploaded_img_filename):
    img_name, _ = os.path.splitext(uploaded_img_filename)
    return os.path.join(get_uploaded_img_dir(), PREPROCESSED_INPUTS_DIR, f"{img_name}.pth")

# TODO: methods to get filenames