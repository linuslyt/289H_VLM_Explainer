import json
import os
from typing import Union, Tuple

CAPTIONING_PROMPT = '\nProvide a one-sentence caption for the provided image.'
UPLOADED_IMG_DIR = "./uploads"
os.makedirs(UPLOADED_IMG_DIR, exist_ok=True)
os.makedirs(os.path.join(UPLOADED_IMG_DIR, "preprocessed"), exist_ok=True)

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

def get_uploaded_img_full_path(uploaded_img_filename: str):
    return os.path.join(get_uploaded_img_dir(), uploaded_img_filename)

def get_output_hidden_state_paths(uploaded_img_path: str, hook_name: str, token_of_interest: str) -> Tuple[str, str]:
    img_name, img_extension = os.path.splitext(uploaded_img_path)
    hidden_state_filename = f"llava_user_inference_{img_name}_{token_of_interest}"
    hidden_state_full_saved_path = os.path.join(get_uploaded_img_dir(), "features", f"{hook_name}_{hidden_state_filename}.pth")
    return hidden_state_filename, hidden_state_full_saved_path

def get_preprocessed_img_stored_path(uploaded_img_filename):
    return os.path.join(get_uploaded_img_dir(), "preprocessed", f"{uploaded_img_filename}.pth")

# TODO: methods to get filenames