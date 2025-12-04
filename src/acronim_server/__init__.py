import json
import os
from typing import Union, Tuple

CAPTIONING_PROMPT = '\nProvide a one-sentence caption for the provided image.'
UPLOADED_IMG_DIR = "./uploads"
os.makedirs(UPLOADED_IMG_DIR, exist_ok=True)

# TODO: typing
def new_event(event_type: str, data: Union[str, dict]):
    # https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format
    return f"event: {event_type}\ndata: {data if type(data) == str else json.dumps(data)}\n\n"

def new_log_event(logger, msg: str):
    logger.info(msg)
    return new_event(event_type="log", data=msg)

def get_uploaded_img_dir():
    return UPLOADED_IMG_DIR

def get_uploaded_img_path(uploaded_img_filename: str):
    return os.path.join(get_uploaded_img_dir(), uploaded_img_filename)

def get_output_hidden_state_path(uploaded_img_path: str, hook_name: str) -> Tuple[str, str]:
    img_name, img_extension = os.path.splitext(uploaded_img_path)
    full_filename = f"llava_user_inference_{img_name}_{TARGET_TOKEN}"
    full_saved_path = os.path.join(OUT_DIR, "features", f"{HOOK_NAMES[0]}_{test_item_filename}.pth")
    return full_filename, full_saved_path

# TODO: methods to get filenames