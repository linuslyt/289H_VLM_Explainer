import asyncio
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
import io
import json
import os
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError
import torch
from typing import Union

from acronim_server import new_event, get_uploaded_img_saved_path, get_grounding_image_dir
from acronim_server.inference import (caption_uploaded_img, get_hidden_state_for_input, get_hidden_states_for_training_samples, 
                                      DICTIONARY_LEARNING_MIN_SAMPLE_SIZE, COCO_TRAIN_FULL_SIZE)
from acronim_server.dictionary_learning import learn_concept_dictionary_for_token
from acronim_server.concept_importance import calculate_concept_importance

from helpers.utils import (get_most_free_gpu)

# TODO: type checking with pydantic

app = FastAPI()

# CORS configuration
origins = [
    "http://localhost",
    "http://localhost:5173",  # Vite port
    "http://localhost:8080",
    # Use "*" only for dev/internal tools to allow ALL connections
    # "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload/", status_code=status.HTTP_201_CREATED)
async def save_image(file: UploadFile = File(...)):
    print("Processing uploaded image...")
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPEG, PNG, and WebP are accepted.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # LLaVA processor (based on CLIP) expects 3 channels.
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Save as JPG. Resizing etc. will be handled by LLaVA processor
        new_filename = f"{file.filename.split('.')[0]}.jpg"
        save_path = get_uploaded_img_saved_path(new_filename)
        image.save(save_path, "JPEG", exist_ok=True)

        return {
            "filename": new_filename,
        }

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/caption-image")
async def caption_image(uploaded_img_path: str):
    print(f"captioning image for {uploaded_img_path}")
    return StreamingResponse(caption_uploaded_img(uploaded_img_path), media_type="text/event-stream")

async def importance_estimation_pipeline(uploaded_img_path: str, token_of_interest: str, sampled_subset_size: int, n_concepts: int, force_recompute: bool):
    # Clamp sampled subset
    sampled_subset_size = max(DICTIONARY_LEARNING_MIN_SAMPLE_SIZE, min(sampled_subset_size, COCO_TRAIN_FULL_SIZE))
    yield new_event(event_type="log", data=f"Clamping sampled_subset_size to {sampled_subset_size}.", passthrough=False)

    # Get hidden state for input instance wrt target token
    async for event_type, data in get_hidden_state_for_input(uploaded_img_path, token_of_interest, force_recompute):
        if event_type == "return":
            print("retdata:", data)
            uploaded_img_hidden_state_path, uploaded_img_hidden_state = data
        else:
            yield new_event(event_type=event_type, data=data, passthrough=False)

    print("here2:", uploaded_img_hidden_state_path)

    # Get hidden states from training data samples that contain token of interest in ground truth caption
    async for event_type, data in get_hidden_states_for_training_samples(token_of_interest, sampled_subset_size, force_recompute):
        if event_type == "return":
            relevant_samples_hidden_state = data
        else:
            yield new_event(event_type=event_type, data=data, passthrough=False)
    
    print(relevant_samples_hidden_state.keys())

    # Learn concept dictionary
    async for event_type, data in learn_concept_dictionary_for_token(token_of_interest, sampled_subset_size, n_concepts, force_recompute):
        if event_type == "return":
            concept_dict_for_token = data
        else:
            yield new_event(event_type=event_type, data=data, passthrough=False)
    
    print(concept_dict_for_token.keys())

    # Calculate concept importance
    async for event_type, data in calculate_concept_importance(token_of_interest, uploaded_img_hidden_state_path, uploaded_img_hidden_state, concept_dict_for_token, n_concepts, force_recompute):
        if event_type == "return":
            results = data
        else:
            yield new_event(event_type=event_type, data=data, passthrough=False)

    # Unprocessed result format:
    # {
    #     "activations": concept_activations,
    #     "importance_scores": concept_importance_scores,
    #     "indices_by_importance": indices_by_importance,
    #     "indices_by_activations": indices_by_activations,
    #     "text_groundings": concept_dict['text_grounding'],
    #     "image_grounding_paths": concept_dict['image_grounding_paths'], # stripped of directory name below before returning
    # }

    print(results)
    results["image_grounding_paths"] = [[os.path.basename(img_path) for img_path in grounding_list] for grounding_list in results["image_grounding_paths"]]
    yield new_event(event_type="return", data=results, passthrough=False)
    return

@app.get("/importance-estimation")
async def importance_estimation(uploaded_img_path: str, token_of_interest: str, 
                                sampled_subset_size: int = 5000, n_concepts: int = 10,
                                force_recompute: bool = False):
    return StreamingResponse(importance_estimation_pipeline(uploaded_img_path, token_of_interest, sampled_subset_size, n_concepts, force_recompute), media_type="text/event-stream")

# Gronuding images can be retrieved by filename - e.g. http://localhost:8000/grounding-images/COCO_train2014_000000095381.jpg
app.mount("/grounding-images", StaticFiles(directory=get_grounding_image_dir()), name="grounding-images")

@app.get("/status")
def get_status(): # Return server and CUDA status
    return {"status": "operational", "cuda_available": torch.cuda.is_available(), "most_free_device_idx": get_most_free_gpu()}

# test Server-Sent Event (SSE) streaming
async def gen_dummy_streaming_output():
    total = 10
    for i in range(total):
        await asyncio.sleep(0.5)  # Simulate work
        percent = int((i + 1) / total * 100)
        # Yield SSE format: "data: <message>\n\n"
        # frontend will read this line by line
        yield f"data: {percent}\n"

@app.get("/test-streaming")
async def test_streaming():
    # Media type 'text/event-stream' is standard for live updates
    return StreamingResponse(gen_dummy_streaming_output(), media_type="text/event-stream")