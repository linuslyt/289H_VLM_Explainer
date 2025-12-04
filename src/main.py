import asyncio
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import io
import json
import os
from pydantic import BaseModel
from PIL import Image, UnidentifiedImageError
import torch
from typing import Union

from helpers.utils import (get_most_free_gpu)

app = FastAPI()

# CORS configuration
origins = [
    "http://localhost",
    "http://localhost:3000",  # TODO: update for frontend port
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

UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
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
        save_path = os.path.join(UPLOAD_DIR, new_filename)
        image.save(save_path, "JPEG", exist_ok=True)

        return {
            "filename": new_filename,
        }

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# TODO: typing
def new_event(event_type: str, data: Union[str, dict]):
    # https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#event_stream_format
    return f"event: {event_type}\ndata: {data if type(data) == str else json.dumps(data)}\n\n"

async def captioning_pipeline():
    await asyncio.sleep(0.5)
    yield new_event(event_type="log", data="Generating image caption...")

    await asyncio.sleep(0.5)
    caption = "placeholder"
    yield new_event(event_type="scores", data=caption) # On frontend, retrieve with event.data

@app.get("/caption-image")
async def caption_image():
    return StreamingResponse(captioning_pipeline(), media_type="text/event-stream")

async def importance_estimation_piipeline():
    await asyncio.sleep(0.5)
    yield new_event(event_type="log", data="Performing dictionary learning...")

    await asyncio.sleep(0.5)
    yield new_event(event_type="log", data="Calculating concept importance...")

    # Scores
    await asyncio.sleep(0.5)
    scores = {
        "importance": [],
        "activations": [],
    }
    yield new_event(event_type="scores", data=scores) # On frontend, retrieve with JSON.parse(event.data);

@app.get("/importance-estimation")
async def importance_estimation():
    return StreamingResponse(importance_estimation_piipeline(), media_type="text/event-stream")

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