from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import asyncio
import torch

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