from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel

import os
import time

model_path = "/models"


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up")
    os.makedirs(model_path, exist_ok=True)

    yield

    print("Shutting down")


app = FastAPI(lifetime=lifespan)


class Image(BaseModel):
    name: str
    id: int
    description: str
    file_path: str


class Model(BaseModel):
    id: int
    file_path: str


@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """<html>
    <head>
        <title>Some HTML in here</title>
    </head>
    <body>
        <h1>Geoguessr AI API</h1>
        <p>This is the root page for the FastAPI.</p>
        <p>GGAI{W0w_Y0u_f0und_m3}</p>
    </body>
    </html>"""


@app.get("/model/{model_id}")
async def read_model(model_id: int):
    return {"model_id": model_id, "model_file_path": "1"}


@app.get("/image/{image_id}")
async def read_image(image_id: int):
    im = Image()
    return {
        "image_id": image_id,
        "image_name": im.name,
        "image_filepath": im.file_path,
        "image_desc": im.description,
    }


@app.post("/submit_image/")
async def submit_image():
    return {"message": "Image has been added"}


@app.get("/predicition/{image_id}")
async def get_prediction(image_id: int):
    long = 1
    lat = 2
    return {"long": long, "lat": lat}


@app.get("/health")
async def ping():
    return {"status": "Healthy!", "Timestamp": time.time()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7200)
