from typing import Union

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import time

app = FastAPI()


class Image(BaseModel):
    name: str
    id: int
    description: str
    file_path: str


class Model(BaseModel):
    id: int
    file_path: str


@app.get("/", response_class=HTMLResponse)
def read_root():
    return """<html>
    <head>
        <title>Some HTML in here</title>
    </head>
    <body>
        <h1>Geoguessr AI API op side</h1>
        <p>This is a simple HTML response from FastAPI.</p>
    </body>
    </html>"""


@app.get("/model/{model_id}")
def read_model(model_id: int, q: Union[str, None] = None):
    return {"model_id": model_id, "model_file_path": "1"}


@app.get("/image/{image_id}")
def read_image(image_id: int, q: Union[str, None] = None):
    im = Image()
    return {
        "image_id": image_id,
        "image_name": im.name,
        "image_filepath": im.file_path,
        "image_desc": im.description,
    }


@app.post("/submit_image/")
def submit_image():
    return {"message": "Image has been added"}


@app.get("/predicition/{image_id}")
def get_prediction(image_id: int):
    long = 1
    lat = 2
    return {"long": long, "lat": lat}


@app.get("/health")
def ping():
    return {"status": "Healthy!", "Timestamp": time.time()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7200)
