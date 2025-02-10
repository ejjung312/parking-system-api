import cv2
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Union
from pydantic import BaseModel
from starlette.responses import Response
from license_plate import LicensePlateDetector

app = FastAPI()
detector = LicensePlateDetector()

def detect_license(image):
    result = detector.detect(image)
    return result

@app.post("/predict_license_plate")
async def predict_license_plate(request: Request):
    file_data = await request.body()  # Raw byte data 읽기
    nparr = np.frombuffer(file_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    processed_img = detect_license(img)

    _, img_encoded = cv2.imencode('.jpg', processed_img)
    return Response(content=img_encoded.tobytes(), media_type="image/jpeg")

@app.get("/")
def read_root():
    return {"Hello": "World"}