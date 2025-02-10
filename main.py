import cv2
import numpy as np
from fastapi import FastAPI, Request
from starlette.responses import Response

from license_plate import LicensePlateDetector
from parking import ParkingDetector

app = FastAPI()
detector = LicensePlateDetector()
parking_detector = ParkingDetector()

@app.post("/predict_license_plate")
async def predict_license_plate(request: Request):
    file_data = await request.body()  # Raw byte data 읽기
    nparr = np.frombuffer(file_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    processed_img = detector.detect(image)

    _, img_encoded = cv2.imencode('.jpg', processed_img)
    return Response(content=img_encoded.tobytes(), media_type="image/jpeg")

@app.post("/predict_parking_monitoring")
async def predict_parking_monitoring(request: Request):
    file_data = await request.body()  # Raw byte data 읽기
    nparr = np.frombuffer(file_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    image = cv2.resize(image, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)

    processed_img = parking_detector.detect(image)

    _, img_encoded = cv2.imencode('.jpg', processed_img)
    return Response(content=img_encoded.tobytes(), media_type="image/jpeg")

@app.get("/")
def read_root():
    return {"Hello": "World"}