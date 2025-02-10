import cv2
import numpy as np
import base64
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
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

    processed_img, license_plate_img, license_plate_text = detector.detect(image)

    _, img_encoded = cv2.imencode('.jpg', processed_img)
    # Base64 인코딩
    processed_img64 = base64.b64encode(img_encoded.tobytes()).decode("utf-8")

    license_plate_img64 = None
    if license_plate_img is not None:
        _, license_plate_img_encoded = cv2.imencode('.jpg', license_plate_img)
        # Base64 인코딩
        license_plate_img64 = base64.b64encode(license_plate_img_encoded.tobytes()).decode("utf-8")

    if license_plate_text is None:
        license_plate_text = ""

    content = {"processed_img": processed_img64, "license_plate_img": license_plate_img64, "license_plate_text": license_plate_text}

    return JSONResponse(content=content)

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

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)