import cv2
from ultralytics import YOLO
from sort.sort import *
from util import *

class LicensePlateDetector:
    def __init__(self):
        self.yolo = YOLO('./model/yolo11n.pt')
        self.licese_plate_detector = YOLO('./model/license_plate_best.pt')

        self.mot_tracker = Sort()

        self.vehicles = [2,3,5,7] # car, motorcycle, bus, truck

    def detect(self, frame):
        detections = self.yolo(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in self.vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        track_ids = self.mot_tracker.update(np.asarray(detections_))

        # detect license plates
        license_plates = self.licese_plate_detector(frame)[0]
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

            if car_id != -1:
                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

                H, W, _ = license_plate_crop.shape

                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                # print(license_plate_text, license_plate_text_score)

                if license_plate_text is not None:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color=(255, 0, 0), thickness=3)

                    return frame, license_plate_crop, license_plate_text

        return frame, None, None