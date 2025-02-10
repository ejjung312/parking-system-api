from ultralytics.solutions import ParkingManagement

class ParkingDetector:
    def __init__(self):
        self.parking_detector = ParkingManagement(
            model="./model/drone_best.pt",
            json_file="./boxes/parking_bounding_boxes.json",
        )

    def detect(self, frame):
        frame = self.parking_detector.process_data(frame)

        return frame