from ultralytics import YOLO
import cv2

class PersonDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):
        results = self.model(frame)[0]
        persons = []

        for result in results.boxes:
            cls_id = int(result.cls[0])
            if cls_id == 0:  # 'person' class
                bbox = result.xyxy[0].tolist()
                persons.append(bbox)
        return persons
