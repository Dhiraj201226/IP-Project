import os
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class PersonDetector:
    def __init__(self, model_name="yolov8n.pt", conf_thresh=0.20, iou_thresh=0.60):
        if not os.path.isabs(model_name) and not os.path.exists(model_name):
            cand = os.path.join(BASE_DIR, model_name)
            if os.path.exists(cand):
                model_name = cand
        self.model = YOLO(model_name)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

    def detect(self, image):
        # Class 0 is person in COCO
        results = self.model.predict(
            image,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            classes=[0],
            verbose=False
        )
        return results[0]
