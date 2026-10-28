from ultralytics import YOLO

class PersonTracker:
    def __init__(self, model_name="yolov8n.pt", method="bytetrack", conf_thresh=0.25):
        self.model = YOLO(model_name)
        self.method = method
        self.conf_thresh = conf_thresh
        # tracker configuration file in ultralytics is "bytetrack.yaml" or "botsort.yaml"
        self.tracker_yaml = f"{self.method}.yaml"

    def track_frame(self, image, persist=True):
        try:
            results = self.model.track(image, persist=persist, conf=self.conf_thresh, classes=[0], tracker=self.tracker_yaml, verbose=False)
            return results[0]
        except Exception:
            # Fallback to standard person detection if linear assignment fails
            results = self.model.predict(image, conf=self.conf_thresh, classes=[0], verbose=False)
            return results[0]
