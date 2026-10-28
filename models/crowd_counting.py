class CrowdCounter:
    def __init__(self, thresholds):
        self.thresholds = thresholds

    def estimate_from_detections(self, num_detections):
        density = "LOW"
        if num_detections >= self.thresholds['high']:
            density = "HIGH"
        elif num_detections >= self.thresholds['medium']:
            density = "MEDIUM"
        return num_detections, density
