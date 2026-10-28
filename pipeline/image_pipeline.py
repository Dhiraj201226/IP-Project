import cv2
import os
import yaml
import time
from models.enhancement import LowLightEnhancer
from models.super_resolution import SuperResolutionModel
from models.detection import PersonDetector
from models.crowd_counting import CrowdCounter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class ImagePipeline:
    def __init__(self, config_path="config.yaml"):
        if not os.path.isabs(config_path) and not os.path.exists(config_path):
            config_path = os.path.join(BASE_DIR, config_path)
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self._enhancer = None
        self._sr = None
        self._detector = None
        self._counter = None

    @property
    def enhancer(self):
        if self._enhancer is None and self.config.get('enhancement', {}).get('enabled', False):
            self._enhancer = LowLightEnhancer(method=self.config['enhancement']['method'])
        return self._enhancer

    @property
    def sr(self):
        if self._sr is None and self.config.get('super_resolution', {}).get('enabled', False):
            sr_model_path = self.config['super_resolution']['model_path']
            if not os.path.isabs(sr_model_path) and not os.path.exists(sr_model_path):
                sr_model_path = os.path.join(BASE_DIR, sr_model_path)
            self._sr = SuperResolutionModel(
                model_path=sr_model_path,
                model_name=self.config['super_resolution']['model_name'],
                scale=self.config['super_resolution'].get('scale', 2),
                method=self.config['super_resolution'].get('method', 'fast')
            )
        return self._sr

    @property
    def detector(self):
        if self._detector is None and self.config.get('detection', {}).get('enabled', False):
            det_model_path = self.config['detection']['model']
            if not os.path.isabs(det_model_path) and not os.path.exists(det_model_path):
                det_model_path = os.path.join(BASE_DIR, det_model_path)
            self._detector = PersonDetector(
                model_name=det_model_path,
                conf_thresh=self.config['runtime']['confidence_threshold']
            )
        return self._detector

    @property
    def counter(self):
        if self._counter is None and self.config.get('crowd_counting', {}).get('enabled', False):
            self._counter = CrowdCounter(self.config['crowd_counting']['density_thresholds'])
        return self._counter

    def run(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")

        results = {}
        processed_image = image.copy()

        start_time = time.time()

        # Enhancement (only runs if enabled)
        if self.config.get('enhancement', {}).get('enabled', False) and self.enhancer:
            processed_image = self.enhancer.enhance(processed_image)
            results['enhanced'] = processed_image.copy()

        # Super Resolution (only runs if enabled)
        if self.config.get('super_resolution', {}).get('enabled', False) and self.sr:
            processed_image = self.sr.upscale(processed_image)
            results['super_resolved'] = processed_image.copy()

        # Detection (only runs if enabled)
        if self.config.get('detection', {}).get('enabled', False) and self.detector:
            det_results = self.detector.detect(processed_image)
            results['detected'] = det_results.plot()

            num_people = len(det_results.boxes)
            results['num_detected'] = num_people

            # Crowd counting (only runs if enabled)
            if self.config.get('crowd_counting', {}).get('enabled', False) and self.counter:
                count, density = self.counter.estimate_from_detections(num_people)
                results['crowd_count'] = count
                results['crowd_density'] = density

        results['processing_time'] = time.time() - start_time
        results['final_image'] = results.get('detected', processed_image)

        return results
