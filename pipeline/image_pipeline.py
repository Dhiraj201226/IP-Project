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
            
        sr_model_path = self.config['super_resolution']['model_path']
        if not os.path.isabs(sr_model_path) and not os.path.exists(sr_model_path):
            sr_model_path = os.path.join(BASE_DIR, sr_model_path)

        det_model_path = self.config['detection']['model']
        if not os.path.isabs(det_model_path) and not os.path.exists(det_model_path):
            det_model_path = os.path.join(BASE_DIR, det_model_path)

        self.enhancer = LowLightEnhancer(method=self.config['enhancement']['method']) if self.config['enhancement']['enabled'] else None
        self.sr = SuperResolutionModel(
            model_path=sr_model_path,
            model_name=self.config['super_resolution']['model_name'],
            scale=self.config['super_resolution']['scale']
        ) if self.config['super_resolution']['enabled'] else None
        
        self.detector = PersonDetector(
            model_name=det_model_path,
            conf_thresh=self.config['runtime']['confidence_threshold']
        ) if self.config['detection']['enabled'] else None
        
        self.counter = CrowdCounter(self.config['crowd_counting']['density_thresholds']) if self.config['crowd_counting']['enabled'] else None

    def run(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")
            
        results = {}
        processed_image = image.copy()
        
        start_time = time.time()
        
        # Enhancement
        if self.enhancer:
            processed_image = self.enhancer.enhance(processed_image)
            results['enhanced'] = processed_image.copy()
            
        # Super Resolution
        if self.sr:
            processed_image = self.sr.upscale(processed_image)
            results['super_resolved'] = processed_image.copy()
            
        # Detection
        if self.detector:
            det_results = self.detector.detect(processed_image)
            results['detected'] = det_results.plot() # Ultralytics returns annotated image with plot()
            
            num_people = len(det_results.boxes)
            results['num_detected'] = num_people
            
            # Crowd counting
            if self.counter:
                count, density = self.counter.estimate_from_detections(num_people)
                results['crowd_count'] = count
                results['crowd_density'] = density
                
        results['processing_time'] = time.time() - start_time
        results['final_image'] = results.get('detected', processed_image)
        
        return results
