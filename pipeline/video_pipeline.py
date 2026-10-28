import cv2
import os
import yaml
import time
from models.enhancement import LowLightEnhancer
from models.super_resolution import SuperResolutionModel
from models.tracking import PersonTracker
from models.crowd_counting import CrowdCounter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class VideoPipeline:
    def __init__(self, config_path="config.yaml"):
        if not os.path.isabs(config_path) and not os.path.exists(config_path):
            config_path = os.path.join(BASE_DIR, config_path)
            
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        sr_model_path = self.config['super_resolution']['model_path']
        if not os.path.isabs(sr_model_path) and not os.path.exists(sr_model_path):
            sr_model_path = os.path.join(BASE_DIR, sr_model_path)

        track_model_path = self.config['detection']['model']
        if not os.path.isabs(track_model_path) and not os.path.exists(track_model_path):
            track_model_path = os.path.join(BASE_DIR, track_model_path)

        self.enhancer = LowLightEnhancer(method=self.config['enhancement']['method']) if self.config['enhancement']['enabled'] else None
        self.sr = SuperResolutionModel(
            model_path=sr_model_path,
            model_name=self.config['super_resolution']['model_name'],
            scale=self.config['super_resolution']['scale']
        ) if self.config['super_resolution']['enabled'] else None
        
        self.tracker = PersonTracker(
            model_name=track_model_path,
            method=self.config['tracking']['method'],
            conf_thresh=self.config['runtime']['confidence_threshold']
        ) if self.config['tracking']['enabled'] else None
        
        self.counter = CrowdCounter(self.config['crowd_counting']['density_thresholds']) if self.config['crowd_counting']['enabled'] else None

    def run(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not read video at {video_path}")
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if self.sr:
            width *= self.config['super_resolution']['scale']
            height *= self.config['super_resolution']['scale']
            
        fourcc = cv2.VideoWriter_fourcc(*'VP80')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_processing_time = 0
        total_people_detected = 0
        unique_ids = set()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            start_time = time.time()
            processed_frame = frame.copy()
            
            if self.enhancer:
                processed_frame = self.enhancer.enhance(processed_frame)
                
            if self.sr:
                processed_frame = self.sr.upscale(processed_frame)
                
            if self.tracker:
                track_results = self.tracker.track_frame(processed_frame)
                processed_frame = track_results.plot()
                
                num_people = len(track_results.boxes)
                total_people_detected += num_people
                
                if track_results.boxes.id is not None:
                    ids = track_results.boxes.id.int().cpu().tolist()
                    unique_ids.update(ids)
                    
            out.write(processed_frame)
            frame_count += 1
            total_processing_time += (time.time() - start_time)
            
        cap.release()
        out.release()
        
        avg_fps = frame_count / total_processing_time if total_processing_time > 0 else 0
        
        return {
            'frames_processed': frame_count,
            'average_fps': avg_fps,
            'total_processing_time': total_processing_time,
            'unique_tracks': len(unique_ids)
        }
