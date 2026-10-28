import cv2
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class SuperResolutionModel:
    def __init__(self, model_path="models/weights/EDSR_x4.pb", model_name="edsr", scale=4):
        self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
        if not os.path.isabs(model_path) and not os.path.exists(model_path):
            cand = os.path.join(BASE_DIR, model_path)
            if os.path.exists(cand):
                model_path = cand
        self.model_path = model_path
        self.model_name = model_name
        self.scale = scale
        self.loaded = False
        
        if os.path.exists(self.model_path):
            self.sr.readModel(self.model_path)
            self.sr.setModel(self.model_name, self.scale)
            self.loaded = True
        else:
            print(f"Warning: Super resolution model not found at {self.model_path}")

    def upscale(self, image):
        if not self.loaded:
            return image
            
        import numpy as np
        image = np.ascontiguousarray(image)
        
        # Prevent OpenCV integer overflow for very large images during x4 upscaling
        h, w = image.shape[:2]
        max_dim = 640
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            
        try:
            return self.sr.upsample(image)
        except Exception as e:
            print(f"Super resolution failed: {e}")
            return image # fallback on failure
