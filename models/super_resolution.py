import cv2
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class SuperResolutionModel:
    def __init__(self, model_path="models/weights/EDSR_x4.pb", model_name="edsr", scale=2, method="fast"):
        self.model_path = model_path
        self.model_name = model_name
        self.scale = scale
        self.method = method # 'fast' (real-time ~10ms) or 'edsr' (deep DNN)
        self.sr = None
        self.loaded = False

    def _load_edsr(self):
        if self.loaded:
            return
        if not os.path.isabs(self.model_path) and not os.path.exists(self.model_path):
            cand = os.path.join(BASE_DIR, self.model_path)
            if os.path.exists(cand):
                self.model_path = cand

        if os.path.exists(self.model_path):
            self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
            self.sr.readModel(self.model_path)
            self.sr.setModel(self.model_name, 4)
            self.loaded = True
        else:
            print(f"Warning: Super resolution model not found at {self.model_path}")

    def upscale(self, image):
        if image is None or image.size == 0:
            return image

        # 1. Fast Super-Resolution (Default: Bicubic + Edge-Preserving Unsharp Mask) ~10ms
        if self.method == "fast":
            h, w = image.shape[:2]
            target_scale = max(1, min(self.scale, 4))
            upscaled = cv2.resize(image, (w * target_scale, h * target_scale), interpolation=cv2.INTER_CUBIC)
            # Unsharp mask to enhance high-frequency edge details
            gaussian = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=1.5)
            sharpened = cv2.addWeighted(upscaled, 1.4, gaussian, -0.4, 0)
            return sharpened

        # 2. Deep EDSR Super-Resolution (CPU intensive, only if explicitly requested)
        self._load_edsr()
        if not self.loaded or self.sr is None:
            return image

        import numpy as np
        image = np.ascontiguousarray(image)
        h, w = image.shape[:2]
        max_dim = 360 # Bound input to prevent multi-minute freezes on CPU
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        try:
            return self.sr.upsample(image)
        except Exception as e:
            print(f"Super resolution failed: {e}")
            return image
