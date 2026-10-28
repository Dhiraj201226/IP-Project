import cv2
import numpy as np

class LowLightEnhancer:
    def __init__(self, method="clahe"):
        self.method = method
        if self.method == "clahe":
            self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def enhance(self, image):
        if self.method == "clahe":
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L-channel
            cl = self.clahe.apply(l)
            
            # Merge channels
            limg = cv2.merge((cl, a, b))
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            # Edge-preserving smoothing to suppress low-light sensor grain
            enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=35, sigmaSpace=35)
            return enhanced
        else:
            # Fallback identity
            return image
