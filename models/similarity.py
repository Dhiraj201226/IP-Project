import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class FeatureExtractor:
    def __init__(self, model_name="resnet18"):
        self.model_name = model_name
        self.is_cnn = "resnet" in model_name.lower()
        if self.is_cnn:
            import torchvision.models as tv_models
            import torchvision.transforms as tv_transforms
            resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
            modules = list(resnet.children())[:-1]
            self.cnn = torch.nn.Sequential(*modules).eval()
            self.transform = tv_transforms.Compose([
                tv_transforms.ToPILImage(),
                tv_transforms.Resize((128, 128)),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            self.model = None
        else:
            if not os.path.isabs(model_name) and not os.path.exists(model_name):
                cand = os.path.join(BASE_DIR, model_name)
                if os.path.exists(cand):
                    model_name = cand
            self.model = YOLO(model_name)

    def extract(self, image):
        """
        Extracts a normalized 1D embedding vector from a BGR image crop.
        """
        if image is None or image.size == 0:
            return None

        if self.is_cnn:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            t = self.transform(rgb).unsqueeze(0)
            with torch.no_grad():
                vec = self.cnn(t).flatten().numpy()
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec

        # YOLO branch
        h, w = image.shape[:2]
        if h < 32 or w < 32:
            image = cv2.resize(image, (max(64, w), max(64, h)), interpolation=cv2.INTER_LINEAR)

        # Ultralytics model.embed returns a list of torch.Tensor
        with torch.no_grad():
            embeddings = self.model.embed(image)
            if embeddings is None or len(embeddings) == 0:
                return None
            vec = embeddings[0].detach().cpu().numpy().flatten()

        # L2 normalization
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

class PartBasedReIDMatcher:
    """
    Universal State-of-the-Art Person Re-Identification (Re-ID) matcher:
    1. Global CNN Backbone (ResNet-18, 128x64): Encodes overall silhouette, posture, height-to-width ratio, and outer garments.
    2. Overlapping Multi-Part Local Slices:
       - Head & Neck (0% - 25%): Hair, facial structure, collar.
       - Upper Torso (20% - 65%): Shirt, jacket, graphics, neckline.
       - Lower Body (55% - 100%): Trousers, shorts, skirts, shoes.
    3. Perceptually Uniform Color Modeling (CIE L*a*b* + Full HSV):
       - CIE L*a*b* (L=Lightness, a=Green/Red, b=Blue/Yellow) accurately distinguishes neutrals
         (black, charcoal, dark grey, light grey, white) as well as all chromatic colors.
       - Full HSV covers all hues, saturations, and values across [0..180, 0..256, 0..256].
    4. Gradient Texture Energy: Captures patterns, stripes, and fabric texture.
    5. Completely general and unbiased: works universally for ANY person with ANY clothing.
    """
    def __init__(self):
        import torchvision.models as tv_models
        import torchvision.transforms as tv_transforms
        resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]
        self.cnn = torch.nn.Sequential(*modules).eval()
        self.transform_global = tv_transforms.Compose([
            tv_transforms.ToPILImage(),
            tv_transforms.Resize((128, 64)),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.transform_part = tv_transforms.Compose([
            tv_transforms.ToPILImage(),
            tv_transforms.Resize((64, 64)),
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_signature(self, crop):
        if crop is None or crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            return None
        h, w = crop.shape[:2]

        # 1. Global CNN representation (128x64 standard Re-ID pedestrian aspect ratio)
        rgb_g = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            v_g = self.cnn(self.transform_global(rgb_g).unsqueeze(0)).flatten().numpy()
        norm_g = np.linalg.norm(v_g)
        v_g = v_g / (norm_g + 1e-7)

        # 2. Overlapping anatomical slices: Head (0-25%), Torso (20-65%), Lower (55-100%)
        slices = [
            crop[0:int(h * 0.25), :],
            crop[int(h * 0.20):int(h * 0.65), :],
            crop[int(h * 0.55):h, :]
        ]
        part_cnns, part_colors, part_textures = [], [], []
        for s in slices:
            if s.size == 0 or s.shape[0] < 4 or s.shape[1] < 4:
                part_cnns.append(np.zeros(512))
                part_colors.append(np.zeros(48 + 32))
                part_textures.append(np.zeros(16))
                continue

            # Deep CNN feature vector per slice
            rgb_s = cv2.cvtColor(s, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                v_p = self.cnn(self.transform_part(rgb_s).unsqueeze(0)).flatten().numpy()
            norm_p = np.linalg.norm(v_p)
            part_cnns.append(v_p / (norm_p + 1e-7))

            # Comprehensive universal color descriptor:
            # CIE L*a*b* (48 bins) captures lightness (separates black, grey, white) and chrominance
            lab = cv2.cvtColor(s, cv2.COLOR_BGR2LAB)
            h_L = cv2.calcHist([lab], [0], None, [16], [0, 256]).flatten()
            h_a = cv2.calcHist([lab], [1], None, [16], [0, 256]).flatten()
            h_b = cv2.calcHist([lab], [2], None, [16], [0, 256]).flatten()
            lab_hist = np.concatenate([h_L, h_a, h_b])
            lab_hist = lab_hist / (np.linalg.norm(lab_hist) + 1e-7)

            # Full HSV (32 bins) across full ranges
            hsv = cv2.cvtColor(s, cv2.COLOR_BGR2HSV)
            h_H = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
            h_S = cv2.calcHist([hsv], [1], None, [8], [0, 256]).flatten()
            h_V = cv2.calcHist([hsv], [2], None, [8], [0, 256]).flatten()
            hsv_hist = np.concatenate([h_H, h_S, h_V])
            hsv_hist = hsv_hist / (np.linalg.norm(hsv_hist) + 1e-7)

            col = np.concatenate([lab_hist, hsv_hist])
            part_colors.append(col / (np.linalg.norm(col) + 1e-7))

            # Texture energy: spatial gradient magnitude (distinguishes solid from patterned fabrics)
            gray = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            tex_hist = cv2.calcHist([mag], [0], None, [16], [0, 256]).flatten()
            part_textures.append(tex_hist / (np.linalg.norm(tex_hist) + 1e-7))

        return v_g, part_cnns, part_colors, part_textures

    def compute_similarity(self, sig1, sig2):
        if sig1 is None or sig2 is None:
            return 0.0
        vg1, cnn1, col1, tex1 = sig1
        vg2, cnn2, col2, tex2 = sig2

        # Global CNN silhouette & posture similarity (30%)
        s_global = max(0.0, float(np.dot(vg1, vg2)))

        # Multi-part similarity: Head (15%), Torso/Upper Body (35%), Lower Body (20%)
        pw = [0.15, 0.35, 0.20]
        s_parts = 0.0
        for i in range(3):
            c_sim = max(0.0, float(np.dot(cnn1[i], cnn2[i])))
            col_sim = max(0.0, float(np.dot(col1[i], col2[i])))
            tex_sim = max(0.0, float(np.dot(tex1[i], tex2[i])))
            part_score = 0.50 * c_sim + 0.40 * col_sim + 0.10 * tex_sim
            s_parts += pw[i] * part_score

        total = 0.30 * s_global + s_parts
        return float(np.clip(total, 0.0, 1.0))

    def rank_candidates(self, ref_image, candidate_crops, bboxes=None):
        if ref_image is None or len(candidate_crops) == 0:
            return [], None
        ref_sig = self.extract_signature(ref_image)
        if ref_sig is None:
            return [], None

        results = []
        for i, crop in enumerate(candidate_crops):
            cand_sig = self.extract_signature(crop)
            sim = self.compute_similarity(ref_sig, cand_sig)
            pct = sim * 100.0
            bbox = bboxes[i] if bboxes is not None and i < len(bboxes) else None
            results.append({
                "index": i,
                "score": sim,
                "percentage": pct,
                "bbox": bbox,
                "crop": crop
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        for rank, item in enumerate(results, start=1):
            item["rank"] = rank
        top_match = results[0] if len(results) > 0 else None
        return results, top_match

class CosineSimilarityMatcher:
    def __init__(self, model_name="part_reid"):
        self.model_name = model_name
        self.use_part_reid = "part" in model_name.lower() or "reid" in model_name.lower()
        if self.use_part_reid:
            self.reid_matcher = PartBasedReIDMatcher()
            self.extractor = None
        else:
            self.reid_matcher = None
            self.extractor = FeatureExtractor(model_name=model_name)

    def compute_similarity(self, vec1, vec2):
        """
        Computes cosine similarity between two normalized vectors.
        Returns float in range [-1.0, 1.0].
        """
        if vec1 is None or vec2 is None:
            return 0.0
        dot = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        sim = dot / (norm1 * norm2)
        return float(np.clip(sim, -1.0, 1.0))

    def rank_candidates(self, ref_image, candidate_crops, bboxes=None):
        """
        Compares ref_image to a list of candidate crops.
        Returns:
            ranked_results: List of dicts sorted by similarity descending.
            top_match: The dict of the highest-ranking candidate (or None if no candidates).
        """
        if self.use_part_reid:
            return self.reid_matcher.rank_candidates(ref_image, candidate_crops, bboxes)

        if ref_image is None or len(candidate_crops) == 0:
            return [], None

        ref_vec = self.extractor.extract(ref_image)
        if ref_vec is None:
            return [], None

        results = []
        for i, crop in enumerate(candidate_crops):
            cand_vec = self.extractor.extract(crop)
            sim = self.compute_similarity(ref_vec, cand_vec)
            pct = max(0.0, sim) * 100.0

            bbox = bboxes[i] if bboxes is not None and i < len(bboxes) else None
            results.append({
                "index": i,
                "score": sim,
                "percentage": pct,
                "bbox": bbox,
                "crop": crop
            })

        # Sort descending by score
        results.sort(key=lambda x: x["score"], reverse=True)

        # Assign 1-based ranks
        for rank, item in enumerate(results, start=1):
            item["rank"] = rank

        top_match = results[0] if len(results) > 0 else None
        return results, top_match
