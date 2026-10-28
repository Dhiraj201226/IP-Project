import cv2
import numpy as np
import time
from models.enhancement import LowLightEnhancer
from models.detection import PersonDetector
from models.face_detector import FaceDetector
from models.similarity import CosineSimilarityMatcher

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class SimilarityPipeline:
    def __init__(self, model_name="part_reid", enhancement_method="clahe"):
        det_model = "yolov8n.pt"
        if not os.path.isabs(det_model) and not os.path.exists(det_model):
            cand = os.path.join(BASE_DIR, det_model)
            if os.path.exists(cand):
                det_model = cand

        self.enhancer = LowLightEnhancer(method=enhancement_method)
        self.person_detector = PersonDetector(model_name=det_model, conf_thresh=0.20)
        self.face_detector = FaceDetector(person_detector=self.person_detector)
        self.person_matcher = CosineSimilarityMatcher(model_name=model_name)
        self.face_matcher = CosineSimilarityMatcher(model_name="resnet18")

    def crop_bounding_box(self, image, bbox, pad_pct=0.05):
        """
        Crops a bounding box [x1, y1, x2, y2] from image with optional padding.
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox

        pad_w = int((x2 - x1) * pad_pct)
        pad_h = int((y2 - y1) * pad_pct)

        x1_pad = max(0, x1 - pad_w)
        y1_pad = max(0, y1 - pad_h)
        x2_pad = min(w, x2 + pad_w)
        y2_pad = min(h, y2 + pad_h)

        if x2_pad <= x1_pad or y2_pad <= y1_pad:
            return None

        return image[y1_pad:y2_pad, x1_pad:x2_pad].copy()

    def run(self, ref_image, target_image, mode="person", enhance=False):
        """
        Executes similarity search and ranking:
        1. Cleans and enhances both images under low light.
        2. Detects persons or faces in target image.
        3. Crops each candidate.
        4. Calculates cosine similarity against ref_image using Deep CNN (ResNet) or YOLO embeddings.
        5. Annotates the target image and isolates the highest-score cropped person.
        """
        start_time = time.time()
        results = {
            "total_detected": 0,
            "ranked_candidates": [],
            "top_match": None,
            "annotated_image": None,
            "processing_time": 0.0
        }

        if ref_image is None or target_image is None:
            return results

        processed_target = target_image.copy()
        processed_ref = ref_image.copy()

        # Clean both images with CLAHE + edge-preserving smoothing
        if enhance:
            processed_target = self.enhancer.enhance(processed_target)
            processed_ref = self.enhancer.enhance(processed_ref)

        # Detect candidates
        bboxes = []
        if mode == "face":
            bboxes = self.face_detector.detect(processed_target)
        else: # person detection (default)
            det = self.person_detector.detect(processed_target)
            if det.boxes is not None and len(det.boxes) > 0:
                xyxy = det.boxes.xyxy.cpu().numpy()
                for box in xyxy:
                    bboxes.append([int(box[0]), int(box[1]), int(box[2]), int(box[3])])

        results["total_detected"] = len(bboxes)
        if len(bboxes) == 0:
            results["annotated_image"] = processed_target
            results["query_crop"] = processed_ref
            results["processing_time"] = time.time() - start_time
            return results

        # Automatically isolate the reference subject (person/face) from ref_image if not already tightly cropped
        query_crop = processed_ref
        if mode == "face":
            ref_faces = self.face_detector.detect(processed_ref)
            if len(ref_faces) > 0:
                largest_face = max(ref_faces, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
                crop_q = self.crop_bounding_box(processed_ref, largest_face, pad_pct=0.08)
                if crop_q is not None:
                    query_crop = crop_q
        else:
            ref_det = self.person_detector.detect(processed_ref)
            if ref_det.boxes is not None and len(ref_det.boxes) > 0:
                ref_boxes = ref_det.boxes.xyxy.cpu().numpy()
                largest_person = max(ref_boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
                b_int = [int(largest_person[0]), int(largest_person[1]), int(largest_person[2]), int(largest_person[3])]
                crop_q = self.crop_bounding_box(processed_ref, b_int, pad_pct=0.05)
                if crop_q is not None:
                    query_crop = crop_q

        results["query_crop"] = query_crop

        # Crop each candidate
        candidate_crops = []
        valid_bboxes = []
        for bbox in bboxes:
            crop = self.crop_bounding_box(processed_target, bbox)
            if crop is not None and crop.shape[0] > 10 and crop.shape[1] > 10:
                candidate_crops.append(crop)
                valid_bboxes.append(bbox)

        if len(candidate_crops) == 0:
            results["annotated_image"] = processed_target
            results["processing_time"] = time.time() - start_time
            return results

        # Rank candidates via selected matcher
        matcher = self.face_matcher if mode == "face" else self.person_matcher
        ranked_candidates, top_match = matcher.rank_candidates(
            query_crop, candidate_crops, valid_bboxes
        )
        results["ranked_candidates"] = ranked_candidates
        results["top_match"] = top_match

        # Annotate target scene image
        annotated = processed_target.copy()
        for cand in ranked_candidates:
            rank = cand["rank"]
            score = cand["score"]
            pct = cand["percentage"]
            x1, y1, x2, y2 = cand["bbox"]

            if rank == 1:
                # Highlight top match in bold vibrant green
                box_color = (0, 230, 64)
                thickness = 3
                label = f"TOP #1: {pct:.1f}% (sim: {score:.3f})"
            else:
                # Other ranks in cyan/yellow
                box_color = (0, 204, 255) if rank <= 3 else (200, 200, 200)
                thickness = 2
                label = f"#{rank}: {pct:.1f}%"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)

            # Label banner
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            label_y1 = max(0, y1 - lh - 8)
            cv2.rectangle(annotated, (x1, label_y1), (x1 + lw + 8, label_y1 + lh + 6), box_color, -1)
            cv2.putText(
                annotated,
                label,
                (x1 + 4, label_y1 + lh + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0) if rank == 1 else (20, 20, 20),
                2 if rank == 1 else 1,
                cv2.LINE_AA
            )

        results["annotated_image"] = annotated
        results["processing_time"] = time.time() - start_time
        return results
