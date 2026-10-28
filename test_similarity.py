import os
import cv2
import numpy as np
from pipeline.similarity_pipeline import SimilarityPipeline

def test_pipeline():
    image_path = os.path.join("data", "test", "images", "image.png")
    if not os.path.exists(image_path):
        image_path = os.path.join("data", "test", "images", "sample_low_light.jpg")

    print(f"Loading test image from: {image_path}")
    target_img = cv2.imread(image_path)
    assert target_img is not None, f"Failed to load image at {image_path}"

    h, w = target_img.shape[:2]
    print(f"Target Image Size: {w}x{h}")

    # Use a subregion of the target image as a reference query
    ref_img = target_img[int(h*0.1):int(h*0.6), int(w*0.1):int(w*0.5)].copy()
    print(f"Reference Image Size: {ref_img.shape[1]}x{ref_img.shape[0]}")

    pipeline = SimilarityPipeline(model_name="yolov8n.pt")

    # 1. Test Person Mode
    print("\n--- Testing Person Detection & Cosine Similarity Ranking ---")
    results = pipeline.run(ref_img, target_img, mode="person", enhance=True)

    print(f"Total Detected: {results['total_detected']}")
    print(f"Processing Time: {results['processing_time']:.3f}s")

    ranked = results["ranked_candidates"]
    top_match = results["top_match"]

    if len(ranked) > 0:
        print(f"Ranked candidates count: {len(ranked)}")
        for item in ranked:
            print(f"  Rank #{item['rank']}: Score={item['score']:.4f} ({item['percentage']:.1f}%), BBox={item['bbox']}")

        assert top_match is not None, "Top match should not be None when candidates exist"
        assert top_match["rank"] == 1, "Top match must have rank 1"
        assert top_match["score"] == ranked[0]["score"], "Top match score must match first ranked item"
        assert top_match["crop"] is not None and top_match["crop"].size > 0, "Top crop must be valid image array"
        print(f"\nTop Match Cropped Dimensions: {top_match['crop'].shape}")
        
        # Verify scores are sorted descending
        scores = [item["score"] for item in ranked]
        assert scores == sorted(scores, reverse=True), "Scores are not sorted in descending order!"
        print("Verification PASSED: Scores are sorted in descending order.")
    else:
        print("No person candidates found with confidence threshold.")

    # 2. Test Face Mode
    print("\n--- Testing Face Detection & Cosine Similarity Ranking ---")
    face_results = pipeline.run(ref_img, target_img, mode="face", enhance=True)
    print(f"Total Faces Detected: {face_results['total_detected']}")
    if face_results["total_detected"] > 0:
        print(f"Top Face Score: {face_results['top_match']['score']:.4f}")

    print("\nAll automated tests completed successfully!")

if __name__ == "__main__":
    test_pipeline()
