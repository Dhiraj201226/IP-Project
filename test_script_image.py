import os
import sys
from pipeline.image_pipeline import ImagePipeline

def main():
    pipeline = ImagePipeline()
    image_path = os.path.join("data", "test", "images", "image.png")
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return
        
    print(f"Running pipeline on {image_path}...")
    results = pipeline.run(image_path)
    print(f"Processing time: {results.get('processing_time', 'N/A'):.2f} seconds")
    print(f"People Detected: {results.get('num_detected', 'N/A')}")
    print(f"Estimated Crowd: {results.get('crowd_count', 'N/A')}")
    print(f"Crowd Density: {results.get('crowd_density', 'N/A')}")

if __name__ == '__main__':
    main()
