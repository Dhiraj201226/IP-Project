import argparse
import cv2
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.image_pipeline import ImagePipeline

def main():
    parser = argparse.ArgumentParser(description="Run Nightvision Image Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input image")
    args = parser.parse_args()

    pipeline = ImagePipeline()
    results = pipeline.run(args.input)
    
    filename = os.path.basename(args.input)
    
    if 'enhanced' in results:
        cv2.imwrite(os.path.join("outputs", "enhanced", filename), results['enhanced'])
    if 'super_resolved' in results:
        cv2.imwrite(os.path.join("outputs", "super_resolved", filename), results['super_resolved'])
    if 'detected' in results:
        cv2.imwrite(os.path.join("outputs", "detected", filename), results['detected'])
        
    cv2.imwrite(os.path.join("outputs", "original", "final_" + filename), results['final_image'])
    
    print("=== Results ===")
    print(f"Processing Time: {results['processing_time']:.3f}s")
    if 'num_detected' in results:
        print(f"People Detected: {results['num_detected']}")
    if 'crowd_count' in results:
        print(f"Estimated Crowd: {results['crowd_count']}")
        print(f"Density: {results['crowd_density']}")
        
    print(f"Output saved to outputs/original/final_{filename}")

if __name__ == "__main__":
    main()
