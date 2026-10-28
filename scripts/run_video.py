import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.video_pipeline import VideoPipeline

def main():
    parser = argparse.ArgumentParser(description="Run Nightvision Video Pipeline")
    parser.add_argument("--input", type=str, required=True, help="Path to input video")
    args = parser.parse_args()

    pipeline = VideoPipeline()
    
    filename = os.path.basename(args.input)
    output_path = os.path.join("outputs", "videos", "final_" + filename)
    
    print(f"Processing video {args.input}...")
    results = pipeline.run(args.input, output_path)
    
    print("\n=== Results ===")
    print(f"Frames Processed: {results['frames_processed']}")
    print(f"Average FPS: {results['average_fps']:.2f}")
    print(f"Total Processing Time: {results['total_processing_time']:.2f}s")
    print(f"Unique People Tracked: {results['unique_tracks']}")
    print(f"Output saved to {output_path}")

if __name__ == "__main__":
    main()
