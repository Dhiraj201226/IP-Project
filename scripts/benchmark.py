import time
import pandas as pd
import matplotlib.pyplot as plt
from pipeline.image_pipeline import ImagePipeline
import os
import cv2

def main():
    print("Running Ablation Benchmark...")
    image_path = "data/test/images/sample_low_light.jpg"
    
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found.")
        return

    # A: Original -> Detection
    pipe_a = ImagePipeline()
    pipe_a.enhancer = None
    pipe_a.sr = None
    res_a = pipe_a.run(image_path)
    cv2.imwrite("outputs/visualizations/baseline_det.jpg", res_a['final_image'])

    # B: Enhanced -> Detection
    pipe_b = ImagePipeline()
    pipe_b.sr = None
    res_b = pipe_b.run(image_path)
    cv2.imwrite("outputs/visualizations/enhanced_det.jpg", res_b['final_image'])

    # C: Enhanced -> SR -> Detection
    pipe_c = ImagePipeline()
    res_c = pipe_c.run(image_path)
    cv2.imwrite("outputs/visualizations/enhanced_sr_det.jpg", res_c['final_image'])

    data = {
        "Experiment": ["Baseline", "Enhanced", "Enhanced+SR"],
        "Detected People": [res_a['num_detected'], res_b['num_detected'], res_c['num_detected']],
        "Processing Time (s)": [res_a['processing_time'], res_b['processing_time'], res_c['processing_time']]
    }
    
    df = pd.DataFrame(data)
    print("\n=== Benchmark Results ===")
    print(df.to_string(index=False))
    
    # Plot
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(df['Experiment'], df['Detected People'], color='skyblue')
    plt.title('Detected People')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    plt.bar(df['Experiment'], df['Processing Time (s)'], color='salmon')
    plt.title('Processing Time')
    plt.ylabel('Seconds')
    
    plt.tight_layout()
    plt.savefig('outputs/visualizations/benchmark_plot.png')
    print("Plot saved to outputs/visualizations/benchmark_plot.png")

if __name__ == "__main__":
    main()
