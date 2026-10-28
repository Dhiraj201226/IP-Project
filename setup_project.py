import os
import urllib.request
import json

base_dir = r"c:\Users\dhira\OneDrive\Desktop\CEP SEM4\1st attempt\ARA-1-Financial-Agent\IP_Project"

directories = [
    "data/test/images",
    "data/test/videos",
    "models",
    "pipeline",
    "evaluation",
    "utils",
    "scripts",
    "docs",
    "outputs/original",
    "outputs/enhanced",
    "outputs/super_resolved",
    "outputs/detected",
    "outputs/videos",
    "outputs/visualizations",
    "outputs/metrics"
]

for d in directories:
    os.makedirs(os.path.join(base_dir, d), exist_ok=True)

# Download a sample low-light image
img_url = "https://raw.githubusercontent.com/csjcai/Zero-DCE/master/data/test_data/LIME/1.jpg"
img_path = os.path.join(base_dir, "data", "test", "images", "sample_low_light.jpg")

try:
    urllib.request.urlretrieve(img_url, img_path)
    print(f"Downloaded sample image to {img_path}")
except Exception as e:
    print(f"Failed to download image: {e}")

# Create a dataset summary json
summary = {
    "total_images": 1 if os.path.exists(img_path) else 0,
    "total_videos": 0,
    "total_annotations": 0,
    "image_dimensions": "Unknown",
    "available_labels": [],
    "available_splits": ["test"],
    "supported_evaluation_tasks": ["Qualitative Enhancement", "Qualitative Detection"]
}

with open(os.path.join(base_dir, "dataset_summary.json"), "w") as f:
    json.dump(summary, f, indent=4)
    
print("Created dataset_summary.json")

