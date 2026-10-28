# NOCTIVISION: Low-Light Image Enhancement, Crowd Density Estimation, Person Detection & Tracking

## 1. Abstract
This project presents an integrated pipeline designed for nighttime and low-light surveillance. It combines low-light enhancement, super-resolution, person detection, and object tracking into a cohesive framework.

## 2. Problem Statement
Nighttime surveillance footage suffers from low illumination, high noise, and low resolution, making downstream computer vision tasks like detection and tracking difficult.

## 3. Dataset Analysis
Due to the absence of a provided primary dataset, a fallback subset of external low-light images was used. Quantitative evaluation (mAP, MAE) was skipped as no ground truth bounding boxes or crowd density maps were available for this subset. 
- **Used Dataset**: Sample external open-source low-light images.
- **Tasks Supported**: Qualitative visual evaluation.

## 4. Methodology & Architecture
The system follows a modular architecture:
1. Input Image/Video
2. Low-Light Enhancement (CLAHE)
3. Super-Resolution (EDSR)
4. Person Detection (YOLOv8) & Tracking (ByteTrack)
5. Crowd Density Estimation (Detection fallback)

## 5. Experimental Results (Ablation)
An ablation study was conducted comparing:
1. Baseline (Original -> Detection)
2. Enhanced (Enhanced -> Detection)
3. Enhanced + SR (Enhanced -> SR -> Detection)

Results showed improved visibility in the enhanced variants, though excessive noise amplification in low-light conditions occasionally impacted YOLOv8's confidence unless appropriately filtered.

## 6. Limitations
- Lack of quantitative benchmark dataset limits the rigorousness of evaluation.
- Crowd counting uses a detection fallback, which struggles in highly dense crowds due to occlusion.

## 7. Conclusion
The Nightvision pipeline successfully integrates enhancement, scaling, detection, and tracking into an easily configurable interface (Streamlit). It demonstrates the utility of pre-processing for surveillance applications.
