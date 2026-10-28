# Nightvision Model Selection

## Low-Light Enhancement
- **Chosen Method**: CLAHE (Contrast Limited Adaptive Histogram Equalization) via OpenCV (L-channel of LAB colorspace).
- **Why**: Zero-DCE and EnlightenGAN weights were not readily available as a standard Hugging Face pipeline, and OpenCV CLAHE provides a highly robust, zero-dependency baseline for image enhancement without any large errors.
- **Input**: BGR Image
- **Output**: Enhanced BGR Image

## Super-Resolution
- **Chosen Method**: EDSR (Enhanced Deep Residual Networks for Single Image Super-Resolution) via OpenCV DNN Superres module.
- **Source**: OpenCV DNN / Saafke EDSR_Tensorflow
- **Why**: It is pretrained, lightweight to run via OpenCV, and does not require complex PyTorch setups or heavy dependency trees.
- **Input**: BGR Image
- **Output**: 4x Upscaled BGR Image

## Person Detection & Tracking
- **Chosen Model**: YOLOv8n (Ultralytics)
- **Source**: Ultralytics / PyPI
- **Why**: YOLOv8 natively supports both high-accuracy object detection and tracking (`ByteTrack` / `BoT-SORT`). It's highly optimized, robust, and explicitly supported for this project structure.
- **Input**: BGR Image
- **Output**: Detected bounding boxes, confidence scores, and persistent tracking IDs.

## Crowd Counting
- **Chosen Method**: Fallback Detection Counting
- **Why**: Standard crowd counting datasets/models (like CSRNet) require specialized density map ground-truth which is not present in our fallback datasets. As per instructions, we use detection count as a fallback and classify density into Low/Medium/High thresholds.
