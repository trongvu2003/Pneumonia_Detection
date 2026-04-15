# Pneumonia Detection - Data Preprocessing

## Overview
This project implements a complete data preprocessing pipeline for chest X-ray images to detect pneumonia using machine learning.

## Preprocessing Steps

### 1. Image Resizing (224×224)
- All images are resized to a consistent 224×224 pixel format
- Maintains aspect ratio using linear interpolation
- Ensures uniform input dimensions for ML models

### 2. CLAHE Enhancement
- **Contrast Limited Adaptive Histogram Equalization**
- Increases contrast in X-ray images
- Clarifies lung regions for better feature detection
- Uses 8×8 tile grid with clipLimit=2.0

### 3. Image Filtering
- **Gaussian Blur**: Reduces noise (5×5 kernel)
- **Median Filter**: Additional noise filtering (5×5 kernel)
- **Canny Edge Detection**: Edge detection (thresholds: 50-150)
- **Sobel Gradient**: Gradient-based edge detection

### 4. Normalization
- Pixel values normalized to [0, 1] range
- Final images saved as uint8 format (0-255 range)
- Ready for ML model input

### 5. Dataset Organization
Processed images saved in structured directory format:
```
data/processed/
├── train/
│   ├── NORMAL/      (1,341 images)
│   └── PNEUMONIA/   (3,875 images)
├── test/
│   ├── NORMAL/      (234 images)
│   └── PNEUMONIA/   (390 images)
└── val/
    ├── NORMAL/      (24 images)
    └── PNEUMONIA/   (23 images)
```

## Dataset Statistics
- **Total Images**: 5,887
- **Image Format**: JPEG
- **Color Space**: Grayscale
- **Dimensions**: 224×224 pixels
- **Value Range**: 0-255 (uint8)

## Usage

### Run Preprocessing
```bash
cd src/preprocessing
python preprocess.py
```

### Verify Dataset
```bash
cd src/preprocessing
python verify_dataset.py
```

## Dependencies
- OpenCV (cv2)
- NumPy
- tqdm
- pathlib

## Output
- Enhanced, filtered, and normalized X-ray images
- Consistent format suitable for CNN-based pneumonia detection
- Maintained class balance and data splits