"""
PNEUMONIA DETECTION - PREPROCESSING PIPELINE
Chỉ thực hiện tiền xử lý ảnh (Chương 2)
1. Read Grayscale
2. Resize to 224x224 (INTER_AREA)
3. CLAHE enhancement (increase contrast)
4. Bilateral Filter (noise reduction)
5. Save processed images
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Configuration
IMAGE_SIZE = 224
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed"
ARCHIVE_PATH = BASE_DIR / "data"

def create_directories():
    """Create output directory structure"""
    for split in ["train", "test", "val"]:
        for category in ["NORMAL", "PNEUMONIA"]:
            output_dir = PROCESSED_DATA_PATH / split / category
            output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Created output directories in {PROCESSED_DATA_PATH}")

def preprocess_single_image(image_path):
    """
    Chỉ thực hiện Tiền xử lý ảnh (Làm sạch và tăng cường)
    """
    try:
        # 1. Đọc ảnh
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Could not read {image_path}")
            return None
            
        # 2. Resize
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        
        # 3. CLAHE (Tăng tương phản)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        # 4. Lọc nhiễu giữ biên (Bilateral Filter)
        smoothed = cv2.bilateralFilter(enhanced, d=5, sigmaColor=75, sigmaSpace=75)
        
        return smoothed
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def save_preprocessed_image(processed_image, output_path, filename):
    """Save processed image as uint8"""
    output_path.mkdir(parents=True, exist_ok=True)
    output_file = output_path / filename
    
    success = cv2.imwrite(str(output_file), processed_image)
    if not success:
        print(f"Failed to save: {output_file}")
    
    return output_file

def process_dataset():
    """Process all images in the dataset"""
    print("=" * 60)
    print("PNEUMONIA DETECTION - PHASE 1: PREPROCESSING")
    print("=" * 60)
    
    create_directories()
    
    total_files = 0
    total_processed = 0
    
    for split in ["train", "test", "val"]:
        split_path = ARCHIVE_PATH / split
        if not split_path.exists():
            print(f" Split {split} not found at {split_path}")
            continue
        
        print(f"\n[PROCESSING] {split.upper()} split:")
        
        for category in ["NORMAL", "PNEUMONIA"]:
            category_path = split_path / category
            if not category_path.exists():
                continue
            
            image_files = list(category_path.glob("*.jpeg")) + list(category_path.glob("*.jpg"))
            
            print(f"\n  [DATA] {category}: Processing {len(image_files)} images...")
            
            for image_path in tqdm(image_files):
                total_files += 1
                
                processed_image = preprocess_single_image(image_path)
                
                if processed_image is not None:
                    output_dir = PROCESSED_DATA_PATH / split / category
                    save_preprocessed_image(processed_image, output_dir, image_path.name)
                    total_processed += 1
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"[OK] Total images: {total_files}")
    print(f"[OK] Successfully processed: {total_processed}")
    print(f"[OK] Output directory: {PROCESSED_DATA_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    process_dataset()