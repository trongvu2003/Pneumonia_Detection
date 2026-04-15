"""
Pneumonia Detection - Data Preprocessing Pipeline
Process X-ray images following the steps:
1. Resize to 224x224
2. CLAHE enhancement (increase contrast)
3. Apply filters (Gaussian Blur, Median Filter, Canny Edge, Sobel)
4. Normalize
5. Save processed images
"""

import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm

# Configuration
IMAGE_SIZE = 224
PROCESSED_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "processed"
ARCHIVE_PATH = Path(__file__).parent.parent / "archive" / "chest_xray"

def create_directories():
    """Create output directory structure"""
    for split in ["train", "test", "val"]:
        for category in ["NORMAL", "PNEUMONIA"]:
            output_dir = PROCESSED_DATA_PATH / split / category
            output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Created output directories in {PROCESSED_DATA_PATH}")

def resize_image(image, size=IMAGE_SIZE):
    """
    Step 1: Resize image to target size
    """
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    return resized

def clahe_enhancement(image):
    """
    Step 2: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    Tăng tương phản ảnh X-ray, làm rõ vùng phổi
    """
    if len(image.shape) == 3:
        # Convert to grayscale if needed
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    return enhanced

def apply_filters(image):
    """
    Step 3: Apply various filters for feature extraction
    - Gaussian Blur: Reduce noise
    - Median Filter: Noise filtering
    - Canny Edge: Edge detection
    - Sobel: Gradient-based edge detection
    """
    # Gaussian Blur - reduce noise
    gaussian = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Median Filter - additional noise filtering
    median = cv2.medianBlur(image, 5)
    
    # Canny Edge detection
    canny = cv2.Canny(image, 50, 150)
    
    # Sobel - gradient
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobel_x, sobel_y)
    
    return {
        'gaussian': gaussian,
        'median': median,
        'canny': canny,
        'sobel': sobel
    }

def normalize_image(image):
    """
    Step 4: Normalize image values to [0, 1] range
    """
    # Convert to float and normalize
    normalized = image.astype(np.float32) / 255.0
    return normalized

def preprocess_single_image(image_path):
    """
    Complete preprocessing pipeline for a single image
    """
    try:
        # Read image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Warning: Could not read {image_path}")
            return None
        
        # Step 1: Resize
        image = resize_image(image)
        
        # Step 2: CLAHE enhancement
        image = clahe_enhancement(image)
        
        # Step 3: Apply filters (we'll use the enhanced image as base)
        filters = apply_filters(image)
        
        # Step 4: Normalize
        normalized = normalize_image(image)
        
        return {
            'original': image,
            'enhanced': normalized,
            'filters': filters
        }
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def save_preprocessed_image(processed_data, output_path, filename):
    """
    Step 5: Save processed images
    Save the enhanced and normalized image
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the enhanced normalized image (multiply by 255 to get uint8 format)
    image_to_save = (processed_data['enhanced'] * 255).astype(np.uint8)
    output_file = output_path / filename
    
    # Debug: print save operation
    success = cv2.imwrite(str(output_file), image_to_save)
    if not success:
        print(f"Failed to save: {output_file}")
    
    return output_file

def process_dataset():
    """
    Process all images in the dataset
    """
    print("=" * 60)
    print("PNEUMONIA DETECTION - DATA PREPROCESSING")
    print("=" * 60)
    
    # Create output directories
    create_directories()
    
    # Process each split (train, test, val)
    total_files = 0
    total_processed = 0
    
    for split in ["train", "test", "val"]:
        split_path = ARCHIVE_PATH / split
        if not split_path.exists():
            print(f"⚠ Split {split} not found at {split_path}")
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
                
                # Preprocess image
                processed = preprocess_single_image(image_path)
                
                if processed is not None:
                    # Save processed image
                    output_dir = PROCESSED_DATA_PATH / split / category
                    output_file = save_preprocessed_image(
                        processed, 
                        output_dir, 
                        image_path.name
                    )
                    total_processed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"[OK] Total images: {total_files}")
    print(f"[OK] Successfully processed: {total_processed}")
    print(f"[OK] Output directory: {PROCESSED_DATA_PATH}")
    print("=" * 60)

if __name__ == "__main__":
    process_dataset()
