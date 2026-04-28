"""
Pneumonia Detection - Feature Extraction Pipeline.

Extract handcrafted features from preprocessed X-ray images:
- HOG  : edge and shape cues          (Chương 3)
- LBP  : local texture histogram      (Chương 3)
- GLCM : statistical texture descriptors (Chương 4)

Outputs are stored in ``data/features``:
- X_train.npy, y_train.npy
- X_val.npy, y_val.npy
- X_test.npy, y_test.npy
- features_summary.csv
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
except ImportError as exc:
    raise ImportError(
        "Missing dependency 'scikit-image'. Install it before running feature extraction."
    ) from exc

try:
    from sklearn.utils import shuffle
except ImportError as exc:
    raise ImportError(
        "Missing dependency 'scikit-learn'. Install it before running feature extraction."
    ) from exc


BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_PATH = BASE_DIR / "data" / "processed"
FEATURE_PATH = BASE_DIR / "data" / "features"

IMG_SIZE = (224, 224)
CLASS_TO_LABEL = {"NORMAL": 0, "PNEUMONIA": 1}
IMAGE_EXTENSIONS = ("*.jpeg", "*.jpg", "*.png")

LBP_POINTS = 8
LBP_RADIUS = 1

#  Số mức lượng tử cho GLCM (giảm từ 256 → 64, nhanh hơn ~16x)
GLCM_LEVELS = 64


def create_feature_directory():
    FEATURE_PATH.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Feature folder ready: {FEATURE_PATH}")


def extract_hog(image):
    """Extract Histogram of Oriented Gradients features. (Chương 3)"""
    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return features.astype(np.float32)


def extract_lbp(image):
    """Extract a normalized Local Binary Pattern histogram. (Chương 3)"""
    lbp = local_binary_pattern(
        image,
        P=LBP_POINTS,
        R=LBP_RADIUS,
        method="uniform",
    )
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, LBP_POINTS + 3),
        range=(0, LBP_POINTS + 2),
    )
    hist = hist.astype(np.float32)
    hist /= hist.sum() + 1e-6
    return hist


def extract_glcm(image):
    """Extract Gray Level Co-occurrence Matrix statistics. (Chương 4)
    
    Quantize image từ 256 → 64 levels trước khi tính GLCM.
    Lý do: Ma trận GLCM với levels=256 có kích thước 256x256 → rất chậm.
           Giảm xuống 64 levels tăng tốc ~16x, độ chính xác không đổi đáng kể.
    """
    # Lượng tử hóa: 256 mức → 64 mức (chia 4)
    image_q = (image // 4).astype(np.uint8)

    glcm = graycomatrix(
        image_q,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )

    features = []
    for prop in ("contrast", "energy", "homogeneity", "correlation"):
        values = graycoprops(glcm, prop).flatten()
        features.extend(values)

    return np.asarray(features, dtype=np.float32)


def list_image_files(class_path):
    image_files = []
    for pattern in IMAGE_EXTENSIONS:
        image_files.extend(class_path.glob(pattern))
    return sorted(image_files)


def extract_features_from_image(image_path):
    """Extract a combined feature vector from one grayscale image."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"[WARNING] Cannot read image: {image_path}")
        return None

    # Ảnh từ processed đã là 224x224, resize là safety check
    image = cv2.resize(image, IMG_SIZE).astype(np.uint8)

    feature_vector = np.concatenate(
        [extract_hog(image), extract_lbp(image), extract_glcm(image)]
    )
    return feature_vector.astype(np.float32)


def process_split(split_name):
    """Process one dataset split and return features, labels, and stats."""
    split_path = PROCESSED_PATH / split_name
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split folder: {split_path}")

    print(f"\n[PROCESSING] {split_name.upper()}")

    X = []
    y = []
    stats = {
        "split": split_name,
        "samples": 0,
        "feature_dim": 0,
        "normal_count": 0,
        "pneumonia_count": 0,
        "failed_images": 0,
    }

    for class_name, label in CLASS_TO_LABEL.items():
        class_path = split_path / class_name
        if not class_path.exists():
            print(f"[WARNING] Missing folder: {class_path}")
            continue

        image_files = list_image_files(class_path)
        print(f"  {class_name}: {len(image_files)} images")

        for image_path in tqdm(image_files, desc=f"{split_name}:{class_name}", leave=False):
            feature_vector = extract_features_from_image(image_path)
            if feature_vector is None:
                stats["failed_images"] += 1
                continue

            X.append(feature_vector)
            y.append(label)

            if class_name == "NORMAL":
                stats["normal_count"] += 1
            else:
                stats["pneumonia_count"] += 1

    if not X:
        raise ValueError(f"No valid images were processed for split '{split_name}'.")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    X, y = shuffle(X, y, random_state=42)

    stats["samples"] = int(X.shape[0])
    stats["feature_dim"] = int(X.shape[1])

    print(f"[INFO] {split_name} shape: {X.shape}")
    return X, y, stats


def save_split(X, y, split_name):
    np.save(FEATURE_PATH / f"X_{split_name}.npy", X)
    np.save(FEATURE_PATH / f"y_{split_name}.npy", y)
    print(f"[OK] Saved {split_name}: X{X.shape}, y{y.shape}")


def save_summary(summary_rows):
    df = pd.DataFrame(summary_rows)
    df.to_csv(FEATURE_PATH / "features_summary.csv", index=False)
    print(f"[OK] Saved summary CSV: {FEATURE_PATH / 'features_summary.csv'}")


def main():
    print("=" * 60)
    print("TASK 2 - FEATURE EXTRACTION")
    print("=" * 60)

    create_feature_directory()

    summary_rows = []
    for split in ("train", "val", "test"):
        X, y, stats = process_split(split)
        save_split(X, y, split)
        summary_rows.append(stats)

    save_summary(summary_rows)

    print("\n[COMPLETE] Feature extraction finished")
    print(f"Output saved at: {FEATURE_PATH}")


if __name__ == "__main__":
    main()