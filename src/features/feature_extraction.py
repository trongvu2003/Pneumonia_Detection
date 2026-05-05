"""
Pneumonia Detection - Feature Extraction Pipeline.

Extract handcrafted features from preprocessed X-ray images:
- HOG  : edge and shape cues
- LBP  : local texture histogram
- GLCM : statistical texture descriptors

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
    raise ImportError("Missing dependency 'scikit-image'.") from exc

try:
    from sklearn.utils import shuffle
    from sklearn.preprocessing import normalize
except ImportError as exc:
    raise ImportError("Missing dependency 'scikit-learn'.") from exc


BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_PATH = BASE_DIR / "data" / "processed"
FEATURE_PATH = BASE_DIR / "data" / "features"

IMG_SIZE = (224, 224)
CLASS_TO_LABEL = {"NORMAL": 0, "PNEUMONIA": 1}
IMAGE_EXTENSIONS = ("*.jpeg", "*.jpg", "*.png")

# LBP: tăng radius+points để capture vùng texture rộng hơn
LBP_POINTS = 24
LBP_RADIUS = 3

GLCM_LEVELS = 64


def create_feature_directory():
    FEATURE_PATH.mkdir(parents=True, exist_ok=True)
    print(f"[OK] Feature folder ready: {FEATURE_PATH}")


# HOG
def extract_hog(image):
    """
    pixels_per_cell (8,8)→(16,16):
      - (8,8) trên 224x224 tạo ra 27x27x2x2x9 ≈ 34K features → quá nhiều
      - (16,16) tạo ra 13x13x2x2x9 ≈ 9K features → vừa đủ, ít noise hơn
    """
    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return features.astype(np.float32)


#  LBP
def extract_lbp(image):
    """
    Tăng P=24, R=3 để capture cấu trúc texture phổi ở vùng rộng hơn.
    Phổi có texture coarse → radius nhỏ (R=1) bỏ qua nhiều thông tin.
    Multi-scale: kết hợp R=1 và R=3 để có cả fine và coarse texture.
    """
    hists = []
    for r, p in [(1, 8), (3, 24)]:  # ← multi-scale LBP
        lbp = local_binary_pattern(image, P=p, R=r, method="uniform")
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=np.arange(0, p + 3),
            range=(0, p + 2),
        )
        hist = hist.astype(np.float32)
        hist /= hist.sum() + 1e-6
        hists.append(hist)
    return np.concatenate(hists)


# GLCM
def extract_glcm(image):
    """
    Thêm distances=[1,3] để capture cả texture gần và xa.
    Phổi bình thường vs viêm phổi có sự khác biệt rõ ở distance lớn hơn.
    """
    image_q = (image // 4).astype(np.uint8)
    glcm = graycomatrix(
        image_q,
        distances=[1, 3],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )
    features = []
    for prop in ("contrast", "energy", "homogeneity", "correlation", "dissimilarity"):
        values = graycoprops(glcm, prop).flatten()
        features.extend(values)
    return np.asarray(features, dtype=np.float32)


# Gabor
def extract_gabor(image):
    """
    Gabor filter capture texture ở nhiều tần số và hướng khác nhau.
    Rất hiệu quả cho ảnh y tế — phổi viêm có texture pattern khác biệt.
    """
    features = []
    # 4 tần số × 4 hướng = 16 filters
    for frequency in [0.1, 0.2, 0.3, 0.4]:
        for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
            kernel = cv2.getGaborKernel(
                ksize=(21, 21),
                sigma=4.0,
                theta=theta,
                lambd=1.0 / frequency,
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_32F,
            )
            filtered = cv2.filter2D(image.astype(np.float32), -1, kernel)
            # Mean + std của response → 2 values per filter
            features.extend([filtered.mean(), filtered.std()])
    return np.asarray(features, dtype=np.float32)  # 16 × 2 = 32 features


# Combine + normalize riêng từng group
def extract_features_from_image(image_path):
    """
    KEY FIX: normalize từng feature group về unit norm riêng
    trước khi concatenate → HOG (9K) không áp đảo GLCM (40) và LBP (36).
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"[WARNING] Cannot read image: {image_path}")
        return None

    image = cv2.resize(image, IMG_SIZE).astype(np.uint8)

    hog_feat = extract_hog(image)
    lbp_feat = extract_lbp(image)
    glcm_feat = extract_glcm(image)
    gabor_feat = extract_gabor(image)

    # Normalize từng group riêng về L2 unit norm ← KEY FIX
    def l2_norm(x):
        norm = np.linalg.norm(x)
        return x / (norm + 1e-8)

    feature_vector = np.concatenate(
        [
            l2_norm(hog_feat),
            l2_norm(lbp_feat),
            l2_norm(glcm_feat),
            l2_norm(gabor_feat),
        ]
    )
    return feature_vector.astype(np.float32)


# Process split
def process_split(split_name):
    split_path = PROCESSED_PATH / split_name
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split folder: {split_path}")

    print(f"\n[PROCESSING] {split_name.upper()}")

    X, y = [], []
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

        image_files = []
        for pattern in IMAGE_EXTENSIONS:
            image_files.extend(class_path.glob(pattern))
        image_files = sorted(image_files)
        print(f"  {class_name}: {len(image_files)} images")

        for image_path in tqdm(
            image_files, desc=f"{split_name}:{class_name}", leave=False
        ):
            fv = extract_features_from_image(image_path)
            if fv is None:
                stats["failed_images"] += 1
                continue
            X.append(fv)
            y.append(label)
            if class_name == "NORMAL":
                stats["normal_count"] += 1
            else:
                stats["pneumonia_count"] += 1

    if not X:
        raise ValueError(f"No valid images processed for split '{split_name}'.")

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    X, y = shuffle(X, y, random_state=42)

    stats["samples"] = int(X.shape[0])
    stats["feature_dim"] = int(X.shape[1])

    print(f"[INFO] {split_name} shape: {X.shape}")
    print(
        f"[INFO] Feature breakdown — HOG: ~{len(extract_hog(np.zeros(IMG_SIZE, np.uint8)))} | "
        f"LBP: ~{len(extract_lbp(np.zeros(IMG_SIZE, np.uint8)))} | "
        f"GLCM: ~{len(extract_glcm(np.zeros(IMG_SIZE, np.uint8)))} | "
        f"Gabor: 32"
    )
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
    print("TASK 2 - FEATURE EXTRACTION (IMPROVED)")
    print("HOG(16x16) + Multi-scale LBP + GLCM(d=1,3) + Gabor")
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
