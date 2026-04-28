"""
Pneumonia Detection - Flask Backend API
Nhận ảnh X-ray từ frontend, trả về kết quả dự đoán SVM / Random Forest
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from pathlib import Path
import numpy as np
import joblib
import cv2
import base64

try:
    from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern
except ImportError:
    raise ImportError("pip install scikit-image")

app = Flask(__name__)
CORS(app)  # Cho phép React gọi API

# Paths 
BASE_DIR    = Path(__file__).resolve().parents[2]
MODELS_PATH = BASE_DIR / "models"

IMG_SIZE    = (224, 224)
LBP_POINTS  = 8
LBP_RADIUS  = 1
GLCM_LEVELS = 64

# Load models khi khởi động server 
print("[LOADING] Loading models...")
try:
    svm    = joblib.load(MODELS_PATH / "svm_model.pkl")
    rf     = joblib.load(MODELS_PATH / "rf_model.pkl")
    scaler = joblib.load(MODELS_PATH / "scaler.pkl")
    pca    = joblib.load(MODELS_PATH / "pca.pkl")
    print("[OK] All models loaded!")
except FileNotFoundError as e:
    print(f"[ERROR] {e}")
    print("Please run train_classical_ml.py first!")
    svm = rf = scaler = pca = None


#Feature extraction (giống feature_extraction.py) 
def extract_hog(image):
    features = hog(
        image, orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return features.astype(np.float32)


def extract_lbp(image):
    lbp = local_binary_pattern(image, P=LBP_POINTS, R=LBP_RADIUS, method="uniform")
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, LBP_POINTS + 3),
        range=(0, LBP_POINTS + 2),
    )
    hist = hist.astype(np.float32)
    hist /= hist.sum() + 1e-6
    return hist


def extract_glcm(image):
    image_q = (image // 4).astype(np.uint8)
    glcm = graycomatrix(
        image_q, distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=GLCM_LEVELS, symmetric=True, normed=True,
    )
    features = []
    for prop in ("contrast", "energy", "homogeneity", "correlation"):
        features.extend(graycoprops(glcm, prop).flatten())
    return np.asarray(features, dtype=np.float32)


def preprocess_image(image_bytes):
    """Đọc ảnh từ bytes, tiền xử lý và trích xuất features"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    # Preprocess (giống preprocess.py)
    image = cv2.resize(image, IMG_SIZE, interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    image = cv2.bilateralFilter(image, d=5, sigmaColor=75, sigmaSpace=75)

    # Feature extraction
    features = np.concatenate([
        extract_hog(image),
        extract_lbp(image),
        extract_glcm(image),
    ])
    return features.astype(np.float32)


# API Routes 
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": svm is not None
    })


@app.route("/predict", methods=["POST"])
def predict():
    if svm is None:
        return jsonify({"error": "Models not loaded"}), 500

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        image_bytes = file.read()

        # Trích xuất features
        features = preprocess_image(image_bytes)
        if features is None:
            return jsonify({"error": "Cannot read image"}), 400

        # Scale + PCA
        features_scaled = scaler.transform([features])
        features_pca    = pca.transform(features_scaled)

        # Dự đoán từ cả 2 model
        CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

        svm_pred    = int(svm.predict(features_pca)[0])
        svm_prob    = svm.predict_proba(features_pca)[0].tolist()

        rf_pred     = int(rf.predict(features_pca)[0])
        rf_prob     = rf.predict_proba(features_pca)[0].tolist()

        # Trả về base64 ảnh để hiển thị trên UI
        img_base64 = base64.b64encode(image_bytes).decode("utf-8")

        return jsonify({
            "image_base64": img_base64,
            "svm": {
                "prediction": CLASS_NAMES[svm_pred],
                "confidence": round(max(svm_prob) * 100, 2),
                "prob_normal": round(svm_prob[0] * 100, 2),
                "prob_pneumonia": round(svm_prob[1] * 100, 2),
            },
            "random_forest": {
                "prediction": CLASS_NAMES[rf_pred],
                "confidence": round(max(rf_prob) * 100, 2),
                "prob_normal": round(rf_prob[0] * 100, 2),
                "prob_pneumonia": round(rf_prob[1] * 100, 2),
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)