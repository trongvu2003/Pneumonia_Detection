"""
Pneumonia Detection - Flask Backend API
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
CORS(app)

BASE_DIR = Path(__file__).resolve().parents[2]
MODELS_PATH = BASE_DIR / "models"

IMG_SIZE = (224, 224)
GLCM_LEVELS = 64

print("[LOADING] Loading models...")
try:
    svm = joblib.load(MODELS_PATH / "svm_model.pkl")
    rf = joblib.load(MODELS_PATH / "rf_model.pkl")
    scaler = joblib.load(MODELS_PATH / "scaler.pkl")
    pca = joblib.load(MODELS_PATH / "pca.pkl")
    print("[OK] All models loaded!")
except FileNotFoundError as e:
    print(f"[ERROR] {e}")
    svm = rf = scaler = pca = None


# Feature extraction (sync với feature_extraction.py) 
def extract_hog(image):
    features = hog(
        image,
        orientations=9,
        pixels_per_cell=(16, 16), 
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    return features.astype(np.float32)


def extract_lbp(image):
    # Multi-scale: (R=1,P=8) + (R=3,P=24)
    hists = []
    for r, p in [(1, 8), (3, 24)]:
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


def extract_glcm(image):
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
        features.extend(graycoprops(glcm, prop).flatten())
    return np.asarray(features, dtype=np.float32)


def extract_gabor(image):
    features = []
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
            features.extend([filtered.mean(), filtered.std()])
    return np.asarray(features, dtype=np.float32)


def l2_norm(x):
    return x / (np.linalg.norm(x) + 1e-8)

# Preprocess ảnh upload
def preprocess_image(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    image = cv2.resize(image, IMG_SIZE).astype(np.uint8)
    hog_feat = extract_hog(image)
    lbp_feat = extract_lbp(image)
    glcm_feat = extract_glcm(image)
    gabor_feat = extract_gabor(image)

    features = np.concatenate(
        [
            l2_norm(hog_feat),
            l2_norm(lbp_feat),
            l2_norm(glcm_feat),
            l2_norm(gabor_feat),
        ]
    )
    return features.astype(np.float32)

#Routes 
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "models_loaded": svm is not None})


@app.route("/predict", methods=["POST"])
def predict():
    if svm is None:
        return jsonify({"error": "Models not loaded"}), 500

    file = request.files.get("file") or request.files.get("image")
    if file is None:
        return (
            jsonify(
                {
                    "error": f"No file uploaded. Received keys: {list(request.files.keys())}"
                }
            ),
            400,
        )

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        image_bytes = file.read()
        print(f"[DEBUG] Image bytes received: {len(image_bytes)} bytes")

        features = preprocess_image(image_bytes)
        if features is None:
            return (
                jsonify({"error": "Cannot decode image — file may be corrupted"}),
                400,
            )

        features_scaled = scaler.transform([features])
        features_pca = pca.transform(features_scaled)

        CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

        svm_pred = int(svm.predict(features_pca)[0])
        svm_prob = svm.predict_proba(features_pca)[0].tolist()

        rf_pred = int(rf.predict(features_pca)[0])
        rf_prob = rf.predict_proba(features_pca)[0].tolist()

        img_base64 = base64.b64encode(image_bytes).decode("utf-8")

        return jsonify(
            {
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
                },
            }
        )

    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
