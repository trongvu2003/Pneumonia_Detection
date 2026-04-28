"""
Pneumonia Detection - Classical ML Training Pipeline
Train SVM và Random Forest trên features HOG + LBP + GLCM
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import joblib

# Paths─────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parents[2]
FEATURE_PATH = BASE_DIR / "data" / "features"
MODELS_PATH  = BASE_DIR / "models"

CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


# Load features
def load_features():
    """Load các file .npy đã trích xuất từ feature_extraction.py"""
    print("[LOADING] Loading extracted features...")

    required_files = [
        "X_train.npy", "y_train.npy",
        "X_val.npy",   "y_val.npy",
        "X_test.npy",  "y_test.npy",
    ]
    for f in required_files:
        if not (FEATURE_PATH / f).exists():
            raise FileNotFoundError(
                f"[ERROR] Missing: {FEATURE_PATH / f}\n"
                "Please run feature_extraction.py first!"
            )

    X_train = np.load(FEATURE_PATH / "X_train.npy")
    y_train = np.load(FEATURE_PATH / "y_train.npy")
    X_val   = np.load(FEATURE_PATH / "X_val.npy")
    y_val   = np.load(FEATURE_PATH / "y_val.npy")
    X_test  = np.load(FEATURE_PATH / "X_test.npy")
    y_test  = np.load(FEATURE_PATH / "y_test.npy")

    print(f"[OK] Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    print(f"[OK] Feature dimension: {X_train.shape[1]}")
    return X_train, y_train, X_val, y_val, X_test, y_test


# Preprocessing
def preprocess_features(X_train, X_val, X_test, n_components=200):
    """
    Chuẩn hóa + PCA để giảm chiều.
    HOG tạo ra ~34,596 features → quá lớn cho SVM, cần giảm chiều.
    """
    print(f"\n[PREPROCESSING] Scaling + PCA (n_components={n_components})...")

    # StandardScaler: đưa features về mean=0, std=1
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    # PCA: giảm từ ~34K → 200 dimensions
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca   = pca.transform(X_val_scaled)
    X_test_pca  = pca.transform(X_test_scaled)

    explained = pca.explained_variance_ratio_.sum() * 100
    print(f"[OK] PCA giữ {explained:.1f}% variance với {n_components} components")
    print(f"[OK] Shape sau PCA: {X_train_pca.shape}")

    return X_train_pca, X_val_pca, X_test_pca, scaler, pca


# Evaluation 
def evaluate_model(model, X, y, dataset_name, model_name):
    """Tính và in đầy đủ metrics"""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    acc  = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec  = recall_score(y, y_pred)
    f1   = f1_score(y, y_pred)
    auc  = roc_auc_score(y, y_prob)
    cm   = confusion_matrix(y, y_pred)

    print(f"\n{'='*60}")
    print(f"{model_name} — {dataset_name.upper()} SET")
    print(f"{'='*60}")
    print(f"Accuracy:  {acc*100:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"\n{classification_report(y, y_pred, target_names=CLASS_NAMES)}")

    return {"model": model_name, "dataset": dataset_name,
            "accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "auc": auc, "cm": cm}


# Plot 
def plot_confusion_matrices(results, save_dir):
    """Vẽ confusion matrix cho tất cả model × dataset"""
    test_results = [r for r in results if r["dataset"] == "Test"]
    n = len(test_results)

    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, r in zip(axes, test_results):
        sns.heatmap(r["cm"], annot=True, fmt='d', cmap='Blues',
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
        ax.set_title(f"{r['model']} (Test)\nAcc: {r['accuracy']*100:.2f}%")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    plt.tight_layout()
    path = save_dir / "classical_ml_confusion_matrices.png"
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"[OK] Confusion matrix saved: {path}")


def plot_comparison(results, save_dir):
    """So sánh các model theo từng metric"""
    test_results = [r for r in results if r["dataset"] == "Test"]
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    labels  = [r["model"] for r in test_results]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, r in enumerate(test_results):
        values = [r[m] for m in metrics]
        ax.bar(x + i * width, values, width, label=r["model"])

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "AUC"])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Classical ML — Model Comparison (Test Set)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = save_dir / "classical_ml_comparison.png"
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"[OK] Comparison plot saved: {path}")


# ── Main ──────────────────────────────────────────────────────────
def main():
    print("\n" + "="*70)
    print("PNEUMONIA DETECTION - CLASSICAL ML TRAINING")
    print("SVM + Random Forest trên features HOG + LBP + GLCM")
    print("="*70)

    MODELS_PATH.mkdir(parents=True, exist_ok=True)

    # ── Load ────────────────────────────────────────────────────
    try:
        X_train, y_train, X_val, y_val, X_test, y_test = load_features()
    except FileNotFoundError as e:
        print(e)
        return False

    # ── Preprocess ──────────────────────────────────────────────
    X_train_pca, X_val_pca, X_test_pca, scaler, pca = preprocess_features(
        X_train, X_val, X_test, n_components=200
    )

    # Gộp train + val để train cuối cùng (dùng val để chọn model)
    X_trainval = np.concatenate([X_train_pca, X_val_pca])
    y_trainval = np.concatenate([y_train, y_val])

    all_results = []

    # ── Model 1: SVM ─────────────────────────────────────────────
    print("\n[STAGE 1] Training SVM...")
    print("-"*70)
    svm = SVC(
        kernel='rbf',
        C=10,
        gamma='scale',
        probability=True,   # cần để tính AUC
        random_state=42
    )
    svm.fit(X_trainval, y_trainval)
    print("[OK] SVM training complete!")

    all_results.append(evaluate_model(svm, X_val_pca,  y_val,  "Val",  "SVM"))
    all_results.append(evaluate_model(svm, X_test_pca, y_test, "Test", "SVM"))

    joblib.dump(svm, MODELS_PATH / "svm_model.pkl")
    print("[OK] SVM model saved")

    # ── Model 2: Random Forest ────────────────────────────────────
    print("\n[STAGE 2] Training Random Forest...")
    print("-"*70)
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_trainval, y_trainval)
    print("[OK] Random Forest training complete!")

    all_results.append(evaluate_model(rf, X_val_pca,  y_val,  "Val",  "Random Forest"))
    all_results.append(evaluate_model(rf, X_test_pca, y_test, "Test", "Random Forest"))

    joblib.dump(rf, MODELS_PATH / "rf_model.pkl")
    print("[OK] Random Forest model saved")

    # ── Save scaler + PCA ─────────────────────────────────────────
    joblib.dump(scaler, MODELS_PATH / "scaler.pkl")
    joblib.dump(pca,    MODELS_PATH / "pca.pkl")
    print("[OK] Scaler + PCA saved")

    # ── Plots ─────────────────────────────────────────────────────
    print("\n[STAGE 3] Saving plots...")
    print("-"*70)
    plot_confusion_matrices(all_results, MODELS_PATH)
    plot_comparison(all_results, MODELS_PATH)

    # ── Summary CSV ───────────────────────────────────────────────
    summary = pd.DataFrame([
        {k: v for k, v in r.items() if k != "cm"}
        for r in all_results
    ])
    summary_path = MODELS_PATH / "classical_ml_results.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[OK] Results saved: {summary_path}")

    print("\n" + "="*70)
    print("CLASSICAL ML TRAINING COMPLETE!")
    print("="*70)
    print("\nCác file đã lưu:")
    print(f"  {MODELS_PATH}/svm_model.pkl")
    print(f"  {MODELS_PATH}/rf_model.pkl")
    print(f"  {MODELS_PATH}/classical_ml_results.csv")
    print(f"  {MODELS_PATH}/classical_ml_confusion_matrices.png")
    print(f"  {MODELS_PATH}/classical_ml_comparison.png")
    print("\nNext step: So sánh kết quả với GoogLeNet trong báo cáo!")
    print("="*70 + "\n")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)