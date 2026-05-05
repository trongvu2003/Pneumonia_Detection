"""
Pneumonia Detection - Classical ML Training Pipeline
SVM + Random Forest
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
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from imblearn.over_sampling import SMOTE
import joblib

BASE_DIR = Path(__file__).resolve().parents[2]
FEATURE_PATH = BASE_DIR / "data" / "features"
MODELS_PATH = BASE_DIR / "models"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]


# Load features
def load_features():
    print("[LOADING] Loading extracted features...")
    for f in [
        "X_train.npy",
        "y_train.npy",
        "X_val.npy",
        "y_val.npy",
        "X_test.npy",
        "y_test.npy",
    ]:
        if not (FEATURE_PATH / f).exists():
            raise FileNotFoundError(
                f"[ERROR] Missing: {FEATURE_PATH / f}\nRun feature_extraction.py first!"
            )

    X_train = np.load(FEATURE_PATH / "X_train.npy")
    y_train = np.load(FEATURE_PATH / "y_train.npy")
    X_val = np.load(FEATURE_PATH / "X_val.npy")
    y_val = np.load(FEATURE_PATH / "y_val.npy")
    X_test = np.load(FEATURE_PATH / "X_test.npy")
    y_test = np.load(FEATURE_PATH / "y_test.npy")

    unique, counts = np.unique(y_train, return_counts=True)
    print(
        f"[OK] Train distribution: { {CLASS_NAMES[int(k)]: v for k, v in zip(unique, counts)} }"
    )
    print(f"[OK] Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


# Preprocessing
def preprocess_features(X_train, X_val, X_test):
    print("\n[PREPROCESSING] Scaling + Auto-PCA (95% variance)...")
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    pca_full = PCA(random_state=42).fit(X_train_s)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_opt = min(int(np.searchsorted(cumvar, 0.95)) + 1, 300)
    print(f"[OK] Optimal PCA components = {n_opt}")

    pca = PCA(n_components=n_opt, random_state=42)
    X_train_pca = pca.fit_transform(X_train_s)
    X_val_pca = pca.transform(X_val_s)
    X_test_pca = pca.transform(X_test_s)

    print(f"[OK] Variance retained: {pca.explained_variance_ratio_.sum()*100:.1f}%")
    print(f"[OK] Shape sau PCA: {X_train_pca.shape}")
    return X_train_pca, X_val_pca, X_test_pca, scaler, pca


# SMOTE
def apply_smote(X, y):
    print("\n[SMOTE] Balancing classes...")
    before = {CLASS_NAMES[int(k)]: v for k, v in zip(*np.unique(y, return_counts=True))}
    print(f"[SMOTE] Before: {before}")
    X_res, y_res = SMOTE(random_state=42, k_neighbors=5).fit_resample(X, y)
    after = {
        CLASS_NAMES[int(k)]: v for k, v in zip(*np.unique(y_res, return_counts=True))
    }
    print(f"[SMOTE] After : {after}")
    return X_res, y_res


# Evaluate
def evaluate_model(model, X, y, dataset_name, model_name):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = {
        "model": model_name,
        "dataset": dataset_name,
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "auc": roc_auc_score(y, y_prob),
        "cm": confusion_matrix(y, y_pred),
    }

    print(f"\n{'='*60}")
    print(f"{model_name} — {dataset_name.upper()} SET")
    print(f"{'='*60}")
    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"ROC-AUC:   {metrics['auc']:.4f}")
    print(classification_report(y, y_pred, target_names=CLASS_NAMES, zero_division=0))
    return metrics


# Plots
def plot_confusion_matrices(results, save_dir):
    test_r = [r for r in results if r["dataset"] == "Test"]
    fig, axes = plt.subplots(1, len(test_r), figsize=(6 * len(test_r), 5))
    if len(test_r) == 1:
        axes = [axes]
    for ax, r in zip(axes, test_r):
        sns.heatmap(
            r["cm"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=ax,
        )
        ax.set_title(
            f"{r['model']} (Test)\nAcc: {r['accuracy']*100:.2f}% | F1: {r['f1']:.4f}"
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
    plt.tight_layout()
    path = save_dir / "classical_ml_confusion_matrices.png"
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"[OK] Saved: {path}")


def plot_comparison(results, save_dir):
    test_r = [r for r in results if r["dataset"] == "Test"]
    metrics = ["accuracy", "precision", "recall", "f1", "auc"]
    x, width = np.arange(len(metrics)), 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, r in enumerate(test_r):
        ax.bar(x + i * width, [r[m] for m in metrics], width, label=r["model"])
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1", "AUC"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Classical ML — Model Comparison (Test Set)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = save_dir / "classical_ml_comparison.png"
    plt.savefig(str(path), dpi=150)
    plt.close()
    print(f"[OK] Saved: {path}")


# Main
def main():
    print("\n" + "=" * 70)
    print("PNEUMONIA DETECTION - CLASSICAL ML v3")
    print("SVM + Random Forest")
    print("=" * 70)

    MODELS_PATH.mkdir(parents=True, exist_ok=True)

    try:
        X_train, y_train, X_val, y_val, X_test, y_test = load_features()
    except FileNotFoundError as e:
        print(e)
        return False

    X_train_pca, X_val_pca, X_test_pca, scaler, pca = preprocess_features(
        X_train, X_val, X_test
    )

    X_tv = np.concatenate([X_train_pca, X_val_pca])
    y_tv = np.concatenate([y_train, y_val])
    X_tv_bal, y_tv_bal = apply_smote(X_tv, y_tv)

    all_results = []

    # Model 1: SVM
    print("\n[STAGE 1] Training SVM...")
    print("-" * 70)
    svm = SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=42,
    )
    svm.fit(X_tv_bal, y_tv_bal)
    print("[OK] SVM training complete!")

    all_results.append(evaluate_model(svm, X_val_pca, y_val, "Val", "SVM"))
    all_results.append(evaluate_model(svm, X_test_pca, y_test, "Test", "SVM"))
    joblib.dump(svm, MODELS_PATH / "svm_model.pkl")
    print("[OK] SVM saved")

    # Model 2: Random Forest
    print("\n[STAGE 2] Training Random Forest...")
    print("-" * 70)
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_tv_bal, y_tv_bal)
    print("[OK] Random Forest training complete!")

    all_results.append(evaluate_model(rf, X_val_pca, y_val, "Val", "Random Forest"))
    all_results.append(evaluate_model(rf, X_test_pca, y_test, "Test", "Random Forest"))
    joblib.dump(rf, MODELS_PATH / "rf_model.pkl")
    print("[OK] Random Forest saved")

    # Save scaler + PCA
    joblib.dump(scaler, MODELS_PATH / "scaler.pkl")
    joblib.dump(pca, MODELS_PATH / "pca.pkl")
    print("[OK] Scaler + PCA saved")

    # Plots + Summary
    print("\n[STAGE 3] Saving plots...")
    plot_confusion_matrices(all_results, MODELS_PATH)
    plot_comparison(all_results, MODELS_PATH)

    summary = pd.DataFrame(
        [{k: v for k, v in r.items() if k != "cm"} for r in all_results]
    )
    summary.to_csv(MODELS_PATH / "classical_ml_results.csv", index=False)

    print("\n" + "=" * 70)
    print("FINAL SUMMARY — TEST SET")
    print("=" * 70)
    test_df = summary[summary["dataset"] == "Test"][
        ["model", "accuracy", "precision", "recall", "f1", "auc"]
    ].round(4)
    print(test_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    for f in [
        "svm_model.pkl",
        "rf_model.pkl",
        "scaler.pkl",
        "pca.pkl",
        "classical_ml_results.csv",
        "classical_ml_confusion_matrices.png",
        "classical_ml_comparison.png",
    ]:
        print(f"  {MODELS_PATH}/{f}")
    print("=" * 70 + "\n")
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
