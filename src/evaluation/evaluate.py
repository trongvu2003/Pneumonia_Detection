"""
Pneumonia Detection - Model Evaluation
Evaluate trained model on test set
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import googlenet, GoogLeNet_Weights
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


class LungXDataset:
    """Simple Dataset for evaluation"""
    
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]['image_path']
        label = self.dataframe.iloc[idx]['label']
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        return image, label


def create_dataframe(root_dir):
    """Create DataFrame from directory structure"""
    data = []
    classes = ["NORMAL", "PNEUMONIA"]
    for label, cls in enumerate(classes):
        class_dir = os.path.join(root_dir, cls)
        if not os.path.exists(class_dir):
            continue
        for img in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img)
            data.append({"image_path": img_path, "label": label})
    return pd.DataFrame(data)


def evaluate(model, dataloader, device, dataset_name="Test"):
    """Evaluate model on dataset"""
    
    model.eval()
    y_true = []
    y_pred = []
    y_prob = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            
            # Handle tuple outputs from GoogLeNet
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(dim=1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_prob.extend(probs[:, 1].cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    cm = confusion_matrix(y_true, y_pred)
    
    # Print results
    print("\n" + "="*70)
    print(f"{dataset_name.upper()} SET - EVALUATION METRICS")
    print("="*70)
    print(f"Accuracy:   {acc*100:.2f}%")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall:     {rec:.4f}")
    print(f"F1-Score:   {f1:.4f}")
    print(f"ROC-AUC:    {auc:.4f}")
    
    # Classification report
    idx_to_class = {0: "NORMAL", 1: "PNEUMONIA"}
    print(f"\nClassification Report ({dataset_name}):")
    print(classification_report(y_true, y_pred, target_names=[idx_to_class[i] for i in range(2)]))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[idx_to_class[i] for i in range(2)],
                yticklabels=[idx_to_class[i] for i in range(2)])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.tight_layout()
    plt.show()
    
    return acc, prec, rec, f1, auc


def main():
    """Main evaluation pipeline"""
    
    # Setup paths
    BASE_DIR = Path(__file__).resolve().parents[2]
    PROCESSED_DATA_PATH = BASE_DIR / "data" / "processed"
    MODELS_PATH = BASE_DIR / "models"
    
    print("\n" + "="*70)
    print("PNEUMONIA DETECTION - EVALUATION")
    print("="*70)
    print(f"Data directory: {PROCESSED_DATA_PATH}")
    print(f"Models directory: {MODELS_PATH}")
    
    # Check if models exist
    model_files = list(MODELS_PATH.glob("*.pth"))
    if not model_files:
        print(f"\n[ERROR] No trained models found in {MODELS_PATH}")
        print("Please run training first!")
        return False
    
    # Use latest model
    best_model_path = max(model_files, key=lambda p: p.stat().st_mtime)
    print(f"\n[OK] Using model: {best_model_path.name}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OK] Using device: {device}")
    
    # Load model
    print("\n[STAGE 1] Loading Model...")
    print("-"*70)
    try:
        weights = GoogLeNet_Weights.IMAGENET1K_V1
        model = googlenet(weights=weights)
        model.fc = nn.Linear(1024, 2)
        
        state_dict = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        print(f"[OK] Model loaded from: {best_model_path}")
    except Exception as e:
        print(f"[ERROR] Model loading failed: {e}")
        return False
    
    # Prepare dataloaders
    print("\n[STAGE 2] Preparing Data...")
    print("-"*70)
    
    IMG_SIZE = 224
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])
    ])
    
    try:
        # Create test dataframe
        test_df = create_dataframe(os.path.join(str(PROCESSED_DATA_PATH), "test"))
        print(f"[OK] Test samples: {len(test_df)}")
        
        # Create test dataset and loader
        test_dataset = LungXDataset(test_df, transform=val_transform)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)
        
        # Create val dataframe for comparison
        val_df = create_dataframe(os.path.join(str(PROCESSED_DATA_PATH), "val"))
        if len(val_df) > 0:
            val_dataset = LungXDataset(val_df, transform=val_transform)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
            print(f"[OK] Val samples: {len(val_df)}")
        else:
            val_loader = None
    except Exception as e:
        print(f"[ERROR] Data preparation failed: {e}")
        return False
    
    # Evaluate on validation set
    if val_loader is not None:
        print("\n[STAGE 3] Evaluating on Validation Set...")
        print("-"*70)
        try:
            val_acc, val_prec, val_rec, val_f1, val_auc = evaluate(
                model, val_loader, device, dataset_name="Validation"
            )
        except Exception as e:
            print(f"[ERROR] Validation evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Evaluate on test set
    print("\n[STAGE 4] Evaluating on Test Set...")
    print("-"*70)
    try:
        test_acc, test_prec, test_rec, test_f1, test_auc = evaluate(
            model, test_loader, device, dataset_name="Test"
        )
    except Exception as e:
        print(f"[ERROR] Test evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE!")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
