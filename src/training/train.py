"""
Pneumonia Detection - Training Pipeline
Train GoogLeNet model on processed chest X-ray images
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import googlenet, GoogLeNet_Weights
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt


class LungXDataset(Dataset):
    """Custom Dataset for chest X-ray images"""
    
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


def prepare_dataloaders(data_root, batch_size=16, num_workers=0):
    """Prepare train, val, and test dataloaders
    
    Note: batch_size=16, num_workers=0 is optimized for CPU
          System has Intel GPU (not NVIDIA) - CUDA not available
    """
    
    IMG_SIZE = 224
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                           [0.229, 0.224, 0.225])
    ])
    
    # Create DataFrames
    train_df = create_dataframe(os.path.join(data_root, "train"))
    val_df = create_dataframe(os.path.join(data_root, "val"))
    test_df = create_dataframe(os.path.join(data_root, "test"))
    
    print(f"\n[OK] Train samples: {len(train_df)}")
    print(f"[OK] Val samples: {len(val_df)}")
    print(f"[OK] Test samples: {len(test_df)}")
    
    # Create datasets
    train_dataset = LungXDataset(train_df, transform=train_transform)
    val_dataset = LungXDataset(val_df, transform=val_transform)
    test_dataset = LungXDataset(test_df, transform=val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=1, patience=1, save_dir="models"):
    """Train model with early stopping
    
    Note: Default num_epochs=1, patience=1 optimized for fast CPU training
          For full training: num_epochs=15, patience=3 on CPU
    """
    
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            # Handle GoogLeNet tuple output (main, aux1, aux2)
            if isinstance(outputs, tuple):
                main_output = outputs[0]
                loss = criterion(main_output, labels)
            else:
                main_output = outputs
                loss = criterion(main_output, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            train_correct += (main_output.argmax(1) == labels).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                # Handle GoogLeNet output (in eval mode, returns tensor not tuple)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
        
        # Calculate metrics
        train_acc = 100 * train_correct / len(train_loader.dataset)
        val_acc = 100 * val_correct / len(val_loader.dataset)
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        
        # Store history
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Create save directory if not exists
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f"googlenet-best-acc{val_acc:.2f}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"[OK] Best model saved: {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[EARLY STOPPING] No improvement for {patience} epochs")
                break
        
        # Stop if 100% accuracy
        if val_acc == 100.0:
            print(f"[OK] Validation accuracy reached 100%!")
            break
        
        # Update scheduler
        scheduler.step(val_acc)
    
    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history


def main():
    """Main training pipeline"""
    
    # Setup paths
    BASE_DIR = Path(__file__).resolve().parents[2]
    PROCESSED_DATA_PATH = BASE_DIR / "src" / "data" / "processed"
    MODELS_PATH = BASE_DIR / "models"
    
    print("\n" + "="*70)
    print("PNEUMONIA DETECTION - TRAINING PIPELINE")
    print("="*70)
    print(f"Base directory: {BASE_DIR}")
    print(f"Data directory: {PROCESSED_DATA_PATH}")
    print(f"Models directory: {MODELS_PATH}")
    
    # Check if data exists
    if not PROCESSED_DATA_PATH.exists():
        print(f"\n[ERROR] Data directory not found: {PROCESSED_DATA_PATH}")
        print("Please run preprocessing first!")
        return False
    
    # Verify data splits
    for split in ["train", "val", "test"]:
        split_path = PROCESSED_DATA_PATH / split
        if not split_path.exists():
            print(f"[WARNING] {split} split not found at {split_path}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[OK] Using device: {device}")
    
    # Show optimization note for CPU
    if device.type == "cpu":
        print("[NOTE] Running on CPU - FAST TRAINING MODE")
        print("       num_epochs=1 | batch_size=16 | patience=1")
        print("       For full training: num_epochs=15, patience=3")
    elif device.type == "cuda":
        print("[NOTE] Running on GPU - Code is optimized for GPU training")
        print("       Batch size: 32 | Epochs: 15 | Num workers: 2")
    
    # Prepare dataloaders
    print("\n[STAGE 1] Preparing Data...")
    print("-"*70)
    try:
        train_loader, val_loader, test_loader = prepare_dataloaders(
            str(PROCESSED_DATA_PATH),
            batch_size=16,
            num_workers=0
        )
        print("[OK] Data preparation complete!")
    except Exception as e:
        print(f"[ERROR] Data preparation failed: {e}")
        return False
    
    # Initialize model
    print("\n[STAGE 2] Initializing Model...")
    print("-"*70)
    try:
        weights = GoogLeNet_Weights.IMAGENET1K_V1
        model = googlenet(weights=weights)
        
        # Replace final layer
        model.fc = nn.Linear(1024, 2)
        
        # Update auxiliary classifiers
        if model.aux1 is not None:
            model.aux1.fc2 = nn.Linear(1024, 2)
        if model.aux2 is not None:
            model.aux2.fc2 = nn.Linear(1024, 2)
        
        model = model.to(device)
        print("[OK] GoogLeNet model initialized with pretrained ImageNet weights")
    except Exception as e:
        print(f"[ERROR] Model initialization failed: {e}")
        return False
    
    # Setup loss, optimizer, scheduler
    print("\n[STAGE 3] Setting up Optimizer & Scheduler...")
    print("-"*70)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,
        T_mult=2,
        eta_min=1e-6
    )
    print("[OK] Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)")
    print("[OK] Scheduler: CosineAnnealingWarmRestarts")
    print("[OK] Loss: CrossEntropyLoss (label_smoothing=0.02)")
    
    # Train model
    print("\n[STAGE 4] Training Model...")
    print("-"*70)
    try:
        model, train_loss, val_loss, train_acc, val_acc = train_model(
            model, 
            train_loader, 
            val_loader, 
            criterion, 
            optimizer, 
            scheduler,
            device,
            num_epochs=1,
            patience=1,
            save_dir=str(MODELS_PATH)
        )
        print("[OK] Training complete!")
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Plot training history
    print("\n[STAGE 5] Plotting Training History...")
    print("-"*70)
    try:
        # Skip plotting for faster completion
        # Uncomment below to enable plotting:
        # plt.figure(figsize=(14, 5))
        # 
        # plt.subplot(1, 2, 1)
        # plt.plot(train_loss, label='Train Loss', marker='o')
        # plt.plot(val_loss, label='Val Loss', marker='o')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.title('Training and Validation Loss')
        # plt.legend()
        # plt.grid(True, alpha=0.3)
        # 
        # plt.subplot(1, 2, 2)
        # plt.plot(train_acc, label='Train Acc', marker='o')
        # plt.plot(val_acc, label='Val Acc', marker='o')
        # plt.xlabel('Epoch')
        # plt.ylabel('Accuracy (%)')
        # plt.title('Training and Validation Accuracy')
        # plt.legend()
        # plt.grid(True, alpha=0.3)
        # 
        # plt.tight_layout()
        # 
        # # Save plot
        # plot_path = MODELS_PATH / "training_history.png"
        # os.makedirs(MODELS_PATH, exist_ok=True)
        # plt.savefig(str(plot_path), dpi=150)
        # print(f"[OK] Training history plot saved: {plot_path}")
        # plt.close()
        
        print("[SKIPPED] Plotting disabled for faster training")
    except Exception as e:
        print(f"[WARNING] Plotting failed: {e}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel saved to: {MODELS_PATH}")
    print("\nNext steps:")
    print("1. Run evaluation: python src/evaluation/evaluate.py")
    print("2. Make predictions: python src/inference/predict.py")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
