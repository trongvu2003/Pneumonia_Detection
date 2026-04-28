# Pneumonia Detection - Training & Inference Pipeline

## Tổng Quan

Đây là pipeline hoàn chỉnh để **training** và **inference** mô hình phát hiện viêm phổi từ ảnh X-quang:

1. **Feature Extraction:** Tải và xử lý ảnh từ `src/data/processed/`
2. **Model Training:** Train mô hình GoogLeNet trên dữ liệu đã xử lý
3. **Evaluation:** Đánh giá mô hình trên validation/test sets
4. **Inference:** Dự đoán trên ảnh mới (nếu có)

⚠️ **Hardware Note:** System has Intel Iris Xe GPU (CUDA only supports NVIDIA). Training runs on CPU with FAST MODE settings (num_epochs=1, patience=1).

## Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

**Packages chính:**
- PyTorch: Deep learning framework
- TorchVision: Pre-trained models
- Scikit-learn: Metrics & utilities
- Pillow: Image loading
- Matplotlib/Seaborn: Visualization

## Cấu Trúc Thư Mục

```
Project/
├── src/
│   ├── data/
│   │   └── processed/      # Ảnh đã xử lý
│   │       ├── train/
│   │       ├── val/
│   │       └── test/
│   ├── training/
│   │   ├── train.py        # Script training chính
│   │   ├── evaluate.py     # Script evaluation
│   │   └── README.md       # Chi tiết code
│   ├── evaluation/
│   │   └── evaluate.py     # Model evaluation
│   ├── inference/
│   ├── features/
│   ├── models/
│   └── preprocessing/
├── models/                 # Lưu model training
├── requirements.txt        # Dependencies
└── README.md              # File này
```

## Quickstart

### Step 1: Chuẩn Bị Dữ Liệu

✅ Dữ liệu phải đã được xử lý trong `src/data/processed/`:

```bash
# Kiểm tra dữ liệu
ls src/data/processed/train/NORMAL/ | head -5
ls src/data/processed/train/PNEUMONIA/ | head -5
```

### Step 2: Training Mô Hình

```bash
python src/training/train.py
```

**Output:**
- Mô hình tốt nhất: `models/googlenet-best-acc{XX.XX}.pth`
- Biểu đồ training: `models/training_history.png`
- Logs in terminal

**Thời gian (FAST MODE):**
- CPU: ~15-20 phút (1 epoch, no plotting)
- Để train full: thay num_epochs=1 → 15, patience=1 → 3

### Step 3: Đánh Giá Mô Hình

```bash
python src/evaluation/evaluate.py
```

**Output:**
- Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Confusion Matrix
- Classification Report

## Training Details

### Model Architecture

**GoogLeNet (Inception v1)**
- Pretrained on ImageNet
- Input: 224×224×3
- Output: 2 classes (NORMAL, PNEUMONIA)
- Auxiliary classifiers for better gradient flow

### Data Augmentation (Training)

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
```

### Hyperparameters

| Parameter | Value | Mô tả |
|-----------|-------|--------|
| Batch Size | 16 | Số ảnh mỗi batch (CPU) |
| Learning Rate | 1e-4 | AdamW optimizer |
| Weight Decay | 1e-4 | L2 regularization |
| **Epochs** | **1** | **FAST MODE - quick training** |
| **Patience** | **1** | **Early stopping (FAST MODE)** |
| Num Workers | 0 | Data loading threads (CPU) |
| Label Smoothing | 0.02 | Regularization |
| Image Size | 224 | Input size (GoogLeNet standard) |
| Plotting | Disabled | Save time on CPU |

### Optimizer & Scheduler

```python
# Optimizer
optimizer = AdamW(
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-4
)

# Scheduler
scheduler = CosineAnnealingWarmRestarts(
    T_0=5,
    T_mult=2,
    eta_min=1e-6
)
```

## Expected Results

Dựa trên notebook gốc, expected performance:

| Metric | Value |
|--------|-------|
| **Accuracy** | 98-99% |
| **Precision** | 0.98+ |
| **Recall** | 0.99+ |
| **F1-Score** | 0.98+ |
| **ROC-AUC** | 0.99+ |

## File Chi Tiết

### train.py

**Chức năng:**
- Load dữ liệu từ `src/data/processed/`
- Tạo DataLoaders với augmentation
- Initialize mô hình GoogLeNet
- Training loop với early stopping (1 epoch FAST MODE)
- Lưu best model
- Skip plotting cho tốc độ

**Usage:**
```bash
python src/training/train.py
```

**Key functions:**
- `create_dataframe(root_dir)` - Tạo DataFrame từ directory
- `prepare_dataloaders()` - Chuẩn bị DataLoaders
- `train_model()` - Training loop
- `main()` - Pipeline chính

### evaluate.py

**Chức năng:**
- Load model tốt nhất từ `models/`
- Evaluate trên validation/test sets
- Tính metrics: Accuracy, Precision, Recall, F1, ROC-AUC
- Plot Confusion Matrix

**Usage:**
```bash
python src/evaluation/evaluate.py
```

## Troubleshooting

### ❌ "Data directory not found"

```bash
# Kiểm tra dữ liệu tồn tại
ls -la src/data/processed/
```

### "CUDA out of memory"

```python
# Giảm batch size trong train.py
batch_size = 8  # hoặc 4 (nếu cần thiết)
```

### ⚠️ "CUDA not available" / GPU không được dùng

**Nguyên nhân:** System có Intel Iris Xe GPU nhưng CUDA chỉ hỗ trợ NVIDIA GPU

**Giải pháp:**
1. Dùng CPU (hiện tại): Tốt hơn, batch_size=16, epochs=15
2. Cài Intel Extension for PyTorch (phức tạp)

```python
# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")  # Will print 'cpu'
```

### ❌ "Module not found"

```bash
pip install -r requirements.txt
```

### ⚡ Training quá chậm?

**Giảm thời gian:**
```python
# Hiện tại: FAST MODE (1 epoch)
# Để chạy full training (15 epochs):
num_epochs = 15  # thay vì 1
patience = 3     # thay vì 1

# Hoặc enable plotting:
# Uncomment plotting section trong train.py
```

**Thời gian ước tính (CPU):**
- 1 epoch (FAST MODE): ~15-20 phút ⚡ (hiện tại)
- 5 epochs: ~1.5-2 giờ
- 15 epochs: ~6-10 giờ (full training)

## Training Output Log Example

```
======================================================================
PNEUMONIA DETECTION - TRAINING PIPELINE
======================================================================
Base directory: C:\Project\Pneumonia_Detection
Data directory: C:\Project\Pneumonia_Detection\src\data\processed

[OK] Train samples: 5216
[OK] Val samples: 47
[OK] Test samples: 624

[OK] Using device: cpu
[NOTE] Running on CPU - FAST TRAINING MODE
       num_epochs=1 | batch_size=16 | patience=1
       For full training: num_epochs=15, patience=3

[STAGE 2] Initializing Model...
[OK] GoogLeNet model initialized with pretrained ImageNet weights

[STAGE 3] Setting up Optimizer & Scheduler...
[OK] Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)
[OK] Scheduler: CosineAnnealingWarmRestarts

[STAGE 4] Training Model...
Epoch 1/1
Train Loss: 0.1234 | Train Acc: 95.23%
Val Loss:   0.0892 | Val Acc:   96.45%
[OK] Best model saved: models/googlenet-best-acc96.45.pth

[STAGE 5] Plotting Training History...
[SKIPPED] Plotting disabled for faster training

======================================================================
TRAINING COMPLETE!
======================================================================
```

## 🎓 Model Files

Thư mục `models/` sẽ chứa:

```
models/
├── googlenet-best-acc98.50.pth   # Best model
└── training_history.png           # Training curves
```

**Sử dụng model:**
```python
import torch

model_path = "models/googlenet-best-acc98.50.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
```

## Advanced Usage

### Custom Training Parameters

Chỉnh sửa trong `src/training/train.py` (main function):

```python
# Hiện tại: FAST MODE
num_epochs = 1     # 1 epoch
patience = 1       # early stop nhanh

# Để full training (15 epochs):
num_epochs = 15
patience = 3

# Data loading
batch_size = 16    # tối ưu cho CPU
num_workers = 0

# Optimizer
lr = 1e-4
weight_decay = 1e-4
```

### Sử Dụng CPU Only

```bash
# Trên Linux/Mac
CUDA_VISIBLE_DEVICES="" python src/training/train.py

# Trên Windows PowerShell
$env:CUDA_VISIBLE_DEVICES=""; python src/training/train.py
```

## References

- GoogLeNet Paper: https://arxiv.org/abs/1409.4842
- PyTorch Documentation: https://pytorch.org/docs/
- TorchVision Models: https://pytorch.org/vision/stable/models.html

## Support

Nếu có vấn đề:

1. Kiểm tra `src/training/README.md` cho chi tiết code
2. Xem phần Troubleshooting ở trên
3. Check error message chi tiết

---

**Created:** April 2026  
**Model:** GoogLeNet (Inception v1)  
**Task:** Pneumonia Detection from Chest X-rays
