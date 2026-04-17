# Training & Evaluation Pipeline - Pneumonia Detection

## Mô tả

Pipeline training và evaluation cho mô hình phát hiện viêm phổi bằng GoogLeNet.

- **Model:** GoogLeNet (Inception v1) với ImageNet pretrained weights
- **Input:** Ảnh X-quang 224×224 từ thư mục `data/processed/`
- **Output:** File mô hình `.pth` trong thư mục `models/`

## Cấu trúc File

```
src/training/
├── train.py           # Script training chính
├── evaluate.py        # Script đánh giá mô hình
└── README.md          # File này
```

## Quick Start

### 1. Chuẩn bị dữ liệu

Đảm bảo dữ liệu đã được xử lý trong `data/processed/`:

```
data/processed/
├── train/
│   ├── NORMAL/    (ảnh)
│   └── PNEUMONIA/ (ảnh)
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

### 2. Cài đặt Dependencies

```bash
pip install torch torchvision scikit-learn matplotlib seaborn pillow pandas numpy
```

### 3. Chạy Training

```bash
# Từ thư mục gốc project
python src/training/train.py
```

**Output:**
- Mô hình tốt nhất được lưu trong `models/googlenet-best-acc{X.XX}.pth`
- Biểu đồ training history được lưu trong `models/training_history.png`

### 4. Đánh giá Mô hình (sau khi train)

```bash
python src/training/evaluate.py
```

**Output:**
- Metrics trên validation/test set: Accuracy, Precision, Recall, F1, ROC-AUC
- Confusion Matrix
- Classification Report

## Chi tiết Code

### train.py

**Các hàm chính:**

#### `LungXDataset(Dataset)`
Custom Dataset class để load ảnh từ directory

#### `create_dataframe(root_dir)`
Tạo Pandas DataFrame từ cấu trúc thư mục:
```python
# Output:
# image_path | label
# img1.jpg   | 0
# img2.jpg   | 1
```

#### `prepare_dataloaders(data_root, batch_size=32, num_workers=0)`
Chuẩn bị DataLoaders với augmentation:

**Train Transform:**
- Resize (224×224)
- Random Horizontal Flip
- Random Rotation (15°)
- Color Jitter
- Normalization (ImageNet stats)

**Val/Test Transform:**
- Resize (224×224)
- Normalization

#### `train_model(...)`
Training loop chính:
- **Loss:** CrossEntropyLoss (label_smoothing=0.02)
- **Optimizer:** AdamW (lr=1e-4, weight_decay=1e-4)
- **Scheduler:** CosineAnnealingWarmRestarts
- **Early Stopping:** Nếu không cải thiện trong `patience` epochs

#### `main()`
Pipeline chính:
1. Kiểm tra dữ liệu
2. Khởi tạo DataLoaders
3. Tải model GoogLeNet pretrained
4. Train mô hình
5. Vẽ biểu đồ training history

### evaluate.py

**Các hàm chính:**

#### `evaluate(model, dataloader, device, dataset_name)`
Đánh giá mô hình trên dataloader:

**Returns:**
- Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Vẽ Confusion Matrix

#### `main()`
Pipeline đánh giá:
1. Tìm model tốt nhất trong `models/`
2. Load model
3. Đánh giá trên validation set (nếu có)
4. Đánh giá trên test set
5. In kết quả chi tiết

## Configuration

Các parameter có thể tùy chỉnh trong `train.py`:

```python
# Data loading
batch_size = 32
num_workers = 0

# Model
IMG_SIZE = 224  # Input image size

# Training
num_epochs = 25
patience = 5  # Early stopping patience

# Optimizer
lr = 1e-4
weight_decay = 1e-4

# Data augmentation
RandomRotation = 15
ColorJitter = 0.2
```

## Model Architecture

**GoogLeNet (Inception v1)**

```
Input: 224×224×3
  ↓
[Pretrained ImageNet weights]
  ↓
[Replace FC layer: 1024 → 2 classes]
  ↓
Output: [NORMAL, PNEUMONIA]
```

**Auxiliary Classifiers:**
- GoogLeNet có 2 auxiliary classifiers (aux1, aux2)
- Được sử dụng chỉ trong training
- Giúp gradient flow tốt hơn

## Training Curve

Khi training xong, file `models/training_history.png` sẽ chứa:

```
┌─────────────────┬──────────────────┐
│ Training Loss   │ Training Accuracy│
│ vs Val Loss     │ vs Val Accuracy  │
└─────────────────┴──────────────────┘
```

## Expected Performance

Dựa trên notebook gốc:

| Metric | Value |
|--------|-------|
| Accuracy | ~98-99% |
| Precision | ~0.98 |
| Recall | ~0.99 |
| F1-Score | ~0.98 |
| ROC-AUC | ~0.99 |

## Troubleshooting

### Error: "Data directory not found"
```bash
# Kiểm tra dữ liệu đã được xử lý
ls data/processed/train/NORMAL/
```

### Error: "CUDA out of memory"
```python
# Giảm batch size trong train.py
batch_size = 16  # hoặc 8
```

### Error: "No module named torch"
```bash
pip install torch torchvision
```

### GPU không được dùng
```python
# Code tự động detect GPU
# Nếu có CUDA, sẽ sử dụng GPU tự động
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Output Files

Sau khi training, thư mục `models/` sẽ chứa:

```
models/
├── googlenet-best-acc98.50.pth    # Model tốt nhất
├── googlenet-best-acc98.75.pth    # (nếu cải thiện)
└── training_history.png            # Biểu đồ training
```

**Sử dụng model để predict:**
```python
import torch
from pathlib import Path

BASE_DIR = Path(".")
model_path = BASE_DIR / "models" / "googlenet-best-acc98.50.pth"

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_path, map_location=device))
```

## Next Steps

1. **Training:** `python src/training/train.py`
2. **Evaluation:** `python src/training/evaluate.py`

## References

- GoogLeNet Paper: https://arxiv.org/abs/1409.4842
- PyTorch Docs: https://pytorch.org/docs/
- TorchVision Models: https://pytorch.org/vision/stable/models.html
