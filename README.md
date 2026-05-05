# 🫁 Phát Hiện Viêm Phổi Từ Ảnh X-Quang Ngực

## 📌 Giới thiệu

Hệ thống phân loại ảnh X-quang ngực thành hai nhãn:

- **NORMAL** — ảnh bình thường
- **PNEUMONIA** — ảnh có dấu hiệu viêm phổi

Phương pháp sử dụng: trích xuất đặc trưng thủ công (HOG, LBP, GLCM, Gabor) kết hợp mô hình học máy cổ điển (SVM, Random Forest), triển khai qua API Flask.

---

## 👥 Thành viên nhóm

| MSSV       | Họ và tên            |
| ---------- | -------------------- |
| 2251120106 | Võ Văn Sáu           |
| 2251120109 | Nguyễn Hoàng Hảo Tâm |
| 2251120072 | Nguyễn Văn Duy       |
| 2251120184 | Đỗ Nguyễn Thiên      |
| 2251120124 | Phạm Công Trứ        |
| 2251120132 | Bùi Trọng Vũ         |

---

## 📁 Cấu Trúc Thư Mục

```
PNEUMONIA_DETECTION/
│
├── data/                        # Dữ liệu ảnh X-quang
│   ├── features/                # Đặc trưng đã trích xuất (lưu dạng .npy hoặc .pkl)
│   ├── processed/               # Ảnh đã qua tiền xử lý
│   ├── train/                   # Tập huấn luyện (NORMAL / PNEUMONIA)
│   ├── val/                     # Tập validation
│   └── test/                    # Tập kiểm tra
│
├── models/                      # Mô hình và kết quả đã huấn luyện
│   ├── svm_model.pkl            # Mô hình SVM đã huấn luyện
│   ├── rf_model.pkl             # Mô hình Random Forest đã huấn luyện
│   ├── scaler.pkl               # StandardScaler đã fit
│   ├── pca.pkl                  # PCA đã fit
│   ├── classical_ml_comparison.png          # Biểu đồ so sánh hiệu suất
│   └── classical_ml_confusion_matrices.png  # Ma trận nhầm lẫn
│
├── src/
│   ├── api/
│   │   └── app.py               # API Flask — endpoint dự đoán
│   ├── features/
│   │   └── feature_extraction.py  # Trích xuất HOG, LBP, GLCM, Gabor
│   ├── preprocessing/
│   │   └── preprocess.py        # Tiền xử lý ảnh (CLAHE, Bilateral Filter)
│   └── training/
│       └── train_classical_ml.py  # Huấn luyện SVM và Random Forest
│
├── requirements.txt             # Danh sách thư viện cần cài
├── .gitignore
└── README.md
```

---

## ⚙️ Cài Đặt Môi Trường

### Yêu cầu

- Python 3.8+
- pip

### Các bước cài đặt

```bash
# 1. Clone repository
git clone <repository-url>
cd PNEUMONIA_DETECTION

# 2. Tạo môi trường ảo
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# 3. Cài đặt các thư viện
pip install -r requirements.txt
```

---

## 🗂️ Dữ Liệu

Bộ dữ liệu sử dụng: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) từ Kaggle.

| Tập        | NORMAL   | PNEUMONIA |
| ---------- | -------- | --------- |
| Train      | 1341     | 3875      |
| Validation | 24       | 23        |
| Test       | 234      | 390       |
| **Tổng**   | **1599** | **4288**  |

**Cách tổ chức thư mục dữ liệu:**

```
data/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── test/
    ├── NORMAL/
    └── PNEUMONIA/
```

---

## 🚀 Hướng Dẫn Sử Dụng

### 1. Tiền xử lý ảnh

```bash
python src/preprocessing/preprocess.py
```

Kết quả ảnh đã xử lý được lưu vào `data/processed/`.

### 2. Trích xuất đặc trưng

```bash
python src/features/feature_extraction.py
```

Vector đặc trưng (HOG + LBP + GLCM + Gabor) được lưu vào `data/features/`.

### 3. Huấn luyện mô hình

```bash
python src/training/train_classical_ml.py
```

Mô hình sau khi huấn luyện được lưu vào `models/`.

### 4. Chạy API dự đoán

```bash
python src/api/app.py
```

API khởi động tại `http://localhost:5000`

---

## 🌐 API Endpoints

### `POST /predict`

Nhận ảnh X-quang và trả về kết quả dự đoán.

**Request:** `multipart/form-data` với field `file` là ảnh X-quang.

**Response:**

```json
{
  "svm": {
    "label": "PNEUMONIA",
    "confidence": 0.92
  },
  "random_forest": {
    "label": "PNEUMONIA",
    "confidence": 0.87
  }
}
```

## 📊 Kết Quả

| Mô hình       | Accuracy | Precision | Recall | F1-score |
| ------------- | -------- | --------- | ------ | -------- |
| SVM (RBF)     | 78.37%   | 74.57%    | 99.23% | 85.15%   |
| Random Forest | 74.04%   | 71.11%    | 98.46% | 82.58%   |

**Nhận xét:** SVM đạt kết quả tốt hơn Random Forest trên tập test. Cả hai mô hình đạt recall cao, phù hợp mục tiêu sàng lọc ban đầu.

---

## 🛠️ Công Nghệ Sử Dụng

| Thư viện           | Mục đích                         |
| ------------------ | -------------------------------- |
| `opencv-python`    | Đọc ảnh, CLAHE, Bilateral Filter |
| `scikit-image`     | Trích xuất HOG, LBP              |
| `scikit-learn`     | SVM, Random Forest, PCA, SMOTE   |
| `imbalanced-learn` | SMOTE cân bằng dữ liệu           |
| `flask`            | Triển khai API                   |
| `numpy`, `scipy`   | Xử lý số liệu                    |

---

## 📚 Tài Liệu Tham Khảo

1. Rajpurkar et al. (2017). _CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning._ arXiv:1711.05225.
2. Kermany et al. (2018). _Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning._ Cell, 172(5).
3. Dalal & Triggs (2005). _Histograms of Oriented Gradients for Human Detection._ CVPR.
4. Ojala et al. (2002). _Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns._ IEEE TPAMI.
5. Haralick et al. (1973). _Textural Features for Image Classification._ IEEE TSMC.
