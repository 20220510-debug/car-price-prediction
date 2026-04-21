# 🚗 PHÂN TÍCH VÀ DỰ ĐOÁN GIÁ XE Ô TÔ

## Giới thiệu đề tài
Đồ án thực hiện phân tích và xây dựng mô hình dự đoán giá xe ô tô cũ sử dụng kỹ thuật Machine Learning.  
Dự án tập trung vào việc làm sạch dữ liệu, khám phá dữ liệu (EDA), trực quan hóa và huấn luyện các mô hình hồi quy để dự đoán giá bán xe.

**Mục tiêu chính:**
- Xây dựng pipeline hoàn chỉnh từ dữ liệu thô đến mô hình dự đoán
- Phân tích các yếu tố ảnh hưởng mạnh nhất đến giá xe
- So sánh hiệu suất giữa Linear Regression và Random Forest Regressor

---

## Cấu trúc thư mục dự án
CarPrice/
├── data/
│   ├── raw/                    ← Dữ liệu gốc (car data.csv)
│   ├── processed/              ← Dữ liệu đã làm sạch
│   └── models/                 ← Mô hình đã huấn luyện (.pkl)
├── notebooks/
│   ├── 01_Data_Cleaning.ipynb
│   ├── 02_EDA_and_Visualization.ipynb
│   └── 03_Modeling_and_Evaluation.ipynb
├── reports/
│   └── figures/                ← Tất cả biểu đồ được lưu
├── src/                        ← (Tùy chọn) Code modules
├── main.py                     ← File chạy toàn bộ pipeline
├── requirements.txt
├── README.md
└── config.py
text---

## Công nghệ và thư viện sử dụng

- **Ngôn ngữ**: Python 3.12 / 3.13
- **Môi trường**: PyCharm + Virtual Environment
- **Thư viện chính**:
  - `pandas`, `numpy` – Xử lý dữ liệu
  - `matplotlib`, `seaborn` – Trực quan hóa
  - `scikit-learn` – Machine Learning
  - `joblib` – Lưu và tải mô hình

---

## Hướng dẫn cài đặt và chạy dự án

### 1. Cài đặt môi trường

```bash
# Clone project (nếu có) hoặc mở thư mục trong PyCharm
cd CarPrice

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt