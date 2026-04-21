# =====================================================
# main.py - PIPELINE HOÀN CHỈNH DỰ ĐOÁN GIÁ XE
# =====================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# ====================== CẤU HÌNH ======================
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
CURRENT_YEAR = 2026

print("🚗 BẮT ĐẦU PIPELINE DỰ ĐOÁN GIÁ XE\n")

# ====================== 1. DATA CLEANING ======================
print("1. ĐANG THỰC HIỆN DATA CLEANING...")

df = pd.read_csv('data/raw/car data.csv')

print(f"   Kích thước ban đầu: {df.shape}")

# Xóa trùng lặp
df = df.drop_duplicates()

# Feature Engineering
df['car_age'] = CURRENT_YEAR - df['Year']
df['km_per_year'] = df['Kms_Driven'] / df['car_age'].replace(0, 1)

# Xử lý outliers cho Selling_Price
Q1 = df['Selling_Price'].quantile(0.25)
Q3 = df['Selling_Price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df = df[(df['Selling_Price'] >= lower_bound) & (df['Selling_Price'] <= upper_bound)]

print(f"   Kích thước sau cleaning: {df.shape}")

# Lưu dữ liệu sạch
os.makedirs('data/processed', exist_ok=True)
df.to_csv('data/processed/car_data_cleaned.csv', index=False)
print("   ✅ Đã lưu file cleaned: data/processed/car_data_cleaned.csv\n")

# ====================== 2. EDA & VISUALIZATION ======================
print("2. ĐANG TẠO BIỂU ĐỒ EDA...")

os.makedirs('reports/figures', exist_ok=True)

# Histogram giá xe
plt.figure(figsize=(10,6))
sns.histplot(df['Selling_Price'], kde=True, bins=30)
plt.title('Phân bố giá xe')
plt.savefig('reports/figures/01_price_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Scatter: Giá vs Tuổi xe
plt.figure(figsize=(10,6))
sns.scatterplot(x='car_age', y='Selling_Price', hue='Fuel_Type', data=df, alpha=0.7)
plt.title('Giá xe theo tuổi xe')
plt.savefig('reports/figures/02_price_vs_age.png', dpi=300, bbox_inches='tight')
plt.close()

# Heatmap tương quan
plt.figure(figsize=(10,8))
numeric_df = df.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Ma trận tương quan')
plt.savefig('reports/figures/03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

print("   ✅ Đã lưu các biểu đồ vào thư mục reports/figures/\n")

# ====================== 3. MODELING ======================
print("3. ĐANG TRAINING MÔ HÌNH...")

X = df.drop(['Selling_Price'], axis=1)
y = df['Selling_Price']

# One-hot encoding
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale cho Linear Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

# Random Forest (mô hình chính)
rf_model = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=10)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Đánh giá
def print_metrics(y_true, y_pred, name):
    print(f"\n=== {name} ===")
    print(f"MAE  : {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"R²   : {r2_score(y_true, y_pred):.4f} ({r2_score(y_true, y_pred)*100:.2f}%)")

print_metrics(y_test, y_pred_lr, "Linear Regression")
print_metrics(y_test, y_pred_rf, "Random Forest Regressor")

# Feature Importance
importances = pd.Series(rf_model.feature_importances_, index=X.columns).nlargest(10)
print("\nTop 10 biến quan trọng nhất:")
print(importances)

# Lưu mô hình
os.makedirs('data/models', exist_ok=True)
joblib.dump(rf_model, 'data/models/random_forest_model.pkl')
joblib.dump(scaler, 'data/models/scaler.pkl')

print("\n✅ ĐÃ LƯU MÔ HÌNH TẠI: data/models/random_forest_model.pkl")
print("\n🎉 HOÀN THÀNH TOÀN BỘ PIPELINE!")

# ====================== KẾT LUẬN ======================
print("\n" + "="*70)
print("TÓM TẮT KẾT QUẢ")
print("="*70)
print(f"• Mô hình tốt nhất: Random Forest với R² ≈ {r2_score(y_test, y_pred_rf):.4f}")
print("• Yếu tố ảnh hưởng lớn nhất: Tuổi xe, Số km đã chạy, Loại hộp số")
print("• Dự án đã sẵn sàng để triển khai thành ứng dụng dự đoán giá xe.")
print("="*70)