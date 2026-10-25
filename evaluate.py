import json
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
DATA_PATH = Path("data/processed/cars_clean_v2.csv")
MODEL_PATH = Path("models/catboost_car_price.cbm")
FEATURE_PATH = Path("models/feature_columns.json")
REPORT_DIR = Path("reports")
RANDOM_STATE = 42
df = pd.read_csv(DATA_PATH)
model = CatBoostRegressor()
model.load_model(str(MODEL_PATH))
with open(FEATURE_PATH, "r", encoding="utf-8") as f:
    config = json.load(f)
with open(REPORT_DIR / "metrics.json", "r", encoding="utf-8") as f:
    saved_metrics = json.load(f)
features = config["features"]
categorical_features = config["categorical_features"]
X = df[features].copy()
y = df["price_million"].copy()
X[categorical_features] = X[categorical_features].fillna("Không rõ").astype(str)
for col in features:
    if col not in categorical_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE)
if len(X_train) != saved_metrics["train_samples"] or len(X_val) != saved_metrics["validation_samples"] or len(X_test) != saved_metrics["test_samples"]:
    raise ValueError("Kích thước dataset đã thay đổi sau khi train. Hãy kiểm tra lại dữ liệu và model.")
def calculate_metrics(y_true, y_pred):
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred))
    }
train_pred = model.predict(X_train)
val_pred = model.predict(X_val)
train_metrics = calculate_metrics(y_train, train_pred)
val_metrics = calculate_metrics(y_val, val_pred)
test_metrics = {
    "MAE": saved_metrics["mae_million"],
    "RMSE": saved_metrics["rmse_million"],
    "R2": saved_metrics["r2_score"]
}
comparison = pd.DataFrame([
    {"Dataset": "Train", "Samples": len(X_train), **train_metrics},
    {"Dataset": "Validation", "Samples": len(X_val), **val_metrics},
    {"Dataset": "Test", "Samples": len(X_test), **test_metrics}
])
print("\n========== ĐÁNH GIÁ CATBOOST ==========")
print(comparison.round(4).to_string(index=False))
comparison.to_csv(REPORT_DIR / "dataset_comparison.csv", index=False, encoding="utf-8-sig")
mae_gap = val_metrics["MAE"] - train_metrics["MAE"]
mae_ratio = val_metrics["MAE"] / train_metrics["MAE"] if train_metrics["MAE"] > 0 else np.nan
print("\n========== KIỂM TRA OVERFITTING ==========")
print(f"Train MAE: {train_metrics['MAE']:.2f} triệu VNĐ")
print(f"Validation MAE: {val_metrics['MAE']:.2f} triệu VNĐ")
print(f"Chênh lệch MAE: {mae_gap:.2f} triệu VNĐ")
print(f"Tỷ lệ Validation MAE / Train MAE: {mae_ratio:.2f}")
results = pd.read_csv(REPORT_DIR / "test_predictions.csv")
required_columns = ["brand", "actual_price", "predicted_price", "absolute_error"]
if not all(col in results.columns for col in required_columns):
    raise ValueError("test_predictions.csv không có đầy đủ các cột cần thiết.")
results["actual_price"] = pd.to_numeric(results["actual_price"], errors="coerce")
results["predicted_price"] = pd.to_numeric(results["predicted_price"], errors="coerce")
results = results.dropna(subset=["actual_price", "predicted_price"]).copy()
results = results[results["actual_price"] > 0].copy()
results["absolute_error"] = (results["actual_price"] - results["predicted_price"]).abs()
results["percentage_error"] = results["absolute_error"] / results["actual_price"] * 100
brand_errors = results.groupby("brand").agg(
    samples=("absolute_error", "size"),
    MAE=("absolute_error", "mean"),
    median_error=("absolute_error", "median"),
    mean_percentage_error=("percentage_error", "mean")
).reset_index()
brand_errors = brand_errors.sort_values("MAE", ascending=False)
brand_errors.to_csv(REPORT_DIR / "errors_by_brand.csv", index=False, encoding="utf-8-sig")
print("\n========== SAI SỐ THEO HÃNG XE ==========")
print("Các hãng có ít nhất 20 mẫu trong tập Test:")
print(brand_errors[brand_errors["samples"] >= 20].round(2).to_string(index=False))
price_bins = [0, 300, 600, 1000, 2000, np.inf]
price_labels = ["Dưới 300 triệu", "300-600 triệu", "600 triệu-1 tỷ", "1-2 tỷ", "Trên 2 tỷ"]
results["price_segment"] = pd.cut(results["actual_price"], bins=price_bins, labels=price_labels, include_lowest=True)
segment_errors = results.groupby("price_segment", observed=True).agg(
    samples=("absolute_error", "size"),
    MAE=("absolute_error", "mean"),
    median_error=("absolute_error", "median"),
    mean_percentage_error=("percentage_error", "mean")
).reset_index()
segment_errors.to_csv(REPORT_DIR / "errors_by_price_segment.csv", index=False, encoding="utf-8-sig")
print("\n========== SAI SỐ THEO PHÂN KHÚC GIÁ ==========")
print(segment_errors.round(2).to_string(index=False))
worst_predictions = results.sort_values("absolute_error", ascending=False).head(20)
worst_predictions.to_csv(REPORT_DIR / "top_20_errors.csv", index=False, encoding="utf-8-sig")
print("\n========== 20 DỰ ĐOÁN SAI NHIỀU NHẤT ==========")
display_columns = ["brand", "grade", "year", "actual_price", "predicted_price", "absolute_error"]
display_columns = [col for col in display_columns if col in worst_predictions.columns]
print(worst_predictions[display_columns].round(2).to_string(index=False))
summary = {
    "train": train_metrics,
    "validation": val_metrics,
    "test": test_metrics,
    "validation_train_mae_gap": float(mae_gap),
    "validation_train_mae_ratio": float(mae_ratio)
}
with open(REPORT_DIR / "evaluation_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=4)
print("\nĐã lưu toàn bộ báo cáo đánh giá vào thư mục reports/")