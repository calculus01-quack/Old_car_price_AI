import json
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
DATA_PATH = Path("data/processed/cars_clean_v2.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
TARGET = "price_million"
RANDOM_STATE = 42
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(DATA_PATH)
print("Kích thước dataset:", df.shape)
print("Số lượng giá trị thiếu:")
print(df.isna().sum())
X = df.drop(columns=[TARGET]).copy()
y = df[TARGET].copy()
categorical_features = ["brand", "grade", "body_type", "origin", "transmission", "drivetrain", "fuel_type"]
numerical_features = [col for col in X.columns if col not in categorical_features]
X[categorical_features] = X[categorical_features].fillna("Không rõ").astype(str)
for col in numerical_features:
    X[col] = pd.to_numeric(X[col], errors="coerce")
print("\nCategorical features:", categorical_features)
print("Numerical features:", numerical_features)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE)
print("\nSố mẫu Train:", len(X_train))
print("Số mẫu Validation:", len(X_val))
print("Số mẫu Test:", len(X_test))
model = CatBoostRegressor(
    iterations=2000,
    depth=6,
    learning_rate=0.05,
    loss_function="RMSE",
    eval_metric="MAE",
    l2_leaf_reg=5,
    random_seed=RANDOM_STATE,
    verbose=100,
    thread_count=-1,
    allow_writing_files=False
)
print("\n===== BẮT ĐẦU TRAIN CATBOOST =====")
model.fit(
    X_train,
    y_train,
    cat_features=categorical_features,
    eval_set=(X_val, y_val),
    early_stopping_rounds=100,
    use_best_model=True
)
print("\nHuấn luyện hoàn tất!")
print("Best iteration:", model.get_best_iteration())
y_pred = model.predict(X_test)
y_pred = np.maximum(y_pred, 0)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print("\n===== KẾT QUẢ ĐÁNH GIÁ =====")
print(f"MAE: {mae:.2f} triệu VNĐ")
print(f"RMSE: {rmse:.2f} triệu VNĐ")
print(f"R² Score: {r2:.4f}")
feature_importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.get_feature_importance()
}).sort_values(by="importance", ascending=False)
print("\n===== FEATURE IMPORTANCE =====")
print(feature_importance.to_string(index=False))
model.save_model(str(MODEL_DIR / "catboost_car_price.cbm"))
with open(MODEL_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
    json.dump({"features": X.columns.tolist(), "categorical_features": categorical_features}, f, ensure_ascii=False, indent=4)
metrics = {
    "model": "CatBoostRegressor",
    "train_samples": len(X_train),
    "validation_samples": len(X_val),
    "test_samples": len(X_test),
    "best_iteration": model.get_best_iteration(),
    "mae_million": float(mae),
    "rmse_million": float(rmse),
    "r2_score": float(r2)
}
with open(REPORT_DIR / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=4)
feature_importance.to_csv(REPORT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig")
results = X_test.copy()
results["actual_price"] = y_test
results["predicted_price"] = y_pred
results["absolute_error"] = np.abs(y_test.to_numpy() - y_pred)
results.to_csv(REPORT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
print("\nĐã lưu model:", MODEL_DIR / "catboost_car_price.cbm")
print("Đã lưu kết quả đánh giá vào thư mục reports/")