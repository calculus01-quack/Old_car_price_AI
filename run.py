import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from catboost import CatBoostRegressor
MODEL_PATH = Path("models/catboost_car_price.cbm")
FEATURE_PATH = Path("models/feature_columns.json")
DATA_PATH = Path("data/processed/cars_clean_v2.csv")
def clean_category(value):
    if pd.isna(value):
        return "Không rõ"
    text = " ".join(str(value).split())
    return text if text not in ["", "-", "--"] else "Không rõ"
def parse_number(value):
    text = value.strip().replace(",", ".")
    if not text:
        return np.nan
    try:
        number = float(text)
    except ValueError:
        raise ValueError("Vui lòng nhập một số hợp lệ.")
    if not np.isfinite(number):
        raise ValueError("Giá trị phải là một số hữu hạn.")
    return number
def parse_mileage(value):
    text = value.strip().lower()
    if not text:
        return np.nan
    text = re.sub(r"\s*km$", "", text).strip()
    if not re.fullmatch(r"\d[\d.,\s]*", text):
        raise ValueError("Số km không hợp lệ.")
    return float(re.sub(r"[.,\s]", "", text))
def parse_fuel_consumption(value):
    text = re.sub(r"\s+", "", value.strip().lower())
    if not text:
        return np.nan
    if text == "l/100km":
        return 1.0
    match = re.fullmatch(r"(\d+(?:[.,]\d+)?)l/100km", text)
    if match:
        return float(match.group(1).replace(",", "."))
    return parse_number(text)
def ask_number(label, minimum=None, maximum=None):
    while True:
        try:
            value = parse_number(input(f"{label}: "))
            if np.isnan(value):
                return np.nan
            if minimum is not None and value < minimum:
                raise ValueError(f"Giá trị phải lớn hơn hoặc bằng {minimum}.")
            if maximum is not None and value > maximum:
                raise ValueError(f"Giá trị phải nhỏ hơn hoặc bằng {maximum}.")
            return value
        except ValueError as error:
            print("Lỗi:", error)
def ask_mileage():
    while True:
        try:
            value = parse_mileage(input("Số km đã đi: "))
            if not np.isnan(value) and value < 0:
                raise ValueError("Số km không được âm.")
            return value
        except ValueError as error:
            print("Lỗi:", error)
def ask_fuel_consumption():
    while True:
        try:
            value = parse_fuel_consumption(input("Tiêu thụ nhiên liệu (L/100Km): "))
            if not np.isnan(value) and value < 0:
                raise ValueError("Mức tiêu thụ không được âm.")
            return value
        except ValueError as error:
            print("Lỗi:", error)
def ask_category(feature, label, dataset):
    print(f"\n========== {label.upper()} ==========")
    choices = sorted(dataset[feature].dropna().astype(str).unique().tolist(), key=str.casefold)
    print(f"Tổng số lựa chọn: {len(choices)}")
    for i, choice in enumerate(choices, start=1):
        print(f"{i}. {choice}")
    while True:
        value = input("\nChọn số thứ tự hoặc nhập tên: ").strip()
        if not value:
            return "Không rõ"
        if value.isdigit():
            index = int(value)
            if 1 <= index <= len(choices):
                return choices[index - 1]
            print("Số thứ tự không tồn tại.")
            continue
        matches = [choice for choice in choices if choice.casefold() == value.casefold()]
        if matches:
            return matches[0]
        print("Giá trị không có trong dataset. Vui lòng chọn lại.")
def load_model():
    if not MODEL_PATH.exists() or not FEATURE_PATH.exists():
        raise FileNotFoundError("Không tìm thấy model. Hãy chạy python train.py trước.")
    model = CatBoostRegressor()
    model.load_model(str(MODEL_PATH))
    with open(FEATURE_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    return model, config["features"], config["categorical_features"]
def input_car(dataset):
    car = {}
    print("\n========== THÔNG TIN XE ==========")
    car["brand"] = ask_category("brand", "Hãng xe", dataset)
    car["grade"] = ask_category("grade", "Mẫu xe / Grade", dataset)
    car["body_type"] = ask_category("body_type", "Dòng xe", dataset)
    car["origin"] = ask_category("origin", "Xuất xứ", dataset)
    car["transmission"] = ask_category("transmission", "Hộp số", dataset)
    car["drivetrain"] = ask_category("drivetrain", "Dẫn động", dataset)
    car["fuel_type"] = ask_category("fuel_type", "Loại nhiên liệu", dataset)
    print("\n========== THÔNG SỐ KỸ THUẬT ==========")
    car["engine_liters"] = ask_number("Dung tích động cơ (L)", minimum=0)
    car["doors"] = ask_number("Số cửa", minimum=1)
    car["seats"] = ask_number("Số chỗ ngồi", minimum=1)
    car["fuel_consumption"] = ask_fuel_consumption()
    car["year"] = ask_number("Năm sản xuất", minimum=1900)
    car["mileage"] = ask_mileage()
    car["mileage_is_zero"] = int(not np.isnan(car["mileage"]) and car["mileage"] == 0)
    car["mileage_missing"] = int(np.isnan(car["mileage"]))
    return car
def prepare_input(car, features, categorical_features):
    missing_features = [feature for feature in features if feature not in car]
    if missing_features:
        raise ValueError(f"Thiếu features: {missing_features}")
    input_df = pd.DataFrame([{feature: car[feature] for feature in features}], columns=features)
    for col in categorical_features:
        input_df[col] = input_df[col].fillna("Không rõ").astype(str)
    for col in features:
        if col not in categorical_features:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce")
    return input_df
def predict_price(model, input_df):
    prediction = float(model.predict(input_df)[0])
    return max(prediction, 0)
def main():
    model, features, categorical_features = load_model()
    if not DATA_PATH.exists():
        raise FileNotFoundError("Không tìm thấy cars_clean_v2.csv. Hãy chạy python preprocess.py trước.")
    dataset = pd.read_csv(DATA_PATH)
    missing_columns = [col for col in categorical_features if col not in dataset.columns]
    if missing_columns:
        raise ValueError(f"Dataset thiếu các cột: {missing_columns}")
    print("\n========== CAR PRICE AI ==========")
    print("Model: CatBoost Regressor")
    print("Trạng thái: Đã tải model thành công")
    print("Số bản ghi trong dataset:", len(dataset))
    print("Số hãng xe:", dataset["brand"].nunique())
    while True:
        car = input_car(dataset)
        input_df = prepare_input(car, features, categorical_features)
        predicted_price = predict_price(model, input_df)
        print("\n========== KẾT QUẢ ==========")
        print("Hãng xe:", car["brand"])
        print("Mẫu xe:", car["grade"])
        print(f"Giá dự đoán: {predicted_price:,.2f} triệu VNĐ")
        print(f"Tương đương: {predicted_price * 1_000_000:,.0f} VNĐ")
        print("=============================")
        again = input("\nBạn muốn dự đoán xe khác? (y/n): ").strip().lower()
        if again != "y":
            print("Kết thúc chương trình.")
            break
if __name__ == "__main__":
    main()