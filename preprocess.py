import pandas as pd
import numpy as np
import re
from pathlib import Path
RAW_PATH = Path("data/raw/car_detail.csv")
OUTPUT_PATH = Path("data/processed/cars_clean_v2.csv")

def parse_price(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if not text or "liên hệ" in text:
        return np.na
    ty_match = re.search(r"(\d+(?:[.,]\d+)?)\s*tỷ", text)
    trieu_match = re.search(r"(\d+(?:[.,]\d+)?)\s*triệu", text)
    if not ty_match and not trieu_match:
        return np.nan
    ty = float(ty_match.group(1).replace(",", ".")) if ty_match else 0
    trieu = float(trieu_match.group(1).replace(",", ".")) if trieu_match else 0
    return ty * 1000 + trieu

def parse_mileage(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    match = re.fullmatch(r"(\d[\d.,\s]*)\s*km", text)
    if not match:
        return np.nan
    return float(re.sub(r"[.,\s]", "", match.group(1)))

def parse_fuel_type(value):
    if pd.isna(value):
        return "Không rõ"
    text = str(value).strip().lower()
    if "hybrid" in text:
        return "Hybrid"
    if "điện" in text:
        return "Điện"
    if "xăng" in text:
        return "Xăng"
    if "dầu" in text or "diesel" in text:
        return "Dầu"
    return "Không rõ"

def parse_engine_liters(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*l\b", text)
    return float(match.group(1).replace(",", ".")) if match else np.nan

def parse_count(value):
    if pd.isna(value):
        return np.nan
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else np.nan

def parse_fuel_consumption(value):
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    text = re.sub(r"\s+", "", text)
    if text == "l/100km":
        return 1.0
    match = re.fullmatch(r"(\d+(?:[.,]\d+)?)l/100km", text)
    return float(match.group(1).replace(",", ".")) if match else np.nan

def clean_category(value):
    if pd.isna(value):
        return "Không rõ"
    text = " ".join(str(value).split())
    return text if text not in ["", "-", "--"] else "Không rõ"

df = pd.read_csv(RAW_PATH)
print("Số dòng ban đầu:", len(df))
print("Số cột ban đầu:", len(df.columns))
df = df[df["Tình trạng"].astype(str).str.strip() == "Xe đã dùng"].copy()
print("Số xe đã sử dụng:", len(df))
df = df.drop_duplicates().copy()
print("Số dòng sau loại trùng:", len(df))
processed = pd.DataFrame(index=df.index)
processed["brand"] = df["Hãng"].apply(clean_category)
processed["grade"] = df["Grade"].apply(clean_category)
processed["body_type"] = df["Dòng xe"].apply(clean_category)
processed["origin"] = df["Xuất xứ"].apply(clean_category)
processed["transmission"] = df["Hộp số"].apply(clean_category)
processed["drivetrain"] = df["Dẫn động"].apply(clean_category)
processed["fuel_type"] = df["Động cơ"].apply(parse_fuel_type)
processed["engine_liters"] = df["Động cơ"].apply(parse_engine_liters)
processed["doors"] = df["Số cửa"].apply(parse_count)
processed["seats"] = df["Số chỗ ngồi"].apply(parse_count)
processed["fuel_consumption"] = df["Tiêu thụ nhiên liệu"].apply(parse_fuel_consumption)
processed["year"] = pd.to_numeric(df["Năm sản xuất"], errors="coerce")
processed["mileage"] = df["Số Km đã đi"].apply(parse_mileage)
processed["mileage_is_zero"] = (processed["mileage"] == 0).astype(int)
processed["price_million"] = df["Giá"].apply(parse_price)
before = len(processed)
processed = processed[processed["price_million"].notna() & (processed["price_million"] > 0)].copy()
print("Số dòng bị loại do giá không hợp lệ:", before - len(processed))
print("Kích thước dữ liệu sau xử lý:", processed.shape)
print("Danh sách cột:")
print(processed.columns.tolist())
print("Số giá trị thiếu:")
print(processed.isna().sum())
missing_percent = (processed.isnull().mean() * 100).round(2)
print("Tỷ lệ thiếu (%):")
print(missing_percent)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
processed.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print("Đã lưu dữ liệu tại:", OUTPUT_PATH)
print("Số features:", len(processed.columns) - 1)
print("Số target:", 1)