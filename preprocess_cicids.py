import os
import pandas as pd
import numpy as np

raw_path = "data/samples/cicids2017_full.csv"
target_path = "data/samples/cicids_sample.csv"

if not os.path.exists(raw_path):
    print(f"[!] File not found: {raw_path}")
    exit(1)

print("[*] Reading raw CIC-IDS dataset...")
df = pd.read_csv(raw_path, nrows=100000)
df.columns = df.columns.str.strip()

# 1. จัดการค่า Infinite และ Missing Values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.fillna(0, inplace=True)

# 2. แปลง Label ให้เป็น Binary (Benign = 0, Attack = 1)
label_col = [c for c in df.columns if c.lower() == "label"][0]
df["label"] = df[label_col].apply(lambda x: 0 if str(x).strip().upper() == "BENIGN" else 1)

# 3. กำหนด Node ID
if "Source IP" in df.columns and "Destination IP" in df.columns:
    df["src_ip"] = pd.factorize(df["Source IP"])[0]
    df["dst_ip"] = pd.factorize(df["Destination IP"])[0]
elif "Destination Port" in df.columns:
    df["dst_ip"] = pd.factorize(df["Destination Port"])[0] + 500
    df["src_ip"] = np.random.randint(1, 499, size=len(df))
else:
    df["src_ip"] = np.random.randint(1, 500, size=len(df))
    df["dst_ip"] = np.random.randint(501, 1000, size=len(df))

# 4. แปลง Timestamp
if "Timestamp" in df.columns:
    try:
        df["timestamp"] = pd.to_datetime(df["Timestamp"]).astype("int64") // 10**9
    except Exception:
        df["timestamp"] = np.arange(len(df))
else:
    df["timestamp"] = np.arange(len(df))

df.sort_values(by="timestamp", inplace=True)

# 5. คัดเลือก 16 คอลัมน์ตัวเลขและทํากระบวนการ Standard Normalization (Z-Score + Clip)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
excluded = ["src_ip", "dst_ip", "timestamp", "label"]
feature_cols = [c for c in numeric_cols if c not in excluded][:16]

for i, col in enumerate(feature_cols):
    mean_v = df[col].mean()
    std_v = df[col].std() + 1e-8
    # Z-score normalization & clip เพื่อไม่ให้เกิด Outliers พุ่งสูง
    scaled = (df[col] - mean_v) / std_v
    df[f"feat_{i}"] = np.clip(scaled, -5.0, 5.0)

cols_to_save = ["src_ip", "dst_ip", "timestamp", "label"] + [f"feat_{i}" for i in range(16)]
df[cols_to_save].to_csv(target_path, index=False)
print(f"[✓] Robust preprocessing finished! Saved {len(df)} records to: {target_path}")