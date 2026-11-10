import os
import sys
from pathlib import Path

# เพิ่ม Root Path (/app หรือ Directory หลัก) เข้า sys.path เพื่อให้เห็นโฟลเดอร์ src
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]  # ชี้ไปที่โฟลเดอร์ root (/app)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# สามารถ import ได้อย่างถูกต้องทั้งใน Docker และ Local
from src.role2_ai.tgn_model import XTGAIATGN
# นำเข้าโมเดลสถาปัตยกรรม XTGAIA TGN
from src.role2_ai.tgn_model import XTGAIATGN

def generate_synthetic_flows(num_samples: int = 120, output_path: str = "data/samples/cicids_sample.csv"):
    """
    สร้างชุดข้อมูลจำลอง cicids_sample.csv อัตโนมัติหากยังไม่มีไฟล์จริงอยู่ในระบบ
    เพื่อให้ Docker สามารถรันเทรนผ่านได้ทันที
    """
    Path("data/samples").mkdir(parents=True, exist_ok=True)
    if os.path.exists(output_path):
        return

    print(f"[*] Generating fixture sample flows at {output_path}...")
    np.random.seed(42)
    src_ips = np.random.randint(1, 20, size=num_samples)
    dst_ips = np.random.randint(21, 50, size=num_samples)
    timestamps = np.sort(np.random.randint(1600000000, 1600010000, size=num_samples))
    labels = np.random.choice([0, 1], size=num_samples, p=[0.8, 0.2])
    
    df = pd.DataFrame({
        "src_ip": src_ips,
        "dst_ip": dst_ips,
        "timestamp": timestamps,
        "dst_port": np.random.choice([80, 443, 445, 22, 3389], size=num_samples),
        "protocol": np.random.choice([6, 17], size=num_samples),
        "flow_duration": np.random.exponential(scale=100, size=num_samples),
        "label": labels
    })
    
    # เพิ่ม Features ตัวเลข 16 มิติสำหรับ Raw Message Vector
    for i in range(16):
        df[f"feat_{i}"] = np.random.normal(0, 1, size=num_samples)
        
    df.to_csv(output_path, index=False)
    print(f"[+] Sample flows fixture generated ({num_samples} flows).")


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Starting XTGAIA TGN Training on device: {device}")

    sample_csv = "data/samples/cicids_sample.csv"
    if not os.path.exists(sample_csv):
        generate_synthetic_flows(output_path=sample_csv)

    # 1. โหลดข้อมูล Flow
    df = pd.read_csv(sample_csv)
    print(f"[*] Loaded {len(df)} flow records.")

    # เตรียม Tensors สำหรับ Temporal Graph
    src = torch.tensor(df["src_ip"].values, dtype=torch.long, device=device)
    dst = torch.tensor(df["dst_ip"].values, dtype=torch.long, device=device)
    t = torch.tensor(df["timestamp"].values, dtype=torch.long, device=device)
    labels = torch.tensor(df["label"].values, dtype=torch.float32, device=device).unsqueeze(1)

    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    if len(feat_cols) == 16:
        raw_msg = torch.tensor(df[feat_cols].values, dtype=torch.float32, device=device)
    else:
        raw_msg = torch.randn(len(df), 16, device=device)

    # 2. สร้างอินสแตนซ์ของโมเดล XTGAIA TGN
    num_nodes = max(src.max().item(), dst.max().item()) + 100
    model = XTGAIATGN(
        num_nodes=num_nodes,
        raw_msg_dim=16,
        memory_dim=128,
        latent_z_dim=8,
        num_heads=8
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    # 3. Training Loop
    epochs = 5
    batch_size = 32
    num_batches = int(np.ceil(len(df) / batch_size))

    print(f"[*] Training TGN for {epochs} epochs (Batch size: {batch_size})...")
    model.train()

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.reset_state()
        epoch_loss = 0.0

        for b in range(num_batches):
            idx = slice(b * batch_size, min((b + 1) * batch_size, len(df)))
            b_src = src[idx]
            b_dst = dst[idx]
            b_t = t[idx]
            b_msg = raw_msg[idx]
            b_y = labels[idx]

            optimizer.zero_grad()

            # อัปเดต Memory ของ TGN
            model.update_memory_events(b_src, b_dst, b_t, b_msg)

            # คำนวณ threat confidence
            p_t, _ = model(b_dst)

            loss = criterion(p_t, b_y)
            loss.backward()
            optimizer.step()

            # Detach memory เพื่อป้องกัน memory leak ใน temporal stream
            model.memory.detach()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_batches
        print(f"    Epoch {epoch:02d}/{epochs:02d} - Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss

    # 4. บันทึกค่าน้ำหนักโมเดลสำหรับส่งมอบให้ทีม
    Path("data/models").mkdir(parents=True, exist_ok=True)
    save_path = "data/models/tgn_best.pt"
    torch.save(model.state_dict(), save_path)
    print(f"[✓] TGN model successfully trained and saved to: {save_path}")


if __name__ == "__main__":
    train()