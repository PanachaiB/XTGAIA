import os
import sys
from pathlib import Path

current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from src.role2_ai.tgn_model import XTGAIATGN

def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.50) -> dict:
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        roc_auc = 0.5
    else:
        rank = np.argsort(y_prob)
        rank_sum = np.sum(np.where(y_true[rank] == 1)[0] + 1)
        roc_auc = float((rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "roc_auc": roc_auc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Starting XTGAIA TGN Training & Evaluation on device: {device}")

    sample_csv = "data/samples/cicids_sample.csv"
    if not os.path.exists(sample_csv):
        print(f"[!] File not found: {sample_csv}")
        return

    df = pd.read_csv(sample_csv)
    print(f"[*] Loaded {len(df)} flow records.")

    # 1. Temporal Coherence Split: 70% Train, 15% Val, 15% Test
    n_total = len(df)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train:n_train + n_val]
    test_df = df.iloc[n_train + n_val:]
    print(f"[*] Dataset Split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    # Tensors Train Set
    src_train = torch.tensor(train_df["src_ip"].values, dtype=torch.long, device=device)
    dst_train = torch.tensor(train_df["dst_ip"].values, dtype=torch.long, device=device)
    t_train = torch.tensor(train_df["timestamp"].values, dtype=torch.long, device=device)
    y_train = torch.tensor(train_df["label"].values, dtype=torch.float32, device=device).unsqueeze(1)

    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    msg_train = torch.tensor(train_df[feat_cols].values, dtype=torch.float32, device=device)

    # 2. Instantiating TGN Model
    num_nodes = max(df["src_ip"].max(), df["dst_ip"].max()) + 100
    model = XTGAIATGN(
        num_nodes=num_nodes,
        raw_msg_dim=len(feat_cols),
        memory_dim=128,
        latent_z_dim=8,
        num_heads=8
    ).to(device)

    # ใช้ Learning Rate 0.005 และ Weight Decay เล็กน้อยเพื่อช่วย Convergence
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    # ปรับ lr จาก 0.005 เป็น 0.0005 เพื่อป้องกัน Loss พุ่งกระฉูด
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
    criterion = nn.BCELoss()

    # 3. Training Loop (เพิ่มเป็น 10 Epochs)
    epochs = 10
    batch_size = 128
    num_batches = int(np.ceil(len(train_df) / batch_size))

    print(f"[*] Training TGN for {epochs} epochs (Batch size: {batch_size})...")
    model.train()
    for epoch in range(1, epochs + 1):
        model.reset_state()
        epoch_loss = 0.0
        for b in range(num_batches):
            idx = slice(b * batch_size, min((b + 1) * batch_size, len(train_df)))
            b_src = src_train[idx]
            b_dst = dst_train[idx]
            b_t = t_train[idx]
            b_msg = msg_train[idx]
            b_y = y_train[idx]

            optimizer.zero_grad()
            model.update_memory_events(b_src, b_dst, b_t, b_msg)
            p_t, _ = model(b_dst)
            loss = criterion(p_t, b_y)
            loss.backward()
            optimizer.step()
            model.memory.detach()
            epoch_loss += loss.item()

        print(f"    Epoch {epoch:02d}/{epochs:02d} - Train Loss: {epoch_loss / num_batches:.4f}")

    # 4. Calibration Phase: Finding Optimal Threshold (omega) on Validation Set
    print("\n[*] Calibrating Optimal Threshold on Validation Set...")
    model.eval()

    src_val = torch.tensor(val_df["src_ip"].values, dtype=torch.long, device=device)
    dst_val = torch.tensor(val_df["dst_ip"].values, dtype=torch.long, device=device)
    t_val = torch.tensor(val_df["timestamp"].values, dtype=torch.long, device=device)
    msg_val = torch.tensor(val_df[feat_cols].values, dtype=torch.float32, device=device)
    y_val = val_df["label"].values

    val_preds = []
    val_batches = int(np.ceil(len(val_df) / batch_size))
    with torch.no_grad():
        for b in range(val_batches):
            idx = slice(b * batch_size, min((b + 1) * batch_size, len(val_df)))
            model.update_memory_events(src_val[idx], dst_val[idx], t_val[idx], msg_val[idx])
            p_t, _ = model(dst_val[idx])
            val_preds.extend(p_t.cpu().numpy().flatten())

    y_val_probs = np.array(val_preds)

    best_threshold = 0.50
    best_val_f1 = -1.0

    # ค้นหา Threshold ที่สร้าง F1-Score สูงสุดบน Validation Set
    for cand_th in np.linspace(np.percentile(y_val_probs, 10), np.percentile(y_val_probs, 90), 100):
        m = compute_metrics(y_val, y_val_probs, threshold=cand_th)
        if m["f1"] > best_val_f1:
            best_val_f1 = m["f1"]
            best_threshold = cand_th

    print(f"[✓] Calibrated Optimal Threshold (ω): {best_threshold:.4f} (Val F1: {best_val_f1:.4f})")

    # 5. Evaluation Phase on Test Set
    print("\n[*] Evaluating on Test Set using Calibrated Threshold...")
    src_test = torch.tensor(test_df["src_ip"].values, dtype=torch.long, device=device)
    dst_test = torch.tensor(test_df["dst_ip"].values, dtype=torch.long, device=device)
    t_test = torch.tensor(test_df["timestamp"].values, dtype=torch.long, device=device)
    msg_test = torch.tensor(test_df[feat_cols].values, dtype=torch.float32, device=device)
    y_test = test_df["label"].values

    all_preds = []
    test_batches = int(np.ceil(len(test_df) / batch_size))

    with torch.no_grad():
        for b in range(test_batches):
            idx = slice(b * batch_size, min((b + 1) * batch_size, len(test_df)))
            model.update_memory_events(src_test[idx], dst_test[idx], t_test[idx], msg_test[idx])
            p_t, _ = model(dst_test[idx])
            all_preds.extend(p_t.cpu().numpy().flatten())

    y_pred_probs = np.array(all_preds)
    metrics = compute_metrics(y_test, y_pred_probs, threshold=best_threshold)

    # 6. Display Performance Summary
    print("\n" + "=" * 50)
    print("      XTGAIA TGN EFFICACY EVALUATION RESULTS      ")
    print("=" * 50)
    print(f"Total Test Samples:  {len(y_test)}")
    print(f"Applied Threshold:   {best_threshold:.4f}")
    print(f"Precision:           {metrics['precision']:.4f} ({metrics['precision'] * 100:.2f}%)")
    print(f"Recall (TPR):        {metrics['recall']:.4f} ({metrics['recall'] * 100:.2f}%)")
    print(f"F1-Score:            {metrics['f1']:.4f} ({metrics['f1'] * 100:.2f}%)")
    print(f"False Positive Rate: {metrics['fpr']:.4f} ({metrics['fpr'] * 100:.2f}%)")
    print(f"False Negative Rate: {metrics['fnr']:.4f} ({metrics['fnr'] * 100:.2f}%)")
    print(f"ROC-AUC:             {metrics['roc_auc']:.4f}")
    print("=" * 50)
    print("\nConfusion Matrix:")
    print(f"  TN: {metrics['tn']:<7} | FP: {metrics['fp']:<7}")
    print(f"  FN: {metrics['fn']:<7} | TP: {metrics['tp']:<7}")

    save_path = "data/models/tgn_best.pt"
    torch.save(model.state_dict(), save_path)
    print(f"\n[✓] Evaluated model saved to: {save_path}")

if __name__ == "__main__":
    train()