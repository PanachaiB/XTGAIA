"""
Role 2 preprocessing: CICIDS2017 flows -> TGN training table + feature metadata.

Method (unchanged from the original script): binary labels (BENIGN = 0, attack = 1),
Z-score normalisation clipped to [-5, 5], chronological order.

Integration fixes:
  * Source and Destination IP are factorised *jointly*, so one host has one TGN
    node id whether it sends or receives (separate factorisation split every host
    into two unrelated memory slots).
  * Timestamps use the dataset's day-first format and the missing-PM fix
    (Role 1 parse_cicids_timestamp) and are rebased to t0 = first event, keeping
    TGN time encodings in a sane range.
  * The 16 edge features are the named FLOW_FEATURES contract shared with Role 1,
    not "the first 16 numeric columns" (which included Source Port).
  * Everything live inference needs to reproduce this exact transform is saved
    to a metadata JSON: ip_to_id, feature names, means/stds, clip, t0.

Usage
  python src/role2_ai/preprocess_cicids.py                      # legacy: data/samples/cicids2017_full.csv
  python src/role2_ai/preprocess_cicids.py --input data/raw/cicids2017/TrafficLabelling/Monday-*.csv ... \
         --benign-frac 0.1 --attack-cap 20000
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.role1_graph.flow_sources import FLOW_FEATURES, parse_cicids_timestamp  # noqa: E402

CLIP = 5.0


def load_flows(paths, benign_frac: float, attack_cap: int, seed: int, nrows=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for order, path in enumerate(paths):
        df = pd.read_csv(path, encoding="latin-1", low_memory=False, skipinitialspace=True, nrows=nrows)
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=["Source IP", "Destination IP"])
        df["label_name"] = df["Label"].astype(str).str.strip()
        df["label"] = (df["label_name"].str.upper() != "BENIGN").astype(int)
        df["ts"] = df["Timestamp"].map(parse_cicids_timestamp)
        df = df.dropna(subset=["ts"])
        df["file_order"] = order
        df["row_order"] = np.arange(len(df))
        benign = df[df["label"] == 0]
        if benign_frac < 1.0:
            benign = benign[rng.random(len(benign)) < benign_frac]
        attacks = []
        for _, grp in df[df["label"] == 1].groupby("label_name"):
            attacks.append(grp if len(grp) <= attack_cap else grp.sample(attack_cap, random_state=seed))
        kept = pd.concat([benign] + attacks)
        print(f"[*] {os.path.basename(path)}: {len(df)} flows -> kept {len(kept)} "
              f"(benign {len(benign)}, attack {int(kept['label'].sum())})")
        frames.append(kept)
    return pd.concat(frames, ignore_index=True)


def build(paths, out_csv: str, meta_path: str, benign_frac=1.0, attack_cap=10**9, seed=42, nrows=None) -> dict:
    df = load_flows(paths, benign_frac, attack_cap, seed, nrows)
    df = df.sort_values(["ts", "file_order", "row_order"], kind="mergesort")

    # 1. infinite / missing values (as before)
    feats = df[FLOW_FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 2. joint node ids
    ips = pd.unique(np.concatenate([df["Source IP"].astype(str).str.strip().values,
                                    df["Destination IP"].astype(str).str.strip().values]))
    ip_to_id = {ip: i for i, ip in enumerate(ips)}
    out = pd.DataFrame({
        "src_ip": df["Source IP"].astype(str).str.strip().map(ip_to_id).values,
        "dst_ip": df["Destination IP"].astype(str).str.strip().map(ip_to_id).values,
    })

    # 3. rebased integer timestamps (seconds since t0)
    t0 = float(df["ts"].min())
    out["timestamp"] = (df["ts"].values - t0).astype(np.int64)
    out["label"] = df["label"].values

    # 4. Z-score + clip (as before), statistics saved for inference
    means, stds = feats.mean(), feats.std().fillna(0.0) + 1e-8
    for i, col in enumerate(FLOW_FEATURES):
        out[f"feat_{i}"] = np.clip((feats[col].values - means[col]) / stds[col], -CLIP, CLIP)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out.to_csv(out_csv, index=False)
    meta = {
        "format": "xtgaia-cicids-meta-v1",
        "source_files": [os.path.basename(p) for p in paths],
        "features": FLOW_FEATURES,
        "means": {c: float(means[c]) for c in FLOW_FEATURES},
        "stds": {c: float(stds[c]) for c in FLOW_FEATURES},
        "clip": CLIP,
        "t0": t0,
        "ip_to_id": ip_to_id,
        "num_known_nodes": len(ip_to_id),
        "rows": int(len(out)),
        "attack_rows": int(out["label"].sum()),
        "label_counts": {k: int(v) for k, v in df["label_name"].value_counts().items()},
        "sampling": {"benign_frac": benign_frac, "attack_cap": attack_cap, "seed": seed},
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(f"[✓] Robust preprocessing finished! Saved {len(out)} records ({meta['attack_rows']} attacks, "
          f"{len(ip_to_id)} hosts) to: {out_csv}\n[✓] Feature metadata: {meta_path}")
    return meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", default=["data/samples/cicids2017_full.csv"],
                    help="CICIDS2017 TrafficLabelling CSV files or globs")
    ap.add_argument("--output", default="data/samples/cicids_sample.csv")
    ap.add_argument("--meta", default="data/models/tgn_feature_meta.json")
    ap.add_argument("--benign-frac", type=float, default=1.0, help="fraction of BENIGN flows kept per file")
    ap.add_argument("--attack-cap", type=int, default=10**9, help="max flows kept per attack label per file")
    ap.add_argument("--nrows", type=int, default=None, help="read only the first N rows of each file")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    paths = sorted({p for pattern in args.input for p in (glob.glob(pattern) or [pattern])})
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"[!] File not found: {', '.join(missing)}")
        sys.exit(1)
    build(paths, args.output, args.meta, args.benign_frac, args.attack_cap, args.seed, args.nrows)


if __name__ == "__main__":
    main()
