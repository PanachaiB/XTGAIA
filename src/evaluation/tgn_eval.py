"""
TGN (Role 2) under the held-out-day protocol on the corrected CICIDS2017.

Training data is prepared with Role 2's preprocessing method (benign fraction, per-label
attack cap, joint IP factorisation, t0-rebased seconds, z-score + clip on FLOW_FEATURES)
and trained with src/role2_ai/train_tgn.py unchanged (its 70/15/15 temporal split and
FPR-budgeted threshold calibration). Friday is then streamed through the live inference
engine (src/role2_ai/inference.py) in time order, exactly as the pipeline consumes it.

  python -m src.evaluation.tgn_eval --seed 0
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from src.evaluation.data import FLOW_FEATURE_MAP, load
from src.evaluation.metrics import detection_metrics
from src.role1_graph.flow_sources import FLOW_FEATURES

CLIP = 5.0
OUT_DIR = "data/processed/eval"


def build_training_table(train: pd.DataFrame, seed: int, benign_frac: float, attack_cap: int, out_csv: str,
                         meta_path: str) -> dict:
    rng = np.random.default_rng(seed)
    parts = []
    for day, g in train.groupby("day", sort=False):
        benign = g[g["y"] == 0]
        benign = benign[rng.random(len(benign)) < benign_frac]
        attacks = [a if len(a) <= attack_cap else a.sample(attack_cap, random_state=seed)
                   for _, a in g[g["y"] == 1].groupby("label")]
        parts.append(pd.concat([benign] + attacks))
    df = pd.concat(parts).sort_values("ts", kind="mergesort")
    cols = [f"f::{FLOW_FEATURE_MAP[c]}" for c in FLOW_FEATURES]
    feats = df[cols].astype(np.float64)
    ips = pd.unique(np.concatenate([df["src"].values, df["dst"].values]))
    ip_to_id = {ip: i for i, ip in enumerate(ips)}
    t0 = float(df["ts"].min())
    out = pd.DataFrame({"src_ip": df["src"].map(ip_to_id).values, "dst_ip": df["dst"].map(ip_to_id).values,
                        "timestamp": (df["ts"].values - t0).astype(np.int64), "label": df["y"].values})
    means, stds = feats.mean(), feats.std().fillna(0.0) + 1e-8
    for i, c in enumerate(cols):
        out[f"feat_{i}"] = np.clip((feats[c].values - means[c]) / stds[c], -CLIP, CLIP)
    out.to_csv(out_csv, index=False)
    names = [FLOW_FEATURE_MAP[c] for c in FLOW_FEATURES]
    meta = {"format": "xtgaia-cicids-meta-v1", "dataset": "CICIDS2017_improved (Engelen et al.)",
            "features": names, "means": {n: float(means[c]) for n, c in zip(names, cols)},
            "stds": {n: float(stds[c]) for n, c in zip(names, cols)}, "clip": CLIP, "t0": t0,
            "ip_to_id": ip_to_id, "num_known_nodes": len(ip_to_id), "rows": int(len(out)),
            "attack_rows": int(out["label"].sum()),
            "sampling": {"benign_frac": benign_frac, "attack_cap": attack_cap, "seed": seed}}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return meta


def evaluate_stream(ckpt: str, test: pd.DataFrame, batch: int = 128) -> dict:
    from src.role2_ai.inference import TGNInferenceEngine
    eng = TGNInferenceEngine(ckpt)
    cols = [f"f::{FLOW_FEATURE_MAP[c]}" for c in FLOW_FEATURES]
    X = test[cols].values.astype(np.float64)
    src, dst, port, ts = test["src"].values, test["dst"].values, test["dst_port"].values, test["ts"].values
    scores = []
    t_start = time.perf_counter()
    for b in range(0, len(test), batch):
        sl = slice(b, b + batch)
        edges = [{"source": s, "target": d, "port": int(p), "timestamp": float(t), "features": x.tolist()}
                 for s, d, p, t, x in zip(src[sl], dst[sl], port[sl], ts[sl], X[sl])]
        scores.extend(eng.observe_batch(edges))
    elapsed = time.perf_counter() - t_start
    m = detection_metrics(test["y"].values, np.array(scores), eng.threshold, test["label"].values)
    m["inference_us_per_flow"] = elapsed * 1e6 / max(len(test), 1)
    m["unseen_hosts"] = eng.unseen_hosts
    return m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--benign-frac", type=float, default=0.1)
    ap.add_argument("--attack-cap", type=int, default=20000)
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "results.jsonl"))
    a = ap.parse_args(argv)
    os.makedirs(OUT_DIR, exist_ok=True)
    from src.role2_ai.train_tgn import train

    t0 = time.time()
    df = load()
    train_df, test_df = df[df["day"] != "friday"], df[df["day"] == "friday"]
    csv = os.path.join(OUT_DIR, f"tgn_train_s{a.seed}.csv")
    meta_path = os.path.join(OUT_DIR, f"tgn_meta_s{a.seed}.json")
    ckpt = os.path.join(OUT_DIR, f"tgn_s{a.seed}.pt")
    meta = build_training_table(train_df, a.seed, a.benign_frac, a.attack_cap, csv, meta_path)
    print(f"[tgn s{a.seed}] training table {meta['rows']:,} rows ({meta['attack_rows']:,} attacks)", flush=True)
    internal = train(csv, meta_path, ckpt, epochs=a.epochs, seed=a.seed)
    m = evaluate_stream(ckpt, test_df)
    rec = {"model": "TGN (Role 2)", "protocol": "heldout", "seed": a.seed, "dataset": "CICIDS2017_improved",
           "train_rows": meta["rows"], "test_rows": int(len(test_df)), "wall_s": time.time() - t0,
           "internal_split_test_metrics": internal, **m}
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    print(f"[tgn s{a.seed}] Friday: P {m['precision']:.4f} R {m['recall']:.4f} F1 {m['f1']:.4f} AUC {m['roc_auc']:.4f} "
          f"TPR@1%FPR {m['tpr_at_fpr_1pct']:.4f} | per-class {m['per_class_recall']}", flush=True)


if __name__ == "__main__":
    main()
