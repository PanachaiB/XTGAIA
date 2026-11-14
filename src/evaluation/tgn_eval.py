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

from src.evaluation.data import FLOW_FEATURE_MAP, feature_columns, load_dataset
from src.evaluation.metrics import (detection_metrics, fixed_fpr_metrics, hierarchical_predictions,
                                    multiclass_metrics, weighted_threshold_metrics)
from src.role1_graph.flow_sources import FLOW_FEATURES

CLIP = 5.0
OUT_DIR = "data/processed/eval"


def tgn_feature_columns(df: pd.DataFrame, dataset: str):
    """CICIDS2017: Role 1's 16 FLOW_FEATURES; NF-UNSW-NB15-v2: all NetFlow v2 numeric features."""
    if dataset == "cicids2017_improved":
        return [f"f::{FLOW_FEATURE_MAP[c]}" for c in FLOW_FEATURES], [FLOW_FEATURE_MAP[c] for c in FLOW_FEATURES]
    cols = feature_columns(df)
    return cols, [c[3:] for c in cols]


def build_training_table(train: pd.DataFrame, seed: int, benign_frac: float, attack_cap: int, out_csv: str,
                         meta_path: str, per_day_split: bool = False, log_transform: bool = False,
                         dataset: str = "cicids2017_improved", eval_rows: pd.DataFrame = None) -> dict:
    """Role 2 training table. `eval_rows` (chronological protocols) are appended unsampled with their
    split tag, so validation/test keep the real distribution and stream after the training rows."""
    rng = np.random.default_rng(seed)
    parts = []
    for day, g in train.groupby("day", sort=False):
        benign = g[g["y"] == 0]
        benign = benign[rng.random(len(benign)) < benign_frac]
        attacks = [a if len(a) <= attack_cap else a.sample(attack_cap, random_state=seed)
                   for _, a in g[g["y"] == 1].groupby("label")]
        parts.append(pd.concat([benign] + attacks))
    df = pd.concat(parts)
    if eval_rows is not None:
        df = df.assign(_split="train")
        df = pd.concat([df, eval_rows])
    df = df.sort_values("ts", kind="mergesort")
    cols, names = tgn_feature_columns(df, dataset)
    feats = df[cols].astype(np.float64)
    if log_transform:                                   # signed log1p for heavy-tailed flow statistics
        feats = np.sign(feats) * np.log1p(np.abs(feats))
    ips = pd.unique(np.concatenate([df["src"].values, df["dst"].values]))
    ip_to_id = {ip: i for i, ip in enumerate(ips)}
    t0 = float(df["ts"].min())
    from src.role2_ai.attack_classes import CLASSES, category_index
    out = pd.DataFrame({"src_ip": df["src"].map(ip_to_id).values, "dst_ip": df["dst"].map(ip_to_id).values,
                        "timestamp": (df["ts"].values - t0).astype(np.int64), "label": df["y"].values,
                        "category": [0 if y == 0 else category_index(l) for y, l in zip(df["y"].values,
                                                                                          df["label"].values)]})
    if eval_rows is not None:
        out["split"] = df["_split"].values
    elif per_day_split:                                 # 70/15/15 temporal split inside every day
        split = np.empty(len(df), dtype=object)
        for _, idx in pd.Series(np.arange(len(df))).groupby(df["day"].values).groups.items():
            idx = np.sort(np.asarray(idx))
            n = len(idx)
            split[idx[: int(n * 0.70)]] = "train"
            split[idx[int(n * 0.70): int(n * 0.85)]] = "val"
            split[idx[int(n * 0.85):]] = "test"
        out["split"] = split
    means, stds = feats.mean(), feats.std().fillna(0.0) + 1e-8
    for i, c in enumerate(cols):
        out[f"feat_{i}"] = np.clip((feats[c].values - means[c]) / stds[c], -CLIP, CLIP)
    out.to_csv(out_csv, index=False)
    meta = {"format": "xtgaia-cicids-meta-v1", "dataset": dataset,
            "features": names, "means": {n: float(means[c]) for n, c in zip(names, cols)},
            "stds": {n: float(stds[c]) for n, c in zip(names, cols)}, "clip": CLIP, "t0": t0,
            "ip_to_id": ip_to_id, "num_known_nodes": len(ip_to_id), "rows": int(len(out)),
            "transform": "signed_log1p" if log_transform else None,
            "attack_rows": int(out["label"].sum()), "classes": CLASSES,
            "sampling": {"benign_frac": benign_frac, "attack_cap": attack_cap, "seed": seed}}
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return meta


def evaluate_stream(ckpt: str, test: pd.DataFrame, batch: int = 128, dataset: str = "cicids2017_improved"):
    from src.role2_ai.inference import TGNInferenceEngine
    eng = TGNInferenceEngine(ckpt)
    cols, _ = tgn_feature_columns(test, dataset)
    X = test[cols].values.astype(np.float64)
    src, dst, port, ts = test["src"].values, test["dst"].values, test["dst_port"].values, test["ts"].values
    scores, cls = [], []
    t_start = time.perf_counter()
    for b in range(0, len(test), batch):
        sl = slice(b, b + batch)
        edges = [{"source": s, "target": d, "port": int(p), "timestamp": float(t), "features": x.tolist()}
                 for s, d, p, t, x in zip(src[sl], dst[sl], port[sl], ts[sl], X[sl])]
        scores.extend(eng.observe_batch(edges))
        if eng.last_category_probs is not None:
            cls.append(eng.last_category_probs)
    elapsed = time.perf_counter() - t_start
    m = detection_metrics(test["y"].values, np.array(scores), eng.threshold, test["label"].values)
    m["inference_us_per_flow"] = elapsed * 1e6 / max(len(test), 1)
    m["unseen_hosts"] = eng.unseen_hosts
    if cls:
        from src.role2_ai.attack_classes import CLASSES, category_index
        y_cat = np.array([0 if y == 0 else category_index(l) for y, l in zip(test["y"].values, test["label"].values)])
        m["multiclass"] = multiclass_metrics(y_cat, hierarchical_predictions(scores, eng.threshold,
                                                                             np.concatenate(cls)), CLASSES)
    return m, np.array(scores)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--benign-frac", type=float, default=0.1)
    ap.add_argument("--attack-cap", type=int, default=20000)
    ap.add_argument("--procedure", choices=["original", "streaming"], default="original",
                    help="original: Role 2 train(); streaming: per-day split, class weights, early stopping")
    ap.add_argument("--dataset", choices=["cicids2017_improved", "nf_unsw_nb15_v2"], default="cicids2017_improved")
    ap.add_argument("--pretrain-2018", action="store_true",
                    help="paper Stage 3: pre-train on CSE-CIC-IDS2018 (improved), then train on CIC-IDS2017")
    ap.add_argument("--pretrain-epochs", type=int, default=3)
    ap.add_argument("--protocol", choices=["heldout", "chrono"], default="heldout")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "results.jsonl"))
    a = ap.parse_args(argv)
    os.makedirs(OUT_DIR, exist_ok=True)
    from src.role2_ai.train_tgn import train, train_streaming
    streaming = a.procedure == "streaming"
    tag = "v2" if streaming else ""

    t0 = time.time()
    init_from = None
    if a.pretrain_2018:
        if a.dataset != "cicids2017_improved":
            raise SystemExit("--pretrain-2018 applies to CIC-IDS2017 (same feature schema)")
        init_from = os.path.join(OUT_DIR, "tgn_pretrain_2018.pt")
        if not os.path.exists(init_from):              # built once, shared by all seeds
            from src.evaluation.data import load_cicids2018_sampled
            d18 = load_cicids2018_sampled()
            csv18, meta18 = os.path.join(OUT_DIR, "tgn_2018_table.csv"), os.path.join(OUT_DIR, "tgn_2018_meta.json")
            m18 = build_training_table(d18, 0, 1.0, 10**9, csv18, meta18, per_day_split=True, log_transform=True)
            print(f"[pretrain 2018] table {m18['rows']:,} rows ({m18['attack_rows']:,} attacks)", flush=True)
            train_streaming(csv18, meta18, init_from, max_epochs=a.pretrain_epochs, seed=0, fit_explainer=False)
    df = load_dataset(a.dataset)
    key = f"{a.dataset[:5]}_{a.protocol}{'_pt18' if a.pretrain_2018 else ''}"
    csv = os.path.join(OUT_DIR, f"tgn{tag}_{key}_train_s{a.seed}.csv")
    meta_path = os.path.join(OUT_DIR, f"tgn{tag}_{key}_meta_s{a.seed}.json")
    ckpt = os.path.join(OUT_DIR, f"tgn{tag}_{key}_s{a.seed}.pt")
    name = ("TGN (XTGAIA, temporal attention" + (", 2017+2018 pre-training)" if a.pretrain_2018 else ")")) \
        if streaming else "TGN (Role 2)"
    if a.protocol == "chrono":
        if not streaming:
            raise SystemExit("--protocol chrono needs --procedure streaming")
        n = len(df)
        tr, va, te = df.iloc[: int(n * 0.70)], df.iloc[int(n * 0.70): int(n * 0.80)], df.iloc[int(n * 0.80):]
        evals = pd.concat([va.assign(_split="val"), te.assign(_split="test")])
        meta = build_training_table(tr, a.seed, a.benign_frac, a.attack_cap, csv, meta_path, log_transform=True,
                                    dataset=a.dataset, eval_rows=evals)
        print(f"[tgn{tag} {key} s{a.seed}] table {meta['rows']:,} rows ({meta['attack_rows']:,} attacks)", flush=True)
        internal, pr = train_streaming(csv, meta_path, ckpt, max_epochs=a.epochs, seed=a.seed, return_predictions=True,
                                       init_from=init_from)
        vm, tm = pr["split"] == "val", pr["split"] == "test"
        m = detection_metrics(pr["y"][tm], pr["preds"][tm], pr["threshold"])
        m["fixed_fpr"] = fixed_fpr_metrics(pr["y"][vm], pr["preds"][vm], pr["y"][tm], pr["preds"][tm])
        m["weighted_threshold"] = weighted_threshold_metrics(pr["y"][vm], pr["preds"][vm], pr["y"][tm], pr["preds"][tm])
        if pr.get("cls_probs") is not None:
            from src.role2_ai.attack_classes import CLASSES
            m["multiclass"] = multiclass_metrics(pr["category"][tm], hierarchical_predictions(
                pr["preds"][tm], pr["threshold"], pr["cls_probs"][tm]), CLASSES)
        test_rows, prevalence = int(tm.sum()), float(pr["y"][tm].mean())
    else:
        train_df, test_df = df[df["day"] != "friday"], df[df["day"] == "friday"]
        meta = build_training_table(train_df, a.seed, a.benign_frac, a.attack_cap, csv, meta_path,
                                    per_day_split=streaming, log_transform=streaming, dataset=a.dataset)
        print(f"[tgn{tag} s{a.seed}] training table {meta['rows']:,} rows ({meta['attack_rows']:,} attacks)", flush=True)
        if streaming:
            internal, pr = train_streaming(csv, meta_path, ckpt, max_epochs=a.epochs, seed=a.seed,
                                           return_predictions=True, init_from=init_from)
        else:
            internal, pr = train(csv, meta_path, ckpt, epochs=a.epochs, seed=a.seed), None
        m, scores = evaluate_stream(ckpt, test_df, dataset=a.dataset)
        if pr is not None:              # thresholds from the per-day validation rows, applied to Friday
            vm = pr["split"] == "val"
            m["fixed_fpr"] = fixed_fpr_metrics(pr["y"][vm], pr["preds"][vm], test_df["y"].values, scores)
            m["weighted_threshold"] = weighted_threshold_metrics(pr["y"][vm], pr["preds"][vm],
                                                                 test_df["y"].values, scores)
        test_rows, prevalence = int(len(test_df)), float(test_df["y"].mean())
    rec = {"model": name, "protocol": a.protocol, "seed": a.seed, "dataset": a.dataset,
           "train_rows": meta["rows"], "test_rows": test_rows, "test_prevalence": prevalence,
           "wall_s": time.time() - t0, "internal_split_test_metrics": internal, **m}
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    ff = m.get("fixed_fpr", {}).get("1%", {})
    print(f"[tgn{tag} {key} s{a.seed}] test: F1 {m['f1']:.4f} AUC {m['roc_auc']:.4f} TPR@1%FPR {m['tpr_at_fpr_1pct']:.4f}"
          f" | @FPR1%(val) F1 {ff.get('f1', float('nan')):.4f} FNR {ff.get('fnr', float('nan')):.4f}"
          f" | multi-class macro-F1 {m.get('multiclass', {}).get('macro_f1', float('nan')):.4f}"
          f" | per-class {m.get('per_class_recall', {})}", flush=True)


if __name__ == "__main__":
    main()
