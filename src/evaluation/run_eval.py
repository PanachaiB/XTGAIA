"""
Run baselines under the XTGAIA evaluation protocol and append one JSON line per run.

  python -m src.evaluation.run_eval --models rf xgb egraphsage --protocols random heldout --seeds 0 1 2
"""
import argparse
import json
import os
import time

import numpy as np

from src.evaluation.baselines import fit_predict_egraphsage, fit_predict_rf, fit_predict_xgb
from src.evaluation.data import feature_columns, load, split, stratified_subsample
from src.evaluation.metrics import detection_metrics

OUT = "data/processed/eval/results.jsonl"
NAMES = {"rf": "Random Forest", "xgb": "XGBoost", "egraphsage": "E-GraphSAGE"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["rf", "xgb", "egraphsage"])
    ap.add_argument("--protocols", nargs="+", default=["random", "heldout"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--gnn-max-train-edges", type=int, default=600000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    t = time.time()
    df = load()
    cols = feature_columns(df)
    print(f"[eval] loaded {len(df):,} flows, {len(cols)} features, attacks {int(df['y'].sum()):,} "
          f"in {time.time() - t:.0f}s", flush=True)
    for protocol in a.protocols:
        for seed in a.seeds:
            tr, te = split(df, protocol, seed)
            for model in a.models:
                t = time.time()
                note = ""
                if model == "egraphsage":
                    tr_m = stratified_subsample(tr, a.gnn_max_train_edges, seed)
                    note = f"trained on stratified subsample of {len(tr_m):,}/{len(tr):,} edges"
                    p, fit_s, pred_s = fit_predict_egraphsage(tr_m, tr_m[cols].values, te, te[cols].values, seed,
                                                              threads=a.threads)
                elif model == "rf":
                    p, fit_s, pred_s = fit_predict_rf(tr[cols].values, tr["y"].values, te[cols].values, seed, a.threads)
                else:
                    p, fit_s, pred_s = fit_predict_xgb(tr[cols].values, tr["y"].values, te[cols].values, seed, a.threads)
                m = detection_metrics(te["y"].values, p, 0.5, te["label"].values)
                rec = {"model": NAMES[model], "protocol": protocol, "seed": seed, "dataset": "CICIDS2017_improved",
                       "train_rows": int(len(tr)), "test_rows": int(len(te)), "fit_s": fit_s,
                       "predict_us_per_flow": pred_s * 1e6 / len(te), "wall_s": time.time() - t, "note": note, **m}
                with open(a.out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, default=float) + "\n")
                print(f"[eval] {NAMES[model]:<14} {protocol:<8} s{seed}: P {m['precision']:.4f} R {m['recall']:.4f} "
                      f"F1 {m['f1']:.4f} AUC {m['roc_auc']:.4f} TPR@1% {m['tpr_at_fpr_1pct']:.4f} "
                      f"TPR@0.1% {m['tpr_at_fpr_0.1pct']:.4f} ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
