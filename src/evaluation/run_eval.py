"""
Run flow-level and graph baselines under the XTGAIA evaluation protocols; one JSON line per run.

  python -m src.evaluation.run_eval --models rf xgb mlp cnn1d egraphsage --protocols pcic_a chrono heldout --seeds 0 1 2 3 4

Protocols with a validation set (pcic_a, chrono, heldout) also report the CCSTGN [11] fixed-FPR
metrics: thresholds chosen on validation for FPR closest to 0.01 / 0.1 / 1 %, F1 and FNR on test.
"""
import argparse
import json
import os
import time

import numpy as np

from src.evaluation.baselines import (fit_predict_egraphsage, fit_predict_multiclass, fit_predict_nn, fit_predict_rf,
                                      fit_predict_xgb)
from src.evaluation.data import feature_columns, load_dataset, split, split3, stratified_subsample
from src.evaluation.metrics import detection_metrics, fixed_fpr_metrics, multiclass_metrics, weighted_threshold_metrics
from src.role2_ai.attack_classes import CLASSES, category_index

OUT = "data/processed/eval/results.jsonl"
NAMES = {"rf": "Random Forest", "xgb": "XGBoost", "mlp": "MLP", "cnn1d": "1D-CNN", "egraphsage": "E-GraphSAGE"}


def run_model(model, tr, va, te, cols, seed, a):
    Xtr, ytr, Xte = tr[cols].values, tr["y"].values, te[cols].values
    Xva = va[cols].values if va is not None else None
    yva = va["y"].values if va is not None else None
    if model == "egraphsage":
        tr_m = stratified_subsample(tr, a.gnn_max_train_edges, seed)
        p, pv, fit_s, pred_s = fit_predict_egraphsage(tr_m, tr_m[cols].values, te, Xte, seed, threads=a.threads,
                                                      va=va, Xva=Xva)
        return p, pv, fit_s, pred_s, f"trained on stratified subsample of {len(tr_m):,}/{len(tr):,} edges"
    if model == "rf":
        return (*fit_predict_rf(Xtr, ytr, Xte, seed, a.threads, Xva, yva), "")
    if model == "xgb":
        return (*fit_predict_xgb(Xtr, ytr, Xte, seed, a.threads, Xva, yva), "")
    return (*fit_predict_nn(model, Xtr, ytr, Xte, seed, a.threads, Xva, yva), "")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["rf", "xgb", "mlp", "cnn1d", "egraphsage"])
    ap.add_argument("--protocols", nargs="+", default=["pcic_a", "chrono", "heldout"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--gnn-max-train-edges", type=int, default=600000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dataset", choices=["cicids2017_improved", "nf_unsw_nb15_v2"], default="cicids2017_improved")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--multiclass", action="store_true",
                    help="also score attack categories (flat multi-class RF / XGBoost; paper Sec. IV.C)")
    a = ap.parse_args(argv)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    t = time.time()
    df = load_dataset(a.dataset)
    cols = feature_columns(df)
    print(f"[eval] loaded {len(df):,} flows, {len(cols)} features, attacks {int(df['y'].sum()):,} "
          f"in {time.time() - t:.0f}s", flush=True)
    for protocol in a.protocols:
        for seed in a.seeds:
            if protocol == "random":
                tr, te = split(df, protocol, seed)
                va = None
            else:
                tr, va, te = split3(df, protocol, seed)
            for model in a.models:
                t = time.time()
                p, pv, fit_s, pred_s, note = run_model(model, tr, va, te, cols, seed, a)
                m = detection_metrics(te["y"].values, p, 0.5, te["label"].values)
                m["accuracy"] = (m["tp"] + m["tn"]) / m["n"]
                rec = {"model": NAMES[model], "protocol": protocol, "seed": seed, "dataset": a.dataset,
                       "train_rows": int(len(tr)), "val_rows": int(len(va)) if va is not None else 0,
                       "test_rows": int(len(te)), "test_prevalence": float(te["y"].mean()),
                       "train_prevalence": float(tr["y"].mean()), "fit_s": fit_s,
                       "predict_us_per_flow": pred_s * 1e6 / len(te), "wall_s": time.time() - t, "note": note, **m}
                if pv is not None:
                    rec["fixed_fpr"] = fixed_fpr_metrics(va["y"].values, pv, te["y"].values, p)
                    rec["weighted_threshold"] = weighted_threshold_metrics(va["y"].values, pv, te["y"].values, p)
                if a.multiclass and model in ("rf", "xgb"):
                    cat = lambda d: np.array([0 if y == 0 else category_index(l)
                                              for y, l in zip(d["y"].values, d["label"].values)])
                    pred_c = fit_predict_multiclass(model, tr[cols].values, cat(tr), te[cols].values, seed, a.threads,
                                                    va[cols].values if va is not None else None,
                                                    cat(va) if va is not None else None)
                    rec["multiclass"] = multiclass_metrics(cat(te), pred_c, CLASSES)
                with open(a.out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, default=float) + "\n")
                ff = rec.get("fixed_fpr", {}).get("1%", {})
                print(f"[eval] {NAMES[model]:<13} {protocol:<8} s{seed}: F1 {m['f1']:.4f} AUC {m['roc_auc']:.4f} "
                      f"FPR {m['fpr']:.4f} | @FPR1%(val): F1 {ff.get('f1', float('nan')):.4f} "
                      f"FNR {ff.get('fnr', float('nan')):.4f} ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
