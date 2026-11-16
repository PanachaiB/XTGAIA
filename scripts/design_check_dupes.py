"""Design check follow-up: does the hybrid gain survive when duplicate flows are removed?

Re-scores saved VALIDATION scores (no test rows) on the validation flows whose exact feature vector never occurs in
train. Arms: node-only (current p_t, destination projection), the C3 flow head (3 seeds, saved scores) and the C2
MLP (3 seeds, retrained with the same seeds; its scores were not saved). Two threshold views:
  re-tuned   the shared threshold rule applied to the duplicate-free subset itself
  carried    ω tuned on the full validation slice, applied to the subset
Appends a section to runs/design_check/summary.md and an entry to docs/paper_deviations.md (B10).
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import design_check as dc  # noqa: E402
import design_check_c3 as c3  # noqa: E402
from src.role2_ai.train_tgn import calibrate_threshold  # noqa: E402

OUT = "runs/design_check/c3_override"
SUMMARY = "runs/design_check/summary.md"


def scores_view(y, s, mask):
    full_thr, _ = calibrate_threshold(y, s, max_fpr=0.01)
    sub = c3.view(y[mask], s[mask])
    carried = c3.full_metrics(y[mask], s[mask], full_thr)
    sub["boot_halfwidth"] = c3.bootstrap_halfwidth(y[mask], s[mask], sub["threshold"])
    return {"retuned": sub, "carried": carried}


def main():
    torch.set_num_threads(int(os.getenv("XTGAIA_TORCH_THREADS", "2") or 2))
    df, feat = dc.load_table()
    split = df["split"].values
    tr, va = np.flatnonzero(split == "train"), np.flatnonzero(split == "val")
    y = df["label"].values
    h = pd.util.hash_pandas_object(df[feat], index=False).values
    dup = np.isin(h[va], h[tr])
    keep = ~dup
    yv = y[va]
    info = {"val_rows": int(len(va)), "duplicate_rows": int(dup.sum()), "kept_rows": int(keep.sum()),
            "val_attacks": int(yv.sum()), "duplicate_attacks": int((dup & (yv == 1)).sum()),
            "kept_attacks": int((keep & (yv == 1)).sum()),
            "kept_attack_share": float(yv[keep].mean())}
    arms = {}
    p_dst = np.load(os.path.join(dc.CACHE, "p_dst.npy"))[va]
    arms["node-only"] = {"-": scores_view(yv, p_dst, keep)}
    arms["flow head (hybrid)"] = {}
    for seed in c3.SEEDS:
        s = np.load(os.path.join(OUT, f"flowhead_val_scores_s{seed}.npy"))
        arms["flow head (hybrid)"][str(seed)] = scores_view(yv, s, keep)
    arms["MLP (C2)"] = {}
    X = df[feat].values
    for seed in c3.SEEDS:
        s, _ = dc.train_head(X[tr], y[tr], X[va], yv, seed, subsample=1.0)
        arms["MLP (C2)"][str(seed)] = scores_view(yv, s, keep)

    def mean_f1(arm, view="retuned"):
        return float(np.mean([v[view]["f1"] for v in arms[arm].values()]))

    def noise(arm):
        f1 = [v["retuned"]["f1"] for v in arms[arm].values()]
        sd = float(np.std(f1, ddof=1)) if len(f1) > 1 else 0.0
        bh = float(np.mean([v["retuned"]["boot_halfwidth"] for v in arms[arm].values()]))
        return max(sd, bh, 0.005)

    gain = mean_f1("flow head (hybrid)") - mean_f1("node-only")
    band = 2 * max(noise("flow head (hybrid)"), noise("node-only"))
    res = {"time": time.strftime("%Y-%m-%d %H:%M"), "subset": info, "arms": arms,
           "gain_retuned": gain, "2x_noise": band, "gain_survives": bool(gain > band),
           "gain_carried": mean_f1("flow head (hybrid)", "carried") - mean_f1("node-only", "carried"),
           "head_minus_mlp_retuned": mean_f1("flow head (hybrid)") - mean_f1("MLP (C2)")}
    with open(os.path.join(OUT, "duplicates_check.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=float)

    L = ["", "### Duplicate-free re-scoring (validation only; flows whose feature vector never occurs in train)", "",
         f"Kept {info['kept_rows']:,} of {info['val_rows']:,} validation flows; removed {info['duplicate_rows']:,} "
         f"duplicates ({info['duplicate_attacks']:,} of {info['val_attacks']:,} attacks). Attack share after removal "
         f"{info['kept_attack_share']:.2%}. Flow labels.", "",
         "| Arm | Seed | F1 re-tuned | F1 carried ω | AUC | TPR@0.1/1/5 %FPR | Precision | MCC |", "|---|---|---|---|---|---|---|---|"]
    for arm, seeds in arms.items():
        for s, v in seeds.items():
            r = v["retuned"]
            L.append(f"| {arm} | {s} | {r['f1']:.4f} | {v['carried']['f1']:.4f} | {r['auc']:.4f} | "
                     f"{r['tpr_at_0.1pct_fpr']:.3f} / {r['tpr_at_1pct_fpr']:.3f} / {r['tpr_at_5pct_fpr']:.3f} | "
                     f"{r['precision']:.4f} | {r['mcc']:.4f} |")
    L += ["", f"Gain (flow head − node-only, re-tuned F1) {gain:+.4f} vs 2 × noise {band:.4f} → "
          f"**{'survives' if res['gain_survives'] else 'does not survive'}** duplicate removal (carried-ω gain "
          f"{res['gain_carried']:+.4f}). Flow head − MLP: {res['head_minus_mlp_retuned']:+.4f}.", ""]
    with open(SUMMARY, encoding="utf-8") as f:
        base = f.read().split("\n### Duplicate-free re-scoring")[0]
    with open(SUMMARY, "w", encoding="utf-8") as f:
        f.write(base.rstrip("\n") + "\n" + "\n".join(L))
    with open(dc.DEVIATIONS, "a", encoding="utf-8") as f:
        f.write(f"\n- **{res['time']} - Duplicate check after C3 (validation only).** Removing the "
                f"{info['duplicate_rows']:,} validation flows whose feature vector occurs in train "
                f"({info['duplicate_attacks']:,} of {info['val_attacks']:,} attacks): re-tuned F1 node-only "
                f"{mean_f1('node-only'):.4f}, flow head {mean_f1('flow head (hybrid)'):.4f}, MLP {mean_f1('MLP (C2)'):.4f}; "
                f"gain {gain:+.4f} vs 2 x noise {band:.4f}: the hybrid gain "
                f"{'survives' if res['gain_survives'] else 'does not survive'} duplicate removal. "
                f"Flow head - MLP {res['head_minus_mlp_retuned']:+.4f}.\n")
    print(json.dumps({k: res[k] for k in ("subset", "gain_retuned", "2x_noise", "gain_survives", "gain_carried",
                                          "head_minus_mlp_retuned")}, default=float), flush=True)


if __name__ == "__main__":
    main()
