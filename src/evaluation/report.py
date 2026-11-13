"""
Aggregate data/processed/eval/results.jsonl into docs/evaluation_results.md (mean +/- 95% CI over seeds).

  python -m src.evaluation.report
"""
import argparse
import collections
import datetime as dt
import json
import os

import numpy as np

from src.evaluation.metrics import summarize

KEYS = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("fpr", "FPR"), ("roc_auc", "ROC-AUC"),
        ("pr_auc", "PR-AUC"), ("tpr_at_fpr_1pct", "TPR@FPR≤1%"), ("tpr_at_fpr_0.1pct", "TPR@FPR≤0.1%")]
PROTOCOLS = {
    "random": "Random stratified 70/30 split over all five days (comparable with most CICIDS2017 papers; "
              "optimistic, temporally adjacent flows fall in both train and test)",
    "heldout": "Held-out day: train Monday–Thursday, test Friday (no temporal leakage; Botnet, PortScan, DDoS "
               "on the unseen day)",
}
ORDER = ["Random Forest", "XGBoost", "E-GraphSAGE", "TGN (Role 2)"]


def fmt(s):
    if s["ci95"] != s["ci95"]:          # NaN: single seed
        return f"{s['mean']:.4f}"
    return f"{s['mean']:.4f} ± {s['ci95']:.4f}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="data/processed/eval/results.jsonl")
    ap.add_argument("--out", default="docs/evaluation_results.md")
    a = ap.parse_args(argv)
    runs = [json.loads(l) for l in open(a.results, encoding="utf-8") if l.strip()]
    groups = collections.defaultdict(list)
    for r in runs:
        groups[(r["protocol"], r["model"])].append(r)

    lines = ["# XTGAIA evaluation results", "",
             f"Generated {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC from `{a.results}` "
             f"({len(runs)} runs). Values are mean ± 95% t-interval half-width over seeds (no interval when a single "
             "seed finished).", "",
             "**Dataset:** CICIDS2017 *improved* (Engelen, Rimmer & Joosen, WTMC 2021; Liu et al., CNS 2022): flows "
             "regenerated from the original PCAPs with a fixed CICFlowMeter and relabelled. Flows labelled "
             "*Attempted* (attack traffic without malicious effect) are treated as BENIGN, as the authors recommend. "
             "Identifiers (flow id, IPs, source port, timestamp) are excluded from flow-level features.", "",
             "**Thresholds:** Random Forest, XGBoost and E-GraphSAGE use 0.5. The TGN uses the threshold calibrated "
             "on its own validation split (max-F1 within a 1% validation FPR budget). ROC-AUC, PR-AUC and "
             "TPR@FPR are threshold-free.", ""]
    for proto, desc in PROTOCOLS.items():
        models = [m for m in ORDER if (proto, m) in groups] + \
                 sorted(m for (p, m) in groups if p == proto and m not in ORDER)
        if not models:
            continue
        lines += [f"## Protocol: {proto}", "", desc, "",
                  "| Model | Seeds | " + " | ".join(h for _, h in KEYS) + " |",
                  "|---|---|" + "---|" * len(KEYS)]
        for m in models:
            rs = groups[(proto, m)]
            s = summarize(rs, keys=[k for k, _ in KEYS])
            lines.append(f"| {m} | {len(rs)} | " + " | ".join(fmt(s[k]) for k, _ in KEYS) + " |")
        lines += ["", "Per-class detection rate at the model's threshold (mean over seeds) and benign flag rate:", "",
                  "| Model | " + " | ".join(sorted({c for m in models for r in groups[(proto, m)]
                                                     for c in r.get("per_class_recall", {})})) + " | Benign flagged |"]
        classes = sorted({c for m in models for r in groups[(proto, m)] for c in r.get("per_class_recall", {})})
        lines.append("|---|" + "---|" * (len(classes) + 1))
        for m in models:
            rs = groups[(proto, m)]
            cells = [f"{np.mean([r['per_class_recall'].get(c, float('nan')) for r in rs]):.3f}" for c in classes]
            lines.append(f"| {m} | " + " | ".join(cells) + f" | {np.mean([r['benign_flag_rate'] for r in rs]):.4f} |")
        notes = sorted({r.get("note") for m in models for r in groups[(proto, m)] if r.get("note")})
        sizes = {(r["train_rows"], r["test_rows"]) for m in models for r in groups[(proto, m)]}
        lines += ["", f"Train/test sizes: {', '.join(f'{a:,}/{b:,}' for a, b in sorted(sizes))}."]
        lines += [f"- {n}" for n in notes] + [""]
    lines += ["## Caveats", "",
              "- Random-split numbers are not comparable with held-out-day numbers; papers that report only random "
              "splits on CICIDS2017 are optimistic.",
              "- E-GraphSAGE trains on a stratified edge subsample for memory reasons (noted per run); it is evaluated "
              "on the full test set.",
              "- The TGN is trained on Role 2's subsampled training table (10% of benign flows, at most 20k flows per "
              "attack label and day) and evaluated by streaming all Friday flows through the live inference engine.",
              "- Cross-dataset evaluation (CICIDS2017 → CSE-CIC-IDS2018) is not included yet.", ""]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[report] {len(runs)} runs -> {a.out}")


if __name__ == "__main__":
    main()
