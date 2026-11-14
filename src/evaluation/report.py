"""
Aggregate data/processed/eval/results.jsonl into a Markdown report (mean ± 95% CI over seeds).

Published baseline numbers are shown only where the comparison is DIRECT (same dataset, task and
reproducible protocol): Vedesh et al. [3] next to P-CIC split A on CICIDS2017, CCSTGN [11] next to
the chronological protocol on NF-UNSW-NB15-v2. Other papers are footnoted (CONTEXT-ONLY).

  python -m src.evaluation.report --out data/processed/eval/evaluation_results.md
"""
import argparse
import collections
import datetime as dt
import json
import os

import numpy as np

from src.evaluation.metrics import summarize

KEYS = [("accuracy", "Acc"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("fpr", "FPR"),
        ("roc_auc", "ROC-AUC"), ("tpr_at_fpr_1pct", "TPR@FPR≤1%"), ("tpr_at_fpr_0.1pct", "TPR@FPR≤0.1%")]
SECTIONS = [
    ("cicids2017_improved", "pcic_a", "CICIDS2017 (corrected), P-CIC split A — random 80/20, benign downsampled in "
     "training only, 10% validation (parity protocol for Vedesh et al. [3])"),
    ("cicids2017_improved", "chrono", "CICIDS2017 (corrected), P-CIC split B — chronological 70/10/20 (CCSTGN [11] "
     "protocol applied to CICIDS2017)"),
    ("cicids2017_improved", "heldout", "CICIDS2017 (corrected), held-out day — train Mon–Thu (last 15% of each day "
     "= validation), test Friday"),
    ("cicids2017_improved", "random", "CICIDS2017 (corrected), legacy random 70/30 (no validation set)"),
    ("nf_unsw_nb15_v2", "chrono", "NF-UNSW-NB15-v2 — chronological 70/10/20, flow index as time (exact CCSTGN [11] "
     "protocol)"),
]
ORDER = ["Random Forest", "XGBoost", "MLP", "1D-CNN", "E-GraphSAGE", "TGN (Role 2)",
         "TGN (Role 2, stage A: training procedure)", "TGN (XTGAIA, temporal attention)",
         "TGN (XTGAIA, temporal attention, 2017+2018 pre-training)"]
PUBLISHED = {
    ("cicids2017_improved", "pcic_a"): {
        "source": "Vedesh K.P. et al. [3], ICSADL 2026 — CICIDS2017 Kaggle repack (~100k flows), random 80/20, "
                  "single run, test prevalence ≈ 43–45 % (derived). Tag: DIRECT after P-CIC (release differs).",
        "rows": [("RF [3]", {"accuracy": 0.9456, "precision": 0.9412, "recall": 0.9502, "f1": 0.9457, "fpr": 0.0488}),
                 ("XGBoost [3]", {"accuracy": 0.9523, "precision": 0.9501, "recall": 0.9546, "f1": 0.9523, "fpr": 0.0402}),
                 ("1D-CNN [3]", {"accuracy": 0.9634, "precision": 0.9601, "recall": 0.9667, "f1": 0.9634, "fpr": 0.0301})]},
    ("nf_unsw_nb15_v2", "chrono"): {
        "source": "Escriche et al. (CCSTGN) [11], CNS 2024 — NF-UNSW-NB15-v2, chronological 70/10/20, 5 runs. "
                  "Tag: DIRECT (same dataset and protocol; our time = flow index as in [11]).",
        "table1": [("MLP [11]", 0.99915, 0.95224, 0.00785, 0.00046), ("CCSTGN-rs [11]", 0.99881, 0.94618, 0.00894, 0.00022),
                   ("CCSTGN-iso [11]", 0.99886, 0.94969, 0.00829, 0.00041),
                   ("CCSTGN-rs (ft) [11]", 0.99884, 0.95037, 0.00816, 0.00057),
                   ("CCSTGN-iso (ft) [11]", 0.99882, 0.95058, 0.00796, 0.00251),
                   ("E-GraphSAGE [11]†", 0.9039, 0.9410, None, None), ("TGN [11]†", 0.8801, 0.9355, None, None),
                   ("3D-IDS [11]†", 0.9155, 0.9545, None, None)],
        "table2": [("MLP [11]", (0.65829, 0.50770), (0.82295, 0.29501), (0.95159, 0.00038)),
                   ("CCSTGN-rs [11]", (0.50331, 0.65988), (0.77020, 0.36579), (0.93667, 0.00007)),
                   ("CCSTGN-iso [11]", (0.22321, 0.87036), (0.60763, 0.56106), (0.94773, 0.00061)),
                   ("CCSTGN-rs (ft) [11]", (0.49784, 0.66585), (0.70193, 0.45329), (0.95029, 0.00080)),
                   ("CCSTGN-iso (ft) [11]", (0.46941, 0.68555), (0.70124, 0.44946), (0.94934, 0.00044))]},
}


def fmt(s):
    if not s or s.get("mean") is None or s["mean"] != s["mean"]:
        return "–"
    return f"{s['mean']:.4f}" if s["ci95"] != s["ci95"] else f"{s['mean']:.4f} ± {s['ci95']:.4f}"


def agg(rs, getter):
    v = np.array([getter(r) for r in rs if getter(r) is not None], dtype=float)
    if not len(v):
        return None
    from scipy import stats
    half = float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
    return {"mean": float(v.mean()), "ci95": half}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="data/processed/eval/results.jsonl")
    ap.add_argument("--out", default="docs/evaluation_results.md")
    a = ap.parse_args(argv)
    runs = [json.loads(l) for l in open(a.results, encoding="utf-8") if l.strip()]
    for r in runs:
        r.setdefault("dataset", "cicids2017_improved")
        r["dataset"] = {"CICIDS2017_improved": "cicids2017_improved"}.get(r["dataset"], r["dataset"])
        if "accuracy" not in r and r.get("n"):
            r["accuracy"] = (r["tp"] + r["tn"]) / r["n"]
    groups = collections.defaultdict(list)
    for r in runs:
        groups[(r["dataset"], r["protocol"], r["model"])].append(r)

    L = ["# XTGAIA detection evaluation", "",
         f"Generated {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC from `{a.results}` ({len(runs)} "
         "runs). Mean ± 95% t-interval half-width over seeds (no interval for a single seed). Default-threshold "
         "metrics use 0.5 for baselines and the validation-calibrated ω for the TGN; *fixed-FPR* and "
         "*weighted-threshold* tables choose every model's threshold on its validation set as in CCSTGN [11]. "
         "Precision, accuracy and F1 depend on test prevalence (given per section); TPR/FNR at fixed FPR and "
         "ROC-AUC do not.", "",
         "**Datasets.** CICIDS2017 *improved* (Engelen et al. 2021 / Liu et al. 2022; *Attempted* → BENIGN). "
         "NF-UNSW-NB15-v2 (Sarhan et al.; 2,390,275 flows, 3.98 % attack).", ""]
    for ds, proto, desc in SECTIONS:
        models = [m for m in ORDER if (ds, proto, m) in groups] + \
                 sorted(m for (d, p, m) in groups if d == ds and p == proto and m not in ORDER)
        if not models:
            continue
        prev = sorted({round(r.get("test_prevalence", float("nan")), 4) for m in models for r in groups[(ds, proto, m)]
                       if r.get("test_prevalence") is not None})
        L += [f"## {desc}", "", f"Test prevalence (attack share): {', '.join(f'{p:.2%}' for p in prev) or 'n/a'}.", "",
              "| Model | Seeds | " + " | ".join(h for _, h in KEYS) + " |", "|---|---|" + "---|" * len(KEYS)]
        for m in models:
            rs = groups[(ds, proto, m)]
            L.append(f"| {m} | {len(rs)} | " + " | ".join(fmt(agg(rs, lambda r, k=k: r.get(k))) for k, _ in KEYS) + " |")
        pub = PUBLISHED.get((ds, proto))
        if pub and "rows" in pub:
            for name, vals in pub["rows"]:
                L.append(f"| *{name}* | 1 | " + " | ".join(f"{vals[k]:.4f}" if k in vals else "–" for k, _ in KEYS) + " |")
            L += ["", f"*Reference rows:* {pub['source']}"]
        if any(r.get("fixed_fpr") for m in models for r in groups[(ds, proto, m)]):
            L += ["", "**Missed detections at fixed FPR** (threshold chosen on validation for FPR closest to the target; "
                  "F1 / FNR on test, as in CCSTGN [11] Table II):", "",
                  "| Model | F1 @0.01% | FNR @0.01% | F1 @0.1% | FNR @0.1% | F1 @1% | FNR @1% | test FPR @1% |",
                  "|---|---|---|---|---|---|---|---|"]
            for m in models:
                rs = [r for r in groups[(ds, proto, m)] if r.get("fixed_fpr")]
                if not rs:
                    continue
                cells = []
                for t in ("0.01%", "0.1%", "1%"):
                    cells += [fmt(agg(rs, lambda r, t=t: r["fixed_fpr"][t]["f1"])),
                              fmt(agg(rs, lambda r, t=t: r["fixed_fpr"][t]["fnr"]))]
                cells.append(fmt(agg(rs, lambda r: r["fixed_fpr"]["1%"]["fpr"])))
                L.append(f"| {m} | " + " | ".join(cells) + " |")
            if pub and "table2" in pub:
                for name, *vals in pub["table2"]:
                    L.append(f"| *{name}* | " + " | ".join(f"{f:.4f} | {n:.4f}" for f, n in vals) + " | – |")
            L += ["", "**Weighted threshold** (ω minimising 0.75·FNR + 0.25·FPR on validation, CCSTGN [11] Table I):", "",
                  "| Model | ROC-AUC | F1 | FPR | FNR |", "|---|---|---|---|---|"]
            for m in models:
                rs = [r for r in groups[(ds, proto, m)] if r.get("weighted_threshold")]
                if rs:
                    L.append(f"| {m} | " + " | ".join(fmt(agg(rs, lambda r, k=k: r["weighted_threshold"][k]))
                                                      for k in ("roc_auc", "f1", "fpr", "fnr")) + " |")
            if pub and "table1" in pub:
                for name, auc, f1, fpr, fnr in pub["table1"]:
                    L.append(f"| *{name}* | {auc:.4f} | {f1:.4f} | {'–' if fpr is None else f'{fpr:.4f}'} | "
                             f"{'–' if fnr is None else f'{fnr:.4f}'} |")
                L += ["", f"*Reference rows:* {pub['source']} † copied in [11] from the 3D-IDS paper (no FPR/FNR)."]
        classes = sorted({c for m in models for r in groups[(ds, proto, m)] for c in r.get("per_class_recall", {})})
        if classes:
            L += ["", "Per-class detection rate at the default threshold (mean over seeds):", "",
                  "| Model | " + " | ".join(classes) + " | Benign flagged |", "|---|" + "---|" * (len(classes) + 1)]
            for m in models:
                rs = groups[(ds, proto, m)]
                cells = [fmt(agg(rs, lambda r, c=c: r.get("per_class_recall", {}).get(c))) for c in classes]
                L.append(f"| {m} | " + " | ".join(cells) + f" | {fmt(agg(rs, lambda r: r.get('benign_flag_rate')))} |")
        if any(r.get("multiclass") for m in models for r in groups[(ds, proto, m)]):
            L += ["", "**Multi-class scoring (attack category).** TGN: hierarchical and consistent with its binary "
                  "output (Benign when p_t < ω, otherwise the Z_t category head's most likely attack class); "
                  "RF / XGBoost: flat multi-class classifiers on the same categories.", "",
                  "| Model | Accuracy | Macro-F1 (all classes) | Macro-F1 (attack classes) | Weighted-F1 |",
                  "|---|---|---|---|---|"]
            for m in models:
                rs = [r for r in groups[(ds, proto, m)] if r.get("multiclass")]
                if rs:
                    L.append(f"| {m} | " + " | ".join(fmt(agg(rs, lambda r, k=k: r["multiclass"][k])) for k in
                                                      ("accuracy", "macro_f1", "macro_f1_attacks", "weighted_f1")) + " |")
        notes = sorted({r.get("note") for m in models for r in groups[(ds, proto, m)] if r.get("note")})
        L += [""] + [f"- {n}" for n in notes] + [""]
    L += ["## Context-only references (not tabulated against our numbers)", "",
          "- Causal-GNN [12]: CIC-IDS2017 ROC-AUC 0.934 as an *unsupervised anomaly detector* (release and split not "
          "stated); not comparable with supervised AUC.",
          "- Ontology-APT [13]: DARPA TRACE (F1 0.92, AUC 0.96); different data and detection unit.",
          "- AIR [5]: private enterprise data, multi-class (mean F1 0.94 ± 0.01).", ""]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"[report] {len(runs)} runs -> {a.out}")


if __name__ == "__main__":
    main()
