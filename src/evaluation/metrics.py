"""Detection metrics for the XTGAIA evaluation protocol (paper Sec. IV.C)."""

from typing import Dict, Iterable, List, Optional

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def tpr_at_fpr(y: np.ndarray, scores: np.ndarray, max_fpr: float) -> float:
    """Highest detection rate achievable while keeping FPR <= max_fpr (threshold-free)."""
    fpr, tpr, _ = roc_curve(y, scores)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def detection_metrics(y, scores, threshold: float, labels: Optional[Iterable[str]] = None) -> Dict:
    y, scores = np.asarray(y).astype(int), np.asarray(scores, dtype=np.float64)
    pred = scores >= threshold
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    out = {
        "threshold": float(threshold), "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "fpr": fp / (fp + tn) if fp + tn else 0.0, "fnr": fn / (fn + tp) if fn + tp else 0.0,
        "roc_auc": float(roc_auc_score(y, scores)) if 0 < y.sum() < len(y) else float("nan"),
        "pr_auc": float(average_precision_score(y, scores)) if y.sum() else float("nan"),
        "tpr_at_fpr_1pct": tpr_at_fpr(y, scores, 0.01), "tpr_at_fpr_0.1pct": tpr_at_fpr(y, scores, 0.001),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": int(len(y)), "positives": int(y.sum()),
    }
    if labels is not None:
        labels = np.asarray(list(labels))
        out["per_class_recall"] = {str(c): float(pred[labels == c].mean())
                                   for c in np.unique(labels) if str(c).upper() != "BENIGN"}
        out["benign_flag_rate"] = float(pred[np.char.upper(labels.astype(str)) == "BENIGN"].mean())
    return out


def summarize(runs: List[Dict], keys=("precision", "recall", "f1", "fpr", "roc_auc", "pr_auc",
                                       "tpr_at_fpr_1pct", "tpr_at_fpr_0.1pct")) -> Dict:
    """mean, std and 95% t-interval half-width over seeds."""
    from scipy import stats
    out = {"n_seeds": len(runs)}
    for k in keys:
        v = np.array([r[k] for r in runs], dtype=float)
        half = float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
        out[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "ci95": half}
    return out
