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


FIXED_FPR_TARGETS = (0.0001, 0.001, 0.01)     # 0.01 %, 0.1 %, 1 % as in CCSTGN [11] Table II


def threshold_for_fpr(y_val, s_val, target: float) -> float:
    """Validation threshold whose FPR is closest to `target` (CCSTGN [11], experiment 2)."""
    y_val, s_val = np.asarray(y_val).astype(int), np.asarray(s_val, dtype=np.float64)
    benign = np.sort(s_val[y_val == 0])
    if not len(benign):
        return 0.5
    cands = np.unique(np.concatenate([np.quantile(benign, np.linspace(0.9, 1.0, 2001)), benign[-2000:]]))
    cands = np.concatenate([cands, [np.nextafter(benign[-1], np.inf)]])
    fpr = 1.0 - np.searchsorted(benign, cands, side="left") / len(benign)     # share of benign >= threshold
    return float(cands[int(np.argmin(np.abs(fpr - target)))])


def threshold_weighted(y_val, s_val, lam: float = 0.75) -> float:
    """CCSTGN [11] experiment 1: ω minimising lam * FNR + (1 - lam) * FPR on the validation set."""
    y_val, s_val = np.asarray(y_val).astype(int), np.asarray(s_val, dtype=np.float64)
    fpr, tpr, thr = roc_curve(y_val, s_val)
    cost = lam * (1.0 - tpr) + (1.0 - lam) * fpr
    th = float(thr[int(np.argmin(cost))])
    return min(th, 1.0) if np.isfinite(th) else 1.0


def weighted_threshold_metrics(y_val, s_val, y_test, s_test, lam: float = 0.75) -> Dict:
    th = threshold_weighted(y_val, s_val, lam)
    m = detection_metrics(y_test, s_test, th)
    return {"threshold": th, "lambda": lam, "f1": m["f1"], "fpr": m["fpr"], "fnr": m["fnr"],
            "precision": m["precision"], "recall": m["recall"], "roc_auc": m["roc_auc"]}


def fixed_fpr_metrics(y_val, s_val, y_test, s_test, targets=FIXED_FPR_TARGETS) -> Dict:
    out = {}
    for tgt in targets:
        th = threshold_for_fpr(y_val, s_val, tgt)
        m = detection_metrics(y_test, s_test, th)
        out[f"{tgt * 100:g}%"] = {"threshold": th, "f1": m["f1"], "fnr": m["fnr"], "fpr": m["fpr"],
                                  "precision": m["precision"], "recall": m["recall"]}
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


def multiclass_metrics(y_true, y_pred, classes) -> Dict:
    """Multi-class scoring of attack categories (paper Sec. IV.C): accuracy, macro / weighted F1 over the
    classes present in the test set, and per-class precision / recall / F1 / support."""
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    present = sorted(set(y_true.tolist()))
    per, f1s, sup = {}, [], []
    for c in present:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per[classes[c]] = {"precision": p, "recall": r, "f1": f, "support": int((y_true == c).sum())}
        f1s.append(f)
        sup.append((y_true == c).sum())
    sup = np.asarray(sup, dtype=float)
    attack = [i for i, c in enumerate(present) if c != 0]
    return {"accuracy": float((y_true == y_pred).mean()) if len(y_true) else 0.0,
            "macro_f1": float(np.mean(f1s)) if f1s else 0.0,
            "macro_f1_attacks": float(np.mean([f1s[i] for i in attack])) if attack else 0.0,
            "weighted_f1": float((np.asarray(f1s) * sup).sum() / sup.sum()) if sup.sum() else 0.0,
            "per_class": per}


def hierarchical_predictions(p, threshold: float, cls_probs) -> np.ndarray:
    """Binary-consistent multi-class prediction: Benign (0) when p < ω, else argmax over attack classes."""
    p, q = np.asarray(p, dtype=float), np.asarray(cls_probs, dtype=float)
    attack = q[:, 1:].argmax(1) + 1
    return np.where(p >= threshold, attack, 0)
