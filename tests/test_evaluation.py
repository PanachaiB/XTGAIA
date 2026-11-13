"""Offline checks of the evaluation protocol (synthetic data; the real runs use CICIDS2017_improved)."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sklearn")

from src.evaluation.baselines import EGraphSAGE, build_graph, fit_predict_egraphsage  # noqa: E402
from src.evaluation.data import split, stratified_subsample  # noqa: E402
from src.evaluation.metrics import detection_metrics, summarize, tpr_at_fpr  # noqa: E402


def synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.3).astype(np.int8)
    df = pd.DataFrame({
        "src": np.where(y == 1, "172.16.0.1", rng.choice(["10.0.0.%d" % i for i in range(20)], n)),
        "dst": rng.choice(["192.168.10.%d" % i for i in range(5)], n),
        "src_port": rng.integers(1024, 65535, n), "dst_port": np.where(y == 1, 22, 443),
        "ts": np.sort(rng.random(n)) * 1e5, "label": np.where(y == 1, "SSH-Patator", "BENIGN"),
        "day": np.where(np.arange(n) < n * 0.8, "monday", "friday"), "y": y,
    })
    df["f::a"] = (y * 5 + rng.normal(0, 1, n)).astype(np.float32)
    df["f::b"] = rng.normal(0, 1, n).astype(np.float32)
    return df


def test_splits_are_stratified_and_disjoint():
    df = synthetic()
    tr, te = split(df, "random", seed=1)
    assert len(set(tr.index) & set(te.index)) == 0 and len(tr) + len(te) == len(df)
    assert abs(te["y"].mean() - df["y"].mean()) < 0.02
    tr, te = split(df, "heldout", seed=0)
    assert set(te["day"]) == {"friday"} and "friday" not in set(tr["day"])
    sub = stratified_subsample(df, 100, 0)
    assert len(sub) <= 102 and abs(sub["y"].mean() - df["y"].mean()) < 0.05


def test_metrics_and_fpr_constrained_tpr():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.9, 0.4, 0.8, 0.95, 0.99])
    m = detection_metrics(y, s, 0.5, ["BENIGN"] * 4 + ["DDoS"] * 4)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (3, 1, 1, 3)
    assert m["per_class_recall"] == {"DDoS": 0.75} and m["benign_flag_rate"] == 0.25
    assert tpr_at_fpr(y, s, 0.0) == 0.5 and tpr_at_fpr(y, s, 0.25) == 1.0
    agg = summarize([m, m])
    assert agg["f1"]["mean"] == pytest.approx(m["f1"]) and agg["n_seeds"] == 2


def test_egraphsage_learns_separable_edges():
    df = synthetic(800)
    tr, te = split(df, "random", seed=0)
    cols = ["f::a", "f::b"]
    p, fit_s, pred_s = fit_predict_egraphsage(tr, tr[cols].values, te, te[cols].values, seed=0, epochs=80, threads=2)
    m = detection_metrics(te["y"].values, p, 0.5)
    assert m["roc_auc"] > 0.95 and p.shape == (len(te),)
    ei, e, n = build_graph(te, te[cols].values)
    assert ei.shape == (2, len(te)) and n <= 2 * len(te)
    assert EGraphSAGE(2)(ei, e.float(), n).shape == (len(te),)
