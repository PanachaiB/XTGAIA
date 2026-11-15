"""Offline checks of the evaluation protocol (synthetic data; the real runs use CICIDS2017_improved)."""
import os
import torch
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
    p, pv, fit_s, pred_s = fit_predict_egraphsage(tr, tr[cols].values, te, te[cols].values, seed=0, epochs=80,
                                                  threads=2)
    assert pv is None                                                   # no validation set passed
    m = detection_metrics(te["y"].values, p, 0.5)
    assert m["roc_auc"] > 0.95 and p.shape == (len(te),)
    ei, e, n = build_graph(te, te[cols].values)
    assert ei.shape == (2, len(te)) and n <= 2 * len(te)
    assert EGraphSAGE(2)(ei, e.float(), n).shape == (len(te),)


def test_three_way_protocols_and_ccstgn_thresholds():
    from src.evaluation.data import split3
    from src.evaluation.metrics import fixed_fpr_metrics, threshold_for_fpr, weighted_threshold_metrics
    df = synthetic(2000, seed=3)
    tr, va, te = split3(df, "pcic_a", seed=0)
    assert len(set(tr.index) & set(te.index)) == 0 and len(set(va.index) & set(te.index)) == 0
    assert (tr["y"] == 0).sum() <= (tr["y"] == 1).sum() + 1          # benign downsampled in train only
    assert abs(te["y"].mean() - df["y"].mean()) < 0.03                 # test keeps the real distribution
    tr, va, te = split3(df, "chrono", seed=0)
    assert tr["ts"].max() <= va["ts"].min() and va["ts"].max() <= te["ts"].min()
    rng = np.random.default_rng(0)
    yv = rng.random(20000) < 0.1
    sv = np.where(yv, rng.uniform(0.5, 1, len(yv)), rng.uniform(0, 0.6, len(yv)))
    th = threshold_for_fpr(yv, sv, 0.01)
    assert abs((sv[~yv] >= th).mean() - 0.01) < 0.002
    ff = fixed_fpr_metrics(yv, sv, yv, sv)
    assert set(ff) == {"0.01%", "0.1%", "1%"} and ff["1%"]["fpr"] == pytest.approx(0.01, abs=0.002)
    wt = weighted_threshold_metrics(yv, sv, yv, sv)
    assert 0.0 <= wt["threshold"] <= 1.0 and wt["fnr"] < 0.2


def test_neural_flow_baselines_run_with_validation():
    from src.evaluation.baselines import fit_predict_nn
    df = synthetic(1500)
    cols = ["f::a", "f::b"] * 3                                         # 6 features for the 1D-CNN pooling
    X = df[cols].values
    for kind in ("mlp", "cnn1d"):
        p, pv, fit_s, pred_s = fit_predict_nn(kind, X[:1000], df["y"].values[:1000], X[1200:], seed=0, threads=2,
                                              Xva=X[1000:1200], yva=df["y"].values[1000:1200], epochs=5, batch=128)
        assert p.shape == (300,) and pv.shape == (200,)
        assert detection_metrics(df["y"].values[1200:], p, 0.5)["roc_auc"] > 0.9


def test_tgn_streaming_chrono_end_to_end(tmp_path):
    from src.evaluation.tgn_eval import build_training_table
    from src.role2_ai.train_tgn import train_streaming
    df = synthetic(900, seed=5)
    df["day"] = "unsw"
    for i in range(14):                                                 # 16 features like the CICIDS contract
        df[f"f::x{i}"] = np.float32(i)
    n = len(df)
    tr, va, te = df.iloc[:630], df.iloc[630:720], df.iloc[720:]
    evals = pd.concat([va.assign(_split="val"), te.assign(_split="test")])
    meta = build_training_table(tr, 0, 1.0, 10**6, str(tmp_path / "t.csv"), str(tmp_path / "m.json"),
                                log_transform=True, dataset="nf_unsw_nb15_v2", eval_rows=evals)
    assert meta["transform"] == "signed_log1p" and meta["rows"] == n
    metrics, pr = train_streaming(str(tmp_path / "t.csv"), str(tmp_path / "m.json"), str(tmp_path / "c.pt"),
                                  max_epochs=2, patience=1, seed=0, batch_size=64, return_predictions=True)
    assert set(np.unique(pr["split"])) == {"train", "val", "test"} and len(pr["preds"]) == n
    assert np.isfinite(pr["preds"]).all()
    # Stage 3: warm start from a pre-trained checkpoint (e.g. CSE-CIC-IDS2018) - weights only, no explainer
    train_streaming(str(tmp_path / "t.csv"), str(tmp_path / "m.json"), str(tmp_path / "c2.pt"),
                         max_epochs=1, patience=1, seed=1, batch_size=64, init_from=str(tmp_path / "c.pt"),
                         fit_explainer=False)
    assert (tmp_path / "c2.pt").exists()
    # progress checkpoint after every epoch; an interrupted run resumes after its last finished epoch
    part = torch.load(str(tmp_path / "c.pt") + ".partial", weights_only=True)
    assert part["last_epoch"] >= 1 and "optimizer" in part
    os.environ["XTGAIA_RESUME"] = "1"
    try:
        train_streaming(str(tmp_path / "t.csv"), str(tmp_path / "m.json"), str(tmp_path / "c.pt"),
                        max_epochs=3, patience=5, seed=0, batch_size=64, fit_explainer=False)
    finally:
        del os.environ["XTGAIA_RESUME"]
    assert torch.load(str(tmp_path / "c.pt") + ".partial", weights_only=True)["last_epoch"] == 3


def test_epoch_seconds_is_unit_independent():
    from src.evaluation.data import epoch_seconds
    s = epoch_seconds(["2017-07-07 08:59:50.315195", "2017-07-07 08:59:50.365195"])
    assert abs(s.iloc[0] - 1499417990.315195) < 1e-3 and abs((s.iloc[1] - s.iloc[0]) - 0.05) < 1e-6


def test_calibrate_threshold_searches_full_range_at_low_prevalence():
    from src.role2_ai.train_tgn import calibrate_threshold, compute_metrics
    rng = np.random.default_rng(0)
    y = (rng.random(20000) < 0.066).astype(int)                  # NF-UNSW validation prevalence
    s = np.where(y == 1, rng.beta(30, 1, len(y)), rng.beta(1, 8, len(y)))
    s[rng.choice(np.flatnonzero(y == 0), 100, replace=False)] = 0.99   # a few confident false positives
    thr, cal = calibrate_threshold(y, s, max_fpr=0.01)
    assert thr > np.percentile(s, 90)                            # the old 10-90th percentile window excluded this
    assert cal["val_fpr"] <= 0.01
    brute = max(compute_metrics(y, s, t)["f1"] for t in np.unique(s) if compute_metrics(y, s, t)["fpr"] <= 0.01)
    assert abs(cal["val_f1"] - brute) < 1e-12                    # exact search
