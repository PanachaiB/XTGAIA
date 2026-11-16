"""Explainer baselines (paper Sec. IV-E; none of [1], [3], [5], [6], [14] publishes explanation timings, so
they are measured here on the same hardware as XTGAIA's PGExplainer / attention rollout / reward decomposition).

Setup mirrors [3] (P-CIC): CICIDS2017 (corrected release), ~100k flows sampled per day, random 80/20 split,
benign downsampled to the attack count in train only, RF / XGBoost / 1D-CNN. Explainers:
  LIME (tabular, 5,000 perturbations)                 all three models   ([6], [1])
  TreeSHAP                                             RF, XGBoost        ([14], [5] use SHAP)
  KernelSHAP (100-row k-means background, 500 evals)   1D-CNN
Per explainer: latency (mean / p95 / std, ms per explanation), deletion fidelity (drop in p(attack) when the
top-5 attributed features are replaced by their training mean), attribution concentration (share of |phi| in
the top 5 features) and stability as in [6] (cosine similarity between attribution vectors of an attack
instance and its nearest neighbour of the same attack class, in standardised feature space).
E-XAI [N5] protocol (spec task 13) on top: descriptive accuracy over a k grid (drop in p(attack) when the top-k
features are replaced by the training mean), sparsity over a τ grid (share of features whose normalised |phi| is
below τ), and stability as top-k overlap of repeated explanations of the same instance (stochastic explainers),
next to the cosine stability of [6].

  python -m src.evaluation.explainer_baselines --instances 300
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.evaluation.baselines import CNN1D, MinMax, signed_log
from src.evaluation.data import DAYS, feature_columns, load_day

TOP_K = 5
K_GRID = (1, 3, 5, 10, 20)
TAU_GRID = (0.01, 0.05, 0.1, 0.25)


def load_pcic(n_total: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = []
    for d in DAYS:                                     # one day at a time keeps peak memory low
        df = load_day(d)
        parts.append(df.sample(n=min(len(df), n_total // len(DAYS)), random_state=int(rng.integers(1 << 31))))
        del df
    return pd.concat(parts, ignore_index=True)


def train_cnn(Xtr, ytr, seed, epochs=8, batch=1024):
    torch.manual_seed(seed)
    sc = MinMax().fit(signed_log(Xtr))
    model = CNN1D(Xtr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    X, y = torch.from_numpy(sc(signed_log(Xtr))), torch.from_numpy(ytr.astype(np.float32))
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            lossf(model(X[idx]).squeeze(-1), y[idx]).backward()
            opt.step()
    model.eval()

    def predict_proba(Z):
        with torch.no_grad():
            p = torch.sigmoid(model(torch.from_numpy(sc(signed_log(np.asarray(Z, dtype=np.float32).reshape(-1, Xtr.shape[1])))))).reshape(-1)
        p = p.numpy().astype(np.float64)
        return np.stack([1 - p, p], 1)
    return predict_proba


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else None


def exai_metrics(phis, X, predict_proba, mean_vec):
    """E-XAI descriptive accuracy (k grid) and sparsity (τ grid)."""
    order = np.argsort(-np.abs(phis), axis=1)
    p0 = predict_proba(X)[:, 1]
    desc = {}
    for k in K_GRID:
        Xd = X.copy()
        for i in range(len(X)):
            Xd[i, order[i, :k]] = mean_vec[order[i, :k]]
        desc[str(k)] = float(np.mean(p0 - predict_proba(Xd)[:, 1]))
    a = np.abs(phis) / np.maximum(np.abs(phis).max(1, keepdims=True), 1e-12)
    return {"descriptive_accuracy_drop": desc, "sparsity": {str(t): float((a < t).mean()) for t in TAU_GRID}}


def topk_overlap(explain, X, runs=3, k=TOP_K):
    """E-XAI stability: mean share of the top-k features shared by repeated explanations of one instance."""
    vals = []
    for x in X:
        tops = [set(np.argsort(-np.abs(np.asarray(explain(x))))[:k]) for _ in range(runs)]
        pairs = [(a, b) for i, a in enumerate(tops) for b in tops[i + 1:]]
        vals.append(np.mean([len(a & b) / k for a, b in pairs]))
    return float(np.mean(vals)) if vals else None


def evaluate_explainer(name, explain, predict_proba, X, labels, mean_vec, std_vec, overlap_n=100):
    lat, phis = [], []
    for x in X:
        t = time.perf_counter()
        phi = explain(x)
        lat.append((time.perf_counter() - t) * 1000.0)
        phis.append(np.asarray(phi, dtype=np.float64))
    phis = np.stack(phis)
    lat = np.asarray(lat)
    # deletion fidelity: replace top-k attributed features with the training mean
    top = np.argsort(-np.abs(phis), axis=1)[:, :TOP_K]
    Xd = X.copy()
    for i in range(len(X)):
        Xd[i, top[i]] = mean_vec[top[i]]
    p0, p1 = predict_proba(X)[:, 1], predict_proba(Xd)[:, 1]
    absphi = np.abs(phis)
    conc = np.take_along_axis(absphi, top, 1).sum(1) / np.maximum(absphi.sum(1), 1e-12)
    # stability [6]: nearest neighbour of the same attack class
    Z = (X - mean_vec) / std_vec
    sims = []
    for i in range(len(X)):
        same = [j for j in range(len(X)) if j != i and labels[j] == labels[i]]
        if not same:
            continue
        j = same[int(np.argmin(((Z[same] - Z[i]) ** 2).sum(1)))]
        c = cosine(phis[i], phis[j])
        if c is not None:
            sims.append(c)
    extra = exai_metrics(phis, X, predict_proba, mean_vec)
    deterministic = name == "TreeSHAP"
    extra["stability_topk_overlap"] = 1.0 if deterministic else topk_overlap(explain, X[:overlap_n])
    extra["stability_topk_overlap_n"] = int(min(len(X), overlap_n))
    return {**extra, "explainer": name, "n": int(len(X)), "latency_ms_mean": float(lat.mean()),
            "latency_ms_p95": float(np.percentile(lat, 95)), "latency_ms_std": float(lat.std()),
            "within_200ms": float((lat <= 200.0).mean()),
            "deletion_fidelity_top5": float(np.mean(p0 - p1)), "top5_concentration": float(conc.mean()),
            "stability_cosine_nn_same_class": float(np.mean(sims)) if sims else None, "stability_pairs": len(sims)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--flows", type=int, default=100000)
    ap.add_argument("--instances", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--kernel-instances", type=int, default=None, help="KernelSHAP instances (default: all)")
    ap.add_argument("--overlap-instances", type=int, default=100, help="instances for top-k overlap stability")
    ap.add_argument("--out", default="data/processed/eval/explainer_baselines.json")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    import shap
    from lime.lime_tabular import LimeTabularExplainer
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    df = load_pcic(a.flows, a.seed)
    cols = feature_columns(df)
    rng = np.random.default_rng(a.seed)
    idx = rng.permutation(len(df))
    cut = int(0.8 * len(df))
    tr, te = df.iloc[idx[:cut]], df.iloc[idx[cut:]]
    att, ben = tr[tr["y"] == 1], tr[tr["y"] == 0]
    tr = pd.concat([att, ben.sample(n=min(len(ben), len(att)), random_state=a.seed)])     # [3]: train only
    Xtr, ytr = tr[cols].values.astype(np.float64), tr["y"].values
    print(f"[xai-baselines] {len(df):,} flows, {len(cols)} features | train {len(tr):,} (balanced) test {len(te):,}",
          flush=True)
    # explain attack instances (the case an analyst investigates), several attack classes
    te_att = te[te["y"] == 1]
    per = max(2, a.instances // max(te_att["label"].nunique(), 1))
    pick = pd.concat([g.sample(n=min(len(g), per), random_state=a.seed) for _, g in te_att.groupby("label")])
    if len(pick) < a.instances:                 # small classes run out: top up from the remaining attack flows
        rest = te_att.drop(index=pick.index)
        pick = pd.concat([pick, rest.sample(n=min(len(rest), a.instances - len(pick)), random_state=a.seed)])
    pick = pick.sample(n=min(len(pick), a.instances), random_state=a.seed)
    X, labels = pick[cols].values.astype(np.float64), pick["label"].values
    mean_vec, std_vec = Xtr.mean(0), Xtr.std(0) + 1e-9

    models = {}
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=a.threads, random_state=a.seed).fit(Xtr, ytr)
    models["Random Forest"] = (rf.predict_proba, rf)
    xgb = XGBClassifier(n_estimators=300, max_depth=7, learning_rate=0.1, tree_method="hist", n_jobs=a.threads,
                        random_state=a.seed).fit(Xtr, ytr)
    models["XGBoost"] = (xgb.predict_proba, xgb)
    models["1D-CNN"] = (train_cnn(Xtr, ytr, a.seed), None)

    lime_exp = LimeTabularExplainer(Xtr, feature_names=cols, class_names=["benign", "attack"], discretize_continuous=True,
                                    random_state=a.seed)
    background = shap.kmeans(Xtr, 100)
    results = []
    for mname, (proba, raw) in models.items():
        def lime_fn(x, proba=proba):
            e = lime_exp.explain_instance(x, proba, num_features=len(cols), num_samples=5000, labels=(1,))
            phi = np.zeros(len(cols))
            for k, w in e.as_map()[1]:
                phi[k] = w
            return phi
        runs = [("LIME", lime_fn)]
        if raw is not None:
            tree = shap.TreeExplainer(raw)

            def tree_fn(x, tree=tree):
                v = tree.shap_values(x.reshape(1, -1))
                v = v[1] if isinstance(v, list) else v
                v = np.asarray(v)
                return v[0, :, 1] if v.ndim == 3 else v[0]
            runs.append(("TreeSHAP", tree_fn))
        else:
            kern = shap.KernelExplainer(lambda Z, proba=proba: proba(Z)[:, 1], background)
            runs.append(("KernelSHAP", lambda x, kern=kern: np.asarray(kern.shap_values(x.reshape(1, -1), nsamples=500,
                                                                                     silent=True))[0]))
        for ename, fn in runs:
            n = len(X) if ename != "KernelSHAP" else min(len(X), a.kernel_instances or len(X))
            r = evaluate_explainer(ename, fn, proba, X[:n], labels[:n], mean_vec, std_vec, a.overlap_instances)
            r["model"] = mname
            results.append(r)
            print(f"  {mname:<13} {ename:<10} latency {r['latency_ms_mean']:8.1f} ms (p95 {r['latency_ms_p95']:8.1f}) | "
                  f"fidelity {r['deletion_fidelity_top5']:.3f} | top-5 share {r['top5_concentration']:.3f} | "
                  f"stability {r['stability_cosine_nn_same_class'] if r['stability_cosine_nn_same_class'] is not None else float('nan'):.3f}"
                  f" | top-5 overlap {r['stability_topk_overlap'] if r['stability_topk_overlap'] is not None else float('nan'):.3f}",
                  flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"setup": {"flows": len(df), "train_balanced": len(tr), "instances": len(X), "seed": a.seed,
                             "threads": a.threads}, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
