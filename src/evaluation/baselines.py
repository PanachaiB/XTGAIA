"""
Baselines for the XTGAIA evaluation protocol.

  RandomForest / XGBoost   flow-level classifiers on all CICFlowMeter features (strong CICIDS2017 baselines)
  E-GraphSAGE              Lo et al., "E-GraphSAGE: A Graph Neural Network based Intrusion Detection
                           System for IoT" (NOMS 2022): endpoints (IP:port) are nodes initialised with
                           ones, flows are edges carrying their features; message m_uv = W_msg[h_u || e_uv],
                           mean aggregation, h_v = ReLU(W_apply[h_v || mean m_uv]), K = 2 layers, edge
                           classification from [z_u || z_v]. Train and test graphs are built separately
                           (inductive), as in the paper.
"""
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ----------------------------------------------------------------------------- tree baselines
# Recipes follow Vedesh et al. [3]: RF with 100 trees and limited depth; XGBoost lr 0.1, depth 7,
# early stopping (on our validation set). Each returns (p_test, p_val, fit_s, predict_s).
def fit_predict_rf(Xtr, ytr, Xte, seed: int, n_jobs: int = 8, Xva=None, yva=None):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=n_jobs, random_state=seed)
    t = time.perf_counter(); clf.fit(Xtr, ytr); fit_s = time.perf_counter() - t
    t = time.perf_counter(); p = clf.predict_proba(Xte)[:, 1]; pred_s = time.perf_counter() - t
    pv = clf.predict_proba(Xva)[:, 1] if Xva is not None else None
    return p, pv, fit_s, pred_s


def fit_predict_xgb(Xtr, ytr, Xte, seed: int, n_jobs: int = 8, Xva=None, yva=None):
    from xgboost import XGBClassifier
    es = Xva is not None
    clf = XGBClassifier(n_estimators=500, max_depth=7, learning_rate=0.1, tree_method="hist", n_jobs=n_jobs,
                        random_state=seed, eval_metric="logloss", early_stopping_rounds=20 if es else None)
    t = time.perf_counter()
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False) if es else clf.fit(Xtr, ytr)
    fit_s = time.perf_counter() - t
    t = time.perf_counter(); p = clf.predict_proba(Xte)[:, 1]; pred_s = time.perf_counter() - t
    pv = clf.predict_proba(Xva)[:, 1] if es else None
    return p, pv, fit_s, pred_s


# ----------------------------------------------------------------------------- neural flow baselines
class MinMax:
    """Min-max scaling to [0, 1] fitted on training data ([3]); applied after a signed log for range."""
    def fit(self, X):
        self.lo, self.hi = X.min(0), X.max(0)
        return self

    def __call__(self, X):
        return np.clip((X - self.lo) / np.where(self.hi > self.lo, self.hi - self.lo, 1.0), 0.0, 1.0).astype(np.float32)


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


class CNN1D(nn.Module):
    """[3]: two 1-D convolution layers over the feature vector, a fully connected layer, sigmoid output."""
    def __init__(self, d):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
                                  nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * (d // 4), 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.fc(self.conv(x.unsqueeze(1))).squeeze(1)


def fit_predict_nn(kind: str, Xtr, ytr, Xte, seed: int, threads: int = 8, Xva=None, yva=None,
                   epochs: int = 10, batch: int = 2048):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(threads)
    sc = MinMax().fit(signed_log(Xtr))
    tr, te = sc(signed_log(Xtr)), sc(signed_log(Xte))
    va = sc(signed_log(Xva)) if Xva is not None else None
    model = (CNN1D if kind == "cnn1d" else MLP)(tr.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    Xt, yt = torch.from_numpy(tr), torch.from_numpy(np.asarray(ytr, dtype=np.float32))
    best, best_state, stale = float("inf"), None, 0
    t = time.perf_counter()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            lossf(model(Xt[idx]), yt[idx]).backward()
            opt.step()
        if va is not None:                         # early stopping on validation loss
            model.eval()
            with torch.no_grad():
                vl = float(lossf(model(torch.from_numpy(va)), torch.from_numpy(np.asarray(yva, dtype=np.float32))))
            if vl < best - 1e-5:
                best, best_state, stale = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                stale += 1
                if stale >= 3:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    fit_s = time.perf_counter() - t
    model.eval()

    def predict(X):
        with torch.no_grad():
            return np.concatenate([torch.sigmoid(model(torch.from_numpy(X[i:i + 65536]))).numpy()
                                   for i in range(0, len(X), 65536)])
    t = time.perf_counter(); p = predict(te); pred_s = time.perf_counter() - t
    return p, (predict(va) if va is not None else None), fit_s, pred_s


# ----------------------------------------------------------------------------- E-GraphSAGE
def signed_log(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


class EGraphSAGE(nn.Module):
    def __init__(self, edim: int, hidden: int = 64):
        super().__init__()
        self.msg1, self.apply1 = nn.Linear(edim + edim, hidden), nn.Linear(edim + hidden, hidden)
        self.msg2, self.apply2 = nn.Linear(hidden + edim, hidden), nn.Linear(hidden + hidden, hidden)
        self.cls = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.edim = edim

    @staticmethod
    def _mean(msgs: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
        out = torch.zeros(n, msgs.size(1), device=msgs.device).index_add_(0, index, msgs)
        cnt = torch.zeros(n, device=msgs.device).index_add_(0, index, torch.ones_like(index, dtype=msgs.dtype))
        return out / cnt.clamp(min=1).unsqueeze(1)

    def embed(self, edge_index: torch.Tensor, e: torch.Tensor, n: int) -> torch.Tensor:
        src, dst = edge_index
        h = torch.ones(n, self.edim, device=e.device)
        m = torch.relu(self.msg1(torch.cat([h[src], e], 1)))
        h = torch.relu(self.apply1(torch.cat([h, self._mean(m, dst, n)], 1)))
        m = torch.relu(self.msg2(torch.cat([h[src], e], 1)))
        h = torch.relu(self.apply2(torch.cat([h, self._mean(m, dst, n)], 1)))
        return h

    def forward(self, edge_index, e, n):
        z = self.embed(edge_index, e, n)
        return self.cls(torch.cat([z[edge_index[0]], z[edge_index[1]]], 1)).squeeze(1)


def build_graph(df: pd.DataFrame, feats: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, int]:
    s = df["src"].values.astype(str) + ":" + df["src_port"].astype(str).values
    d = df["dst"].values.astype(str) + ":" + df["dst_port"].astype(str).values
    codes, uniques = pd.factorize(np.concatenate([s, d]))
    n_e = len(df)
    edge_index = torch.as_tensor(np.stack([codes[:n_e], codes[n_e:]]), dtype=torch.long)
    return edge_index, torch.as_tensor(feats, dtype=torch.float32), len(uniques)


def fit_predict_egraphsage(tr: pd.DataFrame, Xtr, te: pd.DataFrame, Xte, seed: int, epochs: int = 60,
                           threads: int = 8, va: pd.DataFrame = None, Xva=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(threads)
    Ztr, Zte = signed_log(Xtr), signed_log(Xte)
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-6
    Ztr, Zte = np.clip((Ztr - mu) / sd, -10, 10), np.clip((Zte - mu) / sd, -10, 10)
    ei_tr, e_tr, n_tr = build_graph(tr, Ztr)
    y = torch.tensor(np.asarray(tr["y"].values, dtype=np.float32))
    model = EGraphSAGE(e_tr.size(1))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    t = time.perf_counter()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(ei_tr, e_tr, n_tr), y)
        loss.backward()
        opt.step()
    fit_s = time.perf_counter() - t
    model.eval()
    ei_te, e_te, n_te = build_graph(te, Zte)
    t = time.perf_counter()
    with torch.no_grad():
        p = torch.sigmoid(model(ei_te, e_te, n_te)).numpy()
    pred_s = time.perf_counter() - t
    pv = None
    if va is not None:                                     # validation graph built inductively as well
        Zva = np.clip((signed_log(Xva) - mu) / sd, -10, 10)
        ei_va, e_va, n_va = build_graph(va, Zva)
        with torch.no_grad():
            pv = torch.sigmoid(model(ei_va, e_va, n_va)).numpy()
    return p, pv, fit_s, pred_s


def fit_predict_multiclass(kind: str, Xtr, ctr, Xte, seed: int, n_jobs: int = 8, Xva=None, cva=None):
    """Flat multi-class RF / XGBoost on attack categories (attack_classes.CLASSES indices)."""
    classes = np.unique(ctr)
    remap = {c: i for i, c in enumerate(classes)}
    y = np.array([remap[c] for c in ctr])
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=n_jobs, random_state=seed)
        clf.fit(Xtr, y)
    else:
        from xgboost import XGBClassifier
        keep = None if cva is None else np.isin(cva, classes)
        es = cva is not None and keep.any()
        clf = XGBClassifier(n_estimators=500, max_depth=7, learning_rate=0.1, tree_method="hist", n_jobs=n_jobs,
                            random_state=seed, eval_metric="mlogloss", early_stopping_rounds=20 if es else None)
        if es:
            clf.fit(Xtr, y, eval_set=[(Xva[keep], np.array([remap[c] for c in cva[keep]]))], verbose=False)
        else:
            clf.fit(Xtr, y)
    return classes[np.asarray(clf.predict(Xte)).astype(int).ravel()]
