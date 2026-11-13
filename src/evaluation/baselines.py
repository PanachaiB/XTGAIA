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
def fit_predict_rf(Xtr, ytr, Xte, seed: int, n_jobs: int = 8):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=100, n_jobs=n_jobs, random_state=seed)
    t = time.perf_counter(); clf.fit(Xtr, ytr); fit_s = time.perf_counter() - t
    t = time.perf_counter(); p = clf.predict_proba(Xte)[:, 1]; pred_s = time.perf_counter() - t
    return p, fit_s, pred_s


def fit_predict_xgb(Xtr, ytr, Xte, seed: int, n_jobs: int = 8):
    from xgboost import XGBClassifier
    clf = XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, tree_method="hist", subsample=0.8,
                        colsample_bytree=0.8, n_jobs=n_jobs, random_state=seed, eval_metric="logloss")
    t = time.perf_counter(); clf.fit(Xtr, ytr); fit_s = time.perf_counter() - t
    t = time.perf_counter(); p = clf.predict_proba(Xte)[:, 1]; pred_s = time.perf_counter() - t
    return p, fit_s, pred_s


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
                           threads: int = 8) -> Tuple[np.ndarray, float, float]:
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
    return p, fit_s, time.perf_counter() - t
