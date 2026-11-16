"""Design check: should the TGN get a flow-level head? (rules frozen in docs/design_check_rules.md)

State machine, resumable (state in runs/design_check/state.json). Run inside xtgaia-core:

  MSYS_NO_PATHCONV=1 docker compose -f docker/docker-compose.yml run --rm --no-deps -T \
    -e PYTHONUNBUFFERED=1 -e XTGAIA_TORCH_THREADS=14 -e PYTHONPATH=/app:/app/data/pylib \
    xtgaia-core python scripts/design_check.py > runs/design_check/driver.log 2>&1

Restart after an interruption with -e XTGAIA_RESUME=1: finished stages are skipped, and once the 5 h job
has been launched, a restart relaunches it with XTGAIA_RESUME=1 so its trainings continue from their
per-epoch checkpoints.

Stages: embed -> C1 -> C2 -> decide(R1/R2/R3) -> [C3 -> decide(R4)] -> test (final configuration only)
        -> summary -> launch scripts/retrain_role2.sh. Any error -> R5 (current design) -> summary -> launch.
Every decision is appended to docs/paper_deviations.md (B10) with the numbers behind it.
"""
import json
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.evaluation.metrics import detection_metrics  # noqa: E402
from src.role2_ai.tgn_model import XTGAIATGN  # noqa: E402
from src.role2_ai.train_tgn import calibrate_threshold  # noqa: E402

TABLE = os.getenv("DC_TABLE", "data/processed/eval/tgnv2_nf_un_chrono_train_s0.csv")
CKPT = os.getenv("DC_CKPT", "data/processed/eval/tgnv2_nf_un_chrono_s0.pt")
OUT = os.getenv("DC_OUT", "runs/design_check")
DEVIATIONS = os.getenv("DC_DEVIATIONS", "docs/paper_deviations.md")
LIMIT = int(os.getenv("DC_LIMIT", "0") or 0)          # smoke tests only: first N rows of the table
LAUNCH = os.getenv("DC_LAUNCH", "1") == "1"           # smoke tests set 0
LAUNCH_SCRIPT = os.getenv("DC_LAUNCH_SCRIPT", "scripts/retrain_role2.sh")
SEEDS = (0, 1, 2)
HEAD_EPOCHS = int(os.getenv("DC_HEAD_EPOCHS", "6"))
HEAD_SUBSAMPLE = float(os.getenv("DC_HEAD_SUBSAMPLE", "0.5"))
TOPK = 10
STATE = os.path.join(OUT, "state.json")
CACHE = os.getenv("DC_CACHE", os.path.join(OUT, "cache"))   # a rerun may reuse an earlier run's embeddings
EMB_KEYS = ("h_src", "m_src", "h_dst", "m_dst")


# ---------------------------------------------------------------------------------------------- state / log
def load_state():
    if os.path.exists(STATE):
        if os.getenv("XTGAIA_RESUME", "0") != "1":
            sys.exit(f"[design-check] {STATE} exists; restart with XTGAIA_RESUME=1 (or remove {OUT} to start over)")
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    return {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": {}, "decisions": [], "table": TABLE,
            "checkpoint": CKPT, "limit": LIMIT}


def save_state(st):
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2, default=float)
    os.replace(tmp, STATE)


def log_decision(st, rule, outcome, numbers):
    """Append one decision to state and to docs/paper_deviations.md (section B10)."""
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "rule": rule, "outcome": outcome, "numbers": numbers}
    st["decisions"].append(rec)
    save_state(st)
    print(f"[design-check] {rule}: {outcome} | {json.dumps(numbers, default=float)}", flush=True)
    with open(DEVIATIONS, encoding="utf-8") as f:
        text = f.read()
    header = "### B10. Design check: flow-level head (automated decision log)"
    add = ""
    if header not in text:
        add += (f"\n\n{header}\nRules frozen in `docs/design_check_rules.md` before any check ran; validation metrics only; "
                f"summary in `runs/design_check/summary.md`. Entries are appended by `scripts/design_check.py`.\n")
    nums = "; ".join(f"{k} = {_fmt(v)}" for k, v in numbers.items())
    add += f"\n- **{rec['time']} - {rule}: {outcome}.** {nums}\n"
    with open(DEVIATIONS, "a", encoding="utf-8") as f:
        f.write(add)


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k}: {_fmt(x)}" for k, x in v.items()) + "}"
    return str(v)


# ---------------------------------------------------------------------------------------------- data
def load_table():
    head = pd.read_csv(TABLE, nrows=1)
    feat = [c for c in head.columns if c.startswith("feat_")]
    df = pd.read_csv(TABLE, usecols=["src_ip", "dst_ip", "timestamp", "label", "split"] + feat,
                     dtype={c: np.float32 for c in feat})
    if LIMIT and LIMIT < len(df):                    # smoke test: every k-th row, so all splits stay present
        df = df.iloc[:: int(np.ceil(len(df) / LIMIT))].reset_index(drop=True)
    return df, feat


def stage_embed(st, df, feat):
    """One streaming pass of the current TGN: node p_t and embeddings (h, memory) of both endpoints per event."""
    if st["stages"].get("embed", {}).get("done"):
        return
    names = [f"{k}.npy" for k in EMB_KEYS] + ["p_src.npy", "p_dst.npy"]
    if all(os.path.exists(os.path.join(CACHE, f)) for f in names) and             len(np.load(os.path.join(CACHE, "p_dst.npy"), mmap_mode="r")) == len(df):
        # same checkpoint and table -> identical embeddings; streaming again would only repeat the pass
        st["stages"]["embed"] = {"done": True, "rows": len(df), "reused_cache": CACHE}
        save_state(st)
        print(f"[design-check] reusing embeddings from {CACHE}", flush=True)
        return
    os.makedirs(CACHE, exist_ok=True)
    ck = torch.load(CKPT, map_location="cpu", weights_only=True)
    sd = ck["state_dict"]
    md = int(ck["memory_dim"])
    model = XTGAIATGN(num_nodes=int(ck["num_nodes"]), raw_msg_dim=int(ck["raw_msg_dim"]), memory_dim=md,
                      latent_z_dim=int(ck["latent_z_dim"]), num_heads=int(ck["num_heads"]),
                      k1=int(ck.get("k1") or 10), k2=int(ck.get("k2") or 3),
                      num_classes=int(ck.get("num_classes") or 0),
                      memory_skip=sd["z_projector.0.weight"].shape[1] == 2 * md,
                      category_msg=("category_head.0.weight" in sd and sd["category_head.0.weight"].shape[1] > md))
    model.load_state_dict(sd)
    model.eval()
    model.reset_state()
    n = len(df)
    p_src = np.zeros(n, np.float32)
    p_dst = np.zeros(n, np.float32)
    emb = {k: np.lib.format.open_memmap(os.path.join(CACHE, f"{k}.npy"), mode="w+", dtype=np.float16, shape=(n, md))
           for k in EMB_KEYS}
    src = torch.tensor(df["src_ip"].values)
    dst = torch.tensor(df["dst_ip"].values)
    t = torch.tensor(df["timestamp"].values, dtype=torch.long)
    msg = torch.from_numpy(df[feat].values)
    B = 128
    t0 = time.time()
    with torch.no_grad():
        for b in range(0, n, B):
            sl = slice(b, b + B)
            model.update_memory_events(src[sl], dst[sl], t[sl], msg[sl])
            pd_, _, _, nb_d = model.forward_full(dst[sl])
            ps_, _, _, nb_s = model.forward_full(src[sl])
            p_dst[sl] = pd_.squeeze(-1).numpy()
            p_src[sl] = ps_.squeeze(-1).numpy()
            emb["h_src"][sl], emb["m_src"][sl] = nb_s["h"].numpy(), nb_s["mem"].numpy()
            emb["h_dst"][sl], emb["m_dst"][sl] = nb_d["h"].numpy(), nb_d["mem"].numpy()
            if b % 100000 < B:
                print(f"  [embed] {b:,}/{n:,} ({time.time() - t0:.0f}s)", flush=True)
    for a in emb.values():
        a.flush()
    np.save(os.path.join(CACHE, "p_src.npy"), p_src)
    np.save(os.path.join(CACHE, "p_dst.npy"), p_dst)
    st["stages"]["embed"] = {"done": True, "rows": n, "seconds": time.time() - t0, "memory_dim": md}
    save_state(st)


# ---------------------------------------------------------------------------------------------- metrics
def flow_view(y_val, s_val):
    """Threshold rule (every arm): best validation F1 with validation FPR <= 1 %; metrics on validation."""
    thr, cal = calibrate_threshold(y_val, s_val, max_fpr=0.01)
    m = detection_metrics(y_val, s_val, thr)
    return {"threshold": float(thr), "f1": m["f1"], "precision": m["precision"], "recall": m["recall"],
            "fpr": m["fpr"], "auc": m["roc_auc"], "tpr_at_0.1pct_fpr": m["tpr_at_fpr_0.1pct"],
            "tpr_at_1pct_fpr": m["tpr_at_fpr_1pct"]}


def rollups(node, score):
    g = pd.DataFrame({"node": node, "s": score})
    out = {"max": g.groupby("node")["s"].max(),
           "noisy_or": 1.0 - np.exp(g.assign(l=np.log1p(-np.clip(g["s"], 0, 1 - 1e-7))).groupby("node")["l"].sum())}
    g = g.sort_values(["node", "s"], ascending=[True, False])
    out["topk_mean"] = g[g.groupby("node").cumcount() < TOPK].groupby("node")["s"].mean()
    return out


def entity_view(src, dst, y, s_src_side, s_dst_side, thr, order):
    """Node labels and entity metrics (same definition for every arm); roll-up chosen by validation node AUC."""
    from sklearn.metrics import roc_auc_score
    node = np.concatenate([src, dst])
    lab = np.concatenate([y, y])
    score = np.concatenate([s_src_side, s_dst_side])
    when = np.concatenate([order, order])
    node_label = pd.Series(lab, index=node).groupby(level=0).max()
    best = None
    for name, r in rollups(node, score).items():
        yl = node_label.loc[r.index].values
        auc = float(roc_auc_score(yl, r.values)) if 0 < yl.sum() < len(yl) else float("nan")
        if best is None or (np.isfinite(auc) and (not np.isfinite(best[1]) or auc > best[1])):
            best = (name, auc, r)
    flagged = score >= thr
    hit_nodes = set(node[flagged])
    mal = node_label.index[node_label.values == 1]
    ben = node_label.index[node_label.values == 0]
    det = float(np.mean([v in hit_nodes for v in mal])) if len(mal) else float("nan")
    fa = float(np.mean([v in hit_nodes for v in ben])) if len(ben) else float("nan")
    ttfd = []
    att = lab == 1
    for v in mal:
        sel = (node == v) & att
        first_att = when[sel].min()
        fl = when[sel & flagged]
        fl = fl[fl >= first_att]
        if len(fl):
            ttfd.append(float(fl.min() - first_att))
    return {"rollup": best[0], "node_auc": best[1], "nodes": int(len(node_label)), "malicious_nodes": int(len(mal)),
            "entity_detection": det, "benign_node_false_alarm": fa,
            "median_time_to_first_detection": float(np.median(ttfd)) if ttfd else float("nan"),
            "detected_nodes_with_ttfd": len(ttfd)}


# ---------------------------------------------------------------------------------------------- heads
class Head(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_head(X_tr, y_tr, X_va, y_va, seed, epochs=HEAD_EPOCHS, subsample=HEAD_SUBSAMPLE, batch=1024):
    """Small MLP head; epoch kept by validation AUC. Returns validation scores and the model."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    idx = rng.permutation(len(X_tr))[: max(1, int(len(X_tr) * subsample))]
    Xt, yt = torch.from_numpy(X_tr[idx].astype(np.float32)), torch.from_numpy(y_tr[idx].astype(np.float32))
    pos_w = float((len(yt) - yt.sum()) / max(float(yt.sum()), 1.0))
    model = Head(Xt.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    Xv = torch.from_numpy(X_va.astype(np.float32))
    best = (-1.0, None, None)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), batch):
            j = perm[i:i + batch]
            loss = nn.functional.binary_cross_entropy_with_logits(model(Xt[j]), yt[j], pos_weight=torch.tensor(pos_w))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            sv = torch.sigmoid(torch.cat([model(Xv[i:i + 65536]) for i in range(0, len(Xv), 65536)])).numpy()
        auc = float(roc_auc_score(y_va, sv)) if 0 < y_va.sum() < len(y_va) else -1.0
        if auc > best[0]:
            best = (auc, sv.copy(), {k: v.clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    return best[1], model


def head_latency_ms(model, d, reps=200):
    x = torch.randn(128, d)
    model.eval()
    with torch.no_grad():
        for _ in range(20):
            model(x)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            model(x)
            ts.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(ts))


def arm_inputs(arm, df, feat, rows):
    parts = []
    if arm in ("node-only", "hybrid"):
        for k in EMB_KEYS:
            parts.append(np.load(os.path.join(CACHE, f"{k}.npy"), mmap_mode="r")[rows].astype(np.float32))
    if arm in ("flow-only", "hybrid"):
        parts.append(df[feat].values[rows].astype(np.float32))
    return np.concatenate(parts, 1)


def noise_band(values):
    v = np.asarray(values, float)
    return max(2.0 * float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, 0.005)


# ---------------------------------------------------------------------------------------------- stages
def stage_c1(st, df):
    if st["stages"].get("C1", {}).get("done"):
        return
    p_src, p_dst = np.load(os.path.join(CACHE, "p_src.npy")), np.load(os.path.join(CACHE, "p_dst.npy"))
    va = np.flatnonzero(df["split"].values == "val")
    y = df["label"].values[va]
    res = {}
    for name, s in (("dst", p_dst[va]), ("src", p_src[va]), ("max", np.maximum(p_dst, p_src)[va])):
        res[f"flow_{name}"] = flow_view(y, s)
    thr = res["flow_dst"]["threshold"]
    res["node"] = entity_view(df["src_ip"].values[va], df["dst_ip"].values[va], y, p_src[va], p_dst[va], thr, va)
    st["stages"]["C1"] = {"done": True, **res}
    save_state(st)


def stage_c2(st, df, feat):
    if st["stages"].get("C2", {}).get("done"):
        return
    split = df["split"].values
    tr, va = np.flatnonzero(split == "train"), np.flatnonzero(split == "val")
    X = df[feat].values
    y = df["label"].values
    runs = st["stages"].setdefault("C2", {"runs": {}})["runs"]
    for seed in SEEDS:
        if str(seed) in runs:
            continue
        sv, _ = train_head(X[tr], y[tr], X[va], y[va], seed, subsample=1.0)
        runs[str(seed)] = flow_view(y[va], sv)
        save_state(st)
    f1 = [runs[str(s)]["f1"] for s in SEEDS]
    st["stages"]["C2"].update({"done": True, "f1_mean": float(np.mean(f1)), "f1_std": float(np.std(f1, ddof=1)),
                               "noise": noise_band(f1)})
    save_state(st)


def decide_r123(st):
    if "r123" in st["stages"]:
        return st["stages"]["r123"]["rule"]
    c1, c2 = st["stages"]["C1"], st["stages"]["C2"]
    for v in (c1["flow_dst"]["f1"], c2["f1_mean"]):
        if not np.isfinite(v):
            raise RuntimeError("non-finite F1 in C1/C2")
    nums = {"MLP val F1 (mean of 3)": c2["f1_mean"], "MLP sd": c2["f1_std"], "noise band": c2["noise"],
            "TGN C1 val F1 (dst projection)": c1["flow_dst"]["f1"], "TGN C1 val F1 (max projection)": c1["flow_max"]["f1"],
            "TGN C1 val AUC": c1["flow_dst"]["auc"], "MLP val AUC (seed 0)": c2["runs"]["0"]["auc"]}
    if c2["f1_mean"] < 0.96:
        rule, out = "R1", "MLP clearly below ~0.97: pipeline/protocol problem; architecture work stopped, existing job queued"
    elif c1["flow_dst"]["f1"] >= c2["f1_mean"] - c2["noise"]:
        rule, out = "R2", "C1 within seed noise of MLP: labeling-rule gap only; design kept, existing job queued"
    else:
        rule, out = "R3", "C1 far below MLP: running C3 (flow-head arms)"
    if LIMIT and os.getenv("DC_FORCE_RULE"):           # smoke tests only (never honoured on the full table)
        rule, out = os.environ["DC_FORCE_RULE"], f"forced {os.environ['DC_FORCE_RULE']} (smoke test)"
    log_decision(st, rule, out, nums)
    st["stages"]["r123"] = {"rule": rule}
    save_state(st)
    return rule


def stage_c3(st, df, feat):
    split = df["split"].values
    tr, va = np.flatnonzero(split == "train"), np.flatnonzero(split == "val")
    y = df["label"].values
    c3 = st["stages"].setdefault("C3", {"arms": {}})
    for arm in ("flow-only", "node-only", "hybrid"):
        runs = c3["arms"].setdefault(arm, {})
        if len(runs) == len(SEEDS):
            continue
        Xtr, Xva = arm_inputs(arm, df, feat, tr), arm_inputs(arm, df, feat, va)
        for seed in SEEDS:
            if str(seed) in runs:
                continue
            sv, model = train_head(Xtr, y[tr], Xva, y[va], seed)
            fv = flow_view(y[va], sv)
            ev = entity_view(df["src_ip"].values[va], df["dst_ip"].values[va], y[va], sv, sv, fv["threshold"], va)
            # PGExplainer explains node p_t. The heads are trained on detached embeddings, so risk (Eq. 5) and the
            # DQN state (Eq. 9) still consume that same p_t: target and risk/DQN input keep matching. But no head
            # arm uses p_t as its detection score, so the explanation no longer explains what raised the alert.
            runs[str(seed)] = {**fv, "entity": ev, "head_ms_per_128": head_latency_ms(model, Xtr.shape[1]),
                               "pgexplainer_target_matches_risk_dqn_input": True,
                               "pgexplainer_explains_detection_score": False}
            torch.save(model.state_dict(), os.path.join(OUT, f"head_{arm}_s{seed}.pt"))
            save_state(st)
        del Xtr, Xva
    c3["done"] = True
    save_state(st)


def decide_r4(st):
    if "r4" in st["stages"]:
        return st["stages"]["r4"]["adopt"]
    arms = st["stages"]["C3"]["arms"]
    for a in ("node-only", "hybrid"):
        if len(arms.get(a, {})) < len(SEEDS):
            raise RuntimeError(f"C3 arm {a} has fewer than {len(SEEDS)} finished seeds")
    f1 = {a: [arms[a][str(s)]["f1"] for s in SEEDS] for a in arms}
    if not all(np.isfinite(v) for a in f1 for v in f1[a]):
        raise RuntimeError("non-finite F1 in C3")
    mean = {a: float(np.mean(v)) for a, v in f1.items()}
    pooled = float(np.sqrt((np.var(f1["hybrid"], ddof=1) + np.var(f1["node-only"], ddof=1)) / 2))
    gain = mean["hybrid"] - mean["node-only"]
    ent = {a: float(np.nanmean([arms[a][str(s)]["entity"]["entity_detection"] for s in SEEDS])) for a in arms}
    ttfd = {a: float(np.nanmedian([arms[a][str(s)]["entity"]["median_time_to_first_detection"] for s in SEEDS]))
            for a in arms}
    ms = float(np.max([arms["hybrid"][str(s)]["head_ms_per_128"] for s in SEEDS]))
    gain_ok = gain > max(2 * pooled, 0.005)
    ent_ok = ent["hybrid"] >= ent["node-only"] - 0.05
    ttfd_ok = (not np.isfinite(ttfd["node-only"])) or (np.isfinite(ttfd["hybrid"]) and
                                                       ttfd["hybrid"] <= ttfd["node-only"] * 1.10 + 1)
    lat_ok = ms <= 5.0
    adopt = bool(gain_ok and ent_ok and ttfd_ok and lat_ok)
    if LIMIT and os.getenv("DC_FORCE_ADOPT") == "1":   # smoke tests only: exercise the hybrid test path
        adopt = True
    nums = {"val F1 mean": mean, "hybrid - node-only gain": gain, "2 x pooled sd": 2 * pooled,
            "entity detection": ent, "median TTFD (flow index)": ttfd, "hybrid head ms / 128 flows": ms,
            "gain_ok": gain_ok, "entity_ok": ent_ok, "ttfd_ok": ttfd_ok, "latency_ok": lat_ok}
    forced = LIMIT and os.getenv("DC_FORCE_ADOPT") == "1"
    log_decision(st, "R4", ("hybrid flow head ADOPTED" + (" (forced, smoke test)" if forced else "")) if adopt
                 else "hybrid not adopted: current design kept", nums)
    st["stages"]["r4"] = {"adopt": adopt}
    save_state(st)
    return adopt


def stage_test(st, df, feat, adopt):
    """The only access to test rows: the final chosen configuration, thresholds from validation."""
    if st["stages"].get("test", {}).get("done"):
        return
    split = df["split"].values
    va, te = np.flatnonzero(split == "val"), np.flatnonzero(split == "test")
    y = df["label"].values
    if adopt:
        Xva, Xte = arm_inputs("hybrid", df, feat, va), arm_inputs("hybrid", df, feat, te)
        runs = []
        for seed in SEEDS:
            model = Head(Xva.shape[1])
            model.load_state_dict(torch.load(os.path.join(OUT, f"head_hybrid_s{seed}.pt"), weights_only=True))
            model.eval()
            with torch.no_grad():
                sc = lambda X: torch.sigmoid(torch.cat([model(torch.from_numpy(X[i:i + 65536]))  # noqa: E731
                                                        for i in range(0, len(X), 65536)])).numpy()
                sva, ste = sc(Xva), sc(Xte)
            thr, _ = calibrate_threshold(y[va], sva, max_fpr=0.01)
            m = detection_metrics(y[te], ste, thr)
            runs.append({k: m[k] for k in ("f1", "precision", "recall", "fpr", "roc_auc", "tpr_at_fpr_1pct",
                                           "tpr_at_fpr_0.1pct")})
        cfg = "hybrid flow head (3 seeds)"
        res = {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}
    else:
        p_dst = np.load(os.path.join(CACHE, "p_dst.npy"))
        thr, _ = calibrate_threshold(y[va], p_dst[va], max_fpr=0.01)
        m = detection_metrics(y[te], p_dst[te], thr)
        cfg = "current design (node p_t, destination projection)"
        res = {k: m[k] for k in ("f1", "precision", "recall", "fpr", "roc_auc", "tpr_at_fpr_1pct", "tpr_at_fpr_0.1pct")}
    st["stages"]["test"] = {"done": True, "configuration": cfg, **res}
    save_state(st)
    log_decision(st, "Test (once)", cfg, {k: res[k] for k in ("f1", "roc_auc", "tpr_at_fpr_1pct", "tpr_at_fpr_0.1pct")})


def write_summary(st):
    L = ["# Design check summary (flow-level head)", "",
         f"Rules: `docs/design_check_rules.md`. Table `{st['table']}`, checkpoint `{st['checkpoint']}`"
         + (f", first {st['limit']:,} rows only (smoke test)" if st.get("limit") else "") + ". Validation metrics "
         "unless marked test.", ""]
    c1 = st["stages"].get("C1")
    if c1:
        L += ["## C1: current TGN, node p_t projected onto flows (validation)", "",
              "| Projection | F1 | AUC | TPR@0.1%FPR | TPR@1%FPR | FPR | ω |", "|---|---|---|---|---|---|---|"]
        for k in ("flow_dst", "flow_src", "flow_max"):
            r = c1[k]
            L.append(f"| {k[5:]} | {r['f1']:.4f} | {r['auc']:.4f} | {r['tpr_at_0.1pct_fpr']:.4f} | "
                     f"{r['tpr_at_1pct_fpr']:.4f} | {r['fpr']:.4f} | {r['threshold']:.4f} |")
        n = c1["node"]
        L += ["", f"Node level (same model): roll-up **{n['rollup']}**, node AUC {n['node_auc']:.4f} over "
              f"{n['nodes']} nodes ({n['malicious_nodes']} malicious); entity detection {n['entity_detection']:.3f}, "
              f"benign-node false alarms {n['benign_node_false_alarm']:.3f}, median time to first detection "
              f"{n['median_time_to_first_detection']:.1f} flows.", ""]
    c2 = st["stages"].get("C2")
    if c2 and c2.get("done"):
        L += ["## C2: in-house MLP, same flow features / split / threshold rule (validation)", "",
              "| Seed | F1 | AUC | TPR@0.1%FPR | TPR@1%FPR | FPR |", "|---|---|---|---|---|---|"]
        for s, r in c2["runs"].items():
            L.append(f"| {s} | {r['f1']:.4f} | {r['auc']:.4f} | {r['tpr_at_0.1pct_fpr']:.4f} | {r['tpr_at_1pct_fpr']:.4f} "
                     f"| {r['fpr']:.4f} |")
        L += ["", f"Mean F1 {c2['f1_mean']:.4f} ± {c2['f1_std']:.4f} (noise band {c2['noise']:.4f}).", ""]
    c3 = st["stages"].get("C3")
    if c3:
        L += ["## C3: flow-head arms (3 seeds, validation)", "",
              "| Arm | F1 (mean ± sd) | AUC | TPR@1%FPR | Entity detection | Benign-node FA | Median TTFD | Roll-up | "
              "Head ms/128 | PGExplainer target = risk/DQN input |", "|---|---|---|---|---|---|---|---|---|---|"]
        for arm, runs in c3["arms"].items():
            if not runs:
                continue
            g = lambda k: [r[k] for r in runs.values()]  # noqa: E731
            e = lambda k: [r["entity"][k] for r in runs.values()]  # noqa: E731
            L.append(f"| {arm} | {np.mean(g('f1')):.4f} ± {np.std(g('f1'), ddof=1) if len(runs) > 1 else 0:.4f} | "
                     f"{np.mean(g('auc')):.4f} | {np.mean(g('tpr_at_1pct_fpr')):.4f} | "
                     f"{np.nanmean(e('entity_detection')):.3f} | {np.nanmean(e('benign_node_false_alarm')):.3f} | "
                     f"{np.nanmedian(e('median_time_to_first_detection')):.1f} | "
                     f"{max(set(e('rollup')), key=e('rollup').count)} | {np.median(g('head_ms_per_128')):.2f} | "
                     f"{'yes' if all(g('pgexplainer_target_matches_risk_dqn_input')) else 'no'} (explains alert score: "
                     f"{'yes' if any(g('pgexplainer_explains_detection_score')) else 'no'}) |")
        L.append("")
    if st["decisions"]:
        L += ["## Decisions", "", "| Time | Rule | Outcome |", "|---|---|---|"]
        L += [f"| {d['time']} | {d['rule']} | {d['outcome']} |" for d in st["decisions"]]
        L.append("")
    t = st["stages"].get("test")
    if t:
        L += ["## Test split (touched once, final configuration)", "",
              f"{t['configuration']}: F1 {t['f1']:.4f}, AUC {t['roc_auc']:.4f}, TPR@1%FPR {t['tpr_at_fpr_1pct']:.4f}, "
              f"TPR@0.1%FPR {t['tpr_at_fpr_0.1pct']:.4f}, FPR {t['fpr']:.4f}.", ""]
    if st.get("launch"):
        L += ["## Next job", "", f"`{st['launch']['command']}` (XTGAIA_FLOW_HEAD={st['launch']['flow_head'] or 'off'}), "
              f"started {st['launch']['time']}.", ""]
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def launch(st, adopt):
    env = dict(os.environ)
    env["XTGAIA_FLOW_HEAD"] = st["launch"]["flow_head"] if st.get("launch") else ("hybrid" if adopt else "")
    if st.get("launch"):
        env["XTGAIA_RESUME"] = "1"                       # relaunch after a restart continues per-epoch checkpoints
    else:
        env.pop("XTGAIA_RESUME", None)
        st["launch"] = {"command": f"sh {LAUNCH_SCRIPT}", "flow_head": env["XTGAIA_FLOW_HEAD"],
                        "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        save_state(st)
        write_summary(st)
    print(f"[design-check] launching {LAUNCH_SCRIPT} (XTGAIA_FLOW_HEAD={env['XTGAIA_FLOW_HEAD'] or 'off'}, "
          f"resume={env.get('XTGAIA_RESUME', '0')})", flush=True)
    os.execvpe("sh", ["sh", LAUNCH_SCRIPT], env)


def main():
    threads = int(os.getenv("XTGAIA_TORCH_THREADS", "0") or 0)
    if threads:
        torch.set_num_threads(threads)
    os.makedirs(OUT, exist_ok=True)
    st = load_state()
    save_state(st)
    adopt = False
    try:
        df, feat = load_table()
        stage_embed(st, df, feat)
        stage_c1(st, df)
        stage_c2(st, df, feat)
        rule = decide_r123(st)
        if rule == "R3":
            stage_c3(st, df, feat)
            adopt = decide_r4(st)
        stage_test(st, df, feat, adopt)
    except Exception as exc:                              # R5: any error or inconclusive outcome
        traceback.print_exc()
        if "r5" not in st["stages"]:
            log_decision(st, "R5", "errored/inconclusive outcome: current design kept, existing job queued",
                         {"error": f"{type(exc).__name__}: {exc}"[:300]})
            st["stages"]["r5"] = {"error": str(exc)[:500]}
            save_state(st)
        adopt = False
    write_summary(st)
    if LAUNCH:
        launch(st, adopt)


if __name__ == "__main__":
    main()
