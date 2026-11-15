"""C3 flow-head ablation under the documented R1 override (docs/paper_deviations.md B10).

Rules: docs/design_check_rules.md (unchanged). Validation rows only; the test split is not touched.
Reuses the TGN embeddings cached by scripts/design_check.py (runs/design_check/cache).

Arms (3 seeds where something is trained; same split, embeddings and threshold rule):
  flow-only  flow head (MLP over both endpoints' embedding + memory and the flow features), scored per flow
  node-only  the current node score p_t(v) (no head; flows scored by projecting p_t of the destination)
  hybrid     the same flow head, rolled up to a node score (flows keep the flow-head score)
Node labels, one definition per rule, the same in every arm, per (node, 10,000-flow window):
  lenient  node is an endpoint of at least one attack flow in the window
  strict   node is the actual attack entity: it initiates at least one attack flow in the window
Roll-ups (noisy-OR, max, top-10 mean) chosen per arm on validation node AUC under the lenient label.

Run inside xtgaia-core with capped threads so the retrain keeps its cores:
  MSYS_NO_PATHCONV=1 docker compose -f docker/docker-compose.yml run --rm --no-deps -T -e XTGAIA_TORCH_THREADS=2 \
    -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e PYTHONPATH=/app:/app/data/pylib \
    xtgaia-core python scripts/design_check_c3.py [diag|c3|all]
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import design_check as dc  # noqa: E402
from src.role2_ai.train_tgn import calibrate_threshold  # noqa: E402

OUT = os.getenv("C3_OUT", "runs/design_check/c3_override")
CACHE = os.getenv("DC_CACHE", "runs/design_check/cache")
META = os.getenv("DC_META", "data/processed/eval/tgnv2_nf_un_chrono_meta_s0.json")
WINDOW = 10_000
SEEDS = (0, 1, 2)
TOPK = 10
BOOT = 200
STATE = os.path.join(OUT, "state.json")
DRIVER_STATE = "runs/design_check/state.json"


def save(st):
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2, default=float)
    os.replace(tmp, STATE)


def load():
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    return {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "diag": {}, "arms": {}}


# ---------------------------------------------------------------------------------------------- metrics
def mcc(tp, fp, tn, fn):
    d = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / d) if d > 0 else 0.0


def tpr_at(y, s, f):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s)
    ok = fpr <= f
    return float(tpr[ok].max()) if ok.any() else 0.0


def full_metrics(y, s, thr):
    from sklearn.metrics import roc_auc_score
    y, s = np.asarray(y).astype(int), np.asarray(s, float)
    pred = s >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"threshold": float(thr), "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "precision": prec,
            "recall": rec, "fpr": fp / (fp + tn) if fp + tn else 0.0, "mcc": mcc(tp, fp, tn, fn),
            "auc": float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan"),
            "tpr_at_0.1pct_fpr": tpr_at(y, s, 0.001), "tpr_at_1pct_fpr": tpr_at(y, s, 0.01),
            "tpr_at_5pct_fpr": tpr_at(y, s, 0.05), "tp": tp, "fp": fp, "tn": tn, "fn": fn, "n": int(len(y))}


def view(y, s):
    """Shared threshold rule (exact search, best F1 within 1 % validation FPR), metrics on validation."""
    thr, _ = calibrate_threshold(y, s, max_fpr=0.01)
    return full_metrics(y, s, thr)


def bootstrap_halfwidth(y, s, thr, seed=0):
    """Half-width of the 95 % bootstrap CI of F1 at a fixed threshold (Poisson weights over rows)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    pred = np.asarray(s) >= thr
    tp_i, fp_i, fn_i = (pred & (y == 1)).astype(np.float32), (pred & (y == 0)).astype(np.float32), \
        (~pred & (y == 1)).astype(np.float32)
    f1 = []
    for _ in range(BOOT):
        w = rng.poisson(1.0, len(y)).astype(np.float32)
        tp, fp, fn = float(w @ tp_i), float(w @ fp_i), float(w @ fn_i)
        f1.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    lo, hi = np.percentile(f1, [2.5, 97.5])
    return float((hi - lo) / 2)


# ---------------------------------------------------------------------------------------------- nodes
def node_frame(src, dst, y, s_src_side, s_dst_side, order):
    """Long form: one row per (flow, endpoint) with window id, role and score."""
    w = (order - order.min()) // WINDOW
    return pd.DataFrame({"node": np.concatenate([src, dst]), "window": np.concatenate([w, w]),
                         "is_src": np.r_[np.ones(len(src), bool), np.zeros(len(dst), bool)],
                         "y": np.concatenate([y, y]), "s": np.concatenate([s_src_side, s_dst_side]),
                         "t": np.concatenate([order, order])})


def node_labels(long):
    g = long.groupby(["node", "window"])
    lenient = g["y"].max().astype(int)
    strict = long.assign(a=long["y"].astype(bool) & long["is_src"]).groupby(["node", "window"])["a"].max().astype(int)
    degree = g.size()
    return lenient, strict, degree


def rollup_scores(long):
    key = ["node", "window"]
    out = {"max": long.groupby(key)["s"].max(),
           "noisy_or": 1.0 - np.exp(long.assign(l=np.log1p(-np.clip(long["s"], 0, 1 - 1e-7))).groupby(key)["l"].sum())}
    srt = long.sort_values(key + ["s"], ascending=[True, True, False])
    out["topk_mean"] = srt[srt.groupby(key).cumcount() < TOPK].groupby(key)["s"].mean()
    return out


def saturation(score, lenient, degree):
    """Degree vs score for benign and malicious (node, window) pairs: Spearman rho and score by degree quartile."""
    from scipy.stats import spearmanr
    df = pd.DataFrame({"s": score, "y": lenient.loc[score.index].values, "d": degree.loc[score.index].values})
    out = {}
    for lab, name in ((0, "benign"), (1, "malicious")):
        d = df[df["y"] == lab]
        if len(d) < 5:
            continue
        q = pd.qcut(d["d"].rank(method="first"), 4, labels=["Q1 (low degree)", "Q2", "Q3", "Q4 (high degree)"])
        out[name] = {"n": int(len(d)), "spearman_degree_score": float(spearmanr(d["d"], d["s"])[0]),
                     "mean_score_by_degree_quartile": {str(k): float(v) for k, v in d.groupby(q, observed=True)["s"].mean().items()},
                     "share_score_ge_0.99": float((d["s"] >= 0.99).mean())}
    return out


def node_views(long, lenient, strict, degree):
    from sklearn.metrics import roc_auc_score
    rolls = rollup_scores(long)
    sel = {}
    for name, sc in rolls.items():
        yl = lenient.loc[sc.index].values
        sel[name] = float(roc_auc_score(yl, sc.values)) if 0 < yl.sum() < len(yl) else float("nan")
    best = max(sel, key=lambda k: -1 if not np.isfinite(sel[k]) else sel[k])
    sc = rolls[best]
    return {"rollup_selected_on_lenient_auc": best, "rollup_val_auc_lenient": sel,
            "lenient": view(lenient.loc[sc.index].values, sc.values),
            "strict": view(strict.loc[sc.index].values, sc.values),
            "saturation": {k: saturation(v, lenient, degree) for k, v in rolls.items()}}, sc


def entity(long, flow_thr, cats, ipmap):
    """Time to first detection, nodes flagged per attack category, false alerts per 100k flows (flow level)."""
    fl = long["s"] >= flow_thr
    att = long["y"].astype(bool)
    res = {}
    for label, nodes in (("lenient", long.loc[att, "node"].unique()),
                         ("strict", long.loc[att & long["is_src"], "node"].unique())):
        ttfd, det = [], 0
        for v in nodes:
            m = (long["node"].values == v) & att.values
            first = long["t"].values[m].min()
            hit = long["t"].values[m & fl.values]
            hit = hit[hit >= first]
            if len(hit):
                det += 1
                ttfd.append(float(hit.min() - first))
        res[label] = {"malicious_nodes": int(len(nodes)), "entity_detection": det / max(len(nodes), 1),
                      "median_time_to_first_detection_flows": float(np.median(ttfd)) if ttfd else float("nan"),
                      "nodes": sorted(ipmap.get(int(v), str(v)) for v in nodes)}
    per_cat = {}
    flows = long[long["is_src"]]                       # one row per flow
    cat_flow = cats
    for c in np.unique(cat_flow[cat_flow > 0]):
        m = (cat_flow == c)
        mi = np.flatnonzero(m)
        hit_flows = mi[flows["s"].values[mi] >= flow_thr]
        sub = long[long["fid"].isin(hit_flows)] if len(hit_flows) else long.iloc[:0]
        per_cat[str(c)] = {"flows": int(m.sum()), "nodes_flagged": int(sub["node"].nunique()),
                           "attack_entities_flagged": int(sub.loc[sub["is_src"], "node"].nunique())}
    res["nodes_flagged_per_attack_category"] = per_cat
    res["mean_nodes_flagged_per_attack_category"] = float(np.mean([v["nodes_flagged"] for v in per_cat.values()])) \
        if per_cat else float("nan")
    return res


def head_cost(model, d):
    params = int(sum(p.numel() for p in model.parameters()))
    ms128 = dc.head_latency_ms(model, d)
    x = torch.randn(WINDOW, d)
    with torch.no_grad():
        t0 = time.perf_counter()
        model(x)
        mswin = (time.perf_counter() - t0) * 1000.0
    act = 128 * (d + 128 + 64 + 1) * 4
    return {"params": params, "param_bytes": params * 4, "activation_bytes_per_128": act,
            "ms_per_128_flow_batch": ms128, "ms_per_flow": ms128 / 128.0, "ms_per_10k_flow_window": mswin}


# ---------------------------------------------------------------------------------------------- phases
def data():
    df, feat = dc.load_table()
    cat = pd.read_csv(dc.TABLE, usecols=["category"])["category"].values
    if dc.LIMIT and dc.LIMIT < len(cat):
        cat = cat[:: int(np.ceil(len(cat) / dc.LIMIT))]
    with open(META, encoding="utf-8") as f:
        ipmap = {v: k for k, v in json.load(f)["ip_to_id"].items()}
    return df, feat, cat, ipmap


def phase_diag(st, df, feat):
    if st["diag"].get("done"):
        return
    split = df["split"].values
    tr, va = np.flatnonzero(split == "train"), np.flatnonzero(split == "val")
    X, y = df[feat].values, df["label"].values
    d = {"headroom": {"vm_cpus": os.cpu_count(), "c3_torch_threads": torch.get_num_threads(),
                      "note": "retrain uses 14 threads on the 16-vCPU VM; C3 capped at 2"}}
    # (a) MLP: budget-tuned vs fixed 0.5 on the same validation slice (C2 seeds regenerated; deterministic)
    prior = {}
    if os.path.exists(DRIVER_STATE):
        with open(DRIVER_STATE, encoding="utf-8") as f:
            prior = json.load(f)["stages"].get("C2", {}).get("runs", {})
    rows = []
    for seed in SEEDS:
        sv, _ = dc.train_head(X[tr], y[tr], X[va], y[va], seed, subsample=1.0)
        tuned = view(y[va], sv)
        fixed = full_metrics(y[va], sv, 0.5)
        rows.append({"seed": seed, "budget_tuned": tuned, "fixed_0.5": fixed,
                     "matches_driver_C2_f1": abs(tuned["f1"] - prior.get(str(seed), {}).get("f1", np.nan)) < 1e-6})
    d["mlp_threshold"] = rows
    # (b) identical split, features and threshold rule (same table object and the same function)
    d["parity"] = {"split_rows": {s: int((split == s).sum()) for s in ("train", "val", "test")},
                   "mlp_features": feat, "tgn_message_features": feat, "features_identical": True,
                   "threshold_rule": f"{calibrate_threshold.__module__}.{calibrate_threshold.__name__} (shared)",
                   "tgn_scores_from": dc.CKPT, "same_table": dc.TABLE}
    # (c) duplicate flows across train / validation (exact feature vector; and with the endpoints)
    h = pd.util.hash_pandas_object(df[feat], index=False).values               # row hash of the feature vector
    he = pd.util.hash_pandas_object(pd.DataFrame({"h": h, "s": df["src_ip"].values, "d": df["dst_ip"].values}),
                                    index=False).values
    tr_h, va_h = pd.Series(y[tr], index=h[tr]), h[va]
    in_train = np.isin(va_h, tr_h.index.values)
    train_lab = tr_h.groupby(level=0).mean()
    agree = (train_lab.loc[va_h[in_train]].values.round() == y[va][in_train]).mean() if in_train.any() else float("nan")
    d["duplicates"] = {"val_rows_with_identical_features_in_train": float(in_train.mean()),
                       "val_attack_rows_with_identical_features_in_train": float(in_train[y[va] == 1].mean()),
                       "label_agreement_of_those_with_train_majority": float(agree),
                       "val_rows_identical_incl_endpoints_in_train": float(np.isin(he[va], he[tr]).mean()),
                       "duplicate_share_within_train": float(1 - len(np.unique(h[tr])) / len(tr)),
                       "duplicate_share_within_val": float(1 - len(np.unique(va_h)) / len(va))}
    d["done"] = True
    st["diag"] = d
    save(st)


def phase_c3(st, df, feat, cat, ipmap):
    split = df["split"].values
    tr, va = np.flatnonzero(split == "train"), np.flatnonzero(split == "val")
    y = df["label"].values
    src, dst = df["src_ip"].values[va], df["dst_ip"].values[va]
    yv = y[va]
    cats = cat[va]
    os.makedirs(OUT, exist_ok=True)
    # node-only: the current p_t (no seeds; deterministic single checkpoint)
    if "node-only" not in st["arms"]:
        p_src, p_dst = np.load(os.path.join(CACHE, "p_src.npy"))[va], np.load(os.path.join(CACHE, "p_dst.npy"))[va]
        fv = view(yv, p_dst)
        long = node_frame(src, dst, yv, p_src, p_dst, va)
        long["fid"] = np.r_[np.arange(len(va)), np.arange(len(va))]
        lenient, strict, degree = node_labels(long)
        nv, _ = node_views(long, lenient, strict, degree)
        ent = entity(long.assign(s=np.concatenate([p_dst, p_dst])), fv["threshold"], cats, ipmap)
        st["arms"]["node-only"] = {"seeds": {"-": {"flow": fv, "flow_max_projection": view(yv, np.maximum(p_src, p_dst)),
                                                   "node": nv, "entity": ent,
                                                   "boot_halfwidth_flow_f1": bootstrap_halfwidth(yv, p_dst, fv["threshold"]),
                                                   "cost": {"params": 0, "ms_per_flow": 0.0,
                                                            "note": "no extra head; p_t is computed by the TGN anyway"}}},
                                   "flow_score": "p_t of the destination node (current design)",
                                   "node_score": "roll-up of the node's own p_t over its events in the window",
                                   "pgexplainer": {"target_matches_risk_dqn_input": True, "explains_alert_score": True}}
        save(st)
    # flow head (shared by flow-only and hybrid): both endpoints' embedding + memory and flow features
    Xtr = dc.arm_inputs("hybrid", df, feat, tr)
    Xva = dc.arm_inputs("hybrid", df, feat, va)
    for seed in SEEDS:
        key = str(seed)
        if key in st["arms"].get("flow-only", {}).get("seeds", {}) and key in st["arms"].get("hybrid", {}).get("seeds", {}):
            continue
        sv, model = dc.train_head(Xtr, y[tr], Xva, yv, seed)
        np.save(os.path.join(OUT, f"flowhead_val_scores_s{seed}.npy"), sv)
        torch.save(model.state_dict(), os.path.join(OUT, f"flowhead_s{seed}.pt"))
        fv = view(yv, sv)
        long = node_frame(src, dst, yv, sv, sv, va)
        long["fid"] = np.r_[np.arange(len(va)), np.arange(len(va))]
        lenient, strict, degree = node_labels(long)
        nv, _ = node_views(long, lenient, strict, degree)
        ent = entity(long, fv["threshold"], cats, ipmap)
        cost = head_cost(model, Xva.shape[1])
        bh = bootstrap_halfwidth(yv, sv, fv["threshold"], seed)
        st["arms"].setdefault("flow-only", {"seeds": {}, "flow_score": "flow head",
                                            "node_score": "not used (flow-level detection only)",
                                            "pgexplainer": {"target_matches_risk_dqn_input": True,
                                                            "explains_alert_score": False}})
        st["arms"]["flow-only"]["seeds"][key] = {"flow": fv, "entity": ent, "boot_halfwidth_flow_f1": bh, "cost": cost}
        st["arms"].setdefault("hybrid", {"seeds": {}, "flow_score": "flow head",
                                         "node_score": "roll-up of flow-head scores over the node's incident flows",
                                         "pgexplainer": {"target_matches_risk_dqn_input": True,
                                                         "explains_alert_score": False,
                                                         "note": "matches only while risk / DQN keep consuming p_t; "
                                                                 "if the rolled-up node score replaced p_t there, "
                                                                 "PGExplainer would explain a different quantity"}})
        st["arms"]["hybrid"]["seeds"][key] = {"flow": fv, "node": nv, "entity": ent, "boot_halfwidth_flow_f1": bh,
                                              "cost": cost}
        save(st)
        print(f"[c3] seed {seed}: flow F1 {fv['f1']:.4f} AUC {fv['auc']:.4f} | node (lenient) F1 "
              f"{nv['lenient']['f1']:.4f} roll-up {nv['rollup_selected_on_lenient_auc']}", flush=True)


def noise(arm):
    seeds = arm["seeds"]
    f1 = [v["flow"]["f1"] for v in seeds.values()]
    sd = float(np.std(f1, ddof=1)) if len(f1) > 1 else None
    bh = float(np.mean([v["boot_halfwidth_flow_f1"] for v in seeds.values()]))
    return {"seed_sd": sd, "seed_sd_is_exactly_zero": sd == 0.0, "bootstrap_halfwidth": bh,
            "noise": max(sd or 0.0, bh, 0.005), "f1_mean": float(np.mean(f1))}


def decide_r4(st):
    a = st["arms"]
    nz = {k: noise(a[k]) for k in ("node-only", "flow-only", "hybrid")}
    gain = nz["hybrid"]["f1_mean"] - nz["node-only"]["f1_mean"]
    band = 2 * max(nz["hybrid"]["noise"], nz["node-only"]["noise"])
    node_only = a["node-only"]["seeds"]["-"]
    hyb = list(a["hybrid"]["seeds"].values())
    ent_h = float(np.mean([s["entity"]["lenient"]["entity_detection"] for s in hyb]))
    ent_n = node_only["entity"]["lenient"]["entity_detection"]
    tt_h = float(np.nanmedian([s["entity"]["lenient"]["median_time_to_first_detection_flows"] for s in hyb]))
    tt_n = node_only["entity"]["lenient"]["median_time_to_first_detection_flows"]
    ms = float(max(s["cost"]["ms_per_128_flow_batch"] for s in hyb))
    checks = {"gain_ok": bool(gain > band), "entity_ok": bool(ent_h >= ent_n - 0.05),
              "ttfd_ok": bool((not np.isfinite(tt_n)) or (np.isfinite(tt_h) and tt_h <= tt_n * 1.10 + 1)),
              "latency_ok": bool(ms <= 5.0)}
    adopt = all(checks.values())
    st["r4"] = {"adopt": adopt, "noise": nz, "gain_flow_f1_hybrid_minus_node_only": gain, "2x_noise": band,
                "entity_detection_lenient": {"hybrid": ent_h, "node-only": ent_n},
                "median_ttfd_lenient_flows": {"hybrid": tt_h, "node-only": tt_n},
                "hybrid_head_ms_per_128": ms, **checks,
                "node_level_f1_lenient": {"hybrid": float(np.mean([s["node"]["lenient"]["f1"] for s in hyb])),
                                          "node-only": node_only["node"]["lenient"]["f1"]},
                "node_level_f1_strict": {"hybrid": float(np.mean([s["node"]["strict"]["f1"] for s in hyb])),
                                         "node-only": node_only["node"]["strict"]["f1"]}}
    save(st)
    return adopt


def log_b10(st):
    r = st["r4"]
    nz = r["noise"]
    zero = [k for k, v in nz.items() if v["seed_sd_is_exactly_zero"]]
    text = (f"\n- **{time.strftime('%Y-%m-%d %H:%M')} - R4 after C3 (override run): "
            f"{'hybrid flow head ADOPTED' if r['adopt'] else 'hybrid not adopted: current design kept'}.** "
            f"Validation flow F1: node-only {nz['node-only']['f1_mean']:.4f}, flow-only {nz['flow-only']['f1_mean']:.4f}, "
            f"hybrid {nz['hybrid']['f1_mean']:.4f}; gain {r['gain_flow_f1_hybrid_minus_node_only']:+.4f} vs 2 x noise "
            f"{r['2x_noise']:.4f} (noise = max(seed sd, bootstrap half-width, 0.005): hybrid {nz['hybrid']['noise']:.4f}, "
            f"node-only {nz['node-only']['noise']:.4f}"
            + (f"; seed sd exactly 0.000 for {', '.join(zero)}, floor used" if zero else "") + "). "
            f"Entity detection (lenient) hybrid {r['entity_detection_lenient']['hybrid']:.3f} vs node-only "
            f"{r['entity_detection_lenient']['node-only']:.3f}; median time to first detection (lenient, flows) "
            f"{r['median_ttfd_lenient_flows']['hybrid']:.1f} vs {r['median_ttfd_lenient_flows']['node-only']:.1f}; "
            f"hybrid head {r['hybrid_head_ms_per_128']:.2f} ms per 128 flows. Checks: gain {r['gain_ok']}, entity "
            f"{r['entity_ok']}, TTFD {r['ttfd_ok']}, latency {r['latency_ok']}. Node-level F1 lenient: hybrid "
            f"{r['node_level_f1_lenient']['hybrid']:.4f} vs node-only {r['node_level_f1_lenient']['node-only']:.4f}; "
            f"strict: {r['node_level_f1_strict']['hybrid']:.4f} vs {r['node_level_f1_strict']['node-only']:.4f}. "
            "Test split not touched. Details: `runs/design_check/summary.md` (C3 override section).\n")
    with open(dc.DEVIATIONS, "a", encoding="utf-8") as f:
        f.write(text)


def fm(m):
    return (f"{m['f1']:.4f} | {m['auc']:.4f} | {m['tpr_at_0.1pct_fpr']:.3f} / {m['tpr_at_1pct_fpr']:.3f} / "
            f"{m['tpr_at_5pct_fpr']:.3f} | {m['precision']:.4f} | {m['mcc']:.4f} | {m['tp']}/{m['fp']}/{m['tn']}/{m['fn']}")


def noise_text(name, v):
    sd = "n/a (single checkpoint)" if v["seed_sd"] is None else f"{v['seed_sd']:.4f}"
    zero = " (exactly 0.000, floor used)" if v["seed_sd_is_exactly_zero"] else ""
    return f"{name}: seed sd {sd}{zero}, bootstrap {v['bootstrap_halfwidth']:.4f} -> noise {v['noise']:.4f}"


def write_section(st):
    d, a = st["diag"], st["arms"]
    L = ["", "## C3 override run (R1 overridden by the user; R4 unchanged; validation only)", "",
         "Recorded in B10 before running. Noise = max(seed sd, 95 % bootstrap half-width over validation flows, 0.005).",
         "", "### Sanity diagnostics", ""]
    L += ["| MLP seed | F1 budget-tuned (ω) | FPR | F1 at fixed 0.5 | FPR at 0.5 | reproduces C2 |", "|---|---|---|---|---|---|"]
    for r in d["mlp_threshold"]:
        L.append(f"| {r['seed']} | {r['budget_tuned']['f1']:.4f} ({r['budget_tuned']['threshold']:.4f}) | "
                 f"{r['budget_tuned']['fpr']:.4f} | {r['fixed_0.5']['f1']:.4f} | {r['fixed_0.5']['fpr']:.4f} | "
                 f"{'yes' if r['matches_driver_C2_f1'] else 'no'} |")
    du = d["duplicates"]
    L += ["", f"Parity: same table, split ({d['parity']['split_rows']}), the same {len(d['parity']['mlp_features'])} "
          f"flow features for the MLP and the TGN messages, shared threshold rule `{d['parity']['threshold_rule']}`.",
          f"Duplicates: {du['val_rows_with_identical_features_in_train']:.1%} of validation flows have an identical "
          f"feature vector in train ({du['val_attack_rows_with_identical_features_in_train']:.1%} of validation attacks; "
          f"label agreement with the train majority {du['label_agreement_of_those_with_train_majority']:.1%}); "
          f"{du['val_rows_identical_incl_endpoints_in_train']:.1%} including endpoints. Within-split duplicate share: "
          f"train {du['duplicate_share_within_train']:.1%}, validation {du['duplicate_share_within_val']:.1%}.",
          f"Headroom: {d['headroom']['note']}.", ""]
    L += ["### Flow level (validation; flow labels)", "",
          "| Arm | Seed | F1 | AUC | TPR@0.1/1/5 %FPR | Precision | MCC | TP/FP/TN/FN |", "|---|---|---|---|---|---|---|---|"]
    for arm in ("node-only", "flow-only", "hybrid"):
        for s, v in a[arm]["seeds"].items():
            L.append(f"| {arm} | {s} | {fm(v['flow'])} |")
    L += ["", "### Node level (validation; (node, 10,000-flow window) pairs)", "",
          "| Arm | Seed | Label | Roll-up | F1 | AUC | TPR@0.1/1/5 %FPR | Precision | MCC | TP/FP/TN/FN |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for arm in ("node-only", "hybrid"):
        for s, v in a[arm]["seeds"].items():
            for lab in ("lenient", "strict"):
                L.append(f"| {arm} | {s} | {lab} | {v['node']['rollup_selected_on_lenient_auc']} | {fm(v['node'][lab])} |")
    L += ["", "### Entity level (validation; flow threshold ω of each arm)", "",
          "| Arm | Seed | Label | Malicious nodes | Entity detection | Median TTFD (flows) | Mean nodes flagged per attack category |",
          "|---|---|---|---|---|---|---|"]
    for arm in ("node-only", "flow-only", "hybrid"):
        for s, v in a[arm]["seeds"].items():
            for lab in ("lenient", "strict"):
                e = v["entity"][lab]
                L.append(f"| {arm} | {s} | {lab} | {e['malicious_nodes']} | {e['entity_detection']:.3f} | "
                         f"{e['median_time_to_first_detection_flows']:.1f} | "
                         f"{v['entity']['mean_nodes_flagged_per_attack_category']:.1f} |")
    L += ["", "False alerts are reported per 100k validation flows (the dataset has no wall clock, so 'per day' is "
          "not available):", ""]
    for arm in ("node-only", "flow-only", "hybrid"):
        for s, v in a[arm]["seeds"].items():
            f = v["flow"]
            line = f"- {arm} seed {s}: {f['fp'] / f['n'] * 1e5:.0f} false flow alerts per 100k flows"
            if "node" in v:
                for lab in ("lenient", "strict"):
                    line += f"; {v['node'][lab]['fp'] / f['n'] * 1e5:.1f} false node alerts ({lab}) per 100k flows"
            L.append(line)
    L += ["", "### Roll-up saturation check (degree vs score, all roll-ups)", ""]
    for arm in ("node-only", "hybrid"):
        v = list(a[arm]["seeds"].values())[0]
        L.append(f"**{arm}** (seed {list(a[arm]['seeds'])[0]}), validation lenient AUC per roll-up: "
                 + ", ".join(f"{k} {x:.4f}" for k, x in v["node"]["rollup_val_auc_lenient"].items()))
        L += ["", "| Roll-up | Label | n | Spearman(degree, score) | Mean score Q1→Q4 degree | Share ≥ 0.99 |",
              "|---|---|---|---|---|---|"]
        for r, sat in v["node"]["saturation"].items():
            for lab, x in sat.items():
                q = " → ".join(f"{y:.3f}" for y in x["mean_score_by_degree_quartile"].values())
                L.append(f"| {r} | {lab} | {x['n']} | {x['spearman_degree_score']:.3f} | {q} | {x['share_score_ge_0.99']:.3f} |")
        L.append("")
    L += ["### Cost and XAI", "", "| Arm | Params | ms / flow | ms / 128-flow batch | ms / 10k-flow window | PGExplainer target = risk/DQN input | Explains the alert score |",
          "|---|---|---|---|---|---|---|"]
    for arm in ("node-only", "flow-only", "hybrid"):
        v = list(a[arm]["seeds"].values())[0]
        c = v["cost"]
        L.append(f"| {arm} | {c.get('params', 0):,} | {c.get('ms_per_flow', 0):.4f} | {c.get('ms_per_128_flow_batch', 0):.2f} | "
                 f"{c.get('ms_per_10k_flow_window', 0):.1f} | {'yes' if a[arm]['pgexplainer']['target_matches_risk_dqn_input'] else 'no'} | "
                 f"{'yes' if a[arm]['pgexplainer']['explains_alert_score'] else 'no'} |")
    L += ["", a["hybrid"]["pgexplainer"]["note"], ""]
    if "r4" in st:
        r = st["r4"]
        L += ["### R4 (criteria unchanged)", "",
              f"Gain (hybrid − node-only, validation flow F1) {r['gain_flow_f1_hybrid_minus_node_only']:+.4f} vs 2 × noise "
              f"{r['2x_noise']:.4f}; entity detection (lenient) {r['entity_detection_lenient']}; median TTFD (lenient) "
              f"{r['median_ttfd_lenient_flows']}; head {r['hybrid_head_ms_per_128']:.2f} ms / 128 flows. "
              f"Checks: gain {r['gain_ok']}, entity {r['entity_ok']}, TTFD {r['ttfd_ok']}, latency {r['latency_ok']}. "
              f"**{'Hybrid adopted' if r['adopt'] else 'Hybrid not adopted: current design kept'}.**", "",
              "Noise per arm: " + "; ".join(noise_text(k, v) for k, v in r["noise"].items()), ""]
    path = os.getenv("C3_SUMMARY", "runs/design_check/summary.md")
    with open(path, encoding="utf-8") as f:
        base = f.read().split("\n## C3 override run")[0]
    with open(path, "w", encoding="utf-8") as f:
        f.write(base.rstrip("\n") + "\n" + "\n".join(L))


def main():
    threads = int(os.getenv("XTGAIA_TORCH_THREADS", "2") or 2)
    torch.set_num_threads(threads)
    os.makedirs(OUT, exist_ok=True)
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"
    st = load()
    df, feat, cat, ipmap = data()
    if phase in ("diag", "all"):
        phase_diag(st, df, feat)
        print("[c3] diagnostics:", json.dumps({k: st["diag"][k] for k in ("duplicates",)}, default=float), flush=True)
    if phase in ("c3", "all"):
        phase_c3(st, df, feat, cat, ipmap)
        if "r4" not in st:
            decide_r4(st)
            log_b10(st)
    if os.path.exists(os.getenv("C3_SUMMARY", "runs/design_check/summary.md")) and st["diag"].get("done") and st["arms"].get("hybrid"):
        write_section(st)


if __name__ == "__main__":
    main()
