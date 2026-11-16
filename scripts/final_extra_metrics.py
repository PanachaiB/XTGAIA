"""
Extra detection views for the final (tuned) TGN, comparison spec tasks 10-12, on the CICIDS2017 held-out Friday.

The checkpoint's end-of-training memory (Mon-Thu) is replayed over Friday exactly as `tgn_eval` does, while
the node scores of both endpoints are kept. From that one pass:

  * flow level, all flows and non-duplicate flows only (exact feature-vector duplicates of a training row
    removed), at the checkpoint's validation-calibrated threshold ω (task 11);
  * 15-minute windows: window score = max flow score, window label = any attack flow (task 12; the unit of
    Kairos [N2] Table 4, so a later OpTC run can be set against it without changing unit);
  * node level per (host, 15-minute window), node score = max p_t of the host in the window, under both
    label rules (task 10): lenient = endpoint of at least one attack flow, strict = initiates at least one
    attack flow (the attack entity, Orthrus [N4] style); MCC and TP/FP/TN/FN, plus hosts flagged per attack
    window and false node alerts per day at ω.

NF-UNSW-NB15-v2 has no timestamps (time = flow index, as in CCSTGN [11]), so 15-minute windows do not
apply; its node-level and non-duplicate numbers come from the design check (runs/design_check/).

  python scripts/final_extra_metrics.py [--checkpoint data/models/tgn_best.pt] [--out runs/final_metrics]
"""
import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

WINDOW_S = 900.0


def mcc(tp, fp, tn, fn):
    d = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / d) if d > 0 else 0.0


def metrics(y, s, thr):
    from sklearn.metrics import roc_auc_score
    y, s = np.asarray(y).astype(int), np.asarray(s, float)
    pred = s >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"n": int(len(y)), "prevalence": float(y.mean()) if len(y) else 0.0, "precision": p, "recall": r,
            "f1": 2 * p * r / (p + r) if p + r else 0.0, "fpr": fp / (fp + tn) if fp + tn else 0.0,
            "accuracy": (tp + tn) / max(len(y), 1), "mcc": mcc(tp, fp, tn, fn),
            "auc": float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else None,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def score_friday(ckpt, test, batch=128, xai_n=1000):
    from src.evaluation.tgn_eval import tgn_feature_columns
    from src.role2_ai.inference import TGNInferenceEngine
    eng = TGNInferenceEngine(ckpt)
    if eng.explainer is not None:
        eng.explainer.random_reference = True          # evaluation only: k-random-edge fidelity reference
    cols, _ = tgn_feature_columns(test, "cicids2017_improved")
    X = test[cols].values.astype(np.float64)
    src, dst, port, ts = test["src"].values, test["dst"].values, test["dst_port"].values, test["ts"].values
    p_src_all, p_dst_all = np.zeros(len(test)), np.zeros(len(test))
    labels, ys = test["label"].values, test["y"].values
    # explanation sampling: every k-th detected flow, so the >= xai_n explanations spread over the whole day
    expected = max(int((ys == 1).sum() * 0.9), 1)
    every = max(expected // max(xai_n, 1), 1)
    xai, seen = [], 0
    t0 = time.time()
    for b in range(0, len(test), batch):
        sl = slice(b, b + batch)
        edges = [{"source": s, "target": d, "port": int(p), "timestamp": float(t), "features": x.tolist()}
                 for s, d, p, t, x in zip(src[sl], dst[sl], port[sl], ts[sl], X[sl])]
        p_dst_all[sl] = eng.observe_batch(edges)
        p_src_all[sl], _ = eng.score_nodes(list(src[sl]))
        if xai_n and eng.explainer is not None and len(xai) < xai_n * 1.2:
            hit = np.flatnonzero(np.maximum(p_src_all[sl], p_dst_all[sl]) >= eng.threshold)
            for j in hit:
                seen += 1
                if seen % every:
                    continue
                i = b + j
                host = dst[i] if p_dst_all[i] >= p_src_all[i] else src[i]
                t_x = time.perf_counter()
                tiers = eng.explain_tiers(host)
                ms = (time.perf_counter() - t_x) * 1000.0
                if tiers:
                    xai.append({"label": labels[i], "y": int(ys[i]), "ms": ms, "fid_plus": tiers.get("fidelity_plus"),
                                "fid_minus": tiers.get("fidelity_minus"), "sparsity": tiers.get("sparsity"),
                                "fid_plus_l": tiers.get("fidelity_plus_logit"),
                                "fid_minus_l": tiers.get("fidelity_minus_logit"),
                                "fid_rand_l": tiers.get("fidelity_plus_logit_random"),
                                "host": host, "tiers": tiers})
        if b and b % (batch * 1000) == 0:
            print(f"    ... {b:,}/{len(test):,} flows ({time.time() - t0:.0f}s)", flush=True)
    flow = np.maximum(p_src_all, p_dst_all) if eng.flow_readout == "max" else p_dst_all
    return eng, flow, p_src_all, p_dst_all, cols, xai


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="data/models/tgn_best.pt")
    ap.add_argument("--out", default="runs/final_metrics")
    ap.add_argument("--xai-instances", type=int, default=1000)
    a = ap.parse_args(argv)
    from src.evaluation.data import load_dataset
    os.makedirs(os.path.join(REPO, a.out), exist_ok=True)
    df = load_dataset("cicids2017_improved")
    train, test = df[df["day"] != "friday"], df[df["day"] == "friday"].reset_index(drop=True)
    print(f"[*] Friday {len(test):,} flows ({test['y'].mean():.2%} attack); scoring with {a.checkpoint}", flush=True)
    eng, flow, p_src, p_dst, cols, xai = score_friday(os.path.join(REPO, a.checkpoint), test, xai_n=a.xai_instances)
    thr = float(eng.threshold)
    y = test["y"].values.astype(int)
    out = {"checkpoint": a.checkpoint, "fingerprint": eng.fingerprint, "threshold": thr,
           "flow_readout": eng.flow_readout}

    # ---- task 11: all flows vs non-duplicate flows
    key = lambda d: pd.util.hash_pandas_object(d[cols].round(6), index=False)  # noqa: E731
    dup = np.isin(key(test).values, np.unique(key(train).values))
    out["flow_all"] = metrics(y, flow, thr)
    out["flow_non_duplicate"] = metrics(y[~dup], flow[~dup], thr)
    out["duplicates_removed"] = {"n": int(dup.sum()), "share": float(dup.mean()),
                                 "attack_share_of_duplicates": float(y[dup].mean()) if dup.any() else 0.0}

    # ---- task 12: 15-minute windows
    w = ((test["ts"].values - test["ts"].values.min()) // WINDOW_S).astype(int)
    win = pd.DataFrame({"w": w, "y": y, "s": flow}).groupby("w").agg(y=("y", "max"), s=("s", "max"))
    out["window_15min"] = metrics(win["y"].values, win["s"].values, thr)

    # ---- task 10: node level per (host, window), lenient and strict labels
    long = pd.concat([
        pd.DataFrame({"node": test["src"].values, "w": w, "y": y, "is_src": True, "s": p_src}),
        pd.DataFrame({"node": test["dst"].values, "w": w, "y": y, "is_src": False, "s": p_dst})])
    g = long.groupby(["node", "w"])
    node = pd.DataFrame({"s": g["s"].max(), "lenient": g["y"].max().astype(int),
                         "strict": long.assign(a=long["y"].astype(bool) & long["is_src"]).groupby(["node", "w"])["a"]
                         .max().astype(int)})
    out["node_lenient"] = metrics(node["lenient"].values, node["s"].values, thr)
    out["node_strict"] = metrics(node["strict"].values, node["s"].values, thr)
    flagged = node[node["s"] >= thr]
    att_w = node[node["lenient"] == 1].index.get_level_values("w").unique()
    days = (test["ts"].max() - test["ts"].min()) / 86400.0
    out["node_alerting"] = {
        "hosts_flagged_per_attack_window": float(flagged[flagged.index.get_level_values("w").isin(att_w)]
                                                 .groupby(level="w").size().mean()) if len(att_w) else 0.0,
        "false_node_alerts_per_day_strict": float(((node["s"] >= thr) & (node["strict"] == 0)).sum() / max(days, 1 / 24)),
        "false_node_alerts_per_day_lenient": float(((node["s"] >= thr) & (node["lenient"] == 0)).sum() / max(days, 1 / 24)),
        "capture_hours": float(days * 24)}

    # ---- XTGAIA Tier 1+2 explanation benchmark
    if xai:
        from src.evaluation.xai_stability import attribution_vectors, identity_overlap, stability
        lat = np.array([r["ms"] for r in xai])
        num = lambda k: [float(r[k]) for r in xai if r[k] is not None]  # noqa: E731
        stab = stability([{"label": r["label"], "vectors": attribution_vectors(r["tiers"], None)} for r in xai])
        out["xai_tiers12"] = {
            "n": len(xai), "on_attack_flows": int(sum(r["y"] for r in xai)),
            "latency_ms_mean": float(lat.mean()), "latency_ms_p95": float(np.percentile(lat, 95)),
            "latency_ms_std": float(lat.std()), "within_200ms": float((lat <= 200).mean()),
            "fidelity_plus_mean": float(np.mean(num("fid_plus"))) if num("fid_plus") else None,
            "fidelity_minus_mean": float(np.mean(num("fid_minus"))) if num("fid_minus") else None,
            "sparsity_mean": float(np.mean(num("sparsity"))) if num("sparsity") else None,
            "stability_cosine": stab["cosine_mean"], "stability_pairs": stab["pairs"],
            "stability_by_label": stab["by_label"]}
        x_ = out["xai_tiers12"]
        for k_ in ("fid_plus_l", "fid_minus_l", "fid_rand_l"):
            v = np.array(num(k_)) if num(k_) else None
            x_[k_ + "_mean"] = float(v.mean()) if v is not None else None
            x_[k_ + "_median"] = float(np.median(v)) if v is not None else None
        pairs = [(r["fid_plus_l"], r["fid_rand_l"]) for r in xai if r["fid_plus_l"] is not None and r["fid_rand_l"] is not None]
        x_["topk_beats_random_share"] = float(np.mean([a > b for a, b in pairs])) if pairs else None
        x_["identity_overlap"] = identity_overlap(
            [{"host": r["host"], "label": r["label"], "edges": r["tiers"].get("structural_edges")} for r in xai])

    path = os.path.join(REPO, a.out, "cicids_heldout_extra.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    md = ["# Extra detection views: TGN, CICIDS2017 held-out Friday", "",
          f"Checkpoint `{a.checkpoint}` ({eng.fingerprint}), {eng.flow_readout} read-out, threshold ω = {thr:.4f} "
          "(validation-calibrated, not tuned on Friday).", "",
          "| View | Unit | n | Prevalence | Precision | Recall | F1 | FPR | MCC | AUC | TP | FP | TN | FN |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, unit in (("flow_all", "flow"), ("flow_non_duplicate", "flow (non-duplicate)"),
                    ("window_15min", "15-min window"), ("node_lenient", "host × 15 min, lenient"),
                    ("node_strict", "host × 15 min, strict")):
        m = out[k]
        auc = f"{m['auc']:.4f}" if m["auc"] is not None else "–"
        md.append(f"| {k} | {unit} | {m['n']:,} | {m['prevalence']:.3f} | {m['precision']:.4f} | {m['recall']:.4f} | "
                  f"{m['f1']:.4f} | {m['fpr']:.4f} | {m['mcc']:.4f} | {auc} | {m['tp']} | {m['fp']} | {m['tn']} | {m['fn']} |")
    d, na = out["duplicates_removed"], out["node_alerting"]
    md += ["", f"Duplicates removed: {d['n']:,} flows ({d['share']:.1%}), {d['attack_share_of_duplicates']:.1%} of them attacks.",
           f"Hosts flagged per attack window: {na['hosts_flagged_per_attack_window']:.1f}; false node alerts per day: "
           f"{na['false_node_alerts_per_day_strict']:.0f} (strict) / {na['false_node_alerts_per_day_lenient']:.0f} (lenient) "
           f"over a {na['capture_hours']:.1f} h capture."]
    x = out.get("xai_tiers12")
    if x:
        fp = x["fidelity_plus_mean"]
        md += ["", "**XTGAIA Tier 1+2 explanations at detection time** "
               f"({x['n']:,} detections, {x['on_attack_flows']:,} on attack flows): latency mean "
               f"{x['latency_ms_mean']:.1f} ms, p95 {x['latency_ms_p95']:.1f} ms, std {x['latency_ms_std']:.1f} ms "
               f"({x['within_200ms']:.1%} within 200 ms) | fidelity+ "
               f"{fp if fp is None else round(fp, 4)} | fidelity- {x['fidelity_minus_mean']} | sparsity "
               f"{x['sparsity_mean']} | cosine stability (same attack class) {x['stability_cosine']} over "
               f"{x['stability_pairs']} pairs.",
               "", "**Measurement fix (log-odds fidelity, edge-identity stability).** "
               f"fidelity+ (logit) mean {x['fid_plus_l_mean']} / median {x['fid_plus_l_median']}; fidelity- (logit) "
               f"mean {x['fid_minus_l_mean']}; k random edges removed (logit) mean {x['fid_rand_l_mean']} / median "
               f"{x['fid_rand_l_median']}; top-k beats random in {x['topk_beats_random_share']} of explanations | "
               f"edge-identity stability (Jaccard of top-5 edges, consecutive detections of the same host and class) "
               f"{x['identity_overlap']['jaccard_mean']} over {x['identity_overlap']['pairs']} pairs."]
    with open(os.path.join(REPO, a.out, "cicids_heldout_extra.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
