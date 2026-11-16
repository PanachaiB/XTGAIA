"""Explanation stability (paper Sec. IV.E): consistency of explanations across similar attack patterns,
measured as the cosine similarity between attribution vectors of decisions triggered by the same attack
class. Vectors are entity-independent so that two hosts hit by the same attack are comparable:

    structural  top-k PGExplainer edge-mask scores, ranked (importance profile of the 2-hop subgraph)
    temporal    attention-rollout trace resampled to a fixed number of bins (shape of the attack onset)
    strategic   Q-values of all five actions + weighted sub-rewards of the chosen action (Tier 3)
"""
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

TOP_K, TEMPORAL_BINS = 5, 10
TIERS = ("structural", "temporal", "strategic")


def attribution_vectors(tiers: Optional[Dict[str, Any]], decision_xai: Optional[Dict[str, Any]]) -> Dict[str, list]:
    out: Dict[str, list] = {}
    if tiers:
        scores = sorted((float(s) for _, _, s in tiers.get("structural_edges", [])), reverse=True)[:TOP_K]
        out["structural"] = scores + [0.0] * (TOP_K - len(scores))
        trace = list(tiers.get("temporal", {}).get("temporal_trace", []) or [])
        if trace:
            x = np.linspace(0.0, 1.0, len(trace)) if len(trace) > 1 else np.array([0.0])
            out["temporal"] = np.interp(np.linspace(0.0, 1.0, TEMPORAL_BINS), x, np.asarray(trace, float)).tolist() \
                if len(trace) > 1 else [float(trace[0])] * TEMPORAL_BINS
    strat = (decision_xai or {}).get("x_xtgaia_strategic_xai")
    if strat:
        q = [float(strat["q_values"].get(f"a{i}", 0.0)) for i in range(5)]
        sub = strat.get("sub_reward_decomposition", {})
        out["strategic"] = q + [float(sub.get(f"weighted_{c}", 0.0)) for c in ("contain", "disrupt", "cost")]
    return out


def _cos(a, b) -> Optional[float]:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return None
    return float(a @ b / (na * nb))


def stability(records: Iterable[Dict[str, Any]], max_pairs: int = 5000) -> Dict[str, Any]:
    """records: {"label": attack class, "vectors": attribution_vectors(...)}. Benign / unlabelled
    decisions are skipped. Returns the mean pairwise cosine per tier and per attack class."""
    groups: Dict[str, List[Dict[str, list]]] = {}
    for r in records:
        lab = str(r.get("label") or "")
        if lab and lab.upper() != "BENIGN" and r.get("vectors"):
            groups.setdefault(lab, []).append(r["vectors"])
    per_label: Dict[str, Dict[str, Any]] = {}
    pooled: Dict[str, List[float]] = {t: [] for t in TIERS}
    for lab, vecs in groups.items():
        row: Dict[str, Any] = {"n": len(vecs)}
        for tier in TIERS:
            sims = []
            for a, b in list(combinations([v[tier] for v in vecs if tier in v], 2))[:max_pairs]:
                c = _cos(a, b)
                if c is not None:
                    sims.append(c)
            if sims:
                row[tier] = round(float(np.mean(sims)), 4)
                pooled[tier].extend(sims)
        per_label[lab] = row
    return {"cosine_mean": {t: (round(float(np.mean(v)), 4) if v else None) for t, v in pooled.items()},
            "pairs": {t: len(v) for t, v in pooled.items()}, "by_label": per_label}


def identity_overlap(records: Iterable[Dict[str, Any]], k: int = TOP_K) -> Dict[str, Any]:
    """Stability on *which* edges are picked, not on their scores (the score vectors saturate on detected
    nodes). "Similar instances" = consecutive explanations of the same host for the same attack class.
    records: {"host", "label", "edges": [(src, dst, score), ...]} in time order. Returns the mean Jaccard
    overlap of the top-k edge sets over those pairs, overall and per attack class."""
    last: Dict[tuple, set] = {}
    per: Dict[str, List[float]] = {}
    for r in records:
        lab = str(r.get("label") or "")
        if not lab or lab.upper() == "BENIGN":
            continue
        cur = {(str(s), str(d)) for s, d, *_ in (r.get("edges") or [])[:k]}
        key = (str(r.get("host")), lab)
        prev = last.get(key)
        if prev is not None and (prev or cur):
            per.setdefault(lab, []).append(len(prev & cur) / len(prev | cur))
        last[key] = cur
    pooled = [v for vs in per.values() for v in vs]
    return {"jaccard_mean": round(float(np.mean(pooled)), 4) if pooled else None, "pairs": len(pooled),
            "by_label": {lab: {"jaccard_mean": round(float(np.mean(v)), 4), "pairs": len(v)} for lab, v in per.items()}}
