"""Explanation measurement (C5): log-odds fidelity, random-edge reference, edge-identity stability."""
import math

import numpy as np
import torch

from src.evaluation.xai_stability import identity_overlap
from src.role2_ai.explainers import TGNExplainer, _logit
from src.role2_ai.tgn_model import XTGAIATGN
from tests.test_role2_paper import stream


def test_logit_fidelity_matches_probability_fidelity_and_random_reference_is_optional():
    torch.manual_seed(0)
    m = XTGAIATGN(num_nodes=20, raw_msg_dim=16)
    stream(m)
    ex = TGNExplainer(m)
    ex.fit([torch.arange(10)] * 5)
    res = ex.explain(4, "h4", lambda i: f"h{i}")
    assert res["fidelity_plus_logit_random"] is None                 # off in the live tier
    p = res["p_t"]
    p_without = p - res["fidelity_plus"]
    assert math.isclose(res["fidelity_plus_logit"], _logit(p) - _logit(p_without), rel_tol=1e-4, abs_tol=1e-4)
    assert np.sign(res["fidelity_plus_logit"]) == np.sign(res["fidelity_plus"]) or abs(res["fidelity_plus"]) < 1e-6
    ex.random_reference = True
    res2 = ex.explain(4, "h4", lambda i: f"h{i}")
    assert res2["fidelity_plus_logit_random"] is not None and np.isfinite(res2["fidelity_plus_logit_random"])


def test_logit_is_finite_at_saturation():
    assert np.isfinite(_logit(1.0)) and np.isfinite(_logit(0.0))
    assert _logit(0.999) > _logit(0.99) > 0 > _logit(0.01)


def test_identity_overlap_uses_edge_identity_per_host_and_class():
    recs = [
        {"host": "a", "label": "DDoS", "edges": [("x", "a", 1.0), ("y", "a", 1.0)]},
        {"host": "a", "label": "DDoS", "edges": [("x", "a", 1.0), ("z", "a", 1.0)]},      # 1 of 3 shared
        {"host": "b", "label": "DDoS", "edges": [("x", "b", 1.0)]},                       # first for b
        {"host": "a", "label": "BENIGN", "edges": [("q", "a", 1.0)]},                     # skipped
        {"host": "b", "label": "DDoS", "edges": [("x", "b", 1.0)]},                       # identical
    ]
    out = identity_overlap(recs)
    assert out["pairs"] == 2
    assert math.isclose(out["jaccard_mean"], round((1 / 3 + 1.0) / 2, 4))
