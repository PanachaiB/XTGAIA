"""Role 2 paper-fidelity tests: 8-head temporal attention over N2(v), PGExplainer and attention rollout."""
import numpy as np
import pytest
import torch

from src.role2_ai.explainers import TGNExplainer
from src.role2_ai.tgn_model import XTGAIATGN


def stream(model, n=200, nodes=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    src = torch.randint(0, nodes, (n,), generator=g)
    dst = (src + torch.randint(1, nodes, (n,), generator=g)) % nodes
    t = torch.arange(n)
    msg = torch.randn(n, model.raw_msg_dim, generator=g)
    model.eval()
    with torch.no_grad():
        for i in range(0, n, 20):
            model.update_memory_events(src[i:i + 20], dst[i:i + 20], t[i:i + 20], msg[i:i + 20])
    return src, dst


def test_temporal_attention_covers_two_hop_neighbourhood_with_eight_heads():
    torch.manual_seed(0)
    m = XTGAIATGN(num_nodes=20, raw_msg_dim=16)
    stream(m)
    p, z, weights, nb = m.forward_full(torch.tensor([0, 1, 2]))
    L = 1 + m.k1 + m.k1 * m.k2
    assert p.shape == (3, 1) and z.shape == (3, 8) and len(weights) == 2
    assert weights[0].shape == (3, 8, L, L)                              # 8 heads, tokens = v + N1 + N2
    assert bool(nb["valid1"].any()) and bool(nb["valid2"].any())
    assert torch.allclose(weights[0][0, :, 0].sum(-1), torch.ones(8), atol=1e-5)   # rows are distributions
    order = nb["t1"][0][nb["valid1"][0]]
    assert torch.all(order[:-1] >= order[1:])                            # most recent interactions first


def test_token_mask_changes_prediction_and_reset_clears_neighbourhood():
    torch.manual_seed(0)
    m = XTGAIATGN(num_nodes=20, raw_msg_dim=16)
    stream(m)
    n_id = torch.tensor([3])
    with torch.no_grad():
        p_full, _, _, nb = m.forward_full(n_id)
        mask = torch.ones(1, 1 + m.k1 + m.k1 * m.k2)
        mask[0, 1:] = 1e-6
        p_self, _, _, _ = m.forward_full(n_id, token_mask=mask)
    assert not torch.allclose(p_full, p_self)
    m.reset_state()
    assert int((m.nbr_id >= 0).sum()) == 0


def test_pgexplainer_trains_and_explanations_report_fidelity_and_onset():
    torch.manual_seed(0)
    m = XTGAIATGN(num_nodes=20, raw_msg_dim=16)
    stream(m)
    ex = TGNExplainer(m)
    losses = ex.fit([torch.arange(10)] * 30)
    assert len(losses) == 30 and np.isfinite(losses).all()
    assert all(p.requires_grad for p in m.parameters())                  # TGN unfrozen again
    res = ex.explain(4, "h4", lambda i: f"h{i}")
    assert 0 < len(res["structural_edges"]) <= 5 and res["n2_edges"] > 0
    assert all(0.0 <= w <= 1.0 for _, _, w in res["structural_edges"])
    assert res["fidelity_plus"] is not None and 0.0 <= res["sparsity"] < 1.0
    tr = res["temporal"]
    assert len(tr["temporal_trace"]) > 0 and tr["onset_index"] >= 0
    assert abs(sum(tr["temporal_trace"]) - 1.0) < 1e-3
    assert res["latency_ms"]["total"] > 0


def test_engine_bundles_use_paper_explainers(tmp_path):
    from src.role1_graph.flow_sources import FLOW_FEATURES
    from src.role2_ai.inference import TGNInferenceEngine
    torch.manual_seed(0)
    known = ["a", "b", "c"]
    m = XTGAIATGN(num_nodes=40, raw_msg_dim=len(FLOW_FEATURES))
    path = tmp_path / "ck.pt"
    torch.save({"format": "xtgaia-tgn-v1", "state_dict": m.state_dict(), "num_nodes": 40, "raw_msg_dim": 16,
                "memory_dim": 128, "latent_z_dim": 8, "num_heads": 8, "threshold": 0.5,
                "architecture": XTGAIATGN.ARCHITECTURE, "k1": 10, "k2": 3,
                "feature_meta": {"features": FLOW_FEATURES, "means": {c: 0.0 for c in FLOW_FEATURES},
                                 "stds": {c: 1.0 for c in FLOW_FEATURES}, "clip": 5.0, "t0": 0.0,
                                 "ip_to_id": {ip: i for i, ip in enumerate(known)}, "num_known_nodes": 3}}, path)
    eng = TGNInferenceEngine(str(path))
    edges = [{"source": s, "target": d, "port": 22, "timestamp": float(i), "features": [1.0] * 16}
             for i, (s, d) in enumerate([("a", "b"), ("c", "b"), ("a", "c"), ("b", "a")] * 5)]
    eng.observe_batch(edges)
    bundle = eng.explain("b", 0.7)
    ext = bundle["objects"][0]["x_xtgaia_xai"]
    assert ext["structural_attribution"]["explainer_model"] == "PGExplainer"
    assert ext["temporal_attribution"]["rollout_method"] == "AttentionRollout"
    assert ext["explanation_latency_ms"]["total"] > 0 and "fidelity_plus" in ext


def test_category_head_is_hierarchical_and_consistent_with_binary_output():
    from src.evaluation.metrics import hierarchical_predictions, multiclass_metrics
    from src.role2_ai.attack_classes import CLASSES, attack_category
    assert attack_category("DoS Hulk") == "DoS" and attack_category("DDoS attacks-LOIC-HTTP") == "DDoS"
    assert attack_category("FTP-Patator") == "BruteForce" and attack_category("Exploits") == "Exploit"
    assert attack_category("Web Attack - XSS") == "WebAttack" and attack_category("BENIGN") == "Benign"
    m = XTGAIATGN(num_nodes=20, raw_msg_dim=16, num_classes=len(CLASSES))
    p, z, _, nb = m.forward_full(torch.tensor([1, 2]))
    msg = torch.randn(2, 16, requires_grad=True)
    logits = m.category_logits(nb["h"], msg)                             # event-level: node embedding + flow message
    assert logits.shape == (2, len(CLASSES))
    logits.sum().backward()                                              # stop-gradient: the trunk gets no grad
    assert m.z_projector[0].weight.grad is None and m.category_head[0].weight.grad is not None
    assert msg.grad is None
    # the message changes the category, never p_t
    assert not torch.allclose(m.category_logits(nb["h"], msg), m.category_logits(nb["h"], msg + 3.0))
    assert m.last_message(torch.tensor([1, 2])).shape == (2, 16)
    legacy = XTGAIATGN(num_nodes=20, raw_msg_dim=16, num_classes=len(CLASSES), category_msg=False)
    assert legacy.category_head[0].weight.shape[1] == 128                # earlier checkpoints still load
    q = np.zeros((3, len(CLASSES))); q[:, 0] = 0.9; q[:, 2] = 0.1          # head prefers Benign...
    pred = hierarchical_predictions([0.9, 0.2, 0.7], 0.5, q)
    assert pred.tolist() == [2, 0, 2]                                    # ...but class follows the binary verdict
    mm = multiclass_metrics([2, 0, 3], pred, CLASSES)
    assert mm["per_class"]["DDoS"]["recall"] == 1.0 and 0 < mm["macro_f1"] < 1


def test_wazuh_event_unwraps_original_mordor_record():
    import json as _json
    from src.role1_graph.wazuh_mordor import MANAGER_LOCATION, from_wazuh_event
    ev = {"EventID": 1, "Image": "C:\\x.exe", "UtcTime": "2020-09-20 12:16:08.123"}
    alert = {"location": MANAGER_LOCATION, "full_log": _json.dumps(ev), "rule": {"id": "1002", "level": 2},
             "manager": {"name": "wazuh-master"}}
    rec = from_wazuh_event(alert)
    assert rec["EventID"] == 1 and rec["wazuh_rule_id"] == "1002" and rec["wazuh_manager"] == "wazuh-master"
    assert from_wazuh_event({**alert, "location": "/var/log/syslog"}) is None


def test_caldera_stockpile_profiles_map_to_tactic_chains(tmp_path):
    import zipfile
    from src.role3_decision.caldera import caldera_profiles
    z = tmp_path / "stockpile.zip"
    with zipfile.ZipFile(z, "w") as f:
        f.writestr("stockpile-master/data/abilities/discovery/a1.yml",
                   "- id: a1\n  name: arp\n  tactic: discovery\n  technique: {attack_id: T1018, name: x}\n")
        f.writestr("stockpile-master/data/abilities/lateral-movement/a2.yml",
                   "- id: a2\n  name: psexec\n  tactic: lateral-movement\n  technique: {attack_id: T1021.002, name: y}\n")
        f.writestr("stockpile-master/data/adversaries/packs/p1.yml", "id: p1\nname: pack\natomic_ordering: [a2]\n")
        f.writestr("stockpile-master/data/adversaries/w.yml", "id: w\nname: Worm\natomic_ordering: [a1, p1, a1]\n")
    prof = caldera_profiles(str(z))
    assert prof["Worm"] == ["discovery", "lateral-movement", "discovery"] and prof["pack"] == ["lateral-movement"]


def test_vectorised_neighbour_insert_matches_sequential_insert():
    a, b = XTGAIATGN(num_nodes=30, raw_msg_dim=4, k1=3), XTGAIATGN(num_nodes=30, raw_msg_dim=4, k1=3)
    g = torch.Generator().manual_seed(1)
    for step in range(20):
        n = int(torch.randint(1, 40, (1,), generator=g))
        node = torch.randint(0, 6, (n,), generator=g)            # many repeats per node, > K1 in one batch
        other = torch.randint(0, 30, (n,), generator=g)
        t = torch.full((n,), step, dtype=torch.long) + torch.arange(n)
        msg = torch.randn(n, 4, generator=g)
        a._insert_loop(node, other, t, msg)
        b._insert(node, other, t, msg)
    for buf in ("nbr_id", "nbr_t", "nbr_msg", "nbr_ptr"):
        assert torch.equal(getattr(a, buf), getattr(b, buf)), buf
