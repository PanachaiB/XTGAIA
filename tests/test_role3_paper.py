"""Role 3 + pipeline paper-fidelity tests: zone multi-agent DDQN, deployment regimes, online update,
single explanation bundle X_t, Δinf windows, Algorithm 1 write-back and ablation switches."""
import numpy as np
import pytest
import torch

from src.pipeline.orchestrator import build_xt
from src.role3_decision.actions import NUM_ACTIONS, QUARANTINE
from src.role3_decision.double_dqn import QNetwork, ReplayBuffer, make_replay_buffer
from src.role3_decision.graph_interface import QuarantineManager
from src.role3_decision.decision_ledger import DecisionLedger
from src.role3_decision.online_update import OnlineUpdater
from src.role3_decision.soar_dispatcher import SOARDispatcher
from src.role3_decision.state import make_raw_state
from src.role3_decision.zone_env import XTGAIAZoneEnv, load_zone_policies, train_zone_policies
from tests.test_integration import (ATTACKER, WEB, WS, FixedQNetwork, flow, http_calls, inventory,  # noqa: F401
                                    tiny_checkpoint)


# ---------------------------------------------------------------------------- multi-agent zones
def test_zone_env_has_one_defender_per_zone_and_lateral_spread():
    env = XTGAIAZoneEnv(zones={"workstations": {"crit_tiers": [0.1]}, "servers": {"crit_tiers": [1.0]}})
    np.random.seed(3)
    spread = False
    for ep in range(30):
        env.reset()
        assert env.agents == ["attacker", "defender_workstations", "defender_servers"]
        for agent in env.agent_iter():
            obs, r, term, trunc, info = env.last()
            if term or trunc:
                env.step(None)
                continue
            if agent == "attacker":
                env.step(3)                                   # lateral-movement every turn
                spread |= len(env.held) > 1
            else:
                assert obs.shape == (16,) and info["reward_components"].shape == (3,)
                env.step(0)                                   # defenders idle -> the intrusion spreads
    assert spread


def test_independent_zone_policies_train_and_load(tmp_path):
    path = train_zone_policies(episodes=30, save_path=tmp_path / "zones.pt", seed=0, verbose=False)
    models = load_zone_policies(path)
    assert set(models) == {"workstations", "servers"}
    a, b = models["workstations"], models["servers"]
    assert not all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


def test_dispatcher_routes_by_zone_and_passive_mode_only_recommends(http_calls):
    d = SOARDispatcher(model=FixedQNetwork(0), quarantine_manager=QuarantineManager(), ledger=DecisionLedger())
    d.set_zone_policies({"servers": FixedQNetwork(QUARANTINE)}, lambda h: "servers" if h == WEB else "workstations")
    state = make_raw_state(0.9, 0.9, 1.0)
    assert d.predict_action(state, WEB)[0] == QUARANTINE and d.predict_action(state, WS)[0] == 0
    d.mode = "passive"
    import networkx as nx
    g = nx.DiGraph()
    g.add_edge(ATTACKER, WEB, port=22)
    res = d.process_alert(state, WEB, WEB, G_t=g, risk={WEB: 3.0})
    assert res["zone"] == "servers" and res["action"] == QUARANTINE and res["edges_severed"] == 0
    assert not any(c["url"].endswith("webhook_exec-test") for c in http_calls)        # nothing executed
    case = next(c for c in http_calls if c["url"].endswith("/api/v1/case"))["json"]
    assert case["title"].startswith("[XTGAIA][RECOMMENDATION]") and g.has_edge(ATTACKER, WEB)


# ---------------------------------------------------------------------------- online update
def test_online_update_is_guarded_and_swaps_accepted_policy(tmp_path, monkeypatch):
    from src.role3_decision import online_update as ou
    torch.manual_seed(0)
    d = SOARDispatcher(model=QNetwork(), quarantine_manager=QuarantineManager(), ledger=DecisionLedger())
    buf = make_replay_buffer()
    s = np.full(16, 0.5, np.float32)
    for _ in range(8):
        buf.push(s, QUARANTINE, -100.0, s, True, (0.0, 200.0, 0.0))
    up = OnlineUpdater(d, buf)
    up.out_dir = tmp_path
    up.cfg.update(min_new_transitions=2, gradient_steps=5, min_probe_agreement=0.0)
    old = d.model
    assert up.notify() is None                                   # 1 < min_new_transitions
    rec = up.notify()
    assert rec["accepted"] and d.model is not old and (tmp_path / rec["saved"].split("/")[-1].split("\\")[-1]).exists()
    up.cfg.update(min_probe_agreement=1.01)                      # impossible guard -> rejected
    rec = up.update()
    assert not rec["accepted"] and len(up.history) == 2


# ---------------------------------------------------------------------------- X_t bundle
def test_single_explanation_bundle_xt():
    det = {"type": "bundle", "id": "bundle--d", "objects": [{"type": "indicator", "id": "indicator--1"}]}
    dec = {"type": "extension-definition", "x_xtgaia_strategic_xai": {}}
    xt = build_xt(det, dec, {"tiers_1_2": 12.0, "tier_3": 3.0, "total": 13.0})
    types = [o["type"] for o in xt["objects"]]
    assert types == ["indicator", "extension-definition", "x-xtgaia-explanation"]
    meta = xt["objects"][-1]
    assert meta["within_bound"] and meta["tiers"]["structural"] == "PGExplainer" and meta["extraction"] == "parallel"


# ---------------------------------------------------------------------------- pipeline behaviour
def _pipe(tiny_checkpoint, **overrides):
    from src.pipeline import XTGAIAPipeline
    from src.role2_ai.inference import TGNInferenceEngine
    cfg = {"role1": {"asset_inventory": "configs/asset_inventory.yaml", "port_weights": "configs/mitre_t1021_ports.json",
                     "skg_window_ms": 300000, "neo4j_sync": False},
           "role2": {"micro_batch": 128, "inference_window_ms": 500},
           "role3": {"max_dispatches": 5, "zone_policies": False}, "ablation": overrides}
    return XTGAIAPipeline(cfg, engine=TGNInferenceEngine(str(tiny_checkpoint)), neo4j_driver=None)


def test_inference_windows_follow_500ms_event_time(tiny_checkpoint):
    pipe = _pipe(tiny_checkpoint)
    batches = []
    pipe._process_batch = lambda edges, labels: batches.append(len(edges))
    t = 1499440000.0
    pipe.run([flow(WS, WEB, 80, t + dt) for dt in (0.0, 0.1, 0.2, 0.6, 0.65, 2.0)])
    assert batches == [4, 2]                                     # flushed once the window spans >= 500 ms


def test_evicted_nodes_write_back_latent_state(tiny_checkpoint):
    pipe = _pipe(tiny_checkpoint)
    pipe.max_dispatches = 0                                       # eviction only (no containment side effects)
    stored = []
    pipe.graph.store_latent_states = lambda rows: stored.extend(r["ip"] for r in rows)
    pipe.graph.skg.ttl_seconds = 1.0
    t = 1499440000.0
    pipe.run([flow(WS, WEB, 80, t), flow(WS, WEB, 80, t + 0.6), flow(ATTACKER, "192.168.10.3", 445, t + 5.0)])
    assert WS in stored and WEB in stored and WS not in pipe.engine.live   # re-hydrated on return


def test_ablation_switches(tiny_checkpoint, http_calls):
    pipe = _pipe(tiny_checkpoint, risk_assessor=False, closed_loop=False)
    pipe.run([flow(ATTACKER, WEB, 22, 1499440000.0)])
    state, risk, phi = pipe.build_state(WEB, 0.8)
    assert risk == pytest.approx(0.8) and phi[:2] == [0.0, 0.0]          # raw p_t, no criticality
    assert pipe.dispatcher.mutate_enabled is False and pipe.graph.is_blocked(ATTACKER, WEB, 22) is False


# ---------------------------------------------------------------------------- audit fixes
def test_quarantined_node_is_isolated_with_zero_risk_until_rollback():
    import networkx as nx
    qm = QuarantineManager()
    g = nx.DiGraph()
    g.add_edge(ATTACKER, WEB, port=22)
    risk = {WEB: 4.0}
    qm.quarantine(g, risk, WEB)
    assert g.nodes[WEB]["state"] == "Isolated" and risk[WEB] == 0.0 and g.degree(WEB) == 0
    qm.restore(g, WEB)
    assert g.nodes[WEB]["state"] == "Active" and g.has_edge(ATTACKER, WEB)


def test_xai_stability_cosine_by_attack_class():
    from src.evaluation.xai_stability import attribution_vectors, stability
    tiers = {"structural_edges": [("a", "b", 0.9), ("c", "b", 0.4)],
             "temporal": {"temporal_trace": [0.1, 0.2, 0.7]}}
    dec = {"x_xtgaia_strategic_xai": {"q_values": {f"a{i}": float(i) for i in range(5)},
                                      "sub_reward_decomposition": {"weighted_contain": 1.0, "weighted_disrupt": -0.2,
                                                                   "weighted_cost": -0.1}}}
    v = attribution_vectors(tiers, dec)
    assert len(v["structural"]) == 5 and len(v["temporal"]) == 10 and len(v["strategic"]) == 8
    s = stability([{"label": "DDoS", "vectors": v}, {"label": "DDoS", "vectors": v}, {"label": "BENIGN", "vectors": v}])
    assert s["cosine_mean"]["structural"] == pytest.approx(1.0) and s["by_label"]["DDoS"]["n"] == 2


def test_flow_source_reads_corrected_cicids_with_microseconds(tmp_path):
    from src.role1_graph.flow_sources import CICIDSFlowSource, FLOW_FEATURES, IMPROVED_FEATURE_NAMES
    cols = ["Src IP", "Dst IP", "Dst Port", "Protocol", "Timestamp"] + \
           [IMPROVED_FEATURE_NAMES.get(c, c) for c in FLOW_FEATURES] + ["Label", "Attempted Category"]
    rows = [["10.0.0.1", "10.0.0.2", "22", "6", "2017-07-07 08:59:50.000100"] + ["1"] * 16 + ["SSH-Patator", "-1"],
            ["10.0.0.1", "10.0.0.2", "22", "6", "2017-07-07 08:59:50.040200"] + ["2"] * 16 + ["SSH-Patator", "3"]]
    p = tmp_path / "friday.csv"
    p.write_text("\n".join(",".join(r) for r in [cols] + rows))
    ev = list(CICIDSFlowSource([str(p)]))
    assert len(ev) == 2 and abs((ev[1]["timestamp"] - ev[0]["timestamp"]) - 0.0401) < 1e-6
    assert ev[0]["label"] == "SSH-Patator" and ev[1]["label"] == "BENIGN" and ev[1]["features"][0] == 2.0
