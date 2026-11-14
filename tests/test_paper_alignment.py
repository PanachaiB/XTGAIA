"""
Checks that the implementation follows the XTGAIA paper (XTGAIAv3):
  Eq. 8   S_priv(u) of the source host, S_port tiers {1.0 admin, 0.4 web, 0.1 other}
  Alg. 1  TGN latent-state persistence / recovery (512-byte float32 per node)
  Alg. 4  Neo4j soft delete on containment + reactivation on rollback; audit window
          (approve / timeout -> standard transition with Eq. 10 reward)
  Sec. III.B Phase 5  prioritized replay buffer
"""
import numpy as np
import pytest
import torch

from src.role1_graph.host_context import AssetInventory
from src.role1_graph.live_graph import LiveGraphService
from src.role1_graph.risk_engine import ReachabilityRiskEngine
from src.role3_decision.actions import QUARANTINE, RATE_LIMIT
from src.role3_decision.decision_ledger import DecisionLedger, DecisionRecord
from src.role3_decision.double_dqn import PrioritizedReplayBuffer, QNetwork, ReplayBuffer, _optimise
from src.role3_decision.feedback_rollback import FeedbackRollbackHandler
from src.role3_decision.graph_interface import QuarantineManager
from src.role3_decision.state import encode_state, make_raw_state

from tests.test_integration import FLOW_FEATURES, flow, tiny_checkpoint  # noqa: F401  (fixture reuse)


def inv():
    return AssetInventory.from_yaml("configs/asset_inventory.yaml")


# ---------------------------------------------------------------------------- Eq. 8
def test_port_tiers_match_paper():
    eng = ReachabilityRiskEngine(set(), config_path="configs/mitre_t1021_ports.json")
    assert {p: eng.port_weights[p] for p in (445, 3389, 22)} == {445: 1.0, 3389: 1.0, 22: 1.0}
    assert eng.port_weights[80] == eng.port_weights[443] == 0.4
    assert eng.default_port_weight == 0.1


def test_s_priv_is_taken_from_the_source_host():
    a = inv()
    assert (a.privilege("192.168.10.3"), a.privilege("192.168.10.50"), a.privilege("192.168.10.8"),
            a.privilege("8.8.8.8")) == (1.0, 0.6, 0.2, 0.2)
    g = LiveGraphService(a, "configs/mitre_t1021_ports.json", window_ms=60000)
    _, s_vuln, s_priv = g.edge_weight("192.168.10.3", "192.168.10.8", 445)          # DC (root) -> workstation
    assert s_priv == 1.0 and s_vuln == pytest.approx(0.81)
    w_from_ws, _, p_ws = g.edge_weight("192.168.10.8", "192.168.10.3", 445)        # workstation (user) -> DC
    assert p_ws == 0.2
    assert w_from_ws == pytest.approx(1 - (0.5 * 0.75 + 0.3 * 0.2 + 0.2 * 1.0))     # Eq. 7-8


# ---------------------------------------------------------------------------- Alg. 4 Neo4j soft delete
class Recorder:
    def __init__(self):
        self.calls = []

    def on_sever(self, host, edges, isolated):
        self.calls.append(("sever", host, len(edges), isolated))

    def on_restore(self, host, edges):
        self.calls.append(("restore", host, len(edges)))


def test_quarantine_notifies_persistence_listener_for_soft_delete_and_reactivation():
    import networkx as nx
    g, risk, qm, rec = nx.DiGraph(), {"h": 3.0}, QuarantineManager(), Recorder()
    g.add_edge("a", "h", port=445)
    g.add_edge("h", "b", port=22)
    qm.add_listener(rec)
    qm.quarantine(g, risk, "h")
    qm.restore(g, "h")
    assert rec.calls == [("sever", "h", 2, True), ("restore", "h", 2)]


# ---------------------------------------------------------------------------- Alg. 1 latent state
def test_latent_state_roundtrip_is_512_bytes_and_restores_memory(tiny_checkpoint):  # noqa: F811
    from src.role2_ai.inference import TGNInferenceEngine
    src = TGNInferenceEngine(str(tiny_checkpoint))
    edges = [{"source": "172.16.0.1", "target": "192.168.10.50", "port": 22, "timestamp": 1499440000.0 + i,
              "features": [float(i + 1)] * len(FLOW_FEATURES)} for i in range(5)]
    src.observe_batch(edges)
    rows = src.export_latent(["192.168.10.50"])
    assert len(rows) == 1 and len(rows[0]["state"]) == 512                          # 128 x float32
    dst = TGNInferenceEngine(str(tiny_checkpoint))                                  # fresh process
    assert dst.needs_hydration(["192.168.10.50"]) == ["192.168.10.50"]
    assert dst.hydrate({"192.168.10.50": (rows[0]["state"], rows[0]["ts"])}) == 1
    nid = dst.ip_to_id["192.168.10.50"]
    np.testing.assert_allclose(dst.model.memory.memory[nid].detach().numpy(),
                               src.model.memory.memory[src.ip_to_id["192.168.10.50"]].detach().numpy())
    # Algorithm 1 restores the 512-byte latent memory exactly (checked above); p_t(v) additionally
    # depends on the node's recent-interaction neighbourhood, which the paper does not persist.
    p_dst, _ = dst.score("192.168.10.50")
    assert 0.0 <= p_dst <= 1.0


# ---------------------------------------------------------------------------- Alg. 4 audit window
def _record(ledger, case_id, action, ts, s_next_risk):
    s = make_raw_state(0.9, 0.9, 1.0).to_numpy()
    s_next = make_raw_state(0.9, s_next_risk, 1.0).to_numpy()
    ledger.record(DecisionRecord(case_id=case_id, target_host="h", target_ip="h", action=action,
                                 state=s, next_state=s_next, timestamp=ts))
    return s, s_next


def test_audit_window_commits_standard_transition_on_timeout_and_approval(tmp_path):
    ledger, buf = DecisionLedger(), ReplayBuffer(100)
    h = FeedbackRollbackHandler(replay_buffer=buf, ledger=ledger, quarantine_manager=QuarantineManager(),
                                persist_path=tmp_path / "fb.npz")
    s, s_next = _record(ledger, "c1", QUARANTINE, ts=1000.0, s_next_risk=0.0)
    _record(ledger, "c2", RATE_LIMIT, ts=1100.0, s_next_risk=0.9)
    assert ledger.due_for_commit(1200.0, 300) == []                    # inside T_audit
    assert [r.case_id for r in ledger.due_for_commit(1300.0, 300)] == ["c1"]
    out = h.commit("c1", reason="timeout")
    st, a, r, s2, done, comps = buf.buffer[-1]
    e, e2 = encode_state(s), encode_state(s_next)
    expected = 1.0 * (e[9] - e2[9]) - 0.5 * (1.0 * 1.0) - 0.1 * 5        # Eq. 10, crit 1.0, a4 severity 1, cost 5
    assert out["status"] == "COMMITTED" and r == pytest.approx(expected) and a == QUARANTINE and done
    assert ledger.get("c1").committed and ledger.due_for_commit(1300.0, 300) == []
    res = h.approve("c2")                                               # no live-state hook: commits now
    assert res["reason"] == "approved" and ledger.get("c2").committed and len(buf) == 2
    with pytest.raises(ValueError):
        h.approve("c2")
    assert ReplayBuffer(10).load(tmp_path / "fb.npz") == 2              # persisted for training


# ---------------------------------------------------------------------------- PER
def test_prioritized_replay_samples_by_priority_and_updates_from_td(tmp_path):
    torch.manual_seed(0)
    np.random.seed(0)
    buf = PrioritizedReplayBuffer(capacity=8, alpha=1.0, beta_start=0.4, beta_steps=10)
    s = np.zeros(16, np.float32)
    for i in range(8):
        buf.push(s, i % 5, 0.0, s, True, (0.0, 0.0, 0.0))
    buf.update_priorities(np.arange(8), [0.0] * 7 + [100.0])            # last transition dominates
    *_, w, slots = buf.sample_prioritized(64)
    assert (slots == 7).mean() > 0.95 and float(w.max()) == pytest.approx(1.0)
    buf.push(s, 4, -100.0, s, True, (0.0, 200.0, 0.0))                  # new (e.g. rollback) -> max priority
    assert buf.priorities()[-1] == pytest.approx(100.0) and len(buf) == 8  # ring wrapped, oldest evicted
    buf.save(tmp_path / "per.npz")
    again = PrioritizedReplayBuffer(capacity=8)
    assert again.load(tmp_path / "per.npz") == 8
    np.testing.assert_allclose(again.priorities(), buf.priorities())
    online, target = QNetwork(), QNetwork()
    loss = _optimise(online, target, torch.optim.Adam(online.parameters()), buf, 8, 0.95, torch.device("cpu"))
    assert np.isfinite(loss) and buf.beta() > 0.4
