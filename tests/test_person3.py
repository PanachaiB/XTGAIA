"""
tests/test_person3.py: Role 3 (Strategic RL, SOAR execution & rollback) test suite.

Everything here is mock-only: no Shuffle, TheHive, Person 1 graph engine or
Person 2 model is required. HTTP calls are intercepted, Person 1 is replaced by
a fake get_active_graph_and_risk(timestamp) provider, and states come from
XTGAIAStateVector.mock() or schema-valid hand-built vectors.

The policy behaviour grid needs the trained checkpoint (data/models/dqn_policy.pt,
gitignored) and is skipped with a clear reason if it is absent.
"""

import numpy as np
import networkx as nx
import pytest
import torch
import yaml
from fastapi.testclient import TestClient

from src.role3_decision import feedback_rollback as fr
from src.role3_decision import soar_client
from src.role3_decision.actions import ACTIONS, NUM_ACTIONS, QUARANTINE, label
from src.role3_decision.config import load_config, policy_path, reward_weights
from src.role3_decision.decision_ledger import DecisionLedger
from src.role3_decision.double_dqn import (CHANNELS, QNetwork, ReplayBuffer, double_dqn_targets, load_policy,
                                           select_action, train_double_dqn)
from src.role3_decision.feedback_rollback import FeedbackRollbackHandler
from src.role3_decision.graph_interface import QuarantineManager, default_graph_provider
from src.role3_decision.pettingzoo_env import TACTICS, XTGAIAPettingZooEnv, load_caldera_adversary
from src.role3_decision.soar_dispatcher import SOARDispatcher
from src.role3_decision.state import (IDX_CROWN_JEWEL, IDX_IS_ADMIN, IDX_RISK, PHI_FIELDS, asset_criticality,
                                      encode_state, make_raw_state, validate_state)
from src.role3_decision.strategic_xai import StrategicXAIExplainer
from src.schemas.state_vector import XTGAIAStateVector


# ============================================================================ helpers
class FixedQNetwork(QNetwork):
    """Untrained network whose containment head strongly favours one action (deterministic tests)."""

    def __init__(self, favoured: int):
        super().__init__()
        self.favoured = favoured

    def forward_decomposed(self, x):
        q = torch.zeros(x.shape[0], len(CHANNELS), NUM_ACTIONS)
        q[:, 0, self.favoured] = 10.0
        q[:, 1, :] = torch.arange(NUM_ACTIONS, dtype=torch.float32) * 0.1
        return q


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = ""

    def json(self):
        return self._payload


@pytest.fixture
def http_calls(monkeypatch):
    """Intercept every HTTP call made by the SOAR client; TheHive returns a case id."""
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json, "headers": headers or {}})
        if url.endswith("/api/v1/case"):
            return FakeResponse(201, {"_id": "~4242"})
        return FakeResponse(200)

    monkeypatch.setattr(soar_client.requests, "post", fake_post)
    return calls


@pytest.fixture
def soar_env(monkeypatch):
    monkeypatch.setenv("SHUFFLE_BASE_URL", "http://shuffle-backend:5001")
    monkeypatch.setenv("SHUFFLE_EXECUTION_WEBHOOK_URL", "webhook_exec-test")
    monkeypatch.setenv("SHUFFLE_ROLLBACK_WEBHOOK_URL", "webhook_rollback-test")
    monkeypatch.setenv("THEHIVE_URL", "http://thehive:9000")
    monkeypatch.setenv("THEHIVE_API_KEY", "unit-test-key")
    monkeypatch.delenv("SHUFFLE_WEBHOOK_URL", raising=False)
    monkeypatch.delenv("ROLLBACK_WEBHOOK_TOKEN", raising=False)


class FakePerson1:
    """Stands in for Person 1: get_active_graph_and_risk(timestamp) -> (live DiGraph, live risk dict)."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.graph.add_edge("10.0.0.7", "10.0.0.1", port=445, weight=0.3)
        self.graph.add_edge("10.0.0.7", "10.0.0.5", port=22, weight=0.5)
        self.graph.add_edge("10.0.0.9", "10.0.0.7", port=3389, weight=0.4)
        self.graph.add_edge("10.0.0.5", "10.0.0.1", port=1433, weight=0.2)
        self.risk = {"10.0.0.7": 3.2, "10.0.0.1": 0.4, "10.0.0.5": 1.1}
        self.calls = []

    def __call__(self, timestamp):
        self.calls.append(timestamp)
        return self.graph, self.risk


def critical_state():
    return make_raw_state(0.95, 0.93, 1.0, gnn_embedding=np.random.normal(0, 1, 8),
                          conn_count=30, auth_fails=6, cpu_spike=0.8, cvss_max=9.8)


# ============================================================================ state contract
def test_mock_state_is_valid_and_encodes_to_unit_interval():
    sv = XTGAIAStateVector.mock()
    raw = validate_state(sv)
    assert raw.shape == (16,) and raw.dtype == np.float32
    enc = encode_state(sv)
    assert enc.shape == (16,) and enc.dtype == np.float32
    assert np.all((enc >= 0.0) & (enc <= 1.0))


def test_phi_order_matches_shared_schema():
    assert PHI_FIELDS == ["is_admin", "is_crown_jewel", "conn_count", "auth_fails", "cpu_spike", "cvss_max"]
    enc = encode_state(make_raw_state(0.5, 0.5, 0.7))          # crown jewel, not admin
    assert enc[IDX_IS_ADMIN] == 0.0 and enc[IDX_CROWN_JEWEL] == 1.0
    assert asset_criticality(enc) == 0.7                       # exact: env tiers compare with <=
    for crit in (0.1, 0.4, 0.7, 1.0):
        assert asset_criticality(encode_state(make_raw_state(0.5, 0.5, crit))) == crit


@pytest.mark.parametrize("bad, reason", [
    (np.zeros(15, dtype=np.float32), "shape"),
    (np.zeros((1, 16), dtype=np.float32), "shape"),
    (np.zeros(16, dtype=np.float64), "dtype"),
    (np.full(16, np.nan, dtype=np.float32), "NaN"),
    (np.r_[np.zeros(8), 0.5, np.inf, np.zeros(6)].astype(np.float32), "inf"),
    (np.r_[np.zeros(8), 1.5, 0.5, np.zeros(6)].astype(np.float32), "p_t > 1"),
    (np.r_[np.zeros(8), 0.5, -1.0, np.zeros(6)].astype(np.float32), "negative risk"),
    (np.r_[np.zeros(8), 0.5, 0.5, 0.5, 0, 0, 0, 0, 0].astype(np.float32), "non-binary is_admin"),
    (np.r_[np.zeros(8), 0.5, 0.5, 0, 0, 0, 0, 0, 11.0].astype(np.float32), "cvss > 10"),
    ([0.0] * 16, "python list"),
])
def test_invalid_states_are_rejected(bad, reason):
    with pytest.raises(ValueError):
        validate_state(bad)


def test_schema_object_with_wrong_embedding_shape_is_rejected():
    sv = XTGAIAStateVector(np.zeros(7, np.float32), 0.5, 0.5, np.zeros(6, np.float32))
    with pytest.raises(ValueError):
        validate_state(sv)


# ============================================================================ action space
def test_action_space_is_exactly_the_five_playbooks():
    assert NUM_ACTIONS == 5
    assert [ACTIONS[i].name for i in range(5)] == ["NoOp", "RateLimit", "PortIsolation", "RevokeSession", "Quarantine"]
    assert len({a.playbook for a in ACTIONS.values()}) == 5
    env = XTGAIAPettingZooEnv()
    assert env.action_space("defender").n == 5
    assert QNetwork()(torch.zeros(1, 16)).shape == (1, 5)


# ============================================================================ config
def test_reward_weights_are_loaded_from_yaml():
    assert reward_weights() == {"contain": 1.0, "disrupt": 0.5, "cost": 0.1, "rollback": -100.0}
    assert str(policy_path()).replace("\\", "/").endswith("data/models/dqn_policy.pt")


def test_config_override_changes_weights(tmp_path, monkeypatch):
    cfg = dict(load_config())
    cfg["reward_weights"] = {"contain": 2.0, "disrupt": 0.25, "cost": 0.05, "rollback": -50.0}
    path = tmp_path / "weights.yaml"
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("XTGAIA_REWARD_CONFIG", str(path))
    assert reward_weights()["rollback"] == -50.0
    assert XTGAIAPettingZooEnv().w_contain == 2.0


# ============================================================================ Double DQN
def test_qnetwork_total_equals_weighted_heads():
    net = QNetwork()
    x = torch.rand(4, 16)
    heads = net.forward_decomposed(x)
    assert heads.shape == (4, 3, 5)
    w = reward_weights()
    manual = w["contain"] * heads[:, 0] - w["disrupt"] * heads[:, 1] - w["cost"] * heads[:, 2]
    assert torch.allclose(net(x), manual, atol=1e-6)


def test_double_dqn_target_selects_with_online_and_evaluates_with_target():
    online, target = FixedQNetwork(favoured=2), FixedQNetwork(favoured=4)
    s_next = torch.rand(3, 16)
    comps = torch.ones(3, 3)
    done = torch.tensor([0.0, 0.0, 1.0])
    y = double_dqn_targets(online, target, comps, s_next, done, gamma=0.9)
    # online picks a2; target's heads at a2 are (0, 0.2, 0): not target's own argmax (a4 -> 10.0)
    expected_live = torch.tensor([1.0, 1.0 + 0.9 * 0.2, 1.0])
    assert torch.allclose(y[0], expected_live) and torch.allclose(y[1], expected_live)
    assert torch.allclose(y[2], torch.ones(3))                     # terminal: no bootstrap


def test_replay_buffer_roundtrip(tmp_path):
    buf = ReplayBuffer(100)
    for i in range(5):
        buf.push(np.full(16, i / 10, np.float32), i % 5, float(i), np.zeros(16, np.float32), i == 4, (i, 0, 0))
    buf.save(tmp_path / "d.npz")
    other = ReplayBuffer(100)
    assert other.load(tmp_path / "d.npz") == 5
    s, a, r, s2, d, c = other.sample(5)
    assert s.shape == (5, 16) and c.shape == (5, 3)


def test_training_runs_on_simulator_and_saves_policy(tmp_path):
    out = train_double_dqn(episodes=15, save_path=tmp_path / "dqn_policy.pt", replay_buffer=ReplayBuffer(5000),
                           seed=0, verbose=False, load_feedback=False)
    assert out.exists()
    model = load_policy(out)
    assert 0 <= select_action(XTGAIAStateVector.mock(), weights_path=out) < 5
    assert model(torch.as_tensor(encode_state(XTGAIAStateVector.mock())).unsqueeze(0)).shape == (1, 5)


# ============================================================================ PettingZoo env
def test_env_episode_produces_valid_observations_and_decomposed_rewards():
    env = XTGAIAPettingZooEnv()
    env.reset(seed=1)
    w = reward_weights()
    for agent in env.agent_iter(max_iter=200):
        obs, reward, term, trunc, info = env.last()
        assert obs.shape == (16,) and obs.dtype == np.float32 and np.all((obs >= 0) & (obs <= 1))
        if agent == "defender":
            c = info["reward_components"]
            assert reward == pytest.approx(w["contain"] * c[0] - w["disrupt"] * c[1] - w["cost"] * c[2], abs=1e-4)
        if term or trunc:
            action = None
        elif agent == "attacker":
            action = env.scripted_attacker_action()
        else:
            action = env.action_space("defender").sample()
        env.step(action)


def _one_defender_step(p, risk, crit, action):
    env = XTGAIAPettingZooEnv()
    env.reset(seed=0)
    env.p_t, env.risk_t, env.crit_tier = p, risk, crit
    env._update_observations()
    env.step(action)
    return env.rewards["defender"], env.terminations["defender"]


def test_playbook_bonus_only_paid_on_containing_step():
    # Critical crown jewel at risk 0.93: a3 cannot contain in one step (0.93 - 0.5 > 0.2) -> no bonus,
    # a4 contains -> bonus. Quarantine is preferred only because nothing lighter can contain.
    r_a3, done_a3 = _one_defender_step(0.95, 0.93, 1.0, 3)
    r_a4, done_a4 = _one_defender_step(0.95, 0.93, 1.0, QUARANTINE)
    assert not done_a3 and done_a4 and r_a4 > r_a3 and r_a3 < 1.0
    # Medium-risk admin host: a2 contains in one step and keeps its bonus, beating quarantine.
    r_a2, done_a2 = _one_defender_step(0.40, 0.35, 0.4, 2)
    r_q, _ = _one_defender_step(0.40, 0.35, 0.4, QUARANTINE)
    assert done_a2 and r_a2 > 3.0 and r_a2 > r_q


def test_caldera_adversary_profile_loader(tmp_path):
    adv = tmp_path / "adversary.yml"
    adv.write_text(yaml.safe_dump({"name": "test", "atomic_ordering": ["ab1", "ab2", "ab3", "unknown"]}))
    chain = load_caldera_adversary(str(adv), {"ab1": "discovery", "ab2": "credential-access",
                                              "ab3": "lateral-movement"})
    assert chain == ["discovery", "credential-access", "lateral-movement"]
    assert all(t in TACTICS for t in chain)


# ============================================================================ Person 1 interface + topology
def test_quarantine_severs_in_and_out_edges_in_ram_and_resets_risk():
    p1, qm = FakePerson1(), QuarantineManager()
    graph, risk = p1(123.0)
    severed = qm.quarantine(graph, risk, "10.0.0.7")
    assert len(severed) == 3
    assert graph.degree("10.0.0.7") == 0                       # same object Person 1 holds
    assert p1.risk["10.0.0.7"] == 0.0
    assert graph.has_edge("10.0.0.5", "10.0.0.1")              # unrelated edges untouched
    restored = qm.restore(graph, "10.0.0.7")
    assert len(restored) == 3 and graph["10.0.0.7"]["10.0.0.1"]["port"] == 445
    assert qm.pop_prior_risk("10.0.0.7") == 3.2 and qm.pop_prior_risk("10.0.0.7") is None  # snapshot, consumed once


def test_port_isolation_only_removes_matching_port():
    p1, qm = FakePerson1(), QuarantineManager()
    qm.isolate_port(p1.graph, p1.risk, "10.0.0.7", 445)
    assert not p1.graph.has_edge("10.0.0.7", "10.0.0.1") and p1.graph.has_edge("10.0.0.7", "10.0.0.5")
    assert p1.risk["10.0.0.7"] == 3.2


def test_default_provider_fails_clearly_until_person1_exposes_interface():
    import src.role1_graph as role1
    if hasattr(role1, "get_active_graph_and_risk"):
        assert callable(default_graph_provider())
    else:
        with pytest.raises(RuntimeError, match="get_active_graph_and_risk"):
            default_graph_provider()


# ============================================================================ SOAR dispatcher
def test_dispatcher_quarantine_pipeline(http_calls, soar_env):
    p1 = FakePerson1()
    d = SOARDispatcher(model=FixedQNetwork(QUARANTINE), graph_provider=p1,
                       quarantine_manager=QuarantineManager(), ledger=DecisionLedger())
    res = d.process_alert(critical_state(), "10.0.0.7", "10.0.0.7", timestamp=1700000000.0)

    assert res["action"] == QUARANTINE and res["playbook"] == ACTIONS[QUARANTINE].playbook
    shuffle = http_calls[0]
    assert shuffle["url"] == "http://shuffle-backend:5001/api/v1/hooks/webhook_exec-test"
    assert shuffle["json"]["playbook_name"] == "xtgaia_a4_quarantine" and shuffle["json"]["action_code"] == 4
    hive = http_calls[1]
    assert hive["url"] == "http://thehive:9000/api/v1/case"
    assert hive["headers"]["Authorization"] == "Bearer unit-test-key"
    assert res["case_id"] == "~4242" and res["thehive_logged"] and res["soar_dispatched"]
    assert p1.calls == [1700000000.0]                          # Person 1 used only via the agreed interface
    assert res["edges_severed"] == 3 and p1.risk["10.0.0.7"] == 0.0
    rec = d.ledger.get("~4242")
    assert rec.action == QUARANTINE and rec.next_state[IDX_RISK] == 0.0


def test_dispatcher_noop_sends_nothing(http_calls, soar_env):
    d = SOARDispatcher(model=FixedQNetwork(0), graph_provider=FakePerson1(),
                       quarantine_manager=QuarantineManager(), ledger=DecisionLedger())
    res = d.process_alert(make_raw_state(0.05, 0.05, 0.1), "10.0.0.7", "10.0.0.7")
    assert res["action"] == 0 and http_calls == []


def test_dispatcher_fails_closed_without_configuration(http_calls, monkeypatch):
    for var in ("SHUFFLE_EXECUTION_WEBHOOK_URL", "SHUFFLE_WEBHOOK_URL", "THEHIVE_URL", "THEHIVE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    d = SOARDispatcher(model=FixedQNetwork(QUARANTINE), graph_provider=FakePerson1(),
                       quarantine_manager=QuarantineManager(), ledger=DecisionLedger())
    res = d.process_alert(critical_state(), "10.0.0.7", "10.0.0.7")
    assert http_calls == [] and not res["soar_dispatched"] and not res["thehive_logged"]
    assert res["case_id"].startswith("local-")                 # still traceable for rollback


def test_dispatcher_rejects_invalid_state(http_calls, soar_env):
    d = SOARDispatcher(model=FixedQNetwork(QUARANTINE), graph_provider=FakePerson1(),
                       quarantine_manager=QuarantineManager(), ledger=DecisionLedger())
    with pytest.raises(ValueError):
        d.process_alert(np.full(16, np.nan, np.float32), "10.0.0.7", "10.0.0.7")


# ============================================================================ Strategic XAI
def test_xai_decomposition_is_exact_and_consistent_with_policy():
    model = QNetwork()
    xai = StrategicXAIExplainer(model=model)
    state = XTGAIAStateVector.mock()
    exp = xai.explain(state, "10.0.0.7")
    with torch.no_grad():
        q = model(torch.as_tensor(encode_state(state)).unsqueeze(0))[0].numpy()
    for a in range(5):
        assert exp["per_action"][f"a{a}"]["Q_total"] == pytest.approx(float(q[a]), abs=1e-3)
        row = exp["per_action"][f"a{a}"]
        assert {"Q_contain", "Q_disrupt", "Q_cost"} <= set(row)
    assert exp["chosen_action"] == int(q.argmax())
    assert "10.0.0.7" in exp["narrative"] and label(exp["chosen_action"]) in exp["narrative"]


def test_xai_uses_real_state_features():
    exp = StrategicXAIExplainer(model=FixedQNetwork(QUARANTINE)).explain(critical_state(), "dc01")
    ctx = exp["context"]
    assert ctx["is_crown_jewel"] and ctx["is_admin"] and ctx["asset_criticality"] == 1.0
    assert ctx["auth_fails"] == 6.0 and ctx["cvss_max"] == 9.8
    assert exp["decisive_channel"] == "contain"


# ============================================================================ feedback & rollback
def test_independent_blueprint_flow_mock_alert_then_rollback_injects_penalty(http_calls, soar_env, tmp_path,
                                                                           monkeypatch):
    """Blueprint independent test: mock alert -> a4 quarantine -> TheHive Rollback -> R=-100 tuple in D."""
    p1, qm, ledger, buffer = FakePerson1(), QuarantineManager(), DecisionLedger(), ReplayBuffer(1000)
    dispatcher = SOARDispatcher(model=FixedQNetwork(QUARANTINE), graph_provider=p1,
                                quarantine_manager=qm, ledger=ledger)
    alert_state = XTGAIAStateVector.mock()
    result = dispatcher.process_alert(alert_state, "10.0.0.7", "10.0.0.7", timestamp=42.0)
    assert result["action"] == QUARANTINE and p1.graph.degree("10.0.0.7") == 0 and p1.risk["10.0.0.7"] == 0.0

    handler = FeedbackRollbackHandler(replay_buffer=buffer, ledger=ledger, quarantine_manager=qm,
                                      graph_provider=p1, persist_path=tmp_path / "feedback.npz")
    monkeypatch.setattr(fr, "_handler", handler)
    resp = TestClient(fr.app).post("/api/v1/feedback/rollback", json={"case_id": result["case_id"]})
    assert resp.status_code == 200, resp.text
    body = resp.json()

    assert body["penalty_reward_injected"] == -100.0 and body["soar_reverted"] and body["edges_restored"] == 3
    assert http_calls[-1]["url"] == "http://shuffle-backend:5001/api/v1/hooks/webhook_rollback-test"
    assert http_calls[-1]["json"]["rollback_command"] == "UNQUARANTINE_HOST"
    assert p1.graph.degree("10.0.0.7") == 3                                 # unquarantined in RAM
    assert p1.risk["10.0.0.7"] == 3.2                                       # exact pre-quarantine risk restored

    s, a, r, s_next, done, comps = buffer.buffer[-1]
    assert r == -100.0 and a == QUARANTINE and done is True
    np.testing.assert_allclose(s, encode_state(alert_state), atol=1e-6)    # S_t
    assert s_next[IDX_RISK] == 0.0                                          # S_t+1 after quarantine
    w = reward_weights()
    assert w["contain"] * comps[0] - w["disrupt"] * comps[1] - w["cost"] * comps[2] == pytest.approx(-100.0)
    assert ReplayBuffer(10).load(tmp_path / "feedback.npz") == 1            # persisted for training

    again = TestClient(fr.app).post("/api/v1/feedback/rollback", json={"case_id": result["case_id"]})
    assert again.status_code == 400                                         # no double penalty


def test_rollback_with_explicit_vectors_and_token(http_calls, soar_env, tmp_path, monkeypatch):
    buffer = ReplayBuffer(100)
    handler = FeedbackRollbackHandler(replay_buffer=buffer, ledger=DecisionLedger(),
                                      quarantine_manager=QuarantineManager(), persist=False)
    monkeypatch.setattr(fr, "_handler", handler)
    monkeypatch.setenv("ROLLBACK_WEBHOOK_TOKEN", "t0ken")
    s_t = validate_state(XTGAIAStateVector.mock()).tolist()
    payload = {"case_id": "ext-1", "target_host": "h", "target_ip": "1.2.3.4", "executed_action": 2,
               "state_vector": s_t, "next_state_vector": s_t}
    client = TestClient(fr.app)
    assert client.post("/api/v1/feedback/rollback", json=payload).status_code == 401
    ok = client.post("/api/v1/feedback/rollback", json=payload, headers={"X-XTGAIA-Token": "t0ken"})
    assert ok.status_code == 200 and buffer.buffer[-1][2] == -100.0
    bad = dict(payload, state_vector=[0.0] * 15)
    assert client.post("/api/v1/feedback/rollback", json=bad, headers={"X-XTGAIA-Token": "t0ken"}).status_code == 400


# ============================================================================ trained policy behaviour
POLICY_GRID = [
    # (name, p_t, normalised risk, criticality, allowed actions)
    ("Low risk, standard host", 0.05, 0.05, 0.1, [0]),
    ("Low risk, admin host", 0.08, 0.08, 0.4, [0]),
    ("Low risk, crown jewel", 0.10, 0.08, 1.0, [0]),
    ("Medium risk, standard host", 0.35, 0.30, 0.1, [0, 1, 2]),
    ("Medium risk, admin host", 0.40, 0.35, 0.4, [1, 2, 3]),
    ("Medium risk, crown jewel", 0.45, 0.40, 1.0, [1, 2, 3]),
    ("High risk, standard host", 0.70, 0.65, 0.1, [2, 3, 4]),
    ("High risk, admin host", 0.75, 0.70, 0.4, [2, 3, 4]),
    ("High risk, crown jewel", 0.78, 0.75, 1.0, [2, 3, 4]),
    ("Critical, standard host", 0.90, 0.85, 0.1, [2, 3, 4]),
    ("Critical, admin host", 0.92, 0.88, 0.4, [2, 3, 4]),
    ("Critical, crown jewel", 0.95, 0.92, 1.0, [2, 3, 4]),
]


@pytest.mark.skipif(not policy_path().exists(), reason="trained policy missing: run python -m src.role3_decision.double_dqn")
@pytest.mark.parametrize("name, p_t, risk, crit, allowed", POLICY_GRID)
def test_trained_policy_scales_response_with_risk_and_criticality(name, p_t, risk, crit, allowed):
    action = select_action(make_raw_state(p_t, risk, crit))
    assert action in allowed, f"{name}: chose {label(action)}, allowed {allowed}"


@pytest.mark.skipif(not policy_path().exists(), reason="trained policy missing")
def test_trained_policy_and_xai_on_mock_state():
    sv = XTGAIAStateVector.mock()                    # unnormalised values, like live Person 1/2 output
    action = select_action(sv)
    exp = StrategicXAIExplainer().explain(sv, "10.0.0.1")
    assert exp["chosen_action"] == action
