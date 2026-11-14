"""
Cross-role integration tests (offline: Shuffle/TheHive HTTP is intercepted, no Neo4j).

They exercise the real wiring: Role 1 sources/graph/risk -> Role 2 TGN inference + XAI
-> shared XTGAIAStateVector -> Role 3 dispatch -> in-place quarantine on Role 1's live
graph -> enforcement -> analyst rollback. A tiny untrained TGN checkpoint in the
production format stands in for the trained model.
"""
import csv
import datetime as dt
import json

import networkx as nx
import numpy as np
import pytest
import torch

from src.role1_graph import get_active_graph_and_risk, set_live_graph_service
from src.role1_graph.flow_sources import FLOW_FEATURES, CICIDSFlowSource, WazuhAlertSource, parse_cicids_timestamp
from src.role1_graph.host_context import AssetInventory
from src.role1_graph.live_graph import LiveGraphService
from src.role2_ai.inference import STRUCTURAL_EXPLAINER, TEMPORAL_EXPLAINER, TGNInferenceEngine
from src.role2_ai.tgn_model import XTGAIATGN
from src.role3_decision import feedback_rollback as fr
from src.role3_decision import soar_client
from src.role3_decision.actions import NUM_ACTIONS, QUARANTINE
from src.role3_decision.decision_ledger import DecisionLedger
from src.role3_decision.double_dqn import QNetwork, ReplayBuffer
from src.role3_decision.graph_interface import QuarantineManager, default_graph_provider
from src.role3_decision.soar_dispatcher import SOARDispatcher

INTERNAL = "192.168.10.0/24"
WEB, DC, WS, ATTACKER = "192.168.10.50", "192.168.10.3", "192.168.10.8", "172.16.0.1"


# ---------------------------------------------------------------------------- fixtures
class FixedQNetwork(QNetwork):
    def __init__(self, favoured: int):
        super().__init__()
        self.favoured = favoured

    def forward_decomposed(self, x):
        q = torch.zeros(x.shape[0], 3, NUM_ACTIONS)
        q[:, 0, self.favoured] = 10.0
        return q


class FakeResponse:
    def __init__(self, status, body=None):
        self.status_code, self._body = status, body or {}

    def json(self):
        return self._body


@pytest.fixture
def http_calls(monkeypatch):
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "json": json})
        return FakeResponse(201, {"_id": f"~{len(calls)}"}) if url.endswith("/api/v1/case") else FakeResponse(200)

    monkeypatch.setattr(soar_client.requests, "post", fake_post)
    monkeypatch.setenv("SHUFFLE_BASE_URL", "http://shuffle-backend:5001")
    monkeypatch.setenv("SHUFFLE_EXECUTION_WEBHOOK_URL", "webhook_exec-test")
    monkeypatch.setenv("SHUFFLE_ROLLBACK_WEBHOOK_URL", "webhook_rollback-test")
    monkeypatch.setenv("THEHIVE_URL", "http://thehive:9000")
    monkeypatch.setenv("THEHIVE_API_KEY", "unit-test-key")
    monkeypatch.delenv("ROLLBACK_WEBHOOK_TOKEN", raising=False)
    return calls


def inventory():
    return AssetInventory(crown_jewels={DC: 10.0, WEB: 8.0},
                          hosts={DC: {"is_admin": 1, "cvss_max": 7.5}, WEB: {"cvss_max": 9.8}, WS: {"cvss_max": 8.1}},
                          internal_subnets=[INTERNAL])


@pytest.fixture
def tiny_checkpoint(tmp_path):
    torch.manual_seed(0)
    known = [ATTACKER, WEB, DC, WS]
    model = XTGAIATGN(num_nodes=len(known) + 16, raw_msg_dim=len(FLOW_FEATURES))
    path = tmp_path / "tgn_tiny.pt"
    torch.save({
        "format": "xtgaia-tgn-v1", "state_dict": model.state_dict(), "num_nodes": len(known) + 16,
        "architecture": XTGAIATGN.ARCHITECTURE, "k1": model.k1, "k2": model.k2,
        "raw_msg_dim": len(FLOW_FEATURES), "memory_dim": 128, "latent_z_dim": 8, "num_heads": 8,
        "threshold": 0.5, "test_metrics": {"f1": 0.0},
        "feature_meta": {"features": FLOW_FEATURES, "means": {c: 0.0 for c in FLOW_FEATURES},
                         "stds": {c: 1.0 for c in FLOW_FEATURES}, "clip": 5.0, "t0": 1499400000.0,
                         "ip_to_id": {ip: i for i, ip in enumerate(known)}, "num_known_nodes": len(known)},
    }, path)
    return path


def flow(src, dst, port, ts, label="BENIGN"):
    return {"src_ip": src, "dst_ip": dst, "dst_port": port, "timestamp": ts, "proto": "tcp",
            "features": [1.0] * len(FLOW_FEATURES), "label": label}


# ---------------------------------------------------------------------------- Role 1 sources
def test_cicids_timestamps_are_day_first_and_afternoon_fixed():
    t = dt.datetime.fromtimestamp(parse_cicids_timestamp("4/7/2017 1:30"), dt.timezone.utc)
    assert (t.month, t.day, t.hour, t.minute) == (7, 4, 13, 30)          # 4 July, 13:30
    t = dt.datetime.fromtimestamp(parse_cicids_timestamp("03/07/2017 08:55:58"), dt.timezone.utc)
    assert (t.month, t.day, t.hour, t.second) == (7, 3, 8, 58)
    assert parse_cicids_timestamp("not a time") is None


def test_cicids_source_reads_real_column_layout_in_time_order(tmp_path):
    header = ["Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol",
              "Timestamp"] + FLOW_FEATURES + ["Label"]
    rows = [
        ["f1", ATTACKER, "5000", WEB, "22", "6", "7/7/2017 1:05"] + ["2"] * 7 + ["Infinity"] + ["1"] * 8 + ["SSH-Patator"],
        ["f2", WS, "5001", DC, "445", "6", "7/7/2017 12:59"] + ["1"] * 16 + ["BENIGN"],
    ]
    path = tmp_path / "Friday.csv"
    with open(path, "w", newline="", encoding="latin-1") as f:
        w = csv.writer(f)
        w.writerow([" " + h for h in header])          # CICIDS headers carry leading spaces
        w.writerows(rows)
    out = list(CICIDSFlowSource([str(path)]))
    assert [r["label"] for r in out] == ["BENIGN", "SSH-Patator"]          # 12:59 before 13:05
    assert out[1]["dst_port"] == 22 and out[1]["proto"] == "tcp" and len(out[1]["features"]) == 16
    assert out[1]["features"][7] == 0.0                                    # Infinity -> 0


def test_wazuh_alert_maps_to_raw_event_with_auth_failure():
    alert = {"timestamp": "2026-11-10T04:00:00.000+0000", "rule": {"id": "5716", "level": 5,
             "groups": ["syslog", "sshd", "authentication_failed"], "mitre": {"id": ["T1110"]}},
             "agent": {"ip": WEB}, "data": {"srcip": ATTACKER, "dstport": "22"}}
    raw = WazuhAlertSource.to_raw(alert)
    assert raw["src_ip"] == ATTACKER and raw["dst_ip"] == WEB and raw["dst_port"] == 22
    assert raw["auth_failure"] is True and raw["mitre"] == ["T1110"]


# ---------------------------------------------------------------------------- Role 1 graph / risk
def make_graph(**kw):
    return LiveGraphService(inventory(), "configs/mitre_t1021_ports.json", window_ms=300000,
                            host_window_s=300, auth_ports=[22, 445, 3389], **kw)


def test_live_graph_risk_follows_algorithm2_and_phi_is_raw():
    g = make_graph()
    t = 1499440000.0
    for i in range(4):
        g.ingest(flow(ATTACKER, WEB, 22, t + i))                            # repeated SSH attempts
    g.ingest(flow(WS, DC, 445, t + 5))
    assert g.compute_risk(WEB, 0.9) == pytest.approx(0.9 * 8.0)            # crown jewel: d = 0
    assert g.compute_risk(WS, 0.9) > 0.0                                   # path WS -> DC exists
    assert g.compute_risk(ATTACKER, 0.9) > 0.0 and g.compute_risk("8.8.8.8", 0.9) == 0.0
    w = g.skg.graph[WS][DC]["weight"]
    assert 0.0 < w < 1.0                                                   # exploit friction on the edge
    phi = g.phi(WEB)
    assert phi[:2] == [0.0, 1.0] and phi[2] == 1.0 and phi[3] == 3.0 and phi[5] == 9.8  # raw, schema order
    assert g.phi(DC)[0] == 1.0


def test_person1_interface_serves_the_live_graph_to_role3():
    g = make_graph()
    set_live_graph_service(g)
    try:
        graph, risk = default_graph_provider()(0.0)
        assert graph is g.skg.graph and risk is g.risk
        assert get_active_graph_and_risk(0.0)[0] is graph
    finally:
        set_live_graph_service(None)
    with pytest.raises(RuntimeError):
        get_active_graph_and_risk(0.0)


# ---------------------------------------------------------------------------- Role 2 inference
def test_tgn_engine_scores_flows_and_explains_from_model_outputs(tiny_checkpoint):
    eng = TGNInferenceEngine(str(tiny_checkpoint))
    g = make_graph()
    edges = [g.ingest(flow(ATTACKER, WEB, 22, 1499440000.0 + i)) for i in range(3)]
    edges.append(g.ingest(flow("203.0.113.9", WEB, 80, 1499440010.0)))     # host never seen in training
    probs = eng.observe_batch(edges)
    assert len(probs) == 4 and all(0.0 <= p <= 1.0 for p in probs)
    assert eng.unseen_hosts == 1
    p_t, z_t = eng.score(WEB)
    assert z_t.shape == (8,) and z_t.dtype == np.float32
    bundle = eng.explain(WEB, p_t, g.skg.graph, edges[0])
    ext = bundle["objects"][0]["x_xtgaia_xai"]
    # paper Tier 1 / Tier 2 explainers on the temporal-attention TGN
    assert ext["structural_attribution"]["explainer_model"] == "PGExplainer"
    assert ext["temporal_attribution"]["rollout_method"] == "AttentionRollout"
    for e in ext["structural_attribution"]["influential_2hop_edges"]:
        # temporal interactions are undirected in the TGN neighbourhood; the SKG holds them in flow direction
        assert g.skg.graph.has_edge(e["src"], e["dst"]) or g.skg.graph.has_edge(e["dst"], e["src"])
    assert len(ext["temporal_attribution"]["temporal_trace_weights"]) >= 1
    assert ext["trigger_flow"]["target"] == WEB


# ---------------------------------------------------------------------------- full pipeline
def test_pipeline_end_to_end_quarantine_enforcement_and_rollback(tiny_checkpoint, http_calls, tmp_path):
    from src.pipeline import XTGAIAPipeline
    eng = TGNInferenceEngine(str(tiny_checkpoint))
    eng.threshold = 0.0                                          # flag everything: deterministic wiring test
    qm, ledger = QuarantineManager(), DecisionLedger()
    cfg = {"role1": {"asset_inventory": "configs/asset_inventory.yaml", "port_weights": "configs/mitre_t1021_ports.json",
                     "skg_window_ms": 300000, "host_context_window_s": 300, "auth_ports": [22], "neo4j_sync": False},
           "role2": {"micro_batch": 4}, "role3": {"dispatch_cooldown_s": 900, "max_dispatches": 5}}
    pipe = XTGAIAPipeline(cfg, engine=eng, neo4j_driver=None)
    pipe.dispatcher = SOARDispatcher(model=FixedQNetwork(QUARANTINE), graph_provider=pipe.graph.get_active_graph_and_risk,
                                     quarantine_manager=qm, ledger=ledger)
    pipe.enforcement.ledger = ledger
    t = 1499440000.0
    events = [flow(ATTACKER, WEB, 22, t + i, "SSH-Patator") for i in range(3)] + [flow(WS, WEB, 80, t + 3)]
    report = pipe.run(events)

    # Role 3 dispatched exactly once for the web server (cooldown / already-contained after that)
    assert [d["host"] for d in report.dispatches] == [WEB]
    d = report.dispatches[0]
    assert d["action"] == "Quarantine" and d["shuffle"] and d["thehive"] and d["edges_severed"] >= 1
    shuffle = next(c for c in http_calls if c["url"].endswith("webhook_exec-test"))["json"]
    assert shuffle["evidence"]["protected_host"] == WEB and shuffle["evidence"]["counterpart"] in (ATTACKER, WS)
    assert shuffle["detection"]["threat_confidence_pt"] is not None
    assert "label" not in json.dumps(shuffle).lower().replace("labelled", "")  # ground truth never leaves
    case = next(c for c in http_calls if c["url"].endswith("/api/v1/case"))["json"]
    for section in ("Detection context (Role 1)", "Detection XAI (Role 2", "Decision XAI (Role 3"):
        assert section in case["description"]
    graph = pipe.graph.skg.graph
    assert graph.degree(WEB) == 0 and pipe.graph.risk[WEB] == 0.0      # quarantined in Role 1's live graph

    # Enforcement: further traffic to the quarantined host is blocked at ingestion
    blocked_before = pipe.graph.stats["blocked"]
    pipe.run([flow(ATTACKER, WEB, 22, t + 10), flow(WS, DC, 445, t + 11)])
    assert pipe.graph.stats["blocked"] == blocked_before + 1           # only the flow to the quarantined WEB
    # the WS -> DC flow was ingested, flagged (threshold 0) and the fixed policy contained the DC as well
    assert [d["host"] for d in report.dispatches] == [WEB, DC] and pipe.graph.risk[DC] == 0.0

    # Analyst rollback (HTTP API handler) restores the graph and lifts enforcement
    from fastapi.testclient import TestClient
    fr._handler = fr.FeedbackRollbackHandler(replay_buffer=ReplayBuffer(100), ledger=ledger, quarantine_manager=qm,
                                             graph_provider=pipe.graph.get_active_graph_and_risk,
                                             persist_path=tmp_path / "fb.npz")
    resp = TestClient(fr.app).post("/api/v1/feedback/rollback", json={"case_id": d["case_id"]})
    assert resp.status_code == 200 and resp.json()["edges_restored"] == d["edges_severed"]
    assert graph.degree(WEB) >= d["edges_severed"]
    blocked_after_rollback = pipe.graph.stats["blocked"]
    pipe.run([flow(ATTACKER, WEB, 22, t + 20)])
    assert pipe.graph.stats["blocked"] == blocked_after_rollback      # not blocked any more
    assert report.rollbacks_seen and report.rollbacks_seen[0]["host"] == WEB


def test_burn_in_baseline_raises_threshold_without_responding(tiny_checkpoint, http_calls):
    from src.pipeline import XTGAIAPipeline
    eng = TGNInferenceEngine(str(tiny_checkpoint))
    eng.threshold = 0.0                                          # a trained ω that would flag everything
    cfg = {"role1": {"asset_inventory": "configs/asset_inventory.yaml", "port_weights": "configs/mitre_t1021_ports.json",
                     "skg_window_ms": 300000, "neo4j_sync": False},
           "role2": {"micro_batch": 4, "burn_in_events": 8, "burn_in_alert_rate": 0.25},
           "role3": {"max_dispatches": 0}}
    pipe = XTGAIAPipeline(cfg, engine=eng, neo4j_driver=None)
    t = 1499440000.0
    report = pipe.run([flow(WS, WEB, 80, t + i) for i in range(8)])
    assert report.burn_in["events"] == 8 and report.flagged_flows == 0 and not http_calls   # no response during burn-in
    assert report.labelled == 0                                                              # excluded from evaluation
    assert eng.threshold == pytest.approx(report.burn_in["baseline_quantile"]) and eng.threshold > 0.0
    report = pipe.run([flow(WS, WEB, 80, t + 9 + i) for i in range(4)])
    assert report.labelled == 4                                                              # evaluated afterwards


def test_sustained_evidence_gate_blocks_isolated_spikes(tiny_checkpoint, http_calls):
    from src.pipeline import XTGAIAPipeline
    eng = TGNInferenceEngine(str(tiny_checkpoint))
    cfg = {"role1": {"asset_inventory": "configs/asset_inventory.yaml", "port_weights": "configs/mitre_t1021_ports.json",
                     "skg_window_ms": 300000, "neo4j_sync": False},
           "role2": {"micro_batch": 1, "alert_min_events": 4}, "role3": {"max_dispatches": 5}}
    pipe = XTGAIAPipeline(cfg, engine=eng, neo4j_driver=None)
    pipe.dispatcher = SOARDispatcher(model=FixedQNetwork(QUARANTINE), graph_provider=pipe.graph.get_active_graph_and_risk,
                                     quarantine_manager=QuarantineManager(), ledger=pipe.dispatcher.ledger)
    scores = iter([0.9] + [0.01] * 4 + [0.95] * 8)    # one isolated spike, later a sustained run (trend = last 8)
    eng.observe_batch = lambda edges: [eng.history[e["target"]].append((e["timestamp"], p)) or p
                                       for e, p in zip(edges, [next(scores)])]
    eng.threshold = 0.5
    pipe.run([flow(ATTACKER, WEB, 22, 1499440000.0 + i) for i in range(13)])
    assert pipe.report.suppressed["insufficient_evidence"] >= 1          # the isolated 0.9 spike did not escalate
    assert len(pipe.report.dispatches) == 1                               # the sustained run did
