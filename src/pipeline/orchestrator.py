"""
XTGAIA orchestrator: wires the three roles into one streaming system.

Per micro-batch of real events:
  Role 1  raw event -> STIX active edge -> live SKG (+ Neo4j) -> host context
  Role 2  TGN memory update -> p_t per flow (calibrated threshold from training)
  Role 1  for every flagged host: Risk_t = Algorithm 2 with Role 2's p_t
  Shared  XTGAIAStateVector(Z_t, p_t, Risk_t, Phi_t)  (src/schemas/state_vector.py)
  Role 3  Double DQN action -> strategic XAI -> Shuffle -> TheHive case (with Role 1
          evidence and Role 2 detection XAI) -> in-place topology change on Role 1's graph
  Loop    containment is enforced at ingestion until an analyst rolls it back through
          the Role 3 rollback API (penalty transition goes to the replay buffer)

Dataset labels (when a source provides them) are used only for the evaluation
report; they never reach the models, the SOAR payload or TheHive.
"""

import logging
import os
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

from src.role1_graph.host_context import AssetInventory
from src.role1_graph.live_graph import LiveGraphService, set_live_graph_service
from src.role2_ai.inference import TGNInferenceEngine, set_inference_engine
from src.role3_decision.actions import ACTIONS, NOOP, PORT_ISOLATION, QUARANTINE
from src.role3_decision.config import REPO_ROOT, load_config, resolve_path
from src.role3_decision.soar_dispatcher import SOARDispatcher
from src.schemas.state_vector import XTGAIAStateVector

logger = logging.getLogger("XTGAIA-Pipeline")

PHI_NAMES = ["is_admin", "is_crown_jewel", "conn_count", "auth_fails", "cpu_spike", "cvss_max"]


def load_pipeline_config(path: Optional[str] = None) -> Dict[str, Any]:
    p = resolve_path(path or os.getenv("XTGAIA_PIPELINE_CONFIG", "configs/pipeline.yaml"))
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def neo4j_driver_from_env():
    """Neo4j driver from .env (NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD) or None."""
    uri, user, pwd = os.getenv("NEO4J_URI"), os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")
    if not (uri and user and pwd):
        logger.warning("NEO4J_URI/USER/PASSWORD not set; persistent graph sync disabled")
        return None
    try:
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(uri, auth=(user, pwd))
        driver.verify_connectivity()
        return driver
    except Exception as exc:   # detection must keep running without persistence
        logger.warning("Neo4j unavailable (%s); persistent graph sync disabled", exc)
        return None


@dataclass
class Containment:
    case_id: str
    action: int
    port: Optional[int]
    since: float


class Enforcement:
    """Network effect of Role 3's playbooks on the event stream: a quarantined host
    sends/receives nothing, a port-isolated host loses that port. Lifted when the
    analyst rolls the case back (DecisionLedger.rolled_back)."""

    def __init__(self, ledger):
        self.ledger = ledger
        self.active: Dict[str, List[Containment]] = defaultdict(list)

    def add(self, host: str, c: Containment) -> None:
        if c.action in (QUARANTINE, PORT_ISOLATION):
            self.active[host].append(c)

    def refresh(self) -> List[Tuple[str, Containment]]:
        lifted = []
        for host, items in list(self.active.items()):
            keep = []
            for c in items:
                rec = self.ledger.get(c.case_id)
                (lifted.append((host, c)) if rec is not None and rec.rolled_back else keep.append(c))
            if keep:
                self.active[host] = keep
            else:
                self.active.pop(host, None)
        return lifted

    def _blocks_host(self, host: str, port: int) -> bool:
        for c in self.active.get(host, ()):
            if c.action == QUARANTINE or (c.action == PORT_ISOLATION and c.port == port):
                return True
        return False

    def blocks(self, src: str, dst: str, port: int) -> bool:
        return self._blocks_host(src, port) or self._blocks_host(dst, port)



@dataclass
class RunReport:
    events: int = 0
    edges: int = 0
    flagged_flows: int = 0
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    labelled: int = 0
    attack_flows_by_label: Counter = field(default_factory=Counter)
    first_seen_by_label: Dict[str, float] = field(default_factory=dict)      # evaluation only
    detected_by_label: Counter = field(default_factory=Counter)
    policy_actions: Counter = field(default_factory=Counter)
    suppressed: Counter = field(default_factory=Counter)
    dispatches: List[Dict[str, Any]] = field(default_factory=list)
    rollbacks_seen: List[Dict[str, Any]] = field(default_factory=list)
    batch_ms: List[float] = field(default_factory=list)
    burn_in: Dict[str, Any] = field(default_factory=dict)
    started: float = field(default_factory=time.time)

    def detection_metrics(self) -> Dict[str, float]:
        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        fpr = self.fp / (self.fp + self.tn) if self.fp + self.tn else 0.0
        return {"precision": p, "recall": r, "f1": f1, "fpr": fpr}


class XTGAIAPipeline:
    def __init__(self, config: Optional[Dict[str, Any]] = None, dispatcher: Optional[SOARDispatcher] = None,
                 engine: Optional[TGNInferenceEngine] = None, neo4j_driver: Any = "env"):
        self.cfg = config or load_pipeline_config()
        r1, r2, r3 = self.cfg.get("role1", {}), self.cfg.get("role2", {}), self.cfg.get("role3", {})

        # Role 2: trained TGN (fails loudly if not trained yet)
        self.engine = engine or TGNInferenceEngine(str(resolve_path(r2.get("checkpoint", "data/models/tgn_best.pt"))))
        set_inference_engine(self.engine)
        self.micro_batch = int(r2.get("micro_batch", 128))
        # Unlabelled baseline on the live stream: detection runs, no response, then the
        # alert threshold becomes max(trained ω, (1 - alert_rate) quantile of baseline scores).
        self.burn_in_remaining = int(r2.get("burn_in_events", 0))
        self.burn_in_alert_rate = float(r2.get("burn_in_alert_rate", 0.01))
        self._burn_scores: List[float] = []
        self._burn_attacks = 0
        self._burn_host_scores: Dict[str, List[float]] = defaultdict(list)
        self.burn_in_min_host_events = int(r2.get("burn_in_min_host_events", 50))
        self.host_threshold: Dict[str, float] = {}
        # Sustained-evidence gate: a flagged host reaches Role 3 only if the mean of its
        # recent TGN scores (Role 2 temporal trace) is also above the threshold.
        self.alert_min_events = int(r2.get("alert_min_events", 1))

        # Role 1: live graph published through the Person 1 interface
        inventory = AssetInventory.from_yaml(str(resolve_path(r1.get("asset_inventory", "configs/asset_inventory.yaml"))))
        self.inventory = inventory
        driver = neo4j_driver_from_env() if neo4j_driver == "env" and r1.get("neo4j_sync", True) else \
            (None if neo4j_driver == "env" else neo4j_driver)
        self.graph = LiveGraphService(
            inventory, str(resolve_path(r1.get("port_weights", "configs/mitre_t1021_ports.json"))),
            window_ms=float(r1.get("skg_window_ms", 50.0)), host_window_s=float(r1.get("host_context_window_s", 300)),
            auth_ports=r1.get("auth_ports", []), neo4j_driver=driver,
            neo4j_batch_size=int(r1.get("neo4j_batch_size", 500)))
        set_live_graph_service(self.graph)

        # Role 3: dispatcher mutating Role 1's live graph
        self.dispatcher = dispatcher or SOARDispatcher(graph_provider=self.graph.get_active_graph_and_risk)
        self.enforcement = Enforcement(self.dispatcher.ledger)
        self._ledger_version = getattr(self.dispatcher.ledger, "version", None)
        self.graph.is_blocked = self.enforcement.blocks
        self.severity = {int(k[1:]): float(v) for k, v in load_config()["action_severities"].items()}
        self.cooldown_s = float(r3.get("dispatch_cooldown_s", 900))
        self.max_dispatches = int(r3.get("max_dispatches", 25))
        # Alerting vs acting: Role 2's threshold drives detection/XAI; autonomous containment
        # additionally requires the model to call an attack more likely than not.
        self.min_response_p = float(r3.get("min_response_p", 0.0))
        self._last_dispatch: Dict[str, Tuple[float, int]] = {}
        self.report = RunReport()
        self._api_server = None

    # ------------------------------------------------------------------ helpers
    def target_of(self, edge: Dict[str, Any]) -> Tuple[str, str]:
        """(protected host, counterpart): the internal endpoint of a flagged flow is
        the asset to protect (inbound attack -> destination, compromised host calling
        out -> source)."""
        src, dst = edge["source"], edge["target"]
        if self.inventory.is_internal(dst) or not self.inventory.is_internal(src):
            return dst, src
        return src, dst

    def build_state(self, host: str, p_t: float) -> Tuple[XTGAIAStateVector, float, List[float]]:
        _, z_t = self.engine.score(host)                       # Role 2: Z_t of the protected host
        risk = self.graph.compute_risk(host, p_t)              # Role 1: Algorithm 2 with Role 2's p_t
        phi = self.graph.phi(host)                             # Role 1: host context
        state = XTGAIAStateVector(gnn_embedding=np.asarray(z_t, dtype=np.float32),
                                  threat_probability=float(min(max(p_t, 0.0), 1.0)),
                                  topological_risk=float(max(risk, 0.0)),
                                  telemetry_flags=np.asarray(phi, dtype=np.float32))
        return state, risk, phi

    def _suppression(self, host: str, action: int, ts: float) -> Optional[str]:
        if action == NOOP:
            return "policy_noop"
        active = self.enforcement.active.get(host)
        if active and max(self.severity[c.action] for c in active) >= self.severity[action]:
            return "already_contained"
        last = self._last_dispatch.get(host)
        if last and ts - last[0] < self.cooldown_s and self.severity[action] <= self.severity[last[1]]:
            return "cooldown"
        if len(self.report.dispatches) >= self.max_dispatches:
            return "dispatch_cap"
        return None

    # ------------------------------------------------------------------ one micro-batch
    def _finish_burn_in(self) -> None:
        trained = self.engine.threshold
        baseline = float(np.quantile(self._burn_scores, 1.0 - self.burn_in_alert_rate)) if self._burn_scores else trained
        self.engine.threshold = max(trained, baseline)
        # Per-host baselines (UEBA style) for hosts observed often enough: a busy crown jewel
        # has a higher benign score level than a workstation. Never below the global threshold.
        q = 1.0 - self.burn_in_alert_rate
        self.host_threshold = {h: max(self.engine.threshold, float(np.quantile(sc, q)))
                               for h, sc in self._burn_host_scores.items() if len(sc) >= self.burn_in_min_host_events}
        raised = sorted(((h, t) for h, t in self.host_threshold.items() if t > self.engine.threshold),
                        key=lambda x: x[1], reverse=True)
        self.report.burn_in = {
            "host_baselines": len(self.host_threshold),
            "host_thresholds_above_global": {h: round(t, 4) for h, t in raised[:10]},
            "events": len(self._burn_scores), "alert_rate": self.burn_in_alert_rate,
            "trained_threshold": trained, "baseline_quantile": baseline, "threshold": self.engine.threshold,
            "attack_flows_inside_window": self._burn_attacks,   # disclosed only; labels are not used
        }
        logger.info("Burn-in complete: threshold %.4f -> %.4f", trained, self.engine.threshold)

    def _process_batch(self, edges: List[Dict[str, Any]], labels: List[Optional[str]]) -> None:
        t0 = time.perf_counter()
        if self.burn_in_remaining > 0:                                             # baseline: Role 1 + 2 only
            scores = self.engine.observe_batch(edges)
            self._burn_scores.extend(scores)
            for edge, p in zip(edges, scores):
                self._burn_host_scores[self.target_of(edge)[0]].append(p)
            self._burn_attacks += sum(1 for l in labels if l and l.upper() != "BENIGN")
            self.burn_in_remaining -= len(edges)
            if self.burn_in_remaining <= 0:
                self._finish_burn_in()
            return
        probs = self.engine.observe_batch(edges)                                   # Role 2
        thr = self.engine.threshold
        flagged: Dict[str, Tuple[float, Dict[str, Any], str]] = {}
        for edge, p, label in zip(edges, probs, labels):
            host, _ = self.target_of(edge)
            hit = p >= self.host_threshold.get(host, thr)
            self.report.flagged_flows += int(hit)
            if label is not None:
                attack = label.upper() != "BENIGN"
                self.report.labelled += 1
                self.report.tp += int(hit and attack)
                self.report.fp += int(hit and not attack)
                self.report.tn += int(not hit and not attack)
                self.report.fn += int(not hit and attack)
                if attack:
                    self.report.attack_flows_by_label[label] += 1
                    self.report.first_seen_by_label.setdefault(label, float(edge["timestamp"]))
                    self.report.detected_by_label[label] += int(hit)
            if hit:
                if host not in flagged or p > flagged[host][0]:
                    flagged[host] = (p, edge, label or "")
        for host, (p_t, edge, label) in flagged.items():
            mean_p, n = self.engine.trend(host)
            if n < self.alert_min_events or mean_p < self.host_threshold.get(host, thr):
                self.report.suppressed["insufficient_evidence"] += 1
                continue
            if p_t < self.min_response_p or mean_p < self.min_response_p:
                self.report.suppressed["below_response_threshold"] += 1     # alert only, no autonomous action
                continue
            self._decide(host, p_t, edge, label)
        self.report.batch_ms.append((time.perf_counter() - t0) * 1000.0)

    def _decide(self, host: str, p_t: float, edge: Dict[str, Any], label: str) -> None:
        state, risk, phi = self.build_state(host, p_t)
        action, q = self.dispatcher.predict_action(state)
        self.report.policy_actions[ACTIONS[action].name] += 1
        ts = float(edge["timestamp"])
        why = self._suppression(host, action, ts)
        if why:
            self.report.suppressed[why] += 1
            return
        _, counterpart = self.target_of(edge)
        nearest = self.graph.nearest_crown_jewel(host)
        evidence = {
            "protected_host": host, "counterpart": counterpart,
            "trigger_flow": {"source": edge["source"], "target": edge["target"], "port": edge["port"],
                             "timestamp": ts, "stix_bundle_id": edge.get("stix_bundle", {}).get("id")},
            "threat_probability_pt": round(p_t, 4),
            "decision_threshold": round(self.host_threshold.get(host, self.engine.threshold), 4),
            "topological_risk": round(risk, 4), "host_context": dict(zip(PHI_NAMES, phi)),
            "nearest_crown_jewel": None if nearest is None else {"host": nearest[0], "friction_distance": round(nearest[1], 4)},
            "graph": {"nodes": self.graph.skg.graph.number_of_nodes(), "edges": self.graph.skg.graph.number_of_edges()},
        }
        detection_xai = self.engine.explain(host, p_t, self.graph.skg.graph, edge)          # Role 2 XAI
        port = int(edge["port"]) if edge.get("port") else None
        result = self.dispatcher.process_alert(state, host, host, timestamp=ts, target_port=port,
                                               detection_xai=detection_xai, evidence=evidence)  # Role 3
        self._last_dispatch[host] = (ts, action)
        self.enforcement.add(host, Containment(result["case_id"], action, port, ts))
        self.report.dispatches.append({
            "time": ts, "host": host, "counterpart": counterpart, "port": port, "action": ACTIONS[action].name,
            "playbook": ACTIONS[action].playbook, "p_t": round(p_t, 4), "risk": round(risk, 3),
            "phi": dict(zip(PHI_NAMES, phi)), "case_id": result["case_id"],
            "shuffle": result["soar_dispatched"], "thehive": result["thehive_logged"],
            "edges_severed": result["edges_severed"], "topology_error": result["topology_error"],
            "ground_truth_label": label,     # evaluation only; not sent anywhere
            "decision_margin": result["xai_bundle"]["x_xtgaia_strategic_xai"]["decision_margin"],
            "runner_up": result["xai_bundle"]["x_xtgaia_strategic_xai"]["runner_up_comparison"]["runner_up_playbook"],
        })
        logger.info("Dispatched %s for %s (p_t=%.3f risk=%.2f) case=%s", ACTIONS[action].name, host, p_t, risk,
                    result["case_id"])

    def _ledger_changed(self) -> bool:
        version = getattr(self.enforcement.ledger, "version", None)
        if version is None:                      # ledger without change tracking: always re-check
            return True
        if version != self._ledger_version:
            self._ledger_version = version
            return True
        return False

    def refresh_enforcement(self) -> List[Tuple[str, Containment]]:
        lifted = self.enforcement.refresh()
        for host, c in lifted:
            self.report.rollbacks_seen.append({"host": host, "case_id": c.case_id, "action": ACTIONS[c.action].name})
        return lifted

    # ------------------------------------------------------------------ run loop
    def run(self, source: Iterable[Optional[Dict[str, Any]]], max_events: Optional[int] = None,
            progress_every: int = 0) -> RunReport:
        edges: List[Dict[str, Any]] = []
        labels: List[Optional[str]] = []
        for raw in source:
            if raw is None:                          # idle tick from a live source: flush
                if edges:
                    self._process_batch(edges, labels)
                    edges, labels = [], []
                continue
            self.report.events += 1
            if self.enforcement.active and self._ledger_changed():   # analyst rollbacks take effect immediately
                self.refresh_enforcement()
            edge = self.graph.ingest(raw)            # Role 1
            if edge is not None:
                self.report.edges += 1
                edges.append(edge)
                labels.append(raw.get("label"))
                if len(edges) >= self.micro_batch:
                    self._process_batch(edges, labels)
                    edges, labels = [], []
            if progress_every and self.report.events % progress_every == 0:
                m = self.report.detection_metrics()
                print(f"    ... {self.report.events} events | flagged {self.report.flagged_flows} | "
                      f"dispatches {len(self.report.dispatches)} | recall {m['recall']:.3f} precision {m['precision']:.3f}",
                      flush=True)
            if max_events is not None and self.report.events >= max_events:
                break
        if edges:
            self._process_batch(edges, labels)
        self.graph.flush_neo4j()
        return self.report

    # ------------------------------------------------------------------ analyst rollback API
    def serve_rollback_api(self, host: str = "0.0.0.0", port: Optional[int] = None, feedback_path: Optional[str] = None):
        """Run the Role 3 rollback API in this process so it shares the decision
        ledger, quarantine state and live graph with the pipeline. `feedback_path`
        overrides where penalty transitions are persisted (default: paths.feedback_buffer)."""
        import uvicorn
        from src.role3_decision import feedback_rollback as fr

        fr._handler = fr.FeedbackRollbackHandler(ledger=self.dispatcher.ledger,
                                                 quarantine_manager=self.dispatcher.quarantine,
                                                 graph_provider=self.graph.get_active_graph_and_risk,
                                                 persist_path=feedback_path)
        self.rollback_handler = fr._handler
        port = int(port or os.getenv("ROLLBACK_API_PORT", "8008"))
        server = uvicorn.Server(uvicorn.Config(fr.app, host=host, port=port, log_level="warning"))
        threading.Thread(target=server.run, daemon=True, name="xtgaia-rollback-api").start()
        for _ in range(100):
            if server.started:
                break
            time.sleep(0.05)
        self._api_server = server
        return server
