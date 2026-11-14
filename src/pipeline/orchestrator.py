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

import uuid
from concurrent.futures import ThreadPoolExecutor

from src.role1_graph.host_context import AssetInventory
from src.role1_graph.threat_intel import AttackMapper, NVDClient
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
        try:   # silence "property not in database yet" notifications (e.g. first latent_state lookup)
            driver = GraphDatabase.driver(uri, auth=(user, pwd), notifications_min_severity="OFF")
        except TypeError:   # older drivers without notification filtering
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
    latent_recovered: int = 0
    committed: Counter = field(default_factory=Counter)
    latency: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    online_updates: List[Dict[str, Any]] = field(default_factory=list)
    xai_vectors: List[Dict[str, Any]] = field(default_factory=list)          # Sec. IV.E stability, evaluation only
    started: float = field(default_factory=time.time)

    def xai_stability(self) -> Dict[str, Any]:
        from src.evaluation.xai_stability import stability
        return stability(self.xai_vectors)

    def latency_summary(self) -> Dict[str, Dict[str, float]]:
        out = {}
        for k, v in self.latency.items():
            if v:
                a = np.asarray(v, dtype=float)
                out[k] = {"mean": float(a.mean()), "p95": float(np.percentile(a, 95)), "std": float(a.std()),
                          "n": int(len(a))}
        return out

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
        # Δinf: TGN inference window (paper Phase 3: 500 ms); micro_batch caps the events per update
        self.inference_window_s = float(r2.get("inference_window_ms", 500.0)) / 1000.0
        # Ablation switches (paper Sec. IV-F)
        abl = self.cfg.get("ablation", {}) or {}
        self.use_risk_assessor = bool(abl.get("risk_assessor", True))
        self.dual_timescale = bool(abl.get("dual_timescale", True))
        self.closed_loop = bool(abl.get("closed_loop", True))
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
        self.latent_sync_every = int(r2.get("latent_sync_every", 50))   # micro-batches between Neo4j writes
        self._latent_dirty: set = set()
        self._batches = 0

        # Role 1: live graph published through the Person 1 interface
        nvd_cfg = r1.get("nvd", {}) or {}
        nvd = NVDClient(cache_path=str(resolve_path(nvd_cfg.get("cache", "data/cache/nvd_cvss.json"))),
                        enabled=bool(nvd_cfg.get("enabled", False)))
        inventory = AssetInventory.from_yaml(str(resolve_path(r1.get("asset_inventory", "configs/asset_inventory.yaml"))),
                                             nvd=nvd)
        self.inventory = inventory
        self.attack = AttackMapper(str(resolve_path(r1.get("attack_stix", "data/raw/attack/enterprise-attack.json"))))
        driver = neo4j_driver_from_env() if neo4j_driver == "env" and r1.get("neo4j_sync", True) else \
            (None if neo4j_driver == "env" else neo4j_driver)
        self.graph = LiveGraphService(
            inventory, str(resolve_path(r1.get("port_weights", "configs/mitre_t1021_ports.json"))),
            window_ms=float(r1.get("skg_window_ms", 50.0)), host_window_s=float(r1.get("host_context_window_s", 300)),
            auth_ports=r1.get("auth_ports", []), neo4j_driver=driver,
            neo4j_batch_size=int(r1.get("neo4j_batch_size", 500)), attack=self.attack)
        set_live_graph_service(self.graph)
        self.graph.skg.latent_model = getattr(self.engine, "fingerprint", None)

        # Role 3: dispatcher mutating Role 1's live graph
        self.dispatcher = dispatcher or SOARDispatcher(graph_provider=self.graph.get_active_graph_and_risk)
        self.dispatcher.quarantine.add_listener(self.graph)       # Algorithm 4: Neo4j soft delete / reactivate
        self.dispatcher.mode = str(r3.get("mode", "active"))       # deployment regime: active | passive
        self.dispatcher.mutate_enabled = self.closed_loop          # ablation: w/o closed-loop feedback
        if r3.get("zone_policies", True):                          # Phase 5: independent agent per zone
            try:
                from src.role3_decision.zone_env import load_zone_policies
                self.dispatcher.set_zone_policies(load_zone_policies(), self.inventory.zone)
            except (FileNotFoundError, ValueError) as exc:
                logger.warning("Zone policies unavailable (%s); using the single policy", exc)
        self.enforcement = Enforcement(self.dispatcher.ledger)
        self._ledger_version = getattr(self.dispatcher.ledger, "version", None)
        self.graph.is_blocked = self.enforcement.blocks if self.closed_loop else (lambda s, d, p: False)
        self.updater = None
        self._xai_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="xtgaia-xai")
        self.severity = {int(k[1:]): float(v) for k, v in load_config()["action_severities"].items()}
        self.cooldown_s = float(r3.get("dispatch_cooldown_s", 900))
        self.max_dispatches = int(r3.get("max_dispatches", 25))
        # Alerting vs acting: Role 2's threshold drives detection/XAI; autonomous containment
        # additionally requires the model to call an attack more likely than not.
        self.min_response_p = float(r3.get("min_response_p", 0.0))
        # Algorithm 4 audit window T_audit (configs/system_config.yaml analyst_inspection_window_sec)
        self.audit_window_s = float(r3.get("audit_window_s", 300))
        self.feedback_handler = None
        self._last_dispatch: Dict[str, Tuple[float, int]] = {}
        self.report = RunReport()
        self._api_server = None
        self.warm_up()

    def warm_up(self) -> None:
        """One read-only pass through every model (TGN + explainers, each zone's Double DQN + reward decomposition)
        at start-up, so the first real alert does not pay framework initialisation (~100 ms) in its latency."""
        try:
            import torch
            from src.role3_decision.state import make_raw_state
            known = next(iter(self.engine.ip_to_id), None)
            with torch.no_grad():
                self.engine.model.forward_full(torch.tensor([0], dtype=torch.long))
            if known is not None and getattr(self.engine, "explainer", None) is not None:
                self.engine.explain_tiers(known)
            state = make_raw_state(0.5, 0.5, 1.0)
            for zone, members in getattr(self.inventory, "zone_members", {}).items():
                host = next(iter(members), None) if members else None
                if host:
                    self.dispatcher.predict_action(state, host)
                    self.dispatcher.decision_xai(state, host)
            self.dispatcher.predict_action(state, "0.0.0.0")
            self.dispatcher.decision_xai(state, "0.0.0.0")
        except Exception as exc:                       # warm-up must never block start-up
            logger.debug("warm-up skipped: %s", exc)

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
        phi = self.graph.phi(host)                             # Role 1: host context
        if not self.use_risk_assessor:                         # ablation: raw p_t, no criticality / decay
            risk, phi = float(p_t), [0.0, 0.0] + list(phi[2:])
        elif self.dual_timescale:
            risk = self.graph.compute_risk(host, p_t)          # Role 1: Algorithm 2 on the RAM graph
        else:
            risk = self.graph.compute_risk_neo4j(host, p_t)    # ablation: traversals on Neo4j directly
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

    def _observe(self, edges: List[Dict[str, Any]]) -> List[float]:
        """Role 2 memory update with Algorithm 1 latent-state recovery: hosts seen for the first
        time in this process get m_v(t-) from Neo4j (if persisted) before the update."""
        hosts = {e["source"] for e in edges} | {e["target"] for e in edges}
        new = self.engine.needs_hydration(hosts)
        if new:
            states = self.graph.fetch_latent_states(new)
            if states:
                self.report.latent_recovered += self.engine.hydrate(states)
            self.engine.mark_live(new)
        probs = self.engine.observe_batch(edges)
        self._latent_dirty |= hosts
        self._batches += 1
        if self.latent_sync_every and self._batches % self.latent_sync_every == 0:
            self.flush_latent()
        return probs

    def flush_latent(self) -> None:
        if self._latent_dirty:
            self.graph.store_latent_states(self.engine.export_latent(sorted(self._latent_dirty)))
            self._latent_dirty = set()

    def _process_batch(self, edges: List[Dict[str, Any]], labels: List[Optional[str]]) -> None:
        t0 = time.perf_counter()
        if self.burn_in_remaining > 0:                                             # baseline: Role 1 + 2 only
            scores = self._observe(edges)
            self._burn_scores.extend(scores)
            for edge, p in zip(edges, scores):
                self._burn_host_scores[self.target_of(edge)[0]].append(p)
            self._burn_attacks += sum(1 for l in labels if l and l.upper() != "BENIGN")
            self.burn_in_remaining -= len(edges)
            if self.burn_in_remaining <= 0:
                self._finish_burn_in()
            return
        t_an = time.perf_counter()
        probs = self._observe(edges)                                               # Role 2
        if self.dual_timescale and self.use_risk_assessor:                         # Algorithm 2, all v in V_act
            self.graph.compute_all_risks(self.engine.latest_scores(self.graph.host_nodes()))
        t_detect = time.perf_counter()
        self.report.latency["analysis_ms"].append((t_detect - t_an) * 1000.0)
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
                proc = (t_detect - edge.get("_t_ingest", t_detect)) * 1000.0
                self.report.latency["mttd_processing_ms"].append(proc)
                self.report.latency["mttd_window_ms"].append(
                    proc + max(0.0, float(edges[-1]["timestamp"]) - float(edge["timestamp"])) * 1000.0)
                if host not in flagged or p > flagged[host][0]:
                    edge["_t_detect"] = t_detect
                    flagged[host] = (p, edge, label or "")
        self.commit_due(float(edges[-1]["timestamp"]))
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
        t0 = time.perf_counter()
        state, risk, phi = self.build_state(host, p_t)
        action, q = self.dispatcher.predict_action(state, host)
        self.report.latency["decision_ms"].append((time.perf_counter() - t0) * 1000.0)
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
        # Algorithm 3, line 5: parallel extraction of the multi-tiered explanations
        t_x = time.perf_counter()
        f_det = self._xai_pool.submit(self._timed, self.engine.explain, host, p_t, self.graph.skg.graph, edge)
        f_dec = self._xai_pool.submit(self._timed, self.dispatcher.decision_xai, state, host)
        (detection_xai, ms12), (decision_xai, ms3) = f_det.result(), f_dec.result()
        xai_total = (time.perf_counter() - t_x) * 1000.0
        self.report.latency["xai_tiers12_ms"].append(ms12)
        self.report.latency["xai_tier3_ms"].append(ms3)
        self.report.latency["xai_total_ms"].append(xai_total)
        xt = build_xt(detection_xai, decision_xai, {"tiers_1_2": ms12, "tier_3": ms3, "total": xai_total})
        from src.evaluation.xai_stability import attribution_vectors
        self.report.xai_vectors.append({"label": label, "vectors": attribution_vectors(
            getattr(self.engine, "last_tiers", None), decision_xai)})
        port = int(edge["port"]) if edge.get("port") else None
        result = self.dispatcher.process_alert(state, host, host, timestamp=ts, target_port=port,
                                               detection_xai=detection_xai, evidence=evidence,
                                               decision_xai=decision_xai, xai_bundle_xt=xt)       # Role 3
        t_done = time.perf_counter()
        t_enf = result.get("t_enforced", t_done)           # Algorithm 4 Phase 1 done: enforcement + G_t+1
        self.report.latency["execution_ms"].append(result.get("execution_ms", 0.0))
        self.report.latency["mttr_ms"].append((t_enf - edge.get("_t_detect", t_enf)) * 1000.0)
        self.report.latency["end_to_end_ms"].append((t_enf - edge.get("_t_ingest", t_enf)) * 1000.0)
        if "audit_ms" in result:
            self.report.latency["audit_ms"].append(result["audit_ms"])
        self._last_dispatch[host] = (ts, action)
        self.enforcement.add(host, Containment(result["case_id"], action, port, ts))
        self.report.dispatches.append({
            "time": ts, "host": host, "counterpart": counterpart, "port": port, "action": ACTIONS[action].name,
            "playbook": ACTIONS[action].playbook, "p_t": round(p_t, 4), "risk": round(risk, 3),
            "phi": dict(zip(PHI_NAMES, phi)), "case_id": result["case_id"],
            "shuffle": result["soar_dispatched"], "thehive": result["thehive_logged"],
            "edges_severed": result["edges_severed"], "topology_error": result["topology_error"],
            "zone": result.get("zone"), "mode": result.get("mode"), "xai_ms": round(xai_total, 2),
            "xai_fidelity_plus": (getattr(self.engine, "last_tiers", None) or {}).get("fidelity_plus"),
            "xai_fidelity_minus": (getattr(self.engine, "last_tiers", None) or {}).get("fidelity_minus"),
            "xai_sparsity": (getattr(self.engine, "last_tiers", None) or {}).get("sparsity"),
            "xai_within_200ms": xai_total <= 200.0,
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

    # ------------------------------------------------------------------ audit window
    def _feedback(self, feedback_path: Optional[str] = None):
        if self.feedback_handler is None:
            from src.role3_decision.feedback_rollback import FeedbackRollbackHandler
            self.feedback_handler = FeedbackRollbackHandler(
                ledger=self.dispatcher.ledger, quarantine_manager=self.dispatcher.quarantine,
                graph_provider=self.graph.get_active_graph_and_risk, persist_path=feedback_path)
            self.feedback_handler.state_provider = self.current_raw_state
            if (load_config().get("online_update", {}) or {}).get("enabled", True) and self.closed_loop:
                from src.role3_decision.online_update import OnlineUpdater
                self.updater = OnlineUpdater(self.dispatcher, self.feedback_handler.buffer)
                self.feedback_handler.on_feedback.append(self._on_feedback)
        return self.feedback_handler

    def _on_feedback(self, host) -> None:
        rec = self.updater.notify(self.dispatcher.model_for(host)[2]) if self.updater else None
        if rec:
            self.report.online_updates.append(rec)

    @staticmethod
    def _timed(fn, *args):
        t = time.perf_counter()
        out = fn(*args)
        return out, (time.perf_counter() - t) * 1000.0

    def _evict_latent(self, nodes: List[str]) -> None:
        """Algorithm 1: nodes leaving V_act write their latent state back to Neo4j and are
        re-hydrated from it when they re-enter the active window."""
        live = [n for n in nodes if n in self.engine.live]
        if live:
            self.graph.store_latent_states(self.engine.export_latent(live))
        self.engine.live.difference_update(nodes)
        self._latent_dirty.difference_update(nodes)

    def current_raw_state(self, host: str) -> np.ndarray:
        """S_t+1 measured now: Role 2's current p_t and Role 1's current risk/context for the host."""
        p_now, _ = self.engine.score(host)
        state, _, _ = self.build_state(host, p_now)
        return state.to_numpy()

    def commit_due(self, now: float) -> int:
        """Algorithm 4, lines 14-15: approved decisions, or decisions whose T_audit elapsed
        without a rollback, are appended to D as standard transitions."""
        due = self.dispatcher.ledger.due_for_commit(now, self.audit_window_s)
        for rec in due:
            reason = "approved" if rec.approved else "timeout"
            try:
                self._feedback().commit(rec.case_id, self.current_raw_state(rec.target_host), reason)
                self.report.committed[reason] += 1
            except ValueError as exc:
                logger.warning("Commit of %s skipped: %s", rec.case_id, exc)
        return len(due)

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
            t_in = time.perf_counter()
            new_edges = self.graph.ingest_all(raw)    # Role 1: STIX normalisation + typed SKG update
            acq_ms = (time.perf_counter() - t_in) * 1000.0
            self.report.latency["acquisition_ms"].append(acq_ms)
            for edge in new_edges:
                self.report.edges += 1
                if edge.get("features") is None:     # host telemetry: graph/context only (no flow statistics)
                    continue
                edge["_t_ingest"] = t_in
                edges.append(edge)
                labels.append(raw.get("label"))
            gone = self.graph.pop_evicted()
            if gone:
                self._evict_latent(gone)
            # Δinf: flush when the window spans 500 ms of event time or reaches the batch cap
            if edges and (len(edges) >= self.micro_batch or
                          float(edges[-1]["timestamp"]) - float(edges[0]["timestamp"]) >= self.inference_window_s):
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
        self.graph.flush_neo4j(wait=True)
        self.flush_latent()
        return self.report

    # ------------------------------------------------------------------ analyst rollback API
    def serve_rollback_api(self, host: str = "0.0.0.0", port: Optional[int] = None, feedback_path: Optional[str] = None):
        """Run the Role 3 rollback API in this process so it shares the decision
        ledger, quarantine state and live graph with the pipeline. `feedback_path`
        overrides where penalty transitions are persisted (default: paths.feedback_buffer)."""
        import uvicorn
        from src.role3_decision import feedback_rollback as fr

        self.feedback_handler = None
        fr._handler = self._feedback(feedback_path)          # same handler: rollback, approve, commits
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


def build_xt(detection: Dict[str, Any], decision: Dict[str, Any], latency_ms: Dict[str, float]) -> Dict[str, Any]:
    """Algorithm 3, line 12: package structural (Tier 1), temporal (Tier 2) and strategic (Tier 3)
    artefacts into one STIX 2.1 extension bundle X_t."""
    objects = list((detection or {}).get("objects", [])) + ([decision] if decision else [])
    objects.append({
        "type": "x-xtgaia-explanation", "spec_version": "2.1", "id": f"x-xtgaia-explanation--{uuid.uuid4()}",
        "tiers": {"structural": "PGExplainer", "temporal": "AttentionRollout", "strategic": "RewardDecomposition"},
        "extraction": "parallel", "latency_ms": latency_ms, "real_time_bound_ms": 200.0,
        "within_bound": float(latency_ms.get("total", 0.0)) <= 200.0,
    })
    return {"type": "bundle", "id": f"bundle--{uuid.uuid4()}", "objects": objects}
