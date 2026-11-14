"""
XTGAIA Stage 7: SOAR dispatcher.

For each alert/state:
  1. validate + encode S_t, run the Double DQN (eval mode) -> action a_t
  2. build the Stage 6 XAI explanation from the decomposed Q heads
  3. POST the playbook to Shuffle's execution webhook (URL from env)
  4. mutate the live Person 1 graph in RAM (a4 quarantine / a2 port isolation)
  5. open a TheHive case carrying the XAI narrative
  6. record (case_id -> S_t, a_t, S_t+1) so an analyst Rollback can be learned from
"""

import json
import logging
import time
from typing import Any, Dict, Optional, Tuple

import networkx as nx
import torch

from src.role3_decision.actions import ACTIONS, NOOP, PORT_ISOLATION, QUARANTINE
from src.role3_decision.decision_ledger import DecisionLedger, DecisionRecord, get_decision_ledger
from src.role3_decision.double_dqn import QNetwork, load_policy
from src.role3_decision.graph_interface import (GraphRiskProvider, QuarantineManager, default_graph_provider,
                                                get_quarantine_manager)
from src.role3_decision.soar_client import create_thehive_case, post_shuffle
from src.role3_decision.state import IDX_RISK, StateLike, encode_state, validate_state
from src.role3_decision.strategic_xai import StrategicXAIExplainer

logger = logging.getLogger("XTGAIA-SOAR-Dispatcher")

THEHIVE_SEVERITY = {1: 1, 2: 2, 3: 2, 4: 3}   # TheHive: 1 low, 2 medium, 3 high, 4 critical


def _detection_summary(bundle: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Compact view of Role 2's STIX detection bundle for the SOAR payload."""
    if not bundle:
        return {}
    try:
        ext = bundle["objects"][0]["x_xtgaia_xai"]
    except (KeyError, IndexError, TypeError):
        return {}
    return {
        "threat_confidence_pt": ext.get("threat_confidence_pt"),
        "decision_threshold": ext.get("decision_threshold"),
        "top_edges": ext.get("structural_attribution", {}).get("influential_2hop_edges", [])[:3],
        "attack_onset_ms": ext.get("temporal_attribution", {}).get("attack_onset_ms"),
        "attack_duration_ms": ext.get("temporal_attribution", {}).get("attack_duration_ms"),
        "stix_bundle_id": bundle.get("id"),
    }


class SOARDispatcher:
    def __init__(self, weights_path=None, model: Optional[QNetwork] = None,
                 graph_provider: Optional[GraphRiskProvider] = None,
                 quarantine_manager: Optional[QuarantineManager] = None,
                 ledger: Optional[DecisionLedger] = None, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = model if model is not None else load_policy(weights_path, device=self.device)
        self.model.eval()
        self.xai = StrategicXAIExplainer(model=self.model)
        # Paper Phase 5: one independent Double DQN per topological zone (optional; falls back to
        # the single policy for hosts whose zone has no agent).
        self.zone_models: Dict[str, QNetwork] = {}
        self.zone_xai: Dict[str, StrategicXAIExplainer] = {}
        self.zone_of = lambda host: None
        # Deployment regime (paper Phase 5): "active" executes playbooks; "passive" is online
        # observation with human-on-the-loop: decisions become TheHive recommendations only.
        self.mode = "active"
        self._graph_provider = graph_provider
        self.quarantine = quarantine_manager or get_quarantine_manager()
        self.ledger = ledger or get_decision_ledger()

    # ---------------------------------------------------------------- inference
    def set_zone_policies(self, models: Dict[str, QNetwork], zone_of) -> None:
        self.zone_models = dict(models)
        self.zone_xai = {z: StrategicXAIExplainer(model=m) for z, m in self.zone_models.items()}
        self.zone_of = zone_of

    def model_for(self, host: Optional[str]) -> Tuple[QNetwork, StrategicXAIExplainer, Optional[str]]:
        z = self.zone_of(host) if host is not None else None
        if z in self.zone_models:
            return self.zone_models[z], self.zone_xai[z], z
        return self.model, self.xai, None

    def predict_action(self, state: StateLike, host: Optional[str] = None) -> Tuple[int, Dict[str, float]]:
        model, _, _ = self.model_for(host)
        enc = encode_state(state)
        with torch.no_grad():
            q = model(torch.as_tensor(enc, device=self.device).unsqueeze(0))[0].cpu().numpy()
        return int(q.argmax()), {f"a{i}": float(v) for i, v in enumerate(q)}

    def decision_xai(self, state: StateLike, host: str) -> Dict[str, Any]:
        """Tier 3 (reward decomposition) from the host's zone agent."""
        _, xai, _ = self.model_for(host)
        return xai.export_stix_xai_extension(validate_state(state), host)

    # ---------------------------------------------------------------- topology
    def _active_graph(self, timestamp: float, G_t, risk):
        if G_t is not None:
            return G_t, (risk if risk is not None else {})
        provider = self._graph_provider or default_graph_provider()
        return provider(timestamp)

    def mutate_topology(self, graph: nx.Graph, risk: Dict[str, float], host: str, action: int,
                        target_port: Optional[int] = None) -> int:
        if action == QUARANTINE:
            return len(self.quarantine.quarantine(graph, risk, host))
        if action == PORT_ISOLATION:
            if target_port is None:
                logger.warning("a2 PortIsolation for %s without target_port; topology left unchanged", host)
                return 0
            return len(self.quarantine.isolate_port(graph, risk, host, target_port))
        return 0

    # ---------------------------------------------------------------- pipeline
    def process_alert(self, state: StateLike, target_host: str, target_ip: str,
                      timestamp: Optional[float] = None, G_t: Optional[nx.Graph] = None,
                      risk: Optional[Dict[str, float]] = None, target_port: Optional[int] = None,
                      detection_xai: Optional[Dict[str, Any]] = None,
                      evidence: Optional[Dict[str, Any]] = None,
                      decision_xai: Optional[Dict[str, Any]] = None,
                      xai_bundle_xt: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Stages 5-7 for one alert. `detection_xai` is Role 2's STIX detection bundle and
        `evidence` is Role 1's context (attacker, trigger flow, crown-jewel path); both are
        forwarded to Shuffle (summary) and attached to the TheHive case (full)."""
        raw = validate_state(state)
        timestamp = time.time() if timestamp is None else float(timestamp)
        action, q_dict = self.predict_action(raw, target_host)
        _, _, zone = self.model_for(target_host)
        xai = decision_xai or self.decision_xai(raw, target_host)
        playbook = ACTIONS[action]
        passive = self.mode == "passive"

        result: Dict[str, Any] = {
            "action": action, "action_name": playbook.name, "playbook": playbook.playbook,
            "q_values": q_dict, "xai_bundle": xai, "soar_dispatched": False, "thehive_logged": False,
            "case_id": None, "edges_severed": 0, "topology_error": None, "mutated_graph": G_t,
            "zone": zone, "mode": self.mode, "execution_ms": 0.0,
        }
        if action == NOOP:
            logger.info("a0 NoOp for %s; no SOAR dispatch", target_host)
            result["soar_dispatched"] = True
            return result

        t_exec = time.perf_counter()
        if not passive:
            # Algorithm 4, line 2: transmit (v*, a*, X_t) to the Shuffle SOAR webhook
            result["soar_dispatched"] = post_shuffle("execution", {
                "event": "XTGAIA_CONTAINMENT_DISPATCH", "target_host": target_host, "target_ip": target_ip,
                "target_port": target_port, "action_code": action, "action_name": playbook.name,
                "playbook_name": playbook.playbook, "q_values": q_dict, "timestamp": timestamp, "zone": zone,
                "evidence": evidence or {}, "detection": _detection_summary(detection_xai),
                "xai_bundle": xai_bundle_xt,
            })

        next_raw = raw.copy()
        if playbook.severs_edges and not passive and getattr(self, "mutate_enabled", True):
            try:
                graph, live_risk = self._active_graph(timestamp, G_t, risk)
                result["edges_severed"] = self.mutate_topology(graph, live_risk, target_host, action, target_port)
                result["mutated_graph"] = graph
                if target_host in live_risk:
                    next_raw[IDX_RISK] = float(live_risk[target_host])
            except (RuntimeError, KeyError) as exc:
                result["topology_error"] = str(exc)
                logger.error("Topology mutation failed for %s: %s", target_host, exc)

        # Algorithm 4 Phase 1 ends here: SOAR triggered and G_t+1 mutated (MTTR / closed-loop latency end point)
        t_enforced = time.perf_counter()
        result["execution_ms"] = (t_enforced - t_exec) * 1000.0
        result["t_enforced"] = t_enforced
        narrative = xai["x_xtgaia_strategic_xai"]["analyst_narrative"]
        sections = [f"Automated containment: **{playbook.description}** ({playbook.playbook}).", narrative]
        if evidence:
            sections.append("### Detection context (Role 1)\n```json\n"
                            + json.dumps(evidence, indent=2, default=str) + "\n```")
        if detection_xai:
            sections.append("### Detection XAI (Role 2, STIX 2.1)\n```json\n"
                            + json.dumps(detection_xai, indent=2, default=str) + "\n```")
        sections.append("### Decision XAI (Role 3, STIX 2.1)\n```json\n" + json.dumps(xai, indent=2) + "\n```")
        if xai_bundle_xt:
            sections.append("### Explanation bundle X_t (STIX 2.1, Tiers 1-3)\n```json\n"
                            + json.dumps(xai_bundle_xt, indent=2, default=str) + "\n```")
        if passive:
            sections.insert(0, "**Passive deployment (human-on-the-loop observation): recommended action, "
                               "not executed.** Approve or reject in the audit window.")
        case_id = create_thehive_case(
            title=f"[XTGAIA]{'[RECOMMENDATION]' if passive else ''} {playbook.name} on {target_host}",
            description="\n\n".join(sections),
            severity=min(4, THEHIVE_SEVERITY[action] + int(xai["x_xtgaia_strategic_xai"]["context"]["is_crown_jewel"])),
            tags=["XTGAIA", "DoubleDQN", playbook.playbook, f"host:{target_host}"],
        )
        result["thehive_logged"] = case_id is not None
        result["audit_ms"] = (time.perf_counter() - t_enforced) * 1000.0     # Phase 2: TheHive audit case
        result["case_id"] = case_id or self.ledger.local_case_id()
        self.ledger.record(DecisionRecord(
            case_id=result["case_id"], target_host=target_host, target_ip=target_ip, action=action,
            state=raw, next_state=next_raw, timestamp=timestamp, target_port=target_port,
        ))
        return result

    def process_containment_pipeline(self, state_vector: StateLike, target_host: str, target_ip: str,
                                     G_t: Optional[nx.Graph] = None, xai_bundle=None, **kwargs) -> Dict[str, Any]:
        """Backward-compatible entry point used by main.py (xai_bundle is now always computed)."""
        return self.process_alert(state_vector, target_host, target_ip, G_t=G_t, **kwargs)
