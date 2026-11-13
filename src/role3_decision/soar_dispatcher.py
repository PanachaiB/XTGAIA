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
        self._graph_provider = graph_provider
        self.quarantine = quarantine_manager or get_quarantine_manager()
        self.ledger = ledger or get_decision_ledger()

    # ---------------------------------------------------------------- inference
    def predict_action(self, state: StateLike) -> Tuple[int, Dict[str, float]]:
        enc = encode_state(state)
        with torch.no_grad():
            q = self.model(torch.as_tensor(enc, device=self.device).unsqueeze(0))[0].cpu().numpy()
        return int(q.argmax()), {f"a{i}": float(v) for i, v in enumerate(q)}

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
                      evidence: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Stages 5-7 for one alert. `detection_xai` is Role 2's STIX detection bundle and
        `evidence` is Role 1's context (attacker, trigger flow, crown-jewel path); both are
        forwarded to Shuffle (summary) and attached to the TheHive case (full)."""
        raw = validate_state(state)
        timestamp = time.time() if timestamp is None else float(timestamp)
        action, q_dict = self.predict_action(raw)
        xai = self.xai.export_stix_xai_extension(raw, target_host)
        playbook = ACTIONS[action]

        result: Dict[str, Any] = {
            "action": action, "action_name": playbook.name, "playbook": playbook.playbook,
            "q_values": q_dict, "xai_bundle": xai, "soar_dispatched": False, "thehive_logged": False,
            "case_id": None, "edges_severed": 0, "topology_error": None, "mutated_graph": G_t,
        }
        if action == NOOP:
            logger.info("a0 NoOp for %s; no SOAR dispatch", target_host)
            result["soar_dispatched"] = True
            return result

        result["soar_dispatched"] = post_shuffle("execution", {
            "event": "XTGAIA_CONTAINMENT_DISPATCH", "target_host": target_host, "target_ip": target_ip,
            "target_port": target_port, "action_code": action, "action_name": playbook.name,
            "playbook_name": playbook.playbook, "q_values": q_dict, "timestamp": timestamp,
            "evidence": evidence or {}, "detection": _detection_summary(detection_xai),
        })

        next_raw = raw.copy()
        if playbook.severs_edges:
            try:
                graph, live_risk = self._active_graph(timestamp, G_t, risk)
                result["edges_severed"] = self.mutate_topology(graph, live_risk, target_host, action, target_port)
                result["mutated_graph"] = graph
                if target_host in live_risk:
                    next_raw[IDX_RISK] = float(live_risk[target_host])
            except (RuntimeError, KeyError) as exc:
                result["topology_error"] = str(exc)
                logger.error("Topology mutation failed for %s: %s", target_host, exc)

        narrative = xai["x_xtgaia_strategic_xai"]["analyst_narrative"]
        sections = [f"Automated containment: **{playbook.description}** ({playbook.playbook}).", narrative]
        if evidence:
            sections.append("### Detection context (Role 1)\n```json\n"
                            + json.dumps(evidence, indent=2, default=str) + "\n```")
        if detection_xai:
            sections.append("### Detection XAI (Role 2, STIX 2.1)\n```json\n"
                            + json.dumps(detection_xai, indent=2, default=str) + "\n```")
        sections.append("### Decision XAI (Role 3, STIX 2.1)\n```json\n" + json.dumps(xai, indent=2) + "\n```")
        case_id = create_thehive_case(
            title=f"[XTGAIA] {playbook.name} on {target_host}",
            description="\n\n".join(sections),
            severity=min(4, THEHIVE_SEVERITY[action] + int(xai["x_xtgaia_strategic_xai"]["context"]["is_crown_jewel"])),
            tags=["XTGAIA", "DoubleDQN", playbook.playbook, f"host:{target_host}"],
        )
        result["thehive_logged"] = case_id is not None
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
