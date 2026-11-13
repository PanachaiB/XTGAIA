"""
XTGAIA Stage 7: human-in-the-loop feedback and rollback.

When an analyst clicks Rollback in TheHive (false positive):
  1. Shuffle's rollback webhook reverses the playbook (un-quarantine / unblock port)
  2. the edges Role 3 severed are restored on the live Person 1 graph (RAM),
     and the host's pre-containment risk is put back
  3. (S_t, a_t, R = reward_weights.rollback, S_t+1, done=True) is injected into
     the replay buffer D used for training, and persisted to the feedback store
     so a separate training process picks it up.

The penalty is booked on the disruption channel, so the decomposed Q heads
learn that the action was disruptive without being necessary.

Algorithm 4, lines 13-15 (audit window): if the analyst approves the action, or the
inspection window T_audit elapses without a rollback, the standard transition
(S_t, a_t, R_t, S_t+1) is appended to D instead, with R_t from Eq. 10:
    R_t = b1 * R_contain - b2 * R_disrupt - b3 * R_cost
    R_contain = Risk~(S_t) - Risk~(S_t+1)   (normalised risk, as in the simulator)
    R_disrupt = criticality(S_t) * severity(a_t),   R_cost = cost(a_t)
"""

import hmac
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from src.role3_decision.actions import ACTIONS, QUARANTINE
from src.role3_decision.config import feedback_buffer_path, load_config, reward_weights
from src.role3_decision.decision_ledger import DecisionLedger, get_decision_ledger
from src.role3_decision.double_dqn import ReplayBuffer, get_shared_replay_buffer
from src.role3_decision.graph_interface import (GraphRiskProvider, QuarantineManager, default_graph_provider,
                                                get_quarantine_manager)
from src.role3_decision.soar_client import post_shuffle
from src.role3_decision.state import IDX_RISK, asset_criticality, encode_state, validate_state

logger = logging.getLogger("XTGAIA-Feedback-Rollback")


class FeedbackRollbackHandler:
    def __init__(self, replay_buffer: Optional[ReplayBuffer] = None, ledger: Optional[DecisionLedger] = None,
                 quarantine_manager: Optional[QuarantineManager] = None,
                 graph_provider: Optional[GraphRiskProvider] = None, persist_path=None, persist: bool = True):
        self.buffer = replay_buffer if replay_buffer is not None else get_shared_replay_buffer()
        self.ledger = ledger or get_decision_ledger()
        self.quarantine = quarantine_manager or get_quarantine_manager()
        self._graph_provider = graph_provider
        self.persist_path = (persist_path or feedback_buffer_path()) if persist else None
        self.feedback_store = ReplayBuffer(capacity=int(load_config()["hyperparameters"]["replay_buffer_capacity"]))
        if self.persist_path:
            self.feedback_store.load(self.persist_path)
        # Optional live-state hook (set by the pipeline): host -> raw S_t+1 measured at commit time.
        # Without it, approval commits immediately with the S_t+1 recorded at dispatch.
        self.state_provider = None

    # ------------------------------------------------------------------ audit window (Algorithm 4, 13-15)
    @staticmethod
    def standard_reward(s_t, action: int, s_next):
        """Eq. 10 reward and its channels for an executed, non-rolled-back action."""
        cfg = load_config()
        w = reward_weights(cfg)
        e_t, e_next = encode_state(s_t), encode_state(s_next)
        r_contain = float(e_t[IDX_RISK] - e_next[IDX_RISK])
        r_disrupt = asset_criticality(e_t) * float(cfg["action_severities"][f"a{action}"])
        r_cost = float(cfg["action_costs"][f"a{action}"])
        total = w["contain"] * r_contain - w["disrupt"] * r_disrupt - w["cost"] * r_cost
        return float(total), (r_contain, r_disrupt, r_cost)

    def commit(self, case_id: str, next_state_vector=None, reason: str = "timeout") -> Dict[str, Any]:
        rec = self.ledger.get(case_id)
        if rec is None:
            raise ValueError(f"Unknown case {case_id!r}")
        if rec.rolled_back or rec.committed:
            raise ValueError(f"Case {case_id} is already {'rolled back' if rec.rolled_back else 'committed'}")
        s_t = validate_state(np.asarray(rec.state, dtype=np.float32))
        s_next = validate_state(np.asarray(next_state_vector if next_state_vector is not None else rec.next_state,
                                           dtype=np.float32))
        reward, components = self.standard_reward(s_t, rec.action, s_next)
        transition = (encode_state(s_t), int(rec.action), reward, encode_state(s_next), True, components)
        self.buffer.push(*transition)
        self.feedback_store.push(*transition)
        if self.persist_path:
            self.feedback_store.save(self.persist_path)
        self.ledger.mark_committed(case_id)
        logger.info("[Feedback] case %s committed (%s): a=%s R=%.4f", case_id, reason, rec.action, reward)
        return {"status": "COMMITTED", "case_id": case_id, "reason": reason, "reward": reward,
                "components": {"contain": components[0], "disrupt": components[1], "cost": components[2]},
                "buffer_size": len(self.buffer)}

    def approve(self, case_id: str, analyst_comment: str = "") -> Dict[str, Any]:
        rec = self.ledger.get(case_id)
        if rec is None:
            raise ValueError(f"Unknown case {case_id!r}")
        if rec.rolled_back or rec.committed:
            raise ValueError(f"Case {case_id} is already {'rolled back' if rec.rolled_back else 'committed'}")
        self.ledger.mark_approved(case_id)
        if self.state_provider is None:
            return self.commit(case_id, reason="approved")
        return {"status": "APPROVED", "case_id": case_id,
                "note": "committed by the pipeline with the post-action state on its next cycle"}

    def rollback(self, case_id: str, state_vector=None, next_state_vector=None, executed_action: Optional[int] = None,
                 target_host: Optional[str] = None, target_ip: Optional[str] = None,
                 graph=None, risk: Optional[Dict[str, float]] = None, timestamp: Optional[float] = None,
                 analyst_comment: str = "") -> Dict[str, Any]:
        rec = self.ledger.get(case_id)
        if rec is not None and rec.rolled_back:
            raise ValueError(f"Case {case_id} was already rolled back")
        host = target_host or (rec.target_host if rec else None)
        ip = target_ip or (rec.target_ip if rec else None)
        action = executed_action if executed_action is not None else (rec.action if rec else None)
        s_t = state_vector if state_vector is not None else (rec.state if rec else None)
        s_next = next_state_vector if next_state_vector is not None else (rec.next_state if rec else None)
        if None in (host, action) or s_t is None or s_next is None:
            raise ValueError(f"Unknown case {case_id!r} and request lacks host/action/state vectors")
        if action not in ACTIONS:
            raise ValueError(f"executed_action must be in 0..4, got {action}")
        s_t, s_next = validate_state(np.asarray(s_t, dtype=np.float32)), validate_state(np.asarray(s_next, dtype=np.float32))

        # 1. reverse the playbook in Shuffle
        soar_reverted = post_shuffle("rollback", {
            "event": "XTGAIA_ROLLBACK_EXECUTION", "case_id": case_id, "target_host": host, "target_ip": ip,
            "action_to_reverse": int(action), "playbook_to_reverse": ACTIONS[action].playbook,
            "rollback_command": "UNQUARANTINE_HOST" if action == QUARANTINE else f"REVERSE_{ACTIONS[action].name.upper()}",
            "analyst_comment": analyst_comment,
        })

        # 2. restore the live graph (RAM)
        restored, topology_error = 0, None
        if self.quarantine.is_isolated(host):
            try:
                if graph is None:
                    graph, risk = (self._graph_provider or default_graph_provider())(timestamp or 0.0)
                restored = len(self.quarantine.restore(graph, host))
                prior = self.quarantine.pop_prior_risk(host)
                if risk is not None:
                    # exact pre-quarantine value; S_t's (alert-time) risk only if no snapshot exists
                    risk[host] = prior if prior is not None else float(s_t[IDX_RISK])
            except RuntimeError as exc:
                topology_error = str(exc)
                logger.error("Graph restore for %s failed: %s", host, exc)

        # 3. inject the penalty transition into D (and the persistent feedback store)
        penalty = reward_weights()["rollback"]
        components = (0.0, -penalty / reward_weights()["disrupt"], 0.0)
        transition = (encode_state(s_t), int(action), float(penalty), encode_state(s_next), True, components)
        self.buffer.push(*transition)
        self.feedback_store.push(*transition)
        if self.persist_path:
            self.feedback_store.save(self.persist_path)
        if rec is not None:
            self.ledger.mark_rolled_back(case_id)
        logger.info("[Feedback] case %s: injected (a=%s, R=%s) into replay buffer", case_id, action, penalty)

        return {"status": "SUCCESS", "case_id": case_id, "target_host": host, "soar_reverted": soar_reverted,
                "edges_restored": restored, "topology_error": topology_error,
                "penalty_reward_injected": float(penalty), "buffer_size": len(self.buffer)}


# ---------------------------------------------------------------------- HTTP API
class RollbackRequest(BaseModel):
    case_id: str
    target_host: Optional[str] = None
    target_ip: Optional[str] = None
    executed_action: Optional[int] = None
    state_vector: Optional[List[float]] = None        # raw schema S_t (16)
    next_state_vector: Optional[List[float]] = None   # raw schema S_t+1 (16)
    analyst_comment: Optional[str] = "False Positive Override"


class ApproveRequest(BaseModel):
    case_id: str
    analyst_comment: Optional[str] = ""


app = FastAPI(title="XTGAIA Feedback & Rollback API", version="2.1.0")
_handler: Optional[FeedbackRollbackHandler] = None


def get_handler() -> FeedbackRollbackHandler:
    global _handler
    if _handler is None:
        _handler = FeedbackRollbackHandler()
    return _handler


def _check_token(x_xtgaia_token: Optional[str]) -> None:
    expected = os.getenv("ROLLBACK_WEBHOOK_TOKEN")
    if expected:
        if not x_xtgaia_token or not hmac.compare_digest(x_xtgaia_token, expected):
            raise HTTPException(status_code=401, detail="Invalid or missing X-XTGAIA-Token")
    else:
        logger.warning("ROLLBACK_WEBHOOK_TOKEN not set; feedback endpoint is unauthenticated")


@app.post("/api/v1/feedback/approve")
def handle_analyst_approval(req: ApproveRequest, x_xtgaia_token: Optional[str] = Header(default=None)):
    """Analyst confirms the containment inside the audit window (Algorithm 4, line 14)."""
    _check_token(x_xtgaia_token)
    try:
        return get_handler().approve(req.case_id, req.analyst_comment or "")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/v1/feedback/rollback")
def handle_analyst_rollback(req: RollbackRequest, x_xtgaia_token: Optional[str] = Header(default=None)):
    _check_token(x_xtgaia_token)
    try:
        return get_handler().rollback(
            case_id=req.case_id, target_host=req.target_host, target_ip=req.target_ip,
            executed_action=req.executed_action,
            state_vector=None if req.state_vector is None else np.asarray(req.state_vector, dtype=np.float32),
            next_state_vector=None if req.next_state_vector is None else np.asarray(req.next_state_vector, dtype=np.float32),
            analyst_comment=req.analyst_comment or "",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("ROLLBACK_API_PORT", "8008")))
