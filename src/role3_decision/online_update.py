"""
Closed-loop online policy update (paper Phase 5, regime (iii), and Algorithm 4, line 21):

  "Under active deployment, analyst rollback events inject a severe penalty of R = -100 into the
   prioritized replay buffer, continually suppressing false-positive containment triggers."
  D <- D ∪ {(S_t, a*, R = -100, S_t+1)}   ▷ Triggers policy weight update

OnlineUpdater counts feedback transitions (rollback penalties and audit-window commits). Once
`min_new_transitions` have arrived it fine-tunes a copy of the affected zone agent (or the single
policy) on the prioritized buffer with Double DQN updates, and swaps it into the live dispatcher only
if the copy still agrees with the deployed policy on the tier probe states (guard against one
noisy feedback burst rewriting the whole policy). Accepted versions are saved under
paths.online_policies; every attempt is recorded in `history`.
"""

import datetime as dt
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from src.role3_decision.config import load_config, resolve_path
from src.role3_decision.double_dqn import QNetwork, _optimise, policy_summary, save_policy
from src.role3_decision.strategic_xai import StrategicXAIExplainer

logger = logging.getLogger("XTGAIA-Online-Update")


class OnlineUpdater:
    def __init__(self, dispatcher, buffer, cfg: Optional[dict] = None):
        self.dispatcher, self.buffer = dispatcher, buffer
        cfg = cfg or load_config()
        self.cfg = cfg.get("online_update", {}) or {}
        self.enabled = bool(self.cfg.get("enabled", True))
        self.gamma = float(cfg["hyperparameters"]["gamma"])
        self.out_dir = resolve_path(cfg["paths"].get("online_policies", "data/models/online"))
        self.pending: Dict[Optional[str], int] = {}
        self.history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def notify(self, zone: Optional[str] = None, n: int = 1) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return None
        with self._lock:
            self.pending[zone] = self.pending.get(zone, 0) + n
            if self.pending[zone] < int(self.cfg.get("min_new_transitions", 4)):
                return None
            self.pending[zone] = 0
        return self.update(zone)

    def _current(self, zone: Optional[str]) -> QNetwork:
        return self.dispatcher.zone_models.get(zone, self.dispatcher.model) if zone else self.dispatcher.model

    def update(self, zone: Optional[str] = None) -> Dict[str, Any]:
        base = self._current(zone)
        cand, target = QNetwork(), QNetwork()
        cand.load_state_dict(base.state_dict())
        target.load_state_dict(base.state_dict())
        target.eval()
        opt = torch.optim.Adam(cand.parameters(), lr=float(self.cfg.get("learning_rate", 1e-4)))
        batch = int(self.cfg.get("batch_size", 16))
        losses = []
        if len(self.buffer) == 0:
            return {"accepted": False, "reason": "empty buffer"}
        cand.train()
        for _ in range(int(self.cfg.get("gradient_steps", 50))):
            losses.append(_optimise(cand, target, opt, self.buffer, batch, self.gamma, torch.device("cpu")))
        cand.eval()
        before, after = policy_summary(base), policy_summary(cand)
        agreement = sum(before[k] == after[k] for k in before) / len(before)
        accepted = agreement >= float(self.cfg.get("min_probe_agreement", 0.9))
        rec = {"time": dt.datetime.now(dt.timezone.utc).isoformat(), "zone": zone, "accepted": accepted,
               "probe_agreement": agreement, "final_loss": losses[-1] if losses else None,
               "buffer_size": len(self.buffer), "changed_probes": [k for k in before if before[k] != after[k]]}
        if accepted:
            if zone and zone in self.dispatcher.zone_models:
                self.dispatcher.zone_models[zone] = cand
                self.dispatcher.zone_xai[zone] = StrategicXAIExplainer(model=cand)
            else:
                self.dispatcher.model = cand
                self.dispatcher.xai = StrategicXAIExplainer(model=cand)
            stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            path = Path(self.out_dir) / f"policy_{zone or 'single'}_{stamp}.pt"
            save_policy(cand, path)
            rec["saved"] = str(path)
            logger.info("Online update accepted for %s (probe agreement %.2f)", zone or "single policy", agreement)
        else:
            logger.warning("Online update rejected for %s (probe agreement %.2f)", zone or "single policy", agreement)
        self.history.append(rec)
        return rec
