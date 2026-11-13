"""
Decision ledger: remembers each executed containment (keyed by TheHive case id)
so an analyst's Rollback click only needs the case id to recover
(S_t, a_t, S_t+1, host) for the R = rollback-penalty transition.
"""

import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


@dataclass
class DecisionRecord:
    case_id: str
    target_host: str
    target_ip: str
    action: int
    state: np.ndarray          # raw schema S_t
    next_state: np.ndarray     # raw schema S_t+1 (after containment)
    timestamp: float
    target_port: Optional[int] = None
    rolled_back: bool = False
    approved: bool = False     # analyst explicitly approved inside the audit window
    committed: bool = False    # standard transition (S_t, a, R_t, S_t+1) appended to D
    extra: dict = field(default_factory=dict)


class DecisionLedger:
    def __init__(self):
        self._records: Dict[str, DecisionRecord] = {}
        self._lock = threading.Lock()
        self.version = 0          # bumped on every rollback, so consumers can detect changes cheaply

    def record(self, rec: DecisionRecord) -> DecisionRecord:
        with self._lock:
            self._records[rec.case_id] = rec
        return rec

    def get(self, case_id: str) -> Optional[DecisionRecord]:
        return self._records.get(case_id)

    def mark_approved(self, case_id: str) -> None:
        with self._lock:
            if case_id in self._records:
                self._records[case_id].approved = True
                self.version += 1

    def mark_committed(self, case_id: str) -> None:
        with self._lock:
            if case_id in self._records:
                self._records[case_id].committed = True

    def due_for_commit(self, now: float, audit_window_s: float):
        """Algorithm 4, line 14: decisions the analyst approved, or whose inspection window
        T_audit elapsed without a rollback, and that have not been committed yet."""
        with self._lock:
            return [r for r in self._records.values()
                    if not (r.rolled_back or r.committed) and (r.approved or now - r.timestamp >= audit_window_s)]

    def mark_rolled_back(self, case_id: str) -> None:
        with self._lock:
            if case_id in self._records:
                self._records[case_id].rolled_back = True
                self.version += 1

    @staticmethod
    def local_case_id() -> str:
        """Id used when TheHive is unavailable so the decision is still traceable/rollbackable."""
        return f"local-{uuid.uuid4().hex[:12]}"


_shared: Optional[DecisionLedger] = None


def get_decision_ledger() -> DecisionLedger:
    global _shared
    if _shared is None:
        _shared = DecisionLedger()
    return _shared
