"""
Canonical Role 3 action space. Every module (env, Double DQN, XAI, SOAR
dispatcher, rollback) uses this table so indices, names and Shuffle playbook
names can never drift apart.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Action:
    id: int
    name: str              # canonical name used in reports / XAI
    playbook: str          # value sent to Shuffle as `playbook_name`
    description: str
    severs_edges: bool     # does it mutate the in-memory topology?


ACTIONS: Dict[int, Action] = {
    0: Action(0, "NoOp", "xtgaia_a0_noop", "No action / passive monitoring", False),
    1: Action(1, "RateLimit", "xtgaia_a1_rate_limit", "Throttle the host's network traffic", False),
    2: Action(2, "PortIsolation", "xtgaia_a2_port_isolation", "Block the targeted port/protocol", True),
    3: Action(3, "RevokeSession", "xtgaia_a3_revoke_session", "Revoke user sessions / tokens", False),
    4: Action(4, "Quarantine", "xtgaia_a4_quarantine", "Fully quarantine the host", True),
}

NUM_ACTIONS = len(ACTIONS)
NOOP, RATE_LIMIT, PORT_ISOLATION, REVOKE_SESSION, QUARANTINE = range(NUM_ACTIONS)


def label(action_id: int) -> str:
    """Human-readable label, e.g. 'a4 Quarantine'."""
    return f"a{action_id} {ACTIONS[action_id].name}"
