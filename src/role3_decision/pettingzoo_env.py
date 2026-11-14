"""
XTGAIA Stage 5: PettingZoo adversarial multi-agent environment (AEC).

Attacker: replays CALDERA-style adversary profiles, i.e. ordered sequences of
          MITRE ATT&CK tactics. Real CALDERA adversary exports can be loaded
          with `load_caldera_adversary`.
Defender: chooses one of the 5 canonical playbooks (actions.ACTIONS).

Observations are produced exactly like production inputs: the simulator
builds a raw XTGAIAStateVector (shared schema) and passes it through
state.encode_state. The policy therefore sees the same feature space for
simulated, mock and live Person 1/2 data.

Reward (weights from configs/reward_weights.yaml):
    R_t = w_contain * r_contain - w_disrupt * r_disrupt - w_cost * r_cost
Per-channel rewards are exposed in infos["defender"]["reward_components"] so the
decomposed Double DQN can learn Q_contain / Q_disrupt / Q_cost separately.
Tier-alignment shaping is folded into the matching channel (inaction or
insufficiency counts against containment, overkill counts as disruption), so the
scalar total is unchanged.
"""

import functools
from typing import Dict, List, Optional

import numpy as np
import yaml
from gymnasium.spaces import Box, Discrete
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import AgentSelector

from src.role3_decision.actions import NUM_ACTIONS, PORT_ISOLATION, QUARANTINE, RATE_LIMIT, REVOKE_SESSION
from src.role3_decision.config import load_config
from src.role3_decision.state import STATE_DIM, asset_criticality, encode_state, make_raw_state

# MITRE ATT&CK tactics the simulated adversary can execute, with their effect on
# (threat probability p_t, normalised topological risk).
# All ATT&CK enterprise tactics used by the CALDERA stockpile abilities (the first five keep their
# historical indices). Effects: (delta p_t, delta normalised risk) per executed step.
TACTICS: List[str] = ["discovery", "credential-access", "privilege-escalation", "lateral-movement", "exfiltration",
                      "collection", "persistence", "defense-evasion", "execution", "command-and-control", "impact",
                      "initial-access"]
TACTIC_EFFECTS: Dict[str, tuple] = {
    "discovery": (0.10, 0.05),
    "credential-access": (0.20, 0.15),
    "privilege-escalation": (0.25, 0.25),
    "lateral-movement": (0.35, 0.40),
    "exfiltration": (0.30, 0.30),
    "collection": (0.10, 0.10),
    "persistence": (0.15, 0.15),
    "defense-evasion": (0.05, 0.10),     # hides activity: little detection gain, foothold grows
    "execution": (0.15, 0.10),
    "command-and-control": (0.25, 0.20),
    "impact": (0.35, 0.35),
    "initial-access": (0.20, 0.15),
}

# Built-in adversary profiles (ordered tactic chains, CALDERA "atomic ordering" style).
DEFAULT_PROFILES: Dict[str, List[str]] = {
    "recon_only": ["discovery", "discovery", "discovery"],
    "credential_theft": ["discovery", "credential-access", "credential-access", "privilege-escalation"],
    "lateral_worm": ["discovery", "credential-access", "lateral-movement", "lateral-movement"],
    "data_theft": ["discovery", "privilege-escalation", "lateral-movement", "exfiltration"],
}


def default_profiles(cfg: Optional[dict] = None) -> Dict[str, List[str]]:
    """Real CALDERA adversaries (configs/reward_weights.yaml `caldera.stockpile`) when the stockpile is
    present, the built-in tactic chains otherwise."""
    from src.role3_decision.caldera import caldera_profiles
    from src.role3_decision.config import resolve_path
    cfg = cfg or load_config()
    path = (cfg.get("caldera") or {}).get("stockpile", "data/raw/caldera/stockpile.zip")
    try:
        p = resolve_path(path)
        if p.exists():
            profiles = caldera_profiles(str(p), supported=TACTIC_EFFECTS)
            if profiles:
                return profiles
    except Exception:
        pass
    return DEFAULT_PROFILES


def load_caldera_adversary(adversary_yaml: str, ability_tactics: Dict[str, str]) -> List[str]:
    """Convert a CALDERA adversary profile (atomic_ordering of ability IDs) into a tactic chain.

    `ability_tactics` maps CALDERA ability IDs to their ATT&CK tactic (the `tactic`
    field of each ability). Abilities with tactics outside TACTICS are skipped.
    """
    with open(adversary_yaml, "r", encoding="utf-8") as f:
        adv = yaml.safe_load(f)
    chain = [ability_tactics.get(a) for a in adv.get("atomic_ordering", [])]
    chain = [t for t in chain if t in TACTIC_EFFECTS]
    if not chain:
        raise ValueError(f"{adversary_yaml}: no abilities map onto supported tactics {TACTICS}")
    return chain


class XTGAIAPettingZooEnv(AECEnv):
    metadata = {"name": "xtgaia_pettingzoo_v2"}

    def __init__(self, profiles: Optional[Dict[str, List[str]]] = None, max_steps: int = 20):
        super().__init__()
        self.cfg = load_config()
        w = self.cfg["reward_weights"]
        self.w_contain, self.w_disrupt, self.w_cost = float(w["contain"]), float(w["disrupt"]), float(w["cost"])
        self.costs = [float(self.cfg["action_costs"][f"a{i}"]) for i in range(NUM_ACTIONS)]
        self.severities = [float(self.cfg["action_severities"][f"a{i}"]) for i in range(NUM_ACTIONS)]
        self.profiles = profiles or default_profiles(self.cfg)
        self.max_steps = max_steps

        self.possible_agents = ["defender", "attacker"]
        self.agents = self.possible_agents[:]
        self._agent_selector = AgentSelector(self.agents)
        self.action_spaces = {"defender": Discrete(NUM_ACTIONS), "attacker": Discrete(len(TACTICS))}
        box = Box(low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32)
        self.observation_spaces = {"defender": box, "attacker": box}
        self.observations, self.rewards, self._cumulative_rewards = {}, {}, {}
        self.terminations, self.truncations, self.infos = {}, {}, {}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def observe(self, agent):
        return self.observations[agent]

    # ------------------------------------------------------------------ episode setup
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.agents = self.possible_agents[:]
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        zero = np.zeros(3, dtype=np.float32)
        self.infos = {a: {"reward_components": zero.copy()} for a in self.agents}
        self.current_step = 0

        scenario = np.random.choice(["benign", "medium", "high", "critical"], p=[0.35, 0.25, 0.20, 0.20])
        ranges = {"benign": ((0.01, 0.15), (0.01, 0.12)), "medium": ((0.30, 0.55), (0.25, 0.48)),
                  "high": ((0.60, 0.80), (0.55, 0.75)), "critical": ((0.85, 0.98), (0.80, 0.98))}
        (p_lo, p_hi), (r_lo, r_hi) = ranges[scenario]
        self.is_attack_episode = scenario != "benign"
        self.p_t = float(np.random.uniform(p_lo, p_hi))
        self.risk_t = float(np.random.uniform(r_lo, r_hi))       # normalised risk in [0, 1)
        self.crit_tier = float(np.random.choice([0.1, 0.4, 0.7, 1.0]))
        self.profile = list(self.profiles[np.random.choice(list(self.profiles))])
        self.profile_pos = 0
        self.embedding_base = np.random.normal(0.0, 1.0, size=8).astype(np.float32)

        if self.is_attack_episode:
            self.conn_count = float(np.random.uniform(5, 40))
            self.auth_fails = float(np.random.randint(0, 7))
            self.cpu_spike = float(np.random.uniform(0.1, 1.0))
        else:
            self.conn_count = float(np.random.uniform(0, 15))
            self.auth_fails = float(np.random.randint(0, 3))
            self.cpu_spike = float(np.random.uniform(0.0, 0.4))
        self.cvss_max = float(np.random.uniform(0.0, 10.0))

        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self._update_observations()

    def _raw_state(self):
        emb = self.embedding_base + np.random.normal(0.0, 0.1, size=8).astype(np.float32)
        return make_raw_state(self.p_t, self.risk_t, self.crit_tier, gnn_embedding=emb,
                              conn_count=self.conn_count, auth_fails=self.auth_fails,
                              cpu_spike=self.cpu_spike, cvss_max=self.cvss_max, cfg=self.cfg)

    def _update_observations(self):
        enc = encode_state(self._raw_state(), self.cfg)
        self.observations = {"defender": enc, "attacker": enc}

    def scripted_attacker_action(self) -> int:
        """Next tactic of the current CALDERA-style profile (cycles once exhausted)."""
        tactic = self.profile[self.profile_pos % len(self.profile)]
        self.profile_pos += 1
        return TACTICS.index(tactic)

    # ------------------------------------------------------------------ dynamics
    def step(self, action):
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return
        self._cumulative_rewards[agent] = 0.0
        self._clear_rewards()
        self.current_step += 1

        if agent == "attacker":
            self._attacker_step(int(action))
        else:
            self._defender_step(int(action))

        self._accumulate_rewards()
        self._update_observations()
        self.agent_selection = self._agent_selector.next()

    def _attacker_step(self, action: int):
        if not self.is_attack_episode:
            self.p_t = float(np.clip(self.p_t + np.random.uniform(-0.01, 0.01), 0.01, 0.12))
            self.risk_t = float(np.clip(self.risk_t + np.random.uniform(-0.01, 0.01), 0.01, 0.10))
            return
        tactic = TACTICS[action]
        dp, dr = TACTIC_EFFECTS[tactic]
        self.p_t = min(0.99, self.p_t + dp)
        self.risk_t = min(0.99, self.risk_t + dr)
        if tactic == "credential-access":
            self.auth_fails += float(np.random.randint(1, 4))
        elif tactic in ("lateral-movement", "exfiltration", "command-and-control"):
            self.conn_count += float(np.random.uniform(2, 10))
            self.cpu_spike = min(1.0, self.cpu_spike + 0.1)
        self.rewards["attacker"] = float(self.risk_t * 2.0)

    def _defender_step(self, action: int):
        old_risk, old_p = float(self.risk_t), float(self.p_t)
        crit = asset_criticality(self.observations["defender"])

        if action == RATE_LIMIT:
            self.risk_t = max(0.0, self.risk_t - 0.35)
        elif action in (PORT_ISOLATION, REVOKE_SESSION):
            self.risk_t = max(0.0, self.risk_t - 0.50)
        elif action == QUARANTINE:
            self.risk_t = 0.0

        r_contain = old_risk - self.risk_t
        r_disrupt = crit * self.severities[action]
        r_cost = self.costs[action]
        bonus, inaction, insufficient, overkill = self._tier_shaping(action, old_risk, old_p, crit)

        is_benign = old_risk < 0.18 and old_p < 0.18
        contained = (action == QUARANTINE or self.risk_t <= 0.05
                     or (action in (PORT_ISOLATION, REVOKE_SESSION) and self.risk_t <= 0.20)
                     or (action == RATE_LIMIT and crit <= 0.1 and self.risk_t <= 0.25))
        # Playbook bonuses are paid only on the step that actually contains the threat
        # (benign: correct inaction). Otherwise a partial action could farm the bonus
        # every step while the attacker escalates, and a4 would win only by stalling.
        # Result: a4 is preferred strictly when no lighter playbook can contain in one step.
        if not (is_benign or contained):
            bonus = 0.0

        c_contain = r_contain + (bonus - inaction - insufficient) / self.w_contain
        c_disrupt = r_disrupt + overkill / self.w_disrupt
        components = np.array([c_contain, c_disrupt, r_cost], dtype=np.float32)
        total = self.w_contain * c_contain - self.w_disrupt * c_disrupt - self.w_cost * r_cost

        self.rewards["defender"] = float(total)
        self.rewards["attacker"] = -float(total)
        self.infos["defender"] = {"reward_components": components}

        if contained or (is_benign and self.current_step >= 3) or self.current_step >= self.max_steps:
            self.terminations = {a: True for a in self.agents}

    @staticmethod
    def _tier_shaping(action: int, old_risk: float, old_p: float, crit: float):
        """Proportional-response shaping: returns (bonus, inaction, insufficiency, overkill)."""
        bonus = inaction = insufficient = overkill = 0.0
        if old_risk < 0.18 and old_p < 0.18:                       # benign
            if action == 0:
                bonus = 3.5
            else:
                overkill = 3.5
        elif old_risk < 0.50:                                      # medium
            if action == 0:
                inaction = 5.0
            elif action == 4:
                overkill = 5.0
            elif crit <= 0.1:
                bonus, overkill = (4.0, 0.0) if action == 1 else (0.0, 2.0 if action == 2 else 3.0)
            elif crit <= 0.4:
                if action == 2:
                    bonus = 3.5
                elif action == 1:
                    bonus = 2.0
                else:
                    overkill = 1.5
            else:
                if action == 3:
                    bonus = 3.5
                elif action == 2:
                    bonus = 2.0
                else:
                    insufficient = 2.5
        elif old_risk < 0.78:                                      # high
            if action == 0:
                inaction = 6.0
            elif action == 1:
                insufficient = 4.0
            elif crit <= 0.1:
                bonus, overkill = (4.0, 0.0) if action == 2 else (0.0, 2.0 if action == 3 else 3.0)
            elif crit <= 0.4:
                if action in (2, 3):
                    bonus = 3.0
                else:
                    overkill = 2.0
            else:
                if action == 3:
                    bonus = 3.0
                elif action in (2, 4):
                    bonus = 2.0
        else:                                                      # critical
            if action == 0:
                inaction = 8.0
            elif action == 1:
                insufficient = 6.0
            elif crit <= 0.7:
                if action == 3:
                    bonus = 4.0
                elif action == 2:
                    bonus = 3.0
                elif action == 4:
                    overkill = 5.0
            elif old_risk >= 0.90:
                if action == 4:
                    bonus = 6.0
                elif action == 3:
                    bonus = 2.0
                else:
                    insufficient = 4.0
            else:
                if action == 4:
                    bonus = 4.0
                elif action == 3:
                    bonus = 3.0
                elif action == 2:
                    bonus = 0.5
        return bonus, inaction, insufficient, overkill
