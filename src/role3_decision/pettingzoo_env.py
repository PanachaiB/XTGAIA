"""
XTGAIA Stage 5: PettingZoo Adversarial Multi-Agent Environment
Implements turn-based multi-agent interaction between Attacker (MITRE TTPs)
and Defender (Double DQN) with formal decomposed rewards:
R_t = beta1 * R_contain - beta2 * R_disrupt - beta3 * R_cost
"""

import functools
import os
import random
import numpy as np
import yaml
from gymnasium.spaces import Box, Discrete
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import AgentSelector

class XTGAIAPettingZooEnv(AECEnv):
    metadata = {"name": "xtgaia_pettingzoo_v1"}

    def __init__(self, config_path="configs/reward_weights.yaml"):
        super().__init__()
        self.agents = ["attacker", "defender"]
        self.possible_agents = self.agents[:]
        self._agent_selector = AgentSelector(self.agents)

        # Load parameters from YAML
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
            self.beta1 = float(cfg["reward_coefficients"]["beta1"])
            self.beta2 = float(cfg["reward_coefficients"]["beta2"])
            self.beta3 = float(cfg["reward_coefficients"]["beta3"])
            self.costs = [float(cfg["action_costs"][f"a{i}"]) for i in range(5)]
            self.severities = [float(cfg["action_severities"][f"a{i}"]) for i in range(5)]
        else:
            # Mathematical fallbacks
            self.beta1 = 1
            self.beta2 = 0.5
            self.beta3 = 0.1
            self.costs = [0.0, 1.0, 2.0, 2.0, 5.0]
            self.severities = [0.0, 0.1, 0.4, 0.5, 1.0]

        # Action Spaces:
        # Defender: 5 Playbooks (0: Monitor, 1: Rate-Limit, 2: Isolate Port, 3: Revoke Session, 4: Quarantine)
        # Attacker: 3 MITRE TTPs (0: Probe, 1: Exploit, 2: Lateral Move)
        self.action_spaces = {
            "defender": Discrete(5),
            "attacker": Discrete(3),
        }

        # Observation Spaces: 16d state vector S_t = [Z_t (8d) || p_t (1d) || Risk_t (1d) || Phi_t (6d)]
        self.observation_spaces = {
            "defender": Box(low=0.0, high=1.0, shape=(16,), dtype=np.float32),
            "attacker": Box(low=0.0, high=1.0, shape=(16,), dtype=np.float32),
        }

        # Internal state tracking
        self.attack_stage = 0
        self.p_t = 0.05
        self.risk_t = 0.02
        self.asset_criticality = 0.8  # Phi_t
        self.current_step = 0
        self.max_steps = 20

        self.observations = {}
        self.rewards = {}
        self._cumulative_rewards = {}
        self.terminations = {}
        self.truncations = {}
        self.infos = {}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def observe(self, agent):
        """Returns the 16-dimensional observation vector for the active agent."""
        return self.observations[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        self.current_step = 0
        self.max_steps = 20

        # Sample initial scenario: 50% benign baseline, 50% active attack
        scenario_type = random.choice(["benign, medium, critical"])
        if scenario_type == "benign":
            self.attack_stage = 0
            self.p_t = float(np.random.uniform(0.01, 0.12))
            self.risk_t = float(np.random.uniform(0.01, 0.10))
            self.asset_criticality = float(np.random.choice([0.1, 0.4, 0.8, 1.0]))
        elif scenario_type == "medium":
            self.attack_stage = 1
            self.p_t = float(np.random.uniform(0.30, 0.55))
            self.risk_t = float(np.random.uniform(0.25, 0.50))
            self.asset_criticality = float(np.random.choice([0.1, 0.4, 0.8, 1.0]))
        else: # Critical
            self.attack_stage = 2
            self.p_t = float(np.random.uniform(0.70, 0.95))
            self.risk_t = float(np.random.uniform(0.65, 0.95))
            self.asset_criticality = float(np.random.choice([0.4, 0.8, 1.0]))

        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self._update_observations()

    def _update_observations(self):
        # Tie latent graph embedding Z_t (8d) to threat level
        threat_level = float((self.p_t + self.risk_t) / 2.0)
        z_t = np.full((8,), threat_level, dtype=np.float32)

        # Phi_t (6d): [criticality, vulns, exposure, trust, auth, protocols]
        phi_t = np.array(
            [self.asset_criticality, 0.5, 0.1, 0.2, 0.0, 1.0], dtype=np.float32
        )

        state_vec = np.concatenate([
            z_t,
            np.array([self.p_t, self.risk_t], dtype=np.float32),
            phi_t,
        ]).astype(np.float32)

        self.observations = {
            "defender": state_vec,
            "attacker": state_vec,
        }

    def step(self, action):
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return

        self._cumulative_rewards[agent] = 0.0

        self._clear_rewards()
        
        self.current_step += 1

        if agent == "attacker":

            if self.attack_stage == 0:
            # Benign scenario: attacker does not execute an attack
                self.rewards["attacker"] = 0.0
                self.rewards["defender"] = 0.0

            else:
                # Active attack scenario
                if action == 0:  # Probe
                    self.p_t = min(1.0, self.p_t + 0.15)

                elif action == 1:  # Exploit
                    self.attack_stage = max(self.attack_stage, 1)
                    self.p_t = min(1.0, self.p_t + 0.35)
                    self.risk_t = min(1.0, self.risk_t + 0.30)

                elif action == 2:  # Lateral Movement
                    self.attack_stage = max(self.attack_stage, 2)
                    self.p_t = min(1.0, self.p_t + 0.45)
                    self.risk_t = min(1.0, self.risk_t + 0.50)

                self.rewards["attacker"] = float(self.risk_t * 2.0)
                self.rewards["defender"] = 0.0

        elif agent == "defender":
            old_risk = float(self.risk_t)

            # Apply mitigation actions to risk
            if action == 0:  # No Action
                pass
            elif action == 1:  # Rate-Limit
                self.risk_t = max(0.0, self.risk_t - 0.25)
            elif action == 2:  # Isolate Port
                self.risk_t = max(0.0, self.risk_t - 0.50)
            elif action == 3:  # Revoke Session
                self.risk_t = max(0.0, self.risk_t - 0.40)
            elif action == 4:  # Host Quarantine
                self.risk_t = 0.0

            # --- Formal Reward Decomposition ---
            # 1. R_containment: Change in topological risk
            r_contain = old_risk - self.risk_t

            # 2. R_disruption: Asset Criticality * Action Severity
            r_disrupt = self.asset_criticality * self.severities[action]

            # 3. R_cost: Direct execution overhead from YAML
            r_cost = self.costs[action]

            # Inaction penalty if ignoring active high-risk attacks
            inaction_penalty = (old_risk * 3.5) if (action == 0 and old_risk >= 0.25) else 0.0

            # Insuffciency penalty if a lesser containment action was chosen
            insufficiency_penalty = 1.5 if (action == 1 and old_risk >= 0.50) else 0.0

            # Benign baseline incentive: reward doing nothing when system is safe
            benign_bonus = 1.0 if (action == 0 and old_risk < 0.15 and self.p_t < 0.15) else 0.0

            # Base formula: R_t = beta1 * R_contain - beta2 * R_disrupt - beta3 * R_cost
            r_t = (
                (self.beta1 * r_contain)
                - (self.beta2 * r_disrupt)
                - (self.beta3 * r_cost)
                - inaction_penalty
                - insufficiency_penalty
                + benign_bonus
            )

            self.rewards["defender"] = float(r_t)
            self.rewards["attacker"] = -float(r_t)

            # Terminate if threat is eliminated, quarantine is enforced, or step limit reached
            if (
                (old_risk > 0.40 and self.risk_t <= 0.05)
                or self.current_step >= self.max_steps
            ):
                self.terminations = {a: True for a in self.agents}

        self._accumulate_rewards()
        self._update_observations()
        self.agent_selection = self._agent_selector.next()