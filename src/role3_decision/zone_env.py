"""
XTGAIA Stage 5, multi-agent formulation (paper Sec. III-B, Phase 5):

  "Response selection is formulated as a Markov Decision Process and modeled within a
   multi-agent environment using the PettingZoo framework. The system models localized
   security agents operating across distributed topological zones ... Multi-agent
   optimization is conducted via independent Double Deep Q-Networks."

XTGAIAZoneEnv (PettingZoo AEC) has one attacker and one defender agent per topological zone
(e.g. workstations, servers). Each zone's host dynamics, observation (the shared 16-d state
S^i_t = [Z^i_t || p_t, Risk^i_t || Phi^i_t]) and decomposed reward R^i_t = b1 R_contain - b2 R_disrupt
- b3 R_cost are exactly those of the single-zone simulator (XTGAIAPettingZooEnv). The attacker plays
a real MITRE CALDERA adversary profile (stockpile abilities -> ATT&CK tactic chain) inside the zone it currently holds; a lateral-movement tactic spreads
the intrusion into the next zone of the chain (towards critical infrastructure).

train_zone_policies trains one independent reward-decomposed Double DQN per zone agent.
"""

import functools
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.optim as optim
from gymnasium.spaces import Box, Discrete
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import AgentSelector

from src.role3_decision.actions import NUM_ACTIONS, label
from src.role3_decision.config import load_config, resolve_path
from src.role3_decision.double_dqn import QNetwork, _optimise, make_replay_buffer, policy_summary
from src.role3_decision.pettingzoo_env import TACTICS, XTGAIAPettingZooEnv, default_profiles
from src.role3_decision.state import STATE_DIM

ZONE_CHECKPOINT_FORMAT = "xtgaia-zone-double-dqn-v1"


def zone_config(cfg: Optional[dict] = None) -> Dict[str, dict]:
    cfg = cfg or load_config()
    zones = cfg.get("zones") or {"workstations": {"crit_tiers": [0.1, 0.4]}, "servers": {"crit_tiers": [0.7, 1.0]}}
    return {str(k): dict(v or {}) for k, v in zones.items()}


class XTGAIAZoneEnv(AECEnv):
    metadata = {"name": "xtgaia_zones_v1"}

    def __init__(self, zones: Optional[Dict[str, dict]] = None, profiles=None, max_steps: int = 20):
        super().__init__()
        self.zones = zones or zone_config()
        self.zone_names = list(self.zones)
        self.profiles = profiles or default_profiles()
        self.max_steps = max_steps
        self.sims = {z: XTGAIAPettingZooEnv(profiles=self.profiles, max_steps=max_steps) for z in self.zone_names}
        self.defenders = [f"defender_{z}" for z in self.zone_names]
        self.possible_agents = ["attacker"] + self.defenders
        box = Box(low=0.0, high=1.0, shape=(STATE_DIM,), dtype=np.float32)
        self.observation_spaces = {a: box for a in self.possible_agents}
        self.action_spaces = {a: Discrete(NUM_ACTIONS) for a in self.defenders}
        self.action_spaces["attacker"] = Discrete(len(TACTICS))
        self._agent_selector = AgentSelector(self.possible_agents)

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]

    def observe(self, agent):
        return self.observations[agent]

    # ------------------------------------------------------------------ episode
    def _reset_zone(self, z: str, attacked: bool) -> None:
        sim = self.sims[z]
        sim.reset()
        tiers = self.zones[z].get("crit_tiers")
        if tiers:
            sim.crit_tier = float(np.random.choice(tiers))
        if not attacked:                                   # zone starts clean
            sim.is_attack_episode = False
            sim.p_t = float(np.random.uniform(0.01, 0.15))
            sim.risk_t = float(np.random.uniform(0.01, 0.12))
            sim.conn_count = float(np.random.uniform(0, 15))
            sim.auth_fails = float(np.random.randint(0, 3))
            sim.cpu_spike = float(np.random.uniform(0.0, 0.4))
        sim._update_observations()

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self.agents = self.possible_agents[:]
        self.rewards = {a: 0.0 for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        zero = np.zeros(3, dtype=np.float32)
        self.infos = {a: {"reward_components": zero.copy()} for a in self.agents}
        self.current_step = 0
        self.entry = int(np.random.randint(0, len(self.zone_names)))    # zone where the intrusion starts
        self.attack = np.random.random() < 0.65
        self.held = {self.zone_names[self.entry]} if self.attack else set()
        for i, z in enumerate(self.zone_names):
            self._reset_zone(z, attacked=self.attack and i == self.entry)
        self.done_zones = set()
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self._sync_obs()

    def _sync_obs(self):
        self.observations = {f"defender_{z}": self.sims[z].observations["defender"] for z in self.zone_names}
        self.observations["attacker"] = self.sims[self.zone_names[self.entry]].observations["attacker"]

    def scripted_attacker_action(self) -> int:
        z = next(iter(self.held), self.zone_names[self.entry])
        return self.sims[z].scripted_attacker_action()

    # ------------------------------------------------------------------ dynamics
    def step(self, action):
        agent = self.agent_selection
        if self.terminations[agent] or self.truncations[agent]:
            self._was_dead_step(action)
            return
        self._cumulative_rewards[agent] = 0.0
        self._clear_rewards()
        if agent == "attacker":
            self.current_step += 1
            for z in list(self.held):
                sim = self.sims[z]
                sim._attacker_step(int(action))
                if TACTICS[int(action)] == "lateral-movement":       # spread towards the next zone
                    nxt = self.zone_names[(self.zone_names.index(z) + 1) % len(self.zone_names)]
                    if nxt not in self.held and nxt not in self.done_zones:
                        target = self.sims[nxt]
                        target.is_attack_episode = True
                        target.p_t = max(target.p_t, float(np.random.uniform(0.30, 0.55)))
                        target.risk_t = max(target.risk_t, float(np.random.uniform(0.25, 0.48)))
                        self.held.add(nxt)
            self.rewards["attacker"] = float(sum(self.sims[z].rewards.get("attacker", 0.0) for z in self.held))
        else:
            z = agent[len("defender_"):]
            sim = self.sims[z]
            sim.current_step = self.current_step
            sim.terminations = {a: False for a in sim.possible_agents}
            sim._defender_step(int(action))
            self.rewards[agent] = sim.rewards["defender"]
            self.infos[agent] = {"reward_components": sim.infos["defender"]["reward_components"]}
            if any(sim.terminations.values()):
                self.done_zones.add(z)
                self.held.discard(z)
        for z in self.zone_names:
            self.sims[z]._update_observations()
        self._sync_obs()
        if (len(self.done_zones) == len(self.zone_names) or self.current_step >= self.max_steps
                or (not self.held and self.current_step >= 3)):
            self.terminations = {a: True for a in self.agents}
        self._accumulate_rewards()
        self.agent_selection = self._agent_selector.next()


# ============================================================================ training
def train_zone_policies(episodes: int = 10000, save_path=None, seed: Optional[int] = None, verbose: bool = True):
    """Independent reward-decomposed Double DQN per zone agent (own network, target, prioritized buffer)."""
    cfg = load_config()
    hyper = cfg["hyperparameters"]
    gamma, batch = float(hyper["gamma"]), int(hyper["batch_size"])
    eps, eps_min, decay = (float(hyper["exploration"]["initial_epsilon"]), float(hyper["exploration"]["min_epsilon"]),
                           float(hyper["exploration"]["decay_rate"]))
    sync_every = int(hyper["target_update_interval"])
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    env = XTGAIAZoneEnv()
    nets = {a: QNetwork() for a in env.defenders}
    targets = {a: QNetwork() for a in env.defenders}
    for a in env.defenders:
        targets[a].load_state_dict(nets[a].state_dict())
        targets[a].eval()
    opts = {a: optim.Adam(nets[a].parameters(), lr=float(hyper["learning_rate"])) for a in env.defenders}
    buffers = {a: make_replay_buffer(cfg) for a in env.defenders}
    steps = {a: 0 for a in env.defenders}
    device = torch.device("cpu")
    for ep in range(1, episodes + 1):
        env.reset()
        last = {}
        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            done = term or trunc
            if agent in nets and agent in last:
                o, act = last.pop(agent)
                buffers[agent].push(o, act, reward, obs, done, info["reward_components"])
            if done:
                env.step(None)
                continue
            if agent == "attacker":
                env.step(env.scripted_attacker_action())
                continue
            if random.random() < eps:
                a = random.randrange(NUM_ACTIONS)
            else:
                with torch.no_grad():
                    a = int(nets[agent](torch.as_tensor(obs).unsqueeze(0)).argmax(1).item())
            last[agent] = (obs, a)
            steps[agent] += 1
            if len(buffers[agent]) >= batch:
                _optimise(nets[agent], targets[agent], opts[agent], buffers[agent], batch, gamma, device)
            if steps[agent] % sync_every == 0:
                targets[agent].load_state_dict(nets[agent].state_dict())
            env.step(a)
        eps = max(eps_min, eps * decay)
        if verbose and ep % 1000 == 0:
            print(f"Episode {ep}/{episodes} | steps {steps} | epsilon {eps:.3f}", flush=True)
    path = Path(save_path) if save_path else resolve_path(cfg["paths"].get("zone_policy", "data/models/dqn_zone_policies.pt"))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": ZONE_CHECKPOINT_FORMAT, "zones": list(env.zone_names),
                "state_dicts": {a[len("defender_"):]: nets[a].state_dict() for a in env.defenders},
                "state_dim": STATE_DIM, "action_dim": NUM_ACTIONS, "episodes": episodes, "seed": seed}, path)
    if verbose:
        print(f"Zone policies saved to {path}")
        for a in env.defenders:
            nets[a].eval()
            print(f"  {a}: " + ", ".join(f"{k}->{label(v)}" for k, v in policy_summary(nets[a]).items()))
    return path


def load_zone_policies(path=None) -> Dict[str, QNetwork]:
    cfg = load_config()
    path = Path(path) if path else resolve_path(cfg["paths"].get("zone_policy", "data/models/dqn_zone_policies.pt"))
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or ckpt.get("format") != ZONE_CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {ZONE_CHECKPOINT_FORMAT} checkpoint")
    out = {}
    for z, sd in ckpt["state_dicts"].items():
        m = QNetwork(ckpt["state_dim"], ckpt["action_dim"])
        m.load_state_dict(sd)
        m.eval()
        out[z] = m
    return out


if __name__ == "__main__":
    train_zone_policies()
