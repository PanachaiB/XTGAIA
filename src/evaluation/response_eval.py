"""Response-policy evaluation (paper Sec. IV: DRL vs traditional SOAR baselines) in the multi-agent zone
environment driven by real MITRE CALDERA adversary profiles.

Policies (same episodes, same seeds):
  XTGAIA zone DDQN      independent reward-decomposed Double DQN per zone (data/models/dqn_zone_policies.pt)
  Static SOAR playbook  traditional threshold playbook: quarantine when p_t >= 0.5, otherwise no action
  Graded SOAR rules     p_t >= 0.3 rate limit, >= 0.6 port isolation, >= 0.85 quarantine
  No response           never acts

Metrics per policy: mean defender return (paper Eq. 10 reward), containment rate and mean steps to contain
on attack episodes, attack spread (zones reached), false-containment rate (a2-a4 on benign zones), mean
business disruption and action cost per decision.

  python -m src.evaluation.response_eval --episodes 2000
"""
import argparse
import json
import os

import numpy as np
import torch

from src.role3_decision.actions import NUM_ACTIONS, PORT_ISOLATION, QUARANTINE, RATE_LIMIT
from src.role3_decision.zone_env import XTGAIAZoneEnv, load_zone_policies


def static_playbook(p):
    return QUARANTINE if p >= 0.5 else 0


def graded_rules(p):
    return QUARANTINE if p >= 0.85 else PORT_ISOLATION if p >= 0.6 else RATE_LIMIT if p >= 0.3 else 0


def evaluate(policy: str, episodes: int, seed: int, nets=None):
    env = XTGAIAZoneEnv()
    np.random.seed(seed)
    stats = {"return": [], "contained": 0, "attack_eps": 0, "steps_to_contain": [], "spread": [],
             "benign_decisions": 0, "false_containment": 0, "disruption": [], "cost": [], "actions": [0] * NUM_ACTIONS}
    for ep in range(episodes):
        env.reset(seed=seed * 100003 + ep)
        attack = bool(env.held)
        ret, first_contain = 0.0, None
        zones_hit = set(env.held)
        for agent in env.agent_iter():
            obs, r, term, trunc, info = env.last()
            if agent != "attacker":
                ret += r
            if term or trunc:
                env.step(None)
                continue
            if agent == "attacker":
                env.step(env.scripted_attacker_action())
                zones_hit |= env.held
                continue
            z = agent[len("defender_"):]
            sim = env.sims[z]
            benign_zone = not sim.is_attack_episode
            if policy == "ddqn":
                with torch.no_grad():
                    a = int(nets[z](torch.as_tensor(obs).unsqueeze(0)).argmax(1).item())
            elif policy == "static":
                a = static_playbook(sim.p_t)
            elif policy == "graded":
                a = graded_rules(sim.p_t)
            else:
                a = 0
            stats["actions"][a] += 1
            if benign_zone:
                stats["benign_decisions"] += 1
                stats["false_containment"] += int(a >= PORT_ISOLATION)
            held_before = z in env.held
            env.step(a)
            comp = env.infos[agent]["reward_components"]
            stats["disruption"].append(float(comp[1]))
            stats["cost"].append(float(comp[2]))
            if held_before and z in env.done_zones and first_contain is None:
                first_contain = env.current_step
        stats["return"].append(ret)
        stats["spread"].append(len(zones_hit))
        if attack:
            stats["attack_eps"] += 1
            if not env.held:
                stats["contained"] += 1
                stats["steps_to_contain"].append(first_contain or env.current_step)
    n_att = max(stats["attack_eps"], 1)
    return {"mean_return": float(np.mean(stats["return"])),
            "containment_rate": stats["contained"] / n_att,
            "mean_steps_to_contain": float(np.mean(stats["steps_to_contain"])) if stats["steps_to_contain"] else None,
            "mean_zones_reached": float(np.mean(stats["spread"])),
            "false_containment_rate": stats["false_containment"] / max(stats["benign_decisions"], 1),
            "mean_disruption": float(np.mean(stats["disruption"])) if stats["disruption"] else 0.0,
            "mean_cost": float(np.mean(stats["cost"])) if stats["cost"] else 0.0,
            "action_mix": stats["actions"], "episodes": episodes, "attack_episodes": stats["attack_eps"]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--policy-path", default=None)
    ap.add_argument("--out", default="data/processed/eval/response_results.json")
    a = ap.parse_args(argv)
    nets = load_zone_policies(a.policy_path)
    names = {"ddqn": "XTGAIA zone DDQN", "static": "Static SOAR playbook", "graded": "Graded SOAR rules",
             "none": "No response"}
    out = {}
    for key, name in names.items():
        out[name] = evaluate(key, a.episodes, a.seed, nets)
        r = out[name]
        print(f"{name:<22} return {r['mean_return']:8.2f} | contained {r['containment_rate']:.3f} in "
              f"{r['mean_steps_to_contain'] or float('nan'):.2f} steps | zones {r['mean_zones_reached']:.2f} | "
              f"false containment {r['false_containment_rate']:.3f} | disruption {r['mean_disruption']:.3f} "
              f"cost {r['mean_cost']:.2f}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
