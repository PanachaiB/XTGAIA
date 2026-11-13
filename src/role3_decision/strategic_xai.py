"""
XTGAIA Stage 6: Strategic XAI via exact reward decomposition.

The Double DQN has one Q head per reward channel, so for every action
    Q(S, a) = w_contain * Q_contain(S, a) - w_disrupt * Q_disrupt(S, a) - w_cost * Q_cost(S, a)
holds exactly. This module reads those heads (no heuristics), compares the
chosen action with the runner-up channel by channel, and explains the choice
using the real state features (risk, p_t, crown-jewel/admin flags, telemetry).
It works the same on mock, simulated and live Person 1/2 states.
"""

from typing import Any, Dict, Optional

import torch

from src.role3_decision.actions import ACTIONS, NUM_ACTIONS, label
from src.role3_decision.config import reward_weights
from src.role3_decision.double_dqn import CHANNELS, QNetwork, load_policy
from src.role3_decision.state import (IDX_AUTH, IDX_CONN, IDX_CPU, IDX_CROWN_JEWEL, IDX_CVSS, IDX_IS_ADMIN,
                                      IDX_P, IDX_RISK, StateLike, asset_criticality, encode_state,
                                      validate_state)


class StrategicXAIExplainer:
    def __init__(self, model: Optional[QNetwork] = None, weights: Optional[Dict[str, float]] = None):
        self.model = model
        self.weights = weights or reward_weights()

    def _model(self) -> QNetwork:
        if self.model is None:
            self.model = load_policy()
        return self.model

    def decompose(self, state: StateLike) -> Dict[str, Dict[str, float]]:
        """Per-action Q_contain / Q_disrupt / Q_cost (raw heads), weighted contributions and Q_total."""
        enc = encode_state(state)
        with torch.no_grad():
            q_ch = self._model().forward_decomposed(torch.as_tensor(enc).unsqueeze(0))[0].cpu().numpy()
        w = self.weights
        out = {}
        for a in range(NUM_ACTIONS):
            qc, qd, qk = (float(q_ch[i, a]) for i in range(len(CHANNELS)))
            contrib = {"contain": w["contain"] * qc, "disrupt": -w["disrupt"] * qd, "cost": -w["cost"] * qk}
            out[f"a{a}"] = {
                "Q_contain": round(qc, 4), "Q_disrupt": round(qd, 4), "Q_cost": round(qk, 4),
                "weighted_contain": round(contrib["contain"], 4),
                "weighted_disrupt": round(contrib["disrupt"], 4),
                "weighted_cost": round(contrib["cost"], 4),
                "Q_total": round(sum(contrib.values()), 4),
            }
        return out

    def explain(self, state: StateLike, target_host: str = "unknown-host") -> Dict[str, Any]:
        raw = validate_state(state)
        enc = encode_state(raw)
        per_action = self.decompose(raw)
        ranked = sorted(per_action, key=lambda k: per_action[k]["Q_total"], reverse=True)
        chosen, runner = int(ranked[0][1:]), int(ranked[1][1:])
        c, r = per_action[f"a{chosen}"], per_action[f"a{runner}"]
        deltas = {ch: round(c[f"weighted_{ch}"] - r[f"weighted_{ch}"], 4) for ch in CHANNELS}
        context = {
            "threat_probability": round(float(raw[IDX_P]), 4),
            "topological_risk_raw": round(float(raw[IDX_RISK]), 4),
            "topological_risk_norm": round(float(enc[IDX_RISK]), 4),
            "asset_criticality": round(asset_criticality(enc), 2),
            "is_admin": bool(raw[IDX_IS_ADMIN]), "is_crown_jewel": bool(raw[IDX_CROWN_JEWEL]),
            "conn_count": float(raw[IDX_CONN]), "auth_fails": float(raw[IDX_AUTH]),
            "cpu_spike": round(float(raw[IDX_CPU]), 3), "cvss_max": round(float(raw[IDX_CVSS]), 1),
        }
        return {
            "target_host": target_host,
            "chosen_action": chosen, "chosen_name": ACTIONS[chosen].name,
            "runner_up_action": runner, "runner_up_name": ACTIONS[runner].name,
            "q_margin": round(c["Q_total"] - r["Q_total"], 4),
            "channel_advantage_vs_runner_up": deltas,
            "decisive_channel": max(deltas, key=lambda k: deltas[k]),
            "per_action": per_action,
            "context": context,
            "narrative": self._narrative(chosen, runner, deltas, context, target_host),
        }

    @staticmethod
    def _narrative(chosen, runner, deltas, ctx, host) -> str:
        role = ("crown-jewel admin asset" if ctx["is_crown_jewel"] and ctx["is_admin"] else
                "crown-jewel asset" if ctx["is_crown_jewel"] else
                "admin host" if ctx["is_admin"] else "standard host")
        lines = [
            f"XTGAIA selected {label(chosen)} for {host} over runner-up {label(runner)}.",
            f"Context: p_t={ctx['threat_probability']:.2f}, risk={ctx['topological_risk_norm']:.2f} "
            f"(raw {ctx['topological_risk_raw']:.2f}), {role}, criticality={ctx['asset_criticality']:.1f}, "
            f"auth_fails={ctx['auth_fails']:.0f}, conn_count={ctx['conn_count']:.0f}, cvss_max={ctx['cvss_max']:.1f}.",
        ]
        reasons = {
            "contain": "it is expected to contain more of the threat",
            "disrupt": "it causes less business disruption",
            "cost": "it is cheaper to execute",
        }
        favoured = [f"{reasons[ch]} ({d:+.2f})" for ch, d in deltas.items() if d > 0]
        conceded = [f"{ch} {d:+.2f}" for ch, d in deltas.items() if d < 0]
        if favoured:
            lines.append("Chosen because " + "; ".join(favoured) + ".")
        if conceded:
            lines.append("Trade-off accepted vs runner-up: " + ", ".join(conceded) + ".")
        return "\n".join(lines)

    def export_stix_xai_extension(self, state: StateLike, target_host: str = "unknown-host") -> Dict[str, Any]:
        """STIX 2.1 property-extension carrying the Stage 6 explanation."""
        exp = self.explain(state, target_host)
        return {
            "type": "extension-definition",
            "spec_version": "2.1",
            "name": "XTGAIA Strategic Reward Decomposition XAI",
            "extension_types": ["property-extension"],
            "x_xtgaia_strategic_xai": {
                "target_host": target_host,
                "selected_playbook": ACTIONS[exp["chosen_action"]].playbook,
                "action_code": exp["chosen_action"],
                "q_values": {k: v["Q_total"] for k, v in exp["per_action"].items()},
                "decision_margin": exp["q_margin"],
                "sub_reward_decomposition": exp["per_action"][f"a{exp['chosen_action']}"],
                "runner_up_comparison": {
                    "runner_up_playbook": ACTIONS[exp["runner_up_action"]].playbook,
                    "runner_up_breakdown": exp["per_action"][f"a{exp['runner_up_action']}"],
                    "channel_advantage": exp["channel_advantage_vs_runner_up"],
                },
                "context": exp["context"],
                "analyst_narrative": exp["narrative"],
            },
        }


if __name__ == "__main__":
    from src.schemas.state_vector import XTGAIAStateVector
    print(StrategicXAIExplainer().explain(XTGAIAStateVector.mock(), "DB-Primary-01")["narrative"])
