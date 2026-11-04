"""
tests/role3_test.py: Comprehensive Evaluation Suite for XTGAIA Stage 5 (Double DQN)
Includes:
1. Isolated inference test cases (Benign baseline vs Critical compromise).
2. Parametrized Risk x Asset Criticality Policy Robustness Matrix (12 operational scenarios).
"""

import os
import pytest
import torch
import numpy as np
from src.role3_decision.double_dqn import QNetwork, select_action
from src.schemas.state_vector import XTGAIAStateVector

def make_state_vector(p_t: float, risk_t: float, asset_criticality: float) -> torch.Tensor:
    """
    Constructs a 16-dimensional XTGAIA observation vector:
    S_t = [Z_t (8d) || p_t (1d) || Risk_t (1d) || Phi_t (6d)]
    """
    z_val = float((p_t + risk_t) / 2.0)
    z_t = [z_val] * 8
    # Phi_t: [criticality, vulns, exposure, trust, auth, protocols]
    phi_t = [asset_criticality, 0.5, 0.1, 0.2, 0.0, 1.0]

    raw_vec = np.array(z_t + [p_t, risk_t] + phi_t, dtype=np.float32)
    return torch.FloatTensor(raw_vec).unsqueeze(0)

# ===========================================================================
# 1. Robustness Matrix: 12 Operational Scenarios (Risk x Asset Criticality)
# ===========================================================================

TEST_SCENARIOS = [
    # Format: (Scenario Name, p_t, risk_t, asset_criticality, allowed_actions)

    # --- LOW RISK (p_t <= 0.15, risk_t <= 0.10) ---
    # Expected: Monitor (a0) or mild Rate-Limit (a1); absolutely no disruption (a2-a4)
    ("Low Risk, Low Crit (Dev Endpoint)", 0.05, 0.05, 0.1, [0, 1]),
    ("Low Risk, Med Crit (Workstation)", 0.08, 0.08, 0.4, [0, 1]),
    ("Low Risk, Crown Jewel (Domain Controller)", 0.10, 0.08, 1.0, [0, 1]),

    # --- MEDIUM RISK (0.25 <= p_t <= 0.50, 0.25 <= risk_t <= 0.45) ---
    # Expected: Mild containment (a1, a2) or session revocation (a3); no full quarantine (a4)
    ("Medium Risk, Low Crit (Dev Endpoint)", 0.35, 0.30, 0.1, [0, 1, 2]),
    ("Medium Risk, Med Crit (Internal Server)", 0.40, 0.35, 0.4, [1, 2, 3]),
    ("Medium Risk, Crown Jewel (Primary DB)", 0.45, 0.40, 1.0, [1, 2, 3]),

    # --- HIGH RISK (0.65 <= p_t <= 0.80, 0.60 <= risk_t <= 0.75) ---
    # Expected: Decisive intervention (a2, a3, a4); a0 (inaction) and a1 (mild) are insufficient
    ("High Risk, Low Crit (Dev Endpoint)", 0.70, 0.65, 0.1, [2, 3, 4]),
    ("High Risk, Med Crit (Internal Server)", 0.75, 0.70, 0.4, [2, 3, 4]),
    ("High Risk, Crown Jewel (Active Directory)", 0.78, 0.75, 1.0, [2, 3, 4]),

    # --- CRITICAL COMPROMISE (p_t >= 0.90, risk_t >= 0.85) ---
    # Expected: Severe containment / host isolation (a2, a3, a4) to protect Crown Jewels
    ("Critical Threat, Low Crit (Dev Endpoint)", 0.90, 0.85, 0.1, [2, 3, 4]),
    ("Critical Threat, Med Crit (Application Server)", 0.92, 0.88, 0.4, [2, 3, 4]),
    ("Critical Threat, Crown Jewel (Domain Controller / Vault)", 0.95, 0.92, 1.0, [2, 3, 4]),
]

@pytest.mark.parametrize("name, p_t, risk_t, crit, allowed_actions", TEST_SCENARIOS)
def test_agent_policy_robustness_grid(name, p_t, risk_t, crit, allowed_actions):
    """
    Evaluates whether the trained Double DQN policy monotonically scales response
    severity with respect to topological reachability risk and asset criticality.
    """
    weights_path = "data/models/double_dqn_weights.pt"
    assert os.path.exists(weights_path), f"Checkpoint not found at {weights_path}!"

    model = QNetwork()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    s_t = make_state_vector(p_t, risk_t, crit)

    with torch.no_grad():
        q_vals = model(s_t)
        chosen_action = q_vals.argmax(dim=1).item()

    q_summary = {f"a{i}": round(float(q), 3) for i, q in enumerate(q_vals[0])}

    assert chosen_action in allowed_actions, (
        f"\nFAILED [{name}]:"
        f"\n  Selected Action: a_{chosen_action} (Not in allowed set {allowed_actions})"
        f"\n  Input State    : p_t={p_t}, risk_t={risk_t}, Asset Criticality={crit}"
        f"\n  Q-Values       : {q_summary}"
    )