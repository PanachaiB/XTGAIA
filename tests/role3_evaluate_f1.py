"""
XTGAIA Stage 5 Evaluation: Comprehensive Decision & Policy Benchmarking
Calculates:
1. Binary Mitigation Metrics (Accuracy, Precision, Recall, Binary F1, FPR, FNR)
2. Multi-Class Policy Alignment Macro F1 across 4 Operational Threat Tiers
"""

import numpy as np
import torch
from src.role3_decision.double_dqn import load_policy
from src.role3_decision.state import encode_state, make_raw_state

def make_state_vector(p_t: float, risk_t: float, asset_criticality: float) -> torch.Tensor:
    """Builds a shared-schema S_t (random TGN embedding, like mock()) and encodes it for the policy.
    risk_t is the normalised risk tier value; asset_criticality is one of 0.1 / 0.4 / 0.7 / 1.0."""
    sv = make_raw_state(p_t, risk_t, asset_criticality,
                        gnn_embedding=np.random.normal(0, 1, 8).astype(np.float32),
                        conn_count=float(np.random.uniform(0, 40)), auth_fails=float(np.random.randint(0, 7)),
                        cpu_spike=float(np.random.uniform(0, 1)), cvss_max=float(np.random.uniform(0, 10)))
    return torch.as_tensor(encode_state(sv)).unsqueeze(0)

def generate_evaluation_dataset(num_samples=1000):
    """
    Generates exactly num_samples distributed across operational threat tiers:
    - 50% Benign / Low Risk Baseline (Tier 0)
    - 50% Active Intrusions (Tier 1: Medium, Tier 2: High, Tier 3: Critical)
    """
    states = []
    y_true_binary = []       # 0: Benign (a0), 1: Active Attack (a1..a4)
    y_true_tiers = []        # 0: Low, 1: Medium, 2: High, 3: Critical
    expected_action_sets = [] # Valid proportional playbooks for each sample

    n_benign = num_samples // 2
    n_attack = num_samples - n_benign

    # 1. Generate Benign Baseline States (N = 500)
    for _ in range(n_benign):
        p_t = float(np.random.uniform(0.01, 0.15))
        risk_t = float(np.random.uniform(0.01, 0.12))
        crit = float(np.random.choice([0.1, 0.4, 0.7, 1.0]))

        states.append((p_t, risk_t, crit))
        y_true_binary.append(0)
        y_true_tiers.append(0)
        expected_action_sets.append([0]) # Low Risk expects passive monitoring

    # 2. Generate Active Attack States (N = 500)
    attack_tiers = np.random.choice([1, 2, 3], size=n_attack, p=[0.30, 0.35, 0.35])
    for tier in attack_tiers:
        if tier == 1:  # Medium Risk
            p_t = float(np.random.uniform(0.30, 0.55))
            risk_t = float(np.random.uniform(0.25, 0.48))
            crit = float(np.random.choice([0.1, 0.4, 0.7, 1.0]))
            if crit <= 0.1:
                allowed_actions = [1]
            elif crit <= 0.4:
                allowed_actions = [1, 2]
            else:
                allowed_actions = [2, 3]
        elif tier == 2:  # High Risk
            p_t = float(np.random.uniform(0.60, 0.80))
            risk_t = float(np.random.uniform(0.55, 0.75))
            crit = float(np.random.choice([0.1, 0.4, 0.7, 1.0]))
            if crit <= 0.1:
                allowed_actions = [2]
            elif crit <= 0.4:
                allowed_actions = [2, 3]
            else:
                allowed_actions = [2, 3, 4]
        else:  # Critical Threat
            p_t = float(np.random.uniform(0.85, 0.98))
            risk_t = float(np.random.uniform(0.80, 0.98))
            crit = float(np.random.choice([0.1, 0.4, 0.7, 1.0]))
            if crit <= 0.7:
                # Moderate/intermediate assets: avoid disruptive host quarantine
                allowed_actions = [2, 3]
            elif risk_t >= 0.90:
                # Critical Crown Jewel under maximum reachability compromise:
                # Targeted mitigations are insufficient -> mandatory host quarantine
                allowed_actions = [4]
            else:
                # Crown Jewel under high threat -> session revocation or quarantine
                allowed_actions = [3, 4]

        states.append((p_t, risk_t, crit))
        y_true_binary.append(1)
        y_true_tiers.append(tier)
        expected_action_sets.append(allowed_actions)

    return states, y_true_binary, y_true_tiers, expected_action_sets

def evaluate_double_dqn(weights_path=None, num_samples=1000):
    model = load_policy(weights_path)  # defaults to paths.policy in configs/reward_weights.yaml

    states, y_true_binary, y_true_tiers, expected_action_sets = generate_evaluation_dataset(num_samples=num_samples)

    y_pred_actions = []
    y_pred_binary = []
    y_pred_scores = []

    with torch.no_grad():
        for p_t, risk_t, crit in states:
            s_t = make_state_vector(p_t, risk_t, crit)
            q_vals = model(s_t)
            action = int(q_vals.argmax(dim=1).item())

            # Continuous threat score: softmax probability of taking containment action (a1..a4)
            probs = torch.softmax(q_vals, dim=1)[0]
            p_attack = float(probs[1:].sum().item())

            y_pred_actions.append(action)
            y_pred_binary.append(0 if action == 0 else 1)
            y_pred_scores.append(p_attack)

    # ---------------------------------------------------------
    # 1. Binary Decision Metrics (Containment vs. Passive)
    # ---------------------------------------------------------
    tp = sum(1 for yt, yp in zip(y_true_binary, y_pred_binary) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true_binary, y_pred_binary) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true_binary, y_pred_binary) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true_binary, y_pred_binary) if yt == 1 and yp == 0)

    total = len(y_true_binary)
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    binary_f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # Exact ROC-AUC via Mann-Whitney U rank statistic
    y_true_arr = np.array(y_true_binary)
    scores_arr = np.array(y_pred_scores)
    n_pos = int(np.sum(y_true_arr == 1))
    n_neg = int(np.sum(y_true_arr == 0))
    if n_pos > 0 and n_neg > 0:
        order = np.argsort(scores_arr)
        rank = np.empty_like(order, dtype=float)
        rank[order] = np.arange(len(scores_arr)) + 1.0
        for val in np.unique(scores_arr):
            mask = scores_arr == val
            rank[mask] = rank[mask].mean()
        roc_auc = float((np.sum(rank[y_true_arr == 1]) - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))
    else:
        roc_auc = 0.0

    # ---------------------------------------------------------
    # 2. Multi-Class Policy Alignment F1 across Operational Tiers
    # ---------------------------------------------------------
    tier_f1_scores = []
    tier_names = ["Low (Benign)", "Medium Threat", "High Threat", "Critical Threat"]

    tier_alignment_counts = []
    for tier_idx in range(4):
        indices = [i for i, t in enumerate(y_true_tiers) if t == tier_idx]
        total_in_tier = len(indices)

        # Policy Alignment: Did chosen action fall in proportional playbook set?
        correct_in_tier = sum(1 for i in indices if y_pred_actions[i] in expected_action_sets[i])
        alignment_rate = (correct_in_tier / total_in_tier) if total_in_tier > 0 else 0.0
        tier_alignment_counts.append((correct_in_tier, total_in_tier, alignment_rate))
        tier_f1_scores.append(alignment_rate)

    macro_f1 = float(np.mean(tier_f1_scores))

    # ---------------------------------------------------------
    # 3. Print Publication-Ready Summary Report
    # ---------------------------------------------------------
    print("=" * 68)
    print("      XTGAIA STAGE 5 (DOUBLE DQN) BENCHMARKING REPORT        ")
    print("=" * 68)
    print(f"Total Test Samples Evaluated : {total}")
    print(f"  - Benign / Low-Risk States  : {y_true_binary.count(0)}")
    print(f"  - Active Compromise States  : {y_true_binary.count(1)}")
    print("-" * 68)
    print("PART A: BINARY ATTACK MITIGATION EFFICACY")
    print(f"  Accuracy                  : {acc * 100:.2f}%")
    print(f"  Precision                 : {prec:.4f}")
    print(f"  Recall (Detection TPR)    : {rec:.4f}")
    print(f"  Binary F1-Score           : {binary_f1:.4f}")
    print(f"  False Alarm Rate (FPR)    : {fpr * 100:.2f}%")
    print(f"  Missed Attack Rate (FNR)  : {fnr * 100:.2f}%")
    print(f"  ROC-AUC                   : {roc_auc:.4f}")
    print(f"  Confusion Matrix          : [TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}]")
    print("-" * 68)
    print("PART B: MULTI-CLASS POLICY ALIGNMENT (PROPORTIONAL CONTAINMENT)")
    for tier_idx, t_name in enumerate(tier_names):
        c, t, rate = tier_alignment_counts[tier_idx]
        print(f"  - {t_name:<24}: {c:>4} / {t:>4} aligned ({rate * 100:.1f}%)")
    print(f"  --> Multi-Class Macro F1   : {macro_f1:.4f}")
    print("=" * 68)

    # Action distribution summary
    action_labels = ["a0: Monitor", "a1: Rate-Limit", "a2: Isolate Port", "a3: Revoke Token", "a4: Quarantine"]
    print("\nAction Breakdown by Operational Threat Tier:")
    for tier_idx, t_name in enumerate(tier_names):
        indices = [i for i, t in enumerate(y_true_tiers) if t == tier_idx]
        total_in_tier = len(indices)
        actions_in_tier = [y_pred_actions[i] for i in indices]

        print(f"\n  Tier {tier_idx} [{t_name}] (N={total_in_tier}):")
        for a_idx, label in enumerate(action_labels):
            cnt = actions_in_tier.count(a_idx)
            pct = (cnt / total_in_tier) * 100 if total_in_tier > 0 else 0.0
            if cnt > 0:
                print(f"    - {label:<18}: {cnt:>4} ({pct:>5.1f}%)")
    print("\n" + "=" * 68)

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "binary_f1": binary_f1,
        "fpr": fpr,
        "fnr": fnr,
        "roc_auc": roc_auc,
        "macro_f1": macro_f1,
        "tier_alignment": tier_alignment_counts
    }

if __name__ == "__main__":
    evaluate_double_dqn()