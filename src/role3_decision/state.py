"""
Role 3 view of the shared 16-d state contract (src/schemas/state_vector.py).

Contract: every state that enters Role 3 from outside is a *raw* schema vector,
either an XTGAIAStateVector or the float32 array produced by its to_numpy():

    S_t = [Z_t (8) || p_t (1) || Risk_t (1) || Phi_t (6)]
    Phi_t = [is_admin, is_crown_jewel, conn_count, auth_fails, cpu_spike, cvss_max]

`validate_state` rejects anything with the wrong type/shape/dtype, NaN/inf or
out-of-contract values. `encode_state` then maps the raw values onto the
bounded [0, 1] feature space the Double DQN is trained on. Training (env),
inference (dispatcher), XAI and rollback all go through the same encoder, so
mock data and real Person 1/2 data are treated identically.
"""

from typing import Union

import numpy as np

from src.role3_decision.config import load_config
from src.schemas.state_vector import XTGAIAStateVector

STATE_DIM = 16
Z_SLICE = slice(0, 8)
IDX_P = 8
IDX_RISK = 9
PHI_FIELDS = ["is_admin", "is_crown_jewel", "conn_count", "auth_fails", "cpu_spike", "cvss_max"]
IDX_IS_ADMIN, IDX_CROWN_JEWEL, IDX_CONN, IDX_AUTH, IDX_CPU, IDX_CVSS = range(10, 16)

StateLike = Union[XTGAIAStateVector, np.ndarray]


def validate_state(state: StateLike) -> np.ndarray:
    """Return a validated raw (16,) float32 copy of `state` or raise ValueError."""
    if isinstance(state, XTGAIAStateVector):
        try:
            vec = state.to_numpy()
        except AssertionError as exc:  # schema uses asserts; surface them as ValueError
            raise ValueError(f"Invalid XTGAIAStateVector: {exc}") from exc
    elif isinstance(state, np.ndarray):
        vec = state
    else:
        raise ValueError(f"State must be XTGAIAStateVector or np.ndarray, got {type(state).__name__}")

    if vec.shape != (STATE_DIM,):
        raise ValueError(f"State vector must have shape (16,), got {vec.shape}")
    if vec.dtype != np.float32:
        raise ValueError(f"State vector must be float32, got {vec.dtype}")
    if not np.all(np.isfinite(vec)):
        raise ValueError("State vector contains NaN or inf")

    checks = [
        (0.0 <= vec[IDX_P] <= 1.0, "threat_probability p_t must be in [0, 1]"),
        (vec[IDX_RISK] >= 0.0, "topological_risk must be >= 0"),
        (vec[IDX_IS_ADMIN] in (0.0, 1.0), "is_admin must be 0 or 1"),
        (vec[IDX_CROWN_JEWEL] in (0.0, 1.0), "is_crown_jewel must be 0 or 1"),
        (vec[IDX_CONN] >= 0.0, "conn_count must be >= 0"),
        (vec[IDX_AUTH] >= 0.0, "auth_fails must be >= 0"),
        (0.0 <= vec[IDX_CPU] <= 1.0, "cpu_spike must be in [0, 1]"),
        (0.0 <= vec[IDX_CVSS] <= 10.0, "cvss_max must be in [0, 10]"),
    ]
    for ok, msg in checks:
        if not ok:
            raise ValueError(msg)
    return vec.copy()


def encode_state(state: StateLike, cfg: dict = None) -> np.ndarray:
    """Validate a raw schema state and normalise it to the policy's [0, 1] input space."""
    raw = validate_state(state)
    norm = (cfg or load_config())["state_normalization"]
    enc = np.empty(STATE_DIM, dtype=np.float32)
    enc[Z_SLICE] = 1.0 / (1.0 + np.exp(-raw[Z_SLICE]))                      # unbounded embedding -> (0,1)
    enc[IDX_P] = raw[IDX_P]
    enc[IDX_RISK] = 1.0 - np.exp(-raw[IDX_RISK] / float(norm["risk_scale"]))  # saturating risk
    enc[IDX_IS_ADMIN] = raw[IDX_IS_ADMIN]
    enc[IDX_CROWN_JEWEL] = raw[IDX_CROWN_JEWEL]
    enc[IDX_CONN] = min(raw[IDX_CONN] / float(norm["conn_count_cap"]), 1.0)
    enc[IDX_AUTH] = min(raw[IDX_AUTH] / float(norm["auth_fails_cap"]), 1.0)
    enc[IDX_CPU] = raw[IDX_CPU]
    enc[IDX_CVSS] = raw[IDX_CVSS] / float(norm["cvss_max"])
    return enc


def raw_risk_from_normalized(risk_norm: float, cfg: dict = None) -> float:
    """Inverse of the risk normalisation (used by the simulator and tests)."""
    scale = float((cfg or load_config())["state_normalization"]["risk_scale"])
    risk_norm = min(max(float(risk_norm), 0.0), 0.999)
    return float(-scale * np.log(1.0 - risk_norm))


def asset_criticality(encoded: np.ndarray) -> float:
    """Asset criticality derived from the schema flags: 0.1 / 0.4 (admin) / 0.7 (crown jewel) / 1.0 (both)."""
    # Rounded: float32 flags otherwise give 0.10000000149 and break tier comparisons like crit <= 0.1.
    return round(0.1 + 0.3 * float(encoded[IDX_IS_ADMIN]) + 0.6 * float(encoded[IDX_CROWN_JEWEL]), 2)


def flags_for_criticality(crit: float) -> tuple:
    """(is_admin, is_crown_jewel) that reproduce a criticality tier (inverse of asset_criticality)."""
    return {0.1: (0.0, 0.0), 0.4: (1.0, 0.0), 0.7: (0.0, 1.0), 1.0: (1.0, 1.0)}[round(float(crit), 1)]


def make_raw_state(p_t: float, risk_norm: float, crit: float, gnn_embedding: np.ndarray = None,
                   conn_count: float = 5.0, auth_fails: float = 1.0, cpu_spike: float = 0.2,
                   cvss_max: float = 5.0, cfg: dict = None) -> XTGAIAStateVector:
    """Build a schema-valid raw state for a given normalised risk and criticality tier."""
    is_admin, is_cj = flags_for_criticality(crit)
    if gnn_embedding is None:
        gnn_embedding = np.zeros(8, dtype=np.float32)
    return XTGAIAStateVector(
        gnn_embedding=np.asarray(gnn_embedding, dtype=np.float32),
        threat_probability=float(p_t),
        topological_risk=raw_risk_from_normalized(risk_norm, cfg),
        telemetry_flags=np.array([is_admin, is_cj, conn_count, auth_fails, cpu_spike, cvss_max],
                                 dtype=np.float32),
    )
