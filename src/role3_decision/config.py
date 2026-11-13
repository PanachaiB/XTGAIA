"""
Role 3 configuration loader.

Single source of truth for reward weights, action costs/severities, state
normalisation and training hyper-parameters (configs/reward_weights.yaml).
Paths are resolved against the repository root so the code works no matter
which directory it is launched from. Missing keys raise immediately instead of
silently falling back to hardcoded defaults.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "reward_weights.yaml"

# Same .env for host Python and Docker: on the host it is loaded here; in Docker
# compose injects it via env_file (and .env is not inside the image). Variables
# already set in the process environment always win (override=False).
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env", override=False)
except ImportError:  # python-dotenv not installed: rely on the process environment
    pass

_REQUIRED = {
    "reward_weights": ["contain", "disrupt", "cost", "rollback"],
    "action_costs": [f"a{i}" for i in range(5)],
    "action_severities": [f"a{i}" for i in range(5)],
    "state_normalization": ["risk_scale", "conn_count_cap", "auth_fails_cap", "cvss_max"],
    "paths": ["policy", "feedback_buffer"],
    "hyperparameters": ["gamma", "learning_rate", "batch_size",
                        "replay_buffer_capacity", "target_update_interval", "exploration"],
}


def resolve_path(p: str) -> Path:
    """Resolve a config path relative to the repository root."""
    path = Path(p)
    return path if path.is_absolute() else REPO_ROOT / path


@lru_cache(maxsize=8)
def _load(path_str: str) -> Dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Role 3 config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    for section, keys in _REQUIRED.items():
        if section not in cfg:
            raise KeyError(f"{path}: missing section '{section}'")
        missing = [k for k in keys if k not in cfg[section]]
        if missing:
            raise KeyError(f"{path}: section '{section}' missing keys {missing}")
    return cfg


def load_config(path: str = None) -> Dict[str, Any]:
    """Load configs/reward_weights.yaml (override with XTGAIA_REWARD_CONFIG)."""
    chosen = path or os.getenv("XTGAIA_REWARD_CONFIG") or str(DEFAULT_CONFIG_PATH)
    return _load(str(resolve_path(chosen)))


def reward_weights(cfg: Dict[str, Any] = None) -> Dict[str, float]:
    cfg = cfg or load_config()
    return {k: float(v) for k, v in cfg["reward_weights"].items()}


def policy_path(cfg: Dict[str, Any] = None) -> Path:
    cfg = cfg or load_config()
    return resolve_path(cfg["paths"]["policy"])


def feedback_buffer_path(cfg: Dict[str, Any] = None) -> Path:
    cfg = cfg or load_config()
    return resolve_path(cfg["paths"]["feedback_buffer"])
