"""
SHARED SCHEMA: 16-Dimensional State Vector Contract (Algorithm 3)
Shared between Person 1 (Risk), Person 2 (TGN/XAI), and Person 3 (Double DQN)
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class XTGAIAStateVector:
    # 1. 8-dimensional pooled TGN latent embedding (Person 2)
    gnn_embedding: np.ndarray  # shape: (8,), dtype: float32
    # 2. 1-dimensional TGN threat probability (Person 2)
    threat_probability: float  # p_t(v) in [0.0, 1.0]
    # 3. 1-dimensional reachability-decayed risk score (Person 1)
    topological_risk: float    # Risk_t(v) >= 0.0
    # 4. 6-dimensional host context telemetry flags (Person 1/Logs)
    # [is_admin, is_crown_jewel, external_conn_count, failed_logins, cpu_spike, cvss_max]
    telemetry_flags: np.ndarray  # shape: (6,), dtype: float32

    def to_numpy(self) -> np.ndarray:
        """Flattens components into a strictly validated 16-d float32 vector."""
        assert self.gnn_embedding.shape == (8,), "GNN embedding must be 8-dimensional"
        assert self.telemetry_flags.shape == (6,), "Telemetry flags must be 6-dimensional"

        vec = np.concatenate([
            self.gnn_embedding,
            np.array([self.threat_probability, self.topological_risk], dtype=np.float32),
            self.telemetry_flags
        ]).astype(np.float32)

        assert vec.shape == (16,), f"State vector must be exactly 16-d, got {vec.shape}"
        return vec

    @classmethod
    def mock(cls) -> "XTGAIAStateVector":
        """Generates a synthetic 16-d vector for offline testing (used by Person 3)."""
        return cls(
            gnn_embedding=np.random.normal(0, 1, size=(8,)).astype(np.float32),
            threat_probability=float(np.random.uniform(0.7, 0.99)),
            topological_risk=float(np.random.uniform(0.5, 5.0)),
            telemetry_flags=np.array([1.0, 0.0, 15.0, 3.0, 0.85, 9.8], dtype=np.float32)
        )
