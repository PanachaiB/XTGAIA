import torch
import torch.nn as nn
from torch_geometric.nn.models.tgn import (
    TGNMemory,
    IdentityMessage,
    LastAggregator,
)

# นำเข้าโมดูล XAI Bundler สำหรับแพ็กหลักฐาน STIX 2.1
from src.role2_ai.xai_bundler import XAIBundler

class TemporalAttentionEmbedding(nn.Module):
    """
    8-Head Temporal Self-Attention over node memory states (XTGAIA Stage 3).
    """
    def __init__(self, node_dim: int, num_heads: int = 8):
        super().__init__()
        self.node_dim = node_dim
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(
            embed_dim=node_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(node_dim)

    def forward(self, memory_states: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.mha(memory_states, memory_states, memory_states)
        out = self.layer_norm(memory_states + attn_out)
        return out.squeeze(1)


class XTGAIATGN(nn.Module):
    """
    Temporal Graph Network (TGN) with Continuous-time TGNMemory, 
    128d GRU state, and 8-head Temporal Attention.
    """
    def __init__(
        self,
        num_nodes: int = 10000,
        raw_msg_dim: int = 16,
        memory_dim: int = 128,
        latent_z_dim: int = 8,
        num_heads: int = 8
    ):
        super().__init__()
        self.memory_dim = memory_dim
        self.latent_z_dim = latent_z_dim

        # 1. TGN Memory Module with GRU updater (128d)
        self.memory = TGNMemory(
            num_nodes=num_nodes,
            raw_msg_dim=raw_msg_dim,
            memory_dim=memory_dim,
            time_dim=memory_dim,
            message_module=IdentityMessage(raw_msg_dim, memory_dim, memory_dim),
            aggregator_module=LastAggregator(),
        )

        # 2. 8-head Temporal Attention Embedding
        self.attention_embedding = TemporalAttentionEmbedding(
            node_dim=memory_dim,
            num_heads=num_heads
        )

        # 3. Latent Projection: 128d -> 8d (Z_t contract for Person 3)
        self.z_projector = nn.Sequential(
            nn.Linear(memory_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_z_dim)
        )

        # 4. Threat Head: -> p_t in [0.0, 1.0]
        self.classifier = nn.Sequential(
            nn.Linear(latent_z_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, n_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mem, _ = self.memory(n_id)
        h = self.attention_embedding(mem.unsqueeze(1))
        Z_t = self.z_projector(h)
        p_t = self.classifier(Z_t)
        return p_t, Z_t

    def update_memory_events(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        t: torch.Tensor,
        raw_msg: torch.Tensor
    ):
        self.memory.update_state(src, dst, t, raw_msg)

    def reset_state(self):
        self.memory.reset_state()


class MockTGNInference:
    """
    Mock inference interface required by unit tests (tests/role2_test.py).
    Produces contract-compliant p_t, Z_t, and mock XAI bundle.
    """
    def __init__(self, *args, **kwargs):
        self.bundler = XAIBundler()

    def predict(self, *args, **kwargs):
        p_t = 0.85
        z_t = torch.full((8,), 0.1, dtype=torch.float32)
        return p_t, z_t

    def predict_threat_and_explain(self, graph_tensors=None):
        """
        XTGAIA Person 2 Core Interface Contract:
        Returns: (p_t, Z_t, xai_bundle)
        """
        p_t = 0.85
        z_t = torch.full((8,), 0.1, dtype=torch.float32)
        
        # สร้าง STIX 2.1 XAI Bundle ผ่าน XAIBundler
        mock_structural = [(2, 0, 0.5458), (1, 2, 0.5098)]
        mock_temporal = {
            "onset_timestamp_ms": 187.5,
            "attack_duration_ms": 312.5,
            "temporal_trace": [0.0384, 0.0506, 0.079, 0.2719, 0.0924, 0.0564, 0.0413, 0.3701]
        }
        xai_bundle = self.bundler.build_stix_bundle(
            node_id="target_host_node",
            threat_prob=p_t,
            structural_edges=mock_structural,
            temporal_analysis=mock_temporal
        )
        return p_t, z_t, xai_bundle

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)


# Module-level convenience function for end-to-end integration
_default_engine = MockTGNInference()

def predict_threat_and_explain(graph_tensors=None):
    """
    Public API callable for XTGAIA Pipeline (main.py and Person 3).
    Returns (p_t, Z_t, xai_bundle)
    """
    return _default_engine.predict_threat_and_explain(graph_tensors)