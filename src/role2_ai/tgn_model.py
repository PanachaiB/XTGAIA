import torch
import torch.nn as nn

class MockTGNInference(nn.Module):
    """Stub model allowing Person 2 to test tensor pipeline before offline training finishes."""
    def __init__(self):
        super().__init__()
        self.embedding_dim = 8

    def forward(self, edge_index, edge_attr):
        # Outputs mock threat probability and 8d latent embedding
        p_t = 0.88
        z_t = torch.randn(8, dtype=torch.float32)
        return p_t, z_t