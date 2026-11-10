import torch
from src.role2_ai.tgn_model import MockTGNInference

def test_tgn_output_dimensions():
    model = MockTGNInference()
    p_t, z_t = model(None, None)
    assert 0.0 <= p_t <= 1.0
    assert z_t.shape == (8,)