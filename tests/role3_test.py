from src.schemas.state_vector import XTGAIAStateVector
from src.role3_decision.double_dqn import DoubleDQNNetwork

def test_dqn_action_selection():
    dqn = DoubleDQNNetwork()
    mock_state = XTGAIAStateVector.mock().to_numpy()
    action = dqn.select_action(mock_state)
    assert 0 <= action <= 4, "Action must be in range [a0, a4]"