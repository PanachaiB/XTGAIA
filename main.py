"""
XTGAIA Central Pipeline Orchestrator
Connects: Person 1 (Graph/Risk) -> Person 2 (TGN/XAI) -> Person 3 (Double DQN/SOAR)
"""
from src.schemas.state_vector import XTGAIAStateVector
from src.role1_graph.graph_engine import ReachabilityRiskEngine
from src.role2_ai.tgn_model import MockTGNInference
from src.role3_decision.double_dqn import DoubleDQNNetwork

def run_pipeline():
    print("[*] Initializing XTGAIA 7-Stage Pipeline...")
    # 1. Person 1 evaluates graph & risk
    risk_engine = ReachabilityRiskEngine(crown_jewels={"10.0.0.1": 10.0})

    # 2. Person 2 predicts threat & generates XAI
    tgn = MockTGNInference()

    # 3. Assemble 16-d state vector
    state = XTGAIAStateVector.mock().to_numpy()

    # 4. Person 3 selects action & dispatches to SOAR
    dqn = DoubleDQNNetwork()
    action = dqn.select_action(state)
    print(f"[+] Optimal Action Selected: a_{action}")

if __name__ == "__main__":
    run_pipeline()