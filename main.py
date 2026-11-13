"""
XTGAIA Central Pipeline Orchestrator
Connects: Person 1 (Graph/Risk) -> Person 2 (TGN/XAI) -> Person 3 (Double DQN/SOAR)
"""
import networkx as nx
from src.schemas.state_vector import XTGAIAStateVector
from src.role1_graph.graph_engine import ReachabilityRiskEngine
from src.role2_ai.tgn_model import MockTGNInference
from src.role3_decision.soar_dispatcher import SOARDispatcher

def run_pipeline():
    print("[*] Initializing XTGAIA 7-Stage Pipeline...")
    # 1. Person 1 evaluates graph & risk
    risk_engine = ReachabilityRiskEngine(crown_jewels={"10.0.0.1": 10.0})

    # 2. Person 2 predicts threat & generates XAI
    tgn = MockTGNInference()

    # 3. Assemble 16-d state vector
    state = XTGAIAStateVector.mock().to_numpy()

    # 4. Person 3: Double DQN decision + Strategic XAI + SOAR dispatch (Stages 5-7)
    #    SOARDispatcher loads the trained Double DQN, infers the optimal playbook,
    #    generates a Strategic XAI bundle, dispatches to Shuffle SOAR, logs to TheHive,
    #    and mutates the in-memory topology.
    G_t = nx.Graph()
    G_t.add_node("10.0.0.1", criticality=10.0)

    dispatcher = SOARDispatcher()
    result = dispatcher.process_containment_pipeline(
        state_vector=state,
        target_host="10.0.0.1",
        target_ip="10.0.0.1",
        G_t=G_t,
    )

    print(f"[+] Optimal Action Selected: a_{result['action']} ({result['playbook']})")
    print(f"[+] SOAR Dispatched: {result['soar_dispatched']}")
    print(f"[+] TheHive Logged: {result['thehive_logged']}")

if __name__ == "__main__":
    run_pipeline()