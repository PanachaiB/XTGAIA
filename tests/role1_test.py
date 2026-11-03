import networkx as nx
from src.role1_graph.graph_engine import ReachabilityRiskEngine

def test_risk_calculation():
    engine = ReachabilityRiskEngine(crown_jewels={"10.0.0.1": 10.0})
    G = nx.DiGraph()
    G.add_edge("192.168.1.100", "10.0.0.1", weight=0.2) # High exploitability -> low cost 0.2
    risk = engine.calculate_node_risk(G, "192.168.1.100", p_t=0.9)
    assert risk > 0.0, "Risk should be non-zero when a path exists to Crown Jewel"