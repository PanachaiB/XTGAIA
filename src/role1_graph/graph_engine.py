import networkx as nx

class ReachabilityRiskEngine:
    def __init__(self, crown_jewels: dict):
        # crown_jewels: e.g. {"10.0.0.1": 10.0} (Asset Criticality weights)
        self.crown_jewels = crown_jewels

    def calculate_exploit_friction(self, cvss: float, priv_tier: float, port_risk: float) -> float:
        """Algorithm 2: Exploit(u, w) = 0.5*S_vuln + 0.3*S_priv + 0.2*S_port"""
        s_vuln = min(cvss / 10.0, 1.0)
        exploit = 0.50 * s_vuln + 0.30 * priv_tier + 0.20 * port_risk
        return 1.0 - exploit  # Traversal cost omega(u, w)

    def calculate_node_risk(self, graph: nx.DiGraph, node: str, p_t: float) -> float:
        """Computes Dijkstra distance decay toward Crown Jewels."""
        total_risk = 0.0
        if node not in graph:
            return 0.0
        for cj, criticality in self.crown_jewels.items():
            # A crown jewel with no traffic in the current window is not in the graph
            # (nx.has_path would raise NodeNotFound) and is unreachable right now.
            if cj in graph and nx.has_path(graph, node, cj):
                dist = nx.dijkstra_path_length(graph, node, cj, weight='weight')
                total_risk += criticality / ((dist + 1.0) ** 2)
        return float(p_t * total_risk)