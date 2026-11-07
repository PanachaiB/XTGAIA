import networkx as nx
from typing import Dict, Any, List

class ReachabilityRiskEngine:
    def __init__(self, crown_jewels: dict):
        # crown_jewels: เช่น {"10.0.0.1": 10.0} (Asset Criticality weights)
        self.crown_jewels = crown_jewels

    def calculate_exploit_friction(self, cvss: float, priv_tier: float, port_risk: float) -> float:
        """Algorithm 2: Exploit(u, w) = 0.5*S_vuln + 0.3*S_priv + 0.2*S_port"""
        s_vuln = min(cvss / 10.0, 1.0)
        exploit = 0.50 * s_vuln + 0.30 * priv_tier + 0.20 * port_risk
        return max(1.0 - exploit, 0.0001)  # ป้องกัน distance เป็น 0 ใน Dijkstra

    def calculate_node_risk(self, graph: nx.DiGraph, node: str, p_t: float) -> float:
        """Computes Dijkstra distance decay toward Crown Jewels."""
        total_risk = 0.0
        for cj, criticality in self.crown_jewels.items():
            if nx.has_path(graph, node, cj):
                dist = nx.dijkstra_path_length(graph, node, cj, weight='weight')
                total_risk += criticality / ((dist + 1.0) ** 2)
        return float(p_t * total_risk)

    def extract_phi_t(self, graph: nx.DiGraph, node: str, host_telemetry: Dict[str, Any]) -> List[float]:
        """
        สร้าง Host Contextual Telemetry Flags 6 มิติ (Phi_t) ส่งให้ Double DQN (Person 3)
        [is_admin, is_crown_jewel, conn_count, auth_fails, cpu_spike, cvss_max]
        """
        is_admin = float(host_telemetry.get("is_admin", 0))
        is_cj = 1.0 if node in self.crown_jewels else 0.0
        conn_count = float(min(graph.degree(node) if node in graph else 0, 100) / 100.0)
        auth_fails = float(min(host_telemetry.get("auth_fails", 0), 10) / 10.0)
        cpu_spike = float(host_telemetry.get("cpu_spike", 0.0))
        cvss_max = float(host_telemetry.get("cvss_max", 0.0) / 10.0)
        return [is_admin, is_cj, conn_count, auth_fails, cpu_spike, cvss_max]