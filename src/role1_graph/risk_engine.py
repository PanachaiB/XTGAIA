import json
import os
import networkx as nx
from typing import Dict, Any, List, Set

class ReachabilityRiskEngine:
    def __init__(self, crown_jewels: Set[str], config_path: str = "configs/mitre_t1021_ports.json"):
        self.crown_jewels = set(crown_jewels)
        self.port_weights = {}
        self.default_port_weight = 0.2
        self._load_port_config(config_path)

    def _load_port_config(self, config_path: str):
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = json.load(f)
                self.port_weights = {int(k): float(v) for k, v in cfg.get("ports", {}).items()}
                self.default_port_weight = float(cfg.get("default_port_weight", 0.2))

    def compute_edge_friction(self, s_vuln: float, s_priv: float, port: int) -> float:
        """
        สูตรคำนวณ Exploitability Friction:
        omega(u,w) = 1.0 - (0.50 * S_vuln + 0.30 * S_priv + 0.20 * S_port)
        """
        s_port = self.port_weights.get(port, self.default_port_weight)
        
        # ป้องกันค่าน้ำหนักเกิน 1.0
        score = (0.50 * s_vuln) + (0.30 * s_priv) + (0.20 * s_port)
        friction = 1.0 - min(max(score, 0.0), 0.9999)
        return max(friction, 0.0001)  # ป้องกันค่า friction เป็น 0 สำหรับ Dijkstra

    def calculate_risk(self, graph: nx.DiGraph, current_node: str, host_telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Algorithm 2: หาเส้นทางที่สั้นที่สุดไปยัง Crown Jewels ด้วย Dijkstra
        พร้อมสร้าง State Vector Components (Risk_t และ Phi_t 6 มิติ)
        """
        # อัปเดตค่าน้ำหนัก Friction ลงใน Edge ของ Graph
        weighted_graph = graph.copy()
        for u, v, data in weighted_graph.edges(data=True):
            port = data.get("port", 0)
            s_vuln = data.get("s_vuln", 0.5)
            s_priv = data.get("s_priv", 0.0)
            data["weight"] = self.compute_edge_friction(s_vuln, s_priv, port)

        min_path_distance = float("inf")
        for cj in self.crown_jewels:
            if cj in weighted_graph and current_node in weighted_graph:
                try:
                    dist = nx.dijkstra_path_length(weighted_graph, source=current_node, target=cj, weight="weight")
                    min_path_distance = min(min_path_distance, dist)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

        # คำนวณ Quadratic Path Decay: Risk = 1 / (1 + distance^2)
        if min_path_distance == float("inf"):
            risk_t = 0.0
        else:
            risk_t = float(1.0 / (1.0 + (min_path_distance ** 2)))

        # Telemetry Context 6 มิติ (Phi_t)
        # [is_admin, is_crown_jewel, conn_count, auth_fails, cpu_spike, cvss_max]
        phi_t: List[float] = [
            float(host_telemetry.get("is_admin", 0)),
            float(1.0 if current_node in self.crown_jewels else 0.0),
            float(min(graph.degree(current_node) if current_node in graph else 0, 100) / 100.0),
            float(min(host_telemetry.get("auth_fails", 0), 10) / 10.0),
            float(host_telemetry.get("cpu_spike", 0.0)),
            float(host_telemetry.get("cvss_max", 0.0) / 10.0)
        ]

        return {
            "risk_t": risk_t,
            "phi_t": phi_t,
            "min_distance": min_path_distance
        }

