"""
SHARED SCHEMA: STIX 2.1 Cyber Observable and Active Graph Edge Contract
Shared between Person 1 (Graph Ingestion) and Person 2 (GNN / XAI)
"""
from typing import Dict, Any, List

def format_active_edge(src_ip: str, dst_ip: str, dst_port: int, timestamp: float, features: List[float]) -> Dict[str, Any]:
    return {
        "source": src_ip,
        "target": dst_ip,
        "port": dst_port,
        "timestamp": timestamp,
        "features": features  # e.g., byte count, packet duration, protocol flags
    }
