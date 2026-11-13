import time
import networkx as nx
from typing import Dict, Any, Tuple, Optional

class DualTimescaleSKG:
    def __init__(self, ttl_ms: float = 50.0, neo4j_driver: Optional[Any] = None):
        """
        In-Memory RAM Graph ด้วย NetworkX และ TTL 50 ms
        """
        self.graph = nx.DiGraph()
        self.ttl_seconds = ttl_ms / 1000.0  # 50 ms = 0.05 วินาที
        self.driver = neo4j_driver
        self.latest_timestamp = 0.0

    def add_edge(self, source: str, target: str, port: int, timestamp: float, **attributes):
        """
        เพิ่ม Edge เข้ากราฟ พร้อมลบขอบที่หมดอายุ (Sliding Window Eviction)
        """
        self.latest_timestamp = max(self.latest_timestamp, timestamp)
        
        # เพิ่มหรืออัปเดต Edge
        self.graph.add_edge(
            source, 
            target, 
            port=port, 
            timestamp=timestamp, 
            **attributes
        )
        
        # ตัด Edge ที่อายุเกิน TTL 50 ms
        self.evict_stale_edges(self.latest_timestamp)

    def evict_stale_edges(self, current_time: float):
        """
        Algorithm 1: ลบ Edge ทั้งหมดที่มี timestamp < (current_time - 50ms)
        """
        cutoff_time = current_time - self.ttl_seconds
        stale_edges = [
            (u, v) for u, v, data in self.graph.edges(data=True)
            if data.get("timestamp", 0.0) < cutoff_time
        ]
        if stale_edges:
            self.graph.remove_edges_from(stale_edges)

    def sync_to_neo4j(self, u: str, v: str, port: int, timestamp: float):
        """
        อัปเดตสถานะความสัมพันธ์ลงใน Neo4j Persistent Graph
        """
        if not self.driver:
            return
        query = (
            "MERGE (a:Host {ip: $src}) "
            "MERGE (b:Host {ip: $dst}) "
            "MERGE (a)-[r:COMMUNICATES_WITH {port: $port}]->(b) "
            "SET r.last_seen = $timestamp"
        )
        try:
            with self.driver.session() as session:
                session.run(query, src=u, dst=v, port=port, timestamp=timestamp)
        except Exception:
            pass

    def get_active_graph(self) -> nx.DiGraph:
        return self.graph