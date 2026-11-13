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
        previous_latest = self.latest_timestamp
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
        # Same result as evicting on every insert, but the O(E) scan only runs when the
        # window cutoff moves or the new edge is already outside the window; otherwise
        # nothing can have expired (keeps high-rate streams linear).
        if self.latest_timestamp > previous_latest or timestamp < self.latest_timestamp - self.ttl_seconds:
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

    def ensure_neo4j_schema(self):
        """Index on :Host(ip) so MERGE is an index lookup instead of a label scan."""
        if not self.driver:
            return
        with self.driver.session() as session:
            session.run("CREATE INDEX host_ip IF NOT EXISTS FOR (h:Host) ON (h.ip)").consume()

    def sync_batch_to_neo4j(self, edges) -> int:
        """
        Batched version of sync_to_neo4j (same MERGE semantics) for high-rate streams:
        edges = [{"src", "dst", "port", "timestamp"}, ...]. Returns the number written.
        """
        if not self.driver or not edges:
            return 0
        query = (
            "UNWIND $rows AS row "
            "MERGE (a:Host {ip: row.src}) "
            "MERGE (b:Host {ip: row.dst}) "
            "MERGE (a)-[r:COMMUNICATES_WITH {port: row.port}]->(b) "
            "SET r.last_seen = row.timestamp"
        )
        with self.driver.session() as session:
            session.run(query, rows=list(edges)).consume()
        return len(edges)

    def get_active_graph(self) -> nx.DiGraph:
        return self.graph