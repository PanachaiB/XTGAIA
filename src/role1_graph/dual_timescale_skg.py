import ipaddress
import time
import networkx as nx
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional

# Paper SKG schema (Sec. III-A): node and edge types allowed in Cypher labels / relationship types
NODE_LABELS = ("Host", "User", "IPAddress", "Process", "File")
EDGE_LABELS = ("COMMUNICATED", "AUTHENTICATED", "WROTE_FILE", "CREATE_CHILD_PROCESS", "RUN")
NODE_KEY = {"Host": "ip"}            # Host nodes keep their historical `ip` key; other labels use `id`


def _is_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


class DualTimescaleSKG:
    def __init__(self, ttl_ms: float = 50.0, neo4j_driver: Optional[Any] = None):
        """
        In-Memory RAM Graph ด้วย NetworkX และ TTL 50 ms
        """
        self.graph = nx.DiGraph()
        self.ttl_seconds = ttl_ms / 1000.0  # 50 ms = 0.05 วินาที
        self.driver = neo4j_driver
        self.latest_timestamp = 0.0
        # Algorithm 1: nodes that left V_act (no active edge left); drained by the caller so their
        # latent state can be written to Neo4j and re-hydrated when they come back.
        self.evicted_nodes: List[str] = []

    def add_edge(self, source: str, target: str, port: int, timestamp: float, etype: str = "COMMUNICATED",
                 src_type: str = "Host", dst_type: str = "Host", **attributes):
        """
        เพิ่ม Edge เข้ากราฟ พร้อมลบขอบที่หมดอายุ (Sliding Window Eviction)
        Typed SKG edge (paper Sec. III-A): etype in EDGE_LABELS, node types in NODE_LABELS.
        """
        previous_latest = self.latest_timestamp
        self.latest_timestamp = max(self.latest_timestamp, timestamp)

        # V_act <- V_act ∪ {src, dst} (Algorithm 1, line 5)
        for node, ntype in ((source, src_type), (target, dst_type)):
            if node not in self.graph:
                self.graph.add_node(node, ntype=ntype, state="Active")
            elif "ntype" not in self.graph.nodes[node]:
                self.graph.nodes[node]["ntype"] = ntype

        # เพิ่มหรืออัปเดต Edge
        self.graph.add_edge(
            source,
            target,
            port=port,
            timestamp=timestamp,
            etype=etype,
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
        Nodes left without any active edge leave V_act (contained hosts keep their Isolated
        state node so the containment stays visible).
        """
        cutoff_time = current_time - self.ttl_seconds
        stale_edges = [
            (u, v) for u, v, data in self.graph.edges(data=True)
            if data.get("timestamp", 0.0) < cutoff_time
        ]
        if stale_edges:
            self.graph.remove_edges_from(stale_edges)
        # Nodes without any active edge leave V_act - whether their edges expired or were severed
        # by a containment playbook - except contained hosts, which keep their Isolated state.
        gone = [n for n, deg in self.graph.degree() if deg == 0 and self.graph.nodes[n].get("state") != "Isolated"]
        if gone:
            self.graph.remove_nodes_from(gone)
            self.evicted_nodes.extend(gone)

    def pop_evicted(self) -> List[str]:
        out, self.evicted_nodes = self.evicted_nodes, []
        return out

    def sync_to_neo4j(self, u: str, v: str, port: int, timestamp: float):
        """
        อัปเดตสถานะความสัมพันธ์ลงใน Neo4j Persistent Graph
        """
        if not self.driver:
            return
        try:
            self.sync_batch_to_neo4j([{"src": u, "dst": v, "port": port, "timestamp": timestamp}])
        except Exception:
            pass

    def ensure_neo4j_schema(self):
        """Indexes on the node keys so MERGE is an index lookup instead of a label scan."""
        if not self.driver:
            return
        with self.driver.session() as session:
            session.run("CREATE INDEX host_ip IF NOT EXISTS FOR (h:Host) ON (h.ip)").consume()
            for label in ("User", "Process", "File", "IPAddress"):
                session.run(f"CREATE INDEX {label.lower()}_id IF NOT EXISTS FOR (n:{label}) ON (n.id)").consume()

    @staticmethod
    def _merge_node(var: str, label: str, param: str) -> str:
        key = NODE_KEY.get(label, "id")
        return f"MERGE ({var}:{label} {{{key}: {param}}}) "

    def sync_batch_to_neo4j(self, edges) -> int:
        """
        Batched, typed persistence of SKG edges (Algorithm 1, line 18). Each row:
        {src, dst, port, timestamp[, etype, src_type, dst_type, weight, stix_id, technique]}.
        Edges default to Host -COMMUNICATED-> Host. Returns the number written.
        """
        if not self.driver or not edges:
            return 0
        groups: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
        for row in edges:
            et = row.get("etype") or "COMMUNICATED"
            st, dt_ = row.get("src_type") or "Host", row.get("dst_type") or "Host"
            if et not in EDGE_LABELS or st not in NODE_LABELS or dt_ not in NODE_LABELS:
                continue
            groups[(st, et, dt_)].append({
                "src": row["src"], "dst": row["dst"], "port": int(row.get("port") or 0),
                "timestamp": row.get("timestamp"), "weight": row.get("weight"), "stix_id": row.get("stix_id"),
                "technique": row.get("technique"),
                "src_ip": _is_ip(str(row["src"])), "dst_ip": _is_ip(str(row["dst"])),
            })
        n = 0
        with self.driver.session() as session:
            for (st, et, dt_), rows in groups.items():
                rel_key = " {port: row.port}" if et == "COMMUNICATED" else ""
                query = ("UNWIND $rows AS row "
                         + self._merge_node("a", st, "row.src") + self._merge_node("b", dt_, "row.dst")
                         + f"MERGE (a)-[r:{et}{rel_key}]->(b) "
                         "SET r.last_seen = row.timestamp, r.active = true, r.weight = row.weight, "
                         "r.stix_id = row.stix_id, r.technique = row.technique ")
                # Hosts identified by an IP address also carry the paper's IPAddress label
                if st == "Host":
                    query += "FOREACH (_ IN CASE WHEN row.src_ip THEN [1] ELSE [] END | SET a:IPAddress) "
                if dt_ == "Host":
                    query += "FOREACH (_ IN CASE WHEN row.dst_ip THEN [1] ELSE [] END | SET b:IPAddress) "
                session.run(query, rows=rows).consume()
                n += len(rows)
        return n

    def set_edges_active_in_neo4j(self, host: str, edges, active: bool, isolated=None, timestamp=None) -> int:
        """
        Algorithm 4 soft delete / reactivation: flag the given edges inactive (containment) or
        active again (analyst rollback) instead of deleting them, so the node and its history
        stay available for forensic provenance. `isolated` (True/False) also sets Host.state.
        """
        if not self.driver:
            return 0
        groups: Dict[str, List[dict]] = defaultdict(list)
        for u, v, d in edges:
            et = d.get("etype") or "COMMUNICATED"
            if et in EDGE_LABELS:
                groups[et].append({"src": u, "dst": v, "port": int(d.get("port", 0) or 0)})
        with self.driver.session() as session:
            for et, rows in groups.items():
                port_match = " {port: row.port}" if et == "COMMUNICATED" else ""
                query = ("UNWIND $rows AS row "
                         "MATCH (a)-[r:" + et + port_match + "]->(b) "
                         "WHERE coalesce(a.ip, a.id) = row.src AND coalesce(b.ip, b.id) = row.dst "
                         "SET r.active = $active, r.status_changed = $ts")
                session.run(query, rows=rows, active=active, ts=timestamp).consume()
            if isolated is not None:
                session.run("MERGE (h:Host {ip: $ip}) SET h.state = $state, h.state_changed = $ts",
                            ip=host, state="Isolated" if isolated else "Active", ts=timestamp).consume()
        return sum(len(r) for r in groups.values())

    def fetch_latent_states(self, ips):
        """Algorithm 1, line 8: Neo4j.Find(v.ip) -> {ip: (latent_state bytes, latent_ts)}."""
        if not self.driver or not ips:
            return {}
        query = ("UNWIND $ips AS ip MATCH (h:Host {ip: ip}) WHERE h.latent_state IS NOT NULL "
                 "RETURN h.ip AS ip, h.latent_state AS state, h.latent_ts AS ts")
        with self.driver.session() as session:
            return {r["ip"]: (r["state"], r["ts"]) for r in session.run(query, ips=list(ips))}

    def store_latent_states(self, rows) -> int:
        """Persist TGN latent memory (512-byte float32 array per node) for later recovery."""
        if not self.driver or not rows:
            return 0
        query = ("UNWIND $rows AS row MERGE (h:Host {ip: row.ip}) "
                 "SET h.latent_state = row.state, h.latent_ts = row.ts")
        with self.driver.session() as session:
            session.run(query, rows=list(rows)).consume()
        return len(rows)

    def get_active_graph(self) -> nx.DiGraph:
        return self.graph
