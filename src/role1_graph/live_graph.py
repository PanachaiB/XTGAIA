"""
Role 1 live graph service: the single in-RAM Security Knowledge Graph shared by
all three roles.

    ingest(raw)                 STIXNormalizer -> DualTimescaleSKG (+ batched Neo4j sync)
                                -> host context; edges carry Algorithm 2 friction weights
    compute_risk(host, p_t)     Algorithm 2: Risk_t(v) = p_t * sum_c crit_c / (d(v, c) + 1)^2
                                with d = Dijkstra distance over exploit-friction weights
    phi(host)                   Phi_t host context (raw values, schema order)
    get_active_graph_and_risk   the Person 1 interface Role 3 mutates in place

Enforcement: once Role 3 contains a host, flows the playbook blocks are dropped
at ingestion (`is_blocked` predicate supplied by the orchestrator), exactly as
a real firewall/EDR action would stop them reaching the sensor.
"""

import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import networkx as nx

from src.role1_graph.dual_timescale_skg import DualTimescaleSKG
from src.role1_graph.graph_engine import ReachabilityRiskEngine as CriticalityRiskEngine
from src.role1_graph.host_context import AssetInventory, HostContextTracker
from src.role1_graph.risk_engine import ReachabilityRiskEngine as FrictionModel
from src.role1_graph.stix_normalizer import STIXNormalizer

logger = logging.getLogger("XTGAIA-Role1-LiveGraph")

BlockPredicate = Callable[[str, str, int], bool]

# MITRE privilege tiers (configs/system_config.yaml risk_engine.privilege_tiers)
PRIV_ADMIN, PRIV_STANDARD = 1.0, 0.2


class LiveGraphService:
    def __init__(self, inventory: AssetInventory, port_weights_path: str, window_ms: float = 50.0,
                 host_window_s: float = 300.0, auth_ports=(), neo4j_driver: Optional[Any] = None,
                 neo4j_batch_size: int = 500, is_blocked: Optional[BlockPredicate] = None):
        self.inventory = inventory
        self.normalizer = STIXNormalizer()
        self.skg = DualTimescaleSKG(ttl_ms=window_ms, neo4j_driver=neo4j_driver)
        self.friction = FrictionModel(set(inventory.crown_jewels), config_path=port_weights_path)
        self.risk_engine = CriticalityRiskEngine(crown_jewels=inventory.crown_jewels)
        self.context = HostContextTracker(inventory, host_window_s, auth_ports)
        self.risk: Dict[str, float] = {}
        self.is_blocked: BlockPredicate = is_blocked or (lambda s, d, p: False)
        self.neo4j_batch_size = int(neo4j_batch_size)
        self._pending: List[Dict[str, Any]] = []
        self.lock = threading.RLock()
        self.stats = {"ingested": 0, "dropped_malformed": 0, "blocked": 0, "neo4j_synced": 0, "neo4j_errors": 0}
        if neo4j_driver is not None:
            try:
                self.skg.ensure_neo4j_schema()
            except Exception as exc:
                logger.warning("Could not create Neo4j Host(ip) index: %s", exc)

    # ------------------------------------------------------------------ ingestion
    def edge_weight(self, dst: str, port: int) -> Tuple[float, float, float]:
        s_vuln = self.inventory.cvss(dst) / 10.0
        s_priv = PRIV_ADMIN if self.inventory.is_admin(dst) else PRIV_STANDARD
        return self.friction.compute_edge_friction(s_vuln, s_priv, port), s_vuln, s_priv

    def ingest(self, raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Normalise one raw event and add it to the live graph. Returns the
        active edge (shared STIX edge contract) or None if dropped."""
        edge = self.normalizer.normalize(raw)
        if edge is None:
            self.stats["dropped_malformed"] += 1
            return None
        src, dst, port, ts = edge["source"], edge["target"], int(edge["port"]), float(edge["timestamp"])
        if self.is_blocked(src, dst, port):
            self.stats["blocked"] += 1
            return None
        weight, s_vuln, s_priv = self.edge_weight(dst, port)
        with self.lock:
            self.skg.add_edge(src, dst, port=port, timestamp=ts, weight=weight, s_vuln=s_vuln, s_priv=s_priv)
            self.context.observe(edge, raw)
        self.stats["ingested"] += 1
        if self.skg.driver is not None:
            self._pending.append({"src": src, "dst": dst, "port": port, "timestamp": ts})
            if len(self._pending) >= self.neo4j_batch_size:
                self.flush_neo4j()
        return edge

    def flush_neo4j(self) -> None:
        if not self._pending or self.skg.driver is None:
            return
        batch, self._pending = self._pending, []
        try:
            self.stats["neo4j_synced"] += self.skg.sync_batch_to_neo4j(batch)
        except Exception as exc:   # persistence must never stop detection
            self.stats["neo4j_errors"] += 1
            logger.warning("Neo4j batch sync failed (%d edges): %s", len(batch), exc)

    # ------------------------------------------------------------------ Algorithm 2
    def compute_risk(self, host: str, p_t: float) -> float:
        with self.lock:
            graph = self.skg.graph
            if host not in graph:
                risk = 0.0
            else:
                risk = self.risk_engine.calculate_node_risk(graph, host, float(p_t))
            self.risk[host] = float(risk)
            return float(risk)

    def phi(self, host: str) -> List[float]:
        with self.lock:
            return self.context.phi(host)

    def nearest_crown_jewel(self, host: str) -> Optional[Tuple[str, float]]:
        """(crown jewel, friction distance) of the closest reachable crown jewel, for explanations."""
        best = None
        with self.lock:
            g = self.skg.graph
            if host not in g:
                return None
            for cj in self.inventory.crown_jewels:
                if cj in g and nx.has_path(g, host, cj):
                    d = nx.dijkstra_path_length(g, host, cj, weight="weight")
                    if best is None or d < best[1]:
                        best = (cj, float(d))
        return best

    # ------------------------------------------------------------------ Person 1 interface
    def get_active_graph_and_risk(self, timestamp: float = 0.0) -> Tuple[nx.DiGraph, Dict[str, float]]:
        return self.skg.graph, self.risk


_service: Optional[LiveGraphService] = None


def set_live_graph_service(service: Optional[LiveGraphService]) -> None:
    global _service
    _service = service


def get_live_graph_service() -> Optional[LiveGraphService]:
    return _service


def get_active_graph_and_risk(timestamp: float = 0.0) -> Tuple[nx.DiGraph, Dict[str, float]]:
    """Person 1 public interface consumed by Role 3 (graph_interface.default_graph_provider):
    returns the *live* in-RAM graph and risk dict, which Role 3 mutates in place."""
    if _service is None:
        raise RuntimeError("No live graph service is running; start the XTGAIA pipeline first.")
    return _service.get_active_graph_and_risk(timestamp)
