"""
Person 1 integration boundary and in-RAM topology mutation.

Role 3 talks to Person 1 only through
    get_active_graph_and_risk(timestamp) -> (nx.DiGraph, Dict[str, float])
which must return the *live* graph and risk dict held in RAM. Nothing else
from src.role1_graph is imported.

QuarantineManager severs/restores edges on that live graph in place, records
exactly what was removed (with edge attributes) so a TheHive rollback can put
it back, and forces the host's risk entry to 0.0 on quarantine.
"""

import logging
import threading
from typing import Callable, Dict, List, Optional, Tuple

import networkx as nx

logger = logging.getLogger("XTGAIA-Graph-Interface")

GraphRiskProvider = Callable[[float], Tuple[nx.DiGraph, Dict[str, float]]]
Edge = Tuple[str, str, dict]


def default_graph_provider() -> GraphRiskProvider:
    """Resolve Person 1's public function lazily (only the agreed interface is imported)."""
    try:
        from src.role1_graph import get_active_graph_and_risk  # noqa: WPS433 (agreed public API)
    except ImportError as exc:
        raise RuntimeError(
            "Person 1 has not exposed get_active_graph_and_risk(timestamp) in src.role1_graph yet; "
            "pass graph_provider=... explicitly."
        ) from exc
    return get_active_graph_and_risk


class QuarantineManager:
    """Tracks edges severed by Role 3 so they can be restored on analyst rollback."""

    def __init__(self):
        self._severed: Dict[str, List[Edge]] = {}
        # Persistence hooks (Algorithm 4: soft-delete in Neo4j on containment, reactivate on
        # rollback). Listeners implement on_sever(host, edges, isolated) / on_restore(host, edges).
        self.listeners: List[object] = []
        self._prior_risk: Dict[str, float] = {}   # risk before a4 zeroed it, for exact rollback
        self._lock = threading.Lock()

    @staticmethod
    def _incident_edges(graph: nx.Graph, host: str) -> List[Edge]:
        if host not in graph:
            raise KeyError(f"Host {host!r} not present in the active graph")
        if graph.is_directed():
            edges = list(graph.in_edges(host, data=True)) + list(graph.out_edges(host, data=True))
        else:
            edges = list(graph.edges(host, data=True))
        return [(u, v, dict(d)) for u, v, d in edges]

    def quarantine(self, graph: nx.DiGraph, risk: Dict[str, float], host: str) -> List[Edge]:
        """a4: remove every inbound and outbound edge of `host` in place and reset its risk to 0.0."""
        return self._sever(graph, risk, host, self._incident_edges(graph, host), reset_risk=True)

    def isolate_port(self, graph: nx.DiGraph, risk: Dict[str, float], host: str, port: int) -> List[Edge]:
        """a2: remove only the host's edges whose `port` attribute matches."""
        edges = [e for e in self._incident_edges(graph, host) if e[2].get("port") == port]
        return self._sever(graph, risk, host, edges, reset_risk=False)

    def _sever(self, graph, risk, host, edges, reset_risk) -> List[Edge]:
        with self._lock:
            graph.remove_edges_from([(u, v) for u, v, _ in edges])
            self._severed.setdefault(host, []).extend(edges)
            if reset_risk and host in graph:          # Phase 7: the quarantined node stays, state -> Isolated
                graph.nodes[host]["state"] = "Isolated"
            if reset_risk:
                if host in risk:
                    self._prior_risk.setdefault(host, float(risk[host]))  # keep the first, pre-containment value
                risk[host] = 0.0
        logger.info("[Topology] Severed %d edges for %s (risk reset=%s)", len(edges), host, reset_risk)
        self._notify("on_sever", host, edges, reset_risk)
        return edges

    def add_listener(self, listener) -> None:
        if listener not in self.listeners:
            self.listeners.append(listener)

    def _notify(self, hook: str, *args) -> None:
        for listener in list(self.listeners):
            fn = getattr(listener, hook, None)
            if fn is None:
                continue
            try:
                fn(*args)
            except Exception as exc:   # persistence must not break containment
                logger.warning("Quarantine listener %s.%s failed: %s", type(listener).__name__, hook, exc)

    def restore(self, graph: nx.DiGraph, host: str) -> List[Edge]:
        """Rollback: re-insert every edge previously severed for `host`, with original attributes."""
        with self._lock:
            edges = self._severed.pop(host, [])
            for u, v, d in edges:
                graph.add_edge(u, v, **d)
            if host in graph:
                graph.nodes[host]["state"] = "Active"
        logger.info("[Topology] Restored %d edges for %s", len(edges), host)
        self._notify("on_restore", host, edges)
        return edges

    def pop_prior_risk(self, host: str) -> Optional[float]:
        """Risk value the host had before quarantine reset it (None if never reset)."""
        with self._lock:
            return self._prior_risk.pop(host, None)

    def severed_edges(self, host: str) -> List[Edge]:
        return list(self._severed.get(host, []))

    def is_isolated(self, host: str) -> bool:
        return bool(self._severed.get(host))


_shared_manager: Optional[QuarantineManager] = None


def get_quarantine_manager() -> QuarantineManager:
    """Process-wide manager shared by the dispatcher and the rollback API."""
    global _shared_manager
    if _shared_manager is None:
        _shared_manager = QuarantineManager()
    return _shared_manager
