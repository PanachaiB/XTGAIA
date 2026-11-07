import networkx as nx

class DualTimescaleSKG:
    def __init__(self, ttl_ms: float = 50.0):
        self.graph = nx.DiGraph()
        self.ttl = ttl_ms / 1000.0  # แปลงเป็นวินาที

    def add_edge(self, u: str, v: str, timestamp: float, **attrs):
        self.graph.add_edge(u, v, timestamp=timestamp, **attrs)
        self.evict_old_edges(current_time=timestamp)

    def evict_old_edges(self, current_time: float):
        cutoff = current_time - self.ttl
        stale_edges = [
            (u, v) for u, v, data in self.graph.edges(data=True)
            if data.get("timestamp", 0.0) < cutoff
        ]
        self.graph.remove_edges_from(stale_edges)