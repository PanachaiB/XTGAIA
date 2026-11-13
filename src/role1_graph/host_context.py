"""
Role 1 host context (Phi_t): the 6 telemetry flags of the shared state vector,
derived from the asset inventory plus a sliding window over observed traffic.

    Phi_t = [is_admin, is_crown_jewel, conn_count, auth_fails, cpu_spike, cvss_max]   (raw values)

    is_admin / is_crown_jewel / cvss_max   asset inventory (CMDB / vulnerability scanner)
    conn_count   distinct peers the host exchanged traffic with inside the window
                 (raw count; same notion as the graph degree used by risk_engine)
    auth_fails   explicit failed-authentication events (e.g. Wazuh rule groups) plus,
                 for flow-only telemetry, repeated connection attempts from the same
                 source to an authentication port inside the window (brute-force proxy:
                 every attempt after the first one from a source counts)
    cpu_spike    latest agent-reported CPU load in [0, 1]; 0.0 when no agent reports it
"""

import ipaddress
from collections import defaultdict, deque
from typing import Any, Deque, Dict, Iterable, List, Optional, Set, Tuple

import yaml


class AssetInventory:
    # Eq. 8 privilege tiers: S_priv(u) in {1.0 Root/SYSTEM, 0.6 Local Admin, 0.2 User}
    DEFAULT_PRIVILEGE_TIERS = {"root": 1.0, "local_admin": 0.6, "user": 0.2}

    def __init__(self, crown_jewels: Dict[str, float], hosts: Dict[str, Dict[str, Any]],
                 defaults: Optional[Dict[str, Any]] = None, internal_subnets: Iterable[str] = (),
                 privilege_tiers: Optional[Dict[str, float]] = None):
        self.crown_jewels = {str(k): float(v) for k, v in crown_jewels.items()}
        self.hosts = {str(k): dict(v or {}) for k, v in hosts.items()}
        self.defaults = {"is_admin": 0, "cvss_max": 0.0, "privilege": "user", **(defaults or {})}
        self.privilege_tiers = {k: float(v) for k, v in (privilege_tiers or self.DEFAULT_PRIVILEGE_TIERS).items()}
        self.subnets = [ipaddress.ip_network(s) for s in internal_subnets]

    @classmethod
    def from_yaml(cls, path: str) -> "AssetInventory":
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cls(cfg.get("crown_jewels", {}), cfg.get("hosts", {}), cfg.get("defaults"),
                   (cfg.get("network") or {}).get("internal_subnets", []), cfg.get("privilege_tiers"))

    def attr(self, host: str, key: str) -> Any:
        return self.hosts.get(host, {}).get(key, self.defaults.get(key))

    def is_admin(self, host: str) -> bool:
        return bool(self.attr(host, "is_admin"))

    def privilege(self, host: str) -> float:
        """S_priv of a host acting as the source of a hop (Eq. 8)."""
        tier = str(self.attr(host, "privilege") or "user")
        return self.privilege_tiers.get(tier, self.privilege_tiers.get("user", 0.2))

    def is_crown_jewel(self, host: str) -> bool:
        return host in self.crown_jewels

    def cvss(self, host: str) -> float:
        return float(min(max(float(self.attr(host, "cvss_max") or 0.0), 0.0), 10.0))

    def is_internal(self, host: str) -> bool:
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return False
        return any(ip in net for net in self.subnets)


class HostContextTracker:
    def __init__(self, inventory: AssetInventory, window_s: float = 300.0, auth_ports: Iterable[int] = ()):
        self.inventory = inventory
        self.window_s = float(window_s)
        self.auth_ports: Set[int] = {int(p) for p in auth_ports}
        self._conns: Dict[str, Deque[Tuple[float, str]]] = defaultdict(deque)            # host -> (ts, peer)
        self._auth_attempts: Dict[str, Deque[Tuple[float, str]]] = defaultdict(deque)    # host -> (ts, src)
        self._auth_failures: Dict[str, Deque[float]] = defaultdict(deque)                # host -> ts
        self._cpu: Dict[str, float] = {}
        self.now = 0.0

    def _expire(self, q: deque, now: float) -> None:
        cutoff = now - self.window_s
        while q and (q[0][0] if isinstance(q[0], tuple) else q[0]) < cutoff:
            q.popleft()

    def observe(self, edge: Dict[str, Any], raw: Optional[Dict[str, Any]] = None) -> None:
        src, dst, port, ts = edge["source"], edge["target"], int(edge.get("port") or 0), float(edge["timestamp"])
        self.now = max(self.now, ts)
        self._conns[src].append((ts, dst))
        self._conns[dst].append((ts, src))
        if port in self.auth_ports:
            self._auth_attempts[dst].append((ts, src))
        raw = raw or {}
        if raw.get("auth_failure"):
            self._auth_failures[dst].append(ts)
        if raw.get("cpu_spike") is not None:
            self._cpu[dst] = float(raw["cpu_spike"])

    def conn_count(self, host: str) -> int:
        q = self._conns.get(host)
        if not q:
            return 0
        self._expire(q, self.now)
        return len({peer for _, peer in q})

    def auth_fails(self, host: str) -> int:
        attempts = self._auth_attempts.get(host)
        repeats = 0
        if attempts:
            self._expire(attempts, self.now)
            per_src: Dict[str, int] = defaultdict(int)
            for _, src in attempts:
                per_src[src] += 1
            repeats = sum(n - 1 for n in per_src.values())
        failures = self._auth_failures.get(host)
        explicit = 0
        if failures:
            self._expire(failures, self.now)
            explicit = len(failures)
        return repeats + explicit

    def phi(self, host: str) -> List[float]:
        inv = self.inventory
        return [float(inv.is_admin(host)), float(inv.is_crown_jewel(host)), float(self.conn_count(host)),
                float(self.auth_fails(host)), float(self._cpu.get(host, 0.0)), inv.cvss(host)]
