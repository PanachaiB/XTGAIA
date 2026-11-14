"""
Role 1 threat intelligence (paper Phase 4 / Table I, Stage 4):

  NVDClient       CVSS base scores from the NVD REST API 2.0 for the CVEs listed per host in the
                  asset inventory ("live vulnerability feeds via the NVD REST API furnish asset
                  vulnerability scoring"). Results are cached on disk; offline -> cache/inventory.
  AttackMapper    MITRE ATT&CK STIX 2.x corpus: technique id -> name / tactics, used to annotate
                  SKG edges (T1021 remote-service lateral movement by port, process-creation
                  techniques by image), as in "MITRE ATT&CK STIX 2.1 corpus correlates adversary
                  privileges and technique sequences with active enterprise topologies".

Privilege tiers (S_priv) are MITRE's privilege levels - SYSTEM/root, Administrator, User - and are
observed from telemetry by host_context.PrivilegeTracker (current ATT&CK releases no longer carry
x_mitre_permissions_required on techniques).
"""

import json
import logging
import os
import threading
import time
from typing import Dict, Iterable, List, Optional

import requests

logger = logging.getLogger("XTGAIA-Role1-ThreatIntel")

NVD_URL = "https://services.nvd.nist.gov/rest/json/cves/2.0"

# MITRE ATT&CK T1021 (Remote Services) sub-techniques by service port
PORT_TECHNIQUES = {3389: "T1021.001", 445: "T1021.002", 139: "T1021.002", 135: "T1021.003", 22: "T1021.004",
                   5900: "T1021.005", 5985: "T1021.006", 5986: "T1021.006"}
# Process-creation patterns seen in lateral movement telemetry (Mordor) -> ATT&CK technique
IMAGE_TECHNIQUES = {"psexesvc.exe": "T1569.002", "services.exe": "T1569.002", "wmiprvse.exe": "T1047",
                    "wsmprovhost.exe": "T1021.006", "powershell.exe": "T1059.001", "cmd.exe": "T1059.003",
                    "schtasks.exe": "T1053.005", "rundll32.exe": "T1218.011", "regsvr32.exe": "T1218.010",
                    "mshta.exe": "T1218.005", "wmic.exe": "T1047", "sc.exe": "T1543.003"}


class NVDClient:
    def __init__(self, cache_path: str = "data/cache/nvd_cvss.json", enabled: bool = True, timeout: float = 15.0,
                 min_interval_s: float = 6.5):
        self.cache_path, self.enabled, self.timeout = cache_path, enabled, timeout
        self.min_interval_s = min_interval_s          # public rate limit: 5 requests / 30 s without an API key
        self._last = 0.0
        self._lock = threading.Lock()
        self.cache: Dict[str, Optional[float]] = {}
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except (OSError, ValueError):
                self.cache = {}

    @staticmethod
    def parse_base_score(payload: dict) -> Optional[float]:
        try:
            metrics = payload["vulnerabilities"][0]["cve"]["metrics"]
        except (KeyError, IndexError, TypeError):
            return None
        for key in ("cvssMetricV40", "cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
            for m in metrics.get(key, []) or []:
                score = (m.get("cvssData") or {}).get("baseScore")
                if score is not None:
                    return float(score)
        return None

    def base_score(self, cve: str) -> Optional[float]:
        cve = cve.strip().upper()
        if cve in self.cache:
            return self.cache[cve]
        if not self.enabled:
            return None
        with self._lock:
            wait = self.min_interval_s - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                headers = {"apiKey": os.environ["NVD_API_KEY"]} if os.getenv("NVD_API_KEY") else {}
                res = requests.get(NVD_URL, params={"cveId": cve}, headers=headers, timeout=self.timeout)
                self._last = time.time()
                score = self.parse_base_score(res.json()) if res.status_code == 200 else None
            except (requests.RequestException, ValueError) as exc:
                logger.warning("NVD lookup for %s failed: %s", cve, exc)
                return None
            self.cache[cve] = score
            self._save()
        return score

    def max_score(self, cves: Iterable[str]) -> Optional[float]:
        scores = [s for s in (self.base_score(c) for c in cves) if s is not None]
        return max(scores) if scores else None

    def _save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=1, sort_keys=True)
        except OSError as exc:
            logger.warning("Could not write NVD cache: %s", exc)


class AttackMapper:
    """Technique index from the ATT&CK STIX bundle (enterprise-attack.json)."""

    def __init__(self, path: Optional[str] = "data/raw/attack/enterprise-attack.json"):
        self.techniques: Dict[str, Dict] = {}
        self.source = None
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                bundle = json.load(f)
            for o in bundle.get("objects", []):
                if o.get("type") != "attack-pattern" or o.get("revoked") or o.get("x_mitre_deprecated"):
                    continue
                tid = next((r.get("external_id") for r in o.get("external_references", [])
                            if r.get("source_name") == "mitre-attack"), None)
                if tid:
                    self.techniques[tid] = {"id": tid, "name": o.get("name"),
                                            "tactics": [k["phase_name"] for k in o.get("kill_chain_phases", [])
                                                        if k.get("kill_chain_name") == "mitre-attack"]}
            self.source = os.path.basename(path)
        else:
            logger.warning("ATT&CK STIX bundle not found at %s; technique names unavailable", path)

    def describe(self, tid: Optional[str]) -> Optional[Dict]:
        if not tid:
            return None
        return self.techniques.get(tid, {"id": tid, "name": None, "tactics": []})

    def for_port(self, port: int) -> Optional[Dict]:
        return self.describe(PORT_TECHNIQUES.get(int(port or 0)))

    def for_image(self, image: Optional[str]) -> Optional[Dict]:
        if not image:
            return None
        return self.describe(IMAGE_TECHNIQUES.get(os.path.basename(str(image).replace("\\", "/")).lower()))
