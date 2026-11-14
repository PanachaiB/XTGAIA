"""Stage 1 (paper Table I): Mordor telemetry ingested *via Wazuh*.

    Mordor JSON  --append-->  data/wazuh/ingest/mordor.json   (bind mount, Wazuh <localfile> json)
                               Wazuh manager: logcollector -> JSON decoder -> analysisd (rules, MITRE)
    SKG          <--tail---   data/wazuh/archives/archives.json  (logall_json: every decoded event)

The Wazuh event carries the original record in `full_log` (decoded fields in `data`) plus the manager's
rule match (`rule.id/level/mitre`), which is attached to the record as `wazuh_*` side-channel keys
before Role 1 normalises it into typed SKG edges (HostTelemetryNormalizer).
"""
import json
import os
import threading
import time
from typing import Any, Dict, Iterable, Iterator, Optional

from src.role1_graph.telemetry import MordorSource

INGEST_FILE = "data/wazuh/ingest/mordor.json"
ARCHIVES_FILE = "data/wazuh/archives/archives.json"
MANAGER_LOCATION = "/var/ossec/logs/mordor/mordor.json"      # <location> in docker/wazuh/master/ossec.conf


def replay_into_wazuh(paths: Iterable[str], ingest_file: str = INGEST_FILE, eps: float = 0.0) -> int:
    """Append Mordor events (time-ordered) to the file the Wazuh log collector monitors."""
    os.makedirs(os.path.dirname(ingest_file) or ".", exist_ok=True)
    n = 0
    with open(ingest_file, "a", encoding="utf-8") as f:
        for ev in MordorSource(list(paths)):
            f.write(json.dumps(ev, separators=(",", ":")) + "\n")
            n += 1
            if n % 500 == 0:
                f.flush()
            if eps > 0:
                time.sleep(1.0 / eps)
    return n


def from_wazuh_event(alert: Dict[str, Any], location: Optional[str] = MANAGER_LOCATION) -> Optional[Dict[str, Any]]:
    """Wazuh archive/alert entry -> original telemetry record (+ wazuh_* rule metadata), or None."""
    if location and alert.get("location") != location:
        return None
    rec = None
    full = alert.get("full_log")
    if isinstance(full, str) and full.lstrip().startswith("{"):
        try:
            rec = json.loads(full)
        except json.JSONDecodeError:
            rec = None
    if rec is None and isinstance(alert.get("data"), dict):
        rec = dict(alert["data"])
    if not isinstance(rec, dict):
        return None
    rule = alert.get("rule") or {}
    rec["wazuh_rule_id"] = rule.get("id")
    rec["wazuh_rule_level"] = rule.get("level")
    rec["wazuh_mitre"] = (rule.get("mitre") or {}).get("id")
    rec["wazuh_manager"] = (alert.get("manager") or {}).get("name")
    return rec


class WazuhArchiveSource:
    """Tails archives.json from `offset` and yields Mordor records decoded by the manager."""

    def __init__(self, path: str = ARCHIVES_FILE, offset: int = 0, follow: bool = True, idle_timeout: float = 30.0,
                 expected: Optional[int] = None, poll: float = 0.5, location: Optional[str] = MANAGER_LOCATION):
        self.path, self.offset, self.follow = path, offset, follow
        self.idle_timeout, self.expected, self.poll, self.location = idle_timeout, expected, poll, location
        self.seen = 0

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        deadline = time.time() + self.idle_timeout
        while not os.path.exists(self.path):
            if time.time() > deadline:
                return
            time.sleep(self.poll)
        with open(self.path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(self.offset)
            buf = ""
            while True:
                chunk = f.readline()
                if chunk:
                    buf += chunk
                    if not buf.endswith("\n"):            # partial line still being written
                        continue
                    line, buf = buf, ""
                    try:
                        rec = from_wazuh_event(json.loads(line), self.location)
                    except json.JSONDecodeError:
                        continue
                    if rec is not None:
                        self.seen += 1
                        deadline = time.time() + self.idle_timeout
                        yield rec
                        if self.expected is not None and self.seen >= self.expected:
                            return
                    continue
                if not self.follow or time.time() > deadline:
                    return
                time.sleep(self.poll)


class WazuhMordorSource:
    """`main.py run --source mordor --via-wazuh`: replays the Mordor capture into the Wazuh manager and
    streams back what the manager decoded (until every event came back or the stream stays idle)."""

    def __init__(self, paths: Iterable[str], ingest_file: str = INGEST_FILE, archives_file: str = ARCHIVES_FILE,
                 idle_timeout: float = 60.0, eps: float = 0.0):
        self.paths, self.ingest_file, self.archives_file = list(paths), ingest_file, archives_file
        self.idle_timeout, self.eps = idle_timeout, eps
        self.replayed = 0
        self.received = 0

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        offset = os.path.getsize(self.archives_file) if os.path.exists(self.archives_file) else 0
        expected = sum(1 for _ in MordorSource(self.paths))
        writer = threading.Thread(target=lambda: setattr(self, "replayed", replay_into_wazuh(
            self.paths, self.ingest_file, self.eps)), daemon=True)
        writer.start()
        src = WazuhArchiveSource(self.archives_file, offset=offset, idle_timeout=self.idle_timeout, expected=expected)
        for rec in src:
            self.received = src.seen
            yield rec
        writer.join(timeout=5)
