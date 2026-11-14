"""
Role 1 ingestion: real telemetry sources -> raw log dicts for STIXNormalizer.

Every source yields dictionaries with the keys STIXNormalizer understands
(src_ip, dst_ip, dst_port, timestamp, proto, features) plus optional
side-channel keys that never enter the model:
    label          ground-truth label (datasets only; used for evaluation)
    auth_failure   True when the event itself reports a failed authentication
    cpu_spike      host CPU load in [0, 1] when an agent reports it

Supported sources
    CICIDSFlowSource   CICIDS2017 "GeneratedLabelledFlows" (TrafficLabelling) CSVs
    WazuhAlertSource   Wazuh manager alerts.json (one JSON alert per line)
    JSONEventSource    JSON / JSONL event dumps (Mordor / Sysmon style keys)

FLOW_FEATURES is the edge-feature contract handed to Role 2: the TGN raw
message is built from exactly these flow statistics, in this order.
"""

import csv
import datetime as dt
import json
import os
import re
import time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

FLOW_FEATURES: List[str] = [
    "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets",
    "Fwd Packet Length Mean", "Bwd Packet Length Mean",
    "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Fwd IAT Mean",
    "SYN Flag Count", "PSH Flag Count", "ACK Flag Count",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward",
]

# Same statistics in the corrected CICIDS2017 / CSE-CIC-IDS2018 release (Engelen et al., regenerated
# CICFlowMeter), whose timestamps have microsecond resolution - which is what makes the paper's
# Delta_c = 50 ms correlation window meaningful on replayed data.
IMPROVED_FEATURE_NAMES: Dict[str, str] = {
    "Total Fwd Packets": "Total Fwd Packet", "Total Backward Packets": "Total Bwd packets",
    "Total Length of Fwd Packets": "Total Length of Fwd Packet",
    "Total Length of Bwd Packets": "Total Length of Bwd Packet",
    "Init_Win_bytes_forward": "FWD Init Win Bytes", "Init_Win_bytes_backward": "Bwd Init Win Bytes",
}

_PROTO = {"6": "tcp", "17": "udp", "1": "icmp", "0": "hopopt"}
_TS_RE = re.compile(r"^\s*(\d{1,2})/(\d{1,2})/(\d{4})\s+(\d{1,2}):(\d{2})(?::(\d{2}))?\s*([AaPp][Mm])?\s*$")


def parse_cicids_timestamp(value: str, workday_start_hour: int = 8) -> Optional[float]:
    """CICIDS2017 timestamps are day-first ("4/7/2017 8:54" = 4 July 2017) with
    minute resolution, and afternoon times often lack AM/PM ("1:30" = 13:30).
    The testbed only ran 08:00-17:00, so an hour before the workday start is
    read as PM. Returns POSIX seconds (UTC-naive interpretation) or None."""
    m = _TS_RE.match(str(value))
    if not m:
        return None
    day, month, year, hour, minute, sec, ampm = m.groups()
    hour, minute, sec = int(hour), int(minute), int(sec or 0)
    if ampm:
        hour = hour % 12 + (12 if ampm.lower() == "pm" else 0)
    elif hour < workday_start_hour:
        hour += 12
    try:
        stamp = dt.datetime(int(year), int(month), int(day), hour, minute, sec, tzinfo=dt.timezone.utc)
    except ValueError:
        return None
    return stamp.timestamp()


def _parse_iso(value: str) -> Optional[float]:
    """'2017-07-07 08:59:50.123456' (corrected release; naive, read as UTC like the original parser)."""
    try:
        return dt.datetime.fromisoformat(str(value).strip()).replace(tzinfo=dt.timezone.utc).timestamp()
    except ValueError:
        return None


def _to_float(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if f != f or f in (float("inf"), float("-inf")):   # NaN / inf (Flow Bytes/s has both)
        return 0.0
    return f


class CICIDSFlowSource:
    """Streams CICIDS2017 TrafficLabelling CSV rows as raw flow events in time order.

    Rows are sorted by (timestamp, file order) because the CSVs are only roughly
    chronological. Use `start`/`end` (POSIX seconds) to replay a time slice."""

    def __init__(self, paths: Sequence[str], start: Optional[float] = None, end: Optional[float] = None,
                 limit: Optional[int] = None):
        self.paths = [paths] if isinstance(paths, str) else list(paths)
        self.start, self.end, self.limit = start, end, limit

    def _rows(self) -> Iterator[Dict[str, Any]]:
        for path in self.paths:
            with open(path, "r", encoding="latin-1", newline="") as f:
                reader = csv.reader(f)
                header = [h.strip() for h in next(reader)]
                idx = {h: i for i, h in enumerate(header)}
                improved = "Src IP" in idx                     # corrected release (Engelen et al.)
                if improved:
                    for orig, new in (("Source IP", "Src IP"), ("Destination IP", "Dst IP"),
                                      ("Destination Port", "Dst Port"), *IMPROVED_FEATURE_NAMES.items()):
                        idx[orig] = idx.get(new, -1)
                    idx = {k: v for k, v in idx.items() if v >= 0}
                missing = [c for c in ["Source IP", "Destination IP", "Destination Port", "Timestamp", "Label"]
                           + FLOW_FEATURES if c not in idx]
                if missing:
                    raise ValueError(f"{os.path.basename(path)}: missing CICIDS columns {missing}")
                for order, row in enumerate(reader):
                    if len(row) < len(header) or not row[idx["Source IP"]].strip():
                        continue
                    ts = _parse_iso(row[idx["Timestamp"]]) if improved else parse_cicids_timestamp(row[idx["Timestamp"]])
                    if ts is None or (self.start is not None and ts < self.start) or \
                            (self.end is not None and ts >= self.end):
                        continue
                    yield {
                        "src_ip": row[idx["Source IP"]].strip(),
                        "dst_ip": row[idx["Destination IP"]].strip(),
                        "dst_port": int(_to_float(row[idx["Destination Port"]])),
                        "timestamp": ts,
                        "proto": _PROTO.get(row[idx["Protocol"]].strip(), "tcp") if "Protocol" in idx else "tcp",
                        "features": [_to_float(row[idx[c]]) for c in FLOW_FEATURES],
                        "label": ("BENIGN" if improved and "Attempted Category" in idx
                                  and row[idx["Attempted Category"]].strip() not in ("", "-1")
                                  else row[idx["Label"]].strip()),
                        "_order": order,
                    }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        rows = sorted(self._rows(), key=lambda r: (r["timestamp"], r["_order"]))
        for n, r in enumerate(rows):
            if self.limit is not None and n >= self.limit:
                break
            r.pop("_order", None)
            yield r


def _dig(d: Dict[str, Any], *paths: str) -> Any:
    for path in paths:
        cur: Any = d
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                cur = None
                break
            cur = cur[part]
        if cur not in (None, ""):
            return cur
    return None


# Wazuh rule groups that report a failed authentication
_WAZUH_AUTH_FAIL_GROUPS = {"authentication_failed", "authentication_failures", "win_authentication_failed",
                           "invalid_login", "sshd_failed"}


class WazuhAlertSource:
    """Reads Wazuh alerts.json (one alert per line). With follow=True it tails the
    file like `tail -f`, which is how the manager's live alert stream is consumed
    (docker volume wazuh_master_alerts -> /var/ossec/logs/alerts/alerts.json)."""

    def __init__(self, path: str, follow: bool = False, poll_interval: float = 1.0):
        self.path, self.follow, self.poll_interval = path, follow, poll_interval

    @staticmethod
    def to_raw(alert: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        src = _dig(alert, "data.srcip", "data.src_ip", "data.win.eventdata.sourceIp", "data.win.eventdata.ipAddress")
        dst = _dig(alert, "data.dstip", "data.dst_ip", "data.win.eventdata.destinationIp", "agent.ip")
        if not src or not dst:
            return None
        port = _dig(alert, "data.dstport", "data.dst_port", "data.win.eventdata.destinationPort") or 0
        groups = set(_dig(alert, "rule.groups") or [])
        raw = {
            "src_ip": str(src), "dst_ip": str(dst), "dst_port": int(_to_float(port)),
            "timestamp": _dig(alert, "timestamp", "@timestamp"),
            "proto": str(_dig(alert, "data.protocol", "data.win.eventdata.protocol") or "tcp").lower(),
            "auth_failure": bool(groups & _WAZUH_AUTH_FAIL_GROUPS),
            "rule_id": _dig(alert, "rule.id"), "rule_level": _dig(alert, "rule.level"),
            "mitre": _dig(alert, "rule.mitre.id"),
        }
        cpu = _dig(alert, "data.cpu_usage", "data.cpu")
        if cpu is not None:
            raw["cpu_spike"] = min(max(_to_float(cpu) / (100.0 if _to_float(cpu) > 1 else 1.0), 0.0), 1.0)
        return raw

    def __iter__(self) -> Iterator[Optional[Dict[str, Any]]]:
        with open(self.path, "r", encoding="utf-8", errors="replace") as f:
            while True:
                line = f.readline()
                if not line:
                    if not self.follow:
                        return
                    yield None                    # idle tick: lets the pipeline flush a partial batch
                    time.sleep(self.poll_interval)
                    continue
                try:
                    raw = self.to_raw(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if raw is not None:
                    yield raw


class JSONEventSource:
    """JSON array or JSON-lines dump of events with Mordor / Sysmon style keys
    (SourceIp, DestinationIp, DestinationPort, @timestamp ...). Records are
    passed through unchanged; STIXNormalizer resolves the key variants."""

    def __init__(self, path: str):
        self.path = path

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        with open(self.path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        records: Iterable[Dict[str, Any]]
        records = json.loads(text) if text.startswith("[") else (json.loads(l) for l in text.splitlines() if l.strip())
        for rec in records:
            if isinstance(rec, dict):
                yield rec
