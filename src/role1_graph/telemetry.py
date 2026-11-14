"""
Role 1 heterogeneous telemetry (paper Sec. III-A, Security Event Sources + SKG structure).

The SKG has the paper's node types Host / User / IPAddress / Process / File and edge types
COMMUNICATED, AUTHENTICATED, WROTE_FILE, CREATE_CHILD_PROCESS and RUN. Network flows keep using
STIXNormalizer (Host -COMMUNICATED-> Host); this module maps the other Security Event Sources:

  Windows / Sysmon (Mordor, Wazuh-collected endpoints, Active Directory authentication records)
    Sysmon 1   process creation  -> User -RUN-> Process, Process -CREATE_CHILD_PROCESS-> Process
    Sysmon 3   network connection -> Host -COMMUNICATED-> Host           (no flow statistics)
    Sysmon 11  file creation     -> Process -WROTE_FILE-> File
    4624       successful logon  -> User -AUTHENTICATED-> Host
    4625       failed logon      -> host context (auth_fails)
    4672       special privileges -> privilege observation (S_priv tier: SYSTEM / Administrator)
  AWS CloudTrail
    ConsoleLogin                 -> User -AUTHENTICATED-> aws-console (failures -> auth_fails)
    other API calls              -> Host(sourceIPAddress) -COMMUNICATED-> aws service endpoint
  Zeek conn.log                  -> network flow record (STIXNormalizer), CICFlowMeter-style features

Every typed edge carries a STIX 2.1 bundle (ipv4-addr, network-traffic, user-account, process,
file SCOs) and, where it applies, the MITRE ATT&CK technique (AttackMapper).
"""

import datetime as dt
import glob
import hashlib
import io
import json
import os
import zipfile
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from src.role1_graph.flow_sources import FLOW_FEATURES
from src.role1_graph.threat_intel import AttackMapper

NODE_TYPES = ("Host", "User", "IPAddress", "Process", "File")
EDGE_TYPES = ("COMMUNICATED", "AUTHENTICATED", "WROTE_FILE", "CREATE_CHILD_PROCESS", "RUN")
SYSTEM_SIDS = {"S-1-5-18"}                       # LocalSystem
SYSTEM_NAMES = {"system", "nt authority\\system", "local system"}


def parse_time(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace("Z", "+00:00")
    for fmt in (None, "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            t = dt.datetime.fromisoformat(s) if fmt is None else dt.datetime.strptime(s, fmt)
            if t.tzinfo is None:
                t = t.replace(tzinfo=dt.timezone.utc)
            return t.timestamp()
        except ValueError:
            continue
    return None


def _sid(*parts: Any) -> str:
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def _user_id(domain: Any, name: Any) -> Optional[str]:
    name = (str(name or "")).strip()
    if not name or name in {"-", "ANONYMOUS LOGON"}:
        return None
    if "\\" in name:
        return f"user:{name.lower()}"
    domain = (str(domain or "")).strip()
    return f"user:{(domain + chr(92) + name).lower()}" if domain and domain != "-" else f"user:{name.lower()}"


def _host_id(name: Any) -> Optional[str]:
    n = str(name or "").strip().lower()
    return n or None


class HostTelemetryNormalizer:
    """Maps Windows/Sysmon and CloudTrail records to typed SKG edges + host-context observations."""

    def __init__(self, attack: Optional[AttackMapper] = None):
        self.attack = attack
        self.ip_to_host: Dict[str, str] = {}          # learned from Sysmon 3 (which side is the reporting host)

    # ---------------------------------------------------------------- helpers
    def resolve(self, ip: Any) -> Optional[str]:
        ip = str(ip or "").strip()
        if not ip or ip in {"-", "::1", "127.0.0.1"}:
            return None
        return self.ip_to_host.get(ip, ip)

    @staticmethod
    def _edge(src, dst, etype, src_type, dst_type, ts, objects, port=0, attrs=None, technique=None):
        return {"source": src, "target": dst, "etype": etype, "src_type": src_type, "dst_type": dst_type,
                "port": int(port or 0), "timestamp": float(ts), "features": None,
                "stix_bundle": {"type": "bundle", "id": f"bundle--{_sid(src, dst, etype, ts)}", "objects": objects},
                "attrs": attrs or {}, "technique": technique}

    @staticmethod
    def kind(record: Dict[str, Any]) -> str:
        if "EventID" in record and ("Channel" in record or "SourceName" in record):
            return "windows"
        if "eventSource" in record and "eventName" in record:
            return "cloudtrail"
        return "flow"

    def normalize(self, record: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """-> (typed edges, context observations {auth_failure_host, privilege: (node, tier)})."""
        k = self.kind(record)
        if k == "windows":
            return self._windows(record)
        if k == "cloudtrail":
            return self._cloudtrail(record)
        return [], {}

    # ---------------------------------------------------------------- Windows / Sysmon
    def _windows(self, ev: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        eid = int(ev.get("EventID") or 0)
        channel = str(ev.get("Channel") or ev.get("SourceName") or "").lower()
        ts = parse_time(ev.get("UtcTime")) or parse_time(ev.get("@timestamp")) or parse_time(ev.get("EventTime"))
        host = _host_id(ev.get("Hostname") or ev.get("host"))
        if ts is None or host is None:
            return [], {}
        sysmon = "sysmon" in channel
        if sysmon and eid == 1:
            child = f"proc:{host}:{ev.get('ProcessGuid')}"
            parent = f"proc:{host}:{ev.get('ParentProcessGuid')}"
            user = _user_id(None, ev.get("User"))
            tech = self.attack.for_image(ev.get("Image")) if self.attack else None
            proc = {"type": "process", "id": f"process--{_sid(child)}", "pid": ev.get("ProcessId"),
                    "command_line": ev.get("CommandLine"), "x_image": ev.get("Image"), "x_host": host}
            edges = [self._edge(parent, child, "CREATE_CHILD_PROCESS", "Process", "Process", ts, [proc],
                                attrs={"image": ev.get("Image"), "parent_image": ev.get("ParentImage"), "host": host},
                                technique=tech)]
            if user:
                ua = {"type": "user-account", "id": f"user-account--{_sid(user)}", "account_login": user[5:]}
                edges.append(self._edge(user, child, "RUN", "User", "Process", ts, [ua, proc],
                                        attrs={"image": ev.get("Image"), "host": host}, technique=tech))
            ctx = {}
            if str(ev.get("User") or "").lower() in SYSTEM_NAMES:
                ctx["privilege"] = (host, "root")
            return edges, ctx
        if sysmon and eid == 3:
            initiated = str(ev.get("Initiated")).lower() == "true"
            sip, dip = str(ev.get("SourceIp") or ""), str(ev.get("DestinationIp") or "")
            if initiated and sip:
                self.ip_to_host[sip] = host
            elif dip:
                self.ip_to_host[dip] = host
            src, dst = self.resolve(sip), self.resolve(dip)
            if not src or not dst:
                return [], {}
            port = int(ev.get("DestinationPort") or 0)
            objs = [{"type": "ipv4-addr", "id": f"ipv4-addr--{sip}", "value": sip},
                    {"type": "ipv4-addr", "id": f"ipv4-addr--{dip}", "value": dip},
                    {"type": "network-traffic", "id": f"network-traffic--{_sid(sip, dip, port, ts)}",
                     "src_ref": f"ipv4-addr--{sip}", "dst_ref": f"ipv4-addr--{dip}", "dst_port": port,
                     "protocols": [str(ev.get("Protocol") or "tcp").lower()]}]
            tech = self.attack.for_port(port) if self.attack else None
            edge = self._edge(src, dst, "COMMUNICATED", "Host", "Host", ts, objs, port=port,
                              attrs={"image": ev.get("Image"), "user": ev.get("User")}, technique=tech)
            ctx = {"privilege": (host, "root")} if str(ev.get("User") or "").lower() in SYSTEM_NAMES else {}
            return [edge], ctx
        if sysmon and eid == 11:
            proc = f"proc:{host}:{ev.get('ProcessGuid')}"
            path = str(ev.get("TargetFilename") or "")
            if not path:
                return [], {}
            fid = f"file:{host}:{path.lower()}"
            fobj = {"type": "file", "id": f"file--{_sid(fid)}", "name": os.path.basename(path.replace(chr(92), '/')),
                    "x_path": path}
            return [self._edge(proc, fid, "WROTE_FILE", "Process", "File", ts, [fobj],
                               attrs={"image": ev.get("Image"), "host": host})], {}
        if eid == 4624:
            user = _user_id(ev.get("TargetDomainName"), ev.get("TargetUserName"))
            if not user:
                return [], {}
            ua = {"type": "user-account", "id": f"user-account--{_sid(user)}", "account_login": user[5:],
                  "x_logon_type": ev.get("LogonType")}
            src_ip = str(ev.get("IpAddress") or "")
            edges = [self._edge(user, host, "AUTHENTICATED", "User", "Host", ts, [ua],
                                attrs={"logon_type": ev.get("LogonType"), "src_ip": src_ip},
                                technique=self.attack.describe("T1078") if self.attack else None)]
            ctx = {}
            if str(ev.get("TargetUserSid") or "") in SYSTEM_SIDS:
                ctx["privilege"] = (host, "root")
            return edges, ctx
        if eid == 4625:
            return [], {"auth_failure_host": host}
        if eid == 4672:
            subject = _user_id(ev.get("SubjectDomainName"), ev.get("SubjectUserName"))
            tier = "root" if str(ev.get("SubjectUserSid") or "") in SYSTEM_SIDS else "local_admin"
            ctx = {"privilege": (host, tier)}
            if subject:
                ctx["user_privilege"] = (subject, tier)
            return [], ctx
        return [], {}

    # ---------------------------------------------------------------- AWS CloudTrail
    def _cloudtrail(self, rec: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        ts = parse_time(rec.get("eventTime"))
        ident = rec.get("userIdentity") or {}
        user = _user_id(ident.get("accountId"), ident.get("userName") or ident.get("arn") or ident.get("principalId"))
        sip = str(rec.get("sourceIPAddress") or "")
        if ts is None:
            return [], {}
        failed = bool(rec.get("errorCode")) or (rec.get("responseElements") or {}).get("ConsoleLogin") == "Failure"
        if rec.get("eventName") == "ConsoleLogin":
            if failed:
                return [], {"auth_failure_host": "aws-console"}
            if not user:
                return [], {}
            ua = {"type": "user-account", "id": f"user-account--{_sid(user)}", "account_login": user[5:],
                  "x_cloud": "aws"}
            return [self._edge(user, "aws-console", "AUTHENTICATED", "User", "Host", ts, [ua],
                               attrs={"src_ip": sip, "region": rec.get("awsRegion")})], {}
        if not sip or "." not in sip and ":" not in sip:     # AWS-internal callers (e.g. "ec2.amazonaws.com")
            return [], {}
        svc = f"aws:{rec.get('eventSource')}"
        objs = [{"type": "ipv4-addr", "id": f"ipv4-addr--{sip}", "value": sip}]
        ctx = {"auth_failure_host": svc} if failed and "AccessDenied" in str(rec.get("errorCode")) else {}
        return [self._edge(sip, svc, "COMMUNICATED", "Host", "Host", ts, objs, port=443,
                           attrs={"event": rec.get("eventName"), "user": user, "region": rec.get("awsRegion")})], ctx


# ============================================================================ sources
class MordorSource:
    """OTRF Security-Datasets (Mordor) JSON-lines captures, plain or zipped, in time order."""

    def __init__(self, paths: Iterable[str]):
        self.paths = []
        for p in ([paths] if isinstance(paths, str) else paths):
            self.paths.extend(sorted(glob.glob(p)) or [p])

    def _lines(self, path: str) -> Iterator[str]:
        if path.lower().endswith(".zip"):
            with zipfile.ZipFile(path) as z:
                for name in z.namelist():
                    if name.endswith(".json"):
                        with z.open(name) as f:
                            yield from io.TextIOWrapper(f, encoding="utf-8", errors="replace")
        else:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                yield from f

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        events = []
        for p in self.paths:
            for line in self._lines(p):
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = parse_time(ev.get("UtcTime")) or parse_time(ev.get("@timestamp")) or parse_time(ev.get("EventTime"))
                if t is not None:
                    events.append((t, ev))
        events.sort(key=lambda x: x[0])
        for _, ev in events:
            yield ev


class CloudTrailSource:
    """AWS CloudTrail log files ({"Records": [...]}, optionally gzipped JSON) in time order."""

    def __init__(self, paths: Iterable[str]):
        self.paths = []
        for p in ([paths] if isinstance(paths, str) else paths):
            self.paths.extend(sorted(glob.glob(p)) or [p])

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        import gzip
        records = []
        for p in self.paths:
            opener = gzip.open if p.endswith(".gz") else open
            with opener(p, "rt", encoding="utf-8") as f:
                data = json.load(f)
            records.extend(data.get("Records", data if isinstance(data, list) else []))
        records.sort(key=lambda r: parse_time(r.get("eventTime")) or 0.0)
        yield from records


class ZeekConnSource:
    """Zeek conn.log (TSV with #fields header, or JSON lines) -> raw network flow records.

    Features are mapped onto Role 1's FLOW_FEATURES contract where Zeek has an equivalent
    (duration, packets, bytes, rates, inter-arrival mean); unavailable ones are 0."""

    def __init__(self, path: str):
        self.path = path

    @staticmethod
    def features(r: Dict[str, Any]) -> List[float]:
        def num(k):
            try:
                v = float(r.get(k))
                return v if v == v else 0.0
            except (TypeError, ValueError):
                return 0.0
        dur = num("duration")
        op, rp, ob, rb = num("orig_pkts"), num("resp_pkts"), num("orig_bytes"), num("resp_bytes")
        hist = str(r.get("history") or "")
        vals = {
            "Flow Duration": dur * 1e6, "Total Fwd Packets": op, "Total Backward Packets": rp,
            "Total Length of Fwd Packets": ob, "Total Length of Bwd Packets": rb,
            "Fwd Packet Length Mean": ob / op if op else 0.0, "Bwd Packet Length Mean": rb / rp if rp else 0.0,
            "Flow Bytes/s": (ob + rb) / dur if dur else 0.0, "Flow Packets/s": (op + rp) / dur if dur else 0.0,
            "Flow IAT Mean": dur * 1e6 / max(op + rp - 1, 1), "Fwd IAT Mean": dur * 1e6 / max(op - 1, 1),
            "SYN Flag Count": float(hist.lower().count("s")), "PSH Flag Count": 0.0,
            "ACK Flag Count": float(hist.lower().count("a")), "Init_Win_bytes_forward": 0.0,
            "Init_Win_bytes_backward": 0.0,
        }
        return [vals[c] for c in FLOW_FEATURES]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        with open(self.path, "r", encoding="utf-8", errors="replace") as f:
            fields = None
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                if line.startswith("#fields"):
                    fields = line.split("\t")[1:]
                    continue
                if line.startswith("#"):
                    continue
                r = json.loads(line) if line.startswith("{") else (dict(zip(fields, line.split("\t"))) if fields else None)
                if not r or not r.get("id.orig_h") or not r.get("id.resp_h"):
                    continue
                yield {"src_ip": r["id.orig_h"], "dst_ip": r["id.resp_h"], "dst_port": int(float(r.get("id.resp_p") or 0)),
                       "timestamp": float(r.get("ts")), "proto": str(r.get("proto") or "tcp"),
                       "features": self.features(r), "zeek_uid": r.get("uid")}
