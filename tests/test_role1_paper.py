"""Role 1 paper-fidelity tests: heterogeneous SKG, telemetry sources, Algorithm 1/2, NVD and ATT&CK."""
import json

import networkx as nx
import pytest

from src.role1_graph.host_context import AssetInventory
from src.role1_graph.live_graph import LiveGraphService
from src.role1_graph.telemetry import (CloudTrailSource, HostTelemetryNormalizer, MordorSource, ZeekConnSource,
                                       parse_time)
from src.role1_graph.threat_intel import AttackMapper, NVDClient

HOST = "workstation5.theshire.local"


def sysmon(eid, **kw):
    return {"Channel": "Microsoft-Windows-Sysmon/Operational", "EventID": eid, "Hostname": HOST.upper(),
            "UtcTime": kw.pop("UtcTime", "2020-09-20 12:16:08.100"), **kw}


def security(eid, **kw):
    return {"Channel": "Security", "EventID": eid, "Hostname": HOST, "@timestamp": "2020-09-20T12:16:09.000Z", **kw}


def inventory(**kw):
    return AssetInventory(crown_jewels={"10.0.0.1": 10.0, "dc": 8.0}, hosts={"10.0.0.1": {"cvss_max": 9.0}},
                          internal_subnets=["10.0.0.0/24"], **kw)


def service(**kw):
    return LiveGraphService(inventory(), "configs/mitre_t1021_ports.json", window_ms=60000, auth_ports=[22, 445], **kw)


# ---------------------------------------------------------------------------- telemetry -> typed SKG
def test_sysmon_and_security_events_map_to_paper_edge_types():
    n = HostTelemetryNormalizer(AttackMapper(None))
    e, _ = n.normalize(sysmon(1, ProcessGuid="{c}", ParentProcessGuid="{p}", Image=r"C:\Windows\PSEXESVC.exe",
                              ParentImage=r"C:\Windows\services.exe", User="THESHIRE\\pgustavo", CommandLine="x"))
    assert {x["etype"] for x in e} == {"CREATE_CHILD_PROCESS", "RUN"}
    run = next(x for x in e if x["etype"] == "RUN")
    assert run["source"] == "user:theshire\\pgustavo" and run["src_type"] == "User" and run["dst_type"] == "Process"
    assert run["technique"]["id"] == "T1569.002"                                   # PsExec service execution
    assert {o["type"] for o in run["stix_bundle"]["objects"]} == {"user-account", "process"}
    e, _ = n.normalize(sysmon(11, ProcessGuid="{c}", TargetFilename=r"C:\Temp\payload.exe", Image="x"))
    assert e[0]["etype"] == "WROTE_FILE" and e[0]["dst_type"] == "File"
    e, _ = n.normalize(sysmon(3, SourceIp="172.18.39.5", DestinationIp="172.18.38.5", DestinationPort=445,
                              Initiated="true", Protocol="tcp", User="NT AUTHORITY\\SYSTEM", Image="System"))
    assert e[0]["etype"] == "COMMUNICATED" and e[0]["source"] == HOST and e[0]["technique"]["id"] == "T1021.002"
    e, ctx = n.normalize(security(4624, TargetUserName="pgustavo", TargetDomainName="THESHIRE", LogonType=3,
                                  IpAddress="172.18.39.5"))
    assert e[0]["etype"] == "AUTHENTICATED" and e[0]["target"] == HOST
    _, ctx = n.normalize(security(4625))
    assert ctx == {"auth_failure_host": HOST}
    _, ctx = n.normalize(security(4672, SubjectUserName="pgustavo", SubjectDomainName="THESHIRE",
                                  SubjectUserSid="S-1-5-21-1"))
    assert ctx["privilege"] == (HOST, "local_admin") and ctx["user_privilege"][1] == "local_admin"


def test_live_graph_holds_typed_nodes_and_observed_privilege_raises_s_priv():
    g = service()
    for ev in [sysmon(1, ProcessGuid="{c}", ParentProcessGuid="{p}", Image="cmd.exe", User="THESHIRE\\bob"),
               security(4624, TargetUserName="bob", TargetDomainName="THESHIRE", LogonType=3),
               sysmon(3, SourceIp="10.0.0.7", DestinationIp="10.0.0.1", DestinationPort=3389, Initiated="true")]:
        g.ingest_all(ev)
    types = nx.get_node_attributes(g.skg.graph, "ntype")
    assert {"Host", "User", "Process"} <= set(types.values())
    assert g.stats["typed_edges"] == {"CREATE_CHILD_PROCESS": 1, "RUN": 1, "AUTHENTICATED": 1, "COMMUNICATED": 1}
    before = g.edge_weight(HOST, "10.0.0.1", 3389)[2]
    g.ingest_all(security(4672, SubjectUserName="SYSTEM", SubjectUserSid="S-1-5-18"))
    assert before == 0.2 and g.edge_weight(HOST, "10.0.0.1", 3389)[2] == 1.0       # S_priv(u): User -> SYSTEM


def test_inactive_nodes_leave_v_act_and_are_reported():
    g = LiveGraphService(inventory(), "configs/mitre_t1021_ports.json", window_ms=50)      # paper Δc = 50 ms
    flow = lambda s, d, t: {"src_ip": s, "dst_ip": d, "dst_port": 80, "timestamp": t, "features": [0.0] * 16}
    g.ingest_all(flow("10.0.0.7", "10.0.0.1", 100.000))
    g.ingest_all(flow("10.0.0.8", "10.0.0.9", 100.040))
    assert set(g.skg.graph.nodes) == {"10.0.0.7", "10.0.0.1", "10.0.0.8", "10.0.0.9"}
    g.ingest_all(flow("10.0.0.8", "10.0.0.9", 100.070))                            # first edge is > 50 ms old
    assert "10.0.0.7" not in g.skg.graph and sorted(g.pop_evicted()) == ["10.0.0.1", "10.0.0.7"]


def test_all_node_risk_matches_per_node_algorithm2():
    g = service()
    for s, d, port in [("10.0.0.7", "10.0.0.8", 445), ("10.0.0.8", "10.0.0.1", 3389), ("10.0.0.9", "10.0.0.7", 80)]:
        g.ingest_all({"src_ip": s, "dst_ip": d, "dst_port": port, "timestamp": 1.0, "features": [0.0] * 16})
    p = {"10.0.0.7": 0.9, "10.0.0.8": 0.5, "10.0.0.9": 0.2, "10.0.0.1": 1.0}
    allr = g.compute_all_risks(p)
    for v, pv in p.items():
        assert allr[v] == pytest.approx(g.compute_risk(v, pv))


# ---------------------------------------------------------------------------- sources
def test_mordor_cloudtrail_and_zeek_sources(tmp_path):
    mordor = tmp_path / "capture.json"
    mordor.write_text("\n".join(json.dumps(e) for e in [security(4625), sysmon(1, ProcessGuid="a", UtcTime="2020-09-20 12:16:07.000")]))
    evs = list(MordorSource([str(mordor)]))
    assert [e["EventID"] for e in evs] == [1, 4625]                                # time ordered
    ct = tmp_path / "ct.json"
    ct.write_text(json.dumps({"Records": [
        {"eventTime": "2024-01-01T10:00:00Z", "eventSource": "signin.amazonaws.com", "eventName": "ConsoleLogin",
         "sourceIPAddress": "203.0.113.5", "userIdentity": {"userName": "alice", "accountId": "1"},
         "responseElements": {"ConsoleLogin": "Failure"}},
        {"eventTime": "2024-01-01T10:01:00Z", "eventSource": "s3.amazonaws.com", "eventName": "GetObject",
         "sourceIPAddress": "203.0.113.5", "userIdentity": {"userName": "alice"}}]}))
    n = HostTelemetryNormalizer()
    out = [n.normalize(r) for r in CloudTrailSource([str(ct)])]
    assert out[0] == ([], {"auth_failure_host": "aws-console"})
    assert out[1][0][0]["etype"] == "COMMUNICATED" and out[1][0][0]["target"] == "aws:s3.amazonaws.com"
    zeek = tmp_path / "conn.log"
    zeek.write_text("#fields\tts\tuid\tid.orig_h\tid.orig_p\tid.resp_h\tid.resp_p\tproto\tduration\torig_bytes\t"
                    "resp_bytes\torig_pkts\tresp_pkts\thistory\n"
                    "1600000000.5\tC1\t10.0.0.7\t50000\t10.0.0.1\t445\ttcp\t2.0\t100\t300\t4\t2\tShADad\n")
    rec = next(iter(ZeekConnSource(str(zeek))))
    assert rec["dst_port"] == 445 and rec["features"][0] == pytest.approx(2.0e6) and len(rec["features"]) == 16
    assert parse_time("2020-09-20T12:16:09.000Z") == pytest.approx(1600604169.0)


# ---------------------------------------------------------------------------- NVD / ATT&CK
def test_nvd_client_parses_cvss_and_caches(tmp_path, monkeypatch):
    calls = []

    class R:
        status_code = 200

        @staticmethod
        def json():
            return {"vulnerabilities": [{"cve": {"metrics": {"cvssMetricV31": [{"cvssData": {"baseScore": 7.5}}]}}}]}

    monkeypatch.setattr("src.role1_graph.threat_intel.requests.get", lambda *a, **k: calls.append(k) or R())
    nvd = NVDClient(cache_path=str(tmp_path / "nvd.json"), min_interval_s=0)
    assert nvd.base_score("cve-2014-0160") == 7.5 and nvd.base_score("CVE-2014-0160") == 7.5 and len(calls) == 1
    inv = AssetInventory({}, {"h": {"cvss_max": 5.0, "cves": ["CVE-2014-0160"]}}, nvd=nvd)
    assert inv.cvss("h") == 7.5
    assert NVDClient(cache_path=str(tmp_path / "nvd.json"), enabled=False).base_score("CVE-2014-0160") == 7.5


def test_attack_mapper_reads_stix_bundle(tmp_path):
    bundle = {"type": "bundle", "objects": [{"type": "attack-pattern", "name": "SMB/Windows Admin Shares",
              "external_references": [{"source_name": "mitre-attack", "external_id": "T1021.002"}],
              "kill_chain_phases": [{"kill_chain_name": "mitre-attack", "phase_name": "lateral-movement"}]}]}
    p = tmp_path / "attack.json"
    p.write_text(json.dumps(bundle))
    m = AttackMapper(str(p))
    assert m.for_port(445) == {"id": "T1021.002", "name": "SMB/Windows Admin Shares", "tactics": ["lateral-movement"]}
    assert m.for_port(8443) is None
