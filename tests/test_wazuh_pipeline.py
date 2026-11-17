import os
import pytest
from neo4j import GraphDatabase
from src.role1_graph.graph_engine import ReachabilityRiskEngine
from src.role1_graph.dual_timescale_skg import DualTimescaleSKG
from src.role1_graph.stix_normalizer import STIXNormalizer

@pytest.fixture(scope="module")
def neo4j_driver():
    uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "SecurePassword123")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
    except Exception as exc:  # no Neo4j reachable (e.g. GitHub Actions runner): integration test only
        driver.close()
        pytest.skip(f"Neo4j not reachable at {uri}: {type(exc).__name__}")
    yield driver
    # ทำความสะอาดข้อมูลจำลองหลังการทดสอบเสร็จสิ้น (Teardown) ถ้าอยากเห็นกราฟให้คอมเม้นไว้ ถ้าไม่ให้กราฟขึ้นลบคอมเม้นออกให้รันได้
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    driver.close()

def test_wazuh_to_skg_and_risk(neo4j_driver):
    # 1. Wazuh Raw Alerts (โครงสร้างคีย์ตาม STIXNormalizer ของ Role 1)
    wazuh_alert_1 = {
        "timestamp": "2026-11-10T04:00:00.000Z",
        "rule_id": "100201",
        "description": "MITRE T1021.002 SMB Lateral Movement",
        "src_ip": "192.168.1.100",
        "dst_ip": "192.168.1.50",
        "dst_port": 445,
        "proto": "tcp"
    }
    wazuh_alert_2 = {
        "timestamp": "2026-11-10T04:00:00.030Z", # ห่างกัน 30 ms อยู่ใน 50 ms window
        "rule_id": "100202",
        "description": "MITRE T1021.001 RDP Exploit towards Domain Controller",
        "src_ip": "192.168.1.50",
        "dst_ip": "10.0.0.1",
        "dst_port": 3389,
        "proto": "tcp"
    }

    # 2. Phase 1: STIX 2.1 Normalization
    normalizer = STIXNormalizer()
    events = []
    for log in [wazuh_alert_1, wazuh_alert_2]:
        edge_data = normalizer.normalize(log)
        assert edge_data is not None, f"Normalization failed for log: {log['rule_id']}"
        assert "stix_bundle" in edge_data, "Edge must contain full STIX 2.1 bundle"
        events.append(edge_data)

    assert len(events) == 2, "Expected 2 normalized STIX events"

    # 3. Phase 2: Dual-Timescale SKG 50ms & Neo4j Persistent Sync
    skg = DualTimescaleSKG(ttl_ms=50.0, neo4j_driver=neo4j_driver)
    for ev in events:
        src = ev["source"]
        dst = ev["target"]
        port = ev["port"]
        ts = ev["timestamp"]
        weight = 0.2 if port == 445 else 0.1
        
        # เพิ่มลง NetworkX In-Memory Graph (RAM)
        skg.add_edge(src, dst, port=port, timestamp=ts, weight=weight)
        # บันทึกความสัมพันธ์ลง Neo4j
        skg.sync_to_neo4j(src, dst, port=port, timestamp=ts)

    # 4. Phase 4: Topology-Aware Reachability Risk Engine
    engine = ReachabilityRiskEngine(crown_jewels={"10.0.0.1": 10.0})
    risk = engine.calculate_node_risk(skg.graph, "192.168.1.100", p_t=0.9)

    # ตรวจสอบว่าคำนวณ Path มุ่งสู่ Crown Jewel สำเร็จ
    assert risk > 0.0, "Risk score must be positive when path to Crown Jewel exists"