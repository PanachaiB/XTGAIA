import datetime
from typing import Dict, Any, Optional

class STIXNormalizer:
    def __init__(self):
        pass

    def normalize(self, raw_log: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        แปลง Raw Log ให้อยู่ในโครงสร้าง STIX 2.1 SCOs และคืนค่า Active Edge Interface
        """
        if not raw_log:
            return None

        # รองรับทั้งคีย์แบบ Wazuh / Sysmon / Network flows ทั่วไป
        src_ip = raw_log.get("src_ip") or raw_log.get("source_ip") or raw_log.get("SourceIp")
        dst_ip = raw_log.get("dst_ip") or raw_log.get("destination_ip") or raw_log.get("DestinationIp")
        dst_port = raw_log.get("dst_port") or raw_log.get("destination_port") or raw_log.get("DestinationPort", 0)
        timestamp = raw_log.get("timestamp") or raw_log.get("@timestamp") or datetime.datetime.utcnow().timestamp()

        if isinstance(timestamp, str):
            try:
                # แปลง ISO format สู่ POSIX timestamp
                dt = datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                ts_val = dt.timestamp()
            except Exception:
                ts_val = datetime.datetime.utcnow().timestamp()
        else:
            ts_val = float(timestamp)

        if not src_ip or not dst_ip:
            return None

        # โครงสร้าง STIX 2.1 SCO Object
        stix_bundle = {
            "type": "bundle",
            "id": f"bundle--{src_ip}-{dst_ip}-{int(ts_val)}",
            "objects": [
                {"type": "ipv4-addr", "id": f"ipv4-addr--{src_ip}", "value": src_ip},
                {"type": "ipv4-addr", "id": f"ipv4-addr--{dst_ip}", "value": dst_ip},
                {
                    "type": "network-traffic",
                    "id": f"network-traffic--{src_ip}-{dst_ip}-{dst_port}",
                    "src_ref": f"ipv4-addr--{src_ip}",
                    "dst_ref": f"ipv4-addr--{dst_ip}",
                    "dst_port": int(dst_port),
                    "protocols": [raw_log.get("proto", "tcp").lower()]
                }
            ]
        }

        # Shared Interface ตามสัญญาของ Person 1 ส่งต่อให้ระบบ
        active_edge = {
            "source": str(src_ip),
            "target": str(dst_ip),
            "port": int(dst_port),
            "timestamp": ts_val,
            "stix_bundle": stix_bundle,
            "features": raw_log.get("features", [1.0, 0.0, 0.0])
        }
        return active_edge