import json
import time
import uuid
from datetime import datetime, timezone

class XAIBundler:
    """
    XTGAIA Stage 6: Encapsulates structural and temporal XAI artifacts
    into a standardized STIX 2.1 Extension JSON bundle within <50 ms budget.
    """
    def __init__(self, identity_name: str = "XTGAIA-XAI-Engine"):
        self.identity_name = identity_name

    def build_stix_bundle(
        self,
        node_id: int | str,
        threat_prob: float,
        structural_edges: list[tuple[int, int, float]],
        temporal_analysis: dict,
        explainer_model: str = "PGExplainer",
        rollout_method: str = "AttentionRollout",
        extra: dict | None = None
    ) -> dict:
        """
        แปลงผลการวิเคราะห์ XAI ให้เป็น STIX 2.1 Extension Bundle
        """
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        indicator_id = f"indicator--{uuid.uuid4()}"
        bundle_id = f"bundle--{uuid.uuid4()}"

        # โครงสร้างคำอธิบาย XAI Custom Extension
        xai_extension_data = {
            "threat_confidence_pt": round(float(threat_prob), 4),
            "structural_attribution": {
                "influential_2hop_edges": [
                    {"src": u, "dst": v, "importance_weight": weight}
                    for u, v, weight in structural_edges
                ],
                "explainer_model": explainer_model
            },
            "temporal_attribution": {
                "attack_onset_ms": temporal_analysis.get("onset_timestamp_ms", 0.0),
                "attack_duration_ms": temporal_analysis.get("attack_duration_ms", 0.0),
                "temporal_trace_weights": temporal_analysis.get("temporal_trace", []),
                "rollout_method": rollout_method
            }
        }
        if extra:
            xai_extension_data.update(extra)

        # STIX 2.1 Indicator Object พร้อม Custom Properties
        indicator_obj = {
            "type": "indicator",
            "spec_version": "2.1",
            "id": indicator_id,
            "created": now_iso,
            "modified": now_iso,
            "name": f"XTGAIA Threat Alert: Node {node_id}",
            "description": "Temporal Graph Network compromise detection with multi-tiered XAI attribution.",
            "indicator_types": ["malicious-activity"],
            "pattern": f"[network-traffic:dst_ref.value = '{node_id}']",
            "pattern_type": "stix",
            "valid_from": now_iso,
            "x_xtgaia_xai": xai_extension_data
        }

        # STIX 2.1 Bundle Wrapper
        bundle = {
            "type": "bundle",
            "id": bundle_id,
            "objects": [indicator_obj]
        }

        return bundle

    def export_json(self, bundle: dict) -> str:
        """แปลง STIX Bundle เป็น JSON String สำหรับยิงผ่าน Webhook หรือบันทึกไฟล์"""
        return json.dumps(bundle, indent=2)


if __name__ == "__main__":
    start_time = time.perf_counter()
    bundler = XAIBundler()

    # ข้อมูลทดสอบจำลอง
    mock_structural = [(2, 0, 0.5458), (1, 2, 0.5098)]
    mock_temporal = {
        "onset_timestamp_ms": 187.5,
        "attack_duration_ms": 312.5,
        "temporal_trace": [0.0384, 0.0506, 0.079, 0.2719, 0.0924, 0.0564, 0.0413, 0.3701]
    }

    bundle = bundler.build_stix_bundle(
        node_id="192.168.1.105",
        threat_prob=0.885,
        structural_edges=mock_structural,
        temporal_analysis=mock_temporal
    )

    elapsed_ms = (time.perf_counter() - start_time) * 1000
    json_str = bundler.export_json(bundle)

    print(f"[✓] STIX 2.1 Extension Bundle successfully generated.")
    print(f"[✓] Bundle ID: {bundle['id']}")
    print(f"[✓] Execution Time: {elapsed_ms:.2f} ms (Budget: <50.0 ms)")