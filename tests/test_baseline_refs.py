import pytest

from src.evaluation.baseline_refs import ComparabilityError, footnote, load_baselines, side_by_side, table_eligible


@pytest.fixture(scope="module")
def refs():
    return load_baselines()


def test_every_reference_is_tagged(refs):
    assert refs and all(r.tag in {"DIRECT", "PROTOCOL-ONLY", "CONTEXT-ONLY", "QUALITATIVE"} for r in refs.values())


def test_direct_flow_reference_allowed(refs):
    row = side_by_side(refs, "ref11_ccstgn", unit="flow_edge", dataset="nf_unsw_nb15_v2")
    assert row["tag"] == "DIRECT" and "table2_f1_fnr_percent" in row["values"]


def test_context_only_refused(refs):
    with pytest.raises(ComparabilityError):
        side_by_side(refs, "ref12_yu_causal_gnn", unit="flow_edge")
    assert "context" in footnote(refs, "ref12_yu_causal_gnn")


def test_mixed_units_refused(refs):
    with pytest.raises(ComparabilityError):
        side_by_side(refs, "refN4_orthrus", unit="flow_edge")
    row = side_by_side(refs, "refN4_orthrus", unit="node_strict", training="supervised")
    assert any("training regime" in n for n in row["notes"])


def test_direct_dataset_mismatch_refused(refs):
    with pytest.raises(ComparabilityError):
        side_by_side(refs, "ref11_ccstgn", unit="flow_edge", dataset="cicids2017")


def test_air_latency_allowed_detection_refused(refs):
    assert side_by_side(refs, "ref5_air", unit="system", topic="latency")["tag"] == "DIRECT"
    with pytest.raises(ComparabilityError):
        side_by_side(refs, "ref5_air", unit="flow_edge", topic="detection")


def test_table_eligible_flow(refs):
    keys = table_eligible(refs, "flow_edge")
    assert "ref11_ccstgn" in keys and "ref3_vedesh_1dcnn" in keys
    assert "ref12_yu_causal_gnn" not in keys and "refN2_kairos" not in keys
