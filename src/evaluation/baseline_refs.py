"""
Published baseline values (configs/baselines.yaml) with the comparison spec's guard rails (spec tasks 1 and 9).

Every reference carries a comparability tag (DIRECT | PROTOCOL-ONLY | CONTEXT-ONLY | QUALITATIVE), an
evaluation unit (flow_edge, node_strict, node_lenient_2hop, time_window_15min, ...) and a training regime
(supervised, benign_only, ...). `side_by_side` is the only way to put a published value in a results table:

  * CONTEXT-ONLY and QUALITATIVE references are refused (they belong in a footnote: `footnote`);
  * a reference whose unit differs from the table's unit is refused (rules 6 and 8 of the spec);
  * a training-regime mismatch is allowed but flagged, so the table can say so.

  from src.evaluation.baseline_refs import load_baselines, side_by_side
  refs = load_baselines()
  row = side_by_side(refs, "ref11_ccstgn", unit="flow_edge", dataset="nf_unsw_nb15_v2")
"""
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_PATH = os.path.join(REPO, "configs", "baselines.yaml")
TABLE_TAGS = {"DIRECT", "PROTOCOL-ONLY"}


class ComparabilityError(ValueError):
    """Raised when a published value must not appear side by side with ours."""


@dataclass
class BaselineRef:
    key: str
    tag: str
    unit: Optional[str]
    training: Optional[str]
    dataset_key: Optional[str]
    topic_tags: Dict[str, str] = field(default_factory=dict)
    values: Dict[str, Any] = field(default_factory=dict)

    def tag_for(self, topic: str) -> str:
        return self.topic_tags.get(topic, self.tag)


def load_baselines(path: str = DEFAULT_PATH) -> Dict[str, BaselineRef]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    refs = {}
    for key, body in raw.items():
        body = dict(body or {})
        meta = {k: body.pop(k, None) for k in ("tag", "unit", "training", "dataset_key", "topic_tags")}
        if not meta["tag"]:
            raise ValueError(f"{key}: missing comparability tag")
        refs[key] = BaselineRef(key=key, tag=str(meta["tag"]), unit=meta["unit"], training=meta["training"],
                                dataset_key=meta["dataset_key"], topic_tags=dict(meta["topic_tags"] or {}),
                                values=body)
    return refs


def side_by_side(refs: Dict[str, BaselineRef], key: str, unit: str, dataset: Optional[str] = None,
                 topic: str = "detection", training: Optional[str] = "supervised") -> Dict[str, Any]:
    """Return the reference's values for a results table, or raise ComparabilityError."""
    ref = refs[key]
    tag = ref.tag_for(topic)
    if tag not in TABLE_TAGS:
        raise ComparabilityError(f"{key} is {tag} for {topic}: footnote only, never in a side-by-side table")
    if topic == "detection":
        if ref.unit != unit:
            raise ComparabilityError(f"{key} is evaluated per {ref.unit}, the table per {unit}: different units "
                                     "must not share a table")
        if tag == "DIRECT" and dataset is not None and ref.dataset_key != dataset:
            raise ComparabilityError(f"{key} is DIRECT only on {ref.dataset_key}, not {dataset}")
    notes = []
    if tag == "PROTOCOL-ONLY":
        notes.append("protocol only: published numbers are on different data")
    if training and ref.training and ref.training != training:
        notes.append(f"training regime differs ({ref.training} vs {training})")
    return {"key": key, "tag": tag, "unit": ref.unit, "training": ref.training, "notes": notes,
            "values": ref.values}


def footnote(refs: Dict[str, BaselineRef], key: str) -> str:
    ref = refs[key]
    ds = ref.values.get("dataset") or ref.values.get("datasets") or ref.dataset_key
    return (f"{key} ({ref.tag}; unit {ref.unit or 'n/a'}, training {ref.training or 'n/a'}; data: {ds}) - "
            "cited for context, not tabulated")


def table_eligible(refs: Dict[str, BaselineRef], unit: str, topic: str = "detection") -> List[str]:
    out = []
    for key in refs:
        try:
            side_by_side(refs, key, unit=unit, topic=topic, training=None)
            out.append(key)
        except ComparabilityError:
            pass
    return out
