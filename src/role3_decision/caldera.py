"""Real MITRE CALDERA adversary emulation data for the Stage 5 PettingZoo environment (paper Table I:
"parameterized by CALDERA's operational abilities and facts").

Reads the official `stockpile` plugin (github.com/mitre/stockpile; zip or extracted folder):
  data/abilities/<tactic>/<id>.yml   abilities with their ATT&CK tactic and technique
  data/adversaries[/packs]/<id>.yml  adversary profiles: `atomic_ordering` of ability or pack IDs
Each adversary becomes an ordered chain of (tactic, technique) steps that the simulated attacker replays.
"""
import os
import zipfile
from functools import lru_cache
from typing import Dict, List, Tuple

import yaml

DEFAULT_STOCKPILE = "data/raw/caldera/stockpile.zip"


def _iter_yaml(source: str, part: str):
    if os.path.isdir(source):
        for root, _, files in os.walk(source):
            if f"/{part}" in root.replace("\\", "/") + "/":
                for f in files:
                    if f.endswith(".yml"):
                        with open(os.path.join(root, f), encoding="utf-8") as fh:
                            yield fh.read()
    else:
        with zipfile.ZipFile(source) as z:
            for n in z.namelist():
                if f"/data/{part}/" in n and n.endswith(".yml"):
                    yield z.read(n).decode("utf-8")


@lru_cache(maxsize=4)
def load_stockpile(source: str = DEFAULT_STOCKPILE) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """-> (abilities {id: {name, tactic, technique_id, technique}}, adversaries {id: {name, ordering}})."""
    abilities: Dict[str, dict] = {}
    for text in _iter_yaml(source, "abilities"):
        for a in yaml.safe_load(text) or []:
            if isinstance(a, dict) and a.get("id"):
                tech = a.get("technique") or {}
                abilities[a["id"]] = {"name": a.get("name"), "tactic": a.get("tactic"),
                                      "technique_id": tech.get("attack_id"), "technique": tech.get("name")}
    adversaries: Dict[str, dict] = {}
    for text in _iter_yaml(source, "adversaries"):
        doc = yaml.safe_load(text)
        for adv in (doc if isinstance(doc, list) else [doc]):
            if isinstance(adv, dict) and adv.get("id"):
                adversaries[adv["id"]] = {"name": str(adv.get("name", adv["id"])).strip(),
                                          "ordering": list(adv.get("atomic_ordering") or [])}
    return abilities, adversaries


def adversary_chain(adv_id: str, abilities: Dict[str, dict], adversaries: Dict[str, dict],
                    _seen=None) -> List[Tuple[str, str]]:
    """Expand an adversary's atomic ordering (packs resolved recursively) into (tactic, technique_id) steps."""
    seen = _seen or set()
    if adv_id in seen:
        return []
    seen.add(adv_id)
    steps: List[Tuple[str, str]] = []
    for ref in adversaries.get(adv_id, {}).get("ordering", []):
        if ref in abilities and abilities[ref].get("tactic"):
            steps.append((abilities[ref]["tactic"], abilities[ref].get("technique_id") or ""))
        elif ref in adversaries:
            steps.extend(adversary_chain(ref, abilities, adversaries, seen))
    return steps


def caldera_profiles(source: str = DEFAULT_STOCKPILE, supported=None) -> Dict[str, List[str]]:
    """{adversary name: tactic chain} for every adversary whose abilities map onto supported tactics."""
    abilities, adversaries = load_stockpile(source)
    out: Dict[str, List[str]] = {}
    for adv_id, adv in adversaries.items():
        chain = [t for t, _ in adversary_chain(adv_id, abilities, adversaries)
                 if supported is None or t in supported]
        if chain:
            name = adv["name"]
            out[name if name not in out else f"{name} ({adv_id[:8]})"] = chain
    return out
