"""
Build data/samples/demo_warmup.csv: the 30,000 first-hour Friday flows the live demo replays as its warm-up,
so a fresh clone can run the demo without the full CICIDS2017 capture (285 MB).

Same selection as scripts/live_demo.py (Friday 12:00-13:00 in the corrected release's clock, where the capture starts at 12:00 and
the first hour is all benign; first 30,000 flows in time order),
only the columns the flow source reads. CICIDS2017 (corrected release, Engelen et al. 2021; Sharafaldin et
al. 2018) - cite both when redistributing.

  python scripts/make_demo_sample.py      # needs data/raw/cicids2017_improved/friday.csv
"""
import csv
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.role1_graph.flow_sources import FLOW_FEATURES, IMPROVED_FEATURE_NAMES, _parse_iso  # noqa: E402

SRC = os.path.join(REPO, "data", "raw", "cicids2017_improved", "friday.csv")
OUT = os.path.join(REPO, "data", "samples", "demo_warmup.csv")
START, END, N = "2017-07-07 12:00:00", "2017-07-07 13:00:00", 30000


def main():
    t0, t1 = _parse_iso(START), _parse_iso(END)
    feats = [IMPROVED_FEATURE_NAMES.get(c, c) for c in FLOW_FEATURES]
    keep = ["Src IP", "Dst IP", "Dst Port", "Protocol", "Timestamp", "Label", "Attempted Category"] + feats
    rows = []
    with open(SRC, encoding="latin-1", newline="") as f:
        r = csv.reader(f)
        header = [h.strip() for h in next(r)]
        idx = {h: i for i, h in enumerate(header)}
        missing = [c for c in keep if c not in idx]
        if missing:
            sys.exit(f"missing columns in {SRC}: {missing}")
        for order, row in enumerate(r):
            if len(row) < len(header) or not row[idx["Src IP"]].strip():
                continue
            ts = _parse_iso(row[idx["Timestamp"]])
            if ts is None or ts < t0 or ts >= t1:
                continue
            rows.append((ts, order, [row[idx[c]] for c in keep]))
    rows.sort(key=lambda x: (x[0], x[1]))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(keep)
        for _, _, vals in rows[:N]:
            w.writerow(vals)
    print(f"[ok] {min(N, len(rows)):,} flows -> {os.path.relpath(OUT, REPO)} ({os.path.getsize(OUT) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
