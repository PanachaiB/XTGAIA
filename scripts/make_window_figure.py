"""
Figure: the two XTGAIA time windows on real traffic (slides / paper).

  Δc   = 50 ms   NetworkX active graph (DualTimescaleSKG TTL, src/role1_graph/dual_timescale_skg.py)
  Δinf = 500 ms  TGN update window, capped at 128 events (flush rule of src/pipeline/orchestrator.py, A4)

Replays two 4-second slices of the CICIDS2017 (corrected) Friday capture through the real DualTimescaleSKG
and the orchestrator's flush rule (event time, no model needed): a quiet benign stretch and the busiest
second of the DDoS.

  python scripts/make_window_figure.py        # -> docs/figures/fig5_time_windows.png
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from src.role1_graph.dual_timescale_skg import DualTimescaleSKG  # noqa: E402
from src.role1_graph.flow_sources import CICIDSFlowSource  # noqa: E402

FRIDAY = os.path.join(REPO, "data", "raw", "cicids2017_improved", "friday.csv")
OUT = os.path.join(REPO, "docs", "figures", "fig5_time_windows.png")
DC_MS, DINF_S, CAP, SLICE_S = 50.0, 0.5, 128, 4.0
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "hatch.color": "#ffffff", "hatch.linewidth": 1.0})


def load():
    rows = [(r["timestamp"], r["src_ip"], r["dst_ip"], r["dst_port"], r["label"])
            for r in CICIDSFlowSource([FRIDAY])._rows()]
    rows.sort(key=lambda r: r[0])
    return rows


def pick_slices(rows):
    ts = np.array([r[0] for r in rows])
    ddos = ts[[r[4].lower() == "ddos" for r in rows]]
    sec = np.floor(ddos).astype(np.int64)
    vals, counts = np.unique(sec, return_counts=True)
    peak = float(vals[np.argmax(counts)])
    # quiet benign: one hour after the capture starts, first 4 s window with only benign flows
    t = ts[0] + 3600.0
    while True:
        i, j = np.searchsorted(ts, [t, t + SLICE_S])
        if j - i >= 20 and all(rows[k][4].upper() == "BENIGN" for k in range(i, j)):
            break
        t += SLICE_S
    return [("Benign traffic", t), ("DDoS burst", peak - 1.0)]


def simulate(rows, t0):
    ts = np.array([r[0] for r in rows])
    i, j = np.searchsorted(ts, [t0, t0 + SLICE_S])
    skg = DualTimescaleSKG(ttl_ms=DC_MS)
    times, edges, blocks, batch = [], [], [], []
    for k in range(i, j):
        t, s, d, p, _ = rows[k]
        skg.add_edge(s, d, port=p, timestamp=t)
        times.append((t - t0) * 1000.0)
        edges.append(skg.graph.number_of_edges())
        batch.append(t)
        if len(batch) >= CAP or batch[-1] - batch[0] >= DINF_S:            # orchestrator flush rule
            blocks.append(((batch[0] - t0) * 1000.0, (batch[-1] - t0) * 1000.0, len(batch), len(batch) >= CAP))
            batch = []
    if batch:
        blocks.append(((batch[0] - t0) * 1000.0, (batch[-1] - t0) * 1000.0, len(batch), len(batch) >= CAP))
    return np.array(times), np.array(edges), blocks, j - i


def main():
    rows = load()
    slices = pick_slices(rows)
    fig, axes = plt.subplots(2, 2, figsize=(14, 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1], "hspace": 0.32, "wspace": 0.12})
    for col, (name, t0) in enumerate(slices):
        times, edges, blocks, n = simulate(rows, t0)
        a = axes[0, col]
        grid = np.arange(0.0, SLICE_S * 1000.0, 1.0)                         # 1 ms resolution
        in_win = np.searchsorted(times, grid, side="right") - np.searchsorted(times, grid - DC_MS, side="right")
        a.plot(grid, in_win, color=BLUE, linewidth=1.4, label="flows in the last 50 ms")
        a.step(times, edges, where="post", color=INK2, linewidth=1.0, label="host-pair edges in G_act")
        a.set_title(f"{name}  ·  {n / SLICE_S:,.0f} flows/s", loc="left")
        a.set_ylabel("Δc = 50 ms window")
        a.set_ylim(0, max(in_win.max(), edges.max()) * 1.15 + 1)
        if col == 1:
            a.legend(loc="upper left", frameon=False, fontsize=9)
        b = axes[1, col]
        for start, end, cnt, capped in blocks:
            w = max(end - start, 4.0)
            b.bar(start, cnt, width=w, align="edge", color=ORANGE if capped else BLUE,
                  hatch="////" if capped else None, edgecolor=SURFACE, linewidth=1)
        b.axhline(CAP, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
        b.text(SLICE_S * 1000 * 0.995, CAP * 1.04, "cap 128", ha="right", va="bottom", fontsize=8.5, color=INK2)
        b.set_ylabel("events per TGN update")
        b.set_ylim(0, CAP * 1.3)
        b.set_xlabel("event time (ms)")
        capped = sum(1 for x in blocks if x[3])
        b.set_title(f"Δinf = 500 ms: {len(blocks)} updates" + (f", {capped} closed by the cap" if capped else ""),
                    loc="left", fontsize=10, fontweight="normal", color=INK2)
        for ax in (a, b):
            ax.set_xlim(0, SLICE_S * 1000)
            ax.grid(axis="y", color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    fig.suptitle("XTGAIA time windows on real traffic (CICIDS2017 Friday)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.legend(handles=[Patch(facecolor=BLUE, label="TGN update closed by the 500 ms window"),
                        Patch(facecolor=ORANGE, hatch="////", edgecolor=SURFACE, label="closed by the 128-event cap")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.14)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print("[ok]", os.path.relpath(OUT, REPO))


if __name__ == "__main__":
    main()
