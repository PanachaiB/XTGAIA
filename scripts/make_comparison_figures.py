"""
Comparison figures for every topic of the paper's evaluation (Sec. IV-C to IV-E + response policy).

Three groups, encoded by colour AND texture (never colour alone):
  XTGAIA (ours)                   blue, solid
  Baselines run in-house          orange, solid     (our runs of the other models / explainers / playbooks)
  Published (paper-reported)      aqua, hatched     (numbers copied from the comparator papers)

Values are copied from the committed results (sources in comments) - no new measurement here.

  python scripts/make_comparison_figures.py        # -> docs/figures/*.png
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "figures")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
GROUP = {"ours": ("#2a78d6", None, "XTGAIA (ours)"),
         "inhouse": ("#eb6834", None, "Baseline (run in-house)"),
         "published": ("#1baf7a", "////", "Published value"),
         "variant": ("#2a78d6", "xxx", "XTGAIA variant (not deployed)")}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE, "hatch.color": "#ffffff", "hatch.linewidth": 1.0})


def hbar(ax, rows, title, xlabel, fmt="{:.3f}", log=False, xlim=None, better="higher"):
    """rows: [(label, value, group)] top to bottom. Value labels on every bar (few bars; relief rule)."""
    labels = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    groups = [r[2] for r in rows][::-1]
    y = range(len(rows))
    for yi, v, g in zip(y, vals, groups):
        color, hatch, _ = GROUP[g]
        ax.barh(yi, v, height=0.62, color=color, hatch=hatch, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(list(y), labels)
    for t, g in zip(ax.get_yticklabels(), groups):
        if g == "ours":
            t.set_fontweight("bold")
    if log:
        ax.set_xscale("log")
    if xlim:
        ax.set_xlim(*xlim)
    lo, hi = ax.get_xlim()
    box = dict(facecolor=SURFACE, edgecolor="none", pad=0.6)
    for yi, v in zip(y, vals):
        if v < 0 and not log:                       # negative bar: label beyond its left end
            ax.text(v - (hi - lo) * 0.01, yi, fmt.format(v), va="center", ha="right", fontsize=8, color=INK, bbox=box)
            continue
        x = v * 1.08 if log else v + (hi - lo) * 0.01
        ax.text(x, yi, fmt.format(v), va="center", ha="left", fontsize=8, color=INK, bbox=box, zorder=5)
    if not log:
        ax.set_xlim(lo, hi + (hi - lo) * 0.14)
    else:
        ax.set_xlim(lo, hi * 4)
    ax.set_title(title, loc="left")
    arrow = "↑ better" if better == "higher" else "↓ better"
    ax.set_xlabel(f"{xlabel + '  ' if xlabel else ''}{arrow}{'  (log)' if log else ''}")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)


def legend(fig, groups=("ours", "inhouse", "published"), y=0.005):
    handles = [Patch(facecolor=GROUP[g][0], hatch=GROUP[g][1], edgecolor=SURFACE, label=GROUP[g][2]) for g in groups]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, y))


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("[ok]", os.path.relpath(path))


# ------------------------------------------------------------------------------------------------ 1. detection
def detection_unsw():
    # docs/evaluation_results.md, NF-UNSW-NB15-v2 chronological 70/10/20 (CCSTGN [11] protocol); TGN = tuned final
    f1_1 = [("XTGAIA TGN", 0.858, "ours"), ("Random Forest", 0.947, "inhouse"), ("XGBoost", 0.938, "inhouse"),
            ("MLP", 0.936, "inhouse"), ("1D-CNN", 0.937, "inhouse"), ("E-GraphSAGE [N1]", 0.810, "inhouse"),
            ("MLP [11]", 0.952, "published"), ("CCSTGN-rs [11]", 0.937, "published"),
            ("CCSTGN-iso [11]", 0.948, "published"), ("CCSTGN-rs ft [11]", 0.950, "published"),
            ("CCSTGN-iso ft [11]", 0.949, "published")]
    f1_01 = [("XTGAIA TGN", 0.467, "ours"), ("Random Forest", 0.969, "inhouse"), ("XGBoost", 0.962, "inhouse"),
             ("MLP", 0.893, "inhouse"), ("1D-CNN", 0.913, "inhouse"), ("E-GraphSAGE [N1]", 0.445, "inhouse"),
             ("MLP [11]", 0.823, "published"), ("CCSTGN-rs [11]", 0.770, "published"),
             ("CCSTGN-iso [11]", 0.608, "published"), ("CCSTGN-rs ft [11]", 0.702, "published"),
             ("CCSTGN-iso ft [11]", 0.701, "published")]
    auc = [("XTGAIA TGN", 0.9953, "ours"), ("Random Forest", 0.9997, "inhouse"), ("XGBoost", 0.9997, "inhouse"),
           ("MLP", 0.9993, "inhouse"), ("1D-CNN", 0.9993, "inhouse"), ("E-GraphSAGE [N1]", 0.9940, "inhouse"),
           ("CCSTGN-iso [11]", 0.9989, "published"), ("MLP [11]", 0.9991, "published"),
           ("3D-IDS [11]†", 0.9155, "published"), ("E-GraphSAGE [11]†", 0.9039, "published"),
           ("TGN [11]†", 0.8801, "published")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    hbar(axes[0], f1_1, "F1 at FPR 1 %", "F1", xlim=(0.7, 1.0))
    hbar(axes[1], f1_01, "F1 at FPR 0.1 %", "F1", xlim=(0.0, 1.0))
    hbar(axes[2], auc, "ROC-AUC", "ROC-AUC", fmt="{:.4f}", xlim=(0.85, 1.0))
    fig.suptitle("Detection: NF-UNSW-NB15-v2 (CCSTGN protocol)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    legend(fig, y=-0.04)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    save(fig, "fig1a_detection_unsw.png")


def design_check():
    # runs/design_check/summary.md (validation only, NF-UNSW-NB15-v2; flow head rejected by design, B10)
    # hybrid and MLP: mean of 3 seeds; duplicate-free = validation flows whose features never occur in train
    f1 = [("Node-only, dst read-out", 0.856, "ours"), ("Node-only, max read-out", 0.882, "ours"),
          ("Hybrid flow head", 0.950, "variant"), ("MLP (flow features)", 0.955, "inhouse")]
    f1_dedup = [("Node-only, dst read-out", 0.873, "ours"), ("Hybrid flow head", 0.939, "variant"),
                ("MLP (flow features)", 0.944, "inhouse")]
    auc = [("Node-only, dst read-out", 0.9954, "ours"), ("Node-only, max read-out", 0.9957, "ours"),
           ("Hybrid flow head", 0.9984, "variant"), ("MLP (flow features)", 0.9989, "inhouse")]
    tpr = [("Node-only, dst read-out", 0.195, "ours"), ("Node-only, max read-out", 0.203, "ours"),
           ("Hybrid flow head", 0.459, "variant"), ("MLP (flow features)", 0.648, "inhouse")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.0))
    hbar(axes[0], f1, "F1", "", xlim=(0.8, 1.0))
    hbar(axes[1], f1_dedup, "F1, duplicate-free", "", xlim=(0.8, 1.0))
    hbar(axes[2], auc, "ROC-AUC", "", fmt="{:.4f}", xlim=(0.99, 1.0))
    hbar(axes[3], tpr, "TPR at FPR 0.1 %", "", xlim=(0.0, 1.0))
    fig.suptitle("Design check: hybrid flow head (NF-UNSW-NB15-v2, validation)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    legend(fig, groups=("ours", "variant", "inhouse"), y=-0.12)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    save(fig, "fig1c_design_check_flow_head.png")


def detection_cicids():
    # docs/evaluation_results.md, CICIDS2017 held-out Friday (train Mon-Thu, test Friday)
    f1 = [("XTGAIA TGN", 0.970, "ours"),
          ("Random Forest", 0.9705, "inhouse"), ("XGBoost", 0.9934, "inhouse"), ("E-GraphSAGE [N1]", 0.9173, "inhouse")]
    auc = [("XTGAIA TGN", 0.9979, "ours"),
           ("Random Forest", 0.9998, "inhouse"), ("XGBoost", 0.9999, "inhouse"), ("E-GraphSAGE [N1]", 0.9945, "inhouse")]
    fpr = [("XTGAIA TGN", 0.0493, "ours"),
           ("Random Forest", 0.0000, "inhouse"), ("XGBoost", 0.0012, "inhouse"), ("E-GraphSAGE [N1]", 0.0094, "inhouse")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6))
    hbar(axes[0], f1, "F1", "F1", xlim=(0.85, 1.0))
    hbar(axes[1], auc, "ROC-AUC", "ROC-AUC", fmt="{:.4f}", xlim=(0.98, 1.0))
    hbar(axes[2], fpr, "FPR", "FPR", fmt="{:.4f}", better="lower")
    fig.suptitle("Detection: CICIDS2017, held-out Friday",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    legend(fig, groups=("ours", "inhouse"), y=-0.08)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    save(fig, "fig1b_detection_cicids_heldout.png")


# ------------------------------------------------------------------------------------------------ 2. latency
def latency():
    # docs/handoff_latency_and_demo.md (laptop baseline, full SOC stack) vs AIR [5] / Kulkarni [2] reported
    stages = [("Acquisition", 0.07, 8.1), ("Analysis", 37.8, 15.2),
              ("Decision", 42.9, 23.5), ("Execution", 625.8, 11.5),
              ("End-to-end", 943.2, 58.3)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw={"width_ratios": [2.2, 1]})
    ax = axes[0]
    n = len(stages)
    for i, (name, ours, air) in enumerate(stages[::-1]):
        ax.barh(i + 0.18, ours, height=0.34, color=GROUP["ours"][0], edgecolor=SURFACE, linewidth=2)
        ax.barh(i - 0.18, air, height=0.34, color=GROUP["published"][0], hatch="////", edgecolor=SURFACE, linewidth=2)
        ax.text(ours * 1.1, i + 0.18, f"{ours:g}", va="center", fontsize=8, color=INK)
        ax.text(air * 1.1, i - 0.18, f"{air:g}", va="center", fontsize=8, color=INK)
    ax.set_yticks(range(n), [s[0] for s in stages[::-1]])
    ax.set_xscale("log")
    ax.set_xlim(0.03, 10000)
    ax.set_title("Per stage vs AIR [5]", loc="left")
    ax.set_xlabel("ms  ↓ better  (log)")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    hbar(axes[1], [("XTGAIA MTTR", 896.5, "ours"), ("Kulkarni [2]", 1800.0, "published"),
                   ("SOAR, in [2]", 15700.0, "published")],
         "MTTR", "ms", fmt="{:,.0f}", log=True, better="lower")
    fig.suptitle("System latency",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    legend(fig, groups=("ours", "published"), y=-0.05)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    save(fig, "fig2_latency.png")


# ------------------------------------------------------------------------------------------------ 3. XAI
def xai():
    # runs/final_metrics_deployed (XTGAIA, 1,200 detections) and data/processed/eval/explainer_baselines_1000.json
    lat = [("XTGAIA", 25.2, "ours"), ("TreeSHAP (XGBoost)", 1.3, "inhouse"),
           ("TreeSHAP (RF)", 2.4, "inhouse"), ("LIME (XGBoost)", 99.9, "inhouse"), ("LIME (RF)", 144.6, "inhouse"),
           ("LIME (1D-CNN)", 167.4, "inhouse"), ("KernelSHAP (1D-CNN)", 953.9, "inhouse")]
    fid = [("XTGAIA", 0.002, "ours"), ("TreeSHAP (XGBoost)", 0.790, "inhouse"),
           ("TreeSHAP (RF)", 0.266, "inhouse"), ("LIME (XGBoost)", 0.363, "inhouse"), ("LIME (RF)", 0.293, "inhouse"),
           ("LIME (1D-CNN)", 0.105, "inhouse"), ("KernelSHAP (1D-CNN)", 0.461, "inhouse")]
    stab = [("XTGAIA", 0.984, "ours"), ("TreeSHAP (XGBoost)", 0.980, "inhouse"),
            ("TreeSHAP (RF)", 0.983, "inhouse"), ("LIME (XGBoost)", 0.703, "inhouse"), ("LIME (RF)", 0.804, "inhouse"),
            ("LIME (1D-CNN)", 0.848, "inhouse"), ("KernelSHAP (1D-CNN)", 0.731, "inhouse")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    hbar(axes[0], lat, "Latency", "ms", fmt="{:g}", log=True, better="lower")
    axes[0].axvspan(200, axes[0].get_xlim()[1], color="#ecebe7", zorder=0, linewidth=0)
    axes[0].text(200 * 1.08, 6.55, "> 200 ms", fontsize=7.5, color=INK2)
    hbar(axes[1], fid, "Fidelity", "Δp", xlim=(0, 1.0))
    hbar(axes[2], stab, "Stability", "", xlim=(0.5, 1.0))
    fig.suptitle("Explainability",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    legend(fig, groups=("ours", "inhouse"), y=-0.06)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    save(fig, "fig3_xai.png")


# ------------------------------------------------------------------------------------------------ 4. response
def response():
    # docs/evaluation_results.md, CALDERA-driven zone environment (src/evaluation/response_eval.py)
    pol = ["XTGAIA", "Static playbook", "Graded rules", "No response"]
    grp = ["ours", "inhouse", "inhouse", "inhouse"]
    metrics = [("Mean return", [12.99, 7.94, 9.81, -30.86], "higher", "{:.2f}"),
               ("Disruption", [0.050, 1.388, 0.729, 0.000], "lower", "{:.3f}"),
               ("Response cost", [0.41, 0.87, 0.57, 0.00], "lower", "{:.2f}"),
               ("Containment rate", [1.000, 1.000, 1.000, 0.354], "higher", "{:.3f}")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
    for ax, (title, vals, better, fmt) in zip(axes, metrics):
        rows = list(zip(pol, vals, grp))
        hbar(ax, rows, title, "", fmt=fmt, better=better)
        if min(vals) < 0:
            ax.set_xlim(min(vals) * 1.55, max(vals) * 1.6)
            ax.axvline(0, color=INK2, linewidth=0.8)
    fig.suptitle("Response policy (CALDERA adversaries)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    legend(fig, groups=("ours", "inhouse"), y=-0.1)
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    save(fig, "fig4_response.png")


# ------------------------------------------------------------------------------------------------ 0. scorecard
def scorecard():
    # docs/comparison_scorecard.md summary table
    topics = ["Threat detection", "System latency (laptop)", "Explainability", "Response policy"]
    w, t, l = [9, 2, 8, 7], [1, 0, 2, 4], [12, 4, 8, 1]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for i, topic in enumerate(topics[::-1]):
        j = len(topics) - 1 - i
        left = 0
        for v, color, hatch, lab in ((w[j], GROUP["ours"][0], None, "wins"), (t[j], "#b8b7b1", None, "ties"),
                                     (l[j], GROUP["inhouse"][0], "\\\\\\\\", "losses")):
            if v:
                ax.barh(i, v, left=left, height=0.6, color=color, hatch=hatch, edgecolor=SURFACE, linewidth=2)
                ax.text(left + v / 2, i, f"{v}", ha="center", va="center", fontsize=9, color=INK,
                        fontweight="bold", bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.2))
            left += v
        ax.text(left + 0.3, i, f"{w[j]} / {t[j]} / {l[j]}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(topics)), topics[::-1])
    ax.set_xlabel("comparisons")
    ax.set_xlim(0, 27)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Wins / ties / losses per topic", loc="left")
    fig.legend(handles=[Patch(facecolor=GROUP["ours"][0], label="XTGAIA better"),
                        Patch(facecolor="#b8b7b1", label="tie"),
                        Patch(facecolor=GROUP["inhouse"][0], hatch="\\\\\\\\", edgecolor=SURFACE,
                              label="comparator better")],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "fig0_scorecard.png")


if __name__ == "__main__":
    scorecard()
    detection_unsw()
    detection_cicids()
    design_check()
    latency()
    xai()
    response()
