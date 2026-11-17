"""
XTGAIA latency and end-to-end timing benchmark (paper Sec. IV-D), for the hardware hand-off.

One command measures every timing the paper reports, on whatever machine it runs on, and writes a
self-contained result folder (JSON + Markdown) that includes the hardware it ran on:

  * per-stage latency, the four AIR [5] stage boundaries: acquisition, analysis, decision, execution
  * MTTD (processing only and including the Δinf window), MTTR, end-to-end closed loop (ingest -> G_t+1)
  * explanation latency (Tiers 1+2, Tier 3, the parallel bundle X_t) against the 200 ms bound
  * throughput (events/s and flows/s over the replay, start-up excluded)
  * optional TGN micro-benchmark: one memory update + scoring at micro-batch sizes 1 / 16 / 64 / 128,
    isolated from the SOC services (pure model cost)

Usage (inside the xtgaia-core container, SOC stack up for the full closed loop):

  python scripts/latency_bench.py                         # Friday replay, 300k events, 1 run
  python scripts/latency_bench.py --limit 100000 --repeat 3
  python scripts/latency_bench.py --micro-only            # TGN micro-benchmark only (no services needed)
  python scripts/latency_bench.py --no-neo4j --no-soar    # model + graph only (no Neo4j, Shuffle, TheHive)

Results: runs/latency/<YYYYmmdd-HHMMSS>/{latency.json, latency.md}. Nothing is sent anywhere except the
SOC services configured in .env (Shuffle / TheHive / Neo4j), exactly as in a normal run.
"""
import argparse
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np  # noqa: E402

from src.role3_decision.config import resolve_path  # noqa: E402  (loads .env)

DEFAULT_INPUT = "data/raw/cicids2017_improved/friday.csv"

# AIR [5] reported stage times (ms) and the [2] MTTR; comparator columns in the Markdown table.
AIR_MS = {"acquisition_ms": 8.1, "analysis_ms": 15.2, "decision_ms": 23.5, "execution_ms": 11.5,
          "end_to_end_ms": 58.3}
KULKARNI_MTTR_MS = 1800.0
AIR_THROUGHPUT = 18540.0

STAGES = [
    ("acquisition_ms", "Acquisition (parse, STIX, SKG update; per event)"),
    ("analysis_ms", "Analysis (TGN update + Algorithm 2 risk; per micro-batch)"),
    ("decision_ms", "Decision (state vector + zone Double DQN)"),
    ("execution_ms", "Execution (Shuffle -> enforcement -> G_t+1)"),
    ("end_to_end_ms", "End-to-end closed loop (ingest -> G_t+1)"),
    ("mttd_processing_ms", "MTTD, processing only (ingest -> TGN detection)"),
    ("mttd_window_ms", "MTTD incl. Δinf window"),
    ("mttr_ms", "MTTR (detection -> SOAR enforcement)"),
    ("xai_tiers12_ms", "XAI Tiers 1+2 (PGExplainer + attention rollout)"),
    ("xai_tier3_ms", "XAI Tier 3 (reward decomposition)"),
    ("xai_total_ms", "Explanation bundle X_t (parallel tiers)"),
    ("audit_ms", "TheHive audit case (Algorithm 4 Phase 2, after enforcement)"),
]


# ---------------------------------------------------------------------------------------------- hardware
def _read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def hardware_info():
    import torch
    info = {"platform": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
            "logical_cpus": os.cpu_count(), "torch_threads": torch.get_num_threads(),
            "in_container": os.path.exists("/.dockerenv")}
    cpu = next((l.split(":", 1)[1].strip() for l in _read("/proc/cpuinfo").splitlines()
                if l.startswith("model name")), None)
    info["cpu_model"] = cpu or platform.processor() or None
    mem = next((l for l in _read("/proc/meminfo").splitlines() if l.startswith("MemTotal")), None)
    if mem:
        info["ram_gb"] = round(int(mem.split()[1]) / 1024 / 1024, 1)
    else:
        try:
            import psutil
            info["ram_gb"] = round(psutil.virtual_memory().total / 1024 ** 3, 1)
        except ImportError:
            info["ram_gb"] = None
    info["cuda"] = torch.cuda.is_available()
    info["gpu"] = torch.cuda.get_device_name(0) if info["cuda"] else None
    info["git_commit"] = os.getenv("XTGAIA_GIT_COMMIT")
    try:
        info["git_commit"] = info["git_commit"] or subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                                            text=True, timeout=5).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        info["git_commit"] = None
    return info


def service_status():
    import requests
    out = {}
    for name, url in [("thehive", os.getenv("THEHIVE_URL", "").rstrip("/") + "/api/status"),
                      ("shuffle", os.getenv("SHUFFLE_BASE_URL", "").rstrip("/") + "/api/v1/health")]:
        if url.startswith("/"):
            out[name] = "not configured"
            continue
        try:
            out[name] = requests.get(url, timeout=5).status_code
        except Exception as exc:  # noqa: BLE001 - report, never fail
            out[name] = f"unreachable ({type(exc).__name__})"
    return out


# ---------------------------------------------------------------------------------------------- stats
def summarise(values):
    if not values:
        return None
    a = np.asarray(values, dtype=float)
    return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)), "std": float(a.std()),
            "max": float(a.max())}


# ---------------------------------------------------------------------------------------------- full pipeline
def run_pipeline(args, run_idx):
    import main as xt
    from src.pipeline import XTGAIAPipeline

    ns = argparse.Namespace(skg_window_ms=args.skg_window_ms, mode=args.mode, checkpoint=args.checkpoint)
    cfg = xt._config(ns)
    if args.burn_in is not None:
        cfg["role2"]["burn_in_events"] = int(args.burn_in)
    if args.no_soar:                       # model + graph only: hide the SOC endpoints for this process
        for k in ("SHUFFLE_EXECUTION_WEBHOOK_URL", "SHUFFLE_WEBHOOK_URL", "SHUFFLE_ROLLBACK_WEBHOOK_URL",
                  "THEHIVE_URL", "THEHIVE_API_KEY"):
            os.environ.pop(k, None)
    t_start = time.perf_counter()
    pipe = XTGAIAPipeline(cfg, neo4j_driver=None if args.no_neo4j else "env")
    startup_s = time.perf_counter() - t_start
    if pipe.graph.skg.driver is not None:
        pipe.graph.skg.clear_latent_states()       # same starting point as the demo (checkpoint memory)
    source = xt.make_source("cicids", args.input, args.start, args.end, args.limit)
    print(f"[run {run_idx}] replaying {args.limit or 'all'} events from {', '.join(args.input)} "
          f"(start-up {startup_s:.1f}s) ...", flush=True)
    t0 = time.perf_counter()
    report = pipe.run(source, progress_every=args.progress)
    wall = time.perf_counter() - t0
    pipe._xai_pool.shutdown(wait=False)
    lat = {k: summarise(v) for k, v in report.latency.items() if v}
    return {
        "run": run_idx, "startup_s": startup_s, "wall_s": wall, "events": report.events, "edges": report.edges,
        "events_per_s": report.events / wall if wall else None, "flows_per_s": report.edges / wall if wall else None,
        "dispatches": len(report.dispatches),
        "shuffle_ok": sum(1 for d in report.dispatches if d["shuffle"]),
        "thehive_ok": sum(1 for d in report.dispatches if d["thehive"]),
        "batch_ms": summarise(report.batch_ms), "latency_ms": lat,
        "raw_latency_ms": {k: list(map(float, v)) for k, v in report.latency.items()},
        "detection": report.detection_metrics(), "neo4j": pipe.graph.skg.driver is not None,
        "threshold": pipe.engine.threshold, "checkpoint_fingerprint": getattr(pipe.engine, "fingerprint", None),
    }


# ---------------------------------------------------------------------------------------------- micro-benchmark
def micro_benchmark(args):
    """TGN memory update + scoring per micro-batch, on real flows, with no Role 1 / Role 3 work."""
    import torch
    import main as xt
    from src.pipeline import XTGAIAPipeline

    ns = argparse.Namespace(skg_window_ms=args.skg_window_ms, mode=args.mode, checkpoint=args.checkpoint)
    pipe = XTGAIAPipeline(xt._config(ns), neo4j_driver=None)
    edges = []
    for raw in xt.make_source("cicids", args.input, args.start, args.end, 20000):
        for e in pipe.graph.ingest_all(raw):
            if e.get("features") is not None:
                edges.append(e)
        if len(edges) >= 128 * (args.micro_iters + 5):
            break
    out = {}
    with torch.no_grad():
        for size in (1, 16, 64, 128):
            times = []
            for i in range(args.micro_iters + 3):
                batch = edges[(i * size) % max(len(edges) - size, 1):][:size]
                t = time.perf_counter()
                pipe.engine.observe_batch(batch)
                times.append((time.perf_counter() - t) * 1000.0)
            out[str(size)] = summarise(times[3:])        # first 3 = warm-up
            print(f"[micro] batch {size:>3}: mean {out[str(size)]['mean']:.2f} ms, p95 {out[str(size)]['p95']:.2f} ms "
                  f"({out[str(size)]['mean'] / size * 1000:.0f} us/flow)", flush=True)
    pipe._xai_pool.shutdown(wait=False)
    return out


# ---------------------------------------------------------------------------------------------- report
def pooled(runs, key):
    vals = [v for r in runs for v in r["raw_latency_ms"].get(key, [])]
    return summarise(vals)


def write_markdown(path, result):
    hw, runs = result["hardware"], result["runs"]
    L = ["# XTGAIA latency benchmark", "",
         f"Generated {result['generated']} | git {hw.get('git_commit')} | checkpoint "
         f"{runs[0]['checkpoint_fingerprint'] if runs else '-'}", "",
         "## Hardware", "",
         f"- CPU: {hw.get('cpu_model')} ({hw.get('logical_cpus')} logical CPUs, torch threads {hw.get('torch_threads')})",
         f"- RAM: {hw.get('ram_gb')} GB | GPU: {hw.get('gpu') or 'none (CPU inference)'}",
         f"- {hw.get('platform')} | Python {hw.get('python')} | torch {hw.get('torch')} | container: {hw.get('in_container')}",
         f"- Services: {result.get('services')} | Neo4j: {runs[0]['neo4j'] if runs else '-'}", ""]
    if runs:
        L += ["## Replay", "",
              f"Input {', '.join(result['args']['input'])}, {result['args']['limit'] or 'all'} events per run, "
              f"{len(runs)} run(s). Times are processing-only unless stated; pooled over runs.", "",
              "| Run | Events | Wall (s) | Events/s | Flows/s | Dispatches | Shuffle ok | TheHive ok |",
              "|---|---|---|---|---|---|---|---|"]
        for r in runs:
            L.append(f"| {r['run']} | {r['events']:,} | {r['wall_s']:.1f} | {r['events_per_s']:,.0f} | "
                     f"{r['flows_per_s']:,.0f} | {r['dispatches']} | {r['shuffle_ok']} | {r['thehive_ok']} |")
        L += ["", f"Throughput comparator: AIR [5] {AIR_THROUGHPUT:,.0f} events/s (4 × Dell R760 + A100).", "",
              "## Latency (ms)", "",
              "| Stage | n | Mean | p50 | p95 | p99 | Std | Comparator |",
              "|---|---|---|---|---|---|---|---|"]
        for key, name in STAGES:
            s = pooled(runs, key)
            if not s:
                continue
            comp = (f"AIR {AIR_MS[key]} ms" if key in AIR_MS else
                    f"[2] {KULKARNI_MTTR_MS:,.0f} ms" if key == "mttr_ms" else
                    "bound 200 ms" if key == "xai_total_ms" else "–")
            L.append(f"| {name} | {s['n']:,} | {s['mean']:.2f} | {s['p50']:.2f} | {s['p95']:.2f} | {s['p99']:.2f} | "
                     f"{s['std']:.2f} | {comp} |")
        L += ["", "No dispatch means no execution / MTTR / end-to-end samples: the replay window must contain attack "
              "traffic after the burn-in (in the corrected release's clock: Botnet from 13:00, PortScan ~16:30-18:30, DDoS ~18:00-19:30)."]
    if result.get("micro"):
        L += ["", "## TGN micro-benchmark (memory update + scoring, no services)", "",
              "| Micro-batch | Mean (ms) | p95 (ms) | Per flow (µs) |", "|---|---|---|---|"]
        for size, s in result["micro"].items():
            L.append(f"| {size} | {s['mean']:.2f} | {s['p95']:.2f} | {s['mean'] / int(size) * 1000:.0f} |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", nargs="+", default=[DEFAULT_INPUT])
    ap.add_argument("--start", default=None, help="dataset time (UTC) to start; with the default Friday input "
                    "it defaults to 2017-07-07T13:30 (300k events then cover Botnet C2 and the PortScan)")
    ap.add_argument("--end", default=None)
    ap.add_argument("--limit", type=int, default=300000, help="events per run (0 = all)")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--burn-in", type=int, default=None, help="override role2.burn_in_events")
    ap.add_argument("--no-neo4j", action="store_true")
    ap.add_argument("--no-soar", action="store_true", help="skip Shuffle / TheHive (model + graph timing only)")
    ap.add_argument("--micro", action="store_true", help="also run the TGN micro-benchmark")
    ap.add_argument("--micro-only", action="store_true")
    ap.add_argument("--micro-iters", type=int, default=200)
    ap.add_argument("--skg-window-ms", type=float, default=None)
    ap.add_argument("--mode", choices=["active", "passive"], default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--progress", type=int, default=100000)
    ap.add_argument("--out", default=None, help="output folder (default runs/latency/<timestamp>)")
    args = ap.parse_args(argv)
    args.limit = args.limit or None
    if args.start is None and args.input == [DEFAULT_INPUT]:
        args.start = "2017-07-07T13:30"

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = str(resolve_path(args.out or f"runs/latency/{stamp}"))
    os.makedirs(out_dir, exist_ok=True)
    result = {"generated": dt.datetime.now(dt.timezone.utc).isoformat(), "args": vars(args),
              "hardware": hardware_info(), "services": None if args.no_soar else service_status(), "runs": []}
    print(f"[*] hardware: {result['hardware']['cpu_model']} | {result['hardware']['logical_cpus']} CPUs | "
          f"{result['hardware']['ram_gb']} GB | GPU {result['hardware']['gpu']} | services {result['services']}")
    if not args.micro_only:
        for i in range(1, args.repeat + 1):
            result["runs"].append(run_pipeline(args, i))
    if args.micro or args.micro_only:
        result["micro"] = micro_benchmark(args)

    json_path, md_path = os.path.join(out_dir, "latency.json"), os.path.join(out_dir, "latency.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, default=str)
    write_markdown(md_path, result)
    print("\n" + open(md_path, encoding="utf-8").read())
    print(f"[✓] {os.path.relpath(json_path, REPO)}\n[✓] {os.path.relpath(md_path, REPO)}")


if __name__ == "__main__":
    main()
