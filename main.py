"""
XTGAIA Central Pipeline Orchestrator
Connects: Person 1 (Graph/Risk) -> Person 2 (TGN/XAI) -> Person 3 (Double DQN/SOAR)

  python main.py demo                       # full-system demo on the held-out CICIDS2017 Friday
  python main.py run --source cicids --input data/raw/cicids2017/TrafficLabelling/Friday-*.csv
  python main.py run --source wazuh --input /var/ossec/logs/alerts/alerts.json --follow --serve-rollback
  python main.py run --source json  --input data/samples/mordor_sample.json

Prerequisites: a trained TGN (src/role2_ai/preprocess_cicids.py + train_tgn.py) and
a trained policy (data/models/dqn_policy.pt); Shuffle/TheHive/Neo4j settings in .env.
"""
import argparse
import datetime as dt
import glob
import json
import logging
import os
import statistics
import sys
import time

from src.role3_decision.config import REPO_ROOT, resolve_path  # loads .env first

# Corrected CICIDS2017 (microsecond timestamps -> the paper's 50 ms window applies as is); the original
# minute-resolution TrafficLabelling CSVs are the fallback and need a wider --skg-window-ms.
DEFAULT_DEMO_FILES = ["data/raw/cicids2017_improved/friday.csv"]
LEGACY_DEMO_FILES = [
    "data/raw/cicids2017/TrafficLabelling/Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "data/raw/cicids2017/TrafficLabelling/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "data/raw/cicids2017/TrafficLabelling/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]


def _paths(patterns):
    out = []
    for p in patterns:
        hits = sorted(glob.glob(str(resolve_path(p))))
        out.extend(hits or [str(resolve_path(p))])
    return out


def _epoch(value):
    if value is None:
        return None
    return dt.datetime.fromisoformat(value).replace(tzinfo=dt.timezone.utc).timestamp()


def make_source(kind, inputs, start=None, end=None, limit=None, follow=False, via_wazuh=False):
    from src.role1_graph.flow_sources import CICIDSFlowSource, JSONEventSource, WazuhAlertSource
    from src.role1_graph.telemetry import CloudTrailSource, MordorSource, ZeekConnSource
    paths = _paths(inputs)
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        sys.exit(f"[!] input not found: {', '.join(missing)}")
    if kind == "cicids":
        return CICIDSFlowSource(paths, start=_epoch(start), end=_epoch(end), limit=limit)
    if kind == "wazuh":
        return WazuhAlertSource(paths[0], follow=follow)
    if kind == "mordor":
        if via_wazuh:                       # paper Stage 1: Mordor -> Wazuh manager -> archives.json -> SKG
            from src.role1_graph.wazuh_mordor import WazuhMordorSource
            return WazuhMordorSource(paths)
        return MordorSource(paths)
    if kind == "zeek":
        return ZeekConnSource(paths[0])
    if kind == "cloudtrail":
        return CloudTrailSource(paths)
    return JSONEventSource(paths[0])


def print_report(pipe, report, wall_s):
    m = report.detection_metrics()
    print("\n" + "=" * 78)
    print(" XTGAIA RUN REPORT")
    print("=" * 78)
    g = pipe.graph.stats
    print(f" Events read {report.events:,} | ingested {g['ingested']:,} | blocked by containment {g['blocked']:,} "
          f"| malformed {g['dropped_malformed']:,}")
    print(f" Neo4j edges synced {g['neo4j_synced']:,} (errors {g['neo4j_errors']}) | "
          f"live graph now {pipe.graph.skg.graph.number_of_nodes():,} nodes / {pipe.graph.skg.graph.number_of_edges():,} edges")
    if report.batch_ms:
        per_event_us = sum(report.batch_ms) * 1000.0 / max(report.edges, 1)
        p95 = sorted(report.batch_ms)[int(0.95 * (len(report.batch_ms) - 1))]
        print(f" Wall time {wall_s:,.1f}s ({report.events / max(wall_s, 1e-9):,.0f} events/s) | Role2+3 per micro-batch "
              f"mean {statistics.mean(report.batch_ms):.1f} ms, p95 {p95:.1f} ms ({per_event_us:.0f} us/event)")
    if report.burn_in:
        b = report.burn_in
        print(f"\n Burn-in baseline (no response, excluded from evaluation): {b['events']:,} events | trained ω "
              f"{b['trained_threshold']:.4f} -> alert threshold {b['threshold']:.4f} ({b['alert_rate']:.0%} baseline "
              f"alert rate) | attack flows that fell inside the window: {b['attack_flows_inside_window']}")
        print(f"   per-host baselines: {b.get('host_baselines', 0)} hosts; raised above global: "
              f"{b.get('host_thresholds_above_global', {})}")
    print(f"\n Role 2 detection (flow level, threshold {pipe.engine.threshold:.4f}) vs dataset labels "
          f"[{report.labelled:,} labelled flows]")
    print(f"   precision {m['precision']:.4f} | recall {m['recall']:.4f} | F1 {m['f1']:.4f} | FPR {m['fpr']:.4f} "
          f"| TP {report.tp:,} FP {report.fp:,} TN {report.tn:,} FN {report.fn:,}")
    for label, n in report.attack_flows_by_label.most_common():
        print(f"   - {label:<28} detected {report.detected_by_label[label]:>7,} / {n:<7,} "
              f"({report.detected_by_label[label] / n:.1%})")
    print(f"\n Role 3 policy decisions on flagged hosts: {dict(report.policy_actions)}")
    print(f"   suppressed (no new dispatch): {dict(report.suppressed)}")
    print(f"\n Closed loop (Algorithm 4): audit-window commits {dict(report.committed)} (T_audit "
          f"{pipe.audit_window_s:.0f}s) | Neo4j soft-deleted edges {g.get('neo4j_soft_deleted', 0)}, reactivated "
          f"{g.get('neo4j_reactivated', 0)} | Algorithm 1 latent states saved {g.get('latent_saved', 0):,}, "
          f"recovered {report.latent_recovered:,}")
    on_attack = [d for d in report.dispatches if d["ground_truth_label"] not in ("", "BENIGN")]
    print(f"\n Response evaluation (dataset labels, evaluation only): {len(on_attack)}/{len(report.dispatches)} "
          f"dispatches triggered by attack traffic")
    for label, first in sorted(report.first_seen_by_label.items(), key=lambda kv: kv[1]):
        hits = [d for d in report.dispatches if d["ground_truth_label"] == label]
        start = dt.datetime.fromtimestamp(first, dt.timezone.utc).strftime("%H:%M")
        if hits:
            delay = (hits[0]["time"] - first) / 60.0
            print(f"   - {label:<10} first seen {start}: first response {hits[0]['action']} on {hits[0]['host']} "
                  f"after {delay:.0f} min (dataset clock, minute resolution); {len(hits)} dispatch(es)")
        else:
            print(f"   - {label:<10} first seen {start}: no autonomous response")
    print(f"\n Dispatches ({len(report.dispatches)}):")
    for d in report.dispatches:
        when = dt.datetime.fromtimestamp(d["time"], dt.timezone.utc).strftime("%H:%M")
        print(f"   {when} {d['action']:<14} {d['host']:<15} <- {d['counterpart']:<15} p={d['p_t']:.3f} "
              f"risk={d['risk']:<7.3f} auth_fails={int(d['phi']['auth_fails']):<5} conn={int(d['phi']['conn_count']):<4} "
              f"shuffle={'ok' if d['shuffle'] else 'FAIL'} thehive={d['case_id'] if d['thehive'] else 'FAIL'} "
              f"severed={d['edges_severed']} truth={d['ground_truth_label']}")
    lat = report.latency_summary()
    if lat:
        print("\n Latency (ms, mean / p95) - processing-only; MTTD also window-inclusive (Δinf):")
        for k in ("acquisition_ms", "analysis_ms", "decision_ms", "xai_tiers12_ms", "xai_tier3_ms", "xai_total_ms",
                  "execution_ms", "audit_ms", "mttd_processing_ms", "mttd_window_ms", "mttr_ms", "end_to_end_ms"):
            if k in lat:
                print(f"   {k:<20} {lat[k]['mean']:9.2f} / {lat[k]['p95']:9.2f}  (n={lat[k]['n']})")
    stab = report.xai_stability()
    if any(v is not None for v in stab["cosine_mean"].values()):
        print(" XAI stability (mean cosine, same attack class): " + ", ".join(
            f"{t} {v:.3f}" for t, v in stab["cosine_mean"].items() if v is not None))
    g = pipe.graph.stats
    print(f"\n SKG edge types: {g.get('typed_edges', {})} | nodes evicted from V_act: {g.get('evicted_nodes', 0):,}")
    if report.online_updates:
        acc = sum(1 for u in report.online_updates if u.get("accepted"))
        print(f" Online policy updates: {acc}/{len(report.online_updates)} accepted")
    print("=" * 78)


def save_report(pipe, report, path, extra=None):
    out = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(),
        "events": report.events, "edges": report.edges, "graph_stats": pipe.graph.stats,
        "threshold": pipe.engine.threshold, "tgn_training_test_metrics": pipe.engine.test_metrics,
        "detection_metrics": report.detection_metrics(),
        "confusion": {"tp": report.tp, "fp": report.fp, "tn": report.tn, "fn": report.fn},
        "attack_flows_by_label": dict(report.attack_flows_by_label),
        "detected_by_label": dict(report.detected_by_label),
        "policy_actions": dict(report.policy_actions), "suppressed": dict(report.suppressed),
        "dispatches": report.dispatches, "rollbacks_seen": report.rollbacks_seen, "burn_in": report.burn_in,
        "audit_commits": dict(report.committed), "latent_recovered": report.latent_recovered,
        "batch_ms_mean": statistics.mean(report.batch_ms) if report.batch_ms else None,
        "latency_ms": report.latency_summary(), "online_updates": report.online_updates,
        "xai_stability": report.xai_stability(),
        "mode": pipe.dispatcher.mode, "zones": sorted(pipe.dispatcher.zone_models),
        "ablation": {"risk_assessor": pipe.use_risk_assessor, "dual_timescale": pipe.dual_timescale,
                     "closed_loop": pipe.closed_loop},
    }
    if extra:
        out.update(extra)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"[✓] Report written to {os.path.relpath(path, REPO_ROOT)}")


def _config(args, default_window_ms=None):
    from src.pipeline import load_pipeline_config
    cfg = load_pipeline_config()
    window = args.skg_window_ms if args.skg_window_ms is not None else default_window_ms
    if window is not None:
        cfg["role1"]["skg_window_ms"] = float(window)
    if getattr(args, "mode", None):
        cfg.setdefault("role3", {})["mode"] = args.mode
    if getattr(args, "checkpoint", None):
        cfg.setdefault("role2", {})["checkpoint"] = args.checkpoint
    return cfg


def cmd_run(args):
    from src.pipeline import XTGAIAPipeline
    pipe = XTGAIAPipeline(_config(args), neo4j_driver=None if args.no_neo4j else "env")
    if args.serve_rollback:
        pipe.serve_rollback_api()
        print(f"[*] Rollback API on :{os.getenv('ROLLBACK_API_PORT', '8008')}/api/v1/feedback/rollback")
    t0 = time.time()
    report = pipe.run(make_source(args.source, args.input, args.start, args.end, args.limit, args.follow,
                                  getattr(args, "via_wazuh", False)),
                      progress_every=args.progress)
    print_report(pipe, report, time.time() - t0)
    save_report(pipe, report, str(resolve_path(args.report)))


def _service_banner():
    import requests
    rows = []
    for name, url in [("TheHive", os.getenv("THEHIVE_URL", "").rstrip("/") + "/api/status"),
                      ("Shuffle", os.getenv("SHUFFLE_BASE_URL", "").rstrip("/") + "/api/v1/health")]:
        try:
            code = requests.get(url, timeout=5).status_code
        except Exception as exc:
            code = f"unreachable ({type(exc).__name__})"
        rows.append(f"{name} {code}")
    return " | ".join(rows)


def cmd_demo(args):
    import numpy as np
    import requests
    from src.pipeline import XTGAIAPipeline

    print("=" * 78)
    print(" XTGAIA FULL-SYSTEM DEMO  -  real CICIDS2017 traffic, held-out day (Friday 7 July 2017)")
    print("=" * 78)
    # The corrected CICIDS2017 has microsecond timestamps, so the paper's Δc = 50 ms is used as is; only the
    # original minute-resolution TrafficLabelling CSVs need the window widened (5 minutes).
    files = _paths(args.input or (DEFAULT_DEMO_FILES if os.path.exists(resolve_path(DEFAULT_DEMO_FILES[0]))
                                  else LEGACY_DEMO_FILES))
    legacy = any("TrafficLabelling" in f for f in files)
    pipe = XTGAIAPipeline(_config(args, default_window_ms=300000 if legacy else 50),
                          neo4j_driver=None if args.no_neo4j else "env")
    tm = pipe.engine.test_metrics
    print(f"[Role 2] TGN checkpoint: threshold {pipe.engine.threshold:.4f}; training test split "
          f"P {tm.get('precision', 0):.3f} R {tm.get('recall', 0):.3f} F1 {tm.get('f1', 0):.3f} AUC {tm.get('roc_auc', 0):.3f}")
    print(f"[Role 1] SKG window {pipe.cfg['role1']['skg_window_ms']} ms | crown jewels {pipe.inventory.crown_jewels} "
          f"| Neo4j {'on' if pipe.graph.skg.driver else 'off'}")
    print(f"[Role 3] policy {'zone policies ' + str(sorted(pipe.dispatcher.zone_models)) if pipe.dispatcher.zone_models else resolve_path('data/models/dqn_policy.pt').name} | services: {_service_banner()}")
    feedback_path = str(resolve_path("data/processed/demo_feedback_buffer.npz"))
    if os.path.exists(feedback_path):
        os.remove(feedback_path)
    pipe.serve_rollback_api(feedback_path=feedback_path)
    port = int(os.getenv("ROLLBACK_API_PORT", "8008"))
    print(f"[Role 3] analyst rollback API listening on :{port}\n")

    if pipe.graph.skg.driver is not None:              # evaluation replay starts from checkpoint memory
        print(f"[Algorithm 1] cleared {pipe.graph.skg.clear_latent_states():,} persisted latent states "
              f"of model {pipe.graph.skg.latent_model} before replay")
    print(f"[*] Replaying {len(files)} held-out files through Role 1 -> Role 2 -> Role 3 ...")
    t0 = time.time()
    report = pipe.run(make_source("cicids", files, args.start, args.end, args.limit), progress_every=args.progress)
    wall = time.time() - t0
    print_report(pipe, report, wall)

    extra = {"rollback": None, "neo4j": None, "thehive_case_check": None}
    # ---- TheHive: read one case back through the API
    if report.dispatches and report.dispatches[0]["thehive"]:
        cid = report.dispatches[0]["case_id"]
        r = requests.get(f"{os.environ['THEHIVE_URL'].rstrip('/')}/api/v1/case/{cid}",
                         headers={"Authorization": f"Bearer {os.environ['THEHIVE_API_KEY']}"}, timeout=10)
        case = r.json() if r.ok else {}
        desc = case.get("description", "")
        extra["thehive_case_check"] = {"status": r.status_code, "title": case.get("title"),
                                       "severity": case.get("severity"),
                                       "has_role1_evidence": "Detection context (Role 1)" in desc,
                                       "has_role2_xai": "Detection XAI (Role 2" in desc,
                                       "has_role3_xai": "Decision XAI (Role 3" in desc}
        print(f"\n[TheHive] case {cid}: HTTP {r.status_code} '{case.get('title')}' severity {case.get('severity')} | "
              f"evidence sections R1/R2/R3: {extra['thehive_case_check']['has_role1_evidence']}/"
              f"{extra['thehive_case_check']['has_role2_xai']}/{extra['thehive_case_check']['has_role3_xai']}")

    # ---- Analyst rollback over HTTP (prefer a genuine false positive)
    containing = [d for d in report.dispatches if d["action"] in ("Quarantine", "PortIsolation")]
    pool = [d for d in containing if d["ground_truth_label"] == "BENIGN"] or containing or report.dispatches
    if pool:
        d = pool[0]
        g = pipe.graph.skg.graph
        before_edges = g.number_of_edges()
        buf_before = len(pipe.rollback_handler.buffer)
        r = requests.post(f"http://127.0.0.1:{port}/api/v1/feedback/rollback",
                          json={"case_id": d["case_id"], "analyst_comment": "XTGAIA demo: analyst override"},
                          headers={"X-XTGAIA-Token": os.environ.get("ROLLBACK_WEBHOOK_TOKEN", "")}, timeout=30)
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        lifted = pipe.refresh_enforcement()
        buf = pipe.rollback_handler.buffer
        last = buf.buffer[-1] if len(buf) > buf_before else None
        extra["rollback"] = {
            "case_id": d["case_id"], "host": d["host"], "action": d["action"],
            "ground_truth_of_trigger": d["ground_truth_label"], "http": r.status_code, "response": body,
            "edges_before": before_edges, "edges_after": g.number_of_edges(),
            "enforcement_lifted": [h for h, _ in lifted],
            "penalty_reward": None if last is None else float(last[2]),
            "feedback_persisted": os.path.exists(feedback_path),
        }
        print(f"\n[Rollback] analyst rolled back case {d['case_id']} ({d['action']} on {d['host']}, trigger truth "
              f"{d['ground_truth_label']}): HTTP {r.status_code} status={body.get('status')} "
              f"shuffle_reverted={body.get('soar_reverted')} edges_restored={body.get('edges_restored')} | "
              f"enforcement lifted for {extra['rollback']['enforcement_lifted']} | replay tuple reward="
              f"{extra['rollback']['penalty_reward']} | persisted={extra['rollback']['feedback_persisted']}")
        r2 = requests.post(f"http://127.0.0.1:{port}/api/v1/feedback/rollback", json={"case_id": d["case_id"]},
                           headers={"X-XTGAIA-Token": os.environ.get("ROLLBACK_WEBHOOK_TOKEN", "")}, timeout=10)
        r3 = requests.post(f"http://127.0.0.1:{port}/api/v1/feedback/rollback", json={"case_id": d["case_id"]},
                           timeout=10)
        print(f"[Rollback] repeat rollback -> HTTP {r2.status_code}; request without token -> HTTP {r3.status_code}")
        extra["rollback"].update({"repeat_http": r2.status_code, "no_token_http": r3.status_code})
        # Algorithm 4 audit window: approve endpoint on a different case
        other = next((x for x in report.dispatches if x["case_id"] != d["case_id"]), None)
        if other is not None:
            ra = requests.post(f"http://127.0.0.1:{port}/api/v1/feedback/approve", json={"case_id": other["case_id"]},
                               headers={"X-XTGAIA-Token": os.environ.get("ROLLBACK_WEBHOOK_TOKEN", "")}, timeout=10)
            rec = pipe.dispatcher.ledger.get(other["case_id"])
            print(f"[Audit] approve case {other['case_id']} -> HTTP {ra.status_code} "
                  f"({ra.json().get('status') or ra.json().get('detail')}); committed={rec.committed if rec else None}")
            extra["approve"] = {"case_id": other["case_id"], "http": ra.status_code, "body": ra.json()}

    # ---- Neo4j persistent graph
    if pipe.graph.skg.driver is not None:
        with pipe.graph.skg.driver.session() as s:
            hosts = s.run("MATCH (h:Host) RETURN count(h) AS n").single()["n"]
            rels = s.run("MATCH ()-[r:COMMUNICATED]->() RETURN count(r) AS n").single()["n"]
            web = s.run("MATCH (a:Host)-[r:COMMUNICATED]->(b:Host {ip: $ip}) RETURN count(DISTINCT a) AS n",
                        ip="192.168.10.50").single()["n"]
            inactive = s.run("MATCH ()-[r:COMMUNICATED]->() WHERE r.active = false RETURN count(r) AS n").single()["n"]
            states = {r["state"]: r["n"] for r in s.run(
                "MATCH (h:Host) WHERE h.state IS NOT NULL RETURN h.state AS state, count(h) AS n")}
            latent = s.run("MATCH (h:Host) WHERE h.latent_state IS NOT NULL "
                           "RETURN count(h) AS n, max(size(h.latent_state)) AS bytes").single()
        extra["neo4j"] = {"hosts": hosts, "relationships": rels, "peers_of_web_server": web,
                          "inactive_edges": inactive, "host_states": states,
                          "latent_nodes": latent["n"], "latent_bytes": latent["bytes"]}
        print(f"[Neo4j] persistent SKG: {hosts:,} hosts, {rels:,} COMMUNICATED edges; "
              f"{web:,} distinct peers of crown jewel 192.168.10.50")
        print(f"[Neo4j] Algorithm 4 soft delete: {inactive} edges flagged inactive, host states {states} | "
              f"Algorithm 1: {latent['n']:,} hosts with persisted latent state ({latent['bytes']} bytes each)")
        # Algorithm 1 recovery check: a fresh engine (as after a restart) hydrates from Neo4j
        from src.role2_ai.inference import TGNInferenceEngine
        host = "192.168.10.50"
        fresh = TGNInferenceEngine(str(resolve_path(pipe.cfg["role2"]["checkpoint"])))
        p_cold, _ = fresh.score(host)
        restored = fresh.hydrate(pipe.graph.fetch_latent_states([host]))
        p_live, _ = pipe.engine.score(host)
        p_restored, _ = fresh.score(host)
        extra["latent_recovery_check"] = {"host": host, "restored": restored, "p_checkpoint_only": p_cold,
                                          "p_live": p_live, "p_after_recovery": p_restored}
        print(f"[Algorithm 1] fresh engine for {host}: p from checkpoint only {p_cold:.4f} -> after Neo4j recovery "
              f"{p_restored:.4f} (live engine {p_live:.4f})")

    save_report(pipe, report, str(resolve_path(args.report)), extra)


def main(argv=None):
    logging.basicConfig(level=os.getenv("XTGAIA_LOG_LEVEL", "WARNING"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    for name in ("run", "demo"):
        p = sub.add_parser(name)
        p.add_argument("--input", nargs="+", default=None)
        p.add_argument("--start", help="ISO datetime (dataset time, UTC) to start replay, e.g. 2017-07-07T09:00")
        p.add_argument("--end", help="ISO datetime to stop replay")
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--progress", type=int, default=100000)
        p.add_argument("--no-neo4j", action="store_true")
        p.add_argument("--report", default=f"data/processed/{name}_report.json")
        p.add_argument("--skg-window-ms", type=float, default=None,
                       help="Δc override (paper: 50 ms; use 300000 for minute-resolution datasets)")
        p.add_argument("--mode", choices=["active", "passive"], default=None, help="deployment regime")
        p.add_argument("--checkpoint", default=None, help="TGN checkpoint (default: role2.checkpoint in pipeline.yaml)")
        if name == "run":
            p.add_argument("--source", choices=["cicids", "wazuh", "json", "mordor", "zeek", "cloudtrail"],
                           default="cicids")
            p.add_argument("--follow", action="store_true", help="tail a live Wazuh alerts.json")
            p.add_argument("--via-wazuh", action="store_true",
                           help="mordor: ingest through the Wazuh manager (docker compose up -d wazuh-master)")
            p.add_argument("--serve-rollback", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "run":
        if not args.input:
            sys.exit("[!] run needs --input")
        cmd_run(args)
    else:
        if args.cmd is None:
            args = ap.parse_args(["demo"] + (argv or []))
        cmd_demo(args)


if __name__ == "__main__":
    main()
