"""
XTGAIA live demo: type network traffic in by hand and watch every stage react in real time.

The demo runs the real pipeline (the same XTGAIAPipeline as `main.py`): Role 1 STIX normalisation and
the Dual-Timescale SKG, Role 2 TGN memory update / p_t / topology-aware risk, Role 3 zone Double DQN,
the three XAI tiers, the Shuffle / TheHive dispatch and the closed-loop graph mutation (Algorithm 4).
Nothing is simulated except where the traffic comes from: each command injects flows whose statistics
are real CICIDS2017 rows of that kind (benign, port scan, DDoS, botnet), stamped "now".

  python scripts/live_demo.py                    # warm-up on real Friday traffic, then an interactive prompt
  python scripts/live_demo.py --script scripts/demo_script.txt   # scripted run (presentation / rehearsal)
  python scripts/live_demo.py --no-neo4j --no-soar               # model + graph only, no SOC services

Commands at the prompt (type `help`):
  benign [n]                         n real benign flows between their own hosts (default 20)
  scan   [attacker] [victim] [n]     port scan, n flows to successive ports   (default 172.16.0.1 -> 192.168.10.50, 60)
  ddos   [attacker] [victim] [n]     flood to port 80                         (default 172.16.0.1 -> 192.168.10.50, 200)
  bot    [infected] [c2] [n]         botnet beaconing out to a C2 server       (default 192.168.10.15 -> 205.174.165.73, 30)
  flow   <src> <dst> <port> [kind] [n]   custom flows with the statistics of kind (benign|scan|ddos|bot)
  status                             live graph, contained hosts, highest risks
  rollback [case_id|last]            analyst rollback of a containment (Algorithm 4 feedback)
  wait <seconds>                     advance the demo clock (cool-downs and the Δc window are event time)
  quit
"""
import argparse
import datetime as dt
import json
import os
import shlex
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from src.role3_decision.config import resolve_path  # noqa: E402  (loads .env)

FRIDAY = "data/raw/cicids2017_improved/friday.csv"
PRESETS = "data/samples/demo_flow_presets.json"
KINDS = {"benign": "BENIGN", "scan": "Portscan", "ddos": "DDoS", "bot": "Botnet"}
DEFAULTS = {"scan": ("172.16.0.1", "192.168.10.50", 60), "ddos": ("172.16.0.1", "192.168.10.50", 200),
            "bot": ("192.168.10.15", "205.174.165.73", 30)}

USE_COLOR = sys.stdout.isatty() and os.getenv("NO_COLOR") is None


def c(text, code):
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else str(text)


def head(role, text):
    colors = {"Role 1": "36", "Role 2": "35", "Risk": "33", "Role 3": "32", "XAI": "34", "SOAR": "32",
              "Timing": "90", "Loop": "31"}
    print(f"  {c(f'[{role}]', colors.get(role, '0') + ';1'):<20} {text}")


# ---------------------------------------------------------------------------------------------- presets
def build_presets(path_in, path_out, per_label=60):
    """Real CICIDS2017 Friday rows per kind (statistics only; addresses replaced at injection time)."""
    from src.role1_graph.flow_sources import CICIDSFlowSource
    want = {v.lower(): k for k, v in KINDS.items()}
    out = {k: [] for k in KINDS}
    src = CICIDSFlowSource([str(resolve_path(path_in))])
    for raw in src._rows():
        kind = want.get(str(raw["label"]).lower())
        if kind is None or len(out[kind]) >= per_label:
            continue
        out[kind].append({"src_ip": raw["src_ip"], "dst_ip": raw["dst_ip"], "dst_port": raw["dst_port"],
                          "proto": raw["proto"], "features": raw["features"]})
        if all(len(v) >= per_label for v in out.values()):
            break
    os.makedirs(os.path.dirname(str(resolve_path(path_out))), exist_ok=True)
    with open(resolve_path(path_out), "w", encoding="utf-8") as f:
        json.dump({"source": "CICIDS2017 (corrected release), Friday 7 July 2017", "kinds": out}, f)
    print(f"[✓] presets: { {k: len(v) for k, v in out.items()} } -> {path_out}")


# ---------------------------------------------------------------------------------------------- demo engine
class LiveDemo:
    def __init__(self, args):
        import main as xt
        from src.pipeline import XTGAIAPipeline
        if args.no_soar:
            for k in ("SHUFFLE_EXECUTION_WEBHOOK_URL", "SHUFFLE_WEBHOOK_URL", "SHUFFLE_ROLLBACK_WEBHOOK_URL",
                      "THEHIVE_URL", "THEHIVE_API_KEY"):
                os.environ.pop(k, None)
        if not os.path.exists(resolve_path(PRESETS)):
            build_presets(FRIDAY, PRESETS)
        with open(resolve_path(PRESETS), encoding="utf-8") as f:
            self.presets = json.load(f)["kinds"]
        self.cursor = {k: 0 for k in self.presets}
        ns = argparse.Namespace(skg_window_ms=None, mode=args.mode, checkpoint=args.checkpoint)
        cfg = xt._config(ns)
        # baseline threshold learnt on (most of) the warm-up traffic; finished explicitly after the replay
        cfg["role2"]["burn_in_events"] = int(args.warmup * 0.8)
        print(c("=" * 78, "1"))
        print(c(" XTGAIA LIVE DEMO  -  manual traffic through Role 1 -> Role 2 -> Role 3", "1"))
        print(c("=" * 78, "1"))
        t = time.perf_counter()
        self.pipe = XTGAIAPipeline(cfg, neo4j_driver=None if args.no_neo4j else "env")
        p = self.pipe
        print(f"  TGN checkpoint {p.engine.fingerprint} | trained ω {p.engine.threshold:.4f} | Δc "
              f"{p.cfg['role1']['skg_window_ms']:.0f} ms | Δinf {p.inference_window_s * 1000:.0f} ms | "
              f"Neo4j {'on' if p.graph.skg.driver else 'off'} | mode {p.dispatcher.mode} "
              f"(start-up {time.perf_counter() - t:.1f}s)")
        self.feedback_path = str(resolve_path("data/processed/live_demo_feedback.npz"))
        self.port = None
        try:
            p.serve_rollback_api(feedback_path=self.feedback_path, port=args.api_port)
            self.port = args.api_port
        except Exception as exc:  # noqa: BLE001 - the demo still runs without the API
            print(f"  rollback API unavailable ({exc}); `rollback` disabled")
        if p.graph.skg.driver is not None:
            p.graph.skg.clear_latent_states()
        self.clock = None
        self.last_wall = time.time()
        if args.warmup:
            self.warm_up(args.warmup)
        else:
            self.clock = dt.datetime(2017, 7, 7, 12, 0, tzinfo=dt.timezone.utc).timestamp()

    # ------------------------------------------------------------------ clock
    def now(self):
        """Demo event time: continues from the replayed traffic and advances with the wall clock."""
        wall = time.time()
        self.clock += max(wall - self.last_wall, 0.001)
        self.last_wall = wall
        return self.clock

    # ------------------------------------------------------------------ warm-up
    def warm_up(self, n):
        import main as xt
        print(f"\n  Warm-up: replaying {n:,} real benign-hour flows (Friday 09:00) so the SKG, the TGN memory and "
              f"the baseline alert threshold reflect a live network ...", flush=True)
        t = time.perf_counter()
        src = xt.make_source("cicids", [FRIDAY], "2017-07-07T09:00", "2017-07-07T12:00", n)
        last = None

        def tagged():
            nonlocal last
            for raw in src:
                last = raw["timestamp"]
                yield raw
        self.pipe.run(tagged())
        if self.pipe.burn_in_remaining > 0:          # replay ended inside the burn-in: close it on what was seen
            self.pipe.burn_in_remaining = 0
            self.pipe._finish_burn_in()
        self.clock, self.last_wall = float(last or 0), time.time()
        b = self.pipe.report.burn_in or {}
        g = self.pipe.graph.skg.graph
        print(f"  done in {time.perf_counter() - t:.1f}s: live SKG {g.number_of_nodes():,} hosts / "
              f"{g.number_of_edges():,} edges in the Δc window | alert threshold "
              f"{b.get('trained_threshold', 0):.4f} -> {self.pipe.engine.threshold:.4f} "
              f"(1% baseline alert rate) | per-host baselines {b.get('host_baselines', 0)}")
        self.pipe.report.dispatches.clear()

    # ------------------------------------------------------------------ traffic
    def flows(self, kind, n, src=None, dst=None, port=None, port_step=False):
        rows = self.presets[kind]
        out, t = [], self.now()
        for i in range(n):
            r = rows[self.cursor[kind] % len(rows)]
            self.cursor[kind] += 1
            out.append({"src_ip": src or r["src_ip"], "dst_ip": dst or r["dst_ip"],
                        "dst_port": (port + i if port_step else port) if port is not None else r["dst_port"],
                        "proto": r["proto"], "features": list(r["features"]), "timestamp": t + i * 0.002,
                        "label": KINDS[kind]})
        self.clock = out[-1]["timestamp"]
        return out

    def inject(self, raws, title):
        p, rep = self.pipe, self.pipe.report
        print(f"\n{c('>>', '1')} {c(title, '1')}  ({len(raws)} flows)")
        before = {"disp": len(rep.dispatches), "flag": rep.flagged_flows, "supp": dict(rep.suppressed),
                  "acts": dict(rep.policy_actions), "blocked": p.graph.stats["blocked"],
                  "lat": {k: len(v) for k, v in rep.latency.items()}}
        edges, labels, t_acq = [], [], 0.0
        for raw in raws:
            t_in = time.perf_counter()
            new = p.graph.ingest_all(raw)
            t_acq += (time.perf_counter() - t_in) * 1000.0
            for e in new:
                if e.get("features") is None:
                    continue
                e["_t_ingest"] = t_in
                edges.append(e)
                labels.append(raw.get("label"))
        blocked = p.graph.stats["blocked"] - before["blocked"]
        g = p.graph.skg.graph
        stix = (edges[0].get("stix_bundle") or {}).get("id") if edges else None
        head("Role 1", f"{len(edges)} flows normalised to STIX 2.1 ({stix or '-'}) and added to the SKG; "
                       f"{blocked} dropped by active containment | live graph {g.number_of_nodes()} hosts / "
                       f"{g.number_of_edges()} edges | {t_acq:.1f} ms")
        if edges:
            tech = {}
            for e in edges:
                t_ = e.get("technique")
                if isinstance(t_, dict):
                    key = t_.get("technique_id") or t_.get("id") or t_.get("external_id")
                    tech.setdefault(str(key), t_.get("name") or "")
            if tech:
                head("Role 1", "MITRE ATT&CK mapping of the ports: "
                     + ", ".join(f"{k} {v}".strip() for k, v in sorted(tech.items())[:4]))
        for i in range(0, len(edges), p.micro_batch):
            p._process_batch(edges[i:i + p.micro_batch], labels[i:i + p.micro_batch])
        if not edges:
            return
        # Role 2: per involved host
        hosts = []
        for e in edges:
            for h in (p.target_of(e)[0], p.target_of(e)[1]):
                if h not in hosts:
                    hosts.append(h)
        _, risk = p.graph.get_active_graph_and_risk()
        flagged = rep.flagged_flows - before["flag"]
        head("Role 2", f"TGN memory updated ({len(edges)} events); {flagged}/{len(edges)} flows above the alert "
                       f"threshold {p.engine.threshold:.4f}")
        for h in hosts[:4]:
            p_t, z = p.engine.score(h)
            mean_p, n = p.engine.trend(h)
            thr = p.host_threshold.get(h, p.engine.threshold)
            cat = p.engine.classify(h, p_t) or {}
            flag = c("ALERT", "31;1") if p_t >= thr else c("ok", "32")
            head("Role 2", f"{h:<15} p_t {p_t:.3f} (trend {mean_p:.3f} over {n}) thr {thr:.3f} {flag} | "
                           f"category {cat.get('category', '-')} | Z_t[0:3] {[round(float(v), 2) for v in z[:3]]}")
        for h in hosts[:2]:
            if h in risk:
                near = p.graph.nearest_crown_jewel(h)
                head("Risk", f"{h:<15} topological risk {risk[h]:.3f}"
                             + (f" | nearest crown jewel {near[0]} (friction distance {near[1]:.2f})" if near else ""))
        # Role 3
        new_disp = rep.dispatches[before["disp"]:]
        acts = {k: v - before["acts"].get(k, 0) for k, v in rep.policy_actions.items() if v - before["acts"].get(k, 0)}
        supp = {k: v - before["supp"].get(k, 0) for k, v in rep.suppressed.items() if v - before["supp"].get(k, 0)}
        if acts:
            head("Role 3", f"zone Double DQN decisions: {acts}")
        if supp:
            head("Role 3", f"not dispatched: {supp}")
        if not acts and not supp and not new_disp:
            head("Role 3", "no host escalated (nothing above threshold with sustained evidence)")
        for d in new_disp:
            head("Role 3", c(f"{d['action']} on {d['host']} (zone {d['zone']}, playbook {d['playbook']})", "31;1")
                 + f" | p_t {d['p_t']:.3f} risk {d['risk']:.3f} | margin {d['decision_margin']} over "
                   f"{d['runner_up']}")
            tiers = getattr(p.engine, "last_tiers", None) or {}
            se = tiers.get("structural_edges") or []
            if se:
                shown, seen = [], set()
                for e in se:
                    key = tuple(e[:2]) if isinstance(e, (list, tuple)) else str(e)
                    if key in seen:
                        continue
                    seen.add(key)
                    shown.append(f"{e[0]}->{e[1]} ({e[2]:.2f})" if isinstance(e, (list, tuple)) else str(e))
                head("XAI", f"Tier 1 PGExplainer top edges ({len(se)} ranked): " + "; ".join(shown[:3]))
            tmp = tiers.get("temporal") or {}
            if tmp:
                head("XAI", f"Tier 2 attention rollout: onset at event {tmp.get('onset_index')} "
                            f"(+{tmp.get('onset_timestamp_ms', 0):.0f} ms), duration {tmp.get('attack_duration_ms', 0):.0f} ms")
            head("XAI", f"Tier 3 reward decomposition + bundle X_t in {d['xai_ms']:.1f} ms "
                        f"({'within' if d['xai_within_200ms'] else 'OVER'} the 200 ms bound)")
            head("SOAR", f"Shuffle {'dispatched' if d['shuffle'] else 'not reached'} | TheHive case "
                         f"{d['case_id'] if d['thehive'] else d['case_id'] + ' (local, TheHive not reached)'}")
            scope = (f"port {d['port']} on {d['host']} is blocked" if d["action"] == "PortIsolation" else
                     f"all traffic from/to {d['host']} is blocked" if d["action"] == "Quarantine" else
                     f"{d['host']} is under {d['action']}")
            head("Loop", f"G_t+1: {d['edges_severed']} live edges severed; {scope} until rollback or audit commit "
                         f"(Algorithm 4){'' if not d['topology_error'] else ' | error ' + d['topology_error']}")
        # timing for this injection
        def new(key):
            return rep.latency.get(key, [])[before["lat"].get(key, 0):]
        parts = []
        for key, name in (("analysis_ms", "analysis"), ("decision_ms", "decision"), ("xai_total_ms", "XAI"),
                          ("execution_ms", "execution"), ("mttr_ms", "MTTR"), ("end_to_end_ms", "end-to-end")):
            v = new(key)
            if v:
                parts.append(f"{name} {sum(v) / len(v):.1f} ms")
        if parts:
            head("Timing", " | ".join(parts))

    # ------------------------------------------------------------------ other commands
    def status(self):
        p = self.pipe
        g = p.graph.skg.graph
        _, risk = p.graph.get_active_graph_and_risk()
        top = sorted(risk.items(), key=lambda kv: kv[1], reverse=True)[:5]
        print(f"\n  live SKG {g.number_of_nodes()} hosts / {g.number_of_edges()} edges | alert threshold "
              f"{p.engine.threshold:.4f} | dispatches so far {len(p.report.dispatches)}")
        print(f"  contained: { {h: [c_.action for c_ in cs] for h, cs in p.enforcement.active.items()} or 'none'}")
        print(f"  highest risk: {', '.join(f'{h} {r:.3f}' for h, r in top) or '-'}")

    def rollback(self, which):
        import requests
        if self.port is None:
            print("  rollback API not running")
            return
        disp = [d for d in self.pipe.report.dispatches if d["action"] in ("Quarantine", "PortIsolation")] \
            or self.pipe.report.dispatches
        if not disp:
            print("  nothing to roll back")
            return
        d = disp[-1] if which in (None, "last") else next((x for x in disp if x["case_id"] == which), None)
        if d is None:
            print(f"  no dispatch with case id {which}")
            return
        r = requests.post(f"http://127.0.0.1:{self.port}/api/v1/feedback/rollback",
                          json={"case_id": d["case_id"], "analyst_comment": "live demo: analyst override"},
                          headers={"X-XTGAIA-Token": os.environ.get("ROLLBACK_WEBHOOK_TOKEN", "")}, timeout=30)
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        lifted = self.pipe.refresh_enforcement()
        print(f"\n{c('>>', '1')} {c('Analyst rollback of ' + d['action'] + ' on ' + d['host'], '1')}")
        head("Loop", f"HTTP {r.status_code} {body.get('status')} | Shuffle revert {body.get('soar_reverted')} | "
                     f"edges restored {body.get('edges_restored')} | enforcement lifted for {[h for h, _ in lifted]}")
        head("Role 3", "penalty transition (S_t, a_t, R_penalty, S_t+1) stored in the replay buffer for the "
                       "next policy update")

    def handle(self, line):
        try:
            parts = shlex.split(line)
        except ValueError as exc:
            print(f"  {exc}")
            return True
        if not parts:
            return True
        cmd, a = parts[0].lower(), parts[1:]
        try:
            if cmd in ("quit", "exit", "q"):
                return False
            if cmd == "help":
                print(__doc__.split("Commands at the prompt (type `help`):")[1])
            elif cmd == "benign":
                n = int(a[0]) if a else 20
                self.inject(self.flows("benign", n), f"benign traffic: {n} real flows")
            elif cmd in DEFAULTS:
                s, d, n = DEFAULTS[cmd]
                s = a[0] if len(a) > 0 else s
                d = a[1] if len(a) > 1 else d
                n = int(a[2]) if len(a) > 2 else n
                port, step = {"scan": (1, True), "ddos": (80, False), "bot": (8080, False)}[cmd]
                self.inject(self.flows(cmd, n, s, d, port, step), f"{cmd.upper()}: {s} -> {d}")
            elif cmd == "flow":
                if len(a) < 3:
                    print("  usage: flow <src> <dst> <port> [benign|scan|ddos|bot] [n]")
                    return True
                kind = a[3] if len(a) > 3 else "benign"
                if kind not in KINDS:
                    print(f"  kind must be one of {list(KINDS)}")
                    return True
                n = int(a[4]) if len(a) > 4 else 1
                self.inject(self.flows(kind, n, a[0], a[1], int(a[2])), f"custom {kind} flow(s): {a[0]} -> {a[1]}:{a[2]}")
            elif cmd == "status":
                self.status()
            elif cmd == "rollback":
                self.rollback(a[0] if a else None)
            elif cmd == "wait":
                secs = float(a[0]) if a else 60
                self.clock += secs
                print(f"  demo clock advanced {secs:.0f}s")
            else:
                print(f"  unknown command {cmd!r}; type help")
        except Exception as exc:  # noqa: BLE001 - keep the presentation running
            print(f"  error: {type(exc).__name__}: {exc}")
        return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--warmup", type=int, default=30000, help="real Friday flows replayed first (0 = none)")
    ap.add_argument("--script", default=None, help="file with one command per line (# comments allowed)")
    ap.add_argument("--pause", type=float, default=1.5, help="seconds between scripted commands")
    ap.add_argument("--no-neo4j", action="store_true")
    ap.add_argument("--no-soar", action="store_true")
    ap.add_argument("--mode", choices=["active", "passive"], default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--api-port", type=int, default=int(os.getenv("ROLLBACK_API_PORT", "8008")))
    ap.add_argument("--build-presets", action="store_true", help="(re)extract the CICIDS preset flows and exit")
    args = ap.parse_args(argv)
    if args.build_presets:
        build_presets(FRIDAY, PRESETS)
        return
    demo = LiveDemo(args)
    if args.script:
        with open(args.script, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                print(f"\n{c('xtgaia>', '1;36')} {line}")
                if not demo.handle(line):
                    break
                time.sleep(args.pause)
        return
    print("\n  Type `help` for commands. Try: benign, scan, status, rollback, ddos")
    while True:
        try:
            line = input(f"\n{c('xtgaia>', '1;36')} ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not demo.handle(line):
            break


if __name__ == "__main__":
    main()
