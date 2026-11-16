# Hand-off: latency measurement and the live demo

## 1. Latency and end-to-end timing (paper Sec. IV-D)

One command, run on the paper's hardware from the repository root:

```bash
scripts/latency_bench.sh
```

It starts Neo4j, Shuffle and TheHive, waits until TheHive and Shuffle answer (up to 5 minutes), replays 300,000 events of the held-out CICIDS2017 Friday
(from 13:30, so PortScan and DDoS fall after the burn-in) three times through the full pipeline, runs
the TGN micro-benchmark, and writes `runs/latency/<timestamp>/latency.md` and `latency.json`.

Variants:

```bash
scripts/latency_bench.sh --no-stack                      # model + graph only, no SOC services
scripts/latency_bench.sh -- --limit 100000 --repeat 5    # pass options to scripts/latency_bench.py
```

What the table contains (all processing-only unless stated):

| Row | Start → end | Paper / comparator |
|---|---|---|
| Acquisition | raw event → STIX + SKG update (per event) | AIR [5] 8.1 ms |
| Analysis | TGN memory update + Algorithm 2 risk (per micro-batch) | AIR 15.2 ms |
| Decision | state vector + zone Double DQN | AIR 23.5 ms |
| Execution | Shuffle webhook → enforcement → G_t+1 | AIR 11.5 ms |
| End-to-end | ingest → G_t+1 (Algorithm 4 Phase 1, deviation A2) | AIR 58.3 ms |
| MTTD | ingest → TGN detection; also with the Δinf = 500 ms window | none published |
| MTTR | detection → SOAR enforcement | [2] 1,800 ms |
| XAI | Tiers 1+2, Tier 3, the parallel bundle X_t | 200 ms bound |
| Throughput | events/s over the replay, start-up excluded | AIR 18,540 events/s |

The hardware (CPU model, cores, RAM, GPU, torch and git commit) is written into both files.

**Check before reporting**
- The *Dispatches* column must be above 0, otherwise there are no execution, MTTR or end-to-end samples.
- *Shuffle ok* and *TheHive ok* must equal *Dispatches* for a full closed-loop measurement.
- Smoke test on the development laptop (2 CPU threads, training running alongside, no SOC services):
  end-to-end 57.9 ms, MTTR 44.9 ms, X_t 24.6 ms, 355 events/s. With Shuffle and TheHive on, execution
  includes the webhook round trip (≈ 190 ms on the laptop), which is the main difference from AIR.

### Reference baseline: development laptop (2026-11-17)

`scripts/latency_bench.sh`, full SOC stack, 3 × 300,000 events. Run after a WSL restart, with nothing else
running. Raw files: `runs/latency/20261116-182925/`.

Hardware: i5-12500H (16 logical CPUs, torch 8 threads), Docker Desktop WSL2 VM with 8 GB RAM, no GPU, git
5bc628a, tuned CICIDS checkpoint (00b3a95a41623a36). All 66 dispatches reached Shuffle and TheHive.

| Stage | Mean (ms) | p50 | p95 | AIR [5] / [2] |
|---|---|---|---|---|
| Acquisition (per event) | 0.07 | 0.02 | 0.17 | 8.1 |
| Analysis (TGN + risk, per micro-batch) | 37.8 | 27.3 | 93.7 | 15.2 |
| Decision (state + zone Double DQN) | 42.9 | 40.0 | 68.2 | 23.5 |
| Explanation bundle X_t | 63.2 | 54.0 | 114.9 | (200 ms bound) |
| Execution (Shuffle → enforcement → G_t+1) | 625.8 | 342.6 | 1,606.8 | 11.5 |
| End-to-end (ingest → G_t+1) | 943.2 | 554.2 | 2,679.5 | 58.3 |
| MTTD processing only / incl. Δinf | 101.3 / 206.3 | 85.3 / 90.3 | 166.6 / 1,140.3 | – |
| MTTR | 896.5 | 492.7 | 2,640.7 | [2] 1,800 |
| TGN micro-batch of 128 (no services) | 51.4 | – | 67.0 | – |
| Throughput | 353–448 events/s | | | 18,540 events/s |

**Reading it.**
- The model and graph path is ingest → detect → decide → explain:
  - analysis and decision each cost tens of ms (MTTD 101 ms);
  - Tier 3 is about 2 ms, X_t about 63 ms.
- About two thirds of the end-to-end time is the synchronous Shuffle webhook round trip ("Execution").
  On this laptop, Shuffle shares the 8 GB VM with OpenSearch, Cassandra, Elasticsearch, Neo4j and TheHive.
- The mean MTTR (897 ms) beats [2]'s 1,800 ms. AIR's per-stage numbers come from 4 servers and an A100.
- Expect the paper hardware to cut execution sharply if the SOC services have their own resources.
- The earlier demo run reported 191.9 ms execution and 349.9 ms end-to-end. The laptop baseline above is the
  current reference.

## 2. Live demo for the presentation

Type traffic in by hand and watch each stage react: Role 1 (STIX, SKG, ATT&CK mapping), Role 2 (p_t,
alert threshold, attack category, Z_t), topology-aware risk, Role 3 (zone Double DQN action), the three
XAI tiers, the SOAR dispatch, the closed-loop graph mutation, and the analyst rollback.

```bash
docker compose -f docker/docker-compose.yml up -d neo4j shuffle shuffle-backend shuffle-orborus shuffle-opensearch cassandra elasticsearch thehive
```

```bash
docker compose -f docker/docker-compose.yml run --rm -e PYTHONPATH=/app:/app/data/pylib xtgaia-core python scripts/live_demo.py
```

Without the SOC stack, add `--no-neo4j --no-soar` (and `--no-deps` to the compose command).
For a rehearsed run, use `--script scripts/demo_script.txt`: normal traffic, a port scan on the
web server, containment, a DDoS, then an analyst rollback.

Commands at the prompt:

| Command | Effect |
|---|---|
| `benign [n]` | n real benign CICIDS flows |
| `scan [attacker] [victim] [n]` | port scan (default 172.16.0.1 → 192.168.10.50, 60 flows) |
| `ddos [attacker] [victim] [n]` | flood to port 80 |
| `bot [infected] [c2] [n]` | botnet beaconing |
| `flow <src> <dst> <port> [kind] [n]` | custom flows with the statistics of `kind` |
| `status` | live graph, contained hosts, highest risks |
| `rollback [case_id\|last]` | analyst rollback (Algorithm 4 feedback) |
| `wait <seconds>` | advance the demo clock (cool-downs are in event time) |

How it works: on start the demo replays 30,000 real Friday-morning flows (about 40 s) so the SKG, the
TGN memory and the burn-in alert threshold reflect a live network. Each command then injects flows whose
statistics are real CICIDS2017 rows of that kind (`data/samples/demo_flow_presets.json`), stamped with
the demo clock, and runs them through the same `XTGAIAPipeline` code as `main.py`. Use the CICIDS testbed
addresses (192.168.10.x internal, 192.168.10.50 the web server crown jewel, 172.16.0.1 the attacker):
the TGN has node memory for them.
