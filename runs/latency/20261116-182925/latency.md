# XTGAIA latency benchmark

Generated 2026-11-16T18:29:25.768190+00:00 | git 5bc628a | checkpoint 00b3a95a41623a36

## Hardware

- CPU: 12th Gen Intel(R) Core(TM) i5-12500H (16 logical CPUs, torch threads 8)
- RAM: 7.8 GB | GPU: none (CPU inference)
- Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.41 | Python 3.11.18 | torch 2.15.0+cu132 | container: True
- Services: {'thehive': 'unreachable (ConnectionError)', 'shuffle': 200} | Neo4j: True

## Replay

Input data/raw/cicids2017_improved/friday.csv, 300000 events per run, 3 run(s). Times are processing-only unless stated; pooled over runs.

| Run | Events | Wall (s) | Events/s | Flows/s | Dispatches | Shuffle ok | TheHive ok |
|---|---|---|---|---|---|---|---|
| 1 | 300,000 | 670.1 | 448 | 367 | 22 | 22 | 22 |
| 2 | 300,000 | 850.0 | 353 | 289 | 22 | 22 | 22 |
| 3 | 300,000 | 737.9 | 407 | 333 | 22 | 22 | 22 |

Throughput comparator: AIR [5] 18,540 events/s (4 × Dell R760 + A100).

## Latency (ms)

| Stage | n | Mean | p50 | p95 | p99 | Std | Comparator |
|---|---|---|---|---|---|---|---|
| Acquisition (parse, STIX, SKG update; per event) | 900,000 | 0.07 | 0.02 | 0.17 | 0.55 | 17.30 | AIR 8.1 ms |
| Analysis (TGN update + Algorithm 2 risk; per micro-batch) | 25,161 | 37.83 | 27.33 | 93.68 | 130.30 | 39.33 | AIR 15.2 ms |
| Decision (state vector + zone Double DQN) | 2,367 | 42.85 | 40.02 | 68.18 | 97.93 | 78.83 | AIR 23.5 ms |
| Execution (Shuffle -> enforcement -> G_t+1) | 66 | 625.80 | 342.64 | 1606.75 | 4417.12 | 1091.61 | AIR 11.5 ms |
| End-to-end closed loop (ingest -> G_t+1) | 66 | 943.15 | 554.20 | 2679.45 | 6124.25 | 1269.75 | AIR 58.3 ms |
| MTTD, processing only (ingest -> TGN detection) | 249,438 | 101.25 | 85.29 | 166.57 | 364.83 | 94.71 | – |
| MTTD incl. Δinf window | 249,438 | 206.26 | 90.27 | 1140.32 | 1508.89 | 385.09 | – |
| MTTR (detection -> SOAR enforcement) | 66 | 896.52 | 492.65 | 2640.66 | 6052.24 | 1265.79 | [2] 1,800 ms |
| XAI Tiers 1+2 (PGExplainer + attention rollout) | 66 | 62.59 | 53.72 | 114.46 | 141.00 | 28.03 | – |
| XAI Tier 3 (reward decomposition) | 66 | 1.66 | 1.39 | 4.05 | 5.97 | 1.24 | – |
| Explanation bundle X_t (parallel tiers) | 66 | 63.16 | 53.99 | 114.88 | 141.80 | 28.35 | bound 200 ms |
| TheHive audit case (Algorithm 4 Phase 2, after enforcement) | 66 | 1290.48 | 994.21 | 2811.68 | 5960.68 | 1123.71 | – |

No dispatch means no execution / MTTR / end-to-end samples: the replay window must contain attack traffic after the burn-in (Friday 13:55-16:30 has PortScan and DDoS).

## TGN micro-benchmark (memory update + scoring, no services)

| Micro-batch | Mean (ms) | p95 (ms) | Per flow (µs) |
|---|---|---|---|
| 1 | 6.99 | 10.65 | 6989 |
| 16 | 15.11 | 22.87 | 944 |
| 64 | 31.38 | 45.40 | 490 |
| 128 | 51.38 | 67.03 | 401 |
