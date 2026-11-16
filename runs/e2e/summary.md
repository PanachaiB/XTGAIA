# Full end-to-end run (task 2): 2026-11-16 23:04–23:38

`python main.py demo`: Neo4j, Shuffle and TheHive up, tuned CICIDS checkpoint (`evhead-tuned`, max read-out),
held-out Friday (547,557 events) through Role 1 → Role 2 → Role 3 → SOAR → closed loop. Exit 0. Log:
`runs/e2e/demo_full.log`; report: `data/processed/demo_report.json`.

## Every stage ran

| Check | Result |
|---|---|
| Ingestion (Role 1) | 547,557 events; 381,477 ingested into the SKG; 166,080 blocked by active containment; 0 malformed |
| Neo4j persistence | 381,477 edges synced, 0 errors (deadlocks retried by the managed transactions) |
| Burn-in | 25,600 events (no attacks inside the window), alert threshold 0.9693 → 0.9978, 15 per-host baselines |
| Role 2 detection, live (flow level, burn-in threshold) | precision 0.965, recall 0.660, F1 0.784, FPR 0.058 (PortScan 65.6 %, DDoS 66.8 %, Botnet 57.7 %) |
| Role 3 | 2,271 policy decisions → 53 SOAR dispatches (cool-down 2,218, insufficient evidence 3,402 suppressed) |
| Shuffle / TheHive | 53/53 dispatched, 53/53 cases; the case read back over the API has the Role 1, Role 2 and Role 3 evidence sections |
| Analyst rollback (HTTP) | 200 SUCCESS: Shuffle revert, 1 edge restored, enforcement lifted, reward −100 stored and persisted; repeat → 400, no token → 401 |
| Audit window (Algorithm 4) | 53 commits on timeout; approve on a committed case → 400 (already committed) |
| Neo4j state | 8,343 hosts, 22,676 COMMUNICATED edges, soft-deleted edge flagged inactive |
| Algorithm 1 | 110,175 latent saves, 71,344 recoveries; 8,343 hosts with a 512-byte state; a fresh engine restores 192.168.10.50 (p 0.0485 from the checkpoint → 0.0430 after recovery; live 0.0443) |
| XAI | 53 explanation bundles; X_t mean 68.1 ms, p95 201.6 ms; stability (same class) structural 1.000, temporal 1.000, strategic 0.997 |
| Online policy updates | 2/12 accepted |

## Response quality

- 47 of 53 dispatches targeted attack infrastructure:
  - 43 involve the botnet C2 address 205.174.165.73;
  - 4 involve the PortScan/DDoS attacker 172.16.0.1.
- The corrected CICIDS2017 labels most C2 flows *"Botnet - Attempted"* (4,067 of 4,803 flows with
  205.174.165.73). Our loader maps those to BENIGN, following Engelen et al. That is why only 7 dispatches
  carry an attack label.
- 6 dispatches are true false responses. One is a Quarantine of 192.168.10.3, which the run's analyst-rollback
  step then reversed.

## Latency in this run

| Quantity | This run (mean / p95, ms) | Earlier demo run (mean, ms) |
|---|---|---|
| Analysis per micro-batch | 51.1 / 155.8 | 31.5 |
| Decision | 53.6 / 120.0 | 19.6 |
| Execution (Shuffle → enforcement → G_t+1) | 664.7 / 1,945.4 | 191.9 |
| MTTR | 1,096.7 / 3,581.7 | 307.1 |
| End-to-end | 1,170.4 / 4,255.5 | 349.9 |
| TheHive audit (after enforcement) | 1,627.6 / 4,826.8 | 879.7 |

These are worse than the earlier run because the laptop was out of memory. The WSL VM (8 GB) was running the
full SOC stack (Cassandra, Elasticsearch, two OpenSearch/Neo4j services, TheHive) next to the pipeline, with
its 4 GB swap nearly full after the tuning runs. Do not use these as the paper's latency. The latency baseline
(`scripts/latency_bench.sh`) has to run after a WSL restart, and the paper numbers come from the teammate's
hardware run.

## Differences from the offline evaluation

The live F1 (0.784) is below the offline Friday evaluation (0.960) for two reasons:
- the burn-in raises the alert threshold from the validation ω (0.969) to 0.998;
- flows blocked by containment are never scored.
Live operation trades recall for far fewer false alerts (precision 0.965). This is the intended burn-in
behaviour (B4/B8), not a defect.
