# XTGAIA comparison scorecard

Status of every comparative measure in the paper (Sec. IV-C to IV-E plus the response policy),
checked against the updated comparison spec (`baseline_comparison.md`, with [N1]–[N5]). A **win** is a
measured XTGAIA number that is better than a comparator's number on the same metric and unit; a
**loss** is the opposite; **tie** means inside the comparator's 95 % interval. CONTEXT-ONLY values
are never counted. Sources: `docs/evaluation_results.md` (detection, XAI, response) and
`data/processed/demo_report.json` (latency).

Snapshot: 2026-11-16 23:00. TGN rows are the tuned finals (`evhead-tuned`, max read-out, test touched once;
B11). The earlier UNSW max checkpoint (F1 0.863) gives the same win/loss pattern.

## Summary

| Topic | Comparisons counted | Wins | Ties | Losses | Majority? |
|---|---|---|---|---|---|
| Threat detection (IV-C) | 22 | 9 | 1 | 12 | No (allowed exception: note in results + future work) |
| System latency (IV-D) | 6 | 2 | 0 | 4 | **No** on the laptop; hardware-dependent (teammate re-measures) |
| Explainability (IV-E) | 12 | 4 | 0 | 8 | **No** (faster than LIME / KernelSHAP; fidelity ≈ 0; stability not countable) |
| Response policy | 12 | 7 | 4 | 1 | Yes |

## 1. Threat detection (exception: losses go into the results with future work)

### vs CCSTGN [11], NF-UNSW-NB15-v2, chronological 70/10/20 (DIRECT)

Best CCSTGN variant per column; TGN = tuned final, max read-out (1 seed).

| Metric | XTGAIA TGN | Best [11] | Result |
|---|---|---|---|
| ROC-AUC | 0.9953 | 0.9989 (iso) | loss |
| F1 @ weighted threshold | 0.873 | 0.951 (iso-ft) | loss |
| F1 @ FPR 0.01 % | 0.279 | 0.503 (rs) | loss (beats CCSTGN-iso 0.223) |
| F1 @ FPR 0.1 % | 0.467 | 0.770 (rs) | loss |
| F1 @ FPR 1 % | 0.858 | 0.950 (rs-ft) | loss |
| FNR @ FPR 1 % | 0.154 | 0.0001 (rs) | loss |

### vs models quoted in [11] from 3D-IDS (TGN†, E-GraphSAGE†, 3D-IDS†)

| Metric | XTGAIA | TGN† | E-GraphSAGE† | 3D-IDS† |
|---|---|---|---|---|
| ROC-AUC | **0.9953** | 0.8801 | 0.9039 | 0.9155 |
| F1 | 0.873 | 0.9355 | 0.9410 | 0.9545 |

AUC: 3 wins. F1: 3 losses.

### vs E-GraphSAGE [N1], run in-house on the same splits (PROTOCOL-ONLY → our own numbers)

| Dataset / metric | XTGAIA | E-GraphSAGE | Result |
|---|---|---|---|
| UNSW F1 (default) | 0.857 | 0.859 ± 0.015 | tie |
| UNSW ROC-AUC | 0.9953 | 0.9940 | win |
| UNSW F1 @ FPR 0.01 % | 0.279 | 0.219 | win |
| UNSW F1 @ FPR 0.1 % | 0.467 | 0.445 | win |
| UNSW F1 @ FPR 1 % | 0.858 | 0.810 | win |
| UNSW F1 @ weighted threshold | 0.873 | 0.893 | loss |
| CICIDS held-out Friday F1 | 0.960 | 0.917 | win |
| CICIDS held-out Friday ROC-AUC | 0.9965 | 0.9945 | win |

### vs flow classifiers (in-house RF / XGBoost / MLP / 1D-CNN, all protocols)

The flow classifiers beat the TGN on every detection metric on both datasets (e.g. UNSW F1 0.959–0.972
vs 0.857; CICIDS held-out F1 0.970–0.993 vs 0.960). Counted as one loss per dataset (2 losses).

### Not counted

- [3] Vedesh: the TGN has not been run under P-CIC split A, so its published rows cannot be set against
  the TGN. Our in-house replicas of [3]'s models beat its published numbers (F1 ≥ 0.999 vs 0.946–0.963).
- [12] Causal-GNN (unsupervised AUC 0.934), [13] Ontology-APT, [5] AIR, [2] Kulkarni: CONTEXT-ONLY.
- [N2] Kairos, [N3] threaTrace, [N4] Orthrus: need DARPA TC / OpTC provenance data (future work, section D
  of `paper_deviations.md`). Our own node-level numbers under [N4]'s strict rule and [N3]'s lenient rule
  exist from the design check (strict node F1 ≈ 0.11–0.13; lenient saturates at 1.0), but not on their data.

### Write-up for the results section (detection)

XTGAIA's TGN beats the graph-based detectors it can be set against (E-GraphSAGE in-house on both datasets,
and the TGN / E-GraphSAGE / 3D-IDS AUCs quoted in [11]), but not CCSTGN or plain flow classifiers on
NF-UNSW-NB15. The cause is structural: p_t is a node score, and NF-UNSW-NB15 has only 41 hosts, so one
host's score must cover all its benign and malicious flows (the design check showed a flow-level head on
the same model reaches the MLP's F1, 0.950). Tuning within the knobs the paper leaves open did not close the gap (B11). Future-work items:
1. A flow-level read-out alongside p_t (kept out by design; documented in B10).
2. Multi-seed TGN runs (5 seeds, as [11]).
3. Evaluation on provenance data (DARPA TC / OpTC) to compare with [N2]–[N4] on their own units.

## 2. System latency (vs AIR [5] stage by stage, [2] for MTTR)

Laptop baseline: `scripts/latency_bench.sh`, full SOC stack, 3 × 300,000 events, after a WSL restart
(`docs/handoff_latency_and_demo.md`). Hardware: i5-12500H laptop with an 8 GB WSL VM and no GPU; AIR used
4 × Dell R760 + A100. Processing-only means. The teammate's hardware run replaces this table.

| Stage | XTGAIA mean (p50) | Comparator | Result |
|---|---|---|---|
| Acquisition (per event) | 0.07 ms | AIR 8.1 ms | win |
| Analysis (TGN + risk, per micro-batch) | 37.8 ms (27.3) | AIR 15.2 ms | loss |
| Decision (state + Double DQN) | 42.9 ms (40.0) | AIR 23.5 ms | loss |
| Execution (SOAR → enforcement → G_t+1) | 625.8 ms (342.6) | AIR 11.5 ms | loss |
| End-to-end closed loop | 943.2 ms (554.2) | AIR 58.3 ms | loss |
| MTTR (detection → enforcement) | 896.5 ms (492.7) | [2] 1,800 ms | win |
| Throughput | 353–448 events/s | AIR 18,540 events/s | loss (not counted: hardware) |

**2 wins, 4 losses on this laptop.** The earlier demo run gave decision 19.6 ms (a win) and end-to-end
349.9 ms; the clean three-run baseline replaces it.

Notes for the teammate's run:
- About two thirds of end-to-end is the synchronous Shuffle webhook round trip. On the laptop, Shuffle
  shares the 8 GB VM with OpenSearch, Cassandra, Elasticsearch, Neo4j and TheHive.
- Without the SOC services, the model and graph path ran end-to-end in 57.9 ms (AIR 58.3) in the smoke
  test.
- The deciding factors for the paper numbers are the hardware and giving the SOC services their own
  resources.
- MTTD has no published comparator ([5] and [2] do not report it). Laptop value: 101 ms processing-only,
  206 ms including the Δinf window.

## 3. Explainability (all comparator values measured by us; no paper publishes them)

XTGAIA Tier 1+2 (PGExplainer + attention rollout on the TGN), measured at detection time on 1,199 Friday
detections (`runs/final_metrics/`): latency mean 85.9 ms (p95 217.5), fidelity+ about 0, stability 1.000
(degenerate, see below). Post-hoc explainers on [3]'s models (P-CIC setup), 1,000 attack instances each, same
laptop, 8 threads (`data/processed/eval/explainer_baselines_1000.json`):

| Explainer | Latency mean / p95 (ms) | Fidelity (top-5 deletion) | Cosine stability [6] | Top-5 overlap [N5] | Latency vs XTGAIA | Fidelity vs XTGAIA |
|---|---|---|---|---|---|---|
| LIME (RF) | 144.6 / 171.3 | 0.293 | 0.804 | 0.703 | win | loss |
| TreeSHAP (RF) | 2.4 / 3.0 | 0.266 | 0.983 | 1.000 | loss | loss |
| LIME (XGB) | 99.9 / 124.0 | 0.363 | 0.703 | 0.662 | win (narrow) | loss |
| TreeSHAP (XGB) | 1.3 / 2.1 | 0.790 | 0.980 | 1.000 | loss | loss |
| LIME (1D-CNN) | 167.4 / 205.5 | 0.105 | 0.848 | 0.641 | win | loss |
| KernelSHAP (1D-CNN) | 953.9 / 1,314.7 | 0.461 | 0.731 | 0.605 | win | loss |

E-XAI descriptive accuracy (drop in p(attack) with the top-k features removed, k = 1 / 3 / 5 / 10 / 20) and
sparsity (share of features below τ = 0.01 / 0.05 / 0.1 / 0.25 of the largest attribution) are in the JSON.
Example: TreeSHAP (XGB) 0.09 / 0.60 / 0.79 / 0.99 / 1.00, LIME (1D-CNN) 0.01 / 0.05 / 0.10 / 0.10 / 0.22.

**4 wins, 8 losses (stability not counted).** This revises the earlier 10 / 8. The earlier count used a
300-instance run with contended CPU and credited XTGAIA's stability of 1.000. Caveats:
- **Stability is not counted.** XTGAIA's 1.000 over 10,000 same-class pairs is degenerate: detected nodes have
  saturated scores (p_t about 0.999), so its attribution vectors are near-constant and any two are collinear.
  PGExplainer is deterministic per instance, so E-XAI's top-k overlap would also be 1.000 by construction.
- **Fidelity is a real weakness.** Masking the explained neighbourhood does not move p_t, because p_t depends
  mostly on the node's own memory (A1). The definitions also differ (subgraph fidelity+ vs feature deletion).
- **Latency.** XTGAIA beats LIME and KernelSHAP, but TreeSHAP is faster. XTGAIA explains a 2-hop temporal
  graph, while TreeSHAP explains a single flow on a tree model. 93.7 % of XTGAIA explanations meet the 200 ms
  bound.
- **What could change the outcome:**
  1. Have Tier 1 explain the memory path, which should raise fidelity.
  2. Measure stability on edge-identity overlap instead of scores.
  Both are TGN-side changes, not made here.

No comparator: reward decomposition (0.9 ms, exact); explanation bundle within the 200 ms bound (37.8 ms
mean, p95 84.6 ms); AIR's analyst efficiency (needs a defined "manual override", teammate's test).

## 4. Response policy (zone Double DQN vs SOAR playbooks, CALDERA-driven environment)

| Metric | XTGAIA | Static playbook | Graded rules | Results |
|---|---|---|---|---|
| Mean return | 12.99 | 7.94 | 9.81 | 2 wins |
| Containment rate | 1.000 | 1.000 | 1.000 | 2 ties |
| Steps to contain | 1.67 | 1.54 | 1.62 | 2 losses (counted as 1 row) |
| False containment | 0.000 | 0.000 | 0.000 | 2 ties |
| Disruption | 0.050 | 1.388 | 0.729 | 2 wins |
| Cost | 0.41 | 0.87 | 0.57 | 2 wins |
| MTTR vs [2] | 307 ms | 1,800 ms | – | win |

## 5. Open items from the updated spec (section 6 tasks)

| Spec task | Status | Blocked by |
|---|---|---|
| 9. Loader tagging rows with unit and training regime | done (`configs/baselines.yaml`, `src/evaluation/baseline_refs.py`) | – |
| 10. MCC + TP/FP/TN/FN under strict and lenient labels | done on CICIDS Friday with the tuned model: node F1 0.134 lenient / 0.082 strict, MCC 0.18 / 0.13, AUC 0.95 / 0.94 (`runs/final_metrics/`); UNSW from the design check | – |
| 11. Non-duplicate-flow column in every flow table | done for the TGN on CICIDS Friday: F1 0.956 non-duplicate vs 0.960 all (23.4 % duplicates); UNSW from the design check | baselines' non-duplicate rows not rerun |
| 12. 15-minute window metrics from node scores | done on CICIDS Friday: 33 windows, every one flagged at ω (max roll-up), so precision 0.42, MCC 0, AUC 0.69; the window unit saturates on this capture. Not applicable to UNSW (no timestamps) | – |
| 13. E-XAI six metrics, ≥ 1,000 instances | done for latency, descriptive accuracy (k grid), sparsity (τ grid), cosine and top-k overlap stability on 1,000 instances; XTGAIA on 1,199 detections. Robustness / completeness not run (optional in the spec) | – |
| 5. Throughput and hardware log | laptop baseline done (353–448 events/s, hardware logged) | teammate's hardware run |
| TGN under P-CIC split A (for [3]) | not done | compute (≈ 5 h) |
| ≥ 5 TGN seeds | 1 seed | compute (≈ 5–10 h per dataset) |
| 14–16. Provenance runs, PDF extraction, leads | out of scope (future work / reading task) | – |
