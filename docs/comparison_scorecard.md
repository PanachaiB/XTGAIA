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
| System latency (IV-D) | 6 | 3 | 0 | 3 | **No: split**, hardware-dependent (teammate re-measures) |
| Explainability (IV-E) | 18 | 10 | 0 | 8 | Yes, narrowly; fidelity is a clear weakness |
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

Measured on one 16-vCPU workstation (AIR: 4 × Dell R760 + A100). Processing-only times.

| Stage | XTGAIA mean | Comparator | Result |
|---|---|---|---|
| Acquisition (per event) | 0.0 ms | AIR 8.1 ms | win |
| Analysis (TGN + risk, per micro-batch) | 31.5 ms | AIR 15.2 ms | loss |
| Decision (state + Double DQN) | 19.6 ms | AIR 23.5 ms | win |
| Execution (SOAR → enforcement → G_t+1) | 191.9 ms | AIR 11.5 ms | loss |
| End-to-end closed loop | 349.9 ms | AIR 58.3 ms | loss |
| MTTR (detection → enforcement) | 307.1 ms | [2] 1,800 ms | win |

Not yet measured: **throughput** (AIR 18,540 events/s). `scripts/latency_bench.py` now records events/s,
wall time and hardware.

Notes for the teammate's run:
- Execution is dominated by the HTTP round trip to the local Shuffle webhook; it is the main reason the
  end-to-end time is above AIR's. Measure on the paper's hardware with the SOC stack on the same host.
- MTTD has no published comparator ([5] and [2] do not report it).
- Smoke test of `scripts/latency_bench.py` without the SOC services (2 CPU threads, training running
  alongside): end-to-end 57.9 ms (AIR 58.3), analysis 18.6 ms, decision 5.5 ms, execution 0.6 ms, X_t 24.6 ms,
  355 events/s. So the model and graph path already matches AIR on a laptop. The losses above come from the
  Shuffle HTTP round trip and from the earlier demo run's heavier load, not from the TGN or the policy.

## 3. Explainability (all comparator values measured by us; no paper publishes them)

XTGAIA Tier 1+2 (PGExplainer + attention rollout on the TGN) against post-hoc explainers on [3]'s models.

| Explainer | Latency (ms) vs XTGAIA 37.5 | Fidelity vs XTGAIA 0.001 | Stability vs XTGAIA 1.000 / 0.984 |
|---|---|---|---|
| LIME (RF) 231.1 / 0.288 / 0.772 | win | loss | win |
| TreeSHAP (RF) 4.7 / 0.324 / 0.966 | loss | loss | win |
| LIME (XGB) 187.0 / 0.378 / 0.638 | win | loss | win |
| TreeSHAP (XGB) 1.7 / 0.847 / 0.953 | loss | loss | win |
| LIME (1D-CNN) 339.1 / 0.099 / 0.826 | win | loss | win |
| KernelSHAP (1D-CNN) 2302.5 / 0.360 / 0.727 | win | loss | win |

10 wins, 8 losses. Caveats:
- The fidelity definitions differ: fidelity+ on the neighbourhood subgraph vs top-5 feature deletion.
  XTGAIA's ≈ 0 fidelity is a real weakness: p_t depends mostly on the node's own memory (A1).
- XTGAIA's stability rests on 12 same-class pairs.
- The spec's E-XAI [N5] protocol (six metrics, ≥ 1,000 instances) is not done yet; see the open items.

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
| 9. Loader tagging rows with unit and training regime | not done | nothing (code only) |
| 10. MCC + TP/FP/TN/FN under strict and lenient labels | done for the design check (`runs/design_check/`); not in the results doc | final TGN checkpoint |
| 11. Non-duplicate-flow column in every flow table | done once for the design check; not in the results doc | final TGN checkpoint, RAM |
| 12. 15-minute window metrics from node scores | not done | final TGN checkpoint |
| 13. E-XAI six metrics, ≥ 1,000 instances | partial (300 instances; latency, top-5 fidelity, cosine stability) | RAM while training runs |
| 5. Throughput and hardware log | added to `scripts/latency_bench.py` | teammate's hardware run |
| TGN under P-CIC split A (for [3]) | not done | compute (≈ 5 h) |
| ≥ 5 TGN seeds | 1 seed | compute (≈ 5–10 h per dataset) |
| 14–16. Provenance runs, PDF extraction, leads | out of scope (future work / reading task) | – |
