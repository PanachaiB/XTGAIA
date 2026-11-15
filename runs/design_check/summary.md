# Design check summary (flow-level head)

Rules: `docs/design_check_rules.md`. Table `data/processed/eval/tgnv2_nf_un_chrono_train_s0.csv`, checkpoint `data/processed/eval/tgnv2_nf_un_chrono_s0.pt`. Validation metrics unless marked test.

## C1: current TGN, node p_t projected onto flows (validation)

| Projection | F1 | AUC | TPR@0.1%FPR | TPR@1%FPR | FPR | ω |
|---|---|---|---|---|---|---|
| dst | 0.8560 | 0.9954 | 0.1948 | 0.8509 | 0.0077 | 0.9938 |
| src | 0.8128 | 0.9942 | 0.2106 | 0.7815 | 0.0100 | 0.9937 |
| max | 0.8822 | 0.9957 | 0.2031 | 0.8938 | 0.0087 | 0.9942 |

Node level (same model): roll-up **max**, node AUC 1.0000 over 40 nodes (14 malicious); entity detection 1.000, benign-node false alarms 0.000, median time to first detection 0.0 flows.

## C2: in-house MLP, same flow features / split / threshold rule (validation)

| Seed | F1 | AUC | TPR@0.1%FPR | TPR@1%FPR | FPR |
|---|---|---|---|---|---|
| 0 | 0.9547 | 0.9991 | 0.7155 | 1.0000 | 0.0046 |
| 1 | 0.9547 | 0.9989 | 0.6493 | 1.0000 | 0.0065 |
| 2 | 0.9543 | 0.9988 | 0.5807 | 0.9999 | 0.0065 |

Mean F1 0.9546 ± 0.0002 (noise band 0.0050).

## Decisions

| Time | Rule | Outcome |
|---|---|---|
| 2026-11-15 14:40:48 | R1 | MLP clearly below ~0.97: pipeline/protocol problem; architecture work stopped, existing job queued |
| 2026-11-15 14:40:49 | Test (once) | current design (node p_t, destination projection) |

## Test split (touched once, final configuration)

current design (node p_t, destination projection): F1 0.8487, AUC 0.9950, TPR@1%FPR 0.8452, TPR@0.1%FPR 0.2396, FPR 0.0086.

## Next job

`sh scripts/retrain_role2.sh` (XTGAIA_FLOW_HEAD=off), started 2026-11-15 14:40:49.

## C3 override run (R1 overridden by the user; R4 unchanged; validation only)

Recorded in B10 before running. Noise = max(seed sd, 95 % bootstrap half-width over validation flows, 0.005).

### Sanity diagnostics

| MLP seed | F1 budget-tuned (ω) | FPR | F1 at fixed 0.5 | FPR at 0.5 | reproduces C2 |
|---|---|---|---|---|---|
| 0 | 0.9546 (0.8891) | 0.0065 | 0.9394 | 0.0091 | no |
| 1 | 0.9547 (0.8856) | 0.0065 | 0.9377 | 0.0093 | no |
| 2 | 0.9544 (0.9188) | 0.0065 | 0.9343 | 0.0099 | no |

Parity: same table, split ({'train': 207571, 'val': 239028, 'test': 478055}), the same 41 flow features for the MLP and the TGN messages, shared threshold rule `src.role2_ai.train_tgn.calibrate_threshold (shared)`.
Duplicates: 5.1% of validation flows have an identical feature vector in train (19.9% of validation attacks; label agreement with the train majority 100.0%); 0.0% including endpoints. Within-split duplicate share: train 6.6%, validation 3.3%.
Headroom: retrain uses 14 threads on the 16-vCPU VM; C3 capped at 2.

### Flow level (validation; flow labels)

| Arm | Seed | F1 | AUC | TPR@0.1/1/5 %FPR | Precision | MCC | TP/FP/TN/FN |
|---|---|---|---|---|---|---|---|
| node-only | - | 0.8560 | 0.9954 | 0.195 / 0.851 / 1.000 | 0.8839 | 0.8467 | 13045/1713/221594/2676 |
| flow-only | 0 | 0.9488 | 0.9984 | 0.466 / 0.999 / 1.000 | 0.9177 | 0.9457 | 15438/1384/221923/283 |
| flow-only | 1 | 0.9523 | 0.9983 | 0.432 / 1.000 / 1.000 | 0.9180 | 0.9496 | 15554/1390/221917/167 |
| flow-only | 2 | 0.9500 | 0.9984 | 0.478 / 1.000 / 1.000 | 0.9199 | 0.9469 | 15440/1344/221963/281 |
| hybrid | 0 | 0.9488 | 0.9984 | 0.466 / 0.999 / 1.000 | 0.9177 | 0.9457 | 15438/1384/221923/283 |
| hybrid | 1 | 0.9523 | 0.9983 | 0.432 / 1.000 / 1.000 | 0.9180 | 0.9496 | 15554/1390/221917/167 |
| hybrid | 2 | 0.9500 | 0.9984 | 0.478 / 1.000 / 1.000 | 0.9199 | 0.9469 | 15440/1344/221963/281 |

### Node level (validation; (node, 10,000-flow window) pairs)

| Arm | Seed | Label | Roll-up | F1 | AUC | TPR@0.1/1/5 %FPR | Precision | MCC | TP/FP/TN/FN |
|---|---|---|---|---|---|---|---|---|---|
| node-only | - | lenient | max | 1.0000 | 1.0000 | 1.000 / 1.000 / 1.000 | 1.0000 | 1.0000 | 336/0/549/0 |
| node-only | - | strict | max | 0.1284 | 0.8705 | 0.000 / 0.073 / 0.302 | 0.5385 | 0.1688 | 7/6/783/89 |
| hybrid | 0 | lenient | max | 1.0000 | 1.0000 | 1.000 / 1.000 / 1.000 | 1.0000 | 1.0000 | 336/0/549/0 |
| hybrid | 0 | strict | max | 0.1111 | 0.9067 | 0.000 / 0.062 / 0.354 | 0.5000 | 0.1476 | 6/6/783/90 |
| hybrid | 1 | lenient | max | 1.0000 | 1.0000 | 1.000 / 1.000 / 1.000 | 1.0000 | 1.0000 | 336/0/549/0 |
| hybrid | 1 | strict | max | 0.1111 | 0.9017 | 0.000 / 0.062 / 0.323 | 0.5000 | 0.1476 | 6/6/783/90 |
| hybrid | 2 | lenient | max | 1.0000 | 1.0000 | 1.000 / 1.000 / 1.000 | 1.0000 | 1.0000 | 336/0/549/0 |
| hybrid | 2 | strict | max | 0.1273 | 0.8971 | 0.000 / 0.010 / 0.333 | 0.5000 | 0.1596 | 7/7/782/89 |

### Entity level (validation; flow threshold ω of each arm)

| Arm | Seed | Label | Malicious nodes | Entity detection | Median TTFD (flows) | Mean nodes flagged per attack category |
|---|---|---|---|---|---|---|
| node-only | - | lenient | 14 | 1.000 | 0.0 | 13.7 |
| node-only | - | strict | 4 | 1.000 | 0.0 | 13.7 |
| flow-only | 0 | lenient | 14 | 1.000 | 0.0 | 13.7 |
| flow-only | 0 | strict | 4 | 1.000 | 0.0 | 13.7 |
| flow-only | 1 | lenient | 14 | 1.000 | 0.0 | 13.7 |
| flow-only | 1 | strict | 4 | 1.000 | 0.0 | 13.7 |
| flow-only | 2 | lenient | 14 | 1.000 | 0.0 | 13.7 |
| flow-only | 2 | strict | 4 | 1.000 | 0.0 | 13.7 |
| hybrid | 0 | lenient | 14 | 1.000 | 0.0 | 13.7 |
| hybrid | 0 | strict | 4 | 1.000 | 0.0 | 13.7 |
| hybrid | 1 | lenient | 14 | 1.000 | 0.0 | 13.7 |
| hybrid | 1 | strict | 4 | 1.000 | 0.0 | 13.7 |
| hybrid | 2 | lenient | 14 | 1.000 | 0.0 | 13.7 |
| hybrid | 2 | strict | 4 | 1.000 | 0.0 | 13.7 |

False alerts are reported per 100k validation flows (the dataset has no wall clock, so 'per day' is not available):

- node-only seed -: 717 false flow alerts per 100k flows; 0.0 false node alerts (lenient) per 100k flows; 2.5 false node alerts (strict) per 100k flows
- flow-only seed 0: 579 false flow alerts per 100k flows
- flow-only seed 1: 582 false flow alerts per 100k flows
- flow-only seed 2: 562 false flow alerts per 100k flows
- hybrid seed 0: 579 false flow alerts per 100k flows; 0.0 false node alerts (lenient) per 100k flows; 2.5 false node alerts (strict) per 100k flows
- hybrid seed 1: 582 false flow alerts per 100k flows; 0.0 false node alerts (lenient) per 100k flows; 2.5 false node alerts (strict) per 100k flows
- hybrid seed 2: 562 false flow alerts per 100k flows; 0.0 false node alerts (lenient) per 100k flows; 2.9 false node alerts (strict) per 100k flows

### Roll-up saturation check (degree vs score, all roll-ups)

**node-only** (seed -), validation lenient AUC per roll-up: max 1.0000, noisy_or 0.9945, topk_mean 1.0000

| Roll-up | Label | n | Spearman(degree, score) | Mean score Q1→Q4 degree | Share ≥ 0.99 |
|---|---|---|---|---|---|
| max | benign | 549 | -0.324 | 0.489 → 0.000 → 0.000 → 0.000 | 0.000 |
| max | malicious | 336 | 0.059 | 0.997 → 0.997 → 0.997 → 0.996 | 0.976 |
| noisy_or | benign | 549 | 0.232 | 0.495 → 0.000 → 0.000 → 0.000 | 0.069 |
| noisy_or | malicious | 336 | nan | 1.000 → 1.000 → 1.000 → 1.000 | 1.000 |
| topk_mean | benign | 549 | -0.281 | 0.489 → 0.000 → 0.000 → 0.000 | 0.000 |
| topk_mean | malicious | 336 | 0.213 | 0.997 → 0.997 → 0.997 → 0.996 | 0.970 |

**hybrid** (seed 0), validation lenient AUC per roll-up: max 1.0000, noisy_or 0.9945, topk_mean 1.0000

| Roll-up | Label | n | Spearman(degree, score) | Mean score Q1→Q4 degree | Share ≥ 0.99 |
|---|---|---|---|---|---|
| max | benign | 549 | -0.245 | 0.179 → 0.000 → 0.000 → 0.000 | 0.011 |
| max | malicious | 336 | 0.350 | 1.000 → 1.000 → 1.000 → 1.000 | 1.000 |
| noisy_or | benign | 549 | 0.255 | 0.224 → 0.000 → 0.000 → 0.000 | 0.015 |
| noisy_or | malicious | 336 | nan | 1.000 → 1.000 → 1.000 → 1.000 | 1.000 |
| topk_mean | benign | 549 | -0.203 | 0.163 → 0.000 → 0.000 → 0.000 | 0.000 |
| topk_mean | malicious | 336 | 0.603 | 0.999 → 1.000 → 1.000 → 1.000 | 1.000 |

### Cost and XAI

| Arm | Params | ms / flow | ms / 128-flow batch | ms / 10k-flow window | PGExplainer target = risk/DQN input | Explains the alert score |
|---|---|---|---|---|---|---|
| node-only | 0 | 0.0000 | 0.00 | 0.0 | yes | yes |
| flow-only | 79,233 | 0.0011 | 0.14 | 12.3 | yes | no |
| hybrid | 79,233 | 0.0011 | 0.14 | 12.3 | yes | no |

matches only while risk / DQN keep consuming p_t; if the rolled-up node score replaced p_t there, PGExplainer would explain a different quantity

### R4 (criteria unchanged)

Gain (hybrid − node-only, validation flow F1) +0.0944 vs 2 × noise 0.0100; entity detection (lenient) {'hybrid': 1.0, 'node-only': 1.0}; median TTFD (lenient) {'hybrid': 0.0, 'node-only': 0.0}; head 0.14 ms / 128 flows. Checks: gain True, entity True, TTFD True, latency True. **Hybrid adopted.**

Noise per arm: node-only: seed sd n/a (single checkpoint), bootstrap 0.0044 -> noise 0.0050; flow-only: seed sd 0.0018, bootstrap 0.0022 -> noise 0.0050; hybrid: seed sd 0.0018, bootstrap 0.0022 -> noise 0.0050
