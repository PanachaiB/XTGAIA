# Design check summary (flow-level head)

Rules: `docs/design_check_rules.md`. Table `data/processed/eval/tgnv2_nf_un_chrono_train_s0.csv`, checkpoint `data/processed/eval/tgnv2_nf_un_chrono_s0.pt`. Validation metrics unless marked test.

## C1: current TGN, node p_t projected onto flows (validation)

| Projection | F1 | AUC | TPR@0.1%FPR | TPR@1%FPR | FPR | ω |
|---|---|---|---|---|---|---|
| dst | 0.8538 | 0.9954 | 0.1948 | 0.8509 | 0.0100 | 0.9932 |
| src | 0.8126 | 0.9942 | 0.2106 | 0.7815 | 0.0100 | 0.9937 |
| max | 0.8779 | 0.9957 | 0.2031 | 0.8938 | 0.0100 | 0.9940 |

Node level (same model): roll-up **max**, node AUC 1.0000 over 40 nodes (14 malicious); entity detection 1.000, benign-node false alarms 0.000, median time to first detection 0.0 flows.

## C2: in-house MLP, same flow features / split / threshold rule (validation)

| Seed | F1 | AUC | TPR@0.1%FPR | TPR@1%FPR | FPR |
|---|---|---|---|---|---|
| 0 | 0.9337 | 0.9991 | 0.7155 | 1.0000 | 0.0100 |
| 1 | 0.9337 | 0.9989 | 0.6493 | 1.0000 | 0.0100 |
| 2 | 0.9336 | 0.9988 | 0.5807 | 0.9999 | 0.0100 |

Mean F1 0.9336 ± 0.0000 (noise band 0.0050).

## Decisions

| Time | Rule | Outcome |
|---|---|---|
| 2026-11-15 14:13:14 | R1 | MLP clearly below ~0.97: pipeline/protocol problem; architecture work stopped, existing job queued |
| 2026-11-15 14:13:15 | Test (once) | current design (node p_t, destination projection) |

## Test split (touched once, final configuration)

current design (node p_t, destination projection): F1 0.8566, AUC 0.9950, TPR@1%FPR 0.8452, TPR@0.1%FPR 0.2396, FPR 0.0095.

## Next job

`sh scripts/retrain_role2.sh` (XTGAIA_FLOW_HEAD=off), started 2026-11-15 14:13:16.
