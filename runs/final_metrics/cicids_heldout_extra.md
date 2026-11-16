# Extra detection views: tuned TGN, CICIDS2017 held-out Friday

Checkpoint `data/models/tgn_best.pt` (00b3a95a41623a36), max read-out, threshold ω = 0.9693 (validation-calibrated, not tuned on Friday).

| View | Unit | n | Prevalence | Precision | Recall | F1 | FPR | MCC | AUC | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_all | flow | 547,557 | 0.466 | 0.9251 | 0.9965 | 0.9595 | 0.0703 | 0.9241 | 0.9965 | 254063 | 20569 | 272042 | 883 |
| flow_non_duplicate | flow (non-duplicate) | 419,170 | 0.426 | 0.9185 | 0.9956 | 0.9555 | 0.0656 | 0.9224 | 0.9963 | 177827 | 15786 | 224770 | 787 |
| window_15min | 15-min window | 33 | 0.424 | 0.4242 | 1.0000 | 0.5957 | 1.0000 | 0.0000 | 0.6917 | 14 | 19 | 0 | 0 |
| node_lenient | host × 15 min, lenient | 24,020 | 0.002 | 0.0795 | 0.4318 | 0.1343 | 0.0092 | 0.1821 | 0.9526 | 19 | 220 | 23756 | 25 |
| node_strict | host × 15 min, strict | 24,020 | 0.001 | 0.0460 | 0.3667 | 0.0818 | 0.0095 | 0.1271 | 0.9444 | 11 | 228 | 23762 | 19 |

Duplicates removed: 128,387 flows (23.4%), 59.5% of them attacks.
Hosts flagged per attack window: 9.9; false node alerts per day: 680 (strict) / 656 (lenient) over a 8.0 h capture.

**XTGAIA Tier 1+2 explanations at detection time** (1,199 detections, 1,105 on attack flows): latency mean 85.9 ms, p95 217.5 ms, std 491.6 ms (93.7% within 200 ms) | fidelity+ -0.0 | fidelity- 2.8864754747608685e-06 | sparsity 0.8749615512927439 | cosine stability (same attack class) {'structural': 1.0, 'temporal': 1.0, 'strategic': None} over {'structural': 10000, 'temporal': 10000, 'strategic': 0} pairs.
