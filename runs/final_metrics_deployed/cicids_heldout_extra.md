# Extra detection views: deployed (pre-tuning) TGN, CICIDS2017 held-out Friday

Checkpoint `data/models/tgn_best.pt` (1773d9440fff566c), dst read-out, threshold ω = 0.9290 (validation-calibrated, not tuned on Friday).

| View | Unit | n | Prevalence | Precision | Recall | F1 | FPR | MCC | AUC | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow_all | flow | 547,557 | 0.466 | 0.9462 | 0.9950 | 0.9700 | 0.0493 | 0.9436 | 0.9979 | 253660 | 14423 | 278188 | 1286 |
| flow_non_duplicate | flow (non-duplicate) | 419,170 | 0.426 | 0.9271 | 0.9931 | 0.9590 | 0.0580 | 0.9285 | 0.9975 | 177390 | 13941 | 226615 | 1224 |
| window_15min | 15-min window | 33 | 0.424 | 0.4242 | 1.0000 | 0.5957 | 1.0000 | 0.0000 | 0.7970 | 14 | 19 | 0 | 0 |
| node_lenient | host × 15 min, lenient | 24,020 | 0.002 | 0.0223 | 0.8864 | 0.0436 | 0.0712 | 0.1343 | 0.9726 | 39 | 1706 | 22270 | 5 |
| node_strict | host × 15 min, strict | 24,020 | 0.001 | 0.0149 | 0.8667 | 0.0293 | 0.0717 | 0.1082 | 0.9678 | 26 | 1719 | 22271 | 4 |

Duplicates removed: 128,387 flows (23.4%), 59.5% of them attacks.
Hosts flagged per attack window: 74.1; false node alerts per day: 5127 (strict) / 5088 (lenient) over a 8.0 h capture.

**XTGAIA Tier 1+2 explanations at detection time** (1,200 detections, 1,016 on attack flows): latency mean 25.2 ms, p95 51.8 ms, std 17.7 ms (99.8% within 200 ms) | fidelity+ 0.002 | fidelity- 0.017637923533717792 | sparsity 0.8740995833333333 | cosine stability (same attack class) {'structural': 1.0, 'temporal': 0.9964, 'strategic': None} over {'structural': 10000, 'temporal': 10000, 'strategic': 0} pairs.

**Measurement fix (log-odds fidelity, edge-identity stability).** fidelity+ (logit) mean -0.014438278737361981 / median 0.0; fidelity- (logit) mean 0.13969549664777978; k random edges removed (logit) mean -0.004013567137878627 / median 0.0; top-k beats random in 0.075 of explanations | edge-identity stability (Jaccard of top-5 edges, consecutive detections of the same host and class) 0.9836 over 1012 pairs.
