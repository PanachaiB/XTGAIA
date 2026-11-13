# XTGAIA evaluation results

Generated 2026-11-13 20:42 UTC from `data/processed/eval/results.jsonl` (23 runs). Values are mean ± 95% t-interval half-width over seeds (no interval when a single seed finished).

**Dataset:** CICIDS2017 *improved* (Engelen, Rimmer & Joosen, WTMC 2021; Liu et al., CNS 2022): flows regenerated from the original PCAPs with a fixed CICFlowMeter and relabelled. Flows labelled *Attempted* (attack traffic without malicious effect) are treated as BENIGN, as the authors recommend. Identifiers (flow id, IPs, source port, timestamp) are excluded from flow-level features.

**Thresholds:** Random Forest, XGBoost and E-GraphSAGE use 0.5. The TGN uses the threshold calibrated on its own validation split (max-F1 within a 1% validation FPR budget). ROC-AUC, PR-AUC and TPR@FPR are threshold-free.

## Protocol: random

Random stratified 70/30 split over all five days (comparable with most CICIDS2017 papers; optimistic, temporally adjacent flows fall in both train and test)

| Model | Seeds | Precision | Recall | F1 | FPR | ROC-AUC | PR-AUC | TPR@FPR≤1% | TPR@FPR≤0.1% |
|---|---|---|---|---|---|---|---|---|---|
| Random Forest | 3 | 0.9999 ± 0.0000 | 0.9999 ± 0.0000 | 0.9999 ± 0.0000 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| XGBoost | 3 | 0.9999 ± 0.0000 | 0.9999 ± 0.0001 | 0.9999 ± 0.0000 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| E-GraphSAGE | 3 | 0.9663 ± 0.0064 | 0.9813 ± 0.0022 | 0.9737 ± 0.0021 | 0.0109 ± 0.0021 | 0.9968 ± 0.0001 | 0.9836 ± 0.0019 | 0.9808 ± 0.0002 | 0.2562 ± 0.1126 |

Per-class detection rate at the model's threshold (mean over seeds) and benign flag rate:

| Model | Botnet | DDoS | DoS GoldenEye | DoS Hulk | DoS Slowhttptest | DoS Slowloris | FTP-Patator | Heartbleed | Infiltration | Infiltration - Portscan | Portscan | SSH-Patator | Web Attack - Brute Force | Web Attack - SQL Injection | Web Attack - XSS | Benign flagged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Random Forest | 1.000 | 1.000 | 0.999 | 1.000 | 0.995 | 0.999 | 1.000 | 1.000 | 0.727 | 1.000 | 1.000 | 1.000 | 1.000 | 0.250 | 1.000 | 0.0000 |
| XGBoost | 1.000 | 1.000 | 1.000 | 1.000 | 0.997 | 1.000 | 1.000 | 0.889 | 0.818 | 1.000 | 1.000 | 0.999 | 1.000 | 0.750 | 1.000 | 0.0000 |
| E-GraphSAGE | 0.339 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.015 | 0.000 | 0.000 | 0.981 | 0.996 | 0.000 | 1.000 | 1.000 | 1.000 | 0.0109 |

Train/test sizes: 1,469,983/629,993.
- trained on stratified subsample of 599,998/1,469,983 edges

## Protocol: heldout

Held-out day: train Monday–Thursday, test Friday (no temporal leakage; Botnet, PortScan, DDoS on the unseen day)

| Model | Seeds | Precision | Recall | F1 | FPR | ROC-AUC | PR-AUC | TPR@FPR≤1% | TPR@FPR≤0.1% |
|---|---|---|---|---|---|---|---|---|---|
| Random Forest | 3 | 1.0000 ± 0.0000 | 0.9297 ± 0.0979 | 0.9633 ± 0.0530 | 0.0000 ± 0.0000 | 0.9998 ± 0.0002 | 0.9998 ± 0.0002 | 0.9922 ± 0.0008 | 0.9918 ± 0.0007 |
| XGBoost | 3 | 0.9981 ± 0.0000 | 0.9883 ± 0.0002 | 0.9932 ± 0.0001 | 0.0016 ± 0.0000 | 0.9998 ± 0.0000 | 0.9998 ± 0.0000 | 0.9925 ± 0.0015 | 0.9764 ± 0.0173 |
| E-GraphSAGE | 3 | 0.9872 ± 0.0091 | 0.8736 ± 0.5284 | 0.9173 ± 0.3248 | 0.0094 ± 0.0003 | 0.9945 ± 0.0010 | 0.9860 ± 0.0094 | 0.9965 ± 0.0000 | 0.1021 ± 0.4391 |
| TGN (Role 2) | 5 | 0.5083 ± 0.6168 | 0.5985 ± 0.6784 | 0.5410 ± 0.6292 | 0.1453 ± 0.3779 | 0.8952 ± 0.1961 | 0.8315 ± 0.2869 | 0.5983 ± 0.6781 | 0.2235 ± 0.2533 |

Per-class detection rate at the model's threshold (mean over seeds) and benign flag rate:

| Model | Botnet | DDoS | Portscan | Benign flagged |
|---|---|---|---|---|
| Random Forest | 0.000 | 0.945 | 0.925 | 0.0000 |
| XGBoost | 0.000 | 0.995 | 0.989 | 0.0016 |
| E-GraphSAGE | 0.000 | 0.672 | 0.998 | 0.0094 |
| TGN (Role 2) | 0.090 | 0.600 | 0.600 | 0.1453 |

Train/test sizes: 190,052/547,557, 190,210/547,557, 190,423/547,557, 190,630/547,557, 190,651/547,557, 1,552,419/547,557.
- trained on stratified subsample of 599,998/1,552,419 edges

## Caveats

- Random-split numbers are not comparable with held-out-day numbers; papers that report only random splits on CICIDS2017 are optimistic.
- E-GraphSAGE trains on a stratified edge subsample for memory reasons (noted per run); it is evaluated on the full test set.
- The TGN is trained on Role 2's subsampled training table (10% of benign flows, at most 20k flows per attack label and day) and evaluated by streaming all Friday flows through the live inference engine.
- Cross-dataset evaluation (CICIDS2017 → CSE-CIC-IDS2018) is not included yet.

## Findings (overnight run, 14 Nov 2026)

1. **Random split is saturated.** RF and XGBoost reach F1 0.9999 on the random split, in line with the optimistic
   CICIDS2017 literature. Results on the held-out day are the ones that discriminate between models.
2. **Held-out day: XGBoost is the strongest stable baseline** (F1 0.9932 ± 0.0001, TPR@FPR≤1% 0.9925).
   Random Forest is equally good at ranking (AUC 0.9998) but its recall at a 0.5 threshold varies by seed.
3. **Novel attacks are missed by every supervised flow model.** Botnet traffic never appears Monday–Thursday, and
   RF, XGBoost and E-GraphSAGE all detect 0% of it on Friday.
4. **The current Role 2 TGN is unstable across seeds.** Per-seed results on held-out Friday:

   | Seed | Threshold ω (val) | F1 | ROC-AUC | TPR@FPR≤1% | Failure mode |
   |---|---|---|---|---|---|
   | 0 | 0.0260 | 0.716 | 0.636 | 0.000 | training collapse (near-random ranking) |
   | 1 | 0.0127 | 0.995 | 0.996 | 0.997 | – (beats XGBoost) |
   | 2 | 0.2900 | 0.995 | 0.996 | 0.997 | – (beats XGBoost) |
   | 3 | 0.0221 | 0.000 | 0.850 | 0.000 | weak ranking + threshold above all scores |
   | 4 | 0.0197 | 0.000 | 0.998 | 0.997 | threshold miscalibration only |

   When training converges, the TGN matches or exceeds the best baseline (seeds 1, 2, 4 rank Friday almost
   perfectly). Two failure modes cause the large variance:
   - **Training collapse** (seeds 0, 3), consistent with the very noisy per-epoch loss: node-level scoring from a
     single memory slot, attention over a sequence of length 1, and day-ordered batches.
   - **Threshold calibration fails**: Role 2's 70/15/15 temporal split puts late Thursday in validation, which
     contains essentially none of Friday's attack types (validation F1 ≤ 0.06 for every seed), so ω is
     meaningless. Seed 4 shows it directly: AUC 0.998, yet F1 0 because ω is above every Friday score.
5. **Implications for the TGN redesign** (ordered by expected impact): per-day stratified validation so every
   attack family is present when calibrating; edge-level scoring [Z_src ‖ Z_dst ‖ flow features] with temporal
   neighbour attention; early stopping on validation AUC with several restarts; then probability calibration.
