# XTGAIA detection evaluation

Generated 2026-11-15 02:10 UTC from `data/processed/eval/results.jsonl` (58 runs). Mean ± 95% t-interval half-width over seeds (no interval for a single seed). Default-threshold metrics use 0.5 for baselines and the validation-calibrated ω for the TGN; *fixed-FPR* and *weighted-threshold* tables choose every model's threshold on its validation set as in CCSTGN [11]. Precision, accuracy and F1 depend on test prevalence (given per section); TPR/FNR at fixed FPR and ROC-AUC do not.

**Datasets.** CICIDS2017 *improved* (Engelen et al. 2021 / Liu et al. 2022; *Attempted* → BENIGN). NF-UNSW-NB15-v2 (Sarhan et al.; 2,390,275 flows, 3.98 % attack).

## CICIDS2017 (corrected), P-CIC split A — random 80/20, benign downsampled in training only, 10% validation (parity protocol for Vedesh et al. [3])

Test prevalence (attack share): 24.07%.

| Model | Seeds | Acc | Precision | Recall | F1 | FPR | ROC-AUC | TPR@FPR≤1% | TPR@FPR≤0.1% |
|---|---|---|---|---|---|---|---|---|---|
| Random Forest | 4 | 0.9999 ± 0.0000 | 0.9998 ± 0.0000 | 0.9999 ± 0.0000 | 0.9999 ± 0.0000 | 0.0001 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| XGBoost | 4 | 0.9999 ± 0.0000 | 0.9998 ± 0.0001 | 0.9999 ± 0.0000 | 0.9998 ± 0.0000 | 0.0001 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| MLP | 3 | 0.9994 ± 0.0001 | 0.9989 ± 0.0011 | 0.9988 ± 0.0007 | 0.9988 ± 0.0003 | 0.0004 ± 0.0004 | 1.0000 ± 0.0000 | 0.9997 ± 0.0004 | 0.9993 ± 0.0002 |
| 1D-CNN | 3 | 0.9995 ± 0.0002 | 0.9993 ± 0.0001 | 0.9988 ± 0.0007 | 0.9991 ± 0.0004 | 0.0002 ± 0.0000 | 1.0000 ± 0.0000 | 0.9998 ± 0.0002 | 0.9994 ± 0.0003 |
| *RF [3]* | 1 | 0.9456 | 0.9412 | 0.9502 | 0.9457 | 0.0488 | – | – | – |
| *XGBoost [3]* | 1 | 0.9523 | 0.9501 | 0.9546 | 0.9523 | 0.0402 | – | – | – |
| *1D-CNN [3]* | 1 | 0.9634 | 0.9601 | 0.9667 | 0.9634 | 0.0301 | – | – | – |

*Reference rows:* Vedesh K.P. et al. [3], ICSADL 2026 — CICIDS2017 Kaggle repack (~100k flows), random 80/20, single run, test prevalence ≈ 43–45 % (derived). Tag: DIRECT after P-CIC (release differs).

**Missed detections at fixed FPR** (threshold chosen on validation for FPR closest to the target; F1 / FNR on test, as in CCSTGN [11] Table II):

| Model | F1 @0.01% | FNR @0.01% | F1 @0.1% | FNR @0.1% | F1 @1% | FNR @1% | test FPR @1% |
|---|---|---|---|---|---|---|---|
| Random Forest | 0.9998 ± 0.0001 | 0.0001 ± 0.0000 | 0.9983 ± 0.0001 | 0.0000 ± 0.0000 | 0.9838 ± 0.0005 | 0.0000 ± 0.0000 | 0.0104 ± 0.0003 |
| XGBoost | 0.9998 ± 0.0001 | 0.0000 ± 0.0000 | 0.9982 ± 0.0002 | 0.0000 ± 0.0000 | 0.9844 ± 0.0002 | 0.0000 ± 0.0000 | 0.0101 ± 0.0001 |
| MLP | 0.9987 ± 0.0003 | 0.0022 ± 0.0008 | 0.9980 ± 0.0006 | 0.0007 ± 0.0002 | 0.9843 ± 0.0016 | 0.0003 ± 0.0004 | 0.0100 ± 0.0010 |
| 1D-CNN | 0.9987 ± 0.0006 | 0.0022 ± 0.0012 | 0.9980 ± 0.0001 | 0.0006 ± 0.0003 | 0.9839 ± 0.0007 | 0.0002 ± 0.0002 | 0.0103 ± 0.0004 |

**Weighted threshold** (ω minimising 0.75·FNR + 0.25·FPR on validation, CCSTGN [11] Table I):

| Model | ROC-AUC | F1 | FPR | FNR |
|---|---|---|---|---|
| Random Forest | 1.0000 ± 0.0000 | 0.9998 ± 0.0001 | 0.0001 ± 0.0001 | 0.0001 ± 0.0001 |
| XGBoost | 1.0000 ± 0.0000 | 0.9999 ± 0.0000 | 0.0001 ± 0.0000 | 0.0001 ± 0.0000 |
| MLP | 1.0000 ± 0.0000 | 0.9980 ± 0.0009 | 0.0010 ± 0.0007 | 0.0007 ± 0.0004 |
| 1D-CNN | 1.0000 ± 0.0000 | 0.9979 ± 0.0021 | 0.0011 ± 0.0015 | 0.0006 ± 0.0006 |

Per-class detection rate at the default threshold (mean over seeds):

| Model | Botnet | DDoS | DoS GoldenEye | DoS Hulk | DoS Slowhttptest | DoS Slowloris | FTP-Patator | Heartbleed | Infiltration | Infiltration - Portscan | Portscan | SSH-Patator | Web Attack - Brute Force | Web Attack - SQL Injection | Web Attack - XSS | Benign flagged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Random Forest | 0.9983 ± 0.0054 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9971 ± 0.0065 | 0.9990 ± 0.0010 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.7500 ± 0.1137 | 0.9999 ± 0.0001 | 0.9999 ± 0.0001 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.3333 ± 0.4331 | 1.0000 ± 0.0000 | 0.0001 ± 0.0000 |
| XGBoost | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9986 ± 0.0046 | 0.9997 ± 0.0010 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.7857 ± 0.1312 | 0.9999 ± 0.0001 | 0.9999 ± 0.0001 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.5000 ± 0.5304 | 1.0000 ± 0.0000 | 0.0001 ± 0.0000 |
| MLP | 0.9161 ± 0.0851 | 0.9999 ± 0.0001 | 0.9925 ± 0.0140 | 0.9999 ± 0.0003 | 0.9943 ± 0.0124 | 0.9948 ± 0.0097 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.8095 ± 0.2049 | 0.9965 ± 0.0024 | 0.9994 ± 0.0003 | 0.9825 ± 0.0048 | 1.0000 ± 0.0000 | 0.2222 ± 0.4781 | 0.5000 ± 1.2421 | 0.0004 ± 0.0004 |
| 1D-CNN | 0.9456 ± 0.0845 | 0.9999 ± 0.0003 | 0.9941 ± 0.0124 | 1.0000 ± 0.0002 | 0.9923 ± 0.0082 | 0.9935 ± 0.0000 | 1.0000 ± 0.0000 | 0.6667 ± 1.4342 | 0.4762 ± 0.2049 | 0.9960 ± 0.0042 | 0.9994 ± 0.0003 | 0.9865 ± 0.0000 | 1.0000 ± 0.0000 | 0.1111 ± 0.4781 | 1.0000 ± 0.0000 | 0.0002 ± 0.0000 |

**Multi-class scoring (attack category).** TGN: hierarchical and consistent with its binary output (Benign when p_t < ω, otherwise the Z_t category head's most likely attack class); RF / XGBoost: flat multi-class classifiers on the same categories.

| Model | Accuracy | Macro-F1 (all classes) | Macro-F1 (attack classes) | Weighted-F1 |
|---|---|---|---|---|
| Random Forest | 0.9999 | 0.9511 | 0.9450 | 0.9999 |
| XGBoost | 0.9999 | 0.9721 | 0.9687 | 0.9999 |


## CICIDS2017 (corrected), held-out day — train Mon–Thu (last 15% of each day = validation), test Friday

Test prevalence (attack share): 46.56%.

| Model | Seeds | Acc | Precision | Recall | F1 | FPR | ROC-AUC | TPR@FPR≤1% | TPR@FPR≤0.1% |
|---|---|---|---|---|---|---|---|---|---|
| Random Forest | 4 | 0.9736 ± 0.0313 | 1.0000 ± 0.0000 | 0.9434 ± 0.0673 | 0.9705 ± 0.0360 | 0.0000 ± 0.0000 | 0.9998 ± 0.0001 | 0.9922 ± 0.0004 | 0.9918 ± 0.0004 |
| XGBoost | 4 | 0.9939 ± 0.0006 | 0.9986 ± 0.0015 | 0.9882 ± 0.0002 | 0.9934 ± 0.0007 | 0.0012 ± 0.0013 | 0.9999 ± 0.0000 | 0.9926 ± 0.0009 | 0.9800 ± 0.0145 |
| E-GraphSAGE | 3 | 0.9361 ± 0.2461 | 0.9872 ± 0.0091 | 0.8736 ± 0.5284 | 0.9173 ± 0.3248 | 0.0094 ± 0.0003 | 0.9945 ± 0.0010 | 0.9965 ± 0.0000 | 0.1021 ± 0.4391 |
| TGN (Role 2) | 5 | 0.7354 ± 0.2987 | 0.5083 ± 0.6168 | 0.5985 ± 0.6784 | 0.5410 ± 0.6292 | 0.1453 ± 0.3779 | 0.8952 ± 0.1961 | 0.5983 ± 0.6781 | 0.2235 ± 0.2533 |
| TGN (XTGAIA, temporal attention) | 1 | 0.9713 | 0.9462 | 0.9950 | 0.9700 | 0.0493 | 0.9979 | 0.9891 | 0.3696 |

**Missed detections at fixed FPR** (threshold chosen on validation for FPR closest to the target; F1 / FNR on test, as in CCSTGN [11] Table II):

| Model | F1 @0.01% | FNR @0.01% | F1 @0.1% | FNR @0.1% | F1 @1% | FNR @1% | test FPR @1% |
|---|---|---|---|---|---|---|---|
| Random Forest | 0.9887 | 0.0073 | 0.9889 | 0.0028 | 0.9836 | 0.0006 | 0.0286 |
| XGBoost | 0.9931 | 0.0076 | 0.9893 | 0.0023 | 0.9830 | 0.0002 | 0.0300 |
| TGN (XTGAIA, temporal attention) | 0.7486 | 0.4007 | 0.9873 | 0.0200 | 0.9690 | 0.0050 | 0.0512 |

**Weighted threshold** (ω minimising 0.75·FNR + 0.25·FPR on validation, CCSTGN [11] Table I):

| Model | ROC-AUC | F1 | FPR | FNR |
|---|---|---|---|---|
| Random Forest | 0.9998 | 0.9866 | 0.0231 | 0.0007 |
| XGBoost | 0.9999 | 0.9892 | 0.0186 | 0.0005 |
| TGN (XTGAIA, temporal attention) | 0.9979 | 0.9541 | 0.0817 | 0.0022 |

Per-class detection rate at the default threshold (mean over seeds):

| Model | Botnet | DDoS | Portscan | Benign flagged |
|---|---|---|---|---|
| Random Forest | 0.0000 ± 0.0000 | 0.9577 ± 0.1200 | 0.9392 ± 0.0535 | 0.0000 ± 0.0000 |
| XGBoost | 0.0000 ± 0.0000 | 0.9943 ± 0.0037 | 0.9892 ± 0.0021 | 0.0012 ± 0.0013 |
| E-GraphSAGE | 0.0000 ± 0.0000 | 0.6716 ± 1.4131 | 0.9985 ± 0.0017 | 0.0094 ± 0.0003 |
| TGN (Role 2) | 0.0905 ± 0.2512 | 0.6000 ± 0.6801 | 0.6000 ± 0.6801 | 0.1453 ± 0.3779 |
| TGN (XTGAIA, temporal attention) | 0.1196 | 0.9946 | 0.9992 | 0.0493 |

**Multi-class scoring (attack category).** TGN: hierarchical and consistent with its binary output (Benign when p_t < ω, otherwise the Z_t category head's most likely attack class); RF / XGBoost: flat multi-class classifiers on the same categories.

| Model | Accuracy | Macro-F1 (all classes) | Macro-F1 (attack classes) | Weighted-F1 |
|---|---|---|---|---|
| Random Forest | 0.8208 | 0.4955 | 0.3309 | 0.8171 |
| XGBoost | 0.8174 | 0.4858 | 0.3290 | 0.7977 |
| TGN (XTGAIA, temporal attention) | 0.7976 | 0.4305 | 0.2498 | 0.7375 |

- trained on stratified subsample of 599,998/1,552,419 edges

## CICIDS2017 (corrected), legacy random 70/30 (no validation set)

Test prevalence (attack share): n/a.

| Model | Seeds | Acc | Precision | Recall | F1 | FPR | ROC-AUC | TPR@FPR≤1% | TPR@FPR≤0.1% |
|---|---|---|---|---|---|---|---|---|---|
| Random Forest | 3 | 0.9999 ± 0.0000 | 0.9999 ± 0.0000 | 0.9999 ± 0.0000 | 0.9999 ± 0.0000 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| XGBoost | 3 | 1.0000 ± 0.0000 | 0.9999 ± 0.0000 | 0.9999 ± 0.0001 | 0.9999 ± 0.0000 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| E-GraphSAGE | 3 | 0.9873 ± 0.0011 | 0.9663 ± 0.0064 | 0.9813 ± 0.0022 | 0.9737 ± 0.0021 | 0.0109 ± 0.0021 | 0.9968 ± 0.0001 | 0.9808 ± 0.0002 | 0.2562 ± 0.1126 |

Per-class detection rate at the default threshold (mean over seeds):

| Model | Botnet | DDoS | DoS GoldenEye | DoS Hulk | DoS Slowhttptest | DoS Slowloris | FTP-Patator | Heartbleed | Infiltration | Infiltration - Portscan | Portscan | SSH-Patator | Web Attack - Brute Force | Web Attack - SQL Injection | Web Attack - XSS | Benign flagged |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Random Forest | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9991 ± 0.0011 | 1.0000 ± 0.0000 | 0.9949 ± 0.0027 | 0.9988 ± 0.0050 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.7273 ± 0.2258 | 0.9998 ± 0.0003 | 0.9999 ± 0.0001 | 0.9996 ± 0.0016 | 1.0000 ± 0.0000 | 0.2500 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| XGBoost | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.9997 ± 0.0013 | 1.0000 ± 0.0000 | 0.9968 ± 0.0073 | 0.9997 ± 0.0012 | 1.0000 ± 0.0000 | 0.8889 ± 0.4781 | 0.8182 ± 0.2258 | 0.9998 ± 0.0004 | 0.9999 ± 0.0001 | 0.9992 ± 0.0032 | 1.0000 ± 0.0000 | 0.7500 ± 0.0000 | 1.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| E-GraphSAGE | 0.3394 ± 1.4212 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0151 ± 0.0075 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.9811 ± 0.0026 | 0.9957 ± 0.0008 | 0.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0109 ± 0.0021 |

- trained on stratified subsample of 599,998/1,469,983 edges

## NF-UNSW-NB15-v2 — chronological 70/10/20, flow index as time (exact CCSTGN [11] protocol)

Test prevalence (attack share): 7.29%.

| Model | Seeds | Acc | Precision | Recall | F1 | FPR | ROC-AUC | TPR@FPR≤1% | TPR@FPR≤0.1% |
|---|---|---|---|---|---|---|---|---|---|
| Random Forest | 4 | 0.9959 ± 0.0000 | 0.9690 ± 0.0004 | 0.9746 ± 0.0003 | 0.9718 ± 0.0002 | 0.0025 ± 0.0000 | 0.9997 ± 0.0000 | 1.0000 ± 0.0000 | 0.9357 ± 0.0039 |
| XGBoost | 4 | 0.9957 ± 0.0000 | 0.9649 ± 0.0000 | 0.9768 ± 0.0000 | 0.9708 ± 0.0000 | 0.0028 ± 0.0000 | 0.9997 ± 0.0000 | 1.0000 ± 0.0000 | 0.9148 ± 0.0000 |
| MLP | 3 | 0.9938 ± 0.0007 | 0.9357 ± 0.0262 | 0.9834 ± 0.0210 | 0.9588 ± 0.0037 | 0.0053 ± 0.0024 | 0.9993 ± 0.0001 | 1.0000 ± 0.0000 | 0.7825 ± 0.0350 |
| 1D-CNN | 3 | 0.9939 ± 0.0009 | 0.9396 ± 0.0330 | 0.9790 ± 0.0258 | 0.9588 ± 0.0050 | 0.0050 ± 0.0030 | 0.9993 ± 0.0001 | 0.9998 ± 0.0001 | 0.8088 ± 0.0351 |
| E-GraphSAGE | 3 | 0.9791 ± 0.0026 | 0.8454 ± 0.0317 | 0.8733 ± 0.0131 | 0.8590 ± 0.0145 | 0.0126 ± 0.0032 | 0.9940 ± 0.0001 | 0.7597 ± 0.0004 | 0.2702 ± 0.0676 |
| TGN (XTGAIA, temporal attention) | 1 | 0.9795 | 0.8743 | 0.8396 | 0.8566 | 0.0095 | 0.9950 | 0.8452 | 0.2396 |

**Missed detections at fixed FPR** (threshold chosen on validation for FPR closest to the target; F1 / FNR on test, as in CCSTGN [11] Table II):

| Model | F1 @0.01% | FNR @0.01% | F1 @0.1% | FNR @0.1% | F1 @1% | FNR @1% | test FPR @1% |
|---|---|---|---|---|---|---|---|
| Random Forest | 0.8858 ± 0.0131 | 0.2011 ± 0.0216 | 0.9686 ± 0.0004 | 0.0433 ± 0.0014 | 0.9467 ± 0.0002 | 0.0000 ± 0.0000 | 0.0089 ± 0.0000 |
| XGBoost | 0.8560 ± 0.0000 | 0.2491 ± 0.0000 | 0.9620 ± 0.0000 | 0.0566 ± 0.0000 | 0.9383 ± 0.0000 | 0.0000 ± 0.0000 | 0.0103 ± 0.0000 |
| MLP | 0.6767 ± 0.0969 | 0.4862 ± 0.1137 | 0.8934 ± 0.0189 | 0.1776 ± 0.0329 | 0.9355 ± 0.0049 | 0.0000 ± 0.0000 | 0.0108 ± 0.0009 |
| 1D-CNN | 0.6844 ± 0.0506 | 0.4782 ± 0.0592 | 0.9131 ± 0.0213 | 0.1436 ± 0.0377 | 0.9367 ± 0.0088 | 0.0002 ± 0.0002 | 0.0106 ± 0.0016 |
| E-GraphSAGE | 0.2185 ± 0.1541 | 0.8763 ± 0.0995 | 0.4446 ± 0.0165 | 0.7101 ± 0.0146 | 0.8100 ± 0.0284 | 0.2296 ± 0.0596 | 0.0103 ± 0.0017 |
| TGN (XTGAIA, temporal attention) | 0.1785 | 0.9019 | 0.3551 | 0.7820 | 0.8566 | 0.1604 | 0.0095 |
| *MLP [11]* | 0.6583 | 0.5077 | 0.8229 | 0.2950 | 0.9516 | 0.0004 | – |
| *CCSTGN-rs [11]* | 0.5033 | 0.6599 | 0.7702 | 0.3658 | 0.9367 | 0.0001 | – |
| *CCSTGN-iso [11]* | 0.2232 | 0.8704 | 0.6076 | 0.5611 | 0.9477 | 0.0006 | – |
| *CCSTGN-rs (ft) [11]* | 0.4978 | 0.6659 | 0.7019 | 0.4533 | 0.9503 | 0.0008 | – |
| *CCSTGN-iso (ft) [11]* | 0.4694 | 0.6855 | 0.7012 | 0.4495 | 0.9493 | 0.0004 | – |

**Weighted threshold** (ω minimising 0.75·FNR + 0.25·FPR on validation, CCSTGN [11] Table I):

| Model | ROC-AUC | F1 | FPR | FNR |
|---|---|---|---|---|
| Random Forest | 0.9997 ± 0.0000 | 0.9575 ± 0.0007 | 0.0069 ± 0.0001 | 0.0004 ± 0.0003 |
| XGBoost | 0.9997 ± 0.0000 | 0.9585 ± 0.0000 | 0.0068 ± 0.0000 | 0.0004 ± 0.0000 |
| MLP | 0.9993 ± 0.0001 | 0.9511 ± 0.0014 | 0.0081 ± 0.0003 | 0.0001 ± 0.0000 |
| 1D-CNN | 0.9993 ± 0.0001 | 0.9509 ± 0.0052 | 0.0081 ± 0.0009 | 0.0003 ± 0.0001 |
| E-GraphSAGE | 0.9940 ± 0.0001 | 0.8931 ± 0.0254 | 0.0187 ± 0.0049 | 0.0010 ± 0.0016 |
| TGN (XTGAIA, temporal attention) | 0.9950 | 0.8732 | 0.0228 | 0.0002 |
| *MLP [11]* | 0.9991 | 0.9522 | 0.0078 | 0.0005 |
| *CCSTGN-rs [11]* | 0.9988 | 0.9462 | 0.0089 | 0.0002 |
| *CCSTGN-iso [11]* | 0.9989 | 0.9497 | 0.0083 | 0.0004 |
| *CCSTGN-rs (ft) [11]* | 0.9988 | 0.9504 | 0.0082 | 0.0006 |
| *CCSTGN-iso (ft) [11]* | 0.9988 | 0.9506 | 0.0080 | 0.0025 |
| *E-GraphSAGE [11]†* | 0.9039 | 0.9410 | – | – |
| *TGN [11]†* | 0.8801 | 0.9355 | – | – |
| *3D-IDS [11]†* | 0.9155 | 0.9545 | – | – |

*Reference rows:* Escriche et al. (CCSTGN) [11], CNS 2024 — NF-UNSW-NB15-v2, chronological 70/10/20, 5 runs. Tag: DIRECT (same dataset and protocol; our time = flow index as in [11]). † copied in [11] from the 3D-IDS paper (no FPR/FNR).

Per-class detection rate at the default threshold (mean over seeds):

| Model | Analysis | Backdoor | DoS | Exploits | Fuzzers | Generic | Reconnaissance | Shellcode | Worms | Benign flagged |
|---|---|---|---|---|---|---|---|---|---|---|
| Random Forest | 0.8486 ± 0.0022 | 0.9736 ± 0.0051 | 0.9622 ± 0.0027 | 0.9825 ± 0.0004 | 0.9514 ± 0.0005 | 0.9960 ± 0.0008 | 0.9954 ± 0.0009 | 0.9981 ± 0.0025 | 1.0000 ± 0.0000 | 0.0025 ± 0.0000 |
| XGBoost | 0.8675 ± 0.0000 | 0.9908 ± 0.0000 | 0.9684 ± 0.0000 | 0.9834 ± 0.0000 | 0.9504 ± 0.0000 | 0.9981 ± 0.0000 | 0.9984 ± 0.0000 | 0.9961 ± 0.0000 | 1.0000 ± 0.0000 | 0.0028 ± 0.0000 |
| MLP | 0.9449 ± 0.1431 | 0.9912 ± 0.0050 | 0.9533 ± 0.0919 | 0.9829 ± 0.0252 | 0.9764 ± 0.0121 | 0.9964 ± 0.0022 | 0.9988 ± 0.0006 | 0.9949 ± 0.0055 | 1.0000 ± 0.0000 | 0.0053 ± 0.0024 |
| 1D-CNN | 0.9011 ± 0.1237 | 0.9820 ± 0.0323 | 0.9431 ± 0.0869 | 0.9809 ± 0.0245 | 0.9698 ± 0.0409 | 0.9964 ± 0.0015 | 0.9982 ± 0.0022 | 0.9961 ± 0.0000 | 0.9949 ± 0.0221 | 0.0050 ± 0.0030 |
| E-GraphSAGE | 0.9980 ± 0.0000 | 0.9838 ± 0.0038 | 0.8708 ± 0.0089 | 0.6803 ± 0.0281 | 0.9859 ± 0.0134 | 0.9587 ± 0.0006 | 0.9996 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 0.0126 ± 0.0032 |
| TGN (XTGAIA, temporal attention) | – | – | – | – | – | – | – | – | – | – |

**Multi-class scoring (attack category).** TGN: hierarchical and consistent with its binary output (Benign when p_t < ω, otherwise the Z_t category head's most likely attack class); RF / XGBoost: flat multi-class classifiers on the same categories.

| Model | Accuracy | Macro-F1 (all classes) | Macro-F1 (attack classes) | Weighted-F1 |
|---|---|---|---|---|
| Random Forest | 0.9818 | 0.6841 | 0.6493 | 0.9820 |
| XGBoost | 0.9809 | 0.6405 | 0.6008 | 0.9799 |
| TGN (XTGAIA, temporal attention) | 0.9228 | 0.1097 | 0.0120 | 0.9182 |

- trained on stratified subsample of 600,000/1,673,192 edges

## Context-only references (not tabulated against our numbers)

- Causal-GNN [12]: CIC-IDS2017 ROC-AUC 0.934 as an *unsupervised anomaly detector* (release and split not stated); not comparable with supervised AUC.
- Ontology-APT [13]: DARPA TRACE (F1 0.92, AUC 0.96); different data and detection unit.
- AIR [5]: private enterprise data, multi-class (mean F1 0.94 ± 0.01).

## System latency (Sec. IV-D)

Source: `data/processed/demo_report.json`. Processing-only times (comparable to AIR [5]); MTTD is also given including the Δinf = 500 ms window. Hardware differs: AIR ran on 4 × Dell R760 + NVIDIA A100; XTGAIA on one 16-vCPU workstation.

| Stage | XTGAIA mean (ms) | XTGAIA p95 (ms) | AIR [5] (ms) |
|---|---|---|---|
| Acquisition (parse, STIX, graph update; per event) | 0.0 | 0.1 | 8.1 |
| Analysis (TGN + topology-aware risk; per micro-batch) | 31.5 | 74.1 | 15.2 |
| Decision (state + zone Double DQN) | 19.6 | 45.5 | 23.5 |
| Execution (Shuffle -> enforcement -> graph mutation) | 191.9 | 502.9 | 11.5 |
| End-to-end closed loop (ingest -> G_t+1) | 349.9 | 711.7 | 58.3 ± 3.2 |

| Quantity | XTGAIA | Comparator |
|---|---|---|
| MTTD, processing only (ingest -> TGN detection) | 77.6 ms | not reported by [5], [2] |
| MTTD incl. Δinf window | 304.9 ms | not reported |
| MTTR (detection -> SOAR enforcement) | 307.1 ms | [2]: 1,800 ms (traditional SOAR 15,700 ms) |
| TheHive audit case (Algorithm 4 Phase 2, after enforcement) | 879.7 ms | – |
| Explanation bundle X_t (parallel tiers) | 37.8 ms (p95 84.6) | bound 200 ms |

## Explainability (Sec. IV-E)

None of [1], [3], [5], [6], [14] publishes explanation latency, fidelity or stability; the post-hoc explainers they use were measured here on the same hardware ([3]'s P-CIC setup, attack flows, `src/evaluation/explainer_baselines.py`). Stability follows [6]: cosine similarity of attribution vectors of similar attacks.

| Explainer (model) | Mean latency (ms) | p95 (ms) | Fidelity | Stability (cosine) |
|---|---|---|---|---|
| **XTGAIA Tier 1+2: PGExplainer + attention rollout (TGN)** | 37.5 | 83.5 | 0.001 (fidelity+) | 1.000 structural / 0.984 temporal |
| **XTGAIA Tier 3: reward decomposition (Double DQN)** | 0.9 | 2.1 | exact (linear decomposition) | 0.992 |
| LIME (Random Forest) | 231.1 | 304.5 | 0.288 (top-5 deletion) | 0.772 |
| TreeSHAP (Random Forest) | 4.7 | 8.0 | 0.324 (top-5 deletion) | 0.966 |
| LIME (XGBoost) | 187.0 | 270.9 | 0.378 (top-5 deletion) | 0.638 |
| TreeSHAP (XGBoost) | 1.7 | 2.9 | 0.847 (top-5 deletion) | 0.953 |
| LIME (1D-CNN) | 339.1 | 437.3 | 0.099 (top-5 deletion) | 0.826 |
| KernelSHAP (1D-CNN) | 2302.5 | 2836.1 | 0.360 (top-5 deletion) | 0.727 |

XTGAIA structural sparsity (share of the 2-hop neighbourhood outside the top-k): 0.875.

Fidelity definitions differ by row: XTGAIA reports fidelity+ (drop in p_t when the top-k neighbourhood edges are masked); the flow explainers report the drop in p(attack) when their top-5 features are replaced by the training mean.

## Response policy vs SOAR playbooks

Multi-agent zone environment driven by real MITRE CALDERA adversary profiles (`src/evaluation/response_eval.py`), paper reward Eq. 10. [2] reports only aggregate latency; AIR [5]'s response metrics come from a private deployment (context only).

| Policy | Mean return | Containment rate | Steps to contain | False containment | Disruption | Cost |
|---|---|---|---|---|---|---|
| XTGAIA zone DDQN | 12.99 | 1.000 | 1.67 | 0.000 | 0.050 | 0.41 |
| Static SOAR playbook | 7.94 | 1.000 | 1.54 | 0.000 | 1.388 | 0.87 |
| Graded SOAR rules | 9.81 | 1.000 | 1.62 | 0.000 | 0.729 | 0.57 |
| No response | -30.86 | 0.354 | 2.13 | 0.000 | 0.000 | 0.00 |
