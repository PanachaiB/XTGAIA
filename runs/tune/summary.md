# TGN tuning summary (node-only design)

Protocol: `docs/tuning_protocol.md`. Validation only except the final runs (test touched once per dataset).

| Proxy | Knobs | Val F1 | AUC | TPR@0.1%FPR | TPR@1%FPR | Noise | Gain vs base | 2×noise | Minutes | Wins |
|---|---|---|---|---|---|---|---|---|---|---|
| base | defaults | 0.8157 | 0.9937 | 0.071 | 0.784 | 0.0050 | +0.0000 | 0.0000 | 98 | - |
| readout_max | {'XTGAIA_FLOW_READOUT': 'max'} | 0.8672 | 0.9955 | 0.218 | 0.872 | 0.0050 | +0.0515 | 0.0100 | 123 | yes |
| batch64 | {'XTGAIA_BATCH': '64'} | 0.6662 | 0.9914 | 0.076 | 0.570 | 0.0066 | -0.1495 | 0.0132 | 42 | no |
| k1_20 | {'XTGAIA_K1': '20'} | 0.7953 | 0.9932 | 0.058 | 0.754 | 0.0050 | -0.0204 | 0.0100 | 87 | no |
| pw_half | {'XTGAIA_POS_WEIGHT_SCALE': '0.5'} | 0.8220 | 0.9945 | 0.149 | 0.797 | 0.0050 | +0.0063 | 0.0101 | 44 | no |
| lr_1e-4 | {'XTGAIA_LR': '1e-4'} | 0.8120 | 0.9942 | 0.159 | 0.780 | 0.0050 | -0.0037 | 0.0100 | 50 | no |

**Selected:** readout_max {'XTGAIA_FLOW_READOUT': 'max'}

- Final cicids_heldout (test, once): F1 0.9595, AUC 0.9965, TPR@1%FPR 0.9911, TPR@0.1%FPR 0.3166
- Final unsw_chrono (test, once): F1 0.8573, AUC 0.9953, TPR@1%FPR 0.8459, TPR@0.1%FPR 0.2919
