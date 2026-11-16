# TGN tuning protocol (node-only design)

Frozen before any tuning run. Not edited afterwards; every result and selection is logged by
`scripts/tune_tgn.py` to `runs/tune/summary.md` and `docs/paper_deviations.md` (B11).

## Scope

- **Design unchanged.** The node-level p_t(v) (Eq. 4) is the single threat score for alerts, risk (Eq. 5), the
  DQN state (Eq. 9) and the explainers. No flow head. The detached category head (multi-class) stays.
- **Knobs** are training hyper-parameters and the flow-level projection of p_t only:
  - batch size (temporal resolution of memory updates);
  - neighbourhood size k1;
  - learning rate;
  - positive-class weight scale;
  - projection of p_t onto a flow: destination node (live pipeline) or max(p_t(src), p_t(dst)).
- **Where tuning happens.** NF-UNSW-NB15-v2 chronological, the dataset where the TGN trails the comparators. The
  CICIDS2017 held-out run uses the selected training knobs. It keeps the destination projection unless the
  max projection is selected, which then applies to both datasets.

## Consistency with the paper

Fixed by the paper; no knob may change them:
- 128-d GRU memory (Eq. 3);
- 8-head temporal attention over the 2-hop neighbourhood;
- 8-d Z_t;
- node-level p_t(v) (Eq. 4) as the single threat score;
- Δc = 50 ms and Δinf = 500 ms;
- risk (Eq. 5–8), the 16-d DQN state (Eq. 9) and the reward (Eq. 10);
- the three XAI tiers explaining p_t.

The paper does not specify the training batch size, neighbours per hop, learning rate, class weighting, or how a
node score maps onto a flow for flow-level benchmarks, so these are the only knobs. The max projection still
uses only node p_t (a flow is flagged when either endpoint's p_t crosses ω). If it is selected, the paper should
state the projection (B9). Existing documented deviations stay as they are: A1 (memory skip into Z_t), the
detached category head (B2) and the threshold rule (B10 fix).

## Overfitting guards

1. **No test rows during tuning.** Proxy runs train and validate on a copy of the UNSW table with the test rows
   removed. In the chronological split, test rows come after validation, so validation scores are identical
   without them.
2. **Selection uses validation only:** the shared threshold rule (exact search, best F1 at validation
   FPR ≤ 1 %), F1, AUC, and TPR at 0.1 % and 1 % FPR.
3. **Noise.** noise = max(95 % bootstrap half-width of validation F1, 0.005), the same floor as B10.
4. **A knob is adopted only if:**
   - its validation F1 beats the baseline proxy by more than 2 × noise;
   - its validation AUC is not lower than the baseline's by more than 0.001;
   - it does not add more than 50 % to the epoch time.
5. **Combination.** If more than one knob wins, one combined proxy is run. It is adopted only if its F1 is at
   least the best single winner's minus noise; otherwise the best single winner is kept.
6. **Budget.** Proxies run 3 epochs, seed 0. At most 6 proxies in round 1, plus 1 combination. No further rounds,
   so the number of looks at validation stays fixed.
7. **The test split is touched once per dataset**, by the final full runs (10 epochs, early stopping on
   validation AUC) with the selected configuration. Every proxy tried is reported, not only the winner.

## Round 1 proxies (one change each against the baseline)

| Name | Change |
|---|---|
| base | defaults: destination projection, batch 128, k1 10, lr 2e-4, pos-weight scale 1.0 |
| readout_max | projection max(p_t(src), p_t(dst)), trained the same way |
| batch64 | batch 64 |
| k1_20 | k1 = 20 neighbours |
| pw_half | positive-class weight × 0.5 |
| lr_1e-4 | learning rate 1e-4 |
