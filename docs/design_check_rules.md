# Design check: flow-level head for the TGN (decision rules)

Frozen before any check was run. This file is not edited after its first commit; every decision
taken by `scripts/design_check.py` is logged with its numbers in `docs/paper_deviations.md` (B10)
and `runs/design_check/summary.md`.

All decisions use VALIDATION metrics only. The test split is touched once, for the final chosen
configuration.

## Checks (as specified)

C1  Metric parity: score flows from the current TGN by projecting node p_t onto
    each incident flow; report flow-level F1, AUC, TPR@{0.1%,1%}FPR and
    node-level metrics from the SAME model, same split/threshold rule.
C2  In-house MLP on the same flow features/split/threshold rule, 3 seeds.
C3  Small flow head (MLP over both endpoint embeddings + flow features).
    Arms: flow-only, node-only, hybrid. 3 seeds each, reduced epochs/subsample.
    Same node-label definition in every arm. Roll-up options (noisy-OR, max,
    top-k mean) selected on validation.
    Also record: head inference ms, entity-level detection and time to first
    detection, and whether the PGExplainer target still matches the risk/DQN input.

## Decision rules (verbatim)

R1  MLP F1 clearly below ~0.97 (within seed noise of current TGN) ->
    pipeline/protocol problem: stop architecture work, report the diff,
    queue the existing 5h job unchanged.
R2  MLP ~0.97 and C1 flow metrics within seed noise of MLP -> labeling-rule
    gap only: keep design, report both views, queue existing 5h job.
R3  MLP ~0.97, C1 far below -> run C3.
R4  After C3: adopt hybrid only if its validation gain over node-only exceeds
    2x seed std AND entity-level metrics/time-to-first-detection do not degrade
    AND head latency is acceptable. Otherwise keep the current design.
R5  Any inconclusive or errored outcome -> fall back to the current design and
    queue the existing 5h job. Never leave the machine idle.

## Seed-noise note and operational definitions

These definitions turn the rules into fixed thresholds, set before any result was seen.

- **Dataset and model.** NF-UNSW-NB15-v2, chronological 70/10/20 (the CCSTGN [11] protocol). This is
  where the TGN (0.86) sits far below flow models (about 0.96–0.97), which is what the ~0.97 anchor in
  R1–R3 refers to. "Current TGN" is the design-default checkpoint `tgnv2_nf_un_chrono_s0.pt`
  (destination read-out, seed 0). The TGN training table's train/val rows are used for every arm.
- **Threshold rule (every arm, C1 and C2).** ω = the validation threshold with the best F1 among those
  with validation FPR ≤ 1 % (`calibrate_threshold`, as used for the TGN). F1 is reported on validation
  at that ω, together with AUC and TPR at 0.1 % and 1 % FPR.
- **Seed noise.** σ is the sample standard deviation of validation F1 over the 3 seeds of a run. Three
  seeds underestimate spread, so the noise band is **n = max(2σ, 0.005)**. The current TGN has a
  single seed; its noise is taken as the MLP's band (C2) or the matching arm's band (C3).
- **"MLP ~0.97"**: mean MLP validation F1 ≥ 0.96. **"Clearly below ~0.97"** (R1): mean < 0.96.
- **"C1 within seed noise of MLP"** (R2): C1 flow F1 (destination projection) ≥ MLP mean − n_MLP.
  Otherwise **"far below"** (R3).
- **Node label (every arm).** In the validation period, a node is malicious if it is an endpoint
  (source or destination) of at least one attack flow; otherwise benign. Node score = roll-up over
  the node's incident validation flows (TGN: its own p_t at each event). Roll-ups: max, noisy-OR
  (1 − Π(1 − p)) and top-k mean (k = 10), with the roll-up chosen by validation node AUC per arm.
- **Entity-level detection**: share of malicious nodes with at least one incident flow scored ≥ ω,
  plus the benign-node false-alarm share. **Time to first detection**: for each malicious node,
  flow-index distance from its first validation attack flow to its first flagged incident attack flow
  (the dataset has no wall-clock time; [11] also uses flow index as time). Report the median.
- **R4 thresholds.** Gain: mean hybrid flow F1 − mean node-only flow F1 > 2 × pooled σ of the two arms
  (and > 0.005). No degradation: hybrid entity detection ≥ node-only − 0.05, and hybrid median time
  to first detection ≤ node-only's × 1.10 (plus one flow-index unit). Head latency acceptable:
  ≤ 5 ms per 128-flow micro-batch on CPU (the TGN analysis step averages about 31 ms per micro-batch
  live).
- **PGExplainer target.** PGExplainer explains the node p_t that feeds risk (Eq. 5) and the DQN state
  (Eq. 9). An arm "matches" when the score it uses for detection is that same p_t (node-only);
  flow-level scores do not match unless the risk/DQN input is changed too.
- **Inconclusive** (R5): any check that errors, produces NaN, or has fewer than 3 finished seeds.
- **5 h job.** `scripts/retrain_role2.sh` reordered so the CICIDS held-out run goes first, followed by
  the UNSW runs. If R4 adopts the hybrid head, the same jobs run with `XTGAIA_FLOW_HEAD=hybrid`.
