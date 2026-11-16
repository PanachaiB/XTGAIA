# Implementation notes for the XTGAIAv3 paper update

The code implements the system as described in XTGAIAv3 (Phases 1–5, Eq. 1–10, Algorithms 1–4,
Sec. IV evaluation dimensions). This file lists every place where the implementation had to add a
detail the paper does not specify, or had to change something to make the system train or run.
Each item says what the paper says, what the code does, why, and suggested paper wording.

Items are grouped as **(A) changes** (the paper text is no longer exactly accurate) and
**(B) unspecified details** (the paper is silent; the text stays accurate but should state the choice).

---

## A. Changes the paper should reflect

### A1. Z_t reads the node memory as well as the attention embedding (Phase 3 / Eq. 9)
- **Paper:** Z_t(v) ∈ R^8 is the "GNN pooled embedding" from the TGN.
- **Code:** `Z_t = z_projector([h(v) ‖ m_v(t)])`, where h(v) is the 8-head temporal-attention
  embedding over the 2-hop neighbourhood and m_v(t) is the 128-d GRU memory. Still 8-d, still feeds
  p_t (Eq. 4) and the 16-d DQN state (Eq. 9). (`src/role2_ai/tgn_model.py`, `memory_skip=True`)
- **Why:** without the skip connection, early training with only attention plateaued at a
  validation AUC of about 0.87 and was unstable across seeds. The memory skip makes the GRU memory
  directly visible to the readout.
- **Suggested wording:** "Z_t(v) is obtained by projecting the concatenation of the
  temporal-attention embedding and the node memory m_v(t) to R^8."

### A2. MTTR and end-to-end latency end at Algorithm 4 Phase 1
- **Paper (Sec. IV-D):** MTTR runs "from threat detection to the execution of enforcement actions
  via SOAR connectors". End-to-end runs "to post-response graph state mutation (G_t+1)".
- **Code:** MTTR = detection → Shuffle webhook accepted + G_act mutated (Algorithm 4 lines 2–10).
  Opening the TheHive case (Phase 2, lines 11–12) is timed separately as `audit_ms`.
- **Why:** this matches the paper's definitions literally. An earlier version included the TheHive
  call in `execution_ms`, which mixed the human-on-the-loop audit into MTTR. Make it explicit so
  the comparison with AIR [5] / [2] is like-for-like.
- **Suggested wording:** add "TheHive case creation (Phase 2) is reported separately and is not
  part of MTTR."

### A3. Mordor ingestion goes through Wazuh `archives.json`
- **Paper (Table, Stage 1):** Mordor JSON logs are ingested through the Wazuh managers.
- **Code:** the Mordor events are written to a file that the Wazuh manager's log collector reads
  (`localfile`, `log_format json`). The manager is configured with `logall_json yes`, and XTGAIA
  consumes `/var/ossec/logs/archives/archives.json` (`src/role1_graph/wazuh_mordor.py`,
  `docker/wazuh/master/ossec.conf`).
- **Why:** most Mordor events do not match a Wazuh rule, so `alerts.json` would drop them. The
  archive stream carries every decoded event.
- **Suggested wording:** "Wazuh decodes the replayed Mordor events and XTGAIA consumes the
  manager's JSON archive stream."

### A4. Δinf is capped at 128 events per TGN update
- **Paper (Phase 3):** the TGN processes the active graph "over a broader temporal inference window of
  Δinf = 500 ms".
- **Code:** the orchestrator collects Role 1 edges and runs one TGN update when the batch spans 500 ms of event
  time **or** reaches 128 events, whichever comes first (`role2.micro_batch`, `src/pipeline/orchestrator.py`).
  Windows are tumbling, anchored at the first event of each batch; node memory carries state across windows.
- **Why:** within one update every event sees the memory as of the start of the batch (standard TGN batching).
  Training uses 128-event batches, so an uncapped window during a DDoS burst (thousands of flows in 500 ms) would
  score events with far staler memory than training ever saw. On the held-out Friday replay the average batch
  held about 29 events (547,557 events in 18,809 updates), so most updates are closed by the 500 ms rule and the
  cap only binds during bursts.
- **Suggested wording:** "…over a temporal inference window of Δinf = 500 ms (at most 128 events per memory
  update, matching the training batch)." If tuning (B11) selects another training batch size, `micro_batch` is
  set to the same value.

## B. Details the paper leaves unspecified

### B1. TGN optimisation
Adam with lr 2e-4, global gradient-norm clipping at 1.0, BCE with a positive-class weight,
streaming chronological mini-batches. Early stopping keeps the epoch with the best validation AUC
(at most 10 epochs). The decision threshold ω is calibrated on validation (best F1, or a fixed FPR).
lr 5e-4 diverged for the attention model.

### B2. Multi-class scoring (Sec. IV-C, Table Stage 3: "binary/multi-class classification")
- p_t(v) stays the binary malicious probability of Eq. 4. Risk and response use only p_t.
- An auxiliary category head names the attack family of a *detected event*. It reads the attention
  embedding h(v) through a stop-gradient, so it cannot change p_t or Z_t, together with that event's
  raw flow message. The 16-class taxonomy is shared by CIC-IDS2017/2018 and UNSW-NB15
  (`src/role2_ai/attack_classes.py`).
- Why the flow message: the first head read h(v) only. On NF-UNSW-NB15-v2 the same victim hosts
  receive every attack family from the same attacker IPs, so a node embedding cannot separate them
  (TGN macro-F1 0.11 vs RF 0.68). Checkpoints with the earlier head still load
  (`category_msg=False`).
- The multi-class prediction is hierarchical: Benign if p_t < ω, otherwise the argmax over the
  attack classes. Binary and multi-class results therefore never disagree about whether a node is
  malicious.
- Without the stop-gradient, the category loss inverted the binary output during training
  (validation AUC 0.28).

### B3. Pre-training on CSE-CIC-IDS2018
The paper says both datasets pre-train the TGN memory. The code pre-trains on a stratified
CSE-CIC-IDS2018 sample (all benign at 0.5 %, up to 5,000 flows per attack label) and fine-tunes on
CIC-IDS2017 (`tgn_eval --pretrain-2018`).

### B4. Evaluation protocols
- P-CIC split (random 80/20, benign downsampled in train only) for parity with [3].
- Chronological 70/10/20 on NF-UNSW-NB15-v2 for parity with CCSTGN [11].
- Held-out day for CIC-IDS2017: train Mon–Thu, test Friday. Hosts unseen in training are scored
  from checkpoint memory.
- CIC-IDS2017 uses the corrected release (Engelen et al.), with "Attempted" flows relabelled
  benign.

### B5. CALDERA adversaries
The PettingZoo environment's attacker follows real CALDERA stockpile adversary profiles (ability
chains mapped to 12 ATT&CK tactics). It does not run a live CALDERA server.

### B6. Host telemetry versus network flows in the TGN
The TGN is trained on flow records (CIC-IDS2017/2018, UNSW-NB15), so its edge messages are flow
statistics. Host telemetry (Mordor Sysmon/Security events via Wazuh: AUTHENTICATED, WROTE_FILE,
CREATE_CHILD_PROCESS, RUN) updates the typed SKG, the Algorithm 2 risk graph and the context vector
Φ_t (for example `auth_fails`), but does not produce TGN messages. Verified: the Mordor capture
`empire_psexec_dcerpc_tcp_svcctl` yields identical SKG edges whether read directly or through the
Wazuh manager (4,348 events read, 51 typed edges).

### B7. Algorithm 1 latent states are tied to the model that wrote them
Each persisted 512-byte latent state is stored with a fingerprint of the TGN weights
(`Host.latent_model`). Algorithm 1 line 9 only deserialises states whose fingerprint matches the
running model. Otherwise it falls back to 0^128 (line 13). Without this, retraining the model and
then restarting the pipeline loaded memory produced by different weights. An evaluation replay
(`main.py demo`) clears the current model's persisted states first, so repeated replays of the
same day do not start from memory left by the previous replay.

### B8. Testbed details
- The Wazuh master agent port is mapped to host 1516 (configurable) because 1514 is commonly taken
  on the host.
- Hardware differs from Sec. IV-A (the paper states 32 GB RAM and a GPU; the reference runs used an
  8 GB Docker VM on CPU). Hardware-sensitive numbers (latency) should be re-measured on the paper's
  testbed.

### B9. Flow-level read-out of the node-level p_t (open decision)
Eq. 4 is node-level. Flow-level benchmarks (CCSTGN [11], [3]) score each flow, so the evaluation
needs a read-out. The default scores a flow by its destination node, as the live pipeline does.
`XTGAIA_FLOW_READOUT=max` scores (and trains) a flow as max(p_t(src), p_t(dst)): the flow is
malicious if either endpoint is. Re-scoring the existing UNSW checkpoint without retraining:
F1 0.857 (dst) → 0.875 (max), against CCSTGN's 0.946. Trained with the max objective and the
event-level head (4 epochs, seed 0): F1 0.863, AUC 0.9953, TPR@1%FPR 0.858. The multi-class macro-F1
rose from 0.11 to 0.37 (accuracy 0.954; Exploit 0.67, Reconnaissance 0.63, Shellcode 0.59,
Fuzzers 0.54; RF 0.68). Node-level scoring is the structural limit against edge-level comparators. If the trained max read-out is adopted, the paper should state the
flow read-out explicitly.

## Known limitations to state in the paper
- **Risk is sparse under Δc = 50 ms.** Algorithm 2 runs on the active graph, which holds only the last 50 ms of
  edges. A host therefore has nonzero Risk_t (Eq. 5) only while a directed path to a crown jewel exists inside that
  window. In the Friday replay most dispatches had Risk_t = 0, and nonzero values appeared mainly for traffic
  touching crown jewels (e.g. 7.995 for 192.168.10.50). This follows the paper's design, but it means the risk
  component of the DQN state is usually zero. The paper should say so, or justify the window length.
- **Held-out Friday multi-class:** DDoS and Botnet occur only on Friday in CIC-IDS2017, so a model
  trained on Mon–Thu cannot name those categories. Binary detection still flags them (DDoS recall
  0.995). Multi-class results should be reported on the random-split and chronological protocols,
  where every class is seen in training.
- **Botnet on held-out Friday:** recall is low for every supervised model, including ours (TGN 0.12),
  because the class is novel and very small (736 flows).

---

## C. Comparator coverage: what each cited paper allows us to compare

Tags follow the comparison spec: **DIRECT** (same data and protocol, numbers side by side),
**PROTOCOL-ONLY** (we reuse the paper's protocol on our data or rerun on its data), **CONTEXT-ONLY**
(different data or task: cite, do not tabulate), **QUALITATIVE** (no usable numbers),
**MEASURED BY US** (no published value, so we ran the method ourselves).

### C1. Threat detection (Sec. IV-C)

| Ref | Tag | What we compare | Status |
|---|---|---|---|
| [3] Vedesh (RF / XGB / 1D-CNN, CICIDS2017) | DIRECT after P-CIC | Their published rows vs our RF / XGB / MLP / 1D-CNN under P-CIC split A | Done for the baselines (ours F1 ≥ 0.999 vs their 0.946–0.963). The **TGN has not been run under P-CIC split A**; its held-out Friday F1 of 0.970 is a different, harder protocol. Their test prevalence (≈43–45 %, derived) differs from ours (24 %); lead with TPR / FPR / AUC. |
| [11] CCSTGN (NF-UNSW-NB15-v2) | DIRECT (we ran their protocol on their dataset) | Binary AUC / F1 / FPR / FNR, F1 and FNR at FPR {0.01, 0.1, 1} %, λ = 0.75 weighted threshold | Done. Baselines beat it (RF F1 0.972); **the TGN does not** (best F1 0.863 vs 0.946). 1 TGN seed vs their 5. |
| [12] Causal-GNN | CONTEXT-ONLY | Only number: CIC-IDS2017 ROC-AUC 0.934 for an **unsupervised anomaly detector**; release, split and runs not stated | Footnote only. Our supervised AUC (0.998) is not like for like. Their "12.7 % F1 gain on CIC-IDS2017" has no supporting CIC F1 in their tables, so do not repeat it. |
| [13] Ontology-APT | CONTEXT-ONLY | DARPA TRACE event/sequence detection (F1 0.92, AUC 0.96); LANL result has no numbers | Not comparable (different data and detection unit). Comparable only if XTGAIA is run on DARPA TRACE. |
| [5] AIR | CONTEXT-ONLY | Private 18-month financial telemetry, 8-class multi-class (F1 0.94 ± 0.01; the abstract says 0.93) | No AUC, no fixed-FPR metrics, and no benign row in the per-class figure, so no FPR can be derived. We report the matching per-class recall with a benign row so readers can set the two side by side. |
| [2] Kulkarni | CONTEXT-ONLY | Accuracy 96.5 %, FPR 3.2 % (SOAR 88.3 % / 9.1 %), with no dataset, split or run count tied to the numbers | Cite as context only. |
| [6] Khare (LIME stability) | CONTEXT-ONLY for detection | DT / RF / SVM / NB on "CICIDS2018" (their Table II disagrees with their confusion matrices; split not stated) | Not compared. If cited, use the confusion-matrix-derived values (e.g. RF FPR 3.31 %). |
| [14] GAT-XAI | Not comparable | Malware detection from API-call graphs on an unnamed dataset | Not a network-intrusion task; cite only as GAT-plus-explanation related work. |

### C2. System latency (Sec. IV-D)

Measured and reported (AIR [5] stage-by-stage, [2] MTTR); hardware-dependent numbers are left for
the teammate's testbed run. Values the papers do not report, which the paper must not imply:
- **MTTD**: not reported by [5] or [2]. XTGAIA's MTTD has no published comparator.
- **MTTR for AIR [5]**: not reported. Its 6.2 s "time-to-recover after a detection failure" is a
  different quantity and must not be labelled MTTR.
- **[9] CSOC copilots**: position paper with no latency or detection numbers. The draft's Sec. IV-A
  lists it as a latency baseline; remove it from the baseline list and use it for framing only.
- **[3]**: says "low latency" but gives no figures.
- **[11]**: throughput only (about 2 s per 10,000-flow batch on a laptop GPU), not per-event latency.

### C3. Explainability (Sec. IV-E)

None of the cited papers publishes explanation latency, fidelity or a stability value. Every number
in the paper's XAI comparison is therefore **measured by us**: LIME, TreeSHAP and KernelSHAP on
[3]'s P-CIC models, on the same hardware.

| Ref | Tag | What exists in the paper | Consequence |
|---|---|---|---|
| [1] Kalasampath (review) | QUALITATIVE | Statements only (SHAP more stable but costly; LIME lighter) | Cite for framing. |
| [3] Vedesh | QUALITATIVE | Global RF Gini importance only; **no SHAP / LIME timings** | The draft intro cites [3] for "LIME introduces significant explanation latency, while SHAP is too slow". [3] has no such result: cite our own measurement (LIME 187–339 ms, KernelSHAP 2.3 s per explanation). |
| [5] AIR | QUALITATIVE for XAI quality; analyst efficiency only | SHAP + attention rollout; attention weight 0.78 ± 0.05, operator policy coverage 92 %; **no latency, fidelity or stability** | No XAI number to compare. Its analyst numbers (12 interventions per 100 alerts, 4.7 min each) count interventions, not manual overrides; define XTGAIA's metric before setting them side by side (teammate's test). |
| [6] Khare | PROTOCOL-ONLY | Defines cosine stability, **reports no stability value**; does not define "similar instances" | We use their formula with our own definition (nearest neighbour of the same attack class in standardised feature space). The draft's claim that [6] shows explainers are "inconsistent under adversarial noise" is not in [6]: no adversarial experiment. |
| [14] GAT-XAI | QUALITATIVE | Attention + SHAP; **no fidelity, stability or latency** (lists "real-time latency" as a metric without a value) | The draft calls [14] "high-fidelity": unsupported, rephrase. |

### C4. Response policy (Sec. IV, DRL vs SOAR)

- Done: zone Double DQN vs static and graded SOAR playbooks in the CALDERA-driven environment
  (return 12.99 vs 7.94 / 9.81, same containment with far less disruption).
- **AIR [5]** response-quality figures (cost per false positive in CNY, cost-effectiveness ratio
  0.85, ε-Nash 97.3 %) come from its private deployment and game-theoretic model. They are not
  comparable and should be cited as context.
- **[2]** gives only aggregate figures (SOAR response 15.7 s vs 1.8 s), used for MTTR only.

### C5. Open items before the comparison is complete (not waiting on hardware)

1. **Explainer baselines on more instances.** Done: 1,000 attack instances per explainer, with E-XAI's
   descriptive accuracy, sparsity and top-k overlap stability added
   (`data/processed/eval/explainer_baselines_1000.json`). Mean latency: LIME 100–167 ms, KernelSHAP 954 ms,
   TreeSHAP 1–2 ms, against XTGAIA's 85.9 ms at detection time. With XTGAIA's degenerate stability excluded,
   the XAI comparison is 4 wins and 8 losses (`docs/comparison_scorecard.md`).
2. **XTGAIA Tier 1 fidelity.** Confirmed on 1,199 detection-time explanations of the tuned model
   (CICIDS Friday, `runs/final_metrics/`): fidelity+ is about 0 (mean -2e-7) and fidelity- about 0
   (3e-6). Masking the top-k neighbourhood edges does not move p_t. The likely cause is that since
   change A1, Z_t reads the node's own memory directly, so p_t barely depends on the attention
   neighbourhood that PGExplainer explains. Either the structural tier should also explain the memory
   path, or the paper should report fidelity per tier honestly. This is open, and it is a TGN design
   question (not changed here).
3. **XTGAIA stability.** The dedicated benchmark ran: 1,199 detections, 10,000 same-class pairs.
   Structural and temporal cosine are both 1.000, but the value is degenerate, not evidence of a
   stable explainer. Detected nodes have saturated scores (p_t about 0.999), so the top-5 edge scores
   and the rollout trace are near-constant vectors, and any two are collinear. Report the stability
   with this caveat, or measure it on rank or identity overlap (which edges, not their scores).
   Explanation latency on the same sample: mean 85.9 ms, p95 217.5 ms, 93.7 % within the 200 ms bound.
4. **Seeds.** The spec asks for ≥ 5 seeds. Baselines have 3–4; every TGN result is a single seed.
5. **TGN under P-CIC split A** for the direct [3] comparison (TGN-dependent).


### B10. Design check: flow-level head (automated decision log)
Rules frozen in `docs/design_check_rules.md` before any check ran; validation metrics only; summary in `runs/design_check/summary.md`. Entries are appended by `scripts/design_check.py`.

- **2026-11-15 14:13:14 - R1: MLP clearly below ~0.97: pipeline/protocol problem; architecture work stopped, existing job queued.** MLP val F1 (mean of 3) = 0.9336; MLP sd = 0.0000; noise band = 0.0050; TGN C1 val F1 (dst projection) = 0.8538; TGN C1 val F1 (max projection) = 0.8779; TGN C1 val AUC = 0.9954; MLP val AUC (seed 0) = 0.9991

- **2026-11-15 14:13:15 - Test (once): current design (node p_t, destination projection).** f1 = 0.8566; roc_auc = 0.9950; tpr_at_fpr_1pct = 0.8452; tpr_at_fpr_0.1pct = 0.2396

- **2026-11-15 21:40 - Protocol fix after R1 (manual entry).** R1 fired because the shared threshold rule
  (`calibrate_threshold`) only tried thresholds between the 10th and 90th percentile of validation scores plus
  the 1 %-FPR budget point. On NF-UNSW validation (6.6 % attacks) every threshold strict enough to raise
  precision lies above the 90th percentile, so every arm was pinned to exactly 1 % FPR: the MLP reached
  TPR 1.000 at 1 % FPR and AUC 0.999 but only F1 0.934. Fix: exact search over every distinct validation
  score, with the budget rule unchanged (unit test added). The decision rules file is unchanged. The check
  reruns from C1 on the same cached TGN embeddings. Run 1 touched the test split once (current design,
  F1 0.8566); the rerun touches it once more for its own final configuration. Run 1's summary is kept at
  `runs/design_check/run1_threshold_defect/summary.md`. TGN results reported elsewhere that use the
  default calibrated threshold were produced with the old rule; the fixed-FPR tables are unaffected.

- **2026-11-15 14:40:48 - R1: MLP clearly below ~0.97: pipeline/protocol problem; architecture work stopped, existing job queued.** MLP val F1 (mean of 3) = 0.9546; MLP sd = 0.0002; noise band = 0.0050; TGN C1 val F1 (dst projection) = 0.8560; TGN C1 val F1 (max projection) = 0.8822; TGN C1 val AUC = 0.9954; MLP val AUC (seed 0) = 0.9991

- **2026-11-15 14:40:49 - Test (once): current design (node p_t, destination projection).** f1 = 0.8487; roc_auc = 0.9950; tpr_at_fpr_1pct = 0.8452; tpr_at_fpr_0.1pct = 0.2396

- **2026-11-15 22:00 - Override of R1 (manual entry, user decision; recorded before C3 ran).** The
  corrected-rule rerun gave MLP validation F1 0.955 against the 0.96 cut-off, a 0.005 miss. Reasons for the
  override: (a) the cut-off was derived from baselines at a fixed 0.5 threshold on the test split, while 0.955
  is a validation F1 with a budget-tuned threshold on a slice with a different attack share (validation 6.6 %,
  test 7.3 %); (b) R1 is meant to catch pipeline problems, which would show as MLP ≈ TGN, but on the same
  split the MLP is far above the TGN (0.955 vs 0.856 destination projection, 0.882 max projection).
  **R4 adoption criteria are unchanged; the override only permits the C3 experiment to run.**
  Seed-noise floor, declared before any C3 result: noise = max(measured std across the arm's 3 seeds,
  half-width of a 95 % bootstrap CI over validation flows, 0.005 F1). R4's "2x seed std" test uses
  2 × noise. Any measured std of exactly 0.000 is reported as such and the floor is used. The test split
  (already touched twice: run 1 and the corrected-rule rerun) is not touched by C3.

- **2026-11-15 22:29 - R4 after C3 (override run): hybrid flow head ADOPTED.** Validation flow F1: node-only 0.8560, flow-only 0.9504, hybrid 0.9504; gain +0.0944 vs 2 x noise 0.0100 (noise = max(seed sd, bootstrap half-width, 0.005): hybrid 0.0050, node-only 0.0050). Entity detection (lenient) hybrid 1.000 vs node-only 1.000; median time to first detection (lenient, flows) 0.0 vs 0.0; hybrid head 0.14 ms per 128 flows. Checks: gain True, entity True, TTFD True, latency True. Node-level F1 lenient: hybrid 1.0000 vs node-only 1.0000; strict: 0.1165 vs 0.1284. Test split not touched. Details: `runs/design_check/summary.md` (C3 override section).

- **2026-11-15 22:35 - Retrain switched to the hybrid flow head (manual entry, follows R4).** The retrain
  launched after the corrected-rule R1 decision (current design) was stopped a few minutes into its first epoch and
  relaunched as `scripts/retrain_role2.sh` with `XTGAIA_FLOW_HEAD=hybrid` (CICIDS held-out, then UNSW max and
  dst). Caveats recorded with the adoption: (1) the flow head's validation F1 (0.950) is no higher than the plain
  MLP on flow features alone (0.955), so the gain over node-only comes from per-flow features rather than the TGN
  embeddings; (2) 5.1 % of validation flows (19.9 % of validation attacks) have an identical feature vector in train,
  all with the same label, which favours flow-feature models; (3) node-level metrics under the lenient label and
  time to first detection saturate (1.0 and 0 flows) for every arm, and under the strict label every arm is poor
  (node F1 0.11-0.13), so the entity criteria in R4 are weakly informative here.

- **2026-11-15 22:42 - Duplicate check after C3 (validation only).** Removing the 12,132 validation flows whose feature vector occurs in train (3,129 of 15,721 attacks): re-tuned F1 node-only 0.8730, flow head 0.9390, MLP 0.9442; gain +0.0660 vs 2 x noise 0.0100: the hybrid gain survives duplicate removal. Flow head - MLP -0.0052.

- **2026-11-16 - Final decision: node-only design kept (user decision).** Although R4 passed in the override run,
  the hybrid flow head is not adopted: it would change the system architecture (a second threat score) to the point
  where the original design is weakened, and its gain comes from per-flow features (it matches a plain MLP) rather
  than from the TGN. The hybrid retrain was stopped. The node-level p_t(v) of Eq. 4 remains the single threat score
  driving alerts, risk (Eq. 5), the DQN state (Eq. 9) and the explainers. The optional `XTGAIA_FLOW_HEAD` code
  path stays off. The design-check findings (C1-C3, duplicate check) are kept as evidence for the paper's
  discussion of node-level vs flow-level scoring. Next: validation-only tuning of the original TGN (B11).

### B11. Tuning of the original TGN (node-only design)
Protocol frozen in `docs/tuning_protocol.md` before any run: only knobs the paper leaves unspecified (batch size,
neighbours per hop, learning rate, class weighting, flow projection of p_t); validation-only selection on a UNSW
copy without test rows; adoption needs a validation F1 gain > 2 x noise; test touched once per dataset by the final
runs. Entries below are appended by `scripts/tune_tgn.py`.

- **2026-11-16 06:55 - Round 1 (validation, 3 epochs).** base 0.8157; base 0.8157 (gain +0.0000 vs 2x noise 0.0000, AUC 0.9937, 98 min); readout_max 0.8672 (gain +0.0515 vs 2x noise 0.0100, AUC 0.9955, 123 min); batch64 0.6662 (gain -0.1495 vs 2x noise 0.0132, AUC 0.9914, 42 min); k1_20 0.7953 (gain -0.0204 vs 2x noise 0.0100, AUC 0.9932, 87 min); pw_half 0.8220 (gain +0.0063 vs 2x noise 0.0101, AUC 0.9945, 44 min); lr_1e-4 0.8120 (gain -0.0037 vs 2x noise 0.0100, AUC 0.9942, 50 min). Winners: readout_max

- **2026-11-16 06:55 - Selected configuration.** readout_max {'XTGAIA_FLOW_READOUT': 'max'}

- **2026-11-16 16:34 - Final cicids_heldout (test, touched once).** F1 0.9595, AUC 0.9965, TPR@1%FPR 0.9911, TPR@0.1%FPR 0.3166, multi-class macro-F1 0.4247

- **2026-11-16 22:56 - Final unsw_chrono (test, touched once).** F1 0.8573, AUC 0.9953, TPR@1%FPR 0.8459, TPR@0.1%FPR 0.2919, multi-class macro-F1 0.3919

- **2026-11-16 22:57 - Tuning finished.** report regenerated (docs/evaluation_results.md); summary in runs/tune/summary.md

**Outcome and interpretation.**
- Only the flow read-out passed the adoption rule. The TGN dimensions are unchanged (128-d memory, 8 heads,
  2 hops, 8-d Z_t, Δc 50 ms, Δinf 500 ms), and p_t stays the single threat score.
- The max read-out scores a flow as max(p_t(src), p_t(dst)) instead of p_t(dst). Alerts, risk, the DQN state
  and the explainers still use the node score p_t(v); see B9.
- On test, tuning brought no gain on either dataset:

  | Dataset | Tuned | Earlier max read-out checkpoint | Earlier dst read-out |
  |---|---|---|---|
  | NF-UNSW-NB15-v2 F1 / AUC | 0.857 / 0.9953 | 0.863 / 0.9953 | 0.857 / 0.9950 |
  | CICIDS2017 held-out Friday F1 / AUC | 0.960 / 0.9965 | – | 0.970 / 0.9979 |

- On CICIDS the tuned model misses fewer attacks (FNR at 1 % FPR 0.003 vs 0.005; DDoS recall 1.000), but its
  Botnet recall drops to 0.024 (from 0.120) and it raises more false alarms.
- The selection was made on UNSW validation only. The test sets were not used to choose anything, and both
  finals are reported as measured (single seed).
- Conclusion for the paper: within the knobs the paper leaves open, the node-only TGN is at its ceiling on
  these datasets. The UNSW gap to CCSTGN [11] is structural (node scores on 41 hosts; B10). It is reported as
  a detection limitation with the future work listed in `docs/comparison_scorecard.md`.


---

## D. Possible future work (not implemented)

Out of scope for this version. Listed so the paper's future-work section can draw on them.

1. **Node-level (host-event) input alongside flows.** The TGN's edge messages (Eq. 3) are currently 41 flow
   statistics, and supervision is flow labels projected onto the destination node. A second message type (typed
   host events: logons, process creation, file writes, encoded as event type + attributes, with node-type
   embeddings) would let host telemetry (Mordor, Sysmon) update node memory directly, instead of only the graph,
   risk and Φ (B6).
   - **Unchanged by this:** memory, attention, Z_t, p_t (Eq. 4), Algorithms 1–4 and Role 3.
   - **Redefined:** Eq. 3's message, and Eq. 8's port term for non-network edges.
   - **Gained:** node-level supervision of Eq. 4, plus a meaningful "actual attack entity" metric. On flow data,
     every node-score arm reached only node F1 0.11–0.13 for this metric in the design check (B10).
2. **Multi-stage evaluation of the latent-state claim.** No dataset used so far chains attack stages. The datasets
   below have stage or host-compromise labels. Their timestamp precision decides whether Δc = 50 ms and
   Δinf = 500 ms can be exercised as specified.

   | Dataset | Labels | Timestamp precision |
   |---|---|---|
   | DAPT2020 | APT stages; flow format close to CICIDS | probably whole seconds, not verified |
   | Unraveled | APT stages; network and host data | flow precision not verified |
   | LANL 2015 | red-team compromise labels | whole seconds; the standard temporal-graph benchmark |
   | DARPA OpTC | labelled malicious hosts and processes | millisecond |
   | DARPA TC (E3/E5) | labelled provenance | nanosecond |

   Suggested experiments:
   - stage-wise detection rate and delay;
   - persistent vs reset memory (per Δinf window, and on eviction);
   - pipeline restart mid-campaign with Algorithm 1 recovery vs a cold start.
