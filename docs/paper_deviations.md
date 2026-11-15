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
F1 0.857 (dst) → 0.875 (max), against CCSTGN's 0.946. Node-level scoring is the structural limit
against edge-level comparators. If the trained max read-out is adopted, the paper should state the
flow read-out explicitly.

## Known limitations to state in the paper
- **Held-out Friday multi-class:** DDoS and Botnet occur only on Friday in CIC-IDS2017, so a model
  trained on Mon–Thu cannot name those categories. Binary detection still flags them (DDoS recall
  0.995). Multi-class results should be reported on the random-split and chronological protocols,
  where every class is seen in training.
- **Botnet on held-out Friday:** recall is low for every supervised model, including ours (TGN 0.12),
  because the class is novel and very small (736 flows).
