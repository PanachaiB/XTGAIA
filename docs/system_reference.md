# XTGAIA system reference (for presentation Q&A)

What every module and script does, how one event flows through the system, how each model is trained,
and short answers to likely questions. Deviations from the paper are referenced by their IDs in
`docs/paper_deviations.md` (A = change, B = unspecified detail).

---

## 1. The system in one paragraph

XTGAIA is a closed-loop SOC pipeline in three roles:
- **Role 1 (graph and risk)** turns raw telemetry into STIX 2.1 events and keeps a live security knowledge
  graph (SKG) of the last **Δc = 50 ms** in RAM (NetworkX), persisted to Neo4j.
- **Role 2 (detection)** runs a **temporal graph network (TGN)** over that stream in windows of
  **Δinf = 500 ms** (at most 128 events per memory update). It produces a node threat score **p_t(v)**
  (Eq. 4) and an 8-d embedding **Z_t**.
- **Role 1 again** computes a topology-aware **risk** for each host (Eq. 5–8, Algorithm 2): p_t scaled by how
  cheaply an attacker could reach a crown jewel.
- **Role 3 (response)** builds a 16-d state [Z_t ‖ p_t ‖ Risk_t ‖ Φ_t] (Eq. 9). A **zone Double DQN** picks one
  of five graded playbooks (NoOp, Rate limit, Isolate port, Revoke session, Quarantine).
- **Execution:** the action is sent to **Shuffle** (SOAR), logged as a **TheHive** case with a three-tier
  explanation, and the live graph is mutated (G_t+1, Algorithm 4). An analyst rollback undoes it and feeds
  a −100 penalty back to the policy.

```
telemetry ─► Role 1: STIX + 50 ms SKG (Neo4j) ─► Role 2: TGN (500 ms / 128) ─► p_t, Z_t
                ▲                                                     │
                │                         Role 1: risk (Dijkstra to crown jewels)
                │                                                     ▼
   G_t+1 mutation ◄── Shuffle / TheHive ◄── Role 3: zone Double DQN + XAI (Tiers 1–3)
                │
          analyst rollback ──► −100 penalty ──► online policy update
```

---

## 2. Every module and script

### Entry points

| File | Role |
|---|---|
| `main.py` | Command-line entry. `demo` replays the held-out CICIDS2017 Friday through all three roles with Neo4j, Shuffle and TheHive, then checks a TheHive case, an analyst rollback over HTTP, the audit window, Neo4j soft delete and Algorithm 1 recovery. `run` processes any source (CICIDS CSV, Wazuh alerts, Mordor, Zeek, CloudTrail, JSON). |
| `src/pipeline/orchestrator.py` | **The heart of the system** (`XTGAIAPipeline`). It ingests each event (Role 1), batches edges into Δinf windows (500 ms or 128 events), calls the TGN (Role 2), computes risk, and applies the live gates: burn-in threshold, sustained evidence, act-only-if p ≥ 0.5, cooldown (B12). It then calls Role 3, runs the XAI tiers in parallel, enforces containment on the stream (B13), times every stage (MTTD, MTTR, end-to-end) and serves the rollback API. |
| `src/schemas/state_vector.py` | The shared 16-d state contract between the roles (Eq. 9): Z_t (8) ‖ p_t ‖ Risk_t ‖ Φ_t (6). |
| `src/schemas/stix_schema.py` | The shared STIX 2.1 observable and active-edge contract between Role 1 and Role 2. |

### Role 1: ingestion, graph and risk (`src/role1_graph/`)

| File | Role |
|---|---|
| `flow_sources.py` | Readers for CICIDS2017 CSVs (original and corrected release), Wazuh `alerts.json` and JSON dumps. Defines the 16 flow statistics the TGN uses (`FLOW_FEATURES`). |
| `telemetry.py` | Host telemetry (Mordor / Sysmon, Zeek, CloudTrail) turned into the paper's SKG types: Host, User, IPAddress, Process, File nodes; COMMUNICATED, AUTHENTICATED, WROTE_FILE, CREATE_CHILD_PROCESS, RUN edges. |
| `wazuh_mordor.py` | Stage 1 via Wazuh: Mordor events go through the Wazuh manager and are read back from `archives.json` (A3). |
| `stix_normalizer.py` | Phase 1: raw flow → STIX 2.1 bundle and active-edge dict. |
| `dual_timescale_skg.py` | **Δc = 50 ms.** `DualTimescaleSKG`: NetworkX directed graph with a 50 ms TTL on event time (Algorithm 1, lines 3–5, 17). Expired edges are evicted; nodes with no edge leave V_act and write their latent state back. Batched, retried Neo4j writes; latent states tagged with the model fingerprint (B7). |
| `live_graph.py` | `LiveGraphService`: the single in-RAM SKG shared by all roles. Edge friction ω (Eq. 7–8), ATT&CK technique per port, host context, Neo4j sync, latent-state fetch and store, and the hooks Role 3 uses to sever and restore edges. |
| `host_context.py` | Asset inventory (crown jewels, zones, privilege tiers, CVEs) and Φ_t: is_admin, is_crown_jewel, conn_count, auth_fails (300 s window), cpu_spike, cvss_max (B14). |
| `risk_engine.py`, `graph_engine.py` | Algorithm 2: Exploit = 0.5·S_vuln + 0.3·S_priv + 0.2·S_port; ω = 1 − Exploit; Dijkstra to each crown jewel; Risk = p_t · Σ Criticality / (D + 1)² (Eq. 5). |
| `threat_intel.py` | NVD REST API client (CVSS per CVE, cached) and MITRE ATT&CK mapping (port → technique). |

### Role 2: detection and detection XAI (`src/role2_ai/`)

| File | Role |
|---|---|
| `tgn_model.py` | **The TGN** (`XTGAIATGN`): 128-d GRU memory (Eq. 3); 8-head, 2-layer temporal attention over the 2-hop neighbourhood (10 recent neighbours, then 3 of each of theirs); Z_t = projection of [attention embedding ‖ memory] (A1); p_t from Z_t (Eq. 4); auxiliary attack-category head (B2). |
| `train_tgn.py` | Training (section 4.1). `train_streaming` is the procedure used for all reported results; also trains PGExplainer and saves model, feature metadata, threshold and explainer in one checkpoint. |
| `preprocess_cicids.py` | Builds the training table: binary labels, benign sampling, per-label attack cap, IP → node id, time rebased, z-score clipped to ±5. |
| `inference.py` | `TGNInferenceEngine`: loads the checkpoint and applies the TGN to the live stream (`observe_batch` updates memory and scores flows); node scores, Z_t, attack category, Algorithm 1 latent export and hydration, model fingerprint, explanations. |
| `explainers.py` | **XAI Tiers 1 + 2.** Tier 1 PGExplainer: an edge-mask network scores each edge of the 2-hop neighbourhood; reports top-k edges, fidelity+/− (probability and log-odds) and sparsity. Tier 2 attention rollout: Π(0.5·A + 0.5·I) over the layers gives the temporal trace and attack onset. |
| `xai_structural.py`, `xai_temporal.py` | The PGExplainer mask network and the attention-rollout / onset logic used by `explainers.py`. |
| `xai_bundler.py` | Packages Tiers 1–2 into a STIX 2.1 extension bundle (part of X_t). |
| `attack_classes.py` | 16-class attack taxonomy shared by CIC-IDS2017/2018 and UNSW-NB15 (multi-class scoring). |

### Role 3: decision, SOAR and feedback (`src/role3_decision/`)

| File | Role |
|---|---|
| `actions.py` | The five actions (a0–a4), their Shuffle playbook names, and which ones cut graph edges (a2, a4; A5). |
| `config.py` | Loads `configs/reward_weights.yaml`: reward weights, costs, severities, normalisation, hyper-parameters. |
| `state.py` | Validates and normalises the raw 16-d state for the policy (B14). |
| `pettingzoo_env.py` | Training environment (PettingZoo AEC): a CALDERA-driven attacker against a defender. Reward = Eq. 10 channels with proportional-response shaping (A6). |
| `zone_env.py` | Multi-agent version: one independent Double DQN per zone (*servers*, *workstations*); training and loading of zone policies. |
| `caldera.py` | Reads the real MITRE CALDERA stockpile (abilities, adversary profiles mapped to ATT&CK tactics) for the attacker (B5). |
| `double_dqn.py` | **Reward-decomposed Double DQN.** Shared trunk (16 → 64 → 64), three Q heads (contain, disrupt, cost), Q = 1.0·Q_c − 0.5·Q_d − 0.1·Q_k; Double-DQN targets; proportional prioritised replay. |
| `strategic_xai.py` | **XAI Tier 3.** Exact reward decomposition of the chosen action, decision margin over the runner-up, and an analyst narrative. |
| `soar_dispatcher.py` | Per alert: run the policy → Tier 3 → POST to the Shuffle webhook → mutate G_act (Algorithm 4 Phase 1; MTTR ends here, A2) → open a TheHive case with the evidence and X_t (Phase 2) → record in the ledger. |
| `soar_client.py` | HTTP clients for Shuffle webhooks and TheHive 5 (settings from `.env`). |
| `graph_interface.py` | In-RAM topology mutation: quarantine (cut all edges, Risk 0, state Isolated) and isolate port (cut only that port's edges); restore on rollback. |
| `decision_ledger.py` | Remembers each action (case id → state, action, host) for rollback and the audit-window commit. |
| `feedback_rollback.py` | Rollback / approve API (FastAPI). Rollback reverts the Shuffle action, restores edges, lifts enforcement and stores (S, a, −100, S′). Approval or audit-window timeout stores the normal transition (Algorithm 4 Phase 2). |
| `online_update.py` | Closed-loop policy update after ≥ 4 feedback transitions, kept only if ≥ 90 % of probe decisions stay unchanged (B14). |

### Evaluation (`src/evaluation/`)

| File | Role |
|---|---|
| `data.py` | Loads the corrected CICIDS2017 (and NF-UNSW-NB15-v2, CSE-CIC-IDS2018) and builds the protocol splits. |
| `baselines.py`, `run_eval.py` | Random Forest, XGBoost, MLP, 1D-CNN and E-GraphSAGE baselines under each protocol (P-CIC split A, chronological, held-out day); one JSON line per run. |
| `tgn_eval.py` | Trains and evaluates the TGN under each protocol (`train_streaming`), including fixed-FPR and weighted-threshold metrics and the flow read-out (B9). |
| `metrics.py` | F1, FPR, ROC-AUC, TPR at fixed FPR, the threshold search, CCSTGN-style fixed-FPR and weighted-threshold metrics, multi-class metrics. |
| `report.py` | Aggregates all runs into `docs/evaluation_results.md` (mean ± 95 % CI); shows published numbers only where the comparison is DIRECT. |
| `baseline_refs.py` | Published comparator values with tags (DIRECT, PROTOCOL-ONLY, CONTEXT-ONLY, QUALITATIVE); refuses to tabulate context-only values or mixed units. |
| `explainer_baselines.py` | LIME, TreeSHAP and KernelSHAP on the [3]-style models: latency, fidelity, E-XAI metrics, stability (1,000 instances). |
| `xai_stability.py` | Explanation stability: cosine between attribution vectors and edge-identity overlap. |
| `response_eval.py` | Zone Double DQN against static and graded SOAR playbooks and no response, in the CALDERA environment. |

### Scripts (`scripts/`)

| Script | Role |
|---|---|
| `live_demo.py` + `demo_script.txt` | Presentation demo: type traffic, watch every stage react (`docs/live_demo_guide.md`). |
| `make_demo_sample.py` | Builds the committed 30,000-flow warm-up sample from the Friday capture. |
| `latency_bench.py` / `.sh` | One-command latency and end-to-end benchmark for the hardware hand-off (`docs/handoff_latency_and_demo.md`). |
| `design_check.py`, `design_check_c3.py`, `design_check_dupes.py` | The frozen-rule experiment on a flow-level head, plus its duplicate check (B10). |
| `tune_tgn.py` | Validation-only tuning of the TGN under a frozen protocol (B11). |
| `retrain_role2.sh` | Queued Role 2 retraining runs (held-out CICIDS, then UNSW). |
| `final_extra_metrics.py` | Duplicate-free, 15-minute window and strict / lenient node metrics, plus the detection-time XAI benchmark. |
| `make_comparison_figures.py`, `make_window_figure.py` | The figures in `docs/figures/` (comparisons; the two time windows). |

---

## 3. One event, end to end

1. **Ingest (Role 1).** A flow arrives (CSV row, Wazuh alert, …) → STIX 2.1 bundle → edge in the 50 ms SKG
   with friction ω (CVSS of the target, privilege of the source, port class) and an ATT&CK technique.
   Edges older than 50 ms are evicted; hosts leaving V_act write their 128-d memory to Neo4j.
   **Acquisition** is timed here.
2. **Batch (Δinf).** The orchestrator collects edges until they span 500 ms of event time or 128 events.
   A live source also flushes when idle.
3. **Detect (Role 2).** Hosts seen for the first time get their memory restored from Neo4j (Algorithm 1).
   One TGN memory update follows: GRU on [memory ‖ flow features ‖ time]. Then attention over the 2-hop
   neighbourhood gives Z_t and p_t for each flow's destination host. **Analysis** is timed here, and
   **MTTD** ends here.
4. **Risk (Role 1).** For every active host: Dijkstra over ω to each crown jewel, Risk = p_t · Σ
   Criticality / (D + 1)².
5. **Gates (B12).** A flow is flagged if p_t ≥ the alert threshold (set by the burn-in). The host is
   escalated only if its recent mean score is also above it over ≥ 4 events, and acted on only if p_t ≥ 0.5.
6. **Decide (Role 3).** State S = [Z_t ‖ p_t ‖ Risk ‖ Φ_t] → the zone's Double DQN → action and Q-values.
   **Decision** is timed here. A dispatch is suppressed if the policy picked NoOp, the host is already
   contained, or the 15-minute cooldown is active.
7. **Explain (in parallel).** Tier 1 PGExplainer, Tier 2 attention rollout, Tier 3 reward decomposition →
   STIX bundle X_t (200 ms real-time bound).
8. **Act (Algorithm 4 Phase 1).** POST to the Shuffle webhook → a2 / a4 cut edges in G_act (Neo4j soft
   delete; Risk 0 for quarantine) → containment enforced on further traffic. **Execution**, **MTTR** and
   **end-to-end** end here.
9. **Audit (Phase 2).** TheHive case with the evidence and X_t. Analyst approval or a 300 s timeout commits
   the transition to the replay buffer. A rollback reverts Shuffle, restores the edges and stores −100.

---

## 4. Training methods

### 4.1 TGN (Role 2)

| Aspect | Method |
|---|---|
| Data | CICIDS2017, corrected release (Engelen et al.; "Attempted" flows relabelled benign). Held-out-day protocol: train Monday–Thursday, test Friday (an unseen day). Also NF-UNSW-NB15-v2, chronological 70/10/20 (CCSTGN protocol). Optional CSE-CIC-IDS2018 pre-training (B3). |
| Training table | Benign flows sampled at 10 %, at most 20,000 flows per attack label (seeded). Validation and test rows are never sampled. 16 CICFlowMeter statistics, log transform + z-score clipped to ±5. IPs mapped to node ids; time in seconds from t0. Per-day temporal split (70/15/15) so every attack family appears in validation. |
| Architecture | TGN memory 128-d, GRU update, last-message aggregator. Tokens = [memory ‖ flow features ‖ time encoding(Δt)] for the node, its 10 latest interactions and 3 of each neighbour's. Two masked 8-head attention layers. Z_t = MLP([h(v) ‖ m_v]) → 8-d (A1). p_t = sigmoid head on Z_t. Category head on [h(v) (stop-gradient) ‖ flow message]. |
| Procedure (`train_streaming`) | Every epoch replays the whole event stream in time order, as in deployment. Memory sees every event; the loss uses train rows only. Class-weighted BCE (pos_weight = benign / attack). Category cross-entropy on a detached embedding, so p_t is unaffected. Adam (lr 2e-4, weight decay 1e-5), gradient clipping 1.0, batches of 128. Early stopping on validation ROC-AUC (patience 3, at most 10 epochs); keeps the best epoch's weights and end-of-stream memory. |
| Threshold ω | Exact search on validation: best F1 among thresholds with FPR ≤ 1 %. Live, a burn-in on unlabelled traffic raises it to the score exceeded by 1 % of baseline traffic (B12). |
| PGExplainer | Trained afterwards on a frozen copy of the TGN, without labels. The training stream is replayed and the mask network learns to keep p_t with few edges (loss = prediction term + 0.005·mask size + 0.01·mask entropy; Adam lr 3e-3, ≤ 400 steps). |
| Tuning (B11) | Frozen protocol, validation only: batch, neighbours per hop, learning rate, class weighting, flow read-out. Only the max read-out passed; no test-set gain. The deployed model is the pre-tuning, destination read-out model. |
| Design check (B10) | A flow-level head reached the MLP's F1 but was rejected: it adds a second threat score, and its gain comes from per-flow features, not the TGN. |

### 4.2 Double DQN (Role 3)

| Aspect | Method |
|---|---|
| Environment | PettingZoo; attacker replays real CALDERA adversary profiles (ability chains → ATT&CK tactics); one defender agent per zone (servers, workstations); host criticality tiers per zone. |
| State | 16-d S = [Z_t ‖ p_t ‖ Risk ‖ Φ_t], normalised (risk → 1 − e^(−Risk/2), caps on counts, sigmoid on Z_t). |
| Actions / reward | a0–a4 with costs {0, 1, 2, 2, 5}. R = 1.0·R_contain − 0.5·R_disrupt − 0.1·R_cost (Eq. 10). R_contain = risk reduction; R_disrupt = criticality × severity. Proportional-response shaping inside those channels (A6). |
| Network | Shared trunk 16 → 64 → 64, three Q heads (one per reward channel) → exact Tier 3 decomposition. |
| Learning | Double DQN (the online net picks a′, the target net evaluates it), γ 0.95, lr 5e-4, batch 64, target update every 250 steps, ε 1.0 → 0.02, proportional prioritised replay (α 0.6, β 0.4 → 1); `train_zone_policies` runs 10,000 episodes by default. |
| Closed loop | Rollback → (S, a, −100, S′); audit commit → normal transition. Online update after ≥ 4 transitions, kept only if probe decisions stay ≥ 90 % unchanged. |

### 4.3 Baselines

RF, XGBoost, MLP, 1D-CNN (flow features) and E-GraphSAGE (flow-graph GNN), trained in-house on the same
splits and seeds. LIME, TreeSHAP and KernelSHAP measured on the same hardware. Static and graded SOAR
playbooks in the same response environment.

---

## 5. Likely questions, short answers

- **Why a TGN rather than a flow classifier?** Attacks show up as sequences of interactions. The TGN keeps
  per-host memory across time and neighbours, so p_t reflects how a host has been behaving, not just one
  flow. It also gives the embedding Z_t and attention weights that the policy and the explanations use.
- **Why do flow classifiers beat it on UNSW?** p_t is a node score. NF-UNSW-NB15 has only 41 hosts, so one
  score must cover all of a host's benign and malicious flows. A flow-level head matched the MLP (0.950 vs
  0.955) but was rejected by design (B10). It is reported as a limitation with future work.
- **What do the two windows do?** Δc = 50 ms keeps the graph the risk engine walks small and current. Δinf =
  500 ms (capped at 128 events, A4) sets how often the TGN updates memory, matching its training batches.
  See `docs/figures/fig5_time_windows.png`.
- **Why a burn-in threshold?** ω calibrated on the training network does not transfer to a new network:
  65 % of Friday's benign flows exceeded it. The burn-in learns the live baseline without labels.
- **How is it trained without leaking the test set?** Selection uses validation only (frozen protocols in
  `docs/design_check_rules.md` and `docs/tuning_protocol.md`); each test set was touched once.
- **How good are the explanations?** Fast (25 ms; 99.8 % within 200 ms) and stable (edge-identity 0.984).
  But fidelity is about 0: p_t depends mostly on the node's own memory, so masking neighbourhood edges
  barely moves it. Fix: explain the memory path too (future work).
- **Why is end-to-end latency above AIR's?** About two thirds of it is the synchronous Shuffle webhook on a
  laptop VM shared with the SOC stack. Without SOC services, the model path ran in 57.9 ms (AIR 58.3). The
  paper-hardware run will replace the laptop numbers.
- **Why reward decomposition?** Three Q heads make the trade-off exact and visible: containment benefit
  vs business disruption vs cost, and the margin over the runner-up action.
- **What happens on a false positive?** The analyst rolls back: Shuffle reverts, edges and risk are
  restored, and the policy receives a −100 penalty for that state and action. Updates are guarded so one
  rollback cannot destabilise the policy.
- **Is the latent memory persistent?** Yes. Each host's 128-d memory (512 bytes) is written to Neo4j and
  restored on restart, tagged with the model fingerprint so a retrained model never loads stale memory (B7).
  Only a node's own memory is restored, not its neighbours' (B15).
