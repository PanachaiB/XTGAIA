# XTGAIA

Explainable Temporal-Graph AI for autonomous SOC response. Three roles form one pipeline:

| Role | Package | Consumes | Produces |
|---|---|---|---|
| 1 Graph & risk | `src/role1_graph` | real telemetry (CICIDS2017 flows, Wazuh `alerts.json`, Sysmon/Mordor JSON) | STIX 2.1 active edges, live in-RAM SKG (+ Neo4j), host context Φ_t, reachability risk Risk_t |
| 2 TGN & detection XAI | `src/role2_ai` | Role 1 edges | threat probability p_t, 8-d latent Z_t, STIX detection-XAI bundle |
| 3 Decision & SOAR | `src/role3_decision` | `XTGAIAStateVector` (Z_t ‖ p_t ‖ Risk_t ‖ Φ_t) | Double DQN action, decision XAI, Shuffle playbook, TheHive case, in-place containment, rollback feedback |

`src/pipeline/orchestrator.py` wires them: Role 1 ingests → Role 2 scores → Role 1 computes risk with p_t →
shared 16-d state → Role 3 acts on Role 1's live graph → containment is enforced on the stream until an
analyst rolls it back (`POST /api/v1/feedback/rollback`), which feeds an R = −100 transition to the replay buffer.

## Running the integrated system

All commands run in the core container (`docker compose -f docker/docker-compose.yml run --rm xtgaia-core …`);
settings come from `.env` (see `.env.example`) and `configs/pipeline.yaml`.

1. **Data** – CICIDS2017 *GeneratedLabelledFlows* (TrafficLabelling CSVs) in `data/raw/cicids2017/TrafficLabelling/`.
2. **Train Role 2** (Monday–Thursday; Friday is held out for evaluation/demo):
   ```bash
   python src/role2_ai/preprocess_cicids.py --input "data/raw/cicids2017/TrafficLabelling/Monday-*.csv" \
     "data/raw/cicids2017/TrafficLabelling/Tuesday-*.csv" "data/raw/cicids2017/TrafficLabelling/Wednesday-*.csv" \
     "data/raw/cicids2017/TrafficLabelling/Thursday-*.csv" --output data/processed/cicids_train.csv \
     --benign-frac 0.1 --attack-cap 20000
   python src/role2_ai/train_tgn.py --data data/processed/cicids_train.csv
   ```
3. **Role 3 policy** – zone policies `data/models/dqn_zone_policies.pt` (see below); `data/models/dqn_policy.pt` is the single-agent fallback.
4. **Run**
   ```bash
   python main.py demo                                   # full-system demo on the held-out Friday
   python main.py run --source cicids --input "data/raw/cicids2017/TrafficLabelling/Friday-*.csv" --serve-rollback
   python main.py run --source wazuh --input /path/to/alerts.json --follow --serve-rollback
   ```
   Reports are written to `data/processed/*_report.json`.

## Paper components and where they live (XTGAIAv3)

| Paper element | Implementation |
|---|---|
| Stage 1: Mordor telemetry ingested via Wazuh (log collector → JSON decoder → `archives.json`) | `role1_graph/wazuh_mordor.py`, `docker/wazuh/master/ossec.conf`, `main.py run --source mordor --via-wazuh` |
| SKG schema: Host/User/IPAddress/Process/File nodes; COMMUNICATED, AUTHENTICATED, WROTE_FILE, CREATE_CHILD_PROCESS, RUN edges | `role1_graph/dual_timescale_skg.py`, `role1_graph/telemetry.py` (Sysmon 1/3/11, Security 4624/4625/4672, CloudTrail, Zeek) |
| Algorithm 1: Δc = 50 ms window, eviction, latent hydration (512-byte memory per node), async Neo4j writes | `role1_graph/live_graph.py`, `pipeline/orchestrator.py` (`_evict_latent`), `role2_ai/inference.py` |
| Algorithm 2: Eq. 5–8 risk, ω = 1 − (0.5·S_vuln + 0.3·S_priv + 0.2·S_port), NVD CVSS, MITRE ATT&CK | `role1_graph/host_context.py`, `role1_graph/threat_intel.py`, `configs/mitre_t1021_ports.json` |
| TGN: 128-d GRU memory, 8-head temporal attention over a 2-hop neighbourhood, 8-d Z_t, Δinf = 500 ms windows | `role2_ai/tgn_model.py` (`XTGAIATGN`) |
| Binary p_t (Eq. 4) drives risk and response; multi-class scoring uses an attack-category head on Z_t, hierarchical so it never contradicts p_t | `role2_ai/attack_classes.py`, `evaluation/metrics.py` (`multiclass_metrics`) |
| Training: CIC-IDS2017 with CSE-CIC-IDS2018 pre-training (Stage 3) | `role2_ai/train_tgn.py` (`init_from`), `evaluation/tgn_eval.py --pretrain-2018` |
| Algorithm 3 XAI tiers: PGExplainer, attention rollout, reward decomposition → one STIX bundle X_t (parallel, < 200 ms) | `role2_ai/explainers.py`, `role2_ai/inference.py`, `pipeline/orchestrator.py` (`build_xt`) |
| Multi-agent independent Double DQN per network zone (PettingZoo AEC), attacker driven by real MITRE CALDERA adversaries (stockpile abilities → ATT&CK tactic chains) | `role3_decision/zone_env.py`, `role3_decision/caldera.py`, `configs/reward_weights.yaml` (`zones`, `caldera`) |
| Deployment regimes offline / passive (recommend only) / active | `configs/pipeline.yaml` `role3.mode`, `main.py --mode` |
| Algorithm 4: Shuffle, Neo4j soft delete, TheHive audit, rollback R = −100, prioritised replay, guarded online update | `role3_decision/soar_dispatcher.py`, `feedback_rollback.py`, `online_update.py` |
| Ablations (risk assessor, closed loop, Neo4j risk), latency probes, XAI stability (cosine, same attack class) | `configs/pipeline.yaml` `ablation`, `report.latency`, `evaluation/xai_stability.py` |
| Response policy vs traditional SOAR playbooks | `evaluation/response_eval.py` |

Policies: `python -c "from src.role3_decision.zone_env import train_zone_policies; train_zone_policies(10000, 'data/models/dqn_zone_policies.pt', seed=0)"`.
Other telemetry: `python main.py run --source mordor --input data/raw/mordor/<dataset>.json` (`zeek`, `cloudtrail` work the same way).
Paper path through Wazuh: start the master (`docker compose -f docker/docker-compose.yml up -d wazuh-master`), then
`python main.py run --source mordor --via-wazuh --input data/raw/mordor/<dataset>/*.json`. Events are appended to
`data/wazuh/ingest/mordor.json`, decoded by the manager and read back from `data/wazuh/archives/archives.json`
with the manager's rule matches attached.

Data downloads (all gitignored under `data/raw/`): CALDERA stockpile → `data/raw/caldera/stockpile.zip`
(`https://codeload.github.com/mitre/stockpile/zip/refs/heads/master`), MITRE ATT&CK → `data/raw/attack/`,
Mordor → `data/raw/mordor/`, CSE-CIC-IDS2018 improved → `data/raw/cicids2018_improved/`.

## Evaluation protocol (paper Sec. IV.C)

`src/evaluation/` benchmarks detection on three datasets:
- the corrected CICIDS2017 (*CICIDS2017_improved*, in `data/raw/cicids2017_improved/`)
- NF-UNSW-NB15-v2 (`data/raw/nf_unsw_nb15_v2/`), CCSTGN's dataset
- CSE-CIC-IDS2018 improved (`data/raw/cicids2018_improved/*.zip`, read from the zip without extracting), used for pre-training

The protocols are:
- `pcic_a`: random 80/20, for parity with [3]
- `chrono`: chronological 70/10/20, CCSTGN [11]
- `heldout`: train Mon–Thu, test Fri

Thresholds are chosen on validation only, both at fixed FPR (0.01 / 0.1 / 1 %) and by a weighted FNR/FPR rule. The report adds 95 % CIs over seeds.
```bash
python -m src.evaluation.run_eval --dataset nf_unsw_nb15_v2 --protocols chrono --models rf xgb mlp cnn1d egraphsage --seeds 0 1 2
python -m src.evaluation.run_eval --protocols pcic_a heldout chrono --models rf xgb mlp cnn1d --seeds 0 1 2
python -m src.evaluation.tgn_eval --procedure streaming --protocol heldout --pretrain-2018 --seed 0
python -m src.evaluation.tgn_eval --procedure streaming --dataset nf_unsw_nb15_v2 --protocol chrono --seed 0
python -m src.evaluation.run_eval --protocols heldout chrono --models rf xgb --multiclass --seeds 0 1 2   # multi-class baselines
python -m src.evaluation.response_eval --episodes 2000                                            # Role 3 vs SOAR playbooks
python -m src.evaluation.report --out docs/evaluation_results.md
```

Binary detection metrics (ROC-AUC, F1, FPR/FNR at fixed FPR) score p_t. Multi-class metrics (accuracy, macro / weighted F1) score the attack category: Benign when p_t < ω, otherwise the category head's class.

`configs/asset_inventory.yaml` describes the monitored network (crown jewels, admin hosts, CVSS); replace it
with your CMDB / scanner export for another environment.
