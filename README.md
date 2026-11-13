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
3. **Role 3 policy** – `data/models/dqn_policy.pt` (train with `src/role3_decision/double_dqn.py`).
4. **Run**
   ```bash
   python main.py demo                                   # full-system demo on the held-out Friday
   python main.py run --source cicids --input "data/raw/cicids2017/TrafficLabelling/Friday-*.csv" --serve-rollback
   python main.py run --source wazuh --input /path/to/alerts.json --follow --serve-rollback
   ```
   Reports are written to `data/processed/*_report.json`.

`configs/asset_inventory.yaml` describes the monitored network (crown jewels, admin hosts, CVSS); replace it
with your CMDB / scanner export for another environment.
