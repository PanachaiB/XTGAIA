# Running XTGAIA on a new machine

Everything the live demo needs is in this repository:
- the deployed models (`data/models/tgn_best.pt`, `dqn_zone_policies.pt`, `dqn_policy.pt`);
- the demo inputs (`data/samples/demo_warmup.csv`, 30,000 real CICIDS2017 flows, and
  `data/samples/demo_flow_presets.json`);
- the cached NVD scores (`data/cache/nvd_cvss.json`).

The full datasets are only needed to retrain or re-run the evaluation (section 5).

## 1. Prerequisites

- **Docker Desktop** (Windows / macOS) or Docker Engine + Compose v2 (Linux). Give Docker at least 8 GB of
  RAM; 16 GB if you run the full SOC stack.
- **git.**
- About 15 GB of free disk for the images.

## 2. Get the code and build the core image

```bash
git clone <repository-url> XTGAIA
```

```bash
cd XTGAIA
```

```bash
cp .env.example .env
```

```bash
docker compose -f docker/docker-compose.yml build xtgaia-core
```

The first build downloads PyTorch and the other Python packages (about 10–20 minutes).

`.env.example` holds placeholder settings that work for a local test bed. Change the passwords if the
machine is shared.

## 3. Run the live demo (no SOC services needed)

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps xtgaia-core python scripts/live_demo.py --no-neo4j --no-soar
```

The demo warms up on 30,000 real benign flows (about 40 s), then shows the `xtgaia>` prompt. Try `benign 40`,
`scan 172.16.0.1 192.168.10.50 60`, `status`, `ddos 172.16.0.1 192.168.10.50 200`, `rollback last`. The
commands and talking points are in `docs/live_demo_guide.md`.

On Git Bash for Windows, prefix `docker compose ... run` commands with `MSYS_NO_PATHCONV=1 `.

## 4. Full system: Neo4j, Shuffle and TheHive

1. Start the services:

   ```bash
   docker compose -f docker/docker-compose.yml up -d neo4j shuffle shuffle-backend shuffle-orborus shuffle-opensearch cassandra elasticsearch thehive
   ```

   Neo4j works straight away with the `.env` credentials.
2. **TheHive** (http://localhost:9000):
   1. Log in as the default admin (see the TheHive 5 documentation) and create an organisation **XTGAIA**.
   2. Create a user in it with profile **org-admin** and set a password.
   3. Create an **API key** for that user and put it in `.env` as `THEHIVE_API_KEY`.
3. **Shuffle** (http://localhost:3001):
   1. Create the admin account on first visit.
   2. Create two workflows, each starting with a **Webhook** trigger: one for containment, one for rollback.
   3. Put their URLs in `.env` as `SHUFFLE_EXECUTION_WEBHOOK_URL` and `SHUFFLE_ROLLBACK_WEBHOOK_URL`.
   4. The workflows can simply log the payload, or call your firewall / EDR APIs.
4. Run the demo with everything on:

   ```bash
   docker compose -f docker/docker-compose.yml run --rm xtgaia-core python scripts/live_demo.py
   ```

   Each containment now appears as a Shuffle execution and a TheHive case in the XTGAIA organisation.

## 5. Reproducing the training and evaluation (optional, needs the full datasets)

| Data | Where to put it | Used for |
|---|---|---|
| CICIDS2017 corrected release (Engelen et al. 2021) | `data/raw/cicids2017_improved/{monday..friday}.csv` | training, held-out Friday evaluation, `main.py demo`, latency benchmark |
| NF-UNSW-NB15-v2 (Sarhan et al.) | `data/raw/nf_unsw_nb15_v2/` | CCSTGN-protocol evaluation |
| CSE-CIC-IDS2018 (optional) | `data/raw/cicids2018_improved/` | pre-training (B3) |
| MITRE ATT&CK enterprise STIX (`enterprise-attack.json`) | `data/raw/attack/` | ATT&CK technique names in Role 1 (optional; without it the mapping is skipped) |
| MITRE CALDERA stockpile | `data/raw/caldera/stockpile.zip` | Role 3 training environment |

With those in place:
- `python main.py demo` replays the whole Friday through the system.
- `scripts/latency_bench.sh` measures latency (see `docs/handoff_latency_and_demo.md`).
- `python -m src.evaluation.tgn_eval ...` reproduces the evaluation (`docs/evaluation_results.md`).

## Data attribution

`data/samples/demo_warmup.csv` and `data/samples/demo_flow_presets.json` are small extracts of the CICIDS2017
corrected release. Cite I. Sharafaldin, A. H. Lashkari and A. A. Ghorbani (ICISSP 2018) and G. Engelen,
V. Rimmer and W. Joosen (IEEE SPW 2021) when using them.
