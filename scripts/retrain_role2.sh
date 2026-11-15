#!/bin/sh
# Queued Role 2 retraining (event-level category head + flow read-out study). Run from the repo root:
#
#   MSYS_NO_PATHCONV=1 docker compose -f docker/docker-compose.yml run --rm --no-deps -T \
#     -e PYTHONUNBUFFERED=1 -e XTGAIA_TORCH_THREADS=14 -e PYTHONPATH=/app:/app/data/pylib \
#     xtgaia-core sh scripts/retrain_role2.sh > data/processed/eval/retrain_role2.log 2>&1
#
# Needs the SOC stack stopped (8 GB Docker VM). Approximate CPU times: CICIDS held-out ~5 h, UNSW chrono (max) ~5 h,
# UNSW chrono (dst) ~3 h.
# Each job appends one record to data/processed/eval/results.jsonl; the report is regenerated at the end.
# Every job is independent. Each one writes <checkpoint>.partial after every epoch; rerun this script with
# XTGAIA_RESUME=1 (pass -e XTGAIA_RESUME=1) to continue an interrupted job from its last finished epoch.
set -u
export XTGAIA_VARIANT=evhead          # event-level category head (node embedding + flow message, stop-gradient)
# XTGAIA_FLOW_HEAD=hybrid (set by scripts/design_check.py only if rule R4 adopts it) adds the detached flow-level
# head to every job below; unset or empty keeps the current design.
export XTGAIA_FLOW_HEAD="${XTGAIA_FLOW_HEAD:-}"
echo "[retrain_role2] XTGAIA_FLOW_HEAD=${XTGAIA_FLOW_HEAD:-off} resume=${XTGAIA_RESUME:-0}"

# 1. CICIDS2017 held-out Friday with the event-level head (live pipeline read-out: destination node)
XTGAIA_FLOW_READOUT=dst python -m src.evaluation.tgn_eval --procedure streaming --protocol heldout --seed 0 --epochs 10
# 2. NF-UNSW-NB15-v2 chronological (CCSTGN [11] protocol), flow score = max(p_t(src), p_t(dst))
XTGAIA_FLOW_READOUT=max python -m src.evaluation.tgn_eval --procedure streaming --dataset nf_unsw_nb15_v2     --protocol chrono --seed 0 --epochs 10
# 3. NF-UNSW-NB15-v2 with the destination read-out (isolates the head change from the read-out change)
XTGAIA_FLOW_READOUT=dst python -m src.evaluation.tgn_eval --procedure streaming --dataset nf_unsw_nb15_v2     --protocol chrono --seed 0 --epochs 10
python -m src.evaluation.report --out docs/evaluation_results.md
