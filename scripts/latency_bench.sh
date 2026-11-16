#!/usr/bin/env bash
# XTGAIA latency / end-to-end benchmark on this machine (hand-off script).
#
#   scripts/latency_bench.sh                 # full closed loop: starts Neo4j, Shuffle, TheHive, then 3 runs
#   scripts/latency_bench.sh --no-stack      # model + graph only (no SOC services)
#   scripts/latency_bench.sh -- --limit 100000 --repeat 5   # anything after -- goes to latency_bench.py
#
# Results land in runs/latency/<timestamp>/ (latency.md for the paper table, latency.json for the raw data).
set -euo pipefail
cd "$(dirname "$0")/.."

STACK=1
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-stack) STACK=0; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

COMPOSE=(docker compose -f docker/docker-compose.yml)
COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"

if [[ $STACK -eq 1 ]]; then
  echo "[*] starting the SOC stack (Neo4j, Shuffle, TheHive) ..."
  "${COMPOSE[@]}" up -d neo4j shuffle shuffle-backend shuffle-orborus shuffle-opensearch cassandra elasticsearch thehive
  echo "[*] waiting for TheHive and Shuffle to answer (up to 5 min) ..."
  for i in $(seq 1 60); do
    th=$(docker exec xtgaia-thehive curl -s -o /dev/null -w "%{http_code}" http://localhost:9000/api/status 2>/dev/null || true)
    sh=$(docker exec xtgaia-shuffle-backend sh -c "wget -qO- http://localhost:5001/api/v1/health >/dev/null 2>&1 && echo 200" 2>/dev/null || true)
    [[ "$th" == "200" && "$sh" == "200" ]] && { echo "[*] services up after $((i * 5)) s"; break; }
    sleep 5
  done
  sleep 15   # let TheHive finish warming up its index
  ARGS=(--repeat 3 --micro ${EXTRA[@]+"${EXTRA[@]}"})
  DEPS=()
else
  ARGS=(--no-neo4j --no-soar --repeat 3 --micro ${EXTRA[@]+"${EXTRA[@]}"})
  DEPS=(--no-deps)
fi

MSYS_NO_PATHCONV=1 "${COMPOSE[@]}" run --rm ${DEPS[@]+"${DEPS[@]}"} -T \
  -e PYTHONPATH=/app:/app/data/pylib -e XTGAIA_GIT_COMMIT="$COMMIT" \
  xtgaia-core python scripts/latency_bench.py "${ARGS[@]}"
