#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMMON_ENV="$ROOT/scripts/project_runtime_env.sh"
if [ -f "$COMMON_ENV" ]; then
  # shellcheck source=/dev/null
  source "$COMMON_ENV"
fi
PYTHON_BIN="${PYTHON_BIN:-/Users/mac/miniforge3/bin/python3}"
INITIAL_CAPITAL="${INITIAL_CAPITAL:-400}"
LOG_TS="$(date +%Y%m%d_%H%M%S)"
REALIGN_LOG="$ROOT/logs/vnext_btceth_entry_exit_realign_${LOG_TS}.log"

cd "$ROOT"

echo "[mainline-rebuild] reset compare-only lanes"
"$PYTHON_BIN" scripts/reset_vnext_mainline_compare_only.py --initial-capital "$INITIAL_CAPITAL"

echo "[mainline-rebuild] start v1 retrain followup"
if { screen -ls 2>/dev/null || true; } | rg -q "\.vnext_v1_realign[[:space:]]"; then
  screen -S vnext_v1_realign -X quit || true
fi
screen -dmS vnext_v1_realign bash -lc "cd '$ROOT' && '$ROOT/scripts/run_vnext_btceth_entry_exit_followup.sh' >> '$REALIGN_LOG' 2>&1"

echo "[mainline-rebuild] start v1 watcher"
if { screen -ls 2>/dev/null || true; } | rg -q "\.vnext_v1_watch[[:space:]]"; then
  screen -S vnext_v1_watch -X quit || true
fi
screen -dmS vnext_v1_watch bash -lc "cd '$ROOT' && while true; do python scripts/watch_vnext_entry_exit_v1_status.py --run-audit --sample-size 30; sleep 60; done"

echo "[mainline-rebuild] start execution recorder"
if { screen -ls 2>/dev/null || true; } | rg -q "\.vnext_exec_obs[[:space:]]"; then
  screen -S vnext_exec_obs -X quit || true
fi
screen -dmS vnext_exec_obs bash -lc "cd '$ROOT' && /bin/bash scripts/run_vnext_execution_observation_recorder.sh >> '$ROOT/logs/vnext_execution_observation_recorder.log' 2>&1"

echo "[mainline-rebuild] start v2 prepare loop"
if { screen -ls 2>/dev/null || true; } | rg -q "\.vnext_v2_prepare_loop[[:space:]]"; then
  screen -S vnext_v2_prepare_loop -X quit || true
fi
screen -dmS vnext_v2_prepare_loop bash -lc "cd '$ROOT' && /bin/bash scripts/run_vnext_endpoint_v2_prepare_loop.sh >> '$ROOT/logs/vnext_endpoint_v2_prepare_loop.log' 2>&1"

echo "[mainline-rebuild] restart supervisor"
/bin/bash scripts/run_vnext_mainline_supervisor.sh restart

echo "[mainline-rebuild] realign_log=$REALIGN_LOG"
