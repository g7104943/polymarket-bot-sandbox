#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMMON_ENV="$ROOT/scripts/project_runtime_env.sh"
if [ -f "$COMMON_ENV" ]; then
  # shellcheck source=/dev/null
  source "$COMMON_ENV"
fi
PYTHON_BIN="${PYTHON_BIN:-/Users/mac/miniforge3/bin/python3}"
WORKERS="${WORKERS:-8}"
DAYS="${DAYS:-180}"
SKIP_PULL="${SKIP_PULL:-0}"
REFRESH_ALL_PATHS="${REFRESH_ALL_PATHS:-0}"

cd "$ROOT"

if [ "$SKIP_PULL" != "1" ]; then
  echo "[pipeline] pull polymarket 1m path days=${DAYS} workers=${WORKERS}"
  PULL_ARGS=(--assets BTC_USDT ETH_USDT --days "$DAYS" --workers "$WORKERS")
  if [ "$REFRESH_ALL_PATHS" = "1" ]; then
    PULL_ARGS+=(--refresh-all)
  fi
  "$PYTHON_BIN" -u scripts/pull_polymarket_1m_path_vnext.py "${PULL_ARGS[@]}"
else
  echo "[pipeline] skip pull polymarket 1m path"
fi

echo "[pipeline] train entry-exit compare-only models"
"$PYTHON_BIN" -u scripts/train_vnext_btceth_entry_exit_v1.py --force-episodes

echo "[pipeline] start writer"
./reload_launchctl_prediction_writer_vnext_btceth_entry_exit_v1.sh restart

echo "[pipeline] start raw compare-only multi"
./reload_launchctl_multi_trading_monitor_only_entry_exit_v1.sh restart

echo "[pipeline] start overlay reconciler"
./reload_launchctl_overlay_vnext_btceth_entry_exit_v1.sh restart

echo "[pipeline] run one-shot overlay sync"
"$PYTHON_BIN" -u scripts/reconcile_vnext_btceth_entry_exit_overlay.py --once

echo "[pipeline] done"
