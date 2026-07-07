#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/Users/mac/miniforge3/bin/python3}"
WORKERS="${WORKERS:-8}"
DAYS="${DAYS:-2}"
INTERVAL_SECS="${INTERVAL_SECS:-120}"

cd "$ROOT"

while true; do
  echo "[vnext-path-refresh] $(date '+%F %T') pull start days=${DAYS} workers=${WORKERS}"
  "$PYTHON_BIN" -u scripts/pull_polymarket_1m_path_vnext.py --assets BTC_USDT ETH_USDT --days "$DAYS" --workers "$WORKERS" || true
  echo "[vnext-path-refresh] $(date '+%F %T') reconcile once"
  "$PYTHON_BIN" -u scripts/reconcile_vnext_btceth_entry_exit_overlay.py --once || true
  sleep "$INTERVAL_SECS"
done
