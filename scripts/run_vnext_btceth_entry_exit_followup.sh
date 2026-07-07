#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMMON_ENV="$ROOT/scripts/project_runtime_env.sh"
if [ -f "$COMMON_ENV" ]; then
  # shellcheck source=/dev/null
  source "$COMMON_ENV"
fi
PYTHON_BIN="${PYTHON_BIN:-/Users/mac/miniforge3/bin/python3}"
DAYS="${DAYS:-180}"
WORKERS="${WORKERS:-8}"
PULL_PATTERN="pull_polymarket_1m_path_vnext.py"

cd "$ROOT"

echo "[followup] waiting for 1m pull to finish"
while pgrep -f "$PULL_PATTERN" >/dev/null 2>&1; do
  sleep 30
  echo "[followup] still waiting for 1m pull"
done

echo "[followup] 1m pull finished, starting training/startup"
SKIP_PULL=1 "$ROOT/scripts/run_vnext_btceth_entry_exit_pipeline.sh"
