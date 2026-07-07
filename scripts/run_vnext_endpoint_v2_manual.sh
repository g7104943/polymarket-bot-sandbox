#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/Users/mac/miniforge3/bin/python3}"

cd "$ROOT"

ts() {
  date +"%Y-%m-%dT%H:%M:%S%z"
}

log() {
  echo "[vnext-v2][$(ts)] $*"
}

run_stage() {
  local stage="$1"
  shift
  log "start ${stage}"
  "$@"
  log "done ${stage}"
}

if pgrep -f "scripts/train_vnext_btceth_entry_exit_v1.py --force-episodes" >/dev/null 2>&1; then
  log "v1 is still running; refusing to start full v2 pipeline."
  log "Wait for v1 to finish, or set VNEXT_ALLOW_WHILE_V1_RUNNING=1 to bypass."
  if [[ "${VNEXT_ALLOW_WHILE_V1_RUNNING:-0}" != "1" ]]; then
    exit 2
  fi
fi

if pgrep -f 'scripts/vnext_endpoint_v2.py (build-execution-dataset|train-execution|build-profit-relabel|train-profit-alpha)' >/dev/null 2>&1; then
  log "another v2 manual stage is already running; refusing to start duplicate pipeline."
  log "Set VNEXT_ALLOW_DUPLICATE_V2=1 to bypass."
  if [[ "${VNEXT_ALLOW_DUPLICATE_V2:-0}" != "1" ]]; then
    exit 2
  fi
fi

run_stage readiness "$PYTHON_BIN" scripts/vnext_endpoint_v2.py readiness
run_stage assert-manual-ready "$PYTHON_BIN" scripts/vnext_endpoint_v2.py assert-manual-ready
run_stage build-execution-dataset "$PYTHON_BIN" scripts/vnext_endpoint_v2.py build-execution-dataset
run_stage train-execution "$PYTHON_BIN" scripts/vnext_endpoint_v2.py train-execution
run_stage build-profit-relabel "$PYTHON_BIN" scripts/vnext_endpoint_v2.py build-profit-relabel
run_stage train-profit-alpha "$PYTHON_BIN" scripts/vnext_endpoint_v2.py train-profit-alpha
