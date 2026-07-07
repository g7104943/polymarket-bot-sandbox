#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/Users/mac/miniforge3/bin/python3}"

cd "$ROOT"

ts() {
  date +"%Y-%m-%dT%H:%M:%S%z"
}

log() {
  echo "[vnext-v3][$(ts)] $*"
}

run_stage() {
  local stage="$1"
  shift
  log "start ${stage}"
  "$@"
  log "done ${stage}"
}

if pgrep -f 'scripts/vnext_endpoint_v2.py (build-profit-relabel|train-profit-alpha)' >/dev/null 2>&1; then
  log "v2 manual pipeline is still running; refusing to start v3."
  log "Wait for v2 to finish, or set VNEXT_ALLOW_V3_DURING_V2=1 to bypass."
  if [[ "${VNEXT_ALLOW_V3_DURING_V2:-0}" != "1" ]]; then
    exit 2
  fi
fi

if pgrep -f '(train_vnext_sequence_v3.py|audit_vnext_sequence_v3.py|promote_vnext_sequence_v3_monitor.py)' >/dev/null 2>&1; then
  log "another v3 manual stage is already running; refusing to start duplicate pipeline."
  log "Set VNEXT_ALLOW_DUPLICATE_V3=1 to bypass."
  if [[ "${VNEXT_ALLOW_DUPLICATE_V3:-0}" != "1" ]]; then
    exit 2
  fi
fi

for cfg in \
  "$ROOT/data/models/vnext_btc_profit_alpha_v2/config.json" \
  "$ROOT/data/models/vnext_eth_profit_alpha_v2/config.json"
do
  if [[ ! -f "$cfg" ]]; then
    log "missing required v2b config: $cfg"
    exit 2
  fi
done

run_stage train-v3 "$PYTHON_BIN" scripts/train_vnext_sequence_v3.py
run_stage audit-v3 "$PYTHON_BIN" scripts/audit_vnext_sequence_v3.py
run_stage promote-v3 "$PYTHON_BIN" scripts/promote_vnext_sequence_v3_monitor.py
"$ROOT"/reload_launchctl_prediction_writer_vnext_btceth_sequence_v3.sh restart
"$ROOT"/reload_launchctl_multi_trading_monitor_only_sequence_v3.sh restart
"$ROOT"/reload_launchctl_overlay_vnext_btceth_sequence_v3.sh restart
