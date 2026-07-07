#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMMON_ENV="$ROOT/scripts/project_runtime_env.sh"
if [ -f "$COMMON_ENV" ]; then
  # shellcheck source=/dev/null
  source "$COMMON_ENV"
fi
INTERVAL_SEC="${INTERVAL_SEC:-600}"
cd "$ROOT"

while true; do
  echo "[vnext-v2-prepare-loop] $(date '+%F %T') refresh start"
  /bin/bash "$ROOT/scripts/run_vnext_endpoint_v2_prepare.sh" || true
  echo "[vnext-v2-prepare-loop] $(date '+%F %T') refresh done; sleeping ${INTERVAL_SEC}s"
  sleep "$INTERVAL_SEC"
done
