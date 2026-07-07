#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMMON_ENV="$ROOT/scripts/project_runtime_env.sh"
if [ -f "$COMMON_ENV" ]; then
  # shellcheck source=/dev/null
  source "$COMMON_ENV"
fi
PYTHON_BIN="${PYTHON_BIN:-/Users/mac/miniforge3/bin/python3}"

cd "$ROOT"

$PYTHON_BIN scripts/vnext_endpoint_v2.py readiness
$PYTHON_BIN scripts/vnext_endpoint_v2.py build-execution-dataset
