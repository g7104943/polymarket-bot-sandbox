#!/bin/bash
set -euo pipefail

if [ -n "${BASH_SOURCE[0]:-}" ]; then
  PROJECT_ENV_SOURCE="${BASH_SOURCE[0]}"
elif [ -n "${ZSH_VERSION:-}" ]; then
  PROJECT_ENV_SOURCE="${(%):-%x}"
else
  PROJECT_ENV_SOURCE="$0"
fi

PROJECT_ROOT="$(cd "$(dirname "$PROJECT_ENV_SOURCE")/.." && pwd)"
POLYMARKET_ROOT="${PROJECT_ROOT}/polymarket"

PROJECT_RUNTIME_PATH_DEFAULT="/Users/mac/miniforge3/bin:/Users/mac/miniforge3/condabin:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/Users/mac/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export PROJECT_RUNTIME_PATH="${PROJECT_RUNTIME_PATH:-$PROJECT_RUNTIME_PATH_DEFAULT}"
export PATH="$PROJECT_RUNTIME_PATH"

choose_python_bin() {
  local candidates=(
    "${PYTHON_BIN:-}"
    "${TRAIN_PYTHON:-}"
    "/Users/mac/miniforge3/bin/python3"
    "/Users/mac/miniforge3/bin/python"
    "python3"
    "python"
  )
  local p
  for p in "${candidates[@]}"; do
    [ -n "${p}" ] || continue
    if command -v "$p" >/dev/null 2>&1; then
      command -v "$p"
      return 0
    fi
  done
  return 1
}

choose_node_bin() {
  local candidates=(
    "${NODE_BIN:-}"
    "/usr/local/bin/node"
    "/opt/homebrew/bin/node"
    "node"
  )
  local p
  for p in "${candidates[@]}"; do
    [ -n "${p}" ] || continue
    if command -v "$p" >/dev/null 2>&1; then
      command -v "$p"
      return 0
    fi
  done
  return 1
}

if PYTHON_RESOLVED="$(choose_python_bin 2>/dev/null)"; then
  export PYTHON_BIN="$PYTHON_RESOLVED"
fi

if NODE_RESOLVED="$(choose_node_bin 2>/dev/null)"; then
  export NODE_BIN="$NODE_RESOLVED"
fi

export ROOT="${ROOT:-$PROJECT_ROOT}"
export PM="${PM:-$POLYMARKET_ROOT}"
export PROJECT_RUNTIME_ROOT="$ROOT"
export PROJECT_RUNTIME_PM="$PM"
