#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMMON_ENV="$ROOT/scripts/project_runtime_env.sh"
if [ -f "$COMMON_ENV" ]; then
  # shellcheck source=/dev/null
  source "$COMMON_ENV"
fi
SESSION="vnext_mainline_supervisor"
PYTHON_BIN="${PYTHON_BIN:-/Users/mac/miniforge3/bin/python3}"
SCRIPT="$ROOT/scripts/supervise_vnext_mainline.py"
LOG="$ROOT/logs/vnext_mainline_supervisor.log"
LOOP_SECS="${LOOP_SECS:-60}"
PATTERN="$SCRIPT --loop-seconds"

mkdir -p "$ROOT/logs"

start() {
  if { screen -ls 2>/dev/null || true; } | rg -q "\.${SESSION}[[:space:]]"; then
    echo "session=${SESSION} already_running"
    return 0
  fi
  if pgrep -f "$PATTERN" >/dev/null 2>&1; then
    echo "session=${SESSION} already_running_orphan process_pattern=$PATTERN"
    return 0
  fi
  screen -dmS "$SESSION" bash -lc "source '$COMMON_ENV'; cd '$ROOT' && '$PYTHON_BIN' '$SCRIPT' --loop-seconds '$LOOP_SECS' >> '$LOG' 2>&1"
  echo "session=${SESSION} started log=$LOG loop_seconds=$LOOP_SECS"
}

stop() {
  if { screen -ls 2>/dev/null || true; } | rg -q "\.${SESSION}[[:space:]]"; then
    screen -S "$SESSION" -X quit || true
  fi
  pkill -f "$PATTERN" >/dev/null 2>&1 || true
  echo "session=${SESSION} stopped"
}

status() {
  if { screen -ls 2>/dev/null || true; } | rg -q "\.${SESSION}[[:space:]]"; then
    if pgrep -f "$PATTERN" >/dev/null 2>&1; then
      echo "session=${SESSION} running"
    else
      echo "session=${SESSION} screen_only"
    fi
  elif pgrep -f "$PATTERN" >/dev/null 2>&1; then
    echo "session=${SESSION} orphan_running"
  else
    echo "session=${SESSION} not_running"
  fi
}

case "${1:-restart}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  *) echo "usage: $0 {start|stop|restart|status}"; exit 1 ;;
 esac
