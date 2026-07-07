#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/mac/polyfun"
MAX_MB="${1:-200}"
TS="$(date '+%Y%m%d_%H%M%S')"

rotate_one() {
  local f="$1"
  [ -f "$f" ] || return 0
  local bytes
  bytes=$(stat -f%z "$f" 2>/dev/null || echo 0)
  local max_bytes=$((MAX_MB * 1024 * 1024))
  if [ "$bytes" -le "$max_bytes" ]; then
    return 0
  fi

  local rotated="${f}.${TS}.log"
  cp "$f" "$rotated"
  : > "$f"
  gzip -f "$rotated"
  echo "rotated: $f -> ${rotated}.gz"
}

rotate_one "$ROOT/logs/prediction_v5.log"
rotate_one "$ROOT/logs/derivatives_collector.log"
rotate_one "$ROOT/logs/online_learning.log"
rotate_one "/Users/mac/Library/Logs/polyfun/daily_training_scheduler.launchd.log"

exit 0
