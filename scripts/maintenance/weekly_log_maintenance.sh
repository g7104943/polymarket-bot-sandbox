#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/mac/polyfun"

"$ROOT/scripts/maintenance/rotate_active_logs.sh" 200
"$ROOT/scripts/maintenance/cleanup_logs_14d.sh" 14

echo "weekly_log_maintenance_done: $(date '+%F %T %Z')"
