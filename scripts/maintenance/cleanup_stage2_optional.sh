#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/mac/polyfun"

if [ "${1:-}" != "--yes" ]; then
  echo "This step is optional and more aggressive."
  echo "Run with: $0 --yes"
  exit 0
fi

rm -rf "$ROOT/recovery_readonly_20260228_203420"
rm -rf "$ROOT/polymarket/backup_2026-02-28_104845"
rm -rf "$ROOT/.specstory"

echo "stage2 optional cleanup done"
