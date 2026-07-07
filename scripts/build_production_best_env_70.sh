#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPORTS="$ROOT/reports"
OUT="$REPORTS/production_best_70.env"

RUNTIME_ENV_70="$REPORTS/runtime_guard_best_70.env"
RUNTIME_ENV="$REPORTS/runtime_guard_best.env"
ENSEMBLE_MULTI_ENV="$REPORTS/ensemble_params_full_multiseed_best.env"
ENSEMBLE_FULL_ENV="$REPORTS/ensemble_params_full_best.env"
ENSEMBLE_ENV="$REPORTS/ensemble_params_best.env"

# 默认最小置信度可被命令行覆盖
MIN_CONF_70="${1:-0.70}"

if [ -f "$RUNTIME_ENV_70" ]; then
  RT="$RUNTIME_ENV_70"
elif [ -f "$RUNTIME_ENV" ]; then
  RT="$RUNTIME_ENV"
else
  echo "❌ 缺少 $RUNTIME_ENV_70 或 $RUNTIME_ENV"
  exit 1
fi

if [ -f "$ENSEMBLE_MULTI_ENV" ]; then
  ENS="$ENSEMBLE_MULTI_ENV"
elif [ -f "$ENSEMBLE_FULL_ENV" ]; then
  ENS="$ENSEMBLE_FULL_ENV"
elif [ -f "$ENSEMBLE_ENV" ]; then
  ENS="$ENSEMBLE_ENV"
else
  echo "❌ 缺少 $ENSEMBLE_MULTI_ENV 或 $ENSEMBLE_FULL_ENV 或 $ENSEMBLE_ENV"
  exit 1
fi

tmp="$(mktemp)"
{
  echo "# generated_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# source_runtime=$RT"
  echo "# source_ensemble=$ENS"
  echo ""
  cat "$ENS"
  echo ""
  cat "$RT"
  echo ""
  echo "MIN_CONFIDENCE_70=$MIN_CONF_70"
} > "$tmp"

# 去重：同名键保留最后一个值
awk '
  BEGIN { FS="=" }
  /^[[:space:]]*#/ || /^[[:space:]]*$/ { lines[++n]=$0; next }
  {
    key=$1
    val=substr($0, index($0, "=")+1)
    keys[++kcount]=key
    vals[key]=val
  }
  END {
    for (i=1; i<=n; i++) print lines[i]
    seen_count=0
    for (i=1; i<=kcount; i++) {
      key=keys[i]
      if (!(key in seen)) {
        seen[key]=1
        ordered[++seen_count]=key
      }
    }
    for (i=1; i<=seen_count; i++) {
      key=ordered[i]
      print key "=" vals[key]
    }
  }
' "$tmp" > "$OUT"

rm -f "$tmp"
echo "✅ 已生成: $OUT (MIN_CONFIDENCE_70=$MIN_CONF_70)"
