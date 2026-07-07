#!/usr/bin/env bash
set -euo pipefail

ROOT="/Users/mac/polyfun"
DAYS="${1:-14}"

# 只清理 .log，绝不动 json/csv/parquet 等交易与统计数据
find "$ROOT/logs" -type f -name '*.log' -mtime +"$DAYS" -print -delete
find "$ROOT/polymarket" -type f -name '*.log' -mtime +"$DAYS" -print -delete
find "/Users/mac/Library/Logs/polyfun" -type f -name '*.log' -mtime +"$DAYS" -print -delete 2>/dev/null || true

# 清理轮转后的压缩日志（保留 30 天）
find "$ROOT/logs" -type f -name '*.log.*.log.gz' -mtime +30 -print -delete 2>/dev/null || true
find "$ROOT/polymarket" -type f -name '*.log.*.log.gz' -mtime +30 -print -delete 2>/dev/null || true
find "/Users/mac/Library/Logs/polyfun" -type f -name '*.log.*.log.gz' -mtime +30 -print -delete 2>/dev/null || true

exit 0
