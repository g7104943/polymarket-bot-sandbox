#!/bin/bash
# 监控合并交易日志中的「买方竞争」与「无买单深度数据」
# 用于确认：1) 有 bids 时不再被清空  2) 区分 真的没竞争 vs 缺数据
#
# 用法: ./polymarket/scripts/monitor_bid_depth_logs.sh [持续秒数]
# 不传参数则跑 120 秒后退出并打印统计

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PM="$ROOT/polymarket"
DURATION="${1:-120}"

LOG="$PM/multi_ensemble_stdout.log"
[ -f "$LOG" ] || { echo "日志不存在: $LOG"; exit 1; }

echo "监控 $LOG 中 买方竞争 / 无买单深度数据 (${DURATION}s)"
echo "  有「(无买单深度数据)」= 缺 bids 数据；仅有「买方竞争=\$0」= 有数据且竞争为 0"
echo ""

START_LINES=$(wc -l < "$LOG")
sleep "$DURATION"
END_LINES=$(wc -l < "$LOG")
# 只统计监控期间新增的几行里的 买方竞争（取文件末尾一段再 grep）
tail -n 500 "$LOG" | grep "买方竞争" > /tmp/monitor_bid_$$.txt
NEW=$(cat /tmp/monitor_bid_$$.txt)
WITH_NOTE=$(echo "$NEW" | grep "无买单深度数据" | wc -l | tr -d ' ')
ZERO_ONLY=$(echo "$NEW" | grep "买方竞争=\$0)" | grep -v "无买单深度" | wc -l | tr -d ' ')
NONZERO=$(echo "$NEW" | grep -E "买方竞争=\$[1-9]" | wc -l | tr -d ' ')
rm -f /tmp/monitor_bid_$$.txt

echo "--- 统计 (最近 500 行内 买方竞争 相关: 新增约 $((END_LINES - START_LINES)) 行) ---"
echo "  买方竞争=0 且带 (无买单深度数据): $WITH_NOTE 条"
echo "  买方竞争=0 无备注 (有数据、真的没竞争): $ZERO_ONLY 条"
echo "  买方竞争>0: $NONZERO 条"
echo ""
if [ "$WITH_NOTE" -gt 0 ]; then
  echo "  示例 (无买单深度数据):"
  echo "$NEW" | grep "无买单深度数据" | tail -3
  echo ""
fi
if [ "$NONZERO" -gt 0 ]; then
  echo "  示例 (买方竞争>0):"
  echo "$NEW" | grep -E "买方竞争=\$[1-9]" | tail -3
fi
echo "--- 结束 ---"
