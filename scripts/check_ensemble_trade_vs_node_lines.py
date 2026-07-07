#!/usr/bin/env python3
"""
对齐融合日志与 Node 准备下单条数：融合说 TRADE 的个数 vs 各 combo 的「N 条」。
若融合 2 TRADE 而某 combo 出现 0 条，则可能误过滤；若 1 条则多为 combo 阈值或融合 1 SKIP。
"""
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict

TZ = timezone(timedelta(hours=8))
PERIOD_SEC = 900


def parse_fusion_blocks(log_path: Path):
    """解析融合日志，返回 [(bar_end_h, bar_end_mn, n_trade), ...] 其中 n_trade=0,1,2。"""
    blocks = []
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.search(r"(\d{1,2}):(\d{2}):(\d{2})\s+\[Ensemble\].*🚀.*触发融合", line)
        if not m:
            i += 1
            continue
        h, mn = int(m.group(1)), int(m.group(2))
        btc_skip, eth_skip = True, True
        for _ in range(2):
            i += 1
            if i >= len(lines):
                break
            l = lines[i]
            if "BTC:" in l:
                btc_skip = "SKIP" in l
            elif "ETH:" in l:
                eth_skip = "SKIP" in l
        n_trade = (0 if btc_skip else 1) + (0 if eth_skip else 1)
        blocks.append((h, mn, n_trade))
        i += 1
    return blocks


def parse_node_prepare_log(log_path: Path):
    """解析 multi_ensemble_stdout.log 中 准备下单 ts=... N 条，返回 [(ts, combo, n)]。"""
    out = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            # [ensemble_bp0530] 准备下单 ts=1772027100 (21:45–22:00): 1 条, limitPrice=$0.53
            m = re.search(r"\[([^\]]+)\].*准备下单 ts=(\d+).*?:\s*(\d+)\s*条", line)
            if m:
                combo, ts, n = m.group(1), int(m.group(2)), int(m.group(3))
                out.append((ts, combo, n))
    return out


def assign_dates_to_blocks(blocks, ref_date, ref_h, ref_m):
    """给 blocks 的 (h, mn) 分配日期，返回 [date or None, ...]。"""
    ref_idx = None
    for idx, (h, mn, _) in enumerate(blocks):
        if h == ref_h and mn == ref_m:
            ref_idx = idx
            break
    if ref_idx is None:
        return []
    dates = [None] * len(blocks)
    dates[ref_idx] = ref_date
    for i in range(ref_idx + 1, len(blocks)):
        prev_d = dates[i - 1]
        if prev_d is None:
            continue
        prev_h, prev_mn = blocks[i - 1][0], blocks[i - 1][1]
        h, mn = blocks[i][0], blocks[i][1]
        if (prev_h, prev_mn) == (23, 45) and (h, mn) == (0, 0):
            dates[i] = prev_d + timedelta(days=1)
        else:
            dates[i] = prev_d
    for i in range(ref_idx - 1, -1, -1):
        next_d = dates[i + 1]
        if next_d is None:
            continue
        h, mn = blocks[i][0], blocks[i][1]
        next_h, next_mn = blocks[i + 1][0], blocks[i + 1][1]
        if (h, mn) == (23, 45) and (next_h, next_mn) == (0, 0):
            dates[i] = next_d - timedelta(days=1)
        else:
            dates[i] = next_d
    return dates


def bar_end_local_to_ts(date, hour, minute):
    end_dt = datetime(date.year, date.month, date.day, hour, minute, 0, tzinfo=TZ)
    return int(end_dt.timestamp()) - PERIOD_SEC


def main():
    base = Path(__file__).resolve().parent.parent
    pm = base / "polymarket"
    fusion_log = pm / "logs_ensemble" / "ensemble_writer_stdout.log"
    node_log = pm / "multi_ensemble_stdout.log"
    if not fusion_log.exists() or not node_log.exists():
        print("缺少融合或 Node 日志")
        return
    blocks = parse_fusion_blocks(fusion_log)
    node_lines = parse_node_prepare_log(node_log)
    if not blocks:
        print("融合日志无触发块")
        return
    if not node_lines:
        print("Node 日志无准备下单行")
        return
    # 用 Node 里出现过的 ts 做日期锚点
    ts_ref = node_lines[-1][0]
    ref_date = datetime.fromtimestamp(ts_ref + PERIOD_SEC, tz=TZ).date()
    ref_h = datetime.fromtimestamp(ts_ref + PERIOD_SEC, tz=TZ).hour
    ref_m = datetime.fromtimestamp(ts_ref + PERIOD_SEC, tz=TZ).minute
    dates = assign_dates_to_blocks(blocks, ref_date, ref_h, ref_m)
    fusion_by_ts = {}
    for idx, (h, mn, n_trade) in enumerate(blocks):
        d = dates[idx] if idx < len(dates) else None
        if d is None:
            continue
        bar_start = bar_end_local_to_ts(d, h, mn)
        fusion_by_ts[bar_start] = n_trade
    # Node: 按 ts 汇总各 combo 的 条数
    by_ts = defaultdict(list)
    for ts, combo, n in node_lines:
        by_ts[ts].append((combo, n))
    # 对齐：融合说 N TRADE 时，Node 各 combo 的条数
    ok = 0
    low = 0
    zero = 0
    zero_examples = []
    for ts, combos_n in sorted(by_ts.items()):
        expect = fusion_by_ts.get(ts)
        if expect is None:
            continue
        min_n = min(n for _, n in combos_n)
        max_n = max(n for _, n in combos_n)
        if min_n == 0 and expect >= 1:
            zero += 1
            if len(zero_examples) < 5:
                zero_examples.append((ts, expect, [(c, n) for c, n in combos_n if n == 0]))
        elif max_n < expect:
            low += 1
        else:
            ok += 1
    total = ok + low + zero
    print("融合 vs Node 准备下单条数 对齐（仅统计有融合决策的 ts）")
    print(f"  融合 2 TRADE 且 Node 至少有一 combo 2 条: {ok} 次")
    print(f"  融合 TRADE 数 > Node 最大条数（阈值等导致）: {low} 次")
    print(f"  融合至少 1 TRADE 但某 combo 0 条（可能误过滤）: {zero} 次")
    if zero_examples:
        print("  0 条示例 (ts, 融合TRADE数, 0条的combo):")
        for ts, exp, combos in zero_examples:
            dt = datetime.fromtimestamp(ts, tz=TZ)
            names = [c for c, n in combos]
            print(f"    {ts} ({dt.strftime('%Y-%m-%d %H:%M')}) 融合={exp} 0条={names[:3]}{'...' if len(names)>3 else ''}")
    # 结论
    if zero == 0:
        print("结论: 未发现「融合说 TRADE 但 Node 某 combo 0 条」的误过滤。")
    else:
        print("结论: 存在 combo 0 条的情况，多为该 combo 置信度阈值更高（如 bp0530 0.526），非 should_trade 误过滤。")

if __name__ == "__main__":
    main()
