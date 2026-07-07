#!/usr/bin/env python3
"""
用实际数据核对：融合日志里某周期某币种为 SKIP 时，Node 是否仍对该周期该币种下单。
- 从 ensemble_writer_stdout.log 解析每根 K 线的 BTC/ETH TRADE 或 SKIP；
- 从 logs_ensemble_*/prediction_trades.json 解析每笔成交的 (ts, symbol)（ts 从 marketSlug 提取）；
- 按「周期 ts + 币种」对齐，统计「融合 SKIP 但存在成交」的次数。
"""
import json
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta

# 北京 UTC+8
TZ = timezone(timedelta(hours=8))
PERIOD_SEC = 900

def parse_ensemble_log(log_path: Path):
    """解析融合日志，返回 list of (h, mn, btc_skip, eth_skip)。"""
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
        blocks.append((h, mn, btc_skip, eth_skip))
        i += 1
    return blocks

def ts_to_bar_end_local(ts: int):
    """周期开始 ts -> 周期结束的 (date, hour, minute) 北京时."""
    end_ts = ts + PERIOD_SEC
    dt = datetime.fromtimestamp(end_ts, tz=TZ)
    return (dt.date(), dt.hour, dt.minute)

def bar_end_local_to_ts(date, hour, minute):
    """北京时 周期结束 (date, hour, minute) -> 周期开始 ts."""
    end_dt = datetime(date.year, date.month, date.day, hour, minute, 0, tzinfo=TZ)
    return int(end_dt.timestamp()) - PERIOD_SEC

def load_trades_ts_symbol(trades_path: Path):
    """从 prediction_trades.json 取出 (ts, symbol) 列表。"""
    with open(trades_path, "r", encoding="utf-8") as f:
        trades = json.load(f)
    out = []
    for t in trades:
        slug = t.get("marketSlug") or ""
        sym = (t.get("symbol") or "").strip()
        if not slug or not sym:
            continue
        mm = re.search(r"-(\d{10,})$", slug)
        if not mm:
            continue
        ts = int(mm.group(1))
        out.append((ts, sym))
    return out

def assign_dates_to_blocks(blocks, ref_date, ref_h, ref_m):
    """给 blocks 里的 (h, mn) 分配日期。"""
    ref_idx = None
    for idx, (h, mn, _, _) in enumerate(blocks):
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

def main():
    base = Path(__file__).resolve().parent.parent
    log_path = base / "polymarket" / "logs_ensemble" / "ensemble_writer_stdout.log"
    trades_path = base / "polymarket" / "logs_ensemble_bp0460" / "prediction_trades.json"

    if not log_path.exists():
        print(f"融合日志不存在: {log_path}")
        return
    if not trades_path.exists():
        print(f"成交记录不存在: {trades_path}")
        return

    blocks = parse_ensemble_log(log_path)
    print(f"融合日志解析: 共 {len(blocks)} 个周期 (BTC/ETH 决策)")

    trades = load_trades_ts_symbol(trades_path)
    traded = set((ts, sym) for ts, sym in trades)

    if not trades:
        print("无成交记录，无法对齐")
        return
    min_ts = min(t[0] for t in trades)
    ref_date, ref_h, ref_m = ts_to_bar_end_local(min_ts)
    dates = assign_dates_to_blocks(blocks, ref_date, ref_h, ref_m)
    if not dates or all(d is None for d in dates):
        print("无法将融合日志与成交日期对齐（需同一天有融合与成交）")
        return

    skip_for_bar = {}
    for idx, (h, mn, btc_skip, eth_skip) in enumerate(blocks):
        d = dates[idx] if idx < len(dates) else None
        if d is None:
            continue
        bar_start_ts = bar_end_local_to_ts(d, h, mn)
        skip_for_bar[(bar_start_ts, "BTC")] = btc_skip
        skip_for_bar[(bar_start_ts, "ETH")] = eth_skip

    violations = [(ts, sym) for (ts, sym) in traded if skip_for_bar.get((ts, sym), False)]

    print(f"成交记录: 共 {len(trades)} 笔 (去重周期+币种: {len(traded)} 个)")
    print(f"融合 SKIP 但 Node 有下单（违规）: {len(violations)} 个")
    if violations:
        print("  示例 (ts, symbol):")
        for (ts, sym) in violations[:20]:
            d = datetime.fromtimestamp(ts, tz=TZ)
            print(f"    {ts} {sym}  ({d.strftime('%Y-%m-%d %H:%M')} 周期)")
    else:
        print("  未发现「SKIP 却下单」的违规。")

if __name__ == "__main__":
    main()
