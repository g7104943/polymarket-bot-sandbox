#!/usr/bin/env python3
"""
用历史日志统计 exp「该交易但不交易」：v5 预测写 TRADE 的 (ts, symbol)，对应 combo 的 prediction_trades 中无成交。
- 解析 logs_v5_exp{N}/prediction_writer_v5_stdout.log 得到 (ts, symbol, should_trade)；
- 汇总 logs_v5_exp{N}_*/prediction_trades.json 得到该 exp 下所有 (ts, symbol) 成交；
- 对齐用 period_start：v5 日志的 slug ts = period start；trades 的 marketSlug 末尾数字为 period end，period_start = ts - 900。
"""
import json
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta

TZ = timezone(timedelta(hours=8))
PERIOD_SEC = 900


def parse_v5_log(log_path: Path):
    """解析 v5 预测日志，返回 [(ts, symbol, should_trade), ...]。symbol 为 BTC/ETH/XRP。"""
    out = []
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.search(r"slug ts:\s*(\d{10,})", line)
        if not m:
            i += 1
            continue
        ts = int(m.group(1))
        i += 1
        while i < len(lines):
            l = lines[i]
            if "slug ts:" in l or "下次预测触发" in l:
                break
            # 预测汇总里 "    BTC-15m: [DOWN] 52.6% → 交易" 或 "→ 跳过"
            sym_m = re.search(r"(BTC|ETH|XRP)-15m:.*→\s*(交易|跳过)", l)
            if sym_m:
                sym = sym_m.group(1)
                should_trade = sym_m.group(2) == "交易"
                out.append((ts, sym, should_trade))
            i += 1
    return out


def load_trades_for_exp(pm_root: Path, exp_name: str):
    """汇总该 exp 下所有 combo 的 prediction_trades，返回 set of (period_start, symbol)。"""
    traded = set()
    pattern = f"logs_v5_{exp_name}_*"
    for d in pm_root.glob(pattern):
        if not d.is_dir():
            continue
        tf = d / "prediction_trades.json"
        if not tf.exists():
            continue
        try:
            with open(tf, "r", encoding="utf-8") as f:
                trades = json.load(f)
        except Exception:
            continue
        for t in trades:
            slug = t.get("marketSlug") or ""
            sym = (t.get("symbol") or "").strip()
            if not slug or not sym:
                continue
            mm = re.search(r"-(\d{10,})$", slug)
            if not mm:
                continue
            period_end = int(mm.group(1))
            period_start = period_end - PERIOD_SEC
            traded.add((period_start, sym))
    return traded


def main():
    base = Path(__file__).resolve().parent.parent
    pm = base / "polymarket"
    exp_name = "exp10"
    log_path = pm / f"logs_v5_{exp_name}" / "prediction_writer_v5_stdout.log"
    if not log_path.exists():
        print(f"v5 日志不存在: {log_path}")
        return
    blocks = parse_v5_log(log_path)
    traded = load_trades_for_exp(pm, exp_name)
    should_trade_but_no_trade = [(ts, sym) for (ts, sym, st) in blocks if st and (ts, sym) not in traded]
    total_trade_decisions = sum(1 for _, _, st in blocks if st)
    total_skip = sum(1 for _, _, st in blocks if not st)
    print(f"v5 {exp_name} 日志解析: {len(blocks)} 条 (TRADE: {total_trade_decisions}, SKIP: {total_skip})")
    print(f"该 exp 下 combo 成交 (period_start, symbol) 去重: {len(traded)} 个")
    print(f"该交易但不交易（预测 TRADE 但无成交）: {len(should_trade_but_no_trade)} 个")
    if should_trade_but_no_trade:
        for (ts, sym) in should_trade_but_no_trade[:15]:
            dt = datetime.fromtimestamp(ts, tz=TZ)
            print(f"  {ts} {sym}  ({dt.strftime('%Y-%m-%d %H:%M')})")
    else:
        print("  无。")


if __name__ == "__main__":
    main()
