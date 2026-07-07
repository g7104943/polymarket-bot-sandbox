#!/usr/bin/env python3
"""
统计：因价格未达设定而进入监控队列（高价/低价）、以及最终是否在该周期内成交。
用于回答：模拟交易有没有因为价格一直没到买入价而没买？每天每个模型有多少？

用法（项目根目录）:
  python3 scripts/count_price_monitor_missed.py                    # 扫描 polymarket/logs_* 下所有交易日志
  python3 scripts/count_price_monitor_missed.py --logs-dir polymarket  # 指定日志根目录
  python3 scripts/count_price_monitor_missed.py --per-day          # 按日汇总（从周期 ts 推断日期）
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def find_trading_logs(logs_root: Path) -> list[tuple[str, Path]]:
    """返回 [(model_name, log_path), ...]，model_name 为 logs_xxx 的 xxx 部分。"""
    if not logs_root.is_dir():
        return []
    out = []
    for d in sorted(logs_root.iterdir()):
        if not d.is_dir() or not d.name.startswith("logs_"):
            continue
        model_name = d.name  # logs_eth -> logs_eth, logs_gru_eth_55_no1h4h -> logs_gru_eth_55_no1h4h 等
        # 可能的交易 stdout 文件名
        for name in ("trading_stdout.log", "trading_eth_stdout.log", "trading_btc_stdout.log",
                     "trading_xrp_stdout.log", "trading_eth_10_90_stdout.log", "trading_xrp_20_80_stdout.log"):
            p = d / name
            if p.exists():
                out.append((model_name, p))
                break
    return out


def parse_slug_ts(line: str) -> int | None:
    """从包含 *-updown-15m-{ts} 或 slug: *-updown-15m-{ts} 的行中提取 ts。"""
    m = re.search(r"updown-15m-(\d+)", line)
    return int(m.group(1)) if m else None


def parse_period_ts_from_block(lines: list[str], idx: int) -> int | None:
    """从本轮摘要/交易块附近找 target_period_end_ts（slug ts）。往前找 20 行内的 目标周期 或 slug。"""
    start = max(0, idx - 25)
    for i in range(start, idx + 1):
        ts = parse_slug_ts(lines[i])
        if ts is not None:
            return ts
    return None


def ts_to_date_beijing(ts: int) -> str:
    """Unix 秒 -> 北京时间日期 YYYY-MM-DD。"""
    from datetime import datetime, timezone, timedelta
    # ts = 周期开始，东八区
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    beijing_tz = timezone(timedelta(hours=8))
    beijing = dt.astimezone(beijing_tz)
    return beijing.strftime("%Y-%m-%d")


def scan_log_file(log_path: Path) -> dict:
    """
    扫描一个交易 stdout 日志。
    返回:
      high_price_added: [(period_ts, symbol)], 加入高价监控
      low_price_added: [(period_ts, symbol)], 加入低价监控
      triggered: set((period_ts, symbol)), 同周期内后续有「价格监控触发」或「低价监控触发」或出现「模拟成交」等
    """
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    high_added = []  # (period_ts, symbol)
    low_added = []
    triggered = set()  # (period_ts, symbol) 本周期内后来触发了
    # 从「本轮摘要」或「交易 #」块中取 period_ts；从 🔍 ETH: 取 symbol
    i = 0
    while i < len(lines):
        line = lines[i]
        # 加入高价监控
        if "价格过高" in line and "已加入高价监控队列" in line:
            m = re.search(r"🔍\s*(\w+):", line)
            symbol = (m.group(1) if m else "").strip() or "?"
            period_ts = parse_period_ts_from_block(lines, i)
            if period_ts is not None:
                high_added.append((period_ts, symbol))
            else:
                high_added.append((0, symbol))  # 无法解析 ts 时用 0
            i += 1
            continue
        # 加入低价监控
        if "价格过低" in line and ("加入低价监控队列" in line or "已加入低价监控队列" in line):
            m = re.search(r"🔍\s*(\w+):", line)
            symbol = (m.group(1) if m else "").strip() or "?"
            period_ts = parse_period_ts_from_block(lines, i)
            if period_ts is not None:
                low_added.append((period_ts, symbol))
            else:
                low_added.append((0, symbol))
            i += 1
            continue
        # 同周期内后续触发：价格监控触发 / 低价监控触发（说明后来买到了）
        if "价格监控触发" in line or "低价监控触发" in line:
            m = re.search(r"✅\s*(\w+)\s", line)
            symbol = (m.group(1) if m else "").strip() or "?"
            period_ts = parse_slug_ts(line)
            if period_ts is None:
                # 可能在下一行或前几行有 [11:45～12:00 北京时间]，没有 slug 则从前后块找
                period_ts = parse_period_ts_from_block(lines, i)
            if period_ts is not None:
                triggered.add((period_ts, symbol))
            i += 1
            continue
        # 模拟成交 / 已记录：从本块往前找「交易 #N eth-updown-15m-{ts}」或「市场: ETH-15M」
        if "模拟成交" in line or "已记录 (回测" in line:
            period_ts = None
            sym = "?"
            for j in range(max(0, i - 15), i + 1):
                ts = parse_slug_ts(lines[j])
                if ts is not None:
                    period_ts = ts
                sm = re.search(r"市场:\s+(\w+)-15M", lines[j])
                if sm:
                    sym = sm.group(1)
            if period_ts is not None:
                triggered.add((period_ts, sym))
            i += 1
            continue
        i += 1
    return {"high_added": high_added, "low_added": low_added, "triggered": triggered}


def load_executed_slugs(prediction_trades_path: Path) -> set[int]:
    """从 prediction_trades.json 加载已成交的 market slug ts（周期开始 ts）。"""
    out = set()
    if not prediction_trades_path.exists():
        return out
    try:
        data = json.loads(prediction_trades_path.read_text(encoding="utf-8"))
    except Exception:
        return out
    for t in data:
        slug = t.get("marketSlug") or ""
        m = re.search(r"updown-15m-(\d+)", slug)
        if m:
            out.add(int(m.group(1)))
    return out


def main():
    ap = argparse.ArgumentParser(description="统计因价格未达而进入监控/最终未买")
    ap.add_argument("--logs-dir", type=Path, default=Path("polymarket"), help="日志根目录（下有 logs_*）")
    ap.add_argument("--per-day", action="store_true", help="按日汇总")
    args = ap.parse_args()
    logs_root = args.logs_dir
    if not logs_root.is_absolute():
        logs_root = Path(__file__).resolve().parents[1] / logs_root
    pairs = find_trading_logs(logs_root)
    if not pairs:
        print(f"未找到任何交易日志（在 {logs_root} 下查找 logs_*/trading*stdout*.log）")
        return
    print("=" * 60)
    print("  因价格未达设定而进入监控 / 最终未买 统计")
    print("=" * 60)
    print()
    total_high = 0
    total_low = 0
    total_missed_high = 0
    total_missed_low = 0
    by_day: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"high_added": 0, "low_added": 0, "high_missed": 0, "low_missed": 0}))
    for model_name, log_path in pairs:
        res = scan_log_file(log_path)
        trades_path = log_path.parent / "prediction_trades.json"
        executed_slugs = load_executed_slugs(trades_path)
        high_added = res["high_added"]
        low_added = res["low_added"]
        triggered = res["triggered"]
        missed_high = []
        missed_low = []
        for (ts, sym) in high_added:
            total_high += 1
            if ts and (ts, sym) not in triggered and ts not in executed_slugs:
                missed_high.append((ts, sym))
                total_missed_high += 1
            if args.per_day and ts:
                day = ts_to_date_beijing(ts)
                by_day[day][model_name]["high_added"] += 1
                if (ts, sym) not in triggered and ts not in executed_slugs:
                    by_day[day][model_name]["high_missed"] += 1
        for (ts, sym) in low_added:
            total_low += 1
            if ts and (ts, sym) not in triggered and ts not in executed_slugs:
                missed_low.append((ts, sym))
                total_missed_low += 1
            if args.per_day and ts:
                day = ts_to_date_beijing(ts)
                by_day[day][model_name]["low_added"] += 1
                if (ts, sym) not in triggered and ts not in executed_slugs:
                    by_day[day][model_name]["low_missed"] += 1
        print(f"【{model_name}】")
        print(f"  加入高价监控（价格过高）: {len(high_added)} 次")
        print(f"  加入低价监控（价格过低）: {len(low_added)} 次")
        print(f"  高价监控最终未买: {len(missed_high)} 次")
        print(f"  低价监控最终未买: {len(missed_low)} 次")
        print()
    print("─" * 60)
    print("  合计")
    print("  加入高价监控: %d  加入低价监控: %d  高价最终未买: %d  低价最终未买: %d" % (total_high, total_low, total_missed_high, total_missed_low))
    print()
    if args.per_day and by_day:
        print("─" * 60)
        print("  按日统计（每日每模型）")
        print("─" * 60)
        for day in sorted(by_day.keys()):
            print(f"\n  {day}")
            for model in sorted(by_day[day].keys()):
                v = by_day[day][model]
                print(f"    {model}: 高价监控+{v['high_added']} 未买{v['high_missed']} | 低价监控+{v['low_added']} 未买{v['low_missed']}")


if __name__ == "__main__":
    main()
