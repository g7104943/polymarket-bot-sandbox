#!/usr/bin/env python3
"""
按时间段看胜率：诊断「刚开始高、最近骤降」是否成立。
读各组合的 report_summary.json，按日/按周汇总胜率，便于判断是方差、 regime 变化还是过拟合。

用法:
  python3 polymarket/win_rate_by_period.py                    # 所有有 report 的 log 目录
  python3 polymarket/win_rate_by_period.py logs_v5_exp15     # 指定一个组合
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def load_report(log_dir: str) -> dict | None:
    p = SCRIPT_DIR / log_dir / "reports" / "report_summary.json"
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main():
    if len(sys.argv) > 1:
        log_dirs = [d.rstrip("/") for d in sys.argv[1:]]
    else:
        log_dirs = []
        for d in sorted(SCRIPT_DIR.iterdir()):
            if d.is_dir() and d.name.startswith("logs_") and (d / "reports" / "report_summary.json").exists():
                log_dirs.append(d.name)
    if not log_dirs:
        print("未找到任何带 report_summary.json 的 logs_* 目录")
        return 1

    for log_dir in log_dirs:
        data = load_report(log_dir)
        if not data:
            continue
        daily = data.get("dailyStats") or {}
        if not daily:
            summary = data.get("summary") or {}
            wr = summary.get("winRate")
            n = (summary.get("wins") or 0) + (summary.get("losses") or 0)
            print(f"\n{log_dir}: 无按日统计，总胜率 {wr}% (已结 {n} 笔)")
            continue
        dates = sorted(daily.keys())
        print(f"\n{log_dir}: 按日胜率 (共 {len(dates)} 天)")
        print("  日期        已结  胜  负  胜率%    PnL")
        for d in dates:
            s = daily[d]
            t = s.get("trades", 0)
            w = s.get("wins", 0)
            L = s.get("losses", 0)
            wr = s.get("winRate", (w / (w + L) * 100) if (w + L) else 0)
            pnl = s.get("pnl", 0)
            print(f"  {d}  {t:4}  {w:3}  {L:3}  {wr:5.1f}%  {pnl:+.2f}")
        # 前半段 vs 后半段
        if len(dates) >= 2:
            mid = len(dates) // 2
            first_dates, second_dates = dates[:mid], dates[mid:]
            def agg(ds):
                tw, tl, tpnl = 0, 0, 0.0
                for d in ds:
                    s = daily[d]
                    tw += s.get("wins", 0)
                    tl += s.get("losses", 0)
                    tpnl += s.get("pnl", 0)
                return tw, tl, tpnl
            w1, L1, pnl1 = agg(first_dates)
            w2, L2, pnl2 = agg(second_dates)
            n1, n2 = w1 + L1, w2 + L2
            wr1 = (w1 / n1 * 100) if n1 else 0
            wr2 = (w2 / n2 * 100) if n2 else 0
            print(f"  → 前半段({first_dates[0]}~{first_dates[-1]}): 已结{n1} 胜率{wr1:.1f}% PnL{pnl1:+.2f}")
            print(f"  → 后半段({second_dates[0]}~{second_dates[-1]}): 已结{n2} 胜率{wr2:.1f}% PnL{pnl2:+.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
