#!/usr/bin/env python3
"""
Post-only vs Taker 对比报告生成器
═══════════════════════════════════════════════════════════
读取所有 logs_* 目录下的:
  - prediction_trades.json (taker 交易记录)
  - postonly_trades.json   (post-only 影子交易记录)
生成 CSV 对比排名报告。

Usage:
  cd /Users/mac/polyfun
  python3 polymarket/postonly_comparison_report.py
"""

import json
import os
import sys
import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ============================================================
# 配置
# ============================================================

POLYMARKET_DIR = Path(__file__).parent
TAKER_FILE = "prediction_trades.json"
POSTONLY_FILE = "postonly_trades.json"
OUTPUT_CSV = POLYMARKET_DIR.parent / "data" / "postonly_vs_taker_comparison.csv"


def load_json(filepath: Path) -> list:
    if not filepath.exists():
        return []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, IOError):
        return []


def extract_symbol_from_combo(combo_name: str) -> str:
    """从 logs_* 目录名提取币种 (BTC/ETH/SOL/XRP/MULTI)"""
    name = combo_name.lower()
    if name.startswith("v5_exp") or name.startswith("ensemble"):
        return "MULTI"
    for sym in ["btc", "eth", "sol", "xrp"]:
        if sym in name:
            return sym.upper()
    return "MULTI"


def classify_model(combo_name: str) -> str:
    """分类模型类型"""
    name = combo_name.lower()
    if name.startswith("ensemble"):
        return "Ens"
    if name.startswith("v5_exp"):
        return "Exp"
    if "超参" in combo_name:
        return "超参"
    if "gru" in name:
        return "GRU"
    return "旧"


def analyze_taker_trades(trades: list) -> dict:
    """分析 taker 交易记录"""
    settled = [t for t in trades if t.get("result") in ("win", "lose")]
    wins = [t for t in settled if t["result"] == "win"]
    losses = [t for t in settled if t["result"] == "lose"]
    total_pnl = sum(t.get("pnl", 0) for t in settled)
    total_bet = sum(t.get("amount", 0) for t in settled)

    return {
        "total_trades": len(trades),
        "settled": len(settled),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(settled) * 100 if settled else 0,
        "total_pnl": total_pnl,
        "avg_pnl": total_pnl / len(settled) if settled else 0,
        "total_bet": total_bet,
        "roi": total_pnl / total_bet * 100 if total_bet > 0 else 0,
    }


def analyze_postonly_trades(trades: list) -> dict:
    """分析 post-only 影子交易记录"""
    eligible = [t for t in trades if t.get("eligible", False)]
    filled = [t for t in eligible if t.get("filled", False)]
    settled = [t for t in trades if t.get("result") in ("win", "lose", "unfilled")]
    filled_settled = [t for t in settled if t.get("result") in ("win", "lose")]
    wins = [t for t in trades if t.get("result") == "win"]
    losses = [t for t in trades if t.get("result") == "lose"]
    unfilled = [t for t in trades if t.get("result") == "unfilled"]
    total_pnl = sum(t.get("pnl", 0) for t in trades)
    total_bet_filled = sum(t.get("betAmount", 0) for t in filled_settled)

    # 平均排队深度
    depth_values = [t.get("depthAheadAtEntry", 0) for t in eligible]
    avg_depth = sum(depth_values) / len(depth_values) if depth_values else 0

    # 平均成交时间
    fill_times = []
    for t in filled:
        if t.get("createdAt") and t.get("filledAt"):
            try:
                created = datetime.fromisoformat(t["createdAt"].replace("Z", "+00:00"))
                filled_at = datetime.fromisoformat(t["filledAt"].replace("Z", "+00:00"))
                diff_sec = (filled_at - created).total_seconds()
                if diff_sec >= 0:
                    fill_times.append(diff_sec)
            except (ValueError, TypeError):
                pass
    avg_fill_time = sum(fill_times) / len(fill_times) if fill_times else 0

    # 成交方法统计
    price_through = sum(1 for t in filled if t.get("fillMethod") == "price_through")
    queue_consumed = sum(1 for t in filled if t.get("fillMethod") == "queue_consumed")

    return {
        "total": len(trades),
        "eligible": len(eligible),
        "filled": len(filled),
        "unfilled": len(unfilled),
        "fill_rate": len(filled) / len(eligible) * 100 if eligible else 0,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(filled_settled) * 100 if filled_settled else 0,
        "total_pnl": total_pnl,
        "avg_pnl": total_pnl / len(filled_settled) if filled_settled else 0,
        "total_bet": total_bet_filled,
        "roi": total_pnl / total_bet_filled * 100 if total_bet_filled > 0 else 0,
        "avg_depth_ahead": avg_depth,
        "avg_fill_time_sec": avg_fill_time,
        "price_through": price_through,
        "queue_consumed": queue_consumed,
    }


def generate_report():
    """扫描所有 logs_* 目录，生成对比报告"""
    log_dirs = sorted(POLYMARKET_DIR.glob("logs_*"))
    if not log_dirs:
        print("未找到任何 logs_* 目录")
        sys.exit(1)

    rows = []
    total_taker_pnl = 0
    total_postonly_pnl = 0
    total_taker_trades = 0
    total_postonly_eligible = 0
    total_postonly_filled = 0
    dirs_with_postonly = 0

    for log_dir in log_dirs:
        combo_name = log_dir.name.replace("logs_", "")
        symbol = extract_symbol_from_combo(combo_name)

        taker_trades = load_json(log_dir / TAKER_FILE)
        postonly_trades = load_json(log_dir / POSTONLY_FILE)

        if not taker_trades and not postonly_trades:
            continue

        taker = analyze_taker_trades(taker_trades)
        postonly = analyze_postonly_trades(postonly_trades)

        if postonly["total"] > 0:
            dirs_with_postonly += 1

        total_taker_pnl += taker["total_pnl"]
        total_postonly_pnl += postonly["total_pnl"]
        total_taker_trades += taker["settled"]
        total_postonly_eligible += postonly["eligible"]
        total_postonly_filled += postonly["filled"]

        model_type = classify_model(combo_name)
        rows.append({
            "model": combo_name,
            "type": model_type,
            "symbol": symbol,
            "taker_trades": taker["settled"],
            "taker_wins": taker["wins"],
            "taker_losses": taker["losses"],
            "taker_win_rate": round(taker["win_rate"], 1),
            "taker_pnl": round(taker["total_pnl"], 2),
            "taker_avg_pnl": round(taker["avg_pnl"], 2),
            "taker_roi": round(taker["roi"], 2),
            "postonly_eligible": postonly["eligible"],
            "postonly_filled": postonly["filled"],
            "postonly_fill_rate": round(postonly["fill_rate"], 1),
            "postonly_wins": postonly["wins"],
            "postonly_losses": postonly["losses"],
            "postonly_win_rate": round(postonly["win_rate"], 1),
            "postonly_pnl": round(postonly["total_pnl"], 2),
            "postonly_avg_pnl": round(postonly["avg_pnl"], 2),
            "postonly_roi": round(postonly["roi"], 2),
            "avg_depth_ahead": round(postonly["avg_depth_ahead"], 0),
            "avg_fill_time_sec": round(postonly["avg_fill_time_sec"], 1),
            "fill_method_price_through": postonly["price_through"],
            "fill_method_queue_consumed": postonly["queue_consumed"],
        })

    # 按 postonly_pnl 降序排列
    rows.sort(key=lambda r: r["postonly_pnl"], reverse=True)

    # 写入 CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # 打印摘要
    print("=" * 70)
    print("  Post-only vs Taker 对比报告")
    print("=" * 70)
    print(f"  扫描目录: {len(log_dirs)} 个 logs_* 目录")
    print(f"  有效模型: {len(rows)} 个 (有交易记录)")
    print(f"  含 PostOnly 数据: {dirs_with_postonly} 个")
    print()
    print(f"  === Taker 汇总 ===")
    print(f"  总交易数:  {total_taker_trades}")
    print(f"  总 PnL:    ${total_taker_pnl:+,.2f}")
    print()
    print(f"  === Post-only 汇总 ===")
    print(f"  总 eligible: {total_postonly_eligible}")
    print(f"  总 filled:   {total_postonly_filled}")
    print(f"  总 fill rate: {total_postonly_filled/total_postonly_eligible*100:.1f}%" if total_postonly_eligible > 0 else "  总 fill rate: N/A")
    print(f"  总 PnL:      ${total_postonly_pnl:+,.2f}")
    print()

    # 打印 Top 15 by postonly_pnl
    if rows and any(r["postonly_eligible"] > 0 for r in rows):
        print("  === Top 15 Post-only PnL ===")
        header = f"  {'类型':<4} {'模型/组合':<38} {'币种':<6} {'Taker_PnL':>10} {'PO_Fill%':>8} {'PO_PnL':>10} {'PO_WR%':>7} {'Depth':>6}"
        print(header)
        print("  " + "-" * 100)
        top = [r for r in rows if r["postonly_eligible"] > 0][:15]
        for r in top:
            print(f"  {r['type']:<4} {r['model']:<38} {r['symbol']:<6} {r['taker_pnl']:>+10.2f} {r['postonly_fill_rate']:>7.1f}% {r['postonly_pnl']:>+10.2f} {r['postonly_win_rate']:>6.1f}% {r['avg_depth_ahead']:>6.0f}")
    elif rows:
        print("  (尚无 Post-only 影子数据，等待模型重启后积累)")

    # 按 taker_pnl 排名
    print()
    taker_sorted = sorted(rows, key=lambda r: r["taker_pnl"], reverse=True)
    print("  === Top 15 Taker PnL ===")
    header2 = f"  {'类型':<4} {'模型/组合':<38} {'币种':<6} {'笔数':>5} {'胜率':>7} {'PnL':>10} {'ROI':>8}"
    print(header2)
    print("  " + "-" * 90)
    for r in taker_sorted[:15]:
        print(f"  {r['type']:<4} {r['model']:<38} {r['symbol']:<6} {r['taker_trades']:>5} {r['taker_win_rate']:>6.1f}% {r['taker_pnl']:>+10.2f} {r['taker_roi']:>+7.2f}%")

    # 按类型汇总
    print()
    print("  === 按模型类型汇总 ===")
    type_stats = defaultdict(lambda: {"pnl": 0, "trades": 0, "po_pnl": 0, "po_filled": 0, "count": 0})
    for r in rows:
        t = r.get("type", "未知")
        type_stats[t]["pnl"] += r["taker_pnl"]
        type_stats[t]["trades"] += r["taker_trades"]
        type_stats[t]["po_pnl"] += r["postonly_pnl"]
        type_stats[t]["po_filled"] += r["postonly_filled"]
        type_stats[t]["count"] += 1
    for t in ["旧", "超参", "GRU", "Exp", "Ens"]:
        s = type_stats.get(t)
        if not s or s["count"] == 0:
            continue
        print(f"  {t:<4} ({s['count']:>3}个): Taker PnL ${s['pnl']:>+10,.2f} ({s['trades']}笔) | PostOnly PnL ${s['po_pnl']:>+8,.2f} ({s['po_filled']}笔成交)")

    print()
    print(f"  CSV 已保存: {OUTPUT_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    generate_report()
