#!/usr/bin/env python3
"""
Exp9 专用统计脚本 — 11 个组合的全方位分析

板块：
  1. 各组合按最终资金排名（初始 1200）
  2. 各组合按币种盈亏排名（ETH / BTC / XRP）
  3. Exp9 按币种总盈亏排名
  4. 各组合逆向选择 / 成交率统计（从 trading_stdout.log 解析累计统计）

用法：
  python3 polymarket/exp9_stats.py
  或 cd polymarket && python3 exp9_stats.py
"""

import json
import re
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent

# Exp9 初始资金
EXP9_INITIAL_CAPITAL = 800

# 全部 11 个 Exp9 组合
EXP9_LOGS = [
    "logs_v5_exp9_bp0450",
    "logs_v5_exp9_bp0460",
    "logs_v5_exp9_bp0470",
    "logs_v5_exp9_bp0480",
    "logs_v5_exp9_bp0490",
    "logs_v5_exp9_bp0500",
    "logs_v5_exp9_bp0510",
    "logs_v5_exp9_bp0520",
    "logs_v5_exp9_bp0530",
    "logs_v5_exp9_bp_dyn_0450_0530",
    "logs_v5_exp9_bp_dyn_0480_0510",
]

COINS = ["ETH", "BTC"]


def short_name(log_dir: str) -> str:
    return log_dir.replace("logs_v5_exp9_", "")


def load_trades(log_dir: str) -> list:
    """加载已执行交易列表"""
    path = SCRIPT_DIR / log_dir / "prediction_trades.json"
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    trades = []
    for t in data:
        if t.get("status") != "executed":
            continue
        symbol = (t.get("symbol") or "").strip().upper().replace("/USDT", "")
        if not symbol:
            symbol = "UNKNOWN"
        result = t.get("result")
        try:
            pnl = float(t["pnl"]) if t.get("pnl") is not None else None
        except (TypeError, ValueError):
            pnl = None
        try:
            price = float(t["tokenPrice"]) if t.get("tokenPrice") is not None else None
        except (TypeError, ValueError):
            price = None
        try:
            amount = float(t.get("amount") or 0)
        except (TypeError, ValueError):
            amount = 0
        direction = (t.get("direction") or "").strip().upper()
        if direction not in ("UP", "DOWN"):
            direction = None
        trades.append({
            "symbol": symbol,
            "result": result,
            "pnl": pnl,
            "tokenPrice": price,
            "amount": amount,
            "direction": direction,
            "marketSlug": t.get("marketSlug"),
        })
    return trades


def count_by_slug(trades: list) -> dict:
    """按 marketSlug 聚合: 每个 slug 只算 1 次胜/负，PnL 累加。
    返回 {"wins": int, "losses": int, "pending": int, "n_slugs": int, "pnl": float}
    """
    slugs: dict = {}
    for t in trades:
        slug = t.get("marketSlug") or f"_fallback_{id(t)}"
        if slug not in slugs:
            slugs[slug] = {"result": None, "pnl": 0.0}
        if t["pnl"] is not None:
            slugs[slug]["pnl"] += t["pnl"]
        if t["result"] in ("win", "lose"):
            slugs[slug]["result"] = t["result"]
    wins = sum(1 for s in slugs.values() if s["result"] == "win")
    losses = sum(1 for s in slugs.values() if s["result"] == "lose")
    pending = sum(1 for s in slugs.values() if s["result"] is None)
    total_pnl = sum(s["pnl"] for s in slugs.values())
    return {"wins": wins, "losses": losses, "pending": pending, "n_slugs": len(slugs), "pnl": total_pnl}


def load_report_summary(log_dir: str) -> dict | None:
    """从 report_summary.json 读取排名统计"""
    path = SCRIPT_DIR / log_dir / "reports" / "report_summary.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    summary = data.get("summary") if isinstance(data, dict) else None
    if not summary or not isinstance(summary, dict):
        return None
    wins = summary.get("wins")
    losses = summary.get("losses")
    total_pnl = summary.get("totalPnL")
    current_capital = summary.get("currentCapital")
    total_trades = summary.get("totalTrades") or summary.get("completedTrades")
    if wins is None or losses is None or total_trades is None:
        return None
    completed = int(wins) + int(losses)
    wr = (int(wins) / completed * 100) if completed else 0.0
    if current_capital is not None:
        fc = round(float(current_capital), 2)
    elif total_pnl is not None:
        fc = round(EXP9_INITIAL_CAPITAL + float(total_pnl), 2)
    else:
        return None
    pnl_val = float(total_pnl) if total_pnl is not None else (fc - EXP9_INITIAL_CAPITAL)
    return {
        "n_trades": int(total_trades),
        "wins": int(wins),
        "losses": int(losses),
        "completed": completed,
        "win_rate": wr,
        "total_pnl": pnl_val,
        "final_capital": fc,
    }


def parse_adverse_selection(log_dir: str) -> dict | None:
    """从 trading_stdout.log 解析最新的 '累计统计' 行"""
    log_path = SCRIPT_DIR / log_dir / "trading_stdout.log"
    if not log_path.exists():
        return None
    # 读最后 200 行寻找最新累计统计
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return None
    pattern = re.compile(
        r"累计统计:\s*成交率\s*([\d.]+)%\s*\((\d+)/(\d+)\)\s*\|\s*成交均置信\s*([\d.]+|-)%?\s*\|\s*未成交均置信\s*([\d.]+|-)%?"
    )
    last_match = None
    for line in reversed(lines):
        m = pattern.search(line)
        if m:
            last_match = m
            break
    if not last_match:
        return None
    fill_rate_str, filled_str, total_str, avg_filled_str, avg_unfilled_str = last_match.groups()
    return {
        "fill_rate": float(fill_rate_str),
        "filled": int(filled_str),
        "total": int(total_str),
        "avg_conf_filled": float(avg_filled_str) if avg_filled_str != "-" else None,
        "avg_conf_unfilled": float(avg_unfilled_str) if avg_unfilled_str != "-" else None,
    }


def _get_xrp_pnl(log_dir: str) -> float:
    """从 report_summary.json 获取 XRP 的 PnL"""
    path = SCRIPT_DIR / log_dir / "reports" / "report_summary.json"
    if not path.exists():
        return 0.0
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return float(data.get("bySymbol", {}).get("XRP", {}).get("pnl", 0))
    except Exception:
        return 0.0


def main():
    print()
    print("=" * 74)
    print("  Exp9 专用统计（11 个组合 · 初始资金 $800）")
    print("=" * 74)

    # ━━━ 板块 1：按最终资金排名 ━━━
    combo_stats = []
    for log_dir in EXP9_LOGS:
        report = load_report_summary(log_dir)
        if report:
            xrp_pnl = _get_xrp_pnl(log_dir)
            report["total_pnl"] -= xrp_pnl
            report["final_capital"] = EXP9_INITIAL_CAPITAL + report["total_pnl"]
            combo_stats.append({"log_dir": log_dir, **report})
            continue
        trades = load_trades(log_dir)
        trades = [t for t in trades if t["symbol"] != "XRP"]
        total_pnl = sum((t["pnl"] or 0) for t in trades)
        slug_stats = count_by_slug(trades)
        completed = slug_stats["wins"] + slug_stats["losses"]
        wr = (slug_stats["wins"] / completed * 100) if completed else 0
        combo_stats.append({
            "log_dir": log_dir,
            "n_trades": len(trades),
            "wins": slug_stats["wins"],
            "losses": slug_stats["losses"],
            "completed": completed,
            "win_rate": wr,
            "total_pnl": total_pnl,
            "final_capital": round(EXP9_INITIAL_CAPITAL + total_pnl, 2),
            "n_slugs": slug_stats["n_slugs"],
            "pending_slugs": slug_stats["pending"],
        })
    combo_stats.sort(key=lambda x: x["final_capital"], reverse=True)

    print()
    print("─" * 74)
    print("  板块 1: 各组合按最终资金排名")
    print("─" * 74)
    print("  (胜率按 marketSlug 聚合: 每个市场周期只算 1 次胜/负)")
    for rank, s in enumerate(combo_stats, 1):
        name = short_name(s["log_dir"])
        pending_str = f" 待结{s.get('pending_slugs', 0)}" if s.get('pending_slugs', 0) else ""
        print(f"  {rank:2d}. {name:22s}  资金 ${s['final_capital']:8.2f}  "
              f"盈亏 ${s['total_pnl']:+8.2f}  "
              f"已结 {s['completed']:3d} (胜{s['wins']:2d}/负{s['losses']:2d})  "
              f"胜率 {s['win_rate']:5.1f}%  "
              f"({s['n_trades']}单){pending_str}")

    # ━━━ 板块 2：按币种盈亏排名（按 slug 聚合胜率）━━━
    # 收集所有 exp9 交易按 log_dir × symbol 分组，胜率按 slug 聚合
    coin_model_data = defaultdict(lambda: defaultdict(lambda: {
        "pnl": 0.0, "wins": 0, "losses": 0, "completed": 0,
        "n_trades": 0, "pending": 0, "up": 0, "down": 0,
        "prices": [], "slugs": {},
    }))
    for log_dir in EXP9_LOGS:
        trades = load_trades(log_dir)
        for t in trades:
            sym = t["symbol"]
            if sym not in COINS:
                continue
            e = coin_model_data[sym][log_dir]
            e["n_trades"] += 1
            if t["direction"] == "UP":
                e["up"] += 1
            elif t["direction"] == "DOWN":
                e["down"] += 1
            if t["tokenPrice"] is not None:
                e["prices"].append(t["tokenPrice"])
            if t["pnl"] is not None:
                e["pnl"] += t["pnl"]
            # 按 slug 聚合胜率
            slug = t.get("marketSlug") or f"_fallback_{id(t)}"
            if slug not in e["slugs"]:
                e["slugs"][slug] = None
            if t["result"] in ("win", "lose"):
                e["slugs"][slug] = t["result"]
        # 计算 slug 聚合后的胜负数
        for sym2 in COINS:
            e2 = coin_model_data[sym2][log_dir]
            e2["wins"] = sum(1 for r in e2["slugs"].values() if r == "win")
            e2["losses"] = sum(1 for r in e2["slugs"].values() if r == "lose")
            e2["completed"] = e2["wins"] + e2["losses"]
            e2["pending"] = sum(1 for r in e2["slugs"].values() if r is None)

    coin_totals = {}

    print()
    print("─" * 74)
    print("  板块 2: 各组合按币种盈亏排名")
    print("─" * 74)

    for sym in COINS:
        models = coin_model_data.get(sym, {})
        rows = []
        for log_dir in EXP9_LOGS:
            e = models.get(log_dir, {
                "pnl": 0.0, "wins": 0, "losses": 0, "completed": 0,
                "n_trades": 0, "pending": 0, "up": 0, "down": 0, "prices": [],
            })
            rows.append((log_dir, e))
        rows.sort(key=lambda x: x[1]["pnl"], reverse=True)

        total_pnl = sum(r[1]["pnl"] for r in rows)
        total_wins = sum(r[1]["wins"] for r in rows)
        total_losses = sum(r[1]["losses"] for r in rows)
        total_completed = total_wins + total_losses
        total_pending = sum(r[1]["pending"] for r in rows)
        total_up = sum(r[1]["up"] for r in rows)
        total_down = sum(r[1]["down"] for r in rows)
        total_wr = (total_wins / total_completed * 100) if total_completed else 0
        coin_totals[sym] = total_pnl

        print()
        print(f"  ┌─ {sym} ────────────────────────────────────────────────────────────┐")
        print(f"  │ 合计: 已结 {total_completed} 笔 (胜 {total_wins} / 负 {total_losses})"
              f"  待结 {total_pending}  Up {total_up} / Down {total_down}")
        print(f"  │ 胜率 {total_wr:.1f}%  总盈亏 ${total_pnl:+.2f}")
        print(f"  └─────────────────────────────────────────────────────────────────────┘")
        for rank, (log_dir, e) in enumerate(rows, 1):
            wr = (e["wins"] / e["completed"] * 100) if e["completed"] else 0
            avg_p = f"${sum(e['prices'])/len(e['prices']):.4f}" if e["prices"] else "—"
            pending_str = f" 待{e['pending']}" if e["pending"] else ""
            print(f"    {rank:2d}. {short_name(log_dir):22s}  ${e['pnl']:+8.2f}  "
                  f"已结{e['completed']:3d} (胜{e['wins']:2d}/负{e['losses']:2d})  "
                  f"胜率{wr:5.1f}%  均价{avg_p}{pending_str}")

    # ━━━ 板块 3：按币种总盈亏排名 ━━━
    print()
    print("─" * 74)
    print("  板块 3: Exp9 按币种总盈亏排名")
    print("─" * 74)
    sorted_coins = sorted(coin_totals.items(), key=lambda x: x[1], reverse=True)
    grand_total = sum(v for _, v in sorted_coins)
    for rank, (sym, pnl) in enumerate(sorted_coins, 1):
        print(f"    {rank}. {sym:5s}  ${pnl:+.2f}")
    print(f"    ──────────────────")
    print(f"    合计   ${grand_total:+.2f}")

    # ━━━ 板块 4：逆向选择 / 成交率统计 ━━━
    print()
    print("─" * 74)
    print("  板块 4: 逆向选择 · 成交率 · 置信度分析（从实时日志解析）")
    print("─" * 74)
    print(f"  {'组合':<22s}  {'成交率':>8s}  {'成交/总':>8s}  {'成交均置信':>10s}  {'未成交均置信':>12s}  {'差值':>6s}")
    print(f"  {'─'*22}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*12}  {'─'*6}")

    all_filled = 0
    all_total = 0
    sum_conf_filled = 0.0
    sum_conf_unfilled = 0.0
    n_conf_filled = 0
    n_conf_unfilled = 0

    for log_dir in EXP9_LOGS:
        name = short_name(log_dir)
        adv = parse_adverse_selection(log_dir)
        if adv is None:
            print(f"  {name:<22s}  {'—':>8s}  {'—':>8s}  {'—':>10s}  {'—':>12s}  {'—':>6s}")
            continue
        fr = f"{adv['fill_rate']:.1f}%"
        ratio = f"{adv['filled']}/{adv['total']}"
        cf = f"{adv['avg_conf_filled']:.1f}%" if adv['avg_conf_filled'] is not None else "—"
        cu = f"{adv['avg_conf_unfilled']:.1f}%" if adv['avg_conf_unfilled'] is not None else "—"
        # 差值 = 成交均置信 - 未成交均置信（负值 = 逆向选择）
        if adv['avg_conf_filled'] is not None and adv['avg_conf_unfilled'] is not None:
            diff = adv['avg_conf_filled'] - adv['avg_conf_unfilled']
            diff_str = f"{diff:+.1f}%"
        else:
            diff_str = "—"
        print(f"  {name:<22s}  {fr:>8s}  {ratio:>8s}  {cf:>10s}  {cu:>12s}  {diff_str:>6s}")

        all_filled += adv['filled']
        all_total += adv['total']
        if adv['avg_conf_filled'] is not None:
            sum_conf_filled += adv['avg_conf_filled'] * adv['filled']
            n_conf_filled += adv['filled']
        unfilled = adv['total'] - adv['filled']
        if adv['avg_conf_unfilled'] is not None and unfilled > 0:
            sum_conf_unfilled += adv['avg_conf_unfilled'] * unfilled
            n_conf_unfilled += unfilled

    # 汇总行
    print(f"  {'─'*22}  {'─'*8}  {'─'*8}  {'─'*10}  {'─'*12}  {'─'*6}")
    all_fr = f"{(all_filled/all_total*100):.1f}%" if all_total else "—"
    all_ratio = f"{all_filled}/{all_total}"
    all_cf = f"{(sum_conf_filled/n_conf_filled):.1f}%" if n_conf_filled else "—"
    all_cu = f"{(sum_conf_unfilled/n_conf_unfilled):.1f}%" if n_conf_unfilled else "—"
    if n_conf_filled and n_conf_unfilled:
        all_diff = (sum_conf_filled / n_conf_filled) - (sum_conf_unfilled / n_conf_unfilled)
        all_diff_str = f"{all_diff:+.1f}%"
    else:
        all_diff_str = "—"
    print(f"  {'合计':<22s}  {all_fr:>8s}  {all_ratio:>8s}  {all_cf:>10s}  {all_cu:>12s}  {all_diff_str:>6s}")

    print()
    print("  差值 = 成交均置信 - 未成交均置信")
    print("  负值 → 逆向选择（高置信预测反而不成交）")
    print("  正值 → 正向选择（高置信预测更容易成交）")

    print()
    print("=" * 74)
    print("  统计完成")
    print("=" * 74)
    print()


if __name__ == "__main__":
    main()
