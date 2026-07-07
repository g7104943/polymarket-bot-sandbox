#!/usr/bin/env python3
"""
全模型单币三维公平对比排名 — 实盘选型用

三维公平对比：
  维度1 ROI%（收益率）= PnL / 初始资金（单币）
    - 旧模型/Exp/Ensemble 均按单币 $400 计算
    - 意义：「投入 $1 赚了多少」— 直接回答实盘预期收益

  维度2 投注回报率 = PnL / 总下注额
    - 完全不受资金量和下注比例影响
    - 意义：「每赌 $1 赚了多少」— 纯粹衡量模型预测能力

  维度3 下注强度 = 平均每笔下注 / 初始资金
    - 意义：风险偏好指标，越高越激进

背景：
  旧模型: 单币 $400 初始, 固定5%下注 → 每笔~$13-$15
  Exp模型: $800 总额, 按币种隔离 2×$400, Kelly动态 → 每笔~$2-$30
  Ensemble: $800 总额, 按币种隔离 2×$400, Kelly动态

用法：
  python3 polymarket/exp_coin_return_ranking.py
"""

import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

COINS = ["ETH", "BTC", "XRP"]
EXP_COINS = ["ETH", "BTC"]  # Exp 系列不统计 XRP

OLD_INITIAL = 400    # 旧模型单币初始资金
EXP_INITIAL = 400    # Exp/Ensemble 单币初始资金（$800 总额, 按币种隔离 2×$400）

# ─────────────────── GRU EK 组合（活跃，limitPrice=0.46）───────────────────
OLD_MODELS = {
    # ETH
    "logs_gru_eth_ek":              {"coin": "ETH", "label": "GRU_ETH_EK"},
    "logs_gru_eth_no1h4h_ek":      {"coin": "ETH", "label": "GRU_ETH_no1h4h_EK"},
    # BTC
    "logs_gru_btc_no1h4h_ek":      {"coin": "BTC", "label": "GRU_BTC_no1h4h_EK"},
    # SOL
    "logs_gru_sol_ek":              {"coin": "SOL", "label": "GRU_SOL_EK"},
}

# ─────────────────── Exp 系列（Exp10~17, 已移除 Exp8/9）───────────────────
EXP_SERIES = {}
for n in range(10, 18):
    prefix = f"logs_v5_exp{n}"
    bps = ["0450", "0460", "0470", "0480", "0490", "0500", "0510", "0520", "0530"]
    dirs = [f"{prefix}_bp{bp}" for bp in bps]
    dirs += [f"{prefix}_bp_dyn_0450_0530", f"{prefix}_bp_dyn_0480_0510"]
    EXP_SERIES[f"Exp{n}"] = dirs

# Ensemble 多模型融合
_ens_bps = ["0450", "0460", "0470", "0480", "0490", "0500", "0510", "0520", "0530"]
_ens_dirs = [f"logs_ensemble_bp{bp}" for bp in _ens_bps]
_ens_dirs += ["logs_ensemble_bp_dyn_0450_0530", "logs_ensemble_bp_dyn_0480_0510"]
EXP_SERIES["Ensemble"] = _ens_dirs


def _load_report(log_dir: str) -> dict | None:
    """读取 report_summary.json 完整历史数据"""
    rpath = SCRIPT_DIR / log_dir / "reports" / "report_summary.json"
    if not rpath.exists():
        return None
    try:
        with open(rpath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_trades(log_dir: str) -> list:
    """加载已执行交易，包含 amount 字段"""
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
        result = t.get("result")
        try:
            pnl = float(t["pnl"]) if t.get("pnl") is not None else None
        except (TypeError, ValueError):
            pnl = None
        try:
            amount = float(t.get("amount", 0))
        except (TypeError, ValueError):
            amount = 0.0
        slug = t.get("marketSlug") or f"_fallback_{id(t)}"
        trades.append({
            "symbol": symbol, "result": result,
            "pnl": pnl, "slug": slug, "amount": amount,
        })
    return trades


def calc_coin_stats(trades: list, coin: str, log_dir: str = ""):
    """按 slug 去重统计某币的盈亏。优先用 report（完整历史），trades 用于下注金额统计。"""
    coin_trades = [t for t in trades if t["symbol"] == coin]

    # 尝试从 report 获取完整的 PnL/胜负数据
    report = _load_report(log_dir) if log_dir else None
    by_sym = report.get("bySymbol", {}) if report else {}
    sym_report = by_sym.get(coin)

    if sym_report and isinstance(sym_report, dict):
        wins = int(sym_report.get("wins", 0))
        losses = int(sym_report.get("losses", 0))
        pending = int(sym_report.get("pending", 0))
        total_pnl = float(sym_report.get("pnl", 0))
        n_trades = int(sym_report.get("trades", 0))
        completed = wins + losses
    elif not coin_trades:
        return None
    else:
        slugs = {}
        for t in coin_trades:
            slug = t["slug"]
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
        n_trades = len(coin_trades)
        completed = wins + losses

    if completed == 0 and not coin_trades:
        return None

    amounts = [t["amount"] for t in coin_trades if t["amount"] > 0]
    total_bet = sum(amounts)
    avg_bet = (total_bet / len(amounts)) if amounts else 0.0
    return {
        "pnl": total_pnl,
        "wins": wins,
        "losses": losses,
        "pending": pending,
        "completed": completed,
        "n_trades": n_trades,
        "total_bet": total_bet,
        "avg_bet": avg_bet,
    }


def build_row(coin, label, model_type, initial, stats):
    """构建一行数据，计算三维指标"""
    wr = (stats["wins"] / stats["completed"] * 100) if stats["completed"] else 0
    roi = (stats["pnl"] / initial) * 100
    bet_return = (stats["pnl"] / stats["total_bet"] * 100) if stats["total_bet"] > 0 else 0.0
    bet_intensity = (stats["avg_bet"] / initial * 100) if initial > 0 else 0.0
    return {
        "coin": coin,
        "label": label,
        "type": model_type,
        "initial": initial,
        "pnl": stats["pnl"],
        "roi": roi,
        "bet_return": bet_return,
        "bet_intensity": bet_intensity,
        "total_bet": stats["total_bet"],
        "avg_bet": stats["avg_bet"],
        "wins": stats["wins"],
        "losses": stats["losses"],
        "pending": stats["pending"],
        "completed": stats["completed"],
        "wr": wr,
        "n_trades": stats["n_trades"],
    }


def print_row(i, r, show_coin=False):
    """打印一行排名数据"""
    pend = f"+{r['pending']}待" if r["pending"] else ""
    note = ""
    coin_col = f"{r['coin']:4} " if show_coin else ""
    br_str = f"{r['bet_return']:+7.1f}%" if r["total_bet"] > 0 else "    N/A"
    print(f"  {i:3d}  {coin_col}{r['type']:4} {r['label']:26} {r['initial']:>5}  "
          f"{r['roi']:>+7.2f}%  {br_str}  "
          f"${r['avg_bet']:>6.1f} {r['bet_intensity']:>5.2f}%  "
          f"{r['wins']:2d}/{r['losses']:2d}  {r['wr']:5.1f}%  "
          f"{r['completed']:4d}{pend}  {note}")


def main():
    all_rows = []

    # 1) 旧模型 — $400 初始
    for log_dir, info in OLD_MODELS.items():
        trades = load_trades(log_dir)
        if not trades:
            continue
        coin = info["coin"]
        stats = calc_coin_stats(trades, coin)
        if stats is None or stats["completed"] == 0:
            continue
        all_rows.append(build_row(coin, info["label"], "旧", OLD_INITIAL, stats))

    # 2) Exp 系列 — 按币种隔离 $400/币（只统计 ETH/BTC）
    for exp_name, log_dirs in EXP_SERIES.items():
        exp_lower = exp_name.lower()
        log_prefix = f"logs_v5_{exp_lower}_"
        for log_dir in log_dirs:
            trades = load_trades(log_dir)
            if not trades:
                continue
            short = log_dir.replace(log_prefix, "")
            for coin in EXP_COINS:
                stats = calc_coin_stats(trades, coin, log_dir)
                if stats is None or stats["completed"] == 0:
                    continue
                all_rows.append(build_row(
                    coin, f"{exp_name}_{short}", "Exp", EXP_INITIAL, stats))

    if not all_rows:
        print("未找到任何交易数据")
        return

    # ════════════════════════════════════════════════════════
    #  板块1：按 ROI% 排名（主排名 — 收益率维度）
    # ════════════════════════════════════════════════════════
    print("=" * 120)
    print("  全模型单币三维公平对比排名")
    print("  全部按单币 $400 初始资金计算（旧模型: 固定5%下注 | Exp/Ensemble: Kelly动态下注）")
    print("  ROI = PnL÷$400 | 投注回报 = PnL÷总下注额 | 下注强度 = 平均bet÷$400")
    print("=" * 120)

    hdr = (f"  {'#':>3}  {'类型':4} {'模型/组合':26} {'初始$':>5}  "
           f"{'ROI%':>8}  {'投注回报':>8}  "
           f"{'均bet$':>7} {'强度%':>6}  "
           f"{'胜/负':>5}  {'胜率':>6}  {'事件':>4}  备注")
    sep = "  " + "─" * 112

    for coin in COINS:
        coin_rows = [r for r in all_rows if r["coin"] == coin]
        if not coin_rows:
            continue
        coin_rows.sort(key=lambda x: x["roi"], reverse=True)

        n_old = sum(1 for r in coin_rows if r["type"] == "旧")
        n_exp = sum(1 for r in coin_rows if r["type"] == "Exp")

        print()
        print(f"  ━━━ {coin} ROI排名（{n_old} 旧模型 + {n_exp} Exp/Ensemble，均@$400/币）━━━")
        print()
        print(hdr)
        print(sep)

        top_n = min(10, len(coin_rows))
        print(f"  📈 Top {top_n}:")
        for i, r in enumerate(coin_rows[:top_n], 1):
            print_row(i, r)

        bottom_n = min(10, len(coin_rows))
        if len(coin_rows) > top_n:
            print()
            print(f"  📉 Bottom {bottom_n}:")
            start_idx = max(len(coin_rows) - bottom_n, top_n)
            for i, r in enumerate(coin_rows[start_idx:], start_idx + 1):
                print_row(i, r)

        # 对比小结
        old_rows = [r for r in coin_rows if r["type"] == "旧"]
        exp_rows = [r for r in coin_rows if r["type"] == "Exp"]
        print()
        if old_rows:
            avg_roi = sum(r["roi"] for r in old_rows) / len(old_rows)
            avg_br = sum(r["bet_return"] for r in old_rows) / len(old_rows)
            avg_bi = sum(r["bet_intensity"] for r in old_rows) / len(old_rows)
            print(f"  旧模型: 平均ROI {avg_roi:+.2f}% | 平均投注回报 {avg_br:+.1f}% | 平均下注强度 {avg_bi:.2f}%")
        if exp_rows:
            avg_roi = sum(r["roi"] for r in exp_rows) / len(exp_rows)
            avg_br = sum(r["bet_return"] for r in exp_rows) / len(exp_rows)
            avg_bi = sum(r["bet_intensity"] for r in exp_rows) / len(exp_rows)
            print(f"  Exp/Ensemble: 平均ROI {avg_roi:+.2f}% | 平均投注回报 {avg_br:+.1f}% | 平均下注强度 {avg_bi:.2f}%")

    # ════════════════════════════════════════════════════════
    #  板块2：按投注回报率排名（预测能力维度）
    # ════════════════════════════════════════════════════════
    print()
    print("=" * 120)
    print("  按投注回报率排名（PnL ÷ 总下注额）— 纯粹衡量预测能力，不受资金量影响")
    print("=" * 120)

    hdr2 = (f"  {'#':>3}  {'币':4} {'类型':4} {'模型/组合':26} {'初始$':>5}  "
            f"{'投注回报':>8}  {'ROI%':>8}  "
            f"{'总下注$':>9}  {'PnL$':>10}  "
            f"{'胜率':>6}  {'事件':>4}  备注")
    sep2 = "  " + "─" * 112

    # 只看已结>=5的
    br_rows = [r for r in all_rows if r["completed"] >= 5
               and r["total_bet"] > 0 and r["coin"] in COINS]
    br_rows.sort(key=lambda x: x["bet_return"], reverse=True)

    print()
    print(hdr2)
    print(sep2)

    top_br = min(20, len(br_rows))
    print(f"  📈 投注回报 Top {top_br}（每赌$1赚最多）:")
    for i, r in enumerate(br_rows[:top_br], 1):
        pend = f"+{r['pending']}待" if r["pending"] else ""
        print(f"  {i:3d}  {r['coin']:4} {r['type']:4} {r['label']:26} {r['initial']:>5}  "
              f"{r['bet_return']:>+7.1f}%  {r['roi']:>+7.2f}%  "
              f"${r['total_bet']:>8.1f}  ${r['pnl']:>+9.2f}  "
              f"{r['wr']:5.1f}%  {r['completed']:4d}{pend}")

    # ════════════════════════════════════════════════════════
    #  板块3：各币冠军（三维综合）
    # ════════════════════════════════════════════════════════
    print()
    print("=" * 120)
    print("  各币冠军速览（三维对比）")
    print("=" * 120)
    print()
    for coin in COINS:
        coin_valid = [r for r in all_rows
                      if r["coin"] == coin and r["completed"] >= 5 and r["total_bet"] > 0]
        if not coin_valid:
            continue
        best_roi = max(coin_valid, key=lambda x: x["roi"])
        best_br = max(coin_valid, key=lambda x: x["bet_return"])
        print(f"  {coin} ROI冠军:    {best_roi['label']:26} ({best_roi['type']}) "
              f"ROI {best_roi['roi']:+.2f}% | PnL ${best_roi['pnl']:+.2f} | 投注回报 {best_roi['bet_return']:+.1f}% | 强度 {best_roi['bet_intensity']:.2f}%")
        if best_br["label"] != best_roi["label"]:
            print(f"  {coin} 预测力冠军: {best_br['label']:26} ({best_br['type']}) "
                  f"投注回报 {best_br['bet_return']:+.1f}% | PnL ${best_br['pnl']:+.2f} | ROI {best_br['roi']:+.2f}% | 强度 {best_br['bet_intensity']:.2f}%")
        print()

    # ════════════════════════════════════════════════════════
    #  板块4：实盘候选（ROI>0 + 胜率>=50% + 已结>=5）
    # ════════════════════════════════════════════════════════
    print("=" * 120)
    print("  实盘候选（ROI>0 且 胜率>=50% 且 已结>=5 事件）")
    print("=" * 120)
    print()
    candidates = [r for r in all_rows
                  if r["roi"] > 0 and r["wr"] >= 50.0 and r["completed"] >= 5
                  and r["coin"] in COINS]
    candidates.sort(key=lambda x: x["roi"], reverse=True)

    if candidates:
        hdr3 = (f"  {'#':>3}  {'币':4} {'类型':4} {'模型/组合':26} {'初始$':>5}  "
                f"{'ROI%':>8}  {'投注回报':>8}  "
                f"{'均bet$':>7} {'强度%':>6}  "
                f"{'胜率':>6}  {'事件':>4}  备注")
        print(hdr3)
        print("  " + "─" * 112)
        for i, r in enumerate(candidates, 1):
            print_row(i, r, show_coin=True)
        print()
        print(f"  共 {len(candidates)} 个候选")
    else:
        print("  暂无符合条件的候选")

    # ════════════════════════════════════════════════════════
    #  板块5：实际模拟资金收益（$400/币）
    # ════════════════════════════════════════════════════════
    print()
    print("=" * 120)
    print("  实际模拟收益 Top 5（初始 $400/币，按实际 PnL 排名）")
    print("=" * 120)
    print()
    for coin in COINS:
        coin_cands = [r for r in candidates if r["coin"] == coin]
        coin_cands.sort(key=lambda x: x["pnl"], reverse=True)
        coin_cands = coin_cands[:5]
        if not coin_cands:
            print(f"  {coin}: 无正收益候选")
            continue
        for i, r in enumerate(coin_cands, 1):
            final_cap = r["initial"] + r["pnl"]
            print(f"  {coin} #{i} {r['type']:4} {r['label']:26} "
                  f"$400 → ${final_cap:.2f} (PnL ${r['pnl']:+.2f}, ROI {r['roi']:+.2f}%) "
                  f"| 胜率 {r['wr']:.1f}% | 投注回报 {r['bet_return']:+.1f}%")
        print()

    # ════════════════════════════════════════════════════════
    #  板块6：SOL 单独
    # ════════════════════════════════════════════════════════
    sol_rows = [r for r in all_rows if r["coin"] == "SOL"]
    if sol_rows:
        sol_rows.sort(key=lambda x: x["roi"], reverse=True)
        print(f"  ━━━ SOL（仅旧模型@$400）━━━")
        for i, r in enumerate(sol_rows, 1):
            br_str = f"{r['bet_return']:+7.1f}%" if r["total_bet"] > 0 else "    N/A"
            print(f"  {i:3d}  旧  {r['label']:26} {r['initial']:>5}  "
                  f"{r['roi']:>+7.2f}%  {br_str}  "
                  f"${r['avg_bet']:>6.1f} {r['bet_intensity']:>5.2f}%  "
                  f"{r['wins']:2d}/{r['losses']:2d}  {r['wr']:5.1f}%  {r['completed']:4d}")
        print()

    print("=" * 120)
    print("  完成")
    print("=" * 120)


if __name__ == "__main__":
    main()
