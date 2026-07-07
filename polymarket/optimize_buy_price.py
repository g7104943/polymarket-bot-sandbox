#!/usr/bin/env python3
"""
最优限价买入价格分析器 — 数据驱动版

利用 Exp 系列多价位实盘数据，拟合「成交率-胜率-期望值」曲线，
为旧模型（尤其 GRU_ETH_55_dyn_超参）推荐最优限价买入价格 p*。

═══════════════════════════════════════════════════
理论框架：
  每个预测信号的期望 PnL = fill_rate(p) × bet(p) × edge(p)

  其中:
    fill_rate(p) = 在买价 p 时的成交概率（价格越高越容易成交）
    edge(p)      = WR(p) × (1-p)/p - (1-WR(p))   -- 每 $1 下注的期望利润
    (1-p)/p      = 赢时回报倍数（p=0.50 → 1.0×, p=0.45 → 1.22×）

  最优价格 p* 使得 E[PnL 每小时] 最大化
═══════════════════════════════════════════════════

数据来源：
  1. Exp 系列（exp8/13/15/16）9 个固定买价 + 2 个动态价位的实盘交易
  2. 22 个旧模型在 0.51 固定限价下的实盘交易

分析维度：
  A. 经验 PnL 曲线  — 各价位直接 PnL 对比
  B. 成交量曲线      — 各价位交易笔数/小时
  C. 胜率曲线        — 各价位条件胜率（检测逆向选择）
  D. 期望值曲线      — 综合 A+B+C 的理论最优
  E. 旧模型适配      — 将旧模型胜率代入 Exp 成交率曲线

用法：
  python3 polymarket/optimize_buy_price.py
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
COINS = ["ETH", "BTC", "XRP"]
BPS = ["0450", "0460", "0470", "0480", "0490", "0500", "0510", "0520", "0530"]
BP_FLOAT = [0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53]

EXP_NAMES = ["exp8", "exp13", "exp15", "exp16"]
OLD_INITIAL = 400
EXP_INITIAL = 1200

# ────────── GRU EK 组合（活跃，limitPrice=0.46）──────────
OLD_MODELS = {
    "logs_gru_eth_ek":              {"coin": "ETH", "label": "GRU_ETH_EK"},
    "logs_gru_eth_no1h4h_ek":      {"coin": "ETH", "label": "GRU_ETH_no1h4h_EK"},
    "logs_gru_btc_no1h4h_ek":      {"coin": "BTC", "label": "GRU_BTC_no1h4h_EK"},
    "logs_gru_sol_ek":              {"coin": "SOL", "label": "GRU_SOL_EK"},
}


# ═══════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════

def load_trades_extended(log_dir: str) -> list:
    """加载交易记录（含 tokenPrice, confidence, amount, timestamp）"""
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
        try:
            token_price = float(t.get("tokenPrice", 0))
        except (TypeError, ValueError):
            token_price = 0.0
        try:
            confidence = float(t.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0
        ts_str = t.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except Exception:
            ts = None
        trades.append({
            "symbol": symbol, "result": result, "pnl": pnl,
            "amount": amount, "tokenPrice": token_price,
            "confidence": confidence, "timestamp": ts,
            "slug": t.get("marketSlug", ""),
            "limitPrice": float(t.get("limitPriceConfigured", 0) or 0),
        })
    return trades


def compute_stats(trades: list, coin: str) -> dict:
    """计算单币统计：胜率/PnL/交易量/运行时长等"""
    ct = [t for t in trades if t["symbol"] == coin]
    if not ct:
        return None
    wins = sum(1 for t in ct if t["result"] == "win")
    losses = sum(1 for t in ct if t["result"] == "lose")
    settled = wins + losses
    wr = wins / settled if settled else 0
    total_pnl = sum(t["pnl"] for t in ct if t["pnl"] is not None)
    pnl_per_trade = total_pnl / len(ct) if ct else 0
    amounts = [t["amount"] for t in ct if t["amount"] > 0]
    avg_amount = sum(amounts) / len(amounts) if amounts else 0
    prices = [t["tokenPrice"] for t in ct if t["tokenPrice"] > 0]
    avg_price = sum(prices) / len(prices) if prices else 0

    # 运行时长（小时）
    timestamps = [t["timestamp"] for t in ct if t["timestamp"]]
    if len(timestamps) >= 2:
        hours = (max(timestamps) - min(timestamps)).total_seconds() / 3600
    else:
        hours = 1.0
    hours = max(hours, 0.5)

    trades_per_hour = len(ct) / hours
    pnl_per_hour = total_pnl / hours

    return {
        "n_trades": len(ct), "wins": wins, "losses": losses,
        "settled": settled, "wr": wr, "total_pnl": total_pnl,
        "pnl_per_trade": pnl_per_trade, "avg_amount": avg_amount,
        "avg_price": avg_price, "hours": hours,
        "trades_per_hour": trades_per_hour, "pnl_per_hour": pnl_per_hour,
    }


# ═══════════════════════════════════════════════
#  理论计算
# ═══════════════════════════════════════════════

def edge_per_dollar(wr: float, p: float) -> float:
    """每 $1 下注的期望利润: WR × (1-p)/p - (1-WR)"""
    if p <= 0 or p >= 1:
        return 0
    return wr * (1 - p) / p - (1 - wr)


def expected_pnl_per_signal(fill_rate: float, avg_bet: float,
                            wr: float, p: float) -> float:
    """每个预测信号的期望 PnL"""
    edge = edge_per_dollar(wr, p)
    return fill_rate * avg_bet * edge


# ═══════════════════════════════════════════════
#  主分析逻辑
# ═══════════════════════════════════════════════

def main():
    # ── 1. 采集 Exp 多价位数据 ──
    # exp_data[exp][bp][coin] = stats
    exp_data = {}
    for exp in EXP_NAMES:
        exp_data[exp] = {}
        for bp in BPS:
            log_dir = f"logs_v5_{exp}_bp{bp}"
            trades = load_trades_extended(log_dir)
            if not trades:
                continue
            exp_data[exp][bp] = {}
            for coin in COINS:
                stats = compute_stats(trades, coin)
                if stats:
                    exp_data[exp][bp][coin] = stats
        # 动态价位
        for dyn in ["dyn_0450_0530", "dyn_0480_0510"]:
            log_dir = f"logs_v5_{exp}_bp_{dyn}"
            trades = load_trades_extended(log_dir)
            if not trades:
                continue
            exp_data[exp][dyn] = {}
            for coin in COINS:
                stats = compute_stats(trades, coin)
                if stats:
                    exp_data[exp][dyn][coin] = stats

    # ── 2. 采集旧模型数据 ──
    old_data = {}
    for log_dir, info in OLD_MODELS.items():
        trades = load_trades_extended(log_dir)
        if not trades:
            continue
        coin = info["coin"]
        stats = compute_stats(trades, coin)
        if stats and stats["settled"] > 0:
            old_data[info["label"]] = {"coin": coin, "stats": stats}

    # ═══════════════════════════════════════════════════
    #  板块 A：Exp 各价位经验 PnL 曲线
    # ═══════════════════════════════════════════════════
    print("=" * 130)
    print("  最优限价买入价格分析器")
    print("  理论: E[PnL/信号] = 成交率(p) × 下注额(p) × [WR(p) × (1-p)/p - (1-WR(p))]")
    print("  数据: Exp8/13/15/16 各 9 个固定买价实盘交易 + 22 个旧模型@0.51")
    print("=" * 130)

    for coin in COINS:
        print(f"\n{'━' * 130}")
        print(f"  {coin} — 各买价经验数据（所有 Exp 模型合并）")
        print(f"{'━' * 130}")
        print(f"  {'买价':>4}  {'笔数':>5}  {'笔/hr':>6}  {'胜率%':>6}  {'总PnL$':>9}  "
              f"{'PnL/笔':>7}  {'PnL/hr':>8}  {'均token价':>8}  {'均bet$':>7}  "
              f"{'Edge/$ ':>7}  {'理论E[PnL]':>10}")
        print(f"  {'─' * 122}")

        # 合并所有 exp 在同一 bp 的数据
        merged = {}
        for bp_idx, bp in enumerate(BPS):
            p = BP_FLOAT[bp_idx]
            all_n = 0
            all_wins = 0
            all_losses = 0
            all_pnl = 0.0
            all_amounts = []
            all_prices = []
            total_hours = 0.0

            for exp in EXP_NAMES:
                if bp in exp_data.get(exp, {}) and coin in exp_data[exp][bp]:
                    s = exp_data[exp][bp][coin]
                    all_n += s["n_trades"]
                    all_wins += s["wins"]
                    all_losses += s["losses"]
                    all_pnl += s["total_pnl"]
                    total_hours += s["hours"]
                    if s["avg_amount"] > 0:
                        all_amounts.extend([s["avg_amount"]] * s["n_trades"])
                    if s["avg_price"] > 0:
                        all_prices.extend([s["avg_price"]] * s["n_trades"])

            settled = all_wins + all_losses
            wr = all_wins / settled if settled else 0
            avg_amt = sum(all_amounts) / len(all_amounts) if all_amounts else 0
            avg_prc = sum(all_prices) / len(all_prices) if all_prices else 0
            pnl_per_trade = all_pnl / all_n if all_n else 0
            trades_per_hr = all_n / total_hours if total_hours > 0 else 0
            pnl_per_hr = all_pnl / total_hours if total_hours > 0 else 0
            edge = edge_per_dollar(wr, p)
            # 成交率估算：用 trades_per_hour 的相对值
            e_pnl = trades_per_hr * pnl_per_trade  # = pnl_per_hour

            merged[bp] = {
                "p": p, "n": all_n, "wins": all_wins, "losses": all_losses,
                "settled": settled, "wr": wr, "pnl": all_pnl,
                "pnl_per_trade": pnl_per_trade, "avg_amount": avg_amt,
                "avg_price": avg_prc, "trades_per_hr": trades_per_hr,
                "pnl_per_hr": pnl_per_hr, "edge": edge,
                "hours": total_hours,
            }

            marker = ""
            if pnl_per_hr > 0:
                marker = " ✓"
            print(f"  {p:.2f}  {all_n:>5}  {trades_per_hr:>6.1f}  {wr*100:>5.1f}%  "
                  f"${all_pnl:>+8.1f}  {pnl_per_trade:>+7.2f}  "
                  f"${pnl_per_hr:>+7.2f}  {avg_prc:>8.4f}  ${avg_amt:>6.1f}  "
                  f"{edge:>+6.3f}  ${pnl_per_hr:>+9.2f}{marker}")

        # 找最优买价
        best_bp = max(merged.values(), key=lambda x: x["pnl_per_hr"])
        worst_bp = min(merged.values(), key=lambda x: x["pnl_per_hr"])
        print()
        print(f"  >>> {coin} 最优买价(合并): {best_bp['p']:.2f} "
              f"(PnL/hr: ${best_bp['pnl_per_hr']:+.2f}, 胜率: {best_bp['wr']*100:.1f}%, "
              f"笔/hr: {best_bp['trades_per_hr']:.1f})")
        print(f"  >>> {coin} 最差买价(合并): {worst_bp['p']:.2f} "
              f"(PnL/hr: ${worst_bp['pnl_per_hr']:+.2f})")

        # ── 按单个 Exp 分别展示最优 ──
        print()
        print(f"  {coin} — 各 Exp 各自最优买价:")
        for exp in EXP_NAMES:
            exp_bp_stats = []
            for bp in BPS:
                if bp in exp_data.get(exp, {}) and coin in exp_data[exp][bp]:
                    s = exp_data[exp][bp][coin]
                    exp_bp_stats.append((bp, BP_FLOAT[BPS.index(bp)], s))
            if not exp_bp_stats:
                continue
            best = max(exp_bp_stats, key=lambda x: x[2]["pnl_per_hour"])
            print(f"    {exp.upper():>5}: 最优={best[1]:.2f} "
                  f"(PnL/hr=${best[2]['pnl_per_hour']:+.2f}, "
                  f"WR={best[2]['wr']*100:.1f}%, "
                  f"N={best[2]['n_trades']}, {best[2]['hours']:.1f}h)")

    # ═══════════════════════════════════════════════════
    #  板块 B：逆向选择检测
    # ═══════════════════════════════════════════════════
    print()
    print("=" * 130)
    print("  逆向选择检测 — 低价成交的是不是「坏单」？")
    print("  如果低价胜率 < 高价胜率 → 存在逆向选择（市场在你便宜买到时更可能对你不利）")
    print("=" * 130)

    for coin in COINS:
        print(f"\n  {coin}:")
        print(f"    {'买价':>4}  ", end="")
        for exp in EXP_NAMES:
            print(f"  {exp.upper():>6}_WR", end="")
        print(f"  {'合并WR':>7}  判定")
        print(f"    {'─' * 70}")

        all_bp_wr = {}
        for bp_idx, bp in enumerate(BPS):
            p = BP_FLOAT[bp_idx]
            print(f"    {p:.2f}  ", end="")
            wr_list = []
            for exp in EXP_NAMES:
                if bp in exp_data.get(exp, {}) and coin in exp_data[exp][bp]:
                    s = exp_data[exp][bp][coin]
                    wr_list.append(s["wr"])
                    print(f"  {s['wr']*100:>7.1f}%", end="")
                else:
                    print(f"  {'N/A':>8}", end="")
            avg_wr = sum(wr_list) / len(wr_list) if wr_list else 0
            all_bp_wr[bp] = avg_wr
            print(f"  {avg_wr*100:>6.1f}%", end="")
            print()

        # 检测趋势
        wrs = [(BP_FLOAT[i], all_bp_wr.get(bp, 0)) for i, bp in enumerate(BPS)
               if all_bp_wr.get(bp, 0) > 0]
        if len(wrs) >= 3:
            low_wr = sum(w for _, w in wrs[:3]) / 3
            high_wr = sum(w for _, w in wrs[-3:]) / 3
            if high_wr > low_wr * 1.05:
                print(f"    ⚠️  {coin} 存在轻度逆向选择: 低价平均WR={low_wr*100:.1f}% "
                      f"< 高价平均WR={high_wr*100:.1f}%")
                print(f"    → 低价虽然利润率更高，但胜率更低，部分利润会被蚕食")
            elif low_wr > high_wr * 1.05:
                print(f"    ✓  {coin} 无逆向选择: 低价WR={low_wr*100:.1f}% "
                      f"≥ 高价WR={high_wr*100:.1f}% → 低价策略安全")
            else:
                print(f"    ≈  {coin} 胜率无显著趋势: 低价WR={low_wr*100:.1f}% "
                      f"≈ 高价WR={high_wr*100:.1f}%")

    # ═══════════════════════════════════════════════════
    #  板块 C：旧模型当前表现 + 理论最优估算
    # ═══════════════════════════════════════════════════
    print()
    print("=" * 130)
    print("  旧模型（当前 LIMIT_PRICE=0.51）表现 + 最优买价估算")
    print("  方法: 用旧模型的实际胜率 × Exp 的成交率曲线 → 推算在不同买价下的期望 PnL")
    print("=" * 130)

    print(f"\n  {'模型':26}  {'币种':>4}  {'当前WR%':>7}  {'笔数':>4}  {'PnL$':>8}  "
          f"{'当前均价':>7}  {'当前Edge':>8}  {'推荐买价':>8}  {'预期改善':>8}")
    print(f"  {'─' * 115}")

    # 对每个旧模型，用其胜率代入 Exp 成交率曲线计算理论最优价格
    for label, info in sorted(old_data.items(), key=lambda x: -x[1]["stats"]["total_pnl"]):
        coin = info["coin"]
        s = info["stats"]
        wr = s["wr"]
        current_edge = edge_per_dollar(wr, 0.51)

        # 用 Exp 合并数据的 trades_per_hour 作为成交率代理
        # 归一化为相对值（最大 trades_per_hour = 1.0）
        coin_tph = {}
        for bp_idx, bp in enumerate(BPS):
            p = BP_FLOAT[bp_idx]
            tph_sum = 0
            n_exp = 0
            for exp in EXP_NAMES:
                if bp in exp_data.get(exp, {}) and coin in exp_data[exp][bp]:
                    tph_sum += exp_data[exp][bp][coin]["trades_per_hour"]
                    n_exp += 1
            if n_exp > 0:
                coin_tph[p] = tph_sum / n_exp
        if not coin_tph:
            continue

        max_tph = max(coin_tph.values())
        if max_tph <= 0:
            continue

        # 用旧模型实际平均下注额
        avg_bet = s["avg_amount"] if s["avg_amount"] > 0 else 12.0

        # 计算每个价位的理论 E[PnL/hr]
        best_p = 0.51
        best_epnl = 0
        theory_results = []
        for p in [0.44, 0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54]:
            # 插值成交率
            if p in coin_tph:
                rel_fill = coin_tph[p] / max_tph
            else:
                # 线性插值
                ps = sorted(coin_tph.keys())
                if p <= ps[0]:
                    rel_fill = coin_tph[ps[0]] / max_tph * 0.8
                elif p >= ps[-1]:
                    rel_fill = coin_tph[ps[-1]] / max_tph * 1.1
                else:
                    for i in range(len(ps) - 1):
                        if ps[i] <= p <= ps[i + 1]:
                            w = (p - ps[i]) / (ps[i + 1] - ps[i])
                            rel_fill = ((1 - w) * coin_tph[ps[i]]
                                        + w * coin_tph[ps[i + 1]]) / max_tph
                            break
            edge = edge_per_dollar(wr, p)
            epnl = rel_fill * avg_bet * edge * max_tph  # PnL per hour
            theory_results.append((p, rel_fill, edge, epnl))
            if epnl > best_epnl:
                best_epnl = epnl
                best_p = p

        current_epnl = 0
        for p, rf, ed, ep in theory_results:
            if abs(p - 0.51) < 0.005:
                current_epnl = ep
                break

        improvement = ((best_epnl - current_epnl) / abs(current_epnl) * 100
                       if current_epnl != 0 else 0)
        imp_str = f"{improvement:+.0f}%" if current_epnl != 0 else "N/A"
        marker = " ★" if improvement > 10 else ""

        print(f"  {label:26}  {coin:>4}  {wr*100:>6.1f}%  {s['settled']:>4}  "
              f"${s['total_pnl']:>+7.1f}  {s['avg_price']:>7.4f}  "
              f"{current_edge:>+7.3f}  {best_p:>8.2f}  {imp_str:>8}{marker}")

    # ═══════════════════════════════════════════════════
    #  板块 D：重点模型详细分析
    # ═══════════════════════════════════════════════════
    focus_models = [
        ("GRU_ETH_55_dyn_超参", "ETH"),
        ("GRU_ETH_54_超参", "ETH"),
        ("GRU_BTC_57_no1h4h_超参", "BTC"),
        ("GRU_XRP_53_no1h4h_超参", "XRP"),
    ]

    print()
    print("=" * 130)
    print("  重点模型各买价理论 E[PnL/hr] 详细曲线")
    print("  公式: E[PnL/hr](p) = 相对成交率(p) × 平均下注 × [WR×(1-p)/p-(1-WR)] × 基准笔/hr")
    print("=" * 130)

    for model_label, coin in focus_models:
        if model_label not in old_data:
            continue
        s = old_data[model_label]["stats"]
        wr = s["wr"]
        avg_bet = s["avg_amount"] if s["avg_amount"] > 0 else 12.0

        # 获取该币种的 Exp 成交率曲线
        coin_tph = {}
        for bp_idx, bp in enumerate(BPS):
            p = BP_FLOAT[bp_idx]
            tph_sum = 0
            n_exp = 0
            for exp in EXP_NAMES:
                if bp in exp_data.get(exp, {}) and coin in exp_data[exp][bp]:
                    tph_sum += exp_data[exp][bp][coin]["trades_per_hour"]
                    n_exp += 1
            if n_exp > 0:
                coin_tph[p] = tph_sum / n_exp
        if not coin_tph:
            continue
        max_tph = max(coin_tph.values()) if coin_tph else 1

        print(f"\n  ┌─ {model_label} ({coin}) — 胜率={wr*100:.1f}%, 均bet=${avg_bet:.1f}")
        print(f"  │")
        print(f"  │  {'买价':>5}  {'成交率':>6}  {'Edge/$':>7}  {'E[PnL/hr]':>10}  {'图示'}")
        print(f"  │  {'─' * 55}")

        best_p = 0.51
        best_val = -9999
        results = []
        for p_val in [round(0.44 + i * 0.01, 2) for i in range(12)]:
            if p_val in coin_tph:
                rel_fill = coin_tph[p_val] / max_tph
            else:
                ps = sorted(coin_tph.keys())
                if p_val <= ps[0]:
                    rel_fill = coin_tph[ps[0]] / max_tph * 0.7
                elif p_val >= ps[-1]:
                    rel_fill = min(1.0, coin_tph[ps[-1]] / max_tph * 1.15)
                else:
                    for i in range(len(ps) - 1):
                        if ps[i] <= p_val <= ps[i + 1]:
                            w = (p_val - ps[i]) / (ps[i + 1] - ps[i])
                            rel_fill = ((1 - w) * coin_tph[ps[i]]
                                        + w * coin_tph[ps[i + 1]]) / max_tph
                            break

            edge = edge_per_dollar(wr, p_val)
            epnl = rel_fill * avg_bet * edge * max_tph
            results.append((p_val, rel_fill, edge, epnl))
            if epnl > best_val:
                best_val = epnl
                best_p = p_val

        # 归一化柱状图
        max_abs = max(abs(r[3]) for r in results) if results else 1
        for p_val, rf, edge, epnl in results:
            bar_len = int(abs(epnl) / max_abs * 30) if max_abs > 0 else 0
            bar = ("█" * bar_len if epnl >= 0 else "░" * bar_len)
            marker = " ◄── 最优" if abs(p_val - best_p) < 0.005 else ""
            marker += " (当前)" if abs(p_val - 0.51) < 0.005 else ""
            sign = "+" if epnl >= 0 else ""
            print(f"  │  {p_val:.2f}  {rf*100:>5.1f}%  {edge:>+6.3f}  "
                  f"${sign}{epnl:>8.2f}  {bar}{marker}")

        current_epnl = next((e for p, _, _, e in results if abs(p - 0.51) < 0.005), 0)
        if current_epnl != 0 and best_val != 0:
            imp = (best_val - current_epnl) / abs(current_epnl) * 100
            print(f"  │")
            print(f"  │  推荐: {best_p:.2f} → 理论改善 {imp:+.0f}% vs 当前 0.51")
            if best_p < 0.51:
                print(f"  │  ⚠  注意: 价格降低会减少成交量，但每笔利润更高")
            elif best_p > 0.51:
                print(f"  │  ✓  价格提高能增加成交量，虽然每笔利润略低")
        print(f"  └{'─' * 60}")

    # ═══════════════════════════════════════════════════
    #  板块 E：综合建议
    # ═══════════════════════════════════════════════════
    print()
    print("=" * 130)
    print("  综合建议")
    print("=" * 130)
    print()

    # 汇总所有旧模型的推荐
    recommendations = {}
    for label, info in old_data.items():
        coin = info["coin"]
        s = info["stats"]
        wr = s["wr"]
        avg_bet = s["avg_amount"] if s["avg_amount"] > 0 else 12.0

        coin_tph = {}
        for bp_idx, bp in enumerate(BPS):
            p = BP_FLOAT[bp_idx]
            tph_sum = 0
            n_exp = 0
            for exp in EXP_NAMES:
                if bp in exp_data.get(exp, {}) and coin in exp_data[exp][bp]:
                    tph_sum += exp_data[exp][bp][coin]["trades_per_hour"]
                    n_exp += 1
            if n_exp > 0:
                coin_tph[p] = tph_sum / n_exp
        if not coin_tph:
            continue
        max_tph = max(coin_tph.values())
        if max_tph <= 0:
            continue

        best_p = 0.51
        best_val = -9999
        for p_val in [round(0.44 + i * 0.01, 2) for i in range(12)]:
            if p_val in coin_tph:
                rf = coin_tph[p_val] / max_tph
            else:
                ps = sorted(coin_tph.keys())
                if p_val <= ps[0]:
                    rf = coin_tph[ps[0]] / max_tph * 0.7
                elif p_val >= ps[-1]:
                    rf = min(1.0, coin_tph[ps[-1]] / max_tph * 1.15)
                else:
                    for i in range(len(ps) - 1):
                        if ps[i] <= p_val <= ps[i + 1]:
                            w = (p_val - ps[i]) / (ps[i + 1] - ps[i])
                            rf = ((1 - w) * coin_tph[ps[i]]
                                  + w * coin_tph[ps[i + 1]]) / max_tph
                            break
            edge = edge_per_dollar(wr, p_val)
            epnl = rf * avg_bet * edge * max_tph
            if epnl > best_val:
                best_val = epnl
                best_p = p_val

        if coin not in recommendations:
            recommendations[coin] = []
        recommendations[coin].append((label, wr, s["total_pnl"], best_p, s["settled"]))

    for coin in COINS:
        if coin not in recommendations:
            continue
        recs = recommendations[coin]
        # 按 PnL 排序取 top
        recs.sort(key=lambda x: -x[2])
        rec_prices = [r[3] for r in recs if r[4] >= 5]
        if rec_prices:
            avg_rec = sum(rec_prices) / len(rec_prices)
            med_rec = sorted(rec_prices)[len(rec_prices) // 2]
            print(f"  {coin}:")
            print(f"    所有旧模型推荐买价: 平均={avg_rec:.2f}, 中位={med_rec:.2f}")
            for label, wr, pnl, bp, n in recs[:5]:
                marker = " ★" if n >= 10 else ""
                print(f"      {label:26} WR={wr*100:.1f}% PnL=${pnl:+.1f} "
                      f"→ 推荐 {bp:.2f} (基于{n}笔){marker}")
            print()

    print()
    print("  ─── 行动方案 ───")
    print()
    print("  方案 A（保守）: 不改价格，维持 0.51")
    print("    适用于: 样本量太少(<20笔)、推荐价与0.51差异不大的模型")
    print()
    print("  方案 B（推荐）: 将旧模型的买价加入超参优化")
    print("    做法: 修改 optimize_trading_rules.py，增加 buy_price 为可优化参数")
    print("    范围: [0.45, 0.54]，步长 0.01")
    print("    优点: 数据驱动、自动适配每个模型")
    print()
    print("  方案 C（快速验证）: 为 Top 旧模型部署 2-3 个买价")
    print("    做法: 类似 Exp 的多价位部署，让实盘数据说话")
    print("    示例: GRU_ETH_55_dyn_超参 → 分别部署 @0.49 和 @0.52")
    print()
    print("  ⚠ 重要提醒:")
    print("  1. Exp 数据仅 2-3 天，样本量有限，结论需要更多数据验证")
    print("  2. 旧模型和 Exp 模型的信号时机不同，成交率曲线可能有差异")
    print("  3. 市场微观结构随时间变化，最优价格不是固定的")
    print("  4. 对于盈利模型，避免大幅改动；对于亏损模型，可以大胆尝试")
    print()
    print("=" * 130)
    print("  分析完成")
    print("=" * 130)


if __name__ == "__main__":
    main()
