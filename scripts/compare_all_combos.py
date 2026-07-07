#!/usr/bin/env python3
"""
统一回测对比脚本 — 22 个模拟交易组合 + v5 新模型（BTC/ETH/XRP）

功能:
  1. 从 polymarket/logs_*/reports/report_summary.json 读取现有 22 个组合的实战模拟结果
  2. 用 v5 模型预测（v5_test_predictions.parquet）+ 最优交易规则回测 BTC/ETH/XRP
  3. 按最终资金排名，输出对比表格

用法:
  python scripts/compare_all_combos.py

注意:
  - 旧 22 个组合的规则和结果不做任何改变，直接取其模拟盘数据
  - v5 新模型使用 optimal_trading_rules_v3.json 中的最优参数
  - 时间段: 旧组合为实际运行时段，v5 为测试集 15 天（2026-01-24 ~ 2026-02-08）
"""
from __future__ import annotations

import json
import sys
import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

POLYMARKET_DIR = PROJECT_ROOT / "polymarket"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "sentiment_grid_search" / "results"
V5_PRED_FILE = RESULTS_DIR / "v5_test_predictions.parquet"
OPTIMAL_RULES_FILE = RESULTS_DIR / "optimal_trading_rules_v3.json"

# ─── Polymarket 真实约束 ──────────────────────────────────
BUY_PRICE = 0.527       # 默认值，可通过 --buy-price 覆盖
LIQUIDITY_CAP = 60_000.0
LIQUIDITY_BET = 3_000.0
DEFAULT_FEE_RATE = 0.002
DEFAULT_SLIPPAGE_RATE = 0.003


# ═══════════════════════════════════════════════════════════
# Part 1: 读取 22 个现有组合的模拟盘结果
# ═══════════════════════════════════════════════════════════

def load_existing_combos() -> List[Dict[str, Any]]:
    """从 logs_*/reports/report_summary.json 加载所有现有组合的结果。"""
    combos = []
    pattern = str(POLYMARKET_DIR / "logs_*" / "reports" / "report_summary.json")

    for fpath in sorted(glob.glob(pattern)):
        logs_dir = Path(fpath).parents[1].name  # logs_xxx
        combo_name = logs_dir.replace("logs_", "")

        try:
            with open(fpath) as f:
                data = json.load(f)
        except Exception:
            continue

        summary = data.get("summary", {})
        config = data.get("config", {})

        total_trades = summary.get("totalTrades", 0)
        wins = summary.get("wins", 0)
        losses = summary.get("losses", 0)
        win_rate = summary.get("winRate", 0)
        pnl = summary.get("totalPnL", 0)
        current_capital = summary.get("currentCapital", 0)
        initial_capital = summary.get("initialCapital", 400)

        # 从 trade log 计算最大回撤
        max_dd = _calc_max_drawdown_from_trades(POLYMARKET_DIR / logs_dir / "prediction_trades.json", initial_capital)

        # 判断下注方式
        bet_method = _get_bet_method(config, combo_name)

        # 判断组合 ID
        combo_id = _get_combo_id(combo_name, config)

        # 找到币种
        markets = config.get("allowedMarkets", config.get("ALLOWED_MARKETS", ""))
        if isinstance(markets, list):
            markets = ",".join(markets)

        combos.append({
            "排名": 0,
            "组合": combo_name,
            "组合ID": combo_id,
            "币种": markets if markets else "?",
            "下注方式": bet_method,
            "最终资金": round(current_capital, 2),
            "初始资金": round(initial_capital, 2),
            "胜率%": round(win_rate, 1),
            "盈亏%": round(pnl / initial_capital * 100, 1) if initial_capital > 0 else 0,
            "最大回撤%": round(max_dd * 100, 1),
            "交易次数": total_trades,
            "最低/初始%": round(current_capital / initial_capital * 100, 1) if initial_capital > 0 else 0,
            "来源": "模拟盘",
        })

    return combos


def _calc_max_drawdown_from_trades(trades_file: Path, initial_capital: float) -> float:
    """从交易日志计算最大回撤比例。"""
    if not trades_file.exists():
        return 0.0
    try:
        with open(trades_file) as f:
            trades = json.load(f)
    except Exception:
        return 0.0

    if not trades:
        return 0.0

    # 按时间排序
    trades = sorted(trades, key=lambda t: t.get("timestamp", ""))

    capital = initial_capital
    peak = capital
    max_dd = 0.0

    for t in trades:
        pnl = t.get("pnl", 0)
        if pnl is None:
            continue
        if t.get("result") in ("win", "lose"):
            capital += pnl
            peak = max(peak, capital)
            if peak > 0:
                dd = (peak - capital) / peak
                max_dd = max(max_dd, dd)

    return max_dd


def _get_bet_method(config: dict, combo_name: str) -> str:
    """判断下注方式描述。"""
    if "超参" in combo_name:
        # 超参组合使用档位下注
        tiers = []
        for k in ["TIER_P1_PCT", "TIER_P2_PCT", "TIER_P3_PCT", "TIER_P4_PCT"]:
            v = config.get(k)
            if v is not None:
                tiers.append(f"{v}%")
        if tiers:
            return f"档位[{','.join(tiers)}]"
        return "档位下注"
    bet_pct = config.get("BET_SIZE_PERCENT", config.get("betSizePercent", 5))
    return f"固定{bet_pct}%"


def _get_combo_id(combo_name: str, config: dict) -> str:
    """生成简短的组合 ID。"""
    name = combo_name.upper().replace("_超参", "_HP").replace("_DYN", "_DYN")
    return name[:20]


# ═══════════════════════════════════════════════════════════
# Part 2: v5 新模型回测
# ═══════════════════════════════════════════════════════════

def simulate_trading(
    predictions: pd.DataFrame,
    params: Dict[str, Any],
    initial_capital: float = 400.0,
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage_rate: float = DEFAULT_SLIPPAGE_RATE,
    buy_price: float = None,
) -> Dict[str, float]:
    """回测模拟交易（与 optimize_trading_rules.py 一致）。"""
    bp = buy_price if buy_price is not None else BUY_PRICE
    odds = (1.0 - bp) / bp
    total_cost = fee_rate + slippage_rate

    min_conf = params.get("min_confidence", 0.50)
    min_edge = params.get("min_edge", 0.02)
    kelly_frac = params.get("kelly_frac", 0.33)
    bet_pct_normal = params.get("bet_pct_normal", 0.05)
    bet_pct_conservative = params.get("bet_pct_conservative", 0.03)
    conf_tier1 = params.get("conf_tier1_bound", 0.55)
    conf_tier2 = params.get("conf_tier2_bound", 0.60)
    tier1_mult = params.get("tier1_mult", 0.30)
    tier2_mult = params.get("tier2_mult", 0.60)
    tier3_mult = params.get("tier3_mult", 1.00)
    cooldown_bars = int(params.get("cooldown_bars", 0))
    drawdown_halt = params.get("drawdown_halt", 0.20)

    capital = initial_capital
    peak = capital
    min_capital = capital
    n_trades = 0
    wins = 0
    losses = 0
    total_profit = 0.0
    total_loss = 0.0
    cooldown_remaining = 0
    ever_reached_cap = False

    for _, row in predictions.iterrows():
        proba = row["proba_up"]
        actual = int(row["actual"])

        # 冷却期
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue

        # 回撤熔断
        if peak > 0 and (peak - capital) / peak >= drawdown_halt:
            continue

        direction = 1 if proba >= 0.5 else 0
        p = proba if direction == 1 else (1 - proba)

        # Layer 0: 最低置信度
        if p < min_conf:
            continue

        # Layer 1: Edge
        net_p = p - total_cost / 2
        edge = net_p * odds - (1 - net_p)
        if edge < min_edge:
            continue

        # Layer 2: Kelly
        q = 1 - net_p
        kelly_f = (net_p * odds - q) / odds if odds > 0 else 0
        kelly_f = max(0, kelly_f)
        bet_ratio = kelly_f * kelly_frac

        # 档位调整
        if p < conf_tier1:
            bet_ratio *= tier1_mult
        elif p < conf_tier2:
            bet_ratio *= tier2_mult
        else:
            bet_ratio *= tier3_mult

        # 分阶段仓位
        if capital >= LIQUIDITY_CAP:
            ever_reached_cap = True
            bet_amount = LIQUIDITY_BET
        elif ever_reached_cap:
            bet_ratio = min(bet_ratio, bet_pct_conservative)
            bet_amount = capital * bet_ratio
        else:
            bet_ratio = min(bet_ratio, bet_pct_normal)
            bet_amount = capital * bet_ratio

        if bet_amount < 1.0 or bet_amount > capital:
            continue

        n_trades += 1
        correct = (direction == 1 and actual == 1) or (direction == 0 and actual == 0)

        if correct:
            pnl = bet_amount * odds - bet_amount * total_cost
            wins += 1
            total_profit += pnl
        else:
            pnl = -(bet_amount + bet_amount * total_cost)
            losses += 1
            total_loss += abs(pnl)
            cooldown_remaining = cooldown_bars

        capital += pnl
        peak = max(peak, capital)
        min_capital = min(min_capital, capital)

        if capital <= 0:
            capital = 0
            break

    final_capital = capital
    max_dd = (peak - min_capital) / peak if peak > 0 else 0
    win_rate = wins / n_trades if n_trades > 0 else 0
    return_pct = (final_capital - initial_capital) / initial_capital * 100
    pf = total_profit / total_loss if total_loss > 0 else (999 if total_profit > 0 else 0)

    return {
        "final_capital": round(final_capital, 2),
        "return_pct": round(return_pct, 1),
        "win_rate": round(win_rate, 4),
        "max_drawdown": round(max_dd, 4),
        "n_trades": n_trades,
        "wins": wins,
        "losses": losses,
        "profit_factor": round(pf, 2),
        "min_capital": round(min_capital, 2),
    }


def run_v5_backtest(buy_price: float = None) -> List[Dict[str, Any]]:
    """对 v5 模型的测试集预测做回测，分 BTC/ETH/XRP + 合计。"""
    # 加载预测
    if not V5_PRED_FILE.exists():
        print(f"  ⚠️  v5 预测文件不存在: {V5_PRED_FILE}")
        return []

    pred_df = pd.read_parquet(V5_PRED_FILE)
    print(f"  v5 预测: {len(pred_df)} 行, 时间 {pred_df['timestamp'].min()} ~ {pred_df['timestamp'].max()}")

    # 加载最优参数（优先加载对应买价的版本）
    bp_val = buy_price if buy_price is not None else BUY_PRICE
    price_tag = f"bp{bp_val:.3f}".replace(".", "")
    price_specific_file = RESULTS_DIR / f"optimal_trading_rules_v3_{price_tag}.json"
    rules_file = price_specific_file if price_specific_file.exists() else OPTIMAL_RULES_FILE
    if rules_file.exists():
        with open(rules_file) as f:
            opt = json.load(f)
        params = opt["trading_rules"]
        file_bp = opt.get("polymarket_constraints", {}).get("buy_price", "?")
        print(f"  加载参数文件: {rules_file.name} (buy_price={file_bp})")
        # 将 confidence_tiers 转换为 simulate_trading 需要的格式
        tiers = params.get("confidence_tiers", [])
        if tiers and len(tiers) >= 2:
            params["conf_tier1_bound"] = tiers[0][1] if len(tiers[0]) > 1 else 0.55
            params["conf_tier2_bound"] = tiers[1][1] if len(tiers[1]) > 1 else 0.60
            params["tier1_mult"] = tiers[0][2] if len(tiers[0]) > 2 else 0.30
            params["tier2_mult"] = tiers[1][2] if len(tiers[1]) > 2 else 0.60
            params["tier3_mult"] = tiers[2][2] if len(tiers[2]) > 2 else 1.00
        print(f"  最优参数: min_conf={params.get('min_confidence', '?')}, "
              f"min_edge={params.get('min_edge', '?')}, kelly={params.get('kelly_frac', '?')}")
    else:
        print(f"  ⚠️  最优参数文件不存在，使用默认参数")
        params = {
            "min_confidence": 0.54, "min_edge": 0.03, "kelly_frac": 0.33,
            "bet_pct_normal": 0.05, "bet_pct_conservative": 0.03,
            "conf_tier1_bound": 0.55, "conf_tier2_bound": 0.60,
            "tier1_mult": 0.30, "tier2_mult": 0.60, "tier3_mult": 1.00,
            "cooldown_bars": 0, "drawdown_halt": 0.20,
        }

    initial_capital = 400.0
    results = []

    # 分币种回测
    for asset in ["BTC_USDT", "ETH_USDT", "XRP_USDT"]:
        asset_df = pred_df[pred_df["asset"] == asset].sort_values("timestamp").reset_index(drop=True)
        if len(asset_df) == 0:
            continue

        res = simulate_trading(asset_df, params, initial_capital, buy_price=buy_price)
        coin = asset.split("_")[0]
        results.append({
            "排名": 0,
            "组合": f"v5_{coin}",
            "组合ID": f"V5_{coin}_OPT",
            "币种": coin,
            "下注方式": f"Kelly×{params.get('kelly_frac', '?'):.2f}+档位",
            "最终资金": res["final_capital"],
            "初始资金": initial_capital,
            "胜率%": round(res["win_rate"] * 100, 1),
            "盈亏%": res["return_pct"],
            "最大回撤%": round(res["max_drawdown"] * 100, 1),
            "交易次数": res["n_trades"],
            "最低/初始%": round(res["min_capital"] / initial_capital * 100, 1),
            "来源": "v5回测",
        })
        print(f"  {coin}: ${res['final_capital']:.2f} ({res['return_pct']:+.1f}%), "
              f"胜率={res['win_rate']:.1%}, 交易={res['n_trades']}, DD={res['max_drawdown']:.1%}")

    # 全部币种合计（pool）
    all_df = pred_df.sort_values("timestamp").reset_index(drop=True)
    res_all = simulate_trading(all_df, params, initial_capital, buy_price=buy_price)
    results.append({
        "排名": 0,
        "组合": "v5_ALL",
        "组合ID": "V5_ALL_OPT",
        "币种": "BTC+ETH+XRP",
        "下注方式": f"Kelly×{params.get('kelly_frac', '?'):.2f}+档位",
        "最终资金": res_all["final_capital"],
        "初始资金": initial_capital,
        "胜率%": round(res_all["win_rate"] * 100, 1),
        "盈亏%": res_all["return_pct"],
        "最大回撤%": round(res_all["max_drawdown"] * 100, 1),
        "交易次数": res_all["n_trades"],
        "最低/初始%": round(res_all["min_capital"] / initial_capital * 100, 1),
        "来源": "v5回测",
    })
    print(f"  ALL: ${res_all['final_capital']:.2f} ({res_all['return_pct']:+.1f}%), "
          f"胜率={res_all['win_rate']:.1%}, 交易={res_all['n_trades']}, DD={res_all['max_drawdown']:.1%}")

    return results


# ═══════════════════════════════════════════════════════════
# Part 3: 合并排名 + 输出
# ═══════════════════════════════════════════════════════════

def print_ranking(combos: List[Dict[str, Any]], buy_price: float = 0.527):
    """按最终资金排名打印表格。"""
    # 排序
    combos = sorted(combos, key=lambda x: -x["最终资金"])
    for i, c in enumerate(combos):
        c["排名"] = i + 1

    odds = (1.0 - buy_price) / buy_price
    print("\n" + "=" * 130)
    print("  📊 全模型组合对比排名 — 按最终资金降序")
    print("  旧 22 组合: 模拟盘实际结果（规则不变）")
    print(f"  v5 新模型: 历史回测（15 天测试集，买价${buy_price:.3f}，赔率{odds:.4f}）")
    print(f"  ※ 约束: 买价${buy_price:.3f} | 手续费+滑点0.5% | 初始$400")
    print("=" * 130)

    # 表头
    header = (
        f"{'#':>3} {'组合':<25} {'组合ID':<20} {'币种':<12} "
        f"{'下注方式':<18} {'最终资金':>8} {'胜率%':>6} {'盈亏%':>7} "
        f"{'回撤%':>6} {'交易':>5} {'最低%':>6} {'来源':<6}"
    )
    print(header)
    print("─" * 130)

    for c in combos:
        is_v5 = c["来源"] == "v5回测"
        marker = "★" if is_v5 else " "

        line = (
            f"{c['排名']:>3}{marker}"
            f"{'  ' + c['组合']:<25} {c['组合ID']:<20} {c['币种']:<12} "
            f"{c['下注方式']:<18} {'$' + str(c['最终资金']):>8} {c['胜率%']:>5.1f}% "
            f"{c['盈亏%']:>+6.1f}% {c['最大回撤%']:>5.1f}% {c['交易次数']:>5} "
            f"{c['最低/初始%']:>5.1f}% {c['来源']:<6}"
        )
        print(line)

    print("─" * 130)

    # 统计
    old_combos = [c for c in combos if c["来源"] == "模拟盘"]
    new_combos = [c for c in combos if c["来源"] == "v5回测"]

    if old_combos:
        profitable_old = len([c for c in old_combos if c["最终资金"] > c["初始资金"]])
        avg_wr_old = np.mean([c["胜率%"] for c in old_combos])
        avg_pnl_old = np.mean([c["盈亏%"] for c in old_combos])
        print(f"\n  旧 {len(old_combos)} 个组合: 盈利 {profitable_old} 个, "
              f"平均胜率 {avg_wr_old:.1f}%, 平均盈亏 {avg_pnl_old:+.1f}%")

    if new_combos:
        for c in new_combos:
            print(f"  ★ v5 {c['组合']}: ${c['最终资金']:.2f} "
                  f"(胜率{c['胜率%']:.1f}%, 盈亏{c['盈亏%']:+.1f}%, "
                  f"回撤{c['最大回撤%']:.1f}%, {c['交易次数']}笔)")

    print("\n" + "=" * 130)

    return combos


def save_ranking(combos: List[Dict[str, Any]], output_path: Path):
    """保存排名到 CSV。"""
    df = pd.DataFrame(combos)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n  📁 排名已保存: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="全模型组合对比排名")
    parser.add_argument("--buy-price", type=float, default=0.527,
                        help="v5 回测买入价（默认 $0.527；限价单用 0.51 等）")
    args = parser.parse_args()

    bp = args.buy_price
    global BUY_PRICE
    BUY_PRICE = bp  # 更新全局默认值

    odds = (1.0 - bp) / bp
    print("\n" + "=" * 60)
    print("  全模型组合对比 — 22 个模拟盘 + v5 新模型")
    print(f"  v5 买价: ${bp:.3f} | 赔率: {odds:.4f}")
    print("=" * 60)

    # 1. 读取现有 22 个组合
    print("\n📋 读取现有模拟盘组合...")
    existing = load_existing_combos()
    print(f"  找到 {len(existing)} 个组合")

    # 2. v5 新模型回测
    print(f"\n🧪 v5 新模型回测 (BTC/ETH/XRP) @ 买价${bp:.3f}...")
    v5_results = run_v5_backtest(buy_price=bp)

    # 3. 合并 + 排名
    all_combos = existing + v5_results
    ranked = print_ranking(all_combos, buy_price=bp)

    # 4. 保存
    price_tag = f"bp{bp:.3f}".replace(".", "")
    output_csv = RESULTS_DIR / f"combo_ranking_{price_tag}.csv"
    save_ranking(ranked, output_csv)


if __name__ == "__main__":
    main()
