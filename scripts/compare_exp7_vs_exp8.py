#!/usr/bin/env python3
"""
Exp7 vs Exp8 公平对比回测 — 10 对 10 模型直接对比

核心原则:
  ✅ 同一 15 天无泄漏测试集
  ✅ 每个 Exp 使用各自的模型预测 + 各自优化的超参
  ✅ 同一回测引擎（simulate_v5_trading）
  ✅ 0.45~0.53 固定买价（9 个）+ 动态阶梯（1 个）= 10 个模型

用法:
  python scripts/compare_exp7_vs_exp8.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_fair import (
    simulate_v5_trading, _load_v5_params,
    DEFAULT_INITIAL_CAPITAL, DEFAULT_FEE_RATE, DEFAULT_SLIPPAGE_RATE,
    TOTAL_COST, LIQUIDITY_CAP, LIQUIDITY_BET,
)

RESULTS_DIR = PROJECT_ROOT / "experiments" / "sentiment_grid_search" / "results"

# 文件路径
EXP7_PRED = RESULTS_DIR / "exp7_test_predictions.parquet"
EXP8_PRED = RESULTS_DIR / "exp8_test_predictions.parquet"
EXP7_PARAMS_DIR = RESULTS_DIR / "exp7_params"
EXP8_PARAMS_DIR = RESULTS_DIR  # Exp8 在主目录

# 所有买价（包含 0.52 和 0.53）
ALL_FIXED_PRICES = [0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53]
LADDER_PRICES = [0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53]

DATE_FROM = "2026-01-26"
DATE_TO = "2026-02-10"


def run_single(pred_df: pd.DataFrame, params: Dict, bp: float,
               initial_capital: float = DEFAULT_INITIAL_CAPITAL) -> Dict[str, Any]:
    """运行单个买价的回测。"""
    return simulate_v5_trading(pred_df, params, bp, initial_capital)


def run_ladder(pred_df: pd.DataFrame, params: Dict, prices: List[float],
               initial_capital: float = DEFAULT_INITIAL_CAPITAL) -> Dict[str, Any]:
    """运行阶梯模型回测：均分资金到每个价位。"""
    n = len(prices)
    per_cap = initial_capital / n
    total_final = 0
    total_trades = 0
    total_wins = 0
    total_losses = 0
    max_dd = 0

    for bp in prices:
        res = simulate_v5_trading(pred_df, params, bp, per_cap)
        total_final += res["final_capital"]
        total_trades += res["n_trades"]
        total_wins += res["wins"]
        total_losses += res["losses"]
        max_dd = max(max_dd, res["max_drawdown"])

    wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    ret = (total_final - initial_capital) / initial_capital * 100
    return {
        "final_capital": round(total_final, 2),
        "return_pct": round(ret, 1),
        "win_rate": round(wr, 1),
        "max_drawdown": max_dd,
        "n_trades": total_trades,
        "wins": total_wins,
        "losses": total_losses,
    }


def main():
    print("\n" + "█" * 100)
    print("  Exp7 vs Exp8 公平对比回测 — 10 对 10 模型")
    print("█" * 100)
    print(f"  测试期: {DATE_FROM} ~ {DATE_TO} (15 天无泄漏)")
    print(f"  初始资金: ${DEFAULT_INITIAL_CAPITAL}")
    print(f"  手续费+滑点: {TOTAL_COST*100:.1f}%")
    print(f"  模型: 9 固定买价 + 1 动态阶梯 = 10 个")
    print(f"  Exp7: 无 target PM 特征 (234 特征)")
    print(f"  Exp8: 含 5 个 target PM 特征 (240 特征)")

    # ─── 加载预测 ────────────────────────────────
    if not EXP7_PRED.exists():
        print(f"\n❌ Exp7 预测文件不存在: {EXP7_PRED}")
        return
    if not EXP8_PRED.exists():
        print(f"\n❌ Exp8 预测文件不存在: {EXP8_PRED}")
        return

    exp7_df = pd.read_parquet(EXP7_PRED)
    exp8_df = pd.read_parquet(EXP8_PRED)

    # 日期过滤
    for df in [exp7_df, exp8_df]:
        df["date_str"] = df["timestamp"].astype(str).str[:10]
    exp7_df = exp7_df[(exp7_df["date_str"] >= DATE_FROM) & (exp7_df["date_str"] <= DATE_TO)].drop(columns=["date_str"])
    exp8_df = exp8_df[(exp8_df["date_str"] >= DATE_FROM) & (exp8_df["date_str"] <= DATE_TO)].drop(columns=["date_str"])

    exp7_all = exp7_df.sort_values("timestamp").reset_index(drop=True)
    exp8_all = exp8_df.sort_values("timestamp").reset_index(drop=True)

    print(f"\n  Exp7 预测: {len(exp7_all)} bars, proba std={exp7_all['proba_up'].std():.4f}")
    print(f"  Exp8 预测: {len(exp8_all)} bars, proba std={exp8_all['proba_up'].std():.4f}")

    # ─── 逐买价对比 ────────────────────────────────
    print(f"\n{'=' * 120}")
    print(f"  {'买价':<8} {'│':} {'Exp7 资金':>14} {'胜率':>6} {'交易':>6} {'回撤':>6} "
          f"{'│':} {'Exp8 资金':>14} {'胜率':>6} {'交易':>6} {'回撤':>6} "
          f"{'│':} {'Exp8/Exp7':>10} {'胜出':>6}")
    print(f"{'─' * 120}")

    exp7_results = []
    exp8_results = []

    for bp in ALL_FIXED_PRICES:
        tag = f"bp{bp:.3f}".replace(".", "")

        # Load Exp7 params
        e7_file = EXP7_PARAMS_DIR / f"optimal_trading_rules_v3_{tag}.json"
        e7_loaded = _load_v5_params(e7_file)
        if not e7_loaded:
            print(f"  ${bp:.2f}   │ ⚠️ Exp7 参数缺失")
            continue
        e7_params = e7_loaded["params"]

        # Load Exp8 params
        e8_file = EXP8_PARAMS_DIR / f"optimal_trading_rules_v3_{tag}.json"
        e8_loaded = _load_v5_params(e8_file)
        if not e8_loaded:
            print(f"  ${bp:.2f}   │ ... │ ⚠️ Exp8 参数缺失")
            continue
        e8_params = e8_loaded["params"]

        # Run backtests
        r7 = run_single(exp7_all, e7_params, bp)
        r8 = run_single(exp8_all, e8_params, bp)

        exp7_results.append({"bp": bp, **r7})
        exp8_results.append({"bp": bp, **r8})

        ratio = r8["final_capital"] / r7["final_capital"] if r7["final_capital"] > 0 else 999
        winner = "Exp8 ★" if r8["final_capital"] > r7["final_capital"] else "Exp7"

        print(f"  ${bp:.2f}   │ ${r7['final_capital']:>12,.2f} {r7['win_rate']:>5.1f}% {r7['n_trades']:>5} "
              f"{r7['max_drawdown']:>5.1f}% "
              f"│ ${r8['final_capital']:>12,.2f} {r8['win_rate']:>5.1f}% {r8['n_trades']:>5} "
              f"{r8['max_drawdown']:>5.1f}% "
              f"│ {ratio:>9.1f}x {winner}")

    # ─── 动态阶梯 ────────────────────────────────
    print(f"{'─' * 120}")

    e7_dyn_file = EXP7_PARAMS_DIR / "optimal_trading_rules_v3_bp_dyn_0450_0530.json"
    e8_dyn_file = EXP8_PARAMS_DIR / "optimal_trading_rules_v3_bp_dyn_0450_0530.json"
    e7_dyn = _load_v5_params(e7_dyn_file)
    e8_dyn = _load_v5_params(e8_dyn_file)

    if e7_dyn and e8_dyn:
        r7_lad = run_ladder(exp7_all, e7_dyn["params"], LADDER_PRICES)
        r8_lad = run_ladder(exp8_all, e8_dyn["params"], LADDER_PRICES)

        ratio = r8_lad["final_capital"] / r7_lad["final_capital"] if r7_lad["final_capital"] > 0 else 999
        winner = "Exp8 ★" if r8_lad["final_capital"] > r7_lad["final_capital"] else "Exp7"

        print(f"  动态阶梯 │ ${r7_lad['final_capital']:>12,.2f} {r7_lad['win_rate']:>5.1f}% {r7_lad['n_trades']:>5} "
              f"{r7_lad['max_drawdown']:>5.1f}% "
              f"│ ${r8_lad['final_capital']:>12,.2f} {r8_lad['win_rate']:>5.1f}% {r8_lad['n_trades']:>5} "
              f"{r8_lad['max_drawdown']:>5.1f}% "
              f"│ {ratio:>9.1f}x {winner}")

        exp7_results.append({"bp": "ladder", **r7_lad})
        exp8_results.append({"bp": "ladder", **r8_lad})

    # ─── 汇总 ────────────────────────────────
    print(f"{'=' * 120}")

    e7_wins = sum(1 for r7, r8 in zip(exp7_results, exp8_results)
                  if r7["final_capital"] > r8["final_capital"])
    e8_wins = len(exp7_results) - e7_wins

    e7_avg_ret = np.mean([r["return_pct"] for r in exp7_results])
    e8_avg_ret = np.mean([r["return_pct"] for r in exp8_results])
    e7_avg_wr = np.mean([r["win_rate"] for r in exp7_results])
    e8_avg_wr = np.mean([r["win_rate"] for r in exp8_results])

    print(f"\n  ┌─ 汇总 ────────────────────────────────────────────────────────┐")
    print(f"  │  胜出: Exp7 {e7_wins} 次 vs Exp8 {e8_wins} 次"
          f"{' ← Exp8 更优' if e8_wins > e7_wins else ' ← Exp7 更优' if e7_wins > e8_wins else '':<20}│")
    print(f"  │  平均盈亏: Exp7 {e7_avg_ret:>+12.1f}%  vs  Exp8 {e8_avg_ret:>+12.1f}%          │")
    print(f"  │  平均胜率: Exp7 {e7_avg_wr:>8.1f}%     vs  Exp8 {e8_avg_wr:>8.1f}%              │")
    print(f"  └──────────────────────────────────────────────────────────────────┘")

    # ─── 分币种 ────────────────────────────────
    print(f"\n{'─' * 80}")
    print(f"  分币种对比（以 $0.50 买价为例）:")
    print(f"{'─' * 80}")

    bp_for_coin = 0.50
    tag = f"bp{bp_for_coin:.3f}".replace(".", "")
    e7_file = EXP7_PARAMS_DIR / f"optimal_trading_rules_v3_{tag}.json"
    e8_file = EXP8_PARAMS_DIR / f"optimal_trading_rules_v3_{tag}.json"
    e7_loaded = _load_v5_params(e7_file)
    e8_loaded = _load_v5_params(e8_file)

    if e7_loaded and e8_loaded:
        for asset in ["BTC_USDT", "ETH_USDT", "XRP_USDT"]:
            coin = asset.split("_")[0]
            e7_coin = exp7_all[exp7_all["asset"] == asset].sort_values("timestamp").reset_index(drop=True)
            e8_coin = exp8_all[exp8_all["asset"] == asset].sort_values("timestamp").reset_index(drop=True)

            r7 = simulate_v5_trading(e7_coin, e7_loaded["params"], bp_for_coin)
            r8 = simulate_v5_trading(e8_coin, e8_loaded["params"], bp_for_coin)

            winner = "Exp8 ★" if r8["final_capital"] > r7["final_capital"] else "Exp7"
            print(f"  {coin:<5} │ Exp7: ${r7['final_capital']:>10,.2f} ({r7['win_rate']:.1f}%, {r7['n_trades']}笔) "
                  f"│ Exp8: ${r8['final_capital']:>10,.2f} ({r8['win_rate']:.1f}%, {r8['n_trades']}笔) │ {winner}")

    # ─── 保存结果 ────────────────────────────────
    rows = []
    for r7, r8 in zip(exp7_results, exp8_results):
        bp_label = f"${r7['bp']:.2f}" if isinstance(r7['bp'], float) else r7['bp']
        rows.append({
            "买价": bp_label,
            "Exp7_资金": r7["final_capital"],
            "Exp7_盈亏%": r7["return_pct"],
            "Exp7_胜率%": r7["win_rate"],
            "Exp7_交易数": r7["n_trades"],
            "Exp7_回撤%": r7["max_drawdown"],
            "Exp8_资金": r8["final_capital"],
            "Exp8_盈亏%": r8["return_pct"],
            "Exp8_胜率%": r8["win_rate"],
            "Exp8_交易数": r8["n_trades"],
            "Exp8_回撤%": r8["max_drawdown"],
            "Exp8/Exp7": round(r8["final_capital"] / r7["final_capital"], 2) if r7["final_capital"] > 0 else 999,
            "胜出": "Exp8" if r8["final_capital"] > r7["final_capital"] else "Exp7",
        })

    out_csv = RESULTS_DIR / "exp7_vs_exp8_comparison.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n  📁 结果保存: {out_csv}")

    print(f"\n{'█' * 100}")
    print(f"  ✅ Exp7 vs Exp8 对比完成！")
    print(f"{'█' * 100}")


if __name__ == "__main__":
    main()
