#!/usr/bin/env python3
"""
GRU 旧模型 — 限价上限 (LIMIT_PRICE) 网格搜索

只针对两个模型：
  1. GRU_ETH_55_dyn_超参  (WR=79.4%, PnL=+$275)
  2. GRU_ETH_54_超参       (WR=56.4%, PnL=+$187)

固定其他所有超参（阈值、档位 p1~p4 等），只搜索最优 LIMIT_PRICE。

用 3 个数据窗口交叉验证：
  ① 原始训练测试数据（无泄漏起始日至今）
  ② 365 天
  ③ 2024-01-01（ETH ETF 获批）至今

LIMIT_PRICE 含义：限价上限 → 先吃 ask ≤ LIMIT_PRICE，剩余挂在此价等成交/到期退回。
纯只读回测，不修改任何文件，不影响正在跑的模拟交易。

用法：
  python3 scripts/grid_search_buy_price.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ─── 候选买价 ───
BUY_PRICES = [round(0.44 + i * 0.01, 2) for i in range(12)]  # 0.44 ~ 0.55

# ─── 回测公共参数（同 hyperparam_tune_13_combos.py） ───
INITIAL_CAPITAL = 400.0
MIN_BET = 1.0
CAPITAL_THRESHOLD = 60000.0
MAX_BET_CAP = 3000.0
DEFAULT_FEE_RATE = 0.002
DEFAULT_SLIPPAGE = 0.001

# ─── 两个模型的固定参数（从 启动超参组合.sh 提取，不改） ───
#
# GRU_ETH_55_dyn_超参:
#   PROB_THRESHOLD=0.57  →  threshold=0.55, delta=0.02  →  base=0.57
#   P1=8, P2=10, P3=4, P4=2
#
# GRU_ETH_54_超参:
#   PROB_THRESHOLD=0.56  →  threshold=0.54, delta=0.02  →  base=0.56
#   P1=4, P2=10, P3=10, P4=2
#
# 档位边界 b1=2, b2=4, b3=6（默认）

MODELS_CFG = {
    "GRU_ETH_55_dyn_超参": {
        "asset": "ETH_USDT",
        "gru_threshold": 0.55,  # GRU model threshold
        "use_no1h4h": False,    # 用 models_best（有 1h4h）
        # hyperparam_tune 参数
        "threshold": 0.55,
        "b0": 2,  # delta=0.02 → base=0.57
        "b1": 2, "b2": 4, "b3": 6,
        "p1": 8, "p2": 10, "p3": 4, "p4": 2,
    },
    "GRU_ETH_54_超参": {
        "asset": "ETH_USDT",
        "gru_threshold": 0.54,
        "use_no1h4h": False,
        "threshold": 0.54,
        "b0": 2,  # delta=0.02 → base=0.56
        "b1": 2, "b2": 4, "b3": 6,
        "p1": 4, "p2": 10, "p3": 10, "p4": 2,
    },
}


def get_tier(confidence: float, threshold: float, b0: int, b1: int, b2: int, b3: int) -> int:
    """档 1~4，base = threshold + b0*0.01，边界 +b1,+b2,+b3（百分点）。"""
    base = threshold + b0 * 0.01
    c1 = base + b1 * 0.01
    c2 = base + b2 * 0.01
    c3 = base + b3 * 0.01
    if confidence <= c1:
        return 1
    if confidence <= c2:
        return 2
    if confidence <= c3:
        return 3
    return 4


def run_equity_curve(
    trades: List[Dict[str, Any]],
    threshold: float,
    b0: int, b1: int, b2: int, b3: int,
    p1: int, p2: int, p3: int, p4: int,
    fixed_order_price: float,
) -> Dict[str, Any]:
    """跑资金曲线，返回详细统计。"""
    base = threshold + b0 * 0.01
    pcts = {1: p1 * 0.01, 2: p2 * 0.01, 3: p3 * 0.01, 4: p4 * 0.01}
    sorted_trades = sorted(trades, key=lambda t: str(t.get("date") or t.get("timestamp") or ""))

    capital = INITIAL_CAPITAL
    peak_capital = INITIAL_CAPITAL
    wins = losses = 0
    total_bet = 0.0
    equity_curve = [capital]

    for t in sorted_trades:
        if capital < MIN_BET:
            break
        conf = t.get("confidence")
        if conf is None or conf < base:
            continue
        tier = get_tier(conf, threshold, b0, b1, b2, b3)
        ratio = pcts.get(tier, 0)
        if ratio <= 0:
            continue
        # ≥6 万固定 3000，<6 万按总资金比例
        if capital >= CAPITAL_THRESHOLD:
            bet = min(MAX_BET_CAP, capital)
        else:
            bet = capital * ratio
            bet = min(bet, capital)
        if bet < MIN_BET:
            continue
        price = fixed_order_price
        if price <= 0:
            continue
        fee = bet * (DEFAULT_FEE_RATE + DEFAULT_SLIPPAGE)
        if t.get("result") == "win":
            new_pnl = bet * (1.0 / price - 1.0) - fee
            wins += 1
        else:
            new_pnl = -bet - fee
            losses += 1
        capital += new_pnl
        total_bet += bet
        if capital > peak_capital:
            peak_capital = capital
        equity_curve.append(capital)

    # 最大回撤
    peak = INITIAL_CAPITAL
    max_dd = 0.0
    for c in equity_curve:
        if c > peak:
            peak = c
        dd = (peak - c) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    n_trades = wins + losses
    win_rate = wins / n_trades * 100 if n_trades else 0
    pnl = capital - INITIAL_CAPITAL
    return_pct = pnl / INITIAL_CAPITAL * 100
    bet_return = pnl / total_bet * 100 if total_bet > 0 else 0

    return {
        "final_capital": capital,
        "pnl": pnl,
        "return_pct": return_pct,
        "n_trades": n_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "max_drawdown": max_dd,
        "total_bet": total_bet,
        "bet_return": bet_return,
    }


def main():
    print("=" * 110)
    print("  GRU 旧模型 — 限价上限 (LIMIT_PRICE) 网格搜索")
    print("  只针对 GRU_ETH_55_dyn_超参 和 GRU_ETH_54_超参，固定其他参数，只搜索买价")
    print("  纯只读回测，不影响正在跑的模拟交易")
    print("=" * 110)

    from scripts.backtest_gru_regime import get_no_leak_start_date, get_parquet_end_date, get_device

    DATA_RAW = PROJECT_ROOT / "data" / "raw"
    MODELS_BEST = PROJECT_ROOT / "experiments" / "gru_regime_v1" / "outputs" / "models_best"
    MODELS_BEST_NO1H4H = PROJECT_ROOT / "experiments" / "gru_regime_v1" / "outputs" / "models_best_no1h4h"

    device = get_device(use_mps=True)

    # ── 计算样本外数据窗口 ──
    # GRU 模型训练截止日 = no_leak_start，之后的数据才是真正样本外
    no_leak_start = get_no_leak_start_date(DATA_RAW, None, "ETH_USDT")
    end_date = get_parquet_end_date(DATA_RAW, "ETH_USDT")

    oos_start = pd.Timestamp(no_leak_start, tz="UTC")
    oos_end = pd.Timestamp(end_date, tz="UTC")
    total_oos_days = (oos_end - oos_start).days
    print(f"\n  GRU 训练截止日: {no_leak_start}")
    print(f"  样本外数据: {no_leak_start} ~ {end_date} ({total_oos_days}天)")
    print(f"  ⚠ 只使用训练截止日之后的数据，确保零泄漏")

    # 切成 3 个不重叠的等长时间段
    seg_days = total_oos_days // 3
    seg1_end = (oos_start + pd.Timedelta(days=seg_days)).strftime("%Y-%m-%d")
    seg2_start = seg1_end
    seg2_end = (oos_start + pd.Timedelta(days=seg_days * 2)).strftime("%Y-%m-%d")
    seg3_start = seg2_end

    windows = {
        f"样本外Q1({no_leak_start}~{seg1_end})": (no_leak_start, seg1_end),
        f"样本外Q2({seg2_start}~{seg2_end})": (seg2_start, seg2_end),
        f"样本外Q3({seg3_start}~{end_date})": (seg3_start, end_date),
    }

    print()
    print("  数据窗口:")
    for label, (s, e) in windows.items():
        days = (pd.Timestamp(e, tz="UTC") - pd.Timestamp(s, tz="UTC")).days
        print(f"    {label}: {s} ~ {e} ({days}天)")
    print()

    # ── 对每个模型执行 ──
    for model_name, cfg in MODELS_CFG.items():
        print(f"\n{'━' * 110}")
        print(f"  模型: {model_name}")
        base = cfg["threshold"] + cfg["b0"] * 0.01
        print(f"  固定参数: base阈值={base:.2f}, "
              f"p1={cfg['p1']}%, p2={cfg['p2']}%, p3={cfg['p3']}%, p4={cfg['p4']}%, "
              f"当前LIMIT_PRICE=0.51")
        print(f"{'━' * 110}")

        from scripts.hyperparam_tune_13_combos import get_trades_gru

        models_override = MODELS_BEST_NO1H4H if cfg["use_no1h4h"] else MODELS_BEST
        asset = cfg["asset"]

        # ── 对每个窗口生成 GRU 预测 ──
        window_trades = {}
        for label, (start, end) in windows.items():
            print(f"\n  📊 加载窗口 [{label}] ...", end=" ", flush=True)
            t0 = time.time()
            try:
                trades = get_trades_gru(
                    asset=asset,
                    start_date=start,
                    end_date=end,
                    data_src=DATA_RAW,
                    models_override=models_override,
                    device=device,
                )
                elapsed = time.time() - t0
                print(f"✓ {len(trades)} bars ({elapsed:.1f}s)")
                window_trades[label] = trades
            except Exception as e:
                print(f"✗ {e}")

        if not window_trades:
            print("  ⚠ 没有可用窗口数据，跳过")
            continue

        # ── 网格搜索 ──
        print(f"\n  {'买价':>5}", end="")
        for label in window_trades.keys():
            short = label[:8]
            print(f"  │ {short:>8} {'PnL$':>8} {'WR%':>6} {'笔数':>5} {'DD%':>5} {'BetR%':>6}", end="")
        print(f"  │ {'平均PnL$':>9} {'一致':>4}")
        print(f"  {'─' * 5}", end="")
        for _ in window_trades:
            print(f"  │ {'─' * 44}", end="")
        print(f"  │ {'─' * 15}")

        price_results = {}
        for bp in BUY_PRICES:
            results = {}
            pnl_list = []
            print(f"  {bp:.2f}", end="")

            for label, trades in window_trades.items():
                res = run_equity_curve(
                    trades,
                    cfg["threshold"], cfg["b0"], cfg["b1"], cfg["b2"], cfg["b3"],
                    cfg["p1"], cfg["p2"], cfg["p3"], cfg["p4"],
                    fixed_order_price=bp,
                )
                results[label] = res
                pnl_list.append(res["pnl"])

                short = label[:8]
                m = "✓" if res["pnl"] > 0 else " "
                print(f"  │ {short:>8} ${res['pnl']:>+7.1f} "
                      f"{res['win_rate']:>5.1f}% {res['n_trades']:>5} "
                      f"{res['max_drawdown']:>4.1f}% "
                      f"{res['bet_return']:>+5.1f}%{m}", end="")

            avg_pnl = sum(pnl_list) / len(pnl_list) if pnl_list else 0
            all_pos = all(p > 0 for p in pnl_list)
            all_neg = all(p <= 0 for p in pnl_list)
            cons = "全盈" if all_pos else ("全亏" if all_neg else "分歧")
            print(f"  │ ${avg_pnl:>+8.1f} {cons:>4}")

            price_results[bp] = {"results": results, "avg_pnl": avg_pnl,
                                 "pnl_list": pnl_list, "consistency": cons}

        # ── 找最优 ──
        # 只在「全盈」中选最优；若无全盈则退而求「平均 PnL 最高」
        candidates_all_pos = {p: v for p, v in price_results.items() if v["consistency"] == "全盈"}
        if candidates_all_pos:
            best_bp = max(candidates_all_pos.keys(), key=lambda p: candidates_all_pos[p]["avg_pnl"])
            note = "（全盈一致）"
        else:
            best_bp = max(price_results.keys(), key=lambda p: price_results[p]["avg_pnl"])
            note = "（⚠ 无全盈，取平均最佳）"

        best = price_results[best_bp]
        current = price_results.get(0.51, {})

        print()
        print(f"  >>> {model_name} 最优 LIMIT_PRICE: {best_bp:.2f} {note}")
        print(f"      平均 PnL: ${best['avg_pnl']:+.1f}, 一致性: {best['consistency']}")
        if current:
            improvement = best["avg_pnl"] - current["avg_pnl"]
            pct = improvement / abs(current["avg_pnl"]) * 100 if current["avg_pnl"] != 0 else float('inf')
            print(f"      vs 当前 0.51: PnL ${current['avg_pnl']:+.1f} → 改善 ${improvement:+.1f} ({pct:+.0f}%)")

        # ── 柱状图 ──
        print(f"\n  {model_name} — 各买价平均 PnL 柱状图:")
        max_abs = max(abs(v["avg_pnl"]) for v in price_results.values()) or 1
        for bp in BUY_PRICES:
            pr = price_results[bp]
            bar_len = int(abs(pr["avg_pnl"]) / max_abs * 40)
            bar = "█" * bar_len if pr["avg_pnl"] >= 0 else "░" * bar_len
            marker = ""
            if abs(bp - best_bp) < 0.005:
                marker = " ◄── 最优"
            if abs(bp - 0.51) < 0.005:
                marker += " (当前)"
            print(f"    {bp:.2f} ${pr['avg_pnl']:>+8.1f} [{pr['consistency']:>2}] {bar}{marker}")

        # ── 各窗口各自最优 ──
        print(f"\n  各窗口各自最优买价（是否重合？）:")
        window_bests = {}
        for label in window_trades:
            wb = max(BUY_PRICES, key=lambda p: price_results[p]["results"].get(label, {}).get("pnl", -9999))
            wp = price_results[wb]["results"].get(label, {}).get("pnl", 0)
            window_bests[label] = wb
            print(f"    {label[:20]:>20}: {wb:.2f} (PnL ${wp:+.1f})")

        unique_bests = set(window_bests.values())
        if len(unique_bests) == 1:
            print(f"    ✅ 3 个窗口完全重合 → 高置信度！")
        elif max(unique_bests) - min(unique_bests) <= 0.02:
            print(f"    ⚡ 差异 ≤ 2 分 → 基本一致，可取中间值")
        else:
            print(f"    ⚠ 窗口间有分歧（最大差 {(max(unique_bests)-min(unique_bests))*100:.0f} 分），建议保守选择")

    # ── 综合建议 ──
    print()
    print("=" * 110)
    print("  综合建议")
    print("=" * 110)
    print()
    print("  ● 此回测使用 GRU 模型的真实预测，在 3 个数据窗口上交叉验证")
    print("  ● 固定其他所有参数（阈值、p1~p4、delta），只搜索 LIMIT_PRICE")
    print()
    print("  ● LIMIT_PRICE 是限价上限，不是实际成交价：")
    print("    - 实际成交先吃 ask ≤ LIMIT_PRICE 的卖单，通常成交均价 < LIMIT_PRICE")
    print("    - 回测中假设以 LIMIT_PRICE 成交（最保守假设）")
    print("    - 所以实际表现会优于回测结果")
    print()
    print("  ● 如果 3 个窗口的最优买价重合 → 高置信度，可直接修改")
    print("  ● 如果有分歧 → 取中间值或偏保守（靠近 0.51 方向）")
    print()
    print("=" * 110)


if __name__ == "__main__":
    main()
