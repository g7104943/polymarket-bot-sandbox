#!/usr/bin/env python3
"""
GRU 模型 Edge+Kelly 超参搜索（方案 C）

用 Optuna TPE 贝叶斯优化，为每个 GRU 预测源在 bp0460 (买价=0.46) 下
搜索最优 Edge+Kelly 参数：minEdge, kellyFrac, confTiers, drawdownHalt 等。

数据: 2.5 年历史回测预测数据（GRU 模型推理）
方法: 前 N-30 天选参（2 段交叉），最后 30 天冷验证
输出: 最优参数 JSON + 新旧对比报告

用法:
  python scripts/optuna_gru_edge_kelly.py
  python scripts/optuna_gru_edge_kelly.py --trials 500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np  # noqa: F401
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "gru_regime_v1"))

# ─── 常量 ───
BUY_PRICE = 0.46
INITIAL_CAPITAL = 400.0
MIN_BET = 1.0
CAPITAL_THRESHOLD = 60000.0
MAX_BET_CAP = 3000.0
FEE_RATE = 0.002
SLIPPAGE = 0.003
TOTAL_COST = FEE_RATE + SLIPPAGE

COLD_DAYS = 30
LAMBDA_DD = 0.4
DATA_YEARS_DEFAULT = 2.5

# ─── GRU 组合定义（与线上 trader_configs 对应） ───
# 按预测源分组：同一预测源共享 GRU 模型，仅阈值/参数不同
GRU_COMBOS = [
    {"id": "gru_eth",        "asset": "ETH_USDT", "use_no1h4h": False,
     "traders": [
         {"name": "gru_eth_54",         "probThreshold": 0.54, "betSizePercent": 5.0, "limitPrice": 0.53},
         {"name": "gru_eth_55_dyn",     "probThreshold": 0.55, "betSizePercent": 5.0, "limitPrice": 0.53},
         {"name": "gru_eth_54_超参",    "probThreshold": 0.56, "betSizePercent": 5.0, "limitPrice": 0.53},
         {"name": "gru_eth_55_dyn_超参","probThreshold": 0.57, "betSizePercent": 5.0, "limitPrice": 0.53},
     ]},
    {"id": "gru_eth_no1h4h", "asset": "ETH_USDT", "use_no1h4h": True,
     "traders": [
         {"name": "gru_eth_55_no1h4h_超参", "probThreshold": 0.57, "betSizePercent": 5.0, "limitPrice": 0.53},
     ]},
    {"id": "gru_btc_no1h4h", "asset": "BTC_USDT", "use_no1h4h": True,
     "traders": [
         {"name": "gru_btc_57_no1h4h_超参", "probThreshold": 0.59, "betSizePercent": 5.0, "limitPrice": 0.53},
     ]},
    {"id": "gru_sol",        "asset": "SOL_USDT", "use_no1h4h": False,
     "traders": [
         {"name": "gru_sol_52",       "probThreshold": 0.52, "betSizePercent": 5.0, "limitPrice": 0.53},
         {"name": "gru_sol_52_超参",  "probThreshold": 0.54, "betSizePercent": 5.0, "limitPrice": 0.53},
     ]},
]


def get_gru_trades(
    asset: str, start_date: str, end_date: str,
    use_no1h4h: bool = False,
) -> List[Dict[str, Any]]:
    """获取 GRU 回测交易数据。"""
    from scripts.backtest_gru_regime import get_backtest_df_one_asset
    from experiments.gru_regime_v1.src.utils import get_device

    device = get_device(use_mps=False)
    models_dir = PROJECT_ROOT / "experiments" / "gru_regime_v1" / "outputs" / "models_best"
    models_no1h4h = PROJECT_ROOT / "experiments" / "gru_regime_v1" / "outputs" / "models_best_no1h4h"
    models_ov = models_no1h4h if use_no1h4h and models_no1h4h.exists() else models_dir

    merged = get_backtest_df_one_asset(
        asset, test_days=99999, device=device,
        data_src=DATA_RAW, after_date=start_date, end_date=end_date,
        models_best_override=models_ov,
    )

    trades = []
    for _, row in merged.iterrows():
        try:
            pred_prob = float(row["pred_prob"])
        except (TypeError, ValueError, KeyError):
            continue
        pred_up = 1 if pred_prob >= 0.5 else 0
        actual_up = int(row.get("actual_up", 0))
        confidence = pred_prob if pred_up == 1 else (1.0 - pred_prob)
        direction = "UP" if pred_up == 1 else "DOWN"
        result = "win" if pred_up == actual_up else "lose"

        ts = row.get("timestamp") or row.get("date")
        if hasattr(ts, "value"):
            ts_ms = int(ts.value // 10**6)
        elif isinstance(ts, (int, float)):
            ts_ms = int(ts)
        else:
            ts_ms = int(pd.to_datetime(ts).value // 10**6)

        trades.append({
            "timestamp": ts_ms,
            "confidence": confidence,
            "proba_up": pred_prob,
            "direction": direction,
            "result": result,
        })

    trades.sort(key=lambda t: t["timestamp"])
    return trades


COOLDOWN_BARS = 2
HALT_BARS = 96  # ~24h at 15-min K lines

def run_edge_kelly_equity(
    trades: List[Dict],
    min_confidence: float,
    min_edge: float,
    kelly_frac: float,
    bet_pct_normal: float,
    bet_pct_conservative: float,
    conf_tier1_bound: float,
    conf_tier2_bound: float,
    tier1_mult: float,
    tier2_mult: float,
    tier3_mult: float,
    drawdown_halt: float,
    buy_price: float = BUY_PRICE,
) -> Dict[str, Any]:
    """模拟 Edge+Kelly 交易策略的资金曲线，与生产 prediction_executor.ts 对齐。"""
    capital = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    wins = losses = skips = 0
    equity = [capital]
    halted = False
    odds = 1.0 / buy_price - 1.0
    ever_reached_cap = False
    cooldown_remaining = 0
    halt_remaining = 0

    for t in trades:
        if capital < MIN_BET:
            break

        # 24h drawdown halt: skip bars, then reset peak and resume
        if halt_remaining > 0:
            halt_remaining -= 1
            if halt_remaining == 0:
                peak = capital
            skips += 1
            continue

        dd = (peak - capital) / peak if peak > 0 else 0
        if dd >= drawdown_halt:
            halted = True
            halt_remaining = HALT_BARS - 1
            skips += 1
            continue

        # Cooldown after loss (COOLDOWN_BARS K lines)
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            skips += 1
            continue

        conf = t["confidence"]
        if conf < min_confidence:
            skips += 1
            continue

        edge = conf * odds - (1.0 - conf)
        if edge < min_edge:
            skips += 1
            continue

        kelly_raw = max(0.0, (conf * odds - (1.0 - conf)) / odds) if odds > 0 else 0.0
        kelly_adj = kelly_raw * kelly_frac

        if conf < conf_tier1_bound:
            tier_mult = tier1_mult
        elif conf < conf_tier2_bound:
            tier_mult = tier2_mult
        else:
            tier_mult = tier3_mult

        bet_ratio = kelly_adj * tier_mult
        ratio_cap = bet_pct_conservative if ever_reached_cap else bet_pct_normal
        bet_ratio = min(bet_ratio, ratio_cap)

        if capital >= CAPITAL_THRESHOLD:
            if not ever_reached_cap:
                ever_reached_cap = True
            bet = min(MAX_BET_CAP, capital)
        else:
            bet = capital * bet_ratio
            bet = max(min(bet, capital * 0.95), 0)

        if bet < MIN_BET:
            skips += 1
            continue

        fee = bet * TOTAL_COST
        if t["result"] == "win":
            pnl = bet * (1.0 / buy_price - 1.0) - fee
            wins += 1
        else:
            pnl = -bet - fee
            losses += 1
            cooldown_remaining = COOLDOWN_BARS

        capital += pnl
        if capital > peak:
            peak = capital
        equity.append(capital)

    peak_eq = INITIAL_CAPITAL
    max_dd = 0.0
    for c in equity:
        if c > peak_eq:
            peak_eq = c
        if peak_eq > 0:
            dd = (peak_eq - c) / peak_eq
            if dd > max_dd:
                max_dd = dd

    n_trades = wins + losses
    wr = wins / n_trades * 100 if n_trades > 0 else 0

    return {
        "final_capital": capital,
        "n_trades": n_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": wr,
        "max_drawdown": max_dd * 100,
        "skips": skips,
        "halted": halted,
    }


def run_old_simple_equity(
    trades: List[Dict],
    prob_threshold: float,
    bet_pct: float,
    buy_price: float,
) -> Dict[str, Any]:
    """模拟旧的简单模式交易。"""
    capital = INITIAL_CAPITAL
    peak = INITIAL_CAPITAL
    wins = losses = 0
    equity = [capital]

    for t in trades:
        if capital < MIN_BET:
            break
        conf = t["confidence"]
        if conf < prob_threshold:
            continue

        if capital >= CAPITAL_THRESHOLD:
            bet = min(MAX_BET_CAP, capital)
        else:
            bet = capital * bet_pct
            bet = min(bet, capital)

        if bet < MIN_BET:
            continue

        fee = bet * TOTAL_COST
        if t["result"] == "win":
            pnl = bet * (1.0 / buy_price - 1.0) - fee
            wins += 1
        else:
            pnl = -bet - fee
            losses += 1

        capital += pnl
        if capital > peak:
            peak = capital
        equity.append(capital)

    peak_eq = INITIAL_CAPITAL
    max_dd = 0.0
    for c in equity:
        if c > peak_eq:
            peak_eq = c
        if peak_eq > 0:
            dd = (peak_eq - c) / peak_eq
            if dd > max_dd:
                max_dd = dd

    n_trades = wins + losses
    wr = wins / n_trades * 100 if n_trades > 0 else 0
    return {
        "final_capital": capital, "n_trades": n_trades,
        "wins": wins, "losses": losses, "win_rate": wr,
        "max_drawdown": max_dd * 100,
    }


def create_objective(
    train_trades: List[Dict],
    min_conf_range: tuple[float, float] = (0.505, 0.60),
    min_edge_range: tuple[float, float] = (0.01, 0.10),
    tier1_offset_range: tuple[float, float] = (0.01, 0.05),
    tier2_offset_range: tuple[float, float] = (0.03, 0.12),
):
    """创建 Optuna 目标函数。"""
    mid = len(train_trades) // 2
    seg1 = train_trades[:mid]
    seg2 = train_trades[mid:]

    def objective(trial: optuna.Trial) -> float:
        conf_low = max(0.0, min(0.999, float(min_conf_range[0])))
        conf_high = max(conf_low + 1e-6, min(0.999, float(min_conf_range[1])))
        edge_low = max(0.0, float(min_edge_range[0]))
        edge_high = max(edge_low + 1e-6, float(min_edge_range[1]))
        min_confidence = trial.suggest_float("min_confidence", conf_low, conf_high, step=0.005)
        min_edge = trial.suggest_float("min_edge", edge_low, edge_high, step=0.005)
        kelly_frac = trial.suggest_float("kelly_frac", 0.3, 1.0, step=0.05)
        bet_pct_normal = trial.suggest_float("bet_pct_normal", 0.02, 0.15, step=0.005)
        bet_pct_conservative = trial.suggest_float("bet_pct_conservative", 0.01, 0.08, step=0.005)
        t1_low = max(min_confidence + float(tier1_offset_range[0]), 0.0)
        t1_high = min(min_confidence + float(tier1_offset_range[1]), 0.985)
        if t1_low >= t1_high:
            return -1e9
        conf_tier1_bound = trial.suggest_float("conf_tier1_bound", t1_low, t1_high, step=0.005)
        t2_low = max(min_confidence + float(tier2_offset_range[0]), conf_tier1_bound + 0.005)
        t2_high = min(min_confidence + float(tier2_offset_range[1]), 0.995)
        if t2_low >= t2_high:
            return -1e9
        conf_tier2_bound = trial.suggest_float("conf_tier2_bound", t2_low, t2_high, step=0.005)
        tier1_mult = trial.suggest_float("tier1_mult", 0.1, 1.0, step=0.05)
        tier2_mult = trial.suggest_float("tier2_mult", 0.2, 1.2, step=0.05)
        tier3_mult = trial.suggest_float("tier3_mult", 0.3, 1.5, step=0.05)
        drawdown_halt = trial.suggest_float("drawdown_halt", 0.15, 0.50, step=0.05)

        if conf_tier2_bound <= conf_tier1_bound:
            return -1e6

        params = dict(
            min_confidence=min_confidence, min_edge=min_edge,
            kelly_frac=kelly_frac, bet_pct_normal=bet_pct_normal,
            bet_pct_conservative=bet_pct_conservative,
            conf_tier1_bound=conf_tier1_bound, conf_tier2_bound=conf_tier2_bound,
            tier1_mult=tier1_mult, tier2_mult=tier2_mult, tier3_mult=tier3_mult,
            drawdown_halt=drawdown_halt,
        )

        r1 = run_edge_kelly_equity(seg1, **params)
        r2 = run_edge_kelly_equity(seg2, **params)

        if r1["n_trades"] < 15 or r2["n_trades"] < 15:
            return -1e6

        score1 = r1["final_capital"] * (1 - LAMBDA_DD * r1["max_drawdown"] / 100)
        score2 = r2["final_capital"] * (1 - LAMBDA_DD * r2["max_drawdown"] / 100)
        return min(score1, score2)

    return objective


def run_search_for_combo(
    combo: Dict,
    n_trials: int,
    data_years: float = DATA_YEARS_DEFAULT,
    min_conf_range: tuple[float, float] = (0.505, 0.60),
    min_edge_range: tuple[float, float] = (0.01, 0.10),
    tier1_offset_range: tuple[float, float] = (0.01, 0.05),
    tier2_offset_range: tuple[float, float] = (0.03, 0.12),
) -> Dict[str, Any]:
    """对单个 GRU 组合运行 Optuna 搜索。"""
    combo_id = combo["id"]
    asset = combo["asset"]
    use_no1h4h = combo.get("use_no1h4h", False)

    print(f"\n{'='*60}")
    print(f"  搜索: {combo_id} ({asset}, no1h4h={use_no1h4h})")
    print(f"{'='*60}")

    end_date_dt = datetime.now(timezone.utc)
    start_date_dt = end_date_dt - timedelta(days=int(data_years * 365))
    cold_start_dt = end_date_dt - timedelta(days=COLD_DAYS)

    start_date = start_date_dt.strftime("%Y-%m-%d")
    end_date = end_date_dt.strftime("%Y-%m-%d")
    cold_start = cold_start_dt.strftime("%Y-%m-%d")

    print(f"  数据范围: {start_date} ~ {end_date} ({data_years:.1f}年)")
    print(f"  冷验证: {cold_start} ~ {end_date}")

    t0 = time.perf_counter()
    print(f"  加载回测数据...", flush=True)
    all_trades = get_gru_trades(asset, start_date, end_date, use_no1h4h)
    print(f"  总交易: {len(all_trades)} 笔 ({time.perf_counter()-t0:.1f}s)")

    if len(all_trades) < 50:
        print(f"  ⚠️ 数据不足，跳过")
        return {"combo_id": combo_id, "status": "skipped", "reason": "insufficient_data"}

    cold_ms = int(cold_start_dt.timestamp() * 1000)
    train_trades = [t for t in all_trades if t["timestamp"] < cold_ms]
    cold_trades = [t for t in all_trades if t["timestamp"] >= cold_ms]
    print(f"  训练: {len(train_trades)} 笔, 冷验证: {len(cold_trades)} 笔")

    # ─── Optuna 搜索 ───
    print(f"  开始 Optuna 搜索 ({n_trials} trials)...", flush=True)
    t1 = time.perf_counter()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    objective = create_objective(
        train_trades,
        min_conf_range=min_conf_range,
        min_edge_range=min_edge_range,
        tier1_offset_range=tier1_offset_range,
        tier2_offset_range=tier2_offset_range,
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  搜索完成 ({time.perf_counter()-t1:.1f}s)")

    best = study.best_params
    best_score = study.best_value
    print(f"  最优分数: {best_score:.2f}")

    # ─── 冷验证 ───
    r_cold_new = run_edge_kelly_equity(cold_trades, **best)
    r_full_new = run_edge_kelly_equity(all_trades, **best)

    # ─── 对每个旧 trader 跑基线对比 ───
    trader_results = []
    for tr in combo["traders"]:
        old_price = tr["limitPrice"]
        old_threshold = tr["probThreshold"]
        old_bet = tr["betSizePercent"] / 100.0

        r_f_old = run_old_simple_equity(all_trades, old_threshold, old_bet, old_price)
        r_c_old = run_old_simple_equity(cold_trades, old_threshold, old_bet, old_price)
        trader_results.append({
            "name": tr["name"],
            "old_params": tr,
            "full_old": r_f_old,
            "cold_old": r_c_old,
        })

    result = {
        "combo_id": combo_id,
        "asset": asset,
        "status": "completed",
        "best_params": best,
        "best_score": best_score,
        "cold_new": r_cold_new,
        "full_new": r_full_new,
        "trader_baselines": trader_results,
        "n_trials": n_trials,
        "data_range": f"{start_date} ~ {end_date}",
        "traders": [t["name"] for t in combo["traders"]],
    }

    print(f"\n  --- {combo_id} 对比 (新 Edge+Kelly bp0460 vs 各旧 trader) ---")
    for tr in trader_results:
        print(f"\n    旧 trader: {tr['name']} (th={tr['old_params']['probThreshold']}, bp={tr['old_params']['limitPrice']})")
        for label, old_r in [("全量回测", tr["full_old"]), ("冷验证30天", tr["cold_old"])]:
            new_r = r_full_new if label == "全量回测" else r_cold_new
            oc = old_r["final_capital"]
            nc = new_r["final_capital"]
            owr = old_r["win_rate"]
            nwr = new_r["win_rate"]
            print(f"      {label}: 旧${oc:.2f}({owr:.1f}%) → 新${nc:.2f}({nwr:.1f}%) 变化{nc-oc:+.2f}")

    return result


def main():
    parser = argparse.ArgumentParser(description="GRU Edge+Kelly Optuna 超参搜索")
    parser.add_argument("--trials", type=int, default=500, help="每个组合的搜索次数")
    parser.add_argument("--combo", type=str, default=None, help="只搜索指定组合 (如 gru_eth)")
    parser.add_argument("--data-years", type=float, default=DATA_YEARS_DEFAULT, help="回溯数据年数 (0=全量)")
    parser.add_argument("--min-confidence-range", type=str, default="0.505,0.60", help="min_confidence 搜索范围，如 0.60,0.70")
    parser.add_argument("--min-edge-range", type=str, default="0.01,0.10", help="min_edge 搜索范围")
    parser.add_argument("--tier1-offset-range", type=str, default="0.01,0.05", help="tier1 相对 min_conf 偏移范围")
    parser.add_argument("--tier2-offset-range", type=str, default="0.03,0.12", help="tier2 相对 min_conf 偏移范围")
    args = parser.parse_args()

    def _parse_range(raw: str) -> tuple[float, float]:
        parts = [p.strip() for p in str(raw).split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError(f"范围参数格式错误: {raw}")
        lo = float(parts[0])
        hi = float(parts[1])
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi

    min_conf_range = _parse_range(args.min_confidence_range)
    min_edge_range = _parse_range(args.min_edge_range)
    tier1_offset_range = _parse_range(args.tier1_offset_range)
    tier2_offset_range = _parse_range(args.tier2_offset_range)

    data_years = args.data_years if args.data_years > 0 else 20.0

    print("=" * 60)
    print("  GRU Edge+Kelly Optuna 超参搜索 (方案 C)")
    print(f"  买价: {BUY_PRICE}")
    print(f"  数据: {data_years} 年" + (" (全量)" if args.data_years == 0 else ""))
    print(f"  Trials: {args.trials}")
    print(f"  min_conf 搜索: [{min_conf_range[0]:.3f}, {min_conf_range[1]:.3f}]")
    print(f"  组合: {len(GRU_COMBOS)} 个")
    print("=" * 60)

    combos = GRU_COMBOS
    if args.combo:
        combos = [c for c in combos if c["id"] == args.combo]
        if not combos:
            print(f"未找到组合: {args.combo}")
            return

    results = []
    for combo in combos:
        r = run_search_for_combo(
            combo,
            args.trials,
            data_years=data_years,
            min_conf_range=min_conf_range,
            min_edge_range=min_edge_range,
            tier1_offset_range=tier1_offset_range,
            tier2_offset_range=tier2_offset_range,
        )
        results.append(r)

    # ─── 汇总报告 ───
    print("\n" + "=" * 70)
    print("  汇总：全部组合对比 (新 Edge+Kelly bp0460 vs 旧最优 trader)")
    print("=" * 70)
    print(f"  {'组合':20s} {'旧最优资金':>12s} {'新资金':>12s} {'变化':>12s} {'旧WR':>6s} {'新WR':>6s}")
    any_improved = False
    for r in results:
        if r["status"] != "completed":
            print(f"  {r['combo_id']:20s} 跳过: {r.get('reason','')}")
            continue
        best_old = max(r["trader_baselines"], key=lambda x: x["full_old"]["final_capital"])
        oc = best_old["full_old"]["final_capital"]
        nc = r["full_new"]["final_capital"]
        owr = best_old["full_old"]["win_rate"]
        nwr = r["full_new"]["win_rate"]
        improved = nc > oc
        if improved:
            any_improved = True
        mark = "+" if improved else "-"
        print(f"  {r['combo_id']:20s} ${oc:>10.2f} ${nc:>10.2f} {nc-oc:>+10.2f} {owr:>5.1f}% {nwr:>5.1f}% [{mark}]")

    # 保存结果
    out_dir = PROJECT_ROOT / "experiments" / "gru_edge_kelly_optuna"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    for r in results:
        if r["status"] != "completed":
            continue
        out_file = out_dir / f"optimal_{r['combo_id']}_bp0460.json"
        with open(out_file, "w") as f:
            json.dump(r, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n  已保存: {out_file}")

    summary_file = out_dir / f"summary_{ts}.json"
    with open(summary_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  汇总: {summary_file}")

    if any_improved:
        print("\n  ✅ 有组合出现提升，可更新 trader_configs.json")
    else:
        print("\n  ⚠️ 无明显提升，建议保持现有配置")


if __name__ == "__main__":
    main()
