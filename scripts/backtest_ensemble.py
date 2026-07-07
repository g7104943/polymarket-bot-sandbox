#!/usr/bin/env python3
"""
Ensemble v2 融合预测回测 — Log-Odds 融合 + 聚合权重

方法:
  1. 加载所有 v5 模型的 parquet 预测文件
  2. 按 timestamp+asset 对齐，用 log-odds 加权融合
  3. 用统一的 simulate_v5_trading 回测
  4. 输出对比表: 单模型 vs 融合的胜率/PnL/Sharpe

v2 改进 (与 ensemble_prediction_writer.py 同步):
  - 融合方法: 概率平均 → log-odds 加权平均
  - 权重门槛: WR 0.45 → 0.48
  - 排除 Exp8/Exp9 (实盘严重过拟合)
  - CONFIDENCE_SCALE: 1.3 → 1.5

用法:
  python scripts/backtest_ensemble.py
  python scripts/backtest_ensemble.py --consensus 0.7
  python scripts/backtest_ensemble.py --buy-price 0.50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from math import sqrt, log, exp

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_fair import (
    simulate_v5_trading,
    DEFAULT_INITIAL_CAPITAL,
    RESULTS_DIR,
)

# 模型目录名 -> 实验标签 (v2: 排除 Exp8 过拟合模型)
MODEL_MAP = {
    "v5_production": "Exp8",
    "v5_production_sim_noise": "Exp10",
    "v5_production_tv": "Exp13",
    "v5_production_tv_365d": "Exp16",
    "v5_production_no_target_pm": "Exp17",
}

EXCLUDED_FROM_ENSEMBLE = {"Exp8"}

WIN_RATE_FLOOR = 0.48
MIN_TRADES_FOR_WEIGHT = 20
DEFAULT_CONSENSUS_UP = 0.60
DEFAULT_CONSENSUS_DOWN = 0.68
# 兼容旧入口
DEFAULT_CONSENSUS = DEFAULT_CONSENSUS_UP
CONFIDENCE_SCALE = 1.5


def _logit(p):
    """Log-odds transform, vectorized-safe."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))


def _sigmoid(x):
    """Inverse logit, vectorized-safe."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def load_model_predictions() -> Dict[str, pd.DataFrame]:
    """加载所有可用模型的 parquet 预测文件。"""
    available = {}
    for model_name, exp_label in MODEL_MAP.items():
        noisy_path = RESULTS_DIR / f"{model_name}_noisy_test_predictions.parquet"
        std_path = RESULTS_DIR / f"{model_name}_test_predictions.parquet"
        pq_file = noisy_path if noisy_path.exists() else std_path
        if not pq_file.exists():
            pq_file = std_path
        if pq_file.exists():
            df = pd.read_parquet(pq_file)
            available[exp_label] = df
    return available


def compute_weights_from_backtest(
    model_preds: Dict[str, pd.DataFrame],
) -> Dict[str, Dict[str, float]]:
    """v2: 从回测数据计算各模型按资产的权重（前半数据，避免 look-ahead bias）。"""
    weights: Dict[str, Dict[str, float]] = {}

    for label, df in model_preds.items():
        weights[label] = {}
        if label in EXCLUDED_FROM_ENSEMBLE:
            for asset in df["asset"].unique():
                weights[label][asset] = 0.0
            continue

        for asset in df["asset"].unique():
            adf = df[df["asset"] == asset].sort_values("timestamp")
            half = len(adf) // 2
            train_half = adf.iloc[:half]

            if len(train_half) < MIN_TRADES_FOR_WEIGHT:
                weights[label][asset] = 0.5
                continue

            predicted_dir = (train_half["proba_up"] >= 0.5).astype(int)
            correct = (predicted_dir == train_half["actual"]).sum()
            total = len(train_half)
            wr = correct / total
            w = max(0.0, wr - WIN_RATE_FLOOR) * sqrt(total)
            weights[label][asset] = w if w > 0 else 0.0

    return weights


def build_ensemble_predictions(
    model_preds: Dict[str, pd.DataFrame],
    weights: Dict[str, Dict[str, float]],
    consensus_threshold: float = DEFAULT_CONSENSUS,
    consensus_threshold_up: Optional[float] = None,
    consensus_threshold_down: Optional[float] = None,
    confidence_scale: float = CONFIDENCE_SCALE,
    consensus_by_weight: bool = True,
) -> pd.DataFrame:
    """v2: Log-Odds 加权融合。支持按权重算共识(consensus_by_weight)与可调 confidence_scale。"""
    ensemble_labels = [
        l
        for l in model_preds
        if l not in EXCLUDED_FROM_ENSEMBLE and model_preds[l] is not None and not model_preds[l].empty
    ]
    if not ensemble_labels:
        return pd.DataFrame(columns=["timestamp", "asset", "proba_up", "actual", "log_return"])

    dfs = []
    for label in ensemble_labels:
        df = model_preds[label]
        renamed = df[["timestamp", "asset", "proba_up", "actual", "log_return"]].copy()
        renamed = renamed.rename(columns={"proba_up": f"proba_{label}"})
        dfs.append((label, renamed))

    merged = dfs[0][1].copy()
    for label, df in dfs[1:]:
        merged = merged.merge(
            df[["timestamp", "asset", f"proba_{label}"]],
            on=["timestamp", "asset"],
            how="inner",
        )

    if merged.empty:
        return merged[["timestamp", "asset", "actual", "log_return"]].assign(proba_up=pd.Series(dtype=float))[["timestamp", "asset", "proba_up", "actual", "log_return"]]

    n_models = len(ensemble_labels)

    def logit_avg(row):
        asset = row["asset"]
        total_w = 0.0
        weighted_logit_sum = 0.0
        weight_up = 0.0
        n_up_weighted = 0
        n_weighted_total = 0
        for label in ensemble_labels:
            p = row[f"proba_{label}"]
            w = weights.get(label, {}).get(asset, 0.0)
            logit_val = float(_logit(p))
            weighted_logit_sum += w * logit_val
            total_w += w
            if w > 0:
                n_weighted_total += 1
                if p >= 0.5:
                    n_up_weighted += 1
                    weight_up += w
        if total_w > 0:
            avg_p = float(_sigmoid(weighted_logit_sum / total_w))
            direction_up = avg_p >= 0.5
            if consensus_by_weight:
                agree_w = weight_up if direction_up else (total_w - weight_up)
                consensus = agree_w / total_w
            else:
                n_agree = n_up_weighted if direction_up else (n_weighted_total - n_up_weighted)
                consensus = n_agree / n_weighted_total if n_weighted_total > 0 else 0.0
        else:
            avg_p = float(_sigmoid(np.mean([_logit(row[f"proba_{l}"]) for l in ensemble_labels])))
            n_up_all = sum(1 for l in ensemble_labels if row[f"proba_{l}"] >= 0.5)
            consensus = max(n_up_all, n_models - n_up_all) / n_models
        return pd.Series({"proba_up": avg_p, "consensus": consensus})

    ensemble_vals = merged.apply(logit_avg, axis=1)
    if ensemble_vals.empty or "proba_up" not in ensemble_vals.columns:
        return merged[["timestamp", "asset", "actual", "log_return"]].assign(proba_up=pd.Series(dtype=float))[["timestamp", "asset", "proba_up", "actual", "log_return"]]
    merged["proba_up"] = ensemble_vals["proba_up"]
    merged["consensus"] = ensemble_vals["consensus"]

    merged["proba_up"] = (0.5 + (merged["proba_up"] - 0.5) * confidence_scale).clip(0.001, 0.999)

    up_th = float(consensus_threshold_up) if consensus_threshold_up is not None else float(consensus_threshold)
    down_th = float(consensus_threshold_down) if consensus_threshold_down is not None else float(consensus_threshold)
    required_consensus = np.where(merged["proba_up"] >= 0.5, up_th, down_th)
    low_consensus = merged["consensus"] < required_consensus
    merged.loc[low_consensus, "proba_up"] = 0.5

    result = merged[["timestamp", "asset", "proba_up", "actual", "log_return"]].copy()
    return result


def load_params(model_name: str, buy_price: float, rules_version: str = "v3") -> Optional[Dict]:
    """加载模型优化参数。"""
    tag = f"bp{buy_price:.3f}".replace(".", "")
    rules_dir = RESULTS_DIR / model_name
    rules_file = rules_dir / f"optimal_trading_rules_{rules_version}_{tag}.json"
    if not rules_file.exists():
        rules_file = RESULTS_DIR / f"optimal_trading_rules_{rules_version}_{tag}.json"
    if not rules_file.exists():
        return None
    with open(rules_file) as f:
        data = json.load(f)
    return data.get("trading_rules", data.get("params", {}))


def run_backtest(
    buy_price: float = 0.50,
    consensus_threshold: float = DEFAULT_CONSENSUS,
    consensus_threshold_up: Optional[float] = None,
    consensus_threshold_down: Optional[float] = None,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    min_confidence: Optional[float] = None,
):
    """执行完整的单模型 vs 融合对比回测。min_confidence 不为 None 时，融合前仅保留 confidence>=该值的 bar，且 params 强制使用该 min_confidence。"""
    print("=" * 70)
    print("  多模型融合回测对比")
    if consensus_threshold_up is None and consensus_threshold_down is None:
        th_desc = f"{consensus_threshold:.0%}"
    else:
        upv = float(consensus_threshold_up if consensus_threshold_up is not None else consensus_threshold)
        dnv = float(consensus_threshold_down if consensus_threshold_down is not None else consensus_threshold)
        th_desc = f"UP {upv:.0%} / DOWN {dnv:.0%}"
    print(f"  买价: ${buy_price:.2f}  |  共识阈值: {th_desc}  |  初始: ${initial_capital:.0f}" + (f"  |  min_confidence={min_confidence}" if min_confidence is not None else ""))
    print("=" * 70)

    model_preds = load_model_predictions()
    if not model_preds:
        print("❌ 无可用的模型预测 parquet 文件")
        return

    print(f"\n📦 加载了 {len(model_preds)} 个模型:")
    for label, df in model_preds.items():
        dates = f"{df['timestamp'].min().date()} ~ {df['timestamp'].max().date()}"
        print(f"  {label:8s}: {len(df):5d} bars  {dates}")

    # 找重叠时间段（仅按融合有效源计算，避免被已排除模型缩窄窗口）
    window_labels = [l for l in model_preds.keys() if l not in EXCLUDED_FROM_ENSEMBLE]
    date_ranges = [(model_preds[l]["timestamp"].min(), model_preds[l]["timestamp"].max()) for l in window_labels]
    common_start = max(d[0] for d in date_ranges)
    common_end = min(d[1] for d in date_ranges)
    print(f"\n📅 共同时间窗口(融合源): {common_start.date()} ~ {common_end.date()}")

    # 过滤到共同窗口
    filtered = {}
    for label, df in model_preds.items():
        f = df[(df["timestamp"] >= common_start) & (df["timestamp"] <= common_end)].copy()
        filtered[label] = f

    min_confidence_arg = min_confidence
    if min_confidence_arg is not None:
        # 融合前过滤：仅保留 confidence >= min_confidence 的 bar
        min_conf = float(min_confidence_arg)
        for label in list(filtered.keys()):
            df = filtered[label]
            df = df.copy()
            df["_confidence"] = np.maximum(df["proba_up"], 1.0 - df["proba_up"])
            filtered[label] = df[df["_confidence"] >= min_conf].drop(columns=["_confidence"])
        print(f"\n🔒 融合前过滤: 仅保留 confidence >= {min_conf:.2f} 的 bar")
        for label in sorted(filtered.keys()):
            print(f"  {label:8s}: {len(filtered[label])} bars")

    # 计算权重（用前半数据，避免 look-ahead）
    weights = compute_weights_from_backtest(filtered)
    print(f"\n📊 模型权重 (基于前半数据, WR门槛={WIN_RATE_FLOOR:.0%}):")
    for label in sorted(weights.keys()):
        excluded = " ← 排除(过拟合)" if label in EXCLUDED_FROM_ENSEMBLE else ""
        w_str = ", ".join(f"{a}={w:.2f}" for a, w in weights[label].items())
        print(f"  {label:8s}: {w_str}{excluded}")

    # 生成融合预测
    ensemble_df = build_ensemble_predictions(
        filtered,
        weights,
        consensus_threshold=consensus_threshold,
        consensus_threshold_up=consensus_threshold_up,
        consensus_threshold_down=consensus_threshold_down,
    )
    print(f"\n🔀 融合预测: {len(ensemble_df)} bars")
    if ensemble_df.empty:
        print("❌ 融合预测为空：当前阈值/过滤条件下无可交易样本。")
        return

    # 加载 Exp16 的交易参数作为基准
    base_model = "v5_production_tv_365d"  # Exp16
    params = load_params(base_model, buy_price)
    if not params:
        print(f"⚠️ 找不到 {base_model} bp{buy_price} 的参数，使用默认参数")
        params = {
            "min_confidence": 0.50001,
            "min_edge": 0.01638,
            "kelly_frac": 0.95595,
            "bet_pct_normal": 0.06576,
            "bet_pct_conservative": 0.04461,
            "conf_tier1_bound": 0.507,
            "conf_tier2_bound": 0.53,
            "tier1_mult": 0.0881,
            "tier2_mult": 0.8431,
            "tier3_mult": 1.1202,
            "cooldown_bars": 1,
            "drawdown_halt": 0.3471,
        }

    if min_confidence is not None:
        params = dict(params)
        params["min_confidence"] = float(min_confidence)

    # 回测: 各单模型 + 融合
    results = []
    date_from = str(common_start.date())
    date_to = str(common_end.date())

    for label, df in filtered.items():
        if df.empty:
            continue
        df_sorted = df.sort_values("timestamp").reset_index(drop=True)
        res = simulate_v5_trading(df_sorted, params, buy_price, initial_capital)
        results.append({"模型": label, **res})

    # Ensemble
    ens_sorted = ensemble_df.sort_values("timestamp").reset_index(drop=True)
    ens_res = simulate_v5_trading(ens_sorted, params, buy_price, initial_capital)
    results.append({"模型": "ENSEMBLE", **ens_res})

    # 打印对比表
    print(f"\n{'=' * 90}")
    print(f"  回测结果对比  |  买价=${buy_price:.2f}  |  {date_from} ~ {date_to}  |  初始${initial_capital:.0f}")
    print(f"{'=' * 90}")
    header = f"{'模型':>10s} {'最终资金':>10s} {'收益%':>8s} {'胜率%':>7s} {'交易':>5s} {'最大回撤%':>10s} {'Sharpe':>8s}"
    print(header)
    print("-" * 90)

    for r in results:
        name = r["模型"]
        final = r.get("final_capital", initial_capital)
        ret = r.get("return_pct", 0)
        wr = r.get("win_rate", 0)
        trades = r.get("n_trades", 0)
        dd = r.get("max_drawdown", 0)
        sharpe = r.get("sharpe", 0)
        marker = " ★" if name == "ENSEMBLE" else ""
        print(f"{name:>10s} ${final:>9.2f} {ret:>7.1f}% {wr:>6.1f}% {trades:>5d} {dd:>9.1f}% {sharpe:>7.3f}{marker}")

    print(f"{'=' * 90}")

    # Highlight ensemble improvement
    ens_r = [r for r in results if r["模型"] == "ENSEMBLE"][0]
    single_returns = [r["return_pct"] for r in results if r["模型"] != "ENSEMBLE"]
    avg_single = sum(single_returns) / len(single_returns) if single_returns else 0
    best_single = max(single_returns) if single_returns else 0
    ens_ret = ens_r.get("return_pct", 0)

    print(f"\n📈 融合 vs 单模型:")
    print(f"  融合收益:     {ens_ret:+.1f}%")
    print(f"  单模型平均:   {avg_single:+.1f}%")
    print(f"  单模型最佳:   {best_single:+.1f}%")
    print(f"  融合 vs 平均: {ens_ret - avg_single:+.1f}%")
    print(f"  融合 vs 最佳: {ens_ret - best_single:+.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Ensemble 融合预测回测")
    parser.add_argument("--buy-price", type=float, default=0.50, help="买入价格")
    parser.add_argument("--consensus", type=float, default=DEFAULT_CONSENSUS, help="共识阈值")
    parser.add_argument("--consensus-up", type=float, default=None, help="UP 方向共识阈值（可选）")
    parser.add_argument("--consensus-down", type=float, default=None, help="DOWN 方向共识阈值（可选）")
    parser.add_argument("--capital", type=float, default=DEFAULT_INITIAL_CAPITAL, help="初始资金")
    parser.add_argument("--min-confidence", type=float, default=None, help="融合前仅保留 confidence>=该值的 bar，并强制交易层 min_confidence（如 0.70）")
    args = parser.parse_args()

    run_backtest(
        buy_price=args.buy_price,
        consensus_threshold=args.consensus,
        consensus_threshold_up=args.consensus_up,
        consensus_threshold_down=args.consensus_down,
        initial_capital=args.capital,
        min_confidence=args.min_confidence,
    )


if __name__ == "__main__":
    main()
