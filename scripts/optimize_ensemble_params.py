#!/usr/bin/env python3
"""
融合模型超参优化 — 无泄漏回测 + Optuna 搜索共识阈值、置信度缩放

数据与泄漏:
  - 输入: 各模型 *_test_predictions.parquet（与 backtest_ensemble 相同），
    通常为 run_grid/compare_fair 生成的测试集，与训练集无重叠 → 预测本身无泄漏。
  - 权重: 由「全窗口」的前半段时间计算（compute_weights_from_backtest），
    不使用后半段，避免权重看到“未来”。
  - 超参优化: 默认 --eval-split 0.4 表示用「前 60% 时间」做 Optuna 优化、
    「后 40% 时间」仅做一次最终评估并打印，不在优化中用到 → 超参选择无泄漏。
  - 数据天数: 由各 parquet 共同时间窗口决定，脚本会打印天数（例如 ~16 天）。

可调超参（当前）:
  - CONSENSUS_THRESHOLD（0.50~0.75）
  - CONFIDENCE_SCALE（1.0~1.6）
  - CONSENSUS_BY_WEIGHT（True/False）

融合权重参数（WEIGHT_SOFTPLUS_TEMP、WEIGHT_MIN、WR_BLEND_RECENT 等）一起超参的做法:
  - 在回测中复现 ensemble_prediction_writer 的权重公式；
  - 用 parquet 前半段按 bar 模拟 win/loss（预测方向 vs actual → 1/0），
    再按衰减、近期窗口、Bayesian WR、softplus 等用 trial 的权重参数算权重；
  - 同一 trial 内用该权重 + 该 trial 的阈值/scale 做融合回测，目标仍为 Sharpe 或收益%。
  - 本脚本当前只做「共识+置信度」搜索；权重参数扩展见下方实现要点。

用法:
  python scripts/optimize_ensemble_params.py
  python scripts/optimize_ensemble_params.py --trials 80 --eval-split 0.4
  python scripts/optimize_ensemble_params.py --buy-price 0.50 --no-eval-split  # 不划分，全段优化（有优化泄漏）
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import optuna
import numpy as np
from scripts.backtest_ensemble import (
    load_model_predictions,
    compute_weights_from_backtest,
    build_ensemble_predictions,
    load_params,
    EXCLUDED_FROM_ENSEMBLE,
    DEFAULT_CONSENSUS,
    CONFIDENCE_SCALE,
    RESULTS_DIR,
)
from scripts.compare_fair import simulate_v5_trading, DEFAULT_INITIAL_CAPITAL

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _prepare_data():
    """加载预测、共同窗口、权重（与 backtest_ensemble 一致）。"""
    model_preds = load_model_predictions()
    if not model_preds:
        return None, None, None, None, None

    date_ranges = [(df["timestamp"].min(), df["timestamp"].max()) for df in model_preds.values()]
    common_start = max(d[0] for d in date_ranges)
    common_end = min(d[1] for d in date_ranges)

    filtered = {}
    for label, df in model_preds.items():
        f = df[(df["timestamp"] >= common_start) & (df["timestamp"] <= common_end)].copy()
        filtered[label] = f

    weights = compute_weights_from_backtest(filtered)
    # 用于无泄漏划分：全窗口时间跨度（秒）
    total_span = (common_end - common_start).total_seconds()
    return filtered, weights, common_start, common_end, total_span


def _run_backtest(
    filtered,
    weights,
    consensus_threshold_up: float,
    consensus_threshold_down: float,
    confidence_scale: float,
    consensus_by_weight: bool,
    min_edge_up: float,
    min_edge_down: float,
    buy_price: float,
    initial_capital: float,
    time_end_inclusive=None,
    time_start_exclusive=None,
):
    """构建融合预测并跑回测。
    time_end_inclusive: 只保留 timestamp <= 该时刻（优化期）;
    time_start_exclusive: 只保留 timestamp > 该时刻（评估期）。二者至多指定一个。"""
    ensemble_df = build_ensemble_predictions(
        filtered,
        weights,
        consensus_threshold=consensus_threshold_up,
        consensus_threshold_up=consensus_threshold_up,
        consensus_threshold_down=consensus_threshold_down,
        confidence_scale=confidence_scale,
        consensus_by_weight=consensus_by_weight,
    )
    if time_end_inclusive is not None:
        ensemble_df = ensemble_df[ensemble_df["timestamp"] <= time_end_inclusive].copy()
    if time_start_exclusive is not None:
        ensemble_df = ensemble_df[ensemble_df["timestamp"] > time_start_exclusive].copy()
    if len(ensemble_df) == 0:
        return {"sharpe": 0.0, "return_pct": 0.0, "n_trades": 0}

    if 0 < buy_price < 1:
        odds = (1.0 - buy_price) / buy_price
        p = ensemble_df["proba_up"].clip(1e-6, 1 - 1e-6)
        is_up = p >= 0.5
        p_dir = p.where(is_up, 1.0 - p)
        edge = p_dir * odds - (1.0 - p_dir)
        need = np.where(is_up.to_numpy(), float(min_edge_up), float(min_edge_down))
        mask_low_edge = edge.to_numpy() < need
        if np.any(mask_low_edge):
            ensemble_df.loc[mask_low_edge, "proba_up"] = 0.5

    base_model = "v5_production_tv_365d"
    params = load_params(base_model, buy_price)
    if not params:
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
    params = dict(params)
    # 用分方向 edge 作为主门控，统一 min_edge 取较小值避免二次门控扭曲。
    params["min_edge"] = min(float(min_edge_up), float(min_edge_down))
    ens_sorted = ensemble_df.sort_values("timestamp").reset_index(drop=True)
    res = simulate_v5_trading(ens_sorted, params, buy_price, initial_capital)
    return res


def objective(
    trial: optuna.Trial,
    filtered,
    weights,
    buy_price: float,
    initial_capital: float,
    objective_name: str,
    opt_cut_ts,
):
    """opt_cut_ts: 优化期截止时间（含），只在该时间之前的 bar 上回测，避免超参泄漏。"""
    consensus_threshold_up = trial.suggest_float("consensus_threshold_up", 0.50, 0.80)
    consensus_threshold_down = trial.suggest_float(
        "consensus_threshold_down",
        max(0.55, consensus_threshold_up),
        0.92,
    )
    confidence_scale = trial.suggest_float("confidence_scale", 1.0, 1.6)
    consensus_by_weight = trial.suggest_categorical("consensus_by_weight", [True, False])
    min_edge_up = trial.suggest_float("min_edge_up", 0.008, 0.05)
    min_edge_down = trial.suggest_float("min_edge_down", max(0.012, min_edge_up), 0.08)

    res = _run_backtest(
        filtered,
        weights,
        consensus_threshold_up=consensus_threshold_up,
        consensus_threshold_down=consensus_threshold_down,
        confidence_scale=confidence_scale,
        consensus_by_weight=consensus_by_weight,
        min_edge_up=min_edge_up,
        min_edge_down=min_edge_down,
        buy_price=buy_price,
        initial_capital=initial_capital,
        time_end_inclusive=opt_cut_ts,
    )
    if objective_name == "sharpe":
        sharpe = res.get("sharpe") or 0.0
        return -sharpe
    else:
        return_pct = res.get("return_pct") or 0.0
        return -return_pct


def main():
    parser = argparse.ArgumentParser(description="融合模型超参优化（UP/DOWN 分共识阈值 + 分方向 edge + 置信度缩放）")
    parser.add_argument("--trials", type=int, default=60, help="Optuna 试验次数")
    parser.add_argument("--buy-price", type=float, default=0.50, help="买入价格")
    parser.add_argument("--capital", type=float, default=DEFAULT_INITIAL_CAPITAL, help="初始资金")
    parser.add_argument(
        "--objective",
        choices=["sharpe", "return"],
        default="sharpe",
        help="优化目标: sharpe 或 return（收益%%）",
    )
    parser.add_argument(
        "--eval-split",
        type=float,
        default=0.4,
        help="评估集时间占比 (0~1)，优化只在「前 (1-eval_split)」时段做，最后在「后 eval_split」上报一次指标，默认 0.4 即无泄漏",
    )
    parser.add_argument(
        "--no-eval-split",
        action="store_true",
        help="不在时间上划分，全段用于优化（存在超参泄漏，仅用于对比）",
    )
    parser.add_argument("--seed", type=int, default=42, help="Optuna 随机种子")
    args = parser.parse_args()

    eval_split = 0.0 if args.no_eval_split else max(0.0, min(1.0, args.eval_split))

    print("加载回测数据与权重...")
    prepared = _prepare_data()
    if prepared[0] is None or prepared[1] is None:
        print("❌ 无可用的模型预测 parquet，请先生成各模型的 test_predictions.parquet")
        return 1
    filtered, weights, common_start, common_end, total_span = prepared

    days = total_span / 86400.0
    print(f"  共同窗口: {common_start.date()} ~ {common_end.date()}  ({days:.1f} 天)")
    print(f"  优化目标: {args.objective}  |  买价: ${args.buy_price:.2f}  |  trials: {args.trials}")

    if eval_split > 0:
        opt_cut_ts = common_start + (common_end - common_start) * (1.0 - eval_split)
        print(f"  无泄漏划分: 优化期 ~ {opt_cut_ts.date()}  |  评估期 之后 ~ {common_end.date()}  (eval_split={eval_split:.0%})")
    else:
        opt_cut_ts = None
        print("  ⚠️ 未划分评估期，全段用于优化（超参选择存在泄漏）")
    print()

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=15))
    study.optimize(
        lambda t: objective(t, filtered, weights, args.buy_price, args.capital, args.objective, opt_cut_ts),
        n_trials=args.trials,
        show_progress_bar=True,
    )

    best = study.best_trial
    print("=" * 60)
    print("  融合超参优化结果")
    print("=" * 60)
    print(f"  优化期最佳 {args.objective}: {-best.value:.4f}" if args.objective == "sharpe" else f"  优化期最佳收益%: {-best.value:.2f}%")
    print("  建议参数:")
    for k, v in best.params.items():
        if k == "consensus_by_weight":
            print(f"    CONSENSUS_BY_WEIGHT = {v}  # 按权重算共识")
        elif k == "consensus_threshold_up":
            print(f"    CONSENSUS_THRESHOLD_UP = {v:.2f}  # UP 共识阈值")
        elif k == "consensus_threshold_down":
            print(f"    CONSENSUS_THRESHOLD_DOWN = {v:.2f}  # DOWN 共识阈值")
        elif k == "min_edge_up":
            print(f"    MIN_EDGE_UP = {v:.4f}  # UP 最小 edge")
        elif k == "min_edge_down":
            print(f"    MIN_EDGE_DOWN = {v:.4f}  # DOWN 最小 edge")
        elif k == "confidence_scale":
            print(f"    CONFIDENCE_SCALE = {v:.2f}  # 置信度缩放")
        else:
            print(f"    {k} = {v}")

    if eval_split > 0:
        res_eval = _run_backtest(
            filtered,
            weights,
            consensus_threshold_up=best.params["consensus_threshold_up"],
            consensus_threshold_down=best.params["consensus_threshold_down"],
            confidence_scale=best.params["confidence_scale"],
            consensus_by_weight=best.params["consensus_by_weight"],
            min_edge_up=best.params["min_edge_up"],
            min_edge_down=best.params["min_edge_down"],
            buy_price=args.buy_price,
            initial_capital=args.capital,
            time_start_exclusive=opt_cut_ts,
        )
        print(f"  --- 评估期（未参与优化）---")
        print(f"  评估期 Sharpe: {res_eval.get('sharpe', 0):.4f}  |  收益%: {res_eval.get('return_pct', 0):.2f}%  |  交易数: {res_eval.get('n_trades', 0)}")
    else:
        res_eval = None

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "objective": args.objective,
        "trials": int(args.trials),
        "buy_price": float(args.buy_price),
        "capital": float(args.capital),
        "eval_split": float(eval_split),
        "window_days": float(days),
        "common_start": str(common_start),
        "common_end": str(common_end),
        "best_value": float(best.value),
        "best_params": best.params,
        "eval_metrics": res_eval,
    }
    full_path = reports_dir / f"ensemble_params_tuning_{ts}.json"
    best_path = reports_dir / "ensemble_params_best.json"
    env_path = reports_dir / "ensemble_params_best.env"
    full_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    best_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    env_lines = [
        f"CONSENSUS_THRESHOLD_UP={best.params['consensus_threshold_up']}",
        f"CONSENSUS_THRESHOLD_DOWN={best.params['consensus_threshold_down']}",
        f"MIN_EDGE_UP={best.params['min_edge_up']}",
        f"MIN_EDGE_DOWN={best.params['min_edge_down']}",
        f"CONFIDENCE_SCALE={best.params['confidence_scale']}",
        f"CONSENSUS_BY_WEIGHT={'true' if best.params['consensus_by_weight'] else 'false'}",
        "",
    ]
    env_path.write_text("\n".join(env_lines), encoding="utf-8")
    print(f"  已写入: {full_path}")
    print(f"  已更新: {best_path}")
    print(f"  建议环境变量: {env_path}")

    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
