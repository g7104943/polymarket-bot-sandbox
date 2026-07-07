#!/usr/bin/env python3
"""
公平对比脚本：同一无泄漏数据、同一 train/val/test 切分，对比四种方案。
支持多段滚动：每段 test 严格在 train/val 之后，无未来信息泄漏。
可做多少段无泄漏：需原始数据至少 total_window+(rolling-1)*test_days 天（且每段各资产在该 end_date 前都有足够样本）。
例如 test_days=15、total_window=180，做 5 段需 180+60=240 天数据；做 12 段需 180+165=345 天。

方案:
  基线 / 方案 A (FS70) / 方案 B (stack 全特征) / 方案 A+B (fs70+stack)

输出: baseline_metrics.json, fs70_metrics.json, stack_metrics.json, fs70_stack_metrics.json
  --rolling N 时额外: rolling_summary.json（各方案 N 段均值±标准差）, rolling_segments.json（每段明细）

用法:
  python scripts/run_fair_comparison.py
  python scripts/run_fair_comparison.py --rolling 5 --test-days 15 --optuna-trials 20
  python scripts/run_fair_comparison.py --rolling 3 --skip-stack
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 默认回测参数（与 compare_fair 一致，用于 WR/PnL）
DEFAULT_FAIR_PARAMS = {
    "min_confidence": 0.50,
    "min_edge": 0.02,
    "kelly_frac": 0.33,
    "bet_pct_normal": 0.05,
    "bet_pct_conservative": 0.03,
    "conf_tier1_bound": 0.55,
    "conf_tier2_bound": 0.60,
    "tier1_mult": 0.30,
    "tier2_mult": 0.60,
    "tier3_mult": 1.00,
    "cooldown_bars": 0,
    "drawdown_halt": 0.20,
}


def _run_baseline_and_fs70(
    asset_data: Dict,
    sentiment_groups: List[str],
    paths: Dict[str, str],
    results_dir: Path,
    buy_price: float,
    initial_capital: float,
    n_optuna_trials: int,
    test_days: Optional[int] = None,
) -> tuple[Dict, Dict, Optional[Dict], Optional[Dict], Optional[Dict]]:
    """跑 baseline 与 fs70，返回两者 metrics、baseline 的 full splits（供方案 B stack）、fs70 的 reduced splits（供方案 A+B stack）。"""
    from experiments.sentiment_grid_search.run_grid import (
        evaluate_lightgbm,
    )
    from scripts.compare_fair import simulate_v5_trading

    cap = initial_capital
    params = DEFAULT_FAIR_PARAMS

    # 基线: 无特征选择，并返回 full splits 供 方案 B stack 使用
    print("  [1/4] 基线 (无特征选择)...")
    baseline = evaluate_lightgbm(
        asset_data,
        sentiment_groups,
        paths,
        feature_selection_top_pct=0.0,
        return_test_predictions=True,
        return_pooled_and_splits=True,
        n_optuna_trials=n_optuna_trials,
        test_days=test_days,
    )
    if "error" in baseline:
        print(f"    Baseline 失败: {baseline['error']}")
        return {}, {}, None, None, None

    splits_baseline = None
    if "train_val_df" in baseline and "test_df" in baseline and "feature_cols" in baseline:
        splits_baseline = {
            "train_val_df": baseline["train_val_df"],
            "test_df": baseline["test_df"],
            "feature_cols": baseline["feature_cols"],
        }

    baseline_metrics = {
        "test_auc": baseline.get("test_auc"),
        "test_brier": baseline.get("test_brier"),
        "test_accuracy": baseline.get("test_accuracy"),
        "test_samples": baseline.get("test_samples"),
        "num_features": baseline.get("num_features"),
    }
    test_pred_baseline = baseline.get("test_predictions")
    if test_pred_baseline is not None and len(test_pred_baseline) > 0:
        res = simulate_v5_trading(test_pred_baseline, params, buy_price, cap)
        baseline_metrics["win_rate_pct"] = round(res["win_rate"], 2)
        baseline_metrics["return_pct"] = round(res["return_pct"], 2)
        baseline_metrics["n_trades"] = res["n_trades"]
        baseline_metrics["final_capital"] = round(res["final_capital"], 2)

    # 方案 A: FS70（70% 特征选择），并返回筛选后的 splits 供 方案 A+B 使用
    print("  [2/4] 方案 A: FS70 (LGBM gain 前 70% 特征)...")
    fs70 = evaluate_lightgbm(
        asset_data,
        sentiment_groups,
        paths,
        feature_selection_top_pct=0.7,
        return_test_predictions=True,
        return_pooled_and_splits=True,
        n_optuna_trials=n_optuna_trials,
        test_days=test_days,
    )
    if "error" in fs70:
        print(f"    FS70 失败: {fs70['error']}")
        fs70_metrics = {"error": fs70["error"]}
        test_pred_fs70 = None
    else:
        fs70_metrics = {
            "test_auc": fs70.get("test_auc"),
            "test_brier": fs70.get("test_brier"),
            "test_accuracy": fs70.get("test_accuracy"),
            "test_samples": fs70.get("test_samples"),
            "num_features": fs70.get("num_features"),
        }
        test_pred_fs70 = fs70.get("test_predictions")
        if test_pred_fs70 is not None and len(test_pred_fs70) > 0:
            res = simulate_v5_trading(test_pred_fs70, params, buy_price, cap)
            fs70_metrics["win_rate_pct"] = round(res["win_rate"], 2)
            fs70_metrics["return_pct"] = round(res["return_pct"], 2)
            fs70_metrics["n_trades"] = res["n_trades"]
            fs70_metrics["final_capital"] = round(res["final_capital"], 2)

    # 方案 A+B：fs70 筛选后的 splits（先 70% 特征再 stack）
    splits_fs70 = None
    if "train_val_df" in fs70 and "test_df" in fs70 and "feature_cols" in fs70:
        splits_fs70 = {
            "train_val_df": fs70["train_val_df"],
            "test_df": fs70["test_df"],
            "feature_cols": fs70["feature_cols"],
        }

    return baseline_metrics, fs70_metrics, splits_baseline, splits_fs70, None


def _run_stack(
    splits: Dict[str, Any],
    buy_price: float,
    initial_capital: float,
) -> Dict[str, Any]:
    """用 train_val 训练 LGB+XGB stack，在 test 上评估并回测。"""
    from scripts.compare_fair import simulate_v5_trading
    from scripts.stack_models import train_stack, load_stack, predict_proba_blend
    from sklearn.metrics import roc_auc_score, brier_score_loss

    train_val_df = splits["train_val_df"]
    test_df = splits["test_df"]
    feature_cols = splits["feature_cols"]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        data_path = tmp_path / "train_val.parquet"
        # stack_models 需要 target + 数值特征；train_val_df 含 direction_label, sample_weight
        cols = [c for c in feature_cols if c in train_val_df.columns] + ["direction_label"]
        if "sample_weight" in train_val_df.columns:
            cols.append("sample_weight")
        train_val_df[cols].to_parquet(data_path, index=False)

        train_stack(
            data_path,
            target="direction_label",
            out_dir=tmp_path,
            weight_lgb=0.5,
            weight_xgb=0.5,
            n_estimators=300,
        )
        models, weights, feats = load_stack(tmp_path)
        X_te = test_df[[c for c in feats if c in test_df.columns]].fillna(0)
        proba_te = predict_proba_blend(X_te, models, weights, feats)
    y_te = test_df["direction_label"].values
    auc = float(roc_auc_score(y_te, proba_te))
    brier = float(brier_score_loss(y_te, proba_te))
    acc = float(( (proba_te >= 0.5).astype(int) == y_te ).mean())

    test_predictions = pd.DataFrame({
        "timestamp": test_df["timestamp"].values,
        "asset": test_df["_asset_name"].values,
        "proba_up": proba_te,
        "actual": y_te.astype(int),
    })
    res = simulate_v5_trading(test_predictions, DEFAULT_FAIR_PARAMS, buy_price, initial_capital)
    return {
        "test_auc": round(auc, 5),
        "test_brier": round(brier, 5),
        "test_accuracy": round(acc, 4),
        "test_samples": len(test_df),
        "win_rate_pct": round(res["win_rate"], 2),
        "return_pct": round(res["return_pct"], 2),
        "n_trades": res["n_trades"],
        "final_capital": round(res["final_capital"], 2),
    }


def main():
    parser = argparse.ArgumentParser(description="公平对比: baseline / fs70 / stack，同一数据与切分")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="结果目录，默认 experiments/sentiment_grid_search/results")
    parser.add_argument("--buy-price", type=float, default=0.50, help="回测统一买价")
    parser.add_argument("--initial-capital", type=float, default=400.0, help="初始资金")
    parser.add_argument("--skip-stack", action="store_true", help="跳过 LGB+XGB stacking")
    parser.add_argument("--optuna-trials", type=int, default=50, help="Optuna 试验次数（公平对比可减少以加速）")
    parser.add_argument("--rolling", type=int, default=1,
                        help="多段滚动段数（每段 test 严格在 train/val 之后，无泄漏）。1=单段（默认）")
    parser.add_argument("--test-days", type=int, default=15,
                        help="每段测试集天数。滚动时需数据至少 total_window+(rolling-1)*test_days 天")
    parser.add_argument("--assets", type=str, default=None,
                        help="逗号分隔资产，如 BTC_USDT 或 ETH_USDT。不传则用 run_grid 默认三币。单币时可多段滚动更易成功")
    args = parser.parse_args()

    results_dir = args.results_dir or (PROJECT_ROOT / "experiments" / "sentiment_grid_search" / "results")
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    from experiments.sentiment_grid_search.run_grid import (
        get_data_paths,
        check_data_availability,
        build_asset_data_for_fair,
        get_data_latest_ts,
        ASSETS as DEFAULT_ASSETS,
        DEFAULT_DAYS,
        WARMUP_DAYS,
        VAL_DAYS,
        TEST_DAYS,
        EXPERIMENTS,
    )
    from experiments.gru_regime_v1.src.utils import get_device

    if args.assets:
        assets_list = [a.strip().upper() for a in args.assets.split(",") if a.strip()]
        if not assets_list:
            assets_list = list(DEFAULT_ASSETS)
    else:
        assets_list = list(DEFAULT_ASSETS)
    if assets_list == ["BTC_USDT"]:
        out_suffix = "btc"
    elif assets_list == ["ETH_USDT"]:
        out_suffix = "eth"
    else:
        out_suffix = "all"
    print(f"  资产: {assets_list} → 输出后缀 {out_suffix}")

    paths = get_data_paths()
    avail = check_data_availability(paths)
    exp8 = EXPERIMENTS.get(8, EXPERIMENTS[7])
    sentiment_groups = exp8["lgb_extra_groups"]
    missing = [d for d in exp8["requires_data"] if not avail.get(d, False)]
    if missing:
        print(f"  缺少数据: {missing}，将使用可用特征继续")

    test_days = args.test_days
    total_window = DEFAULT_DAYS + WARMUP_DAYS + VAL_DAYS + test_days
    n_rolling = max(1, int(args.rolling))
    model_dir = PROJECT_ROOT / "experiments" / "sentiment_grid_search" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(use_mps=True)

    if n_rolling == 1:
        # 单段
        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=total_window)
        print("构建 asset_data（与 run_grid Phase 1 一致）...")
        asset_data = build_asset_data_for_fair(paths, assets_list, cutoff_date, model_dir, device)
        if not asset_data:
            print("  asset_data 为空，退出")
            return 1
        print("公平对比: 同一 SEED/TEST_DAYS/VAL_DAYS/PURGE（基线 / 方案A / 方案B / 方案A+B）")
        baseline_metrics, fs70_metrics, splits_baseline, splits_fs70, _ = _run_baseline_and_fs70(
            asset_data, sentiment_groups, paths, results_dir,
            buy_price=args.buy_price, initial_capital=args.initial_capital,
            n_optuna_trials=args.optuna_trials, test_days=test_days,
        )
        _write_single_segment_results(
            results_dir, args.buy_price, baseline_metrics, fs70_metrics,
            splits_baseline, splits_fs70, args.skip_stack, args.initial_capital,
            suffix=out_suffix,
        )
        print("\n公平对比完成（不覆盖现网模型目录）。")
        return 0

    # ─── 多段滚动（每段无泄漏：test 严格在 train/val 之后）───
    ref_end = get_data_latest_ts(paths, assets_list)
    if ref_end is None:
        print("  无法获取数据最新时间戳，退出")
        return 1
    required_days = total_window + (n_rolling - 1) * test_days
    cutoff_min = ref_end - pd.Timedelta(days=required_days)
    print(f"多段滚动: 共 {n_rolling} 段, 每段 test {test_days} 天, 需数据至少 {required_days} 天 (ref_end={ref_end.date()}), 资产={assets_list}")

    segments_baseline = []
    segments_fs70 = []
    segments_stack = []
    segments_fs70_stack = []

    for seg in range(n_rolling):
        end_date_seg = ref_end - pd.Timedelta(days=seg * test_days)
        cutoff_seg = end_date_seg - pd.Timedelta(days=total_window)
        print(f"\n{'='*60}")
        print(f"  段 {seg+1}/{n_rolling}: end_date={end_date_seg.date()}, test=[{end_date_seg.date() - pd.Timedelta(days=test_days)} ~ {end_date_seg.date()}]")
        print("构建 asset_data（本段无未来数据）...")
        try:
            asset_data = build_asset_data_for_fair(paths, assets_list, cutoff_seg, model_dir, device, end_date=end_date_seg)
        except Exception as e:
            print(f"  段 {seg+1} 构建 asset_data 失败: {e}，跳过")
            continue
        if not asset_data or len(asset_data) < len(assets_list):
            print(f"  段 {seg+1} asset_data 不足（需 {len(assets_list)} 资产），跳过")
            continue
        try:
            baseline_m, fs70_m, spl_b, spl_fs70, _ = _run_baseline_and_fs70(
                asset_data, sentiment_groups, paths, results_dir,
                buy_price=args.buy_price, initial_capital=args.initial_capital,
                n_optuna_trials=args.optuna_trials, test_days=test_days,
            )
        except Exception as e:
            print(f"  段 {seg+1} baseline/fs70 失败: {e}，跳过")
            continue
        if "error" in baseline_m or "error" in fs70_m:
            print(f"  段 {seg+1} 评估报错，跳过")
            continue
        segments_baseline.append(baseline_m)
        segments_fs70.append(fs70_m)
        stack_m, fs70_stack_m = None, None
        if not args.skip_stack and spl_b is not None:
            try:
                stack_m = _run_stack(spl_b, args.buy_price, args.initial_capital)
            except Exception:
                pass
        if not args.skip_stack and spl_fs70 is not None:
            try:
                fs70_stack_m = _run_stack(spl_fs70, args.buy_price, args.initial_capital)
            except Exception:
                pass
        segments_stack.append(stack_m)
        segments_fs70_stack.append(fs70_stack_m)

    if not segments_baseline:
        print("\n无有效段（可能数据不足或某段某资产样本过少），请减小 --rolling 或检查数据时间范围）。")
        summary = {"buy_price": args.buy_price, "assets": assets_list, "n_segments_valid": 0, "error": "no_valid_segments"}
        with open(results_dir / f"rolling_summary_{out_suffix}.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        return 0

    # 聚合：均值 ± 标准差（只对有效数值）
    def _agg(key: str, rows: List) -> Dict[str, float]:
        vals = [r.get(key) for r in rows if r and isinstance(r.get(key), (int, float))]
        if not vals:
            return {"mean": 0.0, "std": 0.0, "n": 0}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

    summary = {
        "buy_price": args.buy_price,
        "assets": assets_list,
        "n_segments_requested": n_rolling,
        "n_segments_valid": len(segments_baseline),
        "test_days_per_segment": test_days,
        "total_window_days": total_window,
        "required_data_days": required_days,
        "baseline": {k: _agg(k, segments_baseline) for k in ("test_auc", "test_brier", "win_rate_pct", "return_pct", "n_trades", "final_capital")},
        "fs70": {k: _agg(k, segments_fs70) for k in ("test_auc", "test_brier", "win_rate_pct", "return_pct", "n_trades", "final_capital")},
        "stack": {k: _agg(k, [s for s in segments_stack if s]) for k in ("test_auc", "test_brier", "win_rate_pct", "return_pct", "n_trades", "final_capital")},
        "fs70_stack": {k: _agg(k, [s for s in segments_fs70_stack if s]) for k in ("test_auc", "test_brier", "win_rate_pct", "return_pct", "n_trades", "final_capital")},
    }
    summary_path = results_dir / f"rolling_summary_{out_suffix}.json"
    segments_path = results_dir / f"rolling_segments_{out_suffix}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(segments_path, "w") as f:
        json.dump({
            "segments_baseline": segments_baseline,
            "segments_fs70": segments_fs70,
            "segments_stack": segments_stack,
            "segments_fs70_stack": segments_fs70_stack,
        }, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n已写: {summary_path}, {segments_path}")
    print("\n多段滚动对比完成（每段 test 仅用该段前数据训练，无泄漏）。")
    return 0


def _write_single_segment_results(
    results_dir: Path,
    buy_price: float,
    baseline_metrics: Dict,
    fs70_metrics: Dict,
    splits_baseline: Optional[Dict],
    splits_fs70: Optional[Dict],
    skip_stack: bool,
    initial_capital: float,
    suffix: str = "all",
) -> None:
    suf = f"_{suffix}" if suffix != "all" else ""
    with open(results_dir / f"baseline_metrics{suf}.json", "w") as f:
        json.dump({"buy_price": buy_price, "metrics": baseline_metrics}, f, indent=2, ensure_ascii=False)
    with open(results_dir / f"fs70_metrics{suf}.json", "w") as f:
        json.dump({"buy_price": buy_price, "metrics": fs70_metrics}, f, indent=2, ensure_ascii=False)
    print(f"  已写: baseline_metrics{suf}.json, fs70_metrics{suf}.json")
    if skip_stack:
        return
    if splits_baseline is not None:
        try:
            stack_metrics = _run_stack(splits_baseline, buy_price, initial_capital)
            with open(results_dir / f"stack_metrics{suf}.json", "w") as f:
                json.dump({"buy_price": buy_price, "metrics": stack_metrics}, f, indent=2, ensure_ascii=False)
            print(f"  已写: stack_metrics{suf}.json")
        except Exception as e:
            print(f"  方案 B 失败: {e}")
    if splits_fs70 is not None:
        try:
            fs70_stack_metrics = _run_stack(splits_fs70, buy_price, initial_capital)
            with open(results_dir / f"fs70_stack_metrics{suf}.json", "w") as f:
                json.dump({"buy_price": buy_price, "metrics": fs70_stack_metrics}, f, indent=2, ensure_ascii=False)
            print(f"  已写: fs70_stack_metrics{suf}.json")
        except Exception as e:
            print(f"  方案 A+B 失败: {e}")


if __name__ == "__main__":
    sys.exit(main() or 0)
