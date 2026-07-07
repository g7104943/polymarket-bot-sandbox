#!/usr/bin/env python3
"""
融合模型「权重参数 + 阈值」同时超参 — 无泄漏、不需单体模型配合

一、是否需要单体模型配合？
  不需要。权重虽然来自「单体模型的表现」，但回测时我们用的是各模型已经写好的
  *_test_predictions.parquet（每 bar 的 proba_up、actual）。用 parquet 按 bar 模拟
  「该 bar 是否达到交易门槛、若交易则 win/lose」，再按与生产一致的衰减+近期窗口+
  Bayesian WR + softplus 算出权重。因此只需这些 parquet 文件，不需要单体进程再跑或
  再提供任何接口。

二、数据天数与 compare_fair
  - 默认 parquet 来自 backtest_ensemble 同源，共同窗口多为十几天（例如 compare_fair
    的 2026-01-26~02-10）。十几天对「阈值+置信度」尚可，对「权重参数+阈值」同时搜
    容易过拟合，建议尽量加长测试窗口。
  - 做法：用更长测试段生成 parquet 再跑本脚本。例如：
    - compare_fair 加 --date-from / --date-to 拉长测试区间并导出 parquet；
    - 或 walk_forward_validation / run_grid 等生成更长 test_predictions.parquet。
  - 脚本会打印共同窗口天数；若不足 30 天会打提示，仍可继续跑。

三、可超参的融合/权重参数（与 ensemble_prediction_writer 对应）
  融合层:
    - CONSENSUS_THRESHOLD   (建议 0.50~0.75)
    - CONFIDENCE_SCALE      (建议 1.0~1.6)
    - CONSENSUS_BY_WEIGHT   (True/False)
  权重公式:
    - WEIGHT_SOFTPLUS_TEMP  (建议 0.8~2.5)
    - WEIGHT_SOFTPLUS_SCALE (建议 1.0~3.0)
    - WEIGHT_MIN            (建议 0.03~0.10)
    - WEIGHT_SMALL_SAMPLE   (建议 0.2~0.5，<5 笔时用)
  衰减与近期:
    - DECAY_HALFLIFE_HOURS  (建议 6~18)
    - RECENT_WINDOW_HOURS   (建议 12~36)
    - WR_BLEND_RECENT       (建议 0.25~0.45)

用法:
  python scripts/optimize_ensemble_params_full.py --trials 200   # 默认 200 次，单阶段全参数
  python scripts/optimize_ensemble_params_full.py --two-stage    # 推荐：先搜阈值再搜权重，共 240 次
  python scripts/optimize_ensemble_params_full.py --two-stage --stage1-trials 150 --stage2-trials 150
  python scripts/optimize_ensemble_params_full.py --no-weight-params --trials 150   # 只搜阈值/scale
"""
from __future__ import annotations

import argparse
import json
import sys
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from math import log, exp, sqrt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import optuna
from scripts.backtest_ensemble import (
    load_model_predictions,
    build_ensemble_predictions,
    load_params,
    EXCLUDED_FROM_ENSEMBLE,
)
from scripts.compare_fair import simulate_v5_trading, DEFAULT_INITIAL_CAPITAL

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _to_storage_url(storage_arg: str) -> str:
    """将存储参数转为 Optuna storage URL。

    - 若已包含 '://': 视为完整 URL（sqlite/postgresql 等）
    - 否则按本地路径处理并转为 sqlite:///abs/path
    """
    raw = str(storage_arg or "").strip()
    if not raw:
        return ""
    if "://" in raw:
        return raw
    p = Path(raw)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{p}"


def _build_study_base_name(args, common_start, common_end) -> str:
    """构造稳定 study 基名（不含 trials 数，支持后续扩 trial 继续跑）。"""
    payload = {
        "seed": int(args.seed),
        "objective": str(args.objective),
        "buy_price": float(args.buy_price),
        "capital": float(args.capital),
        "eval_split": 0.0 if bool(args.no_eval_split) else float(args.eval_split),
        "no_eval_split": bool(args.no_eval_split),
        "two_stage": bool(args.two_stage),
        "common_start": str(common_start),
        "common_end": str(common_end),
    }
    sig = hashlib.sha1(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return f"{args.study_prefix}_seed{int(args.seed)}_{sig}"


def _fixed_params_sig(fixed: dict) -> str:
    """阶段2 study 名签名：绑定阶段1最优阈值结果。"""
    sig = hashlib.sha1(
        json.dumps(fixed, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return sig[:10]


def _finished_trials_count(study: optuna.study.Study) -> int:
    return sum(1 for t in study.trials if t.state.is_finished())


def _create_study(
    *,
    name: str,
    storage_url: str,
    resume: bool,
    seed: int,
    startup_trials: int,
) -> optuna.study.Study:
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=max(1, int(startup_trials)))
    if storage_url:
        try:
            return optuna.create_study(
                direction="minimize",
                sampler=sampler,
                storage=storage_url,
                study_name=name,
                load_if_exists=bool(resume),
            )
        except optuna.exceptions.DuplicatedStudyError:
            # 非 resume 模式下，同名 study 已存在：自动加时间后缀避免覆盖
            if resume:
                raise
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            alt = f"{name}_{ts}"
            print(f"  ⚠️ study 已存在，自动新建: {alt}")
            return optuna.create_study(
                direction="minimize",
                sampler=sampler,
                storage=storage_url,
                study_name=alt,
                load_if_exists=False,
            )
    return optuna.create_study(direction="minimize", sampler=sampler)


def _compute_weight_from_params(
    wins: float, losses: float, total_trades: float,
    recent_wins: float, recent_losses: float, recent_trades: float,
    weight_min: float, weight_small_sample: float, wr_blend_recent: float,
    weight_softplus_temp: float, weight_softplus_scale: float,
) -> float:
    """与 ensemble_prediction_writer._compute_weight 一致，参数由外部传入（供 trial 使用）。"""
    if total_trades < 0.5:
        return weight_min
    if total_trades < 5:
        return weight_small_sample
    wr = (2.0 + wins) / (4.0 + total_trades)
    if recent_trades >= 5:
        recent_wr = (2.0 + recent_wins) / (4.0 + recent_trades)
        wr = (1.0 - wr_blend_recent) * wr + wr_blend_recent * recent_wr
    wr_score = (wr - 0.5) * 8.0
    confidence = sqrt(min(total_trades, 400))
    x = wr_score * confidence / weight_softplus_temp
    weight = weight_softplus_scale * log(1.0 + exp(x))
    return max(weight_min, weight)


def _simulate_trades_from_parquet(
    model_preds: dict,
    params: dict,
    buy_price: float,
    time_end_inclusive,
) -> dict:
    """从 parquet 前半段（timestamp <= time_end_inclusive）按 bar 模拟每模型每资产的 win/loss 序列。
    返回: {(label, asset): [(ts_epoch, 'win'|'lose'), ...]}"""
    min_conf = params.get("min_confidence", 0.50)
    min_edge = params.get("min_edge", 0.02)
    odds = (1.0 - buy_price) / buy_price
    out = {}
    for label, df in model_preds.items():
        if label in EXCLUDED_FROM_ENSEMBLE:
            continue
        sub = df[df["timestamp"] <= time_end_inclusive].copy() if time_end_inclusive is not None else df.copy()
        for asset in sub["asset"].unique():
            adf = sub[sub["asset"] == asset].sort_values("timestamp")
            if adf.empty:
                out[(label, asset)] = []
                continue
            proba = adf["proba_up"].to_numpy(dtype=float)
            actual = adf["actual"].to_numpy(dtype=int)
            direction_up = proba >= 0.5
            p_dir = np.where(direction_up, proba, 1.0 - proba)
            edge = p_dir * odds - (1.0 - p_dir)
            mask = (p_dir >= min_conf) & (edge >= min_edge)
            if not np.any(mask):
                out[(label, asset)] = []
                continue
            ts_arr = pd.to_datetime(adf["timestamp"]).astype("int64").to_numpy(dtype=np.int64) / 1e9
            dir_int = np.where(direction_up, 1, 0)
            win_mask = dir_int == actual
            result_arr = np.where(win_mask, "win", "lose")
            idx = np.where(mask)[0]
            trades = [(float(ts_arr[i]), str(result_arr[i])) for i in idx]
            out[(label, asset)] = trades
    return out


def _weights_from_simulated_trades(
    trades_by_model_asset: dict,
    weight_cut_ts: float,
    decay_halflife_hours: float,
    decay_min_weight: float,
    recent_window_hours: float,
    weight_min: float,
    weight_small_sample: float,
    wr_blend_recent: float,
    weight_softplus_temp: float,
    weight_softplus_scale: float,
    asset_list: list,
    label_list: list,
) -> dict:
    """用模拟的 (ts, win/lose) 按衰减与近期窗口算权重。weight_cut_ts = 权重数据截止时间（通常=前半段结束）。"""
    decay_lambda = log(2.0) / (decay_halflife_hours * 3600.0)
    recent_cutoff = weight_cut_ts - recent_window_hours * 3600.0
    weights = {}
    for label in label_list:
        weights[label] = {}
        for asset in asset_list:
            trades = trades_by_model_asset.get((label, asset), [])
            total_wins = 0.0
            total_losses = 0.0
            recent_wins = 0
            recent_losses = 0
            for ts, result in trades:
                if ts > weight_cut_ts:
                    continue
                age_s = weight_cut_ts - ts
                decay = max(decay_min_weight, exp(-decay_lambda * age_s))
                if result == "win":
                    total_wins += decay
                else:
                    total_losses += decay
                if recent_cutoff <= ts <= weight_cut_ts:
                    if result == "win":
                        recent_wins += 1
                    else:
                        recent_losses += 1
            total_trades = total_wins + total_losses
            w = _compute_weight_from_params(
                total_wins, total_losses, total_trades,
                float(recent_wins), float(recent_losses), float(recent_wins + recent_losses),
                weight_min, weight_small_sample, wr_blend_recent,
                weight_softplus_temp, weight_softplus_scale,
            )
            weights[label][asset] = w
    return weights


def _prepare_data():
    """与 optimize_ensemble_params 一致。共同窗口只按参与融合的模型计算，排除 EXCLUDED_FROM_ENSEMBLE。"""
    model_preds = load_model_predictions()
    if not model_preds:
        return None, None, None, None, None, None
    preds_for_window = {k: v for k, v in model_preds.items() if k not in EXCLUDED_FROM_ENSEMBLE}
    if not preds_for_window:
        return None, None, None, None, None, None
    date_ranges = [(df["timestamp"].min(), df["timestamp"].max()) for df in preds_for_window.values()]
    common_start = max(d[0] for d in date_ranges)
    common_end = min(d[1] for d in date_ranges)
    filtered = {}
    for label, df in model_preds.items():
        f = df[(df["timestamp"] >= common_start) & (df["timestamp"] <= common_end)].copy()
        filtered[label] = f
    assets = sorted(set().union(*[set(df["asset"].unique()) for df in filtered.values()]))
    labels = [l for l in filtered if l not in EXCLUDED_FROM_ENSEMBLE]
    total_span = (common_end - common_start).total_seconds()
    return filtered, common_start, common_end, total_span, assets, labels


def _run_backtest(
    filtered,
    weights,
    consensus_threshold_up,
    consensus_threshold_down,
    confidence_scale,
    consensus_by_weight,
    min_edge_up,
    min_edge_down,
    buy_price,
    initial_capital,
    forced_min_confidence: float | None = None,
    time_end_inclusive=None,
    time_start_exclusive=None,
):
    """与 optimize_ensemble_params 一致。"""
    ensemble_df = build_ensemble_predictions(
        filtered, weights,
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
        return {"sharpe": 0.0, "return_pct": 0.0, "win_rate": 0.0, "n_trades": 0}

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
            "min_confidence": 0.50001, "min_edge": 0.01638, "kelly_frac": 0.95595,
            "bet_pct_normal": 0.06576, "bet_pct_conservative": 0.04461,
            "conf_tier1_bound": 0.507, "conf_tier2_bound": 0.53,
            "tier1_mult": 0.0881, "tier2_mult": 0.8431, "tier3_mult": 1.1202,
            "cooldown_bars": 1, "drawdown_halt": 0.3471,
        }
    params = dict(params)
    if forced_min_confidence is not None:
        params["min_confidence"] = float(forced_min_confidence)
    params["min_edge"] = min(float(min_edge_up), float(min_edge_down))
    ens_sorted = ensemble_df.sort_values("timestamp").reset_index(drop=True)
    return simulate_v5_trading(ens_sorted, params, buy_price, initial_capital)


def objective(
    trial: optuna.Trial,
    filtered,
    common_start,
    common_end,
    opt_cut_ts,
    weight_cut_ts_epoch,
    assets,
    labels,
    buy_price: float,
    initial_capital: float,
    objective_name: str,
    forced_min_confidence: float | None,
    tune_weight_params: bool,
    trades_by_ma: dict,
    fixed_threshold_params: dict | None = None,
):
    """fixed_threshold_params 非空时：只用其阈值/置信度，只搜权重参数（两阶段第二阶段用）。"""
    if fixed_threshold_params:
        consensus_threshold_up = fixed_threshold_params["consensus_threshold_up"]
        consensus_threshold_down = fixed_threshold_params["consensus_threshold_down"]
        confidence_scale = fixed_threshold_params["confidence_scale"]
        consensus_by_weight = fixed_threshold_params["consensus_by_weight"]
        min_edge_up = fixed_threshold_params["min_edge_up"]
        min_edge_down = fixed_threshold_params["min_edge_down"]
        tune_weight_params = True
    else:
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

    if tune_weight_params:
        decay_halflife_hours = trial.suggest_float("decay_halflife_hours", 6.0, 18.0)
        recent_window_hours = trial.suggest_float("recent_window_hours", 12.0, 36.0)
        wr_blend_recent = trial.suggest_float("wr_blend_recent", 0.25, 0.45)
        weight_softplus_temp = trial.suggest_float("weight_softplus_temp", 0.8, 2.5)
        weight_softplus_scale = trial.suggest_float("weight_softplus_scale", 1.0, 3.0)
        weight_min = trial.suggest_float("weight_min", 0.03, 0.10)
        weight_small_sample = trial.suggest_float("weight_small_sample", 0.2, 0.5)
        decay_min_weight = 0.05
    else:
        decay_halflife_hours = 10.0
        recent_window_hours = 24.0
        wr_blend_recent = 0.35
        weight_softplus_temp = 1.5
        weight_softplus_scale = 2.0
        weight_min = 0.05
        weight_small_sample = 0.30
        decay_min_weight = 0.05

    weights = _weights_from_simulated_trades(
        trades_by_ma, weight_cut_ts_epoch,
        decay_halflife_hours, decay_min_weight, recent_window_hours,
        weight_min, weight_small_sample, wr_blend_recent,
        weight_softplus_temp, weight_softplus_scale,
        assets, labels,
    )
    # labels 已排除 EXCLUDED_FROM_ENSEMBLE，无需再置零
    res = _run_backtest(
        filtered, weights,
        consensus_threshold_up, consensus_threshold_down, confidence_scale, consensus_by_weight,
        min_edge_up, min_edge_down,
        buy_price, initial_capital,
        forced_min_confidence=forced_min_confidence,
        time_end_inclusive=opt_cut_ts,
    )
    def _to_ratio(v: float) -> float:
        # win_rate / max_drawdown 在不同实现里可能是 0~1 或 0~100，统一转为 0~1
        if v > 1.5:
            return v / 100.0
        return v

    if objective_name == "balanced":
        sharpe = float(res.get("sharpe") or 0.0)
        ret_pct = float(res.get("return_pct") or 0.0)
        win_rate = _to_ratio(float(res.get("win_rate") or 0.0))
        max_drawdown = _to_ratio(float(res.get("max_drawdown") or 0.0))
        n_trades = int(res.get("n_trades") or 0)

        # 平衡目标：收益 + 胜率 + 风险调整收益，并对回撤/低样本做惩罚
        sharpe_term = max(-3.0, min(5.0, sharpe))
        ret_term = max(-1.0, min(2.0, ret_pct / 100.0))
        wr_term = win_rate - 0.5
        dd_penalty = max(0.0, max_drawdown)
        trade_penalty = max(0.0, (40 - n_trades) / 40.0)

        balanced_score = (
            1.6 * sharpe_term
            + 2.2 * ret_term
            + 1.2 * wr_term
            - 1.8 * dd_penalty
            - 0.8 * trade_penalty
        )
        return -balanced_score
    if objective_name == "sharpe":
        return -(res.get("sharpe") or 0.0)
    if objective_name == "win_rate":
        return -(res.get("win_rate") or 0.0)  # 最大化胜率%
    return -(res.get("return_pct") or 0.0)


def main():
    parser = argparse.ArgumentParser(description="融合权重+阈值同时超参（无泄漏、不需单体配合）")
    parser.add_argument("--trials", type=int, default=200, help="Optuna 试验次数（单阶段时；两阶段时=stage1+stage2）")
    parser.add_argument("--buy-price", type=float, default=0.50, help="买入价格")
    parser.add_argument("--capital", type=float, default=DEFAULT_INITIAL_CAPITAL, help="初始资金")
    parser.add_argument("--objective", choices=["sharpe", "return", "win_rate", "balanced"], default="balanced",
                        help="优化目标: balanced(平衡) / sharpe(风险调整) / return(收益%%) / win_rate(胜率%%)")
    parser.add_argument("--eval-split", type=float, default=0.4, help="评估集时间占比，默认 0.4 无泄漏")
    parser.add_argument("--no-eval-split", action="store_true", help="全段优化（有泄漏）")
    parser.add_argument("--no-weight-params", action="store_true", help="只搜阈值/scale，不搜权重参数")
    parser.add_argument("--two-stage", action="store_true",
                        help="两阶段：先只搜阈值/置信度，再固定后只搜权重参数，更稳、推荐")
    parser.add_argument("--stage1-trials", type=int, default=120, help="两阶段时阶段1试验数")
    parser.add_argument("--stage2-trials", type=int, default=120, help="两阶段时阶段2试验数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1, help="Optuna 并行 worker 数（默认 1）")
    parser.add_argument(
        "--storage",
        type=str,
        default=str(PROJECT_ROOT / "reports" / "optuna" / "ensemble_full_trials.db"),
        help="Optuna 持久化存储（本地路径或完整 storage URL）",
    )
    parser.add_argument("--study-prefix", type=str, default="ensemble_full", help="study 前缀")
    parser.add_argument("--resume", action="store_true", help="启用 trial 级断点续跑")
    parser.add_argument("--min-confidence", type=float, default=None, help="强制交易层最小置信度（如 0.60~0.70）")
    args = parser.parse_args()

    eval_split = 0.0 if args.no_eval_split else max(0.0, min(1.0, args.eval_split))
    tune_weight_params = not args.no_weight_params
    two_stage = args.two_stage
    stage1_trials = args.stage1_trials
    stage2_trials = args.stage2_trials

    print("加载回测数据...")
    prepared = _prepare_data()
    if prepared[0] is None:
        print("❌ 无可用 parquet，请先生成各模型 test_predictions.parquet")
        return 1
    filtered, common_start, common_end, total_span, assets, labels = prepared
    days = total_span / 86400.0
    print(f"  共同窗口: {common_start.date()} ~ {common_end.date()}  ({days:.1f} 天)")
    if days < 30 and tune_weight_params and not two_stage:
        print("  ⚠️ 数据不足 30 天，权重+阈值同时搜易过拟合，建议拉长测试段或 --no-weight-params 或 --two-stage")
    if two_stage:
        print(f"  优化: 两阶段（先阈值后权重）  |  objective={args.objective}  |  stage1={stage1_trials} + stage2={stage2_trials}")
    else:
        print(f"  优化: {'权重参数+阈值' if tune_weight_params else '仅阈值/置信度'}  |  objective={args.objective}  |  trials={args.trials}")

    opt_cut_ts = None
    if eval_split > 0:
        opt_cut_ts = common_start + (common_end - common_start) * (1.0 - eval_split)
        weight_cut_ts_epoch = opt_cut_ts.timestamp() if hasattr(opt_cut_ts, "timestamp") else pd.Timestamp(opt_cut_ts).timestamp()
        print(f"  无泄漏: 优化期 ~ {opt_cut_ts.date()}  |  评估期 之后 ~ {common_end.date()}")
    else:
        weight_cut_ts_epoch = common_end.timestamp() if hasattr(common_end, "timestamp") else pd.Timestamp(common_end).timestamp()
        print("  ⚠️ 未划分评估期，全段用于优化")

    base_model = "v5_production_tv_365d"
    params_for_trades = load_params(base_model, args.buy_price)
    if not params_for_trades:
        params_for_trades = {"min_confidence": 0.50001, "min_edge": 0.01638}
    if args.min_confidence is not None:
        params_for_trades = dict(params_for_trades)
        params_for_trades["min_confidence"] = float(args.min_confidence)

    print(f"  n_jobs={max(1, int(args.n_jobs))}")
    storage_url = _to_storage_url(args.storage)
    if args.resume:
        if not storage_url:
            print("❌ --resume 需要可用的 --storage")
            return 2
        print(f"  断点续跑: 开启  |  storage={storage_url}")
    elif storage_url:
        print(f"  storage={storage_url}")

    trades_by_ma = _simulate_trades_from_parquet(filtered, params_for_trades, args.buy_price, opt_cut_ts)
    study_base = _build_study_base_name(args, common_start, common_end)
    print(f"  study_base={study_base}")

    if two_stage:
        # 阶段1：只搜阈值/置信度
        print("  [阶段1] 只搜共识阈值(UP/DOWN) / edge阈值(UP/DOWN) / confidence_scale / consensus_by_weight ...")
        stage1_name = f"{study_base}_s1"
        study1 = _create_study(
            name=stage1_name,
            storage_url=storage_url,
            resume=bool(args.resume),
            seed=int(args.seed),
            startup_trials=min(20, int(stage1_trials)),
        )
        done1 = _finished_trials_count(study1)
        remain1 = max(0, int(stage1_trials) - done1)
        print(f"  [阶段1] study={stage1_name}  已完成={done1}  目标={stage1_trials}  剩余={remain1}")
        if remain1 > 0:
            study1.optimize(
                lambda t: objective(
                    t, filtered, common_start, common_end, opt_cut_ts, weight_cut_ts_epoch,
                    assets, labels, args.buy_price, args.capital, args.objective, args.min_confidence,
                    False, trades_by_ma, fixed_threshold_params=None,
                ),
                n_trials=remain1,
                n_jobs=max(1, int(args.n_jobs)),
                show_progress_bar=True,
            )
        done1_after = _finished_trials_count(study1)
        fixed = {
            "consensus_threshold_up": study1.best_trial.params["consensus_threshold_up"],
            "consensus_threshold_down": study1.best_trial.params["consensus_threshold_down"],
            "confidence_scale": study1.best_trial.params["confidence_scale"],
            "consensus_by_weight": study1.best_trial.params["consensus_by_weight"],
            "min_edge_up": study1.best_trial.params["min_edge_up"],
            "min_edge_down": study1.best_trial.params["min_edge_down"],
        }
        print(
            "  [阶段1] 最佳: "
            f"cons_up={fixed['consensus_threshold_up']:.2f} "
            f"cons_down={fixed['consensus_threshold_down']:.2f} "
            f"edge_up={fixed['min_edge_up']:.4f} "
            f"edge_down={fixed['min_edge_down']:.4f} "
            f"scale={fixed['confidence_scale']:.2f} by_weight={fixed['consensus_by_weight']}"
        )
        # 阶段2：固定阈值，只搜权重参数
        print("  [阶段2] 固定阈值，只搜权重参数 ...")
        stage2_name = f"{study_base}_s2_{_fixed_params_sig(fixed)}"
        study2 = _create_study(
            name=stage2_name,
            storage_url=storage_url,
            resume=bool(args.resume),
            seed=int(args.seed) + 1,
            startup_trials=min(20, int(stage2_trials)),
        )
        done2 = _finished_trials_count(study2)
        remain2 = max(0, int(stage2_trials) - done2)
        print(f"  [阶段2] study={stage2_name}  已完成={done2}  目标={stage2_trials}  剩余={remain2}")
        if remain2 > 0:
            study2.optimize(
                lambda t: objective(
                    t, filtered, common_start, common_end, opt_cut_ts, weight_cut_ts_epoch,
                    assets, labels, args.buy_price, args.capital, args.objective, args.min_confidence,
                    True, trades_by_ma, fixed_threshold_params=fixed,
                ),
                n_trials=remain2,
                n_jobs=max(1, int(args.n_jobs)),
                show_progress_bar=True,
            )
        done2_after = _finished_trials_count(study2)
        class _MergedBest:
            params = {**fixed, **study2.best_trial.params}
            value = study2.best_value
        best = _MergedBest()
        stage1_name_out = stage1_name
        stage2_name_out = stage2_name
        single_name_out = ""
        done_single_after = 0
    else:
        single_name = f"{study_base}_single"
        study = _create_study(
            name=single_name,
            storage_url=storage_url,
            resume=bool(args.resume),
            seed=int(args.seed),
            startup_trials=min(20, int(args.trials)),
        )
        done_single = _finished_trials_count(study)
        remain_single = max(0, int(args.trials) - done_single)
        print(f"  [单阶段] study={single_name}  已完成={done_single}  目标={args.trials}  剩余={remain_single}")
        if remain_single > 0:
            study.optimize(
                lambda t: objective(
                    t, filtered, common_start, common_end, opt_cut_ts, weight_cut_ts_epoch,
                    assets, labels, args.buy_price, args.capital, args.objective, args.min_confidence,
                    tune_weight_params, trades_by_ma, fixed_threshold_params=None,
                ),
                n_trials=remain_single,
                n_jobs=max(1, int(args.n_jobs)),
                show_progress_bar=True,
            )
        done_single_after = _finished_trials_count(study)
        best = study.best_trial
        stage1_name_out = ""
        stage2_name_out = ""
        single_name_out = single_name
        done1_after = 0
        done2_after = 0
    print("=" * 60)
    print("  融合超参优化结果（权重+阈值）")
    print("=" * 60)
    if args.objective == "sharpe":
        print(f"  优化期最佳 Sharpe: {-best.value:.4f}")
    elif args.objective == "balanced":
        print(f"  优化期最佳平衡分: {-best.value:.4f}")
    elif args.objective == "win_rate":
        print(f"  优化期最佳胜率%: {-best.value:.1f}")
    else:
        print(f"  优化期最佳收益%: {-best.value:.2f}%")
    print("  建议参数（写入 ensemble_prediction_writer.py）:")
    for k, v in best.params.items():
        if k == "consensus_by_weight":
            print(f"    CONSENSUS_BY_WEIGHT = {v}")
        elif k == "consensus_threshold_up":
            print(f"    CONSENSUS_THRESHOLD_UP = {v:.2f}")
        elif k == "consensus_threshold_down":
            print(f"    CONSENSUS_THRESHOLD_DOWN = {v:.2f}")
        elif k == "min_edge_up":
            print(f"    MIN_EDGE_UP = {v:.4f}")
        elif k == "min_edge_down":
            print(f"    MIN_EDGE_DOWN = {v:.4f}")
        elif k == "confidence_scale":
            print(f"    CONFIDENCE_SCALE = {v:.2f}")
        elif k == "decay_halflife_hours":
            print(f"    DECAY_HALFLIFE_HOURS = {v:.1f}")
        elif k == "recent_window_hours":
            print(f"    RECENT_WINDOW_HOURS = {v:.1f}")
        elif k == "wr_blend_recent":
            print(f"    WR_BLEND_RECENT = {v:.2f}")
        elif k == "weight_softplus_temp":
            print(f"    WEIGHT_SOFTPLUS_TEMP = {v:.2f}")
        elif k == "weight_softplus_scale":
            print(f"    WEIGHT_SOFTPLUS_SCALE = {v:.2f}")
        elif k == "weight_min":
            print(f"    WEIGHT_MIN = {v:.2f}")
        elif k == "weight_small_sample":
            print(f"    WEIGHT_SMALL_SAMPLE = {v:.2f}")
        else:
            print(f"    {k} = {v}")

    if eval_split > 0 and opt_cut_ts is not None:
        w = best.params
        trades = _simulate_trades_from_parquet(filtered, params_for_trades, args.buy_price, opt_cut_ts)
        weight_cut_epoch = opt_cut_ts.timestamp() if hasattr(opt_cut_ts, "timestamp") else pd.Timestamp(opt_cut_ts).timestamp()
        weights_eval = _weights_from_simulated_trades(
            trades, weight_cut_epoch,
            w.get("decay_halflife_hours", 10.0), 0.05,
            w.get("recent_window_hours", 24.0),
            w.get("weight_min", 0.05), w.get("weight_small_sample", 0.30), w.get("wr_blend_recent", 0.35),
            w.get("weight_softplus_temp", 1.5), w.get("weight_softplus_scale", 2.0),
            assets, labels,
        )
        res_eval = _run_backtest(
            filtered, weights_eval,
            best.params["consensus_threshold_up"],
            best.params["consensus_threshold_down"],
            best.params["confidence_scale"],
            best.params["consensus_by_weight"],
            best.params["min_edge_up"],
            best.params["min_edge_down"],
            args.buy_price, args.capital,
            forced_min_confidence=args.min_confidence,
            time_start_exclusive=opt_cut_ts,
        )
        print(f"  --- 评估期（未参与优化）---")
        print(f"  评估期 Sharpe: {res_eval.get('sharpe', 0):.4f}  |  收益%: {res_eval.get('return_pct', 0):.2f}%  |  胜率%: {res_eval.get('win_rate', 0):.1f}  |  交易数: {res_eval.get('n_trades', 0)}")
    else:
        res_eval = None

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "objective": args.objective,
        "seed": int(args.seed),
        "buy_price": float(args.buy_price),
        "capital": float(args.capital),
        "min_confidence_override": (None if args.min_confidence is None else float(args.min_confidence)),
        "eval_split": float(eval_split),
        "resume_enabled": bool(args.resume),
        "storage": storage_url,
        "study_prefix": str(args.study_prefix),
        "window_days": float(days),
        "common_start": str(common_start),
        "common_end": str(common_end),
        "two_stage": bool(two_stage),
        "trials": int(args.trials),
        "stage1_trials": int(stage1_trials),
        "stage2_trials": int(stage2_trials),
        "study_name_stage1": stage1_name_out,
        "study_name_stage2": stage2_name_out,
        "study_name_single": single_name_out,
        "finished_trials_stage1": int(done1_after),
        "finished_trials_stage2": int(done2_after),
        "finished_trials_single": int(done_single_after),
        "best_value": float(best.value),
        "best_params": best.params,
        "eval_metrics": res_eval,
    }
    full_path = reports_dir / f"ensemble_params_full_tuning_{ts}.json"
    best_path = reports_dir / "ensemble_params_full_best.json"
    env_path = reports_dir / "ensemble_params_full_best.env"
    full_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    best_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    env_lines = [
        f"CONSENSUS_THRESHOLD_UP={best.params['consensus_threshold_up']}",
        f"CONSENSUS_THRESHOLD_DOWN={best.params['consensus_threshold_down']}",
        f"MIN_EDGE_UP={best.params['min_edge_up']}",
        f"MIN_EDGE_DOWN={best.params['min_edge_down']}",
        f"CONFIDENCE_SCALE={best.params['confidence_scale']}",
        f"CONSENSUS_BY_WEIGHT={'true' if best.params['consensus_by_weight'] else 'false'}",
    ]
    for key in (
        "decay_halflife_hours",
        "recent_window_hours",
        "wr_blend_recent",
        "weight_softplus_temp",
        "weight_softplus_scale",
        "weight_min",
        "weight_small_sample",
    ):
        if key in best.params:
            env_lines.append(f"{key.upper()}={best.params[key]}")
    env_lines.append("")
    env_path.write_text("\n".join(env_lines), encoding="utf-8")
    print(f"  已写入: {full_path}")
    print(f"  已更新: {best_path}")
    print(f"  建议环境变量: {env_path}")

    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
