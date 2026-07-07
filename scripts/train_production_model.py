#!/usr/bin/env python3
"""
训练 v5 生产模型 — Pooled Exp7 配置 + Optuna 最优超参

流程:
  1. 复用 v5 已训练的 GRU 模型（experiments/sentiment_grid_search/models/）
  2. 构建全量特征（Exp7: Tech+GRU+FGI+CFGI+News+OB+Funding+OI+LS+Polymarket）
  3. 合池 3 币（BTC/ETH/XRP）
  4. 用 Exp7 Optuna 最优超参训练 3 个 Expanding Window LightGBM（60d/90d/120d）
  5. 保存模型 + 特征列表 + 配置到 data/models/v5_production/

用法:
  python scripts/train_production_model.py
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "gru_regime_v1"))

import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss

from experiments.sentiment_grid_search.run_grid import (
    ASSETS, SEED, WINDOW_DAYS_LIST,
    DEFAULT_DAYS, WARMUP_DAYS, TEST_DAYS, VAL_DAYS, PURGE_BARS,
    GRU_HPARAMS, GRU_FEATURE_COLS,
    build_tech_features, extract_embeddings,
    merge_sentiment_to_tech, _prepare_asset_for_pooling,
    _deduplicate_features, get_data_paths, check_data_availability,
)
from experiments.sentiment_grid_search.data_prep import (
    FEATURE_GROUPS, _ensure_datetime_col, load_ohlcv, merge_funding_rate,
)
from experiments.gru_regime_v1.src.utils import get_device
from src.python.feature_engineering import add_multi_timeframe_features

# ─── Exp7 Optuna 最优超参（v5 500 trials 结果）──────────────
EXP7_TUNED_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "n_jobs": 4,
    "verbosity": -1,
    "num_leaves": 212,
    "max_depth": 9,
    "learning_rate": 0.00606389,
    "min_child_samples": 25,
    "feature_fraction": 0.56639651,
    "bagging_fraction": 0.76211669,
    "bagging_freq": 2,
    "lambda_l1": 0.00087003,
    "lambda_l2": 3.64924272,
    "min_split_gain": 0.20273105,
    "path_smooth": 8.31359266,
}

# ─── 正则化参数（P2 改进：大幅简化模型，防止 BTC 过拟合）───────
# 理由:
#   - num_leaves: 212→63, 对 ~34K 样本够用但不会过拟合
#   - max_depth: 9→6, 限制交互阶数
#   - learning_rate: 0.006→0.02, 减少 boosting 轮数
#   - min_child_samples: 25→50, 防止在少量样本上学噪声
#   - feature_fraction: 0.566→0.4, 增加 ensemble diversity
#   - bagging_fraction: 0.762→0.6, 更强的随机采样
#   - lambda_l1: 0.001→0.1, 稀疏正则化
#   - lambda_l2: 3.65→10.0, Ridge 正则化
#   - min_split_gain: 0.2→0.5, 更严格的分裂要求
REGULARIZED_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "n_jobs": 4,
    "verbosity": -1,
    "num_leaves": 63,
    "max_depth": 6,
    "learning_rate": 0.02,
    "min_child_samples": 50,
    "feature_fraction": 0.4,
    "bagging_fraction": 0.6,
    "bagging_freq": 2,
    "lambda_l1": 0.1,
    "lambda_l2": 10.0,
    "min_split_gain": 0.5,
    "path_smooth": 10.0,
}

EXP7_GROUPS = ["fgi_daily", "cfgi", "news", "ob", "funding", "oi", "lsratio", "polymarket_prob"]

# Exp8: Exp7 + 目标市场 Polymarket 概率特征
# 先用 Exp7 超参（后续可通过 run_grid.py 用 Exp8 跑 Optuna 获取新超参）
EXP8_GROUPS = [
    "fgi_daily", "cfgi", "news", "ob", "funding", "oi", "lsratio",
    "polymarket_prob", "polymarket_prob_target",
]

# Exp13/14: Exp8 基础 + TradingView 宏观特征 (去掉 cfgi/news: 用户指定不需要)
EXP13_GROUPS = [
    "fgi_daily", "ob", "funding", "oi", "lsratio",
    "polymarket_prob", "polymarket_prob_target",
    "tv_macro",
]

# Exp15: 365天训练窗口 (基于 Exp8，去掉 cfgi/news 以确保数据覆盖)
EXP15_GROUPS = [
    "fgi_daily", "ob", "funding", "oi", "lsratio",
    "polymarket_prob", "polymarket_prob_target",
]

# Exp16: 365天训练窗口 + TV宏观特征 (基于 Exp13，去掉 cfgi/news)
EXP16_GROUPS = EXP13_GROUPS  # 与 Exp13 相同: fgi_daily + ob + funding + oi + lsratio + pm_prob + pm_target + tv_macro

# Exp17: 去掉 polymarket_prob_target（数据泄漏风险），保留 TV 宏观特征
# 基于 Exp13 但移除 target PM 特征，仅保留 current bar 的 polymarket_prob
EXP17_GROUPS = [
    "fgi_daily", "funding", "oi", "lsratio",
    "polymarket_prob",
    "tv_macro",
]

# Exp15/16 使用更大的 Expanding Window 集成
WINDOW_DAYS_LIST_365 = [120, 240, 365]

# Exp17 使用最大可用数据窗口 (~718d, 受 tv_macro 数据限制)
WINDOW_DAYS_LIST_MAX = [120, 365, 718]

# 默认使用 Exp8（含目标市场 PM 概率）
ACTIVE_GROUPS = EXP8_GROUPS
ACTIVE_EXPERIMENT_NAME = "Exp8 (全特征+目标市场PM)"

OUTPUT_DIR = PROJECT_ROOT / "data" / "models" / "v5_production"

# 新实验目录政策：新实验（特征选择、stacking、新超参等）一律使用新目录/新输出，
# 不覆盖 v5_production 及现有 Exp 的 model-dir，以免影响正在跑的模拟交易。
# 各 Exp 通过 --exp 自动得到不同 OUTPUT_DIR（见 main 内分支）；自定义时请用 --output-dir 指定新路径。

# ─── 模拟K线噪声参数（来自 diagnose_prediction_alignment.py 24h 实测）────
# 模拟 close vs 实际 close 偏差: mean≈0%, σ≈0.13%
SIM_NOISE_STD = 0.0013  # 对数空间标准差
TRADE_OBJECTIVE_EXTREME_9D_UP = 0.10
TRADE_OBJECTIVE_EXTREME_9D_DOWN = -0.10
TRADE_OBJECTIVE_EXTREME_1H_UP = 0.0045
TRADE_OBJECTIVE_EXTREME_1H_DOWN = -0.0045
TRADE_OBJECTIVE_EXTREME_4H_UP = 0.0100
TRADE_OBJECTIVE_EXTREME_4H_DOWN = -0.0100
TRADE_OBJECTIVE_RALLY_4H_UP = 0.0065
TRADE_OBJECTIVE_RALLY_4H_DOWN = -0.0065
TRADE_OBJECTIVE_TREND_BONUS = 1.35
TRADE_OBJECTIVE_RALLY_BONUS = 1.18
TRADE_OBJECTIVE_COUNTERTREND_PENALTY = 0.55
TRADE_OBJECTIVE_RALLY_COUNTERTREND_PENALTY = 0.78
TRADE_OBJECTIVE_HIGH_VOL_FOLLOW_BONUS = 1.08
TRADE_OBJECTIVE_HIGH_VOL_COUNTERTREND_PENALTY = 0.92
CORE10_DIRECTION_WATCH_PATH = PROJECT_ROOT / "reports" / "core10_direction_bucket_watch_latest.json"
CORE10_UNCERTAINTY_GATE_PATH = PROJECT_ROOT / "reports" / "core10_uncertainty_gate_latest.json"


def apply_simulated_candle_noise(tech_df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """在训练数据的 close/high/low/volume 上添加随机噪声, 模拟 T-120s 不完整K线。

    实测数据 (24h, 288条):
      模拟close vs 实际close 偏差均值:     +0.0000%
      模拟close vs 实际close 偏差标准差:   0.1297%
      模拟close vs 实际close 平均绝对偏差: 0.0966%

    方法: 在 log(close) 上加 N(0, σ=0.0013) 噪声, 模拟收盘前2分钟价格微移。
    high/low 也适当添加噪声 (模拟K线不完整时缺少尾部极值)。
    volume 乘以 ~0.87 (模拟只有 13/15 分钟的成交量)。
    """
    rng = np.random.RandomState(seed)
    df = tech_df.copy()
    n = len(df)

    # close: 加对数正态噪声
    log_noise = rng.normal(0, SIM_NOISE_STD, n)
    df["close"] = df["close"] * np.exp(log_noise)

    # high: 可能偏低 (缺少最后2分钟的极端高值)
    high_noise = rng.normal(0, SIM_NOISE_STD * 0.5, n)
    df["high"] = np.maximum(df["high"] * np.exp(high_noise), df["close"])

    # low: 可能偏高 (缺少最后2分钟的极端低值)
    low_noise = rng.normal(0, SIM_NOISE_STD * 0.5, n)
    df["low"] = np.minimum(df["low"] * np.exp(low_noise), df["close"])

    # volume: 约为完整K线的 87% (13/15 分钟)
    vol_factor = rng.uniform(0.83, 0.93, n)
    df["volume"] = df["volume"] * vol_factor

    return df


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _load_direction_watch_bias(symbol: str) -> Dict[str, float]:
    payload = _load_json(CORE10_DIRECTION_WATCH_PATH)
    cells = payload.get("cells") if isinstance(payload.get("cells"), list) else []
    penalty: Dict[str, float] = {"UP": 1.0, "DOWN": 1.0}
    symbol_upper = str(symbol or "").upper().strip()
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        if str(cell.get("symbol") or "").upper().strip() != symbol_upper:
            continue
        recent = cell.get("recent_14d") if isinstance(cell.get("recent_14d"), dict) else {}
        for item in recent.get("bad_bucket_directions") or []:
            if not isinstance(item, dict):
                continue
            bucket_direction = str(item.get("bucket_direction") or "").upper().strip()
            direction = bucket_direction.split(":", 1)[0].strip()
            settled = int(item.get("settled") or 0)
            total_pnl = float(item.get("total_pnl") or 0.0)
            if direction not in penalty:
                continue
            if settled < 8 or total_pnl >= 0:
                continue
            severity = min(0.35, max(0.05, abs(total_pnl) / 600.0))
            penalty[direction] = min(penalty[direction], round(1.0 - severity, 3))
    return penalty


def _load_uncertainty_gate_rules(symbol: str) -> Dict[str, Dict[str, float | str]]:
    payload = _load_json(CORE10_UNCERTAINTY_GATE_PATH)
    symbol_upper = str(symbol or "").upper().strip()
    defaults: Dict[str, Dict[str, float | str]] = {
        "UP": {
            "uncertaintyDispersionMax": 0.18,
            "uncertaintyHighVolDispersionMax": 0.12,
            "uncertaintyChangePointProbMax": 0.86,
            "uncertaintyCountertrendChangePointProbMax": 0.60,
            "uncertaintyDegradedExtraDelta": 0.02,
            "uncertaintyManualAction": "degraded",
        },
        "DOWN": {
            "uncertaintyDispersionMax": 0.15,
            "uncertaintyHighVolDispersionMax": 0.10,
            "uncertaintyChangePointProbMax": 0.80,
            "uncertaintyCountertrendChangePointProbMax": 0.52,
            "uncertaintyDegradedExtraDelta": 0.04,
            "uncertaintyManualAction": "blocked",
        },
    }
    targets = payload.get("targets") if isinstance(payload.get("targets"), list) else []
    for direction in ("UP", "DOWN"):
        rows = [
            row for row in targets
            if isinstance(row, dict)
            and str(row.get("symbol") or "").upper().strip() == symbol_upper
            and str(row.get("direction") or "").upper().strip() == direction
            and isinstance(row.get("rules"), dict)
        ]
        if not rows:
            continue
        agg = dict(defaults[direction])
        agg["uncertaintyDispersionMax"] = min(float((row.get("rules") or {}).get("uncertaintyDispersionMax", agg["uncertaintyDispersionMax"])) for row in rows)
        agg["uncertaintyHighVolDispersionMax"] = min(float((row.get("rules") or {}).get("uncertaintyHighVolDispersionMax", agg["uncertaintyHighVolDispersionMax"])) for row in rows)
        agg["uncertaintyChangePointProbMax"] = min(float((row.get("rules") or {}).get("uncertaintyChangePointProbMax", agg["uncertaintyChangePointProbMax"])) for row in rows)
        agg["uncertaintyCountertrendChangePointProbMax"] = min(float((row.get("rules") or {}).get("uncertaintyCountertrendChangePointProbMax", agg["uncertaintyCountertrendChangePointProbMax"])) for row in rows)
        agg["uncertaintyDegradedExtraDelta"] = max(float((row.get("rules") or {}).get("uncertaintyDegradedExtraDelta", agg["uncertaintyDegradedExtraDelta"])) for row in rows)
        agg["uncertaintyManualAction"] = "blocked" if any(str((row.get("rules") or {}).get("uncertaintyManualAction") or "").lower() == "blocked" for row in rows) else "degraded"
        defaults[direction] = agg
    return defaults


def apply_trade_objective_weighting(
    pooled: pd.DataFrame,
    *,
    assets_by_id: Dict[int, str],
    trend_bonus: float,
    rally_bonus: float,
    countertrend_penalty: float,
    rally_countertrend_penalty: float,
    high_vol_follow_bonus: float,
    high_vol_countertrend_penalty: float,
    uncertainty_aware_weighting: bool,
    uncertainty_penalty_scale: float,
    countertrend_keep_frac: float,
    rally_countertrend_keep_frac: float,
    blocked_keep_frac: float,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    df = pooled.copy()
    summary: Dict[str, Any] = {
        "enabled": True,
        "by_asset": {},
        "rules": {
            "extreme_up": {
                "ret_9d_gte": TRADE_OBJECTIVE_EXTREME_9D_UP,
                "ret_1h_gte": TRADE_OBJECTIVE_EXTREME_1H_UP,
                "ret_4h_gte": TRADE_OBJECTIVE_EXTREME_4H_UP,
            },
            "extreme_down": {
                "ret_9d_lte": TRADE_OBJECTIVE_EXTREME_9D_DOWN,
                "ret_1h_lte": TRADE_OBJECTIVE_EXTREME_1H_DOWN,
                "ret_4h_lte": TRADE_OBJECTIVE_EXTREME_4H_DOWN,
            },
        },
        "params": {
            "trend_bonus": trend_bonus,
            "rally_bonus": rally_bonus,
            "countertrend_penalty": countertrend_penalty,
            "rally_countertrend_penalty": rally_countertrend_penalty,
            "high_vol_follow_bonus": high_vol_follow_bonus,
            "high_vol_countertrend_penalty": high_vol_countertrend_penalty,
            "uncertainty_aware_weighting": bool(uncertainty_aware_weighting),
            "uncertainty_penalty_scale": uncertainty_penalty_scale,
            "countertrend_keep_frac": countertrend_keep_frac,
            "rally_countertrend_keep_frac": rally_countertrend_keep_frac,
            "blocked_keep_frac": blocked_keep_frac,
        },
    }
    if df.empty or "asset_id" not in df.columns or "close" not in df.columns:
        summary["status"] = "skipped_missing_columns"
        return df, summary

    base_weights = df["sample_weight"].astype(float).copy()

    drop_indices: list[int] = []
    for asset_id, asset_name in assets_by_id.items():
        mask = df["asset_id"] == asset_id
        if not bool(mask.any()):
            continue
        asset_df = df.loc[mask].sort_values("timestamp").copy()
        asset_keep = pd.Series(True, index=asset_df.index, dtype=bool)
        close = asset_df["close"].astype(float)
        ret_15m = close.pct_change().fillna(0.0)
        ret_1h = (close / close.shift(4) - 1.0).fillna(0.0)
        ret_4h = (close / close.shift(16) - 1.0).fillna(0.0)
        ret_9d = (close / close.shift(9 * 24 * 4) - 1.0).fillna(0.0)
        rv_1h = ret_15m.rolling(4).std().fillna(0.0)
        rv_4h = ret_15m.rolling(16).std().fillna(0.0)
        rv_percentile = rv_4h.rank(pct=True).fillna(0.5)

        extreme_up = (ret_9d >= TRADE_OBJECTIVE_EXTREME_9D_UP) & (
            (ret_1h >= TRADE_OBJECTIVE_EXTREME_1H_UP) | (ret_4h >= TRADE_OBJECTIVE_EXTREME_4H_UP)
        )
        extreme_down = (ret_9d <= TRADE_OBJECTIVE_EXTREME_9D_DOWN) & (
            (ret_1h <= TRADE_OBJECTIVE_EXTREME_1H_DOWN) | (ret_4h <= TRADE_OBJECTIVE_EXTREME_4H_DOWN)
        )
        rally_up = (~extreme_up) & (~extreme_down) & (
            (ret_4h >= TRADE_OBJECTIVE_RALLY_4H_UP) | ((ret_1h >= TRADE_OBJECTIVE_EXTREME_1H_UP) & (rv_percentile >= 0.68))
        )
        rally_down = (~extreme_up) & (~extreme_down) & (
            (ret_4h <= TRADE_OBJECTIVE_RALLY_4H_DOWN) | ((ret_1h <= TRADE_OBJECTIVE_EXTREME_1H_DOWN) & (rv_percentile >= 0.68))
        )
        high_vol = rv_percentile >= 0.82
        conflict_1h = (np.sign(ret_15m) * np.sign(ret_1h)) < 0
        conflict_4h = (np.sign(ret_15m) * np.sign(ret_4h)) < 0
        trend_flip = (np.sign(ret_1h) * np.sign(ret_4h)) < 0
        burst = (ret_15m.abs() >= (rv_1h.fillna(0.0) * 1.15))
        dispersion_proxy = np.clip(0.55 * rv_percentile + 0.25 * conflict_1h.astype(float) + 0.20 * conflict_4h.astype(float), 0.0, 1.0)
        change_point_proxy = np.clip(0.45 * rv_percentile + 0.30 * trend_flip.astype(float) + 0.25 * burst.astype(float), 0.0, 1.0)

        label_up = asset_df["direction_label"].astype(float) >= 0.5
        label_down = ~label_up
        trend_follow = (extreme_up & label_up) | (extreme_down & label_down)
        countertrend = (extreme_up & label_down) | (extreme_down & label_up)
        rally_follow = (rally_up & label_up) | (rally_down & label_down)
        rally_countertrend = (rally_up & label_down) | (rally_down & label_up)

        direction_penalty = _load_direction_watch_bias(asset_name.replace("_USDT", ""))
        uncertainty_rules = _load_uncertainty_gate_rules(asset_name.replace("_USDT", ""))
        weights = pd.Series(np.ones(len(asset_df)), index=asset_df.index, dtype=float)
        weights.loc[trend_follow] *= trend_bonus
        weights.loc[rally_follow] *= rally_bonus
        weights.loc[countertrend] *= countertrend_penalty
        weights.loc[rally_countertrend] *= rally_countertrend_penalty
        weights.loc[high_vol & (trend_follow | rally_follow)] *= high_vol_follow_bonus
        weights.loc[high_vol & (countertrend | rally_countertrend)] *= high_vol_countertrend_penalty
        weights.loc[label_up] *= float(direction_penalty.get("UP", 1.0))
        weights.loc[label_down] *= float(direction_penalty.get("DOWN", 1.0))
        uncertainty_summary: Dict[str, Dict[str, Any]] = {}
        blocked_union = pd.Series(False, index=asset_df.index, dtype=bool)
        if uncertainty_aware_weighting:
            for direction, label_mask in (("UP", label_up), ("DOWN", label_down)):
                rules = uncertainty_rules.get(direction, {})
                dispersion_cap = float(rules.get("uncertaintyDispersionMax", 0.18))
                high_vol_cap = float(rules.get("uncertaintyHighVolDispersionMax", dispersion_cap))
                cp_cap = float(rules.get("uncertaintyChangePointProbMax", 0.86))
                ct_cp_cap = float(rules.get("uncertaintyCountertrendChangePointProbMax", cp_cap))
                degraded_delta = float(rules.get("uncertaintyDegradedExtraDelta", 0.02))
                manual_action = str(rules.get("uncertaintyManualAction", "degraded")).lower()
                follow_mask = trend_follow if direction == "UP" else trend_follow
                counter_mask = countertrend if direction == "UP" else countertrend
                dispersion_limit = np.where(high_vol, high_vol_cap, dispersion_cap)
                degrade_mask = label_mask & ((dispersion_proxy > dispersion_limit) | (change_point_proxy > cp_cap))
                blocked_mask = label_mask & (
                    (dispersion_proxy > (dispersion_limit + 0.06))
                    | (change_point_proxy > (ct_cp_cap if counter_mask.any() else cp_cap))
                )
                if manual_action == "blocked":
                    blocked_mask = blocked_mask | (label_mask & counter_mask & (change_point_proxy > ct_cp_cap))
                degraded_mult = max(0.30, min(0.95, 1.0 - degraded_delta * 6.0 * uncertainty_penalty_scale))
                blocked_mult = max(0.12, min(0.65, degraded_mult * (0.65 if manual_action == "blocked" else 0.80)))
                weights.loc[degrade_mask] *= degraded_mult
                weights.loc[blocked_mask] *= blocked_mult
                blocked_union = blocked_union | blocked_mask
                low_uncertainty_follow = label_mask & ~degrade_mask & ~blocked_mask & high_vol & follow_mask
                weights.loc[low_uncertainty_follow] *= min(1.15, 1.0 + 0.04 * uncertainty_penalty_scale)
                uncertainty_summary[direction] = {
                    "degrade_rows": int(degrade_mask.sum()),
                    "blocked_rows": int(blocked_mask.sum()),
                    "low_uncertainty_follow_rows": int(low_uncertainty_follow.sum()),
                    "degraded_mult": round(degraded_mult, 4),
                    "blocked_mult": round(blocked_mult, 4),
                    "dispersion_cap": round(dispersion_cap, 4),
                    "high_vol_dispersion_cap": round(high_vol_cap, 4),
                    "change_point_cap": round(cp_cap, 4),
                    "countertrend_change_point_cap": round(ct_cp_cap, 4),
                    "manual_action": manual_action,
                }
        weights = weights.clip(lower=0.15, upper=3.0)

        df.loc[asset_df.index, "sample_weight"] = df.loc[asset_df.index, "sample_weight"].astype(float) * weights
        rng = np.random.RandomState(SEED + int(asset_id) * 101 + 7)
        pruning_summary: Dict[str, Any] = {
            "countertrend_pruned_rows": 0,
            "rally_countertrend_pruned_rows": 0,
            "blocked_pruned_rows": 0,
        }

        def _apply_keep_frac(mask_series: pd.Series, keep_frac: float, summary_key: str) -> None:
            nonlocal asset_keep
            keep_frac = float(max(0.0, min(1.0, keep_frac)))
            if keep_frac >= 1.0:
                return
            eligible = mask_series & asset_keep
            idx = list(asset_df.index[eligible])
            if not idx:
                return
            if keep_frac <= 0.0:
                drop_idx = idx
            else:
                keep_draw = rng.rand(len(idx)) < keep_frac
                drop_idx = [row_idx for row_idx, keep_flag in zip(idx, keep_draw) if not keep_flag]
            if not drop_idx:
                return
            asset_keep.loc[drop_idx] = False
            pruning_summary[summary_key] = int(len(drop_idx))

        _apply_keep_frac(countertrend, countertrend_keep_frac, "countertrend_pruned_rows")
        _apply_keep_frac(rally_countertrend, rally_countertrend_keep_frac, "rally_countertrend_pruned_rows")
        _apply_keep_frac(blocked_union, blocked_keep_frac, "blocked_pruned_rows")
        pruned_idx = list(asset_keep.index[~asset_keep])
        if pruned_idx:
            drop_indices.extend(pruned_idx)
        summary["by_asset"][asset_name] = {
            "rows": int(len(asset_df)),
            "avg_weight_mult": round(float(weights.mean()), 4),
            "trend_follow_rows": int(trend_follow.sum()),
            "countertrend_rows": int(countertrend.sum()),
            "rally_follow_rows": int(rally_follow.sum()),
            "rally_countertrend_rows": int(rally_countertrend.sum()),
            "high_vol_rows": int(high_vol.sum()),
            "direction_penalty": direction_penalty,
            "uncertainty_rules": uncertainty_rules,
            "uncertainty_summary": uncertainty_summary,
            "pruning_summary": pruning_summary,
        }

    if drop_indices:
        df = df.drop(index=sorted(set(drop_indices))).reset_index(drop=True)
    final_weights = df["sample_weight"].astype(float)
    summary["global"] = {
        "mean_before": round(float(base_weights.mean()), 6),
        "mean_after": round(float(final_weights.mean()), 6),
        "median_before": round(float(base_weights.median()), 6),
        "median_after": round(float(final_weights.median()), 6),
        "max_after": round(float(final_weights.max()), 6),
        "min_after": round(float(final_weights.min()), 6),
        "dropped_rows": int(len(set(drop_indices))),
    }
    return df, summary


def _build_empty_embeddings(tech_df: pd.DataFrame) -> pd.DataFrame:
    """为 no-GRU 训练路径提供空嵌入占位表。"""
    if "timestamp_ms" in tech_df.columns:
        ts_col = tech_df["timestamp_ms"]
    elif "timestamp" in tech_df.columns:
        ts_col = tech_df["timestamp"]
    else:
        ts_col = pd.Series(range(len(tech_df)))
    return pd.DataFrame({"timestamp_ms": ts_col}).reset_index(drop=True)


def _load_feature_allowlist(path_str: str) -> list[str]:
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"feature_allowlist_not_found:{path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("--feature-allowlist-file JSON 必须是数组")
        return [str(item).strip() for item in payload if str(item).strip()]
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and (not line.strip().startswith("#"))
    ]


def _write_table(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        path.write_text(df.to_json(orient="records", force_ascii=False, indent=2), encoding="utf-8")


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))


def main():
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="训练 v5 生产模型")
    _parser.add_argument("--cell-id", type=str, default="",
                         help="Core10 cell 级重训入口，如 default/v5_exp13_bp0530/ETH。传入后会切到新的 cell 级树模型训练管道。")
    _parser.add_argument("--profile", type=str, default="",
                         help="Core10 cell/profile-aware 训练时的 profile 标识。")
    _parser.add_argument("--state-conditional", action="store_true",
                         help="启用状态条件训练头。与 --cell-id 一起使用时，会调用新的 Core10 tree pipeline。")
    _parser.add_argument("--abstain-head", action="store_true",
                         help="启用 abstain 头。与 --cell-id 一起使用时，会调用新的 Core10 tree pipeline。")
    _parser.add_argument("--trade-utility-target", action="store_true",
                         help="启用 trade utility 目标。与 --cell-id 一起使用时，会调用新的 Core10 tree pipeline。")
    _parser.add_argument("--exp", type=int, choices=[7, 8, 13, 14, 15, 16, 17], default=None,
                         help="实验版本: 7=Exp7(无target PM), 8=Exp8(含target PM), "
                              "13=Exp13(Exp8+TV宏观,T+0), 14=Exp14(Exp8+TV宏观,T-120s), "
                              "15=Exp15(Exp8基础,365d训练), 16=Exp16(Exp13+TV宏观,365d训练), "
                              "17=Exp17(Exp13去target PM,保留TV宏观). "
                              "不指定则用默认 Exp8")
    _parser.add_argument("--output-dir", type=str, default=None,
                         help="自定义输出目录。不指定则用默认 v5_production")
    _parser.add_argument("--extra-group", action="append", default=[],
                         help="额外启用的特征组（可重复传入），如 --extra-group wallet_cohort")
    _parser.add_argument("--feature-groups", type=str, default="",
                         help="显式覆盖特征组，逗号分隔，如 fgi_daily,ob,funding,oi,lsratio。"
                              "用于单体专项重训时按覆盖率剔除短历史特征。")
    _parser.add_argument("--traditional-only", action="store_true",
                         help="只保留传统主特征层：OHLCV 技术指标 + 多时间框架 + GRU embedding。"
                              "会清空所有 funding/oi/news/orderbook/polymarket 等外挂特征组。")
    _parser.add_argument("--feature-allowlist-file", type=str, default="",
                         help="列级特征白名单文件（txt/json）。训练前按该白名单裁剪特征；asset_id 会自动保留。")
    _parser.add_argument("--feature-stats-out", type=str, default="",
                         help="可选：导出训练使用特征统计（csv/parquet/json）。")
    _parser.add_argument("--feature-importance-out", type=str, default="",
                         help="可选：导出 LightGBM 特征重要性聚合结果（csv/parquet/json）。")
    _parser.add_argument("--pooled-features-out", type=str, default="",
                         help="可选：导出本轮 pooled 特征表（csv/parquet/json），供长窗减法与相关性裁剪使用。")
    _parser.add_argument("--pooled-features-in", type=str, default="",
                         help="可选：直接复用既有 pooled 特征表（csv/parquet/json），跳过 Phase1 特征构建与 GRU 提取。")
    _parser.add_argument("--sim-noise", action="store_true",
                         help="启用模拟K线噪声增强: 在训练数据的 close/high/low 上添加随机噪声, "
                              "模拟 T-120s 时刻不完整K线与实际收盘的偏差 (实测 σ ≈ 0.13%%)")
    _parser.add_argument("--regularized", action="store_true",
                         help="使用正则化 LightGBM 参数 (num_leaves=63, max_depth=6, 更强 L1/L2). "
                              "适合对抗 BTC 过拟合，牺牲一定训练 AUC 换取更好的泛化能力。")
    _parser.add_argument("--btc-weight-boost", type=float, default=1.0,
                         help="BTC 样本权重提升倍数 (默认 1.0 = 不提升). "
                              "设为 1.5~2.0 让模型更重视 BTC 样本，改善 BTC 方向预测。"
                              "注意: asset_id=0 对应 BTC (sorted order: BTC/ETH/XRP)")
    _parser.add_argument("--assets", type=str, default=None,
                         help="逗号分隔资产列表，只训练这些资产，如 BTC_USDT 或 ETH_USDT。"
                              "用于 Core10 专属训练分支，避免继续用全资产合池共享模型。")
    _parser.add_argument("--train-days", type=int, default=0,
                         help="显式覆盖训练窗口天数。0 表示按实验默认值。")
    _parser.add_argument("--window-days-list", type=str, default="",
                         help="显式覆盖集成窗口，如 180,240 或 240,365。空表示按实验默认值。")
    _parser.add_argument("--disable-gru-embeddings", action="store_true",
                         help="训练时不提取 GRU 嵌入，仅使用技术/情绪/目标市场特征。")
    _parser.add_argument("--lgb-params-json", type=str, default="",
                         help="可选：LightGBM 参数覆盖 JSON 文件路径。用于 Core10 promising 变体的模型超参搜索。")
    _parser.add_argument("--lgb-params-name", type=str, default="",
                         help="可选：覆盖后的参数集名称，便于 config/compare/report 标识。")
    _parser.add_argument("--n-estimators-cap", type=int, default=1000,
                         help="LightGBM 最大树数上限。默认 1000。单体专项深搜会显式搜索这个参数。")
    _parser.add_argument("--early-stopping-rounds", type=int, default=50,
                         help="LightGBM 早停轮数。默认 50。单体专项深搜会显式搜索这个参数。")
    _parser.add_argument("--trade-objective-weighting", action="store_true",
                         help="启用交易目标导向的训练样本加权。会提高顺势 extreme/rally 样本权重，降低当前 Core10 已证伪的逆势极端样本权重。")
    _parser.add_argument("--trade-objective-trend-bonus", type=float, default=TRADE_OBJECTIVE_TREND_BONUS)
    _parser.add_argument("--trade-objective-rally-bonus", type=float, default=TRADE_OBJECTIVE_RALLY_BONUS)
    _parser.add_argument("--trade-objective-countertrend-penalty", type=float, default=TRADE_OBJECTIVE_COUNTERTREND_PENALTY)
    _parser.add_argument("--trade-objective-rally-countertrend-penalty", type=float, default=TRADE_OBJECTIVE_RALLY_COUNTERTREND_PENALTY)
    _parser.add_argument("--trade-objective-high-vol-follow-bonus", type=float, default=TRADE_OBJECTIVE_HIGH_VOL_FOLLOW_BONUS)
    _parser.add_argument("--trade-objective-high-vol-countertrend-penalty", type=float, default=TRADE_OBJECTIVE_HIGH_VOL_COUNTERTREND_PENALTY)
    _parser.add_argument("--uncertainty-aware-weighting", action="store_true",
                         help="启用训练期不确定性感知加权。会参考当前 Core10 uncertainty gate，把高波动/高切换/高不确定样本进一步降权。")
    _parser.add_argument("--uncertainty-penalty-scale", type=float, default=1.0,
                         help="训练期不确定性降权强度，默认 1.0。>1 更保守，<1 更宽松。")
    _parser.add_argument("--trade-objective-countertrend-keep-frac", type=float, default=1.0,
                         help="训练期保留的极端逆势样本比例。默认 1.0=不裁剪，<1 会下采样最差逆势样本。")
    _parser.add_argument("--trade-objective-rally-countertrend-keep-frac", type=float, default=1.0,
                         help="训练期保留的 rally 逆势样本比例。默认 1.0=不裁剪。")
    _parser.add_argument("--trade-objective-blocked-keep-frac", type=float, default=1.0,
                         help="训练期保留的 uncertainty blocked 样本比例。默认 1.0=不裁剪。")
    _args = _parser.parse_args()

    if str(_args.cell_id or "").strip():
        tree_script = PROJECT_ROOT / "scripts" / "train_core10_tree_model.py"
        cmd = [
            sys.executable,
            str(tree_script),
            "--cell-id",
            str(_args.cell_id).strip(),
            "--windows",
            str(_args.window_days_list).strip() if str(_args.window_days_list).strip() else "180,365,540",
        ]
        if _args.profile:
            cmd.extend(["--profile", str(_args.profile).strip()])
        if _args.state_conditional:
            cmd.append("--state-conditional")
        if _args.abstain_head:
            cmd.append("--abstain-head")
        if _args.trade_utility_target:
            cmd.append("--trade-utility-target")
        proc = subprocess.run(cmd)
        raise SystemExit(proc.returncode)

    global ACTIVE_GROUPS, ACTIVE_EXPERIMENT_NAME, OUTPUT_DIR
    if _args.exp == 7:
        ACTIVE_GROUPS = EXP7_GROUPS
        ACTIVE_EXPERIMENT_NAME = "Exp7 (无target PM特征)"
    elif _args.exp == 8:
        ACTIVE_GROUPS = EXP8_GROUPS
        ACTIVE_EXPERIMENT_NAME = "Exp8 (全特征+目标市场PM)"
    elif _args.exp in (13, 14):
        ACTIVE_GROUPS = EXP13_GROUPS
        ACTIVE_EXPERIMENT_NAME = f"Exp{_args.exp} (Exp8+TV宏观特征)"
        if _args.exp == 13 and not _args.output_dir:
            OUTPUT_DIR = PROJECT_ROOT / "data" / "models" / "v5_production_tv"
        elif _args.exp == 14 and not _args.output_dir:
            OUTPUT_DIR = PROJECT_ROOT / "data" / "models" / "v5_production_sim_noise_tv"
    elif _args.exp == 15:
        ACTIVE_GROUPS = EXP15_GROUPS
        ACTIVE_EXPERIMENT_NAME = "Exp15 (Exp8基础, 365d训练, 无cfgi/news)"
        if not _args.output_dir:
            OUTPUT_DIR = PROJECT_ROOT / "data" / "models" / "v5_production_365d"
    elif _args.exp == 16:
        ACTIVE_GROUPS = EXP16_GROUPS
        ACTIVE_EXPERIMENT_NAME = "Exp16 (Exp13+TV宏观, 365d训练, 无cfgi/news)"
        if not _args.output_dir:
            OUTPUT_DIR = PROJECT_ROOT / "data" / "models" / "v5_production_tv_365d"
    elif _args.exp == 17:
        ACTIVE_GROUPS = EXP17_GROUPS
        ACTIVE_EXPERIMENT_NAME = "Exp17 (TV宏观+PM当前bar, 无target PM泄漏特征)"
        if not _args.output_dir:
            OUTPUT_DIR = PROJECT_ROOT / "data" / "models" / "v5_production_no_target_pm"

    if _args.output_dir:
        OUTPUT_DIR = Path(_args.output_dir)

    explicit_groups = []
    if str(_args.feature_groups or "").strip():
        explicit_groups = [g.strip() for g in str(_args.feature_groups).split(",") if g.strip()]
        invalid = [g for g in explicit_groups if g not in FEATURE_GROUPS]
        if invalid:
            raise ValueError(f"--feature-groups 存在未知特征组: {invalid}")
        ACTIVE_GROUPS = list(dict.fromkeys(explicit_groups))
        ACTIVE_EXPERIMENT_NAME = f"{ACTIVE_EXPERIMENT_NAME} [explicit_groups]"

    if bool(_args.traditional_only):
        ACTIVE_GROUPS = []
        ACTIVE_EXPERIMENT_NAME = f"{ACTIVE_EXPERIMENT_NAME} [traditional_only]"

    extra_groups = []
    for group in _args.extra_group:
        name = str(group).strip()
        if not name:
            continue
        if name not in FEATURE_GROUPS:
            raise ValueError(f"未知特征组: {name}")
        extra_groups.append(name)
    if extra_groups:
        merged_groups = list(ACTIVE_GROUPS)
        for group in extra_groups:
            if group not in merged_groups:
                merged_groups.append(group)
        ACTIVE_GROUPS = merged_groups
        ACTIVE_EXPERIMENT_NAME = f"{ACTIVE_EXPERIMENT_NAME} + {'+'.join(extra_groups)}"

    # 选择 LightGBM 参数集
    use_regularized = _args.regularized
    active_lgb_params = dict(REGULARIZED_PARAMS if use_regularized else EXP7_TUNED_PARAMS)
    lgb_params_name = "REGULARIZED (P2正则化)" if use_regularized else "EXP7_TUNED (原始)"
    if str(_args.lgb_params_json or "").strip():
        lgb_params_path = Path(str(_args.lgb_params_json)).expanduser().resolve()
        override_params = json.loads(lgb_params_path.read_text(encoding="utf-8"))
        if not isinstance(override_params, dict):
            raise ValueError("--lgb-params-json 必须指向 JSON 对象文件")
        active_lgb_params.update(override_params)
        if str(_args.lgb_params_name or "").strip():
            lgb_params_name = str(_args.lgb_params_name).strip()
        else:
            lgb_params_name = f"OVERRIDE::{lgb_params_path.stem}"
    btc_weight_boost = _args.btc_weight_boost

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = get_data_paths()
    avail = check_data_availability(paths)
    device = get_device(use_mps=True)
    selected_assets = list(ASSETS)
    if _args.assets:
        requested = [a.strip().upper() for a in str(_args.assets).split(",") if a.strip()]
        if not requested:
            raise ValueError("--assets 为空")
        invalid = [a for a in requested if a not in ASSETS]
        if invalid:
            raise ValueError(f"未知资产: {invalid}，可选: {list(ASSETS)}")
        selected_assets = requested

    # 训练窗口和集成窗口选择
    if _args.exp == 17:
        active_train_days = 718  # 最大可用天数 (tv_macro 从 2024-01-01, ~778d 减去 warmup+val+test=60d)
        active_window_days_list = WINDOW_DAYS_LIST_MAX  # [120, 365, 718]
    elif _args.exp in (15, 16):
        active_train_days = 365
        active_window_days_list = WINDOW_DAYS_LIST_365  # [120, 240, 365]
    else:
        active_train_days = DEFAULT_DAYS  # 120
        active_window_days_list = WINDOW_DAYS_LIST      # [60, 90, 120]

    if int(_args.train_days or 0) > 0:
        active_train_days = int(_args.train_days)
    if str(_args.window_days_list or "").strip():
        raw_items = [x.strip() for x in str(_args.window_days_list).split(",") if x.strip()]
        parsed = sorted({int(x) for x in raw_items if int(x) > 0})
        if not parsed:
            raise ValueError("--window-days-list 解析后为空")
        active_window_days_list = parsed

    total_window = active_train_days + WARMUP_DAYS + VAL_DAYS + TEST_DAYS
    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=total_window)

    use_sim_noise = _args.sim_noise

    print(f"{'=' * 70}")
    print(f"  训练 v5 生产模型 — Pooled {ACTIVE_EXPERIMENT_NAME}")
    print(f"  Assets: {', '.join(selected_assets)}")
    print(f"  Window: {total_window}d from {cutoff_date.strftime('%Y-%m-%d')}")
    print(f"  Ensemble: {active_window_days_list}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Data: {avail}")
    if bool(_args.disable_gru_embeddings):
        print("  GRU 嵌入: 关闭（compare-only/no-GRU 路径）")
    print(f"  LightGBM 参数: {lgb_params_name}")
    if use_regularized or str(_args.lgb_params_json or "").strip():
        print(f"    num_leaves={active_lgb_params['num_leaves']}, max_depth={active_lgb_params['max_depth']}, "
              f"lr={active_lgb_params['learning_rate']}, L1={active_lgb_params['lambda_l1']}, L2={active_lgb_params['lambda_l2']}")
    print(f"  训练控制: n_estimators_cap={int(_args.n_estimators_cap)}, early_stopping={int(_args.early_stopping_rounds)}")
    if use_sim_noise:
        print(f"  模拟K线噪声: 启用 (σ={SIM_NOISE_STD*100:.2f}%)")
    if btc_weight_boost != 1.0:
        print(f"  BTC 权重提升: {btc_weight_boost}x")
    if _args.trade_objective_weighting:
        print("  交易目标加权: 启用 (顺势 extreme/rally 提权, countertrend 降权)")
        print(
            "    "
            f"trend={_args.trade_objective_trend_bonus:.2f}, rally={_args.trade_objective_rally_bonus:.2f}, "
            f"counter={_args.trade_objective_countertrend_penalty:.2f}, rally_counter={_args.trade_objective_rally_countertrend_penalty:.2f}"
        )
    if _args.uncertainty_aware_weighting:
        print(f"  不确定性感知加权: 启用 (penalty_scale={_args.uncertainty_penalty_scale:.2f})")
    if (
        float(_args.trade_objective_countertrend_keep_frac) < 1.0
        or float(_args.trade_objective_rally_countertrend_keep_frac) < 1.0
        or float(_args.trade_objective_blocked_keep_frac) < 1.0
    ):
        print(
            "  交易目标剪枝: "
            f"countertrend_keep={float(_args.trade_objective_countertrend_keep_frac):.2f}, "
            f"rally_counter_keep={float(_args.trade_objective_rally_countertrend_keep_frac):.2f}, "
            f"blocked_keep={float(_args.trade_objective_blocked_keep_frac):.2f}"
        )
    print(f"{'=' * 70}")

    pooled_input_path = str(_args.pooled_features_in or "").strip()
    gru_model_dir = PROJECT_ROOT / "experiments" / "sentiment_grid_search" / "models"

    # ═══ Phase 1/2: 构建或复用 pooled 特征 ═══
    print(f"\n{'─' * 70}")
    print(f"  Phase 2: 合池 + 训练 LightGBM 集成")
    print(f"{'─' * 70}")

    if pooled_input_path:
        pooled_path = Path(pooled_input_path).expanduser().resolve()
        pooled = _read_table(pooled_path)
        pooled["timestamp"] = pd.to_datetime(pooled["timestamp"], utc=True, errors="coerce")
        pooled = pooled.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        asset_names = (
            sorted({str(v) for v in pooled["_asset_name"].dropna().astype(str).tolist()})
            if "_asset_name" in pooled.columns else list(selected_assets)
        )
        print(f"  复用 pooled 特征表: {pooled_path}")
        print(f"  合池: {len(pooled)} rows ({len(asset_names)} 币)")
    else:
        print(f"\n{'─' * 70}")
        phase1_label = "加载特征 + 轻量 no-GRU" if bool(_args.disable_gru_embeddings) else "加载特征 + GRU 嵌入"
        print(f"  Phase 1: {phase1_label}")
        print(f"{'─' * 70}")

        asset_data = {}
        for idx, asset in enumerate(selected_assets):
            print(f"\n  {asset}:")

            # 1a. Tech features
            print(f"    构建技术特征...", flush=True)
            tech_df = build_tech_features(paths["data_src"], asset)
            tech_df = _ensure_datetime_col(tech_df, "timestamp")

            # 1a+. 多时间框架特征 (1h/4h KDJ/Range/MACD → mtf_1h_*, mtf_4h_*)
            pre_mtf_cols = len(tech_df.columns)
            try:
                tech_df = add_multi_timeframe_features(tech_df, asset)
                mtf_added = len(tech_df.columns) - pre_mtf_cols
                if mtf_added > 0:
                    print(f"    多时间框架: +{mtf_added} 个 mtf_* 特征 (1h/4h)")
            except Exception as e:
                print(f"    ⚠️ 多时间框架特征跳过: {e}")

            full_len = len(tech_df)
            tech_df = tech_df[tech_df["timestamp"] >= cutoff_date].reset_index(drop=True)
            print(f"    {full_len} → {len(tech_df)} rows, {len(tech_df.columns)} cols")

            if use_sim_noise:
                tech_df = apply_simulated_candle_noise(tech_df, seed=SEED + idx)
                if idx == 0:
                    print(f"    [模拟K线噪声] 已应用 (close/high/low/volume 扰动)")

            if "log_return" not in tech_df.columns:
                tech_df["log_return"] = np.log(tech_df["close"] / tech_df["close"].shift(1))

            if Path(paths.get("funding_path", "")).exists():
                tech_df = merge_funding_rate(tech_df, paths["funding_path"], asset)

            ohlcv_df = load_ohlcv(paths["data_src"], asset)
            ohlcv_df = ohlcv_df[ohlcv_df["timestamp"] >= cutoff_date].reset_index(drop=True)
            if "timestamp_ms" not in tech_df.columns and "timestamp_ms" in ohlcv_df.columns:
                tech_df = pd.merge_asof(
                    tech_df.sort_values("timestamp"),
                    ohlcv_df[["timestamp", "timestamp_ms"]].sort_values("timestamp"),
                    on="timestamp", direction="nearest",
                    tolerance=pd.Timedelta("1min"),
                )

            if bool(_args.disable_gru_embeddings):
                emb_df = _build_empty_embeddings(tech_df)
                print("    跳过 GRU 嵌入: compare-only/no-GRU")
            else:
                model_path = gru_model_dir / f"{asset}_enhanced.pt"
                norm_path = gru_model_dir / f"{asset}_enhanced_normalizer.json"

                if not model_path.exists():
                    print(f"    ⚠️ GRU 模型不存在: {model_path}，需要先运行 run_grid.py")
                    return

                for c in GRU_FEATURE_COLS:
                    if c not in tech_df.columns:
                        tech_df[c] = 0.0

                print(f"    提取 GRU 嵌入...", flush=True)
                emb_df = extract_embeddings(tech_df, model_path, norm_path, device)
                print(f"    嵌入: {len(emb_df)} rows, {emb_df.shape[1] - 1} dims")

            asset_data[asset] = (tech_df, emb_df)

        asset_names = sorted(asset_data.keys())
        frames = []
        for idx, asset in enumerate(asset_names):
            tech_df, emb_df = asset_data[asset]
            prepared = _prepare_asset_for_pooling(
                tech_df, emb_df, ACTIVE_GROUPS, paths, asset, asset_id=idx,
            )
            frames.append(prepared)

        pooled = pd.concat(frames, ignore_index=True)
        pooled = pooled.sort_values("timestamp").reset_index(drop=True)
        print(f"  合池: {len(pooled)} rows ({len(asset_names)} 币)")

    # P3: BTC sample weight boost
    if btc_weight_boost != 1.0:
        # asset_id=0 对应 BTC（sorted order: BTC=0, ETH=1, XRP=2）
        btc_mask = pooled["asset_id"] == 0
        n_btc = btc_mask.sum()
        pooled.loc[btc_mask, "sample_weight"] *= btc_weight_boost
        print(f"  BTC 权重提升: {n_btc} 个 BTC 样本 x{btc_weight_boost} (总 {len(pooled)} 样本)")

    trade_objective_weight_summary: Dict[str, Any] = {"enabled": False}
    if _args.trade_objective_weighting:
        pooled, trade_objective_weight_summary = apply_trade_objective_weighting(
            pooled,
            assets_by_id={idx: asset for idx, asset in enumerate(asset_names)},
            trend_bonus=float(_args.trade_objective_trend_bonus),
            rally_bonus=float(_args.trade_objective_rally_bonus),
            countertrend_penalty=float(_args.trade_objective_countertrend_penalty),
            rally_countertrend_penalty=float(_args.trade_objective_rally_countertrend_penalty),
            high_vol_follow_bonus=float(_args.trade_objective_high_vol_follow_bonus),
            high_vol_countertrend_penalty=float(_args.trade_objective_high_vol_countertrend_penalty),
            uncertainty_aware_weighting=bool(_args.uncertainty_aware_weighting),
            uncertainty_penalty_scale=float(_args.uncertainty_penalty_scale),
            countertrend_keep_frac=float(_args.trade_objective_countertrend_keep_frac),
            rally_countertrend_keep_frac=float(_args.trade_objective_rally_countertrend_keep_frac),
            blocked_keep_frac=float(_args.trade_objective_blocked_keep_frac),
        )
        print("  交易目标加权:")
        global_summary = trade_objective_weight_summary.get("global") if isinstance(trade_objective_weight_summary, dict) else None
        if isinstance(global_summary, dict):
            print(f"    均值 {global_summary['mean_before']:.4f} → {global_summary['mean_after']:.4f}")
        else:
            status = trade_objective_weight_summary.get("status") if isinstance(trade_objective_weight_summary, dict) else "unknown"
            print(f"    ⚠️ 缺少全局加权摘要，status={status}")
        for asset_name, asset_summary in (trade_objective_weight_summary.get("by_asset") or {}).items():
            print(
                f"    {asset_name}: avg_mult={asset_summary['avg_weight_mult']:.3f}, "
                f"trend={asset_summary['trend_follow_rows']}, countertrend={asset_summary['countertrend_rows']}, "
                f"rally={asset_summary['rally_follow_rows']}, rally_ct={asset_summary['rally_countertrend_rows']}"
            )

    # 特征列
    exclude = {
        "timestamp", "timestamp_ms", "date", "asset", "_asset_name",
        "direction_label", "sample_weight",
        "symbol", "open", "high", "low", "close", "volume", "btc_close",
    }
    leaky_prefixes = ("target_", "future_")
    feature_cols = []
    for c in pooled.columns:
        if c in exclude:
            continue
        if any(c.startswith(prefix) for prefix in leaky_prefixes):
            continue
        if c == "asset_id":
            feature_cols.append(c)
            continue
        if pooled[c].dtype not in ["float64", "float32", "int64", "int32"]:
            continue
        if pooled[c].isna().all():
            continue
        feature_cols.append(c)

    numeric_cols = [c for c in feature_cols if c != "asset_id"]
    numeric_cols = _deduplicate_features(pooled, numeric_cols, threshold=0.95)
    feature_cols = numeric_cols + ["asset_id"]

    feature_allowlist_path = str(_args.feature_allowlist_file or "").strip()
    allowlist_used: list[str] = []
    if feature_allowlist_path:
        allowlist_used = _load_feature_allowlist(feature_allowlist_path)
        allowset = set(allowlist_used)
        filtered = [c for c in feature_cols if c == "asset_id" or c in allowset]
        if "asset_id" not in filtered:
            filtered.append("asset_id")
        missing = sorted(set(allowlist_used) - set(feature_cols))
        print(f"  白名单特征: {len(allowlist_used)} -> 命中 {len(filtered)}")
        if missing:
            print(f"  ⚠️ 白名单缺失 {len(missing)} 个特征（忽略）")
        feature_cols = filtered
    print(f"  特征: {len(feature_cols)} (去冗余后)")

    stats_rows = []
    for col in feature_cols:
        series = pooled[col]
        if col == "asset_id":
            stats_rows.append({
                "feature": col,
                "non_null_pct": round(float(series.notna().mean() * 100.0), 4),
                "nunique": int(series.nunique(dropna=True)),
                "std": None,
                "zero_pct": round(float((series.fillna(0) == 0).mean() * 100.0), 4),
                "mean_abs": None,
            })
            continue
        numeric = pd.to_numeric(series, errors="coerce")
        stats_rows.append({
            "feature": col,
            "non_null_pct": round(float(numeric.notna().mean() * 100.0), 4),
            "nunique": int(numeric.nunique(dropna=True)),
            "std": round(float(numeric.std(skipna=True) or 0.0), 10),
            "zero_pct": round(float((numeric.fillna(0) == 0).mean() * 100.0), 4),
            "mean_abs": round(float(numeric.abs().mean(skipna=True) or 0.0), 10),
        })
    feature_stats_df = pd.DataFrame(stats_rows)

    pooled_features_out = str(_args.pooled_features_out or "").strip()
    if pooled_features_out:
        base_cols = ["timestamp", "_asset_name", "asset_id", "direction_label", "sample_weight"]
        pooled_out_cols = [c for c in base_cols if c in pooled.columns] + [c for c in feature_cols if c not in base_cols]
        _write_table(Path(pooled_features_out).expanduser().resolve(), pooled[pooled_out_cols].copy())
        print(f"  ✅ pooled_features_out -> {pooled_features_out}")

    # 日期划分
    latest_ts = pooled["timestamp"].max()
    test_cutoff = latest_ts - pd.Timedelta(days=TEST_DAYS)
    val_cutoff = test_cutoff - pd.Timedelta(days=VAL_DAYS)

    test_df = pooled[pooled["timestamp"] >= test_cutoff].copy()
    val_df = pooled[(pooled["timestamp"] >= val_cutoff) & (pooled["timestamp"] < test_cutoff)].copy()

    X_va = val_df[feature_cols]
    y_va = val_df["direction_label"]
    X_te = test_df[feature_cols]
    y_te = test_df["direction_label"]

    n_assets = len(asset_names)
    purge_rows = PURGE_BARS * n_assets

    # 训练 3 个 Expanding Window 模型
    models = []
    ensemble_probas = []
    ensemble_info = []

    for w_days in active_window_days_list:
        train_cutoff = val_cutoff - pd.Timedelta(days=w_days)
        train_mask = (pooled["timestamp"] >= train_cutoff) & (pooled["timestamp"] < val_cutoff)
        train_df = pooled[train_mask].copy()

        if len(train_df) > purge_rows:
            train_df = train_df.iloc[:-purge_rows]

        print(f"\n  训练窗口 {w_days}d: {len(train_df)} 样本")

        model = lgb.LGBMClassifier(
            **active_lgb_params,
            n_estimators=int(_args.n_estimators_cap),
            random_state=SEED,
        )
        model.fit(
            train_df[feature_cols], train_df["direction_label"],
            sample_weight=train_df["sample_weight"],
            eval_set=[(X_va, y_va)],
            categorical_feature=["asset_id"],
            callbacks=[
                lgb.early_stopping(int(_args.early_stopping_rounds), verbose=False),
                lgb.log_evaluation(0),
            ],
        )

        proba = model.predict_proba(X_te)[:, 1]
        ensemble_probas.append(proba)
        models.append(model)

        val_proba = model.predict_proba(X_va)[:, 1]
        val_auc = roc_auc_score(y_va, val_proba)

        info = {
            "window_days": w_days,
            "train_samples": len(train_df),
            "best_iter": model.best_iteration_,
            "val_auc": round(float(val_auc), 5),
        }
        ensemble_info.append(info)
        print(f"    best_iter={model.best_iteration_}, val_auc={val_auc:.5f}")

    importance_rows = []
    for model, info in zip(models, ensemble_info):
        booster = model.booster_
        feature_names = booster.feature_name()
        gain_values = booster.feature_importance(importance_type="gain")
        split_values = booster.feature_importance(importance_type="split")
        total_gain = float(np.sum(gain_values) or 0.0)
        total_split = float(np.sum(split_values) or 0.0)
        for feature_name, gain_value, split_value in zip(feature_names, gain_values, split_values):
            importance_rows.append({
                "feature": str(feature_name),
                "window_days": int(info["window_days"]),
                "gain": float(gain_value),
                "gain_pct": float(gain_value) / total_gain if total_gain > 0 else 0.0,
                "split": int(split_value),
                "split_pct": float(split_value) / total_split if total_split > 0 else 0.0,
            })
    importance_df = pd.DataFrame(importance_rows)
    if not importance_df.empty:
        importance_agg_df = (
            importance_df.groupby("feature", as_index=False)
            .agg(
                mean_gain=("gain", "mean"),
                total_gain=("gain", "sum"),
                mean_gain_pct=("gain_pct", "mean"),
                mean_split=("split", "mean"),
                total_split=("split", "sum"),
                mean_split_pct=("split_pct", "mean"),
                windows_present=("window_days", "nunique"),
            )
            .sort_values(["mean_gain_pct", "total_gain", "mean_split_pct"], ascending=False)
            .reset_index(drop=True)
        )
    else:
        importance_agg_df = pd.DataFrame(columns=[
            "feature", "mean_gain", "total_gain", "mean_gain_pct",
            "mean_split", "total_split", "mean_split_pct", "windows_present",
        ])

    # 集成评估
    proba_te = np.mean(ensemble_probas, axis=0)
    auc = roc_auc_score(y_te, proba_te)
    brier = brier_score_loss(y_te, proba_te)
    acc = ((proba_te >= 0.5).astype(int) == y_te.values).mean()
    pred_dir = (proba_te >= 0.5).astype(int)
    correct = (pred_dir == y_te.values).astype(float)
    pnl = correct * 2 - 1 - 0.001
    sharpe = pnl.mean() / (pnl.std() + 1e-10) * np.sqrt(252 * 96)

    per_asset_auc = {}
    y_te_np = y_te.values
    asset_name_np = test_df["_asset_name"].values
    for asset in asset_names:
        mask = asset_name_np == asset
        if mask.sum() >= 30:
            per_asset_auc[asset] = round(float(roc_auc_score(y_te_np[mask], proba_te[mask])), 5)

    print(f"\n{'═' * 70}")
    print(f"  生产模型评估")
    print(f"{'═' * 70}")
    print(f"  Pooled AUC: {auc:.5f}")
    print(f"  Sharpe:     {sharpe:.3f}")
    print(f"  Accuracy:   {acc:.4f}")
    print(f"  Brier:      {brier:.5f}")
    for a, v in per_asset_auc.items():
        print(f"  {a}: AUC={v}")

    # ═══ Phase 3: 保存模型 ═══
    print(f"\n{'─' * 70}")
    print(f"  Phase 3: 保存到 {OUTPUT_DIR}")
    print(f"{'─' * 70}")

    # 保存 3 个 LightGBM 模型
    for i, (model, info) in enumerate(zip(models, ensemble_info)):
        model_path = OUTPUT_DIR / f"lgb_{info['window_days']}d.joblib"
        joblib.dump(model, model_path)
        print(f"  ✅ {model_path.name}")

    # 保存特征列表
    feature_list_path = OUTPUT_DIR / "feature_cols.json"
    with open(feature_list_path, "w") as f:
        json.dump(feature_cols, f, indent=2)
    print(f"  ✅ {feature_list_path.name}")

    stats_out = Path(str(_args.feature_stats_out).strip()).expanduser().resolve() if str(_args.feature_stats_out).strip() else OUTPUT_DIR / "feature_stats.csv"
    _write_table(stats_out, feature_stats_df)
    print(f"  ✅ {stats_out.name}")

    importance_out = Path(str(_args.feature_importance_out).strip()).expanduser().resolve() if str(_args.feature_importance_out).strip() else OUTPUT_DIR / "feature_importance.csv"
    _write_table(importance_out, importance_agg_df)
    print(f"  ✅ {importance_out.name}")

    # 保存 asset → id 映射
    asset_map = {asset: idx for idx, asset in enumerate(asset_names)}
    asset_map_path = OUTPUT_DIR / "asset_map.json"
    with open(asset_map_path, "w") as f:
        json.dump(asset_map, f, indent=2)
    print(f"  ✅ {asset_map_path.name}")

    # 复制 GRU 模型引用（指向已有路径，不复制文件）；no-GRU 路径留空
    gru_paths = {}
    if not bool(_args.disable_gru_embeddings):
        for asset in asset_names:
            gru_paths[asset] = {
                "model": str(gru_model_dir / f"{asset}_enhanced.pt"),
                "normalizer": str(gru_model_dir / f"{asset}_enhanced_normalizer.json"),
                "config": str(gru_model_dir / f"{asset}_enhanced_config.json"),
            }

    # 保存完整配置
    config = {
        "version": OUTPUT_DIR.name,
        "trained_at": ts,
        "mode": "pooled",
        "experiment": ACTIVE_EXPERIMENT_NAME,
        "sim_noise_augment": use_sim_noise,
        "sim_noise_std": SIM_NOISE_STD if use_sim_noise else None,
        "btc_weight_boost": btc_weight_boost,
        "trade_objective_weighting": trade_objective_weight_summary,
        "uncertainty_aware_weighting": bool(_args.uncertainty_aware_weighting),
        "assets": asset_names,
        "asset_map": asset_map,
        "feature_groups": ACTIVE_GROUPS,
        "traditional_only": bool(_args.traditional_only),
        "feature_allowlist_file": feature_allowlist_path or None,
        "feature_allowlist_used": allowlist_used,
        "pooled_features_in": pooled_input_path or None,
        "lgb_params": active_lgb_params,
        "lgb_params_name": lgb_params_name,
        "n_estimators_cap": int(_args.n_estimators_cap),
        "early_stopping_rounds": int(_args.early_stopping_rounds),
        "window_days_list": active_window_days_list,
        "ensemble_info": ensemble_info,
        "num_features": len(feature_cols),
        "disable_gru_embeddings": bool(_args.disable_gru_embeddings),
        "gru_hparams": dict(GRU_HPARAMS),
        "gru_feature_cols": list(GRU_FEATURE_COLS),
        "gru_paths": gru_paths,
        "metrics": {
            "pooled_auc": round(float(auc), 5),
            "sharpe": round(float(sharpe), 3),
            "accuracy": round(float(acc), 4),
            "brier": round(float(brier), 5),
            "per_asset_auc": per_asset_auc,
        },
        "data_paths": {k: str(v) for k, v in paths.items()},
        "train_days": active_train_days,
        "val_days": VAL_DAYS,
        "test_days": TEST_DAYS,
    }

    config_path = OUTPUT_DIR / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False, default=str)
    print(f"  ✅ {config_path.name}")

    print(f"\n{'═' * 70}")
    print(f"  生产模型训练完成！")
    print(f"  模型目录: {OUTPUT_DIR}")
    print(f"  文件列表:")
    for f in sorted(OUTPUT_DIR.glob("*")):
        size = f.stat().st_size
        print(f"    {f.name} ({size:,} bytes)")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
