#!/usr/bin/env python3
"""
LightGBM 在线增量学习 — 每日自动微调

在已有 LightGBM 模型上追加少量新树（按波动自适应），让模型持续适应最新市场微观结构。
GRU 编码器不做增量训练（长期时序特征提取器，不需频繁更新）。

数据与备份：每日训练为增量追加，仅覆盖同路径模型文件并保留近期备份，
不删除历史训练数据（raw/parquet 不被脚本删除）。

用法:
    python3 scripts/online_learning_daily.py            # 所有 Exp 模型
    python3 scripts/online_learning_daily.py --dry-run   # 只诊断，不写入模型
    python3 scripts/online_learning_daily.py --exp 13    # 仅更新 Exp13

调度:
    crontab -e → 0 0 * * * cd /Users/mac/polyfun && python3 scripts/online_learning_daily.py >> logs/online_learning.log 2>&1
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
from logging.handlers import RotatingFileHandler
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.sentiment_grid_search.run_grid import (
    ASSETS,
    GRU_FEATURE_COLS,
    GRU_HPARAMS,
    build_tech_features,
    extract_embeddings,
    merge_sentiment_to_tech,
    _prepare_asset_for_pooling,
)
from experiments.sentiment_grid_search.data_prep import (
    _ensure_datetime_col,
    merge_funding_rate,
    load_ohlcv,
)
from experiments.gru_regime_v1.src.utils import get_device
from src.python.feature_engineering import add_multi_timeframe_features

# ─── 常量 ──────────────────────────────────────────────────

INCREMENTAL_LR = 0.005
BASE_NUM_BOOST_ROUND = 12
CALM_NUM_BOOST_ROUND = 8
SHOCK_NUM_BOOST_ROUND = 20
BAGGING_FRACTION = 0.8
MIN_SAMPLES = 50
BACKUP_KEEP_DAYS = max(1, int(os.getenv("ONLINE_LEARNING_BACKUP_KEEP_DAYS", "3") or "3"))
PARAM_SEARCH_NUMERIC_THREADS = max(0, int(os.getenv("PARAM_SEARCH_NUMERIC_THREADS", "0") or "0"))
ROLLBACK_AUC_DROP_ABS = 0.004
ROLLBACK_AUC_DROP_REL = 0.008
ROLLBACK_UTILITY_DROP_ABS = 0.03
ROLLBACK_DOWN_WR_DROP_ABS = 0.05
TRADE_OBJECTIVE_AUC_CATASTROPHIC_MULT = 3.0
UTILITY_MIN_CONFIDENCE = 0.53
UTILITY_MIN_SAMPLES = 120
UTILITY_MIN_DOWN_SAMPLES = 40
RECENCY_HALFLIFE_BARS = 64
SHOCK_SAMPLE_WEIGHT_MULT = 1.8
CALM_VOL_RATIO = 1.08
SHOCK_VOL_RATIO = 1.45
SHOCK_RET_Q = 0.85

EXP_MODEL_DIRS: Dict[int, str] = {
    10: "v5_production_sim_noise",
    13: "v5_production_tv",
    14: "v5_production_sim_noise_tv",
    15: "v5_production_365d",
    16: "v5_production_tv_365d",
    17: "v5_production_no_target_pm",
}

BACKUP_DIR = PROJECT_ROOT / "data" / "models" / "_online_learning_backups"
LOG_FILE = PROJECT_ROOT / "logs" / "online_learning.log"
LAST_SUCCESS_PATH = PROJECT_ROOT / "logs" / "daily_training_last_success.json"
ONLINE_LOG_MAX_MB = int(os.getenv("ONLINE_LEARNING_LOG_MAX_MB", "32") or "32")
ONLINE_LOG_BACKUP_COUNT = int(os.getenv("ONLINE_LEARNING_LOG_BACKUP_COUNT", "5") or "5")
LAST_SUCCESS_RETENTION_HOURS = int(os.getenv("ONLINE_LEARNING_LAST_SUCCESS_RETENTION_HOURS", "30") or "30")

logger = logging.getLogger("online_learning")


def _merge_iso_map(existing: object, updates: Dict[str, str]) -> Dict[str, str]:
    merged: Dict[str, str] = {}
    if isinstance(existing, dict):
        for key, value in existing.items():
            key_s = str(key).strip()
            value_s = str(value).strip()
            if key_s and value_s:
                merged[key_s] = value_s
    for key, value in updates.items():
        key_s = str(key).strip()
        value_s = str(value).strip()
        if not key_s or not value_s:
            continue
        old = merged.get(key_s)
        if old is None:
            merged[key_s] = value_s
            continue
        old_dt = parse_dt(old)
        new_dt = parse_dt(value_s)
        if old_dt is None or (new_dt is not None and new_dt > old_dt):
            merged[key_s] = value_s
    return dict(sorted(merged.items()))


def _merge_str_list(existing: object, additions: list[str]) -> list[str]:
    out = {str(x).strip() for x in additions if str(x).strip()}
    if isinstance(existing, list):
        out.update(str(x).strip() for x in existing if str(x).strip())
    return sorted(out)


def parse_dt(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _calc_abs_returns(df: pd.DataFrame) -> pd.Series:
    """优先用 log_return，否则用 close 计算对数收益绝对值。"""
    if "log_return" in df.columns:
        ret = pd.to_numeric(df["log_return"], errors="coerce")
    elif "close" in df.columns:
        c = pd.to_numeric(df["close"], errors="coerce")
        ret = np.log(c / c.shift(1))
    else:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return ret.abs().replace([np.inf, -np.inf], np.nan)


def _build_adaptive_plan(
    df: pd.DataFrame,
    base_rounds: int = BASE_NUM_BOOST_ROUND,
    calm_rounds: int = CALM_NUM_BOOST_ROUND,
    shock_rounds: int = SHOCK_NUM_BOOST_ROUND,
    recency_halflife_bars: int = RECENCY_HALFLIFE_BARS,
    shock_weight_mult: float = SHOCK_SAMPLE_WEIGHT_MULT,
    calm_vol_ratio: float = CALM_VOL_RATIO,
    shock_vol_ratio: float = SHOCK_VOL_RATIO,
    shock_ret_q: float = SHOCK_RET_Q,
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    构建“近期加权 + 冲击加权”样本权重，并给出本轮增量树数。
    - 平稳期少加树，降低噪声学习；
    - 冲击期多加树，提升 regime shift 适应速度。
    """
    n = len(df)
    if n == 0:
        return pd.Series(dtype=float), {
            "regime": "unknown",
            "vol_ratio": 1.0,
            "last_abs_ret": np.nan,
            "shock_threshold": np.nan,
            "num_boost_round": float(base_rounds),
            "shock_samples": 0.0,
        }

    age = (n - 1 - np.arange(n)).astype(float)
    hl = max(1, int(recency_halflife_bars))
    recency_w = np.exp(-np.log(2.0) * age / hl)
    weights = pd.Series(recency_w, index=df.index, dtype=float)

    abs_ret = _calc_abs_returns(df)
    valid = abs_ret.dropna()
    regime = "normal"
    vol_ratio = 1.0
    last_abs_ret = float(valid.iloc[-1]) if len(valid) > 0 else np.nan
    shock_threshold = np.nan
    shock_samples = 0

    if len(valid) >= 32:
        baseline_window = min(len(valid), 96)
        recent_window = min(len(valid), 24)
        baseline = float(valid.tail(baseline_window).median())
        recent = float(valid.tail(recent_window).mean())
        vol_ratio = recent / max(baseline, 1e-12)

        tail = valid.tail(baseline_window)
        shock_threshold = float(tail.quantile(max(0.50, min(0.99, shock_ret_q))))
        shock_cut = shock_threshold * 1.15
        is_shock = (vol_ratio >= max(1.0, float(shock_vol_ratio))) or (last_abs_ret >= shock_cut)

        if is_shock:
            regime = "shock"
            shock_mask = (abs_ret >= shock_threshold).fillna(False)
            shock_samples = int(shock_mask.sum())
            if shock_samples > 0:
                weights.loc[shock_mask] *= max(1.0, float(shock_weight_mult))
        elif vol_ratio <= max(1.0, float(calm_vol_ratio)):
            regime = "calm"

    if regime == "shock":
        num_boost_round = max(1, int(shock_rounds))
    elif regime == "calm":
        num_boost_round = max(1, int(calm_rounds))
    else:
        num_boost_round = max(1, int(base_rounds))

    mean_w = float(weights.mean()) if len(weights) > 0 else 1.0
    if mean_w > 0:
        weights = weights / mean_w
    weights = weights.clip(lower=0.1, upper=5.0)

    return weights, {
        "regime": regime,
        "vol_ratio": float(vol_ratio),
        "last_abs_ret": float(last_abs_ret),
        "shock_threshold": float(shock_threshold),
        "num_boost_round": float(num_boost_round),
        "shock_samples": float(shock_samples),
    }


def _calc_validation_trade_utility(
    y_true: pd.Series,
    proba_up: np.ndarray,
    min_confidence: float = UTILITY_MIN_CONFIDENCE,
) -> Dict[str, Optional[float]]:
    """在验证集上计算“可交易样本”的方向效用代理指标。"""
    y = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
    p = np.asarray(proba_up, dtype=float)
    n = min(len(y), len(p))
    if n <= 0:
        return {
            "n": 0.0,
            "wr": None,
            "pnl_proxy": None,
            "down_n": 0.0,
            "down_wr": None,
            "down_pnl_proxy": None,
        }

    y = y[:n].astype(int)
    p = np.clip(p[:n], 1e-6, 1 - 1e-6)
    pred_up = p >= 0.5
    pred = pred_up.astype(int)
    confidence = np.where(pred_up, p, 1.0 - p)
    take = confidence >= float(min_confidence)
    if int(take.sum()) <= 0:
        return {
            "n": 0.0,
            "wr": None,
            "pnl_proxy": None,
            "down_n": 0.0,
            "down_wr": None,
            "down_pnl_proxy": None,
        }

    wins = (pred == y).astype(float)
    edge = 2.0 * confidence - 1.0
    pnl_proxy = np.where(wins > 0.5, edge, -edge)

    down_take = take & (~pred_up)
    down_wins = wins[down_take] if int(down_take.sum()) > 0 else np.array([], dtype=float)
    down_edge = edge[down_take] if int(down_take.sum()) > 0 else np.array([], dtype=float)
    down_pnl_proxy = np.where(down_wins > 0.5, down_edge, -down_edge) if len(down_wins) > 0 else np.array([], dtype=float)

    return {
        "n": float(int(take.sum())),
        "wr": float(wins[take].mean()) if int(take.sum()) > 0 else None,
        "pnl_proxy": float(pnl_proxy[take].mean()) if int(take.sum()) > 0 else None,
        "down_n": float(int(down_take.sum())),
        "down_wr": float(down_wins.mean()) if len(down_wins) > 0 else None,
        "down_pnl_proxy": float(down_pnl_proxy.mean()) if len(down_pnl_proxy) > 0 else None,
    }


# ─── 数据准备 ──────────────────────────────────────────────

def _build_recent_data(
    config: dict,
    hours: int = 48,
    device: str = "cpu",
    need_mtf: bool = False,
    as_of: datetime | None = None,
) -> Optional[pd.DataFrame]:
    """
    构建最近 N 小时的特征+标签数据（复用训练流程）。

    使用 48h 而非 24h 是因为:
    - 技术指标需要 lookback 窗口（RSI/EMA 等至少 14-50 根 bar）
    - GRU 需要 64 根 bar 的历史序列
    - 48h = 192 根 15m bar, 减去 warmup 后有效数据约 24h

    Args:
        need_mtf: 是否构建 MTF 特征。仅当模型的 feature_cols 包含 mtf_* 时传 True。
    """
    paths = config.get("data_paths", {})
    data_src = paths.get("data_src", str(PROJECT_ROOT / "data" / "raw"))
    assets = config.get("assets", ASSETS)
    feature_groups = config.get("feature_groups", [])
    gru_model_dir = PROJECT_ROOT / "experiments" / "sentiment_grid_search" / "models"

    now = as_of or datetime.now(timezone.utc)
    cutoff = now - pd.Timedelta(hours=hours + 200 * 15 / 60)

    frames = []
    for idx, asset in enumerate(sorted(assets)):
        try:
            tech_df = build_tech_features(data_src, asset, tail_rows=2500)
            tech_df = _ensure_datetime_col(tech_df, "timestamp")

            if need_mtf:
                pre_cols = len(tech_df.columns)
                try:
                    tech_df = add_multi_timeframe_features(tech_df, asset)
                    mtf_n = len(tech_df.columns) - pre_cols
                    if mtf_n > 0:
                        logger.debug(f"  {asset}: +{mtf_n} mtf features")
                except Exception as e:
                    logger.warning(f"  {asset}: MTF 特征构建失败 ({e})，跳过该 asset 避免用零值训练")
                    continue

            if "log_return" not in tech_df.columns:
                tech_df["log_return"] = np.log(
                    tech_df["close"] / tech_df["close"].shift(1)
                )

            funding_path = paths.get("funding_path", "")
            if Path(funding_path).exists():
                tech_df = merge_funding_rate(tech_df, funding_path, asset)

            ohlcv_df = load_ohlcv(data_src, asset)
            if "timestamp_ms" not in tech_df.columns and "timestamp_ms" in ohlcv_df.columns:
                tech_df = pd.merge_asof(
                    tech_df.sort_values("timestamp"),
                    ohlcv_df[["timestamp", "timestamp_ms"]].sort_values("timestamp"),
                    on="timestamp",
                    direction="nearest",
                    tolerance=pd.Timedelta("1min"),
                )

            for c in GRU_FEATURE_COLS:
                if c not in tech_df.columns:
                    tech_df[c] = 0.0

            model_path = gru_model_dir / f"{asset}_enhanced.pt"
            norm_path = gru_model_dir / f"{asset}_enhanced_normalizer.json"
            if model_path.exists():
                emb_df = extract_embeddings(tech_df, model_path, norm_path, device)
            else:
                emb_df = pd.DataFrame({"timestamp_ms": tech_df["timestamp"]})

            prepared = _prepare_asset_for_pooling(
                tech_df, emb_df, feature_groups, paths, asset, asset_id=idx,
            )
            frames.append(prepared)
        except Exception as e:
            logger.warning(f"  跳过 {asset}: {e}")
            continue

    if not frames:
        return None

    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled.sort_values("timestamp").reset_index(drop=True)

    cutoff_ts = pd.Timestamp(cutoff)
    upper_ts = pd.Timestamp(now)
    if pooled["timestamp"].dt.tz is None:
        cutoff_ts = cutoff_ts.tz_localize(None)
        upper_ts = upper_ts.tz_localize(None) if upper_ts.tzinfo is not None else upper_ts
    elif cutoff_ts.tzinfo is None:
        cutoff_ts = cutoff_ts.tz_localize("UTC")
        upper_ts = upper_ts.tz_localize("UTC") if upper_ts.tzinfo is None else upper_ts
    recent = pooled[(pooled["timestamp"] >= cutoff_ts) & (pooled["timestamp"] <= upper_ts)].reset_index(drop=True)

    logger.info(f"  数据: {len(pooled)} 总行, {len(recent)} 近期行 ({hours}h, as_of={now.isoformat()})")
    return recent


# ─── 增量训练核心 ────────────────────────────────────────────

def _incremental_train_one_model(
    model_path: Path,
    X_new: pd.DataFrame,
    y_new: pd.Series,
    sample_weight: Optional[pd.Series] = None,
    feature_cols: Optional[List[str]] = None,
    num_boost_round: int = BASE_NUM_BOOST_ROUND,
    regime: str = "normal",
    rollback_auc_drop_abs: float = ROLLBACK_AUC_DROP_ABS,
    rollback_auc_drop_rel: float = ROLLBACK_AUC_DROP_REL,
    rollback_utility_drop_abs: float = ROLLBACK_UTILITY_DROP_ABS,
    rollback_down_wr_drop_abs: float = ROLLBACK_DOWN_WR_DROP_ABS,
    trade_objective_priority: bool = False,
    trade_objective_auc_catastrophic_mult: float = TRADE_OBJECTIVE_AUC_CATASTROPHIC_MULT,
    utility_min_confidence: float = UTILITY_MIN_CONFIDENCE,
    utility_min_samples: int = UTILITY_MIN_SAMPLES,
    utility_min_down_samples: int = UTILITY_MIN_DOWN_SAMPLES,
    slot01_positive_writeback: bool = False,
    dry_run: bool = False,
    defer_writeback: bool = False,
) -> Dict:
    """
    对单个 LightGBM joblib 模型执行增量训练。

    Returns dict with keys:
        success, old_trees, new_trees, old_auc, new_auc, model_path, backed_up_to
    """
    result = {
        "success": False,
        "training_completed": False,
        "model_path": str(model_path),
        "old_trees": 0,
        "new_trees": 0,
        "old_auc": None,
        "new_auc": None,
        "backed_up_to": None,
        "skipped_reason": None,
        "num_boost_round": int(num_boost_round),
        "regime": regime,
        "auc_warning": None,
        "utility_warning": None,
        "trade_objective_priority": bool(trade_objective_priority),
        "trade_objective_ready": False,
        "auc_softened_by_trade_objective": False,
        "validation_support_status": "not_evaluated",
        "utility_sample_threshold": int(min(int(utility_min_samples), 20)),
        "down_sample_threshold": int(min(int(utility_min_down_samples), 8)),
        "veto_due_to_threshold_only": False,
        "old_utility_n": None,
        "new_utility_n": None,
        "old_utility_wr": None,
        "new_utility_wr": None,
        "old_utility_pnl": None,
        "new_utility_pnl": None,
        "old_down_n": None,
        "new_down_n": None,
        "old_down_wr": None,
        "new_down_wr": None,
        "metrics_ready_for_writeback": False,
        "writeback_allowed": False,
        "trained_but_not_written": False,
        "writeback_veto_reason": None,
        "strict_improvement": False,
        "family_atomic_mode": bool(defer_writeback),
        "family_atomic_pending": False,
        "family_atomic_committed": False,
        "family_atomic_veto_reason": None,
        "slot01_positive_writeback_mode": bool(slot01_positive_writeback),
        "slot01_positive_evaluation_baseline": "slot01_current_live_contract" if slot01_positive_writeback else "",
        "slot01_positive_evaluation_scope": "slot01_only_current_live_contract" if slot01_positive_writeback else "",
        "slot01_positive_gate_semantics": (
            "challenger_vs_current_slot01_live_contract_proxy_metrics"
            if slot01_positive_writeback
            else ""
        ),
        "slot01_positive_gain": False,
        "slot01_positive_gain_reasons": [],
        "slot01_observation_only": False,
    }

    try:
        old_model = joblib.load(model_path)
    except Exception as e:
        result["skipped_reason"] = f"加载失败: {e}"
        return result

    old_booster = old_model.booster_
    result["old_trees"] = old_booster.num_trees()

    if feature_cols:
        X_all = X_new[feature_cols]
    else:
        X_all = X_new
    y_all = y_new

    if len(y_all) < MIN_SAMPLES:
        result["skipped_reason"] = f"样本不足: {len(y_all)} < {MIN_SAMPLES}"
        return result

    # Time-based split: 前 80% 训练，后 20% 验证（按时间顺序，无 shuffle）
    val_ratio = 0.20
    split_idx = int(len(X_all) * (1 - val_ratio))
    X_train, X_val = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
    y_train, y_val = y_all.iloc[:split_idx], y_all.iloc[split_idx:]
    sw_train = sample_weight.iloc[:split_idx] if sample_weight is not None else None

    old_proba_val = old_model.predict_proba(X_val)[:, 1]
    try:
        result["old_auc"] = round(float(roc_auc_score(y_val, old_proba_val)), 5)
    except ValueError:
        result["old_auc"] = None
    old_utility = _calc_validation_trade_utility(y_val, old_proba_val, utility_min_confidence)
    result["old_utility_n"] = old_utility.get("n")
    result["old_utility_wr"] = old_utility.get("wr")
    result["old_utility_pnl"] = old_utility.get("pnl_proxy")
    result["old_down_n"] = old_utility.get("down_n")
    result["old_down_wr"] = old_utility.get("down_wr")
    result["_val_y"] = y_val.to_numpy(dtype=int)
    result["_old_val_proba"] = np.asarray(old_proba_val, dtype=float)

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp_model_path = tmp.name
    old_booster.save_model(tmp_model_path)

    params = old_booster.params.copy() if hasattr(old_booster, "params") else {}
    params.update({
        "learning_rate": INCREMENTAL_LR,
        "bagging_fraction": BAGGING_FRACTION,
        "bagging_freq": 1,
        "verbosity": -1,
    })
    if PARAM_SEARCH_NUMERIC_THREADS > 0:
        params["num_threads"] = int(PARAM_SEARCH_NUMERIC_THREADS)
    if "num_iterations" in params:
        del params["num_iterations"]
    if "num_trees" in params:
        del params["num_trees"]

    cat_features = []
    if "asset_id" in X_train.columns:
        cat_features = ["asset_id"]

    train_set = lgb.Dataset(
        X_train,
        label=y_train,
        weight=sw_train.values if sw_train is not None else None,
        categorical_feature=cat_features if cat_features else "auto",
        free_raw_data=False,
    )

    use_rounds = max(1, int(num_boost_round))
    new_booster = lgb.train(
        params,
        train_set,
        num_boost_round=use_rounds,
        init_model=tmp_model_path,
    )

    Path(tmp_model_path).unlink(missing_ok=True)
    result["new_trees"] = new_booster.num_trees()

    new_proba_val = new_booster.predict(X_val)
    try:
        result["new_auc"] = round(float(roc_auc_score(y_val, new_proba_val)), 5)
    except ValueError:
        result["new_auc"] = None
    new_utility = _calc_validation_trade_utility(y_val, new_proba_val, utility_min_confidence)
    result["new_utility_n"] = new_utility.get("n")
    result["new_utility_wr"] = new_utility.get("wr")
    result["new_utility_pnl"] = new_utility.get("pnl_proxy")
    result["new_down_n"] = new_utility.get("down_n")
    result["new_down_wr"] = new_utility.get("down_wr")
    result["_new_val_proba"] = np.asarray(new_proba_val, dtype=float)
    result["training_completed"] = True

    old_utility_n = float(old_utility.get("n") or 0.0)
    new_utility_n = float(new_utility.get("n") or 0.0)
    old_utility_pnl = old_utility.get("pnl_proxy")
    new_utility_pnl = new_utility.get("pnl_proxy")
    utility_sample_threshold = float(min(int(utility_min_samples), 20))
    down_sample_threshold = float(min(int(utility_min_down_samples), 8))
    utility_ready = (
        old_utility_n >= utility_sample_threshold
        and new_utility_n >= utility_sample_threshold
        and old_utility_pnl is not None
        and new_utility_pnl is not None
    )
    result["trade_objective_ready"] = bool(utility_ready)

    old_down_n = float(old_utility.get("down_n") or 0.0)
    new_down_n = float(new_utility.get("down_n") or 0.0)
    old_down_wr = old_utility.get("down_wr")
    new_down_wr = new_utility.get("down_wr")
    down_ready = (
        old_down_n >= down_sample_threshold
        and new_down_n >= down_sample_threshold
        and old_down_wr is not None
        and new_down_wr is not None
    )
    result["trade_objective_ready"] = bool(result["trade_objective_ready"] or down_ready)
    auc_ready = result["old_auc"] is not None and result["new_auc"] is not None
    if not auc_ready:
        result["validation_support_status"] = "auc_not_ready"
    elif utility_ready and down_ready:
        result["validation_support_status"] = "ready"
    elif utility_ready:
        result["validation_support_status"] = "auc_and_utility_ready"
    elif down_ready:
        result["validation_support_status"] = "auc_and_down_ready"
    else:
        result["validation_support_status"] = "auc_only_ready"
        result["veto_due_to_threshold_only"] = True
    result["metrics_ready_for_writeback"] = bool(auc_ready)
    if not result["metrics_ready_for_writeback"] and not defer_writeback:
        result["trained_but_not_written"] = True
        result["writeback_veto_reason"] = "insufficient_validation_support"
        result["skipped_reason"] = "训练完成但验证支持不足，禁止写盘"
        return result

    regressions = []
    improvements = []
    if utility_ready and old_utility_pnl is not None and new_utility_pnl is not None and float(new_utility_pnl) < float(old_utility_pnl):
        regressions.append("utility_regression")
        result["utility_warning"] = f"pnl_proxy {float(old_utility_pnl):.5f} -> {float(new_utility_pnl):.5f}"
    elif utility_ready and old_utility_pnl is not None and new_utility_pnl is not None and float(new_utility_pnl) > float(old_utility_pnl):
        improvements.append("utility_pnl")

    if down_ready and old_down_wr is not None and new_down_wr is not None and float(new_down_wr) < float(old_down_wr):
        regressions.append("down_wr_regression")
    elif down_ready and old_down_wr is not None and new_down_wr is not None and float(new_down_wr) > float(old_down_wr):
        improvements.append("down_wr")

    if auc_ready and float(result["new_auc"]) < float(result["old_auc"]):
        regressions.append("auc_regression")
        result["auc_warning"] = f"{result['old_auc']} -> {result['new_auc']}"
    elif auc_ready and float(result["new_auc"]) > float(result["old_auc"]):
        improvements.append("auc")

    if trade_objective_priority and "auc_regression" in regressions:
        old_auc = float(result["old_auc"])
        new_auc = float(result["new_auc"])
        auc_drop = max(0.0, old_auc - new_auc)
        catastrophic_auc_drop = max(
            float(rollback_auc_drop_abs),
            abs(old_auc) * float(rollback_auc_drop_rel),
        ) * float(trade_objective_auc_catastrophic_mult)
        result["trade_objective_auc_drop"] = auc_drop
        result["trade_objective_auc_catastrophic_threshold"] = catastrophic_auc_drop

    result["local_regressions"] = list(regressions)
    result["local_improvements"] = list(improvements)

    if defer_writeback:
        # family 模式下不在单个 window 维度直接卡死，统一交给 family 集成效果决策。
        result["strict_improvement"] = bool(improvements)
        result["writeback_allowed"] = bool(auc_ready and result["training_completed"])
    elif slot01_positive_writeback:
        # 一号槽位专属增量门：只看一号槽位当前模型的正向收益，不再让其他 family/window 联合否决。
        # 这里先用验证集交易效用代理做训练写回门；真钱上线前仍由后续长窗/运行时核验报告兜底。
        reasons: list[str] = []
        pass_gate = True
        if not utility_ready:
            pass_gate = False
            reasons.append("交易收益样本不足，不能证明一号槽位正增益")
        elif old_utility_pnl is None or new_utility_pnl is None or float(new_utility_pnl) <= float(old_utility_pnl):
            pass_gate = False
            reasons.append("交易收益代理未改善")
        else:
            reasons.append("交易收益代理改善")

        old_wr = old_utility.get("wr")
        new_wr = new_utility.get("wr")
        if old_wr is not None and new_wr is not None:
            wr_drop = float(old_wr) - float(new_wr)
            if wr_drop > float(rollback_down_wr_drop_abs):
                pass_gate = False
                reasons.append(f"总体胜率下降过大:{wr_drop:.5f}")
            elif wr_drop > 0:
                reasons.append(f"总体胜率轻微下降:{wr_drop:.5f}")

        if down_ready and old_down_wr is not None and new_down_wr is not None:
            down_drop = float(old_down_wr) - float(new_down_wr)
            if down_drop > float(rollback_down_wr_drop_abs):
                pass_gate = False
                reasons.append(f"向下胜率下降过大:{down_drop:.5f}")
            elif down_drop > 0:
                reasons.append(f"向下胜率轻微下降:{down_drop:.5f}")

        if auc_ready and result["old_auc"] is not None and result["new_auc"] is not None:
            old_auc = float(result["old_auc"])
            new_auc = float(result["new_auc"])
            auc_drop = max(0.0, old_auc - new_auc)
            catastrophic_auc_drop = max(
                float(rollback_auc_drop_abs),
                abs(old_auc) * float(rollback_auc_drop_rel),
            ) * float(trade_objective_auc_catastrophic_mult)
            result["trade_objective_auc_drop"] = auc_drop
            result["trade_objective_auc_catastrophic_threshold"] = catastrophic_auc_drop
            if auc_drop > catastrophic_auc_drop:
                pass_gate = False
                reasons.append(f"基础识别能力灾难性下降:{auc_drop:.5f}")
            elif auc_drop > 0:
                reasons.append(f"基础识别能力轻微下降:{auc_drop:.5f}")

        result["slot01_positive_gain"] = bool(pass_gate)
        result["slot01_positive_gain_reasons"] = reasons
        result["slot01_observation_only"] = not bool(pass_gate)
        if not pass_gate:
            result["trained_but_not_written"] = True
            result["writeback_veto_reason"] = "slot01_positive_gate:" + ";".join(reasons)
            result["skipped_reason"] = "一号槽位正增益门未通过，进入观察但不写盘"
            return result

        result["strict_improvement"] = True
        result["writeback_allowed"] = True
    else:
        if regressions:
            result["trained_but_not_written"] = True
            result["writeback_veto_reason"] = ",".join(regressions)
            result["skipped_reason"] = "写盘 veto: " + ", ".join(regressions)
            return result

        if not improvements:
            result["trained_but_not_written"] = True
            result["writeback_veto_reason"] = "no_strict_improvement"
            result["skipped_reason"] = "训练完成但无严格改进，禁止写盘"
            return result

        result["strict_improvement"] = True
        result["writeback_allowed"] = True

    if dry_run:
        result["success"] = True
        result["skipped_reason"] = "dry-run, 通过写盘门但未写入"
        return result

    new_model = copy.deepcopy(old_model)
    new_model._Booster = new_booster
    try:
        new_model.n_estimators = new_booster.num_trees()
    except (AttributeError, TypeError):
        pass
    try:
        new_model._best_iteration = new_booster.num_trees()
    except (AttributeError, TypeError):
        pass

    if defer_writeback:
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as staged_tmp:
            staged_model_path = staged_tmp.name
        joblib.dump(new_model, staged_model_path)
        result["family_atomic_pending"] = True
        result["_staged_model_path"] = staged_model_path
        result["skipped_reason"] = "等待 family 原子提交"
        return result

    today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    backup_dir = BACKUP_DIR / model_path.parent.name
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{model_path.stem}_{today_str}.joblib"
    shutil.copy2(model_path, backup_path)
    result["backed_up_to"] = str(backup_path)

    joblib.dump(new_model, model_path)
    result["success"] = True
    return result


def _calc_family_validation_metrics(
    y_val: np.ndarray,
    proba_up: np.ndarray,
    utility_min_confidence: float,
) -> Dict[str, Optional[float]]:
    metrics = _calc_validation_trade_utility(
        pd.Series(np.asarray(y_val, dtype=int)),
        np.asarray(proba_up, dtype=float),
        utility_min_confidence,
    )
    try:
        auc = float(roc_auc_score(np.asarray(y_val, dtype=int), np.asarray(proba_up, dtype=float)))
    except ValueError:
        auc = None
    metrics["auc"] = auc
    return metrics


def _family_metric_tuple(metrics: Dict[str, Optional[float]]) -> tuple:
    return (
        float(metrics.get("pnl_proxy") or -1e9),
        float(metrics.get("down_wr") or -1e9),
        float(metrics.get("auc") or -1e9),
    )


def _finalize_family_atomic_writeback(
    family_results: List[Dict],
    *,
    dry_run: bool = False,
    rollback_auc_drop_abs: float = ROLLBACK_AUC_DROP_ABS,
    rollback_auc_drop_rel: float = ROLLBACK_AUC_DROP_REL,
    utility_min_confidence: float = UTILITY_MIN_CONFIDENCE,
    utility_min_samples: int = UTILITY_MIN_SAMPLES,
    utility_min_down_samples: int = UTILITY_MIN_DOWN_SAMPLES,
) -> Tuple[List[Dict], bool]:
    """对同一 exp family 做联合判定，并原子提交被批准的 window 子集。"""
    if not family_results:
        return family_results, False

    atomic_rows = [row for row in family_results if row.get("family_atomic_mode")]
    if not atomic_rows:
        return family_results, any(bool(row.get("success")) for row in family_results)

    required_rows = [
        row
        for row in atomic_rows
        if row.get("training_completed")
        or row.get("trained_but_not_written")
        or row.get("writeback_allowed")
        or row.get("success")
        or row.get("skipped_reason")
    ]
    if dry_run:
        # Recommendation/search paths only need the validation predictions to decide
        # whether a family-level candidate would have been selected. Requiring a
        # staged model here forces LightGBM model serialization that is immediately
        # discarded, and made nightly param search appear hung during finalist
        # evaluation.
        candidate_rows = [
            row
            for row in required_rows
            if row.get("writeback_allowed")
            and row.get("_new_val_proba") is not None
            and row.get("_old_val_proba") is not None
        ]
    else:
        candidate_rows = [
            row
            for row in required_rows
            if row.get("family_atomic_pending")
            and str(row.get("_staged_model_path") or "").strip()
            and row.get("_new_val_proba") is not None
            and row.get("_old_val_proba") is not None
        ]
    y_val = None
    utility_min_confidence = UTILITY_MIN_CONFIDENCE
    old_cols: List[np.ndarray] = []
    for row in required_rows:
        if y_val is None and row.get("_val_y") is not None:
            y_val = np.asarray(row.get("_val_y"), dtype=int)
        if row.get("_old_val_proba") is not None:
            old_cols.append(np.asarray(row.get("_old_val_proba"), dtype=float))
    if y_val is None or not old_cols:
        candidate_rows = []

    family_utility_sample_threshold = float(min(int(utility_min_samples), 20))
    family_down_sample_threshold = float(min(int(utility_min_down_samples), 8))

    def _family_metric_state(metrics: Dict[str, Optional[float]]) -> Dict[str, bool]:
        try:
            n = float(metrics.get("n") or 0.0)
            down_n = float(metrics.get("down_n") or 0.0)
        except Exception:
            n = 0.0
            down_n = 0.0
        return {
            "auc_ready": metrics.get("auc") is not None,
            "utility_ready": metrics.get("pnl_proxy") is not None and n >= family_utility_sample_threshold,
            "down_ready": metrics.get("down_wr") is not None and down_n >= family_down_sample_threshold,
        }

    family_reason = "family_atomic_veto:no_family_candidate_selected"
    selected_rows: List[Dict] = []
    if candidate_rows and y_val is not None:
        ordered_rows = sorted(required_rows, key=lambda row: int(row.get("window") or 0))
        row_to_idx = {id(row): idx for idx, row in enumerate(ordered_rows)}
        old_stack = np.column_stack([np.asarray(row.get("_old_val_proba"), dtype=float) for row in ordered_rows])
        base_proba = np.mean(old_stack, axis=1)
        base_metrics = _calc_family_validation_metrics(y_val, base_proba, utility_min_confidence)
        base_state = _family_metric_state(base_metrics)
        base_auc = float(base_metrics.get("auc") or 0.0)
        auc_tolerance = max(float(rollback_auc_drop_abs), abs(base_auc) * float(rollback_auc_drop_rel))
        best_choice: Optional[List[Dict]] = None
        best_score: Optional[tuple] = None
        best_metrics: Optional[Dict[str, Optional[float]]] = None

        for row in required_rows:
            row["family_base_metrics"] = dict(base_metrics)
            row["family_selected_metrics"] = None
            row["family_selected_windows"] = []
            row["family_support_ready"] = bool(base_state.get("auc_ready"))
            row["family_support_thresholds"] = {
                "utility_min_samples": int(family_utility_sample_threshold),
                "down_sample_threshold": int(family_down_sample_threshold),
            }

        if not base_state.get("auc_ready"):
            family_reason = "family_atomic_veto:family_auc_not_ready"

        if base_state.get("auc_ready"):
            for subset_size in range(1, len(candidate_rows) + 1):
                for subset in itertools.combinations(candidate_rows, subset_size):
                    candidate_stack = old_stack.copy()
                    for row in subset:
                        candidate_stack[:, row_to_idx[id(row)]] = np.asarray(row.get("_new_val_proba"), dtype=float)
                    candidate_proba = np.mean(candidate_stack, axis=1)
                    metrics = _calc_family_validation_metrics(y_val, candidate_proba, utility_min_confidence)
                    metrics_state = _family_metric_state(metrics)
                    if not metrics_state.get("auc_ready"):
                        continue
                    utility_ready = bool(base_state.get("utility_ready") and metrics_state.get("utility_ready"))
                    down_ready = bool(base_state.get("down_ready") and metrics_state.get("down_ready"))
                    if utility_ready and float(metrics["pnl_proxy"]) < float(base_metrics["pnl_proxy"]):
                        continue
                    if down_ready and float(metrics["down_wr"]) < float(base_metrics["down_wr"]):
                        continue
                    if float(metrics["auc"]) < (float(base_metrics["auc"]) - auc_tolerance):
                        continue
                    utility_delta = (
                        float(metrics["pnl_proxy"]) - float(base_metrics["pnl_proxy"])
                        if utility_ready
                        else None
                    )
                    down_delta = (
                        float(metrics["down_wr"]) - float(base_metrics["down_wr"])
                        if down_ready
                        else None
                    )
                    auc_delta = float(metrics["auc"]) - float(base_metrics["auc"])
                    improvement_flags = [auc_delta > 0.0]
                    if utility_delta is not None:
                        improvement_flags.append(utility_delta > 0.0)
                    if down_delta is not None:
                        improvement_flags.append(down_delta > 0.0)
                    if not any(improvement_flags):
                        continue
                    score = (
                        utility_delta if utility_delta is not None else -1e-9,
                        down_delta if down_delta is not None else -1e-9,
                        auc_delta,
                        -len(subset),
                    )
                    if best_score is None or score > best_score:
                        best_score = score
                        best_choice = list(subset)
                        best_metrics = dict(metrics)
            if best_choice:
                selected_rows = best_choice
                family_reason = None
            else:
                family_reason = "family_atomic_veto:no_family_improvement"

        if best_metrics is not None:
            selected_windows = sorted(int(row.get("window") or 0) for row in selected_rows)
            for row in required_rows:
                row["family_selected_metrics"] = dict(best_metrics)
                row["family_selected_windows"] = list(selected_windows)

    selected_ids = {id(row) for row in selected_rows}

    if selected_rows:
        if dry_run:
            for row in selected_rows:
                row["success"] = True
                row["family_atomic_pending"] = False
                row["family_atomic_committed"] = True
                row["family_atomic_veto_reason"] = None
                row["skipped_reason"] = "dry-run, family 联合写盘通过"
            for row in required_rows:
                if id(row) in selected_ids:
                    continue
                staged_path = str(row.get("_staged_model_path") or "").strip()
                if staged_path:
                    Path(staged_path).unlink(missing_ok=True)
                row["success"] = False
                row["writeback_allowed"] = False
                row["trained_but_not_written"] = True
                row["family_atomic_pending"] = False
                row["family_atomic_committed"] = False
                if not row.get("writeback_veto_reason"):
                    row["writeback_veto_reason"] = "family_atomic_not_selected"
                row["family_atomic_veto_reason"] = str(row.get("writeback_veto_reason") or "family_atomic_not_selected")
                row["skipped_reason"] = "family 联合判定未选中"
            return family_results, True

        committed_dirs: set[str] = set()
        today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        commit_targets = []
        for row in selected_rows:
            model_path = Path(str(row.get("model_path") or ""))
            staged_model_path = Path(str(row.get("_staged_model_path") or ""))
            if not model_path.exists() or not staged_model_path.exists():
                family_reason = "family_atomic_veto:missing_staged_model"
                selected_rows = []
                selected_ids = set()
                break
            backup_dir = BACKUP_DIR / model_path.parent.name
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup_path = backup_dir / f"{model_path.stem}_{today_str}.joblib"
            commit_targets.append((row, model_path, staged_model_path, backup_path))
        if selected_rows:
            committed_so_far: List[Tuple[Path, Path]] = []
            try:
                for _, model_path, _, backup_path in commit_targets:
                    shutil.copy2(model_path, backup_path)
                for row, model_path, staged_model_path, backup_path in commit_targets:
                    shutil.move(str(staged_model_path), str(model_path))
                    row["backed_up_to"] = str(backup_path)
                    row["success"] = True
                    row["family_atomic_pending"] = False
                    row["family_atomic_committed"] = True
                    row["family_atomic_veto_reason"] = None
                    row["skipped_reason"] = None
                    committed_dirs.add(str(model_path.parent))
                    committed_so_far.append((model_path, backup_path))
            except Exception:
                for model_path, backup_path in committed_so_far:
                    if backup_path.exists():
                        shutil.copy2(backup_path, model_path)
                family_reason = "family_atomic_veto:commit_rollback"
                selected_rows = []
                selected_ids = set()
        if selected_rows:
            for row in required_rows:
                if id(row) in selected_ids:
                    continue
                staged_path = str(row.get("_staged_model_path") or "").strip()
                if staged_path:
                    Path(staged_path).unlink(missing_ok=True)
                row["success"] = False
                row["writeback_allowed"] = False
                row["trained_but_not_written"] = True
                row["family_atomic_pending"] = False
                row["family_atomic_committed"] = False
                if not row.get("writeback_veto_reason"):
                    row["writeback_veto_reason"] = "family_atomic_not_selected"
                row["family_atomic_veto_reason"] = str(row.get("writeback_veto_reason") or "family_atomic_not_selected")
                row["skipped_reason"] = "family 联合判定未选中"
            return family_results, bool(committed_dirs)

    for row in required_rows:
        staged_path = str(row.get("_staged_model_path") or "").strip()
        if staged_path:
            Path(staged_path).unlink(missing_ok=True)
        row["success"] = False
        row["family_atomic_pending"] = False
        row["family_atomic_committed"] = False
        row["family_atomic_veto_reason"] = family_reason
        row["writeback_allowed"] = False
        row["trained_but_not_written"] = True
        base_reason = str(row.get("writeback_veto_reason") or "").strip()
        if not base_reason:
            row["writeback_veto_reason"] = family_reason
        elif family_reason not in base_reason:
            row["writeback_veto_reason"] = f"{base_reason},{family_reason}"
        row["skipped_reason"] = "family 联合判定未通过"
    return family_results, False


# ─── 备份清理 ────────────────────────────────────────────────

def _cleanup_old_backups():
    """删除超过 BACKUP_KEEP_DAYS 天的备份文件。"""
    if not BACKUP_DIR.exists():
        return
    cutoff = datetime.now(timezone.utc).timestamp() - BACKUP_KEEP_DAYS * 86400
    removed = 0
    for f in BACKUP_DIR.rglob("*.joblib"):
        if f.stat().st_mtime < cutoff:
            f.unlink()
            removed += 1
    if removed > 0:
        logger.info(f"  清理 {removed} 个过期备份 (>{BACKUP_KEEP_DAYS}天)")


def _bootstrap_model_dir_if_needed(target_dir: Path, source_dir: Path) -> bool:
    """
    当 target 缺少 config.json 时，从 source 引导基础文件。
    返回是否完成了有效引导。
    """
    try:
        source_dir = source_dir.resolve()
    except Exception:
        pass
    if not source_dir.exists():
        return False
    source_cfg = source_dir / "config.json"
    if not source_cfg.exists():
        return False

    target_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for name in ("config.json", "feature_cols.json"):
        src = source_dir / name
        dst = target_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
    for src in source_dir.glob("lgb_*d.joblib"):
        dst = target_dir / src.name
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
    return copied > 0


# ─── 主流程 ──────────────────────────────────────────────────

def run_incremental(
    exp_ids: Optional[List[int]] = None,
    dry_run: bool = False,
    model_dir_map_json: Optional[str] = None,
    bootstrap_model_dir_map_json: Optional[str] = None,
    hours: int = 48,
    base_rounds: int = BASE_NUM_BOOST_ROUND,
    calm_rounds: int = CALM_NUM_BOOST_ROUND,
    shock_rounds: int = SHOCK_NUM_BOOST_ROUND,
    recency_halflife_bars: int = RECENCY_HALFLIFE_BARS,
    shock_weight_mult: float = SHOCK_SAMPLE_WEIGHT_MULT,
    calm_vol_ratio: float = CALM_VOL_RATIO,
    shock_vol_ratio: float = SHOCK_VOL_RATIO,
    shock_ret_q: float = SHOCK_RET_Q,
    rollback_auc_drop_abs: float = ROLLBACK_AUC_DROP_ABS,
    rollback_auc_drop_rel: float = ROLLBACK_AUC_DROP_REL,
    rollback_utility_drop_abs: float = ROLLBACK_UTILITY_DROP_ABS,
    rollback_down_wr_drop_abs: float = ROLLBACK_DOWN_WR_DROP_ABS,
    trade_objective_priority: bool = False,
    trade_objective_auc_catastrophic_mult: float = TRADE_OBJECTIVE_AUC_CATASTROPHIC_MULT,
    utility_min_confidence: float = UTILITY_MIN_CONFIDENCE,
    utility_min_samples: int = UTILITY_MIN_SAMPLES,
    utility_min_down_samples: int = UTILITY_MIN_DOWN_SAMPLES,
    history_job_id: Optional[str] = None,
    history_mode: Optional[str] = None,
    history_family: Optional[str] = None,
    history_asset: Optional[str] = None,
    family_atomic_writeback: bool = False,
    slot01_positive_writeback: bool = False,
    as_of_timestamp: Optional[str] = None,
    history_timestamp: Optional[str] = None,
):
    """对指定 Exp 模型执行增量训练。"""
    as_of_dt = parse_dt(as_of_timestamp) if as_of_timestamp else None
    history_dt = parse_dt(history_timestamp) if history_timestamp else None
    models_base = PROJECT_ROOT / "data" / "models"
    custom_model_dirs: Dict[int, Path] = {}
    bootstrap_model_dirs: Dict[int, Path] = {}

    if model_dir_map_json:
        raw_map = json.loads(Path(model_dir_map_json).read_text(encoding="utf-8"))
        if not isinstance(raw_map, dict):
            raise ValueError("--model-dir-map-json 必须是 exp_id -> model_dir 的 JSON 对象")
        for key, value in raw_map.items():
            exp_key = int(str(key))
            model_path = Path(str(value))
            if not model_path.is_absolute():
                model_path = (PROJECT_ROOT / model_path).resolve()
            custom_model_dirs[exp_key] = model_path

    if bootstrap_model_dir_map_json:
        raw_bootstrap = json.loads(Path(bootstrap_model_dir_map_json).read_text(encoding="utf-8"))
        if not isinstance(raw_bootstrap, dict):
            raise ValueError("--bootstrap-model-dir-map-json 必须是 exp_id -> source_model_dir 的 JSON 对象")
        for key, value in raw_bootstrap.items():
            exp_key = int(str(key))
            src_path = Path(str(value))
            if not src_path.is_absolute():
                src_path = (PROJECT_ROOT / src_path).resolve()
            bootstrap_model_dirs[exp_key] = src_path

    if exp_ids is None:
        exp_ids = sorted(set(EXP_MODEL_DIRS.keys()) | set(custom_model_dirs.keys()))

    device = str(get_device())
    logger.info(f"设备: {device}")
    logger.info(f"目标实验: {exp_ids}")

    all_results = []
    success_dirs: set[str] = set()

    family_atomic_writeback = bool(
        (not slot01_positive_writeback)
        and (family_atomic_writeback or str(history_mode or "").strip() in {"core10_only", "mainline_runtime_pool"})
    )

    for exp_id in exp_ids:
        model_dir = custom_model_dirs.get(exp_id)
        dir_name = ""
        if model_dir is None:
            dir_name = EXP_MODEL_DIRS.get(exp_id, "")
            if not dir_name:
                logger.warning(f"Exp{exp_id}: 未知的模型目录映射")
                continue
            model_dir = models_base / dir_name
        else:
            try:
                dir_name = str(model_dir.relative_to(models_base))
            except ValueError:
                dir_name = model_dir.name
        config_path = model_dir / "config.json"
        feature_cols_path = model_dir / "feature_cols.json"

        if not config_path.exists():
            source_dir = bootstrap_model_dirs.get(exp_id)
            if source_dir is None and exp_id in EXP_MODEL_DIRS:
                source_dir = (models_base / EXP_MODEL_DIRS[exp_id]).resolve()
            if source_dir is not None:
                logger.warning(f"Exp{exp_id}: config.json 不存在，尝试 bootstrap: target={model_dir} source={source_dir}")
                ok = _bootstrap_model_dir_if_needed(model_dir, source_dir)
                if ok:
                    logger.info(f"Exp{exp_id}: bootstrap 成功")
                else:
                    logger.warning(f"Exp{exp_id}: bootstrap 失败，跳过")
            if not config_path.exists():
                logger.warning(f"Exp{exp_id}: config.json 不存在 ({model_dir})")
                continue

        logger.info(f"\n{'─' * 50}")
        logger.info(f"  Exp{exp_id} ({dir_name})")
        logger.info(f"{'─' * 50}")

        with open(config_path) as f:
            config = json.load(f)

        feature_cols = None
        if feature_cols_path.exists():
            with open(feature_cols_path) as f:
                feature_cols = json.load(f)

        need_mtf = feature_cols is not None and any(c.startswith("mtf_") for c in feature_cols)

        window_days_list = config.get("window_days_list", [60, 90, 120])

        try:
            recent_data = _build_recent_data(config, hours=hours, device=device, need_mtf=need_mtf, as_of=as_of_dt)
        except Exception as e:
            logger.exception(f"  Exp{exp_id}: 构建近期数据失败: {e}")
            all_results.append(
                {
                    "exp": exp_id,
                    "window": None,
                    "success": False,
                    "skipped_reason": f"build_recent_data 失败: {e}",
                }
            )
            continue
        if recent_data is None or len(recent_data) < MIN_SAMPLES:
            logger.warning(f"  Exp{exp_id}: 数据不足, 跳过")
            continue

        if feature_cols:
            avail_cols = [c for c in feature_cols if c in recent_data.columns]
            missing = set(feature_cols) - set(avail_cols)
            if missing:
                logger.info(f"  缺失 {len(missing)} 个特征, 填充0")
                for c in missing:
                    recent_data[c] = 0.0

        adaptive_w, adaptive_info = _build_adaptive_plan(
            recent_data,
            base_rounds=base_rounds,
            calm_rounds=calm_rounds,
            shock_rounds=shock_rounds,
            recency_halflife_bars=recency_halflife_bars,
            shock_weight_mult=shock_weight_mult,
            calm_vol_ratio=calm_vol_ratio,
            shock_vol_ratio=shock_vol_ratio,
            shock_ret_q=shock_ret_q,
        )

        logger.info(
            "  自适应: regime=%s, vol_ratio=%.2f, rounds=%d, shock_samples=%d",
            adaptive_info["regime"],
            adaptive_info["vol_ratio"],
            int(adaptive_info["num_boost_round"]),
            int(adaptive_info["shock_samples"]),
        )

        X_all = recent_data[feature_cols] if feature_cols else recent_data
        y_all = recent_data["direction_label"]
        sw_raw = recent_data.get("sample_weight")
        if sw_raw is not None:
            sw = pd.to_numeric(sw_raw, errors="coerce").fillna(1.0).clip(lower=0.1, upper=10.0)
        else:
            sw = pd.Series(np.ones(len(recent_data), dtype=float), index=recent_data.index)
        sw = (sw * adaptive_w).astype(float)
        sw_mean = float(sw.mean()) if len(sw) > 0 else 1.0
        if sw_mean > 0:
            sw = sw / sw_mean

        exp_results: List[Dict] = []
        for w_days in window_days_list:
            model_path = model_dir / f"lgb_{w_days}d.joblib"
            if not model_path.exists():
                logger.warning(f"  {model_path.name}: 不存在")
                continue

            logger.info(f"  训练 {model_path.name}...")
            try:
                res = _incremental_train_one_model(
                    model_path=model_path,
                    X_new=X_all,
                    y_new=y_all,
                    sample_weight=sw,
                    feature_cols=feature_cols,
                    num_boost_round=int(adaptive_info["num_boost_round"]),
                    regime=str(adaptive_info["regime"]),
                    rollback_auc_drop_abs=rollback_auc_drop_abs,
                    rollback_auc_drop_rel=rollback_auc_drop_rel,
                    rollback_utility_drop_abs=rollback_utility_drop_abs,
                    rollback_down_wr_drop_abs=rollback_down_wr_drop_abs,
                    trade_objective_priority=trade_objective_priority,
                    trade_objective_auc_catastrophic_mult=trade_objective_auc_catastrophic_mult,
                    utility_min_confidence=utility_min_confidence,
                    utility_min_samples=utility_min_samples,
                    utility_min_down_samples=utility_min_down_samples,
                    slot01_positive_writeback=slot01_positive_writeback,
                    dry_run=dry_run,
                    defer_writeback=family_atomic_writeback,
                )
            except Exception as e:
                logger.exception(f"    ❌ {model_path.name} 增量训练异常: {e}")
                exp_results.append(
                    {
                        "exp": exp_id,
                        "window": w_days,
                        "success": False,
                        "skipped_reason": f"增量训练异常: {e}",
                        "family_atomic_mode": bool(family_atomic_writeback),
                        "family_atomic_pending": False,
                        "family_atomic_committed": False,
                        "family_atomic_veto_reason": None,
                    }
                )
                continue
            exp_results.append({"exp": exp_id, "window": w_days, **res})

        if family_atomic_writeback and exp_results:
            exp_results, committed = _finalize_family_atomic_writeback(
                exp_results,
                dry_run=dry_run,
                rollback_auc_drop_abs=rollback_auc_drop_abs,
                rollback_auc_drop_rel=rollback_auc_drop_rel,
                utility_min_confidence=utility_min_confidence,
                utility_min_samples=utility_min_samples,
                utility_min_down_samples=utility_min_down_samples,
            )
            if committed and not dry_run:
                success_dirs.add(str(model_dir))
                if feature_cols_path.exists():
                    try:
                        feature_cols_path.touch()
                    except OSError as e:
                        logger.warning(f"  touch feature_cols.json 失败: {e}")

        for res in exp_results:
            sanitized = {k: v for k, v in res.items() if not str(k).startswith("_")}
            all_results.append(sanitized)
            if sanitized["success"]:
                if not family_atomic_writeback and not dry_run:
                    success_dirs.add(str(model_dir))
                    if feature_cols_path.exists():
                        try:
                            feature_cols_path.touch()
                        except OSError as e:
                            logger.warning(f"  touch feature_cols.json 失败: {e}")
                logger.info(
                    f"    ✅ {sanitized['old_trees']} → {sanitized['new_trees']} 棵树, "
                    f"AUC: {sanitized['old_auc']} → {sanitized['new_auc']} | rounds={sanitized['num_boost_round']} | regime={sanitized['regime']}"
                    f" | utility(pnl): {sanitized['old_utility_pnl']} → {sanitized['new_utility_pnl']}"
                    f"{' (dry-run)' if dry_run else ''}"
                )
                if sanitized.get("auc_warning"):
                    logger.warning(f"      ⚠️ AUC 小幅回落: {sanitized['auc_warning']}")
                if sanitized.get("utility_warning"):
                    logger.warning(f"      ⚠️ 效用小幅回落: {sanitized['utility_warning']}")
            elif sanitized.get("trained_but_not_written"):
                logger.warning(
                    f"    🛑 训练完成但禁止写盘: {sanitized.get('writeback_veto_reason') or sanitized.get('skipped_reason', '未知')}"
                )
            else:
                logger.warning(f"    ⏭️  跳过: {sanitized.get('skipped_reason', '未知')}")

    _cleanup_old_backups()

    if not dry_run and success_dirs:
        try:
            data = {}
            if LAST_SUCCESS_PATH.exists():
                try:
                    data = json.loads(LAST_SUCCESS_PATH.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    pass
            exp_dir_updated_at: dict[str, str] = {}
            for model_dir_str in sorted(success_dirs):
                model_dir = Path(model_dir_str)
                try:
                    joblibs = list(model_dir.glob("*.joblib"))
                    if not joblibs:
                        continue
                    best_mtime = max(f.stat().st_mtime for f in joblibs)
                    exp_dir_updated_at[str(model_dir)] = datetime.fromtimestamp(
                        best_mtime, timezone.utc
                    ).isoformat()
                except OSError:
                    continue
            data["exp_updated_at"] = datetime.now(timezone.utc).isoformat()
            merged_exp_dir_updated_at = _merge_iso_map(data.get("exp_dir_updated_at"), exp_dir_updated_at)
            cutoff = datetime.now(timezone.utc) - timedelta(hours=LAST_SUCCESS_RETENTION_HOURS)
            merged_exp_dir_updated_at = {
                model_dir: ts
                for model_dir, ts in merged_exp_dir_updated_at.items()
                if parse_dt(ts) is not None
                and parse_dt(ts) >= cutoff
                and Path(model_dir).is_dir()
                and any(Path(model_dir).glob("*.joblib"))
            }
            data["exp_dir_updated_at"] = dict(sorted(merged_exp_dir_updated_at.items()))
            data["exp_dirs"] = _merge_str_list(data.get("exp_dirs"), list(data["exp_dir_updated_at"].keys()))
            data["exp_dirs"] = [model_dir for model_dir in data["exp_dirs"] if model_dir in data["exp_dir_updated_at"]]
            LAST_SUCCESS_PATH.parent.mkdir(parents=True, exist_ok=True)
            LAST_SUCCESS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(f"  已更新 {LAST_SUCCESS_PATH.name}: 本轮新增 exp_dirs={list(success_dirs)}")
        except OSError as e:
            logger.warning(f"  写入 last_success 失败: {e}")

    success_count = sum(1 for r in all_results if r["success"])
    trained_not_written_count = sum(1 for r in all_results if r.get("trained_but_not_written"))
    skip_count = len(all_results) - success_count
    logger.info(f"\n{'═' * 50}")
    logger.info(f"  完成: {success_count} 写盘成功, {trained_not_written_count} 训练完成但未写盘, {skip_count} 未通过")
    logger.info(f"{'═' * 50}")

    log_entry = {
        "timestamp": (history_dt or datetime.now(timezone.utc)).isoformat(),
        "entry_type": "online_learning_daily",
        "job_id": str(history_job_id or ""),
        "mode": str(history_mode or ""),
        "family": str(history_family or ""),
        "asset": str(history_asset or ""),
        "as_of_timestamp": as_of_dt.isoformat() if as_of_dt else None,
        "history_timestamp_override": history_dt.isoformat() if history_dt else None,
        "results": all_results,
        "summary": {
            "success": success_count,
            "trained_but_not_written": trained_not_written_count,
            "skipped": skip_count,
        },
    }
    log_path = PROJECT_ROOT / "logs" / "online_learning_history.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return all_results


# ─── GRU LightGBM 增量训练 ────────────────────────────────────

GRU_ASSETS = ["BTC_USDT", "ETH_USDT", "SOL_USDT"]
GRU_MODELS_BEST = PROJECT_ROOT / "experiments" / "gru_regime_v1" / "outputs" / "models_best"

def _build_gru_recent_data(
    asset: str,
    hours: int = 48,
) -> Optional[Tuple[pd.DataFrame, List[str]]]:
    """
    为 GRU LightGBM 构建最近 N 小时的特征数据（复用 prediction_writer_gru 流程）。
    返回 (merged_df, feature_cols) 或 None。
    """
    import os
    import time as _time
    import torch

    _t0 = _time.time()
    logger.info(f"  GRU {asset}: 开始导入模块...")
    from src.python.prediction_writer_gru import (
        load_gru_model, merge_embeddings,
    )
    from src.python.feature_engineering import build_features
    from src.python.data_fetcher import load_ohlcv as _load_ohlcv_fetcher
    logger.info(f"  GRU {asset}: 导入完成 ({_time.time()-_t0:.1f}s)")

    symbol_map = {"BTC_USDT": "BTC/USDT", "ETH_USDT": "ETH/USDT", "SOL_USDT": "SOL/USDT"}
    symbol = symbol_map.get(asset)
    if not symbol:
        return None

    device = torch.device("cpu")
    logger.info(f"  GRU {asset}: 加载模型...")
    try:
        model_data = load_gru_model(asset, device)
    except FileNotFoundError as e:
        logger.warning(f"  GRU {asset}: 模型加载失败 - {e}")
        return None
    logger.info(f"  GRU {asset}: 模型加载完成 ({_time.time()-_t0:.1f}s)")

    feature_cols = model_data["feature_cols"]
    if feature_cols is None:
        logger.warning(f"  GRU {asset}: 无法获取 feature_cols")
        return None

    logger.info(f"  GRU {asset}: 加载数据...")
    try:
        df = _load_ohlcv_fetcher(symbol, "15m")
    except Exception as e:
        logger.warning(f"  GRU {asset}: 数据加载失败 - {e}")
        return None
    logger.info(f"  GRU {asset}: 数据 {len(df)} 行 ({_time.time()-_t0:.1f}s)")

    if df.empty or len(df) < 200:
        return None

    tail = min(2500, len(df))
    df = df.tail(tail).reset_index(drop=True)

    logger.info(f"  GRU {asset}: 构建特征...")
    df_feat = build_features(df.copy(), symbol)
    logger.info(f"  GRU {asset}: 特征完成 ({_time.time()-_t0:.1f}s), MTF...")
    try:
        df_feat = add_multi_timeframe_features(df_feat, symbol)
    except Exception:
        pass
    logger.info(f"  GRU {asset}: MTF 完成 ({_time.time()-_t0:.1f}s), embedding...")

    from experiments.gru_regime_v1.src.data_loader import compute_features as _gru_compute, create_sliding_windows as _gru_windows
    encoder = model_data["encoder"]
    normalizer = model_data["normalizer"]
    train_config = model_data["train_config"]
    gru_feature_cols = train_config["feature_cols"]
    lookback = train_config["hyperparams"]["lookback"]

    encoder = encoder.to(device)
    encoder.eval()

    df_emb_raw = df.copy()
    df_emb_raw = _gru_compute(df_emb_raw)
    df_emb_raw["volatility_label"] = 0.0
    X_seq, _, ts_seq = _gru_windows(df_emb_raw, lookback, gru_feature_cols)
    X_seq = normalizer.transform_array(X_seq)
    if len(X_seq) == 0:
        return None

    with torch.no_grad():
        batch_x = torch.FloatTensor(X_seq).to(device)
        embeddings = encoder.get_embedding(batch_x).cpu().numpy()
    logger.info(f"  GRU {asset}: embedding 完成 ({_time.time()-_t0:.1f}s), {len(embeddings)} 行")

    ts_ms = np.asarray(ts_seq, dtype=np.int64) if not (hasattr(ts_seq, "dtype") and np.issubdtype(ts_seq.dtype, np.datetime64)) else pd.to_datetime(ts_seq).astype("int64") // 10**6
    emb_dict = {"timestamp": ts_ms}
    for j in range(embeddings.shape[1]):
        emb_dict[f"emb_{j}"] = embeddings[:, j]
    emb_df = pd.DataFrame(emb_dict)

    merged, _ = merge_embeddings(df_feat, emb_df, fill_strategy="zero")
    logger.info(f"  GRU {asset}: merge 完成 ({_time.time()-_t0:.1f}s), {len(merged)} 行")
    if len(merged) < MIN_SAMPLES:
        return None

    now = datetime.now(timezone.utc)
    cutoff = now - pd.Timedelta(hours=hours + 200 * 15 / 60)
    cutoff_ts = pd.Timestamp(cutoff)
    if "timestamp" in merged.columns:
        ts_col = merged["timestamp"]
        if hasattr(ts_col.dtype, "tz") and ts_col.dt.tz is not None:
            cutoff_ts = cutoff_ts.tz_localize("UTC") if cutoff_ts.tzinfo is None else cutoff_ts
        else:
            cutoff_ts = cutoff_ts.tz_localize(None) if cutoff_ts.tzinfo is not None else cutoff_ts
        merged = merged[ts_col >= cutoff_ts].reset_index(drop=True)

    if "close" in merged.columns:
        merged["direction_label"] = (merged["close"].shift(-1) > merged["close"]).astype(int)
        merged = merged.dropna(subset=["direction_label"]).reset_index(drop=True)

    if len(merged) < MIN_SAMPLES:
        return None

    return merged, feature_cols


def run_gru_incremental(
    assets: Optional[List[str]] = None,
    dry_run: bool = False,
    hours: int = 48,
    base_rounds: int = BASE_NUM_BOOST_ROUND,
    calm_rounds: int = CALM_NUM_BOOST_ROUND,
    shock_rounds: int = SHOCK_NUM_BOOST_ROUND,
    recency_halflife_bars: int = RECENCY_HALFLIFE_BARS,
    shock_weight_mult: float = SHOCK_SAMPLE_WEIGHT_MULT,
    calm_vol_ratio: float = CALM_VOL_RATIO,
    shock_vol_ratio: float = SHOCK_VOL_RATIO,
    shock_ret_q: float = SHOCK_RET_Q,
    rollback_auc_drop_abs: float = ROLLBACK_AUC_DROP_ABS,
    rollback_auc_drop_rel: float = ROLLBACK_AUC_DROP_REL,
    rollback_utility_drop_abs: float = ROLLBACK_UTILITY_DROP_ABS,
    rollback_down_wr_drop_abs: float = ROLLBACK_DOWN_WR_DROP_ABS,
    trade_objective_priority: bool = False,
    trade_objective_auc_catastrophic_mult: float = TRADE_OBJECTIVE_AUC_CATASTROPHIC_MULT,
    utility_min_confidence: float = UTILITY_MIN_CONFIDENCE,
    utility_min_samples: int = UTILITY_MIN_SAMPLES,
    utility_min_down_samples: int = UTILITY_MIN_DOWN_SAMPLES,
):
    """对 GRU LightGBM 头部模型执行增量训练。"""
    if assets is None:
        assets = GRU_ASSETS

    all_results = []
    for asset in assets:
        logger.info(f"\n{'─' * 50}")
        logger.info(f"  GRU {asset}")
        logger.info(f"{'─' * 50}")

        model_path = GRU_MODELS_BEST / asset / "lightgbm_with_embedding.joblib"
        if not model_path.exists():
            logger.warning(f"  GRU {asset}: 模型不存在 ({model_path})")
            continue

        data_result = _build_gru_recent_data(asset, hours=hours)
        if data_result is None:
            logger.warning(f"  GRU {asset}: 数据不足, 跳过")
            continue

        merged, feature_cols = data_result
        avail_cols = [c for c in feature_cols if c in merged.columns]
        missing = set(feature_cols) - set(avail_cols)
        if missing:
            logger.info(f"  缺失 {len(missing)} 个特征, 填充0")
            for c in missing:
                merged[c] = 0.0

        adaptive_w, adaptive_info = _build_adaptive_plan(
            merged,
            base_rounds=base_rounds,
            calm_rounds=calm_rounds,
            shock_rounds=shock_rounds,
            recency_halflife_bars=recency_halflife_bars,
            shock_weight_mult=shock_weight_mult,
            calm_vol_ratio=calm_vol_ratio,
            shock_vol_ratio=shock_vol_ratio,
            shock_ret_q=shock_ret_q,
        )

        logger.info(
            "  自适应: regime=%s, vol_ratio=%.2f, rounds=%d, shock_samples=%d",
            adaptive_info["regime"],
            adaptive_info["vol_ratio"],
            int(adaptive_info["num_boost_round"]),
            int(adaptive_info["shock_samples"]),
        )

        X_all = merged[feature_cols]
        y_all = merged["direction_label"]
        sw = pd.Series(np.ones(len(merged), dtype=float), index=merged.index)
        sw = (sw * adaptive_w).astype(float)
        sw_mean = float(sw.mean()) if len(sw) > 0 else 1.0
        if sw_mean > 0:
            sw = sw / sw_mean

        logger.info(f"  训练 lightgbm_with_embedding.joblib ({len(X_all)} 样本)...")
        res = _incremental_train_one_model(
            model_path=model_path,
            X_new=X_all,
            y_new=y_all,
            sample_weight=sw,
            feature_cols=feature_cols,
            num_boost_round=int(adaptive_info["num_boost_round"]),
            regime=str(adaptive_info["regime"]),
            rollback_auc_drop_abs=rollback_auc_drop_abs,
            rollback_auc_drop_rel=rollback_auc_drop_rel,
            rollback_utility_drop_abs=rollback_utility_drop_abs,
            rollback_down_wr_drop_abs=rollback_down_wr_drop_abs,
            trade_objective_priority=trade_objective_priority,
            trade_objective_auc_catastrophic_mult=trade_objective_auc_catastrophic_mult,
            utility_min_confidence=utility_min_confidence,
            utility_min_samples=utility_min_samples,
            utility_min_down_samples=utility_min_down_samples,
            dry_run=dry_run,
        )
        all_results.append({"asset": asset, "model": "gru_lgb", **res})

        if res["success"]:
            logger.info(
                f"    ✅ {res['old_trees']} → {res['new_trees']} 棵树, "
                f"AUC: {res['old_auc']} → {res['new_auc']} | rounds={res['num_boost_round']} | regime={res['regime']}"
                f" | utility(pnl): {res['old_utility_pnl']} → {res['new_utility_pnl']}"
                f"{' (dry-run)' if dry_run else ''}"
            )
            if res.get("auc_warning"):
                logger.warning(f"      ⚠️ AUC 小幅回落: {res['auc_warning']}")
            if res.get("utility_warning"):
                logger.warning(f"      ⚠️ 效用小幅回落: {res['utility_warning']}")
        elif res.get("trained_but_not_written"):
            logger.warning(
                f"    🛑 训练完成但禁止写盘: {res.get('writeback_veto_reason') or res.get('skipped_reason', '未知')}"
            )
        else:
            logger.warning(f"    ⏭️  跳过: {res.get('skipped_reason', '未知')}")

    _cleanup_old_backups()

    success_count = sum(1 for r in all_results if r["success"])
    trained_not_written_count = sum(1 for r in all_results if r.get("trained_but_not_written"))
    skip_count = len(all_results) - success_count
    logger.info(f"\n  GRU 完成: {success_count} 写盘成功, {trained_not_written_count} 训练完成但未写盘, {skip_count} 未通过")

    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "gru",
        "results": all_results,
        "summary": {
            "success": success_count,
            "trained_but_not_written": trained_not_written_count,
            "skipped": skip_count,
        },
    }
    log_path = PROJECT_ROOT / "logs" / "online_learning_history.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return all_results


# ─── CLI ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LightGBM 在线增量学习")
    parser.add_argument("--exp", type=int, nargs="+", help="指定 Exp ID (默认全部)")
    parser.add_argument("--gru", action="store_true", help="训练 GRU LightGBM 头部模型")
    parser.add_argument("--gru-assets", nargs="+", help="GRU 训练的 asset (默认 BTC/ETH/SOL)")
    parser.add_argument("--all", action="store_true", help="训练所有模型 (Exp + GRU)")
    parser.add_argument("--dry-run", action="store_true", help="只诊断, 不写入模型")
    parser.add_argument("--hours", type=int, default=48, help="使用最近 N 小时数据 (默认48)")
    parser.add_argument("--base-rounds", type=int, default=BASE_NUM_BOOST_ROUND, help=f"普通波动增量树数 (默认 {BASE_NUM_BOOST_ROUND})")
    parser.add_argument("--calm-rounds", type=int, default=CALM_NUM_BOOST_ROUND, help=f"平稳波动增量树数 (默认 {CALM_NUM_BOOST_ROUND})")
    parser.add_argument("--shock-rounds", type=int, default=SHOCK_NUM_BOOST_ROUND, help=f"冲击波动增量树数 (默认 {SHOCK_NUM_BOOST_ROUND})")
    parser.add_argument("--recency-halflife-bars", type=int, default=RECENCY_HALFLIFE_BARS, help=f"近期样本半衰期(15m bars) (默认 {RECENCY_HALFLIFE_BARS})")
    parser.add_argument("--shock-weight-mult", type=float, default=SHOCK_SAMPLE_WEIGHT_MULT, help=f"冲击样本加权倍数 (默认 {SHOCK_SAMPLE_WEIGHT_MULT})")
    parser.add_argument("--calm-vol-ratio", type=float, default=CALM_VOL_RATIO, help=f"平稳判定波动比阈值 (默认 {CALM_VOL_RATIO})")
    parser.add_argument("--shock-vol-ratio", type=float, default=SHOCK_VOL_RATIO, help=f"冲击判定波动比阈值 (默认 {SHOCK_VOL_RATIO})")
    parser.add_argument("--shock-ret-q", type=float, default=SHOCK_RET_Q, help=f"冲击分位阈值Q (默认 {SHOCK_RET_Q})")
    parser.add_argument("--rollback-auc-drop-abs", type=float, default=ROLLBACK_AUC_DROP_ABS, help=f"AUC绝对回滚阈值 (默认 {ROLLBACK_AUC_DROP_ABS})")
    parser.add_argument("--rollback-auc-drop-rel", type=float, default=ROLLBACK_AUC_DROP_REL, help=f"AUC相对回滚阈值 (默认 {ROLLBACK_AUC_DROP_REL})")
    parser.add_argument("--rollback-utility-drop-abs", type=float, default=ROLLBACK_UTILITY_DROP_ABS, help=f"效用pnl代理绝对回滚阈值 (默认 {ROLLBACK_UTILITY_DROP_ABS})")
    parser.add_argument("--rollback-down-wr-drop-abs", type=float, default=ROLLBACK_DOWN_WR_DROP_ABS, help=f"DOWN胜率绝对回滚阈值 (默认 {ROLLBACK_DOWN_WR_DROP_ABS})")
    parser.add_argument("--trade-objective-priority", action="store_true", help="优先按交易效用/方向表现判定是否回滚，AUC 仅作为灾难性降级兜底")
    parser.add_argument("--trade-objective-auc-catastrophic-mult", type=float, default=TRADE_OBJECTIVE_AUC_CATASTROPHIC_MULT, help=f"trade objective 优先时 AUC 灾难性回滚倍数 (默认 {TRADE_OBJECTIVE_AUC_CATASTROPHIC_MULT})")
    parser.add_argument("--utility-min-confidence", type=float, default=UTILITY_MIN_CONFIDENCE, help=f"效用统计最小置信度 (默认 {UTILITY_MIN_CONFIDENCE})")
    parser.add_argument("--utility-min-samples", type=int, default=UTILITY_MIN_SAMPLES, help=f"效用统计最小样本 (默认 {UTILITY_MIN_SAMPLES})")
    parser.add_argument("--utility-min-down-samples", type=int, default=UTILITY_MIN_DOWN_SAMPLES, help=f"DOWN效用统计最小样本 (默认 {UTILITY_MIN_DOWN_SAMPLES})")
    parser.add_argument("--model-dir-map-json", type=str, default="", help="自定义 exp_id -> model_dir JSON，用于 Core10-only 增量训练")
    parser.add_argument("--bootstrap-model-dir-map-json", type=str, default="", help="自定义 exp_id -> source_model_dir JSON；目标目录缺配置时自动 bootstrap")
    parser.add_argument("--history-job-id", type=str, default="", help="写入 online_learning_history 的 job_id，用于 grouped 透明归因")
    parser.add_argument("--history-mode", type=str, default="", help="写入 online_learning_history 的 mode 标签")
    parser.add_argument("--history-family", type=str, default="", help="写入 online_learning_history 的 family 标签")
    parser.add_argument("--history-asset", type=str, default="", help="写入 online_learning_history 的 asset 标签")
    parser.add_argument("--as-of-timestamp", type=str, default="", help="按指定 UTC/ISO 时刻构建训练数据窗口，用于历史缺口回放")
    parser.add_argument("--history-timestamp", type=str, default="", help="覆盖 online_learning_history 的记录时间戳，用于历史 coverage 补齐")
    parser.add_argument("--family-atomic-writeback", action="store_true", help="同一 exp family 联合判定后再原子提交被批准的 window 子集")
    parser.add_argument("--slot01-positive-writeback", action="store_true", help="一号槽位专属正增益写回门；不使用家族联合否决")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细日志")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    # 当由 scheduler 重定向 stdout 到 online_learning.log 时，关闭本地 FileHandler，
    # 避免同一日志被重复写两遍（I/O 放大 + 观察误判）。
    if os.getenv("ONLINE_LEARNING_NO_FILE_HANDLER", "0") != "1":
        handlers.append(
            RotatingFileHandler(
                LOG_FILE,
                maxBytes=max(8, ONLINE_LOG_MAX_MB) * 1024 * 1024,
                backupCount=max(1, ONLINE_LOG_BACKUP_COUNT),
                encoding="utf-8",
            )
        )
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )

    logger.info(f"═══ LightGBM 在线增量学习 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ═══")

    run_exp = args.all or (not args.gru)
    run_gru = args.all or args.gru

    if run_exp:
        run_incremental(
            exp_ids=args.exp,
            dry_run=args.dry_run,
            model_dir_map_json=(args.model_dir_map_json or None),
            bootstrap_model_dir_map_json=(args.bootstrap_model_dir_map_json or None),
            hours=args.hours,
            base_rounds=args.base_rounds,
            calm_rounds=args.calm_rounds,
            shock_rounds=args.shock_rounds,
            recency_halflife_bars=args.recency_halflife_bars,
            shock_weight_mult=args.shock_weight_mult,
            calm_vol_ratio=args.calm_vol_ratio,
            shock_vol_ratio=args.shock_vol_ratio,
            shock_ret_q=args.shock_ret_q,
            rollback_auc_drop_abs=args.rollback_auc_drop_abs,
            rollback_auc_drop_rel=args.rollback_auc_drop_rel,
            rollback_utility_drop_abs=args.rollback_utility_drop_abs,
            rollback_down_wr_drop_abs=args.rollback_down_wr_drop_abs,
            trade_objective_priority=args.trade_objective_priority,
            trade_objective_auc_catastrophic_mult=args.trade_objective_auc_catastrophic_mult,
            utility_min_confidence=args.utility_min_confidence,
            utility_min_samples=args.utility_min_samples,
            utility_min_down_samples=args.utility_min_down_samples,
            history_job_id=(args.history_job_id or None),
            history_mode=(args.history_mode or None),
            history_family=(args.history_family or None),
            history_asset=(args.history_asset or None),
            family_atomic_writeback=bool(args.family_atomic_writeback),
            slot01_positive_writeback=bool(args.slot01_positive_writeback),
            as_of_timestamp=(args.as_of_timestamp or None),
            history_timestamp=(args.history_timestamp or None),
        )
    if run_gru:
        gru_assets = args.gru_assets if args.gru_assets else None
        run_gru_incremental(
            assets=gru_assets,
            dry_run=args.dry_run,
            hours=args.hours,
            base_rounds=args.base_rounds,
            calm_rounds=args.calm_rounds,
            shock_rounds=args.shock_rounds,
            recency_halflife_bars=args.recency_halflife_bars,
            shock_weight_mult=args.shock_weight_mult,
            calm_vol_ratio=args.calm_vol_ratio,
            shock_vol_ratio=args.shock_vol_ratio,
            shock_ret_q=args.shock_ret_q,
            rollback_auc_drop_abs=args.rollback_auc_drop_abs,
            rollback_auc_drop_rel=args.rollback_auc_drop_rel,
            rollback_utility_drop_abs=args.rollback_utility_drop_abs,
            rollback_down_wr_drop_abs=args.rollback_down_wr_drop_abs,
            trade_objective_priority=args.trade_objective_priority,
            trade_objective_auc_catastrophic_mult=args.trade_objective_auc_catastrophic_mult,
            utility_min_confidence=args.utility_min_confidence,
            utility_min_samples=args.utility_min_samples,
            utility_min_down_samples=args.utility_min_down_samples,
        )


if __name__ == "__main__":
    main()
