#!/usr/bin/env python3
"""
GRU LightGBM 在线增量学习 — 独立入口

与 online_learning_daily.py 分离，避免 run_grid 导入链干扰 PyTorch 推理。

数据与备份：每日训练为增量追加，仅覆盖同路径模型文件并保留近期备份，
不删除历史训练数据（raw/parquet 不被脚本删除）。

用法:
    python3 scripts/online_learning_gru.py            # 所有 GRU 资产
    python3 scripts/online_learning_gru.py --dry-run   # 只诊断
    python3 scripts/online_learning_gru.py --assets BTC_USDT  # 单个资产
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
from logging.handlers import RotatingFileHandler
import shutil
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import os
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

import torch
torch.set_num_threads(1)

from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.python.prediction_writer_gru import load_gru_model, merge_embeddings
from src.python.feature_engineering import build_features, add_multi_timeframe_features
from src.python.data_fetcher import load_ohlcv
from experiments.gru_regime_v1.src.data_loader import (
    compute_features as _gru_compute,
    create_sliding_windows as _gru_windows,
)

INCREMENTAL_LR = 0.005
BASE_NUM_BOOST_ROUND = 12
CALM_NUM_BOOST_ROUND = 8
SHOCK_NUM_BOOST_ROUND = 20
BAGGING_FRACTION = 0.8
MIN_SAMPLES = 50
BACKUP_KEEP_DAYS = max(1, int(os.getenv("ONLINE_LEARNING_BACKUP_KEEP_DAYS", "3") or "3"))
ROLLBACK_AUC_DROP_ABS = 0.004
ROLLBACK_AUC_DROP_REL = 0.008
ROLLBACK_UTILITY_DROP_ABS = 0.03
ROLLBACK_DOWN_WR_DROP_ABS = 0.05
UTILITY_MIN_CONFIDENCE = 0.53
UTILITY_MIN_SAMPLES = 120
UTILITY_MIN_DOWN_SAMPLES = 40
RECENCY_HALFLIFE_BARS = 64
SHOCK_SAMPLE_WEIGHT_MULT = 1.8
CALM_VOL_RATIO = 1.08
SHOCK_VOL_RATIO = 1.45
SHOCK_RET_Q = 0.85

GRU_ASSETS = ["BTC_USDT", "ETH_USDT", "SOL_USDT"]
GRU_MODELS_BEST = PROJECT_ROOT / "experiments" / "gru_regime_v1" / "outputs" / "models_best"
BACKUP_DIR = PROJECT_ROOT / "data" / "models" / "_online_learning_backups"
LOG_FILE = PROJECT_ROOT / "logs" / "online_learning.log"
LAST_SUCCESS_PATH = PROJECT_ROOT / "logs" / "daily_training_last_success.json"
ONLINE_LOG_MAX_MB = int(os.getenv("ONLINE_LEARNING_LOG_MAX_MB", "32") or "32")
ONLINE_LOG_BACKUP_COUNT = int(os.getenv("ONLINE_LEARNING_LOG_BACKUP_COUNT", "5") or "5")
LAST_SUCCESS_RETENTION_HOURS = int(os.getenv("ONLINE_LEARNING_LAST_SUCCESS_RETENTION_HOURS", "30") or "30")

logger = logging.getLogger("online_learning_gru")


def _parse_dt(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


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
        old_dt = _parse_dt(old)
        new_dt = _parse_dt(value_s)
        if old_dt is None or (new_dt is not None and new_dt > old_dt):
            merged[key_s] = value_s
    return dict(sorted(merged.items()))


def _merge_str_list(existing: object, additions: list[str]) -> list[str]:
    out = {str(x).strip() for x in additions if str(x).strip()}
    if isinstance(existing, list):
        out.update(str(x).strip() for x in existing if str(x).strip())
    return sorted(out)


def _calc_abs_returns(df: pd.DataFrame) -> pd.Series:
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
    down_pnl_proxy = (
        np.where(down_wins > 0.5, down_edge, -down_edge)
        if len(down_wins) > 0
        else np.array([], dtype=float)
    )

    return {
        "n": float(int(take.sum())),
        "wr": float(wins[take].mean()) if int(take.sum()) > 0 else None,
        "pnl_proxy": float(pnl_proxy[take].mean()) if int(take.sum()) > 0 else None,
        "down_n": float(int(down_take.sum())),
        "down_wr": float(down_wins.mean()) if len(down_wins) > 0 else None,
        "down_pnl_proxy": float(down_pnl_proxy.mean()) if len(down_pnl_proxy) > 0 else None,
    }


def _incremental_train(
    model_path,
    X_new,
    y_new,
    sample_weight=None,
    feature_cols=None,
    num_boost_round: int = BASE_NUM_BOOST_ROUND,
    regime: str = "normal",
    rollback_auc_drop_abs: float = ROLLBACK_AUC_DROP_ABS,
    rollback_auc_drop_rel: float = ROLLBACK_AUC_DROP_REL,
    rollback_utility_drop_abs: float = ROLLBACK_UTILITY_DROP_ABS,
    rollback_down_wr_drop_abs: float = ROLLBACK_DOWN_WR_DROP_ABS,
    utility_min_confidence: float = UTILITY_MIN_CONFIDENCE,
    utility_min_samples: int = UTILITY_MIN_SAMPLES,
    utility_min_down_samples: int = UTILITY_MIN_DOWN_SAMPLES,
    dry_run=False,
):
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
    }
    try:
        old_model = joblib.load(model_path)
    except Exception as e:
        result["skipped_reason"] = f"加载失败: {e}"
        return result

    old_booster = old_model.booster_
    result["old_trees"] = old_booster.num_trees()
    X_all = X_new[feature_cols] if feature_cols else X_new
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
        pass
    old_utility = _calc_validation_trade_utility(y_val, old_proba_val, utility_min_confidence)
    result["old_utility_n"] = old_utility.get("n")
    result["old_utility_wr"] = old_utility.get("wr")
    result["old_utility_pnl"] = old_utility.get("pnl_proxy")
    result["old_down_n"] = old_utility.get("down_n")
    result["old_down_wr"] = old_utility.get("down_wr")

    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp_path = tmp.name
    old_booster.save_model(tmp_path)
    params = old_booster.params.copy() if hasattr(old_booster, "params") else {}
    params.update({"learning_rate": INCREMENTAL_LR, "bagging_fraction": BAGGING_FRACTION, "bagging_freq": 1, "verbosity": -1})
    for k in ("num_iterations", "num_trees"):
        params.pop(k, None)

    train_set = lgb.Dataset(
        X_train,
        label=y_train,
        weight=sw_train.values if sw_train is not None else None,
        free_raw_data=False,
    )
    use_rounds = max(1, int(num_boost_round))
    new_booster = lgb.train(params, train_set, num_boost_round=use_rounds, init_model=tmp_path)
    Path(tmp_path).unlink(missing_ok=True)
    result["new_trees"] = new_booster.num_trees()

    new_proba_val = new_booster.predict(X_val)
    try:
        result["new_auc"] = round(float(roc_auc_score(y_val, new_proba_val)), 5)
    except ValueError:
        pass
    new_utility = _calc_validation_trade_utility(y_val, new_proba_val, utility_min_confidence)
    result["new_utility_n"] = new_utility.get("n")
    result["new_utility_wr"] = new_utility.get("wr")
    result["new_utility_pnl"] = new_utility.get("pnl_proxy")
    result["new_down_n"] = new_utility.get("down_n")
    result["new_down_wr"] = new_utility.get("down_wr")
    result["training_completed"] = True

    old_utility_n = float(old_utility.get("n") or 0.0)
    new_utility_n = float(new_utility.get("n") or 0.0)
    old_utility_pnl = old_utility.get("pnl_proxy")
    new_utility_pnl = new_utility.get("pnl_proxy")
    utility_ready = (
        old_utility_n >= float(utility_min_samples)
        and new_utility_n >= float(utility_min_samples)
        and old_utility_pnl is not None
        and new_utility_pnl is not None
    )
    old_down_n = float(old_utility.get("down_n") or 0.0)
    new_down_n = float(new_utility.get("down_n") or 0.0)
    old_down_wr = old_utility.get("down_wr")
    new_down_wr = new_utility.get("down_wr")
    down_ready = (
        old_down_n >= float(utility_min_down_samples)
        and new_down_n >= float(utility_min_down_samples)
        and old_down_wr is not None
        and new_down_wr is not None
    )
    auc_ready = result["old_auc"] is not None and result["new_auc"] is not None
    result["metrics_ready_for_writeback"] = bool(auc_ready and utility_ready and down_ready)
    if not result["metrics_ready_for_writeback"]:
        result["trained_but_not_written"] = True
        result["writeback_veto_reason"] = "insufficient_validation_support"
        result["skipped_reason"] = "训练完成但验证支持不足，禁止写盘"
        return result

    regressions = []
    improvements = []
    if float(new_utility_pnl) < float(old_utility_pnl):
        regressions.append("utility_regression")
        result["utility_warning"] = f"pnl_proxy {float(old_utility_pnl):.5f} -> {float(new_utility_pnl):.5f}"
    elif float(new_utility_pnl) > float(old_utility_pnl):
        improvements.append("utility_pnl")

    if float(new_down_wr) < float(old_down_wr):
        regressions.append("down_wr_regression")
    elif float(new_down_wr) > float(old_down_wr):
        improvements.append("down_wr")

    if float(result["new_auc"]) < float(result["old_auc"]):
        regressions.append("auc_regression")
        result["auc_warning"] = f"{result['old_auc']} -> {result['new_auc']}"
    elif float(result["new_auc"]) > float(result["old_auc"]):
        improvements.append("auc")

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

    today_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    backup_dir = BACKUP_DIR / model_path.parent.name
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{model_path.stem}_{today_str}.joblib"
    shutil.copy2(model_path, backup_path)
    result["backed_up_to"] = str(backup_path)

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
    joblib.dump(new_model, model_path)
    result["success"] = True
    return result


def run(
    assets=None,
    dry_run=False,
    hours=48,
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
    utility_min_confidence: float = UTILITY_MIN_CONFIDENCE,
    utility_min_samples: int = UTILITY_MIN_SAMPLES,
    utility_min_down_samples: int = UTILITY_MIN_DOWN_SAMPLES,
):
    if assets is None:
        assets = GRU_ASSETS
    device = torch.device("cpu")
    symbol_map = {"BTC_USDT": "BTC/USDT", "ETH_USDT": "ETH/USDT", "SOL_USDT": "SOL/USDT"}
    all_results = []

    for asset in assets:
        symbol = symbol_map.get(asset)
        if not symbol:
            continue
        logger.info(f"\n{'─'*50}\n  GRU {asset}\n{'─'*50}")
        t0 = time.time()

        model_path = GRU_MODELS_BEST / asset / "lightgbm_with_embedding.joblib"
        if not model_path.exists():
            logger.warning(f"  模型不存在: {model_path}")
            continue

        try:
            model_data = load_gru_model(asset, device)
        except Exception as e:
            logger.warning(f"  模型加载失败: {e}")
            continue
        feature_cols = model_data["feature_cols"]
        if feature_cols is None:
            continue

        df = load_ohlcv(symbol, "15m")
        if df.empty or len(df) < 200:
            logger.warning(f"  数据不足")
            continue
        df = df.tail(2500).reset_index(drop=True)
        logger.info(f"  数据: {len(df)} 行 ({time.time()-t0:.1f}s)")

        df_feat = build_features(df.copy(), symbol)
        try:
            df_feat = add_multi_timeframe_features(df_feat, symbol)
        except Exception:
            pass
        logger.info(f"  特征完成 ({time.time()-t0:.1f}s)")

        encoder = model_data["encoder"].to(device)
        encoder.eval()
        normalizer = model_data["normalizer"]
        train_config = model_data["train_config"]
        gru_fc = train_config["feature_cols"]
        lookback = train_config["hyperparams"]["lookback"]

        df_e = df.copy()
        df_e = _gru_compute(df_e)
        df_e["volatility_label"] = 0.0
        X_seq, _, ts_seq = _gru_windows(df_e, lookback, gru_fc)
        X_seq = normalizer.transform_array(X_seq)
        if len(X_seq) == 0:
            continue

        with torch.no_grad():
            batch_x = torch.FloatTensor(X_seq).to(device)
            embeddings = encoder.get_embedding(batch_x).cpu().numpy()
        logger.info(f"  Embedding: {embeddings.shape} ({time.time()-t0:.1f}s)")

        ts_ms = np.asarray(ts_seq, dtype=np.int64) if not (hasattr(ts_seq, "dtype") and np.issubdtype(ts_seq.dtype, np.datetime64)) else pd.to_datetime(ts_seq).astype("int64") // 10**6
        emb_dict = {"timestamp": ts_ms}
        for j in range(embeddings.shape[1]):
            emb_dict[f"emb_{j}"] = embeddings[:, j]
        emb_df = pd.DataFrame(emb_dict)

        merged, _ = merge_embeddings(df_feat, emb_df, fill_strategy="zero")
        if "timestamp" in merged.columns:
            merged = merged.sort_values("timestamp").reset_index(drop=True)
        if "close" in merged.columns:
            merged["direction_label"] = (merged["close"].shift(-1) > merged["close"]).astype(int)
            merged = merged.dropna(subset=["direction_label"]).reset_index(drop=True)
        if len(merged) < MIN_SAMPLES:
            continue

        cutoff_hours = hours + 200 * 15 / 60
        cutoff_ms = int((time.time() - cutoff_hours * 3600) * 1000)
        if "timestamp" in merged.columns:
            tc = merged["timestamp"]
            if tc.dtype == np.int64 or tc.dtype == np.float64:
                merged = merged[tc >= cutoff_ms].reset_index(drop=True)
            else:
                cutoff_dt = pd.Timestamp.now() - pd.Timedelta(hours=cutoff_hours)
                try:
                    merged = merged[tc >= cutoff_dt].reset_index(drop=True)
                except TypeError:
                    pass

        missing = set(feature_cols) - set(merged.columns)
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
        logger.info(f"  训练样本: {len(X_all)} ({time.time()-t0:.1f}s)")

        res = _incremental_train(
            model_path,
            X_all,
            y_all,
            sample_weight=sw,
            feature_cols=feature_cols,
            num_boost_round=int(adaptive_info["num_boost_round"]),
            regime=str(adaptive_info["regime"]),
            rollback_auc_drop_abs=rollback_auc_drop_abs,
            rollback_auc_drop_rel=rollback_auc_drop_rel,
            rollback_utility_drop_abs=rollback_utility_drop_abs,
            rollback_down_wr_drop_abs=rollback_down_wr_drop_abs,
            utility_min_confidence=utility_min_confidence,
            utility_min_samples=utility_min_samples,
            utility_min_down_samples=utility_min_down_samples,
            dry_run=dry_run,
        )
        all_results.append({"asset": asset, **res})
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
            logger.warning(f"    ⏭️ 跳过: {res.get('skipped_reason')}")

    s = sum(1 for r in all_results if r["success"])
    trained_not_written = sum(1 for r in all_results if r.get("trained_but_not_written"))
    logger.info(f"\n  GRU 完成: {s} 写盘成功, {trained_not_written} 训练完成但未写盘, {len(all_results)-s} 未通过")

    if not dry_run and s > 0:
        try:
            data = {}
            if LAST_SUCCESS_PATH.exists():
                try:
                    data = json.loads(LAST_SUCCESS_PATH.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    pass
            gru_asset_updated_at: dict[str, str] = {}
            for r in all_results:
                if not r.get("success"):
                    continue
                asset = r.get("asset")
                model_path_str = r.get("model_path")
                if not asset or not model_path_str:
                    continue
                try:
                    mtime = Path(model_path_str).stat().st_mtime
                    gru_asset_updated_at[str(asset)] = datetime.fromtimestamp(
                        mtime, timezone.utc
                    ).isoformat()
                except OSError:
                    continue
            data["gru_updated_at"] = datetime.now(timezone.utc).isoformat()
            merged_gru_asset_updated_at = _merge_iso_map(data.get("gru_asset_updated_at"), gru_asset_updated_at)
            cutoff = datetime.now(timezone.utc) - timedelta(hours=LAST_SUCCESS_RETENTION_HOURS)
            merged_gru_asset_updated_at = {
                asset: ts
                for asset, ts in merged_gru_asset_updated_at.items()
                if _parse_dt(ts) is not None and _parse_dt(ts) >= cutoff
            }
            data["gru_asset_updated_at"] = dict(sorted(merged_gru_asset_updated_at.items()))
            data["gru_assets"] = _merge_str_list(data.get("gru_assets"), list(data["gru_asset_updated_at"].keys()))
            data["gru_assets"] = [asset for asset in data["gru_assets"] if asset in data["gru_asset_updated_at"]]
            LAST_SUCCESS_PATH.parent.mkdir(parents=True, exist_ok=True)
            LAST_SUCCESS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            logger.info(f"  已更新 {LAST_SUCCESS_PATH.name}: 本轮新增 gru_assets={[r['asset'] for r in all_results if r['success']]}")
        except OSError as e:
            logger.warning(f"  写入 last_success 失败: {e}")

    log_path = PROJECT_ROOT / "logs" / "online_learning_history.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "type": "gru", "results": all_results, "summary": {"success": s, "trained_but_not_written": trained_not_written, "skipped": len(all_results)-s}}, ensure_ascii=False) + "\n")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GRU LightGBM 在线增量学习")
    parser.add_argument("--assets", nargs="+", help="指定 asset (默认 BTC/ETH/SOL)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--hours", type=int, default=48)
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
    parser.add_argument("--utility-min-confidence", type=float, default=UTILITY_MIN_CONFIDENCE, help=f"效用统计最小置信度 (默认 {UTILITY_MIN_CONFIDENCE})")
    parser.add_argument("--utility-min-samples", type=int, default=UTILITY_MIN_SAMPLES, help=f"效用统计最小样本 (默认 {UTILITY_MIN_SAMPLES})")
    parser.add_argument("--utility-min-down-samples", type=int, default=UTILITY_MIN_DOWN_SAMPLES, help=f"DOWN效用统计最小样本 (默认 {UTILITY_MIN_DOWN_SAMPLES})")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    # 当由 scheduler 重定向 stdout 到 online_learning.log 时，关闭本地 FileHandler，
    # 避免同一日志被重复写两遍。
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
    logger.info(f"═══ GRU 在线增量学习 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} ═══")
    run(
        assets=args.assets,
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
        utility_min_confidence=args.utility_min_confidence,
        utility_min_samples=args.utility_min_samples,
        utility_min_down_samples=args.utility_min_down_samples,
    )
