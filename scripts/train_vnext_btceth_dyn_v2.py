#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "gru_regime_v1"))


def _preload_local_package(name: str, init_path: Path) -> None:
    if name in sys.modules or not init_path.exists():
        return
    spec = importlib.util.spec_from_file_location(
        name,
        init_path,
        submodule_search_locations=[str(init_path.parent)],
    )
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


_preload_local_package("src", PROJECT_ROOT / "src" / "__init__.py")
_preload_local_package("src.python", PROJECT_ROOT / "src" / "python" / "__init__.py")

from experiments.sentiment_grid_search.data_prep import FEATURE_GROUPS
from experiments.sentiment_grid_search.run_grid import build_tech_features, get_data_paths, merge_sentiment_to_tech
from src.python.feature_engineering import add_multi_timeframe_features

REPORTS = PROJECT_ROOT / "reports"
MODELS = PROJECT_ROOT / "data" / "models"

BTC_MODEL_DIR = MODELS / "vnext_btc_dyn_v2"
ETH_MODEL_DIR = MODELS / "vnext_eth_dyn_v2"
TRAIN_REPORT = REPORTS / "vnext_btceth_dyn_v2_training_latest.json"
BTC_LEADERBOARD = REPORTS / "vnext_btc_dyn_v2_leaderboard_latest.json"
ETH_LEADERBOARD = REPORTS / "vnext_eth_dyn_v2_leaderboard_latest.json"
COMPARE_REPORT = REPORTS / "vnext_btceth_dyn_v2_compare_latest.json"

SEARCH_WORKERS = 8
COARSE_PRICE_GRID = {
    "BTC_USDT": [0.44, 0.46, 0.48, 0.50, 0.52, 0.54],
    "ETH_USDT": [0.43, 0.45, 0.47, 0.49, 0.51, 0.53],
}
WINDOW_SETS: list[list[int]] = [
    [60, 90, 120],
    [90, 120, 180],
    [120, 180, 240],
    [180, 240, 360],
]
TRAIN_DAYS_CANDIDATES = [90, 120, 180, 240, 360]
PARAM_TEMPLATES: dict[str, dict[str, Any]] = {
    "conservative": {
        "num_leaves": 31,
        "max_depth": 5,
        "learning_rate": 0.03,
        "min_child_samples": 80,
        "feature_fraction": 0.70,
        "bagging_fraction": 0.75,
        "bagging_freq": 2,
        "lambda_l1": 0.30,
        "lambda_l2": 12.0,
        "max_bin": 255,
        "min_split_gain": 0.05,
    },
    "balanced": {
        "num_leaves": 63,
        "max_depth": 6,
        "learning_rate": 0.02,
        "min_child_samples": 50,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.85,
        "bagging_freq": 2,
        "lambda_l1": 0.10,
        "lambda_l2": 10.0,
        "max_bin": 255,
        "min_split_gain": 0.03,
    },
    "elastic": {
        "num_leaves": 95,
        "max_depth": 8,
        "learning_rate": 0.015,
        "min_child_samples": 30,
        "feature_fraction": 0.90,
        "bagging_fraction": 0.90,
        "bagging_freq": 1,
        "lambda_l1": 0.05,
        "lambda_l2": 6.0,
        "max_bin": 383,
        "min_split_gain": 0.01,
    },
    "deep_regularized": {
        "num_leaves": 127,
        "max_depth": 9,
        "learning_rate": 0.012,
        "min_child_samples": 20,
        "feature_fraction": 0.95,
        "bagging_fraction": 0.90,
        "bagging_freq": 1,
        "lambda_l1": 0.15,
        "lambda_l2": 14.0,
        "max_bin": 511,
        "min_split_gain": 0.08,
    },
}
BTC_BUNDLES: dict[str, list[str]] = {
    "btc_micro": ["ob", "funding", "oi", "lsratio", "polymarket_prob_target"],
    "btc_micro_pm": ["ob", "funding", "oi", "lsratio", "polymarket_prob", "polymarket_prob_target"],
    "btc_micro_pm_fgi": ["fgi_daily", "ob", "funding", "oi", "lsratio", "polymarket_prob", "polymarket_prob_target"],
    "btc_regime_lite": ["fgi_daily", "ob", "funding", "oi", "lsratio", "polymarket_prob", "polymarket_prob_target"],
}
ETH_BUNDLES: dict[str, list[str]] = {
    "eth_micro": ["ob", "funding", "oi", "lsratio", "polymarket_prob_target"],
    "eth_micro_pm": ["ob", "funding", "oi", "lsratio", "polymarket_prob", "polymarket_prob_target"],
    "eth_micro_pm_fgi": ["fgi_daily", "ob", "funding", "oi", "lsratio", "polymarket_prob", "polymarket_prob_target"],
    "eth_wallet": ["fgi_daily", "ob", "funding", "oi", "lsratio", "polymarket_prob", "polymarket_prob_target", "wallet_cohort"],
    "eth_wallet_regime": ["fgi_daily", "ob", "funding", "oi", "lsratio", "polymarket_prob", "polymarket_prob_target", "wallet_cohort"],
}
REGIME_LITE_COLS = [
    "ret_1h",
    "ret_4h",
    "ret_1d",
    "rv_1h",
    "rv_4h",
    "range_expansion",
    "trend_alignment_lite",
    "trend_flip_risk_lite",
]
TECH_EXCLUDE = {
    "timestamp", "timestamp_ms", "date", "symbol", "open", "high", "low", "close", "volume",
    "direction_label", "sample_weight", "price_proxy", "take_label", "price_label", "best_price_utility",
    "next_return", "move_score", "volatility_ref", "direction_up_label", "direction_down_label",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-arr))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _round_float(v: Any, digits: int = 6) -> Any:
    try:
        fv = float(v)
    except Exception:
        return v
    if math.isnan(fv) or math.isinf(fv):
        return None
    return round(fv, digits)


def _recent_weight_multiplier(ts: pd.Series) -> np.ndarray:
    latest = ts.max()
    age_days = (latest - ts).dt.total_seconds() / 86400.0
    out = np.ones(len(ts), dtype=float)
    out = np.where(age_days <= 30.0, out * 1.3, out)
    out = np.where(age_days <= 14.0, out * (1.8 / 1.3), out)
    out = np.where(age_days <= 3.0, out * (2.5 / 1.8), out)
    return out


def _ensure_regime_lite_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = pd.to_numeric(out.get("close"), errors="coerce").astype(float)
    ret_15m = close.pct_change().fillna(0.0)
    out["ret_1h"] = (close / close.shift(4) - 1.0).fillna(0.0)
    out["ret_4h"] = (close / close.shift(16) - 1.0).fillna(0.0)
    out["ret_1d"] = (close / close.shift(96) - 1.0).fillna(0.0)
    out["rv_1h"] = ret_15m.rolling(4).std().fillna(0.0)
    out["rv_4h"] = ret_15m.rolling(16).std().fillna(0.0)
    bar_range = (pd.to_numeric(out.get("high"), errors="coerce") - pd.to_numeric(out.get("low"), errors="coerce")).abs()
    out["range_expansion"] = (bar_range / close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["trend_alignment_lite"] = (
        np.sign(out["ret_1h"].to_numpy(dtype=float)) * np.sign(out["ret_4h"].to_numpy(dtype=float))
    ).astype(float)
    out["trend_flip_risk_lite"] = (
        np.sign(out["ret_1h"].to_numpy(dtype=float)) * np.sign(out["ret_1d"].to_numpy(dtype=float)) < 0
    ).astype(int)
    return out


def _infer_prob_proxy(df: pd.DataFrame) -> pd.Series:
    if "target_logit_p" in df.columns:
        return pd.Series(sigmoid(df["target_logit_p"].to_numpy(dtype=float)), index=df.index)
    if "logit_p" in df.columns:
        return pd.Series(sigmoid(df["logit_p"].to_numpy(dtype=float)), index=df.index)
    return pd.Series(0.50, index=df.index, dtype=float)


def _asset_bundle_map(asset: str) -> dict[str, list[str]]:
    return BTC_BUNDLES if asset == "BTC_USDT" else ETH_BUNDLES


def _price_grid_for_asset(asset: str, center_prices: Sequence[float] | None = None) -> list[str]:
    if center_prices is None:
        return [f"{p:.2f}" for p in COARSE_PRICE_GRID[asset]] + ["ABSTAIN"]
    refined: set[float] = set()
    for price in center_prices:
        rounded = round(float(price), 2)
        for delta in (-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03):
            refined.add(round(rounded + delta, 2))
    lo = 0.42 if asset == "BTC_USDT" else 0.41
    hi = 0.56 if asset == "BTC_USDT" else 0.55
    prices = sorted(p for p in refined if lo <= p <= hi)
    return [f"{p:.2f}" for p in prices] + ["ABSTAIN"]


def _required_group_cols(groups: Sequence[str]) -> set[str]:
    cols: set[str] = set()
    for group in groups:
        cols.update(FEATURE_GROUPS.get(group, []))
    return cols


def _choose_feature_cols(df: pd.DataFrame, bundle_name: str, groups: Sequence[str]) -> list[str]:
    selected: list[str] = []
    required_cols = _required_group_cols(groups)
    for col in df.columns:
        if col in TECH_EXCLUDE:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and not df[col].isna().all():
            selected.append(col)
    if bundle_name.endswith("regime_lite") or bundle_name.endswith("wallet_regime"):
        for col in REGIME_LITE_COLS:
            if col in df.columns and col not in selected:
                selected.append(col)
        coverage = {col: float(df[col].notna().mean()) for col in REGIME_LITE_COLS if col in df.columns}
        if coverage and min(coverage.values()) < 0.95:
            return []
    # 只保留选中特征组中要求的列 + 通用技术列
    optional_group_cols = set().union(*[set(v) for k, v in FEATURE_GROUPS.items() if k not in {"base", *groups}])
    final = [c for c in selected if c not in optional_group_cols or c in required_cols]
    return sorted(dict.fromkeys(final))


def _default_utility_cfg() -> dict[str, float]:
    return {
        "abstain_cost_bias": 0.08,
        "fill_penalty_scale": 0.18,
        "move_cap": 2.0,
        "wrong_loss_scale": 1.0,
    }


def _derive_labels(df: pd.DataFrame, asset: str, price_actions: Sequence[str], utility_cfg: dict[str, float]) -> pd.DataFrame:
    out = df.copy()
    next_close = pd.to_numeric(out["close"], errors="coerce").shift(-1)
    out["next_return"] = (next_close / pd.to_numeric(out["close"], errors="coerce") - 1.0)
    out["direction_label"] = np.where(out["next_return"].isna(), np.nan, (out["next_return"] > 0).astype(float))
    out = out.dropna(subset=["direction_label", "next_return"]).reset_index(drop=True)
    out["direction_label"] = out["direction_label"].astype(int)
    out["price_proxy"] = _infer_prob_proxy(out).clip(0.01, 0.99)

    vol_a = pd.to_numeric(out.get("atr_pct_14"), errors="coerce") if "atr_pct_14" in out.columns else pd.Series(0.0, index=out.index)
    vol_b = pd.to_numeric(out.get("volatility_10"), errors="coerce") if "volatility_10" in out.columns else pd.Series(0.0, index=out.index)
    out["volatility_ref"] = np.maximum.reduce([
        np.nan_to_num(vol_a.to_numpy(dtype=float), nan=0.0),
        np.nan_to_num(vol_b.to_numpy(dtype=float), nan=0.0),
        np.full(len(out), 0.002, dtype=float),
    ])
    move_cap = float(utility_cfg["move_cap"])
    out["move_score"] = np.clip(np.abs(out["next_return"].to_numpy(dtype=float)) / out["volatility_ref"].to_numpy(dtype=float), 0.0, move_cap)

    price_candidates = [p for p in price_actions if p != "ABSTAIN"]
    utility_by_price: dict[str, np.ndarray] = {}
    proxy = out["price_proxy"].to_numpy(dtype=float)
    move_score = out["move_score"].to_numpy(dtype=float)
    abstain_cost_bias = float(utility_cfg["abstain_cost_bias"])
    fill_penalty_scale = float(utility_cfg["fill_penalty_scale"])
    wrong_loss_scale = float(utility_cfg["wrong_loss_scale"])
    edge_mass = np.maximum(move_score - 0.35, 0.0)
    quality_scale = 0.65 + np.minimum(np.abs(proxy - 0.5) * 2.0, 0.35)

    for price_text in price_candidates:
        price = float(price_text)
        filled = proxy <= price
        price_penalty = price * 0.18 * max(0.6, wrong_loss_scale)
        slippage_penalty = np.maximum(price - proxy, 0.0) * fill_penalty_scale
        utility = np.where(
            filled,
            edge_mass * quality_scale * (1.0 - price) - price_penalty - slippage_penalty,
            0.0,
        )
        utility_by_price[price_text] = utility

    utility_matrix = np.vstack([utility_by_price[p] for p in price_candidates]).T
    best_price_idx = np.argmax(utility_matrix, axis=1)
    best_price_utility = utility_matrix[np.arange(len(out)), best_price_idx]
    best_actions = np.asarray(price_candidates, dtype=object)[best_price_idx]
    take_label = (best_price_utility > abstain_cost_bias).astype(int)
    out["take_label"] = take_label
    out["price_label"] = np.where(take_label == 1, best_actions, "ABSTAIN")
    out["best_price_utility"] = best_price_utility
    out["sample_weight"] = _recent_weight_multiplier(out["timestamp"]) * (1.0 + 0.15 * np.clip(move_score, 0.0, 2.0))
    return out


@dataclass
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def _split_recent(df: pd.DataFrame, train_days: int, val_days: int = 4, test_days: int = 4) -> SplitData:
    latest = df["timestamp"].max()
    test_cutoff = latest - pd.Timedelta(days=test_days)
    val_cutoff = test_cutoff - pd.Timedelta(days=val_days)
    train_cutoff = val_cutoff - pd.Timedelta(days=train_days)
    train = df[(df["timestamp"] >= train_cutoff) & (df["timestamp"] < val_cutoff)].copy()
    val = df[(df["timestamp"] >= val_cutoff) & (df["timestamp"] < test_cutoff)].copy()
    test = df[df["timestamp"] >= test_cutoff].copy()
    return SplitData(train=train, val=val, test=test)


def _lgb_binary(params: dict[str, Any], n_jobs: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        metric="auc",
        n_estimators=1600,
        verbosity=-1,
        n_jobs=n_jobs,
        **params,
    )


def _lgb_multiclass(params: dict[str, Any], num_class: int, n_jobs: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_class,
        metric="multi_logloss",
        n_estimators=1600,
        verbosity=-1,
        n_jobs=n_jobs,
        **params,
    )


def _fit_heads(split: SplitData, feature_cols: Sequence[str], params: dict[str, Any], price_actions: Sequence[str], lgb_jobs: int) -> dict[str, Any]:
    X_tr = split.train[list(feature_cols)]
    X_va = split.val[list(feature_cols)]
    X_te = split.test[list(feature_cols)]
    w_tr = split.train["sample_weight"].to_numpy(dtype=float)

    gate_model = _lgb_binary(params, n_jobs=lgb_jobs)
    gate_model.fit(
        X_tr,
        split.train["take_label"].to_numpy(dtype=int),
        sample_weight=w_tr,
        eval_set=[(X_va, split.val["take_label"].to_numpy(dtype=int))],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    dir_model = _lgb_binary(params, n_jobs=lgb_jobs)
    dir_model.fit(
        X_tr,
        split.train["direction_label"].to_numpy(dtype=int),
        sample_weight=w_tr,
        eval_set=[(X_va, split.val["direction_label"].to_numpy(dtype=int))],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    price_to_id = {name: idx for idx, name in enumerate(price_actions)}
    y_price_tr = split.train["price_label"].map(price_to_id).to_numpy(dtype=int)
    y_price_va = split.val["price_label"].map(price_to_id).to_numpy(dtype=int)
    price_model = _lgb_multiclass(params, num_class=len(price_actions), n_jobs=lgb_jobs)
    price_model.fit(
        X_tr,
        y_price_tr,
        sample_weight=w_tr,
        eval_set=[(X_va, y_price_va)],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    return {
        "gate_model": gate_model,
        "dir_model": dir_model,
        "price_model": price_model,
        "X_va": X_va,
        "X_te": X_te,
        "price_to_id": price_to_id,
        "id_to_price": {idx: name for name, idx in price_to_id.items()},
        "split": split,
    }


def _apply_calibration_meta(meta: dict[str, Any], probs: np.ndarray) -> np.ndarray:
    base = np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6)
    kind = str((meta or {}).get("selected") or (meta or {}).get("kind") or "none")
    if kind == "temperature":
        temp = float(meta.get("temperature") or 1.0)
        logit = np.log(base / (1.0 - base))
        return 1.0 / (1.0 + np.exp(-(logit / temp)))
    if kind == "isotonic":
        x = np.asarray(meta.get("x_thresholds") or [], dtype=float)
        y = np.asarray(meta.get("y_thresholds") or [], dtype=float)
        if len(x) >= 2 and len(x) == len(y):
            return np.clip(np.interp(base, x, y, left=y[0], right=y[-1]), 1e-6, 1.0 - 1e-6)
    return base


def _select_calibration(
    dir_val_prob: np.ndarray,
    y_val: np.ndarray,
    gate_val_prob: np.ndarray,
    price_val_labels: np.ndarray,
    take_threshold: float,
) -> dict[str, Any]:
    base_prob = np.clip(np.asarray(dir_val_prob, dtype=float), 1e-6, 1.0 - 1e-6)
    y_val = np.asarray(y_val, dtype=int)
    baseline_trade_ratio = float(((gate_val_prob >= take_threshold) & (price_val_labels != "ABSTAIN")).mean()) if len(y_val) else 0.0
    candidates: list[tuple[dict[str, Any], np.ndarray]] = [({"selected": "none", "kind": "none"}, base_prob)]
    for temp in [0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.25]:
        logit = np.log(base_prob / (1.0 - base_prob))
        cal = 1.0 / (1.0 + np.exp(-(logit / temp)))
        candidates.append(({"selected": "temperature", "kind": "temperature", "temperature": temp}, cal))
    for bins in [6, 8, 10, 12, 14, 18]:
        try:
            ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            ir.fit(base_prob, y_val)
            cal = np.clip(ir.predict(base_prob), 1e-6, 1.0 - 1e-6)
            x = getattr(ir, "X_thresholds_", None)
            y = getattr(ir, "y_thresholds_", None)
            if x is None or y is None:
                continue
            candidates.append(({
                "selected": "isotonic",
                "kind": "isotonic",
                "bins": bins,
                "x_thresholds": [float(v) for v in x],
                "y_thresholds": [float(v) for v in y],
            }, cal))
        except Exception:
            continue

    best_score = float("inf")
    best_meta = {"selected": "none", "kind": "none"}
    for meta, cal_prob in candidates:
        chosen_dir = (cal_prob >= 0.5).astype(int)
        conf = np.maximum(cal_prob, 1.0 - cal_prob)
        wrong = chosen_dir != y_val
        high_conf_loss_rate = float((wrong & (conf >= 0.60)).mean()) if len(y_val) else 1.0
        brier = brier_score_loss(y_val, cal_prob) if len(y_val) else 1.0
        trade_ratio = float(((gate_val_prob >= take_threshold) & (price_val_labels != "ABSTAIN")).mean()) if len(y_val) else 0.0
        if baseline_trade_ratio > 0 and trade_ratio < baseline_trade_ratio * 0.85:
            continue
        score = brier + high_conf_loss_rate
        if score < best_score:
            best_score = score
            best_meta = dict(meta)
            best_meta["brier"] = _round_float(brier, 6)
            best_meta["high_conf_loss_rate"] = _round_float(high_conf_loss_rate, 6)
            best_meta["trade_ratio"] = _round_float(trade_ratio, 6)
    return best_meta


def _simulate_eval(split: SplitData, dir_prob: np.ndarray, gate_prob: np.ndarray, price_labels: np.ndarray, proxy_price: np.ndarray, take_threshold: float) -> dict[str, Any]:
    y = split.test["direction_label"].to_numpy(dtype=int)
    chosen_dir = (dir_prob >= 0.5).astype(int)
    conf = np.maximum(dir_prob, 1.0 - dir_prob)
    take_trade = (gate_prob >= take_threshold) & (price_labels != "ABSTAIN")
    limit_prices = np.array(
        [float(label) if str(label) != "ABSTAIN" else np.nan for label in price_labels],
        dtype=float,
    )
    fills = take_trade & (proxy_price <= limit_prices)
    correct = fills & (chosen_dir == y)
    wrong = fills & (chosen_dir != y)
    pnl = np.zeros(len(y), dtype=float)
    pnl[correct] = 1.0 - limit_prices[correct]
    pnl[wrong] = -limit_prices[wrong]
    cumulative = np.cumsum(pnl)
    peaks = np.maximum.accumulate(np.maximum(cumulative, 0.0))
    drawdowns = peaks - cumulative
    high_conf_losses = wrong & (conf >= 0.60)
    streak = 0
    max_streak = 0
    for flag in high_conf_losses:
        if flag:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    trades = int(fills.sum())
    avg_limit = float(np.nanmean(limit_prices[fills])) if trades else None
    win_rate = float(correct.sum() / trades) if trades else 0.0
    avg_expectancy = float(pnl[fills].mean()) if trades else 0.0
    return {
        "rows": int(len(y)),
        "trades": trades,
        "fills": trades,
        "abstain_rate": _round_float(1.0 - float(take_trade.mean())),
        "win_rate": _round_float(win_rate),
        "total_pnl": _round_float(float(pnl.sum()), 4),
        "avg_expectancy": _round_float(avg_expectancy, 6),
        "max_drawdown": _round_float(float(drawdowns.max()) if len(drawdowns) else 0.0, 4),
        "high_conf_loss_rate": _round_float(float(high_conf_losses.sum() / trades) if trades else 0.0, 6),
        "max_same_direction_loss_streak": int(max_streak),
        "avg_limit_price": _round_float(avg_limit, 4) if avg_limit is not None else None,
    }


def _candidate_score(metrics: dict[str, Any], baseline_trades: int | None = None) -> float:
    trades = int(metrics.get("trades") or 0)
    if baseline_trades and trades < int(round(baseline_trades * 0.85)):
        return -1e9
    return (
        0.35 * float(metrics.get("avg_expectancy") or 0.0)
        + 0.25 * float(metrics.get("total_pnl") or 0.0)
        - 0.20 * float(metrics.get("max_drawdown") or 0.0)
        - 0.15 * float(metrics.get("high_conf_loss_rate") or 0.0)
        - 0.05 * max(0, (baseline_trades or trades) - trades)
    )


def _build_asset_frame(asset: str, bundle_name: str, groups: Sequence[str]) -> pd.DataFrame:
    paths = get_data_paths()
    tech = build_tech_features(paths["data_src"], asset, tail_rows=0, freq_min=15)
    tech = add_multi_timeframe_features(tech, asset)
    merged = merge_sentiment_to_tech(tech, list(groups), paths, asset)
    if bundle_name.endswith("regime_lite") or bundle_name.endswith("wallet_regime"):
        merged = _ensure_regime_lite_features(merged)
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    return merged


def _run_stage_a(asset: str) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    bundle_map = _asset_bundle_map(asset)
    rows: list[dict[str, Any]] = []
    frames: dict[str, pd.DataFrame] = {}
    utility_cfg = _default_utility_cfg()
    for bundle_name, groups in bundle_map.items():
        df = _build_asset_frame(asset, bundle_name, groups)
        frames[bundle_name] = df
        feature_cols = _choose_feature_cols(df, bundle_name, groups)
        if not feature_cols:
            rows.append({"asset": asset, "bundle": bundle_name, "status": "skipped_no_features"})
            continue
        coarse_actions = _price_grid_for_asset(asset)
        labeled = _derive_labels(df, asset, coarse_actions, utility_cfg)
        for train_days in TRAIN_DAYS_CANDIDATES:
            split = _split_recent(labeled, train_days=train_days)
            if len(split.train) < 1200 or len(split.val) < 240 or len(split.test) < 240:
                rows.append({
                    "asset": asset,
                    "bundle": bundle_name,
                    "train_days": train_days,
                    "status": "skipped_insufficient_rows",
                    "train_rows": len(split.train),
                    "val_rows": len(split.val),
                    "test_rows": len(split.test),
                })
                continue
            for template_name, params in PARAM_TEMPLATES.items():
                fit = _fit_heads(split, feature_cols, params, coarse_actions, lgb_jobs=8)
                dir_test = fit["dir_model"].predict_proba(fit["X_te"])[:, 1]
                gate_test = fit["gate_model"].predict_proba(fit["X_te"])[:, 1]
                price_test = fit["price_model"].predict(fit["X_te"])
                price_labels = np.asarray([fit["id_to_price"][int(i)] for i in price_test], dtype=object)
                metrics = _simulate_eval(
                    split,
                    dir_prob=np.asarray(dir_test, dtype=float),
                    gate_prob=np.asarray(gate_test, dtype=float),
                    price_labels=price_labels,
                    proxy_price=split.test["price_proxy"].to_numpy(dtype=float),
                    take_threshold=0.50,
                )
                rows.append({
                    "asset": asset,
                    "bundle": bundle_name,
                    "groups": list(groups),
                    "feature_cols_count": len(feature_cols),
                    "train_days": train_days,
                    "template": template_name,
                    "params": params,
                    "utility_cfg": utility_cfg,
                    "metrics": metrics,
                    "score": _round_float(_candidate_score(metrics), 6),
                    "status": "ok",
                })
    leaderboard = pd.DataFrame(rows)
    if not leaderboard.empty and "score" in leaderboard.columns:
        leaderboard = leaderboard.sort_values(["score"], ascending=[False], na_position="last").reset_index(drop=True)
    return leaderboard, frames


def _objective_factory(asset: str, bundle_name: str, df: pd.DataFrame, groups: Sequence[str], train_days: int, coarse_top_prices: Sequence[float]):
    feature_cols = _choose_feature_cols(df, bundle_name, groups)
    fine_actions = _price_grid_for_asset(asset, center_prices=coarse_top_prices)

    def objective(trial: optuna.Trial) -> float:
        utility_cfg = {
            "abstain_cost_bias": trial.suggest_float("abstain_cost_bias", 0.0, 0.4),
            "fill_penalty_scale": trial.suggest_float("fill_penalty_scale", 0.05, 0.35),
            "move_cap": trial.suggest_float("move_cap", 1.2, 2.5),
            "wrong_loss_scale": trial.suggest_float("wrong_loss_scale", 0.7, 1.4),
        }
        labeled = _derive_labels(df, asset, fine_actions, utility_cfg)
        split = _split_recent(labeled, train_days=train_days)
        if len(split.train) < 1200 or len(split.val) < 240 or len(split.test) < 240:
            raise optuna.TrialPruned()
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 150),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.50, 0.98),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.50, 0.98),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.3),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.001, 3.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1.0, 40.0, log=True),
            "max_bin": trial.suggest_int("max_bin", 127, 511),
        }
        take_threshold = trial.suggest_float("take_threshold", 0.45, 0.70)
        fit = _fit_heads(split, feature_cols, params, fine_actions, lgb_jobs=1)
        dir_val = fit["dir_model"].predict_proba(fit["X_va"])[:, 1]
        dir_test = fit["dir_model"].predict_proba(fit["X_te"])[:, 1]
        gate_val = fit["gate_model"].predict_proba(fit["X_va"])[:, 1]
        gate_test = fit["gate_model"].predict_proba(fit["X_te"])[:, 1]
        price_val = fit["price_model"].predict(fit["X_va"])
        price_test = fit["price_model"].predict(fit["X_te"])
        price_val_labels = np.asarray([fit["id_to_price"][int(i)] for i in price_val], dtype=object)
        price_test_labels = np.asarray([fit["id_to_price"][int(i)] for i in price_test], dtype=object)
        cal_meta = _select_calibration(
            dir_val_prob=np.asarray(dir_val, dtype=float),
            y_val=split.val["direction_label"].to_numpy(dtype=int),
            gate_val_prob=np.asarray(gate_val, dtype=float),
            price_val_labels=price_val_labels,
            take_threshold=float(take_threshold),
        )
        cal_test = _apply_calibration_meta(cal_meta, np.asarray(dir_test, dtype=float))
        metrics = _simulate_eval(
            split,
            dir_prob=cal_test,
            gate_prob=np.asarray(gate_test, dtype=float),
            price_labels=price_test_labels,
            proxy_price=split.test["price_proxy"].to_numpy(dtype=float),
            take_threshold=float(take_threshold),
        )
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("calibration", cal_meta)
        trial.set_user_attr("price_actions", fine_actions)
        trial.set_user_attr("utility_cfg", utility_cfg)
        trial.set_user_attr("feature_cols", feature_cols)
        return _candidate_score(metrics)

    return objective


def _window_list_for_train_days(train_days: int) -> list[int]:
    for ws in WINDOW_SETS:
        if max(ws) >= train_days:
            return list(ws)
    return list(WINDOW_SETS[-1])


def _train_asset(asset: str) -> dict[str, Any]:
    model_dir = BTC_MODEL_DIR if asset == "BTC_USDT" else ETH_MODEL_DIR
    leaderboard_path = BTC_LEADERBOARD if asset == "BTC_USDT" else ETH_LEADERBOARD
    stage_a, frames = _run_stage_a(asset)
    write_json(leaderboard_path, {
        "generated_at": utc_now_iso(),
        "asset": asset,
        "stage": "coarse",
        "rows": stage_a.to_dict(orient="records") if not stage_a.empty else [],
    })
    top_stage_a = stage_a[stage_a["status"] == "ok"].head(3).to_dict(orient="records") if not stage_a.empty else []
    finalists: list[dict[str, Any]] = []
    for rank, row in enumerate(top_stage_a, start=1):
        bundle_name = str(row["bundle"])
        groups = list((_asset_bundle_map(asset))[bundle_name])
        train_days = int(row["train_days"])
        df = frames[bundle_name]
        coarse_actions = _price_grid_for_asset(asset)
        labeled = _derive_labels(df, asset, coarse_actions, _default_utility_cfg())
        split = _split_recent(labeled, train_days=train_days)
        top_prices: list[float] = []
        for key in split.train["price_label"].value_counts().index:
            if key != "ABSTAIN":
                top_prices.append(float(key))
            if len(top_prices) >= 2:
                break
        if not top_prices:
            top_prices = [0.50 if asset == "BTC_USDT" else 0.49]
        objective = _objective_factory(asset, bundle_name, df, groups, train_days, top_prices)
        study = optuna.create_study(direction="maximize", study_name=f"{asset}_{bundle_name}_{rank}")
        study.optimize(objective, n_trials=120, n_jobs=SEARCH_WORKERS, show_progress_bar=False)
        best = study.best_trial
        finalists.append({
            "rank": rank,
            "bundle": bundle_name,
            "groups": groups,
            "train_days": train_days,
            "seed_template": row.get("template"),
            "coarse_score": row.get("score"),
            "best_value": _round_float(best.value, 6),
            "best_params": best.params,
            "best_metrics": best.user_attrs.get("metrics", {}),
            "best_calibration": best.user_attrs.get("calibration", {}),
            "price_actions": best.user_attrs.get("price_actions", []),
            "utility_cfg": best.user_attrs.get("utility_cfg", {}),
            "feature_cols": best.user_attrs.get("feature_cols", []),
        })
    finalists = sorted(finalists, key=lambda x: float(x.get("best_value") or -1e9), reverse=True)
    if not finalists:
        raise RuntimeError(f"no finalists for {asset}")
    winner = finalists[0]
    bundle_name = str(winner["bundle"])
    groups = list((_asset_bundle_map(asset))[bundle_name])
    feature_cols = list(winner["feature_cols"])
    price_actions = list(winner["price_actions"])
    utility_cfg = dict(winner["utility_cfg"])
    labeled = _derive_labels(frames[bundle_name], asset, price_actions, utility_cfg)
    windows_to_use = _window_list_for_train_days(int(winner["train_days"]))
    dir_files: list[str] = []
    gate_files: list[str] = []
    price_files: list[str] = []
    rows_meta: list[dict[str, Any]] = []
    params = dict(winner["best_params"])
    take_threshold = float(params.pop("take_threshold"))
    for w in windows_to_use:
        split = _split_recent(labeled, train_days=w)
        if len(split.train) < 1200 or len(split.val) < 240 or len(split.test) < 240:
            rows_meta.append({"window_days": w, "status": "skipped_insufficient_rows"})
            continue
        fit = _fit_heads(split, feature_cols, params, price_actions, lgb_jobs=8)
        dir_val = fit["dir_model"].predict_proba(fit["X_va"])[:, 1]
        gate_val = fit["gate_model"].predict_proba(fit["X_va"])[:, 1]
        price_val = fit["price_model"].predict(fit["X_va"])
        price_val_labels = np.asarray([fit["id_to_price"][int(i)] for i in price_val], dtype=object)
        cal_meta = _select_calibration(
            dir_val_prob=np.asarray(dir_val, dtype=float),
            y_val=split.val["direction_label"].to_numpy(dtype=int),
            gate_val_prob=np.asarray(gate_val, dtype=float),
            price_val_labels=price_val_labels,
            take_threshold=take_threshold,
        )
        dir_file = f"direction_lgb_{w}d.joblib"
        gate_file = f"edge_gate_lgb_{w}d.joblib"
        price_file = f"price_head_lgb_{w}d.joblib"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(fit["dir_model"], model_dir / dir_file)
        joblib.dump(fit["gate_model"], model_dir / gate_file)
        joblib.dump(fit["price_model"], model_dir / price_file)
        dir_files.append(dir_file)
        gate_files.append(gate_file)
        price_files.append(price_file)
        rows_meta.append({
            "window_days": w,
            "status": "trained",
            "direction_best_iteration": int(getattr(fit["dir_model"], "best_iteration_", 0) or 0),
            "gate_best_iteration": int(getattr(fit["gate_model"], "best_iteration_", 0) or 0),
            "price_best_iteration": int(getattr(fit["price_model"], "best_iteration_", 0) or 0),
            "take_threshold": _round_float(take_threshold, 6),
            "calibration": cal_meta,
        })
    config = {
        "generated_at": utc_now_iso(),
        "version": ("vnext_btc_dyn_v2" if asset == "BTC_USDT" else "vnext_eth_dyn_v2"),
        "asset": asset,
        "compare_only": True,
        "monitor_only": True,
        "mode": "three_head_compare_only",
        "feature_bundle": bundle_name,
        "feature_groups": groups,
        "feature_cols": feature_cols,
        "window_days_list": windows_to_use,
        "best_train_days_seed": int(winner["train_days"]),
        "price_actions": price_actions,
        "take_threshold": _round_float(take_threshold, 6),
        "best_params": winner["best_params"],
        "best_metrics": winner["best_metrics"],
        "calibration": rows_meta[0]["calibration"] if rows_meta and rows_meta[0].get("calibration") else winner.get("best_calibration", {}),
        "utility_cfg": utility_cfg,
        "model_files": {
            "direction": dir_files,
            "edge_gate": gate_files,
            "price_head": price_files,
        },
        "training_rows": rows_meta,
        "notes": [
            "serious compare-only v2",
            "btc_eth_separate_heads",
            "dynamic_price_coarse_plus_fine",
            "no_cfgi_news_gru_external",
            "post_init_weighted",
        ],
    }
    write_json(model_dir / "config.json", config)
    write_json(model_dir / "feature_cols.json", feature_cols)
    write_json(model_dir / "asset_map.json", {asset: 0})
    write_json(leaderboard_path, {
        "generated_at": utc_now_iso(),
        "asset": asset,
        "stage": "final",
        "coarse_rows": stage_a.to_dict(orient="records") if not stage_a.empty else [],
        "finalists": finalists,
        "winner": winner,
        "model_dir": str(model_dir),
    })
    return {
        "asset": asset,
        "model_dir": str(model_dir),
        "winner": winner,
        "training_rows": rows_meta,
        "leaderboard_path": str(leaderboard_path),
    }


def _evaluate_v1(asset: str, split: SplitData) -> dict[str, Any]:
    root = MODELS / "vnext_btceth_v1"
    if not root.exists():
        return {}
    cfg = json.loads((root / "config.json").read_text(encoding="utf-8"))
    v1_feature_cols = json.loads((root / "feature_cols.json").read_text(encoding="utf-8"))
    asset_map = cfg.get("asset_map") or json.loads((root / "asset_map.json").read_text(encoding="utf-8"))
    asset_id = int(asset_map[asset])
    X = split.test.copy()
    for col in v1_feature_cols:
        if col not in X.columns:
            X[col] = 0.0
    if "asset_id" not in X.columns:
        X["asset_id"] = asset_id
    X["asset_id"] = asset_id
    X = X[v1_feature_cols]
    probas = []
    for window in cfg.get("window_days_list", [60, 90, 120]):
        model = joblib.load(root / f"lgb_{window}d.joblib")
        probas.append(model.predict_proba(X)[:, 1])
    dir_prob = np.mean(probas, axis=0)
    conf = np.maximum(dir_prob, 1.0 - dir_prob)
    price_labels = np.array(["0.50"] * len(X), dtype=object)
    gate_prob = np.where(conf >= 0.53, 1.0, 0.0)
    return _simulate_eval(
        split=split,
        dir_prob=np.asarray(dir_prob, dtype=float),
        gate_prob=np.asarray(gate_prob, dtype=float),
        price_labels=price_labels,
        proxy_price=split.test["price_proxy"].to_numpy(dtype=float),
        take_threshold=0.50,
    )


def _exp10_reference(asset: str) -> Any:
    p = REPORTS / "mainline_exp10_drawdown_diagnosis_latest.json"
    if not p.exists():
        return {}
    payload = json.loads(p.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    selected = [r for r in rows if isinstance(r, dict) and r.get("symbol") == asset.split("_")[0] and str(r.get("family") or "") == "v5_exp10"]
    selected.sort(key=lambda x: float(x.get("total_pnl") or 0.0))
    return selected[:4]


def _write_compare_report(results: dict[str, Any]) -> None:
    write_json(COMPARE_REPORT, {"generated_at": utc_now_iso(), **results})


def main() -> int:
    parser = argparse.ArgumentParser(description="Train serious BTC/ETH compare-only v2")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    btc_result = _train_asset("BTC_USDT")
    eth_result = _train_asset("ETH_USDT")

    compare_payload: dict[str, Any] = {
        "generated_at": utc_now_iso(),
        "status": "trained",
        "scope": "compare_only_v2_btc_eth",
        "assets": {},
        "notes": [
            "not mainline replacement",
            "compare_only only",
            "ranked by post-init utility metrics, not AUC",
            "dynamic price head enabled",
            "expanded hyperparameter and price search",
        ],
    }
    for asset, result in [("BTC_USDT", btc_result), ("ETH_USDT", eth_result)]:
        model_dir = Path(result["model_dir"])
        config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
        bundle_name = config["feature_bundle"]
        groups = config["feature_groups"]
        df = _build_asset_frame(asset, bundle_name, groups)
        price_actions = config["price_actions"]
        labeled = _derive_labels(df, asset, price_actions, config["utility_cfg"])
        split = _split_recent(labeled, train_days=int(config["best_train_days_seed"]))
        feature_cols = config["feature_cols"]
        X = split.test.copy()
        for col in feature_cols:
            if col not in X.columns:
                X[col] = 0.0
        X = X[feature_cols]
        dir_probs = []
        gate_probs = []
        price_probs = []
        for name in config["model_files"]["direction"]:
            model = joblib.load(model_dir / name)
            dir_probs.append(model.predict_proba(X)[:, 1])
        for name in config["model_files"]["edge_gate"]:
            model = joblib.load(model_dir / name)
            gate_probs.append(model.predict_proba(X)[:, 1])
        for name in config["model_files"]["price_head"]:
            model = joblib.load(model_dir / name)
            price_probs.append(model.predict_proba(X))
        dir_prob = np.mean(dir_probs, axis=0)
        gate_prob = np.mean(gate_probs, axis=0)
        price_pred = np.argmax(np.mean(price_probs, axis=0), axis=1)
        id_to_price = {idx: name for idx, name in enumerate(price_actions)}
        price_labels = np.asarray([id_to_price[int(i)] for i in price_pred], dtype=object)
        dir_prob = _apply_calibration_meta(config.get("calibration") or {}, np.asarray(dir_prob, dtype=float))
        metrics_v2 = _simulate_eval(
            split=split,
            dir_prob=np.asarray(dir_prob, dtype=float),
            gate_prob=np.asarray(gate_prob, dtype=float),
            price_labels=price_labels,
            proxy_price=split.test["price_proxy"].to_numpy(dtype=float),
            take_threshold=float(config["take_threshold"]),
        )
        metrics_v1 = _evaluate_v1(asset, split)
        compare_payload["assets"][asset] = {
            "v2": metrics_v2,
            "v1": metrics_v1,
            "winner": config.get("best_metrics", {}),
            "model_dir": str(model_dir),
            "exp10_reference": _exp10_reference(asset),
        }

    _write_compare_report(compare_payload)
    write_json(TRAIN_REPORT, {
        "generated_at": utc_now_iso(),
        "status": "trained",
        "scope": "vnext_btceth_dyn_v2",
        "assets": {
            "BTC_USDT": btc_result,
            "ETH_USDT": eth_result,
        },
        "compare_report": str(COMPARE_REPORT),
    })
    print(json.dumps({
        "status": "trained",
        "btc_model_dir": str(BTC_MODEL_DIR),
        "eth_model_dir": str(ETH_MODEL_DIR),
        "compare_report": str(COMPARE_REPORT),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
