#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import lightgbm as lgb
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops.core10_retrain_common import (
    ACTION_LABELS,
    ACTION_TO_ID,
    DATA,
    STATE_LABELS,
    STATE_TO_ID,
    _add_trade_objective_labels,
    build_cell_dataset,
    choose_tree_feature_cols,
    load_core_cell_map,
    normalize_label_config,
    resolve_current_model_dir,
    train_val_test_split,
    utc_now_iso,
    write_json,
)


TREE_ROOT = DATA / "models"

DEFAULT_TREE_PARAMS = {
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": 7,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 4.0,
}

DEFAULT_ACTION_CONFIG = {
    "action_utility_threshold": 0.10,
    "action_margin_min": 0.05,
    "action_positive_weight_boost": 1.35,
    "action_countertrend_weight_boost": 1.50,
    "abstain_weight_boost": 1.40,
    "observed_action_positive_roi_min": 0.02,
    "observed_action_bad_roi_max": -0.02,
    "observed_action_margin_min": 0.03,
    "observed_action_weight_boost": 2.25,
    "recent_hard_negative_days": 14.0,
    "recent_hard_negative_weight_boost": 2.50,
    "recent_hard_negative_up_weight_boost": 3.50,
    "recent_hard_negative_down_weight_boost": 2.00,
    "trend_weight": 0.45,
    "chop_penalty": 0.20,
    "abstain_gate_blend": 0.50,
    "action_margin_power": 1.0,
    "market_final_recent_negative_up_weight_boost": 4.00,
    "market_final_recent_negative_down_weight_boost": 2.50,
}

ACTION_LABEL_VERSION = "observed_trade_overlay_v2"
MARKET_ACTION_LABEL_VERSION = "market_settled_overlay_v3"
MARKET_FINAL_ACTION_LABEL_VERSION = "market_final_action_v1"


def _normalize_tree_params(tree_params: Dict[str, Any] | None = None) -> Dict[str, Any]:
    params = dict(DEFAULT_TREE_PARAMS)
    if isinstance(tree_params, dict):
        for key in params:
            raw = tree_params.get(key)
            if raw is None:
                continue
            try:
                params[key] = float(raw) if key not in {"num_leaves", "max_depth", "min_data_in_leaf", "bagging_freq"} else int(raw)
            except Exception:
                continue
    return params


def _normalize_action_config(action_config: Dict[str, Any] | None = None) -> Dict[str, float]:
    cfg = dict(DEFAULT_ACTION_CONFIG)
    if isinstance(action_config, dict):
        for key in cfg:
            raw = action_config.get(key)
            if raw is None:
                continue
            try:
                cfg[key] = float(raw)
            except Exception:
                continue
    return cfg


def resolve_action_label_version(label_config: Dict[str, Any] | None = None) -> str:
    cfg = normalize_label_config(label_config)
    if float(cfg.get("market_final_action_strength") or 0.0) > 0:
        return MARKET_FINAL_ACTION_LABEL_VERSION
    if float(cfg.get("market_trade_overlay_strength") or 0.0) > 0:
        return MARKET_ACTION_LABEL_VERSION
    return ACTION_LABEL_VERSION


def candidate_dir_for_cell(cell_id: str, output_tag: str | None = None) -> Path:
    safe = cell_id.replace("/", "__").replace("-", "_").lower()
    if output_tag:
        safe_tag = output_tag.replace("/", "__").replace("-", "_").lower()
        return TREE_ROOT / f"core10_action_{safe}__{safe_tag}"
    return TREE_ROOT / f"core10_action_{safe}"


def _fit_state_model(X_tr, y_tr, w_tr, X_va, y_va, tree_params: Dict[str, Any]):
    train_set = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
    valid_set = lgb.Dataset(X_va, label=y_va, reference=train_set)
    params = {
        "objective": "multiclass",
        "num_class": len(STATE_LABELS),
        "learning_rate": float(tree_params["learning_rate"]),
        "num_leaves": int(tree_params["num_leaves"]),
        "max_depth": int(tree_params["max_depth"]),
        "min_data_in_leaf": int(tree_params["min_data_in_leaf"]),
        "feature_fraction": float(tree_params["feature_fraction"]),
        "bagging_fraction": float(tree_params["bagging_fraction"]),
        "bagging_freq": int(tree_params["bagging_freq"]),
        "lambda_l2": float(tree_params["lambda_l2"]),
        "num_threads": 1,
        "verbosity": -1,
        "metric": "multi_logloss",
    }
    return lgb.train(
        params,
        train_set,
        num_boost_round=400,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )


def _fit_action_model(X_tr, y_tr, w_tr, X_va, y_va, tree_params: Dict[str, Any]):
    train_set = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
    valid_set = lgb.Dataset(X_va, label=y_va, reference=train_set)
    params = {
        "objective": "multiclass",
        "num_class": len(ACTION_LABELS),
        "learning_rate": float(tree_params["learning_rate"]),
        "num_leaves": int(tree_params["num_leaves"]),
        "max_depth": int(tree_params["max_depth"]),
        "min_data_in_leaf": int(tree_params["min_data_in_leaf"]),
        "feature_fraction": float(tree_params["feature_fraction"]),
        "bagging_fraction": float(tree_params["bagging_fraction"]),
        "bagging_freq": int(tree_params["bagging_freq"]),
        "lambda_l2": float(tree_params["lambda_l2"]),
        "num_threads": 1,
        "verbosity": -1,
        "metric": "multi_logloss",
    }
    return lgb.train(
        params,
        train_set,
        num_boost_round=400,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )


def _fit_abstain(X_tr, y_tr, w_tr, X_va, y_va, tree_params: Dict[str, Any]):
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=400,
        learning_rate=float(tree_params["learning_rate"]),
        num_leaves=int(tree_params["num_leaves"]),
        max_depth=int(tree_params["max_depth"]),
        min_child_samples=int(tree_params["min_data_in_leaf"]),
        feature_fraction=float(tree_params["feature_fraction"]),
        bagging_fraction=float(tree_params["bagging_fraction"]),
        bagging_freq=int(tree_params["bagging_freq"]),
        lambda_l2=float(tree_params["lambda_l2"]),
        n_jobs=1,
        verbosity=-1,
    )
    model.fit(
        X_tr,
        y_tr,
        sample_weight=w_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    return model


def _build_action_labels(df, action_config: Dict[str, float], label_config: Dict[str, Any] | None = None):
    out = df.copy()
    label_cfg = normalize_label_config(label_config)
    up_util = out["up_utility_label"].to_numpy(dtype=float)
    down_util = out["down_utility_label"].to_numpy(dtype=float)
    abstain = out["abstain_label"].to_numpy(dtype=int)
    threshold = float(action_config["action_utility_threshold"])
    margin_min = float(action_config["action_margin_min"])
    best_is_up = up_util >= down_util
    best_util = np.maximum(up_util, down_util)
    margin = np.abs(up_util - down_util)
    action = np.full(len(out), "ABSTAIN", dtype=object)
    trade_mask = (abstain == 0) & (best_util > threshold) & (margin >= margin_min)
    action[trade_mask & best_is_up] = "UP"
    action[trade_mask & (~best_is_up)] = "DOWN"

    if "market_trade_best_roi" in out.columns and float(np.nan_to_num(out.get("market_trade_settled_markets", 0), nan=0.0).sum()) > 0:
        up_roi = out["market_trade_roi_up"].to_numpy(dtype=float) if "market_trade_roi_up" in out.columns else np.full(len(out), np.nan)
        down_roi = out["market_trade_roi_down"].to_numpy(dtype=float) if "market_trade_roi_down" in out.columns else np.full(len(out), np.nan)
        up_count = out["market_trade_count_up"].to_numpy(dtype=float) if "market_trade_count_up" in out.columns else np.zeros(len(out), dtype=float)
        down_count = out["market_trade_count_down"].to_numpy(dtype=float) if "market_trade_count_down" in out.columns else np.zeros(len(out), dtype=float)
        observed_margin = out["market_trade_roi_margin"].to_numpy(dtype=float) if "market_trade_roi_margin" in out.columns else np.zeros(len(out), dtype=float)
        observed_trade_weight = out["market_trade_settled_markets"].to_numpy(dtype=float) if "market_trade_settled_markets" in out.columns else np.ones(len(out), dtype=float)
    else:
        up_roi = out["trade_roi_up"].to_numpy(dtype=float) if "trade_roi_up" in out.columns else np.full(len(out), np.nan)
        down_roi = out["trade_roi_down"].to_numpy(dtype=float) if "trade_roi_down" in out.columns else np.full(len(out), np.nan)
        up_count = out["trade_count_up"].to_numpy(dtype=float) if "trade_count_up" in out.columns else np.zeros(len(out), dtype=float)
        down_count = out["trade_count_down"].to_numpy(dtype=float) if "trade_count_down" in out.columns else np.zeros(len(out), dtype=float)
        observed_margin = np.where(
            np.isfinite(up_roi) & np.isfinite(down_roi),
            np.abs(up_roi - down_roi),
            np.maximum(np.abs(np.nan_to_num(up_roi, nan=0.0)), np.abs(np.nan_to_num(down_roi, nan=0.0))),
        )
        observed_trade_weight = np.ones(len(out), dtype=float)
    observed_min = float(action_config["observed_action_positive_roi_min"])
    observed_bad = float(action_config["observed_action_bad_roi_max"])
    observed_margin_min = float(action_config["observed_action_margin_min"])
    observed_mask = (up_count + down_count) > 0
    observed_best_up = np.nan_to_num(up_roi, nan=-999.0) >= np.nan_to_num(down_roi, nan=-999.0)
    observed_best_roi = np.maximum(np.nan_to_num(up_roi, nan=-999.0), np.nan_to_num(down_roi, nan=-999.0))
    observed_action = np.full(len(out), "ABSTAIN", dtype=object)
    observed_trade_mask = observed_mask & (observed_best_roi >= observed_min) & (observed_margin >= observed_margin_min)
    observed_action[observed_trade_mask & observed_best_up] = "UP"
    observed_action[observed_trade_mask & (~observed_best_up)] = "DOWN"
    observed_abstain_mask = observed_mask & ((observed_best_roi <= observed_bad) | (observed_margin < observed_margin_min))
    observed_action[observed_abstain_mask] = "ABSTAIN"
    action = np.where(observed_mask, observed_action, action)
    follow = out["follow_side"].to_numpy(dtype=object) if "follow_side" in out.columns else np.full(len(out), "NEUTRAL", dtype=object)

    market_final_strength = float(label_cfg.get("market_final_action_strength") or 0.0)
    market_final_trade_mask = np.zeros(len(out), dtype=bool)
    market_final_abstain_mask = np.zeros(len(out), dtype=bool)
    regime_cluster_trade_mask = np.zeros(len(out), dtype=bool)
    regime_cluster_abstain_mask = np.zeros(len(out), dtype=bool)
    if market_final_strength > 0 and observed_mask.any():
        final_good = float(label_cfg["market_final_action_good_roi_min"])
        final_bad = float(label_cfg["market_final_action_bad_roi_max"])
        final_margin = float(label_cfg["market_final_action_margin_min"])
        market_final_trade_mask = observed_mask & (observed_best_roi >= final_good) & (observed_margin >= final_margin)
        market_final_abstain_mask = observed_mask & (
            (observed_best_roi <= final_bad) | (observed_margin < final_margin)
        )
        market_final_action = action.copy()
        market_final_action[market_final_trade_mask & observed_best_up] = "UP"
        market_final_action[market_final_trade_mask & (~observed_best_up)] = "DOWN"
        market_final_action[market_final_abstain_mask] = "ABSTAIN"
        action = np.where(observed_mask, market_final_action, action)

    regime_cluster_strength = float(label_cfg.get("regime_cluster_hard_negative_strength") or 0.0)
    if regime_cluster_strength > 0 and observed_mask.any():
        cluster_good = float(label_cfg["regime_cluster_good_roi_min"])
        cluster_bad = float(label_cfg["regime_cluster_bad_roi_max"])
        cluster_margin = float(label_cfg["regime_cluster_margin_min"])
        down_regime = observed_mask & (follow == "DOWN")
        up_regime = observed_mask & (follow == "UP")
        regime_good_down = down_regime & np.isfinite(down_roi) & (down_roi >= cluster_good) & (observed_margin >= cluster_margin)
        regime_good_up = up_regime & np.isfinite(up_roi) & (up_roi >= cluster_good) & (observed_margin >= cluster_margin)
        regime_bad_up = down_regime & np.isfinite(up_roi) & (up_roi <= cluster_bad)
        regime_bad_down = up_regime & np.isfinite(down_roi) & (down_roi <= cluster_bad)
        regime_cluster_trade_mask = regime_good_down | regime_good_up
        regime_cluster_abstain_mask = (regime_bad_up & ~regime_good_down) | (regime_bad_down & ~regime_good_up)
        regime_cluster_action = action.copy()
        regime_cluster_action[regime_good_down] = "DOWN"
        regime_cluster_action[regime_good_up] = "UP"
        regime_cluster_action[regime_cluster_abstain_mask] = "ABSTAIN"
        action = np.where(observed_mask, regime_cluster_action, action)

    out["action_label"] = [ACTION_TO_ID[a] for a in action]

    base = ((out["up_head_weight"].to_numpy(dtype=float) + out["down_head_weight"].to_numpy(dtype=float)) / 2.0)
    is_trade = action != "ABSTAIN"
    is_counter = is_trade & (follow != "NEUTRAL") & (action != follow)
    action_weight = base * np.where(is_trade, float(action_config["action_positive_weight_boost"]), 1.0)
    action_weight = action_weight * np.where(is_counter, float(action_config["action_countertrend_weight_boost"]), 1.0)
    observed_weight_boost = np.where(
        observed_mask,
        float(action_config["observed_action_weight_boost"]) * np.clip(observed_trade_weight, 1.0, 3.0),
        1.0,
    )
    action_weight = action_weight * observed_weight_boost
    latest_ts = out["timestamp"].max() if "timestamp" in out.columns and len(out) else None
    if latest_ts is not None:
        age_days = (latest_ts - out["timestamp"]).dt.total_seconds().div(86400.0).to_numpy(dtype=float)
    else:
        age_days = np.full(len(out), 999.0, dtype=float)
    recent_hard_negative_days = float(action_config["recent_hard_negative_days"])
    recent_hard_negative = (
        observed_mask
        & (age_days <= recent_hard_negative_days)
        & (
            ((action == "UP") & np.isfinite(up_roi) & (up_roi <= observed_bad))
            | ((action == "DOWN") & np.isfinite(down_roi) & (down_roi <= observed_bad))
        )
    )
    recent_hard_negative_up = recent_hard_negative & (action == "UP")
    recent_hard_negative_down = recent_hard_negative & (action == "DOWN")
    action_weight = action_weight * np.where(recent_hard_negative, float(action_config["recent_hard_negative_weight_boost"]), 1.0)
    action_weight = action_weight * np.where(
        recent_hard_negative_up,
        float(action_config["recent_hard_negative_up_weight_boost"]) / max(float(action_config["recent_hard_negative_weight_boost"]), 1e-9),
        1.0,
    )
    action_weight = action_weight * np.where(
        recent_hard_negative_down,
        float(action_config["recent_hard_negative_down_weight_boost"]) / max(float(action_config["recent_hard_negative_weight_boost"]), 1e-9),
        1.0,
    )
    abstain_weight = out["abstain_head_weight"].to_numpy(dtype=float) * np.where(abstain == 1, float(action_config["abstain_weight_boost"]), 1.0)
    abstain_weight = abstain_weight * np.where(observed_abstain_mask, float(action_config["observed_action_weight_boost"]) * np.clip(observed_trade_weight, 1.0, 3.0), 1.0)
    if market_final_strength > 0:
        trade_boost = 1.0 + (float(label_cfg["market_final_action_weight_boost"]) - 1.0) * market_final_strength
        abstain_boost = 1.0 + (float(label_cfg["market_final_abstain_weight_boost"]) - 1.0) * market_final_strength
        action_weight = action_weight * np.where(
            market_final_trade_mask,
            trade_boost * np.clip(observed_trade_weight, 1.0, 3.0),
            1.0,
        )
        abstain_weight = abstain_weight * np.where(
            market_final_abstain_mask,
            abstain_boost * np.clip(observed_trade_weight, 1.0, 3.0),
            1.0,
        )
        recent_final_negative_days = float(label_cfg["market_final_recent_negative_days"])
        final_bad = float(label_cfg["market_final_action_bad_roi_max"])
        market_final_recent_negative = (
            observed_mask
            & (age_days <= recent_final_negative_days)
            & (
                ((action == "UP") & np.isfinite(up_roi) & (up_roi <= final_bad))
                | ((action == "DOWN") & np.isfinite(down_roi) & (down_roi <= final_bad))
            )
        )
        market_final_recent_negative_up = market_final_recent_negative & (action == "UP")
        market_final_recent_negative_down = market_final_recent_negative & (action == "DOWN")
        action_weight = action_weight * np.where(
            market_final_recent_negative_up,
            1.0 + (float(action_config["market_final_recent_negative_up_weight_boost"]) - 1.0) * market_final_strength,
            1.0,
        )
        action_weight = action_weight * np.where(
            market_final_recent_negative_down,
            1.0 + (float(action_config["market_final_recent_negative_down_weight_boost"]) - 1.0) * market_final_strength,
            1.0,
        )
    else:
        market_final_recent_negative = np.zeros(len(out), dtype=bool)
    if regime_cluster_strength > 0:
        cluster_trade_boost = 1.0 + (float(label_cfg["regime_cluster_action_weight_boost"]) - 1.0) * regime_cluster_strength
        cluster_abstain_boost = 1.0 + (float(label_cfg["regime_cluster_abstain_weight_boost"]) - 1.0) * regime_cluster_strength
        action_weight = action_weight * np.where(
            regime_cluster_trade_mask,
            cluster_trade_boost * np.clip(observed_trade_weight, 1.0, 3.0),
            1.0,
        )
        abstain_weight = abstain_weight * np.where(
            regime_cluster_abstain_mask,
            cluster_abstain_boost * np.clip(observed_trade_weight, 1.0, 3.0),
            1.0,
        )
    out["action_weight"] = action_weight
    out["abstain_train_weight"] = abstain_weight
    out["action_name"] = action
    out["observed_trade_label"] = observed_action
    out["recent_hard_negative"] = recent_hard_negative.astype(int)
    out["market_final_trade_mask"] = market_final_trade_mask.astype(int)
    out["market_final_abstain_mask"] = market_final_abstain_mask.astype(int)
    out["market_final_recent_negative"] = market_final_recent_negative.astype(int)
    out["regime_cluster_trade_mask"] = regime_cluster_trade_mask.astype(int)
    out["regime_cluster_abstain_mask"] = regime_cluster_abstain_mask.astype(int)
    return out


def train_one(
    cell_id: str,
    windows: List[int],
    force: bool = False,
    label_config: Dict[str, Any] | None = None,
    tree_params: Dict[str, Any] | None = None,
    action_config: Dict[str, Any] | None = None,
    output_tag: str | None = None,
    label_each_split: bool = False,
    split_config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    cells = load_core_cell_map()
    if cell_id not in cells:
        raise KeyError(f"unknown cell_id: {cell_id}")
    cell = cells[cell_id]
    label_config = normalize_label_config(label_config)
    tree_params = _normalize_tree_params(tree_params)
    action_config = _normalize_action_config(action_config)
    split_config = dict(split_config or {})
    action_label_version = resolve_action_label_version(label_config)
    output_dir = candidate_dir_for_cell(cell_id, output_tag=output_tag)
    output_dir.mkdir(parents=True, exist_ok=True)

    if force:
        for path in output_dir.glob("*"):
            if path.is_file():
                path.unlink()

    base_df = build_cell_dataset(cell, max_window_days=max(windows), force_rebuild=force, label_config=label_config)
    labeled_all = None if label_each_split else _build_action_labels(base_df, action_config, label_config=label_config)
    feature_cols = choose_tree_feature_cols(base_df, cell)
    rows_meta = []

    for window_days in windows:
        split_source = base_df if label_each_split else labeled_all
        split = train_val_test_split(split_source, window_days, **split_config)
        train_df, val_df, test_df = split["train"].copy(), split["val"].copy(), split["test"].copy()
        if label_each_split:
            train_df = _build_action_labels(
                _add_trade_objective_labels(train_df, label_config=label_config),
                action_config,
                label_config=label_config,
            )
            val_df = _build_action_labels(
                _add_trade_objective_labels(val_df, label_config=label_config),
                action_config,
                label_config=label_config,
            )
            test_df = _build_action_labels(
                _add_trade_objective_labels(test_df, label_config=label_config),
                action_config,
                label_config=label_config,
            )
        if len(train_df) < 800 or len(val_df) < 120 or len(test_df) < 120:
            rows_meta.append({
                "window_days": window_days,
                "status": "skipped_insufficient_rows",
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
            })
            continue

        for col in feature_cols:
            if col not in train_df.columns:
                train_df[col] = 0.0
                val_df[col] = 0.0
                test_df[col] = 0.0

        X_tr = train_df[feature_cols]
        X_va = val_df[feature_cols]
        y_state_tr = train_df["state_label"].map(STATE_TO_ID).astype(int)
        y_state_va = val_df["state_label"].map(STATE_TO_ID).astype(int)
        y_action_tr = train_df["action_label"].astype(int)
        y_action_va = val_df["action_label"].astype(int)

        state_model = _fit_state_model(X_tr, y_state_tr, train_df["state_head_weight"], X_va, y_state_va, tree_params)
        action_model = _fit_action_model(X_tr, y_action_tr, train_df["action_weight"], X_va, y_action_va, tree_params)
        abstain_model = _fit_abstain(X_tr, train_df["abstain_label"], train_df["abstain_train_weight"], X_va, val_df["abstain_label"], tree_params)

        joblib.dump(state_model, output_dir / f"state_lgb_{window_days}d.joblib")
        joblib.dump(action_model, output_dir / f"action_lgb_{window_days}d.joblib")
        joblib.dump(abstain_model, output_dir / f"abstain_lgb_{window_days}d.joblib")
        rows_meta.append({
            "window_days": window_days,
            "status": "trained",
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
            "up_rate": round(float((train_df["action_name"] == "UP").mean()), 6),
            "down_rate": round(float((train_df["action_name"] == "DOWN").mean()), 6),
            "abstain_rate": round(float((train_df["action_name"] == "ABSTAIN").mean()), 6),
        })

    (output_dir / "feature_cols.json").write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    config = {
        "generated_at": utc_now_iso(),
        "mode": "core10_tree_action_specialist",
        "action_label_version": action_label_version,
        "cell_id": cell_id,
        "profile": cell["profile"],
        "trader": cell["trader"],
        "symbol": cell["symbol"],
        "asset": f"{cell['symbol']}_USDT",
        "state_labels": STATE_LABELS,
        "action_labels": ACTION_LABELS,
        "window_days_list": windows,
        "feature_cols_count": len(feature_cols),
        "current_active_model": str(resolve_current_model_dir(cell)),
        "label_config": label_config,
        "tree_params": tree_params,
        "action_config": action_config,
        "action_scoring": {
            "trend_weight": action_config["trend_weight"],
            "chop_penalty": action_config["chop_penalty"],
            "abstain_gate_blend": action_config["abstain_gate_blend"],
            "action_margin_power": action_config["action_margin_power"],
        },
        "split_labeling_mode": "per_split_no_recent_leak_v1" if label_each_split else "full_dataset_before_split",
        "split_config": split_config,
        "output_tag": output_tag or "",
        "head_files": {
            "state": [f"state_lgb_{w}d.joblib" for w in windows if (output_dir / f"state_lgb_{w}d.joblib").exists()],
            "action": [f"action_lgb_{w}d.joblib" for w in windows if (output_dir / f"action_lgb_{w}d.joblib").exists()],
            "abstain": [f"abstain_lgb_{w}d.joblib" for w in windows if (output_dir / f"abstain_lgb_{w}d.joblib").exists()],
        },
        "training_rows": rows_meta,
    }
    write_json(output_dir / "config.json", config)
    return {
        "cell_id": cell_id,
        "output_dir": str(output_dir),
        "trained_windows": [row["window_days"] for row in rows_meta if row["status"] == "trained"],
        "rows": rows_meta,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--windows", default="180,365,540")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--label-config-json", default="")
    parser.add_argument("--tree-params-json", default="")
    parser.add_argument("--action-config-json", default="")
    parser.add_argument("--label-each-split", action="store_true")
    parser.add_argument("--split-config-json", default="")
    args = parser.parse_args()

    windows = [int(part) for part in str(args.windows).split(",") if str(part).strip()]
    label_config = json.loads(args.label_config_json) if str(args.label_config_json).strip() else None
    tree_params = json.loads(args.tree_params_json) if str(args.tree_params_json).strip() else None
    action_config = json.loads(args.action_config_json) if str(args.action_config_json).strip() else None
    split_config = json.loads(args.split_config_json) if str(args.split_config_json).strip() else None
    result = train_one(
        args.cell_id,
        windows,
        force=args.force,
        label_config=label_config,
        tree_params=tree_params,
        action_config=action_config,
        output_tag=(args.output_tag or None),
        label_each_split=bool(args.label_each_split),
        split_config=split_config,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
