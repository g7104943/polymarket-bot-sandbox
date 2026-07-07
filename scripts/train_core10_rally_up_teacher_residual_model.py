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
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops.core10_retrain_common import (
    DATA,
    build_cell_dataset,
    choose_tree_feature_cols,
    load_core_cell_map,
    normalize_label_config,
    predict_current_active_proba,
    resolve_current_model_dir,
    train_val_test_split,
    utc_now_iso,
    write_json,
)
from scripts.train_core10_action_tree_model import (
    _build_action_labels,
    _normalize_action_config,
    _normalize_tree_params,
    resolve_action_label_version,
)

TREE_ROOT = DATA / "models"
DEFAULT_ROUTE_STATES = ["rally_up", "extreme_up"]

DEFAULT_RESIDUAL_CONFIG = {
    "good_roi_min": 0.05,
    "bad_roi_max": 0.00,
    "margin_min": 0.08,
    "teacher_missed_positive_edge_max": 0.10,
    "teacher_false_positive_edge_min": 0.18,
    "teacher_feature_weight": 1.0,
    "positive_weight_boost": 3.0,
    "hard_negative_weight_boost": 4.0,
    "teacher_missed_positive_boost": 5.0,
    "teacher_false_positive_boost": 4.5,
    "neutral_weight_scale": 0.75,
    "recent_hard_negative_days": 21.0,
    "recent_hard_negative_boost": 5.5,
}


class _ConstantBinaryModel:
    def __init__(self, positive_prob: float):
        self.positive_prob = float(np.clip(positive_prob, 0.0, 1.0))

    def predict_proba(self, X):
        n = len(X)
        pos = np.full(n, self.positive_prob, dtype=float)
        return np.column_stack([1.0 - pos, pos])


def _normalize_residual_config(cfg: Dict[str, Any] | None = None) -> Dict[str, float]:
    out = dict(DEFAULT_RESIDUAL_CONFIG)
    if isinstance(cfg, dict):
        for key in out:
            raw = cfg.get(key)
            if raw is None:
                continue
            try:
                out[key] = float(raw)
            except Exception:
                continue
    return out


def candidate_dir_for_cell(cell_id: str, output_tag: str | None = None) -> Path:
    safe = cell_id.replace("/", "__").replace("-", "_").lower()
    if output_tag:
        safe_tag = output_tag.replace("/", "__").replace("-", "_").lower()
        return TREE_ROOT / f"core10_rally_up_teacher_residual_{safe}__{safe_tag}"
    return TREE_ROOT / f"core10_rally_up_teacher_residual_{safe}"


def _fit_binary_model(X_tr, y_tr, w_tr, X_va, y_va, tree_params: Dict[str, Any]):
    y_tr_arr = np.asarray(y_tr, dtype=int)
    y_va_arr = np.asarray(y_va, dtype=int)
    unique = np.unique(y_tr_arr)
    if len(unique) < 2:
        return _ConstantBinaryModel(float(unique[0]) if len(unique) == 1 else 0.0)

    train_set = lgb.Dataset(X_tr, label=y_tr_arr, weight=w_tr)
    valid_set = lgb.Dataset(X_va, label=y_va_arr, reference=train_set)
    params = {
        "objective": "binary",
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
        "metric": "binary_logloss",
    }
    return lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )


def _augment_teacher_features(df, model_dir: Path) -> List[str]:
    asset = "ETH_USDT"
    teacher_p_up = predict_current_active_proba(model_dir, df, asset)
    teacher_edge = np.maximum(teacher_p_up - 0.5, 0.0)
    teacher_margin = np.abs(teacher_p_up - 0.5)
    teacher_abs_proxy = np.clip((0.06 - teacher_margin) / 0.06, 0.0, 1.0)
    df["teacher_p_up"] = teacher_p_up
    df["teacher_up_edge"] = teacher_edge
    df["teacher_margin"] = teacher_margin
    df["teacher_abs_proxy"] = teacher_abs_proxy
    df["teacher_edge_rank_20"] = pd.qcut(df["teacher_up_edge"].rank(method="first"), q=20, labels=False, duplicates="drop").astype(float) / 19.0 if len(df) >= 20 else 0.0
    return ["teacher_p_up", "teacher_up_edge", "teacher_margin", "teacher_abs_proxy", "teacher_edge_rank_20"]


def _build_residual_labels(df, residual_config: Dict[str, float]) -> None:
    up_roi = df["market_trade_roi_up"].to_numpy(dtype=float) if "market_trade_roi_up" in df.columns else np.full(len(df), np.nan)
    margin = df["market_trade_roi_margin"].to_numpy(dtype=float) if "market_trade_roi_margin" in df.columns else np.zeros(len(df), dtype=float)
    observed_mask = np.isfinite(up_roi)
    good = observed_mask & (up_roi >= float(residual_config["good_roi_min"])) & (margin >= float(residual_config["margin_min"]))
    bad = observed_mask & (up_roi <= float(residual_config["bad_roi_max"]))
    action_name = df["action_name"].to_numpy(dtype=object)
    teacher_edge = df["teacher_up_edge"].to_numpy(dtype=float)
    latest_ts = df["timestamp"].max()
    age_days = (latest_ts - df["timestamp"]).dt.total_seconds().div(86400.0).to_numpy(dtype=float)
    recent_bad_up = (
        (age_days <= float(residual_config["recent_hard_negative_days"]))
        & (action_name == "UP")
        & (
            bad
            | (df.get("market_final_recent_negative", 0).to_numpy(dtype=float) > 0 if "market_final_recent_negative" in df.columns else False)
            | (df.get("recent_hard_negative", 0).to_numpy(dtype=float) > 0 if "recent_hard_negative" in df.columns else False)
        )
    )
    teacher_missed_positive = good & (teacher_edge <= float(residual_config["teacher_missed_positive_edge_max"]))
    teacher_false_positive = (action_name != "UP") & (teacher_edge >= float(residual_config["teacher_false_positive_edge_min"]))

    residual_positive = good
    abstain_label = ~(good)

    weight = np.full(len(df), float(residual_config["neutral_weight_scale"]), dtype=float)
    weight[residual_positive] = float(residual_config["positive_weight_boost"])
    weight[bad] *= float(residual_config["hard_negative_weight_boost"])
    weight[teacher_missed_positive] *= float(residual_config["teacher_missed_positive_boost"])
    weight[teacher_false_positive] *= float(residual_config["teacher_false_positive_boost"])
    weight[recent_bad_up] *= float(residual_config["recent_hard_negative_boost"])

    df["residual_positive_label"] = residual_positive.astype(int)
    df["residual_abstain_label"] = abstain_label.astype(int)
    df["residual_weight"] = weight.astype(float)
    df["teacher_missed_positive"] = teacher_missed_positive.astype(int)
    df["teacher_false_positive"] = teacher_false_positive.astype(int)
    df["recent_bad_up_label"] = recent_bad_up.astype(int)


def train_one(
    cell_id: str,
    windows: List[int],
    force: bool = False,
    label_config: Dict[str, Any] | None = None,
    tree_params: Dict[str, Any] | None = None,
    action_config: Dict[str, Any] | None = None,
    residual_config: Dict[str, Any] | None = None,
    route_states: List[str] | None = None,
    output_tag: str | None = None,
) -> Dict[str, Any]:
    cells = load_core_cell_map()
    if cell_id not in cells:
        raise KeyError(f"unknown cell_id: {cell_id}")
    cell = cells[cell_id]
    label_config = normalize_label_config(label_config)
    tree_params = _normalize_tree_params(tree_params)
    action_config = _normalize_action_config(action_config)
    residual_config = _normalize_residual_config(residual_config)
    route_states = list(route_states or DEFAULT_ROUTE_STATES)
    action_label_version = resolve_action_label_version(label_config) + "__rally_up_teacher_residual_v1"
    output_dir = candidate_dir_for_cell(cell_id, output_tag=output_tag)
    output_dir.mkdir(parents=True, exist_ok=True)

    if force:
        for path in output_dir.glob("*"):
            if path.is_file():
                path.unlink()

    df = build_cell_dataset(cell, max_window_days=max(windows), force_rebuild=force, label_config=label_config)
    df = _build_action_labels(df, action_config, label_config=label_config)
    teacher_feature_cols = _augment_teacher_features(df, resolve_current_model_dir(cell))
    df = df.loc[df["state_label"].isin(route_states)].copy().reset_index(drop=True)
    _build_residual_labels(df, residual_config)

    feature_cols = choose_tree_feature_cols(df, cell)
    for col in teacher_feature_cols:
        if col not in feature_cols:
            feature_cols.append(col)

    rows_meta = []
    for window_days in windows:
        split = train_val_test_split(df, window_days)
        train_df, val_df, test_df = split["train"].copy(), split["val"].copy(), split["test"].copy()
        if len(train_df) < 500 or len(val_df) < 80 or len(test_df) < 80:
            rows_meta.append({"window_days": window_days, "status": "skipped_insufficient_rows", "train_rows": len(train_df), "val_rows": len(val_df), "test_rows": len(test_df)})
            continue
        for col in feature_cols:
            if col not in train_df.columns:
                train_df[col] = 0.0
                val_df[col] = 0.0
                test_df[col] = 0.0
        X_tr = train_df[feature_cols]
        X_va = val_df[feature_cols]
        residual_model = _fit_binary_model(
            X_tr,
            train_df["residual_positive_label"],
            train_df["residual_weight"],
            X_va,
            val_df["residual_positive_label"],
            tree_params,
        )
        abstain_model = _fit_binary_model(
            X_tr,
            train_df["residual_abstain_label"],
            train_df["residual_weight"],
            X_va,
            val_df["residual_abstain_label"],
            tree_params,
        )
        joblib.dump(residual_model, output_dir / f"residual_up_lgb_{window_days}d.joblib")
        joblib.dump(abstain_model, output_dir / f"residual_abstain_lgb_{window_days}d.joblib")
        rows_meta.append(
            {
                "window_days": window_days,
                "status": "trained",
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
                "positive_rate": round(float(train_df["residual_positive_label"].mean()), 6),
                "teacher_missed_positive_rate": round(float(train_df["teacher_missed_positive"].mean()), 6),
                "teacher_false_positive_rate": round(float(train_df["teacher_false_positive"].mean()), 6),
                "recent_bad_up_rate": round(float(train_df["recent_bad_up_label"].mean()), 6),
            }
        )

    (output_dir / "feature_cols.json").write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    config = {
        "generated_at": utc_now_iso(),
        "mode": "core10_rally_up_teacher_residual",
        "action_label_version": action_label_version,
        "cell_id": cell_id,
        "profile": cell["profile"],
        "trader": cell["trader"],
        "symbol": cell["symbol"],
        "asset": f"{cell['symbol']}_USDT",
        "window_days_list": windows,
        "feature_cols_count": len(feature_cols),
        "current_active_model": str(resolve_current_model_dir(cell)),
        "label_config": label_config,
        "tree_params": tree_params,
        "action_config": action_config,
        "residual_config": residual_config,
        "route_states": route_states,
        "output_tag": output_tag or "",
        "head_files": {
            "residual_up": [f"residual_up_lgb_{w}d.joblib" for w in windows if (output_dir / f"residual_up_lgb_{w}d.joblib").exists()],
            "residual_abstain": [f"residual_abstain_lgb_{w}d.joblib" for w in windows if (output_dir / f"residual_abstain_lgb_{w}d.joblib").exists()],
        },
        "training_rows": rows_meta,
    }
    write_json(output_dir / "config.json", config)
    return {"cell_id": cell_id, "output_dir": str(output_dir), "trained_windows": [r["window_days"] for r in rows_meta if r["status"] == "trained"], "rows": rows_meta}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--windows", default="180,365")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--label-config-json", default="")
    parser.add_argument("--tree-params-json", default="")
    parser.add_argument("--action-config-json", default="")
    parser.add_argument("--residual-config-json", default="")
    parser.add_argument("--route-states-json", default="")
    args = parser.parse_args()

    windows = [int(part) for part in str(args.windows).split(",") if str(part).strip()]
    label_config = json.loads(args.label_config_json) if str(args.label_config_json).strip() else None
    tree_params = json.loads(args.tree_params_json) if str(args.tree_params_json).strip() else None
    action_config = json.loads(args.action_config_json) if str(args.action_config_json).strip() else None
    residual_config = json.loads(args.residual_config_json) if str(args.residual_config_json).strip() else None
    route_states = json.loads(args.route_states_json) if str(args.route_states_json).strip() else None
    result = train_one(
        args.cell_id,
        windows,
        force=args.force,
        label_config=label_config,
        tree_params=tree_params,
        action_config=action_config,
        residual_config=residual_config,
        route_states=route_states,
        output_tag=(args.output_tag or None),
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
