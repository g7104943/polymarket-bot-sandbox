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

TREE_ROOT = DATA / "models"
DEFAULT_ROUTE_STATES = ["rally_up", "extreme_up"]

DEFAULT_TREE_PARAMS = {
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "min_data_in_leaf": 100,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 4.0,
}

DEFAULT_EVENT_CONFIG = {
    "horizon_bars": 16.0,
    "tp_mult": 1.10,
    "sl_mult": 0.90,
    "min_tp": 0.0040,
    "min_sl": 0.0035,
    "win_weight_boost": 2.5,
    "loss_weight_boost": 3.0,
    "recent_negative_days": 21.0,
    "recent_negative_boost": 3.5,
    "teacher_feature_weight": 1.0,
}


class _ConstantBinaryModel:
    def __init__(self, positive_prob: float):
        self.positive_prob = float(np.clip(positive_prob, 0.0, 1.0))

    def predict_proba(self, X):
        n = len(X)
        pos = np.full(n, self.positive_prob, dtype=float)
        return np.column_stack([1.0 - pos, pos])


def _normalize_tree_params(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    out = dict(DEFAULT_TREE_PARAMS)
    if isinstance(cfg, dict):
        for key in out:
            raw = cfg.get(key)
            if raw is None:
                continue
            try:
                out[key] = float(raw) if key not in {"num_leaves", "max_depth", "min_data_in_leaf", "bagging_freq"} else int(raw)
            except Exception:
                continue
    return out


def _normalize_event_config(cfg: Dict[str, Any] | None = None) -> Dict[str, float]:
    out = dict(DEFAULT_EVENT_CONFIG)
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
        return TREE_ROOT / f"core10_rally_up_event_ranker_{safe}__{safe_tag}"
    return TREE_ROOT / f"core10_rally_up_event_ranker_{safe}"


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
    return ["teacher_p_up", "teacher_up_edge", "teacher_margin", "teacher_abs_proxy"]


def _build_event_labels(df, event_config: Dict[str, float]) -> None:
    horizon = int(event_config["horizon_bars"])
    tp_mult = float(event_config["tp_mult"])
    sl_mult = float(event_config["sl_mult"])
    min_tp = float(event_config["min_tp"])
    min_sl = float(event_config["min_sl"])
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    rv = df["rv_4h"].to_numpy(dtype=float) if "rv_4h" in df.columns else np.full(len(df), 0.004, dtype=float)
    latest_ts = df["timestamp"].max()
    age_days = (latest_ts - df["timestamp"]).dt.total_seconds().div(86400.0).to_numpy(dtype=float)

    rank_target = np.zeros(len(df), dtype=float)
    abstain_label = np.ones(len(df), dtype=int)
    weight = np.ones(len(df), dtype=float)
    tp_arr = np.maximum(rv * tp_mult, min_tp)
    sl_arr = np.maximum(rv * sl_mult, min_sl)

    for i in range(len(df)):
        if i + 1 >= len(df):
            continue
        end = min(len(df), i + 1 + horizon)
        up_bar = close[i] * (1.0 + tp_arr[i])
        down_bar = close[i] * (1.0 - sl_arr[i])
        label = 0.0
        abstain = 1
        hit_idx = None
        hit_side = None
        for j in range(i + 1, end):
            if high[j] >= up_bar:
                hit_idx = j
                hit_side = "UP"
                break
            if low[j] <= down_bar:
                hit_idx = j
                hit_side = "DOWN"
                break
        if hit_side == "UP":
            abstain = 0
            label = 1.0 / (1.0 + (hit_idx - i))
            weight[i] *= float(event_config["win_weight_boost"])
        elif hit_side == "DOWN":
            abstain = 1
            label = -1.0 / (1.0 + (hit_idx - i))
            weight[i] *= float(event_config["loss_weight_boost"])
            if age_days[i] <= float(event_config["recent_negative_days"]):
                weight[i] *= float(event_config["recent_negative_boost"])
        rank_target[i] = label
        abstain_label[i] = abstain

    df["event_rank_target"] = rank_target
    df["event_abstain_label"] = abstain_label
    df["event_weight"] = weight


def _fit_regressor(X_tr, y_tr, w_tr, X_va, y_va, tree_params: Dict[str, Any]):
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=500,
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


def _fit_binary(X_tr, y_tr, w_tr, X_va, y_va, tree_params: Dict[str, Any]):
    y_tr_arr = np.asarray(y_tr, dtype=int)
    unique = np.unique(y_tr_arr)
    if len(unique) < 2:
        return _ConstantBinaryModel(float(unique[0]) if len(unique) == 1 else 1.0)
    train_set = lgb.Dataset(X_tr, label=y_tr_arr, weight=w_tr)
    valid_set = lgb.Dataset(X_va, label=np.asarray(y_va, dtype=int), reference=train_set)
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


def train_one(
    cell_id: str,
    windows: List[int],
    force: bool = False,
    label_config: Dict[str, Any] | None = None,
    tree_params: Dict[str, Any] | None = None,
    event_config: Dict[str, Any] | None = None,
    route_states: List[str] | None = None,
    output_tag: str | None = None,
) -> Dict[str, Any]:
    cells = load_core_cell_map()
    if cell_id not in cells:
        raise KeyError(f"unknown cell_id: {cell_id}")
    cell = cells[cell_id]
    label_config = normalize_label_config(label_config)
    tree_params = _normalize_tree_params(tree_params)
    event_config = _normalize_event_config(event_config)
    route_states = list(route_states or DEFAULT_ROUTE_STATES)
    output_dir = candidate_dir_for_cell(cell_id, output_tag=output_tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for path in output_dir.glob("*"):
            if path.is_file():
                path.unlink()

    df = build_cell_dataset(cell, max_window_days=max(windows), force_rebuild=force, label_config=label_config)
    _augment_teacher_features(df, resolve_current_model_dir(cell))
    df = df.loc[df["state_label"].isin(route_states)].copy().reset_index(drop=True)
    _build_event_labels(df, event_config)
    feature_cols = choose_tree_feature_cols(df, cell)
    for col in ["teacher_p_up", "teacher_up_edge", "teacher_margin", "teacher_abs_proxy"]:
        if col not in feature_cols:
            feature_cols.append(col)

    rows_meta = []
    for window_days in windows:
        split = train_val_test_split(df, window_days)
        train_df, val_df, test_df = split["train"].copy(), split["val"].copy(), split["test"].copy()
        if len(train_df) < 800 or len(val_df) < 120 or len(test_df) < 120:
            rows_meta.append({"window_days": window_days, "status": "skipped_insufficient_rows", "train_rows": len(train_df), "val_rows": len(val_df), "test_rows": len(test_df)})
            continue
        for col in feature_cols:
            if col not in train_df.columns:
                train_df[col] = 0.0
                val_df[col] = 0.0
                test_df[col] = 0.0
        X_tr = train_df[feature_cols]
        X_va = val_df[feature_cols]
        rank_model = _fit_regressor(
            X_tr,
            train_df["event_rank_target"].to_numpy(dtype=float),
            train_df["event_weight"].to_numpy(dtype=float),
            X_va,
            val_df["event_rank_target"].to_numpy(dtype=float),
            tree_params,
        )
        abstain_model = _fit_binary(
            X_tr,
            train_df["event_abstain_label"].to_numpy(dtype=int),
            train_df["event_weight"].to_numpy(dtype=float),
            X_va,
            val_df["event_abstain_label"].to_numpy(dtype=int),
            tree_params,
        )
        joblib.dump(rank_model, output_dir / f"event_rank_lgb_{window_days}d.joblib")
        joblib.dump(abstain_model, output_dir / f"event_abstain_lgb_{window_days}d.joblib")
        rows_meta.append(
            {
                "window_days": window_days,
                "status": "trained",
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
                "positive_rank_rows": int((train_df["event_rank_target"] > 0).sum()),
                "negative_rank_rows": int((train_df["event_rank_target"] < 0).sum()),
                "abstain_rows": int((train_df["event_abstain_label"] == 1).sum()),
            }
        )

    (output_dir / "feature_cols.json").write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    config = {
        "generated_at": utc_now_iso(),
        "mode": "core10_rally_up_event_ranker",
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
        "event_config": event_config,
        "route_states": route_states,
        "output_tag": output_tag or "",
        "head_files": {
            "event_rank": [f"event_rank_lgb_{w}d.joblib" for w in windows if (output_dir / f"event_rank_lgb_{w}d.joblib").exists()],
            "event_abstain": [f"event_abstain_lgb_{w}d.joblib" for w in windows if (output_dir / f"event_abstain_lgb_{w}d.joblib").exists()],
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
    parser.add_argument("--event-config-json", default="")
    parser.add_argument("--route-states-json", default="")
    args = parser.parse_args()

    windows = [int(part) for part in str(args.windows).split(",") if str(part).strip()]
    label_config = json.loads(args.label_config_json) if str(args.label_config_json).strip() else None
    tree_params = json.loads(args.tree_params_json) if str(args.tree_params_json).strip() else None
    event_config = json.loads(args.event_config_json) if str(args.event_config_json).strip() else None
    route_states = json.loads(args.route_states_json) if str(args.route_states_json).strip() else None
    result = train_one(
        args.cell_id,
        windows,
        force=args.force,
        label_config=label_config,
        tree_params=tree_params,
        event_config=event_config,
        route_states=route_states,
        output_tag=(args.output_tag or None),
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
