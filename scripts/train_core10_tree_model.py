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
    STATE_LABELS,
    STATE_TO_ID,
    build_cell_dataset,
    choose_tree_feature_cols,
    load_core_cell_map,
    normalize_label_config,
    resolve_current_model_dir,
    utc_now_iso,
    write_json,
    train_val_test_split,
)


TREE_ROOT = DATA / "models"


DEFAULT_TREE_PARAMS = {
    "learning_rate": 0.03,
    "num_leaves": 63,
    "max_depth": 7,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.75,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 4.0,
}


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


def candidate_dir_for_cell(cell_id: str, output_tag: str | None = None) -> Path:
    safe = cell_id.replace("/", "__").replace("-", "_").lower()
    if output_tag:
        safe_tag = output_tag.replace("/", "__").replace("-", "_").lower()
        return TREE_ROOT / f"core10_tree_{safe}__{safe_tag}"
    return TREE_ROOT / f"core10_tree_{safe}"


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
    model = lgb.train(
        params,
        train_set,
        num_boost_round=400,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    return model


def _fit_regressor(X_tr, y_tr, w_tr, X_va, y_va, tree_params: Dict[str, Any]):
    model = lgb.LGBMRegressor(
        objective="regression",
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


def train_one(
    cell_id: str,
    windows: List[int],
    force: bool = False,
    label_config: Dict[str, Any] | None = None,
    tree_params: Dict[str, Any] | None = None,
    output_tag: str | None = None,
) -> Dict[str, Any]:
    cells = load_core_cell_map()
    if cell_id not in cells:
        raise KeyError(f"unknown cell_id: {cell_id}")
    cell = cells[cell_id]
    label_config = normalize_label_config(label_config)
    tree_params = _normalize_tree_params(tree_params)
    output_dir = candidate_dir_for_cell(cell_id, output_tag=output_tag)
    output_dir.mkdir(parents=True, exist_ok=True)

    if force:
        for path in output_dir.glob("*"):
            if path.is_file():
                path.unlink()

    df = build_cell_dataset(cell, max_window_days=max(windows), force_rebuild=force, label_config=label_config)
    feature_cols = choose_tree_feature_cols(df, cell)
    rows_meta = []

    for window_days in windows:
        split = train_val_test_split(df, window_days)
        train_df, val_df, test_df = split["train"], split["val"], split["test"]
        if len(train_df) < 800 or len(val_df) < 120 or len(test_df) < 120:
            rows_meta.append(
                {
                    "window_days": window_days,
                    "status": "skipped_insufficient_rows",
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "test_rows": len(test_df),
                }
            )
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

        state_model = _fit_state_model(X_tr, y_state_tr, train_df["state_head_weight"], X_va, y_state_va, tree_params)
        up_model = _fit_regressor(X_tr, train_df["up_utility_label"], train_df["up_head_weight"], X_va, val_df["up_utility_label"], tree_params)
        down_model = _fit_regressor(X_tr, train_df["down_utility_label"], train_df["down_head_weight"], X_va, val_df["down_utility_label"], tree_params)
        abstain_model = _fit_abstain(X_tr, train_df["abstain_label"], train_df["abstain_head_weight"], X_va, val_df["abstain_label"], tree_params)

        joblib.dump(state_model, output_dir / f"state_lgb_{window_days}d.joblib")
        joblib.dump(up_model, output_dir / f"up_utility_lgb_{window_days}d.joblib")
        joblib.dump(down_model, output_dir / f"down_utility_lgb_{window_days}d.joblib")
        joblib.dump(abstain_model, output_dir / f"abstain_lgb_{window_days}d.joblib")
        rows_meta.append(
            {
                "window_days": window_days,
                "status": "trained",
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
                "state_best_iteration": int(getattr(state_model, "best_iteration", 0) or 0),
                "up_best_iteration": int(getattr(up_model, "best_iteration_", 0) or 0),
                "down_best_iteration": int(getattr(down_model, "best_iteration_", 0) or 0),
                "abstain_best_iteration": int(getattr(abstain_model, "best_iteration_", 0) or 0),
            }
        )

    (output_dir / "feature_cols.json").write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    config = {
        "generated_at": utc_now_iso(),
        "mode": "core10_tree_state_conditional",
        "cell_id": cell_id,
        "profile": cell["profile"],
        "trader": cell["trader"],
        "symbol": cell["symbol"],
        "asset": f"{cell['symbol']}_USDT",
        "state_labels": STATE_LABELS,
        "window_days_list": windows,
        "feature_cols_count": len(feature_cols),
        "current_active_model": str(resolve_current_model_dir(cell)),
        "label_config": label_config,
        "tree_params": tree_params,
        "output_tag": output_tag or "",
        "head_files": {
            "state": [f"state_lgb_{w}d.joblib" for w in windows if (output_dir / f"state_lgb_{w}d.joblib").exists()],
            "up_utility": [f"up_utility_lgb_{w}d.joblib" for w in windows if (output_dir / f"up_utility_lgb_{w}d.joblib").exists()],
            "down_utility": [f"down_utility_lgb_{w}d.joblib" for w in windows if (output_dir / f"down_utility_lgb_{w}d.joblib").exists()],
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
    parser.add_argument("--profile")
    parser.add_argument("--state-conditional", action="store_true")
    parser.add_argument("--abstain-head", action="store_true")
    parser.add_argument("--trade-utility-target", action="store_true")
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--label-config-json", default="")
    parser.add_argument("--tree-params-json", default="")
    args = parser.parse_args()

    windows = [int(part) for part in str(args.windows).split(",") if str(part).strip()]
    label_config = json.loads(args.label_config_json) if str(args.label_config_json).strip() else None
    tree_params = json.loads(args.tree_params_json) if str(args.tree_params_json).strip() else None
    result = train_one(args.cell_id, windows, force=args.force, label_config=label_config, tree_params=tree_params, output_tag=(args.output_tag or None))
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
