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


def candidate_dir_for_cell(cell_id: str, output_tag: str | None = None) -> Path:
    safe = cell_id.replace("/", "__").replace("-", "_").lower()
    if output_tag:
        safe_tag = output_tag.replace("/", "__").replace("-", "_").lower()
        return TREE_ROOT / f"core10_rally_up_precision_{safe}__{safe_tag}"
    return TREE_ROOT / f"core10_rally_up_precision_{safe}"


def _fit_binary_model(X_tr, y_tr, w_tr, X_va, y_va, tree_params: Dict[str, Any]):
    model = lgb.LGBMClassifier(
        objective="binary",
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


def train_one(
    cell_id: str,
    windows: List[int],
    force: bool = False,
    label_config: Dict[str, Any] | None = None,
    tree_params: Dict[str, Any] | None = None,
    action_config: Dict[str, Any] | None = None,
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
    route_states = list(route_states or DEFAULT_ROUTE_STATES)
    action_label_version = resolve_action_label_version(label_config) + "__rally_up_precision_v1"
    output_dir = candidate_dir_for_cell(cell_id, output_tag=output_tag)
    output_dir.mkdir(parents=True, exist_ok=True)

    if force:
        for path in output_dir.glob("*"):
            if path.is_file():
                path.unlink()

    df = build_cell_dataset(cell, max_window_days=max(windows), force_rebuild=force, label_config=label_config)
    df = _build_action_labels(df, action_config, label_config=label_config)
    df = df.loc[df["state_label"].isin(route_states)].copy().reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f"{cell_id} no rows for route_states={route_states}")

    feature_cols = choose_tree_feature_cols(df, cell)
    rows_meta = []

    for window_days in windows:
        split = train_val_test_split(df, window_days)
        train_df, val_df, test_df = split["train"].copy(), split["val"].copy(), split["test"].copy()
        if len(train_df) < 500 or len(val_df) < 80 or len(test_df) < 80:
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
        y_tr = (train_df["action_name"] == "UP").astype(int)
        y_va = (val_df["action_name"] == "UP").astype(int)
        sample_weight = train_df["action_weight"].to_numpy(dtype=float).copy()
        sample_weight *= np.where(y_tr.to_numpy(dtype=int) == 1, 1.5, 1.0)

        model = _fit_binary_model(X_tr, y_tr, sample_weight, X_va, y_va, tree_params)
        joblib.dump(model, output_dir / f"rally_up_precision_lgb_{window_days}d.joblib")
        rows_meta.append(
            {
                "window_days": window_days,
                "status": "trained",
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "test_rows": len(test_df),
                "up_rate": round(float((train_df["action_name"] == "UP").mean()), 6),
                "abstain_like_rate": round(float((train_df["action_name"] != "UP").mean()), 6),
            }
        )

    (output_dir / "feature_cols.json").write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    config = {
        "generated_at": utc_now_iso(),
        "mode": "core10_rally_up_precision_specialist",
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
        "route_states": route_states,
        "output_tag": output_tag or "",
        "head_files": {
            "rally_up_precision": [f"rally_up_precision_lgb_{w}d.joblib" for w in windows if (output_dir / f"rally_up_precision_lgb_{w}d.joblib").exists()],
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
    parser.add_argument("--windows", default="180,365")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--label-config-json", default="")
    parser.add_argument("--tree-params-json", default="")
    parser.add_argument("--action-config-json", default="")
    parser.add_argument("--route-states-json", default="")
    args = parser.parse_args()

    windows = [int(part) for part in str(args.windows).split(",") if str(part).strip()]
    label_config = json.loads(args.label_config_json) if str(args.label_config_json).strip() else None
    tree_params = json.loads(args.tree_params_json) if str(args.tree_params_json).strip() else None
    action_config = json.loads(args.action_config_json) if str(args.action_config_json).strip() else None
    route_states = json.loads(args.route_states_json) if str(args.route_states_json).strip() else None
    result = train_one(
        args.cell_id,
        windows,
        force=args.force,
        label_config=label_config,
        tree_params=tree_params,
        action_config=action_config,
        route_states=route_states,
        output_tag=(args.output_tag or None),
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
