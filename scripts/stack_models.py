#!/usr/bin/env python3
"""
Model stacking / blending: train LightGBM + XGBoost (optional) on the same data,
blend their predict_proba, and save for use in run_grid or production.

与 run_grid 的多窗口 LGB 集成是两条线：本脚本为独立工具，不写入 v5 模型目录；
接入 prediction_writer_v5 为可选后续步骤。

Usage:
  python scripts/stack_models.py --data path/to/features.parquet --target direction_label --out-dir out/
  python scripts/stack_models.py --data path/to/features.csv --target direction_label --out-dir out/ --weight-lgb 0.6 --weight-xgb 0.4

Output:
  out/stack_lgb.joblib, out/stack_xgb.joblib (if XGBoost available), out/stack_config.json
  Config lists "models" and "weights" for predict_proba blending.

Optional dependency: pip install xgboost  (if not installed, only LightGBM is used)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import lightgbm as lgb


def _load_data(path: Path, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not in {list(df.columns)}")
    y = df[target]
    X = df.drop(columns=[target])
    return X, y


def _get_features(X: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    exclude = set(exclude or [])
    obj_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    return [c for c in X.columns if c not in exclude and c not in obj_cols]


def train_stack(
    data_path: Path,
    target: str,
    out_dir: Path,
    weight_lgb: float = 0.5,
    weight_xgb: float = 0.5,
    weight_catboost: float = 0.0,
    lgb_params: Optional[Dict] = None,
    n_estimators: int = 500,
    random_state: int = 42,
) -> Dict:
    """
    Train LightGBM and optionally XGBoost、CatBoost on the same data; save models and config.
    Weights are normalized (weight_catboost=0 则不训练 CatBoost；需 pip install catboost)。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y = _load_data(data_path, target)
    feature_cols = _get_features(X, exclude=["timestamp", "asset_id"])
    X = X[feature_cols].copy()
    X = X.fillna(0)

    # Optional sample_weight column (same df)
    w = None
    if "sample_weight" in X.columns:
        w = X["sample_weight"].copy()
        X = X.drop(columns=["sample_weight"])
        feature_cols = [c for c in feature_cols if c != "sample_weight"]

    models = {}
    weights = {}

    # LightGBM (required)
    lgb_params = dict(lgb_params or {})
    lgb_params.setdefault("objective", "binary")
    lgb_params.setdefault("metric", "auc")
    lgb_params.setdefault("verbosity", -1)
    lgb_params.setdefault("random_state", random_state)
    lgb_params.setdefault("n_estimators", n_estimators)
    cat_cols = [c for c in ["asset_id"] if c in X.columns]
    model_lgb = lgb.LGBMClassifier(**lgb_params)
    model_lgb.fit(
        X, y,
        sample_weight=w,
        categorical_feature=cat_cols if cat_cols else "auto",
        callbacks=[lgb.log_evaluation(0)],
    )
    models["lgb"] = model_lgb
    weights["lgb"] = weight_lgb

    # XGBoost (optional)
    try:
        import xgboost as xgb
        model_xgb = xgb.XGBClassifier(
            n_estimators=n_estimators,
            objective="binary:logistic",
            eval_metric="auc",
            use_label_encoder=False,
            random_state=random_state,
            verbosity=0,
        )
        model_xgb.fit(X, y, sample_weight=w, verbose=False)
        models["xgb"] = model_xgb
        weights["xgb"] = weight_xgb
    except ImportError:
        pass

    # CatBoost (optional，weight_catboost > 0 且已安装时加入融合)
    if weight_catboost > 0:
        try:
            import catboost as cb
            model_cb = cb.CatBoostClassifier(
                iterations=n_estimators,
                loss_function="Logloss",
                eval_metric="AUC",
                random_state=random_state,
                verbose=0,
            )
            model_cb.fit(X, y, sample_weight=w)
            models["catboost"] = model_cb
            weights["catboost"] = weight_catboost
        except ImportError:
            pass

    # Normalize weights
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    # Save
    joblib.dump(model_lgb, out_dir / "stack_lgb.joblib")
    if "xgb" in models:
        joblib.dump(models["xgb"], out_dir / "stack_xgb.joblib")
    if "catboost" in models:
        joblib.dump(models["catboost"], out_dir / "stack_catboost.joblib")
    config = {
        "feature_cols": feature_cols,
        "target": target,
        "models": list(models.keys()),
        "weights": weights,
        "random_state": random_state,
    }
    with open(out_dir / "stack_config.json", "w") as f:
        json.dump(config, f, indent=2)

    return {"models": models, "weights": weights, "config": config, "feature_cols": feature_cols}


def load_stack(out_dir: Path) -> Tuple[List, List[float], List[str]]:
    """Load blended stack from out_dir; returns (list of models), (weights), feature_cols."""
    out_dir = Path(out_dir)
    with open(out_dir / "stack_config.json") as f:
        config = json.load(f)
    models = []
    weights = []
    for name in config["models"]:
        path = out_dir / f"stack_{name}.joblib"
        if path.exists():
            models.append(joblib.load(path))
            weights.append(config["weights"][name])
    total_w = sum(weights)
    weights = [w / total_w for w in weights]
    return models, weights, config["feature_cols"]


def predict_proba_blend(X: pd.DataFrame, models: List, weights: List[float], feature_cols: List[str]) -> np.ndarray:
    """Return blended P(y=1) for X (must contain feature_cols)."""
    X = X[feature_cols].fillna(0)
    p = np.zeros(len(X))
    for m, w in zip(models, weights):
        p += w * m.predict_proba(X)[:, 1]
    return p


def main():
    ap = argparse.ArgumentParser(description="Train stacked LightGBM + XGBoost and save")
    ap.add_argument("--data", type=Path, required=True, help="CSV or parquet with features + target")
    ap.add_argument("--target", default="direction_label", help="Target column name")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory for models and config")
    ap.add_argument("--weight-lgb", type=float, default=0.5, help="LightGBM 融合权重")
    ap.add_argument("--weight-xgb", type=float, default=0.5, help="XGBoost 融合权重")
    ap.add_argument("--weight-catboost", type=float, default=0.0, help="CatBoost 融合权重，>0 时加入（需 pip install catboost）")
    ap.add_argument("--n-estimators", type=int, default=500, help="Trees per model")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_stack(
        args.data,
        args.target,
        args.out_dir,
        weight_lgb=args.weight_lgb,
        weight_xgb=args.weight_xgb,
        weight_catboost=args.weight_catboost,
        n_estimators=args.n_estimators,
        random_state=args.seed,
    )
    print("Stack trained and saved to", args.out_dir)


if __name__ == "__main__":
    main()
