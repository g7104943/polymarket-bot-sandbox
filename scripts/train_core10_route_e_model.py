#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier, LGBMRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops.core10_retrain_common import (  # noqa: E402
    CACHE_DIR,
    DATA,
    build_cell_dataset,
    choose_tree_feature_cols,
    load_core_cell_map,
    normalize_label_config,
    predict_current_active_proba,
    resolve_current_model_dir,
    sanitize_cell_id,
    train_val_test_split,
    utc_now_iso,
    write_json,
)
from scripts.train_core10_sequence_model import (  # noqa: E402
    MultiHeadSequenceModel,
    _normalize_sequence_params,
    finetune_cell,
    get_device,
)

TREE_ROOT = DATA / "models"

REGIME_GROUPS: Dict[str, Tuple[str, ...]] = {
    "global": ("normal", "rally_up", "rally_down", "extreme_up", "extreme_down", "high_vol_chop"),
    "normal": ("normal",),
    "rally_up": ("rally_up",),
    "rally_down": ("rally_down",),
    "extreme_or_high_vol": ("extreme_up", "extreme_down", "high_vol_chop"),
}

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

DEFAULT_ROUTE_E_SCORING = {
    "normal_action_scale": 0.75,
    "extreme_action_scale": 0.55,
    "rally_bias_weight": 0.12,
}


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


def _normalize_route_e_scoring(cfg: Dict[str, Any] | None = None) -> Dict[str, float]:
    out = dict(DEFAULT_ROUTE_E_SCORING)
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
    safe = sanitize_cell_id(cell_id)
    if output_tag:
        safe_tag = sanitize_cell_id(output_tag)
        return TREE_ROOT / f"core10_route_e_{safe}__{safe_tag}"
    return TREE_ROOT / f"core10_route_e_{safe}"


def _sequence_output_tag(output_tag: str | None) -> str:
    base = output_tag or "route_e"
    return f"{base}__seq"


def _regime_bucket(state_label: str) -> str:
    state = str(state_label or "")
    if state == "rally_up":
        return "rally_up"
    if state == "rally_down":
        return "rally_down"
    if state in {"extreme_up", "extreme_down", "high_vol_chop"}:
        return "extreme_or_high_vol"
    return "normal"


def _augment_teacher_features(df: pd.DataFrame, model_dir: Path, asset: str) -> None:
    teacher_p_up = predict_current_active_proba(model_dir, df, asset)
    teacher_margin = np.abs(teacher_p_up - 0.5)
    teacher_abs_proxy = np.clip((0.06 - teacher_margin) / 0.06, 0.0, 1.0)
    df["teacher_p_up"] = teacher_p_up
    df["teacher_margin"] = teacher_margin
    df["teacher_abs_proxy"] = teacher_abs_proxy
    df["teacher_p_down"] = 1.0 - teacher_p_up


def _embedding_cache_path(cell_id: str, sequence_tag: str, label_config: Dict[str, Any], sequence_params: Dict[str, Any], max_window_days: int) -> Path:
    digest = hashlib.sha1(
        json.dumps(
            {
                "cell_id": cell_id,
                "sequence_tag": sequence_tag,
                "label_config": label_config,
                "sequence_params": sequence_params,
                "max_window_days": max_window_days,
            },
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()[:12]
    return CACHE_DIR / f"{sanitize_cell_id(cell_id)}__route_e_seq_emb__{digest}.parquet"


def _build_sequence_embeddings(
    df: pd.DataFrame,
    cell_id: str,
    label_config: Dict[str, Any],
    sequence_params: Dict[str, Any],
    sequence_tag: str,
    force: bool = False,
) -> Tuple[pd.DataFrame, Path]:
    cache_path = _embedding_cache_path(cell_id, sequence_tag, label_config, sequence_params, len(df))
    if cache_path.exists() and not force:
        emb_df = pd.read_parquet(cache_path)
        emb_df["timestamp"] = pd.to_datetime(emb_df["timestamp"], utc=True)
        merged = df.merge(emb_df, on="timestamp", how="left")
        emb_cols = [col for col in merged.columns if col.startswith("seq_emb_")]
        for col in emb_cols:
            merged[col] = merged[col].fillna(0.0)
        return merged, Path(json.loads((cache_path.with_suffix(".json")).read_text(encoding="utf-8")).get("sequence_model_dir"))

    seq_info = finetune_cell(
        cell_id,
        force=False,
        label_config=label_config,
        sequence_params=sequence_params,
        output_tag=sequence_tag,
    )
    seq_dir = Path(str(seq_info["output_dir"]))
    try:
        ckpt = torch.load(seq_dir / "model.pt", map_location="cpu")
    except Exception:
        seq_info = finetune_cell(
            cell_id,
            force=True,
            label_config=label_config,
            sequence_params=sequence_params,
            output_tag=sequence_tag,
        )
        seq_dir = Path(str(seq_info["output_dir"]))
        ckpt = torch.load(seq_dir / "model.pt", map_location="cpu")
    feature_cols = list(ckpt["feature_cols"])
    device = get_device()
    model = MultiHeadSequenceModel(input_dim=len(feature_cols), sequence_params=sequence_params).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    work = df.copy().reset_index(drop=True)
    for col in feature_cols:
        if col not in work.columns:
            work[col] = 0.0
    x_all = work[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
    seq_len = int(sequence_params["seq_len"])
    emb_dim = int(sequence_params["embedding_dim"])
    emb = np.zeros((len(work), emb_dim), dtype=np.float32)
    batch_size = 256
    valid_idx = list(range(max(0, seq_len - 1), len(work)))
    with torch.no_grad():
        for start in range(0, len(valid_idx), batch_size):
            idxs = valid_idx[start : start + batch_size]
            if not idxs:
                continue
            batch = np.stack([x_all[idx - seq_len + 1 : idx + 1] for idx in idxs], axis=0)
            x = torch.tensor(batch, dtype=torch.float32, device=device)
            _, emb_batch = model.backbone(x)
            emb[np.asarray(idxs, dtype=int)] = emb_batch.detach().cpu().numpy().astype(np.float32)

    emb_cols = {}
    for i in range(emb_dim):
        emb_cols[f"seq_emb_{i:02d}"] = emb[:, i]
    emb_df = pd.DataFrame({"timestamp": work["timestamp"], **emb_cols})
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    emb_df.to_parquet(cache_path, index=False)
    cache_path.with_suffix(".json").write_text(
        json.dumps({"sequence_model_dir": str(seq_dir)}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    merged = work.merge(emb_df, on="timestamp", how="left")
    return merged, seq_dir


def _ensure_binary_classes_present(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str], label_col: str, weight_col: str) -> Tuple[List[int], pd.DataFrame]:
    added: List[int] = []
    present = set(train_df[label_col].astype(int).unique().tolist())
    missing = [cls for cls in (0, 1) if cls not in present]
    if not missing:
        return added, train_df
    fallback_frames = [train_df, val_df, test_df]
    for cls in missing:
        source_row = None
        for frame in fallback_frames:
            hits = frame[frame[label_col] == cls]
            if not hits.empty:
                source_row = hits.iloc[[0]].copy()
                break
        if source_row is None:
            source_row = train_df.iloc[[0]].copy()
        for col in feature_cols:
            if col not in source_row.columns:
                source_row[col] = 0.0
        source_row[label_col] = int(cls)
        source_row[weight_col] = 1e-6
        train_df = pd.concat([train_df, source_row], ignore_index=True)
        added.append(int(cls))
    return added, train_df


def _fit_regressor(train_df: pd.DataFrame, feature_cols: List[str], target_col: str, weight_col: str, params: Dict[str, Any]) -> LGBMRegressor:
    model = LGBMRegressor(
        objective="regression",
        n_estimators=400,
        learning_rate=float(params["learning_rate"]),
        num_leaves=int(params["num_leaves"]),
        max_depth=int(params["max_depth"]),
        min_data_in_leaf=int(params["min_data_in_leaf"]),
        feature_fraction=float(params["feature_fraction"]),
        bagging_fraction=float(params["bagging_fraction"]),
        bagging_freq=int(params["bagging_freq"]),
        reg_lambda=float(params["lambda_l2"]),
        verbosity=-1,
    )
    model.fit(
        train_df[feature_cols],
        train_df[target_col].to_numpy(dtype=float),
        sample_weight=train_df[weight_col].to_numpy(dtype=float),
    )
    return model


def _fit_binary_classifier(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str], label_col: str, weight_col: str, params: Dict[str, Any]) -> Tuple[LGBMClassifier, List[int]]:
    added, train_df = _ensure_binary_classes_present(train_df, val_df, test_df, feature_cols, label_col, weight_col)
    model = LGBMClassifier(
        objective="binary",
        n_estimators=400,
        learning_rate=float(params["learning_rate"]),
        num_leaves=int(params["num_leaves"]),
        max_depth=int(params["max_depth"]),
        min_data_in_leaf=int(params["min_data_in_leaf"]),
        feature_fraction=float(params["feature_fraction"]),
        bagging_fraction=float(params["bagging_fraction"]),
        bagging_freq=int(params["bagging_freq"]),
        reg_lambda=float(params["lambda_l2"]),
        verbosity=-1,
    )
    model.fit(
        train_df[feature_cols],
        train_df[label_col].to_numpy(dtype=int),
        sample_weight=train_df[weight_col].to_numpy(dtype=float),
    )
    return model, added


def train_one(
    cell_id: str,
    windows: List[int],
    force: bool = False,
    label_config: Dict[str, Any] | None = None,
    tree_params: Dict[str, Any] | None = None,
    sequence_params: Dict[str, Any] | None = None,
    route_e_scoring: Dict[str, Any] | None = None,
    output_tag: str | None = None,
    sequence_tag: str | None = None,
) -> Dict[str, Any]:
    cells = load_core_cell_map()
    if cell_id not in cells:
        raise KeyError(f"unknown cell_id: {cell_id}")
    cell = cells[cell_id]
    label_config = normalize_label_config(label_config)
    tree_params = _normalize_tree_params(tree_params)
    sequence_params = _normalize_sequence_params(sequence_params)
    route_e_scoring = _normalize_route_e_scoring(route_e_scoring)
    output_dir = candidate_dir_for_cell(cell_id, output_tag=output_tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for path in output_dir.glob("*"):
            if path.is_file():
                path.unlink()

    df = build_cell_dataset(cell, max_window_days=max(windows), force_rebuild=False, label_config=label_config)
    asset = f"{cell['symbol']}_USDT"
    _augment_teacher_features(df, resolve_current_model_dir(cell), asset)
    seq_tag = sequence_tag or _sequence_output_tag(output_tag)
    df, sequence_model_dir = _build_sequence_embeddings(df, cell_id, label_config, sequence_params, seq_tag, force=force)
    df["route_e_regime"] = df["state_label"].map(_regime_bucket).fillna("normal")

    feature_cols = choose_tree_feature_cols(df, cell)
    for col in ["teacher_p_up", "teacher_p_down", "teacher_margin", "teacher_abs_proxy"]:
        if col not in feature_cols:
            feature_cols.append(col)
    for col in [c for c in df.columns if c.startswith("seq_emb_")]:
        if col not in feature_cols:
            feature_cols.append(col)

    head_files: Dict[str, Dict[str, List[str]]] = {
        "up_utility": {},
        "down_utility": {},
        "abstain": {},
    }
    training_rows: List[Dict[str, Any]] = []
    for window_days in windows:
        split = train_val_test_split(df, window_days)
        train_df = split["train"].copy()
        val_df = split["val"].copy()
        test_df = split["test"].copy()
        if len(train_df) < 1200 or len(val_df) < 150 or len(test_df) < 150:
            training_rows.append(
                {
                    "window_days": int(window_days),
                    "regime": "all",
                    "status": "skipped_insufficient_rows",
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "test_rows": len(test_df),
                }
            )
            continue
        for regime_name, state_labels in REGIME_GROUPS.items():
            train_reg = train_df[train_df["state_label"].isin(state_labels)].copy()
            val_reg = val_df[val_df["state_label"].isin(state_labels)].copy()
            test_reg = test_df[test_df["state_label"].isin(state_labels)].copy()
            if len(train_reg) < 300 or len(val_reg) < 50 or len(test_reg) < 50:
                training_rows.append(
                    {
                        "window_days": int(window_days),
                        "regime": regime_name,
                        "status": "skipped_sparse_regime",
                        "train_rows": len(train_reg),
                        "val_rows": len(val_reg),
                        "test_rows": len(test_reg),
                    }
                )
                continue
            for col in feature_cols:
                if col not in train_reg.columns:
                    train_reg[col] = 0.0
                    val_reg[col] = 0.0
                    test_reg[col] = 0.0

            up_model = _fit_regressor(train_reg, feature_cols, "up_utility_label", "up_head_weight", tree_params)
            down_model = _fit_regressor(train_reg, feature_cols, "down_utility_label", "down_head_weight", tree_params)
            abstain_model, added_classes = _fit_binary_classifier(
                train_reg,
                val_reg,
                test_reg,
                feature_cols,
                "abstain_label",
                "abstain_head_weight",
                tree_params,
            )

            up_name = f"up_utility__{regime_name}__{int(window_days)}d.joblib"
            down_name = f"down_utility__{regime_name}__{int(window_days)}d.joblib"
            abstain_name = f"abstain__{regime_name}__{int(window_days)}d.joblib"
            joblib.dump(up_model, output_dir / up_name)
            joblib.dump(down_model, output_dir / down_name)
            joblib.dump(abstain_model, output_dir / abstain_name)
            head_files["up_utility"].setdefault(regime_name, []).append(up_name)
            head_files["down_utility"].setdefault(regime_name, []).append(down_name)
            head_files["abstain"].setdefault(regime_name, []).append(abstain_name)
            training_rows.append(
                {
                    "window_days": int(window_days),
                    "regime": regime_name,
                    "status": "trained",
                    "train_rows": len(train_reg),
                    "val_rows": len(val_reg),
                    "test_rows": len(test_reg),
                    "abstain_positive_rows": int((train_reg["abstain_label"] == 1).sum()),
                    "abstain_anchor_added_classes": added_classes,
                }
            )

    config = {
        "generated_at": utc_now_iso(),
        "mode": "core10_route_e_regime_local_bandit_ranker",
        "route": "E",
        "cell_id": cell_id,
        "profile": cell["profile"],
        "trader": cell["trader"],
        "symbol": cell["symbol"],
        "asset": asset,
        "window_days_list": [int(x) for x in windows],
        "feature_cols_count": len(feature_cols),
        "feature_cols": feature_cols,
        "sequence_model_dir": str(sequence_model_dir),
        "sequence_output_tag": seq_tag,
        "current_active_model": str(resolve_current_model_dir(cell)),
        "label_config": label_config,
        "tree_params": tree_params,
        "sequence_params": sequence_params,
        "route_e_scoring": route_e_scoring,
        "head_files": head_files,
        "training_rows": training_rows,
        "regime_groups": {key: list(vals) for key, vals in REGIME_GROUPS.items()},
    }
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "feature_cols.json", feature_cols)
    return {
        "cell_id": cell_id,
        "output_dir": str(output_dir),
        "sequence_model_dir": str(sequence_model_dir),
        "trained_windows": sorted({row["window_days"] for row in training_rows if row["status"] == "trained"}),
        "rows": training_rows,
    }


def _mean_prediction(models: Iterable[Any], X: pd.DataFrame, binary: bool = False) -> np.ndarray:
    rows = []
    for model in models:
        if binary:
            rows.append(model.predict_proba(X)[:, 1])
        else:
            rows.append(model.predict(X))
    if not rows:
        return np.zeros(len(X), dtype=float)
    return np.mean(rows, axis=0)


def _attach_runtime_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    current_active_model = Path(str(cfg.get("current_active_model") or ""))
    asset = str(cfg.get("asset") or "")
    if current_active_model.exists() and asset:
        _augment_teacher_features(out, current_active_model, asset)
    sequence_model_dir = Path(str(cfg.get("sequence_model_dir") or ""))
    if sequence_model_dir.exists():
        sequence_params = _normalize_sequence_params(cfg.get("sequence_params"))
        ckpt = torch.load(sequence_model_dir / "model.pt", map_location="cpu")
        seq_feature_cols = list(ckpt["feature_cols"])
        device = get_device()
        model = MultiHeadSequenceModel(input_dim=len(seq_feature_cols), sequence_params=sequence_params).to(device)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        model.eval()
        for col in seq_feature_cols:
            if col not in out.columns:
                out[col] = 0.0
        x_all = out[seq_feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        seq_len = int(sequence_params["seq_len"])
        emb_dim = int(sequence_params["embedding_dim"])
        emb = np.zeros((len(out), emb_dim), dtype=np.float32)
        batch_size = 256
        valid_idx = list(range(max(0, seq_len - 1), len(out)))
        with torch.no_grad():
            for start in range(0, len(valid_idx), batch_size):
                idxs = valid_idx[start : start + batch_size]
                if not idxs:
                    continue
                batch = np.stack([x_all[idx - seq_len + 1 : idx + 1] for idx in idxs], axis=0)
                x = torch.tensor(batch, dtype=torch.float32, device=device)
                _, emb_batch = model.backbone(x)
                emb[np.asarray(idxs, dtype=int)] = emb_batch.detach().cpu().numpy().astype(np.float32)
        for i in range(emb_dim):
            out[f"seq_emb_{i:02d}"] = emb[:, i]
    return out


def predict_route_e_candidate_scores(model_dir: Path, df: pd.DataFrame) -> Dict[str, np.ndarray]:
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    feature_cols = json.loads((model_dir / "feature_cols.json").read_text(encoding="utf-8"))
    X = _attach_runtime_features(df, cfg)
    X["route_e_regime"] = X["state_label"].map(_regime_bucket).fillna("normal")
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0.0
    scoring = _normalize_route_e_scoring(cfg.get("route_e_scoring"))
    model_cache: Dict[Tuple[str, str], List[Any]] = {}
    head_files = cfg.get("head_files", {}) if isinstance(cfg.get("head_files"), dict) else {}

    def _load_models(head: str, regime: str) -> List[Any]:
        key = (head, regime)
        if key in model_cache:
            return model_cache[key]
        names = []
        if isinstance(head_files.get(head), dict):
            names = list(head_files.get(head, {}).get(regime, []) or [])
            if not names and regime != "global":
                names = list(head_files.get(head, {}).get("global", []) or [])
        models = [joblib.load(model_dir / name) for name in names if (model_dir / name).exists()]
        model_cache[key] = models
        return models

    up_score = np.zeros(len(X), dtype=float)
    down_score = np.zeros(len(X), dtype=float)
    abstain_score = np.ones(len(X), dtype=float)
    regimes = X["route_e_regime"].astype(str).to_numpy(dtype=object)
    for regime_name in ("normal", "rally_up", "rally_down", "extreme_or_high_vol"):
        mask = regimes == regime_name
        if not mask.any():
            continue
        X_reg = X.loc[mask, feature_cols]
        up_models = _load_models("up_utility", regime_name)
        down_models = _load_models("down_utility", regime_name)
        abstain_models = _load_models("abstain", regime_name)
        raw_up = np.maximum(_mean_prediction(up_models, X_reg, binary=False), 0.0)
        raw_down = np.maximum(_mean_prediction(down_models, X_reg, binary=False), 0.0)
        raw_abstain = np.clip(_mean_prediction(abstain_models, X_reg, binary=True), 0.0, 1.0)
        if regime_name == "normal":
            raw_up *= float(scoring["normal_action_scale"])
            raw_down *= float(scoring["normal_action_scale"])
        elif regime_name == "extreme_or_high_vol":
            raw_up *= float(scoring["extreme_action_scale"])
            raw_down *= float(scoring["extreme_action_scale"])
        elif regime_name == "rally_up":
            raw_up *= 1.0 + float(scoring["rally_bias_weight"])
            raw_down *= max(0.0, 1.0 - float(scoring["rally_bias_weight"]))
        elif regime_name == "rally_down":
            raw_down *= 1.0 + float(scoring["rally_bias_weight"])
            raw_up *= max(0.0, 1.0 - float(scoring["rally_bias_weight"]))
        up_score[mask] = raw_up
        down_score[mask] = raw_down
        abstain_score[mask] = raw_abstain

    return {
        "up_score": np.asarray(up_score, dtype=float),
        "down_score": np.asarray(down_score, dtype=float),
        "abstain_score": np.asarray(abstain_score, dtype=float),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--windows", default="180,365,540")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-tag", default="")
    parser.add_argument("--label-config-json", default="")
    parser.add_argument("--tree-params-json", default="")
    parser.add_argument("--sequence-params-json", default="")
    parser.add_argument("--route-e-scoring-json", default="")
    parser.add_argument("--sequence-tag", default="")
    args = parser.parse_args()

    windows = [int(x.strip()) for x in str(args.windows).split(",") if x.strip()]
    label_config = json.loads(args.label_config_json) if str(args.label_config_json).strip() else None
    tree_params = json.loads(args.tree_params_json) if str(args.tree_params_json).strip() else None
    sequence_params = json.loads(args.sequence_params_json) if str(args.sequence_params_json).strip() else None
    route_e_scoring = json.loads(args.route_e_scoring_json) if str(args.route_e_scoring_json).strip() else None
    result = train_one(
        args.cell_id,
        windows,
        force=args.force,
        label_config=label_config,
        tree_params=tree_params,
        sequence_params=sequence_params,
        route_e_scoring=route_e_scoring,
        output_tag=args.output_tag or None,
        sequence_tag=args.sequence_tag or None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
