#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops.core10_retrain_common import (  # noqa: E402
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

TREE_ROOT = DATA / 'models'

DEFAULT_TREE_PARAMS = {
    'learning_rate': 0.03,
    'num_leaves': 31,
    'max_depth': 6,
    'min_data_in_leaf': 100,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l2': 4.0,
}

DEFAULT_BANDIT_CONFIG = {
    'abstain_utility_max': 0.01,
    'action_gap_min': 0.01,
    'reward_weight_scale': 4.0,
    'recent_days': 21.0,
    'recent_action_boost': 2.0,
    'countertrend_boost': 1.5,
}


def _normalize_tree_params(cfg: Dict[str, Any] | None = None) -> Dict[str, Any]:
    out = dict(DEFAULT_TREE_PARAMS)
    if isinstance(cfg, dict):
        for key in out:
            raw = cfg.get(key)
            if raw is None:
                continue
            try:
                out[key] = float(raw) if key not in {'num_leaves', 'max_depth', 'min_data_in_leaf', 'bagging_freq'} else int(raw)
            except Exception:
                continue
    return out


def _normalize_bandit_config(cfg: Dict[str, Any] | None = None) -> Dict[str, float]:
    out = dict(DEFAULT_BANDIT_CONFIG)
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
    safe = cell_id.replace('/', '__').replace('-', '_').lower()
    if output_tag:
        safe_tag = output_tag.replace('/', '__').replace('-', '_').lower()
        return TREE_ROOT / f'core10_bandit_policy_{safe}__{safe_tag}'
    return TREE_ROOT / f'core10_bandit_policy_{safe}'


def _augment_teacher_features(df, model_dir: Path, asset: str) -> None:
    teacher_p_up = predict_current_active_proba(model_dir, df, asset)
    teacher_margin = np.abs(teacher_p_up - 0.5)
    teacher_abs_proxy = np.clip((0.06 - teacher_margin) / 0.06, 0.0, 1.0)
    df['teacher_p_up'] = teacher_p_up
    df['teacher_margin'] = teacher_margin
    df['teacher_abs_proxy'] = teacher_abs_proxy


def _build_bandit_labels(df, bandit_config: Dict[str, float]) -> None:
    up = df['up_utility_label'].to_numpy(dtype=float)
    down = df['down_utility_label'].to_numpy(dtype=float)
    abstain_floor = float(bandit_config['abstain_utility_max'])
    gap_min = float(bandit_config['action_gap_min'])
    labels = np.full(len(df), 2, dtype=int)  # 0=UP,1=DOWN,2=ABSTAIN
    up_mask = (up > abstain_floor) & (up >= down + gap_min)
    down_mask = (down > abstain_floor) & (down >= up + gap_min)
    labels[up_mask] = 0
    labels[down_mask] = 1
    reward_mag = np.maximum.reduce([np.abs(up), np.abs(down), np.zeros(len(df), dtype=float)])
    weights = 1.0 + reward_mag * float(bandit_config['reward_weight_scale'])
    latest_ts = df['timestamp'].max()
    age_days = (latest_ts - df['timestamp']).dt.total_seconds().div(86400.0)
    recent_days = float(bandit_config['recent_days'])
    if recent_days > 0:
        weights[(labels != 2) & (age_days <= recent_days)] *= float(bandit_config['recent_action_boost'])
    follow_side = df['follow_side'].astype(str).to_numpy(dtype=object)
    countertrend = ((labels == 0) & (follow_side == 'DOWN')) | ((labels == 1) & (follow_side == 'UP'))
    weights[countertrend] *= float(bandit_config['countertrend_boost'])
    df['bandit_action_label'] = labels
    df['bandit_weight'] = weights


def _ensure_all_action_classes_present(train_df, val_df, test_df, feature_cols: List[str]):
    """Add tiny-weight anchor rows so LightGBM sees all 3 action classes."""
    added: List[int] = []
    present = set(train_df['bandit_action_label'].astype(int).unique().tolist())
    missing = [cls for cls in (0, 1, 2) if cls not in present]
    if not missing:
        return added, train_df

    fallback_frames = [train_df, val_df, test_df]
    for cls in missing:
        source_row = None
        for frame in fallback_frames:
            hits = frame[frame['bandit_action_label'] == cls]
            if not hits.empty:
                source_row = hits.iloc[[0]].copy()
                break
        if source_row is None:
            source_row = train_df.iloc[[0]].copy()
        for col in feature_cols:
            if col not in source_row.columns:
                source_row[col] = 0.0
        source_row['bandit_action_label'] = int(cls)
        source_row['bandit_weight'] = 1e-6
        train_df = pd.concat([train_df, source_row], ignore_index=True)
        added.append(int(cls))
    return added, train_df


def train_one(
    cell_id: str,
    windows: List[int],
    force: bool = False,
    label_config: Dict[str, Any] | None = None,
    tree_params: Dict[str, Any] | None = None,
    bandit_config: Dict[str, Any] | None = None,
    output_tag: str | None = None,
) -> Dict[str, Any]:
    cells = load_core_cell_map()
    if cell_id not in cells:
        raise KeyError(f'unknown cell_id: {cell_id}')
    cell = cells[cell_id]
    asset = f"{cell['symbol']}_USDT"
    label_config = normalize_label_config(label_config)
    tree_params = _normalize_tree_params(tree_params)
    bandit_config = _normalize_bandit_config(bandit_config)
    output_dir = candidate_dir_for_cell(cell_id, output_tag=output_tag)
    output_dir.mkdir(parents=True, exist_ok=True)
    if force:
        for path in output_dir.glob('*'):
            if path.is_file():
                path.unlink()

    df = build_cell_dataset(cell, max_window_days=max(windows), force_rebuild=force, label_config=label_config)
    _augment_teacher_features(df, resolve_current_model_dir(cell), asset)
    _build_bandit_labels(df, bandit_config)
    feature_cols = choose_tree_feature_cols(df, cell)
    for col in ['teacher_p_up', 'teacher_margin', 'teacher_abs_proxy']:
        if col not in feature_cols:
            feature_cols.append(col)

    rows_meta = []
    for window_days in windows:
        split = train_val_test_split(df, window_days)
        train_df, val_df, test_df = split['train'].copy(), split['val'].copy(), split['test'].copy()
        if len(train_df) < 1200 or len(val_df) < 150 or len(test_df) < 150:
            rows_meta.append({'window_days': window_days, 'status': 'skipped_insufficient_rows', 'train_rows': len(train_df), 'val_rows': len(val_df), 'test_rows': len(test_df)})
            continue
        for col in feature_cols:
            if col not in train_df.columns:
                train_df[col] = 0.0
                val_df[col] = 0.0
                test_df[col] = 0.0
        added_classes, train_df = _ensure_all_action_classes_present(train_df, val_df, test_df, feature_cols)
        model = LGBMClassifier(
            objective='multiclass',
            num_class=3,
            n_estimators=400,
            learning_rate=float(tree_params['learning_rate']),
            num_leaves=int(tree_params['num_leaves']),
            max_depth=int(tree_params['max_depth']),
            min_data_in_leaf=int(tree_params['min_data_in_leaf']),
            feature_fraction=float(tree_params['feature_fraction']),
            bagging_fraction=float(tree_params['bagging_fraction']),
            bagging_freq=int(tree_params['bagging_freq']),
            reg_lambda=float(tree_params['lambda_l2']),
            verbosity=-1,
        )
        model.fit(
            train_df[feature_cols],
            train_df['bandit_action_label'].to_numpy(dtype=int),
            sample_weight=train_df['bandit_weight'].to_numpy(dtype=float),
            eval_set=[(val_df[feature_cols], val_df['bandit_action_label'].to_numpy(dtype=int))],
            eval_metric='multi_logloss',
        )
        joblib.dump(model, output_dir / f'bandit_policy_lgb_{window_days}d.joblib')
        rows_meta.append({
            'window_days': window_days,
            'status': 'trained',
            'train_rows': len(train_df),
            'val_rows': len(val_df),
            'test_rows': len(test_df),
            'label_up_rows': int((train_df['bandit_action_label'] == 0).sum()),
            'label_down_rows': int((train_df['bandit_action_label'] == 1).sum()),
            'label_abstain_rows': int((train_df['bandit_action_label'] == 2).sum()),
            'anchor_added_classes': added_classes,
        })

    (output_dir / 'feature_cols.json').write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    config = {
        'generated_at': utc_now_iso(),
        'mode': 'core10_bandit_policy',
        'cell_id': cell_id,
        'profile': cell['profile'],
        'trader': cell['trader'],
        'symbol': cell['symbol'],
        'asset': asset,
        'window_days_list': windows,
        'feature_cols_count': len(feature_cols),
        'current_active_model': str(resolve_current_model_dir(cell)),
        'label_config': label_config,
        'tree_params': tree_params,
        'bandit_config': bandit_config,
        'output_tag': output_tag or '',
        'head_files': {
            'bandit_policy': [f'bandit_policy_lgb_{w}d.joblib' for w in windows if (output_dir / f'bandit_policy_lgb_{w}d.joblib').exists()],
        },
        'training_rows': rows_meta,
    }
    write_json(output_dir / 'config.json', config)
    return {
        'cell_id': cell_id,
        'output_dir': str(output_dir),
        'trained_windows': [r['window_days'] for r in rows_meta if r['status'] == 'trained'],
        'rows': rows_meta,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell-id', required=True)
    parser.add_argument('--windows', default='180,365')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--output-tag', default='')
    args = parser.parse_args()
    windows = [int(x.strip()) for x in str(args.windows).split(',') if x.strip()]
    result = train_one(args.cell_id, windows, force=args.force, output_tag=args.output_tag or None)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
