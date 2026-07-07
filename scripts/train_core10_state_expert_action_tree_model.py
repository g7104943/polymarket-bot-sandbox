#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops.core10_retrain_common import (
    ACTION_LABELS,
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
)
from scripts.train_core10_action_tree_model import (
    _build_action_labels,
    _fit_abstain,
    _fit_action_model,
    _fit_state_model,
    _normalize_action_config,
    _normalize_tree_params,
    resolve_action_label_version,
)
from scripts.ops.core10_retrain_common import train_val_test_split

TREE_ROOT = DATA / 'models'

EXPERT_GROUPS = {
    'rally_up': ['rally_up', 'extreme_up'],
    'rally_down': ['rally_down', 'extreme_down'],
    'fallback': ['normal', 'high_vol_chop'],
}

DEFAULT_EXPERT_CONFIG = {
    'expert_in_weight': 2.5,
    'expert_out_weight': 0.20,
}


def _normalize_expert_config(cfg: Dict[str, Any] | None = None) -> Dict[str, float]:
    out = dict(DEFAULT_EXPERT_CONFIG)
    if isinstance(cfg, dict):
        for k in out:
            raw = cfg.get(k)
            if raw is None:
                continue
            try:
                out[k] = float(raw)
            except Exception:
                continue
    return out


def candidate_dir_for_cell(cell_id: str, output_tag: str | None = None) -> Path:
    safe = cell_id.replace('/', '__').replace('-', '_').lower()
    if output_tag:
        safe_tag = output_tag.replace('/', '__').replace('-', '_').lower()
        return TREE_ROOT / f'core10_state_expert_action_{safe}__{safe_tag}'
    return TREE_ROOT / f'core10_state_expert_action_{safe}'


def train_one(
    cell_id: str,
    windows: List[int],
    force: bool = False,
    label_config: Dict[str, Any] | None = None,
    tree_params: Dict[str, Any] | None = None,
    action_config: Dict[str, Any] | None = None,
    expert_config: Dict[str, Any] | None = None,
    output_tag: str | None = None,
) -> Dict[str, Any]:
    cells = load_core_cell_map()
    if cell_id not in cells:
        raise KeyError(f'unknown cell_id: {cell_id}')
    cell = cells[cell_id]
    label_config = normalize_label_config(label_config)
    tree_params = _normalize_tree_params(tree_params)
    action_config = _normalize_action_config(action_config)
    expert_config = _normalize_expert_config(expert_config)
    action_label_version = resolve_action_label_version(label_config) + '__state_expert_v1'
    output_dir = candidate_dir_for_cell(cell_id, output_tag=output_tag)
    output_dir.mkdir(parents=True, exist_ok=True)

    if force:
        for path in output_dir.glob('*'):
            if path.is_file():
                path.unlink()

    df = build_cell_dataset(cell, max_window_days=max(windows), force_rebuild=force, label_config=label_config)
    df = _build_action_labels(df, action_config, label_config=label_config)
    feature_cols = choose_tree_feature_cols(df, cell)
    rows_meta = []

    for window_days in windows:
        split = train_val_test_split(df, window_days)
        train_df, val_df, test_df = split['train'].copy(), split['val'].copy(), split['test'].copy()
        if len(train_df) < 800 or len(val_df) < 120 or len(test_df) < 120:
            rows_meta.append({
                'window_days': window_days,
                'status': 'skipped_insufficient_rows',
                'train_rows': len(train_df),
                'val_rows': len(val_df),
                'test_rows': len(test_df),
            })
            continue

        for col in feature_cols:
            if col not in train_df.columns:
                train_df[col] = 0.0
                val_df[col] = 0.0
                test_df[col] = 0.0

        X_tr = train_df[feature_cols]
        X_va = val_df[feature_cols]
        y_state_tr = train_df['state_label'].map(STATE_TO_ID).astype(int)
        y_state_va = val_df['state_label'].map(STATE_TO_ID).astype(int)
        y_action_tr = train_df['action_label'].astype(int)
        y_action_va = val_df['action_label'].astype(int)

        state_model = _fit_state_model(X_tr, y_state_tr, train_df['state_head_weight'], X_va, y_state_va, tree_params)
        joblib.dump(state_model, output_dir / f'state_lgb_{window_days}d.joblib')

        expert_stats = {}
        for expert_name, states in EXPERT_GROUPS.items():
            tr_mask = train_df['state_label'].isin(states).to_numpy(dtype=bool)
            va_mask = val_df['state_label'].isin(states).to_numpy(dtype=bool)
            action_weight = train_df['action_weight'].to_numpy(dtype=float).copy()
            abstain_weight = train_df['abstain_train_weight'].to_numpy(dtype=float).copy()
            action_weight[tr_mask] *= float(expert_config['expert_in_weight'])
            abstain_weight[tr_mask] *= float(expert_config['expert_in_weight'])
            action_weight[~tr_mask] *= float(expert_config['expert_out_weight'])
            abstain_weight[~tr_mask] *= float(expert_config['expert_out_weight'])

            action_model = _fit_action_model(X_tr, y_action_tr, action_weight, X_va, y_action_va, tree_params)
            abstain_model = _fit_abstain(X_tr, train_df['abstain_label'], abstain_weight, X_va, val_df['abstain_label'], tree_params)
            joblib.dump(action_model, output_dir / f'action_{expert_name}_lgb_{window_days}d.joblib')
            joblib.dump(abstain_model, output_dir / f'abstain_{expert_name}_lgb_{window_days}d.joblib')
            expert_stats[expert_name] = {
                'train_rows_in_group': int(tr_mask.sum()),
                'val_rows_in_group': int(va_mask.sum()),
                'train_share': round(float(tr_mask.mean()), 6),
            }

        rows_meta.append({
            'window_days': window_days,
            'status': 'trained',
            'train_rows': len(train_df),
            'val_rows': len(val_df),
            'test_rows': len(test_df),
            'experts': expert_stats,
            'up_rate': round(float((train_df['action_name'] == 'UP').mean()), 6),
            'down_rate': round(float((train_df['action_name'] == 'DOWN').mean()), 6),
            'abstain_rate': round(float((train_df['action_name'] == 'ABSTAIN').mean()), 6),
        })

    (output_dir / 'feature_cols.json').write_text(json.dumps(feature_cols, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    config = {
        'generated_at': utc_now_iso(),
        'mode': 'core10_state_expert_action_specialist',
        'action_label_version': action_label_version,
        'cell_id': cell_id,
        'profile': cell['profile'],
        'trader': cell['trader'],
        'symbol': cell['symbol'],
        'asset': f"{cell['symbol']}_USDT",
        'state_labels': STATE_LABELS,
        'action_labels': ACTION_LABELS,
        'window_days_list': windows,
        'feature_cols_count': len(feature_cols),
        'current_active_model': str(resolve_current_model_dir(cell)),
        'label_config': label_config,
        'tree_params': tree_params,
        'action_config': action_config,
        'expert_config': expert_config,
        'expert_groups': EXPERT_GROUPS,
        'output_tag': output_tag or '',
        'head_files': {
            'state': [f'state_lgb_{w}d.joblib' for w in windows if (output_dir / f'state_lgb_{w}d.joblib').exists()],
            'expert_action': {
                expert: [f'action_{expert}_lgb_{w}d.joblib' for w in windows if (output_dir / f'action_{expert}_lgb_{w}d.joblib').exists()]
                for expert in EXPERT_GROUPS
            },
            'expert_abstain': {
                expert: [f'abstain_{expert}_lgb_{w}d.joblib' for w in windows if (output_dir / f'abstain_{expert}_lgb_{w}d.joblib').exists()]
                for expert in EXPERT_GROUPS
            },
        },
        'training_rows': rows_meta,
    }
    write_json(output_dir / 'config.json', config)
    return {
        'cell_id': cell_id,
        'output_dir': str(output_dir),
        'trained_windows': [row['window_days'] for row in rows_meta if row['status'] == 'trained'],
        'rows': rows_meta,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell-id', required=True)
    parser.add_argument('--windows', default='180,365')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--output-tag', default='')
    parser.add_argument('--label-config-json', default='')
    parser.add_argument('--tree-params-json', default='')
    parser.add_argument('--action-config-json', default='')
    parser.add_argument('--expert-config-json', default='')
    args = parser.parse_args()

    windows = [int(part) for part in str(args.windows).split(',') if str(part).strip()]
    label_config = json.loads(args.label_config_json) if str(args.label_config_json).strip() else None
    tree_params = json.loads(args.tree_params_json) if str(args.tree_params_json).strip() else None
    action_config = json.loads(args.action_config_json) if str(args.action_config_json).strip() else None
    expert_config = json.loads(args.expert_config_json) if str(args.expert_config_json).strip() else None
    result = train_one(
        args.cell_id,
        windows,
        force=args.force,
        label_config=label_config,
        tree_params=tree_params,
        action_config=action_config,
        expert_config=expert_config,
        output_tag=(args.output_tag or None),
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
