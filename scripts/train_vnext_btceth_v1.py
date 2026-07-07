#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / 'scripts' / 'train_production_model.py'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'models' / 'vnext_btceth_v1'
REPORT_PATH = PROJECT_ROOT / 'reports' / 'vnext_btceth_v1_training_latest.json'
LGB_PARAMS_PATH = PROJECT_ROOT / 'reports' / 'vnext_btceth_v1_lgb_params_latest.json'

FEATURE_GROUPS = [
    'fgi_daily',
    'ob',
    'funding',
    'oi',
    'lsratio',
    'polymarket_prob',
    'polymarket_prob_target',
]

DEFAULT_ARGS = {
    'assets': 'BTC_USDT,ETH_USDT',
    'feature_groups': ','.join(FEATURE_GROUPS),
    'train_days': '120',
    'window_days_list': '60,90,120',
    'n_estimators_cap': '600',
    'early_stopping_rounds': '40',
    'trade_objective_countertrend_penalty': '0.70',
    'trade_objective_rally_countertrend_penalty': '0.85',
    'trade_objective_high_vol_follow_bonus': '1.05',
    'trade_objective_high_vol_countertrend_penalty': '0.95',
}

LGB_COMPARE_PARAMS = {
    'n_jobs': 8,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def _write_lgb_params_snapshot() -> None:
    # 训练脚本要求这里是“纯 LightGBM 参数字典”，不能混入元信息。
    _write_json(LGB_PARAMS_PATH, dict(LGB_COMPARE_PARAMS))


def build_cmd(force: bool) -> list[str]:
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        '--output-dir',
        str(OUTPUT_DIR),
        '--assets',
        DEFAULT_ARGS['assets'],
        '--feature-groups',
        DEFAULT_ARGS['feature_groups'],
        '--train-days',
        DEFAULT_ARGS['train_days'],
        '--window-days-list',
        DEFAULT_ARGS['window_days_list'],
        '--n-estimators-cap',
        DEFAULT_ARGS['n_estimators_cap'],
        '--early-stopping-rounds',
        DEFAULT_ARGS['early_stopping_rounds'],
        '--disable-gru-embeddings',
        '--regularized',
        '--lgb-params-json',
        str(LGB_PARAMS_PATH),
        '--lgb-params-name',
        'VNEXT_COMPARE_M4',
        '--trade-objective-weighting',
        '--uncertainty-aware-weighting',
        '--trade-objective-countertrend-penalty',
        DEFAULT_ARGS['trade_objective_countertrend_penalty'],
        '--trade-objective-rally-countertrend-penalty',
        DEFAULT_ARGS['trade_objective_rally_countertrend_penalty'],
        '--trade-objective-high-vol-follow-bonus',
        DEFAULT_ARGS['trade_objective_high_vol_follow_bonus'],
        '--trade-objective-high-vol-countertrend-penalty',
        DEFAULT_ARGS['trade_objective_high_vol_countertrend_penalty'],
    ]
    if force:
        cmd.append('--sim-noise')
    return cmd


def patch_config() -> Dict[str, Any]:
    config_path = OUTPUT_DIR / 'config.json'
    cfg = _read_json(config_path)
    cfg['version'] = 'vnext_btceth_v1'
    cfg['experiment'] = 'vnext_btceth_v1 (btc/eth compare-only, regularized, no cfgi/news)'
    cfg['compare_only'] = True
    cfg['monitor_only'] = True
    cfg['disable_gru_embeddings'] = True
    cfg['compare_group'] = 'vnext_btceth_compare'
    cfg['compare_traders'] = ['vnext_btc_v1', 'vnext_eth_v1']
    cfg['post_init_emphasis'] = {
        'method': 'short_horizon_train_window_proxy',
        'train_days': 120,
        'notes': '用较短训练窗口和交易目标加权，避免旧 regime 主导。',
    }
    cfg['design_notes'] = {
        'kept_simple': True,
        'skip_cfgi': True,
        'external_lobster_data': False,
        'absorb_lessons': [
            'no_family_wide_retrain',
            'no_production_replacement_first',
            'no_cfgi',
            'trade_objective_over_direction_only',
            'compare_only_lane',
        ],
        'borrowed_ideas': [
            'abstain_on_bad_markets',
            'confidence_calibration',
            'market_selection_before_more_guard_layers',
        ],
    }
    _write_json(config_path, cfg)
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description='Train compare-only BTC/ETH vnext model')
    parser.add_argument('--force', action='store_true', help='Retrain even if model artifacts already exist')
    args = parser.parse_args()

    existing = OUTPUT_DIR / 'config.json'
    if existing.exists() and not args.force:
        cfg = patch_config()
        _write_json(REPORT_PATH, {
            'generated_at': utc_now_iso(),
            'status': 'reused_existing_model',
            'output_dir': str(OUTPUT_DIR),
            'assets': cfg.get('assets', []),
            'feature_groups': cfg.get('feature_groups', []),
            'metrics': cfg.get('metrics', {}),
            'lgb_params_snapshot': str(LGB_PARAMS_PATH),
        })
        print(f'reused_existing_model={OUTPUT_DIR}')
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_lgb_params_snapshot()
    cmd = build_cmd(force=args.force)
    env = dict(os.environ)
    env.setdefault('MPLCONFIGDIR', '/tmp/mpl_vnext')
    env.setdefault('PYTHONUNBUFFERED', '1')
    env.setdefault('OMP_NUM_THREADS', '8')
    env.setdefault('OPENBLAS_NUM_THREADS', '8')
    env.setdefault('MKL_NUM_THREADS', '8')
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    if proc.returncode != 0:
        _write_json(REPORT_PATH, {
            'generated_at': utc_now_iso(),
            'status': 'training_failed',
            'returncode': proc.returncode,
            'cmd': cmd,
            'lgb_params_snapshot': str(LGB_PARAMS_PATH),
        })
        return proc.returncode

    cfg = patch_config()
    _write_json(REPORT_PATH, {
        'generated_at': utc_now_iso(),
        'status': 'trained',
        'output_dir': str(OUTPUT_DIR),
        'cmd': cmd,
        'assets': cfg.get('assets', []),
        'feature_groups': cfg.get('feature_groups', []),
        'metrics': cfg.get('metrics', {}),
        'window_days_list': cfg.get('window_days_list', []),
        'design_notes': cfg.get('design_notes', {}),
        'lgb_params_snapshot': str(LGB_PARAMS_PATH),
    })
    print(f'trained_model_dir={OUTPUT_DIR}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
