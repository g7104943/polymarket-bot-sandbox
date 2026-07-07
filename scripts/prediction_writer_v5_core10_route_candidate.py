#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLY = PROJECT_ROOT / 'polymarket'
REPORTS = PROJECT_ROOT / 'reports'
LOGS = PROJECT_ROOT / 'logs'
LOGS.mkdir(parents=True, exist_ok=True)

import sys

sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops.core10_retrain_common import (  # noqa: E402
    build_cell_dataset,
    load_core_cell_map,
    normalize_policy_config,
    policy_actions_from_scores,
)
from scripts.ops.generate_core10_route_a_cell_search_latest import _combine_scores  # noqa: E402

ROLLOUT = REPORTS / 'core10_route_rollout_latest.json'
DEFAULT_SLEEP_SEC = 30
DEFAULT_FILE_MAX_AGE_SEC = 5 * 60


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _compute_target_period(now_ts: int) -> int:
    return (now_ts // 900) * 900


def _confidence_value(up_score: float, down_score: float) -> float:
    return float(np.clip(max(up_score, down_score), 0.0, 1.0))


def _find_job(cell_id: str) -> Dict[str, Any]:
    rollout = _load_json(ROLLOUT)
    if str(rollout.get('winning_route') or '') != 'A':
        return {}
    for row in rollout.get('jobs', []):
        if isinstance(row, dict) and str(row.get('cell_id') or '') == cell_id:
            return row
    return {}


def _load_job_spec(cell_id: str) -> Dict[str, Any]:
    job = _find_job(cell_id)
    up_dir = Path(str(job.get('up_candidate_dir') or '')) if job else Path('')
    down_dir = Path(str(job.get('down_candidate_dir') or '')) if job and job.get('down_candidate_dir') else None
    cfg_source = up_dir if up_dir.exists() else (down_dir if down_dir and down_dir.exists() else None)
    model_cfg = _load_json(cfg_source / 'config.json') if cfg_source else {}
    label_config = model_cfg.get('label_config') if isinstance(model_cfg.get('label_config'), dict) else {}
    window_days = [int(x) for x in (model_cfg.get('window_days_list') or [365]) if str(x).strip()]
    return {
        'candidate_ready': bool(job) and up_dir.exists(),
        'job': job,
        'up_dir': up_dir,
        'down_dir': down_dir if down_dir and down_dir.exists() else None,
        'label_config': label_config,
        'window_days': window_days or [365],
    }


def _build_prediction_payload(cell_id: str, output_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cell = load_core_cell_map()[cell_id]
    spec = _load_job_spec(cell_id)
    policy_cfg = normalize_policy_config((spec['job'] or {}).get('policy_config') if spec['job'] else None)
    guard_depth = int((spec['job'] or {}).get('guard_depth') or 0)
    route_states_by_side = (spec['job'] or {}).get('route_states_by_side') or {}

    df = build_cell_dataset(
        cell,
        max_window_days=max(spec['window_days']),
        force_rebuild=True,
        label_config=spec['label_config'],
    )
    infer_df = df.tail(2048).copy().reset_index(drop=True)
    if spec['candidate_ready']:
        scores = _combine_scores(infer_df, spec['up_dir'], spec['down_dir'])
    else:
        scores = {
            'up_score': np.zeros(len(infer_df), dtype=float),
            'down_score': np.zeros(len(infer_df), dtype=float),
            'abstain_score': np.ones(len(infer_df), dtype=float),
        }
    actions = policy_actions_from_scores(
        scores['up_score'],
        scores['down_score'],
        scores['abstain_score'],
        policy_config=policy_cfg,
    )

    idx = len(infer_df) - 1
    last = infer_df.iloc[idx]
    up_score = float(scores['up_score'][idx])
    down_score = float(scores['down_score'][idx])
    abstain_score = float(scores['abstain_score'][idx])
    action = str(actions[idx]) if idx >= 0 else 'ABSTAIN'
    state_label = str(last.get('state_label') or 'unknown')

    should_trade = bool(spec['candidate_ready'] and action != 'ABSTAIN')
    skip_reason = None
    if not spec['candidate_ready']:
        skip_reason = 'route_rollout_not_ready'
    elif guard_depth >= 1:
        allowed = set(str(x) for x in (route_states_by_side.get(action) or []) if str(x))
        if allowed and state_label not in allowed:
            should_trade = False
            skip_reason = f'route_state_not_eligible:{state_label}'
    if should_trade is False and skip_reason is None:
        skip_reason = 'policy_abstain'

    direction = 'UP' if up_score >= down_score else 'DOWN'
    confidence = _confidence_value(up_score, down_score)
    now = datetime.now(timezone.utc)
    now_ts = int(now.timestamp())
    target_period_end_ts = _compute_target_period(now_ts)
    prediction = {
        'symbol': f"{cell['symbol']}/USDT",
        'timeframe': '15m',
        'direction': direction,
        'confidence': round(confidence, 6),
        'timestamp': now.isoformat(),
        'details': {
            'route': 'A',
            'cell_id': cell_id,
            'candidate_trader_name': str((spec['job'] or {}).get('candidate_trader_name') or ''),
            'route_states_by_side': route_states_by_side,
            'guard_depth': guard_depth,
            'up_score': round(up_score, 6),
            'down_score': round(down_score, 6),
            'abstain_score': round(abstain_score, 6),
            'trade_decision': {
                'should_trade': should_trade,
                'skip_reason': skip_reason,
            },
            'policy_config': policy_cfg,
            'model_version': f"core10_route_candidate::A::{cell_id}",
        },
    }
    payload = {
        'timestamp': now.isoformat(),
        'target_period_end_ts': target_period_end_ts,
        'model_version': f"core10_route_candidate::A::{cell_id}",
        'phase': 1,
        'limit_price': 0.53,
        'bet_fraction_this_phase': 1.0,
        'max_sweep_price': 0.54,
        'predictions': {f"{cell['symbol']}_USDT_15m": prediction},
    }
    meta = {
        'target_period_end_ts': target_period_end_ts,
        'candidate_ready': spec['candidate_ready'],
        'cell_id': cell_id,
        'route_state': state_label,
        'should_trade': should_trade,
        'skip_reason': skip_reason,
        'output_file': str(output_path),
    }
    return payload, meta


def _write_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def _should_refresh(output_path: Path, target_period_end_ts: int, last_slot: int | None, last_sig: str | None, next_sig: str, max_age_sec: int) -> bool:
    if last_slot is None or target_period_end_ts != last_slot:
        return True
    if last_sig != next_sig:
        return True
    if not output_path.exists():
        return True
    try:
        age = time.time() - output_path.stat().st_mtime
    except OSError:
        return True
    return age >= max_age_sec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell-id', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--sleep-sec', type=int, default=DEFAULT_SLEEP_SEC)
    parser.add_argument('--max-age-sec', type=int, default=DEFAULT_FILE_MAX_AGE_SEC)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    output_path = Path(args.output)
    last_slot: int | None = None
    last_sig: str | None = None

    while True:
        spec = _load_job_spec(args.cell_id)
        next_sig = '|'.join([
            args.cell_id,
            str(spec['up_dir']),
            str(spec['down_dir']),
            json.dumps((spec['job'] or {}).get('policy_config') or {}, sort_keys=True, ensure_ascii=False),
            json.dumps((spec['job'] or {}).get('route_states_by_side') or {}, sort_keys=True, ensure_ascii=False),
            str((spec['job'] or {}).get('guard_depth') or 0),
            str(spec['candidate_ready']),
        ])
        target_period_end_ts = _compute_target_period(int(time.time()))
        if _should_refresh(output_path, target_period_end_ts, last_slot, last_sig, next_sig, max(30, int(args.max_age_sec))):
            try:
                payload, meta = _build_prediction_payload(args.cell_id, output_path)
                _write_atomic(output_path, payload)
                last_slot = int(meta['target_period_end_ts'])
                last_sig = next_sig
                logging.info(
                    'route candidate write ok cell=%s state=%s should_trade=%s skip=%s output=%s',
                    args.cell_id,
                    meta['route_state'],
                    meta['should_trade'],
                    meta['skip_reason'],
                    output_path,
                )
            except Exception as exc:
                logging.exception('route candidate write failed: %s', exc)
        if args.once:
            return 0
        time.sleep(max(10, int(args.sleep_sec)))


if __name__ == '__main__':
    raise SystemExit(main())
