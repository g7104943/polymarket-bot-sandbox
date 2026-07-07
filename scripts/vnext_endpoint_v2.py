#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_vnext_btceth_entry_exit_v1 import (
    BUY_GRIDS,
    EXIT_FAMILIES,
    SELL_QUOTE_POLICIES,
    _apply_calibration_meta,
    _bucketize_metric,
    _candidate_train_days,
    _confidence_bucket,
    _direction_trend_bucket,
    _fit_heads,
    _iter_purged_splits,
    _select_calibration_from_split,
    _simulate_family_policy,
    _spread_bucket_to_proxy,
    _spread_metric_from_row,
    _trend_score_from_row,
    _utility_adjusted,
    _vol_metric_from_row,
    polymarket_taker_fee_rate,
)
from scripts.vnext_execution_common import (
    ASSETS,
    ASSET_SYMBOL,
    EXECUTION_LABEL_PATHS,
    EXECUTION_REPORT,
    HISTORICAL_EXECUTION_BOOTSTRAP_REPORT,
    LIFECYCLE_EVENT_PATHS,
    ORDERBOOK_EVENT_PATHS,
    PROFIT_ALPHA_REPORT,
    PROFIT_RELABEL_REPORT,
    READINESS_REPORT,
    RELABELED_EPISODE_PATHS,
    SYMBOL_ASSET,
    V1_EPISODE_PATHS,
    V1_MODEL_DIRS,
    V2A_MODEL_DIRS,
    V2B_MODEL_DIRS,
    default_execution_bootstrap,
    filter_rows_after_reset,
    load_json,
    load_jsonl,
    load_latest_reset_report,
    load_overlay_rows,
    normalize_fill_status,
    utc_now_iso,
    write_json,
    write_readiness_report,
)
from scripts.vnext_entry_replay import replay_entry_trace
from scripts.vnext_stage_common import find_illegal_feature_cols

SEARCH_WORKERS = 8
MIN_TRADES = 30
MIN_SNAPSHOTS = 1000
MIN_FILL_SAMPLES = 20
MIN_HIST_SETTLED_TRADES = 1000
MIN_HIST_FILL_EVENTS = 1000
MIN_TRAIN_ROWS = 800
MIN_VAL_ROWS = 180
MIN_TEST_ROWS = 180
V1_ALIGNMENT_AUDIT = PROJECT_ROOT / 'reports' / 'vnext_entry_exit_v1_alignment_audit_latest.json'

BOOTSTRAP_QUEUE_DEPTH = {'BTC_USDT': 850.0, 'ETH_USDT': 900.0}
BOOTSTRAP_TIMEOUT_PENALTY = 0.015
BOOTSTRAP_NO_FILL_PENALTY = 0.020
BOOTSTRAP_HIGH_CONF_LOSS_PENALTY = 0.010
BOOTSTRAP_QUEUE_WAIT_PENALTY_PER_SEC = 0.00015

LEGACY_PARTIAL_RE = re.compile(
    r"\[挂单部分成交\]\s+(BTC|ETH)\s+@\$(?P<limit>[\d.]+):\s+成交\s+\$(?P<filled>[\d.]+)\s+\((?P<ratio>[\d.]+)%\),\s+剩余冻结\s+\$(?P<remaining>[\d.]+)\s+\(买方竞争=\$(?P<competition>[\d.]+)\)"
)
LEGACY_FULL_RE = re.compile(
    r"\[挂单成交\]\s+(BTC|ETH)\s+@\$(?P<limit>[\d.]+):\s+完全成交\s+\$(?P<filled>[\d.]+)\s+\(best_ask=\$(?P<best_ask>[\d.]+),\s+均价=\$(?P<avg_price>[\d.]+),\s+买方竞争=\$(?P<competition>[\d.]+)\)"
)
LEGACY_TIMEOUT_RE = re.compile(r"(BTC|ETH).{0,32}(最终未买|未成交超时|超时未成交)")

PARAM_TEMPLATES: dict[str, dict[str, Any]] = {
    'conservative': {
        'num_leaves': 31, 'max_depth': 5, 'learning_rate': 0.03, 'min_child_samples': 80,
        'feature_fraction': 0.70, 'bagging_fraction': 0.75, 'bagging_freq': 2,
        'lambda_l1': 0.30, 'lambda_l2': 12.0, 'max_bin': 255, 'min_split_gain': 0.05,
    },
    'balanced': {
        'num_leaves': 63, 'max_depth': 6, 'learning_rate': 0.02, 'min_child_samples': 50,
        'feature_fraction': 0.80, 'bagging_fraction': 0.85, 'bagging_freq': 2,
        'lambda_l1': 0.10, 'lambda_l2': 10.0, 'max_bin': 255, 'min_split_gain': 0.03,
    },
    'elastic': {
        'num_leaves': 95, 'max_depth': 8, 'learning_rate': 0.015, 'min_child_samples': 30,
        'feature_fraction': 0.90, 'bagging_fraction': 0.90, 'bagging_freq': 1,
        'lambda_l1': 0.05, 'lambda_l2': 6.0, 'max_bin': 383, 'min_split_gain': 0.01,
    },
}


@dataclass
class ExecutionProfile:
    fill_rate: float
    partial_fill_rate: float
    timeout_rate: float
    avg_partial_fill_ratio: float
    avg_queue_wait_seconds: float
    avg_entry_queue_depth: float
    avg_exit_queue_depth: float
    exit_family: str
    sell_quote_policy: str
    source: str


@dataclass
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def _round_float(v: Any, digits: int = 6) -> Any:
    try:
        fv = float(v)
    except Exception:
        return v
    if math.isnan(fv) or math.isinf(fv):
        return None
    return round(fv, digits)


def _safe_mean(values: Sequence[float], default: float = 0.0) -> float:
    arr = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(arr)) if arr else default


def _default_entry_fill_mode(row: dict[str, Any]) -> str:
    return 'bootstrap_taker' if str(row.get('status') or '').lower() == 'executed' else 'queued'


def _spread_bucket_from_value(val: float) -> str:
    return _bucketize_metric(float(val), 0.01, 0.03, labels=('tight', 'normal', 'wide'))


def _load_observation_rows(asset: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return (
        filter_rows_after_reset(load_jsonl(ORDERBOOK_EVENT_PATHS[asset])),
        filter_rows_after_reset(load_jsonl(LIFECYCLE_EVENT_PATHS[asset])),
    )


def _iter_legacy_exp_log_dirs() -> list[Path]:
    poly_dir = PROJECT_ROOT / 'polymarket'
    out: list[Path] = []
    for path in sorted(poly_dir.glob('logs*')):
        if not path.is_dir():
            continue
        name = path.name
        if not (name.startswith('logs_v5_exp') or name.startswith('logs_core10_v5_exp') or name.startswith('logs_70_logs_v5_exp')):
            continue
        out.append(path)
    return out


def _load_first_trade_file(log_dir: Path) -> list[dict[str, Any]]:
    for filename in ('prediction_trades.simulation.json', 'prediction_trades.json'):
        path = log_dir / filename
        if path.exists():
            payload = load_json(path)
            if isinstance(payload, list):
                return [row for row in payload if isinstance(row, dict)]
    return []


def _parse_legacy_log_events(log_dir: Path) -> list[dict[str, Any]]:
    path = log_dir / 'trading_stdout.log'
    if not path.exists():
        return []
    try:
        text = path.read_text(encoding='utf-8', errors='replace')
    except Exception:
        return []
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        m = LEGACY_PARTIAL_RE.search(line)
        if m:
            symbol = str(m.group(1)).upper()
            rows.append({
                'asset': SYMBOL_ASSET.get(symbol),
                'symbol': symbol,
                'fill_status': 'partial_fill',
                'partial_fill_ratio': float(m.group('ratio')) / 100.0,
                'entry_limit_price': float(m.group('limit')),
                'filled_usd': float(m.group('filled')),
                'remaining_usd': float(m.group('remaining')),
                'queue_competition_usd': float(m.group('competition')),
                'best_ask': None,
                'avg_fill_price': None,
            })
            continue
        m = LEGACY_FULL_RE.search(line)
        if m:
            symbol = str(m.group(1)).upper()
            rows.append({
                'asset': SYMBOL_ASSET.get(symbol),
                'symbol': symbol,
                'fill_status': 'filled',
                'partial_fill_ratio': 1.0,
                'entry_limit_price': float(m.group('limit')),
                'filled_usd': float(m.group('filled')),
                'remaining_usd': 0.0,
                'queue_competition_usd': float(m.group('competition')),
                'best_ask': float(m.group('best_ask')),
                'avg_fill_price': float(m.group('avg_price')),
            })
            continue
        m = LEGACY_TIMEOUT_RE.search(line)
        if m:
            symbol = str(m.group(1)).upper()
            rows.append({
                'asset': SYMBOL_ASSET.get(symbol),
                'symbol': symbol,
                'fill_status': 'timeout_unfilled',
                'partial_fill_ratio': 0.0,
                'entry_limit_price': None,
                'filled_usd': 0.0,
                'remaining_usd': 0.0,
                'queue_competition_usd': None,
                'best_ask': None,
                'avg_fill_price': None,
            })
    return [row for row in rows if row.get('asset') in ASSETS]


def _build_historical_execution_bootstrap() -> dict[str, Any]:
    prior_weight = 50.0
    assets_payload: dict[str, Any] = {}
    grouped_events: dict[str, list[dict[str, Any]]] = {asset: [] for asset in ASSETS}
    grouped_trades: dict[str, list[dict[str, Any]]] = {asset: [] for asset in ASSETS}
    scanned_dirs: list[str] = []
    for log_dir in _iter_legacy_exp_log_dirs():
        scanned_dirs.append(str(log_dir))
        for row in _load_first_trade_file(log_dir):
            symbol = str(row.get('symbol') or '').upper()
            asset = SYMBOL_ASSET.get(symbol)
            if asset not in ASSETS:
                continue
            grouped_trades[asset].append(row)
        for event in _parse_legacy_log_events(log_dir):
            asset = str(event.get('asset') or '')
            if asset in grouped_events:
                grouped_events[asset].append(event)
    for asset in ASSETS:
        bootstrap = default_execution_bootstrap(asset)
        defaults = bootstrap['defaults']
        trade_rows = grouped_trades[asset]
        events = grouped_events[asset]
        partial_events = [row for row in events if row['fill_status'] == 'partial_fill']
        full_events = [row for row in events if row['fill_status'] == 'filled']
        timeout_events = [row for row in events if row['fill_status'] == 'timeout_unfilled']
        observed_fill_events = len(partial_events) + len(full_events)
        observed_outcomes = observed_fill_events + len(timeout_events)
        fill_prior = float(defaults['fill_rate']) * prior_weight
        timeout_prior = float(defaults['timeout_rate']) * prior_weight
        partial_prior = float(defaults['partial_fill_rate']) * prior_weight
        partial_ratio_prior = float(defaults['avg_partial_fill_ratio']) * prior_weight
        fill_rate = (fill_prior + observed_fill_events) / (prior_weight + observed_outcomes) if observed_outcomes > 0 else float(defaults['fill_rate'])
        timeout_rate = (timeout_prior + len(timeout_events)) / (prior_weight + observed_outcomes) if observed_outcomes > 0 else float(defaults['timeout_rate'])
        partial_fill_rate = (partial_prior + len(partial_events)) / (prior_weight + observed_fill_events) if observed_fill_events > 0 else float(defaults['partial_fill_rate'])
        avg_partial_fill_ratio = (
            (partial_ratio_prior + sum(float(row['partial_fill_ratio']) for row in partial_events)) / (prior_weight + len(partial_events))
            if partial_events else float(defaults['avg_partial_fill_ratio'])
        )
        fill_rate = min(0.98, max(0.80, fill_rate))
        timeout_rate = min(0.15, max(0.02, timeout_rate))
        partial_fill_rate = min(0.80, max(0.05, partial_fill_rate))
        avg_partial_fill_ratio = min(0.97, max(0.35, avg_partial_fill_ratio))
        queue_comp = [float(row['queue_competition_usd']) for row in events if row.get('queue_competition_usd') is not None]
        best_asks = [float(row['best_ask']) for row in full_events if row.get('best_ask') is not None]
        avg_trade_pnl = _safe_mean([float(row.get('pnl') or 0.0) for row in trade_rows if row.get('pnl') is not None], 0.0)
        settled = [
            row for row in trade_rows
            if str(row.get('status') or '').lower() == 'executed' and str(row.get('result') or '').lower() in {'win', 'lose'}
        ]
        assets_payload[asset] = {
            'asset': asset,
            'symbol': ASSET_SYMBOL[asset],
            'legacy_log_dirs_scanned': len(scanned_dirs),
            'legacy_trade_rows': int(len(trade_rows)),
            'legacy_settled_trades': int(len(settled)),
            'partial_fill_events': int(len(partial_events)),
            'full_fill_events': int(len(full_events)),
            'timeout_events': int(len(timeout_events)),
            'avg_partial_fill_ratio': _round_float(avg_partial_fill_ratio, 6),
            'avg_queue_competition_usd': _round_float(_safe_mean(queue_comp, 0.0), 6),
            'avg_best_ask_on_fill': _round_float(_safe_mean(best_asks, 0.0), 6),
            'avg_trade_pnl': _round_float(avg_trade_pnl, 6),
            'bootstrap_defaults': {
                'fill_rate': _round_float(fill_rate, 6),
                'partial_fill_rate': _round_float(partial_fill_rate, 6),
                'timeout_rate': _round_float(timeout_rate, 6),
                'avg_partial_fill_ratio': _round_float(avg_partial_fill_ratio, 6),
                'avg_queue_wait_seconds': defaults['avg_queue_wait_seconds'],
                'default_exit_family': defaults['default_exit_family'],
                'default_sell_quote_policy': defaults['default_sell_quote_policy'],
                'avg_entry_queue_depth': _round_float(max(_safe_mean(queue_comp, BOOTSTRAP_QUEUE_DEPTH[asset]), BOOTSTRAP_QUEUE_DEPTH[asset]), 6),
                'avg_exit_queue_depth': _round_float(max(_safe_mean(queue_comp, BOOTSTRAP_QUEUE_DEPTH[asset]), BOOTSTRAP_QUEUE_DEPTH[asset]), 6),
            },
        }
    payload = {
        'generated_at': utc_now_iso(),
        'scope': 'vnext_execution_v2_historical_bootstrap',
        'source': 'legacy_exp_logs_and_trades',
        'assets': assets_payload,
    }
    write_json(HISTORICAL_EXECUTION_BOOTSTRAP_REPORT, payload)
    return payload


def _load_or_build_historical_execution_bootstrap() -> dict[str, Any]:
    payload = load_json(HISTORICAL_EXECUTION_BOOTSTRAP_REPORT)
    assets = payload.get('assets') if isinstance(payload, dict) else None
    if isinstance(assets, dict) and assets:
        return payload
    return _build_historical_execution_bootstrap()


def _readiness_with_historical_bootstrap() -> dict[str, Any]:
    readiness = write_readiness_report()
    historical = _load_or_build_historical_execution_bootstrap()
    for asset, row in readiness.get('assets', {}).items():
        hist = historical.get('assets', {}).get(asset, {}) if isinstance(historical, dict) else {}
        row['historical_bootstrap'] = {
            'legacy_trade_rows': int(hist.get('legacy_trade_rows') or 0),
            'legacy_settled_trades': int(hist.get('legacy_settled_trades') or 0),
            'partial_fill_events': int(hist.get('partial_fill_events') or 0),
            'full_fill_events': int(hist.get('full_fill_events') or 0),
            'timeout_events': int(hist.get('timeout_events') or 0),
        }
    write_json(READINESS_REPORT, readiness)
    return {
        **readiness,
        'historical_bootstrap_report': str(HISTORICAL_EXECUTION_BOOTSTRAP_REPORT),
    }


def _load_or_refresh_readiness() -> dict[str, Any]:
    readiness = load_json(READINESS_REPORT)
    assets = readiness.get('assets') if isinstance(readiness, dict) else None
    if isinstance(assets, dict) and assets:
        return readiness
    return _readiness_with_historical_bootstrap()


def _build_legacy_seed_dataset_for_asset(asset: str, historical_payload: dict[str, Any] | None, target_rows: int) -> pd.DataFrame:
    hist = ((historical_payload or {}).get('assets') or {}).get(asset) if isinstance(historical_payload, dict) else None
    if not isinstance(hist, dict):
        return pd.DataFrame()
    defaults = hist.get('bootstrap_defaults') if isinstance(hist.get('bootstrap_defaults'), dict) else {}
    partial_events = int(hist.get('partial_fill_events') or 0)
    full_events = int(hist.get('full_fill_events') or 0)
    timeout_events = int(hist.get('timeout_events') or 0)
    total_events = partial_events + full_events + timeout_events
    if total_events <= 0 or target_rows <= 0:
        return pd.DataFrame()

    def _alloc(count: int) -> int:
        return max(0, min(count, int(round(target_rows * (count / total_events)))))

    partial_n = _alloc(partial_events)
    full_n = _alloc(full_events)
    timeout_n = _alloc(timeout_events)
    allocated = partial_n + full_n + timeout_n
    if allocated < target_rows:
        full_n += target_rows - allocated

    avg_partial_fill_ratio = float(defaults.get('avg_partial_fill_ratio') or 0.48)
    avg_queue_wait_seconds = float(defaults.get('avg_queue_wait_seconds') or 22.0)
    avg_entry_queue_depth = float(defaults.get('avg_entry_queue_depth') or BOOTSTRAP_QUEUE_DEPTH[asset])
    avg_exit_queue_depth = float(defaults.get('avg_exit_queue_depth') or BOOTSTRAP_QUEUE_DEPTH[asset])
    default_exit_family = str(defaults.get('default_exit_family') or 'TIME_EXIT')
    default_sell_quote_policy = str(defaults.get('default_sell_quote_policy') or 'TIMEOUT_THEN_HIT_BID')
    avg_trade_pnl = float(hist.get('avg_trade_pnl') or 0.0)
    rows: list[dict[str, Any]] = []

    for idx in range(full_n):
        rows.append({
            'asset': asset,
            'symbol': ASSET_SYMBOL[asset],
            'trade_id': f'legacy_{asset}_filled_{idx}',
            'market_slug': 'legacy_bootstrap',
            'decision_ts': None,
            'status': 'executed',
            'result': 'win' if avg_trade_pnl > 0 else 'lose' if avg_trade_pnl < 0 else '',
            'direction': '',
            'take_trade': True,
            'confidence': 0.60,
            'entry_limit_price': 0.0,
            'entry_fill_mode': 'legacy_bootstrap',
            'exit_fill_mode': 'legacy_bootstrap',
            'fill_status': 'filled',
            'partial_fill_ratio': 1.0,
            'queue_wait_seconds': avg_queue_wait_seconds,
            'entry_queue_depth': avg_entry_queue_depth,
            'exit_queue_depth': avg_exit_queue_depth,
            'selected_exit_family': default_exit_family,
            'selected_sell_quote_policy': default_sell_quote_policy,
            'baseline_hold_pnl': avg_trade_pnl,
            'overlay_pnl': avg_trade_pnl,
            'entry_best_ask': None,
            'entry_best_bid': None,
            'exit_best_ask': None,
            'exit_best_bid': None,
            'avg_spread': 0.02,
            'snapshot_count': 0,
            'execution_calibration': 'legacy_bootstrap',
            'source': 'legacy_exp',
        })
    for idx in range(partial_n):
        partial_pnl = avg_trade_pnl * max(avg_partial_fill_ratio, 0.25)
        rows.append({
            'asset': asset,
            'symbol': ASSET_SYMBOL[asset],
            'trade_id': f'legacy_{asset}_partial_{idx}',
            'market_slug': 'legacy_bootstrap',
            'decision_ts': None,
            'status': 'executed',
            'result': 'win' if partial_pnl > 0 else 'lose' if partial_pnl < 0 else '',
            'direction': '',
            'take_trade': True,
            'confidence': 0.58,
            'entry_limit_price': 0.0,
            'entry_fill_mode': 'legacy_bootstrap',
            'exit_fill_mode': 'legacy_bootstrap',
            'fill_status': 'partial_fill',
            'partial_fill_ratio': avg_partial_fill_ratio,
            'queue_wait_seconds': avg_queue_wait_seconds,
            'entry_queue_depth': avg_entry_queue_depth,
            'exit_queue_depth': avg_exit_queue_depth,
            'selected_exit_family': default_exit_family,
            'selected_sell_quote_policy': default_sell_quote_policy,
            'baseline_hold_pnl': partial_pnl,
            'overlay_pnl': partial_pnl,
            'entry_best_ask': None,
            'entry_best_bid': None,
            'exit_best_ask': None,
            'exit_best_bid': None,
            'avg_spread': 0.02,
            'snapshot_count': 0,
            'execution_calibration': 'legacy_bootstrap',
            'source': 'legacy_exp',
        })
    for idx in range(timeout_n):
        rows.append({
            'asset': asset,
            'symbol': ASSET_SYMBOL[asset],
            'trade_id': f'legacy_{asset}_timeout_{idx}',
            'market_slug': 'legacy_bootstrap',
            'decision_ts': None,
            'status': 'timeout_unfilled',
            'result': 'lose',
            'direction': '',
            'take_trade': True,
            'confidence': 0.55,
            'entry_limit_price': 0.0,
            'entry_fill_mode': 'legacy_bootstrap',
            'exit_fill_mode': 'legacy_bootstrap',
            'fill_status': 'timeout_unfilled',
            'partial_fill_ratio': 0.0,
            'queue_wait_seconds': avg_queue_wait_seconds,
            'entry_queue_depth': avg_entry_queue_depth,
            'exit_queue_depth': avg_exit_queue_depth,
            'selected_exit_family': default_exit_family,
            'selected_sell_quote_policy': default_sell_quote_policy,
            'baseline_hold_pnl': -BOOTSTRAP_NO_FILL_PENALTY,
            'overlay_pnl': -BOOTSTRAP_NO_FILL_PENALTY,
            'entry_best_ask': None,
            'entry_best_bid': None,
            'exit_best_ask': None,
            'exit_best_bid': None,
            'avg_spread': 0.02,
            'snapshot_count': 0,
            'execution_calibration': 'legacy_bootstrap',
            'source': 'legacy_exp',
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df['fill_success'] = (df['fill_status'] == 'filled').astype(int)
        df['partial_fill_flag'] = (df['fill_status'] == 'partial_fill').astype(int)
        df['timeout_flag'] = (df['fill_status'] == 'timeout_unfilled').astype(int)
        df['win_flag'] = df['overlay_pnl'].astype(float) > 0
    return df


def _latest_lifecycle_by_trade_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        trade_id = str(row.get('trade_id') or '').strip()
        if not trade_id:
            continue
        ts = str(row.get('ts') or row.get('generated_at') or '')
        prev = out.get(trade_id)
        if prev is None or ts >= str(prev.get('ts') or prev.get('generated_at') or ''):
            out[trade_id] = row
    return out


def _summarize_orderbook_rows(rows: list[dict[str, Any]], trade_id: str, market_slug: str) -> dict[str, Any]:
    matched = [
        row for row in rows
        if str(row.get('trade_id') or '') == trade_id or str(row.get('market_slug') or '') == market_slug
    ]
    if not matched:
        return {
            'snapshot_count': 0,
            'entry_best_ask': None,
            'entry_best_bid': None,
            'exit_best_ask': None,
            'exit_best_bid': None,
            'avg_spread': None,
            'avg_bid_depth_top3': None,
            'avg_ask_depth_top3': None,
        }
    matched = sorted(matched, key=lambda r: str(r.get('ts') or r.get('generated_at') or ''))
    spreads = [float(r.get('spread') or 0.0) for r in matched if r.get('spread') is not None]
    bid_depths = [float(r.get('bid_depth_top3') or 0.0) for r in matched if r.get('bid_depth_top3') is not None]
    ask_depths = [float(r.get('ask_depth_top3') or 0.0) for r in matched if r.get('ask_depth_top3') is not None]
    first = matched[0]
    last = matched[-1]
    return {
        'snapshot_count': int(len(matched)),
        'entry_best_ask': _round_float(first.get('best_ask'), 6),
        'entry_best_bid': _round_float(first.get('best_bid'), 6),
        'exit_best_ask': _round_float(last.get('best_ask'), 6),
        'exit_best_bid': _round_float(last.get('best_bid'), 6),
        'avg_spread': _round_float(_safe_mean(spreads, 0.0), 6),
        'avg_bid_depth_top3': _round_float(_safe_mean(bid_depths, 0.0), 6),
        'avg_ask_depth_top3': _round_float(_safe_mean(ask_depths, 0.0), 6),
    }


def _build_execution_dataset_for_asset(asset: str) -> pd.DataFrame:
    overlay_rows = load_overlay_rows(asset)
    orderbook_rows, lifecycle_rows = _load_observation_rows(asset)
    lifecycle_by_trade = _latest_lifecycle_by_trade_id(lifecycle_rows)
    records: list[dict[str, Any]] = []
    for row in overlay_rows:
        if str(row.get('status') or '').lower() != 'executed':
            continue
        if str(row.get('result') or '').lower() not in {'win', 'lose'}:
            continue
        if str(row.get('exit_reason') or '').lower().startswith('fallback_hold_missing_path'):
            continue
        overlay_pnl = row.get('overlay_pnl') if row.get('overlay_pnl') is not None else row.get('pnl')
        if overlay_pnl is None:
            continue
        trade_id = str(row.get('id') or '')
        market_slug = str(row.get('marketSlug') or '')
        symbol = str(row.get('symbol') or ASSET_SYMBOL[asset]).upper()
        lifecycle = lifecycle_by_trade.get(trade_id, {})
        book = _summarize_orderbook_rows(orderbook_rows, trade_id, market_slug)
        fill_status = normalize_fill_status(lifecycle.get('fill_status'))
        if fill_status == 'unknown':
            fill_status = 'filled' if str(row.get('status') or '').lower() == 'executed' else 'queued'
        partial_fill_ratio = lifecycle.get('partial_fill_ratio')
        if partial_fill_ratio is None:
            partial_fill_ratio = 1.0 if fill_status == 'filled' else 0.5 if fill_status == 'partial_fill' else 0.0
        queue_wait_seconds = lifecycle.get('queue_wait_seconds')
        if queue_wait_seconds is None:
            queue_wait_seconds = 0.0
        record = {
            'asset': asset,
            'symbol': symbol,
            'trade_id': trade_id,
            'market_slug': market_slug,
            'decision_ts': row.get('decision_ts') or row.get('timestamp'),
            'status': str(row.get('status') or ''),
            'result': str(row.get('result') or ''),
            'direction': str(row.get('direction') or ''),
            'take_trade': bool(row.get('take_trade', True)),
            'confidence': float(row.get('confidence') or 0.0),
            'entry_limit_price': float(row.get('entry_limit_price') or row.get('limitPriceConfigured') or 0.0),
            'entry_fill_mode': str(lifecycle.get('entry_fill_mode') or _default_entry_fill_mode(row)),
            'exit_fill_mode': str(lifecycle.get('exit_fill_mode') or 'overlay_proxy'),
            'fill_status': fill_status,
            'partial_fill_ratio': float(partial_fill_ratio),
            'queue_wait_seconds': float(queue_wait_seconds),
            'entry_queue_depth': float(lifecycle.get('entry_queue_depth') or book.get('avg_ask_depth_top3') or BOOTSTRAP_QUEUE_DEPTH[asset]),
            'exit_queue_depth': float(lifecycle.get('exit_queue_depth') or book.get('avg_bid_depth_top3') or BOOTSTRAP_QUEUE_DEPTH[asset]),
            'selected_exit_family': str(row.get('selected_exit_family') or row.get('exit_family') or 'TIME_EXIT'),
            'selected_sell_quote_policy': str(row.get('selected_sell_quote_policy') or row.get('sell_quote_policy') or 'TIMEOUT_THEN_HIT_BID'),
            'baseline_hold_pnl': float(row.get('baseline_hold_pnl') or 0.0),
            'overlay_pnl': float(overlay_pnl or 0.0),
            'entry_best_ask': book.get('entry_best_ask'),
            'entry_best_bid': book.get('entry_best_bid'),
            'exit_best_ask': book.get('exit_best_ask'),
            'exit_best_bid': book.get('exit_best_bid'),
            'avg_spread': book.get('avg_spread'),
            'snapshot_count': int(book.get('snapshot_count') or 0),
            'execution_calibration': 'observed' if int(book.get('snapshot_count') or 0) > 0 or lifecycle else 'bootstrap',
            'source': 'current_overlay',
        }
        records.append(record)
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df['fill_status'] = df['fill_status'].map(normalize_fill_status)
    df['fill_success'] = (df['fill_status'] == 'filled').astype(int)
    df['partial_fill_flag'] = (df['fill_status'] == 'partial_fill').astype(int)
    df['timeout_flag'] = (df['fill_status'] == 'timeout_unfilled').astype(int)
    df['win_flag'] = df['overlay_pnl'].astype(float) > 0
    return df


def build_execution_datasets() -> dict[str, Any]:
    readiness = _readiness_with_historical_bootstrap()
    historical = load_json(HISTORICAL_EXECUTION_BOOTSTRAP_REPORT)
    reset_payload = load_latest_reset_report()
    assets_payload: dict[str, Any] = {}
    for asset in ASSETS:
        current_df = _build_execution_dataset_for_asset(asset)
        target_legacy_rows = max(0, 120 - int(len(current_df)))
        legacy_df = _build_legacy_seed_dataset_for_asset(asset, historical, target_rows=target_legacy_rows)
        frames = [df for df in (current_df, legacy_df) if not df.empty]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not df.empty:
            EXECUTION_LABEL_PATHS[asset].parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(EXECUTION_LABEL_PATHS[asset], index=False)
        assets_payload[asset] = {
            'asset': asset,
            'rows': int(len(df)),
            'current_rows': int(len(current_df)),
            'legacy_seed_rows': int(len(legacy_df)),
            'execution_label_path': str(EXECUTION_LABEL_PATHS[asset]),
            'fill_status_counts': {str(k): int(v) for k, v in (df['fill_status'].value_counts().to_dict() if not df.empty else {}).items()},
            'avg_overlay_pnl': _round_float(float(df['overlay_pnl'].mean()) if not df.empty else 0.0, 6),
        }
    payload = {
        'generated_at': utc_now_iso(),
        'scope': 'vnext_execution_v2_dataset',
        'reset_generated_at': reset_payload.get('generated_at'),
        'readiness': readiness,
        'assets': assets_payload,
    }
    write_json(EXECUTION_REPORT, payload)
    return payload


def _execution_lookup_key(exit_family: str, sell_quote_policy: str, avg_spread: float | None) -> str:
    spread_bucket = _spread_bucket_from_value(float(avg_spread or 0.0))
    return '|'.join([str(exit_family), str(sell_quote_policy), spread_bucket])


def _train_execution_for_asset(asset: str) -> dict[str, Any]:
    df = pd.read_parquet(EXECUTION_LABEL_PATHS[asset]) if EXECUTION_LABEL_PATHS[asset].exists() else pd.DataFrame()
    model_dir = V2A_MODEL_DIRS[asset]
    model_dir.mkdir(parents=True, exist_ok=True)
    bootstrap = default_execution_bootstrap(asset)
    historical = _load_or_build_historical_execution_bootstrap()
    hist_asset = ((historical.get('assets') or {}).get(asset) if isinstance(historical, dict) else None) or {}
    hist_defaults = hist_asset.get('bootstrap_defaults') if isinstance(hist_asset.get('bootstrap_defaults'), dict) else {}
    if hist_defaults:
        bootstrap = {
            **bootstrap,
            'execution_calibration': 'historical_bootstrap',
            'defaults': {
                **bootstrap['defaults'],
                **{k: v for k, v in hist_defaults.items() if v is not None},
            },
        }
    if not df.empty and 'source' not in df.columns:
        df['source'] = 'current_overlay'
    current_df = df[df['source'] != 'legacy_exp'].copy() if not df.empty else pd.DataFrame()
    legacy_df = df[df['source'] == 'legacy_exp'].copy() if not df.empty else pd.DataFrame()
    current_fill_samples = int(
        current_df['fill_status'].isin(['filled', 'partial_fill', 'timeout_unfilled']).sum()
    ) if not current_df.empty else 0
    current_snapshots = int(current_df['snapshot_count'].sum()) if not current_df.empty else 0
    observed_ready = (
        int(len(current_df)) >= MIN_TRADES
        and current_snapshots >= MIN_SNAPSHOTS
        and current_fill_samples >= MIN_FILL_SAMPLES
    )
    if df.empty:
        execution_calibration = {
            **bootstrap,
            'status': 'bootstrap_no_rows',
            'current_cycle_only': True,
            'sample_counts': {
                'rows': 0,
                'current_rows': 0,
                'legacy_seed_rows': 0,
                'snapshots': 0,
                'current_snapshots': 0,
                'fill_samples': 0,
                'current_fill_samples': 0,
                'historical_trade_rows': int(hist_asset.get('legacy_trade_rows') or 0),
                'historical_settled_trades': int(hist_asset.get('legacy_settled_trades') or 0),
            },
        }
        fill_policy_lookup = {
            'generated_at': utc_now_iso(),
            'asset': asset,
            'status': 'bootstrap_no_rows',
            'current_cycle_only': True,
            'default': bootstrap['defaults'],
            'rows': {},
        }
        queue_policy_lookup = {
            'generated_at': utc_now_iso(),
            'asset': asset,
            'status': 'bootstrap_no_rows',
            'current_cycle_only': True,
            'default': {
                'avg_queue_wait_seconds': bootstrap['defaults']['avg_queue_wait_seconds'],
                'avg_entry_queue_depth': BOOTSTRAP_QUEUE_DEPTH[asset],
                'avg_exit_queue_depth': BOOTSTRAP_QUEUE_DEPTH[asset],
            },
            'rows': {},
        }
    elif not observed_ready:
        execution_calibration = {
            **bootstrap,
            'status': 'bootstrap_warmup',
            'current_cycle_only': True,
            'sample_counts': {
                'rows': int(len(df)),
                'current_rows': int(len(current_df)),
                'legacy_seed_rows': int(len(legacy_df)),
                'snapshots': int(df['snapshot_count'].sum()) if not df.empty else 0,
                'current_snapshots': current_snapshots,
                'fill_samples': int(df['fill_status'].isin(['filled', 'partial_fill', 'timeout_unfilled']).sum()) if not df.empty else 0,
                'current_fill_samples': current_fill_samples,
                'historical_trade_rows': int(hist_asset.get('legacy_trade_rows') or 0),
                'historical_settled_trades': int(hist_asset.get('legacy_settled_trades') or 0),
            },
        }
        fill_policy_lookup = {
            'generated_at': utc_now_iso(),
            'asset': asset,
            'status': 'bootstrap_warmup',
            'current_cycle_only': True,
            'default': bootstrap['defaults'],
            'rows': {},
        }
        queue_policy_lookup = {
            'generated_at': utc_now_iso(),
            'asset': asset,
            'status': 'bootstrap_warmup',
            'current_cycle_only': True,
            'default': {
                'avg_queue_wait_seconds': bootstrap['defaults']['avg_queue_wait_seconds'],
                'avg_entry_queue_depth': float(bootstrap['defaults'].get('avg_entry_queue_depth') or BOOTSTRAP_QUEUE_DEPTH[asset]),
                'avg_exit_queue_depth': float(bootstrap['defaults'].get('avg_exit_queue_depth') or BOOTSTRAP_QUEUE_DEPTH[asset]),
            },
            'rows': {},
        }
    else:
        observed_df = current_df.copy()
        fill_rows: dict[str, Any] = {}
        for key, group in observed_df.groupby(observed_df.apply(lambda r: _execution_lookup_key(r['selected_exit_family'], r['selected_sell_quote_policy'], r['avg_spread']), axis=1)):
            any_fill_mask = group['fill_status'].isin(['filled', 'partial_fill'])
            any_fill_rate = float(any_fill_mask.mean())
            partial_fill_rate = (
                float((group.loc[any_fill_mask, 'fill_status'] == 'partial_fill').mean())
                if int(any_fill_mask.sum()) > 0 else 0.0
            )
            timeout_rate = float((group['fill_status'] == 'timeout_unfilled').mean())
            fill_rows[str(key)] = {
                'samples': int(len(group)),
                'fill_rate': _round_float(min(0.99, max(0.10, any_fill_rate)), 6),
                'partial_fill_rate': _round_float(min(0.90, max(0.0, partial_fill_rate)), 6),
                'timeout_rate': _round_float(min(0.50, max(0.0, timeout_rate)), 6),
                'avg_partial_fill_ratio': _round_float(float(group['partial_fill_ratio'].mean()), 6),
                'avg_queue_wait_seconds': _round_float(float(group['queue_wait_seconds'].mean()), 6),
                'avg_entry_queue_depth': _round_float(float(group['entry_queue_depth'].mean()), 6),
                'avg_exit_queue_depth': _round_float(float(group['exit_queue_depth'].mean()), 6),
                'avg_overlay_pnl': _round_float(float(group['overlay_pnl'].mean()), 6),
                'exit_family': str(group['selected_exit_family'].iloc[0]),
                'sell_quote_policy': str(group['selected_sell_quote_policy'].iloc[0]),
                'spread_bucket': str(key).rsplit('|', 1)[-1],
            }
        if fill_rows:
            default_key, default_row = max(fill_rows.items(), key=lambda item: (float(item[1]['avg_overlay_pnl'] or 0.0), item[1]['samples']))
            default_lookup = {
                'fill_rate': default_row['fill_rate'],
                'partial_fill_rate': default_row['partial_fill_rate'],
                'timeout_rate': default_row['timeout_rate'],
                'avg_partial_fill_ratio': default_row['avg_partial_fill_ratio'],
                'avg_queue_wait_seconds': default_row['avg_queue_wait_seconds'],
                'avg_entry_queue_depth': default_row['avg_entry_queue_depth'],
                'avg_exit_queue_depth': default_row['avg_exit_queue_depth'],
                'default_exit_family': default_row['exit_family'],
                'default_sell_quote_policy': default_row['sell_quote_policy'],
                'default_key': default_key,
            }
        else:
            default_lookup = bootstrap['defaults']
        execution_calibration = {
            'generated_at': utc_now_iso(),
            'asset': asset,
            'status': 'trained',
            'execution_calibration': 'observed',
            'current_cycle_only': True,
            'sample_counts': {
                'rows': int(len(df)),
                'current_rows': int(len(current_df)),
                'legacy_seed_rows': int(len(legacy_df)),
                'snapshots': int(df['snapshot_count'].sum()),
                'current_snapshots': current_snapshots,
                'fill_samples': int(((df['fill_status'].isin(['filled', 'partial_fill', 'timeout_unfilled'])).sum())),
                'current_fill_samples': current_fill_samples,
                'historical_trade_rows': int(hist_asset.get('legacy_trade_rows') or 0),
                'historical_settled_trades': int(hist_asset.get('legacy_settled_trades') or 0),
            },
            'defaults': default_lookup,
        }
        fill_policy_lookup = {
            'generated_at': utc_now_iso(),
            'asset': asset,
            'status': 'trained',
            'current_cycle_only': True,
            'default': default_lookup,
            'rows': fill_rows,
        }
        queue_policy_lookup = {
            'generated_at': utc_now_iso(),
            'asset': asset,
            'status': 'trained',
            'current_cycle_only': True,
            'default': {
                'avg_queue_wait_seconds': default_lookup['avg_queue_wait_seconds'],
                'avg_entry_queue_depth': default_lookup['avg_entry_queue_depth'],
                'avg_exit_queue_depth': default_lookup['avg_exit_queue_depth'],
            },
            'rows': {
                key: {
                    'avg_queue_wait_seconds': row['avg_queue_wait_seconds'],
                    'avg_entry_queue_depth': row['avg_entry_queue_depth'],
                    'avg_exit_queue_depth': row['avg_exit_queue_depth'],
                    'timeout_rate': row['timeout_rate'],
                    'samples': row['samples'],
                }
                for key, row in fill_rows.items()
            },
        }
    write_json(model_dir / 'execution_calibration.json', execution_calibration)
    write_json(model_dir / 'fill_policy_lookup.json', fill_policy_lookup)
    write_json(model_dir / 'queue_policy_lookup.json', queue_policy_lookup)
    write_json(model_dir / 'config.json', {
        'generated_at': utc_now_iso(),
        'asset': asset,
        'scope': 'vnext_execution_v2_prepare',
        'stage': 'v2a_execution',
        'status': execution_calibration['status'],
        'depends_on_v1_model_dir': str(V1_MODEL_DIRS[asset]),
        'execution_label_path': str(EXECUTION_LABEL_PATHS[asset]),
        'notes': ['manual_activation_only', 'queue_aware_exit_prepared', 'bootstrap_allowed_when_samples_sparse'],
    })
    return {
        'asset': asset,
        'model_dir': str(model_dir),
        'execution_calibration': execution_calibration,
        'fill_policy_lookup_rows': len(fill_policy_lookup.get('rows') or {}),
    }


def train_execution() -> dict[str, Any]:
    readiness = _readiness_with_historical_bootstrap()
    reset_payload = load_latest_reset_report()
    assets_payload = {asset: _train_execution_for_asset(asset) for asset in ASSETS}
    payload = {
        'generated_at': utc_now_iso(),
        'scope': 'vnext_execution_v2_training',
        'reset_generated_at': reset_payload.get('generated_at'),
        'readiness': readiness,
        'assets': assets_payload,
    }
    write_json(EXECUTION_REPORT, payload)
    return payload


def _load_execution_profile(asset: str, exit_family: str, sell_quote_policy: str, spread_metric: float) -> ExecutionProfile:
    lookup = load_json(V2A_MODEL_DIRS[asset] / 'fill_policy_lookup.json') or {}
    rows = lookup.get('rows') if isinstance(lookup, dict) else {}
    default = (lookup.get('default') if isinstance(lookup, dict) else None) or default_execution_bootstrap(asset)['defaults']
    key = _execution_lookup_key(exit_family, sell_quote_policy, spread_metric)
    selected = rows.get(key) if isinstance(rows, dict) else None
    if not isinstance(selected, dict):
        selected = {
            'fill_rate': default.get('fill_rate', 0.92),
            'partial_fill_rate': default.get('partial_fill_rate', 0.08),
            'timeout_rate': default.get('timeout_rate', 0.05),
            'avg_partial_fill_ratio': default.get('avg_partial_fill_ratio', 0.48),
            'avg_queue_wait_seconds': default.get('avg_queue_wait_seconds', 22.0),
            'avg_entry_queue_depth': default.get('avg_entry_queue_depth', BOOTSTRAP_QUEUE_DEPTH[asset]),
            'avg_exit_queue_depth': default.get('avg_exit_queue_depth', BOOTSTRAP_QUEUE_DEPTH[asset]),
            'exit_family': exit_family,
            'sell_quote_policy': sell_quote_policy,
        }
        source = 'default'
    else:
        source = 'lookup'
    return ExecutionProfile(
        fill_rate=float(selected.get('fill_rate') or 0.92),
        partial_fill_rate=float(selected.get('partial_fill_rate') or 0.08),
        timeout_rate=float(selected.get('timeout_rate') or 0.05),
        avg_partial_fill_ratio=float(selected.get('avg_partial_fill_ratio') or 0.48),
        avg_queue_wait_seconds=float(selected.get('avg_queue_wait_seconds') or 22.0),
        avg_entry_queue_depth=float(selected.get('avg_entry_queue_depth') or BOOTSTRAP_QUEUE_DEPTH[asset]),
        avg_exit_queue_depth=float(selected.get('avg_exit_queue_depth') or BOOTSTRAP_QUEUE_DEPTH[asset]),
        exit_family=str(selected.get('exit_family') or exit_family),
        sell_quote_policy=str(selected.get('sell_quote_policy') or sell_quote_policy),
        source=source,
    )


def _execution_adjusted_utility(
    asset: str,
    pnl: float,
    mae: float,
    confidence: float,
    trend_bucket: str,
    exit_family: str,
    sell_quote_policy: str,
    spread_metric: float,
    objective_cfg: dict[str, float],
) -> tuple[float, float, ExecutionProfile]:
    if exit_family == 'HOLD_TO_EXPIRY':
        profile = ExecutionProfile(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, exit_family, sell_quote_policy, 'hold_to_expiry')
        adjusted_pnl = float(pnl)
        utility = _utility_adjusted(adjusted_pnl, trend_bucket, exit_family, sell_quote_policy, mae, objective_cfg)
        return adjusted_pnl, utility, profile
    profile = _load_execution_profile(asset, exit_family, sell_quote_policy, spread_metric)
    realized_fill_ratio = max(0.0, min(1.0, profile.fill_rate))
    if realized_fill_ratio <= 0:
        adjusted_pnl = -BOOTSTRAP_NO_FILL_PENALTY
    else:
        partial_ratio = (1.0 - profile.partial_fill_rate) + profile.partial_fill_rate * profile.avg_partial_fill_ratio
        adjusted_pnl = float(pnl) * realized_fill_ratio * partial_ratio
    utility = _utility_adjusted(adjusted_pnl, trend_bucket, exit_family, sell_quote_policy, mae, objective_cfg)
    utility -= BOOTSTRAP_TIMEOUT_PENALTY * profile.timeout_rate
    utility -= BOOTSTRAP_NO_FILL_PENALTY * max(0.0, 1.0 - profile.fill_rate)
    utility -= BOOTSTRAP_QUEUE_WAIT_PENALTY_PER_SEC * profile.avg_queue_wait_seconds
    if adjusted_pnl <= 0 and confidence >= 0.60:
        utility -= BOOTSTRAP_HIGH_CONF_LOSS_PENALTY
    return adjusted_pnl, float(utility), profile


def _objective_cfg_v2() -> dict[str, float]:
    return {
        'trend_bonus': 1.20,
        'countertrend_penalty': 0.75,
        'abstain_cost_bias': 0.02,
        'drawdown_penalty': 0.25,
        'sell_aggressiveness_bias': 0.10,
        'take_profit_preference': 0.05,
    }


def _relabel_row_v2(asset: str, row: pd.Series, objective_cfg: dict[str, float]) -> dict[str, Any] | None:
    t_sec_list = json.loads(str(row['path_t_sec_json']))
    up_path = json.loads(str(row['path_up_json']))
    market_start_ts = int(row['target_start_ts'])
    prediction_ts = int(row.get('prediction_ts') or pd.Timestamp(row['timestamp']).timestamp())
    market_slug = str(row.get('market_slug') or f"{'btc' if asset == 'BTC_USDT' else 'eth'}-updown-15m-{market_start_ts}")
    actual_up = bool(int(row['actual_up']) == 1) if 'actual_up' in row.index else bool(float(row.get('next_return', 0.0)) > 0.0)
    vol_metric = float(row.get('vol_metric') or _vol_metric_from_row(row))
    spread_metric = float(row.get('spread_metric') or _spread_metric_from_row(row))
    trend_score = float(row.get('trend_score') or _trend_score_from_row(row))
    action_utility_map: dict[str, float] = {}
    action_pnl_map: dict[str, float] = {}
    action_meta_map: dict[str, dict[str, Any]] = {}
    best: dict[str, Any] | None = None
    for direction in ('UP', 'DOWN'):
        direction_prices = up_path if direction == 'UP' else [float(np.clip(1.0 - p, 0.0, 1.0)) for p in up_path]
        won = actual_up if direction == 'UP' else (not actual_up)
        trend_bucket = _direction_trend_bucket(direction, trend_score)
        for buy_price in BUY_GRIDS[asset]:
            action_key = f'{direction}|{buy_price:.2f}'
            entry_trace = replay_entry_trace(
                prediction_ts=prediction_ts,
                market_start_ts=market_start_ts,
                limit_price=float(buy_price),
                t_sec_list=t_sec_list,
                direction_prices=direction_prices,
                market_slug=market_slug,
            )
            if entry_trace.fill_status != 'filled' or entry_trace.fill_price is None or entry_trace.fill_ts is None:
                action_utility_map[action_key] = 0.0
                action_pnl_map[action_key] = 0.0
                action_meta_map[action_key] = {
                    'buy_label': f'{buy_price:.2f}',
                    'best_exit_family': 'HOLD_TO_EXPIRY',
                    'best_sell_quote_policy': 'TIMEOUT_THEN_HIT_BID',
                    'entry_fill_proxy': entry_trace.first_observed_price,
                    'entry_proxy_ts': entry_trace.fill_ts or entry_trace.market_discovered_ts or prediction_ts,
                    'avg_hold_seconds': 0,
                    'profile_source': 'unfilled_entry',
                    'entry_trace': entry_trace.to_dict(),
                }
                continue
            local_best = None
            for family in EXIT_FAMILIES:
                for policy in SELL_QUOTE_POLICIES:
                    spread_proxy = _spread_bucket_to_proxy(asset, _spread_bucket_from_value(spread_metric))
                    pnl, exit_price, exit_ts, exit_reason, mae, _mfe = _simulate_family_policy(
                        amount=1.0,
                        entry_price=float(entry_trace.fill_price),
                        entry_ts=int(entry_trace.fill_ts),
                        market_start_ts=market_start_ts,
                        won=won,
                        t_sec_list=t_sec_list,
                        direction_prices=direction_prices,
                        exit_family=family,
                        sell_quote_policy=policy,
                        spread_proxy=spread_proxy,
                    )
                    confidence = max(float(row.get('confidence') or 0.60), 0.50)
                    adjusted_pnl, adjusted_utility, profile = _execution_adjusted_utility(
                        asset=asset,
                        pnl=float(pnl),
                        mae=float(mae),
                        confidence=confidence,
                        trend_bucket=trend_bucket,
                        exit_family=family,
                        sell_quote_policy=policy,
                        spread_metric=spread_metric,
                        objective_cfg=objective_cfg,
                    )
                    candidate = {
                        'direction': direction,
                        'buy_label': f'{buy_price:.2f}',
                        'best_exit_family': family,
                        'best_sell_quote_policy': policy,
                        'best_utility': adjusted_utility,
                        'execution_net_pnl': adjusted_pnl,
                        'baseline_hold_pnl': float(pnl),
                        'entry_fill_proxy': float(entry_trace.fill_price),
                        'entry_proxy_ts': int(entry_trace.fill_ts),
                        'avg_hold_seconds': int(exit_ts - int(entry_trace.fill_ts)),
                        'exit_reason': exit_reason,
                        'profile_source': profile.source,
                        'entry_trace': entry_trace.to_dict(),
                    }
                    if local_best is None or float(candidate['best_utility']) > float(local_best['best_utility']):
                        local_best = candidate
            if local_best is None:
                continue
            action_utility_map[action_key] = float(local_best['best_utility'])
            action_pnl_map[action_key] = float(local_best['execution_net_pnl'])
            action_meta_map[action_key] = {
                'buy_label': local_best['buy_label'],
                'best_exit_family': local_best['best_exit_family'],
                'best_sell_quote_policy': local_best['best_sell_quote_policy'],
                'entry_fill_proxy': local_best['entry_fill_proxy'],
                'entry_proxy_ts': local_best['entry_proxy_ts'],
                'avg_hold_seconds': local_best['avg_hold_seconds'],
                'profile_source': local_best['profile_source'],
                'entry_trace': local_best['entry_trace'],
            }
            if best is None or float(local_best['best_utility']) > float(best['best_utility']):
                best = local_best
    if best is None:
        return None
    take_label = int(float(best['best_utility']) > float(objective_cfg['abstain_cost_bias']))
    direction_target = 1 if str(best['direction']) == 'UP' else 0
    return {
        'gate_label': take_label,
        'direction_target': direction_target,
        'buy_label': best['buy_label'] if take_label else 'ABSTAIN',
        'best_exit_family': best['best_exit_family'] if take_label else 'HOLD_TO_EXPIRY',
        'best_sell_quote_policy': best['best_sell_quote_policy'] if take_label else 'TIMEOUT_THEN_HIT_BID',
        'best_utility': float(best['best_utility']),
        'execution_net_pnl': float(best['execution_net_pnl']),
        'entry_proxy_ts': int(best['entry_proxy_ts']),
        'entry_fill_proxy': float(best['entry_fill_proxy']),
        'action_utility_map_json': json.dumps({k: round(v, 6) for k, v in action_utility_map.items()}, separators=(',', ':')),
        'action_pnl_map_json': json.dumps({k: round(v, 6) for k, v in action_pnl_map.items()}, separators=(',', ':')),
        'action_meta_map_json': json.dumps(action_meta_map, separators=(',', ':')),
        'vol_metric': vol_metric,
        'spread_metric': spread_metric,
        'trend_score': trend_score,
    }


def build_profit_relabel() -> dict[str, Any]:
    objective_cfg = _objective_cfg_v2()
    reset_payload = load_latest_reset_report()
    assets_payload: dict[str, Any] = {}
    write_json(PROFIT_RELABEL_REPORT, {
        'generated_at': utc_now_iso(),
        'scope': 'vnext_profit_relabel_v2',
        'status': 'running',
        'reset_generated_at': reset_payload.get('generated_at'),
        'assets': assets_payload,
        'objective_cfg': objective_cfg,
    })
    for asset in ASSETS:
        src = pd.read_parquet(V1_EPISODE_PATHS[asset])
        rows: list[dict[str, Any]] = []
        for _, row in src.iterrows():
            relabeled = _relabel_row_v2(asset, row, objective_cfg)
            if not relabeled:
                continue
            record = row.to_dict()
            record.update(relabeled)
            record['sample_weight'] = float(record.get('sample_weight') or 1.0) * float(np.clip(1.0 + float(relabeled['best_utility']), 0.5, 2.0))
            rows.append(record)
        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError(f'no relabeled rows built for {asset}')
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        RELABELED_EPISODE_PATHS[asset].parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(RELABELED_EPISODE_PATHS[asset], index=False)
        assets_payload[asset] = {
            'asset': asset,
            'rows': int(len(df)),
            'path': str(RELABELED_EPISODE_PATHS[asset]),
            'avg_best_utility': _round_float(float(df['best_utility'].mean()), 6),
            'avg_execution_net_pnl': _round_float(float(df['execution_net_pnl'].mean()), 6),
            'take_rate': _round_float(float(df['gate_label'].mean()), 6),
        }
        write_json(PROFIT_RELABEL_REPORT, {
            'generated_at': utc_now_iso(),
            'scope': 'vnext_profit_relabel_v2',
            'status': 'running',
            'reset_generated_at': reset_payload.get('generated_at'),
            'assets': assets_payload,
            'objective_cfg': objective_cfg,
        })
    payload = {
        'generated_at': utc_now_iso(),
        'scope': 'vnext_profit_relabel_v2',
        'status': 'completed',
        'reset_generated_at': reset_payload.get('generated_at'),
        'assets': assets_payload,
        'objective_cfg': objective_cfg,
    }
    write_json(PROFIT_RELABEL_REPORT, payload)
    return payload


def _load_v1_feature_cols(asset: str) -> list[str]:
    path = V1_MODEL_DIRS[asset] / 'feature_cols.json'
    payload = load_json(path)
    if not isinstance(payload, list):
        raise RuntimeError(f'missing v1 feature_cols for {asset}: {path}')
    cols = [str(col) for col in payload if isinstance(col, str)]
    illegal = find_illegal_feature_cols(cols)
    if illegal:
        raise RuntimeError(f'illegal v1 feature_cols for {asset}: {illegal}')
    return cols


def _load_v1_buy_actions(asset: str) -> list[str]:
    cfg = load_json(V1_MODEL_DIRS[asset] / 'config.json')
    payload = cfg.get('buy_actions') if isinstance(cfg, dict) else None
    if not isinstance(payload, list):
        raise RuntimeError(f'missing v1 buy_actions for {asset}')
    return [str(v) for v in payload]


def _candidate_score_v2(metrics: dict[str, Any]) -> float:
    trades = int(metrics.get('trades') or 0)
    total_pnl = float(metrics.get('total_pnl') or 0.0)
    abstain_rate = float(metrics.get('abstain_rate') or 1.0)
    if total_pnl <= 0 or trades < MIN_TRADES or abstain_rate >= 0.70:
        return -1e9
    return (
        1.0 * total_pnl
        - 0.35 * float(metrics.get('max_drawdown') or 0.0)
        + 0.25 * float(metrics.get('avg_expectancy') or 0.0)
        - 0.20 * float(metrics.get('high_conf_loss_rate') or 0.0)
        + 0.10 * float(metrics.get('win_rate') or 0.0)
    )


def _alignment_audit_ok(audit: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
    assets = (audit or {}).get('assets') or {}
    if not isinstance(assets, dict) or not assets:
        return False, {'reason': 'missing_or_waiting_alignment_audit'}
    asset_rows: dict[str, Any] = {}
    ok_assets = 0
    total_assets = 0
    for asset, row in assets.items():
        if not isinstance(row, dict):
            continue
        total_assets += 1
        status = str(row.get('status') or '')
        match_rate = float(row.get('fill_status_match_rate') or 0.0)
        fill_err = row.get('avg_abs_fill_price_error')
        fill_err_ok = fill_err is None or float(fill_err) <= 0.08
        asset_ok = status == 'ok' and match_rate >= 0.5 and fill_err_ok
        if asset_ok:
            ok_assets += 1
        asset_rows[str(asset)] = {
            'status': status,
            'fill_status_match_rate': _round_float(match_rate, 6),
            'avg_abs_fill_price_error': _round_float(fill_err, 6) if fill_err is not None else None,
            'ready': asset_ok,
        }
    return ok_assets == total_assets and total_assets > 0, {
        'assets': asset_rows,
        'ok_assets': ok_assets,
        'total_assets': total_assets,
    }


def _historical_bootstrap_ready(row: dict[str, Any]) -> tuple[bool, dict[str, int]]:
    hist = row.get('historical_bootstrap') if isinstance(row, dict) else None
    if not isinstance(hist, dict):
        return False, {
            'legacy_settled_trades': 0,
            'legacy_fill_events': 0,
        }
    settled = int(hist.get('legacy_settled_trades') or 0)
    fill_events = int(hist.get('partial_fill_events') or 0) + int(hist.get('full_fill_events') or 0)
    ready = settled >= MIN_HIST_SETTLED_TRADES and fill_events >= MIN_HIST_FILL_EVENTS
    return ready, {
        'legacy_settled_trades': settled,
        'legacy_fill_events': fill_events,
    }


def _simulate_profit_eval(rows: pd.DataFrame, dir_prob: np.ndarray, gate_prob: np.ndarray, buy_labels: np.ndarray, take_threshold: float) -> dict[str, Any]:
    rows = rows.sort_values('target_start_ts').reset_index(drop=True)
    conf = np.maximum(dir_prob, 1.0 - dir_prob)
    total_pnl = 0.0
    trades = 0
    wins = 0
    losses = 0
    abstained = 0
    pnls: list[float] = []
    hold_secs: list[int] = []
    buy_prices: list[float] = []
    chosen_dirs: list[str] = []
    high_conf_losses = 0
    exit_reason_counts: dict[str, int] = {}
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for i, row in rows.iterrows():
        if float(gate_prob[i]) < take_threshold:
            abstained += 1
            continue
        buy_label = str(buy_labels[i])
        if buy_label == 'ABSTAIN':
            abstained += 1
            continue
        direction = 'UP' if float(dir_prob[i]) >= 0.5 else 'DOWN'
        key = f'{direction}|{buy_label}'
        utility_map = json.loads(str(row.get('action_utility_map_json') or '{}'))
        pnl_map = json.loads(str(row.get('action_pnl_map_json') or '{}'))
        meta_map = json.loads(str(row.get('action_meta_map_json') or '{}'))
        pnl = float(pnl_map.get(key, 0.0))
        trades += 1
        total_pnl += pnl
        pnls.append(pnl)
        buy_prices.append(float(buy_label))
        chosen_dirs.append(direction)
        hold_secs.append(int((meta_map.get(key) or {}).get('avg_hold_seconds') or 0))
        exit_family = str((meta_map.get(key) or {}).get('best_exit_family') or 'HOLD_TO_EXPIRY')
        exit_policy = str((meta_map.get(key) or {}).get('best_sell_quote_policy') or 'TIMEOUT_THEN_HIT_BID')
        exit_reason_counts[f'{exit_family}:{exit_policy}'] = exit_reason_counts.get(f'{exit_family}:{exit_policy}', 0) + 1
        if pnl > 0:
            wins += 1
        else:
            losses += 1
            if conf[i] >= 0.60:
                high_conf_losses += 1
        cum += pnl
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    streak = 0
    max_streak = 0
    prev_dir = None
    for pnl, direction in zip(pnls, chosen_dirs):
        if pnl <= 0 and direction == prev_dir:
            streak += 1
        elif pnl <= 0:
            streak = 1
        else:
            streak = 0
        prev_dir = direction if pnl <= 0 else None
        max_streak = max(max_streak, streak)
    return {
        'rows': int(len(rows)),
        'trades': int(trades),
        'win_rate': _round_float(float(wins / trades) if trades else 0.0),
        'total_pnl': _round_float(total_pnl, 6),
        'avg_expectancy': _round_float(float(np.mean(pnls)) if pnls else 0.0, 6),
        'max_drawdown': _round_float(max_dd, 6),
        'high_conf_loss_rate': _round_float(float(high_conf_losses / trades) if trades else 0.0, 6),
        'max_same_direction_loss_streak': int(max_streak),
        'abstain_rate': _round_float(float(abstained / len(rows)) if len(rows) else 0.0, 6),
        'avg_buy_price': _round_float(float(np.mean(buy_prices)) if buy_prices else 0.0, 4),
        'avg_exit_hold_seconds': _round_float(float(np.mean(hold_secs)) if hold_secs else 0.0, 2),
        'exit_reason_breakdown': {k: int(v) for k, v in sorted(exit_reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))},
    }


def _evaluate_profit_candidate(df: pd.DataFrame, feature_cols: Sequence[str], params: dict[str, Any], buy_actions: Sequence[str], train_days: int, take_threshold: float) -> dict[str, Any]:
    folds = _iter_purged_splits(df, train_days=train_days)
    if not folds:
        raise RuntimeError('no valid purged folds')
    metrics_list: list[dict[str, Any]] = []
    best_calibration: dict[str, Any] = {'selected': 'none', 'kind': 'none'}
    best_fit: dict[str, Any] | None = None
    for idx, split in enumerate(folds):
        fit = _fit_heads(split, feature_cols, params, buy_actions, lgb_jobs=1)
        cal_meta = _select_calibration_from_split(fit, take_threshold)
        if idx == 0:
            best_calibration = cal_meta
            best_fit = fit
        dir_test = fit['dir_model'].predict_proba(fit['X_te'])[:, 1]
        gate_test = fit['gate_model'].predict_proba(fit['X_te'])[:, 1]
        buy_test = fit['buy_model'].predict(fit['X_te'])
        buy_test_labels = np.asarray([fit['id_to_price'][int(i)] for i in buy_test], dtype=object)
        metrics = _simulate_profit_eval(split.test, _apply_calibration_meta(cal_meta, np.asarray(dir_test, dtype=float)), np.asarray(gate_test, dtype=float), buy_test_labels, take_threshold)
        metrics_list.append(metrics)
    agg = {
        'trades': int(round(float(np.mean([m['trades'] for m in metrics_list])))) if metrics_list else 0,
        'total_pnl': _round_float(float(np.mean([m['total_pnl'] for m in metrics_list])) if metrics_list else 0.0, 6),
        'max_drawdown': _round_float(float(np.mean([m['max_drawdown'] for m in metrics_list])) if metrics_list else 0.0, 6),
        'avg_expectancy': _round_float(float(np.mean([m['avg_expectancy'] for m in metrics_list])) if metrics_list else 0.0, 6),
        'high_conf_loss_rate': _round_float(float(np.mean([m['high_conf_loss_rate'] for m in metrics_list])) if metrics_list else 0.0, 6),
        'win_rate': _round_float(float(np.mean([m['win_rate'] for m in metrics_list])) if metrics_list else 0.0, 6),
        'abstain_rate': _round_float(float(np.mean([m['abstain_rate'] for m in metrics_list])) if metrics_list else 1.0, 6),
        'max_same_direction_loss_streak': int(max([m['max_same_direction_loss_streak'] for m in metrics_list] or [0])),
        'avg_buy_price': _round_float(float(np.mean([m['avg_buy_price'] for m in metrics_list])) if metrics_list else 0.0, 4),
        'avg_exit_hold_seconds': _round_float(float(np.mean([m['avg_exit_hold_seconds'] for m in metrics_list])) if metrics_list else 0.0, 2),
        'folds': metrics_list,
    }
    return {'metrics': agg, 'calibration': best_calibration, 'fit': best_fit}


def _train_profit_alpha_for_asset(asset: str) -> dict[str, Any]:
    df = pd.read_parquet(RELABELED_EPISODE_PATHS[asset])
    if df.empty:
        raise RuntimeError(f'empty relabeled dataset for {asset}')
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    feature_cols = [col for col in _load_v1_feature_cols(asset) if col in df.columns]
    buy_actions = _load_v1_buy_actions(asset)
    effective_days = int(min(180, math.floor((df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400.0)))
    candidates = _candidate_train_days(effective_days)
    if not candidates:
        candidates = [90]

    coarse_rows: list[dict[str, Any]] = []
    for train_days in candidates:
        folds = _iter_purged_splits(df, train_days=train_days, max_folds=1)
        if not folds:
            coarse_rows.append({'train_days': train_days, 'status': 'skipped_no_recent_fold'})
            continue
        split = folds[0]
        for template_name, params in PARAM_TEMPLATES.items():
            try:
                fit = _fit_heads(split, feature_cols, params, buy_actions, lgb_jobs=8)
                cal_meta = _select_calibration_from_split(fit, take_threshold=0.50)
                dir_test = fit['dir_model'].predict_proba(fit['X_te'])[:, 1]
                gate_test = fit['gate_model'].predict_proba(fit['X_te'])[:, 1]
                buy_test = fit['buy_model'].predict(fit['X_te'])
                buy_labels = np.asarray([fit['id_to_price'][int(i)] for i in buy_test], dtype=object)
                metrics = _simulate_profit_eval(split.test, _apply_calibration_meta(cal_meta, np.asarray(dir_test, dtype=float)), np.asarray(gate_test, dtype=float), buy_labels, 0.50)
                coarse_rows.append({
                    'asset': asset,
                    'train_days': train_days,
                    'template': template_name,
                    'params': params,
                    'metrics': metrics,
                    'score': _round_float(_candidate_score_v2(metrics), 6),
                    'status': 'ok',
                })
            except Exception as exc:
                coarse_rows.append({'asset': asset, 'train_days': train_days, 'template': template_name, 'status': f'skipped_fit:{exc}'})
    coarse = pd.DataFrame(coarse_rows)
    ok = coarse[coarse['status'] == 'ok'].sort_values('score', ascending=False).head(3).to_dict(orient='records') if not coarse.empty else []
    if not ok:
        raise RuntimeError(f'no viable profit-alpha candidates for {asset}')

    finalists: list[dict[str, Any]] = []
    for row in ok:
        train_days = int(row['train_days'])

        def objective(trial: optuna.Trial) -> float:
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 31, 255),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.08, log=True),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 150),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.50, 0.98),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.50, 0.98),
                'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
                'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.3),
                'lambda_l1': trial.suggest_float('lambda_l1', 0.001, 3.0, log=True),
                'lambda_l2': trial.suggest_float('lambda_l2', 1.0, 40.0, log=True),
                'max_bin': trial.suggest_int('max_bin', 127, 511),
            }
            take_threshold = trial.suggest_float('take_threshold', 0.46, 0.70)
            result = _evaluate_profit_candidate(df, feature_cols, params, buy_actions, train_days, take_threshold)
            metrics = result['metrics']
            trial.set_user_attr('metrics', metrics)
            trial.set_user_attr('calibration', result['calibration'])
            trial.set_user_attr('feature_cols', list(feature_cols))
            trial.set_user_attr('train_days', train_days)
            trial.set_user_attr('take_threshold', take_threshold)
            return _candidate_score_v2(metrics)

        study = optuna.create_study(direction='maximize', study_name=f'{asset}_profit_alpha_v2_{train_days}')
        study.optimize(objective, n_trials=120, n_jobs=SEARCH_WORKERS, show_progress_bar=False)
        best = study.best_trial
        finalists.append({
            'train_days': train_days,
            'seed_template': row.get('template'),
            'coarse_score': row.get('score'),
            'best_value': _round_float(best.value, 6),
            'best_params': best.params,
            'best_metrics': best.user_attrs.get('metrics', {}),
            'best_calibration': best.user_attrs.get('calibration', {}),
            'feature_cols': best.user_attrs.get('feature_cols', []),
        })
    finalists = sorted(finalists, key=lambda x: float(x.get('best_value') or -1e9), reverse=True)
    winner = finalists[0]
    params = dict(winner['best_params'])
    take_threshold = float(params.pop('take_threshold'))
    model_dir = V2B_MODEL_DIRS[asset]
    model_dir.mkdir(parents=True, exist_ok=True)

    direction_files: list[str] = []
    gate_files: list[str] = []
    buy_files: list[str] = []
    training_rows: list[dict[str, Any]] = []
    for window_days in sorted(set(candidates)):
        folds = _iter_purged_splits(df, train_days=window_days, max_folds=1)
        if not folds:
            training_rows.append({'window_days': window_days, 'status': 'skipped_no_recent_fold'})
            continue
        split = folds[0]
        fit = _fit_heads(split, feature_cols, params, buy_actions, lgb_jobs=8)
        cal_meta = _select_calibration_from_split(fit, take_threshold)
        dir_test = fit['dir_model'].predict_proba(fit['X_te'])[:, 1]
        gate_test = fit['gate_model'].predict_proba(fit['X_te'])[:, 1]
        buy_test = fit['buy_model'].predict(fit['X_te'])
        buy_labels = np.asarray([fit['id_to_price'][int(i)] for i in buy_test], dtype=object)
        metrics = _simulate_profit_eval(split.test, _apply_calibration_meta(cal_meta, np.asarray(dir_test, dtype=float)), np.asarray(gate_test, dtype=float), buy_labels, take_threshold)
        direction_file = f'direction_lgb_{window_days}d.joblib'
        gate_file = f'gate_lgb_{window_days}d.joblib'
        buy_file = f'buy_lgb_{window_days}d.joblib'
        joblib.dump(fit['dir_model'], model_dir / direction_file)
        joblib.dump(fit['gate_model'], model_dir / gate_file)
        joblib.dump(fit['buy_model'], model_dir / buy_file)
        direction_files.append(direction_file)
        gate_files.append(gate_file)
        buy_files.append(buy_file)
        training_rows.append({'window_days': window_days, 'status': 'trained', 'metrics': metrics, 'calibration': cal_meta, 'take_threshold': _round_float(take_threshold, 6)})

    write_json(model_dir / 'config.json', {
        'generated_at': utc_now_iso(),
        'asset': asset,
        'version': 'vnext_btc_profit_alpha_v2' if asset == 'BTC_USDT' else 'vnext_eth_profit_alpha_v2',
        'scope': 'profit_relabel_alpha_v2',
        'depends_on_execution_model_dir': str(V2A_MODEL_DIRS[asset]),
        'source_relabeled_dataset': str(RELABELED_EPISODE_PATHS[asset]),
        'feature_cols': feature_cols,
        'buy_actions': buy_actions,
        'take_threshold': _round_float(take_threshold, 6),
        'best_params': params,
        'best_metrics': winner['best_metrics'],
        'calibration': winner.get('best_calibration', {}),
        'model_files': {'direction': direction_files, 'gate': gate_files, 'buy': buy_files},
        'notes': ['manual_activation_only', 'profit_relabeled_alpha', 'depends_on_v2a_execution'],
    })
    write_json(model_dir / 'feature_cols.json', feature_cols)
    write_json(model_dir / 'hyperparams_snapshot.json', {'take_threshold': take_threshold, 'lgbm_params': params, 'winner': winner})
    return {
        'asset': asset,
        'model_dir': str(model_dir),
        'winner': winner,
        'training_rows': training_rows,
    }


def train_profit_alpha() -> dict[str, Any]:
    reset_payload = load_latest_reset_report()
    assets_payload: dict[str, Any] = {}
    write_json(PROFIT_ALPHA_REPORT, {
        'generated_at': utc_now_iso(),
        'scope': 'vnext_profit_alpha_v2_training',
        'status': 'running',
        'reset_generated_at': reset_payload.get('generated_at'),
        'assets': assets_payload,
    })
    for asset in ASSETS:
        assets_payload[asset] = _train_profit_alpha_for_asset(asset)
        write_json(PROFIT_ALPHA_REPORT, {
            'generated_at': utc_now_iso(),
            'scope': 'vnext_profit_alpha_v2_training',
            'status': 'running',
            'reset_generated_at': reset_payload.get('generated_at'),
            'assets': assets_payload,
        })
    payload = {
        'generated_at': utc_now_iso(),
        'scope': 'vnext_profit_alpha_v2_training',
        'status': 'completed',
        'reset_generated_at': reset_payload.get('generated_at'),
        'assets': assets_payload,
    }
    write_json(PROFIT_ALPHA_REPORT, payload)
    return payload


def assert_manual_ready() -> tuple[dict[str, Any], int]:
    readiness = _load_or_refresh_readiness()
    alignment_audit = load_json(V1_ALIGNMENT_AUDIT) or {}
    alignment_ok, alignment_meta = _alignment_audit_ok(alignment_audit if isinstance(alignment_audit, dict) else {})
    assets_payload: dict[str, Any] = {}
    ready = True
    has_current_anchor = False
    alignment_assets = alignment_meta.get('assets') if isinstance(alignment_meta, dict) else {}
    if not isinstance(alignment_assets, dict):
        alignment_assets = {}
    for asset in ASSETS:
        row = (readiness.get('assets') or {}).get(asset, {})
        settled = int(row.get('settled_compare_only_trades') or 0)
        snapshots = int(row.get('orderbook_snapshots') or 0)
        fill_samples = int(row.get('fillable_partial_timeout_samples') or 0)
        observed_ready = (
            settled >= MIN_TRADES
            and snapshots >= MIN_SNAPSHOTS
            and fill_samples >= MIN_FILL_SAMPLES
        )
        hist_ready, hist_meta = _historical_bootstrap_ready(row if isinstance(row, dict) else {})
        align_row = alignment_assets.get(asset) if isinstance(alignment_assets, dict) else {}
        align_status = str((align_row or {}).get('status') or '')
        align_asset_ok = bool((align_row or {}).get('ready'))
        bootstrap_assisted = hist_ready and align_asset_ok and not observed_ready
        asset_ready = observed_ready or bootstrap_assisted
        if align_asset_ok:
            has_current_anchor = True
        if observed_ready:
            mode = 'observed'
        elif bootstrap_assisted:
            mode = 'bootstrap_assisted'
        else:
            mode = 'blocked'
        assets_payload[asset] = {
            'asset': asset,
            'ready': asset_ready,
            'mode': mode,
            'alignment_status': align_status,
            'alignment_ready': align_asset_ok,
            'observed_ready': observed_ready,
            'historical_bootstrap_ready': hist_ready,
            'settled_compare_only_trades': settled,
            'orderbook_snapshots': snapshots,
            'fillable_partial_timeout_samples': fill_samples,
            'historical_bootstrap': hist_meta,
            'minimums': {
                'settled_compare_only_trades': MIN_TRADES,
                'orderbook_snapshots': MIN_SNAPSHOTS,
                'fillable_partial_timeout_samples': MIN_FILL_SAMPLES,
                'historical_settled_trades': MIN_HIST_SETTLED_TRADES,
                'historical_fill_events': MIN_HIST_FILL_EVENTS,
            },
        }
        ready = ready and asset_ready
    ready = bool(ready and has_current_anchor and alignment_ok)
    payload = {
        'generated_at': utc_now_iso(),
        'scope': 'vnext_execution_v2_manual_gate',
        'reset_generated_at': readiness.get('reset_generated_at'),
        'ready': ready,
        'alignment_audit_path': str(V1_ALIGNMENT_AUDIT),
        'alignment_ok': bool(alignment_ok),
        'has_current_anchor': has_current_anchor,
        'alignment_meta': alignment_meta,
        'assets': assets_payload,
    }
    return payload, (0 if ready else 2)


def main() -> int:
    ap = argparse.ArgumentParser(description='Prepare and train execution-aware v2 endpoint components for BTC/ETH compare-only lane')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('readiness')
    sub.add_parser('build-execution-dataset')
    sub.add_parser('train-execution')
    sub.add_parser('build-profit-relabel')
    sub.add_parser('train-profit-alpha')
    sub.add_parser('assert-manual-ready')
    args = ap.parse_args()

    os.environ.setdefault('OMP_NUM_THREADS', '8')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
    os.environ.setdefault('MKL_NUM_THREADS', '8')
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if args.cmd == 'readiness':
        payload = _readiness_with_historical_bootstrap()
    elif args.cmd == 'build-execution-dataset':
        payload = build_execution_datasets()
    elif args.cmd == 'train-execution':
        payload = train_execution()
    elif args.cmd == 'build-profit-relabel':
        payload = build_profit_relabel()
    elif args.cmd == 'train-profit-alpha':
        payload = train_profit_alpha()
    elif args.cmd == 'assert-manual-ready':
        payload, code = assert_manual_ready()
        print(json.dumps(payload, ensure_ascii=False))
        return code
    else:
        raise RuntimeError(f'unknown cmd: {args.cmd}')
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
