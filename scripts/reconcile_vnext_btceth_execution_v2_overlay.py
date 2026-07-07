#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_vnext_btceth_entry_exit_v1 import (
    _hold_to_expiry_pnl,
    _simulate_family_policy,
    _spread_bucket_to_proxy,
)
from scripts.vnext_stage_common import atomic_write_json, load_execution_policy, load_json, round_float

RAW_LOG_DIRS = {
    'BTC': PROJECT_ROOT / 'polymarket' / 'logs_vnext_btc_execution_v2_raw',
    'ETH': PROJECT_ROOT / 'polymarket' / 'logs_vnext_eth_execution_v2_raw',
}
OVERLAY_LOG_DIRS = {
    'BTC': PROJECT_ROOT / 'polymarket' / 'logs_vnext_btc_execution_v2',
    'ETH': PROJECT_ROOT / 'polymarket' / 'logs_vnext_eth_execution_v2',
}
PATH_FILES = {
    'BTC': PROJECT_ROOT / 'data' / 'polymarket_1m_btc_usdt.parquet',
    'ETH': PROJECT_ROOT / 'data' / 'polymarket_1m_eth_usdt.parquet',
}
LEDGER_PATH = PROJECT_ROOT / 'polymarket' / 'predictions_vnext_btceth_execution_v2_ledger.json'
OUT_FILE = 'prediction_trades.simulation.json'
SLEEP_SECS = 20
SLUG_TS_RE = re.compile(r'-(\d{10})$')


def _load_trade_array(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    return [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []


def _load_path_groups(path: Path) -> dict[int, tuple[list[int], list[float]]]:
    if not path.exists():
        raise FileNotFoundError(f'missing path file: {path}')
    df = pd.read_parquet(path)
    if not {'15m_start_ts', 't_sec', 'p'}.issubset(df.columns):
        raise ValueError(f'invalid path columns: {path}')
    df = df.sort_values(['15m_start_ts', 't_sec']).reset_index(drop=True)
    out: dict[int, tuple[list[int], list[float]]] = {}
    for start_ts, group in df.groupby('15m_start_ts', sort=True):
        out[int(start_ts)] = (
            [int(v) for v in group['t_sec'].tolist()],
            [float(v) for v in group['p'].tolist()],
        )
    return out


def _slug_start_ts(market_slug: str) -> int | None:
    m = SLUG_TS_RE.search(str(market_slug or '').strip())
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _parse_ts(value: Any) -> int | None:
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except Exception:
            return None
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return int(datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp())
    except Exception:
        return None


def _fill_status_from_profile(fill_rate: float, partial_fill_rate: float, timeout_rate: float) -> str:
    if fill_rate <= 0.05 or timeout_rate >= 0.75:
        return 'timeout_unfilled'
    if partial_fill_rate >= 0.10:
        return 'partial_fill'
    return 'filled'


def _execution_adjusted_pnl(base_pnl: float, asset: str, exit_family: str, sell_quote_policy: str, spread_metric: float | None) -> tuple[float, dict[str, Any]]:
    if exit_family == 'HOLD_TO_EXPIRY':
        profile = {
            'selected_exit_family': exit_family,
            'selected_sell_quote_policy': sell_quote_policy,
            'fill_rate': 1.0,
            'partial_fill_rate': 0.0,
            'timeout_rate': 0.0,
            'avg_partial_fill_ratio': 1.0,
            'avg_queue_wait_seconds': 0.0,
            'avg_entry_queue_depth': 0.0,
            'avg_exit_queue_depth': 0.0,
            'policy_source': 'hold_to_expiry',
        }
        return float(base_pnl), profile
    profile = load_execution_policy(asset, exit_family, sell_quote_policy, spread_metric)
    fill_rate = max(0.0, min(1.0, float(profile.get('fill_rate') or 0.0)))
    partial_rate = max(0.0, min(1.0, float(profile.get('partial_fill_rate') or 0.0)))
    partial_ratio = max(0.0, min(1.0, (1.0 - partial_rate) + partial_rate * float(profile.get('avg_partial_fill_ratio') or 0.0)))
    if fill_rate <= 0.0:
        adjusted = -0.02
    else:
        adjusted = float(base_pnl) * fill_rate * partial_ratio
    return float(adjusted), profile


def _overlay_entry(raw: dict[str, Any], ledger: dict[str, Any] | None, paths: dict[int, tuple[list[int], list[float]]]) -> dict[str, Any]:
    symbol = str(raw.get('symbol') or '').upper()
    asset = 'BTC_USDT' if symbol == 'BTC' else 'ETH_USDT'
    market_slug = str(raw.get('marketSlug') or '')
    market_start_ts = _slug_start_ts(market_slug)

    base = dict(raw)
    base['role'] = 'compare_only'
    base['model_variant'] = str((ledger or {}).get('model_variant') or 'vnext_execution_v2')
    base['entry_limit_price'] = round_float((ledger or {}).get('entry_limit_price') or raw.get('limitPriceConfigured') or 0.0, 4)
    base['selected_exit_family'] = str((ledger or {}).get('selected_exit_family') or 'TIME_EXIT')
    base['selected_sell_quote_policy'] = str((ledger or {}).get('selected_sell_quote_policy') or 'TIMEOUT_THEN_HIT_BID')
    base['baseline_hold_pnl'] = round_float(raw.get('pnl') if raw.get('pnl') is not None else 0.0, 6)
    base['overlay_pnl'] = raw.get('pnl')
    base['exit_reason'] = 'pending'
    base['exit_price'] = None
    base['result'] = raw.get('result')
    base['avg_hold_seconds'] = None
    base['entry_fill_mode'] = 'runtime_executor'
    base['exit_fill_mode'] = 'queue_aware_v2'
    base['fill_status'] = 'queued'
    base['partial_fill_ratio'] = 0.0
    base['queue_wait_seconds'] = None
    base['entry_queue_depth'] = None
    base['exit_queue_depth'] = None

    if str(raw.get('status') or '').lower() != 'executed':
        return base
    if str(raw.get('result') or '').lower() not in {'win', 'lose'}:
        base['result'] = 'pending'
        base['pnl'] = None
        return base
    if market_start_ts is None or market_start_ts not in paths:
        base['exit_reason'] = 'fallback_hold_missing_path'
        return base

    path_t, path_up = paths[market_start_ts]
    direction = str(raw.get('direction') or 'UP').upper()
    direction_prices = path_up if direction == 'UP' else [float(max(0.0, min(1.0, 1.0 - p))) for p in path_up]
    entry_ts = _parse_ts((ledger or {}).get('entry_proxy_ts')) or _parse_ts(raw.get('timestamp')) or market_start_ts
    entry_price = float(raw.get('tokenPrice') or raw.get('limitPriceConfigured') or (ledger or {}).get('entry_limit_price') or 0.50)
    amount = float(raw.get('amount') or 0.0)
    won = str(raw.get('result') or '').lower() == 'win'
    baseline_hold_pnl = float(raw.get('pnl') if raw.get('pnl') is not None else _hold_to_expiry_pnl(amount, entry_price, won))
    spread_bucket = str((ledger or {}).get('spread_bucket') or 'normal')
    spread_proxy = _spread_bucket_to_proxy(asset, spread_bucket)
    exit_family = str((ledger or {}).get('selected_exit_family') or 'TIME_EXIT')
    sell_quote_policy = str((ledger or {}).get('selected_sell_quote_policy') or 'TIMEOUT_THEN_HIT_BID')
    spread_metric = float((ledger or {}).get('spread_metric') or 0.0)
    base_overlay_pnl, exit_price, exit_ts, exit_reason, _mae, _mfe = _simulate_family_policy(
        amount=amount,
        entry_price=entry_price,
        entry_ts=entry_ts,
        market_start_ts=market_start_ts,
        won=won,
        t_sec_list=path_t,
        direction_prices=direction_prices,
        exit_family=exit_family,
        sell_quote_policy=sell_quote_policy,
        spread_proxy=spread_proxy,
    )
    adjusted_pnl, profile = _execution_adjusted_pnl(base_overlay_pnl, asset, exit_family, sell_quote_policy, spread_metric)
    fill_status = _fill_status_from_profile(float(profile.get('fill_rate') or 0.0), float(profile.get('partial_fill_rate') or 0.0), float(profile.get('timeout_rate') or 0.0))
    overlay_result = 'win' if adjusted_pnl > 0 else 'lose'
    base.update({
        'entry_limit_price': round_float((ledger or {}).get('entry_limit_price') or raw.get('limitPriceConfigured') or entry_price, 4),
        'limitPriceConfigured': round_float((ledger or {}).get('entry_limit_price') or raw.get('limitPriceConfigured') or entry_price, 4),
        'exit_family': exit_family,
        'sell_quote_policy': sell_quote_policy,
        'exit_price': round_float(exit_price, 4),
        'exit_reason': exit_reason,
        'baseline_hold_pnl': round_float(baseline_hold_pnl, 6),
        'overlay_pnl': round_float(adjusted_pnl, 6),
        'raw_overlay_pnl': round_float(base_overlay_pnl, 6),
        'pnl': round_float(adjusted_pnl, 6),
        'result': overlay_result,
        'settledAt': datetime.fromtimestamp(exit_ts, tz=timezone.utc).isoformat().replace('+00:00', 'Z'),
        'avg_hold_seconds': int(exit_ts - entry_ts),
        'stoppedOut': exit_reason.startswith('stop_loss') or exit_reason.startswith('fast_fail'),
        'entry_fill_mode': 'runtime_executor',
        'exit_fill_mode': 'queue_aware_v2',
        'fill_status': fill_status,
        'partial_fill_ratio': round_float(profile.get('avg_partial_fill_ratio'), 6),
        'queue_wait_seconds': round_float(profile.get('avg_queue_wait_seconds'), 6),
        'entry_queue_depth': round_float(profile.get('avg_entry_queue_depth'), 6),
        'exit_queue_depth': round_float(profile.get('avg_exit_queue_depth'), 6),
        'role': 'compare_only',
    })
    return base


def _rebuild_asset(symbol: str, ledger_markets: dict[str, dict[str, Any]], paths: dict[int, tuple[list[int], list[float]]]) -> list[dict[str, Any]]:
    raw_path = RAW_LOG_DIRS[symbol] / OUT_FILE
    rows = _load_trade_array(raw_path)
    out: list[dict[str, Any]] = []
    for row in rows:
        market_slug = str(row.get('marketSlug') or '')
        ledger = ledger_markets.get(market_slug)
        out.append(_overlay_entry(row, ledger, paths))
    out.sort(key=lambda r: str(r.get('timestamp') or ''))
    return out


def run_once() -> None:
    ledger_payload = load_json(LEDGER_PATH)
    ledger_markets = dict((ledger_payload or {}).get('markets') or {}) if isinstance(ledger_payload, dict) else {}
    path_groups = {symbol: _load_path_groups(path) for symbol, path in PATH_FILES.items()}
    for symbol, out_dir in OVERLAY_LOG_DIRS.items():
        overlay_rows = _rebuild_asset(symbol, ledger_markets, path_groups[symbol])
        atomic_write_json(out_dir / OUT_FILE, overlay_rows)


def main() -> int:
    ap = argparse.ArgumentParser(description='Reconcile queue-aware execution v2 overlay logs for monitor_all_orders')
    ap.add_argument('--once', action='store_true')
    args = ap.parse_args()
    if args.once:
        run_once()
        return 0
    err_path = PROJECT_ROOT / 'logs' / 'reconcile_vnext_btceth_execution_v2_overlay.error.log'
    err_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            run_once()
        except Exception as exc:
            with err_path.open('a', encoding='utf-8') as fh:
                fh.write(f"{datetime.now().isoformat()} {exc}\n")
        time.sleep(SLEEP_SECS)


if __name__ == '__main__':
    raise SystemExit(main())
