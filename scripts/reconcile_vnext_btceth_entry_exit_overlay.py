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

RAW_LOG_DIRS = {
    'BTC': PROJECT_ROOT / 'polymarket' / 'logs_vnext_btc_entry_exit_v1_raw',
    'ETH': PROJECT_ROOT / 'polymarket' / 'logs_vnext_eth_entry_exit_v1_raw',
}
OVERLAY_LOG_DIRS = {
    'BTC': PROJECT_ROOT / 'polymarket' / 'logs_vnext_btc_entry_exit_v1',
    'ETH': PROJECT_ROOT / 'polymarket' / 'logs_vnext_eth_entry_exit_v1',
}
PATH_FILES = {
    'BTC': PROJECT_ROOT / 'data' / 'polymarket_1m_btc_usdt.parquet',
    'ETH': PROJECT_ROOT / 'data' / 'polymarket_1m_eth_usdt.parquet',
}
LEDGER_PATH = PROJECT_ROOT / 'polymarket' / 'predictions_vnext_btceth_entry_exit_v1_ledger.json'
OUT_FILE = 'prediction_trades.simulation.json'
SLEEP_SECS = 20
SLUG_TS_RE = re.compile(r'-(\d{10})$')


def _round(v: Any, digits: int = 6) -> Any:
    try:
        fv = float(v)
    except Exception:
        return v
    if math.isnan(fv) or math.isinf(fv):
        return None
    return round(fv, digits)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def _load_trade_array(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
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


def _overlay_entry(raw: dict[str, Any], ledger: dict[str, Any] | None, paths: dict[int, tuple[list[int], list[float]]]) -> dict[str, Any]:
    symbol = str(raw.get('symbol') or '').upper()
    asset = 'BTC_USDT' if symbol == 'BTC' else 'ETH_USDT'
    market_slug = str(raw.get('marketSlug') or '')
    market_start_ts = _slug_start_ts(market_slug)

    base = dict(raw)
    base['role'] = 'compare_only'
    base['model_variant'] = str((ledger or {}).get('model_variant') or 'vnext_entry_exit_v1')
    base['entry_limit_price'] = _round((ledger or {}).get('entry_limit_price') or raw.get('limitPriceConfigured') or 0.0, 4)
    base['selected_exit_family'] = str((ledger or {}).get('selected_exit_family') or 'HOLD_TO_EXPIRY')
    base['selected_sell_quote_policy'] = str((ledger or {}).get('selected_sell_quote_policy') or 'MID_MINUS_0.01')
    base['confidence_bucket'] = str((ledger or {}).get('confidence_bucket') or 'mid')
    base['trend_bucket'] = str((ledger or {}).get('trend_bucket') or 'mixed')
    base['vol_bucket'] = str((ledger or {}).get('vol_bucket') or 'mid')
    base['spread_bucket'] = str((ledger or {}).get('spread_bucket') or 'normal')
    base['baseline_hold_pnl'] = _round(raw.get('pnl') if raw.get('pnl') is not None else 0.0, 6)
    base['overlay_pnl'] = raw.get('pnl')
    base['exit_reason'] = 'pending'
    base['exit_price'] = None
    base['result'] = raw.get('result')
    base['avg_hold_seconds'] = None

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
    spread_proxy = _spread_bucket_to_proxy(asset, str((ledger or {}).get('spread_bucket') or 'normal'))
    exit_family = str((ledger or {}).get('selected_exit_family') or 'HOLD_TO_EXPIRY')
    sell_quote_policy = str((ledger or {}).get('selected_sell_quote_policy') or 'MID_MINUS_0.01')
    overlay_pnl, exit_price, exit_ts, exit_reason, _mae, _mfe = _simulate_family_policy(
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
    overlay_result = 'win' if overlay_pnl > 0 else 'lose'
    base.update({
        'entry_limit_price': _round((ledger or {}).get('entry_limit_price') or raw.get('limitPriceConfigured') or entry_price, 4),
        'limitPriceConfigured': _round((ledger or {}).get('entry_limit_price') or raw.get('limitPriceConfigured') or entry_price, 4),
        'exit_family': exit_family,
        'sell_quote_policy': sell_quote_policy,
        'exit_price': _round(exit_price, 4),
        'exit_reason': exit_reason,
        'baseline_hold_pnl': _round(baseline_hold_pnl, 6),
        'overlay_pnl': _round(overlay_pnl, 6),
        'pnl': _round(overlay_pnl, 6),
        'result': overlay_result,
        'settledAt': datetime.fromtimestamp(exit_ts, tz=timezone.utc).isoformat().replace('+00:00', 'Z'),
        'avg_hold_seconds': int(exit_ts - entry_ts),
        'stoppedOut': exit_reason.startswith('stop_loss') or exit_reason.startswith('fast_fail'),
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
    ledger_payload = _load_json(LEDGER_PATH)
    ledger_markets = dict((ledger_payload or {}).get('markets') or {}) if isinstance(ledger_payload, dict) else {}
    path_groups = {symbol: _load_path_groups(path) for symbol, path in PATH_FILES.items()}
    for symbol, out_dir in OVERLAY_LOG_DIRS.items():
        overlay_rows = _rebuild_asset(symbol, ledger_markets, path_groups[symbol])
        _atomic_write_json(out_dir / OUT_FILE, overlay_rows)


def main() -> int:
    ap = argparse.ArgumentParser(description='Reconcile compare-only entry-exit overlay logs for monitor_all_orders')
    ap.add_argument('--once', action='store_true')
    args = ap.parse_args()
    if args.once:
        run_once()
        return 0
    err_path = PROJECT_ROOT / 'logs' / 'reconcile_vnext_btceth_entry_exit_overlay.error.log'
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
