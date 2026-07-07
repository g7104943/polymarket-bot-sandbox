#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_vnext_btceth_entry_exit_v1 import polymarket_taker_fee_rate
from scripts.vnext_entry_replay import (
    ENTRY_REPLAY_PRESTART_SEC,
    market_end_from_start_ts,
    replay_entry_trace,
)
from scripts.vnext_execution_common import (
    ASSETS,
    ASSET_SYMBOL,
    REPORTS_DIR,
    V1_EPISODE_PATHS,
    load_ledger_markets,
    load_overlay_rows,
    load_raw_rows,
    utc_now_iso,
    write_json,
)

OUTPUT_PATH = REPORTS_DIR / 'vnext_entry_exit_v1_alignment_audit_latest.json'
PATH_FILES = {
    'BTC_USDT': PROJECT_ROOT / 'data' / 'polymarket_1m_btc_usdt.parquet',
    'ETH_USDT': PROJECT_ROOT / 'data' / 'polymarket_1m_eth_usdt.parquet',
}


def _round(v: Any, digits: int = 6) -> Any:
    try:
        fv = float(v)
    except Exception:
        return v
    if math.isnan(fv) or math.isinf(fv):
        return None
    return round(fv, digits)


def _slug_start_ts(slug: str | None) -> int | None:
    if not slug:
        return None
    try:
        return int(str(slug).rsplit('-', 1)[-1])
    except Exception:
        return None


def _iso_to_sec(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(pd.Timestamp(text).timestamp())
    except Exception:
        return None


def _load_path_map(asset: str) -> dict[int, tuple[list[int], list[float]]]:
    path = PATH_FILES[asset]
    if not path.exists():
        return {}
    df = pd.read_parquet(path)
    if df.empty:
        return {}
    out: dict[int, tuple[list[int], list[float]]] = {}
    for start_ts, group in df.groupby('15m_start_ts', sort=True):
        g = group.sort_values('t_sec')
        out[int(start_ts)] = (
            [int(v) for v in g['t_sec'].tolist()],
            [float(v) for v in g['p'].tolist()],
        )
    return out


def _expected_pnl(raw_row: dict[str, Any]) -> float | None:
    amount = raw_row.get('amount')
    buy_price = raw_row.get('tokenPrice')
    result = str(raw_row.get('result') or '').lower()
    try:
        amount_f = float(amount)
        buy_f = float(buy_price)
    except Exception:
        return None
    if buy_f <= 0 or amount_f <= 0:
        return None
    if result == 'lose':
        return -amount_f
    if result == 'win':
        tokens_after_fee = (amount_f / buy_f) * (1.0 - polymarket_taker_fee_rate(buy_f))
        return float(tokens_after_fee - amount_f)
    return None


def _audit_asset(asset: str, sample_size: int, path_map: dict[int, tuple[list[int], list[float]]]) -> dict[str, Any]:
    symbol = ASSET_SYMBOL[asset]
    raw_rows = [row for row in load_raw_rows(asset) if str(row.get('status') or '').lower() == 'executed']
    overlay_by_id = {str(row.get('id') or ''): row for row in load_overlay_rows(asset)}
    ledger = {
        slug: row for slug, row in load_ledger_markets().items()
        if str((row or {}).get('asset') or '') == asset
    }
    take_markets = {
        slug: row for slug, row in ledger.items()
        if bool(row.get('take_trade'))
    }
    raw_by_slug = {str(row.get('marketSlug') or ''): row for row in raw_rows}

    if not take_markets:
        return {
            'asset': asset,
            'status': 'waiting_no_take_markets',
            'settled_raw_trades': int(len(raw_rows)),
            'take_markets': 0,
        }

    fill_match = 0
    fill_compared = 0
    fill_price_errors: list[float] = []
    pnl_errors: list[float] = []
    overlay_errors: list[float] = []
    sample_rows: list[dict[str, Any]] = []
    no_fill_markets = 0

    for slug, ledger_row in sorted(take_markets.items()):
        market_start_ts = int(
            ledger_row.get('market_start_ts')
            or ledger_row.get('decision_ts')
            or _slug_start_ts(slug)
            or 0
        )
        if market_start_ts <= 0:
            continue
        prediction_ts = int(
            ledger_row.get('prediction_ts')
            or _iso_to_sec(ledger_row.get('prediction_timestamp'))
            or (market_start_ts - ENTRY_REPLAY_PRESTART_SEC)
        )
        direction = str(ledger_row.get('direction') or 'UP').upper()
        limit_price = float(ledger_row.get('entry_limit_price') or 0.0)
        path = path_map.get(int(market_start_ts))
        if not path:
            continue
        t_sec_list, up_path = path
        direction_prices = up_path if direction == 'UP' else [float(max(0.0, min(1.0, 1.0 - p))) for p in up_path]
        replay = replay_entry_trace(
            prediction_ts=prediction_ts,
            market_start_ts=market_start_ts,
            limit_price=limit_price,
            t_sec_list=t_sec_list,
            direction_prices=direction_prices,
            market_slug=slug,
        )
        raw_row = raw_by_slug.get(slug)
        runtime_fill_status = 'filled' if raw_row else 'timeout_unfilled'
        fill_compared += 1
        if replay.fill_status == runtime_fill_status:
            fill_match += 1
        if raw_row is None:
            no_fill_markets += 1
            continue
        overlay_row = overlay_by_id.get(str(raw_row.get('id') or ''))
        runtime_fill_price = float(raw_row.get('tokenPrice') or 0.0)
        if replay.fill_price is not None and runtime_fill_price > 0:
            fill_price_errors.append(abs(float(replay.fill_price) - runtime_fill_price))
        expected_pnl = _expected_pnl(raw_row)
        raw_pnl = raw_row.get('pnl')
        try:
            raw_pnl_f = float(raw_pnl)
        except Exception:
            raw_pnl_f = None
        if expected_pnl is not None and raw_pnl_f is not None:
            pnl_errors.append(abs(expected_pnl - raw_pnl_f))
        if overlay_row and raw_pnl_f is not None:
            overlay_pnl = overlay_row.get('overlay_pnl')
            try:
                overlay_pnl_f = float(overlay_pnl)
            except Exception:
                overlay_pnl_f = None
            if overlay_pnl_f is not None:
                overlay_errors.append(abs(overlay_pnl_f - raw_pnl_f))
        if len(sample_rows) < sample_size:
            sample_rows.append({
                'market_slug': slug,
                'symbol': symbol,
                'direction': direction,
                'limit_price': _round(limit_price, 4),
                'prediction_ts': prediction_ts,
                'market_start_ts': market_start_ts,
                'market_end_ts': market_end_from_start_ts(market_start_ts),
                'replay_fill_status': replay.fill_status,
                'replay_fill_ts': replay.fill_ts,
                'replay_fill_price': _round(replay.fill_price, 6),
                'runtime_fill_status': runtime_fill_status,
                'runtime_fill_ts': _iso_to_sec(raw_row.get('timestamp')),
                'runtime_fill_price': _round(runtime_fill_price, 6),
                'raw_pnl': _round(raw_pnl_f, 6),
                'expected_raw_pnl': _round(expected_pnl, 6),
                'overlay_pnl': _round((overlay_row or {}).get('overlay_pnl'), 6),
            })

    return {
        'asset': asset,
        'status': 'ok' if fill_compared else 'waiting_no_comparable_rows',
        'take_markets': int(len(take_markets)),
        'executed_raw_trades': int(len(raw_rows)),
        'no_fill_markets': int(no_fill_markets),
        'fill_status_match_rate': _round(fill_match / fill_compared if fill_compared else 0.0, 6),
        'avg_abs_fill_price_error': _round(sum(fill_price_errors) / len(fill_price_errors), 6) if fill_price_errors else None,
        'avg_abs_raw_pnl_formula_error': _round(sum(pnl_errors) / len(pnl_errors), 6) if pnl_errors else None,
        'avg_abs_overlay_vs_raw_pnl_error': _round(sum(overlay_errors) / len(overlay_errors), 6) if overlay_errors else None,
        'sample_rows': sample_rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description='Audit current-cycle v1 entry replay alignment against runtime fills and pnl')
    ap.add_argument('--sample-size', type=int, default=20)
    ap.add_argument('--output', type=Path, default=OUTPUT_PATH)
    args = ap.parse_args()

    assets_payload = {}
    for asset in ASSETS:
        assets_payload[asset] = _audit_asset(asset, int(args.sample_size), _load_path_map(asset))
    payload = {
        'generated_at': utc_now_iso(),
        'scope': 'vnext_entry_exit_v1_alignment_audit',
        'sample_size': int(args.sample_size),
        'assets': assets_payload,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
