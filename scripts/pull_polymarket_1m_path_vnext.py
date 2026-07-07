#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DIR = DATA_DIR / 'raw'
GAMMA_API = 'https://gamma-api.polymarket.com'
CLOB_URL = 'https://clob.polymarket.com/prices-history'
ASSET_TO_SLUG = {'BTC_USDT': 'btc', 'ETH_USDT': 'eth'}
DEFAULT_TIMEOUT = 20
ENTRY_REPLAY_PRESTART_SEC = 900
MARKET_DURATION_SEC = 900


def _meta_path(asset: str) -> Path:
    return DATA_DIR / f'polymarket_1m_{asset.lower()}_meta.json'


def _load_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_meta(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def _load_15m_start_ts(asset: str, days: int) -> list[int]:
    path = RAW_DIR / f'{asset.lower()}_15m.parquet'
    df = pd.read_parquet(path, columns=['timestamp'])
    ts = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    cutoff = ts.max() - pd.Timedelta(days=days)
    df = df.loc[ts >= cutoff].copy()
    vals = (pd.to_numeric(df['timestamp'], errors='coerce').astype('int64') // 1000).tolist()
    return [int(v) for v in vals]


def _fetch_event(session: requests.Session, slug: str) -> dict[str, Any] | None:
    try:
        r = session.get(f'{GAMMA_API}/events/slug/{slug}', timeout=DEFAULT_TIMEOUT)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


def _has_market(session: requests.Session, asset: str, start_ts: int) -> bool:
    slug = f"{ASSET_TO_SLUG[asset]}-updown-15m-{start_ts}"
    event = _fetch_event(session, slug)
    return _token_id_from_event(event) is not None


def _token_id_from_event(event: dict[str, Any] | None) -> str | None:
    if not isinstance(event, dict):
        return None
    markets = event.get('markets') or []
    if not markets:
        return None
    market = markets[0]
    cids = market.get('clobTokenIds')
    if isinstance(cids, str):
        if cids.startswith('['):
            try:
                vals = json.loads(cids)
                return str(vals[0]) if vals else None
            except Exception:
                return None
        parts = [x.strip() for x in cids.split(',') if x.strip()]
        return parts[0] if parts else None
    if isinstance(cids, list):
        return str(cids[0]) if cids else None
    return None


def _fetch_history(session: requests.Session, token_id: str, start_ts: int, end_ts: int) -> list[dict[str, Any]]:
    try:
        r = session.get(
            CLOB_URL,
            params={'market': token_id, 'startTs': start_ts, 'endTs': end_ts, 'fidelity': 1},
            timeout=DEFAULT_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        hist = data.get('history') or []
        if isinstance(hist, list):
            return [h for h in hist if isinstance(h, dict) and 't' in h and 'p' in h]
    except Exception:
        return []
    return []


def _fetch_one(asset: str, start_ts: int) -> dict[str, Any]:
    slug = f"{ASSET_TO_SLUG[asset]}-updown-15m-{start_ts}"
    with requests.Session() as session:
        event = _fetch_event(session, slug)
        token_id = _token_id_from_event(event)
        if not token_id:
            return {'status': 'no_market', 'start_ts': int(start_ts), 'rows': []}
        replay_start_ts = int(start_ts) - ENTRY_REPLAY_PRESTART_SEC
        replay_end_ts = int(start_ts) + MARKET_DURATION_SEC
        hist = _fetch_history(session, token_id, replay_start_ts, replay_end_ts)
        if not hist:
            return {'status': 'empty_history', 'start_ts': int(start_ts), 'rows': []}
        rows = []
        for row in hist:
            try:
                rows.append({'15m_start_ts': int(start_ts), 't_sec': int(row['t']), 'p': float(row['p'])})
            except Exception:
                continue
        if not rows:
            return {'status': 'empty_history', 'start_ts': int(start_ts), 'rows': []}
        return {'status': 'ok', 'start_ts': int(start_ts), 'rows': rows}


def _iter_missing(starts: Iterable[int], existing: set[int], known_unavailable: set[int]) -> list[int]:
    return [int(ts) for ts in starts if int(ts) not in existing and int(ts) not in known_unavailable]


def _write_parquet(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    new_df = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_parquet(path)
        df = pd.concat([old, new_df], ignore_index=True)
        df = df.drop_duplicates(subset=['15m_start_ts', 't_sec'], keep='last').sort_values(['15m_start_ts', 't_sec']).reset_index(drop=True)
    else:
        df = new_df.sort_values(['15m_start_ts', 't_sec']).reset_index(drop=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _summarize_meta(asset: str, requested_days: int, starts: list[int], existing: set[int], known_unavailable: set[int]) -> dict[str, Any]:
    available_sorted = sorted(existing)
    unavailable_sorted = sorted(known_unavailable)
    first_available_ts = int(available_sorted[0]) if available_sorted else None
    latest_available_ts = int(available_sorted[-1]) if available_sorted else None
    effective_path_days = None
    if first_available_ts is not None and latest_available_ts is not None:
        effective_path_days = round((latest_available_ts - first_available_ts) / 86400.0, 3)
    return {
        'generated_at': pd.Timestamp.now(tz='UTC').isoformat(),
        'asset': asset,
        'requested_days': int(requested_days),
        'bars_total': int(len(starts)),
        'available_bars': int(len(existing)),
        'known_unavailable_bars': int(len(known_unavailable)),
        'first_available_start_ts': first_available_ts,
        'latest_available_start_ts': latest_available_ts,
        'effective_path_days': effective_path_days,
        'known_unavailable_start_ts': unavailable_sorted,
    }


def _discover_prefix_unavailable(asset: str, starts: list[int], existing: set[int], known_unavailable: set[int]) -> set[int]:
    candidates = [int(ts) for ts in starts if int(ts) not in existing and int(ts) not in known_unavailable]
    if not candidates:
        return set()
    with requests.Session() as session:
        if _has_market(session, asset, candidates[0]):
            return set()
        lo = 0
        hi = len(candidates) - 1
        first_ok_idx: int | None = None
        while lo <= hi:
            mid = (lo + hi) // 2
            if _has_market(session, asset, candidates[mid]):
                first_ok_idx = mid
                hi = mid - 1
            else:
                lo = mid + 1
    if first_ok_idx is None:
        return set(candidates)
    return set(candidates[:first_ok_idx])


def main() -> int:
    ap = argparse.ArgumentParser(description='Download Polymarket 1m path data for BTC/ETH compare-only entry-exit models')
    ap.add_argument('--assets', nargs='+', default=['BTC_USDT', 'ETH_USDT'])
    ap.add_argument('--days', type=int, default=180)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit-bars', type=int, default=0)
    ap.add_argument('--refresh-all', action='store_true')
    args = ap.parse_args()

    for asset in args.assets:
        asset = asset.strip().upper()
        if asset not in ASSET_TO_SLUG:
            raise SystemExit(f'unsupported asset: {asset}')
        out_path = DATA_DIR / f'polymarket_1m_{asset.lower()}.parquet'
        meta_path = _meta_path(asset)
        starts = _load_15m_start_ts(asset, days=args.days)
        if args.limit_bars > 0:
            starts = starts[-int(args.limit_bars):]
        existing: set[int] = set()
        if args.refresh_all and out_path.exists():
            out_path.unlink()
        if out_path.exists():
            try:
                existing_df = pd.read_parquet(out_path, columns=['15m_start_ts'])
                existing = {int(v) for v in existing_df['15m_start_ts'].dropna().astype('int64').tolist()}
            except Exception:
                existing = set()
        meta = _load_meta(meta_path)
        known_unavailable = {
            int(v) for v in (meta.get('known_unavailable_start_ts') or [])
            if str(v).strip()
        }
        prefix_unavailable = _discover_prefix_unavailable(asset, starts, existing, known_unavailable)
        if prefix_unavailable:
            known_unavailable.update(prefix_unavailable)
            _write_meta(meta_path, _summarize_meta(asset, args.days, starts, existing, known_unavailable))
        missing = _iter_missing(starts, existing, known_unavailable)
        print(
            f'[1m-path] {asset}: bars_total={len(starts)} missing={len(missing)} '
            f'known_unavailable={len(known_unavailable)} out={out_path}'
        )
        collected: list[dict[str, Any]] = []
        newly_available: set[int] = set()
        newly_unavailable: set[int] = set()
        if missing:
            with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
                futures = {ex.submit(_fetch_one, asset, ts): ts for ts in missing}
                done = 0
                for fut in as_completed(futures):
                    ts = futures[fut]
                    try:
                        result = fut.result()
                    except Exception:
                        result = {'status': 'exception', 'start_ts': int(ts), 'rows': []}
                    rows = list(result.get('rows') or [])
                    status = str(result.get('status') or '')
                    if status == 'ok' and rows:
                        collected.extend(rows)
                        newly_available.add(int(ts))
                    elif status in {'no_market', 'empty_history'}:
                        newly_unavailable.add(int(ts))
                    done += 1
                    if done % 100 == 0 or done == len(missing):
                        print(
                            f'[1m-path] {asset}: {done}/{len(missing)} bars fetched, '
                            f'rows_buffer={len(collected)} ok={len(newly_available)} '
                            f'unavailable={len(newly_unavailable)}'
                        )
                    if done % 200 == 0 or done == len(missing):
                        if collected:
                            _write_parquet(out_path, collected)
                            collected = []
                            existing.update(newly_available)
                        if newly_unavailable:
                            known_unavailable.update(newly_unavailable)
                        _write_meta(meta_path, _summarize_meta(asset, args.days, starts, existing, known_unavailable))
        elif not out_path.exists():
            _write_parquet(out_path, [])
        if newly_available:
            existing.update(newly_available)
        if newly_unavailable:
            known_unavailable.update(newly_unavailable)
        _write_meta(meta_path, _summarize_meta(asset, args.days, starts, existing, known_unavailable))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
