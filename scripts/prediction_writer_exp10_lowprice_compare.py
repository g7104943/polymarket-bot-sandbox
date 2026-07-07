#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLY = PROJECT_ROOT / 'polymarket'
REPORT = PROJECT_ROOT / 'reports' / 'exp10_lowprice_selection_latest.json'
DEFAULT_SOURCE_MAX_AGE_SEC = 1800
ASSETS = ('BTC', 'ETH')


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def get_confidence(entry: dict[str, Any]) -> float | None:
    try:
        c = entry.get('confidence')
        if c is not None:
            return float(c)
    except Exception:
        pass
    try:
        p = float((entry.get('details') or {}).get('proba_up'))
        return max(p, 1.0 - p)
    except Exception:
        return None


def dynamic_limit_price(confidence: float, threshold: float, low: float, high: float) -> float:
    denom = max(1e-9, 1.0 - float(threshold))
    scale = min(1.0, max(0.0, (float(confidence) - float(threshold)) / denom))
    price = float(low) + (float(high) - float(low)) * scale
    return round(price, 4)


def resolve_source_file(asset_payload: dict[str, Any]) -> Path | None:
    source_file = asset_payload.get('sourcePredictionFile')
    if isinstance(source_file, str) and source_file.strip():
        return Path(source_file)
    suffix = str(asset_payload.get('sourcePredictionSuffix') or '').strip()
    if suffix:
        return POLY / f'predictions{suffix}.json'
    return None


def build_empty_payload(source_file: Path | None, selection_mode: str | None, reason: str) -> dict[str, Any]:
    return {
        'timestamp': utc_now_iso(),
        'target_period_end_ts': None,
        'phase': 1,
        'limit_price': None,
        'source_file': source_file.name if source_file else None,
        'source_age_sec': None,
        'source_stale': reason == 'source_stale',
        'source_missing': reason == 'source_missing',
        'selection_mode': selection_mode,
        'predictions': {},
    }


def build_asset_output(asset: str, asset_payload: dict[str, Any], source_max_age_sec: int) -> tuple[Path, dict[str, Any]]:
    winner = asset_payload.get('winner') or {}
    selection = winner.get('selection')
    derived_file = Path(winner.get('derivedPredictionFile') or POLY / f'predictions_exp10_lowprice_{asset.lower()}.json')
    source_file = resolve_source_file(asset_payload)
    if not source_file or not source_file.exists():
        return derived_file, build_empty_payload(source_file, selection, 'source_missing')

    raw = load_json(source_file)
    if not isinstance(raw, dict):
        return derived_file, build_empty_payload(source_file, selection, 'source_missing')

    source_age_sec = max(0.0, time.time() - source_file.stat().st_mtime)
    if source_age_sec > float(source_max_age_sec):
        out = build_empty_payload(source_file, selection, 'source_stale')
        out['source_age_sec'] = round(source_age_sec, 3)
        out['target_period_end_ts'] = raw.get('target_period_end_ts')
        out['phase'] = raw.get('phase', 1)
        return derived_file, out

    predictions = raw.get('predictions') or {}
    if not isinstance(predictions, dict):
        predictions = {}
    first_entry = next((v for v in predictions.values() if isinstance(v, dict)), None)
    rules = load_json(Path(winner.get('rulesPath') or '')) if winner.get('rulesPath') else None
    if not isinstance(rules, dict):
        rules = {}

    limit_price: float | None = None
    if selection == 'fixed-opt':
        try:
            limit_price = round(float((rules.get('polymarket_constraints') or {}).get('buy_price')), 4)
        except Exception:
            limit_price = None
    elif selection == 'dynamic-band':
        try:
            low, high = (rules.get('polymarket_constraints') or {}).get('buy_price_range') or [0.30, 0.40]
            threshold = float((rules.get('trading_rules') or {}).get('min_confidence'))
            confidence = get_confidence(first_entry or {})
            if confidence is not None:
                limit_price = dynamic_limit_price(confidence, threshold, float(low), float(high))
        except Exception:
            limit_price = None

    out = dict(raw)
    out['timestamp'] = utc_now_iso()
    out['limit_price'] = limit_price
    out['source_file'] = source_file.name
    out['source_age_sec'] = round(source_age_sec, 3)
    out['source_stale'] = False
    out['selection_mode'] = selection
    out['base_trader_name'] = 'v5_exp10_bp0470'
    out['price_bounds'] = [0.30, 0.40]
    out['winner_reason'] = winner.get('reason')
    out['winner_rules_path'] = winner.get('rulesPath')
    out['predictions'] = predictions if limit_price is not None else {}
    return derived_file, out


def run_once(source_max_age_sec: int) -> dict[str, Any]:
    report = load_json(REPORT)
    if not isinstance(report, dict):
        raise RuntimeError(f'missing low-price selection report: {REPORT}')
    assets = report.get('assets') if isinstance(report.get('assets'), dict) else {}
    summary: dict[str, Any] = {'generatedAt': utc_now_iso(), 'assets': {}}
    for asset in ASSETS:
        payload = assets.get(asset)
        if not isinstance(payload, dict):
            continue
        derived_file, out = build_asset_output(asset, payload, source_max_age_sec)
        atomic_write_json(derived_file, out)
        summary['assets'][asset] = {
            'output': str(derived_file),
            'selection_mode': out.get('selection_mode'),
            'limit_price': out.get('limit_price'),
            'source_stale': out.get('source_stale', False),
            'prediction_count': len(out.get('predictions') or {}),
        }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description='Write derived exp10 low-price prediction files for BTC/ETH.')
    ap.add_argument('--loop', type=float, default=0.0, help='poll interval seconds; 0 = run once')
    ap.add_argument('--source-max-age-sec', type=int, default=int(os.environ.get('EXP10_LOWPRICE_SOURCE_MAX_AGE_SEC') or DEFAULT_SOURCE_MAX_AGE_SEC))
    args = ap.parse_args()

    while True:
        summary = run_once(max(60, int(args.source_max_age_sec)))
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        if args.loop <= 0:
            break
        time.sleep(max(1.0, float(args.loop)))


if __name__ == '__main__':
    main()
