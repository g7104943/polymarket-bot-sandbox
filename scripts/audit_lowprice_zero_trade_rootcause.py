#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.exp_lowprice_selected_common import load_json

POLY = ROOT / 'polymarket'
REPORT = ROOT / 'reports' / 'exp_lowprice_zero_trade_rootcause_latest.json'
GUARD_DIR = POLY / 'logs'
DEFAULT_SOURCE_CONFIG = POLY / 'trader_configs.json'
ARCHIVE_ROOT = POLY / 'archived_lowprice_resets'

TARGET_NAMES = {
    'v5_exp10_bp0300_btc',
    'v5_exp10_bp0300_eth',
    'v5_exp10_bp_dyn_0300_0330_eth',
}


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def load_guard(group: str) -> dict[str, Any]:
    path = GUARD_DIR / f'runtime_trade_guards_{group}.json'
    if not path.exists():
        return {}
    return load_json(path)


def guard_trader_row(guard_payload: dict[str, Any], trader_name: str) -> dict[str, Any] | None:
    traders = (((guard_payload or {}).get('guard_evidence_v1') or {}).get('traders') or [])
    for row in traders:
        if isinstance(row, dict) and str(row.get('traderName') or '') == trader_name:
            return row
    return None


def latest_archive_default_config() -> Path | None:
    if not ARCHIVE_ROOT.exists():
        return None
    candidates = sorted(
        [p / 'polymarket' / 'trader_configs_monitor_only_lowprice.json' for p in ARCHIVE_ROOT.iterdir() if p.is_dir()],
        key=lambda p: p.parent.parent.name,
        reverse=True,
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def source_runtime_trader_name(source_name: str, symbol: str) -> str:
    return f'{source_name}__simulation_{symbol.lower()}'


def classify(low_guard: dict[str, Any] | None, source_guard: dict[str, Any] | None, prediction_file_exists: bool = True) -> str:
    low_abstain = int((low_guard or {}).get('abstainedByMarketSelection') or 0)
    source_abstain = int((source_guard or {}).get('abstainedByMarketSelection') or 0)
    low_eval = int((low_guard or {}).get('evaluated') or 0)
    if low_abstain > 0 and source_abstain > 0:
        return 'source_same_cycle_preselector_abstain'
    if low_abstain > 0 and source_abstain == 0:
        return 'lowprice_extra_preselector_block'
    if low_eval == 0 and not prediction_file_exists:
        return 'stale_or_missing'
    if low_eval == 0:
        return 'no_signal'
    return 'runtime_guard_blocked'


def main() -> None:
    archive_cfg = latest_archive_default_config()
    if not archive_cfg:
        payload = {'generatedAt': now_iso(), 'ok': False, 'issue': 'missing_archived_default_lowprice_config'}
        dump_json(REPORT, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        raise SystemExit(1)
    rows = [row for row in load_json(archive_cfg) if str(row.get('name') or '') in TARGET_NAMES]
    source_rows = {str(row.get('name') or ''): row for row in load_json(DEFAULT_SOURCE_CONFIG) if isinstance(row, dict)}
    low_guard_payload = load_guard('lowprice_default_selected')
    out_rows = []
    issue_count = 0
    for row in rows:
        name = str(row.get('name') or '')
        symbol = str(row.get('lowPriceSymbol') or '').upper()
        source_name = str(row.get('lowPriceSourceTrader') or '')
        source_row = source_rows.get(source_name) or {}
        source_group = str(source_row.get('group') or '')
        source_guard_payload = load_guard(source_group) if source_group else {}
        low_guard = guard_trader_row(low_guard_payload, name)
        source_guard = guard_trader_row(source_guard_payload, source_runtime_trader_name(source_name, symbol))
        pred_suffix = str(row.get('predictionSuffixBySymbol', {}).get(symbol) or row.get('predictionSuffix') or '').strip()
        pred_path = POLY / f'predictions{pred_suffix}.json' if pred_suffix else None
        classification = classify(low_guard, source_guard, bool(pred_path and pred_path.exists()))
        out = {
            'name': name,
            'source_trader': source_name,
            'symbol': symbol,
            'source_group': source_group,
            'classification': classification,
            'lowprice_guard': low_guard,
            'source_guard': source_guard,
            'prediction_file': str(pred_path) if pred_path else None,
            'prediction_file_exists': bool(pred_path and pred_path.exists()),
        }
        if classification == 'lowprice_extra_preselector_block':
            issue_count += 1
        out_rows.append(out)
    payload = {
        'generatedAt': now_iso(),
        'archive_config': str(archive_cfg),
        'row_count': len(out_rows),
        'issue_count': issue_count,
        'rows': out_rows,
        'ok': issue_count == 0,
    }
    dump_json(REPORT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload['ok']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
