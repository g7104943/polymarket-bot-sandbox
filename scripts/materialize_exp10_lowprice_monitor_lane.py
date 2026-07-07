#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLY = PROJECT_ROOT / 'polymarket'
REPORT = PROJECT_ROOT / 'reports' / 'exp10_lowprice_selection_latest.json'
TRADER_CONFIGS = POLY / 'trader_configs.json'
BASE_TRADER_NAME = 'v5_exp10_bp0470'
GROUP = 'v5_exp10_lowprice_compare'
INITIAL_CAPITAL = 400.0
MONITOR_FILE = POLY / 'monitor_only_traders_exp10_lowprice.json'
ACTIVE_FILE = POLY / 'active_traders_monitor_only_exp10_lowprice.json'
CONFIG_FILE = POLY / 'trader_configs_monitor_only_exp10_lowprice.json'


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


def load_base_row() -> dict[str, Any]:
    rows = load_json(TRADER_CONFIGS)
    if not isinstance(rows, list):
        raise RuntimeError(f'bad trader config payload: {TRADER_CONFIGS}')
    row = next((r for r in rows if isinstance(r, dict) and r.get('name') == BASE_TRADER_NAME), None)
    if not isinstance(row, dict):
        raise RuntimeError(f'missing base trader: {BASE_TRADER_NAME}')
    return row


def build_trader_row(base: dict[str, Any], asset: str, selection_row: dict[str, Any]) -> dict[str, Any]:
    winner = selection_row.get('winner') or {}
    suffix = str(winner.get('derivedPredictionSuffix') or '').strip()
    if not suffix:
        raise RuntimeError(f'missing derivedPredictionSuffix for {asset}')
    row = dict(base)
    row['name'] = f'v5_exp10_lowprice_{asset.lower()}'
    row['group'] = GROUP
    row['profile'] = 'default'
    row['role'] = 'compare_only'
    row['logsDir'] = f'logs_v5_exp10_lowprice_{asset.lower()}'
    row['predictionSuffix'] = suffix
    row['predictionSuffixBySymbol'] = {asset: suffix}
    row['rulesJsonPath'] = str(winner.get('rulesPath') or row.get('rulesJsonPath') or '')
    row.pop('predictionFallbackSuffix', None)
    row['allowedMarkets'] = asset
    row['initialCapital'] = INITIAL_CAPITAL
    row['perCoinCapital'] = False
    row['numTradingAssets'] = 1
    row['runtimeSplitBySymbol'] = False
    row['usePredictionLimitPrice'] = True
    row['limitPrice'] = None
    row['limitPriceLadder'] = None
    row['lowPriceBaseTraderName'] = BASE_TRADER_NAME
    row['lowPriceSelectionMode'] = winner.get('selection')
    row['lowPriceSelectionReason'] = winner.get('reason')
    row['lowPriceSelectionRulesPath'] = winner.get('rulesPath')
    row['lowPriceSelectionReport'] = str(REPORT)
    row['lowPricePriceBounds'] = [0.30, 0.40]
    return row


def main() -> None:
    report = load_json(REPORT)
    if not isinstance(report, dict) or not isinstance(report.get('assets'), dict):
        raise RuntimeError(f'missing selection report: {REPORT}')
    base = load_base_row()
    trader_rows: list[dict[str, Any]] = []
    monitor_rows: list[dict[str, Any]] = []
    trader_names: list[str] = []
    for asset in ('BTC', 'ETH'):
        asset_row = report['assets'].get(asset)
        if not isinstance(asset_row, dict):
            continue
        winner = asset_row.get('winner') or {}
        if not winner.get('selection'):
            continue
        row = build_trader_row(base, asset, asset_row)
        trader_rows.append(row)
        trader_names.append(str(row['name']))
        monitor_rows.append({
            'name': row['name'],
            'group': row['group'],
            'profile': row['profile'],
            'logsDir': row['logsDir'],
            'predictionSuffix': row['predictionSuffix'],
            'allowedMarkets': row['allowedMarkets'],
            'role': row['role'],
            'symbol': asset,
            'selectionMode': row['lowPriceSelectionMode'],
        })

    atomic_write_json(CONFIG_FILE, trader_rows)
    atomic_write_json(ACTIVE_FILE, {
        'generatedAt': utc_now_iso(),
        'source': 'materialize_exp10_lowprice_monitor_lane.py',
        'scope': 'exp10_lowprice_compare_only',
        'traderNames': trader_names,
        'active_traders': trader_names,
        'groups': [GROUP] if trader_names else [],
    })
    atomic_write_json(MONITOR_FILE, {
        'generatedAt': utc_now_iso(),
        'scope': 'compare_only_monitor_lane',
        'traders': monitor_rows,
    })
    print(json.dumps({
        'status': 'ok',
        'config': str(CONFIG_FILE),
        'active': str(ACTIVE_FILE),
        'monitor': str(MONITOR_FILE),
        'traders': trader_names,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
