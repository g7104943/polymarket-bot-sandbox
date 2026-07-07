#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.exp_lowprice_selected_common import PROFILES, dump_json, load_json

POLY = ROOT / 'polymarket'
OPT_REPORT = ROOT / 'reports' / 'exp_lowprice_selected_optimization_latest.json'
REPORT = ROOT / 'reports' / 'exp_lowprice_runtime_audit_latest.json'
ASSET_KEY = {'BTC': 'BTC_USDT_15m', 'ETH': 'ETH_USDT_15m', 'XRP': 'XRP_USDT_15m'}
PROFILE_CONFIG_MAP = {
    'default': POLY / 'trader_configs.json',
    '70': POLY / 'trader_configs_70.json',
}
GUARD_DIR = POLY / 'logs'


def pgrep_count(pattern: str) -> int:
    proc = subprocess.run(['bash', '-lc', f"pgrep -f '{pattern}' | wc -l"], capture_output=True, text=True)
    try:
        return int(proc.stdout.strip() or '0')
    except Exception:
        return 0


def resolve_prediction_suffixes(row: dict[str, Any], symbol: str) -> tuple[str | None, str | None]:
    by_symbol = row.get('predictionSuffixBySymbol') if isinstance(row.get('predictionSuffixBySymbol'), dict) else {}
    primary = str(by_symbol.get(symbol) or '').strip() or None
    base = str(row.get('predictionSuffix') or '').strip() or None
    return primary, base


def _inspect_single_prediction_file(path: Path | None, asset_key: str | None) -> dict[str, Any]:
    source = {
        'prediction_file': str(path) if path else None,
        'prediction_file_exists': bool(path and path.exists()),
        'prediction_file_age_sec': None,
        'prediction_file_fresh': False,
        'prediction_payload_present': False,
        'prediction_direction': None,
        'prediction_confidence': None,
        'latest_should_trade': None,
        'latest_skip_reason': None,
    }
    if not path or not path.exists():
        return source
    source['prediction_file_age_sec'] = round(time.time() - path.stat().st_mtime, 3)
    source['prediction_file_fresh'] = bool(source['prediction_file_age_sec'] <= 1800.0)
    try:
        payload = load_json(path)
    except Exception:
        return source
    predictions = payload.get('predictions') if isinstance(payload, dict) else None
    if not isinstance(predictions, dict):
        return source
    cell = predictions.get(asset_key) if asset_key else None
    if isinstance(cell, dict):
        source['prediction_payload_present'] = True
        source['prediction_direction'] = cell.get('direction')
        source['prediction_confidence'] = cell.get('confidence')
        details = cell.get('details') if isinstance(cell.get('details'), dict) else {}
        trade_decision = details.get('trade_decision') if isinstance(details, dict) else {}
        source['latest_should_trade'] = trade_decision.get('should_trade')
        source['latest_skip_reason'] = trade_decision.get('skip_reason')
    return source


def inspect_prediction_source_for_row(row: dict[str, Any], symbol: str) -> dict[str, Any]:
    symbol = str(symbol or '').upper()
    primary_suffix, base_suffix = resolve_prediction_suffixes(row, symbol)
    pred_path = POLY / f'predictions{primary_suffix}.json' if primary_suffix else None
    fallback_path = None
    if base_suffix and base_suffix != primary_suffix:
        fallback_path = POLY / f'predictions{base_suffix}.json'
    asset_key = ASSET_KEY.get(symbol)
    primary = _inspect_single_prediction_file(pred_path, asset_key)
    fallback = _inspect_single_prediction_file(fallback_path, asset_key)
    effective = primary
    resolution = 'primary'
    if not primary.get('prediction_file_fresh') and fallback.get('prediction_file_fresh'):
        effective = fallback
        resolution = 'fallback_base'
    elif primary.get('prediction_file_exists') and not primary.get('prediction_file_fresh') and fallback.get('prediction_file_exists'):
        resolution = 'primary_stale_fallback_unusable'
    return {
        'prediction_suffix': primary_suffix or base_suffix,
        'primary_prediction_suffix': primary_suffix,
        'base_prediction_suffix': base_suffix,
        'prediction_resolution': resolution,
        'prediction_fallback_available': bool(fallback_path),
        **effective,
        'primary_prediction_file': primary.get('prediction_file'),
        'primary_prediction_file_exists': primary.get('prediction_file_exists'),
        'primary_prediction_file_age_sec': primary.get('prediction_file_age_sec'),
        'primary_prediction_file_fresh': primary.get('prediction_file_fresh'),
        'primary_prediction_payload_present': primary.get('prediction_payload_present'),
        'fallback_prediction_file': fallback.get('prediction_file'),
        'fallback_prediction_file_exists': fallback.get('prediction_file_exists'),
        'fallback_prediction_file_age_sec': fallback.get('prediction_file_age_sec'),
        'fallback_prediction_file_fresh': fallback.get('prediction_file_fresh'),
        'fallback_prediction_payload_present': fallback.get('prediction_payload_present'),
    }


def profile_config_rows(profile: str) -> dict[str, dict[str, Any]]:
    rows = load_json(PROFILE_CONFIG_MAP[profile])
    return {str(row.get('name') or ''): row for row in rows if isinstance(row, dict) and str(row.get('name') or '').strip()}


def source_config_row(profile: str, name: str) -> dict[str, Any] | None:
    return profile_config_rows(profile).get(name)


def load_guard_file(group: str) -> dict[str, Any]:
    path = GUARD_DIR / f'runtime_trade_guards_{group}.json'
    if not path.exists():
        return {}
    try:
        return load_json(path)
    except Exception:
        return {}


def guard_trader_row(guard_payload: dict[str, Any], trader_name: str) -> dict[str, Any] | None:
    traders = (((guard_payload or {}).get('guard_evidence_v1') or {}).get('traders') or [])
    for row in traders:
        if isinstance(row, dict) and str(row.get('traderName') or '') == trader_name:
            return row
    return None


def decision_cells(guard_payload: dict[str, Any], trader_name: str, symbol: str) -> list[dict[str, Any]]:
    cells = (((guard_payload or {}).get('execution_decisions_v1') or {}).get('cells') or [])
    out = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        if str(cell.get('traderName') or '') != trader_name:
            continue
        if str(cell.get('symbol') or '').upper() != symbol.upper():
            continue
        out.append(cell)
    return out


def likely_guard_signals(trader_guard: dict[str, Any] | None, decision_cells_for_trader: list[dict[str, Any]]) -> list[str]:
    signals: list[str] = []
    if trader_guard:
        if int(trader_guard.get('abstainedByMarketSelection') or 0) > 0:
            signals.append('preselector_abstain')
        if int(trader_guard.get('skippedByThresholdDrift') or 0) > 0:
            signals.append('threshold_drift')
        if int(trader_guard.get('skippedByExpectancyGate') or 0) > 0:
            signals.append('expectancy_gate')
        if int(trader_guard.get('skippedBySelector') or 0) > 0:
            signals.append('selector_gate')
        if int(trader_guard.get('skippedByRegime') or 0) > 0:
            signals.append('regime_gate')
        if int(trader_guard.get('skippedByShockRisk') or 0) > 0:
            signals.append('shock_risk')
        if int(trader_guard.get('skippedByComboPause') or 0) > 0:
            signals.append('combo_pause')
        if int(trader_guard.get('skippedByDownBreaker') or 0) > 0:
            signals.append('down_breaker')
        if int(trader_guard.get('skippedByUpRisk') or 0) > 0:
            signals.append('up_risk')
        if int(trader_guard.get('skippedByExtremeMarket') or 0) > 0:
            signals.append('extreme_market')
        if int(trader_guard.get('skippedByDirectionLossCooldown') or 0) > 0:
            signals.append('direction_loss_cooldown')
        if int(trader_guard.get('skippedByDrawdownAcceleration') or 0) > 0:
            signals.append('drawdown_acceleration')
    for cell in decision_cells_for_trader:
        reason = str(cell.get('primaryBlockReason') or '')
        if reason.startswith('threshold_') and 'threshold' not in signals:
            signals.append(reason)
        if reason == 'data_quality_halt' and 'data_quality' not in signals:
            signals.append('data_quality')
        if str(cell.get('decisionStatus') or '') == 'abstained' and 'preselector_abstain' not in signals:
            signals.append('preselector_abstain')
    return signals


def source_runtime_trader_name(source_row: dict[str, Any], symbol: str) -> str:
    base = str(source_row.get('name') or '')
    split = f'{base}__simulation_{symbol.lower()}'
    return split


def load_trade_count(logs_dir: str, symbol: str) -> tuple[bool, int]:
    runtime_dir = POLY / logs_dir
    sim_path = runtime_dir / 'prediction_trades.simulation.json'
    if not sim_path.exists():
        split = POLY / f'{logs_dir}__simulation_{symbol.lower()}' / 'prediction_trades.simulation.json'
        sim_path = split
    if not sim_path.exists():
        return False, 0
    try:
        payload = load_json(sim_path)
    except Exception:
        return True, 0
    if not isinstance(payload, list):
        return True, 0
    return True, len([row for row in payload if isinstance(row, dict) and str(row.get('symbol') or '').upper() == symbol.upper()])


def pending_order_count(logs_dir: str, symbol: str) -> int:
    runtime_dir = POLY / logs_dir
    sim_path = runtime_dir / 'pending_sim_orders.simulation.json'
    if not sim_path.exists():
        split = POLY / f'{logs_dir}__simulation_{symbol.lower()}' / 'pending_sim_orders.simulation.json'
        sim_path = split
    if not sim_path.exists():
        return 0
    try:
        payload = load_json(sim_path)
    except Exception:
        return 0
    if not isinstance(payload, list):
        return 0
    return len([row for row in payload if isinstance(row, dict) and str(row.get('symbol') or symbol).upper() == symbol.upper()])


def first_trade_limit_info(logs_dir: str, symbol: str, selection_mode: str, finalist: dict[str, Any]) -> tuple[Any, Any]:
    runtime_dir = POLY / logs_dir
    sim_path = runtime_dir / 'prediction_trades.simulation.json'
    if not sim_path.exists():
        split = POLY / f'{logs_dir}__simulation_{symbol.lower()}' / 'prediction_trades.simulation.json'
        sim_path = split
    if not sim_path.exists():
        return None, None
    try:
        payload = load_json(sim_path)
    except Exception:
        return None, None
    rows = [row for row in payload if isinstance(row, dict) and str(row.get('symbol') or '').upper() == symbol.upper()]
    if not rows:
        return None, None
    first = rows[0]
    got = first.get('limitPriceConfigured')
    if selection_mode == 'fixed_price':
        expected = round(float(finalist.get('selected_buy_price') or 0.0), 2)
        return got, (round(float(got or 0.0), 2) == expected)
    levels = [round(0.30 + 0.01 * i, 2) for i in range(15)]
    if got is None:
        return got, False
    return got, (round(float(got), 2) in levels)


def classify_row(source: dict[str, Any], trade_count: int, pending_count: int, guard_signals: list[str]) -> str:
    if trade_count > 0:
        return 'has_runtime_trades'
    if pending_count > 0:
        return 'queued_no_fill'
    if not source.get('prediction_file_exists') or not source.get('prediction_file_fresh'):
        return 'stale_or_missing'
    if not source.get('prediction_payload_present'):
        return 'prediction_empty'
    if 'preselector_abstain' in guard_signals:
        return 'preselector_abstain'
    if guard_signals:
        return 'runtime_guard_blocked'
    if source.get('latest_should_trade') is False:
        return 'no_signal'
    if source.get('latest_should_trade') is True:
        return 'should_trade_no_order_observed'
    return 'no_signal'


def classify_source_alignment(classification: str, source_guard_signals: list[str], source_payload_present: bool, source_fresh: bool) -> str:
    if classification == 'preselector_abstain':
        if 'preselector_abstain' in source_guard_signals:
            return 'source_same_cycle_preselector_abstain'
        if source_guard_signals:
            return f'different_source_guard:{",".join(source_guard_signals)}'
        return 'source_guard_unknown'
    if classification == 'prediction_empty':
        if source_payload_present:
            return 'lowprice_prediction_empty_only'
        if source_fresh:
            return 'matches_source_prediction_empty'
        return 'source_prediction_stale_or_missing'
    return 'not_applicable'


def finalists_map(optimization: dict[str, Any], profile: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in optimization['profiles'][profile]:
        for finalist in row.get('runtime_finalists') or []:
            out[(row['source_trader'], row['symbol'], finalist['finalist_key'])] = finalist
    return out


def audit_profile(profile: str, optimization: dict[str, Any]) -> dict[str, Any]:
    config_path = PROFILES[profile]['out_config']
    if not config_path.exists():
        return {
            'profile': profile,
            'group': PROFILES[profile]['group'],
            'pid_count': 0,
            'classification_counts': {'missing_out_config': 1},
            'sample_trade_checks': [],
        }
    rows = load_json(config_path)
    group = PROFILES[profile]['group']
    group_guard = load_guard_file(group)
    finalists = finalists_map(optimization, profile)
    sample_trade_checks = []
    class_counts: dict[str, int] = {}
    for row in rows:
        symbol = str(row.get('lowPriceSymbol') or '').upper()
        source_trader = str(row.get('lowPriceSourceTrader') or '')
        finalist = finalists.get((source_trader, symbol, str(row.get('lowPriceFinalistKey') or '')), {})
        source = inspect_prediction_source_for_row(row, symbol)
        trader_guard = guard_trader_row(group_guard, str(row.get('name') or ''))
        decs = decision_cells(group_guard, str(row.get('name') or ''), symbol)
        guard_signals = likely_guard_signals(trader_guard, decs)
        has_sim_log, trade_count = load_trade_count(str(row.get('logsDir') or ''), symbol)
        pending_count = pending_order_count(str(row.get('logsDir') or ''), symbol)
        source_profile = str(row.get('lowPriceSourceProfile') or profile)
        source_name = str(row.get('lowPriceSourceTrader') or '')
        source_row = source_config_row(source_profile, source_name)
        source_group = str(source_row.get('group') or '') if isinstance(source_row, dict) else ''
        source_guard = load_guard_file(source_group) if source_group else {}
        source_runtime_name = source_runtime_trader_name(source_row, symbol) if isinstance(source_row, dict) else ''
        source_guard_row = guard_trader_row(source_guard, source_runtime_name) if source_runtime_name else None
        source_decisions = decision_cells(source_guard, source_runtime_name, symbol) if source_runtime_name else []
        source_guard_signals = likely_guard_signals(source_guard_row, source_decisions)
        source_truth = inspect_prediction_source_for_row(source_row, symbol) if isinstance(source_row, dict) else {
            'prediction_file_exists': False,
            'prediction_file_fresh': False,
            'prediction_payload_present': False,
        }
        classification = classify_row(source, trade_count, pending_count, guard_signals)
        class_counts[classification] = class_counts.get(classification, 0) + 1
        first_limit, first_limit_check = first_trade_limit_info(str(row.get('logsDir') or ''), symbol, str(row.get('lowPriceSelectionMode') or ''), finalist)
        sample_trade_checks.append({
            'name': row.get('name'),
            'logsDir': row.get('logsDir'),
            'selectionMode': row.get('lowPriceSelectionMode'),
            'trade_count': trade_count,
            'pending_count': pending_count,
            'has_sim_log': has_sim_log,
            'classification': classification,
            'recent_guard_signals': guard_signals,
            'source_profile': source_profile,
            'source_group': source_group,
            'source_runtime_trader_name': source_runtime_name,
            'source_recent_guard_signals': source_guard_signals,
            'source_guard_alignment': classify_source_alignment(classification, source_guard_signals, bool(source_truth.get('prediction_payload_present')), bool(source_truth.get('prediction_file_fresh'))),
            'first_trade_limitPriceConfigured': first_limit,
            'first_trade_limit_check': first_limit_check,
            **source,
            'source_prediction_file': source_truth.get('prediction_file'),
            'source_prediction_file_exists': source_truth.get('prediction_file_exists'),
            'source_prediction_file_fresh': source_truth.get('prediction_file_fresh'),
            'source_prediction_payload_present': source_truth.get('prediction_payload_present'),
        })
    return {
        'profile': profile,
        'group': group,
        'pid_count': pgrep_count(f'dist/multi_prediction_index.js --group {group}'),
        'classification_counts': class_counts,
        'sample_trade_checks': sample_trade_checks,
    }


def main() -> None:
    optimization = load_json(OPT_REPORT)
    payload = {
        'generatedAt': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'profiles': {profile: audit_profile(profile, optimization) for profile in ('default', '70')},
    }
    dump_json(REPORT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
