#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.exp_lowprice_selected_common import (
    DYNAMIC_FULL_LADDER_RANGE,
    PRICE_BOUNDS,
    PRICE_LEVELS,
    PROFILES,
    SELECTED_CELLS,
    dump_json,
    expected_clone_initial_capital,
    format_ladder_from_levels,
    iter_fixed_dynamic_compare_rows,
    iter_selected_baselines,
    load_json,
    resolve_baseline,
    resolve_prediction_suffix,
)

POLY = ROOT / 'polymarket'
OPT_REPORT = ROOT / 'reports' / 'exp_lowprice_selected_optimization_latest.json'
MATERIALIZED = ROOT / 'reports' / 'exp_lowprice_matrix_materialized_latest.json'
REPORT = ROOT / 'reports' / 'exp_lowprice_prelaunch_audit_latest.json'
SOURCE_MAX_AGE_SEC = 1800
FULL_LADDER = format_ladder_from_levels(PRICE_LEVELS)
BLOCKING_ISSUE_TYPES = {
    'missing_out_config',
    'missing_baseline',
    'missing_selection_source',
    'duplicate_name',
    'duplicate_logs',
    'missing_prediction_file',
    'missing_rules_file',
}

ALLOWED_DIFF_KEYS = {
    'name', 'group', 'profile', 'role', 'logsDir', 'initialCapital', 'predictionSuffix', 'predictionSuffixBySymbol',
    'predictionFallbackSuffix',
    'disablePostOnlyShadowTracking',
    'rulesJsonPath', 'allowedMarkets', 'runtimeSplitBySymbol', 'numTradingAssets', 'limitPrice',
    'limitPriceLadder', 'usePredictionLimitPrice', 'perCoinCapital', 'runtimeSeedCapitalFromReports',
    'runtimeIndependentCellCapital', 'lowPriceSourceTrader',
    'lowPriceSourceProfile', 'lowPriceSymbol', 'lowPriceSelectionMode', 'lowPriceSelectedBuyPrice',
    'lowPriceSelectedBuyPriceRange', 'lowPriceBounds', 'lowPriceExperiment', 'lowPriceSourceAllowedMarkets',
    'lowPriceSourceInitialCapital', 'lowPriceExpectedCloneInitialCapital', 'lowPriceBypassMarketSelection',
    'lowPriceBypassThresholdBase',
    'lowPriceSourceResolvedBuyPrice', 'lowPriceSourceResolvedBuyPriceRange', 'lowPriceSourceSizingReferencePrice',
    'lowPriceFinalistRank', 'lowPriceFinalistKey', 'lowPriceDynamicMinTotalAmountUsd',
}


def classify_selection_source_gap(row: dict[str, Any]) -> str:
    scientific_status = str(row.get('scientificEligibilityStatus') or '').strip().lower()
    disabled_reason = str(row.get('disabledReason') or '').strip().lower()
    if scientific_status == 'disabled_by_operator' or disabled_reason.startswith('disabled_by_operator'):
        return 'archival_disabled_without_selection_source'
    cutover_status = str(row.get('lowpriceCutoverStatus') or '').strip()
    live_consumption_status = str(row.get('liveConsumptionStatus') or '').strip()
    prediction_surface_choice = str(row.get('predictionSurfaceChoice') or '').strip().lower()
    if (
        prediction_surface_choice == 'base'
        and (
            cutover_status == 'lowprice_candidate_missing_for_cutover'
            or live_consumption_status == 'lowprice_candidate_missing_for_cutover'
        )
    ):
        return 'live_base_surface_without_selection_source'
    return 'missing_selection_source'


def resolve_prediction_target(primary_suffix: str, fallback_suffix: str | None) -> tuple[Path, Path | None, Path | None]:
    primary = POLY / f"predictions{primary_suffix}.json"
    fallback = POLY / f"predictions{fallback_suffix}.json" if fallback_suffix else None
    now = time.time()
    if primary.exists() and (now - primary.stat().st_mtime) <= SOURCE_MAX_AGE_SEC:
        return primary, primary, fallback
    if fallback and fallback.exists() and (now - fallback.stat().st_mtime) <= SOURCE_MAX_AGE_SEC:
        return fallback, primary, fallback
    return primary, primary, fallback


def normalize(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k not in ALLOWED_DIFF_KEYS}


def finalists_map(optimization: dict[str, Any], profile: str) -> dict[tuple[str, str, str], dict[str, Any]]:
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in optimization['profiles'][profile]:
        for finalist in row.get('runtime_finalists') or []:
            out[(row['source_trader'], row['symbol'], finalist['finalist_key'])] = finalist
    return out


def selection_identity(source_trader: str, symbol: str, selection_mode: str, selected_buy_price: Any, selected_buy_price_range: Any) -> tuple[Any, ...]:
    return (
        str(source_trader or ''),
        str(symbol or '').upper(),
        str(selection_mode or ''),
        round(float(selected_buy_price), 2) if selected_buy_price is not None else None,
        tuple(round(float(x), 2) for x in (selected_buy_price_range or [])),
    )


def audit_profile(profile: str, optimization: dict[str, Any], materialized: dict[str, Any], baselines: dict[tuple[str, str, str], dict[str, Any]]) -> dict[str, Any]:
    config_path = PROFILES[profile]['out_config']
    if not config_path.exists():
        return {
            'profile': profile,
            'clone_rows': 0,
            'selected_cells': len(SELECTED_CELLS[profile]),
            'issue_count': 1,
            'blocking_issue_count': 1,
            'issues': [{'type': 'missing_out_config', 'file': str(config_path)}],
            'materialized_summary': materialized.get('profiles', {}).get(profile),
        }
    rows = load_json(config_path)
    issues: list[dict[str, Any]] = []
    optimization_finalists = {
        selection_identity(row['source_trader'], row['symbol'], finalist['selection_mode'], finalist.get('selected_buy_price'), finalist.get('selected_buy_price_range')): finalist
        for row in optimization['profiles'][profile]
        for finalist in (row.get('runtime_finalists') or [])
    }
    fixed_dynamic_rows = {
        selection_identity(row['source_trader'], row['symbol'], row['selection_mode'], row.get('selected_buy_price'), row.get('selected_buy_price_range')): row
        for row in iter_fixed_dynamic_compare_rows(profile)
    }
    expected_finalists = len(optimization_finalists)
    expected_rows = len(set(optimization_finalists.keys()) | set(fixed_dynamic_rows.keys()))
    if len(rows) != expected_rows:
        issues.append({'type': 'materialized_row_count_mismatch', 'expected': expected_rows, 'got': len(rows)})

    seen_names = set()
    seen_logs = set()
    for row in rows:
        source_trader = str(row.get('lowPriceSourceTrader') or '')
        symbol = str(row.get('lowPriceSymbol') or '').upper()
        baseline = baselines.get((profile, source_trader, symbol))
        name = str(row.get('name') or '')
        logs = str(row.get('logsDir') or '')
        selection_mode = str(row.get('lowPriceSelectionMode') or '')
        identity = selection_identity(
            source_trader,
            symbol,
            selection_mode,
            row.get('lowPriceSelectedBuyPrice'),
            row.get('lowPriceSelectedBuyPriceRange'),
        )
        finalist = optimization_finalists.get(identity)
        fixed_dynamic = fixed_dynamic_rows.get(identity)
        expected_selection = fixed_dynamic or finalist
        forced_row = bool(row.get('forcedLowpriceMaterialization'))

        if not baseline:
            if forced_row:
                issues.append({
                    'type': 'forced_row_without_selected_baseline',
                    'name': name,
                    'source_trader': source_trader,
                    'symbol': symbol,
                })
                continue
            issues.append({'type': 'missing_baseline', 'name': name, 'source_trader': source_trader, 'symbol': symbol})
            continue
        if not expected_selection:
            issues.append({
                'type': classify_selection_source_gap(row),
                'name': name,
                'source_trader': source_trader,
                'symbol': symbol,
                'scientificEligibilityStatus': row.get('scientificEligibilityStatus'),
                'disabledReason': row.get('disabledReason'),
                'lowpriceCutoverStatus': row.get('lowpriceCutoverStatus'),
                'liveConsumptionStatus': row.get('liveConsumptionStatus'),
                'predictionSurfaceChoice': row.get('predictionSurfaceChoice'),
            })
            continue
        if name in seen_names:
            issues.append({'type': 'duplicate_name', 'name': name})
        seen_names.add(name)
        if logs in seen_logs:
            issues.append({'type': 'duplicate_logs', 'logsDir': logs})
        seen_logs.add(logs)

        if normalize(row) != normalize(baseline['source_row']):
            issues.append({'type': 'unexpected_field_drift', 'name': name})

        expected_initial = expected_clone_initial_capital(baseline['source_row'])
        got_initial = round(float(row.get('initialCapital') or 0), 2)
        if got_initial != expected_initial:
            issues.append({'type': 'initial_capital_mismatch', 'name': name, 'expected': expected_initial, 'got': got_initial})

        expected_primary_suffix = resolve_prediction_suffix(baseline['source_row'], symbol)
        expected_generic_suffix = str(baseline['source_row'].get('predictionSuffix') or '').strip() or expected_primary_suffix
        expected_fallback_suffix = expected_generic_suffix if expected_generic_suffix != expected_primary_suffix else None
        clone_suffix = str(row.get('predictionSuffix') or '').strip()
        if clone_suffix != expected_generic_suffix:
            issues.append({'type': 'prediction_suffix_mismatch', 'name': name, 'expected': expected_generic_suffix, 'got': clone_suffix})
        if row.get('predictionSuffixBySymbol') != {symbol: expected_primary_suffix}:
            issues.append({
                'type': 'prediction_suffix_by_symbol_mismatch',
                'name': name,
                'expected': {symbol: expected_primary_suffix},
                'got': row.get('predictionSuffixBySymbol'),
            })
        got_fallback = str(row.get('predictionFallbackSuffix') or '').strip() or None
        if got_fallback != expected_fallback_suffix:
            issues.append({
                'type': 'prediction_fallback_suffix_mismatch',
                'name': name,
                'expected': expected_fallback_suffix,
                'got': got_fallback,
            })
        pred_file, primary_file, fallback_file = resolve_prediction_target(expected_primary_suffix, expected_fallback_suffix)
        if not pred_file.exists():
            issues.append({
                'type': 'missing_prediction_file',
                'name': name,
                'file': str(primary_file),
                'fallback_file': str(fallback_file) if fallback_file else None,
            })
        else:
            age = time.time() - pred_file.stat().st_mtime
            if age > SOURCE_MAX_AGE_SEC:
                issues.append({
                    'type': 'stale_prediction_file',
                    'name': name,
                    'age_sec': round(age, 1),
                    'file': str(pred_file),
                })

        rules_path = Path(str(row.get('rulesJsonPath') or ''))
        if not rules_path.is_absolute():
            rules_path = ROOT / rules_path
        if not rules_path.exists():
            issues.append({'type': 'missing_rules_file', 'name': name, 'file': str(rules_path)})
            continue
        payload = load_json(rules_path)
        poly = payload.get('polymarket_constraints') if isinstance(payload.get('polymarket_constraints'), dict) else {}

        if row.get('runtimeSplitBySymbol') is not False:
            issues.append({'type': 'runtime_split_not_disabled', 'name': name})
        if str(row.get('allowedMarkets') or '').upper() != symbol:
            issues.append({'type': 'allowed_markets_mismatch', 'name': name, 'got': row.get('allowedMarkets')})
        if row.get('runtimeIndependentCellCapital') is not True:
            issues.append({'type': 'independent_cell_capital_missing', 'name': name})
        if row.get('perCoinCapital') is not False:
            issues.append({'type': 'per_coin_capital_should_be_false', 'name': name})
        if row.get('runtimeSeedCapitalFromReports') is not False:
            issues.append({'type': 'runtime_seed_from_reports_should_be_false', 'name': name})
        if row.get('lowPriceBounds') != PRICE_BOUNDS:
            issues.append({'type': 'bounds_mismatch', 'name': name, 'got': row.get('lowPriceBounds')})
        if row.get('lowPriceBypassMarketSelection') is not True:
            issues.append({'type': 'preselector_bypass_missing', 'name': name})
        if row.get('lowPriceBypassThresholdBase') is not True:
            issues.append({'type': 'threshold_base_bypass_missing', 'name': name})
        if round(float(row.get('lowPriceSourceSizingReferencePrice') or 0.0), 4) != round(float(baseline.get('resolved_source_sizing_reference_price') or 0.0), 4):
            issues.append({'type': 'source_sizing_reference_price_mismatch', 'name': name})

        if selection_mode == 'fixed_price':
            exp_price = round(float(expected_selection['selected_buy_price']), 2)
            got = round(float(poly.get('buy_price') or 0), 2)
            if got != exp_price:
                issues.append({'type': 'rules_buy_price_mismatch', 'name': name, 'expected': exp_price, 'got': got})
            if poly.get('buy_price_range') not in (None, []):
                issues.append({'type': 'rules_range_should_be_null', 'name': name})
            if round(float(row.get('limitPrice') or 0), 2) != exp_price:
                issues.append({'type': 'limit_price_mismatch', 'name': name, 'expected': exp_price, 'got': row.get('limitPrice')})
            if row.get('limitPriceLadder') not in (None, ''):
                issues.append({'type': 'limit_ladder_should_be_empty', 'name': name})
            if row.get('lowPriceDynamicMinTotalAmountUsd') not in (None, ''):
                issues.append({'type': 'fixed_dynamic_min_total_should_be_null', 'name': name})
        else:
            exp_range = [round(float(x), 2) for x in (expected_selection.get('selected_buy_price_range') or DYNAMIC_FULL_LADDER_RANGE)]
            got_range = poly.get('buy_price_range') if isinstance(poly.get('buy_price_range'), list) else None
            if got_range != exp_range:
                issues.append({'type': 'rules_range_mismatch', 'name': name, 'expected': exp_range, 'got': got_range})
            if row.get('limitPrice') not in (None, ''):
                issues.append({'type': 'dynamic_limit_price_should_be_null', 'name': name, 'got': row.get('limitPrice')})
            if str(row.get('limitPriceLadder') or '') != FULL_LADDER:
                issues.append({'type': 'dynamic_ladder_mismatch', 'name': name, 'expected': FULL_LADDER, 'got': row.get('limitPriceLadder')})
            if round(float(row.get('lowPriceDynamicMinTotalAmountUsd') or 0.0), 2) != 15.0:
                issues.append({'type': 'dynamic_min_total_amount_mismatch', 'name': name, 'got': row.get('lowPriceDynamicMinTotalAmountUsd')})

    return {
        'profile': profile,
        'clone_rows': len(rows),
        'expected_finalists': expected_finalists,
        'expected_materialized_rows': expected_rows,
        'selected_target': len(SELECTED_CELLS[profile]),
        'fixed_dynamic_compare_rows_total': len(fixed_dynamic_rows),
        'fixed_dynamic_compare_rows': list(fixed_dynamic_rows.values()),
        'issues': issues,
        'issue_count': len(issues),
        'blocking_issue_count': sum(1 for issue in issues if str(issue.get('type') or '') in BLOCKING_ISSUE_TYPES),
        'materialized_summary': materialized.get('profiles', {}).get(profile) if isinstance(materialized.get('profiles'), dict) else None,
    }


def main() -> None:
    optimization = load_json(OPT_REPORT)
    materialized = load_json(MATERIALIZED)
    baselines = {(b['profile'], b['source_trader'], b['symbol']): b for b in iter_selected_baselines()}
    for profile in PROFILES:
        for row in iter_fixed_dynamic_compare_rows(profile):
            key = (profile, str(row['source_trader']), str(row['symbol']).upper())
            if key not in baselines:
                baselines[key] = resolve_baseline(profile, row['source_trader'], row['symbol'])
    profiles = {profile: audit_profile(profile, optimization, materialized, baselines) for profile in ('default', '70')}
    issue_count_total = sum(int(v.get('issue_count') or 0) for v in profiles.values())
    blocking_issue_count_total = sum(int(v.get('blocking_issue_count') or 0) for v in profiles.values())
    fixed_dynamic_compare_rows_total = sum(int(v.get('fixed_dynamic_compare_rows_total') or 0) for v in profiles.values())
    fixed_dynamic_compare_rows = {
        profile: list(v.get('fixed_dynamic_compare_rows') or [])
        for profile, v in profiles.items()
    }
    ok = all(int(v.get('blocking_issue_count') or 0) == 0 for v in profiles.values())
    generated_at = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    payload = {
        'generatedAt': generated_at,
        'generated_at': generated_at,
        'status': 'ok' if ok else 'warning',
        'profiles': profiles,
        'ok': ok,
        'summary': {
            'open_p0': 0,
            'open_p1': 0 if ok else 1,
            'issue_count': issue_count_total,
            'blocking_issue_count': blocking_issue_count_total,
            'profiles_total': len(profiles),
            'fixed_dynamic_compare_rows_total': fixed_dynamic_compare_rows_total,
            'current_conclusion': (
                'lowprice_prelaunch_blocking_contract_is_clean_observational_drift_remains'
                if ok and issue_count_total > 0
                else 'lowprice_prelaunch_contract_is_consistent'
                if ok
                else 'lowprice_prelaunch_contract_has_open_blocking_gaps'
            ),
        },
        'current_conclusion': (
            'lowprice_prelaunch_blocking_contract_is_clean_observational_drift_remains'
            if ok and issue_count_total > 0
            else 'lowprice_prelaunch_contract_is_consistent'
            if ok
            else 'lowprice_prelaunch_contract_has_open_blocking_gaps'
        ),
        'fixed_dynamic_compare_rows_total': fixed_dynamic_compare_rows_total,
        'fixed_dynamic_compare_rows': fixed_dynamic_compare_rows,
    }
    dump_json(REPORT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if blocking_issue_count_total > 0:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
