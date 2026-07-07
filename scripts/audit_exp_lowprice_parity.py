#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
POLY = ROOT / 'polymarket'
REPORT = ROOT / 'reports' / 'exp_lowprice_parity_audit_latest.json'
RUNTIME_AUDIT = ROOT / 'reports' / 'exp_lowprice_runtime_audit_latest.json'

from scripts.exp_lowprice_selected_common import resolve_prediction_suffix
from scripts.exp_lowprice_selected_common import iter_fixed_dynamic_compare_rows

PROFILE_MAP = {
    'default': {
        'source_config': POLY / 'trader_configs.json',
        'clone_config': POLY / 'trader_configs_monitor_only_lowprice.json',
        'clone_active': POLY / 'active_traders_monitor_only_lowprice.json',
        'launch_log': ROOT / 'logs' / 'lowprice_default_selected.launchd.log',
    },
    '70': {
        'source_config': POLY / 'trader_configs_70.json',
        'clone_config': POLY / 'trader_configs_monitor_only_lowprice_70.json',
        'clone_active': POLY / 'active_traders_monitor_only_lowprice_70.json',
        'launch_log': ROOT / 'logs' / 'lowprice_70_selected.launchd.log',
    },
}

EXPECTED_OVERRIDE_KEYS = {
    'name', 'group', 'profile', 'role', 'initialCapital', 'perCoinCapital', 'runtimeSeedCapitalFromReports',
    'runtimeIndependentCellCapital', 'allowedMarkets', 'runtimeSplitBySymbol', 'numTradingAssets',
    'predictionSuffix', 'predictionSuffixBySymbol', 'predictionFallbackSuffix', 'rulesJsonPath',
    'disablePostOnlyShadowTracking',
    'lowPriceSourceTrader', 'lowPriceSourceProfile', 'lowPriceSymbol',
    'lowPriceSelectionMode', 'lowPriceSelectedBuyPrice', 'lowPriceSelectedBuyPriceRange', 'lowPriceBounds',
    'lowPriceExperiment', 'lowPriceSourceAllowedMarkets', 'lowPriceSourceInitialCapital',
    'lowPriceExpectedCloneInitialCapital', 'limitPrice', 'limitPriceLadder', 'usePredictionLimitPrice', 'logsDir',
    'lowPriceBypassMarketSelection', 'lowPriceBypassThresholdBase', 'lowPriceSourceResolvedBuyPrice', 'lowPriceSourceResolvedBuyPriceRange',
    'lowPriceSourceSizingReferencePrice', 'lowPriceFinalistRank', 'lowPriceFinalistKey',
    'lowPriceDynamicMinTotalAmountUsd',
}

CRITICAL_PARITY_KEYS = [
    'probThreshold', 'minEdge', 'kellyFrac', 'betPctNormal', 'betPctConservative', 'confTier1Bound',
    'confTier2Bound', 'tier1Mult', 'tier2Mult', 'tier3Mult', 'cooldownBars', 'drawdownHalt',
    'selectorMode', 'selectorModeBySymbol', 'selectorEligibleBySymbol', 'selectorScopeBySymbol',
    'regimeMode', 'regimeModeBySymbol', 'calibrationMode', 'calibrationModeBySymbolDirection',
    'calibrationStatsMode', 'calibrationCheckSeconds', 'expectancyGateMode', 'thresholdDriftMode',
    'metaLabelMode', 'jointOptimizationMode',
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def now_iso() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def value_equal(a: Any, b: Any) -> bool:
    return a == b


def parse_allowed_markets(raw: Any) -> list[str]:
    if isinstance(raw, list):
        vals = [str(x).strip().upper() for x in raw if str(x).strip()]
    else:
        vals = [part.strip().upper() for part in str(raw or '').split(',') if part.strip()]
    return sorted(set(v for v in vals if v))


def expected_initial_for_clone(source: dict[str, Any], symbol: str) -> float:
    initial = round(float(source.get('initialCapital') or 0.0), 2)
    markets = parse_allowed_markets(source.get('allowedMarkets'))
    if bool(source.get('perCoinCapital')) and symbol and len(markets) > 1:
        return round(initial / len(markets), 2)
    return initial


def load_runtime_sample_lookup() -> dict[str, Any]:
    if not RUNTIME_AUDIT.exists():
        return {}
    payload = load_json(RUNTIME_AUDIT)
    lookup: dict[str, Any] = {}
    for profile_data in (payload.get('profiles') or {}).values():
        for sample in profile_data.get('sample_trade_checks') or []:
            lookup[str(sample.get('name') or '')] = sample
    return lookup


def runtime_diagnosis(clone: dict[str, Any], runtime_sample_lookup: dict[str, Any]) -> dict[str, Any]:
    sample = runtime_sample_lookup.get(str(clone.get('name') or ''), {})
    return {
        'classification': sample.get('classification'),
        'recent_guard_signals': sample.get('recent_guard_signals'),
        'source_guard_alignment': sample.get('source_guard_alignment'),
        'trade_count': sample.get('trade_count'),
        'first_trade_limit_check': sample.get('first_trade_limit_check'),
        'first_trade_limitPriceConfigured': sample.get('first_trade_limitPriceConfigured'),
        'prediction_resolution': sample.get('prediction_resolution'),
        'prediction_file_exists': sample.get('prediction_file_exists'),
        'prediction_file_fresh': sample.get('prediction_file_fresh'),
        'prediction_payload_present': sample.get('prediction_payload_present'),
    }


def selection_identity(source_trader: str, symbol: str, selection_mode: str, selected_buy_price: Any, selected_buy_price_range: Any) -> tuple[Any, ...]:
    return (
        str(source_trader or ''),
        str(symbol or '').upper(),
        str(selection_mode or ''),
        round(float(selected_buy_price), 2) if selected_buy_price is not None else None,
        tuple(round(float(x), 2) for x in (selected_buy_price_range or [])),
    )


def audit_profile(profile: str, runtime_sample_lookup: dict[str, Any]) -> dict[str, Any]:
    meta = PROFILE_MAP[profile]
    if not meta['clone_config'].exists():
        return {
            'profile': profile,
            'clone_count': 0,
            'issue_count': 1,
            'issue_counts': {'missing_clone_config': 1},
            'mode_registry_warning_present': False,
            'rows': [],
        }
    sources = load_json(meta['source_config'])
    clones = load_json(meta['clone_config'])
    source_map = {str(r.get('name') or ''): r for r in sources if isinstance(r, dict)}
    rows = []
    issue_counts: dict[str, int] = {}
    issue_count = 0
    fixed_dynamic_expected = {
        selection_identity(row['source_trader'], row['symbol'], row['selection_mode'], row.get('selected_buy_price'), row.get('selected_buy_price_range'))
        for row in iter_fixed_dynamic_compare_rows(profile)
    }
    fixed_dynamic_rows: list[dict[str, Any]] = []
    for clone in clones:
        source_name = str(clone.get('lowPriceSourceTrader') or '')
        symbol = str(clone.get('lowPriceSymbol') or clone.get('allowedMarkets') or '').strip().upper()
        source = source_map.get(source_name)
        issues: list[str] = []
        if not source:
            issues.append('source_trader_missing')
            rows.append({'name': clone.get('name'), 'issues': issues})
            issue_count += len(issues)
            continue
        changed: dict[str, dict[str, Any]] = {}
        for key in sorted(set(source.keys()) | set(clone.keys())):
            a = source.get(key, '__MISSING__')
            b = clone.get(key, '__MISSING__')
            if not value_equal(a, b):
                changed[key] = {'source': a, 'clone': b}
        unexpected = sorted(k for k in changed.keys() if k not in EXPECTED_OVERRIDE_KEYS)
        if unexpected:
            issues.extend([f'unexpected_diff:{k}' for k in unexpected])
        for key in CRITICAL_PARITY_KEYS:
            if source.get(key) != clone.get(key):
                issues.append(f'critical_parity_drift:{key}')

        expected_suffix = resolve_prediction_suffix(source, symbol)
        clone_suffix = str(clone.get('predictionSuffix') or '').strip()
        if not clone_suffix:
            issues.append('prediction_suffix_missing')
        if clone.get('predictionSuffixBySymbol') != {symbol: expected_suffix}:
            issues.append('prediction_suffix_by_symbol_mismatch')
        if str(clone.get('allowedMarkets') or '').strip().upper() != symbol:
            issues.append('allowed_markets_symbol_mismatch')
        if round(float(clone.get('initialCapital') or 0.0), 2) != expected_initial_for_clone(source, symbol):
            issues.append('initial_capital_semantics_mismatch')
        if profile == '70' and clone_suffix and '_70' not in clone_suffix:
            issues.append('profile70_suffix_not_70')
        if clone.get('runtimeIndependentCellCapital') is not True:
            issues.append('independent_cell_capital_missing')
        if clone.get('perCoinCapital') is not False:
            issues.append('per_coin_capital_should_be_false')
        if clone.get('runtimeSeedCapitalFromReports') is not False:
            issues.append('runtime_seed_from_reports_should_be_false')
        if clone.get('lowPriceBypassMarketSelection') is not True:
            issues.append('market_selection_bypass_missing')
        if clone.get('lowPriceBypassThresholdBase') is not True:
            issues.append('threshold_base_bypass_missing')
        if clone.get('lowPriceSelectionMode') == 'fixed_price':
            if clone.get('limitPrice') is None or clone.get('limitPriceLadder') not in (None, ''):
                issues.append('fixed_mode_price_fields_invalid')
            if clone.get('lowPriceDynamicMinTotalAmountUsd') not in (None, ''):
                issues.append('fixed_mode_dynamic_min_total_set')
        elif clone.get('lowPriceSelectionMode') == 'dynamic_range':
            if clone.get('limitPrice') is not None:
                issues.append('dynamic_mode_limit_price_invalid')
            if str(clone.get('limitPriceLadder') or '') != '0.30,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39,0.40,0.41,0.42,0.43,0.44':
                issues.append('dynamic_mode_ladder_invalid')
            if round(float(clone.get('lowPriceDynamicMinTotalAmountUsd') or 0.0), 2) != 15.0:
                issues.append('dynamic_mode_min_total_invalid')
        diag = runtime_diagnosis(clone, runtime_sample_lookup)
        if selection_identity(source_name, symbol, clone.get('lowPriceSelectionMode'), clone.get('lowPriceSelectedBuyPrice'), clone.get('lowPriceSelectedBuyPriceRange')) in fixed_dynamic_expected:
            fixed_dynamic_rows.append({
                'name': clone.get('name'),
                'source_trader': source_name,
                'symbol': symbol,
                'selection_mode': clone.get('lowPriceSelectionMode'),
                'selected_buy_price': clone.get('lowPriceSelectedBuyPrice'),
                'selected_buy_price_range': clone.get('lowPriceSelectedBuyPriceRange'),
            })
        rows.append({
            'profile': profile,
            'name': clone.get('name'),
            'source_trader': source_name,
            'symbol': symbol,
            'selection_mode': clone.get('lowPriceSelectionMode'),
            'expected_structural_overrides': sorted(k for k in changed.keys() if k in EXPECTED_OVERRIDE_KEYS),
            'unexpected_diffs': unexpected,
            'issues': issues,
            'runtime_diagnosis': diag,
        })
        for issue in issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        issue_count += len(issues)

    launch_log = meta['launch_log']
    mode_registry_warning = False
    if launch_log.exists():
        try:
            mode_registry_warning = 'mode registry 警告' in launch_log.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            mode_registry_warning = False
    return {
        'profile': profile,
        'clone_count': len(rows),
        'issue_count': issue_count,
        'issue_counts': dict(sorted(issue_counts.items())),
        'mode_registry_warning_present': mode_registry_warning,
        'fixed_dynamic_compare_rows_total': len(fixed_dynamic_rows),
        'fixed_dynamic_compare_rows': fixed_dynamic_rows,
        'rows': rows,
    }


def main() -> None:
    runtime_sample_lookup = load_runtime_sample_lookup()
    profiles = {
        'default': audit_profile('default', runtime_sample_lookup),
        '70': audit_profile('70', runtime_sample_lookup),
    }
    issue_count_total = sum(int(v.get('issue_count') or 0) for v in profiles.values())
    fixed_dynamic_compare_rows_total = sum(int(v.get('fixed_dynamic_compare_rows_total') or 0) for v in profiles.values())
    ok = all((profiles[p]['issue_count'] == 0) for p in profiles)
    generated_at = now_iso()
    payload = {
        'generatedAt': generated_at,
        'generated_at': generated_at,
        'profiles': profiles,
        'status': 'ok' if ok else 'warning',
        'summary': {
            'open_p0': 0,
            'open_p1': 0 if ok else 1,
            'issue_count': issue_count_total,
            'profiles_total': len(profiles),
            'fixed_dynamic_compare_rows_total': fixed_dynamic_compare_rows_total,
            'current_conclusion': (
                'lowprice_clone_parity_is_consistent'
                if ok
                else 'lowprice_clone_parity_has_open_gaps'
            ),
        },
    }
    payload['ok'] = ok
    payload['current_conclusion'] = (
        'lowprice_clone_parity_is_consistent'
        if ok
        else 'lowprice_clone_parity_has_open_gaps'
    )
    dump_json(REPORT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
