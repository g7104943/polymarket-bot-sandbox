#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POLY = ROOT / 'polymarket'
REPORTS = ROOT / 'reports'
MODELS = ROOT / 'data' / 'models'
LOWPRICE_RULES_DIR = POLY / 'lowprice_rules_selected'
ARCHIVE_ROOT = POLY / 'archived_lowprice_resets'
PRICE_LEVELS = [round(0.30 + 0.01 * i, 2) for i in range(15)]
PRICE_BOUNDS = [0.30, 0.44]
DYNAMIC_FULL_LADDER_RANGE = [0.30, 0.44]
FIXED_DYNAMIC_COMPARE_RANGE = [0.30, 0.44]
FIXED_DYNAMIC_COMPARE_SOURCE_TRADER = 'v5_exp10_bp_dyn_0480_0510'
FIXED_FINALIST_COUNT = 2
MIN_HOLDOUT_TRADES = {
    'BTC': 50,
    'ETH': 50,
    'XRP': 30,
}
MIN_EXECUTION_ADJUSTED_FILL_RATIO = {
    'BTC': 0.10,
    'ETH': 0.10,
    'XRP': 0.03,
}
MIN_FILLABLE_RATE = {
    'BTC': 0.10,
    'ETH': 0.10,
    'XRP': 0.03,
}
XRP_1M_COVERAGE_THRESHOLD = 0.95
TOTAL_DAYS = 180
HOLDOUT_DAYS = 15
TRAIN_DAYS = TOTAL_DAYS - HOLDOUT_DAYS
INITIAL_CAPITAL = 400.0
SETTLEMENT_COST = 0.015 + 0.003
SIM_BET_PCT_NORMAL_CAP = 0.05
SIM_BET_PCT_CONSERVATIVE_CAP = 0.03
DYN_RE = re.compile(r'bp_dyn_(\d{4})_(\d{4})', re.IGNORECASE)
ASSET_NAME = {'BTC': 'BTC_USDT', 'ETH': 'ETH_USDT', 'XRP': 'XRP_USDT'}
SELECTOR_TARGET_SYMBOLS = {'BTC', 'ETH'}
CALIBRATION_TARGET_SYMBOLS = {'BTC', 'ETH'}
REGIME_TARGET_SYMBOLS = {'BTC', 'ETH'}
EXPECTANCY_TARGET_SYMBOLS = {'BTC', 'ETH'}
ASSET_1M_PATH = {
    'BTC': ROOT / 'data' / 'polymarket_1m_btc_usdt.parquet',
    'ETH': ROOT / 'data' / 'polymarket_1m_eth_usdt.parquet',
    'XRP': ROOT / 'data' / 'polymarket_1m_xrp_usdt.parquet',
}
ASSET_15M_RAW_PATH = {
    'BTC': ROOT / 'data' / 'raw' / 'btc_usdt_15m.parquet',
    'ETH': ROOT / 'data' / 'raw' / 'eth_usdt_15m.parquet',
    'XRP': ROOT / 'data' / 'raw' / 'xrp_usdt_15m.parquet',
}
BOOTSTRAP_REPORT = REPORTS / 'vnext_execution_v2_historical_bootstrap_latest.json'
REGIME_RUNTIME_DIR = POLY / 'logs' / 'runtime'
EXPECTANCY_RUNTIME_DIR = POLY / 'logs' / 'runtime'
DEFAULT_EXECUTION_DEFAULTS = {
    'fill_rate': 0.92,
    'partial_fill_rate': 0.08,
    'timeout_rate': 0.08,
    'avg_partial_fill_ratio': 0.48,
    'avg_queue_wait_seconds': 22.0,
}
PROFILES = {
    'default': {
        'config': POLY / 'trader_configs.json',
        'active': POLY / 'active_traders.json',
        'out_config': POLY / 'trader_configs_monitor_only_lowprice.json',
        'out_active': POLY / 'active_traders_monitor_only_lowprice.json',
        'out_monitor': POLY / 'monitor_only_traders_lowprice.json',
        'group': 'lowprice_default_selected',
        'logs_prefix': 'logs_lowprice_selected_',
        'cmp_prefix': 'cmp_lowprice_',
    },
    '70': {
        'config': POLY / 'trader_configs_70.json',
        'active': POLY / 'active_traders_70.json',
        'out_config': POLY / 'trader_configs_monitor_only_lowprice_70.json',
        'out_active': POLY / 'active_traders_monitor_only_lowprice_70.json',
        'out_monitor': POLY / 'monitor_only_traders_lowprice_70.json',
        'group': 'lowprice_70_selected',
        'logs_prefix': 'logs_70_lowprice_selected_',
        'cmp_prefix': 'cmp_lowprice_70_',
    },
}
RETIRED_LOWPRICE_BASELINES: dict[str, set[tuple[str, str]]] = {
    'default': {
        ('v5_exp15_bp0510', 'XRP'),
        ('v5_exp15_bp0520', 'XRP'),
        ('v5_exp10_bp0450', 'XRP'),
        ('v5_exp10_bp0460', 'XRP'),
        ('v5_exp10_bp0470', 'XRP'),
        ('v5_exp10_bp0480', 'XRP'),
        ('v5_exp10_bp0490', 'XRP'),
        ('v5_exp10_bp_dyn_0480_0510', 'ETH'),
    },
    '70': {
        ('v5_exp15_bp0510', 'XRP'),
        ('v5_exp15_bp0520', 'XRP'),
        ('v5_exp10_bp0500', 'BTC'),
    },
}
LOWPRICE_SELECTION_ALLOWLIST: dict[str, dict[tuple[str, str], set[tuple[str, float | None, tuple[float, ...]]]]] = {
    '70': {
        # 0.45-0.48 are ordinary profile-70 threshold simulations now.
        # Lowprice is reserved for buy prices below 0.44, and this source is
        # fully blocked from the lowprice materializer to prevent duplicate rows.
        ('v5_exp16_bp0530', 'ETH'): set(),
    },
}
RETIRED_MAINLINE_CELLS: dict[str, set[tuple[str, str]]] = {
    'default': {
        ('v5_exp10_bp0470', 'XRP'),
        ('v5_exp16_bp0530', 'BTC'),
    },
    '70': {
        ('v5_exp10_bp0470', 'XRP'),
        ('v5_exp16_bp0530', 'BTC'),
        ('v5_exp16_bp0530', 'ETH'),
    },
}
SELECTED_CELLS = {
    'default': [
        ('v5_exp16_bp0500', 'XRP'),
        ('v5_exp10_bp0470', 'BTC'),
        ('v5_exp10_bp0480', 'ETH'),
        ('v5_exp10_bp_dyn_0450_0530', 'BTC'),
        ('v5_exp10_bp0460', 'ETH'),
        ('v5_exp16_bp0520', 'BTC'),
        ('v5_exp16_bp0530', 'BTC'),
        ('v5_exp10_bp0490', 'ETH'),
        ('v5_exp10_bp0450', 'BTC'),
        ('v5_exp13_bp0530', 'BTC'),
        ('v5_exp14_bp0530', 'XRP'),
        ('v5_exp10_bp0490', 'BTC'),
    ],
    '70': [
        ('v5_exp13_bp0530', 'XRP'),
        ('v5_exp10_bp0470', 'BTC'),
        ('v5_exp16_bp0520', 'XRP'),
        ('v5_exp10_bp0520', 'BTC'),
        ('v5_exp10_bp_dyn_0450_0530', 'BTC'),
        ('v5_exp10_bp_dyn_0480_0510', 'BTC'),
        ('v5_exp10_bp0460', 'ETH'),
        ('v5_exp14_bp0530', 'ETH'),
        ('v5_exp10_bp0480', 'ETH'),
        ('v5_exp14_bp0530', 'BTC'),
    ],
}

FIXED_DYNAMIC_COMPARE_ROWS = {
    profile: [
        {
            'source_trader': FIXED_DYNAMIC_COMPARE_SOURCE_TRADER,
            'symbol': symbol,
            'selection_mode': 'dynamic_range',
            'selected_buy_price': None,
            'selected_buy_price_range': list(FIXED_DYNAMIC_COMPARE_RANGE),
            'finalist_rank': None,
            'finalist_key': 'fixed_dynamic_0300_0440',
            'selection_origin': 'fixed_dynamic_compare_row',
        }
        for symbol in ('BTC', 'ETH', 'XRP')
    ]
    for profile in PROFILES
}

FIXED_DYNAMIC_RUNTIME_ROW_OVERRIDES: dict[str, dict[tuple[str, str], dict[str, Any]]] = {
    '70': {
        ('v5_exp10_bp_dyn_0480_0510', 'ETH'): {
            'name': 'v5_exp10_bp_dyn_0300_0440_eth',
            'logsDir': 'logs_70_lowprice_selected_v5_exp10_bp_dyn_0300_0440_eth',
            'role': 'compare_only',
        },
    },
}


def is_retired_lowprice_baseline(profile: str, source_trader: str, symbol: str) -> bool:
    profile_key = str(profile or '').strip()
    source_trader_key = str(source_trader or '').strip()
    symbol_key = str(symbol or '').strip().upper()
    return (source_trader_key, symbol_key) in RETIRED_LOWPRICE_BASELINES.get(profile_key, set())


def _selection_price(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        return round(float(raw), 2)
    except Exception:
        return None


def _selection_range(raw: Any) -> tuple[float, ...]:
    if not isinstance(raw, list):
        return ()
    try:
        return tuple(round(float(item), 2) for item in raw)
    except Exception:
        return ()


def lowprice_selection_allowed(
    profile: str,
    source_trader: str,
    symbol: str,
    selection_mode: str,
    selected_buy_price: Any = None,
    selected_buy_price_range: Any = None,
) -> bool:
    profile_key = str(profile or '').strip()
    source_trader_key = str(source_trader or '').strip()
    symbol_key = str(symbol or '').strip().upper()
    allow_by_source = LOWPRICE_SELECTION_ALLOWLIST.get(profile_key, {})
    allowed = allow_by_source.get((source_trader_key, symbol_key))
    if allowed is None:
        return True
    identity = (
        str(selection_mode or '').strip() or 'fixed_price',
        _selection_price(selected_buy_price),
        _selection_range(selected_buy_price_range),
    )
    return identity in allowed


def is_retired_mainline_cell(profile: str, trader: str, symbol: str) -> bool:
    profile_key = str(profile or '').strip()
    trader_key = str(trader or '').strip()
    symbol_key = str(symbol or '').strip().upper()
    return (trader_key, symbol_key) in RETIRED_MAINLINE_CELLS.get(profile_key, set())


def is_retired_monitor_cell(
    profile: str,
    scope: str,
    trader: str,
    symbol: str,
    *,
    source_trader: str | None = None,
) -> bool:
    scope_key = str(scope or '').strip()
    if scope_key == 'lowprice':
        resolved_source = str(source_trader or trader or '').strip()
        return is_retired_lowprice_baseline(profile, resolved_source, symbol)
    return is_retired_mainline_cell(profile, trader, symbol)


def selected_cells_for_profile(profile: str) -> list[tuple[str, str]]:
    retired_raw = [
        (name, symbol)
        for name, symbol in SELECTED_CELLS.get(profile, [])
        if is_retired_lowprice_baseline(profile, name, symbol)
    ]
    if retired_raw:
        raise RuntimeError(
            f"retired lowprice baseline still present in SELECTED_CELLS[{profile}]: {retired_raw}"
        )
    return [
        (name, symbol)
        for name, symbol in SELECTED_CELLS.get(profile, [])
        if not is_retired_lowprice_baseline(profile, name, symbol)
    ]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def active_names(path: Path) -> set[str]:
    payload = load_json(path)
    names = payload.get('traderNames') if isinstance(payload, dict) else []
    return {str(x).strip() for x in names if isinstance(x, str) and str(x).strip()}


def parse_symbols(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(x).strip().upper() for x in raw if str(x).strip()]
    return [x.strip().upper() for x in str(raw or '').split(',') if x.strip()]


def expected_clone_initial_capital(row: dict[str, Any]) -> float:
    _ = row
    return round(INITIAL_CAPITAL, 2)


def resolve_prediction_suffix(row: dict[str, Any], symbol: str) -> str:
    by_symbol = row.get('predictionSuffixBySymbol') if isinstance(row.get('predictionSuffixBySymbol'), dict) else {}
    symbol_candidates = [str(symbol or ''), str(symbol or '').upper(), str(symbol or '').lower()]
    mapped = None
    for key in symbol_candidates:
        value = by_symbol.get(key)
        if str(value or '').strip():
            mapped = value
            break
    suffix = str(mapped or row.get('predictionSuffix') or '').strip()
    if not suffix:
        raise RuntimeError(f'missing prediction suffix for {row.get("name")} {symbol}')
    return suffix


def resolve_prediction_file(suffix: str) -> Path:
    return POLY / f'predictions{suffix}.json'


def rules_path_from_row(row: dict[str, Any]) -> Path | None:
    raw = str(row.get('rulesJsonPath') or '').strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path


def load_rules_payload(row: dict[str, Any]) -> dict[str, Any]:
    path = rules_path_from_row(row)
    if not path or not path.exists():
        return {}
    try:
        payload = load_json(path)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def model_dir_for(name: str, symbol: str) -> Path:
    return MODELS / f'v5_core10_{name}_{symbol.lower()}'


def parse_name_dynamic_range(name: str) -> list[float] | None:
    match = DYN_RE.search(name or '')
    if not match:
        return None
    low = round(int(match.group(1)) / 1000.0, 2)
    high = round(int(match.group(2)) / 1000.0, 2)
    return [low, high]


def format_ladder_from_range(price_range: list[float]) -> str:
    lo, hi = [round(float(x), 2) for x in price_range]
    values = []
    cur = lo
    while cur <= hi + 1e-9:
        values.append(f'{cur:.2f}')
        cur = round(cur + 0.01, 2)
    return ','.join(values)


def format_ladder_from_levels(levels: list[float]) -> str:
    return ','.join(f'{round(float(x), 2):.2f}' for x in levels)


def sanitize_candidate_tag(raw: str | None) -> str:
    text = str(raw or '').strip()
    if not text:
        return ''
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', text).strip('._-')


def derive_lowprice_family_id(
    selection_mode: str,
    selected_buy_price: float | None,
    selected_buy_price_range: list[float] | None,
    finalist_key: str | None = None,
) -> str:
    key = str(finalist_key or '').strip()
    if key:
        return key
    if selection_mode == 'fixed_price':
        if selected_buy_price is None:
            raise RuntimeError('missing selected_buy_price for fixed_price family id')
        return f"fixed_bp{int(round(float(selected_buy_price) * 1000)):04d}"
    if not isinstance(selected_buy_price_range, list) or len(selected_buy_price_range) != 2:
        raise RuntimeError('missing selected_buy_price_range for dynamic_range family id')
    lo = int(round(float(selected_buy_price_range[0]) * 1000))
    hi = int(round(float(selected_buy_price_range[1]) * 1000))
    return f"dynamic_{lo:04d}_{hi:04d}"


def derive_lowprice_family_rule_version(candidate_tag: str | None = None) -> str:
    tag = sanitize_candidate_tag(candidate_tag)
    return f'candidate::{tag}' if tag else 'incumbent'


def lowprice_rules_path(
    profile: str,
    source_trader: str,
    symbol: str,
    selection_mode: str,
    selected_buy_price: float | None,
    selected_buy_price_range: list[float] | None,
    candidate_tag: str | None = None,
) -> Path:
    out_dir = LOWPRICE_RULES_DIR / profile
    tag = sanitize_candidate_tag(candidate_tag)
    if tag:
        out_dir = out_dir / '_candidates' / tag
    if selection_mode == 'fixed_price':
        if selected_buy_price is None:
            raise RuntimeError(f'missing selected_buy_price for fixed_price rules path: {profile}:{source_trader}:{symbol}')
        suffix = f"bp{int(round(float(selected_buy_price) * 1000)):04d}"
    else:
        if not isinstance(selected_buy_price_range, list) or len(selected_buy_price_range) != 2:
            raise RuntimeError(f'missing selected_buy_price_range for dynamic_range rules path: {profile}:{source_trader}:{symbol}')
        suffix = (
            f"bpdyn_{int(round(float(selected_buy_price_range[0]) * 1000)):04d}_"
            f"{int(round(float(selected_buy_price_range[1]) * 1000)):04d}"
        )
    return out_dir / f'{source_trader}_{symbol.lower()}_{suffix}.json'


def ensure_lowprice_rules_file(
    profile: str,
    source_trader: str,
    symbol: str,
    selection_mode: str,
    selected_buy_price: float | None,
    selected_buy_price_range: list[float] | None,
    baseline: dict[str, Any],
    extra_metadata: dict[str, Any] | None = None,
    candidate_tag: str | None = None,
) -> str:
    out_path = lowprice_rules_path(
        profile=profile,
        source_trader=source_trader,
        symbol=symbol,
        selection_mode=selection_mode,
        selected_buy_price=selected_buy_price,
        selected_buy_price_range=selected_buy_price_range,
        candidate_tag=candidate_tag,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(json.dumps(baseline['resolved_non_price_rules']))
    poly = payload.get('polymarket_constraints') if isinstance(payload.get('polymarket_constraints'), dict) else {}
    if selection_mode == 'fixed_price':
        if selected_buy_price is None:
            raise RuntimeError(f'missing selected_buy_price for fixed_price rules file: {profile}:{source_trader}:{symbol}')
        price = round(float(selected_buy_price), 2)
        poly['buy_price'] = price
        poly['buy_price_range'] = None
        poly['odds'] = round((1.0 - price) / price, 4)
    else:
        if not isinstance(selected_buy_price_range, list) or len(selected_buy_price_range) != 2:
            raise RuntimeError(f'missing selected_buy_price_range for dynamic_range rules file: {profile}:{source_trader}:{symbol}')
        lo = round(float(selected_buy_price_range[0]), 2)
        hi = round(float(selected_buy_price_range[1]), 2)
        mid = round((lo + hi) / 2.0, 4)
        poly['buy_price'] = mid
        poly['buy_price_range'] = [lo, hi]
        poly['odds'] = round((1.0 - mid) / mid, 4)
    payload['polymarket_constraints'] = poly
    metadata = payload.get('metadata') if isinstance(payload.get('metadata'), dict) else {}
    metadata = dict(metadata)
    metadata.update({
        'source_trader': source_trader,
        'profile': profile,
        'symbol': symbol,
        'selection_mode': selection_mode,
        'lowprice_family_id': derive_lowprice_family_id(
            selection_mode=selection_mode,
            selected_buy_price=selected_buy_price,
            selected_buy_price_range=selected_buy_price_range,
            finalist_key=(extra_metadata or {}).get('lowpriceFamilyId') if isinstance(extra_metadata, dict) else None,
        ),
        'family_rule_version': derive_lowprice_family_rule_version(candidate_tag),
        'selected_buy_price': None if selected_buy_price is None else round(float(selected_buy_price), 4),
        'selected_buy_price_range': selected_buy_price_range,
        'generated_at': now_iso(),
    })
    if isinstance(extra_metadata, dict):
        metadata.update(extra_metadata)
    payload['metadata'] = metadata
    dump_json(out_path, payload)
    return str(out_path)


def build_dynamic_candidates(source_range: list[float]) -> list[list[float]]:
    width = round(float(source_range[1]) - float(source_range[0]), 2)
    if width <= 0:
        raise RuntimeError(f'invalid source dynamic range: {source_range}')
    end_max = round(PRICE_BOUNDS[1] - width, 2)
    candidates: list[list[float]] = []
    cur = PRICE_BOUNDS[0]
    while cur <= end_max + 1e-9:
        candidates.append([round(cur, 2), round(cur + width, 2)])
        cur = round(cur + 0.01, 2)
    return candidates


def normalize_nonprice_rules(payload: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(payload or {}))
    poly = out.get('polymarket_constraints') if isinstance(out.get('polymarket_constraints'), dict) else {}
    if isinstance(poly, dict):
        poly.pop('buy_price', None)
        poly.pop('buy_price_range', None)
        poly.pop('odds', None)
        out['polymarket_constraints'] = poly
    return out


def runtime_params_from_cfg(cfg: dict[str, Any]) -> dict[str, float]:
    return {
        'min_confidence': float(cfg['probThreshold']),
        'min_edge': float(cfg['minEdge']),
        'kelly_frac': float(cfg['kellyFrac']),
        'bet_pct_normal': min(float(cfg['betPctNormal']), SIM_BET_PCT_NORMAL_CAP),
        'bet_pct_conservative': min(float(cfg['betPctConservative']), SIM_BET_PCT_CONSERVATIVE_CAP),
        'conf_tier1_bound': float(cfg['confTier1Bound']),
        'conf_tier2_bound': float(cfg['confTier2Bound']),
        'tier1_mult': float(cfg['tier1Mult']),
        'tier2_mult': float(cfg['tier2Mult']),
        'tier3_mult': float(cfg['tier3Mult']),
        'cooldown_bars': int(cfg['cooldownBars']),
        'drawdown_halt': float(cfg['drawdownHalt']),
    }


def runtime_mode_controls_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in (
        'selectorMode',
        'selectorModeBySymbol',
        'selectorEligibleBySymbol',
        'selectorScopeBySymbol',
        'regimeMode',
        'regimeModeBySymbol',
        'calibrationMode',
        'calibrationModeBySymbolDirection',
        'calibrationStatsMode',
        'calibrationCheckSeconds',
        'expectancyGateMode',
        'expectancyGateModeBySymbolDirection',
        'thresholdDriftMode',
        'thresholdDriftModeBySymbolDirection',
        'metaLabelMode',
        'metaLabelModeBySymbolDirection',
        'jointOptimizationMode',
        'jointOptimizationExpectedPayoffMode',
        'runtimeGuardMode',
        'overlayMode',
    ):
        if key in cfg and cfg.get(key) is not None:
            out[key] = cfg.get(key)
    return out


def _symbol_direction_candidates(symbol: str, direction: str) -> list[str]:
    symbol = str(symbol or '')
    direction = str(direction or '').upper()
    return [
        f'{symbol}_{direction}',
        f'{symbol.upper()}_{direction}',
        f'{symbol.lower()}_{direction}',
        f'{symbol}__{direction}',
        f'{symbol.upper()}__{direction}',
        f'{symbol.lower()}__{direction}',
    ]


def resolve_selector_mode_from_cfg(cfg: dict[str, Any], symbol: str) -> str:
    by_symbol = cfg.get('selectorModeBySymbol') if isinstance(cfg.get('selectorModeBySymbol'), dict) else {}
    raw = str(by_symbol.get(symbol) or by_symbol.get(str(symbol).upper()) or cfg.get('selectorMode') or 'off').lower()
    if raw in {'shadow', 'enforce'}:
        return raw
    return 'off'


def selector_target_symbol(symbol: str) -> bool:
    return str(symbol or '').upper() in SELECTOR_TARGET_SYMBOLS


def selector_runtime_eligible_from_cfg(cfg: dict[str, Any], symbol: str) -> bool:
    symbol = str(symbol or '').upper()
    if not selector_target_symbol(symbol):
        return True
    mode = resolve_selector_mode_from_cfg(cfg, symbol)
    if mode in {'off', 'shadow'}:
        return True
    eligible = cfg.get('selectorEligibleBySymbol') if isinstance(cfg.get('selectorEligibleBySymbol'), dict) else {}
    configured = eligible.get(symbol)
    if isinstance(configured, bool):
        return configured
    return True


def resolve_calibration_mode_from_cfg(cfg: dict[str, Any], symbol: str, direction: str) -> str:
    by_symbol_direction = cfg.get('calibrationModeBySymbolDirection') if isinstance(cfg.get('calibrationModeBySymbolDirection'), dict) else {}
    raw = None
    for key in _symbol_direction_candidates(symbol, direction):
        if key in by_symbol_direction:
            raw = by_symbol_direction[key]
            break
    if raw is None:
        raw = cfg.get('calibrationMode') or 'off'
    raw = str(raw).lower()
    if raw in {'shadow', 'enforce'}:
        return raw
    return 'off'


def calibration_target_symbol(symbol: str) -> bool:
    return str(symbol or '').upper() in CALIBRATION_TARGET_SYMBOLS


def calibration_overrides_from_cfg(cfg: dict[str, Any], symbol: str, direction: str) -> dict[str, Any]:
    by_symbol_direction = cfg.get('calibrationBySymbolDirection') if isinstance(cfg.get('calibrationBySymbolDirection'), dict) else {}
    for key in _symbol_direction_candidates(symbol, direction):
        hit = by_symbol_direction.get(key)
        if isinstance(hit, dict):
            return hit
    return {}


def regime_target_symbol(symbol: str) -> bool:
    return str(symbol or '').upper() in REGIME_TARGET_SYMBOLS


def resolve_regime_mode_from_cfg(cfg: dict[str, Any], symbol: str) -> str:
    by_symbol = cfg.get('regimeModeBySymbol') if isinstance(cfg.get('regimeModeBySymbol'), dict) else {}
    raw = str(by_symbol.get(symbol) or by_symbol.get(str(symbol).upper()) or cfg.get('regimeMode') or 'off').lower()
    if raw in {'shadow', 'enforce'}:
        return raw
    return 'off'


def resolve_regime_state(profile: str, cfg: dict[str, Any], symbol: str) -> dict[str, Any]:
    symbol = str(symbol or '').upper()
    mode = resolve_regime_mode_from_cfg(cfg, symbol)
    state = {
        'mode': mode,
        'active': False,
        'directionPolicy': 'BOTH',
        'policyMode': 'neutral',
        'policyConfidence': 0.0,
        'thresholdDelta': 0.0,
        'regimeId': 'none',
        'reasonCode': 'off_mode' if mode == 'off' else 'no_data',
        'changePointProb': None,
        'policyDisagreementFlags': [],
    }
    if not regime_target_symbol(symbol):
        state['reasonCode'] = 'symbol_not_target'
        return state
    if mode == 'off':
        return state
    path = REGIME_RUNTIME_DIR / f'regime_gates_v1_{profile}.json'
    if not path.exists():
        return state
    try:
        payload = load_json(path)
    except Exception:
        state['reasonCode'] = 'parse_error'
        return state
    rows = payload.get('symbols') if isinstance(payload, dict) else []
    if not isinstance(rows, list):
        return state
    matched = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get('symbol') or '').strip().upper() != symbol:
            continue
        source_mode = str(row.get('sourceMode') or 'simulation').lower()
        if source_mode in {'simulation', 'live'} and source_mode != 'simulation':
            continue
        matched = row
        break
    if not isinstance(matched, dict):
        state['reasonCode'] = 'symbol_missing'
        return state
    direction_policy_raw = str(matched.get('directionPolicy') or 'BOTH').strip().upper()
    direction_policy = direction_policy_raw if direction_policy_raw in {'BOTH', 'UP', 'DOWN', 'NONE'} else 'BOTH'
    policy_mode = 'one_sided' if str(matched.get('policyMode') or '').strip().lower() == 'one_sided' else 'neutral'
    threshold_delta_raw = float(matched.get('thresholdDelta') or 0.0)
    threshold_delta = max(-0.05, min(0.05, threshold_delta_raw))
    policy_confidence_raw = float(matched.get('policyConfidence') or 0.0)
    change_point_prob_raw = matched.get('changePointProb')
    try:
        change_point_prob = float(change_point_prob_raw) if change_point_prob_raw is not None else None
    except Exception:
        change_point_prob = None
    disagreement_flags = matched.get('policyDisagreementFlags')
    state.update({
        'active': bool(matched.get('active')),
        'directionPolicy': direction_policy,
        'policyMode': policy_mode,
        'policyConfidence': max(0.0, min(1.0, policy_confidence_raw)),
        'thresholdDelta': threshold_delta,
        'regimeId': str(matched.get('regimeId') or 'unknown'),
        'reasonCode': str(matched.get('reasonCode') or 'ok'),
        'changePointProb': None if change_point_prob is None else max(0.0, min(1.0, change_point_prob)),
        'policyDisagreementFlags': [str(x).strip() for x in disagreement_flags] if isinstance(disagreement_flags, list) else [],
    })
    return state


def expectancy_target_symbol(symbol: str) -> bool:
    return str(symbol or '').upper() in EXPECTANCY_TARGET_SYMBOLS


def resolve_expectancy_mode_from_cfg(cfg: dict[str, Any], symbol: str, direction: str) -> str:
    by_symbol_direction = cfg.get('expectancyGateModeBySymbolDirection') if isinstance(cfg.get('expectancyGateModeBySymbolDirection'), dict) else {}
    raw = None
    for key in _symbol_direction_candidates(symbol, direction):
        if key in by_symbol_direction:
            raw = by_symbol_direction[key]
            break
    if raw is None:
        raw = cfg.get('expectancyGateMode') or 'off'
    raw = str(raw).lower()
    if raw in {'shadow', 'enforce'}:
        return raw
    return 'off'


def resolve_expectancy_state(profile: str, trader_name: str, cfg: dict[str, Any], symbol: str, direction: str) -> dict[str, Any]:
    symbol = str(symbol or '').upper()
    direction = str(direction or '').upper()
    mode = resolve_expectancy_mode_from_cfg(cfg, symbol, direction)
    state = {
        'mode': mode,
        'status': 'normal',
        'blocked': False,
        'degradedBetScale': 1.0,
        'degradedExtraDelta': 0.0,
        'reasonCode': 'off_mode' if mode == 'off' else 'no_data',
        'sampleCount': 0,
        'stats': {},
    }
    if not expectancy_target_symbol(symbol):
        state['reasonCode'] = 'symbol_not_target'
        return state
    if mode == 'off':
        return state
    path = EXPECTANCY_RUNTIME_DIR / f'expectancy_gates_v1_{profile}.json'
    if not path.exists():
        return state
    try:
        payload = load_json(path)
    except Exception:
        state['reasonCode'] = 'parse_error'
        return state
    cells = payload.get('cells') if isinstance(payload, dict) else []
    if not isinstance(cells, list):
        return state
    matched = None
    trader_name = str(trader_name or '')
    for row in cells:
        if not isinstance(row, dict):
            continue
        if str(row.get('profile') or profile) != profile:
            continue
        if str(row.get('traderName') or '') != trader_name:
            continue
        if str(row.get('symbol') or '').strip().upper() != symbol:
            continue
        if str(row.get('direction') or '').strip().upper() != direction:
            continue
        source_mode = str(row.get('sourceMode') or 'simulation').lower()
        if source_mode in {'simulation', 'live'} and source_mode != 'simulation':
            continue
        matched = row
        break
    if not isinstance(matched, dict):
        state['reasonCode'] = 'cell_missing'
        return state
    stats = matched.get('stats') if isinstance(matched.get('stats'), dict) else {}
    sample_count_raw = matched.get('sampleCount')
    if sample_count_raw is None and isinstance(stats, dict):
        sample_count_raw = stats.get('trades')
    try:
        sample_count = int(sample_count_raw or 0)
    except Exception:
        sample_count = 0
    state.update({
        'status': str(matched.get('status') or 'normal'),
        'blocked': bool(matched.get('blocked')),
        'degradedBetScale': float(matched.get('degradedBetScale') or 1.0),
        'degradedExtraDelta': float(matched.get('degradedExtraDelta') or 0.0),
        'reasonCode': str(matched.get('reasonCode') or 'ok'),
        'sampleCount': max(0, sample_count),
        'stats': stats,
    })
    return state


def ladder_range_from_cfg(row: dict[str, Any]) -> list[float] | None:
    ladder = str(row.get('limitPriceLadder') or '').strip()
    if not ladder:
        return None
    vals = [round(float(x.strip()), 2) for x in ladder.split(',') if x.strip()]
    if not vals:
        return None
    return [min(vals), max(vals)]


def source_runtime_dir(row: dict[str, Any], symbol: str) -> str:
    base = str(row.get('logsDir') or '').strip()
    split = f'{base}__simulation_{symbol.lower()}'
    return split if (POLY / split).is_dir() else base


def selected_row(profile: str, name: str) -> dict[str, Any]:
    rows = load_json(PROFILES[profile]['config'])
    row = next((r for r in rows if isinstance(r, dict) and str(r.get('name') or '') == name), None)
    if not isinstance(row, dict):
        raise RuntimeError(f'missing trader config for {profile}:{name}')
    return row


def resolve_baseline(profile: str, name: str, symbol: str) -> dict[str, Any]:
    row = selected_row(profile, name)
    rules_payload = load_rules_payload(row)
    rules_poly = rules_payload.get('polymarket_constraints') if isinstance(rules_payload.get('polymarket_constraints'), dict) else {}
    name_range = parse_name_dynamic_range(name)
    rules_buy = rules_poly.get('buy_price') if isinstance(rules_poly, dict) else None
    rules_range = rules_poly.get('buy_price_range') if isinstance(rules_poly, dict) else None
    flags: list[str] = []

    ladder_range = ladder_range_from_cfg(row)
    if name_range is not None or ladder_range is not None:
        resolved_mode = 'dynamic'
        if ladder_range is not None:
            source_range = ladder_range
        elif name_range is not None:
            source_range = name_range
        elif isinstance(rules_range, list) and len(rules_range) == 2:
            source_range = [round(float(rules_range[0]), 2), round(float(rules_range[1]), 2)]
        else:
            raise RuntimeError(f'dynamic trader missing price range truth: {profile}:{name}:{symbol}')
        if name_range is not None and source_range != name_range:
            flags.append(f'name_config_range_mismatch:{name_range}!={source_range}')
        if isinstance(rules_range, list) and len(rules_range) == 2:
            rr = [round(float(rules_range[0]), 2), round(float(rules_range[1]), 2)]
            if rr != source_range:
                flags.append(f'config_rules_range_mismatch:{source_range}!={rr}')
        elif rules_range is None:
            flags.append('rules_range_missing')
        source_price = None
    else:
        resolved_mode = 'fixed'
        cfg_price = row.get('limitPrice')
        if cfg_price is None and rules_buy is not None:
            flags.append('config_limit_missing_rules_buy_present')
        if cfg_price is not None and rules_buy is not None:
            cfg_p = round(float(cfg_price), 2)
            rules_p = round(float(rules_buy), 2)
            if cfg_p != rules_p:
                flags.append(f'config_rules_buy_mismatch:{cfg_p}!={rules_p}')
        if cfg_price is None:
            raise RuntimeError(f'fixed trader missing config.limitPrice: {profile}:{name}:{symbol}')
        source_price = round(float(cfg_price), 2)
        source_range = None

    ladder = str(row.get('limitPriceLadder') or '').strip()
    if resolved_mode == 'dynamic':
        if not ladder:
            flags.append('config_ladder_missing')
        else:
            ladder_vals = [round(float(x.strip()), 2) for x in ladder.split(',') if x.strip()]
            ladder_minmax = [min(ladder_vals), max(ladder_vals)] if ladder_vals else None
            if ladder_minmax and ladder_minmax != source_range:
                flags.append(f'config_ladder_range_mismatch:{ladder_minmax}!={source_range}')
    elif ladder:
        flags.append('fixed_has_limit_ladder')

    model_dir = model_dir_for(name, symbol)
    if not model_dir.exists():
        raise RuntimeError(f'missing model dir for {profile}:{name}:{symbol}: {model_dir}')

    suffix = resolve_prediction_suffix(row, symbol)
    active = active_names(PROFILES[profile]['active'])
    source_markets = parse_symbols(row.get('allowedMarkets'))
    source_sizing_reference_price = (
        round(float(source_price), 4)
        if source_price is not None
        else round((float(source_range[0]) + float(source_range[1])) / 2.0, 4)
    )
    return {
        'profile': profile,
        'source_trader': name,
        'symbol': symbol,
        'active_in_profile': name in active,
        'resolved_price_mode': resolved_mode,
        'resolved_source_buy_price': source_price,
        'resolved_source_buy_price_range': source_range,
        'resolved_source_sizing_reference_price': source_sizing_reference_price,
        'resolved_source_limit_ladder': ladder if ladder else None,
        'source_prediction_suffix': suffix,
        'source_prediction_file': str(resolve_prediction_file(suffix)),
        'source_logs_dir': source_runtime_dir(row, symbol),
        'source_rules_path': str(rules_path_from_row(row)) if rules_path_from_row(row) else None,
        'source_allowed_markets': source_markets,
        'source_market_count': len(source_markets),
        'source_initial_capital': float(row.get('initialCapital') or INITIAL_CAPITAL),
        'expected_clone_initial_capital': expected_clone_initial_capital(row),
        'model_dir': str(model_dir),
        'config_limit_price': row.get('limitPrice'),
        'config_limit_price_ladder': row.get('limitPriceLadder'),
        'config_use_prediction_limit_price': row.get('usePredictionLimitPrice'),
        'config_rules_mismatch_flags': flags,
        'resolved_non_price_rules': normalize_nonprice_rules(rules_payload),
        'source_row': row,
        'runtime_params': runtime_params_from_cfg(row),
        'runtime_mode_controls': runtime_mode_controls_from_cfg(row),
        'selector_mode': resolve_selector_mode_from_cfg(row, symbol),
        'selector_runtime_eligible': selector_runtime_eligible_from_cfg(row, symbol),
        'calibration_modes': {
            'UP': resolve_calibration_mode_from_cfg(row, symbol, 'UP'),
            'DOWN': resolve_calibration_mode_from_cfg(row, symbol, 'DOWN'),
        },
        'calibration_overrides': {
            'UP': calibration_overrides_from_cfg(row, symbol, 'UP'),
            'DOWN': calibration_overrides_from_cfg(row, symbol, 'DOWN'),
        },
        'regime_mode': resolve_regime_mode_from_cfg(row, symbol),
        'regime_state': resolve_regime_state(profile, row, symbol),
        'expectancy_modes': {
            'UP': resolve_expectancy_mode_from_cfg(row, symbol, 'UP'),
            'DOWN': resolve_expectancy_mode_from_cfg(row, symbol, 'DOWN'),
        },
        'expectancy_states': {
            'UP': resolve_expectancy_state(profile, name, row, symbol, 'UP'),
            'DOWN': resolve_expectancy_state(profile, name, row, symbol, 'DOWN'),
        },
    }


def iter_selected_baselines() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for profile in SELECTED_CELLS:
        for name, symbol in selected_cells_for_profile(profile):
            out.append(resolve_baseline(profile, name, symbol))
    return out


def iter_active_lowprice_source_baselines(
    profiles: list[str] | None = None,
    symbols: list[str] | None = None,
) -> list[dict[str, Any]]:
    target_profiles = profiles or list(PROFILES.keys())
    target_symbols = {str(sym).strip().upper() for sym in (symbols or []) if str(sym).strip()}
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for profile in target_profiles:
        spec = PROFILES[profile]
        active = active_names(spec['out_active'])
        payload = load_json(spec['out_config'])
        rows = payload if isinstance(payload, list) else []
        for row in rows:
            if not isinstance(row, dict):
                continue
            trader_name = str(row.get('name') or '').strip()
            if trader_name not in active:
                continue
            source_trader = str(row.get('lowPriceSourceTrader') or row.get('sourceTrader') or '').strip()
            symbol = str(row.get('lowPriceSymbol') or row.get('allowedMarkets') or '').strip().upper()
            if not source_trader or not symbol:
                continue
            if target_symbols and symbol not in target_symbols:
                continue
            key = (profile, source_trader, symbol)
            if key in seen:
                continue
            seen.add(key)
            out.append(resolve_baseline(profile, source_trader, symbol))
    return out


def iter_fixed_dynamic_compare_rows(profile: str | None = None) -> list[dict[str, Any]]:
    if profile is not None:
        return [dict(row) for row in FIXED_DYNAMIC_COMPARE_ROWS.get(profile, [])]
    rows: list[dict[str, Any]] = []
    for key in FIXED_DYNAMIC_COMPARE_ROWS:
        rows.extend(iter_fixed_dynamic_compare_rows(key))
    return rows


def fixed_dynamic_runtime_row_override(profile: str, source_trader: str, symbol: str) -> dict[str, Any] | None:
    profile_key = str(profile or '').strip()
    source_key = str(source_trader or '').strip()
    symbol_key = str(symbol or '').strip().upper()
    override = FIXED_DYNAMIC_RUNTIME_ROW_OVERRIDES.get(profile_key, {}).get((source_key, symbol_key))
    return dict(override) if isinstance(override, dict) else None


def execution_defaults_for_asset(symbol: str) -> dict[str, float]:
    asset_name = ASSET_NAME[symbol]
    try:
        payload = load_json(BOOTSTRAP_REPORT)
    except Exception:
        payload = {}
    defaults = (((payload or {}).get('assets') or {}).get(asset_name) or {}).get('bootstrap_defaults') or {}
    result = dict(DEFAULT_EXECUTION_DEFAULTS)
    for key in result:
        if key in defaults:
            result[key] = float(defaults[key])
    return result
