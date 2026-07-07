#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numba
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.exp_lowprice_selected_common import (
    ASSET_1M_PATH,
    ASSET_NAME,
    DYNAMIC_FULL_LADDER_RANGE,
    FIXED_FINALIST_COUNT,
    HOLDOUT_DAYS,
    INITIAL_CAPITAL,
    LOWPRICE_RULES_DIR,
    MIN_EXECUTION_ADJUSTED_FILL_RATIO,
    MIN_FILLABLE_RATE,
    MIN_HOLDOUT_TRADES,
    PRICE_LEVELS,
    PROFILES,
    REPORTS,
    SELECTED_CELLS,
    SETTLEMENT_COST,
    TOTAL_DAYS,
    calibration_target_symbol,
    derive_lowprice_family_id,
    derive_lowprice_family_rule_version,
    dump_json,
    ensure_lowprice_rules_file,
    execution_defaults_for_asset,
    expectancy_target_symbol,
    format_ladder_from_levels,
    format_ladder_from_range,
    iter_active_lowprice_source_baselines,
    iter_selected_baselines,
    now_iso,
    regime_target_symbol,
    selector_target_symbol,
    XRP_1M_COVERAGE_THRESHOLD,
)
from scripts.optimize_trading_rules import generate_predictions

CACHE_DIR = PROJECT_ROOT / 'data' / 'processed' / 'exp_lowprice_selected'
BASELINE_REPORT = REPORTS / 'exp_lowprice_selected_baselines_latest.json'
OPT_REPORT = REPORTS / 'exp_lowprice_selected_optimization_latest.json'
POLY_SNAPSHOT_PATH = PROJECT_ROOT / 'data' / 'sentiment' / 'polymarket_ob_snapshots.parquet'
CALIBRATION_TEMPERATURE_DEFAULT = 1.0
CALIBRATION_ALPHA_DEFAULT = 0.0
CALIBRATION_BETA_DEFAULT = 1.0
QUEUE_AWARE_RUNTIME_HAIRCUT = 0.95


def _round(x: float | None, n: int = 6) -> float | None:
    if x is None:
        return None
    v = float(x)
    if not math.isfinite(v):
        return None
    return round(v, n)


def _write_rules_file(
    profile: str,
    source_trader: str,
    symbol: str,
    selection_mode: str,
    selected_buy_price: float | None,
    selected_buy_price_range: list[float] | None,
    baseline: dict[str, Any],
    metrics: dict[str, Any],
    *,
    candidate_tag: str | None = None,
    lowprice_family_id: str | None = None,
    family_rule_version: str | None = None,
    stale_order_policy: str = 'off',
    stale_cancel_min_age_sec: int = 0,
    stale_cancel_min_drift_ticks: int = 0,
    stale_cancel_min_time_to_expiry_sec: int = 0,
) -> str:
    return ensure_lowprice_rules_file(
        profile=profile,
        source_trader=source_trader,
        symbol=symbol,
        selection_mode=selection_mode,
        selected_buy_price=selected_buy_price,
        selected_buy_price_range=selected_buy_price_range,
        baseline=baseline,
        candidate_tag=candidate_tag,
        extra_metadata={
            'lowpriceFamilyId': lowprice_family_id,
            'family_rule_version': family_rule_version or derive_lowprice_family_rule_version(candidate_tag),
            'holdout_total_pnl': metrics.get('holdout', {}).get('total_pnl'),
            'train_total_pnl': metrics.get('train', {}).get('total_pnl'),
            'staleOrderPolicy': stale_order_policy,
            'staleCancelMinAgeSec': int(max(0, stale_cancel_min_age_sec)),
            'staleCancelMinDriftTicks': int(max(0, stale_cancel_min_drift_ticks)),
            'staleCancelMinTimeToExpirySec': int(max(0, stale_cancel_min_time_to_expiry_sec)),
        },
    )


def _parse_csv_set(raw: str | None) -> set[str]:
    return {part.strip() for part in str(raw or '').split(',') if part.strip()}


def _parse_upper_csv_set(raw: str | None) -> set[str]:
    return {part.strip().upper() for part in str(raw or '').split(',') if part.strip()}


def _prediction_cache_path(baseline: dict[str, Any], days: int) -> Path:
    return CACHE_DIR / f"{baseline['source_trader']}_{baseline['symbol'].lower()}_{days}d.parquet"


def _load_or_generate_predictions(baseline: dict[str, Any], days: int, refresh: bool) -> pd.DataFrame:
    cache_path = _prediction_cache_path(baseline, days)
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)
    df = generate_predictions(model_dir=Path(baseline['model_dir']), apply_noise=True, test_days=days)
    df = df[df['asset'] == ASSET_NAME[baseline['symbol']]].copy().reset_index(drop=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


def _apply_calibration_method(confidence: float, overrides: dict[str, Any]) -> float:
    x = min(1.0 - 1e-6, max(1e-6, float(confidence)))
    method = str(overrides.get('calibrationMethod') or 'identity').lower()
    if method == 'temperature':
        temperature = max(0.05, float(overrides.get('calibrationTemperature') or CALIBRATION_TEMPERATURE_DEFAULT))
        logit = math.log(x / (1.0 - x))
        return 1.0 / (1.0 + math.exp(-logit / temperature))
    if method == 'sigmoid':
        alpha = float(overrides.get('calibrationAlpha') or CALIBRATION_ALPHA_DEFAULT)
        beta = float(overrides.get('calibrationBeta') or CALIBRATION_BETA_DEFAULT)
        logit = math.log(x / (1.0 - x))
        return 1.0 / (1.0 + math.exp(-(alpha + beta * logit)))
    if method == 'isotonic':
        xs = overrides.get('calibrationIsotonicX') if isinstance(overrides.get('calibrationIsotonicX'), list) else []
        ys = overrides.get('calibrationIsotonicY') if isinstance(overrides.get('calibrationIsotonicY'), list) else []
        pts: list[tuple[float, float]] = []
        for vx, vy in zip(xs, ys):
            try:
                fx = float(vx)
                fy = float(vy)
            except Exception:
                continue
            if math.isfinite(fx) and math.isfinite(fy):
                pts.append((fx, fy))
        if len(pts) >= 2:
            pts.sort(key=lambda item: item[0])
            if x <= pts[0][0]:
                return min(1.0, max(0.0, pts[0][1]))
            if x >= pts[-1][0]:
                return min(1.0, max(0.0, pts[-1][1]))
            for idx in range(1, len(pts)):
                left_x, left_y = pts[idx - 1]
                right_x, right_y = pts[idx]
                if x <= right_x:
                    span = max(1e-9, right_x - left_x)
                    w = (x - left_x) / span
                    return min(1.0, max(0.0, left_y + (right_y - left_y) * w))
    return x


def _apply_runtime_like_controls(predictions: pd.DataFrame, baseline: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = predictions.copy()
    symbol = str(baseline['symbol']).upper()
    selector_modeled = selector_target_symbol(symbol)
    calibration_modeled = calibration_target_symbol(symbol)
    regime_modeled = regime_target_symbol(symbol)
    expectancy_modeled = expectancy_target_symbol(symbol)
    selector_blocked_rows = 0
    calibration_applied_rows = 0
    calibration_considered_rows = 0
    regime_blocked_rows = 0
    expectancy_blocked_rows = 0
    expectancy_degraded_rows = 0

    if 'proba_up' not in out.columns:
        raise RuntimeError(f'missing proba_up for {baseline["source_trader"]} {symbol}')

    raw = pd.to_numeric(out['proba_up'], errors='coerce').astype(float).to_numpy()
    adjusted = raw.copy()
    threshold_extra = np.zeros(len(adjusted), dtype=np.float64)
    direction_blocked = np.zeros(len(adjusted), dtype=np.int8)
    bet_scale = np.ones(len(adjusted), dtype=np.float64)
    if selector_modeled and baseline.get('selector_mode') == 'enforce' and not bool(baseline.get('selector_runtime_eligible')):
        adjusted[:] = 0.5
        selector_blocked_rows = int(len(adjusted))
    elif calibration_modeled:
        up_mode = str((baseline.get('calibration_modes') or {}).get('UP') or 'off').lower()
        down_mode = str((baseline.get('calibration_modes') or {}).get('DOWN') or 'off').lower()
        up_overrides = dict((baseline.get('calibration_overrides') or {}).get('UP') or {})
        down_overrides = dict((baseline.get('calibration_overrides') or {}).get('DOWN') or {})
        for idx, p in enumerate(adjusted):
            if not math.isfinite(float(p)):
                continue
            direction = 'UP' if p >= 0.5 else 'DOWN'
            confidence = float(p if direction == 'UP' else 1.0 - p)
            mode = up_mode if direction == 'UP' else down_mode
            overrides = up_overrides if direction == 'UP' else down_overrides
            if mode != 'enforce':
                continue
            calibration_considered_rows += 1
            calibrated = _apply_calibration_method(confidence, overrides)
            new_p = calibrated if direction == 'UP' else (1.0 - calibrated)
            if abs(new_p - p) > 1e-12:
                calibration_applied_rows += 1
            adjusted[idx] = new_p

    if regime_modeled and baseline.get('regime_mode') == 'enforce':
        regime_state = dict(baseline.get('regime_state') or {})
        threshold_delta = max(-0.05, min(0.05, float(regime_state.get('thresholdDelta') or 0.0)))
        threshold_extra[:] = threshold_delta
        direction_policy = str(regime_state.get('directionPolicy') or 'BOTH').upper()
        if direction_policy in {'UP', 'DOWN', 'NONE'}:
            implied_up = adjusted >= 0.5
            if direction_policy == 'NONE':
                direction_blocked[:] = 1
            elif direction_policy == 'UP':
                direction_blocked[:] = (~implied_up).astype(np.int8)
            elif direction_policy == 'DOWN':
                direction_blocked[:] = implied_up.astype(np.int8)
            regime_blocked_rows = int(direction_blocked.sum())

    if expectancy_modeled:
        up_mode = str((baseline.get('expectancy_modes') or {}).get('UP') or 'off').lower()
        down_mode = str((baseline.get('expectancy_modes') or {}).get('DOWN') or 'off').lower()
        up_state = dict((baseline.get('expectancy_states') or {}).get('UP') or {})
        down_state = dict((baseline.get('expectancy_states') or {}).get('DOWN') or {})
        for idx, p in enumerate(adjusted):
            if not math.isfinite(float(p)):
                continue
            direction = 'UP' if p >= 0.5 else 'DOWN'
            mode = up_mode if direction == 'UP' else down_mode
            state = up_state if direction == 'UP' else down_state
            if mode != 'enforce':
                continue
            if bool(state.get('blocked')):
                direction_blocked[idx] = 1
                expectancy_blocked_rows += 1
                continue
            status = str(state.get('status') or 'normal').lower()
            if status == 'degraded':
                bet_scale[idx] = max(0.0, min(1.0, float(state.get('degradedBetScale') or 1.0)))
                threshold_extra[idx] += max(0.0, float(state.get('degradedExtraDelta') or 0.0))
                expectancy_degraded_rows += 1

    out['raw_proba_up'] = raw
    out['proba_up'] = adjusted
    out['runtime_threshold_extra'] = threshold_extra
    out['runtime_direction_blocked'] = direction_blocked
    out['runtime_bet_scale'] = bet_scale
    return out, {
        'selector_modeled': bool(selector_modeled),
        'selector_mode': baseline.get('selector_mode'),
        'selector_runtime_eligible': bool(baseline.get('selector_runtime_eligible', True)),
        'selector_blocked_rows': int(selector_blocked_rows),
        'calibration_modeled': bool(calibration_modeled),
        'calibration_modes': baseline.get('calibration_modes'),
        'calibration_considered_rows': int(calibration_considered_rows),
        'calibration_applied_rows': int(calibration_applied_rows),
        'regime_modeled': bool(regime_modeled),
        'regime_mode': baseline.get('regime_mode'),
        'regime_state': baseline.get('regime_state'),
        'regime_blocked_rows': int(regime_blocked_rows),
        'expectancy_modeled': bool(expectancy_modeled),
        'expectancy_modes': baseline.get('expectancy_modes'),
        'expectancy_states': baseline.get('expectancy_states'),
        'expectancy_blocked_rows': int(expectancy_blocked_rows),
        'expectancy_degraded_rows': int(expectancy_degraded_rows),
    }


def _load_poly_prob_snapshot_touch_map(symbol: str) -> pd.DataFrame:
    symbol_key = symbol.lower()
    prob_path = PROJECT_ROOT / 'data' / 'sentiment' / f'polymarket_prob_{symbol_key}_usdt.parquet'
    target_path = PROJECT_ROOT / 'data' / 'sentiment' / f'polymarket_prob_target_{symbol_key}_usdt.parquet'
    snapshot_rows: list[pd.DataFrame] = []

    if prob_path.exists():
        prob = pd.read_parquet(prob_path)
        if 'timestamp_s' in prob.columns and 'raw_p_last' in prob.columns:
            tmp = prob[['timestamp_s', 'raw_p_last']].copy()
            tmp['price'] = pd.to_numeric(tmp['raw_p_last'], errors='coerce')
            tmp['15m_start_ts'] = (pd.to_numeric(tmp['timestamp_s'], errors='coerce') // 900) * 900
            snapshot_rows.append(tmp[['15m_start_ts', 'price']])

    if target_path.exists():
        target = pd.read_parquet(target_path)
        if 'timestamp_s' in target.columns and 'target_raw_p_last' in target.columns:
            tmp = target[['timestamp_s', 'target_raw_p_last']].copy()
            tmp['price'] = pd.to_numeric(tmp['target_raw_p_last'], errors='coerce')
            tmp['15m_start_ts'] = (pd.to_numeric(tmp['timestamp_s'], errors='coerce') // 900) * 900
            snapshot_rows.append(tmp[['15m_start_ts', 'price']])

    if POLY_SNAPSHOT_PATH.exists():
        ob = pd.read_parquet(POLY_SNAPSHOT_PATH, columns=['timestamp_s', 'symbol', 'best_ask'])
        ob = ob[ob['symbol'].astype(str).str.upper() == f'{symbol.upper()}USDT'].copy()
        if not ob.empty:
            ob['price'] = pd.to_numeric(ob['best_ask'], errors='coerce')
            ob['15m_start_ts'] = (pd.to_numeric(ob['timestamp_s'], errors='coerce') // 900) * 900
            snapshot_rows.append(ob[['15m_start_ts', 'price']])

    if not snapshot_rows:
        raise RuntimeError(f'missing Polymarket fallback data for {symbol}')

    combined = pd.concat(snapshot_rows, ignore_index=True)
    combined = combined.dropna(subset=['15m_start_ts', 'price']).copy()
    combined['15m_start_ts'] = pd.to_numeric(combined['15m_start_ts'], errors='coerce').astype('int64')
    combined['price'] = pd.to_numeric(combined['price'], errors='coerce').astype(float)
    combined = combined[(combined['price'] > 0) & (combined['price'] < 1)].copy()
    out = combined.groupby('15m_start_ts', as_index=False).agg(
        path_min_price=('price', 'min'),
        path_max_price=('price', 'max'),
        path_obs=('price', 'size'),
    )
    return out


def _load_path_touch_map(symbol: str, allow_prob_snapshot_fallback: bool = False) -> tuple[pd.DataFrame, str]:
    path_file = ASSET_1M_PATH[symbol]
    if path_file.exists():
        df = pd.read_parquet(path_file, columns=['15m_start_ts', 't_sec', 'p'])
        df = df.dropna(subset=['15m_start_ts', 't_sec', 'p']).copy()
        df['15m_start_ts'] = pd.to_numeric(df['15m_start_ts'], errors='coerce').astype('int64')
        df['t_sec'] = pd.to_numeric(df['t_sec'], errors='coerce').astype('int64')
        df['p'] = pd.to_numeric(df['p'], errors='coerce').astype(float)
        df = df[df['t_sec'] >= df['15m_start_ts']].copy()
        out = df.groupby('15m_start_ts', as_index=False).agg(
            path_min_price=('p', 'min'),
            path_max_price=('p', 'max'),
            path_obs=('p', 'size'),
        )
        return out, 'polymarket_1m'
    if allow_prob_snapshot_fallback:
        return _load_poly_prob_snapshot_touch_map(symbol), 'polymarket_prob_snapshot_fallback'
    raise RuntimeError(f'missing polymarket 1m path for {symbol}')


def _attach_path_touch(predictions: pd.DataFrame, symbol: str, xrp_path_mode: str | None = None) -> tuple[pd.DataFrame, str]:
    out = predictions.copy()
    ts = pd.to_datetime(out['timestamp'], utc=True)
    out['market_start_ts'] = (ts.astype('int64') // 10**9).astype('int64')
    allow_fallback = str(symbol).upper() == 'XRP'
    if allow_fallback and xrp_path_mode == 'polymarket_prob_snapshot_fallback':
        touch, source = _load_path_touch_map(symbol, allow_prob_snapshot_fallback=True)
    else:
        touch, source = _load_path_touch_map(symbol, allow_prob_snapshot_fallback=False)
    out = out.merge(touch, how='left', left_on='market_start_ts', right_on='15m_start_ts')
    out['path_has_coverage'] = out['path_min_price'].notna()
    out['path_min_price'] = pd.to_numeric(out['path_min_price'], errors='coerce')
    out['path_max_price'] = pd.to_numeric(out['path_max_price'], errors='coerce')
    out['path_obs'] = pd.to_numeric(out['path_obs'], errors='coerce').fillna(0).astype(int)
    coverage_rate = float(out['path_has_coverage'].mean()) if len(out) else 0.0
    if allow_fallback and xrp_path_mode != 'polymarket_1m' and coverage_rate < XRP_1M_COVERAGE_THRESHOLD:
        touch, source = _load_path_touch_map(symbol, allow_prob_snapshot_fallback=True)
        out = predictions.copy()
        out['market_start_ts'] = (ts.astype('int64') // 10**9).astype('int64')
        out = out.merge(touch, how='left', left_on='market_start_ts', right_on='15m_start_ts')
        out['path_has_coverage'] = out['path_min_price'].notna()
        out['path_min_price'] = pd.to_numeric(out['path_min_price'], errors='coerce')
        out['path_max_price'] = pd.to_numeric(out['path_max_price'], errors='coerce')
        out['path_obs'] = pd.to_numeric(out['path_obs'], errors='coerce').fillna(0).astype(int)
    return out, source


def _resolve_xrp_path_mode(baselines: list[dict[str, Any]], refresh_predictions: bool) -> tuple[str, dict[str, Any]]:
    xrp_baseline = next((b for b in baselines if b['symbol'] == 'XRP'), None)
    if xrp_baseline is None:
        return 'not_applicable', {'mode': 'not_applicable', 'coverage_rate': None}
    if not ASSET_1M_PATH['XRP'].exists():
        return 'polymarket_prob_snapshot_fallback', {
            'mode': 'polymarket_prob_snapshot_fallback',
            'coverage_rate': 0.0,
            'reason': 'missing_1m_file',
            'threshold': XRP_1M_COVERAGE_THRESHOLD,
        }
    predictions = _load_or_generate_predictions(xrp_baseline, TOTAL_DAYS, refresh_predictions)
    probe = predictions.copy()
    ts = pd.to_datetime(probe['timestamp'], utc=True)
    probe['market_start_ts'] = (ts.astype('int64') // 10**9).astype('int64')
    touch, _ = _load_path_touch_map('XRP', allow_prob_snapshot_fallback=False)
    probe = probe.merge(touch, how='left', left_on='market_start_ts', right_on='15m_start_ts')
    coverage_rate = float(probe['path_min_price'].notna().mean()) if len(probe) else 0.0
    if coverage_rate >= XRP_1M_COVERAGE_THRESHOLD:
        return 'polymarket_1m', {
            'mode': 'polymarket_1m',
            'coverage_rate': _round(coverage_rate, 6),
            'threshold': XRP_1M_COVERAGE_THRESHOLD,
        }
    return 'polymarket_prob_snapshot_fallback', {
        'mode': 'polymarket_prob_snapshot_fallback',
        'coverage_rate': _round(coverage_rate, 6),
        'reason': 'coverage_below_threshold',
        'threshold': XRP_1M_COVERAGE_THRESHOLD,
    }


def _build_execution_rows(symbol: str) -> dict[str, Any]:
    defaults = execution_defaults_for_asset(symbol)
    rows: dict[str, Any] = {}
    for price in PRICE_LEVELS:
        expected_fill_ratio = defaults['fill_rate'] * ((1.0 - defaults['partial_fill_rate']) + defaults['partial_fill_rate'] * defaults['avg_partial_fill_ratio'])
        rows[f'{price:.4f}'] = {
            'price': price,
            'fill_rate': _round(defaults['fill_rate'], 6),
            'partial_fill_rate': _round(defaults['partial_fill_rate'], 6),
            'timeout_rate': _round(defaults['timeout_rate'], 6),
            'avg_partial_fill_ratio': _round(defaults['avg_partial_fill_ratio'], 6),
            'avg_queue_wait_seconds': _round(defaults['avg_queue_wait_seconds'], 6),
            'expected_fill_ratio': _round(expected_fill_ratio, 6),
        }
    return rows


def _build_execution_arrays(price_rows: dict[str, Any]) -> dict[str, np.ndarray]:
    price_points = np.array(sorted(float(k) for k in price_rows.keys()), dtype=np.float64)
    out: dict[str, np.ndarray] = {'price_points': price_points}
    for field in ('fill_rate', 'partial_fill_rate', 'timeout_rate', 'avg_partial_fill_ratio', 'avg_queue_wait_seconds', 'expected_fill_ratio'):
        out[field] = np.array([float(price_rows[f'{p:.4f}'][field]) for p in price_points], dtype=np.float64)
    return out


def _interp_metric(exec_arrays: dict[str, np.ndarray], prices: np.ndarray, field: str) -> np.ndarray:
    return np.interp(prices, exec_arrays['price_points'], exec_arrays[field], left=exec_arrays[field][0], right=exec_arrays[field][-1])


@numba.njit(cache=True)
def _simulate_trading_core_fillmask(
    probas, actuals, odds_arr, fill_ratio_arr, trade_weight_arr, threshold_extra_arr, direction_blocked_arr, bet_scale_arr,
    initial_capital, settlement_cost,
    min_conf, min_edge, kelly_frac,
    bet_pct_normal, bet_pct_conservative,
    tier1_bound, tier2_bound,
    tier1_mult, tier2_mult, tier3_mult,
    cooldown, dd_halt,
    liquidity_cap, liquidity_bet,
    halt_duration_bars, recovery_target,
):
    n = len(probas)
    pnl_arr = np.zeros(n, dtype=np.float64)
    capital = initial_capital
    peak_capital = initial_capital
    max_drawdown = 0.0
    wins = 0.0
    losses = 0.0
    trade_equiv = 0.0
    cooldown_remaining = 0
    halted = False
    halt_start_bar = 0
    ever_reached_cap = False
    for i in range(n):
        proba = probas[i]
        actual = actuals[i]
        dd = (peak_capital - capital) / peak_capital if peak_capital > 0.0 else 0.0
        if halted:
            bars_since_halt = i - halt_start_bar
            capital_recovered = capital >= peak_capital * recovery_target
            time_expired = bars_since_halt >= halt_duration_bars
            if capital_recovered or time_expired:
                halted = False
                # Align with runtime/compare_fair semantics: once a drawdown halt
                # cooling window ends, reset the peak to current capital so the
                # strategy is not immediately re-halted by the stale pre-halt peak.
                peak_capital = capital
            else:
                continue
        if dd >= dd_halt:
            halted = True
            halt_start_bar = i
            continue
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue
        if direction_blocked_arr[i] != 0:
            continue
        if proba >= 0.5:
            direction = 1
            confidence = proba
        else:
            direction = 0
            confidence = 1.0 - proba
        required_conf = min(0.999, max(0.0, min_conf + threshold_extra_arr[i]))
        if confidence < required_conf:
            continue
        odds_this = odds_arr[i]
        p = confidence
        q = 1.0 - p
        edge = p * odds_this - q
        if edge < min_edge:
            continue
        realized_fill_ratio = fill_ratio_arr[i]
        trade_weight = trade_weight_arr[i]
        if realized_fill_ratio <= 0.0 or trade_weight <= 0.0:
            continue
        if odds_this > 0.0:
            kelly_f = (p * odds_this - q) / odds_this
        else:
            kelly_f = 0.0
        if kelly_f < 0.0:
            kelly_f = 0.0
        bet_ratio = kelly_f * kelly_frac
        if confidence < tier1_bound:
            bet_ratio *= tier1_mult
        elif confidence < tier2_bound:
            bet_ratio *= tier2_mult
        else:
            bet_ratio *= tier3_mult
        if capital >= liquidity_cap:
            ever_reached_cap = True
            bet_amount = liquidity_bet
        elif ever_reached_cap:
            if bet_ratio > bet_pct_conservative:
                bet_ratio = bet_pct_conservative
            bet_amount = capital * bet_ratio
        else:
            if bet_ratio > bet_pct_normal:
                bet_ratio = bet_pct_normal
            bet_amount = capital * bet_ratio
        cap_limit = capital * 0.95
        bet_amount *= max(0.0, bet_scale_arr[i])
        effective_bet_cap = capital * (bet_pct_conservative if ever_reached_cap else bet_pct_normal)
        if bet_amount > effective_bet_cap:
            bet_amount = effective_bet_cap
        if bet_amount > cap_limit:
            bet_amount = cap_limit
        if bet_amount < 1.0:
            continue
        correct = (direction == 1 and actual == 1) or (direction == 0 and actual == 0)
        realized_notional = bet_amount * realized_fill_ratio
        if realized_notional <= 0.0:
            continue
        if correct:
            pnl = realized_notional * odds_this - realized_notional * settlement_cost
            wins += trade_weight
            cooldown_remaining = 0
        else:
            pnl = -realized_notional
            losses += trade_weight
            if cooldown > 0:
                cooldown_remaining = cooldown
        trade_equiv += trade_weight
        capital += pnl
        pnl_arr[i] = pnl
        if capital > peak_capital:
            peak_capital = capital
        dd = (peak_capital - capital) / peak_capital if peak_capital > 0.0 else 0.0
        if dd > max_drawdown:
            max_drawdown = dd
        if capital <= 0.0:
            capital = 0.0
            break
    return capital, wins, losses, trade_equiv, max_drawdown, ever_reached_cap, pnl_arr


def _summarize(predictions: pd.DataFrame, pnl_arr: np.ndarray, final_capital: float, wins: float, losses: float, trade_equiv: float, max_drawdown: float, ever_reached_cap: bool) -> dict[str, Any]:
    n_trades_equiv = float(max(0.0, trade_equiv))
    total_pnl = float(final_capital - INITIAL_CAPITAL)
    win_rate = float(wins / n_trades_equiv) if n_trades_equiv > 1e-9 else 0.0
    nonzero = pnl_arr[pnl_arr != 0]
    if len(nonzero) > 2:
        sharpe = float(nonzero.mean() / (nonzero.std() + 1e-10) * np.sqrt(252 * 96))
    else:
        sharpe = 0.0
    gross_profit = float(nonzero[nonzero > 0].sum()) if len(nonzero) else 0.0
    gross_loss = abs(float(nonzero[nonzero < 0].sum())) if len(nonzero) else 0.0
    profit_factor = gross_profit / (gross_loss + 1e-10) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    avg_expectancy = float(total_pnl / n_trades_equiv) if n_trades_equiv > 1e-9 else 0.0
    return {
        'final_capital': _round(final_capital, 6),
        'total_pnl': _round(total_pnl, 6),
        'n_trades': int(round(n_trades_equiv)),
        'trade_equiv': _round(n_trades_equiv, 6),
        'wins_equiv': _round(wins, 6),
        'losses_equiv': _round(losses, 6),
        'win_rate': _round(win_rate, 6),
        'max_drawdown': _round(max_drawdown, 6),
        'profit_factor': _round(profit_factor, 6),
        'avg_expectancy': _round(avg_expectancy, 6),
        'sharpe': _round(sharpe, 6),
        'abstain_rate': _round(float(1.0 - (n_trades_equiv / max(1.0, float(len(predictions))))), 6),
        'ever_reached_cap': bool(ever_reached_cap),
    }


def _simulate_core(predictions: pd.DataFrame, params: dict[str, float], prices: np.ndarray, realized_fill_ratio: np.ndarray, trade_weight_arr: np.ndarray) -> dict[str, Any]:
    probas = np.ascontiguousarray(predictions['proba_up'].values, dtype=np.float64)
    actuals = np.ascontiguousarray(predictions['actual'].values, dtype=np.int64)
    odds_arr = np.ascontiguousarray((1.0 - prices) / prices, dtype=np.float64)
    threshold_extra_arr = np.ascontiguousarray(pd.to_numeric(predictions['runtime_threshold_extra'], errors='coerce').fillna(0.0).values, dtype=np.float64)
    direction_blocked_arr = np.ascontiguousarray(pd.to_numeric(predictions['runtime_direction_blocked'], errors='coerce').fillna(0).astype(np.int8).values, dtype=np.int8)
    bet_scale_arr = np.ascontiguousarray(pd.to_numeric(predictions['runtime_bet_scale'], errors='coerce').fillna(1.0).values, dtype=np.float64)
    num_assets = int(predictions['asset'].nunique()) if 'asset' in predictions.columns else 1
    cooldown = int(params['cooldown_bars']) * max(1, num_assets)
    final_capital, wins, losses, trade_equiv, max_drawdown, ever_reached_cap, pnl_arr = _simulate_trading_core_fillmask(
        probas,
        actuals,
        odds_arr,
        realized_fill_ratio,
        trade_weight_arr,
        threshold_extra_arr,
        direction_blocked_arr,
        bet_scale_arr,
        INITIAL_CAPITAL,
        SETTLEMENT_COST,
        float(params['min_confidence']),
        float(params['min_edge']),
        float(params['kelly_frac']),
        float(params['bet_pct_normal']),
        float(params['bet_pct_conservative']),
        float(params['conf_tier1_bound']),
        float(params['conf_tier2_bound']),
        float(params['tier1_mult']),
        float(params['tier2_mult']),
        float(params['tier3_mult']),
        cooldown,
        float(params['drawdown_halt']),
        60000.0,
        3000.0,
        96,
        0.85,
    )
    return _summarize(predictions, pnl_arr, float(final_capital), float(wins), float(losses), float(trade_equiv), float(max_drawdown), bool(ever_reached_cap))


def _simulate_fixed(predictions: pd.DataFrame, params: dict[str, float], price: float, exec_arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    path_min = pd.to_numeric(predictions['path_min_price'], errors='coerce').to_numpy(dtype=np.float64)
    prices = np.full(len(predictions), float(price), dtype=np.float64)
    fill_ratio = np.interp(prices, exec_arrays['price_points'], exec_arrays['expected_fill_ratio'])
    fill_rate = np.interp(prices, exec_arrays['price_points'], exec_arrays['fill_rate'])
    touch = np.isfinite(path_min) & (path_min <= prices)
    realized_fill_ratio = np.ascontiguousarray(fill_ratio * touch.astype(np.float64), dtype=np.float64)
    trade_weight = np.ascontiguousarray(fill_rate * touch.astype(np.float64), dtype=np.float64)
    out = _simulate_core(predictions, params, prices, realized_fill_ratio, trade_weight)
    out['selection_mode'] = 'fixed_price'
    out['selected_buy_price'] = _round(price, 4)
    out['pathCoverageRate'] = _round(float(np.mean(np.isfinite(path_min))) if len(path_min) else 0.0, 6)
    out['fillableRate'] = _round(float(np.mean(touch)) if len(touch) else 0.0, 6)
    out['executionAdjustedFillRatio'] = _round(float(np.mean(realized_fill_ratio)) if len(realized_fill_ratio) else 0.0, 6)
    return out


def _simulate_fixed_runtime_equivalent_queue_check(
    predictions: pd.DataFrame,
    params: dict[str, float],
    price: float,
    exec_arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    path_min = pd.to_numeric(predictions['path_min_price'], errors='coerce').to_numpy(dtype=np.float64)
    prices = np.full(len(predictions), float(price), dtype=np.float64)
    fill_ratio = np.interp(prices, exec_arrays['price_points'], exec_arrays['expected_fill_ratio'])
    fill_rate = np.interp(prices, exec_arrays['price_points'], exec_arrays['fill_rate'])
    queue_wait = np.interp(prices, exec_arrays['price_points'], exec_arrays['avg_queue_wait_seconds'])
    touch = np.isfinite(path_min) & (path_min <= prices)
    realized_fill_ratio = np.ascontiguousarray(fill_ratio * QUEUE_AWARE_RUNTIME_HAIRCUT * touch.astype(np.float64), dtype=np.float64)
    trade_weight = np.ascontiguousarray(fill_rate * touch.astype(np.float64), dtype=np.float64)
    out = _simulate_core(predictions, params, prices, realized_fill_ratio, trade_weight)
    out['selection_mode'] = 'fixed_price'
    out['selected_buy_price'] = _round(price, 4)
    out['pathCoverageRate'] = _round(float(np.mean(np.isfinite(path_min))) if len(path_min) else 0.0, 6)
    out['fillableRate'] = _round(float(np.mean(touch)) if len(touch) else 0.0, 6)
    out['executionAdjustedFillRatio'] = _round(float(np.mean(realized_fill_ratio)) if len(realized_fill_ratio) else 0.0, 6)
    out['runtimeEquivalentQueueCheck'] = {
        'mode': 'simulation_only',
        'haircut': QUEUE_AWARE_RUNTIME_HAIRCUT,
        'uses_expected_fill_ratio_proxy': True,
        'uses_fill_rate_trade_weight': True,
        'uses_avg_queue_wait_seconds_reference': _round(float(np.mean(queue_wait)) if len(queue_wait) else 0.0, 6),
    }
    return out


def _simulate_dynamic(predictions: pd.DataFrame, params: dict[str, float], price_range: list[float], exec_arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    path_min = pd.to_numeric(predictions['path_min_price'], errors='coerce').to_numpy(dtype=np.float64)
    rungs = np.array([round(x, 2) for x in np.arange(price_range[0], price_range[1] + 0.001, 0.01)], dtype=np.float64)
    rung_fill_ratio = np.interp(rungs, exec_arrays['price_points'], exec_arrays['expected_fill_ratio'])
    rung_fill_rate = np.interp(rungs, exec_arrays['price_points'], exec_arrays['fill_rate'])
    if len(path_min):
        touch_matrix = (path_min.reshape(-1, 1) <= rungs.reshape(1, -1)) & np.isfinite(path_min).reshape(-1, 1)
        realized_matrix = touch_matrix.astype(np.float64) * rung_fill_ratio.reshape(1, -1)
        weight_matrix = touch_matrix.astype(np.float64) * rung_fill_rate.reshape(1, -1)
        realized_fill_ratio = realized_matrix.mean(axis=1)
        trade_weight = weight_matrix.mean(axis=1)
        weighted_price_num = (realized_matrix * rungs.reshape(1, -1)).sum(axis=1)
        weighted_price_den = realized_matrix.sum(axis=1)
        midpoint = np.full(len(path_min), round((price_range[0] + price_range[1]) / 2.0, 4), dtype=np.float64)
        prices = np.divide(weighted_price_num, weighted_price_den, out=midpoint.copy(), where=weighted_price_den > 0)
        touch = touch_matrix.any(axis=1)
    else:
        realized_fill_ratio = np.zeros(0, dtype=np.float64)
        trade_weight = np.zeros(0, dtype=np.float64)
        prices = np.zeros(0, dtype=np.float64)
        touch = np.zeros(0, dtype=np.bool_)
    out = _simulate_core(predictions, params, np.ascontiguousarray(prices, dtype=np.float64), np.ascontiguousarray(realized_fill_ratio, dtype=np.float64), np.ascontiguousarray(trade_weight, dtype=np.float64))
    out['selection_mode'] = 'dynamic_range'
    out['selected_buy_price_range'] = [round(float(price_range[0]), 2), round(float(price_range[1]), 2)]
    out['dynamic_ladder'] = format_ladder_from_range(price_range)
    out['pathCoverageRate'] = _round(float(np.mean(np.isfinite(path_min))) if len(path_min) else 0.0, 6)
    out['fillableRate'] = _round(float(np.mean(touch)) if len(touch) else 0.0, 6)
    out['executionAdjustedFillRatio'] = _round(float(np.mean(realized_fill_ratio)) if len(realized_fill_ratio) else 0.0, 6)
    return out


def _split_train_holdout(df: pd.DataFrame, holdout_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values('timestamp').reset_index(drop=True)
    cutoff = pd.to_datetime(df['timestamp']).max() - pd.Timedelta(days=holdout_days)
    train_df = df[pd.to_datetime(df['timestamp']) < cutoff].copy().reset_index(drop=True)
    holdout_df = df[pd.to_datetime(df['timestamp']) >= cutoff].copy().reset_index(drop=True)
    return train_df, holdout_df


def _candidate_sort_key(metrics: dict[str, Any]) -> tuple[float, float, float, int]:
    return (
        float(metrics.get('total_pnl') or 0.0),
        -float(metrics.get('max_drawdown') or 0.0),
        float(metrics.get('avg_expectancy') or 0.0),
        int(metrics.get('n_trades') or 0),
    )


def _fixed_candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, float, float, int, float]:
    holdout = candidate.get('holdout') if isinstance(candidate.get('holdout'), dict) else {}
    train = candidate.get('train') if isinstance(candidate.get('train'), dict) else {}
    return (
        float(holdout.get('total_pnl') or 0.0),
        -float(holdout.get('max_drawdown') or 0.0),
        float(holdout.get('avg_expectancy') or 0.0),
        int(holdout.get('n_trades') or 0),
        float(train.get('total_pnl') or 0.0),
    )


def _fixed_candidate_gate(symbol: str, holdout_metrics: dict[str, Any]) -> tuple[bool, list[str], dict[str, float]]:
    symbol = str(symbol or '').upper()
    thresholds = {
        'min_holdout_trades': int(MIN_HOLDOUT_TRADES.get(symbol, 30)),
        'min_execution_adjusted_fill_ratio': float(MIN_EXECUTION_ADJUSTED_FILL_RATIO.get(symbol, 0.03)),
        'min_fillable_rate': float(MIN_FILLABLE_RATE.get(symbol, 0.03)),
    }
    reasons: list[str] = []
    holdout_trades = int(holdout_metrics.get('n_trades') or 0)
    exec_ratio = float(holdout_metrics.get('executionAdjustedFillRatio') or 0.0)
    fillable_rate = float(holdout_metrics.get('fillableRate') or 0.0)
    if holdout_trades < thresholds['min_holdout_trades']:
        reasons.append('holdout_trade_sample_below_floor')
    if exec_ratio < thresholds['min_execution_adjusted_fill_ratio']:
        reasons.append('execution_adjusted_fill_ratio_below_floor')
    if fillable_rate < thresholds['min_fillable_rate']:
        reasons.append('fillable_rate_below_floor')
    return (len(reasons) == 0), reasons, thresholds


def _fixed_runtime_equivalent_queue_gate(symbol: str, holdout_metrics: dict[str, Any]) -> tuple[bool, list[str], dict[str, float]]:
    eligible, reasons, thresholds = _fixed_candidate_gate(symbol, holdout_metrics)
    runtime_thresholds = dict(thresholds)
    runtime_thresholds['queue_haircut'] = QUEUE_AWARE_RUNTIME_HAIRCUT
    runtime_reasons = [f'runtime_queue_check:{reason}' for reason in reasons]
    return eligible, runtime_reasons, runtime_thresholds


def _edge_winner_explained(winner: dict[str, Any] | None, fixed_candidates: list[dict[str, Any]]) -> bool | None:
    if not winner:
        return None
    try:
        winner_price = round(float((winner.get('candidate') or {}).get('buy_price')), 2)
    except Exception:
        return None
    if winner_price not in {PRICE_LEVELS[0], PRICE_LEVELS[-1]}:
        return None
    neighbor_target = round(winner_price + 0.01, 2) if winner_price == PRICE_LEVELS[0] else round(winner_price - 0.01, 2)
    neighbor = next(
        (
            candidate for candidate in fixed_candidates
            if round(float((candidate.get('candidate') or {}).get('buy_price')), 2) == neighbor_target
        ),
        None,
    )
    if not neighbor:
        return False
    return _fixed_candidate_sort_key(winner) > _fixed_candidate_sort_key(neighbor)


def _fixed_selection_status(
    eligible_candidates: list[dict[str, Any]],
    winner: dict[str, Any] | None,
    fixed_candidates: list[dict[str, Any]],
) -> tuple[str, bool | None]:
    if not eligible_candidates or winner is None:
        return 'sample_thin_keep_under_watch', None
    edge_explained = _edge_winner_explained(winner, fixed_candidates)
    if edge_explained is True:
        return 'edge_winner_but_explained', True
    if edge_explained is False:
        return 'selection_science_gap', False
    return 'scientifically_stable', None


def _optimize_one(
    baseline: dict[str, Any],
    refresh_predictions: bool,
    xrp_path_mode: str | None = None,
    *,
    candidate_tag: str | None = None,
    stale_order_policy: str = 'off',
    stale_cancel_min_age_sec: int = 0,
    stale_cancel_min_drift_ticks: int = 0,
    stale_cancel_min_time_to_expiry_sec: int = 0,
) -> dict[str, Any]:
    predictions = _load_or_generate_predictions(baseline, TOTAL_DAYS, refresh_predictions)
    predictions, runtime_modeling = _apply_runtime_like_controls(predictions, baseline)
    predictions, path_source = _attach_path_touch(predictions, baseline['symbol'], xrp_path_mode=xrp_path_mode)
    train_df, holdout_df = _split_train_holdout(predictions, HOLDOUT_DAYS)
    exec_arrays = _build_execution_arrays(_build_execution_rows(baseline['symbol']))
    params = baseline['runtime_params']

    candidates: list[dict[str, Any]] = []
    if baseline['resolved_price_mode'] == 'fixed':
        for price in PRICE_LEVELS:
            train_metrics = _simulate_fixed(train_df, params, price, exec_arrays)
            holdout_metrics = _simulate_fixed(holdout_df, params, price, exec_arrays)
            full_metrics = _simulate_fixed(predictions, params, price, exec_arrays)
            queue_train_metrics = _simulate_fixed_runtime_equivalent_queue_check(train_df, params, price, exec_arrays)
            queue_holdout_metrics = _simulate_fixed_runtime_equivalent_queue_check(holdout_df, params, price, exec_arrays)
            queue_full_metrics = _simulate_fixed_runtime_equivalent_queue_check(predictions, params, price, exec_arrays)
            base_eligible_for_finalist, base_ineligibility_reasons, thresholds = _fixed_candidate_gate(baseline['symbol'], holdout_metrics)
            queue_check_passed, queue_check_reasons, queue_thresholds = _fixed_runtime_equivalent_queue_gate(baseline['symbol'], queue_holdout_metrics)
            eligible_for_finalist = bool(base_eligible_for_finalist and queue_check_passed)
            ineligibility_reasons = [*base_ineligibility_reasons, *queue_check_reasons]
            candidates.append({
                'selection_mode': 'fixed_price',
                'candidate': {'buy_price': price},
                'train': train_metrics,
                'holdout': holdout_metrics,
                'full': full_metrics,
                'runtime_equivalent_queue_check': {
                    'train': queue_train_metrics,
                    'holdout': queue_holdout_metrics,
                    'full': queue_full_metrics,
                    'passed': queue_check_passed,
                    'reasons': queue_check_reasons,
                    'thresholds': queue_thresholds,
                    'mode': 'simulation_only',
                },
                'base_eligible_for_finalist': base_eligible_for_finalist,
                'base_ineligibility_reasons': base_ineligibility_reasons,
                'eligible_for_finalist': eligible_for_finalist,
                'ineligibility_reasons': ineligibility_reasons,
                'eligibility_thresholds': thresholds,
            })
    else:
        price_range = list(DYNAMIC_FULL_LADDER_RANGE)
        train_metrics = _simulate_dynamic(train_df, params, price_range, exec_arrays)
        holdout_metrics = _simulate_dynamic(holdout_df, params, price_range, exec_arrays)
        full_metrics = _simulate_dynamic(predictions, params, price_range, exec_arrays)
        candidates.append({
            'selection_mode': 'dynamic_range',
            'candidate': {'buy_price_range': price_range},
            'train': train_metrics,
            'holdout': holdout_metrics,
            'full': full_metrics,
        })

    if baseline['resolved_price_mode'] == 'fixed':
        candidates.sort(key=_fixed_candidate_sort_key, reverse=True)
        eligible_candidates = [candidate for candidate in candidates if bool(candidate.get('eligible_for_finalist'))]
        finalists = eligible_candidates[:FIXED_FINALIST_COUNT]
        winner = eligible_candidates[0] if eligible_candidates else None
        winner_selection_basis = 'holdout_first_train_tiebreak'
        fixed_price_selection_status, edge_winner_explained = _fixed_selection_status(eligible_candidates, winner, candidates)
    else:
        candidates.sort(key=lambda x: _candidate_sort_key(x['train']), reverse=True)
        finalists = candidates[:1]
        winner = candidates[0]
        winner_selection_basis = 'dynamic_range_contract_unchanged'
        fixed_price_selection_status = 'dynamic_range_contract_unchanged'
        edge_winner_explained = None

    runtime_finalists: list[dict[str, Any]] = []
    for idx, finalist in enumerate(finalists, start=1):
        if finalist['selection_mode'] == 'fixed_price':
            selected_price = float(finalist['candidate']['buy_price'])
            selected_range = None
            finalist_key = f'fixed_bp{int(round(selected_price * 1000)):04d}'
        else:
            selected_price = None
            selected_range = [round(float(x), 2) for x in finalist['candidate']['buy_price_range']]
            finalist_key = f'dynamic_{int(round(selected_range[0] * 1000)):04d}_{int(round(selected_range[1] * 1000)):04d}'
        family_rule_version = derive_lowprice_family_rule_version(candidate_tag)
        rules_path = _write_rules_file(
            baseline['profile'],
            baseline['source_trader'],
            baseline['symbol'],
            finalist['selection_mode'],
            selected_price,
            selected_range,
            baseline,
            finalist,
            candidate_tag=candidate_tag,
            lowprice_family_id=finalist_key,
            family_rule_version=family_rule_version,
            stale_order_policy=stale_order_policy,
            stale_cancel_min_age_sec=stale_cancel_min_age_sec,
            stale_cancel_min_drift_ticks=stale_cancel_min_drift_ticks,
            stale_cancel_min_time_to_expiry_sec=stale_cancel_min_time_to_expiry_sec,
        )
        runtime_finalists.append({
            'finalist_rank': idx,
            'finalist_key': finalist_key,
            'lowpriceFamilyId': finalist_key,
            'familyRuleVersion': family_rule_version,
            'selection_mode': finalist['selection_mode'],
            'selected_buy_price': selected_price,
            'selected_buy_price_range': selected_range,
            'dynamic_ladder': format_ladder_from_levels(PRICE_LEVELS) if finalist['selection_mode'] == 'dynamic_range' else None,
            'train_metrics': finalist['train'],
            'holdout_metrics': finalist['holdout'],
            'full_metrics': finalist['full'],
            'rulesJsonPath': rules_path,
            'finalist_for_compare_only': True,
            'eligible_for_finalist': finalist.get('eligible_for_finalist'),
            'base_eligible_for_finalist': finalist.get('base_eligible_for_finalist'),
            'ineligibility_reasons': finalist.get('ineligibility_reasons'),
            'base_ineligibility_reasons': finalist.get('base_ineligibility_reasons'),
            'eligibility_thresholds': finalist.get('eligibility_thresholds'),
            'runtime_equivalent_queue_check': finalist.get('runtime_equivalent_queue_check'),
            'staleOrderPolicy': stale_order_policy,
            'staleCancelMinAgeSec': int(max(0, stale_cancel_min_age_sec)),
            'staleCancelMinDriftTicks': int(max(0, stale_cancel_min_drift_ticks)),
            'staleCancelMinTimeToExpirySec': int(max(0, stale_cancel_min_time_to_expiry_sec)),
        })

    if winner and winner['selection_mode'] == 'fixed_price':
        best_selected_price = float(winner['candidate']['buy_price'])
        best_selected_range = None
    elif winner and winner['selection_mode'] == 'dynamic_range':
        best_selected_price = None
        best_selected_range = [round(float(x), 2) for x in winner['candidate']['buy_price_range']]
    else:
        best_selected_price = None
        best_selected_range = None
    return {
        'profile': baseline['profile'],
        'source_trader': baseline['source_trader'],
        'symbol': baseline['symbol'],
        'resolved_price_mode': baseline['resolved_price_mode'],
        'resolved_source_buy_price': baseline['resolved_source_buy_price'],
        'resolved_source_buy_price_range': baseline['resolved_source_buy_price_range'],
        'resolved_source_sizing_reference_price': baseline['resolved_source_sizing_reference_price'],
        'config_rules_mismatch_flags': baseline['config_rules_mismatch_flags'],
        'selection_mode': winner['selection_mode'] if winner else 'fixed_price',
        'selected_buy_price': best_selected_price,
        'selected_buy_price_range': best_selected_range,
        'selection_basis': (
            'offline_screen_then_compare_only_with_runtime_equivalent_queue_check'
            if baseline['resolved_price_mode'] == 'fixed'
            else 'offline_screen_then_compare_only'
        ),
        'candidate_tag': str(candidate_tag or ''),
        'family_rule_version': derive_lowprice_family_rule_version(candidate_tag),
        'stale_order_policy': stale_order_policy,
        'stale_cancel_min_age_sec': int(max(0, stale_cancel_min_age_sec)),
        'stale_cancel_min_drift_ticks': int(max(0, stale_cancel_min_drift_ticks)),
        'stale_cancel_min_time_to_expiry_sec': int(max(0, stale_cancel_min_time_to_expiry_sec)),
        'winner_selection_basis': winner_selection_basis,
        'runtime_equivalent_queue_check_required_for_finalists': baseline['resolved_price_mode'] == 'fixed',
        'eligible_candidate_count': sum(1 for candidate in candidates if bool(candidate.get('eligible_for_finalist'))) if baseline['resolved_price_mode'] == 'fixed' else None,
        'winner_is_grid_edge': (
            best_selected_price in {PRICE_LEVELS[0], PRICE_LEVELS[-1]}
            if baseline['resolved_price_mode'] == 'fixed' and best_selected_price is not None
            else None
        ),
        'edge_winner_explained': edge_winner_explained,
        'fixed_price_selection_status': fixed_price_selection_status,
        'path_source': path_source,
        'runtime_modeling': runtime_modeling,
        'train_metrics': winner['train'] if winner else None,
        'holdout_metrics': winner['holdout'] if winner else None,
        'full_metrics': winner['full'] if winner else None,
        'candidate_count': len(candidates),
        'candidates': candidates,
        'runtime_finalists': runtime_finalists,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description='Optimize selected low-price exp cells using 180d 1m path.')
    ap.add_argument('--refresh-predictions', action='store_true')
    ap.add_argument('--profiles', default='', help='Comma-separated profiles to include (default,70)')
    ap.add_argument('--symbols', default='', help='Comma-separated symbols to include (e.g. BTC,ETH)')
    ap.add_argument('--source-traders', default='', help='Comma-separated source traders to include')
    ap.add_argument('--active-lowprice-families', action='store_true', help='Discover baselines from current active lowprice compare/live families instead of static selected cells')
    ap.add_argument('--candidate-tag', default='', help='Write rules into candidate namespace instead of incumbent path')
    ap.add_argument('--output-report', default='', help='Optional alternate optimization report path')
    ap.add_argument('--output-baseline-report', default='', help='Optional alternate baseline report path')
    ap.add_argument('--stale-order-policy', default='off', choices=['off', 'cancel_only_when_stale'])
    ap.add_argument('--stale-cancel-min-age-sec', type=int, default=0)
    ap.add_argument('--stale-cancel-min-drift-ticks', type=int, default=0)
    ap.add_argument('--stale-cancel-min-time-to-expiry-sec', type=int, default=0)
    args = ap.parse_args()

    baseline_report_path = Path(args.output_baseline_report).expanduser() if args.output_baseline_report else BASELINE_REPORT
    optimization_report_path = Path(args.output_report).expanduser() if args.output_report else OPT_REPORT

    allowed_profiles = _parse_csv_set(args.profiles) or set(PROFILES.keys())
    allowed_symbols = _parse_upper_csv_set(args.symbols)
    allowed_source_traders = _parse_csv_set(args.source_traders)

    baseline_source = iter_active_lowprice_source_baselines(
        profiles=sorted(allowed_profiles),
        symbols=sorted(allowed_symbols),
    ) if args.active_lowprice_families else iter_selected_baselines()
    baselines = [
        baseline for baseline in baseline_source
        if baseline['profile'] in allowed_profiles
        and (not allowed_symbols or str(baseline['symbol']).upper() in allowed_symbols)
        and (not allowed_source_traders or str(baseline['source_trader']) in allowed_source_traders)
    ]
    if not baselines:
        raise SystemExit('no baselines matched the requested lowprice tuning scope')

    xrp_path_mode, xrp_path_mode_metadata = _resolve_xrp_path_mode(baselines, args.refresh_predictions)
    baseline_payload = {
        'generatedAt': now_iso(),
        'days': TOTAL_DAYS,
        'holdoutDays': HOLDOUT_DAYS,
        'xrpPathMode': xrp_path_mode,
        'xrpPathModeMetadata': xrp_path_mode_metadata,
        'profiles': {profile: [] for profile in PROFILES},
    }
    for baseline in baselines:
        baseline_payload['profiles'][baseline['profile']].append({
            k: v for k, v in baseline.items() if k not in {'source_row', 'resolved_non_price_rules', 'runtime_params'}
        })
    dump_json(baseline_report_path, baseline_payload)

    results = {
        'generatedAt': now_iso(),
        'days': TOTAL_DAYS,
        'holdoutDays': HOLDOUT_DAYS,
        'candidateTag': str(args.candidate_tag or ''),
        'profilesRequested': sorted(allowed_profiles),
        'symbolsRequested': sorted(allowed_symbols),
        'sourceTradersRequested': sorted(allowed_source_traders),
        'staleOrderPolicy': str(args.stale_order_policy),
        'staleCancelMinAgeSec': int(max(0, args.stale_cancel_min_age_sec)),
        'staleCancelMinDriftTicks': int(max(0, args.stale_cancel_min_drift_ticks)),
        'staleCancelMinTimeToExpirySec': int(max(0, args.stale_cancel_min_time_to_expiry_sec)),
        'xrpPathMode': xrp_path_mode,
        'xrpPathModeMetadata': xrp_path_mode_metadata,
        'profiles': {profile: [] for profile in PROFILES},
    }
    for baseline in baselines:
        result = _optimize_one(
            baseline,
            args.refresh_predictions,
            xrp_path_mode=xrp_path_mode,
            candidate_tag=args.candidate_tag,
            stale_order_policy=args.stale_order_policy,
            stale_cancel_min_age_sec=int(max(0, args.stale_cancel_min_age_sec)),
            stale_cancel_min_drift_ticks=int(max(0, args.stale_cancel_min_drift_ticks)),
            stale_cancel_min_time_to_expiry_sec=int(max(0, args.stale_cancel_min_time_to_expiry_sec)),
        )
        results['profiles'][baseline['profile']].append(result)
    results['ok'] = True
    dump_json(optimization_report_path, results)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
