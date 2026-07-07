#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.python.feature_engineering import add_multi_timeframe_features
from experiments.sentiment_grid_search.data_prep import FEATURE_GROUPS
from experiments.sentiment_grid_search.run_grid import build_tech_features, get_data_paths, merge_sentiment_to_tech
from scripts.vnext_entry_replay import (
    EntryExecutionTrace,
    market_end_from_start_ts,
    replay_entry_trace,
)

REPORTS = PROJECT_ROOT / 'reports'
MODELS = PROJECT_ROOT / 'data' / 'models'
PROCESSED = PROJECT_ROOT / 'data' / 'processed'
DATA_DIR = PROJECT_ROOT / 'data'
WALLET_PANEL = PROJECT_ROOT / 'reports' / 'wallet_cohort_panel_round3_top200.parquet'

BTC_MODEL_DIR = MODELS / 'vnext_btc_entry_exit_v1'
ETH_MODEL_DIR = MODELS / 'vnext_eth_entry_exit_v1'
BTC_EPISODES = PROCESSED / 'vnext_entry_exit_episodes_btc_usdt.parquet'
ETH_EPISODES = PROCESSED / 'vnext_entry_exit_episodes_eth_usdt.parquet'
BTC_LEADERBOARD = REPORTS / 'vnext_btc_entry_exit_v1_leaderboard_latest.json'
ETH_LEADERBOARD = REPORTS / 'vnext_eth_entry_exit_v1_leaderboard_latest.json'
COMPARE_REPORT = REPORTS / 'vnext_btceth_entry_exit_v1_compare_latest.json'
TRAIN_REPORT = REPORTS / 'vnext_btceth_entry_exit_v1_training_latest.json'

SEARCH_WORKERS = 8
FEATURE_COMPLETE_DAYS = 180
MIN_EFFECTIVE_FEATURE_DAYS = 120
VAL_DAYS = 14
TEST_DAYS = 14
EMBARGO_BARS = 8
MIN_TRAIN_ROWS = 1200
MIN_VAL_ROWS = 240
MIN_TEST_ROWS = 240
TRAIN_DAYS_CANDIDATES = [90, 120, 150]
ENABLE_WALLET_COHORT_ABLATION = os.getenv('VNEXT_ENABLE_WALLET_COHORT_ABLATION', '0').strip().lower() in {'1', 'true', 'yes', 'on'}
BUY_GRIDS = {
    'BTC_USDT': [0.42, 0.44, 0.46, 0.48, 0.50, 0.52, 0.54, 0.56],
    'ETH_USDT': [0.41, 0.43, 0.45, 0.47, 0.49, 0.51, 0.53, 0.55],
}
EXIT_FAMILIES = [
    'HOLD_TO_EXPIRY',
    'TIME_EXIT',
    'STOP_LOSS',
    'TAKE_PROFIT',
    'TRAILING_EXIT',
    'PROFIT_PROTECT',
    'FAST_FAIL_EXIT',
]
SELL_QUOTE_POLICIES = [
    'HIT_BID',
    'MID_MINUS_0.01',
    'MID_MINUS_0.02',
    'MID_MINUS_SPREAD_HALF',
    'MID_MINUS_SPREAD_ONE',
]
PARAM_TEMPLATES: dict[str, dict[str, Any]] = {
    'conservative': {
        'num_leaves': 31, 'max_depth': 5, 'learning_rate': 0.03, 'min_child_samples': 80,
        'feature_fraction': 0.70, 'bagging_fraction': 0.75, 'bagging_freq': 2,
        'lambda_l1': 0.30, 'lambda_l2': 12.0, 'max_bin': 255, 'min_split_gain': 0.05,
    },
    'balanced': {
        'num_leaves': 63, 'max_depth': 6, 'learning_rate': 0.02, 'min_child_samples': 50,
        'feature_fraction': 0.80, 'bagging_fraction': 0.85, 'bagging_freq': 2,
        'lambda_l1': 0.10, 'lambda_l2': 10.0, 'max_bin': 255, 'min_split_gain': 0.03,
    },
    'elastic': {
        'num_leaves': 95, 'max_depth': 8, 'learning_rate': 0.015, 'min_child_samples': 30,
        'feature_fraction': 0.90, 'bagging_fraction': 0.90, 'bagging_freq': 1,
        'lambda_l1': 0.05, 'lambda_l2': 6.0, 'max_bin': 383, 'min_split_gain': 0.01,
    },
    'deep_regularized': {
        'num_leaves': 127, 'max_depth': 9, 'learning_rate': 0.012, 'min_child_samples': 20,
        'feature_fraction': 0.95, 'bagging_fraction': 0.90, 'bagging_freq': 1,
        'lambda_l1': 0.15, 'lambda_l2': 14.0, 'max_bin': 511, 'min_split_gain': 0.08,
    },
}
BTC_RECIPES = [
    {
        'name': 'btc_fast_trend',
        'macd': {'fast': 8, 'slow': 21, 'signal': 5},
        'rsi_periods': [6, 12],
        'bollinger': {'period': 14, 'mult': 1.8},
        'volume_period': 14,
        'stochastic': {'k_period': 9, 'd_period': 3},
        'cci_period': 14,
        'mfi_period': 10,
        'donchian_period': 10,
        'roc_periods': [1, 2, 4, 8, 16],
        'atr_periods': [7, 14],
        'price_position_periods': [5, 10, 20, 40],
        'volatility_periods': [5, 10, 20],
        'lag_periods': [1, 2, 4, 8],
        'ema_periods': [8, 21, 55],
        'ema_slope_periods': [8, 21, 55],
        'adx_period': 10,
        'bullish_windows': [3, 5, 7],
    },
    {
        'name': 'btc_balanced',
        'macd': {'fast': 12, 'slow': 26, 'signal': 9},
        'rsi_periods': [7, 14],
        'bollinger': {'period': 20, 'mult': 2.0},
        'volume_period': 20,
        'stochastic': {'k_period': 14, 'd_period': 3},
        'cci_period': 20,
        'mfi_period': 14,
        'donchian_period': 20,
        'roc_periods': [1, 3, 5, 10, 20],
        'atr_periods': [14, 20],
        'price_position_periods': [5, 10, 20, 50],
        'volatility_periods': [5, 10, 20],
        'lag_periods': [1, 2, 3, 5, 10],
        'ema_periods': [5, 10, 20, 50, 100],
        'ema_slope_periods': [10, 20, 50],
        'adx_period': 14,
        'bullish_windows': [3, 5, 7, 9, 11],
    },
    {
        'name': 'btc_slow_regime',
        'macd': {'fast': 16, 'slow': 35, 'signal': 9},
        'rsi_periods': [14, 21],
        'bollinger': {'period': 30, 'mult': 2.2},
        'volume_period': 30,
        'stochastic': {'k_period': 21, 'd_period': 5},
        'cci_period': 30,
        'mfi_period': 21,
        'donchian_period': 30,
        'roc_periods': [1, 5, 15, 30],
        'atr_periods': [14, 21],
        'price_position_periods': [10, 20, 40, 80],
        'volatility_periods': [10, 20, 40],
        'lag_periods': [1, 3, 6, 12],
        'ema_periods': [10, 20, 50, 100],
        'ema_slope_periods': [20, 50, 100],
        'adx_period': 21,
        'bullish_windows': [5, 10, 20],
    },
]
ETH_RECIPES = [
    {
        'name': 'eth_micro',
        'macd': {'fast': 8, 'slow': 21, 'signal': 5},
        'rsi_periods': [6, 12],
        'bollinger': {'period': 14, 'mult': 1.8},
        'volume_period': 14,
        'stochastic': {'k_period': 9, 'd_period': 3},
        'cci_period': 14,
        'mfi_period': 10,
        'donchian_period': 10,
        'roc_periods': [1, 2, 4, 8, 16],
        'atr_periods': [7, 14],
        'price_position_periods': [5, 10, 20, 40],
        'volatility_periods': [5, 10, 20],
        'lag_periods': [1, 2, 4, 8],
        'ema_periods': [8, 21, 55],
        'ema_slope_periods': [8, 21, 55],
        'adx_period': 10,
        'bullish_windows': [3, 5, 7],
    },
    {
        'name': 'eth_balanced',
        'macd': {'fast': 12, 'slow': 26, 'signal': 9},
        'rsi_periods': [7, 14],
        'bollinger': {'period': 20, 'mult': 2.0},
        'volume_period': 20,
        'stochastic': {'k_period': 14, 'd_period': 3},
        'cci_period': 20,
        'mfi_period': 14,
        'donchian_period': 20,
        'roc_periods': [1, 3, 5, 10, 20],
        'atr_periods': [14, 20],
        'price_position_periods': [5, 10, 20, 50],
        'volatility_periods': [5, 10, 20],
        'lag_periods': [1, 2, 3, 5, 10],
        'ema_periods': [5, 10, 20, 50, 100],
        'ema_slope_periods': [10, 20, 50],
        'adx_period': 14,
        'bullish_windows': [3, 5, 7, 9, 11],
    },
    {
        'name': 'eth_regime',
        'macd': {'fast': 16, 'slow': 35, 'signal': 9},
        'rsi_periods': [14, 21],
        'bollinger': {'period': 30, 'mult': 2.2},
        'volume_period': 30,
        'stochastic': {'k_period': 21, 'd_period': 5},
        'cci_period': 30,
        'mfi_period': 21,
        'donchian_period': 30,
        'roc_periods': [1, 5, 15, 30],
        'atr_periods': [14, 21],
        'price_position_periods': [10, 20, 40, 80],
        'volatility_periods': [10, 20, 40],
        'lag_periods': [1, 3, 6, 12],
        'ema_periods': [10, 20, 50, 100],
        'ema_slope_periods': [20, 50, 100],
        'adx_period': 21,
        'bullish_windows': [5, 10, 20],
    },
]
ASSET_RECIPES = {'BTC_USDT': BTC_RECIPES, 'ETH_USDT': ETH_RECIPES}
ASSET_GROUPS = {
    'BTC_USDT': {
        'btc_core': ['ob', 'funding', 'oi', 'lsratio', 'polymarket_prob'],
        'btc_core_fgi': ['fgi_daily', 'ob', 'funding', 'oi', 'lsratio', 'polymarket_prob'],
    },
    'ETH_USDT': {
        'eth_core': ['ob', 'funding', 'oi', 'lsratio', 'polymarket_prob'],
        'eth_core_fgi': ['fgi_daily', 'ob', 'funding', 'oi', 'lsratio', 'polymarket_prob'],
    },
}
TECH_EXCLUDE = {
    'timestamp', 'timestamp_ms', 'date', 'symbol', 'open', 'high', 'low', 'close', 'volume',
    'target_start_ts', 'market_start_ts', 'market_end_ts', 'prediction_ts', 'next_close', 'next_return',
    'gate_label', 'direction_target', 'buy_label', 'best_utility', 'baseline_hold_pnl',
    'entry_fill_proxy', 'entry_fill_status', 'entry_fill_ts', 'entry_fill_fraction',
    'path_t_sec_json', 'path_up_json', 'entry_proxy_ts', 'decision_ts', 'target_period_end_ts',
    'market_discovered_ts', 'order_submitted_ts', 'timeout_ts', 'best_exit_family',
    'best_sell_quote_policy', 'best_action_key', 'entry_action_meta_json',
    'entry_action_utility_json', 'entry_trace_json', 'sample_weight', 'vol_metric', 'spread_metric',
    'trend_score', 'actual_up',
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def _round_float(v: Any, digits: int = 6) -> Any:
    try:
        fv = float(v)
    except Exception:
        return v
    if math.isnan(fv) or math.isinf(fv):
        return None
    return round(fv, digits)


def polymarket_taker_fee_rate(price: float) -> float:
    p = float(price)
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return 0.25 * (p * (1.0 - p)) ** 2


def _recent_weight_multiplier(ts: pd.Series) -> np.ndarray:
    latest = ts.max()
    age_days = (latest - ts).dt.total_seconds() / 86400.0
    out = np.ones(len(ts), dtype=float)
    out = np.where(age_days <= 30.0, out * 1.3, out)
    out = np.where(age_days <= 14.0, out * (1.8 / 1.3), out)
    out = np.where(age_days <= 3.0, out * (2.5 / 1.8), out)
    return out


def _wallet_panel_available() -> bool:
    return WALLET_PANEL.exists()


def _asset_bundle_map(asset: str) -> dict[str, list[str]]:
    mapping = dict(ASSET_GROUPS[asset])
    if asset == 'ETH_USDT' and ENABLE_WALLET_COHORT_ABLATION and _wallet_panel_available():
        mapping['eth_wallet'] = ['fgi_daily', 'ob', 'funding', 'oi', 'lsratio', 'polymarket_prob', 'wallet_cohort']
    return mapping


def _required_group_cols(groups: Sequence[str]) -> set[str]:
    cols: set[str] = set()
    for group in groups:
        cols.update(FEATURE_GROUPS.get(group, []))
    return cols


def _choose_feature_cols(df: pd.DataFrame, groups: Sequence[str]) -> list[str]:
    selected: list[str] = []
    required_cols = _required_group_cols(groups)
    optional_group_cols = set().union(*[set(v) for k, v in FEATURE_GROUPS.items() if k not in {'base', *groups}])
    for col in df.columns:
        if col in TECH_EXCLUDE:
            continue
        if col == 'target_prob_quality' or col == 'target_prob_source':
            continue
        if col.startswith('target_'):
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and not df[col].isna().all():
            if col in optional_group_cols and col not in required_cols:
                continue
            selected.append(col)
    return sorted(dict.fromkeys(selected))


def _candidate_train_days(effective_feature_days: int) -> list[int]:
    usable_days = int(max(0, effective_feature_days) - VAL_DAYS - TEST_DAYS)
    candidates = [int(v) for v in TRAIN_DAYS_CANDIDATES if int(v) <= usable_days]
    if not candidates and usable_days >= 60:
        candidates = [int(usable_days)]
    return sorted(dict.fromkeys(v for v in candidates if v >= 60))


def _build_asset_frame(asset: str, recipe: dict[str, Any], groups: Sequence[str]) -> pd.DataFrame:
    paths = get_data_paths()
    tech = build_tech_features(paths['data_src'], asset, tail_rows=0, freq_min=15, recipe=recipe)
    tech = add_multi_timeframe_features(tech, asset)
    merged = merge_sentiment_to_tech(tech, list(groups), paths, asset)
    merged = merged.sort_values('timestamp').reset_index(drop=True)
    merged['timestamp'] = pd.to_datetime(merged['timestamp'], utc=True)
    merged = merged[merged['timestamp'] >= (merged['timestamp'].max() - pd.Timedelta(days=FEATURE_COMPLETE_DAYS + 5))].reset_index(drop=True)
    return merged


def _load_path_groups(asset: str) -> dict[int, tuple[list[int], list[float]]]:
    path = DATA_DIR / f'polymarket_1m_{asset.lower()}.parquet'
    if not path.exists():
        raise FileNotFoundError(f'missing 1m path file: {path}')
    df = pd.read_parquet(path)
    if not {'15m_start_ts', 't_sec', 'p'}.issubset(df.columns):
        raise ValueError(f'invalid path file columns for {path}')
    df = df.sort_values(['15m_start_ts', 't_sec']).reset_index(drop=True)
    out: dict[int, tuple[list[int], list[float]]] = {}
    for bar_ts, group in df.groupby('15m_start_ts', sort=True):
        out[int(bar_ts)] = (
            [int(v) for v in group['t_sec'].tolist()],
            [float(v) for v in group['p'].tolist()],
        )
    return out


def _hold_to_expiry_pnl(amount: float, buy_price: float, won: bool) -> float:
    tokens_after_fee = (amount / buy_price) * (1.0 - polymarket_taker_fee_rate(buy_price))
    if won:
        return tokens_after_fee - amount
    return -amount


def _exit_sell_pnl(amount: float, buy_price: float, sell_price: float) -> float:
    tokens_after_fee = (amount / buy_price) * (1.0 - polymarket_taker_fee_rate(buy_price))
    cash_out = tokens_after_fee * sell_price * (1.0 - polymarket_taker_fee_rate(sell_price))
    return cash_out - amount


def _spread_metric_from_row(row: pd.Series) -> float:
    for col in ('ob_spread_ratio_last', 'ob_spread_ratio_mean', 'roll_spread_pct_20', 'roll_spread_20'):
        val = row.get(col)
        if val is None:
            continue
        try:
            fv = float(val)
        except Exception:
            continue
        if math.isfinite(fv):
            return abs(fv)
    return 0.0


def _vol_metric_from_row(row: pd.Series) -> float:
    for col in ('atr_pct_14', 'atr_pct_20', 'volatility_10', 'volatility_20', 'rv_1h', 'rv_4h'):
        val = row.get(col)
        if val is None:
            continue
        try:
            fv = float(val)
        except Exception:
            continue
        if math.isfinite(fv):
            return abs(fv)
    return 0.0


def _trend_score_from_row(row: pd.Series) -> float:
    vals: list[float] = []
    for col in ('ret_1h', 'ret_4h', 'ret_1d', 'ema_10_20_diff', 'ema_20_50_diff'):
        val = row.get(col)
        if val is None:
            continue
        try:
            fv = float(val)
        except Exception:
            continue
        if math.isfinite(fv):
            vals.append(float(np.sign(fv)))
    return float(np.sum(vals)) if vals else 0.0


def _direction_trend_bucket(direction: str, trend_score: float) -> str:
    signed = trend_score if direction == 'UP' else -trend_score
    if signed >= 1.5:
        return 'aligned'
    if signed <= -1.0:
        return 'counter'
    return 'mixed'


def _confidence_bucket(confidence: float) -> str:
    if confidence >= 0.62:
        return 'high'
    if confidence >= 0.56:
        return 'mid'
    return 'low'


def _bucketize_metric(value: float, q1: float, q2: float, labels: Sequence[str] = ('low', 'mid', 'high')) -> str:
    if value <= q1:
        return labels[0]
    if value <= q2:
        return labels[1]
    return labels[2]


def _spread_bucket_to_proxy(asset: str, bucket: str) -> float:
    if asset == 'BTC_USDT':
        mapping = {'tight': 0.01, 'normal': 0.015, 'wide': 0.02}
    else:
        mapping = {'tight': 0.01, 'normal': 0.02, 'wide': 0.03}
    return mapping.get(bucket, mapping['normal'])


def _quote_exit_price(mid: float, spread_proxy: float, policy: str) -> float:
    if policy == 'HIT_BID':
        out = mid - spread_proxy
    elif policy == 'MID_MINUS_0.01':
        out = mid - 0.01
    elif policy == 'MID_MINUS_0.02':
        out = mid - 0.02
    elif policy == 'MID_MINUS_SPREAD_HALF':
        out = mid - max(spread_proxy / 2.0, 0.005)
    else:
        out = mid - max(spread_proxy, 0.01)
    return float(np.clip(out, 0.01, 0.99))


def _simulate_family_policy(
    amount: float,
    entry_price: float,
    entry_ts: int,
    market_start_ts: int,
    won: bool,
    t_sec_list: Sequence[int],
    direction_prices: Sequence[float],
    exit_family: str,
    sell_quote_policy: str,
    spread_proxy: float,
) -> tuple[float, float, int, str, float, float]:
    seq = [(int(t), float(p)) for t, p in zip(t_sec_list, direction_prices) if int(t) >= entry_ts]
    if not seq:
        settle_price = 1.0 if won else 0.0
        pnl = _hold_to_expiry_pnl(amount, entry_price, won)
        return pnl, settle_price, market_start_ts + 900, 'fallback_hold_no_path', 0.0, 0.0

    mae = 0.0
    mfe = 0.0
    peak = entry_price
    for _t, p in seq:
        mae = max(mae, max(0.0, entry_price - p))
        mfe = max(mfe, max(0.0, p - entry_price))
        peak = max(peak, p)

    def _exit(mid: float, ts: int, reason: str) -> tuple[float, float, int, str, float, float]:
        exit_price = _quote_exit_price(mid, spread_proxy, sell_quote_policy)
        pnl = _exit_sell_pnl(amount, entry_price, exit_price)
        return pnl, exit_price, ts, reason, mae, mfe

    if exit_family == 'HOLD_TO_EXPIRY':
        settle_price = 1.0 if won else 0.0
        pnl = _hold_to_expiry_pnl(amount, entry_price, won)
        return pnl, settle_price, market_start_ts + 900, 'hold_to_expiry', mae, mfe

    if exit_family == 'TIME_EXIT':
        target_ts = market_start_ts + 840
        chosen = next(((t, p) for t, p in seq if t >= target_ts), seq[-1])
        return _exit(chosen[1], int(chosen[0]), 'time_exit_t60')

    if exit_family == 'STOP_LOSS':
        threshold = entry_price - 0.05
        hit = next(((t, p) for t, p in seq if p <= threshold), None)
        if hit is not None:
            return _exit(hit[1], int(hit[0]), 'stop_loss_0.05')
        settle_price = 1.0 if won else 0.0
        pnl = _hold_to_expiry_pnl(amount, entry_price, won)
        return pnl, settle_price, market_start_ts + 900, 'stop_loss_not_hit_hold', mae, mfe

    if exit_family == 'TAKE_PROFIT':
        threshold = entry_price + 0.06
        hit = next(((t, p) for t, p in seq if p >= threshold), None)
        if hit is not None:
            return _exit(hit[1], int(hit[0]), 'take_profit_0.06')
        settle_price = 1.0 if won else 0.0
        pnl = _hold_to_expiry_pnl(amount, entry_price, won)
        return pnl, settle_price, market_start_ts + 900, 'take_profit_not_hit_hold', mae, mfe

    if exit_family == 'TRAILING_EXIT':
        armed = False
        peak = -1.0
        for t, p in seq:
            if not armed and p >= entry_price + 0.04:
                armed = True
                peak = p
            elif armed:
                peak = max(peak, p)
                if p <= peak - 0.02:
                    return _exit(p, int(t), 'trailing_exit')
        settle_price = 1.0 if won else 0.0
        pnl = _hold_to_expiry_pnl(amount, entry_price, won)
        return pnl, settle_price, market_start_ts + 900, 'trailing_not_hit_hold', mae, mfe

    if exit_family == 'PROFIT_PROTECT':
        armed = False
        peak = -1.0
        for t, p in seq:
            if not armed and p >= entry_price + 0.03:
                armed = True
                peak = p
            elif armed:
                peak = max(peak, p)
                protect_floor = max(entry_price + 0.01, peak - 0.015)
                if p <= protect_floor:
                    return _exit(p, int(t), 'profit_protect')
        settle_price = 1.0 if won else 0.0
        pnl = _hold_to_expiry_pnl(amount, entry_price, won)
        return pnl, settle_price, market_start_ts + 900, 'profit_protect_not_hit_hold', mae, mfe

    if exit_family == 'FAST_FAIL_EXIT':
        fast_end = entry_ts + 180
        for t, p in seq:
            if t <= fast_end and p <= entry_price - 0.03:
                return _exit(p, int(t), 'fast_fail_early')
            if t > fast_end:
                break
        guard_end = min(market_start_ts + 300, market_start_ts + 900)
        for t, p in seq:
            if t <= guard_end and p <= entry_price - 0.02:
                return _exit(p, int(t), 'fast_fail_guard')
        settle_price = 1.0 if won else 0.0
        pnl = _hold_to_expiry_pnl(amount, entry_price, won)
        return pnl, settle_price, market_start_ts + 900, 'fast_fail_not_hit_hold', mae, mfe

    settle_price = 1.0 if won else 0.0
    pnl = _hold_to_expiry_pnl(amount, entry_price, won)
    return pnl, settle_price, market_start_ts + 900, 'fallback_hold', mae, mfe


def _utility_adjusted(
    pnl: float,
    trend_bucket: str,
    exit_family: str,
    sell_quote_policy: str,
    mae: float,
    objective_cfg: dict[str, float],
) -> float:
    utility = float(pnl)
    if trend_bucket == 'aligned':
        utility *= float(objective_cfg['trend_bonus'])
    elif trend_bucket == 'counter':
        utility *= float(objective_cfg['countertrend_penalty'])
    utility -= float(objective_cfg['drawdown_penalty']) * float(mae)
    if exit_family in {'TAKE_PROFIT', 'TRAILING_EXIT', 'PROFIT_PROTECT'} and pnl > 0:
        utility += float(objective_cfg['take_profit_preference']) * min(float(pnl), 0.10)
    if sell_quote_policy in {'HIT_BID', 'MID_MINUS_SPREAD_ONE'}:
        utility += float(objective_cfg['sell_aggressiveness_bias']) * 0.001
    return utility


def _default_objective_cfg() -> dict[str, float]:
    return {
        'trend_bonus': 1.20,
        'countertrend_penalty': 0.75,
        'abstain_cost_bias': 0.02,
        'drawdown_penalty': 0.25,
        'sell_aggressiveness_bias': 0.10,
        'take_profit_preference': 0.05,
    }


def _best_labels_for_row(asset: str, row: pd.Series, t_sec_list: Sequence[int], up_path: Sequence[float], objective_cfg: dict[str, float]) -> dict[str, Any] | None:
    market_start_ts = int(row['target_start_ts'])
    target_period_end_ts = market_end_from_start_ts(market_start_ts)
    prediction_ts = int(pd.Timestamp(row['timestamp']).timestamp())
    market_slug = f"{'btc' if asset == 'BTC_USDT' else 'eth'}-updown-15m-{market_start_ts}"
    actual_up = bool(float(row['next_return']) > 0.0)
    vol_metric = _vol_metric_from_row(row)
    spread_metric = _spread_metric_from_row(row)
    trend_score = _trend_score_from_row(row)
    spread_proxy = _spread_bucket_to_proxy(asset, 'normal')
    action_utility_map: dict[str, float] = {}
    action_meta_map: dict[str, dict[str, Any]] = {}
    best: dict[str, Any] | None = None
    for direction in ('UP', 'DOWN'):
        direction_prices = up_path if direction == 'UP' else [float(np.clip(1.0 - p, 0.0, 1.0)) for p in up_path]
        won = actual_up if direction == 'UP' else (not actual_up)
        trend_bucket = _direction_trend_bucket(direction, trend_score)
        for buy_price in BUY_GRIDS[asset]:
            action_key = f'{direction}|{buy_price:.2f}'
            entry_trace = replay_entry_trace(
                prediction_ts=prediction_ts,
                market_start_ts=market_start_ts,
                limit_price=float(buy_price),
                t_sec_list=t_sec_list,
                direction_prices=direction_prices,
                market_slug=market_slug,
            )
            if entry_trace.fill_status != 'filled' or entry_trace.fill_price is None or entry_trace.fill_ts is None:
                candidate = {
                    'action_key': action_key,
                    'direction': direction,
                    'buy_label': f'{buy_price:.2f}',
                    'best_exit_family': 'HOLD_TO_EXPIRY',
                    'best_sell_quote_policy': 'MID_MINUS_0.01',
                    'best_utility': 0.0,
                    'baseline_hold_pnl': 0.0,
                    'entry_fill_proxy': float(entry_trace.first_observed_price) if entry_trace.first_observed_price is not None else np.nan,
                    'entry_fill_status': entry_trace.fill_status,
                    'entry_fill_ts': entry_trace.fill_ts,
                    'entry_fill_fraction': float(entry_trace.fill_fraction),
                    'entry_proxy_ts': int(entry_trace.fill_ts or entry_trace.market_discovered_ts or prediction_ts),
                    'market_discovered_ts': entry_trace.market_discovered_ts,
                    'order_submitted_ts': entry_trace.order_submitted_ts,
                    'timeout_ts': int(entry_trace.timeout_ts),
                    'target_period_end_ts': int(entry_trace.target_period_end_ts),
                    'exit_reason': 'entry_unfilled',
                    'vol_metric': vol_metric,
                    'spread_metric': spread_metric,
                    'trend_score': trend_score,
                }
                action_utility_map[action_key] = 0.0
                action_meta_map[action_key] = {
                    'direction': direction,
                    'buy_label': f'{buy_price:.2f}',
                    'best_exit_family': 'HOLD_TO_EXPIRY',
                    'best_sell_quote_policy': 'MID_MINUS_0.01',
                    'baseline_hold_pnl': 0.0,
                    'entry_trace': entry_trace.to_dict(),
                    'exit_reason': 'entry_unfilled',
                }
                if best is None or float(candidate['best_utility']) > float(best['best_utility']):
                    best = candidate
                continue
            baseline_hold = _hold_to_expiry_pnl(1.0, float(entry_trace.fill_price), won)
            baseline_utility = _utility_adjusted(baseline_hold, trend_bucket, 'HOLD_TO_EXPIRY', 'MID_MINUS_0.01', 0.0, objective_cfg)
            local_best = {
                'action_key': action_key,
                'direction': direction,
                'buy_label': f'{buy_price:.2f}',
                'best_exit_family': 'HOLD_TO_EXPIRY',
                'best_sell_quote_policy': 'MID_MINUS_0.01',
                'best_utility': baseline_utility,
                'baseline_hold_pnl': baseline_hold,
                'entry_fill_proxy': float(entry_trace.fill_price),
                'entry_fill_status': entry_trace.fill_status,
                'entry_fill_ts': int(entry_trace.fill_ts),
                'entry_fill_fraction': float(entry_trace.fill_fraction),
                'entry_proxy_ts': int(entry_trace.fill_ts),
                'market_discovered_ts': entry_trace.market_discovered_ts,
                'order_submitted_ts': entry_trace.order_submitted_ts,
                'timeout_ts': int(entry_trace.timeout_ts),
                'target_period_end_ts': int(entry_trace.target_period_end_ts),
                'exit_reason': 'hold_to_expiry',
                'vol_metric': vol_metric,
                'spread_metric': spread_metric,
                'trend_score': trend_score,
            }
            for family in EXIT_FAMILIES:
                for policy in SELL_QUOTE_POLICIES:
                    pnl, exit_price, exit_ts, exit_reason, mae, _mfe = _simulate_family_policy(
                        amount=1.0,
                        entry_price=float(entry_trace.fill_price),
                        entry_ts=int(entry_trace.fill_ts),
                        market_start_ts=market_start_ts,
                        won=won,
                        t_sec_list=t_sec_list,
                        direction_prices=direction_prices,
                        exit_family=family,
                        sell_quote_policy=policy,
                        spread_proxy=spread_proxy,
                    )
                    utility = _utility_adjusted(pnl, trend_bucket, family, policy, mae, objective_cfg)
                    if utility > float(local_best['best_utility']):
                        local_best = {
                            'direction': direction,
                            'buy_label': f'{buy_price:.2f}',
                            'best_exit_family': family,
                            'best_sell_quote_policy': policy,
                            'best_utility': utility,
                            'baseline_hold_pnl': baseline_hold,
                            'entry_fill_proxy': float(entry_trace.fill_price),
                            'entry_fill_status': entry_trace.fill_status,
                            'entry_fill_ts': int(entry_trace.fill_ts),
                            'entry_fill_fraction': float(entry_trace.fill_fraction),
                            'entry_proxy_ts': int(entry_trace.fill_ts),
                            'market_discovered_ts': entry_trace.market_discovered_ts,
                            'order_submitted_ts': entry_trace.order_submitted_ts,
                            'timeout_ts': int(entry_trace.timeout_ts),
                            'target_period_end_ts': int(entry_trace.target_period_end_ts),
                            'exit_reason': exit_reason,
                            'exit_price': exit_price,
                            'exit_ts': exit_ts,
                            'vol_metric': vol_metric,
                            'spread_metric': spread_metric,
                            'trend_score': trend_score,
                        }
            action_utility_map[action_key] = float(local_best['best_utility'])
            action_meta_map[action_key] = {
                'direction': direction,
                'buy_label': f'{buy_price:.2f}',
                'best_exit_family': str(local_best['best_exit_family']),
                'best_sell_quote_policy': str(local_best['best_sell_quote_policy']),
                'baseline_hold_pnl': _round_float(local_best['baseline_hold_pnl'], 6),
                'entry_trace': entry_trace.to_dict(),
                'exit_reason': str(local_best['exit_reason']),
                'exit_price': _round_float(local_best.get('exit_price')),
                'exit_ts': int(local_best.get('exit_ts') or target_period_end_ts),
            }
            if best is None or float(local_best['best_utility']) > float(best['best_utility']):
                best = local_best
    if best is None:
        return None
    take_label = int(float(best['best_utility']) > float(objective_cfg['abstain_cost_bias']))
    direction_target = 1 if str(best['direction']) == 'UP' else 0
    return {
        'actual_up': int(actual_up),
        'gate_label': take_label,
        'direction_target': direction_target,
        'buy_label': best['buy_label'] if take_label else 'ABSTAIN',
        'best_exit_family': best['best_exit_family'] if take_label else 'HOLD_TO_EXPIRY',
        'best_sell_quote_policy': best['best_sell_quote_policy'] if take_label else 'MID_MINUS_0.01',
        'best_utility': float(best['best_utility']),
        'baseline_hold_pnl': float(best.get('baseline_hold_pnl') or 0.0),
        'entry_fill_proxy': float(best.get('entry_fill_proxy') or np.nan),
        'entry_fill_status': str(best.get('entry_fill_status') or 'unknown'),
        'entry_fill_ts': int(best.get('entry_fill_ts') or 0),
        'entry_fill_fraction': float(best.get('entry_fill_fraction') or 0.0),
        'entry_proxy_ts': int(best.get('entry_proxy_ts') or prediction_ts),
        'market_discovered_ts': int(best.get('market_discovered_ts') or 0),
        'order_submitted_ts': int(best.get('order_submitted_ts') or 0),
        'timeout_ts': int(best.get('timeout_ts') or target_period_end_ts),
        'target_period_end_ts': int(best.get('target_period_end_ts') or target_period_end_ts),
        'best_action_key': str(best.get('action_key') or 'ABSTAIN'),
        'entry_action_utility_json': json.dumps(
            {k: _round_float(v, 6) for k, v in action_utility_map.items()},
            ensure_ascii=False,
            separators=(',', ':'),
        ),
        'entry_action_meta_json': json.dumps(action_meta_map, ensure_ascii=False, separators=(',', ':')),
        'entry_trace_json': json.dumps(
            action_meta_map.get(str(best.get('action_key') or ''), {}).get('entry_trace') or {},
            ensure_ascii=False,
            separators=(',', ':'),
        ),
        'exit_reason': best.get('exit_reason') or 'hold_to_expiry',
        'vol_metric': float(best.get('vol_metric') or 0.0),
        'spread_metric': float(best.get('spread_metric') or 0.0),
        'trend_score': float(best.get('trend_score') or 0.0),
    }


@dataclass
class SplitData:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def _iter_purged_splits(df: pd.DataFrame, train_days: int, max_folds: int = 3) -> list[SplitData]:
    work = df.sort_values('timestamp').reset_index(drop=True)
    latest = pd.to_datetime(work['timestamp'], utc=True).max()
    embargo = pd.Timedelta(minutes=15 * EMBARGO_BARS)
    folds: list[SplitData] = []
    for fold_idx in range(max_folds):
        test_end = latest - pd.Timedelta(days=TEST_DAYS * fold_idx)
        test_start = test_end - pd.Timedelta(days=TEST_DAYS)
        val_end = test_start - embargo
        val_start = val_end - pd.Timedelta(days=VAL_DAYS)
        train_end = val_start - embargo
        train_start = train_end - pd.Timedelta(days=train_days)
        train = work[(work['timestamp'] >= train_start) & (work['timestamp'] < train_end)].copy()
        val = work[(work['timestamp'] >= val_start) & (work['timestamp'] < val_end)].copy()
        test = work[(work['timestamp'] >= test_start) & (work['timestamp'] < test_end)].copy()
        if len(train) < MIN_TRAIN_ROWS or len(val) < MIN_VAL_ROWS or len(test) < MIN_TEST_ROWS:
            continue
        folds.append(SplitData(train=train, val=val, test=test))
    return folds


def _lgb_binary(params: dict[str, Any], n_jobs: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(objective='binary', metric='auc', n_estimators=1600, verbosity=-1, n_jobs=n_jobs, **params)


def _lgb_multiclass(params: dict[str, Any], num_class: int, n_jobs: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(objective='multiclass', num_class=num_class, metric='multi_logloss', n_estimators=1600, verbosity=-1, n_jobs=n_jobs, **params)


def _fit_heads(split: SplitData, feature_cols: Sequence[str], params: dict[str, Any], buy_actions: Sequence[str], lgb_jobs: int) -> dict[str, Any]:
    X_tr = split.train[list(feature_cols)]
    X_va = split.val[list(feature_cols)]
    X_te = split.test[list(feature_cols)]
    w_tr = split.train['sample_weight'].to_numpy(dtype=float)

    gate_model = _lgb_binary(params, n_jobs=lgb_jobs)
    gate_model.fit(X_tr, split.train['gate_label'].to_numpy(dtype=int), sample_weight=w_tr, eval_set=[(X_va, split.val['gate_label'].to_numpy(dtype=int))], callbacks=[lgb.early_stopping(100, verbose=False)])

    dir_model = _lgb_binary(params, n_jobs=lgb_jobs)
    dir_model.fit(X_tr, split.train['direction_target'].to_numpy(dtype=int), sample_weight=w_tr, eval_set=[(X_va, split.val['direction_target'].to_numpy(dtype=int))], callbacks=[lgb.early_stopping(100, verbose=False)])

    price_to_id = {name: idx for idx, name in enumerate(buy_actions)}
    tr_take = split.train[split.train['gate_label'] == 1].copy()
    va_take = split.val[split.val['gate_label'] == 1].copy()
    if len(tr_take) < 120 or len(va_take) < 40:
        raise RuntimeError('insufficient positive take rows for buy head')
    buy_model = _lgb_multiclass(params, num_class=len(buy_actions), n_jobs=lgb_jobs)
    buy_model.fit(
        tr_take[list(feature_cols)],
        tr_take['buy_label'].map(price_to_id).to_numpy(dtype=int),
        sample_weight=tr_take['sample_weight'].to_numpy(dtype=float),
        eval_set=[(va_take[list(feature_cols)], va_take['buy_label'].map(price_to_id).to_numpy(dtype=int))],
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )

    return {
        'gate_model': gate_model,
        'dir_model': dir_model,
        'buy_model': buy_model,
        'X_va': X_va,
        'X_te': X_te,
        'price_to_id': price_to_id,
        'id_to_price': {idx: name for name, idx in price_to_id.items()},
        'split': split,
    }


def _apply_calibration_meta(meta: dict[str, Any], probs: np.ndarray) -> np.ndarray:
    base = np.clip(np.asarray(probs, dtype=float), 1e-6, 1.0 - 1e-6)
    kind = str((meta or {}).get('selected') or (meta or {}).get('kind') or 'none')
    if kind == 'temperature':
        temp = float(meta.get('temperature') or 1.0)
        logit = np.log(base / (1.0 - base))
        return 1.0 / (1.0 + np.exp(-(logit / temp)))
    if kind == 'isotonic':
        x = np.asarray(meta.get('x_thresholds') or [], dtype=float)
        y = np.asarray(meta.get('y_thresholds') or [], dtype=float)
        if len(x) >= 2 and len(x) == len(y):
            return np.clip(np.interp(base, x, y, left=y[0], right=y[-1]), 1e-6, 1.0 - 1e-6)
    return base


def _bucket_edges(df: pd.DataFrame) -> dict[str, list[float]]:
    vol = pd.to_numeric(df['vol_metric'], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    spread = pd.to_numeric(df['spread_metric'], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
    vol_q = [float(vol.quantile(0.33)) if len(vol) else 0.0, float(vol.quantile(0.66)) if len(vol) else 0.0]
    spread_q = [float(spread.quantile(0.33)) if len(spread) else 0.0, float(spread.quantile(0.66)) if len(spread) else 0.0]
    return {'confidence': [0.56, 0.62], 'vol': vol_q, 'spread': spread_q}


def _build_bucket_key(asset: str, confidence: float, trend_score: float, direction: str, vol_metric: float, spread_metric: float, edges: dict[str, list[float]]) -> tuple[str, str, str, str]:
    conf_bucket = _confidence_bucket(confidence)
    trend_bucket = _direction_trend_bucket(direction, trend_score)
    vol_bucket = _bucketize_metric(float(vol_metric), float(edges['vol'][0]), float(edges['vol'][1]))
    spread_bucket = _bucketize_metric(float(spread_metric), float(edges['spread'][0]), float(edges['spread'][1]), labels=('tight', 'normal', 'wide'))
    return conf_bucket, trend_bucket, vol_bucket, spread_bucket


def _select_calibration(dir_val_prob: np.ndarray, y_val: np.ndarray, gate_val_prob: np.ndarray, buy_val_labels: np.ndarray, rows: pd.DataFrame, take_threshold: float) -> dict[str, Any]:
    base_prob = np.clip(np.asarray(dir_val_prob, dtype=float), 1e-6, 1.0 - 1e-6)
    y_val = np.asarray(y_val, dtype=int)
    candidates: list[tuple[dict[str, Any], np.ndarray]] = [({'selected': 'none', 'kind': 'none'}, base_prob)]
    for temp in [0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.25]:
        logit = np.log(base_prob / (1.0 - base_prob))
        cal = 1.0 / (1.0 + np.exp(-(logit / temp)))
        candidates.append(({'selected': 'temperature', 'kind': 'temperature', 'temperature': temp}, cal))
    try:
        ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
        ir.fit(base_prob, y_val)
        cal = np.clip(ir.predict(base_prob), 1e-6, 1.0 - 1e-6)
        x = getattr(ir, 'X_thresholds_', None)
        y = getattr(ir, 'y_thresholds_', None)
        if x is not None and y is not None and len(x) >= 2:
            candidates.append(({
                'selected': 'isotonic', 'kind': 'isotonic',
                'x_thresholds': [float(v) for v in x],
                'y_thresholds': [float(v) for v in y],
            }, cal))
    except Exception:
        pass

    best_meta = {'selected': 'none', 'kind': 'none'}
    best_score = -1e18
    for meta, probs in candidates:
        metrics = _simulate_eval(rows, probs, gate_val_prob, buy_val_labels, take_threshold)
        if int(metrics['trades']) < max(1, int(round(len(rows) * 0.05))):
            continue
        brier = float(brier_score_loss(y_val, probs)) if len(y_val) else 1.0
        score = _candidate_score(metrics) - brier
        if score > best_score:
            best_score = score
            best_meta = dict(meta)
            best_meta['brier'] = _round_float(brier, 6)
            best_meta['val_metrics'] = metrics
    return best_meta


def _build_exit_policy_lookup(rows: pd.DataFrame, dir_prob: np.ndarray, buy_labels: np.ndarray, edges: dict[str, list[float]], asset: str) -> dict[str, Any]:
    lookup_rows: dict[str, dict[str, Any]] = {}
    fallback_scores: dict[tuple[str, str], list[float]] = {}
    for i, row in rows.reset_index(drop=True).iterrows():
        if int(row['gate_label']) != 1:
            continue
        buy_label = str(buy_labels[i])
        if buy_label == 'ABSTAIN':
            continue
        confidence = max(float(dir_prob[i]), 1.0 - float(dir_prob[i]))
        direction = 'UP' if float(dir_prob[i]) >= 0.5 else 'DOWN'
        bucket = _build_bucket_key(asset, confidence, float(row['trend_score']), direction, float(row['vol_metric']), float(row['spread_metric']), edges)
        key = '|'.join(bucket)
        pair = (str(row['best_exit_family']), str(row['best_sell_quote_policy']))
        slot = lookup_rows.setdefault(key, {'count': 0, 'pair_scores': {}, 'bucket': {'confidence_bucket': bucket[0], 'trend_bucket': bucket[1], 'vol_bucket': bucket[2], 'spread_bucket': bucket[3]}})
        slot['count'] += 1
        pair_key = '|'.join(pair)
        pair_slot = slot['pair_scores'].setdefault(pair_key, {'count': 0, 'utility_sum': 0.0, 'exit_family': pair[0], 'sell_quote_policy': pair[1]})
        pair_slot['count'] += 1
        pair_slot['utility_sum'] += float(row['best_utility'])
        fallback_scores.setdefault(pair, []).append(float(row['best_utility']))
    rendered: dict[str, Any] = {}
    for key, slot in lookup_rows.items():
        if int(slot['count']) < 40:
            continue
        best_pair = max(slot['pair_scores'].values(), key=lambda x: (x['utility_sum'] / max(1, x['count']), x['count']))
        rendered[key] = {
            **slot['bucket'],
            'samples': int(slot['count']),
            'selected_exit_family': best_pair['exit_family'],
            'selected_sell_quote_policy': best_pair['sell_quote_policy'],
            'avg_utility': _round_float(best_pair['utility_sum'] / max(1, best_pair['count']), 6),
        }
    if fallback_scores:
        fallback_pair = max(fallback_scores.items(), key=lambda kv: np.mean(kv[1]))[0]
    else:
        fallback_pair = ('HOLD_TO_EXPIRY', 'MID_MINUS_0.01' if asset == 'BTC_USDT' else 'MID_MINUS_0.02')
    return {
        'generated_at': utc_now_iso(),
        'asset': asset,
        'bucket_edges': edges,
        'default': {
            'selected_exit_family': fallback_pair[0],
            'selected_sell_quote_policy': fallback_pair[1],
        },
        'rows': rendered,
    }


def _simulate_eval(rows: pd.DataFrame, dir_prob: np.ndarray, gate_prob: np.ndarray, buy_labels: np.ndarray, take_threshold: float) -> dict[str, Any]:
    rows = rows.sort_values('target_start_ts').reset_index(drop=True)
    conf = np.maximum(dir_prob, 1.0 - dir_prob)
    total_pnl = 0.0
    trades = 0
    wins = 0
    losses = 0
    abstained = 0
    pnls: list[float] = []
    hold_secs: list[int] = []
    buy_prices: list[float] = []
    chosen_dirs: list[str] = []
    high_conf_losses = 0
    exit_reason_counts: dict[str, int] = {}
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for i, row in rows.iterrows():
        if float(gate_prob[i]) < take_threshold:
            abstained += 1
            continue
        buy_label = str(buy_labels[i])
        if buy_label == 'ABSTAIN':
            abstained += 1
            continue
        direction = 'UP' if float(dir_prob[i]) >= 0.5 else 'DOWN'
        action_key = f'{direction}|{buy_label}'
        action_meta_map = json.loads(str(row['entry_action_meta_json'])) if str(row.get('entry_action_meta_json') or '').strip() else {}
        action_meta = action_meta_map.get(action_key) if isinstance(action_meta_map, dict) else None
        if not isinstance(action_meta, dict):
            abstained += 1
            continue
        entry_trace = action_meta.get('entry_trace') if isinstance(action_meta.get('entry_trace'), dict) else {}
        if str((entry_trace or {}).get('fill_status') or '') != 'filled':
            abstained += 1
            continue
        t_sec_list = json.loads(str(row['path_t_sec_json']))
        up_path = json.loads(str(row['path_up_json']))
        actual_up = bool(int(row['actual_up']) == 1)
        won = actual_up if direction == 'UP' else (not actual_up)
        direction_prices = up_path if direction == 'UP' else [float(np.clip(1.0 - p, 0.0, 1.0)) for p in up_path]
        market_start_ts = int(row['target_start_ts'])
        entry_ts = int((entry_trace or {}).get('fill_ts') or row['entry_proxy_ts'])
        entry_price = float((entry_trace or {}).get('fill_price') or row.get('entry_fill_proxy') or buy_label)
        spread_bucket = _bucketize_metric(float(row['spread_metric']), 0.000001, 0.00001, labels=('tight', 'normal', 'wide'))
        spread_proxy = _spread_bucket_to_proxy('BTC_USDT' if 'BTC' in str(row.get('symbol', 'BTC')) else 'ETH_USDT', spread_bucket)
        pnl, _exit_price, exit_ts, exit_reason, _mae, _mfe = _simulate_family_policy(
            amount=1.0,
            entry_price=entry_price,
            entry_ts=entry_ts,
            market_start_ts=market_start_ts,
            won=won,
            t_sec_list=t_sec_list,
            direction_prices=direction_prices,
            exit_family=str(action_meta.get('best_exit_family') or row['best_exit_family']),
            sell_quote_policy=str(action_meta.get('best_sell_quote_policy') or row['best_sell_quote_policy']),
            spread_proxy=spread_proxy,
        )
        trades += 1
        total_pnl += pnl
        pnls.append(float(pnl))
        buy_prices.append(float(entry_price))
        chosen_dirs.append(direction)
        hold_secs.append(int(exit_ts - entry_ts))
        exit_reason_counts[exit_reason] = exit_reason_counts.get(exit_reason, 0) + 1
        if pnl > 0:
            wins += 1
        else:
            losses += 1
            if conf[i] >= 0.60:
                high_conf_losses += 1
        cum += pnl
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    streak = 0
    max_streak = 0
    prev_dir = None
    for pnl, direction in zip(pnls, chosen_dirs):
        if pnl <= 0 and direction == prev_dir:
            streak += 1
        elif pnl <= 0:
            streak = 1
        else:
            streak = 0
        prev_dir = direction if pnl <= 0 else None
        max_streak = max(max_streak, streak)
    return {
        'rows': int(len(rows)),
        'trades': int(trades),
        'win_rate': _round_float(float(wins / trades) if trades else 0.0),
        'total_pnl': _round_float(total_pnl, 6),
        'avg_expectancy': _round_float(float(np.mean(pnls)) if pnls else 0.0, 6),
        'max_drawdown': _round_float(max_dd, 6),
        'high_conf_loss_rate': _round_float(float(high_conf_losses / trades) if trades else 0.0, 6),
        'max_same_direction_loss_streak': int(max_streak),
        'abstain_rate': _round_float(float(abstained / len(rows)) if len(rows) else 0.0, 6),
        'avg_buy_price': _round_float(float(np.mean(buy_prices)) if buy_prices else 0.0, 4),
        'avg_exit_hold_seconds': _round_float(float(np.mean(hold_secs)) if hold_secs else 0.0, 2),
        'exit_reason_breakdown': {k: int(v) for k, v in sorted(exit_reason_counts.items(), key=lambda kv: (-kv[1], kv[0]))},
    }


def _candidate_score(metrics: dict[str, Any]) -> float:
    trades = int(metrics.get('trades') or 0)
    total_pnl = float(metrics.get('total_pnl') or 0.0)
    abstain_rate = float(metrics.get('abstain_rate') or 1.0)
    if total_pnl <= 0 or trades < 30 or abstain_rate >= 0.70:
        return -1e9
    return (
        1.00 * total_pnl
        - 0.35 * float(metrics.get('max_drawdown') or 0.0)
        + 0.25 * float(metrics.get('avg_expectancy') or 0.0)
        - 0.20 * float(metrics.get('high_conf_loss_rate') or 0.0)
        + 0.10 * float(metrics.get('win_rate') or 0.0)
    )


def _is_viable_metrics(metrics: dict[str, Any]) -> bool:
    return _candidate_score(metrics) > -1e8


def _select_calibration_from_split(fit: dict[str, Any], take_threshold: float) -> dict[str, Any]:
    split = fit['split']
    dir_val = fit['dir_model'].predict_proba(fit['X_va'])[:, 1]
    gate_val = fit['gate_model'].predict_proba(fit['X_va'])[:, 1]
    buy_val = fit['buy_model'].predict(fit['X_va'])
    buy_val_labels = np.asarray([fit['id_to_price'][int(i)] for i in buy_val], dtype=object)
    return _select_calibration(np.asarray(dir_val, dtype=float), split.val['direction_target'].to_numpy(dtype=int), np.asarray(gate_val, dtype=float), buy_val_labels, split.val, take_threshold)


def _evaluate_candidate(df: pd.DataFrame, feature_cols: Sequence[str], params: dict[str, Any], buy_actions: Sequence[str], train_days: int, take_threshold: float, calibration_only: bool = False) -> dict[str, Any]:
    folds = _iter_purged_splits(df, train_days=train_days)
    if not folds:
        raise RuntimeError('no valid purged folds')
    metrics_list: list[dict[str, Any]] = []
    best_calibration: dict[str, Any] = {'selected': 'none', 'kind': 'none'}
    best_fit: dict[str, Any] | None = None
    for idx, split in enumerate(folds):
        fit = _fit_heads(split, feature_cols, params, buy_actions, lgb_jobs=1)
        cal_meta = _select_calibration_from_split(fit, take_threshold)
        if idx == 0:
            best_calibration = cal_meta
            best_fit = fit
        dir_test = fit['dir_model'].predict_proba(fit['X_te'])[:, 1]
        gate_test = fit['gate_model'].predict_proba(fit['X_te'])[:, 1]
        buy_test = fit['buy_model'].predict(fit['X_te'])
        buy_test_labels = np.asarray([fit['id_to_price'][int(i)] for i in buy_test], dtype=object)
        metrics = _simulate_eval(split.test, _apply_calibration_meta(cal_meta, np.asarray(dir_test, dtype=float)), np.asarray(gate_test, dtype=float), buy_test_labels, take_threshold)
        metrics_list.append(metrics)
    agg = {
        'trades': int(round(float(np.mean([m['trades'] for m in metrics_list])))) if metrics_list else 0,
        'total_pnl': _round_float(float(np.mean([m['total_pnl'] for m in metrics_list])) if metrics_list else 0.0, 6),
        'max_drawdown': _round_float(float(np.mean([m['max_drawdown'] for m in metrics_list])) if metrics_list else 0.0, 6),
        'avg_expectancy': _round_float(float(np.mean([m['avg_expectancy'] for m in metrics_list])) if metrics_list else 0.0, 6),
        'high_conf_loss_rate': _round_float(float(np.mean([m['high_conf_loss_rate'] for m in metrics_list])) if metrics_list else 0.0, 6),
        'win_rate': _round_float(float(np.mean([m['win_rate'] for m in metrics_list])) if metrics_list else 0.0, 6),
        'abstain_rate': _round_float(float(np.mean([m['abstain_rate'] for m in metrics_list])) if metrics_list else 1.0, 6),
        'max_same_direction_loss_streak': int(max([m['max_same_direction_loss_streak'] for m in metrics_list] or [0])),
        'avg_buy_price': _round_float(float(np.mean([m['avg_buy_price'] for m in metrics_list])) if metrics_list else 0.0, 4),
        'avg_exit_hold_seconds': _round_float(float(np.mean([m['avg_exit_hold_seconds'] for m in metrics_list])) if metrics_list else 0.0, 2),
        'folds': metrics_list,
    }
    return {'metrics': agg, 'calibration': best_calibration, 'fit': best_fit}


def _objective_factory(asset: str, df: pd.DataFrame, recipe: dict[str, Any], bundle_name: str, groups: Sequence[str], train_days: int, feature_cols: Sequence[str]):
    buy_actions = [f'{p:.2f}' for p in BUY_GRIDS[asset]]

    def objective(trial: optuna.Trial) -> float:
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.08, log=True),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 150),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.50, 0.98),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.50, 0.98),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.3),
            'lambda_l1': trial.suggest_float('lambda_l1', 0.001, 3.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1.0, 40.0, log=True),
            'max_bin': trial.suggest_int('max_bin', 127, 511),
        }
        take_threshold = trial.suggest_float('take_threshold', 0.46, 0.70)
        result = _evaluate_candidate(df, feature_cols, params, buy_actions, train_days, take_threshold)
        metrics = result['metrics']
        trial.set_user_attr('metrics', metrics)
        trial.set_user_attr('calibration', result['calibration'])
        trial.set_user_attr('feature_cols', list(feature_cols))
        trial.set_user_attr('recipe', recipe)
        trial.set_user_attr('bundle', bundle_name)
        trial.set_user_attr('groups', list(groups))
        trial.set_user_attr('train_days', train_days)
        trial.set_user_attr('take_threshold', take_threshold)
        return _candidate_score(metrics)

    return objective


def _build_episode_frame(asset: str, recipe: dict[str, Any], groups: Sequence[str], objective_cfg: dict[str, float]) -> pd.DataFrame:
    base = _build_asset_frame(asset, recipe, groups)
    base['next_close'] = pd.to_numeric(base['close'], errors='coerce').shift(-1)
    base['next_return'] = (pd.to_numeric(base['next_close'], errors='coerce') / pd.to_numeric(base['close'], errors='coerce') - 1.0)
    base = base.dropna(subset=['next_return']).reset_index(drop=True)
    base['prediction_ts'] = (base['timestamp'].astype('int64') // 10**9).astype(int)
    base['target_start_ts'] = (base['prediction_ts'] + 900).astype(int)
    base['market_start_ts'] = base['target_start_ts']
    base['market_end_ts'] = base['market_start_ts'] + 900
    path_groups = _load_path_groups(asset)
    rows: list[dict[str, Any]] = []
    for _, row in base.iterrows():
        start_ts = int(row['target_start_ts'])
        path = path_groups.get(start_ts)
        if not path:
            continue
        t_sec_list, up_path = path
        labels = _best_labels_for_row(asset, row, t_sec_list, up_path, objective_cfg)
        if not labels:
            continue
        record = row.to_dict()
        record.update(labels)
        record['decision_ts'] = int(start_ts)
        record['market_start_ts'] = int(start_ts)
        record['market_end_ts'] = int(market_end_from_start_ts(start_ts))
        record['path_t_sec_json'] = json.dumps([int(v) for v in t_sec_list], separators=(',', ':'))
        record['path_up_json'] = json.dumps([round(float(v), 6) for v in up_path], separators=(',', ':'))
        rows.append(record)
    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError(f'no episodes built for {asset}')
    out['timestamp'] = pd.to_datetime(out['timestamp'], utc=True)
    common_end = min(base['timestamp'].max(), out['timestamp'].max())
    max_window_cutoff = common_end - pd.Timedelta(days=FEATURE_COMPLETE_DAYS)
    out = out[(out['timestamp'] >= max_window_cutoff) & (out['timestamp'] <= common_end)].reset_index(drop=True)
    if out.empty:
        raise RuntimeError(f'no capped episodes built for {asset}')
    path_span_days = float((out['timestamp'].max() - out['timestamp'].min()).total_seconds() / 86400.0) if len(out) else 0.0
    effective_feature_days = int(min(FEATURE_COMPLETE_DAYS, math.floor(path_span_days)))
    if effective_feature_days < MIN_EFFECTIVE_FEATURE_DAYS:
        raise RuntimeError(
            f'{asset}: insufficient path-supported history for serious training '
            f'(effective_days={effective_feature_days}, min_days={MIN_EFFECTIVE_FEATURE_DAYS})'
        )
    effective_cutoff = common_end - pd.Timedelta(days=effective_feature_days)
    base_window_rows = int(((base['timestamp'] >= effective_cutoff) & (base['timestamp'] <= common_end)).sum())
    out = out[(out['timestamp'] >= effective_cutoff) & (out['timestamp'] <= common_end)].reset_index(drop=True)
    out['sample_weight'] = _recent_weight_multiplier(out['timestamp'])
    coverage_ratio = float(len(out) / base_window_rows) if base_window_rows > 0 else 0.0
    if coverage_ratio < 0.85:
        raise RuntimeError(
            f'{asset}: 1m path coverage too incomplete for serious training '
            f'(effective_days={effective_feature_days}, coverage={coverage_ratio:.3f})'
        )
    out.attrs['effective_feature_days'] = int(effective_feature_days)
    out.attrs['path_span_days'] = _round_float(path_span_days, 3)
    out.attrs['path_coverage_ratio'] = _round_float(coverage_ratio, 6)
    out.attrs['path_common_end'] = common_end.isoformat()
    return out


def _write_episode_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def _run_stage_a(asset: str, objective_cfg: dict[str, float], force_rebuild_episodes: bool = False) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    episode_cache: dict[str, pd.DataFrame] = {}
    for recipe in ASSET_RECIPES[asset]:
        for bundle_name, groups in _asset_bundle_map(asset).items():
            cache_key = f"{recipe['name']}::{bundle_name}"
            try:
                episodes = _build_episode_frame(asset, recipe, groups, objective_cfg)
                episode_cache[cache_key] = episodes
            except Exception as exc:
                rows.append({'asset': asset, 'recipe': recipe['name'], 'bundle': bundle_name, 'groups': list(groups), 'status': f'episode_error:{exc}'})
                continue
            feature_cols = _choose_feature_cols(episodes, groups)
            if not feature_cols:
                rows.append({'asset': asset, 'recipe': recipe['name'], 'bundle': bundle_name, 'groups': list(groups), 'status': 'skipped_no_features'})
                continue
            train_day_candidates = _candidate_train_days(int(episodes.attrs.get('effective_feature_days') or FEATURE_COMPLETE_DAYS))
            if not train_day_candidates:
                rows.append({
                    'asset': asset,
                    'recipe': recipe['name'],
                    'bundle': bundle_name,
                    'groups': list(groups),
                    'status': f"skipped_no_train_days:effective_days={episodes.attrs.get('effective_feature_days')}",
                })
                continue
            for train_days in train_day_candidates:
                folds = _iter_purged_splits(episodes, train_days=train_days, max_folds=1)
                if not folds:
                    rows.append({'asset': asset, 'recipe': recipe['name'], 'bundle': bundle_name, 'train_days': train_days, 'status': 'skipped_no_recent_fold'})
                    continue
                split = folds[0]
                for template_name, params in PARAM_TEMPLATES.items():
                    try:
                        fit = _fit_heads(split, feature_cols, params, [f'{p:.2f}' for p in BUY_GRIDS[asset]], lgb_jobs=8)
                        cal_meta = _select_calibration_from_split(fit, take_threshold=0.50)
                        dir_test = fit['dir_model'].predict_proba(fit['X_te'])[:, 1]
                        gate_test = fit['gate_model'].predict_proba(fit['X_te'])[:, 1]
                        buy_test = fit['buy_model'].predict(fit['X_te'])
                        buy_labels = np.asarray([fit['id_to_price'][int(i)] for i in buy_test], dtype=object)
                        metrics = _simulate_eval(split.test, _apply_calibration_meta(cal_meta, np.asarray(dir_test, dtype=float)), np.asarray(gate_test, dtype=float), buy_labels, take_threshold=0.50)
                        rows.append({
                            'asset': asset,
                            'recipe': recipe['name'],
                            'bundle': bundle_name,
                            'groups': list(groups),
                            'feature_cols_count': len(feature_cols),
                            'effective_feature_days': int(episodes.attrs.get('effective_feature_days') or FEATURE_COMPLETE_DAYS),
                            'path_coverage_ratio': episodes.attrs.get('path_coverage_ratio'),
                            'train_days': train_days,
                            'template': template_name,
                            'params': params,
                            'metrics': metrics,
                            'score': _round_float(_candidate_score(metrics), 6),
                            'status': 'ok',
                        })
                    except Exception as exc:
                        rows.append({'asset': asset, 'recipe': recipe['name'], 'bundle': bundle_name, 'train_days': train_days, 'template': template_name, 'status': f'skipped_fit:{exc}'})
    leaderboard = pd.DataFrame(rows)
    if not leaderboard.empty and 'score' in leaderboard.columns:
        leaderboard = leaderboard.sort_values(['score'], ascending=[False], na_position='last').reset_index(drop=True)
    return leaderboard, episode_cache


def _train_asset(asset: str, force_rebuild_episodes: bool = False) -> dict[str, Any]:
    model_dir = BTC_MODEL_DIR if asset == 'BTC_USDT' else ETH_MODEL_DIR
    episodes_path = BTC_EPISODES if asset == 'BTC_USDT' else ETH_EPISODES
    leaderboard_path = BTC_LEADERBOARD if asset == 'BTC_USDT' else ETH_LEADERBOARD
    objective_cfg = _default_objective_cfg()
    stage_a, episode_cache = _run_stage_a(asset, objective_cfg, force_rebuild_episodes=force_rebuild_episodes)
    write_json(leaderboard_path, {'generated_at': utc_now_iso(), 'asset': asset, 'stage': 'coarse', 'rows': stage_a.to_dict(orient='records') if not stage_a.empty else []})
    ok = (
        stage_a[(stage_a['status'] == 'ok') & (pd.to_numeric(stage_a['score'], errors='coerce') > -1e8)]
        .head(3)
        .to_dict(orient='records')
        if not stage_a.empty else []
    )
    if not ok:
        raise RuntimeError(
            f'no viable stage-a candidates for {asset}; '
            'all candidates failed profit/trade/abstain gates'
        )

    finalists: list[dict[str, Any]] = []
    for row in ok:
        recipe_name = str(row['recipe'])
        bundle_name = str(row['bundle'])
        recipe = next(r for r in ASSET_RECIPES[asset] if r['name'] == recipe_name)
        groups = list(_asset_bundle_map(asset)[bundle_name])
        cache_key = f'{recipe_name}::{bundle_name}'
        episodes = episode_cache[cache_key]
        feature_cols = _choose_feature_cols(episodes, groups)
        objective = _objective_factory(asset, episodes, recipe, bundle_name, groups, int(row['train_days']), feature_cols)
        study = optuna.create_study(direction='maximize', study_name=f'{asset}_{recipe_name}_{bundle_name}_{row["template"]}')
        study.optimize(objective, n_trials=120, n_jobs=SEARCH_WORKERS, show_progress_bar=False)
        best = study.best_trial
        finalists.append({
            'recipe_name': recipe_name,
            'recipe': recipe,
            'bundle': bundle_name,
            'groups': groups,
            'train_days': int(best.user_attrs.get('train_days') or row['train_days']),
            'seed_template': row.get('template'),
            'coarse_score': row.get('score'),
            'best_value': _round_float(best.value, 6),
            'best_params': best.params,
            'best_metrics': best.user_attrs.get('metrics', {}),
            'best_calibration': best.user_attrs.get('calibration', {}),
            'feature_cols': best.user_attrs.get('feature_cols', []),
        })
    finalists = sorted(finalists, key=lambda x: float(x.get('best_value') or -1e9), reverse=True)
    finalists = [row for row in finalists if _is_viable_metrics(row.get('best_metrics') or {})]
    if not finalists:
        raise RuntimeError(
            f'no viable finalists for {asset}; '
            'all optimized candidates failed profit/trade/abstain gates'
        )
    winner = finalists[0]
    recipe = dict(winner['recipe'])
    groups = list(winner['groups'])
    cache_key = f"{winner['recipe_name']}::{winner['bundle']}"
    base_episodes = episode_cache[cache_key]
    episodes = base_episodes.copy().reset_index(drop=True)
    episodes.attrs = dict(getattr(base_episodes, 'attrs', {}) or {})
    feature_cols = list(winner['feature_cols'])
    buy_actions = [f'{p:.2f}' for p in BUY_GRIDS[asset]]
    params = dict(winner['best_params'])
    take_threshold = float(params.pop('take_threshold'))
    model_dir.mkdir(parents=True, exist_ok=True)
    effective_feature_days = int(episodes.attrs.get('effective_feature_days') or FEATURE_COMPLETE_DAYS)
    effective_window_days = _candidate_train_days(effective_feature_days)
    if not effective_window_days:
        raise RuntimeError(f'no valid final window_days_list for {asset} effective_days={effective_feature_days}')

    # 保存 episode 快照（最终赢家配置）
    _write_episode_parquet(episodes, episodes_path)

    # 训练资产有效窗口内的多窗口集成
    direction_files: list[str] = []
    gate_files: list[str] = []
    buy_files: list[str] = []
    training_rows: list[dict[str, Any]] = []
    lookup_inputs: list[pd.DataFrame] = []
    lookup_dir_probs: list[np.ndarray] = []
    lookup_buy_labels: list[np.ndarray] = []
    bucket_edges = _bucket_edges(episodes)

    for window_days in effective_window_days:
        folds = _iter_purged_splits(episodes, train_days=window_days, max_folds=1)
        if not folds:
            training_rows.append({'window_days': window_days, 'status': 'skipped_no_recent_fold'})
            continue
        split = folds[0]
        fit = _fit_heads(split, feature_cols, params, buy_actions, lgb_jobs=8)
        cal_meta = _select_calibration_from_split(fit, take_threshold)

        dir_test = fit['dir_model'].predict_proba(fit['X_te'])[:, 1]
        gate_test = fit['gate_model'].predict_proba(fit['X_te'])[:, 1]
        buy_test = fit['buy_model'].predict(fit['X_te'])
        buy_test_labels = np.asarray([fit['id_to_price'][int(i)] for i in buy_test], dtype=object)
        cal_dir_test = _apply_calibration_meta(cal_meta, np.asarray(dir_test, dtype=float))
        metrics = _simulate_eval(split.test, cal_dir_test, np.asarray(gate_test, dtype=float), buy_test_labels, take_threshold)
        lookup_inputs.append(split.test.reset_index(drop=True))
        lookup_dir_probs.append(cal_dir_test)
        lookup_buy_labels.append(buy_test_labels)

        direction_file = f'direction_lgb_{window_days}d.joblib'
        gate_file = f'gate_lgb_{window_days}d.joblib'
        buy_file = f'buy_lgb_{window_days}d.joblib'
        joblib.dump(fit['dir_model'], model_dir / direction_file)
        joblib.dump(fit['gate_model'], model_dir / gate_file)
        joblib.dump(fit['buy_model'], model_dir / buy_file)
        direction_files.append(direction_file)
        gate_files.append(gate_file)
        buy_files.append(buy_file)
        training_rows.append({
            'window_days': window_days,
            'status': 'trained',
            'metrics': metrics,
            'direction_best_iteration': int(getattr(fit['dir_model'], 'best_iteration_', 0) or 0),
            'gate_best_iteration': int(getattr(fit['gate_model'], 'best_iteration_', 0) or 0),
            'buy_best_iteration': int(getattr(fit['buy_model'], 'best_iteration_', 0) or 0),
            'take_threshold': _round_float(take_threshold, 6),
            'calibration': cal_meta,
        })

    lookup_rows = pd.concat(lookup_inputs, ignore_index=True) if lookup_inputs else episodes.tail(500).reset_index(drop=True)
    lookup_dir = np.concatenate(lookup_dir_probs) if lookup_dir_probs else np.full(len(lookup_rows), 0.5)
    lookup_buy = np.concatenate(lookup_buy_labels) if lookup_buy_labels else np.asarray(['ABSTAIN'] * len(lookup_rows), dtype=object)
    exit_lookup = _build_exit_policy_lookup(lookup_rows, lookup_dir, lookup_buy, bucket_edges, asset)

    config = {
        'generated_at': utc_now_iso(),
        'version': 'vnext_btc_entry_exit_v1' if asset == 'BTC_USDT' else 'vnext_eth_entry_exit_v1',
        'asset': asset,
        'compare_only': True,
        'monitor_only': True,
        'mode': 'entry_exit_overlay_compare_only',
        'decision_mode': 't_plus_zero',
        'feature_complete_days_cap': FEATURE_COMPLETE_DAYS,
        'effective_feature_days': effective_feature_days,
        'path_coverage_ratio': episodes.attrs.get('path_coverage_ratio'),
        'path_span_days': episodes.attrs.get('path_span_days'),
        'feature_bundle': winner['bundle'],
        'feature_groups': groups,
        'feature_cols': feature_cols,
        'indicator_recipe': recipe,
        'window_days_list': effective_window_days,
        'best_train_days_seed': int(winner['train_days']),
        'buy_actions': buy_actions,
        'exit_families': EXIT_FAMILIES,
        'sell_quote_policies': SELL_QUOTE_POLICIES,
        'take_threshold': _round_float(take_threshold, 6),
        'best_params': winner['best_params'],
        'best_metrics': winner['best_metrics'],
        'calibration': next((row['calibration'] for row in training_rows if row.get('status') == 'trained' and row.get('calibration')), winner.get('best_calibration', {})),
        'bucket_edges': bucket_edges,
        'model_files': {
            'direction': direction_files,
            'gate': gate_files,
            'buy': buy_files,
        },
        'training_rows': training_rows,
        'notes': [
            'compare_only only',
            'btc_eth_separate_heads',
            'non_gru_stage1',
            't_plus_zero_decision',
            '1m_path_labels_overlay_only',
            'no_cfgi_news_gru_prob_target',
            'profit_first_selection',
            'wallet_cohort_disabled_by_default',
            'asset_specific_effective_window',
        ],
    }
    write_json(model_dir / 'config.json', config)
    write_json(model_dir / 'feature_cols.json', feature_cols)
    write_json(model_dir / 'asset_map.json', {asset: 0})
    write_json(model_dir / 'indicator_recipe.json', recipe)
    write_json(model_dir / 'hyperparams_snapshot.json', {'take_threshold': take_threshold, 'lgbm_params': winner['best_params']})
    write_json(model_dir / 'exit_policy_lookup.json', exit_lookup)
    write_json(leaderboard_path, {
        'generated_at': utc_now_iso(),
        'asset': asset,
        'stage': 'final',
        'coarse_rows': stage_a.to_dict(orient='records') if not stage_a.empty else [],
        'finalists': finalists,
        'winner': winner,
        'model_dir': str(model_dir),
        'episodes_path': str(episodes_path),
        'effective_feature_days': effective_feature_days,
        'path_coverage_ratio': episodes.attrs.get('path_coverage_ratio'),
    })
    return {
        'asset': asset,
        'model_dir': str(model_dir),
        'winner': winner,
        'episodes_path': str(episodes_path),
        'leaderboard_path': str(leaderboard_path),
        'effective_feature_days': effective_feature_days,
        'path_coverage_ratio': episodes.attrs.get('path_coverage_ratio'),
        'bucket_edges': bucket_edges,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description='Train BTC/ETH compare-only entry+exit models (profit-first, 180d feature-complete)')
    ap.add_argument('--force-episodes', action='store_true')
    args = ap.parse_args()

    os.environ.setdefault('OMP_NUM_THREADS', '8')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
    os.environ.setdefault('MKL_NUM_THREADS', '8')
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    btc = _train_asset('BTC_USDT', force_rebuild_episodes=args.force_episodes)
    eth = _train_asset('ETH_USDT', force_rebuild_episodes=args.force_episodes)
    compare_payload = {
        'generated_at': utc_now_iso(),
        'status': 'trained',
        'scope': 'compare_only_entry_exit_v1',
        'selection_mode': 'absolute_profit_first',
        'assets': {
            'BTC_USDT': {
                'model_dir': btc['model_dir'],
                'winner': btc['winner'],
                'effective_feature_days': btc.get('effective_feature_days'),
                'path_coverage_ratio': btc.get('path_coverage_ratio'),
            },
            'ETH_USDT': {
                'model_dir': eth['model_dir'],
                'winner': eth['winner'],
                'effective_feature_days': eth.get('effective_feature_days'),
                'path_coverage_ratio': eth.get('path_coverage_ratio'),
            },
        },
        'notes': [
            'compare_only only',
            'not optimized against running exp models',
            'monitor_all_orders handles later side-by-side observation',
            'win_rate defined by pnl > 0',
        ],
    }
    write_json(COMPARE_REPORT, compare_payload)
    write_json(TRAIN_REPORT, {
        'generated_at': utc_now_iso(),
        'status': 'trained',
        'scope': 'vnext_btceth_entry_exit_v1',
        'assets': {'BTC_USDT': btc, 'ETH_USDT': eth},
        'compare_report': str(COMPARE_REPORT),
    })
    print(json.dumps({'status': 'trained', 'btc_model_dir': str(BTC_MODEL_DIR), 'eth_model_dir': str(ETH_MODEL_DIR), 'compare_report': str(COMPARE_REPORT)}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
