#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / 'reports'
POLY_DIR = PROJECT_ROOT / 'polymarket'
MODELS_DIR = PROJECT_ROOT / 'data' / 'models'

ASSETS = ('BTC_USDT', 'ETH_USDT')
ASSET_SYMBOL = {'BTC_USDT': 'BTC', 'ETH_USDT': 'ETH'}
SYMBOL_ASSET = {v: k for k, v in ASSET_SYMBOL.items()}

V1_MODEL_DIRS = {
    'BTC_USDT': MODELS_DIR / 'vnext_btc_entry_exit_v1',
    'ETH_USDT': MODELS_DIR / 'vnext_eth_entry_exit_v1',
}
V2A_MODEL_DIRS = {
    'BTC_USDT': MODELS_DIR / 'vnext_btc_execution_v2',
    'ETH_USDT': MODELS_DIR / 'vnext_eth_execution_v2',
}
V2B_MODEL_DIRS = {
    'BTC_USDT': MODELS_DIR / 'vnext_btc_profit_alpha_v2',
    'ETH_USDT': MODELS_DIR / 'vnext_eth_profit_alpha_v2',
}
V3_MODEL_DIRS = {
    'BTC_USDT': MODELS_DIR / 'vnext_btc_sequence_v3',
    'ETH_USDT': MODELS_DIR / 'vnext_eth_sequence_v3',
}

V2_MONITOR_TEMPLATE = POLY_DIR / 'monitor_only_traders_execution_v2.template.json'
V2_ACTIVE_TEMPLATE = POLY_DIR / 'active_traders_monitor_only_execution_v2.template.json'
V2_CONFIG_TEMPLATE = POLY_DIR / 'trader_configs_monitor_only_execution_v2.template.json'
V2_MONITOR_FILE = POLY_DIR / 'monitor_only_traders_execution_v2.json'
V2_ACTIVE_FILE = POLY_DIR / 'active_traders_monitor_only_execution_v2.json'
V2_CONFIG_FILE = POLY_DIR / 'trader_configs_monitor_only_execution_v2.json'

V3_MONITOR_TEMPLATE = POLY_DIR / 'monitor_only_traders_sequence_v3.template.json'
V3_ACTIVE_TEMPLATE = POLY_DIR / 'active_traders_monitor_only_sequence_v3.template.json'
V3_CONFIG_TEMPLATE = POLY_DIR / 'trader_configs_monitor_only_sequence_v3.template.json'
V3_MONITOR_FILE = POLY_DIR / 'monitor_only_traders_sequence_v3.json'
V3_ACTIVE_FILE = POLY_DIR / 'active_traders_monitor_only_sequence_v3.json'
V3_CONFIG_FILE = POLY_DIR / 'trader_configs_monitor_only_sequence_v3.json'

ILLEGAL_FEATURE_COLS = {
    'next_close', 'next_return', 'sample_weight',
    'prediction_ts', 'target_start_ts', 'market_start_ts', 'market_end_ts', 'decision_ts',
    'actual_up', 'gate_label', 'direction_target', 'buy_label',
    'best_utility', 'baseline_hold_pnl', 'execution_net_pnl',
    'entry_fill_proxy', 'entry_fill_status', 'entry_fill_ts', 'entry_fill_fraction',
    'entry_proxy_ts', 'market_discovered_ts', 'order_submitted_ts', 'timeout_ts',
    'target_period_end_ts', 'best_action_key', 'entry_action_utility_json',
    'entry_action_meta_json', 'entry_trace_json', 'action_utility_map_json',
    'action_pnl_map_json', 'action_meta_map_json', 'exit_reason',
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def round_float(v: Any, digits: int = 6) -> Any:
    try:
        fv = float(v)
    except Exception:
        return v
    if math.isnan(fv) or math.isinf(fv):
        return None
    return round(fv, digits)


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


def load_trade_array(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    return [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []


def current_cycle_trade_stats(path: Path) -> dict[str, Any]:
    rows = load_trade_array(path)
    wins = 0
    losses = 0
    pending = 0
    pnl = 0.0
    for row in rows:
        result = str(row.get('result') or '').lower()
        if result == 'win':
            wins += 1
        elif result == 'lose':
            losses += 1
        else:
            pending += 1
        try:
            pnl += float(row.get('pnl') or 0.0)
        except Exception:
            pass
    return {
        'rows': int(len(rows)),
        'wins': int(wins),
        'losses': int(losses),
        'pending': int(pending),
        'pnl': round_float(pnl, 6),
    }


def load_v1_runtime_config(asset: str) -> dict[str, Any]:
    cfg = load_json(V1_MODEL_DIRS[asset] / 'config.json')
    if not isinstance(cfg, dict):
        raise RuntimeError(f'missing v1 config for {asset}')
    return cfg


def spread_bucket_from_value(val: float) -> str:
    fv = float(val or 0.0)
    if fv < 0.01:
        return 'tight'
    if fv < 0.03:
        return 'normal'
    return 'wide'


def execution_lookup_key(exit_family: str, sell_quote_policy: str, spread_metric: float | None) -> str:
    return '|'.join([str(exit_family), str(sell_quote_policy), spread_bucket_from_value(float(spread_metric or 0.0))])


def load_execution_policy(asset: str, exit_family: str, sell_quote_policy: str, spread_metric: float | None) -> dict[str, Any]:
    lookup = load_json(V2A_MODEL_DIRS[asset] / 'fill_policy_lookup.json')
    rows = (lookup or {}).get('rows') if isinstance(lookup, dict) else {}
    default = (lookup or {}).get('default') if isinstance(lookup, dict) else None
    if not isinstance(default, dict):
        default = {
            'fill_rate': 0.92,
            'partial_fill_rate': 0.08,
            'timeout_rate': 0.05,
            'avg_partial_fill_ratio': 0.48,
            'avg_queue_wait_seconds': 22.0,
            'avg_entry_queue_depth': 850.0 if asset == 'BTC_USDT' else 900.0,
            'avg_exit_queue_depth': 850.0 if asset == 'BTC_USDT' else 900.0,
            'default_exit_family': 'TIME_EXIT',
            'default_sell_quote_policy': 'TIMEOUT_THEN_HIT_BID',
        }
    key = execution_lookup_key(exit_family, sell_quote_policy, spread_metric)
    selected = rows.get(key) if isinstance(rows, dict) else None
    if isinstance(selected, dict):
        return {
            'selected_exit_family': str(selected.get('exit_family') or exit_family),
            'selected_sell_quote_policy': str(selected.get('sell_quote_policy') or sell_quote_policy),
            'fill_rate': float(selected.get('fill_rate') or default.get('fill_rate') or 0.92),
            'partial_fill_rate': float(selected.get('partial_fill_rate') or default.get('partial_fill_rate') or 0.08),
            'timeout_rate': float(selected.get('timeout_rate') or default.get('timeout_rate') or 0.05),
            'avg_partial_fill_ratio': float(selected.get('avg_partial_fill_ratio') or default.get('avg_partial_fill_ratio') or 0.48),
            'avg_queue_wait_seconds': float(selected.get('avg_queue_wait_seconds') or default.get('avg_queue_wait_seconds') or 22.0),
            'avg_entry_queue_depth': float(selected.get('avg_entry_queue_depth') or default.get('avg_entry_queue_depth') or (850.0 if asset == 'BTC_USDT' else 900.0)),
            'avg_exit_queue_depth': float(selected.get('avg_exit_queue_depth') or default.get('avg_exit_queue_depth') or (850.0 if asset == 'BTC_USDT' else 900.0)),
            'policy_source': 'lookup',
            'spread_bucket': spread_bucket_from_value(float(spread_metric or 0.0)),
            'lookup_key': key,
            'avg_overlay_pnl': round_float(selected.get('avg_overlay_pnl'), 6),
            'samples': int(selected.get('samples') or 0),
        }
    return {
        'selected_exit_family': str(default.get('default_exit_family') or exit_family),
        'selected_sell_quote_policy': str(default.get('default_sell_quote_policy') or sell_quote_policy),
        'fill_rate': float(default.get('fill_rate') or 0.92),
        'partial_fill_rate': float(default.get('partial_fill_rate') or 0.08),
        'timeout_rate': float(default.get('timeout_rate') or 0.05),
        'avg_partial_fill_ratio': float(default.get('avg_partial_fill_ratio') or 0.48),
        'avg_queue_wait_seconds': float(default.get('avg_queue_wait_seconds') or 22.0),
        'avg_entry_queue_depth': float(default.get('avg_entry_queue_depth') or (850.0 if asset == 'BTC_USDT' else 900.0)),
        'avg_exit_queue_depth': float(default.get('avg_exit_queue_depth') or (850.0 if asset == 'BTC_USDT' else 900.0)),
        'policy_source': 'default',
        'spread_bucket': spread_bucket_from_value(float(spread_metric or 0.0)),
        'lookup_key': key,
        'avg_overlay_pnl': None,
        'samples': 0,
    }


def choose_default_execution_policy(asset: str, spread_metric: float | None) -> dict[str, Any]:
    lookup = load_json(V2A_MODEL_DIRS[asset] / 'fill_policy_lookup.json')
    rows = (lookup or {}).get('rows') if isinstance(lookup, dict) else {}
    spread_bucket = spread_bucket_from_value(float(spread_metric or 0.0))
    best_row: dict[str, Any] | None = None
    best_score = -1e18
    if isinstance(rows, dict):
        for key, row in rows.items():
            if not isinstance(row, dict):
                continue
            if str(key).rsplit('|', 1)[-1] != spread_bucket:
                continue
            score = float(row.get('avg_overlay_pnl') or 0.0) * max(float(row.get('fill_rate') or 0.0), 0.05)
            score -= 0.05 * float(row.get('timeout_rate') or 0.0)
            score += 0.01 * min(int(row.get('samples') or 0), 20)
            if score > best_score:
                best_score = score
                best_row = row
    if isinstance(best_row, dict):
        return load_execution_policy(
            asset,
            str(best_row.get('exit_family') or 'TIME_EXIT'),
            str(best_row.get('sell_quote_policy') or 'TIMEOUT_THEN_HIT_BID'),
            spread_metric,
        )
    return load_execution_policy(asset, 'TIME_EXIT', 'TIMEOUT_THEN_HIT_BID', spread_metric)


def ensure_empty_trade_log(path: Path) -> None:
    if not path.exists():
        atomic_write_json(path, [])


def materialize_template(src: Path, dst: Path) -> None:
    payload = load_json(src)
    if payload is None:
        raise RuntimeError(f'missing template: {src}')
    if isinstance(payload, dict):
        payload['generatedAt'] = datetime.now().isoformat()
    atomic_write_json(dst, payload)


def load_parquet_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def find_illegal_feature_cols(feature_cols: list[str] | tuple[str, ...]) -> list[str]:
    return sorted({str(col) for col in feature_cols if str(col) in ILLEGAL_FEATURE_COLS})
