#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
MODELS_DIR = DATA_DIR / 'models'
REPORTS_DIR = PROJECT_ROOT / 'reports'
POLYMARKET_DIR = PROJECT_ROOT / 'polymarket'
LOGS_DIR = PROJECT_ROOT / 'logs'

ASSETS = ('BTC_USDT', 'ETH_USDT')
ASSET_SYMBOL = {'BTC_USDT': 'BTC', 'ETH_USDT': 'ETH'}
SYMBOL_ASSET = {v: k for k, v in ASSET_SYMBOL.items()}

RAW_LOG_DIRS = {
    'BTC_USDT': POLYMARKET_DIR / 'logs_vnext_btc_entry_exit_v1_raw',
    'ETH_USDT': POLYMARKET_DIR / 'logs_vnext_eth_entry_exit_v1_raw',
}
OVERLAY_LOG_DIRS = {
    'BTC_USDT': POLYMARKET_DIR / 'logs_vnext_btc_entry_exit_v1',
    'ETH_USDT': POLYMARKET_DIR / 'logs_vnext_eth_entry_exit_v1',
}
V1_MODEL_DIRS = {
    'BTC_USDT': MODELS_DIR / 'vnext_btc_entry_exit_v1',
    'ETH_USDT': MODELS_DIR / 'vnext_eth_entry_exit_v1',
}
V1_EPISODE_PATHS = {
    'BTC_USDT': PROCESSED_DIR / 'vnext_entry_exit_episodes_btc_usdt.parquet',
    'ETH_USDT': PROCESSED_DIR / 'vnext_entry_exit_episodes_eth_usdt.parquet',
}
LEDGER_PATH = POLYMARKET_DIR / 'predictions_vnext_btceth_entry_exit_v1_ledger.json'

ORDERBOOK_EVENT_PATHS = {
    'BTC_USDT': PROCESSED_DIR / 'vnext_execution_orderbook_btc_usdt.jsonl',
    'ETH_USDT': PROCESSED_DIR / 'vnext_execution_orderbook_eth_usdt.jsonl',
}
LIFECYCLE_EVENT_PATHS = {
    'BTC_USDT': PROCESSED_DIR / 'vnext_execution_lifecycle_btc_usdt.jsonl',
    'ETH_USDT': PROCESSED_DIR / 'vnext_execution_lifecycle_eth_usdt.jsonl',
}
EXECUTION_LABEL_PATHS = {
    'BTC_USDT': PROCESSED_DIR / 'vnext_execution_labels_btc_usdt.parquet',
    'ETH_USDT': PROCESSED_DIR / 'vnext_execution_labels_eth_usdt.parquet',
}
RELABELED_EPISODE_PATHS = {
    'BTC_USDT': PROCESSED_DIR / 'vnext_profit_relabel_btc_usdt_v2.parquet',
    'ETH_USDT': PROCESSED_DIR / 'vnext_profit_relabel_eth_usdt_v2.parquet',
}
V2A_MODEL_DIRS = {
    'BTC_USDT': MODELS_DIR / 'vnext_btc_execution_v2',
    'ETH_USDT': MODELS_DIR / 'vnext_eth_execution_v2',
}
V2B_MODEL_DIRS = {
    'BTC_USDT': MODELS_DIR / 'vnext_btc_profit_alpha_v2',
    'ETH_USDT': MODELS_DIR / 'vnext_eth_profit_alpha_v2',
}
READINESS_REPORT = REPORTS_DIR / 'vnext_execution_v2_readiness_latest.json'
EXECUTION_REPORT = REPORTS_DIR / 'vnext_execution_v2_training_latest.json'
PROFIT_RELABEL_REPORT = REPORTS_DIR / 'vnext_profit_relabel_v2_latest.json'
PROFIT_ALPHA_REPORT = REPORTS_DIR / 'vnext_profit_alpha_v2_training_latest.json'
HISTORICAL_EXECUTION_BOOTSTRAP_REPORT = REPORTS_DIR / 'vnext_execution_v2_historical_bootstrap_latest.json'
RESET_REPORT = REPORTS_DIR / 'vnext_entry_exit_reset_latest.json'

MONITOR_TEMPLATE_PATH = POLYMARKET_DIR / 'monitor_only_traders_execution_v2.template.json'
ACTIVE_TEMPLATE_PATH = POLYMARKET_DIR / 'active_traders_monitor_only_execution_v2.template.json'
TRADER_CONFIG_TEMPLATE_PATH = POLYMARKET_DIR / 'trader_configs_monitor_only_execution_v2.template.json'


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open('a', encoding='utf-8') as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + '\n')


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _parse_iso_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace('Z', '+00:00')).astimezone(timezone.utc)
    except Exception:
        return None


def _row_event_dt(row: dict[str, Any]) -> datetime | None:
    for key in ('ts', 'generated_at', 'timestamp', 'settledAt'):
        dt = _parse_iso_dt(row.get(key))
        if dt is not None:
            return dt
    return None


def load_latest_reset_report() -> dict[str, Any]:
    payload = load_json(RESET_REPORT)
    return payload if isinstance(payload, dict) else {}


def load_reset_cutoff_dt() -> datetime | None:
    payload = load_latest_reset_report()
    return _parse_iso_dt(payload.get('generated_at'))


def filter_rows_after_reset(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cutoff = load_reset_cutoff_dt()
    if cutoff is None:
        return rows
    out: list[dict[str, Any]] = []
    for row in rows:
        dt = _row_event_dt(row)
        if dt is None or dt >= cutoff:
            out.append(row)
    return out


def load_trade_array(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    return [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []


def overlay_trade_path(asset: str) -> Path:
    return OVERLAY_LOG_DIRS[asset] / 'prediction_trades.simulation.json'


def raw_trade_path(asset: str) -> Path:
    return RAW_LOG_DIRS[asset] / 'prediction_trades.simulation.json'


def load_overlay_rows(asset: str) -> list[dict[str, Any]]:
    return filter_rows_after_reset(load_trade_array(overlay_trade_path(asset)))


def load_raw_rows(asset: str) -> list[dict[str, Any]]:
    return filter_rows_after_reset(load_trade_array(raw_trade_path(asset)))


def load_ledger_markets() -> dict[str, dict[str, Any]]:
    payload = load_json(LEDGER_PATH)
    if not isinstance(payload, dict):
        return {}
    markets = payload.get('markets')
    return {str(k): v for k, v in markets.items()} if isinstance(markets, dict) else {}


def normalize_fill_status(value: Any) -> str:
    text = str(value or '').strip().lower()
    if text in {'filled', 'partial_fill', 'timeout_unfilled', 'queued', 'skipped'}:
        return text
    if text in {'partial', 'partial_filled'}:
        return 'partial_fill'
    if text in {'timeout', 'unfilled', 'no_fill'}:
        return 'timeout_unfilled'
    return 'filled' if text else 'unknown'


def default_execution_bootstrap(asset: str) -> dict[str, Any]:
    sell_defaults = {
        'BTC_USDT': 'TIMEOUT_THEN_HIT_BID',
        'ETH_USDT': 'TIMEOUT_THEN_HIT_BID',
    }
    return {
        'generated_at': utc_now_iso(),
        'asset': asset,
        'execution_calibration': 'bootstrap',
        'defaults': {
            'fill_rate': 0.92,
            'partial_fill_rate': 0.08,
            'timeout_rate': 0.05,
            'avg_partial_fill_ratio': 0.48,
            'avg_queue_wait_seconds': 22.0,
            'default_exit_family': 'TIME_EXIT',
            'default_sell_quote_policy': sell_defaults[asset],
        },
    }


def build_readiness_summary() -> dict[str, Any]:
    ledger = load_ledger_markets()
    reset_payload = load_latest_reset_report()
    assets: dict[str, Any] = {}
    for asset in ASSETS:
        overlay_rows = load_overlay_rows(asset)
        orderbook_rows = filter_rows_after_reset(load_jsonl(ORDERBOOK_EVENT_PATHS[asset]))
        lifecycle_rows = filter_rows_after_reset(load_jsonl(LIFECYCLE_EVENT_PATHS[asset]))
        settled = [
            row for row in overlay_rows
            if str(row.get('status') or '').lower() == 'executed'
            and str(row.get('result') or '').lower() in {'win', 'lose'}
        ]
        lifecycle_fill_samples = [
            row for row in lifecycle_rows
            if normalize_fill_status(row.get('fill_status')) in {'filled', 'partial_fill', 'timeout_unfilled'}
        ]
        assets[asset] = {
            'asset': asset,
            'symbol': ASSET_SYMBOL[asset],
            'settled_compare_only_trades': int(len(settled)),
            'orderbook_snapshots': int(len(orderbook_rows)),
            'fillable_partial_timeout_samples': int(len(lifecycle_fill_samples)),
            'ledger_markets': int(sum(1 for row in ledger.values() if str(row.get('asset') or '') == asset)),
            'execution_label_path': str(EXECUTION_LABEL_PATHS[asset]),
            'relabeled_episode_path': str(RELABELED_EPISODE_PATHS[asset]),
        }
    return {
        'generated_at': utc_now_iso(),
        'scope': 'vnext_execution_v2_readiness',
        'reset_generated_at': reset_payload.get('generated_at'),
        'assets': assets,
    }


def write_readiness_report() -> dict[str, Any]:
    payload = build_readiness_summary()
    write_json(READINESS_REPORT, payload)
    return payload


def load_parquet_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)
