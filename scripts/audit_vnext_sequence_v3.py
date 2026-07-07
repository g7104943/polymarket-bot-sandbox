#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.vnext_stage_common import (
    ASSETS,
    POLY_DIR,
    REPORTS_DIR,
    V3_MODEL_DIRS,
    current_cycle_trade_stats,
    find_illegal_feature_cols,
    load_json,
    utc_now_iso,
)
from scripts.vnext_execution_common import load_latest_reset_report

TRAINING_REPORT = REPORTS_DIR / 'vnext_sequence_v3_training_latest.json'
OUTPUT_PATH = REPORTS_DIR / 'vnext_sequence_v3_audit_latest.json'
V3_OVERLAY_LOGS = {
    'BTC_USDT': POLY_DIR / 'logs_vnext_btc_sequence_v3' / 'prediction_trades.simulation.json',
    'ETH_USDT': POLY_DIR / 'logs_vnext_eth_sequence_v3' / 'prediction_trades.simulation.json',
}


def _asset_report(asset: str) -> dict[str, Any]:
    payload = load_json(TRAINING_REPORT)
    asset_payload = ((payload.get('assets') or {}) if isinstance(payload, dict) else {}).get(asset) or {}
    metrics = asset_payload.get('final_metrics') or {}
    model_dir = V3_MODEL_DIRS[asset]
    has_config = (model_dir / 'config.json').exists()
    cfg = load_json(model_dir / 'config.json') if has_config else {}
    illegal_feature_cols = find_illegal_feature_cols(list((cfg or {}).get('feature_cols') or [])) if isinstance(cfg, dict) else []
    total_pnl = float(metrics.get('total_pnl') or 0.0)
    trades = int(metrics.get('trades') or 0)
    abstain_rate = float(metrics.get('abstain_rate') or 1.0)
    ready = bool(has_config and not illegal_feature_cols and total_pnl > 0 and trades >= 30 and abstain_rate < 0.70)
    return {
        'asset': asset,
        'model_dir': str(model_dir),
        'has_config': has_config,
        'illegal_feature_cols': illegal_feature_cols,
        'training_asset_payload': asset_payload,
        'training_metrics': metrics,
        'runtime_overlay_stats': current_cycle_trade_stats(V3_OVERLAY_LOGS[asset]),
        'ready_for_monitor': ready,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description='Audit v3 sequence artifacts before monitor promotion')
    ap.add_argument('--output', type=Path, default=OUTPUT_PATH)
    args = ap.parse_args()
    assets = {asset: _asset_report(asset) for asset in ASSETS}
    payload = {
        'generated_at': utc_now_iso(),
        'scope': 'vnext_sequence_v3_audit',
        'reset_generated_at': load_latest_reset_report().get('generated_at'),
        'ready_for_monitor': all(bool(row.get('ready_for_monitor')) for row in assets.values()),
        'depends_on_training_report': str(TRAINING_REPORT),
        'assets': assets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
