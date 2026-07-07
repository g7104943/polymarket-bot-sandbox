#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.vnext_execution_common import EXECUTION_REPORT, PROFIT_ALPHA_REPORT, V2A_MODEL_DIRS, V2B_MODEL_DIRS
from scripts.vnext_stage_common import (
    ASSETS,
    POLY_DIR,
    REPORTS_DIR,
    current_cycle_trade_stats,
    find_illegal_feature_cols,
    load_json,
    round_float,
    utc_now_iso,
)
from scripts.vnext_execution_common import load_latest_reset_report

OUTPUT_PATH = REPORTS_DIR / 'vnext_execution_v2_audit_latest.json'
V2_OVERLAY_LOGS = {
    'BTC_USDT': POLY_DIR / 'logs_vnext_btc_execution_v2' / 'prediction_trades.simulation.json',
    'ETH_USDT': POLY_DIR / 'logs_vnext_eth_execution_v2' / 'prediction_trades.simulation.json',
}


def _asset_report(asset: str) -> dict[str, Any]:
    exec_payload = load_json(EXECUTION_REPORT) or {}
    profit_payload = load_json(PROFIT_ALPHA_REPORT) or {}
    exec_asset = ((exec_payload.get('assets') or {}) if isinstance(exec_payload, dict) else {}).get(asset) or {}
    profit_asset = ((profit_payload.get('assets') or {}) if isinstance(profit_payload, dict) else {}).get(asset) or {}
    winner = profit_asset.get('winner') if isinstance(profit_asset, dict) else {}
    best_metrics = winner.get('best_metrics') if isinstance(winner, dict) else {}
    overlay_stats = current_cycle_trade_stats(V2_OVERLAY_LOGS[asset])
    exec_cfg = V2A_MODEL_DIRS[asset] / 'execution_calibration.json'
    alpha_cfg = V2B_MODEL_DIRS[asset] / 'config.json'
    has_exec = exec_cfg.exists()
    has_alpha = alpha_cfg.exists()
    alpha_cfg_payload = load_json(alpha_cfg) if has_alpha else {}
    illegal_feature_cols = find_illegal_feature_cols(list((alpha_cfg_payload or {}).get('feature_cols') or [])) if isinstance(alpha_cfg_payload, dict) else []
    total_pnl = float(best_metrics.get('total_pnl') or 0.0)
    trades = int(best_metrics.get('trades') or 0)
    abstain_rate = float(best_metrics.get('abstain_rate') or 1.0)
    audit_passed = bool(has_exec and has_alpha and not illegal_feature_cols and total_pnl > 0 and trades >= 30 and abstain_rate < 0.70)
    return {
        'asset': asset,
        'execution_model_dir': str(V2A_MODEL_DIRS[asset]),
        'profit_alpha_model_dir': str(V2B_MODEL_DIRS[asset]),
        'has_execution_calibration': has_exec,
        'has_profit_alpha_config': has_alpha,
        'illegal_feature_cols': illegal_feature_cols,
        'execution_status': str((exec_asset.get('execution_calibration') or {}).get('status') or 'missing'),
        'profit_alpha_winner': winner,
        'profit_alpha_best_metrics': best_metrics,
        'runtime_overlay_stats': overlay_stats,
        'audit_passed': audit_passed,
        'ready_for_v3': audit_passed,
        'ready_for_monitor': False,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description='Audit v2 execution-aware endpoint artifacts before monitor promotion')
    ap.add_argument('--output', type=Path, default=OUTPUT_PATH)
    args = ap.parse_args()
    assets = {asset: _asset_report(asset) for asset in ASSETS}
    payload = {
        'generated_at': utc_now_iso(),
        'scope': 'vnext_execution_v2_audit',
        'reset_generated_at': load_latest_reset_report().get('generated_at'),
        'audit_passed': all(bool(row.get('audit_passed')) for row in assets.values()),
        'ready_for_v3': all(bool(row.get('ready_for_v3')) for row in assets.values()),
        'ready_for_monitor': False,
        'assets': assets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
