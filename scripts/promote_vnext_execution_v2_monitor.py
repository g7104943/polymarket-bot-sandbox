#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.vnext_stage_common import (
    POLY_DIR,
    V2_ACTIVE_FILE,
    V2_ACTIVE_TEMPLATE,
    V2_CONFIG_FILE,
    V2_CONFIG_TEMPLATE,
    V2_MONITOR_FILE,
    V2_MONITOR_TEMPLATE,
    atomic_write_json,
    load_json,
    materialize_template,
)

AUDIT_PATH = Path('/Users/mac/polyfun/reports/vnext_execution_v2_audit_latest.json')
PROMOTION_REPORT = Path('/Users/mac/polyfun/reports/vnext_execution_v2_promotion_latest.json')
LOG_PATHS = [
    POLY_DIR / 'logs_vnext_btc_execution_v2' / 'prediction_trades.simulation.json',
    POLY_DIR / 'logs_vnext_eth_execution_v2' / 'prediction_trades.simulation.json',
    POLY_DIR / 'logs_vnext_btc_execution_v2_raw' / 'prediction_trades.simulation.json',
    POLY_DIR / 'logs_vnext_eth_execution_v2_raw' / 'prediction_trades.simulation.json',
]


def main() -> int:
    ap = argparse.ArgumentParser(description='Materialize v2 compare-only monitor lane configs once audit passes')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    audit = load_json(AUDIT_PATH)
    if not args.force:
        raise SystemExit('v2 monitor promotion is disabled by design; v2 is internal-only and must not be promoted')
    if not isinstance(audit, dict) or not bool(audit.get('audit_passed')):
        raise SystemExit('v2 audit has not passed; refusing forced promotion without a valid audit')
    materialize_template(V2_MONITOR_TEMPLATE, V2_MONITOR_FILE)
    materialize_template(V2_ACTIVE_TEMPLATE, V2_ACTIVE_FILE)
    materialize_template(V2_CONFIG_TEMPLATE, V2_CONFIG_FILE)
    for path in LOG_PATHS:
        atomic_write_json(path, [])
    payload = {
        'generated_at': __import__('datetime').datetime.now().isoformat(),
        'scope': 'vnext_execution_v2_monitor_promotion',
        'audit_ready_for_monitor': False,
        'audit_passed': bool((audit or {}).get('audit_passed')) if isinstance(audit, dict) else False,
        'monitor_file': str(V2_MONITOR_FILE),
        'active_file': str(V2_ACTIVE_FILE),
        'config_file': str(V2_CONFIG_FILE),
        'log_paths': [str(p) for p in LOG_PATHS],
    }
    atomic_write_json(PROMOTION_REPORT, payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
