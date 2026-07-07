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
    V2_CONFIG_FILE,
    V2_MONITOR_FILE,
    V3_ACTIVE_FILE,
    V3_ACTIVE_TEMPLATE,
    V3_CONFIG_FILE,
    V3_CONFIG_TEMPLATE,
    V3_MONITOR_FILE,
    V3_MONITOR_TEMPLATE,
    atomic_write_json,
    load_json,
    materialize_template,
)

AUDIT_PATH = Path('/Users/mac/polyfun/reports/vnext_sequence_v3_audit_latest.json')
PROMOTION_REPORT = Path('/Users/mac/polyfun/reports/vnext_sequence_v3_promotion_latest.json')
V1_MONITOR_FILE = POLY_DIR / 'monitor_only_traders_entry_exit.json'
LOG_PATHS = [
    POLY_DIR / 'logs_vnext_btc_sequence_v3' / 'prediction_trades.simulation.json',
    POLY_DIR / 'logs_vnext_eth_sequence_v3' / 'prediction_trades.simulation.json',
    POLY_DIR / 'logs_vnext_btc_sequence_v3_raw' / 'prediction_trades.simulation.json',
    POLY_DIR / 'logs_vnext_eth_sequence_v3_raw' / 'prediction_trades.simulation.json',
]


def main() -> int:
    ap = argparse.ArgumentParser(description='Materialize v3 compare-only monitor lane configs once audit passes')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    audit = load_json(AUDIT_PATH)
    if not args.force:
        if not isinstance(audit, dict) or not bool(audit.get('ready_for_monitor')):
            raise SystemExit('v3 audit not ready for monitor; use --force to bypass')
    materialize_template(V3_MONITOR_TEMPLATE, V3_MONITOR_FILE)
    materialize_template(V3_ACTIVE_TEMPLATE, V3_ACTIVE_FILE)
    materialize_template(V3_CONFIG_TEMPLATE, V3_CONFIG_FILE)
    atomic_write_json(V1_MONITOR_FILE, {'traders': []})
    atomic_write_json(V2_MONITOR_FILE, {'traders': []})
    atomic_write_json(V2_ACTIVE_FILE, {'groups': []})
    atomic_write_json(V2_CONFIG_FILE, [])
    for path in LOG_PATHS:
        atomic_write_json(path, [])
    payload = {
        'generated_at': __import__('datetime').datetime.now().isoformat(),
        'scope': 'vnext_sequence_v3_monitor_promotion',
        'audit_ready_for_monitor': bool((audit or {}).get('ready_for_monitor')) if isinstance(audit, dict) else False,
        'monitor_file': str(V3_MONITOR_FILE),
        'active_file': str(V3_ACTIVE_FILE),
        'config_file': str(V3_CONFIG_FILE),
        'cleared_internal_monitor_files': [
            str(V1_MONITOR_FILE),
            str(V2_MONITOR_FILE),
            str(V2_ACTIVE_FILE),
            str(V2_CONFIG_FILE),
        ],
        'log_paths': [str(p) for p in LOG_PATHS],
    }
    atomic_write_json(PROMOTION_REPORT, payload)
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
