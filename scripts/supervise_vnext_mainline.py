#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
LOGS = ROOT / 'logs'
POLY = ROOT / 'polymarket'
PYTHON = '/Users/mac/miniforge3/bin/python3'

V1_STATUS = REPORTS / 'vnext_entry_exit_v1_realign_status_latest.json'
V1_AUDIT = REPORTS / 'vnext_entry_exit_v1_alignment_audit_latest.json'
V2_READY = REPORTS / 'vnext_execution_v2_readiness_latest.json'
V2_AUDIT = REPORTS / 'vnext_execution_v2_audit_latest.json'
V3_AUDIT = REPORTS / 'vnext_sequence_v3_audit_latest.json'
STATE_PATH = REPORTS / 'vnext_mainline_supervisor_latest.json'

V2_MANUAL = ROOT / 'scripts' / 'run_vnext_endpoint_v2_manual.sh'
V3_MANUAL = ROOT / 'scripts' / 'run_vnext_sequence_v3_manual.sh'
V1_AUDIT_SCRIPT = ROOT / 'scripts' / 'audit_vnext_entry_exit_v1_alignment.py'
V2_AUDIT_SCRIPT = ROOT / 'scripts' / 'audit_vnext_execution_v2.py'
V3_AUDIT_SCRIPT = ROOT / 'scripts' / 'audit_vnext_sequence_v3.py'
V2_ENDPOINT = ROOT / 'scripts' / 'vnext_endpoint_v2.py'

V3_WRITER = ROOT / 'reload_launchctl_prediction_writer_vnext_btceth_sequence_v3.sh'
V3_MULTI = ROOT / 'reload_launchctl_multi_trading_monitor_only_sequence_v3.sh'
V3_OVERLAY = ROOT / 'reload_launchctl_overlay_vnext_btceth_sequence_v3.sh'

V2_BTC_LOG = POLY / 'logs_vnext_btc_execution_v2' / 'prediction_trades.simulation.json'
V2_ETH_LOG = POLY / 'logs_vnext_eth_execution_v2' / 'prediction_trades.simulation.json'
V2_CONFIGS = [
    ROOT / 'data' / 'models' / 'vnext_btc_profit_alpha_v2' / 'config.json',
    ROOT / 'data' / 'models' / 'vnext_eth_profit_alpha_v2' / 'config.json',
]
V3_CONFIGS = [
    ROOT / 'data' / 'models' / 'vnext_btc_sequence_v3' / 'config.json',
    ROOT / 'data' / 'models' / 'vnext_eth_sequence_v3' / 'config.json',
]


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def _run(cmd: list[str], label: str) -> dict[str, Any]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        'label': label,
        'cmd': cmd,
        'returncode': proc.returncode,
        'stdout': proc.stdout.strip()[-4000:],
        'stderr': proc.stderr.strip()[-4000:],
        'ran_at': datetime.now().isoformat(),
    }


def _has_all(paths: list[Path]) -> bool:
    return all(path.exists() for path in paths)


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except Exception:
        return 0.0


def _needs_refresh(report_path: Path, deps: list[Path]) -> bool:
    if not report_path.exists():
        return True
    report_mtime = _mtime(report_path)
    dep_mtime = max((_mtime(dep) for dep in deps if dep.exists()), default=0.0)
    return dep_mtime > report_mtime


def _pgrep(pattern: str) -> bool:
    return subprocess.run(['pgrep', '-f', pattern], capture_output=True).returncode == 0


def _service_running(kind: str) -> bool:
    patterns = {
        'v2_manual': r"scripts/vnext_endpoint_v2.py (build-execution-dataset|train-execution|build-profit-relabel|train-profit-alpha)",
        'v2_writer': 'prediction_writer_vnext_btceth_execution_v2.py',
        'v2_multi': 'multi_prediction_index.js --group vnext_btceth_execution_v2_compare',
        'v2_overlay': 'reconcile_vnext_btceth_execution_v2_overlay.py',
        'v3_manual': r"(train_vnext_sequence_v3.py|audit_vnext_sequence_v3.py|promote_vnext_sequence_v3_monitor.py)",
        'v3_writer': 'prediction_writer_vnext_btceth_sequence_v3.py',
        'v3_multi': 'multi_prediction_index.js --group vnext_btceth_sequence_v3_compare',
        'v3_overlay': 'reconcile_vnext_btceth_sequence_v3_overlay.py',
    }
    pattern = patterns.get(kind)
    return _pgrep(pattern) if pattern else False


def _trade_stats(path: Path) -> dict[str, Any]:
    rows = _load_json(path)
    if not isinstance(rows, list):
        return {'rows': 0, 'wins': 0, 'losses': 0, 'pending': 0, 'pnl': 0.0}
    wins = losses = pending = 0
    pnl = 0.0
    for row in rows:
        if not isinstance(row, dict):
            continue
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
    return {'rows': len(rows), 'wins': wins, 'losses': losses, 'pending': pending, 'pnl': round(pnl, 6)}


def _v1_runtime_ready(status: dict[str, Any]) -> bool:
    if not isinstance(status, dict):
        return False
    stage = str(status.get('stage') or '')
    counts = status.get('v1_overlay_counts') or {}
    btc = int(((counts.get('BTC') or {}).get('rows') or 0))
    eth = int(((counts.get('ETH') or {}).get('rows') or 0))
    return stage == 'runtime' and (btc > 0 or eth > 0)


def _alignment_ok(audit: dict[str, Any]) -> bool:
    assets = (audit or {}).get('assets') or {}
    if not isinstance(assets, dict) or not assets:
        return False
    ok_assets = 0
    for row in assets.values():
        if not isinstance(row, dict):
            continue
        status = str(row.get('status') or '')
        match_rate = float(row.get('fill_status_match_rate') or 0.0)
        fill_err = row.get('avg_abs_fill_price_error')
        if status == 'ok' and match_rate >= 0.5 and (fill_err is None or float(fill_err) <= 0.08):
            ok_assets += 1
    return ok_assets == len(assets)


def _v2_runtime_ready() -> bool:
    btc = _trade_stats(V2_BTC_LOG)
    eth = _trade_stats(V2_ETH_LOG)
    return btc['rows'] >= 30 and eth['rows'] >= 30 and btc['pnl'] > 0 and eth['pnl'] > 0


def _v2_audit_passed(audit: dict[str, Any]) -> bool:
    if not isinstance(audit, dict):
        return False
    return bool(audit.get('audit_passed'))


def _v2_manual_gate() -> dict[str, Any]:
    proc = subprocess.run([PYTHON, str(V2_ENDPOINT), 'assert-manual-ready'], capture_output=True, text=True)
    payload = {}
    try:
        payload = json.loads(proc.stdout.strip() or '{}')
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    payload['_returncode'] = proc.returncode
    if proc.stderr.strip():
        payload['_stderr'] = proc.stderr.strip()[-2000:]
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description='Supervise v1 -> v2 -> v3 mainline progression')
    ap.add_argument('--loop-seconds', type=int, default=60)
    ap.add_argument('--once', action='store_true')
    args = ap.parse_args()

    while True:
        state = {
            'generated_at': datetime.now().isoformat(),
            'loop_seconds': int(args.loop_seconds),
            'actions': [],
            'summary': {},
        }
        v1_status = _load_json(V1_STATUS) or {}
        v1_audit = _load_json(V1_AUDIT) or {}
        v2_audit = _load_json(V2_AUDIT) or {}
        v3_audit = _load_json(V3_AUDIT) or {}
        v2_manual_gate = _v2_manual_gate()

        state['summary']['v1_stage'] = v1_status.get('stage')
        state['summary']['v1_counts'] = v1_status.get('v1_overlay_counts')
        state['summary']['alignment_ok'] = _alignment_ok(v1_audit)
        state['summary']['v2_manual_ready'] = bool(v2_manual_gate.get('ready'))
        state['summary']['v2_manual_gate'] = v2_manual_gate
        state['summary']['v2_audit_passed'] = _v2_audit_passed(v2_audit)
        state['summary']['v2_ready_for_monitor'] = False
        state['summary']['v2_runtime_ready_for_v3'] = False
        state['summary']['v3_ready_for_monitor'] = bool((v3_audit or {}).get('ready_for_monitor')) if isinstance(v3_audit, dict) else False
        state['summary']['v2_models_present'] = _has_all(V2_CONFIGS)
        state['summary']['v3_models_present'] = _has_all(V3_CONFIGS)
        state['summary']['services'] = {
            'v2_manual': _service_running('v2_manual'),
            'v2_writer': _service_running('v2_writer'),
            'v2_multi': _service_running('v2_multi'),
            'v2_overlay': _service_running('v2_overlay'),
            'v3_manual': _service_running('v3_manual'),
            'v3_writer': _service_running('v3_writer'),
            'v3_multi': _service_running('v3_multi'),
            'v3_overlay': _service_running('v3_overlay'),
        }

        if _v1_runtime_ready(v1_status):
            state['actions'].append(_run([PYTHON, str(V1_AUDIT_SCRIPT), '--sample-size', '30'], 'v1_alignment_audit'))
            v1_audit = _load_json(V1_AUDIT) or {}
            state['summary']['alignment_ok'] = _alignment_ok(v1_audit)

        if (
            state['summary']['v2_manual_ready']
            and not state['summary']['v2_models_present']
            and not state['summary']['services']['v2_manual']
        ):
            state['actions'].append(_run(['bash', str(V2_MANUAL)], 'v2_manual_pipeline'))
            state['actions'].append(_run([PYTHON, str(V2_AUDIT_SCRIPT)], 'v2_audit'))
            v2_audit = _load_json(V2_AUDIT) or {}
            state['summary']['v2_audit_passed'] = _v2_audit_passed(v2_audit)
            state['summary']['v2_ready_for_monitor'] = False
            state['summary']['v2_models_present'] = _has_all(V2_CONFIGS)

        if state['summary']['v2_models_present'] and _needs_refresh(V2_AUDIT, V2_CONFIGS):
            state['actions'].append(_run([PYTHON, str(V2_AUDIT_SCRIPT)], 'v2_audit_refresh'))
            v2_audit = _load_json(V2_AUDIT) or {}
            state['summary']['v2_audit_passed'] = _v2_audit_passed(v2_audit)
            state['summary']['v2_ready_for_monitor'] = False

        if (
            state['summary']['v2_models_present']
            and state['summary']['v2_audit_passed']
            and not state['summary']['v3_models_present']
            and not state['summary']['services']['v2_manual']
            and not state['summary']['services']['v3_manual']
        ):
            state['actions'].append(_run(['bash', str(V3_MANUAL)], 'v3_manual_pipeline'))
            state['actions'].append(_run([PYTHON, str(V3_AUDIT_SCRIPT)], 'v3_audit'))
            v3_audit = _load_json(V3_AUDIT) or {}
            state['summary']['v3_ready_for_monitor'] = bool((v3_audit or {}).get('ready_for_monitor')) if isinstance(v3_audit, dict) else False
            state['summary']['v3_models_present'] = _has_all(V3_CONFIGS)

        if state['summary']['v3_models_present'] and _needs_refresh(V3_AUDIT, V3_CONFIGS):
            state['actions'].append(_run([PYTHON, str(V3_AUDIT_SCRIPT)], 'v3_audit_refresh'))
            v3_audit = _load_json(V3_AUDIT) or {}
            state['summary']['v3_ready_for_monitor'] = bool((v3_audit or {}).get('ready_for_monitor')) if isinstance(v3_audit, dict) else False

        if state['summary']['v3_ready_for_monitor']:
            if not state['summary']['services']['v3_writer']:
                state['actions'].append(_run(['bash', str(V3_WRITER), 'restart'], 'v3_writer_restart'))
            if not state['summary']['services']['v3_multi']:
                state['actions'].append(_run(['bash', str(V3_MULTI), 'restart'], 'v3_multi_restart'))
            if not state['summary']['services']['v3_overlay']:
                state['actions'].append(_run(['bash', str(V3_OVERLAY), 'restart'], 'v3_overlay_restart'))

        _write_json(STATE_PATH, state)
        if args.once:
            break
        time.sleep(max(15, int(args.loop_seconds)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
