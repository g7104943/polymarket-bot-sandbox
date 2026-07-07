#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports'
OPS = ROOT / 'scripts' / 'ops'
REPORT = REPORTS / 'live_candidate_validation_latest.json'
REFRESH_CMDS = [
    ['python3', str(OPS / 'generate_execution_calibration_contract_latest.py')],
    ['python3', str(OPS / 'generate_limit_price_fill_realism_gate_latest.py')],
    ['python3', str(OPS / 'generate_live_rollout_evidence_latest.py')],
    ['python3', str(OPS / 'generate_live_monitor_readiness_latest.py')],
    ['python3', str(OPS / 'generate_live_switch_candidate_review_latest.py')],
    ['python3', str(OPS / 'generate_live_candidate_validation_latest.py')],
]


def _load_json(path: Path, default: Any) -> Any:
    try:
        if not path.exists():
            return default
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return default


def _refresh_reports() -> list[str]:
    failures: list[str] = []
    for cmd in REFRESH_CMDS:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if proc.returncode != 0:
            failures.append(f"{Path(cmd[-1]).name}:{proc.returncode}")
    return failures


def _print_table(rows: list[dict[str, Any]], profile: str) -> None:
    width = 164
    print()
    print('=' * width)
    print('  Live Candidate Monitor')
    print(f"  刷新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | profile={profile}")
    print('=' * width)
    fmt = "  {:<34} {:<10} {:<10} {:<16} {:>9} {:>9} {:>9} {:>9} {:>10}"
    print(fmt.format('候选', '运行态', '准入', '验证态', '原始$', '固定低价$', '动态低价$', '近实盘$', '差额$'))
    print('-' * width)
    for row in rows:
        print(
            fmt.format(
                str(row.get('short') or '')[:34],
                str(row.get('operationalReadinessStatus') or '-')[:10],
                str(row.get('promotionEligibilityStatus') or '-')[:10],
                str(row.get('candidateStage') or '-')[:16],
                f"{float(row.get('rawSimPnlUsd') or 0.0):+.2f}",
                f"{float(row.get('fixedLowpricePnlUsd') or 0.0):+.2f}",
                f"{float(row.get('dynamicLowpricePnlUsd') or 0.0):+.2f}",
                f"{float(row.get('liveEquivalentPnlUsd') or 0.0):+.2f}",
                f"{float(row.get('rawToExecutableGapUsd') or 0.0):+.2f}",
            )
        )
    print('-' * width)
    print(f"  合计: {len(rows)} 行候选")


def main() -> int:
    ap = argparse.ArgumentParser(description='独立 live 候选视图')
    ap.add_argument('--profile', choices=('default', '70', 'all'), default='70')
    ap.add_argument('--interval', '-i', type=int, default=60)
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--no-refresh', action='store_true')
    args = ap.parse_args()

    def _run_once() -> None:
        failures = [] if args.no_refresh else _refresh_reports()
        payload = _load_json(REPORT, {})
        rows = payload.get('rows') if isinstance(payload.get('rows'), list) else []
        if args.profile != 'all':
            rows = [row for row in rows if str(row.get('profile') or '') == args.profile]
        _print_table(rows, args.profile)
        if failures:
            print(f"  报告刷新失败: {' '.join(failures)}")

    if args.once:
        _run_once()
        return 0

    print('  按 Ctrl+C 退出')
    try:
        while True:
            _run_once()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print('\n  已停止')
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
