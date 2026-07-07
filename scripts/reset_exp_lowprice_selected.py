#!/usr/bin/env python3
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.exp_lowprice_selected_common import ARCHIVE_ROOT, POLY, REPORTS, dump_json, now_iso

REPORT = REPORTS / 'exp_lowprice_selected_reset_latest.json'

POLY_PATTERNS = [
    'logs_lowprice_*',
    'logs_70_lowprice_*',
    'logs_v5_exp10_lowprice*',
    'lowprice_rules',
    'lowprice_rules_selected',
    'active_traders_monitor_only_lowprice*.json',
    'monitor_only_traders_lowprice*.json',
    'trader_configs_monitor_only_lowprice*.json',
    'active_traders_monitor_only_exp10_lowprice.json',
    'monitor_only_traders_exp10_lowprice.json',
    'trader_configs_monitor_only_exp10_lowprice.json',
    'predictions_exp10_lowprice*.json',
]
REPORT_PATTERNS = [
    'exp_lowprice_*',
    'exp10_lowprice_*',
]
ROOT_LOG_PATTERNS = [
    'lowprice_*.launchd.log',
]


def move_matches(base: Path, pattern: str, archive_root: Path, moved: list[str]) -> None:
    for path in sorted(base.glob(pattern)):
        if not path.exists():
            continue
        rel = path.relative_to(ROOT)
        target = archive_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
        moved.append(str(rel))


def main() -> None:
    archive_dir = ARCHIVE_ROOT / now_iso().replace(':', '').replace('-', '').replace('T', '_').replace('Z', '')
    moved: list[str] = []
    archive_dir.mkdir(parents=True, exist_ok=True)

    for pattern in POLY_PATTERNS:
        move_matches(POLY, pattern, archive_dir, moved)
    for pattern in REPORT_PATTERNS:
        move_matches(REPORTS, pattern, archive_dir, moved)
    logs_dir = ROOT / 'logs'
    if logs_dir.exists():
        for pattern in ROOT_LOG_PATTERNS:
            move_matches(logs_dir, pattern, archive_dir, moved)

    payload = {
        'generatedAt': now_iso(),
        'archiveDir': str(archive_dir),
        'movedCount': len(moved),
        'moved': moved,
    }
    dump_json(REPORT, payload)
    print(payload)


if __name__ == '__main__':
    main()
