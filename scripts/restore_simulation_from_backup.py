#!/usr/bin/env python3
"""
从 polymarket/backup_YYYY-MM-DD_HHMMSS 恢复模拟交易数据（恢复误执行「按前30重启」前的状态）。

用法:
  python scripts/restore_simulation_from_backup.py              # 使用最新备份
  python scripts/restore_simulation_from_backup.py --list       # 只列出备份
  python scripts/restore_simulation_from_backup.py backup_2026-02-28_211355  # 指定备份目录名
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLYMARKET_DIR = PROJECT_ROOT / "polymarket"

FILES_TO_RESTORE = [
    "prediction_trades.json",
    "postonly_trades.json",
    "halt_state.json",
    "executed_target_ts.json",
]
FILE_GLOBS_TO_RESTORE = [
    "prediction_trades*.json",
    "pending_sim_orders*.json",
    "pending_order_ledger*.jsonl",
]
RESTORE_REPORTS = True


def main() -> int:
    parser = argparse.ArgumentParser(description="从 backup_* 恢复模拟交易数据")
    parser.add_argument("backup_name", nargs="?", help="备份目录名，如 backup_2026-02-28_211355；不传则用最新")
    parser.add_argument("--list", action="store_true", help="只列出可用备份后退出")
    parser.add_argument("--dry-run", action="store_true", help="只打印将要恢复的文件，不写入")
    args = parser.parse_args()

    if not POLYMARKET_DIR.exists():
        print(f"❌ 目录不存在: {POLYMARKET_DIR}", file=sys.stderr)
        return 1

    backups = sorted(
        (d for d in POLYMARKET_DIR.iterdir() if d.is_dir() and d.name.startswith("backup_")),
        key=lambda x: x.name,
        reverse=True,
    )
    if not backups:
        print("❌ 未找到任何 backup_* 目录", file=sys.stderr)
        return 1

    if args.list:
        print("可用备份（按时间倒序）:")
        for b in backups:
            print(f"  {b.name}")
        return 0

    if args.backup_name:
        backup_root = POLYMARKET_DIR / args.backup_name
        if not backup_root.exists() or not backup_root.is_dir():
            print(f"❌ 备份不存在: {backup_root}", file=sys.stderr)
            return 1
    else:
        backup_root = backups[0]
        print(f"使用最新备份: {backup_root.name}")

    restored_files = 0
    for subdir in sorted(backup_root.iterdir()):
        if not subdir.is_dir() or not subdir.name.startswith("logs_"):
            continue
        dest = POLYMARKET_DIR / subdir.name
        if not dest.exists() or not dest.is_dir():
            continue
        for fname in FILES_TO_RESTORE:
            src = subdir / fname
            if src.exists():
                dst = dest / fname
                if args.dry_run:
                    print(f"  [DRY] {subdir.name}/{fname} -> {dest.name}/")
                else:
                    shutil.copy2(str(src), str(dst))
                restored_files += 1
        restored_names = set(FILES_TO_RESTORE)
        for pattern in FILE_GLOBS_TO_RESTORE:
            for src in sorted(subdir.glob(pattern)):
                if src.name in restored_names:
                    continue
                dst = dest / src.name
                if args.dry_run:
                    print(f"  [DRY] {subdir.name}/{src.name} -> {dest.name}/")
                else:
                    shutil.copy2(str(src), str(dst))
                restored_files += 1
                restored_names.add(src.name)
        if RESTORE_REPORTS:
            reports_src = subdir / "reports"
            reports_dest = dest / "reports"
            if reports_src.exists() and reports_src.is_dir():
                reports_dest.mkdir(parents=True, exist_ok=True)
                for f in reports_src.iterdir():
                    if f.is_file():
                        if args.dry_run:
                            print(f"  [DRY] {subdir.name}/reports/{f.name} -> {dest.name}/reports/")
                        else:
                            shutil.copy2(str(f), str(reports_dest / f.name))
                        restored_files += 1

    print(f"{'[DRY] ' if args.dry_run else ''}恢复完成: {restored_files} 个文件，来自 {backup_root.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
