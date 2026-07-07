#!/usr/bin/env python3
"""
归档所有 logs_* 目录中的交易数据，为全新模拟交易做准备。

归档内容: prediction_trades*.json, pending_sim_orders*.json, pending_order_ledger*.jsonl, legacy postonly_trades.json, reports/ 目录, halt_state.json, executed_target_ts.json
归档目标: polymarket/backup_YYYY-MM-DD_HHMMSS/
保留: 日志目录结构不变（启动脚本需要它们存在）

用法:
  python scripts/archive_trade_data.py
"""
from __future__ import annotations

import os
import shutil
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLYMARKET_DIR = PROJECT_ROOT / "polymarket"

FILES_TO_ARCHIVE = [
    "prediction_trades.json",
    "postonly_trades.json",
    "halt_state.json",
    "executed_target_ts.json",
]

FILE_GLOBS_TO_ARCHIVE = [
    "prediction_trades*.json",
    "pending_sim_orders*.json",
    "pending_order_ledger*.jsonl",
]

DIRS_TO_ARCHIVE = [
    "reports",
]


def main():
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    backup_root = POLYMARKET_DIR / f"backup_{ts}"

    log_dirs = sorted(d for d in POLYMARKET_DIR.iterdir()
                      if d.is_dir() and d.name.startswith("logs_"))

    print(f"{'=' * 60}")
    print(f"  交易数据归档")
    print(f"  源目录: {POLYMARKET_DIR}")
    print(f"  备份目标: {backup_root.name}/")
    print(f"  待处理: {len(log_dirs)} 个 logs_* 目录")
    print(f"{'=' * 60}")

    archived_files = 0
    archived_dirs = 0

    for log_dir in log_dirs:
        rel = log_dir.name
        any_found = False

        archived_names: set[str] = set()
        for fname in FILES_TO_ARCHIVE:
            src = log_dir / fname
            if src.exists():
                dst_dir = backup_root / rel
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst_dir / fname))
                archived_files += 1
                any_found = True
                archived_names.add(fname)

        for pattern in FILE_GLOBS_TO_ARCHIVE:
            for src in sorted(log_dir.glob(pattern)):
                if src.name in archived_names:
                    continue
                dst_dir = backup_root / rel
                dst_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst_dir / src.name))
                archived_files += 1
                any_found = True
                archived_names.add(src.name)

        for dname in DIRS_TO_ARCHIVE:
            src_dir = log_dir / dname
            if src_dir.exists() and any(src_dir.iterdir()):
                dst_dir = backup_root / rel / dname
                dst_dir.mkdir(parents=True, exist_ok=True)
                for f in list(src_dir.iterdir()):
                    if f.is_file():
                        shutil.move(str(f), str(dst_dir / f.name))
                        archived_files += 1
                archived_dirs += 1
                any_found = True

        if any_found:
            print(f"  {rel}")

    print(f"\n{'=' * 60}")
    print(f"  归档完成: {archived_files} 个文件, {archived_dirs} 个报告目录")
    print(f"  备份位置: {backup_root}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
