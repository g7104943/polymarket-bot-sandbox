#!/usr/bin/env python3
"""
重置所有 logs_* 目录下的模拟交易状态（用于重新初始化后开跑）。

行为:
  - 清空/重建 prediction_trades.json 为 []
  - 删除 pending/legacy-postonly/halt/executed_target 等运行态文件
  - 清空 reports/ 下文件（目录保留）
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLYMARKET_DIR = PROJECT_ROOT / "polymarket"

RESET_FILES = [
    "postonly_trades.json",
    "halt_state.json",
    "executed_target_ts.json",
]

RESET_GLOBS = [
    "prediction_trades*.json",
    "pending_sim_orders*.json",
    "pending_order_ledger*.jsonl",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Reset all simulation logs_* states")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not POLYMARKET_DIR.exists():
        raise SystemExit(f"missing: {POLYMARKET_DIR}")

    log_dirs = sorted(d for d in POLYMARKET_DIR.iterdir() if d.is_dir() and d.name.startswith("logs_"))
    changed = 0
    reports_cleared = 0

    for d in log_dirs:
        for trade_file in sorted(d.glob("prediction_trades*.json")):
            if args.dry_run:
                print(f"[DRY] reset {trade_file}")
            else:
                trade_file.write_text("[]\n", encoding="utf-8")
            changed += 1
        if not any(d.glob("prediction_trades*.json")):
            trades = d / "prediction_trades.json"
            if args.dry_run:
                print(f"[DRY] reset {trades}")
            else:
                trades.write_text("[]\n", encoding="utf-8")
            changed += 1

        for fn in RESET_FILES:
            p = d / fn
            if p.exists():
                if args.dry_run:
                    print(f"[DRY] remove {p}")
                else:
                    p.unlink(missing_ok=True)
                changed += 1

        for pattern in RESET_GLOBS:
            for p in sorted(d.glob(pattern)):
                if p.name == "prediction_trades.json":
                    continue
                if p.name in RESET_FILES:
                    continue
                if args.dry_run:
                    print(f"[DRY] remove {p}")
                else:
                    p.unlink(missing_ok=True)
                changed += 1

        rdir = d / "reports"
        if rdir.exists() and rdir.is_dir():
            for f in rdir.iterdir():
                if f.is_file():
                    if args.dry_run:
                        print(f"[DRY] remove report {f}")
                    else:
                        f.unlink(missing_ok=True)
                    reports_cleared += 1
        elif not args.dry_run:
            rdir.mkdir(parents=True, exist_ok=True)

    summary = {
        "log_dirs": len(log_dirs),
        "changed_files": changed,
        "reports_cleared": reports_cleared,
        "dry_run": args.dry_run,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
