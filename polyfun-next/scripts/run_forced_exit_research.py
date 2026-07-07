#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polyfun_next.forced_exit_research import run_forced_exit_research


def main() -> int:
    ap = argparse.ArgumentParser(description="Run strict forced-exit research for ETH/BTC 15m/1h/4h.")
    ap.add_argument("--cache", default="/Users/mac/polyfun/reports/cache")
    ap.add_argument("--reports", default="/Users/mac/polyfun/reports")
    ap.add_argument("--engine", choices=["lightgbm", "catboost"], default="lightgbm")
    args = ap.parse_args()
    payload = run_forced_exit_research(cache_dir=args.cache, reports_dir=args.reports, engine=args.engine)
    print(Path(args.reports) / "forced_exit_15m_1h_4h_absolute_compare_latest.md")
    print(payload["uniqueVerdict"]["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
