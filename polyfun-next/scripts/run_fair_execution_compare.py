#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polyfun_next.fair_compare import compare_methods, load_episodes, write_outputs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eth-episodes", default="/Users/mac/polyfun/data/processed/vnext_entry_exit_episodes_eth_usdt.parquet")
    ap.add_argument("--btc-episodes", default="/Users/mac/polyfun/data/processed/vnext_entry_exit_episodes_btc_usdt.parquet")
    ap.add_argument("--reports", default="/Users/mac/polyfun/reports")
    args = ap.parse_args()

    all_metrics = []
    for label, path in [("ETH", args.eth_episodes), ("BTC", args.btc_episodes)]:
        df = load_episodes(path)
        metrics = compare_methods(df, ["180d", "365d", "all"])
        for m in metrics:
            object.__setattr__(m, "name", f"{label}_{m.name}")
        all_metrics.extend(metrics)
    write_outputs(args.reports, all_metrics)
    print(Path(args.reports) / "fair_execution_compare_latest.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
