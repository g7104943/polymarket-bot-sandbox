#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polyfun_next.value_system import run_value_system


def main() -> int:
    ap = argparse.ArgumentParser(description="Rebuild ETH 15m value-first strategy research.")
    ap.add_argument("--episodes", default="/Users/mac/polyfun/data/processed/vnext_entry_exit_episodes_eth_usdt.parquet")
    ap.add_argument("--candidate-stream", default="/Users/mac/polyfun/polyfun-next/runtime/eth15m_5y_candidate_stream.jsonl")
    ap.add_argument("--features", default="/Users/mac/polyfun/data/processed/vnext_profit_relabel_eth_usdt_v2.parquet")
    ap.add_argument("--reports", default="/Users/mac/polyfun/reports")
    args = ap.parse_args()
    payload = run_value_system(
        episode_path=args.episodes,
        candidate_path=args.candidate_stream,
        feature_path=args.features,
        reports_dir=args.reports,
    )
    print(Path(args.reports) / "strategy_value_system_absolute_compare_latest.md")
    print(payload["uniqueVerdict"]["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

