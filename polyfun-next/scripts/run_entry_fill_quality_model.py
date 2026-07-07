#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polyfun_next.quality_model import (
    compare_quality_methods,
    load_quality_frame,
    run_walk_forward_quality,
    write_quality_outputs,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="/Users/mac/polyfun/data/processed/vnext_entry_exit_episodes_eth_usdt.parquet")
    ap.add_argument("--candidate-stream", default="/Users/mac/polyfun/polyfun-next/runtime/eth15m_5y_candidate_stream.jsonl")
    ap.add_argument("--features", default="/Users/mac/polyfun/data/processed/vnext_profit_relabel_eth_usdt_v2.parquet")
    ap.add_argument("--reports", default="/Users/mac/polyfun/reports")
    ap.add_argument("--feature-mode", default="strict", choices=["strict", "microstructure"])
    args = ap.parse_args()

    df, features, audit = load_quality_frame(args.episodes, args.candidate_stream, args.features, feature_mode=args.feature_mode)
    scored, blocks = run_walk_forward_quality(df, features)
    metrics = compare_quality_methods(scored, ["180d", "365d", "all"])
    write_quality_outputs(args.reports, metrics, audit, blocks, scored)
    print(Path(args.reports) / "polyfun_next_entry_fill_quality_model_latest.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
