#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polyfun_next.arb_canary import ArbExecutionEngine, load_arb_config, scan_orderbook_arbitrage
from polyfun_next.ledger import JsonlLedger
from polyfun_next.official import ClobV2SdkOfficialClient, DryRunOfficialClient


def main() -> int:
    ap = argparse.ArgumentParser(description="Dry-run or guarded live execution for the best arb canary opportunity")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=50)
    args = ap.parse_args()

    cfg = load_arb_config(args.config)
    report = scan_orderbook_arbitrage(cfg, market_limit=args.limit)
    if not report.opportunities:
        print(json.dumps({"status": "no_opportunity", "markets_seen": report.markets_seen}, indent=2))
        return 2
    official = DryRunOfficialClient() if args.dry_run or not cfg.live_enabled else ClobV2SdkOfficialClient(cfg)  # type: ignore[arg-type]
    engine = ArbExecutionEngine(cfg, official, JsonlLedger(ROOT / "runtime" / "arb_canary_ledger.jsonl"))
    statuses = engine.execute(report.opportunities[0], dry_run=args.dry_run)
    print(json.dumps([s.__dict__ for s in statuses], indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
