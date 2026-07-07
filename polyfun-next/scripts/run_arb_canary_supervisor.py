#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polyfun_next.arb_canary import load_arb_config, scan_orderbook_arbitrage, to_jsonable
from polyfun_next.ledger import JsonlLedger


def main() -> int:
    ap = argparse.ArgumentParser(description="Periodic read-only arb canary scanner")
    ap.add_argument("--config", required=True)
    ap.add_argument("--duration-seconds", type=int, default=24 * 3600)
    ap.add_argument("--interval-seconds", type=int, default=60)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    cfg = load_arb_config(args.config)
    ledger = JsonlLedger(ROOT / "runtime" / "arb_canary_supervisor.jsonl")
    deadline = time.time() + args.duration_seconds
    loops = 0
    best_edge = 0.0
    while time.time() < deadline:
        started = time.time()
        try:
            report = scan_orderbook_arbitrage(cfg, market_limit=args.limit)
            best_edge = max(best_edge, report.opportunities[0].edge_usd if report.opportunities else 0.0)
            ledger.append(
                "arb_scan_tick",
                {
                    "markets_seen": report.markets_seen,
                    "markets_eligible": report.markets_eligible,
                    "markets_with_books": report.markets_with_books,
                    "opportunities": [to_jsonable(opp) for opp in report.opportunities[:5]],
                    "error_count": len(report.errors),
                },
            )
            print(
                json.dumps(
                    {
                        "tick": loops,
                        "markets_seen": report.markets_seen,
                        "opportunities": len(report.opportunities),
                        "best_edge": report.opportunities[0].edge_usd if report.opportunities else 0.0,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        except Exception as exc:
            ledger.append("arb_scan_error", {"error": repr(exc)})
            print(json.dumps({"tick": loops, "error": repr(exc)}, ensure_ascii=False), flush=True)
        loops += 1
        sleep_for = max(0.0, args.interval_seconds - (time.time() - started))
        time.sleep(sleep_for)
    ledger.append("arb_supervisor_finished", {"loops": loops, "best_edge": best_edge})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
