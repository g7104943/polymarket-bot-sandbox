#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polyfun_next.candidate_source import JsonlCandidateSource
from polyfun_next.config import load_config
from polyfun_next.execution import ExecutionEngine
from polyfun_next.ledger import JsonlLedger
from polyfun_next.official import ClobV2SdkOfficialClient, DryRunOfficialClient
from polyfun_next.policy import CanaryPolicy
from polyfun_next.types import CandidateSignal, OrderbookQuote


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--candidate-jsonl")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--current-funds", type=float, default=850.0)
    # Temporary quote inputs until a dedicated orderbook adapter is enabled.
    ap.add_argument("--best-bid", type=float, default=0.48)
    ap.add_argument("--best-ask", type=float, default=0.50)
    ap.add_argument("--ask-depth-shares", type=float, default=100.0)
    ap.add_argument("--bid-depth-shares", type=float, default=100.0)
    ap.add_argument("--minutes-remaining", type=float, default=13.5)
    ap.add_argument("--completed-trades", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    policy = CanaryPolicy(cfg)
    candidate = _candidate(args.candidate_jsonl)
    quote = OrderbookQuote(
        best_bid=args.best_bid,
        best_ask=args.best_ask,
        bid_depth_shares=args.bid_depth_shares,
        ask_depth_shares=args.ask_depth_shares,
        minutes_remaining=args.minutes_remaining,
    )
    plan = policy.build_order_plan(
        candidate,
        quote,
        current_funds_usd=args.current_funds,
        completed_trades=args.completed_trades,
    )
    if plan is None:
        print("candidate rejected by canary policy")
        return 2
    official = DryRunOfficialClient() if args.dry_run or not cfg.live_enabled else ClobV2SdkOfficialClient(cfg)
    engine = ExecutionEngine(
        cfg,
        official=official,
        ledger=JsonlLedger(ROOT / "runtime" / "canary_ledger.jsonl"),
    )
    status = engine.submit(plan, dry_run=args.dry_run)
    print(status)
    return 0


def _candidate(path: str | None) -> CandidateSignal:
    if path:
        found = JsonlCandidateSource(path).latest()
        if found is None:
            raise RuntimeError(f"no candidate in {path}")
        return found
    # Deterministic smoke candidate; not a real model output.
    return CandidateSignal(
        symbol="ETH",
        period="15m",
        market_slug="dry-run-eth-updown-15m",
        condition_id="0xDRYRUN",
        token_id="0",
        side="BUY",
        model_score=0.56,
    )


if __name__ == "__main__":
    raise SystemExit(main())
