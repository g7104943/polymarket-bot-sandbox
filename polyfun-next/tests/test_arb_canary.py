from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polyfun_next.arb_canary import (
    ArbCanaryConfig,
    ArbExecutionEngine,
    ArbMarket,
    BookLevel,
    MarketOutcome,
    OrderBook,
    find_arbitrage_opportunity,
)
from polyfun_next.ledger import JsonlLedger
from polyfun_next.official import DryRunOfficialClient
from polyfun_next.types import OrderTruth


def cfg(**kwargs):
    base = dict(
        system_name="test",
        live_enabled=False,
        clob_host="https://clob.polymarket.com",
        gamma_host="https://gamma-api.polymarket.com",
        pusd_address="0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB",
        max_total_cost_usd=5.0,
        target_payout_usd=5.0,
        min_edge_usd=0.03,
        min_edge_pct=0.005,
        min_liquidity_usd=1000.0,
        min_volume_usd=100.0,
        min_order_shares=5.0,
        max_outcomes=20,
        scan_limit=10,
        order_type="FOK",
        allow_non_atomic_live_execution=False,
        official_truth_required=True,
    )
    base.update(kwargs)
    c = ArbCanaryConfig(**base)
    c.validate()
    return c


def market(outcomes=("YES", "NO")):
    return ArbMarket(
        market_id="m1",
        condition_id="0xabc",
        slug="test-market",
        question="test?",
        liquidity=10000,
        volume=10000,
        outcomes=[MarketOutcome(f"token{idx}", outcome) for idx, outcome in enumerate(outcomes)],
    )


def book(token_id, ask_price, ask_size=10):
    return OrderBook(token_id, asks=[BookLevel(ask_price, ask_size)], bids=[])


class ArbCanaryTest(unittest.TestCase):
    def test_binary_complete_set_opportunity(self):
        m = market()
        books = {"token0": book("token0", 0.49), "token1": book("token1", 0.49)}
        opp = find_arbitrage_opportunity(m, books, cfg())
        self.assertIsNotNone(opp)
        self.assertAlmostEqual(opp.total_cost_usd, 4.9)
        self.assertAlmostEqual(opp.edge_usd, 0.1)

    def test_rejects_negative_edge(self):
        m = market()
        books = {"token0": book("token0", 0.51), "token1": book("token1", 0.51)}
        self.assertIsNone(find_arbitrage_opportunity(m, books, cfg()))

    def test_rejects_insufficient_depth(self):
        m = market()
        books = {"token0": book("token0", 0.49, 4.9), "token1": book("token1", 0.49, 10)}
        self.assertIsNone(find_arbitrage_opportunity(m, books, cfg()))

    def test_multi_outcome_opportunity(self):
        m = market(("A", "B", "C"))
        books = {
            "token0": book("token0", 0.30),
            "token1": book("token1", 0.30),
            "token2": book("token2", 0.30),
        }
        opp = find_arbitrage_opportunity(m, books, cfg())
        self.assertIsNotNone(opp)
        self.assertAlmostEqual(opp.total_cost_usd, 4.5)
        self.assertEqual(len(opp.legs), 3)

    def test_dry_run_execution_never_posts_live(self):
        m = market()
        books = {"token0": book("token0", 0.49), "token1": book("token1", 0.49)}
        opp = find_arbitrage_opportunity(m, books, cfg())
        assert opp is not None
        engine = ArbExecutionEngine(
            cfg(),
            DryRunOfficialClient(),
            JsonlLedger(ROOT / "runtime" / "test_arb_canary.jsonl"),
        )
        statuses = engine.execute(opp, dry_run=True)
        self.assertEqual(len(statuses), 2)
        self.assertTrue(all(s.truth == OrderTruth.DRY_RUN for s in statuses))


if __name__ == "__main__":
    unittest.main()
