import unittest
from pathlib import Path
import sys
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polyfun_next.constants import LIVE_ACK_ENV, LIVE_ACK_VALUE
from polyfun_next.config import load_config
from polyfun_next.canary_state import CanaryState, load_state, opened_position, save_state
from polyfun_next.execution import ExecutionEngine
from polyfun_next.ledger import JsonlLedger
from polyfun_next.official import DryRunOfficialClient
from polyfun_next.policy import CanaryPolicy
from polyfun_next.types import CandidateSignal, OrderbookQuote, OrderTruth
from polyfun_next.types import OfficialOrderStatus


class _RejectedWithOrderIdOfficialClient(DryRunOfficialClient):
    def __init__(self):
        self.get_order_called = False

    def post_buy_order_with_type(self, plan, order_type: str):
        return OfficialOrderStatus(
            order_id="0xabc123",
            truth=OrderTruth.OFFICIAL_REJECTED,
            raw={"post_exception": "no orders found to match with FAK order"},
            matched_shares=0.0,
            message="postOrder raised with orderID, but no active/matched order exists",
        )

    def get_order(self, order_id: str):
        self.get_order_called = True
        return OfficialOrderStatus(
            order_id=order_id,
            truth=OrderTruth.OFFICIAL_MISSING,
            raw={},
            matched_shares=0.0,
            message="empty getOrder",
        )


class ExecutionGuardTest(unittest.TestCase):
    def test_dry_run_never_submits(self):
        cfg = load_config(ROOT / "config" / "canary.eth15m.example.json")
        c = CandidateSignal("ETH", "15m", "m", "c", "t", "BUY", 0.56)
        q = OrderbookQuote(0.49, 0.50, 100, 100, 13.5)
        plan = CanaryPolicy(cfg).build_order_plan(c, q, 850)
        engine = ExecutionEngine(cfg, DryRunOfficialClient(), JsonlLedger(ROOT / "runtime" / "test.jsonl"))
        status = engine.submit(plan, dry_run=True)
        self.assertEqual(status.truth, OrderTruth.DRY_RUN)
        self.assertIsNone(status.order_id)

    def test_dry_run_take_profit_never_sells(self):
        cfg = load_config(ROOT / "config" / "canary.eth15m.example.json")
        q = OrderbookQuote(0.61, 0.62, 100, 100, 10)
        exit_plan = CanaryPolicy(cfg).build_take_profit_exit(
            token_id="t", market_slug="m", entry_price=0.50, shares=10, quote=q
        )
        self.assertIsNone(exit_plan)

    def test_fak_rejection_with_order_id_is_not_overwritten_by_missing_get_order(self):
        cfg = load_config(ROOT / "config" / "canary.eth15m.example.json")
        cfg = cfg.__class__(**{**cfg.__dict__, "live_enabled": True})
        c = CandidateSignal("ETH", "15m", "m", "c", "t", "BUY", 0.56)
        q = OrderbookQuote(0.49, 0.50, 100, 100, 13.5)
        plan = CanaryPolicy(cfg).build_order_plan(c, q, 850)
        official = _RejectedWithOrderIdOfficialClient()
        engine = ExecutionEngine(cfg, official, JsonlLedger(ROOT / "runtime" / "test.jsonl"))

        old_ack = os.environ.get(LIVE_ACK_ENV)
        os.environ[LIVE_ACK_ENV] = LIVE_ACK_VALUE
        try:
            status = engine.submit(plan, dry_run=False)
        finally:
            if old_ack is None:
                os.environ.pop(LIVE_ACK_ENV, None)
            else:
                os.environ[LIVE_ACK_ENV] = old_ack

        self.assertEqual(status.truth, OrderTruth.OFFICIAL_REJECTED)
        self.assertFalse(official.get_order_called)

    def test_state_round_trip(self):
        p = ROOT / "runtime" / "test_state.json"
        pos = opened_position(token_id="t", market_slug="m", entry_price=0.5, shares=10, target_price=0.6, buy_order_id="o")
        save_state(p, CanaryState(open_position=pos, consecutive_anomalies=1))
        loaded = load_state(p)
        self.assertEqual(loaded.open_position.token_id, "t")
        self.assertEqual(loaded.consecutive_anomalies, 1)


if __name__ == "__main__":
    unittest.main()
