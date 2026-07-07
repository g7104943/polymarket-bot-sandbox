import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polyfun_next.config import load_config
from polyfun_next.policy import CanaryPolicy
from polyfun_next.types import CandidateSignal, OrderbookQuote


class CanaryPolicyTest(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(ROOT / "config" / "canary.eth15m.example.json")
        self.policy = CanaryPolicy(self.cfg)

    def test_fixed_one_percent_size(self):
        self.assertEqual(round(self.policy.order_notional(850.0, completed_trades=0), 2), 8.50)
        self.assertEqual(round(self.policy.order_notional(850.0, completed_trades=100), 2), 8.50)
        self.assertEqual(round(self.policy.order_notional(1000.0, stage_cap=0), 2), 10.00)

    def test_reject_btc(self):
        c = CandidateSignal("BTC", "15m", "m", "c", "t", "BUY", 0.7)
        q = OrderbookQuote(0.49, 0.50, 100, 100, 10)
        self.assertIsNone(self.policy.build_order_plan(c, q, 715))

    def test_accept_safe_eth(self):
        c = CandidateSignal("ETH", "15m", "m", "c", "t", "BUY", 0.56)
        q = OrderbookQuote(0.49, 0.50, 100, 100, 13.5)
        plan = self.policy.build_order_plan(c, q, 850)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.order_type, "FAK")
        self.assertEqual(plan.cancel_after_seconds, 0)
        self.assertEqual(round(plan.notional_usd, 2), 8.50)

    def test_reject_without_model_score_threshold(self):
        c = CandidateSignal("ETH", "15m", "m", "c", "t", "BUY", 0.549)
        q = OrderbookQuote(0.49, 0.50, 100, 100, 13.5)
        self.assertIsNone(self.policy.build_order_plan(c, q, 850))

    def test_accept_at_archive_model_score_threshold(self):
        c = CandidateSignal("ETH", "15m", "m", "c", "t", "BUY", 0.551)
        q = OrderbookQuote(0.49, 0.50, 100, 100, 13.5)
        self.assertIsNotNone(self.policy.build_order_plan(c, q, 850))

    def test_reject_when_depth_cannot_cover_one_percent_notional(self):
        c = CandidateSignal("ETH", "15m", "m", "c", "t", "BUY", 0.56)
        q = OrderbookQuote(0.49, 0.50, 10.0, 10.0, 13.5)
        self.assertIsNone(self.policy.build_order_plan(c, q, 850))

    def test_absolute_spread_is_not_a_gate_anymore(self):
        c = CandidateSignal("ETH", "15m", "m", "c", "t", "BUY", 0.60)
        q = OrderbookQuote(0.40, 0.50, 100, 100, 13.5)
        self.assertIsNotNone(self.policy.build_order_plan(c, q, 850))

    def test_accept_above_reference_max_entry_price_when_price_gate_disabled(self):
        c = CandidateSignal("ETH", "15m", "m", "c", "t", "BUY", 0.60)
        q = OrderbookQuote(0.50, 0.525, 100, 100, 13.5)
        plan = self.policy.build_order_plan(c, q, 850)
        self.assertIsNotNone(plan)
        self.assertGreater(plan.price, self.cfg.max_entry_price)

    def test_accept_high_price_when_value_edge_vs_price_gate_disabled(self):
        c = CandidateSignal("ETH", "15m", "m", "c", "t", "BUY", 0.80)
        q = OrderbookQuote(0.89, 0.90, 100, 100, 13.5)
        plan = self.policy.build_order_plan(c, q, 850)
        self.assertIsNotNone(plan)
        self.assertEqual(round(plan.price, 2), 0.91)

    def test_aux_direction_gate_is_configured_off_by_default(self):
        self.assertFalse(self.cfg.aux_direction_gate_enabled)
        self.assertEqual(self.cfg.aux_gate_mode, "weighted")
        self.assertEqual(self.cfg.aux_weights["4h"], 0.45)
        self.assertFalse(self.cfg.aux_hard_veto_enabled)

    def test_reject_outside_entry_window(self):
        c = CandidateSignal("ETH", "15m", "m", "c", "t", "BUY", 0.60)
        too_early = OrderbookQuote(0.49, 0.50, 100, 100, 14.75)
        too_late = OrderbookQuote(0.49, 0.50, 100, 100, 11.75)
        self.assertIsNone(self.policy.build_order_plan(c, too_early, 850))
        self.assertIsNone(self.policy.build_order_plan(c, too_late, 850))

    def test_reject_unstable_quote_jump(self):
        c = CandidateSignal("ETH", "15m", "m", "c", "t", "BUY", 0.60)
        prev = OrderbookQuote(0.49, 0.50, 100, 100, 13.55)
        now = OrderbookQuote(0.49, 0.515, 100, 100, 13.5)
        self.assertIsNone(self.policy.build_order_plan(c, now, 850, previous_quote=prev))

    def test_take_profit_exit_disabled_by_default(self):
        q_low = OrderbookQuote(0.59, 0.60, 100, 100, 10)
        self.assertIsNone(
            self.policy.build_take_profit_exit(
                token_id="t", market_slug="m", entry_price=0.50, shares=10, quote=q_low
            )
        )
        q_hit = OrderbookQuote(0.61, 0.62, 100, 100, 10)
        self.assertIsNone(
            self.policy.build_take_profit_exit(
                token_id="t", market_slug="m", entry_price=0.50, shares=10, quote=q_hit
            )
        )

    def test_reject_late_market(self):
        c = CandidateSignal("ETH", "15m", "m", "c", "t", "BUY", 0.56)
        q = OrderbookQuote(0.49, 0.50, 100, 100, 6.5)
        self.assertIsNone(self.policy.build_order_plan(c, q, 850))


if __name__ == "__main__":
    unittest.main()
