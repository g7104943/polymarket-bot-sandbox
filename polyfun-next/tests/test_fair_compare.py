import unittest
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polyfun_next.fair_compare import _apply_cancel_5m, _metrics, _max_drawdown


class FairCompareTest(unittest.TestCase):
    def test_cancel_5m_keeps_early_fill_and_cancels_late(self):
        df = pd.DataFrame([
            {"trace_fill_ts": 100, "trace_order_submitted_ts": 0, "trace_fill_fraction": 1.0, "trace_fill_price": 0.5, "baseline_hold_pnl": 1},
            {"trace_fill_ts": 301, "trace_order_submitted_ts": 0, "trace_fill_fraction": 1.0, "trace_fill_price": 0.5, "baseline_hold_pnl": -1},
        ])
        out = _apply_cancel_5m(df)
        self.assertEqual(out.loc[0, "sim_fill_fraction"], 1.0)
        self.assertEqual(out.loc[1, "sim_fill_fraction"], 0.0)

    def test_metrics_winner_loser_fill_rates(self):
        df = pd.DataFrame([
            {"sim_fill_fraction": 0.0, "sim_pnl": 0.0, "baseline_hold_pnl": 1, "market_slug": "w0"},
            {"sim_fill_fraction": 1.0, "sim_pnl": 5.0, "baseline_hold_pnl": 1, "market_slug": "w1"},
            {"sim_fill_fraction": 1.0, "sim_pnl": -5.0, "baseline_hold_pnl": -1, "market_slug": "l1"},
        ])
        m = _metrics("x", "test", df)
        self.assertEqual(m.wins, 1)
        self.assertEqual(m.losses, 1)
        self.assertAlmostEqual(m.winner_fill_rate_pct, 50.0)
        self.assertAlmostEqual(m.loser_fill_rate_pct, 100.0)

    def test_max_drawdown(self):
        self.assertEqual(_max_drawdown(pd.Series([10, -3, -4, 2])), 7.0)


if __name__ == "__main__":
    unittest.main()
