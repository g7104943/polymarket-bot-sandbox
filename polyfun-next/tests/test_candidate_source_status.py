import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polyfun_next.candidate_source import CandidateSourceError, JsonlCandidateSource
from polyfun_next.official import _quote_from_orderbook, _status_from_raw_order
from polyfun_next.status import ledger_summary
from polyfun_next.types import OrderTruth


class CandidateAndStatusTest(unittest.TestCase):
    def test_candidate_source_requires_eth15m(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "candidates.jsonl"
            p.write_text(json.dumps({
                "symbol": "ETH",
                "period": "15m",
                "market_slug": "m",
                "condition_id": "c",
                "token_id": "t",
                "side": "BUY",
                "model_score": 0.56,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }) + "\n")
            self.assertEqual(JsonlCandidateSource(p).latest().symbol, "ETH")

    def test_candidate_source_rejects_btc(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "candidates.jsonl"
            p.write_text(json.dumps({
                "symbol": "BTC",
                "period": "15m",
                "market_slug": "m",
                "condition_id": "c",
                "token_id": "t",
                "side": "BUY",
                "model_score": 0.56,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }) + "\n")
            with self.assertRaises(CandidateSourceError):
                JsonlCandidateSource(p).latest()

    def test_candidate_source_preserves_live_model_metadata(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "candidates.jsonl"
            p.write_text(json.dumps({
                "symbol": "ETH",
                "period": "15m",
                "market_slug": "m",
                "condition_id": "c",
                "token_id": "t",
                "side": "UP",
                "model_score": 0.56,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source": "integrated_main_lightgbm_full_base15_edge0.05_archive_rerank_v1",
                "selected_candidate": "integrated_lightgbm_full_base15_edge0.05_6c54c8a7b1d2593a",
                "train_window": "full",
                "feature_mode": "base15",
                "edge": 0.05,
            }) + "\n")
            latest = JsonlCandidateSource(p).latest()
            self.assertEqual(latest.live_model_profile, "integrated_main_lightgbm_full_base15_edge0.05_archive_rerank_v1")
            self.assertEqual(latest.selected_candidate, "integrated_lightgbm_full_base15_edge0.05_6c54c8a7b1d2593a")
            self.assertEqual(latest.train_window, "full")
            self.assertEqual(latest.feature_mode, "base15")
            self.assertEqual(latest.edge, 0.05)

    def test_status_mapping(self):
        self.assertEqual(
            _status_from_raw_order("abc", {"status": "CANCELED", "matchedShares": "0"}).truth,
            OrderTruth.OFFICIAL_CANCELED_ZERO_FILL,
        )
        self.assertEqual(
            _status_from_raw_order("abc", {"status": "OPEN", "matchedShares": "0"}).truth,
            OrderTruth.OFFICIAL_OPEN,
        )
        self.assertEqual(
            _status_from_raw_order("abc", {"status": "OPEN", "matchedShares": "1.2"}).truth,
            OrderTruth.OFFICIAL_FILLED,
        )

    def test_ledger_summary_counts_missing(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ledger.jsonl"
            p.write_text(json.dumps({"payload": {"status": {"truth": "official_missing"}}}) + "\n")
            self.assertEqual(ledger_summary(p)["official_missing"], 1)

    def test_quote_from_orderbook_picks_best_levels(self):
        q = _quote_from_orderbook(
            {
                "bids": [{"price": "0.49", "size": "10"}, {"price": "0.51", "size": "7"}],
                "asks": [{"price": "0.54", "size": "8"}, {"price": "0.53", "size": "9"}],
            },
            minutes_remaining=10,
        )
        self.assertEqual(q.best_bid, 0.51)
        self.assertEqual(q.bid_depth_shares, 7)
        self.assertEqual(q.best_ask, 0.53)
        self.assertEqual(q.ask_depth_shares, 9)


if __name__ == "__main__":
    unittest.main()
