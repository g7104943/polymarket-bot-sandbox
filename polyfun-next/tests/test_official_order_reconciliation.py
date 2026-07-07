import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polyfun_next.official import ClobV2SdkOfficialClient, _extract_order_id_from_exception
from polyfun_next.types import OfficialOrderStatus, OrderTruth


class _DummyOfficial:
    def __init__(self, follow_up: OfficialOrderStatus):
        self.follow_up = follow_up

    def get_order(self, order_id: str) -> OfficialOrderStatus:
        return self.follow_up


class OfficialOrderReconciliationTest(unittest.TestCase):
    def test_extract_order_id_from_v2_exception_payload(self):
        exc = RuntimeError(
            "ApiException[400] {'errorMsg': 'no orders found to match with FAK order', "
            "'orderID': '0xcc5d245abc1234567890abcdefabcdefabcdefabcdef'}"
        )
        self.assertEqual(
            _extract_order_id_from_exception(exc),
            "0xcc5d245abc1234567890abcdefabcdefabcdefabcdef",
        )

    def test_post_exception_with_matched_order_id_is_not_rejected(self):
        order_id = "0xabababababababababababababababab"
        exc = RuntimeError(f"ApiException[400] {{'orderID': '{order_id}'}}")
        dummy = _DummyOfficial(
            OfficialOrderStatus(
                order_id=order_id,
                truth=OrderTruth.OFFICIAL_FILLED,
                raw={"status": "MATCHED", "matchedShares": "1.0"},
                matched_shares=1.0,
                message="official matched shares > 0",
            )
        )

        status = ClobV2SdkOfficialClient._reconcile_post_exception(
            dummy, exc, order_type="FAK", side="buy", plan={"notional_usd": 1.0}
        )

        self.assertIsNotNone(status)
        self.assertEqual(status.order_id, order_id)
        self.assertEqual(status.truth, OrderTruth.OFFICIAL_FILLED)
        self.assertEqual(status.matched_shares, 1.0)


if __name__ == "__main__":
    unittest.main()
