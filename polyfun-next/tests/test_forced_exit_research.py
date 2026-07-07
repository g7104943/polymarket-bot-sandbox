import pandas as pd

from polyfun_next.forced_exit_research import _fixed_forced_rows, _reference_rows, _choose_verdict


def _sample_df():
    return pd.DataFrame({
        "dt": pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:15:00Z"]),
        "asset": ["ETH", "ETH"],
        "timeframe": ["15m", "15m"],
        "side": ["UP", "DOWN"],
        "won_hold": [1, 0],
        "pnl__hold_to_expiry": [35.75, -35.75],
        "pnl__time_half": [10.0, -5.0],
        "pnl__time_80pct": [8.0, -4.0],
    })


def test_forced_rows_never_hold_to_settlement():
    rows = _fixed_forced_rows(_sample_df(), "ETH", "15m", "180d")
    assert rows
    assert all(r.hold_to_settlement_count == 0 for r in rows)
    assert all(not r.method.startswith("对照") for r in rows)


def test_reference_rows_are_not_live_candidates():
    rows = _reference_rows(_sample_df(), "ETH", "15m", "180d")
    assert rows[0].hold_to_settlement_count == 2
    verdict = _choose_verdict(rows)
    assert verdict["status"] == "no_live_candidate"
