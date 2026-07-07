import pandas as pd

from polyfun_next.realistic_backtest import _simulate, compare_realistic_methods


def test_limit_order_winner_not_filled_when_price_never_touches_limit():
    df = pd.DataFrame([
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "event_time": pd.Timestamp("2026-01-01T00:00:00Z"),
            "direction_target": 1,
            "actual_up": 1,
            "won_if_filled": True,
            "path_t_sec_json": [0, 60, 120, 300],
            "path_up_json": [0.50, 0.55, 0.60, 0.70],
            "market_slug": "m1",
            "best_action_key": "UP|0.49",
        }
    ])
    rows = compare_realistic_methods(df, ["all"])
    by_name = {r.method: r for r in rows}
    assert by_name["直接吃卖价_立即成交"].fills == 1
    assert by_name["挂便宜1分_5分钟取消"].fills == 0


def test_limit_order_loser_filled_when_price_moves_down():
    df = pd.DataFrame([
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "event_time": pd.Timestamp("2026-01-01T00:00:00Z"),
            "direction_target": 1,
            "actual_up": 0,
            "won_if_filled": False,
            "path_t_sec_json": [0, 60, 120, 300],
            "path_up_json": [0.50, 0.49, 0.48, 0.40],
            "market_slug": "m2",
            "best_action_key": "UP|0.49",
        }
    ])
    rows = compare_realistic_methods(df, ["all"])
    by_name = {r.method: r for r in rows}
    assert by_name["挂便宜1分_5分钟取消"].fills == 1
    assert by_name["挂便宜1分_5分钟取消"].losses == 1
    assert by_name["挂便宜1分_5分钟取消"].pnl < 0


def test_candidate_side_overrides_leaky_direction_target():
    df = pd.DataFrame([
        {
            "actual_up": 0,
            "direction_target": 1,  # episode field says UP, candidate stream says DOWN.
            "candidate_side": "DOWN",
            "path_t_sec_json": [0, 60],
            "path_up_json": [0.50, 0.40],
        }
    ])
    out = _simulate(df, {"kind": "taker", "limit_offset": 0.0})
    assert out.loc[0, "sim_pnl"] > 0


def test_quality_model_no_future_label_features():
    from polyfun_next.quality_model import _select_feature_columns
    df = pd.DataFrame({
        "safe_return_1": [0.1],
        "actual_up": [1],
        "next_return": [0.2],
        "best_action_key": ["UP"],
        "path_up_json": ["[]"],
        "logit_p": [0.55],
        "ob_spread_ratio_mean": [0.01],
    })
    cols = _select_feature_columns(df, "strict")
    assert "safe_return_1" in cols
    assert "logit_p" in cols
    assert "actual_up" not in cols
    assert "next_return" not in cols
    assert "best_action_key" not in cols
    assert "path_up_json" not in cols
    assert "ob_spread_ratio_mean" not in cols
