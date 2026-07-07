from __future__ import annotations

from datetime import datetime, timedelta, timezone

from polyfun_next.risk_v2 import OfficialTrade, RiskV2Policy, simulate_official_risk_policy


def _trade(i: int, won: bool, *, price: float = 0.55, score: float = 0.56) -> OfficialTrade:
    dt = datetime(2026, 5, 1, tzinfo=timezone.utc) + timedelta(minutes=15 * i)
    cost = 10.0
    shares = cost / price
    return OfficialTrade(
        market_slug=f"eth-updown-15m-{1777600000 + i * 900}",
        dt=dt,
        first_ts=dt.isoformat(),
        order_ids=(f"order-{i}",),
        side="UP",
        won=won,
        actual_cost=cost,
        actual_shares=shares,
        actual_avg_price=price,
        actual_pnl=(shares - cost) if won else -cost,
        model_score=score,
        shock_candidate_id="06173b68b0d86431",
        strategy_profile="new_shock_filter_top159_cluster_06173",
    )


def test_loss_streak_policy_pauses_after_three_settled_losses():
    rows = [_trade(0, False), _trade(1, False), _trade(2, False), _trade(3, False), _trade(4, True)]
    policy = RiskV2Policy(name="loss3", loss_streak_n=3, loss_pause_bars=4, update_lag_bars=0)
    result = simulate_official_risk_policy(rows, policy, initial_funds=100.0)
    assert result["keptTrades"] == 3
    assert result["skippedTrades"] == 2
    assert result["interceptedLosers"] == 1
    assert result["interceptedWinners"] == 1
    assert result["pauseCount"] >= 1


def test_execution_failure_is_not_part_of_trade_loss_policy():
    rows = [_trade(0, True), _trade(1, True)]
    policy = RiskV2Policy(name="loss3", loss_streak_n=3, loss_pause_bars=4, update_lag_bars=0)
    result = simulate_official_risk_policy(rows, policy, initial_funds=100.0)
    assert result["keptTrades"] == 2
    assert result["skippedTrades"] == 0


def test_pending_result_lag_delays_loss_streak_pause():
    rows = [_trade(0, False), _trade(1, False), _trade(2, False), _trade(3, False)]
    immediate = simulate_official_risk_policy(rows, RiskV2Policy(name="loss3", loss_streak_n=3, loss_pause_bars=4, update_lag_bars=0), initial_funds=100.0)
    delayed = simulate_official_risk_policy(rows, RiskV2Policy(name="loss3lag4", loss_streak_n=3, loss_pause_bars=4, update_lag_bars=4), initial_funds=100.0)
    assert immediate["skippedTrades"] > delayed["skippedTrades"]
    assert delayed["keptTrades"] == 4


def test_high_price_loss_guard_blocks_high_price_loss_cluster():
    rows = [_trade(0, False, price=0.66), _trade(1, False, price=0.67), _trade(2, False, price=0.50)]
    policy = RiskV2Policy(name="highprice", high_price_min=0.65, high_price_loss_streak_n=2, high_price_pause_bars=4, update_lag_bars=0)
    result = simulate_official_risk_policy(rows, policy, initial_funds=100.0)
    assert result["keptTrades"] == 2
    assert result["skippedTrades"] == 1
    assert result["interceptedLosers"] == 1


def test_day_drawdown_policy_uses_realized_official_pnl():
    rows = [_trade(0, False, price=0.55), _trade(1, False, price=0.55), _trade(2, True, price=0.55)]
    policy = RiskV2Policy(name="daydd", day_drawdown_fraction=0.10, day_pause_bars=8, update_lag_bars=0)
    result = simulate_official_risk_policy(rows, policy, initial_funds=100.0)
    assert result["keptTrades"] == 1
    assert result["skippedTrades"] == 2


def test_state_machine_rolling_pnl_triggers_yellow_pause():
    rows = [
        _trade(0, False, price=0.55),
        _trade(1, False, price=0.55),
        _trade(2, False, price=0.55),
        _trade(3, True, price=0.55),
    ]
    policy = RiskV2Policy(
        name="sm",
        family="state_machine",
        state_yellow_window=2,
        state_yellow_min_pnl=-15.0,
        state_yellow_pause_bars=4,
        update_lag_bars=0,
    )
    result = simulate_official_risk_policy(rows, policy, initial_funds=100.0)
    assert result["keptTrades"] == 2
    assert result["skippedTrades"] == 2
    assert result["skipReasons"]


def test_state_machine_expired_pause_does_not_extend_without_new_settlement():
    rows = [
        _trade(0, False, price=0.55),
        _trade(1, False, price=0.55),
        _trade(2, False, price=0.55),
        _trade(10, True, price=0.55),
    ]
    policy = RiskV2Policy(
        name="sm_loss",
        family="state_machine",
        state_red_loss_streak_n=3,
        state_red_pause_bars=4,
        update_lag_bars=0,
    )
    result = simulate_official_risk_policy(rows, policy, initial_funds=100.0)
    assert result["keptTrades"] == 4
    assert result["skippedTrades"] == 0


def test_state_machine_high_price_loss_uses_state_price_threshold():
    rows = [
        _trade(0, False, price=0.66),
        _trade(1, False, price=0.67),
        _trade(2, True, price=0.50),
    ]
    policy = RiskV2Policy(
        name="sm_hp",
        family="state_machine",
        state_red_high_price_min=0.65,
        state_red_high_price_loss_streak_n=2,
        state_red_pause_bars=4,
        update_lag_bars=0,
    )
    result = simulate_official_risk_policy(rows, policy, initial_funds=100.0)
    assert result["keptTrades"] == 2
    assert result["skippedTrades"] == 1
    assert result["interceptedWinners"] == 1


def test_state_machine_score_price_loss_requires_both_price_and_score():
    rows = [
        _trade(0, False, price=0.62, score=0.57),
        _trade(1, False, price=0.62, score=0.57),
        _trade(2, False, price=0.50, score=0.57),
        _trade(3, False, price=0.62, score=0.61),
        _trade(4, True, price=0.62, score=0.57),
    ]
    policy = RiskV2Policy(
        name="sm_score_price",
        family="state_machine",
        state_red_score_price_min=0.60,
        state_red_score_price_max=0.58,
        state_red_score_price_loss_streak_n=2,
        state_red_pause_bars=4,
        update_lag_bars=0,
    )
    result = simulate_official_risk_policy(rows, policy, initial_funds=100.0)
    assert result["keptTrades"] == 2
    assert result["skippedTrades"] == 3
    assert result["interceptedLosers"] == 2
    assert result["interceptedWinners"] == 1
