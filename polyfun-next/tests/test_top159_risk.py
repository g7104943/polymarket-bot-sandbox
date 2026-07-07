from __future__ import annotations

from datetime import datetime, timedelta, timezone

from polyfun_next.canary_state import CanaryState, load_state
from polyfun_next.config import load_config
from polyfun_next.policy import CanaryPolicy
from polyfun_next.risk import evaluate_top159_risk, state_with_current_risk_day


def _cfg():
    return load_config("config/canary.eth15m.example.json")


def test_daily_high_water_drawdown_pauses_at_twenty_percent():
    cfg = _cfg()
    state = CanaryState(open_position=None, day_start_funds_usd=900.0, day_high_water_funds_usd=1000.0)
    decision = evaluate_top159_risk(cfg, state, current_funds_usd=800.0)
    assert not decision.allowed
    assert "daily_loss_stop" in decision.reasons
    assert decision.metrics
    assert decision.metrics["day_high_water_funds_usd"] == 1000.0
    assert decision.metrics["daily_loss_trigger_funds_usd"] == 800.0


def test_daily_high_water_drawdown_does_not_pause_at_19_9_percent():
    cfg = _cfg()
    state = CanaryState(open_position=None, day_start_funds_usd=900.0, day_high_water_funds_usd=1000.0)
    decision = evaluate_top159_risk(cfg, state, current_funds_usd=801.0)
    assert decision.allowed
    assert decision.reasons == []


def test_active_daily_pause_blocks_until_expiry():
    cfg = _cfg()
    now = datetime(2026, 5, 1, tzinfo=timezone.utc)
    state = CanaryState(open_position=None, risk_pause_until=(now + timedelta(hours=1)).isoformat())
    decision = evaluate_top159_risk(cfg, state, current_funds_usd=850.0, now=now)
    assert not decision.allowed
    assert "daily_loss_pause_active" in decision.reasons


def test_expired_daily_pause_resets_day_high_water_and_unblocks():
    now = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    state = CanaryState(
        open_position=None,
        day_key="2026-05-01",
        day_start_funds_usd=1000.0,
        day_high_water_funds_usd=1000.0,
        current_funds_usd=780.0,
        risk_pause_until=(now - timedelta(minutes=1)).isoformat(),
    )
    updated = state_with_current_risk_day(state, current_funds_usd=780.0, now=now)
    assert updated.risk_pause_until is None
    assert updated.day_start_funds_usd == 780.0
    assert updated.day_high_water_funds_usd == 780.0
    decision = evaluate_top159_risk(_cfg(), updated, current_funds_usd=780.0, now=now)
    assert decision.allowed


def test_failure_pause_after_five_execution_failures():
    cfg = _cfg()
    state = CanaryState(open_position=None, consecutive_fak_failures=5, consecutive_fok_failures=5)
    decision = evaluate_top159_risk(cfg, state, current_funds_usd=850.0)
    assert not decision.allowed
    assert "execution_failure_pause" in decision.reasons


def test_expired_failure_pause_does_not_block():
    cfg = _cfg()
    now = datetime(2026, 5, 1, tzinfo=timezone.utc)
    state = CanaryState(
        open_position=None,
        consecutive_fak_failures=0,
        failure_pause_until=(now - timedelta(seconds=1)).isoformat(),
    )
    decision = evaluate_top159_risk(cfg, state, current_funds_usd=850.0, now=now)
    assert decision.allowed


def test_expired_failure_pause_clears_old_execution_failures():
    now = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    state = CanaryState(
        open_position=None,
        consecutive_fak_failures=5,
        consecutive_fok_failures=5,
        failure_pause_until=(now - timedelta(minutes=1)).isoformat(),
    )
    updated = state_with_current_risk_day(state, current_funds_usd=850.0, now=now)
    assert updated.failure_pause_until is None
    assert updated.consecutive_fak_failures == 0
    assert updated.consecutive_fok_failures == 0
    decision = evaluate_top159_risk(_cfg(), updated, current_funds_usd=850.0, now=now)
    assert decision.allowed


def test_legacy_fok_failure_counter_still_blocks():
    cfg = _cfg()
    state = CanaryState(open_position=None, consecutive_fok_failures=5)
    decision = evaluate_top159_risk(cfg, state, current_funds_usd=850.0)
    assert not decision.allowed
    assert decision.metrics
    assert decision.metrics["consecutive_fak_failures"] == 5


def test_legacy_state_file_maps_fok_counter_to_fak_counter(tmp_path):
    path = tmp_path / "state.json"
    path.write_text('{"open_position": null, "consecutive_fok_failures": 4}', encoding="utf-8")
    state = load_state(path)
    assert state.consecutive_fak_failures == 4
    assert state.consecutive_fok_failures == 4


def test_removed_risk_gates_no_longer_block():
    cfg = _cfg()
    state = CanaryState(
        open_position=None,
        high_water_funds_usd=1000.0,
        recent_trade_results=[1] * 20 + [0] * 80,
        recent_entry_prices=[0.58] * 50,
        winner_fill_count=80,
        winner_order_count=100,
        loser_fill_count=95,
        loser_order_count=100,
    )
    decision = evaluate_top159_risk(cfg, state, current_funds_usd=930.0)
    assert decision.allowed
    assert decision.reasons == []


def test_policy_always_uses_one_percent_notional():
    cfg = _cfg()
    policy = CanaryPolicy(cfg)
    assert round(policy.order_notional(1000.0, completed_trades=0), 2) == 10.0
    assert round(policy.order_notional(1000.0, completed_trades=100, stage_cap=0), 2) == 10.0


def test_risk_day_state_is_persistable_on_first_check():
    now = datetime(2026, 5, 1, 8, 0, tzinfo=timezone.utc)
    state = CanaryState(open_position=None)
    updated = state_with_current_risk_day(state, current_funds_usd=850.0, now=now)
    assert updated.day_key == "2026-05-01"
    assert updated.day_start_funds_usd == 850.0
    assert updated.current_funds_usd == 850.0
    assert updated.high_water_funds_usd == 850.0


def test_risk_day_state_resets_on_new_day():
    now = datetime(2026, 5, 2, 0, 1, tzinfo=timezone.utc)
    state = CanaryState(
        open_position=None,
        day_key="2026-05-01",
        day_start_funds_usd=850.0,
        current_funds_usd=830.0,
        high_water_funds_usd=860.0,
        day_high_water_funds_usd=858.0,
    )
    updated = state_with_current_risk_day(state, current_funds_usd=840.0, now=now)
    assert updated.day_key == "2026-05-02"
    assert updated.day_start_funds_usd == 840.0
    assert updated.day_high_water_funds_usd == 840.0
    assert updated.current_funds_usd == 840.0
    assert updated.high_water_funds_usd == 860.0


def test_risk_day_state_keeps_same_day_start():
    now = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    state = CanaryState(
        open_position=None,
        day_key="2026-05-01",
        day_start_funds_usd=850.0,
        current_funds_usd=845.0,
        high_water_funds_usd=850.0,
        day_high_water_funds_usd=855.0,
    )
    updated = state_with_current_risk_day(state, current_funds_usd=840.0, now=now)
    assert updated.day_key == "2026-05-01"
    assert updated.day_start_funds_usd == 850.0
    assert updated.day_high_water_funds_usd == 855.0
    assert updated.current_funds_usd == 840.0


def test_risk_day_uses_beijing_calendar_boundary():
    # 2026-05-01 16:01 UTC is 2026-05-02 00:01 in Beijing.
    now = datetime(2026, 5, 1, 16, 1, tzinfo=timezone.utc)
    state = CanaryState(
        open_position=None,
        day_key="2026-05-01",
        day_start_funds_usd=900.0,
        day_high_water_funds_usd=1000.0,
        current_funds_usd=790.0,
    )
    updated = state_with_current_risk_day(state, current_funds_usd=850.0, now=now)
    assert updated.day_key == "2026-05-02"
    assert updated.day_start_funds_usd == 850.0
    assert updated.day_high_water_funds_usd == 850.0
