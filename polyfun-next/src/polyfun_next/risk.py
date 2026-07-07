from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from .canary_state import CanaryState
from .config import CanaryConfig

BEIJING_TZ = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reasons: list[str]
    metrics: dict[str, float | int | str | None] | None = None


def evaluate_top159_risk(
    config: CanaryConfig,
    state: CanaryState,
    *,
    current_funds_usd: float,
    now: datetime | None = None,
) -> RiskDecision:
    """Simplified top159 risk gate.

    The only live pauses are:
    - daily loss reaches 20%, pause 24h;
    - consecutive FOK/FAK failures reaches 5, pause 4h.
    """
    now = now or datetime.now(timezone.utc)
    day_key = now.astimezone(BEIJING_TZ).date().isoformat()
    day_start = float(state.day_start_funds_usd or current_funds_usd)
    day_high_water = float(
        state.day_high_water_funds_usd
        if state.day_high_water_funds_usd is not None
        else (state.high_water_funds_usd if state.high_water_funds_usd is not None else day_start)
    )
    if state.day_key and state.day_key != day_key:
        day_start = current_funds_usd
        day_high_water = current_funds_usd
    day_high_water = max(day_high_water, float(current_funds_usd))

    daily_loss = (day_high_water - current_funds_usd) / day_high_water if day_high_water > 0 else 0.0
    reasons: list[str] = []
    active_daily_pause = _is_active_pause(state.risk_pause_until, now)
    active_failure_pause = _is_active_pause(state.failure_pause_until, now)
    if active_daily_pause:
        reasons.append("daily_loss_pause_active")
    if active_failure_pause:
        reasons.append("execution_failure_pause_active")
    if daily_loss >= config.daily_loss_stop_fraction:
        reasons.append("daily_loss_stop")
    execution_failures = max(
        int(getattr(state, "consecutive_fak_failures", 0)),
        int(getattr(state, "consecutive_fok_failures", 0)),
    )
    if execution_failures >= config.failure_pause_count:
        reasons.append("execution_failure_pause")

    return RiskDecision(
        allowed=not reasons,
        reasons=reasons,
        metrics={
            "current_funds_usd": round(float(current_funds_usd), 8),
            "day_start_funds_usd": round(float(day_start), 8),
            "day_high_water_funds_usd": round(float(day_high_water), 8),
            "daily_loss_fraction": round(float(daily_loss), 8),
            "daily_drawdown_from_high_fraction": round(float(daily_loss), 8),
            "daily_loss_trigger_funds_usd": round(float(day_high_water) * (1.0 - config.daily_loss_stop_fraction), 8),
            "consecutive_fak_failures": execution_failures,
            "consecutive_fok_failures": execution_failures,
            "risk_pause_until": state.risk_pause_until,
            "failure_pause_until": state.failure_pause_until,
            "day_key": day_key,
        },
    )


def state_with_current_risk_day(
    state: CanaryState,
    *,
    current_funds_usd: float,
    now: datetime | None = None,
) -> CanaryState:
    """Return state with a persisted day key/start balance for daily-loss checks."""
    now = now or datetime.now(timezone.utc)
    day_key = now.astimezone(BEIJING_TZ).date().isoformat()
    expired_daily_pause = bool(state.risk_pause_until) and not _is_active_pause(state.risk_pause_until, now)
    expired_failure_pause = bool(state.failure_pause_until) and not _is_active_pause(state.failure_pause_until, now)
    day_start = state.day_start_funds_usd
    if state.day_key != day_key or day_start is None or expired_daily_pause:
        day_start = float(current_funds_usd)
        day_high_water = float(current_funds_usd)
        risk_pause_until = None
    else:
        day_high_water = float(
            state.day_high_water_funds_usd
            if state.day_high_water_funds_usd is not None
            else (state.high_water_funds_usd if state.high_water_funds_usd is not None else current_funds_usd)
        )
        day_high_water = max(day_high_water, float(current_funds_usd))
        risk_pause_until = state.risk_pause_until
    high_water = max(float(state.high_water_funds_usd or current_funds_usd), float(current_funds_usd))
    failure_pause_until = state.failure_pause_until
    consecutive_fak_failures = state.consecutive_fak_failures
    consecutive_fok_failures = state.consecutive_fok_failures
    if expired_failure_pause:
        # A 5x FAK/FOK execution-failure pause is a timed cooldown, not a
        # permanent latch. Once it expires, clear the failure counter so the
        # next supervisor tick can trade again instead of immediately
        # re-opening another 4h pause from the same old failures.
        failure_pause_until = None
        consecutive_fak_failures = 0
        consecutive_fok_failures = 0

    return replace(
        state,
        day_key=day_key,
        day_start_funds_usd=float(day_start),
        day_high_water_funds_usd=float(day_high_water),
        current_funds_usd=float(current_funds_usd),
        high_water_funds_usd=high_water,
        risk_pause_until=risk_pause_until,
        failure_pause_until=failure_pause_until,
        consecutive_fak_failures=consecutive_fak_failures,
        consecutive_fok_failures=consecutive_fok_failures,
    )


def _is_active_pause(value: str | None, now: datetime) -> bool:
    if not value:
        return False
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt > now
