from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CanaryPosition:
    token_id: str
    market_slug: str
    entry_price: float
    shares: float
    target_price: float
    buy_order_id: str | None
    opened_at: str


@dataclass(frozen=True)
class CanaryState:
    open_position: CanaryPosition | None
    consecutive_anomalies: int = 0
    # top159 uses FAK orders. Keep the legacy FOK field for old state files and
    # reports, but make FAK the canonical runtime counter.
    consecutive_fak_failures: int = 0
    completed_filled_trades: int = 0
    consecutive_fok_failures: int = 0
    consecutive_losses: int = 0
    current_funds_usd: float | None = None
    high_water_funds_usd: float | None = None
    # Daily high-water mark used by the 20% top159 live pause. This is
    # deliberately separate from the legacy/global high-water display value.
    day_high_water_funds_usd: float | None = None
    day_start_funds_usd: float | None = None
    day_key: str | None = None
    loss_pause_bars_remaining: int = 0
    recent_trade_results: list[int] | None = None
    recent_entry_prices: list[float] | None = None
    winner_fill_count: int = 0
    winner_order_count: int = 0
    loser_fill_count: int = 0
    loser_order_count: int = 0
    risk_pause_until: str | None = None
    failure_pause_until: str | None = None


def empty_state() -> CanaryState:
    return CanaryState(open_position=None, consecutive_anomalies=0)


def load_state(path: str | Path) -> CanaryState:
    p = Path(path)
    if not p.exists():
        return empty_state()
    raw = json.loads(p.read_text())
    pos_raw = raw.get("open_position")
    pos = CanaryPosition(**pos_raw) if isinstance(pos_raw, dict) else None
    execution_failures = int(raw.get("consecutive_fak_failures", raw.get("consecutive_fok_failures", 0)))
    return CanaryState(
        open_position=pos,
        consecutive_anomalies=int(raw.get("consecutive_anomalies", 0)),
        completed_filled_trades=int(raw.get("completed_filled_trades", 0)),
        consecutive_fak_failures=execution_failures,
        consecutive_fok_failures=execution_failures,
        consecutive_losses=int(raw.get("consecutive_losses", 0)),
        current_funds_usd=_optional_float(raw.get("current_funds_usd")),
        high_water_funds_usd=_optional_float(raw.get("high_water_funds_usd")),
        day_high_water_funds_usd=_optional_float(raw.get("day_high_water_funds_usd")),
        day_start_funds_usd=_optional_float(raw.get("day_start_funds_usd")),
        day_key=raw.get("day_key"),
        loss_pause_bars_remaining=int(raw.get("loss_pause_bars_remaining", 0)),
        recent_trade_results=[int(x) for x in (raw.get("recent_trade_results") or [])][-100:],
        recent_entry_prices=[float(x) for x in (raw.get("recent_entry_prices") or [])][-50:],
        winner_fill_count=int(raw.get("winner_fill_count", 0)),
        winner_order_count=int(raw.get("winner_order_count", 0)),
        loser_fill_count=int(raw.get("loser_fill_count", 0)),
        loser_order_count=int(raw.get("loser_order_count", 0)),
        risk_pause_until=raw.get("risk_pause_until"),
        failure_pause_until=raw.get("failure_pause_until"),
    )


def save_state(path: str | Path, state: CanaryState) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = asdict(state)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def opened_position(*, token_id: str, market_slug: str, entry_price: float, shares: float, target_price: float, buy_order_id: str | None) -> CanaryPosition:
    return CanaryPosition(
        token_id=token_id,
        market_slug=market_slug,
        entry_price=float(entry_price),
        shares=float(shares),
        target_price=float(target_price),
        buy_order_id=buy_order_id,
        opened_at=datetime.now(timezone.utc).isoformat(),
    )


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)
