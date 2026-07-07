from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class OrderTruth(str, Enum):
    DRY_RUN = "dry_run"
    OFFICIAL_OPEN = "official_open"
    OFFICIAL_FILLED = "official_filled"
    OFFICIAL_CANCELED_ZERO_FILL = "official_canceled_zero_fill"
    OFFICIAL_REJECTED = "official_rejected"
    OFFICIAL_MISSING = "official_missing"


@dataclass(frozen=True)
class CandidateSignal:
    symbol: str
    period: str
    market_slug: str
    condition_id: str
    token_id: str
    side: str
    model_score: float
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str | None = None
    live_model_profile: str | None = None
    selected_candidate: str | None = None
    train_window: str | None = None
    feature_mode: str | None = None
    edge: float | None = None
    strategy_profile: str | None = None
    base_model_profile: str | None = None
    base_selected_candidate: str | None = None
    shock_filter_enabled: bool = False
    shock_condition: bool = False
    shock_gate_probability: float | None = None
    shock_gate_threshold: float | None = None
    shock_action: str | None = None
    shock_reason: str | None = None
    shock_profile: str | None = None
    shock_candidate_id: str | None = None
    shock_model_engine: str | None = None
    shock_model_hyper: dict[str, Any] | None = None
    calibration_router_enabled: bool = False
    calibration_router_profile: str | None = None
    calibration_router_candidate_id: str | None = None
    calibration_router_shadow_only: bool = False
    router_probability: float | None = None
    router_required_probability: float | None = None
    router_combo_score: float | None = None
    router_combo_threshold: float | None = None
    router_policy_mode: str | None = None
    router_daily_cap_mode: str | None = None
    router_action: str | None = None
    router_reason: str | None = None
    router_model_key: str | None = None
    router_feature_mode: str | None = None


@dataclass(frozen=True)
class OrderbookQuote:
    best_bid: Optional[float]
    best_ask: Optional[float]
    bid_depth_shares: float
    ask_depth_shares: float
    minutes_remaining: float
    source: str = "rest"
    source_age_seconds: float | None = None
    source_updated_at: str | None = None

    @property
    def spread_pct(self) -> Optional[float]:
        if self.best_bid is None or self.best_ask is None or self.best_ask <= 0:
            return None
        mid = (self.best_bid + self.best_ask) / 2
        if mid <= 0:
            return None
        return (self.best_ask - self.best_bid) / mid


@dataclass(frozen=True)
class OrderPlan:
    candidate: CandidateSignal
    price: float
    size_shares: float
    notional_usd: float
    cancel_after_seconds: int
    reason: str
    order_type: str = "FOK"
    risk_stage: int = 0


@dataclass(frozen=True)
class ExitPlan:
    token_id: str
    market_slug: str
    size_shares: float
    target_price: float
    worst_sell_price: float
    reason: str


@dataclass(frozen=True)
class OfficialOrderStatus:
    order_id: Optional[str]
    truth: OrderTruth
    raw: dict[str, Any]
    matched_shares: float = 0.0
    message: str = ""
