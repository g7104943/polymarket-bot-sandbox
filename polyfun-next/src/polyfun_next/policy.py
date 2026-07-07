from __future__ import annotations

from .config import CanaryConfig
from typing import Optional

from .types import CandidateSignal, ExitPlan, OrderPlan, OrderbookQuote


class CanaryPolicy:
    """Small, boring, execution-first policy for ETH 15m canary orders."""

    def __init__(self, config: CanaryConfig):
        self.config = config

    def order_notional(self, current_funds_usd: float, *, completed_trades: int = 0, stage_cap: int | None = None) -> float:
        """Fixed 1% stake. completed_trades/stage_cap are ignored for backward compatibility."""
        notional = current_funds_usd * self.config.stake_fraction
        if self.config.max_order_usd > 0:
            return min(self.config.max_order_usd, notional)
        return notional

    def risk_stage(self, completed_trades: int, *, stage_cap: int | None = None) -> int:
        """Legacy display field only; staged sizing has been removed."""
        return 0

    def build_order_plan(
        self,
        candidate: CandidateSignal,
        quote: OrderbookQuote,
        current_funds_usd: float,
        completed_trades: int = 0,
        previous_quote: OrderbookQuote | None = None,
        stage_cap: int | None = None,
    ) -> Optional[OrderPlan]:
        if self.entry_rejection_reasons(candidate, quote, current_funds_usd, previous_quote=previous_quote):
            return None

        notional = self.order_notional(current_funds_usd)
        size = notional / quote.best_ask
        protection_price = min(0.98, quote.best_ask + self.config.max_quote_jump_price)
        if self.config.enforce_max_entry_price and protection_price > self.config.max_entry_price:
            protection_price = quote.best_ask

        stage = 0
        price_gate = "price-gates-off" if (
            not self.config.enforce_max_entry_price and not self.config.enforce_value_edge_vs_price
        ) else "price-gates-on"
        return OrderPlan(
            candidate=candidate,
            price=protection_price,
            size_shares=size,
            notional_usd=notional,
            cancel_after_seconds=self.config.cancel_after_seconds,
            order_type=self.config.preferred_order_type.upper(),
            risk_stage=stage,
            reason=(
                f"newslot1 top159: immediate FAK buy passed model-score, depth, "
                f"quote-stability, fixed-1%-stake, and entry-window gates ({price_gate})"
            ),
        )

    def entry_rejection_reasons(
        self,
        candidate: CandidateSignal,
        quote: OrderbookQuote,
        current_funds_usd: float,
        *,
        previous_quote: OrderbookQuote | None = None,
    ) -> list[str]:
        reasons: list[str] = []
        if candidate.symbol != self.config.symbol or candidate.period != self.config.period:
            reasons.append("candidate_symbol_or_period_mismatch")
            return reasons
        if quote.minutes_remaining < self.config.min_market_minutes_remaining:
            reasons.append("market_too_close_to_end")
        elapsed_seconds = max(0.0, 15.0 * 60.0 - quote.minutes_remaining * 60.0)
        if elapsed_seconds < self.config.entry_window_start_seconds:
            reasons.append("before_entry_window")
        if elapsed_seconds > self.config.entry_window_end_seconds:
            reasons.append("after_entry_window")
        if quote.best_bid is None or quote.best_ask is None:
            reasons.append("missing_best_bid_or_ask")
            return reasons
        if quote.best_ask <= 0 or quote.best_ask >= 0.99:
            reasons.append("invalid_best_ask")
        if self.config.enforce_max_entry_price and quote.best_ask > self.config.max_entry_price:
            reasons.append("best_ask_above_max_entry_price")
        if previous_quote is not None and previous_quote.best_ask is not None:
            if abs(quote.best_ask - previous_quote.best_ask) > self.config.max_quote_jump_price:
                reasons.append("ask_jump_too_large")
        if previous_quote is not None and previous_quote.best_bid is not None and quote.best_bid is not None:
            if abs(quote.best_bid - previous_quote.best_bid) > self.config.max_quote_jump_price:
                reasons.append("bid_jump_too_large")
        if candidate.model_score < 0.5 + self.config.min_value_edge:
            reasons.append("model_score_below_edge")
        if self.config.enforce_value_edge_vs_price and (candidate.model_score - quote.best_ask) < self.config.min_value_edge:
            reasons.append("model_value_edge_below_min_edge")

        notional = self.order_notional(current_funds_usd)
        size = notional / quote.best_ask
        if size < self.config.min_expected_fill_shares:
            reasons.append("order_size_below_min_expected_shares")
        if quote.ask_depth_shares < size * self.config.min_depth_multiplier:
            reasons.append("ask_depth_insufficient")
        if quote.bid_depth_shares < self.config.min_expected_fill_shares:
            reasons.append("bid_depth_insufficient_for_future_exit_check")
        return reasons

    def take_profit_target(self, entry_price: float) -> float:
        return min(0.99, entry_price * (1.0 + self.config.take_profit_pct))

    def build_take_profit_exit(
        self,
        *,
        token_id: str,
        market_slug: str,
        entry_price: float,
        shares: float,
        quote: OrderbookQuote,
    ) -> ExitPlan | None:
        if not self.config.allow_take_profit:
            return None
        if shares <= 0 or quote.best_bid is None or quote.best_bid <= 0:
            return None
        target = self.take_profit_target(entry_price)
        if quote.best_bid < target:
            return None
        if quote.bid_depth_shares < shares:
            return None
        worst = max(0.01, quote.best_bid * (1.0 - self.config.max_exit_slippage_pct))
        return ExitPlan(
            token_id=token_id,
            market_slug=market_slug,
            size_shares=shares,
            target_price=target,
            worst_sell_price=worst,
            reason="ETH15m canary: fixed 20% take-profit reached; sell with protected market order",
        )
