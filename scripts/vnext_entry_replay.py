#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence


ENTRY_REPLAY_PRESTART_SEC = 900
MARKET_DURATION_SEC = 900


@dataclass(frozen=True)
class EntryExecutionTrace:
    prediction_ts: int
    decision_ts: int
    target_period_end_ts: int
    market_slug: str | None
    market_discovered_ts: int | None
    order_submitted_ts: int | None
    limit_price: float
    fill_status: str
    fill_ts: int | None
    fill_price: float | None
    fill_fraction: float
    timeout_ts: int
    window_start_ts: int
    window_end_ts: int
    first_observed_price: float | None
    min_observed_price: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def market_start_from_prediction_ts(prediction_ts: int) -> int:
    return int(prediction_ts) + ENTRY_REPLAY_PRESTART_SEC


def market_end_from_start_ts(market_start_ts: int) -> int:
    return int(market_start_ts) + MARKET_DURATION_SEC


def replay_entry_trace(
    *,
    prediction_ts: int,
    market_start_ts: int,
    limit_price: float,
    t_sec_list: Sequence[int],
    direction_prices: Sequence[float],
    market_slug: str | None = None,
) -> EntryExecutionTrace:
    target_period_end_ts = market_end_from_start_ts(market_start_ts)
    window_start_ts = int(prediction_ts)
    window_end_ts = int(target_period_end_ts)
    timeout_ts = int(target_period_end_ts)

    if limit_price <= 0:
        return EntryExecutionTrace(
            prediction_ts=int(prediction_ts),
            decision_ts=int(market_start_ts),
            target_period_end_ts=target_period_end_ts,
            market_slug=market_slug,
            market_discovered_ts=None,
            order_submitted_ts=None,
            limit_price=float(limit_price),
            fill_status='skipped',
            fill_ts=None,
            fill_price=None,
            fill_fraction=0.0,
            timeout_ts=timeout_ts,
            window_start_ts=window_start_ts,
            window_end_ts=window_end_ts,
            first_observed_price=None,
            min_observed_price=None,
        )

    seq = [
        (int(t), float(p))
        for t, p in zip(t_sec_list, direction_prices)
        if window_start_ts <= int(t) <= window_end_ts
    ]
    if not seq:
        return EntryExecutionTrace(
            prediction_ts=int(prediction_ts),
            decision_ts=int(market_start_ts),
            target_period_end_ts=target_period_end_ts,
            market_slug=market_slug,
            market_discovered_ts=None,
            order_submitted_ts=None,
            limit_price=float(limit_price),
            fill_status='no_market_data',
            fill_ts=None,
            fill_price=None,
            fill_fraction=0.0,
            timeout_ts=timeout_ts,
            window_start_ts=window_start_ts,
            window_end_ts=window_end_ts,
            first_observed_price=None,
            min_observed_price=None,
        )

    market_discovered_ts = int(seq[0][0])
    order_submitted_ts = market_discovered_ts
    first_observed_price = float(seq[0][1])
    min_observed_price = float(min(p for _, p in seq))

    for t, p in seq:
        if float(p) <= float(limit_price) + 1e-12:
            return EntryExecutionTrace(
                prediction_ts=int(prediction_ts),
                decision_ts=int(market_start_ts),
                target_period_end_ts=target_period_end_ts,
                market_slug=market_slug,
                market_discovered_ts=market_discovered_ts,
                order_submitted_ts=order_submitted_ts,
                limit_price=float(limit_price),
                fill_status='filled',
                fill_ts=int(t),
                fill_price=float(p),
                fill_fraction=1.0,
                timeout_ts=timeout_ts,
                window_start_ts=window_start_ts,
                window_end_ts=window_end_ts,
                first_observed_price=first_observed_price,
                min_observed_price=min_observed_price,
            )

    return EntryExecutionTrace(
        prediction_ts=int(prediction_ts),
        decision_ts=int(market_start_ts),
        target_period_end_ts=target_period_end_ts,
        market_slug=market_slug,
        market_discovered_ts=market_discovered_ts,
        order_submitted_ts=order_submitted_ts,
        limit_price=float(limit_price),
        fill_status='timeout_unfilled',
        fill_ts=None,
        fill_price=None,
        fill_fraction=0.0,
        timeout_ts=timeout_ts,
        window_start_ts=window_start_ts,
        window_end_ts=window_end_ts,
        first_observed_price=first_observed_price,
        min_observed_price=min_observed_price,
    )
