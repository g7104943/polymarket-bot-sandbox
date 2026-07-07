from __future__ import annotations

import json
import math
import os
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from .constants import CLOB_PRODUCTION_HOST, LIVE_ACK_ENV, LIVE_ACK_VALUE, PUSD_ADDRESS
from .execution import LiveSafetyError
from .ledger import JsonlLedger
from .official import OfficialClient
from .types import CandidateSignal, OfficialOrderStatus, OrderPlan, OrderTruth


@dataclass(frozen=True)
class ArbCanaryConfig:
    system_name: str
    live_enabled: bool
    clob_host: str
    gamma_host: str
    pusd_address: str
    max_total_cost_usd: float
    target_payout_usd: float
    min_edge_usd: float
    min_edge_pct: float
    min_liquidity_usd: float
    min_volume_usd: float
    min_order_shares: float
    max_outcomes: int
    scan_limit: int
    order_type: str
    allow_non_atomic_live_execution: bool
    official_truth_required: bool

    def validate(self) -> None:
        problems: list[str] = []
        if self.clob_host != CLOB_PRODUCTION_HOST:
            problems.append("clob_host must be production CLOB host")
        if self.pusd_address.lower() != PUSD_ADDRESS.lower():
            problems.append("pUSD collateral address mismatch")
        if self.max_total_cost_usd > 5:
            problems.append("max_total_cost_usd cannot exceed 5U in canary")
        if self.target_payout_usd > 5:
            problems.append("target_payout_usd cannot exceed 5U in canary")
        if self.target_payout_usd < self.min_order_shares:
            problems.append("target_payout_usd must cover minimum order shares")
        if self.min_edge_usd <= 0 or self.min_edge_pct <= 0:
            problems.append("edge thresholds must be positive")
        if self.order_type.upper() not in {"FOK", "FAK"}:
            problems.append("arb canary only allows FOK/FAK")
        if not self.official_truth_required:
            problems.append("official_truth_required must stay true")
        if problems:
            raise ValueError("; ".join(problems))


@dataclass(frozen=True)
class MarketOutcome:
    token_id: str
    outcome: str


@dataclass(frozen=True)
class ArbMarket:
    market_id: str
    condition_id: str
    slug: str
    question: str
    liquidity: float
    volume: float
    outcomes: list[MarketOutcome]


@dataclass(frozen=True)
class BookLevel:
    price: float
    size: float


@dataclass(frozen=True)
class OrderBook:
    token_id: str
    asks: list[BookLevel]
    bids: list[BookLevel]


@dataclass(frozen=True)
class ArbLeg:
    token_id: str
    outcome: str
    shares: float
    worst_price: float
    cost_usd: float
    levels_used: int


@dataclass(frozen=True)
class ArbOpportunity:
    market: ArbMarket
    legs: list[ArbLeg]
    payout_usd: float
    total_cost_usd: float
    edge_usd: float
    edge_pct: float
    order_type: str
    reason: str
    scanned_at: str


@dataclass(frozen=True)
class ArbScanReport:
    generated_at: str
    markets_seen: int
    markets_eligible: int
    markets_with_books: int
    opportunities: list[ArbOpportunity]
    errors: list[str]


def load_arb_config(path: str | Path) -> ArbCanaryConfig:
    cfg = ArbCanaryConfig(**json.loads(Path(path).read_text()))
    cfg.validate()
    return cfg


class GammaMarketClient:
    def __init__(self, gamma_host: str, clob_host: str, *, timeout: float = 10.0):
        self.gamma_host = gamma_host.rstrip("/")
        self.clob_host = clob_host.rstrip("/")
        self.timeout = timeout

    def fetch_markets(self, *, limit: int) -> list[ArbMarket]:
        params = {
            "active": "true",
            "closed": "false",
            "limit": str(limit),
            "order": "liquidity",
            "ascending": "false",
        }
        raw = self._get_json(f"{self.gamma_host}/markets?{urllib.parse.urlencode(params)}")
        if not isinstance(raw, list):
            return []
        markets: list[ArbMarket] = []
        for item in raw:
            market = _market_from_gamma(item)
            if market is not None:
                markets.append(market)
        return markets

    def fetch_orderbook(self, token_id: str) -> OrderBook | None:
        raw = self._get_json(f"{self.clob_host}/book?{urllib.parse.urlencode({'token_id': token_id})}")
        if not isinstance(raw, dict):
            return None
        return _orderbook_from_raw(token_id, raw)

    def _get_json(self, url: str) -> Any:
        request = urllib.request.Request(url, headers={"User-Agent": "polyfun-next-arb-canary/0.1"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))


def find_arbitrage_opportunity(
    market: ArbMarket,
    books: dict[str, OrderBook],
    config: ArbCanaryConfig,
) -> ArbOpportunity | None:
    if len(market.outcomes) < 2 or len(market.outcomes) > config.max_outcomes:
        return None
    if market.liquidity < config.min_liquidity_usd or market.volume < config.min_volume_usd:
        return None

    target_shares = min(config.target_payout_usd, config.max_total_cost_usd)
    if target_shares < config.min_order_shares:
        return None

    legs: list[ArbLeg] = []
    total_cost = 0.0
    for outcome in market.outcomes:
        book = books.get(outcome.token_id)
        if book is None:
            return None
        fill = _cost_to_buy_shares(book.asks, target_shares)
        if fill is None:
            return None
        cost, worst_price, levels_used = fill
        total_cost += cost
        legs.append(
            ArbLeg(
                token_id=outcome.token_id,
                outcome=outcome.outcome,
                shares=target_shares,
                worst_price=worst_price,
                cost_usd=cost,
                levels_used=levels_used,
            )
        )

    payout = target_shares
    edge = payout - total_cost
    edge_pct = edge / payout if payout > 0 else 0.0
    if total_cost > config.max_total_cost_usd:
        return None
    if edge < config.min_edge_usd or edge_pct < config.min_edge_pct:
        return None
    return ArbOpportunity(
        market=market,
        legs=legs,
        payout_usd=payout,
        total_cost_usd=total_cost,
        edge_usd=edge,
        edge_pct=edge_pct,
        order_type=config.order_type.upper(),
        reason="complete-set ask package costs less than guaranteed payout",
        scanned_at=datetime.now(timezone.utc).isoformat(),
    )


def scan_orderbook_arbitrage(config: ArbCanaryConfig, *, market_limit: Optional[int] = None) -> ArbScanReport:
    client = GammaMarketClient(config.gamma_host, config.clob_host)
    markets = client.fetch_markets(limit=market_limit or config.scan_limit)
    opportunities: list[ArbOpportunity] = []
    errors: list[str] = []
    eligible = 0
    with_books = 0
    for market in markets:
        if (
            len(market.outcomes) < 2
            or len(market.outcomes) > config.max_outcomes
            or market.liquidity < config.min_liquidity_usd
            or market.volume < config.min_volume_usd
        ):
            continue
        eligible += 1
        books: dict[str, OrderBook] = {}
        failed = False
        for outcome in market.outcomes:
            try:
                book = client.fetch_orderbook(outcome.token_id)
            except Exception as exc:
                errors.append(f"{market.slug}:{outcome.outcome}: orderbook error: {exc!r}")
                failed = True
                break
            if book is None:
                failed = True
                break
            books[outcome.token_id] = book
        if failed:
            continue
        with_books += 1
        opp = find_arbitrage_opportunity(market, books, config)
        if opp is not None:
            opportunities.append(opp)
    opportunities.sort(key=lambda o: (o.edge_usd, o.edge_pct), reverse=True)
    return ArbScanReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        markets_seen=len(markets),
        markets_eligible=eligible,
        markets_with_books=with_books,
        opportunities=opportunities,
        errors=errors,
    )


class ArbExecutionEngine:
    def __init__(self, config: ArbCanaryConfig, official: OfficialClient, ledger: JsonlLedger):
        self.config = config
        self.official = official
        self.ledger = ledger

    def execute(self, opportunity: ArbOpportunity, *, dry_run: bool = True) -> list[OfficialOrderStatus]:
        self.ledger.append("arb_opportunity_plan", {"opportunity": to_jsonable(opportunity)})
        if dry_run or not self.config.live_enabled:
            statuses = [
                OfficialOrderStatus(
                    order_id=None,
                    truth=OrderTruth.DRY_RUN,
                    raw={"leg": to_jsonable(leg), "order_type": opportunity.order_type},
                    message="dry-run or live disabled",
                )
                for leg in opportunity.legs
            ]
            self.ledger.append("arb_not_submitted", {"statuses": [to_jsonable(s) for s in statuses]})
            return statuses

        if os.environ.get(LIVE_ACK_ENV) != LIVE_ACK_VALUE:
            raise LiveSafetyError(
                f"live arb blocked: set {LIVE_ACK_ENV}={LIVE_ACK_VALUE} to acknowledge canary risk"
            )
        if not self.config.allow_non_atomic_live_execution:
            raise LiveSafetyError(
                "live arb blocked: multi-leg execution is non-atomic; set allow_non_atomic_live_execution only after manual acceptance"
            )

        statuses: list[OfficialOrderStatus] = []
        for leg in opportunity.legs:
            plan = _order_plan_for_leg(opportunity, leg, self.config)
            posted = self.official.post_buy_order_with_type(plan, opportunity.order_type)
            self.ledger.append("arb_leg_post_result", {"status": to_jsonable(posted)})
            statuses.append(posted)
            if posted.truth in {OrderTruth.OFFICIAL_REJECTED, OrderTruth.OFFICIAL_MISSING}:
                self.ledger.append(
                    "arb_unbalanced_exposure_risk",
                    {"failed_leg": to_jsonable(leg), "statuses": [to_jsonable(s) for s in statuses]},
                )
                break
        return statuses


def _order_plan_for_leg(opportunity: ArbOpportunity, leg: ArbLeg, config: ArbCanaryConfig) -> OrderPlan:
    candidate = CandidateSignal(
        symbol="ARB",
        period="complete_set",
        market_slug=opportunity.market.slug,
        condition_id=opportunity.market.condition_id,
        token_id=leg.token_id,
        side="BUY",
        model_score=1.0,
    )
    return OrderPlan(
        candidate=candidate,
        price=leg.worst_price,
        size_shares=leg.shares,
        notional_usd=leg.cost_usd,
        cancel_after_seconds=0,
        reason=f"arb canary {opportunity.order_type}: buy complete-set leg {leg.outcome}",
    )


def _cost_to_buy_shares(asks: Iterable[BookLevel], target_shares: float) -> tuple[float, float, int] | None:
    remaining = target_shares
    cost = 0.0
    worst = 0.0
    used = 0
    for level in sorted(asks, key=lambda x: x.price):
        if remaining <= 1e-9:
            break
        take = min(remaining, level.size)
        cost += take * level.price
        worst = max(worst, level.price)
        remaining -= take
        used += 1
    if remaining > 1e-9:
        return None
    if not math.isfinite(cost) or cost <= 0 or worst <= 0:
        return None
    return cost, worst, used


def _market_from_gamma(item: Any) -> ArbMarket | None:
    if not isinstance(item, dict):
        return None
    token_ids = _parse_jsonish(item.get("clobTokenIds"))
    outcomes = _parse_jsonish(item.get("outcomes"))
    if not isinstance(token_ids, list) or len(token_ids) < 2:
        return None
    if not isinstance(outcomes, list) or len(outcomes) != len(token_ids):
        outcomes = [f"outcome_{idx + 1}" for idx in range(len(token_ids))]
    parsed_outcomes = [
        MarketOutcome(token_id=str(token_id), outcome=str(outcomes[idx]))
        for idx, token_id in enumerate(token_ids)
        if str(token_id)
    ]
    if len(parsed_outcomes) != len(token_ids):
        return None
    return ArbMarket(
        market_id=str(item.get("id") or ""),
        condition_id=str(item.get("conditionId") or item.get("condition_id") or ""),
        slug=str(item.get("slug") or ""),
        question=str(item.get("question") or item.get("title") or ""),
        liquidity=_to_float(item.get("liquidity")),
        volume=_to_float(item.get("volume")),
        outcomes=parsed_outcomes,
    )


def _orderbook_from_raw(token_id: str, raw: dict[str, Any]) -> OrderBook:
    asks = [_level_from_raw(level) for level in raw.get("asks", [])]
    bids = [_level_from_raw(level) for level in raw.get("bids", [])]
    return OrderBook(
        token_id=token_id,
        asks=[level for level in asks if level is not None],
        bids=[level for level in bids if level is not None],
    )


def _level_from_raw(level: Any) -> BookLevel | None:
    if isinstance(level, dict):
        price = _to_float(level.get("price") or level.get("p") or level.get("px"))
        size = _to_float(level.get("size") or level.get("s") or level.get("shares"))
    elif isinstance(level, (list, tuple)) and len(level) >= 2:
        price = _to_float(level[0])
        size = _to_float(level[1])
    else:
        return None
    if price <= 0 or size <= 0:
        return None
    return BookLevel(price=price, size=size)


def _parse_jsonish(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return None
    return value


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return {key: to_jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value
