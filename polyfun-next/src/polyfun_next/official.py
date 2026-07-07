from __future__ import annotations

import importlib.util
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Optional, Protocol

from .config import CanaryConfig
from .types import ExitPlan, OfficialOrderStatus, OrderPlan, OrderTruth, OrderbookQuote
from .ws_cache import load_ws_quote


@dataclass(frozen=True)
class SdkStatus:
    package: str
    installed: bool
    message: str


def check_v2_sdk() -> SdkStatus:
    installed = importlib.util.find_spec("py_clob_client_v2") is not None
    if installed:
        return SdkStatus("py_clob_client_v2", True, "V2 Python SDK importable")
    return SdkStatus(
        "py_clob_client_v2",
        False,
        "V2 Python SDK is not installed in this interpreter; install py-clob-client-v2 before live mode",
    )


class OfficialClient(Protocol):
    def post_buy_order(self, plan: OrderPlan) -> OfficialOrderStatus: ...

    def post_buy_order_with_type(self, plan: OrderPlan, order_type: str) -> OfficialOrderStatus: ...

    def post_sell_order(self, plan: ExitPlan) -> OfficialOrderStatus: ...

    def get_order(self, order_id: str) -> OfficialOrderStatus: ...

    def cancel_order(self, order_id: str) -> OfficialOrderStatus: ...

    def get_orderbook_quote(self, token_id: str, minutes_remaining: float = 10.0) -> OrderbookQuote: ...


class DryRunOfficialClient:
    """Client used in tests and dry-runs. It never reports a real official order."""

    def post_buy_order(self, plan: OrderPlan) -> OfficialOrderStatus:
        return self.post_buy_order_with_type(plan, "FAK")

    def post_buy_order_with_type(self, plan: OrderPlan, order_type: str) -> OfficialOrderStatus:
        return OfficialOrderStatus(
            order_id=None,
            truth=OrderTruth.DRY_RUN,
            raw={"plan": _safe_plan(plan), "mode": "dry_run", "order_type": order_type},
            message="dry-run: no official order submitted",
        )

    def post_sell_order(self, plan: ExitPlan) -> OfficialOrderStatus:
        return OfficialOrderStatus(
            order_id=None,
            truth=OrderTruth.DRY_RUN,
            raw={"plan": _safe_raw(plan.__dict__), "mode": "dry_run"},
            message="dry-run: no official sell order submitted",
        )

    def get_order(self, order_id: str) -> OfficialOrderStatus:
        return OfficialOrderStatus(
            order_id=order_id,
            truth=OrderTruth.OFFICIAL_MISSING,
            raw={},
            message="dry-run client has no official order store",
        )

    def cancel_order(self, order_id: str) -> OfficialOrderStatus:
        return OfficialOrderStatus(
            order_id=order_id,
            truth=OrderTruth.DRY_RUN,
            raw={"mode": "dry_run"},
            message="dry-run: no official order canceled",
        )

    def get_orderbook_quote(self, token_id: str, minutes_remaining: float = 10.0) -> OrderbookQuote:
        return OrderbookQuote(best_bid=0.50, best_ask=0.51, bid_depth_shares=100.0, ask_depth_shares=100.0, minutes_remaining=minutes_remaining)


class ClobV2SdkOfficialClient:
    """Thin adapter around py-clob-client-v2.

    Environment variables are intentionally explicit so this new project does not silently reuse
    legacy slot1/slot2 credentials with ambiguous semantics.
    """

    def __init__(self, config: CanaryConfig):
        try:
            from py_clob_client_v2 import ApiCreds, ClobClient
        except Exception as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("py_clob_client_v2 is required for live mode") from exc

        private_key = _required_env("POLYFUN_NEXT_PRIVATE_KEY")
        api_key = _required_env("POLYFUN_NEXT_API_KEY")
        api_secret = _required_env("POLYFUN_NEXT_API_SECRET")
        api_passphrase = _required_env("POLYFUN_NEXT_API_PASSPHRASE")
        funder = os.environ.get("POLYFUN_NEXT_FUNDER") or None
        sig_raw = os.environ.get("POLYFUN_NEXT_SIGNATURE_TYPE")
        if funder and not sig_raw:
            raise RuntimeError("missing POLYFUN_NEXT_SIGNATURE_TYPE for proxy/funder wallet; slot1 top159 should use 2 for Gnosis Safe")
        signature_type = int(sig_raw or "0")

        self.config = config
        creds = ApiCreds(api_key=api_key, api_secret=api_secret, api_passphrase=api_passphrase)
        self._client = ClobClient(
            config.clob_host,
            137,
            key=private_key,
            creds=creds,
            signature_type=signature_type,
            funder=funder,
            use_server_time=True,
            retry_on_error=True,
        )

    def post_buy_order(self, plan: OrderPlan) -> OfficialOrderStatus:
        return self.post_buy_order_with_type(plan, "FAK")

    def post_buy_order_with_type(self, plan: OrderPlan, order_type: str) -> OfficialOrderStatus:
        from py_clob_client_v2 import MarketOrderArgs, OrderType, Side
        sdk_order_type = getattr(OrderType, order_type.upper(), OrderType.FAK)

        order_args = MarketOrderArgs(
            token_id=plan.candidate.token_id,
            amount=float(plan.notional_usd),
            side=Side.BUY,
            price=float(plan.price),
            order_type=sdk_order_type,
        )
        try:
            raw = self._client.create_and_post_market_order(order_args, order_type=sdk_order_type)
        except Exception as exc:  # pragma: no cover - live path
            reconciled = self._reconcile_post_exception(exc, order_type=order_type, side="buy", plan=_safe_plan(plan))
            if reconciled is not None:
                return reconciled
            return OfficialOrderStatus(
                order_id=None,
                truth=OrderTruth.OFFICIAL_REJECTED,
                raw={"exception": repr(exc), "order_type": order_type},
                message=f"CLOB V2 SDK rejected or failed while posting {order_type.upper()} buy order",
            )
        order_id = _extract_order_id(raw)
        if not order_id:
            return OfficialOrderStatus(
                order_id=None,
                truth=OrderTruth.OFFICIAL_MISSING,
                raw=_safe_raw(raw),
                message="postOrder returned no recognizable official order id",
            )
        return OfficialOrderStatus(
            order_id=order_id,
            truth=OrderTruth.OFFICIAL_OPEN,
            raw=_safe_raw(raw),
            message="postOrder returned official order id",
        )

    def post_sell_order(self, plan: ExitPlan) -> OfficialOrderStatus:
        from py_clob_client_v2 import MarketOrderArgs, OrderType, Side

        order_args = MarketOrderArgs(
            token_id=plan.token_id,
            amount=float(plan.size_shares),
            side=Side.SELL,
            price=float(plan.worst_sell_price),
            order_type=OrderType.FAK,
        )
        try:
            raw = self._client.create_and_post_market_order(order_args, order_type=OrderType.FAK)
        except Exception as exc:  # pragma: no cover - live path
            reconciled = self._reconcile_post_exception(exc, order_type="FAK", side="sell", plan=_safe_raw(plan.__dict__))
            if reconciled is not None:
                return reconciled
            return OfficialOrderStatus(
                order_id=None,
                truth=OrderTruth.OFFICIAL_REJECTED,
                raw={"exception": repr(exc), "plan": _safe_raw(plan.__dict__)},
                message="CLOB V2 SDK rejected or failed while posting FAK sell order",
            )
        order_id = _extract_order_id(raw)
        if not order_id:
            return OfficialOrderStatus(
                order_id=None,
                truth=OrderTruth.OFFICIAL_MISSING,
                raw=_safe_raw(raw),
                message="sell postOrder returned no recognizable official order id",
            )
        return OfficialOrderStatus(
            order_id=order_id,
            truth=OrderTruth.OFFICIAL_OPEN,
            raw=_safe_raw(raw),
            message="sell postOrder returned official order id",
        )

    def get_order(self, order_id: str) -> OfficialOrderStatus:
        try:
            raw = self._client.get_order(order_id)
        except Exception as exc:  # pragma: no cover - live path
            return OfficialOrderStatus(
                order_id=order_id,
                truth=OrderTruth.OFFICIAL_MISSING,
                raw={"exception": repr(exc)},
                message="getOrder failed",
            )
        return _status_from_raw_order(order_id, raw)

    def cancel_order(self, order_id: str) -> OfficialOrderStatus:
        from py_clob_client_v2 import OrderPayload

        try:
            raw = self._client.cancel_order(OrderPayload(orderID=order_id))
        except Exception as exc:  # pragma: no cover - live path
            return OfficialOrderStatus(
                order_id=order_id,
                truth=OrderTruth.OFFICIAL_REJECTED,
                raw={"exception": repr(exc)},
                message="cancelOrder failed",
            )
        return OfficialOrderStatus(
            order_id=order_id,
            truth=OrderTruth.OFFICIAL_CANCELED_ZERO_FILL,
            raw=_safe_raw(raw),
            message="cancel request submitted; follow-up getOrder required for final matched amount",
        )

    def get_orderbook_quote(self, token_id: str, minutes_remaining: float = 10.0) -> OrderbookQuote:
        if self.config.websocket_quote_enabled:
            quote = load_ws_quote(
                token_id,
                minutes_remaining=minutes_remaining,
                cache_path=self.config.websocket_market_cache_path,
                max_age_seconds=self.config.websocket_quote_max_age_seconds,
            )
            if quote is not None:
                return quote
        raw = self._client.get_order_book(token_id)
        return _quote_from_orderbook(raw, minutes_remaining)

    def _reconcile_post_exception(
        self,
        exc: Exception,
        *,
        order_type: str,
        side: str,
        plan: dict[str, Any],
    ) -> OfficialOrderStatus | None:  # pragma: no cover - exercised with fake client tests
        """Resolve V2 SDK exceptions that still include an official order id.

        The CLOB V2 market-order helper can raise on no-match/partial edge cases
        while returning an orderID in the error payload. Treating that as a pure
        rejection is unsafe: activity/balance may already have changed. If an
        order id exists, official getOrder is the truth source.
        """

        order_id = _extract_order_id_from_exception(exc)
        if not order_id:
            return None

        confirmed = self.get_order(order_id)
        raw = {
            "post_exception": repr(exc),
            "order_type": order_type,
            "side": side,
            "plan": plan,
            "follow_up_get_order": confirmed.raw,
        }
        if confirmed.truth != OrderTruth.OFFICIAL_MISSING:
            return OfficialOrderStatus(
                order_id=order_id,
                truth=confirmed.truth,
                raw=raw,
                matched_shares=confirmed.matched_shares,
                message=f"postOrder raised, but orderID reconciled by official getOrder as {confirmed.truth.value}",
            )
        return OfficialOrderStatus(
            order_id=order_id,
            truth=OrderTruth.OFFICIAL_REJECTED,
            raw=raw,
            matched_shares=0.0,
            message="postOrder raised with orderID, but follow-up getOrder could not find an active/matched order",
        )


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"missing required live env var: {name}")
    return value


def _extract_order_id(raw: Any) -> Optional[str]:
    if isinstance(raw, dict):
        for key in ("orderID", "orderId", "order_id", "id", "hash"):
            value = raw.get(key)
            if value:
                return str(value)
        nested = raw.get("order")
        if isinstance(nested, dict):
            return _extract_order_id(nested)
    if isinstance(raw, str):
        return _extract_order_id_from_text(raw)
    return None


def _extract_order_id_from_exception(exc: Exception) -> Optional[str]:
    return _extract_order_id_from_text(repr(exc)) or _extract_order_id_from_text(str(exc))


def _extract_order_id_from_text(text: str) -> Optional[str]:
    # Prefer explicit orderID fields over arbitrary hashes in the message.
    explicit = re.search(r"['\"]?orderID['\"]?\s*[:=]\s*['\"]?(0x[a-fA-F0-9]{32,})", text)
    if explicit:
        return explicit.group(1)
    loose = re.search(r"\borderID\b.*?(0x[a-fA-F0-9]{32,})", text)
    if loose:
        return loose.group(1)
    return None


def _status_from_raw_order(order_id: str, raw: Any) -> OfficialOrderStatus:
    safe = _safe_raw(raw)
    status = str(_dig(safe, "status") or _dig(safe, "state") or "").upper()
    matched = _to_float(
        _dig(safe, "matchedShares")
        or _dig(safe, "matched_shares")
        or _dig(safe, "sizeMatched")
        or _dig(safe, "size_matched")
        or 0.0
    )
    if not raw:
        return OfficialOrderStatus(order_id, OrderTruth.OFFICIAL_MISSING, safe, matched, "empty getOrder")
    if matched > 0:
        return OfficialOrderStatus(order_id, OrderTruth.OFFICIAL_FILLED, safe, matched, "official matched shares > 0")
    if status in {"CANCELED", "CANCELLED", "EXPIRED"}:
        return OfficialOrderStatus(order_id, OrderTruth.OFFICIAL_CANCELED_ZERO_FILL, safe, matched, "official canceled with zero fill")
    if status in {"OPEN", "LIVE", "RESTING"}:
        return OfficialOrderStatus(order_id, OrderTruth.OFFICIAL_OPEN, safe, matched, "official open order")
    return OfficialOrderStatus(order_id, OrderTruth.OFFICIAL_OPEN, safe, matched, "official order exists; status not mapped")


def _quote_from_orderbook(raw: Any, minutes_remaining: float) -> OrderbookQuote:
    safe = _safe_raw(raw)
    bids = _dig(safe, "bids") or []
    asks = _dig(safe, "asks") or []
    best_bid, bid_depth = _best_level(bids, is_bid=True)
    best_ask, ask_depth = _best_level(asks, is_bid=False)
    return OrderbookQuote(
        best_bid=best_bid,
        best_ask=best_ask,
        bid_depth_shares=bid_depth,
        ask_depth_shares=ask_depth,
        minutes_remaining=minutes_remaining,
        source="rest",
    )


def _best_level(levels: Any, *, is_bid: bool) -> tuple[float | None, float]:
    if not isinstance(levels, list):
        return None, 0.0
    parsed: list[tuple[float, float]] = []
    for level in levels:
        if isinstance(level, dict):
            price = _to_float(level.get("price") or level.get("p") or level.get("px"))
            size = _to_float(level.get("size") or level.get("s") or level.get("shares"))
        elif isinstance(level, (list, tuple)) and len(level) >= 2:
            price = _to_float(level[0]); size = _to_float(level[1])
        else:
            continue
        if price > 0 and size > 0:
            parsed.append((price, size))
    if not parsed:
        return None, 0.0
    price, size = (max if is_bid else min)(parsed, key=lambda x: x[0])
    return price, size


def _dig(raw: Any, key: str) -> Any:
    if isinstance(raw, dict):
        if key in raw:
            return raw[key]
        for value in raw.values():
            found = _dig(value, key)
            if found is not None:
                return found
    return None


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _safe_raw(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if hasattr(raw, "__dict__"):
        return dict(raw.__dict__)
    return {"raw_repr": repr(raw)}


def _safe_plan(plan: OrderPlan) -> dict[str, Any]:
    return {
        "symbol": plan.candidate.symbol,
        "period": plan.candidate.period,
        "market_slug": plan.candidate.market_slug,
        "token_id": plan.candidate.token_id,
        "price": plan.price,
        "size_shares": plan.size_shares,
        "notional_usd": plan.notional_usd,
        "cancel_after_seconds": plan.cancel_after_seconds,
    }

# Attach query helpers without changing the minimal OfficialClient protocol.
def _clob_get_balance_allowance(self) -> dict[str, Any]:  # pragma: no cover - live path
    from py_clob_client_v2 import AssetType, BalanceAllowanceParams

    raw = self._client.get_balance_allowance(
        BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
    )
    return _safe_raw(raw)


def _clob_get_open_orders(self) -> list[Any]:  # pragma: no cover - live path
    raw = self._client.get_open_orders()
    return raw if isinstance(raw, list) else [raw]


ClobV2SdkOfficialClient.get_balance_allowance = _clob_get_balance_allowance
ClobV2SdkOfficialClient.get_open_orders = _clob_get_open_orders
