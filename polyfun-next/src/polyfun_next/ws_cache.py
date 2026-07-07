from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .types import OrderbookQuote

DEFAULT_ROOT = Path(os.environ.get("POLYFUN_NEXT_ROOT", "/Users/mac/polyfun/polyfun-next"))
DEFAULT_CACHE_PATH = DEFAULT_ROOT / "runtime" / "top159_ws_market_cache.json"


def parse_utc(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _float_or_none(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if out > 0 else None


def _cache_path(path: str | None = None) -> Path:
    return Path(path).expanduser() if path else DEFAULT_CACHE_PATH


def load_ws_quote(
    token_id: str,
    *,
    minutes_remaining: float,
    cache_path: str | None = None,
    max_age_seconds: float = 3.0,
    now: datetime | None = None,
) -> OrderbookQuote | None:
    """Return a fresh WebSocket orderbook quote for token_id, or None.

    This is intentionally strict: wrong token, stale timestamps, missing bid/ask,
    or zero depth all fall back to REST. The websocket cache is an acceleration
    layer, not a truth replacement.
    """
    path = _cache_path(cache_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if str(payload.get("token_id") or payload.get("asset_id") or "") != str(token_id):
        return None
    if str(payload.get("status") or "").lower() not in {"ready", "ok"}:
        return None
    updated = parse_utc(payload.get("updated_at") or payload.get("last_message_at"))
    if updated is None:
        return None
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    age = (now.astimezone(timezone.utc) - updated).total_seconds()
    if age < 0 or age > float(max_age_seconds):
        return None
    best_bid = _float_or_none(payload.get("best_bid"))
    best_ask = _float_or_none(payload.get("best_ask"))
    bid_depth = _float_or_none(payload.get("bid_depth_shares"))
    ask_depth = _float_or_none(payload.get("ask_depth_shares"))
    if best_bid is None or best_ask is None or bid_depth is None or ask_depth is None:
        return None
    if best_bid >= best_ask:
        return None
    return OrderbookQuote(
        best_bid=best_bid,
        best_ask=best_ask,
        bid_depth_shares=bid_depth,
        ask_depth_shares=ask_depth,
        minutes_remaining=minutes_remaining,
        source="websocket",
        source_age_seconds=round(age, 3),
        source_updated_at=updated.isoformat(),
    )


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
