from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from polyfun_next.ws_cache import load_ws_quote


def write_cache(path: Path, **kwargs):
    payload = {
        "status": "ready",
        "token_id": "token-a",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "best_bid": 0.49,
        "best_ask": 0.51,
        "bid_depth_shares": 100.0,
        "ask_depth_shares": 120.0,
        **kwargs,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_loads_fresh_matching_websocket_quote(tmp_path: Path):
    cache = tmp_path / "ws.json"
    write_cache(cache)
    quote = load_ws_quote("token-a", minutes_remaining=13.5, cache_path=str(cache), max_age_seconds=3.0)
    assert quote is not None
    assert quote.source == "websocket"
    assert quote.best_ask == 0.51
    assert quote.ask_depth_shares == 120.0


def test_ignores_stale_websocket_quote(tmp_path: Path):
    cache = tmp_path / "ws.json"
    write_cache(cache, updated_at=(datetime.now(timezone.utc) - timedelta(seconds=20)).isoformat())
    assert load_ws_quote("token-a", minutes_remaining=13.5, cache_path=str(cache), max_age_seconds=3.0) is None


def test_ignores_wrong_token_websocket_quote(tmp_path: Path):
    cache = tmp_path / "ws.json"
    write_cache(cache, token_id="token-b")
    assert load_ws_quote("token-a", minutes_remaining=13.5, cache_path=str(cache), max_age_seconds=3.0) is None


def test_ignores_crossed_or_incomplete_book(tmp_path: Path):
    cache = tmp_path / "ws.json"
    write_cache(cache, best_bid=0.52, best_ask=0.51)
    assert load_ws_quote("token-a", minutes_remaining=13.5, cache_path=str(cache), max_age_seconds=3.0) is None
