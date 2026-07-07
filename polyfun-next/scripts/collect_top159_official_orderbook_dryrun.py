#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path("/Users/mac/polyfun")
NEXT = ROOT / "polyfun-next"
REPORTS = ROOT / "reports"
RUNTIME = NEXT / "runtime"
DEFAULT_OUT = RUNTIME / "top159_official_orderbook_dryrun.jsonl"
DEFAULT_REPORT = REPORTS / "top159_official_orderbook_dryrun_latest.json"
PID_FILE = RUNTIME / "top159_official_orderbook_dryrun.pid"
GENERATOR = NEXT / "scripts" / "generate_top159_live_candidate.py"

sys.path.insert(0, str(NEXT / "src"))

from polyfun_next.config import load_config  # noqa: E402
from polyfun_next.policy import CanaryPolicy  # noqa: E402
from polyfun_next.types import CandidateSignal, OrderbookQuote  # noqa: E402


def load_generator():
    spec = importlib.util.spec_from_file_location("top159_live_generator_for_dryrun", GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load generator: {GENERATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["top159_live_generator_for_dryrun"] = module
    spec.loader.exec_module(module)
    return module


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime | pd.Timestamp) -> str:
    return pd.Timestamp(dt).tz_convert("UTC").isoformat() if isinstance(dt, pd.Timestamp) else dt.isoformat()


def beijing_iso(dt: datetime) -> str:
    return pd.Timestamp(dt).tz_convert("Asia/Shanghai").isoformat()


def http_json(url: str, timeout: int = 10) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "polyfun-next-top159-protection-dryrun/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_orderbook(token_id: str) -> dict[str, Any]:
    query = urllib.parse.urlencode({"token_id": token_id})
    url = f"https://clob.polymarket.com/book?{query}"
    try:
        raw = http_json(url)
        if isinstance(raw, dict) and (raw.get("bids") is not None or raw.get("asks") is not None):
            return raw
    except Exception as first_exc:
        # Some API builds also accept asset_id. Keep this fallback explicit in the raw report.
        query = urllib.parse.urlencode({"asset_id": token_id})
        fallback_url = f"https://clob.polymarket.com/book?{query}"
        raw = http_json(fallback_url)
        if isinstance(raw, dict):
            raw["_token_id_query_error"] = repr(first_exc)
            raw["_fallback_url"] = fallback_url
        return raw
    return raw if isinstance(raw, dict) else {"raw": raw}


def to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def parse_levels(levels: Any) -> list[dict[str, float]]:
    parsed: list[dict[str, float]] = []
    if not isinstance(levels, list):
        return parsed
    for level in levels:
        if isinstance(level, dict):
            price = to_float(level.get("price") or level.get("p") or level.get("px"))
            size = to_float(level.get("size") or level.get("s") or level.get("shares"))
        elif isinstance(level, (list, tuple)) and len(level) >= 2:
            price = to_float(level[0])
            size = to_float(level[1])
        else:
            continue
        if price is not None and size is not None and price > 0 and size > 0:
            parsed.append({"price": price, "size": size})
    return parsed


def summarize_book(raw: dict[str, Any]) -> dict[str, Any]:
    bids = parse_levels(raw.get("bids"))
    asks = parse_levels(raw.get("asks"))
    best_bid = max((x["price"] for x in bids), default=None)
    best_ask = min((x["price"] for x in asks), default=None)
    bid_depth_top3 = sum(x["size"] for x in sorted(bids, key=lambda x: x["price"], reverse=True)[:3])
    ask_depth_top3 = sum(x["size"] for x in sorted(asks, key=lambda x: x["price"])[:3])
    spread = None
    if best_bid is not None and best_ask is not None:
        spread = best_ask - best_bid
    return {
        "bestBid": best_bid,
        "bestAsk": best_ask,
        "spread": spread,
        "bidDepthTop3Shares": bid_depth_top3,
        "askDepthTop3Shares": ask_depth_top3,
        "bidLevels": len(bids),
        "askLevels": len(asks),
        "minOrderSize": raw.get("min_order_size"),
        "tickSize": raw.get("tick_size"),
        "lastTradePrice": raw.get("last_trade_price"),
        "hash": raw.get("hash"),
        "timestamp": raw.get("timestamp"),
    }


def evaluate_policy(config_path: Path, signal: dict[str, Any], market: dict[str, Any] | None, book: dict[str, Any] | None) -> dict[str, Any]:
    cfg = load_config(config_path)
    policy = CanaryPolicy(cfg)
    if not market or not book:
        return {"evaluated": False, "reasons": ["missing_market_or_orderbook"]}
    summary = summarize_book(book)
    if summary["bestAsk"] is None or summary["bestBid"] is None:
        return {"evaluated": False, "reasons": ["missing_best_bid_or_ask"], "orderbook": summary}
    end = pd.to_datetime(market.get("endDate"), utc=True, errors="coerce")
    minutes_remaining = 10.0
    if not pd.isna(end):
        minutes_remaining = max(0.0, (end - pd.Timestamp(utc_now())).total_seconds() / 60.0)
    candidate = CandidateSignal(
        symbol="ETH",
        period="15m",
        market_slug=str(market.get("market_slug") or market.get("slug") or ""),
        condition_id=str(market.get("condition_id") or market.get("conditionId") or ""),
        token_id=str(market.get("token_id") or ""),
        side=str(signal.get("side") or ""),
        model_score=float(signal.get("model_score") or 0.0),
    )
    quote = OrderbookQuote(
        best_bid=summary["bestBid"],
        best_ask=summary["bestAsk"],
        bid_depth_shares=float(summary["bidDepthTop3Shares"] or 0.0),
        ask_depth_shares=float(summary["askDepthTop3Shares"] or 0.0),
        minutes_remaining=minutes_remaining,
    )
    reasons = policy.entry_rejection_reasons(candidate, quote, current_funds_usd=cfg.base_capital_usd)
    plan = None if reasons else policy.build_order_plan(candidate, quote, current_funds_usd=cfg.base_capital_usd)
    return {
        "evaluated": True,
        "accepted": not reasons,
        "reasons": reasons,
        "orderbook": summary,
        "notionalUsd": policy.order_notional(cfg.base_capital_usd),
        "plannedPrice": plan.price if plan else None,
        "plannedShares": plan.size_shares if plan else None,
        "valueEdge": (float(signal.get("model_score") or 0.0) - float(summary["bestAsk"])) if summary["bestAsk"] is not None else None,
    }


def collect_once(config_path: Path, out_path: Path, report_path: Path, edge_override: float | None = None) -> dict[str, Any]:
    gen = load_generator()
    base = gen._load_module("crypto_search_dryrun_once", gen.BASE_SCRIPT)
    fill = gen._load_module("fill_search_dryrun_once", gen.FILL_SCRIPT)
    params = gen.top159_params()
    if edge_override is not None:
        params = dict(params)
        params["edge"] = edge_override
        params["edge_override_reason"] = "official_orderbook_dryrun_only"
    signal = gen.current_top159_signal(base, fill, params)
    market = gen.find_eth15m_market(signal["candidate_start"], signal["side"])
    now = utc_now()
    start_ts = pd.Timestamp(signal["candidate_start"])
    elapsed = (pd.Timestamp(now) - start_ts).total_seconds()
    entry_window = {"elapsedSeconds": elapsed, "start": 30, "end": 180, "allowed": 30 <= elapsed <= 180}
    book = None
    book_error = None
    if market and market.get("token_id"):
        try:
            book = fetch_orderbook(str(market["token_id"]))
        except Exception as exc:
            book_error = repr(exc)
    policy_eval = evaluate_policy(config_path, signal, market, book) if book else {"evaluated": False, "reasons": ["orderbook_fetch_failed"], "error": book_error}
    row = {
        "observedAt": now.isoformat(),
        "observedAtBeijing": beijing_iso(now),
        "source": "official_current_clob_orderbook_dryrun",
        "trading": "none",
        "configPath": str(config_path),
        "signal": {
            "candidateStart": signal.get("candidate_start"),
            "probUp": signal.get("prob_up"),
            "side": signal.get("side"),
            "modelScore": signal.get("model_score"),
            "params": signal.get("params"),
            "featureCount": signal.get("feature_count"),
            "trainRows": signal.get("train_rows"),
            "modelCache": signal.get("model_cache"),
        },
        "market": market,
        "entryWindow": entry_window,
        "orderbook": summarize_book(book) if book else None,
        "orderbookRawError": book_error,
        "policyEvaluation": policy_eval,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(row, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return row


def main() -> int:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    parser = argparse.ArgumentParser(description="Collect official current CLOB orderbook samples for top159 protection-price research. No trading.")
    parser.add_argument("--config", default=str(NEXT / "config" / "canary.eth15m.json"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--edge-override", type=float, default=None)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=25)
    args = parser.parse_args()
    while True:
        row = collect_once(Path(args.config), Path(args.out), Path(args.report), edge_override=args.edge_override)
        print(json.dumps({"ok": True, "observedAtBeijing": row["observedAtBeijing"], "accepted": row["policyEvaluation"].get("accepted"), "reasons": row["policyEvaluation"].get("reasons")}, ensure_ascii=False))
        if not args.loop:
            return 0
        time.sleep(max(5, args.poll_seconds))


if __name__ == "__main__":
    raise SystemExit(main())
