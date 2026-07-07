#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.optimize_trading_rules import generate_predictions
import numba

POLY = PROJECT_ROOT / "polymarket"
REPORTS = PROJECT_ROOT / "reports"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "sentiment_grid_search" / "results" / "exp10_lowprice_compare"
CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "exp10_lowprice_compare"
TRADER_CONFIGS = POLY / "trader_configs.json"
HISTORICAL_EXECUTION_BOOTSTRAP_REPORT = PROJECT_ROOT / "reports" / "vnext_execution_v2_historical_bootstrap_latest.json"
EXECUTION_CALIBRATION_REPORT = REPORTS / "exp10_lowprice_execution_calibration_latest.json"
BASE_TRADER_NAME = "v5_exp10_bp0470"
INITIAL_CAPITAL = 400.0
SETTLEMENT_COST = 0.015 + 0.003
LOW_PRICE = 0.30
HIGH_PRICE = 0.40
FIXED_STEP = 0.005
TOTAL_DAYS = 180
HOLDOUT_DAYS = 15
QUEUE_CALIBRATION_MAX_BOOK_ROWS = 50000
ASSET_ROWS = {
    "BTC": {
        "asset_name": "BTC_USDT",
        "model_dir": PROJECT_ROOT / "data" / "models" / "v5_core10_v5_exp10_bp0470_btc",
        "source_prediction_suffix": "_core10_v5_exp10_bp0470_btc",
        "source_prediction_file": POLY / "predictions_core10_v5_exp10_bp0470_btc.json",
        "out_prediction_suffix": "_exp10_lowprice_btc",
        "out_prediction_file": POLY / "predictions_exp10_lowprice_btc.json",
        "path_file": PROJECT_ROOT / "data" / "polymarket_1m_btc_usdt.parquet",
        "orderbook_file": PROJECT_ROOT / "data" / "processed" / "vnext_execution_orderbook_btc_usdt.jsonl",
    },
    "ETH": {
        "asset_name": "ETH_USDT",
        "model_dir": PROJECT_ROOT / "data" / "models" / "v5_core10_v5_exp10_bp0470_eth",
        "source_prediction_suffix": "_core10_v5_exp10_bp0470_eth",
        "source_prediction_file": POLY / "predictions_core10_v5_exp10_bp0470_eth.json",
        "out_prediction_suffix": "_exp10_lowprice_eth",
        "out_prediction_file": POLY / "predictions_exp10_lowprice_eth.json",
        "path_file": PROJECT_ROOT / "data" / "polymarket_1m_eth_usdt.parquet",
        "orderbook_file": PROJECT_ROOT / "data" / "processed" / "vnext_execution_orderbook_eth_usdt.jsonl",
    },
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _round(x: float | None, n: int = 6) -> float | None:
    if x is None or not math.isfinite(float(x)):
        return None
    return round(float(x), n)


def _load_base_trader() -> Dict[str, Any]:
    rows = json.loads(TRADER_CONFIGS.read_text(encoding="utf-8"))
    row = next((r for r in rows if r.get("name") == BASE_TRADER_NAME), None)
    if not isinstance(row, dict):
        raise RuntimeError(f"base trader not found: {BASE_TRADER_NAME}")
    return row


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _runtime_params_from_cfg(cfg: Dict[str, Any]) -> Dict[str, float]:
    return {
        "min_confidence": float(cfg["probThreshold"]),
        "min_edge": float(cfg["minEdge"]),
        "kelly_frac": float(cfg["kellyFrac"]),
        "bet_pct_normal": float(cfg["betPctNormal"]),
        "bet_pct_conservative": float(cfg["betPctConservative"]),
        "conf_tier1_bound": float(cfg["confTier1Bound"]),
        "conf_tier2_bound": float(cfg["confTier2Bound"]),
        "tier1_mult": float(cfg["tier1Mult"]),
        "tier2_mult": float(cfg["tier2Mult"]),
        "tier3_mult": float(cfg["tier3Mult"]),
        "cooldown_bars": int(cfg["cooldownBars"]),
        "drawdown_halt": float(cfg["drawdownHalt"]),
    }


def _build_rules_payload(base_cfg: Dict[str, Any], mode: str, buy_price: float | None, buy_price_range: list[float] | None, extra: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        "trading_rules": {
            "min_confidence": float(base_cfg["probThreshold"]),
            "min_edge": float(base_cfg["minEdge"]),
            "kelly_frac": float(base_cfg["kellyFrac"]),
            "bet_pct_normal": float(base_cfg["betPctNormal"]),
            "bet_pct_conservative": float(base_cfg["betPctConservative"]),
            "confidence_tiers": [
                [float(base_cfg["probThreshold"]), float(base_cfg["confTier1Bound"]), float(base_cfg["tier1Mult"])],
                [float(base_cfg["confTier1Bound"]), float(base_cfg["confTier2Bound"]), float(base_cfg["tier2Mult"])],
                [float(base_cfg["confTier2Bound"]), 1.0, float(base_cfg["tier3Mult"])],
            ],
            "cooldown_bars": int(base_cfg["cooldownBars"]),
            "drawdown_halt": float(base_cfg["drawdownHalt"]),
        },
        "polymarket_constraints": {
            "buy_price": float(buy_price if buy_price is not None else sum(buy_price_range or [LOW_PRICE, HIGH_PRICE]) / 2.0),
            "buy_price_range": buy_price_range,
            "odds": _round((1.0 - float(buy_price if buy_price is not None else sum(buy_price_range or [LOW_PRICE, HIGH_PRICE]) / 2.0)) / float(buy_price if buy_price is not None else sum(buy_price_range or [LOW_PRICE, HIGH_PRICE]) / 2.0), 4),
            "fee_rate": 0.015,
            "slippage_rate": 0.003,
            "liquidity_cap": 60000,
            "liquidity_bet": 3000,
        },
        "search_config": {
            "method": "bp0470_field_locked_lowprice_selection",
            "days": TOTAL_DAYS,
            "holdout_days": HOLDOUT_DAYS,
            "fixed_grid_step": FIXED_STEP,
            "selection_mode": mode,
        },
        "optimized_at": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "metadata": {
            "base_trader_name": BASE_TRADER_NAME,
            "base_prediction_family": "exp10_bp0470_default",
            "selection_mode": mode,
            **extra,
        },
    }
    return payload


def _confidence_from_proba(proba: np.ndarray) -> np.ndarray:
    return np.maximum(proba, 1.0 - proba)


def _dynamic_limit_prices(probas: np.ndarray, threshold: float, low: float, high: float) -> np.ndarray:
    conf = _confidence_from_proba(probas)
    denom = max(1e-9, 1.0 - threshold)
    scale = np.clip((conf - threshold) / denom, 0.0, 1.0)
    prices = low + (high - low) * scale
    return np.round(prices.astype(np.float64), 4)


def _asset_name(asset: str) -> str:
    return str(ASSET_ROWS[asset]["asset_name"])


def _load_execution_bootstrap_defaults(asset: str) -> Dict[str, float]:
    payload = _load_json(HISTORICAL_EXECUTION_BOOTSTRAP_REPORT)
    defaults = (((payload or {}).get("assets") or {}).get(_asset_name(asset)) or {}).get("bootstrap_defaults") or {}
    return {
        "fill_rate": float(defaults.get("fill_rate") or 0.92),
        "partial_fill_rate": float(defaults.get("partial_fill_rate") or 0.08),
        "timeout_rate": float(defaults.get("timeout_rate") or max(0.0, 1.0 - float(defaults.get("fill_rate") or 0.92))),
        "avg_partial_fill_ratio": float(defaults.get("avg_partial_fill_ratio") or 0.48),
        "avg_queue_wait_seconds": float(defaults.get("avg_queue_wait_seconds") or 22.0),
    }


def _normalize_book_side(levels: Iterable[dict[str, Any]] | None) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    for level in levels or []:
        try:
            price = float(level.get("price"))
        except Exception:
            continue
        if not math.isfinite(price) or price <= 0:
            continue
        size_usd = level.get("sizeUsd")
        if size_usd is None:
            try:
                shares = float(level.get("sizeShares") or level.get("size") or 0.0)
                size_usd = shares * price
            except Exception:
                size_usd = 0.0
        try:
            size_usd_f = float(size_usd)
        except Exception:
            continue
        if not math.isfinite(size_usd_f) or size_usd_f <= 0:
            continue
        out.append({"price": price, "sizeUsd": size_usd_f})
    return out


def _calc_weighted_avg_buy_fill(asks: list[dict[str, float]], limit_price: float, request_usd: float) -> tuple[float, float]:
    sorted_asks = sorted((a for a in asks if a["price"] <= limit_price and a["sizeUsd"] > 0), key=lambda x: x["price"])
    remaining = float(request_usd)
    total_cost = 0.0
    total_shares = 0.0
    for ask in sorted_asks:
        if remaining <= 0:
            break
        consume = min(float(ask["sizeUsd"]), remaining)
        total_cost += consume
        total_shares += consume / float(ask["price"])
        remaining -= consume
    avg_price = (total_cost / total_shares) if total_shares > 0 else float(limit_price)
    avg_price = min(max(avg_price, 0.01), float(limit_price))
    return avg_price, max(0.0, total_cost)


def _estimate_bid_competition_usd(bids: list[dict[str, float]], limit_price: float) -> float:
    return float(sum(float(b["sizeUsd"]) for b in bids if float(b["price"]) >= limit_price))


def _simulate_queue_aware_buy_fill(asks: list[dict[str, float]], bids: list[dict[str, float]], limit_price: float, request_usd: float) -> Dict[str, float | str]:
    avg_price, raw_filled_usd = _calc_weighted_avg_buy_fill(asks, limit_price, request_usd)
    queue_competition_usd = _estimate_bid_competition_usd(bids, limit_price)
    effective_fillable_usd = max(0.0, raw_filled_usd - queue_competition_usd)
    filled_usd = min(float(request_usd), effective_fillable_usd)
    if filled_usd >= request_usd:
        fill_status = "filled"
    elif filled_usd > 0:
        fill_status = "partial_fill"
    elif raw_filled_usd > 0:
        fill_status = "queued"
    else:
        fill_status = "timeout_unfilled"
    return {
        "fill_status": fill_status,
        "filled_usd": filled_usd,
        "avg_fill_price": avg_price,
        "queue_competition_usd": queue_competition_usd,
        "partial_fill_ratio": (filled_usd / request_usd) if request_usd > 0 else 0.0,
    }


def _load_recent_book_rows(asset: str, max_rows: int = QUEUE_CALIBRATION_MAX_BOOK_ROWS) -> list[dict[str, Any]]:
    path = Path(ASSET_ROWS[asset]["orderbook_file"])
    if not path.exists():
        return []
    rows: deque[dict[str, Any]] = deque(maxlen=max_rows)
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("event_type") != "book":
                continue
            asks_top = row.get("asks_top") or []
            bids_top = row.get("bids_top") or []
            if not asks_top or not bids_top:
                continue
            rows.append(row)
    return list(rows)


def _build_execution_calibration(asset: str, base_params: Dict[str, float], low: float, high: float, step: float) -> Dict[str, Any]:
    defaults = _load_execution_bootstrap_defaults(asset)
    request_usd = max(1.0, INITIAL_CAPITAL * float(base_params["bet_pct_normal"]))
    book_rows = _load_recent_book_rows(asset)
    prices = []
    x = float(low)
    while x <= float(high) + 1e-9:
        prices.append(round(x, 4))
        x += float(step)

    price_rows: dict[str, Any] = {}
    for price in prices:
        best_ask_touch = 0
        valid_best_ask = 0
        for row in book_rows:
            best_ask = row.get("best_ask")
            try:
                best_ask_f = float(best_ask)
            except Exception:
                continue
            if not math.isfinite(best_ask_f) or best_ask_f <= 0:
                continue
            valid_best_ask += 1
            if best_ask_f <= price:
                best_ask_touch += 1
        sample_count = valid_best_ask
        best_ask_touch_rate = (best_ask_touch / valid_best_ask) if valid_best_ask > 0 else 0.0
        fill_rate = float(defaults["fill_rate"])
        partial_fill_rate = float(defaults["partial_fill_rate"])
        timeout_rate = float(defaults["timeout_rate"])
        avg_partial_fill_ratio = float(defaults["avg_partial_fill_ratio"])
        avg_queue_wait_seconds = float(defaults["avg_queue_wait_seconds"])
        expected_fill_ratio = fill_rate * ((1.0 - partial_fill_rate) + partial_fill_rate * avg_partial_fill_ratio)
        price_rows[f"{price:.4f}"] = {
            "price": price,
            "sample_count": int(sample_count),
            "recent_best_ask_touch_rate": _round(best_ask_touch_rate, 6),
            "queued_rate": _round(max(0.0, 1.0 - fill_rate - timeout_rate), 6),
            "fill_rate": _round(fill_rate, 6),
            "partial_fill_rate": _round(partial_fill_rate, 6),
            "timeout_rate": _round(timeout_rate, 6),
            "avg_partial_fill_ratio": _round(avg_partial_fill_ratio, 6),
            "avg_queue_wait_seconds": _round(avg_queue_wait_seconds, 6),
            "expected_fill_ratio": _round(expected_fill_ratio, 6),
        }

    return {
        "asset": asset,
        "source": "historical_execution_bootstrap_with_recent_best_ask_touch_audit",
        "request_usd": _round(request_usd, 6),
        "snapshot_count": int(len(book_rows)),
        "defaults": {
            k: _round(v, 6) for k, v in defaults.items()
        },
        "rows": price_rows,
    }


def _build_execution_arrays(price_rows: Dict[str, Any]) -> Dict[str, np.ndarray]:
    price_points = np.array(sorted(float(k) for k in price_rows.keys()), dtype=np.float64)
    out: Dict[str, np.ndarray] = {"price_points": price_points}
    for field in ("fill_rate", "partial_fill_rate", "timeout_rate", "avg_partial_fill_ratio", "avg_queue_wait_seconds", "expected_fill_ratio"):
        out[field] = np.array([float(price_rows[f"{p:.4f}"][field]) for p in price_points], dtype=np.float64)
    return out


def _interp_metric(exec_arrays: Dict[str, np.ndarray], prices: np.ndarray, field: str) -> np.ndarray:
    return np.interp(prices, exec_arrays["price_points"], exec_arrays[field], left=exec_arrays[field][0], right=exec_arrays[field][-1])





def _load_path_touch_map(asset: str) -> pd.DataFrame:
    path_file = ASSET_ROWS[asset]["path_file"]
    if not Path(path_file).exists():
        raise RuntimeError(f"missing path file for {asset}: {path_file}")
    df = pd.read_parquet(path_file, columns=["15m_start_ts", "t_sec", "p"])
    df = df.dropna(subset=["15m_start_ts", "t_sec", "p"]).copy()
    df["15m_start_ts"] = pd.to_numeric(df["15m_start_ts"], errors="coerce").astype("int64")
    df["t_sec"] = pd.to_numeric(df["t_sec"], errors="coerce").astype("int64")
    df["p"] = pd.to_numeric(df["p"], errors="coerce").astype(float)
    # Conservative fillability gate: only count touches from market start onward, not the pre-start replay window.
    df = df[df["t_sec"] >= df["15m_start_ts"]].copy()
    out = df.groupby("15m_start_ts", as_index=False).agg(
        path_min_price=("p", "min"),
        path_max_price=("p", "max"),
        path_obs=("p", "size"),
    )
    return out


def _attach_path_touch(predictions: pd.DataFrame, asset: str) -> pd.DataFrame:
    out = predictions.copy()
    ts = pd.to_datetime(out["timestamp"], utc=True)
    out["market_start_ts"] = (ts.astype("int64") // 10**9).astype("int64")
    touch = _load_path_touch_map(asset)
    out = out.merge(touch, how="left", left_on="market_start_ts", right_on="15m_start_ts")
    out["path_has_coverage"] = out["path_min_price"].notna()
    out["path_min_price"] = pd.to_numeric(out["path_min_price"], errors="coerce")
    out["path_max_price"] = pd.to_numeric(out["path_max_price"], errors="coerce")
    out["path_obs"] = pd.to_numeric(out["path_obs"], errors="coerce").fillna(0).astype(int)
    return out


@numba.njit(cache=True)
def _simulate_trading_core_fillmask(
    probas, actuals, odds_arr, fill_ratio_arr, fill_trade_weight_arr,
    initial_capital, settlement_cost,
    min_conf, min_edge, kelly_frac,
    bet_pct_normal, bet_pct_conservative,
    tier1_bound, tier2_bound,
    tier1_mult, tier2_mult, tier3_mult,
    cooldown, dd_halt,
    liquidity_cap, liquidity_bet,
    halt_duration_bars, recovery_target,
):
    n = len(probas)
    pnl_arr = np.zeros(n, dtype=np.float64)
    capital = initial_capital
    peak_capital = initial_capital
    max_drawdown = 0.0
    wins = 0.0
    losses = 0.0
    trade_equiv = 0.0
    cooldown_remaining = 0
    halted = False
    halt_start_bar = 0
    ever_reached_cap = False
    for i in range(n):
        proba = probas[i]
        actual = actuals[i]
        dd = (peak_capital - capital) / peak_capital if peak_capital > 0.0 else 0.0
        if halted:
            bars_since_halt = i - halt_start_bar
            capital_recovered = capital >= peak_capital * recovery_target
            time_expired = bars_since_halt >= halt_duration_bars
            if capital_recovered or time_expired:
                halted = False
            else:
                continue
        if dd >= dd_halt:
            halted = True
            halt_start_bar = i
            continue
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue
        if proba >= 0.5:
            direction = 1
            confidence = proba
        else:
            direction = 0
            confidence = 1.0 - proba
        if confidence < min_conf:
            continue
        odds_this = odds_arr[i]
        p = confidence
        q = 1.0 - p
        edge = p * odds_this - q
        if edge < min_edge:
            continue
        realized_fill_ratio = fill_ratio_arr[i]
        trade_weight = fill_trade_weight_arr[i]
        if realized_fill_ratio <= 0.0 or trade_weight <= 0.0:
            continue
        if odds_this > 0.0:
            kelly_f = (p * odds_this - q) / odds_this
        else:
            kelly_f = 0.0
        if kelly_f < 0.0:
            kelly_f = 0.0
        bet_ratio = kelly_f * kelly_frac
        if confidence < tier1_bound:
            bet_ratio *= tier1_mult
        elif confidence < tier2_bound:
            bet_ratio *= tier2_mult
        else:
            bet_ratio *= tier3_mult
        if capital >= liquidity_cap:
            ever_reached_cap = True
            bet_amount = liquidity_bet
        elif ever_reached_cap:
            if bet_ratio > bet_pct_conservative:
                bet_ratio = bet_pct_conservative
            bet_amount = capital * bet_ratio
        else:
            if bet_ratio > bet_pct_normal:
                bet_ratio = bet_pct_normal
            bet_amount = capital * bet_ratio
        cap_limit = capital * 0.95
        if bet_amount > cap_limit:
            bet_amount = cap_limit
        if bet_amount < 1.0:
            continue
        correct = (direction == 1 and actual == 1) or (direction == 0 and actual == 0)
        realized_notional = bet_amount * realized_fill_ratio
        if realized_notional <= 0.0:
            continue
        if correct:
            pnl = realized_notional * odds_this - realized_notional * settlement_cost
            wins += trade_weight
            cooldown_remaining = 0
        else:
            pnl = -realized_notional
            losses += trade_weight
            if cooldown > 0:
                cooldown_remaining = cooldown
        trade_equiv += trade_weight
        capital += pnl
        pnl_arr[i] = pnl
        if capital > peak_capital:
            peak_capital = capital
        dd = (peak_capital - capital) / peak_capital if peak_capital > 0.0 else 0.0
        if dd > max_drawdown:
            max_drawdown = dd
        if capital <= 0.0:
            capital = 0.0
            break
    return capital, wins, losses, trade_equiv, max_drawdown, ever_reached_cap, pnl_arr


def _summarize(predictions: pd.DataFrame, pnl_arr: np.ndarray, final_capital: float, wins: float, losses: float, trade_equiv: float, max_drawdown: float, ever_reached_cap: bool) -> Dict[str, Any]:
    n_trades_equiv = float(max(0.0, trade_equiv))
    n_trades = int(round(n_trades_equiv))
    total_pnl = float(final_capital - INITIAL_CAPITAL)
    win_rate = float(wins / n_trades_equiv) if n_trades_equiv > 1e-9 else 0.0
    nonzero = pnl_arr[pnl_arr != 0]
    if len(nonzero) > 2:
        sharpe = float(nonzero.mean() / (nonzero.std() + 1e-10) * np.sqrt(252 * 96))
    else:
        sharpe = 0.0
    gross_profit = float(nonzero[nonzero > 0].sum()) if len(nonzero) else 0.0
    gross_loss = abs(float(nonzero[nonzero < 0].sum())) if len(nonzero) else 0.0
    profit_factor = gross_profit / (gross_loss + 1e-10) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)
    avg_expectancy = float(total_pnl / n_trades_equiv) if n_trades_equiv > 1e-9 else 0.0
    abstain_rate = float(1.0 - (n_trades_equiv / max(1.0, float(len(predictions)))))
    return {
        "final_capital": _round(final_capital, 6),
        "total_pnl": _round(total_pnl, 6),
        "n_trades": n_trades,
        "trade_equiv": _round(n_trades_equiv, 6),
        "wins": int(round(wins)),
        "losses": int(round(losses)),
        "wins_equiv": _round(wins, 6),
        "losses_equiv": _round(losses, 6),
        "win_rate": _round(win_rate, 6),
        "max_drawdown": _round(max_drawdown, 6),
        "profit_factor": _round(profit_factor, 6),
        "avg_expectancy": _round(avg_expectancy, 6),
        "sharpe": _round(sharpe, 6),
        "abstain_rate": _round(abstain_rate, 6),
        "ever_reached_cap": bool(ever_reached_cap),
    }


def _simulate_with_price_array(predictions: pd.DataFrame, params: Dict[str, float], prices: np.ndarray, exec_arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    probas = np.ascontiguousarray(predictions["proba_up"].values, dtype=np.float64)
    actuals = np.ascontiguousarray(predictions["actual"].values, dtype=np.int64)
    odds_arr = np.ascontiguousarray((1.0 - prices) / prices, dtype=np.float64)
    path_min = pd.to_numeric(predictions.get("path_min_price"), errors="coerce")
    path_touch = np.ascontiguousarray((path_min.values <= prices) & np.isfinite(path_min.values), dtype=np.bool_)
    expected_fill_ratio = _interp_metric(exec_arrays, prices, "expected_fill_ratio")
    fill_rate = _interp_metric(exec_arrays, prices, "fill_rate")
    realized_fill_ratio = np.ascontiguousarray(expected_fill_ratio * path_touch.astype(np.float64), dtype=np.float64)
    trade_weight_arr = np.ascontiguousarray(fill_rate * path_touch.astype(np.float64), dtype=np.float64)
    num_assets = int(predictions["asset"].nunique()) if "asset" in predictions.columns else 1
    cooldown = int(params["cooldown_bars"]) * max(1, num_assets)
    final_capital, wins, losses, trade_equiv, max_drawdown, ever_reached_cap, pnl_arr = _simulate_trading_core_fillmask(
        probas,
        actuals,
        odds_arr,
        realized_fill_ratio,
        trade_weight_arr,
        INITIAL_CAPITAL,
        SETTLEMENT_COST,
        float(params["min_confidence"]),
        float(params["min_edge"]),
        float(params["kelly_frac"]),
        float(params["bet_pct_normal"]),
        float(params["bet_pct_conservative"]),
        float(params["conf_tier1_bound"]),
        float(params["conf_tier2_bound"]),
        float(params["tier1_mult"]),
        float(params["tier2_mult"]),
        float(params["tier3_mult"]),
        cooldown,
        float(params["drawdown_halt"]),
        60000.0,
        3000.0,
        96,
        0.85,
    )
    out = _summarize(predictions, pnl_arr, float(final_capital), float(wins), float(losses), float(trade_equiv), float(max_drawdown), bool(ever_reached_cap))
    out["pathCoverageRate"] = _round(float(np.mean(np.isfinite(path_min.values))) if len(predictions) else 0.0, 6)
    out["fillableRate"] = _round(float(np.mean(path_touch)) if len(path_touch) else 0.0, 6)
    out["executionAdjustedFillRate"] = _round(float(np.mean(trade_weight_arr)) if len(trade_weight_arr) else 0.0, 6)
    out["executionAdjustedFillRatio"] = _round(float(np.mean(realized_fill_ratio)) if len(realized_fill_ratio) else 0.0, 6)
    return out


def _simulate_fixed(predictions: pd.DataFrame, params: Dict[str, float], buy_price: float, exec_arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    prices = np.full(len(predictions), float(buy_price), dtype=np.float64)
    out = _simulate_with_price_array(predictions, params, prices, exec_arrays)
    out["buy_price"] = _round(buy_price, 4)
    return out


def _simulate_dynamic(predictions: pd.DataFrame, params: Dict[str, float], low: float, high: float, exec_arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    probas = np.ascontiguousarray(predictions["proba_up"].values, dtype=np.float64)
    prices = _dynamic_limit_prices(probas, float(params["min_confidence"]), low, high)
    out = _simulate_with_price_array(predictions, params, prices, exec_arrays)
    out["buy_price_range"] = [_round(low, 4), _round(high, 4)]
    out["dynamic_price_min"] = _round(float(prices.min()) if len(prices) else low, 4)
    out["dynamic_price_max"] = _round(float(prices.max()) if len(prices) else high, 4)
    out["dynamic_price_mean"] = _round(float(prices.mean()) if len(prices) else (low + high) / 2.0, 4)
    return out


def _train_best_fixed(train_df: pd.DataFrame, params: Dict[str, float], low: float, high: float, step: float, exec_arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
    candidates: list[Dict[str, Any]] = []
    x = low
    while x <= high + 1e-9:
        res = _simulate_fixed(train_df, params, round(x, 4), exec_arrays)
        candidates.append(res)
        x += step
    candidates.sort(key=lambda r: (float(r["total_pnl"]), -float(r["max_drawdown"]), float(r["avg_expectancy"]), int(r["n_trades"]), -float(r["buy_price"])), reverse=True)
    return {"best": candidates[0], "all": candidates}


def _split_train_holdout(df: pd.DataFrame, holdout_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("timestamp").reset_index(drop=True)
    cutoff = pd.to_datetime(df["timestamp"]).max() - pd.Timedelta(days=holdout_days)
    train_df = df[pd.to_datetime(df["timestamp"]) < cutoff].copy().reset_index(drop=True)
    holdout_df = df[pd.to_datetime(df["timestamp"]) >= cutoff].copy().reset_index(drop=True)
    return train_df, holdout_df


def _candidate_passes(full_metrics: Dict[str, Any], holdout_metrics: Dict[str, Any]) -> bool:
    full_trade_equiv = float(full_metrics.get("trade_equiv") or full_metrics["n_trades"] or 0.0)
    holdout_trade_equiv = float(holdout_metrics.get("trade_equiv") or holdout_metrics["n_trades"] or 0.0)
    return (
        full_trade_equiv >= 30.0
        and holdout_trade_equiv >= 8.0
        and float(full_metrics.get("total_pnl") or 0.0) > 0.0
        and float(holdout_metrics.get("total_pnl") or 0.0) > 0.0
        and float(full_metrics.get("executionAdjustedFillRatio") or 0.0) > 0.0
        and float(full_metrics.get("pathCoverageRate") or 0.0) >= 0.80
    )


def _choose_winner(fixed_holdout: Dict[str, Any], dynamic_holdout: Dict[str, Any], fixed_full: Dict[str, Any], dynamic_full: Dict[str, Any]) -> Dict[str, Any]:
    fixed_ok = _candidate_passes(fixed_full, fixed_holdout)
    dynamic_ok = _candidate_passes(dynamic_full, dynamic_holdout)
    if fixed_ok and not dynamic_ok:
        return {"winner": "fixed-opt", "reason": "dynamic_failed_minimum_gates"}
    if dynamic_ok and not fixed_ok:
        return {"winner": "dynamic-band", "reason": "fixed_failed_minimum_gates"}
    if not fixed_ok and not dynamic_ok:
        return {"winner": None, "reason": "both_failed_minimum_gates"}

    pnl_diff = float(fixed_holdout["total_pnl"]) - float(dynamic_holdout["total_pnl"])
    if abs(pnl_diff) > 8.0:
        return {"winner": "fixed-opt" if pnl_diff > 0 else "dynamic-band", "reason": "holdout_total_pnl"}
    dd_diff = float(fixed_holdout["max_drawdown"]) - float(dynamic_holdout["max_drawdown"])
    if abs(dd_diff) > 1e-9:
        return {"winner": "fixed-opt" if dd_diff < 0 else "dynamic-band", "reason": "holdout_max_drawdown_tiebreak"}
    exp_diff = float(fixed_holdout["avg_expectancy"]) - float(dynamic_holdout["avg_expectancy"])
    if abs(exp_diff) > 1e-9:
        return {"winner": "fixed-opt" if exp_diff > 0 else "dynamic-band", "reason": "holdout_avg_expectancy_tiebreak"}
    trade_diff = int(fixed_holdout["n_trades"]) - int(dynamic_holdout["n_trades"])
    if trade_diff != 0:
        return {"winner": "fixed-opt" if trade_diff > 0 else "dynamic-band", "reason": "holdout_trade_count_tiebreak"}
    return {"winner": "fixed-opt", "reason": "default_fixed_tie"}


def _prediction_cache_path(asset: str, days: int) -> Path:
    return CACHE_DIR / f"exp10_lowprice_predictions_{asset.lower()}_{days}d.parquet"


def _load_or_generate_predictions(asset: str, model_dir: Path, days: int, refresh: bool) -> pd.DataFrame:
    cache_path = _prediction_cache_path(asset, days)
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)
    df = generate_predictions(model_dir=model_dir, apply_noise=True, test_days=days)
    asset_name = ASSET_ROWS[asset]["asset_name"]
    df = df[df["asset"] == asset_name].copy().reset_index(drop=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Select exp10 low-price winners for BTC/ETH using fixed vs dynamic low-price head-to-head.")
    ap.add_argument("--days", type=int, default=TOTAL_DAYS)
    ap.add_argument("--holdout-days", type=int, default=HOLDOUT_DAYS)
    ap.add_argument("--low", type=float, default=LOW_PRICE)
    ap.add_argument("--high", type=float, default=HIGH_PRICE)
    ap.add_argument("--step", type=float, default=FIXED_STEP)
    ap.add_argument("--refresh-predictions", action="store_true")
    args = ap.parse_args()

    base_cfg = _load_base_trader()
    base_params = _runtime_params_from_cfg(base_cfg)
    generated_at = _now_iso()
    report: Dict[str, Any] = {
        "generatedAt": generated_at,
        "selection_window": {"days": args.days, "holdout_days": args.holdout_days},
        "baseTraderName": BASE_TRADER_NAME,
        "selectionMode": "holdout_head_to_head",
        "offlineSelectionScope": "price-layer only, runtime guard parity enforced online via cloned trader config",
        "priceBounds": {"low": args.low, "high": args.high, "fixedStep": args.step},
        "windowDays": args.days,
        "holdoutDays": args.holdout_days,
        "assets": {},
    }
    exec_report: Dict[str, Any] = {
        "generatedAt": generated_at,
        "scope": "exp10_lowprice_execution_calibration",
        "priceBounds": {"low": args.low, "high": args.high, "fixedStep": args.step},
        "assets": {},
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    for asset, meta in ASSET_ROWS.items():
        predictions = _load_or_generate_predictions(asset, meta["model_dir"], args.days, args.refresh_predictions)
        predictions = _attach_path_touch(predictions, asset)
        exec_calibration = _build_execution_calibration(asset, base_params, args.low, args.high, args.step)
        exec_arrays = _build_execution_arrays(exec_calibration["rows"])
        train_df, holdout_df = _split_train_holdout(predictions, args.holdout_days)
        fixed_train = _train_best_fixed(train_df, base_params, args.low, args.high, args.step, exec_arrays)
        fixed_price = float(fixed_train["best"]["buy_price"])

        fixed_holdout = _simulate_fixed(holdout_df, base_params, fixed_price, exec_arrays)
        dynamic_holdout = _simulate_dynamic(holdout_df, base_params, args.low, args.high, exec_arrays)
        fixed_full = _simulate_fixed(predictions, base_params, fixed_price, exec_arrays)
        dynamic_full = _simulate_dynamic(predictions, base_params, args.low, args.high, exec_arrays)
        winner_decision = _choose_winner(fixed_holdout, dynamic_holdout, fixed_full, dynamic_full)
        winner = winner_decision["winner"]

        fixed_rules = _build_rules_payload(
            base_cfg,
            mode="fixed-opt",
            buy_price=fixed_price,
            buy_price_range=None,
            extra={
                "asset": asset,
                "holdout_total_pnl": fixed_holdout["total_pnl"],
                "full_total_pnl": fixed_full["total_pnl"],
                "source_prediction_suffix": meta["source_prediction_suffix"],
            },
        )
        dynamic_rules = _build_rules_payload(
            base_cfg,
            mode="dynamic-band",
            buy_price=None,
            buy_price_range=[round(args.low, 4), round(args.high, 4)],
            extra={
                "asset": asset,
                "holdout_total_pnl": dynamic_holdout["total_pnl"],
                "full_total_pnl": dynamic_full["total_pnl"],
                "source_prediction_suffix": meta["source_prediction_suffix"],
                "dynamic_price_formula": "low + (high-low) * clamp((confidence-probThreshold)/(1-probThreshold), 0, 1)",
            },
        )

        fixed_rules_path = RESULTS_DIR / f"optimal_trading_rules_exp10_lowprice_{asset.lower()}_fixedopt.json"
        dynamic_rules_path = RESULTS_DIR / f"optimal_trading_rules_exp10_lowprice_{asset.lower()}_dynamicband.json"
        winner_rules_path = RESULTS_DIR / f"optimal_trading_rules_exp10_lowprice_{asset.lower()}_winner.json"
        _write_json(fixed_rules_path, fixed_rules)
        _write_json(dynamic_rules_path, dynamic_rules)
        if winner == "fixed-opt":
            _write_json(winner_rules_path, fixed_rules)
        elif winner == "dynamic-band":
            _write_json(winner_rules_path, dynamic_rules)

        report["assets"][asset] = {
            "asset": asset,
            "sourceModelDir": str(meta["model_dir"]),
            "sourcePredictionSuffix": meta["source_prediction_suffix"],
            "derivedPredictionSuffix": meta["out_prediction_suffix"],
            "rows": int(len(predictions)),
            "pathCoverageRows": int(predictions["path_has_coverage"].sum()),
            "pathCoverageRate": _round(float(predictions["path_has_coverage"].mean()) if len(predictions) else 0.0, 6),
            "trainRows": int(len(train_df)),
            "holdoutRows": int(len(holdout_df)),
            "fixed": {
                "trainBest": fixed_train["best"],
                "holdout": fixed_holdout,
                "full": fixed_full,
                "rulesPath": str(fixed_rules_path),
            },
            "dynamic": {
                "holdout": dynamic_holdout,
                "full": dynamic_full,
                "rulesPath": str(dynamic_rules_path),
                "buy_price_range": [round(args.low, 4), round(args.high, 4)],
            },
            "winner": {
                "selection": winner,
                "reason": winner_decision["reason"],
                "rulesPath": str(winner_rules_path) if winner else None,
                "derivedPredictionFile": str(meta["out_prediction_file"]),
                "derivedPredictionSuffix": meta["out_prediction_suffix"],
                "fixedPassesMinimumGates": _candidate_passes(fixed_full, fixed_holdout),
                "dynamicPassesMinimumGates": _candidate_passes(dynamic_full, dynamic_holdout),
            },
            "executionCalibration": {
                "source": exec_calibration["source"],
                "requestUsd": exec_calibration["request_usd"],
                "snapshotCount": exec_calibration["snapshot_count"],
                "defaults": exec_calibration["defaults"],
            },
        }
        exec_report["assets"][asset] = exec_calibration

    latest = REPORTS / "exp10_lowprice_selection_latest.json"
    stamped = REPORTS / f"exp10_lowprice_selection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    _write_json(latest, report)
    _write_json(stamped, report)
    _write_json(EXECUTION_CALIBRATION_REPORT, exec_report)
    print(json.dumps({
        "status": "ok",
        "report": str(latest),
        "assets": {k: v["winner"] for k, v in report["assets"].items()},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
