#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POLY = ROOT / "polymarket"
REPORT = ROOT / "reports" / "simulation_money_audit_latest.json"

CONFIG_SETS = {
    "baseline_default": {
        "profile": "default",
        "kind": "baseline",
        "config": POLY / "trader_configs.json",
        "active": POLY / "active_traders.json",
    },
    "baseline_70": {
        "profile": "70",
        "kind": "baseline",
        "config": POLY / "trader_configs_70.json",
        "active": POLY / "active_traders_70.json",
    },
    "lowprice_default": {
        "profile": "default",
        "kind": "lowprice",
        "config": POLY / "trader_configs_monitor_only_lowprice.json",
    },
    "lowprice_70": {
        "profile": "70",
        "kind": "lowprice",
        "config": POLY / "trader_configs_monitor_only_lowprice_70.json",
    },
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def active_names(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    payload = load_json(path)
    names = payload.get("traderNames") or payload.get("active_traders") or []
    return {str(x).strip() for x in names if str(x).strip()}


def parse_allowed_markets(raw: Any) -> list[str]:
    if isinstance(raw, list):
        vals = [str(x).strip().upper() for x in raw if str(x).strip()]
    else:
        vals = [part.strip().upper() for part in str(raw or "").split(",") if part.strip()]
    return sorted(set(x for x in vals if x))


def resolve_row_symbols(row: dict[str, Any], kind: str) -> list[str]:
    if kind == "lowprice":
        symbol = str(row.get("lowPriceSymbol") or "").upper()
        return [symbol] if symbol else []
    return parse_allowed_markets(row.get("allowedMarkets"))


def effective_symbol_initial_capital(row: dict[str, Any], symbol: str, kind: str) -> float:
    initial = float(row.get("initialCapital") or 0.0)
    if kind == "lowprice":
        return initial
    markets = parse_allowed_markets(row.get("allowedMarkets"))
    if bool(row.get("perCoinCapital")) and symbol and len(markets) > 1:
        return round(initial / len(markets), 4)
    return round(initial, 4)


def runtime_dirs(logs_dir: str, symbol: str) -> list[Path]:
    if not logs_dir:
        return []
    base = POLY / logs_dir
    split = POLY / f"{logs_dir}__simulation_{symbol.lower()}"
    if split.is_dir():
        return [split]
    return [base]


def iter_trade_log_paths(dirs: list[Path]) -> list[Path]:
    out: list[Path] = []
    for runtime_dir in dirs:
        for name in ("prediction_trades.simulation.json", "prediction_trades.json"):
            path = runtime_dir / name
            if path.exists():
                out.append(path)
    seen: set[str] = set()
    uniq: list[Path] = []
    for path in out:
        key = str(path)
        if key not in seen:
            uniq.append(path)
            seen.add(key)
    return uniq


def load_trade_rows(dirs: list[Path], symbol: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for path in iter_trade_log_paths(dirs):
        try:
            payload = load_json(path)
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not isinstance(row, dict):
                continue
            if str(row.get("symbol") or "").upper() != symbol.upper():
                continue
            key = (
                row.get("id"),
                row.get("marketSlug"),
                row.get("timestamp"),
                row.get("direction"),
                row.get("amount"),
                row.get("tokenPrice"),
                row.get("pnl"),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(dict(row))
    return rows


def load_pending_frozen(dirs: list[Path], symbol: str) -> float:
    total = 0.0
    for runtime_dir in dirs:
        path = runtime_dir / "pending_sim_orders.simulation.json"
        if not path.exists():
            continue
        try:
            payload = load_json(path)
        except Exception:
            continue
        if not isinstance(payload, list):
            continue
        for row in payload:
            if not isinstance(row, dict):
                continue
            row_symbol = str(row.get("symbol") or symbol).upper()
            if row_symbol != symbol.upper():
                continue
            total += float(row.get("frozenAmount") or 0.0)
    return round(total, 4)


def load_pending_rows(dirs: list[Path], symbol: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for runtime_dir in dirs:
        for name in ("pending_sim_orders.simulation.json", "pending_sim_orders.json"):
            path = runtime_dir / name
            if not path.exists():
                continue
            try:
                payload = load_json(path)
            except Exception:
                continue
            if not isinstance(payload, list):
                continue
            for row in payload:
                if not isinstance(row, dict):
                    continue
                row_symbol = str(row.get("symbol") or symbol).upper()
                if row_symbol != symbol.upper():
                    continue
                key = (
                    row.get("id"),
                    row.get("marketSlug"),
                    row.get("targetPeriodEndTs"),
                    row.get("limitPrice"),
                    row.get("frozenAmount"),
                )
                if key in seen:
                    continue
                seen.add(key)
                rows.append(dict(row))
    return rows


def parse_ladder_prices(raw: Any) -> list[float]:
    if isinstance(raw, list):
        vals = raw
    else:
        vals = str(raw or "").split(",")
    out: list[float] = []
    for item in vals:
        try:
            out.append(round(float(item), 2))
        except Exception:
            continue
    return out


def dynamic_ladder_audit(row: dict[str, Any], trades: list[dict[str, Any]], pending_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ladder_prices = parse_ladder_prices(row.get("limitPriceLadder"))
    expected_keys = [f"{price:.2f}" for price in ladder_prices]
    expected_set = set(expected_keys)
    fills_by_price: dict[str, dict[str, Any]] = {}
    pending_by_price: dict[str, dict[str, Any]] = {}

    for trade in trades:
        if trade.get("limitPriceConfigured") is None:
            continue
        try:
            key = f"{float(trade.get('limitPriceConfigured')):.2f}"
        except Exception:
            continue
        bucket = fills_by_price.setdefault(key, {"filled_amount": 0.0, "fills": 0})
        bucket["filled_amount"] = round(bucket["filled_amount"] + float(trade.get("amount") or 0.0), 4)
        bucket["fills"] += 1

    for pending in pending_rows:
        if pending.get("limitPrice") is None:
            continue
        try:
            key = f"{float(pending.get('limitPrice')):.2f}"
        except Exception:
            continue
        bucket = pending_by_price.setdefault(key, {"pending_frozen": 0.0, "pending_orders": 0})
        bucket["pending_frozen"] = round(bucket["pending_frozen"] + float(pending.get("frozenAmount") or 0.0), 4)
        bucket["pending_orders"] += 1

    all_keys = sorted(set(fills_by_price) | set(pending_by_price) | expected_set, key=float)
    rung_rows: list[dict[str, Any]] = []
    total_filled = 0.0
    total_pending = 0.0
    unexpected_prices: list[str] = []
    for key in all_keys:
        if expected_set and key not in expected_set:
            unexpected_prices.append(key)
        fill_info = fills_by_price.get(key, {})
        pending_info = pending_by_price.get(key, {})
        filled_amount = round(float(fill_info.get("filled_amount") or 0.0), 4)
        pending_frozen = round(float(pending_info.get("pending_frozen") or 0.0), 4)
        total_filled = round(total_filled + filled_amount, 4)
        total_pending = round(total_pending + pending_frozen, 4)
        rung_rows.append(
            {
                "limit_price": float(key),
                "expected_in_ladder": key in expected_set,
                "fills": int(fill_info.get("fills") or 0),
                "filled_amount": filled_amount,
                "pending_orders": int(pending_info.get("pending_orders") or 0),
                "pending_frozen": pending_frozen,
            }
        )

    return {
        "expected_rung_count": len(expected_keys),
        "expected_prices": expected_keys,
        "active_rung_count": len([r for r in rung_rows if r["fills"] or r["pending_orders"]]),
        "rungs": rung_rows,
        "totals": {
            "filled_amount": total_filled,
            "pending_frozen": total_pending,
        },
        "unexpected_prices": unexpected_prices,
    }


def dedup_latest_market_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_slug: dict[str, dict[str, Any]] = {}
    for row in rows:
        slug = str(row.get("marketSlug") or row.get("market_slug") or "").strip()
        if not slug:
            continue
        ts = str(row.get("settledAt") or row.get("timestamp") or "")
        current = by_slug.get(slug)
        if current is None or ts >= str(current.get("settledAt") or current.get("timestamp") or ""):
            by_slug[slug] = row
    return list(by_slug.values())


def summary_symbol_metrics(dirs: list[Path], symbol: str) -> dict[str, Any]:
    out = {
        "summary_files": [],
        "fills": 0,
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "pending": 0,
        "total_pnl": 0.0,
        "initial_capital": None,
        "current_capital": None,
        "capital_change": None,
    }
    initial_vals: list[float] = []
    current_vals: list[float] = []
    cap_changes: list[float] = []
    for runtime_dir in dirs:
        path = runtime_dir / "reports" / "report_summary.simulation.json"
        if not path.exists():
            continue
        try:
            payload = load_json(path)
        except Exception:
            continue
        out["summary_files"].append(str(path))
        by_symbol = payload.get("bySymbol") if isinstance(payload, dict) else {}
        sym = by_symbol.get(symbol.upper()) if isinstance(by_symbol, dict) else {}
        if not isinstance(sym, dict):
            continue
        out["fills"] += int(sym.get("fills") or 0)
        out["trades"] += int(sym.get("trades") or 0)
        out["wins"] += int(sym.get("wins") or 0)
        out["losses"] += int(sym.get("losses") or 0)
        out["pending"] += int(sym.get("pending") or 0)
        out["total_pnl"] += float(sym.get("pnl") or 0.0)
        if sym.get("initialCapital") is not None:
            initial_vals.append(float(sym.get("initialCapital")))
        if sym.get("currentCapital") is not None:
            current_vals.append(float(sym.get("currentCapital")))
        if sym.get("capitalChange") is not None:
            cap_changes.append(float(sym.get("capitalChange")))
    out["total_pnl"] = round(float(out["total_pnl"]), 4)
    if initial_vals:
        out["initial_capital"] = round(sum(initial_vals), 4)
    if current_vals:
        out["current_capital"] = round(sum(current_vals), 4)
    if cap_changes:
        out["capital_change"] = round(sum(cap_changes), 4)
    return out


def money_truth(rows: list[dict[str, Any]], initial_capital: float, pending_frozen: float) -> dict[str, Any]:
    executed = [row for row in rows if str(row.get("status") or "").lower() == "executed"]
    fills = [row for row in executed if row.get("amount") is not None]
    dedup = dedup_latest_market_rows(executed)
    settled = [row for row in dedup if str(row.get("result") or "").lower() in {"win", "lose"}]
    pending = [row for row in dedup if str(row.get("result") or "").lower() not in {"win", "lose"}]
    pending_executed_amount = round(
        sum(float(row.get("amount") or 0.0) for row in executed if str(row.get("result") or "").lower() not in {"win", "lose"}),
        4,
    )
    pnl_sum = round(sum(float(row.get("pnl") or 0.0) for row in executed), 4)
    capital = float(initial_capital)
    for row in sorted(executed, key=lambda r: str(r.get("timestamp") or "")):
        capital -= float(row.get("amount") or 0.0)
        result = str(row.get("result") or "").lower()
        if result == "win":
            capital += float(row.get("amount") or 0.0) + float(row.get("pnl") or 0.0)
        elif result == "lose" and row.get("stoppedOut") and row.get("pnl") is not None:
            capital += float(row.get("amount") or 0.0) + float(row.get("pnl") or 0.0)
    capital -= float(pending_frozen or 0.0)
    capital = round(capital, 4)
    return {
        "fills": len(fills),
        "dedup_markets": len(dedup),
        "settled_markets": len(settled),
        "pending_markets": len(pending),
        "total_amount": round(sum(float(row.get("amount") or 0.0) for row in executed), 4),
        "total_pnl": pnl_sum,
        "initial_capital": round(float(initial_capital), 4),
        "pending_frozen": round(float(pending_frozen or 0.0), 4),
        "pending_executed_amount": pending_executed_amount,
        "unsettled_exposure": round(pending_executed_amount + float(pending_frozen or 0.0), 4),
        "current_capital_path": capital,
        "capital_change_path": round(capital - float(initial_capital), 4),
    }


def float_eq(a: Any, b: Any, tol: float = 1e-6) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def audit_row(profile: str, kind: str, row: dict[str, Any], symbol: str) -> dict[str, Any]:
    logs_dir = str(row.get("logsDir") or "")
    dirs = runtime_dirs(logs_dir, symbol)
    trades = load_trade_rows(dirs, symbol)
    pending_rows = load_pending_rows(dirs, symbol)
    summary = summary_symbol_metrics(dirs, symbol)
    initial_capital = effective_symbol_initial_capital(row, symbol, kind)
    pending_frozen = load_pending_frozen(dirs, symbol)
    truth = money_truth(trades, initial_capital, pending_frozen)
    hard_issues: list[str] = []
    soft_issues: list[str] = []

    if kind == "lowprice":
        source_initial = row.get("lowPriceSourceInitialCapital")
        source_markets = row.get("lowPriceSourceAllowedMarkets")
        expected_clone_initial = row.get("lowPriceExpectedCloneInitialCapital")
        try:
            if expected_clone_initial is None and source_initial is not None and source_markets:
                if isinstance(source_markets, list):
                    market_count = len([str(x).strip() for x in source_markets if str(x).strip()])
                else:
                    market_count = len([x.strip() for x in str(source_markets).split(",") if x.strip()])
                expected_clone_initial = float(source_initial) / max(1, market_count)
            if expected_clone_initial is not None and not float_eq(initial_capital, float(expected_clone_initial), tol=0.01):
                hard_issues.append("lowprice_initial_capital_semantics_mismatch")
        except Exception:
            hard_issues.append("lowprice_initial_capital_semantics_invalid")

    for trade in trades:
        token_price = trade.get("tokenPrice")
        limit_price = trade.get("limitPriceConfigured")
        if token_price is not None and limit_price is not None:
            try:
                if float(token_price) - float(limit_price) > 1e-9:
                    hard_issues.append("token_price_exceeds_limit_price")
                    break
            except Exception:
                hard_issues.append("invalid_token_or_limit_price")
                break

    pnl_diff = None
    market_diff = None
    pending_diff = None
    current_capital_diff = None
    capital_change_diff = None
    summary_open_fill_lag = False
    if summary["summary_files"]:
        pnl_diff = round(float(truth["total_pnl"]) - float(summary["total_pnl"] or 0.0), 4)
        market_diff = int(truth["dedup_markets"]) - int(summary["trades"] or 0)
        pending_diff = int(truth["pending_markets"]) - int(summary["pending"] or 0)
        expected_unsettled_diff = round(-float(truth["unsettled_exposure"] or 0.0), 4)
        unresolved_current_diff = None
        unresolved_cap_change_diff = None
        if summary["current_capital"] is not None:
            current_capital_diff = round(float(truth["current_capital_path"]) - float(summary["current_capital"] or 0.0), 4)
            unresolved_current_diff = round(current_capital_diff - expected_unsettled_diff, 4)
            if abs(current_capital_diff) > 1.0 and abs(unresolved_current_diff) > 1.0:
                hard_issues.append("summary_current_capital_mismatch")
        if summary["capital_change"] is not None:
            capital_change_diff = round(float(truth["capital_change_path"]) - float(summary["capital_change"] or 0.0), 4)
            unresolved_cap_change_diff = round(capital_change_diff - expected_unsettled_diff, 4)
            if abs(capital_change_diff) > 1.0 and abs(unresolved_cap_change_diff) > 1.0:
                hard_issues.append("summary_capital_change_mismatch")
        summary_open_fill_lag = (
            int(truth["pending_markets"]) > 0
            and int(summary["trades"] or 0) == 0
            and int(summary["pending"] or 0) == 0
            and abs(float(pnl_diff or 0.0)) <= 1.0
        )
        explained_pending_summary_gap = (
            int(truth["pending_markets"]) > 0
            and market_diff == pending_diff == int(truth["pending_markets"])
            and abs(float(unresolved_current_diff or 0.0)) <= 1.0
            and abs(float(unresolved_cap_change_diff or 0.0)) <= 1.0
        )
        if abs(float(pnl_diff or 0.0)) > 1.0:
            hard_issues.append("summary_pnl_mismatch")
        if market_diff not in (0, None) and not summary_open_fill_lag and not explained_pending_summary_gap:
            soft_issues.append("summary_trade_count_mismatch")
        if pending_diff not in (0, None) and not summary_open_fill_lag and not explained_pending_summary_gap:
            soft_issues.append("summary_pending_count_mismatch")

    selected_limit_check = None
    ladder_audit = None
    if trades and kind == "lowprice":
        selection_mode = str(row.get("lowPriceSelectionMode") or "")
        if selection_mode == "fixed_price":
            target = row.get("lowPriceSelectedBuyPrice")
            checks = [float_eq(trade.get("limitPriceConfigured"), target, tol=0.005) for trade in trades if trade.get("limitPriceConfigured") is not None]
            selected_limit_check = bool(checks) and all(checks)
            if selected_limit_check is False:
                hard_issues.append("lowprice_fixed_limit_mismatch")
        elif selection_mode == "dynamic_range":
            rng = row.get("lowPriceSelectedBuyPriceRange") or []
            if isinstance(rng, list) and len(rng) == 2:
                lo = float(rng[0])
                hi = float(rng[1])
                checks = []
                for trade in trades:
                    if trade.get("limitPriceConfigured") is None:
                        continue
                    try:
                        px = float(trade.get("limitPriceConfigured"))
                    except Exception:
                        continue
                    checks.append(lo - 1e-9 <= px <= hi + 1e-9)
                selected_limit_check = bool(checks) and all(checks)
                if selected_limit_check is False:
                    hard_issues.append("lowprice_dynamic_limit_out_of_range")
    if kind == "lowprice" and str(row.get("lowPriceSelectionMode") or "") == "dynamic_range":
        ladder_audit = dynamic_ladder_audit(row, trades, pending_rows)
        if ladder_audit["unexpected_prices"]:
            hard_issues.append("lowprice_dynamic_unexpected_ladder_price")
        if not float_eq(ladder_audit["totals"]["filled_amount"], truth["total_amount"], tol=0.01):
            hard_issues.append("lowprice_dynamic_filled_amount_mismatch")
        if not float_eq(ladder_audit["totals"]["pending_frozen"], pending_frozen, tol=0.01):
            hard_issues.append("lowprice_dynamic_pending_frozen_mismatch")

    status = "ok"
    if not trades and kind == "lowprice":
        status = "no_runtime_trades_yet"
    elif hard_issues:
        status = "warning"
    elif soft_issues:
        status = "soft_warning"

    return {
        "profile": profile,
        "kind": kind,
        "name": row.get("name"),
        "symbol": symbol,
        "logsDir": logs_dir,
        "runtime_dirs": [str(x) for x in dirs],
        "status": status,
        "issues": sorted(set(hard_issues + soft_issues)),
        "hard_issues": sorted(set(hard_issues)),
        "soft_issues": sorted(set(soft_issues)),
        "trade_log_truth": truth,
        "summary_aggregate": summary,
        "differences": {
            "pnl_minus_summary_pnl": pnl_diff,
            "fills_minus_summary_fills": None,
            "markets_minus_summary_trades": market_diff,
            "pending_minus_summary_pending": pending_diff,
            "current_capital_minus_summary": current_capital_diff,
            "capital_change_minus_summary": capital_change_diff,
            "expected_unsettled_diff": expected_unsettled_diff if summary["summary_files"] else None,
            "unresolved_current_capital_diff": unresolved_current_diff if summary["summary_files"] else None,
            "unresolved_capital_change_diff": unresolved_cap_change_diff if summary["summary_files"] else None,
            "summary_open_fill_lag": summary_open_fill_lag,
        },
        "capital_semantics": {
            "initial_capital": initial_capital,
            "source_initial_capital": row.get("lowPriceSourceInitialCapital"),
            "source_allowed_markets": row.get("lowPriceSourceAllowedMarkets"),
            "expected_clone_initial_capital": row.get("lowPriceExpectedCloneInitialCapital"),
        },
        "limit_price_check": {
            "selection_mode": row.get("lowPriceSelectionMode"),
            "selected_buy_price": row.get("lowPriceSelectedBuyPrice"),
            "selected_buy_price_range": row.get("lowPriceSelectedBuyPriceRange"),
            "selected_limit_check": selected_limit_check,
        },
        "dynamic_ladder_audit": ladder_audit,
    }


def iter_target_rows() -> list[tuple[str, str, dict[str, Any]]]:
    out: list[tuple[str, str, dict[str, Any]]] = []
    for meta in CONFIG_SETS.values():
        rows = load_json(meta["config"])
        active = active_names(meta.get("active"))
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "")
            if meta["kind"] == "baseline" and name not in active:
                continue
            out.append((meta["profile"], meta["kind"], row))
    return out


def main() -> None:
    rows_out: list[dict[str, Any]] = []
    hard_issue_counts: dict[str, int] = {}
    soft_issue_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    for profile, kind, row in iter_target_rows():
        for symbol in resolve_row_symbols(row, kind):
            result = audit_row(profile, kind, row, symbol)
            rows_out.append(result)
            status = result["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
            for issue in result["hard_issues"]:
                hard_issue_counts[issue] = hard_issue_counts.get(issue, 0) + 1
            for issue in result["soft_issues"]:
                soft_issue_counts[issue] = soft_issue_counts.get(issue, 0) + 1

    payload = {
        "generatedAt": now_iso(),
        "status_counts": dict(sorted(status_counts.items())),
        "hard_issue_counts": dict(sorted(hard_issue_counts.items())),
        "soft_issue_counts": dict(sorted(soft_issue_counts.items())),
        "rows": rows_out,
        "ok": not bool(hard_issue_counts),
    }
    dump_json(REPORT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
