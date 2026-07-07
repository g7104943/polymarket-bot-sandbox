#!/usr/bin/env python3
"""
Live/Sim 共存版汇总脚本

核心特性:
1. 动态发现组合目录（配置 + 磁盘），不依赖固定白名单。
2. 支持拆分运行目录（如 logs_xxx__live_eth / logs_xxx__simulation_btc）。
3. 默认按 mode 隔离展示，不混算。
4. 兼容旧文件:
   - report_summary.<mode>.json -> report_summary.json
   - prediction_trades.<mode>.json -> prediction_trades.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
CORE_UNIVERSE_LATEST = SCRIPT_DIR.parent / "reports" / "core_universe_latest.json"
ALWAYS_INCLUDE_GROUPS = {"ensemble"}
MODE_SET = {"simulation", "live", "backtest"}
MODE_SUFFIX_RE = re.compile(r"__(simulation|live|backtest)(?:_[a-z0-9_]+)?$", re.IGNORECASE)


@dataclass
class ComboStats:
    log_dir: str
    base_log_dir: str
    mode: str
    initial_capital: float
    total_trades: int
    completed: int
    wins: int
    losses: int
    pending: int
    total_pnl: float
    final_capital: float
    avg_buy_price: float
    avg_buy_price_win: float
    avg_buy_price_lose: float
    up_count: int
    down_count: int


@dataclass
class ComboSymbolStats:
    log_dir: str
    base_log_dir: str
    mode: str
    symbol: str
    initial_capital: float
    total_trades: int
    wins: int
    losses: int
    pending: int
    total_pnl: float
    final_capital: float
    avg_buy_price: float
    avg_buy_price_win: float
    avg_buy_price_lose: float
    up_count: int
    down_count: int
    capital_is_estimated: bool


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_scope_cells(scope: str) -> Set[Tuple[str, str, str]]:
    if scope == "all" or not CORE_UNIVERSE_LATEST.exists():
        return set()
    payload = _load_json(CORE_UNIVERSE_LATEST)
    if not isinstance(payload, dict):
        return set()
    rows = payload.get("core_cells" if scope == "core" else "shadow_cells")
    out: Set[Tuple[str, str, str]] = set()
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            out.add(
                (
                    str(row.get("profile") or ""),
                    str(row.get("trader") or ""),
                    str(row.get("symbol") or "").upper(),
                )
            )
    return out


def _extract_active_names(raw: Any) -> Set[str]:
    if isinstance(raw, list):
        return {str(x).strip() for x in raw if isinstance(x, str) and str(x).strip()}
    if isinstance(raw, dict):
        out: Set[str] = set()
        for key in ("traderNames", "active_traders"):
            vals = raw.get(key)
            if isinstance(vals, list):
                out.update(str(x).strip() for x in vals if isinstance(x, str) and str(x).strip())
        return out
    return set()


def _extract_active_groups(raw: Any) -> Set[str]:
    if isinstance(raw, dict):
        vals = raw.get("groups")
        if isinstance(vals, list):
            return {str(x).strip() for x in vals if isinstance(x, str) and str(x).strip()}
    return set()


def _cfg_paths(profile: str) -> Tuple[Path, Path]:
    if profile == "70":
        return SCRIPT_DIR / "trader_configs_70.json", SCRIPT_DIR / "active_traders_70.json"
    return SCRIPT_DIR / "trader_configs.json", SCRIPT_DIR / "active_traders.json"


def _discover_base_dirs(profile: str, active_only: bool) -> List[str]:
    cfg_path, active_path = _cfg_paths(profile)
    cfg_raw = _load_json(cfg_path)
    if not isinstance(cfg_raw, list):
        return []

    all_dirs = []
    for row in cfg_raw:
        if not isinstance(row, dict):
            continue
        d = str(row.get("logsDir", "")).strip()
        if d:
            all_dirs.append(d)

    if not active_only:
        return sorted(set(all_dirs))

    active_raw = _load_json(active_path)
    active_names = _extract_active_names(active_raw)
    active_groups = _extract_active_groups(active_raw)
    use_group_fallback = (len(active_names) == 0 and len(active_groups) > 0)

    selected: Set[str] = set()
    for row in cfg_raw:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        group = str(row.get("group", "")).strip()
        logs_dir = str(row.get("logsDir", "")).strip()
        if not logs_dir:
            continue
        if name in active_names:
            selected.add(logs_dir)
            continue
        if group in ALWAYS_INCLUDE_GROUPS:
            selected.add(logs_dir)
            continue
        if use_group_fallback and group and group in active_groups:
            selected.add(logs_dir)

    return sorted(selected if selected else set(all_dirs))


def _mode_from_log_dir(log_dir: str) -> str:
    m = MODE_SUFFIX_RE.search(log_dir)
    if m:
        return m.group(1).lower()
    return "simulation"


def _base_log_dir(log_dir: str) -> str:
    return MODE_SUFFIX_RE.sub("", log_dir)


def _display_combo_name(log_dir: str) -> str:
    name = _base_log_dir(log_dir)
    for prefix in ("logs_70_", "logs_"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    if name.startswith("logs_"):
        name = name[len("logs_"):]
    if name.startswith("v5_"):
        name = name[len("v5_"):]
    if name.endswith("_wallet"):
        base = name[:-len("_wallet")]
        return f"wallet-{base}"
    return name


def _should_include_mode(mode: str, mode_filter: str) -> bool:
    if mode_filter == "both":
        return mode in MODE_SET
    return mode == mode_filter


def _expand_runtime_dirs(base_dirs: Iterable[str], profile: str, mode_filter: str) -> List[str]:
    prefix = "logs_70_" if profile == "70" else "logs_"
    all_fs_dirs = [d for d in SCRIPT_DIR.iterdir() if d.is_dir() and d.name.startswith(prefix)]
    by_base: Dict[str, Set[str]] = defaultdict(set)
    for d in all_fs_dirs:
        by_base[_base_log_dir(d.name)].add(d.name)

    out: Set[str] = set()
    for base in base_dirs:
        candidates = by_base.get(base, set())
        if not candidates and (SCRIPT_DIR / base).is_dir():
            candidates = {base}
        for c in candidates:
            mode = _mode_from_log_dir(c)
            if _should_include_mode(mode, mode_filter):
                out.add(c)
    return sorted(out)


def _all_runtime_dirs(profile: str, mode_filter: str) -> List[str]:
    prefix = "logs_70_" if profile == "70" else "logs_"
    out: List[str] = []
    for d in sorted(SCRIPT_DIR.iterdir(), key=lambda p: p.name):
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        mode = _mode_from_log_dir(d.name)
        if _should_include_mode(mode, mode_filter):
            out.append(d.name)
    return out


def _read_mode_file(base: Path, stem: str, mode: str) -> Any:
    # 优先 mode 文件
    mode_path = base / f"{stem}.{mode}.json"
    if mode_path.exists():
        data = _load_json(mode_path)
        if data is not None:
            return data
    # 回退兼容文件
    merged_path = base / f"{stem}.json"
    if merged_path.exists():
        return _load_json(merged_path)
    return None


def _infer_initial_capital(log_dir: str) -> float:
    base = _base_log_dir(log_dir)
    if "exp" in base or "ensemble" in base:
        return 800.0
    return 400.0


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _trade_list(log_dir: str, mode: str) -> List[Dict[str, Any]]:
    data = _read_mode_file(SCRIPT_DIR / log_dir, "prediction_trades", mode)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    return []


def _summary_dict(log_dir: str, mode: str) -> Optional[Dict[str, Any]]:
    data = _read_mode_file(SCRIPT_DIR / log_dir / "reports", "report_summary", mode)
    return data if isinstance(data, dict) else None


def _compute_combo_stats(log_dir: str, mode: str) -> ComboStats:
    report = _summary_dict(log_dir, mode)
    trades = _trade_list(log_dir, mode)

    if report and isinstance(report.get("summary"), dict):
        summary = report["summary"]
        total_trades = int(summary.get("totalTrades") or summary.get("completedTrades") or 0)
        wins = int(summary.get("wins") or 0)
        losses = int(summary.get("losses") or 0)
        pending = int(summary.get("pending") or 0)
        completed = wins + losses
        total_pnl = _safe_float(summary.get("totalPnL"), 0.0)
        cfg = report.get("config") if isinstance(report.get("config"), dict) else {}
        initial = _safe_float(cfg.get("initialCapital"), _infer_initial_capital(log_dir))
        final_capital = _safe_float(summary.get("currentCapital"), initial + total_pnl)
    else:
        executed = [t for t in trades if str(t.get("status", "")).lower() == "executed"]
        by_slug: Dict[str, Dict[str, Any]] = {}
        for t in executed:
            slug = str(t.get("marketSlug") or f"_no_slug_{id(t)}")
            row = by_slug.setdefault(slug, {"result": None})
            r = str(t.get("result") or "").lower()
            if r in ("win", "lose"):
                row["result"] = r
        wins = sum(1 for v in by_slug.values() if v.get("result") == "win")
        losses = sum(1 for v in by_slug.values() if v.get("result") == "lose")
        pending = sum(1 for v in by_slug.values() if v.get("result") is None)
        completed = wins + losses
        total_trades = len(by_slug)
        total_pnl = sum(_safe_float(t.get("pnl"), 0.0) for t in executed if t.get("pnl") is not None)
        initial = _infer_initial_capital(log_dir)
        final_capital = initial + total_pnl

    executed_trades = [t for t in trades if str(t.get("status", "")).lower() == "executed"]
    buy_prices = []
    buy_prices_win = []
    buy_prices_lose = []
    up_count = 0
    down_count = 0
    for t in executed_trades:
        direction = str(t.get("direction") or "").upper()
        if direction == "UP":
            up_count += 1
        elif direction == "DOWN":
            down_count += 1
        price = t.get("tokenPrice")
        if price is None:
            continue
        p = _safe_float(price, -1)
        if p <= 0:
            continue
        buy_prices.append(p)
        result = str(t.get("result") or "").lower()
        if result == "win":
            buy_prices_win.append(p)
        elif result == "lose":
            buy_prices_lose.append(p)

    avg_buy = sum(buy_prices) / len(buy_prices) if buy_prices else 0.0
    avg_buy_win = sum(buy_prices_win) / len(buy_prices_win) if buy_prices_win else 0.0
    avg_buy_lose = sum(buy_prices_lose) / len(buy_prices_lose) if buy_prices_lose else 0.0

    return ComboStats(
        log_dir=log_dir,
        base_log_dir=_base_log_dir(log_dir),
        mode=mode,
        initial_capital=round(initial, 2),
        total_trades=total_trades,
        completed=completed,
        wins=wins,
        losses=losses,
        pending=pending,
        total_pnl=round(total_pnl, 2),
        final_capital=round(final_capital, 2),
        avg_buy_price=round(avg_buy, 5),
        avg_buy_price_win=round(avg_buy_win, 5),
        avg_buy_price_lose=round(avg_buy_lose, 5),
        up_count=up_count,
        down_count=down_count,
    )


def _merge_stats_by_base(rows: List[ComboStats]) -> List[ComboStats]:
    grouped: Dict[str, List[ComboStats]] = defaultdict(list)
    for r in rows:
        grouped[r.base_log_dir].append(r)

    merged: List[ComboStats] = []
    for base, parts in grouped.items():
        # split 运行目录（如 __live_eth + __simulation_btc）合并时，
        # 初始资金应取各分片初始资金之和，而不是任意一片。
        initial = round(sum(x.initial_capital for x in parts), 2)
        total_trades = sum(x.total_trades for x in parts)
        completed = sum(x.completed for x in parts)
        wins = sum(x.wins for x in parts)
        losses = sum(x.losses for x in parts)
        pending = sum(x.pending for x in parts)
        total_pnl = round(sum(x.total_pnl for x in parts), 2)
        final_capital = round(initial + total_pnl, 2)
        up_count = sum(x.up_count for x in parts)
        down_count = sum(x.down_count for x in parts)

        def _weighted_avg(vals: List[Tuple[float, int]]) -> float:
            numer = sum(v * n for v, n in vals if v > 0 and n > 0)
            denom = sum(n for v, n in vals if v > 0 and n > 0)
            return round(numer / denom, 5) if denom > 0 else 0.0

        avg_buy = _weighted_avg([(x.avg_buy_price, x.up_count + x.down_count) for x in parts])
        avg_buy_win = _weighted_avg([(x.avg_buy_price_win, x.wins) for x in parts])
        avg_buy_lose = _weighted_avg([(x.avg_buy_price_lose, x.losses) for x in parts])

        merged.append(
            ComboStats(
                log_dir=base,
                base_log_dir=base,
                mode="merged",
                initial_capital=initial,
                total_trades=total_trades,
                completed=completed,
                wins=wins,
                losses=losses,
                pending=pending,
                total_pnl=total_pnl,
                final_capital=final_capital,
                avg_buy_price=avg_buy,
                avg_buy_price_win=avg_buy_win,
                avg_buy_price_lose=avg_buy_lose,
                up_count=up_count,
                down_count=down_count,
            )
        )
    return merged


def _compute_combo_symbol_stats(log_dir: str, mode: str) -> List[ComboSymbolStats]:
    report = _summary_dict(log_dir, mode)
    trades = _trade_list(log_dir, mode)

    by_symbol_report: Dict[str, Any] = {}
    initial_capital_total = _infer_initial_capital(log_dir)
    per_coin_capital: Dict[str, float] = {}
    configured_symbols: List[str] = []
    if report and isinstance(report.get("summary"), dict):
        summary = report["summary"]
        cfg = report.get("config") if isinstance(report.get("config"), dict) else {}
        initial_capital_total = _safe_float(cfg.get("initialCapital"), _safe_float(summary.get("initialCapital"), initial_capital_total))
        if isinstance(cfg.get("perCoinCapital"), dict):
            per_coin_capital = {
                str(sym).upper().replace("/USDT", "").strip(): _safe_float(v, 0.0)
                for sym, v in cfg.get("perCoinCapital", {}).items()
                if str(sym).strip()
            }
        if isinstance(cfg.get("allowedMarkets"), list):
            configured_symbols = [
                str(sym).upper().replace("/USDT", "").strip()
                for sym in cfg.get("allowedMarkets", [])
                if str(sym).strip()
            ]
        if isinstance(report.get("bySymbol"), dict):
            by_symbol_report = {
                str(sym).upper().replace("/USDT", "").strip(): row
                for sym, row in report["bySymbol"].items()
                if isinstance(row, dict) and str(sym).strip()
            }

    trade_by_symbol: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "pending": 0,
            "pnl": 0.0,
            "up_count": 0,
            "down_count": 0,
            "_prices": [],
            "_prices_win": [],
            "_prices_lose": [],
        }
    )

    for t in trades:
        if str(t.get("status", "")).lower() != "executed":
            continue
        symbol = str(t.get("symbol") or "").upper().replace("/USDT", "").strip()
        if not symbol:
            continue
        row = trade_by_symbol[symbol]
        row["trades"] += 1
        result = str(t.get("result") or "").lower()
        if result == "win":
            row["wins"] += 1
        elif result == "lose":
            row["losses"] += 1
        else:
            row["pending"] += 1
        direction = str(t.get("direction") or "").upper()
        if direction == "UP":
            row["up_count"] += 1
        elif direction == "DOWN":
            row["down_count"] += 1
        if t.get("pnl") is not None:
            row["pnl"] += _safe_float(t.get("pnl"), 0.0)
        price = t.get("tokenPrice")
        p = _safe_float(price, -1) if price is not None else -1
        if p > 0:
            row["_prices"].append(p)
            if result == "win":
                row["_prices_win"].append(p)
            elif result == "lose":
                row["_prices_lose"].append(p)

    symbols = sorted(set(by_symbol_report.keys()) | set(trade_by_symbol.keys()))
    if not symbols:
        return []

    symbol_count = len(configured_symbols) if configured_symbols else len(symbols)
    initial_per_symbol = round(initial_capital_total / symbol_count, 2) if symbol_count > 0 else 0.0

    out: List[ComboSymbolStats] = []
    for symbol in symbols:
        report_row = by_symbol_report.get(symbol, {})
        trade_row = trade_by_symbol.get(symbol, {})
        prices = list(trade_row.get("_prices", []))
        prices_win = list(trade_row.get("_prices_win", []))
        prices_lose = list(trade_row.get("_prices_lose", []))

        total_trades = int(report_row.get("trades") or trade_row.get("trades") or 0)
        wins = int(report_row.get("wins") or trade_row.get("wins") or 0)
        losses = int(report_row.get("losses") or trade_row.get("losses") or 0)
        pending = int(report_row.get("pending") or trade_row.get("pending") or 0)
        total_pnl = round(_safe_float(report_row.get("pnl"), _safe_float(trade_row.get("pnl"), 0.0)), 2)
        real_coin_capital = per_coin_capital.get(symbol)
        capital_is_estimated = not (real_coin_capital is not None and real_coin_capital > 0)
        final_capital = round(real_coin_capital, 2) if not capital_is_estimated else round(initial_per_symbol + total_pnl, 2)
        avg_buy = round(sum(prices) / len(prices), 5) if prices else 0.0
        avg_buy_win = round(sum(prices_win) / len(prices_win), 5) if prices_win else 0.0
        avg_buy_lose = round(sum(prices_lose) / len(prices_lose), 5) if prices_lose else 0.0
        up_count = int(report_row.get("upCount") or trade_row.get("up_count") or 0)
        down_count = int(report_row.get("downCount") or trade_row.get("down_count") or 0)

        out.append(
            ComboSymbolStats(
                log_dir=log_dir,
                base_log_dir=_base_log_dir(log_dir),
                mode=mode,
                symbol=symbol,
                initial_capital=initial_per_symbol,
                total_trades=total_trades,
                wins=wins,
                losses=losses,
                pending=pending,
                total_pnl=total_pnl,
                final_capital=final_capital,
                avg_buy_price=avg_buy,
                avg_buy_price_win=avg_buy_win,
                avg_buy_price_lose=avg_buy_lose,
                up_count=up_count,
                down_count=down_count,
                capital_is_estimated=capital_is_estimated,
            )
        )
    return out


def _merge_symbol_stats_by_base(rows: List[ComboSymbolStats]) -> List[ComboSymbolStats]:
    grouped: Dict[Tuple[str, str], List[ComboSymbolStats]] = defaultdict(list)
    for r in rows:
        grouped[(r.base_log_dir, r.symbol)].append(r)

    merged: List[ComboSymbolStats] = []
    for (base, symbol), parts in grouped.items():
        initial = round(sum(x.initial_capital for x in parts), 2)
        total_trades = sum(x.total_trades for x in parts)
        wins = sum(x.wins for x in parts)
        losses = sum(x.losses for x in parts)
        pending = sum(x.pending for x in parts)
        total_pnl = round(sum(x.total_pnl for x in parts), 2)
        final_capital = round(initial + total_pnl, 2)
        up_count = sum(x.up_count for x in parts)
        down_count = sum(x.down_count for x in parts)

        def _weighted_avg(vals: List[Tuple[float, int]]) -> float:
            numer = sum(v * n for v, n in vals if v > 0 and n > 0)
            denom = sum(n for v, n in vals if v > 0 and n > 0)
            return round(numer / denom, 5) if denom > 0 else 0.0

        avg_buy = _weighted_avg([(x.avg_buy_price, x.up_count + x.down_count) for x in parts])
        avg_buy_win = _weighted_avg([(x.avg_buy_price_win, x.wins) for x in parts])
        avg_buy_lose = _weighted_avg([(x.avg_buy_price_lose, x.losses) for x in parts])

        merged.append(
            ComboSymbolStats(
                log_dir=base,
                base_log_dir=base,
                mode="merged",
                symbol=symbol,
                initial_capital=initial,
                total_trades=total_trades,
                wins=wins,
                losses=losses,
                pending=pending,
                total_pnl=total_pnl,
                final_capital=final_capital,
                avg_buy_price=avg_buy,
                avg_buy_price_win=avg_buy_win,
                avg_buy_price_lose=avg_buy_lose,
                up_count=up_count,
                down_count=down_count,
                capital_is_estimated=all(x.capital_is_estimated for x in parts),
            )
        )
    return merged


def _collect_symbol_stats(rows: List[ComboStats], mode_filter: str, profile: str, active_only: bool) -> Dict[str, Dict[str, float]]:
    if active_only:
        base_dirs = _discover_base_dirs(profile=profile, active_only=True)
        runtime_dirs = _expand_runtime_dirs(base_dirs, profile=profile, mode_filter=mode_filter)
    else:
        runtime_dirs = _all_runtime_dirs(profile=profile, mode_filter=mode_filter)

    out = defaultdict(lambda: {
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "pending": 0,
        "pnl": 0.0,
        "avg_buy": 0.0,
        "avg_buy_win": 0.0,
        "avg_buy_lose": 0.0,
        "_prices": [],
        "_prices_win": [],
        "_prices_lose": [],
    })

    for log_dir in runtime_dirs:
        mode = _mode_from_log_dir(log_dir)
        trades = _trade_list(log_dir, mode)
        for t in trades:
            if str(t.get("status", "")).lower() != "executed":
                continue
            symbol = str(t.get("symbol") or "").upper().replace("/USDT", "").strip()
            if not symbol:
                continue
            row = out[symbol]
            row["trades"] += 1
            result = str(t.get("result") or "").lower()
            if result == "win":
                row["wins"] += 1
            elif result == "lose":
                row["losses"] += 1
            else:
                row["pending"] += 1
            if t.get("pnl") is not None:
                row["pnl"] += _safe_float(t.get("pnl"), 0.0)
            price = t.get("tokenPrice")
            p = _safe_float(price, -1) if price is not None else -1
            if p > 0:
                row["_prices"].append(p)
                if result == "win":
                    row["_prices_win"].append(p)
                elif result == "lose":
                    row["_prices_lose"].append(p)

    for symbol, row in out.items():
        prices = row.pop("_prices")
        prices_win = row.pop("_prices_win")
        prices_lose = row.pop("_prices_lose")
        row["avg_buy"] = round(sum(prices) / len(prices), 5) if prices else 0.0
        row["avg_buy_win"] = round(sum(prices_win) / len(prices_win), 5) if prices_win else 0.0
        row["avg_buy_lose"] = round(sum(prices_lose) / len(prices_lose), 5) if prices_lose else 0.0
        row["pnl"] = round(row["pnl"], 2)
    return out


def print_rankings(rows: List[ComboStats], mode_filter: str, merged: bool, profile: str, active_only: bool, scope: str) -> None:
    print("=" * 110)
    print("  模型组合资金排名（Live/Sim 隔离）")
    print("=" * 110)
    print(
        f"  范围: {'active + 常驻组' if active_only else '全部目录'}"
        f" | profile={profile} | mode={mode_filter} | merge_modes={'on' if merged else 'off'} | scope={scope}"
    )
    print("-" * 110)
    header = (
        f"{'排名':>4}  {'模型':<42} {'mode':<11} {'交易':>6} {'胜/负':>9} "
        f"{'胜率%':>6} {'总盈亏':>12} {'最终资金':>12} {'均价(胜/负)':>18}"
    )
    print(header)
    print("-" * 110)
    for idx, s in enumerate(rows, start=1):
        avg_pair = f"{s.avg_buy_price_win:.3f}/{s.avg_buy_price_lose:.3f}" if s.avg_buy_price_win or s.avg_buy_price_lose else "-"
        completed = s.wins + s.losses
        win_rate = (s.wins / completed * 100.0) if completed else 0.0
        display_name = _display_combo_name(s.log_dir)
        print(
            f"{idx:>4}  {display_name:<42} {s.mode:<11} {s.total_trades:>6} "
            f"{f'{s.wins}/{s.losses}':>9} {win_rate:>6.1f} "
            f"${s.total_pnl:>+11.2f} ${s.final_capital:>11.2f} {avg_pair:>18}"
        )
    print("-" * 110)
    total_pnl = sum(r.total_pnl for r in rows)
    print(f"  合计组合: {len(rows)} | 总盈亏: ${total_pnl:+.2f}")
    print("=" * 110)


def print_combo_symbol_rankings(rows: List[ComboSymbolStats], mode_filter: str, merged: bool, profile: str, active_only: bool, scope: str) -> None:
    print("=" * 124)
    print("  模型组合+币种资金排名（Live/Sim 隔离）")
    print("=" * 124)
    print(
        f"  范围: {'active + 常驻组' if active_only else '全部目录'}"
        f" | profile={profile} | mode={mode_filter} | merge_modes={'on' if merged else 'off'} | scope={scope}"
    )
    print("  说明: 总盈亏按币种实际成交/报告统计；最终资金优先读取真实 perCoinCapital，缺失时才回退到均分初始资金估算")
    print("-" * 124)
    header = (
        f"{'排名':>4}  {'模型':<36} {'币种':<6} {'mode':<11} {'交易':>6} {'胜/负':>9} "
        f"{'胜率%':>6} {'总盈亏':>12} {'最终资金':>12} {'均价(胜/负)':>18}"
    )
    print(header)
    print("-" * 124)
    for idx, s in enumerate(rows, start=1):
        avg_pair = f"{s.avg_buy_price_win:.3f}/{s.avg_buy_price_lose:.3f}" if s.avg_buy_price_win or s.avg_buy_price_lose else "-"
        completed = s.wins + s.losses
        win_rate = (s.wins / completed * 100.0) if completed else 0.0
        display_name = _display_combo_name(s.log_dir)
        print(
            f"{idx:>4}  {display_name:<36} {s.symbol:<6} {s.mode:<11} {s.total_trades:>6} "
            f"{f'{s.wins}/{s.losses}':>9} {win_rate:>6.1f} "
            f"${s.total_pnl:>+11.2f} ${s.final_capital:>11.2f} {avg_pair:>18}"
        )
    print("-" * 124)
    total_pnl = sum(r.total_pnl for r in rows)
    print(f"  合计模型组合+币种: {len(rows)} | 总盈亏: ${total_pnl:+.2f}")
    print("=" * 124)


def print_symbol_summary(symbol_stats: Dict[str, Dict[str, float]]) -> None:
    if not symbol_stats:
        print("\n未发现可统计的币种交易。")
        return
    print("\n" + "=" * 110)
    print("  按币种汇总（已执行交易）")
    print("=" * 110)
    print(f"{'币种':<8} {'交易':>8} {'胜/负':>11} {'总盈亏':>12} {'均买价':>10} {'胜单均价':>10} {'负单均价':>10}")
    print("-" * 110)
    for symbol in sorted(symbol_stats.keys()):
        s = symbol_stats[symbol]
        wl = f"{int(s['wins'])}/{int(s['losses'])}"
        print(
            f"{symbol:<8} {int(s['trades']):>8} {wl:>11} "
            f"${s['pnl']:>+11.2f} {s['avg_buy']:>10.4f} {s['avg_buy_win']:>10.4f} {s['avg_buy_lose']:>10.4f}"
        )
    print("=" * 110)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live/Sim 共存版组合汇总")
    parser.add_argument("--all-combos", action="store_true", help="扫描所有 logs_* 目录（默认仅 active + 常驻组）")
    parser.add_argument("--profile", choices=("default", "70"), default="default")
    parser.add_argument("--mode", choices=("simulation", "live", "both"), default="simulation")
    parser.add_argument("--merge-modes", action="store_true", help="按 base 组合合并 simulation/live/backtest")
    parser.add_argument("--scope", choices=("core", "shadow", "all"), default="core", help="默认主收益只看 core，shadow 单独查看")
    parser.add_argument("--rank-by", choices=("combo-symbol", "combo"), default="combo-symbol", help="默认按 模型组合+币种 排名，可切回旧的组合排名")
    args = parser.parse_args()

    active_only = not args.all_combos
    mode_filter = args.mode
    profile = args.profile
    scope_cells = _load_scope_cells(args.scope)

    if active_only:
        base_dirs = _discover_base_dirs(profile=profile, active_only=True)
        runtime_dirs = _expand_runtime_dirs(base_dirs, profile=profile, mode_filter=mode_filter)
    else:
        runtime_dirs = _all_runtime_dirs(profile=profile, mode_filter=mode_filter)

    if not runtime_dirs:
        print("未发现可统计目录。")
        return

    rows = [_compute_combo_stats(d, _mode_from_log_dir(d)) for d in runtime_dirs]
    if args.rank_by == "combo":
        if args.merge_modes:
            rows = _merge_stats_by_base(rows)
        if scope_cells:
            filtered_rows = []
            for row in rows:
                display_name = _display_combo_name(row.log_dir)
                trader = display_name.replace("ens_", "ensemble_")
                trader = f"v5_{trader}" if trader.startswith("exp") else trader
                symbols = {key[2] for key in scope_cells if key[0] == profile and key[1] == trader}
                if symbols:
                    filtered_rows.append(row)
            rows = filtered_rows
        rows.sort(key=lambda x: x.final_capital, reverse=True)
        print_rankings(rows, mode_filter=mode_filter, merged=args.merge_modes, profile=profile, active_only=active_only, scope=args.scope)
    else:
        symbol_rows: List[ComboSymbolStats] = []
        for d in runtime_dirs:
            symbol_rows.extend(_compute_combo_symbol_stats(d, _mode_from_log_dir(d)))
        if args.merge_modes:
            symbol_rows = _merge_symbol_stats_by_base(symbol_rows)
        if scope_cells:
            filtered_rows = []
            for row in symbol_rows:
                display_name = _display_combo_name(row.log_dir)
                trader = display_name.replace("ens_", "ensemble_")
                trader = f"v5_{trader}" if trader.startswith("exp") else trader
                if (profile, trader, row.symbol) in scope_cells:
                    filtered_rows.append(row)
            symbol_rows = filtered_rows
        symbol_rows.sort(key=lambda x: x.final_capital, reverse=True)
        print_combo_symbol_rankings(symbol_rows, mode_filter=mode_filter, merged=args.merge_modes, profile=profile, active_only=active_only, scope=args.scope)

    symbol_stats = _collect_symbol_stats(rows, mode_filter=mode_filter, profile=profile, active_only=active_only)
    print_symbol_summary(symbol_stats)


if __name__ == "__main__":
    main()
