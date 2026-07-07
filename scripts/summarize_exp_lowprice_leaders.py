#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLY = PROJECT_ROOT / 'polymarket'
REPORTS = PROJECT_ROOT / 'reports'

CONFIGS = {
    'default': POLY / 'trader_configs_monitor_only_lowprice.json',
    '70': POLY / 'trader_configs_monitor_only_lowprice_70.json',
}

OUT_LATEST = REPORTS / 'exp_lowprice_leaderboard_latest.json'
OUT_STAMPED = REPORTS / f"exp_lowprice_leaderboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace('+00:00', 'Z')


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_sim_trades(logs_dir: str) -> list[dict[str, Any]]:
    path = POLY / logs_dir / 'prediction_trades.simulation.json'
    if not path.exists():
        return []
    try:
        rows = load_json(path)
    except Exception:
        return []
    return rows if isinstance(rows, list) else []


def trade_pnl(row: dict[str, Any]) -> float:
    val = row.get('pnl')
    if val is None:
        val = row.get('netPnl')
    try:
        return float(val)
    except Exception:
        return 0.0


def trade_fill_status(row: dict[str, Any]) -> str:
    for key in ('fillStatus', 'fill_status'):
        val = row.get(key)
        if isinstance(val, str) and val:
            return val
    if row.get('partialFillRatio') not in (None, 0, 1):
        return 'partial_fill'
    return 'filled'


def partial_ratio(row: dict[str, Any]) -> float | None:
    for key in ('partialFillRatio', 'partial_fill_ratio'):
        val = row.get(key)
        if val is None:
            continue
        try:
            return float(val)
        except Exception:
            return None
    return None


def summarize_trader(row: dict[str, Any]) -> dict[str, Any]:
    trades = read_sim_trades(row['logsDir'])
    settled = [t for t in trades if str(t.get('status', '')).lower() not in ('pending', 'open')]
    pnls = [trade_pnl(t) for t in settled]
    wins = sum(1 for p in pnls if p > 0)
    losses = sum(1 for p in pnls if p <= 0)
    total = sum(pnls)
    expectancy = total / len(settled) if settled else 0.0

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for pnl in pnls:
        equity += pnl
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)

    timeout_count = 0
    partial_count = 0
    fill_like_count = 0
    last_ts = None
    for t in trades:
        fill_status = trade_fill_status(t)
        if fill_status in {'timeout_unfilled', 'timeout', 'unfilled'}:
            timeout_count += 1
        if fill_status in {'filled', 'partial_fill'}:
            fill_like_count += 1
        pr = partial_ratio(t)
        if fill_status == 'partial_fill' or (pr is not None and 0 < pr < 1):
            partial_count += 1
        ts = t.get('timestamp') or t.get('createdAt') or t.get('predictionTimestamp')
        if isinstance(ts, str) and (last_ts is None or ts > last_ts):
            last_ts = ts

    fill_rate = fill_like_count / len(trades) if trades else 0.0
    timeout_rate = timeout_count / len(trades) if trades else 0.0
    partial_rate = partial_count / len(trades) if trades else 0.0

    return {
        'name': row['name'],
        'source_trader': row.get('lowPriceSourceTrader'),
        'symbol': row.get('lowPriceSymbol'),
        'bucket': row.get('lowPriceBucket'),
        'logs_dir': row['logsDir'],
        'trades': len(trades),
        'settled_trades': len(settled),
        'wins': wins,
        'losses': losses,
        'total_pnl': round(total, 4),
        'avg_expectancy': round(expectancy, 6),
        'max_drawdown': round(max_drawdown, 4),
        'fill_rate': round(fill_rate, 6),
        'timeout_rate': round(timeout_rate, 6),
        'partial_fill_rate': round(partial_rate, 6),
        'latest_ts': last_ts,
    }


def sort_key(summary: dict[str, Any]) -> tuple[Any, ...]:
    return (
        summary['total_pnl'],
        -summary['max_drawdown'],
        summary['avg_expectancy'],
        summary['fill_rate'],
        -summary['timeout_rate'],
        summary['trades'],
        -(summary['bucket'] or 0),
    )


def choose_leader(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = sorted(candidates, key=sort_key, reverse=True)
    leader = dict(ranked[0])
    if len(ranked) > 1:
        runner_up = ranked[1]
        leader['runner_up'] = {
            'name': runner_up['name'],
            'bucket': runner_up['bucket'],
            'total_pnl': runner_up['total_pnl'],
            'max_drawdown': runner_up['max_drawdown'],
            'avg_expectancy': runner_up['avg_expectancy'],
            'trades': runner_up['trades'],
        }
        leader['pnl_edge_vs_runner_up'] = round(leader['total_pnl'] - runner_up['total_pnl'], 4)
    return leader


def build_profile(profile: str, path: Path) -> dict[str, Any]:
    rows = load_json(path)
    trader_summaries = [summarize_trader(r) for r in rows]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for summary in trader_summaries:
        grouped[(summary['source_trader'], summary['symbol'])].append(summary)

    leaders = []
    nonzero = 0
    for (source_trader, symbol), candidates in sorted(grouped.items()):
        leader = choose_leader(candidates)
        leader['source_trader'] = source_trader
        leader['symbol'] = symbol
        leader['candidate_count'] = len(candidates)
        leaders.append(leader)
        if any(c['trades'] > 0 for c in candidates):
            nonzero += 1

    leaders_sorted = sorted(leaders, key=lambda x: (
        x['total_pnl'], -x['max_drawdown'], x['avg_expectancy'], x['fill_rate'], -x['timeout_rate'], x['trades']
    ), reverse=True)

    return {
        'profile': profile,
        'clone_traders': len(rows),
        'source_cells': len(grouped),
        'source_cells_with_any_trades': nonzero,
        'leaders': leaders_sorted,
    }


def main() -> int:
    report = {
        'generatedAt': iso_now(),
        'profiles': {profile: build_profile(profile, path) for profile, path in CONFIGS.items()},
    }
    OUT_LATEST.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    OUT_STAMPED.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
