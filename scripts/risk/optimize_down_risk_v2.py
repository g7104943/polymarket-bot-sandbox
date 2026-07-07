#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import json
import math
import os
import sys
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PANEL = PROJECT_ROOT / "reports" / "down_risk_v2_panel.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "down_risk_v2_tuning_report.json"


@dataclass
class TradeEvent:
    ts: int
    pnl: float
    confidence: float
    win: int
    weight: float


@dataclass
class CellData:
    profile: str
    symbol: str
    cell_id: str
    base_down_conf_threshold: float
    trades: List[TradeEvent]


@dataclass
class Params:
    wr_soft: float
    wr_hard: float
    pnl_soft: float
    pnl_hard: float
    min_trades_soft: int
    min_trades_hard: int
    soft_extra_delta: float
    soft_bet_scale: float
    hard_hold_bars: int
    lookback_fast_hours: int
    min_trades_fast: int
    wr_fast_blocked: float
    pnl_fast_blocked: float
    release_checks: int
    release_wr: float
    release_pnl: float
    release_confirm_hours: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "downRiskWrSoft": self.wr_soft,
            "downRiskWrHard": self.wr_hard,
            "downRiskPnlSoft": self.pnl_soft,
            "downRiskPnlHard": self.pnl_hard,
            "downRiskMinTradesSoft": self.min_trades_soft,
            "downRiskMinTradesHard": self.min_trades_hard,
            "downRiskSoftExtraDelta": self.soft_extra_delta,
            "downRiskSoftBetScale": self.soft_bet_scale,
            "downRiskHardHoldBars": self.hard_hold_bars,
            "downRiskLookbackHoursFast": self.lookback_fast_hours,
            "downRiskMinTradesFast": self.min_trades_fast,
            "downRiskWrFastBlocked": self.wr_fast_blocked,
            "downRiskPnlFastBlocked": self.pnl_fast_blocked,
            "downRiskReleaseChecks": self.release_checks,
            "downRiskReleaseWr": self.release_wr,
            "downRiskReleasePnl": self.release_pnl,
            "downRiskReleaseConfirmHours": self.release_confirm_hours,
        }


def default_action_params(profile: str) -> Dict[str, Any]:
    return {
        "soft_extra_delta": 0.02,
        "soft_bet_scale": 0.60,
        "hard_hold_bars": 6 if profile == "70" else 4,
        "lookback_fast_hours": 2,
        "min_trades_fast": 4,
        "wr_fast_blocked": 0.25,
        "pnl_fast_blocked": -0.03,
        "release_checks": 3,
        "release_wr": 0.45,
        "release_pnl": -0.05,
        "release_confirm_hours": 6,
    }


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="DOWN 风控 V2.2：基于 K线/特征面板的分层超参")
    ap.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--window-days", type=int, default=180)
    ap.add_argument("--half-life-days", type=int, default=30)
    ap.add_argument("--assets", type=str, default="BTC,ETH")
    ap.add_argument("--recent-days", type=int, default=21)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--max-trades-per-cell", type=int, default=1500)
    ap.add_argument("--recent-retain-ratio", type=float, default=0.40)
    ap.add_argument("--progress-interval", type=int, default=400)
    ap.add_argument("--cv-protocol", type=str, default="walk_forward", choices=["walk_forward", "none"])
    ap.add_argument("--train-min-days", type=int, default=180)
    ap.add_argument("--test-days", type=int, default=14)
    ap.add_argument("--step-days", type=int, default=14)
    ap.add_argument("--purge-hours", type=int, default=24)
    ap.add_argument("--clusters", type=str, default="all", help="default_BTC,default_ETH,70_BTC,70_ETH or all")
    ap.add_argument("--checkpoint-dir", type=Path, default=None)
    ap.add_argument("--max-workers", type=int, default=1)
    return ap.parse_args()


def _json_write_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _json_read(path: Path) -> Optional[Dict[str, Any]]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


def _cluster_result_path(base: Optional[Path], cluster_key: str) -> Optional[Path]:
    if base is None:
        return None
    return base / cluster_key / "cluster_result.json"


def evenly_pick(items: List[TradeEvent], k: int) -> List[TradeEvent]:
    if k <= 0:
        return []
    if k >= len(items):
        return items
    n = len(items)
    out: List[TradeEvent] = []
    for i in range(k):
        idx = int(i * n / k)
        out.append(items[idx])
    return out


def downsample_trades(
    trades: List[TradeEvent],
    max_trades_per_cell: int,
    recent_retain_ratio: float,
) -> Tuple[List[TradeEvent], int, int]:
    total = len(trades)
    if max_trades_per_cell <= 0 or total <= max_trades_per_cell:
        return trades, total, total

    if max_trades_per_cell == 1:
        return [trades[-1]], total, 1

    recent_ratio = _clamp(float(recent_retain_ratio), 0.05, 0.95)
    recent_n = int(round(max_trades_per_cell * recent_ratio))
    recent_n = max(1, min(max_trades_per_cell - 1, recent_n))
    old_n = max_trades_per_cell - recent_n

    split = max(0, total - recent_n)
    older = trades[:split]
    recent = trades[split:]

    sampled_old = evenly_pick(older, old_n) if old_n > 0 else []
    sampled = sampled_old + recent
    if len(sampled) > max_trades_per_cell:
        sampled = sampled[-max_trades_per_cell:]
    return sampled, total, len(sampled)


def load_panel(panel_path: Path, assets: List[str]) -> pd.DataFrame:
    if not panel_path.exists():
        raise RuntimeError(f"panel not found: {panel_path}")
    df = pd.read_parquet(panel_path)
    required = {
        "ts", "profile", "traderName", "symbol", "cell_id", "confidence", "win", "pnl", "down_conf_threshold"
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"panel missing columns: {sorted(missing)}")

    df = df.copy()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df = df[df["symbol"].isin(assets)]
    if df.empty:
        raise RuntimeError("panel empty after asset filter")

    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["win"] = pd.to_numeric(df["win"], errors="coerce")
    df["pnl"] = pd.to_numeric(df["pnl"], errors="coerce")
    df["down_conf_threshold"] = pd.to_numeric(df["down_conf_threshold"], errors="coerce")
    df = df.dropna(subset=["ts", "confidence", "win", "pnl", "down_conf_threshold"])
    if df.empty:
        raise RuntimeError("panel empty after numeric cleanup")

    df["profile"] = df["profile"].astype(str)
    df["traderName"] = df["traderName"].astype(str)
    df["cell_id"] = df["cell_id"].astype(str)
    return df


def build_cells(
    df: pd.DataFrame,
    cutoff_ts: int,
    half_life_days: int,
    max_trades_per_cell: int,
    recent_retain_ratio: float,
) -> Tuple[Dict[Tuple[str, str], List[CellData]], Dict[str, Any]]:
    max_ts = int(df["ts"].max())
    x = df[df["ts"] >= cutoff_ts].copy()
    if x.empty:
        raise RuntimeError("no rows in optimization window")

    half_life_sec = max(1, int(half_life_days) * 86400)
    ln2 = math.log(2.0)
    x["age_sec"] = (max_ts - x["ts"]).clip(lower=0)
    x["weight"] = x["age_sec"].apply(lambda z: math.exp(-ln2 * float(z) / float(half_life_sec)))

    buckets: Dict[Tuple[str, str], List[CellData]] = {
        ("default", "BTC"): [],
        ("default", "ETH"): [],
        ("70", "BTC"): [],
        ("70", "ETH"): [],
    }
    bucket_raw_trades: Dict[Tuple[str, str], int] = {k: 0 for k in buckets}
    bucket_sampled_trades: Dict[Tuple[str, str], int] = {k: 0 for k in buckets}

    for (profile, symbol, cell_id), g in x.groupby(["profile", "symbol", "cell_id"], sort=False):
        if (profile, symbol) not in buckets:
            continue
        g = g.sort_values("ts")
        base_thr = float(g["down_conf_threshold"].median())
        trades_raw = [
            TradeEvent(
                ts=int(r.ts),
                pnl=float(r.pnl),
                confidence=float(r.confidence),
                win=int(r.win),
                weight=float(r.weight),
            )
            for r in g.itertuples(index=False)
        ]
        if not trades_raw:
            continue
        trades, before_n, after_n = downsample_trades(
            trades_raw,
            max_trades_per_cell=max_trades_per_cell,
            recent_retain_ratio=recent_retain_ratio,
        )
        bucket_raw_trades[(profile, symbol)] += int(before_n)
        bucket_sampled_trades[(profile, symbol)] += int(after_n)
        buckets[(profile, symbol)].append(
            CellData(
                profile=profile,
                symbol=symbol,
                cell_id=str(cell_id),
                base_down_conf_threshold=_clamp(base_thr, 0.01, 0.999),
                trades=trades,
            )
        )

    meta = {
        "rows_window": int(len(x)),
        "max_ts": max_ts,
        "min_ts": int(x["ts"].min()),
        "cells": {f"{k[0]}_{k[1]}": len(v) for k, v in buckets.items()},
        "trades_before_sampling": {f"{k[0]}_{k[1]}": int(v) for k, v in bucket_raw_trades.items()},
        "trades_after_sampling": {f"{k[0]}_{k[1]}": int(v) for k, v in bucket_sampled_trades.items()},
        "sampling": {
            "maxTradesPerCell": int(max_trades_per_cell),
            "recentRetainRatio": float(recent_retain_ratio),
        },
    }
    return buckets, meta


def wr_post(wins: int, n: int) -> float:
    return (wins + 2.0) / (n + 4.0)


def trim_window(window: deque[Tuple[int, int, float]], cutoff_ts: int) -> None:
    while window and window[0][0] < cutoff_ts:
        window.popleft()


def compute_stats(window: deque[Tuple[int, int, float]]) -> Tuple[int, int, float, float]:
    n = len(window)
    if n <= 0:
        return 0, 0, 0.0, 0.5
    wins = sum(x[1] for x in window)
    pnl = sum(x[2] for x in window)
    return n, wins, pnl, wr_post(wins, n)


def build_walk_forward_folds(
    min_ts: int,
    max_ts: int,
    train_min_days: int,
    test_days: int,
    step_days: int,
    purge_hours: int,
) -> List[Tuple[int, int, int, int]]:
    folds: List[Tuple[int, int, int, int]] = []
    train_start = int(min_ts)
    train_end = int(min_ts + max(1, int(train_min_days)) * 86400)
    purge_sec = max(0, int(purge_hours)) * 3600
    test_sec = max(1, int(test_days)) * 86400
    step_sec = max(1, int(step_days)) * 86400
    while train_end + purge_sec + test_sec <= int(max_ts):
        test_start = train_end + purge_sec
        test_end = test_start + test_sec
        folds.append((train_start, train_end, test_start, test_end))
        train_end += step_sec
    return folds


def baseline_cell_metrics_interval(cell: CellData, start_ts: int, end_ts: int, history_start_ts: int) -> Dict[str, float]:
    rows = [t for t in cell.trades if history_start_ts <= t.ts < end_ts]
    input_w = 0.0
    executed_w = 0.0
    wins_w = 0.0
    hard_blocked_w = 0.0
    suppressed_w = 0.0
    net_pnl = 0.0
    equity = 0.0
    peak = 0.0
    mdd = 0.0

    for t in rows:
        in_test = start_ts <= t.ts < end_ts
        if in_test:
            input_w += t.weight
            executed_w += t.weight
            wins_w += t.weight * (1.0 if t.win else 0.0)
            pnl_eff = t.pnl
            net_pnl += pnl_eff * t.weight
            equity += pnl_eff * t.weight
            peak = max(peak, equity)
            mdd = max(mdd, peak - equity)
    return {
        "input": input_w,
        "executed": executed_w,
        "wins": wins_w,
        "net_pnl": net_pnl,
        "mdd": mdd,
        "hard_blocked": hard_blocked_w,
        "suppressed": suppressed_w,
    }


def simulate_cell_interval(
    cell: CellData,
    p: Params,
    start_ts: int,
    end_ts: int,
    history_start_ts: int,
) -> Dict[str, float]:
    rows = [t for t in cell.trades if history_start_ts <= t.ts < end_ts]

    window6: deque[Tuple[int, int, float]] = deque()
    window24: deque[Tuple[int, int, float]] = deque()
    window_fast: deque[Tuple[int, int, float]] = deque()
    window_release_confirm: deque[Tuple[int, int, float]] = deque()
    tier = "normal"
    hard_until_ts = 0
    release_streak = 0

    input_w = 0.0
    executed_w = 0.0
    wins_w = 0.0
    hard_blocked_w = 0.0
    suppressed_w = 0.0
    net_pnl = 0.0
    equity = 0.0
    peak = 0.0
    mdd = 0.0

    for t in rows:
        trim_window(window6, t.ts - 6 * 3600)
        trim_window(window24, t.ts - 24 * 3600)
        trim_window(window_fast, t.ts - p.lookback_fast_hours * 3600)
        trim_window(window_release_confirm, t.ts - p.release_confirm_hours * 3600)
        n6, _, pnl6, wr6 = compute_stats(window6)
        n24, _, pnl24, wr24 = compute_stats(window24)
        n_fast, _, pnl_fast, wr_fast = compute_stats(window_fast)
        n_rel_confirm, _, pnl_rel_confirm, wr_rel_confirm = compute_stats(window_release_confirm)
        r6 = (pnl6 / n6) if n6 > 0 else 0.0
        r24 = (pnl24 / n24) if n24 > 0 else 0.0
        r_fast = (pnl_fast / n_fast) if n_fast > 0 else 0.0
        r_rel_confirm = (pnl_rel_confirm / n_rel_confirm) if n_rel_confirm > 0 else 0.0

        soft_trigger = n6 >= p.min_trades_soft and (wr6 < p.wr_soft or r6 < p.pnl_soft)
        hard_trigger_main = n6 >= p.min_trades_hard and (wr6 < p.wr_hard and r6 < p.pnl_hard)
        hard_trigger_fast = (
            n_fast >= p.min_trades_fast
            and (wr_fast < p.wr_fast_blocked or r_fast < p.pnl_fast_blocked)
        )
        hard_trigger = hard_trigger_main or hard_trigger_fast
        next_tier = "hard" if hard_trigger else ("soft" if soft_trigger else "normal")

        if wr24 >= 0.50 and r24 >= 0:
            if next_tier == "hard":
                next_tier = "soft"
            elif next_tier == "soft":
                next_tier = "normal"

        if tier != "hard" and next_tier == "hard":
            hard_until_ts = t.ts + p.hard_hold_bars * 900
            release_streak = 0

        if tier == "hard":
            if t.ts < hard_until_ts:
                next_tier = "hard"
            else:
                release_pass_main = wr6 >= p.release_wr and r6 >= p.release_pnl
                release_pass_confirm = wr_rel_confirm >= p.release_wr and r_rel_confirm >= p.release_pnl
                release_pass = release_pass_main and release_pass_confirm
                if release_pass:
                    release_streak += 1
                    if release_streak >= p.release_checks:
                        next_tier = "soft" if soft_trigger else "normal"
                        release_streak = 0
                        hard_until_ts = 0
                    else:
                        next_tier = "hard"
                else:
                    release_streak = 0
                    next_tier = "hard"
                    hard_until_ts = t.ts + p.hard_hold_bars * 900

        tier = next_tier
        blocked = tier == "hard" and t.ts < hard_until_ts
        required_conf = cell.base_down_conf_threshold + (p.soft_extra_delta if tier == "soft" else 0.0)
        required_conf = _clamp(required_conf, 0.01, 0.999)
        scale = p.soft_bet_scale if tier == "soft" else 1.0
        passes_threshold = t.confidence >= required_conf

        if t.ts >= start_ts:
            input_w += t.weight
            if blocked:
                hard_blocked_w += t.weight
                suppressed_w += t.weight
            elif not passes_threshold:
                suppressed_w += t.weight
            else:
                pnl_eff = t.pnl * scale
                executed_w += t.weight
                wins_w += t.weight * (1.0 if t.win else 0.0)
                net_pnl += pnl_eff * t.weight
                equity += pnl_eff * t.weight
                peak = max(peak, equity)
                mdd = max(mdd, peak - equity)

        if blocked or not passes_threshold:
            continue

        pnl_eff_all = t.pnl * scale
        window6.append((t.ts, 1 if t.win else 0, pnl_eff_all))
        window24.append((t.ts, 1 if t.win else 0, pnl_eff_all))
        window_fast.append((t.ts, 1 if t.win else 0, pnl_eff_all))
        window_release_confirm.append((t.ts, 1 if t.win else 0, pnl_eff_all))

    return {
        "input": input_w,
        "executed": executed_w,
        "wins": wins_w,
        "net_pnl": net_pnl,
        "mdd": mdd,
        "hard_blocked": hard_blocked_w,
        "suppressed": suppressed_w,
    }


def evaluate_cluster_oos(
    cells: List[CellData],
    p: Params,
    folds: List[Tuple[int, int, int, int]],
) -> Dict[str, float]:
    rows: List[Dict[str, float]] = []
    for train_start, _train_end, test_start, test_end in folds:
        rows.extend(
            simulate_cell_interval(
                c,
                p,
                start_ts=test_start,
                end_ts=test_end,
                history_start_ts=train_start,
            )
            for c in cells
        )
    return aggregate_metrics(rows)


def baseline_cluster_oos(
    cells: List[CellData],
    folds: List[Tuple[int, int, int, int]],
) -> Dict[str, float]:
    rows: List[Dict[str, float]] = []
    for train_start, _train_end, test_start, test_end in folds:
        rows.extend(
            baseline_cell_metrics_interval(
                c,
                start_ts=test_start,
                end_ts=test_end,
                history_start_ts=train_start,
            )
            for c in cells
        )
    return aggregate_metrics(rows)


def baseline_cell_metrics(cell: CellData, cutoff_ts: Optional[int]) -> Dict[str, float]:
    rows = cell.trades if cutoff_ts is None else [t for t in cell.trades if t.ts >= cutoff_ts]
    input_w = 0.0
    executed_w = 0.0
    wins_w = 0.0
    hard_blocked_w = 0.0
    suppressed_w = 0.0
    net_pnl = 0.0
    equity = 0.0
    peak = 0.0
    mdd = 0.0

    for t in rows:
        input_w += t.weight
        executed_w += t.weight
        wins_w += t.weight * (1.0 if t.win else 0.0)
        pnl_eff = t.pnl
        net_pnl += pnl_eff * t.weight
        equity += pnl_eff * t.weight
        peak = max(peak, equity)
        mdd = max(mdd, peak - equity)

    return {
        "input": input_w,
        "executed": executed_w,
        "wins": wins_w,
        "net_pnl": net_pnl,
        "mdd": mdd,
        "hard_blocked": hard_blocked_w,
        "suppressed": suppressed_w,
    }


def simulate_cell(cell: CellData, p: Params, cutoff_ts: Optional[int]) -> Dict[str, float]:
    rows = cell.trades if cutoff_ts is None else [t for t in cell.trades if t.ts >= cutoff_ts]

    window6: deque[Tuple[int, int, float]] = deque()
    window24: deque[Tuple[int, int, float]] = deque()
    window_fast: deque[Tuple[int, int, float]] = deque()
    window_release_confirm: deque[Tuple[int, int, float]] = deque()

    tier = "normal"
    hard_until_ts = 0
    release_streak = 0

    input_w = 0.0
    executed_w = 0.0
    wins_w = 0.0
    hard_blocked_w = 0.0
    suppressed_w = 0.0

    net_pnl = 0.0
    equity = 0.0
    peak = 0.0
    mdd = 0.0

    for t in rows:
        input_w += t.weight

        trim_window(window6, t.ts - 6 * 3600)
        trim_window(window24, t.ts - 24 * 3600)
        trim_window(window_fast, t.ts - p.lookback_fast_hours * 3600)
        trim_window(window_release_confirm, t.ts - p.release_confirm_hours * 3600)
        n6, _, pnl6, wr6 = compute_stats(window6)
        n24, _, pnl24, wr24 = compute_stats(window24)
        n_fast, _, pnl_fast, wr_fast = compute_stats(window_fast)
        n_rel_confirm, _, pnl_rel_confirm, wr_rel_confirm = compute_stats(window_release_confirm)
        r6 = (pnl6 / n6) if n6 > 0 else 0.0
        r24 = (pnl24 / n24) if n24 > 0 else 0.0
        r_fast = (pnl_fast / n_fast) if n_fast > 0 else 0.0
        r_rel_confirm = (pnl_rel_confirm / n_rel_confirm) if n_rel_confirm > 0 else 0.0

        soft_trigger = n6 >= p.min_trades_soft and (wr6 < p.wr_soft or r6 < p.pnl_soft)
        hard_trigger_main = n6 >= p.min_trades_hard and (wr6 < p.wr_hard and r6 < p.pnl_hard)
        hard_trigger_fast = (
            n_fast >= p.min_trades_fast
            and (wr_fast < p.wr_fast_blocked or r_fast < p.pnl_fast_blocked)
        )
        hard_trigger = hard_trigger_main or hard_trigger_fast
        next_tier = "hard" if hard_trigger else ("soft" if soft_trigger else "normal")

        if wr24 >= 0.50 and r24 >= 0:
            if next_tier == "hard":
                next_tier = "soft"
            elif next_tier == "soft":
                next_tier = "normal"

        if tier != "hard" and next_tier == "hard":
            hard_until_ts = t.ts + p.hard_hold_bars * 900
            release_streak = 0

        if tier == "hard":
            if t.ts < hard_until_ts:
                next_tier = "hard"
            else:
                release_pass_main = wr6 >= p.release_wr and r6 >= p.release_pnl
                release_pass_confirm = wr_rel_confirm >= p.release_wr and r_rel_confirm >= p.release_pnl
                release_pass = release_pass_main and release_pass_confirm
                if release_pass:
                    release_streak += 1
                    if release_streak >= p.release_checks:
                        next_tier = "soft" if soft_trigger else "normal"
                        release_streak = 0
                        hard_until_ts = 0
                    else:
                        next_tier = "hard"
                else:
                    release_streak = 0
                    next_tier = "hard"
                    hard_until_ts = t.ts + p.hard_hold_bars * 900

        tier = next_tier
        blocked = tier == "hard" and t.ts < hard_until_ts
        if blocked:
            hard_blocked_w += t.weight
            suppressed_w += t.weight
            continue

        required_conf = cell.base_down_conf_threshold + (p.soft_extra_delta if tier == "soft" else 0.0)
        required_conf = _clamp(required_conf, 0.01, 0.999)
        if t.confidence < required_conf:
            suppressed_w += t.weight
            continue

        scale = p.soft_bet_scale if tier == "soft" else 1.0
        pnl_eff = t.pnl * scale

        executed_w += t.weight
        wins_w += t.weight * (1.0 if t.win else 0.0)
        net_pnl += pnl_eff * t.weight
        equity += pnl_eff * t.weight
        peak = max(peak, equity)
        mdd = max(mdd, peak - equity)

        window6.append((t.ts, 1 if t.win else 0, pnl_eff))
        window24.append((t.ts, 1 if t.win else 0, pnl_eff))
        window_fast.append((t.ts, 1 if t.win else 0, pnl_eff))
        window_release_confirm.append((t.ts, 1 if t.win else 0, pnl_eff))

    return {
        "input": input_w,
        "executed": executed_w,
        "wins": wins_w,
        "net_pnl": net_pnl,
        "mdd": mdd,
        "hard_blocked": hard_blocked_w,
        "suppressed": suppressed_w,
    }


def aggregate_metrics(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    out = {
        "input": 0.0,
        "executed": 0.0,
        "wins": 0.0,
        "net_pnl": 0.0,
        "mdd": 0.0,
        "hard_blocked": 0.0,
        "suppressed": 0.0,
    }
    for r in rows:
        for k in out:
            out[k] += float(r.get(k, 0.0))
    out["win_rate"] = (out["wins"] / out["executed"]) if out["executed"] > 0 else 0.0
    out["hard_block_rate"] = (out["hard_blocked"] / out["input"]) if out["input"] > 0 else 0.0
    out["trade_suppression"] = (out["suppressed"] / out["input"]) if out["input"] > 0 else 0.0
    return out


def score_metrics(m: Dict[str, float]) -> float:
    return (
        0.45 * m["net_pnl"]
        - 0.35 * m["mdd"]
        - 0.15 * m["hard_block_rate"]
        - 0.05 * m["trade_suppression"]
    )


def satisfies_constraints(candidate: Dict[str, float], baseline: Dict[str, float]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    b_pnl = baseline["net_pnl"]
    c_pnl = candidate["net_pnl"]
    if b_pnl > 0 and c_pnl < b_pnl * 0.95:
        reasons.append("net_pnl_drop_gt_5pct")

    b_mdd = baseline["mdd"]
    c_mdd = candidate["mdd"]
    if b_mdd > 0 and c_mdd > b_mdd * 0.85:
        reasons.append("mdd_improve_lt_15pct")

    if candidate["hard_block_rate"] > 0.35:
        reasons.append("hard_block_rate_gt_35pct")

    return (len(reasons) == 0, reasons)


def evaluate_cluster(cells: List[CellData], p: Params, cutoff_ts: Optional[int]) -> Dict[str, float]:
    return aggregate_metrics(simulate_cell(c, p, cutoff_ts=cutoff_ts) for c in cells)


def baseline_cluster(cells: List[CellData], cutoff_ts: Optional[int]) -> Dict[str, float]:
    return aggregate_metrics(baseline_cell_metrics(c, cutoff_ts=cutoff_ts) for c in cells)


def _chunk_items(items: List[Any], workers: int) -> List[List[Any]]:
    if workers <= 1 or len(items) <= 1:
        return [list(items)]
    chunk_size = max(1, int(math.ceil(len(items) / float(max(1, workers)))))
    return [list(items[idx : idx + chunk_size]) for idx in range(0, len(items), chunk_size)]


def _evaluate_down_trigger_chunk(payload: Dict[str, Any]) -> Dict[str, Any]:
    cells: List[CellData] = list(payload["cells"])
    folds = list(payload["folds"])
    baseline_oos = dict(payload["baseline_oos"])
    defaults = dict(payload["defaults"])
    trigger_grid = list(payload["trigger_grid"])
    candidates: List[Tuple[float, Params]] = []
    for wr_soft, wr_hard, pnl_soft, pnl_hard, min_soft, min_hard in trigger_grid:
        p = Params(
            wr_soft=wr_soft,
            wr_hard=wr_hard,
            pnl_soft=pnl_soft,
            pnl_hard=pnl_hard,
            min_trades_soft=min_soft,
            min_trades_hard=min_hard,
            soft_extra_delta=defaults["soft_extra_delta"],
            soft_bet_scale=defaults["soft_bet_scale"],
            hard_hold_bars=defaults["hard_hold_bars"],
            lookback_fast_hours=defaults["lookback_fast_hours"],
            min_trades_fast=defaults["min_trades_fast"],
            wr_fast_blocked=defaults["wr_fast_blocked"],
            pnl_fast_blocked=defaults["pnl_fast_blocked"],
            release_checks=defaults["release_checks"],
            release_wr=defaults["release_wr"],
            release_pnl=defaults["release_pnl"],
            release_confirm_hours=defaults["release_confirm_hours"],
        )
        moos = evaluate_cluster_oos(cells, p, folds=folds)
        ok_oos, _ = satisfies_constraints(moos, baseline_oos)
        s = score_metrics(moos)
        if not ok_oos:
            s -= 1e9
        candidates.append((float(s), p))
    return {"candidates": candidates, "evaluated": len(trigger_grid)}


def _evaluate_down_action_chunk(payload: Dict[str, Any]) -> Dict[str, Any]:
    cells: List[CellData] = list(payload["cells"])
    folds = list(payload["folds"])
    baseline_180 = dict(payload["baseline_180"])
    baseline_oos = dict(payload["baseline_oos"])
    recent_cutoff_ts = int(payload["recent_cutoff_ts"])
    trigger_p: Params = payload["trigger_params"]
    action_rows = list(payload["action_rows"])

    feasible_count = 0
    best_score = -10**18
    best_params: Optional[Params] = None
    best_180: Optional[Dict[str, float]] = None
    best_oos: Optional[Dict[str, float]] = None
    best_21: Optional[Dict[str, float]] = None
    best_reasons: List[str] = []

    for soft_delta, soft_scale, hold_bars, rel_checks, rel_wr, rel_pnl in action_rows:
        p = Params(
            wr_soft=trigger_p.wr_soft,
            wr_hard=trigger_p.wr_hard,
            pnl_soft=trigger_p.pnl_soft,
            pnl_hard=trigger_p.pnl_hard,
            min_trades_soft=trigger_p.min_trades_soft,
            min_trades_hard=trigger_p.min_trades_hard,
            soft_extra_delta=soft_delta,
            soft_bet_scale=soft_scale,
            hard_hold_bars=hold_bars,
            lookback_fast_hours=trigger_p.lookback_fast_hours,
            min_trades_fast=trigger_p.min_trades_fast,
            wr_fast_blocked=trigger_p.wr_fast_blocked,
            pnl_fast_blocked=trigger_p.pnl_fast_blocked,
            release_checks=rel_checks,
            release_wr=rel_wr,
            release_pnl=rel_pnl,
            release_confirm_hours=trigger_p.release_confirm_hours,
        )
        m180 = evaluate_cluster(cells, p, cutoff_ts=None)
        m21 = evaluate_cluster(cells, p, cutoff_ts=recent_cutoff_ts)
        moos = evaluate_cluster_oos(cells, p, folds=folds)
        ok180, reasons180 = satisfies_constraints(m180, baseline_180)
        ok_oos, reasons_oos = satisfies_constraints(moos, baseline_oos)
        ok = ok180 and ok_oos
        reasons = list(reasons180) + [f"oos_{r}" for r in reasons_oos]
        s = score_metrics(moos)
        if ok:
            feasible_count += 1
        else:
            s -= 1e9
        if s > best_score:
            best_score = s
            best_params = p
            best_180 = m180
            best_oos = moos
            best_21 = m21
            best_reasons = reasons
    return {
        "best_score": float(best_score),
        "best_params": best_params,
        "best_180": best_180,
        "best_oos": best_oos,
        "best_21": best_21,
        "best_reasons": list(best_reasons),
        "feasible_count": int(feasible_count),
        "evaluated": len(action_rows),
    }


def _evaluate_down_fast_refine_chunk(payload: Dict[str, Any]) -> Dict[str, Any]:
    cells: List[CellData] = list(payload["cells"])
    folds = list(payload["folds"])
    baseline_180 = dict(payload["baseline_180"])
    baseline_oos = dict(payload["baseline_oos"])
    recent_cutoff_ts = int(payload["recent_cutoff_ts"])
    base: Params = payload["base_params"]
    grid = list(payload["grid"])

    feasible_count = 0
    best_score = -10**18
    best_params = base
    best_180 = evaluate_cluster(cells, base, cutoff_ts=None)
    best_21 = evaluate_cluster(cells, base, cutoff_ts=recent_cutoff_ts)
    best_oos = evaluate_cluster_oos(cells, base, folds=folds)
    ok180, reasons180 = satisfies_constraints(best_180, baseline_180)
    ok_oos, reasons_oos = satisfies_constraints(best_oos, baseline_oos)
    best_reasons = list(reasons180) + [f"oos_{r}" for r in reasons_oos]
    if ok180 and ok_oos:
        best_score = score_metrics(best_oos)
        feasible_count = 1
    else:
        best_score = score_metrics(best_oos) - 1e9

    for lookback_fast_hours, min_trades_fast, wr_fast_blocked, pnl_fast_blocked, release_confirm_hours in grid:
        p = Params(
            wr_soft=base.wr_soft,
            wr_hard=base.wr_hard,
            pnl_soft=base.pnl_soft,
            pnl_hard=base.pnl_hard,
            min_trades_soft=base.min_trades_soft,
            min_trades_hard=base.min_trades_hard,
            soft_extra_delta=base.soft_extra_delta,
            soft_bet_scale=base.soft_bet_scale,
            hard_hold_bars=base.hard_hold_bars,
            lookback_fast_hours=int(lookback_fast_hours),
            min_trades_fast=int(min_trades_fast),
            wr_fast_blocked=float(wr_fast_blocked),
            pnl_fast_blocked=float(pnl_fast_blocked),
            release_checks=base.release_checks,
            release_wr=base.release_wr,
            release_pnl=base.release_pnl,
            release_confirm_hours=int(release_confirm_hours),
        )
        m180 = evaluate_cluster(cells, p, cutoff_ts=None)
        m21 = evaluate_cluster(cells, p, cutoff_ts=recent_cutoff_ts)
        moos = evaluate_cluster_oos(cells, p, folds=folds)
        ok180, reasons180 = satisfies_constraints(m180, baseline_180)
        ok_oos, reasons_oos = satisfies_constraints(moos, baseline_oos)
        ok = ok180 and ok_oos
        reasons = list(reasons180) + [f"oos_{r}" for r in reasons_oos]
        s = score_metrics(moos)
        if ok:
            feasible_count += 1
        else:
            s -= 1e9
        if s > best_score:
            best_score = s
            best_params = p
            best_180 = m180
            best_21 = m21
            best_oos = moos
            best_reasons = reasons
    return {
        "best_score": float(best_score),
        "best_params": best_params,
        "best_180": best_180,
        "best_oos": best_oos,
        "best_21": best_21,
        "best_reasons": list(best_reasons),
        "feasible_count": int(feasible_count),
        "evaluated": len(grid),
    }


def search_cluster(
    profile: str,
    symbol: str,
    cells: List[CellData],
    recent_cutoff_ts: int,
    top_k: int,
    cv_protocol: str,
    train_min_days: int,
    test_days: int,
    step_days: int,
    purge_hours: int,
    progress_interval: int,
    inner_workers: int = 1,
) -> Dict[str, Any]:
    baseline_180 = baseline_cluster(cells, cutoff_ts=None)
    baseline_21 = baseline_cluster(cells, cutoff_ts=recent_cutoff_ts)
    all_min_ts = min(t.ts for c in cells for t in c.trades)
    all_max_ts = max(t.ts for c in cells for t in c.trades)
    if cv_protocol == "walk_forward":
        folds = build_walk_forward_folds(
            min_ts=all_min_ts,
            max_ts=all_max_ts,
            train_min_days=train_min_days,
            test_days=test_days,
            step_days=step_days,
            purge_hours=purge_hours,
        )
        if not folds:
            # 数据不足时，退化为单个近窗 OOS（仍保持时间先后）
            test_end = all_max_ts
            test_start = max(all_min_ts, test_end - max(1, test_days) * 86400)
            train_end = max(all_min_ts, test_start - max(0, purge_hours) * 3600)
            folds = [(all_min_ts, train_end, test_start, test_end)]
    else:
        folds = [(all_min_ts, all_max_ts, recent_cutoff_ts, all_max_ts)]
    baseline_oos = baseline_cluster_oos(cells, folds=folds)
    baseline_score_180 = score_metrics(baseline_180)
    baseline_score_21 = score_metrics(baseline_21)
    baseline_score_oos = score_metrics(baseline_oos)

    cluster_name = f"{profile}_{symbol}"
    defaults = default_action_params(profile)

    def refine_fast_lane(base: Params) -> Dict[str, Any]:
        best_score = -10**18
        best_params = base
        best_180 = evaluate_cluster(cells, base, cutoff_ts=None)
        best_21 = evaluate_cluster(cells, base, cutoff_ts=recent_cutoff_ts)
        best_oos = evaluate_cluster_oos(cells, base, folds=folds)
        ok180, reasons180 = satisfies_constraints(best_180, baseline_180)
        ok_oos, reasons_oos = satisfies_constraints(best_oos, baseline_oos)
        best_reasons = list(reasons180) + [f"oos_{r}" for r in reasons_oos]
        if ok180 and ok_oos:
            best_score = score_metrics(best_oos)
        else:
            best_score = score_metrics(best_oos) - 1e9

        evaluated = 0
        feasible_count = 0
        grid = list(itertools.product(
            [1, 2, 3],
            [2, 3, 4, 5],
            [0.20, 0.25, 0.30, 0.35],
            [-0.01, -0.02, -0.03, -0.04],
            [4, 6, 8, 10],
        ))
        total = len(grid)
        chunks = _chunk_items(grid, inner_workers)
        if inner_workers > 1 and len(chunks) > 1:
            with ProcessPoolExecutor(max_workers=min(inner_workers, len(chunks))) as executor:
                future_map = {
                    executor.submit(
                        _evaluate_down_fast_refine_chunk,
                        {
                            "cells": cells,
                            "folds": folds,
                            "baseline_180": baseline_180,
                            "baseline_oos": baseline_oos,
                            "recent_cutoff_ts": recent_cutoff_ts,
                            "base_params": base,
                            "grid": chunk,
                        },
                    ): chunk
                    for chunk in chunks
                }
                for future in as_completed(future_map):
                    row = future.result()
                    evaluated += int(row.get("evaluated") or 0)
                    feasible_count += int(row.get("feasible_count") or 0)
                    s = float(row.get("best_score") or -10**18)
                    if s > best_score:
                        best_score = s
                        best_params = row.get("best_params") or best_params
                        best_180 = row.get("best_180") or best_180
                        best_21 = row.get("best_21") or best_21
                        best_oos = row.get("best_oos") or best_oos
                        best_reasons = list(row.get("best_reasons") or best_reasons)
                    if progress_interval > 0:
                        print(f"[cluster {cluster_name}] fast_refine {min(evaluated, total)}/{total}")
        else:
            row = _evaluate_down_fast_refine_chunk(
                {
                    "cells": cells,
                    "folds": folds,
                    "baseline_180": baseline_180,
                    "baseline_oos": baseline_oos,
                    "recent_cutoff_ts": recent_cutoff_ts,
                    "base_params": base,
                    "grid": grid,
                }
            )
            evaluated = int(row.get("evaluated") or 0)
            feasible_count = int(row.get("feasible_count") or 0)
            best_score = float(row.get("best_score") or best_score)
            best_params = row.get("best_params") or best_params
            best_180 = row.get("best_180") or best_180
            best_21 = row.get("best_21") or best_21
            best_oos = row.get("best_oos") or best_oos
            best_reasons = list(row.get("best_reasons") or best_reasons)
            if progress_interval > 0:
                print(f"[cluster {cluster_name}] fast_refine {min(evaluated, total)}/{total}")
        return {
            "best_score": float(best_score),
            "best_params": best_params,
            "best_180": best_180,
            "best_oos": best_oos,
            "best_21": best_21,
            "best_reasons": list(best_reasons),
            "feasible_count": int(feasible_count),
            "evaluated": int(evaluated),
            "search_pass": "fast_refine",
        }

    def build_trigger_grid(
        wr_soft_values: List[float],
        wr_hard_values: List[float],
        pnl_soft_values: List[float],
        pnl_hard_values: List[float],
        min_soft_values: List[int],
        min_hard_values: List[int],
    ) -> List[Tuple[float, float, float, float, int, int]]:
        out: List[Tuple[float, float, float, float, int, int]] = []
        for wr_soft, wr_hard, pnl_soft, pnl_hard, min_soft, min_hard in itertools.product(
            wr_soft_values,
            wr_hard_values,
            pnl_soft_values,
            pnl_hard_values,
            min_soft_values,
            min_hard_values,
        ):
            if wr_hard > wr_soft:
                continue
            if min_hard < min_soft:
                continue
            out.append((wr_soft, wr_hard, pnl_soft, pnl_hard, min_soft, min_hard))
        return out

    def run_search_pass(
        trigger_grid: List[Tuple[float, float, float, float, int, int]],
        action_grid: List[Tuple[float, float, int, int, float, float]],
        pass_label: str,
    ) -> Dict[str, Any]:
        trigger_candidates: List[Tuple[float, Params]] = []
        print(
            f"[cluster {cluster_name}] cells={len(cells)} {pass_label} "
            f"trigger_grid={len(trigger_grid)} top_k={int(max(1, int(top_k)))}"
        )
        trigger_chunks = _chunk_items(trigger_grid, inner_workers)
        if inner_workers > 1 and len(trigger_chunks) > 1:
            processed = 0
            with ProcessPoolExecutor(max_workers=min(inner_workers, len(trigger_chunks))) as executor:
                future_map = {
                    executor.submit(
                        _evaluate_down_trigger_chunk,
                        {
                            "cells": cells,
                            "folds": folds,
                            "baseline_oos": baseline_oos,
                            "defaults": defaults,
                            "trigger_grid": chunk,
                        },
                    ): chunk
                    for chunk in trigger_chunks
                }
                for future in as_completed(future_map):
                    row = future.result()
                    trigger_candidates.extend(list(row.get("candidates") or []))
                    processed += int(row.get("evaluated") or 0)
                    if progress_interval > 0:
                        print(f"[cluster {cluster_name}] {pass_label} trigger_phase {min(processed, len(trigger_grid))}/{len(trigger_grid)}")
        else:
            row = _evaluate_down_trigger_chunk(
                {
                    "cells": cells,
                    "folds": folds,
                    "baseline_oos": baseline_oos,
                    "defaults": defaults,
                    "trigger_grid": trigger_grid,
                }
            )
            trigger_candidates.extend(list(row.get("candidates") or []))
            if progress_interval > 0:
                print(f"[cluster {cluster_name}] {pass_label} trigger_phase {len(trigger_grid)}/{len(trigger_grid)}")

        trigger_candidates.sort(key=lambda x: x[0], reverse=True)
        top_candidates = trigger_candidates[: max(1, int(top_k))]
        print(f"[cluster {cluster_name}] {pass_label} action_grid={len(action_grid)}")

        feasible_count = 0
        best_score = -10**18
        best_params: Optional[Params] = None
        best_180: Optional[Dict[str, float]] = None
        best_oos: Optional[Dict[str, float]] = None
        best_21: Optional[Dict[str, float]] = None
        best_reasons: List[str] = []
        evaluated = 0

        total_actions = len(top_candidates) * len(action_grid)
        action_chunks_payloads: List[Dict[str, Any]] = []
        for _, trigger_p in top_candidates:
            for chunk in _chunk_items(action_grid, inner_workers):
                action_chunks_payloads.append(
                    {
                        "cells": cells,
                        "folds": folds,
                        "baseline_180": baseline_180,
                        "baseline_oos": baseline_oos,
                        "recent_cutoff_ts": recent_cutoff_ts,
                        "trigger_params": trigger_p,
                        "action_rows": chunk,
                    }
                )
        if inner_workers > 1 and len(action_chunks_payloads) > 1:
            action_idx = 0
            with ProcessPoolExecutor(max_workers=min(inner_workers, len(action_chunks_payloads))) as executor:
                future_map = {executor.submit(_evaluate_down_action_chunk, payload): payload for payload in action_chunks_payloads}
                for future in as_completed(future_map):
                    row = future.result()
                    evaluated += int(row.get("evaluated") or 0)
                    action_idx += int(row.get("evaluated") or 0)
                    feasible_count += int(row.get("feasible_count") or 0)
                    s = float(row.get("best_score") or -10**18)
                    if s > best_score:
                        best_score = s
                        best_params = row.get("best_params") or best_params
                        best_180 = row.get("best_180") or best_180
                        best_oos = row.get("best_oos") or best_oos
                        best_21 = row.get("best_21") or best_21
                        best_reasons = list(row.get("best_reasons") or best_reasons)
                    if progress_interval > 0:
                        print(f"[cluster {cluster_name}] {pass_label} action_phase {min(action_idx, total_actions)}/{total_actions}")
        else:
            for payload in action_chunks_payloads:
                row = _evaluate_down_action_chunk(payload)
                evaluated += int(row.get("evaluated") or 0)
                feasible_count += int(row.get("feasible_count") or 0)
                s = float(row.get("best_score") or -10**18)
                if s > best_score:
                    best_score = s
                    best_params = row.get("best_params") or best_params
                    best_180 = row.get("best_180") or best_180
                    best_oos = row.get("best_oos") or best_oos
                    best_21 = row.get("best_21") or best_21
                    best_reasons = list(row.get("best_reasons") or best_reasons)
                if progress_interval > 0:
                    print(f"[cluster {cluster_name}] {pass_label} action_phase {min(evaluated, total_actions)}/{total_actions}")

        assert best_params is not None and best_180 is not None and best_oos is not None and best_21 is not None
        return {
            "best_score": float(best_score),
            "best_params": best_params,
            "best_180": best_180,
            "best_oos": best_oos,
            "best_21": best_21,
            "best_reasons": list(best_reasons),
            "feasible_count": int(feasible_count),
            "trigger_grid_size": len(trigger_grid),
            "action_grid_size": len(action_grid),
            "evaluated": int(evaluated),
            "search_pass": pass_label,
        }

    pass1_trigger_grid = build_trigger_grid(
        wr_soft_values=[0.40, 0.42, 0.44, 0.46],
        wr_hard_values=[0.34, 0.36, 0.38, 0.40],
        pnl_soft_values=[-0.08, -0.10, -0.12, -0.15],
        pnl_hard_values=[-0.14, -0.18, -0.22, -0.25, -0.30],
        min_soft_values=[4, 6, 8, 10, 12],
        min_hard_values=[8, 10, 12, 14],
    )
    pass1_hold_set = [4, 6, 8, 10] if profile != "70" else [6, 8, 10, 12]
    pass1_action_grid = list(
        itertools.product(
            [0.02, 0.03, 0.04, 0.05, 0.06],
            [0.30, 0.40, 0.50, 0.60],
            pass1_hold_set,
            [3, 4, 5],
            [0.42, 0.45, 0.47, 0.50],
            [-0.08, -0.05, -0.03, 0.00, 0.02],
        )
    )
    pass1 = run_search_pass(pass1_trigger_grid, pass1_action_grid, "pass1")

    pass2: Optional[Dict[str, Any]] = None
    selected = pass1
    if pass1["best_reasons"]:
        pass2_trigger_grid = build_trigger_grid(
            wr_soft_values=[0.42, 0.44, 0.46, 0.48],
            wr_hard_values=[0.28, 0.30, 0.32, 0.34, 0.35],
            pnl_soft_values=[-0.10, -0.12, -0.15, -0.18],
            pnl_hard_values=[-0.22, -0.25, -0.30, -0.35, -0.40],
            min_soft_values=[6, 8, 10, 12, 14],
            min_hard_values=[10, 12, 14, 16, 18],
        )
        pass2_hold_set = [6, 8, 10, 12] if profile != "70" else [8, 10, 12, 14]
        pass2_action_grid = list(
            itertools.product(
                [0.03, 0.04, 0.05, 0.06, 0.07, 0.08],
                [0.20, 0.25, 0.30, 0.35, 0.40, 0.50],
                pass2_hold_set,
                [2, 3, 4, 5],
                [0.45, 0.47, 0.50, 0.52, 0.55],
                [-0.08, -0.05, -0.03, 0.00],
            )
        )
        pass2 = run_search_pass(pass2_trigger_grid, pass2_action_grid, "pass2")
        selected = min(
            [pass1, pass2],
            key=lambda x: (len(x["best_reasons"]), -float(x["best_score"])),
        )

    fast_refine = refine_fast_lane(selected["best_params"])
    selected = min(
        [selected, fast_refine],
        key=lambda x: (len(x["best_reasons"]), -float(x["best_score"])),
    )

    best_params = selected["best_params"]
    best_180 = selected["best_180"]
    best_oos = selected["best_oos"]
    best_21 = selected["best_21"]
    best_score = float(selected["best_score"])
    best_reasons = list(selected["best_reasons"])
    search_pass = str(selected["search_pass"])
    final_is_feasible = len(best_reasons) == 0

    min_ts = min(t.ts for c in cells for t in c.trades)
    max_ts = max(t.ts for c in cells for t in c.trades)
    span_days = (max_ts - min_ts) / 86400.0 if max_ts >= min_ts else 0.0

    return {
        "profile": profile,
        "symbol": symbol,
        "cells": len(cells),
        "input_trades_180": int(round(baseline_180["input"])),
        "data_span_days": span_days,
        "oldest_sample_utc": datetime.fromtimestamp(min_ts, tz=timezone.utc).isoformat(),
        "latest_sample_utc": datetime.fromtimestamp(max_ts, tz=timezone.utc).isoformat(),
        "baseline_180": {**baseline_180, "score": baseline_score_180},
        "baseline_21": {**baseline_21, "score": baseline_score_21},
        "baseline_oos": {**baseline_oos, "score": baseline_score_oos, "folds": len(folds)},
        "best": {
            "params": best_params.as_dict(),
            "metrics_180": {**best_180, "score": score_metrics(best_180)},
            "metrics_21": {**best_21, "score": score_metrics(best_21)},
            "metrics_oos": {**best_oos, "score": score_metrics(best_oos), "folds": len(folds)},
            "combined_score": best_score,
            "constraint_violations": best_reasons,
            "search_pass": search_pass,
            "needs_review": not final_is_feasible,
            "final_is_feasible": final_is_feasible,
            "delta_vs_baseline_180": {
                "net_pnl": best_180["net_pnl"] - baseline_180["net_pnl"],
                "mdd": best_180["mdd"] - baseline_180["mdd"],
                "hard_block_rate": best_180["hard_block_rate"] - baseline_180["hard_block_rate"],
                "trade_suppression": best_180["trade_suppression"] - baseline_180["trade_suppression"],
            },
            "delta_vs_baseline_oos": {
                "net_pnl": best_oos["net_pnl"] - baseline_oos["net_pnl"],
                "mdd": best_oos["mdd"] - baseline_oos["mdd"],
                "hard_block_rate": best_oos["hard_block_rate"] - baseline_oos["hard_block_rate"],
                "trade_suppression": best_oos["trade_suppression"] - baseline_oos["trade_suppression"],
            },
        },
        "search": {
            "trigger_grid_size": int(selected.get("trigger_grid_size", pass1.get("trigger_grid_size", 0))),
            "trigger_top_k": int(max(1, int(top_k))),
            "action_grid_size": int(selected.get("action_grid_size", pass1.get("action_grid_size", 0))),
            "evaluated_action_candidates": int(selected.get("evaluated", 0)),
            "feasible_count": int(selected.get("feasible_count", 0)),
            "search_pass": search_pass,
            "pass2_attempted": bool(pass2 is not None),
            "pass1_feasible_count": int(pass1["feasible_count"]),
            "pass2_feasible_count": int(pass2["feasible_count"]) if pass2 is not None else 0,
            "fast_refine_feasible_count": int(fast_refine["feasible_count"]),
            "fast_refine_evaluated": int(fast_refine["evaluated"]),
            "cv_protocol": cv_protocol,
            "cv_folds": len(folds),
            "train_min_days": int(train_min_days),
            "test_days": int(test_days),
            "step_days": int(step_days),
            "purge_hours": int(purge_hours),
        },
    }


def run_cluster_search_task(payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    return (
        str(payload["cluster_name"]),
        search_cluster(
            profile=str(payload["profile"]),
            symbol=str(payload["symbol"]),
            cells=list(payload["cells"]),
            recent_cutoff_ts=int(payload["recent_cutoff_ts"]),
            top_k=int(payload["top_k"]),
            cv_protocol=str(payload["cv_protocol"]),
            train_min_days=int(payload["train_min_days"]),
            test_days=int(payload["test_days"]),
            step_days=int(payload["step_days"]),
            purge_hours=int(payload["purge_hours"]),
            progress_interval=int(payload["progress_interval"]),
            inner_workers=int(payload.get("inner_workers") or 1),
        ),
    )


def main() -> int:
    args = parse_args()
    if args.checkpoint_dir is None:
        args.checkpoint_dir = args.output.parent / f"{args.output.stem}_checkpoint"
    assets = [x.strip().upper() for x in str(args.assets).split(",") if x.strip()]
    assets = [x for x in assets if x in {"BTC", "ETH"}]
    if not assets:
        raise RuntimeError("assets empty after filter; only BTC,ETH supported")

    df = load_panel(args.panel, assets)
    max_ts = int(df["ts"].max())
    cutoff_ts = max_ts - int(args.window_days) * 86400
    recent_cutoff_ts = max_ts - int(args.recent_days) * 86400

    buckets, panel_meta = build_cells(
        df,
        cutoff_ts=cutoff_ts,
        half_life_days=int(args.half_life_days),
        max_trades_per_cell=int(args.max_trades_per_cell),
        recent_retain_ratio=float(args.recent_retain_ratio),
    )

    all_cluster_keys = ["default_BTC", "default_ETH", "70_BTC", "70_ETH"]
    req = str(args.clusters or "all").strip()
    if req.lower() == "all":
        selected_clusters = set(all_cluster_keys)
    else:
        selected_clusters = {x.strip() for x in req.split(",") if x.strip() in set(all_cluster_keys)}
        if not selected_clusters:
            raise RuntimeError(f"--clusters invalid: {req}")

    clusters: Dict[str, Any] = {}
    pending_tasks: List[Dict[str, Any]] = []
    for key in [("default", "BTC"), ("default", "ETH"), ("70", "BTC"), ("70", "ETH")]:
        profile, symbol = key
        cluster_name = f"{profile}_{symbol}"
        if cluster_name not in selected_clusters:
            continue
        cluster_result_path = _cluster_result_path(args.checkpoint_dir, cluster_name)
        if cluster_result_path is not None and cluster_result_path.exists():
            cached = _json_read(cluster_result_path)
            if cached:
                clusters[cluster_name] = cached
                print(f"[resume] cluster_cached={cluster_name}", flush=True)
                continue
        cells = buckets.get(key, [])
        if not cells:
            raise RuntimeError(f"cluster has no cells: {profile}-{symbol}")
        pending_tasks.append(
            {
                "cluster_name": cluster_name,
                "profile": profile,
                "symbol": symbol,
                "cells": cells,
                "recent_cutoff_ts": recent_cutoff_ts,
                "top_k": int(args.top_k),
                "cv_protocol": str(args.cv_protocol),
                "train_min_days": int(args.train_min_days),
                "test_days": int(args.test_days),
                "step_days": int(args.step_days),
                "purge_hours": int(args.purge_hours),
                "progress_interval": int(args.progress_interval),
                "cluster_result_path": str(cluster_result_path) if cluster_result_path is not None else None,
            }
        )

    max_workers = max(1, int(args.max_workers or 1))
    outer_workers = min(max_workers, len(pending_tasks)) if pending_tasks else 1
    inner_workers = max(1, int(math.ceil(float(max_workers) / float(max(1, outer_workers)))))
    for task in pending_tasks:
        task["inner_workers"] = inner_workers
    if outer_workers > 1 and len(pending_tasks) > 1:
        with ProcessPoolExecutor(max_workers=outer_workers) as executor:
            future_map = {executor.submit(run_cluster_search_task, task): task for task in pending_tasks}
            for future in as_completed(future_map):
                task = future_map[future]
                cluster_name, result = future.result()
                clusters[cluster_name] = result
                cluster_result_path_raw = task.get("cluster_result_path")
                if cluster_result_path_raw:
                    _json_write_atomic(Path(str(cluster_result_path_raw)), result)
    else:
        for task in pending_tasks:
            cluster_name, result = run_cluster_search_task(task)
            clusters[cluster_name] = result
            cluster_result_path_raw = task.get("cluster_result_path")
            if cluster_result_path_raw:
                _json_write_atomic(Path(str(cluster_result_path_raw)), result)

    payload = {
        "generatedAt": datetime.now(tz=timezone.utc).isoformat(),
        "source": {
            "panel": str(args.panel),
            "clusters": sorted(selected_clusters),
            "assets": assets,
            "windowDays": int(args.window_days),
            "halfLifeDays": int(args.half_life_days),
            "recentDays": int(args.recent_days),
            "topK": int(args.top_k),
            "cvProtocol": str(args.cv_protocol),
            "trainMinDays": int(args.train_min_days),
            "testDays": int(args.test_days),
            "stepDays": int(args.step_days),
            "purgeHours": int(args.purge_hours),
            "maxTradesPerCell": int(args.max_trades_per_cell),
            "recentRetainRatio": float(args.recent_retain_ratio),
            "maxTsUtc": datetime.fromtimestamp(max_ts, tz=timezone.utc).isoformat(),
            "cutoffTsUtc": datetime.fromtimestamp(cutoff_ts, tz=timezone.utc).isoformat(),
            "recentCutoffTsUtc": datetime.fromtimestamp(recent_cutoff_ts, tz=timezone.utc).isoformat(),
        },
        "panelMeta": panel_meta,
        "clusters": clusters,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] report => {args.output}")
    for key, val in clusters.items():
        score = val["best"]["combined_score"]
        params = val["best"]["params"]
        print(f"  - {key}: combined_score={score:.4f} params={params}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
