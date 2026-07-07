#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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
DEFAULT_PANEL = PROJECT_ROOT / "reports" / "shock_risk_v2_panel.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "shock_risk_v2_tuning_report.json"


@dataclass
class TradeEvent:
    ts: int
    pnl: float
    win: int
    weight: float


@dataclass
class CellData:
    profile: str
    symbol: str
    direction: str
    cell_id: str
    events: List[TradeEvent]


@dataclass
class TriggerParams:
    window_minutes: int
    min_trades: int
    wr_post: float
    pnl_per_trade: float
    vol_percentile: int


@dataclass
class ActionParams:
    hold_hours: int
    release_checks: int
    release_wr: float
    release_pnl: float


@dataclass
class Params:
    trigger: TriggerParams
    action: ActionParams

    def as_dict(self) -> Dict[str, Any]:
        return {
            "shockRiskWindowMinutes": int(self.trigger.window_minutes),
            "shockRiskMinTrades": int(self.trigger.min_trades),
            "shockRiskWrPost": float(self.trigger.wr_post),
            "shockRiskPnlPerTrade": float(self.trigger.pnl_per_trade),
            "shockRiskVolPercentile": int(self.trigger.vol_percentile),
            "shockRiskHoldHours": int(self.action.hold_hours),
            "shockRiskReleaseChecks": int(self.action.release_checks),
            "shockRiskReleaseWr": float(self.action.release_wr),
            "shockRiskReleasePnl": float(self.action.release_pnl),
        }


def params_from_dict(d: Dict[str, Any]) -> Params:
    return Params(
        trigger=TriggerParams(
            window_minutes=int(d["shockRiskWindowMinutes"]),
            min_trades=int(d["shockRiskMinTrades"]),
            wr_post=float(d["shockRiskWrPost"]),
            pnl_per_trade=float(d["shockRiskPnlPerTrade"]),
            vol_percentile=int(d["shockRiskVolPercentile"]),
        ),
        action=ActionParams(
            hold_hours=int(d["shockRiskHoldHours"]),
            release_checks=int(d["shockRiskReleaseChecks"]),
            release_wr=float(d["shockRiskReleaseWr"]),
            release_pnl=float(d["shockRiskReleasePnl"]),
        ),
    )


def _json_write_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _json_read(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _cluster_dir(base: Optional[Path], cluster_key: str) -> Optional[Path]:
    if base is None:
        return None
    return base / cluster_key


def _cluster_result_path(base: Optional[Path], cluster_key: str) -> Optional[Path]:
    d = _cluster_dir(base, cluster_key)
    return None if d is None else d / "cluster_result.json"


def _pass_phase_path(base: Optional[Path], cluster_key: str, pass_label: str, phase: str) -> Optional[Path]:
    d = _cluster_dir(base, cluster_key)
    return None if d is None else d / f"{pass_label}_{phase}.json"


def _trigger_key(t: Tuple[int, int, float, float, int]) -> str:
    window_m, min_trades, wr, pnl, vol_pctl = t
    return f"{int(window_m)}|{int(min_trades)}|{float(wr):.6f}|{float(pnl):.6f}|{int(vol_pctl)}"


def _action_key(trigger: TriggerParams, a: Tuple[int, int, float, float]) -> str:
    hold_h, rel_checks, rel_wr, rel_pnl = a
    return (
        f"{int(trigger.window_minutes)}|{int(trigger.min_trades)}|{float(trigger.wr_post):.6f}|"
        f"{float(trigger.pnl_per_trade):.6f}|{int(trigger.vol_percentile)}|"
        f"{int(hold_h)}|{int(rel_checks)}|{float(rel_wr):.6f}|{float(rel_pnl):.6f}"
    )


def _candidate_from_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(item)
    out["params"] = params_from_dict(item["params"])
    return out


def _candidate_to_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(item)
    params = out.get("params")
    if isinstance(params, Params):
        out["params"] = params.as_dict()
    return out


def _is_action_ckpt_compatible(payload: Optional[Dict[str, Any]]) -> bool:
    if not payload or not bool(payload.get("complete")):
        return False
    best = payload.get("best") or {}
    if not isinstance(best, dict):
        return False
    # new coverage fields introduced in v1.3+
    if "trigger_fold_coverage_recent" not in best:
        return False
    snap = best.get("constraints_snapshot") or {}
    if not isinstance(snap, dict):
        return False
    return "min_trigger_fold_coverage_recent" in snap and "trigger_coverage_recent_folds" in snap


def _is_cluster_result_compatible(payload: Optional[Dict[str, Any]]) -> bool:
    if not payload or not isinstance(payload, dict):
        return False
    best = (payload.get("best") or {})
    if not isinstance(best, dict):
        return False
    if "trigger_fold_coverage_recent" not in best:
        return False
    snap = best.get("constraints_snapshot") or {}
    if not isinstance(snap, dict):
        return False
    return "min_trigger_fold_coverage_recent" in snap and "trigger_coverage_recent_folds" in snap


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SHOCK 风控 V2：无泄漏 walk-forward 超参")
    ap.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--window-days", type=int, default=365)
    ap.add_argument("--half-life-days", type=int, default=30)
    ap.add_argument("--assets", type=str, default="BTC,ETH")
    ap.add_argument("--recent-days", type=int, default=21)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--max-trades-per-cell", type=int, default=1500)
    ap.add_argument("--progress-interval", type=int, default=300)
    ap.add_argument("--cv-protocol", type=str, default="walk_forward", choices=["walk_forward", "none"])
    ap.add_argument("--train-min-days", type=int, default=180)
    ap.add_argument("--test-days", type=int, default=14)
    ap.add_argument("--step-days", type=int, default=14)
    ap.add_argument("--purge-hours", type=int, default=24)
    ap.add_argument("--min-trigger-count", type=float, default=5.0)
    ap.add_argument("--min-trigger-rate", type=float, default=0.0001)
    ap.add_argument("--min-trigger-fold-coverage", type=float, default=0.70)
    ap.add_argument("--min-trigger-fold-coverage-recent", type=float, default=0.70)
    ap.add_argument("--trigger-coverage-recent-folds", type=int, default=5)
    ap.add_argument("--net-drop-max", type=float, default=0.08)
    ap.add_argument("--mdd-improve-min", type=float, default=0.10)
    ap.add_argument("--suppression-max", type=float, default=0.40)
    ap.add_argument("--wr-drop-max", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--checkpoint-dir", type=Path, default=None)
    ap.add_argument("--clusters", type=str, default="all")
    return ap.parse_args()


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def load_panel(path: Path, assets: List[str]) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"panel not found: {path}")
    df = pd.read_parquet(path)
    required = {"ts", "profile", "traderName", "symbol", "direction", "cell_id", "win", "pnl"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"panel missing columns: {sorted(missing)}")
    df = df.copy()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["direction"] = df["direction"].astype(str).str.upper()
    df = df[df["symbol"].isin(assets)]
    df = df[df["direction"].isin(["UP", "DOWN"])]
    if df.empty:
        raise RuntimeError("panel empty after filters")

    for col in ("ts", "win", "pnl"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["ts", "win", "pnl"])
    if df.empty:
        raise RuntimeError("panel empty after numeric cleanup")

    df["profile"] = df["profile"].astype(str)
    df["traderName"] = df["traderName"].astype(str)
    df["cell_id"] = df["cell_id"].astype(str)
    return df


def downsample_events(events: List[TradeEvent], limit: int) -> Tuple[List[TradeEvent], int, int]:
    total = len(events)
    if limit <= 0 or total <= limit:
        return events, total, total
    sampled = events[-limit:]
    return sampled, total, len(sampled)


def build_cells(
    df: pd.DataFrame,
    cutoff_ts: int,
    half_life_days: int,
    max_trades_per_cell: int,
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
    raw_cnt: Dict[Tuple[str, str], int] = {k: 0 for k in buckets}
    sampled_cnt: Dict[Tuple[str, str], int] = {k: 0 for k in buckets}

    for (profile, symbol, direction, cell_id), g in x.groupby(
        ["profile", "symbol", "direction", "cell_id"], sort=False
    ):
        key = (profile, symbol)
        if key not in buckets:
            continue
        g = g.sort_values("ts")
        events_raw = [
            TradeEvent(
                ts=int(r.ts),
                pnl=float(r.pnl),
                win=int(r.win),
                weight=float(r.weight),
            )
            for r in g.itertuples(index=False)
        ]
        if not events_raw:
            continue
        events, before_n, after_n = downsample_events(events_raw, max_trades_per_cell)
        raw_cnt[key] += int(before_n)
        sampled_cnt[key] += int(after_n)
        buckets[key].append(
            CellData(
                profile=str(profile),
                symbol=str(symbol),
                direction=str(direction),
                cell_id=str(cell_id),
                events=events,
            )
        )

    meta = {
        "rows_window": int(len(x)),
        "max_ts": max_ts,
        "min_ts": int(x["ts"].min()),
        "cells": {f"{k[0]}_{k[1]}": len(v) for k, v in buckets.items()},
        "trades_before_sampling": {f"{k[0]}_{k[1]}": int(v) for k, v in raw_cnt.items()},
        "trades_after_sampling": {f"{k[0]}_{k[1]}": int(v) for k, v in sampled_cnt.items()},
        "sampling": {"maxTradesPerCell": int(max_trades_per_cell)},
    }
    return buckets, meta


def wr_post(wins: int, n: int) -> float:
    return (wins + 2.0) / (n + 4.0)


def compute_mdd(equity: List[float]) -> float:
    if not equity:
        return 0.0
    peak = equity[0]
    worst = 0.0
    for x in equity:
        peak = max(peak, x)
        dd = peak - x
        if dd > worst:
            worst = dd
    return float(worst)


def build_folds(
    min_ts: int,
    max_ts: int,
    protocol: str,
    train_min_days: int,
    test_days: int,
    step_days: int,
) -> List[Tuple[int, int]]:
    if protocol == "none":
        return [(min_ts, max_ts + 1)]
    train_min_sec = max(1, train_min_days) * 86400
    test_sec = max(1, test_days) * 86400
    step_sec = max(1, step_days) * 86400

    folds: List[Tuple[int, int]] = []
    test_start = min_ts + train_min_sec
    while test_start + test_sec <= max_ts:
        folds.append((test_start, test_start + test_sec))
        test_start += step_sec
    if not folds:
        start = max(min_ts, max_ts - test_sec)
        folds.append((start, max_ts + 1))
    return folds


def evaluate_cell_on_fold(
    events: List[TradeEvent],
    params: Params,
    test_start: int,
    test_end: int,
    purge_hours: int,
) -> Dict[str, float]:
    if not events:
        return {
            "net": 0.0,
            "baseline_net": 0.0,
            "mdd": 0.0,
            "baseline_mdd": 0.0,
            "suppression": 0.0,
            "active_rate": 0.0,
            "false_trigger_rate": 0.0,
            "trades": 0.0,
            "triggers": 0.0,
        }
    events = sorted(events, key=lambda e: e.ts)
    purge_sec = max(0, int(purge_hours) * 3600)
    seed_end = test_start - purge_sec

    history: List[TradeEvent] = []
    for e in events:
        if e.ts < seed_end:
            history.append(e)
        else:
            break

    shock_active_until = 0
    release_streak = 0
    suppression = 0
    trigger_count = 0
    false_trigger_count = 0
    tested = 0
    baseline_wins = 0
    baseline_trades = 0
    cand_wins = 0
    cand_trades = 0
    baseline_equity = [0.0]
    equity = [0.0]

    for i, e in enumerate(events):
        if e.ts < test_start:
            if e.ts >= seed_end:
                # purge 区间不参与历史
                continue
            continue
        if e.ts >= test_end:
            break

        cutoff_main = e.ts - int(params.trigger.window_minutes) * 60
        cutoff_1h = e.ts - 3600
        cutoff_4h = e.ts - 4 * 3600
        cutoff_24h = e.ts - 24 * 3600

        hist_main = [h for h in history if h.ts >= cutoff_main]
        hist_1h = [h for h in history if h.ts >= cutoff_1h]
        hist_4h = [h for h in history if h.ts >= cutoff_4h]
        hist_24h = [h for h in history if h.ts >= cutoff_24h]

        n_main = len(hist_main)
        wins_main = int(sum(1 for h in hist_main if h.win == 1))
        pnl_main = float(sum(h.pnl for h in hist_main))
        pnl_per_trade_main = pnl_main / n_main if n_main > 0 else 0.0
        wr_main = wr_post(wins_main, n_main)
        pnl_1h = float(sum(h.pnl for h in hist_1h)) if hist_1h else 0.0
        pnl_4h = float(sum(h.pnl for h in hist_4h)) if hist_4h else 0.0
        avg_abs_1h = float(sum(abs(h.pnl) for h in hist_1h) / len(hist_1h)) if hist_1h else 0.0
        avg_abs_4h = float(sum(abs(h.pnl) for h in hist_4h) / len(hist_4h)) if hist_4h else 0.0
        avg_abs_24h = float(sum(abs(h.pnl) for h in hist_24h) / len(hist_24h)) if hist_24h else 0.0
        abs_vol = sorted(abs(h.pnl) for h in hist_24h)
        if abs_vol:
            q_idx = int((len(abs_vol) - 1) * (_clamp(params.trigger.vol_percentile / 100.0, 0.0, 1.0)))
            vol_pctl = abs_vol[max(0, min(len(abs_vol) - 1, q_idx))]
        else:
            vol_pctl = 0.0

        vol_ratio = avg_abs_1h / max(avg_abs_24h, 1e-9) if avg_abs_24h > 0 else 0.0
        vol_ratio_threshold = 1.0 + max(0.0, (float(params.trigger.vol_percentile) - 50.0) / 100.0)
        vol_ratio_gate = avg_abs_24h > 0 and vol_ratio >= vol_ratio_threshold
        vol_quantile_gate = (avg_abs_1h >= vol_pctl) or (avg_abs_4h >= vol_pctl)
        vol_gate = vol_ratio_gate or vol_quantile_gate
        loss_gate = pnl_1h < 0 or pnl_4h < 0
        severe_pnl = min(
            float(params.trigger.pnl_per_trade) * 1.8,
            float(params.trigger.pnl_per_trade) - 0.01,
        )
        normal_trigger = (
            n_main >= params.trigger.min_trades
            and wr_main < params.trigger.wr_post
            and pnl_per_trade_main < params.trigger.pnl_per_trade
            and vol_gate
            and loss_gate
        )
        severe_trigger = (
            n_main >= max(3, params.trigger.min_trades // 2)
            and pnl_per_trade_main < severe_pnl
            and vol_quantile_gate
            and loss_gate
        )
        triggered = normal_trigger or severe_trigger

        if shock_active_until <= e.ts and triggered:
            shock_active_until = e.ts + params.action.hold_hours * 3600
            release_streak = 0
            trigger_count += 1
            next_slice = [x.pnl for x in events[i + 1:i + 5]]
            if next_slice and sum(next_slice) >= 0:
                false_trigger_count += 1

        if shock_active_until <= e.ts and shock_active_until > 0:
            release_pass = wr_main >= params.action.release_wr and pnl_per_trade_main >= params.action.release_pnl
            if release_pass:
                release_streak += 1
                if release_streak >= params.action.release_checks:
                    shock_active_until = 0
                    release_streak = 0
            else:
                release_streak = 0
                shock_active_until = e.ts + params.action.hold_hours * 3600

        active = shock_active_until > e.ts
        tested += 1
        baseline_trades += 1
        baseline_wins += int(e.win == 1)
        baseline_next = baseline_equity[-1] + e.weight * e.pnl
        baseline_equity.append(baseline_next)
        if active:
            suppression += 1
            equity.append(equity[-1])
        else:
            cand_trades += 1
            cand_wins += int(e.win == 1)
            equity.append(equity[-1] + e.weight * e.pnl)

        history.append(e)

    baseline_net = baseline_equity[-1]
    net = equity[-1]
    out = {
        "net": float(net),
        "baseline_net": float(baseline_net),
        "mdd": compute_mdd(equity),
        "baseline_mdd": compute_mdd(baseline_equity),
        "suppression": float(suppression / tested) if tested > 0 else 0.0,
        "active_rate": float(sum(1 for _ in equity if False)),  # placeholder overwritten below
        "false_trigger_rate": float(false_trigger_count / trigger_count) if trigger_count > 0 else 0.0,
        "trades": float(tested),
        "triggers": float(trigger_count),
        "baseline_wins": float(baseline_wins),
        "baseline_trades": float(baseline_trades),
        "cand_wins": float(cand_wins),
        "cand_trades": float(cand_trades),
    }
    # active_rate 用 suppression 近似（被阻断比例）
    out["active_rate"] = out["suppression"]
    return out


def aggregate_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {
            "net_pnl": 0.0,
            "baseline_net_pnl": 0.0,
            "mdd": 0.0,
            "baseline_mdd": 0.0,
            "suppression_rate": 0.0,
            "false_trigger_rate": 0.0,
            "trade_count": 0.0,
            "trigger_count": 0.0,
            "baseline_wins": 0.0,
            "baseline_trades": 0.0,
            "cand_wins": 0.0,
            "cand_trades": 0.0,
        }
    n = float(len(rows))
    baseline_wins = float(sum(r.get("baseline_wins", 0.0) for r in rows))
    baseline_trades = float(sum(r.get("baseline_trades", 0.0) for r in rows))
    cand_wins = float(sum(r.get("cand_wins", 0.0) for r in rows))
    cand_trades = float(sum(r.get("cand_trades", 0.0) for r in rows))
    return {
        "net_pnl": float(sum(r["net"] for r in rows)),
        "baseline_net_pnl": float(sum(r["baseline_net"] for r in rows)),
        "mdd": float(sum(r["mdd"] for r in rows) / n),
        "baseline_mdd": float(sum(r["baseline_mdd"] for r in rows) / n),
        "suppression_rate": float(sum(r["suppression"] for r in rows) / n),
        "false_trigger_rate": float(sum(r["false_trigger_rate"] for r in rows) / n),
        "trade_count": float(sum(r["trades"] for r in rows)),
        "trigger_count": float(sum(r["triggers"] for r in rows)),
        "baseline_wins": baseline_wins,
        "baseline_trades": baseline_trades,
        "cand_wins": cand_wins,
        "cand_trades": cand_trades,
        "baseline_wr": baseline_wins / baseline_trades if baseline_trades > 0 else 0.5,
        "cand_wr": cand_wins / cand_trades if cand_trades > 0 else 0.5,
    }


def score_metrics(m: Dict[str, float]) -> float:
    return (
        0.40 * float(m["net_pnl"])
        - 0.40 * float(m["mdd"])
        - 0.10 * float(m["suppression_rate"])
        - 0.10 * float(m["false_trigger_rate"])
    )


def evaluate_params(
    cells: List[CellData],
    params: Params,
    folds: List[Tuple[int, int]],
    purge_hours: int,
    min_trigger_count: float,
    min_trigger_rate: float,
    min_trigger_fold_coverage: float,
    min_trigger_fold_coverage_recent: float,
    trigger_coverage_recent_folds: int,
    net_drop_max: float,
    mdd_improve_min: float,
    suppression_max: float,
    wr_drop_max: float,
) -> Dict[str, Any]:
    per_fold: List[Dict[str, float]] = []
    per_fold_trigger_counts: List[float] = []
    for test_start, test_end in folds:
        fold_rows: List[Dict[str, float]] = []
        for cell in cells:
            row = evaluate_cell_on_fold(cell.events, params, test_start, test_end, purge_hours)
            fold_rows.append(row)
        fold_metrics = aggregate_metrics(fold_rows)
        per_fold.append(fold_metrics)
        per_fold_trigger_counts.append(float(fold_metrics.get("trigger_count", 0.0)))

    if per_fold:
        agg = {
            "net_pnl": float(sum(x["net_pnl"] for x in per_fold) / len(per_fold)),
            "baseline_net_pnl": float(sum(x["baseline_net_pnl"] for x in per_fold) / len(per_fold)),
            "mdd": float(sum(x["mdd"] for x in per_fold) / len(per_fold)),
            "baseline_mdd": float(sum(x["baseline_mdd"] for x in per_fold) / len(per_fold)),
            "suppression_rate": float(sum(x["suppression_rate"] for x in per_fold) / len(per_fold)),
            "false_trigger_rate": float(sum(x["false_trigger_rate"] for x in per_fold) / len(per_fold)),
            "trade_count": float(sum(x["trade_count"] for x in per_fold)),
            "trigger_count": float(sum(x["trigger_count"] for x in per_fold)),
            "baseline_wins": float(sum(x.get("baseline_wins", 0.0) for x in per_fold)),
            "baseline_trades": float(sum(x.get("baseline_trades", 0.0) for x in per_fold)),
            "cand_wins": float(sum(x.get("cand_wins", 0.0) for x in per_fold)),
            "cand_trades": float(sum(x.get("cand_trades", 0.0) for x in per_fold)),
        }
        agg["baseline_wr"] = agg["baseline_wins"] / agg["baseline_trades"] if agg["baseline_trades"] > 0 else 0.5
        agg["cand_wr"] = agg["cand_wins"] / agg["cand_trades"] if agg["cand_trades"] > 0 else 0.5
    else:
        agg = aggregate_metrics([])

    score = score_metrics(agg)
    baseline = float(agg["baseline_net_pnl"])
    net = float(agg["net_pnl"])
    if baseline > 0:
        net_drop = max(0.0, (baseline - net) / baseline)
    else:
        net_drop = 0.0 if net >= baseline else 1.0
    baseline_mdd = float(agg["baseline_mdd"])
    mdd = float(agg["mdd"])
    mdd_improve = ((baseline_mdd - mdd) / baseline_mdd) if baseline_mdd > 0 else 0.0
    suppression = float(agg["suppression_rate"])
    trigger_count = float(agg["trigger_count"])
    trade_count = max(1.0, float(agg["trade_count"]))
    trigger_rate = trigger_count / trade_count
    fold_count = int(len(per_fold_trigger_counts))
    nonzero_fold_count = int(sum(1 for x in per_fold_trigger_counts if float(x) > 0.0))
    trigger_fold_coverage = (
        float(nonzero_fold_count / fold_count) if fold_count > 0 else 0.0
    )
    recent_n = int(max(1, trigger_coverage_recent_folds))
    recent_counts = per_fold_trigger_counts[-recent_n:] if per_fold_trigger_counts else []
    recent_fold_count = int(len(recent_counts))
    recent_nonzero_fold_count = int(sum(1 for x in recent_counts if float(x) > 0.0))
    trigger_fold_coverage_recent = (
        float(recent_nonzero_fold_count / recent_fold_count) if recent_fold_count > 0 else 0.0
    )
    baseline_wr = float(agg.get("baseline_wr", 0.5))
    cand_wr = float(agg.get("cand_wr", 0.5))
    wr_drop = max(0.0, baseline_wr - cand_wr)

    violations: List[str] = []
    if net_drop > float(net_drop_max):
        violations.append("net_drop_gt_max")
    if mdd_improve < float(mdd_improve_min):
        violations.append("mdd_improve_lt_min")
    if suppression > float(suppression_max):
        violations.append("suppression_gt_max")
    if wr_drop > float(wr_drop_max):
        violations.append("wr_drop_gt_max")
    if trigger_count < float(min_trigger_count):
        violations.append("trigger_count_lt_min")
    if trigger_rate < float(min_trigger_rate):
        violations.append("trigger_rate_lt_min")
    global_cov_ok = trigger_fold_coverage >= float(min_trigger_fold_coverage)
    recent_cov_ok = trigger_fold_coverage_recent >= float(min_trigger_fold_coverage_recent)
    if not (global_cov_ok or recent_cov_ok):
        violations.append("trigger_fold_coverage_lt_min")

    return {
        "score": float(score),
        "metrics": agg,
        "constraint_violations": violations,
        "is_feasible": len(violations) == 0,
        "fold_trigger_counts": per_fold_trigger_counts,
        "nonzero_fold_count": nonzero_fold_count,
        "fold_count": fold_count,
        "trigger_fold_coverage": trigger_fold_coverage,
        "recent_fold_count": recent_fold_count,
        "recent_nonzero_fold_count": recent_nonzero_fold_count,
        "trigger_fold_coverage_recent": trigger_fold_coverage_recent,
        "constraints_snapshot": {
            "net_drop_max": float(net_drop_max),
            "mdd_improve_min": float(mdd_improve_min),
            "suppression_max": float(suppression_max),
            "wr_drop_max": float(wr_drop_max),
            "min_trigger_count": float(min_trigger_count),
            "min_trigger_rate": float(min_trigger_rate),
            "min_trigger_fold_coverage": float(min_trigger_fold_coverage),
            "min_trigger_fold_coverage_recent": float(min_trigger_fold_coverage_recent),
            "trigger_coverage_recent_folds": int(trigger_coverage_recent_folds),
        },
        "wr_drop": wr_drop,
    }


def optimize_cluster(
    cluster_key: str,
    cells: List[CellData],
    folds: List[Tuple[int, int]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if not cells:
        return {
            "cells": 0,
            "folds": len(folds),
            "best": None,
            "candidates": [],
        }

    def run_search_pass(
        trigger_grid: List[Tuple[int, int, float, float, int]],
        action_grid: List[Tuple[int, int, float, float]],
        pass_label: str,
    ) -> Dict[str, Any]:
        trigger_ckpt = _pass_phase_path(args.checkpoint_dir, cluster_key, pass_label, "trigger")
        action_ckpt = _pass_phase_path(args.checkpoint_dir, cluster_key, pass_label, "action")
        if action_ckpt is not None and action_ckpt.exists():
            cached = _json_read(action_ckpt)
            if _is_action_ckpt_compatible(cached):
                return {
                    "best": _candidate_from_payload(cached["best"]),
                    "candidates": [_candidate_from_payload(x) for x in cached.get("candidates", [])],
                    "trigger_grid_size": int(cached.get("trigger_grid_size", len(trigger_grid))),
                    "action_grid_size": int(cached.get("action_grid_size", len(action_grid))),
                    "evaluated": int(cached.get("evaluated", 0)),
                    "feasible_count": int(cached.get("feasible_count", 0)),
                    "search_pass": str(cached.get("search_pass", pass_label)),
                }
            # stale/incompatible checkpoint: force recompute under current scoring/constraints
            try:
                action_ckpt.unlink(missing_ok=True)
            except Exception:
                pass

        default_action = ActionParams(hold_hours=12, release_checks=3, release_wr=0.45, release_pnl=-0.05)
        trigger_candidates: List[Dict[str, Any]] = []
        trigger_done: set[str] = set()
        trigger_cached = _json_read(trigger_ckpt) if (trigger_ckpt is not None and trigger_ckpt.exists()) else None
        if trigger_cached:
            trigger_candidates = [_candidate_from_payload(x) for x in trigger_cached.get("candidates", [])]
            trigger_done = {str(x) for x in trigger_cached.get("done_keys", [])}

        for i, trig in enumerate(trigger_grid, start=1):
            trig_key = _trigger_key(trig)
            if trig_key not in trigger_done:
                window_m, min_trades, wr, pnl, vol_pctl = trig
                params = Params(
                    trigger=TriggerParams(
                        window_minutes=int(window_m),
                        min_trades=int(min_trades),
                        wr_post=float(wr),
                        pnl_per_trade=float(pnl),
                        vol_percentile=int(vol_pctl),
                    ),
                    action=default_action,
                )
                ev = evaluate_params(
                    cells,
                    params,
                    folds,
                    args.purge_hours,
                    args.min_trigger_count,
                    args.min_trigger_rate,
                    args.min_trigger_fold_coverage,
                    args.min_trigger_fold_coverage_recent,
                    args.trigger_coverage_recent_folds,
                    args.net_drop_max,
                    args.mdd_improve_min,
                    args.suppression_max,
                    args.wr_drop_max,
                )
                trigger_candidates.append(
                    {
                        "params": params,
                        "score": float(ev["score"]),
                        "is_feasible": bool(ev["is_feasible"]),
                        "constraint_violations": list(ev["constraint_violations"]),
                        "fold_trigger_counts": list(ev["fold_trigger_counts"]),
                        "nonzero_fold_count": int(ev["nonzero_fold_count"]),
                        "fold_count": int(ev["fold_count"]),
                        "trigger_fold_coverage": float(ev["trigger_fold_coverage"]),
                        "recent_fold_count": int(ev["recent_fold_count"]),
                        "recent_nonzero_fold_count": int(ev["recent_nonzero_fold_count"]),
                        "trigger_fold_coverage_recent": float(ev["trigger_fold_coverage_recent"]),
                        "constraints_snapshot": dict(ev["constraints_snapshot"]),
                    }
                )
                trigger_done.add(trig_key)
            if args.progress_interval > 0 and (i % args.progress_interval == 0 or i == len(trigger_grid)):
                if trigger_ckpt is not None:
                    _json_write_atomic(
                        trigger_ckpt,
                        {
                            "complete": False,
                            "search_pass": pass_label,
                            "done_keys": sorted(trigger_done),
                            "candidates": [_candidate_to_payload(x) for x in trigger_candidates],
                        },
                    )
                print(f"[cluster {cluster_key}] {pass_label} trigger_phase {i}/{len(trigger_grid)}", flush=True)

        trigger_candidates.sort(key=lambda x: (0 if x["is_feasible"] else 1, -x["score"]))
        top_triggers = trigger_candidates[: max(1, int(args.top_k))]
        if trigger_ckpt is not None:
            _json_write_atomic(
                trigger_ckpt,
                {
                    "complete": True,
                    "search_pass": pass_label,
                    "done_keys": sorted(trigger_done),
                    "candidates": [_candidate_to_payload(x) for x in trigger_candidates],
                    "top_triggers": [_candidate_to_payload(x) for x in top_triggers],
                },
            )

        final_candidates: List[Dict[str, Any]] = []
        action_done: set[str] = set()
        action_cached = _json_read(action_ckpt) if (action_ckpt is not None and action_ckpt.exists()) else None
        if action_cached:
            final_candidates = [_candidate_from_payload(x) for x in action_cached.get("candidates", [])]
            action_done = {str(x) for x in action_cached.get("done_keys", [])}
        total_action = len(top_triggers) * len(action_grid)
        action_idx = 0
        for tr in top_triggers:
            for hold_h, rel_checks, rel_wr, rel_pnl in action_grid:
                action_idx += 1
                p = tr["params"]
                act_tuple = (hold_h, rel_checks, rel_wr, rel_pnl)
                act_key = _action_key(p.trigger, act_tuple)
                if act_key not in action_done:
                    params = Params(
                        trigger=p.trigger,
                        action=ActionParams(
                            hold_hours=int(hold_h),
                            release_checks=int(rel_checks),
                            release_wr=float(rel_wr),
                            release_pnl=float(rel_pnl),
                        ),
                    )
                    ev = evaluate_params(
                        cells,
                        params,
                        folds,
                        args.purge_hours,
                        args.min_trigger_count,
                        args.min_trigger_rate,
                        args.min_trigger_fold_coverage,
                        args.min_trigger_fold_coverage_recent,
                        args.trigger_coverage_recent_folds,
                        args.net_drop_max,
                        args.mdd_improve_min,
                        args.suppression_max,
                        args.wr_drop_max,
                    )
                    final_candidates.append(
                        {
                            "params": params,
                            "score": float(ev["score"]),
                            "is_feasible": bool(ev["is_feasible"]),
                            "constraint_violations": list(ev["constraint_violations"]),
                            "metrics": ev["metrics"],
                            "fold_trigger_counts": list(ev["fold_trigger_counts"]),
                            "nonzero_fold_count": int(ev["nonzero_fold_count"]),
                            "fold_count": int(ev["fold_count"]),
                            "trigger_fold_coverage": float(ev["trigger_fold_coverage"]),
                            "recent_fold_count": int(ev["recent_fold_count"]),
                            "recent_nonzero_fold_count": int(ev["recent_nonzero_fold_count"]),
                            "trigger_fold_coverage_recent": float(ev["trigger_fold_coverage_recent"]),
                            "constraints_snapshot": dict(ev["constraints_snapshot"]),
                        }
                    )
                    action_done.add(act_key)
                if args.progress_interval > 0 and (action_idx % args.progress_interval == 0 or action_idx == total_action):
                    if action_ckpt is not None:
                        snapshot = sorted(final_candidates, key=lambda x: (0 if x["is_feasible"] else 1, -x["score"]))
                        _json_write_atomic(
                            action_ckpt,
                            {
                                "complete": False,
                                "search_pass": pass_label,
                                "done_keys": sorted(action_done),
                                "trigger_grid_size": len(trigger_grid),
                                "action_grid_size": len(action_grid),
                                "evaluated": len(final_candidates),
                                "feasible_count": sum(1 for x in final_candidates if bool(x["is_feasible"])),
                                "candidates": [_candidate_to_payload(x) for x in snapshot[:50]],
                            },
                        )
                    print(f"[cluster {cluster_key}] {pass_label} action_phase {action_idx}/{total_action}", flush=True)

        final_candidates.sort(key=lambda x: (0 if x["is_feasible"] else 1, -x["score"]))
        best = final_candidates[0]
        result = {
            "best": best,
            "candidates": final_candidates[:10],
            "trigger_grid_size": len(trigger_grid),
            "action_grid_size": len(action_grid),
            "evaluated": len(final_candidates),
            "feasible_count": sum(1 for x in final_candidates if bool(x["is_feasible"])),
            "search_pass": pass_label,
        }
        if action_ckpt is not None:
            _json_write_atomic(
                action_ckpt,
                {
                    "complete": True,
                    "search_pass": pass_label,
                    "done_keys": sorted(action_done),
                    "trigger_grid_size": len(trigger_grid),
                    "action_grid_size": len(action_grid),
                    "evaluated": len(final_candidates),
                    "feasible_count": sum(1 for x in final_candidates if bool(x["is_feasible"])),
                    "best": _candidate_to_payload(best),
                    "candidates": [_candidate_to_payload(x) for x in final_candidates[:10]],
                },
            )
        return result

    pass1_trigger_grid = list(
        itertools.product(
            [60, 75, 90, 120],
            [12, 16, 20, 25],
            [0.30, 0.35, 0.40, 0.45],
            [-0.02, -0.04, -0.06, -0.08],
            [60, 70, 80, 85],
        )
    )
    pass1_action_grid = list(
        itertools.product(
            [8, 12, 16],
            [2, 3, 4],
            [0.43, 0.45, 0.47],
            [-0.08, -0.05, -0.03, 0.0],
        )
    )
    pass1 = run_search_pass(pass1_trigger_grid, pass1_action_grid, "pass1")
    selected = pass1
    pass2: Optional[Dict[str, Any]] = None
    pass3: Optional[Dict[str, Any]] = None
    if not bool(pass1["best"]["is_feasible"]):
        pass2_trigger_grid = list(
            itertools.product(
                [45, 60, 90, 120],
                [8, 12, 16, 20],
                [0.35, 0.40, 0.45, 0.50],
                [-0.01, -0.02, -0.04, -0.06],
                [55, 60, 70, 80],
            )
        )
        pass2_action_grid = list(
            itertools.product(
                [6, 8, 12],
                [2, 3, 4],
                [0.43, 0.45, 0.47],
                [-0.08, -0.05, -0.03, 0.0],
            )
        )
        pass2 = run_search_pass(pass2_trigger_grid, pass2_action_grid, "pass2")
        selected = min(
            [pass1, pass2],
            key=lambda x: (
                0 if bool(x["best"]["is_feasible"]) else 1,
                len(x["best"]["constraint_violations"]),
                -float(x["best"]["score"]),
            ),
        )
    if not bool(selected["best"]["is_feasible"]):
        viol = set(str(x) for x in selected["best"].get("constraint_violations", []))
        if {"trigger_count_lt_min", "trigger_rate_lt_min", "trigger_fold_coverage_lt_min"} & viol:
            pass3_trigger_grid = list(
                itertools.product(
                    [45, 60, 75],
                    [6, 8, 10, 12],
                    [0.35, 0.40, 0.45, 0.50],
                    [-0.01, -0.02, -0.03, -0.04, -0.06],
                    [55, 60, 65, 70, 75],
                )
            )
            pass3_action_grid = list(
                itertools.product(
                    [6, 8, 12],
                    [2, 3, 4],
                    [0.43, 0.45, 0.47],
                    [-0.08, -0.05, -0.03, 0.0],
                )
            )
            pass3 = run_search_pass(pass3_trigger_grid, pass3_action_grid, "pass3")
            selected = min(
                [selected, pass3],
                key=lambda x: (
                    0 if bool(x["best"]["is_feasible"]) else 1,
                    len(x["best"]["constraint_violations"]),
                    -float(x["best"]["score"]),
                ),
            )

    best = selected["best"]
    best_out = {
        "params": best["params"].as_dict(),
        "score": float(best["score"]),
        "is_feasible": bool(best["is_feasible"]),
        "constraint_violations": list(best["constraint_violations"]),
        "metrics": best["metrics"],
        "fold_trigger_counts": list(best.get("fold_trigger_counts", [])),
        "nonzero_fold_count": int(best.get("nonzero_fold_count", 0)),
        "fold_count": int(best.get("fold_count", 0)),
        "trigger_fold_coverage": float(best.get("trigger_fold_coverage", 0.0)),
        "recent_fold_count": int(best.get("recent_fold_count", 0)),
        "recent_nonzero_fold_count": int(best.get("recent_nonzero_fold_count", 0)),
        "trigger_fold_coverage_recent": float(best.get("trigger_fold_coverage_recent", 0.0)),
        "constraints_snapshot": dict(best.get("constraints_snapshot", {})),
        "search_pass": str(selected["search_pass"]),
        "needs_review": not bool(best["is_feasible"]),
        "final_is_feasible": bool(best["is_feasible"]),
    }
    cands_out = []
    for cand in selected["candidates"]:
        cands_out.append(
            {
                "params": cand["params"].as_dict(),
                "score": float(cand["score"]),
                "is_feasible": bool(cand["is_feasible"]),
                "constraint_violations": list(cand["constraint_violations"]),
                "metrics": cand["metrics"],
                "fold_trigger_counts": list(cand.get("fold_trigger_counts", [])),
                "nonzero_fold_count": int(cand.get("nonzero_fold_count", 0)),
                "fold_count": int(cand.get("fold_count", 0)),
                "trigger_fold_coverage": float(cand.get("trigger_fold_coverage", 0.0)),
                "recent_fold_count": int(cand.get("recent_fold_count", 0)),
                "recent_nonzero_fold_count": int(cand.get("recent_nonzero_fold_count", 0)),
                "trigger_fold_coverage_recent": float(cand.get("trigger_fold_coverage_recent", 0.0)),
                "constraints_snapshot": dict(cand.get("constraints_snapshot", {})),
            }
        )
    result = {
        "cells": len(cells),
        "folds": len(folds),
        "best": best_out,
        "candidates": cands_out,
        "search": {
            "trigger_grid_size": int(selected["trigger_grid_size"]),
            "action_grid_size": int(selected["action_grid_size"]),
            "evaluated_action_candidates": int(selected["evaluated"]),
            "feasible_count": int(selected["feasible_count"]),
            "search_pass": str(selected["search_pass"]),
            "pass2_attempted": bool(pass2 is not None),
            "pass3_attempted": bool(pass3 is not None),
            "pass1_feasible_count": int(pass1["feasible_count"]),
            "pass2_feasible_count": int(pass2["feasible_count"]) if pass2 is not None else 0,
            "pass3_feasible_count": int(pass3["feasible_count"]) if pass3 is not None else 0,
        },
    }
    cluster_result_path = _cluster_result_path(args.checkpoint_dir, cluster_key)
    if cluster_result_path is not None:
        _json_write_atomic(cluster_result_path, result)
    return result


def run_cluster_task(
    cluster_key: str,
    cells: List[CellData],
    folds: List[Tuple[int, int]],
    args: argparse.Namespace,
) -> Tuple[str, Dict[str, Any]]:
    return cluster_key, optimize_cluster(cluster_key, cells, folds, args)


def main() -> int:
    args = parse_args()
    if args.checkpoint_dir is None:
        args.checkpoint_dir = args.output.parent / f"{args.output.stem}_checkpoint"
    assets = [x.strip().upper() for x in str(args.assets).split(",") if x.strip()]
    df = load_panel(args.panel, assets)

    max_ts = int(df["ts"].max())
    cutoff_ts = max_ts - int(args.window_days) * 86400
    buckets, build_meta = build_cells(
        df=df,
        cutoff_ts=cutoff_ts,
        half_life_days=int(args.half_life_days),
        max_trades_per_cell=int(args.max_trades_per_cell),
    )

    folds = build_folds(
        min_ts=int(build_meta["min_ts"]),
        max_ts=int(build_meta["max_ts"]),
        protocol=str(args.cv_protocol),
        train_min_days=int(args.train_min_days),
        test_days=int(args.test_days),
        step_days=int(args.step_days),
    )

    all_cluster_keys = ["default_BTC", "default_ETH", "70_BTC", "70_ETH"]
    requested = str(args.clusters or "all").strip()
    if requested.lower() == "all":
        selected_clusters = set(all_cluster_keys)
    else:
        selected_clusters = {
            x.strip()
            for x in requested.split(",")
            if x.strip() in set(all_cluster_keys)
        }
        if not selected_clusters:
            raise RuntimeError(f"--clusters invalid: {requested}")

    cluster_specs = []
    for profile, symbol in [("default", "BTC"), ("default", "ETH"), ("70", "BTC"), ("70", "ETH")]:
        key = f"{profile}_{symbol}"
        if key not in selected_clusters:
            continue
        cells = buckets.get((profile, symbol), [])
        print(f"[cluster {key}] cells={len(cells)} top_k={int(args.top_k)}", flush=True)
        cluster_specs.append((key, cells))

    clusters_out: Dict[str, Any] = {}
    pending_cluster_specs = []
    for key, cells in cluster_specs:
        cluster_result_path = _cluster_result_path(args.checkpoint_dir, key)
        cached = _json_read(cluster_result_path) if (cluster_result_path is not None and cluster_result_path.exists()) else None
        if _is_cluster_result_compatible(cached):
            clusters_out[key] = cached
            print(f"[resume] cluster_cached={key}", flush=True)
        else:
            if cluster_result_path is not None and cluster_result_path.exists():
                try:
                    cluster_result_path.unlink(missing_ok=True)
                except Exception:
                    pass
            pending_cluster_specs.append((key, cells))

    workers = max(1, min(int(args.workers), len(pending_cluster_specs) if pending_cluster_specs else 1))
    if pending_cluster_specs and workers == 1:
        for key, cells in pending_cluster_specs:
            clusters_out[key] = optimize_cluster(key, cells, folds, args)
    elif pending_cluster_specs:
        print(f"[parallel] workers={workers} clusters={len(pending_cluster_specs)}", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(run_cluster_task, key, cells, folds, args): key
                for key, cells in pending_cluster_specs
            }
            for future in as_completed(futures):
                key, result = future.result()
                clusters_out[key] = result
                print(f"[parallel] cluster_done={key}", flush=True)

    # Keep stable output ordering in the final report.
    clusters_out = {key: clusters_out[key] for key, _ in cluster_specs}

    summary = {
        "generatedAt": datetime.now(tz=timezone.utc).isoformat(),
        "panel": str(args.panel),
        "output": str(args.output),
        "options": {
            "window_days": int(args.window_days),
            "half_life_days": int(args.half_life_days),
            "recent_days": int(args.recent_days),
            "cv_protocol": str(args.cv_protocol),
            "train_min_days": int(args.train_min_days),
            "test_days": int(args.test_days),
            "step_days": int(args.step_days),
            "purge_hours": int(args.purge_hours),
            "assets": assets,
            "workers": int(args.workers),
            "constraints": {
                "net_drop_max": float(args.net_drop_max),
                "mdd_improve_min": float(args.mdd_improve_min),
                "suppression_max": float(args.suppression_max),
                "wr_drop_max": float(args.wr_drop_max),
                "min_trigger_count": float(args.min_trigger_count),
                "min_trigger_rate": float(args.min_trigger_rate),
                "min_trigger_fold_coverage": float(args.min_trigger_fold_coverage),
                "min_trigger_fold_coverage_recent": float(args.min_trigger_fold_coverage_recent),
                "trigger_coverage_recent_folds": int(args.trigger_coverage_recent_folds),
            },
        },
        "build_meta": build_meta,
        "folds": [{"test_start": int(a), "test_end": int(b)} for a, b in folds],
        "clusters": clusters_out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] report => {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
