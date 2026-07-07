#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PANEL = PROJECT_ROOT / "reports" / "combo_pause_v1_panel.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "combo_pause_v1_tuning_report.json"


@dataclass
class Event:
    ts: int
    pnl: float
    win: int
    base_capital: float


@dataclass
class Params:
    comboPauseDrawdown2h: float
    comboPauseHoldMinutes: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            "comboPauseDrawdown2h": float(self.comboPauseDrawdown2h),
            "comboPauseHoldMinutes": int(self.comboPauseHoldMinutes),
        }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_panel(path: Path, assets: Iterable[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"panel not found: {path}")
    df = pd.read_parquet(path)
    required = {"ts", "profile", "traderName", "symbol", "direction", "cluster_id", "cell_id", "pnl", "win", "base_capital"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"panel missing columns: {sorted(missing)}")
    out = df.copy()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["direction"] = out["direction"].astype(str).str.upper()
    out = out[out["direction"].isin(["UP", "DOWN"])]
    if assets:
        aset = {str(x).upper() for x in assets}
        out = out[out["symbol"].isin(aset)]
    for col in ("ts", "pnl", "win", "base_capital"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["ts", "pnl", "win", "base_capital"])
    if out.empty:
        raise RuntimeError("combo panel empty after cleanup")
    return out


def build_cluster_events(df: pd.DataFrame) -> Dict[str, List[Event]]:
    out: Dict[str, List[Event]] = {}
    for cluster_id, g in df.groupby("cluster_id", sort=False):
        events = [
            Event(
                ts=int(row.ts),
                pnl=float(row.pnl),
                win=int(row.win),
                base_capital=max(0.0, float(row.base_capital)),
            )
            for row in g.sort_values("ts").itertuples(index=False)
        ]
        if events:
            out[str(cluster_id)] = events
    return out


def make_folds(
    events: List[Event],
    *,
    window_days: int,
    train_min_days: int,
    test_days: int,
    step_days: int,
    purge_hours: int,
    recent_days: int,
) -> Tuple[List[Tuple[int, int, int, int]], str]:
    if not events:
        return [], "empty"
    min_ts = int(events[0].ts)
    max_ts = int(events[-1].ts)
    folds: List[Tuple[int, int, int, int]] = []
    step_sec = max(1, int(step_days)) * 86400
    test_sec = max(1, int(test_days)) * 86400
    purge_sec = max(0, int(purge_hours)) * 3600
    train_min_sec = max(1, int(train_min_days)) * 86400
    window_sec = max(int(window_days), int(train_min_days)) * 86400
    cursor = min_ts + train_min_sec
    while True:
        train_end = cursor
        test_start = train_end + purge_sec
        test_end = test_start + test_sec
        if test_end > max_ts + 1:
            break
        train_start = max(min_ts, train_end - window_sec)
        if train_end - train_start >= train_min_sec:
            folds.append((train_start, train_end, test_start, test_end))
        cursor += step_sec
    if folds:
        return folds, "walk_forward"

    test_end = max_ts + 1
    test_start = max(min_ts, max_ts - max(1, int(recent_days)) * 86400)
    train_end = max(min_ts, test_start - purge_sec)
    train_start = min_ts
    if test_end > test_start:
        return [(train_start, train_end, test_start, test_end)], "recent_fallback"
    return [], "empty"


def _max_drawdown(pnls: List[float]) -> float:
    eq = 0.0
    peak = 0.0
    mdd = 0.0
    for pnl in pnls:
        eq += float(pnl)
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    return float(mdd)


def _simulate_fold(events: List[Event], params: Params, train_start: int, train_end: int, test_start: int, test_end: int) -> Dict[str, float]:
    history: Deque[Tuple[int, float]] = deque()
    history_pnl = 0.0
    hold_sec = int(params.comboPauseHoldMinutes) * 60
    active_until = 0

    base_net = 0.0
    base_wins = 0
    base_input = 0
    base_pnls: List[float] = []

    cand_net = 0.0
    cand_wins = 0
    cand_input = 0
    cand_exec = 0
    cand_pnls: List[float] = []

    warmup_start = max(train_start, test_start - 2 * 3600)
    for ev in events:
        if ev.ts < warmup_start or ev.ts >= test_end:
            continue
        cutoff = ev.ts - 2 * 3600
        while history and history[0][0] < cutoff:
            _, old_pnl = history.popleft()
            history_pnl -= float(old_pnl)

        base_cap = max(1e-9, float(ev.base_capital) if math.isfinite(ev.base_capital) and ev.base_capital > 0 else 1.0)
        current_drawdown = max(0.0, -history_pnl / base_cap)
        in_test = test_start <= ev.ts < test_end

        if in_test:
            base_input += 1
            base_net += float(ev.pnl)
            base_pnls.append(float(ev.pnl))
            if int(ev.win) == 1:
                base_wins += 1

        if active_until > ev.ts or current_drawdown >= params.comboPauseDrawdown2h:
            if current_drawdown >= params.comboPauseDrawdown2h:
                active_until = max(active_until, ev.ts + hold_sec)
            if in_test:
                cand_input += 1
            continue

        history.append((ev.ts, float(ev.pnl)))
        history_pnl += float(ev.pnl)

        if in_test:
            cand_input += 1
            cand_exec += 1
            cand_net += float(ev.pnl)
            cand_pnls.append(float(ev.pnl))
            if int(ev.win) == 1:
                cand_wins += 1

        current_after = max(0.0, -history_pnl / base_cap)
        if current_after >= params.comboPauseDrawdown2h:
            active_until = max(active_until, ev.ts + hold_sec)

    return {
        "base_net_pnl": float(base_net),
        "cand_net_pnl": float(cand_net),
        "base_mdd": _max_drawdown(base_pnls),
        "cand_mdd": _max_drawdown(cand_pnls),
        "base_wr": float(base_wins / base_input) if base_input else 0.0,
        "cand_wr": float(cand_wins / cand_exec) if cand_exec else 0.0,
        "base_input": float(base_input),
        "cand_input": float(cand_input),
        "cand_executed": float(cand_exec),
    }


def evaluate_candidate(events: List[Event], params: Params, folds: List[Tuple[int, int, int, int]]) -> Dict[str, Any]:
    fold_rows: List[Dict[str, float]] = []
    for train_start, train_end, test_start, test_end in folds:
        fold_rows.append(_simulate_fold(events, params, train_start, train_end, test_start, test_end))
    if not fold_rows:
        return {
            "base_net_pnl": 0.0,
            "cand_net_pnl": 0.0,
            "base_mdd": 0.0,
            "cand_mdd": 0.0,
            "base_wr": 0.0,
            "cand_wr": 0.0,
            "wr_drop": 0.0,
            "trade_suppression": 0.0,
            "base_input": 0.0,
            "cand_input": 0.0,
            "cand_executed": 0.0,
            "fold_count": 0,
        }
    base_net = sum(x["base_net_pnl"] for x in fold_rows)
    cand_net = sum(x["cand_net_pnl"] for x in fold_rows)
    base_mdd = max((x["base_mdd"] for x in fold_rows), default=0.0)
    cand_mdd = max((x["cand_mdd"] for x in fold_rows), default=0.0)
    base_input = sum(x["base_input"] for x in fold_rows)
    cand_input = sum(x["cand_input"] for x in fold_rows)
    cand_executed = sum(x["cand_executed"] for x in fold_rows)
    base_wr_num = sum(x["base_wr"] * x["base_input"] for x in fold_rows)
    cand_wr_num = sum(x["cand_wr"] * x["cand_executed"] for x in fold_rows)
    base_wr = base_wr_num / base_input if base_input > 0 else 0.0
    cand_wr = cand_wr_num / cand_executed if cand_executed > 0 else 0.0
    wr_drop = max(0.0, base_wr - cand_wr)
    suppression = max(0.0, 1.0 - (cand_executed / base_input)) if base_input > 0 else 0.0
    return {
        "base_net_pnl": float(base_net),
        "cand_net_pnl": float(cand_net),
        "base_mdd": float(base_mdd),
        "cand_mdd": float(cand_mdd),
        "base_wr": float(base_wr),
        "cand_wr": float(cand_wr),
        "wr_drop": float(wr_drop),
        "trade_suppression": float(suppression),
        "base_input": float(base_input),
        "cand_input": float(cand_input),
        "cand_executed": float(cand_executed),
        "fold_count": int(len(fold_rows)),
        "fold_metrics": fold_rows,
    }


def admission_tier(metrics: Dict[str, Any]) -> Tuple[str, List[str]]:
    base_net = float(metrics.get("base_net_pnl", 0.0))
    cand_net = float(metrics.get("cand_net_pnl", 0.0))
    base_mdd = float(metrics.get("base_mdd", 0.0))
    cand_mdd = float(metrics.get("cand_mdd", 0.0))
    wr_drop = float(metrics.get("wr_drop", 0.0))
    suppression = float(metrics.get("trade_suppression", 0.0))
    mdd_improve = ((base_mdd - cand_mdd) / base_mdd) if base_mdd > 0 else 0.0
    violations: List[str] = []
    if cand_net < base_net:
        violations.append("net_pnl_lt_baseline")
    if mdd_improve < 0.10:
        violations.append("mdd_improve_lt_10pct")
    if wr_drop > 0.01:
        violations.append("wr_drop_gt_1pct")
    if suppression > 0.30:
        violations.append("trade_suppression_gt_30pct")
    if cand_net >= base_net and mdd_improve >= 0.10 and wr_drop <= 0.01 and suppression <= 0.30:
        return "A", violations
    if cand_net >= base_net and cand_mdd <= base_mdd and wr_drop <= 0.015 and suppression <= 0.40:
        return "B", violations
    return "C", violations


def candidate_score(metrics: Dict[str, Any], tier: str) -> float:
    tier_rank = {"A": 2.0, "B": 1.0, "C": 0.0}.get(tier, 0.0)
    base_net = float(metrics.get("base_net_pnl", 0.0))
    cand_net = float(metrics.get("cand_net_pnl", 0.0))
    base_mdd = float(metrics.get("base_mdd", 0.0))
    cand_mdd = float(metrics.get("cand_mdd", 0.0))
    suppression = float(metrics.get("trade_suppression", 0.0))
    wr_drop = float(metrics.get("wr_drop", 0.0))
    mdd_improve = (base_mdd - cand_mdd)
    return tier_rank * 1_000_000.0 + (cand_net - base_net) * 1_000.0 + mdd_improve * 10.0 - suppression * 100.0 - wr_drop * 100.0


def _cluster_dir(base: Optional[Path], cluster: str) -> Optional[Path]:
    if base is None:
        return None
    return base / cluster


def _write_cluster_result(base: Optional[Path], cluster: str, payload: Dict[str, Any]) -> None:
    d = _cluster_dir(base, cluster)
    if d is None:
        return
    d.mkdir(parents=True, exist_ok=True)
    (d / "cluster_result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Directional combo pause optimization from settled trades")
    ap.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--clusters", type=str, default="all")
    ap.add_argument("--assets", type=str, default="BTC,ETH")
    ap.add_argument("--window-days", type=int, default=365)
    ap.add_argument("--train-min-days", type=int, default=180)
    ap.add_argument("--test-days", type=int, default=14)
    ap.add_argument("--step-days", type=int, default=14)
    ap.add_argument("--purge-hours", type=int, default=24)
    ap.add_argument("--half-life-days", type=int, default=30, help="compat only; unused for combo pause")
    ap.add_argument("--recent-days", type=int, default=14)
    ap.add_argument("--drawdown-grid", type=str, default="0.03,0.04,0.05,0.06,0.08,0.10,0.12,0.15")
    ap.add_argument("--hold-grid", type=str, default="30,45,60,90,120,180")
    ap.add_argument("--checkpoint-dir", type=Path, default=None)
    return ap.parse_args()


def _parse_float_grid(raw: str) -> List[float]:
    vals: List[float] = []
    for part in str(raw).split(","):
        try:
            vals.append(float(part.strip()))
        except Exception:
            continue
    return sorted(set(v for v in vals if math.isfinite(v) and v > 0))


def _parse_int_grid(raw: str) -> List[int]:
    vals: List[int] = []
    for part in str(raw).split(","):
        try:
            vals.append(int(float(part.strip())))
        except Exception:
            continue
    return sorted(set(v for v in vals if v > 0))


def main() -> int:
    args = parse_args()
    assets = [x.strip().upper() for x in str(args.assets).split(",") if x.strip()]
    df = load_panel(args.panel, assets)
    clusters_all = build_cluster_events(df)
    wanted = {x.strip() for x in str(args.clusters).split(",") if x.strip() and x.strip().lower() != "all"}
    selected_clusters = {k: v for k, v in clusters_all.items() if (not wanted) or k in wanted}
    if not selected_clusters:
        raise SystemExit("combo selected clusters empty")

    drawdowns = _parse_float_grid(args.drawdown_grid)
    holds = _parse_int_grid(args.hold_grid)
    if not drawdowns or not holds:
        raise SystemExit("combo grid empty")

    output: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "panel": str(args.panel),
            "assets": assets,
            "window_days": int(args.window_days),
            "train_min_days": int(args.train_min_days),
            "test_days": int(args.test_days),
            "step_days": int(args.step_days),
            "purge_hours": int(args.purge_hours),
            "recent_days": int(args.recent_days),
        },
        "clusters": {},
        "summary": {"total_clusters": 0, "ready_clusters": 0},
    }

    for cluster_id, events in selected_clusters.items():
        folds, protocol_kind = make_folds(
            events,
            window_days=int(args.window_days),
            train_min_days=int(args.train_min_days),
            test_days=int(args.test_days),
            step_days=int(args.step_days),
            purge_hours=int(args.purge_hours),
            recent_days=int(args.recent_days),
        )
        best_payload: Optional[Dict[str, Any]] = None
        best_score = -1e18
        for dd, hold in itertools.product(drawdowns, holds):
            params = Params(comboPauseDrawdown2h=float(dd), comboPauseHoldMinutes=int(hold))
            metrics = evaluate_candidate(events, params, folds)
            tier, violations = admission_tier(metrics)
            score = candidate_score(metrics, tier)
            candidate_payload = {
                "params": params.as_dict(),
                "metrics_oos": metrics,
                "admission_tier": tier,
                "constraint_violations": violations,
                "final_is_feasible": tier in {"A", "B"},
                "delta_vs_baseline_oos": {
                    "net_pnl": float(metrics["cand_net_pnl"] - metrics["base_net_pnl"]),
                    "mdd": float(metrics["cand_mdd"] - metrics["base_mdd"]),
                },
                "score": score,
            }
            if score > best_score:
                best_score = score
                best_payload = candidate_payload
        if best_payload is None:
            continue
        cluster_payload = {
            "protocol": protocol_kind,
            "events": int(len(events)),
            "ts_min_utc": datetime.fromtimestamp(int(events[0].ts), tz=timezone.utc).isoformat(),
            "ts_max_utc": datetime.fromtimestamp(int(events[-1].ts), tz=timezone.utc).isoformat(),
            "fold_count": int(len(folds)),
            "best": best_payload,
        }
        output["clusters"][cluster_id] = cluster_payload
        output["summary"]["total_clusters"] += 1
        if bool(best_payload.get("final_is_feasible")):
            output["summary"]["ready_clusters"] += 1
        _write_cluster_result(args.checkpoint_dir, cluster_id, cluster_payload)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, args.output)
    print(args.output)
    print(json.dumps(output["summary"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
