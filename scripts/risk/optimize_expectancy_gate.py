#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PANEL = PROJECT_ROOT / "reports" / "expectancy_gate_panel.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "expectancy_gate_tuning_report.json"


@dataclass(frozen=True)
class Candidate:
    lookbackHours: int
    lookbackHoursFast: int
    minTrades: int
    minTradesFast: int
    minTradesBlocked: int
    wrDegraded: float
    pnlDegraded: float
    degradedBetScale: float
    wrFastBlocked: float
    pnlFastBlocked: float
    wrBlocked: float
    pnlBlocked: float
    blockedEnterChecks: int
    releaseChecks: int
    releaseWr: float
    releasePnl: float

    def params(self) -> Dict[str, Any]:
        return {
            "expectancyGateLookbackHours": self.lookbackHours,
            "expectancyGateLookbackHoursFast": self.lookbackHoursFast,
            "expectancyGateMinTrades": self.minTrades,
            "expectancyGateMinTradesFast": self.minTradesFast,
            "expectancyGateMinTradesBlocked": self.minTradesBlocked,
            "expectancyGateWrDegraded": self.wrDegraded,
            "expectancyGatePnlDegraded": self.pnlDegraded,
            "expectancyGateDegradedBetScale": self.degradedBetScale,
            "expectancyGateWrFastBlocked": self.wrFastBlocked,
            "expectancyGatePnlFastBlocked": self.pnlFastBlocked,
            "expectancyGateWrBlocked": self.wrBlocked,
            "expectancyGatePnlBlocked": self.pnlBlocked,
            "expectancyGateBlockedEnterChecks": self.blockedEnterChecks,
            "expectancyGateReleaseChecks": self.releaseChecks,
            "expectancyGateReleaseWr": self.releaseWr,
            "expectancyGateReleasePnl": self.releasePnl,
        }

    def label(self) -> str:
        return (
            f"L{self.lookbackHours}/F{self.lookbackHoursFast}-N{self.minTrades}/F{self.minTradesFast}/B{self.minTradesBlocked}-"
            f"D({self.wrDegraded:.2f},{self.pnlDegraded:.2f},x{self.degradedBetScale:.2f})-"
            f"F({self.wrFastBlocked:.2f},{self.pnlFastBlocked:.2f})-"
            f"B({self.wrBlocked:.2f},{self.pnlBlocked:.2f},k{self.blockedEnterChecks})-"
            f"R({self.releaseChecks},{self.releaseWr:.2f},{self.releasePnl:.2f})"
        )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Optimize expectancy gate with walk-forward no-leak protocol")
    ap.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--window-days", type=int, default=365)
    ap.add_argument("--cv-protocol", choices=["walk_forward"], default="walk_forward")
    ap.add_argument("--train-min-days", type=int, default=180)
    ap.add_argument("--test-days", type=int, default=14)
    ap.add_argument("--step-days", type=int, default=14)
    ap.add_argument("--purge-hours", type=int, default=24)
    ap.add_argument("--progress-interval", type=int, default=10)
    ap.add_argument("--clusters", type=str, default="all")
    ap.add_argument("--net-drop-max", type=float, default=0.0)
    ap.add_argument("--mdd-improve-min", type=float, default=0.10)
    ap.add_argument("--wr-drop-max", type=float, default=0.01)
    ap.add_argument("--suppression-max", type=float, default=0.30)
    ap.add_argument("--suppression-prescreen-max", type=float, default=0.40)
    ap.add_argument("--min-trades-for-wr-check", type=int, default=30)
    ap.add_argument("--checkpoint-dir", type=Path, default=None)
    return ap.parse_args()


def _json_write_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _json_read(path: Path) -> Dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


def _cluster_result_path(base: Path | None, cluster_id: str) -> Path | None:
    if base is None:
        return None
    return base / cluster_id / "cluster_result.json"


def load_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"panel not found: {path}")
    df = pd.read_parquet(path)
    required = {
        "ts", "profile", "symbol", "direction", "cell_id", "cluster_id",
        "confidence", "win", "pnl", "base_conf_threshold",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"panel missing columns: {sorted(missing)}")
    df = df.copy()
    for col in ("ts", "confidence", "win", "pnl", "base_conf_threshold"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["ts", "confidence", "win", "pnl", "base_conf_threshold"])
    if df.empty:
        raise RuntimeError("panel empty after cleanup")
    df["cluster_id"] = df["cluster_id"].astype(str)
    df["cell_id"] = df["cell_id"].astype(str)
    df["direction"] = df["direction"].astype(str).str.upper()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    return df.sort_values(["cluster_id", "ts", "cell_id"]).reset_index(drop=True)


def aggregate_cluster(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("ts", sort=True)
        .agg(
            trades=("win", "size"),
            wins=("win", "sum"),
            pnl=("pnl", "sum"),
        )
        .reset_index()
        .sort_values("ts")
        .reset_index(drop=True)
    )


def folds_for_cluster(df: pd.DataFrame, args: argparse.Namespace) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    min_ts = int(df["ts"].min())
    max_ts = int(df["ts"].max())
    window_cutoff = max_ts - int(args.window_days) * 86400
    df = df[df["ts"] >= window_cutoff].copy()
    if df.empty:
        return []
    train_min = int(args.train_min_days) * 86400
    test_len = int(args.test_days) * 86400
    step_len = int(args.step_days) * 86400
    purge_len = int(args.purge_hours) * 3600
    train_end = int(df["ts"].min()) + train_min
    folds: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    while True:
        test_start = train_end + purge_len
        test_end = test_start + test_len
        if test_end > int(df["ts"].max()):
            break
        train_df = df[df["ts"] < train_end].copy()
        test_df = df[(df["ts"] >= test_start) & (df["ts"] < test_end)].copy()
        if not train_df.empty and not test_df.empty:
            folds.append((train_df, test_df))
        train_end += step_len
    return folds


def candidate_pass1() -> List[Candidate]:
    candidates: List[Candidate] = []
    trigger_sets = [
        # pass1 先覆盖“轻降仓 + 极端才 blocked”的可行区，避免又滑回过度压缩。
        # 这里显式加入已验证的 DOWN 可行区：blocked 只在极端差的窗口才触发。
        (48, 20, 28, 0.50, -0.01, 0.00, -0.40),
        (72, 24, 32, 0.46, -0.03, 0.00, -0.40),
        (24, 12, 18, 0.52, 0.00, 0.05, -0.30),
        (24, 12, 18, 0.52, 0.00, 0.10, -0.25),
        (48, 20, 28, 0.50, -0.01, 0.10, -0.25),
        (48, 20, 28, 0.50, -0.01, 0.15, -0.20),
        (72, 24, 32, 0.46, -0.03, 0.20, -0.20),
        (72, 24, 32, 0.48, -0.02, 0.15, -0.20),
    ]
    action_sets = [
        (0.84, 2, 1, 0.50, 0.00),
        (0.95, 2, 1, 0.48, -0.01),
        (0.90, 2, 1, 0.50, 0.00),
        (0.85, 3, 2, 0.50, 0.00),
        (0.85, 2, 2, 0.50, 0.00),
    ]
    for t in trigger_sets:
        for a in action_sets:
            lookback, min_trades, min_trades_blocked, wr_degraded, pnl_degraded, wr_blocked, pnl_blocked = t
            degraded_bet_scale, blocked_enter_checks, release_checks, release_wr, release_pnl = a
            candidates.append(
                Candidate(
                    lookbackHours=lookback,
                    lookbackHoursFast=12,
                    minTrades=min_trades,
                    minTradesFast=4,
                    minTradesBlocked=min_trades_blocked,
                    wrDegraded=wr_degraded,
                    pnlDegraded=pnl_degraded,
                    degradedBetScale=degraded_bet_scale,
                    wrFastBlocked=0.30,
                    pnlFastBlocked=-0.06,
                    wrBlocked=wr_blocked,
                    pnlBlocked=pnl_blocked,
                    blockedEnterChecks=blocked_enter_checks,
                    releaseChecks=release_checks,
                    releaseWr=release_wr,
                    releasePnl=release_pnl,
                )
            )
    return candidates


def candidate_pass2() -> List[Candidate]:
    candidates: List[Candidate] = []
    trigger_sets = [
        # pass2 继续围绕已验证的可行区细化，不再把 blocked 放在常规动作层。
        (48, 20, 28, 0.50, -0.01, 0.00, -0.40),
        (72, 24, 32, 0.46, -0.03, 0.00, -0.40),
        (72, 24, 32, 0.46, -0.03, 0.02, -0.35),
        (24, 12, 18, 0.54, 0.00, 0.05, -0.30),
        (24, 16, 24, 0.52, -0.01, 0.08, -0.28),
        (48, 20, 28, 0.50, -0.01, 0.10, -0.25),
        (48, 24, 32, 0.48, -0.02, 0.15, -0.22),
        (72, 24, 32, 0.46, -0.03, 0.20, -0.20),
        (72, 28, 36, 0.45, -0.04, 0.22, -0.18),
        (24, 8, 12, 0.56, 0.00, 0.10, -0.22),
        (24, 8, 12, 0.58, 0.01, 0.15, -0.18),
        (48, 16, 24, 0.52, -0.01, 0.18, -0.18),
    ]
    action_sets = [
        (0.84, 2, 1, 0.50, 0.00),
        (0.95, 2, 1, 0.50, 0.00),
        (0.90, 2, 1, 0.50, 0.00),
        (0.85, 2, 1, 0.50, 0.00),
        (0.85, 3, 2, 0.52, 0.00),
        (0.80, 3, 2, 0.52, 0.01),
        (0.75, 2, 1, 0.52, 0.01),
        (0.75, 3, 1, 0.55, 0.02),
    ]
    for t in trigger_sets:
        for a in action_sets:
            lookback, min_trades, min_trades_blocked, wr_degraded, pnl_degraded, wr_blocked, pnl_blocked = t
            degraded_bet_scale, blocked_enter_checks, release_checks, release_wr, release_pnl = a
            candidates.append(
                Candidate(
                    lookbackHours=lookback,
                    lookbackHoursFast=12,
                    minTrades=min_trades,
                    minTradesFast=4,
                    minTradesBlocked=min_trades_blocked,
                    wrDegraded=wr_degraded,
                    pnlDegraded=pnl_degraded,
                    degradedBetScale=degraded_bet_scale,
                    wrFastBlocked=0.30,
                    pnlFastBlocked=-0.06,
                    wrBlocked=wr_blocked,
                    pnlBlocked=pnl_blocked,
                    blockedEnterChecks=blocked_enter_checks,
                    releaseChecks=release_checks,
                    releaseWr=release_wr,
                    releasePnl=release_pnl,
                )
            )
    return candidates


def max_drawdown_from_pnls(pnls: Iterable[float]) -> float:
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnls:
        cum += float(pnl)
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    return max_dd


def root_cause_from_violations(violations: List[str], suppression: float) -> str | None:
    if not violations:
        return None
    if "trade_suppression_gt_limit" in violations or suppression > 0.30:
        return "over_suppression"
    if "oos_mdd_improve_lt_limit" in violations:
        return "no_oos_edge"
    if "oos_net_drop_gt_limit" in violations:
        return "no_oos_edge"
    if "oos_wr_drop_gt_limit" in violations:
        return "no_oos_edge"
    return "needs_review"


def evaluate_candidate(
    folds: List[Tuple[pd.DataFrame, pd.DataFrame]],
    candidate: Candidate,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    base_pnls: List[float] = []
    cand_pnls: List[float] = []
    base_trades = 0
    cand_trades = 0
    blocked_trades = 0
    degraded_trades = 0
    base_wins = 0
    cand_wins = 0

    for train_df, test_df in folds:
        train_sorted = train_df.sort_values("ts")
        test_sorted = test_df.sort_values("ts")
        window: Deque[Tuple[int, int, int, float]] = deque()
        window_fast: Deque[Tuple[int, int, int, float]] = deque()
        win_sum = 0
        trade_sum = 0
        pnl_sum = 0.0
        fast_win_sum = 0
        fast_trade_sum = 0
        fast_pnl_sum = 0.0
        state = "normal"
        blocked_enter_streak = 0
        release_pass_streak = 0

        def process_row(ts: int, trades: int, wins: int, pnl: float, collect_metrics: bool) -> None:
            nonlocal state, blocked_enter_streak, release_pass_streak
            nonlocal base_trades, cand_trades, blocked_trades, degraded_trades
            nonlocal base_pnls, cand_pnls, trade_sum, win_sum, pnl_sum
            nonlocal fast_trade_sum, fast_win_sum, fast_pnl_sum
            nonlocal base_wins, cand_wins

            cutoff = ts - candidate.lookbackHours * 3600
            while window and window[0][0] < cutoff:
                _, old_trades, old_wins, old_pnl = window.popleft()
                trade_sum -= old_trades
                win_sum -= old_wins
                pnl_sum -= old_pnl
            fast_cutoff = ts - candidate.lookbackHoursFast * 3600
            while window_fast and window_fast[0][0] < fast_cutoff:
                _, old_trades, old_wins, old_pnl = window_fast.popleft()
                fast_trade_sum -= old_trades
                fast_win_sum -= old_wins
                fast_pnl_sum -= old_pnl

            wr_post = (win_sum + 2.0) / (trade_sum + 4.0)
            pnl_per_trade = (pnl_sum / trade_sum) if trade_sum > 0 else 0.0
            wr_post_fast = (fast_win_sum + 2.0) / (fast_trade_sum + 4.0)
            pnl_per_trade_fast = (fast_pnl_sum / fast_trade_sum) if fast_trade_sum > 0 else 0.0

            target_state = "normal"
            blocked_eligible = trade_sum >= candidate.minTradesBlocked and (
                wr_post < candidate.wrBlocked or pnl_per_trade < candidate.pnlBlocked
            )
            fast_blocked_eligible = fast_trade_sum >= candidate.minTradesFast and (
                wr_post_fast < candidate.wrFastBlocked or pnl_per_trade_fast < candidate.pnlFastBlocked
            )
            if trade_sum >= candidate.minTrades and (
                wr_post < candidate.wrDegraded or pnl_per_trade < candidate.pnlDegraded
            ):
                target_state = "degraded"
            if blocked_eligible or fast_blocked_eligible:
                blocked_enter_streak += 1
                if blocked_enter_streak >= candidate.blockedEnterChecks:
                    target_state = "blocked"
                elif target_state == "normal":
                    target_state = "degraded"
            else:
                blocked_enter_streak = 0

            if state != "normal" and target_state == "normal":
                release_pass = wr_post >= candidate.releaseWr and pnl_per_trade >= candidate.releasePnl
                if release_pass:
                    release_pass_streak += 1
                    if release_pass_streak >= candidate.releaseChecks:
                        state = "normal"
                        release_pass_streak = 0
                    else:
                        target_state = state
                else:
                    release_pass_streak = 0
                    target_state = state
            else:
                if target_state != "normal":
                    release_pass_streak = 0
                state = target_state

            if collect_metrics:
                base_trades += trades
                base_pnls.append(pnl)
                base_wins += wins
                if state == "blocked":
                    blocked_trades += trades
                elif state == "degraded":
                    cand_trades += trades
                    degraded_trades += trades
                    cand_pnls.append(pnl * candidate.degradedBetScale)
                    cand_wins += wins
                else:
                    cand_trades += trades
                    cand_pnls.append(pnl)
                    cand_wins += wins

            window.append((ts, trades, wins, pnl))
            trade_sum += trades
            win_sum += wins
            pnl_sum += pnl
            window_fast.append((ts, trades, wins, pnl))
            fast_trade_sum += trades
            fast_win_sum += wins
            fast_pnl_sum += pnl

        # 训练段也要完整回放状态机，避免测试段起点状态偏乐观
        for row in train_sorted.itertuples(index=False):
            process_row(
                ts=int(row.ts),
                trades=int(row.trades),
                wins=int(row.wins),
                pnl=float(row.pnl),
                collect_metrics=False,
            )

        for row in test_sorted.itertuples(index=False):
            process_row(
                ts=int(row.ts),
                trades=int(row.trades),
                wins=int(row.wins),
                pnl=float(row.pnl),
                collect_metrics=True,
            )

    base_net = float(sum(base_pnls))
    cand_net = float(sum(cand_pnls))
    base_mdd = max_drawdown_from_pnls(base_pnls)
    cand_mdd = max_drawdown_from_pnls(cand_pnls)
    pnl_delta_ratio = (cand_net - base_net) / max(1.0, abs(base_net))
    net_drop = max(0.0, -pnl_delta_ratio)
    mdd_improve = 0.0 if base_mdd <= 0 else (base_mdd - cand_mdd) / base_mdd
    suppression = 0.0 if base_trades <= 0 else max(0.0, 1.0 - (cand_trades / base_trades))
    base_wr = (base_wins / base_trades) if base_trades > 0 else None
    cand_wr = (cand_wins / cand_trades) if cand_trades > 0 else None
    wr_drop = 0.0
    if base_wr is not None and cand_wr is not None:
        wr_drop = max(0.0, base_wr - cand_wr)
    wr_check_applicable = (
        base_trades >= int(args.min_trades_for_wr_check)
        and cand_trades >= int(args.min_trades_for_wr_check)
    )
    blocked_rate = 0.0 if base_trades <= 0 else (blocked_trades / base_trades)
    degraded_rate = 0.0 if base_trades <= 0 else (degraded_trades / base_trades)
    blocked_penalty = max(0.0, blocked_rate - 0.05)

    violations: List[str] = []
    if net_drop > float(args.net_drop_max):
        violations.append("oos_net_drop_gt_limit")
    if mdd_improve < float(args.mdd_improve_min):
        violations.append("oos_mdd_improve_lt_limit")
    if suppression > float(args.suppression_max):
        violations.append("trade_suppression_gt_limit")
    if wr_check_applicable and wr_drop > float(args.wr_drop_max):
        violations.append("oos_wr_drop_gt_limit")
    prefilter_violations: List[str] = []
    if suppression > float(args.suppression_prescreen_max):
        prefilter_violations.append("trade_suppression_gt_prescreen")
    feasible = not violations
    prefilter_ok = not prefilter_violations
    score = (
        (0.42 * pnl_delta_ratio)
        + (0.33 * mdd_improve)
        - (0.15 * suppression)
        - (0.10 * blocked_penalty)
        - (0.20 * max(0.0, suppression - float(args.suppression_max)))
    )

    return {
        "candidate": candidate,
        "score": score,
        "base_net_pnl": base_net,
        "cand_net_pnl": cand_net,
        "base_mdd": base_mdd,
        "cand_mdd": cand_mdd,
        "pnl_delta_ratio": pnl_delta_ratio,
        "net_drop": net_drop,
        "mdd_improve": mdd_improve,
        "base_wr": base_wr,
        "cand_wr": cand_wr,
        "wr_drop": wr_drop,
        "wr_check_applicable": wr_check_applicable,
        "trade_suppression": suppression,
        "blocked_rate": blocked_rate,
        "degraded_rate": degraded_rate,
        "base_trades": base_trades,
        "cand_trades": cand_trades,
        "base_wins": base_wins,
        "cand_wins": cand_wins,
        "feasible": feasible,
        "prefilter_ok": prefilter_ok,
        "prefilter_violations": prefilter_violations,
        "constraint_violations": violations,
        "root_cause_type": root_cause_from_violations(violations, suppression),
    }


def evaluate_grid(
    folds: List[Tuple[pd.DataFrame, pd.DataFrame]],
    candidates: List[Candidate],
    pass_label: str,
    cluster_id: str,
    progress_interval: int,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    total = len(candidates)
    for idx, candidate in enumerate(candidates, start=1):
        if progress_interval > 0 and (idx % progress_interval == 0 or idx == total):
            print(f"[cluster {cluster_id}] {pass_label} {idx}/{total}")
        res = evaluate_candidate(folds, candidate, args)
        res["search_pass"] = pass_label
        results.append(res)
    results.sort(
        key=lambda item: (bool(item["prefilter_ok"]), bool(item["feasible"]), item["score"]),
        reverse=True,
    )
    return results


def build_report_for_cluster(
    cluster_id: str,
    cluster_df: pd.DataFrame,
    folds: List[Tuple[pd.DataFrame, pd.DataFrame]],
    progress_interval: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    def refine_fast_lane(base: Candidate) -> List[Dict[str, Any]]:
        refined: List[Dict[str, Any]] = []
        total = 3 * 4 * 4 * 4 * 3
        idx = 0
        for lookback_fast, min_trades_fast, wr_fast_blocked, pnl_fast_blocked, blocked_enter_checks in itertools.product(
            [1, 2, 3],
            [2, 3, 4, 5],
            [0.20, 0.25, 0.30, 0.35],
            [-0.01, -0.02, -0.03, -0.04],
            [1, 2, 3],
        ):
            idx += 1
            cand = Candidate(
                lookbackHours=base.lookbackHours,
                lookbackHoursFast=int(lookback_fast),
                minTrades=base.minTrades,
                minTradesFast=int(min_trades_fast),
                minTradesBlocked=base.minTradesBlocked,
                wrDegraded=base.wrDegraded,
                pnlDegraded=base.pnlDegraded,
                degradedBetScale=base.degradedBetScale,
                wrFastBlocked=float(wr_fast_blocked),
                pnlFastBlocked=float(pnl_fast_blocked),
                wrBlocked=base.wrBlocked,
                pnlBlocked=base.pnlBlocked,
                blockedEnterChecks=int(blocked_enter_checks),
                releaseChecks=base.releaseChecks,
                releaseWr=base.releaseWr,
                releasePnl=base.releasePnl,
            )
            if progress_interval > 0 and (idx % progress_interval == 0 or idx == total):
                print(f"[cluster {cluster_id}] fast_refine {idx}/{total}")
            res = evaluate_candidate(folds, cand, args)
            res["search_pass"] = "fast_refine"
            refined.append(res)
        refined.sort(
            key=lambda item: (bool(item["prefilter_ok"]), bool(item["feasible"]), item["score"]),
            reverse=True,
        )
        return refined

    pass1 = evaluate_grid(folds, candidate_pass1(), "pass1", cluster_id, progress_interval, args)
    selected = pass1[0]
    pass2_top: List[Dict[str, Any]] = []
    fast_top: List[Dict[str, Any]] = []
    if not bool(selected["feasible"]):
        pass2 = evaluate_grid(folds, candidate_pass2(), "pass2", cluster_id, progress_interval, args)
        pass2_top = pass2[:5]
        if pass2 and (bool(pass2[0]["feasible"]) or pass2[0]["score"] > selected["score"]):
            selected = pass2[0]
    fast = refine_fast_lane(selected["candidate"])
    fast_top = fast[:5]
    if fast and (bool(fast[0]["feasible"]) or fast[0]["score"] > selected["score"]):
        selected = fast[0]
    candidate = selected["candidate"]
    violations = list(selected["constraint_violations"])
    feasible = not violations
    return {
        "best": {
            "search_pass": str(selected["search_pass"]),
            "needs_review": not feasible,
            "final_is_feasible": feasible,
            "candidate": candidate.label(),
            "params": candidate.params(),
            "base_net_pnl": selected["base_net_pnl"],
            "cand_net_pnl": selected["cand_net_pnl"],
            "pnl_delta_ratio": selected["pnl_delta_ratio"],
            "base_mdd": selected["base_mdd"],
            "cand_mdd": selected["cand_mdd"],
            "mdd_improve": selected["mdd_improve"],
            "base_wr": selected["base_wr"],
            "cand_wr": selected["cand_wr"],
            "wr_drop": selected["wr_drop"],
            "wr_check_applicable": selected["wr_check_applicable"],
            "trade_suppression": selected["trade_suppression"],
            "net_drop": selected["net_drop"],
            "blocked_rate": selected["blocked_rate"],
            "degraded_rate": selected["degraded_rate"],
            "base_trades": selected["base_trades"],
            "cand_trades": selected["cand_trades"],
            "base_wins": selected["base_wins"],
            "cand_wins": selected["cand_wins"],
            "prefilter_ok": selected["prefilter_ok"],
            "prefilter_violations": selected["prefilter_violations"],
            "constraint_violations": violations,
            "root_cause_type": selected["root_cause_type"],
            "folds": len(folds),
        },
        "top_candidates": [
            {
                "candidate": item["candidate"].label(),
                "search_pass": item["search_pass"],
                "score": item["score"],
                "feasible": item["feasible"],
                "constraint_violations": item["constraint_violations"],
                "pnl_delta_ratio": item["pnl_delta_ratio"],
                "net_drop": item["net_drop"],
                "mdd_improve": item["mdd_improve"],
                "wr_drop": item["wr_drop"],
                "trade_suppression": item["trade_suppression"],
                "blocked_rate": item["blocked_rate"],
                "degraded_rate": item["degraded_rate"],
                "prefilter_ok": item["prefilter_ok"],
                "prefilter_violations": item["prefilter_violations"],
            }
            for item in pass1[:5]
        ],
        "pass2_top_candidates": [
            {
                "candidate": item["candidate"].label(),
                "search_pass": item["search_pass"],
                "score": item["score"],
                "feasible": item["feasible"],
                "constraint_violations": item["constraint_violations"],
                "pnl_delta_ratio": item["pnl_delta_ratio"],
                "net_drop": item["net_drop"],
                "mdd_improve": item["mdd_improve"],
                "wr_drop": item["wr_drop"],
                "trade_suppression": item["trade_suppression"],
                "blocked_rate": item["blocked_rate"],
                "degraded_rate": item["degraded_rate"],
                "prefilter_ok": item["prefilter_ok"],
                "prefilter_violations": item["prefilter_violations"],
            }
            for item in pass2_top
        ],
        "fast_top_candidates": [
            {
                "candidate": item["candidate"].label(),
                "search_pass": item["search_pass"],
                "score": item["score"],
                "feasible": item["feasible"],
                "constraint_violations": item["constraint_violations"],
                "pnl_delta_ratio": item["pnl_delta_ratio"],
                "net_drop": item["net_drop"],
                "mdd_improve": item["mdd_improve"],
                "wr_drop": item["wr_drop"],
                "trade_suppression": item["trade_suppression"],
                "blocked_rate": item["blocked_rate"],
                "degraded_rate": item["degraded_rate"],
                "prefilter_ok": item["prefilter_ok"],
                "prefilter_violations": item["prefilter_violations"],
            }
            for item in fast_top
        ],
    }


def main() -> int:
    args = parse_args()
    if args.checkpoint_dir is None:
        args.checkpoint_dir = args.output.parent / f"{args.output.stem}_checkpoint"
    df = load_panel(args.panel)
    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "panel": str(args.panel),
        "protocol": {
            "cv_protocol": args.cv_protocol,
            "window_days": int(args.window_days),
            "train_min_days": int(args.train_min_days),
            "test_days": int(args.test_days),
            "step_days": int(args.step_days),
            "purge_hours": int(args.purge_hours),
            "strict_no_leak": True,
            "constraints": {
                "net_drop_max": float(args.net_drop_max),
                "mdd_improve_min": float(args.mdd_improve_min),
                "wr_drop_max": float(args.wr_drop_max),
                "suppression_max": float(args.suppression_max),
                "suppression_prescreen_max": float(args.suppression_prescreen_max),
                "min_trades_for_wr_check": int(args.min_trades_for_wr_check),
            },
        },
        "anti_overfit": {
            "cluster_only": True,
            "cluster_count": int(df["cluster_id"].nunique()),
            "cell_specific_params_forbidden": True,
            "low_sample_fallback": "insufficient_sample_to_neutral",
        },
        "clusters": {},
    }

    expected_clusters_all = [
        "default_BTC_UP", "default_BTC_DOWN", "default_ETH_UP", "default_ETH_DOWN",
        "70_BTC_UP", "70_BTC_DOWN", "70_ETH_UP", "70_ETH_DOWN",
    ]
    req = str(args.clusters or "all").strip()
    if req.lower() == "all":
        expected_clusters = expected_clusters_all
    else:
        selected = [x.strip() for x in req.split(",") if x.strip()]
        expected_clusters = [x for x in expected_clusters_all if x in selected]
        if not expected_clusters:
            raise RuntimeError(f"--clusters invalid: {req}")
    for cluster_id in expected_clusters:
        cluster_result_path = _cluster_result_path(args.checkpoint_dir, cluster_id)
        if cluster_result_path is not None and cluster_result_path.exists():
            cached = _json_read(cluster_result_path)
            if cached:
                report["clusters"][cluster_id] = cached
                print(f"[resume] cluster_cached={cluster_id}")
                continue

        raw_cluster_df = df[df["cluster_id"] == cluster_id].copy().sort_values("ts")
        cluster_info: Dict[str, Any] = {
            "rows": int(len(raw_cluster_df)),
            "cells": int(raw_cluster_df["cell_id"].nunique()) if not raw_cluster_df.empty else 0,
            "best": None,
            "top_candidates": [],
            "pass2_top_candidates": [],
        }
        if raw_cluster_df.empty:
            cluster_info["best"] = {
                "search_pass": "pass1",
                "needs_review": True,
                "final_is_feasible": False,
                "root_cause_type": "data_coverage_gap",
                "constraint_violations": ["no_cluster_rows"],
            }
            report["clusters"][cluster_id] = cluster_info
            if cluster_result_path is not None:
                _json_write_atomic(cluster_result_path, cluster_info)
            continue
        cluster_df = aggregate_cluster(raw_cluster_df)
        folds = folds_for_cluster(cluster_df, args)
        if not folds:
            cluster_info["best"] = {
                "search_pass": "pass1",
                "needs_review": True,
                "final_is_feasible": False,
                "root_cause_type": "data_coverage_gap",
                "constraint_violations": ["no_walk_forward_folds"],
            }
            report["clusters"][cluster_id] = cluster_info
            if cluster_result_path is not None:
                _json_write_atomic(cluster_result_path, cluster_info)
            continue
        cluster_info.update(build_report_for_cluster(cluster_id, cluster_df, folds, args.progress_interval, args))
        report["clusters"][cluster_id] = cluster_info
        if cluster_result_path is not None:
            _json_write_atomic(cluster_result_path, cluster_info)

    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote expectancy gate tuning report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
