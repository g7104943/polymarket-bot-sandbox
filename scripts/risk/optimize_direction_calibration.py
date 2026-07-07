#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PANEL = PROJECT_ROOT / "reports" / "direction_calibration_panel.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "direction_calibration_tuning_report.json"


@dataclass(frozen=True)
class Candidate:
    method: str
    temperature: float = 1.0
    alpha: float = 0.0
    beta: float = 1.0
    bins: int = 0

    def label(self) -> str:
        if self.method == "temperature":
            return f"temperature(T={self.temperature:.2f})"
        if self.method == "sigmoid":
            return f"sigmoid(a={self.alpha:.2f},b={self.beta:.2f})"
        if self.method == "isotonic":
            return f"isotonic(bins={self.bins})"
        return "identity"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Optimize direction-level calibration with walk-forward no-leak protocol")
    ap.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--window-days", type=int, default=365)
    ap.add_argument("--cv-protocol", choices=["walk_forward"], default="walk_forward")
    ap.add_argument("--train-min-days", type=int, default=180)
    ap.add_argument("--test-days", type=int, default=14)
    ap.add_argument("--step-days", type=int, default=14)
    ap.add_argument("--purge-hours", type=int, default=24)
    ap.add_argument("--ece-bins", type=int, default=10)
    ap.add_argument("--progress-interval", type=int, default=20)
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
    return df.sort_values(["cluster_id", "cell_id", "ts"]).reset_index(drop=True)


def candidate_grid() -> List[Candidate]:
    out: List[Candidate] = [Candidate("identity")]
    # 优先覆盖低抑制区间，但保留足够形状以寻找“低抑制+回撤改善”的折中点。
    for temperature in (0.75, 0.85, 0.95, 1.00, 1.05, 1.15, 1.30):
        out.append(Candidate("temperature", temperature=temperature))
    for alpha in (-0.25, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05):
        for beta in (0.75, 0.85, 0.95, 1.0, 1.05, 1.15):
            out.append(Candidate("sigmoid", alpha=alpha, beta=beta))
    for bins in (4, 5, 6, 7, 8, 10):
        out.append(Candidate("isotonic", bins=bins))
    return out


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


def fit_isotonic(xs: np.ndarray, ys: np.ndarray, bins: int) -> Tuple[List[float], List[float]]:
    frame = pd.DataFrame({"x": xs, "y": ys}).sort_values("x").reset_index(drop=True)
    frame["bin"] = pd.qcut(frame.index, q=min(bins, len(frame)), labels=False, duplicates="drop")
    grouped = frame.groupby("bin", sort=True).agg(x=("x", "mean"), y=("y", "mean")).reset_index(drop=True)
    x_vals = grouped["x"].astype(float).tolist()
    y_vals = grouped["y"].astype(float).tolist()
    mono_y = list(np.maximum.accumulate(np.array(y_vals, dtype=float)))
    return x_vals, mono_y


def fit_calibrator(train_df: pd.DataFrame, candidate: Candidate) -> Dict[str, Any]:
    conf = train_df["confidence"].to_numpy(dtype=float)
    wins = train_df["win"].to_numpy(dtype=float)
    if candidate.method == "isotonic":
        x_vals, y_vals = fit_isotonic(conf, wins, bins=max(2, candidate.bins))
        return {"method": "isotonic", "x": x_vals, "y": y_vals}
    return {
        "method": candidate.method,
        "temperature": candidate.temperature,
        "alpha": candidate.alpha,
        "beta": candidate.beta,
    }


def apply_calibrator(confidence: np.ndarray, model: Dict[str, Any]) -> np.ndarray:
    conf = np.clip(confidence.astype(float), 1e-6, 1 - 1e-6)
    method = str(model.get("method") or "identity")
    if method == "temperature":
        temp = max(0.05, float(model.get("temperature", 1.0)))
        logit = np.log(conf / (1.0 - conf))
        return 1.0 / (1.0 + np.exp(-(logit / temp)))
    if method == "sigmoid":
        alpha = float(model.get("alpha", 0.0))
        beta = float(model.get("beta", 1.0))
        logit = np.log(conf / (1.0 - conf))
        return 1.0 / (1.0 + np.exp(-(alpha + beta * logit)))
    if method == "isotonic":
        xs = np.array(model.get("x") or [], dtype=float)
        ys = np.array(model.get("y") or [], dtype=float)
        if len(xs) >= 2 and len(xs) == len(ys):
            return np.interp(conf, xs, ys, left=ys[0], right=ys[-1])
    return conf


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, bins: int) -> float:
    if len(y_true) == 0:
        return 0.0
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    n = len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi >= 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        total += abs(float(y_true[mask].mean()) - float(y_prob[mask].mean())) * (mask.sum() / n)
    return float(total)


def max_drawdown_from_pnls(pnls: Iterable[float]) -> float:
    cum = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnls:
        cum += float(pnl)
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)
    return max_dd


def evaluate_candidate(
    cluster_df: pd.DataFrame,
    folds: List[Tuple[pd.DataFrame, pd.DataFrame]],
    candidate: Candidate,
    ece_bins: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    brier_values: List[float] = []
    ece_values: List[float] = []
    cand_pnls: List[float] = []
    base_pnls: List[float] = []
    cand_trades = 0
    base_trades = 0
    cand_wins = 0
    base_wins = 0
    for train_df, test_df in folds:
        model = fit_calibrator(train_df, candidate)
        calibrated = apply_calibrator(test_df["confidence"].to_numpy(dtype=float), model)
        y_true = test_df["win"].to_numpy(dtype=float)
        brier_values.append(float(np.mean((calibrated - y_true) ** 2)))
        ece_values.append(ece_score(y_true, calibrated, bins=ece_bins))
        base_mask = test_df["confidence"].to_numpy(dtype=float) >= test_df["base_conf_threshold"].to_numpy(dtype=float)
        cand_mask = calibrated >= test_df["base_conf_threshold"].to_numpy(dtype=float)
        base_selected = test_df.loc[base_mask, ["ts", "pnl"]].sort_values("ts")
        cand_selected = test_df.loc[cand_mask, ["ts", "pnl"]].sort_values("ts")
        base_pnls.extend(base_selected["pnl"].astype(float).tolist())
        cand_pnls.extend(cand_selected["pnl"].astype(float).tolist())
        base_trades += int(base_mask.sum())
        cand_trades += int(cand_mask.sum())
        if base_mask.any():
            base_wins += int(y_true[base_mask].sum())
        if cand_mask.any():
            cand_wins += int(y_true[cand_mask].sum())

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
        (-0.55 * float(np.mean(brier_values) if brier_values else 0.0))
        + (-0.20 * float(np.mean(ece_values) if ece_values else 0.0))
        + (0.15 * mdd_improve)
        + (0.10 * pnl_delta_ratio)
        - (0.20 * max(0.0, suppression - float(args.suppression_max)))
    )
    return {
        "candidate": candidate,
        "score": score,
        "brier": float(np.mean(brier_values) if brier_values else 0.0),
        "ece": float(np.mean(ece_values) if ece_values else 0.0),
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
        "base_trades": base_trades,
        "cand_trades": cand_trades,
        "base_wins": base_wins,
        "cand_wins": cand_wins,
        "feasible": feasible,
        "prefilter_ok": prefilter_ok,
        "prefilter_violations": prefilter_violations,
        "constraint_violations": violations,
    }


def build_final_params(cluster_df: pd.DataFrame, candidate: Candidate) -> Dict[str, Any]:
    model = fit_calibrator(cluster_df, candidate)
    params: Dict[str, Any] = {
        "calibrationMethod": str(model.get("method") or candidate.method),
        "calibrationLookbackHours": 72,
        "calibrationMinSamples": 24,
    }
    if params["calibrationMethod"] == "temperature":
        params["calibrationTemperature"] = float(model.get("temperature", candidate.temperature))
    elif params["calibrationMethod"] == "sigmoid":
        params["calibrationAlpha"] = float(model.get("alpha", candidate.alpha))
        params["calibrationBeta"] = float(model.get("beta", candidate.beta))
    elif params["calibrationMethod"] == "isotonic":
        params["calibrationIsotonicX"] = [float(x) for x in (model.get("x") or [])]
        params["calibrationIsotonicY"] = [float(y) for y in (model.get("y") or [])]
    return params


def infer_root_cause(violations: List[str]) -> str | None:
    if not violations:
        return None
    if "trade_suppression_gt_limit" in violations:
        return "over_suppression"
    if "oos_mdd_improve_lt_limit" in violations:
        return "no_oos_edge"
    if "oos_net_drop_gt_limit" in violations:
        return "no_oos_edge"
    if "oos_wr_drop_gt_limit" in violations:
        return "no_oos_edge"
    return "needs_review"


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
        "clusters": {},
        "anti_overfit": {
            "cluster_only": True,
            "cluster_count": int(df["cluster_id"].nunique()),
            "cell_specific_params_forbidden": True,
        },
    }
    candidates = candidate_grid()
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
    progress = 0
    for cluster_id in expected_clusters:
        cluster_result_path = _cluster_result_path(args.checkpoint_dir, cluster_id)
        if cluster_result_path is not None and cluster_result_path.exists():
            cached = _json_read(cluster_result_path)
            if cached:
                report["clusters"][cluster_id] = cached
                print(f"[resume] cluster_cached={cluster_id}")
                continue

        cluster_df = df[df["cluster_id"] == cluster_id].copy().sort_values("ts")
        cluster_info: Dict[str, Any] = {
            "rows": int(len(cluster_df)),
            "cells": int(cluster_df["cell_id"].nunique()) if not cluster_df.empty else 0,
            "best": None,
            "top_candidates": [],
        }
        if cluster_df.empty:
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

        evaluations: List[Dict[str, Any]] = []
        for candidate in candidates:
            progress += 1
            if args.progress_interval > 0 and progress % args.progress_interval == 0:
                print(f"[progress] {cluster_id}: {progress}/{len(expected_clusters) * len(candidates)}")
            evaluations.append(evaluate_candidate(cluster_df, folds, candidate, args.ece_bins, args))
        evaluations.sort(
            key=lambda x: (bool(x["prefilter_ok"]), bool(x["feasible"]), x["score"]),
            reverse=True,
        )
        best = evaluations[0]
        best_candidate: Candidate = best["candidate"]
        best_params = build_final_params(cluster_df, best_candidate)
        cluster_info["best"] = {
            "search_pass": "pass1",
            "needs_review": not bool(best["feasible"]),
            "final_is_feasible": bool(best["feasible"]),
            "candidate": best_candidate.label(),
            "params": best_params,
            "brier": best["brier"],
            "ece": best["ece"],
            "base_net_pnl": best["base_net_pnl"],
            "cand_net_pnl": best["cand_net_pnl"],
            "pnl_delta_ratio": best["pnl_delta_ratio"],
            "net_drop": best["net_drop"],
            "base_mdd": best["base_mdd"],
            "cand_mdd": best["cand_mdd"],
            "mdd_improve": best["mdd_improve"],
            "base_wr": best["base_wr"],
            "cand_wr": best["cand_wr"],
            "wr_drop": best["wr_drop"],
            "wr_check_applicable": best["wr_check_applicable"],
            "trade_suppression": best["trade_suppression"],
            "prefilter_ok": best["prefilter_ok"],
            "prefilter_violations": best["prefilter_violations"],
            "constraint_violations": best["constraint_violations"],
            "root_cause_type": infer_root_cause(best["constraint_violations"]),
            "folds": len(folds),
        }
        cluster_info["top_candidates"] = [
            {
                "candidate": ev["candidate"].label(),
                "score": ev["score"],
                "feasible": ev["feasible"],
                "constraint_violations": ev["constraint_violations"],
                "pnl_delta_ratio": ev["pnl_delta_ratio"],
                "net_drop": ev["net_drop"],
                "mdd_improve": ev["mdd_improve"],
                "wr_drop": ev["wr_drop"],
                "trade_suppression": ev["trade_suppression"],
                "prefilter_ok": ev["prefilter_ok"],
                "prefilter_violations": ev["prefilter_violations"],
            }
            for ev in evaluations[:5]
        ]
        report["clusters"][cluster_id] = cluster_info
        if cluster_result_path is not None:
            _json_write_atomic(cluster_result_path, cluster_info)

    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote calibration tuning report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
