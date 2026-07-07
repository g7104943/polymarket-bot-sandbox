#!/usr/bin/env python3
from __future__ import annotations

"""Targeted top159 shock/environment filter search.

Research-only. It never mutates live configs, live model profiles, ledgers, or
order state. This search starts from the corrected shock feature alignment:
Kline features are visible only at candle close time.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOP159_SHOCK_EXTREME_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path("/Users/mac/polyfun")
NEXT = ROOT / "polyfun-next"
REPORTS = ROOT / "reports"
SCRIPT_DIR = NEXT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_top159_shock_filter_extreme_search as ext  # type: ignore  # noqa: E402

BJ = timezone(timedelta(hours=8))
START_BANKROLL = 850.0

CURRENT_061_PARAMS: dict[str, Any] = {
    "family": "cluster_gate",
    "action": "raise_score",
    "candidate_id": "06173b68b0d86431",
    "cluster_id": "ef7216a66ff15ccb",
    "min_cluster_hits": 2,
    "shock_score_min": 0.58,
    "any_clusters": [
        ["15m_same_as_top159", "1h_vol_ge_1.6", "4h_vol_ge_1.1"],
        ["15m_same_as_top159", "4h_vol_ge_1.1", "1h_pos_high"],
        ["15m_same_as_top159", "1h_pos_low"],
        ["15m_same_as_top159", "1h_same_as_top159", "1h_trend_down"],
    ],
}

OUT_AUDIT = REPORTS / "top159_shock_filter_cluster_bug_audit_latest.json"
OUT_CLUSTERS = REPORTS / "top159_shock_filter_cluster_bad_clusters_latest.json"
OUT_CANDIDATES = REPORTS / "top159_shock_filter_cluster_candidates_latest.jsonl"
OUT_CHECKPOINT = REPORTS / "top159_shock_filter_cluster_checkpoint_latest.json"
OUT_LEADERBOARD = REPORTS / "top159_shock_filter_cluster_leaderboard_latest.json"
OUT_COMPARE = REPORTS / "top159_shock_filter_cluster_compare_latest.json"
OUT_TABLE = REPORTS / "top159_shock_filter_cluster_profit_table_latest.md"
OUT_VERDICT = REPORTS / "top159_shock_filter_cluster_verdict_latest.md"
OUT_LOG = REPORTS / "top159_shock_filter_cluster_run.log"

G_ENRICHED: pd.DataFrame | None = None
G_ATOMS: dict[str, dict[str, np.ndarray]] | None = None
G_PERIOD_VALS: dict[str, pd.DataFrame] | None = None


def apply_output_suffix(suffix: str | None) -> None:
    """Make independent shard workers write independent files.

    This avoids the unsafe pattern that caused 4 workers to contend for the
    same jsonl/checkpoint files. The merge step can then combine shard files
    deterministically.
    """
    global OUT_AUDIT, OUT_CLUSTERS, OUT_CANDIDATES, OUT_CHECKPOINT, OUT_LEADERBOARD, OUT_COMPARE, OUT_TABLE, OUT_VERDICT, OUT_LOG
    suffix = (suffix or "").strip()
    if not suffix:
        return

    def add_suffix(path: Path) -> Path:
        return path.with_name(f"{path.stem}_{suffix}{path.suffix}")

    OUT_AUDIT = add_suffix(OUT_AUDIT)
    OUT_CLUSTERS = add_suffix(OUT_CLUSTERS)
    OUT_CANDIDATES = add_suffix(OUT_CANDIDATES)
    OUT_CHECKPOINT = add_suffix(OUT_CHECKPOINT)
    OUT_LEADERBOARD = add_suffix(OUT_LEADERBOARD)
    OUT_COMPARE = add_suffix(OUT_COMPARE)
    OUT_TABLE = add_suffix(OUT_TABLE)
    OUT_VERDICT = add_suffix(OUT_VERDICT)
    OUT_LOG = add_suffix(OUT_LOG)


def bj_now() -> datetime:
    return datetime.now(BJ)


def iso(dt: datetime) -> str:
    return dt.astimezone(BJ).isoformat(timespec="seconds")


def log(msg: str) -> None:
    line = f"[{iso(bj_now())}] {msg}"
    print(line, flush=True)
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with OUT_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def stable_key(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def atom_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    score = pd.to_numeric(df["score15"], errors="coerce").fillna(0.0)
    direction = df["direction"].astype(str).str.upper()
    out["dir_UP"] = (direction == "UP").to_numpy()
    out["dir_DOWN"] = (direction == "DOWN").to_numpy()
    out["score_055_057"] = ((score >= 0.55) & (score < 0.57)).to_numpy()
    out["score_057_060"] = ((score >= 0.57) & (score < 0.60)).to_numpy()
    out["score_060_064"] = ((score >= 0.60) & (score < 0.64)).to_numpy()
    out["score_ge_064"] = (score >= 0.64).to_numpy()
    out["score_lt_058"] = (score < 0.58).to_numpy()
    out["score_lt_060"] = (score < 0.60).to_numpy()

    dt = pd.to_datetime(df["dt"], utc=True, errors="coerce").dt.tz_convert("Asia/Shanghai")
    hour = dt.dt.hour.fillna(-1).astype(int)
    for start in [0, 4, 8, 12, 16, 20]:
        out[f"bj_hour_{start:02d}_{start+3:02d}"] = ((hour >= start) & (hour <= start + 3)).to_numpy()

    for tf in ["15m", "1h", "4h"]:
        for flag in ["same_as_top159", "opposes_top159", "terminal_chase", "exhaustion_wick"]:
            col = f"{tf}_{flag}"
            if col in df:
                out[col] = df[col].fillna(False).astype(bool).to_numpy()
        cdir = df.get(f"{tf}_candle_dir", pd.Series("", index=df.index)).astype(str)
        out[f"{tf}_candle_up"] = (cdir == "up").to_numpy()
        out[f"{tf}_candle_down"] = (cdir == "down").to_numpy()
        trend = df.get(f"{tf}_trend_state", pd.Series("", index=df.index)).astype(str)
        out[f"{tf}_trend_up"] = (trend == "up").to_numpy()
        out[f"{tf}_trend_down"] = (trend == "down").to_numpy()
        out[f"{tf}_trend_mixed"] = (trend == "mixed").to_numpy()
        for th in [0.35, 0.45, 0.55, 0.65]:
            out[f"{tf}_body_ge_{th:.2f}"] = (pd.to_numeric(df[f"{tf}_body_ratio"], errors="coerce").fillna(0.0) >= th).to_numpy()
        for th in [0.35, 0.45, 0.55, 0.65, 0.75]:
            out[f"{tf}_rangeq_ge_{th:.2f}"] = (pd.to_numeric(df[f"{tf}_range_q"], errors="coerce").fillna(0.0) >= th).to_numpy()
        for th in [1.1, 1.3, 1.6]:
            out[f"{tf}_vol_ge_{th:.1f}"] = (pd.to_numeric(df[f"{tf}_volume_mult"], errors="coerce").fillna(0.0) >= th).to_numpy()
        pos = pd.to_numeric(df[f"{tf}_pos20"], errors="coerce").fillna(0.5)
        out[f"{tf}_pos_high"] = (pos >= 0.75).to_numpy()
        out[f"{tf}_pos_low"] = (pos <= 0.25).to_numpy()
    # Remove very rare or universal atoms per dataframe later during mining.
    return out


def build_atom_store(enriched: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    return {
        period: atom_masks(enriched[enriched["period_name"] == period].reset_index(drop=True))
        for period in ["gate_train_for_180d", "gate_train_for_365d", "validation_180d", "validation_365d"]
    }


def build_period_vals(enriched: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        period: enriched[enriched["period_name"] == period].reset_index(drop=True)
        for period in ["gate_train_for_180d", "gate_train_for_365d", "validation_180d", "validation_365d"]
    }


def mask_for_atoms(atom_store: dict[str, dict[str, np.ndarray]], period: str, atoms: list[str]) -> np.ndarray:
    masks = atom_store[period]
    if not atoms:
        first = next(iter(masks.values()))
        return np.ones(len(first), dtype=bool)
    out = masks[atoms[0]].copy()
    for atom in atoms[1:]:
        out &= masks[atom]
    return out


def cluster_stats(df: pd.DataFrame, mask: np.ndarray) -> dict[str, Any]:
    won = df["won"].astype(bool).to_numpy()
    n = int(mask.sum())
    wins = int((won & mask).sum())
    losses = n - wins
    return {
        "n": n,
        "wins": wins,
        "losses": losses,
        "winRatePct": round(100.0 * wins / n, 6) if n else 0.0,
        "loserMinusWinner": losses - wins,
    }


def mine_bad_clusters(enriched: pd.DataFrame, atoms: dict[str, dict[str, np.ndarray]], limit: int) -> list[dict[str, Any]]:
    train180 = enriched[enriched["period_name"] == "gate_train_for_180d"].reset_index(drop=True)
    train365 = enriched[enriched["period_name"] == "gate_train_for_365d"].reset_index(drop=True)
    atom_names = sorted(set(atoms["gate_train_for_180d"]).intersection(atoms["gate_train_for_365d"]))

    # Keep atoms that are neither too rare nor universal in both training windows.
    usable: list[str] = []
    for a in atom_names:
        n180 = int(atoms["gate_train_for_180d"][a].sum())
        n365 = int(atoms["gate_train_for_365d"][a].sum())
        if n180 < 40 or n365 < 40:
            continue
        if n180 > len(train180) * 0.98 and n365 > len(train365) * 0.98:
            continue
        usable.append(a)

    combos: list[tuple[str, ...]] = []
    combos.extend((a,) for a in usable)
    # Two- and three-atom combos are where most useful environment clusters live.
    # Restrict to deterministic top atoms by individual badness to keep runtime bounded.
    indiv = []
    for a in usable:
        s180 = cluster_stats(train180, mask_for_atoms(atoms, "gate_train_for_180d", [a]))
        s365 = cluster_stats(train365, mask_for_atoms(atoms, "gate_train_for_365d", [a]))
        score = min(s180["loserMinusWinner"], s365["loserMinusWinner"]) * 8 + min(s180["n"], s365["n"]) * 0.01
        indiv.append((score, a))
    indiv.sort(reverse=True)
    top_atoms = [a for _s, a in indiv[:70]]
    combos.extend(tuple(c) for c in combinations(top_atoms, 2))
    combos.extend(tuple(c) for c in combinations(top_atoms[:36], 3))

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for c in combos:
        key = "|".join(c)
        if key in seen:
            continue
        seen.add(key)
        m180 = mask_for_atoms(atoms, "gate_train_for_180d", list(c))
        m365 = mask_for_atoms(atoms, "gate_train_for_365d", list(c))
        s180 = cluster_stats(train180, m180)
        s365 = cluster_stats(train365, m365)
        if min(s180["n"], s365["n"]) < 30:
            continue
        min_edge = min(s180["loserMinusWinner"], s365["loserMinusWinner"])
        base180 = float(train180["won"].mean() * 100.0)
        base365 = float(train365["won"].mean() * 100.0)
        # Prefer clusters that are bad in both training windows and not too tiny.
        score = (
            min_edge * 12.0
            + max(0.0, base180 - s180["winRatePct"]) * 2.0
            + max(0.0, base365 - s365["winRatePct"]) * 2.0
            + min(s180["n"], s365["n"]) * 0.015
            - (0 if min(s180["n"], s365["n"]) >= 80 else 15)
        )
        rows.append({
            "cluster_id": stable_key({"atoms": c}),
            "atoms": list(c),
            "score": round(score, 6),
            "train180": s180,
            "train365": s365,
        })
    rows.sort(key=lambda r: (r["score"], min(r["train180"]["loserMinusWinner"], r["train365"]["loserMinusWinner"])), reverse=True)
    return rows[:limit]


def candidate_from_cluster(cluster: dict[str, Any], action: str, score_min: float | None = None) -> dict[str, Any]:
    p = {
        "family": "cluster_gate",
        "cluster_id": cluster["cluster_id"],
        "atoms": cluster["atoms"],
        "action": action,
    }
    if score_min is not None:
        p["shock_score_min"] = round(float(score_min), 4)
    return p


def generate_candidates(clusters: list[dict[str, Any]], max_candidates: int, shard_index: int = 0, shard_count: int = 1) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    top = clusters
    top_n = int(os.environ.get("TOP159_CLUSTER_TOP", "80"))
    if top_n > 0:
        top = clusters[:min(top_n, len(clusters))]
    expanded = os.environ.get("TOP159_CLUSTER_EXPANDED", "1").strip() != "0"
    deep = os.environ.get("TOP159_CLUSTER_DEEP", "0").strip() == "1"
    start_offset = int(os.environ.get("TOP159_CLUSTER_START_OFFSET", "0"))
    stop_at = start_offset + max_candidates
    dedup = os.environ.get("TOP159_CLUSTER_DEDUP", "1").strip() != "0"
    single_thresholds = [0.555, 0.56, 0.565, 0.57, 0.575, 0.58, 0.59, 0.60, 0.62, 0.64, 0.67, 0.70, 0.74] if expanded else [0.56, 0.57, 0.58, 0.59, 0.60, 0.62, 0.64]
    pair_top = int(os.environ.get("TOP159_CLUSTER_PAIR_TOP", "80" if expanded else "45"))
    triple_top = int(os.environ.get("TOP159_CLUSTER_TRIPLE_TOP", "70" if deep else ("42" if expanded else "0")))
    quad_top = int(os.environ.get("TOP159_CLUSTER_QUAD_TOP", "46" if deep else "0"))
    global_variants: list[list[str]] = [[]]
    if deep:
        global_variants.extend([
            ["dir_UP"],
            ["dir_DOWN"],
            ["score_lt_058"],
            ["score_lt_060"],
            ["dir_UP", "score_lt_060"],
            ["dir_DOWN", "score_lt_060"],
        ])
    seen: set[str] = set()
    total_seen = 0

    def maybe_add(raw: dict[str, Any]) -> bool:
        nonlocal total_seen
        sid = stable_key(raw)
        if dedup and sid in seen:
            return False
        if dedup:
            seen.add(sid)
        if total_seen >= stop_at:
            return True
        if total_seen >= start_offset:
            raw["candidate_id"] = sid
            if total_seen % max(1, shard_count) == shard_index:
                rows.append(raw)
        total_seen += 1
        return total_seen >= stop_at

    def with_global(raw: dict[str, Any], global_atoms: list[str]) -> dict[str, Any]:
        if not global_atoms:
            return raw
        out = dict(raw)
        out["global_atoms"] = list(global_atoms)
        return out

    for cl in top:
        for gv in global_variants:
            if maybe_add(with_global(candidate_from_cluster(cl, "hard_block"), gv)):
                return rows, total_seen
            for ss in single_thresholds:
                if maybe_add(with_global(candidate_from_cluster(cl, "raise_score", ss), gv)):
                    return rows, total_seen
    # OR two bad clusters. Keep simple hard block / soft threshold variants.
    for a, b in combinations(top[:pair_top], 2):
        atoms_any = [a["atoms"], b["atoms"]]
        for min_hits in ([1, 2] if deep else [1]):
            base = {
                "family": "cluster_gate",
                "cluster_id": stable_key({"any_clusters": [a["cluster_id"], b["cluster_id"]], "min_hits": min_hits}),
                "any_clusters": atoms_any,
                "min_cluster_hits": min_hits,
                "action": "hard_block",
            }
            for gv in global_variants:
                if maybe_add(with_global(dict(base), gv)):
                    return rows, total_seen
                for ss in ([0.555, 0.56, 0.57, 0.58, 0.59, 0.60, 0.62, 0.65, 0.68, 0.72] if expanded else [0.58, 0.60, 0.62]):
                    q = dict(base)
                    q["action"] = "raise_score"
                    q["shock_score_min"] = ss
                    if maybe_add(with_global(q, gv)):
                        return rows, total_seen
    # OR three bad clusters. This is the targeted version of "combine several
    # small weak spots"; validation still decides, so it is not accepted just
    # because it looks good in the mining window.
    if triple_top > 0:
        for a, b, c in combinations(top[:triple_top], 3):
            atoms_any = [a["atoms"], b["atoms"], c["atoms"]]
            for min_hits in ([1, 2, 3] if deep else [1]):
                base = {
                    "family": "cluster_gate",
                    "cluster_id": stable_key({"any_clusters": [a["cluster_id"], b["cluster_id"], c["cluster_id"]], "min_hits": min_hits}),
                    "any_clusters": atoms_any,
                    "min_cluster_hits": min_hits,
                    "action": "hard_block",
                }
                for gv in global_variants:
                    if maybe_add(with_global(dict(base), gv)):
                        return rows, total_seen
                    for ss in ([0.56, 0.58, 0.60, 0.62, 0.65, 0.68] if deep else [0.58, 0.60, 0.62, 0.65]):
                        q = dict(base)
                        q["action"] = "raise_score"
                        q["shock_score_min"] = ss
                        if maybe_add(with_global(q, gv)):
                            return rows, total_seen
    if quad_top > 0:
        for a, b, c, d in combinations(top[:quad_top], 4):
            atoms_any = [a["atoms"], b["atoms"], c["atoms"], d["atoms"]]
            for min_hits in [1, 2, 3]:
                base = {
                    "family": "cluster_gate",
                    "cluster_id": stable_key({"any_clusters": [a["cluster_id"], b["cluster_id"], c["cluster_id"], d["cluster_id"]], "min_hits": min_hits}),
                    "any_clusters": atoms_any,
                    "min_cluster_hits": min_hits,
                    "action": "hard_block",
                }
                for gv in global_variants:
                    if maybe_add(with_global(dict(base), gv)):
                        return rows, total_seen
                    for ss in [0.58, 0.60, 0.62, 0.65, 0.68]:
                        q = dict(base)
                        q["action"] = "raise_score"
                        q["shock_score_min"] = ss
                        if maybe_add(with_global(q, gv)):
                            return rows, total_seen
    return rows, total_seen


def condition_for_candidate(atom_store: dict[str, dict[str, np.ndarray]], period: str, params: dict[str, Any]) -> np.ndarray:
    if "any_clusters" in params:
        masks = [mask_for_atoms(atom_store, period, list(atoms)) for atoms in params["any_clusters"]]
        min_hits = int(params.get("min_cluster_hits", 1))
        if min_hits <= 1:
            out = masks[0].copy()
            for m in masks[1:]:
                out |= m
        else:
            votes = np.zeros_like(masks[0], dtype=np.int16)
            for m in masks:
                votes += m.astype(np.int16)
            out = votes >= min_hits
        for atom in params.get("global_atoms", []) or []:
            out &= atom_store[period][atom]
        return out
    out = mask_for_atoms(atom_store, period, list(params.get("atoms", [])))
    for atom in params.get("global_atoms", []) or []:
        out &= atom_store[period][atom]
    return out


def fast_metric(val: pd.DataFrame, keep: np.ndarray, cond: np.ndarray, params: dict[str, Any], window: str) -> dict[str, Any]:
    won = val["won"].astype(bool).to_numpy()
    keep = keep.astype(bool)
    cond = cond.astype(bool)
    trades = int(keep.sum())
    wins = int((won & keep).sum())
    losses = trades - wins
    blocked = ~keep
    blocked_winners = int((won & blocked).sum())
    blocked_losers = int((~won & blocked).sum())
    cond_count = int(cond.sum())
    cond_kept = cond & keep
    return {
        "window": window,
        "params": params,
        "trades": trades,
        "base_trades": int(len(val)),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(100.0 * wins / trades, 6) if trades else 0.0,
        "base_win_rate_pct": round(100.0 * float(won.mean()), 6) if len(won) else 0.0,
        "retention_rate": round(100.0 * trades / max(1, len(val)), 6),
        "blocked_trades": int(blocked.sum()),
        "blocked_winners": blocked_winners,
        "blocked_losers": blocked_losers,
        "blocked_loser_minus_winner": blocked_losers - blocked_winners,
        "condition_base_trades": cond_count,
        "condition_base_win_rate_pct": round(100.0 * float(won[cond].mean()), 6) if cond_count else 0.0,
        "condition_kept_trades": int(cond_kept.sum()),
        "condition_kept_win_rate_pct": round(100.0 * float(won[cond_kept].mean()), 6) if int(cond_kept.sum()) else 0.0,
    }


def evaluate_candidate(enriched: pd.DataFrame, atom_store: dict[str, dict[str, np.ndarray]], params: dict[str, Any]) -> dict[str, Any]:
    out = {"candidate_id": params["candidate_id"], "params": params}
    strict = True
    period_vals = G_PERIOD_VALS or build_period_vals(enriched)
    for window, period, min_trades in [("180d", "validation_180d", 100), ("365d", "validation_365d", 200)]:
        val = period_vals[period]
        cond = condition_for_candidate(atom_store, period, params)
        if params["action"] == "hard_block":
            keep = ~cond
        else:
            keep = (~cond) | (pd.to_numeric(val["score15"], errors="coerce").fillna(0.0).to_numpy() >= float(params["shock_score_min"]))
        row = fast_metric(val, keep, cond, params, window)
        out[f"w{window[:3]}"] = row
        if row["trades"] < min_trades:
            strict = False
        if window == "365d" and row["retention_rate"] < 45.0:
            strict = False
        if row["blocked_losers"] <= row["blocked_winners"]:
            strict = False
    out["strict_pass"] = strict
    return out


def init_worker() -> None:
    global G_ENRICHED, G_ATOMS, G_PERIOD_VALS
    G_ENRICHED, _truth = ext.load_or_build_enriched()
    G_ATOMS = build_atom_store(G_ENRICHED)
    G_PERIOD_VALS = build_period_vals(G_ENRICHED)


def evaluate_chunk(params_chunk: list[dict[str, Any]]) -> list[dict[str, Any]]:
    global G_ENRICHED, G_ATOMS
    if G_ENRICHED is None or G_ATOMS is None:
        init_worker()
    assert G_ENRICHED is not None and G_ATOMS is not None
    rows = []
    for p in params_chunk:
        try:
            rows.append(evaluate_candidate(G_ENRICHED, G_ATOMS, p))
        except Exception as exc:
            rows.append({"candidate_id": p.get("candidate_id") or stable_key(p), "params": p, "error": repr(exc)})
    return rows


def row_score(row: dict[str, Any]) -> tuple:
    w180 = row.get("w180") or {}
    w365 = row.get("w365") or {}
    edge180 = int(w180.get("blocked_losers", 0)) - int(w180.get("blocked_winners", 0))
    edge365 = int(w365.get("blocked_losers", 0)) - int(w365.get("blocked_winners", 0))
    min_edge = min(edge180, edge365)
    min_ret = min(safe_float(w180.get("retention_rate")), safe_float(w365.get("retention_rate")))
    min_trades = min(int(w180.get("trades", 0)), int(w365.get("trades", 0)))
    wr_sum = safe_float(w180.get("win_rate_pct")) + safe_float(w365.get("win_rate_pct"))
    cond_n = min(int(w180.get("condition_base_trades", 0)), int(w365.get("condition_base_trades", 0)))
    return (int(bool(row.get("strict_pass"))), min_edge, edge180 + edge365, wr_sum, min_ret, cond_n, min_trades)


def compact(row: dict[str, Any]) -> dict[str, Any]:
    return {k: row.get(k) for k in ["candidate_id", "strict_pass", "params", "w180", "w365", "pressure180", "pressure365", "fak2_180", "fak2_365", "error"] if k in row}


def chunked(seq: list[dict[str, Any]], size: int):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


def append_candidates(rows: list[dict[str, Any]]) -> None:
    OUT_CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CANDIDATES.open("a", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(compact(r), ensure_ascii=False, sort_keys=True, default=str) + "\n")


def rewrite_candidates(rows: list[dict[str, Any]]) -> None:
    OUT_CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CANDIDATES.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(compact(r), ensure_ascii=False, sort_keys=True, default=str) + "\n")


def load_done_ids() -> set[str]:
    done = set()
    if OUT_CANDIDATES.exists():
        with OUT_CANDIDATES.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                    if obj.get("candidate_id"):
                        done.add(obj["candidate_id"])
                except Exception:
                    pass
    return done


def leaderboard(rows: list[dict[str, Any]], limit: int = 100) -> list[dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for r in rows:
        if r.get("error"):
            continue
        cid = r.get("candidate_id")
        if not cid:
            continue
        if cid not in best or row_score(r) > row_score(best[cid]):
            best[cid] = r
    vals = list(best.values())
    vals.sort(key=row_score, reverse=True)
    return [compact(v) for v in vals[:limit]]


def write_checkpoint(started: datetime, deadline: datetime, planned: int, done: int, rows: list[dict[str, Any]], status: str) -> None:
    write_json(OUT_CHECKPOINT, {
        "status": status,
        "started_at_bj": iso(started),
        "updated_at_bj": iso(bj_now()),
        "deadline_bj": iso(deadline),
        "planned_candidates": planned,
        "completed_candidates": done,
        "candidate_file": str(OUT_CANDIDATES),
        "top": leaderboard(rows, 30),
        "live_config_mutated": False,
        "notes": "Research-only targeted cluster shock search; live top159 untouched.",
    })


def summarize_final(enriched: pd.DataFrame, atom_store: dict[str, dict[str, np.ndarray]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    lb = leaderboard(rows, 50)
    best = lb[0] if lb else None
    validation_rows: list[dict[str, Any]] = []
    cluster_named = [("current_live_06173_cluster_gate", CURRENT_061_PARAMS)]
    if best:
        cluster_named.append(("cluster_targeted_best", best["params"]))
    blocked_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for label, p in cluster_named:
        for window, period in [("180d", "validation_180d"), ("365d", "validation_365d")]:
            val = enriched[enriched["period_name"] == period].reset_index(drop=True)
            cond = condition_for_candidate(atom_store, period, p)
            if p["action"] == "hard_block":
                keep = ~cond
            else:
                keep = (~cond) | (pd.to_numeric(val["score15"], errors="coerce").fillna(0.0).to_numpy() >= float(p["shock_score_min"]))
            selected = val[keep].copy().sort_values("dt").reset_index(drop=True)
            validation_rows.append(ext.summarize_compound(selected, label, window, "full_fill_buy_0.50", 0.50))
            validation_rows.append(ext.summarize_compound(selected, label, window, "fak_pressure_buy_0.52", 0.52))
            validation_rows.append(ext.summarize_toxic_mc(selected, label, window))
            won = val["won"].astype(bool).to_numpy()
            blocked = ~keep
            blocked_lookup[(label, window)] = {
                "拦截赢家": int((won & blocked).sum()),
                "拦截输单": int((~won & blocked).sum()),
                "保留率": round(100.0 * int(keep.sum()) / max(1, len(keep)), 6),
            }
    # Compare with baseline/current 2c736 using existing final validator.
    compare_params = [
        ("current_new_archive_top159", {"family": "baseline"}),
        ("prior_2c736", {
            "family": "model_gate",
            "action": "model_gate",
            "rule_mode": "any_terminal",
            "body_min": 0.5,
            "range_q_min": 0.4,
            "volume_mult_min": 1.1,
            "model_engine": "logistic",
            "min_gate_win_prob": 0.53,
            "model_hyper": {"engine": "logistic", "C": 0.08},
        }),
    ]
    prior_rows, _maps = ext.final_validation_rows(enriched, compare_params)
    archived_rows, archived_audit = archived_real_rows_cluster(enriched, atom_store, cluster_named)
    return {
        "updated_at_bj": iso(bj_now()),
        "recommended_candidate": best,
        "leaderboard": lb,
        "validationRows": prior_rows + validation_rows,
        "archivedRealRows": archived_rows,
        "archivedRealAudit": archived_audit,
        "blockedLookup": {f"{k[0]}::{k[1]}": v for k, v in blocked_lookup.items()},
        "live_config_mutated": False,
    }


def archived_real_rows_cluster(
    enriched: pd.DataFrame,
    atom_store: dict[str, dict[str, np.ndarray]],
    named_params: list[tuple[str, dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay cluster-gate keep masks on archived real ETH 15m trades.

    The generic helper in ``run_top159_shock_filter_extreme_search`` cannot
    evaluate ``family=cluster_gate`` candidates because those are built from
    mined atom masks instead of the older body/range shock params. This helper
    keeps the archive comparison on the same exact mask logic used by the 061
    live gate and this search.
    """
    fill_rows, scan_audit = ext.archive.load_live_fill_rows()
    all_eth = ext.archive.aggregate_logical(fill_rows, "全部归档ETH15m live去重")
    strict_mask = fill_rows["sourcePath"].str.contains(
        "slot01|forcedtrade|base_forcedtrade|__live_eth|logs_v5_exp10_bp0480__live_eth",
        case=False,
        regex=True,
    )
    strict = (
        ext.archive.aggregate_logical(fill_rows[strict_mask].copy(), "严格旧slot1/ETH相关live去重")
        if strict_mask.any()
        else all_eth.iloc[0:0]
    )
    rows: list[dict[str, Any]] = []
    val = enriched[enriched["period_name"] == "validation_180d"].reset_index(drop=True)
    for label, params in named_params:
        cond = condition_for_candidate(atom_store, "validation_180d", params)
        if params.get("action") == "hard_block":
            keep = ~cond
        else:
            keep = (~cond) | (
                pd.to_numeric(val["score15"], errors="coerce").fillna(0.0).to_numpy()
                >= float(params.get("shock_score_min", 1.0))
            )
        pred = val.assign(keep=keep.astype(bool), cond=cond.astype(bool))
        for scope, old in [("严格旧slot1/ETH相关live去重", strict), ("全部归档ETH15m live去重", all_eth)]:
            if old.empty:
                continue
            merged = old.merge(
                pred[["dt", "pred_up15", "score15", "won", "keep", "cond"]],
                left_on="marketStart",
                right_on="dt",
                how="left",
            )
            s = merged[merged["keep"].fillna(False).astype(bool)].copy().sort_values("marketStart")
            if s.empty:
                rows.append(
                    {
                        "scope": scope,
                        "name": label,
                        "oldRealMarkets": int(len(old)),
                        "selectedTrades": 0,
                        "skippedTrades": int(len(old)),
                    }
                )
                continue
            model_won = s["pred_up15"].astype(bool).to_numpy() == s["actualUp"].astype(bool).to_numpy()
            one = np.where(model_won, 1.0, -1.0)
            fak52 = np.where(model_won, (1.0 / 0.52) - 1.0, -1.0)
            old_direction_up = s["direction"].astype(str).str.upper().to_numpy() == "UP"
            s["sameDirectionAsOld"] = s["pred_up15"].astype(bool).to_numpy() == old_direction_up
            same = s[s["sameDirectionAsOld"]].copy()
            same_pnl = same["pnl"].astype(float).tolist()
            skipped = old[~old["marketStart"].isin(s["marketStart"])]
            rows.append(
                {
                    "scope": scope,
                    "name": label,
                    "oldRealMarkets": int(len(old)),
                    "selectedTrades": int(len(s)),
                    "skippedTrades": int(len(old) - len(s)),
                    "skippedOldWinners": int(skipped["won"].sum()) if len(skipped) else 0,
                    "skippedOldLosers": int((~skipped["won"]).sum()) if len(skipped) else 0,
                    "wins": int(model_won.sum()),
                    "losses": int(len(model_won) - int(model_won.sum())),
                    "winRatePct": round(100.0 * float(model_won.mean()), 6),
                    "oneUnitPnl": round(float(one.sum()), 6),
                    "oneUnitMaxDrawdown": ext.max_drawdown_from_pnl(one.tolist()),
                    "fak52Pnl": round(float(fak52.sum()), 6),
                    "fak52MaxDrawdown": ext.max_drawdown_from_pnl(fak52.tolist()),
                    "sameDirectionExecutableTrades": int(len(same)),
                    "sameDirectionActualPnlUsd": round(float(sum(same_pnl)), 6) if same_pnl else 0.0,
                    "sameDirectionActualMaxDrawdownUsd": ext.max_drawdown_from_pnl(same_pnl),
                    "shockConditionTrades": int(s["cond"].fillna(False).astype(bool).sum()),
                    "avgScore": round(float(s["score15"].mean()), 6),
                    "setHash": stable_key(s[["marketSlug", "pred_up15", "actualUp"]].to_dict("records")),
                }
            )
    audit = {
        "scanAudit": scan_audit,
        "strictTrades": int(len(strict)),
        "allArchivedTrades": int(len(all_eth)),
        "archivedRangeUtc": [str(all_eth["marketStart"].min()), str(all_eth["marketStart"].max())]
        if len(all_eth)
        else [None, None],
        "truthLimit": "archive rows are pure prediction replay on old real markets; they are not proof of new-model live execution/fill quality",
    }
    return rows, audit


def row_money(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("method") == "slot1_toxic_monte_carlo":
        return {
            "盈亏": row.get("compoundPnlP50"),
            "期末资金": row.get("endingBankrollP50"),
            "最大回撤": row.get("maxDrawdownP50"),
            "毒性P5": row.get("endingBankrollP5"),
            "毒性P50": row.get("endingBankrollP50"),
            "毒性P95": row.get("endingBankrollP95"),
        }
    return {
        "盈亏": row.get("compoundPnl"),
        "期末资金": row.get("endingBankroll"),
        "最大回撤": row.get("maxDrawdownUsd"),
        "毒性P5": None,
        "毒性P50": None,
        "毒性P95": None,
    }


def fmt_num(x: Any, nd: int = 2) -> str:
    try:
        if x is None:
            return "-"
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return "-"
        return f"{v:,.{nd}f}"
    except Exception:
        return "-"


def compact_profit_table(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rec = payload.get("recommended_candidate") or {}
    blocked_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for key, value in (payload.get("blockedLookup") or {}).items():
        try:
            name, window = str(key).split("::", 1)
            blocked_lookup[(name, window)] = value
        except Exception:
            continue
    for label in ["current_new_archive_top159", "prior_2c736"]:
        # Baseline and prior rows are generated by ext; only cluster best has
        # explicit blocked counts in this script.
        pass
    if rec:
        for window_key, window in [("w180", "180d"), ("w365", "365d")]:
            w = rec.get(window_key) or {}
            blocked_lookup[("cluster_targeted_best", window)] = {
                "拦截赢家": w.get("blocked_winners"),
                "拦截输单": w.get("blocked_losers"),
                "保留率": w.get("retention_rate"),
            }
    rows: list[dict[str, Any]] = []
    for row in payload.get("validationRows", []):
        money = row_money(row)
        key = (row.get("name"), row.get("window"))
        blocked = blocked_lookup.get(key, {})
        rows.append({
            "配置": row.get("name"),
            "窗口": row.get("window"),
            "口径": row.get("method"),
            "交易数": row.get("trades"),
            "胜负": f"{row.get('wins')}/{row.get('losses')}",
            "胜率": row.get("winRatePct"),
            "盈亏": money["盈亏"],
            "期末资金": money["期末资金"],
            "最大回撤": money["最大回撤"],
            "毒性P5": money["毒性P5"],
            "毒性P50": money["毒性P50"],
            "毒性P95": money["毒性P95"],
            "拦截赢家": blocked.get("拦截赢家", "-"),
            "拦截输单": blocked.get("拦截输单", "-"),
            "保留率": blocked.get("保留率", "-"),
        })
    for row in payload.get("archivedRealRows", []):
        scope = row.get("scope") or "历史归档"
        rows.append({
            "配置": row.get("name"),
            "窗口": scope,
            "口径": "archive_fak52_pure_prediction",
            "交易数": row.get("selectedTrades"),
            "胜负": f"{row.get('wins')}/{row.get('losses')}",
            "胜率": row.get("winRatePct"),
            "盈亏": row.get("fak52Pnl"),
            "期末资金": None,
            "最大回撤": row.get("fak52MaxDrawdown"),
            "毒性P5": None,
            "毒性P50": None,
            "毒性P95": None,
            "拦截赢家": row.get("skippedOldWinners", "-"),
            "拦截输单": row.get("skippedOldLosers", "-"),
            "保留率": round(100.0 * float(row.get("selectedTrades", 0)) / max(1.0, float(row.get("oldRealMarkets", 0))), 6)
            if row.get("oldRealMarkets")
            else "-",
        })
        rows.append({
            "配置": row.get("name"),
            "窗口": scope,
            "口径": "archive_old_same_direction_actual_pnl",
            "交易数": row.get("sameDirectionExecutableTrades"),
            "胜负": "-",
            "胜率": None,
            "盈亏": row.get("sameDirectionActualPnlUsd"),
            "期末资金": None,
            "最大回撤": row.get("sameDirectionActualMaxDrawdownUsd"),
            "毒性P5": None,
            "毒性P50": None,
            "毒性P95": None,
            "拦截赢家": row.get("skippedOldWinners", "-"),
            "拦截输单": row.get("skippedOldLosers", "-"),
            "保留率": round(100.0 * float(row.get("selectedTrades", 0)) / max(1.0, float(row.get("oldRealMarkets", 0))), 6)
            if row.get("oldRealMarkets")
            else "-",
        })
    return rows


def render_profit_table(payload: dict[str, Any]) -> str:
    rows = compact_profit_table(payload)
    headers = ["配置", "窗口", "口径", "交易数", "胜负", "胜率", "盈亏", "期末资金", "最大回撤", "毒性P5", "毒性P50", "毒性P95", "拦截赢家", "拦截输单", "保留率"]
    out = [
        "# top159 冲击门盈亏对比表",
        "",
        f"更新时间（北京时间）：{payload.get('updated_at_bj')}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        vals = []
        for h in headers:
            v = r.get(h)
            if h in {"胜率", "保留率"} and isinstance(v, (int, float)):
                vals.append(fmt_num(v, 2) + "%")
            elif h in {"盈亏", "期末资金", "最大回撤", "毒性P5", "毒性P50", "毒性P95"}:
                vals.append(fmt_num(v, 2))
            else:
                vals.append(str(v))
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out) + "\n"


def render_verdict(payload: dict[str, Any]) -> str:
    best = payload.get("recommended_candidate") or {}
    lines = [
        "# top159 错单簇定向搜索结论",
        "",
        f"更新时间（北京时间）：{payload.get('updated_at_bj')}",
        "",
    ]
    if not best:
        lines.append("结论：没有生成有效候选，live 不改。")
    else:
        w180 = best.get("w180") or {}
        w365 = best.get("w365") or {}
        min_edge = min(int(w180.get("blocked_losers", 0)) - int(w180.get("blocked_winners", 0)), int(w365.get("blocked_losers", 0)) - int(w365.get("blocked_winners", 0)))
        if min_edge >= 25:
            decision = "候选达到影子验证边际，可进入24小时影子，不直接改真钱。"
        else:
            decision = "候选边际仍不足25，只列观察，不建议切真钱。"
        lines += [
            f"结论：{decision}",
            "",
            "## 最强候选",
            "```json",
            json.dumps(best, ensure_ascii=False, indent=2, default=str),
            "```",
            "",
            f"完整盈亏表：{OUT_TABLE}",
        ]
    return "\n".join(lines) + "\n"


def parse_deadline(s: str | None) -> datetime:
    if s:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=BJ)
        return dt.astimezone(BJ)
    return bj_now() + timedelta(hours=5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline-bj", default=os.environ.get("TOP159_CLUSTER_DEADLINE_BJ"))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("TOP159_CLUSTER_WORKERS", "4")))
    ap.add_argument("--chunk-size", type=int, default=int(os.environ.get("TOP159_CLUSTER_CHUNK_SIZE", "320")))
    ap.add_argument("--max-candidates", type=int, default=int(os.environ.get("TOP159_CLUSTER_MAX_CANDIDATES", "20000")))
    ap.add_argument("--cluster-limit", type=int, default=int(os.environ.get("TOP159_CLUSTER_LIMIT", "80")))
    ap.add_argument("--shard-index", type=int, default=int(os.environ.get("TOP159_CLUSTER_SHARD_INDEX", "0")))
    ap.add_argument("--shard-count", type=int, default=int(os.environ.get("TOP159_CLUSTER_SHARD_COUNT", "1")))
    ap.add_argument("--output-suffix", default=os.environ.get("TOP159_CLUSTER_OUTPUT_SUFFIX", ""))
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    if args.shard_count < 1:
        raise SystemExit("--shard-count must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise SystemExit("--shard-index must be in [0, shard_count)")
    apply_output_suffix(args.output_suffix)

    if args.fresh:
        for p in [OUT_CANDIDATES, OUT_CHECKPOINT, OUT_LEADERBOARD, OUT_COMPARE, OUT_VERDICT, OUT_CLUSTERS, OUT_AUDIT, OUT_LOG]:
            if p.exists():
                p.unlink()

    started = bj_now()
    deadline = parse_deadline(args.deadline_bj)
    log(f"cluster targeted search started; deadline={iso(deadline)} workers={args.workers}")
    enriched, truth = ext.load_or_build_enriched()
    atoms = build_atom_store(enriched)
    audit = ext.bug_audit(enriched)
    audit.update({
        "created_at_bj": iso(bj_now()),
        "feature_alignment_version": truth.get("shockFeatureAlignmentVersion"),
        "research_only": True,
        "live_configs_touched": False,
        "search_type": "bad_cluster_targeted",
    })
    write_json(OUT_AUDIT, audit)
    clusters = mine_bad_clusters(enriched, atoms, args.cluster_limit)
    write_json(OUT_CLUSTERS, {"generated_at_bj": iso(bj_now()), "clusters": clusters, "cluster_count": len(clusters)})
    candidates, total_generated = generate_candidates(clusters, args.max_candidates, args.shard_index, args.shard_count)
    if args.shard_count > 1:
        log(f"shard={args.shard_index}/{args.shard_count} generated_total={total_generated} shard_candidates={len(candidates)}")
    done = load_done_ids()
    candidates = [c for c in candidates if c["candidate_id"] not in done]
    all_rows: list[dict[str, Any]] = []
    if OUT_CANDIDATES.exists():
        with OUT_CANDIDATES.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    all_rows.append(json.loads(line))
                except Exception:
                    pass
    planned = len(candidates) + len(done)
    persist_mode = os.environ.get("TOP159_CLUSTER_PERSIST_MODE", "all").strip().lower()
    keep_top = int(os.environ.get("TOP159_CLUSTER_KEEP_TOP", "5000"))
    if persist_mode not in {"all", "top"}:
        persist_mode = "all"
    completed_count = len(done)

    def consume_rows(rows: list[dict[str, Any]]) -> None:
        nonlocal all_rows, completed_count
        completed_count += len(rows)
        all_rows.extend(rows)
        if keep_top > 0 and len(all_rows) > keep_top * 2:
            all_rows = leaderboard(all_rows, keep_top)
        if persist_mode == "all":
            append_candidates(rows)

    write_checkpoint(started, deadline, planned, completed_count, all_rows, "running")
    last_cp = time.time()
    submitted = 0
    chunks = list(chunked(candidates, args.chunk_size))
    executor_kind = os.environ.get("TOP159_CLUSTER_EXECUTOR", "thread").strip().lower()
    # Process workers duplicate the full enriched dataframe on macOS and can be
    # killed silently under memory pressure. Threads share the in-memory cache
    # and are enough here because the hot path is vectorized NumPy/Pandas.
    if executor_kind not in {"thread", "process"}:
        executor_kind = "thread"
    log(f"executor={executor_kind} chunks={len(chunks)} candidates={len(candidates)}")
    global G_ENRICHED, G_ATOMS, G_PERIOD_VALS
    initializer = None
    if executor_kind == "thread":
        G_ENRICHED, G_ATOMS = enriched, atoms
        G_PERIOD_VALS = build_period_vals(enriched)
        executor_cls = ThreadPoolExecutor
    else:
        executor_cls = ProcessPoolExecutor
        initializer = init_worker
    with executor_cls(max_workers=args.workers, initializer=initializer) as pool:
        futures = []
        for chunk in chunks:
            if bj_now() >= deadline - timedelta(minutes=15):
                break
            futures.append(pool.submit(evaluate_chunk, chunk))
            submitted += len(chunk)
            if len(futures) >= args.workers * 3:
                for fut in as_completed(futures[:args.workers]):
                    try:
                        rows = fut.result()
                    except Exception as exc:
                        log(f"worker_error={repr(exc)}")
                        rows = []
                    consume_rows(rows)
                futures = futures[args.workers:]
                if time.time() - last_cp >= 15 * 60:
                    write_checkpoint(started, deadline, planned, completed_count, all_rows, "running")
                    log(f"checkpoint completed={completed_count} submitted={submitted} top={leaderboard(all_rows, 1)[:1]}")
                    last_cp = time.time()
        for fut in as_completed(futures):
            try:
                rows = fut.result()
            except Exception as exc:
                log(f"worker_error={repr(exc)}")
                rows = []
            consume_rows(rows)
            if time.time() - last_cp >= 15 * 60:
                write_checkpoint(started, deadline, planned, completed_count, all_rows, "running")
                log(f"checkpoint completed={completed_count} submitted={submitted} top={leaderboard(all_rows, 1)[:1]}")
                last_cp = time.time()
            if bj_now() >= deadline - timedelta(minutes=8):
                break
    if persist_mode == "top":
        rewrite_candidates(leaderboard(all_rows, keep_top))
    write_json(OUT_LEADERBOARD, {
        "updated_at_bj": iso(bj_now()),
        "rows": leaderboard(all_rows, 500),
        "live_config_mutated": False,
    })
    write_checkpoint(started, deadline, planned, completed_count, all_rows, "finalizing")
    compare = summarize_final(enriched, atoms, all_rows)
    compare["audit"] = {"bugAudit": audit, "clusterCount": len(clusters), "plannedCandidates": planned, "completedCandidates": completed_count, "persistMode": persist_mode, "keepTop": keep_top}
    write_json(OUT_COMPARE, compare)
    OUT_TABLE.write_text(render_profit_table(compare), encoding="utf-8")
    OUT_VERDICT.write_text(render_verdict(compare), encoding="utf-8")
    write_checkpoint(started, deadline, planned, completed_count, all_rows, "complete")
    log(f"complete compare={OUT_COMPARE} verdict={OUT_VERDICT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
