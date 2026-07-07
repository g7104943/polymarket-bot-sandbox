#!/usr/bin/env python3
from __future__ import annotations

"""Full-field long-running hyperopt for the current 061 cluster gate.

Research only. This script does not mutate live configs, live model profiles,
ledgers, orders, claim state, or monitors.

Locked evaluation:
  - closed_candle_available_at_v2 alignment inherited from audited cluster data
  - 850U initial bankroll
  - 1% current bankroll per trade
  - primary buy price 0.55, full-fill assumption
"""

import concurrent.futures as cf
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(key, os.environ.get("TOP159_061_FULLFIELD_THREADS", "1"))
os.environ.setdefault("TOP159_SHOCK_EXTREME_THREADS", os.environ.get("TOP159_061_FULLFIELD_THREADS", "1"))

import numpy as np
import pandas as pd

ROOT = Path("/Users/mac/polyfun")
REPORTS = ROOT / "reports"
PRECISION_SCRIPT = ROOT / "polyfun-next" / "scripts" / "run_top159_061_precision_hyperopt.py"

START_BANKROLL = 850.0
STAKE_PCT = 0.01
PRIMARY_BUY_PRICE = 0.55
BUY_PRICES = [0.55, 0.60]
RNG_SEED = 20260507
WORKERS = int(os.environ.get("TOP159_061_FULLFIELD_WORKERS", "4"))
BATCH_SIZE = int(os.environ.get("TOP159_061_FULLFIELD_BATCH_SIZE", "800"))
MAX_CANDIDATES = int(os.environ.get("TOP159_061_FULLFIELD_MAX_CANDIDATES", "350000"))
MAX_SECONDS_ENV = os.environ.get("TOP159_061_FULLFIELD_MAX_SECONDS")
RESET = os.environ.get("TOP159_061_FULLFIELD_RESET", "0").strip() == "1"

OUT_AUDIT_MD = REPORTS / "top159_061_fullfield_hyperopt_bug_audit_latest.md"
OUT_AUDIT_JSON = REPORTS / "top159_061_fullfield_hyperopt_bug_audit_latest.json"
OUT_RESULTS = REPORTS / "top159_061_fullfield_hyperopt_results_latest.jsonl"
OUT_LEADERBOARD_MD = REPORTS / "top159_061_fullfield_hyperopt_leaderboard_latest.md"
OUT_LEADERBOARD_JSON = REPORTS / "top159_061_fullfield_hyperopt_leaderboard_latest.json"
OUT_COMPARE_MD = REPORTS / "top159_061_fullfield_hyperopt_061_compare_latest.md"
OUT_COMPARE_JSON = REPORTS / "top159_061_fullfield_hyperopt_061_compare_latest.json"
OUT_VERDICT_MD = REPORTS / "top159_061_fullfield_hyperopt_unique_verdict_latest.md"
OUT_VERDICT_JSON = REPORTS / "top159_061_fullfield_hyperopt_unique_verdict_latest.json"
OUT_CHECKPOINT = REPORTS / "top159_061_fullfield_hyperopt_checkpoint_latest.json"
OUT_PID = REPORTS / "top159_061_fullfield_hyperopt.pid"

BASELINE_061 = {
    "180d": {"trades": 3942, "wins": 2324, "losses": 1618, "winRatePct": 58.954845, "compoundPnl": 11494.591625, "endingBankroll": 12344.591625, "maxDrawdownUsd": 1481.713373, "returnDrawdownRatio": 7.757635, "monthlyPositiveRatio": 1.0, "setHash": "b35313c05e5b66d2"},
    "365d": {"trades": 9152, "wins": 5246, "losses": 3906, "winRatePct": 57.320804, "compoundPnl": 27033.917055, "endingBankroll": 27883.917055, "maxDrawdownUsd": 3499.248929, "returnDrawdownRatio": 7.725634, "monthlyPositiveRatio": 0.846154, "setHash": "ced1cf82642d8f0d"},
}

BJ = timezone(timedelta(hours=8))
_G_PERIOD_VALS: dict[str, pd.DataFrame] | None = None
_G_ATOMS: dict[str, dict[str, np.ndarray]] | None = None


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


precision = load_module("top159_precision_base_for_fullfield", PRECISION_SCRIPT)
base = precision.base
archive = precision.archive


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def bj_now() -> str:
    return pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S CST")


def seconds_until_next_8am_bj() -> int:
    now = datetime.now(BJ)
    target = now.replace(hour=8, minute=0, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return max(60, int((target - now).total_seconds()))


def max_seconds() -> int:
    if MAX_SECONDS_ENV:
        return int(MAX_SECONDS_ENV)
    return seconds_until_next_8am_bj()


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:16]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def max_drawdown(equity: np.ndarray) -> tuple[float, float]:
    if equity.size == 0:
        return 0.0, 0.0
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    i = int(np.argmax(dd))
    mx = float(dd[i])
    denom = float(peak[i]) if peak[i] > 1e-12 else 0.0
    return mx, mx / denom if denom else 0.0


def monthly_positive_ratio(equity: np.ndarray, dt: pd.Series) -> float:
    if equity.size == 0:
        return 0.0
    f = pd.DataFrame({"dt": pd.to_datetime(dt, utc=True, errors="coerce"), "equity": equity}).dropna().sort_values("dt")
    if f.empty:
        return 0.0
    f["month"] = f["dt"].dt.to_period("M").astype(str)
    prev = START_BANKROLL
    vals = []
    for _, g in f.groupby("month", sort=True):
        end = float(g["equity"].iloc[-1])
        vals.append(end - prev)
        prev = end
    return round(sum(v > 0 for v in vals) / len(vals), 6) if vals else 0.0


def curve_metrics(rows: pd.DataFrame, name: str, window: str, buy_price: float) -> dict[str, Any]:
    sel = rows.sort_values("dt").reset_index(drop=True).copy()
    won = sel["won"].astype(bool).to_numpy()
    eq = START_BANKROLL
    curve = np.empty(len(sel), dtype=float)
    win_ret = 1.0 / float(buy_price) - 1.0
    for i, ok in enumerate(won):
        stake = eq * STAKE_PCT
        eq += stake * (win_ret if ok else -1.0)
        eq = max(eq, 0.0)
        curve[i] = eq
    mxdd, mxdd_pct = max_drawdown(curve)
    wins = int(won.sum())
    losses = int(len(won) - wins)
    return {
        "name": name,
        "window": window,
        "buyPrice": float(buy_price),
        "trades": int(len(sel)),
        "wins": wins,
        "losses": losses,
        "winRatePct": round(100.0 * wins / len(sel), 6) if len(sel) else 0.0,
        "compoundPnl": round(float(eq - START_BANKROLL), 6),
        "endingBankroll": round(float(eq), 6),
        "maxDrawdownUsd": round(float(mxdd), 6),
        "maxDrawdownPct": round(float(mxdd_pct) * 100.0, 6),
        "returnDrawdownRatio": round(float(eq - START_BANKROLL) / mxdd, 6) if mxdd > 1e-12 else (999.0 if eq > START_BANKROLL else 0.0),
        "monthlyPositiveRatio": monthly_positive_ratio(curve, sel["dt"]),
        "setHash": stable_hash(sel[["dt", "pred_up15", "label_up", "won"]].to_dict("records")) if len(sel) else "empty",
    }


def build_truth() -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]], dict[str, pd.DataFrame], dict[str, Any]]:
    enriched, truth = base.ext.load_or_build_enriched()
    atom_store = base.build_atom_store(enriched)
    period_vals = base.build_period_vals(enriched)
    return enriched, atom_store, period_vals, truth


def current_061_params() -> dict[str, Any]:
    p = dict(base.CURRENT_061_PARAMS)
    p["candidate_id"] = "current_live_06173_cluster_gate"
    return p


def mask_for_atoms(atom_store: dict[str, dict[str, np.ndarray]], period: str, atoms: list[str]) -> np.ndarray:
    return base.mask_for_atoms(atom_store, period, atoms)


def condition_for_candidate(atom_store: dict[str, dict[str, np.ndarray]], period: str, params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return condition mask and risk vector for a candidate."""
    if params.get("action") in {"weighted_cluster_score", "weighted_hard", "weighted_soft"}:
        clusters = params.get("any_clusters", []) or []
        if not clusters:
            first = next(iter(atom_store[period].values()))
            return np.zeros_like(first, dtype=bool), np.zeros_like(first, dtype=float)
        weights = params.get("weights") or [1.0] * len(clusters)
        risk = np.zeros_like(next(iter(atom_store[period].values())), dtype=float)
        for atoms, weight in zip(clusters, weights):
            risk += mask_for_atoms(atom_store, period, list(atoms)).astype(float) * float(weight)
        cond = risk >= float(params.get("risk_min", 1.0))
        for atom in params.get("global_atoms", []) or []:
            gm = atom_store[period][atom]
            cond &= gm
            risk = np.where(gm, risk, 0.0)
        return cond.astype(bool), risk
    cond = base.condition_for_candidate(atom_store, period, params)
    return cond.astype(bool), cond.astype(float)


def keep_mask(val: pd.DataFrame, cond: np.ndarray, risk: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    score = pd.to_numeric(val["score15"], errors="coerce").fillna(0.0).to_numpy()
    action = params.get("action")
    if action == "hard_block":
        keep = ~cond
    elif action == "raise_score":
        keep = (~cond) | (score >= float(params["shock_score_min"]))
    elif action == "soft_deduct":
        keep = (score - np.where(cond, float(params["deduct"]), 0.0)) >= float(params.get("base_score_min", 0.55))
    elif action == "directional_raise":
        direction = val["direction"].astype(str).str.upper().to_numpy()
        up_min = float(params.get("up_score_min", params.get("shock_score_min", 0.58)))
        down_min = float(params.get("down_score_min", params.get("shock_score_min", 0.58)))
        req = np.where(direction == "UP", up_min, down_min)
        keep = (~cond) | (score >= req)
    elif action == "score_segment_raise":
        low = score < float(params.get("low_score_cut", 0.58))
        keep = (~cond) | (~low) | (score >= float(params.get("shock_score_min", 0.59)))
    elif action == "dynamic_threshold":
        direction = val["direction"].astype(str).str.upper().to_numpy()
        req = np.full(len(score), float(params.get("base_score_min", 0.55)), dtype=float)
        req += np.where(cond, float(params.get("bad_extra", 0.02)), 0.0)
        req += np.where((direction == "UP") & cond, float(params.get("up_extra", 0.0)), 0.0)
        req += np.where((direction == "DOWN") & cond, float(params.get("down_extra", 0.0)), 0.0)
        req += np.where((score < float(params.get("low_score_cut", 0.58))) & cond, float(params.get("low_extra", 0.0)), 0.0)
        keep = score >= req
    elif action == "weighted_cluster_score":
        req = float(params.get("base_score_min", 0.55)) + np.minimum(risk, float(params.get("risk_cap", 4.0))) * float(params.get("risk_slope", 0.01))
        req = np.minimum(req, float(params.get("max_score_min", 0.66)))
        keep = (risk < float(params.get("risk_min", 1.0))) | (score >= req)
    elif action == "weighted_hard":
        keep = risk < float(params.get("risk_min", 1.0))
    elif action == "weighted_soft":
        keep = (score - risk * float(params.get("risk_deduct", 0.01))) >= float(params.get("base_score_min", 0.55))
    else:
        keep = np.ones(len(val), dtype=bool)
    return keep.astype(bool)


def evaluate_candidate(period_vals: dict[str, pd.DataFrame], atom_store: dict[str, dict[str, np.ndarray]], params: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    price_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    name = params.get("candidate_id") or stable_hash(params)
    for window, period in [("180d", "validation_180d"), ("365d", "validation_365d")]:
        val = period_vals[period].reset_index(drop=True)
        cond, risk = condition_for_candidate(atom_store, period, params)
        keep = keep_mask(val, cond, risk, params)
        selected = val[keep].copy().sort_values("dt").reset_index(drop=True)
        won = val["won"].astype(bool).to_numpy()
        blocked = ~keep
        blocked_rows.append({
            "window": window,
            "blockedTrades": int(blocked.sum()),
            "blockedWinners": int((won & blocked).sum()),
            "blockedLosers": int((~won & blocked).sum()),
            "retentionRate": round(100.0 * int(keep.sum()) / max(1, len(keep)), 6),
            "conditionTrades": int(cond.sum()),
            "conditionWinRatePct": round(100.0 * float(won[cond].mean()), 6) if int(cond.sum()) else 0.0,
            "avgRisk": round(float(np.nanmean(risk)), 6) if len(risk) else 0.0,
        })
        for bp in BUY_PRICES:
            r = curve_metrics(selected, name, window, bp)
            r.update({"blocked": blocked_rows[-1], "action": params.get("action"), "candidateId": name})
            price_rows.append(r)
            if bp == PRIMARY_BUY_PRICE:
                rows.append(r)
    return {"name": name, "params": params, "rows": rows, "priceRows": price_rows, "blockedRows": blocked_rows}


def init_worker(period_vals: dict[str, pd.DataFrame], atom_store: dict[str, dict[str, np.ndarray]]) -> None:
    global _G_PERIOD_VALS, _G_ATOMS
    _G_PERIOD_VALS = period_vals
    _G_ATOMS = atom_store


def eval_batch(params_batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if _G_PERIOD_VALS is None or _G_ATOMS is None:
        raise RuntimeError("worker not initialized")
    out = []
    for params in params_batch:
        try:
            out.append(evaluate_candidate(_G_PERIOD_VALS, _G_ATOMS, params))
        except Exception as exc:
            out.append({"name": params.get("candidate_id") or stable_hash(params), "params": params, "error": repr(exc)[:800]})
    return out


def pass_gate(c: dict[str, Any]) -> tuple[bool, list[str]]:
    rows = {r["window"]: r for r in c.get("rows", [])}
    reasons: list[str] = []
    for w in ["180d", "365d"]:
        r = rows.get(w)
        b = BASELINE_061[w]
        if not r:
            reasons.append(f"{w}_missing")
            continue
        if r["compoundPnl"] <= b["compoundPnl"]:
            reasons.append(f"{w}_pnl_not_above_061")
        if w == "365d" and r["winRatePct"] < b["winRatePct"]:
            reasons.append("365d_winrate_below_061")
        if r["maxDrawdownUsd"] > b["maxDrawdownUsd"] * 1.05:
            reasons.append(f"{w}_drawdown_too_high")
        if r["trades"] < max(100 if w == "180d" else 200, int(b["trades"] * 0.45)):
            reasons.append(f"{w}_too_few_trades")
    return not reasons, reasons


def score_candidate(c: dict[str, Any]) -> float:
    rows = {r["window"]: r for r in c.get("rows", [])}
    if set(rows) != {"180d", "365d"}:
        return -1e18
    b180, b365 = BASELINE_061["180d"], BASELINE_061["365d"]
    brs = {r["window"]: r for r in c.get("blockedRows", [])}
    block_edge = (brs.get("180d", {}).get("blockedLosers", 0) - brs.get("180d", {}).get("blockedWinners", 0)) + (brs.get("365d", {}).get("blockedLosers", 0) - brs.get("365d", {}).get("blockedWinners", 0))
    return (
        (rows["180d"]["compoundPnl"] - b180["compoundPnl"]) / max(1.0, b180["compoundPnl"]) * 4000
        + (rows["365d"]["compoundPnl"] - b365["compoundPnl"]) / max(1.0, b365["compoundPnl"]) * 5500
        + (rows["365d"]["winRatePct"] - b365["winRatePct"]) * 35
        - (rows["180d"]["maxDrawdownUsd"] / max(1.0, b180["maxDrawdownUsd"])) * 20
        - (rows["365d"]["maxDrawdownUsd"] / max(1.0, b365["maxDrawdownUsd"])) * 20
        + block_edge * 0.08
    )


def archive_rows(period_vals: dict[str, pd.DataFrame], atom_store: dict[str, dict[str, np.ndarray]], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    if not candidate or not candidate.get("params"):
        return []
    out: list[dict[str, Any]] = []
    try:
        fill_rows, _ = archive.load_live_fill_rows()
        all_eth = archive.aggregate_logical(fill_rows, "全部归档ETH15m live去重")
        strict_mask = fill_rows["sourcePath"].str.contains("slot01|forcedtrade|base_forcedtrade|__live_eth|logs_v5_exp10_bp0480__live_eth", case=False, regex=True)
        strict = archive.aggregate_logical(fill_rows[strict_mask].copy(), "严格旧slot1/ETH相关live去重") if strict_mask.any() else all_eth.iloc[0:0]
        val = period_vals["validation_180d"].reset_index(drop=True)
        cond, risk = condition_for_candidate(atom_store, "validation_180d", candidate["params"])
        keep = keep_mask(val, cond, risk, candidate["params"])
        pred = val.assign(keep=keep.astype(bool))
        for old in [strict, all_eth]:
            if old.empty:
                continue
            merged = old.merge(pred[["dt", "pred_up15", "score15", "keep"]], left_on="marketStart", right_on="dt", how="left")
            chosen = merged[merged["keep"].fillna(False).astype(bool)].copy().sort_values("marketStart")
            if chosen.empty:
                out.append({"scope": old["scopeName"].iloc[0], "oldRealMarkets": int(len(old)), "selectedTrades": 0})
                continue
            won = chosen["pred_up15"].astype(bool).to_numpy() == chosen["actualUp"].astype(bool).to_numpy()
            arch_sel = pd.DataFrame({"dt": chosen["marketStart"], "label_up": chosen["actualUp"].astype(bool), "pred_up15": chosen["pred_up15"].astype(bool), "won": won})
            m = curve_metrics(arch_sel, candidate["name"], str(old["scopeName"].iloc[0]), PRIMARY_BUY_PRICE)
            out.append({"scope": old["scopeName"].iloc[0], "oldRealMarkets": int(len(old)), "selectedTrades": int(len(chosen)), "wins": int(won.sum()), "losses": int(len(won)-int(won.sum())), "winRatePct": round(100*float(won.mean()), 6), "compoundPnl": m["compoundPnl"], "endingBankroll": m["endingBankroll"], "maxDrawdownUsd": m["maxDrawdownUsd"], "setHash": stable_hash(chosen[["marketSlug", "pred_up15", "actualUp"]].to_dict("records"))})
    except Exception as exc:
        out.append({"scope": "archive_error", "error": repr(exc)[:600]})
    return out


def run_audit(period_vals: dict[str, pd.DataFrame], atom_store: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    current = evaluate_candidate(period_vals, atom_store, current_061_params())
    current2 = evaluate_candidate(period_vals, atom_store, current_061_params())
    repeat = []
    for a, b in zip(current["rows"], current2["rows"]):
        repeat.append({"window": a["window"], "hash1": a["setHash"], "hash2": b["setHash"], "pnl1": a["compoundPnl"], "pnl2": b["compoundPnl"], "passed": a["setHash"] == b["setHash"] and a["compoundPnl"] == b["compoundPnl"]})
    baseline_match = []
    for r in current["rows"]:
        b = BASELINE_061[r["window"]]
        baseline_match.append({"window": r["window"], "trades": r["trades"], "winRatePct": r["winRatePct"], "pnl": r["compoundPnl"], "expectedPnl": b["compoundPnl"], "passed": abs(r["compoundPnl"] - b["compoundPnl"]) < 1e-6 and r["trades"] == b["trades"]})
    # Random-label sanity: selected 061 rows should lose its real edge when
    # labels are replaced by independent 50/50 outcomes. A pure permutation
    # would preserve the original win rate and is not a useful sanity check.
    rng = np.random.default_rng(RNG_SEED)
    shuffle_rows = []
    for window, period in [("180d", "validation_180d"), ("365d", "validation_365d")]:
        val = period_vals[period].reset_index(drop=True)
        cond, risk = condition_for_candidate(atom_store, period, current_061_params())
        keep = keep_mask(val, cond, risk, current_061_params())
        selected = val[keep].copy().reset_index(drop=True)
        fake = selected.copy()
        fake["won"] = rng.random(len(fake)) < 0.5
        m = curve_metrics(fake, "shuffle", window, PRIMARY_BUY_PRICE)
        shuffle_rows.append({"window": window, "trades": int(len(fake)), "winRatePct": m["winRatePct"], "pnl": m["compoundPnl"]})
    audit = {"generatedAt": now_iso(), "beijingTime": bj_now(), "researchOnlyNoLiveChange": True, "timePolicy": "closed_candle_available_at_v2", "current061Replay": baseline_match, "repeatability": repeat, "shuffleSanity": shuffle_rows, "passed": all(x["passed"] for x in baseline_match) and all(x["passed"] for x in repeat)}
    write_json(OUT_AUDIT_JSON, audit)
    write_text(OUT_AUDIT_MD, "\n".join(["# 061 全字段长跑超参：审计", "", f"- 北京时间：`{audit['beijingTime']}`", f"- 时间口径：`{audit['timePolicy']}`", f"- 当前061复现：`{baseline_match}`", f"- 重复运行：`{repeat}`", f"- 随机标签 sanity：`{shuffle_rows}`", f"- 审计通过：`{audit['passed']}`"]) + "\n")
    return audit


def mine_clusters(enriched: pd.DataFrame, atom_store: dict[str, dict[str, np.ndarray]], limit: int = 260) -> list[dict[str, Any]]:
    clusters = base.mine_bad_clusters(enriched, atom_store, limit=limit)
    for atoms in base.CURRENT_061_PARAMS.get("any_clusters", []):
        clusters.append({"cluster_id": stable_hash({"current061": atoms}), "atoms": list(atoms), "score": 999.0, "source": "current061"})
    seed = [
        ["15m_same_as_top159", "1h_same_as_top159", "4h_rangeq_ge_0.65"],
        ["15m_same_as_top159", "1h_vol_ge_1.6", "4h_vol_ge_1.3"],
        ["15m_same_as_top159", "score_lt_060", "1h_trend_down"],
        ["15m_same_as_top159", "score_lt_058", "4h_pos_high"],
    ]
    for atoms in seed:
        clusters.append({"cluster_id": stable_hash({"seed": atoms}), "atoms": atoms, "score": 850.0, "source": "seed"})
    dedup: dict[str, dict[str, Any]] = {}
    for c in clusters:
        dedup[c["cluster_id"]] = c
    rows = list(dedup.values())
    rows.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return rows


def useful_global_atoms(atom_store: dict[str, dict[str, np.ndarray]]) -> list[list[str]]:
    names = sorted(set(atom_store["validation_180d"]).intersection(atom_store["validation_365d"]))
    keep = [[]]
    preferred_prefixes = [
        "dir_", "score_", "bj_hour_",
        "15m_trend_", "1h_trend_", "4h_trend_",
        "15m_pos_", "1h_pos_", "4h_pos_",
        "15m_vol_ge_", "1h_vol_ge_", "4h_vol_ge_",
        "15m_rangeq_ge_", "1h_rangeq_ge_", "4h_rangeq_ge_",
        "15m_same_as_top159", "1h_same_as_top159", "4h_same_as_top159",
        "15m_opposes_top159", "1h_opposes_top159", "4h_opposes_top159",
    ]
    singles = [n for n in names if any(n.startswith(p) or n == p for p in preferred_prefixes)]
    for n in singles:
        # Avoid ultra-rare global constraints.
        n180 = int(atom_store["validation_180d"][n].sum())
        n365 = int(atom_store["validation_365d"][n].sum())
        if n180 >= 80 and n365 >= 160:
            keep.append([n])
    # Add a few deterministic two-atom environment gates without exploding too much.
    pairs = [
        ["dir_UP", "score_lt_060"], ["dir_DOWN", "score_lt_060"],
        ["1h_trend_down", "score_lt_060"], ["1h_trend_up", "score_lt_060"],
        ["4h_trend_down", "score_lt_060"], ["4h_trend_up", "score_lt_060"],
    ]
    for p in pairs:
        if all(x in names for x in p):
            keep.append(p)
    limit = int(os.environ.get("TOP159_061_FULLFIELD_GLOBAL_LIMIT", "90"))
    return keep[:limit]


def param_candidate_id(raw: dict[str, Any]) -> str:
    payload = dict(raw)
    payload.pop("candidate_id", None)
    payload["timePolicy"] = "closed_candle_available_at_v2"
    return stable_hash(payload)


def with_id(raw: dict[str, Any]) -> dict[str, Any]:
    r = dict(raw)
    r["candidate_id"] = param_candidate_id(r)
    return r


def candidate_stream(clusters: list[dict[str, Any]], global_atoms: list[list[str]]) -> Iterable[dict[str, Any]]:
    yield current_061_params()
    expanded = os.environ.get("TOP159_061_FULLFIELD_EXPANDED", "0").strip() == "1"
    fast_new_first = expanded and os.environ.get("TOP159_061_FULLFIELD_FAST_NEW_FIRST", "0").strip() == "1"
    fast_new_only = fast_new_first and os.environ.get("TOP159_061_FULLFIELD_FAST_NEW_ONLY", "0").strip() == "1"
    base_score_grid = [round(x, 3) for x in np.arange(0.55, 0.6401, 0.005)] + [0.66, 0.68, 0.70]
    base_deduct_grid = [round(x, 3) for x in np.arange(0.005, 0.0501, 0.005)]
    base_low_cuts = [0.57, 0.58, 0.60, 0.62]
    score_grid = list(base_score_grid)
    deduct_grid = list(base_deduct_grid)
    low_cuts = list(base_low_cuts)
    if expanded:
        score_grid = sorted(set(score_grid + [round(x, 4) for x in np.arange(0.5525, 0.6626, 0.0025)] + [0.675, 0.685, 0.695, 0.72]))
        deduct_grid = sorted(set(deduct_grid + [round(x, 4) for x in np.arange(0.0025, 0.0801, 0.0025)]))
        low_cuts = sorted(set(low_cuts + [0.555, 0.56, 0.565, 0.575, 0.585, 0.59, 0.61, 0.64]))
    top_single = clusters[: int(os.environ.get("TOP159_061_FULLFIELD_TOP_SINGLE", "180"))]
    top_combo = clusters[: int(os.environ.get("TOP159_061_FULLFIELD_TOP_COMBO", "90"))]
    top_weight = clusters[: int(os.environ.get("TOP159_061_FULLFIELD_TOP_WEIGHT", "80"))]

    seen: set[str] = set()

    def emit(raw: dict[str, Any]):
        r = with_id(raw)
        if r["candidate_id"] in seen:
            return None
        seen.add(r["candidate_id"])
        return r

    # Resume runs already have a large completed result set.  If we enumerate the
    # old search space first, the main process can burn a core skipping hundreds
    # of thousands of known candidates before workers receive any new work.  This
    # front-loads the newly expanded regions so ProcessPool workers start quickly.
    if fast_new_first:
        new_scores = [x for x in score_grid if x not in set(base_score_grid)]
        new_deducts = [x for x in deduct_grid if x not in set(base_deduct_grid)]
        new_lows = [x for x in low_cuts if x not in set(base_low_cuts)]
        tail_single = top_single[180:] or top_single
        tail_combo = top_combo[90:] or top_combo
        tail_weight = top_weight[80:] or top_weight
        tail_globals = global_atoms[90:] or global_atoms[:30]

        for cl in tail_single:
            atoms = list(cl["atoms"])
            for ga in tail_globals:
                base_raw = {"family": "cluster_gate", "atoms": atoms, "global_atoms": ga}
                r = emit(base_raw | {"action": "hard_block"})
                if r: yield r
                for ss in new_scores:
                    r = emit(base_raw | {"action": "raise_score", "shock_score_min": ss})
                    if r: yield r
                for dd in new_deducts:
                    r = emit(base_raw | {"action": "soft_deduct", "deduct": dd, "base_score_min": 0.55})
                    if r: yield r
                for low in new_lows:
                    for ss in [0.565, 0.575, 0.585, 0.595, 0.61, 0.63, 0.65]:
                        r = emit(base_raw | {"action": "score_segment_raise", "low_score_cut": low, "shock_score_min": ss})
                        if r: yield r

        for k, topn in [(2, min(len(tail_combo), 40)), (3, min(len(tail_combo), 32)), (4, min(len(tail_combo), 24))]:
            if topn < k:
                continue
            for combo in combinations(tail_combo[:topn], k):
                any_clusters = [list(c["atoms"]) for c in combo]
                for ga in tail_globals[:40]:
                    for min_hits in range(1, min(k, 3) + 1):
                        raw_base = {"family": "cluster_gate", "any_clusters": any_clusters, "min_cluster_hits": min_hits, "global_atoms": ga}
                        for ss in [0.5525, 0.5575, 0.5625, 0.5675, 0.5725, 0.5775, 0.5825, 0.5875, 0.5925, 0.605, 0.615, 0.635, 0.675]:
                            r = emit(raw_base | {"action": "raise_score", "shock_score_min": ss})
                            if r: yield r
                        for dd in [0.0025, 0.0075, 0.0125, 0.0175, 0.025, 0.035, 0.055, 0.07]:
                            r = emit(raw_base | {"action": "soft_deduct", "deduct": dd, "base_score_min": 0.55})
                            if r: yield r
                        r = emit(raw_base | {"action": "hard_block"})
                        if r: yield r
                        for up_min in [0.5525, 0.5625, 0.5725, 0.585, 0.605, 0.635]:
                            for down_min in [0.5525, 0.5625, 0.5725, 0.585, 0.605, 0.635]:
                                r = emit(raw_base | {"action": "directional_raise", "up_score_min": up_min, "down_score_min": down_min})
                                if r: yield r

        for k, topn in [(2, min(len(tail_weight), 34)), (3, min(len(tail_weight), 28)), (4, min(len(tail_weight), 22))]:
            if topn < k:
                continue
            for combo in combinations(tail_weight[:topn], k):
                any_clusters = [list(c["atoms"]) for c in combo]
                for ga in tail_globals[:30]:
                    for risk_min in [0.5, 1.0, 1.25, 1.5, 1.75, 2.25, 2.75, 3.25]:
                        raw_base = {"family": "weighted_cluster_gate", "any_clusters": any_clusters, "weights": [1.0] * k, "risk_min": risk_min, "global_atoms": ga}
                        for slope in [0.0025, 0.0075, 0.0125, 0.0175, 0.025, 0.04]:
                            r = emit(raw_base | {"action": "weighted_cluster_score", "base_score_min": 0.55, "risk_slope": slope, "risk_cap": 4.0, "max_score_min": 0.66})
                            if r: yield r
                        for dd in [0.0025, 0.0075, 0.0125, 0.0175, 0.025, 0.035]:
                            r = emit(raw_base | {"action": "weighted_soft", "base_score_min": 0.55, "risk_deduct": dd})
                            if r: yield r
                        r = emit(raw_base | {"action": "weighted_hard"})
                        if r: yield r

        if fast_new_only:
            return

    for cl in top_single:
        atoms = list(cl["atoms"])
        for ga in global_atoms:
            base_raw = {"family": "cluster_gate", "atoms": atoms, "global_atoms": ga}
            r = emit(base_raw | {"action": "hard_block"})
            if r: yield r
            for ss in score_grid:
                r = emit(base_raw | {"action": "raise_score", "shock_score_min": ss})
                if r: yield r
            for dd in deduct_grid:
                r = emit(base_raw | {"action": "soft_deduct", "deduct": dd, "base_score_min": 0.55})
                if r: yield r
            for low in low_cuts:
                for ss in [0.58, 0.59, 0.60, 0.62, 0.64]:
                    r = emit(base_raw | {"action": "score_segment_raise", "low_score_cut": low, "shock_score_min": ss})
                    if r: yield r

    # OR / min-hit cluster combinations.
    for k, topn in [(2, 80), (3, 58), (4, 42)]:
        for combo in combinations(top_combo[:topn], k):
            any_clusters = [list(c["atoms"]) for c in combo]
            for ga in global_atoms[:40]:
                for min_hits in range(1, min(k, 3) + 1):
                    raw_base = {"family": "cluster_gate", "any_clusters": any_clusters, "min_cluster_hits": min_hits, "global_atoms": ga}
                    for ss in [0.555, 0.56, 0.565, 0.57, 0.575, 0.58, 0.585, 0.59, 0.60, 0.62, 0.64]:
                        r = emit(raw_base | {"action": "raise_score", "shock_score_min": ss})
                        if r: yield r
                    for dd in [0.005, 0.01, 0.015, 0.02, 0.03, 0.04]:
                        r = emit(raw_base | {"action": "soft_deduct", "deduct": dd, "base_score_min": 0.55})
                        if r: yield r
                    r = emit(raw_base | {"action": "hard_block"})
                    if r: yield r
                    for up_min in [0.56, 0.57, 0.58, 0.60, 0.62]:
                        for down_min in [0.56, 0.57, 0.58, 0.60, 0.62]:
                            r = emit(raw_base | {"action": "directional_raise", "up_score_min": up_min, "down_score_min": down_min})
                            if r: yield r

    # Dynamic threshold variants on stronger clusters.
    for cl in top_single[:120]:
        atoms = list(cl["atoms"])
        for ga in global_atoms[:50]:
            raw_base = {"family": "cluster_gate", "atoms": atoms, "global_atoms": ga}
            for bad_extra in [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.06]:
                for low_extra in [0.0, 0.005, 0.01, 0.02]:
                    for up_extra, down_extra in [(0.0, 0.0), (0.005, 0.0), (0.0, 0.005), (0.01, 0.0), (0.0, 0.01)]:
                        r = emit(raw_base | {"action": "dynamic_threshold", "base_score_min": 0.55, "bad_extra": bad_extra, "low_score_cut": 0.58, "low_extra": low_extra, "up_extra": up_extra, "down_extra": down_extra})
                        if r: yield r

    # Weighted cluster risk score variants.
    for k, topn in [(2, 70), (3, 54), (4, 38)]:
        for combo in combinations(top_weight[:topn], k):
            any_clusters = [list(c["atoms"]) for c in combo]
            weights_sets = [[1.0] * k]
            if k >= 3:
                weights_sets.append([1.5] + [1.0] * (k - 1))
                weights_sets.append([1.0, 1.5] + [1.0] * (k - 2))
            for weights in weights_sets:
                for ga in global_atoms[:30]:
                    for risk_min in [1.0, 1.5, 2.0, 2.5, 3.0]:
                        raw_base = {"family": "weighted_cluster_gate", "any_clusters": any_clusters, "weights": weights, "risk_min": risk_min, "global_atoms": ga}
                        for slope in [0.005, 0.01, 0.015, 0.02, 0.03]:
                            r = emit(raw_base | {"action": "weighted_cluster_score", "base_score_min": 0.55, "risk_slope": slope, "risk_cap": 4.0, "max_score_min": 0.66})
                            if r: yield r
                        for dd in [0.005, 0.01, 0.015, 0.02]:
                            r = emit(raw_base | {"action": "weighted_soft", "base_score_min": 0.55, "risk_deduct": dd})
                            if r: yield r
                        r = emit(raw_base | {"action": "weighted_hard"})
                        if r: yield r


def load_done_results() -> tuple[set[str], list[dict[str, Any]]]:
    done: set[str] = set()
    rows: list[dict[str, Any]] = []
    if RESET:
        for p in [OUT_RESULTS, OUT_CHECKPOINT, OUT_LEADERBOARD_JSON, OUT_COMPARE_JSON, OUT_VERDICT_JSON, OUT_LEADERBOARD_MD, OUT_COMPARE_MD, OUT_VERDICT_MD]:
            if p.exists():
                p.unlink()
        return done, rows
    if os.environ.get("TOP159_061_FULLFIELD_SKIP_DONE_LOAD", "0").strip() == "1":
        return done, rows
    if not OUT_RESULTS.exists():
        return done, rows
    with OUT_RESULTS.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            cid = obj.get("name") or obj.get("params", {}).get("candidate_id")
            if cid:
                done.add(cid)
                rows.append(obj)
    return done, rows


def append_results(rows: list[dict[str, Any]]) -> None:
    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_RESULTS.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for c in results:
        if not c.get("rows"):
            continue
        ok, reasons = pass_gate(c)
        ranked.append({**c, "passed": ok, "reasons": reasons, "score": score_candidate(c)})
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


def result_candidate_id(c: dict[str, Any] | None) -> str:
    if not c:
        return ""
    return str((c.get("params") or {}).get("candidate_id") or c.get("name") or "")


def is_current_061_candidate(c: dict[str, Any] | None) -> bool:
    return result_candidate_id(c) == "current_live_06173_cluster_gate"


def best_non_current_candidate(ranked: list[dict[str, Any]]) -> dict[str, Any] | None:
    for c in ranked:
        if not is_current_061_candidate(c):
            return c
    return None


def render(payload: dict[str, Any]) -> None:
    selected = payload.get("selected")
    display_selected = selected if selected and selected.get("passed") else None
    best_observation = payload.get("bestObservation")
    top = payload.get("topCandidates", [])
    lines = ["# 061 全字段长跑超参：061公平对比", "", f"- 北京时间：`{payload['beijingTime']}`", "- 口径：`850U初始 / 每笔1% / 满成交 / 买价0.55`", "- live动作：`无，研究只读`", "", "## 061基线 vs 当前最强候选", "", "|配置|窗口|交易数|胜/负|胜率|盈亏|期末资金|最大回撤|收益回撤比|拦截赢家|拦截输单|保留率|哈希|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for w in ["180d", "365d"]:
        b = BASELINE_061[w]
        lines.append(f"|当前061|{w}|{b['trades']}|{b['wins']}/{b['losses']}|{b['winRatePct']:.2f}%|{b['compoundPnl']:,.2f}|{b['endingBankroll']:,.2f}|{b['maxDrawdownUsd']:,.2f}|{b['returnDrawdownRatio']:.2f}|-|-|-|`{b['setHash']}`|")
    if display_selected:
        blocked = {r["window"]: r for r in display_selected.get("blockedRows", [])}
        for r in display_selected.get("rows", []):
            br = blocked.get(r["window"], {})
            lines.append(f"|全字段严格通过候选|{r['window']}|{r['trades']}|{r['wins']}/{r['losses']}|{r['winRatePct']:.2f}%|{r['compoundPnl']:,.2f}|{r['endingBankroll']:,.2f}|{r['maxDrawdownUsd']:,.2f}|{r['returnDrawdownRatio']:.2f}|{br.get('blockedWinners','-')}|{br.get('blockedLosers','-')}|{br.get('retentionRate','-')}|`{r['setHash']}`|")
    else:
        lines.append("|全字段严格通过候选|-|-|-|-|-|-|-|-|-|-|-|无|")
    if best_observation:
        lines += ["", "## 最接近但未通过候选（观察，不可上线）", "", "|窗口|交易数|胜/负|胜率|盈亏|期末资金|最大回撤|收益回撤比|拦截赢家|拦截输单|失败原因|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        blocked = {r["window"]: r for r in best_observation.get("blockedRows", [])}
        reasons = ",".join(best_observation.get("reasons", []))
        for r in best_observation.get("rows", []):
            br = blocked.get(r["window"], {})
            lines.append(f"|{r['window']}|{r['trades']}|{r['wins']}/{r['losses']}|{r['winRatePct']:.2f}%|{r['compoundPnl']:,.2f}|{r['endingBankroll']:,.2f}|{r['maxDrawdownUsd']:,.2f}|{r['returnDrawdownRatio']:.2f}|{br.get('blockedWinners','-')}|{br.get('blockedLosers','-')}|{reasons}|")
    lines += ["", "## 前30候选", "", "|排名|动作|交易数365|365胜率|365盈亏|365回撤|拦截赢家365|拦截输单365|通过|失败原因|", "|---:|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for i, c in enumerate(top[:30], 1):
        rows = {r["window"]: r for r in c.get("rows", [])}
        brs = {r["window"]: r for r in c.get("blockedRows", [])}
        r365 = rows.get("365d", {})
        b365 = brs.get("365d", {})
        lines.append(f"|{i}|{c.get('params',{}).get('action')}|{r365.get('trades',0)}|{r365.get('winRatePct',0):.2f}%|{r365.get('compoundPnl',0):,.2f}|{r365.get('maxDrawdownUsd',0):,.2f}|{b365.get('blockedWinners','-')}|{b365.get('blockedLosers','-')}|{c.get('passed', False)}|{','.join(c.get('reasons', []))}|")
    write_text(OUT_COMPARE_MD, "\n".join(lines) + "\n")
    write_text(OUT_LEADERBOARD_MD, "\n".join(lines) + "\n")
    v = payload.get("verdict", {})
    vlines = ["# 061 全字段长跑超参：唯一结论", "", f"- 状态：`{v.get('status')}`", f"- 结论：{v.get('message')}", "", "## 历史归档真实单纯预测复核", "", "|范围|旧真实市场数|选中|胜/负|胜率|盈亏|最大回撤|", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in payload.get("archiveRows", []):
        vlines.append(f"|{r.get('scope')}|{r.get('oldRealMarkets',0)}|{r.get('selectedTrades',0)}|{r.get('wins',0)}/{r.get('losses',0)}|{r.get('winRatePct',0):.2f}%|{r.get('compoundPnl',0):,.2f}|{r.get('maxDrawdownUsd',0):,.2f}|")
    write_text(OUT_VERDICT_MD, "\n".join(vlines) + "\n")


def write_progress(results: list[dict[str, Any]], started: float, audit: dict[str, Any], period_vals: dict[str, pd.DataFrame], atom_store: dict[str, dict[str, np.ndarray]], generated: int, skipped: int, finished: bool) -> None:
    ranked = rank_results(results)
    strict = [r for r in ranked if r.get("passed")]
    selected = strict[0] if strict else (ranked[0] if ranked else None)
    best_observation = best_non_current_candidate(ranked)
    arch = archive_rows(period_vals, atom_store, selected) if selected else []
    arch_bad = any(r.get("selectedTrades", 0) >= 30 and r.get("winRatePct", 100) < 53.0 for r in arch)
    status = "candidate_beats_061" if selected and selected.get("passed") and not arch_bad else "running_no_pass_yet" if not finished else "no_candidate_beats_061"
    msg = "找到比当前061更强的全字段候选，只进入影子验证，不改真钱。" if status == "candidate_beats_061" else ("全字段搜索仍在运行，当前尚无严格通过候选。" if not finished else "全字段搜索没有找到同时提升180/365、归档不差的候选；当前真钱继续保留061。")
    payload = {"generatedAt": now_iso(), "beijingTime": bj_now(), "researchOnlyNoLiveChange": True, "finished": finished, "elapsedSeconds": round(time.time() - started, 3), "workers": WORKERS, "batchSize": BATCH_SIZE, "maxCandidates": MAX_CANDIDATES, "generatedCandidates": generated, "skippedExisting": skipped, "evaluatedResults": len(results), "strictPass": len(strict), "audit": audit, "baseline061": BASELINE_061, "topCandidates": ranked[:300], "selected": selected, "bestObservation": best_observation, "archiveRows": arch, "verdict": {"status": status, "message": msg}}
    write_json(OUT_CHECKPOINT, payload)
    write_json(OUT_LEADERBOARD_JSON, payload)
    write_json(OUT_COMPARE_JSON, payload)
    write_json(OUT_VERDICT_JSON, payload["verdict"] | {"selected": selected, "archiveRows": arch})
    render(payload)


def chunks(stream: Iterable[dict[str, Any]], done: set[str], max_candidates: int) -> Iterable[tuple[list[dict[str, Any]], int, int]]:
    batch: list[dict[str, Any]] = []
    generated = 0
    skipped = 0
    for p in stream:
        generated += 1
        if generated > max_candidates:
            break
        cid = p.get("candidate_id") or with_id(p)["candidate_id"]
        p["candidate_id"] = cid
        if cid in done:
            skipped += 1
            continue
        batch.append(p)
        if len(batch) >= BATCH_SIZE:
            yield batch, generated, skipped
            batch = []
    if batch:
        yield batch, generated, skipped


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    OUT_PID.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    started = time.time()
    limit_seconds = max_seconds()
    enriched, atom_store, period_vals, _truth = build_truth()
    audit = run_audit(period_vals, atom_store)
    if not audit.get("passed"):
        write_progress([], started, audit, period_vals, atom_store, generated=0, skipped=0, finished=True)
        raise SystemExit("audit failed")
    clusters = mine_clusters(enriched, atom_store, limit=int(os.environ.get("TOP159_061_FULLFIELD_CLUSTER_LIMIT", "260")))
    globals_ = useful_global_atoms(atom_store)
    done, results = load_done_results()
    last_progress = time.time()
    generated_latest = 0
    skipped_latest = 0

    with cf.ProcessPoolExecutor(max_workers=WORKERS, initializer=init_worker, initargs=(period_vals, atom_store)) as ex:
        pending: list[cf.Future] = []
        stream = chunks(candidate_stream(clusters, globals_), done, MAX_CANDIDATES)
        exhausted = False
        while True:
            while len(pending) < WORKERS * 2 and not exhausted and time.time() - started < limit_seconds:
                try:
                    batch, generated_latest, skipped_latest = next(stream)
                except StopIteration:
                    exhausted = True
                    break
                pending.append(ex.submit(eval_batch, batch))
            if not pending:
                break
            done_futs, pending_set = cf.wait(pending, timeout=2.0, return_when=cf.FIRST_COMPLETED)
            pending = list(pending_set)
            for fut in done_futs:
                rows = fut.result()
                results.extend(rows)
                append_results(rows)
                for row in rows:
                    cid = row.get("name") or row.get("params", {}).get("candidate_id")
                    if cid:
                        done.add(cid)
            if time.time() - last_progress >= 900:
                write_progress(results, started, audit, period_vals, atom_store, generated_latest, skipped_latest, finished=False)
                last_progress = time.time()
            if time.time() - started >= limit_seconds and not pending:
                break
            if exhausted and not pending:
                break
    finished = exhausted or generated_latest >= MAX_CANDIDATES or time.time() - started >= limit_seconds
    write_progress(results, started, audit, period_vals, atom_store, generated_latest, skipped_latest, finished=finished)


if __name__ == "__main__":
    main()
