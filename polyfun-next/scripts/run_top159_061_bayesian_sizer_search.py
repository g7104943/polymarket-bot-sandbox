#!/usr/bin/env python3
from __future__ import annotations

"""Bayesian online calibration + dynamic sizing for current ETH061.

Research only. Does not mutate live configs, orders, ledgers, claim state, or
monitor state.

Locked research口径:
  - 061 remains the directional/selection core.
  - 180d/365d use audited closed-candle ETH061 universe.
  - Primary simulated buy price: 0.55, current-bankroll stake base: 1%.
  - Official replay uses actual official-filled average buy price and official settlement cache.
"""

import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(key, os.environ.get("TOP159_061_BAYES_THREADS", "1"))

import numpy as np
import pandas as pd

ROOT = Path("/Users/mac/polyfun")
NEXT = ROOT / "polyfun-next"
REPORTS = ROOT / "reports"
SCRIPTS = NEXT / "scripts"
BASE_SCRIPT = SCRIPTS / "run_top159_061_score_calibration_router.py"
ACTUAL_SCRIPT = SCRIPTS / "run_top159_061_full_calibration_actual_live_replay.py"

OUT_AUDIT_MD = REPORTS / "top159_061_bayesian_sizer_bug_audit_latest.md"
OUT_AUDIT_JSON = REPORTS / "top159_061_bayesian_sizer_bug_audit_latest.json"
OUT_LEADERBOARD_JSON = REPORTS / "top159_061_bayesian_sizer_leaderboard_latest.json"
OUT_LEADERBOARD_MD = REPORTS / "top159_061_bayesian_sizer_leaderboard_latest.md"
OUT_COMPARE_JSON = REPORTS / "top159_061_bayesian_sizer_180_365_official_compare_latest.json"
OUT_COMPARE_MD = REPORTS / "top159_061_bayesian_sizer_180_365_official_compare_latest.md"
OUT_VERDICT_JSON = REPORTS / "top159_061_bayesian_sizer_unique_verdict_latest.json"
OUT_VERDICT_MD = REPORTS / "top159_061_bayesian_sizer_unique_verdict_latest.md"

START_BANKROLL = 850.0
BUY_PRICE = 0.55
STAKE_PCT = 0.01
RNG_SEED = 20260511

BASELINE_061 = {
    "180d": {"trades": 3942, "wins": 2324, "losses": 1618, "winRatePct": 58.954845, "compoundPnl": 11494.591625, "endingBankroll": 12344.591625, "maxDrawdownUsd": 1481.713373, "returnDrawdownRatio": 7.757635, "monthlyPositiveRatio": 1.0, "setHash": "b35313c05e5b66d2"},
    "365d": {"trades": 9152, "wins": 5246, "losses": 3906, "winRatePct": 57.320804, "compoundPnl": 27033.917055, "endingBankroll": 27883.917055, "maxDrawdownUsd": 3499.248929, "returnDrawdownRatio": 7.725634, "monthlyPositiveRatio": 0.846154, "setHash": "ced1cf82642d8f0d"},
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

base = load_module("bayes_sizer_base061", BASE_SCRIPT)
actual = load_module("bayes_sizer_actual_replay", ACTUAL_SCRIPT)
shock = actual.shock


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def bj_now() -> str:
    return pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S CST")


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


@dataclass(frozen=True)
class SizerPolicy:
    name: str
    dims: tuple[str, ...]
    prior_strength: float
    min_count: int
    weak_cut: float
    strong_cut: float
    very_strong_cut: float
    weak_mult: float
    strong_mult: float
    very_strong_mult: float
    skip_cut: float | None = None
    skip_min_count: int = 999999
    update_lag: int = 2
    use_price_break_even: bool = True

    def key(self) -> str:
        return stable_hash(asdict(self))


def max_drawdown(curve: np.ndarray) -> tuple[float, float]:
    if curve.size == 0:
        return 0.0, 0.0
    peak = np.maximum.accumulate(curve)
    dd = peak - curve
    idx = int(np.argmax(dd))
    mx = float(dd[idx])
    pct = float(mx / peak[idx]) if peak[idx] > 1e-12 else 0.0
    return mx, pct


def monthly_positive_ratio(equity: np.ndarray, dt: pd.Series) -> float:
    if equity.size == 0:
        return 0.0
    frame = pd.DataFrame({"dt": pd.to_datetime(dt, utc=True, errors="coerce"), "equity": equity}).dropna().sort_values("dt")
    if frame.empty:
        return 0.0
    frame["month"] = frame["dt"].dt.to_period("M").astype(str)
    prev = START_BANKROLL
    vals: list[float] = []
    for _, g in frame.groupby("month", sort=True):
        end = float(g["equity"].iloc[-1])
        vals.append(end - prev)
        prev = end
    return round(sum(v > 0 for v in vals) / len(vals), 6) if vals else 0.0


def current_061_selected(period_vals: dict[str, pd.DataFrame], atom_store: dict[str, dict[str, np.ndarray]], period: str) -> pd.DataFrame:
    df = period_vals[period].copy().reset_index(drop=True)
    mask = base.current_061_mask(atom_store, period, df)
    out = df[mask].copy().sort_values("dt").reset_index(drop=True)
    out["current061_keep"] = True
    return out


def cluster_hit_count(df: pd.DataFrame) -> np.ndarray:
    same15 = df.get("15m_same_as_top159", pd.Series(False, index=df.index)).fillna(False).astype(bool).to_numpy()
    same1h = df.get("1h_same_as_top159", pd.Series(False, index=df.index)).fillna(False).astype(bool).to_numpy()
    vol1h = pd.to_numeric(df.get("1h_volume_mult", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0).to_numpy()
    vol4h = pd.to_numeric(df.get("4h_volume_mult", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0).to_numpy()
    pos1h = pd.to_numeric(df.get("1h_pos20", pd.Series(0.5, index=df.index)), errors="coerce").fillna(0.5).to_numpy()
    trend1h = df.get("1h_trend_state", pd.Series("", index=df.index)).fillna("").astype(str).to_numpy()
    conds = [
        same15 & (vol1h >= 1.6) & (vol4h >= 1.1),
        same15 & (vol4h >= 1.1) & (pos1h >= 0.75),
        same15 & (pos1h <= 0.25),
        same15 & same1h & (trend1h == "down"),
    ]
    return np.vstack(conds).sum(axis=0).astype(int)


def add_bucket_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    score = pd.to_numeric(out["score15"], errors="coerce").fillna(0.0)
    out["score_bin"] = pd.cut(score, bins=[0.0, 0.57, 0.60, 0.64, 1.0], labels=["055_057", "057_060", "060_064", "ge064"], include_lowest=True).astype(str)
    out["score_bin_fine"] = pd.cut(score, bins=[0.0, 0.56, 0.57, 0.58, 0.60, 0.62, 0.65, 1.0], labels=["055_056", "056_057", "057_058", "058_060", "060_062", "062_065", "ge065"], include_lowest=True).astype(str)
    bj = pd.to_datetime(out["dt"], utc=True, errors="coerce").dt.tz_convert("Asia/Shanghai")
    hour = bj.dt.hour.fillna(0).astype(int)
    out["hour_bucket"] = pd.cut(hour, bins=[-1, 3, 7, 11, 15, 19, 23], labels=["00_03", "04_07", "08_11", "12_15", "16_19", "20_23"]).astype(str)
    out["direction"] = out["direction"].astype(str).str.upper()
    for tf in ["15m", "1h", "4h"]:
        out[f"{tf}_trend_bucket"] = out.get(f"{tf}_trend_state", pd.Series("missing", index=out.index)).fillna("missing").astype(str)
        pos = pd.to_numeric(out.get(f"{tf}_pos20", pd.Series(0.5, index=out.index)), errors="coerce").fillna(0.5)
        out[f"{tf}_pos_bucket"] = np.where(pos >= 0.75, "high", np.where(pos <= 0.25, "low", "mid"))
        rq = pd.to_numeric(out.get(f"{tf}_range_q", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
        out[f"{tf}_range_bucket"] = np.where(rq >= 0.70, "high", np.where(rq >= 0.45, "mid", "low"))
        vm = pd.to_numeric(out.get(f"{tf}_volume_mult", pd.Series(0.0, index=out.index)), errors="coerce").fillna(0.0)
        out[f"{tf}_vol_bucket"] = np.where(vm >= 1.6, "high", np.where(vm >= 1.1, "mid", "low"))
        for flag in ["same_as_top159", "opposes_top159", "terminal_chase", "exhaustion_wick"]:
            col = f"{tf}_{flag}"
            out[col] = out.get(col, pd.Series(False, index=out.index)).fillna(False).astype(bool).astype(int).astype(str)
    out["cluster_hits"] = cluster_hit_count(out).astype(str)
    out["shock_any"] = out.get("shock_base_any", pd.Series(False, index=out.index)).fillna(False).astype(bool).astype(int).astype(str)
    out["same_stack"] = (
        out.get("15m_same_as_top159", pd.Series(False, index=out.index)).fillna(False).astype(bool).astype(int)
        + out.get("1h_same_as_top159", pd.Series(False, index=out.index)).fillna(False).astype(bool).astype(int)
        + out.get("4h_same_as_top159", pd.Series(False, index=out.index)).fillna(False).astype(bool).astype(int)
    ).astype(str)
    return out


def bucket_key(row: pd.Series, dims: Iterable[str]) -> tuple[Any, ...]:
    return tuple(row.get(d, "missing") for d in dims)


def init_counts(train_df: pd.DataFrame, dims: tuple[str, ...]) -> tuple[dict[tuple[Any, ...], list[int]], float]:
    train = train_df.sort_values("dt").reset_index(drop=True)
    if "score_bin" not in train.columns:
        train = add_bucket_features(train)
    wins = train["won"].astype(bool).to_numpy()
    global_p = float(wins.mean()) if len(wins) else 0.57
    counts: dict[tuple[Any, ...], list[int]] = {}
    key_arrays = [train[d].astype(str).to_numpy(dtype=object) if d in train.columns else np.full(len(train), "missing", dtype=object) for d in dims]
    for k, won in zip(zip(*key_arrays), wins):
        c = counts.setdefault(k, [0, 0])
        c[0] += int(bool(won))
        c[1] += 1
    return counts, global_p


def p_hat_for(counts: dict[tuple[Any, ...], list[int]], key: tuple[Any, ...], prior_strength: float, global_p: float) -> tuple[float, int]:
    wins, n = counts.get(key, [0, 0])
    p = (wins + float(prior_strength) * global_p) / (n + float(prior_strength)) if (n + prior_strength) > 0 else global_p
    return float(p), int(n)


def multiplier_for(policy: SizerPolicy, p: float, n: int, buy_price: float) -> tuple[float, str]:
    # Use buy_price as breakeven probability when enabled. Under fixed 0.55,
    # p_hat is compared directly with 0.55; under actual replay, it adapts to
    # the actual observed buy price.
    shift = float(buy_price - BUY_PRICE) if policy.use_price_break_even else 0.0
    weak_cut = policy.weak_cut + shift
    strong_cut = policy.strong_cut + shift
    very_cut = policy.very_strong_cut + shift
    skip_cut = None if policy.skip_cut is None else policy.skip_cut + shift
    if skip_cut is not None and n >= policy.skip_min_count and p < skip_cut:
        return 0.0, "skip"
    if n >= policy.min_count and p < weak_cut:
        return policy.weak_mult, "down"
    if p >= very_cut:
        return policy.very_strong_mult, "up_strong"
    if p >= strong_cut:
        return policy.strong_mult, "up"
    return 1.0, "base"


def simulate_online(train_df: pd.DataFrame, val_df: pd.DataFrame, policy: SizerPolicy, name: str, window: str, initial: float, buy_price: float, stake_fraction: float, actual_price_col: str | None = None) -> dict[str, Any]:
    train = train_df if "score_bin" in train_df.columns else add_bucket_features(train_df)
    val = val_df if "score_bin" in val_df.columns else add_bucket_features(val_df)
    val = val.sort_values("dt").reset_index(drop=True)
    counts, global_p = init_counts(train, policy.dims)
    equity = float(initial)
    curve: list[float] = []
    actions = {"skip": 0, "down": 0, "base": 0, "up": 0, "up_strong": 0}
    pending_updates: list[tuple[tuple[Any, ...], bool]] = []
    stake_mults: list[float] = []
    router_ps: list[float] = []
    router_ns: list[int] = []
    key_arrays = [val[d].astype(str).to_numpy(dtype=object) if d in val.columns else np.full(len(val), "missing", dtype=object) for d in policy.dims]
    keys = list(zip(*key_arrays))
    won_arr = val["won"].astype(bool).to_numpy()
    dt_arr = pd.to_datetime(val["dt"], utc=True, errors="coerce").to_numpy()
    pred_arr = val.get("pred_up15", pd.Series(False, index=val.index)).astype(bool).to_numpy()
    label_arr = val.get("label_up", pd.Series(False, index=val.index)).astype(bool).to_numpy()
    if actual_price_col and actual_price_col in val.columns:
        price_arr = pd.to_numeric(val[actual_price_col], errors="coerce").fillna(float(buy_price)).to_numpy(dtype=float)
        price_arr = np.where(price_arr > 0, price_arr, float(buy_price))
    else:
        price_arr = np.full(len(val), float(buy_price), dtype=float)
    actual_cost_arr = pd.to_numeric(val.get("actual_cost", pd.Series(0.0, index=val.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    actual_pnl_arr = pd.to_numeric(val.get("actual_pnl", pd.Series(0.0, index=val.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    actual_shares_arr = pd.to_numeric(val.get("actual_shares", pd.Series(0.0, index=val.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    selected_dt: list[Any] = []
    selected_pred: list[bool] = []
    selected_label: list[bool] = []
    selected_won: list[bool] = []
    selected_mult: list[float] = []
    selected_actual_cost = 0.0
    selected_actual_pnl = 0.0
    selected_actual_shares = 0.0
    for i, (k, won) in enumerate(zip(keys, won_arr)):
        # Results are not instantly usable in live. Apply updates with a small lag.
        if len(pending_updates) > policy.update_lag:
            k_old, won_old = pending_updates.pop(0)
            c = counts.setdefault(k_old, [0, 0])
            c[0] += int(won_old)
            c[1] += 1
        p, n = p_hat_for(counts, k, policy.prior_strength, global_p)
        row_buy_price = float(price_arr[i])
        mult, action = multiplier_for(policy, p, n, row_buy_price)
        pending_updates.append((k, bool(won)))
        actions[action] += 1
        if mult <= 0:
            continue
        stake = equity * float(stake_fraction) * float(mult)
        if bool(won):
            equity += stake * (1.0 / row_buy_price - 1.0)
        else:
            equity -= stake
        equity = max(0.0, equity)
        curve.append(equity)
        stake_mults.append(mult)
        router_ps.append(p)
        router_ns.append(n)
        selected_dt.append(dt_arr[i])
        selected_pred.append(bool(pred_arr[i]))
        selected_label.append(bool(label_arr[i]))
        selected_won.append(bool(won))
        selected_mult.append(float(mult))
        selected_actual_cost += float(actual_cost_arr[i])
        selected_actual_pnl += float(actual_pnl_arr[i])
        selected_actual_shares += float(actual_shares_arr[i])
    arr = np.asarray(curve, dtype=float)
    mxdd, mxdd_pct = max_drawdown(arr)
    wins = int(np.asarray(selected_won, dtype=bool).sum()) if selected_won else 0
    losses = int(len(selected_won) - wins)
    ret_dd = round(float(equity - initial) / mxdd, 6) if mxdd > 1e-12 else (999.0 if equity > initial else 0.0)
    actual_cost = selected_actual_cost
    actual_pnl = selected_actual_pnl
    weighted_buy = selected_actual_cost / selected_actual_shares if selected_actual_shares else 0.0
    hash_rows = [
        {"dt": str(dt), "pred_up15": pred, "label_up": label, "won": won, "stake_mult": mult}
        for dt, pred, label, won, mult in zip(selected_dt, selected_pred, selected_label, selected_won, selected_mult)
    ]
    return {
        "name": name,
        "window": window,
        "buyPrice": float(buy_price),
        "trades": int(len(selected_won)),
        "wins": wins,
        "losses": losses,
        "winRatePct": round(100.0 * wins / len(selected_won), 6) if len(selected_won) else 0.0,
        "avgStakeMultiplier": round(float(np.mean(stake_mults)), 6) if stake_mults else 0.0,
        "compoundPnl": round(float(equity - initial), 6),
        "endingBankroll": round(float(equity), 6),
        "maxDrawdownUsd": round(float(mxdd), 6),
        "maxDrawdownPct": round(float(mxdd_pct) * 100.0, 6),
        "returnDrawdownRatio": ret_dd,
        "monthlyPositiveRatio": monthly_positive_ratio(arr, pd.Series(selected_dt)) if selected_dt else 0.0,
        "downsizeCount": int(actions["down"]),
        "upsizeCount": int(actions["up"] + actions["up_strong"]),
        "skipCount": int(actions["skip"]),
        "actionCounts": actions,
        "avgRouterP": round(float(np.mean(router_ps)), 6) if router_ps else 0.0,
        "avgBucketN": round(float(np.mean(router_ns)), 2) if router_ns else 0.0,
        "actualOfficialCost": round(actual_cost, 6),
        "actualOfficialPnl": round(actual_pnl, 6),
        "weightedAvgBuyPrice": round(weighted_buy, 6),
        "setHash": stable_hash(hash_rows) if hash_rows else "empty",
    }


def fixed_061_metrics(rows: pd.DataFrame, name: str, window: str, initial: float, buy_price: float, stake_fraction: float, actual_price_col: str | None = None) -> dict[str, Any]:
    pol = SizerPolicy(name="fixed", dims=("score_bin", "direction"), prior_strength=100, min_count=999999, weak_cut=0, strong_cut=9, very_strong_cut=9, weak_mult=1, strong_mult=1, very_strong_mult=1, skip_cut=None, update_lag=2)
    return simulate_online(rows.iloc[:0].copy(), rows, pol, name, window, initial, buy_price, stake_fraction, actual_price_col=actual_price_col)


def policy_grid() -> list[SizerPolicy]:
    dims_grid = [
        ("score_bin", "direction"),
        ("score_bin_fine", "direction"),
        ("score_bin", "direction", "hour_bucket"),
        ("score_bin", "direction", "1h_trend_bucket"),
        ("score_bin", "direction", "4h_trend_bucket"),
        ("score_bin", "direction", "1h_trend_bucket", "4h_trend_bucket"),
        ("score_bin", "direction", "1h_pos_bucket"),
        ("score_bin", "direction", "4h_pos_bucket"),
        ("score_bin", "direction", "1h_range_bucket", "4h_range_bucket"),
        ("score_bin", "direction", "1h_vol_bucket", "4h_vol_bucket"),
        ("score_bin", "direction", "cluster_hits"),
        ("score_bin", "direction", "shock_any"),
        ("score_bin", "direction", "same_stack"),
        ("score_bin", "direction", "hour_bucket", "1h_trend_bucket"),
        ("score_bin", "direction", "hour_bucket", "cluster_hits"),
        ("score_bin_fine", "direction", "1h_trend_bucket", "4h_pos_bucket"),
    ]
    profiles = [
        ("gentle", 0.550, 0.565, 0.585, 0.75, 1.10, 1.25, None, 999999),
        ("balanced", 0.550, 0.570, 0.595, 0.50, 1.20, 1.50, None, 999999),
        ("conservative", 0.545, 0.570, 0.600, 0.50, 1.10, 1.25, None, 999999),
        ("skip_soft", 0.550, 0.570, 0.595, 0.75, 1.15, 1.35, 0.535, 150),
        ("skip_mid", 0.5525, 0.575, 0.600, 0.50, 1.15, 1.35, 0.540, 120),
        ("skip_strict", 0.555, 0.580, 0.610, 0.50, 1.20, 1.50, 0.545, 80),
    ]
    out: list[SizerPolicy] = []
    for dims in dims_grid:
        for prior in [50, 100, 200, 400, 800]:
            for min_count in [20, 50, 100, 200]:
                for lag in [1, 2, 4]:
                    for label, weak, strong, very, wmult, smult, vsmult, skip_cut, skip_min in profiles:
                        out.append(SizerPolicy(name=f"{label}_{stable_hash({'d': dims, 'p': prior, 'm': min_count, 'l': lag})}", dims=dims, prior_strength=prior, min_count=min_count, weak_cut=weak, strong_cut=strong, very_strong_cut=very, weak_mult=wmult, strong_mult=smult, very_strong_mult=vsmult, skip_cut=skip_cut, skip_min_count=skip_min, update_lag=lag))
    seen = set(); uniq = []
    for p in out:
        k = p.key()
        if k not in seen:
            seen.add(k); uniq.append(p)
    return uniq


def score_candidate(c: dict[str, Any], official_base: dict[str, Any] | None) -> float:
    rows = {r["window"]: r for r in c.get("rows", [])}
    if set(rows) != {"180d", "365d"}:
        return -1e18
    b180, b365 = BASELINE_061["180d"], BASELINE_061["365d"]
    official_bonus = 0.0
    off = c.get("officialReplay")
    if official_base and off:
        official_bonus = (off.get("actualOfficialPnl", 0.0) - official_base.get("actualOfficialPnl", 0.0)) * 25 + (official_base.get("maxDrawdownUsd", 0.0) - off.get("maxDrawdownUsd", 0.0)) * 3
    return (
        (rows["180d"]["compoundPnl"] - b180["compoundPnl"]) / max(1.0, b180["compoundPnl"]) * 2500
        + (rows["365d"]["compoundPnl"] - b365["compoundPnl"]) / max(1.0, b365["compoundPnl"]) * 4500
        - max(0.0, rows["180d"]["maxDrawdownUsd"] / b180["maxDrawdownUsd"] - 1.05) * 500
        - max(0.0, rows["365d"]["maxDrawdownUsd"] / b365["maxDrawdownUsd"] - 1.05) * 900
        + (rows["365d"].get("avgStakeMultiplier", 1.0) - 1.0) * 25
        + official_bonus
    )


def pass_gate(c: dict[str, Any], official_base: dict[str, Any] | None) -> tuple[bool, list[str]]:
    rows = {r["window"]: r for r in c.get("rows", [])}
    reasons = []
    for w in ["180d", "365d"]:
        if w not in rows:
            reasons.append(f"{w}_missing"); continue
        b = BASELINE_061[w]
        if rows[w]["trades"] < max(100 if w == "180d" else 200, int(b["trades"] * 0.70)):
            reasons.append(f"{w}_too_few")
        if rows[w]["compoundPnl"] <= b["compoundPnl"]:
            reasons.append(f"{w}_pnl_not_above_061")
        if rows[w]["maxDrawdownUsd"] > b["maxDrawdownUsd"] * 1.05:
            reasons.append(f"{w}_dd_too_high")
    off = c.get("officialReplay")
    if official_base and off:
        if off.get("actualOfficialPnl", -1e9) < official_base.get("actualOfficialPnl", 0.0) and off.get("maxDrawdownUsd", 1e9) >= official_base.get("maxDrawdownUsd", 0.0):
            reasons.append("official_worse_pnl_and_not_lower_dd")
    return not reasons, reasons


def evaluate_policy(policy: SizerPolicy, train180: pd.DataFrame, val180: pd.DataFrame, train365: pd.DataFrame, val365: pd.DataFrame, actual_train: pd.DataFrame | None, actual_rows: pd.DataFrame | None, official_base: dict[str, Any] | None) -> dict[str, Any]:
    rows = [
        simulate_online(train180, val180, policy, policy.key(), "180d", START_BANKROLL, BUY_PRICE, STAKE_PCT),
        simulate_online(train365, val365, policy, policy.key(), "365d", START_BANKROLL, BUY_PRICE, STAKE_PCT),
    ]
    out = {"name": policy.key(), "policy": asdict(policy), "rows": rows}
    if actual_train is not None and actual_rows is not None and not actual_rows.empty:
        cfg = json.loads(actual.CONFIG.read_text(encoding="utf-8"))
        initial = float(cfg.get("base_capital_usd") or 847.091209)
        stake_fraction = float(cfg.get("stake_fraction") or 0.01)
        out["officialReplay"] = simulate_online(actual_train, actual_rows, policy, policy.key(), "official_actual", initial, BUY_PRICE, stake_fraction, actual_price_col="actual_avg_price")
    ok, reasons = pass_gate(out, official_base)
    out["passed"] = ok
    out["failReasons"] = reasons
    out["score"] = score_candidate(out, official_base)
    return out


def run_audit(period_vals, atom_store) -> dict[str, Any]:
    checks = []
    for w, period in [("180d", "validation_180d"), ("365d", "validation_365d")]:
        rows = current_061_selected(period_vals, atom_store, period)
        m = fixed_061_metrics(rows, "current061", w, START_BANKROLL, BUY_PRICE, STAKE_PCT)
        b = BASELINE_061[w]
        checks.append({"window": w, "trades": m["trades"], "pnl": m["compoundPnl"], "expectedPnl": b["compoundPnl"], "maxDrawdown": m["maxDrawdownUsd"], "expectedDrawdown": b["maxDrawdownUsd"], "passed": m["trades"] == b["trades"] and abs(m["compoundPnl"] - b["compoundPnl"]) < 1e-6 and abs(m["maxDrawdownUsd"] - b["maxDrawdownUsd"]) < 1e-6})
    audit = {"generatedAt": now_iso(), "beijingTime": bj_now(), "researchOnlyNoLiveChange": True, "timePolicy": "closed_candle_available_at_v2", "baselineReplay": checks, "passed": all(x["passed"] for x in checks), "notes": "Bayesian buckets are walk-forward and updated only after a configurable completed-trade lag."}
    write_json(OUT_AUDIT_JSON, audit)
    write_text(OUT_AUDIT_MD, "# 061 贝叶斯在线校准 + 动态仓位：审计\n\n" + f"- 北京时间：`{audit['beijingTime']}`\n- live动作：`无，研究只读`\n- 当前061复现：`{checks}`\n- 审计通过：`{audit['passed']}`\n")
    return audit


def format_compare(payload: dict[str, Any]) -> str:
    lines = [
        "# 061 贝叶斯在线校准 + 动态仓位：对比表",
        "",
        f"- 北京时间：`{payload['beijingTime']}`",
        "- live动作：`无，研究只读`",
        "- 口径：`850U初始 / 买价0.55 / 满成交 / 基础仓位1% / 动态仓位只用过去样本校准`",
        "",
        "## 180天 / 365天",
        "",
        "|配置|窗口|交易数|胜/负|胜率|平均仓位倍数|盈亏|期末资金|最大回撤|收益回撤比|降仓|加仓|跳过|哈希|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for w, b in BASELINE_061.items():
        lines.append(f"|当前061固定1%|{w}|{b['trades']}|{b['wins']}/{b['losses']}|{b['winRatePct']:.2f}%|1.000|{b['compoundPnl']:,.2f}|{b['endingBankroll']:,.2f}|{b['maxDrawdownUsd']:,.2f}|{b['returnDrawdownRatio']:.2f}|0|0|0|`{b['setHash']}`|")
    for label, c in [("动态仓位最强", payload.get("bestSizerCandidate")), ("动态仓位+软跳过最强", payload.get("bestSkipCandidate")), ("综合最强", payload.get("bestCandidate"))]:
        if not c: continue
        for r in c.get("rows", []):
            lines.append(f"|{label}|{r['window']}|{r['trades']}|{r['wins']}/{r['losses']}|{r['winRatePct']:.2f}%|{r['avgStakeMultiplier']:.3f}|{r['compoundPnl']:,.2f}|{r['endingBankroll']:,.2f}|{r['maxDrawdownUsd']:,.2f}|{r['returnDrawdownRatio']:.2f}|{r['downsizeCount']}|{r['upsizeCount']}|{r['skipCount']}|`{r['setHash']}`|")
    lines += ["", "## 061真实交易官方实际订单回放", "", "|配置|真实订单数|胜/负|胜率|平均仓位倍数|复利重放盈亏|官方实际盈亏|最大回撤|加仓|降仓|跳过|加权买价|哈希|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    ob = payload.get("officialBaselineCandidate")
    if ob:
        lines.append(f"|当前061官方单固定1%|{ob['trades']}|{ob['wins']}/{ob['losses']}|{ob['winRatePct']:.2f}%|1.000|{ob['compoundPnl']:+,.2f}|{ob.get('actualOfficialPnl',0):+,.2f}|{ob['maxDrawdownUsd']:,.2f}|0|0|0|{ob.get('weightedAvgBuyPrice',0):.4f}|`{ob['setHash']}`|")
    for label, c in [("动态仓位最强", payload.get("bestSizerCandidate")), ("动态仓位+软跳过最强", payload.get("bestSkipCandidate")), ("综合最强", payload.get("bestCandidate"))]:
        r = (c or {}).get("officialReplay")
        if not r: continue
        lines.append(f"|{label}|{r['trades']}|{r['wins']}/{r['losses']}|{r['winRatePct']:.2f}%|{r['avgStakeMultiplier']:.3f}|{r['compoundPnl']:+,.2f}|{r.get('actualOfficialPnl',0):+,.2f}|{r['maxDrawdownUsd']:,.2f}|{r['upsizeCount']}|{r['downsizeCount']}|{r['skipCount']}|{r.get('weightedAvgBuyPrice',0):.4f}|`{r['setHash']}`|")
    return "\n".join(lines) + "\n"


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    enriched, atom_store, period_vals = base.build_truth()
    audit = run_audit(period_vals, atom_store)
    if not audit["passed"]:
        raise SystemExit("baseline audit failed")

    train180 = add_bucket_features(current_061_selected(period_vals, atom_store, "gate_train_for_180d"))
    train365 = add_bucket_features(current_061_selected(period_vals, atom_store, "gate_train_for_365d"))
    val180 = add_bucket_features(current_061_selected(period_vals, atom_store, "validation_180d"))
    val365 = add_bucket_features(current_061_selected(period_vals, atom_store, "validation_365d"))

    actual_rows = actual.load_actual_logical_rows(strict_061=False)
    actual_train = None
    official_base = None
    if not actual_rows.empty:
        min_dt = pd.to_datetime(actual_rows["dt"], utc=True).min()
        hist = enriched[pd.to_datetime(enriched["dt"], utc=True) < min_dt].sort_values("dt").drop_duplicates("dt", keep="last").reset_index(drop=True)
        hist = shock.enrich_shock_features(hist)
        # Use the live-deployable historical 061 rows as the prior for actual replay.
        masks = base.base.atom_masks(hist)
        cond = base.base.condition_for_candidate(masks, "unused", base.base.CURRENT_061_PARAMS) if False else None
        hist_atom = base.base.atom_masks(hist)
        condition = base.base.condition_for_candidate({"h": hist_atom}, "h", base.base.CURRENT_061_PARAMS)
        score = pd.to_numeric(hist["score15"], errors="coerce").fillna(0.0).to_numpy()
        actual_train = add_bucket_features(hist[((~condition) | (score >= float(base.base.CURRENT_061_PARAMS["shock_score_min"])))].copy().reset_index(drop=True))
        actual_rows_enriched = shock.enrich_shock_features(actual_rows[["dt", "label_up", "pred_up15", "direction", "score15", "won", "period_name", "market_slug", "actual_cost", "actual_shares", "actual_avg_price", "actual_pnl"]].copy())
        actual_rows = add_bucket_features(actual_rows_enriched.sort_values("dt").reset_index(drop=True))
        cfg = json.loads(actual.CONFIG.read_text(encoding="utf-8"))
        initial = float(cfg.get("base_capital_usd") or 847.091209)
        stake_fraction = float(cfg.get("stake_fraction") or 0.01)
        official_base = fixed_061_metrics(actual_rows, "current_061_official", "official_actual", initial, BUY_PRICE, stake_fraction, actual_price_col="actual_avg_price")

    policies = policy_grid()
    candidates = []
    for i, pol in enumerate(policies, start=1):
        c = evaluate_policy(pol, train180, val180, train365, val365, actual_train, actual_rows, official_base)
        candidates.append(c)
        if i % 500 == 0:
            top = sorted(candidates, key=lambda x: x["score"], reverse=True)[:50]
            write_json(REPORTS / "top159_061_bayesian_sizer_checkpoint_latest.json", {"generatedAt": now_iso(), "beijingTime": bj_now(), "done": i, "total": len(policies), "top": top})
    candidates.sort(key=lambda x: x["score"], reverse=True)
    strict = [c for c in candidates if c.get("passed")]
    sizer = [c for c in candidates if c["policy"].get("skip_cut") is None]
    skip = [c for c in candidates if c["policy"].get("skip_cut") is not None]
    best = strict[0] if strict else candidates[0]
    best_sizer = sorted(sizer, key=lambda x: x["score"], reverse=True)[0] if sizer else None
    best_skip = sorted(skip, key=lambda x: x["score"], reverse=True)[0] if skip else None
    payload = {
        "generatedAt": now_iso(),
        "beijingTime": bj_now(),
        "researchOnlyNoLiveChange": True,
        "audit": audit,
        "policyCount": len(policies),
        "strictPass": len(strict),
        "baseline061": BASELINE_061,
        "officialBaselineCandidate": official_base,
        "bestCandidate": best,
        "bestSizerCandidate": best_sizer,
        "bestSkipCandidate": best_skip,
        "topCandidates": candidates[:200],
        "verdict": {
            "status": "candidate_passed" if strict else "no_candidate_passed",
            "message": "贝叶斯动态仓位找到严格通过候选；仍只建议影子验证，不直接改真钱。" if strict else "贝叶斯动态仓位没有找到同时打过061和官方回放的严格候选；当前真钱继续保留061。",
        },
    }
    write_json(OUT_LEADERBOARD_JSON, payload)
    write_text(OUT_LEADERBOARD_MD, "# 061 贝叶斯在线校准 + 动态仓位：排行榜\n\n" + f"- 北京时间：`{payload['beijingTime']}`\n- 候选数：`{len(policies)}`；严格通过：`{len(strict)}`\n\n" + "|排名|候选|180盈亏/回撤|365盈亏/回撤|官方盈亏/回撤|通过|失败原因|\n|---:|---|---:|---:|---:|---|---|\n" + "\n".join([f"|{i+1}|`{c['name']}`|{c['rows'][0]['compoundPnl']:,.2f}/{c['rows'][0]['maxDrawdownUsd']:,.2f}|{c['rows'][1]['compoundPnl']:,.2f}/{c['rows'][1]['maxDrawdownUsd']:,.2f}|{(c.get('officialReplay') or {}).get('actualOfficialPnl',0):+,.2f}/{(c.get('officialReplay') or {}).get('maxDrawdownUsd',0):,.2f}|{c.get('passed')}|`{','.join(c.get('failReasons') or [])}`|" for i,c in enumerate(candidates[:80])]) + "\n")
    write_json(OUT_COMPARE_JSON, payload)
    write_text(OUT_COMPARE_MD, format_compare(payload))
    write_json(OUT_VERDICT_JSON, payload["verdict"] | {"bestCandidate": best, "bestSizerCandidate": best_sizer, "bestSkipCandidate": best_skip})
    write_text(OUT_VERDICT_MD, f"# 061 贝叶斯在线校准 + 动态仓位：唯一结论\n\n- 状态：`{payload['verdict']['status']}`\n- 结论：{payload['verdict']['message']}\n")
    print(OUT_COMPARE_MD)
    print(json.dumps({"status": payload['verdict']['status'], "policyCount": len(policies), "strictPass": len(strict), "best": best.get('name')}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
