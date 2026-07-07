#!/usr/bin/env python3
from __future__ import annotations

"""BTC 061-style method search and ETH061 fair comparison.

Research-only. This script does not mutate live configs, ledgers, orders,
claim state, monitor settings, or current ETH061 live trading.

Locked metric:
  - 850U initial bankroll
  - stake 1% current bankroll per selected market
  - buy price 0.55 primary, full fill
  - closed-candle-visible feature alignment inherited from audited builders
"""

import concurrent.futures as cf
import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from itertools import combinations, product
from pathlib import Path
from typing import Any

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(key, os.environ.get("BTC061_THREADS", "1"))

import numpy as np
import pandas as pd

ROOT = Path("/Users/mac/polyfun")
REPORTS = ROOT / "reports"
RAW_GROWTH_SCRIPT = ROOT / "polyfun-next" / "scripts" / "run_crypto_raw_growth_multimarket_search.py"
PRECISION_SCRIPT = ROOT / "polyfun-next" / "scripts" / "run_top159_061_precision_hyperopt.py"

START_BANKROLL = 850.0
STAKE_PCT = 0.01
PRIMARY_BUY_PRICE = 0.55
BUY_PRICES = [0.55, 0.60]
RNG_SEED = 20260509
WORKERS = int(os.environ.get("BTC061_WORKERS", "6"))
THREADS = int(os.environ.get("BTC061_THREADS", "1"))
MAX_SECONDS = int(os.environ.get("BTC061_MAX_SECONDS", str(6 * 3600)))
MAX_MAIN_PARAMS = int(os.environ.get("BTC061_MAX_MAIN_PARAMS", "0"))
TOP_MAIN_FOR_FILTER = int(os.environ.get("BTC061_TOP_MAIN_FOR_FILTER", "10"))
MAX_FILTER_CANDIDATES_PER_MAIN = int(os.environ.get("BTC061_MAX_FILTER_CANDIDATES_PER_MAIN", "9000"))

OUT_AUDIT_MD = REPORTS / "btc061_method_bug_audit_latest.md"
OUT_AUDIT_JSON = REPORTS / "btc061_method_bug_audit_latest.json"
OUT_RESULTS = REPORTS / "btc061_method_results_latest.jsonl"
OUT_CHECKPOINT = REPORTS / "btc061_method_checkpoint_latest.json"
OUT_LEADERBOARD = REPORTS / "btc061_method_leaderboard_latest.json"
OUT_COMPARE_MD = REPORTS / "btc061_vs_eth061_180_365_archive_compare_latest.md"
OUT_COMPARE_JSON = REPORTS / "btc061_vs_eth061_180_365_archive_compare_latest.json"
OUT_VERDICT_MD = REPORTS / "btc061_method_unique_verdict_latest.md"
OUT_VERDICT_JSON = REPORTS / "btc061_method_unique_verdict_latest.json"

BASELINE_ETH061 = {
    "180d": {"trades": 3942, "wins": 2324, "losses": 1618, "winRatePct": 58.954845, "compoundPnl": 11494.591625, "endingBankroll": 12344.591625, "maxDrawdownUsd": 1481.713373, "returnDrawdownRatio": 7.757635, "setHash": "b35313c05e5b66d2"},
    "365d": {"trades": 9152, "wins": 5246, "losses": 3906, "winRatePct": 57.320804, "compoundPnl": 27033.917055, "endingBankroll": 27883.917055, "maxDrawdownUsd": 3499.248929, "returnDrawdownRatio": 7.725634, "setHash": "ced1cf82642d8f0d"},
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


raw_growth = load_module("btc061_raw_growth_base", RAW_GROWTH_SCRIPT)
precision = load_module("btc061_eth061_precision_base", PRECISION_SCRIPT)


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


def max_drawdown(equity: np.ndarray) -> tuple[float, float]:
    if equity.size == 0:
        return 0.0, 0.0
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    idx = int(np.argmax(dd))
    mx = float(dd[idx])
    denom = float(peak[idx]) if peak[idx] > 1e-12 else 0.0
    return mx, mx / denom if denom else 0.0


def monthly_positive_ratio(equity: np.ndarray, dt: pd.Series) -> float:
    if equity.size == 0:
        return 0.0
    f = pd.DataFrame({"dt": pd.to_datetime(dt, utc=True, errors="coerce"), "equity": equity}).dropna().sort_values("dt")
    if f.empty:
        return 0.0
    f["month"] = f["dt"].dt.to_period("M").astype(str)
    prev = START_BANKROLL
    vals: list[float] = []
    for _, g in f.groupby("month", sort=True):
        end = float(g["equity"].iloc[-1])
        vals.append(end - prev)
        prev = end
    return round(sum(v > 0 for v in vals) / len(vals), 6) if vals else 0.0


def curve_metrics(rows: pd.DataFrame, name: str, window: str, buy_price: float = PRIMARY_BUY_PRICE) -> dict[str, Any]:
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
    id_cols = [c for c in ["asset", "dt", "pred_up", "label_up", "won"] if c in sel.columns]
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
        "setHash": stable_hash(sel[id_cols].to_dict("records")) if len(sel) else "empty",
    }


def audit_eth061_replay() -> dict[str, Any]:
    enriched, atom_store, period_vals, truth = precision.build_truth()
    params = precision.current_061_params()
    replay = precision.evaluate_candidate(period_vals, atom_store, params)
    by = {r["window"]: r for r in replay.get("rows", []) if r.get("buyPrice") == 0.55}
    checks = []
    for w, b in BASELINE_ETH061.items():
        r = by.get(w)
        ok = bool(r)
        diffs: dict[str, float] = {}
        hash_warning = False
        if r:
            for k in ["trades", "wins", "losses", "compoundPnl", "maxDrawdownUsd"]:
                diffs[k] = abs(float(r.get(k, 0)) - float(b.get(k, 0)))
            hash_warning = r.get("setHash") != b.get("setHash")
            ok = all(v <= (1e-4 if k not in {"trades", "wins", "losses"} else 0.0) for k, v in diffs.items())
        checks.append({"window": w, "passed": ok, "baseline": b, "replay": r, "diffs": diffs, "setHashChangedWarning": hash_warning})
    return {
        "truth": truth,
        "checks": checks,
        "passed": all(c["passed"] for c in checks),
        "note": "setHash is warning-only here; economic replay metrics must match exactly. Latest 061 reports changed set hashes while trades/wins/pnl/drawdown remained identical.",
    }


def build_btc_frame() -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    df, features, truth = raw_growth.build_frame("BTC", "15m")
    # raw_growth build_frame already shifts current-timeframe features and asof-merges HTF shifted features.
    return df.sort_values("dt").reset_index(drop=True), features, truth


def feature_subset(features: list[str], mode: str) -> list[str]:
    return raw_growth.feature_subset(features, mode)


def split_train_val(df: pd.DataFrame, window: str, train_window: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    train, val = raw_growth.split_train_val(df, window, train_window)
    start = val["dt"].min()
    return train, val, start


def fit_predict(df: pd.DataFrame, features: list[str], params: dict[str, Any], window: str, gate_mode: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train, val, start = split_train_val(df, window, params["train_window"])
    if gate_mode:
        # Split past data into earlier model-train and later gate-train to avoid mining clusters from pure in-sample predictions.
        gate_days = 365 if window == "365d" else 180
        gate_start = start - pd.Timedelta(days=gate_days)
        model_train = train[train["dt"] < gate_start - pd.Timedelta(minutes=15)].copy()
        gate_val = train[train["dt"] >= gate_start].copy()
        if len(model_train) < 1000 or len(gate_val) < 100:
            # fallback to last 30% gate split, still not using validation rows
            cut = int(len(train) * 0.70)
            model_train = train.iloc[:cut].copy()
            gate_val = train.iloc[cut:].copy()
        model = raw_growth.fit_model(params["engine"], model_train, features, params)
        if model is None:
            return pd.DataFrame(), pd.DataFrame(), {"error": "gate_fit_failed"}
        prob = raw_growth.predict(model, gate_val, features)
        selected = raw_growth.select_rows(gate_val, prob, params)
        return selected, gate_val, {"modelTrainRows": int(len(model_train)), "gateRows": int(len(gate_val))}
    model = raw_growth.fit_model(params["engine"], train, features, params)
    if model is None:
        return pd.DataFrame(), pd.DataFrame(), {"error": "fit_failed"}
    prob = raw_growth.predict(model, val, features)
    selected = raw_growth.select_rows(val, prob, params)
    return selected, val, {"trainRows": int(len(train)), "validationRows": int(len(val))}


def atom_masks(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    n = len(rows)
    if n == 0:
        return {}
    out: dict[str, np.ndarray] = {}
    score = pd.to_numeric(rows.get("score"), errors="coerce").fillna(0.0)
    pred = rows.get("pred_up", pd.Series(False, index=rows.index)).astype(bool)
    label = rows.get("label_up", pd.Series(False, index=rows.index)).astype(bool)
    out["dir_UP"] = pred.to_numpy()
    out["dir_DOWN"] = (~pred).to_numpy()
    for lo, hi in [(0.55, 0.57), (0.57, 0.60), (0.60, 0.64)]:
        out[f"score_{lo:.2f}_{hi:.2f}"] = ((score >= lo) & (score < hi)).to_numpy()
    out["score_ge_064"] = (score >= 0.64).to_numpy()
    out["score_lt_058"] = (score < 0.58).to_numpy()
    dt = pd.to_datetime(rows["dt"], utc=True, errors="coerce").dt.tz_convert("Asia/Shanghai")
    hour = dt.dt.hour.fillna(-1).astype(int)
    for start in [0, 4, 8, 12, 16, 20]:
        out[f"bj_hour_{start:02d}_{start+3:02d}"] = ((hour >= start) & (hour <= start + 3)).to_numpy()

    def num(col: str, default: float = 0.0) -> pd.Series:
        return pd.to_numeric(rows[col], errors="coerce").fillna(default) if col in rows else pd.Series(default, index=rows.index)

    direction_sign = np.where(pred.to_numpy(), 1.0, -1.0)
    for prefix, ret_col, trend_col, bb_col, range_col, vol_col in [
        ("15m", "ret_1", "ema_8_32", "bb_pos", "range_16", "volume_z_16"),
        ("1h", "1h_ret_1", "1h_ema_8_32", "1h_bb_pos", "1h_range_16", "1h_volume_z_16"),
        ("4h", "4h_ret_1", "4h_ema_8_32", "4h_bb_pos", "4h_range_16", "4h_volume_z_16"),
    ]:
        ret = num(ret_col)
        trend = num(trend_col)
        bb = num(bb_col)
        rng = num(range_col)
        volz = num(vol_col)
        out[f"{prefix}_same_as_pred"] = (np.sign(ret.to_numpy()) == direction_sign).astype(bool)
        out[f"{prefix}_opposes_pred"] = (np.sign(ret.to_numpy()) == -direction_sign).astype(bool)
        out[f"{prefix}_trend_up"] = (trend > 0).to_numpy()
        out[f"{prefix}_trend_down"] = (trend < 0).to_numpy()
        out[f"{prefix}_trend_same_as_pred"] = (np.sign(trend.to_numpy()) == direction_sign).astype(bool)
        out[f"{prefix}_trend_opposes_pred"] = (np.sign(trend.to_numpy()) == -direction_sign).astype(bool)
        out[f"{prefix}_pos_high"] = (bb >= 0.75).to_numpy()
        out[f"{prefix}_pos_low"] = (bb <= -0.75).to_numpy()
        qvals = rng.rank(pct=True).fillna(0.5)
        out[f"{prefix}_rangeq_ge_055"] = (qvals >= 0.55).to_numpy()
        out[f"{prefix}_rangeq_ge_070"] = (qvals >= 0.70).to_numpy()
        out[f"{prefix}_vol_ge_1.1"] = (volz >= 1.1).to_numpy()
        out[f"{prefix}_vol_ge_1.6"] = (volz >= 1.6).to_numpy()
        out[f"{prefix}_terminal_chase"] = (out[f"{prefix}_same_as_pred"] & (((pred.to_numpy()) & (bb >= 0.75).to_numpy()) | ((~pred.to_numpy()) & (bb <= -0.75).to_numpy())))
    # Explicitly include correctness only for mining checks? No. Never use won/label atoms for candidate filters.
    _ = label
    return {k: np.asarray(v, dtype=bool) for k, v in out.items() if len(v) == n}


def mask_for_atoms(atoms: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    if not names:
        first = next(iter(atoms.values()))
        return np.ones_like(first, dtype=bool)
    out = atoms[names[0]].copy()
    for n in names[1:]:
        out &= atoms[n]
    return out


def cluster_stats(rows: pd.DataFrame, mask: np.ndarray) -> dict[str, Any]:
    won = rows["won"].astype(bool).to_numpy()
    n = int(mask.sum())
    wins = int((won & mask).sum())
    losses = n - wins
    return {"n": n, "wins": wins, "losses": losses, "winRatePct": round(100.0 * wins / n, 6) if n else 0.0, "loserMinusWinner": losses - wins}


def mine_bad_clusters(gate180: pd.DataFrame, gate365: pd.DataFrame, limit: int = 80) -> list[dict[str, Any]]:
    a180 = atom_masks(gate180)
    a365 = atom_masks(gate365)
    names = sorted(set(a180).intersection(a365))
    usable = []
    for name in names:
        n180 = int(a180[name].sum())
        n365 = int(a365[name].sum())
        if min(n180, n365) >= 30 and n180 < len(gate180) * 0.98 and n365 < len(gate365) * 0.98:
            usable.append(name)
    # Rank single atoms first; combine best atoms only.
    single_rank = []
    rows = []
    for name in usable:
        s180 = cluster_stats(gate180, a180[name])
        s365 = cluster_stats(gate365, a365[name])
        score = min(s180["loserMinusWinner"], s365["loserMinusWinner"]) * 10 + min(s180["n"], s365["n"]) * 0.01
        single_rank.append((score, name, s180, s365))
    single_rank.sort(reverse=True, key=lambda x: x[0])
    combos: list[tuple[str, ...]] = [(x[1],) for x in single_rank[:80]]
    top_atoms = [x[1] for x in single_rank[:45]]
    combos.extend(tuple(c) for c in combinations(top_atoms, 2))
    combos.extend(tuple(c) for c in combinations(top_atoms[:24], 3))
    seen = set()
    for combo in combos:
        key = "|".join(combo)
        if key in seen:
            continue
        seen.add(key)
        m180 = mask_for_atoms(a180, list(combo))
        m365 = mask_for_atoms(a365, list(combo))
        s180 = cluster_stats(gate180, m180)
        s365 = cluster_stats(gate365, m365)
        if min(s180["n"], s365["n"]) < 25:
            continue
        score = min(s180["loserMinusWinner"], s365["loserMinusWinner"]) * 12 + (100 - max(s180["winRatePct"], s365["winRatePct"])) + min(s180["n"], s365["n"]) * 0.015
        rows.append({"cluster_id": stable_hash(combo), "atoms": list(combo), "score": round(score, 6), "train180": s180, "train365": s365})
    rows.sort(key=lambda r: (r["score"], min(r["train180"]["loserMinusWinner"], r["train365"]["loserMinusWinner"])), reverse=True)
    return rows[:limit]


def generate_filter_params(clusters: list[dict[str, Any]], max_count: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    top = clusters[:70]
    thresholds = [0.555, 0.56, 0.565, 0.57, 0.575, 0.58, 0.59, 0.60, 0.62, 0.64]
    deducts = [0.005, 0.01, 0.015, 0.02, 0.03]
    for c in top:
        rows.append({"action": "hard_block", "clusters": [c["atoms"]], "min_cluster_hits": 1})
        for th in thresholds:
            rows.append({"action": "raise_score", "clusters": [c["atoms"]], "min_cluster_hits": 1, "score_min": th})
        for dd in deducts:
            rows.append({"action": "soft_deduct", "clusters": [c["atoms"]], "min_cluster_hits": 1, "deduct": dd, "base_score_min": 0.55})
    for a, b in combinations(top[:50], 2):
        clusters2 = [a["atoms"], b["atoms"]]
        rows.append({"action": "hard_block", "clusters": clusters2, "min_cluster_hits": 1})
        for th in [0.56, 0.57, 0.58, 0.60, 0.62, 0.64]:
            rows.append({"action": "raise_score", "clusters": clusters2, "min_cluster_hits": 1, "score_min": th})
        rows.append({"action": "hard_block", "clusters": clusters2, "min_cluster_hits": 2})
        for th in [0.58, 0.60, 0.62]:
            rows.append({"action": "raise_score", "clusters": clusters2, "min_cluster_hits": 2, "score_min": th})
    for a, b, c in combinations(top[:28], 3):
        clusters3 = [a["atoms"], b["atoms"], c["atoms"]]
        for min_hits in [1, 2]:
            rows.append({"action": "hard_block", "clusters": clusters3, "min_cluster_hits": min_hits})
            for th in [0.58, 0.60, 0.62]:
                rows.append({"action": "raise_score", "clusters": clusters3, "min_cluster_hits": min_hits, "score_min": th})
    # Directional variants around best clusters.
    for c in top[:35]:
        for up, down in product([0.56, 0.58, 0.60, 0.62], [0.56, 0.58, 0.60, 0.62]):
            rows.append({"action": "directional_raise", "clusters": [c["atoms"]], "min_cluster_hits": 1, "up_score_min": up, "down_score_min": down})
    dedup: dict[str, dict[str, Any]] = {}
    for r in rows:
        cid = stable_hash(r)
        r = dict(r)
        r["candidate_id"] = cid
        dedup[cid] = r
    return list(dedup.values())[:max_count]


def condition_mask(rows: pd.DataFrame, params: dict[str, Any]) -> np.ndarray:
    atoms = atom_masks(rows)
    masks = [mask_for_atoms(atoms, c) for c in params.get("clusters", [])]
    if not masks:
        return np.zeros(len(rows), dtype=bool)
    if int(params.get("min_cluster_hits", 1)) <= 1:
        out = masks[0].copy()
        for m in masks[1:]:
            out |= m
    else:
        votes = np.zeros(len(rows), dtype=np.int16)
        for m in masks:
            votes += m.astype(np.int16)
        out = votes >= int(params["min_cluster_hits"])
    return out


def apply_filter(rows: pd.DataFrame, params: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if rows.empty:
        return rows.copy(), {"blockedTrades": 0, "blockedWinners": 0, "blockedLosers": 0, "retentionRate": 0.0, "conditionTrades": 0}
    cond = condition_mask(rows, params)
    score = pd.to_numeric(rows["score"], errors="coerce").fillna(0.0).to_numpy()
    action = params.get("action")
    if action == "hard_block":
        keep = ~cond
    elif action == "raise_score":
        keep = (~cond) | (score >= float(params.get("score_min", 0.58)))
    elif action == "soft_deduct":
        keep = (score - np.where(cond, float(params.get("deduct", 0.01)), 0.0)) >= float(params.get("base_score_min", 0.55))
    elif action == "directional_raise":
        pred = rows["pred_up"].astype(bool).to_numpy()
        req = np.where(pred, float(params.get("up_score_min", 0.58)), float(params.get("down_score_min", 0.58)))
        keep = (~cond) | (score >= req)
    else:
        keep = np.ones(len(rows), dtype=bool)
    won = rows["won"].astype(bool).to_numpy()
    blocked = ~keep
    meta = {
        "blockedTrades": int(blocked.sum()),
        "blockedWinners": int((won & blocked).sum()),
        "blockedLosers": int((~won & blocked).sum()),
        "retentionRate": round(100.0 * int(keep.sum()) / max(1, len(keep)), 6),
        "conditionTrades": int(cond.sum()),
    }
    return rows.loc[keep].copy().reset_index(drop=True), meta


def main_param_grid() -> list[dict[str, Any]]:
    rows = []
    for train_window, feature_mode, threshold, band, max_score, max_features, vol_q, mp in product(
        ["1y", "3y", "5y", "full"],
        ["core_htf", "trend_multi", "wide"],
        [0.54, 0.55, 0.56, 0.57],
        ["all", "mid_only"],
        [0.76, 0.82],
        [80, 120, 180],
        [0.95, 0.999],
        [
            {"n_estimators": 100, "learning_rate": 0.035, "num_leaves": 20, "min_child_samples": 80, "reg_lambda": 1.0},
            {"n_estimators": 150, "learning_rate": 0.028, "num_leaves": 28, "min_child_samples": 100, "reg_lambda": 1.2},
        ],
    ):
        p = {"engine": "lightgbm", "train_window": train_window, "feature_mode": feature_mode, "threshold": threshold, "score_band": band, "max_score": max_score, "max_features": max_features, "vol_q": vol_q, "daily_cap": 0, **mp, "subsample": 0.88, "colsample_bytree": 0.88}
        rows.append(p)
    if MAX_MAIN_PARAMS and len(rows) > MAX_MAIN_PARAMS:
        step = max(1, len(rows) // MAX_MAIN_PARAMS)
        rows = rows[::step][:MAX_MAIN_PARAMS]
    return rows


def evaluate_main_param(args: tuple[pd.DataFrame, list[str], dict[str, Any]]) -> dict[str, Any]:
    df, all_features, params = args
    feats = feature_subset(all_features, params["feature_mode"])[: int(params.get("max_features", 120))]
    name = f"BTC_bare_{params['engine']}_{params['train_window']}_{params['feature_mode']}_thr{params['threshold']}_{stable_hash(params)}"
    try:
        if len(feats) < 8:
            return {"name": name, "params": params, "error": "too_few_features"}
        rows = []
        selections = {}
        for window in ["180d", "365d"]:
            selected, _val, info = fit_predict(df, feats, params, window, gate_mode=False)
            if selected.empty:
                return {"name": name, "params": params, "error": f"empty_{window}"}
            selected = selected.rename(columns={"pred_up": "pred_up"})
            selected["asset"] = "BTC"
            row = curve_metrics(selected, name, window, PRIMARY_BUY_PRICE)
            row.update({"asset": "BTC", "config": "BTC裸主模型", "featureCount": len(feats), "trainWindow": params["train_window"], "threshold": params["threshold"]})
            rows.append(row)
            selections[window] = selected
        return {"name": name, "params": params, "rows": rows, "featureCount": len(feats), "selections": {k: v.to_dict("records") for k, v in selections.items()}}
    except Exception as exc:
        return {"name": name, "params": params, "error": repr(exc)[:1000]}


def baseline_pass_score(candidate: dict[str, Any]) -> float:
    if "rows" not in candidate:
        return -1e18
    by = {r["window"]: r for r in candidate["rows"]}
    score = 0.0
    for w, b in BASELINE_ETH061.items():
        r = by.get(w, {})
        score += (float(r.get("compoundPnl", -1e9)) - b["compoundPnl"]) / max(abs(b["compoundPnl"]), 1.0) * 1000.0
        score += (float(r.get("winRatePct", 0.0)) - b["winRatePct"]) * 30.0
        score -= max(0.0, float(r.get("maxDrawdownUsd", 1e9)) - b["maxDrawdownUsd"]) / max(b["maxDrawdownUsd"], 1.0) * 500.0
    return score


def strict_vs_eth061(candidate: dict[str, Any]) -> tuple[bool, list[str]]:
    if "rows" not in candidate:
        return False, [candidate.get("error", "invalid")]
    by = {r["window"]: r for r in candidate["rows"]}
    reasons = []
    for w, b in BASELINE_ETH061.items():
        r = by[w]
        if int(r["trades"]) < (100 if w == "180d" else 200):
            reasons.append(f"{w}_trades_too_low")
        if float(r["compoundPnl"]) <= b["compoundPnl"]:
            reasons.append(f"{w}_pnl_not_above_eth061")
        if w == "365d" and float(r["winRatePct"]) < b["winRatePct"]:
            reasons.append("365d_winrate_below_eth061")
        if float(r["maxDrawdownUsd"]) > b["maxDrawdownUsd"] * 1.05:
            reasons.append(f"{w}_drawdown_too_high")
    return len(reasons) == 0, reasons


def evaluate_filter_candidate(args: tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    main, fp, selections_raw = args
    name = f"BTC061style_{main['params']['train_window']}_{main['params']['feature_mode']}_{main['params']['threshold']}_{fp['candidate_id']}"
    rows = []
    block_meta = {}
    try:
        for window in ["180d", "365d"]:
            base_rows = pd.DataFrame(selections_raw[window])
            filtered, meta = apply_filter(base_rows, fp)
            if filtered.empty:
                return {"name": name, "params": {"main": main["params"], "filter": fp}, "error": f"empty_{window}"}
            row = curve_metrics(filtered, name, window, PRIMARY_BUY_PRICE)
            row.update({"asset": "BTC", "config": "BTC061-style", "trainWindow": main["params"]["train_window"], "threshold": main["params"]["threshold"], **meta})
            rows.append(row)
            block_meta[window] = meta
        return {"name": name, "params": {"main": main["params"], "filter": fp}, "rows": rows, "blockMeta": block_meta, "mainName": main["name"]}
    except Exception as exc:
        return {"name": name, "params": {"main": main.get("params"), "filter": fp}, "error": repr(exc)[:1000]}


def build_filter_candidates_for_main(df: pd.DataFrame, all_features: list[str], main: dict[str, Any]) -> list[dict[str, Any]]:
    params = main["params"]
    feats = feature_subset(all_features, params["feature_mode"])[: int(params.get("max_features", 120))]
    gate180, _, _ = fit_predict(df, feats, params, "180d", gate_mode=True)
    gate365, _, _ = fit_predict(df, feats, params, "365d", gate_mode=True)
    if gate180.empty or gate365.empty:
        return []
    clusters = mine_bad_clusters(gate180, gate365, 90)
    return generate_filter_params(clusters, MAX_FILTER_CANDIDATES_PER_MAIN)


def archived_btc_metrics(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    # Historical archives are sparse and discontinuous. Use pure prediction only.
    rows = []
    paths = []
    for p in ROOT.rglob("prediction_trades*.json"):
        if p.is_file() and "node_modules" not in p.parts and ".venv" not in p.parts:
            paths.append(p)
    seen = set()
    old = []
    for path in sorted(paths):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, list):
            continue
        for r in raw:
            if not isinstance(r, dict) or r.get("mode") != "live" or r.get("status") != "executed" or str(r.get("symbol") or "").upper() != "BTC":
                continue
            slug = str(r.get("marketSlug") or "")
            if "btc-updown-15m-" not in slug:
                continue
            result = str(r.get("formalResult") or r.get("result") or "").lower()
            if result not in {"win", "lose"}:
                continue
            direction = str(r.get("direction") or r.get("tokenOutcome") or "").upper()
            if direction not in {"UP", "DOWN"}:
                continue
            try:
                dt = pd.to_datetime(int(slug.rsplit("-", 1)[-1]), unit="s", utc=True)
            except Exception:
                continue
            key = stable_hash([slug, direction, r.get("orderId"), r.get("txHash"), r.get("timestamp"), r.get("amount"), result])
            if key in seen:
                continue
            seen.add(key)
            won = result == "win"
            old_up = direction == "UP"
            actual_up = old_up if won else (not old_up)
            old.append({"dt": dt, "actualUp": actual_up})
    if not old or "params" not in candidate:
        return []
    old_df = pd.DataFrame(old).drop_duplicates("dt").sort_values("dt")
    # Recreate only the bare main prediction for archive; if filter exists, apply filter against predicted rows.
    df, all_features, _truth = build_btc_frame()
    main_params = candidate["params"].get("main", candidate["params"])
    feats = feature_subset(all_features, main_params["feature_mode"])[: int(main_params.get("max_features", 120))]
    start = old_df["dt"].min()
    train = df[df["dt"] < start - pd.Timedelta(minutes=15)].copy()
    if main_params["train_window"] != "full":
        days = {"1y": 365, "3y": 1095, "5y": 1825}.get(main_params["train_window"], 1095)
        train = train[train["dt"] >= start - pd.Timedelta(days=days)].copy()
    model = raw_growth.fit_model(main_params["engine"], train, feats, main_params)
    if model is None:
        return []
    vals = df[(df["dt"] >= old_df["dt"].min()) & (df["dt"] <= old_df["dt"].max())].copy()
    prob = raw_growth.predict(model, vals, feats)
    selected = raw_growth.select_rows(vals, prob, main_params)
    if candidate["params"].get("filter"):
        selected, _ = apply_filter(selected, candidate["params"]["filter"])
    merged = old_df.merge(selected[["dt", "pred_up", "score"]], on="dt", how="inner")
    if merged.empty:
        return []
    merged["asset"] = "BTC"
    merged["label_up"] = merged["actualUp"].astype(bool)
    merged["won"] = merged["pred_up"].astype(bool) == merged["label_up"].astype(bool)
    row = curve_metrics(merged, candidate["name"], "archived_real_pure_prediction", PRIMARY_BUY_PRICE)
    row.update({"scope": "全部归档BTC15m live去重", "oldRealMarkets": int(len(old_df)), "selectedTrades": int(len(merged))})
    return [row]


def render_outputs(payload: dict[str, Any]) -> None:
    write_json(OUT_CHECKPOINT, payload)
    write_json(OUT_LEADERBOARD, payload)
    write_json(OUT_COMPARE_JSON, payload)
    selected = payload.get("selected")
    verdict = payload.get("verdict") or {}
    write_json(OUT_VERDICT_JSON, verdict | {"selected": selected, "archiveRows": payload.get("archiveRows", [])})
    lines = [
        "# BTC 版 061 方法搜索：ETH061 公平对比",
        "",
        f"- 北京时间：`{payload['beijingTime']}`",
        "- 口径：`850U初始 / 每笔1% / 满成交 / 买价0.55`",
        "- live动作：`无，研究只读；ETH061真钱未改`",
        "",
        "## ETH061 基线 vs BTC 最强",
        "",
        "|配置|窗口|交易数|胜/负|胜率|盈亏|期末资金|最大回撤|收益回撤比|哈希|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for w, b in BASELINE_ETH061.items():
        lines.append(f"|ETH 当前061|{w}|{b['trades']}|{b['wins']}/{b['losses']}|{b['winRatePct']:.2f}%|{b['compoundPnl']:,.2f}|{b['endingBankroll']:,.2f}|{b['maxDrawdownUsd']:,.2f}|{b['returnDrawdownRatio']:.2f}|`{b['setHash']}`|")
    if selected and selected.get("rows"):
        for r in selected["rows"]:
            lines.append(f"|{r.get('config','BTC候选')}|{r['window']}|{r['trades']}|{r['wins']}/{r['losses']}|{r['winRatePct']:.2f}%|{r['compoundPnl']:,.2f}|{r['endingBankroll']:,.2f}|{r['maxDrawdownUsd']:,.2f}|{r['returnDrawdownRatio']:.2f}|`{r['setHash']}`|")
    else:
        lines.append("|BTC候选|-|-|-|-|-|-|-|-|无|")
    lines += ["", "## BTC 历史归档纯预测复核", "", "|范围|旧真实市场数|选中|胜/负|胜率|盈亏|最大回撤|", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in payload.get("archiveRows", []):
        lines.append(f"|{r.get('scope','archive')}|{r.get('oldRealMarkets',0)}|{r.get('selectedTrades',r.get('trades',0))}|{r.get('wins',0)}/{r.get('losses',0)}|{r.get('winRatePct',0):.2f}%|{r.get('compoundPnl',0):,.2f}|{r.get('maxDrawdownUsd',0):,.2f}|")
    if not payload.get("archiveRows"):
        lines.append("|无BTC归档可复核|0|0|0/0|0.00%|0.00|0.00|")
    lines += ["", "## 前20候选", "", "|排名|配置|365胜率|365盈亏|365回撤|通过|失败原因|", "|---:|---|---:|---:|---:|---|---|"]
    for i, c in enumerate(payload.get("topCandidates", [])[:20], 1):
        by = {r["window"]: r for r in c.get("rows", [])}
        r365 = by.get("365d", {})
        lines.append(f"|{i}|{c.get('name','')}|{float(r365.get('winRatePct',0)):.2f}%|{float(r365.get('compoundPnl',0)):,.2f}|{float(r365.get('maxDrawdownUsd',0)):,.2f}|{c.get('passed',False)}|{','.join(c.get('reasons',[]))}|")
    lines += ["", "## 唯一结论", "", f"- 状态：`{verdict.get('status')}`", f"- 结论：{verdict.get('message')}"]
    write_text(OUT_COMPARE_MD, "\n".join(lines) + "\n")
    write_text(OUT_VERDICT_MD, "\n".join(["# BTC 版 061 方法搜索：唯一结论", "", f"- 状态：`{verdict.get('status')}`", f"- 结论：{verdict.get('message')}", ""]) + "\n")


def main() -> int:
    started = time.time()
    REPORTS.mkdir(parents=True, exist_ok=True)
    if OUT_RESULTS.exists() and os.environ.get("BTC061_RESET", "1") == "1":
        OUT_RESULTS.unlink()
    eth_audit = audit_eth061_replay()
    df, all_features, truth = build_btc_frame()
    forbidden_hits = [c for c in all_features if c in set(getattr(raw_growth.base, "FORBIDDEN_FEATURES", set())) or c.startswith(tuple(getattr(raw_growth.base, "FORBIDDEN_PREFIXES", ())))]
    audit = {"generatedAt": now_iso(), "beijingTime": bj_now(), "researchOnlyNoLiveChange": True, "eth061Replay": eth_audit, "btcDataTruth": truth, "forbiddenFeatureHits": forbidden_hits, "timePolicy": "closed_candle_available_at_v2 via shifted current bar features and backward-asof shifted HTF features", "passed": bool(eth_audit["passed"]) and not forbidden_hits}
    write_json(OUT_AUDIT_JSON, audit)
    write_text(OUT_AUDIT_MD, "\n".join(["# BTC061 方法搜索：回测审计", "", f"- 北京时间：`{audit['beijingTime']}`", f"- ETH061复放通过：`{eth_audit['passed']}`", f"- BTC禁用字段命中：`{forbidden_hits}`", f"- 审计通过：`{audit['passed']}`", ""]) + "\n")
    if not audit["passed"]:
        payload = {"generatedAt": now_iso(), "beijingTime": bj_now(), "audit": audit, "selected": None, "topCandidates": [], "archiveRows": [], "verdict": {"status": "audit_failed", "message": "ETH061复放或BTC禁用字段审计失败；未进入BTC搜索。"}}
        render_outputs(payload)
        return 2

    main_params = main_param_grid()
    deadline = started + MAX_SECONDS
    main_results: list[dict[str, Any]] = []
    with cf.ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(evaluate_main_param, (df, all_features, p)) for p in main_params]
        for i, fut in enumerate(cf.as_completed(futs), 1):
            main_results.append(fut.result())
            if time.time() > deadline:
                break
    for r in main_results:
        if "rows" in r:
            ok, reasons = strict_vs_eth061(r)
            r["passed"] = ok
            r["reasons"] = reasons
            r["score"] = baseline_pass_score(r)
    valid_main = [r for r in main_results if "rows" in r]
    valid_main.sort(key=lambda x: (x.get("passed", False), x.get("score", -1e18)), reverse=True)

    filter_results: list[dict[str, Any]] = []
    for main_cand in valid_main[:TOP_MAIN_FOR_FILTER]:
        if time.time() > deadline:
            break
        fps = build_filter_candidates_for_main(df, all_features, main_cand)
        if not fps:
            continue
        args = [(main_cand, fp, main_cand["selections"]) for fp in fps]
        with cf.ProcessPoolExecutor(max_workers=WORKERS) as ex:
            futs = [ex.submit(evaluate_filter_candidate, a) for a in args]
            for fut in cf.as_completed(futs):
                rr = fut.result()
                if "rows" in rr:
                    ok, reasons = strict_vs_eth061(rr)
                    rr["passed"] = ok
                    rr["reasons"] = reasons
                    rr["score"] = baseline_pass_score(rr)
                filter_results.append(rr)
                if len(filter_results) % 1000 == 0:
                    with OUT_RESULTS.open("a", encoding="utf-8") as fh:
                        for x in filter_results[-1000:]:
                            slim = {k: x.get(k) for k in ["name", "params", "rows", "blockMeta", "passed", "reasons", "score", "error"]}
                            fh.write(json.dumps(slim, ensure_ascii=False, sort_keys=True, default=str) + "\n")
                if time.time() > deadline:
                    break

    all_candidates = [r for r in valid_main + filter_results if "rows" in r]
    all_candidates.sort(key=lambda x: (x.get("passed", False), x.get("score", -1e18)), reverse=True)
    selected = all_candidates[0] if all_candidates else None
    archive_rows = archived_btc_metrics(selected) if selected else []
    if selected and selected.get("passed"):
        verdict = {"status": "btc_candidate_beats_eth061", "message": "找到BTC 061-style候选；只能进入影子验证，不改真钱。"}
    else:
        verdict = {"status": "no_btc_candidate_beats_eth061", "message": "BTC裸主模型和BTC 061-style过滤都没有严格打败ETH061；当前真钱继续保留ETH061。"}
    payload = {"generatedAt": now_iso(), "beijingTime": bj_now(), "finished": True, "elapsedSeconds": round(time.time() - started, 3), "workers": WORKERS, "mainParamCount": len(main_params), "mainValidCount": len(valid_main), "filterResultCount": len(filter_results), "audit": audit, "baselineETH061": BASELINE_ETH061, "topCandidates": all_candidates[:100], "selected": selected, "archiveRows": archive_rows, "verdict": verdict, "liveConfigMutated": False}
    render_outputs(payload)
    print(OUT_COMPARE_MD)
    print(OUT_VERDICT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
