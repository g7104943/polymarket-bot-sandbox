#!/usr/bin/env python3
from __future__ import annotations

"""Raw-kline compound-growth main-model search for top159 successor.

Research only. This script does not mutate live configs, caches, ledgers,
process state, order submission, or monitor settings.

Purpose:
  - Stop adding small external filters to 061.
  - Train a new ETH 15m main model over all raw 15m markets.
  - Let the model choose UP/DOWN and whether to trade.
  - Evaluate only the user's requested fair raw-kline proxy:
      start 850U, stake 1% current bankroll, buy price 0.50, full fill.
"""

import concurrent.futures as cf
import hashlib
import importlib.util
import itertools
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

THREADS = os.environ.get("TOP159_RAW_GROWTH_THREADS", "1")
for key in ["OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(key, THREADS)

import numpy as np
import pandas as pd

ROOT = Path("/Users/mac/polyfun")
NEXT = ROOT / "polyfun-next"
SCRIPTS = NEXT / "scripts"
REPORTS = ROOT / "reports"
EXTREME_SCRIPT = SCRIPTS / "run_top159_integrated_main_extreme_search.py"
ARCHIVE_SCRIPT = ROOT / "scripts" / "ops" / "run_top159_all_archived_real_eth_compare.py"
CLUSTER_COMPARE_JSON = REPORTS / "top159_061_deep_v2_cluster_180_365_archive_compare_latest.json"

WORKERS = int(os.environ.get("TOP159_RAW_GROWTH_WORKERS", "4"))
MAX_SECONDS = int(os.environ.get("TOP159_RAW_GROWTH_MAX_SECONDS", str(6 * 3600)))
CHECKPOINT_SECONDS = int(os.environ.get("TOP159_RAW_GROWTH_CHECKPOINT_SECONDS", "900"))
PARAM_LIMIT = int(os.environ.get("TOP159_RAW_GROWTH_PARAM_LIMIT", "0"))
MAX_TRAIN_ROWS = int(os.environ.get("TOP159_RAW_GROWTH_MAX_TRAIN_ROWS", "180000"))
RNG_SEED = 20260505

START_BANKROLL = 850.0
STAKE_PCT = 0.01
BUY_PRICE = 0.50

OUT_AUDIT_MD = REPORTS / "top159_raw_growth_mainmodel_v1_bug_audit_latest.md"
OUT_AUDIT_JSON = REPORTS / "top159_raw_growth_mainmodel_v1_bug_audit_latest.json"
OUT_RESULTS = REPORTS / "top159_raw_growth_mainmodel_v1_results_latest.jsonl"
OUT_CHECKPOINT = REPORTS / "top159_raw_growth_mainmodel_v1_checkpoint_latest.json"
OUT_LEADERBOARD = REPORTS / "top159_raw_growth_mainmodel_v1_leaderboard_latest.json"
OUT_COMPARE_MD = REPORTS / "top159_raw_growth_mainmodel_v1_180_365_archive_compare_latest.md"
OUT_COMPARE_JSON = REPORTS / "top159_raw_growth_mainmodel_v1_180_365_archive_compare_latest.json"
OUT_VERDICT_MD = REPORTS / "top159_raw_growth_mainmodel_v1_unique_verdict_latest.md"
OUT_VERDICT_JSON = REPORTS / "top159_raw_growth_mainmodel_v1_unique_verdict_latest.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M = load_module("raw_growth_integrated_extreme", EXTREME_SCRIPT)
archive = load_module("raw_growth_archive_compare", ARCHIVE_SCRIPT)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def bj_now() -> str:
    return pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S CST")


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def max_drawdown(equity: np.ndarray) -> tuple[float, float, int, int]:
    if equity.size == 0:
        return 0.0, 0.0, -1, -1
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    trough = int(np.argmax(dd))
    max_dd = float(dd[trough])
    peak_i = int(np.argmax(equity[: trough + 1])) if trough >= 0 else -1
    denom = float(peak[trough]) if trough >= 0 else 0.0
    return max_dd, (max_dd / denom if denom > 1e-12 else 0.0), peak_i, trough


def monthly_positive_ratio(equity: np.ndarray, dt: pd.Series) -> float:
    if equity.size == 0:
        return 0.0
    frame = pd.DataFrame({"dt": pd.to_datetime(dt, utc=True, errors="coerce"), "equity": equity})
    frame = frame.dropna().sort_values("dt")
    if frame.empty:
        return 0.0
    frame["month"] = frame["dt"].dt.to_period("M").astype(str)
    prev = START_BANKROLL
    vals: list[float] = []
    for _, g in frame.groupby("month", sort=True):
        end = float(g["equity"].iloc[-1])
        vals.append(end - prev)
        prev = end
    return round(sum(x > 0 for x in vals) / len(vals), 6) if vals else 0.0


def compound_metrics(selected: pd.DataFrame, name: str, window: str, method: str = "raw_full_fill_buy_0.50") -> dict[str, Any]:
    sel = selected.sort_values("dt").reset_index(drop=True).copy()
    won = sel["won"].astype(bool).to_numpy()
    eq = START_BANKROLL
    curve = np.empty(len(sel), dtype=float)
    for i, ok in enumerate(won):
        stake = eq * STAKE_PCT
        ret = (1.0 / BUY_PRICE) - 1.0 if ok else -1.0
        eq += stake * ret
        if eq < 0:
            eq = 0.0
        curve[i] = eq
    max_dd, max_dd_pct, peak_i, trough_i = max_drawdown(curve)
    wins = int(won.sum())
    losses = int(len(won) - wins)
    return {
        "name": name,
        "window": window,
        "method": method,
        "trades": int(len(sel)),
        "wins": wins,
        "losses": losses,
        "winRatePct": round(100.0 * wins / len(sel), 6) if len(sel) else 0.0,
        "compoundPnl": round(float(eq - START_BANKROLL), 6),
        "endingBankroll": round(float(eq), 6),
        "maxDrawdownUsd": round(max_dd, 6),
        "maxDrawdownPct": round(max_dd_pct * 100.0, 6),
        "returnDrawdownRatio": round(float(eq - START_BANKROLL) / max_dd, 6) if max_dd > 1e-12 else (999.0 if eq > START_BANKROLL else 0.0),
        "monthlyPositiveRatio": monthly_positive_ratio(curve, sel["dt"]) if len(sel) else 0.0,
        "drawdownPeakTime": str(pd.to_datetime(sel["dt"], utc=True).iloc[peak_i]) if peak_i >= 0 and len(sel) else None,
        "drawdownTroughTime": str(pd.to_datetime(sel["dt"], utc=True).iloc[trough_i]) if trough_i >= 0 and len(sel) else None,
        "setHash": stable_hash(sel[["dt", "pred_up15", "label_up", "won"]].to_dict("records")),
    }


def split_train_val(df: pd.DataFrame, window: str, train_window: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    end = df["dt"].max()
    days = 180 if window == "180d" else 365
    start = end - pd.Timedelta(days=days)
    val = df[df["dt"] >= start].copy()
    if train_window == "full":
        train = df[df["dt"] < start].copy()
    else:
        train_days = {"1y": 365, "3y": 1095, "5y": 1825, "full": None}[train_window]
        train = df[(df["dt"] < start) & (df["dt"] >= start - pd.Timedelta(days=int(train_days)))].copy()
    # Four-hour embargo. It is conservative and avoids adjacent-window leakage.
    train = train[train["dt"] < start - pd.Timedelta(hours=4)].copy()
    if MAX_TRAIN_ROWS > 0 and len(train) > MAX_TRAIN_ROWS:
        train = train.sort_values("dt").iloc[-MAX_TRAIN_ROWS:].copy()
    return train, val


def feature_subset(features: list[str], mode: str) -> list[str]:
    if mode == "base15":
        out = [c for c in features if not c.startswith(("1h_", "4h_", "daily_"))]
    elif mode == "base_plus_1h4h":
        out = [c for c in features if not c.startswith("daily_")]
    elif mode == "base_plus_daily":
        out = [c for c in features if not c.startswith(("1h_", "4h_"))]
    elif mode == "base_plus_4h_daily":
        out = [c for c in features if not c.startswith("1h_")]
    elif mode == "trend_multi":
        keys = ("ret_", "ema_", "ema_dist_", "rsi", "bb_pos", "range_", "vol_")
        out = [c for c in features if any(k in c for k in keys)]
    elif mode == "wide":
        out = list(features)
    else:
        out = list(features)
    return out


def fit_model(engine: str, train: pd.DataFrame, feats: list[str], params: dict[str, Any], random_labels: bool = False):
    if len(train) < 1000 or not feats:
        return None
    x = train[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = train["label_up"].astype(int).to_numpy()
    if random_labels:
        y = np.random.default_rng(RNG_SEED + len(train) + len(feats)).permutation(y)
    if len(np.unique(y)) < 2:
        return None
    if engine == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=600, C=float(params.get("C", 1.0)), random_state=RNG_SEED))
    elif engine == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=int(params["n_estimators"]),
            learning_rate=float(params["learning_rate"]),
            num_leaves=int(params["num_leaves"]),
            min_child_samples=int(params["min_child_samples"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_lambda=float(params["reg_lambda"]),
            random_state=RNG_SEED,
            n_jobs=int(THREADS),
            verbose=-1,
        )
    elif engine == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            reg_lambda=float(params["reg_lambda"]),
            random_state=RNG_SEED,
            n_jobs=int(THREADS),
            eval_metric="logloss",
            verbosity=0,
        )
    elif engine == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            iterations=int(params["n_estimators"]),
            depth=int(params["depth"]),
            learning_rate=float(params["learning_rate"]),
            l2_leaf_reg=float(params["reg_lambda"]),
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=RNG_SEED,
            thread_count=int(THREADS),
            verbose=False,
        )
    else:
        raise ValueError(engine)
    model.fit(x, y)
    return model


def predict_prob(model: Any, val: pd.DataFrame, feats: list[str]) -> np.ndarray:
    x = val[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def select_rows(val: pd.DataFrame, prob: np.ndarray, params: dict[str, Any]) -> pd.DataFrame:
    pred_up = prob >= 0.5
    score = np.maximum(prob, 1.0 - prob)
    mask = score >= 0.5 + float(params["edge"])
    if params.get("score_band") == "mid_only":
        mask &= score <= float(params.get("max_score", 0.76))
    if params.get("vol_q", 0.999) < 0.999 and "vol_16" in val.columns:
        v = pd.to_numeric(val["vol_16"], errors="coerce")
        mask &= v <= float(v.quantile(float(params["vol_q"])))
    selected = val.loc[mask, ["dt", "label_up"]].copy()
    selected["pred_up15"] = pred_up[mask].astype(bool)
    selected["score15"] = score[mask]
    # Optional daily density cap: keep strongest signals per UTC day.
    cap = int(params.get("daily_cap", 0) or 0)
    if cap > 0 and not selected.empty:
        selected["day"] = pd.to_datetime(selected["dt"], utc=True).dt.floor("D")
        selected = (
            selected.sort_values(["day", "score15"], ascending=[True, False])
            .groupby("day", sort=False)
            .head(cap)
            .sort_values("dt")
            .drop(columns=["day"])
        )
    selected = selected.reset_index(drop=True)
    selected["won"] = selected["pred_up15"].to_numpy() == selected["label_up"].astype(bool).to_numpy()
    return selected


def evaluate_params(df: pd.DataFrame, features: list[str], params: dict[str, Any], name: str) -> dict[str, Any] | None:
    feats = feature_subset(features, params["feature_mode"])
    max_features = int(params.get("max_features", 0) or 0)
    if max_features > 0:
        feats = feats[:max_features]
    if len(feats) < 8:
        return None
    rows = []
    for window in ["180d", "365d"]:
        train, val = split_train_val(df, window, params["train_window"])
        model = fit_model(params["engine"], train, feats, params)
        if model is None:
            return None
        prob = predict_prob(model, val, feats)
        selected = select_rows(val, prob, params)
        if selected.empty:
            return None
        row = compound_metrics(selected, name, window)
        row.update({
            "engine": params["engine"],
            "trainWindow": params["train_window"],
            "featureMode": params["feature_mode"],
            "edge": params["edge"],
            "dailyCap": int(params.get("daily_cap", 0) or 0),
            "featureCount": int(len(feats)),
        })
        rows.append(row)
    return {"name": name, "params": params, "rows": rows, "featureCount": int(len(feats))}


def current_061_rows() -> dict[str, dict[str, Any]]:
    if not CLUSTER_COMPARE_JSON.exists():
        raise FileNotFoundError(CLUSTER_COMPARE_JSON)
    payload = json.loads(CLUSTER_COMPARE_JSON.read_text())
    rows: dict[str, dict[str, Any]] = {}
    for r in payload["validationRows"]:
        if r.get("name") == "current_live_06173_cluster_gate" and r.get("method") == "full_fill_buy_0.50":
            rows[r["window"]] = dict(r)
    if set(rows) != {"180d", "365d"}:
        raise RuntimeError("current061 baseline rows missing")
    return rows


def pass_gate(candidate: dict[str, Any], baseline: dict[str, dict[str, Any]]) -> tuple[bool, list[str]]:
    by = {r["window"]: r for r in candidate["rows"]}
    reasons = []
    # Strict user's current raw fair rule.
    for window in ["180d", "365d"]:
        r = by[window]
        b = baseline[window]
        if r["compoundPnl"] <= b["compoundPnl"]:
            reasons.append(f"{window}_pnl_not_above_061")
        if window == "365d" and r["winRatePct"] < b["winRatePct"]:
            reasons.append("365d_winrate_below_061")
        if window == "180d" and r["winRatePct"] < b["winRatePct"] - 0.15:
            reasons.append("180d_winrate_too_low")
        if r["maxDrawdownUsd"] > b["maxDrawdownUsd"] * 1.05:
            reasons.append(f"{window}_drawdown_too_high")
        if r["trades"] < max(100 if window == "180d" else 200, int(b["trades"] * 0.45)):
            reasons.append(f"{window}_too_few_trades")
    return not reasons, reasons


def score_candidate(candidate: dict[str, Any], baseline: dict[str, dict[str, Any]]) -> float:
    by = {r["window"]: r for r in candidate["rows"]}
    pnl_gain = (by["180d"]["compoundPnl"] - baseline["180d"]["compoundPnl"]) / max(1.0, baseline["180d"]["compoundPnl"])
    pnl_gain += (by["365d"]["compoundPnl"] - baseline["365d"]["compoundPnl"]) / max(1.0, baseline["365d"]["compoundPnl"])
    wr_gain = (by["180d"]["winRatePct"] - baseline["180d"]["winRatePct"]) + (by["365d"]["winRatePct"] - baseline["365d"]["winRatePct"])
    dd_penalty = by["365d"]["maxDrawdownUsd"] / max(1.0, baseline["365d"]["maxDrawdownUsd"])
    rd = by["180d"]["returnDrawdownRatio"] + by["365d"]["returnDrawdownRatio"]
    return pnl_gain * 10000.0 + wr_gain * 20.0 + rd * 5.0 - dd_penalty * 10.0


def param_grid() -> list[dict[str, Any]]:
    engines = [x for x in os.environ.get("TOP159_RAW_GROWTH_ENGINES", "logistic,lightgbm,xgboost").split(",") if x]
    rows: list[dict[str, Any]] = []
    train_windows = ["1y", "3y", "5y", "full"]
    feature_modes = ["base15", "base_plus_1h4h", "base_plus_daily", "base_plus_4h_daily", "trend_multi", "wide"]
    edges = [0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.08, 0.09]
    score_bands = [("all", 1.0), ("mid_only", 0.72), ("mid_only", 0.76), ("mid_only", 0.80)]
    vol_qs = [0.80, 0.90, 0.95, 0.999]
    max_features_grid = [60, 120, 180]
    daily_caps = [0, 32, 24, 18, 12]
    for engine, tw, fm, edge, band, vol_q, mf, cap in itertools.product(
        engines, train_windows, feature_modes, edges, score_bands, vol_qs, max_features_grid, daily_caps
    ):
        score_band, max_score = band
        if engine == "logistic":
            for C in [0.25, 0.5, 0.75, 1.0, 2.0]:
                rows.append({
                    "engine": engine, "train_window": tw, "feature_mode": fm, "edge": edge,
                    "score_band": score_band, "max_score": max_score, "vol_q": vol_q,
                    "max_features": mf, "daily_cap": cap, "C": C,
                    "n_estimators": 1, "learning_rate": 0.04, "num_leaves": 16,
                    "min_child_samples": 80, "subsample": 0.9, "colsample_bytree": 0.9,
                    "reg_lambda": 1.0, "depth": 3,
                })
            continue
        model_sets = [
            (80, 0.025, 16, 80, 0.5, 3),
            (120, 0.035, 24, 80, 0.8, 4),
            (180, 0.025, 32, 100, 1.2, 4),
            (240, 0.018, 40, 140, 2.0, 5),
        ]
        for ne, lr, leaves, mcs, reg, depth in model_sets:
            rows.append({
                "engine": engine, "train_window": tw, "feature_mode": fm, "edge": edge,
                "score_band": score_band, "max_score": max_score, "vol_q": vol_q,
                "max_features": mf, "daily_cap": cap, "C": 1.0,
                "n_estimators": ne, "learning_rate": lr, "num_leaves": leaves,
                "min_child_samples": mcs, "subsample": 0.88, "colsample_bytree": 0.88,
                "reg_lambda": reg, "depth": depth,
            })
    if PARAM_LIMIT > 0 and len(rows) > PARAM_LIMIT:
        rng = np.random.default_rng(RNG_SEED)
        order = rng.permutation(len(rows))
        rows = [rows[int(i)] for i in order[:PARAM_LIMIT]]
    return rows


_DF: pd.DataFrame | None = None
_FEATURES: list[str] | None = None


def init_worker(df: pd.DataFrame, features: list[str]) -> None:
    global _DF, _FEATURES
    _DF = df
    _FEATURES = features


def eval_worker(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    if _DF is None or _FEATURES is None:
        raise RuntimeError("worker not initialized")
    idx, params = item
    name = f"rawgrowth_{params['engine']}_{params['train_window']}_{params['feature_mode']}_edge{params['edge']}_cap{params.get('daily_cap', 0)}_{stable_hash(params)}"
    try:
        out = evaluate_params(_DF, _FEATURES, params, name)
        return out if out is not None else {"name": name, "params": params, "error": "empty_or_fit_failed"}
    except Exception as exc:
        return {"name": name, "params": params, "error": repr(exc)[:800]}


def archive_rows_for_candidate(df: pd.DataFrame, features: list[str], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        fill_rows, _scan_audit = archive.load_live_fill_rows()
        all_eth = archive.aggregate_logical(fill_rows, "全部归档ETH15m live去重")
        strict_mask = fill_rows["sourcePath"].str.contains("slot01|forcedtrade|base_forcedtrade|__live_eth|logs_v5_exp10_bp0480__live_eth", case=False, regex=True)
        strict = archive.aggregate_logical(fill_rows[strict_mask].copy(), "严格旧slot1/ETH相关live去重") if strict_mask.any() else all_eth.iloc[0:0]
        params = candidate["params"]
        feats = feature_subset(features, params["feature_mode"])[: int(params.get("max_features", 0) or 999999)]
        for old in [strict, all_eth]:
            if old.empty:
                continue
            start = old["marketStart"].min()
            end = old["marketStart"].max()
            train = df[df["dt"] < start - pd.Timedelta(hours=4)].copy()
            if params["train_window"] != "full":
                days = {"1y": 365, "3y": 1095, "5y": 1825}[params["train_window"]]
                train = train[train["dt"] >= start - pd.Timedelta(days=days)].copy()
            val = df[(df["dt"] >= start) & (df["dt"] <= end)].copy()
            model = fit_model(params["engine"], train, feats, params)
            if model is None:
                continue
            prob = predict_prob(model, val, feats)
            sel = select_rows(val, prob, params)
            pred = sel[["dt", "pred_up15", "score15"]].copy()
            merged = old.merge(pred, left_on="marketStart", right_on="dt", how="left")
            chosen = merged[merged["pred_up15"].notna()].copy().sort_values("marketStart")
            if chosen.empty:
                rows.append({"scope": old["scopeName"].iloc[0], "name": candidate["name"], "oldRealMarkets": int(len(old)), "selectedTrades": 0})
                continue
            model_won = chosen["pred_up15"].astype(bool).to_numpy() == chosen["actualUp"].astype(bool).to_numpy()
            # Compound 850/1%/buy0.50 over archived selected windows.
            arch_sel = pd.DataFrame({
                "dt": chosen["marketStart"].to_numpy(),
                "label_up": chosen["actualUp"].astype(bool).to_numpy(),
                "pred_up15": chosen["pred_up15"].astype(bool).to_numpy(),
                "won": model_won,
            })
            metrics = compound_metrics(arch_sel, candidate["name"], str(old["scopeName"].iloc[0]), "archive_raw_buy_0.50")
            rows.append({
                "scope": old["scopeName"].iloc[0],
                "name": candidate["name"],
                "oldRealMarkets": int(len(old)),
                "selectedTrades": int(len(chosen)),
                "skippedTrades": int(len(old) - len(chosen)),
                "wins": int(model_won.sum()),
                "losses": int(len(model_won) - int(model_won.sum())),
                "winRatePct": round(100.0 * float(model_won.mean()), 6),
                "compoundPnl": metrics["compoundPnl"],
                "endingBankroll": metrics["endingBankroll"],
                "maxDrawdownUsd": metrics["maxDrawdownUsd"],
                "avgScore": round(float(chosen["score15"].mean()), 6),
                "setHash": stable_hash(chosen[["marketSlug", "pred_up15", "actualUp"]].to_dict("records")),
            })
    except Exception as exc:
        rows.append({"scope": "archive_error", "name": candidate.get("name"), "error": repr(exc)[:800]})
    return rows


def bug_audit(df: pd.DataFrame, features: list[str], baseline: dict[str, dict[str, Any]]) -> dict[str, Any]:
    params = {
        "engine": "logistic", "train_window": "3y", "feature_mode": "base_plus_1h4h", "edge": 0.05,
        "score_band": "all", "max_score": 1.0, "vol_q": 0.999, "max_features": 120, "daily_cap": 0,
        "C": 1.0, "n_estimators": 1, "learning_rate": 0.04, "num_leaves": 16, "min_child_samples": 80,
        "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0, "depth": 3,
    }
    c1 = evaluate_params(df, features, params, "audit_repeat")
    c2 = evaluate_params(df, features, params, "audit_repeat")
    repeat = []
    if c1 and c2:
        for a, b in zip(c1["rows"], c2["rows"]):
            repeat.append({
                "window": a["window"],
                "hash1": a["setHash"],
                "hash2": b["setHash"],
                "pnl1": a["compoundPnl"],
                "pnl2": b["compoundPnl"],
                "passed": a["setHash"] == b["setHash"] and a["compoundPnl"] == b["compoundPnl"],
            })
    random_label = {"status": "not_run", "passed": False}
    try:
        feats = feature_subset(features, params["feature_mode"])[: int(params["max_features"])]
        train, val = split_train_val(df, "365d", params["train_window"])
        model = fit_model(params["engine"], train, feats, params, random_labels=True)
        if model is not None:
            prob = predict_prob(model, val, feats)
            sel = select_rows(val, prob, params)
            wr = float(sel["won"].mean() * 100.0) if len(sel) else 0.0
            random_label = {"status": "ok", "selectedTrades": int(len(sel)), "winRatePct": round(wr, 6), "passed": len(sel) < 50 or 43.0 <= wr <= 57.0}
    except Exception as exc:
        random_label = {"status": "error", "error": repr(exc)[:300], "passed": False}
    forbidden = [c for c in features if c in M.base.FORBIDDEN_FEATURES or c.startswith(tuple(M.base.FORBIDDEN_PREFIXES))]
    audit = {
        "generatedAt": now_iso(),
        "beijingTime": bj_now(),
        "researchOnlyNoLiveChange": True,
        "fairMetric": "850U initial, stake 1% current bankroll, buy price 0.50, full fill",
        "timePolicy": "15m/1h/4h features are shifted to already closed candles; 4h embargo before validation",
        "forbiddenFeatureHits": forbidden,
        "repeatability": repeat,
        "randomLabelAudit": random_label,
        "baseline061": baseline,
    }
    audit["passed"] = not forbidden and bool(repeat) and all(r["passed"] for r in repeat) and bool(random_label.get("passed"))
    write_json(OUT_AUDIT_JSON, audit)
    write_text(OUT_AUDIT_MD, "\n".join([
        "# top159 原始复利主模型：回测器审计",
        "",
        f"- 北京时间：`{audit['beijingTime']}`",
        f"- 研究动作：`research_only_no_live_change`",
        f"- 公平口径：`{audit['fairMetric']}`",
        f"- 时间口径：`{audit['timePolicy']}`",
        f"- 禁用字段命中：`{forbidden}`",
        f"- 重复运行：`{repeat}`",
        f"- 随机标签测试：`{random_label}`",
        f"- 审计通过：`{audit['passed']}`",
    ]) + "\n")
    return audit


def write_progress(results: list[dict[str, Any]], baseline: dict[str, dict[str, Any]], total: int, started: float, data_truth: dict[str, Any], audit: dict[str, Any], finished: bool = False) -> None:
    valid = [r for r in results if r.get("rows")]
    rows = []
    for c in valid:
        ok, reasons = pass_gate(c, baseline)
        rows.append({**c, "passed": ok, "reasons": reasons, "score": score_candidate(c, baseline)})
    rows.sort(key=lambda r: r["score"], reverse=True)
    strict = [r for r in rows if r["passed"]]
    selected = strict[0] if strict else (rows[0] if rows else None)
    archive_rows = archive_rows_for_candidate(_DF_MAIN, _FEATURES_MAIN, selected) if selected and _DF_MAIN is not None and _FEATURES_MAIN is not None else []
    payload = {
        "generatedAt": now_iso(),
        "beijingTime": bj_now(),
        "researchOnlyNoLiveChange": True,
        "finished": finished,
        "elapsedSeconds": round(time.time() - started, 3),
        "workers": WORKERS,
        "totalCandidates": total,
        "doneCount": len(results),
        "validCount": len(valid),
        "strictPassCount": len(strict),
        "dataTruth": data_truth,
        "audit": audit,
        "baseline061": baseline,
        "topCandidates": rows[:300],
        "strictPass": strict[:100],
        "selectedForArchiveCompare": selected,
        "archivedRealRowsForSelected": archive_rows,
        "liveConfigMutated": False,
    }
    write_json(OUT_CHECKPOINT, payload)
    write_json(OUT_LEADERBOARD, payload)
    write_json(OUT_COMPARE_JSON, payload)
    status = "candidate_beats_061" if selected and selected.get("passed") else "no_candidate_beats_061"
    write_json(OUT_VERDICT_JSON, {"generatedAt": payload["generatedAt"], "status": status, "selected": selected, "archivedRealRowsForSelected": archive_rows, "liveAction": "research_only_no_live_change"})
    render_markdown(payload)


def render_markdown(payload: dict[str, Any]) -> None:
    baseline = payload["baseline061"]
    selected = payload.get("selectedForArchiveCompare")
    lines = [
        "# top159 原始K线复利目标主模型：180/365/历史归档对比",
        "",
        f"- 北京时间：`{payload['beijingTime']}`",
        "- 口径：`850U初始 / 每笔当前资金1% / 买价0.50 / 满成交 / 原始K线方向`",
        "- live动作：`research_only_no_live_change`",
        "",
        "## 180天 / 365天",
        "",
        "|配置|窗口|训练窗|模型|特征组|交易数|胜/负|胜率|盈亏|期末资金|最大回撤|收益回撤比|月正收益|哈希|",
        "|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for w in ["180d", "365d"]:
        b = baseline[w]
        lines.append(
            f"|当前061|{w}|-|061_cluster_gate|-|{b['trades']}|{b['wins']}/{b['losses']}|{b['winRatePct']:.2f}%|"
            f"{b['compoundPnl']:,.2f}|{b['endingBankroll']:,.2f}|{b['maxDrawdownUsd']:,.2f}|{b['returnDrawdownRatio']:.2f}|{b.get('monthlyPositiveRatio', 0):.2%}|`{b['setHash']}`|"
        )
    if selected and selected.get("rows"):
        for r in selected["rows"]:
            lines.append(
                f"|{selected['name']}|{r['window']}|{r.get('trainWindow','-')}|{r.get('engine','-')}|{r.get('featureMode','-')}|"
                f"{r['trades']}|{r['wins']}/{r['losses']}|{r['winRatePct']:.2f}%|{r['compoundPnl']:,.2f}|"
                f"{r['endingBankroll']:,.2f}|{r['maxDrawdownUsd']:,.2f}|{r['returnDrawdownRatio']:.2f}|{r['monthlyPositiveRatio']:.2%}|`{r['setHash']}`|"
            )
    lines += ["", "## 历史归档纯预测复核", "", "|范围|配置|交易数|胜/负|胜率|盈亏|期末资金|最大回撤|", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in payload.get("archivedRealRowsForSelected", []):
        if "error" in r:
            lines.append(f"|{r.get('scope')}|{r.get('name')}|错误|-|-|-|-|-|")
            continue
        lines.append(
            f"|{r['scope']}|{r['name']}|{r['selectedTrades']}|{r.get('wins',0)}/{r.get('losses',0)}|"
            f"{r.get('winRatePct',0):.2f}%|{r.get('compoundPnl',0):,.2f}|{r.get('endingBankroll',0):,.2f}|{r.get('maxDrawdownUsd',0):,.2f}|"
        )
    status = "通过" if selected and selected.get("passed") else "未通过"
    lines += [
        "",
        "## 结论",
        "",
        f"- 严格过门状态：`{status}`。",
        "- 若未通过：不改真钱061；这说明当前 ETH15m 原始K线方向模型仍没有可靠超过061。",
    ]
    write_text(OUT_COMPARE_MD, "\n".join(lines) + "\n")
    write_text(OUT_VERDICT_MD, "\n".join(lines) + "\n")


_DF: pd.DataFrame | None = None
_FEATURES: list[str] | None = None
_DF_MAIN: pd.DataFrame | None = None
_FEATURES_MAIN: list[str] | None = None


def init_worker(df: pd.DataFrame, features: list[str]) -> None:
    global _DF, _FEATURES
    _DF = df
    _FEATURES = features


def eval_worker(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    if _DF is None or _FEATURES is None:
        raise RuntimeError("worker not initialized")
    idx, params = item
    name = f"rawgrowth_{params['engine']}_{params['train_window']}_{params['feature_mode']}_edge{params['edge']}_cap{params.get('daily_cap',0)}_{stable_hash(params)}"
    try:
        out = evaluate_params(_DF, _FEATURES, params, name)
        return out if out is not None else {"name": name, "params": params, "error": "empty_or_fit_failed"}
    except Exception as exc:
        return {"name": name, "params": params, "error": repr(exc)[:800]}


def run() -> int:
    global _DF_MAIN, _FEATURES_MAIN
    started = time.time()
    print(f"[raw-growth] start {bj_now()} workers={WORKERS} maxSeconds={MAX_SECONDS}", flush=True)
    df, features, truth = M.build_integrated_frame()
    _DF_MAIN, _FEATURES_MAIN = df, features
    baseline = current_061_rows()
    audit = bug_audit(df, features, baseline)
    params = param_grid()
    total = len(params)
    results: list[dict[str, Any]] = []
    done_names: set[str] = set()
    if OUT_RESULTS.exists():
        for line in OUT_RESULTS.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                results.append(row)
                done_names.add(row.get("name"))
            except Exception:
                pass
    pending = []
    for i, p in enumerate(params):
        name = f"rawgrowth_{p['engine']}_{p['train_window']}_{p['feature_mode']}_edge{p['edge']}_cap{p.get('daily_cap',0)}_{stable_hash(p)}"
        if name not in done_names:
            pending.append((i, p))
    print(f"[raw-growth] rows={len(df)} features={len(features)} total={total} done={len(results)} pending={len(pending)} audit={audit['passed']}", flush=True)
    write_progress(results, baseline, total, started, truth, audit, finished=False)
    last = time.time()
    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_RESULTS.open("a", encoding="utf-8") as fh:
        with cf.ProcessPoolExecutor(max_workers=WORKERS, initializer=init_worker, initargs=(df, features)) as ex:
            futures: dict[Any, tuple[int, dict[str, Any]]] = {}
            it = iter(pending)
            for _ in range(max(4, WORKERS * 2)):
                try:
                    item = next(it)
                except StopIteration:
                    break
                futures[ex.submit(eval_worker, item)] = item
            while futures:
                if time.time() - started >= MAX_SECONDS:
                    break
                done, _ = cf.wait(futures, timeout=5, return_when=cf.FIRST_COMPLETED)
                for fut in done:
                    item = futures.pop(fut)
                    try:
                        row = fut.result()
                    except Exception as exc:
                        row = {"name": f"candidate_{item[0]}", "params": item[1], "error": repr(exc)[:800]}
                    fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")
                    fh.flush()
                    results.append(row)
                    try:
                        nxt = next(it)
                        futures[ex.submit(eval_worker, nxt)] = nxt
                    except StopIteration:
                        pass
                if time.time() - last >= CHECKPOINT_SECONDS:
                    write_progress(results, baseline, total, started, truth, audit, finished=False)
                    print(f"[raw-growth] checkpoint {len(results)}/{total}", flush=True)
                    last = time.time()
            for fut in futures:
                fut.cancel()
    finished = len(results) >= total
    write_progress(results, baseline, total, started, truth, audit, finished=finished)
    print(json.dumps({
        "status": "finished" if finished else "checkpointed",
        "done": len(results),
        "total": total,
        "compare": str(OUT_COMPARE_MD),
        "verdict": str(OUT_VERDICT_MD),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
