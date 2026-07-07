#!/usr/bin/env python3
from __future__ import annotations

"""Shock-aware integrated top159 main model research.

Research-only:
  - does not mutate live configs,
  - does not restart trading,
  - does not submit orders.

This script trains a new main model that directly consumes 15m/base features,
closed-candle 1h/4h shock features, and fold-safe wrong-cluster encodings.
It is intentionally separate from the live shock gate.
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

THREADS = os.environ.get("TOP159_SHOCK_AWARE_THREADS", "1")
for key in ["OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(key, THREADS)

import numpy as np
import pandas as pd

ROOT = Path("/Users/mac/polyfun")
REPORTS = ROOT / "reports"
SCRIPT_DIR = ROOT / "polyfun-next" / "scripts"
EXTREME_SCRIPT = SCRIPT_DIR / "run_top159_integrated_main_extreme_search.py"
SHOCK_SCRIPT = SCRIPT_DIR / "run_top159_shock_candle_filter_research.py"
ARCHIVE_SCRIPT = ROOT / "scripts" / "ops" / "run_top159_all_archived_real_eth_compare.py"

WORKERS = int(os.environ.get("TOP159_SHOCK_AWARE_WORKERS", "4"))
MAX_SECONDS = int(os.environ.get("TOP159_SHOCK_AWARE_MAX_SECONDS", str(6 * 3600)))
CHECKPOINT_SECONDS = int(os.environ.get("TOP159_SHOCK_AWARE_CHECKPOINT_SECONDS", "900"))
PARAM_LIMIT = int(os.environ.get("TOP159_SHOCK_AWARE_PARAM_LIMIT", "0"))
MC_TRIALS = int(os.environ.get("TOP159_SHOCK_AWARE_MC_TRIALS", "600"))
MAX_TRAIN_ROWS = int(os.environ.get("TOP159_SHOCK_AWARE_MAX_TRAIN_ROWS", "180000"))
RNG_SEED = 20260504

OUT_AUDIT_MD = REPORTS / "top159_shock_aware_integrated_main_bug_audit_latest.md"
OUT_AUDIT_JSON = REPORTS / "top159_shock_aware_integrated_main_bug_audit_latest.json"
OUT_RESULTS = REPORTS / "top159_shock_aware_integrated_main_results_latest.jsonl"
OUT_CHECKPOINT = REPORTS / "top159_shock_aware_integrated_main_checkpoint_latest.json"
OUT_LEADER_MD = REPORTS / "top159_shock_aware_integrated_main_leaderboard_latest.md"
OUT_LEADER_JSON = REPORTS / "top159_shock_aware_integrated_main_leaderboard_latest.json"
OUT_COMPARE_MD = REPORTS / "top159_shock_aware_integrated_main_180_365_archived_compare_latest.md"
OUT_COMPARE_JSON = REPORTS / "top159_shock_aware_integrated_main_180_365_archived_compare_latest.json"
OUT_VERDICT_MD = REPORTS / "top159_shock_aware_integrated_main_unique_verdict_latest.md"
OUT_VERDICT_JSON = REPORTS / "top159_shock_aware_integrated_main_unique_verdict_latest.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M = load_module("shock_aware_integrated_base", EXTREME_SCRIPT)
shock = load_module("shock_aware_candle_features", SHOCK_SCRIPT)
archive = load_module("shock_aware_archive_compare", ARCHIVE_SCRIPT)
M.aux.MC_TRIALS = MC_TRIALS
M.base.TRIALS = MC_TRIALS


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


def attach_closed_candle_shock_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Attach 15m/1h/4h shock features by closed-candle available_at time."""
    out = df.sort_values("dt").copy()
    out["ts_ns"] = pd.to_datetime(out["dt"], utc=True).map(lambda x: pd.Timestamp(x).value).astype("int64")
    left = out[["ts_ns"]].sort_values("ts_ns")
    added: list[str] = []
    truth: dict[str, Any] = {
        "shockFeatureAlignmentVersion": getattr(shock, "SHOCK_FEATURE_ALIGNMENT_VERSION", "unknown"),
        "shockFeatureAlignment": "candle_available_at=open_time+interval; merge_asof backward; only closed candles visible",
    }
    for tf in ["15m", "1h", "4h"]:
        raw = M.load_raw("ETH", tf)
        feats = shock.candle_features(raw, tf).sort_values("ts_ns").drop(columns=["dt"])
        merged = pd.merge_asof(left, feats, on="ts_ns", direction="backward", allow_exact_matches=True).sort_index()
        cols_numeric = [
            f"{tf}_body_ratio", f"{tf}_range_pct", f"{tf}_range_q", f"{tf}_volume_mult",
            f"{tf}_close_pos", f"{tf}_upper_wick_ratio", f"{tf}_lower_wick_ratio", f"{tf}_pos20",
        ]
        for col in cols_numeric:
            new = f"shock_{col}"
            out[new] = pd.to_numeric(merged[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            added.append(new)
        out[f"shock_{tf}_big_body_default"] = merged[f"{tf}_is_big_body_default"].fillna(False).astype(int).to_numpy()
        added.append(f"shock_{tf}_big_body_default")
        for state in ["up", "down", "mixed"]:
            new = f"shock_{tf}_trend_{state}"
            out[new] = (merged[f"{tf}_trend_state"].astype(str) == state).astype(int).to_numpy()
            added.append(new)
        for cdir in ["up", "down"]:
            new = f"shock_{tf}_candle_{cdir}"
            out[new] = (merged[f"{tf}_candle_dir"].astype(str) == cdir).astype(int).to_numpy()
            added.append(new)
        truth[f"{tf}Rows"] = int(len(feats))
    out = out.drop(columns=["ts_ns"])
    return out, added, truth


def add_environment_bins(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    out["bj_hour_bucket"] = pd.to_datetime(out["dt"], utc=True).dt.tz_convert("Asia/Shanghai").dt.hour.fillna(0).astype(int) // 4
    # Purely pre-trade environment bins. These are not target encodings yet.
    out["cluster_1h4h_state"] = (
        out.get("shock_1h_candle_up", 0).astype(int).astype(str) + "_" +
        out.get("shock_4h_candle_up", 0).astype(int).astype(str) + "_" +
        out.get("shock_1h_trend_up", 0).astype(int).astype(str) + "_" +
        out.get("shock_4h_trend_up", 0).astype(int).astype(str)
    )
    out["cluster_shock_position"] = (
        pd.cut(pd.to_numeric(out.get("shock_1h_pos20", 0.5), errors="coerce"), bins=[-0.01, 0.25, 0.75, 1.01], labels=["low", "mid", "high"]).astype(str) + "_" +
        pd.cut(pd.to_numeric(out.get("shock_4h_pos20", 0.5), errors="coerce"), bins=[-0.01, 0.25, 0.75, 1.01], labels=["low", "mid", "high"]).astype(str) + "_" +
        out.get("shock_1h_big_body_default", 0).astype(int).astype(str) + "_" +
        out.get("shock_4h_big_body_default", 0).astype(int).astype(str)
    )
    out["cluster_time_vol"] = (
        out["bj_hour_bucket"].astype(str) + "_" +
        pd.cut(pd.to_numeric(out.get("shock_1h_range_q", 0.5), errors="coerce"), bins=[-0.01, 0.35, 0.65, 1.01], labels=["calm", "mid", "wide"]).astype(str) + "_" +
        pd.cut(pd.to_numeric(out.get("shock_4h_range_q", 0.5), errors="coerce"), bins=[-0.01, 0.35, 0.65, 1.01], labels=["calm", "mid", "wide"]).astype(str)
    )
    # Numeric safe bins for tree models.
    safe = ["bj_hour_bucket"]
    for col in ["cluster_1h4h_state", "cluster_shock_position", "cluster_time_vol"]:
        codes, _ = pd.factorize(out[col].astype(str), sort=True)
        ncol = f"{col}_code"
        out[ncol] = codes.astype(float)
        safe.append(ncol)
    return out, safe


def oof_target_encode(train: pd.DataFrame, val: pd.DataFrame, keys: list[str], n_folds: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Time-safe target encodings for wrong-cluster/environment priors.

    Training rows use only earlier folds. Validation rows use the full training
    window. This makes the feature fold-safe and prevents validation labels from
    defining bad clusters.
    """
    tr = train.copy().sort_values("dt").reset_index(drop=True)
    va = val.copy().sort_values("dt").reset_index(drop=True)
    y = tr["label_up"].astype(float)
    global_mean = float(y.mean()) if len(y) else 0.5
    added: list[str] = []
    fold_id = np.floor(np.arange(len(tr)) * n_folds / max(1, len(tr))).astype(int)
    for key in keys:
        rate_col = f"te_{key}_up_rate"
        cnt_col = f"te_{key}_count_log"
        tr[rate_col] = global_mean
        tr[cnt_col] = 0.0
        for fold in range(n_folds):
            hist = tr[fold_id < fold]
            idx = fold_id == fold
            if hist.empty or not idx.any():
                continue
            stats = hist.groupby(key)["label_up"].agg(["mean", "count"])
            tr.loc[idx, rate_col] = tr.loc[idx, key].map(stats["mean"]).fillna(global_mean).astype(float)
            tr.loc[idx, cnt_col] = np.log1p(tr.loc[idx, key].map(stats["count"]).fillna(0).astype(float))
        stats_full = tr.groupby(key)["label_up"].agg(["mean", "count"])
        va[rate_col] = va[key].map(stats_full["mean"]).fillna(global_mean).astype(float)
        va[cnt_col] = np.log1p(va[key].map(stats_full["count"]).fillna(0).astype(float))
        added += [rate_col, cnt_col]
    return tr, va, added


def build_shock_aware_frame() -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    df, base_features, truth = M.build_integrated_frame()
    df, shock_features, shock_truth = attach_closed_candle_shock_features(df)
    df, cluster_safe = add_environment_bins(df)
    features = list(dict.fromkeys(base_features + shock_features + cluster_safe))
    forbidden = set(M.base.FORBIDDEN_FEATURES)
    prefixes = tuple(M.base.FORBIDDEN_PREFIXES)
    clean: list[str] = []
    for c in features:
        if c in forbidden or c.startswith(prefixes):
            continue
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if int(s.notna().sum()) >= max(500, int(len(df) * 0.50)):
                df[c] = s
                clean.append(c)
    data_truth = {
        **truth,
        **shock_truth,
        "finalRows": int(len(df)),
        "baseFeatureCount": int(len(base_features)),
        "shockFeatureCount": int(len(shock_features)),
        "clusterSafeFeatureCount": int(len(cluster_safe)),
        "finalFeatureCount": int(len(clean)),
        "wrongClusterFeaturePolicy": "fold-safe target encodings are generated inside each train/validation split; validation labels never define encodings",
    }
    return df.dropna(subset=["dt", "label_up"]).sort_values("dt").reset_index(drop=True), clean, data_truth


def split_train_val(df: pd.DataFrame, window: str, train_window: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    end = df["dt"].max()
    days = 180 if window == "180d" else 365
    start = end - pd.Timedelta(days=days)
    val = df[df["dt"] >= start].copy()
    if train_window == "full":
        train = df[df["dt"] < start].copy()
    else:
        train_days = {"1y": 365, "3y": 1095, "5y": 1825}[train_window]
        train = df[(df["dt"] < start) & (df["dt"] >= start - pd.Timedelta(days=train_days))].copy()
    if MAX_TRAIN_ROWS > 0 and len(train) > MAX_TRAIN_ROWS:
        train = train.sort_values("dt").iloc[-MAX_TRAIN_ROWS:].copy()
    # Purge adjacent samples between train/validation.
    embargo_start = start - pd.Timedelta(hours=4)
    train = train[train["dt"] < embargo_start].copy()
    return train, val


def feature_subset(features: list[str], mode: str) -> list[str]:
    if mode == "base15":
        out = [c for c in features if not c.startswith(("1h_", "4h_", "daily_", "shock_", "te_", "cluster_"))]
    elif mode == "base_plus_shock":
        out = [c for c in features if c.startswith("shock_") or not c.startswith(("1h_", "4h_", "daily_", "te_", "cluster_"))]
    elif mode == "base_plus_1h4h_shock":
        out = [c for c in features if not c.startswith(("daily_", "te_"))]
    elif mode == "base_plus_all_shock":
        out = [c for c in features if not c.startswith("te_")]
    elif mode == "shock_only":
        out = [c for c in features if c.startswith("shock_") or c.startswith("cluster_")]
    elif mode == "base_plus_shock_cluster":
        out = list(features)
    else:
        out = list(features)
    return out


def fit_model(engine: str, train: pd.DataFrame, feats: list[str], params: dict[str, Any]):
    return M.fit_model(engine, train, feats, params)


def predict(model: Any, val: pd.DataFrame, feats: list[str]) -> np.ndarray:
    return M.predict(model, val, feats)


def select_rows(val: pd.DataFrame, prob: np.ndarray, params: dict[str, Any]) -> pd.DataFrame:
    return M.select_rows(val, prob, params)


def evaluate_params(df: pd.DataFrame, features: list[str], params: dict[str, Any], name: str) -> dict[str, Any] | None:
    base_feats = feature_subset(features, params["feature_mode"])
    max_features = int(params.get("max_features", 120))
    if max_features > 0:
        # Preserve shock features while still limiting the wide historical feature pool.
        shock_feats = [c for c in base_feats if c.startswith(("shock_", "cluster_"))]
        other_feats = [c for c in base_feats if c not in shock_feats]
        base_feats = list(dict.fromkeys(other_feats[:max_features] + shock_feats[:80]))
    if len(base_feats) < 10:
        return None
    rows: list[dict[str, Any]] = []
    for window in ["180d", "365d"]:
        train, val = split_train_val(df, window, params["train_window"])
        feats = list(base_feats)
        if params.get("use_cluster_te", False):
            train, val, te_feats = oof_target_encode(train, val, ["cluster_1h4h_state", "cluster_shock_position", "cluster_time_vol"])
            feats = list(dict.fromkeys(feats + te_feats))
        if len(train) < 1000 or len(val) < 100:
            return None
        model = fit_model(params["engine"], train, feats, params)
        if model is None:
            return None
        prob = predict(model, val, feats)
        selected = select_rows(val, prob, params)
        if selected.empty:
            return None
        row = M.aux.curve_metrics(selected, name, window, "shock_aware_integrated_main")
        row.update({
            "engine": params["engine"],
            "trainWindow": params["train_window"],
            "featureMode": params["feature_mode"],
            "edge": params["edge"],
            "featureCount": int(len(feats)),
            "useClusterTE": bool(params.get("use_cluster_te", False)),
            "compoundPnlP50": round(float(row["endingBankrollP50"] - 850.0), 6),
        })
        rows.append(row)
    return {"name": name, "params": params, "rows": rows, "featureCount": int(rows[0]["featureCount"])}


def current_base_rows() -> dict[str, dict[str, Any]]:
    return M.base_rows()


def pass_gate(candidate: dict[str, Any], base_by_window: dict[str, dict[str, Any]]) -> tuple[bool, list[str]]:
    by = {r["window"]: r for r in candidate["rows"]}
    reasons: list[str] = []
    for w in ["180d", "365d"]:
        r = by[w]
        b = base_by_window[w]
        if r["trades"] < max(100 if w == "180d" else 200, int(b["trades"] * 0.45)):
            reasons.append(f"{w}_trade_count_too_low")
        if r["fak2EndingBankroll"] <= 850:
            reasons.append(f"{w}_fak2_not_profitable")
        if r["endingBankrollP50"] < b["endingBankrollP50"] * 0.995 and r["endingBankrollP5"] <= b["endingBankrollP5"]:
            reasons.append(f"{w}_toxicity_not_better")
        if r["maxDrawdownP50"] > b["maxDrawdownP50"] * 1.10:
            reasons.append(f"{w}_drawdown_worse")
    return not reasons, reasons


def score_candidate(candidate: dict[str, Any], base_by_window: dict[str, dict[str, Any]]) -> float:
    by = {r["window"]: r for r in candidate["rows"]}
    gain_p50 = sum(by[w]["endingBankrollP50"] - base_by_window[w]["endingBankrollP50"] for w in ["180d", "365d"])
    gain_p5 = sum(by[w]["endingBankrollP5"] - base_by_window[w]["endingBankrollP5"] for w in ["180d", "365d"])
    fak = sum(by[w]["fak2EndingBankroll"] - 850.0 for w in ["180d", "365d"])
    win = by["180d"]["winRatePct"] + by["365d"]["winRatePct"]
    dd = by["365d"]["maxDrawdownP50"]
    return gain_p50 * 2.0 + gain_p5 + fak * 0.2 + win * 20.0 - dd * 0.4


def param_grid() -> list[dict[str, Any]]:
    engines = [x for x in os.environ.get("TOP159_SHOCK_AWARE_ENGINES", "logistic,lightgbm,xgboost,catboost").split(",") if x]
    rows: list[dict[str, Any]] = []
    feature_modes = ["base15", "base_plus_shock", "base_plus_1h4h_shock", "base_plus_all_shock", "base_plus_shock_cluster"]
    train_windows = ["1y", "3y", "5y", "full"]
    edges = [0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.08]
    max_features_grid = [60, 100, 160, 240]
    score_bands = [("all", 1.0), ("mid_only", 0.72), ("mid_only", 0.76), ("mid_only", 0.80)]
    cluster_flags = [False, True]
    for engine, tw, fm, edge, mf, band, use_te in itertools.product(engines, train_windows, feature_modes, edges, max_features_grid, score_bands, cluster_flags):
        score_band, max_score = band
        if engine == "logistic":
            for c in [0.08, 0.15, 0.35, 0.75, 1.5, 3.0]:
                rows.append({
                    "engine": engine, "train_window": tw, "feature_mode": fm, "edge": edge,
                    "max_features": mf, "score_band": score_band, "max_score": max_score,
                    "use_cluster_te": use_te, "C": c,
                    "n_estimators": 1, "learning_rate": 0.04, "num_leaves": 16,
                    "min_child_samples": 80, "subsample": 0.9, "colsample_bytree": 0.9,
                    "reg_lambda": 1.0, "depth": 3,
                })
        else:
            presets = [
                (80, 0.025, 12, 80, 1.0, 3),
                (120, 0.035, 16, 80, 1.0, 4),
                (160, 0.025, 24, 120, 2.0, 4),
                (220, 0.018, 32, 160, 3.0, 5),
            ]
            for ne, lr, leaves, mcs, reg, depth in presets:
                rows.append({
                    "engine": engine, "train_window": tw, "feature_mode": fm, "edge": edge,
                    "max_features": mf, "score_band": score_band, "max_score": max_score,
                    "use_cluster_te": use_te,
                    "n_estimators": ne, "learning_rate": lr, "num_leaves": leaves,
                    "min_child_samples": mcs, "subsample": 0.88, "colsample_bytree": 0.88,
                    "reg_lambda": reg, "depth": depth, "C": 1.0,
                })
    if PARAM_LIMIT > 0 and len(rows) > PARAM_LIMIT:
        rng = np.random.default_rng(RNG_SEED)
        rows = [rows[int(i)] for i in rng.permutation(len(rows))[:PARAM_LIMIT]]
    return rows


_DF: pd.DataFrame | None = None
_FEATURES: list[str] | None = None


def init_worker(df: pd.DataFrame, features: list[str]) -> None:
    global _DF, _FEATURES
    _DF = df
    _FEATURES = features
    M.aux.MC_TRIALS = MC_TRIALS
    M.base.TRIALS = MC_TRIALS


def eval_worker(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    if _DF is None or _FEATURES is None:
        raise RuntimeError("worker not initialized")
    idx, params = item
    name = f"shockaware_{params['engine']}_{params['train_window']}_{params['feature_mode']}_edge{params['edge']}_{stable_hash(params)}"
    try:
        out = evaluate_params(_DF, _FEATURES, params, name)
        return out if out is not None else {"name": name, "params": params, "error": "empty_or_fit_failed"}
    except Exception as exc:
        return {"name": name, "params": params, "error": repr(exc)[:800]}


def archived_rows_for_candidate(df: pd.DataFrame, features: list[str], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        fill_rows, scan_audit = archive.load_live_fill_rows()
        all_eth = archive.aggregate_logical(fill_rows, "全部归档ETH15m live去重")
        strict_mask = fill_rows["sourcePath"].str.contains("slot01|forcedtrade|base_forcedtrade|__live_eth|logs_v5_exp10_bp0480__live_eth", case=False, regex=True)
        strict = archive.aggregate_logical(fill_rows[strict_mask].copy(), "严格旧slot1/ETH相关live去重") if strict_mask.any() else all_eth.iloc[0:0]
        old_scopes = [strict, all_eth]
        rows: list[dict[str, Any]] = []
        params = candidate["params"]
        feats0 = feature_subset(features, params["feature_mode"])
        if params.get("max_features", 0):
            feats0 = feats0[: int(params["max_features"])]
        for old in old_scopes:
            if old.empty:
                continue
            start = old["marketStart"].min()
            end = old["marketStart"].max()
            train = df[df["dt"] < start].copy()
            if params["train_window"] != "full":
                days = {"1y": 365, "3y": 1095, "5y": 1825}[params["train_window"]]
                train = train[train["dt"] >= start - pd.Timedelta(days=days)].copy()
            val = df[(df["dt"] >= start) & (df["dt"] <= end)].copy()
            feats = list(feats0)
            if params.get("use_cluster_te", False):
                train, val, te_feats = oof_target_encode(train, val, ["cluster_1h4h_state", "cluster_shock_position", "cluster_time_vol"])
                feats = list(dict.fromkeys(feats + te_feats))
            model = fit_model(params["engine"], train, feats, params)
            if model is None:
                continue
            prob = predict(model, val, feats)
            sel = select_rows(val, prob, params)
            pred = sel[["dt", "pred_up15", "score15"]].copy()
            merged = old.merge(pred, left_on="marketStart", right_on="dt", how="left")
            chosen = merged[merged["pred_up15"].notna()].copy().sort_values("marketStart")
            if chosen.empty:
                rows.append({"scope": old["scopeName"].iloc[0], "name": candidate["name"], "oldRealMarkets": int(len(old)), "selectedTrades": 0})
                continue
            model_won = chosen["pred_up15"].astype(bool).to_numpy() == chosen["actualUp"].astype(bool).to_numpy()
            one = np.where(model_won, 1.0, -1.0)
            fak52 = np.where(model_won, (1.0 / 0.52) - 1.0, -1.0)
            same_dir = chosen["pred_up15"].astype(bool).to_numpy() == (chosen["direction"].astype(str).str.upper() == "UP").to_numpy()
            same = chosen[same_dir]
            rows.append({
                "scope": old["scopeName"].iloc[0],
                "name": candidate["name"],
                "oldRealMarkets": int(len(old)),
                "selectedTrades": int(len(chosen)),
                "skippedTrades": int(len(old) - len(chosen)),
                "wins": int(model_won.sum()),
                "losses": int(len(model_won) - int(model_won.sum())),
                "winRatePct": round(100.0 * float(model_won.mean()), 6),
                "oneUnitPnl": round(float(one.sum()), 6),
                "oneUnitMaxDrawdown": archive.max_drawdown(one.tolist()),
                "fak52Pnl": round(float(fak52.sum()), 6),
                "fak52MaxDrawdown": archive.max_drawdown(fak52.tolist()),
                "sameDirectionExecutableTrades": int(len(same)),
                "sameDirectionActualPnlUsd": round(float(same["pnl"].astype(float).sum()), 6) if len(same) else 0.0,
                "sameDirectionActualMaxDrawdownUsd": archive.max_drawdown(same["pnl"].astype(float).tolist()) if len(same) else 0.0,
                "avgScore": round(float(chosen["score15"].mean()), 6),
                "setHash": stable_hash(chosen[["marketSlug", "pred_up15", "actualUp"]].to_dict("records")),
            })
        return rows
    except Exception as exc:
        return [{"scope": "archive_error", "name": candidate.get("name"), "error": repr(exc)[:500]}]


def bug_audit(df: pd.DataFrame, features: list[str], base_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    params = {
        "engine": "logistic", "train_window": "3y", "feature_mode": "base_plus_shock",
        "edge": 0.05, "max_features": 100, "score_band": "all", "max_score": 1.0,
        "use_cluster_te": True, "C": 0.75, "n_estimators": 1, "learning_rate": 0.04,
        "num_leaves": 16, "min_child_samples": 80, "subsample": 0.9, "colsample_bytree": 0.9,
        "reg_lambda": 1.0, "depth": 3,
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
                "p50_1": a["endingBankrollP50"],
                "p50_2": b["endingBankrollP50"],
                "passed": a["setHash"] == b["setHash"] and a["endingBankrollP50"] == b["endingBankrollP50"],
            })
    forbidden_hits = [c for c in features if c in M.base.FORBIDDEN_FEATURES or c.startswith(tuple(M.base.FORBIDDEN_PREFIXES))]
    random_label = {"status": "not_run", "passed": False}
    try:
        train, val = split_train_val(df, "365d", "3y")
        feats = feature_subset(features, "base_plus_shock")[:100]
        train, val, te_feats = oof_target_encode(train, val, ["cluster_1h4h_state", "cluster_shock_position", "cluster_time_vol"])
        feats = list(dict.fromkeys(feats + te_feats))
        model = M.fit_model("logistic", train.assign(label_up=np.random.default_rng(RNG_SEED).permutation(train["label_up"].astype(int))), feats, params)
        if model is not None:
            prob = M.predict(model, val, feats)
            sel = M.select_rows(val, prob, params)
            wr = float(sel["won"].mean() * 100.0) if len(sel) else 0.0
            random_label = {"status": "ok", "selectedTrades": int(len(sel)), "winRatePct": round(wr, 6), "passed": len(sel) < 50 or 43.0 <= wr <= 57.0}
    except Exception as exc:
        random_label = {"status": "error", "error": repr(exc)[:300], "passed": False}
    audit = {
        "generatedAt": now_iso(),
        "beijingTime": bj_now(),
        "researchOnlyNoLiveChange": True,
        "closedCandlePolicy": "1h/4h shock features use available_at=open_time+interval and merge_asof backward",
        "purgedWalkForward": "validation train windows embargo four hours before validation start",
        "wrongClusterEncoding": "time-safe OOF target encoding on train folds; validation encoded from training only",
        "forbiddenFeatureHits": forbidden_hits,
        "repeatability": repeat,
        "randomLabelAudit": random_label,
        "baseRows": base_rows,
    }
    audit["passed"] = not forbidden_hits and bool(repeat) and all(r["passed"] for r in repeat) and bool(random_label.get("passed"))
    write_json(OUT_AUDIT_JSON, audit)
    lines = [
        "# top159 冲击特征一体化主模型：回测器审计",
        "",
        f"- 北京时间：`{audit['beijingTime']}`",
        f"- 研究动作：`research_only_no_live_change`",
        f"- 审计结论：`{'通过' if audit['passed'] else '未通过'}`",
        f"- 未来字段命中：`{forbidden_hits}`",
        f"- 随机标签测试：`{random_label}`",
    ]
    write_text(OUT_AUDIT_MD, "\n".join(lines) + "\n")
    return audit


def compact_candidate(c: dict[str, Any], base_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ok, reasons = pass_gate(c, base_rows)
    return {
        "name": c["name"],
        "params": c["params"],
        "rows": c["rows"],
        "passed": ok,
        "reasons": reasons,
        "score": score_candidate(c, base_rows),
    }


def load_existing_results() -> list[dict[str, Any]]:
    rows = []
    if OUT_RESULTS.exists():
        for line in OUT_RESULTS.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def write_progress(results: list[dict[str, Any]], base_rows: dict[str, dict[str, Any]], total: int, started: float, data_truth: dict[str, Any], audit: dict[str, Any], finished: bool = False) -> None:
    valid = [r for r in results if r.get("rows")]
    candidates = [compact_candidate(r, base_rows) for r in valid]
    candidates.sort(key=lambda r: r["score"], reverse=True)
    strict = [c for c in candidates if c["passed"]]
    selected = strict[0] if strict else (candidates[0] if candidates else None)
    archive_rows: list[dict[str, Any]] = []
    if selected:
        # Only run archive compare for the current best to keep checkpoints light.
        try:
            global _DF, _FEATURES
            if _DF is not None and _FEATURES is not None:
                archive_rows = archived_rows_for_candidate(_DF, _FEATURES, selected)
        except Exception as exc:
            archive_rows = [{"scope": "archive_error", "error": repr(exc)[:500]}]
    payload = {
        "generatedAt": now_iso(),
        "beijingTime": bj_now(),
        "finished": finished,
        "elapsedSeconds": round(time.time() - started, 3),
        "workers": WORKERS,
        "workerThreads": THREADS,
        "totalCandidates": total,
        "doneCount": len(results),
        "validCount": len(valid),
        "strictPassCount": len(strict),
        "dataTruth": data_truth,
        "audit": audit,
        "baseRows": base_rows,
        "topCandidates": candidates[:300],
        "strictPass": strict[:100],
        "selectedForArchiveCompare": selected,
        "archivedRealRowsForSelected": archive_rows,
        "liveConfigMutated": False,
    }
    write_json(OUT_CHECKPOINT, payload)
    write_json(OUT_LEADER_JSON, payload)
    write_json(OUT_COMPARE_JSON, payload)
    verdict = {
        "generatedAt": payload["generatedAt"],
        "status": "strict_candidate_found" if strict else ("best_candidate_observation_only" if selected else "no_candidate"),
        "selected": selected,
        "archivedRealRowsForSelected": archive_rows,
        "liveAction": "research_only_no_live_change",
    }
    write_json(OUT_VERDICT_JSON, verdict)
    render_markdown(payload, verdict)


def fmt(v: Any, nd: int = 2) -> str:
    if v is None:
        return "-"
    try:
        f = float(v)
        if not np.isfinite(f):
            return "-"
        return f"{f:,.{nd}f}"
    except Exception:
        return str(v)


def render_markdown(payload: dict[str, Any], verdict: dict[str, Any]) -> None:
    lines = [
        "# top159 冲击特征一体化主模型排行榜",
        "",
        f"- 北京时间：`{payload['beijingTime']}`",
        f"- 完成：`{payload['doneCount']}/{payload['totalCandidates']}`",
        f"- 严格通过：`{payload['strictPassCount']}`",
        "- live动作：`research_only_no_live_change`",
        "",
        "|配置|窗口|训练窗|模型|特征组|交易数|胜/负|胜率|盈亏P50|期末P50|P5|P95|最大回撤|收益回撤比|FAK+2资金|月正收益|保留率|哈希|",
        "|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    base_rows = payload["baseRows"]
    for r in [base_rows["180d"], base_rows["365d"]]:
        pnl = r["endingBankrollP50"] - 850.0
        ratio = pnl / r["maxDrawdownP50"] if r["maxDrawdownP50"] else 0
        lines.append(f"|current_top159|{r['window']}|-|baseline|-|{r['trades']}|{r['wins']}/{r['losses']}|{fmt(r['winRatePct'])}|{fmt(pnl)}|{fmt(r['endingBankrollP50'])}|{fmt(r['endingBankrollP5'])}|{fmt(r['endingBankrollP95'])}|{fmt(r['maxDrawdownP50'])}|{fmt(ratio)}|{fmt(r['fak2EndingBankroll'])}|{fmt(r['monthlyPositiveRatioP50'])}|100.00|`{r['setHash']}`|")
    for c in payload.get("topCandidates", [])[:35]:
        for r in c["rows"]:
            p = c["params"]
            b = base_rows[r["window"]]
            pnl = r["endingBankrollP50"] - 850.0
            ratio = pnl / r["maxDrawdownP50"] if r["maxDrawdownP50"] else 0
            retention = 100.0 * r["trades"] / max(1, b["trades"])
            lines.append(f"|{c['name']}|{r['window']}|{p.get('train_window')}|{p.get('engine')}|{p.get('feature_mode')}|{r['trades']}|{r['wins']}/{r['losses']}|{fmt(r['winRatePct'])}|{fmt(pnl)}|{fmt(r['endingBankrollP50'])}|{fmt(r['endingBankrollP5'])}|{fmt(r['endingBankrollP95'])}|{fmt(r['maxDrawdownP50'])}|{fmt(ratio)}|{fmt(r['fak2EndingBankroll'])}|{fmt(r['monthlyPositiveRatioP50'])}|{fmt(retention)}|`{r['setHash']}`|")
    write_text(OUT_LEADER_MD, "\n".join(lines) + "\n")

    compare = [
        "# top159 冲击特征一体化主模型：180/365/历史归档对比",
        "",
        f"- 北京时间：`{payload['beijingTime']}`",
        "- 历史归档说明：旧真实交易不是新模型真实成交，只是同一历史市场窗口上的纯方向/同向成交复用检验。",
        "",
        "## 180天 / 365天",
        "",
    ]
    compare.extend(lines[7:])
    compare += ["", "## 历史归档真实单（当前最佳候选）", "", "|范围|配置|旧市场数|选中交易|胜/负|胜率|1U盈亏|1U回撤|FAK0.52盈亏|FAK0.52回撤|同向旧实盘笔数|同向旧实盘盈亏|同向旧实盘回撤|平均分|哈希|", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in payload.get("archivedRealRowsForSelected", []):
        if r.get("error"):
            compare.append(f"|{r.get('scope')}|{r.get('name')}|-|-|-|-|-|-|-|-|-|-|-|-|{r.get('error')}|")
        else:
            compare.append(f"|{r.get('scope')}|{r.get('name')}|{r.get('oldRealMarkets')}|{r.get('selectedTrades')}|{r.get('wins')}/{r.get('losses')}|{fmt(r.get('winRatePct'))}|{fmt(r.get('oneUnitPnl'))}|{fmt(r.get('oneUnitMaxDrawdown'))}|{fmt(r.get('fak52Pnl'))}|{fmt(r.get('fak52MaxDrawdown'))}|{r.get('sameDirectionExecutableTrades')}|{fmt(r.get('sameDirectionActualPnlUsd'))}|{fmt(r.get('sameDirectionActualMaxDrawdownUsd'))}|{fmt(r.get('avgScore'), 4)}|`{r.get('setHash')}`|")
    write_text(OUT_COMPARE_MD, "\n".join(compare) + "\n")

    vlines = [
        "# top159 冲击特征一体化主模型唯一结论",
        "",
        f"- 北京时间：`{payload['beijingTime']}`",
        f"- 状态：`{verdict['status']}`",
        "- live动作：`research_only_no_live_change`",
    ]
    sel = verdict.get("selected")
    if sel:
        vlines += [
            f"- 当前选中：`{sel['name']}`",
            f"- 参数：`{json.dumps(sel['params'], ensure_ascii=False, sort_keys=True)}`",
            f"- 对比表：`{OUT_COMPARE_MD}`",
        ]
    else:
        vlines.append("- 没有有效候选。")
    write_text(OUT_VERDICT_MD, "\n".join(vlines) + "\n")


def run() -> int:
    started = time.time()
    print(f"[shock-aware-integrated] start {bj_now()} workers={WORKERS} maxSeconds={MAX_SECONDS}", flush=True)
    df, features, truth = build_shock_aware_frame()
    base_rows = current_base_rows()
    audit = bug_audit(df, features, base_rows)
    print(f"[shock-aware-integrated] rows={len(df)} features={len(features)} auditPassed={audit['passed']}", flush=True)
    params = param_grid()
    total = len(params)
    existing = load_existing_results()
    done = {r.get("name") for r in existing if r.get("name")}
    pending: list[tuple[int, dict[str, Any]]] = []
    for i, p in enumerate(params):
        name = f"shockaware_{p['engine']}_{p['train_window']}_{p['feature_mode']}_edge{p['edge']}_{stable_hash(p)}"
        if name not in done:
            pending.append((i, p))
    print(f"[shock-aware-integrated] total={total} existing={len(existing)} pending={len(pending)}", flush=True)
    global _DF, _FEATURES
    _DF, _FEATURES = df, features
    write_progress(existing, base_rows, total, started, truth, audit, finished=False)
    last = time.time()
    results = list(existing)
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
                done_futs, _ = cf.wait(futures, timeout=5, return_when=cf.FIRST_COMPLETED)
                for fut in done_futs:
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
                    write_progress(results, base_rows, total, started, truth, audit, finished=False)
                    print(f"[shock-aware-integrated] checkpoint {len(results)}/{total}", flush=True)
                    last = time.time()
            for fut in futures:
                fut.cancel()
    finished = len(results) >= total
    write_progress(results, base_rows, total, started, truth, audit, finished=finished)
    print(json.dumps({
        "status": "finished" if finished else "checkpointed",
        "done": len(results),
        "total": total,
        "leaderboard": str(OUT_LEADER_MD),
        "compare": str(OUT_COMPARE_MD),
        "verdict": str(OUT_VERDICT_MD),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
