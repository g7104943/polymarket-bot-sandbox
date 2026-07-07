#!/usr/bin/env python3
from __future__ import annotations

"""BTC near-ETH061 multi-route research search.

Research-only. Does not mutate live configs, ledgers, orders, claim state,
monitor settings, or the current ETH061 live trading process.

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
from pathlib import Path
from typing import Any

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(key, os.environ.get("BTC_NEAR061_THREADS", "1"))

import numpy as np
import pandas as pd

ROOT = Path("/Users/mac/polyfun")
REPORTS = ROOT / "reports"
SCRIPTS = ROOT / "polyfun-next" / "scripts"
BTC061_SCRIPT = SCRIPTS / "run_btc061_method_search.py"
RAW_GROWTH_SCRIPT = SCRIPTS / "run_crypto_raw_growth_multimarket_search.py"

START_BANKROLL = 850.0
STAKE_PCT = 0.01
PRIMARY_BUY_PRICE = 0.55
PRESSURE_BUY_PRICE = 0.60
RNG_SEED = 20260509
WORKERS = int(os.environ.get("BTC_NEAR061_WORKERS", "6"))
THREADS = int(os.environ.get("BTC_NEAR061_THREADS", "1"))
MAX_SECONDS = int(os.environ.get("BTC_NEAR061_MAX_SECONDS", str(2 * 3600)))
PARAM_LIMIT = int(os.environ.get("BTC_NEAR061_PARAM_LIMIT", "0"))
TOP_FOR_ARCHIVE = int(os.environ.get("BTC_NEAR061_TOP_FOR_ARCHIVE", "10"))
TOP_MAIN_FOR_FILTER = int(os.environ.get("BTC_NEAR061_TOP_MAIN_FOR_FILTER", "12"))
MAX_FILTERS_PER_MAIN = int(os.environ.get("BTC_NEAR061_MAX_FILTERS_PER_MAIN", "12000"))

OUT_AUDIT_MD = REPORTS / "btc_near061_multiroute_bug_audit_latest.md"
OUT_AUDIT_JSON = REPORTS / "btc_near061_multiroute_bug_audit_latest.json"
OUT_RESULTS = REPORTS / "btc_near061_multiroute_results_latest.jsonl"
OUT_CHECKPOINT = REPORTS / "btc_near061_multiroute_checkpoint_latest.json"
OUT_LEADERBOARD = REPORTS / "btc_near061_multiroute_leaderboard_latest.json"
OUT_COMPARE_MD = REPORTS / "btc_near061_vs_eth061_compare_latest.md"
OUT_COMPARE_JSON = REPORTS / "btc_near061_vs_eth061_compare_latest.json"
OUT_VERDICT_MD = REPORTS / "btc_near061_unique_verdict_latest.md"
OUT_VERDICT_JSON = REPORTS / "btc_near061_unique_verdict_latest.json"

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


btc061 = load_module("btc_near061_old_method", BTC061_SCRIPT)
raw_growth = load_module("btc_near061_raw_growth", RAW_GROWTH_SCRIPT)

G_BASE_FRAME: tuple[pd.DataFrame, list[str], dict[str, Any]] | None = None
G_CROSS_FRAME: tuple[pd.DataFrame, list[str], dict[str, Any]] | None = None


def init_worker() -> None:
    global G_BASE_FRAME, G_CROSS_FRAME
    # Build once per worker; this is much faster than rebuilding cross-asset features for every candidate.
    G_BASE_FRAME = build_base_frame()
    G_CROSS_FRAME = build_cross_frame()


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


def prefix_merge_asof(left: pd.DataFrame, right: pd.DataFrame, cols: list[str], prefix: str) -> tuple[pd.DataFrame, list[str]]:
    r = right[["dt"] + cols].copy().sort_values("dt")
    rename = {c: f"{prefix}_{c}" for c in cols}
    r = r.rename(columns=rename)
    l = left.sort_values("dt").copy()
    l["dt"] = pd.to_datetime(l["dt"], utc=True, errors="coerce")
    r["dt"] = pd.to_datetime(r["dt"], utc=True, errors="coerce")
    out = pd.merge_asof(l, r, on="dt", direction="backward")
    return out, list(rename.values())


def build_base_frame() -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    return btc061.build_btc_frame()


def build_cross_frame() -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    df, features, truth = build_base_frame()
    added_all: list[str] = []
    for asset in ["ETH", "SOL", "XRP"]:
        try:
            odf, ofeats, otruth = raw_growth.build_frame(asset, "15m")
        except Exception as exc:
            truth[f"cross{asset}Error"] = repr(exc)[:300]
            continue
        # Keep compact, high-signal cross-market features only.
        keys = ("ret_", "ema_", "ema_dist_", "rsi", "bb_pos", "range_", "vol_", "volume_z")
        cols = [c for c in ofeats if any(k in c for k in keys)][:90]
        df, added = prefix_merge_asof(df, odf, cols, asset.lower())
        added_all.extend(added)
        truth[f"cross{asset}Rows"] = int(len(odf))
        truth[f"cross{asset}FeatureCount"] = int(len(added))
    # Relative-strength handcrafted features. All underlying columns are shifted/aligned already.
    rel_cols: list[str] = []
    for asset in ["eth", "sol", "xrp"]:
        for col in ["ret_1", "ret_4", "ret_16", "ema_8_32", "bb_pos", "range_16"]:
            oc = f"{asset}_{col}"
            if col in df.columns and oc in df.columns:
                rc = f"rel_btc_minus_{asset}_{col}"
                df[rc] = pd.to_numeric(df[col], errors="coerce") - pd.to_numeric(df[oc], errors="coerce")
                rel_cols.append(rc)
    all_features = features + added_all + rel_cols
    clean: list[str] = []
    forbidden = set(getattr(raw_growth.base, "FORBIDDEN_FEATURES", set()))
    forbidden_prefix = tuple(getattr(raw_growth.base, "FORBIDDEN_PREFIXES", ()))
    for c in all_features:
        if c in forbidden or c.startswith(forbidden_prefix):
            continue
        s = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if int(s.notna().sum()) >= max(500, int(len(df) * 0.45)):
            df[c] = s
            clean.append(c)
    truth["crossFinalFeatureCount"] = len(clean)
    return df.sort_values("dt").reset_index(drop=True), clean, truth


def feature_subset(features: list[str], mode: str) -> list[str]:
    if mode == "relative_strength":
        keys = ("rel_btc_minus", "eth_", "sol_", "xrp_", "ret_", "ema_", "bb_pos", "range_", "vol_")
        return [c for c in features if any(k in c for k in keys)]
    return raw_growth.feature_subset(features, mode)


def split_train_val(df: pd.DataFrame, window: str, train_window: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    train, val = raw_growth.split_train_val(df, window, train_window)
    return train, val, val["dt"].min()


def fit_model(params: dict[str, Any], train: pd.DataFrame, features: list[str]):
    return raw_growth.fit_model(params["engine"], train, features, params)


def predict(model: Any, val: pd.DataFrame, features: list[str]) -> np.ndarray:
    return raw_growth.predict(model, val, features)


def select_standard(val: pd.DataFrame, prob: np.ndarray, params: dict[str, Any]) -> pd.DataFrame:
    return raw_growth.select_rows(val, prob, params)


def select_dynamic(val: pd.DataFrame, train: pd.DataFrame, prob: np.ndarray, params: dict[str, Any]) -> pd.DataFrame:
    pred_up = prob >= 0.5
    score = np.maximum(prob, 1.0 - prob)
    req = np.full(len(val), float(params.get("threshold", 0.55)), dtype=float)
    vol_col = next((c for c in ["vol_16", "range_16", "1h_range_16", "4h_range_16"] if c in val.columns and c in train.columns), None)
    if vol_col:
        tv = pd.to_numeric(train[vol_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(tv):
            lo = float(tv.quantile(0.33)); hi = float(tv.quantile(0.66))
            vv = pd.to_numeric(val[vol_col], errors="coerce").fillna(tv.median()).to_numpy()
            req = np.where(vv <= lo, float(params.get("thr_low_vol", req[0])), req)
            req = np.where((vv > lo) & (vv <= hi), float(params.get("thr_mid_vol", req[0])), req)
            req = np.where(vv > hi, float(params.get("thr_high_vol", req[0])), req)
    trend_col = next((c for c in ["1h_ema_8_32", "4h_ema_8_32", "ema_8_32"] if c in val.columns), None)
    if trend_col:
        trend = pd.to_numeric(val[trend_col], errors="coerce").fillna(0.0).to_numpy()
        same = np.sign(trend) == np.where(pred_up, 1.0, -1.0)
        req = np.where(same, req + float(params.get("same_trend_adj", 0.0)), req)
        req = np.where(~same, req + float(params.get("opp_trend_adj", 0.0)), req)
    if params.get("directional", False):
        req = np.where(pred_up, req + float(params.get("up_adj", 0.0)), req + float(params.get("down_adj", 0.0)))
    max_score = float(params.get("max_score", 1.0))
    mask = (score >= req) & (score <= max_score)
    out = val.loc[mask, ["dt", "asset", "timeframe", "label_up"]].copy().reset_index(drop=True)
    out["pred_up"] = pred_up[mask].astype(bool)
    out["score"] = score[mask]
    out["required_score"] = req[mask]
    out["won"] = out["pred_up"].to_numpy() == out["label_up"].astype(bool).to_numpy()
    daily_cap = int(params.get("daily_cap", 0) or 0)
    if daily_cap > 0 and len(out):
        out["day"] = pd.to_datetime(out["dt"], utc=True).dt.floor("D")
        out = out.sort_values(["day", "score"], ascending=[True, False]).groupby("day", group_keys=False).head(daily_cap)
        out = out.drop(columns=["day"]).sort_values("dt").reset_index(drop=True)
    return out


def get_frame(frame_kind: str) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    if frame_kind == "cross":
        return G_CROSS_FRAME if G_CROSS_FRAME is not None else build_cross_frame()
    return G_BASE_FRAME if G_BASE_FRAME is not None else build_base_frame()


def evaluate_route_param(args: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    frame_kind, params = args
    name = f"{params['route']}_{params['engine']}_{params['train_window']}_{params['feature_mode']}_thr{params['threshold']}_{stable_hash(params)}"
    try:
        df, features, truth = get_frame(frame_kind)
        feats = feature_subset(features, params["feature_mode"])[: int(params.get("max_features", 140))]
        if len(feats) < 8:
            return {"name": name, "params": params, "frameKind": frame_kind, "error": "too_few_features"}
        rows = []
        selections: dict[str, list[dict[str, Any]]] = {}
        for window in ["180d", "365d"]:
            train, val, _start = split_train_val(df, window, params["train_window"])
            model = fit_model(params, train, feats)
            if model is None:
                return {"name": name, "params": params, "frameKind": frame_kind, "error": f"fit_failed_{window}"}
            prob = predict(model, val, feats)
            if params.get("selection_mode") == "dynamic":
                selected = select_dynamic(val, train, prob, params)
            else:
                selected = select_standard(val, prob, params)
            if selected.empty:
                return {"name": name, "params": params, "frameKind": frame_kind, "error": f"empty_{window}"}
            selected["asset"] = "BTC"
            row = curve_metrics(selected, name, window, PRIMARY_BUY_PRICE)
            row.update({"asset": "BTC", "config": params["route"], "engine": params["engine"], "trainWindow": params["train_window"], "featureMode": params["feature_mode"], "threshold": params["threshold"], "featureCount": len(feats)})
            rows.append(row)
            # Keep selected rows for top candidates only; caller may discard if not needed.
            selections[window] = selected[["dt", "asset", "timeframe", "label_up", "pred_up", "score", "won"]].to_dict("records")
        return {"name": name, "params": params, "frameKind": frame_kind, "rows": rows, "featureCount": len(feats), "dataTruth": truth, "selections": selections}
    except Exception as exc:
        return {"name": name, "params": params, "frameKind": frame_kind, "error": repr(exc)[:1200]}


def candidate_score(candidate: dict[str, Any]) -> float:
    if "rows" not in candidate:
        return -1e18
    by = {r["window"]: r for r in candidate["rows"]}
    # Hard evidence floor: tiny-sample candidates can be useful diagnostics, but must not lead the board.
    if int(by.get("180d", {}).get("trades", 0)) < 100 or int(by.get("365d", {}).get("trades", 0)) < 200:
        return -1e12 + int(by.get("180d", {}).get("trades", 0)) + int(by.get("365d", {}).get("trades", 0))
    score = 0.0
    for w, b in BASELINE_ETH061.items():
        r = by.get(w, {})
        pnl_ratio = float(r.get("compoundPnl", -1e9)) / max(b["compoundPnl"], 1.0)
        dd_ratio = float(r.get("maxDrawdownUsd", 1e9)) / max(b["maxDrawdownUsd"], 1.0)
        score += pnl_ratio * 700.0
        score += (float(r.get("winRatePct", 0.0)) - b["winRatePct"]) * 25.0
        score += float(r.get("returnDrawdownRatio", 0.0)) * 20.0
        score -= max(0.0, dd_ratio - 1.0) * 500.0
    return round(score, 6)


def classify(candidate: dict[str, Any]) -> tuple[str, list[str]]:
    if "rows" not in candidate:
        return "invalid", [candidate.get("error", "invalid")]
    by = {r["window"]: r for r in candidate["rows"]}
    reasons: list[str] = []
    strict = True
    near = True
    observe = True
    for w, b in BASELINE_ETH061.items():
        r = by[w]
        min_trades = 100 if w == "180d" else 200
        if r["trades"] < min_trades:
            reasons.append(f"{w}_trades_too_low")
            strict = near = observe = False
        if r["compoundPnl"] <= b["compoundPnl"]:
            strict = False
        if w == "365d" and r["winRatePct"] < b["winRatePct"]:
            strict = False
        if r["maxDrawdownUsd"] > b["maxDrawdownUsd"] * 1.05:
            strict = False
        pnl_floor = 0.70 if w == "180d" else 0.75
        if r["compoundPnl"] < b["compoundPnl"] * pnl_floor:
            near = False
        if w == "365d" and r["winRatePct"] < b["winRatePct"] - 0.40:
            near = False
        if r["maxDrawdownUsd"] > b["maxDrawdownUsd"]:
            near = False
        if r["returnDrawdownRatio"] < b["returnDrawdownRatio"]:
            near = False
        if r["compoundPnl"] < b["compoundPnl"] * 0.45 or r["maxDrawdownUsd"] > b["maxDrawdownUsd"] * 0.75:
            observe = False
    if strict:
        return "strict", []
    if near:
        return "near", []
    if observe:
        return "observe", []
    if not reasons:
        reasons.append("not_enough_vs_eth061")
    return "fail", reasons


def route_param_grid() -> list[tuple[str, dict[str, Any]]]:
    params: list[tuple[str, dict[str, Any]]] = []
    model_grid = [
        {"engine": "lightgbm", "n_estimators": 100, "learning_rate": 0.035, "num_leaves": 20, "min_child_samples": 80, "reg_lambda": 1.0, "subsample": 0.88, "colsample_bytree": 0.88},
        {"engine": "lightgbm", "n_estimators": 160, "learning_rate": 0.026, "num_leaves": 28, "min_child_samples": 100, "reg_lambda": 1.2, "subsample": 0.88, "colsample_bytree": 0.88},
        {"engine": "xgboost", "n_estimators": 120, "learning_rate": 0.035, "depth": 3, "reg_lambda": 1.0, "subsample": 0.88, "colsample_bytree": 0.88},
        {"engine": "logistic", "C": 0.5},
        {"engine": "logistic", "C": 1.2},
    ]
    # A/D: expanded BTC061-style and low drawdown variants.
    for train_window in ["1y", "3y", "5y", "full"]:
        for feature_mode in ["core_htf", "trend_multi", "wide"]:
            for threshold in [0.54, 0.55, 0.56, 0.57, 0.58, 0.60]:
                for score_band, max_score in [("all", 1.0), ("mid_only", 0.76), ("mid_only", 0.82), ("high_only", 1.0)]:
                    for daily_cap in [0, 24, 32, 48]:
                        for max_features in [80, 140, 220]:
                            for mg in model_grid:
                                route = "BTC061_full" if daily_cap == 0 and threshold <= 0.57 else "BTC_low_drawdown"
                                p = {"route": route, "selection_mode": "standard", "train_window": train_window, "feature_mode": feature_mode, "threshold": threshold, "score_band": score_band, "max_score": max_score, "high_min": max(0.58, threshold), "daily_cap": daily_cap, "vol_q": 0.999, "max_features": max_features, **mg}
                                params.append(("base", p))
    # B: volatility/trend dynamic threshold specialists.
    for train_window in ["1y", "3y", "5y", "full"]:
        for feature_mode in ["trend_multi", "wide"]:
            for base_thr in [0.545, 0.55, 0.56]:
                for low, mid, high in [(0.54, 0.55, 0.57), (0.55, 0.56, 0.58), (0.56, 0.57, 0.60), (0.55, 0.58, 0.56)]:
                    for same_adj, opp_adj in [(-0.005, 0.01), (0.0, 0.015), (-0.01, 0.02), (0.005, 0.0)]:
                        for mg in model_grid[:3]:
                            p = {"route": "BTC_regime_specialist", "selection_mode": "dynamic", "train_window": train_window, "feature_mode": feature_mode, "threshold": base_thr, "thr_low_vol": low, "thr_mid_vol": mid, "thr_high_vol": high, "same_trend_adj": same_adj, "opp_trend_adj": opp_adj, "directional": True, "up_adj": 0.0, "down_adj": 0.0, "daily_cap": 0, "score_band": "all", "max_score": 1.0, "max_features": 180, **mg}
                            params.append(("base", p))
    # C: relative strength frame.
    for train_window in ["1y", "3y", "5y", "full"]:
        for threshold in [0.54, 0.55, 0.56, 0.57, 0.58]:
            for daily_cap in [0, 24, 32]:
                for max_features in [100, 180, 260]:
                    for mg in model_grid[:4]:
                        p = {"route": "BTC_relative_strength", "selection_mode": "standard", "train_window": train_window, "feature_mode": "relative_strength", "threshold": threshold, "score_band": "all", "max_score": 1.0, "daily_cap": daily_cap, "vol_q": 0.999, "max_features": max_features, **mg}
                        params.append(("cross", p))
    if PARAM_LIMIT > 0 and len(params) > PARAM_LIMIT:
        step = max(1, len(params) // PARAM_LIMIT)
        params = params[::step][:PARAM_LIMIT]
    return params


def audit() -> dict[str, Any]:
    eth_audit = btc061.audit_eth061_replay()
    df, features, truth = build_base_frame()
    cross_df, cross_features, cross_truth = build_cross_frame()
    forbidden = set(getattr(raw_growth.base, "FORBIDDEN_FEATURES", set()))
    forbidden_prefix = tuple(getattr(raw_growth.base, "FORBIDDEN_PREFIXES", ()))
    hits = [c for c in features + cross_features if c in forbidden or c.startswith(forbidden_prefix)]
    passed = bool(eth_audit["passed"]) and not hits and len(df) > 10000 and len(cross_df) > 10000
    out = {
        "generatedAt": now_iso(),
        "beijingTime": bj_now(),
        "researchOnlyNoLiveChange": True,
        "eth061Replay": eth_audit,
        "btcBaseTruth": truth,
        "btcCrossTruth": cross_truth,
        "forbiddenFeatureHits": hits,
        "timePolicy": "closed_candle_available_at_v2: shifted current-bar features and backward-asof shifted HTF/cross-asset features",
        "passed": passed,
    }
    write_json(OUT_AUDIT_JSON, out)
    write_text(OUT_AUDIT_MD, "\n".join([
        "# BTC near-ETH061 多路线搜索：审计",
        "",
        f"- 北京时间：`{out['beijingTime']}`",
        "- 研究动作：`research_only_no_live_change`",
        "- 公平口径：`850U初始 / 每笔1% / 买价0.55 / 满成交`",
        f"- ETH061复放通过：`{eth_audit['passed']}`",
        f"- BTC基础数据：`{truth.get('start')} -> {truth.get('end')}`，行数 `{truth.get('finalRows')}`，特征 `{truth.get('featureCount')}`",
        f"- 跨币特征数：`{cross_truth.get('crossFinalFeatureCount')}`",
        f"- 禁用字段命中：`{hits}`",
        f"- 审计通过：`{passed}`",
        "",
    ]) + "\n")
    return out


def slim_candidate(c: dict[str, Any], keep_selections: bool = False) -> dict[str, Any]:
    keys = ["name", "params", "frameKind", "rows", "featureCount", "class", "classReasons", "score", "archiveRows", "pressureRows", "error"]
    out = {k: c.get(k) for k in keys if k in c}
    if keep_selections and "selections" in c:
        out["selections"] = c["selections"]
    return out


def evaluate_filter_candidate_local(args: tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]]]) -> dict[str, Any]:
    main, fp, selections_raw = args
    name = f"BTC061style_{main['params']['train_window']}_{main['params']['feature_mode']}_{main['params']['threshold']}_{fp['candidate_id']}"
    rows = []
    block_meta = {}
    selections: dict[str, list[dict[str, Any]]] = {}
    try:
        for window in ["180d", "365d"]:
            base_rows = pd.DataFrame(selections_raw[window])
            filtered, meta = btc061.apply_filter(base_rows, fp)
            if filtered.empty:
                return {"name": name, "params": {"main": main["params"], "filter": fp}, "error": f"empty_{window}"}
            row = curve_metrics(filtered, name, window, PRIMARY_BUY_PRICE)
            row.update({"asset": "BTC", "config": "BTC061-style", "trainWindow": main["params"]["train_window"], "threshold": main["params"]["threshold"], **meta})
            rows.append(row)
            block_meta[window] = meta
            keep_cols = [c for c in ["dt", "asset", "timeframe", "label_up", "pred_up", "score", "won"] if c in filtered.columns]
            selections[window] = filtered[keep_cols].to_dict("records")
        return {"name": name, "params": {"main": main["params"], "filter": fp}, "rows": rows, "blockMeta": block_meta, "mainName": main["name"], "selections": selections}
    except Exception as exc:
        return {"name": name, "params": {"main": main.get("params"), "filter": fp}, "error": repr(exc)[:1000]}


def pressure_rows(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    if "selections" not in candidate:
        return []
    rows = []
    for window, records in candidate["selections"].items():
        df = pd.DataFrame(records)
        if not df.empty:
            r = curve_metrics(df, candidate["name"], window, PRESSURE_BUY_PRICE)
            r.update({"asset": "BTC", "config": candidate.get("params", {}).get("route"), "pressure": "buy_0.60"})
            rows.append(r)
    return rows


def archived_rows(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    # Current generic archive scoring supports the candidate's main route and selection mode.
    try:
        old = btc061.archived_btc_metrics(candidate)
        if old:
            return old
    except Exception:
        pass
    return []


def render(payload: dict[str, Any]) -> None:
    write_json(OUT_CHECKPOINT, payload)
    write_json(OUT_LEADERBOARD, payload)
    write_json(OUT_COMPARE_JSON, payload)
    write_json(OUT_VERDICT_JSON, payload.get("verdict", {}) | {"selected": payload.get("selected")})
    selected = payload.get("selected") or {}
    verdict = payload.get("verdict") or {}
    lines = [
        "# BTC near-ETH061 多路线搜索结果",
        "",
        f"- 北京时间：`{payload.get('beijingTime')}`",
        "- live动作：`无，研究只读；ETH061真钱未改`",
        "- 主口径：`850U初始 / 每笔1% / 买价0.55 / 满成交`",
        "",
        "## ETH061 vs BTC 最强",
        "",
        "|配置|窗口|交易数|胜/负|胜率|盈亏|期末资金|最大回撤|收益回撤比|类别|哈希|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for w, b in BASELINE_ETH061.items():
        lines.append(f"|ETH 当前061|{w}|{b['trades']}|{b['wins']}/{b['losses']}|{b['winRatePct']:.2f}%|{b['compoundPnl']:,.2f}|{b['endingBankroll']:,.2f}|{b['maxDrawdownUsd']:,.2f}|{b['returnDrawdownRatio']:.2f}|基线|`{b['setHash']}`|")
    if selected.get("rows"):
        for r in selected["rows"]:
            lines.append(f"|{r.get('config','BTC候选')}|{r['window']}|{r['trades']}|{r['wins']}/{r['losses']}|{r['winRatePct']:.2f}%|{r['compoundPnl']:,.2f}|{r['endingBankroll']:,.2f}|{r['maxDrawdownUsd']:,.2f}|{r['returnDrawdownRatio']:.2f}|{selected.get('class')}|`{r['setHash']}`|")
    else:
        lines.append("|BTC候选|-|-|-|-|-|-|-|-|无|无|")
    lines += ["", "## BTC 0.60买价压力", "", "|窗口|交易数|胜率|盈亏|最大回撤|", "|---|---:|---:|---:|---:|"]
    for r in selected.get("pressureRows", []) or []:
        lines.append(f"|{r['window']}|{r['trades']}|{r['winRatePct']:.2f}%|{r['compoundPnl']:,.2f}|{r['maxDrawdownUsd']:,.2f}|")
    if not selected.get("pressureRows"):
        lines.append("|无|-|-|-|-|")
    lines += ["", "## BTC 历史归档纯预测", "", "|范围|旧真实市场数|选中|胜/负|胜率|盈亏|最大回撤|", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in selected.get("archiveRows", []) or []:
        lines.append(f"|{r.get('scope','archive')}|{r.get('oldRealMarkets',0)}|{r.get('selectedTrades',r.get('trades',0))}|{r.get('wins',0)}/{r.get('losses',0)}|{r.get('winRatePct',0):.2f}%|{r.get('compoundPnl',0):,.2f}|{r.get('maxDrawdownUsd',0):,.2f}|")
    if not selected.get("archiveRows"):
        lines.append("|无可复核|0|0|0/0|0.00%|0.00|0.00|")
    lines += ["", "## 前20候选", "", "|排名|路线|配置|365胜率|365盈亏|365回撤|类别|原因|", "|---:|---|---|---:|---:|---:|---|---|"]
    for i, c in enumerate(payload.get("topCandidates", [])[:20], 1):
        by = {r["window"]: r for r in c.get("rows", [])}
        r365 = by.get("365d", {})
        lines.append(f"|{i}|{c.get('params',{}).get('route') or c.get('params',{}).get('main',{}).get('route','BTC061-style')}|{c.get('name','')}|{float(r365.get('winRatePct',0)):.2f}%|{float(r365.get('compoundPnl',0)):,.2f}|{float(r365.get('maxDrawdownUsd',0)):,.2f}|{c.get('class')}|{','.join(c.get('classReasons',[]))}|")
    lines += ["", "## 唯一结论", "", f"- 状态：`{verdict.get('status')}`", f"- 结论：{verdict.get('message')}"]
    write_text(OUT_COMPARE_MD, "\n".join(lines) + "\n")
    write_text(OUT_VERDICT_MD, "\n".join(["# BTC near-ETH061：唯一结论", "", f"- 状态：`{verdict.get('status')}`", f"- 结论：{verdict.get('message')}", ""]) + "\n")


def main() -> int:
    started = time.time()
    REPORTS.mkdir(parents=True, exist_ok=True)
    if OUT_RESULTS.exists() and os.environ.get("BTC_NEAR061_RESET", "1") == "1":
        OUT_RESULTS.unlink()
    audit_payload = audit()
    if not audit_payload["passed"]:
        payload = {"generatedAt": now_iso(), "beijingTime": bj_now(), "audit": audit_payload, "selected": None, "topCandidates": [], "verdict": {"status": "audit_failed", "message": "审计失败；未进入BTC搜索。"}}
        render(payload)
        return 2

    grid = route_param_grid()
    deadline = started + MAX_SECONDS
    results: list[dict[str, Any]] = []
    with cf.ProcessPoolExecutor(max_workers=WORKERS, initializer=init_worker) as ex:
        futs = [ex.submit(evaluate_route_param, item) for item in grid]
        for i, fut in enumerate(cf.as_completed(futs), 1):
            r = fut.result()
            if "rows" in r:
                cls, reasons = classify(r)
                r["class"] = cls
                r["classReasons"] = reasons
                r["score"] = candidate_score(r)
            results.append(r)
            if i % 250 == 0:
                with OUT_RESULTS.open("a", encoding="utf-8") as fh:
                    for x in results[-250:]:
                        fh.write(json.dumps(slim_candidate(x), ensure_ascii=False, sort_keys=True, default=str) + "\n")
                partial = sorted([x for x in results if "rows" in x], key=lambda x: (x.get("class") in {"strict", "near", "observe"}, x.get("score", -1e18)), reverse=True)[:50]
                write_json(OUT_CHECKPOINT, {"generatedAt": now_iso(), "beijingTime": bj_now(), "finished": False, "done": i, "total": len(grid), "topCandidates": [slim_candidate(x) for x in partial]})
            if time.time() > deadline:
                break

    valid = [r for r in results if "rows" in r]
    valid.sort(key=lambda x: (x.get("class") == "strict", x.get("class") == "near", x.get("class") == "observe", x.get("score", -1e18)), reverse=True)

    # Add BTC061-style filter search on top route candidates, reusing audited old filter machinery.
    filter_results: list[dict[str, Any]] = []
    base_candidates = [c for c in valid if c.get("frameKind") == "base" and c.get("params", {}).get("route") in {"BTC061_full", "BTC_low_drawdown"}][:TOP_MAIN_FOR_FILTER]
    try:
        df_base, all_features, _ = build_base_frame()
        for main_cand in base_candidates:
            if time.time() > deadline:
                break
            fps = btc061.build_filter_candidates_for_main(df_base, all_features, main_cand)[:MAX_FILTERS_PER_MAIN]
            args = [(main_cand, fp, main_cand["selections"]) for fp in fps]
            with cf.ProcessPoolExecutor(max_workers=WORKERS) as ex:
                futs = [ex.submit(evaluate_filter_candidate_local, a) for a in args]
                for fut in cf.as_completed(futs):
                    rr = fut.result()
                    if "rows" in rr:
                        cls, reasons = classify(rr)
                        rr["class"] = cls
                        rr["classReasons"] = reasons
                        rr["score"] = candidate_score(rr)
                    filter_results.append(rr)
                    if len(filter_results) % 1000 == 0:
                        with OUT_RESULTS.open("a", encoding="utf-8") as fh:
                            for x in filter_results[-1000:]:
                                fh.write(json.dumps(slim_candidate(x), ensure_ascii=False, sort_keys=True, default=str) + "\n")
                    if time.time() > deadline:
                        break
    except Exception as exc:
        filter_results.append({"name": "filter_stage_error", "error": repr(exc)[:1200]})

    all_candidates = [r for r in valid + filter_results if "rows" in r]
    all_candidates.sort(key=lambda x: (x.get("class") == "strict", x.get("class") == "near", x.get("class") == "observe", x.get("score", -1e18)), reverse=True)
    selected = all_candidates[0] if all_candidates else None
    if selected:
        selected = dict(selected)
        selected["pressureRows"] = pressure_rows(selected)
        selected["archiveRows"] = archived_rows(selected)

    if selected and selected.get("class") == "strict":
        verdict = {"status": "strict_btc_candidate_found", "message": "找到严格打败ETH061的BTC候选；只能进入影子验证，不直接真钱。"}
    elif selected and selected.get("class") == "near":
        verdict = {"status": "near_btc_candidate_found", "message": "找到接近ETH061的BTC候选；优先做影子验证，不直接真钱。"}
    elif selected and selected.get("class") == "observe":
        verdict = {"status": "observation_btc_candidate_only", "message": "仅找到观察级BTC候选；回撤或收益回撤有参考，但不建议上线真钱。"}
    else:
        verdict = {"status": "no_usable_btc_candidate", "message": "五路线BTC搜索未找到可接近ETH061的候选；当前真钱继续保留ETH061。"}

    payload = {
        "generatedAt": now_iso(),
        "beijingTime": bj_now(),
        "finished": True,
        "elapsedSeconds": round(time.time() - started, 3),
        "workers": WORKERS,
        "paramCount": len(grid),
        "evaluatedCount": len(results),
        "filterEvaluatedCount": len(filter_results),
        "audit": audit_payload,
        "baselineETH061": BASELINE_ETH061,
        "topCandidates": [slim_candidate(c) for c in all_candidates[:100]],
        "selected": slim_candidate(selected or {}, keep_selections=False),
        "verdict": verdict,
        "liveConfigMutated": False,
    }
    render(payload)
    print(OUT_COMPARE_MD)
    print(OUT_VERDICT_MD)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
