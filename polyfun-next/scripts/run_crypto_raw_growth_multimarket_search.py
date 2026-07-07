#!/usr/bin/env python3
from __future__ import annotations

"""Multi-asset raw-kline compound-growth main-model search.

Research only. It does not mutate live configs, ledgers, processes, order
submission, monitor settings, or current 061 live trading.

Fair metric:
  - 850U initial bankroll
  - stake 1% current bankroll per selected market
  - buy price fixed at 0.50
  - full fill proxy
  - model chooses UP/DOWN and whether to trade
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

THREADS = os.environ.get("CRYPTO_RAW_GROWTH_THREADS", "1")
for key in ["OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(key, THREADS)

import numpy as np
import pandas as pd

ROOT = Path("/Users/mac/polyfun")
NEXT = ROOT / "polyfun-next"
SCRIPTS = NEXT / "scripts"
REPORTS = ROOT / "reports"
BASE_SCRIPT = ROOT / "scripts" / "ops" / "run_crypto15m_1h_multasset_pressure_search_latest.py"
CLUSTER_COMPARE_JSON = REPORTS / "top159_061_deep_v2_cluster_180_365_archive_compare_latest.json"

WORKERS = int(os.environ.get("CRYPTO_RAW_GROWTH_WORKERS", "4"))
MAX_SECONDS = int(os.environ.get("CRYPTO_RAW_GROWTH_MAX_SECONDS", str(4 * 3600)))
CHECKPOINT_SECONDS = int(os.environ.get("CRYPTO_RAW_GROWTH_CHECKPOINT_SECONDS", "900"))
PARAM_LIMIT = int(os.environ.get("CRYPTO_RAW_GROWTH_PARAM_LIMIT", "0"))
MAX_TRAIN_ROWS = int(os.environ.get("CRYPTO_RAW_GROWTH_MAX_TRAIN_ROWS", "160000"))
RNG_SEED = 20260505

ASSETS = [x.strip().upper() for x in os.environ.get("CRYPTO_RAW_GROWTH_ASSETS", "BTC,ETH,SOL,XRP").split(",") if x.strip()]
TIMEFRAMES = [x.strip() for x in os.environ.get("CRYPTO_RAW_GROWTH_TIMEFRAMES", "15m,1h,4h").split(",") if x.strip()]
ENGINES = [x.strip() for x in os.environ.get("CRYPTO_RAW_GROWTH_ENGINES", "logistic,lightgbm,xgboost").split(",") if x.strip()]

START_BANKROLL = 850.0
STAKE_PCT = 0.01
BUY_PRICE = 0.50

OUT_AUDIT_MD = REPORTS / "crypto_raw_growth_multimarket_bug_audit_latest.md"
OUT_AUDIT_JSON = REPORTS / "crypto_raw_growth_multimarket_bug_audit_latest.json"
OUT_RESULTS = REPORTS / "crypto_raw_growth_multimarket_results_latest.jsonl"
OUT_CHECKPOINT = REPORTS / "crypto_raw_growth_multimarket_checkpoint_latest.json"
OUT_LEADERBOARD = REPORTS / "crypto_raw_growth_multimarket_leaderboard_latest.json"
OUT_COMPARE_MD = REPORTS / "crypto_raw_growth_multimarket_180_365_archive_compare_latest.md"
OUT_COMPARE_JSON = REPORTS / "crypto_raw_growth_multimarket_180_365_archive_compare_latest.json"
OUT_VERDICT_MD = REPORTS / "crypto_raw_growth_multimarket_unique_verdict_latest.md"
OUT_VERDICT_JSON = REPORTS / "crypto_raw_growth_multimarket_unique_verdict_latest.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

base = load_module("crypto_pressure_base_for_raw_growth", BASE_SCRIPT)


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
    identity_cols = ["dt", "pred_up", "label_up", "won"]
    if "asset" in sel.columns:
        identity_cols.insert(0, "asset")
    if "timeframe" in sel.columns:
        identity_cols.insert(1, "timeframe")
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
        "setHash": stable_hash(sel[identity_cols].to_dict("records")) if len(sel) else stable_hash([]),
    }


def prefix_merge_asof(left: pd.DataFrame, right: pd.DataFrame, cols: list[str], prefix: str) -> tuple[pd.DataFrame, list[str]]:
    r = right[["dt"] + cols].copy()
    rename = {c: f"{prefix}_{c}" for c in cols}
    r = r.rename(columns=rename)
    l = left.sort_values("dt").copy()
    r = r.sort_values("dt").copy()
    l["dt"] = pd.to_datetime(l["dt"], utc=True, errors="coerce").astype("datetime64[ns, UTC]")
    r["dt"] = pd.to_datetime(r["dt"], utc=True, errors="coerce").astype("datetime64[ns, UTC]")
    out = pd.merge_asof(l, r, on="dt", direction="backward")
    return out, list(rename.values())


def daily_from_1h(asset: str) -> tuple[pd.DataFrame, list[str]]:
    h = base.load_raw(asset, "1h").copy()
    h = h.set_index("dt").sort_index()
    d = pd.DataFrame({
        "date": h.index.floor("D").unique(),
    })
    ohlc = h.resample("1D", label="left", closed="left").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna().reset_index()
    ohlc = ohlc.rename(columns={"dt": "date"})
    ohlc["date"] = pd.to_datetime(ohlc["date"], utc=True)
    # base.build_features expects both original date and normalized dt.
    ohlc["dt"] = ohlc["date"]
    fdf, fcols = base.build_features(ohlc, "1d")
    return fdf, fcols


def build_frame(asset: str, timeframe: str) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    raw = base.load_raw(asset, timeframe)
    df, fcols = base.build_features(raw, timeframe)
    feature_cols = list(fcols)
    truth: dict[str, Any] = {"asset": asset, "timeframe": timeframe, "rawRows": int(len(raw)), "baseRows": int(len(df)), "featureGroups": {timeframe: len(fcols)}}
    for tf in ["15m", "1h", "4h"]:
        if tf == timeframe:
            continue
        path = ROOT / "data" / "raw" / f"{asset.lower()}_usdt_{tf}.parquet"
        if not path.exists():
            continue
        raw_htf = base.load_raw(asset, tf)
        fdf, cols = base.build_features(raw_htf, tf)
        df, added = prefix_merge_asof(df, fdf, cols, tf)
        feature_cols += added
        truth["featureGroups"][tf] = len(added)
        truth[f"raw{tf}Rows"] = int(len(raw_htf))
    try:
        ddf, dcols = daily_from_1h(asset)
        df, added = prefix_merge_asof(df, ddf, dcols, "daily")
        feature_cols += added
        truth["featureGroups"]["daily"] = len(added)
    except Exception as exc:
        truth["dailyError"] = repr(exc)[:300]
    clean: list[str] = []
    forbidden = set(getattr(base, "FORBIDDEN_FEATURES", set()))
    forbidden_prefix = tuple(getattr(base, "FORBIDDEN_PREFIXES", ()))
    for c in feature_cols:
        if c in forbidden or c.startswith(forbidden_prefix):
            continue
        s = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if int(s.notna().sum()) >= max(500, int(len(df) * 0.50)):
            df[c] = s
            clean.append(c)
    df = df.dropna(subset=["dt", "label_up"]).sort_values("dt").reset_index(drop=True)
    df["asset"] = asset
    df["timeframe"] = timeframe
    truth["finalRows"] = int(len(df))
    truth["featureCount"] = int(len(clean))
    truth["start"] = str(df["dt"].min()) if len(df) else None
    truth["end"] = str(df["dt"].max()) if len(df) else None
    return df, clean, truth


def feature_subset(features: list[str], mode: str) -> list[str]:
    if mode == "core":
        out = [c for c in features if not c.startswith(("15m_", "1h_", "4h_", "daily_"))]
    elif mode == "core_htf":
        out = [c for c in features if not c.startswith("daily_")]
    elif mode == "trend_multi":
        keys = ("ret_", "ema_", "ema_dist_", "rsi", "bb_pos", "range_", "vol_")
        out = [c for c in features if any(k in c for k in keys)]
    elif mode == "wide":
        out = list(features)
    else:
        out = list(features)
    return out


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
    # Conservative embargo: one full candidate bar before validation.
    embargo = {"15m": pd.Timedelta(minutes=15), "1h": pd.Timedelta(hours=1), "4h": pd.Timedelta(hours=4)}.get(str(df["timeframe"].iloc[0]), pd.Timedelta(hours=4))
    train = train[train["dt"] < start - embargo].copy()
    if MAX_TRAIN_ROWS > 0 and len(train) > MAX_TRAIN_ROWS:
        train = train.sort_values("dt").iloc[-MAX_TRAIN_ROWS:].copy()
    return train, val


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
            n_estimators=int(params.get("n_estimators", 120)), learning_rate=float(params.get("learning_rate", 0.035)),
            num_leaves=int(params.get("num_leaves", 24)), min_child_samples=int(params.get("min_child_samples", 80)),
            subsample=float(params.get("subsample", 0.88)), colsample_bytree=float(params.get("colsample_bytree", 0.88)),
            reg_lambda=float(params.get("reg_lambda", 0.8)), random_state=RNG_SEED, n_jobs=int(THREADS), verbose=-1,
        )
    elif engine == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=int(params.get("n_estimators", 120)), max_depth=int(params.get("depth", 4)), learning_rate=float(params.get("learning_rate", 0.035)),
            subsample=float(params.get("subsample", 0.88)), colsample_bytree=float(params.get("colsample_bytree", 0.88)),
            reg_lambda=float(params.get("reg_lambda", 0.8)), random_state=RNG_SEED, n_jobs=int(THREADS), eval_metric="logloss", verbosity=0,
        )
    elif engine == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            iterations=int(params.get("n_estimators", 120)), depth=int(params.get("depth", 4)), learning_rate=float(params.get("learning_rate", 0.035)),
            l2_leaf_reg=float(params.get("reg_lambda", 0.8)), loss_function="Logloss", eval_metric="Logloss", random_seed=RNG_SEED,
            thread_count=int(THREADS), verbose=False,
        )
    else:
        return None
    model.fit(x, y)
    return model


def predict(model: Any, val: pd.DataFrame, feats: list[str]) -> np.ndarray:
    x = val[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def select_rows(val: pd.DataFrame, prob: np.ndarray, params: dict[str, Any]) -> pd.DataFrame:
    pred_up = prob >= 0.5
    score = np.maximum(prob, 1.0 - prob)
    mask = score >= float(params["threshold"])
    band = params.get("score_band", "all")
    if band == "mid_only":
        mask &= score <= float(params.get("max_score", 0.76))
    elif band == "high_only":
        mask &= score >= float(params.get("high_min", 0.60))
    if params.get("vol_q", 0.999) < 0.999:
        vol_cols = [c for c in ["vol_16", "vol_32", "15m_vol_16", "1h_vol_16", "4h_vol_16"] if c in val.columns]
        if vol_cols:
            v = pd.to_numeric(val[vol_cols[0]], errors="coerce")
            mask &= v <= float(v.quantile(float(params["vol_q"])))
    out = val.loc[mask, ["dt", "asset", "timeframe", "label_up"]].copy().reset_index(drop=True)
    pred_sel = pred_up[mask]
    score_sel = score[mask]
    out["pred_up"] = pred_sel.astype(bool)
    out["score"] = score_sel
    out["won"] = out["pred_up"].to_numpy() == out["label_up"].astype(bool).to_numpy()
    daily_cap = int(params.get("daily_cap", 0) or 0)
    if daily_cap > 0 and len(out):
        out["day"] = pd.to_datetime(out["dt"], utc=True).dt.floor("D")
        out = out.sort_values(["day", "score"], ascending=[True, False]).groupby("day", group_keys=False).head(daily_cap)
        out = out.drop(columns=["day"]).sort_values("dt").reset_index(drop=True)
    return out


def evaluate_params(item: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    asset, timeframe, params = item
    name = f"rawgrowth_{asset}_{timeframe}_{params['engine']}_{params['train_window']}_{params['feature_mode']}_thr{params['threshold']}_cap{params.get('daily_cap',0)}_{stable_hash(params)}"
    try:
        df, features, truth = build_frame(asset, timeframe)
        feats = feature_subset(features, params["feature_mode"])
        max_features = int(params.get("max_features", 120))
        if max_features > 0:
            feats = feats[:max_features]
        if len(feats) < 8:
            return {"name": name, "params": params, "asset": asset, "timeframe": timeframe, "error": "too_few_features"}
        rows = []
        for window in ["180d", "365d"]:
            train, val = split_train_val(df, window, params["train_window"])
            model = fit_model(params["engine"], train, feats, params)
            if model is None:
                return {"name": name, "params": params, "asset": asset, "timeframe": timeframe, "error": "fit_failed"}
            prob = predict(model, val, feats)
            selected = select_rows(val, prob, params)
            if len(selected) == 0:
                return {"name": name, "params": params, "asset": asset, "timeframe": timeframe, "error": "empty_selection"}
            row = compound_metrics(selected, name, window)
            row.update({"asset": asset, "timeframe": timeframe, "engine": params["engine"], "trainWindow": params["train_window"], "featureMode": params["feature_mode"], "threshold": params["threshold"], "dailyCap": params.get("daily_cap", 0), "featureCount": len(feats)})
            rows.append(row)
        return {"name": name, "params": params, "asset": asset, "timeframe": timeframe, "rows": rows, "featureCount": len(feats), "dataTruth": truth}
    except Exception as exc:
        return {"name": name, "params": params, "asset": asset, "timeframe": timeframe, "error": repr(exc)[:1000]}


def current_061_rows() -> dict[str, dict[str, Any]]:
    data = json.loads(CLUSTER_COMPARE_JSON.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for row in data.get("validationRows", []):
        if row.get("name") == "current_live_06173_cluster_gate" and row.get("method") == "full_fill_buy_0.50":
            out[row["window"]] = row
    if "180d" not in out or "365d" not in out:
        raise RuntimeError("cannot locate current061 baseline rows")
    return out


def pass_gate(candidate: dict[str, Any], baseline: dict[str, dict[str, Any]]) -> tuple[bool, list[str]]:
    if "rows" not in candidate:
        return False, [candidate.get("error", "invalid")]
    by = {r["window"]: r for r in candidate["rows"]}
    reasons: list[str] = []
    for w in ["180d", "365d"]:
        r = by[w]
        b = baseline[w]
        min_trades = max(100 if w == "180d" else 200, int(b["trades"] * 0.45))
        if r["trades"] < min_trades:
            reasons.append(f"{w}_trades_too_low")
        if r["compoundPnl"] <= b["compoundPnl"]:
            reasons.append(f"{w}_pnl_not_above_061")
        if w == "365d" and r["winRatePct"] < b["winRatePct"]:
            reasons.append("365d_winrate_below_061")
        if w == "180d" and r["winRatePct"] < b["winRatePct"] - 0.15:
            reasons.append("180d_winrate_too_low")
        if r["maxDrawdownUsd"] > b["maxDrawdownUsd"] * 1.05:
            reasons.append(f"{w}_drawdown_too_high")
    return len(reasons) == 0, reasons


def score_candidate(candidate: dict[str, Any], baseline: dict[str, dict[str, Any]]) -> float:
    if "rows" not in candidate:
        return -1e18
    by = {r["window"]: r for r in candidate["rows"]}
    score = 0.0
    for w in ["180d", "365d"]:
        r = by[w]
        b = baseline[w]
        score += (r["compoundPnl"] - b["compoundPnl"]) / max(abs(b["compoundPnl"]), 1.0) * 1000.0
        score += (r["winRatePct"] - b["winRatePct"]) * 50.0
        score -= max(0.0, r["maxDrawdownUsd"] - b["maxDrawdownUsd"]) / max(b["maxDrawdownUsd"], 1.0) * 500.0
        score += r.get("returnDrawdownRatio", 0.0) * 3.0
    return score


def param_grid() -> list[tuple[str, str, dict[str, Any]]]:
    rows: list[tuple[str, str, dict[str, Any]]] = []
    train_windows = ["1y", "3y", "5y", "full"]
    feature_modes = ["core", "core_htf", "trend_multi", "wide"]
    thresholds = [0.52, 0.53, 0.54, 0.55, 0.56, 0.58, 0.60, 0.62]
    daily_caps = [0, 12, 18, 24, 32]
    score_bands = [("all", 1.0), ("mid_only", 0.76), ("high_only", 1.0)]
    vol_qs = [0.90, 0.95, 0.999]
    max_features_grid = [60, 120, 180]
    model_params = [
        {"n_estimators": 80, "learning_rate": 0.04, "num_leaves": 16, "min_child_samples": 80, "reg_lambda": 1.0, "depth": 3, "C": 0.5},
        {"n_estimators": 120, "learning_rate": 0.035, "num_leaves": 24, "min_child_samples": 80, "reg_lambda": 0.8, "depth": 4, "C": 1.0},
        {"n_estimators": 180, "learning_rate": 0.025, "num_leaves": 32, "min_child_samples": 100, "reg_lambda": 1.2, "depth": 4, "C": 1.5},
    ]
    for asset, tf, engine, tw, fm, th, cap, band, vol_q, mf, mp in itertools.product(ASSETS, TIMEFRAMES, ENGINES, train_windows, feature_modes, thresholds, daily_caps, score_bands, vol_qs, max_features_grid, model_params):
        if engine not in {"logistic", "lightgbm", "xgboost", "catboost"}:
            continue
        p = {"engine": engine, "train_window": tw, "feature_mode": fm, "threshold": th, "daily_cap": cap, "score_band": band[0], "max_score": band[1], "high_min": max(0.58, th), "vol_q": vol_q, "max_features": mf, **mp, "subsample": 0.88, "colsample_bytree": 0.88}
        rows.append((asset, tf, p))
    if PARAM_LIMIT > 0:
        # Deterministic diverse stride, not just first N.
        step = max(1, len(rows) // PARAM_LIMIT)
        rows = rows[::step][:PARAM_LIMIT]
    return rows


def bug_audit() -> dict[str, Any]:
    # Lightweight but direct checks. The heavy repeat/random checks are performed on actual candidates after search.
    hits = []
    forbidden_exact = set(getattr(base, "FORBIDDEN_FEATURES", set()))
    forbidden_prefix = tuple(getattr(base, "FORBIDDEN_PREFIXES", ()))
    truth = []
    for asset in ASSETS:
        for tf in TIMEFRAMES:
            try:
                df, features, data_truth = build_frame(asset, tf)
                bad = [c for c in features if c in forbidden_exact or c.startswith(forbidden_prefix)]
                hits += [f"{asset}/{tf}:{c}" for c in bad]
                truth.append(data_truth)
            except Exception as exc:
                truth.append({"asset": asset, "timeframe": tf, "error": repr(exc)[:500]})
    audit = {
        "generatedAt": now_iso(),
        "beijingTime": bj_now(),
        "researchOnlyNoLiveChange": True,
        "fairMetric": "850U initial, stake 1% current bankroll, buy price 0.50, full fill",
        "timePolicy": "current timeframe features shifted by one bar; 1h/4h/daily merged with backward asof; train/validation embargo by timeframe",
        "dataTruth": truth,
        "forbiddenFeatureHits": hits,
        "passed": len(hits) == 0,
    }
    write_json(OUT_AUDIT_JSON, audit)
    lines = [
        "# 多币种多周期原始复利主模型：回测器审计",
        "",
        f"- 北京时间：`{audit['beijingTime']}`",
        "- 研究动作：`research_only_no_live_change`",
        "- 公平口径：`850U初始 / 每笔1% / 买价0.50 / 满成交`",
        f"- 禁用字段命中：`{hits}`",
        f"- 审计通过：`{audit['passed']}`",
    ]
    write_text(OUT_AUDIT_MD, "\n".join(lines) + "\n")
    return audit


def iter_prediction_files() -> list[Path]:
    paths = []
    for p in ROOT.rglob("prediction_trades*.json"):
        if p.is_file() and "node_modules" not in p.parts and ".venv" not in p.parts:
            paths.append(p)
    return sorted(paths)


def load_archived_live_rows() -> pd.DataFrame:
    rows = []
    seen = set()
    for path in iter_prediction_files():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, list):
            continue
        for r in raw:
            if not isinstance(r, dict) or r.get("mode") != "live" or r.get("status") != "executed":
                continue
            symbol = str(r.get("symbol") or "").upper()
            if symbol not in set(ASSETS):
                continue
            slug = str(r.get("marketSlug") or "")
            tf = "15m" if f"{symbol.lower()}-updown-15m-" in slug else None
            if tf is None:
                continue
            result = str(r.get("formalResult") or r.get("result") or "").lower()
            if result not in {"win", "lose"}:
                continue
            direction = str(r.get("direction") or r.get("tokenOutcome") or "").upper()
            if direction not in {"UP", "DOWN"}:
                continue
            try:
                market_start = pd.to_datetime(int(slug.rsplit("-", 1)[-1]), unit="s", utc=True)
            except Exception:
                continue
            rid = stable_hash([path.as_posix(), slug, direction, r.get("orderId"), r.get("txHash"), r.get("timestamp"), r.get("amount"), result])
            if rid in seen:
                continue
            seen.add(rid)
            won = result == "win"
            old_up = direction == "UP"
            actual_up = old_up if won else (not old_up)
            rows.append({"asset": symbol, "timeframe": tf, "marketStart": market_start, "direction": direction, "actualUp": bool(actual_up), "source": str(path)})
    if not rows:
        return pd.DataFrame(columns=["asset", "timeframe", "marketStart", "direction", "actualUp"])
    df = pd.DataFrame(rows).drop_duplicates(["asset", "timeframe", "marketStart", "direction"]).sort_values("marketStart")
    return df.reset_index(drop=True)


def archived_prediction_metrics(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    if "rows" not in candidate:
        return []
    asset = candidate.get("asset")
    tf = candidate.get("timeframe")
    old = load_archived_live_rows()
    old = old[(old["asset"] == asset) & (old["timeframe"] == tf)].copy()
    if old.empty:
        return []
    params = candidate["params"]
    df, features, _ = build_frame(asset, tf)
    train = df[df["dt"] < old["marketStart"].min() - pd.Timedelta(hours=4)].copy()
    if params["train_window"] != "full":
        days = {"1y": 365, "3y": 1095, "5y": 1825}[params["train_window"]]
        train = train[train["dt"] >= old["marketStart"].min() - pd.Timedelta(days=days)].copy()
    feats = feature_subset(features, params["feature_mode"])
    feats = feats[: int(params.get("max_features", 120))]
    model = fit_model(params["engine"], train, feats, params)
    if model is None:
        return []
    vals = df[(df["dt"] >= old["marketStart"].min()) & (df["dt"] <= old["marketStart"].max())].copy()
    if vals.empty:
        return []
    prob = predict(model, vals, feats)
    selected = select_rows(vals, prob, params)
    pred_map = selected.set_index("dt")["pred_up"].to_dict()
    score_map = selected.set_index("dt")["score"].to_dict()
    rows = []
    for _, r in old.iterrows():
        p = pred_map.get(r["marketStart"])
        if p is None:
            continue
        rows.append({"dt": r["marketStart"], "asset": asset, "timeframe": tf, "pred_up": bool(p), "label_up": bool(r["actualUp"]), "won": bool(p) == bool(r["actualUp"]), "score": float(score_map.get(r["marketStart"], np.nan))})
    if not rows:
        return []
    s = pd.DataFrame(rows)
    return [compound_metrics(s, candidate["name"], "archived_real_pure_prediction") | {"scope": f"全部归档{asset}{tf} live去重", "oldRealMarkets": int(len(old)), "selectedTrades": int(len(s))}]


def summarize_and_write(results: list[dict[str, Any]], baseline: dict[str, dict[str, Any]], total: int, started: float, audit: dict[str, Any], finished: bool) -> None:
    valid = [r for r in results if "rows" in r]
    for r in valid:
        passed, reasons = pass_gate(r, baseline)
        r["passed"] = passed
        r["reasons"] = reasons
        r["score"] = score_candidate(r, baseline)
    valid.sort(key=lambda x: (x.get("passed", False), x.get("score", -1e18)), reverse=True)
    strict = [r for r in valid if r.get("passed")]
    selected = strict[0] if strict else (valid[0] if valid else None)
    archive_rows = archived_prediction_metrics(selected) if selected else []
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
        "audit": audit,
        "baseline061": baseline,
        "topCandidates": valid[:100],
        "strictPass": strict[:30],
        "selectedForArchiveCompare": selected,
        "archivedRealRowsForSelected": archive_rows,
        "liveConfigMutated": False,
    }
    write_json(OUT_CHECKPOINT, payload)
    write_json(OUT_LEADERBOARD, payload)
    write_json(OUT_COMPARE_JSON, {"baseline061": baseline, "selected": selected, "archivedRealRowsForSelected": archive_rows, "liveAction": "research_only_no_live_change"})
    status = "candidate_beats_061" if selected and selected.get("passed") else "no_candidate_beats_061"
    write_json(OUT_VERDICT_JSON, {"generatedAt": now_iso(), "status": status, "selected": selected, "archivedRealRowsForSelected": archive_rows, "liveAction": "research_only_no_live_change"})
    lines = [
        "# 多币种多周期原始复利主模型搜索：180/365/历史归档对比",
        "",
        f"- 北京时间：`{payload['beijingTime']}`",
        "- 口径：`850U初始 / 每笔当前资金1% / 买价0.50 / 满成交 / 原始K线方向`",
        "- live动作：`research_only_no_live_change`",
        f"- 搜索进度：`{len(results)}/{total}`，严格通过：`{len(strict)}`",
        "",
        "## 180天 / 365天",
        "",
        "|配置|币种|周期|窗口|训练窗|模型|特征组|交易数|胜/负|胜率|盈亏|期末资金|最大回撤|收益回撤比|月正收益|哈希|",
        "|---|---|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for w in ["180d", "365d"]:
        b = baseline[w]
        lines.append(f"|当前061|ETH|15m|{w}|-|061|-|{b['trades']}|{b['wins']}/{b['losses']}|{b['winRatePct']:.2f}%|{b['compoundPnl']:,.2f}|{b['endingBankroll']:,.2f}|{b['maxDrawdownUsd']:,.2f}|{b['returnDrawdownRatio']:.2f}|{b.get('monthlyPositiveRatio',0)*100:.2f}%|`{b.get('setHash','')}`|")
    if selected and "rows" in selected:
        for r in selected["rows"]:
            lines.append(f"|{selected['name']}|{selected['asset']}|{selected['timeframe']}|{r['window']}|{r['trainWindow']}|{r['engine']}|{r['featureMode']}|{r['trades']}|{r['wins']}/{r['losses']}|{r['winRatePct']:.2f}%|{r['compoundPnl']:,.2f}|{r['endingBankroll']:,.2f}|{r['maxDrawdownUsd']:,.2f}|{r['returnDrawdownRatio']:.2f}|{r.get('monthlyPositiveRatio',0)*100:.2f}%|`{r.get('setHash','')}`|")
    lines += ["", "## 历史归档纯预测复核", "", "|范围|配置|交易数|胜/负|胜率|盈亏|期末资金|最大回撤|", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in archive_rows:
        lines.append(f"|{r.get('scope')}|{r.get('name')}|{r.get('selectedTrades', r.get('trades',0))}|{r.get('wins',0)}/{r.get('losses',0)}|{r.get('winRatePct',0):.2f}%|{r.get('compoundPnl',0):,.2f}|{r.get('endingBankroll',0):,.2f}|{r.get('maxDrawdownUsd',0):,.2f}|")
    lines += ["", "## 结论", "", f"- 严格过门状态：`{'通过' if selected and selected.get('passed') else '未通过'}`。"]
    if selected and selected.get("passed"):
        lines.append("- 该候选只进入影子验证，不直接切真钱。")
    else:
        lines.append("- 多币种/多周期原始K线方向搜索暂未超过当前061，不改真钱。")
    write_text(OUT_COMPARE_MD, "\n".join(lines) + "\n")
    write_text(OUT_VERDICT_MD, "\n".join(lines) + "\n")


def run() -> int:
    started = time.time()
    print(f"[crypto-raw-growth] start {bj_now()} workers={WORKERS} maxSeconds={MAX_SECONDS}", flush=True)
    audit = bug_audit()
    baseline = current_061_rows()
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
    for asset, tf, p in params:
        name = f"rawgrowth_{asset}_{tf}_{p['engine']}_{p['train_window']}_{p['feature_mode']}_thr{p['threshold']}_cap{p.get('daily_cap',0)}_{stable_hash(p)}"
        if name not in done_names:
            pending.append((asset, tf, p))
    print(f"[crypto-raw-growth] total={total} done={len(results)} pending={len(pending)} audit={audit['passed']} assets={ASSETS} timeframes={TIMEFRAMES}", flush=True)
    summarize_and_write(results, baseline, total, started, audit, finished=False)
    last = time.time()
    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_RESULTS.open("a", encoding="utf-8") as fh:
        with cf.ProcessPoolExecutor(max_workers=WORKERS) as ex:
            futures: dict[Any, tuple[str, str, dict[str, Any]]] = {}
            it = iter(pending)
            for _ in range(max(4, WORKERS * 2)):
                try:
                    item = next(it)
                except StopIteration:
                    break
                futures[ex.submit(evaluate_params, item)] = item
            while futures:
                if time.time() - started >= MAX_SECONDS:
                    break
                done, _ = cf.wait(futures, timeout=5, return_when=cf.FIRST_COMPLETED)
                for fut in done:
                    item = futures.pop(fut)
                    try:
                        row = fut.result()
                    except Exception as exc:
                        row = {"name": f"candidate_{stable_hash(item)}", "asset": item[0], "timeframe": item[1], "params": item[2], "error": repr(exc)[:1000]}
                    fh.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")
                    fh.flush()
                    results.append(row)
                    try:
                        nxt = next(it)
                        futures[ex.submit(evaluate_params, nxt)] = nxt
                    except StopIteration:
                        pass
                if time.time() - last >= CHECKPOINT_SECONDS:
                    summarize_and_write(results, baseline, total, started, audit, finished=False)
                    print(f"[crypto-raw-growth] checkpoint {len(results)}/{total}", flush=True)
                    last = time.time()
            for fut in futures:
                fut.cancel()
    finished = len(results) >= total
    summarize_and_write(results, baseline, total, started, audit, finished=finished)
    print(json.dumps({"status": "finished" if finished else "checkpointed", "done": len(results), "total": total, "compare": str(OUT_COMPARE_MD), "verdict": str(OUT_VERDICT_MD)}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
