#!/usr/bin/env python3
from __future__ import annotations

"""ETH-only regime expert main-model search.

Research only. It does not mutate live top159 configs, ledgers, process state,
order submission, monitor settings, or current 061 live trading.

Fair metric:
  - ETH only
  - 850U initial bankroll
  - stake 1% current bankroll per selected market
  - buy price fixed at 0.50
  - full fill raw-kline proxy
  - model chooses UP/DOWN/NO_TRADE

Idea:
  Split ETH 15m markets into pre-trade regimes, train small specialist models
  per regime using only past data, and compare against current 061 under the
  same raw-kline fair metric.
"""

import concurrent.futures as cf
import hashlib
import importlib.util
import itertools
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

THREADS = os.environ.get("ETH_REGIME_EXPERT_THREADS", "1")
for key in ["OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(key, THREADS)

import numpy as np
import pandas as pd

ROOT = Path("/Users/mac/polyfun")
NEXT = ROOT / "polyfun-next"
SCRIPTS = NEXT / "scripts"
REPORTS = ROOT / "reports"
BASE_SCRIPT = SCRIPTS / "run_top159_integrated_main_extreme_search.py"
ARCHIVE_SCRIPT = ROOT / "scripts" / "ops" / "run_top159_all_archived_real_eth_compare.py"
CLUSTER_COMPARE_JSON = REPORTS / "top159_061_deep_v2_cluster_180_365_archive_compare_latest.json"

WORKERS = int(os.environ.get("ETH_REGIME_EXPERT_WORKERS", "4"))
MAX_SECONDS = int(os.environ.get("ETH_REGIME_EXPERT_MAX_SECONDS", str(5 * 3600)))
CHECKPOINT_SECONDS = int(os.environ.get("ETH_REGIME_EXPERT_CHECKPOINT_SECONDS", "900"))
PARAM_LIMIT = int(os.environ.get("ETH_REGIME_EXPERT_PARAM_LIMIT", "0"))
MAX_TRAIN_ROWS = int(os.environ.get("ETH_REGIME_EXPERT_MAX_TRAIN_ROWS", "180000"))
RNG_SEED = 20260505

START_BANKROLL = 850.0
STAKE_PCT = 0.01
BUY_PRICE = 0.50

OUT_AUDIT_MD = REPORTS / "eth_regime_expert_mainmodel_bug_audit_latest.md"
OUT_AUDIT_JSON = REPORTS / "eth_regime_expert_mainmodel_bug_audit_latest.json"
OUT_RESULTS = REPORTS / "eth_regime_expert_mainmodel_results_latest.jsonl"
OUT_CHECKPOINT = REPORTS / "eth_regime_expert_mainmodel_checkpoint_latest.json"
OUT_LEADERBOARD = REPORTS / "eth_regime_expert_mainmodel_leaderboard_latest.json"
OUT_COMPARE_MD = REPORTS / "eth_regime_expert_mainmodel_180_365_archive_compare_latest.md"
OUT_COMPARE_JSON = REPORTS / "eth_regime_expert_mainmodel_180_365_archive_compare_latest.json"
OUT_VERDICT_MD = REPORTS / "eth_regime_expert_mainmodel_unique_verdict_latest.md"
OUT_VERDICT_JSON = REPORTS / "eth_regime_expert_mainmodel_unique_verdict_latest.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base = load_module("eth_regime_integrated_base", BASE_SCRIPT)
archive = load_module("eth_regime_archive_compare", ARCHIVE_SCRIPT)


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
    id_cols = [c for c in ["dt", "regime", "pred_up", "label_up", "won"] if c in sel.columns]
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
        "setHash": stable_hash(sel[id_cols].to_dict("records")) if len(sel) else stable_hash([]),
    }


def current_061_rows() -> dict[str, dict[str, Any]]:
    payload = json.loads(CLUSTER_COMPARE_JSON.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for row in payload.get("validationRows", []):
        if row.get("name") == "current_live_06173_cluster_gate" and row.get("method") == "full_fill_buy_0.50":
            out[row["window"]] = dict(row)
    if set(out) != {"180d", "365d"}:
        raise RuntimeError("cannot locate current061 baseline rows")
    return out


def build_frame() -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    df, features, truth = base.build_integrated_frame()
    # Add transparent, numeric pre-trade environment helpers. They are built from
    # shifted/closed features already present in build_integrated_frame.
    out = df.copy().sort_values("dt").reset_index(drop=True)
    out["bj_hour"] = pd.to_datetime(out["dt"], utc=True).dt.tz_convert("Asia/Shanghai").dt.hour.astype(float)
    out["bj_hour_bucket"] = (out["bj_hour"] // 4).astype(float)
    extra = ["bj_hour", "bj_hour_bucket"]
    for a, b, name in [
        ("1h_ema_8_32", "4h_ema_8_32", "trend_1h4h_agree"),
        ("ema_8_32", "1h_ema_8_32", "trend_15m1h_agree"),
    ]:
        if a in out.columns and b in out.columns:
            out[name] = (np.sign(pd.to_numeric(out[a], errors="coerce")) == np.sign(pd.to_numeric(out[b], errors="coerce"))).astype(float)
            extra.append(name)
    for c in ["vol_16", "vol_32", "range_16", "range_32", "1h_vol_16", "1h_range_16", "4h_vol_16", "4h_range_16", "bb_pos", "1h_bb_pos", "4h_bb_pos"]:
        if c in out.columns:
            s = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            out[f"env_{c}_rank"] = s.rank(pct=True).astype(float)
            extra.append(f"env_{c}_rank")
    clean = list(dict.fromkeys(features + extra))
    forbidden = set(base.base.FORBIDDEN_FEATURES)
    prefixes = tuple(base.base.FORBIDDEN_PREFIXES)
    final: list[str] = []
    for c in clean:
        if c in forbidden or c.startswith(prefixes):
            continue
        if c in out.columns:
            s = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if int(s.notna().sum()) >= max(500, int(len(out) * 0.50)):
                out[c] = s
                final.append(c)
    truth = dict(truth)
    truth["regimeExtraFeatures"] = extra
    truth["finalRows"] = int(len(out))
    truth["finalFeatureCount"] = int(len(final))
    return out.dropna(subset=["dt", "label_up"]).sort_values("dt").reset_index(drop=True), final, truth


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
    elif mode == "trend_multi":
        keys = ("ret_", "ema_", "ema_dist_", "rsi", "bb_pos", "range_", "vol_", "env_")
        out = [c for c in features if any(k in c for k in keys)]
    elif mode == "wide":
        out = list(features)
    else:
        out = list(features)
    return out


def fit_model(engine: str, train: pd.DataFrame, feats: list[str], params: dict[str, Any], random_labels: bool = False):
    if len(train) < int(params.get("min_fit_rows", 300)) or not feats:
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
            n_estimators=int(params.get("n_estimators", 90)), learning_rate=float(params.get("learning_rate", 0.035)),
            num_leaves=int(params.get("num_leaves", 16)), min_child_samples=int(params.get("min_child_samples", 80)),
            subsample=float(params.get("subsample", 0.88)), colsample_bytree=float(params.get("colsample_bytree", 0.88)),
            reg_lambda=float(params.get("reg_lambda", 1.0)), random_state=RNG_SEED, n_jobs=int(THREADS), verbose=-1,
        )
    elif engine == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=int(params.get("n_estimators", 90)), max_depth=int(params.get("depth", 3)), learning_rate=float(params.get("learning_rate", 0.035)),
            subsample=float(params.get("subsample", 0.88)), colsample_bytree=float(params.get("colsample_bytree", 0.88)),
            reg_lambda=float(params.get("reg_lambda", 1.0)), random_state=RNG_SEED, n_jobs=int(THREADS), eval_metric="logloss", verbosity=0,
        )
    else:
        return None
    model.fit(x, y)
    return model


def predict_prob(model: Any, val: pd.DataFrame, feats: list[str]) -> np.ndarray:
    x = val[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def regime_key_from_thresholds(df: pd.DataFrame, params: dict[str, Any], thresholds: dict[str, tuple[float, float]]) -> pd.Series:
    mode = params.get("regime_mode", "trend_vol")
    parts: list[pd.Series] = []
    if mode in {"trend_vol", "trend_pos", "full_env", "time_trend", "shock_like"}:
        c1 = "1h_ema_8_32" if "1h_ema_8_32" in df.columns else "ema_8_32"
        c2 = "4h_ema_8_32" if "4h_ema_8_32" in df.columns else c1
        t1 = np.sign(pd.to_numeric(df.get(c1, 0), errors="coerce").fillna(0.0)).astype(int).astype(str)
        t2 = np.sign(pd.to_numeric(df.get(c2, 0), errors="coerce").fillna(0.0)).astype(int).astype(str)
        parts.append("tr" + t1 + t2)
    if mode in {"trend_vol", "full_env", "shock_like"}:
        vc = "1h_vol_16" if "1h_vol_16" in df.columns else ("vol_16" if "vol_16" in df.columns else "range_16")
        lo, hi = thresholds.get(vc, (0.0, 0.0))
        s = pd.to_numeric(df.get(vc, 0), errors="coerce").fillna(0.0)
        b = pd.Series(np.where(s <= lo, "lo", np.where(s >= hi, "hi", "mid")), index=df.index)
        parts.append("vol" + b)
    if mode in {"trend_pos", "full_env", "shock_like"}:
        pc = "1h_bb_pos" if "1h_bb_pos" in df.columns else ("bb_pos" if "bb_pos" in df.columns else "ema_dist_32")
        lo, hi = thresholds.get(pc, (-0.5, 0.5))
        s = pd.to_numeric(df.get(pc, 0), errors="coerce").fillna(0.0)
        b = pd.Series(np.where(s <= lo, "low", np.where(s >= hi, "high", "mid")), index=df.index)
        parts.append("pos" + b)
    if mode in {"time_trend", "full_env"}:
        h = pd.to_datetime(df["dt"], utc=True).dt.tz_convert("Asia/Shanghai").dt.hour.astype(int) // 4
        parts.append("h" + h.astype(str))
    if mode == "shock_like":
        rc = "1h_range_16" if "1h_range_16" in df.columns else ("range_16" if "range_16" in df.columns else "vol_16")
        lo, hi = thresholds.get(rc, (0.0, 0.0))
        s = pd.to_numeric(df.get(rc, 0), errors="coerce").fillna(0.0)
        b = pd.Series(np.where(s >= hi, "shock", "normal"), index=df.index)
        parts.append("rng" + b)
    if not parts:
        return pd.Series("all", index=df.index)
    out = parts[0].astype(str)
    for p in parts[1:]:
        out = out + "|" + p.astype(str)
    return out.astype(str)


def fit_thresholds(train: pd.DataFrame, params: dict[str, Any]) -> dict[str, tuple[float, float]]:
    cols = ["1h_vol_16", "vol_16", "range_16", "1h_range_16", "1h_bb_pos", "bb_pos", "ema_dist_32"]
    qlo = float(params.get("regime_q_low", 0.33))
    qhi = float(params.get("regime_q_high", 0.67))
    out: dict[str, tuple[float, float]] = {}
    for c in cols:
        if c in train.columns:
            s = pd.to_numeric(train[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(s) > 100:
                out[c] = (float(s.quantile(qlo)), float(s.quantile(qhi)))
    return out


def select_from_prob(val: pd.DataFrame, prob: np.ndarray, params: dict[str, Any], regime: pd.Series | None = None) -> pd.DataFrame:
    pred_up = prob >= 0.5
    score = np.maximum(prob, 1.0 - prob)
    mask = score >= float(params.get("threshold", 0.55))
    if params.get("score_band") == "mid_only":
        mask &= score <= float(params.get("max_score", 0.76))
    elif params.get("score_band") == "high_only":
        mask &= score >= float(params.get("high_min", params.get("threshold", 0.58)))
    out = val.loc[mask, ["dt", "label_up"]].copy().reset_index(drop=True)
    out["pred_up"] = pred_up[mask].astype(bool)
    out["score"] = score[mask]
    if regime is not None:
        out["regime"] = regime.loc[val.index[mask]].astype(str).to_numpy()
    else:
        out["regime"] = "global"
    out["won"] = out["pred_up"].to_numpy() == out["label_up"].astype(bool).to_numpy()
    cap = int(params.get("daily_cap", 0) or 0)
    if cap > 0 and len(out):
        out["day"] = pd.to_datetime(out["dt"], utc=True).dt.floor("D")
        out = out.sort_values(["day", "score"], ascending=[True, False]).groupby("day", sort=False).head(cap).drop(columns=["day"]).sort_values("dt").reset_index(drop=True)
    return out


def internal_active_regimes(train: pd.DataFrame, feats: list[str], params: dict[str, Any], thresholds: dict[str, tuple[float, float]]) -> set[str]:
    # Fit on earlier training segment, judge regimes on the later training segment only.
    tr = train.sort_values("dt").copy().reset_index(drop=True)
    if len(tr) < 2000:
        return set()
    cut = int(len(tr) * float(params.get("internal_train_frac", 0.78)))
    fit = tr.iloc[:cut].copy()
    hold = tr.iloc[cut:].copy()
    if len(hold) < 300:
        return set()
    rg_hold = regime_key_from_thresholds(hold, params, thresholds)
    engine = params["engine"]
    # Use a global internal model as a stable low-variance judge for which regimes are tradable.
    model = fit_model(engine, fit, feats, params)
    if model is None:
        return set()
    prob = predict_prob(model, hold, feats)
    selected = select_from_prob(hold, prob, params, rg_hold)
    if selected.empty:
        return set()
    min_sig = int(params.get("min_regime_signals", 40))
    min_wr = float(params.get("min_regime_wr", 0.515))
    active: set[str] = set()
    for key, g in selected.groupby("regime"):
        if len(g) >= min_sig and float(g["won"].mean()) >= min_wr:
            active.add(str(key))
    return active


def evaluate_regime_expert(df: pd.DataFrame, features: list[str], params: dict[str, Any], name: str) -> dict[str, Any] | None:
    feats = feature_subset(features, params["feature_mode"])
    max_features = int(params.get("max_features", 120))
    if max_features > 0:
        feats = feats[:max_features]
    if len(feats) < 8:
        return None
    rows: list[dict[str, Any]] = []
    for window in ["180d", "365d"]:
        train, val = split_train_val(df, window, params["train_window"])
        if len(train) < 1000 or len(val) < 100:
            return None
        thresholds = fit_thresholds(train, params)
        rg_train = regime_key_from_thresholds(train, params, thresholds)
        rg_val = regime_key_from_thresholds(val, params, thresholds)
        active = internal_active_regimes(train, feats, params, thresholds)
        selected_parts: list[pd.DataFrame] = []
        expert_min_rows = int(params.get("expert_min_rows", 700))
        fallback = str(params.get("fallback", "no_trade"))
        # Train one expert per active regime. This is the actual mixture-of-experts path.
        for key in sorted(active):
            tr_sub = train.loc[rg_train == key].copy()
            va_sub = val.loc[rg_val == key].copy()
            if len(tr_sub) < expert_min_rows or len(va_sub) == 0:
                continue
            model = fit_model(params["engine"], tr_sub, feats, params)
            if model is None:
                continue
            prob = predict_prob(model, va_sub, feats)
            selected_parts.append(select_from_prob(va_sub, prob, params, rg_val.loc[va_sub.index]))
        if fallback == "global_active":
            rest = val.loc[~rg_val.isin(active)].copy()
            if len(rest):
                model = fit_model(params["engine"], train, feats, params)
                if model is not None:
                    prob = predict_prob(model, rest, feats)
                    selected_parts.append(select_from_prob(rest, prob, params, rg_val.loc[rest.index]))
        elif fallback == "global_all" and not selected_parts:
            model = fit_model(params["engine"], train, feats, params)
            if model is not None:
                prob = predict_prob(model, val, feats)
                selected_parts.append(select_from_prob(val, prob, params, rg_val))
        if not selected_parts:
            return None
        selected = pd.concat(selected_parts, ignore_index=True).drop_duplicates(["dt"]).sort_values("dt").reset_index(drop=True)
        if selected.empty:
            return None
        row = compound_metrics(selected, name, window, "regime_expert_raw_full_fill_buy_0.50")
        row.update({
            "engine": params["engine"],
            "trainWindow": params["train_window"],
            "featureMode": params["feature_mode"],
            "regimeMode": params["regime_mode"],
            "threshold": params["threshold"],
            "activeRegimeCount": int(len(active)),
            "fallback": fallback,
            "featureCount": int(len(feats)),
        })
        rows.append(row)
    return {"name": name, "params": params, "rows": rows, "featureCount": int(rows[0]["featureCount"])}


def pass_gate(candidate: dict[str, Any], baseline: dict[str, dict[str, Any]]) -> tuple[bool, list[str]]:
    by = {r["window"]: r for r in candidate.get("rows", [])}
    reasons: list[str] = []
    for w in ["180d", "365d"]:
        if w not in by:
            reasons.append(f"{w}_missing")
            continue
        r = by[w]
        b = baseline[w]
        if r["compoundPnl"] <= b["compoundPnl"]:
            reasons.append(f"{w}_pnl_not_above_061")
        if w == "365d" and r["winRatePct"] < b["winRatePct"]:
            reasons.append("365d_winrate_below_061")
        if w == "180d" and r["winRatePct"] < b["winRatePct"] - 0.15:
            reasons.append("180d_winrate_too_low")
        if r["maxDrawdownUsd"] > b["maxDrawdownUsd"] * 1.05:
            reasons.append(f"{w}_drawdown_too_high")
        if r["trades"] < max(100 if w == "180d" else 200, int(b["trades"] * 0.45)):
            reasons.append(f"{w}_too_few_trades")
    return not reasons, reasons


def score_candidate(candidate: dict[str, Any], baseline: dict[str, dict[str, Any]]) -> float:
    if "rows" not in candidate:
        return -1e18
    by = {r["window"]: r for r in candidate["rows"]}
    score = 0.0
    for w in ["180d", "365d"]:
        r = by[w]
        b = baseline[w]
        score += (r["compoundPnl"] - b["compoundPnl"]) / max(abs(b["compoundPnl"]), 1.0) * 10000.0
        score += (r["winRatePct"] - b["winRatePct"]) * 50.0
        score -= max(0.0, r["maxDrawdownUsd"] - b["maxDrawdownUsd"]) / max(b["maxDrawdownUsd"], 1.0) * 500.0
        score += r.get("returnDrawdownRatio", 0.0) * 2.0
    return score


def param_grid() -> list[dict[str, Any]]:
    engines = [x for x in os.environ.get("ETH_REGIME_EXPERT_ENGINES", "logistic,lightgbm,xgboost").split(",") if x]
    train_windows = ["1y", "3y", "5y", "full"]
    feature_modes = ["base15", "base_plus_1h4h", "base_plus_daily", "trend_multi", "wide"]
    regime_modes = ["trend_vol", "trend_pos", "time_trend", "full_env", "shock_like"]
    thresholds = [0.52, 0.53, 0.54, 0.55, 0.56, 0.58, 0.60]
    fallbacks = ["no_trade", "global_active"]
    min_regime_signals = [30, 50, 80]
    min_regime_wrs = [0.505, 0.515, 0.525, 0.54]
    expert_min_rows = [500, 900, 1400]
    rows: list[dict[str, Any]] = []
    for engine, tw, fm, rm, th, fb, ms, mw, er in itertools.product(engines, train_windows, feature_modes, regime_modes, thresholds, fallbacks, min_regime_signals, min_regime_wrs, expert_min_rows):
        if engine == "logistic":
            for c in [0.15, 0.35, 0.75, 1.5, 3.0]:
                rows.append({"engine": engine, "train_window": tw, "feature_mode": fm, "regime_mode": rm, "threshold": th, "fallback": fb, "min_regime_signals": ms, "min_regime_wr": mw, "expert_min_rows": er, "C": c, "n_estimators": 1, "learning_rate": 0.04, "num_leaves": 12, "min_child_samples": 80, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0, "depth": 3, "max_features": 120})
        else:
            presets = [(70, 0.04, 12, 80, 1.0, 3), (110, 0.035, 16, 100, 1.2, 3), (150, 0.025, 24, 120, 2.0, 4)]
            for ne, lr, leaves, mcs, reg, depth in presets:
                rows.append({"engine": engine, "train_window": tw, "feature_mode": fm, "regime_mode": rm, "threshold": th, "fallback": fb, "min_regime_signals": ms, "min_regime_wr": mw, "expert_min_rows": er, "C": 1.0, "n_estimators": ne, "learning_rate": lr, "num_leaves": leaves, "min_child_samples": mcs, "subsample": 0.88, "colsample_bytree": 0.88, "reg_lambda": reg, "depth": depth, "max_features": 120})
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


def eval_worker(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    if _DF is None or _FEATURES is None:
        raise RuntimeError("worker not initialized")
    idx, params = item
    name = f"regimeexpert_{params['engine']}_{params['train_window']}_{params['feature_mode']}_{params['regime_mode']}_thr{params['threshold']}_{stable_hash(params)}"
    try:
        out = evaluate_regime_expert(_DF, _FEATURES, params, name)
        return out if out is not None else {"name": name, "params": params, "error": "empty_or_fit_failed"}
    except Exception as exc:
        return {"name": name, "params": params, "error": repr(exc)[:1000]}


def archive_rows_for_candidate(df: pd.DataFrame, features: list[str], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        fill_rows, _ = archive.load_live_fill_rows()
        all_eth = archive.aggregate_logical(fill_rows, "全部归档ETH15m live去重")
        strict_mask = fill_rows["sourcePath"].str.contains("slot01|forcedtrade|base_forcedtrade|__live_eth|logs_v5_exp10_bp0480__live_eth", case=False, regex=True)
        strict = archive.aggregate_logical(fill_rows[strict_mask].copy(), "严格旧slot1/ETH相关live去重") if strict_mask.any() else all_eth.iloc[0:0]
        for old in [strict, all_eth]:
            if old.empty:
                continue
            params = candidate["params"]
            start = old["marketStart"].min()
            end = old["marketStart"].max()
            train = df[df["dt"] < start - pd.Timedelta(hours=4)].copy()
            if params["train_window"] != "full":
                days = {"1y": 365, "3y": 1095, "5y": 1825}[params["train_window"]]
                train = train[train["dt"] >= start - pd.Timedelta(days=days)].copy()
            val = df[(df["dt"] >= start) & (df["dt"] <= end)].copy()
            if train.empty or val.empty:
                continue
            feats = feature_subset(features, params["feature_mode"])[: int(params.get("max_features", 120))]
            thresholds = fit_thresholds(train, params)
            rg_train = regime_key_from_thresholds(train, params, thresholds)
            rg_val = regime_key_from_thresholds(val, params, thresholds)
            active = internal_active_regimes(train, feats, params, thresholds)
            selected_parts: list[pd.DataFrame] = []
            for key in sorted(active):
                tr_sub = train.loc[rg_train == key].copy()
                va_sub = val.loc[rg_val == key].copy()
                if len(tr_sub) < int(params.get("expert_min_rows", 700)) or len(va_sub) == 0:
                    continue
                model = fit_model(params["engine"], tr_sub, feats, params)
                if model is None:
                    continue
                prob = predict_prob(model, va_sub, feats)
                selected_parts.append(select_from_prob(va_sub, prob, params, rg_val.loc[va_sub.index]))
            if not selected_parts:
                rows.append({"scope": old["scopeName"].iloc[0], "name": candidate["name"], "oldRealMarkets": int(len(old)), "selectedTrades": 0})
                continue
            selected = pd.concat(selected_parts, ignore_index=True).drop_duplicates(["dt"]).sort_values("dt")
            pred = selected[["dt", "pred_up", "score"]].copy()
            merged = old.merge(pred, left_on="marketStart", right_on="dt", how="left")
            chosen = merged[merged["pred_up"].notna()].copy().sort_values("marketStart")
            if chosen.empty:
                rows.append({"scope": old["scopeName"].iloc[0], "name": candidate["name"], "oldRealMarkets": int(len(old)), "selectedTrades": 0})
                continue
            model_won = chosen["pred_up"].astype(bool).to_numpy() == chosen["actualUp"].astype(bool).to_numpy()
            arch_sel = pd.DataFrame({"dt": chosen["marketStart"].to_numpy(), "label_up": chosen["actualUp"].astype(bool).to_numpy(), "pred_up": chosen["pred_up"].astype(bool).to_numpy(), "won": model_won, "regime": "archive"})
            metrics = compound_metrics(arch_sel, candidate["name"], str(old["scopeName"].iloc[0]), "archive_raw_buy_0.50")
            rows.append({"scope": old["scopeName"].iloc[0], "name": candidate["name"], "oldRealMarkets": int(len(old)), "selectedTrades": int(len(chosen)), "skippedTrades": int(len(old) - len(chosen)), "wins": int(model_won.sum()), "losses": int(len(model_won) - int(model_won.sum())), "winRatePct": round(100.0 * float(model_won.mean()), 6), "compoundPnl": metrics["compoundPnl"], "endingBankroll": metrics["endingBankroll"], "maxDrawdownUsd": metrics["maxDrawdownUsd"], "avgScore": round(float(chosen["score"].mean()), 6), "setHash": stable_hash(chosen[["marketSlug", "pred_up", "actualUp"]].to_dict("records"))})
    except Exception as exc:
        rows.append({"scope": "archive_error", "name": candidate.get("name"), "error": repr(exc)[:800]})
    return rows


def bug_audit(df: pd.DataFrame, features: list[str], baseline: dict[str, dict[str, Any]]) -> dict[str, Any]:
    params = {"engine": "logistic", "train_window": "3y", "feature_mode": "base_plus_1h4h", "regime_mode": "trend_vol", "threshold": 0.55, "fallback": "no_trade", "min_regime_signals": 30, "min_regime_wr": 0.505, "expert_min_rows": 500, "C": 0.75, "n_estimators": 1, "learning_rate": 0.04, "num_leaves": 12, "min_child_samples": 80, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0, "depth": 3, "max_features": 120}
    c1 = evaluate_regime_expert(df, features, params, "audit_repeat")
    c2 = evaluate_regime_expert(df, features, params, "audit_repeat")
    repeat = []
    if c1 and c2:
        for a, b in zip(c1["rows"], c2["rows"]):
            repeat.append({"window": a["window"], "hash1": a["setHash"], "hash2": b["setHash"], "pnl1": a["compoundPnl"], "pnl2": b["compoundPnl"], "passed": a["setHash"] == b["setHash"] and a["compoundPnl"] == b["compoundPnl"]})
    random_label = {"status": "not_run", "passed": False}
    try:
        train, val = split_train_val(df, "365d", "3y")
        feats = feature_subset(features, "base_plus_1h4h")[:120]
        # Keep regime discovery normal, but randomize labels inside model fitting by replacing train label.
        train2 = train.copy()
        train2["label_up"] = np.random.default_rng(RNG_SEED).permutation(train2["label_up"].astype(int).to_numpy())
        thresholds = fit_thresholds(train2, params)
        active = internal_active_regimes(train2, feats, params, thresholds)
        rg_train = regime_key_from_thresholds(train2, params, thresholds)
        rg_val = regime_key_from_thresholds(val, params, thresholds)
        parts: list[pd.DataFrame] = []
        for key in sorted(active):
            tr_sub = train2.loc[rg_train == key].copy()
            va_sub = val.loc[rg_val == key].copy()
            if len(tr_sub) < 500 or len(va_sub) == 0:
                continue
            model = fit_model("logistic", tr_sub, feats, params)
            if model is None:
                continue
            prob = predict_prob(model, va_sub, feats)
            parts.append(select_from_prob(va_sub, prob, params, rg_val.loc[va_sub.index]))
        if parts:
            sel = pd.concat(parts, ignore_index=True).drop_duplicates(["dt"])
            wr = float(sel["won"].mean() * 100.0) if len(sel) else 0.0
            random_label = {"status": "ok", "selectedTrades": int(len(sel)), "winRatePct": round(wr, 6), "passed": len(sel) < 50 or 43.0 <= wr <= 57.0}
        else:
            random_label = {"status": "ok_empty", "selectedTrades": 0, "winRatePct": 0.0, "passed": True}
    except Exception as exc:
        random_label = {"status": "error", "error": repr(exc)[:300], "passed": False}
    forbidden = [c for c in features if c in base.base.FORBIDDEN_FEATURES or c.startswith(tuple(base.base.FORBIDDEN_PREFIXES))]
    audit = {"generatedAt": now_iso(), "beijingTime": bj_now(), "researchOnlyNoLiveChange": True, "fairMetric": "850U initial, stake 1%, buy 0.50, full fill", "timePolicy": "closed_candle_available_at_v2 via shifted base features and daily closed bars", "regimePolicy": "regime thresholds and active regime selection are learned only from the training window", "forbiddenFeatureHits": forbidden, "repeatability": repeat, "randomLabelAudit": random_label, "baseline061": baseline}
    audit["passed"] = not forbidden and bool(repeat) and all(r["passed"] for r in repeat) and bool(random_label.get("passed"))
    write_json(OUT_AUDIT_JSON, audit)
    lines = ["# ETH 分环境专家模型：回测器审计", "", f"- 北京时间：`{audit['beijingTime']}`", "- 研究动作：`research_only_no_live_change`", "- 当前真钱：不改 061。", f"- 禁用字段命中：`{forbidden}`", f"- 重复运行：`{repeat}`", f"- 随机标签测试：`{random_label}`", f"- 审计通过：`{audit['passed']}`"]
    write_text(OUT_AUDIT_MD, "\n".join(lines) + "\n")
    return audit


def summarize(results: list[dict[str, Any]], baseline: dict[str, dict[str, Any]], total: int, started: float, data_truth: dict[str, Any], audit: dict[str, Any], finished: bool) -> None:
    valid = [r for r in results if r.get("rows")]
    for r in valid:
        ok, reasons = pass_gate(r, baseline)
        r["passed"] = ok
        r["reasons"] = reasons
        r["score"] = score_candidate(r, baseline)
    valid.sort(key=lambda x: (x.get("passed", False), x.get("score", -1e18)), reverse=True)
    strict = [r for r in valid if r.get("passed")]
    selected = strict[0] if strict else (valid[0] if valid else None)
    archive_rows = archive_rows_for_candidate(_DF_MAIN, _FEATURES_MAIN, selected) if selected and _DF_MAIN is not None and _FEATURES_MAIN is not None else []
    payload = {"generatedAt": now_iso(), "beijingTime": bj_now(), "researchOnlyNoLiveChange": True, "finished": finished, "elapsedSeconds": round(time.time() - started, 3), "workers": WORKERS, "totalCandidates": total, "doneCount": len(results), "validCount": len(valid), "strictPassCount": len(strict), "dataTruth": data_truth, "audit": audit, "baseline061": baseline, "topCandidates": valid[:100], "strictPass": strict[:30], "selectedForArchiveCompare": selected, "archivedRealRowsForSelected": archive_rows, "liveConfigMutated": False}
    write_json(OUT_CHECKPOINT, payload)
    write_json(OUT_LEADERBOARD, payload)
    write_json(OUT_COMPARE_JSON, payload)
    status = "candidate_beats_061" if selected and selected.get("passed") else "no_candidate_beats_061"
    write_json(OUT_VERDICT_JSON, {"generatedAt": payload["generatedAt"], "status": status, "selected": selected, "archivedRealRowsForSelected": archive_rows, "liveAction": "research_only_no_live_change"})
    render_markdown(payload)


def render_markdown(payload: dict[str, Any]) -> None:
    baseline = payload["baseline061"]
    selected = payload.get("selectedForArchiveCompare")
    lines = ["# ETH 分环境专家模型：180/365/历史归档对比", "", f"- 北京时间：`{payload['beijingTime']}`", "- 口径：`850U初始 / 每笔当前资金1% / 买价0.50 / 满成交 / ETH原始K线方向`", "- live动作：`research_only_no_live_change`", f"- 搜索进度：`{payload['doneCount']}/{payload['totalCandidates']}`，严格通过：`{payload['strictPassCount']}`", "", "## 180天 / 365天", "", "|配置|窗口|训练窗|模型|特征组|环境|交易数|胜/负|胜率|盈亏|期末资金|最大回撤|收益回撤比|月正收益|哈希|", "|---|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for w in ["180d", "365d"]:
        b = baseline[w]
        lines.append(f"|当前061|{w}|-|061_cluster_gate|-|-|{b['trades']}|{b['wins']}/{b['losses']}|{b['winRatePct']:.2f}%|{b['compoundPnl']:,.2f}|{b['endingBankroll']:,.2f}|{b['maxDrawdownUsd']:,.2f}|{b['returnDrawdownRatio']:.2f}|{b.get('monthlyPositiveRatio', 0):.2%}|`{b['setHash']}`|")
    if selected and selected.get("rows"):
        for r in selected["rows"]:
            p = selected.get("params", {})
            lines.append(f"|{selected['name']}|{r['window']}|{r.get('trainWindow','-')}|{r.get('engine','-')}|{r.get('featureMode','-')}|{r.get('regimeMode','-')}|{r['trades']}|{r['wins']}/{r['losses']}|{r['winRatePct']:.2f}%|{r['compoundPnl']:,.2f}|{r['endingBankroll']:,.2f}|{r['maxDrawdownUsd']:,.2f}|{r['returnDrawdownRatio']:.2f}|{r['monthlyPositiveRatio']:.2%}|`{r['setHash']}`|")
    lines += ["", "## 历史归档纯预测复核", "", "|范围|配置|交易数|胜/负|胜率|盈亏|期末资金|最大回撤|", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in payload.get("archivedRealRowsForSelected", []):
        if "error" in r:
            lines.append(f"|{r.get('scope')}|{r.get('name')}|错误|-|-|-|-|-|")
        else:
            lines.append(f"|{r['scope']}|{r['name']}|{r['selectedTrades']}|{r.get('wins',0)}/{r.get('losses',0)}|{r.get('winRatePct',0):.2f}%|{r.get('compoundPnl',0):,.2f}|{r.get('endingBankroll',0):,.2f}|{r.get('maxDrawdownUsd',0):,.2f}|")
    status = "通过" if selected and selected.get("passed") else "未通过"
    lines += ["", "## 结论", "", f"- 严格过门状态：`{status}`。", "- 若未通过：不改真钱061；说明 ETH15m 分环境专家模型仍未可靠超过061。"]
    write_text(OUT_COMPARE_MD, "\n".join(lines) + "\n")
    write_text(OUT_VERDICT_MD, "\n".join(lines) + "\n")


_DF_MAIN: pd.DataFrame | None = None
_FEATURES_MAIN: list[str] | None = None


def run() -> int:
    global _DF_MAIN, _FEATURES_MAIN
    started = time.time()
    print(f"[regime-expert] start {bj_now()} workers={WORKERS} maxSeconds={MAX_SECONDS}", flush=True)
    df, features, truth = build_frame()
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
        name = f"regimeexpert_{p['engine']}_{p['train_window']}_{p['feature_mode']}_{p['regime_mode']}_thr{p['threshold']}_{stable_hash(p)}"
        if name not in done_names:
            pending.append((i, p))
    print(f"[regime-expert] rows={len(df)} features={len(features)} total={total} done={len(results)} pending={len(pending)} audit={audit['passed']}", flush=True)
    summarize(results, baseline, total, started, truth, audit, finished=False)
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
                    summarize(results, baseline, total, started, truth, audit, finished=False)
                    print(f"[regime-expert] checkpoint {len(results)}/{total}", flush=True)
                    last = time.time()
            for fut in futures:
                fut.cancel()
    finished = len(results) >= total
    summarize(results, baseline, total, started, truth, audit, finished=finished)
    print(json.dumps({"status": "finished" if finished else "checkpointed", "done": len(results), "total": total, "compare": str(OUT_COMPARE_MD), "verdict": str(OUT_VERDICT_MD)}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
