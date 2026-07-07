#!/usr/bin/env python3
from __future__ import annotations

"""ETH 15m delayed-entry confirmation research.

Research only. Does not mutate live top159 configs, ledgers, process state,
order submission, monitor settings, or current 061 live trading.

Fair metric:
  - ETH 15m only
  - 850U initial bankroll
  - stake 1% current bankroll per selected market
  - full-fill proxy with buy prices 0.50/0.52/0.55
  - decisions at market open +1/+2/+3 minutes using only elapsed 1m bars
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

THREADS = os.environ.get("ETH_DELAYED_ENTRY_THREADS", "1")
for key in ["OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(key, THREADS)

import numpy as np
import pandas as pd

ROOT = Path("/Users/mac/polyfun")
NEXT = ROOT / "polyfun-next"
SCRIPTS = NEXT / "scripts"
REPORTS = ROOT / "reports"
RAW = ROOT / "data" / "raw"
BASE_SCRIPT = SCRIPTS / "run_top159_integrated_main_extreme_search.py"
ARCHIVE_SCRIPT = ROOT / "scripts" / "ops" / "run_top159_all_archived_real_eth_compare.py"
CLUSTER_COMPARE_JSON = REPORTS / "top159_061_deep_v2_cluster_180_365_archive_compare_latest.json"

WORKERS = int(os.environ.get("ETH_DELAYED_ENTRY_WORKERS", "4"))
MAX_SECONDS = int(os.environ.get("ETH_DELAYED_ENTRY_MAX_SECONDS", str(5 * 3600)))
CHECKPOINT_SECONDS = int(os.environ.get("ETH_DELAYED_ENTRY_CHECKPOINT_SECONDS", "900"))
PARAM_LIMIT = int(os.environ.get("ETH_DELAYED_ENTRY_PARAM_LIMIT", "0"))
MAX_TRAIN_ROWS = int(os.environ.get("ETH_DELAYED_ENTRY_MAX_TRAIN_ROWS", "180000"))
RNG_SEED = 20260506

START_BANKROLL = 850.0
STAKE_PCT = 0.01
MAIN_BUY_PRICE = 0.50
PRESSURE_BUY_PRICES = [0.50, 0.52, 0.55]

OUT_DATA_TRUTH = REPORTS / "eth_delayed_entry_data_truth_latest.json"
OUT_AUDIT_MD = REPORTS / "eth_delayed_entry_bug_audit_latest.md"
OUT_AUDIT_JSON = REPORTS / "eth_delayed_entry_bug_audit_latest.json"
OUT_RESULTS = REPORTS / "eth_delayed_entry_results_latest.jsonl"
OUT_CHECKPOINT = REPORTS / "eth_delayed_entry_checkpoint_latest.json"
OUT_CONFIRM_LEADER = REPORTS / "eth_delayed_entry_061_confirm_leaderboard_latest.json"
OUT_MAIN_LEADER = REPORTS / "eth_delayed_entry_mainmodel_leaderboard_latest.json"
OUT_COMPARE_MD = REPORTS / "eth_delayed_entry_180_365_archive_compare_latest.md"
OUT_COMPARE_JSON = REPORTS / "eth_delayed_entry_180_365_archive_compare_latest.json"
OUT_VERDICT_MD = REPORTS / "eth_delayed_entry_unique_verdict_latest.md"
OUT_VERDICT_JSON = REPORTS / "eth_delayed_entry_unique_verdict_latest.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base = load_module("eth_delayed_base", BASE_SCRIPT)
archive = load_module("eth_delayed_archive", ARCHIVE_SCRIPT)


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


def compound_metrics(selected: pd.DataFrame, name: str, window: str, buy_price: float, method: str) -> dict[str, Any]:
    sel = selected.sort_values("dt").reset_index(drop=True).copy()
    won = sel["won"].astype(bool).to_numpy()
    eq = START_BANKROLL
    curve = np.empty(len(sel), dtype=float)
    for i, ok in enumerate(won):
        stake = eq * STAKE_PCT
        ret = (1.0 / buy_price) - 1.0 if ok else -1.0
        eq += stake * ret
        if eq < 0:
            eq = 0.0
        curve[i] = eq
    max_dd, max_dd_pct, peak_i, trough_i = max_drawdown(curve)
    wins = int(won.sum())
    losses = int(len(won) - wins)
    id_cols = [c for c in ["dt", "decisionDelayMin", "pred_up", "label_up", "won"] if c in sel.columns]
    return {
        "name": name,
        "window": window,
        "method": method,
        "buyPrice": buy_price,
        "decisionDelayMin": int(sel["decisionDelayMin"].mode().iloc[0]) if len(sel) and "decisionDelayMin" in sel.columns else None,
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


def fixed_1m_timestamp(ts: pd.Series) -> pd.Series:
    s = pd.to_numeric(ts, errors="coerce").astype("Int64")
    # Binance rows before 2025 are milliseconds; newer collector rows were stored as microseconds.
    unit = np.where(s.astype("float64") > 1e15, "us", "ms")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns, UTC]")
    ms_mask = unit == "ms"
    us_mask = unit == "us"
    out.loc[ms_mask] = pd.to_datetime(s.loc[ms_mask].astype("int64"), unit="ms", utc=True, errors="coerce")
    out.loc[us_mask] = pd.to_datetime(s.loc[us_mask].astype("int64"), unit="us", utc=True, errors="coerce")
    return out


def load_1m_fixed() -> tuple[pd.DataFrame, dict[str, Any]]:
    path = RAW / "eth_usdt_1m.parquet"
    raw = pd.read_parquet(path)
    dt = fixed_1m_timestamp(raw["timestamp"])
    df = raw.copy()
    df["dt"] = dt
    before = int(len(df))
    df = df.dropna(subset=["dt", "open", "high", "low", "close"]).sort_values("dt").drop_duplicates("dt", keep="last").reset_index(drop=True)
    diffs = df["dt"].diff().dt.total_seconds().dropna()
    audit = {
        "path": str(path),
        "rawRows": before,
        "fixedRows": int(len(df)),
        "badOrDuplicateRowsDropped": int(before - len(df)),
        "start": str(df["dt"].min()) if len(df) else None,
        "end": str(df["dt"].max()) if len(df) else None,
        "modeIntervalSeconds": float(diffs.mode().iloc[0]) if len(diffs) else None,
        "gapGt90s": int((diffs > 90).sum()) if len(diffs) else 0,
        "timestampUnitPolicy": "row-wise: >1e15 microseconds, else milliseconds",
    }
    return df, audit


def build_base_frame() -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    df, features, truth = base.build_integrated_frame()
    out = df.copy().sort_values("dt").reset_index(drop=True)
    clean = list(features)
    truth = dict(truth)
    truth["finalRowsBeforeDelayed"] = int(len(out))
    return out, clean, truth


def add_elapsed_features(base_df: pd.DataFrame, one_min: pd.DataFrame, delay_min: int) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    left = base_df.copy().sort_values("dt").reset_index(drop=True)
    m = one_min.set_index("dt").sort_index()
    starts = pd.DatetimeIndex(pd.to_datetime(left["dt"], utc=True))
    offsets = [pd.Timedelta(minutes=i) for i in range(delay_min)]
    # At decision +N, only bars with open time start..start+N-1 are visible.
    open0 = pd.to_numeric(m["open"].reindex(starts), errors="coerce").to_numpy(dtype=float)
    close_n = pd.to_numeric(m["close"].reindex(starts + offsets[-1]), errors="coerce").to_numpy(dtype=float)
    high_stack = []
    low_stack = []
    vol_stack = []
    for off in offsets:
        idx = starts + off
        high_stack.append(pd.to_numeric(m["high"].reindex(idx), errors="coerce").to_numpy(dtype=float))
        low_stack.append(pd.to_numeric(m["low"].reindex(idx), errors="coerce").to_numpy(dtype=float))
        if "volume" in m.columns:
            vol_stack.append(pd.to_numeric(m["volume"].reindex(idx), errors="coerce").to_numpy(dtype=float))
    high_arr = np.vstack(high_stack)
    low_arr = np.vstack(low_stack)
    hi = np.nanmax(high_arr, axis=0)
    lo = np.nanmin(low_arr, axis=0)
    if vol_stack:
        vol = np.nansum(np.vstack(vol_stack), axis=0)
    else:
        vol = np.zeros(len(left), dtype=float)
    observed = (~np.isnan(open0)) & (~np.isnan(close_n)) & (~np.isnan(hi)) & (~np.isnan(lo))
    if delay_min > 1:
        observed &= np.sum(~np.isnan(high_arr), axis=0) >= delay_min
        observed &= np.sum(~np.isnan(low_arr), axis=0) >= delay_min
    rng = np.maximum(hi - lo, 1e-12)
    body = close_n - open0
    prev15 = pd.to_numeric(left["ret_1"], errors="coerce").fillna(0.0).to_numpy(dtype=float) if "ret_1" in left.columns else np.zeros(len(left), dtype=float)
    feats = pd.DataFrame({
        f"de{delay_min}_ret": close_n / open0 - 1.0,
        f"de{delay_min}_body_ratio": np.abs(body) / rng,
        f"de{delay_min}_range_pct": rng / open0,
        f"de{delay_min}_close_pos": (close_n - lo) / rng,
        f"de{delay_min}_upper_wick_ratio": (hi - np.maximum(open0, close_n)) / rng,
        f"de{delay_min}_lower_wick_ratio": (np.minimum(open0, close_n) - lo) / rng,
        f"de{delay_min}_volume": vol,
        f"de{delay_min}_same_as_prev15": (np.sign(body) == np.sign(prev15)).astype(float),
    })
    feats.loc[~observed, :] = np.nan
    out = pd.concat([left, feats], axis=1)
    feature_cols = [c for c in feats.columns]
    for c in feature_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    truth = {
        "decisionDelayMin": delay_min,
        "missingElapsedRows": int((~observed).sum()),
        "elapsedFeatureCount": int(len(feature_cols)),
        "decisionFeaturePolicy": f"uses 1m bars with open_time in [market_start, market_start+{delay_min}m); no later bars",
    }
    return out, feature_cols, truth


def feature_subset(features: list[str], mode: str) -> list[str]:
    if mode == "elapsed_only":
        return [c for c in features if c.startswith("de")]
    if mode == "base15_elapsed":
        return [c for c in features if c.startswith("de") or not c.startswith(("1h_", "4h_", "daily_"))]
    if mode == "base_htf_elapsed":
        return [c for c in features if c.startswith("de") or not c.startswith("daily_")]
    if mode == "trend_elapsed":
        keys = ("ret_", "ema_", "ema_dist_", "rsi", "bb_pos", "range_", "vol_", "de")
        return [c for c in features if any(k in c for k in keys)]
    return list(features)


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


def fit_model(engine: str, train: pd.DataFrame, feats: list[str], params: dict[str, Any], random_labels: bool = False):
    if len(train) < 1000 or not feats:
        return None
    x = train[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = train["label_up"].astype(int).to_numpy()
    if random_labels:
        y = np.random.default_rng(RNG_SEED + len(train) + len(feats) + int(params.get("random_seed_offset", 0))).permutation(y)
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
            n_estimators=int(params.get("n_estimators", 100)), learning_rate=float(params.get("learning_rate", 0.035)),
            num_leaves=int(params.get("num_leaves", 16)), min_child_samples=int(params.get("min_child_samples", 80)),
            subsample=float(params.get("subsample", 0.88)), colsample_bytree=float(params.get("colsample_bytree", 0.88)),
            reg_lambda=float(params.get("reg_lambda", 1.0)), random_state=RNG_SEED, n_jobs=int(THREADS), verbose=-1,
        )
    elif engine == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=int(params.get("n_estimators", 100)), max_depth=int(params.get("depth", 3)), learning_rate=float(params.get("learning_rate", 0.035)),
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


def select_rows(val: pd.DataFrame, prob: np.ndarray, params: dict[str, Any], delay: int, confirm_mode: str) -> pd.DataFrame:
    pred_up = prob >= 0.5
    score = np.maximum(prob, 1.0 - prob)
    mask = score >= float(params.get("threshold", 0.55))
    if params.get("score_band") == "mid_only":
        mask &= score <= float(params.get("max_score", 0.76))
    if confirm_mode == "early_agreement":
        # The first N minutes must agree with the model direction. This is not the
        # live 061 direction; it is a clean delayed-entry confirmation baseline.
        support = np.sign(pd.to_numeric(val.get(f"de{delay}_ret", 0), errors="coerce").fillna(0.0).to_numpy())
        mask &= support != 0
        mask &= (support > 0) == pred_up
    out = val.loc[mask, ["dt", "label_up"]].copy().reset_index(drop=True)
    out["pred_up"] = pred_up[mask].astype(bool)
    out["score"] = score[mask]
    out["decisionDelayMin"] = delay
    out["won"] = out["pred_up"].to_numpy() == out["label_up"].astype(bool).to_numpy()
    cap = int(params.get("daily_cap", 0) or 0)
    if cap > 0 and len(out):
        out["day"] = pd.to_datetime(out["dt"], utc=True).dt.floor("D")
        out = out.sort_values(["day", "score"], ascending=[True, False]).groupby("day", sort=False).head(cap).drop(columns=["day"]).sort_values("dt").reset_index(drop=True)
    return out


def evaluate_params(frames: dict[int, tuple[pd.DataFrame, list[str]]], params: dict[str, Any], name: str) -> dict[str, Any] | None:
    delay = int(params["delay"])
    df, all_features = frames[delay]
    feats = feature_subset(all_features, params["feature_mode"])
    max_features = int(params.get("max_features", 120))
    if max_features > 0:
        # Always preserve delayed features, cap older feature pool.
        delayed = [c for c in feats if c.startswith("de")]
        other = [c for c in feats if c not in delayed]
        feats = list(dict.fromkeys(other[:max_features] + delayed))
    if len(feats) < 8:
        return None
    rows = []
    for window in ["180d", "365d"]:
        train, val = split_train_val(df, window, params["train_window"])
        model = fit_model(params["engine"], train, feats, params)
        if model is None:
            return None
        prob = predict_prob(model, val, feats)
        selected = select_rows(val, prob, params, delay, str(params.get("confirm_mode", "mainmodel")))
        if selected.empty:
            return None
        main = compound_metrics(selected, name, window, MAIN_BUY_PRICE, "delayed_entry_raw_full_fill_buy_0.50")
        pressure = {f"buy{p:.2f}": compound_metrics(selected, name, window, p, f"delayed_entry_raw_full_fill_buy_{p:.2f}") for p in PRESSURE_BUY_PRICES}
        main.update({"engine": params["engine"], "trainWindow": params["train_window"], "featureMode": params["feature_mode"], "threshold": params["threshold"], "confirmMode": params.get("confirm_mode", "mainmodel"), "featureCount": int(len(feats)), "buyPricePressure": pressure})
        rows.append(main)
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
        score += (r["winRatePct"] - b["winRatePct"]) * 60.0
        score -= max(0.0, r["maxDrawdownUsd"] - b["maxDrawdownUsd"]) / max(b["maxDrawdownUsd"], 1.0) * 500.0
        score += r.get("returnDrawdownRatio", 0.0) * 2.0
    return score


def param_grid() -> list[dict[str, Any]]:
    engines = [x for x in os.environ.get("ETH_DELAYED_ENTRY_ENGINES", "logistic,lightgbm,xgboost").split(",") if x]
    delays = [1, 2, 3]
    train_windows = ["1y", "3y", "5y", "full"]
    feature_modes = ["elapsed_only", "base15_elapsed", "base_htf_elapsed", "trend_elapsed", "wide"]
    thresholds = [0.52, 0.53, 0.54, 0.55, 0.56, 0.58, 0.60]
    confirm_modes = ["mainmodel", "early_agreement"]
    rows: list[dict[str, Any]] = []
    for engine, delay, tw, fm, th, cm in itertools.product(engines, delays, train_windows, feature_modes, thresholds, confirm_modes):
        if engine == "logistic":
            for c in [0.15, 0.35, 0.75, 1.5, 3.0]:
                rows.append({"engine": engine, "delay": delay, "train_window": tw, "feature_mode": fm, "threshold": th, "confirm_mode": cm, "C": c, "n_estimators": 1, "learning_rate": 0.04, "num_leaves": 12, "min_child_samples": 80, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0, "depth": 3, "max_features": 120})
        else:
            for ne, lr, leaves, mcs, reg, depth in [(80, 0.04, 12, 80, 1.0, 3), (120, 0.035, 20, 100, 1.2, 3), (180, 0.025, 28, 140, 2.0, 4)]:
                rows.append({"engine": engine, "delay": delay, "train_window": tw, "feature_mode": fm, "threshold": th, "confirm_mode": cm, "C": 1.0, "n_estimators": ne, "learning_rate": lr, "num_leaves": leaves, "min_child_samples": mcs, "subsample": 0.88, "colsample_bytree": 0.88, "reg_lambda": reg, "depth": depth, "max_features": 120})
    if PARAM_LIMIT > 0 and len(rows) > PARAM_LIMIT:
        rng = np.random.default_rng(RNG_SEED)
        rows = [rows[int(i)] for i in rng.permutation(len(rows))[:PARAM_LIMIT]]
    return rows


_FRAMES: dict[int, tuple[pd.DataFrame, list[str]]] | None = None


def init_worker(frames: dict[int, tuple[pd.DataFrame, list[str]]]) -> None:
    global _FRAMES
    _FRAMES = frames


def eval_worker(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    if _FRAMES is None:
        raise RuntimeError("worker not initialized")
    idx, params = item
    name = f"delayed_{params['confirm_mode']}_{params['engine']}_{params['train_window']}_{params['feature_mode']}_d{params['delay']}_thr{params['threshold']}_{stable_hash(params)}"
    try:
        out = evaluate_params(_FRAMES, params, name)
        return out if out is not None else {"name": name, "params": params, "error": "empty_or_fit_failed"}
    except Exception as exc:
        return {"name": name, "params": params, "error": repr(exc)[:1000]}


def archive_rows_for_candidate(frames: dict[int, tuple[pd.DataFrame, list[str]]], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        fill_rows, _ = archive.load_live_fill_rows()
        all_eth = archive.aggregate_logical(fill_rows, "全部归档ETH15m live去重")
        strict_mask = fill_rows["sourcePath"].str.contains("slot01|forcedtrade|base_forcedtrade|__live_eth|logs_v5_exp10_bp0480__live_eth", case=False, regex=True)
        strict = archive.aggregate_logical(fill_rows[strict_mask].copy(), "严格旧slot1/ETH相关live去重") if strict_mask.any() else all_eth.iloc[0:0]
        params = candidate["params"]
        delay = int(params["delay"])
        df, features = frames[delay]
        feats = feature_subset(features, params["feature_mode"])
        delayed = [c for c in feats if c.startswith("de")]
        other = [c for c in feats if c not in delayed]
        feats = list(dict.fromkeys(other[: int(params.get("max_features", 120))] + delayed))
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
            if train.empty or val.empty:
                continue
            model = fit_model(params["engine"], train, feats, params)
            if model is None:
                continue
            prob = predict_prob(model, val, feats)
            sel = select_rows(val, prob, params, delay, str(params.get("confirm_mode", "mainmodel")))
            pred = sel[["dt", "pred_up", "score"]].copy()
            merged = old.merge(pred, left_on="marketStart", right_on="dt", how="left")
            chosen = merged[merged["pred_up"].notna()].copy().sort_values("marketStart")
            if chosen.empty:
                rows.append({"scope": old["scopeName"].iloc[0], "name": candidate["name"], "oldRealMarkets": int(len(old)), "selectedTrades": 0})
                continue
            model_won = chosen["pred_up"].astype(bool).to_numpy() == chosen["actualUp"].astype(bool).to_numpy()
            arch_sel = pd.DataFrame({"dt": chosen["marketStart"].to_numpy(), "label_up": chosen["actualUp"].astype(bool).to_numpy(), "pred_up": chosen["pred_up"].astype(bool).to_numpy(), "won": model_won, "decisionDelayMin": delay})
            metrics = compound_metrics(arch_sel, candidate["name"], str(old["scopeName"].iloc[0]), MAIN_BUY_PRICE, "archive_raw_buy_0.50")
            rows.append({"scope": old["scopeName"].iloc[0], "name": candidate["name"], "oldRealMarkets": int(len(old)), "selectedTrades": int(len(chosen)), "skippedTrades": int(len(old) - len(chosen)), "wins": int(model_won.sum()), "losses": int(len(model_won) - int(model_won.sum())), "winRatePct": round(100.0 * float(model_won.mean()), 6), "compoundPnl": metrics["compoundPnl"], "endingBankroll": metrics["endingBankroll"], "maxDrawdownUsd": metrics["maxDrawdownUsd"], "avgScore": round(float(chosen["score"].mean()), 6), "setHash": stable_hash(chosen[["marketSlug", "pred_up", "actualUp"]].to_dict("records"))})
    except Exception as exc:
        rows.append({"scope": "archive_error", "name": candidate.get("name"), "error": repr(exc)[:800]})
    return rows


def bug_audit(base_df: pd.DataFrame, frames: dict[int, tuple[pd.DataFrame, list[str]]], one_min_truth: dict[str, Any], baseline: dict[str, dict[str, Any]]) -> dict[str, Any]:
    params = {"engine": "logistic", "delay": 2, "train_window": "3y", "feature_mode": "base15_elapsed", "threshold": 0.55, "confirm_mode": "mainmodel", "C": 0.75, "n_estimators": 1, "learning_rate": 0.04, "num_leaves": 12, "min_child_samples": 80, "subsample": 0.9, "colsample_bytree": 0.9, "reg_lambda": 1.0, "depth": 3, "max_features": 120}
    c1 = evaluate_params(frames, params, "audit_repeat")
    c2 = evaluate_params(frames, params, "audit_repeat")
    repeat = []
    if c1 and c2:
        for a, b in zip(c1["rows"], c2["rows"]):
            repeat.append({"window": a["window"], "hash1": a["setHash"], "hash2": b["setHash"], "pnl1": a["compoundPnl"], "pnl2": b["compoundPnl"], "passed": a["setHash"] == b["setHash"] and a["compoundPnl"] == b["compoundPnl"]})
    random_label = {"status": "not_run", "passed": False}
    try:
        df, features = frames[2]
        feats = feature_subset(features, "base15_elapsed")[:120]
        train, val = split_train_val(df, "365d", "3y")
        rnd_params = dict(params)
        rnd_params["threshold"] = 0.52
        total_sel = 0
        total_win = 0
        runs = []
        for seed in range(8):
            rnd_params["random_seed_offset"] = seed * 997
            model = fit_model("logistic", train, feats, rnd_params, random_labels=True)
            if model is None:
                continue
            prob = predict_prob(model, val, feats)
            sel = select_rows(val, prob, rnd_params, 2, "mainmodel")
            wins = int(sel["won"].sum()) if len(sel) else 0
            total_sel += int(len(sel))
            total_win += wins
            wr_i = float(wins * 100.0 / len(sel)) if len(sel) else 0.0
            runs.append({"seed": seed, "selectedTrades": int(len(sel)), "winRatePct": round(wr_i, 6)})
        wr = float(total_win * 100.0 / total_sel) if total_sel else 0.0
        random_label = {"status": "ok", "runs": runs, "totalSelectedTrades": total_sel, "weightedWinRatePct": round(wr, 6), "passed": total_sel >= 500 and 47.0 <= wr <= 53.0}
    except Exception as exc:
        random_label = {"status": "error", "error": repr(exc)[:300], "passed": False}
    forbidden_hits = []
    for _, features in frames.values():
        forbidden_hits += [c for c in features if c in base.base.FORBIDDEN_FEATURES or c.startswith(tuple(base.base.FORBIDDEN_PREFIXES))]
    audit = {"generatedAt": now_iso(), "beijingTime": bj_now(), "researchOnlyNoLiveChange": True, "oneMinuteDataTruth": one_min_truth, "fairMetric": "850U initial, stake 1%, buy 0.50, full fill", "decisionPolicy": "+N minute features use 1m bars in [market_start, market_start+N minutes), no later bars", "forbiddenFeatureHits": sorted(set(forbidden_hits)), "repeatability": repeat, "randomLabelAudit": random_label, "baseline061": baseline}
    audit["passed"] = one_min_truth.get("modeIntervalSeconds") == 60.0 and not audit["forbiddenFeatureHits"] and bool(repeat) and all(r["passed"] for r in repeat) and bool(random_label.get("passed"))
    write_json(OUT_AUDIT_JSON, audit)
    write_json(OUT_DATA_TRUTH, {"generatedAt": audit["generatedAt"], "oneMinuteDataTruth": one_min_truth, "frameRows": {str(k): int(len(v[0])) for k, v in frames.items()}})
    lines = ["# ETH 开盘后确认模型：回测器审计", "", f"- 北京时间：`{audit['beijingTime']}`", "- live动作：`research_only_no_live_change`", f"- 1分钟数据：`{one_min_truth}`", f"- 禁用字段命中：`{audit['forbiddenFeatureHits']}`", f"- 重复运行：`{repeat}`", f"- 随机标签测试：`{random_label}`", f"- 审计通过：`{audit['passed']}`"]
    write_text(OUT_AUDIT_MD, "\n".join(lines) + "\n")
    return audit


def summarize(results: list[dict[str, Any]], baseline: dict[str, dict[str, Any]], total: int, started: float, data_truth: dict[str, Any], audit: dict[str, Any], frames: dict[int, tuple[pd.DataFrame, list[str]]], finished: bool) -> None:
    valid = [r for r in results if r.get("rows")]
    for r in valid:
        ok, reasons = pass_gate(r, baseline)
        r["passed"] = ok
        r["reasons"] = reasons
        r["score"] = score_candidate(r, baseline)
    valid.sort(key=lambda x: (x.get("passed", False), x.get("score", -1e18)), reverse=True)
    strict = [r for r in valid if r.get("passed")]
    selected = strict[0] if strict else (valid[0] if valid else None)
    archive_rows = archive_rows_for_candidate(frames, selected) if selected else []
    confirm = [r for r in valid if r.get("params", {}).get("confirm_mode") == "early_agreement"][:100]
    main = [r for r in valid if r.get("params", {}).get("confirm_mode") == "mainmodel"][:100]
    payload = {"generatedAt": now_iso(), "beijingTime": bj_now(), "researchOnlyNoLiveChange": True, "finished": finished, "elapsedSeconds": round(time.time() - started, 3), "workers": WORKERS, "totalCandidates": total, "doneCount": len(results), "validCount": len(valid), "strictPassCount": len(strict), "dataTruth": data_truth, "audit": audit, "baseline061": baseline, "topCandidates": valid[:150], "confirmGateCandidates": confirm, "mainModelCandidates": main, "strictPass": strict[:50], "selectedForArchiveCompare": selected, "archivedRealRowsForSelected": archive_rows, "liveConfigMutated": False}
    write_json(OUT_CHECKPOINT, payload)
    write_json(OUT_CONFIRM_LEADER, {"generatedAt": payload["generatedAt"], "top": confirm, "liveConfigMutated": False})
    write_json(OUT_MAIN_LEADER, {"generatedAt": payload["generatedAt"], "top": main, "liveConfigMutated": False})
    write_json(OUT_COMPARE_JSON, payload)
    status = "candidate_beats_061" if selected and selected.get("passed") else "no_candidate_beats_061"
    write_json(OUT_VERDICT_JSON, {"generatedAt": payload["generatedAt"], "status": status, "selected": selected, "archivedRealRowsForSelected": archive_rows, "liveAction": "research_only_no_live_change"})
    render_markdown(payload)


def render_markdown(payload: dict[str, Any]) -> None:
    baseline = payload["baseline061"]
    selected = payload.get("selectedForArchiveCompare")
    lines = ["# ETH 开盘后确认模型：180/365/历史归档对比", "", f"- 北京时间：`{payload['beijingTime']}`", "- 口径：`850U初始 / 每笔当前资金1% / 买价0.50 / 满成交 / ETH原始K线方向`", "- live动作：`research_only_no_live_change`", f"- 搜索进度：`{payload['doneCount']}/{payload['totalCandidates']}`，严格通过：`{payload['strictPassCount']}`", "", "## 180天 / 365天", "", "|配置|窗口|延迟|模式|训练窗|模型|特征组|交易数|胜/负|胜率|盈亏|期末资金|最大回撤|收益回撤比|月正收益|哈希|", "|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for w in ["180d", "365d"]:
        b = baseline[w]
        lines.append(f"|当前061|{w}|0|061|-|061_cluster_gate|-|{b['trades']}|{b['wins']}/{b['losses']}|{b['winRatePct']:.2f}%|{b['compoundPnl']:,.2f}|{b['endingBankroll']:,.2f}|{b['maxDrawdownUsd']:,.2f}|{b['returnDrawdownRatio']:.2f}|{b.get('monthlyPositiveRatio', 0):.2%}|`{b['setHash']}`|")
    if selected and selected.get("rows"):
        p = selected.get("params", {})
        for r in selected["rows"]:
            lines.append(f"|{selected['name']}|{r['window']}|{r.get('decisionDelayMin','-')}|{r.get('confirmMode','-')}|{r.get('trainWindow','-')}|{r.get('engine','-')}|{r.get('featureMode','-')}|{r['trades']}|{r['wins']}/{r['losses']}|{r['winRatePct']:.2f}%|{r['compoundPnl']:,.2f}|{r['endingBankroll']:,.2f}|{r['maxDrawdownUsd']:,.2f}|{r['returnDrawdownRatio']:.2f}|{r['monthlyPositiveRatio']:.2%}|`{r['setHash']}`|")
    lines += ["", "## 买价压力", "", "|配置|窗口|买价|期末资金|盈亏|最大回撤|", "|---|---:|---:|---:|---:|---:|"]
    if selected and selected.get("rows"):
        for r in selected["rows"]:
            for key, pr in r.get("buyPricePressure", {}).items():
                lines.append(f"|{selected['name']}|{r['window']}|{pr['buyPrice']:.2f}|{pr['endingBankroll']:,.2f}|{pr['compoundPnl']:,.2f}|{pr['maxDrawdownUsd']:,.2f}|")
    lines += ["", "## 历史归档纯预测复核", "", "|范围|配置|交易数|胜/负|胜率|盈亏|期末资金|最大回撤|", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in payload.get("archivedRealRowsForSelected", []):
        if "error" in r:
            lines.append(f"|{r.get('scope')}|{r.get('name')}|错误|-|-|-|-|-|")
        else:
            lines.append(f"|{r['scope']}|{r['name']}|{r['selectedTrades']}|{r.get('wins',0)}/{r.get('losses',0)}|{r.get('winRatePct',0):.2f}%|{r.get('compoundPnl',0):,.2f}|{r.get('endingBankroll',0):,.2f}|{r.get('maxDrawdownUsd',0):,.2f}|")
    status = "通过" if selected and selected.get("passed") else "未通过"
    lines += ["", "## 结论", "", f"- 严格过门状态：`{status}`。", "- 若未通过：不改真钱061；说明开盘后确认模型仍未可靠超过061。"]
    write_text(OUT_COMPARE_MD, "\n".join(lines) + "\n")
    write_text(OUT_VERDICT_MD, "\n".join(lines) + "\n")


_FRAMES: dict[int, tuple[pd.DataFrame, list[str]]] | None = None


def init_worker(frames: dict[int, tuple[pd.DataFrame, list[str]]]) -> None:
    global _FRAMES
    _FRAMES = frames


def eval_worker(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    if _FRAMES is None:
        raise RuntimeError("worker not initialized")
    idx, params = item
    name = f"delayed_{params['confirm_mode']}_{params['engine']}_{params['train_window']}_{params['feature_mode']}_d{params['delay']}_thr{params['threshold']}_{stable_hash(params)}"
    try:
        out = evaluate_params(_FRAMES, params, name)
        return out if out is not None else {"name": name, "params": params, "error": "empty_or_fit_failed"}
    except Exception as exc:
        return {"name": name, "params": params, "error": repr(exc)[:1000]}


def run() -> int:
    started = time.time()
    print(f"[delayed-entry] start {bj_now()} workers={WORKERS} maxSeconds={MAX_SECONDS}", flush=True)
    one_min, one_truth = load_1m_fixed()
    base_df, base_features, base_truth = build_base_frame()
    frames: dict[int, tuple[pd.DataFrame, list[str]]] = {}
    delay_truth: dict[str, Any] = {}
    for delay in [1, 2, 3]:
        df, delayed_features, truth = add_elapsed_features(base_df, one_min, delay)
        features = list(dict.fromkeys(base_features + delayed_features))
        frames[delay] = (df, features)
        delay_truth[str(delay)] = truth
    data_truth = {"oneMinute": one_truth, "base": base_truth, "delays": delay_truth}
    baseline = current_061_rows()
    audit = bug_audit(base_df, frames, one_truth, baseline)
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
        name = f"delayed_{p['confirm_mode']}_{p['engine']}_{p['train_window']}_{p['feature_mode']}_d{p['delay']}_thr{p['threshold']}_{stable_hash(p)}"
        if name not in done_names:
            pending.append((i, p))
    print(f"[delayed-entry] baseRows={len(base_df)} total={total} done={len(results)} pending={len(pending)} audit={audit['passed']}", flush=True)
    summarize(results, baseline, total, started, data_truth, audit, frames, finished=False)
    last = time.time()
    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with OUT_RESULTS.open("a", encoding="utf-8") as fh:
        with cf.ProcessPoolExecutor(max_workers=WORKERS, initializer=init_worker, initargs=(frames,)) as ex:
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
                    summarize(results, baseline, total, started, data_truth, audit, frames, finished=False)
                    print(f"[delayed-entry] checkpoint {len(results)}/{total}", flush=True)
                    last = time.time()
            for fut in futures:
                fut.cancel()
    finished = len(results) >= total
    summarize(results, baseline, total, started, data_truth, audit, frames, finished=finished)
    print(json.dumps({"status": "finished" if finished else "checkpointed", "done": len(results), "total": total, "compare": str(OUT_COMPARE_MD), "verdict": str(OUT_VERDICT_MD)}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
