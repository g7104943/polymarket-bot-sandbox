#!/usr/bin/env python3
from __future__ import annotations

"""ETH 15m order-flow + meta-label main model search.

Research only. This script does not read or mutate live trading configs,
process state, ledgers, claim state, or monitor settings.

Fair metric requested by the user:
  - start bankroll 850U
  - stake 1% of current bankroll per selected trade
  - full fill at buy prices 0.50 / 0.52 / 0.55
  - model chooses UP / DOWN / NO_TRADE

Important time policy:
  - 15m features are shifted and known before each market opens.
  - 1m order-flow proxy features use available_at = minute_open + 1 minute.
  - 1h/4h/cross-asset features are merged only after their candle is closed.
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

THREADS = os.environ.get("ETH15M_ORDERFLOW_THREADS", "1")
for key in ["OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(key, THREADS)

import numpy as np
import pandas as pd

ROOT = Path("/Users/mac/polyfun")
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
SCRIPTS = ROOT / "polyfun-next" / "scripts"
BASE_SCRIPT = ROOT / "scripts" / "ops" / "run_crypto15m_1h_multasset_pressure_search_latest.py"
ARCHIVE_SCRIPT = ROOT / "scripts" / "ops" / "run_top159_all_archived_real_eth_compare.py"
CLUSTER_COMPARE_JSON = REPORTS / "top159_061_deep_v2_cluster_180_365_archive_compare_latest.json"

WORKERS = int(os.environ.get("ETH15M_ORDERFLOW_WORKERS", "4"))
MAX_SECONDS = int(os.environ.get("ETH15M_ORDERFLOW_MAX_SECONDS", str(5 * 3600)))
PARAM_LIMIT = int(os.environ.get("ETH15M_ORDERFLOW_PARAM_LIMIT", "2600"))
MAX_TRAIN_ROWS = int(os.environ.get("ETH15M_ORDERFLOW_MAX_TRAIN_ROWS", "180000"))
RNG_SEED = 20260506
START_BANKROLL = 850.0
STAKE_PCT = 0.01
BUY_PRICES = [0.50, 0.52, 0.55]

OUT_DATA_TRUTH_MD = REPORTS / "eth15m_orderflow_data_truth_latest.md"
OUT_DATA_TRUTH_JSON = REPORTS / "eth15m_orderflow_data_truth_latest.json"
OUT_LABEL_AUDIT_MD = REPORTS / "eth15m_orderflow_label_audit_latest.md"
OUT_LABEL_AUDIT_JSON = REPORTS / "eth15m_orderflow_label_audit_latest.json"
OUT_RESULTS = REPORTS / "eth15m_orderflow_metalabel_results_latest.jsonl"
OUT_LEADERBOARD_JSON = REPORTS / "eth15m_orderflow_metalabel_leaderboard_latest.json"
OUT_LEADERBOARD_MD = REPORTS / "eth15m_orderflow_metalabel_leaderboard_latest.md"
OUT_COMPARE_JSON = REPORTS / "eth15m_orderflow_061_compare_latest.json"
OUT_COMPARE_MD = REPORTS / "eth15m_orderflow_061_compare_latest.md"
OUT_PRICE_JSON = REPORTS / "eth15m_orderflow_price_pressure_latest.json"
OUT_PRICE_MD = REPORTS / "eth15m_orderflow_price_pressure_latest.md"
OUT_VERDICT_JSON = REPORTS / "eth15m_orderflow_unique_verdict_latest.json"
OUT_VERDICT_MD = REPORTS / "eth15m_orderflow_unique_verdict_latest.md"
OUT_CHECKPOINT = REPORTS / "eth15m_orderflow_metalabel_checkpoint_latest.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base = load_module("eth15m_orderflow_base", BASE_SCRIPT)
archive = load_module("eth15m_orderflow_archive", ARCHIVE_SCRIPT)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def bj_now() -> str:
    return pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S CST")


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:16]


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


def coerce_dt_from_timestamp(df: pd.DataFrame) -> pd.Series:
    if "date" in df.columns:
        d = pd.to_datetime(df["date"], utc=True, errors="coerce")
        if d.notna().mean() > 0.8:
            return d
    col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    x = pd.to_numeric(df[col], errors="coerce").astype("float64")
    # Some local 1m files have mixed millisecond/microsecond rows. Normalize row-wise
    # to seconds instead of choosing one global unit and risking out-of-bounds dates.
    ax = x.abs()
    seconds = x.copy()
    seconds = seconds.where(ax <= 1e17, x / 1e9)   # nanoseconds
    ax = seconds.abs()
    seconds = seconds.where(ax <= 1e14, seconds / 1e6)  # microseconds
    ax = seconds.abs()
    seconds = seconds.where(ax <= 1e11, seconds / 1e3)  # milliseconds
    return pd.to_datetime(seconds, unit="s", utc=True, errors="coerce")


def load_raw_ohlcv(asset: str, tf: str) -> pd.DataFrame:
    path = RAW / f"{asset.lower()}_usdt_{tf}.parquet"
    df = pd.read_parquet(path)
    df["dt"] = coerce_dt_from_timestamp(df)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["dt", "open", "high", "low", "close"]).sort_values("dt").reset_index(drop=True)


def max_drawdown(equity: np.ndarray) -> tuple[float, float]:
    if len(equity) == 0:
        return 0.0, 0.0
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    mx = float(dd.max())
    i = int(dd.argmax())
    pct = mx / float(peak[i]) if peak[i] > 1e-12 else 0.0
    return mx, pct


def monthly_positive_ratio(equity: np.ndarray, dt: pd.Series) -> float:
    if len(equity) == 0:
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
    sel = rows.sort_values("dt").reset_index(drop=True)
    won = sel["won"].astype(bool).to_numpy()
    eq = START_BANKROLL
    curve = np.empty(len(sel), dtype=float)
    win_ret = (1.0 / float(buy_price)) - 1.0
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
        "buyPrice": buy_price,
        "trades": int(len(sel)),
        "wins": wins,
        "losses": losses,
        "winRatePct": round(100.0 * wins / len(sel), 6) if len(sel) else 0.0,
        "compoundPnl": round(float(eq - START_BANKROLL), 6),
        "endingBankroll": round(float(eq), 6),
        "maxDrawdownUsd": round(mxdd, 6),
        "maxDrawdownPct": round(mxdd_pct * 100.0, 6),
        "returnDrawdownRatio": round(float(eq - START_BANKROLL) / mxdd, 6) if mxdd > 1e-12 else (999.0 if eq > START_BANKROLL else 0.0),
        "monthlyPositiveRatio": monthly_positive_ratio(curve, sel["dt"]) if len(sel) else 0.0,
        "setHash": stable_hash(sel[["dt", "pred_up15", "label_up", "won"]].to_dict("records")),
    }


def current_061_rows() -> dict[str, dict[str, Any]]:
    payload = json.loads(CLUSTER_COMPARE_JSON.read_text())
    out: dict[str, dict[str, Any]] = {}
    for r in payload.get("validationRows", []):
        if r.get("name") == "current_live_06173_cluster_gate" and r.get("method") == "full_fill_buy_0.50":
            out[r["window"]] = dict(r)
    if set(out) != {"180d", "365d"}:
        raise RuntimeError("061 baseline missing")
    return out


def prefix_merge_asof(left: pd.DataFrame, right: pd.DataFrame, cols: list[str], prefix: str, time_col: str = "dt") -> tuple[pd.DataFrame, list[str]]:
    r = right[[time_col] + cols].copy().rename(columns={time_col: "merge_dt"})
    rename = {c: f"{prefix}_{c}" for c in cols}
    r = r.rename(columns=rename)
    l = left.copy()
    # merge_asof requires identical datetime dtype precision. Normalize both sides.
    l["dt"] = pd.to_datetime(l["dt"], utc=True, errors="coerce").astype("datetime64[ns, UTC]")
    r["merge_dt"] = pd.to_datetime(r["merge_dt"], utc=True, errors="coerce").astype("datetime64[ns, UTC]")
    l = l.dropna(subset=["dt"]).sort_values("dt")
    r = r.dropna(subset=["merge_dt"]).sort_values("merge_dt")
    out = pd.merge_asof(l, r, left_on="dt", right_on="merge_dt", direction="backward").drop(columns=["merge_dt"])
    return out, list(rename.values())


def build_1m_orderflow_features(asset: str) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    raw = load_raw_ohlcv(asset, "1m")
    raw = raw.sort_values("dt").reset_index(drop=True)
    close = raw["close"].astype(float)
    high = raw["high"].astype(float)
    low = raw["low"].astype(float)
    vol = raw["volume"].astype(float)
    ret1 = close.pct_change()
    raw["available_at"] = raw["dt"] + pd.Timedelta(minutes=1)
    feats: list[str] = []
    for n in [3, 5, 15, 30, 60]:
        raw[f"of_ret_{n}m"] = close.pct_change(n)
        raw[f"of_volatility_{n}m"] = ret1.rolling(n).std()
        raw[f"of_range_{n}m"] = (high.rolling(n).max() - low.rolling(n).min()) / close
        raw[f"of_volume_sum_{n}m"] = vol.rolling(n).sum()
        raw[f"of_volume_z_{n}m"] = (vol - vol.rolling(n * 4).mean()) / (vol.rolling(n * 4).std() + 1e-12)
        # Price-volume directional proxy. We do not have true taker buy volume in long 1m history.
        signed = np.sign(ret1.fillna(0.0)) * vol
        raw[f"of_signed_volume_ratio_{n}m"] = signed.rolling(n).sum() / (vol.rolling(n).sum() + 1e-12)
        raw[f"of_up_minute_ratio_{n}m"] = (ret1 > 0).astype(float).rolling(n).mean()
        feats += [
            f"of_ret_{n}m", f"of_volatility_{n}m", f"of_range_{n}m", f"of_volume_sum_{n}m",
            f"of_volume_z_{n}m", f"of_signed_volume_ratio_{n}m", f"of_up_minute_ratio_{n}m",
        ]
    for c in feats:
        raw[c] = pd.to_numeric(raw[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
    audit = {
        "asset": asset,
        "rows": int(len(raw)),
        "start": str(raw["dt"].min()),
        "end": str(raw["dt"].max()),
        "columns": list(raw.columns),
        "trueTakerBuyAvailable": any("taker" in c.lower() for c in raw.columns),
        "note": "Long-history 1m has price/volume only; taker buy/trade count are unavailable, so signed-volume proxy is used.",
    }
    return raw[["available_at"] + feats].dropna(subset=["available_at"]).sort_values("available_at"), feats, audit


def build_cross_15m_features(asset: str) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    raw = load_raw_ohlcv(asset, "15m")
    df, feats = base.build_features(raw, "15m")
    keep = [c for c in feats if c.startswith(("ret_", "vol_", "range_", "volume_z_", "ema_dist_")) or c in {"ema_8_32", "ema_16_64", "rsi_14", "bb_pos"}]
    audit = {"asset": asset, "rows": int(len(df)), "start": str(df["dt"].min()), "end": str(df["dt"].max()), "features": len(keep)}
    return df[["dt"] + keep].copy(), keep, audit


def build_dataset() -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    eth15 = load_raw_ohlcv("eth", "15m")
    df, base_feats = base.build_features(eth15, "15m")
    feature_cols = list(base_feats)
    audit: dict[str, Any] = {
        "generatedAt": now_iso(),
        "researchOnlyNoLiveChange": True,
        "eth15mRowsRaw": int(len(eth15)),
        "eth15mRowsAfterBaseFeatures": int(len(df)),
        "featureGroups": {"base15": len(base_feats)},
        "dataSourceNote": "Main long-window order-flow features use Binance/local 1m price-volume proxy. Polymarket/orderbook short windows are not used as long-window proof.",
    }
    # ETH 1m order-flow proxy.
    eth1m, eth1m_feats, eth1m_audit = build_1m_orderflow_features("eth")
    df, added = prefix_merge_asof(df, eth1m, eth1m_feats, "eth1m", time_col="available_at")
    feature_cols += added
    audit["featureGroups"]["eth1m_orderflow_proxy"] = len(added)
    audit["eth1m"] = eth1m_audit
    # BTC 1m relative flow.
    btc1m, btc1m_feats, btc1m_audit = build_1m_orderflow_features("btc")
    df, added = prefix_merge_asof(df, btc1m, btc1m_feats, "btc1m", time_col="available_at")
    feature_cols += added
    audit["featureGroups"]["btc1m_orderflow_proxy"] = len(added)
    audit["btc1m"] = btc1m_audit
    # 1h / 4h ETH state and cross-asset 15m linkage.
    for asset, tf, prefix in [("eth", "1h", "eth1h"), ("eth", "4h", "eth4h")]:
        raw = load_raw_ohlcv(asset, tf)
        fdf, fcols = base.build_features(raw, tf)
        # Features are shifted inside build_features. Candle dt is open time; shifted features for that row are available at that row's open.
        df, added = prefix_merge_asof(df, fdf, fcols, prefix, time_col="dt")
        feature_cols += added
        audit["featureGroups"][prefix] = len(added)
    for asset in ["btc", "sol", "xrp"]:
        try:
            cdf, ccols, ca = build_cross_15m_features(asset)
            df, added = prefix_merge_asof(df, cdf, ccols, f"{asset}15", time_col="dt")
            feature_cols += added
            audit["featureGroups"][f"{asset}15_linkage"] = len(added)
            audit[f"{asset}15"] = ca
        except Exception as exc:
            audit[f"{asset}15_error"] = repr(exc)
    # Labels beyond direction.
    ret = (df["close"] / df["open"] - 1.0).astype(float)
    abs_ret = ret.abs()
    # trailing quantile shifted to avoid using current label in threshold.
    df["label_strong_move"] = (abs_ret > abs_ret.rolling(8000, min_periods=1000).quantile(0.60).shift(1)).astype(int)
    df["label_clean_up"] = ((df["label_up"].astype(int) == 1) & (df["label_strong_move"] == 1)).astype(int)
    df["label_clean_down"] = ((df["label_up"].astype(int) == 0) & (df["label_strong_move"] == 1)).astype(int)
    df["label_weak_edge"] = (df["label_strong_move"] == 0).astype(int)
    # Triple-barrier proxy from current 15m OHLC path. It is label-only, never used as feature.
    up_first = ((df["high"] / df["open"] - 1.0) >= 0.0015) & ((df["open"] / df["low"] - 1.0) < 0.0015)
    down_first = ((df["open"] / df["low"] - 1.0) >= 0.0015) & ((df["high"] / df["open"] - 1.0) < 0.0015)
    df["label_triple_barrier_up_clean"] = (up_first | ((df["label_up"] == 1) & ~down_first)).astype(int)
    df["label_triple_barrier_down_clean"] = (down_first | ((df["label_up"] == 0) & ~up_first)).astype(int)
    forbidden_exact = set(base.FORBIDDEN_FEATURES) | {"label_up", "label_strong_move", "label_clean_up", "label_clean_down", "label_weak_edge", "label_triple_barrier_up_clean", "label_triple_barrier_down_clean"}
    clean: list[str] = []
    for c in feature_cols:
        if c in forbidden_exact or c.startswith(tuple(base.FORBIDDEN_PREFIXES)):
            continue
        s = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if int(s.notna().sum()) >= max(1000, int(len(df) * 0.45)):
            df[c] = s
            clean.append(c)
    df = df.dropna(subset=["dt", "label_up"]).sort_values("dt").reset_index(drop=True)
    audit["finalRows"] = int(len(df))
    audit["featureCount"] = int(len(clean))
    audit["labelRates"] = {
        "upPct": round(100 * float(df["label_up"].mean()), 6),
        "strongMovePct": round(100 * float(df["label_strong_move"].mean()), 6),
        "weakEdgePct": round(100 * float(df["label_weak_edge"].mean()), 6),
    }
    return df, clean, audit


def feature_subset(features: list[str], mode: str) -> list[str]:
    if mode == "base15":
        return [c for c in features if not c.startswith(("eth1m_", "btc1m_", "eth1h_", "eth4h_", "btc15_", "sol15_", "xrp15_"))]
    if mode == "orderflow_only":
        return [c for c in features if c.startswith(("eth1m_", "btc1m_"))]
    if mode == "base_orderflow":
        return [c for c in features if not c.startswith(("eth1h_", "eth4h_", "btc15_", "sol15_", "xrp15_"))]
    if mode == "base_orderflow_htf":
        return [c for c in features if not c.startswith(("btc15_", "sol15_", "xrp15_"))]
    if mode == "wide":
        return list(features)
    if mode == "linkage_wide":
        return [c for c in features if c.startswith(("ret_", "vol_", "range_", "volume_z_", "ema_", "rsi", "bb", "eth1m_", "btc1m_", "btc15_", "sol15_", "xrp15_", "eth1h_", "eth4h_"))]
    return list(features)


def split_train_val(df: pd.DataFrame, window: str, train_window: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    end = df["dt"].max()
    days = 180 if window == "180d" else 365
    start = end - pd.Timedelta(days=days)
    val = df[df["dt"] >= start].copy()
    if train_window == "full":
        train = df[df["dt"] < start].copy()
    else:
        td = {"1y": 365, "3y": 1095, "5y": 1825}[train_window]
        train = df[(df["dt"] < start) & (df["dt"] >= start - pd.Timedelta(days=td))].copy()
    train = train[train["dt"] < start - pd.Timedelta(hours=4)].copy()
    if MAX_TRAIN_ROWS > 0 and len(train) > MAX_TRAIN_ROWS:
        train = train.sort_values("dt").iloc[-MAX_TRAIN_ROWS:].copy()
    return train, val


def fit_model(engine: str, train: pd.DataFrame, feats: list[str], label: str, params: dict[str, Any], random_labels: bool = False):
    if len(train) < 1500 or len(feats) < 4:
        return None
    x = train[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = train[label].astype(int).to_numpy()
    if random_labels:
        y = np.random.default_rng(RNG_SEED + len(train) + len(feats)).permutation(y)
    if len(np.unique(y)) < 2:
        return None
    if engine == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=700, C=float(params.get("C", 1.0)), random_state=RNG_SEED))
    elif engine == "lightgbm":
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=int(params.get("n_estimators", 120)), learning_rate=float(params.get("learning_rate", 0.035)),
            num_leaves=int(params.get("num_leaves", 24)), min_child_samples=int(params.get("min_child_samples", 90)),
            subsample=float(params.get("subsample", 0.88)), colsample_bytree=float(params.get("colsample_bytree", 0.88)),
            reg_lambda=float(params.get("reg_lambda", 1.0)), random_state=RNG_SEED, n_jobs=int(THREADS), verbose=-1,
        )
    elif engine == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=int(params.get("n_estimators", 120)), max_depth=int(params.get("depth", 3)), learning_rate=float(params.get("learning_rate", 0.035)),
            subsample=float(params.get("subsample", 0.88)), colsample_bytree=float(params.get("colsample_bytree", 0.88)),
            reg_lambda=float(params.get("reg_lambda", 1.0)), random_state=RNG_SEED, n_jobs=int(THREADS), eval_metric="logloss", verbosity=0,
        )
    elif engine == "catboost":
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            iterations=int(params.get("n_estimators", 120)), depth=int(params.get("depth", 3)), learning_rate=float(params.get("learning_rate", 0.035)),
            l2_leaf_reg=float(params.get("reg_lambda", 1.0)), loss_function="Logloss", random_seed=RNG_SEED, thread_count=int(THREADS), verbose=False,
        )
    else:
        raise ValueError(engine)
    model.fit(x, y)
    return model


def predict_prob(model: Any, val: pd.DataFrame, feats: list[str]) -> np.ndarray:
    x = val[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def select_rows(val: pd.DataFrame, p_up: np.ndarray, p_quality: np.ndarray, params: dict[str, Any]) -> pd.DataFrame:
    pred_up = p_up >= 0.5
    dir_conf = np.maximum(p_up, 1.0 - p_up)
    weight = float(params.get("quality_weight", 0.0))
    score = dir_conf + weight * (p_quality - 0.5)
    mask = score >= float(params["threshold"])
    band = params.get("score_band", "all")
    if band == "mid_only":
        mask &= score <= float(params.get("max_score", 0.80))
    cap = int(params.get("daily_cap", 0) or 0)
    selected = val.loc[mask, ["dt", "label_up"]].copy()
    selected["pred_up15"] = pred_up[mask].astype(bool)
    selected["score15"] = score[mask]
    selected["dir_conf"] = dir_conf[mask]
    selected["quality_prob"] = p_quality[mask]
    if cap > 0 and not selected.empty:
        selected["day"] = pd.to_datetime(selected["dt"], utc=True).dt.floor("D")
        selected = selected.sort_values(["day", "score15"], ascending=[True, False]).groupby("day", sort=False).head(cap).sort_values("dt").drop(columns=["day"])
    selected = selected.reset_index(drop=True)
    selected["won"] = selected["pred_up15"].to_numpy() == selected["label_up"].astype(bool).to_numpy()
    return selected


def evaluate_params(df: pd.DataFrame, features: list[str], params: dict[str, Any]) -> dict[str, Any] | None:
    feats = feature_subset(features, params["feature_mode"])
    max_features = int(params.get("max_features", 0) or 0)
    if max_features > 0:
        feats = feats[:max_features]
    if len(feats) < 8:
        return None
    name = f"orderflow_meta_{params['engine']}_{params['train_window']}_{params['feature_mode']}_thr{params['threshold']}_qw{params.get('quality_weight',0)}_{stable_hash(params)}"
    rows: list[dict[str, Any]] = []
    price_rows: list[dict[str, Any]] = []
    for window in ["180d", "365d"]:
        train, val = split_train_val(df, window, params["train_window"])
        dir_model = fit_model(params["engine"], train, feats, "label_up", params)
        q_model = fit_model(params["quality_engine"], train, feats, params.get("quality_label", "label_strong_move"), params)
        if dir_model is None or q_model is None:
            return None
        p_up = predict_prob(dir_model, val, feats)
        p_q = predict_prob(q_model, val, feats)
        selected = select_rows(val, p_up, p_q, params)
        if selected.empty:
            return None
        for bp in BUY_PRICES:
            pr = curve_metrics(selected, name, window, bp)
            pr.update({
                "engine": params["engine"], "qualityEngine": params["quality_engine"], "trainWindow": params["train_window"],
                "featureMode": params["feature_mode"], "threshold": params["threshold"], "qualityWeight": params.get("quality_weight", 0),
                "qualityLabel": params.get("quality_label", "label_strong_move"), "dailyCap": int(params.get("daily_cap", 0) or 0), "featureCount": len(feats),
            })
            price_rows.append(pr)
            if bp == 0.50:
                rows.append(pr)
    return {"name": name, "params": params, "rows": rows, "priceRows": price_rows, "featureCount": len(feats)}


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
    try:
        out = evaluate_params(_DF, _FEATURES, params)
        return out if out else {"params": params, "error": "empty_or_fit_failed"}
    except Exception as exc:
        return {"params": params, "error": repr(exc)[:800]}


def param_grid() -> list[dict[str, Any]]:
    engines = [x for x in os.environ.get("ETH15M_ORDERFLOW_ENGINES", "logistic,lightgbm,xgboost").split(",") if x]
    quality_engines = [x for x in os.environ.get("ETH15M_ORDERFLOW_QUALITY_ENGINES", "logistic,lightgbm").split(",") if x]
    rows: list[dict[str, Any]] = []
    for engine, qeng, tw, fm, thr, qw, qlabel, band, cap, mf in itertools.product(
        engines,
        quality_engines,
        ["1y", "3y", "5y", "full"],
        ["base15", "orderflow_only", "base_orderflow", "base_orderflow_htf", "linkage_wide", "wide"],
        [0.52, 0.535, 0.55, 0.565, 0.58, 0.60, 0.62, 0.64, 0.66],
        [0.0, 0.15, 0.30, 0.50, 0.75],
        ["label_strong_move", "label_triple_barrier_up_clean"],
        [("all", 1.0), ("mid_only", 0.76), ("mid_only", 0.82)],
        [0, 48, 32, 24],
        [80, 140, 220],
    ):
        score_band, max_score = band
        if engine == "logistic" and qeng == "logistic":
            for C in [0.2, 0.5, 1.0, 2.0]:
                rows.append({"engine": engine, "quality_engine": qeng, "train_window": tw, "feature_mode": fm, "threshold": thr, "quality_weight": qw, "quality_label": qlabel, "score_band": score_band, "max_score": max_score, "daily_cap": cap, "max_features": mf, "C": C, "n_estimators": 100, "learning_rate": 0.035, "num_leaves": 24, "min_child_samples": 90, "subsample": 0.88, "colsample_bytree": 0.88, "reg_lambda": 1.0, "depth": 3})
        else:
            for ne, lr, leaves, mcs, reg, depth in [(80, 0.035, 16, 100, 1.0, 3), (140, 0.025, 24, 120, 1.5, 4), (220, 0.018, 32, 160, 2.0, 4)]:
                rows.append({"engine": engine, "quality_engine": qeng, "train_window": tw, "feature_mode": fm, "threshold": thr, "quality_weight": qw, "quality_label": qlabel, "score_band": score_band, "max_score": max_score, "daily_cap": cap, "max_features": mf, "C": 1.0, "n_estimators": ne, "learning_rate": lr, "num_leaves": leaves, "min_child_samples": mcs, "subsample": 0.88, "colsample_bytree": 0.88, "reg_lambda": reg, "depth": depth})
    if PARAM_LIMIT and len(rows) > PARAM_LIMIT:
        rng = np.random.default_rng(RNG_SEED)
        idx = rng.permutation(len(rows))[:PARAM_LIMIT]
        rows = [rows[int(i)] for i in idx]
    return rows


def baseline_pass(candidate: dict[str, Any], baseline: dict[str, dict[str, Any]]) -> tuple[bool, list[str]]:
    by = {r["window"]: r for r in candidate.get("rows", []) if r.get("buyPrice") == 0.50}
    reasons = []
    for w in ["180d", "365d"]:
        if w not in by:
            reasons.append(f"{w}_missing")
            continue
        r, b = by[w], baseline[w]
        if r["compoundPnl"] <= b["compoundPnl"]:
            reasons.append(f"{w}_pnl_not_above_061")
        if w == "365d" and r["winRatePct"] < b["winRatePct"]:
            reasons.append("365d_winrate_below_061")
        if r["maxDrawdownUsd"] > b["maxDrawdownUsd"] * 1.05:
            reasons.append(f"{w}_drawdown_too_high")
        if r["trades"] < max(100 if w == "180d" else 200, int(b["trades"] * 0.45)):
            reasons.append(f"{w}_too_few_trades")
    price_by = {(r["window"], r["buyPrice"]): r for r in candidate.get("priceRows", [])}
    for w in ["180d", "365d"]:
        if price_by.get((w, 0.52), {}).get("compoundPnl", -1e9) <= 0:
            reasons.append(f"{w}_buy052_not_alive")
    return not reasons, reasons


def score_candidate(candidate: dict[str, Any], baseline: dict[str, dict[str, Any]]) -> float:
    by = {r["window"]: r for r in candidate.get("rows", []) if r.get("buyPrice") == 0.50}
    if set(by) != {"180d", "365d"}:
        return -1e18
    return (
        (by["180d"]["compoundPnl"] - baseline["180d"]["compoundPnl"]) / max(1.0, baseline["180d"]["compoundPnl"]) * 2000
        + (by["365d"]["compoundPnl"] - baseline["365d"]["compoundPnl"]) / max(1.0, baseline["365d"]["compoundPnl"]) * 5000
        + (by["180d"]["winRatePct"] - baseline["180d"]["winRatePct"]) * 20
        + (by["365d"]["winRatePct"] - baseline["365d"]["winRatePct"]) * 30
        - by["365d"]["maxDrawdownUsd"] / max(1.0, baseline["365d"]["maxDrawdownUsd"]) * 20
    )


def archive_rows_for_candidate(df: pd.DataFrame, features: list[str], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not candidate or not candidate.get("params"):
        return rows
    try:
        fill_rows, _ = archive.load_live_fill_rows()
        scopes = []
        all_eth = archive.aggregate_logical(fill_rows, "全部归档ETH15m live去重")
        strict_mask = fill_rows["sourcePath"].str.contains("slot01|forcedtrade|base_forcedtrade|__live_eth|logs_v5_exp10_bp0480__live_eth", case=False, regex=True)
        strict = archive.aggregate_logical(fill_rows[strict_mask].copy(), "严格旧slot1/ETH相关live去重") if strict_mask.any() else all_eth.iloc[0:0]
        scopes = [strict, all_eth]
        params = candidate["params"]
        feats = feature_subset(features, params["feature_mode"])[: int(params.get("max_features", 0) or 999999)]
        for old in scopes:
            if old.empty:
                continue
            start, end = old["marketStart"].min(), old["marketStart"].max()
            train = df[df["dt"] < start - pd.Timedelta(hours=4)].copy()
            if params["train_window"] != "full":
                days = {"1y": 365, "3y": 1095, "5y": 1825}[params["train_window"]]
                train = train[train["dt"] >= start - pd.Timedelta(days=days)].copy()
            val = df[(df["dt"] >= start) & (df["dt"] <= end)].copy()
            dm = fit_model(params["engine"], train, feats, "label_up", params)
            qm = fit_model(params["quality_engine"], train, feats, params.get("quality_label", "label_strong_move"), params)
            if dm is None or qm is None:
                continue
            sel = select_rows(val, predict_prob(dm, val, feats), predict_prob(qm, val, feats), params)
            pred = sel[["dt", "pred_up15", "score15"]].copy()
            merged = old.merge(pred, left_on="marketStart", right_on="dt", how="left")
            chosen = merged[merged["pred_up15"].notna()].copy().sort_values("marketStart")
            if chosen.empty:
                rows.append({"scope": old["scopeName"].iloc[0], "selectedTrades": 0, "oldRealMarkets": int(len(old))})
                continue
            won = chosen["pred_up15"].astype(bool).to_numpy() == chosen["actualUp"].astype(bool).to_numpy()
            arch_sel = pd.DataFrame({"dt": chosen["marketStart"], "label_up": chosen["actualUp"].astype(bool), "pred_up15": chosen["pred_up15"].astype(bool), "won": won})
            m = curve_metrics(arch_sel, candidate["name"], str(old["scopeName"].iloc[0]), 0.50)
            rows.append({"scope": old["scopeName"].iloc[0], "oldRealMarkets": int(len(old)), "selectedTrades": int(len(chosen)), "wins": int(won.sum()), "losses": int(len(won) - int(won.sum())), "winRatePct": round(100 * float(won.mean()), 6), "compoundPnl": m["compoundPnl"], "endingBankroll": m["endingBankroll"], "maxDrawdownUsd": m["maxDrawdownUsd"], "setHash": stable_hash(chosen[["marketSlug", "pred_up15", "actualUp"]].to_dict("records"))})
    except Exception as exc:
        rows.append({"scope": "archive_error", "error": repr(exc)[:600]})
    return rows


def run_audit(df: pd.DataFrame, features: list[str], baseline: dict[str, dict[str, Any]]) -> dict[str, Any]:
    params = {"engine": "logistic", "quality_engine": "logistic", "train_window": "3y", "feature_mode": "base_orderflow", "threshold": 0.56, "quality_weight": 0.3, "quality_label": "label_strong_move", "score_band": "all", "max_score": 1.0, "daily_cap": 0, "max_features": 140, "C": 1.0, "n_estimators": 100, "learning_rate": 0.035, "num_leaves": 24, "min_child_samples": 90, "subsample": 0.88, "colsample_bytree": 0.88, "reg_lambda": 1.0, "depth": 3}
    c1 = evaluate_params(df, features, params)
    c2 = evaluate_params(df, features, params)
    repeat = []
    if c1 and c2:
        for a, b in zip(c1["rows"], c2["rows"]):
            repeat.append({"window": a["window"], "hash1": a["setHash"], "hash2": b["setHash"], "pnl1": a["compoundPnl"], "pnl2": b["compoundPnl"], "passed": a["setHash"] == b["setHash"] and a["compoundPnl"] == b["compoundPnl"]})
    random_label = {"status": "not_run", "passed": False}
    try:
        feats = feature_subset(features, params["feature_mode"])[: int(params["max_features"])]
        train, val = split_train_val(df, "365d", params["train_window"])
        dm = fit_model(params["engine"], train, feats, "label_up", params, random_labels=True)
        qm = fit_model(params["quality_engine"], train, feats, params["quality_label"], params, random_labels=True)
        if dm and qm:
            sel = select_rows(val, predict_prob(dm, val, feats), predict_prob(qm, val, feats), params)
            wr = float(sel["won"].mean() * 100.0) if len(sel) else 0.0
            random_label = {"status": "ok", "selectedTrades": int(len(sel)), "winRatePct": round(wr, 6), "passed": len(sel) < 50 or 43.0 <= wr <= 57.0}
    except Exception as exc:
        random_label = {"status": "error", "error": repr(exc)[:500], "passed": False}
    forbidden_hits = [c for c in features if c.startswith(tuple(base.FORBIDDEN_PREFIXES)) or c in base.FORBIDDEN_FEATURES]
    audit = {"generatedAt": now_iso(), "beijingTime": bj_now(), "researchOnlyNoLiveChange": True, "timePolicy": "closed_candle_available_at_v2 plus 1m available_at=open+1min", "forbiddenFeatureHits": forbidden_hits, "repeatability": repeat, "randomLabelAudit": random_label, "passed": not forbidden_hits and bool(repeat) and all(r["passed"] for r in repeat) and bool(random_label.get("passed")), "baseline061": baseline}
    write_json(OUT_LABEL_AUDIT_JSON, audit)
    write_text(OUT_LABEL_AUDIT_MD, "\n".join(["# ETH15m 订单流元标签：标签与回测审计", "", f"- 北京时间：`{audit['beijingTime']}`", f"- 时间口径：`{audit['timePolicy']}`", f"- 禁用字段命中：`{forbidden_hits}`", f"- 重复运行：`{repeat}`", f"- 随机标签：`{random_label}`", f"- 审计通过：`{audit['passed']}`"]) + "\n")
    return audit


def render_outputs(payload: dict[str, Any]) -> None:
    baseline = payload["baseline061"]
    top = payload.get("topCandidates", [])
    selected = payload.get("selected")
    lines = ["# ETH15m 订单流元标签模型：061公平对比", "", f"- 北京时间：`{payload['beijingTime']}`", "- 口径：`850U初始 / 每笔1% / 满成交 / 买价0.50`", "- live动作：`无，研究只读`", "", "## 180天 / 365天主表", "", "|配置|窗口|交易数|胜/负|胜率|盈亏|期末资金|最大回撤|收益回撤比|月正收益|买价|哈希|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|" ]
    for w in ["180d", "365d"]:
        b = baseline[w]
        lines.append(f"|当前061|{w}|{b['trades']}|{b['wins']}/{b['losses']}|{b['winRatePct']:.2f}%|{b['compoundPnl']:,.2f}|{b['endingBankroll']:,.2f}|{b['maxDrawdownUsd']:,.2f}|{b['returnDrawdownRatio']:.2f}|{b.get('monthlyPositiveRatio',0):.2%}|0.50|`{b['setHash']}`|")
    if selected:
        for r in selected.get("rows", []):
            lines.append(f"|订单流元标签最佳|{r['window']}|{r['trades']}|{r['wins']}/{r['losses']}|{r['winRatePct']:.2f}%|{r['compoundPnl']:,.2f}|{r['endingBankroll']:,.2f}|{r['maxDrawdownUsd']:,.2f}|{r['returnDrawdownRatio']:.2f}|{r['monthlyPositiveRatio']:.2%}|{r['buyPrice']:.2f}|`{r['setHash']}`|")
    lines += ["", "## 前10候选（买价0.50主口径）", "", "|排名|配置|180天盈亏|180天胜率|365天盈亏|365天胜率|365天回撤|通过061门槛|失败原因|", "|---:|---|---:|---:|---:|---:|---:|---|---|"]
    for i, c in enumerate(top[:10], 1):
        by = {r["window"]: r for r in c.get("rows", [])}
        reasons = ",".join(c.get("reasons", []))
        lines.append(f"|{i}|`{c['name']}`|{by.get('180d',{}).get('compoundPnl',0):,.2f}|{by.get('180d',{}).get('winRatePct',0):.2f}%|{by.get('365d',{}).get('compoundPnl',0):,.2f}|{by.get('365d',{}).get('winRatePct',0):.2f}%|{by.get('365d',{}).get('maxDrawdownUsd',0):,.2f}|{c.get('passed', False)}|{reasons}|")
    write_text(OUT_COMPARE_MD, "\n".join(lines) + "\n")

    p_lines = ["# ETH15m 订单流元标签：买价压力表", "", "|配置|窗口|买价|交易数|胜率|盈亏|期末资金|最大回撤|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    if selected:
        for r in selected.get("priceRows", []):
            p_lines.append(f"|订单流元标签最佳|{r['window']}|{r['buyPrice']:.2f}|{r['trades']}|{r['winRatePct']:.2f}%|{r['compoundPnl']:,.2f}|{r['endingBankroll']:,.2f}|{r['maxDrawdownUsd']:,.2f}|")
    write_text(OUT_PRICE_MD, "\n".join(p_lines) + "\n")

    verdict = payload.get("verdict", {})
    v_lines = ["# ETH15m 订单流元标签：唯一结论", "", f"- 状态：`{verdict.get('status')}`", f"- 结论：{verdict.get('message')}", "", "## 历史归档纯预测复核", "", "|范围|旧真实市场数|新模型选中|胜/负|胜率|盈亏|最大回撤|", "|---|---:|---:|---:|---:|---:|---:|" ]
    for r in payload.get("archiveRows", []):
        v_lines.append(f"|{r.get('scope')}|{r.get('oldRealMarkets',0)}|{r.get('selectedTrades',0)}|{r.get('wins',0)}/{r.get('losses',0)}|{r.get('winRatePct',0):.2f}%|{r.get('compoundPnl',0):,.2f}|{r.get('maxDrawdownUsd',0):,.2f}|")
    write_text(OUT_VERDICT_MD, "\n".join(v_lines) + "\n")


def write_progress(results: list[dict[str, Any]], baseline: dict[str, dict[str, Any]], total: int, start_ts: float, df: pd.DataFrame, features: list[str], data_truth: dict[str, Any], label_audit: dict[str, Any], finished: bool = False) -> None:
    valid = [r for r in results if r.get("rows")]
    rows = []
    for c in valid:
        ok, reasons = baseline_pass(c, baseline)
        rows.append({**c, "passed": ok, "reasons": reasons, "score": score_candidate(c, baseline)})
    rows.sort(key=lambda x: x["score"], reverse=True)
    strict = [r for r in rows if r.get("passed")]
    selected = strict[0] if strict else (rows[0] if rows else None)
    archive_rows = archive_rows_for_candidate(df, features, selected) if selected else []
    status = "candidate_beats_061" if selected and selected.get("passed") else "no_candidate_beats_061"
    msg = "找到通过061门槛的订单流元标签候选，只能进入影子验证，不改真钱。" if status == "candidate_beats_061" else "没有候选在180/365和买价0.52压力下同时打穿061。ETH15m方向模型仍以061为真钱基线。"
    payload = {"generatedAt": now_iso(), "beijingTime": bj_now(), "researchOnlyNoLiveChange": True, "finished": finished, "elapsedSeconds": round(time.time() - start_ts, 3), "workers": WORKERS, "done": len(results), "total": total, "valid": len(valid), "strictPass": len(strict), "dataTruth": data_truth, "labelAudit": label_audit, "baseline061": baseline, "topCandidates": rows[:200], "strictPassCandidates": strict[:50], "selected": selected, "archiveRows": archive_rows, "verdict": {"status": status, "message": msg}}
    write_json(OUT_CHECKPOINT, payload)
    write_json(OUT_LEADERBOARD_JSON, payload)
    write_json(OUT_COMPARE_JSON, payload)
    write_json(OUT_PRICE_JSON, payload)
    write_json(OUT_VERDICT_JSON, payload["verdict"] | {"selected": selected, "archiveRows": archive_rows, "generatedAt": payload["generatedAt"]})
    render_outputs(payload)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    baseline = current_061_rows()
    df, features, data_truth = build_dataset()
    write_json(OUT_DATA_TRUTH_JSON, data_truth)
    write_text(OUT_DATA_TRUTH_MD, "\n".join(["# ETH15m 订单流数据真相", "", f"- 北京时间：`{bj_now()}`", f"- 最终样本：`{data_truth['finalRows']}`", f"- 特征数：`{data_truth['featureCount']}`", f"- 1分钟数据说明：{data_truth['eth1m']['note']}", f"- ETH 1m 覆盖：`{data_truth['eth1m']['start']} -> {data_truth['eth1m']['end']}`", f"- BTC 1m 覆盖：`{data_truth['btc1m']['start']} -> {data_truth['btc1m']['end']}`", f"- 标签比例：`{data_truth['labelRates']}`", "", "本脚本只做研究，不改真实交易。"] ) + "\n")
    label_audit = run_audit(df, features, baseline)
    params = param_grid()
    if OUT_RESULTS.exists():
        OUT_RESULTS.unlink()
    results: list[dict[str, Any]] = []
    last_write = time.time()
    with cf.ProcessPoolExecutor(max_workers=WORKERS, initializer=init_worker, initargs=(df, features)) as ex:
        futs = {ex.submit(eval_worker, (i, p)): i for i, p in enumerate(params)}
        for fut in cf.as_completed(futs):
            results.append(fut.result())
            with OUT_RESULTS.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(results[-1], ensure_ascii=False, default=str) + "\n")
            if time.time() - last_write > 300:
                write_progress(results, baseline, len(params), started, df, features, data_truth, label_audit, finished=False)
                last_write = time.time()
            if time.time() - started > MAX_SECONDS:
                break
    write_progress(results, baseline, len(params), started, df, features, data_truth, label_audit, finished=True)


if __name__ == "__main__":
    main()
