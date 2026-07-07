#!/usr/bin/env python3
from __future__ import annotations

"""ETH15m next-10 strategy search at buy price 0.55.

Research only. This script does not mutate live 061/top159 configs, ledgers,
order state, claim state, monitor state, or any live process.

Locked evaluation:
  - 850U initial bankroll
  - 1% of current bankroll per selected market
  - buy price 0.55 primary, 0.60 pressure
  - full fill, ignore fill toxicity/盘口
  - one logical trade per ETH 15m market

Time policy:
  - imported dataset uses closed-candle/available data policy from audited
    orderflow script.
  - extra labels are label-only and never used as features.
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

THREADS = os.environ.get("ETH15M_NEXT10_THREADS", "1")
for key in ["OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(key, THREADS)
os.environ.setdefault("ETH15M_ORDERFLOW_THREADS", THREADS)
os.environ.setdefault("ETH15M_ORDERFLOW_MAX_TRAIN_ROWS", os.environ.get("ETH15M_NEXT10_MAX_TRAIN_ROWS", "120000"))

import numpy as np
import pandas as pd

ROOT = Path("/Users/mac/polyfun")
REPORTS = ROOT / "reports"
ORDERFLOW_SCRIPT = ROOT / "polyfun-next" / "scripts" / "run_eth15m_orderflow_metalabel_search.py"
ARCHIVE_SCRIPT = ROOT / "scripts" / "ops" / "run_top159_all_archived_real_eth_compare.py"

START_BANKROLL = 850.0
STAKE_PCT = 0.01
PRIMARY_BUY_PRICE = 0.55
BUY_PRICES = [0.55, 0.60]
RNG_SEED = 20260507
WORKERS = int(os.environ.get("ETH15M_NEXT10_WORKERS", "4"))
MAX_SECONDS = int(os.environ.get("ETH15M_NEXT10_MAX_SECONDS", str(4 * 3600)))
PARAM_LIMIT = int(os.environ.get("ETH15M_NEXT10_PARAM_LIMIT", "420"))

OUT_AUDIT_MD = REPORTS / "eth15m_next10_strategy_bug_audit_latest.md"
OUT_AUDIT_JSON = REPORTS / "eth15m_next10_strategy_bug_audit_latest.json"
OUT_RESULTS = REPORTS / "eth15m_next10_strategy_results_latest.jsonl"
OUT_LEADERBOARD_MD = REPORTS / "eth15m_next10_strategy_leaderboard_latest.md"
OUT_LEADERBOARD_JSON = REPORTS / "eth15m_next10_strategy_leaderboard_latest.json"
OUT_COMPARE_MD = REPORTS / "eth15m_next10_strategy_061_compare_latest.md"
OUT_COMPARE_JSON = REPORTS / "eth15m_next10_strategy_061_compare_latest.json"
OUT_VERDICT_MD = REPORTS / "eth15m_next10_strategy_unique_verdict_latest.md"
OUT_VERDICT_JSON = REPORTS / "eth15m_next10_strategy_unique_verdict_latest.json"
OUT_DATA_MD = REPORTS / "eth15m_next10_strategy_data_truth_latest.md"
OUT_DATA_JSON = REPORTS / "eth15m_next10_strategy_data_truth_latest.json"
OUT_CHECKPOINT = REPORTS / "eth15m_next10_strategy_checkpoint_latest.json"

BASELINE_061 = {
    "180d": {"trades": 3942, "wins": 2324, "losses": 1618, "winRatePct": 58.954845, "compoundPnl": 11494.591625, "endingBankroll": 12344.591625, "maxDrawdownUsd": 1481.713373, "returnDrawdownRatio": 7.757635, "monthlyPositiveRatio": 1.0, "setHash": "b35313c05e5b66d2"},
    "365d": {"trades": 9152, "wins": 5246, "losses": 3906, "winRatePct": 57.320804, "compoundPnl": 27033.917055, "endingBankroll": 27883.917055, "maxDrawdownUsd": 3499.248929, "returnDrawdownRatio": 7.725634, "monthlyPositiveRatio": 0.846154, "setHash": "ced1cf82642d8f0d"},
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


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


of = load_module("next10_orderflow_base", ORDERFLOW_SCRIPT)
archive = load_module("next10_archive", ARCHIVE_SCRIPT)
of.MAX_TRAIN_ROWS = int(os.environ.get("ETH15M_NEXT10_MAX_TRAIN_ROWS", "120000"))


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
        "monthlyPositiveRatio": monthly_positive_ratio(curve, sel["dt"]) if len(sel) else 0.0,
        "setHash": stable_hash(sel[["dt", "pred_up15", "label_up", "won"]].to_dict("records")) if len(sel) else "empty",
    }


def feature_subset(features: list[str], mode: str) -> list[str]:
    base = of.feature_subset(features, mode if mode in {"base15", "orderflow_only", "base_orderflow", "base_orderflow_htf", "linkage_wide", "wide"} else "wide")
    if mode == "htf_only":
        return [c for c in features if c.startswith(("eth1h_", "eth4h_"))]
    if mode == "cross_only":
        return [c for c in features if c.startswith(("btc15_", "sol15_", "xrp15_", "btc1m_"))]
    if mode == "volatility":
        return [c for c in features if any(s in c.lower() for s in ["vol", "range", "atr", "bb", "rsi", "body"])]
    if mode == "sequence_proxy":
        return [c for c in features if c.startswith(("eth1m_", "btc1m_")) or any(s in c.lower() for s in ["ret", "volume", "range"])]
    return base


def build_dataset() -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    df, features, audit = of.build_dataset()
    df = df.sort_values("dt").reset_index(drop=True).copy()
    # Label-only multi-horizon targets. They are not features.
    future_close_30 = df["close"].shift(-1)
    future_close_60 = df["close"].shift(-3)
    df["label_up30"] = (future_close_30 > df["open"]).astype(int)
    df["label_up60"] = (future_close_60 > df["open"]).astype(int)
    # Safe present-time regime helpers are features; they only depend on already built shifted features.
    numeric = df[features].select_dtypes(include=[np.number]) if features else pd.DataFrame(index=df.index)
    if not numeric.empty:
        # Use broad proxies without inventing future fields.
        vol_cols = [c for c in features if any(s in c.lower() for s in ["range", "vol", "atr", "bb_width"])]
        if vol_cols:
            v = pd.to_numeric(df[vol_cols[0]], errors="coerce").fillna(0.0)
            df["regime_vol_rank"] = v.rolling(2000, min_periods=200).rank(pct=True).shift(1).fillna(0.5)
            features.append("regime_vol_rank")
        htf_cols = [c for c in features if c.startswith("eth1h_") and "ret" in c.lower()]
        if htf_cols:
            h = pd.to_numeric(df[htf_cols[0]], errors="coerce").fillna(0.0)
            df["regime_1h_up"] = (h > 0).astype(float)
            features.append("regime_1h_up")
    df = df.dropna(subset=["dt", "label_up", "label_up30", "label_up60"]).reset_index(drop=True)
    audit["next10ExtraLabels"] = {"label_up30_pct": round(100 * float(df["label_up30"].mean()), 6), "label_up60_pct": round(100 * float(df["label_up60"].mean()), 6)}
    audit["next10FeatureCount"] = len(features)
    return df, features, audit


def split_train_val(df: pd.DataFrame, window: str, train_window: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return of.split_train_val(df, window, train_window)


def fit_model(engine: str, train: pd.DataFrame, feats: list[str], label: str, params: dict[str, Any], random_labels: bool = False):
    return of.fit_model(engine, train, feats, label, params, random_labels=random_labels)


def predict_prob(model: Any, val: pd.DataFrame, feats: list[str]) -> np.ndarray:
    return of.predict_prob(model, val, feats)


def safe_fit_predict(engine: str, train: pd.DataFrame, val: pd.DataFrame, feats: list[str], label: str, params: dict[str, Any], random_labels: bool = False) -> np.ndarray | None:
    m = fit_model(engine, train, feats, label, params, random_labels=random_labels)
    if m is None:
        return None
    return predict_prob(m, val, feats)


def apply_daily_cap(selected: pd.DataFrame, cap: int) -> pd.DataFrame:
    if cap <= 0 or selected.empty:
        return selected.sort_values("dt").reset_index(drop=True)
    s = selected.copy()
    s["day"] = pd.to_datetime(s["dt"], utc=True).dt.floor("D")
    s = s.sort_values(["day", "score15"], ascending=[True, False]).groupby("day", sort=False).head(cap)
    return s.sort_values("dt").drop(columns=["day"]).reset_index(drop=True)


def selected_from_scores(val: pd.DataFrame, pred_up: np.ndarray, score: np.ndarray, threshold: float, params: dict[str, Any]) -> pd.DataFrame:
    mask = np.asarray(score) >= threshold
    if params.get("score_band") == "mid_only":
        mask &= np.asarray(score) <= float(params.get("max_score", 0.82))
    if params.get("vol_bucket") in {"low", "mid", "high"} and "regime_vol_rank" in val.columns:
        vr = pd.to_numeric(val["regime_vol_rank"], errors="coerce").fillna(0.5).to_numpy()
        if params["vol_bucket"] == "low":
            mask &= vr < 0.33
        elif params["vol_bucket"] == "mid":
            mask &= (vr >= 0.33) & (vr < 0.66)
        else:
            mask &= vr >= 0.66
    selected = val.loc[mask, ["dt", "label_up"]].copy()
    selected["pred_up15"] = np.asarray(pred_up)[mask].astype(bool)
    selected["score15"] = np.asarray(score)[mask]
    selected["won"] = selected["pred_up15"].to_numpy() == selected["label_up"].astype(bool).to_numpy()
    return apply_daily_cap(selected.reset_index(drop=True), int(params.get("daily_cap", 0) or 0))


def eval_strategy(df: pd.DataFrame, features: list[str], params: dict[str, Any], random_labels: bool = False) -> dict[str, Any] | None:
    feats = feature_subset(features, params.get("feature_mode", "wide"))[: int(params.get("max_features", 160) or 160)]
    if len(feats) < 6:
        return None
    name = f"next10_{params['family']}_{params['engine']}_{params['train_window']}_{params.get('feature_mode','wide')}_thr{params['threshold']}_{stable_hash(params)}"
    price_rows: list[dict[str, Any]] = []
    primary_rows: list[dict[str, Any]] = []
    for window in ["180d", "365d"]:
        train, val = split_train_val(df, window, params["train_window"])
        if len(train) < 2000 or len(val) < 200:
            return None
        p15 = safe_fit_predict(params["engine"], train, val, feats, "label_up", params, random_labels=random_labels)
        if p15 is None:
            return None
        pred = p15 >= 0.5
        conf = np.maximum(p15, 1.0 - p15)
        score = conf.copy()
        fam = params["family"]
        if fam == "multi_objective_compound":
            pq = safe_fit_predict(params.get("quality_engine", params["engine"]), train, val, feats, params.get("quality_label", "label_strong_move"), params, random_labels=random_labels)
            if pq is None: return None
            score = conf + float(params.get("quality_weight", 0.25)) * (pq - 0.5)
        elif fam == "confidence_bucket":
            # Use calibrated confidence band; this is intentionally simple and auditable.
            score = conf
        elif fam == "multi_horizon_consensus":
            p30 = safe_fit_predict(params["engine"], train, val, feats, "label_up30", params, random_labels=random_labels)
            p60 = safe_fit_predict(params["engine"], train, val, feats, "label_up60", params, random_labels=random_labels)
            if p30 is None or p60 is None: return None
            d30, d60 = p30 >= 0.5, p60 >= 0.5
            agree = (pred == d30) & (pred == d60)
            score = (conf + np.maximum(p30, 1-p30) + np.maximum(p60, 1-p60)) / 3.0
            score = np.where(agree, score, 0.0)
        elif fam == "cross_lag_lead":
            score = conf
        elif fam == "orderflow_proxy":
            pq = safe_fit_predict(params.get("quality_engine", "logistic"), train, val, feats, "label_strong_move", params, random_labels=random_labels)
            if pq is None: return None
            score = conf + float(params.get("quality_weight", 0.35)) * (pq - 0.5)
        elif fam == "volatility_regime":
            score = conf
        elif fam == "reversal_continuation_dual":
            # Two-expert proxy: require 1h state support for continuation or let high confidence handle reversal.
            score = conf.copy()
            if "regime_1h_up" in val.columns:
                h_up = pd.to_numeric(val["regime_1h_up"], errors="coerce").fillna(0.5).to_numpy() > 0.5
                same = (pred & h_up) | ((~pred) & (~h_up))
                score = np.where(same, score + float(params.get("same_bonus", 0.015)), score - float(params.get("opp_penalty", 0.015)))
        elif fam == "purged_metalabel":
            # Train a quality model on clean directional labels matching selected direction.
            p_clean_up = safe_fit_predict(params.get("quality_engine", "logistic"), train, val, feats, "label_clean_up", params, random_labels=random_labels)
            p_clean_down = safe_fit_predict(params.get("quality_engine", "logistic"), train, val, feats, "label_clean_down", params, random_labels=random_labels)
            if p_clean_up is None or p_clean_down is None: return None
            pq = np.where(pred, p_clean_up, p_clean_down)
            score = conf + float(params.get("quality_weight", 0.4)) * (pq - 0.5)
        elif fam == "dynamic_threshold":
            score = conf.copy()
            if "regime_vol_rank" in val.columns:
                vr = pd.to_numeric(val["regime_vol_rank"], errors="coerce").fillna(0.5).to_numpy()
                score = score - np.where(vr > 0.66, float(params.get("high_vol_penalty", 0.015)), 0.0)
                score = score + np.where(vr < 0.33, float(params.get("low_vol_bonus", 0.005)), 0.0)
        elif fam == "conservative_vote_ensemble":
            ps: list[np.ndarray] = [p15]
            for eng in ["logistic", "lightgbm"]:
                if eng == params["engine"]:
                    continue
                p = safe_fit_predict(eng, train, val, feats, "label_up", {**params, "engine": eng}, random_labels=random_labels)
                if p is not None:
                    ps.append(p)
            if len(ps) < 2:
                return None
            dirs = [(p >= 0.5) for p in ps]
            up_votes = np.sum(np.vstack(dirs), axis=0)
            pred = up_votes >= (len(ps) / 2.0)
            agree_ratio = np.maximum(up_votes, len(ps)-up_votes) / len(ps)
            avgp = np.mean(np.vstack(ps), axis=0)
            score = np.maximum(avgp, 1-avgp) * agree_ratio
        else:
            return None
        selected = selected_from_scores(val, pred, score, float(params["threshold"]), params)
        if selected.empty:
            return None
        for bp in BUY_PRICES:
            r = curve_metrics(selected, name, window, bp)
            r.update({"strategyFamily": fam, "engine": params["engine"], "trainWindow": params["train_window"], "featureMode": params.get("feature_mode", "wide"), "threshold": params["threshold"], "featureCount": len(feats)})
            price_rows.append(r)
            if bp == PRIMARY_BUY_PRICE:
                primary_rows.append(r)
    return {"name": name, "params": params, "rows": primary_rows, "priceRows": price_rows, "featureCount": len(feats)}


_DF: pd.DataFrame | None = None
_FEATURES: list[str] | None = None


def init_worker(df: pd.DataFrame, features: list[str]) -> None:
    global _DF, _FEATURES
    _DF, _FEATURES = df, features


def eval_worker(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    if _DF is None or _FEATURES is None:
        raise RuntimeError("worker not initialized")
    idx, params = item
    try:
        out = eval_strategy(_DF, _FEATURES, params)
        return out if out else {"params": params, "error": "empty_or_fit_failed"}
    except Exception as exc:
        return {"params": params, "error": repr(exc)[:800]}


def strategy_grid() -> list[dict[str, Any]]:
    engines = [x for x in os.environ.get("ETH15M_NEXT10_ENGINES", "logistic,lightgbm").split(",") if x]
    train_windows = ["1y", "3y", "5y", "full"]
    rows: list[dict[str, Any]] = []
    common_tree = {"n_estimators": 100, "learning_rate": 0.035, "num_leaves": 24, "min_child_samples": 100, "subsample": 0.88, "colsample_bytree": 0.88, "reg_lambda": 1.2, "depth": 3, "C": 1.0}
    # 1 多目标复利主模型
    for eng in engines:
        for tw in train_windows:
            for fm in ["base_orderflow_htf", "wide", "linkage_wide"]:
                for thr in [0.55, 0.565, 0.58, 0.60, 0.62]:
                    for qw in [0.15, 0.35, 0.55]:
                        rows.append({**common_tree, "family": "multi_objective_compound", "engine": eng, "quality_engine": "logistic", "train_window": tw, "feature_mode": fm, "threshold": thr, "quality_weight": qw, "quality_label": "label_strong_move", "daily_cap": 0, "max_features": 180})
    # 2 高置信分桶
    for eng in engines:
        for tw in train_windows:
            for thr in [0.565, 0.58, 0.60, 0.62, 0.64, 0.66]:
                rows.append({**common_tree, "family": "confidence_bucket", "engine": eng, "train_window": tw, "feature_mode": "base_orderflow_htf", "threshold": thr, "score_band": "all", "daily_cap": 0, "max_features": 160})
                rows.append({**common_tree, "family": "confidence_bucket", "engine": eng, "train_window": tw, "feature_mode": "wide", "threshold": thr, "score_band": "mid_only", "max_score": 0.82, "daily_cap": 48, "max_features": 220})
    # 3 短中长一致
    for eng in engines:
        for tw in ["3y", "5y", "full"]:
            for thr in [0.535, 0.55, 0.565, 0.58, 0.60]:
                rows.append({**common_tree, "family": "multi_horizon_consensus", "engine": eng, "train_window": tw, "feature_mode": "base_orderflow_htf", "threshold": thr, "daily_cap": 0, "max_features": 180})
    # 4 跨币领先滞后
    for eng in engines:
        for tw in train_windows:
            for thr in [0.54, 0.555, 0.57, 0.59, 0.61]:
                rows.append({**common_tree, "family": "cross_lag_lead", "engine": eng, "train_window": tw, "feature_mode": "linkage_wide", "threshold": thr, "daily_cap": 0, "max_features": 180})
    # 5 订单流代理
    for eng in engines:
        for tw in train_windows:
            for thr in [0.54, 0.555, 0.57, 0.59, 0.61]:
                for qw in [0.25, 0.45, 0.65]:
                    rows.append({**common_tree, "family": "orderflow_proxy", "engine": eng, "quality_engine": "logistic", "train_window": tw, "feature_mode": "base_orderflow", "threshold": thr, "quality_weight": qw, "daily_cap": 0, "max_features": 170})
    # 6 波动状态专用
    for eng in engines:
        for tw in train_windows:
            for bucket in ["low", "mid", "high"]:
                for thr in [0.535, 0.55, 0.565, 0.58, 0.60]:
                    rows.append({**common_tree, "family": "volatility_regime", "engine": eng, "train_window": tw, "feature_mode": "wide", "threshold": thr, "vol_bucket": bucket, "daily_cap": 0, "max_features": 200})
    # 7 反转/延续双专家
    for eng in engines:
        for tw in train_windows:
            for thr in [0.54, 0.555, 0.57, 0.585, 0.60]:
                for pen in [0.0, 0.015, 0.03]:
                    rows.append({**common_tree, "family": "reversal_continuation_dual", "engine": eng, "train_window": tw, "feature_mode": "base_orderflow_htf", "threshold": thr, "same_bonus": 0.01, "opp_penalty": pen, "daily_cap": 0, "max_features": 180})
    # 8 净化元标签
    for eng in engines:
        for tw in train_windows:
            for thr in [0.535, 0.55, 0.565, 0.58, 0.60]:
                for qw in [0.25, 0.45, 0.65]:
                    rows.append({**common_tree, "family": "purged_metalabel", "engine": eng, "quality_engine": "logistic", "train_window": tw, "feature_mode": "base_orderflow_htf", "threshold": thr, "quality_weight": qw, "daily_cap": 0, "max_features": 180})
    # 9 动态阈值
    for eng in engines:
        for tw in train_windows:
            for thr in [0.54, 0.555, 0.57, 0.585, 0.60]:
                for hp in [0.0, 0.015, 0.03]:
                    rows.append({**common_tree, "family": "dynamic_threshold", "engine": eng, "train_window": tw, "feature_mode": "wide", "threshold": thr, "high_vol_penalty": hp, "low_vol_bonus": 0.005, "daily_cap": 0, "max_features": 220})
    # 10 保守投票集成
    for eng in engines:
        for tw in ["3y", "5y", "full"]:
            for fm in ["base_orderflow_htf", "wide"]:
                for thr in [0.535, 0.55, 0.565, 0.58, 0.60]:
                    rows.append({**common_tree, "family": "conservative_vote_ensemble", "engine": eng, "train_window": tw, "feature_mode": fm, "threshold": thr, "daily_cap": 0, "max_features": 200})
    if PARAM_LIMIT and len(rows) > PARAM_LIMIT:
        rng = np.random.default_rng(RNG_SEED)
        # Keep at least several from each family, then random-fill.
        keep: list[dict[str, Any]] = []
        for fam in sorted({r["family"] for r in rows}):
            fam_rows = [r for r in rows if r["family"] == fam]
            idx = rng.permutation(len(fam_rows))[: max(8, PARAM_LIMIT // 20)]
            keep.extend([fam_rows[int(i)] for i in idx])
        remaining = [r for r in rows if r not in keep]
        need = max(0, PARAM_LIMIT - len(keep))
        if need:
            idx = rng.permutation(len(remaining))[:need]
            keep.extend([remaining[int(i)] for i in idx])
        rows = keep[:PARAM_LIMIT]
    return rows


def pass_gate(candidate: dict[str, Any]) -> tuple[bool, list[str]]:
    by = {r["window"]: r for r in candidate.get("rows", []) if r.get("buyPrice") == PRIMARY_BUY_PRICE}
    reasons: list[str] = []
    for w in ["180d", "365d"]:
        if w not in by:
            reasons.append(f"{w}_missing")
            continue
        r = by[w]
        b = BASELINE_061[w]
        if r["compoundPnl"] <= b["compoundPnl"]:
            reasons.append(f"{w}_pnl_not_above_061")
        if w == "365d" and r["winRatePct"] < b["winRatePct"]:
            reasons.append("365d_winrate_below_061")
        if r["maxDrawdownUsd"] > b["maxDrawdownUsd"] * 1.05:
            reasons.append(f"{w}_drawdown_too_high")
        if r["trades"] < (100 if w == "180d" else 200):
            reasons.append(f"{w}_too_few_trades")
    p60 = {(r["window"], r["buyPrice"]): r for r in candidate.get("priceRows", [])}
    for w in ["180d", "365d"]:
        if p60.get((w, 0.60), {}).get("compoundPnl", -1e9) <= 0:
            reasons.append(f"{w}_buy060_not_alive")
    return not reasons, reasons


def candidate_score(c: dict[str, Any]) -> float:
    by = {r["window"]: r for r in c.get("rows", [])}
    if set(by) != {"180d", "365d"}:
        return -1e18
    b180, b365 = BASELINE_061["180d"], BASELINE_061["365d"]
    return (
        (by["180d"]["compoundPnl"] - b180["compoundPnl"]) / max(1.0, b180["compoundPnl"]) * 2500
        + (by["365d"]["compoundPnl"] - b365["compoundPnl"]) / max(1.0, b365["compoundPnl"]) * 5000
        + (by["365d"]["winRatePct"] - b365["winRatePct"]) * 35
        - by["365d"]["maxDrawdownUsd"] / max(1.0, b365["maxDrawdownUsd"]) * 25
    )


def archive_rows_for_candidate(df: pd.DataFrame, features: list[str], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    if not candidate or not candidate.get("params"):
        return []
    rows: list[dict[str, Any]] = []
    try:
        fill_rows, _ = archive.load_live_fill_rows()
        all_eth = archive.aggregate_logical(fill_rows, "全部归档ETH15m live去重")
        strict_mask = fill_rows["sourcePath"].str.contains("slot01|forcedtrade|base_forcedtrade|__live_eth|logs_v5_exp10_bp0480__live_eth", case=False, regex=True)
        strict = archive.aggregate_logical(fill_rows[strict_mask].copy(), "严格旧slot1/ETH相关live去重") if strict_mask.any() else all_eth.iloc[0:0]
        params = candidate["params"]
        feats = feature_subset(features, params.get("feature_mode", "wide"))[: int(params.get("max_features", 160) or 160)]
        for old in [strict, all_eth]:
            if old.empty:
                continue
            start, end = old["marketStart"].min(), old["marketStart"].max()
            train = df[df["dt"] < start - pd.Timedelta(hours=4)].copy()
            if params["train_window"] != "full":
                days = {"1y": 365, "3y": 1095, "5y": 1825}[params["train_window"]]
                train = train[train["dt"] >= start - pd.Timedelta(days=days)].copy()
            val = df[(df["dt"] >= start) & (df["dt"] <= end)].copy()
            pred = eval_strategy_window(train, val, feats, params)
            if pred is None or pred.empty:
                rows.append({"scope": old["scopeName"].iloc[0], "oldRealMarkets": int(len(old)), "selectedTrades": 0})
                continue
            merged = old.merge(pred[["dt", "pred_up15", "score15"]], left_on="marketStart", right_on="dt", how="left")
            chosen = merged[merged["pred_up15"].notna()].copy().sort_values("marketStart")
            if chosen.empty:
                rows.append({"scope": old["scopeName"].iloc[0], "oldRealMarkets": int(len(old)), "selectedTrades": 0})
                continue
            won = chosen["pred_up15"].astype(bool).to_numpy() == chosen["actualUp"].astype(bool).to_numpy()
            arch_sel = pd.DataFrame({"dt": chosen["marketStart"], "label_up": chosen["actualUp"].astype(bool), "pred_up15": chosen["pred_up15"].astype(bool), "won": won})
            m = curve_metrics(arch_sel, candidate["name"], str(old["scopeName"].iloc[0]), PRIMARY_BUY_PRICE)
            rows.append({"scope": old["scopeName"].iloc[0], "oldRealMarkets": int(len(old)), "selectedTrades": int(len(chosen)), "wins": int(won.sum()), "losses": int(len(won)-int(won.sum())), "winRatePct": round(100 * float(won.mean()), 6), "compoundPnl": m["compoundPnl"], "endingBankroll": m["endingBankroll"], "maxDrawdownUsd": m["maxDrawdownUsd"], "setHash": stable_hash(chosen[["marketSlug", "pred_up15", "actualUp"]].to_dict("records"))})
    except Exception as exc:
        rows.append({"scope": "archive_error", "error": repr(exc)[:600]})
    return rows


def eval_strategy_window(train: pd.DataFrame, val: pd.DataFrame, feats: list[str], params: dict[str, Any]) -> pd.DataFrame | None:
    # Same core as eval_strategy for one arbitrary validation window, for archive replay.
    p15 = safe_fit_predict(params["engine"], train, val, feats, "label_up", params)
    if p15 is None:
        return None
    pred = p15 >= 0.5
    conf = np.maximum(p15, 1.0 - p15)
    score = conf.copy()
    fam = params["family"]
    if fam in {"multi_objective_compound", "orderflow_proxy"}:
        pq = safe_fit_predict(params.get("quality_engine", params["engine"]), train, val, feats, params.get("quality_label", "label_strong_move"), params)
        if pq is None: return None
        score = conf + float(params.get("quality_weight", 0.3)) * (pq - 0.5)
    elif fam == "multi_horizon_consensus":
        p30 = safe_fit_predict(params["engine"], train, val, feats, "label_up30", params)
        p60 = safe_fit_predict(params["engine"], train, val, feats, "label_up60", params)
        if p30 is None or p60 is None: return None
        d30, d60 = p30 >= 0.5, p60 >= 0.5
        score = np.where((pred == d30) & (pred == d60), (conf + np.maximum(p30,1-p30) + np.maximum(p60,1-p60))/3.0, 0.0)
    elif fam == "reversal_continuation_dual" and "regime_1h_up" in val.columns:
        h_up = pd.to_numeric(val["regime_1h_up"], errors="coerce").fillna(0.5).to_numpy() > 0.5
        same = (pred & h_up) | ((~pred) & (~h_up))
        score = np.where(same, score + float(params.get("same_bonus", 0.015)), score - float(params.get("opp_penalty", 0.015)))
    elif fam == "purged_metalabel":
        pu = safe_fit_predict(params.get("quality_engine", "logistic"), train, val, feats, "label_clean_up", params)
        pdn = safe_fit_predict(params.get("quality_engine", "logistic"), train, val, feats, "label_clean_down", params)
        if pu is None or pdn is None: return None
        pq = np.where(pred, pu, pdn)
        score = conf + float(params.get("quality_weight", 0.4)) * (pq - 0.5)
    elif fam == "dynamic_threshold" and "regime_vol_rank" in val.columns:
        vr = pd.to_numeric(val["regime_vol_rank"], errors="coerce").fillna(0.5).to_numpy()
        score = score - np.where(vr > 0.66, float(params.get("high_vol_penalty", 0.015)), 0.0)
    elif fam == "conservative_vote_ensemble":
        ps = [p15]
        for eng in ["logistic", "lightgbm"]:
            if eng == params["engine"]: continue
            p = safe_fit_predict(eng, train, val, feats, "label_up", {**params, "engine": eng})
            if p is not None: ps.append(p)
        if len(ps) >= 2:
            dirs = [(p >= 0.5) for p in ps]
            up_votes = np.sum(np.vstack(dirs), axis=0)
            pred = up_votes >= (len(ps)/2.0)
            avgp = np.mean(np.vstack(ps), axis=0)
            score = np.maximum(avgp, 1-avgp) * (np.maximum(up_votes, len(ps)-up_votes) / len(ps))
    return selected_from_scores(val, pred, score, float(params["threshold"]), params)


def run_audit(df: pd.DataFrame, features: list[str], params: list[dict[str, Any]]) -> dict[str, Any]:
    sample = next((p for p in params if p["family"] == "multi_objective_compound"), params[0])
    repeat = []
    c1 = eval_strategy(df, features, sample)
    c2 = eval_strategy(df, features, sample)
    if c1 and c2:
        for a, b in zip(c1["rows"], c2["rows"]):
            repeat.append({"window": a["window"], "hash1": a["setHash"], "hash2": b["setHash"], "pnl1": a["compoundPnl"], "pnl2": b["compoundPnl"], "passed": a["setHash"] == b["setHash"] and a["compoundPnl"] == b["compoundPnl"]})
    random_label = {"status": "not_run", "passed": False}
    try:
        feats = feature_subset(features, sample.get("feature_mode", "wide"))[: int(sample.get("max_features", 160))]
        train, val = split_train_val(df, "365d", sample["train_window"])
        rand = {**sample, "threshold": max(0.52, float(sample["threshold"])-0.02)}
        p = safe_fit_predict(rand["engine"], train, val, feats, "label_up", rand, random_labels=True)
        if p is not None:
            pred = p >= 0.5
            conf = np.maximum(p, 1-p)
            sel = selected_from_scores(val, pred, conf, float(rand["threshold"]), rand)
            wr = float(sel["won"].mean()*100) if len(sel) else 0.0
            random_label = {"status": "ok", "selectedTrades": int(len(sel)), "winRatePct": round(wr, 6), "passed": len(sel) < 50 or 43.0 <= wr <= 57.0}
    except Exception as exc:
        random_label = {"status": "error", "error": repr(exc)[:500], "passed": False}
    forbidden = [c for c in features if c.startswith(tuple(of.base.FORBIDDEN_PREFIXES)) or c in of.base.FORBIDDEN_FEATURES or c.startswith("label_")]
    audit = {"generatedAt": now_iso(), "beijingTime": bj_now(), "researchOnlyNoLiveChange": True, "timePolicy": "closed_candle_available_at_v2 + 1m available_at=open+1min", "forbiddenFeatureHits": forbidden, "repeatability": repeat, "randomLabelAudit": random_label, "passed": not forbidden and bool(repeat) and all(r["passed"] for r in repeat) and bool(random_label.get("passed"))}
    write_json(OUT_AUDIT_JSON, audit)
    write_text(OUT_AUDIT_MD, "\n".join(["# ETH15m 十方案统一验证：审计", "", f"- 北京时间：`{audit['beijingTime']}`", f"- 时间口径：`{audit['timePolicy']}`", f"- 禁用字段命中：`{forbidden}`", f"- 重复运行：`{repeat}`", f"- 随机标签：`{random_label}`", f"- 审计通过：`{audit['passed']}`"]) + "\n")
    return audit


def render(payload: dict[str, Any]) -> None:
    top = payload.get("topCandidates", [])
    selected = payload.get("selected")
    lines = ["# ETH15m 十方案统一验证：061公平对比", "", f"- 北京时间：`{payload['beijingTime']}`", "- 口径：`850U初始 / 每笔1% / 满成交 / 买价0.55`", "- live动作：`无，研究只读`", "", "## 061基线 vs 最强候选", "", "|配置|窗口|交易数|胜/负|胜率|盈亏|期末资金|最大回撤|收益回撤比|月正收益|哈希|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for w in ["180d", "365d"]:
        b = BASELINE_061[w]
        lines.append(f"|当前061|{w}|{b['trades']}|{b['wins']}/{b['losses']}|{b['winRatePct']:.2f}%|{b['compoundPnl']:,.2f}|{b['endingBankroll']:,.2f}|{b['maxDrawdownUsd']:,.2f}|{b['returnDrawdownRatio']:.2f}|{b['monthlyPositiveRatio']:.2%}|`{b['setHash']}`|")
    if selected:
        for r in selected.get("rows", []):
            lines.append(f"|十方案最佳|{r['window']}|{r['trades']}|{r['wins']}/{r['losses']}|{r['winRatePct']:.2f}%|{r['compoundPnl']:,.2f}|{r['endingBankroll']:,.2f}|{r['maxDrawdownUsd']:,.2f}|{r['returnDrawdownRatio']:.2f}|{r['monthlyPositiveRatio']:.2%}|`{r['setHash']}`|")
    lines += ["", "## 前30候选", "", "|排名|方案|模型|训练窗|特征组|180盈亏|180胜率|365盈亏|365胜率|365回撤|通过|失败原因|", "|---:|---|---|---|---|---:|---:|---:|---:|---:|---|---|"]
    for i, c in enumerate(top[:30], 1):
        by = {r["window"]: r for r in c.get("rows", [])}
        p = c.get("params", {})
        lines.append(f"|{i}|{p.get('family')}|{p.get('engine')}|{p.get('train_window')}|{p.get('feature_mode')}|{by.get('180d',{}).get('compoundPnl',0):,.2f}|{by.get('180d',{}).get('winRatePct',0):.2f}%|{by.get('365d',{}).get('compoundPnl',0):,.2f}|{by.get('365d',{}).get('winRatePct',0):.2f}%|{by.get('365d',{}).get('maxDrawdownUsd',0):,.2f}|{c.get('passed', False)}|{','.join(c.get('reasons', []))}|")
    write_text(OUT_COMPARE_MD, "\n".join(lines) + "\n")
    write_text(OUT_LEADERBOARD_MD, "\n".join(lines) + "\n")

    v = payload.get("verdict", {})
    vlines = ["# ETH15m 十方案统一验证：唯一结论", "", f"- 状态：`{v.get('status')}`", f"- 结论：{v.get('message')}", "", "## 历史归档真实单纯预测复核", "", "|范围|旧真实市场数|选中|胜/负|胜率|盈亏|最大回撤|", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in payload.get("archiveRows", []):
        vlines.append(f"|{r.get('scope')}|{r.get('oldRealMarkets',0)}|{r.get('selectedTrades',0)}|{r.get('wins',0)}/{r.get('losses',0)}|{r.get('winRatePct',0):.2f}%|{r.get('compoundPnl',0):,.2f}|{r.get('maxDrawdownUsd',0):,.2f}|")
    write_text(OUT_VERDICT_MD, "\n".join(vlines) + "\n")


def write_progress(results: list[dict[str, Any]], total: int, started: float, df: pd.DataFrame, features: list[str], data_truth: dict[str, Any], audit: dict[str, Any], finished: bool) -> None:
    valid = [r for r in results if r.get("rows")]
    ranked = []
    for c in valid:
        ok, reasons = pass_gate(c)
        ranked.append({**c, "passed": ok, "reasons": reasons, "score": candidate_score(c)})
    ranked.sort(key=lambda x: x["score"], reverse=True)
    strict = [r for r in ranked if r.get("passed")]
    selected = strict[0] if strict else (ranked[0] if ranked else None)
    archive_rows = archive_rows_for_candidate(df, features, selected) if selected else []
    archive_bad = False
    for r in archive_rows:
        if r.get("selectedTrades", 0) >= 30 and r.get("winRatePct", 100) < 53.0:
            archive_bad = True
    status = "candidate_beats_061" if selected and selected.get("passed") and not archive_bad else "no_candidate_beats_061"
    msg = "找到十方案中可进入24小时影子验证的候选；本轮不改真钱。" if status == "candidate_beats_061" else "十个方案没有找到同时打败061、0.60压力不崩且归档不差的候选；当前真钱继续保留061。"
    payload = {"generatedAt": now_iso(), "beijingTime": bj_now(), "researchOnlyNoLiveChange": True, "finished": finished, "elapsedSeconds": round(time.time()-started, 3), "workers": WORKERS, "done": len(results), "total": total, "valid": len(valid), "strictPass": len(strict), "dataTruth": data_truth, "audit": audit, "baseline061": BASELINE_061, "topCandidates": ranked[:300], "selected": selected, "archiveRows": archive_rows, "verdict": {"status": status, "message": msg}}
    write_json(OUT_CHECKPOINT, payload)
    write_json(OUT_LEADERBOARD_JSON, payload)
    write_json(OUT_COMPARE_JSON, payload)
    write_json(OUT_VERDICT_JSON, payload["verdict"] | {"selected": selected, "archiveRows": archive_rows})
    render(payload)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    df, features, data_truth = build_dataset()
    params = strategy_grid()
    data_payload = {"generatedAt": now_iso(), "beijingTime": bj_now(), "researchOnlyNoLiveChange": True, "rows": int(len(df)), "featureCount": int(len(features)), "sourceAudit": data_truth, "families": sorted({p["family"] for p in params}), "paramCount": len(params), "baseline061": BASELINE_061}
    write_json(OUT_DATA_JSON, data_payload)
    write_text(OUT_DATA_MD, "\n".join(["# ETH15m 十方案统一验证：数据真相", "", f"- 北京时间：`{data_payload['beijingTime']}`", f"- 样本数：`{data_payload['rows']}`", f"- 特征数：`{data_payload['featureCount']}`", f"- 候选参数数：`{data_payload['paramCount']}`", f"- 方案族：`{data_payload['families']}`", "", "本脚本只做研究，不改真实交易。"])+"\n")
    audit = run_audit(df, features, params)
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
                write_progress(results, len(params), started, df, features, data_truth, audit, finished=False)
                last_write = time.time()
            if time.time() - started > MAX_SECONDS:
                break
    write_progress(results, len(params), started, df, features, data_truth, audit, finished=True)


if __name__ == "__main__":
    main()
