#!/usr/bin/env python3
from __future__ import annotations

"""ETH 15m high-price 0.65 main-model search.

Research only. This script does not read or mutate live trading configs,
process state, ledgers, claim state, or monitor settings.

User-requested fair metric:
  - start bankroll 850U
  - stake 1% of current bankroll per selected trade
  - primary buy price 0.65, auxiliary 0.60 / 0.70
  - full fill, ignore winner/loser fill toxicity
  - model chooses UP / DOWN / NO_TRADE

Important math:
  - At buy price 0.65, a win earns (1/0.65 - 1) = 53.846% of stake.
  - A loss loses 100% of stake.
  - Break-even win rate is exactly 65% before other costs.

Time policy:
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

THREADS = os.environ.get("ETH15M_HIGHPRICE_THREADS", "1")
# Make the imported orderflow module obey this script's thread budget.
os.environ.setdefault("ETH15M_ORDERFLOW_THREADS", THREADS)
for key in ["OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(key, THREADS)

import numpy as np
import pandas as pd

ROOT = Path("/Users/mac/polyfun")
REPORTS = ROOT / "reports"
ORDERFLOW_SCRIPT = ROOT / "polyfun-next" / "scripts" / "run_eth15m_orderflow_metalabel_search.py"
ARCHIVE_SCRIPT = ROOT / "scripts" / "ops" / "run_top159_all_archived_real_eth_compare.py"

WORKERS = int(os.environ.get("ETH15M_HIGHPRICE_WORKERS", "4"))
MAX_SECONDS = int(os.environ.get("ETH15M_HIGHPRICE_MAX_SECONDS", str(4 * 3600)))
PARAM_LIMIT = int(os.environ.get("ETH15M_HIGHPRICE_PARAM_LIMIT", "900"))
MAX_TRAIN_ROWS = int(os.environ.get("ETH15M_HIGHPRICE_MAX_TRAIN_ROWS", os.environ.get("ETH15M_ORDERFLOW_MAX_TRAIN_ROWS", "180000")))
RNG_SEED = 20260506
START_BANKROLL = 850.0
STAKE_PCT = 0.01
PRIMARY_BUY_PRICE = 0.65
BUY_PRICES = [0.65, 0.60, 0.70]
MAX_ALLOWED_DRAWDOWN_USD = START_BANKROLL * 0.25
TARGET_DRAWDOWN_USD = START_BANKROLL * 0.15

OUT_DATA_TRUTH_MD = REPORTS / "eth15m_highprice065_data_truth_latest.md"
OUT_DATA_TRUTH_JSON = REPORTS / "eth15m_highprice065_data_truth_latest.json"
OUT_LABEL_AUDIT_MD = REPORTS / "eth15m_highprice065_label_audit_latest.md"
OUT_LABEL_AUDIT_JSON = REPORTS / "eth15m_highprice065_label_audit_latest.json"
OUT_RESULTS = REPORTS / "eth15m_highprice065_results_latest.jsonl"
OUT_LEADERBOARD_JSON = REPORTS / "eth15m_highprice065_leaderboard_latest.json"
OUT_LEADERBOARD_MD = REPORTS / "eth15m_highprice065_leaderboard_latest.md"
OUT_COMPARE_JSON = REPORTS / "eth15m_highprice065_180_365_archive_compare_latest.json"
OUT_COMPARE_MD = REPORTS / "eth15m_highprice065_180_365_archive_compare_latest.md"
OUT_VERDICT_JSON = REPORTS / "eth15m_highprice065_unique_verdict_latest.json"
OUT_VERDICT_MD = REPORTS / "eth15m_highprice065_unique_verdict_latest.md"
OUT_CHECKPOINT = REPORTS / "eth15m_highprice065_checkpoint_latest.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


of = load_module("eth15m_highprice_orderflow_base", ORDERFLOW_SCRIPT)
archive = load_module("eth15m_highprice_archive", ARCHIVE_SCRIPT)
of.MAX_TRAIN_ROWS = MAX_TRAIN_ROWS


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


def feature_subset(features: list[str], mode: str) -> list[str]:
    return of.feature_subset(features, mode)


def split_train_val(df: pd.DataFrame, window: str, train_window: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return of.split_train_val(df, window, train_window)


def max_drawdown(equity: np.ndarray) -> tuple[float, float]:
    return of.max_drawdown(equity)


def monthly_positive_ratio(equity: np.ndarray, dt: pd.Series) -> float:
    return of.monthly_positive_ratio(equity, dt)


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
        "buyPrice": float(buy_price),
        "winReturnPerStake": round(win_ret, 10),
        "breakEvenWinRatePct": round(100.0 * buy_price, 6),
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
        "setHash": stable_hash(sel[["dt", "pred_up15", "label_up", "won"]].to_dict("records")),
    }


def fit_model(engine: str, train: pd.DataFrame, feats: list[str], label: str, params: dict[str, Any], random_labels: bool = False):
    # Reuse the already audited model factory.
    return of.fit_model(engine, train, feats, label, params, random_labels=random_labels)


def predict_prob(model: Any, val: pd.DataFrame, feats: list[str]) -> np.ndarray:
    return of.predict_prob(model, val, feats)


def calibrated_predict(engine: str, train: pd.DataFrame, val: pd.DataFrame, feats: list[str], label: str, params: dict[str, Any], random_labels: bool = False) -> np.ndarray | None:
    calibration = params.get("calibration", "none")
    if calibration == "none":
        m = fit_model(engine, train, feats, label, params, random_labels=random_labels)
        return predict_prob(m, val, feats) if m is not None else None
    train = train.sort_values("dt").copy()
    if len(train) < 6000:
        m = fit_model(engine, train, feats, label, params, random_labels=random_labels)
        return predict_prob(m, val, feats) if m is not None else None
    cut = int(len(train) * 0.82)
    fit_part = train.iloc[:cut].copy()
    cal_part = train.iloc[cut:].copy()
    if len(fit_part) < 2500 or len(cal_part) < 1000:
        m = fit_model(engine, train, feats, label, params, random_labels=random_labels)
        return predict_prob(m, val, feats) if m is not None else None
    m = fit_model(engine, fit_part, feats, label, params, random_labels=random_labels)
    if m is None:
        return None
    p_cal = predict_prob(m, cal_part, feats)
    y_cal = cal_part[label].astype(int).to_numpy()
    p_val = predict_prob(m, val, feats)
    try:
        if calibration == "sigmoid":
            from sklearn.linear_model import LogisticRegression
            cal = LogisticRegression(max_iter=300, C=1.0, random_state=RNG_SEED)
            cal.fit(p_cal.reshape(-1, 1), y_cal)
            return np.asarray(cal.predict_proba(p_val.reshape(-1, 1))[:, 1], dtype=float)
        if calibration == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            cal = IsotonicRegression(out_of_bounds="clip")
            cal.fit(p_cal, y_cal)
            return np.asarray(cal.predict(p_val), dtype=float)
    except Exception:
        return p_val
    return p_val


def select_rows(val: pd.DataFrame, p_up: np.ndarray, p_quality: np.ndarray, params: dict[str, Any]) -> pd.DataFrame:
    pred_up = p_up >= 0.5
    dir_conf = np.maximum(p_up, 1.0 - p_up)
    q = np.asarray(p_quality, dtype=float)
    score_mode = params.get("score_mode", "dir_only")
    q_weight = float(params.get("quality_weight", 0.0))
    if score_mode == "weighted_avg":
        score = (1.0 - q_weight) * dir_conf + q_weight * q
    elif score_mode == "quality_boost":
        score = np.clip(dir_conf + q_weight * (q - 0.5), 0.0, 1.0)
    elif score_mode == "quality_gate":
        score = dir_conf.copy()
    else:
        score = dir_conf.copy()
    mask = score >= float(params["threshold"])
    if score_mode == "quality_gate":
        mask &= q >= float(params.get("quality_threshold", 0.55))
    band = params.get("score_band", "all")
    if band == "mid_only":
        mask &= score <= float(params.get("max_score", 0.90))
    if band == "high_only":
        mask &= score >= float(params.get("min_high_score", params["threshold"]))
    selected = val.loc[mask, ["dt", "label_up"]].copy()
    selected["pred_up15"] = pred_up[mask].astype(bool)
    selected["score15"] = score[mask]
    selected["dir_conf"] = dir_conf[mask]
    selected["quality_prob"] = q[mask]
    cap = int(params.get("daily_cap", 0) or 0)
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
    name = (
        f"highprice065_{params['engine']}_{params['train_window']}_{params['feature_mode']}"
        f"_thr{params['threshold']}_score{params.get('score_mode','dir_only')}"
        f"_cal{params.get('calibration','none')}_{stable_hash(params)}"
    )
    main_rows: list[dict[str, Any]] = []
    price_rows: list[dict[str, Any]] = []
    selected_counts: dict[str, int] = {}
    for window in ["180d", "365d"]:
        train, val = split_train_val(df, window, params["train_window"])
        p_up = calibrated_predict(params["engine"], train, val, feats, "label_up", params)
        p_q = calibrated_predict(params["quality_engine"], train, val, feats, params.get("quality_label", "label_strong_move"), params)
        if p_up is None or p_q is None:
            return None
        selected = select_rows(val, p_up, p_q, params)
        if selected.empty:
            return None
        selected_counts[window] = int(len(selected))
        for bp in BUY_PRICES:
            pr = curve_metrics(selected, name, window, bp)
            pr.update({
                "engine": params["engine"],
                "qualityEngine": params["quality_engine"],
                "trainWindow": params["train_window"],
                "featureMode": params["feature_mode"],
                "threshold": params["threshold"],
                "scoreMode": params.get("score_mode", "dir_only"),
                "qualityWeight": params.get("quality_weight", 0.0),
                "qualityThreshold": params.get("quality_threshold"),
                "qualityLabel": params.get("quality_label", "label_strong_move"),
                "calibration": params.get("calibration", "none"),
                "dailyCap": int(params.get("daily_cap", 0) or 0),
                "featureCount": len(feats),
            })
            price_rows.append(pr)
            if abs(bp - PRIMARY_BUY_PRICE) < 1e-9:
                main_rows.append(pr)
    return {"name": name, "params": params, "rows": main_rows, "priceRows": price_rows, "featureCount": len(feats), "selectedCounts": selected_counts}


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
        return {"params": params, "error": repr(exc)[:900]}


def param_grid() -> list[dict[str, Any]]:
    engines = [x for x in os.environ.get("ETH15M_HIGHPRICE_ENGINES", "logistic,lightgbm,xgboost").split(",") if x]
    q_engines = [x for x in os.environ.get("ETH15M_HIGHPRICE_QUALITY_ENGINES", "logistic,lightgbm").split(",") if x]
    rows: list[dict[str, Any]] = []
    thresholds = [0.60, 0.62, 0.64, 0.65, 0.66, 0.67, 0.68, 0.70, 0.72, 0.75, 0.78]
    for engine, qeng, tw, fm, thr, smode, qw, qthr, band, cap, mf, cal in itertools.product(
        engines,
        q_engines,
        ["1y", "3y", "5y", "full"],
        ["base15", "base_orderflow", "base_orderflow_htf", "linkage_wide", "wide"],
        thresholds,
        ["dir_only", "weighted_avg", "quality_boost", "quality_gate"],
        [0.0, 0.15, 0.30, 0.50],
        [0.52, 0.55, 0.58, 0.62],
        [("all", 1.0), ("mid_only", 0.82), ("mid_only", 0.90), ("high_only", 1.0)],
        [0, 48, 32, 24, 18, 12],
        [80, 140, 220, 316],
        ["none", "sigmoid"],
    ):
        score_band, max_score = band
        if smode != "quality_gate" and qthr != 0.52:
            continue
        if smode == "dir_only" and qw != 0.0:
            continue
        if score_band == "high_only" and thr < 0.66:
            continue
        common = {
            "engine": engine,
            "quality_engine": qeng,
            "train_window": tw,
            "feature_mode": fm,
            "threshold": thr,
            "score_mode": smode,
            "quality_weight": qw,
            "quality_threshold": qthr,
            "quality_label": "label_strong_move",
            "score_band": score_band,
            "max_score": max_score,
            "min_high_score": thr,
            "daily_cap": cap,
            "max_features": mf,
            "calibration": cal,
        }
        if engine == "logistic" and qeng == "logistic":
            for C in [0.08, 0.15, 0.3, 0.6, 1.0, 2.0]:
                rows.append(common | {"C": C, "n_estimators": 120, "learning_rate": 0.03, "num_leaves": 24, "min_child_samples": 90, "subsample": 0.88, "colsample_bytree": 0.88, "reg_lambda": 1.0, "depth": 3})
        else:
            for ne, lr, leaves, mcs, reg, depth in [(80, 0.04, 12, 160, 2.0, 2), (120, 0.03, 16, 140, 1.5, 3), (180, 0.022, 24, 180, 2.5, 3), (240, 0.016, 32, 240, 3.0, 4)]:
                rows.append(common | {"C": 1.0, "n_estimators": ne, "learning_rate": lr, "num_leaves": leaves, "min_child_samples": mcs, "subsample": 0.86, "colsample_bytree": 0.86, "reg_lambda": reg, "depth": depth})
    if PARAM_LIMIT and len(rows) > PARAM_LIMIT:
        rng = np.random.default_rng(RNG_SEED)
        idx = rng.permutation(len(rows))[:PARAM_LIMIT]
        rows = [rows[int(i)] for i in idx]
    return rows


def highprice_pass(candidate: dict[str, Any]) -> tuple[bool, list[str]]:
    by = {r["window"]: r for r in candidate.get("rows", []) if abs(r.get("buyPrice", 0) - PRIMARY_BUY_PRICE) < 1e-9}
    reasons: list[str] = []
    for window, min_trades in [("180d", 100), ("365d", 200)]:
        r = by.get(window)
        if not r:
            reasons.append(f"{window}_missing")
            continue
        if r["trades"] < min_trades:
            reasons.append(f"{window}_too_few_trades")
        if r["winRatePct"] <= 65.0:
            reasons.append(f"{window}_winrate_not_above_65")
        if r["compoundPnl"] <= 0:
            reasons.append(f"{window}_pnl_not_positive")
        if r["maxDrawdownUsd"] > MAX_ALLOWED_DRAWDOWN_USD:
            reasons.append(f"{window}_drawdown_over_25pct")
    # 0.60 must also survive. 0.70 is reported as stress, not a hard gate.
    price_by = {(r["window"], r["buyPrice"]): r for r in candidate.get("priceRows", [])}
    for window in ["180d", "365d"]:
        if price_by.get((window, 0.60), {}).get("compoundPnl", -1e9) <= 0:
            reasons.append(f"{window}_buy060_not_positive")
    return not reasons, reasons


def candidate_score(candidate: dict[str, Any]) -> float:
    by = {r["window"]: r for r in candidate.get("rows", []) if abs(r.get("buyPrice", 0) - PRIMARY_BUY_PRICE) < 1e-9}
    if set(by) != {"180d", "365d"}:
        return -1e18
    # Sorting priority requested: 365d win rate, 365d drawdown, 180d stability, pnl, trades.
    return (
        by["365d"]["winRatePct"] * 10000
        - by["365d"]["maxDrawdownUsd"] * 4
        + by["180d"]["winRatePct"] * 1500
        + np.log1p(max(0.0, by["365d"]["compoundPnl"])) * 300
        + min(by["365d"]["trades"], 2000) * 0.5
        - max(0, 100 - by["180d"]["trades"]) * 2000
    )


def archive_rows_for_candidate(df: pd.DataFrame, features: list[str], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not candidate or not candidate.get("params"):
        return rows
    try:
        fill_rows, _ = archive.load_live_fill_rows()
        all_eth = archive.aggregate_logical(fill_rows, "全部归档ETH15m live去重")
        strict_mask = fill_rows["sourcePath"].str.contains("slot01|forcedtrade|base_forcedtrade|__live_eth|logs_v5_exp10_bp0480__live_eth", case=False, regex=True)
        strict = archive.aggregate_logical(fill_rows[strict_mask].copy(), "严格旧slot1/ETH相关live去重") if strict_mask.any() else all_eth.iloc[0:0]
        params = candidate["params"]
        feats = feature_subset(features, params["feature_mode"])
        max_features = int(params.get("max_features", 0) or 0)
        if max_features > 0:
            feats = feats[:max_features]
        for old in [strict, all_eth]:
            if old.empty:
                continue
            start, end = old["marketStart"].min(), old["marketStart"].max()
            train = df[df["dt"] < start - pd.Timedelta(hours=4)].copy()
            if params["train_window"] != "full":
                days = {"1y": 365, "3y": 1095, "5y": 1825}[params["train_window"]]
                train = train[train["dt"] >= start - pd.Timedelta(days=days)].copy()
            val = df[(df["dt"] >= start) & (df["dt"] <= end)].copy()
            p_up = calibrated_predict(params["engine"], train, val, feats, "label_up", params)
            p_q = calibrated_predict(params["quality_engine"], train, val, feats, params.get("quality_label", "label_strong_move"), params)
            if p_up is None or p_q is None:
                continue
            sel = select_rows(val, p_up, p_q, params)
            pred = sel[["dt", "pred_up15", "score15"]].copy()
            merged = old.merge(pred, left_on="marketStart", right_on="dt", how="left")
            chosen = merged[merged["pred_up15"].notna()].copy().sort_values("marketStart")
            if chosen.empty:
                rows.append({"scope": old["scopeName"].iloc[0], "selectedTrades": 0, "oldRealMarkets": int(len(old)), "archivePass55": False})
                continue
            won = chosen["pred_up15"].astype(bool).to_numpy() == chosen["actualUp"].astype(bool).to_numpy()
            arch_sel = pd.DataFrame({"dt": chosen["marketStart"], "label_up": chosen["actualUp"].astype(bool), "pred_up15": chosen["pred_up15"].astype(bool), "won": won})
            m = curve_metrics(arch_sel, candidate["name"], str(old["scopeName"].iloc[0]), PRIMARY_BUY_PRICE)
            win_rate = round(100 * float(won.mean()), 6)
            rows.append({
                "scope": old["scopeName"].iloc[0],
                "oldRealMarkets": int(len(old)),
                "selectedTrades": int(len(chosen)),
                "wins": int(won.sum()),
                "losses": int(len(won) - int(won.sum())),
                "winRatePct": win_rate,
                "compoundPnl": m["compoundPnl"],
                "endingBankroll": m["endingBankroll"],
                "maxDrawdownUsd": m["maxDrawdownUsd"],
                "archivePass55": bool(win_rate >= 55.0) if len(chosen) >= 30 else False,
                "setHash": stable_hash(chosen[["marketSlug", "pred_up15", "actualUp"]].to_dict("records")),
            })
    except Exception as exc:
        rows.append({"scope": "archive_error", "error": repr(exc)[:600], "archivePass55": False})
    return rows


def run_audit(df: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    params = {
        "engine": "logistic", "quality_engine": "logistic", "train_window": "3y", "feature_mode": "base_orderflow_htf",
        "threshold": 0.65, "score_mode": "dir_only", "quality_weight": 0.0, "quality_threshold": 0.55,
        "quality_label": "label_strong_move", "score_band": "all", "max_score": 1.0, "daily_cap": 0,
        "max_features": 140, "calibration": "sigmoid", "C": 0.6, "n_estimators": 120, "learning_rate": 0.03,
        "num_leaves": 16, "min_child_samples": 140, "subsample": 0.86, "colsample_bytree": 0.86, "reg_lambda": 1.5, "depth": 3,
    }
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
        p_up = calibrated_predict(params["engine"], train, val, feats, "label_up", params, random_labels=True)
        p_q = calibrated_predict(params["quality_engine"], train, val, feats, params["quality_label"], params, random_labels=True)
        if p_up is not None and p_q is not None:
            sel = select_rows(val, p_up, p_q, params)
            wr = float(sel["won"].mean() * 100.0) if len(sel) else 0.0
            random_label = {"status": "ok", "selectedTrades": int(len(sel)), "winRatePct": round(wr, 6), "passed": len(sel) < 50 or 43.0 <= wr <= 57.0}
    except Exception as exc:
        random_label = {"status": "error", "error": repr(exc)[:500], "passed": False}
    forbidden_hits = [c for c in features if c.startswith(tuple(of.base.FORBIDDEN_PREFIXES)) or c in of.base.FORBIDDEN_FEATURES]
    audit = {
        "generatedAt": now_iso(),
        "beijingTime": bj_now(),
        "researchOnlyNoLiveChange": True,
        "primaryBuyPrice": PRIMARY_BUY_PRICE,
        "breakEvenWinRatePct": 65.0,
        "winReturnPerStake": round(1.0 / PRIMARY_BUY_PRICE - 1.0, 10),
        "timePolicy": "closed_candle_available_at_v2 plus 1m available_at=open+1min",
        "forbiddenFeatureHits": forbidden_hits,
        "repeatability": repeat,
        "randomLabelAudit": random_label,
        "passed": not forbidden_hits and bool(repeat) and all(r["passed"] for r in repeat) and bool(random_label.get("passed")),
    }
    write_json(OUT_LABEL_AUDIT_JSON, audit)
    write_text(OUT_LABEL_AUDIT_MD, "\n".join([
        "# ETH15m 高价0.65模型：标签与回测审计", "",
        f"- 北京时间：`{audit['beijingTime']}`",
        f"- 主买价：`{PRIMARY_BUY_PRICE}`；打平胜率：`65%`；赢单收益/本金：`{audit['winReturnPerStake']}`",
        f"- 时间口径：`{audit['timePolicy']}`",
        f"- 禁用字段命中：`{forbidden_hits}`",
        f"- 重复运行：`{repeat}`",
        f"- 随机标签：`{random_label}`",
        f"- 审计通过：`{audit['passed']}`",
    ]) + "\n")
    return audit


def rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [r for r in results if r.get("rows")]
    out = []
    for c in valid:
        ok, reasons = highprice_pass(c)
        out.append(c | {"passed": ok, "reasons": reasons, "score": candidate_score(c)})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def render_outputs(payload: dict[str, Any]) -> None:
    top = payload.get("topCandidates", [])
    selected = payload.get("selected")
    lines = [
        "# ETH15m 高价0.65模型：180/365/归档对比", "",
        f"- 北京时间：`{payload['beijingTime']}`",
        "- 口径：`850U初始 / 每笔1% / 满成交 / 主买价0.65`",
        "- 数学：买价0.65时，赢单赚本金53.85%，输单亏100%，打平胜率65%。",
        "- live动作：`无，研究只读`", "",
        "## 最佳候选主表", "",
        "|配置|窗口|买价|交易数|胜/负|胜率|盈亏|期末资金|最大回撤|收益回撤比|月正收益|哈希|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    if selected:
        for r in selected.get("priceRows", []):
            if r["buyPrice"] in BUY_PRICES:
                lines.append(f"|高价0.65最佳|{r['window']}|{r['buyPrice']:.2f}|{r['trades']}|{r['wins']}/{r['losses']}|{r['winRatePct']:.2f}%|{r['compoundPnl']:,.2f}|{r['endingBankroll']:,.2f}|{r['maxDrawdownUsd']:,.2f}|{r['returnDrawdownRatio']:.2f}|{r['monthlyPositiveRatio']:.2%}|`{r['setHash']}`|")
    lines += ["", "## 前15候选（按365天胜率与回撤排序）", "", "|排名|配置|180胜率|180盈亏|180回撤|365胜率|365盈亏|365回撤|交易数365|通过|失败原因|", "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for i, c in enumerate(top[:15], 1):
        by = {r["window"]: r for r in c.get("rows", [])}
        reasons = ",".join(c.get("reasons", []))
        lines.append(f"|{i}|`{c['name']}`|{by.get('180d',{}).get('winRatePct',0):.2f}%|{by.get('180d',{}).get('compoundPnl',0):,.2f}|{by.get('180d',{}).get('maxDrawdownUsd',0):,.2f}|{by.get('365d',{}).get('winRatePct',0):.2f}%|{by.get('365d',{}).get('compoundPnl',0):,.2f}|{by.get('365d',{}).get('maxDrawdownUsd',0):,.2f}|{by.get('365d',{}).get('trades',0)}|{c.get('passed', False)}|{reasons}|")
    lines += ["", "## 历史归档真实单纯预测复核", "", "|范围|旧真实市场数|新模型选中|胜/负|胜率|盈亏|最大回撤|是否>=55%|", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in payload.get("archiveRows", []):
        lines.append(f"|{r.get('scope')}|{r.get('oldRealMarkets',0)}|{r.get('selectedTrades',0)}|{r.get('wins',0)}/{r.get('losses',0)}|{r.get('winRatePct',0):.2f}%|{r.get('compoundPnl',0):,.2f}|{r.get('maxDrawdownUsd',0):,.2f}|{r.get('archivePass55', False)}|")
    write_text(OUT_COMPARE_MD, "\n".join(lines) + "\n")

    lb_lines = ["# ETH15m 高价0.65模型：候选排行榜", "", "|排名|配置|训练窗|模型|特征组|阈值|校准|365胜率|365交易|365盈亏|365回撤|通过|", "|---:|---|---|---|---|---:|---|---:|---:|---:|---:|---|"]
    for i, c in enumerate(top[:100], 1):
        by = {r["window"]: r for r in c.get("rows", [])}
        p = c.get("params", {})
        lb_lines.append(f"|{i}|`{c['name']}`|{p.get('train_window')}|{p.get('engine')}/{p.get('quality_engine')}|{p.get('feature_mode')}|{p.get('threshold')}|{p.get('calibration')}|{by.get('365d',{}).get('winRatePct',0):.2f}%|{by.get('365d',{}).get('trades',0)}|{by.get('365d',{}).get('compoundPnl',0):,.2f}|{by.get('365d',{}).get('maxDrawdownUsd',0):,.2f}|{c.get('passed', False)}|")
    write_text(OUT_LEADERBOARD_MD, "\n".join(lb_lines) + "\n")

    verdict = payload.get("verdict", {})
    v_lines = [
        "# ETH15m 高价0.65模型：唯一结论", "",
        f"- 状态：`{verdict.get('status')}`",
        f"- 结论：{verdict.get('message')}", "",
        "## 说明", "",
        "- 这条线专门服务 `平均买价0.65`，不再用 `0.50` 高交易频率表硬凑。",
        "- 如果没有候选同时满足 180/365 胜率 >65%、正盈亏、低回撤、归档不差，就不能进入真钱。",
    ]
    write_text(OUT_VERDICT_MD, "\n".join(v_lines) + "\n")


def write_progress(results: list[dict[str, Any]], total: int, start_ts: float, df: pd.DataFrame, features: list[str], data_truth: dict[str, Any], label_audit: dict[str, Any], finished: bool = False) -> None:
    ranked = rank_results(results)
    strict = [r for r in ranked if r.get("passed")]
    selected = strict[0] if strict else (ranked[0] if ranked else None)
    archive_rows = archive_rows_for_candidate(df, features, selected) if selected else []
    archive_ok = bool(archive_rows) and all((r.get("archivePass55") or r.get("selectedTrades", 0) < 30) for r in archive_rows if not r.get("error"))
    live_eligible = bool(selected and selected.get("passed") and archive_ok)
    status = "highprice065_candidate_found" if live_eligible else "no_highprice065_candidate"
    msg = "找到满足0.65高价口径的候选，只能进入24小时影子验证，不改真钱。" if live_eligible else "没有候选同时满足0.65买价下180/365胜率>65%、正盈亏、低回撤和归档复核。当前不应切真钱。"
    payload = {
        "generatedAt": now_iso(),
        "beijingTime": bj_now(),
        "researchOnlyNoLiveChange": True,
        "finished": finished,
        "elapsedSeconds": round(time.time() - start_ts, 3),
        "workers": WORKERS,
        "done": len(results),
        "total": total,
        "valid": len([r for r in results if r.get("rows")]),
        "strictPass": len(strict),
        "primaryBuyPrice": PRIMARY_BUY_PRICE,
        "breakEvenWinRatePct": 65.0,
        "dataTruth": data_truth,
        "labelAudit": label_audit,
        "topCandidates": ranked[:300],
        "strictPassCandidates": strict[:80],
        "selected": selected,
        "archiveRows": archive_rows,
        "verdict": {"status": status, "message": msg, "archiveOk": archive_ok},
    }
    write_json(OUT_CHECKPOINT, payload)
    write_json(OUT_LEADERBOARD_JSON, payload)
    write_json(OUT_COMPARE_JSON, payload)
    write_json(OUT_VERDICT_JSON, payload["verdict"] | {"selected": selected, "archiveRows": archive_rows, "generatedAt": payload["generatedAt"]})
    render_outputs(payload)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    df, features, data_truth = of.build_dataset()
    data_truth = dict(data_truth)
    data_truth.update({
        "generatedAtHighPrice065": now_iso(),
        "researchOnlyNoLiveChange": True,
        "primaryBuyPrice": PRIMARY_BUY_PRICE,
        "auxBuyPrices": BUY_PRICES,
        "breakEvenWinRatePct": 65.0,
        "winReturnPerStakeAt065": round(1.0 / PRIMARY_BUY_PRICE - 1.0, 10),
        "note": "This script optimizes for high-price 0.65. It does not mutate current 061 live trading.",
    })
    write_json(OUT_DATA_TRUTH_JSON, data_truth)
    write_text(OUT_DATA_TRUTH_MD, "\n".join([
        "# ETH15m 高价0.65模型：数据真相", "",
        f"- 北京时间：`{bj_now()}`",
        f"- 最终样本：`{data_truth['finalRows']}`",
        f"- 特征数：`{data_truth['featureCount']}`",
        f"- 主买价：`{PRIMARY_BUY_PRICE}`；打平胜率：`65%`；赢单收益/本金：`{data_truth['winReturnPerStakeAt065']}`",
        f"- ETH 1m 覆盖：`{data_truth['eth1m']['start']} -> {data_truth['eth1m']['end']}`",
        f"- 1分钟说明：{data_truth['eth1m']['note']}",
        f"- 标签比例：`{data_truth['labelRates']}`",
        "", "本脚本只做研究，不改真实交易。",
    ]) + "\n")
    label_audit = run_audit(df, features)
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
                write_progress(results, len(params), started, df, features, data_truth, label_audit, finished=False)
                last_write = time.time()
            if time.time() - started > MAX_SECONDS:
                break
    write_progress(results, len(params), started, df, features, data_truth, label_audit, finished=True)


if __name__ == "__main__":
    main()
