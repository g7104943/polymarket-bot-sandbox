#!/usr/bin/env python3
from __future__ import annotations

"""Score-calibrated router for the current ETH061 candidate universe.

Research only.  This script does not mutate live configs, ledgers, orders,
claim state, monitor state, or model profiles.

Goal:
  - Keep the audited closed-candle alignment from the 061 research stack.
  - Stop adding more hand-built clusters.
  - Learn, from past candidates only, whether the current base signal is worth
    trading under the 0.55 buy-price research口径.
"""

import hashlib
import importlib.util
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for key in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(key, os.environ.get("TOP159_061_CALIB_THREADS", "1"))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional local dependency
    LGBMClassifier = None  # type: ignore

ROOT = Path("/Users/mac/polyfun")
REPORTS = ROOT / "reports"
BASE_SCRIPT = ROOT / "polyfun-next" / "scripts" / "run_top159_shock_filter_cluster_targeted_search.py"
ARCHIVE_SCRIPT = ROOT / "scripts" / "ops" / "run_top159_all_archived_real_eth_compare.py"

START_BANKROLL = 850.0
STAKE_PCT = 0.01
BUY_PRICE = 0.55
STRESS_BUY_PRICE = 0.60
RNG_SEED = 20260509

OUT_AUDIT_MD = REPORTS / "top159_061_score_calibration_router_bug_audit_latest.md"
OUT_AUDIT_JSON = REPORTS / "top159_061_score_calibration_router_bug_audit_latest.json"
OUT_LEADERBOARD_MD = REPORTS / "top159_061_score_calibration_router_leaderboard_latest.md"
OUT_LEADERBOARD_JSON = REPORTS / "top159_061_score_calibration_router_leaderboard_latest.json"
OUT_COMPARE_MD = REPORTS / "top159_061_score_calibration_router_061_compare_latest.md"
OUT_COMPARE_JSON = REPORTS / "top159_061_score_calibration_router_061_compare_latest.json"
OUT_VERDICT_MD = REPORTS / "top159_061_score_calibration_router_unique_verdict_latest.md"
OUT_VERDICT_JSON = REPORTS / "top159_061_score_calibration_router_unique_verdict_latest.json"

BASELINE_061 = {
    "180d": {
        "trades": 3942,
        "wins": 2324,
        "losses": 1618,
        "winRatePct": 58.954845,
        "compoundPnl": 11494.591625,
        "endingBankroll": 12344.591625,
        "maxDrawdownUsd": 1481.713373,
        "returnDrawdownRatio": 7.757635,
        "monthlyPositiveRatio": 1.0,
        "setHash": "b35313c05e5b66d2",
    },
    "365d": {
        "trades": 9152,
        "wins": 5246,
        "losses": 3906,
        "winRatePct": 57.320804,
        "compoundPnl": 27033.917055,
        "endingBankroll": 27883.917055,
        "maxDrawdownUsd": 3499.248929,
        "returnDrawdownRatio": 7.725634,
        "monthlyPositiveRatio": 0.846154,
        "setHash": "ced1cf82642d8f0d",
    },
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base = load_module("score_calib_cluster_base", BASE_SCRIPT)
archive = load_module("score_calib_archive", ARCHIVE_SCRIPT)


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
    frame = pd.DataFrame({"dt": pd.to_datetime(dt, utc=True, errors="coerce"), "equity": equity}).dropna()
    if frame.empty:
        return 0.0
    frame = frame.sort_values("dt")
    frame["month"] = frame["dt"].dt.to_period("M").astype(str)
    prev = START_BANKROLL
    vals = []
    for _, group in frame.groupby("month", sort=True):
        end = float(group["equity"].iloc[-1])
        vals.append(end - prev)
        prev = end
    return round(sum(v > 0 for v in vals) / len(vals), 6) if vals else 0.0


def curve_metrics(rows: pd.DataFrame, name: str, window: str, buy_price: float) -> dict[str, Any]:
    selected = rows.sort_values("dt").reset_index(drop=True).copy()
    won = selected["won"].astype(bool).to_numpy()
    equity = START_BANKROLL
    curve = np.empty(len(selected), dtype=float)
    win_ret = 1.0 / float(buy_price) - 1.0
    for i, ok in enumerate(won):
        stake = equity * STAKE_PCT
        equity += stake * (win_ret if ok else -1.0)
        equity = max(equity, 0.0)
        curve[i] = equity
    mxdd, mxdd_pct = max_drawdown(curve)
    wins = int(won.sum())
    losses = int(len(won) - wins)
    return {
        "name": name,
        "window": window,
        "buyPrice": float(buy_price),
        "trades": int(len(selected)),
        "wins": wins,
        "losses": losses,
        "winRatePct": round(100.0 * wins / len(selected), 6) if len(selected) else 0.0,
        "compoundPnl": round(float(equity - START_BANKROLL), 6),
        "endingBankroll": round(float(equity), 6),
        "maxDrawdownUsd": round(float(mxdd), 6),
        "maxDrawdownPct": round(float(mxdd_pct) * 100.0, 6),
        "returnDrawdownRatio": round(float(equity - START_BANKROLL) / mxdd, 6) if mxdd > 1e-12 else (999.0 if equity > START_BANKROLL else 0.0),
        "monthlyPositiveRatio": monthly_positive_ratio(curve, selected["dt"]),
        "setHash": stable_hash(selected[["dt", "pred_up15", "label_up", "won"]].to_dict("records")) if len(selected) else "empty",
    }


def build_truth() -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]], dict[str, pd.DataFrame]]:
    enriched, _truth = base.ext.load_or_build_enriched()
    atom_store = base.build_atom_store(enriched)
    period_vals = base.build_period_vals(enriched)
    return enriched, atom_store, period_vals


def current_061_mask(atom_store: dict[str, dict[str, np.ndarray]], period: str, val: pd.DataFrame) -> np.ndarray:
    cond = base.condition_for_candidate(atom_store, period, base.CURRENT_061_PARAMS)
    score = pd.to_numeric(val["score15"], errors="coerce").fillna(0.0).to_numpy()
    return ((~cond) | (score >= float(base.CURRENT_061_PARAMS["shock_score_min"]))).astype(bool)


def one_hot(series: pd.Series, prefix: str) -> pd.DataFrame:
    return pd.get_dummies(series.fillna("missing").astype(str), prefix=prefix, dtype=float)


def make_feature_frame(df: pd.DataFrame, feature_mode: str, atom_masks: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["score15"] = pd.to_numeric(df["score15"], errors="coerce").fillna(0.0)
    out["score_margin"] = out["score15"] - 0.55
    out["is_up"] = (df["direction"].astype(str).str.upper() == "UP").astype(float)
    bj = pd.to_datetime(df["dt"], utc=True, errors="coerce").dt.tz_convert("Asia/Shanghai")
    hour = bj.dt.hour.fillna(0).astype(float)
    out["bj_hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["bj_hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    numeric_cols = []
    for tf in ["15m", "1h", "4h"]:
        for suffix in [
            "body_ratio",
            "range_pct",
            "range_q",
            "volume_mult",
            "close_pos",
            "upper_wick_ratio",
            "lower_wick_ratio",
            "pos20",
        ]:
            col = f"{tf}_{suffix}"
            if col in df.columns:
                numeric_cols.append(col)
        for flag in ["same_as_top159", "opposes_top159", "terminal_chase", "exhaustion_wick", "is_big_body_default"]:
            col = f"{tf}_{flag}"
            if col in df.columns:
                out[col] = df[col].fillna(False).astype(bool).astype(float)
        if f"{tf}_trend_state" in df.columns:
            out = pd.concat([out, one_hot(df[f"{tf}_trend_state"], f"{tf}_trend")], axis=1)
        if f"{tf}_candle_dir" in df.columns:
            out = pd.concat([out, one_hot(df[f"{tf}_candle_dir"], f"{tf}_candle")], axis=1)
    for col in numeric_cols:
        out[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if feature_mode in {"atoms_core", "atoms_all"} and atom_masks:
        allowed = []
        for name, arr in atom_masks.items():
            if feature_mode == "atoms_core":
                if not (
                    name.startswith("dir_")
                    or name.startswith("score_")
                    or name.startswith("bj_hour_")
                    or "same_as_top159" in name
                    or "opposes_top159" in name
                    or name.endswith("_pos_high")
                    or name.endswith("_pos_low")
                    or "_trend_" in name
                    or "_vol_ge_" in name
                    or "_rangeq_ge_" in name
                ):
                    continue
            n = int(np.asarray(arr, dtype=bool).sum())
            if 25 <= n <= len(df) - 25:
                allowed.append(name)
        for name in sorted(allowed)[:160]:
            out[f"atom__{name}"] = np.asarray(atom_masks[name], dtype=float)
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def align_features(train_x: pd.DataFrame, calib_x: pd.DataFrame, val_x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = sorted(set(train_x.columns) | set(calib_x.columns) | set(val_x.columns))
    return train_x.reindex(columns=cols, fill_value=0.0), calib_x.reindex(columns=cols, fill_value=0.0), val_x.reindex(columns=cols, fill_value=0.0)


def split_train_calib(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ordered = df.sort_values("dt").index.to_numpy()
    cut = max(100, int(len(ordered) * 0.75))
    cut = min(cut, len(ordered) - 50)
    return ordered[:cut], ordered[cut:]


def make_model(model_name: str):
    if model_name.startswith("logit_l2_"):
        c = float(model_name.rsplit("_", 1)[1])
        return make_pipeline(StandardScaler(with_mean=False), LogisticRegression(C=c, penalty="l2", max_iter=1000, class_weight="balanced", solver="liblinear"))
    if model_name.startswith("logit_l1_"):
        c = float(model_name.rsplit("_", 1)[1])
        return make_pipeline(StandardScaler(with_mean=False), LogisticRegression(C=c, penalty="l1", max_iter=1000, class_weight="balanced", solver="liblinear"))
    if model_name == "hgb_shallow":
        return HistGradientBoostingClassifier(max_leaf_nodes=12, learning_rate=0.035, max_iter=140, l2_regularization=0.5, random_state=RNG_SEED)
    if model_name == "rf_shallow":
        return RandomForestClassifier(n_estimators=160, max_depth=5, min_samples_leaf=80, class_weight="balanced_subsample", random_state=RNG_SEED, n_jobs=1)
    if model_name == "lgbm_shallow" and LGBMClassifier is not None:
        return LGBMClassifier(
            n_estimators=180,
            learning_rate=0.025,
            num_leaves=12,
            min_child_samples=80,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            random_state=RNG_SEED,
            n_jobs=1,
            verbose=-1,
        )
    raise ValueError(model_name)


def fit_predict_one_window(
    period_vals: dict[str, pd.DataFrame],
    atom_store: dict[str, dict[str, np.ndarray]],
    window: str,
    feature_mode: str,
    model_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_period = "gate_train_for_180d" if window == "180d" else "gate_train_for_365d"
    val_period = "validation_180d" if window == "180d" else "validation_365d"
    train_df = period_vals[train_period].reset_index(drop=True)
    val_df = period_vals[val_period].reset_index(drop=True)
    train_idx, calib_idx = split_train_calib(train_df)

    train_x_all = make_feature_frame(train_df, feature_mode, atom_store[train_period])
    val_x = make_feature_frame(val_df, feature_mode, atom_store[val_period])
    core_x = train_x_all.loc[train_idx]
    calib_x = train_x_all.loc[calib_idx]
    core_x, calib_x, val_x = align_features(core_x, calib_x, val_x)

    y_core = train_df.loc[train_idx, "won"].astype(bool).astype(int).to_numpy()
    y_calib = train_df.loc[calib_idx, "won"].astype(bool).astype(int).to_numpy()
    model = make_model(model_name)
    model.fit(core_x, y_core)
    raw_calib = model.predict_proba(calib_x)[:, 1]
    raw_val = model.predict_proba(val_x)[:, 1]

    # A second-stage logistic calibration keeps the router from trusting raw
    # tree scores as if they were real probabilities.
    calibrator = LogisticRegression(C=0.25, penalty="l2", max_iter=1000, solver="lbfgs")
    calibrator.fit(raw_calib.reshape(-1, 1), y_calib)
    p_val = calibrator.predict_proba(raw_val.reshape(-1, 1))[:, 1]
    p_calib = calibrator.predict_proba(raw_calib.reshape(-1, 1))[:, 1]
    loss = float(log_loss(y_calib, np.clip(p_calib, 1e-6, 1 - 1e-6), labels=[0, 1]))

    out = val_df.copy()
    out["router_prob"] = p_val
    out["current061_keep"] = current_061_mask(atom_store, val_period, val_df)
    meta = {
        "window": window,
        "trainPeriod": train_period,
        "valPeriod": val_period,
        "featureMode": feature_mode,
        "model": model_name,
        "features": int(len(core_x.columns)),
        "trainRows": int(len(core_x)),
        "calibRows": int(len(calib_x)),
        "calibLogLoss": round(loss, 6),
    }
    return out, meta


@dataclass(frozen=True)
class Policy:
    scope: str
    prob_min: float
    score_min: float
    daily_cap: int | None
    require_061: bool

    def name(self, feature_mode: str, model_name: str) -> str:
        return stable_hash({
            "router": "score_calibration_v1",
            "featureMode": feature_mode,
            "model": model_name,
            "scope": self.scope,
            "probMin": self.prob_min,
            "scoreMin": self.score_min,
            "dailyCap": self.daily_cap,
            "require061": self.require_061,
        })


def apply_policy(df: pd.DataFrame, policy: Policy) -> pd.DataFrame:
    sel = df.copy()
    mask = (pd.to_numeric(sel["router_prob"], errors="coerce").fillna(0.0) >= policy.prob_min) & (
        pd.to_numeric(sel["score15"], errors="coerce").fillna(0.0) >= policy.score_min
    )
    if policy.require_061:
        mask &= sel["current061_keep"].astype(bool)
    selected = sel[mask].copy()
    if policy.daily_cap and not selected.empty:
        selected["bj_day"] = pd.to_datetime(selected["dt"], utc=True).dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m-%d")
        selected = (
            selected.sort_values(["bj_day", "router_prob"], ascending=[True, False])
            .groupby("bj_day", group_keys=False)
            .head(policy.daily_cap)
            .sort_values("dt")
        )
    return selected.sort_values("dt").reset_index(drop=True)


def evaluate_router_candidate(window_preds: dict[str, pd.DataFrame], policy: Policy, feature_mode: str, model_name: str) -> dict[str, Any]:
    name = policy.name(feature_mode, model_name)
    rows = []
    stress = []
    for w in ["180d", "365d"]:
        selected = apply_policy(window_preds[w], policy)
        r = curve_metrics(selected, name, w, BUY_PRICE)
        s = curve_metrics(selected, name, w, STRESS_BUY_PRICE)
        base_count = len(window_preds[w])
        current061 = int(window_preds[w]["current061_keep"].astype(bool).sum())
        rows.append(r | {"retentionRateVsBase": round(100 * len(selected) / max(1, base_count), 6), "retentionRateVs061": round(100 * len(selected) / max(1, current061), 6)})
        stress.append(s)
    return {
        "name": name,
        "kind": "score_calibration_router",
        "featureMode": feature_mode,
        "model": model_name,
        "policy": policy.__dict__,
        "rows": rows,
        "stressRows": stress,
    }


def pass_gate(candidate: dict[str, Any]) -> tuple[bool, list[str]]:
    rows = {r["window"]: r for r in candidate["rows"]}
    reasons = []
    for w in ["180d", "365d"]:
        r = rows.get(w)
        b = BASELINE_061[w]
        if not r:
            reasons.append(f"{w}_missing")
            continue
        if r["trades"] < (100 if w == "180d" else 200):
            reasons.append(f"{w}_too_few")
        if r["trades"] < int(b["trades"] * 0.45):
            reasons.append(f"{w}_retention_too_low")
        if r["compoundPnl"] <= b["compoundPnl"]:
            reasons.append(f"{w}_pnl_not_above_061")
        if w == "365d" and r["winRatePct"] < b["winRatePct"]:
            reasons.append("365d_winrate_below_061")
        if r["maxDrawdownUsd"] > b["maxDrawdownUsd"] * 1.05:
            reasons.append(f"{w}_drawdown_too_high")
    return not reasons, reasons


def score_candidate(candidate: dict[str, Any]) -> float:
    rows = {r["window"]: r for r in candidate["rows"]}
    if set(rows) != {"180d", "365d"}:
        return -1e18
    b180, b365 = BASELINE_061["180d"], BASELINE_061["365d"]
    return (
        (rows["180d"]["compoundPnl"] - b180["compoundPnl"]) / max(1.0, b180["compoundPnl"]) * 3000
        + (rows["365d"]["compoundPnl"] - b365["compoundPnl"]) / max(1.0, b365["compoundPnl"]) * 5000
        + (rows["365d"]["winRatePct"] - b365["winRatePct"]) * 45
        - (rows["180d"]["maxDrawdownUsd"] / max(1.0, b180["maxDrawdownUsd"])) * 10
        - (rows["365d"]["maxDrawdownUsd"] / max(1.0, b365["maxDrawdownUsd"])) * 10
    )


def baseline_rows(period_vals: dict[str, pd.DataFrame], atom_store: dict[str, dict[str, np.ndarray]]) -> list[dict[str, Any]]:
    out = []
    for window, period in [("180d", "validation_180d"), ("365d", "validation_365d")]:
        val = period_vals[period].reset_index(drop=True)
        keep = current_061_mask(atom_store, period, val)
        selected = val[keep].copy()
        out.append(curve_metrics(selected, "current_live_06173_cluster_gate", window, BUY_PRICE))
    return out


def archive_rows(candidate: dict[str, Any], window_preds: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    out = []
    try:
        fill_rows, _ = archive.load_live_fill_rows()
        all_eth = archive.aggregate_logical(fill_rows, "全部归档ETH15m live去重")
        strict_mask = fill_rows["sourcePath"].str.contains(
            "slot01|forcedtrade|base_forcedtrade|__live_eth|logs_v5_exp10_bp0480__live_eth",
            case=False,
            regex=True,
        )
        strict = archive.aggregate_logical(fill_rows[strict_mask].copy(), "严格旧slot1/ETH相关live去重") if strict_mask.any() else all_eth.iloc[0:0]
        policy = Policy(**candidate["policy"])
        pred = window_preds["180d"].copy()
        selected = apply_policy(pred, policy)
        # apply_policy sorts and resets the selected frame index, so marking by
        # integer index would corrupt archived-real replay.  The 15m market
        # start time is the stable logical key in this dataset.
        selected_dt = set(pd.to_datetime(selected["dt"], utc=True, errors="coerce").dropna().astype("int64").tolist())
        pred_dt = pd.to_datetime(pred["dt"], utc=True, errors="coerce").astype("int64")
        pred["selected"] = pred_dt.isin(selected_dt)
        pred_small = pred[["dt", "pred_up15", "score15", "router_prob", "selected"]]
        for old in [strict, all_eth]:
            if old.empty:
                continue
            merged = old.merge(pred_small, left_on="marketStart", right_on="dt", how="left")
            chosen = merged[merged["selected"].fillna(False).astype(bool)].copy().sort_values("marketStart")
            if chosen.empty:
                out.append({"scope": old["scopeName"].iloc[0], "oldRealMarkets": int(len(old)), "selectedTrades": 0})
                continue
            won = chosen["pred_up15"].astype(bool).to_numpy() == chosen["actualUp"].astype(bool).to_numpy()
            arch_sel = pd.DataFrame({"dt": chosen["marketStart"], "label_up": chosen["actualUp"].astype(bool), "pred_up15": chosen["pred_up15"].astype(bool), "won": won})
            m = curve_metrics(arch_sel, candidate["name"], str(old["scopeName"].iloc[0]), BUY_PRICE)
            out.append(
                {
                    "scope": old["scopeName"].iloc[0],
                    "oldRealMarkets": int(len(old)),
                    "selectedTrades": int(len(chosen)),
                    "wins": int(won.sum()),
                    "losses": int(len(won) - int(won.sum())),
                    "winRatePct": round(100 * float(won.mean()), 6),
                    "compoundPnl": m["compoundPnl"],
                    "endingBankroll": m["endingBankroll"],
                    "maxDrawdownUsd": m["maxDrawdownUsd"],
                    "avgRouterProb": round(float(chosen["router_prob"].mean()), 6),
                    "setHash": stable_hash(chosen[["marketSlug", "pred_up15", "actualUp"]].to_dict("records")),
                }
            )
    except Exception as exc:
        out.append({"scope": "archive_error", "error": repr(exc)[:700]})
    return out


def run_audit(period_vals: dict[str, pd.DataFrame], atom_store: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    replay = baseline_rows(period_vals, atom_store)
    repeat = baseline_rows(period_vals, atom_store)
    baseline_match = []
    for a, b in zip(replay, repeat):
        expected = BASELINE_061[a["window"]]
        baseline_match.append(
            {
                "window": a["window"],
                "trades": a["trades"],
                "winRatePct": a["winRatePct"],
                "pnl": a["compoundPnl"],
                "expectedPnl": expected["compoundPnl"],
                "repeatHashMatch": a["setHash"] == b["setHash"],
                "passed": a["trades"] == expected["trades"] and abs(a["compoundPnl"] - expected["compoundPnl"]) < 1e-6 and a["setHash"] == b["setHash"],
            }
        )
    rng = np.random.default_rng(RNG_SEED)
    shuffle = []
    for r, (window, period) in zip(replay, [("180d", "validation_180d"), ("365d", "validation_365d")]):
        val = period_vals[period].reset_index(drop=True)
        keep = current_061_mask(atom_store, period, val)
        fake = val[keep].copy()
        fake["won"] = rng.random(len(fake)) < 0.5
        m = curve_metrics(fake, "random_label", window, BUY_PRICE)
        shuffle.append({"window": window, "trades": len(fake), "randomWinRatePct": m["winRatePct"], "randomPnl": m["compoundPnl"]})
    audit = {
        "generatedAt": now_iso(),
        "beijingTime": bj_now(),
        "researchOnlyNoLiveChange": True,
        "timePolicy": "closed_candle_available_at_v2",
        "baselineReplay": baseline_match,
        "randomLabelSanity": shuffle,
        "passed": all(x["passed"] for x in baseline_match),
    }
    write_json(OUT_AUDIT_JSON, audit)
    write_text(
        OUT_AUDIT_MD,
        "\n".join(
            [
                "# 061 分数校准路由：审计",
                "",
                f"- 北京时间：`{audit['beijingTime']}`",
                f"- 时间口径：`{audit['timePolicy']}`",
                f"- 当前061复现：`{baseline_match}`",
                f"- 随机标签 sanity：`{shuffle}`",
                f"- 审计通过：`{audit['passed']}`",
            ]
        )
        + "\n",
    )
    return audit


def format_compare(payload: dict[str, Any]) -> str:
    lines = [
        "# 061 分数校准路由：公平对比",
        "",
        f"- 北京时间：`{payload['beijingTime']}`",
        "- live动作：`无，研究只读`",
        "- 口径：`850U初始 / 每笔1% / 满成交 / 买价0.55`",
        "",
        "|配置|窗口|交易数|胜/负|胜率|盈亏|期末资金|最大回撤|收益回撤比|月正收益|哈希|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for w, b in BASELINE_061.items():
        lines.append(
            f"|当前061|{w}|{b['trades']}|{b['wins']}/{b['losses']}|{b['winRatePct']:.2f}%|{b['compoundPnl']:,.2f}|{b['endingBankroll']:,.2f}|{b['maxDrawdownUsd']:,.2f}|{b['returnDrawdownRatio']:.2f}|{b['monthlyPositiveRatio']:.2%}|`{b['setHash']}`|"
        )
    cand = payload.get("bestCandidate")
    if cand:
        for r in cand.get("rows", []):
            lines.append(
                f"|校准路由最强|{r['window']}|{r['trades']}|{r['wins']}/{r['losses']}|{r['winRatePct']:.2f}%|{r['compoundPnl']:,.2f}|{r['endingBankroll']:,.2f}|{r['maxDrawdownUsd']:,.2f}|{r['returnDrawdownRatio']:.2f}|{r['monthlyPositiveRatio']:.2%}|`{r['setHash']}`|"
            )
    lines += ["", "## 0.60 买价压力", "", "|配置|窗口|交易数|胜率|盈亏|最大回撤|", "|---|---:|---:|---:|---:|---:|"]
    if cand:
        for r in cand.get("stressRows", []):
            lines.append(f"|校准路由最强|{r['window']}|{r['trades']}|{r['winRatePct']:.2f}%|{r['compoundPnl']:,.2f}|{r['maxDrawdownUsd']:,.2f}|")
    lines += ["", "## 历史归档真实单纯预测复核", "", "|范围|旧真实市场数|选中|胜/负|胜率|盈亏|最大回撤|", "|---|---:|---:|---:|---:|---:|---:|"]
    for a in payload.get("archiveRows", []):
        if "error" in a:
            lines.append(f"|{a.get('scope')}|-|-|-|-|错误：{a.get('error')}|-|")
        else:
            lines.append(
                f"|{a.get('scope')}|{a.get('oldRealMarkets')}|{a.get('selectedTrades')}|{a.get('wins','-')}/{a.get('losses','-')}|{float(a.get('winRatePct',0)):.2f}%|{float(a.get('compoundPnl',0)):,.2f}|{float(a.get('maxDrawdownUsd',0)):,.2f}|"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    enriched, atom_store, period_vals = build_truth()
    audit = run_audit(period_vals, atom_store)
    if not audit["passed"]:
        raise SystemExit("061 baseline replay audit failed; refusing to run router search")

    feature_modes = ["base_num", "atoms_core", "atoms_all"]
    models = ["logit_l2_0.1", "logit_l2_0.3", "logit_l2_1.0", "logit_l1_0.1", "hgb_shallow", "rf_shallow"]
    if LGBMClassifier is not None:
        models.append("lgbm_shallow")

    policies = [
        Policy(scope="base", prob_min=p, score_min=s, daily_cap=cap, require_061=req)
        for req in [False, True]
        for p in np.round(np.arange(0.52, 0.641, 0.005), 3)
        for s in [0.55, 0.56, 0.57, 0.58]
        for cap in [None, 24, 32, 48]
    ]

    candidates: list[dict[str, Any]] = []
    model_meta: list[dict[str, Any]] = []
    for feature_mode in feature_modes:
        for model_name in models:
            try:
                preds = {}
                metas = []
                for window in ["180d", "365d"]:
                    pred, meta = fit_predict_one_window(period_vals, atom_store, window, feature_mode, model_name)
                    preds[window] = pred
                    metas.append(meta)
                model_meta.extend(metas)
                for policy in policies:
                    c = evaluate_router_candidate(preds, policy, feature_mode, model_name)
                    ok, reasons = pass_gate(c)
                    c["passed"] = ok
                    c["failReasons"] = reasons
                    c["score"] = score_candidate(c)
                    candidates.append(c)
            except Exception as exc:
                candidates.append({"name": stable_hash({"featureMode": feature_mode, "model": model_name, "error": repr(exc)}), "featureMode": feature_mode, "model": model_name, "error": repr(exc)})

    valid = [c for c in candidates if "rows" in c]
    valid.sort(key=score_candidate, reverse=True)
    strict = [c for c in valid if c.get("passed")]
    selected = strict[0] if strict else (valid[0] if valid else None)
    archive_eval = archive_rows(selected, {"180d": fit_predict_one_window(period_vals, atom_store, "180d", selected["featureMode"], selected["model"])[0]}) if selected else []
    payload = {
        "generatedAt": now_iso(),
        "beijingTime": bj_now(),
        "researchOnlyNoLiveChange": True,
        "timePolicy": "closed_candle_available_at_v2",
        "audit": audit,
        "modelMeta": model_meta,
        "baseline061": BASELINE_061,
        "evaluatedCandidates": len(candidates),
        "strictPass": len(strict),
        "topCandidates": valid[:100],
        "bestCandidate": selected,
        "archiveRows": archive_eval,
        "verdict": {
            "status": "candidate_beats_061" if selected and selected.get("passed") else "no_candidate_beats_061",
            "message": "校准路由找到严格超过当前061的候选，只能进影子验证，不改真钱。"
            if selected and selected.get("passed")
            else "校准路由未找到严格超过当前061的候选；当前真钱继续保留061。",
        },
    }
    write_json(OUT_LEADERBOARD_JSON, payload)
    write_text(
        OUT_LEADERBOARD_MD,
        "# 061 分数校准路由：排行榜\n\n"
        + f"- 北京时间：`{payload['beijingTime']}`\n"
        + f"- 评估候选：`{payload['evaluatedCandidates']}`\n"
        + f"- 严格通过：`{payload['strictPass']}`\n\n"
        + "|排名|模型|特征|策略|180天 盈亏/胜率/回撤|365天 盈亏/胜率/回撤|通过|\n"
        + "|---:|---|---|---|---:|---:|---|\n"
        + "\n".join(
            f"|{i+1}|{c.get('model')}|{c.get('featureMode')}|`{c.get('policy')}`|"
            f"{c['rows'][0]['compoundPnl']:,.2f}/{c['rows'][0]['winRatePct']:.2f}%/{c['rows'][0]['maxDrawdownUsd']:,.2f}|"
            f"{c['rows'][1]['compoundPnl']:,.2f}/{c['rows'][1]['winRatePct']:.2f}%/{c['rows'][1]['maxDrawdownUsd']:,.2f}|"
            f"{c.get('passed')}|"
            for i, c in enumerate(valid[:30])
        )
        + "\n",
    )
    compare = {"generatedAt": now_iso(), "beijingTime": bj_now(), "baseline061": BASELINE_061, "bestCandidate": selected, "archiveRows": archive_eval}
    write_json(OUT_COMPARE_JSON, compare)
    write_text(OUT_COMPARE_MD, format_compare(compare))
    write_json(OUT_VERDICT_JSON, payload["verdict"] | {"bestCandidate": selected})
    write_text(OUT_VERDICT_MD, f"# 061 分数校准路由：唯一结论\n\n- 状态：`{payload['verdict']['status']}`\n- 结论：{payload['verdict']['message']}\n")


if __name__ == "__main__":
    main()
