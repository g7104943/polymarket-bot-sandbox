#!/usr/bin/env python3
from __future__ import annotations

"""Fuller score-calibration router search for ETH061.

Research only. Does not touch live configs, orders, ledgers, claim state, or
monitor state.

Compared with the first calibration router, this expands:
  - model families and hyperparameters
  - probability calibration strength
  - score/probability policies
  - daily caps
  - direction-specific and combined score/probability gates

Locked metric:
  - 850U initial bankroll
  - stake 1% current bankroll
  - buy price 0.55 primary, full fill
  - closed_candle_available_at_v2 inherited from audited 061 universe
"""

import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    os.environ.setdefault(key, os.environ.get("TOP159_061_CALIB_FULL_THREADS", "1"))

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None  # type: ignore

ROOT = Path("/Users/mac/polyfun")
REPORTS = ROOT / "reports"
BASE_ROUTER = ROOT / "polyfun-next" / "scripts" / "run_top159_061_score_calibration_router.py"

START_BANKROLL = 850.0
STAKE_PCT = 0.01
BUY_PRICE = 0.55
STRESS_BUY_PRICE = 0.60
RNG_SEED = 20260510

OUT_AUDIT_MD = REPORTS / "top159_061_score_calibration_router_full_hyperopt_bug_audit_latest.md"
OUT_AUDIT_JSON = REPORTS / "top159_061_score_calibration_router_full_hyperopt_bug_audit_latest.json"
OUT_LEADERBOARD_MD = REPORTS / "top159_061_score_calibration_router_full_hyperopt_leaderboard_latest.md"
OUT_LEADERBOARD_JSON = REPORTS / "top159_061_score_calibration_router_full_hyperopt_leaderboard_latest.json"
OUT_COMPARE_MD = REPORTS / "top159_061_score_calibration_router_full_hyperopt_061_compare_latest.md"
OUT_COMPARE_JSON = REPORTS / "top159_061_score_calibration_router_full_hyperopt_061_compare_latest.json"
OUT_VERDICT_MD = REPORTS / "top159_061_score_calibration_router_full_hyperopt_unique_verdict_latest.md"
OUT_VERDICT_JSON = REPORTS / "top159_061_score_calibration_router_full_hyperopt_unique_verdict_latest.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base = load_module("score_calibration_router_base", BASE_ROUTER)


BASELINE_061 = base.BASELINE_061

_PERIOD_VALS: dict[str, pd.DataFrame] | None = None
_ATOM_STORE: dict[str, pd.DataFrame] | None = None
_POLICIES: list["Policy"] | None = None


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


def make_model(cfg: dict[str, Any]):
    kind = cfg["kind"]
    if kind == "logit":
        return make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(
                C=float(cfg["C"]),
                penalty=str(cfg.get("penalty", "l2")),
                max_iter=1200,
                solver="liblinear",
                class_weight=cfg.get("class_weight"),
            ),
        )
    if kind == "hgb":
        return HistGradientBoostingClassifier(
            max_leaf_nodes=int(cfg["max_leaf_nodes"]),
            learning_rate=float(cfg["learning_rate"]),
            max_iter=int(cfg["max_iter"]),
            l2_regularization=float(cfg["l2"]),
            min_samples_leaf=int(cfg.get("min_samples_leaf", 40)),
            random_state=RNG_SEED,
        )
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=int(cfg["n_estimators"]),
            max_depth=int(cfg["max_depth"]),
            min_samples_leaf=int(cfg["min_samples_leaf"]),
            class_weight="balanced_subsample",
            random_state=RNG_SEED,
            n_jobs=1,
        )
    if kind == "extra":
        return ExtraTreesClassifier(
            n_estimators=int(cfg["n_estimators"]),
            max_depth=int(cfg["max_depth"]),
            min_samples_leaf=int(cfg["min_samples_leaf"]),
            class_weight="balanced",
            random_state=RNG_SEED,
            n_jobs=1,
        )
    if kind == "lgbm" and LGBMClassifier is not None:
        return LGBMClassifier(
            n_estimators=int(cfg["n_estimators"]),
            learning_rate=float(cfg["learning_rate"]),
            num_leaves=int(cfg["num_leaves"]),
            min_child_samples=int(cfg["min_child_samples"]),
            subsample=float(cfg.get("subsample", 0.85)),
            colsample_bytree=float(cfg.get("colsample_bytree", 0.85)),
            reg_lambda=float(cfg.get("reg_lambda", 1.0)),
            random_state=RNG_SEED,
            n_jobs=1,
            verbose=-1,
        )
    raise ValueError(cfg)


def model_grid() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for c in [0.03, 0.05, 0.08, 0.1, 0.2, 0.35, 0.6, 1.0, 1.8, 3.0]:
        rows.append({"kind": "logit", "penalty": "l2", "C": c, "class_weight": "balanced"})
    for c in [0.03, 0.05, 0.08, 0.1, 0.2, 0.35, 0.6, 1.0]:
        rows.append({"kind": "logit", "penalty": "l1", "C": c, "class_weight": "balanced"})
    for leaves in [8, 12, 16, 24]:
        for lr, iters in [(0.02, 220), (0.035, 160), (0.055, 110)]:
            rows.append({"kind": "hgb", "max_leaf_nodes": leaves, "learning_rate": lr, "max_iter": iters, "l2": 0.5, "min_samples_leaf": 60})
    for depth, leaf in [(4, 80), (5, 80), (6, 100)]:
        rows.append({"kind": "rf", "n_estimators": 160, "max_depth": depth, "min_samples_leaf": leaf})
        rows.append({"kind": "extra", "n_estimators": 180, "max_depth": depth, "min_samples_leaf": leaf})
    if LGBMClassifier is not None:
        for leaves in [8, 12, 16, 24]:
            for lr, n in [(0.018, 260), (0.028, 180), (0.045, 120)]:
                rows.append({"kind": "lgbm", "n_estimators": n, "learning_rate": lr, "num_leaves": leaves, "min_child_samples": 80, "reg_lambda": 2.0})
    return rows


@dataclass(frozen=True)
class Policy:
    mode: str
    prob_min: float = 0.535
    score_min: float = 0.55
    daily_cap: int | None = None
    require_061: bool = True
    up_prob_min: float | None = None
    down_prob_min: float | None = None
    combo_min: float | None = None
    combo_alpha: float = 0.0
    low_score_cut: float | None = None
    low_prob_extra: float = 0.0

    def key(self, feature_mode: str, model_key: str, calib_c: float) -> str:
        return stable_hash({"feature": feature_mode, "model": model_key, "calibC": calib_c, "policy": self.__dict__})


def policy_grid(stage: str = "full") -> list[Policy]:
    rows: list[Policy] = []
    if stage == "coarse":
        caps = [None, 20, 24, 32, 48]
        for require_061 in [True, False]:
            for cap in caps:
                for p in [0.515, 0.525, 0.535, 0.545, 0.555, 0.565, 0.58, 0.60]:
                    for s in [0.55, 0.555, 0.56, 0.57, 0.58]:
                        rows.append(Policy(mode="plain", prob_min=p, score_min=s, daily_cap=cap, require_061=require_061))
        for cap in [None, 24, 32]:
            for require_061 in [True, False]:
                for up in [0.525, 0.545, 0.565, 0.58]:
                    for down in [0.525, 0.545, 0.565, 0.58]:
                        rows.append(Policy(mode="directional", score_min=0.55, daily_cap=cap, require_061=require_061, up_prob_min=up, down_prob_min=down))
        for cap in [None, 24, 32]:
            for require_061 in [True, False]:
                for alpha in [0.5, 1.0, 1.5]:
                    for cm in [0.545, 0.555, 0.565, 0.58]:
                        rows.append(Policy(mode="combo", score_min=0.55, daily_cap=cap, require_061=require_061, combo_min=cm, combo_alpha=alpha))
        for cap in [None, 24, 32]:
            for p in [0.525, 0.545, 0.565]:
                for cut in [0.57, 0.60]:
                    for extra in [0.01, 0.02]:
                        rows.append(Policy(mode="low_score_extra", prob_min=p, score_min=0.55, daily_cap=cap, require_061=True, low_score_cut=cut, low_prob_extra=extra))
        # Deduplicate below.
    else:
        if stage != "full":
            raise ValueError(stage)
        caps = [None, 12, 16, 20, 24, 28, 32, 40, 48]
        prob_grid = [round(x, 4) for x in np.arange(0.505, 0.651, 0.005)]
        fine_prob = [round(x, 4) for x in np.arange(0.5225, 0.5926, 0.0025)]
        score_grid = [0.55, 0.5525, 0.555, 0.5575, 0.56, 0.565, 0.57, 0.575, 0.58, 0.59, 0.60]
        for require_061 in [True, False]:
            for cap in caps:
                for p in sorted(set(prob_grid + fine_prob)):
                    for s in score_grid:
                        rows.append(Policy(mode="plain", prob_min=p, score_min=s, daily_cap=cap, require_061=require_061))
        for cap in [None, 16, 20, 24, 32, 48]:
            for require_061 in [True, False]:
                for up in [0.52, 0.53, 0.54, 0.55, 0.565, 0.58, 0.60]:
                    for down in [0.52, 0.53, 0.54, 0.55, 0.565, 0.58, 0.60]:
                        rows.append(Policy(mode="directional", score_min=0.55, daily_cap=cap, require_061=require_061, up_prob_min=up, down_prob_min=down))
        for cap in [None, 20, 24, 32, 48]:
            for require_061 in [True, False]:
                for alpha in [0.25, 0.5, 0.75, 1.0, 1.5]:
                    for cm in [0.535, 0.545, 0.555, 0.565, 0.575, 0.59, 0.61]:
                        rows.append(Policy(mode="combo", score_min=0.55, daily_cap=cap, require_061=require_061, combo_min=cm, combo_alpha=alpha))
        for cap in [None, 20, 24, 32, 48]:
            for p in [0.52, 0.53, 0.54, 0.55, 0.565]:
                for cut in [0.57, 0.58, 0.60, 0.62]:
                    for extra in [0.005, 0.01, 0.02, 0.03]:
                        rows.append(Policy(mode="low_score_extra", prob_min=p, score_min=0.55, daily_cap=cap, require_061=True, low_score_cut=cut, low_prob_extra=extra))
    # Deduplicate, because the grids intentionally overlap around the useful region.
    seen = set()
    out = []
    for r in rows:
        k = tuple(sorted(r.__dict__.items()))
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def fit_predict_one_window(period_vals, atom_store, window: str, feature_mode: str, model_cfg: dict[str, Any], calib_c: float):
    train_period = "gate_train_for_180d" if window == "180d" else "gate_train_for_365d"
    val_period = "validation_180d" if window == "180d" else "validation_365d"
    train_df = period_vals[train_period].reset_index(drop=True)
    val_df = period_vals[val_period].reset_index(drop=True)
    train_idx, calib_idx = base.split_train_calib(train_df)
    train_x_all = base.make_feature_frame(train_df, feature_mode, atom_store[train_period])
    val_x = base.make_feature_frame(val_df, feature_mode, atom_store[val_period])
    core_x = train_x_all.loc[train_idx]
    calib_x = train_x_all.loc[calib_idx]
    core_x, calib_x, val_x = base.align_features(core_x, calib_x, val_x)
    y_core = train_df.loc[train_idx, "won"].astype(bool).astype(int).to_numpy()
    y_calib = train_df.loc[calib_idx, "won"].astype(bool).astype(int).to_numpy()
    model = make_model(model_cfg)
    model.fit(core_x, y_core)
    raw_calib = model.predict_proba(calib_x)[:, 1]
    raw_val = model.predict_proba(val_x)[:, 1]
    calibrator = LogisticRegression(C=calib_c, penalty="l2", max_iter=1000, solver="lbfgs")
    calibrator.fit(raw_calib.reshape(-1, 1), y_calib)
    p_val = calibrator.predict_proba(raw_val.reshape(-1, 1))[:, 1]
    out = val_df.copy()
    out["router_prob"] = p_val
    out["current061_keep"] = base.current_061_mask(atom_store, val_period, val_df)
    meta = {"window": window, "modelCfg": model_cfg, "featureMode": feature_mode, "calibC": calib_c, "features": int(len(core_x.columns)), "trainRows": int(len(core_x)), "calibRows": int(len(calib_x))}
    return out, meta


def apply_policy(df: pd.DataFrame, policy: Policy) -> pd.DataFrame:
    score = pd.to_numeric(df["score15"], errors="coerce").fillna(0.0)
    prob = pd.to_numeric(df["router_prob"], errors="coerce").fillna(0.0)
    mask = score >= policy.score_min
    if policy.require_061:
        mask &= df["current061_keep"].astype(bool)
    if policy.mode == "plain":
        mask &= prob >= policy.prob_min
        rank_score = prob
    elif policy.mode == "directional":
        direction = df["direction"].astype(str).str.upper()
        req = np.where(direction == "UP", float(policy.up_prob_min), float(policy.down_prob_min))
        mask &= prob.to_numpy() >= req
        rank_score = prob
    elif policy.mode == "combo":
        combo = prob + policy.combo_alpha * (score - 0.55)
        mask &= combo >= float(policy.combo_min)
        rank_score = combo
    elif policy.mode == "low_score_extra":
        req = policy.prob_min + np.where(score < float(policy.low_score_cut), policy.low_prob_extra, 0.0)
        mask &= prob.to_numpy() >= req
        rank_score = prob
    else:
        raise ValueError(policy)
    selected = df[mask].copy()
    if policy.daily_cap and not selected.empty:
        selected["_rank_score"] = np.asarray(rank_score)[mask.to_numpy()]
        selected["bj_day"] = pd.to_datetime(selected["dt"], utc=True).dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m-%d")
        selected = (
            selected.sort_values(["bj_day", "_rank_score"], ascending=[True, False])
            .groupby("bj_day", group_keys=False)
            .head(policy.daily_cap)
            .drop(columns=["_rank_score"])
            .sort_values("dt")
        )
    return selected.sort_values("dt").reset_index(drop=True)


def fast_curve_metrics(rows: pd.DataFrame, name: str, window: str, buy_price: float) -> dict[str, Any]:
    selected = rows.sort_values("dt").reset_index(drop=True)
    n = int(len(selected))
    if n == 0:
        return base.curve_metrics(selected, name, window, buy_price)
    won = selected["won"].astype(bool).to_numpy()
    win_ret = 1.0 / float(buy_price) - 1.0
    step = 1.0 + STAKE_PCT * np.where(won, win_ret, -1.0)
    curve = START_BANKROLL * np.cumprod(step)
    equity = float(curve[-1])
    running_max = np.maximum.accumulate(curve)
    drawdowns = running_max - curve
    mxdd = float(drawdowns.max()) if len(drawdowns) else 0.0
    mxdd_pct = float((drawdowns / np.maximum(running_max, 1e-12)).max()) if len(drawdowns) else 0.0
    wins = int(won.sum())
    losses = int(n - wins)
    return {
        "name": name,
        "window": window,
        "buyPrice": float(buy_price),
        "trades": n,
        "wins": wins,
        "losses": losses,
        "winRatePct": round(100.0 * wins / n, 6),
        "compoundPnl": round(float(equity - START_BANKROLL), 6),
        "endingBankroll": round(equity, 6),
        "maxDrawdownUsd": round(mxdd, 6),
        "maxDrawdownPct": round(mxdd_pct * 100.0, 6),
        "returnDrawdownRatio": round(float(equity - START_BANKROLL) / mxdd, 6) if mxdd > 1e-12 else (999.0 if equity > START_BANKROLL else 0.0),
        "monthlyPositiveRatio": base.monthly_positive_ratio(curve, selected["dt"]),
        "setHash": base.stable_hash(selected[["dt", "pred_up15", "label_up", "won"]].to_dict("records")),
    }


def evaluate_candidate(preds: dict[str, pd.DataFrame], policy: Policy, feature_mode: str, model_key: str, model_cfg: dict[str, Any], calib_c: float) -> dict[str, Any]:
    name = policy.key(feature_mode, model_key, calib_c)
    rows = []
    stress_rows = []
    for w in ["180d", "365d"]:
        selected = apply_policy(preds[w], policy)
        r = fast_curve_metrics(selected, name, w, BUY_PRICE)
        s = fast_curve_metrics(selected, name, w, STRESS_BUY_PRICE)
        current061 = int(preds[w]["current061_keep"].astype(bool).sum())
        r.update({"retentionRateVs061": round(100.0 * len(selected) / max(1, current061), 6)})
        rows.append(r)
        stress_rows.append(s)
    return {"name": name, "featureMode": feature_mode, "modelKey": model_key, "modelCfg": model_cfg, "calibC": calib_c, "policy": policy.__dict__, "rows": rows, "stressRows": stress_rows}


def pass_gate(c: dict[str, Any]) -> tuple[bool, list[str]]:
    rows = {r["window"]: r for r in c["rows"]}
    reasons = []
    for w in ["180d", "365d"]:
        r = rows.get(w)
        b = BASELINE_061[w]
        if not r:
            reasons.append(f"{w}_missing")
            continue
        if r["trades"] < max(100 if w == "180d" else 200, int(b["trades"] * 0.45)):
            reasons.append(f"{w}_too_few")
        if r["compoundPnl"] <= b["compoundPnl"]:
            reasons.append(f"{w}_pnl_not_above_061")
        if w == "365d" and r["winRatePct"] < b["winRatePct"]:
            reasons.append("365d_winrate_below_061")
        if r["maxDrawdownUsd"] > b["maxDrawdownUsd"] * 1.05:
            reasons.append(f"{w}_drawdown_too_high")
    return not reasons, reasons


def score_candidate(c: dict[str, Any]) -> float:
    rows = {r["window"]: r for r in c["rows"]}
    if set(rows) != {"180d", "365d"}:
        return -1e18
    b180, b365 = BASELINE_061["180d"], BASELINE_061["365d"]
    return (
        (rows["180d"]["compoundPnl"] - b180["compoundPnl"]) / b180["compoundPnl"] * 3000
        + (rows["365d"]["compoundPnl"] - b365["compoundPnl"]) / b365["compoundPnl"] * 5000
        + (rows["365d"]["winRatePct"] - b365["winRatePct"]) * 50
        - max(0.0, rows["180d"]["maxDrawdownUsd"] / b180["maxDrawdownUsd"] - 1) * 500
        - max(0.0, rows["365d"]["maxDrawdownUsd"] / b365["maxDrawdownUsd"] - 1) * 500
    )


def run_model_task(task: tuple[str, dict[str, Any], float]) -> dict[str, Any]:
    """Fit one model/calibrator setup and return only its best candidates.

    The full grid can evaluate millions of policy variants. Keeping every row
    in memory is unnecessary; each worker keeps the strongest local candidates
    and returns aggregate counts.
    """
    if _PERIOD_VALS is None or _ATOM_STORE is None or _POLICIES is None:
        raise RuntimeError("worker globals not initialized")
    feature_mode, cfg, calib_c = task
    model_key = stable_hash(cfg)
    try:
        preds: dict[str, pd.DataFrame] = {}
        meta_pair = []
        for window in ["180d", "365d"]:
            p, m = fit_predict_one_window(_PERIOD_VALS, _ATOM_STORE, window, feature_mode, cfg, calib_c)
            preds[window] = p
            meta_pair.append(m)

        best: list[dict[str, Any]] = []
        strict_best: list[dict[str, Any]] = []
        strict_count = 0
        evaluated = 0
        for policy in _POLICIES:
            c = evaluate_candidate(preds, policy, feature_mode, model_key, cfg, calib_c)
            ok, reasons = pass_gate(c)
            c["passed"] = ok
            c["failReasons"] = reasons
            c["score"] = score_candidate(c)
            strict_count += int(ok)
            evaluated += 1
            best.append(c)
            if len(best) > 80:
                best.sort(key=score_candidate, reverse=True)
                del best[80:]
            if ok:
                strict_best.append(c)
                if len(strict_best) > 80:
                    strict_best.sort(key=score_candidate, reverse=True)
                    del strict_best[80:]
        best.sort(key=score_candidate, reverse=True)
        strict_best.sort(key=score_candidate, reverse=True)
        return {
            "ok": True,
            "evaluated": evaluated,
            "strict": strict_count,
            "top": best[:80],
            "strictTop": strict_best[:80],
            "meta": meta_pair,
            "modelKey": model_key,
        }
    except Exception as exc:
        return {
            "ok": False,
            "evaluated": 1,
            "strict": 0,
            "top": [],
            "strictTop": [],
            "meta": [],
            "error": {
                "name": stable_hash({"cfg": cfg, "feature": feature_mode, "calib": calib_c, "error": repr(exc)}),
                "featureMode": feature_mode,
                "modelKey": model_key,
                "modelCfg": cfg,
                "calibC": calib_c,
                "error": repr(exc)[:800],
            },
        }


def init_worker(policies: list[Policy]) -> None:
    global _PERIOD_VALS, _ATOM_STORE, _POLICIES
    _POLICIES = policies
    _enriched, _ATOM_STORE, _PERIOD_VALS = base.build_truth()


def format_compare(payload: dict[str, Any]) -> str:
    lines = [
        "# 061 分数校准路由全量超参：公平对比",
        "",
        f"- 北京时间：`{payload['beijingTime']}`",
        "- live动作：`无，研究只读`",
        "- 口径：`850U初始 / 每笔1% / 满成交 / 买价0.55`",
        "",
        "|配置|窗口|交易数|胜/负|胜率|盈亏|期末资金|最大回撤|收益回撤比|月正收益|哈希|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for w, b in BASELINE_061.items():
        lines.append(f"|当前061|{w}|{b['trades']}|{b['wins']}/{b['losses']}|{b['winRatePct']:.2f}%|{b['compoundPnl']:,.2f}|{b['endingBankroll']:,.2f}|{b['maxDrawdownUsd']:,.2f}|{b['returnDrawdownRatio']:.2f}|{b['monthlyPositiveRatio']:.2%}|`{b['setHash']}`|")
    c = payload.get("bestCandidate")
    if c:
        for r in c["rows"]:
            lines.append(f"|全量校准路由最强|{r['window']}|{r['trades']}|{r['wins']}/{r['losses']}|{r['winRatePct']:.2f}%|{r['compoundPnl']:,.2f}|{r['endingBankroll']:,.2f}|{r['maxDrawdownUsd']:,.2f}|{r['returnDrawdownRatio']:.2f}|{r['monthlyPositiveRatio']:.2%}|`{r['setHash']}`|")
    lines += ["", "## 0.60 买价压力", "", "|配置|窗口|交易数|胜率|盈亏|最大回撤|", "|---|---:|---:|---:|---:|---:|"]
    if c:
        for r in c["stressRows"]:
            lines.append(f"|全量校准路由最强|{r['window']}|{r['trades']}|{r['winRatePct']:.2f}%|{r['compoundPnl']:,.2f}|{r['maxDrawdownUsd']:,.2f}|")
    return "\n".join(lines) + "\n"


def run_audit(period_vals, atom_store) -> dict[str, Any]:
    rows = base.baseline_rows(period_vals, atom_store)
    checks = []
    for r in rows:
        b = BASELINE_061[r["window"]]
        checks.append({"window": r["window"], "trades": r["trades"], "pnl": r["compoundPnl"], "expectedPnl": b["compoundPnl"], "passed": r["trades"] == b["trades"] and abs(r["compoundPnl"] - b["compoundPnl"]) < 1e-6})
    audit = {"generatedAt": base.now_iso(), "beijingTime": base.bj_now(), "researchOnlyNoLiveChange": True, "timePolicy": "closed_candle_available_at_v2", "baselineReplay": checks, "passed": all(x["passed"] for x in checks)}
    write_json(OUT_AUDIT_JSON, audit)
    write_text(OUT_AUDIT_MD, "# 061 分数校准路由全量超参：审计\n\n" + f"- 北京时间：`{audit['beijingTime']}`\n- 当前061复现：`{checks}`\n- 审计通过：`{audit['passed']}`\n")
    return audit


def run_task_batch(tasks: list[tuple[str, dict[str, Any], float]], policies: list[Policy], workers: int, checkpoint_label: str) -> tuple[list[dict[str, Any]], int, int, list[dict[str, Any]], list[dict[str, Any]]]:
    global _POLICIES
    _POLICIES = policies
    candidates: list[dict[str, Any]] = []
    strict_candidates: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    metas: list[dict[str, Any]] = []
    evaluated_total = 0
    strict_total = 0
    ctx = mp.get_context(os.environ.get("TOP159_061_CALIB_MP_MODE", "spawn"))
    use_initializer = ctx.get_start_method() != "fork"
    kwargs = {"initializer": init_worker, "initargs": (policies,)} if use_initializer else {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, **kwargs) as ex:
        futs = [ex.submit(run_model_task, t) for t in tasks]
        for idx, fut in enumerate(as_completed(futs), start=1):
            res = fut.result()
            evaluated_total += int(res.get("evaluated", 0))
            strict_total += int(res.get("strict", 0))
            candidates.extend(res.get("top", []))
            strict_candidates.extend(res.get("strictTop", []))
            metas.extend(res.get("meta", []))
            if not res.get("ok"):
                errors.append(res.get("error", {}))
            if idx % 1 == 0 or idx == len(futs):
                candidates.sort(key=score_candidate, reverse=True)
                del candidates[800:]
                strict_candidates.sort(key=score_candidate, reverse=True)
                del strict_candidates[800:]
                checkpoint = {
                    "generatedAt": base.now_iso(),
                    "beijingTime": base.bj_now(),
                    "stage": checkpoint_label,
                    "doneTasks": idx,
                    "totalTasks": len(futs),
                    "evaluatedCandidates": evaluated_total,
                    "strictPass": strict_total,
                    "topCandidates": candidates[:50],
                    "topStrictCandidates": strict_candidates[:50],
                    "errors": errors[:20],
                }
                write_json(REPORTS / "top159_061_score_calibration_router_full_hyperopt_checkpoint_latest.json", checkpoint)
    candidates.sort(key=score_candidate, reverse=True)
    strict_candidates.sort(key=score_candidate, reverse=True)
    merged: list[dict[str, Any]] = []
    seen = set()
    for c in strict_candidates[:800] + candidates[:800]:
        k = c.get("name")
        if k in seen:
            continue
        seen.add(k)
        merged.append(c)
    return merged[:1600], evaluated_total, strict_total, metas, errors


def main() -> None:
    global _PERIOD_VALS, _ATOM_STORE, _POLICIES
    REPORTS.mkdir(parents=True, exist_ok=True)
    enriched, atom_store, period_vals = base.build_truth()
    audit = run_audit(period_vals, atom_store)
    if not audit["passed"]:
        raise SystemExit("baseline audit failed")
    feature_modes = ["base_num", "atoms_core", "atoms_all"]
    coarse_policies = policy_grid("coarse")
    full_policies = policy_grid("full")
    model_cfgs = model_grid()
    calib_grid = [0.08, 0.15, 0.25, 0.45, 0.75]
    _PERIOD_VALS = period_vals
    _ATOM_STORE = atom_store
    tasks = [(feature_mode, cfg, calib_c) for feature_mode in feature_modes for cfg in model_cfgs for calib_c in calib_grid]
    workers = max(1, int(os.environ.get("TOP159_061_CALIB_FULL_WORKERS", "4")))
    coarse_candidates, coarse_eval, coarse_strict, coarse_metas, coarse_errors = run_task_batch(tasks, coarse_policies, workers, "coarse_model_screen")
    seen_task_keys = []
    seen = set()
    for c in coarse_candidates:
        k = (c.get("featureMode"), json.dumps(c.get("modelCfg"), sort_keys=True, default=str), c.get("calibC"))
        if k in seen:
            continue
        seen.add(k)
        seen_task_keys.append(k)
        if len(seen_task_keys) >= int(os.environ.get("TOP159_061_CALIB_FULL_TOP_TASKS", "40")):
            break
    focused_tasks: list[tuple[str, dict[str, Any], float]] = []
    for feature_mode, cfg, calib_c in tasks:
        k = (feature_mode, json.dumps(cfg, sort_keys=True, default=str), calib_c)
        if k in seen:
            focused_tasks.append((feature_mode, cfg, calib_c))
    candidates, full_eval, full_strict, full_metas, full_errors = run_task_batch(focused_tasks, full_policies, workers, "focused_full_policy_search")
    evaluated_total = coarse_eval + full_eval
    strict_total = coarse_strict + full_strict
    metas = coarse_metas[:25] + full_metas[:75]
    errors = coarse_errors + full_errors
    valid = [c for c in candidates if "rows" in c]
    valid.sort(key=score_candidate, reverse=True)
    strict = [c for c in valid if c.get("passed")]
    selected = strict[0] if strict else (valid[0] if valid else None)
    payload = {
        "generatedAt": base.now_iso(),
        "beijingTime": base.bj_now(),
        "researchOnlyNoLiveChange": True,
        "audit": audit,
        "evaluatedCandidates": evaluated_total,
        "strictPass": strict_total,
        "workers": workers,
        "modelCount": len(model_cfgs),
        "coarsePolicyCount": len(coarse_policies),
        "fullPolicyCount": len(full_policies),
        "focusedTaskCount": len(focused_tasks),
        "calibrationCount": len(calib_grid),
        "modelMeta": metas[:50],
        "errors": errors[:50],
        "baseline061": BASELINE_061,
        "topCandidates": valid[:200],
        "bestCandidate": selected,
        "verdict": {
            "status": "candidate_beats_061" if selected and selected.get("passed") else "no_candidate_beats_061",
            "message": "全量校准路由找到严格超过当前061的候选，只进入影子验证，不改真钱。"
            if selected and selected.get("passed")
            else "全量校准路由没有找到严格超过当前061的候选；当前真钱继续保留061。",
        },
    }
    write_json(OUT_LEADERBOARD_JSON, payload)
    write_text(
        OUT_LEADERBOARD_MD,
        "# 061 分数校准路由全量超参：排行榜\n\n"
        + f"- 北京时间：`{payload['beijingTime']}`\n"
        + f"- 模型数：`{payload['modelCount']}`；粗筛策略数：`{payload['coarsePolicyCount']}`；精搜策略数：`{payload['fullPolicyCount']}`；精搜模型配置数：`{payload['focusedTaskCount']}`；校准器数：`{payload['calibrationCount']}`\n"
        + f"- 评估候选：`{payload['evaluatedCandidates']}`；严格通过：`{payload['strictPass']}`\n\n"
        + "|排名|特征|模型|校准C|策略|180天 盈亏/胜率/回撤|365天 盈亏/胜率/回撤|通过|\n"
        + "|---:|---|---|---:|---|---:|---:|---|\n"
        + "\n".join(
            f"|{i+1}|{c.get('featureMode')}|`{c.get('modelCfg')}`|{c.get('calibC')}|`{c.get('policy')}`|"
            f"{c['rows'][0]['compoundPnl']:,.2f}/{c['rows'][0]['winRatePct']:.2f}%/{c['rows'][0]['maxDrawdownUsd']:,.2f}|"
            f"{c['rows'][1]['compoundPnl']:,.2f}/{c['rows'][1]['winRatePct']:.2f}%/{c['rows'][1]['maxDrawdownUsd']:,.2f}|{c.get('passed')}|"
            for i, c in enumerate(valid[:40])
        )
        + "\n",
    )
    compare = {"generatedAt": base.now_iso(), "beijingTime": base.bj_now(), "baseline061": BASELINE_061, "bestCandidate": selected}
    write_json(OUT_COMPARE_JSON, compare)
    write_text(OUT_COMPARE_MD, format_compare(compare))
    write_json(OUT_VERDICT_JSON, payload["verdict"] | {"bestCandidate": selected})
    write_text(OUT_VERDICT_MD, f"# 061 分数校准路由全量超参：唯一结论\n\n- 状态：`{payload['verdict']['status']}`\n- 结论：{payload['verdict']['message']}\n")


if __name__ == "__main__":
    main()
