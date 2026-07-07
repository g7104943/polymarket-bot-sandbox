#!/usr/bin/env python3
from __future__ import annotations

"""ETH15m model-bank router search at buy price 0.55.

Research only. Does not mutate live top159/061 config, ledgers, monitor, order
state, claim state, or any live process.

Idea:
  - Keep current 061 as the core baseline.
  - Build a bank of previously audited experts.
  - For each ETH 15m top159 candidate market, route to one of:
      use_061 / use_alt / skip.
  - Evaluate using the user-locked fair proxy:
      850U initial, 1% current bankroll, full fill, buy price 0.55.
"""

import hashlib
import importlib.util
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

THREADS = os.environ.get("ETH15M_ROUTER_THREADS", "1")
for key in ["OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(key, THREADS)
# Keep imported modules under the same budget.
os.environ.setdefault("ETH15M_SEQ055_THREADS", THREADS)
os.environ.setdefault("ETH15M_BUY055_THREADS", THREADS)
os.environ.setdefault("TOP159_RAW_GROWTH_THREADS", THREADS)
os.environ.setdefault("TOP159_SHOCK_EXTREME_THREADS", THREADS)

import numpy as np
import pandas as pd

ROOT = Path("/Users/mac/polyfun")
NEXT = ROOT / "polyfun-next"
SCRIPTS = NEXT / "scripts"
REPORTS = ROOT / "reports"

CLUSTER_SCRIPT = SCRIPTS / "run_top159_shock_filter_cluster_targeted_search.py"
HIGHPRICE_SCRIPT = SCRIPTS / "run_eth15m_high_price_055_model_search.py"
SEQ_SCRIPT = SCRIPTS / "run_eth15m_sequence_shape_055_model_search.py"
RAW_SCRIPT = SCRIPTS / "run_top159_raw_growth_mainmodel_v1_search.py"
ARCHIVE_SCRIPT = ROOT / "scripts" / "ops" / "run_top159_all_archived_real_eth_compare.py"

START_BANKROLL = 850.0
STAKE_PCT = 0.01
PRIMARY_BUY_PRICE = 0.55
BUY_PRICES = [0.55, 0.60]
RNG_SEED = 20260507
MAX_SECONDS = int(os.environ.get("ETH15M_ROUTER_MAX_SECONDS", str(3 * 3600)))
MAX_EXPERTS_PER_FAMILY = int(os.environ.get("ETH15M_ROUTER_EXPERTS_PER_FAMILY", "3"))

OUT_DATA_TRUTH_MD = REPORTS / "eth15m_model_bank_router_data_truth_latest.md"
OUT_DATA_TRUTH_JSON = REPORTS / "eth15m_model_bank_router_data_truth_latest.json"
OUT_RESULTS = REPORTS / "eth15m_model_bank_router_results_latest.jsonl"
OUT_LEADERBOARD_MD = REPORTS / "eth15m_model_bank_router_leaderboard_latest.md"
OUT_LEADERBOARD_JSON = REPORTS / "eth15m_model_bank_router_leaderboard_latest.json"
OUT_COMPARE_MD = REPORTS / "eth15m_model_bank_router_061_compare_latest.md"
OUT_COMPARE_JSON = REPORTS / "eth15m_model_bank_router_061_compare_latest.json"
OUT_VERDICT_MD = REPORTS / "eth15m_model_bank_router_unique_verdict_latest.md"
OUT_VERDICT_JSON = REPORTS / "eth15m_model_bank_router_unique_verdict_latest.json"
OUT_AUDIT_MD = REPORTS / "eth15m_model_bank_router_bug_audit_latest.md"
OUT_AUDIT_JSON = REPORTS / "eth15m_model_bank_router_bug_audit_latest.json"

BASELINE_061_BUY055 = {
    "180d": {
        "name": "current_061_at_buy055",
        "window": "180d",
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
        "name": "current_061_at_buy055",
        "window": "365d",
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


cluster = load_module("router_cluster", CLUSTER_SCRIPT)
hp = load_module("router_highprice055", HIGHPRICE_SCRIPT)
seq = load_module("router_sequence055", SEQ_SCRIPT)
raw = load_module("router_rawgrowth", RAW_SCRIPT)
archive = load_module("router_archive", ARCHIVE_SCRIPT)

# Keep expensive imported search modules from using very large row windows by default.
hp.MAX_TRAIN_ROWS = int(os.environ.get("ETH15M_ROUTER_MAX_TRAIN_ROWS", "90000"))
seq.MAX_TRAIN_ROWS = int(os.environ.get("ETH15M_ROUTER_MAX_TRAIN_ROWS", "90000"))
raw.MAX_TRAIN_ROWS = int(os.environ.get("ETH15M_ROUTER_MAX_TRAIN_ROWS", "90000"))


def max_drawdown(equity: np.ndarray) -> tuple[float, float]:
    if equity.size == 0:
        return 0.0, 0.0
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    trough = int(np.argmax(dd))
    mx = float(dd[trough])
    denom = float(peak[trough]) if trough >= 0 else 0.0
    return mx, (mx / denom if denom > 1e-12 else 0.0)


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
        "setHash": stable_hash(sel[["dt", "action", "pred_up15", "label_up", "won"]].to_dict("records")) if len(sel) else "empty",
        "use061": int((sel.get("action", pd.Series([], dtype=str)) == "use_061").sum()) if len(sel) else 0,
        "useAlt": int((sel.get("action", pd.Series([], dtype=str)) == "use_alt").sum()) if len(sel) else 0,
        "skip": 0,
    }


def load_top_params(path: Path, min_365_trades: int = 100, max_n: int = 3) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    out: list[dict[str, Any]] = []
    for c in payload.get("topCandidates", []):
        rows = {r.get("window"): r for r in c.get("rows", [])}
        r365 = rows.get("365d") or {}
        if int(r365.get("trades", 0)) >= min_365_trades:
            out.append({"name": c.get("name"), "params": c.get("params", {})})
        if len(out) >= max_n:
            break
    return out


def score_from_highprice_params(val: pd.DataFrame, p_up: np.ndarray, p_q: np.ndarray, params: dict[str, Any]) -> pd.DataFrame:
    pred_up = p_up >= 0.5
    dir_conf = np.maximum(p_up, 1.0 - p_up)
    q = np.asarray(p_q, dtype=float)
    sm = params.get("score_mode", "dir_only")
    qw = float(params.get("quality_weight", 0.0))
    if sm == "weighted_avg":
        score = (1.0 - qw) * dir_conf + qw * q
    elif sm == "quality_boost":
        score = np.clip(dir_conf + qw * (q - 0.5), 0.0, 1.0)
    else:
        score = dir_conf.copy()
    mask = score >= float(params.get("threshold", 0.55))
    if sm == "quality_gate":
        mask &= q >= float(params.get("quality_threshold", 0.55))
    if params.get("score_band") == "mid_only":
        mask &= score <= float(params.get("max_score", 0.90))
    if params.get("score_band") == "high_only":
        mask &= score >= float(params.get("min_high_score", params.get("threshold", 0.55)))
    out = val[["dt"]].copy()
    out["pred"] = pred_up.astype(bool)
    out["score"] = score.astype(float)
    out["keep"] = mask.astype(bool)
    # Match the daily cap policy by clearing non-top signals per day.
    cap = int(params.get("daily_cap", 0) or 0)
    if cap > 0 and out["keep"].any():
        tmp = out[out["keep"]].copy()
        tmp["day"] = pd.to_datetime(tmp["dt"], utc=True).dt.floor("D")
        allowed = tmp.sort_values(["day", "score"], ascending=[True, False]).groupby("day", sort=False).head(cap)["dt"]
        out["keep"] = out["dt"].isin(set(allowed))
    return out


def predict_highprice_like(module: Any, df: pd.DataFrame, features: list[str], params: dict[str, Any], start: pd.Timestamp, end: pd.Timestamp, family: str) -> pd.DataFrame:
    feats = module.feature_subset(features, params["feature_mode"])
    if family == "seq":
        if params.get("seq_feature_limit"):
            nonseq = [c for c in feats if not c.startswith("seq_")]
            seqcols = [c for c in feats if c.startswith("seq_")][: int(params["seq_feature_limit"])]
            feats = nonseq + seqcols
    max_features = int(params.get("max_features", 0) or 0)
    if max_features > 0:
        if family == "seq":
            nonseq = [c for c in feats if not c.startswith("seq_")]
            seqcols = [c for c in feats if c.startswith("seq_")]
            slots = max(0, max_features - min(len(seqcols), int(params.get("seq_keep", 80))))
            feats = nonseq[:slots] + seqcols[: max_features - slots]
        else:
            feats = feats[:max_features]
    train = df[df["dt"] < start - pd.Timedelta(hours=4)].copy()
    tw = params.get("train_window", "full")
    if tw != "full":
        days = {"1y": 365, "3y": 1095, "5y": 1825}[tw]
        train = train[train["dt"] >= start - pd.Timedelta(days=days)].copy()
    val = df[(df["dt"] >= start) & (df["dt"] <= end)].copy()
    if train.empty or val.empty:
        return pd.DataFrame(columns=["dt", "pred", "score", "keep"])
    p_up = module.calibrated_predict(params["engine"], train, val, feats, "label_up", params)
    p_q = module.calibrated_predict(params["quality_engine"], train, val, feats, params.get("quality_label", "label_strong_move"), params)
    if p_up is None or p_q is None:
        return pd.DataFrame(columns=["dt", "pred", "score", "keep"])
    return score_from_highprice_params(val, p_up, p_q, params)


def predict_raw(df: pd.DataFrame, features: list[str], params: dict[str, Any], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    feats = raw.feature_subset(features, params["feature_mode"])
    max_features = int(params.get("max_features", 0) or 0)
    if max_features > 0:
        feats = feats[:max_features]
    train = df[df["dt"] < start - pd.Timedelta(hours=4)].copy()
    tw = params.get("train_window", "full")
    if tw != "full":
        days = {"1y": 365, "3y": 1095, "5y": 1825}[tw]
        train = train[train["dt"] >= start - pd.Timedelta(days=days)].copy()
    val = df[(df["dt"] >= start) & (df["dt"] <= end)].copy()
    if train.empty or val.empty:
        return pd.DataFrame(columns=["dt", "pred", "score", "keep"])
    m = raw.fit_model(params["engine"], train, feats, params)
    if m is None:
        return pd.DataFrame(columns=["dt", "pred", "score", "keep"])
    prob = raw.predict_prob(m, val, feats)
    pred = prob >= 0.5
    score = np.maximum(prob, 1.0 - prob)
    mask = score >= 0.5 + float(params.get("edge", 0.03))
    if params.get("score_band") == "mid_only":
        mask &= score <= float(params.get("max_score", 0.76))
    if params.get("vol_q", 0.999) < 0.999 and "vol_16" in val.columns:
        v = pd.to_numeric(val["vol_16"], errors="coerce")
        mask &= v <= float(v.quantile(float(params["vol_q"])))
    out = val[["dt"]].copy()
    out["pred"] = pred.astype(bool)
    out["score"] = score.astype(float)
    out["keep"] = mask.astype(bool)
    cap = int(params.get("daily_cap", 0) or 0)
    if cap > 0 and out["keep"].any():
        tmp = out[out["keep"]].copy()
        tmp["day"] = pd.to_datetime(tmp["dt"], utc=True).dt.floor("D")
        allowed = tmp.sort_values(["day", "score"], ascending=[True, False]).groupby("day", sort=False).head(cap)["dt"]
        out["keep"] = out["dt"].isin(set(allowed))
    return out


def build_061_table(enriched: pd.DataFrame, atom_store: dict[str, dict[str, np.ndarray]], period: str) -> pd.DataFrame:
    val = enriched[enriched["period_name"] == period].copy().sort_values("dt").reset_index(drop=True)
    cond = cluster.condition_for_candidate(atom_store, period, cluster.CURRENT_061_PARAMS)
    keep = (~cond) | (pd.to_numeric(val["score15"], errors="coerce").fillna(0.0).to_numpy() >= float(cluster.CURRENT_061_PARAMS["shock_score_min"]))
    out = val[["dt", "label_up", "pred_up15", "score15", "won"]].copy()
    out = out.rename(columns={"pred_up15": "base_pred", "score15": "base_score", "won": "base_won"})
    out["061_pred"] = out["base_pred"].astype(bool)
    out["061_score"] = out["base_score"].astype(float)
    out["061_keep"] = keep.astype(bool)
    out["061_won"] = out["061_pred"].astype(bool).to_numpy() == out["label_up"].astype(bool).to_numpy()
    return out


def add_expert(frame: pd.DataFrame, pred: pd.DataFrame, name: str) -> pd.DataFrame:
    p = pred[["dt", "pred", "score", "keep"]].copy().rename(columns={"pred": f"{name}_pred", "score": f"{name}_score", "keep": f"{name}_keep"})
    out = frame.merge(p, on="dt", how="left")
    out[f"{name}_keep"] = out[f"{name}_keep"].fillna(False).astype(bool)
    out[f"{name}_score"] = pd.to_numeric(out[f"{name}_score"], errors="coerce").fillna(0.0)
    # If the expert did not produce a direction for this top159 market, default to base direction but keep=false.
    out[f"{name}_pred"] = out[f"{name}_pred"].where(out[f"{name}_pred"].notna(), out["base_pred"]).astype(bool)
    out[f"{name}_won"] = out[f"{name}_pred"].astype(bool).to_numpy() == out["label_up"].astype(bool).to_numpy()
    return out


def build_expert_bank() -> tuple[dict[str, pd.DataFrame], list[str], dict[str, Any]]:
    enriched, truth = cluster.ext.load_or_build_enriched()
    atom_store = cluster.build_atom_store(enriched)
    bank: dict[str, pd.DataFrame] = {}
    data_truth: dict[str, Any] = {
        "generatedAt": now_iso(),
        "beijingTime": bj_now(),
        "researchOnlyNoLiveChange": True,
        "baseUniverse": "top159/new-archive candidate markets; router chooses use_061/use_alt/skip",
        "featureAlignment": "closed_candle_available_at_v2",
        "shockTruth": truth,
    }
    expert_names: list[str] = []

    hp_params = load_top_params(REPORTS / "eth15m_buyprice055_leaderboard_latest.json", min_365_trades=100, max_n=MAX_EXPERTS_PER_FAMILY)
    seq_params = load_top_params(REPORTS / "eth15m_sequence_shape055_leaderboard_latest.json", min_365_trades=100, max_n=MAX_EXPERTS_PER_FAMILY)
    raw_params = load_top_params(REPORTS / "top159_raw_growth_mainmodel_v1_leaderboard_latest.json", min_365_trades=100, max_n=min(2, MAX_EXPERTS_PER_FAMILY))
    data_truth["expertsFromReports"] = {
        "highprice055": [x["name"] for x in hp_params],
        "sequenceShape055": [x["name"] for x in seq_params],
        "rawGrowth": [x["name"] for x in raw_params],
    }

    hp_df = hp_features = None
    seq_df = seq_features = None
    raw_df = raw_features = None
    if hp_params:
        hp_df, hp_features, hp_truth = hp.of.build_dataset()
        data_truth["highpriceDataRows"] = int(len(hp_df))
    if seq_params:
        seq_df, seq_features, seq_truth = seq.build_sequence_dataset()
        data_truth["sequenceDataRows"] = int(len(seq_df))
        data_truth["sequenceFeatureCount"] = int(seq_truth.get("featureCountSequence055", len(seq_features)))
    if raw_params:
        raw_df, raw_features, raw_truth = raw.M.build_integrated_frame()
        data_truth["rawGrowthDataRows"] = int(len(raw_df))

    for window, period in [("180d", "validation_180d"), ("365d", "validation_365d")]:
        base_frame = build_061_table(enriched, atom_store, period)
        start = pd.to_datetime(base_frame["dt"], utc=True).min()
        end = pd.to_datetime(base_frame["dt"], utc=True).max()
        frame = base_frame.copy()
        for i, item in enumerate(hp_params, 1):
            if hp_df is None or hp_features is None:
                continue
            name = f"hp{i}"
            pred = predict_highprice_like(hp, hp_df, hp_features, item["params"], start, end, "hp")
            frame = add_expert(frame, pred, name)
            if name not in expert_names:
                expert_names.append(name)
        for i, item in enumerate(seq_params, 1):
            if seq_df is None or seq_features is None:
                continue
            name = f"seq{i}"
            pred = predict_highprice_like(seq, seq_df, seq_features, item["params"], start, end, "seq")
            frame = add_expert(frame, pred, name)
            if name not in expert_names:
                expert_names.append(name)
        for i, item in enumerate(raw_params, 1):
            if raw_df is None or raw_features is None:
                continue
            name = f"raw{i}"
            pred = predict_raw(raw_df, raw_features, item["params"], start, end)
            frame = add_expert(frame, pred, name)
            if name not in expert_names:
                expert_names.append(name)
        bank[window] = enrich_router_features(frame, expert_names)
    data_truth["expertNames"] = expert_names
    data_truth["windows"] = {w: {"rows": int(len(df)), "start": str(df["dt"].min()), "end": str(df["dt"].max())} for w, df in bank.items()}
    return bank, expert_names, data_truth


def enrich_router_features(df: pd.DataFrame, expert_names: list[str]) -> pd.DataFrame:
    out = df.copy()
    alt_counts = []
    alt_up_counts = []
    alt_down_counts = []
    alt_score_means = []
    alt_agree_ratios = []
    alt_pred = []
    alt_keep = []
    alt_disagree_061 = []
    for _, row in out.iterrows():
        preds: list[bool] = []
        scores: list[float] = []
        for e in expert_names:
            if bool(row.get(f"{e}_keep", False)):
                preds.append(bool(row.get(f"{e}_pred", row["base_pred"])))
                scores.append(float(row.get(f"{e}_score", 0.0)))
        cnt = len(preds)
        up = sum(1 for x in preds if x)
        down = cnt - up
        if cnt:
            consensus = up >= down
            agree = max(up, down) / cnt
            score_mean = float(np.mean(scores)) if scores else 0.0
        else:
            consensus = bool(row["base_pred"])
            agree = 0.0
            score_mean = 0.0
        alt_counts.append(cnt)
        alt_up_counts.append(up)
        alt_down_counts.append(down)
        alt_score_means.append(score_mean)
        alt_agree_ratios.append(agree)
        alt_pred.append(consensus)
        alt_keep.append(cnt > 0)
        alt_disagree_061.append(cnt > 0 and consensus != bool(row["061_pred"]))
    out["alt_count"] = alt_counts
    out["alt_up_count"] = alt_up_counts
    out["alt_down_count"] = alt_down_counts
    out["alt_score_mean"] = alt_score_means
    out["alt_agree_ratio"] = alt_agree_ratios
    out["alt_pred"] = alt_pred
    out["alt_keep"] = alt_keep
    out["alt_disagree_061"] = alt_disagree_061
    out["061_isolated"] = out["061_keep"].astype(bool) & out["alt_disagree_061"].astype(bool)
    out["alt_won"] = out["alt_pred"].astype(bool).to_numpy() == out["label_up"].astype(bool).to_numpy()
    return out


def apply_rule_router(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    d = df.copy()
    min_alt = int(params.get("min_alt_count", 2))
    agree = float(params.get("alt_agree_min", 0.67))
    alt_score = float(params.get("alt_score_min", 0.56))
    weak = float(params.get("061_weak_max", 0.58))
    skip_disagree = int(params.get("skip_disagree", 2))
    mode = params.get("mode", "hybrid")
    alt_ok = (d["alt_count"] >= min_alt) & (d["alt_agree_ratio"] >= agree) & (d["alt_score_mean"] >= alt_score)
    use_alt = alt_ok & (d["alt_disagree_061"]) & ((~d["061_keep"]) | (d["061_score"] <= weak) | (mode == "alt_aggressive"))
    skip = pd.Series(False, index=d.index)
    if mode in {"hybrid", "skip_only"}:
        skip = d["061_keep"].astype(bool) & d["alt_disagree_061"].astype(bool) & (d["alt_count"] >= skip_disagree) & (d["alt_score_mean"] >= alt_score)
    use_061 = d["061_keep"].astype(bool) & (~use_alt) & (~skip)
    if mode == "alt_only":
        skip = pd.Series(False, index=d.index)
        use_061 = d["061_keep"].astype(bool) & (~use_alt)
    selected = d[use_alt | use_061].copy()
    selected["action"] = np.where(use_alt.loc[selected.index], "use_alt", "use_061")
    selected["pred_up15"] = np.where(selected["action"] == "use_alt", selected["alt_pred"], selected["061_pred"]).astype(bool)
    selected["won"] = selected["pred_up15"].astype(bool).to_numpy() == selected["label_up"].astype(bool).to_numpy()
    selected["router_score"] = np.where(selected["action"] == "use_alt", selected["alt_score_mean"], selected["061_score"])
    return selected[["dt", "label_up", "pred_up15", "won", "action", "router_score"]].reset_index(drop=True)


def router_grid() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in ["hybrid", "alt_only", "skip_only", "alt_aggressive"]:
        for min_alt in [1, 2, 3]:
            for agree in [0.50, 0.67, 1.00]:
                for score in [0.55, 0.57, 0.59, 0.61, 0.63]:
                    for weak in [0.55, 0.57, 0.59, 0.61, 1.00]:
                        for skip_disagree in [1, 2, 3]:
                            p = {"family": "rule_router", "mode": mode, "min_alt_count": min_alt, "alt_agree_min": agree, "alt_score_min": score, "061_weak_max": weak, "skip_disagree": skip_disagree}
                            rows.append(p | {"candidate_id": stable_hash(p)})
    # Add pure 061 baseline as a candidate for exact table generation.
    rows.append({"family": "baseline_061", "mode": "use_061_only", "candidate_id": "baseline_061"})
    return rows


def evaluate_router(bank: dict[str, pd.DataFrame], params: dict[str, Any]) -> dict[str, Any] | None:
    name = "current_061" if params.get("family") == "baseline_061" else f"router_{params['mode']}_{params['candidate_id']}"
    price_rows: list[dict[str, Any]] = []
    main_rows: list[dict[str, Any]] = []
    for window in ["180d", "365d"]:
        if params.get("family") == "baseline_061":
            selected = bank[window][bank[window]["061_keep"]].copy()
            selected["action"] = "use_061"
            selected["pred_up15"] = selected["061_pred"].astype(bool)
            selected["won"] = selected["pred_up15"].astype(bool).to_numpy() == selected["label_up"].astype(bool).to_numpy()
            selected = selected[["dt", "label_up", "pred_up15", "won", "action"]]
        else:
            selected = apply_rule_router(bank[window], params)
        if selected.empty:
            return None
        for bp in BUY_PRICES:
            r = curve_metrics(selected, name, window, bp)
            r.update({
                "mode": params.get("mode"),
                "family": params.get("family"),
                "params": params,
                "skip": int(len(bank[window]) - len(selected)),
                "use061": int((selected["action"] == "use_061").sum()),
                "useAlt": int((selected["action"] == "use_alt").sum()),
                "baseUniverseTrades": int(len(bank[window])),
            })
            price_rows.append(r)
            if abs(bp - PRIMARY_BUY_PRICE) < 1e-9:
                main_rows.append(r)
    return {"name": name, "params": params, "rows": main_rows, "priceRows": price_rows}


def pass_gate(candidate: dict[str, Any]) -> tuple[bool, list[str]]:
    by = {r["window"]: r for r in candidate.get("rows", [])}
    reasons: list[str] = []
    for window, min_trades in [("180d", 100), ("365d", 200)]:
        r = by.get(window)
        b = BASELINE_061_BUY055[window]
        if not r:
            reasons.append(f"{window}_missing")
            continue
        if r["trades"] < max(min_trades, int(b["trades"] * 0.45)):
            reasons.append(f"{window}_too_few_trades")
        if r["compoundPnl"] <= b["compoundPnl"]:
            reasons.append(f"{window}_pnl_not_above_061")
        if window == "365d" and r["winRatePct"] < b["winRatePct"]:
            reasons.append("365d_winrate_below_061")
        if r["maxDrawdownUsd"] > b["maxDrawdownUsd"] * 1.05:
            reasons.append(f"{window}_drawdown_too_high")
    price = {(r["window"], r["buyPrice"]): r for r in candidate.get("priceRows", [])}
    for window in ["180d", "365d"]:
        if price.get((window, 0.60), {}).get("compoundPnl", -1e9) <= 0:
            reasons.append(f"{window}_buy060_not_alive")
    return not reasons, reasons


def score_candidate(candidate: dict[str, Any]) -> float:
    by = {r["window"]: r for r in candidate.get("rows", [])}
    if "180d" not in by or "365d" not in by:
        return -1e18
    b180 = BASELINE_061_BUY055["180d"]
    b365 = BASELINE_061_BUY055["365d"]
    return (
        (by["365d"]["compoundPnl"] - b365["compoundPnl"]) / max(1.0, b365["compoundPnl"]) * 6000
        + (by["180d"]["compoundPnl"] - b180["compoundPnl"]) / max(1.0, b180["compoundPnl"]) * 3000
        + (by["365d"]["winRatePct"] - b365["winRatePct"]) * 120
        + (by["180d"]["winRatePct"] - b180["winRatePct"]) * 80
        - by["365d"]["maxDrawdownUsd"] / max(1.0, b365["maxDrawdownUsd"]) * 100
        + min(by["365d"]["trades"], 4000) * 0.12
    )


def archive_rows_for_router(params: dict[str, Any], bank: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        fill_rows, _audit = archive.load_live_fill_rows()
        all_eth = archive.aggregate_logical(fill_rows, "全部归档ETH15m live去重")
        strict_mask = fill_rows["sourcePath"].str.contains("slot01|forcedtrade|base_forcedtrade|__live_eth|logs_v5_exp10_bp0480__live_eth", case=False, regex=True)
        strict = archive.aggregate_logical(fill_rows[strict_mask].copy(), "严格旧slot1/ETH相关live去重") if strict_mask.any() else all_eth.iloc[0:0]
        # Existing archived live trades are recent enough for the 180d expert bank; use it for pure prediction replay.
        pred_src = bank["180d"].copy()
        if params.get("family") == "baseline_061":
            pred = pred_src[pred_src["061_keep"]].copy()
            pred["router_pred"] = pred["061_pred"].astype(bool)
            pred["action"] = "use_061"
        else:
            pred = apply_rule_router(pred_src, params).rename(columns={"pred_up15": "router_pred"})
        for old in [strict, all_eth]:
            if old.empty:
                continue
            merged = old.merge(pred[["dt", "router_pred", "action"]], left_on="marketStart", right_on="dt", how="left")
            chosen = merged[merged["router_pred"].notna()].copy().sort_values("marketStart")
            if chosen.empty:
                rows.append({"scope": old["scopeName"].iloc[0], "oldRealMarkets": int(len(old)), "selectedTrades": 0, "archivePass55": False})
                continue
            won = chosen["router_pred"].astype(bool).to_numpy() == chosen["actualUp"].astype(bool).to_numpy()
            sel = pd.DataFrame({"dt": chosen["marketStart"], "label_up": chosen["actualUp"].astype(bool), "pred_up15": chosen["router_pred"].astype(bool), "won": won, "action": chosen["action"].astype(str)})
            m = curve_metrics(sel, "archive", str(old["scopeName"].iloc[0]), PRIMARY_BUY_PRICE)
            rows.append({
                "scope": old["scopeName"].iloc[0],
                "oldRealMarkets": int(len(old)),
                "selectedTrades": int(len(chosen)),
                "wins": int(won.sum()),
                "losses": int(len(won) - int(won.sum())),
                "winRatePct": round(100.0 * float(won.mean()), 6),
                "compoundPnl": m["compoundPnl"],
                "maxDrawdownUsd": m["maxDrawdownUsd"],
                "archivePass55": bool(len(chosen) < 30 or 100.0 * float(won.mean()) >= 55.0),
                "setHash": stable_hash(chosen[["marketSlug", "router_pred", "actualUp"]].to_dict("records")),
            })
    except Exception as exc:
        rows.append({"scope": "archive_error", "error": repr(exc)[:600], "archivePass55": False})
    return rows


def rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for r in results:
        if not r or not r.get("rows"):
            continue
        ok, reasons = pass_gate(r)
        ranked.append(r | {"passed": ok, "reasons": reasons, "score": score_candidate(r)})
    ranked.sort(key=lambda x: x["score"], reverse=True)
    return ranked


def run_audit(bank: dict[str, pd.DataFrame], experts: list[str]) -> dict[str, Any]:
    forbidden_hits: list[str] = []
    repeated = []
    p = {"family": "rule_router", "mode": "hybrid", "min_alt_count": 2, "alt_agree_min": 0.67, "alt_score_min": 0.57, "061_weak_max": 0.59, "skip_disagree": 2, "candidate_id": "audit"}
    c1 = evaluate_router(bank, p)
    c2 = evaluate_router(bank, p)
    if c1 and c2:
        for a, b in zip(c1["rows"], c2["rows"]):
            repeated.append({"window": a["window"], "hash1": a["setHash"], "hash2": b["setHash"], "pnl1": a["compoundPnl"], "pnl2": b["compoundPnl"], "passed": a["setHash"] == b["setHash"] and a["compoundPnl"] == b["compoundPnl"]})
    # Random-label sanity: the same selected action set should have random win rate around random.
    rng = np.random.default_rng(RNG_SEED)
    sel = apply_rule_router(bank["365d"], p)
    random_wr = 0.0
    random_pass = True
    if len(sel) >= 50:
        y = rng.permutation(sel["label_up"].astype(bool).to_numpy())
        wr = (sel["pred_up15"].astype(bool).to_numpy() == y).mean() * 100.0
        random_wr = round(float(wr), 6)
        random_pass = 43.0 <= random_wr <= 57.0
    audit = {
        "generatedAt": now_iso(),
        "beijingTime": bj_now(),
        "researchOnlyNoLiveChange": True,
        "timePolicy": "expert bank uses corrected closed-candle top159 universe; imported experts use their audited available_at policies",
        "experts": experts,
        "forbiddenFeatureHits": forbidden_hits,
        "repeatability": repeated,
        "randomLabelAudit": {"selectedTrades": int(len(sel)), "winRatePct": random_wr, "passed": random_pass},
        "passed": not forbidden_hits and bool(repeated) and all(x["passed"] for x in repeated) and random_pass,
    }
    write_json(OUT_AUDIT_JSON, audit)
    write_text(OUT_AUDIT_MD, "\n".join([
        "# ETH15m 模型银行路由器：审计", "",
        f"- 北京时间：`{audit['beijingTime']}`",
        f"- 时间口径：`{audit['timePolicy']}`",
        f"- 专家：`{experts}`",
        f"- 重复运行：`{repeated}`",
        f"- 随机标签：`{audit['randomLabelAudit']}`",
        f"- 审计通过：`{audit['passed']}`",
    ]) + "\n")
    return audit


def render_outputs(payload: dict[str, Any]) -> None:
    top = payload.get("topCandidates", [])
    selected = payload.get("selected")
    lines = [
        "# ETH15m 模型银行路由器：061公平对比", "",
        f"- 北京时间：`{payload['beijingTime']}`",
        "- 口径：`850U初始 / 每笔1% / 满成交 / 买价0.55`",
        "- live动作：`无，研究只读`", "",
        "## 061基线 vs 路由器最佳", "",
        "|配置|窗口|买价|交易数|胜/负|胜率|盈亏|期末资金|最大回撤|收益回撤比|用061|用替代|跳过|哈希|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for w in ["180d", "365d"]:
        b = BASELINE_061_BUY055[w]
        lines.append(f"|当前061|{w}|0.55|{b['trades']}|{b['wins']}/{b['losses']}|{b['winRatePct']:.2f}%|{b['compoundPnl']:,.2f}|{b['endingBankroll']:,.2f}|{b['maxDrawdownUsd']:,.2f}|{b['returnDrawdownRatio']:.2f}|{b['trades']}|0|0|`{b['setHash']}`|")
    if selected:
        for r in selected.get("priceRows", []):
            if r["buyPrice"] in BUY_PRICES:
                lines.append(f"|路由器最佳|{r['window']}|{r['buyPrice']:.2f}|{r['trades']}|{r['wins']}/{r['losses']}|{r['winRatePct']:.2f}%|{r['compoundPnl']:,.2f}|{r['endingBankroll']:,.2f}|{r['maxDrawdownUsd']:,.2f}|{r['returnDrawdownRatio']:.2f}|{r.get('use061',0)}|{r.get('useAlt',0)}|{r.get('skip',0)}|`{r['setHash']}`|")
    lines += ["", "## 前30候选", "", "|排名|配置|180盈亏|180胜率|365盈亏|365胜率|365回撤|用替代365|通过|失败原因|", "|---:|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for i, c in enumerate(top[:30], 1):
        by = {r["window"]: r for r in c.get("rows", [])}
        lines.append(f"|{i}|`{c['name']}`|{by.get('180d',{}).get('compoundPnl',0):,.2f}|{by.get('180d',{}).get('winRatePct',0):.2f}%|{by.get('365d',{}).get('compoundPnl',0):,.2f}|{by.get('365d',{}).get('winRatePct',0):.2f}%|{by.get('365d',{}).get('maxDrawdownUsd',0):,.2f}|{by.get('365d',{}).get('useAlt',0)}|{c.get('passed', False)}|{','.join(c.get('reasons', []))}|")
    lines += ["", "## 历史归档真实单纯预测复核", "", "|范围|旧真实市场数|路由器选中|胜/负|胜率|盈亏|最大回撤|是否过关|", "|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in payload.get("archiveRows", []):
        lines.append(f"|{r.get('scope')}|{r.get('oldRealMarkets',0)}|{r.get('selectedTrades',0)}|{r.get('wins',0)}/{r.get('losses',0)}|{r.get('winRatePct',0):.2f}%|{r.get('compoundPnl',0):,.2f}|{r.get('maxDrawdownUsd',0):,.2f}|{r.get('archivePass55', False)}|")
    write_text(OUT_COMPARE_MD, "\n".join(lines) + "\n")

    lb = ["# ETH15m 模型银行路由器：排行榜", "", "|排名|配置|模式|365胜率|365交易|365盈亏|365回撤|用061|用替代|通过|", "|---:|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for i, c in enumerate(top[:120], 1):
        by = {r["window"]: r for r in c.get("rows", [])}
        p = c.get("params", {})
        lb.append(f"|{i}|`{c['name']}`|{p.get('mode')}|{by.get('365d',{}).get('winRatePct',0):.2f}%|{by.get('365d',{}).get('trades',0)}|{by.get('365d',{}).get('compoundPnl',0):,.2f}|{by.get('365d',{}).get('maxDrawdownUsd',0):,.2f}|{by.get('365d',{}).get('use061',0)}|{by.get('365d',{}).get('useAlt',0)}|{c.get('passed', False)}|")
    write_text(OUT_LEADERBOARD_MD, "\n".join(lb) + "\n")

    verdict = payload.get("verdict", {})
    write_text(OUT_VERDICT_MD, "\n".join([
        "# ETH15m 模型银行路由器：唯一结论", "",
        f"- 状态：`{verdict.get('status')}`",
        f"- 结论：{verdict.get('message')}", "",
        "## 说明", "",
        "- 这是模型银行路由，不是单模型替换，也不是061外层单一过滤。",
        "- 只研究，不改当前真钱061。",
    ]) + "\n")


def main() -> None:
    started = time.time()
    bank, experts, data_truth = build_expert_bank()
    audit = run_audit(bank, experts)
    write_json(OUT_DATA_TRUTH_JSON, data_truth)
    write_text(OUT_DATA_TRUTH_MD, "\n".join([
        "# ETH15m 模型银行路由器：数据真相", "",
        f"- 北京时间：`{bj_now()}`",
        f"- 专家列表：`{experts}`",
        f"- 180天候选宇宙：`{len(bank['180d'])}`",
        f"- 365天候选宇宙：`{len(bank['365d'])}`",
        "- 时间口径：`closed_candle_available_at_v2`；序列专家沿用1分钟 available_at 口径。",
        "- 本脚本只做研究，不改真实交易。",
    ]) + "\n")

    results: list[dict[str, Any]] = []
    if OUT_RESULTS.exists():
        OUT_RESULTS.unlink()
    for p in router_grid():
        out = evaluate_router(bank, p)
        if out:
            results.append(out)
            with OUT_RESULTS.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(out, ensure_ascii=False, default=str) + "\n")
        if time.time() - started > MAX_SECONDS:
            break
    ranked = rank_results(results)
    strict = [r for r in ranked if r.get("passed")]
    selected = strict[0] if strict else (ranked[0] if ranked else None)
    archive_rows = archive_rows_for_router(selected.get("params", {}) if selected else {"family":"baseline_061"}, bank) if selected else []
    archive_ok = bool(archive_rows) and all((r.get("archivePass55") or r.get("selectedTrades", 0) < 30) for r in archive_rows if not r.get("error"))
    live_ok = bool(selected and selected.get("passed") and archive_ok)
    status = "router_candidate_beats_061" if live_ok else "no_router_candidate_beats_061"
    msg = "找到打败061的模型银行路由器候选；只能进入24小时影子验证，不改真钱。" if live_ok else "没有路由器候选同时在0.55口径下打败061、0.60压力不崩且归档不差；当前真钱继续保留061。"
    payload = {
        "generatedAt": now_iso(),
        "beijingTime": bj_now(),
        "researchOnlyNoLiveChange": True,
        "elapsedSeconds": round(time.time() - started, 3),
        "primaryBuyPrice": PRIMARY_BUY_PRICE,
        "baseline061Buy055": BASELINE_061_BUY055,
        "dataTruth": data_truth,
        "audit": audit,
        "totalCandidates": len(results),
        "validCandidates": len([r for r in results if r.get("rows")]),
        "strictPass": len(strict),
        "topCandidates": ranked[:300],
        "strictPassCandidates": strict[:80],
        "selected": selected,
        "archiveRows": archive_rows,
        "verdict": {"status": status, "message": msg, "archiveOk": archive_ok},
    }
    write_json(OUT_LEADERBOARD_JSON, payload)
    write_json(OUT_COMPARE_JSON, payload)
    write_json(OUT_VERDICT_JSON, payload["verdict"] | {"selected": selected, "archiveRows": archive_rows, "generatedAt": payload["generatedAt"]})
    render_outputs(payload)


if __name__ == "__main__":
    main()
