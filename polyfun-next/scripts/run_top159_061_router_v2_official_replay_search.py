#!/usr/bin/env python3
from __future__ import annotations

"""Router V2 search for ETH061 with official actual-order replay.

Research only. Does not mutate live configs, orders, ledgers, claim state, or
monitor state.

Locked assumptions:
  - Base live strategy remains 061.
  - Router may only keep/block candidates already allowed by 061.
  - Daily caps, if enabled, are chronological by Beijing day. No future daily
    sorting is allowed.
  - 180d/365d use the audited closed-candle research universe.
  - Actual live-order replay uses official-filled logical orders and official
    settlement results from the top159 monitor settlement cache.
"""

import importlib.util
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(key, os.environ.get("TOP159_061_ROUTER_V2_THREADS", "1"))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path("/Users/mac/polyfun")
NEXT = ROOT / "polyfun-next"
REPORTS = ROOT / "reports"
SCRIPTS = NEXT / "scripts"

LIVE_ORDER_SCRIPT = SCRIPTS / "run_top159_061_score_calibration_router_live_order_hyperopt.py"
ACTUAL_REPLAY_SCRIPT = SCRIPTS / "run_top159_061_full_calibration_actual_live_replay.py"

OUT_AUDIT_JSON = REPORTS / "top159_061_router_v2_official_replay_bug_audit_latest.json"
OUT_AUDIT_MD = REPORTS / "top159_061_router_v2_official_replay_bug_audit_latest.md"
OUT_LEADERBOARD_JSON = REPORTS / "top159_061_router_v2_official_replay_leaderboard_latest.json"
OUT_LEADERBOARD_MD = REPORTS / "top159_061_router_v2_official_replay_leaderboard_latest.md"
OUT_COMPARE_JSON = REPORTS / "top159_061_router_v2_official_replay_compare_latest.json"
OUT_COMPARE_MD = REPORTS / "top159_061_router_v2_official_replay_compare_latest.md"
OUT_VERDICT_JSON = REPORTS / "top159_061_router_v2_official_replay_unique_verdict_latest.json"
OUT_VERDICT_MD = REPORTS / "top159_061_router_v2_official_replay_unique_verdict_latest.md"
OUT_CHECKPOINT_JSON = REPORTS / "top159_061_router_v2_official_replay_checkpoint_latest.json"

START_BANKROLL = 850.0
BUY_PRICE = 0.55
STAKE_PCT = 0.01


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


live = load_module("router_v2_live_order", LIVE_ORDER_SCRIPT)
actual = load_module("router_v2_actual_replay", ACTUAL_REPLAY_SCRIPT)
full = live.full
BASELINE_061 = live.BASELINE_061


@dataclass(frozen=True)
class V2Policy:
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
    daily_cap_mode: str = "chronological"

    def to_live_policy(self):
        return live.Policy(**asdict(self))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def bj_now() -> str:
    return pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S CST")


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


def policy_grid_v2() -> list[V2Policy]:
    rows: list[V2Policy] = []
    # Keep this search live-orderable: only chronological caps and require_061=True.
    fast = os.environ.get("TOP159_061_ROUTER_V2_FAST", "0") == "1"
    if fast:
        caps = [None, 24, 48]
        # Deliberately compact but covers the useful live-order shapes found in prior runs.
        for p in [0.515, 0.525, 0.535, 0.545, 0.555, 0.565]:
            for s in [0.55, 0.555, 0.56, 0.57]:
                rows.append(V2Policy(mode="plain", prob_min=p, score_min=s, daily_cap=None))
        for up in [0.52, 0.53, 0.545, 0.565]:
            for down in [0.52, 0.53, 0.545, 0.565]:
                rows.append(V2Policy(mode="directional", score_min=0.55, daily_cap=None, up_prob_min=up, down_prob_min=down))
        for cap in caps:
            for alpha in [0.75, 1.0, 1.5]:
                for cm in [0.535, 0.545, 0.555, 0.565]:
                    rows.append(V2Policy(mode="combo", score_min=0.55, daily_cap=cap, combo_min=cm, combo_alpha=alpha))
        for p in [0.52, 0.535, 0.55, 0.565]:
            for cut in [0.57, 0.60]:
                for extra in [0.01, 0.02, 0.03]:
                    rows.append(V2Policy(mode="low_score_extra", prob_min=p, score_min=0.55, daily_cap=None, low_score_cut=cut, low_prob_extra=extra))
        seen = set()
        uniq = []
        for r in rows:
            key = json.dumps(asdict(r), sort_keys=True)
            if key not in seen:
                seen.add(key); uniq.append(r)
        return uniq
    caps = [None, 12, 16, 20, 24, 32, 48]
    for cap in caps:
        for p in [0.505, 0.515, 0.525, 0.5325, 0.535, 0.5375, 0.5425, 0.545, 0.55, 0.5575, 0.565, 0.58, 0.60]:
            for s in [0.55, 0.5525, 0.555, 0.5575, 0.56, 0.565, 0.57, 0.58]:
                rows.append(V2Policy(mode="plain", prob_min=p, score_min=s, daily_cap=cap))
    for cap in caps:
        for up in [0.515, 0.525, 0.535, 0.545, 0.555, 0.565, 0.58]:
            for down in [0.515, 0.525, 0.535, 0.545, 0.555, 0.565, 0.58]:
                rows.append(V2Policy(mode="directional", score_min=0.55, daily_cap=cap, up_prob_min=up, down_prob_min=down))
    for cap in caps:
        for alpha in [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
            for cm in [0.525, 0.535, 0.54, 0.545, 0.55, 0.555, 0.565, 0.575, 0.59]:
                for s in [0.55, 0.555, 0.56, 0.57]:
                    rows.append(V2Policy(mode="combo", score_min=s, daily_cap=cap, combo_min=cm, combo_alpha=alpha))
    for cap in caps:
        for p in [0.515, 0.525, 0.535, 0.545, 0.555, 0.565]:
            for cut in [0.56, 0.57, 0.58, 0.60, 0.62]:
                for extra in [0.005, 0.01, 0.015, 0.02, 0.03, 0.04]:
                    rows.append(V2Policy(mode="low_score_extra", prob_min=p, score_min=0.55, daily_cap=cap, low_score_cut=cut, low_prob_extra=extra))
    # Deduplicate by JSON representation.
    seen = set()
    uniq = []
    for r in rows:
        key = json.dumps(asdict(r), sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    return uniq


def load_seed_tasks() -> list[tuple[str, dict[str, Any], float]]:
    tasks: list[tuple[str, dict[str, Any], float]] = []
    paths = [
        REPORTS / "top159_061_score_calibration_router_live_order_hyperopt_leaderboard_latest.json",
        REPORTS / "top159_061_score_calibration_router_live_order_hyperopt_061_compare_latest.json",
        REPORTS / "top159_061_score_calibration_router_full_hyperopt_061_compare_latest.json",
    ]
    for path in paths:
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        candidates = []
        if isinstance(payload.get("topCandidates"), list):
            candidates.extend(payload.get("topCandidates") or [])
        if isinstance(payload.get("bestCandidate"), dict):
            candidates.append(payload["bestCandidate"])
        for c in candidates:
            if not isinstance(c, dict) or not c.get("featureMode") or not c.get("modelCfg"):
                continue
            tasks.append((str(c["featureMode"]), dict(c["modelCfg"]), float(c.get("calibC") or 0.75)))
            if len(tasks) >= int(os.environ.get("TOP159_061_ROUTER_V2_MAX_SEED_TASKS", "36")):
                break
    # Add a few robust fallback configs if reports are absent or too narrow.
    fallback = [
        ("atoms_core", {"kind": "lgbm", "n_estimators": 180, "learning_rate": 0.028, "num_leaves": 24, "min_child_samples": 80, "reg_lambda": 2.0}, 0.75),
        ("base_num", {"kind": "lgbm", "n_estimators": 180, "learning_rate": 0.028, "num_leaves": 16, "min_child_samples": 80, "reg_lambda": 2.0}, 0.75),
        ("atoms_core", {"kind": "logit", "penalty": "l2", "C": 0.35, "class_weight": "balanced"}, 0.45),
        ("atoms_all", {"kind": "lgbm", "n_estimators": 180, "learning_rate": 0.028, "num_leaves": 16, "min_child_samples": 80, "reg_lambda": 2.0}, 0.45),
    ]
    tasks.extend(fallback)
    seen = set()
    uniq = []
    for t in tasks:
        key = json.dumps({"f": t[0], "m": t[1], "c": t[2]}, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(t)
    return uniq[: int(os.environ.get("TOP159_061_ROUTER_V2_MAX_TASKS", "40"))]


def curve_metrics_actual(rows: pd.DataFrame, name: str, initial: float, stake_fraction: float) -> dict[str, Any]:
    selected = rows.sort_values("dt").reset_index(drop=True)
    equity = float(initial)
    curve: list[float] = []
    for _, r in selected.iterrows():
        stake = equity * float(stake_fraction)
        avg_price = float(r.get("actual_avg_price") or 0.0)
        if avg_price <= 0:
            continue
        if bool(r.get("won")):
            equity += stake * (1.0 / avg_price - 1.0)
        else:
            equity -= stake
        equity = max(equity, 0.0)
        curve.append(equity)
    arr = np.asarray(curve, dtype=float)
    max_dd = 0.0
    if len(arr):
        peak = np.maximum.accumulate(arr)
        max_dd = float((peak - arr).max())
    wins = int(selected["won"].astype(bool).sum()) if not selected.empty else 0
    losses = int(len(selected) - wins)
    actual_cost = float(pd.to_numeric(selected.get("actual_cost", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()) if not selected.empty else 0.0
    actual_pnl = float(pd.to_numeric(selected.get("actual_pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()) if not selected.empty else 0.0
    return {
        "name": name,
        "logicalTrades": int(len(selected)),
        "wins": wins,
        "losses": losses,
        "winRatePct": round(100.0 * wins / len(selected), 6) if len(selected) else 0.0,
        "initialFunds": round(float(initial), 6),
        "endingFunds": round(float(equity), 6),
        "compoundPnl": round(float(equity - initial), 6),
        "maxDrawdownUsd": round(float(max_dd), 6),
        "returnDrawdownRatio": round(float(equity - initial) / max_dd, 6) if max_dd > 1e-12 else (999.0 if equity > initial else 0.0),
        "actualOfficialCost": round(actual_cost, 6),
        "actualOfficialPnl": round(actual_pnl, 6),
        "logicalAvgBuyPrice": round(float(selected["actual_avg_price"].mean()), 6) if len(selected) else 0.0,
        "weightedAvgBuyPrice": round(float(actual_cost / selected["actual_shares"].sum()), 6) if len(selected) and float(selected["actual_shares"].sum()) else 0.0,
        "setHash": full.stable_hash(selected[["market_slug", "side", "won", "actual_avg_price"]].to_dict("records")) if len(selected) else "empty",
    }


def build_actual_router_cache(enriched: pd.DataFrame, live_df: pd.DataFrame, feature_modes: set[str]) -> dict[str, Any]:
    train = enriched.sort_values("dt").drop_duplicates("dt", keep="last").reset_index(drop=True)
    train = train[pd.to_datetime(train["dt"], utc=True) < pd.to_datetime(live_df["dt"], utc=True).min()].copy().reset_index(drop=True)
    train_atom = full.base.base.atom_masks(train)
    shock = actual.shock
    base_live = live_df[["dt", "label_up", "pred_up15", "direction", "score15", "won", "period_name"]].copy()
    live_enriched = shock.enrich_shock_features(base_live)
    live_atom = full.base.base.atom_masks(live_enriched)
    train_idx, calib_idx = full.base.split_train_calib(train)
    feature_cache: dict[str, dict[str, pd.DataFrame]] = {}
    for fm in sorted(feature_modes):
        feature_cache[fm] = {
            "train_x_all": full.base.make_feature_frame(train, fm, train_atom),
            "live_x": full.base.make_feature_frame(live_enriched, fm, live_atom),
        }
    return {
        "train": train,
        "train_idx": train_idx,
        "calib_idx": calib_idx,
        "y_core": train.loc[train_idx, "won"].astype(bool).astype(int).to_numpy(),
        "y_calib": train.loc[calib_idx, "won"].astype(bool).astype(int).to_numpy(),
        "features": feature_cache,
    }


def train_live_router_generic_cached(candidate: dict[str, Any], live_df: pd.DataFrame, cache: dict[str, Any]) -> pd.DataFrame:
    fm = candidate["featureMode"]
    train_x_all = cache["features"][fm]["train_x_all"]
    live_x_raw = cache["features"][fm]["live_x"]
    train_idx = cache["train_idx"]
    calib_idx = cache["calib_idx"]
    core_x = train_x_all.loc[train_idx]
    calib_x = train_x_all.loc[calib_idx]
    core_x, calib_x, live_x = full.base.align_features(core_x, calib_x, live_x_raw)
    model = full.make_model(candidate["modelCfg"])
    model.fit(core_x, cache["y_core"])
    raw_calib = model.predict_proba(calib_x)[:, 1]
    raw_live = model.predict_proba(live_x)[:, 1]
    calibrator = LogisticRegression(C=float(candidate.get("calibC") or 0.75), penalty="l2", max_iter=1000, solver="lbfgs")
    calibrator.fit(raw_calib.reshape(-1, 1), cache["y_calib"])
    out = live_df.copy()
    out["router_prob"] = calibrator.predict_proba(raw_live.reshape(-1, 1))[:, 1]
    out["current061_keep"] = True
    return out


def official_replay_for_policy(scored: pd.DataFrame, policy: V2Policy, name: str, initial: float, stake_fraction: float) -> dict[str, Any]:
    selected = live.apply_policy_live_order(scored, policy.to_live_policy())
    skipped = scored[~scored["market_slug"].isin(set(selected["market_slug"]))].copy().sort_values("dt")
    row = curve_metrics_actual(selected, name, initial, stake_fraction)
    row.update({
        "allOfficialTrades": int(len(scored)),
        "keptTrades": int(len(selected)),
        "skippedTrades": int(len(skipped)),
        "interceptedWinners": int(skipped["won"].astype(bool).sum()) if not skipped.empty else 0,
        "interceptedLosers": int((~skipped["won"].astype(bool)).sum()) if not skipped.empty else 0,
        "retentionRateVsOfficial061": round(100.0 * len(selected) / max(1, len(scored)), 6),
        "officialTruthSource": "top159_monitor_official_settlement_fix_latest + official-filled ledger candidates",
        "pendingExcluded": True,
    })
    return row


def score_v2(candidate: dict[str, Any], baseline_actual: dict[str, Any], d3e3_actual: dict[str, Any]) -> float:
    rows = {r["window"]: r for r in candidate["rows"]}
    actual_row = candidate["officialReplay"]
    if set(rows) != {"180d", "365d"}:
        return -1e18
    # Strongly prioritize official actual replay, then 365d, then 180d.
    actual_gain = actual_row["actualOfficialPnl"] - baseline_actual["actualOfficialPnl"]
    actual_compound_gain = actual_row["compoundPnl"] - baseline_actual["compoundPnl"]
    actual_dd_gain = baseline_actual["maxDrawdownUsd"] - actual_row["maxDrawdownUsd"]
    d3e3_gap = actual_row["actualOfficialPnl"] - d3e3_actual["actualOfficialPnl"]
    b180, b365 = BASELINE_061["180d"], BASELINE_061["365d"]
    return (
        actual_gain * 90
        + actual_compound_gain * 20
        + actual_dd_gain * 6
        + d3e3_gap * 25
        + (rows["365d"]["compoundPnl"] - b365["compoundPnl"]) / max(1.0, b365["compoundPnl"]) * 2500
        + (rows["180d"]["compoundPnl"] - b180["compoundPnl"]) / max(1.0, b180["compoundPnl"]) * 1000
        + (rows["365d"]["winRatePct"] - b365["winRatePct"]) * 30
        - max(0.0, rows["365d"]["maxDrawdownUsd"] / max(1.0, b365["maxDrawdownUsd"]) - 1.05) * 600
    )


def candidate_passes(candidate: dict[str, Any], baseline_actual: dict[str, Any], d3e3_candidate: dict[str, Any] | None) -> tuple[bool, list[str]]:
    reasons = []
    rows = {r["window"]: r for r in candidate.get("rows", [])}
    for w in ["180d", "365d"]:
        if w not in rows:
            reasons.append(f"{w}_missing")
            continue
        b = BASELINE_061[w]
        if rows[w]["trades"] < max(100 if w == "180d" else 200, int(b["trades"] * 0.45)):
            reasons.append(f"{w}_too_few")
        if w == "365d" and rows[w]["compoundPnl"] < b["compoundPnl"] * 0.95:
            reasons.append("365d_pnl_too_weak_vs_061")
        if w == "180d" and rows[w]["compoundPnl"] < b["compoundPnl"] * 0.90:
            reasons.append("180d_pnl_too_weak_vs_061")
    actual_row = candidate.get("officialReplay") or {}
    if actual_row.get("logicalTrades", 0) < 50:
        reasons.append("official_too_few")
    if actual_row.get("actualOfficialPnl", -1e9) < baseline_actual.get("actualOfficialPnl", 0) and actual_row.get("maxDrawdownUsd", 1e9) >= baseline_actual.get("maxDrawdownUsd", 0):
        reasons.append("official_replay_worse_pnl_and_not_lower_dd")
    if d3e3_candidate is not None:
        d3_rows = {r["window"]: r for r in d3e3_candidate.get("rows", [])}
        if "365d" in d3_rows and "365d" in rows and rows["365d"]["compoundPnl"] < d3_rows["365d"]["compoundPnl"] * 0.90:
            reasons.append("365d_much_weaker_than_d3e3")
    return not reasons, reasons


def evaluate_task(task: tuple[str, dict[str, Any], float], policies: list[V2Policy], period_vals, atom_store, actual_rows: pd.DataFrame, actual_cache: dict[str, Any], initial: float, stake_fraction: float) -> list[dict[str, Any]]:
    feature_mode, cfg, calib_c = task
    model_key = full.stable_hash(cfg)
    preds = {}
    for window in ["180d", "365d"]:
        p, _m = full.fit_predict_one_window(period_vals, atom_store, window, feature_mode, cfg, calib_c)
        preds[window] = p
    scored_actual = train_live_router_generic_cached({"featureMode": feature_mode, "modelCfg": cfg, "calibC": calib_c}, actual_rows, actual_cache)
    out = []
    for pol in policies:
        c = live.evaluate_candidate(preds, pol.to_live_policy(), feature_mode, model_key, cfg, calib_c)
        c["policy"] = asdict(pol)
        c["featureMode"] = feature_mode
        c["modelKey"] = model_key
        c["modelCfg"] = cfg
        c["calibC"] = calib_c
        c["officialReplay"] = official_replay_for_policy(scored_actual, pol, c["name"], initial, stake_fraction)
        out.append(c)
    return out


def run_audit(period_vals, atom_store) -> dict[str, Any]:
    audit = live.run_audit(period_vals, atom_store)
    # This additionally records that the official settlement cache is present.
    settlement_path = actual.SETTLEMENT
    details = []
    if settlement_path.exists():
        try:
            settlement = json.loads(settlement_path.read_text(encoding="utf-8"))
            details = settlement.get("details") or []
        except Exception:
            details = []
    audit["officialReplayInput"] = {
        "settlementPath": str(settlement_path),
        "exists": settlement_path.exists(),
        "details": len(details),
        "source": "top159 monitor official settlement cache refreshed from official/Gamma where available",
    }
    write_json(OUT_AUDIT_JSON, audit)
    write_text(
        OUT_AUDIT_MD,
        "# 路由 V2 官方真实订单回放：审计\n\n"
        f"- 北京时间：`{bj_now()}`\n"
        f"- 061基线复现：`{audit.get('baselineReplay')}`\n"
        f"- 官方回放输入：`{audit['officialReplayInput']}`\n"
        f"- 审计通过：`{audit.get('passed')}`\n",
    )
    return audit


def fmt_table(payload: dict[str, Any]) -> str:
    best = payload.get("bestCandidate")
    d3 = payload.get("d3e3Candidate")
    lines = [
        "# 路由 V2 官方真实订单回放：对比表",
        "",
        f"- 北京时间：`{payload['beijingTime']}`",
        "- live动作：`无，研究只读`",
        "- 180/365口径：`850U初始 / 每笔1% / 买价0.55 / 满成交 / 真实时间顺序`",
        "- 官方真实订单口径：`官方成交成本 + 官方结算结果；本地策略盈亏不参与真相`",
        "",
        "## 180天 / 365天",
        "",
        "|配置|窗口|交易数|胜/负|胜率|盈亏|期末资金|最大回撤|收益回撤比|保留率|哈希|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for w, b in BASELINE_061.items():
        lines.append(f"|当前061|{w}|{b['trades']}|{b['wins']}/{b['losses']}|{b['winRatePct']:.2f}%|{b['compoundPnl']:,.2f}|{b['endingBankroll']:,.2f}|{b['maxDrawdownUsd']:,.2f}|{b['returnDrawdownRatio']:.2f}|100.00%|`{b['setHash']}`|")
    if d3:
        for r in d3.get("rows", []):
            lines.append(f"|d3e3真实顺序路由|{r['window']}|{r['trades']}|{r['wins']}/{r['losses']}|{r['winRatePct']:.2f}%|{r['compoundPnl']:,.2f}|{r['endingBankroll']:,.2f}|{r['maxDrawdownUsd']:,.2f}|{r['returnDrawdownRatio']:.2f}|{r.get('retentionRateVs061',0):.2f}%|`{r['setHash']}`|")
    if best:
        for r in best.get("rows", []):
            lines.append(f"|路由V2最强|{r['window']}|{r['trades']}|{r['wins']}/{r['losses']}|{r['winRatePct']:.2f}%|{r['compoundPnl']:,.2f}|{r['endingBankroll']:,.2f}|{r['maxDrawdownUsd']:,.2f}|{r['returnDrawdownRatio']:.2f}|{r.get('retentionRateVs061',0):.2f}%|`{r['setHash']}`|")
    lines += [
        "",
        "## 061真实订单官方回放",
        "",
        "|配置|真实订单数|保留|跳过|胜/负|胜率|复利重放盈亏|官方实际盈亏|最大回撤|拦截赢家|拦截输单|加权买价|哈希|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for label, c in [("当前061真实单全部执行", payload.get("officialBaselineCandidate")), ("d3e3真实顺序路由", d3), ("路由V2最强", best)]:
        if not c:
            continue
        r = c.get("officialReplay") if label != "当前061真实单全部执行" else c
        lines.append(
            f"|{label}|{r.get('allOfficialTrades', r.get('logicalTrades', 0))}|{r.get('keptTrades', r.get('logicalTrades', 0))}|{r.get('skippedTrades', 0)}|"
            f"{r.get('wins',0)}/{r.get('losses',0)}|{float(r.get('winRatePct',0)):.2f}%|{float(r.get('compoundPnl',0)):+,.2f}|"
            f"{float(r.get('actualOfficialPnl',0)):+,.2f}|{float(r.get('maxDrawdownUsd',0)):,.2f}|{r.get('interceptedWinners',0)}|{r.get('interceptedLosers',0)}|"
            f"{float(r.get('weightedAvgBuyPrice',0)):.4f}|`{r.get('setHash','-')}`|"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    _enriched, atom_store, period_vals = full.base.build_truth()
    audit = run_audit(period_vals, atom_store)
    if not audit.get("passed"):
        raise SystemExit("baseline audit failed; stop before V2 search")

    cfg = json.loads(actual.CONFIG.read_text(encoding="utf-8"))
    initial = float(cfg.get("base_capital_usd") or 847.091209)
    stake_fraction = float(cfg.get("stake_fraction") or 0.01)
    actual_rows = actual.load_actual_logical_rows(strict_061=False)
    if actual_rows.empty:
        raise RuntimeError("no official actual top159 rows available for replay")
    official_baseline = curve_metrics_actual(actual_rows, "current_061_all_actual_orders", initial, stake_fraction)
    official_baseline.update({
        "allOfficialTrades": int(len(actual_rows)),
        "keptTrades": int(len(actual_rows)),
        "skippedTrades": 0,
        "interceptedWinners": 0,
        "interceptedLosers": 0,
    })

    tasks = load_seed_tasks()
    policies = policy_grid_v2()
    # Always include exact d3e3 task/policy for reference.
    d3payload = json.loads((REPORTS / "top159_061_score_calibration_router_live_order_hyperopt_061_compare_latest.json").read_text(encoding="utf-8"))
    d3best = d3payload["bestCandidate"]
    d3task = (d3best["featureMode"], d3best["modelCfg"], float(d3best["calibC"]))
    if d3task not in tasks:
        tasks.insert(0, d3task)
    d3_policy = V2Policy(**d3best["policy"])
    if d3_policy not in policies:
        policies.insert(0, d3_policy)

    if os.environ.get("TOP159_061_ROUTER_V2_FAST", "0") == "1":
        tasks = [t for t in tasks if t[0] in {"base_num", "atoms_core"}] or tasks
    feature_modes = {t[0] for t in tasks}
    actual_cache = build_actual_router_cache(_enriched, actual_rows, feature_modes)

    all_candidates: list[dict[str, Any]] = []
    d3_candidate: dict[str, Any] | None = None
    done = 0
    total = len(tasks)
    for task in tasks:
        batch = evaluate_task(task, policies, period_vals, atom_store, actual_rows, actual_cache, initial, stake_fraction)
        for c in batch:
            if c.get("featureMode") == d3best.get("featureMode") and c.get("modelCfg") == d3best.get("modelCfg") and abs(float(c.get("calibC")) - float(d3best.get("calibC"))) < 1e-12 and c.get("policy") == d3best.get("policy"):
                d3_candidate = c
            all_candidates.append(c)
        done += 1
        # Keep checkpoint lightweight.
        ranked_preview = sorted(all_candidates, key=lambda x: score_v2(x, official_baseline, d3_candidate["officialReplay"] if d3_candidate else official_baseline), reverse=True)[:30]
        write_json(OUT_CHECKPOINT_JSON, {
            "generatedAt": now_iso(),
            "beijingTime": bj_now(),
            "doneTasks": done,
            "totalTasks": total,
            "evaluatedCandidates": len(all_candidates),
            "topPreview": ranked_preview,
        })

    if d3_candidate is None:
        # Fallback: evaluate exact d3e3 alone if equality check missed due to JSON float/int quirks.
        d3_batch = evaluate_task(d3task, [d3_policy], period_vals, atom_store, actual_rows, actual_cache, initial, stake_fraction)
        d3_candidate = d3_batch[0]
        all_candidates.extend(d3_batch)

    for c in all_candidates:
        ok, reasons = candidate_passes(c, official_baseline, d3_candidate)
        c["passedV2Gate"] = ok
        c["failReasonsV2"] = reasons
        c["scoreV2"] = score_v2(c, official_baseline, d3_candidate["officialReplay"])

    ranked = sorted(all_candidates, key=lambda x: x.get("scoreV2", -1e18), reverse=True)
    strict = [c for c in ranked if c.get("passedV2Gate")]
    best = strict[0] if strict else ranked[0]
    verdict_status = "v2_candidate_passed" if strict else "no_v2_candidate_passed"
    verdict_msg = (
        "路由V2找到同时兼顾官方真实订单回放和180/365的候选；仍只建议影子验证，不直接切真钱。"
        if strict
        else "路由V2没有找到严格优于 d3e3 且官方真实订单回放不差的候选；真钱继续保留061，d3e3继续只作影子参考。"
    )
    payload = {
        "generatedAt": now_iso(),
        "beijingTime": bj_now(),
        "researchOnlyNoLiveChange": True,
        "audit": audit,
        "tasks": len(tasks),
        "policies": len(policies),
        "evaluatedCandidates": len(all_candidates),
        "strictPass": len(strict),
        "baseline061": BASELINE_061,
        "officialBaselineCandidate": official_baseline,
        "d3e3Candidate": d3_candidate,
        "bestCandidate": best,
        "topCandidates": ranked[:200],
        "verdict": {"status": verdict_status, "message": verdict_msg},
    }
    write_json(OUT_LEADERBOARD_JSON, payload)
    write_json(OUT_COMPARE_JSON, payload)
    write_text(OUT_COMPARE_MD, fmt_table(payload))

    md = [
        "# 路由 V2 官方真实订单回放：排行榜",
        "",
        f"- 北京时间：`{payload['beijingTime']}`",
        f"- 任务数：`{payload['tasks']}`；政策数：`{payload['policies']}`；候选数：`{payload['evaluatedCandidates']}`；严格通过：`{payload['strictPass']}`",
        "",
        "|排名|候选|特征|模型|校准C|策略|官方实际盈亏|官方回撤|180盈亏|365盈亏|通过|失败原因|",
        "|---:|---|---|---|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for i, c in enumerate(ranked[:80], start=1):
        rows = {r["window"]: r for r in c["rows"]}
        ar = c["officialReplay"]
        md.append(
            f"|{i}|`{c['name']}`|{c.get('featureMode')}|`{c.get('modelCfg')}`|{c.get('calibC')}|`{c.get('policy')}`|"
            f"{ar.get('actualOfficialPnl',0):+,.2f}|{ar.get('maxDrawdownUsd',0):,.2f}|{rows['180d']['compoundPnl']:,.2f}|{rows['365d']['compoundPnl']:,.2f}|{c.get('passedV2Gate')}|`{','.join(c.get('failReasonsV2') or [])}`|"
        )
    write_text(OUT_LEADERBOARD_MD, "\n".join(md) + "\n")
    verdict = payload["verdict"] | {"bestCandidate": best, "d3e3Candidate": d3_candidate, "officialBaselineCandidate": official_baseline}
    write_json(OUT_VERDICT_JSON, verdict)
    write_text(OUT_VERDICT_MD, f"# 路由 V2 官方真实订单回放：唯一结论\n\n- 状态：`{verdict_status}`\n- 结论：{verdict_msg}\n")
    print(OUT_COMPARE_MD)
    print(json.dumps({"status": verdict_status, "evaluated": len(all_candidates), "strictPass": len(strict), "best": best.get("name")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
