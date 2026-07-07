#!/usr/bin/env python3
from __future__ import annotations

"""Precision hyperopt for current 061 cluster gate.

Research only. Does not mutate live configs, ledgers, monitor, order state, or
claim state.

Locked metric:
  - 850U initial bankroll
  - stake 1% current bankroll per trade
  - buy price 0.55 primary, full fill
  - closed_candle_available_at_v2 feature alignment via cluster targeted base
"""

import concurrent.futures as cf
import hashlib
import importlib.util
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(key, os.environ.get("TOP159_061_PRECISION_THREADS", "1"))
os.environ.setdefault("TOP159_SHOCK_EXTREME_THREADS", os.environ.get("TOP159_061_PRECISION_THREADS", "1"))

import numpy as np
import pandas as pd

ROOT = Path("/Users/mac/polyfun")
REPORTS = ROOT / "reports"
BASE_SCRIPT = ROOT / "polyfun-next" / "scripts" / "run_top159_shock_filter_cluster_targeted_search.py"
ARCHIVE_SCRIPT = ROOT / "scripts" / "ops" / "run_top159_all_archived_real_eth_compare.py"

START_BANKROLL = 850.0
STAKE_PCT = 0.01
PRIMARY_BUY_PRICE = 0.55
BUY_PRICES = [0.55, 0.60]
RNG_SEED = 20260507
WORKERS = int(os.environ.get("TOP159_061_PRECISION_WORKERS", "4"))
MAX_SECONDS = int(os.environ.get("TOP159_061_PRECISION_MAX_SECONDS", str(3 * 3600)))
PARAM_LIMIT = int(os.environ.get("TOP159_061_PRECISION_PARAM_LIMIT", "1200"))

OUT_AUDIT_MD = REPORTS / "top159_061_precision_hyperopt_bug_audit_latest.md"
OUT_AUDIT_JSON = REPORTS / "top159_061_precision_hyperopt_bug_audit_latest.json"
OUT_RESULTS = REPORTS / "top159_061_precision_hyperopt_results_latest.jsonl"
OUT_LEADERBOARD_MD = REPORTS / "top159_061_precision_hyperopt_leaderboard_latest.md"
OUT_LEADERBOARD_JSON = REPORTS / "top159_061_precision_hyperopt_leaderboard_latest.json"
OUT_COMPARE_MD = REPORTS / "top159_061_precision_hyperopt_061_compare_latest.md"
OUT_COMPARE_JSON = REPORTS / "top159_061_precision_hyperopt_061_compare_latest.json"
OUT_VERDICT_MD = REPORTS / "top159_061_precision_hyperopt_unique_verdict_latest.md"
OUT_VERDICT_JSON = REPORTS / "top159_061_precision_hyperopt_unique_verdict_latest.json"
OUT_CHECKPOINT = REPORTS / "top159_061_precision_hyperopt_checkpoint_latest.json"

BASELINE_061 = {
    "180d": {"trades": 3942, "wins": 2324, "losses": 1618, "winRatePct": 58.954845, "compoundPnl": 11494.591625, "endingBankroll": 12344.591625, "maxDrawdownUsd": 1481.713373, "returnDrawdownRatio": 7.757635, "monthlyPositiveRatio": 1.0, "setHash": "b35313c05e5b66d2"},
    "365d": {"trades": 9152, "wins": 5246, "losses": 3906, "winRatePct": 57.320804, "compoundPnl": 27033.917055, "endingBankroll": 27883.917055, "maxDrawdownUsd": 3499.248929, "returnDrawdownRatio": 7.725634, "monthlyPositiveRatio": 0.846154, "setHash": "ced1cf82642d8f0d"},
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


base = load_module("precision_cluster_base", BASE_SCRIPT)
archive = load_module("precision_archive", ARCHIVE_SCRIPT)


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
        "monthlyPositiveRatio": monthly_positive_ratio(curve, sel["dt"]),
        "setHash": stable_hash(sel[["dt", "pred_up15", "label_up", "won"]].to_dict("records")) if len(sel) else "empty",
    }


def build_truth() -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]], dict[str, pd.DataFrame], dict[str, Any]]:
    enriched, truth = base.ext.load_or_build_enriched()
    atom_store = base.build_atom_store(enriched)
    period_vals = base.build_period_vals(enriched)
    return enriched, atom_store, period_vals, truth


def condition_for_candidate(atom_store: dict[str, dict[str, np.ndarray]], period: str, params: dict[str, Any]) -> np.ndarray:
    return base.condition_for_candidate(atom_store, period, params)


def keep_mask(val: pd.DataFrame, cond: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    score = pd.to_numeric(val["score15"], errors="coerce").fillna(0.0).to_numpy()
    if params["action"] == "hard_block":
        keep = ~cond
    elif params["action"] == "raise_score":
        keep = (~cond) | (score >= float(params["shock_score_min"]))
    elif params["action"] == "soft_deduct":
        keep = (score - np.where(cond, float(params["deduct"]), 0.0)) >= float(params.get("base_score_min", 0.55))
    elif params["action"] == "directional_raise":
        direction = val["direction"].astype(str).str.upper().to_numpy()
        up_min = float(params.get("up_score_min", params.get("shock_score_min", 0.58)))
        down_min = float(params.get("down_score_min", params.get("shock_score_min", 0.58)))
        req = np.where(direction == "UP", up_min, down_min)
        keep = (~cond) | (score >= req)
    elif params["action"] == "score_segment_raise":
        low = score < float(params.get("low_score_cut", 0.58))
        keep = (~cond) | (~low) | (score >= float(params.get("shock_score_min", 0.59)))
    else:
        keep = np.ones(len(val), dtype=bool)
    return keep.astype(bool)


def evaluate_candidate(period_vals: dict[str, pd.DataFrame], atom_store: dict[str, dict[str, np.ndarray]], params: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    price_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    name = params.get("candidate_id") or stable_hash(params)
    for window, period in [("180d", "validation_180d"), ("365d", "validation_365d")]:
        val = period_vals[period].reset_index(drop=True)
        cond = condition_for_candidate(atom_store, period, params)
        keep = keep_mask(val, cond, params)
        selected = val[keep].copy().sort_values("dt").reset_index(drop=True)
        won = val["won"].astype(bool).to_numpy()
        blocked = ~keep
        blocked_rows.append({
            "window": window,
            "blockedTrades": int(blocked.sum()),
            "blockedWinners": int((won & blocked).sum()),
            "blockedLosers": int((~won & blocked).sum()),
            "retentionRate": round(100.0 * int(keep.sum()) / max(1, len(keep)), 6),
            "conditionTrades": int(cond.sum()),
            "conditionWinRatePct": round(100.0 * float(won[cond].mean()), 6) if int(cond.sum()) else 0.0,
        })
        for bp in BUY_PRICES:
            r = curve_metrics(selected, name, window, bp)
            r.update({"blocked": blocked_rows[-1], "action": params.get("action"), "candidateId": name})
            price_rows.append(r)
            if bp == PRIMARY_BUY_PRICE:
                rows.append(r)
    return {"name": name, "params": params, "rows": rows, "priceRows": price_rows, "blockedRows": blocked_rows}


def current_061_params() -> dict[str, Any]:
    p = dict(base.CURRENT_061_PARAMS)
    p["candidate_id"] = "current_live_06173_cluster_gate"
    return p


def mine_clusters(enriched: pd.DataFrame, atom_store: dict[str, dict[str, np.ndarray]], limit: int = 120) -> list[dict[str, Any]]:
    clusters = base.mine_bad_clusters(enriched, atom_store, limit=limit)
    # Ensure current 061 clusters and known 82e374 clusters are included.
    for atoms in base.CURRENT_061_PARAMS.get("any_clusters", []):
        clusters.append({"cluster_id": stable_hash({"atoms": atoms}), "atoms": list(atoms), "score": 999.0, "source": "current061"})
    known = [["15m_same_as_top159", "1h_same_as_top159", "4h_rangeq_ge_0.65"], ["15m_same_as_top159", "1h_vol_ge_1.6", "4h_vol_ge_1.3"]]
    for atoms in known:
        clusters.append({"cluster_id": stable_hash({"atoms": atoms}), "atoms": atoms, "score": 800.0, "source": "prior82e374"})
    dedup: dict[str, dict[str, Any]] = {}
    for c in clusters:
        dedup[c["cluster_id"]] = c
    return list(dedup.values())


def gen_candidates(clusters: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    # Current 061 exact replay.
    rows.append(current_061_params())
    top = clusters[:80]
    score_grid = [round(x, 3) for x in np.arange(0.55, 0.6401, 0.005)]
    deduct_grid = [round(x, 3) for x in np.arange(0.005, 0.0501, 0.005)]
    # Single cluster, precise action variants.
    for c in top:
        atoms = c["atoms"]
        for ss in score_grid:
            rows.append({"family": "cluster_gate", "candidate_id": stable_hash({"single": atoms, "act": "raise", "ss": ss}), "atoms": atoms, "action": "raise_score", "shock_score_min": ss})
        for dd in deduct_grid:
            rows.append({"family": "cluster_gate", "candidate_id": stable_hash({"single": atoms, "act": "deduct", "dd": dd}), "atoms": atoms, "action": "soft_deduct", "deduct": dd, "base_score_min": 0.55})
        rows.append({"family": "cluster_gate", "candidate_id": stable_hash({"single": atoms, "act": "hard"}), "atoms": atoms, "action": "hard_block"})
    # Multi-cluster OR, current-like and prior-like.
    for k, topn in [(2, 48), (3, 34), (4, 24)]:
        for combo in combinations(top[:topn], k):
            any_clusters = [c["atoms"] for c in combo]
            for min_hits in range(1, min(k, 3) + 1):
                for ss in [0.555, 0.56, 0.565, 0.57, 0.575, 0.58, 0.585, 0.59, 0.60, 0.62, 0.64]:
                    rows.append({"family": "cluster_gate", "candidate_id": stable_hash({"any": any_clusters, "min": min_hits, "act": "raise", "ss": ss}), "any_clusters": any_clusters, "min_cluster_hits": min_hits, "action": "raise_score", "shock_score_min": ss})
                for dd in [0.005, 0.01, 0.015, 0.02, 0.03, 0.04]:
                    rows.append({"family": "cluster_gate", "candidate_id": stable_hash({"any": any_clusters, "min": min_hits, "act": "deduct", "dd": dd}), "any_clusters": any_clusters, "min_cluster_hits": min_hits, "action": "soft_deduct", "deduct": dd, "base_score_min": 0.55})
                rows.append({"family": "cluster_gate", "candidate_id": stable_hash({"any": any_clusters, "min": min_hits, "act": "hard"}), "any_clusters": any_clusters, "min_cluster_hits": min_hits, "action": "hard_block"})
    # Direction-specific variants on strongest clusters.
    for c in top[:50]:
        atoms = c["atoms"]
        for up_min in [0.56, 0.57, 0.58, 0.60, 0.62]:
            for down_min in [0.56, 0.57, 0.58, 0.60, 0.62]:
                rows.append({"family": "cluster_gate", "candidate_id": stable_hash({"dir": atoms, "up": up_min, "down": down_min}), "atoms": atoms, "action": "directional_raise", "up_score_min": up_min, "down_score_min": down_min})
    # Dedup and deterministic sample if too many.
    dedup: dict[str, dict[str, Any]] = {}
    for r in rows:
        dedup[r["candidate_id"]] = r
    rows = list(dedup.values())
    if PARAM_LIMIT and len(rows) > PARAM_LIMIT:
        rng = np.random.default_rng(RNG_SEED)
        # Always keep exact 061 and all current/prior seed clusters first.
        must = [r for r in rows if r["candidate_id"] == "current_live_06173_cluster_gate"]
        rest = [r for r in rows if r not in must]
        idx = rng.permutation(len(rest))[: max(0, PARAM_LIMIT - len(must))]
        rows = must + [rest[int(i)] for i in idx]
    return rows


_G_PERIOD_VALS: dict[str, pd.DataFrame] | None = None
_G_ATOMS: dict[str, dict[str, np.ndarray]] | None = None


def init_worker(period_vals: dict[str, pd.DataFrame], atom_store: dict[str, dict[str, np.ndarray]]) -> None:
    global _G_PERIOD_VALS, _G_ATOMS
    _G_PERIOD_VALS = period_vals
    _G_ATOMS = atom_store


def eval_worker(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    if _G_PERIOD_VALS is None or _G_ATOMS is None:
        raise RuntimeError("worker not initialized")
    _, params = item
    try:
        return evaluate_candidate(_G_PERIOD_VALS, _G_ATOMS, params)
    except Exception as exc:
        return {"name": params.get("candidate_id"), "params": params, "error": repr(exc)[:800]}


def pass_gate(c: dict[str, Any]) -> tuple[bool, list[str]]:
    rows = {r["window"]: r for r in c.get("rows", [])}
    reasons: list[str] = []
    for w in ["180d", "365d"]:
        r = rows.get(w)
        b = BASELINE_061[w]
        if not r:
            reasons.append(f"{w}_missing")
            continue
        if r["compoundPnl"] <= b["compoundPnl"]:
            reasons.append(f"{w}_pnl_not_above_061")
        if w == "365d" and r["winRatePct"] < b["winRatePct"]:
            reasons.append("365d_winrate_below_061")
        if r["maxDrawdownUsd"] > b["maxDrawdownUsd"] * 1.05:
            reasons.append(f"{w}_drawdown_too_high")
        if r["trades"] < max(100 if w == "180d" else 200, int(b["trades"] * 0.45)):
            reasons.append(f"{w}_too_few_trades")
    return not reasons, reasons


def score_candidate(c: dict[str, Any]) -> float:
    rows = {r["window"]: r for r in c.get("rows", [])}
    if set(rows) != {"180d", "365d"}:
        return -1e18
    b180, b365 = BASELINE_061["180d"], BASELINE_061["365d"]
    return (
        (rows["180d"]["compoundPnl"] - b180["compoundPnl"]) / max(1.0, b180["compoundPnl"]) * 2500
        + (rows["365d"]["compoundPnl"] - b365["compoundPnl"]) / max(1.0, b365["compoundPnl"]) * 5000
        + (rows["365d"]["winRatePct"] - b365["winRatePct"]) * 35
        - rows["365d"]["maxDrawdownUsd"] / max(1.0, b365["maxDrawdownUsd"]) * 20
    )


def archive_rows(period_vals: dict[str, pd.DataFrame], atom_store: dict[str, dict[str, np.ndarray]], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    if not candidate or not candidate.get("params"):
        return []
    out: list[dict[str, Any]] = []
    try:
        fill_rows, _ = archive.load_live_fill_rows()
        all_eth = archive.aggregate_logical(fill_rows, "全部归档ETH15m live去重")
        strict_mask = fill_rows["sourcePath"].str.contains("slot01|forcedtrade|base_forcedtrade|__live_eth|logs_v5_exp10_bp0480__live_eth", case=False, regex=True)
        strict = archive.aggregate_logical(fill_rows[strict_mask].copy(), "严格旧slot1/ETH相关live去重") if strict_mask.any() else all_eth.iloc[0:0]
        # Archive dates are best covered by validation_180d in existing logic.
        val = period_vals["validation_180d"].reset_index(drop=True)
        cond = condition_for_candidate(atom_store, "validation_180d", candidate["params"])
        keep = keep_mask(val, cond, candidate["params"])
        pred = val.assign(keep=keep.astype(bool))
        for old in [strict, all_eth]:
            if old.empty:
                continue
            merged = old.merge(pred[["dt", "pred_up15", "score15", "keep"]], left_on="marketStart", right_on="dt", how="left")
            chosen = merged[merged["keep"].fillna(False).astype(bool)].copy().sort_values("marketStart")
            if chosen.empty:
                out.append({"scope": old["scopeName"].iloc[0], "oldRealMarkets": int(len(old)), "selectedTrades": 0})
                continue
            won = chosen["pred_up15"].astype(bool).to_numpy() == chosen["actualUp"].astype(bool).to_numpy()
            arch_sel = pd.DataFrame({"dt": chosen["marketStart"], "label_up": chosen["actualUp"].astype(bool), "pred_up15": chosen["pred_up15"].astype(bool), "won": won})
            m = curve_metrics(arch_sel, candidate["name"], str(old["scopeName"].iloc[0]), PRIMARY_BUY_PRICE)
            out.append({"scope": old["scopeName"].iloc[0], "oldRealMarkets": int(len(old)), "selectedTrades": int(len(chosen)), "wins": int(won.sum()), "losses": int(len(won)-int(won.sum())), "winRatePct": round(100*float(won.mean()), 6), "compoundPnl": m["compoundPnl"], "endingBankroll": m["endingBankroll"], "maxDrawdownUsd": m["maxDrawdownUsd"], "setHash": stable_hash(chosen[["marketSlug", "pred_up15", "actualUp"]].to_dict("records"))})
    except Exception as exc:
        out.append({"scope": "archive_error", "error": repr(exc)[:600]})
    return out


def run_audit(period_vals: dict[str, pd.DataFrame], atom_store: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    current = evaluate_candidate(period_vals, atom_store, current_061_params())
    current2 = evaluate_candidate(period_vals, atom_store, current_061_params())
    repeat = []
    for a, b in zip(current["rows"], current2["rows"]):
        repeat.append({"window": a["window"], "hash1": a["setHash"], "hash2": b["setHash"], "pnl1": a["compoundPnl"], "pnl2": b["compoundPnl"], "passed": a["setHash"] == b["setHash"] and a["compoundPnl"] == b["compoundPnl"]})
    baseline_match = []
    for r in current["rows"]:
        b = BASELINE_061[r["window"]]
        baseline_match.append({"window": r["window"], "trades": r["trades"], "winRatePct": r["winRatePct"], "pnl": r["compoundPnl"], "expectedPnl": b["compoundPnl"], "passed": abs(r["compoundPnl"] - b["compoundPnl"]) < 1e-6 and r["trades"] == b["trades"]})
    audit = {"generatedAt": now_iso(), "beijingTime": bj_now(), "researchOnlyNoLiveChange": True, "timePolicy": "closed_candle_available_at_v2", "current061Replay": baseline_match, "repeatability": repeat, "passed": all(x["passed"] for x in baseline_match) and all(x["passed"] for x in repeat)}
    write_json(OUT_AUDIT_JSON, audit)
    write_text(OUT_AUDIT_MD, "\n".join(["# 061 本体精细化超参：审计", "", f"- 北京时间：`{audit['beijingTime']}`", f"- 时间口径：`{audit['timePolicy']}`", f"- 当前061复现：`{baseline_match}`", f"- 重复运行：`{repeat}`", f"- 审计通过：`{audit['passed']}`"]) + "\n")
    return audit


def render(payload: dict[str, Any]) -> None:
    selected = payload.get("selected")
    top = payload.get("topCandidates", [])
    lines = ["# 061 本体精细化超参：061公平对比", "", f"- 北京时间：`{payload['beijingTime']}`", "- 口径：`850U初始 / 每笔1% / 满成交 / 买价0.55`", "- live动作：`无，研究只读`", "", "## 061基线 vs 最强候选", "", "|配置|窗口|交易数|胜/负|胜率|盈亏|期末资金|最大回撤|收益回撤比|拦截赢家|拦截输单|保留率|哈希|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for w in ["180d", "365d"]:
        b = BASELINE_061[w]
        lines.append(f"|当前061|{w}|{b['trades']}|{b['wins']}/{b['losses']}|{b['winRatePct']:.2f}%|{b['compoundPnl']:,.2f}|{b['endingBankroll']:,.2f}|{b['maxDrawdownUsd']:,.2f}|{b['returnDrawdownRatio']:.2f}|-|-|-|`{b['setHash']}`|")
    if selected:
        blocked = {r["window"]: r for r in selected.get("blockedRows", [])}
        for r in selected.get("rows", []):
            br = blocked.get(r["window"], {})
            lines.append(f"|精细化最佳|{r['window']}|{r['trades']}|{r['wins']}/{r['losses']}|{r['winRatePct']:.2f}%|{r['compoundPnl']:,.2f}|{r['endingBankroll']:,.2f}|{r['maxDrawdownUsd']:,.2f}|{r['returnDrawdownRatio']:.2f}|{br.get('blockedWinners','-')}|{br.get('blockedLosers','-')}|{br.get('retentionRate','-')}|`{r['setHash']}`|")
    lines += ["", "## 前30候选", "", "|排名|动作|交易数365|365胜率|365盈亏|365回撤|拦截赢家365|拦截输单365|通过|失败原因|", "|---:|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for i, c in enumerate(top[:30], 1):
        rows = {r["window"]: r for r in c.get("rows", [])}
        brs = {r["window"]: r for r in c.get("blockedRows", [])}
        r365 = rows.get("365d", {})
        b365 = brs.get("365d", {})
        lines.append(f"|{i}|{c.get('params',{}).get('action')}|{r365.get('trades',0)}|{r365.get('winRatePct',0):.2f}%|{r365.get('compoundPnl',0):,.2f}|{r365.get('maxDrawdownUsd',0):,.2f}|{b365.get('blockedWinners','-')}|{b365.get('blockedLosers','-')}|{c.get('passed', False)}|{','.join(c.get('reasons', []))}|")
    write_text(OUT_COMPARE_MD, "\n".join(lines) + "\n")
    write_text(OUT_LEADERBOARD_MD, "\n".join(lines) + "\n")
    v = payload.get("verdict", {})
    vlines = ["# 061 本体精细化超参：唯一结论", "", f"- 状态：`{v.get('status')}`", f"- 结论：{v.get('message')}", "", "## 历史归档真实单纯预测复核", "", "|范围|旧真实市场数|选中|胜/负|胜率|盈亏|最大回撤|", "|---|---:|---:|---:|---:|---:|---:|"]
    for r in payload.get("archiveRows", []):
        vlines.append(f"|{r.get('scope')}|{r.get('oldRealMarkets',0)}|{r.get('selectedTrades',0)}|{r.get('wins',0)}/{r.get('losses',0)}|{r.get('winRatePct',0):.2f}%|{r.get('compoundPnl',0):,.2f}|{r.get('maxDrawdownUsd',0):,.2f}|")
    write_text(OUT_VERDICT_MD, "\n".join(vlines) + "\n")


def write_progress(results: list[dict[str, Any]], total: int, started: float, period_vals: dict[str, pd.DataFrame], atom_store: dict[str, dict[str, np.ndarray]], audit: dict[str, Any], finished: bool) -> None:
    valid = [r for r in results if r.get("rows")]
    ranked = []
    for c in valid:
        ok, reasons = pass_gate(c)
        ranked.append({**c, "passed": ok, "reasons": reasons, "score": score_candidate(c)})
    ranked.sort(key=lambda x: x["score"], reverse=True)
    strict = [r for r in ranked if r.get("passed")]
    selected = strict[0] if strict else (ranked[0] if ranked else None)
    arch = archive_rows(period_vals, atom_store, selected) if selected else []
    arch_bad = any(r.get("selectedTrades", 0) >= 30 and r.get("winRatePct", 100) < 53.0 for r in arch)
    status = "candidate_beats_061" if selected and selected.get("passed") and not arch_bad else "no_candidate_beats_061"
    msg = "找到比当前061更强的精细化候选，只进入影子验证，不改真钱。" if status == "candidate_beats_061" else "061精细化超参没有找到同时提升180/365、归档不差的候选；当前真钱继续保留061。"
    payload = {"generatedAt": now_iso(), "beijingTime": bj_now(), "researchOnlyNoLiveChange": True, "finished": finished, "elapsedSeconds": round(time.time()-started, 3), "workers": WORKERS, "done": len(results), "total": total, "valid": len(valid), "strictPass": len(strict), "audit": audit, "baseline061": BASELINE_061, "topCandidates": ranked[:300], "selected": selected, "archiveRows": arch, "verdict": {"status": status, "message": msg}}
    write_json(OUT_CHECKPOINT, payload)
    write_json(OUT_LEADERBOARD_JSON, payload)
    write_json(OUT_COMPARE_JSON, payload)
    write_json(OUT_VERDICT_JSON, payload["verdict"] | {"selected": selected, "archiveRows": arch})
    render(payload)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    started = time.time()
    enriched, atom_store, period_vals, truth = build_truth()
    audit = run_audit(period_vals, atom_store)
    clusters = mine_clusters(enriched, atom_store, limit=140)
    params = gen_candidates(clusters)
    if OUT_RESULTS.exists():
        OUT_RESULTS.unlink()
    results: list[dict[str, Any]] = []
    last_write = time.time()
    with cf.ProcessPoolExecutor(max_workers=WORKERS, initializer=init_worker, initargs=(period_vals, atom_store)) as ex:
        futs = {ex.submit(eval_worker, (i, p)): i for i, p in enumerate(params)}
        for fut in cf.as_completed(futs):
            results.append(fut.result())
            with OUT_RESULTS.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(results[-1], ensure_ascii=False, default=str) + "\n")
            if time.time() - last_write > 300:
                write_progress(results, len(params), started, period_vals, atom_store, audit, finished=False)
                last_write = time.time()
            if time.time() - started > MAX_SECONDS:
                break
    write_progress(results, len(params), started, period_vals, atom_store, audit, finished=True)


if __name__ == "__main__":
    main()
