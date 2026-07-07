#!/usr/bin/env python3
from __future__ import annotations

"""Live-orderable score-calibration router search for ETH061.

Research only. This script fixes the key flaw in the earlier full hyperopt:
when daily_cap is used, candidates are selected chronologically within each
Beijing day instead of sorting by the full day's future router probability.
"""

import importlib.util
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(key, os.environ.get("TOP159_061_CALIB_LIVE_ORDER_THREADS", "1"))

import numpy as np
import pandas as pd

ROOT = Path("/Users/mac/polyfun")
REPORTS = ROOT / "reports"
FULL_PATH = ROOT / "polyfun-next" / "scripts" / "run_top159_061_score_calibration_router_full_hyperopt.py"

OUT_AUDIT_MD = REPORTS / "top159_061_score_calibration_router_live_order_hyperopt_bug_audit_latest.md"
OUT_AUDIT_JSON = REPORTS / "top159_061_score_calibration_router_live_order_hyperopt_bug_audit_latest.json"
OUT_LEADERBOARD_MD = REPORTS / "top159_061_score_calibration_router_live_order_hyperopt_leaderboard_latest.md"
OUT_LEADERBOARD_JSON = REPORTS / "top159_061_score_calibration_router_live_order_hyperopt_leaderboard_latest.json"
OUT_COMPARE_MD = REPORTS / "top159_061_score_calibration_router_live_order_hyperopt_061_compare_latest.md"
OUT_COMPARE_JSON = REPORTS / "top159_061_score_calibration_router_live_order_hyperopt_061_compare_latest.json"
OUT_VERDICT_MD = REPORTS / "top159_061_score_calibration_router_live_order_hyperopt_unique_verdict_latest.md"
OUT_VERDICT_JSON = REPORTS / "top159_061_score_calibration_router_live_order_hyperopt_unique_verdict_latest.json"
OUT_CHECKPOINT = REPORTS / "top159_061_score_calibration_router_live_order_hyperopt_checkpoint_latest.json"

START_BANKROLL = 850.0
BUY_PRICE = 0.55
STRESS_BUY_PRICE = 0.60
STAKE_PCT = 0.01


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


full = load_module("score_calibration_router_full_base_live_order", FULL_PATH)
BASELINE_061 = full.BASELINE_061

_PERIOD_VALS: dict[str, pd.DataFrame] | None = None
_ATOM_STORE: dict[str, pd.DataFrame] | None = None
_POLICIES: list["Policy"] | None = None


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
    daily_cap_mode: str = "chronological"

    def key(self, feature_mode: str, model_key: str, calib_c: float) -> str:
        return full.stable_hash({"feature": feature_mode, "model": model_key, "calibC": calib_c, "policy": self.__dict__})


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


def convert_policy(p: Any) -> Policy:
    d = dict(p.__dict__)
    d["daily_cap_mode"] = "chronological"
    return Policy(**d)


def policy_grid(stage: str) -> list[Policy]:
    return [convert_policy(p) for p in full.policy_grid(stage)]


def apply_policy_live_order(df: pd.DataFrame, policy: Policy) -> pd.DataFrame:
    score = pd.to_numeric(df["score15"], errors="coerce").fillna(0.0)
    prob = pd.to_numeric(df["router_prob"], errors="coerce").fillna(0.0)
    mask = score >= policy.score_min
    if policy.require_061:
        mask &= df["current061_keep"].astype(bool)
    if policy.mode == "plain":
        mask &= prob >= policy.prob_min
    elif policy.mode == "directional":
        direction = df["direction"].astype(str).str.upper()
        req = np.where(direction == "UP", float(policy.up_prob_min), float(policy.down_prob_min))
        mask &= prob.to_numpy() >= req
    elif policy.mode == "combo":
        combo = prob + policy.combo_alpha * (score - 0.55)
        mask &= combo >= float(policy.combo_min)
    elif policy.mode == "low_score_extra":
        req = policy.prob_min + np.where(score < float(policy.low_score_cut), policy.low_prob_extra, 0.0)
        mask &= prob.to_numpy() >= req
    else:
        raise ValueError(policy)
    selected = df[mask].copy().sort_values("dt")
    if policy.daily_cap and not selected.empty:
        selected["bj_day"] = pd.to_datetime(selected["dt"], utc=True).dt.tz_convert("Asia/Shanghai").dt.strftime("%Y-%m-%d")
        selected = selected.groupby("bj_day", group_keys=False).head(int(policy.daily_cap)).sort_values("dt")
    return selected.reset_index(drop=True)


def curve_metrics(rows: pd.DataFrame, name: str, window: str, buy_price: float) -> dict[str, Any]:
    selected = rows.sort_values("dt").reset_index(drop=True)
    n = int(len(selected))
    if n == 0:
        return full.base.curve_metrics(selected, name, window, buy_price)
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
        "monthlyPositiveRatio": full.base.monthly_positive_ratio(curve, selected["dt"]),
        "setHash": full.base.stable_hash(selected[["dt", "pred_up15", "label_up", "won"]].to_dict("records")),
    }


def evaluate_candidate(preds: dict[str, pd.DataFrame], policy: Policy, feature_mode: str, model_key: str, model_cfg: dict[str, Any], calib_c: float) -> dict[str, Any]:
    name = policy.key(feature_mode, model_key, calib_c)
    rows = []
    stress_rows = []
    for w in ["180d", "365d"]:
        selected = apply_policy_live_order(preds[w], policy)
        r = curve_metrics(selected, name, w, BUY_PRICE)
        s = curve_metrics(selected, name, w, STRESS_BUY_PRICE)
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


def init_worker(policies: list[Policy]) -> None:
    global _PERIOD_VALS, _ATOM_STORE, _POLICIES
    _POLICIES = policies
    _enriched, _ATOM_STORE, _PERIOD_VALS = full.base.build_truth()


def run_model_task(task: tuple[str, dict[str, Any], float]) -> dict[str, Any]:
    if _PERIOD_VALS is None or _ATOM_STORE is None or _POLICIES is None:
        raise RuntimeError("worker globals not initialized")
    feature_mode, cfg, calib_c = task
    model_key = full.stable_hash(cfg)
    try:
        preds: dict[str, pd.DataFrame] = {}
        meta_pair = []
        for window in ["180d", "365d"]:
            p, m = full.fit_predict_one_window(_PERIOD_VALS, _ATOM_STORE, window, feature_mode, cfg, calib_c)
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
        return {"ok": True, "evaluated": evaluated, "strict": strict_count, "top": best[:80], "strictTop": strict_best[:80], "meta": meta_pair, "modelKey": model_key}
    except Exception as exc:
        return {"ok": False, "evaluated": 1, "strict": 0, "top": [], "strictTop": [], "meta": [], "error": {"name": full.stable_hash({"cfg": cfg, "feature": feature_mode, "calib": calib_c, "error": repr(exc)}), "featureMode": feature_mode, "modelKey": model_key, "modelCfg": cfg, "calibC": calib_c, "error": repr(exc)[:800]}}


def run_audit(period_vals, atom_store) -> dict[str, Any]:
    rows = full.base.baseline_rows(period_vals, atom_store)
    checks = []
    for r in rows:
        b = BASELINE_061[r["window"]]
        checks.append({"window": r["window"], "trades": r["trades"], "pnl": r["compoundPnl"], "expectedPnl": b["compoundPnl"], "passed": r["trades"] == b["trades"] and abs(r["compoundPnl"] - b["compoundPnl"]) < 1e-6})
    # Directly test that chronological cap differs from future rank cap on the known old winner.
    old_compare = REPORTS / "top159_061_score_calibration_router_full_hyperopt_061_compare_latest.json"
    cap_check = None
    if old_compare.exists():
        data = json.loads(old_compare.read_text())
        old = data.get("bestCandidate") or {}
        try:
            pol = Policy(**(old.get("policy") | {"daily_cap_mode": "chronological"}))
            sample, _ = full.fit_predict_one_window(period_vals, atom_store, "365d", old["featureMode"], old["modelCfg"], old["calibC"])
            chronological = curve_metrics(apply_policy_live_order(sample, pol), "chronological", "365d", BUY_PRICE)
            future_rank = old["rows"][1]
            cap_check = {
                "oldFutureRankPnl": future_rank.get("compoundPnl"),
                "chronologicalPnl": chronological.get("compoundPnl"),
                "provesFutureRankNotUsed": chronological.get("setHash") != future_rank.get("setHash"),
                "chronologicalHash": chronological.get("setHash"),
            }
        except Exception as exc:
            cap_check = {"error": repr(exc)}
    audit = {"generatedAt": full.base.now_iso(), "beijingTime": full.base.bj_now(), "researchOnlyNoLiveChange": True, "timePolicy": "closed_candle_available_at_v2", "dailyCapPolicy": "chronological_first_n_by_beijing_day_no_future_sort", "baselineReplay": checks, "dailyCapAudit": cap_check, "passed": all(x["passed"] for x in checks)}
    write_json(OUT_AUDIT_JSON, audit)
    write_text(OUT_AUDIT_MD, "# 061 校准路由真实顺序版：审计\n\n" + f"- 北京时间：`{audit['beijingTime']}`\n- live动作：`无，研究只读`\n- 当前061复现：`{checks}`\n- 每日上限口径：`{audit['dailyCapPolicy']}`\n- 每日上限审计：`{cap_check}`\n- 审计通过：`{audit['passed']}`\n")
    return audit


def run_task_batch(tasks: list[tuple[str, dict[str, Any], float]], policies: list[Policy], workers: int, label: str):
    candidates: list[dict[str, Any]] = []
    strict_candidates: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    metas: list[dict[str, Any]] = []
    evaluated_total = 0
    strict_total = 0
    ctx = mp.get_context(os.environ.get("TOP159_061_CALIB_LIVE_ORDER_MP_MODE", "spawn"))
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=init_worker, initargs=(policies,)) as ex:
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
            candidates.sort(key=score_candidate, reverse=True)
            del candidates[800:]
            strict_candidates.sort(key=score_candidate, reverse=True)
            del strict_candidates[800:]
            write_json(OUT_CHECKPOINT, {"generatedAt": full.base.now_iso(), "beijingTime": full.base.bj_now(), "stage": label, "doneTasks": idx, "totalTasks": len(futs), "evaluatedCandidates": evaluated_total, "strictPass": strict_total, "topCandidates": candidates[:50], "topStrictCandidates": strict_candidates[:50], "errors": errors[:20]})
    merged: list[dict[str, Any]] = []
    seen = set()
    for c in strict_candidates[:800] + candidates[:800]:
        k = c.get("name")
        if k in seen:
            continue
        seen.add(k)
        merged.append(c)
    merged.sort(key=score_candidate, reverse=True)
    return merged[:1600], evaluated_total, strict_total, metas, errors


def format_compare(payload: dict[str, Any]) -> str:
    lines = [
        "# 061 校准路由真实顺序版：公平对比",
        "",
        f"- 北京时间：`{payload['beijingTime']}`",
        "- live动作：`无，研究只读`",
        "- 口径：`850U初始 / 每笔1% / 满成交 / 买价0.55 / 每日上限按时间顺序累计`",
        "",
        "|配置|窗口|交易数|胜/负|胜率|盈亏|期末资金|最大回撤|收益回撤比|月正收益|哈希|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for w, b in BASELINE_061.items():
        lines.append(f"|当前061|{w}|{b['trades']}|{b['wins']}/{b['losses']}|{b['winRatePct']:.2f}%|{b['compoundPnl']:,.2f}|{b['endingBankroll']:,.2f}|{b['maxDrawdownUsd']:,.2f}|{b['returnDrawdownRatio']:.2f}|{b['monthlyPositiveRatio']:.2%}|`{b['setHash']}`|")
    c = payload.get("bestCandidate")
    if c:
        for r in c["rows"]:
            lines.append(f"|真实顺序校准路由最强|{r['window']}|{r['trades']}|{r['wins']}/{r['losses']}|{r['winRatePct']:.2f}%|{r['compoundPnl']:,.2f}|{r['endingBankroll']:,.2f}|{r['maxDrawdownUsd']:,.2f}|{r['returnDrawdownRatio']:.2f}|{r['monthlyPositiveRatio']:.2%}|`{r['setHash']}`|")
    lines += ["", "## 0.60 买价压力", "", "|配置|窗口|交易数|胜率|盈亏|最大回撤|", "|---|---:|---:|---:|---:|---:|"]
    if c:
        for r in c["stressRows"]:
            lines.append(f"|真实顺序校准路由最强|{r['window']}|{r['trades']}|{r['winRatePct']:.2f}%|{r['compoundPnl']:,.2f}|{r['maxDrawdownUsd']:,.2f}|")
    return "\n".join(lines) + "\n"


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    _enriched, atom_store, period_vals = full.base.build_truth()
    audit = run_audit(period_vals, atom_store)
    if not audit["passed"]:
        raise SystemExit("baseline audit failed")
    workers = max(1, int(os.environ.get("TOP159_061_CALIB_LIVE_ORDER_WORKERS", "4")))
    feature_modes = ["base_num", "atoms_core", "atoms_all"]
    model_cfgs = full.model_grid()
    calib_grid = [0.08, 0.15, 0.25, 0.45, 0.75]
    tasks = [(feature_mode, cfg, calib_c) for feature_mode in feature_modes for cfg in model_cfgs for calib_c in calib_grid]
    coarse_policies = policy_grid("coarse")
    full_policies = policy_grid("full")
    coarse_candidates, coarse_eval, coarse_strict, coarse_metas, coarse_errors = run_task_batch(tasks, coarse_policies, workers, "coarse_live_order_model_screen")
    seen = set()
    for c in coarse_candidates:
        k = (c.get("featureMode"), json.dumps(c.get("modelCfg"), sort_keys=True, default=str), c.get("calibC"))
        seen.add(k)
        if len(seen) >= int(os.environ.get("TOP159_061_CALIB_LIVE_ORDER_TOP_TASKS", "40")):
            break
    focused = []
    for feature_mode, cfg, calib_c in tasks:
        k = (feature_mode, json.dumps(cfg, sort_keys=True, default=str), calib_c)
        if k in seen:
            focused.append((feature_mode, cfg, calib_c))
    candidates, full_eval, full_strict, full_metas, full_errors = run_task_batch(focused, full_policies, workers, "focused_live_order_policy_search")
    valid = [c for c in candidates if "rows" in c]
    valid.sort(key=score_candidate, reverse=True)
    strict = [c for c in valid if c.get("passed")]
    selected = strict[0] if strict else (valid[0] if valid else None)
    payload = {
        "generatedAt": full.base.now_iso(),
        "beijingTime": full.base.bj_now(),
        "researchOnlyNoLiveChange": True,
        "audit": audit,
        "dailyCapPolicy": "chronological_first_n_by_beijing_day_no_future_sort",
        "evaluatedCandidates": coarse_eval + full_eval,
        "strictPass": coarse_strict + full_strict,
        "workers": workers,
        "modelCount": len(model_cfgs),
        "coarsePolicyCount": len(coarse_policies),
        "fullPolicyCount": len(full_policies),
        "focusedTaskCount": len(focused),
        "modelMeta": (coarse_metas[:25] + full_metas[:75]),
        "errors": (coarse_errors + full_errors)[:50],
        "baseline061": BASELINE_061,
        "topCandidates": valid[:200],
        "bestCandidate": selected,
        "verdict": {
            "status": "candidate_beats_061" if selected and selected.get("passed") else "no_candidate_beats_061",
            "message": "真实顺序校准路由找到严格超过当前061的候选；仍只进入影子验证，不改真钱。" if selected and selected.get("passed") else "真实顺序校准路由没有找到严格超过当前061的候选；当前真钱继续保留061。",
        },
    }
    write_json(OUT_LEADERBOARD_JSON, payload)
    lines = [
        "# 061 校准路由真实顺序版：排行榜",
        "",
        f"- 北京时间：`{payload['beijingTime']}`",
        f"- 评估候选：`{payload['evaluatedCandidates']}`；严格通过：`{payload['strictPass']}`",
        "- 每日上限口径：`北京时间内按发生顺序累计，不按未来分数排序`",
        "",
        "|排名|特征|模型|校准C|策略|180天 盈亏/胜率/回撤|365天 盈亏/胜率/回撤|通过|",
        "|---:|---|---|---:|---|---:|---:|---|",
    ]
    for i, c in enumerate(valid[:40]):
        lines.append(
            f"|{i+1}|{c.get('featureMode')}|`{c.get('modelCfg')}`|{c.get('calibC')}|`{c.get('policy')}`|"
            f"{c['rows'][0]['compoundPnl']:,.2f}/{c['rows'][0]['winRatePct']:.2f}%/{c['rows'][0]['maxDrawdownUsd']:,.2f}|"
            f"{c['rows'][1]['compoundPnl']:,.2f}/{c['rows'][1]['winRatePct']:.2f}%/{c['rows'][1]['maxDrawdownUsd']:,.2f}|{c.get('passed')}|"
        )
    write_json(OUT_LEADERBOARD_JSON, payload)
    write_text(OUT_LEADERBOARD_MD, "\n".join(lines) + "\n")
    compare = {"generatedAt": full.base.now_iso(), "beijingTime": full.base.bj_now(), "dailyCapPolicy": payload["dailyCapPolicy"], "baseline061": BASELINE_061, "bestCandidate": selected}
    write_json(OUT_COMPARE_JSON, compare)
    write_text(OUT_COMPARE_MD, format_compare(compare))
    write_json(OUT_VERDICT_JSON, payload["verdict"] | {"bestCandidate": selected})
    write_text(OUT_VERDICT_MD, f"# 061 校准路由真实顺序版：唯一结论\n\n- 状态：`{payload['verdict']['status']}`\n- 结论：{payload['verdict']['message']}\n")


if __name__ == "__main__":
    main()
