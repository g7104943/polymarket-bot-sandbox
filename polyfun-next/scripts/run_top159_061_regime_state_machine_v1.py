#!/usr/bin/env python3
"""Research-only 061 regime state machine V1.

The live 061 strategy is not changed.  This script reuses the audited 061/V2
data chain and evaluates online regime states using only past settled trades.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


ROOT = Path("/Users/mac/polyfun")
SCRIPTS = ROOT / "polyfun-next" / "scripts"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)
V2_PATH = SCRIPTS / "run_top159_061_bayesian_sizer_v2_search.py"

BUY_PRICE = 0.55
INITIAL_FUNDS = 850.0
BASE_STAKE = 0.01

OUT_AUDIT_JSON = REPORTS / "top159_061_regime_state_machine_v1_bug_audit_latest.json"
OUT_AUDIT_MD = REPORTS / "top159_061_regime_state_machine_v1_bug_audit_latest.md"
OUT_LEADERBOARD_JSON = REPORTS / "top159_061_regime_state_machine_v1_leaderboard_latest.json"
OUT_LEADERBOARD_MD = REPORTS / "top159_061_regime_state_machine_v1_leaderboard_latest.md"
OUT_COMPARE_JSON = REPORTS / "top159_061_regime_state_machine_v1_180_365_official_compare_latest.json"
OUT_COMPARE_MD = REPORTS / "top159_061_regime_state_machine_v1_180_365_official_compare_latest.md"
OUT_VERDICT_JSON = REPORTS / "top159_061_regime_state_machine_v1_unique_verdict_latest.json"
OUT_VERDICT_MD = REPORTS / "top159_061_regime_state_machine_v1_unique_verdict_latest.md"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


v2 = load_module(V2_PATH, "regime_v2_source")


@dataclass(frozen=True)
class RegimePolicy:
    name: str
    window: int
    strong_win: float
    weak_win: float
    strong_ret: float
    weak_ret: float
    dd_soft: float
    dd_hard: float
    recovery_level: float
    loss_streak_trigger: int
    pause_bars: int
    stake_weak: float
    stake_normal: float
    stake_strong: float
    stake_abnormal: float
    use_env_confirm: bool
    update_lag: int = 2


@dataclass
class RegimeMetrics:
    rows: int
    trades: int
    skipped: int
    wins: int
    losses: int
    winRate: float
    avgStakePct: float
    avgStakeMultiplier: float
    compoundPnl: float
    endingFunds: float
    maxDrawdown: float
    returnDrawdown: float
    strongCount: int
    normalCount: int
    weakCount: int
    abnormalCount: int
    setHash: str
    originalLiveAmountPnl: Optional[float] = None
    actualPriceDynamicPnl: Optional[float] = None
    weightedAvgBuyPrice: Optional[float] = None


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def digest(parts: Iterable[Any]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode())
        h.update(b"|")
    return h.hexdigest()[:16]


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def money(x: float) -> str:
    return f"{x:,.2f}"


def ret_for_win(won: bool, q: float = BUY_PRICE) -> float:
    return (1.0 / q - 1.0) if won else -1.0


def policy_grid() -> List[RegimePolicy]:
    out: List[RegimePolicy] = []
    stake_sets = {
        "conservative": (0.0075, 0.0100, 0.0115, 0.0050),
        "balanced": (0.0065, 0.0100, 0.0130, 0.0050),
        "aggressive": (0.0050, 0.0100, 0.0130, 0.0050),
    }
    for window in [20, 50, 100, 200]:
        for strong_win in [0.58, 0.60, 0.62]:
            for weak_win in [0.48, 0.50, 0.52]:
                if weak_win >= strong_win:
                    continue
                for dd_soft in [0.03, 0.05, 0.07]:
                    for dd_hard in [0.08, 0.10, 0.12]:
                        if dd_hard <= dd_soft:
                            continue
                        for recovery in [0.98, 0.99, 1.0]:
                            for loss_trigger in [3, 4, 5]:
                                for pause_bars in [0, 4, 8, 12]:
                                    for label, stakes in stake_sets.items():
                                        for env in [False, True]:
                                            nm = digest([window, strong_win, weak_win, dd_soft, dd_hard, recovery, loss_trigger, pause_bars, label, env])
                                            out.append(RegimePolicy(
                                                name=f"regime_v1_{nm}",
                                                window=window,
                                                strong_win=strong_win,
                                                weak_win=weak_win,
                                                strong_ret=0.055,
                                                weak_ret=0.0,
                                                dd_soft=dd_soft,
                                                dd_hard=dd_hard,
                                                recovery_level=recovery,
                                                loss_streak_trigger=loss_trigger,
                                                pause_bars=pause_bars,
                                                stake_weak=stakes[0],
                                                stake_normal=stakes[1],
                                                stake_strong=stakes[2],
                                                stake_abnormal=stakes[3],
                                                use_env_confirm=env,
                                            ))
    # Keep runtime bounded while still covering all families.
    selected: List[RegimePolicy] = []
    for window in [20, 50, 100, 200]:
        chunk = [p for p in out if p.window == window]
        idxs = np.linspace(0, len(chunk) - 1, min(180, len(chunk)), dtype=int)
        selected.extend([chunk[int(i)] for i in idxs])
    return selected


def env_supports_row(row: pd.Series) -> bool:
    direction = str(row.get("direction", ""))
    same_1h = not bool(row.get("1h_opposes_top159", False))
    same_4h = not bool(row.get("4h_opposes_top159", False))
    vol_ok = str(row.get("1h_vol_bucket", "")) != "hi" or str(row.get("4h_vol_bucket", "")) != "hi"
    trend = str(row.get("1h_trend_bucket", ""))
    if direction == "UP":
        return (same_1h or same_4h) and trend != "down" and vol_ok
    if direction == "DOWN":
        return (same_1h or same_4h) and trend != "up" and vol_ok
    return same_1h and same_4h


def choose_state(policy: RegimePolicy, recent_wins: deque, recent_rets: deque, funds: float, peak: float, loss_streak: int, row: pd.Series) -> str:
    dd = (peak - funds) / peak if peak > 0 else 0.0
    if dd >= policy.dd_hard or loss_streak >= policy.loss_streak_trigger:
        return "abnormal"
    if len(recent_wins) >= max(5, policy.window // 2):
        wr = float(np.mean(recent_wins))
        rr = float(np.mean(recent_rets))
        if wr <= policy.weak_win or rr <= policy.weak_ret or dd >= policy.dd_soft:
            return "weak"
        strong = wr >= policy.strong_win and rr >= policy.strong_ret and dd < policy.dd_soft
        if strong and (not policy.use_env_confirm or env_supports_row(row)):
            return "strong"
    return "normal"


def simulate(train: pd.DataFrame, val: pd.DataFrame, policy: Optional[RegimePolicy], *, actual_mode: bool = False) -> RegimeMetrics:
    train = train.sort_values("dt").reset_index(drop=True)
    val = val.sort_values("dt").reset_index(drop=True)
    max_window = 200 if policy is None else policy.window
    recent_wins: deque = deque(maxlen=max_window)
    recent_rets: deque = deque(maxlen=max_window)
    for _, r in train.tail(max_window).iterrows():
        won = bool(r["won"])
        recent_wins.append(1 if won else 0)
        recent_rets.append(ret_for_win(won, BUY_PRICE))

    funds = INITIAL_FUNDS
    peak = funds
    max_dd = 0.0
    trades = wins = losses = skipped = 0
    strong = normal = weak = abnormal = 0
    stake_pcts: List[float] = []
    pending: deque = deque()
    parts: List[Any] = []
    loss_streak = 0
    pause_remaining = 0
    original_pnl = 0.0 if actual_mode else None
    weighted_cost = 0.0
    weighted_amt = 0.0

    def flush_until(i: int):
        nonlocal loss_streak
        while pending and pending[0][0] <= i:
            _, won, q = pending.popleft()
            recent_wins.append(1 if won else 0)
            recent_rets.append(ret_for_win(won, q))

    for i, row in val.iterrows():
        flush_until(i)
        q = BUY_PRICE
        if actual_mode and "actual_avg_price" in row and pd.notna(row["actual_avg_price"]):
            aq = float(row["actual_avg_price"])
            if 0.01 <= aq <= 0.99:
                q = aq
        won = bool(row["won"])
        if policy is None:
            state = "normal"
            stake_pct = BASE_STAKE
        else:
            if pause_remaining > 0:
                state = "abnormal"
                stake_pct = 0.0
                pause_remaining -= 1
            else:
                state = choose_state(policy, recent_wins, recent_rets, funds, peak, loss_streak, row)
                if state == "strong":
                    stake_pct = policy.stake_strong
                elif state == "weak":
                    stake_pct = policy.stake_weak
                elif state == "abnormal":
                    stake_pct = policy.stake_abnormal
                    if policy.pause_bars:
                        pause_remaining = policy.pause_bars
                        stake_pct = 0.0
                else:
                    stake_pct = policy.stake_normal
        pending.append((i + (2 if policy is None else policy.update_lag), won, q))

        if stake_pct <= 0:
            skipped += 1
            if won:
                # Skipped winners/losers are not trades, but they affect future
                # state after settlement via pending.
                pass
            parts.append((row.get("dt"), "skip", won, state))
            continue

        stake = funds * stake_pct
        pnl = stake * (1.0 / q - 1.0) if won else -stake
        funds += pnl
        peak = max(peak, funds)
        max_dd = max(max_dd, peak - funds)
        trades += 1
        wins += 1 if won else 0
        losses += 0 if won else 1
        loss_streak = 0 if won else loss_streak + 1
        stake_pcts.append(stake_pct)
        if state == "strong":
            strong += 1
        elif state == "weak":
            weak += 1
        elif state == "abnormal":
            abnormal += 1
        else:
            normal += 1
        if actual_mode:
            if original_pnl is not None:
                original_pnl += float(row.get("actual_pnl", 0.0) or 0.0)
            cost = float(row.get("actual_cost", 0.0) or 0.0)
            if cost > 0:
                weighted_cost += cost
                weighted_amt += cost * q
        parts.append((row.get("dt"), stake_pct, won, round(funds, 6), state))

    while pending:
        _, won, q = pending.popleft()
        recent_wins.append(1 if won else 0)
        recent_rets.append(ret_for_win(won, q))

    win_rate = wins / trades if trades else 0.0
    pnl = funds - INITIAL_FUNDS
    return RegimeMetrics(
        rows=len(val),
        trades=trades,
        skipped=skipped,
        wins=wins,
        losses=losses,
        winRate=win_rate,
        avgStakePct=float(np.mean(stake_pcts)) if stake_pcts else 0.0,
        avgStakeMultiplier=(float(np.mean(stake_pcts)) / BASE_STAKE) if stake_pcts else 0.0,
        compoundPnl=pnl,
        endingFunds=funds,
        maxDrawdown=max_dd,
        returnDrawdown=(pnl / max_dd if max_dd > 0 else 0.0),
        strongCount=strong,
        normalCount=normal,
        weakCount=weak,
        abnormalCount=abnormal,
        setHash=digest(parts),
        originalLiveAmountPnl=original_pnl,
        actualPriceDynamicPnl=pnl if actual_mode else None,
        weightedAvgBuyPrice=(weighted_amt / weighted_cost if weighted_cost > 0 else None),
    )


def m(met: RegimeMetrics) -> Dict[str, Any]:
    return asdict(met)


def baseline_replicates(base180: RegimeMetrics, base365: RegimeMetrics) -> Dict[str, Any]:
    exp = {
        "180d": {"trades": 3942, "pnl": 11494.59, "dd": 1481.71},
        "365d": {"trades": 9152, "pnl": 27033.92, "dd": 3499.25},
    }
    return {
        "180d": {
            "trades": base180.trades,
            "pnl": base180.compoundPnl,
            "dd": base180.maxDrawdown,
            "passed": base180.trades == exp["180d"]["trades"] and abs(base180.compoundPnl - exp["180d"]["pnl"]) < 1.0 and abs(base180.maxDrawdown - exp["180d"]["dd"]) < 1.0,
        },
        "365d": {
            "trades": base365.trades,
            "pnl": base365.compoundPnl,
            "dd": base365.maxDrawdown,
            "passed": base365.trades == exp["365d"]["trades"] and abs(base365.compoundPnl - exp["365d"]["pnl"]) < 1.0 and abs(base365.maxDrawdown - exp["365d"]["dd"]) < 1.0,
        },
    }


def score_candidate(metrics: Dict[str, RegimeMetrics], base: Dict[str, RegimeMetrics], official: Optional[RegimeMetrics]) -> Tuple[float, List[str], bool]:
    reasons: List[str] = []
    if metrics["365d"].compoundPnl < base["365d"].compoundPnl * 1.20:
        reasons.append("365d_pnl_below_20pct_lift")
    if metrics["180d"].compoundPnl < base["180d"].compoundPnl * 1.10:
        reasons.append("180d_pnl_below_10pct_lift")
    if metrics["365d"].maxDrawdown > base["365d"].maxDrawdown * 1.05:
        reasons.append("365d_dd_above_5pct_limit")
    if metrics["180d"].maxDrawdown > base["180d"].maxDrawdown * 1.05:
        reasons.append("180d_dd_above_5pct_limit")
    if metrics["365d"].trades < base["365d"].trades * 0.90:
        reasons.append("365d_retention_below_90pct")
    if official and official.actualPriceDynamicPnl is not None and official.originalLiveAmountPnl is not None:
        if official.actualPriceDynamicPnl < official.originalLiveAmountPnl:
            reasons.append("official_replay_worse_than_061")
    lift365 = metrics["365d"].compoundPnl - base["365d"].compoundPnl
    lift180 = metrics["180d"].compoundPnl - base["180d"].compoundPnl
    dd_penalty = max(0.0, metrics["365d"].maxDrawdown - base["365d"].maxDrawdown)
    score = lift365 + 0.35 * lift180 - 2.0 * dd_penalty
    if official and official.actualPriceDynamicPnl is not None and official.originalLiveAmountPnl is not None:
        score += 10.0 * (official.actualPriceDynamicPnl - official.originalLiveAmountPnl)
    return score, reasons, not reasons


def main() -> None:
    data = v2.prepare_data()
    base180 = simulate(data["train180"], data["val180"], None)
    base365 = simulate(data["train365"], data["val365"], None)
    official_base = None
    if not data["official_rows"].empty:
        official_base = simulate(data["official_train"], data["official_rows"], None, actual_mode=True)

    baseline = {"180d": base180, "365d": base365}
    base_check = baseline_replicates(base180, base365)

    policies = policy_grid()
    results: List[Dict[str, Any]] = []
    for pol in policies:
        r180 = simulate(data["train180"], data["val180"], pol)
        r365 = simulate(data["train365"], data["val365"], pol)
        official = None
        if not data["official_rows"].empty:
            official = simulate(data["official_train"], data["official_rows"], pol, actual_mode=True)
        metrics = {"180d": r180, "365d": r365}
        score, reasons, passed = score_candidate(metrics, baseline, official)
        results.append({
            "rankScore": score,
            "candidateId": digest([pol.name, r180.setHash, r365.setHash, official.setHash if official else "none"]),
            "policy": asdict(pol),
            "strictPass": passed,
            "failReasons": reasons,
            "metrics": {"180d": m(r180), "365d": m(r365), "official": m(official) if official else None},
        })
    results.sort(key=lambda x: x["rankScore"], reverse=True)
    top = results[0] if results else None
    strict = [r for r in results if r["strictPass"]]

    repeat = []
    for row in results[:10]:
        pol = RegimePolicy(**row["policy"])
        r180 = simulate(data["train180"], data["val180"], pol)
        r365 = simulate(data["train365"], data["val365"], pol)
        repeat.append({
            "candidateId": row["candidateId"],
            "passed": r180.setHash == row["metrics"]["180d"]["setHash"] and r365.setHash == row["metrics"]["365d"]["setHash"],
        })

    rng = np.random.default_rng(7)
    rand_val365 = data["val365"].copy()
    # A permutation keeps 061's >55% unconditional win rate intact, which is
    # still profitable at 0.55 even when features carry no information.  For a
    # leakage smoke test we need labels independent of the feature stream.
    rand_val365["won"] = rng.random(len(rand_val365)) < 0.5
    random_best = -10**18
    for row in results[:30]:
        pol = RegimePolicy(**row["policy"])
        rr = simulate(data["train365"], rand_val365, pol)
        random_best = max(random_best, rr.compoundPnl)
    random_label = {"bestRandomPnl": random_best, "passed": random_best < base365.compoundPnl * 0.35}

    audit = {
        "researchOnlyNoLiveChange": True,
        "timePolicy": "closed_candle_available_at_v2 via audited V2 data chain",
        "baselineReplication": base_check,
        "randomLabelAudit": random_label,
        "repeatability": repeat,
        "passed": all(v["passed"] for v in base_check.values()) and random_label["passed"] and all(r["passed"] for r in repeat),
    }
    write_json(OUT_AUDIT_JSON, audit)
    OUT_AUDIT_MD.write_text(
        "# 061 Regime State Machine V1 审计\n\n"
        f"- 061基线复现: {'通过' if all(v['passed'] for v in base_check.values()) else '失败'}\n"
        f"- 随机标签测试: {'通过' if random_label['passed'] else '失败'}\n"
        f"- 重复运行哈希: {'通过' if all(r['passed'] for r in repeat) else '失败'}\n",
        encoding="utf-8",
    )

    write_json(OUT_LEADERBOARD_JSON, results[:200])
    lines = ["# 061 Regime State Machine V1 排行榜", "", "|排名|候选|通过|180天盈亏|180天回撤|365天盈亏|365天回撤|官方动态盈亏|失败原因|", "|---:|---|---|---:|---:|---:|---:|---:|---|"]
    for i, row in enumerate(results[:20], 1):
        met = row["metrics"]
        official = met.get("official") or {}
        lines.append(f"|{i}|`{row['candidateId']}`|{row['strictPass']}|{money(met['180d']['compoundPnl'])}|{money(met['180d']['maxDrawdown'])}|{money(met['365d']['compoundPnl'])}|{money(met['365d']['maxDrawdown'])}|{money(official.get('actualPriceDynamicPnl') or 0.0)}|{','.join(row['failReasons'])}|")
    OUT_LEADERBOARD_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    compare = {
        "baseline061": {"180d": m(base180), "365d": m(base365), "official": m(official_base) if official_base else None},
        "topCandidate": top,
        "strictPassCount": len(strict),
    }
    write_json(OUT_COMPARE_JSON, compare)
    md = ["# 061 Regime State Machine V1 对比", "", "|配置|窗口|交易数|胜/负|胜率|平均仓位|强势|正常|弱势|异常|盈亏|最大回撤|收益回撤比|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, row in [("当前061", {"180d": m(base180), "365d": m(base365)}), ("V1最佳状态机", top["metrics"] if top else {})]:
        for w in ["180d", "365d"]:
            mm = row[w]
            md.append(f"|{name}|{w}|{mm['trades']}|{mm['wins']}/{mm['losses']}|{pct(mm['winRate'])}|{mm['avgStakeMultiplier']:.3f}x|{mm['strongCount']}|{mm['normalCount']}|{mm['weakCount']}|{mm['abnormalCount']}|{money(mm['compoundPnl'])}|{money(mm['maxDrawdown'])}|{mm['returnDrawdown']:.2f}|")
    md.append("")
    md.append("## 官方实际订单回放")
    md.append("")
    md.append("|配置|订单数|胜/负|胜率|原始061官方盈亏|动态重算盈亏|最大回撤|平均买价|")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, mm in [("当前061", m(official_base) if official_base else None), ("V1最佳状态机", (top["metrics"].get("official") if top else None))]:
        if not mm:
            continue
        md.append(f"|{name}|{mm['trades']}|{mm['wins']}/{mm['losses']}|{pct(mm['winRate'])}|{money(mm.get('originalLiveAmountPnl') or 0.0)}|{money(mm.get('actualPriceDynamicPnl') or 0.0)}|{money(mm['maxDrawdown'])}|{(mm.get('weightedAvgBuyPrice') or 0):.4f}|")
    md.append("")
    md.append(f"严格通过候选数: {len(strict)}")
    OUT_COMPARE_MD.write_text("\n".join(md) + "\n", encoding="utf-8")

    status = "REGIME_STATE_MACHINE_CANDIDATE_FOUND" if strict else "NO_REGIME_STATE_MACHINE_CANDIDATE"
    verdict = {
        "status": status,
        "message": "找到严格通过候选，只允许进入影子验证。" if strict else "没有状态机候选同时满足收益、回撤和官方回放门槛；继续保留当前061。",
        "strictPassCount": len(strict),
        "selected": strict[0] if strict else top,
    }
    write_json(OUT_VERDICT_JSON, verdict)
    OUT_VERDICT_MD.write_text(
        "# 061 Regime State Machine V1 唯一结论\n\n"
        f"- 严格通过候选数: `{len(strict)}`\n"
        f"- 结论: `{verdict['status']}; keep live 061`" + "\n",
        encoding="utf-8",
    )
    print(verdict["status"])


if __name__ == "__main__":
    main()
