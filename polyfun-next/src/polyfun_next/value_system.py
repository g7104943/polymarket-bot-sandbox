from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .quality_model import compare_quality_methods, load_quality_frame, run_walk_forward_quality
from .realistic_backtest import _side_path

STAKE_USD = 5.0
SELL_HAIRCUT = 0.01


@dataclass(frozen=True)
class ActionSpec:
    name: str
    kind: str
    take_profit: float = 0.0
    stop_loss: float = 0.0
    trailing_arm: float = 0.0
    trailing_back: float = 0.0
    time_fraction: float = 1.0


@dataclass(frozen=True)
class ValueMetrics:
    method: str
    window: str
    candidates: int
    fills: int
    wins: int
    losses: int
    win_rate_pct: float
    pnl: float
    max_drawdown: float
    pnl_drawdown_ratio: float
    avg_buy_price: float
    avg_exit_price: float
    take_profit_count: int
    stop_loss_count: int
    trailing_count: int
    time_exit_count: int
    hold_to_settle_count: int
    skipped_count: int
    settlement_winner_fill_rate_pct: float
    settlement_loser_fill_rate_pct: float
    set_hash: str
    note: str


def action_space() -> list[ActionSpec]:
    actions = [ActionSpec("hold_to_settlement", "hold")]
    for tp in [0.10, 0.15, 0.20, 0.25, 0.30]:
        for sl in [0.10, 0.15, 0.20, 0.25, 0.30]:
            actions.append(ActionSpec(f"tp{int(tp*100)}_sl{int(sl*100)}", "tp_sl", take_profit=tp, stop_loss=sl))
    for tp, sl, frac in [(0.15, 0.20, 0.50), (0.20, 0.20, 0.50), (0.20, 0.20, 0.75), (0.25, 0.20, 0.75)]:
        actions.append(
            ActionSpec(
                f"tp{int(tp*100)}_sl{int(sl*100)}_time{int(frac*100)}",
                "tp_sl",
                take_profit=tp,
                stop_loss=sl,
                time_fraction=frac,
            )
        )
    for arm in [0.15, 0.20, 0.25]:
        for back in [0.05, 0.08, 0.10]:
            actions.append(ActionSpec(f"trail{int(arm*100)}_back{int(back*100)}", "trail", trailing_arm=arm, trailing_back=back))
    for frac in [0.25, 0.50, 0.75]:
        actions.append(ActionSpec(f"time_exit_{int(frac*100)}pct", "time", time_fraction=frac))
    return actions


ACTIONS = action_space()
TEMPLATE_ACTION_NAMES = [
    "hold_to_settlement",
    "tp15_sl20",
    "tp20_sl20",
    "tp20_sl30",
    "tp25_sl20",
    "tp20_sl20_time75",
    "tp20_sl20_time50",
    "trail20_back8",
    "time_exit_50pct",
    "time_exit_75pct",
]
TEMPLATE_ACTIONS = [_action for _action in ACTIONS if _action.name in TEMPLATE_ACTION_NAMES]
GATES = {
    "all": "全部候选",
    "entry_gate": "价值门",
    "entry_fill_gate": "价值+成交门",
    "rule_gate": "硬规则门",
}


def run_value_system(
    *,
    episode_path: str | Path,
    candidate_path: str | Path,
    feature_path: str | Path,
    reports_dir: str | Path,
) -> dict[str, Any]:
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)

    strict_df, features, audit = load_quality_frame(episode_path, candidate_path, feature_path, feature_mode="strict")
    # Use monthly walk-forward blocks for this value-system layer. It preserves the key rule
    # (future blocks only use past data) while avoiding dozens of repeated model fits.
    scored, blocks = run_walk_forward_quality(strict_df, features, warmup_days=45, block_days=30, validation_days=21)
    eligible = scored[scored["quality_block_id"] >= 0].copy().sort_values("event_time").reset_index(drop=True)

    bugcheck = _bugcheck(eligible)
    rows = _absolute_rows(eligible, ["real", "180d", "365d"])
    wf = _walk_forward_template_strategy(eligible)
    for window in ["real", "180d", "365d"]:
        rows.append(_metrics_for_method("自适应_价值成交风控门", window, _window_df(wf, window), _window_df(eligible, window), "每个未来块只用过去数据选择门和退出动作。"))

    verdict = _choose_verdict(rows)
    payload = {
        "status": "ok_value_system_rebuilt",
        "truthPolicy": "official/token-path real window first; raw proxy is context only; no live if sell-side execution proof is missing",
        "dataAudit": {
            "rows": int(len(strict_df)),
            "eligibleRows": int(len(eligible)),
            "firstEventTime": str(strict_df["event_time"].min()),
            "lastEventTime": str(strict_df["event_time"].max()),
            "featureCount": len(features),
            "featureHash": audit.feature_set_hash,
            "blocks": blocks,
        },
        "bugcheck": bugcheck,
        "rows": [asdict(r) for r in rows],
        "uniqueVerdict": verdict,
        "deprecatedLiveIdeas": [
            "fixed_20pct_take_profit_without_stop_loss",
            "old_slot1_slot2_dynamic_3_4_5pct",
            "cheap_limit_order_waiting_without_fill_toxicity_model",
        ],
    }
    (reports / "strategy_value_system_absolute_compare_latest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (reports / "strategy_value_system_absolute_compare_latest.md").write_text(
        _markdown(payload), encoding="utf-8"
    )
    (reports / "strategy_value_system_unique_verdict_latest.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (reports / "strategy_value_system_unique_verdict_latest.md").write_text(
        _verdict_markdown(payload), encoding="utf-8"
    )
    (reports / "strategy_value_system_bugcheck_latest.json").write_text(
        json.dumps(bugcheck, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return payload


def _absolute_rows(df: pd.DataFrame, windows: Iterable[str]) -> list[ValueMetrics]:
    methods: list[tuple[str, str, ActionSpec, str]] = [
        ("当前基线_吃卖价持有到结算", "all", ACTIONS[0], "保证买入，持有到结算。"),
        ("价值门_持有到结算", "entry_gate", ACTIONS[0], "只做入场价值模型通过的单。"),
        ("价值成交门_持有到结算", "entry_fill_gate", ACTIONS[0], "入场价值和成交质量都通过。"),
        ("硬规则门_持有到结算", "rule_gate", ACTIONS[0], "过去验证选择的简单信心和买价规则。"),
        ("风控门_tp20_sl20", "all", _action_by_name("tp20_sl20"), "固定止盈20%、固定止损20%，两边都存在。"),
        ("风控门_tp15_sl20", "all", _action_by_name("tp15_sl20"), "固定止盈15%、固定止损20%。"),
        ("风控门_tp20_sl20_time75", "all", _action_by_name("tp20_sl20_time75"), "固定止盈20%、止损20%，没触发则75%周期时间退出。"),
        ("风控门_tp20_sl20_time50", "all", _action_by_name("tp20_sl20_time50"), "固定止盈20%、止损20%，没触发则50%周期时间退出。"),
        ("风控门_trail20_back8", "all", _action_by_name("trail20_back8"), "盈利20%后回撤8%卖出。"),
        ("风控门_time_exit_50pct", "all", _action_by_name("time_exit_50pct"), "持有半个周期后卖出。"),
        ("价值成交门_tp20_sl20", "entry_fill_gate", _action_by_name("tp20_sl20"), "成交质量通过后再用20/20风控。"),
    ]
    rows: list[ValueMetrics] = []
    for window in windows:
        wdf = _window_df(df, window)
        for method, gate, action, note in methods:
            sim = _simulate_template(wdf, gate, action)
            rows.append(_metrics_for_method(method, window, sim, wdf, note))
    return rows


def _walk_forward_template_strategy(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out_parts: list[pd.DataFrame] = []
    for block_id in sorted(df["quality_block_id"].dropna().unique()):
        block = df[df["quality_block_id"] == block_id].copy()
        block_start = block["event_time"].min()
        hist = df[df["event_time"] < block_start].copy()
        if len(hist) < 250:
            chosen_gate, chosen_action = "all", ACTIONS[0]
        else:
            chosen_gate, chosen_action = _choose_template_from_history(hist)
        sim = _simulate_template(block, chosen_gate, chosen_action)
        sim["chosen_gate"] = chosen_gate
        sim["chosen_action"] = chosen_action.name
        out_parts.append(sim)
    return pd.concat(out_parts, ignore_index=True) if out_parts else df.iloc[0:0].copy()


def _choose_template_from_history(hist: pd.DataFrame) -> tuple[str, ActionSpec]:
    best: tuple[float, str, ActionSpec] | None = None
    for gate in GATES:
        for action in TEMPLATE_ACTIONS:
            sim = _simulate_template(hist, gate, action)
            m = _metrics_for_method("hist", "hist", sim, hist, "")
            if m.fills < max(80, int(hist.shape[0] * 0.25)):
                continue
            if m.pnl <= 0 or m.max_drawdown <= 0:
                continue
            score = (m.pnl_drawdown_ratio * 2.5) + (m.win_rate_pct / 20.0) + (m.pnl / 500.0) - (m.max_drawdown / 500.0)
            if best is None or score > best[0]:
                best = (score, gate, action)
    if best is None:
        return "all", ACTIONS[0]
    return best[1], best[2]


def _simulate_template(df: pd.DataFrame, gate_name: str, action: ActionSpec) -> pd.DataFrame:
    x = df.copy()
    if x.empty:
        return _empty_sim(x)
    gate = pd.Series(True, index=x.index)
    if gate_name != "all":
        gate = x.get(gate_name, False).astype(bool)
    pnls: list[float] = []
    entry_prices: list[float] = []
    exit_prices: list[float] = []
    reasons: list[str] = []
    filled: list[bool] = []
    settlement_won: list[bool] = []
    for idx, row in x.iterrows():
        if not bool(gate.loc[idx]):
            pnls.append(0.0)
            entry_prices.append(np.nan)
            exit_prices.append(np.nan)
            reasons.append("skipped_by_gate")
            filled.append(False)
            settlement_won.append(bool(row.get("won_if_filled", False)))
            continue
        _, path = _side_path(row)
        pnl, entry, exit_price, reason = _simulate_action(path, bool(row.get("won_if_filled", False)), action)
        pnls.append(pnl)
        entry_prices.append(entry)
        exit_prices.append(exit_price)
        reasons.append(reason)
        filled.append(True)
        settlement_won.append(bool(row.get("won_if_filled", False)))
    x["value_sim_pnl"] = pnls
    x["value_entry_price"] = entry_prices
    x["value_exit_price"] = exit_prices
    x["value_reason"] = reasons
    x["value_filled"] = filled
    x["value_settlement_won"] = settlement_won
    return x


def _simulate_action(path: list[float], settlement_won: bool, action: ActionSpec) -> tuple[float, float, float, str]:
    if not path:
        return 0.0, np.nan, np.nan, "no_path"
    arr = np.asarray([float(np.clip(p, 0.001, 0.999)) for p in path], dtype=float)
    entry = float(arr[0])

    def sell(i: int, reason: str) -> tuple[float, float, float, str]:
        i = max(0, min(i, len(arr) - 1))
        sell_price = float(np.clip(arr[i] - SELL_HAIRCUT, 0.001, 0.999))
        return STAKE_USD * (sell_price / entry - 1.0), entry, sell_price, reason

    if action.kind == "hold":
        exit_price = 1.0 if settlement_won else 0.0
        return (STAKE_USD * (1.0 / entry - 1.0) if settlement_won else -STAKE_USD), entry, exit_price, "hold_to_settlement"
    if action.kind == "tp_sl":
        tp = entry * (1.0 + action.take_profit)
        sl = entry * (1.0 - action.stop_loss)
        for i, p in enumerate(arr):
            if p >= tp:
                return sell(i, f"take_profit_{int(action.take_profit*100)}")
            if p <= sl:
                return sell(i, f"stop_loss_{int(action.stop_loss*100)}")
        exit_price = 1.0 if settlement_won else 0.0
        if action.time_fraction < 1.0:
            return sell(int(round((len(arr) - 1) * action.time_fraction)), f"time_exit_after_no_barrier_{int(action.time_fraction*100)}")
        return (STAKE_USD * (1.0 / entry - 1.0) if settlement_won else -STAKE_USD), entry, exit_price, "hold_after_no_barrier"
    if action.kind == "trail":
        armed = False
        peak = entry
        arm = entry * (1.0 + action.trailing_arm)
        for i, p in enumerate(arr):
            if not armed and p >= arm:
                armed = True
                peak = p
            elif armed:
                peak = max(peak, p)
                if p <= peak * (1.0 - action.trailing_back):
                    return sell(i, f"trailing_{int(action.trailing_arm*100)}_{int(action.trailing_back*100)}")
        exit_price = 1.0 if settlement_won else 0.0
        return (STAKE_USD * (1.0 / entry - 1.0) if settlement_won else -STAKE_USD), entry, exit_price, "hold_after_no_trailing"
    if action.kind == "time":
        return sell(int(round((len(arr) - 1) * action.time_fraction)), f"time_exit_{int(action.time_fraction*100)}")
    raise ValueError(f"unknown action: {action}")


def _metrics_for_method(method: str, window: str, sim: pd.DataFrame, candidates: pd.DataFrame, note: str) -> ValueMetrics:
    if candidates.empty:
        return ValueMetrics(method, window, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "empty", note)
    filled = sim[sim["value_filled"].astype(bool)].copy() if "value_filled" in sim.columns else sim.iloc[0:0].copy()
    pnls = filled["value_sim_pnl"].astype(float) if len(filled) else pd.Series([], dtype=float)
    wins = int((pnls > 0).sum())
    losses = int((pnls < 0).sum())
    pnl = float(pnls.sum()) if len(pnls) else 0.0
    dd = _max_drawdown(pnls)
    reasons = filled["value_reason"].astype(str) if len(filled) else pd.Series([], dtype=str)
    settlement_winners = candidates[candidates["won_if_filled"].astype(bool)]
    settlement_losers = candidates[~candidates["won_if_filled"].astype(bool)]
    filled_winners = filled[filled["value_settlement_won"].astype(bool)] if len(filled) else filled
    filled_losers = filled[~filled["value_settlement_won"].astype(bool)] if len(filled) else filled
    return ValueMetrics(
        method=method,
        window=window,
        candidates=int(len(candidates)),
        fills=int(len(filled)),
        wins=wins,
        losses=losses,
        win_rate_pct=round(100.0 * wins / len(filled), 4) if len(filled) else 0.0,
        pnl=round(pnl, 4),
        max_drawdown=round(dd, 4),
        pnl_drawdown_ratio=round(pnl / dd, 6) if dd > 0 else 0.0,
        avg_buy_price=round(float(filled["value_entry_price"].mean()), 4) if len(filled) else 0.0,
        avg_exit_price=round(float(filled["value_exit_price"].mean()), 4) if len(filled) else 0.0,
        take_profit_count=int(reasons.str.startswith("take_profit").sum()) if len(reasons) else 0,
        stop_loss_count=int(reasons.str.startswith("stop_loss").sum()) if len(reasons) else 0,
        trailing_count=int(reasons.str.startswith("trailing").sum()) if len(reasons) else 0,
        time_exit_count=int(reasons.str.startswith("time_exit").sum()) if len(reasons) else 0,
        hold_to_settle_count=int(reasons.str.startswith("hold").sum()) if len(reasons) else 0,
        skipped_count=int(len(candidates) - len(filled)),
        settlement_winner_fill_rate_pct=round(100.0 * len(filled_winners) / len(settlement_winners), 4) if len(settlement_winners) else 0.0,
        settlement_loser_fill_rate_pct=round(100.0 * len(filled_losers) / len(settlement_losers), 4) if len(settlement_losers) else 0.0,
        set_hash=_hash_set(filled),
        note=note,
    )


def _choose_verdict(rows: list[ValueMetrics]) -> dict[str, Any]:
    by_key = {(r.method, r.window): r for r in rows}
    baseline_real = by_key.get(("当前基线_吃卖价持有到结算", "real"))
    baseline_180 = by_key.get(("当前基线_吃卖价持有到结算", "180d"))
    baseline_365 = by_key.get(("当前基线_吃卖价持有到结算", "365d"))
    if baseline_real is None:
        return {"status": "error_missing_baseline"}
    candidates = sorted({r.method for r in rows if r.method != baseline_real.method})
    passing = []
    for method in candidates:
        real = by_key.get((method, "real"))
        w180 = by_key.get((method, "180d"))
        w365 = by_key.get((method, "365d"))
        if real is None or w180 is None or w365 is None or baseline_180 is None or baseline_365 is None:
            continue
        tail_ok = real.hold_to_settle_count == 0
        real_ok = tail_ok and real.win_rate_pct >= baseline_real.win_rate_pct and real.max_drawdown < baseline_real.max_drawdown and real.pnl >= baseline_real.pnl * 0.97
        proxy_ok = (w180.pnl >= baseline_180.pnl * 0.97 and w365.pnl >= baseline_365.pnl * 0.97)
        fill_ok = real.settlement_winner_fill_rate_pct + 2.0 >= real.settlement_loser_fill_rate_pct
        if real_ok and proxy_ok and fill_ok and real.fills >= 50:
            improvement_score = (
                (real.win_rate_pct - baseline_real.win_rate_pct)
                + (baseline_real.max_drawdown - real.max_drawdown) / max(1.0, baseline_real.max_drawdown)
                + (real.pnl - baseline_real.pnl) / max(1.0, abs(baseline_real.pnl))
            )
            passing.append((improvement_score, method, real, w180, w365))
    if not passing:
        return {
            "status": "no_live_candidate",
            "reason": "没有候选同时满足：真实窗口胜率不低、最大回撤更低、盈亏不差、180/365不塌、赢家成交率不明显低于输家成交率，并且不能继续持有到结算。",
            "baselineReal": asdict(baseline_real),
        }
    passing.sort(key=lambda x: x[0], reverse=True)
    _, method, real, w180, w365 = passing[0]
    return {
        "status": "candidate_found_requires_canary",
        "method": method,
        "real": asdict(real),
        "w180": asdict(w180),
        "w365": asdict(w365),
        "reason": "候选通过研究门槛，但仍需要官网小额金丝雀证明订单、卖出、撤单、领取闭环。",
    }


def _bugcheck(df: pd.DataFrame) -> dict[str, Any]:
    repeated = bool(df["event_time"].duplicated().any()) if "event_time" in df else False
    invalid_block = int((df.get("quality_block_id", pd.Series(dtype=int)) < 0).sum()) if len(df) else 0
    paths_empty = 0
    sample = df.head(200)
    for _, row in sample.iterrows():
        _, path = _side_path(row)
        if not path:
            paths_empty += 1
    return {
        "duplicateEventTimes": repeated,
        "invalidBlockRows": invalid_block,
        "emptyPathInFirst200": paths_empty,
        "windowIsolation": "180d/365d/real are sliced independently from event_time",
        "futureLabelPolicy": "candidate direction comes from pre-entry candidate stream; actual result and token path are only labels/replay",
    }


def _window_df(df: pd.DataFrame, window: str) -> pd.DataFrame:
    if df.empty or window in {"all", "real"}:
        return df.copy()
    days = int(window.rstrip("d"))
    end = df["event_time"].max()
    return df[df["event_time"] >= end - pd.Timedelta(days=days)].copy()


def _action_by_name(name: str) -> ActionSpec:
    for action in ACTIONS:
        if action.name == name:
            return action
    raise KeyError(name)


def _max_drawdown(pnls: pd.Series) -> float:
    if pnls.empty:
        return 0.0
    curve = pnls.astype(float).cumsum()
    return float((curve.cummax() - curve).max())


def _hash_set(df: pd.DataFrame) -> str:
    if df.empty:
        return "empty"
    cols = [c for c in ["event_time", "candidate_side", "source", "train_window", "verify_window"] if c in df.columns]
    if not cols:
        raw = [str(i) for i in df.index]
    else:
        raw = df[cols].astype(str).agg("|".join, axis=1).tolist()
    return hashlib.sha256("\n".join(sorted(raw)).encode()).hexdigest()[:16]


def _empty_sim(x: pd.DataFrame) -> pd.DataFrame:
    x["value_sim_pnl"] = []
    x["value_entry_price"] = []
    x["value_exit_price"] = []
    x["value_reason"] = []
    x["value_filled"] = []
    x["value_settlement_won"] = []
    return x


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# 先证明能成交，再证明有正期望：ETH 15m 价值系统回测",
        "",
        "## 数据审计",
        f"- 候选总数：{payload['dataAudit']['rows']}",
        f"- 可走前评估数：{payload['dataAudit']['eligibleRows']}",
        f"- 时间范围：{payload['dataAudit']['firstEventTime']} 到 {payload['dataAudit']['lastEventTime']}",
        f"- 特征数：{payload['dataAudit']['featureCount']}，哈希：`{payload['dataAudit']['featureHash']}`",
        f"- 防漏洞：{payload['bugcheck']}",
        "",
        "## 绝对对比表",
        "|方法|窗口|候选数|成交数|胜/负|胜率|盈亏|最大回撤|收益回撤比|平均买价|平均退出价|止盈|止损|时间退出|持有到期|跳过|赢家成交率|输家成交率|集合哈希|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in payload["rows"]:
        lines.append(
            f"|{r['method']}|{r['window']}|{r['candidates']}|{r['fills']}|{r['wins']}/{r['losses']}|{r['win_rate_pct']}%|{r['pnl']}|{r['max_drawdown']}|{r['pnl_drawdown_ratio']}|{r['avg_buy_price']}|{r['avg_exit_price']}|{r['take_profit_count']}|{r['stop_loss_count']}|{r['time_exit_count']}|{r['hold_to_settle_count']}|{r['skipped_count']}|{r['settlement_winner_fill_rate_pct']}%|{r['settlement_loser_fill_rate_pct']}%|`{r['set_hash']}`|"
        )
    lines += [
        "",
        "## 口径",
        "- 胜/负按实际退出后的盈亏计算，不再按最终结算方向硬算。",
        "- `赢家成交率/输家成交率` 仍按最终结算赢家/输家的进入比例，用来检查赢单买不上、输单买满。",
        "- `20%止盈无止损` 没有进入候选；所有风控候选都带止损、移动止盈或时间退出。",
    ]
    return "\n".join(lines) + "\n"


def _verdict_markdown(payload: dict[str, Any]) -> str:
    v = payload["uniqueVerdict"]
    lines = ["# 价值系统唯一结论", "", f"- 状态：`{v.get('status')}`"]
    if v.get("status") == "no_live_candidate":
        lines.append(f"- 结论：{v.get('reason')}")
        lines.append("- 处理：不恢复真钱，不上线 20% 无止损，不上线旧 slot1/slot2。")
    else:
        lines.append(f"- 候选：`{v.get('method')}`")
        lines.append(f"- 说明：{v.get('reason')}")
    return "\n".join(lines) + "\n"
