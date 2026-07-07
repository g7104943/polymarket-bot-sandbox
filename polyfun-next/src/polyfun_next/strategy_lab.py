from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .quality_model import (
    load_quality_frame,
    run_walk_forward_quality,
    compare_quality_methods,
)
from .realistic_backtest import RealisticMetrics, _metrics, _side_path, _simulate

STAKE_USD = 5.0


@dataclass(frozen=True)
class StrategyVerdict:
    status: str
    live_candidate: str | None
    reason: str
    baseline_method: str
    baseline_pnl: float
    baseline_win_rate_pct: float
    baseline_max_drawdown: float
    checked_candidates: int


def run_strategy_lab(
    *,
    episode_path: str | Path,
    candidate_path: str | Path,
    feature_path: str | Path,
    reports_dir: str | Path,
) -> dict[str, Any]:
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)

    strict_df, strict_features, strict_audit = load_quality_frame(episode_path, candidate_path, feature_path, feature_mode="strict")
    strict_scored, strict_blocks = run_walk_forward_quality(strict_df, strict_features)
    # quality_model uses "all" for the full Polymarket-aligned real window.
    # _tag_metrics renames it to "real" so reports keep the user-facing truth layer clear.
    windows = ["180d", "365d", "all"]
    strict_metrics = compare_quality_methods(strict_scored, windows)

    micro_df, micro_features, micro_audit = load_quality_frame(episode_path, candidate_path, feature_path, feature_mode="microstructure")
    micro_scored, micro_blocks = run_walk_forward_quality(micro_df, micro_features)
    micro_metrics = compare_quality_methods(micro_scored, windows)

    exit_metrics = compare_exit_actions(strict_scored, windows)
    combined = _tag_metrics(strict_metrics, "严格特征") + _tag_metrics(micro_metrics, "盘口聚合特征") + _tag_metrics(exit_metrics, "退出动作")
    verdict = choose_verdict(combined)
    proxy = _load_proxy_context()
    payload = {
        "status": "ok_strategy_lab_complete",
        "strictAudit": asdict(strict_audit),
        "microstructureAudit": asdict(micro_audit),
        "strictBlocks": strict_blocks,
        "microstructureBlocks": micro_blocks,
        "metrics": combined,
        "proxyContext": proxy,
        "verdict": asdict(verdict),
    }
    (reports / "strategy_lab_candidate_loop_leaderboard_latest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (reports / "strategy_lab_unique_verdict_latest.md").write_text(_markdown(payload), encoding="utf-8")
    (reports / "strategy_lab_real_window_compare_latest.md").write_text(_markdown_real(combined, verdict), encoding="utf-8")
    (reports / "strategy_lab_180_365_proxy_compare_latest.md").write_text(_markdown_proxy(proxy), encoding="utf-8")
    (reports / "strategy_lab_canary_contract_latest.md").write_text(_markdown_canary(verdict), encoding="utf-8")
    data_truth = {
        "realRows": int(len(strict_df)),
        "eligibleRows": int((strict_scored["quality_block_id"] >= 0).sum()),
        "realStart": str(strict_df["event_time"].min()),
        "realEnd": str(strict_df["event_time"].max()),
        "candidatePath": str(candidate_path),
        "episodePath": str(episode_path),
        "featurePath": str(feature_path),
        "truthPolicy": "Polymarket token path is highest backtest truth; raw kline proxy is context only",
    }
    (reports / "strategy_lab_data_truth_latest.json").write_text(json.dumps(data_truth, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def compare_exit_actions(scored: pd.DataFrame, windows: list[str]) -> list[RealisticMetrics]:
    eligible = scored[scored["quality_block_id"] >= 0].copy()
    methods = [
        ("同窗口基线_吃卖价持有到期", eligible, {"kind": "hold"}, "同一窗口全吃价并持有到期。"),
        ("固定止盈10_吃卖价", eligible, {"kind": "tp", "tp": 0.10}, "价格上涨10%卖出，否则到期。"),
        ("固定止盈20_吃卖价", eligible, {"kind": "tp", "tp": 0.20}, "价格上涨20%卖出，否则到期。"),
        ("固定止损20_吃卖价", eligible, {"kind": "sl", "sl": 0.20}, "价格下跌20%卖出，否则到期。"),
        ("移动止盈25回撤5_吃卖价", eligible, {"kind": "trail", "arm": 0.25, "back": 0.05}, "盈利25%后回撤5%卖出，否则到期。"),
        ("时间退出75%_吃卖价", eligible, {"kind": "time", "frac": 0.75}, "持有到本周期75%时间卖出。"),
        ("硬规则+持有到期", eligible[eligible.get("rule_gate", False)].copy(), {"kind": "hold"}, "过去验证选出的信心+买价规则。"),
    ]
    out: list[RealisticMetrics] = []
    for window in windows:
        for method, df, spec, note in methods:
            wdf = _window_df(df, window)
            sim = _simulate_exit(wdf, spec)
            out.append(_metrics(method, window, sim, note))
    return out


def choose_verdict(rows: list[dict[str, Any]]) -> StrategyVerdict:
    baseline = _find_metric(rows, "real", "同窗口基线_吃卖价") or _find_metric(rows, "real", "同窗口基线_吃卖价持有到期")
    if baseline is None:
        raise RuntimeError("missing baseline")
    candidates = []
    numerically_strong_but_not_live = []
    for row in rows:
        m = row["metric"]
        if m["window"] != "real" or "基线" in m["method"] or m["method"] == "不交易":
            continue
        pnl_ok = m["pnl"] >= baseline["pnl"] or (m["pnl"] >= baseline["pnl"] * 0.97 and m["pnl_drawdown_ratio"] >= baseline["pnl_drawdown_ratio"] * 1.2)
        win_ok = m["win_rate_pct"] >= baseline["win_rate_pct"]
        dd_ok = m["max_drawdown"] < baseline["max_drawdown"]
        fill_ok = m["winner_fill_rate_pct"] + 1e-9 >= m["loser_fill_rate_pct"] - 2.0
        if pnl_ok and win_ok and dd_ok and fill_ok and m["fills"] >= 50:
            if row["family"] == "退出动作" and m["method"] not in {"硬规则+持有到期"}:
                # Early-exit methods currently use the token price path as a conservative research proxy,
                # but we do not yet have enough sell-side order book proof to assert live executable exits.
                numerically_strong_but_not_live.append(row)
                continue
            candidates.append(row)
    if not candidates:
        extra = ""
        if numerically_strong_but_not_live:
            best_proxy = max(numerically_strong_but_not_live, key=lambda r: (r["metric"]["pnl_drawdown_ratio"], r["metric"]["pnl"]))
            extra = (
                f" 数字最强的研究候选是 `{best_proxy['family']} / {best_proxy['metric']['method']}`，"
                "但提前卖出缺少卖出盘口可成交证明，不能直接进真钱。"
            )
        return StrategyVerdict(
            status="no_live_candidate",
            live_candidate=None,
            reason="所有候选都未同时满足真钱上线门槛：真实窗口胜率不低、回撤更低、盈亏不差、成交偏差不过分、并且买入/卖出都能被官网执行证据证明。" + extra,
            baseline_method=baseline["method"],
            baseline_pnl=float(baseline["pnl"]),
            baseline_win_rate_pct=float(baseline["win_rate_pct"]),
            baseline_max_drawdown=float(baseline["max_drawdown"]),
            checked_candidates=sum(1 for r in rows if r["metric"]["window"] == "real"),
        )
    best = max(candidates, key=lambda r: (r["metric"]["pnl_drawdown_ratio"], r["metric"]["pnl"], r["metric"]["win_rate_pct"]))
    return StrategyVerdict(
        status="canary_candidate_found",
        live_candidate=f"{best['family']} / {best['metric']['method']}",
        reason="候选满足真实窗口最低门槛；仍需小额金丝雀验证官方订单链。",
        baseline_method=baseline["method"],
        baseline_pnl=float(baseline["pnl"]),
        baseline_win_rate_pct=float(baseline["win_rate_pct"]),
        baseline_max_drawdown=float(baseline["max_drawdown"]),
        checked_candidates=sum(1 for r in rows if r["metric"]["window"] == "real"),
    )


def _simulate_exit(df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    x = df.copy()
    if x.empty:
        x["sim_fill_fraction"] = []
        x["sim_fill_price"] = []
        x["sim_fill_delay_sec"] = []
        x["sim_pnl"] = []
        return x
    side_paths = x.apply(_side_path, axis=1)
    pnls = []
    prices = []
    delays = []
    for idx, (_, token_path) in enumerate(side_paths):
        won = bool(x.iloc[idx].get("won_if_filled", False))
        if len(token_path) == 0:
            pnls.append(0.0); prices.append(np.nan); delays.append(np.nan); continue
        entry = float(token_path[0])
        pnl = _pnl_for_exit(token_path, won, spec)
        pnls.append(pnl); prices.append(entry); delays.append(0.0)
    x["sim_fill_fraction"] = 1.0
    x["sim_fill_price"] = prices
    x["sim_fill_delay_sec"] = delays
    x["sim_pnl"] = pnls
    return x


def _pnl_for_exit(path: list[float], won: bool, spec: dict[str, Any]) -> float:
    arr = np.asarray(path, dtype=float)
    entry = max(0.001, float(arr[0]))
    kind = spec.get("kind", "hold")
    if kind == "tp":
        hits = np.flatnonzero(arr >= entry * (1 + float(spec["tp"])))
        if len(hits):
            sell = max(0.001, min(0.999, float(arr[int(hits[0])])))
            return STAKE_USD * (sell / entry - 1.0)
    elif kind == "sl":
        hits = np.flatnonzero(arr <= entry * (1 - float(spec["sl"])))
        if len(hits):
            sell = max(0.001, min(0.999, float(arr[int(hits[0])])))
            return STAKE_USD * (sell / entry - 1.0)
    elif kind == "trail":
        arm = entry * (1 + float(spec["arm"]))
        peak = entry
        armed = False
        for p in arr:
            p = float(p)
            if not armed and p >= arm:
                armed = True; peak = p
            elif armed:
                peak = max(peak, p)
                if p <= peak * (1 - float(spec["back"])):
                    sell = max(0.001, min(0.999, p))
                    return STAKE_USD * (sell / entry - 1.0)
    elif kind == "time":
        i = int(round((len(arr) - 1) * float(spec["frac"])))
        sell = max(0.001, min(0.999, float(arr[i])))
        return STAKE_USD * (sell / entry - 1.0)
    if won:
        return STAKE_USD * (1.0 / entry - 1.0)
    return -STAKE_USD


def _tag_metrics(metrics: list[RealisticMetrics], family: str) -> list[dict[str, Any]]:
    out = []
    for m in metrics:
        d = asdict(m)
        if d["window"] == "all":
            d["window"] = "real"
        out.append({"family": family, "metric": d})
    return out


def _find_metric(rows: list[dict[str, Any]], window: str, method: str) -> dict[str, Any] | None:
    for r in rows:
        if r["metric"]["window"] == window and r["metric"]["method"] == method:
            return r["metric"]
    return None


def _window_df(df: pd.DataFrame, window: str) -> pd.DataFrame:
    if df.empty or window in {"all", "real"}:
        return df.copy()
    days = int(window.rstrip("d"))
    end = df["event_time"].max()
    return df[df["event_time"] >= end - pd.Timedelta(days=days)].copy()


def _load_proxy_context() -> dict[str, Any]:
    p = Path("/Users/mac/polyfun/reports/proxy_180_365_fair_compare_latest.json")
    if not p.exists():
        return {"status": "missing"}
    data = json.loads(p.read_text())
    return {
        "status": "available_proxy_only",
        "source": str(p),
        "note": "raw kline proxy only; not used for live acceptance",
        "strictProxyRows": data.get("strictProxyRows", [])[:12],
    }


def _markdown(payload: dict[str, Any]) -> str:
    verdict = payload["verdict"]
    lines = ["# Strategy Lab 自循环唯一结论", ""]
    lines.append("## 结论")
    lines.append(f"- 状态：`{verdict['status']}`")
    lines.append(f"- 可上线候选：`{verdict['live_candidate']}`")
    lines.append(f"- 原因：{verdict['reason']}")
    lines.append("")
    lines.append("## 真实窗口候选榜")
    lines += _table([r for r in payload["metrics"] if r["metric"]["window"] == "real"])
    lines.append("")
    lines.append("## 说明")
    lines.append("- `real` 是当前可对齐的 Polymarket token 路径窗口，不使用原始K线冒充真钱。")
    lines.append("- 3/5/7年代理只作为压力测试上下文，不参与真钱通过判定。")
    lines.append("- 若没有 live candidate，系统明确禁止为了上线硬凑策略。")
    return "\n".join(lines) + "\n"


def _markdown_real(rows: list[dict[str, Any]], verdict: StrategyVerdict) -> str:
    return "\n".join(["# Strategy Lab 真实窗口对比", "", f"结论：{verdict.status}，{verdict.reason}", ""] + _table([r for r in rows if r["metric"]["window"] == "real"])) + "\n"


def _markdown_proxy(proxy: dict[str, Any]) -> str:
    lines = ["# Strategy Lab 180/365 代理上下文", "", f"状态：`{proxy.get('status')}`", "", "说明：原始K线代理不证明真钱可成交，只用于压力测试。", ""]
    rows = proxy.get("strictProxyRows", [])
    if rows:
        lines.append("|对象|窗口|交易数|胜率|盈亏|最大回撤|收益回撤比|备注|")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
        for r in rows:
            lines.append(f"|{r.get('object')}|{r.get('window')}|{r.get('trades')}|{r.get('winRate')}%|{r.get('pnl')}|{r.get('maxDrawdown')}|{r.get('pnlDdRatio')}|{r.get('note','')}|")
    return "\n".join(lines) + "\n"


def _markdown_canary(verdict: StrategyVerdict) -> str:
    if verdict.status != "canary_candidate_found":
        return "# Strategy Lab 金丝雀合同\n\n当前没有满足真钱门槛的候选，不生成上线合同。\n"
    return f"# Strategy Lab 金丝雀合同\n\n候选：{verdict.live_candidate}\n\n仅允许 5U 小额金丝雀，必须 OFFICIAL_MISSING=0。\n"


def _table(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["|家族|方法|交易数|胜/负|胜率|盈亏|最大回撤|收益回撤比|赢家成交率|输家成交率|均价|哈希|", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        m = r["metric"]
        lines.append(f"|{r['family']}|{m['method']}|{m['fills']}|{m['wins']}/{m['losses']}|{m['win_rate_pct']:.4f}%|{m['pnl']:.4f}|{m['max_drawdown']:.4f}|{m['pnl_drawdown_ratio']:.6f}|{m['winner_fill_rate_pct']:.4f}%|{m['loser_fill_rate_pct']:.4f}%|{m['avg_fill_price']:.4f}|`{m['set_hash']}`|")
    return lines
