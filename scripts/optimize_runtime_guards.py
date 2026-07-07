#!/usr/bin/env python3
"""
运行时风控离线超参（不影响在线进程）

目标：
1) DOWN 方向阈值加严（down_threshold_delta）
2) DOWN 熔断参数（lookback/min_trades/min_winrate/min_pnl_per_trade/hold_bars）

基于各组合已结算交易日志 prediction_trades.json 做回放，输出每个 group 的推荐参数。
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLYMARKET_DIR = PROJECT_ROOT / "polymarket"
TRADER_CONFIG_PATH_DEFAULT = POLYMARKET_DIR / "trader_configs.json"
TRADER_CONFIG_PATH_70 = POLYMARKET_DIR / "trader_configs_70.json"
REPORTS_DIR = PROJECT_ROOT / "reports"

# 与 multi_prediction_index.ts 保持一致
GROUP_DOWN_DELTA_MAP: Dict[str, float] = {
    "v5_exp10": 0.09,
    "v5_exp11": 0.09,
    "v5_exp13": 0.06,
    "v5_exp14": 0.04,
    "v5_exp15": 0.06,
    "v5_exp16": 0.02,
    "v5_exp17": 0.05,
    "ensemble": 0.08,
    "gru_all": 0.09,
}
DEFAULT_DOWN_DELTA = 0.04

# 与 multi_prediction_index.ts 默认一致
BASE_LOOKBACK_HOURS = 8
BASE_MIN_TRADES = 30
BASE_MIN_WINRATE = 0.40
BASE_MIN_PNL_PER_TRADE = -0.20
BASE_HOLD_BARS = 8
BASE_DQ_API_ERROR_DEGRADE = 0.12
BASE_DQ_API_ERROR_HALT = 0.25
BASE_DQ_API_MIN_OBS = 30
BASE_DQ_BET_SCALE_DEGRADED = 0.60
BASE_DQ_RECOVERY_BARS = 6
BASE_DQ_RECOVERY_STABLE_CHECKS = 2


@dataclass
class TraderCfg:
    name: str
    group: str
    logs_dir: str
    base_prob_threshold: float
    current_down_delta: float


@dataclass
class Trade:
    ts: int
    direction: str
    result: str
    pnl: float
    confidence: float


@dataclass
class GuardParams:
    down_delta: float
    lookback_hours: int
    min_trades: int
    min_winrate: float
    min_pnl_per_trade: float
    hold_bars: int


def _parse_ts(val: Any) -> Optional[int]:
    if not val or not isinstance(val, str):
        return None
    s = val.strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return None


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if v != v:  # NaN
            return float(default)
        return v
    except Exception:
        return float(default)


def _pnl_from_entry(e: Dict[str, Any]) -> float:
    # 优先 simulatedPnl（模拟成交侧），其次 pnl
    v = e.get("simulatedPnl")
    if isinstance(v, (int, float)):
        return float(v)
    v = e.get("pnl")
    if isinstance(v, (int, float)):
        return float(v)
    return 0.0


def _load_trader_cfgs(groups_filter: Optional[set[str]], trader_config_path: Path) -> List[TraderCfg]:
    if not trader_config_path.exists():
        raise FileNotFoundError(f"配置不存在: {trader_config_path}")
    data = json.loads(trader_config_path.read_text(encoding="utf-8"))
    out: List[TraderCfg] = []
    seen_logs = set()
    for row in data:
        if not isinstance(row, dict):
            continue
        group = str(row.get("group", "")).strip()
        if not group:
            continue
        if groups_filter and group not in groups_filter:
            continue
        logs_dir = str(row.get("logsDir", "")).strip()
        if not logs_dir or logs_dir in seen_logs:
            continue
        seen_logs.add(logs_dir)
        base_thr = _to_float(row.get("probThreshold"), 0.55)
        current_delta = GROUP_DOWN_DELTA_MAP.get(group, DEFAULT_DOWN_DELTA)
        out.append(
            TraderCfg(
                name=str(row.get("name", logs_dir)),
                group=group,
                logs_dir=logs_dir,
                base_prob_threshold=base_thr,
                current_down_delta=current_delta,
            )
        )
    return out


def _load_trades_for_cfg(cfg: TraderCfg, since_ts: Optional[int]) -> List[Trade]:
    path = POLYMARKET_DIR / cfg.logs_dir / "prediction_trades.json"
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    out: List[Trade] = []
    for e in raw:
        if not isinstance(e, dict):
            continue
        result = str(e.get("result", "")).lower()
        if result not in ("win", "lose"):
            continue
        direction = str(e.get("direction", "")).upper()
        if direction not in ("UP", "DOWN"):
            continue
        ts = _parse_ts(e.get("settledAt")) or _parse_ts(e.get("timestamp"))
        if ts is None:
            continue
        if since_ts is not None and ts < since_ts:
            continue
        conf = _to_float(e.get("confidence"), 0.0)
        out.append(
            Trade(
                ts=ts,
                direction=direction,
                result=result,
                pnl=_pnl_from_entry(e),
                confidence=conf,
            )
        )
    out.sort(key=lambda x: x.ts)
    return out


def _simulate_one(cfg: TraderCfg, trades: List[Trade], p: GuardParams) -> Dict[str, Any]:
    if not trades:
        return {
            "input_trades": 0,
            "executed_trades": 0,
            "wins": 0,
            "win_rate": 0.0,
            "pnl": 0.0,
            "down_input": 0,
            "down_executed": 0,
            "down_wins": 0,
            "down_win_rate": 0.0,
            "skipped_by_delta": 0,
            "skipped_by_breaker": 0,
            "breaker_triggers": 0,
        }

    down_required = min(0.999, max(0.0, cfg.base_prob_threshold + p.down_delta))
    hold_secs = int(p.hold_bars) * 900
    lookback_secs = int(p.lookback_hours) * 3600

    executed = 0
    wins = 0
    pnl = 0.0
    down_input = 0
    down_executed = 0
    down_wins = 0
    down_pnl = 0.0
    skipped_by_delta = 0
    skipped_by_breaker = 0
    breaker_triggers = 0

    down_until_ts = 0
    # 队列元素: (ts, win_flag, pnl)
    down_window = deque()
    dw_n = 0
    dw_w = 0
    dw_pnl = 0.0

    def trim_window(now_ts: int) -> None:
        nonlocal dw_n, dw_w, dw_pnl
        cutoff = now_ts - lookback_secs
        while down_window and down_window[0][0] < cutoff:
            _, wflag, pval = down_window.popleft()
            dw_n -= 1
            if wflag:
                dw_w -= 1
            dw_pnl -= pval

    for t in trades:
        trim_window(t.ts)
        if t.direction == "DOWN":
            down_input += 1
            if t.confidence < down_required:
                skipped_by_delta += 1
                continue
            if t.ts < down_until_ts:
                skipped_by_breaker += 1
                continue

        executed += 1
        pnl += t.pnl
        if t.result == "win":
            wins += 1

        if t.direction == "DOWN":
            down_executed += 1
            down_pnl += t.pnl
            is_win = t.result == "win"
            if is_win:
                down_wins += 1
            down_window.append((t.ts, is_win, t.pnl))
            dw_n += 1
            if is_win:
                dw_w += 1
            dw_pnl += t.pnl
            if dw_n >= int(p.min_trades):
                wr = dw_w / max(1, dw_n)
                pnl_per_trade = dw_pnl / max(1, dw_n)
                if wr < float(p.min_winrate) or pnl_per_trade < float(p.min_pnl_per_trade):
                    new_until = t.ts + hold_secs
                    if new_until > down_until_ts:
                        down_until_ts = new_until
                        breaker_triggers += 1

    return {
        "input_trades": len(trades),
        "executed_trades": executed,
        "wins": wins,
        "win_rate": (wins / executed) if executed > 0 else 0.0,
        "pnl": pnl,
        "down_input": down_input,
        "down_executed": down_executed,
        "down_wins": down_wins,
        "down_pnl": down_pnl,
        "down_win_rate": (down_wins / down_executed) if down_executed > 0 else 0.0,
        "skipped_by_delta": skipped_by_delta,
        "skipped_by_breaker": skipped_by_breaker,
        "breaker_triggers": breaker_triggers,
        "down_required_conf": down_required,
    }


def _aggregate_metrics(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    out = Counter()
    pnl = 0.0
    down_pnl = 0.0
    for r in rows:
        out["input_trades"] += int(r.get("input_trades", 0))
        out["executed_trades"] += int(r.get("executed_trades", 0))
        out["wins"] += int(r.get("wins", 0))
        out["down_input"] += int(r.get("down_input", 0))
        out["down_executed"] += int(r.get("down_executed", 0))
        out["down_wins"] += int(r.get("down_wins", 0))
        out["skipped_by_delta"] += int(r.get("skipped_by_delta", 0))
        out["skipped_by_breaker"] += int(r.get("skipped_by_breaker", 0))
        out["breaker_triggers"] += int(r.get("breaker_triggers", 0))
        pnl += float(r.get("pnl", 0.0))
        down_pnl += float(r.get("down_pnl", 0.0))
    executed = int(out["executed_trades"])
    down_executed = int(out["down_executed"])
    out_dict = dict(out)
    out_dict["pnl"] = pnl
    out_dict["down_pnl"] = down_pnl
    out_dict["win_rate"] = (int(out["wins"]) / executed) if executed > 0 else 0.0
    out_dict["down_win_rate"] = (int(out["down_wins"]) / down_executed) if down_executed > 0 else 0.0
    out_dict["down_pnl_per_trade"] = (down_pnl / down_executed) if down_executed > 0 else 0.0
    return out_dict


def _score_guard_candidate(metrics: Dict[str, Any], baseline: Dict[str, Any], min_trade_ratio: float) -> float:
    pnl = float(metrics.get("pnl", 0.0))
    executed = int(metrics.get("executed_trades", 0))
    baseline_exec = max(1, int(baseline.get("executed_trades", 0)))
    trade_ratio = executed / baseline_exec
    score = pnl
    wr_delta = float(metrics.get("win_rate", 0.0)) - float(baseline.get("win_rate", 0.0))
    down_wr_delta = float(metrics.get("down_win_rate", 0.0)) - float(baseline.get("down_win_rate", 0.0))
    # 在收益相近时优先“方向修复”和“整体胜率改善”
    score += 320.0 * wr_delta + 680.0 * down_wr_delta
    if trade_ratio < min_trade_ratio:
        # 过度减少交易时施加惩罚，避免“靠几笔幸存交易刷高收益”。
        score -= (min_trade_ratio - trade_ratio) * 2000.0
    return score


def _score_with_data_quality(
    guard_score: float,
    metrics: Dict[str, Any],
    baseline: Dict[str, Any],
    *,
    dq_api_degrade: float,
    dq_api_halt: float,
    dq_api_min_obs: int,
    dq_bet_scale_degraded: float,
    dq_recovery_bars: int,
    dq_recovery_stable_checks: int,
) -> float:
    score = float(guard_score)

    # 结构性约束：halt 阈值必须高于 degrade
    if dq_api_halt <= dq_api_degrade:
        return score - 5000.0
    gap = dq_api_halt - dq_api_degrade
    if gap < 0.05:
        score -= (0.05 - gap) * 1800.0

    # 结合当前组的下跌侧风险强弱，给数据质量门控一个“目标保守度”
    baseline_down_wr = float(baseline.get("down_win_rate", 0.0))
    baseline_down_pnl_pt = float(baseline.get("down_pnl_per_trade", 0.0))
    risk_need = max(0.0, 0.45 - baseline_down_wr) + max(0.0, -baseline_down_pnl_pt / 0.25)
    risk_need = min(1.0, risk_need)

    conservativeness = (
        0.35 * (1.0 - min(1.0, max(0.0, dq_api_halt)))
        + 0.45 * (1.0 - min(1.0, max(0.0, dq_bet_scale_degraded)))
        + 0.15 * min(1.0, max(0.0, dq_recovery_bars / 12.0))
        + 0.05 * min(1.0, max(0.0, dq_recovery_stable_checks / 3.0))
    )
    target_conservativeness = min(1.0, 0.35 + 0.55 * risk_need)
    score -= abs(conservativeness - target_conservativeness) * 800.0

    # 极端保守会让交易质量下降，给轻惩罚
    trade_ratio = int(metrics.get("executed_trades", 0)) / max(1, int(baseline.get("executed_trades", 0)))
    if trade_ratio < 0.75 and dq_bet_scale_degraded < 0.5:
        score -= (0.75 - trade_ratio) * 450.0

    # dq_api_min_obs 过大时会让门控对异常迟钝，过小则抖动，偏离中位值给轻惩罚
    score -= abs(int(dq_api_min_obs) - 30) * 6.0
    return score


def _weighted_mode(values: List[int], weights: List[float]) -> int:
    bucket: Dict[int, float] = {}
    for v, w in zip(values, weights):
        bucket[v] = bucket.get(v, 0.0) + w
    return max(bucket.items(), key=lambda x: x[1])[0]


def _weighted_avg(values: List[float], weights: List[float], digits: int = 3) -> float:
    sw = sum(weights)
    if sw <= 0:
        return round(float(values[0]), digits) if values else 0.0
    return round(sum(v * w for v, w in zip(values, weights)) / sw, digits)


def _pick_middle_float(values: List[float], digits: int = 3) -> float:
    if not values:
        return 0.0
    vals = sorted(float(v) for v in values)
    mid = vals[len(vals) // 2]
    return round(mid, digits)


def _pick_middle_int(values: List[int]) -> int:
    if not values:
        return 0
    vals = sorted(int(v) for v in values)
    return int(vals[len(vals) // 2])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="运行时风控超参回放")
    ap.add_argument("--profile", choices=("default", "70"), default="default", help="default=主线 logs_*；70=独立70链路 logs_70_*")
    ap.add_argument("--trader-config", default="", help="可选：自定义 trader 配置文件路径（默认按 --profile 选择）")
    ap.add_argument("--groups", default="", help="仅优化这些 group，逗号分隔；空=自动取配置中全部")
    ap.add_argument("--window-days", type=int, default=7, help="仅使用最近 N 天已结算交易（默认 7）")
    ap.add_argument("--min-trade-ratio", type=float, default=0.60, help="候选参数最少保留的交易比例（相对基线，默认 0.60）")
    ap.add_argument("--delta-grid", default="0.02,0.04,0.06,0.08,0.10", help="DOWN 阈值增量网格")
    ap.add_argument("--lookback-grid", default="6,8,12", help="熔断回看小时网格")
    ap.add_argument("--min-trades-grid", default="20,30,40", help="熔断最小样本网格")
    ap.add_argument("--min-winrate-grid", default="0.35,0.40,0.45", help="熔断胜率阈值网格")
    ap.add_argument("--min-pnl-grid", default="-0.30,-0.20,-0.10", help="熔断单笔PnL阈值网格")
    ap.add_argument("--hold-bars-grid", default="6,8,12", help="熔断保持 bars 网格")
    ap.add_argument("--dq-api-degrade-grid", default="0.10,0.12,0.15", help="数据质量降级 API 错误率阈值网格")
    ap.add_argument("--dq-api-halt-grid", default="0.20,0.25,0.30", help="数据质量停单 API 错误率阈值网格")
    ap.add_argument("--dq-api-min-obs-grid", default="20,30,40", help="API 错误率最小样本阈值网格")
    ap.add_argument("--dq-bet-scale-degraded-grid", default="0.40,0.60,0.80", help="degraded 模式仓位缩放网格")
    ap.add_argument("--dq-recovery-bars-grid", default="4,6,8,12", help="门控恢复滞后 bars 网格")
    ap.add_argument("--dq-recovery-stable-checks-grid", default="1,2,3", help="恢复前稳定检查次数网格")
    ap.add_argument("--top-k", type=int, default=5, help="每组输出前 K 个候选")
    return ap.parse_args()


def _parse_grid_floats(s: str) -> List[float]:
    vals = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    return vals


def _parse_grid_ints(s: str) -> List[int]:
    vals = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    return vals


def main() -> int:
    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    profile = str(args.profile)
    if str(args.trader_config).strip():
        trader_config_path = Path(str(args.trader_config).strip()).expanduser()
        if not trader_config_path.is_absolute():
            trader_config_path = (PROJECT_ROOT / trader_config_path).resolve()
    else:
        trader_config_path = TRADER_CONFIG_PATH_70 if profile == "70" else TRADER_CONFIG_PATH_DEFAULT

    groups_filter = {g.strip() for g in args.groups.split(",") if g.strip()} or None
    cfgs = _load_trader_cfgs(groups_filter, trader_config_path=trader_config_path)
    if not cfgs:
        print("未找到可优化的 trader 配置。")
        return 1

    now_ts = int(datetime.now(tz=timezone.utc).timestamp())
    since_ts = now_ts - int(args.window_days) * 86400 if args.window_days > 0 else None

    trades_by_cfg: Dict[str, List[Trade]] = {}
    for c in cfgs:
        trades_by_cfg[c.logs_dir] = _load_trades_for_cfg(c, since_ts)

    by_group: Dict[str, List[TraderCfg]] = {}
    for c in cfgs:
        by_group.setdefault(c.group, []).append(c)

    delta_grid = _parse_grid_floats(args.delta_grid)
    lookback_grid = _parse_grid_ints(args.lookback_grid)
    min_trades_grid = _parse_grid_ints(args.min_trades_grid)
    min_winrate_grid = _parse_grid_floats(args.min_winrate_grid)
    min_pnl_grid = _parse_grid_floats(args.min_pnl_grid)
    hold_grid = _parse_grid_ints(args.hold_bars_grid)
    dq_api_degrade_grid = _parse_grid_floats(args.dq_api_degrade_grid)
    dq_api_halt_grid = _parse_grid_floats(args.dq_api_halt_grid)
    dq_api_min_obs_grid = _parse_grid_ints(args.dq_api_min_obs_grid)
    dq_bet_scale_degraded_grid = _parse_grid_floats(args.dq_bet_scale_degraded_grid)
    dq_recovery_bars_grid = _parse_grid_ints(args.dq_recovery_bars_grid)
    dq_recovery_stable_checks_grid = _parse_grid_ints(args.dq_recovery_stable_checks_grid)
    baseline_dq = {
        "dq_api_degrade": BASE_DQ_API_ERROR_DEGRADE,
        "dq_api_halt": BASE_DQ_API_ERROR_HALT,
        "dq_api_min_obs": BASE_DQ_API_MIN_OBS,
        "dq_bet_scale_degraded": BASE_DQ_BET_SCALE_DEGRADED,
        "dq_recovery_bars": BASE_DQ_RECOVERY_BARS,
        "dq_recovery_stable_checks": BASE_DQ_RECOVERY_STABLE_CHECKS,
    }

    all_results: Dict[str, Any] = {}
    best_by_group: Dict[str, Any] = {}

    for group, items in sorted(by_group.items()):
        # 基线：当前 down_delta + 当前默认 breaker
        baseline_rows = []
        group_current_delta = float(items[0].current_down_delta if items else DEFAULT_DOWN_DELTA)
        for c in items:
            p = GuardParams(
                down_delta=c.current_down_delta,
                lookback_hours=BASE_LOOKBACK_HOURS,
                min_trades=BASE_MIN_TRADES,
                min_winrate=BASE_MIN_WINRATE,
                min_pnl_per_trade=BASE_MIN_PNL_PER_TRADE,
                hold_bars=BASE_HOLD_BARS,
            )
            baseline_rows.append(_simulate_one(c, trades_by_cfg.get(c.logs_dir, []), p))
        baseline = _aggregate_metrics(baseline_rows)
        baseline_guard_score = _score_guard_candidate(baseline, baseline, float(args.min_trade_ratio))
        baseline_score = _score_with_data_quality(
            baseline_guard_score,
            baseline,
            baseline,
            dq_api_degrade=baseline_dq["dq_api_degrade"],
            dq_api_halt=baseline_dq["dq_api_halt"],
            dq_api_min_obs=baseline_dq["dq_api_min_obs"],
            dq_bet_scale_degraded=baseline_dq["dq_bet_scale_degraded"],
            dq_recovery_bars=baseline_dq["dq_recovery_bars"],
            dq_recovery_stable_checks=baseline_dq["dq_recovery_stable_checks"],
        )

        # 阶段1：先筛 guard 参数（真实交易回放）
        guard_ranked: List[Tuple[float, Dict[str, Any]]] = []
        for (delta, lb, mn, mwr, mpnl, hold) in itertools.product(
            delta_grid, lookback_grid, min_trades_grid, min_winrate_grid, min_pnl_grid, hold_grid
        ):
            p = GuardParams(
                down_delta=float(delta),
                lookback_hours=int(lb),
                min_trades=int(mn),
                min_winrate=float(mwr),
                min_pnl_per_trade=float(mpnl),
                hold_bars=int(hold),
            )
            rows = [_simulate_one(c, trades_by_cfg.get(c.logs_dir, []), p) for c in items]
            agg = _aggregate_metrics(rows)
            guard_score = _score_guard_candidate(agg, baseline, float(args.min_trade_ratio))
            agg["guard_score"] = guard_score
            agg["guard_params"] = {
                "down_delta": p.down_delta,
                "lookback_hours": p.lookback_hours,
                "min_trades": p.min_trades,
                "min_winrate": p.min_winrate,
                "min_pnl_per_trade": p.min_pnl_per_trade,
                "hold_bars": p.hold_bars,
            }
            guard_ranked.append((guard_score, agg))

        guard_ranked.sort(
            key=lambda x: (
                x[0],
                float(x[1].get("pnl", 0.0)),
                float(x[1].get("win_rate", 0.0)),
                int(x[1].get("executed_trades", 0)),
            ),
            reverse=True,
        )

        # 阶段2：在 guard top 候选上联合搜索数据质量门控参数
        guard_top_n = max(12, int(args.top_k) * 4)
        guard_top = [r for _, r in guard_ranked[:guard_top_n]]
        ranked: List[Tuple[float, Dict[str, Any]]] = []

        baseline_entry = dict(baseline)
        baseline_entry["guard_score"] = baseline_guard_score
        baseline_entry["score"] = baseline_score
        baseline_entry["params"] = {
            "down_delta": group_current_delta,
            "lookback_hours": BASE_LOOKBACK_HOURS,
            "min_trades": BASE_MIN_TRADES,
            "min_winrate": BASE_MIN_WINRATE,
            "min_pnl_per_trade": BASE_MIN_PNL_PER_TRADE,
            "hold_bars": BASE_HOLD_BARS,
            "dq_api_degrade": baseline_dq["dq_api_degrade"],
            "dq_api_halt": baseline_dq["dq_api_halt"],
            "dq_api_min_obs": baseline_dq["dq_api_min_obs"],
            "dq_bet_scale_degraded": baseline_dq["dq_bet_scale_degraded"],
            "dq_recovery_bars": baseline_dq["dq_recovery_bars"],
            "dq_recovery_stable_checks": baseline_dq["dq_recovery_stable_checks"],
        }
        baseline_entry["is_current"] = True
        ranked.append((baseline_score, baseline_entry))

        for guard in guard_top:
            gp = dict(guard.get("guard_params", {}))
            for (
                dq_api_degrade,
                dq_api_halt,
                dq_api_min_obs,
                dq_bet_scale_degraded,
                dq_recovery_bars,
                dq_recovery_stable_checks,
            ) in itertools.product(
                dq_api_degrade_grid,
                dq_api_halt_grid,
                dq_api_min_obs_grid,
                dq_bet_scale_degraded_grid,
                dq_recovery_bars_grid,
                dq_recovery_stable_checks_grid,
            ):
                score = _score_with_data_quality(
                    float(guard.get("guard_score", 0.0)),
                    guard,
                    baseline,
                    dq_api_degrade=float(dq_api_degrade),
                    dq_api_halt=float(dq_api_halt),
                    dq_api_min_obs=int(dq_api_min_obs),
                    dq_bet_scale_degraded=float(dq_bet_scale_degraded),
                    dq_recovery_bars=int(dq_recovery_bars),
                    dq_recovery_stable_checks=int(dq_recovery_stable_checks),
                )
                cand = dict(guard)
                cand["score"] = score
                cand["params"] = {
                    **gp,
                    "dq_api_degrade": float(dq_api_degrade),
                    "dq_api_halt": float(dq_api_halt),
                    "dq_api_min_obs": int(dq_api_min_obs),
                    "dq_bet_scale_degraded": float(dq_bet_scale_degraded),
                    "dq_recovery_bars": int(dq_recovery_bars),
                    "dq_recovery_stable_checks": int(dq_recovery_stable_checks),
                }
                ranked.append((score, cand))

        ranked.sort(
            key=lambda x: (
                x[0],
                float(x[1].get("pnl", 0.0)),
                float(x[1].get("win_rate", 0.0)),
                int(x[1].get("executed_trades", 0)),
            ),
            reverse=True,
        )
        topk = [r for _, r in ranked[: max(1, int(args.top_k))]]
        best = topk[0]
        delta_pnl = float(best["pnl"]) - float(baseline["pnl"])
        delta_wr = float(best["win_rate"]) - float(baseline["win_rate"])
        delta_down_wr = float(best["down_win_rate"]) - float(baseline["down_win_rate"])
        best_summary = {
            "baseline": baseline,
            "best": {
                **best,
                "delta_vs_baseline": {
                    "pnl": delta_pnl,
                    "win_rate": delta_wr,
                    "down_win_rate": delta_down_wr,
                    "executed_trades": int(best["executed_trades"]) - int(baseline["executed_trades"]),
                },
                "keep_current": bool(best.get("is_current")),
            },
            "top": topk,
            "n_traders": len(items),
            "guard_candidates_evaluated": len(guard_ranked),
            "dq_candidates_evaluated": len(ranked),
        }
        all_results[group] = best_summary
        best_by_group[group] = best_summary["best"]["params"]

    # 生成全局推荐（按各组 baseline executed_trades 加权）
    groups = sorted(all_results.keys())
    weights = [max(1.0, float(all_results[g]["baseline"].get("executed_trades", 0))) for g in groups]

    dq_api_degrade_global = _weighted_avg(
        [float(best_by_group[g]["dq_api_degrade"]) for g in groups], weights, digits=3
    )
    dq_api_halt_global = _weighted_avg(
        [float(best_by_group[g]["dq_api_halt"]) for g in groups], weights, digits=3
    )
    dq_api_halt_global = max(round(dq_api_degrade_global + 0.05, 3), dq_api_halt_global)

    global_reco = {
        "down_delta_default": _weighted_avg(
            [float(best_by_group[g]["down_delta"]) for g in groups], weights, digits=3
        ),
        "down_breaker_lookback_hours": _weighted_mode(
            [int(best_by_group[g]["lookback_hours"]) for g in groups], weights
        ),
        "down_breaker_min_trades": _weighted_mode(
            [int(best_by_group[g]["min_trades"]) for g in groups], weights
        ),
        "down_breaker_min_winrate": _weighted_avg(
            [float(best_by_group[g]["min_winrate"]) for g in groups], weights, digits=3
        ),
        "down_breaker_min_pnl_per_trade": _weighted_avg(
            [float(best_by_group[g]["min_pnl_per_trade"]) for g in groups], weights, digits=3
        ),
        "down_breaker_hold_bars": _weighted_mode(
            [int(best_by_group[g]["hold_bars"]) for g in groups], weights
        ),
        "data_quality_api_error_degrade": dq_api_degrade_global,
        "data_quality_api_error_halt": dq_api_halt_global,
        "data_quality_api_min_obs": _weighted_mode(
            [int(best_by_group[g]["dq_api_min_obs"]) for g in groups], weights
        ),
        "data_quality_bet_scale_degraded": _weighted_avg(
            [float(best_by_group[g]["dq_bet_scale_degraded"]) for g in groups], weights, digits=2
        ),
        "data_quality_recovery_bars": _weighted_mode(
            [int(best_by_group[g]["dq_recovery_bars"]) for g in groups], weights
        ),
        "data_quality_recovery_stable_checks": _weighted_mode(
            [int(best_by_group[g]["dq_recovery_stable_checks"]) for g in groups], weights
        ),
    }

    out = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "profile": profile,
        "trader_config": str(trader_config_path),
        "window_days": int(args.window_days),
        "groups": groups,
        "grids": {
            "delta": delta_grid,
            "lookback_hours": lookback_grid,
            "min_trades": min_trades_grid,
            "min_winrate": min_winrate_grid,
            "min_pnl_per_trade": min_pnl_grid,
            "hold_bars": hold_grid,
            "dq_api_error_degrade": dq_api_degrade_grid,
            "dq_api_error_halt": dq_api_halt_grid,
            "dq_api_min_obs": dq_api_min_obs_grid,
            "dq_bet_scale_degraded": dq_bet_scale_degraded_grid,
            "dq_recovery_bars": dq_recovery_bars_grid,
            "dq_recovery_stable_checks": dq_recovery_stable_checks_grid,
        },
        "global_recommendation": global_reco,
        "by_group": all_results,
    }

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "_70" if profile == "70" else ""
    report_file = REPORTS_DIR / f"runtime_guard_tuning{suffix}_{ts}.json"
    report_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    best_file = REPORTS_DIR / f"runtime_guard_best_params{suffix}.json"
    best_file.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    env_file = REPORTS_DIR / f"runtime_guard_best{suffix}.env"
    env_text = "\n".join(
        [
            f"DOWN_THRESHOLD_DELTA_DEFAULT={global_reco['down_delta_default']}",
            f"DOWN_THRESHOLD_DELTA={global_reco['down_delta_default']}",
            f"DOWN_BREAKER_LOOKBACK_HOURS={global_reco['down_breaker_lookback_hours']}",
            f"DOWN_BREAKER_MIN_TRADES={global_reco['down_breaker_min_trades']}",
            f"DOWN_BREAKER_MIN_WINRATE={global_reco['down_breaker_min_winrate']}",
            f"DOWN_BREAKER_MIN_PNL_PER_TRADE={global_reco['down_breaker_min_pnl_per_trade']}",
            f"DOWN_BREAKER_HOLD_BARS={global_reco['down_breaker_hold_bars']}",
            f"DATA_QUALITY_API_ERROR_DEGRADE={global_reco['data_quality_api_error_degrade']}",
            f"DATA_QUALITY_API_ERROR_HALT={global_reco['data_quality_api_error_halt']}",
            f"DATA_QUALITY_API_MIN_OBS={global_reco['data_quality_api_min_obs']}",
            f"DATA_QUALITY_BET_SCALE_DEGRADED={global_reco['data_quality_bet_scale_degraded']}",
            "DATA_QUALITY_BET_SCALE_HALT=0.00",
            f"DATA_QUALITY_RECOVERY_BARS={global_reco['data_quality_recovery_bars']}",
            f"DATA_QUALITY_RECOVERY_STABLE_CHECKS={global_reco['data_quality_recovery_stable_checks']}",
            "",
        ]
    )
    env_file.write_text(env_text, encoding="utf-8")

    print(f"已写入: {report_file}")
    print(f"已更新: {best_file}")
    print(f"建议环境变量: {env_file}")
    for g in groups:
        b = all_results[g]["best"]
        d = b["delta_vs_baseline"]
        p = b["params"]
        print(
            f"[{g}] pnl {d['pnl']:+.2f}, wr {d['win_rate']:+.3f}, down_wr {d['down_win_rate']:+.3f}, "
            f"trades {d['executed_trades']:+d} | delta={p['down_delta']:.2f}, "
            f"breaker={p['lookback_hours']}h/{p['min_trades']}/{p['min_winrate']:.2f}/{p['min_pnl_per_trade']:.2f}/{p['hold_bars']}bar | "
            f"dq={p['dq_api_degrade']:.2f}/{p['dq_api_halt']:.2f}/obs{p['dq_api_min_obs']}/scale{p['dq_bet_scale_degraded']:.2f}/"
            f"rec{p['dq_recovery_bars']}x{p['dq_recovery_stable_checks']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
