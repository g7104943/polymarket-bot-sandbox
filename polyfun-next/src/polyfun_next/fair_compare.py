from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Any

import numpy as np
import pandas as pd

STAKE_USD = 5.0
CANCEL_AFTER_SECONDS = 300


@dataclass(frozen=True)
class Metrics:
    name: str
    window: str
    trades: int
    actual_fills: int
    wins: int
    losses: int
    win_rate_pct: float
    pnl: float
    max_drawdown: float
    pnl_drawdown_ratio: float
    winner_fill_rate_pct: float
    loser_fill_rate_pct: float
    winner_avg_fill_fraction: float
    loser_avg_fill_fraction: float
    avg_fill_fraction: float
    cancels: int
    official_missing: int
    set_hash: str


def load_episodes(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "entry_trace_json" not in df.columns:
        raise ValueError(f"{path} missing entry_trace_json")
    traces = df["entry_trace_json"].apply(_loads_json)
    trace_df = pd.json_normalize(traces).add_prefix("trace_")
    out = pd.concat([df.reset_index(drop=True), trace_df.reset_index(drop=True)], axis=1)
    out["event_time"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["trace_fill_fraction"] = pd.to_numeric(out.get("trace_fill_fraction", 0), errors="coerce").fillna(0.0)
    out["trace_fill_ts"] = pd.to_numeric(out.get("trace_fill_ts", np.nan), errors="coerce")
    out["trace_order_submitted_ts"] = pd.to_numeric(out.get("trace_order_submitted_ts", np.nan), errors="coerce")
    out["trace_timeout_ts"] = pd.to_numeric(out.get("trace_timeout_ts", np.nan), errors="coerce")
    out["trace_fill_price"] = pd.to_numeric(out.get("trace_fill_price", np.nan), errors="coerce")
    return out


def compare_methods(df: pd.DataFrame, windows: Iterable[str]) -> list[Metrics]:
    rows: list[Metrics] = []
    for window in windows:
        wdf = _window_df(df, window)
        rows.append(_metrics("hold_to_expiry_raw_fill", window, _apply_raw_fill(wdf)))
        rows.append(_metrics("cancel_after_5m", window, _apply_cancel_5m(wdf)))
        rows.append(_metrics("polyfun_next_canary_gate", window, _apply_canary_gate(wdf)))
    return rows


def metrics_to_markdown(metrics: list[Metrics], title: str) -> str:
    lines = [f"# {title}", ""]
    lines.append("|方法|窗口|交易数|实际成交数|胜/负|胜率|盈亏|最大回撤|收益回撤比|赢家成交率|输家成交率|赢家平均成交|输家平均成交|平均成交|取消数|官方缺失|集合哈希|")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for m in metrics:
        lines.append(
            f"|{m.name}|{m.window}|{m.trades}|{m.actual_fills}|{m.wins}/{m.losses}|{m.win_rate_pct:.4f}%|{m.pnl:.4f}|{m.max_drawdown:.4f}|{m.pnl_drawdown_ratio:.6f}|{m.winner_fill_rate_pct:.4f}%|{m.loser_fill_rate_pct:.4f}%|{m.winner_avg_fill_fraction:.4f}|{m.loser_avg_fill_fraction:.4f}|{m.avg_fill_fraction:.4f}|{m.cancels}|{m.official_missing}|`{m.set_hash}`|"
        )
    lines.append("")
    lines.append("## 口径说明")
    lines.append("- `hold_to_expiry_raw_fill`：使用 episode 里的成交轨迹，不额外加取消或盘口门。")
    lines.append("- `cancel_after_5m`：订单提交后超过 300 秒才成交的部分视为取消。")
    lines.append("- `polyfun_next_canary_gate`：5U 金丝雀盘口门代理，过滤成交比例不足、价差过大、剩余时间不足的样本。")
    lines.append("- 这些仍是本地真实/代理混合数据审计，不等于真钱上线结果；官网订单状态仍是最高真相。")
    return "\n".join(lines) + "\n"


def _loads_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def _window_df(df: pd.DataFrame, window: str) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if window == "all":
        return df.copy()
    days = int(window.rstrip("d"))
    end = df["event_time"].max()
    start = end - pd.Timedelta(days=days)
    return df[df["event_time"] >= start].copy()


def _apply_raw_fill(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["sim_fill_fraction"] = x["trace_fill_fraction"].clip(0, 1)
    x["sim_canceled"] = x["sim_fill_fraction"] <= 0
    x["sim_pnl"] = x.apply(_pnl_for_row, axis=1)
    return x


def _apply_cancel_5m(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    delay = x["trace_fill_ts"] - x["trace_order_submitted_ts"]
    late = delay > CANCEL_AFTER_SECONDS
    x["sim_fill_fraction"] = np.where(late, 0.0, x["trace_fill_fraction"].clip(0, 1))
    x["sim_canceled"] = x["sim_fill_fraction"] <= 0
    x["sim_pnl"] = x.apply(_pnl_for_row, axis=1)
    return x


def _apply_canary_gate(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    spread_ok = pd.to_numeric(x.get("spread_metric", 0), errors="coerce").fillna(0) <= 0.08
    fill_ok = x["trace_fill_fraction"].fillna(0) >= 0.8
    remaining = x["trace_timeout_ts"] - x["trace_order_submitted_ts"]
    time_ok = remaining >= 420
    price = x["trace_fill_price"].fillna(1.0)
    price_ok = (price > 0.01) & (price < 0.99)
    keep = spread_ok & fill_ok & time_ok & price_ok
    x["sim_fill_fraction"] = np.where(keep, x["trace_fill_fraction"].clip(0, 1), 0.0)
    x["sim_canceled"] = ~keep
    x["sim_pnl"] = x.apply(_pnl_for_row, axis=1)
    return x


def _pnl_for_row(row: pd.Series) -> float:
    frac = float(row.get("sim_fill_fraction", 0.0) or 0.0)
    if frac <= 0:
        return 0.0
    price = row.get("trace_fill_price")
    try:
        price = float(price)
    except Exception:
        price = np.nan
    if not np.isfinite(price) or price <= 0:
        price = 0.5
    won = _won(row)
    notional = STAKE_USD * frac
    if won:
        # Buy shares at price; winning payout is 1 per share.
        return notional * (1.0 / price - 1.0)
    return -notional


def _won(row: pd.Series) -> bool:
    # Highest-priority truth for vnext episodes:
    # actual_up is the resolved market direction, direction_target is the side
    # chosen before execution. baseline_hold_pnl in these files can be an
    # action/utility field and is not safe as a win/loss label.
    if "actual_up" in row and "direction_target" in row and pd.notna(row["actual_up"]) and pd.notna(row["direction_target"]):
        try:
            return int(row["actual_up"]) == int(row["direction_target"])
        except Exception:
            pass
    if "baseline_hold_pnl" in row and pd.notna(row["baseline_hold_pnl"]):
        try:
            return float(row["baseline_hold_pnl"]) > 0
        except Exception:
            pass
    if "won_hold" in row and pd.notna(row["won_hold"]):
        return bool(row["won_hold"])
    if "exit_reason" in row:
        return str(row["exit_reason"]).lower() == "win"
    return False


def _metrics(name: str, window: str, df: pd.DataFrame) -> Metrics:
    if df.empty:
        return Metrics(name, window, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "empty")
    filled = df[df["sim_fill_fraction"] > 0].copy()
    wins_mask = filled.apply(_won, axis=1) if not filled.empty else pd.Series([], dtype=bool)
    wins = int(wins_mask.sum()) if not filled.empty else 0
    losses = int(len(filled) - wins)
    pnl_series = filled["sim_pnl"].astype(float) if not filled.empty else pd.Series([], dtype=float)
    pnl = float(pnl_series.sum()) if not filled.empty else 0.0
    dd = _max_drawdown(pnl_series)
    ratio = pnl / dd if dd > 0 else (0.0 if pnl == 0 else np.inf)
    winner_df = filled[wins_mask] if not filled.empty else filled
    loser_df = filled[~wins_mask] if not filled.empty else filled
    winner_candidates = df[df.apply(_won, axis=1)]
    loser_candidates = df[~df.apply(_won, axis=1)]
    return Metrics(
        name=name,
        window=window,
        trades=int(len(df)),
        actual_fills=int(len(filled)),
        wins=wins,
        losses=losses,
        win_rate_pct=100.0 * wins / len(filled) if len(filled) else 0.0,
        pnl=pnl,
        max_drawdown=dd,
        pnl_drawdown_ratio=float(ratio),
        winner_fill_rate_pct=100.0 * len(winner_df) / len(winner_candidates) if len(winner_candidates) else 0.0,
        loser_fill_rate_pct=100.0 * len(loser_df) / len(loser_candidates) if len(loser_candidates) else 0.0,
        winner_avg_fill_fraction=float(winner_df["sim_fill_fraction"].mean()) if len(winner_df) else 0.0,
        loser_avg_fill_fraction=float(loser_df["sim_fill_fraction"].mean()) if len(loser_df) else 0.0,
        avg_fill_fraction=float(filled["sim_fill_fraction"].mean()) if len(filled) else 0.0,
        cancels=int((df["sim_fill_fraction"] <= 0).sum()),
        official_missing=0,
        set_hash=_hash_set(filled),
    )


def _max_drawdown(pnl_series: pd.Series) -> float:
    if pnl_series.empty:
        return 0.0
    curve = pnl_series.cumsum()
    peak = curve.cummax()
    return float((peak - curve).max())


def _hash_set(df: pd.DataFrame) -> str:
    if df.empty:
        return "empty"
    keys = []
    for col in ["market_slug", "trace_market_slug", "timestamp", "side", "best_action_key"]:
        if col in df.columns:
            keys.append(df[col].astype(str))
    if not keys:
        raw = df.index.astype(str).to_series(index=df.index)
    else:
        raw = keys[0]
        for s in keys[1:]:
            raw = raw + "|" + s
    h = hashlib.sha256("\n".join(sorted(raw.tolist())).encode()).hexdigest()
    return h[:16]


def write_outputs(report_dir: str | Path, metrics: list[Metrics]) -> None:
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fair_execution_compare_latest.json").write_text(
        json.dumps([asdict(m) for m in metrics], indent=2, ensure_ascii=False)
    )
    (out / "fair_execution_compare_latest.md").write_text(
        metrics_to_markdown(metrics, "预测市场真实成交约束公平对比")
    )
