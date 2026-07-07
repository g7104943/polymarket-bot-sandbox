from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Any

import numpy as np
import pandas as pd

STAKE_USD = 5.0
CANCEL_AFTER_SECONDS = 300


@dataclass(frozen=True)
class RealisticMetrics:
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
    winner_fill_rate_pct: float
    loser_fill_rate_pct: float
    winner_avg_fill_fraction: float
    loser_avg_fill_fraction: float
    avg_fill_price: float
    avg_fill_delay_sec: float
    cancels: int
    set_hash: str
    note: str


def load_episode_candidates(path: str | Path, *, require_non_abstain: bool = True) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = ["timestamp", "direction_target", "actual_up", "path_t_sec_json", "path_up_json"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    out = df.copy()
    out["event_time"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["event_time"])
    if require_non_abstain and "best_action_key" in out.columns:
        out = out[out["best_action_key"].astype(str) != "ABSTAIN"].copy()
    out["won_if_filled"] = out.apply(_won_if_filled, axis=1)
    return out.reset_index(drop=True)


def load_candidates_with_episode_paths(
    episode_path: str | Path,
    candidate_path: str | Path,
) -> pd.DataFrame:
    """Join a pre-entry candidate stream to episode token paths.

    The candidate stream is the only allowed source of trade direction. Episode labels are used
    only for settlement and token path replay. This avoids the hindsight leak from
    ``direction_target`` / ``best_action_key`` in the episode files.
    """
    ep = pd.read_parquet(episode_path)
    cand = pd.read_json(candidate_path, lines=True)
    required_cand = ["timestamp", "side"]
    missing = [c for c in required_cand if c not in cand.columns]
    if missing:
        raise ValueError(f"{candidate_path} missing columns: {missing}")
    required_ep = ["timestamp", "actual_up", "path_t_sec_json", "path_up_json"]
    missing_ep = [c for c in required_ep if c not in ep.columns]
    if missing_ep:
        raise ValueError(f"{episode_path} missing columns: {missing_ep}")

    ep = ep.copy()
    cand = cand.copy()
    ep["event_time"] = pd.to_datetime(ep["timestamp"], utc=True, errors="coerce")
    cand["event_time"] = pd.to_datetime(cand["timestamp"], utc=True, errors="coerce")
    cand = cand.dropna(subset=["event_time"]).drop_duplicates("event_time", keep="last")
    keep_cols = [
        "event_time",
        "side",
        "model_score",
        "source",
        "train_window",
        "verify_window",
        "entry_price",
    ]
    keep_cols = [c for c in keep_cols if c in cand.columns]
    merged = ep.merge(cand[keep_cols], on="event_time", how="inner", suffixes=("", "_candidate"))
    if merged.empty:
        raise ValueError("candidate stream has no timestamp overlap with episode paths")
    merged["candidate_side"] = merged["side"].astype(str).str.upper()
    merged["won_if_filled"] = merged.apply(_won_if_filled, axis=1)
    return merged.reset_index(drop=True)


def compare_realistic_methods(df: pd.DataFrame, windows: Iterable[str]) -> list[RealisticMetrics]:
    rows: list[RealisticMetrics] = []
    specs = [
        ("直接吃卖价_立即成交", {"kind": "taker", "limit_offset": 0.0, "cancel_seconds": None, "price_min": 0.0, "price_max": 1.0}),
        ("挂便宜1分_5分钟取消", {"kind": "limit", "limit_offset": -0.01, "cancel_seconds": CANCEL_AFTER_SECONDS, "price_min": 0.0, "price_max": 1.0}),
        ("挂便宜2分_5分钟取消", {"kind": "limit", "limit_offset": -0.02, "cancel_seconds": CANCEL_AFTER_SECONDS, "price_min": 0.0, "price_max": 1.0}),
        ("挂便宜4分_5分钟取消", {"kind": "limit", "limit_offset": -0.04, "cancel_seconds": CANCEL_AFTER_SECONDS, "price_min": 0.0, "price_max": 1.0}),
        ("polyfun-next盘口门_吃卖价", {"kind": "taker", "limit_offset": 0.0, "cancel_seconds": None, "price_min": 0.05, "price_max": 0.55}),
        ("polyfun-next盘口门_便宜1分5分钟", {"kind": "limit", "limit_offset": -0.01, "cancel_seconds": CANCEL_AFTER_SECONDS, "price_min": 0.05, "price_max": 0.55}),
        ("不交易", {"kind": "none"}),
    ]
    for window in windows:
        wdf = _window_df(df, window)
        for method, spec in specs:
            sim = _simulate(wdf, spec)
            rows.append(_metrics(method, window, sim, _note_for(method)))
    return rows


def metrics_to_markdown(metrics: list[RealisticMetrics], title: str) -> str:
    lines = [f"# {title}", ""]
    lines.append("|方法|窗口|候选数|实际成交数|胜/负|成交后胜率|盈亏|最大回撤|收益回撤比|赢家成交率|输家成交率|赢家平均成交|输家平均成交|平均成交价|平均成交延迟秒|取消数|集合哈希|备注|")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
    for m in metrics:
        lines.append(
            f"|{m.method}|{m.window}|{m.candidates}|{m.fills}|{m.wins}/{m.losses}|{m.win_rate_pct:.4f}%|{m.pnl:.4f}|{m.max_drawdown:.4f}|{m.pnl_drawdown_ratio:.6f}|{m.winner_fill_rate_pct:.4f}%|{m.loser_fill_rate_pct:.4f}%|{m.winner_avg_fill_fraction:.4f}|{m.loser_avg_fill_fraction:.4f}|{m.avg_fill_price:.4f}|{m.avg_fill_delay_sec:.1f}|{m.cancels}|`{m.set_hash}`|{m.note}|"
        )
    lines.append("")
    lines.append("## 口径说明")
    lines.append("- 本表只用 Polymarket token 路径代理真实盘口，不用原始K线最终涨跌冒充成交。")
    lines.append("- `直接吃卖价`：假设按当时可见 token 价格立刻买入，测试信号在保证成交、价格更差时是否仍有边际。")
    lines.append("- `挂便宜N分`：限价比当时价格低 N 美分，5分钟内价格触及才成交，否则取消。")
    lines.append("- `polyfun-next盘口门`：只做价格在 0.05 到 0.55 之间的候选；本地没有完整深度时，深度不能被证明。")
    lines.append("- 如果赢家成交率明显低于输家成交率，就是成交选择偏差；这种情况下方向胜率不能当真钱依据。")
    return "\n".join(lines) + "\n"


def write_outputs(report_dir: str | Path, metrics: list[RealisticMetrics], *, stem: str = "polyfun_next_realistic_backtest_latest") -> None:
    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{stem}.json").write_text(json.dumps([asdict(m) for m in metrics], indent=2, ensure_ascii=False), encoding="utf-8")
    (out / f"{stem}.md").write_text(metrics_to_markdown(metrics, "polyfun-next 真实盘口成交约束回测"), encoding="utf-8")


def _window_df(df: pd.DataFrame, window: str) -> pd.DataFrame:
    if df.empty or window == "all":
        return df.copy()
    days = int(window.rstrip("d"))
    end = df["event_time"].max()
    start = end - pd.Timedelta(days=days)
    return df[df["event_time"] >= start].copy()


def _simulate(df: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    x = df.copy()
    if x.empty:
        x["sim_fill_fraction"] = []
        x["sim_fill_price"] = []
        x["sim_fill_delay_sec"] = []
        x["sim_pnl"] = []
        return x
    if "won_if_filled" not in x.columns or "candidate_side" in x.columns:
        x["won_if_filled"] = x.apply(_won_if_filled, axis=1)
    if spec.get("kind") == "none":
        x["sim_fill_fraction"] = 0.0
        x["sim_fill_price"] = np.nan
        x["sim_fill_delay_sec"] = np.nan
        x["sim_pnl"] = 0.0
        return x

    prices_and_times = x.apply(_side_path, axis=1)
    x["_times"] = [pt[0] for pt in prices_and_times]
    x["_prices"] = [pt[1] for pt in prices_and_times]
    x["_entry_price"] = x["_prices"].apply(lambda p: float(p[0]) if len(p) else np.nan)

    price_min = float(spec.get("price_min", 0.0))
    price_max = float(spec.get("price_max", 1.0))
    gate = x["_entry_price"].between(price_min, price_max, inclusive="both")

    fill_fraction = []
    fill_price = []
    fill_delay = []
    for _, row in x.iterrows():
        if not bool(gate.loc[row.name]):
            fill_fraction.append(0.0); fill_price.append(np.nan); fill_delay.append(np.nan); continue
        prices = row["_prices"]
        times = row["_times"]
        if len(prices) == 0 or len(times) == 0 or not np.isfinite(row["_entry_price"]):
            fill_fraction.append(0.0); fill_price.append(np.nan); fill_delay.append(np.nan); continue
        if spec.get("kind") == "taker":
            fill_fraction.append(1.0); fill_price.append(float(prices[0])); fill_delay.append(0.0); continue
        limit = max(0.01, min(0.99, float(prices[0]) + float(spec.get("limit_offset", 0.0))))
        cancel_seconds = spec.get("cancel_seconds")
        first_t = float(times[0])
        chosen = None
        for t, p in zip(times, prices):
            if cancel_seconds is not None and float(t) - first_t > float(cancel_seconds):
                break
            if float(p) <= limit:
                chosen = (float(t), float(p))
                break
        if chosen is None:
            fill_fraction.append(0.0); fill_price.append(np.nan); fill_delay.append(np.nan)
        else:
            fill_fraction.append(1.0); fill_price.append(chosen[1]); fill_delay.append(chosen[0] - first_t)
    x["sim_fill_fraction"] = fill_fraction
    x["sim_fill_price"] = fill_price
    x["sim_fill_delay_sec"] = fill_delay
    x["sim_pnl"] = x.apply(_pnl_for_row, axis=1)
    return x


def _side_path(row: pd.Series) -> tuple[list[float], list[float]]:
    times = _loads_list(row.get("path_t_sec_json"))
    up_prices = _loads_list(row.get("path_up_json"))
    n = min(len(times), len(up_prices))
    times = [float(v) for v in times[:n] if _is_num(v)]
    up_prices = [float(v) for v in up_prices[:n] if _is_num(v)]
    n = min(len(times), len(up_prices))
    times = times[:n]
    up_prices = [max(0.001, min(0.999, p)) for p in up_prices[:n]]
    direction = _candidate_direction(row)
    if direction == 1:
        prices = up_prices
    else:
        prices = [max(0.001, min(0.999, 1.0 - p)) for p in up_prices]
    return times, prices


def _pnl_for_row(row: pd.Series) -> float:
    frac = float(row.get("sim_fill_fraction", 0.0) or 0.0)
    price = row.get("sim_fill_price")
    if frac <= 0 or not _is_num(price) or float(price) <= 0:
        return 0.0
    notional = STAKE_USD * frac
    if bool(row.get("won_if_filled", False)):
        return notional * (1.0 / float(price) - 1.0)
    return -notional


def _won_if_filled(row: pd.Series) -> bool:
    return int(row.get("actual_up")) == _candidate_direction(row)


def _candidate_direction(row: pd.Series) -> int:
    side = str(row.get("candidate_side", row.get("side", ""))).upper()
    if side in {"UP", "YES", "LONG"}:
        return 1
    if side in {"DOWN", "NO", "SHORT"}:
        return 0
    return int(row.get("direction_target", 1))


def _metrics(method: str, window: str, df: pd.DataFrame, note: str) -> RealisticMetrics:
    if df.empty:
        return RealisticMetrics(method, window, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, "empty", note)
    fills = df[df["sim_fill_fraction"] > 0].copy()
    wins_mask = fills["won_if_filled"].astype(bool) if len(fills) else pd.Series([], dtype=bool)
    wins = int(wins_mask.sum()) if len(fills) else 0
    losses = int(len(fills) - wins)
    pnl_series = fills["sim_pnl"].astype(float) if len(fills) else pd.Series([], dtype=float)
    pnl = float(pnl_series.sum()) if len(fills) else 0.0
    dd = _max_drawdown(pnl_series)
    winner_candidates = df[df["won_if_filled"].astype(bool)]
    loser_candidates = df[~df["won_if_filled"].astype(bool)]
    winner_fills = fills[fills["won_if_filled"].astype(bool)] if len(fills) else fills
    loser_fills = fills[~fills["won_if_filled"].astype(bool)] if len(fills) else fills
    return RealisticMetrics(
        method=method,
        window=window,
        candidates=int(len(df)),
        fills=int(len(fills)),
        wins=wins,
        losses=losses,
        win_rate_pct=100.0 * wins / len(fills) if len(fills) else 0.0,
        pnl=pnl,
        max_drawdown=dd,
        pnl_drawdown_ratio=pnl / dd if dd > 0 else 0.0,
        winner_fill_rate_pct=100.0 * len(winner_fills) / len(winner_candidates) if len(winner_candidates) else 0.0,
        loser_fill_rate_pct=100.0 * len(loser_fills) / len(loser_candidates) if len(loser_candidates) else 0.0,
        winner_avg_fill_fraction=float(winner_fills["sim_fill_fraction"].mean()) if len(winner_fills) else 0.0,
        loser_avg_fill_fraction=float(loser_fills["sim_fill_fraction"].mean()) if len(loser_fills) else 0.0,
        avg_fill_price=float(fills["sim_fill_price"].mean()) if len(fills) else 0.0,
        avg_fill_delay_sec=float(fills["sim_fill_delay_sec"].mean()) if len(fills) else 0.0,
        cancels=int((df["sim_fill_fraction"] <= 0).sum()),
        set_hash=_hash_set(fills),
        note=note,
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
    cols = [c for c in ["market_slug", "timestamp", "candidate_side", "source", "best_action_key", "direction_target"] if c in df.columns]
    if not cols:
        raw = [str(i) for i in df.index]
    else:
        raw = df[cols].astype(str).agg("|".join, axis=1).tolist()
    return hashlib.sha256("\n".join(sorted(raw)).encode()).hexdigest()[:16]


def _loads_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, str):
        try:
            out = json.loads(value)
            return out if isinstance(out, list) else []
        except Exception:
            return []
    return []


def _is_num(x: Any) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def _note_for(method: str) -> str:
    if "直接吃" in method:
        return "保证成交；用来检验方向信号扣除买价后是否还有边际。"
    if "便宜" in method:
        return "模拟限价等待；若输家成交率显著更高，说明成交选择偏差有毒。"
    if "盘口门" in method:
        return "polyfun-next 盘口门代理；当前缺完整盘口深度，只能证明价格门。"
    if method == "不交易":
        return "风险为零的基准。"
    return ""
