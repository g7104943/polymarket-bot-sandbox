from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PANEL = PROJECT_ROOT / "reports" / "wallet_cohort_panel_round3_top200.parquet"

DEFAULT_BUCKET_HOURS = 12
DEFAULT_FLOW_LOOKBACK_HOURS = 24
DEFAULT_WALLET_LOOKBACK_DAYS = 30.0
DEFAULT_WALLET_RECENT_DAYS = 7.0
DEFAULT_TOP_K = 10
DEFAULT_MIN_TRADES = 5

WALLET_FEATURE_COLS = [
    "wallet_up_cost_24h",
    "wallet_down_cost_24h",
    "wallet_total_cost_24h",
    "wallet_up_count_24h",
    "wallet_down_count_24h",
    "wallet_total_count_24h",
    "wallet_signed_flow_24h",
    "wallet_agreement_24h",
    "wallet_flow_bias_24h",
    "wallet_signal_available",
]


def _wr_post(wins: int, trades: int) -> float:
    return (wins + 2.0) / (trades + 4.0) if trades >= 0 else 0.5


def _compute_drawdown(pnls: Iterable[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for pnl in pnls:
        equity += float(pnl)
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return float(max_dd)


def _to_epoch_seconds(ts_like: Sequence[pd.Timestamp] | pd.Series) -> np.ndarray:
    ts = pd.to_datetime(ts_like, utc=True)
    return (ts.astype("int64") // 10**9).astype(np.int64)


@lru_cache(maxsize=8)
def _load_wallet_panel_cached(path_str: str, mtime_ns: int) -> pd.DataFrame:
    path = Path(path_str)
    df = pd.read_parquet(path)
    required = {"wallet", "symbol", "direction", "entryTs", "closeTs", "realizedPnl", "isWin", "cost"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"wallet cohort panel missing columns: {sorted(missing)}")
    out = df.copy()
    out["wallet"] = out["wallet"].astype(str).str.lower()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    out["direction"] = out["direction"].astype(str).str.upper()
    for col in ("entryTs", "closeTs", "realizedPnl", "isWin", "cost"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["entryTs", "closeTs", "realizedPnl", "isWin", "cost"])
    out["entryTs"] = out["entryTs"].astype(np.int64)
    out["closeTs"] = out["closeTs"].astype(np.int64)
    out["isWin"] = out["isWin"].astype(np.int64)
    out["cost"] = out["cost"].astype(np.float64)
    out["realizedPnl"] = out["realizedPnl"].astype(np.float64)
    return out.sort_values(["symbol", "direction", "wallet", "entryTs"]).reset_index(drop=True)


def load_wallet_panel(path: Path = DEFAULT_PANEL) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return _load_wallet_panel_cached(str(path.resolve()), path.stat().st_mtime_ns)


def zero_wallet_feature_frame(index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame({col: np.zeros(len(index), dtype=np.float64) for col in WALLET_FEATURE_COLS}, index=index)


def _prepare_flow_arrays(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    if df.empty:
        return {
            "entryTs": np.array([], dtype=np.int64),
            "cumCost": np.array([0.0], dtype=np.float64),
            "cumCount": np.array([0], dtype=np.int64),
        }
    ordered = df.sort_values("entryTs")
    entry_ts = ordered["entryTs"].to_numpy(dtype=np.int64)
    cost = ordered["cost"].to_numpy(dtype=np.float64)
    cum_cost = np.concatenate(([0.0], np.cumsum(cost, dtype=np.float64)))
    cum_count = np.arange(len(entry_ts) + 1, dtype=np.int64)
    return {"entryTs": entry_ts, "cumCost": cum_cost, "cumCount": cum_count}


def _window_cost_and_count(flow: Dict[str, np.ndarray], starts: np.ndarray, ends: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    entry_ts = flow["entryTs"]
    if entry_ts.size == 0:
        zeros_f = np.zeros(len(starts), dtype=np.float64)
        zeros_i = np.zeros(len(starts), dtype=np.int64)
        return zeros_f, zeros_i
    start_idx = np.searchsorted(entry_ts, starts, side="left")
    end_idx = np.searchsorted(entry_ts, ends, side="left")
    cost = flow["cumCost"][end_idx] - flow["cumCost"][start_idx]
    count = flow["cumCount"][end_idx] - flow["cumCount"][start_idx]
    return cost, count


def _score_wallets_for_anchor(
    panel: pd.DataFrame,
    symbol: str,
    direction: str,
    anchor_ts: int,
    lookback_days: float,
    recent_days: float,
    min_trades: int,
    top_k: int,
) -> Set[str]:
    cutoff_ts = anchor_ts - int(lookback_days * 86400)
    recent_cutoff_ts = anchor_ts - int(recent_days * 86400)
    scoped = panel[
        (panel["symbol"] == symbol)
        & (panel["direction"] == direction)
        & (panel["closeTs"] <= anchor_ts)
        & (panel["closeTs"] >= cutoff_ts)
    ]
    if scoped.empty:
        return set()

    rows: List[Dict[str, float | str]] = []
    for wallet, g in scoped.groupby("wallet", sort=True):
        trades = int(len(g))
        if trades < min_trades:
            continue
        g = g.sort_values("closeTs")
        wins = int(g["isWin"].sum())
        net_pnl = float(g["realizedPnl"].sum())
        pnl_per_trade = net_pnl / trades if trades else 0.0
        wr_post = _wr_post(wins, trades)
        recent = g[g["closeTs"] >= recent_cutoff_ts]
        recent_trades = int(len(recent))
        recent_pnl = float(recent["realizedPnl"].sum()) if recent_trades else 0.0
        recent_wr_post = _wr_post(int(recent["isWin"].sum()), recent_trades) if recent_trades else 0.5
        max_dd = _compute_drawdown(g["realizedPnl"].tolist())
        consistency = float((g["realizedPnl"] > 0).rolling(5, min_periods=1).mean().mean())
        score = (
            0.35 * wr_post
            + 0.15 * recent_wr_post
            + 0.25 * pnl_per_trade
            + 0.15 * (recent_pnl / max(1, recent_trades))
            + 0.10 * consistency
            - 0.02 * max_dd
        )
        rows.append({"wallet": str(wallet), "score": score, "recentPnl": recent_pnl})

    if not rows:
        return set()

    score_df = pd.DataFrame(rows).sort_values(["score", "recentPnl"], ascending=[False, False])
    return {str(w).lower() for w in score_df.head(top_k)["wallet"].tolist()}


def add_wallet_cohort_features(
    df: pd.DataFrame,
    asset: str,
    *,
    panel_path: Path = DEFAULT_PANEL,
    bucket_hours: int = DEFAULT_BUCKET_HOURS,
    flow_lookback_hours: int = DEFAULT_FLOW_LOOKBACK_HOURS,
    wallet_lookback_days: float = DEFAULT_WALLET_LOOKBACK_DAYS,
    wallet_recent_days: float = DEFAULT_WALLET_RECENT_DAYS,
    top_k: int = DEFAULT_TOP_K,
    wallet_min_trades: int = DEFAULT_MIN_TRADES,
) -> pd.DataFrame:
    out = df.copy()
    for col in WALLET_FEATURE_COLS:
        out[col] = 0.0

    symbol = str(asset).split("_")[0].upper()
    if symbol != "ETH":
        return out
    if not panel_path.exists():
        return out

    try:
        panel = load_wallet_panel(panel_path)
    except Exception:
        return out
    if panel.empty:
        return out

    ts_arr = _to_epoch_seconds(out["timestamp"])
    if len(ts_arr) == 0:
        return out

    bucket_sec = max(1, int(bucket_hours) * 3600)
    flow_lookback_sec = max(1, int(flow_lookback_hours) * 3600)
    anchors = (ts_arr // bucket_sec) * bucket_sec
    unique_anchors = np.unique(anchors)

    up_cost = np.zeros(len(out), dtype=np.float64)
    down_cost = np.zeros(len(out), dtype=np.float64)
    up_count = np.zeros(len(out), dtype=np.int64)
    down_count = np.zeros(len(out), dtype=np.int64)

    direction_views = {
        "UP": panel[(panel["symbol"] == symbol) & (panel["direction"] == "UP")],
        "DOWN": panel[(panel["symbol"] == symbol) & (panel["direction"] == "DOWN")],
    }

    for anchor_ts in unique_anchors:
        mask = anchors == anchor_ts
        if not np.any(mask):
            continue
        idx = np.flatnonzero(mask)
        ts_slice = ts_arr[idx]
        starts = ts_slice - flow_lookback_sec

        selected = {
            direction: _score_wallets_for_anchor(
                panel=panel,
                symbol=symbol,
                direction=direction,
                anchor_ts=int(anchor_ts),
                lookback_days=wallet_lookback_days,
                recent_days=wallet_recent_days,
                min_trades=wallet_min_trades,
                top_k=top_k,
            )
            for direction in ("UP", "DOWN")
        }

        for direction, wallets in selected.items():
            if not wallets:
                continue
            view = direction_views[direction]
            scoped = view[view["wallet"].isin(wallets)]
            flow = _prepare_flow_arrays(scoped)
            cost, count = _window_cost_and_count(flow, starts, ts_slice)
            if direction == "UP":
                up_cost[idx] = cost
                up_count[idx] = count
            else:
                down_cost[idx] = cost
                down_count[idx] = count

    total_cost = up_cost + down_cost
    total_count = up_count + down_count
    signed_flow = up_cost - down_cost
    agreement = np.zeros(len(out), dtype=np.float64)
    nz = total_cost > 0
    agreement[nz] = np.abs(signed_flow[nz]) / total_cost[nz]
    flow_bias = np.zeros(len(out), dtype=np.float64)
    flow_bias[nz] = signed_flow[nz] / total_cost[nz]
    available = (total_count > 0).astype(np.float64)

    out["wallet_up_cost_24h"] = up_cost
    out["wallet_down_cost_24h"] = down_cost
    out["wallet_total_cost_24h"] = total_cost
    out["wallet_up_count_24h"] = up_count.astype(np.float64)
    out["wallet_down_count_24h"] = down_count.astype(np.float64)
    out["wallet_total_count_24h"] = total_count.astype(np.float64)
    out["wallet_signed_flow_24h"] = signed_flow
    out["wallet_agreement_24h"] = agreement
    out["wallet_flow_bias_24h"] = flow_bias
    out["wallet_signal_available"] = available
    return out
