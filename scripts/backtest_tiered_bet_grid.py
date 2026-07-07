#!/usr/bin/env python3
"""
8 个 GRU 组合的档位下注比例网格回测（阈值+B+p，可选校准）：
- 仅跑 models_best / models_best_no1h4h 下 8 个模型（ETH/BTC/XRP/SOL 各 2），与模拟盘无关。
- 数据：data/raw/*.parquet + GRU 推理得到交易序列。推荐配置下用 --val-no-leak --multi-segment 2 时，
  选参与打分只用「无泄漏起始日→parquet 末尾」这段数据，并在其内切 2 段做多段验证，即全程无泄漏。
- 超参含义：对「已超过模型阈值」的预测，再按「超过多少」(Δ+B0,B1,B2,B3) 分档，每档下注比例 p1～p4（%）。
  与回测规则一致：总资金 <6 万按比例下注，≥6 万单笔封顶 3000（实现为 下注=min(资金×比例, 3000)）。
- 粗网格点数：Δ 6 点 × B0 6 × B123 120 × p 5^4=625 ≈ 270 万点/组合，8 组合约 2160 万次评估；多段 2 段时每点跑 2 段回测。
  耗时与机器有关，约数十分钟～数小时，加 --workers 5 可明显缩短。
- 可选 --calibration-grid 增加校准维度；输出 ranking_val_365d 等，列含方案、Δ、p1～p4、校准。

唯一推荐配置（无泄漏 + 粗网格 + 2 段最差段 + top5 + 稳健配置，可加 --workers 5 加速）：
  python scripts/backtest_tiered_bet_grid.py --scientific --val-no-leak --coarse --multi-segment 2 --segment-aggregate min --threshold-top 5 --robust-config --workers 5

其他用法：
  python scripts/backtest_tiered_bet_grid.py --scientific --val-no-leak
  python scripts/backtest_tiered_bet_grid.py --scientific --calibration-grid
  python scripts/backtest_tiered_bet_grid.py --scientific --resume
  python scripts/backtest_tiered_bet_grid.py --sim-12-only [--val-no-leak]
"""

import argparse
import pickle
import signal
import statistics
import time
import importlib.util
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set
from itertools import combinations, combinations_with_replacement, product

import numpy as np
import pandas as pd

# 校准选项：无 / isotonic / sigmoid，在验证段上对比胜率与最终资金
CALIBRATION_OPTIONS = ["无", "isotonic", "sigmoid"]

# 断点续跑：检查点路径与状态（在 main 里设置）
_CHECKPOINT_PATH: Optional[Path] = None
_RESUME_INTERRUPTED = False

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"

# 仅 8 个 GRU 组合（models_best / models_best_no1h4h），无其他组合、无基线
COMBO_META: Dict[str, Dict[str, Any]] = {}

INITIAL_CAPITAL = 400.0
MIN_BET = 1.0
# 与「400 初始、≥6万固定3000、<6万按比例」一致：流动性封顶，单笔最大 3000
MAX_BET_CAP = 3000.0

# 与 backtest_gru_regime 一致：手续费 + 滑点（--scientific 时用公式重算 PnL）
DEFAULT_FEE_RATE = 0.001
DEFAULT_SLIPPAGE = 0.001
# scientific 时固定下单单价（模拟交易平均价），可用 --order-price 覆盖
FIXED_ORDER_PRICE_DEFAULT = 0.527

# 8 个 GRU 仅用两套模型目录（与模拟盘无关），4 资产×2 = 8
# models_best = 用了 1h/4h 当特征的 4 个模型；models_best_no1h4h = 没用 1h4h 的 4 个模型
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "gru_regime_v1"
MODELS_BEST = EXPERIMENT_ROOT / "outputs" / "models_best"  # 有 1h4h 特征
MODELS_BEST_NO1H4H = EXPERIMENT_ROOT / "outputs" / "models_best_no1h4h"  # 无 1h4h 特征
# 8 个 GRU 模型配置：用了 1h4h 当特征的用 use_1h4h=True(models_best)，没用 1h4h 的用 use_no1h4h=True(models_best_no1h4h)
GRU_MODEL_CONFIGS: List[Dict[str, Any]] = [
    {"key": "GRU_ETH_USDT", "asset": "ETH_USDT", "use_1h4h": True, "symbol": "ETH"},       # models_best 有1h4h
    {"key": "GRU_ETH_USDT_no1h4h", "asset": "ETH_USDT", "use_no1h4h": True, "symbol": "ETH"},  # models_best_no1h4h 无1h4h
    {"key": "GRU_BTC_USDT", "asset": "BTC_USDT", "use_1h4h": True, "symbol": "BTC"},
    {"key": "GRU_BTC_USDT_no1h4h", "asset": "BTC_USDT", "use_no1h4h": True, "symbol": "BTC"},
    {"key": "GRU_XRP_USDT", "asset": "XRP_USDT", "use_1h4h": True, "symbol": "XRP"},
    {"key": "GRU_XRP_USDT_no1h4h", "asset": "XRP_USDT", "use_no1h4h": True, "symbol": "XRP"},
    {"key": "GRU_SOL_USDT", "asset": "SOL_USDT", "use_1h4h": True, "symbol": "SOL"},
    {"key": "GRU_SOL_USDT_no1h4h", "asset": "SOL_USDT", "use_no1h4h": True, "symbol": "SOL"},
]
GRU_COMBOS = [c["key"] for c in GRU_MODEL_CONFIGS]
NO1H4H_COMBOS = [c["key"] for c in GRU_MODEL_CONFIGS if c.get("use_no1h4h")]
for _c in GRU_MODEL_CONFIGS:
    k, sym = _c["key"], _c["symbol"]
    COMBO_META[k] = {"symbol": sym, "label": "GRU_{}_1h4h".format(sym) if _c.get("use_1h4h") else "GRU_{}_no1h4h".format(sym), "dynamic": "无", "calibration": "有"}
# 阈值搜索范围 [0.5, 0.6]（含），步长默认 0.01 → 11 个点（0.50,0.51,…,0.60）
THRESHOLD_SEARCH_MIN = 0.50
THRESHOLD_SEARCH_MAX = 0.60
# 综合评分权重：最终资金、胜率、最大回撤（越小越好）、最低/初始（越高越好）
SCORE_WEIGHTS = (0.35, 0.25, 0.25, 0.15)  # (资金, 胜率, -回撤, 最低/初始)
# GRU 缩减网格：p1～p4 取值 [2,3,4,5,6,7,8,9,10]，共 C(9+4-1,4)=495 组；阈值 11 点 × B0 6 点 × B123 120 × 495 ≈ 392 万/组合
# --fast 时：阈值步长 0.02(6 点)、p 取 [2,4,6,8,10](70 组) → 6×6×120×70≈30 万/组合，约 13 倍加速，精度略降
# --coarse 粗网格：阈值步长 0.02(6 点)、p 取 [2,4,6,8,10] 的 5^4=625 组（product），搜索点约减半、更稳
P_REDUCED_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10]
P_FAST_VALUES = [2, 4, 6, 8, 10]  # --fast 时 p 只取 5 个值，70 组（combinations_with_replacement）
P_COARSE_VALUES = [2, 4, 6, 8, 10]  # --coarse 时 p 取 5^4=625 组（product）


def _trades_to_date_str(t: Dict[str, Any]) -> str:
    """从单笔 trade 取日期 YYYY-MM-DD。"""
    d = t.get("date") or t.get("timestamp")
    if d is None:
        return ""
    if isinstance(d, (int, float)):
        return pd.to_datetime(d, unit="ms", utc=True).strftime("%Y-%m-%d")
    return str(d)[:10]


def _split_trades_by_segments(trades: List[Dict[str, Any]], n_segments: int) -> List[List[Dict[str, Any]]]:
    """将 trades 按日期排序后切为 n_segments 段（按时间顺序均分），每段返回一个列表。"""
    if not trades or n_segments <= 1:
        return [trades] if trades else []
    sorted_trades = sorted(trades, key=lambda t: _trades_to_date_str(t))
    n = len(sorted_trades)
    seg_size = n // n_segments
    remainder = n % n_segments
    segments = []
    start = 0
    for i in range(n_segments):
        size = seg_size + (1 if i < remainder else 0)
        segments.append(sorted_trades[start : start + size])
        start += size
    return segments


def _parquet_date_range(data_src: Path, asset: str) -> Tuple[str, str]:
    """从 parquet 读时间范围，返回 (min_date, max_date) YYYY-MM-DD。"""
    filepath = data_src / f"{asset.lower()}_15m.parquet"
    if not filepath.exists():
        raise FileNotFoundError(f"parquet 不存在: {filepath}")
    df = pd.read_parquet(filepath)
    if "timestamp" not in df.columns:
        raise ValueError("parquet 需含 timestamp 列")
    ts = df["timestamp"]
    mn, mx = int(ts.min()), int(ts.max())
    d_min = pd.to_datetime(mn, unit="ms", utc=True).strftime("%Y-%m-%d")
    d_max = pd.to_datetime(mx, unit="ms", utc=True).strftime("%Y-%m-%d")
    return (d_min, d_max)


def _df_to_trades(merged_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """将 GRU 回测 DataFrame（含 pred_prob, pred_up, actual_up）转为档位回测用的 trades 列表。
    confidence 用 effective_prob（押 up 用 pred_prob，押 down 用 1-pred_prob）；result 仅按预测与实际是否一致（已去掉止损 reversal）。
    """
    if "pred_prob" not in merged_df.columns or "actual_up" not in merged_df.columns:
        return []
    trades = []
    for _, row in merged_df.iterrows():
        ts = row.get("timestamp")
        if ts is None:
            ts = row.get("date")
        if pd.isna(ts):
            continue
        if hasattr(ts, "value"):
            ts = int(ts.value // 10**6)
        elif not isinstance(ts, (int, float)):
            ts = int(pd.to_datetime(ts).value // 10**6)
        dt = row.get("date")
        if dt is None:
            dt = pd.to_datetime(ts, unit="ms", utc=True)
        date_str = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)[:10]
        try:
            pred_prob = float(row["pred_prob"])
        except (TypeError, ValueError, KeyError):
            continue
        pred_up = int(row.get("pred_up", 1 if pred_prob >= 0.5 else 0))
        actual_up = int(row.get("actual_up", 0))
        effective = pred_prob if pred_up == 1 else (1.0 - pred_prob)
        result = "win" if (actual_up == 1 and pred_up == 1) or (actual_up == 0 and pred_up == 0) else "loss"
        trades.append({
            "timestamp": ts,
            "date": date_str,
            "confidence": effective,
            "result": result,
            "amount": 1.0,
        })
    return trades


def _fit_calibrator(train_trades: List[Dict[str, Any]], method: str):
    """在训练段上拟合校准器：X=confidence, y=result(win=1)。method 为 isotonic 或 sigmoid。"""
    if not train_trades or method == "无":
        return None
    X = np.array([t["confidence"] for t in train_trades], dtype=float).reshape(-1, 1)
    y = np.array([1 if t.get("result") == "win" else 0 for t in train_trades], dtype=int)
    if np.unique(y).size < 2:
        return None
    if method == "isotonic":
        from sklearn.isotonic import IsotonicRegression
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(X.ravel(), y)
        return cal
    if method == "sigmoid":
        from sklearn.linear_model import LogisticRegression
        cal = LogisticRegression()
        cal.fit(X, y)
        return cal
    return None


def _apply_calibration_to_trades(trades: List[Dict[str, Any]], calibrator, method: str) -> List[Dict[str, Any]]:
    """对 trades 的 confidence 做校准，返回新列表（不修改原列表）。"""
    if not trades or calibrator is None or method == "无":
        return list(trades)
    probs = np.array([t["confidence"] for t in trades], dtype=float)
    if method == "isotonic":
        calibrated = np.clip(calibrator.predict(probs), 0.0, 1.0)
    elif method == "sigmoid":
        calibrated = np.clip(calibrator.predict_proba(probs.reshape(-1, 1))[:, 1], 0.0, 1.0)
    else:
        return list(trades)
    out = []
    for i, t in enumerate(trades):
        t2 = dict(t)
        t2["confidence"] = float(calibrated[i])
        out.append(t2)
    return out


def _run_equity_curve(
    trades: List[Dict[str, Any]],
    get_bet_ratio_fn: Any,
    use_scientific: bool,
    cap_bet_3000: bool,
    fixed_order_price: Optional[float],
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage: float = DEFAULT_SLIPPAGE,
) -> Tuple[float, List[float], int, int]:
    """
    公共资金曲线：按时间序遍历 trades，每笔由 get_bet_ratio_fn(trade, capital, peak_capital) 得到 (ratio, include)。
    若 include 为 False 则跳过；否则下注 bet=min(capital*ratio, MAX_BET_CAP)，算 new_pnl，更新 capital/peak。
    返回 (final_capital, equity_curve, wins, losses)。
    """
    sorted_trades = sorted(
        enumerate(trades),
        key=lambda x: (str(x[1].get("timestamp") or ""), x[0]),
    )
    sorted_trades = [t for _, t in sorted_trades]
    capital = INITIAL_CAPITAL
    peak_capital = INITIAL_CAPITAL
    wins = 0
    losses = 0
    equity_curve: List[float] = [capital]
    for t in sorted_trades:
        if capital < MIN_BET:
            break
        ratio, include = get_bet_ratio_fn(t, capital, peak_capital)
        if not include or ratio <= 0:
            continue
        raw_bet = capital * ratio
        bet = min(raw_bet, MAX_BET_CAP) if cap_bet_3000 else raw_bet
        bet = min(bet, capital)
        if bet < MIN_BET:
            continue
        amount = t.get("amount")
        amount_used = t.get("actualAmount") if t.get("actualAmount") is not None and float(t.get("actualAmount", 0)) > 0 else amount
        if amount_used is None or float(amount_used) <= 0:
            continue
        if use_scientific:
            price = fixed_order_price if fixed_order_price is not None and fixed_order_price > 0 else t.get("tokenPrice")
            if price is None or price <= 0:
                continue
            fee = bet * (fee_rate + slippage)
            if t.get("result") == "win":
                new_pnl = bet * (1.0 / price - 1.0) - fee
                wins += 1
            else:
                new_pnl = -bet - fee
                losses += 1
        else:
            pnl = t.get("pnl")
            if pnl is None:
                continue
            scale = bet / float(amount_used)
            new_pnl = scale * pnl
            if new_pnl > 0:
                wins += 1
            else:
                losses += 1
        capital += new_pnl
        if capital > peak_capital:
            peak_capital = capital
        equity_curve.append(capital)
    return (capital, equity_curve, wins, losses)


def get_tier(confidence: float, threshold: float, b0: int, b1: int, b2: int, b3: int) -> int:
    """
    根据置信度与档位边界返回档位 1～4。
    档 1: [threshold+B0, threshold+B0+B1]
    档 2: (threshold+B0+B1, threshold+B0+B2]
    档 3: (threshold+B0+B2, threshold+B0+B3]
    档 4: > threshold+B0+B3
    单位：confidence 为小数（0.55），b0,b1,b2,b3 为整数百分点（0～10），换算为 0.01。
    """
    base = threshold + b0 * 0.01
    c1 = base + b1 * 0.01  # 档1 上界
    c2 = base + b2 * 0.01  # 档2 上界
    c3 = base + b3 * 0.01  # 档3 上界
    if confidence <= c1:
        return 1
    if confidence <= c2:
        return 2
    if confidence <= c3:
        return 3
    return 4


def simulate_combo_stats(
    trades: List[Dict[str, Any]],
    threshold: float,
    b0: int,
    b1: int,
    b2: int,
    b3: int,
    p1: int,
    p2: int,
    p3: int,
    p4: int,
    use_scientific: bool = False,
    cap_bet_3000: bool = True,
    fixed_order_price: Optional[float] = None,
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage: float = DEFAULT_SLIPPAGE,
) -> Dict[str, Any]:
    """按档位规则跑资金曲线，返回交易次数、胜败、最终资金、回撤等，供排名表与网格搜索用。"""
    base = threshold + b0 * 0.01
    pcts = {1: p1 * 0.01, 2: p2 * 0.01, 3: p3 * 0.01, 4: p4 * 0.01}

    def get_ratio(t: Dict, cap: float, peak: float) -> Tuple[float, bool]:
        if t["confidence"] < base:
            return (0.0, False)
        tier = get_tier(t["confidence"], threshold, b0, b1, b2, b3)
        return (pcts[tier], True)

    capital, equity_curve, wins, losses = _run_equity_curve(
        trades, get_ratio, use_scientific, cap_bet_3000, fixed_order_price, fee_rate, slippage
    )
    n_trades = wins + losses
    peak = INITIAL_CAPITAL
    max_dd = 0.0
    min_cap = INITIAL_CAPITAL
    for c in equity_curve:
        if c > peak:
            peak = c
        if peak > 0:
            dd = (peak - c) / peak
            if dd > max_dd:
                max_dd = dd
        if c < min_cap:
            min_cap = c
    min_over_initial = (min_cap / INITIAL_CAPITAL * 100.0) if INITIAL_CAPITAL > 0 else 100.0
    profit_pct = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100.0 if INITIAL_CAPITAL > 0 else 0.0
    win_rate_pct = (wins / n_trades * 100.0) if n_trades > 0 else 0.0
    return {
        "n_trades": n_trades,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": win_rate_pct,
        "final_capital": capital,
        "pnl": capital - INITIAL_CAPITAL,
        "profit_pct": profit_pct,
        "max_drawdown_pct": max_dd * 100.0,
        "min_over_initial_pct": min_over_initial,
    }


def _ranking_label(
    log_dir: str,
    thr: float,
    params: Tuple[int, int, int, int, int, int, int, int],
    meta: Dict[str, Any],
) -> str:
    """排名表最后一列：GRU_币种_阈值_no1h4h_B0_p 或 GRU_币种_阈值_B0_p。"""
    b0, b1, b2, b3, p1, p2, p3, p4 = params
    sym = meta.get("symbol", "")
    p_str = f"{p1},{p2},{p3},{p4}"
    if log_dir in NO1H4H_COMBOS:
        return f"GRU_{sym}_{thr:.2f}_no1h4h_{b0}_{p_str}"
    return f"GRU_{sym}_{thr:.2f}_{b0}_{p_str}"


def _composite_score(stats: Dict[str, Any], weights: Tuple[float, float, float, float] = SCORE_WEIGHTS) -> float:
    """综合评分：资金、胜率、回撤（越小越好）、最低/初始（越高越好）。越高越好。"""
    w1, w2, w3, w4 = weights
    cap = stats.get("final_capital", 0.0) / INITIAL_CAPITAL if INITIAL_CAPITAL > 0 else 0.0
    win = stats.get("win_rate_pct", 0.0) / 100.0
    dd = min(1.0, max(0.0, stats.get("max_drawdown_pct", 100.0) / 100.0))
    min_init = stats.get("min_over_initial_pct", 0.0) / 100.0
    return w1 * cap + w2 * win - w3 * dd + w4 * min_init


def grid_search_gru_with_threshold(
    trades: List[Dict[str, Any]],
    threshold_min: float = THRESHOLD_SEARCH_MIN,
    threshold_max: float = THRESHOLD_SEARCH_MAX,
    threshold_step: float = 0.01,
    search_b0: bool = True,
    use_scientific: bool = False,
    cap_bet_3000: bool = True,
    fixed_order_price: Optional[float] = None,
    reduced_grid: bool = True,
    fast_grid: bool = False,
    coarse_grid: bool = False,
    workers: int = 1,
) -> List[Tuple[float, Tuple[int, int, int, int, int, int, int, int], float, Dict[str, Any]]]:
    """
    对 GRU 组合做超参三块网格搜索（① 阈值 ② B ③ p），按综合评分排序。
    fast_grid=True：阈值步长 0.02(6 点)、p 取 [2,4,6,8,10](70 组)，约 13 倍加速。
    coarse_grid=True：阈值步长 0.02、p 取 [2,4,6,8,10] 的 5^4=625 组，更粗更稳。
    workers>1：多进程并行，进一步加速。
    """
    step = 0.02 if (fast_grid or coarse_grid) else threshold_step
    thresholds = []
    t = threshold_min
    while t <= threshold_max + 1e-9:
        thresholds.append(round(t, 2))
        t += step
    if reduced_grid:
        b0_range = list(range(0, 11, 2))
        if coarse_grid:
            p_values = P_COARSE_VALUES
            p1234_list = [tuple(x) for x in product(p_values, repeat=4)]  # 5^4=625
        else:
            p_values = P_FAST_VALUES if fast_grid else P_REDUCED_VALUES
            p1234_list = list(combinations_with_replacement(p_values, 4))
    else:
        b0_range = list(range(0, 11)) if search_b0 else [0]
        p1234_list = list(combinations_with_replacement(range(1, 11), 4))
    b123_list = [(b1, b2, b3) for b1, b2, b3 in combinations(range(1, 11), 3)]
    total_points = len(thresholds) * len(b0_range) * len(b123_list) * len(p1234_list)

    def eval_one(params: Tuple) -> Tuple:
        th, b0, b1, b2, b3, p1, p2, p3, p4 = params
        st = simulate_combo_stats(
            trades, th, b0, b1, b2, b3, p1, p2, p3, p4,
            use_scientific=use_scientific,
            cap_bet_3000=cap_bet_3000,
            fixed_order_price=fixed_order_price,
        )
        return (th, (b0, b1, b2, b3, p1, p2, p3, p4), _composite_score(st), st)

    all_params = [
        (th, b0, b1, b2, b3, p1, p2, p3, p4)
        for th in thresholds
        for b0 in b0_range
        for (b1, b2, b3) in b123_list
        for (p1, p2, p3, p4) in p1234_list
    ]

    def _format_eta(seconds: float) -> str:
        if not (seconds > 0 and seconds < 1e9):
            return "—"
        if seconds < 60:
            return "{:.0f}秒".format(seconds)
        if seconds < 3600:
            return "{:.0f}分{:.0f}秒".format(seconds // 60, seconds % 60)
        return "{:.0f}时{:.0f}分".format(seconds // 3600, (seconds % 3600) // 60)

    PROGRESS_EVERY = 5000
    results: List[Tuple[float, Tuple[int, int, int, int, int, int, int, int], float, Dict[str, Any]]] = []
    t0_grid = time.perf_counter()
    if workers <= 1:
        done = 0
        for params in all_params:
            results.append(eval_one(params))
            done += 1
            if done % PROGRESS_EVERY == 0 or done == total_points:
                elapsed = time.perf_counter() - t0_grid
                rate = done / elapsed if elapsed > 0 else 0
                eta_s = (total_points - done) / rate if rate > 0 else 0
                print("    已计算 {}/{} 组  已用 {:.1f}s  预计剩余 {}".format(done, total_points, elapsed, _format_eta(eta_s)), flush=True)
    else:
        from multiprocessing import Pool
        with Pool(workers, initializer=_init_worker, initargs=(trades, use_scientific, cap_bet_3000, fixed_order_price)) as pool:
            for r in pool.imap_unordered(_eval_one_worker, all_params, chunksize=500):
                results.append(r)
                done = len(results)
                if done % PROGRESS_EVERY == 0 or done == total_points:
                    elapsed = time.perf_counter() - t0_grid
                    rate = done / elapsed if elapsed > 0 else 0
                    eta_s = (total_points - done) / rate if rate > 0 else 0
                    print("    已计算 {}/{} 组  已用 {:.1f}s  预计剩余 {}".format(done, total_points, elapsed, _format_eta(eta_s)), flush=True)
    results.sort(key=lambda x: x[2], reverse=True)
    return results


def grid_search_gru_with_calibration(
    train_trades: List[Dict[str, Any]],
    val_trades: List[Dict[str, Any]],
    threshold_min: float = THRESHOLD_SEARCH_MIN,
    threshold_max: float = THRESHOLD_SEARCH_MAX,
    threshold_step: float = 0.02,
    search_b0: bool = True,
    use_scientific: bool = False,
    cap_bet_3000: bool = True,
    fixed_order_price: Optional[float] = None,
    coarse_grid: bool = True,
) -> List[Tuple[float, Tuple[int, int, int, int, int, int, int, int], float, Dict[str, Any], str]]:
    """
    带校准维度的网格搜索：对 无/isotonic/sigmoid 在训练段拟合（isotonic/sigmoid），在验证段应用后回测。
    返回 (th, params, score, st, calibration_label) 列表，按 score 降序。
    """
    val_by_cal: Dict[str, List[Dict[str, Any]]] = {"无": list(val_trades)}
    for method in ("isotonic", "sigmoid"):
        cal = _fit_calibrator(train_trades, method)
        val_by_cal[method] = _apply_calibration_to_trades(val_trades, cal, method)

    step = threshold_step
    thresholds = []
    t = threshold_min
    while t <= threshold_max + 1e-9:
        thresholds.append(round(t, 2))
        t += step
    b0_range = list(range(0, 11, 2))
    p1234_list = [tuple(x) for x in product(P_COARSE_VALUES, repeat=4)]
    b123_list = [(b1, b2, b3) for b1, b2, b3 in combinations(range(1, 11), 3)]

    def _fmt_eta(s: float) -> str:
        if not (s > 0 and s < 1e9):
            return "—"
        if s < 60:
            return "{:.0f}秒".format(s)
        if s < 3600:
            return "{:.0f}分{:.0f}秒".format(s // 60, s % 60)
        return "{:.0f}时{:.0f}分".format(s // 3600, (s % 3600) // 60)

    results: List[Tuple[float, Tuple[int, int, int, int, int, int, int, int], float, Dict[str, Any], str]] = []
    total_points = len(thresholds) * len(b0_range) * len(b123_list) * len(p1234_list) * len(CALIBRATION_OPTIONS)
    done = 0
    t0_grid = time.perf_counter()
    for th in thresholds:
        for b0 in b0_range:
            for (b1, b2, b3) in b123_list:
                for (p1, p2, p3, p4) in p1234_list:
                    for cal_label in CALIBRATION_OPTIONS:
                        trades_cal = val_by_cal.get(cal_label, val_trades)
                        st = simulate_combo_stats(
                            trades_cal, th, b0, b1, b2, b3, p1, p2, p3, p4,
                            use_scientific=use_scientific,
                            cap_bet_3000=cap_bet_3000,
                            fixed_order_price=fixed_order_price,
                        )
                        score = _composite_score(st)
                        params = (b0, b1, b2, b3, p1, p2, p3, p4)
                        results.append((th, params, score, st, cal_label))
                        done += 1
                        if done % 5000 == 0 or done == total_points:
                            elapsed = time.perf_counter() - t0_grid
                            rate = done / elapsed if elapsed > 0 else 0
                            eta_s = (total_points - done) / rate if rate > 0 else 0
                            print("    已计算 {}/{} 组  已用 {:.1f}s  预计剩余 {}".format(done, total_points, elapsed, _fmt_eta(eta_s)), flush=True)
    results.sort(key=lambda x: x[2], reverse=True)
    return results


def _init_worker_multisegment(
    trades_segments: List[List[Dict[str, Any]]],
    segment_aggregate: str,
    use_scientific: bool,
    cap_bet_3000: bool,
    fixed_order_price: Optional[float],
) -> None:
    """多段验证多进程 worker 初始化。"""
    global _WORKER_SEGMENTS, _WORKER_SEGMENT_AGG, _WORKER_OPTS
    _WORKER_SEGMENTS = trades_segments
    _WORKER_SEGMENT_AGG = segment_aggregate
    _WORKER_OPTS = dict(use_scientific=use_scientific, cap_bet_3000=cap_bet_3000, fixed_order_price=fixed_order_price)


_WORKER_SEGMENTS: List[List[Dict[str, Any]]] = []
_WORKER_SEGMENT_AGG: str = "mean"


def _eval_one_multisegment_worker(params: Tuple) -> Tuple:
    """多段验证单点评估（读全局 _WORKER_SEGMENTS）。"""
    th, b0, b1, b2, b3, p1, p2, p3, p4 = params
    caps = []
    all_st = []
    for seg_trades in _WORKER_SEGMENTS:
        if not seg_trades:
            caps.append(INITIAL_CAPITAL)
            all_st.append({})
            continue
        st = simulate_combo_stats(
            seg_trades, th, b0, b1, b2, b3, p1, p2, p3, p4, **_WORKER_OPTS
        )
        caps.append(st["final_capital"])
        all_st.append(st)
    if _WORKER_SEGMENT_AGG == "min":
        score = min(caps) if caps else 0.0
    else:
        score = sum(caps) / len(caps) if caps else 0.0
    merged = all_st[0].copy() if all_st else {}
    merged["segment_final_capitals"] = caps
    merged["aggregate_score"] = score
    return (th, (b0, b1, b2, b3, p1, p2, p3, p4), score, merged)


def grid_search_gru_multisegment(
    trades_segments: List[List[Dict[str, Any]]],
    segment_aggregate: str = "mean",
    threshold_min: float = THRESHOLD_SEARCH_MIN,
    threshold_max: float = THRESHOLD_SEARCH_MAX,
    threshold_step: float = 0.02,
    search_b0: bool = True,
    use_scientific: bool = False,
    cap_bet_3000: bool = True,
    fixed_order_price: Optional[float] = None,
    coarse_grid: bool = True,
    workers: int = 1,
) -> List[Tuple[float, Tuple[int, int, int, int, int, int, int, int], float, Dict[str, Any]]]:
    """
    多段验证：对每段 trades 分别回测，得分 = mean(各段最终资金) 或 min(各段最终资金)。
    数据仅用无泄漏区间内切成的多段，选参更稳。coarse_grid 默认 True（步长 0.02、p 取 5^4）。
    workers>1 时多进程并行。
    """
    step = threshold_step
    thresholds = []
    t = threshold_min
    while t <= threshold_max + 1e-9:
        thresholds.append(round(t, 2))
        t += step
    b0_range = list(range(0, 11, 2))
    p1234_list = [tuple(x) for x in product(P_COARSE_VALUES, repeat=4)]
    b123_list = [(b1, b2, b3) for b1, b2, b3 in combinations(range(1, 11), 3)]
    total_points = len(thresholds) * len(b0_range) * len(b123_list) * len(p1234_list)

    all_params = [
        (th, b0, b1, b2, b3, p1, p2, p3, p4)
        for th in thresholds
        for b0 in b0_range
        for (b1, b2, b3) in b123_list
        for (p1, p2, p3, p4) in p1234_list
    ]

    def _format_eta(seconds: float) -> str:
        if not (seconds > 0 and seconds < 1e9):
            return "—"
        if seconds < 60:
            return "{:.0f}秒".format(seconds)
        if seconds < 3600:
            return "{:.0f}分{:.0f}秒".format(seconds // 60, seconds % 60)
        return "{:.0f}时{:.0f}分".format(seconds // 3600, (seconds % 3600) // 60)

    PROGRESS_EVERY = 5000
    results: List[Tuple[float, Tuple[int, int, int, int, int, int, int, int], float, Dict[str, Any]]] = []
    t0_grid = time.perf_counter()
    if workers <= 1:
        done = 0
        for params in all_params:
            th, b0, b1, b2, b3, p1, p2, p3, p4 = params
            caps = []
            all_st = []
            for seg_trades in trades_segments:
                if not seg_trades:
                    caps.append(INITIAL_CAPITAL)
                    all_st.append({})
                    continue
                st = simulate_combo_stats(
                    seg_trades, th, b0, b1, b2, b3, p1, p2, p3, p4,
                    use_scientific=use_scientific,
                    cap_bet_3000=cap_bet_3000,
                    fixed_order_price=fixed_order_price,
                )
                caps.append(st["final_capital"])
                all_st.append(st)
            score = min(caps) if caps and segment_aggregate == "min" else (sum(caps) / len(caps) if caps else 0.0)
            merged = all_st[0].copy() if all_st else {}
            merged["segment_final_capitals"] = caps
            merged["aggregate_score"] = score
            results.append((th, (b0, b1, b2, b3, p1, p2, p3, p4), score, merged))
            done += 1
            if done % PROGRESS_EVERY == 0 or done == total_points:
                elapsed = time.perf_counter() - t0_grid
                rate = done / elapsed if elapsed > 0 else 0
                eta_s = (total_points - done) / rate if rate > 0 else 0
                print("    已计算 {}/{} 组  已用 {:.1f}s  预计剩余 {}".format(done, total_points, elapsed, _format_eta(eta_s)), flush=True)
    else:
        from multiprocessing import Pool
        with Pool(
            workers,
            initializer=_init_worker_multisegment,
            initargs=(trades_segments, segment_aggregate, use_scientific, cap_bet_3000, fixed_order_price),
        ) as pool:
            for r in pool.imap_unordered(_eval_one_multisegment_worker, all_params, chunksize=500):
                results.append(r)
                done = len(results)
                if done % PROGRESS_EVERY == 0 or done == total_points:
                    elapsed = time.perf_counter() - t0_grid
                    rate = done / elapsed if elapsed > 0 else 0
                    eta_s = (total_points - done) / rate if rate > 0 else 0
                    print("    已计算 {}/{} 组  已用 {:.1f}s  预计剩余 {}".format(done, total_points, elapsed, _format_eta(eta_s)), flush=True)
    results.sort(key=lambda x: x[2], reverse=True)
    return results


def _init_worker(trades: List[Dict], use_scientific: bool, cap_bet_3000: bool, fixed_order_price: Optional[float]) -> None:
    """多进程 worker 初始化：把 trades 和选项放进全局，供 _eval_one_worker 使用。"""
    global _WORKER_TRADES, _WORKER_OPTS
    _WORKER_TRADES = trades
    _WORKER_OPTS = dict(use_scientific=use_scientific, cap_bet_3000=cap_bet_3000, fixed_order_price=fixed_order_price)


_WORKER_TRADES: List[Dict[str, Any]] = []
_WORKER_OPTS: Dict[str, Any] = {}


def _eval_one_worker(params: Tuple) -> Tuple:
    """多进程单点评估（读全局 _WORKER_TRADES）。"""
    th, b0, b1, b2, b3, p1, p2, p3, p4 = params
    st = simulate_combo_stats(
        _WORKER_TRADES, th, b0, b1, b2, b3, p1, p2, p3, p4, **_WORKER_OPTS
    )
    return (th, (b0, b1, b2, b3, p1, p2, p3, p4), _composite_score(st), st)


def main():
    parser = argparse.ArgumentParser(description="8 个 GRU 组合档位下注网格回测（仅搜阈值+B+p）")
    parser.add_argument("--no-b0", action="store_true", help="不搜索 B0，固定 B0=0")
    parser.add_argument(
        "--scientific",
        action="store_true",
        help="与 backtest_gru_regime 一致：每笔用公式重算 PnL（手续费+滑点，赢=bet*(1/价-1)-fee，输=-bet-fee）",
    )
    parser.add_argument(
        "--no-cap",
        action="store_true",
        help="不封顶单笔 3000：下注=资金×比例；默认封顶（与 400/≥6万固定3000 规则一致，bet=min(资金×比例,3000)）",
    )
    parser.add_argument(
        "--order-price",
        type=float,
        default=None,
        help="scientific 时固定下单单价；默认 0.527（模拟交易平均价），不传则用每笔 tokenPrice",
    )
    parser.add_argument(
        "--data-src",
        type=str,
        default=None,
        help="parquet 目录，默认 data/raw",
    )
    parser.add_argument(
        "--val-days",
        type=int,
        default=365,
        help="留出最后 N 天做验证集：选参在「除最后 N 天」上，排名与最终资金在「最后 N 天」上；0=不切分",
    )
    parser.add_argument(
        "--val-no-leak",
        action="store_true",
        help="验证期起点取 max(最后N天起点, 无泄漏起始日)，保证验证集无泄漏",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
        help="阈值搜索步长，默认 0.01（0.5,0.51,…,0.6 共 11 点）",
    )
    parser.add_argument(
        "--threshold-top",
        type=int,
        default=3,
        help="每个 GRU 组合取综合评分前 N 组进入排名与输出，默认 3（共 24 组）",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="两个排名 CSV 输出目录，默认项目根目录",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="快速网格：阈值步长 0.02、p 取 [2,4,6,8,10]，约 13 倍加速，精度略降",
    )
    parser.add_argument(
        "--coarse",
        action="store_true",
        help="粗网格：阈值步长 0.02、p1~p4 仅取 2,4,6,8,10 共 5^4=625 组，点数约减半、过拟合弱",
    )
    parser.add_argument(
        "--multi-segment",
        type=int,
        default=0,
        metavar="N",
        help="多段验证：在无泄漏区间内切 N 段(2 或 3)，每段分别回测，按 --segment-aggregate 聚合得分选参；数据仅用无泄漏起点到 parquet 末尾",
    )
    parser.add_argument(
        "--segment-aggregate",
        type=str,
        choices=("mean", "min"),
        default="mean",
        help="多段验证时的聚合方式：mean=平均各段最终资金，min=最差段（逼每段都不差）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="网格搜索并行进程数，默认 1；多核可设 4 等以进一步加速",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从上次检查点续跑（Ctrl+C 中断后，用 --resume 从未完成的组合继续）",
    )
    parser.add_argument(
        "--robust-config",
        action="store_true",
        help="对每组合 top-K 取超参中位数作为稳健配置，再跑一次验证并输出",
    )
    parser.add_argument(
        "--calibration-grid",
        action="store_true",
        help="超参中加入校准维度：在训练段拟合 isotonic/sigmoid，验证段上对比 无/isotonic/sigmoid，CSV 含方案、Δ、p1～p4、校准",
    )
    parser.add_argument(
        "--sim-12-only",
        action="store_true",
        help="仅跑模拟盘 12 组合验证：与档位网格同一验证期（--val-days/--val-no-leak），用现有 12 组合的模型+阈值+有无校准+有无动态，输出 gru_regime_backtest_ranking_sim12.csv",
    )
    args = parser.parse_args()

    _ds = (getattr(args, "data_src", None) or "").strip()
    data_src = Path(_ds) if _ds else DATA_RAW
    if not data_src.is_absolute():
        data_src = PROJECT_ROOT / data_src

    # ========== 仅跑模拟盘 12 组合验证（与 tiered 同一验证期，无档位网格） ==========
    if getattr(args, "sim_12_only", False):
        sys.path.insert(0, str(PROJECT_ROOT))
        spec = importlib.util.spec_from_file_location(
            "bgr", PROJECT_ROOT / "scripts" / "backtest_gru_regime.py"
        )
        bgr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bgr)
        val_days = getattr(args, "val_days", 365)
        val_no_leak = getattr(args, "val_no_leak", False)
        if val_no_leak:
            after_date = bgr.get_no_leak_start_date(data_src, None, "BTC_USDT")
            print("  [--sim-12-only] 验证期对齐无泄漏起始日: {}（{} 天）".format(after_date, val_days))
        else:
            try:
                _min, max_date = _parquet_date_range(data_src, "BTC_USDT")
                after_dt = pd.Timestamp(max_date, tz="UTC") - pd.Timedelta(days=val_days)
                after_date = after_dt.strftime("%Y-%m-%d")
                print("  [--sim-12-only] 验证期: {} 起 {} 天（parquet 截止 {}）".format(after_date, val_days, max_date))
            except Exception as e:
                print("  无法取验证起点: {}，改用无泄漏起始日".format(e))
                after_date = bgr.get_no_leak_start_date(data_src, None, "BTC_USDT")
        out_dir = getattr(args, "out_dir", None)
        out_dir = Path(out_dir) if out_dir else PROJECT_ROOT
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        from scripts.backtest_gru_regime_report import run_report
        order_price = getattr(args, "order_price", None) or 0.503
        df = run_report(
            after_date=after_date,
            year_days=val_days,
            order_price=order_price,
            run_sim_12_combos=True,
            max_trades_per_day=0,
            data_src=data_src,
        )
        if df.empty:
            print("  无有效记录")
            return
        csv_path = out_dir / "gru_regime_backtest_ranking_sim12.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print("\n  已写出: {} （12 组合按最终资金降序）".format(csv_path))
        print(df.head(12).to_string(index=False))
        return

    print("加载 8 个 GRU 组合（parquet + GRU 推理，各用对应模型目录）…")
    sys.path.insert(0, str(PROJECT_ROOT))
    spec = importlib.util.spec_from_file_location(
        "bgr", PROJECT_ROOT / "scripts" / "backtest_gru_regime.py"
    )
    bgr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bgr)
    _get_backtest_df = bgr.get_backtest_df_one_asset
    _get_device = bgr.get_device
    device = _get_device(use_mps=not getattr(args, "no_mps", False))
    trades_by_combo: Dict[str, List[Dict[str, Any]]] = {}
    for cfg in GRU_MODEL_CONFIGS:
        asset = cfg["asset"]
        models_override = MODELS_BEST if cfg.get("use_1h4h") else MODELS_BEST_NO1H4H
        try:
            min_date, max_date = _parquet_date_range(data_src, asset)
        except Exception as e:
            print("  跳过 {}: {}".format(cfg["key"], e))
            trades_by_combo[cfg["key"]] = []
            continue
        try:
            merged = _get_backtest_df(
                asset, test_days=99999, device=device, data_src=data_src,
                after_date=min_date, end_date=max_date,
                models_best_override=models_override,
            )
        except Exception as e:
            print("  跳过 {}: {}".format(cfg["key"], e))
            trades_by_combo[cfg["key"]] = []
            continue
        trades = _df_to_trades(merged)
        trades_by_combo[cfg["key"]] = trades
        print("  {}: {} 笔".format(cfg["key"], len(trades)))
    total_trades = sum(len(t) for t in trades_by_combo.values())
    if total_trades == 0:
        print("未成功加载任何 GRU 交易序列，请检查 data/raw/*.parquet 与 GRU 模型")
        return

    val_days = getattr(args, "val_days", 365)
    val_no_leak = getattr(args, "val_no_leak", False)
    train_by_combo: Dict[str, List[Dict[str, Any]]] = {}
    val_by_combo: Dict[str, List[Dict[str, Any]]] = {}
    val_end_global = ""
    if val_days > 0:
        all_dates = []
        for tlist in trades_by_combo.values():
            for t in tlist or []:
                d = t.get("date") or t.get("timestamp")
                if d:
                    if isinstance(d, (int, float)):
                        d = pd.to_datetime(d, unit="ms", utc=True).strftime("%Y-%m-%d")
                    else:
                        d = str(d)[:10]
                    all_dates.append(d)
        if not all_dates:
            val_days = 0
        else:
            val_end_global = max(all_dates)
            val_start_dt = pd.Timestamp(val_end_global, tz="UTC") - pd.Timedelta(days=val_days)
            val_start_cand = val_start_dt.strftime("%Y-%m-%d")
            get_no_leak = getattr(bgr, "get_no_leak_start_date", None) if val_no_leak else None
            for cfg in GRU_MODEL_CONFIGS:
                key, asset = cfg["key"], cfg["asset"]
                trades = trades_by_combo.get(key, [])
                if not trades:
                    train_by_combo[key] = []
                    val_by_combo[key] = []
                    continue
                if val_no_leak and get_no_leak:
                    try:
                        no_leak = get_no_leak(data_src, asset=asset)
                        val_start = max(val_start_cand, no_leak)
                    except Exception:
                        val_start = val_start_cand
                else:
                    val_start = val_start_cand
                train_by_combo[key] = [t for t in trades if (t.get("date") or str(t.get("timestamp", ""))[:10]) < val_start]
                val_by_combo[key] = [t for t in trades if (t.get("date") or str(t.get("timestamp", ""))[:10]) >= val_start]
            n_train = sum(len(v) for v in train_by_combo.values())
            n_val = sum(len(v) for v in val_by_combo.values())
            print("  留出验证期: {} 天，验证区间 {} ~ {}（选参在训练段，排名在验证段）".format(val_days, val_start_cand, val_end_global))
            if val_no_leak:
                print("  已对齐无泄漏起始日（--val-no-leak）")
            print("  训练段 {} 笔，验证段 {} 笔".format(n_train, n_val))
    if val_days == 0:
        train_by_combo = {k: list(v) for k, v in trades_by_combo.items()}
        val_by_combo = {k: list(v) for k, v in trades_by_combo.items()}
        print("  未切分（--val-days 0），选参与排名均在全量数据上")

    # 校准网格：每组合在训练段拟合 isotonic/sigmoid，验证段得到 无/isotonic/sigmoid 三份，便于对比胜率与最终资金
    calibration_grid = getattr(args, "calibration_grid", False)
    val_by_cal_by_combo: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    if calibration_grid and val_days > 0 and multi_segment not in (2, 3):
        for key in GRU_COMBOS:
            train_t = train_by_combo.get(key, [])
            val_t = val_by_combo.get(key, [])
            val_by_cal_by_combo[key] = {"无": list(val_t)}
            for method in ("isotonic", "sigmoid"):
                cal = _fit_calibrator(train_t, method)
                val_by_cal_by_combo[key][method] = _apply_calibration_to_trades(val_t, cal, method)
        print("  校准网格: 每组合 无/isotonic/sigmoid 三档，验证段上对比")

    # 多段验证：仅用无泄漏起点→parquet 末尾，切成 N 段，每段回测后按 mean/min 聚合得分选参
    multi_segment = getattr(args, "multi_segment", 0)
    segments_by_combo: Dict[str, List[List[Dict[str, Any]]]] = {}
    no_leak_trades_by_combo: Dict[str, List[Dict[str, Any]]] = {}
    if multi_segment in (2, 3):
        get_no_leak = getattr(bgr, "get_no_leak_start_date", None)
        for cfg in GRU_MODEL_CONFIGS:
            key, asset = cfg["key"], cfg["asset"]
            trades = trades_by_combo.get(key, [])
            if not trades:
                segments_by_combo[key] = []
                no_leak_trades_by_combo[key] = []
                continue
            if get_no_leak:
                try:
                    no_leak_start = get_no_leak(data_src, asset=asset)
                except Exception:
                    no_leak_start = None
            else:
                no_leak_start = None
            if no_leak_start:
                no_leak_trades = [t for t in trades if _trades_to_date_str(t) >= no_leak_start]
            else:
                no_leak_trades = list(trades)
            no_leak_trades_by_combo[key] = no_leak_trades
            segments_by_combo[key] = _split_trades_by_segments(no_leak_trades, multi_segment)
        n_seg_trades = [sum(len(s) for s in segments_by_combo.get(k, [])) for k in GRU_COMBOS]
        print("  多段验证: 无泄漏区间内切 {} 段，选参按 --segment-aggregate 聚合（各段笔数: {}）".format(
            multi_segment, n_seg_trades))

    search_b0 = not args.no_b0
    use_scientific = getattr(args, "scientific", False)
    # GRU 的 trades 来自 _df_to_trades，无 pnl/tokenPrice；非 scientific 时每笔 pnl 为 None 会被跳过，回测无效
    if not use_scientific and total_trades > 0:
        any_has_pnl_or_price = False
        for tlist in trades_by_combo.values():
            for t in (tlist or [])[:200]:
                if t.get("pnl") is not None:
                    any_has_pnl_or_price = True
                    break
                try:
                    if t.get("tokenPrice") is not None and float(t.get("tokenPrice", 0) or 0) > 0:
                        any_has_pnl_or_price = True
                        break
                except (TypeError, ValueError):
                    pass
            if any_has_pnl_or_price:
                break
        if not any_has_pnl_or_price:
            print("错误：当前交易序列来自 GRU 推理，无 pnl/tokenPrice 字段；非 --scientific 时每笔会被跳过，回测无意义。请加上 --scientific。")
            return
    cap_bet_3000 = not getattr(args, "no_cap", False)
    fixed_order_price = getattr(args, "order_price", None)
    if use_scientific and fixed_order_price is None:
        fixed_order_price = FIXED_ORDER_PRICE_DEFAULT
    threshold_step = getattr(args, "threshold_step", 0.01)
    threshold_top = getattr(args, "threshold_top", 3)
    out_dir = getattr(args, "out_dir", None)
    out_dir = Path(out_dir) if out_dir else PROJECT_ROOT
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    print("\n8 个 GRU 组合：搜索 阈值(0.5～0.6)+B+p（缩减网格），每组合取前 {} 组，共 {} 组进入排名".format(threshold_top, 8 * threshold_top))
    if getattr(args, "fast", False):
        print("  加速: --fast 快速网格（阈值步长 0.02、p 取 [2,4,6,8,10]，约 13 倍加速）")
    if getattr(args, "coarse", False):
        print("  粗网格: --coarse（阈值步长 0.02、p1~p4 取 2,4,6,8,10 共 5^4=625 组）")
    if multi_segment in (2, 3):
        print("  多段验证: {} 段，聚合方式 {}".format(multi_segment, getattr(args, "segment_aggregate", "mean")))
    workers = getattr(args, "workers", 1)
    if workers > 1:
        print("  加速: --workers {} 多进程".format(workers))
    if use_scientific:
        print("  规则: --scientific（手续费+滑点），单价 {:.3f}，单笔封顶 3000".format(fixed_order_price))
    else:
        print("  规则: 默认（用历史 PnL 缩放）")
    if not cap_bet_3000:
        print("  下注: 不封顶")

    global _CHECKPOINT_PATH, _RESUME_INTERRUPTED
    out_dir.mkdir(parents=True, exist_ok=True)
    _CHECKPOINT_PATH = out_dir / "backtest_tiered_bet_grid_checkpoint.pkl"

    # 断点续跑：加载检查点
    top_k_results_by_combo: Dict[str, List[Tuple[float, Tuple[int, int, int, int, int, int, int, int], float, Dict[str, Any]]]] = {}
    completed_combos: Set[str] = set()
    if getattr(args, "resume", False) and _CHECKPOINT_PATH.exists():
        try:
            with open(_CHECKPOINT_PATH, "rb") as f:
                ck = pickle.load(f)
            top_k_results_by_combo = dict(ck.get("top_k_results_by_combo", {}))
            completed_combos = set(ck.get("completed_combos", []))
            print("  已加载检查点：{} 个组合已完成，续跑中 …".format(len(completed_combos)))
        except Exception as e:
            print("  检查点加载失败: {}，从头跑".format(e))

    def _save_checkpoint() -> None:
        if _CHECKPOINT_PATH is None:
            return
        try:
            with open(_CHECKPOINT_PATH, "wb") as f:
                pickle.dump({"completed_combos": list(completed_combos), "top_k_results_by_combo": top_k_results_by_combo}, f)
        except Exception as e:
            print("  检查点保存失败: {}".format(e))

    def _sigint_handler(sig: int, frame: Any) -> None:
        global _RESUME_INTERRUPTED
        _RESUME_INTERRUPTED = True
        _save_checkpoint()
        print("\n  收到 Ctrl+C，已保存进度。续跑请用: python scripts/backtest_tiered_bet_grid.py --scientific --resume [其他参数同本次]")
        sys.exit(0)

    signal.signal(signal.SIGINT, _sigint_handler)

    # 每组合取前 threshold_top 组：(threshold, params, score, train_stats)
    t0_total = time.perf_counter()
    for idx, log_dir in enumerate(GRU_COMBOS, 1):
        # 断点续跑：已完成的组合直接使用保存结果
        if log_dir in completed_combos:
            label = COMBO_META.get(log_dir, {}).get("label", log_dir)
            print("  [{}/8] {}: 使用已保存结果（--resume）".format(idx, label))
            continue
        # 多段验证时在无泄漏各段上选参；否则有验证期时在训练段上选参，无则全量
        use_multisegment = multi_segment in (2, 3)
        segments = segments_by_combo.get(log_dir, []) if use_multisegment else []
        trades = (train_by_combo.get(log_dir, []) if val_days > 0 and not use_multisegment else trades_by_combo.get(log_dir, []))
        if use_multisegment:
            if not segments or not any(segments):
                trades = no_leak_trades_by_combo.get(log_dir, [])
            else:
                trades = None  # 用 segments 跑，不取单段
        if not use_multisegment and not trades:
            top_k_results_by_combo[log_dir] = []
            completed_combos.add(log_dir)
            _save_checkpoint()
            continue
        if use_multisegment and not segments:
            top_k_results_by_combo[log_dir] = []
            completed_combos.add(log_dir)
            _save_checkpoint()
            continue
        label = COMBO_META.get(log_dir, {}).get("label", log_dir)
        t0 = time.perf_counter()
        if use_multisegment:
            print("  [{}/8] {}: 多段验证 阈值(0.5～0.6 步0.02) + B + p（5^4）…".format(idx, label))
            gru_results = grid_search_gru_multisegment(
                segments,
                segment_aggregate=getattr(args, "segment_aggregate", "mean"),
                threshold_step=0.02,
                search_b0=search_b0,
                use_scientific=use_scientific,
                cap_bet_3000=cap_bet_3000,
                fixed_order_price=fixed_order_price,
                coarse_grid=True,
                workers=getattr(args, "workers", 1),
            )
        elif calibration_grid and val_days > 0:
            train_t = train_by_combo.get(log_dir, [])
            val_t = val_by_combo.get(log_dir, [])
            print("  [{}/8] {}: 搜索 阈值+B+p+校准(无/isotonic/sigmoid) …".format(idx, label))
            gru_results = grid_search_gru_with_calibration(
                train_t,
                val_t,
                threshold_step=0.02,
                search_b0=search_b0,
                use_scientific=use_scientific,
                cap_bet_3000=cap_bet_3000,
                fixed_order_price=fixed_order_price,
                coarse_grid=True,
            )
        else:
            print("  [{}/8] {}: 搜索 阈值(0.5～0.6 步{}) + B + p …".format(idx, label, threshold_step))
            gru_results = grid_search_gru_with_threshold(
                trades,
                threshold_min=THRESHOLD_SEARCH_MIN,
                threshold_max=THRESHOLD_SEARCH_MAX,
                threshold_step=threshold_step,
                search_b0=search_b0,
                use_scientific=use_scientific,
                cap_bet_3000=cap_bet_3000,
                fixed_order_price=fixed_order_price,
                reduced_grid=True,
                fast_grid=getattr(args, "fast", False),
                coarse_grid=getattr(args, "coarse", False),
                workers=getattr(args, "workers", 1),
            )
        elapsed = time.perf_counter() - t0
        if gru_results:
            top_n = gru_results[:threshold_top]
            top_k_results_by_combo[log_dir] = top_n
            item0 = top_n[0]
            best_th = item0[0]
            best_params = item0[1]
            best_score = item0[2]
            best_st = item0[3]
            best_cal = item0[4] if len(item0) == 5 else "无"
            b0, b1, b2, b3, p1, p2, p3, p4 = best_params
            cap_disp = best_st.get("aggregate_score", best_st["final_capital"])
            caps_str = ""
            if "segment_final_capitals" in best_st:
                caps_str = " 各段=" + ",".join("{:.0f}".format(c) for c in best_st["segment_final_capitals"])
            print("      最佳 校准={} 阈值={:.2f} B0={} p={},{},{},{}  得分={:.3f} 资金={:.0f}{} 胜率={:.1f}% 回撤={:.1f}%  ({:.1f}s)".format(
                best_cal, best_th, b0, p1, p2, p3, p4, best_score, cap_disp, caps_str, best_st["win_rate_pct"], best_st["max_drawdown_pct"], elapsed))
            print("      前 {} 组（按得分）:".format(len(top_n)))
            for i, item in enumerate(top_n, 1):
                th, params, score, st = item[0], item[1], item[2], item[3]
                cal = item[4] if len(item) == 5 else "无"
                b0, b1, b2, b3, p1, p2, p3, p4 = params
                cap_d = st.get("aggregate_score", st["final_capital"])
                seg_str = " 各段=" + ",".join("{:.0f}".format(c) for c in st["segment_final_capitals"]) if "segment_final_capitals" in st else ""
                print("        {:2d}. 校准={} 阈值={:.2f} B0={} p={},{},{},{}  分={:.3f} 资金={:.0f}{} 回撤={:.1f}%".format(
                    i, cal, th, b0, p1, p2, p3, p4, score, cap_d, seg_str, st["max_drawdown_pct"]))
        else:
            top_k_results_by_combo[log_dir] = []
            print("      无有效结果")
        completed_combos.add(log_dir)
        _save_checkpoint()
    elapsed_total = time.perf_counter() - t0_total
    print(f"  8 组合网格搜索总耗时: {elapsed_total:.1f} 秒 ({elapsed_total/60:.1f} 分钟)")

    # 构建每组合前 threshold_top 行：在验证段/全量/无泄漏全段上跑一遍得到 val_stats；校准时用对应校准的验证段
    if multi_segment in (2, 3):
        rank_trades_src = no_leak_trades_by_combo
    else:
        rank_trades_src = val_by_combo if val_days > 0 else trades_by_combo
    all_entries: List[Tuple[str, int, float, Tuple[int, int, int, int, int, int, int, int], float, Dict[str, Any], Dict[str, Any], str]] = []
    for log_dir in GRU_COMBOS:
        for rank_in_combo, item in enumerate(top_k_results_by_combo.get(log_dir, []), 1):
            th, params, train_score, train_st = item[0], item[1], item[2], item[3]
            cal = item[4] if len(item) == 5 else "无"
            if calibration_grid and val_by_cal_by_combo:
                trades = (val_by_cal_by_combo.get(log_dir) or {}).get(cal, rank_trades_src.get(log_dir, []))
            else:
                trades = rank_trades_src.get(log_dir, [])
            b0, b1, b2, b3, p1, p2, p3, p4 = params
            val_st = simulate_combo_stats(
                trades, th, b0, b1, b2, b3, p1, p2, p3, p4,
                use_scientific=use_scientific,
                cap_bet_3000=cap_bet_3000,
                fixed_order_price=fixed_order_price,
            )
            all_entries.append((log_dir, rank_in_combo, th, params, train_score, train_st, val_st, cal))
    n_entries = len(all_entries)

    # 按验证段最终资金降序（用于 CSV1 与控制台排名表）
    by_val_capital = sorted(all_entries, key=lambda x: x[6]["final_capital"], reverse=True)
    total_pnl_all = sum(e[6]["final_capital"] - INITIAL_CAPITAL for e in all_entries)

    print("\n" + "=" * 70)
    if val_days > 0:
        print("  {} 组排名（验证段，每组合 {} 组共 {} 组，按验证段最终资金降序）".format(n_entries, threshold_top, n_entries))
    else:
        print("  {} 组排名（每组合 {} 组共 {} 组，按最终资金降序）".format(n_entries, threshold_top, n_entries))
    print("=" * 70)
    if val_days > 0:
        print("  验证段总 PnL（{} 组各自曲线之和）: ${:+.2f}  下表为验证段最终资金/胜率/回撤".format(n_entries, total_pnl_all))
    else:
        print("  总 PnL（{} 组）: ${:+.2f}".format(n_entries, total_pnl_all))
    print(f"  {'排名':<4} {'组合内':<4} {'币种':<4} {'周期':<4} {'交易次数':<6} {'胜场':<4} {'败场':<4} {'胜率%':<6} {'最终资金':<8} {'盈亏%':<8} {'最大回撤%':<8} {'最低/初始%':<10} {'阈值':<4} 组合")
    print("  " + "-" * 110)
    for r, (log_dir, rank_in_combo, th, params, _ts, _tst, val_st, _cal) in enumerate(by_val_capital, 1):
        meta = COMBO_META.get(log_dir, {"symbol": "-", "label": log_dir, "dynamic": "无", "calibration": "有"})
        b0, b1, b2, b3, p1, p2, p3, p4 = params
        rank_label = _ranking_label(log_dir, th, params, meta)
        print(f"  {r:<4} {rank_in_combo:<4} {meta['symbol']:<4} {'15m':<4} {val_st['n_trades']:<6} {val_st['wins']:<4} {val_st['losses']:<4} {val_st['win_rate_pct']:.1f}%   {val_st['final_capital']:.0f}     {val_st['profit_pct']:+.1f}%   {val_st['max_drawdown_pct']:.1f}%      {val_st['min_over_initial_pct']:.1f}%       {th:.2f}   {rank_label}")
    print("=" * 70)

    # 两个 CSV：每组合保留 top-K 行（共 8*K 行），中文表头；多段验证时增加各段资金列
    CSV_CN_HEADERS = {
        "rank_val": "排名", "row": "排名", "combo": "组合", "symbol": "币种", "rank_in_combo": "组内名次",
        "scheme": "方案", "calibration": "校准",
        "threshold": "阈值", "b0": "B0", "b1": "B1", "b2": "B2", "b3": "B3",
        "p1": "p1", "p2": "p2", "p3": "p3", "p4": "p4",
        "train_score": "训练综合分", "train_final_capital": "训练最终资金",
        "train_win_rate_pct": "训练胜率%", "train_max_drawdown_pct": "训练最大回撤%",
        "val_final_capital": "验证最终资金", "val_win_rate_pct": "验证胜率%",
        "val_max_drawdown_pct": "验证最大回撤%", "val_profit_pct": "验证盈亏%",
        "val_n_trades": "验证交易次数", "val_wins": "验证胜场", "val_losses": "验证败场",
        "val_min_over_initial_pct": "验证最低/初始%",
        "aggregate_score": "聚合得分",
        "val_seg1_final_capital": "验证段1最终资金", "val_seg2_final_capital": "验证段2最终资金",
        "val_seg3_final_capital": "验证段3最终资金",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    csv1_path = out_dir / "ranking_val_365d.csv"
    csv2_path = out_dir / "ranking_top{}_per_combo.csv".format(threshold_top)

    def row_for_csv(rank_key: str, rank_val: int, combo: str, rank_in_combo: int, th: float, params: Tuple[int, int, int, int, int, int, int, int], train_score: float, train_st: Dict[str, Any], val_st: Dict[str, Any], cal: str = "无") -> Dict[str, Any]:
        b0, b1, b2, b3, p1, p2, p3, p4 = params
        meta = COMBO_META.get(combo, {"symbol": "-", "label": combo})
        scheme = "{}_Δ{:.2f}_p{},{},{},{}_校准{}".format(combo, th, p1, p2, p3, p4, cal)
        out = {
            rank_key: rank_val,
            "combo": combo,
            "symbol": meta.get("symbol", "-"),
            "rank_in_combo": rank_in_combo,
            "scheme": scheme,
            "calibration": cal,
            "threshold": th,
            "b0": b0, "b1": b1, "b2": b2, "b3": b3,
            "p1": p1, "p2": p2, "p3": p3, "p4": p4,
            "train_score": train_score,
            "train_final_capital": train_st.get("final_capital"),
            "train_win_rate_pct": train_st.get("win_rate_pct"),
            "train_max_drawdown_pct": train_st.get("max_drawdown_pct"),
            "val_final_capital": val_st.get("final_capital"),
            "val_win_rate_pct": val_st.get("win_rate_pct"),
            "val_max_drawdown_pct": val_st.get("max_drawdown_pct"),
            "val_profit_pct": val_st.get("profit_pct"),
            "val_n_trades": val_st.get("n_trades"),
            "val_wins": val_st.get("wins"),
            "val_losses": val_st.get("losses"),
            "val_min_over_initial_pct": val_st.get("min_over_initial_pct"),
        }
        if "aggregate_score" in train_st:
            out["aggregate_score"] = train_st["aggregate_score"]
        seg_caps = train_st.get("segment_final_capitals") or []
        for i, cap in enumerate(seg_caps):
            if i < 3:
                out["val_seg{}_final_capital".format(i + 1)] = cap
        return out

    # CSV1: 全部 top-K 行（8*K 行），按验证最终资金降序
    rows1 = []
    for r, entry in enumerate(
        sorted(all_entries, key=lambda x: x[6]["final_capital"], reverse=True), 1
    ):
        log_dir, rank_in_combo, th, params, train_score, train_st, val_st, cal = entry
        rows1.append(row_for_csv("rank_val", r, log_dir, rank_in_combo, th, params, train_score, train_st, val_st, cal))
    df1 = pd.DataFrame(rows1).rename(columns=CSV_CN_HEADERS)
    df1.to_csv(csv1_path, index=False, encoding="utf-8-sig")
    print("  已写出: {} （按验证最终资金降序，每组合 top{} 共 {} 行）".format(csv1_path, threshold_top, len(rows1)))

    # CSV2: 全部 top-K 行，按组合+训练/聚合得分降序（每组合内 1..K）
    rows2 = []
    for entry in all_entries:
        log_dir, rank_in_combo, th, params, train_score, train_st, val_st, cal = entry
        rows2.append(row_for_csv("row", rank_in_combo, log_dir, rank_in_combo, th, params, train_score, train_st, val_st, cal))
    def _row_sort_key(row: Dict[str, Any]) -> Tuple[str, float]:
        sc = row.get("train_score") or row.get("aggregate_score") or 0
        return (row["combo"], -(float(sc) if sc is not None else 0))
    rows2_sorted = sorted(rows2, key=_row_sort_key)
    df2 = pd.DataFrame(rows2_sorted).rename(columns=CSV_CN_HEADERS)
    df2.to_csv(csv2_path, index=False, encoding="utf-8-sig")
    print("  已写出: {} （每组合 top{} 按得分降序，共 {} 行）".format(csv2_path, threshold_top, len(rows2)))

    # 可选：每组合 top-K 超参取中位数作为「稳健配置」，再跑一次验证并输出
    if getattr(args, "robust_config", False) and all_entries:
        robust_rows = []
        for log_dir in GRU_COMBOS:
            top_n = top_k_results_by_combo.get(log_dir, [])
            if not top_n:
                continue
            th_list = [x[0] for x in top_n]
            b0_list = [x[1][0] for x in top_n]
            b1_list = [x[1][1] for x in top_n]
            b2_list = [x[1][2] for x in top_n]
            b3_list = [x[1][3] for x in top_n]
            p1_list = [x[1][4] for x in top_n]
            p2_list = [x[1][5] for x in top_n]
            p3_list = [x[1][6] for x in top_n]
            p4_list = [x[1][7] for x in top_n]
            med_th = round(statistics.median(th_list), 2)
            med_b0 = int(round(statistics.median(b0_list)))
            med_b1 = int(round(statistics.median(b1_list)))
            med_b2 = int(round(statistics.median(b2_list)))
            med_b3 = int(round(statistics.median(b3_list)))
            med_p1 = int(round(statistics.median(p1_list)))
            med_p2 = int(round(statistics.median(p2_list)))
            med_p3 = int(round(statistics.median(p3_list)))
            med_p4 = int(round(statistics.median(p4_list)))
            trades_robust = rank_trades_src.get(log_dir, [])
            if not trades_robust:
                continue
            robust_st = simulate_combo_stats(
                trades_robust, med_th, med_b0, med_b1, med_b2, med_b3, med_p1, med_p2, med_p3, med_p4,
                use_scientific=use_scientific,
                cap_bet_3000=cap_bet_3000,
                fixed_order_price=fixed_order_price,
            )
            meta = COMBO_META.get(log_dir, {"symbol": "-", "label": log_dir})
            robust_rows.append({
                "combo": log_dir,
                "symbol": meta.get("symbol", "-"),
                "threshold": med_th,
                "b0": med_b0, "b1": med_b1, "b2": med_b2, "b3": med_b3,
                "p1": med_p1, "p2": med_p2, "p3": med_p3, "p4": med_p4,
                "val_final_capital": robust_st["final_capital"],
                "val_win_rate_pct": robust_st["win_rate_pct"],
                "val_max_drawdown_pct": robust_st["max_drawdown_pct"],
            })
        if robust_rows:
            csv_robust = out_dir / "ranking_robust_config.csv"
            df_robust = pd.DataFrame(robust_rows)
            df_robust.to_csv(csv_robust, index=False, encoding="utf-8-sig")
            print("  已写出: {} （每组合 top-{} 中位数稳健配置验证，{} 行）".format(csv_robust, threshold_top, len(robust_rows)))


if __name__ == "__main__":
    main()
