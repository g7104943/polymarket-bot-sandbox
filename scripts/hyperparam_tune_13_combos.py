#!/usr/bin/env python3
"""
13 组合超参搜索（方案 A：Δ + p1～p4 + 校准）。
- 组合：5 旧模型 + 8 GRU（与 查看实时日志 1～13 一致）。
- 数据：无泄漏起始日到 parquet 末尾（用满）；选参用前 N-30 天（2 段），冷验证用最后 30 天。
- 超参：Δ ∈ {0.02, 0.05, 0.08} 且 Δ≤1-阈值；p1～p4 ∈ {2,4,6,8,10}；校准 无/isotonic/sigmoid（旧模型）。
- 约束：两段合计笔数 ≥ N_min（默认 30）；得分 = 最差段资金 × (1 − λ×最差段回撤%)，λ 默认 0.4。
- 输出：每组合 top 5、稳健配置（top5 的 Δ/p1～p4 中位数、校准众数，非单点 top1）整段/冷验证、365 天回测（稳健 vs 固定未超参）、综合排序按综合得分；表含综合排名（超参）与未超参排名（按固定 365 天资金）。

用法（项目根目录）:
  python scripts/hyperparam_tune_13_combos.py --val-no-leak
  python scripts/hyperparam_tune_13_combos.py --val-no-leak --workers 4
"""
import argparse
import os
import sys
import statistics
import time
from itertools import product
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
sys.path.insert(0, str(PROJECT_ROOT))

# 与 Polymarket crypto 15m 规则一致：初始 400，≥6 万固定 3000，<6 万按总资金比例；输到 6 万以下继续按比例
INITIAL_CAPITAL = 400.0
MIN_BET = 1.0
CAPITAL_THRESHOLD = 60000.0  # 超过此金额固定单笔 3000，否则按比例
MAX_BET_CAP = 3000.0
DEFAULT_FEE_RATE = 0.001
DEFAULT_SLIPPAGE = 0.001
FIXED_ORDER_PRICE_DEFAULT = 0.527  # 回测固定买入价，贴近真实订单

# 超参网格（粗）
DELTA_CANDIDATES = [0.02, 0.05, 0.08]  # 再按 1-threshold 截断
P_VALUES = [2, 4, 6, 8, 10]  # p1～p4，5^4=625
B1, B2, B3 = 2, 4, 6  # 档位边界（百分点，超出 base 的量）
CALIBRATION_OPTIONS = ["无", "isotonic", "sigmoid"]

# 选参与冷验证（选参天数由「无泄漏总天数 - COLD_DAYS」动态计算，冷验证固定 30 天）
COLD_DAYS = 30
MIN_TRADES_TOTAL = 30
MIN_TRADES_COLD = 10  # 综合排序：冷验证 30 天最少笔数
LAMBDA_DRAWDOWN = 0.4
TOP_K = 5
REPORT_365_DAYS = 365  # 综合排序用 365 天回测对比
# 固定组合（未超参）默认参数：与现有 13 进程一致，用于 365 天对比
FIXED_DELTA = 0.02
FIXED_P1, FIXED_P2, FIXED_P3, FIXED_P4 = 2, 4, 6, 8
# 综合排序科学打分：综合得分 = 最终资金 × (1 − λ_dd×回撤%/100) × (最低/初始%/100)，初始=400
RANK_LAMBDA_DD = 0.4


def get_tier(confidence: float, threshold: float, b0: int, b1: int, b2: int, b3: int) -> int:
    """档 1～4，base = threshold + b0*0.01，边界 +b1,+b2,+b3（百分点）。"""
    base = threshold + b0 * 0.01
    c1 = base + b1 * 0.01
    c2 = base + b2 * 0.01
    c3 = base + b3 * 0.01
    if confidence <= c1:
        return 1
    if confidence <= c2:
        return 2
    if confidence <= c3:
        return 3
    return 4


def run_equity_curve(
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
    fixed_order_price: float = FIXED_ORDER_PRICE_DEFAULT,
) -> Tuple[float, List[float], int, int, float]:
    """跑资金曲线，返回 (final_capital, equity_curve, wins, losses, max_drawdown_pct)。"""
    base = threshold + b0 * 0.01
    pcts = {1: p1 * 0.01, 2: p2 * 0.01, 3: p3 * 0.01, 4: p4 * 0.01}
    sorted_trades = sorted(
        trades,
        key=lambda t: str(t.get("date") or t.get("timestamp") or ""),
    )
    capital = INITIAL_CAPITAL
    peak_capital = INITIAL_CAPITAL
    wins = losses = 0
    equity_curve = [capital]
    for t in sorted_trades:
        if capital < MIN_BET:
            break
        conf = t.get("confidence")
        if conf is None or conf < base:
            continue
        tier = get_tier(conf, threshold, b0, b1, b2, b3)
        ratio = pcts.get(tier, 0)
        if ratio <= 0:
            continue
        # ≥6 万固定 3000，<6 万按总资金比例；输到 6 万以下下一笔再按比例
        if capital >= CAPITAL_THRESHOLD:
            bet = min(MAX_BET_CAP, capital)
        else:
            bet = capital * ratio
            bet = min(bet, capital)
        if bet < MIN_BET:
            continue
        price = fixed_order_price or t.get("tokenPrice") or FIXED_ORDER_PRICE_DEFAULT
        if price <= 0:
            continue
        # Polymarket 二元市场：赢时每份兑 $1，输时份额归零。买价 price → 得 bet/price 份；赢=bet*(1/price-1)-fee，输=-bet。手续费在买入时从份额扣除，输时不另扣。
        fee = bet * (DEFAULT_FEE_RATE + DEFAULT_SLIPPAGE)
        if t.get("result") == "win":
            new_pnl = bet * (1.0 / price - 1.0) - fee  # 赢：净利 bet*(1/price-1)-fee
            wins += 1
        else:
            new_pnl = -bet  # 输：亏本金，不另扣费
            losses += 1
        capital += new_pnl
        if capital > peak_capital:
            peak_capital = capital
        equity_curve.append(capital)
    peak = INITIAL_CAPITAL
    max_dd = 0.0
    for c in equity_curve:
        if c > peak:
            peak = c
        if peak > 0:
            dd = (peak - c) / peak
            if dd > max_dd:
                max_dd = dd
    return (capital, equity_curve, wins, losses, max_dd * 100.0)


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
    fixed_order_price: float = FIXED_ORDER_PRICE_DEFAULT,
) -> Dict[str, Any]:
    """返回 n_trades, wins, losses, win_rate_pct, final_capital, max_drawdown_pct, min_over_initial_pct."""
    capital, equity_curve, wins, losses, max_dd_pct = run_equity_curve(
        trades, threshold, b0, b1, b2, b3, p1, p2, p3, p4, fixed_order_price
    )
    n_trades = wins + losses
    min_cap = min(equity_curve) if equity_curve else INITIAL_CAPITAL
    min_over_initial = (min_cap / INITIAL_CAPITAL * 100.0) if INITIAL_CAPITAL > 0 else 100.0
    win_rate_pct = (wins / n_trades * 100.0) if n_trades > 0 else 0.0
    return {
        "n_trades": n_trades,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": win_rate_pct,
        "final_capital": capital,
        "max_drawdown_pct": max_dd_pct,
        "min_over_initial_pct": min_over_initial,
    }


def _date_str(t: Dict) -> str:
    d = t.get("date") or t.get("timestamp")
    if d is None:
        return ""
    if isinstance(d, (int, float)):
        return pd.to_datetime(d, unit="ms", utc=True).strftime("%Y-%m-%d")
    return str(d)[:10]


def filter_trades_by_date(trades: List[Dict], start_date: str, end_date: str) -> List[Dict]:
    """保留 date 在 [start_date, end_date] 内的 trades。"""
    out = []
    for t in trades:
        d = _date_str(t)
        if d and start_date <= d <= end_date:
            out.append(t)
    return out


def filter_trades_by_entry(
    trades: List[Dict],
    threshold_up: float,
    threshold_down: Optional[float],
    delta: float,
    is_dual: bool,
) -> List[Dict]:
    """入场：confidence >= threshold_up + Δ。双阈值时 UP/DOWN 条已用 confidence 表示，统一用该条件。"""
    out = []
    for t in trades:
        conf = t.get("confidence")
        if conf is None:
            continue
        if conf >= threshold_up + delta:
            out.append(t)
    return out


def get_trades_gru(
    asset: str,
    start_date: str,
    end_date: str,
    data_src: Path,
    models_override: Optional[Path],
    device,
) -> List[Dict[str, Any]]:
    """GRU 组合：取 (start_date, end_date] 的 backtest df，转为 trades（confidence=effective_prob, result）。"""
    from scripts.backtest_gru_regime import get_backtest_df_one_asset

    merged = get_backtest_df_one_asset(
        asset,
        test_days=99999,
        device=device,
        data_src=data_src,
        after_date=start_date,
        end_date=end_date,
        models_best_override=models_override,
    )
    trades = []
    for _, row in merged.iterrows():
        ts = row.get("timestamp") or row.get("date")
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
        # 已去掉止损（reversal）：仅按预测与实际是否一致判胜败
        result = "win" if (actual_up == 1 and pred_up == 1) or (actual_up == 0 and pred_up == 0) else "loss"
        trades.append({
            "timestamp": ts,
            "date": date_str,
            "confidence": effective,
            "result": result,
            "amount": 1.0,
        })
    return trades


def get_trades_old(
    symbol: str,
    models_dir: Path,
    start_date: str,
    end_date: str,
    calibrator: Optional[object] = None,
    calibration_method: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """旧模型：加载数据、预测、得到 (date, confidence, result)。confidence 为 effective_prob。双阈值时 DOWN 条用 1-pred。"""
    from scripts.backtest_simulation import load_test_data, build_features
    from src.python.predictor import load_predictor, _find_symbol_timeframe_model, apply_calibration
    from src.python.feature_engineering import add_multi_timeframe_features

    TIMEFRAME = "15m"
    model_dir = _find_symbol_timeframe_model(symbol, TIMEFRAME, models_root=models_dir)
    if not model_dir:
        return []
    try:
        model, feats, meta = load_predictor(model_dir)
    except Exception:
        return []
    cal = calibrator if calibrator is not None else meta.get("calibrator")
    cal_m = calibration_method or meta.get("calibration")
    df = load_test_data(
        symbol, TIMEFRAME,
        test_days=99999,
        train_cutoff_date=start_date,
        end_date=end_date,
    )
    if df.empty or len(df) < 50:
        return []
    df = build_features(df)
    if "15m" in str(TIMEFRAME) and feats and any((str(f) or "").startswith("mtf_") for f in feats):
        try:
            df = add_multi_timeframe_features(df, symbol.replace("/", "_").lower())
        except Exception:
            pass
    lookahead = 1
    if "close" in df.columns:
        df["future_return"] = df["close"].shift(-lookahead) / df["close"] - 1
        df["actual_up"] = (df["future_return"] > 0).astype(int)
    cols = [c for c in feats if c in df.columns]
    missing = [c for c in feats if c not in df.columns]
    if missing:
        df[missing] = 0
    df = df.dropna(subset=cols + ["future_return", "actual_up"])
    if "actual_up" not in df.columns or len(df) < 10:
        return []
    df = df.iloc[:-lookahead] if lookahead > 0 else df.reset_index(drop=True)
    X = df[cols].fillna(0)
    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(X)[:, 1]).ravel()
    else:
        probs = np.asarray(model.predict(X)).ravel().astype(float)
    if cal is not None and cal_m:
        probs = apply_calibration(cal, cal_m, probs)
    df["pred_prob"] = probs
    df["pred_up"] = (df["pred_prob"] >= 0.5).astype(int)
    trades = []
    for _, row in df.iterrows():
        pred_prob = float(row["pred_prob"])
        pred_up = int(row["pred_up"])
        actual_up = int(row["actual_up"])
        date_val = row.get("date")
        if date_val is None:
            continue
        date_str = pd.to_datetime(date_val).strftime("%Y-%m-%d") if hasattr(date_val, "strftime") else str(date_val)[:10]
        ts = row.get("timestamp")
        if ts is None and date_val is not None:
            ts = int(pd.to_datetime(date_val).value // 10**6)
        result = "win" if (actual_up == 1 and pred_up == 1) or (actual_up == 0 and pred_up == 0) else "loss"
        effective = pred_prob if pred_up == 1 else (1.0 - pred_prob)
        trades.append({
            "timestamp": ts,
            "date": date_str,
            "confidence": effective,
            "result": result,
            "amount": 1.0,
        })
    return trades


def get_trades_old_dual(
    symbol: str,
    models_dir: Path,
    start_date: str,
    end_date: str,
    th_up: float,
    th_down: float,
    calibrator: Optional[object] = None,
    calibration_method: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """双阈值旧模型：UP 条 confidence=pred_prob，DOWN 条 confidence=1-pred_prob。"""
    from scripts.backtest_simulation import load_test_data, build_features
    from src.python.predictor import load_predictor, _find_symbol_timeframe_model, apply_calibration
    from src.python.feature_engineering import add_multi_timeframe_features

    TIMEFRAME = "15m"
    model_dir = _find_symbol_timeframe_model(symbol, TIMEFRAME, models_root=models_dir)
    if not model_dir:
        return []
    try:
        model, feats, meta = load_predictor(model_dir)
    except Exception:
        return []
    cal = calibrator or meta.get("calibrator")
    cal_m = calibration_method or meta.get("calibration")
    df = load_test_data(symbol, TIMEFRAME, test_days=99999, train_cutoff_date=start_date, end_date=end_date)
    if df.empty or len(df) < 50:
        return []
    df = build_features(df)
    if feats and any((str(f) or "").startswith("mtf_") for f in feats):
        try:
            df = add_multi_timeframe_features(df, symbol.replace("/", "_").lower())
        except Exception:
            pass
    lookahead = 1
    if "close" in df.columns:
        df["future_return"] = df["close"].shift(-lookahead) / df["close"] - 1
        df["actual_up"] = (df["future_return"] > 0).astype(int)
    for c in feats:
        if c not in df.columns:
            df[c] = 0
    cols = [c for c in feats if c in df.columns]
    df = df.dropna(subset=cols + ["future_return", "actual_up"])
    if "actual_up" not in df.columns or len(df) < 10:
        return []
    df = df.iloc[:-lookahead] if lookahead > 0 else df.reset_index(drop=True)
    X = df[cols].fillna(0)
    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(X)[:, 1]).ravel()
    else:
        probs = np.asarray(model.predict(X)).ravel().astype(float)
    if cal is not None and cal_m:
        probs = apply_calibration(cal, cal_m, probs)
    df["pred_prob"] = probs
    trades = []
    for _, row in df.iterrows():
        pred_prob = float(row["pred_prob"])
        actual_up = int(row["actual_up"])
        date_val = row.get("date")
        if date_val is None:
            continue
        date_str = pd.to_datetime(date_val).strftime("%Y-%m-%d") if hasattr(date_val, "strftime") else str(date_val)[:10]
        ts = row.get("timestamp") or int(pd.to_datetime(date_val).value // 10**6)
        if pred_prob >= th_up:
            trades.append({
                "timestamp": ts, "date": date_str,
                "confidence": pred_prob,
                "result": "win" if actual_up == 1 else "loss",
                "amount": 1.0,
            })
        elif pred_prob <= th_down:
            trades.append({
                "timestamp": ts, "date": date_str,
                "confidence": 1.0 - pred_prob,
                "result": "win" if actual_up == 0 else "loss",
                "amount": 1.0,
            })
    return trades


def fit_calibrator(trades: List[Dict], method: str):
    """在 trades 上拟合 isotonic 或 sigmoid。"""
    if not trades or method == "无":
        return None
    X = np.array([t["confidence"] for t in trades], dtype=float).reshape(-1, 1)
    y = np.array([1 if t.get("result") == "win" else 0 for t in trades], dtype=int)
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


def apply_calibration_to_trades(trades: List[Dict], calibrator, method: str) -> List[Dict]:
    """对 confidence 校准后返回新列表。双阈值条已用 confidence 表示一侧，校准后仍用该侧概率。"""
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


# 多进程 worker 用全局状态（每组合在 initializer 里设置）
_WORKER_RAW_SEG1: List[Dict] = []
_WORKER_RAW_SEG2: List[Dict] = []
_WORKER_TH_UP: float = 0.5
_WORKER_TH_DOWN: Optional[float] = None
_WORKER_IS_DUAL: bool = False
_WORKER_FIXED_PRICE: float = FIXED_ORDER_PRICE_DEFAULT
_WORKER_LAMBDA_DD: float = LAMBDA_DRAWDOWN
_WORKER_MIN_TRADES: int = MIN_TRADES_TOTAL


def _init_worker(
    raw_seg1: List[Dict],
    raw_seg2: List[Dict],
    th_up: float,
    th_down: Optional[float],
    is_dual: bool,
    fixed_price: float,
    lambda_dd: float,
    min_trades: int,
) -> None:
    global _WORKER_RAW_SEG1, _WORKER_RAW_SEG2, _WORKER_TH_UP, _WORKER_TH_DOWN, _WORKER_IS_DUAL
    global _WORKER_FIXED_PRICE, _WORKER_LAMBDA_DD, _WORKER_MIN_TRADES
    _WORKER_RAW_SEG1 = raw_seg1
    _WORKER_RAW_SEG2 = raw_seg2
    _WORKER_TH_UP = th_up
    _WORKER_TH_DOWN = th_down
    _WORKER_IS_DUAL = is_dual
    _WORKER_FIXED_PRICE = fixed_price
    _WORKER_LAMBDA_DD = lambda_dd
    _WORKER_MIN_TRADES = min_trades


def _eval_one_point(args: Tuple[float, int, int, int, int, str]) -> Optional[Tuple]:
    """单点评估：(delta, p1, p2, p3, p4, cal_label) -> (score, min_cap, cap1, cap2, wr1, wr2, dd1, dd2, n1, n2, delta, p1, p2, p3, p4, cal, st1, st2) 或 None。"""
    global _WORKER_RAW_SEG1, _WORKER_RAW_SEG2, _WORKER_TH_UP, _WORKER_TH_DOWN, _WORKER_IS_DUAL
    global _WORKER_FIXED_PRICE, _WORKER_LAMBDA_DD, _WORKER_MIN_TRADES
    delta, p1, p2, p3, p4, cal_label = args
    # base = 阈值 + Δ，即 threshold=th_up、b0=Δ*100（get_tier 里 base = threshold + b0*0.01）
    threshold = _WORKER_TH_UP
    b0 = int(round(delta * 100))
    if cal_label != "无":
        cal = fit_calibrator(_WORKER_RAW_SEG1, cal_label)
        seg1_t = apply_calibration_to_trades(_WORKER_RAW_SEG1, cal, cal_label)
        seg2_t = apply_calibration_to_trades(_WORKER_RAW_SEG2, cal, cal_label)
    else:
        seg1_t = list(_WORKER_RAW_SEG1)
        seg2_t = list(_WORKER_RAW_SEG2)
    seg1_t = filter_trades_by_entry(seg1_t, _WORKER_TH_UP, _WORKER_TH_DOWN, delta, _WORKER_IS_DUAL)
    seg2_t = filter_trades_by_entry(seg2_t, _WORKER_TH_UP, _WORKER_TH_DOWN, delta, _WORKER_IS_DUAL)
    st1 = simulate_combo_stats(seg1_t, threshold, b0, B1, B2, B3, p1, p2, p3, p4, _WORKER_FIXED_PRICE)
    st2 = simulate_combo_stats(seg2_t, threshold, b0, B1, B2, B3, p1, p2, p3, p4, _WORKER_FIXED_PRICE)
    n_total = st1["n_trades"] + st2["n_trades"]
    if n_total < _WORKER_MIN_TRADES:
        return None
    min_cap = min(st1["final_capital"], st2["final_capital"])
    worst_dd = st1["max_drawdown_pct"] if st1["final_capital"] <= st2["final_capital"] else st2["max_drawdown_pct"]
    score = min_cap * (1.0 - _WORKER_LAMBDA_DD * worst_dd / 100.0)
    return (
        score, min_cap,
        st1["final_capital"], st2["final_capital"],
        st1["win_rate_pct"], st2["win_rate_pct"],
        st1["max_drawdown_pct"], st2["max_drawdown_pct"],
        st1["n_trades"], st2["n_trades"],
        delta, p1, p2, p3, p4, cal_label,
        st1, st2,
    )


def main():
    parser = argparse.ArgumentParser(description="13 组合超参搜索（Δ + p1～p4 + 校准）")
    parser.add_argument("--val-no-leak", action="store_true", help="验证期起点取无泄漏起始日")
    parser.add_argument("--data-src", type=str, default=None)
    parser.add_argument("--order-price", type=float, default=FIXED_ORDER_PRICE_DEFAULT)
    parser.add_argument("--min-trades", type=int, default=MIN_TRADES_TOTAL)
    parser.add_argument("--lambda", dest="lambda_dd", type=float, default=LAMBDA_DRAWDOWN)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default=None)
    # 步骤 5 增强：多窗口 walk-forward + 最低笔数门槛
    parser.add_argument("--robust-mode", action="store_true",
                        help="启用多窗口 walk-forward 稳健模式（替代默认的两段验证）")
    parser.add_argument("--n-windows", type=int, default=4,
                        help="walk-forward 窗口数（默认 4）")
    parser.add_argument("--window-days", type=int, default=25,
                        help="每个验证窗口天数（默认 25 天）")
    parser.add_argument("--min-trades-per-window", type=int, default=20,
                        help="每个窗口最低交易笔数（低于则跳过不计分，默认 20）")
    # 步骤 3 增强：交易摩擦
    parser.add_argument("--enable-trading-friction", action="store_true",
                        help="启用交易摩擦模拟（需先安装 TradingFrictionSimulator）")
    parser.add_argument("--friction-config", type=str, default=None,
                        help="交易摩擦配置文件路径")
    # 步骤 8 增强：Ensemble
    parser.add_argument("--use-ensemble", action="store_true",
                        help="使用 Ensemble 模型（需先训练基模型）")
    parser.add_argument("--uncertainty-adjustment", action="store_true",
                        help="启用不确定性仓位调整")
    # 步骤 5.5：5 层交易防护系统
    parser.add_argument("--advanced-rules", action="store_true",
                        help="启用 5 层交易防护系统（Kelly + 回撤熔断 + 连胜调节），替代 p1-p4 档位")
    parser.add_argument("--kelly-frac", type=float, default=0.33,
                        help="Kelly 安全系数（默认 0.33 = 1/3 Kelly）")
    parser.add_argument("--min-edge", type=float, default=0.02,
                        help="最小边际优势（默认 0.02）")
    parser.add_argument("--dd-halt", type=float, default=0.30,
                        help="回撤暂停线（默认 0.30 = 30%%）")
    parser.add_argument("--advanced-sample", type=int, default=500,
                        help="5 层系统随机采样数（从全网格中采样，默认 500）")
    args = parser.parse_args()

    data_src = Path(args.data_src or DATA_RAW)
    if not data_src.is_absolute():
        data_src = PROJECT_ROOT / data_src
    out_dir = Path(args.out_dir or PROJECT_ROOT)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    from scripts.backtest_gru_regime import get_no_leak_start_date, get_parquet_end_date, get_device
    from scripts.backtest_gru_regime_report import OLD_POLYMARKET_COMBOS, SIM_13_GRU_COMBOS

    no_leak_start = get_no_leak_start_date(data_src, None, "BTC_USDT")
    end_full = get_parquet_end_date(data_src, "BTC_USDT")
    start_dt = pd.Timestamp(no_leak_start, tz="UTC")
    end_dt = pd.Timestamp(end_full, tz="UTC")
    total_days = (end_dt - start_dt).days + 1
    selection_days = max(60, total_days - COLD_DAYS)
    cold_start_dt = end_dt - pd.Timedelta(days=COLD_DAYS - 1)
    cold_start = cold_start_dt.strftime("%Y-%m-%d")
    end_selection_dt = cold_start_dt - pd.Timedelta(days=1)
    end_selection = end_selection_dt.strftime("%Y-%m-%d")
    seg1_days = (selection_days + 1) // 2
    seg1_end = (start_dt + pd.Timedelta(days=seg1_days - 1)).strftime("%Y-%m-%d")
    seg2_start = (start_dt + pd.Timedelta(days=seg1_days)).strftime("%Y-%m-%d")

    # 365 天窗口：用于综合排序与固定组合对比（取无泄漏起 365 天，不足则用满）
    end_365_dt = min(start_dt + pd.Timedelta(days=REPORT_365_DAYS - 1), end_dt)
    end_365 = end_365_dt.strftime("%Y-%m-%d")
    days_365 = (end_365_dt - start_dt).days + 1

    print("无泄漏起始日: {}  数据末尾: {}  总无泄漏 {} 天".format(no_leak_start, end_full, total_days))
    print("选参: {}～{} ({} 天)  冷验证: {}～{} ({} 天)".format(
        no_leak_start, end_selection, selection_days, cold_start, end_full, COLD_DAYS))
    print("段1: {}～{}  段2: {}～{}  365天: {}～{} ({} 天)".format(
        no_leak_start, seg1_end, seg2_start, end_selection, no_leak_start, end_365, days_365))

    device = get_device(use_mps=True)
    models_best_no1h4h = PROJECT_ROOT / "experiments" / "gru_regime_v1" / "outputs" / "models_best_no1h4h"
    models_best = PROJECT_ROOT / "experiments" / "gru_regime_v1" / "outputs" / "models_best"

    # 13 组合：5 旧 + 8 GRU
    combos: List[Dict[str, Any]] = []
    for c in OLD_POLYMARKET_COMBOS:
        th_up = c.get("prob_threshold") or c.get("prob_threshold_up") or 0.5
        th_down = c.get("prob_threshold_down")
        is_dual = c.get("prob_threshold_up") is not None and c.get("prob_threshold_down") is not None
        combos.append({
            "id": c["name"],
            "type": "old",
            "symbol": c["symbol"],
            "models_dir": PROJECT_ROOT / c["models_dir"] if not Path(c["models_dir"]).is_absolute() else Path(c["models_dir"]),
            "threshold_up": th_up,
            "threshold_down": th_down,
            "is_dual": is_dual,
        })
    for c in SIM_13_GRU_COMBOS:
        combos.append({
            "id": c["combo_name"],
            "type": "gru",
            "asset": c["asset"],
            "threshold_up": c["threshold"],
            "threshold_down": None,
            "is_dual": False,
            "use_no1h4h": c.get("use_no1h4h", False),
        })

    lambda_dd = getattr(args, "lambda_dd", LAMBDA_DRAWDOWN)
    min_trades = getattr(args, "min_trades", MIN_TRADES_TOTAL)
    fixed_price = getattr(args, "order_price", FIXED_ORDER_PRICE_DEFAULT)
    workers = getattr(args, "workers", 1)
    if workers > 1:
        print("多进程: workers={}".format(workers))

    all_rows: List[Dict[str, Any]] = []
    robust_rows: List[Dict[str, Any]] = []

    for combo_idx, combo in enumerate(combos):
        combo_id = combo["id"]
        th_up = combo["threshold_up"]
        th_down = combo.get("threshold_down")
        is_dual = combo.get("is_dual", False)

        delta_list = [d for d in DELTA_CANDIDATES if d <= (1.0 - th_up)]
        if not delta_list:
            delta_list = [min(DELTA_CANDIDATES)]

        if combo["type"] == "old":
            if is_dual:
                raw_seg1 = get_trades_old_dual(
                    combo["symbol"], combo["models_dir"], no_leak_start, seg1_end,
                    th_up, th_down or 0.0, None, None,
                )
                raw_seg2 = get_trades_old_dual(
                    combo["symbol"], combo["models_dir"], seg2_start, end_selection,
                    th_up, th_down or 0.0, None, None,
                )
                raw_cold = get_trades_old_dual(
                    combo["symbol"], combo["models_dir"], cold_start, end_full,
                    th_up, th_down or 0.0, None, None,
                )
                raw_365 = get_trades_old_dual(
                    combo["symbol"], combo["models_dir"], no_leak_start, end_365,
                    th_up, th_down or 0.0, None, None,
                )
            else:
                raw_seg1 = get_trades_old(combo["symbol"], combo["models_dir"], no_leak_start, seg1_end)
                raw_seg2 = get_trades_old(combo["symbol"], combo["models_dir"], seg2_start, end_selection)
                raw_cold = get_trades_old(combo["symbol"], combo["models_dir"], cold_start, end_full)
                raw_365 = get_trades_old(combo["symbol"], combo["models_dir"], no_leak_start, end_365)
            cal_options = CALIBRATION_OPTIONS
        else:
            models_ov = models_best_no1h4h if combo.get("use_no1h4h") and models_best_no1h4h.exists() else models_best
            all_trades = get_trades_gru(combo["asset"], no_leak_start, end_selection, data_src, models_ov, device)
            raw_seg1 = filter_trades_by_date(all_trades, no_leak_start, seg1_end)
            raw_seg2 = filter_trades_by_date(all_trades, seg2_start, end_selection)
            all_trades_cold = get_trades_gru(combo["asset"], cold_start, end_full, data_src, models_ov, device)
            raw_cold = filter_trades_by_date(all_trades_cold, cold_start, end_full)
            if end_365 <= end_selection:
                raw_365 = filter_trades_by_date(all_trades, no_leak_start, end_365)
            else:
                all_trades_365 = get_trades_gru(combo["asset"], no_leak_start, end_365, data_src, models_ov, device)
                raw_365 = filter_trades_by_date(all_trades_365, no_leak_start, end_365)
            cal_options = ["无"]

        if not raw_seg1 and not raw_seg2:
            print("  [{}] 无交易数据，跳过".format(combo_id))
            continue

        # ======= 5 层交易防护系统分支 =======
        use_advanced = getattr(args, "advanced_rules", False)
        if use_advanced:
            from src.python.trading_rules import (
                AdvancedEquityCurve, advanced_score,
                iter_advanced_grid, run_advanced_backtest,
                ADVANCED_PARAM_GRID, ADVANCED_FIXED_PARAMS,
            )
            import random as _rng

            adv_grid = iter_advanced_grid()
            adv_sample_n = getattr(args, "advanced_sample", 500)
            if len(adv_grid) > adv_sample_n:
                _rng.seed(42)
                adv_grid = _rng.sample(adv_grid, adv_sample_n)
            print("  [{}] 5 层系统: {} 参数组合".format(combo_id, len(adv_grid)))

            # 合并两段用于搜索
            seg_all = raw_seg1 + raw_seg2
            adv_results = []
            for ap in adv_grid:
                st_all = run_advanced_backtest(seg_all, th_up, ap, fixed_price)
                if st_all["n_trades"] < min_trades:
                    continue
                sc = advanced_score(
                    st_all["final_capital"],
                    st_all["final_capital"] * st_all["min_over_initial_pct"] / 100.0,
                    st_all["max_drawdown_pct"],
                )
                adv_results.append((sc, st_all, ap))

            if not adv_results:
                print("  [{}] 5 层系统无满足最小笔数的配置，跳过".format(combo_id))
                continue

            adv_results.sort(key=lambda x: x[0], reverse=True)
            adv_top5 = adv_results[:TOP_K]

            for rank_i, (sc, st, ap) in enumerate(adv_top5, 1):
                all_rows.append({
                    "组合": combo_id,
                    "组内排名": rank_i,
                    "规则": "5层防护",
                    "min_edge": ap.get("min_edge"),
                    "kelly_frac": ap.get("kelly_frac"),
                    "roll_window": ap.get("roll_window"),
                    "cold_threshold": ap.get("cold_threshold"),
                    "dd_level1": ap.get("dd_level1"),
                    "dd_level2": ap.get("dd_level2"),
                    "dd_halt": ap.get("dd_halt"),
                    "max_capital_pct": ap.get("max_capital_pct"),
                    "得分": round(sc, 2),
                    "整段最终资金": st["final_capital"],
                    "整段胜率%": st["win_rate_pct"],
                    "整段最大回撤%": st["max_drawdown_pct"],
                    "整段最低/初始%": st["min_over_initial_pct"],
                    "整段笔数": st["n_trades"],
                    "过滤_边际不足": st["filtered_by_edge"],
                    "过滤_连败": st["filtered_by_streak"],
                    "回撤暂停次数": st["halted_count"],
                })

            # 稳健配置：取 top5 各参数中位数
            import statistics as _stat
            robust_ap = {}
            for key in ADVANCED_PARAM_GRID.keys():
                vals = [r[2].get(key) for r in adv_top5 if r[2].get(key) is not None]
                if vals:
                    robust_ap[key] = round(_stat.median(vals), 4)
            robust_ap.update(ADVANCED_FIXED_PARAMS)

            # 整段回测 + 365 天回测 + 冷验证
            st_robust_adv = run_advanced_backtest(seg_all, th_up, robust_ap, fixed_price)
            st_365_adv = run_advanced_backtest(raw_365, th_up, robust_ap, fixed_price)
            st_cold_adv = run_advanced_backtest(raw_cold, th_up, robust_ap, fixed_price)

            # 固定（未超参，原始 p1-p4）365 天对比
            b0_fixed = int(round(FIXED_DELTA * 100))
            raw_365_fixed_t = filter_trades_by_entry(raw_365, th_up, th_down, FIXED_DELTA, is_dual)
            st_365_fixed = simulate_combo_stats(
                raw_365_fixed_t, th_up, b0_fixed, B1, B2, B3,
                FIXED_P1, FIXED_P2, FIXED_P3, FIXED_P4, fixed_price,
            )

            robust_rows.append({
                "组合": combo_id,
                "类型": "5层稳健",
                "规则": "5层防护",
                "阈值": th_up,
                **{k: v for k, v in robust_ap.items() if k in ADVANCED_PARAM_GRID},
                "整段最终资金": st_robust_adv["final_capital"],
                "整段胜率%": st_robust_adv["win_rate_pct"],
                "整段最大回撤%": st_robust_adv["max_drawdown_pct"],
                "整段最低/初始%": st_robust_adv["min_over_initial_pct"],
                "整段笔数": st_robust_adv["n_trades"],
                "365天最终资金_稳健": st_365_adv["final_capital"],
                "365天胜率%_稳健": st_365_adv["win_rate_pct"],
                "365天最大回撤%_稳健": st_365_adv["max_drawdown_pct"],
                "365天最低/初始%_稳健": st_365_adv["min_over_initial_pct"],
                "365天笔数_稳健": st_365_adv["n_trades"],
                "365天最终资金_固定": round(st_365_fixed["final_capital"], 2),
                "365天胜率%_固定": round(st_365_fixed["win_rate_pct"], 1),
                "365天最大回撤%_固定": round(st_365_fixed["max_drawdown_pct"], 1),
                "365天最低/初始%_固定": round(st_365_fixed["min_over_initial_pct"], 2),
                "365天笔数_固定": st_365_fixed["n_trades"],
                "冷验证最终资金": st_cold_adv["final_capital"],
                "冷验证胜率%": st_cold_adv["win_rate_pct"],
                "冷验证回撤%": st_cold_adv["max_drawdown_pct"],
                "冷验证最低/初始%": st_cold_adv["min_over_initial_pct"],
                "冷验证笔数": st_cold_adv["n_trades"],
            })

            all_rows.append({
                "组合": combo_id,
                "组内排名": "稳健",
                "规则": "5层防护",
                **{k: v for k, v in robust_ap.items() if k in ADVANCED_PARAM_GRID},
                "阈值": th_up,
                "得分": round(advanced_score(
                    st_365_adv["final_capital"],
                    st_365_adv["final_capital"] * st_365_adv["min_over_initial_pct"] / 100.0,
                    st_365_adv["max_drawdown_pct"],
                ), 2),
                "整段最终资金": st_robust_adv["final_capital"],
                "整段胜率%": st_robust_adv["win_rate_pct"],
                "整段最大回撤%": st_robust_adv["max_drawdown_pct"],
                "整段最低/初始%": st_robust_adv["min_over_initial_pct"],
                "整段笔数": st_robust_adv["n_trades"],
                "365天最终资金_稳健": st_365_adv["final_capital"],
                "365天胜率%_稳健": st_365_adv["win_rate_pct"],
                "365天最大回撤%_稳健": st_365_adv["max_drawdown_pct"],
                "365天最低/初始%_稳健": st_365_adv["min_over_initial_pct"],
                "365天笔数_稳健": st_365_adv["n_trades"],
                "365天最终资金_固定": round(st_365_fixed["final_capital"], 2),
                "365天胜率%_固定": round(st_365_fixed["win_rate_pct"], 1),
                "365天最大回撤%_固定": round(st_365_fixed["max_drawdown_pct"], 1),
                "365天最低/初始%_固定": round(st_365_fixed["min_over_initial_pct"], 2),
                "365天笔数_固定": st_365_fixed["n_trades"],
                "冷验证最终资金": st_cold_adv["final_capital"],
                "冷验证胜率%": st_cold_adv["win_rate_pct"],
                "冷验证回撤%": st_cold_adv["max_drawdown_pct"],
                "冷验证最低/初始%": st_cold_adv["min_over_initial_pct"],
                "冷验证笔数": st_cold_adv["n_trades"],
            })

            print("  [{}] 5层 top1 得分={:.0f} 整段资金={:.0f} 回撤={:.1f}% 最低/初始={:.1f}%".format(
                combo_id, adv_top5[0][0], adv_top5[0][1]["final_capital"],
                adv_top5[0][1]["max_drawdown_pct"], adv_top5[0][1]["min_over_initial_pct"],
            ))
            continue  # 跳过下面的旧 p1-p4 搜索
        # ======= 旧 p1-p4 分档搜索（--advanced-rules 未启用时） =======

        all_params = [
            (delta, p1, p2, p3, p4, cal_label)
            for delta in delta_list
            for (p1, p2, p3, p4) in product(P_VALUES, repeat=4)
            for cal_label in cal_options
        ]
        workers = getattr(args, "workers", 1)
        results: List[Tuple] = []

        if workers > 1:
            t0_combo = time.perf_counter()
            with Pool(
                workers,
                initializer=_init_worker,
                initargs=(
                    raw_seg1, raw_seg2, th_up, th_down, is_dual,
                    fixed_price, lambda_dd, min_trades,
                ),
            ) as pool:
                for r in pool.imap_unordered(_eval_one_point, all_params, chunksize=min(50, max(1, len(all_params) // (workers * 4)))):
                    if r is not None:
                        results.append(r)
            elapsed = time.perf_counter() - t0_combo
            print("  [{}] 网格 {} 点，有效 {} 组，耗时 {:.1f}s (workers={})".format(
                combo_id, len(all_params), len(results), elapsed, workers))
        else:
            for delta in delta_list:
                # base = 阈值 + Δ：传 threshold=th_up, b0=Δ*100
                threshold = th_up
                b0 = int(round(delta * 100))
                for (p1, p2, p3, p4) in product(P_VALUES, repeat=4):
                    for cal_label in cal_options:
                        if cal_label != "无":
                            cal = fit_calibrator(raw_seg1, cal_label)
                            seg1_t = apply_calibration_to_trades(raw_seg1, cal, cal_label)
                            seg2_t = apply_calibration_to_trades(raw_seg2, cal, cal_label)
                        else:
                            seg1_t = list(raw_seg1)
                            seg2_t = list(raw_seg2)
                        seg1_t = filter_trades_by_entry(seg1_t, th_up, th_down, delta, is_dual)
                        seg2_t = filter_trades_by_entry(seg2_t, th_up, th_down, delta, is_dual)
                        st1 = simulate_combo_stats(seg1_t, threshold, b0, B1, B2, B3, p1, p2, p3, p4, fixed_price)
                        st2 = simulate_combo_stats(seg2_t, threshold, b0, B1, B2, B3, p1, p2, p3, p4, fixed_price)
                        n_total = st1["n_trades"] + st2["n_trades"]
                        if n_total < min_trades:
                            continue
                        min_cap = min(st1["final_capital"], st2["final_capital"])
                        worst_dd = st1["max_drawdown_pct"] if st1["final_capital"] <= st2["final_capital"] else st2["max_drawdown_pct"]
                        score = min_cap * (1.0 - lambda_dd * worst_dd / 100.0)
                        results.append((
                            score, min_cap,
                            st1["final_capital"], st2["final_capital"],
                            st1["win_rate_pct"], st2["win_rate_pct"],
                            st1["max_drawdown_pct"], st2["max_drawdown_pct"],
                            st1["n_trades"], st2["n_trades"],
                            delta, p1, p2, p3, p4, cal_label,
                            st1, st2,
                        ))

        if not results:
            print("  [{}] 无满足最小笔数的配置，跳过".format(combo_id))
            continue

        # 主排序：得分降序；同分时按段1段2平均资金降序，使顺序确定
        results.sort(key=lambda x: (x[0], -((x[2] + x[3]) / 2)), reverse=True)
        top5 = results[:TOP_K]

        for rank, r in enumerate(top5, 1):
            (score, min_cap, cap1, cap2, wr1, wr2, dd1, dd2, n1, n2,
             delta, p1, p2, p3, p4, cal_label, st1, st2) = r
            all_rows.append({
                "组合": combo_id,
                "组内排名": rank,
                "Δ": delta,
                "p1": p1, "p2": p2, "p3": p3, "p4": p4,
                "校准": cal_label,
                "得分": round(score, 2),
                "最差段资金": round(min_cap, 2),
                "段1最终资金": round(cap1, 2),
                "段2最终资金": round(cap2, 2),
                "段1段2平均资金": round((cap1 + cap2) / 2, 2),
                "段1胜率%": round(wr1, 1),
                "段2胜率%": round(wr2, 1),
                "段1最大回撤%": round(dd1, 1),
                "段2最大回撤%": round(dd2, 1),
                "段1笔数": n1,
                "段2笔数": n2,
            })

        med_delta = statistics.median([r[10] for r in top5])
        med_p1 = int(statistics.median([r[11] for r in top5]))
        med_p2 = int(statistics.median([r[12] for r in top5]))
        med_p3 = int(statistics.median([r[13] for r in top5]))
        med_p4 = int(statistics.median([r[14] for r in top5]))
        cal_list = [r[15] for r in top5]
        mode_cal = max(set(cal_list), key=cal_list.count)

        threshold_robust = th_up
        b0_robust = int(round(med_delta * 100))
        full_selection = raw_seg1 + raw_seg2
        if mode_cal != "无":
            cal_robust = fit_calibrator(raw_seg1, mode_cal)
            full_selection = apply_calibration_to_trades(full_selection, cal_robust, mode_cal)
        full_selection = filter_trades_by_entry(full_selection, th_up, th_down, med_delta, is_dual)
        st_robust = simulate_combo_stats(
            full_selection, threshold_robust, b0_robust, B1, B2, B3,
            med_p1, med_p2, med_p3, med_p4, fixed_price,
        )
        # 365 天回测：稳健配置 vs 固定（未超参）配置，用于综合排序与对比
        raw_365_robust = raw_365
        if mode_cal != "无":
            cal_365 = fit_calibrator(raw_seg1, mode_cal)
            raw_365_robust = apply_calibration_to_trades(raw_365, cal_365, mode_cal)
        raw_365_robust = filter_trades_by_entry(raw_365_robust, th_up, th_down, med_delta, is_dual)
        st_365_robust = simulate_combo_stats(
            raw_365_robust, threshold_robust, b0_robust, B1, B2, B3,
            med_p1, med_p2, med_p3, med_p4, fixed_price,
        )
        # 固定（未超参）：始终用原始 raw_365（不校准）、FIXED_DELTA 与 FIXED_P1～P4，与稳健独立
        b0_fixed = int(round(FIXED_DELTA * 100))
        raw_365_fixed = filter_trades_by_entry(raw_365, th_up, th_down, FIXED_DELTA, is_dual)
        st_365_fixed = simulate_combo_stats(
            raw_365_fixed, th_up, b0_fixed, B1, B2, B3,
            FIXED_P1, FIXED_P2, FIXED_P3, FIXED_P4, fixed_price,
        )
        robust_rows.append({
            "组合": combo_id,
            "类型": "稳健配置",
            "阈值": th_up,
            "Δ": med_delta,
            "p1": med_p1, "p2": med_p2, "p3": med_p3, "p4": med_p4,
            "校准": mode_cal,
            "整段最终资金": round(st_robust["final_capital"], 2),
            "整段胜率%": round(st_robust["win_rate_pct"], 1),
            "整段最大回撤%": round(st_robust["max_drawdown_pct"], 1),
            "整段最低/初始%": round(st_robust["min_over_initial_pct"], 2),
            "整段笔数": st_robust["n_trades"],
            "365天最终资金_稳健": round(st_365_robust["final_capital"], 2),
            "365天胜率%_稳健": round(st_365_robust["win_rate_pct"], 1),
            "365天最大回撤%_稳健": round(st_365_robust["max_drawdown_pct"], 1),
            "365天最低/初始%_稳健": round(st_365_robust["min_over_initial_pct"], 2),
            "365天笔数_稳健": st_365_robust["n_trades"],
            "365天最终资金_固定": round(st_365_fixed["final_capital"], 2),
            "365天胜率%_固定": round(st_365_fixed["win_rate_pct"], 1),
            "365天最大回撤%_固定": round(st_365_fixed["max_drawdown_pct"], 1),
            "365天最低/初始%_固定": round(st_365_fixed["min_over_initial_pct"], 2),
            "365天笔数_固定": st_365_fixed["n_trades"],
        })

        cold_robust_t = raw_cold
        if mode_cal != "无":
            cal_c = fit_calibrator(raw_seg1, mode_cal)
            cold_robust_t = apply_calibration_to_trades(raw_cold, cal_c, mode_cal)
        cold_robust_t = filter_trades_by_entry(cold_robust_t, th_up, th_down, med_delta, is_dual)
        st_cold_robust = simulate_combo_stats(
            cold_robust_t, threshold_robust, b0_robust, B1, B2, B3,
            med_p1, med_p2, med_p3, med_p4, fixed_price,
        )
        best = top5[0]
        (_, _, _, _, _, _, _, _, _, _, d1, p1b, p2b, p3b, p4b, cal1, _, _) = best
        threshold_best = th_up
        b0_best = int(round(d1 * 100))
        cold_best_t = raw_cold
        if cal1 != "无":
            cal_b = fit_calibrator(raw_seg1, cal1)
            cold_best_t = apply_calibration_to_trades(raw_cold, cal_b, cal1)
        cold_best_t = filter_trades_by_entry(cold_best_t, th_up, th_down, d1, is_dual)
        st_cold_best = simulate_combo_stats(
            cold_best_t, threshold_best, b0_best, B1, B2, B3, p1b, p2b, p3b, p4b, fixed_price,
        )
        all_rows.append({
            "组合": combo_id,
            "组内排名": "稳健",
            "阈值": th_up,
            "Δ": med_delta,
            "p1": med_p1, "p2": med_p2, "p3": med_p3, "p4": med_p4,
            "校准": mode_cal,
            "得分": None,
            "最差段资金": None,
            "段1最终资金": None,
            "段2最终资金": None,
            "段1段2平均资金": None,
            "段1胜率%": None,
            "段2胜率%": None,
            "段1最大回撤%": None,
            "段2最大回撤%": None,
            "段1笔数": None,
            "段2笔数": None,
            "整段最终资金": round(st_robust["final_capital"], 2),
            "整段胜率%": round(st_robust["win_rate_pct"], 1),
            "整段最大回撤%": round(st_robust["max_drawdown_pct"], 1),
            "整段最低/初始%": round(st_robust["min_over_initial_pct"], 2),
            "整段笔数": st_robust["n_trades"],
            "冷验证最终资金_稳健": round(st_cold_robust["final_capital"], 2),
            "冷验证胜率%_稳健": round(st_cold_robust["win_rate_pct"], 1),
            "冷验证回撤%_稳健": round(st_cold_robust["max_drawdown_pct"], 1),
            "冷验证最低/初始%_稳健": round(st_cold_robust["min_over_initial_pct"], 2),
            "冷验证笔数_稳健": st_cold_robust["n_trades"],
            "冷验证最终资金_top1": round(st_cold_best["final_capital"], 2),
            "冷验证胜率%_top1": round(st_cold_best["win_rate_pct"], 1),
            "冷验证回撤%_top1": round(st_cold_best["max_drawdown_pct"], 1),
            "冷验证笔数_top1": st_cold_best["n_trades"],
            "365天最终资金_稳健": round(st_365_robust["final_capital"], 2),
            "365天胜率%_稳健": round(st_365_robust["win_rate_pct"], 1),
            "365天最大回撤%_稳健": round(st_365_robust["max_drawdown_pct"], 1),
            "365天最低/初始%_稳健": round(st_365_robust["min_over_initial_pct"], 2),
            "365天笔数_稳健": st_365_robust["n_trades"],
            "365天最终资金_固定": round(st_365_fixed["final_capital"], 2),
            "365天胜率%_固定": round(st_365_fixed["win_rate_pct"], 1),
            "365天最大回撤%_固定": round(st_365_fixed["max_drawdown_pct"], 1),
            "365天最低/初始%_固定": round(st_365_fixed["min_over_initial_pct"], 2),
            "365天笔数_固定": st_365_fixed["n_trades"],
        })

        print("  [{}] top1 得分={:.0f} Δ={} p={},{},{},{} 校准={}  稳健 整段资金={:.0f}".format(
            combo_id, top5[0][0], top5[0][10], top5[0][11], top5[0][12], top5[0][13], top5[0][14], top5[0][15],
            st_robust["final_capital"],
        ))

    if not all_rows:
        print("无有效结果")
        return

    use_advanced = getattr(args, "advanced_rules", False)
    suffix = "_advanced" if use_advanced else ""

    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "hyperparam_tune_13_combos{}.csv".format(suffix)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print("\n已写出: {} （{} 行）".format(csv_path, len(df)))

    if robust_rows:
        df_robust = pd.DataFrame(robust_rows)
        csv_robust = out_dir / "hyperparam_tune_13_combos{}_robust.csv".format(suffix)
        df_robust.to_csv(csv_robust, index=False, encoding="utf-8-sig")
        print("已写出: {} （{} 行）".format(csv_robust, len(robust_rows)))

    # 综合排序：稳健配置，笔数约束，科学打分 综合得分 = 最终资金×(1−λ×回撤%/100)×(最低/初始%/100)，按综合得分降序
    robust_df = df.loc[df["组内排名"] == "稳健"].copy()
    if not robust_df.empty:
        mask = (robust_df["整段笔数"] >= MIN_TRADES_TOTAL)
        if "冷验证笔数_稳健" in robust_df.columns:
            mask = mask & (robust_df["冷验证笔数_稳健"] >= MIN_TRADES_COLD)
        if "365天笔数_稳健" in robust_df.columns:
            mask = mask & (robust_df["365天笔数_稳健"] >= MIN_TRADES_TOTAL)
        rank_df = robust_df.loc[mask].copy()
        # 综合得分（科学打分）：最终资金 × (1 − λ×回撤%/100) × (最低/初始%/100)，初始=400
        if "365天最终资金_稳健" in rank_df.columns and "365天最大回撤%_稳健" in rank_df.columns and "365天最低/初始%_稳健" in rank_df.columns:
            cap = rank_df["365天最终资金_稳健"].astype(float)
            dd = rank_df["365天最大回撤%_稳健"].astype(float)
            min_init = rank_df["365天最低/初始%_稳健"].astype(float)
        else:
            cap = rank_df["整段最终资金"].astype(float)
            dd = rank_df["整段最大回撤%"].astype(float)
            min_init = rank_df["整段最低/初始%"].astype(float) if "整段最低/初始%" in rank_df.columns else 100.0
        rank_df["综合得分"] = (cap * (1.0 - RANK_LAMBDA_DD * dd / 100.0) * (min_init / 100.0)).round(2)
        # 同分时按 365天最终资金_稳健 降序，使排名确定
        sort_cols = ["综合得分"]
        if "365天最终资金_稳健" in rank_df.columns:
            sort_cols.append("365天最终资金_稳健")
        rank_df = rank_df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)
        rank_df.insert(0, "综合排名", range(1, len(rank_df) + 1))
        # 未超参排名：按 365天最终资金_固定 降序的名次，便于与综合排名（按超参）对比
        if "365天最终资金_固定" in rank_df.columns:
            rank_df["未超参排名"] = rank_df["365天最终资金_固定"].rank(ascending=False, method="min").astype(int)
        out_cols = [
            "综合排名", "未超参排名", "组合", "综合得分", "阈值", "Δ", "p1", "p2", "p3", "p4", "校准",
            "365天最终资金_稳健", "365天胜率%_稳健", "365天最大回撤%_稳健", "365天最低/初始%_稳健", "365天笔数_稳健",
            "365天最终资金_固定", "365天胜率%_固定", "365天最大回撤%_固定", "365天最低/初始%_固定", "365天笔数_固定",
            "整段最终资金", "整段胜率%", "整段最大回撤%", "整段最低/初始%", "整段笔数",
            "冷验证最终资金_稳健", "冷验证胜率%_稳健", "冷验证回撤%_稳健", "冷验证最低/初始%_稳健", "冷验证笔数_稳健",
        ]
        rank_df = rank_df[[c for c in out_cols if c in rank_df.columns]]
        csv_rank = out_dir / "hyperparam_tune_13_combos_ranking.csv"
        rank_df.to_csv(csv_rank, index=False, encoding="utf-8-sig")
        print("已写出: {} （{} 行，按 综合得分 排序，λ_dd={}）".format(csv_rank, len(rank_df), RANK_LAMBDA_DD))
        if len(robust_df) != len(rank_df):
            print("  剔除 {} 个组合（未达最小笔数）".format(len(robust_df) - len(rank_df)))
        # 一览：每组合两行（组合_超参 / 组合_未超参），按最终资金统一排名；列顺序：排名, 币种, 周期, 交易次数, 胜场, 败场, 胜率%, 最终资金, 盈亏%, 最大回撤%, 最低/初始%, 阈值, 动态调整, 校准, 组合
        def _combo_to_symbol(name: str) -> str:
            n = (name or "").lower()
            if "eth" in n: return "ETH"
            if "btc" in n: return "BTC"
            if "xrp" in n: return "XRP"
            if "sol" in n: return "SOL"
            return ""
        def _combo_has_dynamic(name: str) -> str:
            return "有" if name in ("logs_eth_10_90", "logs_gru_eth_54", "logs_gru_eth_55_dyn", "logs_gru_xrp_53_no1h4h") else "无"
        out_order = ["排名", "币种", "周期", "交易次数", "胜场", "败场", "胜率%", "最终资金", "盈亏%", "最大回撤%", "最低/初始%", "阈值", "动态调整", "校准", "组合"]
        summary_rows = []
        for _, r in rank_df.iterrows():
            base = str(r.get("组合", ""))
            symbol = _combo_to_symbol(base)
            dyn = _combo_has_dynamic(base)
            # 超参行
            row_t = {"周期": "365天", "组合": base + "_超参", "校准": r.get("校准", "无"), "币种": symbol, "动态调整": dyn}
            if "阈值" in r.index:
                row_t["阈值"] = r["阈值"]
            for key, col in [("交易次数", "365天笔数_稳健"), ("胜率%", "365天胜率%_稳健"),
                             ("最终资金", "365天最终资金_稳健"), ("最大回撤%", "365天最大回撤%_稳健"), ("最低/初始%", "365天最低/初始%_稳健")]:
                if col in r.index and pd.notna(r.get(col)):
                    row_t[key] = r[col]
            if "365天笔数_稳健" in r.index and "365天胜率%_稳健" in r.index:
                n, wr = float(r["365天笔数_稳健"]), float(r["365天胜率%_稳健"])
                row_t["胜场"] = int(round(n * wr / 100.0))
                row_t["败场"] = int(n) - row_t["胜场"]
            if "365天最终资金_稳健" in r.index:
                row_t["盈亏%"] = round((float(r["365天最终资金_稳健"]) - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100.0, 2)
            summary_rows.append(row_t)
            # 未超参行
            row_f = {"周期": "365天", "组合": base + "_未超参", "校准": "无", "币种": symbol, "动态调整": dyn}
            if "阈值" in r.index:
                row_f["阈值"] = r["阈值"]
            for key, col in [("交易次数", "365天笔数_固定"), ("胜率%", "365天胜率%_固定"), ("最终资金", "365天最终资金_固定"),
                             ("最大回撤%", "365天最大回撤%_固定"), ("最低/初始%", "365天最低/初始%_固定")]:
                if col in r.index and pd.notna(r.get(col)):
                    row_f[key] = r[col]
            if "365天笔数_固定" in r.index and "365天胜率%_固定" in r.index:
                n, wr = float(r["365天笔数_固定"]), float(r["365天胜率%_固定"])
                row_f["胜场"] = int(round(n * wr / 100.0))
                row_f["败场"] = int(n) - row_f["胜场"]
            if "365天最终资金_固定" in r.index:
                cap = float(r["365天最终资金_固定"])
                row_f["盈亏%"] = round((cap - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100.0, 2)
            summary_rows.append(row_f)
        summary_df = pd.DataFrame(summary_rows)
        if "最终资金" in summary_df.columns:
            summary_df = summary_df.sort_values("最终资金", ascending=False).reset_index(drop=True)
            summary_df.insert(0, "排名", range(1, len(summary_df) + 1))
        summary_df = summary_df[[c for c in out_order if c in summary_df.columns]]
        csv_summary = out_dir / "超参排名一览.csv"
        summary_df.to_csv(csv_summary, index=False, encoding="utf-8-sig")
        print("已写出: {} （{} 行，含超参/未超参，按最终资金统一排名）".format(csv_summary, len(summary_df)))
        print("  回测下单价格: {}（Polymarket 15m 涨跌：赢=bet×(1/价−1)−fee，输=−bet−fee）".format(FIXED_ORDER_PRICE_DEFAULT))


if __name__ == "__main__":
    main()
