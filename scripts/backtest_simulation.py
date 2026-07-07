"""
回测模拟：使用训练好的 LightGBM 模型，模拟 Polymarket 下注，计算各币种/周期的胜率和盈亏。

功能：
  - 模拟初始资金（默认 400 USDC），每次下注比例（默认 5%）
  - 按 symbol × timeframe 分别统计胜率
  - 生成可视化图表和 HTML 报告
  - 排名显示哪个币/周期胜率最高

用法：
  python scripts/backtest_simulation.py --initial-capital 400 --bet-ratio 0.05
  python scripts/backtest_simulation.py --output-html report.html

注意：本脚本不使用任何内存自动清理功能（如 gc.collect()），以保持最佳性能。
"""

import json
import sys
import os
import argparse
import math
import warnings
# 明确禁用垃圾回收的自动清理，使用系统默认行为
# 不导入 gc 模块，不调用任何内存清理函数
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np

# 抑制 scikit-learn 版本不匹配警告（不影响主要功能）
warnings.filterwarnings('ignore', message='.*InconsistentVersionWarning.*')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.python.data_fetcher import SYMBOLS, TIMEFRAMES, load_ohlcv
from src.python.feature_engineering import build_features, get_feature_columns, get_primary_feature_columns, add_multi_timeframe_features
from src.python.predictor import load_predictor, apply_calibration, _latest_model_dir, _find_all_timeframe_models, _find_symbol_timeframe_model

# ============================================================
# Polymarket 规则：
# - 以 50c (0.5 USDC) 买入一份合约
# - 赢了：合约价值 1 USDC，投入翻倍（净赚 100%）
# - 输了：合约价值 0 USDC，投入归零（净亏 100%）
# - 最小下注：1 USDC
# - 每次下注：当前资金 x 下注比例（动态变化）
# - 回测中资金 < 1 USDC 时不再继续后续天数（无法满足最小下注）
# ============================================================
# Polymarket：买 Yes at 价格 P，赢则每股赎回 $1（payout=本金/P），输则 $0
WIN_VALUE = 1.0    # 赢了合约价值 1 USDC/股
LOSE_VALUE = 0.0   # 输了合约价值 0
MIN_BET = 1.0      # 最小下注 1 USDC；资金 < MIN_BET 时停止该组回测
# 回测默认：6 万初始、固定每笔 3000、0.54 价格；有动态=回撤>20% 用 1500，>10% 用 2250
DEFAULT_ORDER_PRICE = 0.5
DEFAULT_INITIAL_CAPITAL = 60000.0
FIXED_BET_AMOUNT = 3000.0

# 现实交易参数
DEFAULT_FEE_RATE = 0.001      # 手续费 0.1%（买入+卖出共约 0.2%）
DEFAULT_SLIPPAGE = 0.001      # 滑点 0.1%
DEFAULT_MAX_TRADES_PER_DAY = 20  # 每日最多交易次数（Polymarket 一般没那么多市场）


def load_test_data(
    symbol: str,
    timeframe: str,
    test_days: int = 90,
    train_cutoff_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    加载回测数据（样本外数据）。
    
    参数:
        symbol: 交易对
        timeframe: 时间周期
        test_days: 回测天数（无 train_cutoff/end_date 时用最近 test_days 天）
        train_cutoff_date: 训练数据截止日期（ISO格式），回测只用此日期之后的数据
        end_date: 回测结束日（YYYY-MM-DD）；与 train_cutoff_date 同时指定时取 (train_cutoff_date, end_date]，用于固定 365 天窗口
    
    重要：回测数据必须是模型没见过的数据（样本外），否则会有数据泄露！
    """
    df = load_ohlcv(symbol, timeframe)
    if df.empty:
        return df
    
    # 确保有日期列
    if "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    
    now_utc = pd.Timestamp.now(tz="UTC")
    cutoff_recent = now_utc - pd.Timedelta(days=test_days)

    if train_cutoff_date:
        train_cutoff = pd.to_datetime(train_cutoff_date, utc=True)
        df = df[df["date"] > train_cutoff].copy()
        if end_date:
            end_ts = pd.to_datetime(end_date, utc=True)
            df = df[df["date"] <= end_ts].copy()
        else:
            df = df[df["date"] >= cutoff_recent].copy()
    else:
        df = df[df["date"] >= cutoff_recent].copy()

    return df.reset_index(drop=True)


def _run_trading_loop_on_df(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    initial_capital: float,
    order_price: Optional[float],
    prob_threshold: float = 0.55,
    prob_threshold_up: Optional[float] = None,
    prob_threshold_down: Optional[float] = None,
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage: float = DEFAULT_SLIPPAGE,
    max_trades_per_day: int = DEFAULT_MAX_TRADES_PER_DAY,
    use_fixed_bet: bool = True,
    use_smart_bet: bool = False,
    smart_bet_capital_threshold: float = 60000.0,
    bet_ratio: float = 0.05,
    enable_dynamic_bet_ratio: bool = False,
    enable_high_vol_filter: bool = False,
    high_vol_params: Optional[Dict[str, Any]] = None,
    stop_loss_pct: Optional[float] = None,
    consecutive_loss_limit: Optional[int] = None,
    risk_pause_k_bars: Optional[int] = None,
    safe_print_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """对已含 pred_prob、pred_up、actual_up、date、high_vol_skip 的 df 跑交易循环，返回与 run_backtest_for_pair 相同结构。"""
    _print = safe_print_fn if safe_print_fn is not None else (lambda msg, end="\n": None)
    order_price_this = order_price if order_price is not None else DEFAULT_ORDER_PRICE
    capital = initial_capital
    equity_curve = [capital]
    trade_history = []
    trade_outcomes = []
    wins = 0
    losses = 0
    peak_capital = capital
    max_drawdown = 0.0
    trading_paused = False
    pause_reason = None
    consecutive_losses = 0
    paused_k_bars_remaining = 0
    original_bet_ratio = bet_ratio
    current_date = None
    trades_today = 0
    total_fees = 0.0
    use_dual = prob_threshold_up is not None and prob_threshold_down is not None

    for idx, row in df.iterrows():
        if capital < MIN_BET:
            break
        if capital > peak_capital:
            peak_capital = capital
        if paused_k_bars_remaining > 0:
            paused_k_bars_remaining -= 1
            if paused_k_bars_remaining <= 0:
                trading_paused = False
                pause_reason = None
                consecutive_losses = 0
            else:
                equity_curve.append(capital)
                continue
        if trading_paused:
            equity_curve.append(capital)
            continue
        if stop_loss_pct is not None:
            drawdown_from_peak = (peak_capital - capital) / peak_capital if peak_capital > 0 else 0
            if drawdown_from_peak >= stop_loss_pct:
                trading_paused = True
                pause_reason = f"从峰值回撤 {drawdown_from_peak*100:.1f}% 超过止损"
                equity_curve.append(capital)
                continue
        if consecutive_loss_limit is not None and consecutive_losses >= consecutive_loss_limit:
            pause_k_bars = risk_pause_k_bars or 8
            paused_k_bars_remaining = pause_k_bars
            trading_paused = True
            pause_reason = f"连续亏损 {consecutive_losses} 次"
            equity_curve.append(capital)
            continue

        prob = row["pred_prob"]
        pred = row["pred_up"]
        actual = row["actual_up"]
        row_date = row.get("date")
        if row_date is not None:
            trade_date = pd.to_datetime(row_date).date() if hasattr(row_date, "date") else pd.to_datetime(row_date).date()
            if current_date != trade_date:
                current_date = trade_date
                trades_today = 0
        if max_trades_per_day > 0 and trades_today >= max_trades_per_day:
            continue
        effective_prob = prob if pred == 1 else (1 - prob)
        if use_dual:
            if pred == 1 and prob < (prob_threshold_up or 0):
                continue
            if pred == 0 and prob >= (prob_threshold_down or 1):
                continue
        else:
            if effective_prob < prob_threshold:
                continue
        if enable_high_vol_filter and row.get("high_vol_skip"):
            scale = (high_vol_params or {}).get("high_vol_bet_scale")
            if scale is None or scale <= 0:
                equity_curve.append(capital)
                continue
        if capital < MIN_BET:
            break
        if use_smart_bet:
            if capital > smart_bet_capital_threshold:
                if enable_dynamic_bet_ratio and peak_capital > 0:
                    dd = (peak_capital - capital) / peak_capital
                    bet_amount = 1500.0 if dd > 0.2 else (2250.0 if dd > 0.1 else FIXED_BET_AMOUNT)
                else:
                    bet_amount = FIXED_BET_AMOUNT
            else:
                effective_ratio = bet_ratio
                if enable_dynamic_bet_ratio and peak_capital > 0:
                    dd = (peak_capital - capital) / peak_capital
                    effective_ratio = bet_ratio * (0.5 if dd > 0.2 else 0.75 if dd > 0.1 else 1.0)
                bet_amount = max(MIN_BET, min(capital * effective_ratio, capital - 1.0))
        elif use_fixed_bet:
            if enable_dynamic_bet_ratio and peak_capital > 0:
                dd = (peak_capital - capital) / peak_capital
                bet_amount = 1500.0 if dd > 0.2 else (2250.0 if dd > 0.1 else FIXED_BET_AMOUNT)
            else:
                bet_amount = FIXED_BET_AMOUNT
        else:
            if enable_dynamic_bet_ratio and peak_capital > 0:
                dd = (peak_capital - capital) / peak_capital
                dynamic_bet_ratio = bet_ratio * (0.5 if dd > 0.2 else 0.75 if dd > 0.1 else 1.0)
                bet_amount = capital * dynamic_bet_ratio
            else:
                bet_amount = capital * bet_ratio
            if bet_amount < MIN_BET:
                bet_amount = MIN_BET if capital >= MIN_BET else 0
        if bet_amount <= 0 or capital < bet_amount:
            break
        if enable_high_vol_filter and row.get("high_vol_skip"):
            scale = (high_vol_params or {}).get("high_vol_bet_scale")
            if scale is not None and scale > 0:
                bet_amount = bet_amount * scale
                if bet_amount < MIN_BET:
                    equity_curve.append(capital)
                    continue
        fee = bet_amount * (fee_rate + slippage)
        total_fees += fee
        capital -= (bet_amount + fee)
        correct = (pred == actual)
        if correct:
            consecutive_losses = 0
            capital += bet_amount / order_price_this
            pnl = bet_amount / order_price_this - bet_amount - fee
            wins += 1
        else:
            consecutive_losses += 1
            pnl = -bet_amount - fee
            losses += 1
        capital = max(capital, 0)
        trades_today += 1
        equity_curve.append(capital)
        trade_outcomes.append(correct)
        trade_history.append({
            "date": str(row.get("date", idx)),
            "pred": "UP" if pred == 1 else "DOWN",
            "actual": "UP" if actual == 1 else "DOWN",
            "prob": round(effective_prob, 4),
            "correct": correct,
            "bet": round(bet_amount, 2),
            "contracts": round(bet_amount / order_price_this, 2),
            "fee": round(fee, 4),
            "pnl": round(pnl, 2),
            "capital": round(capital, 2),
        })
        if peak_capital > 0:
            drawdown = (peak_capital - capital) / peak_capital
            max_drawdown = max(max_drawdown, max(0.0, min(1.0, drawdown)))
        if capital < MIN_BET:
            break

    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    profit_pct = (capital - initial_capital) / initial_capital * 100
    min_capital = min(equity_curve) if equity_curve else initial_capital
    min_over_initial_pct = (min_capital / initial_capital * 100) if initial_capital > 0 else 100.0
    min_over_initial_pct = max(0.0, min(1000.0, min_over_initial_pct))
    max_drawdown_pct = round(max(0, min(1, max_drawdown)) * 100, 2)
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "final_capital": round(capital, 2),
        "profit_pct": round(profit_pct, 2),
        "max_drawdown": max_drawdown_pct,
        "min_capital": round(min_capital, 2),
        "min_over_initial_pct": round(min_over_initial_pct, 2),
        "total_fees": round(total_fees, 2),
        "equity_curve": equity_curve,
        "trade_history": trade_history,
        "trading_paused": trading_paused,
        "pause_reason": pause_reason,
        "risk_controls": None,
    }


def run_backtest_for_pair(
    symbol: str,
    timeframe: str,
    model,
    feature_names: List[str],
    initial_capital: float = 400.0,
    bet_ratio: float = 0.05,
    test_days: int = 90,
    prob_threshold: float = 0.55,

    prob_threshold_up: Optional[float] = None,
    prob_threshold_down: Optional[float] = None,
    train_cutoff_date: Optional[str] = None,
    end_date: Optional[str] = None,
    lookahead: int = 1,
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage: float = DEFAULT_SLIPPAGE,
    max_trades_per_day: int = DEFAULT_MAX_TRADES_PER_DAY,
    calibrator: Optional[object] = None,
    calibration_method: Optional[str] = None,
    stop_loss_pct: Optional[float] = None,
    consecutive_loss_limit: Optional[int] = None,
    risk_pause_k_bars: Optional[int] = None,
    # 策略层过滤参数（新模型优化 ④）
    require_ema_trend: bool = False,
    min_atr_pct: Optional[float] = None,
    min_volume_ratio: Optional[float] = None,
    # 动态调整下注比例（根据回撤）
    enable_dynamic_bet_ratio: bool = False,
    # 若传入则用该价格作为下单价格（如 0.54）；否则按阈值
    order_price: Optional[float] = None,
    # True=6万+固定3000（有动态时1500/2250）；False=400+5%
    use_fixed_bet: bool = True,
    # 贴近实盘：资金>threshold 时固定 3000，否则按总资金×bet_ratio（如 5%）；与 use_fixed_bet 二选一
    use_smart_bet: bool = False,
    smart_bet_capital_threshold: float = 60000.0,
    # 高波动过滤：与 backtest_gru_regime 一致，跳过 atr/vol 高波 bar（仅回测对比用，不改变实盘）
    enable_high_vol_filter: bool = False,
    high_vol_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    对单个 symbol+timeframe 运行回测。
    
    返回:
        {
            "symbol": str,
            "timeframe": str,
            "total_trades": int,
            "wins": int,
            "losses": int,
            "win_rate": float,
            "final_capital": float,
            "profit_pct": float,
            "max_drawdown": float,
            "equity_curve": List[float],
            "trade_history": List[dict],
        }
    """
    # 加载回测数据（样本外数据）
    import sys
    import time
    import os
    
    # 使用锁确保输出不混乱（多进程时）
    try:
        from threading import Lock
        _output_lock = Lock()
    except:
        _output_lock = None
    
    def safe_print(msg, end="\n"):
        """线程安全的打印"""
        if _output_lock:
            with _output_lock:
                sys.stdout.write(f"[{symbol} {timeframe}] {msg}{end}")
                sys.stdout.flush()
        else:
            sys.stdout.write(f"[{symbol} {timeframe}] {msg}{end}")
            sys.stdout.flush()
    
    safe_print("加载数据...", end=" ")
    df = load_test_data(symbol, timeframe, test_days, train_cutoff_date=train_cutoff_date, end_date=end_date)
    if df.empty or len(df) < 100:
        safe_print("❌ 数据不足")
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "error": "数据不足",
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "final_capital": initial_capital,
            "profit_pct": 0.0,
            "max_drawdown": 0.0,
            "min_capital": initial_capital,
            "min_over_initial_pct": 100.0,
            "equity_curve": [initial_capital],
            "trade_history": [],
        }
    safe_print(f"✅ ({len(df)} 根K线)", end=" ")
    
    # 计算特征（如果使用新特征集 v4，特征数量多，可能较慢）
    safe_print(f"计算特征...", end=" ")
    start_time = time.time()
    df = build_features(df)
    elapsed = time.time() - start_time
    safe_print(f"✅ ({elapsed:.1f}秒)", end=" ")
    
    # 若模型含 mtf_*（15m 训练时用了 --add-multi-timeframe），须补上 1h/4h 特征，否则特征数与训练不一致会报错
    if timeframe == "15m" and feature_names and any((str(f) or "").startswith("mtf_") for f in feature_names):
        safe_print("加载多时间框架特征（1h/4h）...", end=" ")
        mtf_start = time.time()
        df = add_multi_timeframe_features(df, symbol)
        mtf_elapsed = time.time() - mtf_start
        safe_print(f"✅ ({mtf_elapsed:.1f}秒)", end=" ")
    
    if enable_high_vol_filter:
        from scripts.backtest_gru_regime import add_high_vol_skip_column
        if high_vol_params:
            skip_params = {k: v for k, v in high_vol_params.items() if k in ("atr_n_days", "atr_quantile", "vol_ratio_threshold", "combine")}
            add_high_vol_skip_column(df, **skip_params)
        else:
            add_high_vol_skip_column(df)
    else:
        df["high_vol_skip"] = False
    
    safe_print("")  # 换行
    
    # 确保特征列存在且顺序与模型一致
    cols = [c for c in feature_names if c in df.columns]
    if not cols:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "error": "特征列不匹配",
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "final_capital": initial_capital,
            "profit_pct": 0.0,
            "max_drawdown": 0.0,
            "min_capital": initial_capital,
            "min_over_initial_pct": 100.0,
            "equity_curve": [initial_capital],
            "trade_history": [],
        }
    if len(cols) != len(feature_names):
        missing = [f for f in feature_names if f not in df.columns]
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "error": f"特征数不一致: 模型需 {len(feature_names)} 个，仅找到 {len(cols)} 个。若为 15m 且含 mtf_*，请确保 data/raw 存在 {symbol.replace('/', '_')}_1h.parquet 和 _4h.parquet。缺失示例: {missing[:5]}{'...' if len(missing) > 5 else ''}",
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "final_capital": initial_capital,
            "profit_pct": 0.0,
            "max_drawdown": 0.0,
            "min_capital": initial_capital,
            "min_over_initial_pct": 100.0,
            "equity_curve": [initial_capital],
            "trade_history": [],
        }
    
    # 创建标签（未来 N 根 K 线涨跌，与训练时 lookahead 一致）
    df["future_return"] = df["close"].shift(-lookahead) / df["close"] - 1
    df["actual_up"] = (df["future_return"] > 0).astype(int)
    
    # 去除 NaN（排除最后 lookahead 行，没有未来数据）
    df = df.dropna(subset=cols + ["future_return", "actual_up"])
    df = df.iloc[:-lookahead] if lookahead > 0 else df
    
    if len(df) < 50:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "error": "有效数据不足",
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "final_capital": initial_capital,
            "profit_pct": 0.0,
            "max_drawdown": 0.0,
            "min_capital": initial_capital,
            "min_over_initial_pct": 100.0,
            "equity_curve": [initial_capital],
            "trade_history": [],
        }
    
    # 预测：双阈值逻辑需要真实概率，必须用 predict_proba；predict() 返回 0/1 类别会导致 20-80 等双阈值永远不触发
    X = df[cols]
    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(X)[:, 1]).ravel()
    else:
        probs = np.asarray(model.predict(X)).ravel()
    if calibrator is not None and calibration_method:
        probs = apply_calibration(calibrator, calibration_method, probs)
    df["pred_prob"] = probs
    df["pred_up"] = (probs >= 0.5).astype(int)
    
    # 模拟交易（符合 Polymarket 规则，含手续费和每日限制）
    safe_print(f"开始回测循环（{len(df)} 根K线）...")
    capital = initial_capital
    equity_curve = [capital]
    trade_history = []
    trade_outcomes = []  # 只记录 correct，第二遍根据当前资金重算 bet_amount
    wins = 0
    losses = 0
    peak_capital = capital
    max_drawdown = 0.0
    
    # 风险控制参数
    # stop_loss_pct: 止损百分比（如 0.3 表示从峰值回撤 30% 后停止交易，不自动恢复）
    # consecutive_loss_limit: 连续亏损次数限制（如 5 表示连续亏损 5 次后触发暂停）
    # risk_pause_k_bars: 连续亏损触发后，停止交易 N 根 K 线（如 8 表示停止 8 根 K 线后自动恢复）
    
    # 风险控制状态
    trading_paused = False
    pause_reason = None
    consecutive_losses = 0
    paused_k_bars_remaining = 0  # 剩余暂停的 K 线数（连续亏损触发后使用）
    # 动态调整下注比例相关
    original_bet_ratio = bet_ratio  # 保存原始下注比例，用于动态调整恢复
    
    # 每日交易计数
    current_date = None
    trades_today = 0
    total_fees = 0.0
    
    for idx, row in df.iterrows():
        # 资金 < 1 USDC 不再继续后续天数（无法满足最小下注 1 USDC）
        if capital < MIN_BET:
            break

        # 更新峰值（在风险控制检查之前）
        if capital > peak_capital:
            peak_capital = capital

        # ============================================================
        # 风险控制检查
        # ============================================================
        # 如果正在暂停中（K 线计数），每根 K 线减 1
        if paused_k_bars_remaining > 0:
            paused_k_bars_remaining -= 1
            if paused_k_bars_remaining <= 0:
                # 暂停时间到，恢复交易
                trading_paused = False
                pause_reason = None
                consecutive_losses = 0  # 重置连续亏损计数
            else:
                # 仍在暂停中，跳过交易
                equity_curve.append(capital)
                continue
        
        # 如果已经暂停（但 paused_k_bars_remaining 为 0，可能是止损触发的），跳过交易
        if trading_paused:
            equity_curve.append(capital)
            continue
        
        # 1. 止损检查（从峰值回撤）- 触发后停止交易，不自动恢复
        if stop_loss_pct is not None:
            drawdown_from_peak = (peak_capital - capital) / peak_capital if peak_capital > 0 else 0
            if drawdown_from_peak >= stop_loss_pct:
                trading_paused = True
                pause_reason = f"从峰值回撤 {drawdown_from_peak*100:.1f}% 超过止损 {stop_loss_pct*100:.1f}%，停止交易"
                equity_curve.append(capital)
                continue
        
        # 2. 连续亏损限制 - 触发后停止交易 N 根 K 线，然后自动恢复
        if consecutive_loss_limit is not None and consecutive_losses >= consecutive_loss_limit:
            pause_k_bars = risk_pause_k_bars or 8
            paused_k_bars_remaining = pause_k_bars
            trading_paused = True
            pause_reason = f"连续亏损 {consecutive_losses} 次，超过限制 {consecutive_loss_limit}，停止交易 {pause_k_bars} 根 K 线"
            equity_curve.append(capital)
            continue

        prob = row["pred_prob"]
        pred = row["pred_up"]
        actual = row["actual_up"]
        
        # 获取当前日期（限制每日交易次数）
        row_date = row.get("date")
        if row_date is not None:
            if hasattr(row_date, 'date'):
                trade_date = row_date.date()
            else:
                trade_date = pd.to_datetime(row_date).date()
            
            if current_date != trade_date:
                current_date = trade_date
                trades_today = 0
        
        # 每日交易次数限制
        if max_trades_per_day > 0 and trades_today >= max_trades_per_day:
            continue
        
        # 有效概率 = 预测方向的概率（与训练时一致，见 docs/概率与阈值逻辑.md）
        effective_prob = prob if pred == 1 else (1 - prob)

        # 阈值过滤：单/双阈值与训练时一致（模型输出 P(UP)，UP 时 prob≥up、DOWN 时 prob<down）
        use_dual = prob_threshold_up is not None and prob_threshold_down is not None
        if use_dual:
            # 双阈值：UP 仅当 P(UP) >= up；DOWN 仅当 P(UP) < down（与训练时 P(UP) 语义一致）
            if pred == 1:
                if prob < prob_threshold_up:
                    continue
            else:
                if prob >= prob_threshold_down:
                    continue
        else:
            if effective_prob < prob_threshold:
                continue
        
        # 高波动过滤：scale 为 None 或 0 时该 bar 跳过；否则后续按 scale 缩小下注（少交易，与 backtest_gru_regime 一致）
        if enable_high_vol_filter and row.get("high_vol_skip"):
            high_vol_bet_scale = (high_vol_params or {}).get("high_vol_bet_scale")
            if high_vol_bet_scale is None or high_vol_bet_scale <= 0:
                equity_curve.append(capital)
                continue
        
        # ============================================================
        # 策略层过滤（新模型优化 ④：EMA/ATR/Volume）
        # ============================================================
        if require_ema_trend or min_atr_pct is not None or min_volume_ratio is not None:
            # EMA 趋势确认：UP 需 ema_10 > ema_20，DOWN 需 ema_10 < ema_20
            # 注意：特征工程中使用的是 ema_10 和 ema_20，不是 ema_8 和 ema_21
            if require_ema_trend:
                ema_10 = row.get("ema_10")
                ema_20 = row.get("ema_20")
                if ema_10 is not None and ema_20 is not None:
                    if pred == 1 and ema_10 <= ema_20:
                        continue  # UP 但 EMA 趋势不符
                    elif pred == 0 and ema_10 >= ema_20:
                        continue  # DOWN 但 EMA 趋势不符
            
            # ATR 过滤：仅当 ATR_pct_14 >= min_atr_pct 才交易
            if min_atr_pct is not None:
                atr_pct_14 = row.get("atr_pct_14")
                if atr_pct_14 is not None and atr_pct_14 < min_atr_pct:
                    continue  # ATR 过低，跳过交易
            
            # 成交量过滤：仅当 volume_ratio >= min_volume_ratio 才交易
            if min_volume_ratio is not None:
                volume_ratio = row.get("volume_ratio")
                if volume_ratio is not None and volume_ratio < min_volume_ratio:
                    continue  # 成交量不足，跳过交易
        
        # ============================================================
        # 下注：贴近实盘(use_smart_bet) 初始400、>6万固定3000/≤6万5%；有动态时按回撤减仓与资金逻辑一致
        # ============================================================
        if capital < MIN_BET:
            break
        if use_smart_bet:
            if capital > smart_bet_capital_threshold:
                # >6万：无动态固定3000；有动态按回撤 3000/2250/1500
                if enable_dynamic_bet_ratio and peak_capital > 0:
                    current_drawdown = (peak_capital - capital) / peak_capital
                    if current_drawdown > 0.2:
                        bet_amount = 1500.0
                    elif current_drawdown > 0.1:
                        bet_amount = 2250.0
                    else:
                        bet_amount = FIXED_BET_AMOUNT
                else:
                    bet_amount = FIXED_BET_AMOUNT
            else:
                # ≤6万：按当前总资金5%；有动态时按回撤缩小比例（0.5/0.75）
                effective_ratio = bet_ratio
                if enable_dynamic_bet_ratio and peak_capital > 0:
                    current_drawdown = (peak_capital - capital) / peak_capital
                    if current_drawdown > 0.2:
                        effective_ratio = bet_ratio * 0.5
                    elif current_drawdown > 0.1:
                        effective_ratio = bet_ratio * 0.75
                bet_amount = max(MIN_BET, min(capital * effective_ratio, capital - 1.0))
            if capital < bet_amount:
                break
        elif use_fixed_bet:
            if enable_dynamic_bet_ratio and peak_capital > 0:
                current_drawdown = (peak_capital - capital) / peak_capital
                if current_drawdown > 0.2:
                    bet_amount = 1500.0
                elif current_drawdown > 0.1:
                    bet_amount = 2250.0
                else:
                    bet_amount = FIXED_BET_AMOUNT
            else:
                bet_amount = FIXED_BET_AMOUNT
            if capital < bet_amount:
                break
        else:
            if enable_dynamic_bet_ratio and peak_capital > 0:
                current_drawdown = (peak_capital - capital) / peak_capital
                dynamic_bet_ratio = original_bet_ratio * (0.5 if current_drawdown > 0.2 else 0.75 if current_drawdown > 0.1 else 1.0)
                bet_amount = capital * dynamic_bet_ratio
            else:
                bet_amount = capital * bet_ratio
            if bet_amount < MIN_BET:
                if capital >= MIN_BET:
                    bet_amount = MIN_BET
                else:
                    break
        order_price_this = order_price if order_price is not None else DEFAULT_ORDER_PRICE

        # 高波少交易：高波 bar 下注额乘以 scale，不足 MIN_BET 则跳过
        if enable_high_vol_filter and row.get("high_vol_skip"):
            high_vol_bet_scale = (high_vol_params or {}).get("high_vol_bet_scale")
            if high_vol_bet_scale is not None and high_vol_bet_scale > 0:
                bet_amount = bet_amount * high_vol_bet_scale
                if bet_amount < MIN_BET:
                    equity_curve.append(capital)
                    continue
        
        # ============================================================
        # Polymarket 规则：买 Yes at 价格 P，赢则每股赎回 $1，输则 $0
        # - 合约数 = bet_amount / P
        # - 买入时：扣除 bet_amount + fee
        # - 赢：payout = bet_amount / P（每股 $1），净赚 = bet_amount/P - bet_amount - fee
        # - 输：payout = 0，净亏 = -bet_amount - fee
        # ============================================================
        contracts = bet_amount / order_price_this  # 能买的合约数
        
        # 计算手续费和滑点（买入时扣除）
        fee = bet_amount * (fee_rate + slippage)
        total_fees += fee
        
        # 买入时先扣除投入和手续费
        capital -= (bet_amount + fee)
        
        correct = (pred == actual)
        
        # 更新连续亏损计数
        if correct:
            consecutive_losses = 0  # 重置连续亏损计数
        else:
            consecutive_losses += 1
        
        if correct:
            # 赢：每股赎回 $1，得到 bet_amount / order_price_this
            capital += bet_amount / order_price_this
            pnl = bet_amount / order_price_this - bet_amount - fee  # 净赚
            wins += 1
        else:
            # 输：合约价值 0
            pnl = -bet_amount - fee  # 净亏
            losses += 1
        
        capital = max(capital, 0)  # 不能为负
        trades_today += 1
        equity_curve.append(capital)
        
        # 记录交易（并保存用于第二遍按胜率价格重算）
        trade_outcomes.append(correct)
        trade_history.append({
            "date": str(row.get("date", idx)),
            "pred": "UP" if pred == 1 else "DOWN",
            "actual": "UP" if actual == 1 else "DOWN",
            "prob": round(effective_prob, 4),
            "correct": correct,
            "bet": round(bet_amount, 2),
            "contracts": round(contracts, 2),
            "fee": round(fee, 4),
            "pnl": round(pnl, 2),
            "capital": round(capital, 2),
        })
        
        # 更新最大回撤（峰值已在循环开始处更新）
        # drawdown计算公式: (peak_capital - capital) / peak_capital
        # 如果capital > peak_capital（不应该发生，但可能由于浮点误差），drawdown会是负数，应该限制为0
        # 如果capital < 0（不应该发生，因为第446行限制了capital >= 0），drawdown可能>1，应该限制为1
        if peak_capital > 0:
            # 计算回撤比例（0-1之间）
            drawdown = (peak_capital - capital) / peak_capital
            # 确保drawdown在0-1之间：
            # - 如果capital > peak_capital，drawdown为负，限制为0（没有回撤）
            # - 如果capital < 0（理论上不应该发生），drawdown > 1，限制为1（100%回撤）
            drawdown = max(0.0, min(1.0, drawdown))
        else:
            # peak_capital为0，无法计算回撤，设为0
            drawdown = 0.0
        max_drawdown = max(max_drawdown, drawdown)
        
        # 破产检查：资金 < 1 USDC 则不再继续后续天数
        if capital < MIN_BET:
            break
    
    total_trades = wins + losses
    win_rate = wins / total_trades if total_trades > 0 else 0.0
    # 简化版：固定 order_price=0.54，无需第二遍重算
    profit_pct = (capital - initial_capital) / initial_capital * 100
    min_capital = min(equity_curve) if equity_curve else initial_capital
    
    # 计算最低/初始%
    # 公式: (min_capital / initial_capital) * 100
    # 如果min_capital为负（不应该发生，因为capital被限制为>=0），结果会是负数
    # 如果min_capital异常大（比如资金增长很多倍），结果会很大
    if initial_capital > 0:
        min_over_initial_pct = (min_capital / initial_capital) * 100
        # 限制在合理范围：
        # - 最小0%（如果min_capital为负，限制为0%）
        # - 最大1000%（允许资金增长到10倍，超过这个值可能是计算错误）
        min_over_initial_pct = max(0.0, min(1000.0, min_over_initial_pct))
    else:
        # initial_capital为0，无法计算，设为100%（表示没有变化）
        min_over_initial_pct = 100.0
    
    # 确保max_drawdown在0-1之间（比例），然后转换为百分比
    max_drawdown_pct = round(max(0, min(1, max_drawdown)) * 100, 2)

    result = {
        "symbol": symbol,
        "timeframe": timeframe,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "final_capital": round(capital, 2),
        "profit_pct": round(profit_pct, 2),
        "max_drawdown": max_drawdown_pct,  # 已经是百分比（0-100）
        "min_capital": round(min_capital, 2),
        "min_over_initial_pct": round(min_over_initial_pct, 2),  # 已经是百分比（0-1000）
        "total_fees": round(total_fees, 2),
        "equity_curve": equity_curve,
        "trade_history": trade_history,
        "trading_paused": trading_paused,
        "pause_reason": pause_reason,
        "risk_controls": {
            "stop_loss_pct": stop_loss_pct,
            "consecutive_loss_limit": consecutive_loss_limit,
            "risk_pause_k_bars": risk_pause_k_bars,
        } if any([stop_loss_pct, consecutive_loss_limit, risk_pause_k_bars]) else None,
    }
    # 双阈值且 0 笔交易时返回 pred_prob 范围，便于诊断（如 XRP 20-80 未达 0.8/0.2）
    if total_trades == 0 and use_dual and "pred_prob" in df.columns:
        result["prob_min"] = float(df["pred_prob"].min())
        result["prob_max"] = float(df["pred_prob"].max())
    return result


def run_backtest_for_pair_v5(
    symbol: str,
    timeframe: str,
    model_dir: Path,
    after_date: str,
    end_date: str,
    test_days: int = 90,
    initial_capital: float = 400.0,
    bet_ratio: float = 0.05,
    prob_threshold: float = 0.55,
    lookahead: int = 1,
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage: float = DEFAULT_SLIPPAGE,
    max_trades_per_day: int = DEFAULT_MAX_TRADES_PER_DAY,
    order_price: Optional[float] = None,
    use_fixed_bet: bool = True,
    use_smart_bet: bool = False,
    smart_bet_capital_threshold: float = 60000.0,
    enable_dynamic_bet_ratio: bool = False,
    enable_high_vol_filter: bool = False,
    high_vol_params: Optional[Dict[str, Any]] = None,
    stop_loss_pct: Optional[float] = None,
    consecutive_loss_limit: Optional[int] = None,
    risk_pause_k_bars: Optional[int] = None,
    safe_print_fn: Optional[Any] = None,
    v5_prediction_cache: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, Any]:
    """v5 回测：用 V5Predictor 历史预测得到 pred_prob/pred_up，再走同一套交易循环（_run_trading_loop_on_df）。
    与 run_backtest_for_pair 返回结构一致。"""
    model_dir = Path(model_dir)
    if not model_dir.exists():
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "error": "v5 model_dir 不存在",
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "final_capital": initial_capital,
            "profit_pct": 0.0,
            "max_drawdown": 0.0,
            "min_capital": initial_capital,
            "min_over_initial_pct": 100.0,
            "equity_curve": [initial_capital],
            "trade_history": [],
        }

    df = load_test_data(symbol, timeframe, test_days, train_cutoff_date=after_date, end_date=end_date)
    if df.empty or len(df) < 100:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "error": "数据不足",
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "final_capital": initial_capital,
            "profit_pct": 0.0,
            "max_drawdown": 0.0,
            "min_capital": initial_capital,
            "min_over_initial_pct": 100.0,
            "equity_curve": [initial_capital],
            "trade_history": [],
        }

    df = build_features(df)
    if "date" not in df.columns and "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    asset = symbol.replace("/", "_")
    start_date = df["date"].min().strftime("%Y-%m-%d") if hasattr(df["date"].min(), "strftime") else str(df["date"].min())[:10]
    end_date_use = df["date"].max().strftime("%Y-%m-%d") if hasattr(df["date"].max(), "strftime") else str(df["date"].max())[:10]

    cache_key = (str(model_dir), symbol, start_date, end_date_use) if v5_prediction_cache is not None else None
    if cache_key is not None and v5_prediction_cache is not None and cache_key in v5_prediction_cache:
        v5_df = v5_prediction_cache[cache_key].copy()
    else:
        try:
            from scripts.prediction_writer_v5 import V5Predictor
            predictor = V5Predictor(model_dir)
            v5_df = predictor.predict_historical(asset, start_date, end_date_use)
        except Exception as e:
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "error": "v5 历史预测失败: " + str(e),
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "final_capital": initial_capital,
                "profit_pct": 0.0,
                "max_drawdown": 0.0,
                "min_capital": initial_capital,
                "min_over_initial_pct": 100.0,
                "equity_curve": [initial_capital],
                "trade_history": [],
            }
        if cache_key is not None and v5_prediction_cache is not None:
            v5_prediction_cache[cache_key] = v5_df.copy()

    if v5_df.empty or "pred_prob" not in v5_df.columns:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "error": "v5 历史预测无结果",
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "final_capital": initial_capital,
            "profit_pct": 0.0,
            "max_drawdown": 0.0,
            "min_capital": initial_capital,
            "min_over_initial_pct": 100.0,
            "equity_curve": [initial_capital],
            "trade_history": [],
        }

    v5_df = v5_df[["timestamp", "pred_prob", "pred_up"]].copy()
    # 统一为 datetime64[ns, UTC] 再 merge，避免 datetime64[ms, UTC] 与 datetime64[ns] 合并报错
    df["date"] = pd.to_datetime(df["date"], utc=True)
    v5_df["timestamp"] = pd.to_datetime(v5_df["timestamp"], utc=True)
    df = df.merge(v5_df, left_on="date", right_on="timestamp", how="inner")
    # 合并后必须按时间排序，否则 shift(-lookahead) 会错位，actual_up 与 pred 不对应，胜率失真
    df = df.sort_values("date").reset_index(drop=True)
    # 去掉 v5 的 timestamp 列（若左侧有 timestamp 会变成 timestamp_x/timestamp_y，删右则）
    for c in ["timestamp", "timestamp_y"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    df["future_return"] = df["close"].shift(-lookahead) / df["close"] - 1
    df["actual_up"] = (df["future_return"] > 0).astype(int)
    df = df.dropna(subset=["future_return", "actual_up"])
    if lookahead > 0:
        df = df.iloc[:-lookahead]

    if len(df) < 50:
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "error": "有效数据不足",
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "final_capital": initial_capital,
            "profit_pct": 0.0,
            "max_drawdown": 0.0,
            "min_capital": initial_capital,
            "min_over_initial_pct": 100.0,
            "equity_curve": [initial_capital],
            "trade_history": [],
        }

    if enable_high_vol_filter:
        from scripts.backtest_gru_regime import add_high_vol_skip_column
        if high_vol_params:
            skip_params = {k: v for k, v in high_vol_params.items() if k in ("atr_n_days", "atr_quantile", "vol_ratio_threshold", "combine")}
            add_high_vol_skip_column(df, **skip_params)
        else:
            add_high_vol_skip_column(df)
    else:
        df["high_vol_skip"] = False

    return _run_trading_loop_on_df(
        df,
        symbol,
        timeframe,
        initial_capital,
        order_price,
        prob_threshold=prob_threshold,
        fee_rate=fee_rate,
        slippage=slippage,
        max_trades_per_day=max_trades_per_day,
        use_fixed_bet=use_fixed_bet,
        use_smart_bet=use_smart_bet,
        smart_bet_capital_threshold=smart_bet_capital_threshold,
        bet_ratio=bet_ratio,
        enable_dynamic_bet_ratio=enable_dynamic_bet_ratio,
        enable_high_vol_filter=enable_high_vol_filter,
        high_vol_params=high_vol_params,
        stop_loss_pct=stop_loss_pct,
        consecutive_loss_limit=consecutive_loss_limit,
        risk_pause_k_bars=risk_pause_k_bars,
        safe_print_fn=safe_print_fn,
    )


def _run_single_backtest(
    args_tuple: Tuple,
) -> Dict[str, Any]:
    """
    单个 symbol+timeframe 回测的辅助函数（用于并行处理）。
    """
    (symbol, tf, initial_capital, bet_ratio, test_days, prob_threshold,
     prob_threshold_up, prob_threshold_down, primary_only,
     fee_rate, slippage, max_trades_per_day, models_dir_str, stop_loss_pct,
     consecutive_loss_limit, risk_pause_k_bars, tf_models_dict, model_path_str,
     require_ema_trend, min_atr_pct, min_volume_ratio, enable_dynamic_bet_ratio) = args_tuple
    
    # 转换回 Path 对象
    models_dir = Path(models_dir_str) if models_dir_str else None
    tf_models = {k: Path(v) for k, v in tf_models_dict.items()} if tf_models_dict else {}
    
    try:
        # ⚡ 性能优化：直接使用预先查找的模型路径，避免重复查找
        if model_path_str:
            model_path = Path(model_path_str)
        elif tf in tf_models:
            # 其次使用 timeframe 专用模型
            model_path = tf_models[tf]
        else:
            # 最后使用通用模型（需要查找，但这种情况应该很少）
            model_path = _latest_model_dir(timeframe=tf)
        
        # 加载模型
        model, feature_names, meta = load_predictor(model_path)
        
        # 从元数据获取训练截止日期和 lookahead（防止数据泄露）
        train_cutoff_date = meta.get("train_cutoff_date")
        lookahead = meta.get("lookahead", 1)  # 默认 1（只看下一根）
        
        # 如果只用主要特征，过滤
        if primary_only:
            feature_names = [f for f in feature_names if f.startswith(("kdj_", "rang_"))]
        
        result = run_backtest_for_pair(
            symbol, tf, model, feature_names,
            initial_capital=initial_capital,
            bet_ratio=bet_ratio,
            test_days=test_days,
            prob_threshold=prob_threshold,
            prob_threshold_up=prob_threshold_up,
            prob_threshold_down=prob_threshold_down,
            train_cutoff_date=train_cutoff_date,
            lookahead=lookahead,
            fee_rate=fee_rate,
            slippage=slippage,
            max_trades_per_day=max_trades_per_day,
            calibrator=meta.get("calibrator"),
            calibration_method=meta.get("calibration"),
            stop_loss_pct=stop_loss_pct,
            consecutive_loss_limit=consecutive_loss_limit,
            risk_pause_k_bars=risk_pause_k_bars,
            require_ema_trend=require_ema_trend,
            min_atr_pct=min_atr_pct,
            min_volume_ratio=min_volume_ratio,
            enable_dynamic_bet_ratio=enable_dynamic_bet_ratio,
        )
        return result
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        return {
            "symbol": symbol,
            "timeframe": tf,
            "error": error_msg,
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "final_capital": initial_capital,
            "profit_pct": 0.0,
            "max_drawdown": 0.0,
            "min_capital": initial_capital,
            "min_over_initial_pct": 100.0,
            "equity_curve": [initial_capital],
            "trade_history": [],
        }


def run_full_backtest(
    initial_capital: float = 400.0,
    bet_ratio: float = 0.05,
    test_days: int = 90,
    prob_threshold: float = 0.55,
    prob_threshold_up: Optional[float] = None,
    prob_threshold_down: Optional[float] = None,
    primary_only: bool = False,
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage: float = DEFAULT_SLIPPAGE,
    max_trades_per_day: int = DEFAULT_MAX_TRADES_PER_DAY,
    models_dir: Optional[Path] = None,
    stop_loss_pct: Optional[float] = None,
    consecutive_loss_limit: Optional[int] = None,
    risk_pause_k_bars: Optional[int] = None,
    n_jobs: int = 1,
    timeframes: Optional[List[str]] = None,
    # 策略层过滤参数（新模型优化 ④）
    require_ema_trend: bool = False,
    min_atr_pct: Optional[float] = None,
    min_volume_ratio: Optional[float] = None,
    # 只回测指定的币对（None 表示回测所有币对）
    symbols: Optional[List[str]] = None,
    # 动态调整下注比例（根据回撤）
    enable_dynamic_bet_ratio: bool = False,
) -> Dict[str, Any]:
    """
    对所有 symbol × timeframe 运行回测。
    
    参数:
        fee_rate: 手续费率（默认 0.1%）
        slippage: 滑点（默认 0.1%）
        max_trades_per_day: 每日最多交易次数（默认 3，0=不限制）
        models_dir: 模型根目录，默认 data/models；可指定 data/models_setA 等用于多套模型对比。
        n_jobs: 并行进程数，1=串行，-1=使用全部CPU核心，N=使用N个进程
        timeframes: 要回测的时间周期列表，默认使用TIMEFRAMES（所有周期）
    
    重要：回测自动使用模型元数据中的 train_cutoff_date，确保只用样本外数据！
    
    返回完整的回测结果。
    """
    # 使用指定的timeframes或默认的TIMEFRAMES
    test_timeframes = timeframes if timeframes is not None else TIMEFRAMES
    
    # 使用指定的symbols或默认的SYMBOLS
    test_symbols = symbols if symbols is not None else SYMBOLS
    
    results = []
    
    # ⚡ 性能优化：预先查找所有模型路径，避免每个任务都重复查找（96个目录查找很慢）
    print("🔍 预先查找所有模型路径（避免重复查找）...", end=" ", flush=True)
    model_paths_cache = {}  # {(symbol, tf): model_path_str}
    
    # 尝试加载按 timeframe 分别训练的模型（备用）
    tf_models = _find_all_timeframe_models(models_root=models_dir)
    # 转换为字典（可序列化，用于传递给子进程）
    tf_models_dict = {k: str(v) for k, v in tf_models.items()}
    
    # 预先查找所有 symbol+timeframe 组合的模型路径
    for symbol in test_symbols:
        for tf in test_timeframes:
            symbol_tf_model = _find_symbol_timeframe_model(symbol, tf, models_root=models_dir)
            if symbol_tf_model:
                model_paths_cache[(symbol, tf)] = str(symbol_tf_model)
            elif tf in tf_models:
                model_paths_cache[(symbol, tf)] = str(tf_models[tf])
            else:
                # 最后使用通用模型（需要查找）
                try:
                    model_paths_cache[(symbol, tf)] = str(_latest_model_dir(timeframe=tf))
                except:
                    model_paths_cache[(symbol, tf)] = None
    
    print(f"✅ 找到 {len([v for v in model_paths_cache.values() if v])} 个模型路径")
    
    # 用于只打印一次「样本外」说明
    _printed_cutoff = [False]
    
    # 统计信息
    total_combinations = len(test_symbols) * len(test_timeframes)
    
    # 准备任务列表（models_dir 需要转换为字符串以便序列化）
    models_dir_str = str(models_dir) if models_dir else None
    tasks = []
    for symbol in test_symbols:
        for tf in test_timeframes:
            model_path_str = model_paths_cache.get((symbol, tf))
            tasks.append((
                symbol, tf, initial_capital, bet_ratio, test_days, prob_threshold,
                prob_threshold_up, prob_threshold_down, primary_only,
                fee_rate, slippage, max_trades_per_day, models_dir_str, stop_loss_pct,
                consecutive_loss_limit, risk_pause_k_bars, tf_models_dict, model_path_str,
                require_ema_trend, min_atr_pct, min_volume_ratio, enable_dynamic_bet_ratio
            ))
    
    # 确定并行数
    if n_jobs == -1:
        n_jobs = os.cpu_count() or 4
    n_jobs = min(n_jobs, total_combinations)
    
    print(f"📊 开始回测 {total_combinations} 个组合（{len(test_symbols)} 个交易对 × {len(test_timeframes)} 个时间周期）...")
    if n_jobs > 1:
        print(f"🚀 并行处理: {n_jobs} 个进程同时运行")
    else:
        print(f"📝 串行处理: 逐个运行")
    
    # ⚡ 性能优化：不预先加载模型，改为在第一个任务完成后打印样本外说明
    # 这样可以避免第一个模型被加载两次（一次这里，一次在任务中）
    _sample_out_info_printed = [False]
    
    # 并行或串行执行
    if n_jobs > 1:
        # 并行处理
        completed = 0
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_task = {executor.submit(_run_single_backtest, task): task for task in tasks}
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                symbol, tf = task[0], task[1]
                completed += 1
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    # ⚡ 性能优化：在第一个任务完成后打印样本外说明（避免重复加载模型）
                    if not _sample_out_info_printed[0] and "error" not in result:
                        # 从结果中获取元数据（如果任务中已经加载了模型）
                        # 或者从任务参数中获取模型路径
                        model_path_str = task[-1]  # 最后一个参数是 model_path_str
                        if model_path_str:
                            try:
                                model_path = Path(model_path_str)
                                if model_path.exists():
                                    _, _, meta = load_predictor(model_path)
                                    train_cutoff_date = meta.get("train_cutoff_date")
                                    if train_cutoff_date:
                                        print(f"  [样本外] 回测只用 {train_cutoff_date[:10]} 之后的数据（训练时已排除，无泄露）")
                                    else:
                                        print(f"  [⚠️] 模型无 train_cutoff_date，回测用最近 test_days 天，可能含训练数据")
                                    _sample_out_info_printed[0] = True
                            except:
                                pass  # 如果加载失败，跳过说明
                    
                    if "error" in result:
                        print(f"[{completed}/{total_combinations}] {symbol} {tf}: ❌ {result['error']}")
                    else:
                        wr = result["win_rate"] * 100
                        pnl = result["profit_pct"]
                        print(f"[{completed}/{total_combinations}] {symbol} {tf}: ✅ 胜率: {wr:.1f}%, 盈亏: {pnl:+.1f}%, 交易次数: {result['total_trades']}")
                except Exception as e:
                    print(f"[{completed}/{total_combinations}] {symbol} {tf}: ❌ 异常: {e}")
                    results.append({
                        "symbol": symbol,
                        "timeframe": tf,
                        "error": str(e),
                        "total_trades": 0,
                        "wins": 0,
                        "losses": 0,
                        "win_rate": 0.0,
                        "final_capital": initial_capital,
                        "profit_pct": 0.0,
                        "max_drawdown": 0.0,
                        "min_capital": initial_capital,
                        "min_over_initial_pct": 100.0,
                        "equity_curve": [initial_capital],
                        "trade_history": [],
                    })
    else:
        # 串行处理（保持原有逻辑，便于调试）
        current = 0
        for task in tasks:
            symbol, tf = task[0], task[1]
            current += 1
            print(f"[{current}/{total_combinations}] 回测 {symbol} {tf}...", end=" ", flush=True)
            
            result = _run_single_backtest(task)
            results.append(result)
            
            # ⚡ 性能优化：在第一个任务完成后打印样本外说明（避免重复加载模型）
            if not _sample_out_info_printed[0] and "error" not in result:
                model_path_str = task[-1]  # 最后一个参数是 model_path_str
                if model_path_str:
                    try:
                        model_path = Path(model_path_str)
                        if model_path.exists():
                            _, _, meta = load_predictor(model_path)
                            train_cutoff_date = meta.get("train_cutoff_date")
                            if train_cutoff_date:
                                print(f"  [样本外] 回测只用 {train_cutoff_date[:10]} 之后的数据（训练时已排除，无泄露）")
                            else:
                                print(f"  [⚠️] 模型无 train_cutoff_date，回测用最近 test_days 天，可能含训练数据")
                            _sample_out_info_printed[0] = True
                    except:
                        pass  # 如果加载失败，跳过说明
            
            if "error" in result:
                print(f"❌ {result['error']}")
            else:
                wr = result["win_rate"] * 100
                pnl = result["profit_pct"]
                print(f"✅ 胜率: {wr:.1f}%, 盈亏: {pnl:+.1f}%, 交易次数: {result['total_trades']}")
    
    config = {
        "initial_capital": initial_capital,
        "bet_ratio": bet_ratio,
        "test_days": test_days,
        "primary_only": primary_only,
        "fee_rate": fee_rate,
        "slippage": slippage,
        "max_trades_per_day": max_trades_per_day,
        "models_dir": str(models_dir) if models_dir is not None else "data/models",
    }
    if prob_threshold_up is not None and prob_threshold_down is not None:
        config["prob_threshold_up"] = prob_threshold_up
        config["prob_threshold_down"] = prob_threshold_down
    else:
        config["prob_threshold"] = prob_threshold
    return {
        "results": results,
        "config": config,
        "timestamp": datetime.now().isoformat(),
    }


def generate_summary_table(results: List[Dict]) -> pd.DataFrame:
    """生成汇总表格"""
    rows = []
    for r in results:
        rows.append({
            "币种": r["symbol"].replace("/USDT", ""),
            "周期": r["timeframe"],
            "交易次数": r["total_trades"],
            "胜场": r["wins"],
            "败场": r["losses"],
            "胜率%": round(r["win_rate"] * 100, 1),
            "最终资金": r["final_capital"],
            "盈亏%": r["profit_pct"],
            "最大回撤%": r["max_drawdown"],
            "最低/初始%": r.get("min_over_initial_pct", 100),
        })
    
    df = pd.DataFrame(rows)
    # 按胜率排序
    df = df.sort_values("胜率%", ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # 排名从 1 开始
    df.index.name = "排名"
    
    return df


def generate_html_report(
    backtest_data: Dict[str, Any],
    output_path: Optional[str] = None,
) -> str:
    """生成 HTML 可视化报告"""
    results = backtest_data["results"]
    config = backtest_data["config"]
    
    # 生成汇总表
    summary_df = generate_summary_table(results)
    
    # 计算整体统计
    total_trades = sum(r["total_trades"] for r in results)
    total_wins = sum(r["wins"] for r in results)
    overall_win_rate = total_wins / total_trades * 100 if total_trades > 0 else 0
    
    avg_profit = np.mean([r["profit_pct"] for r in results if r["total_trades"] > 0])
    best = max(results, key=lambda x: x["win_rate"])
    worst = min(results, key=lambda x: x["win_rate"] if x["total_trades"] > 0 else 1)
    
    # 按币种汇总
    by_symbol = {}
    for r in results:
        s = r["symbol"]
        if s not in by_symbol:
            by_symbol[s] = {"wins": 0, "total": 0, "profit": []}
        by_symbol[s]["wins"] += r["wins"]
        by_symbol[s]["total"] += r["total_trades"]
        by_symbol[s]["profit"].append(r["profit_pct"])
    
    symbol_summary = []
    for s, d in by_symbol.items():
        wr = d["wins"] / d["total"] * 100 if d["total"] > 0 else 0
        avg_p = np.mean(d["profit"])
        symbol_summary.append({"symbol": s, "win_rate": wr, "avg_profit": avg_p})
    symbol_summary.sort(key=lambda x: x["win_rate"], reverse=True)
    
    # 按周期汇总
    by_tf = {}
    for r in results:
        tf = r["timeframe"]
        if tf not in by_tf:
            by_tf[tf] = {"wins": 0, "total": 0, "profit": []}
        by_tf[tf]["wins"] += r["wins"]
        by_tf[tf]["total"] += r["total_trades"]
        by_tf[tf]["profit"].append(r["profit_pct"])
    
    tf_summary = []
    for tf, d in by_tf.items():
        wr = d["wins"] / d["total"] * 100 if d["total"] > 0 else 0
        avg_p = np.mean(d["profit"])
        tf_summary.append({"timeframe": tf, "win_rate": wr, "avg_profit": avg_p})
    tf_summary.sort(key=lambda x: x["win_rate"], reverse=True)
    
    # 生成权益曲线数据（取有交易的）
    equity_data = []
    for r in results:
        if r["total_trades"] > 0:
            equity_data.append({
                "name": f"{r['symbol'].replace('/USDT', '')}_{r['timeframe']}",
                # 最近 100 个点
                "data": r["equity_curve"][-100:],
            })
    
    thr_desc = (
        f"双阈值 UP≥{int(config.get('prob_threshold_up', 0)*100)}% DOWN<{int(config.get('prob_threshold_down', 0)*100)}%"
        if "prob_threshold_up" in config and "prob_threshold_down" in config
        else f"概率阈值 {config['prob_threshold']*100}%"
    )
    
    # HTML 模板
    html = f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LightGBM 回测报告</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ 
            text-align: center; 
            margin-bottom: 30px; 
            font-size: 2.5em;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .stat-card h3 {{ color: #888; font-size: 0.9em; margin-bottom: 10px; }}
        .stat-card .value {{ font-size: 2em; font-weight: bold; }}
        .stat-card .value.positive {{ color: #00ff88; }}
        .stat-card .value.negative {{ color: #ff4757; }}
        .section {{ 
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 25px;
            backdrop-filter: blur(10px);
        }}
        .section h2 {{ 
            margin-bottom: 20px; 
            padding-bottom: 10px;
            border-bottom: 2px solid rgba(255,255,255,0.1);
        }}
        table {{ 
            width: 100%; 
            border-collapse: collapse; 
            margin-top: 15px;
        }}
        th, td {{ 
            padding: 12px 15px; 
            text-align: left; 
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        th {{ 
            background: rgba(0,212,255,0.2); 
            font-weight: 600;
        }}
        tr:hover {{ background: rgba(255,255,255,0.05); }}
        .win-rate {{ font-weight: bold; }}
        .win-rate.high {{ color: #00ff88; }}
        .win-rate.medium {{ color: #ffd93d; }}
        .win-rate.low {{ color: #ff4757; }}
        .profit.positive {{ color: #00ff88; }}
        .profit.negative {{ color: #ff4757; }}
        .rank {{ 
            display: inline-block;
            width: 30px;
            height: 30px;
            line-height: 30px;
            text-align: center;
            border-radius: 50%;
            font-weight: bold;
        }}
        .rank-1 {{ background: linear-gradient(135deg, #ffd700, #ffaa00); color: #000; }}
        .rank-2 {{ background: linear-gradient(135deg, #c0c0c0, #a0a0a0); color: #000; }}
        .rank-3 {{ background: linear-gradient(135deg, #cd7f32, #b87333); color: #fff; }}
        .chart-container {{ 
            height: 400px; 
            margin-top: 20px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        .config-info {{
            font-size: 0.9em;
            color: #888;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 LightGBM 预测回测报告</h1>
        
        <!-- 总体统计 -->
        <div class="stats-grid">
            <div class="stat-card">
                <h3>总交易次数</h3>
                <div class="value">{total_trades}</div>
            </div>
            <div class="stat-card">
                <h3>总体胜率</h3>
                <div class="value {'positive' if overall_win_rate >= 50 else 'negative'}">{overall_win_rate:.1f}%</div>
            </div>
            <div class="stat-card">
                <h3>平均盈亏</h3>
                <div class="value {'positive' if avg_profit >= 0 else 'negative'}">{avg_profit:+.1f}%</div>
            </div>
            <div class="stat-card">
                <h3>最佳组合</h3>
                <div class="value positive">{best['symbol'].replace('/USDT', '')}_{best['timeframe']}</div>
            </div>
        </div>
        
        <!-- 按币种排名 -->
        <div class="section">
            <h2>🏆 币种胜率排名</h2>
            <div class="summary-grid">
                {"".join(f'''
                <div class="stat-card">
                    <h3>{s["symbol"].replace("/USDT", "")}</h3>
                    <div class="value {'positive' if s["win_rate"] >= 50 else 'negative'}">{s["win_rate"]:.1f}%</div>
                    <div class="profit {'positive' if s["avg_profit"] >= 0 else 'negative'}">平均盈亏: {s["avg_profit"]:+.1f}%</div>
                </div>
                ''' for s in symbol_summary)}
            </div>
        </div>
        
        <!-- 按周期排名 -->
        <div class="section">
            <h2>⏱️ 周期胜率排名</h2>
            <div class="summary-grid">
                {"".join(f'''
                <div class="stat-card">
                    <h3>{t["timeframe"]}</h3>
                    <div class="value {'positive' if t["win_rate"] >= 50 else 'negative'}">{t["win_rate"]:.1f}%</div>
                    <div class="profit {'positive' if t["avg_profit"] >= 0 else 'negative'}">平均盈亏: {t["avg_profit"]:+.1f}%</div>
                </div>
                ''' for t in tf_summary)}
            </div>
        </div>
        
        <!-- 详细排名表 -->
        <div class="section">
            <h2>📋 详细排名（按胜率）</h2>
            <table>
                <thead>
                    <tr>
                        <th>排名</th>
                        <th>币种</th>
                        <th>周期</th>
                        <th>交易次数</th>
                        <th>胜/败</th>
                        <th>胜率</th>
                        <th>最终资金</th>
                        <th>盈亏%</th>
                        <th>最大回撤%</th>
                        <th>最低/初始%</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(f'''
                    <tr>
                        <td><span class="rank rank-{i+1 if i < 3 else ''}">{i+1}</span></td>
                        <td>{summary_df.iloc[i]["币种"]}</td>
                        <td>{summary_df.iloc[i]["周期"]}</td>
                        <td>{summary_df.iloc[i]["交易次数"]}</td>
                        <td>{summary_df.iloc[i]["胜场"]}/{summary_df.iloc[i]["败场"]}</td>
                        <td class="win-rate {'high' if summary_df.iloc[i]["胜率%"] >= 55 else 'medium' if summary_df.iloc[i]["胜率%"] >= 50 else 'low'}">{summary_df.iloc[i]["胜率%"]:.1f}%</td>
                        <td>${summary_df.iloc[i]["最终资金"]:.2f}</td>
                        <td class="profit {'positive' if summary_df.iloc[i]["盈亏%"] >= 0 else 'negative'}">{summary_df.iloc[i]["盈亏%"]:+.1f}%</td>
                        <td>{summary_df.iloc[i]["最大回撤%"]:.1f}%</td>
                        <td>{summary_df.iloc[i]["最低/初始%"]:.1f}%</td>
                    </tr>
                    ''' for i in range(len(summary_df)))}
                </tbody>
            </table>
        </div>
        
        <!-- 权益曲线 -->
        <div class="section">
            <h2>📈 资金曲线（最近交易）</h2>
            <div class="chart-container">
                <canvas id="equityChart"></canvas>
            </div>
        </div>
        
        <div class="config-info">
            <p>回测配置：初始资金 ${config['initial_capital']}，每次下注 {config['bet_ratio']*100}%，{thr_desc}，回测天数 {config['test_days']} 天</p>
            <p>指标说明：最大回撤% = 从权益曲线历史高点的最大跌幅；最低/初始% = 曾到达的最低资金/初始资金（越高越好，100%表示从未低于初始资金，接近0%表示曾几乎归零，风险极高）。</p>
            <p>生成时间：{backtest_data['timestamp']}</p>
        </div>
    </div>
    
    <script>
        // 权益曲线图表
        const ctx = document.getElementById('equityChart').getContext('2d');
        const datasets = {json.dumps([
            {
                "label": e["name"],
                "data": e["data"],
                "borderWidth": 2,
                "fill": False,
                "tension": 0.1,
            } for e in equity_data[:6]
        ])};
        // 最多显示 6 条线
        
        // 随机颜色
        const colors = ['#00d4ff', '#00ff88', '#ffd93d', '#ff4757', '#7b2cbf', '#ff9f43'];
        datasets.forEach((d, i) => {{
            d.borderColor = colors[i % colors.length];
            d.backgroundColor = colors[i % colors.length] + '20';
        }});
        
        new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: Array.from({{length: Math.max(...datasets.map(d => d.data.length))}}, (_, i) => i + 1),
                datasets: datasets
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{
                        labels: {{ color: '#eee' }}
                    }}
                }},
                scales: {{
                    x: {{
                        title: {{ display: true, text: '交易序号', color: '#888' }},
                        ticks: {{ color: '#888' }},
                        grid: {{ color: 'rgba(255,255,255,0.1)' }}
                    }},
                    y: {{
                        title: {{ display: true, text: '资金 (USDC)', color: '#888' }},
                        ticks: {{ color: '#888' }},
                        grid: {{ color: 'rgba(255,255,255,0.1)' }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    
    # 保存 HTML
    if output_path:
        Path(output_path).write_text(html, encoding="utf-8")
        print(f"\n📊 HTML 报告已保存: {output_path}")
    
    return html


def print_summary(backtest_data: Dict[str, Any]):
    """打印控制台摘要"""
    results = backtest_data["results"]
    config = backtest_data["config"]
    
    summary_df = generate_summary_table(results)
    
    print("\n" + "=" * 70)
    print("📊 LightGBM 预测回测结果")
    print("=" * 70)
    thr_str = (
        f"双阈值 UP≥{config['prob_threshold_up']*100:.0f}% DOWN<{config['prob_threshold_down']*100:.0f}%"
        if "prob_threshold_up" in config and "prob_threshold_down" in config
        else f"概率阈值 {config['prob_threshold']*100}%"
    )
    print(f"配置: 初始资金 ${config['initial_capital']}, 每次下注 {config['bet_ratio']*100}%, "
          f"{thr_str}, 回测 {config['test_days']} 天")
    print("=" * 70)
    
    print("\n🏆 胜率排名:")
    print(summary_df.to_string())
    print("  ※ 最大回撤% = 从权益曲线历史高点的最大跌幅；最低/初始% = 曾到达的最低资金/初始资金（越高越好，100%表示从未低于初始资金，接近0%表示曾几乎归零，风险极高）。")
    
    # 整体统计
    total_trades = sum(r["total_trades"] for r in results)
    total_wins = sum(r["wins"] for r in results)
    overall_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    
    print(f"\n📈 整体统计:")
    print(f"  总交易次数: {total_trades}")
    print(f"  总体胜率: {overall_wr:.1f}%")
    
    # 最佳和最差
    valid_results = [r for r in results if r["total_trades"] > 0]
    if valid_results:
        best = max(valid_results, key=lambda x: x["win_rate"])
        worst = min(valid_results, key=lambda x: x["win_rate"])
        print(f"\n🥇 最佳: {best['symbol']} {best['timeframe']} - 胜率 {best['win_rate']*100:.1f}%, 盈亏 {best['profit_pct']:+.1f}%")
        print(f"🥉 最差: {worst['symbol']} {worst['timeframe']} - 胜率 {worst['win_rate']*100:.1f}%, 盈亏 {worst['profit_pct']:+.1f}%")


def main():
    # 立即输出，确保用户知道脚本已启动
    print("=" * 70)
    print("📊 LightGBM 回测模拟脚本启动中...")
    print("=" * 70)
    import sys
    sys.stdout.flush()  # 确保输出立即显示
    
    ap = argparse.ArgumentParser(description="LightGBM 模型回测模拟（符合 Polymarket 规则，样本外测试）")
    ap.add_argument("--initial-capital", type=float, default=400.0, help="初始资金（USDC），默认 400")
    ap.add_argument("--bet-ratio", type=float, default=0.05, help="每次下注比例，默认 0.05 (5%%)")
    ap.add_argument("--test-days", type=int, default=90, help="回测天数：只使用最近 N 天的数据。有 train_cutoff 时仍会保证样本外（不早于 train_cutoff），同时截断为最近 test_days 天")
    ap.add_argument("--prob-threshold", type=float, default=None, help="概率阈值（单阈值），默认从配置读取")
    ap.add_argument("--prob-threshold-up", type=float, default=None, metavar="PCT", help="双阈值：UP 仅当 prob ≥ 此值（新模型优化）。与 --prob-threshold-down 同时指定时启用双阈值")
    ap.add_argument("--prob-threshold-down", type=float, default=None, metavar="PCT", help="双阈值：DOWN 仅当 prob < 此值（新模型优化）。与 --prob-threshold-up 同时指定时启用双阈值")
    ap.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help=f"手续费率，默认 {DEFAULT_FEE_RATE}")
    ap.add_argument("--slippage", type=float, default=DEFAULT_SLIPPAGE, help=f"滑点，默认 {DEFAULT_SLIPPAGE}")
    ap.add_argument("--max-trades-per-day", type=int, default=DEFAULT_MAX_TRADES_PER_DAY, help=f"每日最多交易次数，默认 {DEFAULT_MAX_TRADES_PER_DAY}，0=不限制")
    ap.add_argument("--primary-only", action="store_true", help="只使用主要特征（KDJ、Range）")
    ap.add_argument("--models-dir", type=str, default=None, help="模型根目录，默认 data/models；可指定 data/models_setA 等，用于多套模型对比")
    ap.add_argument("--output-html", type=str, default=None, help="输出 HTML 报告路径")
    ap.add_argument("--output-json", type=str, default=None, help="输出 JSON 结果路径")
    ap.add_argument("--stop-loss-pct", type=float, default=None, metavar="PCT", help="止损百分比（0-1），如 0.3 表示从峰值回撤 30%% 后停止交易（不自动恢复），默认不限制")
    ap.add_argument("--consecutive-loss-limit", type=int, default=None, metavar="N", help="连续亏损次数限制，如 5 表示连续亏损 5 次后触发暂停，默认不限制")
    ap.add_argument("--risk-pause-k-bars", type=int, default=None, metavar="N", help="连续亏损触发后，停止交易 N 根 K 线（如 8 表示停止 8 根 K 线后自动恢复），默认不限制")
    ap.add_argument("--enable-dynamic-bet-ratio", action="store_true", help="启用动态调整下注比例：回撤>20%%时减半，回撤>10%%时减少25%%，回撤≤10%%时恢复原始比例（与模拟交易逻辑一致）")
    ap.add_argument("--cpu-threads", type=int, default=None, metavar="N", help="CPU 线程数限制（可选），如 10 表示限制为 10 线程。默认不限制，使用系统默认值。可通过环境变量 OMP_NUM_THREADS 等控制")
    ap.add_argument("--n-jobs", type=int, default=1, metavar="N", help="并行进程数，1=串行（默认），-1=使用全部CPU核心，N=使用N个进程。并行可以大幅提升速度，充分利用CPU")
    args = ap.parse_args()
    
    # 如果用户通过命令行参数设置了 CPU 线程数，则设置环境变量
    if args.cpu_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(args.cpu_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(args.cpu_threads)
        print(f"🔧 CPU 线程数限制: {args.cpu_threads}（通过 --cpu-threads 参数设置）")
        sys.stdout.flush()
    # 否则完全使用系统默认值，不进行任何设置
    else:
        print(f"🔧 CPU 线程数: 使用系统默认值（未限制）")
        sys.stdout.flush()
    
    # 读取概率阈值配置
    prob_threshold = args.prob_threshold
    prob_threshold_up = args.prob_threshold_up
    prob_threshold_down = args.prob_threshold_down
    use_dual = prob_threshold_up is not None and prob_threshold_down is not None
    if use_dual:
        prob_threshold = 0.55  # 双阈值模式下单阈值未使用，传默认值即可
    elif prob_threshold is None:
        config_path = PROJECT_ROOT / "config" / "polymarket_threshold.json"
        if config_path.exists():
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            prob_threshold = cfg.get("prob_threshold", 0.55)
        else:
            prob_threshold = 0.55
    
    print(f"🚀 开始回测（样本外测试，含手续费/滑点/交易限制）...")
    print(f"  初始资金: ${args.initial_capital}")
    print(f"  下注比例: {args.bet_ratio * 100}%")
    if use_dual:
        print(f"  概率阈值: 双阈值 UP≥{prob_threshold_up*100:.0f}% DOWN<{prob_threshold_down*100:.0f}%（新模型优化）")
    else:
        print(f"  概率阈值: {prob_threshold * 100}%")
    print(f"  手续费: {args.fee_rate * 100}%, 滑点: {args.slippage * 100}%")
    print(f"  每日最多交易: {args.max_trades_per_day} 次 (0=不限)")
    
    # 风险控制参数
    if args.stop_loss_pct or args.consecutive_loss_limit or args.risk_pause_k_bars or args.enable_dynamic_bet_ratio:
        print(f"\n  🛡️ 风险控制:")
        if args.stop_loss_pct:
            print(f"    止损: 从峰值回撤 {args.stop_loss_pct * 100:.1f}% 后停止交易（不自动恢复）")
        if args.consecutive_loss_limit:
            print(f"    连续亏损限制: {args.consecutive_loss_limit} 次")
        if args.risk_pause_k_bars:
            print(f"    连续亏损暂停: 触发后停止交易 {args.risk_pause_k_bars} 根 K 线后自动恢复")
        if args.enable_dynamic_bet_ratio:
            print(f"    下注方式: 动态调整下注比例（回撤>20%%减半，>10%%减25%%，≤10%%恢复原始 {args.bet_ratio * 100}%%）")
        else:
            print(f"    下注方式: 固定下注比例 {args.bet_ratio * 100}%")
    else:
        print(f"  ⚠️  未启用风险控制（建议使用 --stop-loss-pct 0.3 等参数避免大回撤）")
    print(f"  回测天数: {args.test_days}")
    models_dir = Path(args.models_dir).resolve() if args.models_dir else None
    if models_dir:
        print(f"  模型目录: {models_dir}")
    print()
    
    # 运行回测
    backtest_data = run_full_backtest(
        initial_capital=args.initial_capital,
        bet_ratio=args.bet_ratio,
        test_days=args.test_days,
        prob_threshold=prob_threshold,
        prob_threshold_up=prob_threshold_up,
        prob_threshold_down=prob_threshold_down,
        primary_only=args.primary_only,
        fee_rate=args.fee_rate,
        slippage=args.slippage,
        max_trades_per_day=args.max_trades_per_day,
        models_dir=models_dir,
        stop_loss_pct=args.stop_loss_pct,
        consecutive_loss_limit=args.consecutive_loss_limit,
        risk_pause_k_bars=args.risk_pause_k_bars,
        enable_dynamic_bet_ratio=args.enable_dynamic_bet_ratio,
        n_jobs=args.n_jobs,
    )
    
    # 打印摘要
    print_summary(backtest_data)
    
    # 输出 HTML
    html_path = args.output_html or str(PROJECT_ROOT / "data" / "backtest_report.html")
    generate_html_report(backtest_data, html_path)
    
    # 输出 JSON
    if args.output_json:
        # 去除 equity_curve 和 trade_history 以减小文件大小
        export_data = {
            "config": backtest_data["config"],
            "timestamp": backtest_data["timestamp"],
            "results": [
                {k: v for k, v in r.items() if k not in ("equity_curve", "trade_history")}
                for r in backtest_data["results"]
            ]
        }
        Path(args.output_json).write_text(
            json.dumps(export_data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        print(f"📄 JSON 结果已保存: {args.output_json}")


if __name__ == "__main__":
    main()
