"""
特征工程：读取 Kdj.txt、Rang.txt，计算 KDJ、Range、MACD、OBV、RSI、布林带、成交量等，
新增 ROC、ATR、时间特征等，生成特征矩阵与标签(下一根K线涨跌)。
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from .indicators.kdj_range import add_range_kdj_features
from .indicators.macd import calculate_macd
from .indicators.obv import calculate_obv
from .indicators.traditional import (
    calculate_rsi,
    calculate_bollinger_bands,
    calculate_volume_indicators,
)
from .indicators.advanced import (
    calculate_stochastic,
    calculate_cci,
    calculate_mfi,
    calculate_vwap,
    calculate_donchian_channel,
    calculate_hurst_exponent,
    calculate_volume_delta,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============ 新增特征函数 ============

def calculate_roc(df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
    """
    计算多周期 ROC（Rate of Change）。
    ROC = (close - close_n) / close_n * 100
    """
    periods = periods or [1, 3, 5, 10, 20]
    res = df.copy()
    close = df["close"]
    for p in periods:
        res[f"roc_{p}"] = (close - close.shift(p)) / close.shift(p).replace(0, np.nan) * 100
    return res


def calculate_atr(df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
    """
    计算 ATR（Average True Range）和 ATR 百分比。
    """
    periods = periods or [14, 20]
    res = df.copy()
    high, low, close = df["high"], df["low"], df["close"]
    
    # True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    res["true_range"] = tr
    
    for p in periods:
        atr = tr.rolling(window=p).mean()
        res[f"atr_{p}"] = atr
        # ATR 百分比（相对于收盘价）
        res[f"atr_pct_{p}"] = atr / close.replace(0, np.nan) * 100
    
    return res


def calculate_price_position(df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
    """
    计算价格相对高低点位置。
    position = (close - low_N) / (high_N - low_N)
    范围 0~1，0 表示在最低点，1 表示在最高点
    """
    periods = periods or [5, 10, 20, 50]
    res = df.copy()
    close = df["close"]
    
    for p in periods:
        low_n = df["low"].rolling(window=p).min()
        high_n = df["high"].rolling(window=p).max()
        range_n = high_n - low_n
        res[f"price_pos_{p}"] = (close - low_n) / range_n.replace(0, np.nan)
    
    return res


def calculate_volatility(df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
    """
    计算历史波动率（收益率的标准差）。
    """
    periods = periods or [5, 10, 20]
    res = df.copy()
    returns = df["close"].pct_change()
    
    for p in periods:
        res[f"volatility_{p}"] = returns.rolling(window=p).std() * 100
    
    return res


def calculate_bb_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算价格在布林带内的位置。
    bb_position = (close - bb_lower) / (bb_upper - bb_lower)
    范围通常 0~1，可能超出
    """
    res = df.copy()
    if "bb_upper" in res.columns and "bb_lower" in res.columns:
        bb_range = res["bb_upper"] - res["bb_lower"]
        res["bb_position"] = (res["close"] - res["bb_lower"]) / bb_range.replace(0, np.nan)
    return res


def calculate_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    增强成交量特征：成交量变化率、价量背离信号。
    """
    res = df.copy()
    vol = df["volume"]
    close = df["close"]
    
    # 成交量变化率（多周期）
    for p in [1, 3, 5]:
        res[f"volume_change_{p}"] = vol.pct_change(periods=p) * 100
    
    # 相对成交量（vs 20 周期均值），如果还没计算过
    if "volume_ratio" not in res.columns:
        vol_ma = vol.rolling(window=20).mean()
        res["volume_ratio"] = vol / vol_ma.replace(0, np.nan)
    
    # 价量背离信号
    # 价格上涨但成交量下降 = 1（看跌背离）
    # 价格下跌但成交量上升 = -1（看涨背离）
    price_up = close > close.shift(1)
    price_down = close < close.shift(1)
    vol_up = vol > vol.shift(1)
    vol_down = vol < vol.shift(1)
    
    divergence = pd.Series(0, index=df.index)
    divergence[price_up & vol_down] = 1   # 看跌背离
    divergence[price_down & vol_up] = -1  # 看涨背离
    res["price_vol_divergence"] = divergence
    
    return res


def calculate_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算时间特征：小时、星期几、是否 UTC 0 点。
    需要 df 有 timestamp 或 date 列。
    """
    res = df.copy()
    
    # 尝试获取时间信息
    dt = None
    if "timestamp" in res.columns:
        dt = pd.to_datetime(res["timestamp"], unit="ms", utc=True)
    elif "date" in res.columns:
        dt = pd.to_datetime(res["date"], utc=True)
    
    if dt is not None:
        # 小时 (0-23)
        res["hour"] = dt.dt.hour
        # 星期几 (0=周一, 6=周日)
        res["day_of_week"] = dt.dt.dayofweek
        # 是否 UTC 0 点（日线开盘时间）
        res["is_utc_midnight"] = (dt.dt.hour == 0).astype(int)
        # 是否周末（周六周日）
        res["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        # 小时的正弦/余弦编码（捕捉周期性）
        res["hour_sin"] = np.sin(2 * np.pi * dt.dt.hour / 24)
        res["hour_cos"] = np.cos(2 * np.pi * dt.dt.hour / 24)
        # 星期的正弦/余弦编码
        res["dow_sin"] = np.sin(2 * np.pi * dt.dt.dayofweek / 7)
        res["dow_cos"] = np.cos(2 * np.pi * dt.dt.dayofweek / 7)
    
    return res


def calculate_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算动量相关特征。
    """
    res = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    
    # 收益率
    res["return_1"] = close.pct_change() * 100
    res["return_3"] = close.pct_change(periods=3) * 100
    res["return_5"] = close.pct_change(periods=5) * 100
    
    # 收益率的移动平均（趋势）
    returns = close.pct_change()
    res["return_ma_5"] = returns.rolling(5).mean() * 100
    res["return_ma_10"] = returns.rolling(10).mean() * 100
    
    # 连续涨跌天数
    up = (close > close.shift(1)).astype(int)
    down = (close < close.shift(1)).astype(int)
    
    # 计算连续上涨/下跌的天数
    up_streak = pd.Series(0, index=df.index, dtype=float)
    down_streak = pd.Series(0, index=df.index, dtype=float)
    
    for i in range(1, len(df)):
        if up.iloc[i] == 1:
            up_streak.iloc[i] = up_streak.iloc[i-1] + 1
        if down.iloc[i] == 1:
            down_streak.iloc[i] = down_streak.iloc[i-1] + 1
    
    res["up_streak"] = up_streak
    res["down_streak"] = down_streak
    
    return res


def calculate_lag_features(df: pd.DataFrame, lags: List[int] = None) -> pd.DataFrame:
    """
    计算滞后特征 - 前几根 K 线的信息。
    这对预测下一根 K 线非常重要！
    新增：RSI_lag1~3, MACD_hist_lag1~3, ATR_lag1~3 等关键指标的滞后版本
    """
    lags = lags or [1, 2, 3, 5, 10]
    res = df.copy()
    close = df["close"]
    volume = df["volume"]
    
    for lag in lags:
        # 价格滞后
        res[f"close_lag_{lag}"] = close.shift(lag)
        res[f"return_lag_{lag}"] = close.pct_change().shift(lag) * 100
        
        # 成交量滞后
        res[f"volume_lag_{lag}"] = volume.shift(lag)
        
        # 相对于滞后价格的变化
        res[f"close_vs_lag_{lag}"] = (close - close.shift(lag)) / close.shift(lag).replace(0, np.nan) * 100
    
    # 关键指标的滞后版本（滞后 1-3）
    key_indicators = {
        "rsi": "rsi_14" if "rsi_14" in res.columns else "rsi",
        "rsi_7": "rsi_7",
        "macd_hist": "macd_hist",
        "macd_line": "macd_line",
        "macd_signal": "macd_signal",
        "atr_14": "atr_14",
        "kdj_k": "kdj_k",
        "kdj_d": "kdj_d",
        "kdj_j": "kdj_j",
        "stoch_k": "stoch_k",
        "stoch_d": "stoch_d",
        "cci": "cci",
        "mfi": "mfi",
    }
    
    for name, col in key_indicators.items():
        if col in res.columns:
            for lag in [1, 2, 3]:
                res[f"{name}_lag{lag}"] = res[col].shift(lag)
    
    return res


def calculate_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算特征交叉 - 组合多个指标提升预测力。
    """
    res = df.copy()
    
    # KDJ 和 RSI 的组合信号
    if "kdj_k" in res.columns and "rsi" in res.columns:
        res["kdj_rsi_diff"] = res["kdj_k"] - res["rsi"]
        res["kdj_rsi_avg"] = (res["kdj_k"] + res["rsi"]) / 2
    
    # KDJ K-D 差值（动量）
    if "kdj_k" in res.columns and "kdj_d" in res.columns:
        res["kdj_kd_diff"] = res["kdj_k"] - res["kdj_d"]
        res["kdj_kd_cross_strength"] = res["kdj_kd_diff"] - res["kdj_kd_diff"].shift(1)
    
    # Range 和价格位置的组合
    if "rang_osc" in res.columns and "price_pos_20" in res.columns:
        res["rang_price_combo"] = res["rang_osc"] * res["price_pos_20"]
    
    # 布林带位置和 KDJ 的组合
    if "bb_position" in res.columns and "kdj_j" in res.columns:
        res["bb_kdj_combo"] = res["bb_position"] * res["kdj_j"] / 100
    
    # 成交量和价格动量的组合
    if "volume_ratio" in res.columns and "return_1" in res.columns:
        res["vol_momentum"] = res["volume_ratio"] * np.sign(res["return_1"])
    
    # ATR 标准化的价格变动
    if "atr_14" in res.columns and "return_1" in res.columns:
        res["return_atr_normalized"] = res["return_1"] / (res["atr_pct_14"].replace(0, np.nan))
    
    return res


def calculate_trend_features(
    df: pd.DataFrame,
    ema_periods: Optional[List[int]] = None,
    ema_slope_periods: Optional[List[int]] = None,
    adx_period: int = 14,
) -> pd.DataFrame:
    """
    计算趋势相关特征。
    """
    res = df.copy()
    close = df["close"]
    
    ema_periods = ema_periods or [5, 10, 20, 50, 100]
    ema_slope_periods = ema_slope_periods or [10, 20, 50]

    # 多周期 EMA
    for period in ema_periods:
        ema = close.ewm(span=period, adjust=False).mean()
        res[f"ema_{period}"] = ema
        res[f"close_vs_ema_{period}"] = (close - ema) / ema * 100

    # EMA 斜率（趋势强度）
    for period in ema_slope_periods:
        ema = close.ewm(span=period, adjust=False).mean()
        res[f"ema_slope_{period}"] = ema.diff(5) / ema.shift(5) * 100
    
    # 多 EMA 交叉信号
    if "ema_10" in res.columns and "ema_20" in res.columns:
        res["ema_10_20_diff"] = res["ema_10"] - res["ema_20"]
        res["ema_10_20_cross"] = np.sign(res["ema_10_20_diff"]) - np.sign(res["ema_10_20_diff"].shift(1))
    
    if "ema_20" in res.columns and "ema_50" in res.columns:
        res["ema_20_50_diff"] = res["ema_20"] - res["ema_50"]
    
    # ADX-like 趋势强度（简化版）
    high = df["high"]
    low = df["low"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr_14 = tr.rolling(adx_period).mean()

    plus_di = 100 * plus_dm.rolling(adx_period).mean() / atr_14.replace(0, np.nan)
    minus_di = 100 * minus_dm.rolling(adx_period).mean() / atr_14.replace(0, np.nan)
    
    res["plus_di"] = plus_di
    res["minus_di"] = minus_di
    res["di_diff"] = plus_di - minus_di
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    res["adx"] = dx.rolling(adx_period).mean()

    return res


def calculate_pattern_features(df: pd.DataFrame, bullish_windows: Optional[List[int]] = None) -> pd.DataFrame:
    """
    计算简单的价格形态特征。
    """
    res = df.copy()
    open_p = df["open"]
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    # K 线实体大小
    body = (close - open_p).abs()
    full_range = high - low
    res["body_ratio"] = body / full_range.replace(0, np.nan)
    
    # 上下影线
    upper_shadow = high - pd.concat([open_p, close], axis=1).max(axis=1)
    lower_shadow = pd.concat([open_p, close], axis=1).min(axis=1) - low
    res["upper_shadow_ratio"] = upper_shadow / full_range.replace(0, np.nan)
    res["lower_shadow_ratio"] = lower_shadow / full_range.replace(0, np.nan)
    
    # 阳线/阴线
    res["is_bullish"] = (close > open_p).astype(int)
    bullish = (close > open_p).astype(int)
    
    bullish_windows = bullish_windows or [3, 5, 7, 9, 11]

    # 最近 N 根中阳线比例，用比例便于跨 N 比较、更稳定
    for N in bullish_windows:
        cnt = bullish.rolling(N, min_periods=1).sum()
        res[f"recent_{N}_bullish_ratio"] = (cnt / N).astype(float)
    
    # 指数加权（衰减记忆）：近期阳线权重大，halflife 2=更快、5=稍慢
    res["bullish_ewm_hl2"] = bullish.ewm(halflife=2, min_periods=1).mean()
    res["bullish_ewm_hl5"] = bullish.ewm(halflife=5, min_periods=1).mean()
    
    # 二值门限：≥0.6 视为短期强多，≤0.4 视为短期强空（基于 5 根）
    r5 = res["recent_5_bullish_ratio"]
    res["strong_bull"] = (r5 >= 0.6).astype(int)
    res["strong_bear"] = (r5 <= 0.4).astype(int)
    
    # 连续同向 K 线数
    res["consecutive_bullish"] = bullish.groupby((bullish != bullish.shift()).cumsum()).cumcount() + 1
    res["consecutive_bullish"] = res["consecutive_bullish"] * bullish
    
    bearish = (close < open_p).astype(int)
    res["consecutive_bearish"] = bearish.groupby((bearish != bearish.shift()).cumsum()).cumcount() + 1
    res["consecutive_bearish"] = res["consecutive_bearish"] * bearish
    
    return res


# ============ 特征列判断 ============

# 主要特征：KDJ 和 Range Oscillator（来自 KDJRange.py）
PRIMARY_FEATURE_PREFIXES = ("kdj_", "rang_")

# 辅助特征前缀
SECONDARY_FEATURE_PREFIXES = (
    "macd_", "obv_", "bb_", "roc_", "atr_", 
    "price_pos_", "volatility_", "volume_change_", "return_",
    "close_lag_", "volume_lag_", "close_vs_lag_", "return_lag_",
    "kdj_rsi_", "kdj_kd_", "rang_price_", "bb_kdj_", "vol_momentum",
    "ema_", "close_vs_ema_", "di_", "plus_di", "minus_di", "adx",
    "body_ratio", "upper_shadow_", "lower_shadow_", "is_bullish",
    "consecutive_", "return_atr_", "recent_", "bullish_ewm_",
    # 🆕 新增特征前缀
    "rsi_", "stoch_", "cci", "mfi", "vwap", "dc_", "volume_delta",
    "rsi_lag", "macd_hist_lag", "macd_line_lag", "macd_signal_lag",
    "atr_lag", "kdj_lag", "stoch_lag", "cci_lag", "mfi_lag",
    "btc_", "corr_", "hurst_"
)

# 基础单列特征
# v2 新增特征前缀
V2_FEATURE_PREFIXES = (
    "return_skew_", "return_kurtosis_", "return_autocorr_", "variance_ratio_",
    "parkinson_vol_", "garman_klass_vol_", "vol_of_vol_", "realized_vs_parkinson_",
    "amihud_illiq_", "roll_spread_", "range_volume_ratio", "kyle_lambda_",
    "vol_regime_", "trend_consistency_", "mean_reversion_", "volume_regime_",
    "vol_volume_divergence", "trend_dir_", "trend_alignment", "trend_conflict_",
    "vol_ratio_4vs16", "signed_volume_momentum",
    "funding_rate", "oi_change", "long_short_ratio", "taker_buy_sell",
)

BASIC_FEATURES = (
    "rsi", "rsi_14", "rsi_7", "rsi_14_norm", "rsi_7_norm",
    "volume_ma", "volume_ratio", "true_range", "price_vol_divergence",
    "strong_bull", "strong_bear", "cci", "mfi", "vwap", "close_vs_vwap",
    "hurst_exponent", "volume_delta", "volume_delta_pct"
)

# 时间特征
TIME_FEATURES = (
    "hour", "day_of_week", "is_utc_midnight", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos"
)

# 动量特征
MOMENTUM_FEATURES = ("up_streak", "down_streak")


def _is_primary_feature(c: str) -> bool:
    """判断是否为主要特征（KDJ、Range）"""
    for p in PRIMARY_FEATURE_PREFIXES:
        if c.startswith(p):
            return True
    return False


def _is_feature_col(c: str, include_secondary: bool = True) -> bool:
    """
    判断是否为特征列。
    
    参数:
        c: 列名
        include_secondary: 是否包含辅助特征（默认 True）
    """
    if c in ("open", "high", "low", "close", "volume", "date", "timestamp", "symbol", "timeframe"):
        return False
    if c.startswith("mtf_"):
        return True
    
    # 主要特征（KDJ、Range）- 始终包含
    if _is_primary_feature(c):
        return True
    
    if not include_secondary:
        return False
    
    # 辅助特征
    if c in BASIC_FEATURES:
        return True
    if c in TIME_FEATURES:
        return True
    if c in MOMENTUM_FEATURES:
        return True
    for p in SECONDARY_FEATURE_PREFIXES:
        if c.startswith(p):
            return True
    for p in V2_FEATURE_PREFIXES:
        if c.startswith(p):
            return True
    return False


def calculate_cross_asset_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    计算跨资产相关性特征 — 适用于所有非 BTC 币种。

    BTC 是加密市场的领头羊，其价格变动通常领先其他币种。
    特征包括：
    - BTC 收益率滞后（领先信号）
    - 滚动相关系数（联动强度）
    - BTC/Alt 价格比（相对强弱）
    - 相关性变化速率（联动关系是否在变化）
    """
    res = df.copy()

    # BTC 是基准，不需要对 BTC 自身计算
    if "BTC" in symbol.upper():
        return res

    try:
        btc_path = DATA_RAW / "btc_usdt_15m.parquet"
        if not btc_path.exists():
            for tf in ["1h", "4h"]:
                btc_path = DATA_RAW / f"btc_usdt_{tf}.parquet"
                if btc_path.exists():
                    break
            else:
                return res

        btc_df = pd.read_parquet(btc_path)
        if "timestamp" not in btc_df.columns or "timestamp" not in res.columns:
            return res

        btc_df = btc_df.sort_values("timestamp")
        res_sorted = res.sort_values("timestamp")

        merged = pd.merge_asof(
            res_sorted[["timestamp", "close"]],
            btc_df[["timestamp", "close"]].rename(columns={"close": "btc_close"}),
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=15).total_seconds() * 1000,
        )

        res = res.sort_values("timestamp").reset_index(drop=True)
        merged = merged.sort_values("timestamp").reset_index(drop=True)

        if len(merged) == len(res):
            res["btc_close"] = merged["btc_close"].values
            btc_returns = res["btc_close"].pct_change() * 100
            alt_returns = res["close"].pct_change() * 100

            # BTC/Alt 价格比（相对强弱）
            res["btc_alt_ratio"] = res["btc_close"] / res["close"].replace(0, np.nan)

            # BTC 收益率滞后 1~3（领先信号）
            for lag in [1, 2, 3]:
                res[f"btc_return_lag{lag}"] = btc_returns.shift(lag)

            # 滚动相关系数（20 和 50 bars）
            for w in [20, 50]:
                if len(res) >= w:
                    res[f"corr_btc_{w}"] = alt_returns.rolling(w, min_periods=w // 2).corr(btc_returns)

            # 相关性变化速率：相关系数的一阶差分
            if f"corr_btc_20" in res.columns:
                res["corr_btc_change"] = res["corr_btc_20"].diff(5)

            # BTC-Alt 收益率差（spread）：正=BTC跑赢，负=Alt跑赢
            res["btc_alt_spread"] = btc_returns - alt_returns
            res["btc_alt_spread_ma5"] = res["btc_alt_spread"].rolling(5).mean()

    except Exception:
        pass

    return res


# ═══════════════════════ v2 新增特征函数 ═══════════════════════


def calculate_statistical_features(df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
    """
    收益率分布的高阶矩与统计特征 — 与传统技术指标正交。

    - 偏度（Skewness）：正偏=右尾长（突然大涨可能），负偏=左尾长（闪崩风险）
    - 峰度（Kurtosis）：>0=肥尾（极端波动多），<0=薄尾
    - 自相关（Autocorrelation）：正=趋势延续，负=均值回复
    - 方差比（Variance Ratio）：>1=趋势市，<1=均值回复市
    """
    periods = periods or [20, 50]
    res = df.copy()
    returns = df["close"].pct_change()

    for p in periods:
        # 偏度
        res[f"return_skew_{p}"] = returns.rolling(p, min_periods=max(p // 2, 5)).skew()
        # 峰度（excess kurtosis）
        res[f"return_kurtosis_{p}"] = returns.rolling(p, min_periods=max(p // 2, 5)).kurt()

    # 向量化自相关（避免 rolling.apply 的性能问题）
    for lag in [1, 3, 5]:
        r_shift = returns.shift(lag)
        roll_cov = returns.rolling(20, min_periods=10).cov(r_shift)
        roll_std = returns.rolling(20, min_periods=10).std()
        roll_std_lag = r_shift.rolling(20, min_periods=10).std()
        denom = roll_std * roll_std_lag
        res[f"return_autocorr_{lag}"] = (roll_cov / denom.replace(0, np.nan)).clip(-1, 1)

    # 方差比 VR(q) = Var(q-period return) / (q * Var(1-period return))
    # VR > 1 → 趋势；VR < 1 → 均值回复；VR ≈ 1 → 随机游走
    for q in [5, 10]:
        ret_q = df["close"].pct_change(q)
        var_q = ret_q.rolling(20, min_periods=10).var()
        var_1 = returns.rolling(20, min_periods=10).var()
        res[f"variance_ratio_{q}"] = (var_q / (q * var_1).replace(0, np.nan)).clip(0, 5)

    return res


def calculate_advanced_volatility(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    高级波动率估计 — 比 close-to-close 更高效地利用 OHLC 信息。

    - Parkinson：仅用 high-low，效率是 close-close 的 5 倍
    - Garman-Klass：用 OHLC 四个价格，效率最高
    - 波动率的波动率（Vol-of-Vol）：捕捉波动率状态变化
    - 已实现波动率 vs Parkinson 比率：差异大→有跳跃/缺口
    """
    res = df.copy()
    high = df["high"]
    low = df["low"]
    open_p = df["open"]
    close = df["close"]

    # Parkinson 波动率：σ_P = sqrt(1/(4nln2) * Σ(ln(H/L))^2)
    hl_log2 = (np.log(high / low.replace(0, np.nan))) ** 2
    res[f"parkinson_vol_{period}"] = np.sqrt(
        hl_log2.rolling(period, min_periods=period // 2).mean() / (4 * np.log(2))
    )

    # Garman-Klass 波动率：更高效地利用 OHLC
    # σ_GK = sqrt(1/n * Σ[0.5*ln(H/L)^2 - (2ln2-1)*ln(C/O)^2])
    co_log2 = (np.log(close / open_p.replace(0, np.nan))) ** 2
    gk_term = 0.5 * hl_log2 - (2 * np.log(2) - 1) * co_log2
    res[f"garman_klass_vol_{period}"] = np.sqrt(
        gk_term.rolling(period, min_periods=period // 2).mean().clip(lower=0)
    )

    # 波动率的波动率（Vol-of-Vol）— 波动率是否在剧烈变化
    ret_vol = df["close"].pct_change().abs()
    rolling_vol = ret_vol.rolling(period, min_periods=period // 2).mean()
    res[f"vol_of_vol_{period}"] = rolling_vol.rolling(period, min_periods=period // 2).std()

    # 已实现波动率 vs Parkinson 比率 — 差异大说明存在跳跃/缺口
    realized = df["close"].pct_change().rolling(period, min_periods=period // 2).std()
    parkinson = res[f"parkinson_vol_{period}"]
    res[f"realized_vs_parkinson_{period}"] = (
        realized / parkinson.replace(0, np.nan)
    ).clip(0, 10)

    return res


def calculate_microstructure_proxies(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    从 OHLCV 估计微观结构指标 — 无需订单簿数据。

    - Amihud 非流动性：|return|/volume，高=价格易被成交量推动
    - Roll 价差估计：从价格自协方差估计 bid-ask spread
    - 范围-成交量比：(high-low)/volume，衡量每单位成交量的价格影响
    - Kyle Lambda：价格冲击系数（回归斜率）
    """
    res = df.copy()
    returns = df["close"].pct_change()
    volume = df["volume"]
    high = df["high"]
    low = df["low"]
    close = df["close"]

    # Amihud 非流动性比率：ILLIQ = mean(|r_t| / V_t)
    # 高值 = 低流动性（少量交易推动价格大幅变化）
    illiq = returns.abs() / volume.replace(0, np.nan)
    res[f"amihud_illiq_{period}"] = illiq.rolling(period, min_periods=period // 2).mean()

    # Roll 价差估计：Spread = 2 * sqrt(max(-Cov(Δp_t, Δp_{t-1}), 0))
    # 利用价格序列负自协方差来估计 bid-ask spread
    dp = close.diff()
    dp_lag = dp.shift(1)
    roll_cov = dp.rolling(period, min_periods=period // 2).cov(dp_lag)
    res[f"roll_spread_{period}"] = 2 * np.sqrt((-roll_cov).clip(lower=0))
    # 标准化为百分比
    res[f"roll_spread_pct_{period}"] = res[f"roll_spread_{period}"] / close.replace(0, np.nan) * 100

    # 范围-成交量比：(H-L)/Volume，每单位成交量引起的价格变动幅度
    range_pct = (high - low) / close.replace(0, np.nan)
    res["range_volume_ratio"] = range_pct / volume.replace(0, np.nan)
    res[f"range_volume_ratio_ma_{period}"] = res["range_volume_ratio"].rolling(
        period, min_periods=period // 2
    ).mean()

    # Kyle Lambda（简化版）：用 |Δprice| vs signed_volume 的比率
    # 衡量单位成交量的价格冲击大小
    signed_vol = volume * np.sign(returns)
    abs_dp = close.diff().abs()
    lambda_ratio = abs_dp / signed_vol.abs().replace(0, np.nan)
    res[f"kyle_lambda_{period}"] = lambda_ratio.rolling(
        period, min_periods=period // 2
    ).mean()

    return res


def calculate_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    市场状态（Regime）量化 — 帮助模型识别当前处于什么市场环境。

    - 波动率状态分位数：当前波动率在近期分布中的位置（0-1）
    - 趋势持续性：连续同向 bar 的动量强度
    - 均值回复 Z-score：价格偏离均线的标准化程度
    - 成交量状态：当前成交量在近期分布中的位置
    """
    res = df.copy()
    returns = df["close"].pct_change()
    close = df["close"]
    volume = df["volume"]

    # 波动率状态分位数（50 bar 窗口）
    # 0 = 极低波动，1 = 极高波动
    vol = returns.abs().rolling(10).mean()
    res["vol_regime_quantile"] = vol.rolling(50, min_periods=20).rank(pct=True)

    # 趋势持续性：用过去 20 bar 收益率的符号一致性
    # 值接近 1 = 持续上涨，接近 -1 = 持续下跌，接近 0 = 震荡
    sign_returns = np.sign(returns)
    res["trend_consistency_20"] = sign_returns.rolling(20, min_periods=10).mean()

    # 均值回复 Z-score：(close - EMA50) / rolling_std
    # |Z| > 2 → 超买/超卖，回归概率高
    ema50 = close.ewm(span=50, adjust=False).mean()
    rolling_std = close.rolling(50, min_periods=20).std()
    res["mean_reversion_zscore"] = (close - ema50) / rolling_std.replace(0, np.nan)

    # 成交量状态分位数（50 bar 窗口）
    res["volume_regime_quantile"] = volume.rolling(50, min_periods=20).rank(pct=True)

    # 波动率-成交量分歧：高波动低成交量→危险信号
    vol_q = res["vol_regime_quantile"]
    vol_v_q = res["volume_regime_quantile"]
    res["vol_volume_divergence"] = vol_q - vol_v_q

    return res


def calculate_multi_scale_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    多尺度趋势一致性 — 捕捉不同时间尺度趋势是否对齐。

    将 15m 数据上采样到 1h/4h/1d 趋势方向，当所有尺度一致时信号最强。
    这些特征与单一尺度的 EMA/ROC 不同，因为它衡量的是"趋势对齐程度"。
    """
    res = df.copy()
    close = df["close"]

    # 各尺度的趋势方向（用 EMA 斜率的符号表示）
    # 1h = 4 bars, 4h = 16 bars, 1d = 96 bars
    scales = {"1h": 4, "4h": 16, "1d": 96}
    trend_cols = []

    for name, bars in scales.items():
        ema = close.ewm(span=bars, adjust=False).mean()
        slope = ema.diff(max(bars // 4, 1))
        trend = np.sign(slope)
        col = f"trend_dir_{name}"
        res[col] = trend
        trend_cols.append(col)

    # 多尺度趋势一致性得分：所有尺度方向之和 / 尺度数
    # +1 = 所有尺度看涨，-1 = 所有尺度看跌，0 = 分歧
    res["trend_alignment"] = res[trend_cols].sum(axis=1) / len(trend_cols)

    # 趋势一致性绝对值（不分方向，纯衡量"是否一致"）
    res["trend_alignment_abs"] = res["trend_alignment"].abs()

    # 短期 vs 长期趋势冲突信号
    # 短期(1h)和长期(1d)方向不一致时 = 1
    res["trend_conflict_1h_1d"] = (
        res["trend_dir_1h"] != res["trend_dir_1d"]
    ).astype(int)

    return res


def calculate_vol_and_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    波动率比率 + 带方向的成交量动量 — 与现有特征正交。

    - vol_ratio_4vs16: 短期(4bar≈1h) vs 中期(16bar≈4h) 波动率之比。
      >1 = 短期波动加剧（突发行情），<1 = 短期相对平静。
      不同于 ATR/BB_width 等绝对波动率，这是一个无量纲的"波动率加速度"。
    - signed_volume_momentum: 用价格方向加权的累计成交量差额滚动值。
      正值 = 近期上涨伴随放量 > 下跌放量（买方主导）
      负值 = 下跌放量 > 上涨放量（卖方主导）
      不同于 OBV（纯累计），这里用滚动窗口做归一化，捕捉近期趋势。
    """
    res = df.copy()
    close = df["close"]
    volume = df["volume"]

    # ---------- vol_ratio_4vs16 ----------
    returns = close.pct_change()
    vol_short = returns.rolling(4, min_periods=2).std()
    vol_long = returns.rolling(16, min_periods=8).std()
    res["vol_ratio_4vs16"] = vol_short / vol_long.replace(0, np.nan)

    # ---------- signed_volume_momentum ----------
    # 每根 bar 的"带符号成交量"= sign(return) * volume
    signed_vol = np.sign(returns) * volume
    # 12-bar 滚动均值，再除以成交量均值做归一化（无量纲）
    sv_ma = signed_vol.rolling(12, min_periods=6).mean()
    vol_ma = volume.rolling(12, min_periods=6).mean()
    res["signed_volume_momentum"] = sv_ma / vol_ma.replace(0, np.nan)

    return res


def default_indicator_recipe() -> Dict[str, Any]:
    return {
        "macd": {"fast": 12, "slow": 26, "signal": 9},
        "rsi_periods": [14, 7],
        "bollinger": {"period": 20, "mult": 2.0},
        "volume_period": 20,
        "stochastic": {"k_period": 14, "d_period": 3},
        "cci_period": 20,
        "mfi_period": 14,
        "donchian_period": 20,
        "roc_periods": [1, 3, 5, 10, 20],
        "atr_periods": [14, 20],
        "price_position_periods": [5, 10, 20, 50],
        "volatility_periods": [5, 10, 20],
        "lag_periods": [1, 2, 3, 5, 10],
        "ema_periods": [5, 10, 20, 50, 100],
        "ema_slope_periods": [10, 20, 50],
        "adx_period": 14,
        "bullish_windows": [3, 5, 7, 9, 11],
    }


def build_features(df: pd.DataFrame, symbol: Optional[str] = None, recipe: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    """
    在 OHLCV 上计算全部指标。要求列: open, high, low, close, volume。
    
    参数:
        symbol: 交易对名称（如 "ETH/USDT"），用于跨资产特征计算
    """
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    for need in ("open", "high", "low", "close", "volume"):
        if need not in df.columns:
            raise ValueError(f"DataFrame 需要列: {need}")

    recipe_cfg = default_indicator_recipe()
    if recipe:
        for key, value in recipe.items():
            recipe_cfg[key] = value

    # 核心指标（KDJ + Range）
    df = add_range_kdj_features(df, config_dir=PROJECT_ROOT / "config")
    
    # 传统技术指标
    macd_cfg = recipe_cfg.get("macd") or {}
    df = calculate_macd(
        df,
        fast=int(macd_cfg.get("fast", 12)),
        slow=int(macd_cfg.get("slow", 26)),
        signal=int(macd_cfg.get("signal", 9)),
    )                                          # MACD
    df = calculate_obv(df)                     # OBV
    df = calculate_rsi(df, periods=list(recipe_cfg.get("rsi_periods") or [14, 7]))   # RSI
    bb_cfg = recipe_cfg.get("bollinger") or {}
    df = calculate_bollinger_bands(
        df,
        period=int(bb_cfg.get("period", 20)),
        mult=float(bb_cfg.get("mult", 2.0)),
    )                                          # Bollinger Bands
    df = calculate_volume_indicators(df, period=int(recipe_cfg.get("volume_period", 20)))  # Volume indicators
    
    # 🆕 新增高级指标
    stoch_cfg = recipe_cfg.get("stochastic") or {}
    df = calculate_stochastic(
        df,
        k_period=int(stoch_cfg.get("k_period", 14)),
        d_period=int(stoch_cfg.get("d_period", 3)),
    )                                                       # Stochastic Oscillator
    df = calculate_cci(df, period=int(recipe_cfg.get("cci_period", 20)))              # CCI
    df = calculate_mfi(df, period=int(recipe_cfg.get("mfi_period", 14)))              # MFI
    df = calculate_vwap(df, reset_period="D")              # VWAP (日内重置)
    df = calculate_donchian_channel(df, period=int(recipe_cfg.get("donchian_period", 20)))  # Donchian Channel
    df = calculate_volume_delta(df, period=10)             # Volume Delta
    
    # 动量和波动率
    df = calculate_roc(df, periods=list(recipe_cfg.get("roc_periods") or [1, 3, 5, 10, 20]))  # ROC 多周期
    df = calculate_atr(df, periods=list(recipe_cfg.get("atr_periods") or [14, 20]))            # ATR
    df = calculate_price_position(df, periods=list(recipe_cfg.get("price_position_periods") or [5, 10, 20, 50]))
    df = calculate_volatility(df, periods=list(recipe_cfg.get("volatility_periods") or [5, 10, 20]))
    df = calculate_bb_position(df)            # 布林带位置
    df = calculate_volume_features(df)        # 增强成交量特征
    df = calculate_time_features(df)          # 时间特征
    df = calculate_momentum_features(df)      # 动量特征
    
    # 🆕 强化特征（提升预测力）
    df = calculate_lag_features(df, lags=list(recipe_cfg.get("lag_periods") or [1, 2, 3, 5, 10]))  # 滞后特征
    df = calculate_cross_features(df)         # 特征交叉
    df = calculate_trend_features(
        df,
        ema_periods=list(recipe_cfg.get("ema_periods") or [5, 10, 20, 50, 100]),
        ema_slope_periods=list(recipe_cfg.get("ema_slope_periods") or [10, 20, 50]),
        adx_period=int(recipe_cfg.get("adx_period", 14)),
    )                                         # 趋势特征（EMA、ADX）
    df = calculate_pattern_features(df, bullish_windows=list(recipe_cfg.get("bullish_windows") or [3, 5, 7, 9, 11]))  # K线形态特征
    
    # 🆕 跨资产相关性（所有非 BTC 币种 vs BTC）
    if symbol:
        df = calculate_cross_asset_features(df, symbol)
    
    # 🆕 Hurst Exponent（趋势持久性）
    # ⚡ 性能优化：完全跳过计算，填充默认值 0.5（随机游走）
    df["hurst_exponent"] = 0.5

    # ═══════════════════════════════════════════════════════════
    # 🆕 v2 新增特征 — 与现有技术指标正交的信息维度
    # ═══════════════════════════════════════════════════════════
    df = calculate_statistical_features(df)       # 收益率分布高阶矩 + 自相关 + 方差比
    df = calculate_advanced_volatility(df)        # Parkinson / Garman-Klass / 波动率的波动率
    df = calculate_microstructure_proxies(df)     # Amihud 非流动性 / Roll 价差估计 / 范围成交量比
    df = calculate_regime_features(df)            # 波动率/趋势 状态量化
    df = calculate_multi_scale_trend(df)          # 多尺度趋势一致性（1h/4h/1d 方向）
    df = calculate_vol_and_flow_features(df)      # 波动率比率 + 带方向成交量动量

    return df


def get_feature_columns(df: pd.DataFrame, primary_only: bool = False) -> List[str]:
    """
    获取特征列。
    
    参数:
        df: DataFrame
        primary_only: 如果为 True，只返回主要特征（KDJ、Range）
    """
    return [c for c in df.columns if _is_feature_col(c, include_secondary=not primary_only)]


def get_primary_feature_columns(df: pd.DataFrame) -> List[str]:
    """只获取主要特征（KDJ、Range）"""
    return [c for c in df.columns if _is_primary_feature(c)]


def build_labels(
    df: pd.DataFrame,
    threshold_pct: Optional[float] = None,
    use_three_class: bool = False,
    lookahead: int = 1,
    use_dynamic_threshold: bool = False,
    atr_multiplier: float = 1.0,
    atr_period: int = 14,
) -> pd.Series:
    """
    标签：未来 N 根 K 线的方向。
    
    参数:
        threshold_pct: 可选的涨跌幅阈值（百分比，如 0.1 表示 0.1%）
                      若设置，只有涨跌幅超过阈值才标记为 UP/DOWN
        use_three_class: 若为 True 且设置了 threshold_pct，使用三分类:
                        UP=2, FLAT=1, DOWN=0
        lookahead: 预测未来多少根 K 线的方向（默认 1，建议 3-5 更稳定）
        use_dynamic_threshold: 步骤 4 增强——若为 True，用 ATR 动态阈值代替固定 threshold_pct
        atr_multiplier: ATR 乘数（默认 1.0），dynamic_threshold = ATR_pct * atr_multiplier
        atr_period: ATR 计算周期（默认 14）
    
    返回:
        标签 Series，二分类: 0=DOWN, 1=UP
                      三分类: 0=DOWN, 1=FLAT, 2=UP
    """
    close = df["close"]
    
    # 预测未来 N 根 K 线后的收盘价
    future_close = close.shift(-lookahead)
    
    # 计算收益率（百分比）
    pct_change = (future_close - close) / close * 100

    # 步骤 4 增强：ATR 动态阈值
    if use_dynamic_threshold:
        # 计算 ATR 百分比（相对于 close），作为动态阈值
        high = df["high"]
        low = df["low"]
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=atr_period, min_periods=1).mean()
        # 转为百分比（与 pct_change 同单位）
        dynamic_threshold = (atr / close * 100) * atr_multiplier

        if use_three_class:
            labels = pd.Series(1, index=df.index)  # 默认 FLAT
            labels[pct_change > dynamic_threshold] = 2   # UP
            labels[pct_change < -dynamic_threshold] = 0  # DOWN
        else:
            labels = pd.Series(np.nan, index=df.index)
            labels[pct_change > dynamic_threshold] = 1   # UP
            labels[pct_change < -dynamic_threshold] = 0  # DOWN
        return labels
    
    if threshold_pct is None:
        # 原始逻辑：简单涨跌
        return (future_close > close).astype(int)
    
    if use_three_class:
        # 三分类：DOWN=0, FLAT=1, UP=2
        labels = pd.Series(1, index=df.index)  # 默认 FLAT
        labels[pct_change > threshold_pct] = 2   # UP
        labels[pct_change < -threshold_pct] = 0  # DOWN
        return labels
    else:
        # 二分类带阈值：只有超过阈值才标记，其他为 NaN（训练时会被过滤）
        labels = pd.Series(np.nan, index=df.index)
        labels[pct_change > threshold_pct] = 1   # UP
        labels[pct_change < -threshold_pct] = 0  # DOWN
        return labels


DATA_RAW = PROJECT_ROOT / "data" / "raw"
MTF_CANDIDATE_COLS = ["ema_10_20_diff", "rang_osc", "kdj_k", "rsi", "bb_position"]


def _to_ts_ms(d: pd.DataFrame) -> pd.Series:
    if "timestamp" in d.columns:
        t = d["timestamp"]
        return (pd.to_numeric(t, errors="coerce") // 1).astype("int64")  # 保持 ms 或转为 ms
    if "date" in d.columns:
        return pd.to_datetime(d["date"], utc=True).astype("int64") // 10**6
    raise ValueError("需要 timestamp 或 date 列")


def add_multi_timeframe_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    在 15m 的 build_features 结果上，按时间向后对齐合并 1h、4h 的高层特征（mtf_1h_*, mtf_4h_*）。
    仅当 data/raw 下存在对应 {symbol}_{1h|4h}.parquet 时添加。

    防泄漏：只用「已收盘」的高周期 K 线。1h 的 _tm 为开盘，收盘= _tm+1h；4h 收盘= _tm+4h。
    按 _tm_avail = 收盘时间 做 merge_asof(backward)，保证 15m 时刻只用在其之前已收盘的 1h/4h。
    """
    df = df.copy()
    try:
        df["_tm"] = _to_ts_ms(df)
    except ValueError:
        return df
    sym = symbol.replace("/", "_").lower()
    # 周期(秒) -> 毫秒
    tf_seconds = {"1h": 3600, "4h": 4 * 3600}

    for tf, prefix in [("1h", "mtf_1h_"), ("4h", "mtf_4h_")]:
        path = DATA_RAW / f"{sym}_{tf}.parquet"
        if not path.exists():
            continue
        try:
            h = pd.read_parquet(path)
            # ⚠️ 性能警告：这里需要为1h/4h数据计算特征，可能较慢
            # 对于120天的数据，1h约2000根，4h约500根，但特征计算仍然需要时间
            h = build_features(h)
        except Exception as e:
            logger.warning("add_multi_timeframe_features: 跳过 %s（读取 parquet 或 build_features 失败）: %s", tf, e)
            continue
        take = [c for c in MTF_CANDIDATE_COLS if c in h.columns]
        if not take:
            continue
        try:
            h["_tm"] = _to_ts_ms(h)
        except ValueError as e:
            logger.warning("add_multi_timeframe_features: 跳过 %s（时间戳解析失败）: %s", tf, e)
            continue
        # 仅当该根 K 线已收盘后才可用：_tm 为开盘，收盘 = _tm + 周期
        h["_tm_avail"] = h["_tm"] + tf_seconds[tf] * 1000
        h = h[["_tm_avail"] + take].dropna(how="all", subset=take).sort_values("_tm_avail")
        h = h.rename(columns={c: prefix + c for c in take})
        df = pd.merge_asof(df.sort_values("_tm"), h, left_on="_tm", right_on="_tm_avail", direction="backward")
        df = df.drop(columns=["_tm_avail"], errors="ignore")

    # 末行 mtf_ 可能因 1h/4h 未更新到最新而 NaN（merge_asof 无 backward 匹配），用 ffill 兜底避免「最后一行特征含 NaN」
    mtf_cols = [c for c in df.columns if c.startswith("mtf_")]
    if mtf_cols:
        df[mtf_cols] = df[mtf_cols].ffill()

    df = df.drop(columns=["_tm"], errors="ignore")
    return df


def prepare_train_data(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    label_threshold_pct: Optional[float] = None,
    use_three_class: bool = False,
    primary_only: bool = False,
    lookahead: int = 1,
    add_multi_timeframe: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    从 OHLCV 构建特征和标签。返回 (X, y)。

    参数:
        df: OHLCV 数据
        feature_cols: 可选的特征列列表
        label_threshold_pct: 涨跌幅阈值（百分比），如 0.1 表示 0.1%
        use_three_class: 是否使用三分类
        lookahead: 预测未来多少根 K 线（默认 1，建议 3-5 更稳定）
        primary_only: 如果为 True，只使用主要特征（KDJ、Range）
        add_multi_timeframe: 若 True 且 base 为 15m，合并 1h/4h 的 mtf_* 特征
    """
    # 获取 symbol 用于跨资产特征
    symbol = None
    if "symbol" in df.columns:
        symbol = str(df["symbol"].iloc[0]) if len(df) > 0 else None
    
    df = build_features(df, symbol=symbol)
    if add_multi_timeframe and "timeframe" in df.columns and "symbol" in df.columns:
        tf = str(df["timeframe"].iloc[0]).lower() if len(df) else ""
        if tf == "15m":
            sym = str(df["symbol"].iloc[0])
            df = add_multi_timeframe_features(df, sym)
    if feature_cols:
        cols = feature_cols
    elif primary_only:
        cols = get_primary_feature_columns(df)
    else:
        cols = get_feature_columns(df)
    cols = [c for c in cols if c in df.columns]
    if not cols:
        raise ValueError("没有可用的特征列")
    
    # 构建标签（预测未来 lookahead 根 K 线的方向）
    y = build_labels(df, threshold_pct=label_threshold_pct, use_three_class=use_three_class, lookahead=lookahead)
    X = df[cols].copy()
    
    # 排除最后 lookahead 行（没有未来数据）
    X = X.iloc[:-lookahead]
    y = y.iloc[:-lookahead]
    mask = X.notna().all(axis=1) & y.notna()
    return X[mask], y[mask]


def get_feature_names_from_frame(df: pd.DataFrame) -> List[str]:
    return get_feature_columns(build_features(df))
