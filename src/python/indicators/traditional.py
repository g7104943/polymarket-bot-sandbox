"""RSI、布林带、成交量等传统指标"""

import pandas as pd
import numpy as np


def calculate_rsi(df: pd.DataFrame, periods: list = [14, 7]) -> pd.DataFrame:
    """
    RSI（相对强弱指标）。添加 rsi_14, rsi_7, rsi_14_norm, rsi_7_norm
    标准化到 0-1 范围
    """
    res = df.copy()
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    
    for period in periods:
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        res[f"rsi_{period}"] = rsi
        # 标准化到 0-1
        res[f"rsi_{period}_norm"] = rsi / 100.0
    
    # 保持向后兼容：rsi = rsi_14
    if 14 in periods:
        res["rsi"] = res["rsi_14"]
    return res


def calculate_bollinger_bands(
    df: pd.DataFrame, period: int = 20, mult: float = 2.0
) -> pd.DataFrame:
    """
    布林带。添加 bb_upper, bb_middle, bb_lower, bb_width, bb_percent_b
    %B = (close - bb_lower) / (bb_upper - bb_lower)，表示价格在布林带中的位置
    """
    res = df.copy()
    close = df["close"]
    mid = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    res["bb_middle"] = mid
    res["bb_upper"] = mid + mult * std
    res["bb_lower"] = mid - mult * std
    res["bb_width"] = (res["bb_upper"] - res["bb_lower"]) / res["bb_middle"].replace(0, np.nan)
    # %B：价格在布林带中的位置（0-1，可能超出）
    bb_range = res["bb_upper"] - res["bb_lower"]
    res["bb_percent_b"] = (close - res["bb_lower"]) / bb_range.replace(0, np.nan)
    return res


def calculate_volume_indicators(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """成交量: volume_ma, volume_ratio"""
    res = df.copy()
    vol = df["volume"]
    res["volume_ma"] = vol.rolling(window=period).mean()
    res["volume_ratio"] = vol / res["volume_ma"].replace(0, np.nan)
    return res
