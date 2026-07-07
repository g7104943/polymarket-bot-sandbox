"""MACD 指标"""

import pandas as pd


def calculate_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """计算 MACD，添加 macd_line, macd_signal, macd_hist"""
    res = df.copy()
    close = df["close"]
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    res["macd_line"] = ema_fast - ema_slow
    res["macd_signal"] = res["macd_line"].ewm(span=signal, adjust=False).mean()
    res["macd_hist"] = res["macd_line"] - res["macd_signal"]
    return res
