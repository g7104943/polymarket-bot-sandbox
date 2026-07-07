"""OBV (On-Balance Volume) 指标"""

import pandas as pd
import numpy as np


def calculate_obv(df: pd.DataFrame) -> pd.DataFrame:
    """计算 OBV，添加 obv, obv_ema"""
    res = df.copy()
    close = df["close"]
    vol = df["volume"]
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    res["obv"] = (vol * direction).cumsum()
    res["obv_ema"] = res["obv"].ewm(span=20, adjust=False).mean()
    return res
