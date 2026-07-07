"""
KDJ and Range Oscillator (Zeiierman) - 从 Kdj.txt, Rang.txt 读取参数
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional


def _load_kdj_config(config_dir: Optional[Path] = None) -> Dict[str, int]:
    """从 config/Kdj.txt 读取参数"""
    config_dir = config_dir or Path(__file__).resolve().parents[2] / "config"
    path = config_dir / "Kdj.txt"
    params = {"rsv_length": 9, "k_smooth": 3, "d_smooth": 3}
    if path.exists():
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            line = line.split("#")[0].strip()
            if "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if k in params:
                    params[k] = int(v)
    return params


def _load_rang_config(config_dir: Optional[Path] = None) -> Dict[str, float]:
    """从 config/Rang.txt 读取参数"""
    config_dir = config_dir or Path(__file__).resolve().parents[2] / "config"
    path = config_dir / "Rang.txt"
    params = {"range_period": 20, "range_multiplier": 2.0}
    if path.exists():
        for line in path.read_text(encoding="utf-8").strip().splitlines():
            line = line.split("#")[0].strip()
            if "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if k == "range_period":
                    params[k] = int(v)
                elif k == "range_multiplier":
                    params[k] = float(v)
    return params


class RangeOscillator:
    """Range Oscillator (Zeiierman). 参数来自 Rang.txt: range_period->length, range_multiplier->mult"""

    def __init__(
        self,
        length: Optional[int] = None,
        mult: Optional[float] = None,
        config_dir: Optional[Path] = None,
    ):
        cfg = _load_rang_config(config_dir)
        self.length = length if length is not None else cfg["range_period"]
        self.mult = mult if mult is not None else cfg["range_multiplier"]

    def _atr(self, df: pd.DataFrame, period: int = 2000) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        actual = min(period, len(df))
        atr = tr.rolling(window=actual).mean()
        if atr.isna().all():
            atr = tr.rolling(window=min(200, len(df))).mean()
        return atr.bfill()

    def _weighted_ma(self, close: pd.Series) -> pd.Series:
        """
        向量化实现的加权移动平均。
        权重 = abs(close[i] - close[i-1]) / close[i-1]
        """
        close_arr = close.values.astype(float)
        n = len(close_arr)
        
        # 计算价格变化率作为权重
        # d = abs(close[i] - close[i-1])
        # w = d / close[i-1]
        price_diff = np.abs(np.diff(close_arr))
        prev_close = close_arr[:-1]
        
        # 避免除零
        with np.errstate(divide='ignore', invalid='ignore'):
            weights = np.where(prev_close != 0, price_diff / prev_close, 0)
        
        # 结果数组
        out = np.full(n, np.nan)
        
        # 使用滑动窗口计算（向量化）
        for i in range(self.length - 1, n):
            # 窗口范围：[i-length+1, i] 对应的权重索引
            start = max(0, i - self.length + 1)
            end = i
            
            if end <= 0:
                continue
                
            # 权重索引对应 close[start:end] -> close[start+1:end+1] 的变化
            w_start = max(0, start)
            w_end = end
            
            if w_end <= w_start:
                continue
            
            window_weights = weights[w_start:w_end]
            window_close = close_arr[w_start + 1:w_end + 1]
            
            sw = np.sum(window_weights)
            if sw != 0:
                sc = np.sum(window_close * window_weights)
                out[i] = sc / sw
        
        return pd.Series(out, index=close.index)

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        res = df.copy()
        atr = self._atr(df)
        range_atr = atr * self.mult
        ma = self._weighted_ma(df["close"])

        # 向量化计算 in_range（简化：检查当前价格是否在范围内）
        close_arr = df["close"].values
        ma_arr = ma.values
        range_atr_arr = range_atr.values
        
        # 简化的 in_range 计算：|close - ma| <= range_atr
        with np.errstate(invalid='ignore'):
            in_range_arr = np.abs(close_arr - ma_arr) <= range_atr_arr
        in_range = pd.Series(in_range_arr, index=df.index)

        # 向量化计算 trend
        trend = np.zeros(len(df), dtype=int)
        trend[close_arr > ma_arr] = 1
        trend[close_arr < ma_arr] = -1
        # NaN 位置设为 0
        trend[np.isnan(ma_arr)] = 0
        trend = pd.Series(trend, index=df.index)

        osc = np.where(
            pd.isna(range_atr) | (range_atr == 0) | pd.isna(ma),
            np.nan,
            100 * (df["close"] - ma) / range_atr,
        )
        res["rang_osc"] = osc
        res["rang_ma"] = ma
        res["rang_upper"] = ma + range_atr
        res["rang_lower"] = ma - range_atr
        res["rang_in_range"] = in_range.astype(int)
        res["rang_trend"] = trend
        res["rang_break_up"] = ((df["close"] > ma + range_atr).fillna(False)).astype(int)
        res["rang_break_down"] = ((df["close"] < ma - range_atr).fillna(False)).astype(int)
        res["rang_range_width"] = range_atr
        return res


class KDJ:
    """KDJ 指标。参数来自 Kdj.txt: rsv_length->n, k_smooth->m1, d_smooth->m2"""

    def __init__(
        self,
        n: Optional[int] = None,
        m1: Optional[int] = None,
        m2: Optional[int] = None,
        config_dir: Optional[Path] = None,
    ):
        cfg = _load_kdj_config(config_dir)
        self.n = n if n is not None else cfg["rsv_length"]
        self.m1 = m1 if m1 is not None else cfg["k_smooth"]
        self.m2 = m2 if m2 is not None else cfg["d_smooth"]

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        res = df.copy()
        low_n = df["low"].rolling(window=self.n).min()
        high_n = df["high"].rolling(window=self.n).max()
        rsv = np.where(
            high_n == low_n,
            50.0,
            np.where(
                pd.isna(high_n) | pd.isna(low_n),
                np.nan,
                100 * (df["close"] - low_n) / (high_n - low_n),
            ),
        )
        rsv = pd.Series(rsv, index=df.index)
        k = rsv.ewm(span=self.m1, adjust=False).mean()
        d = k.ewm(span=self.m2, adjust=False).mean()
        j = 3 * k - 2 * d
        res["kdj_k"] = k
        res["kdj_d"] = d
        res["kdj_j"] = j
        res["kdj_cross_up"] = ((k > d) & (k.shift(1) <= d.shift(1))).astype(int)
        res["kdj_cross_down"] = ((k < d) & (k.shift(1) >= d.shift(1))).astype(int)
        res["kdj_overbought"] = ((k > 80) & (d > 80)).astype(int)
        res["kdj_oversold"] = ((k < 20) & (d < 20)).astype(int)
        return res


def add_range_kdj_features(
    df: pd.DataFrame,
    kdj_params: Optional[Dict[str, int]] = None,
    rang_params: Optional[Dict[str, float]] = None,
    config_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """同时添加 Range 和 KDJ。可选覆盖 Kdj.txt/Rang.txt 参数。"""
    kw_r = {}
    if rang_params:
        if "range_period" in rang_params:
            kw_r["length"] = rang_params["range_period"]
        if "range_multiplier" in rang_params:
            kw_r["mult"] = rang_params["range_multiplier"]
    if not kw_r:
        kw_r["config_dir"] = config_dir
    rang = RangeOscillator(**kw_r) if kw_r else RangeOscillator(config_dir=config_dir)

    kw_k = {}
    if kdj_params:
        if "rsv_length" in kdj_params:
            kw_k["n"] = kdj_params["rsv_length"]
        if "k_smooth" in kdj_params:
            kw_k["m1"] = kdj_params["k_smooth"]
        if "d_smooth" in kdj_params:
            kw_k["m2"] = kdj_params["d_smooth"]
    if not kw_k:
        kw_k["config_dir"] = config_dir
    kdj = KDJ(**kw_k) if kw_k else KDJ(config_dir=config_dir)

    df = rang.calculate(df)
    df = kdj.calculate(df)
    return df
