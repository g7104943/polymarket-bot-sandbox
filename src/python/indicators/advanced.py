"""
高级技术指标：Stochastic, CCI, MFI, VWAP, Donchian Channel, Hurst Exponent 等
"""

import pandas as pd
import numpy as np
from typing import Optional


def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """
    计算随机振荡器 (Stochastic Oscillator)。
    添加 stoch_k, stoch_d
    """
    res = df.copy()
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    # %K = (close - low_N) / (high_N - low_N) * 100
    low_n = low.rolling(window=k_period).min()
    high_n = high.rolling(window=k_period).max()
    stoch_k = 100 * (close - low_n) / (high_n - low_n).replace(0, np.nan)
    
    # %D = %K 的移动平均
    stoch_d = stoch_k.rolling(window=d_period).mean()
    
    res["stoch_k"] = stoch_k
    res["stoch_d"] = stoch_d
    
    return res


def calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    计算商品通道指数 (Commodity Channel Index, CCI)。
    添加 cci
    """
    res = df.copy()
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    # 典型价格
    tp = (high + low + close) / 3
    
    # 典型价格的移动平均
    tp_ma = tp.rolling(window=period).mean()
    
    # 平均偏差
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean(), raw=False)
    
    # CCI = (TP - TP_MA) / (0.015 * MAD)
    res["cci"] = (tp - tp_ma) / (0.015 * mad.replace(0, np.nan))
    
    return res


def calculate_mfi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    计算资金流量指数 (Money Flow Index, MFI)。
    添加 mfi
    """
    res = df.copy()
    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df["volume"]
    
    # 典型价格
    tp = (high + low + close) / 3
    
    # 原始资金流
    raw_money_flow = tp * volume
    
    # 正/负资金流
    positive_flow = raw_money_flow.copy()
    negative_flow = raw_money_flow.copy()
    
    # 价格上涨时为正，下跌时为负
    price_change = tp.diff()
    positive_flow[price_change <= 0] = 0
    negative_flow[price_change >= 0] = 0
    
    # 滚动求和
    positive_sum = positive_flow.rolling(window=period).sum()
    negative_sum = negative_flow.rolling(window=period).sum()
    
    # MFI = 100 - (100 / (1 + 正资金流 / 负资金流))
    money_ratio = positive_sum / negative_sum.replace(0, np.nan)
    res["mfi"] = 100 - (100 / (1 + money_ratio))
    
    return res


def calculate_vwap(df: pd.DataFrame, reset_period: Optional[str] = None) -> pd.DataFrame:
    """
    计算量加权平均价 (Volume Weighted Average Price, VWAP)。
    添加 vwap
    
    参数:
        reset_period: 重置周期，如 'D' (每日), 'H' (每小时), None (不重置)
    """
    res = df.copy()
    high = df["high"]
    low = df["low"]
    close = df["close"]
    volume = df["volume"]
    
    # 典型价格
    tp = (high + low + close) / 3
    
    # 价格 * 成交量
    pv = tp * volume
    
    if reset_period:
        # 按周期重置（如每日重置）
        if "timestamp" in res.columns:
            dt = pd.to_datetime(res["timestamp"], unit="ms", utc=True)
        elif "date" in res.columns:
            dt = pd.to_datetime(res["date"], utc=True)
        else:
            reset_period = None
        
        if reset_period:
            # 按周期分组计算累计 VWAP
            # 使用 tz_localize(None) 移除时区信息，避免警告
            # dt 是 Series，需要使用 .dt 访问器
            if dt.dt.tz is not None:
                dt_naive = dt.dt.tz_localize(None)
            else:
                dt_naive = dt
            # 使用 .dt.to_period() 而不是 .to_period()
            pv_cumsum = pv.groupby(dt_naive.dt.to_period(reset_period)).cumsum()
            vol_cumsum = volume.groupby(dt_naive.dt.to_period(reset_period)).cumsum()
            res["vwap"] = pv_cumsum / vol_cumsum.replace(0, np.nan)
        else:
            # 全局累计
            res["vwap"] = pv.cumsum() / volume.cumsum().replace(0, np.nan)
    else:
        # 全局累计（不重置）
        res["vwap"] = pv.cumsum() / volume.cumsum().replace(0, np.nan)
    
    # VWAP 相对价格位置
    res["close_vs_vwap"] = (close - res["vwap"]) / res["vwap"].replace(0, np.nan) * 100
    
    return res


def calculate_donchian_channel(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """
    计算唐奇安通道 (Donchian Channel)。
    添加 dc_upper, dc_lower, dc_middle, dc_width, dc_position
    """
    res = df.copy()
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    # 上轨：N 周期最高价
    dc_upper = high.rolling(window=period).max()
    
    # 下轨：N 周期最低价
    dc_lower = low.rolling(window=period).min()
    
    # 中轨
    dc_middle = (dc_upper + dc_lower) / 2
    
    # 通道宽度
    dc_width = dc_upper - dc_lower
    
    # 价格在通道中的位置 (0-1)
    dc_position = (close - dc_lower) / dc_width.replace(0, np.nan)
    
    res["dc_upper"] = dc_upper
    res["dc_lower"] = dc_lower
    res["dc_middle"] = dc_middle
    res["dc_width"] = dc_width
    res["dc_position"] = dc_position
    
    return res


def calculate_hurst_exponent(series: pd.Series, max_lag: int = 20, window: int = 100) -> pd.Series:
    """
    计算 Hurst 指数（趋势持久性）。
    
    参数:
        series: 价格序列
        max_lag: 最大滞后周期
        window: 滚动窗口大小
    
    返回:
        Hurst 指数 Series（0-1之间，>0.5 表示趋势，<0.5 表示均值回归）
    """
    def _hurst(ts, lags):
        """计算单个时间窗口的 Hurst 指数（使用 R/S 方法）"""
        if len(ts) < lags + 1:
            return np.nan
        
        # 使用 R/S (Rescaled Range) 方法
        lags_to_test = [lag for lag in range(2, min(lags + 1, len(ts) // 2)) if lag < len(ts)]
        if len(lags_to_test) < 2:
            return np.nan
        
        rs_values = []
        lag_values = []
        
        for lag in lags_to_test:
            # 将序列分成多个子序列
            n_chunks = len(ts) // lag
            if n_chunks < 1:
                continue
            
            rs_list = []
            for i in range(n_chunks):
                chunk = ts.iloc[i * lag:(i + 1) * lag]
                if len(chunk) < 2:
                    continue
                
                # 计算均值
                mean_chunk = chunk.mean()
                
                # 计算累积偏差
                cumdev = (chunk - mean_chunk).cumsum()
                
                # R: 范围
                R = cumdev.max() - cumdev.min()
                
                # S: 标准差
                S = chunk.std()
                
                if S > 0:
                    rs_list.append(R / S)
            
            if len(rs_list) > 0:
                rs_values.append(np.mean(rs_list))
                lag_values.append(lag)
        
        if len(rs_values) < 2:
            return np.nan
        
        # 线性回归 log(R/S) vs log(lag)
        log_rs = np.log(rs_values)
        log_lag = np.log(lag_values)
        
        # 简单线性回归
        x_mean = np.mean(log_lag)
        y_mean = np.mean(log_rs)
        
        numerator = np.sum((log_lag - x_mean) * (log_rs - y_mean))
        denominator = np.sum((log_lag - x_mean) ** 2)
        
        if denominator == 0:
            return np.nan
        
        hurst = numerator / denominator
        return np.clip(hurst, 0, 1)  # 限制在 0-1 之间
    
    # 滚动窗口计算
    hurst_values = []
    window = min(window, len(series) // 2)  # 确保窗口不会太大
    
    for i in range(len(series)):
        if i < window:
            hurst_values.append(np.nan)
        else:
            window_series = series.iloc[i - window:i + 1]
            h = _hurst(window_series, max_lag)
            hurst_values.append(h if not np.isnan(h) else np.nan)
    
    return pd.Series(hurst_values, index=series.index)


def calculate_volume_delta(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    """
    计算成交量 Delta：当前 volume - rolling mean。
    添加 volume_delta
    """
    res = df.copy()
    volume = df["volume"]
    volume_ma = volume.rolling(window=period).mean()
    res["volume_delta"] = volume - volume_ma
    res["volume_delta_pct"] = (res["volume_delta"] / volume_ma.replace(0, np.nan)) * 100
    return res
