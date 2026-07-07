"""
Range Oscillator (Zeiierman) and KDJ Indicators - Python Implementation
适用于 FreqAI 策略或技术分析
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


class RangeOscillator:
    """
    Range Oscillator (Zeiierman) 指标
    
    参数:
        length: 最小范围长度，默认 50
        mult: 范围宽度乘数，默认 2.0
        levels: 热力图级别数，默认 2
        heat_thresh: 每级别最小触及次数，默认 1
    """
    
    def __init__(
        self,
        length: int = 50,
        mult: float = 2.0,
        levels: int = 2,
        heat_thresh: int = 1
    ):
        self.length = length
        self.mult = mult
        self.levels = levels
        self.heat_thresh = heat_thresh
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 2000) -> pd.Series:
        """计算 ATR (Average True Range)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 使用可用的最大周期计算 ATR
        actual_period = min(period, len(df))
        atr = tr.rolling(window=actual_period).mean()
        
        # 如果周期太大，回退到较小周期
        if atr.isna().all():
            atr = tr.rolling(window=min(200, len(df))).mean()
        
        return atr.fillna(method='bfill')
    
    def calculate_weighted_ma(self, close: pd.Series) -> pd.Series:
        """计算加权移动平均"""
        ma_values = []
        
        for i in range(len(close)):
            if i < self.length - 1:
                ma_values.append(np.nan)
                continue
            
            sum_weighted_close = 0.0
            sum_weights = 0.0
            
            for j in range(self.length):
                idx = i - j
                if idx < 1:
                    break
                
                delta = abs(close.iloc[idx] - close.iloc[idx - 1])
                w = delta / close.iloc[idx - 1] if close.iloc[idx - 1] != 0 else 0
                
                sum_weighted_close += close.iloc[idx] * w
                sum_weights += w
            
            ma = sum_weighted_close / sum_weights if sum_weights != 0 else np.nan
            ma_values.append(ma)
        
        return pd.Series(ma_values, index=close.index)
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 Range Oscillator
        
        返回:
            包含以下列的 DataFrame:
            - rang_osc: 振荡器值
            - rang_ma: 加权移动平均
            - rang_upper: 上轨
            - rang_lower: 下轨
            - rang_in_range: 是否在范围内
            - rang_trend: 趋势方向 (1=上升, -1=下降)
            - rang_break_up: 向上突破
            - rang_break_down: 向下突破
        """
        result = df.copy()
        
        # 计算 ATR
        atr = self.calculate_atr(df)
        range_atr = atr * self.mult
        
        # 计算加权移动平均
        ma = self.calculate_weighted_ma(df['close'])
        
        # 检查是否在范围内
        in_range_list = []
        for i in range(len(df)):
            if i < self.length - 1 or pd.isna(ma.iloc[i]):
                in_range_list.append(False)
                continue
            
            distances = []
            for j in range(self.length):
                idx = i - j
                if idx >= 0 and not pd.isna(ma.iloc[i]):
                    distances.append(abs(df['close'].iloc[idx] - ma.iloc[i]))
            
            max_dist = max(distances) if distances else 0
            in_range = max_dist <= range_atr.iloc[i] if not pd.isna(range_atr.iloc[i]) else False
            in_range_list.append(in_range)
        
        in_range = pd.Series(in_range_list, index=df.index)
        
        # 计算趋势方向
        trend_dir = pd.Series(0, index=df.index)
        for i in range(len(df)):
            if pd.isna(ma.iloc[i]):
                trend_dir.iloc[i] = 0
            elif df['close'].iloc[i] > ma.iloc[i]:
                trend_dir.iloc[i] = 1
            elif df['close'].iloc[i] < ma.iloc[i]:
                trend_dir.iloc[i] = -1
            else:
                trend_dir.iloc[i] = trend_dir.iloc[i-1] if i > 0 else 0
        
        # 计算振荡器
        osc = pd.Series(index=df.index, dtype=float)
        for i in range(len(df)):
            if pd.isna(range_atr.iloc[i]) or range_atr.iloc[i] == 0 or pd.isna(ma.iloc[i]):
                osc.iloc[i] = np.nan
            else:
                osc.iloc[i] = 100 * (df['close'].iloc[i] - ma.iloc[i]) / range_atr.iloc[i]
        
        # 突破检测
        break_up = (df['close'] > ma + range_atr).fillna(False)
        break_down = (df['close'] < ma - range_atr).fillna(False)
        
        # 添加结果到 DataFrame
        result['rang_osc'] = osc
        result['rang_ma'] = ma
        result['rang_upper'] = ma + range_atr
        result['rang_lower'] = ma - range_atr
        result['rang_in_range'] = in_range
        result['rang_trend'] = trend_dir
        result['rang_break_up'] = break_up.astype(int)
        result['rang_break_down'] = break_down.astype(int)
        result['rang_range_width'] = range_atr
        
        return result


class KDJ:
    """
    KDJ 指标 (Stochastic Oscillator with J line)
    
    参数:
        n: K 周期，默认 9
        m1: K 平滑周期，默认 3
        m2: D 平滑周期，默认 3
    """
    
    def __init__(self, n: int = 9, m1: int = 3, m2: int = 3):
        self.n = n
        self.m1 = m1
        self.m2 = m2
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 KDJ 指标
        
        返回:
            包含以下列的 DataFrame:
            - kdj_k: K 值
            - kdj_d: D 值
            - kdj_j: J 值
            - kdj_cross_up: K 线上穿 D 线
            - kdj_cross_down: K 线下穿 D 线
            - kdj_overbought: 超买信号 (K>80, D>80)
            - kdj_oversold: 超卖信号 (K<20, D<20)
        """
        result = df.copy()
        
        # 计算 RSV (Raw Stochastic Value)
        low_n = df['low'].rolling(window=self.n).min()
        high_n = df['high'].rolling(window=self.n).max()
        
        rsv = pd.Series(index=df.index, dtype=float)
        for i in range(len(df)):
            if pd.isna(high_n.iloc[i]) or pd.isna(low_n.iloc[i]):
                rsv.iloc[i] = np.nan
            elif high_n.iloc[i] == low_n.iloc[i]:
                rsv.iloc[i] = 50.0  # 避免除零
            else:
                rsv.iloc[i] = 100 * (df['close'].iloc[i] - low_n.iloc[i]) / (high_n.iloc[i] - low_n.iloc[i])
        
        # 计算 K 值 (RSV 的移动平均)
        k = rsv.ewm(span=self.m1, adjust=False).mean()
        
        # 计算 D 值 (K 的移动平均)
        d = k.ewm(span=self.m2, adjust=False).mean()
        
        # 计算 J 值
        j = 3 * k - 2 * d
        
        # 交叉信号
        cross_up = ((k > d) & (k.shift(1) <= d.shift(1))).astype(int)
        cross_down = ((k < d) & (k.shift(1) >= d.shift(1))).astype(int)
        
        # 超买超卖信号
        overbought = ((k > 80) & (d > 80)).astype(int)
        oversold = ((k < 20) & (d < 20)).astype(int)
        
        # 添加结果
        result['kdj_k'] = k
        result['kdj_d'] = d
        result['kdj_j'] = j
        result['kdj_cross_up'] = cross_up
        result['kdj_cross_down'] = cross_down
        result['kdj_overbought'] = overbought
        result['kdj_oversold'] = oversold
        
        return result


def add_range_kdj_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    便捷函数：同时添加 Range Oscillator 和 KDJ 指标
    
    参数:
        df: 包含 'open', 'high', 'low', 'close', 'volume' 的 DataFrame
    
    返回:
        添加了所有指标的 DataFrame
    """
    # 初始化指标
    rang = RangeOscillator(length=50, mult=2.0)
    kdj = KDJ(n=9, m1=3, m2=3)
    
    # 计算指标
    df = rang.calculate(df)
    df = kdj.calculate(df)
    
    return df


# 使用示例
if __name__ == '__main__':
    # 创建测试数据
    dates = pd.date_range('2024-01-01', periods=200, freq='1h')
    np.random.seed(42)
    
    # 模拟价格数据
    close = 100 + np.cumsum(np.random.randn(200) * 0.5)
    high = close + np.random.rand(200) * 2
    low = close - np.random.rand(200) * 2
    open_price = close + np.random.randn(200) * 0.3
    volume = np.random.randint(1000, 10000, 200)
    
    test_df = pd.DataFrame({
        'date': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    
    # 计算指标
    result = add_range_kdj_features(test_df)
    
    # 显示结果
    print("Range Oscillator 指标:")
    print(result[['date', 'close', 'rang_osc', 'rang_ma', 'rang_trend', 'rang_break_up', 'rang_break_down']].tail(10))
    
    print("\nKDJ 指标:")
    print(result[['date', 'close', 'kdj_k', 'kdj_d', 'kdj_j', 'kdj_cross_up', 'kdj_cross_down']].tail(10))
    
    # 特征统计
    print("\n指标列:")
    kdj_cols = [col for col in result.columns if col.startswith('kdj_')]
    rang_cols = [col for col in result.columns if col.startswith('rang_')]
    print(f"KDJ 特征: {kdj_cols}")
    print(f"Range Oscillator 特征: {rang_cols}")