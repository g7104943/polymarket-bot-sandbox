"""
分层采样器（方案 B）：对长期历史数据按时间距离做分层采样，
保留罕见事件（黑天鹅、监管冲击），同时减少计算成本。

用法：
    from src.python.data.stratified_sampler import stratified_sampling
    df_sampled = stratified_sampling(df, end_date=datetime.now())

与方案 A（滚动 90 天）互补：
- 方案 A 负责「新鲜度」—— 用最近 90 天训练，匹配当前市场
- 方案 B 负责「覆盖面」—— 保留 2 年尾部事件，作为 Ensemble 长期模型
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple


def stratified_sampling(
    df: pd.DataFrame,
    end_date: Optional[datetime] = None,
    cutoff_days: Optional[List[int]] = None,
    sample_rates: Optional[List[float]] = None,
    timestamp_col: str = "timestamp",
    seed: int = 42,
) -> pd.DataFrame:
    """
    按时间距离做分层采样。

    参数:
        df: 输入 DataFrame，需含时间列
        end_date: 采样截止日（默认=现在 UTC）
        cutoff_days: 分层边界（天），如 [90, 180, 365] 表示 4 层：
            0-90 天 / 90-180 天 / 180-365 天 / 365 天以上
        sample_rates: 每层的采样率，如 [1.0, 0.5, 0.25, 0.14]
            第一层（最近）全采样，越远采样越稀疏
        timestamp_col: 时间列名（毫秒时间戳或 datetime）
        seed: 随机种子

    返回:
        采样后的 DataFrame（按时间排序）
    """
    if cutoff_days is None:
        cutoff_days = [90, 180, 365]
    if sample_rates is None:
        sample_rates = [1.0, 0.5, 0.25, 0.14]

    if len(sample_rates) != len(cutoff_days) + 1:
        raise ValueError(
            f"sample_rates 长度 ({len(sample_rates)}) 须等于 cutoff_days 长度 + 1 ({len(cutoff_days) + 1})"
        )

    if end_date is None:
        end_date = datetime.now(timezone.utc)
    elif end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    # 将 cutoff_days 转为毫秒时间戳边界
    boundaries_ms = []
    for days in cutoff_days:
        dt = end_date - timedelta(days=days)
        boundaries_ms.append(int(dt.timestamp() * 1000))

    # 获取时间列（支持毫秒时间戳或 datetime）
    if df[timestamp_col].dtype in ("int64", "float64"):
        ts = df[timestamp_col]
    else:
        ts = pd.to_datetime(df[timestamp_col], utc=True).astype("int64") // 10**6

    # 分层采样
    rng = np.random.RandomState(seed)
    result_indices = []

    # 第 0 层：最近 cutoff_days[0] 天 → sample_rates[0]
    mask_0 = ts >= boundaries_ms[0]
    idx_0 = df.index[mask_0]
    n_sample_0 = max(1, int(len(idx_0) * sample_rates[0]))
    if sample_rates[0] >= 1.0:
        result_indices.append(idx_0)
    else:
        result_indices.append(rng.choice(idx_0, size=n_sample_0, replace=False))

    # 中间层
    for i in range(len(cutoff_days) - 1):
        mask = (ts < boundaries_ms[i]) & (ts >= boundaries_ms[i + 1])
        idx = df.index[mask]
        if len(idx) == 0:
            continue
        rate = sample_rates[i + 1]
        n_sample = max(1, int(len(idx) * rate))
        if rate >= 1.0:
            result_indices.append(idx)
        else:
            result_indices.append(rng.choice(idx, size=n_sample, replace=False))

    # 最后一层：cutoff_days[-1] 天以前 → sample_rates[-1]
    mask_last = ts < boundaries_ms[-1]
    idx_last = df.index[mask_last]
    if len(idx_last) > 0:
        rate = sample_rates[-1]
        n_sample = max(1, int(len(idx_last) * rate))
        if rate >= 1.0:
            result_indices.append(idx_last)
        else:
            result_indices.append(rng.choice(idx_last, size=n_sample, replace=False))

    # 合并、排序
    all_idx = np.concatenate([np.asarray(idx) for idx in result_indices])
    all_idx = np.unique(all_idx)
    sampled = df.loc[all_idx].copy()

    sort_col = timestamp_col if timestamp_col in sampled.columns else sampled.columns[0]
    sampled = sampled.sort_values(sort_col).reset_index(drop=True)

    return sampled


def get_sampling_stats(
    df: pd.DataFrame,
    df_sampled: pd.DataFrame,
    end_date: Optional[datetime] = None,
    cutoff_days: Optional[List[int]] = None,
    timestamp_col: str = "timestamp",
) -> dict:
    """
    返回各层的原始/采样行数、采样率统计。

    返回:
        {
            "total_original": int,
            "total_sampled": int,
            "reduction_pct": float,
            "layers": [{"range": "0-90d", "original": N, "sampled": M, "rate": 0.xx}, ...]
        }
    """
    if cutoff_days is None:
        cutoff_days = [90, 180, 365]
    if end_date is None:
        end_date = datetime.now(timezone.utc)
    elif end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    boundaries_ms = [int((end_date - timedelta(days=d)).timestamp() * 1000) for d in cutoff_days]

    def _get_ts(frame):
        if frame[timestamp_col].dtype in ("int64", "float64"):
            return frame[timestamp_col]
        return pd.to_datetime(frame[timestamp_col], utc=True).astype("int64") // 10**6

    ts_orig = _get_ts(df)
    ts_samp = _get_ts(df_sampled)

    layers = []
    ranges = [f"0-{cutoff_days[0]}d"]
    for i in range(len(cutoff_days) - 1):
        ranges.append(f"{cutoff_days[i]}-{cutoff_days[i+1]}d")
    ranges.append(f">{cutoff_days[-1]}d")

    bounds_extended = [float("inf")] + boundaries_ms + [0]
    for i, label in enumerate(ranges):
        hi = bounds_extended[i]
        lo = bounds_extended[i + 1]
        if hi == float("inf"):
            mask_o = ts_orig >= lo
            mask_s = ts_samp >= lo
        else:
            mask_o = (ts_orig < hi) & (ts_orig >= lo)
            mask_s = (ts_samp < hi) & (ts_samp >= lo)
        n_o = int(mask_o.sum())
        n_s = int(mask_s.sum())
        layers.append({
            "range": label,
            "original": n_o,
            "sampled": n_s,
            "rate": round(n_s / n_o, 3) if n_o > 0 else 0.0,
        })

    return {
        "total_original": len(df),
        "total_sampled": len(df_sampled),
        "reduction_pct": round((1 - len(df_sampled) / len(df)) * 100, 1) if len(df) > 0 else 0.0,
        "layers": layers,
    }
