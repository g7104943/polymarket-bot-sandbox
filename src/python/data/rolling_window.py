"""
滚动窗口训练数据策略（方案 A）：
- 训练集：最近 train_days 天（默认 90 天）
- 验证集：最近 val_days 天（默认 30 天，与训练集不重叠）
- 每月丢弃最老 30 天、加入最新 30 天后重训

用法：
    from src.python.data.rolling_window import get_rolling_window_data
    train_df, val_df = get_rolling_window_data(full_df, end_date=datetime.now())

与方案 B（分层采样 2 年）互补：
- 方案 A 负责「新鲜度」—— 始终匹配当前市场状态
- 方案 B 负责「覆盖面」—— 保留罕见事件，作为 Ensemble 长期成员
"""

import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple


def get_rolling_window_data(
    df: pd.DataFrame,
    end_date: Optional[datetime] = None,
    train_days: int = 90,
    val_days: int = 30,
    timestamp_col: str = "timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    从完整 DataFrame 中截取滚动窗口的训练集与验证集。

    时间线: |--- 更早 ---|--- train_days ---|--- val_days ---| end_date
                        train_start      val_start       end_date

    参数:
        df: 完整历史数据，需含时间列
        end_date: 窗口右端（默认=现在 UTC）
        train_days: 训练窗口天数（默认 90）
        val_days: 验证窗口天数（默认 30）
        timestamp_col: 时间列名

    返回:
        (train_df, val_df)
    """
    if end_date is None:
        end_date = datetime.now(timezone.utc)
    elif end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    val_start = end_date - timedelta(days=val_days)
    train_start = val_start - timedelta(days=train_days)

    # 转为毫秒时间戳边界
    end_ms = int(end_date.timestamp() * 1000)
    val_start_ms = int(val_start.timestamp() * 1000)
    train_start_ms = int(train_start.timestamp() * 1000)

    # 获取时间列
    if df[timestamp_col].dtype in ("int64", "float64"):
        ts = df[timestamp_col]
    else:
        ts = pd.to_datetime(df[timestamp_col], utc=True).astype("int64") // 10**6

    train_mask = (ts >= train_start_ms) & (ts < val_start_ms)
    val_mask = (ts >= val_start_ms) & (ts <= end_ms)

    train_df = df.loc[train_mask].copy().reset_index(drop=True)
    val_df = df.loc[val_mask].copy().reset_index(drop=True)

    return train_df, val_df


def get_rolling_window_info(
    end_date: Optional[datetime] = None,
    train_days: int = 90,
    val_days: int = 30,
) -> dict:
    """返回滚动窗口的日期信息，方便日志打印。"""
    if end_date is None:
        end_date = datetime.now(timezone.utc)
    elif end_date.tzinfo is None:
        end_date = end_date.replace(tzinfo=timezone.utc)

    val_start = end_date - timedelta(days=val_days)
    train_start = val_start - timedelta(days=train_days)

    return {
        "train_start": train_start.strftime("%Y-%m-%d"),
        "train_end": val_start.strftime("%Y-%m-%d"),
        "val_start": val_start.strftime("%Y-%m-%d"),
        "val_end": end_date.strftime("%Y-%m-%d"),
        "train_days": train_days,
        "val_days": val_days,
        "total_days": train_days + val_days,
    }


def should_retrain_monthly(last_trained_date: Optional[str] = None) -> bool:
    """
    判断是否应该触发月度重训练。

    规则：当前月份与上次训练月份不同时触发。
    """
    now = datetime.now(timezone.utc)
    if last_trained_date is None:
        return True
    try:
        last = pd.to_datetime(last_trained_date, utc=True)
        return (now.year, now.month) != (last.year, last.month)
    except Exception:
        return True
