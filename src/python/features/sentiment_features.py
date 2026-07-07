"""
情绪特征工程模块（Step 6.5）。

将情绪数据合并到 15m 特征集，并生成派生特征。

情绪数据源：
  - Fear & Greed Index (日级, Alternative.me) → 前向填充到 15m
  - 新闻情绪评分 (15min, CryptoCompare + VADER)
  - 新闻量 (15min, CryptoCompare)
  - 社交量近似 (15min)
  - 派生: sentiment_spike, sentiment_change_rate, news_volume_spike

用法：
  from src.python.features.sentiment_features import (
      merge_sentiment_to_15m,
      add_sentiment_derived,
      load_and_prepare_sentiment,
  )
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


# ───────────────────────────────────────────────────
# 核心特征列
# ───────────────────────────────────────────────────

SENTIMENT_RAW_COLS = [
    "fear_greed_index",
    "news_volume_15m",
    "sentiment_score_15m",
    "social_volume_15m",
]

SENTIMENT_DERIVED_COLS = [
    "sentiment_spike",
    "sentiment_change_rate",
    "fng_state",
    "sentiment_direction",
    "news_volume_ma20",
    "news_volume_spike",
    "sentiment_ma20",
    "sentiment_std20",
    "fng_ma7",
    "fng_change_1d",
]

ALL_SENTIMENT_COLS = SENTIMENT_RAW_COLS + SENTIMENT_DERIVED_COLS


# ───────────────────────────────────────────────────
# 合并函数
# ───────────────────────────────────────────────────

def merge_sentiment_to_15m(
    features_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
    timestamp_col: str = "timestamp",
) -> pd.DataFrame:
    """
    将情绪数据合并到主 15m 特征集。

    Parameters
    ----------
    features_df : pd.DataFrame
        现有 15m 特征（KDJ, MACD, RSI, OB 特征等）。
    sentiment_df : pd.DataFrame
        情绪数据（fear_greed_index, news_volume_15m, sentiment_score_15m 等）。
    timestamp_col : str
        时间戳列名。

    Returns
    -------
    pd.DataFrame
        合并后的特征集（left join，前向填充缺失值）。
    """
    features = features_df.copy()
    sentiment = sentiment_df.copy()

    # 确保 timestamp 是 datetime，且统一为 tz-naive ns
    features[timestamp_col] = pd.to_datetime(features[timestamp_col]).dt.tz_localize(None).astype("datetime64[ns]")
    sentiment[timestamp_col] = pd.to_datetime(sentiment[timestamp_col]).dt.tz_localize(None).astype("datetime64[ns]")

    # 去除 sentiment 中的重复时间戳（保留最后一条）
    sentiment = sentiment.drop_duplicates(subset=[timestamp_col], keep="last")

    # 对齐：用 merge_asof 按最近时间匹配（情绪数据可能不完全对齐 K 线边界）
    features = features.sort_values(timestamp_col)
    sentiment = sentiment.sort_values(timestamp_col)

    # 确定要合并的列（避免重复）
    merge_cols = [timestamp_col] + [
        c for c in SENTIMENT_RAW_COLS if c in sentiment.columns
    ]
    # 加入已有的派生列
    for c in ["sentiment_spike", "sentiment_change_rate"]:
        if c in sentiment.columns:
            merge_cols.append(c)

    sentiment_subset = sentiment[merge_cols].copy()

    merged = pd.merge_asof(
        features,
        sentiment_subset,
        on=timestamp_col,
        direction="backward",  # 用最近的过去数据（不泄漏未来）
        tolerance=pd.Timedelta("30min"),  # 最多容忍 30 分钟差距
    )

    # 前向填充（如果某 15min 无情绪数据）
    cols_to_fill = [c for c in SENTIMENT_RAW_COLS if c in merged.columns]
    merged[cols_to_fill] = merged[cols_to_fill].ffill()

    # 对仍缺失的值填默认值
    defaults = {
        "fear_greed_index": 50,  # 中性
        "news_volume_15m": 0,
        "sentiment_score_15m": 0.0,
        "social_volume_15m": 0,
        "sentiment_spike": False,
        "sentiment_change_rate": 0.0,
    }
    for col, default in defaults.items():
        if col in merged.columns:
            merged[col] = merged[col].fillna(default)

    return merged


# ───────────────────────────────────────────────────
# 派生特征
# ───────────────────────────────────────────────────

def add_sentiment_derived(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加派生情绪特征。

    新增列：
      - fng_state: 分类 (extreme_fear / fear / neutral / greed / extreme_greed)
      - sentiment_direction: 方向 (+1 / 0 / -1)
      - news_volume_ma20: 新闻量 20 期移动平均
      - news_volume_spike: 新闻量 > 1.5 * MA20
      - sentiment_ma20: 情绪评分 20 期移动平均
      - sentiment_std20: 情绪评分 20 期滚动标准差
      - fng_ma7: FGI 7 日移动平均
      - fng_change_1d: FGI 日变化量
    """
    result = df.copy()

    # 1. 恐慌/贪婪状态（分类编码）
    if "fear_greed_index" in result.columns:
        result["fng_state"] = pd.cut(
            result["fear_greed_index"].astype(float),
            bins=[-1, 20, 40, 60, 80, 101],
            labels=[0, 1, 2, 3, 4],  # 0=极度恐慌 ... 4=极度贪婪
        ).astype(float)

        # FGI 7 日移动平均（96 根 15min K线 ≈ 1 天）
        result["fng_ma7"] = (
            result["fear_greed_index"]
            .rolling(window=96 * 7, min_periods=96)
            .mean()
        )

        # FGI 日变化量
        result["fng_change_1d"] = (
            result["fear_greed_index"].astype(float)
            - result["fear_greed_index"].shift(96).astype(float)
        )

    # 2. 情绪方向
    if "sentiment_change_rate" in result.columns:
        result["sentiment_direction"] = result["sentiment_change_rate"].apply(
            lambda x: 1 if x > 0.05 else (-1 if x < -0.05 else 0)
        )

    # 3. 新闻量统计
    if "news_volume_15m" in result.columns:
        result["news_volume_ma20"] = (
            result["news_volume_15m"].rolling(window=20, min_periods=5).mean()
        )
        result["news_volume_spike"] = (
            result["news_volume_15m"] > 1.5 * result["news_volume_ma20"]
        ).astype(float)

    # 4. 情绪评分滚动统计
    if "sentiment_score_15m" in result.columns:
        result["sentiment_ma20"] = (
            result["sentiment_score_15m"].rolling(window=20, min_periods=5).mean()
        )
        result["sentiment_std20"] = (
            result["sentiment_score_15m"].rolling(window=20, min_periods=5).std()
        )

    # 填充 NaN
    for col in SENTIMENT_DERIVED_COLS:
        if col in result.columns:
            result[col] = result[col].fillna(0)

    return result


# ───────────────────────────────────────────────────
# 便捷加载函数
# ───────────────────────────────────────────────────

def load_and_prepare_sentiment(
    fng_history_path: str = "data/sentiment/fear_greed_history_daily.parquet",
    news_history_path: str = "data/sentiment/news_sentiment_history_15m.parquet",
    realtime_path: str = "data/sentiment/sentiment_realtime_15m.parquet",
) -> pd.DataFrame:
    """
    加载并合并历史 FGI + 历史新闻情绪 + 实时 15m 情绪数据。

    数据优先级：
      1. 历史新闻情绪（15min 粒度，CryptoCompare 回填）— 训练主力
      2. 实时数据（15min 粒度，收集时间短但最新）
      3. FGI 历史（日级，2018 至今）— 补充 fear_greed_index

    Parameters
    ----------
    fng_history_path : str
        Fear & Greed 日级历史文件。
    news_history_path : str
        CryptoCompare 历史新闻情绪 15min 文件（由 pull_sentiment_news_history.py 生成）。
    realtime_path : str
        实时 15min 情绪数据文件。

    Returns
    -------
    pd.DataFrame
        合并后的情绪数据（以 15min 粒度为主，FGI 前向填充）。
    """
    dfs = []

    # 1. 加载历史新闻情绪（15min 粒度，CryptoCompare + VADER 回填）
    news_path = Path(news_history_path)
    if news_path.exists():
        news_hist = pd.read_parquet(news_path)
        news_hist["timestamp"] = pd.to_datetime(news_hist["timestamp"])
        # 统一 tz-naive
        if news_hist["timestamp"].dt.tz is not None:
            news_hist["timestamp"] = news_hist["timestamp"].dt.tz_localize(None)
        dfs.append(news_hist)
        print(f"  加载历史新闻情绪: {len(news_hist)} 条 "
              f"({news_hist['timestamp'].min().date()} ~ {news_hist['timestamp'].max().date()})")

    # 2. 加载实时数据（有完整 6 列，但记录少）
    rt_path = Path(realtime_path)
    if rt_path.exists():
        rt = pd.read_parquet(rt_path)
        rt["timestamp"] = pd.to_datetime(rt["timestamp"])
        if rt["timestamp"].dt.tz is not None:
            rt["timestamp"] = rt["timestamp"].dt.tz_localize(None)

        # 如果有历史新闻数据，只追加历史之后的实时记录（避免重叠）
        if dfs:
            latest_hist = dfs[0]["timestamp"].max()
            rt = rt[rt["timestamp"] > latest_hist]

        if len(rt) > 0:
            dfs.append(rt)
            print(f"  加载实时情绪: {len(rt)} 条")

    # 3. 加载 FGI 历史（日级，补充 fear_greed_index 到所有 15min 行）
    fng_path = Path(fng_history_path)
    if fng_path.exists():
        fng = pd.read_parquet(fng_path)
        fng["timestamp"] = pd.to_datetime(fng["timestamp"])
        if fng["timestamp"].dt.tz is not None:
            fng["timestamp"] = fng["timestamp"].dt.tz_localize(None)

        # 只保留 FGI 列
        fng_cols = ["timestamp", "fear_greed_index"]
        if "fng_classification" in fng.columns:
            fng_cols.append("fng_classification")
        fng = fng[fng_cols]

        if len(fng) > 0:
            print(f"  加载 FGI 历史: {len(fng)} 条 "
                  f"({fng['timestamp'].min().date()} ~ {fng['timestamp'].max().date()})")

    if not dfs:
        # 没有新闻数据也没有实时数据，只有 FGI → 返回 FGI 日级数据
        if fng_path.exists():
            print("  [INFO] 仅有 FGI 历史（日级），无 15min 新闻情绪")
            return fng
        print("  [WARN] 无情绪数据可加载")
        return pd.DataFrame(columns=["timestamp"] + SENTIMENT_RAW_COLS)

    # 合并新闻+实时数据
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    # 将 FGI 日级数据 merge 到 15min 数据上
    if fng_path.exists() and len(fng) > 0:
        # merge_asof: 每条 15min 记录匹配最近的 FGI 日数据（backward）
        fng_for_merge = fng[["timestamp", "fear_greed_index"]].copy()
        fng_for_merge = fng_for_merge.sort_values("timestamp")
        combined = combined.sort_values("timestamp")

        # 先删掉 combined 中可能已有的 FGI 列（来自实时数据）
        if "fear_greed_index" in combined.columns:
            combined = combined.drop(columns=["fear_greed_index"])

        combined = pd.merge_asof(
            combined,
            fng_for_merge,
            on="timestamp",
            direction="backward",
        )

    # 前向填充 FGI（日级 → 15min）
    if "fear_greed_index" in combined.columns:
        combined["fear_greed_index"] = combined["fear_greed_index"].ffill()
        combined["fear_greed_index"] = combined["fear_greed_index"].fillna(50)

    # 确保所有必要列存在
    for col, default in [
        ("news_volume_15m", 0),
        ("sentiment_score_15m", 0.0),
        ("social_volume_15m", 0),
        ("fear_greed_index", 50),
    ]:
        if col not in combined.columns:
            combined[col] = default

    return combined


def get_sentiment_feature_names() -> List[str]:
    """返回所有情绪特征名列表。"""
    return ALL_SENTIMENT_COLS.copy()
