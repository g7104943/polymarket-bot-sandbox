#!/usr/bin/env python3
"""
一次性拉取历史情绪数据（用于回测和训练）。

数据源：
  1. Alternative.me Fear & Greed Index — 完全免费、无 key、全历史 2018 至今
     API: https://api.alternative.me/fng/?limit=0

用法：
  python scripts/pull_sentiment_history.py
  python scripts/pull_sentiment_history.py --output data/sentiment/fear_greed_history_daily.parquet

输出：
  data/sentiment/fear_greed_history_daily.parquet
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests


def pull_fear_greed_history(output_path: Path) -> pd.DataFrame:
    """
    拉取 Fear & Greed Index 全历史（2018 至今）。

    来源：Alternative.me（完全免费、无 API key、无速率限制）

    返回 DataFrame 列：
      - timestamp (datetime)
      - fear_greed_index (int, 0-100)
      - value_classification (str: Extreme Fear / Fear / Neutral / Greed / Extreme Greed)
    """
    print("拉取 Alternative.me Fear & Greed Index 全历史...")
    url = "https://api.alternative.me/fng/?limit=0"

    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        print(f"[ERROR] 请求失败: {e}")
        sys.exit(1)

    if "data" not in data:
        print(f"[ERROR] 响应格式异常: {list(data.keys())}")
        sys.exit(1)

    records = data["data"]
    df = pd.DataFrame(records)

    # 解析字段
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="s")
    df["fear_greed_index"] = df["value"].astype(int)
    df = df.rename(columns={"value_classification": "fng_classification"})
    df = df[["timestamp", "fear_greed_index", "fng_classification"]]

    # 按时间排序（从早到晚）
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"  总记录数: {len(df)}")
    print(f"  时间范围: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    print(f"  今日 FGI:  {df['fear_greed_index'].iloc[-1]} ({df['fng_classification'].iloc[-1]})")
    print(f"  已保存到:  {output_path}")

    # 打印分布统计
    print("\n  FGI 分布统计:")
    print(f"    均值:    {df['fear_greed_index'].mean():.1f}")
    print(f"    中位数:  {df['fear_greed_index'].median():.0f}")
    print(f"    标准差:  {df['fear_greed_index'].std():.1f}")
    print(f"    最小值:  {df['fear_greed_index'].min()}")
    print(f"    最大值:  {df['fear_greed_index'].max()}")

    # 打印各分类占比
    print("\n  分类分布:")
    for cls, count in df["fng_classification"].value_counts().items():
        pct = count / len(df) * 100
        print(f"    {cls:<15s}: {count:>5d} ({pct:.1f}%)")

    return df


def parse_args():
    p = argparse.ArgumentParser(description="拉取历史情绪数据")
    p.add_argument(
        "--output",
        type=str,
        default="data/sentiment/fear_greed_history_daily.parquet",
        help="输出 parquet 文件路径",
    )
    return p.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)

    print("=" * 60)
    print("  历史情绪数据拉取（一次性）")
    print("=" * 60)

    # 1. Fear & Greed Index (必做，完全免费)
    fng = pull_fear_greed_history(output_path)

    print("\n" + "=" * 60)
    print("  拉取完成")
    print("=" * 60)
    print(f"\n  下一步：启动实时收集守护进程")
    print(f"    python scripts/collect_sentiment_realtime.py")


if __name__ == "__main__":
    main()
