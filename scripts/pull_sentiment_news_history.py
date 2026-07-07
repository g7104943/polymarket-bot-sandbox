#!/usr/bin/env python3
"""
回填 CryptoCompare 历史新闻情绪数据，聚合到 15 分钟桶。

CryptoCompare News API:
  - 完全免费、无需 API key
  - 支持 lTs 参数翻页（获取 published_on < lTs 的文章）
  - 每页约 50 篇文章

流程：
  1. 从当前时间往回翻页拉取所有新闻
  2. 对每篇新闻标题+摘要做 VADER 情绪评分
  3. 按 15 分钟桶聚合：news_volume, sentiment_score, social_volume
  4. 保存为 parquet

用法：
  python scripts/pull_sentiment_news_history.py
  python scripts/pull_sentiment_news_history.py --days 912 --output data/sentiment/news_sentiment_history_15m.parquet
  python scripts/pull_sentiment_news_history.py --days 912 --resume   # 断点续传，跳过已有数据
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
except ImportError:
    print("[ERROR] vaderSentiment 未安装: pip install vaderSentiment")
    sys.exit(1)


SOCIAL_KEYWORDS = {"social", "reddit", "twitter", "telegram", "community", "x.com"}


def fetch_news_page(lTs: int | None = None, max_retries: int = 5) -> list[dict]:
    """
    拉取一页 CryptoCompare 新闻（带重试）。

    Parameters
    ----------
    lTs : int or None
        返回 published_on < lTs 的文章。None = 最新。
    max_retries : int
        最大重试次数（每次指数退避）。

    Returns
    -------
    list[dict]
        文章列表，每篇包含 published_on, title, body, tags, categories 等。
    """
    url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"
    if lTs is not None:
        url += f"&lTs={lTs}"

    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            data = r.json()
            return data.get("Data", [])
        except Exception as e:
            wait = min(2 ** attempt, 30)
            if attempt < max_retries - 1:
                print(f"  [WARN] 请求失败 (重试 {attempt+1}/{max_retries}, 等{wait}s): {type(e).__name__}")
                time.sleep(wait)
            else:
                print(f"  [ERROR] 请求失败 (已耗尽重试): {e}")
                return []
    return []


def score_article(article: dict) -> dict:
    """对单篇文章计算 VADER 情绪 + 分类。"""
    title = article.get("title", "")
    body_snippet = article.get("body", "")[:200]
    text = f"{title}. {body_snippet}"

    score = _vader.polarity_scores(text)["compound"]

    tags = (article.get("tags", "") + " " + article.get("categories", "")).lower()
    is_social = any(kw in tags for kw in SOCIAL_KEYWORDS)

    return {
        "published_on": article.get("published_on", 0),
        "title": title[:120],
        "sentiment_score": round(score, 4),
        "is_social": is_social,
    }


def aggregate_to_buckets(articles: list[dict], bucket_minutes: int = 15) -> pd.DataFrame:
    """
    将文章列表聚合到指定分钟桶。

    Parameters
    ----------
    articles : list[dict]
        经过 score_article 的文章列表。
    bucket_minutes : int
        桶大小（分钟），支持 5 或 15。

    返回 DataFrame 列：
      - timestamp (datetime, 桶起始)
      - news_volume_{bucket_minutes}m (int)
      - sentiment_score_{bucket_minutes}m (float, VADER 均值)
      - social_volume_{bucket_minutes}m (int)
    """
    if not articles:
        return pd.DataFrame()

    suffix = f"{bucket_minutes}m"
    df = pd.DataFrame(articles)
    df["timestamp"] = pd.to_datetime(df["published_on"], unit="s", utc=True)

    # 向下取整到指定分钟
    df["bucket"] = df["timestamp"].dt.floor(f"{bucket_minutes}min")

    vol_col = f"news_volume_{suffix}"
    sent_col = f"sentiment_score_{suffix}"
    soc_col = f"social_volume_{suffix}"

    grouped = df.groupby("bucket").agg(
        **{vol_col: ("sentiment_score", "count"),
           sent_col: ("sentiment_score", "mean"),
           soc_col: ("is_social", "sum")},
    ).reset_index()

    grouped = grouped.rename(columns={"bucket": "timestamp"})
    grouped[sent_col] = grouped[sent_col].round(4)
    grouped[soc_col] = grouped[soc_col].astype(int)

    return grouped.sort_values("timestamp").reset_index(drop=True)


def aggregate_to_15m(articles: list[dict]) -> pd.DataFrame:
    """向后兼容：聚合到 15 分钟桶。"""
    return aggregate_to_buckets(articles, bucket_minutes=15)


def main():
    parser = argparse.ArgumentParser(description="回填 CryptoCompare 历史新闻情绪数据")
    parser.add_argument(
        "--days", type=int, default=912,
        help="回溯天数（默认 912，约 2.5 年）",
    )
    parser.add_argument(
        "--output", type=str,
        default="data/sentiment/news_sentiment_history_15m.parquet",
        help="输出 parquet 路径",
    )
    parser.add_argument(
        "--delay", type=float, default=0.3,
        help="每页请求间隔秒数（礼貌限速，默认 0.3）",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="断点续传：从已有数据的最早时间继续向前拉取",
    )
    parser.add_argument(
        "--bucket-minutes", type=int, default=15, choices=[5, 15],
        help="聚合桶大小（分钟），默认 15; Exp13 用 5",
    )
    args = parser.parse_args()

    bucket_minutes = args.bucket_minutes
    bucket_suffix = f"{bucket_minutes}m"

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    target_start = now - timedelta(days=args.days)
    target_ts = int(target_start.timestamp())

    # ── Resume logic: load existing data and continue backward ──
    existing_df = None
    resume_from_ts = None
    if args.resume and output_path.exists():
        existing_df = pd.read_parquet(output_path)
        if not existing_df.empty:
            earliest_existing = existing_df["timestamp"].min()
            # Handle tz-aware timestamps
            if hasattr(earliest_existing, "tz") and earliest_existing.tz is not None:
                resume_from_ts = int(earliest_existing.timestamp())
            else:
                resume_from_ts = int(earliest_existing.timestamp())
            print(f"  [RESUME] 已有数据 {len(existing_df)} 桶，最早: {earliest_existing}")
            print(f"  [RESUME] 从 {earliest_existing} 继续向前拉取到 {target_start.strftime('%Y-%m-%d')}")

    print("=" * 60)
    print("  CryptoCompare 历史新闻情绪回填")
    print("=" * 60)
    print(f"  回溯: {args.days} 天 ({target_start.strftime('%Y-%m-%d')} ~ {now.strftime('%Y-%m-%d')})")
    print(f"  输出: {args.output}")
    print(f"  限速: {args.delay}s/页")
    if resume_from_ts:
        print(f"  断点续传: 从 {datetime.fromtimestamp(resume_from_ts, tz=timezone.utc).strftime('%Y-%m-%d %H:%M')} 向前")
    print("=" * 60)

    all_articles = []
    lTs = resume_from_ts  # None = 从最新开始; resume = 从已有数据最早处继续
    page = 0
    earliest_so_far = resume_from_ts if resume_from_ts else int(now.timestamp())
    consecutive_empty = 0

    while earliest_so_far > target_ts:
        articles = fetch_news_page(lTs)

        if not articles:
            consecutive_empty += 1
            if consecutive_empty >= 3:
                print(f"\n  连续 3 页空结果，API 可能已到底。停止。")
                break
            time.sleep(1)
            continue

        consecutive_empty = 0

        # 计算 VADER 情绪
        scored = [score_article(a) for a in articles]
        all_articles.extend(scored)

        # 翻页：用本页最早文章的时间戳
        timestamps = [a["published_on"] for a in articles if a.get("published_on", 0) > 0]
        if timestamps:
            earliest_page = min(timestamps)
            earliest_so_far = earliest_page
            lTs = earliest_page
        else:
            break

        page += 1
        earliest_dt = datetime.fromtimestamp(earliest_so_far, tz=timezone.utc)
        if page % 20 == 0:
            print(
                f"  [{page:>4d} 页] {len(all_articles):>6d} 篇 | "
                f"最早: {earliest_dt.strftime('%Y-%m-%d %H:%M')} | "
                f"距目标: {(earliest_so_far - target_ts) / 86400:.0f} 天"
            )

        # 每 200 页保存一次中间结果（防止长时间跑到一半丢失）
        if page % 200 == 0 and all_articles:
            checkpoint_arts = [a for a in all_articles if a["published_on"] >= target_ts]
            if checkpoint_arts:
                ckpt = aggregate_to_buckets(checkpoint_arts, bucket_minutes=bucket_minutes)
                ckpt.to_parquet(output_path, index=False)
                print(f"  [CHECKPOINT] 已保存 {len(ckpt)} 桶 → {output_path}")

        time.sleep(args.delay)

    print(f"\n  总计: {len(all_articles)} 篇文章, {page} 页")

    if not all_articles:
        print("  [ERROR] 无文章拉取到")
        sys.exit(1)

    # 过滤目标时间范围内的文章
    all_articles = [a for a in all_articles if a["published_on"] >= target_ts]
    print(f"  目标范围内: {len(all_articles)} 篇")

    # 聚合到指定分钟桶
    print(f"  聚合到 {bucket_minutes} 分钟桶...")
    result = aggregate_to_buckets(all_articles, bucket_minutes=bucket_minutes)

    # 合并已有数据（断点续传）
    if existing_df is not None and not existing_df.empty:
        print(f"  [RESUME] 合并: 已有 {len(existing_df)} 桶 + 新增 {len(result)} 桶")
        result = pd.concat([existing_df, result], ignore_index=True)
        result = result.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        print(f"  [RESUME] 合并后: {len(result)} 桶")

    if result.empty:
        print("  [ERROR] 聚合结果为空")
        sys.exit(1)

    # 保存
    result.to_parquet(output_path, index=False)

    vol_col = f"news_volume_{bucket_suffix}"
    sent_col = f"sentiment_score_{bucket_suffix}"
    soc_col = f"social_volume_{bucket_suffix}"

    print(f"\n  结果:")
    print(f"    {bucket_suffix} 桶数: {len(result)}")
    print(f"    时间范围: {result['timestamp'].min()} ~ {result['timestamp'].max()}")
    print(f"    平均每桶新闻量: {result[vol_col].mean():.2f}")
    print(f"    平均情绪评分: {result[sent_col].mean():.4f}")
    print(f"    含 social 桶占比: {(result[soc_col] > 0).mean():.1%}")
    print(f"    已保存: {output_path}")

    # 按天统计概览
    result["date"] = result["timestamp"].dt.date
    daily = result.groupby("date").agg(
        news_count=(vol_col, "sum"),
        avg_sentiment=(sent_col, "mean"),
        buckets=(vol_col, "count"),
    )
    print(f"\n  日统计（最近 7 天）:")
    print(daily.tail(7).to_string())

    print(f"\n  下一步: 重训模型时用 load_and_prepare_sentiment() 自动加载此数据")


if __name__ == "__main__":
    main()
