#!/usr/bin/env python3
"""
实时情绪数据收集守护进程（每 15 分钟循环）。

数据源（全部免费，已验证可用）：
  1. CryptoCompare News API — 免费、无需 API key、实时更新
     https://min-api.cryptocompare.com/data/v2/news/?lang=EN
  2. Alternative.me Fear & Greed Index — 免费、无 key、日更新
     https://api.alternative.me/fng/?limit=1
  3. VADER Sentiment — 本地 NLP 计算，适合金融文本

收集的指标（6 个）：
  - fear_greed_index      (0-100, 日级, 前向填充到 15m)
  - news_volume_15m       (最近 15 分钟新闻数量)
  - sentiment_score_15m   (-1 到 +1, VADER 对新闻标题评分均值)
  - social_volume_15m     (近似: 含 social/reddit/twitter 关键词的新闻数)
  - sentiment_spike        (bool: |当前 - MA20| > 2*STD)
  - sentiment_change_rate  (当前与上一条的变化率)

用法：
  python scripts/collect_sentiment_realtime.py
  python scripts/collect_sentiment_realtime.py --interval 15 --output data/sentiment/sentiment_realtime_15m.parquet

  # 后台运行
  nohup python scripts/collect_sentiment_realtime.py > logs/sentiment_collector.log 2>&1 &
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# VADER 情绪分析器（比 TextBlob 更适合金融文本）
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
except ImportError:
    _vader = None
    print("[WARN] vaderSentiment 未安装，使用简化情绪评分。")
    print("  安装命令: pip install vaderSentiment")


# ───────────────────────────────────────────────────
# 数据源函数
# ───────────────────────────────────────────────────

def fetch_cryptocompare_news(lookback_minutes: int = 20) -> dict:
    """
    从 CryptoCompare 拉取最近的加密货币新闻。

    API: https://min-api.cryptocompare.com/data/v2/news/?lang=EN
    完全免费、无需 API key、无速率限制。

    返回:
      {
        "news_volume_15m": int,       # 最近 15min 新闻数量
        "sentiment_score_15m": float, # VADER 情绪评分均值 (-1 ~ +1)
        "social_volume_15m": int,     # 含 social 关键词的新闻数（近似）
        "top_headlines": list,        # 前 3 条标题
      }
    """
    url = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN"

    data = None
    for attempt in range(1, 31):
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            data = r.json()
            break
        except requests.RequestException as e:
            print(f"  [WARN] CryptoCompare 请求失败 (attempt {attempt}/30): {e}")
            if attempt < 30:
                time.sleep(2)

    if data is None:
        return {
            "news_volume_15m": 0,
            "sentiment_score_15m": 0.0,
            "social_volume_15m": 0,
            "top_headlines": [],
        }

    articles = data.get("Data", [])
    if not articles:
        return {
            "news_volume_15m": 0,
            "sentiment_score_15m": 0.0,
            "social_volume_15m": 0,
            "top_headlines": [],
        }

    # 筛选最近 N 分钟的新闻
    now_ts = int(datetime.now(timezone.utc).timestamp())
    cutoff_ts = now_ts - lookback_minutes * 60

    recent = [a for a in articles if a.get("published_on", 0) >= cutoff_ts]
    news_volume = len(recent)

    # VADER 情绪评分
    sentiments = []
    for a in recent:
        title = a.get("title", "")
        body_snippet = a.get("body", "")[:200]  # 只取前 200 字符
        text = f"{title}. {body_snippet}"

        if _vader is not None:
            score = _vader.polarity_scores(text)["compound"]
        else:
            # 简化替代：基于关键词
            pos_words = ["surge", "rally", "bullish", "soar", "gain", "rise", "up", "high", "record"]
            neg_words = ["crash", "plunge", "bearish", "dump", "fall", "drop", "down", "low", "fear"]
            text_lower = text.lower()
            pos_count = sum(1 for w in pos_words if w in text_lower)
            neg_count = sum(1 for w in neg_words if w in text_lower)
            score = (pos_count - neg_count) / max(pos_count + neg_count, 1)

        sentiments.append(score)

    sentiment_score = sum(sentiments) / len(sentiments) if sentiments else 0.0

    # 社交量近似：含 social/reddit/twitter/telegram 相关标签/分类的新闻数
    social_keywords = {"social", "reddit", "twitter", "telegram", "community", "x.com"}
    social_count = 0
    for a in recent:
        tags = (a.get("tags", "") + " " + a.get("categories", "")).lower()
        if any(kw in tags for kw in social_keywords):
            social_count += 1

    top_headlines = [a.get("title", "")[:80] for a in recent[:3]]

    return {
        "news_volume_15m": news_volume,
        "sentiment_score_15m": round(sentiment_score, 4),
        "social_volume_15m": social_count,
        "top_headlines": top_headlines,
    }


def fetch_cfgi_webhook(port: int = 3002) -> dict | None:
    """
    从本地 CFGI webhook server 读取最新推送数据。

    CFGI Webhook 推送模式：免费无限（不扣 credits）。
    需要先启动 webhook server (node scripts/cfgi_webhook_server.js)
    并用 ngrok 暴露端口，在 CFGI dashboard 配置 webhook URL。

    返回: dict 或 None（server 未运行或无数据时）
    """
    try:
        r = requests.get(f"http://localhost:{port}/api/webHook", timeout=2)
        if r.status_code == 200:
            data = r.json()
            return {
                "fear_greed_index": data.get("cfgi", data.get("fear_greed_index")),
                "social_volume_15m": data.get("social_volume"),
                "social_sentiment": data.get("social_sentiment"),
            }
    except Exception:
        pass  # webhook server 未运行，静默跳过
    return None


def fetch_fear_greed_latest() -> int | None:
    """
    从 Alternative.me 拉取最新 Fear & Greed Index。

    API: https://api.alternative.me/fng/?limit=1
    完全免费、无 key。日更新，但我们每 15min 检查一次（前向填充）。

    返回: int (0-100) 或 None（请求失败时）
    """
    url = "https://api.alternative.me/fng/?limit=1"

    for attempt in range(1, 31):
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            data = r.json()
            return int(data["data"][0]["value"])
        except Exception as e:
            print(f"  [WARN] Alternative.me FGI 请求失败 (attempt {attempt}/30): {e}")
            if attempt < 30:
                time.sleep(2)

    return None


# ───────────────────────────────────────────────────
# 收集器主类
# ───────────────────────────────────────────────────

class SentimentCollector:
    """15 分钟情绪数据收集器。"""

    COLUMNS = [
        "timestamp",
        "fear_greed_index",
        "news_volume_15m",
        "sentiment_score_15m",
        "social_volume_15m",
        "sentiment_spike",
        "sentiment_change_rate",
    ]

    def __init__(self, output_path: str = "data/sentiment/sentiment_realtime_15m.parquet"):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_history()
        self._last_fgi: int | None = None

    def _load_history(self):
        """加载历史数据。"""
        if self.output_path.exists():
            self.df = pd.read_parquet(self.output_path)
            print(f"  已加载历史: {len(self.df)} 条记录")
        else:
            self.df = pd.DataFrame(columns=self.COLUMNS)
            print("  无历史数据，从零开始")

    def _calculate_derived(self, new_row: dict) -> dict:
        """计算派生特征: sentiment_spike, sentiment_change_rate。"""
        if len(self.df) < 20:
            new_row["sentiment_spike"] = False
            new_row["sentiment_change_rate"] = 0.0
            return new_row

        # MA20 和 STD20
        recent = self.df["sentiment_score_15m"].tail(20)
        ma20 = recent.mean()
        std20 = recent.std()

        # Spike: |当前 - MA20| > 2 * STD20
        current = new_row["sentiment_score_15m"]
        new_row["sentiment_spike"] = bool(
            std20 > 0.001 and abs(current - ma20) > 2 * std20
        )

        # Change rate
        prev = self.df["sentiment_score_15m"].iloc[-1]
        if abs(prev) > 0.001:
            new_row["sentiment_change_rate"] = round(
                (current - prev) / abs(prev), 4
            )
        else:
            new_row["sentiment_change_rate"] = 0.0

        return new_row

    # ── 历史文件路径（prediction_writer_v5.py 读取的文件） ──
    FGI_HISTORY_PATH = Path("data/sentiment/fear_greed_history_daily.parquet")
    NEWS_HISTORY_PATH = Path("data/sentiment/news_sentiment_history_15m.parquet")

    @staticmethod
    def _fgi_classification(value: int) -> str:
        """FGI 值 → 分类字符串（与 pull_sentiment_history.py 一致）。"""
        if value <= 24:
            return "Extreme Fear"
        elif value <= 49:
            return "Fear"
        elif value == 50:
            return "Neutral"
        elif value <= 74:
            return "Greed"
        else:
            return "Extreme Greed"

    @staticmethod
    def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
        """先写 tmp 再 rename，保证原子性。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(suffix=".parquet", dir=str(path.parent))
        os.close(fd)
        try:
            df.to_parquet(tmp, index=False)
            os.replace(tmp, str(path))
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise

    @staticmethod
    def _to_utc_series(series: pd.Series) -> pd.Series:
        """逐元素解析，兼容混合时区/混合格式时间戳。"""
        return series.apply(lambda v: pd.to_datetime(v, errors="coerce", utc=True))

    def _append_to_fgi_file(self, fgi_value: int) -> None:
        """追加 FGI 到 fear_greed_history_daily.parquet（按日去重）。"""
        try:
            today_utc = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            today_date = today_utc.date()
            new_row = pd.DataFrame([{
                "timestamp": today_utc,
                "fear_greed_index": int(fgi_value),
                "fng_classification": self._fgi_classification(fgi_value),
            }])

            if self.FGI_HISTORY_PATH.exists():
                existing = pd.read_parquet(self.FGI_HISTORY_PATH)
                # 按日去重：如果今天已有则更新
                existing_ts = self._to_utc_series(existing["timestamp"])
                existing["_date"] = existing_ts.dt.date
                new_row["_date"] = today_date
                if today_date in existing["_date"].values:
                    # 更新今天的值
                    mask = existing["_date"] == today_date
                    existing.loc[mask, "timestamp"] = today_utc
                    existing.loc[mask, "fear_greed_index"] = int(fgi_value)
                    existing.loc[mask, "fng_classification"] = self._fgi_classification(fgi_value)
                    combined = existing.drop(columns=["_date"])
                else:
                    combined = pd.concat(
                        [existing.drop(columns=["_date"]), new_row.drop(columns=["_date"])],
                        ignore_index=True,
                    )
                combined["timestamp"] = self._to_utc_series(combined["timestamp"])
                combined = combined[combined["timestamp"].notna()].copy()
                combined = combined.sort_values("timestamp").reset_index(drop=True)
            else:
                combined = new_row.drop(columns=["_date"])
                combined["timestamp"] = self._to_utc_series(combined["timestamp"])

            self._atomic_write_parquet(combined, self.FGI_HISTORY_PATH)
            print(f"  [FGI] 已更新 fear_greed_history_daily.parquet (FGI={fgi_value}, 总 {len(combined)} 天)")
        except Exception as e:
            print(f"  [WARN] FGI 历史文件写入失败: {e}")

    def _append_to_news_file(self, news_data: dict) -> None:
        """追加 news 到 news_sentiment_history_15m.parquet（按 15m 桶去重）。"""
        try:
            now = datetime.now(timezone.utc)
            bucket = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
            new_row = pd.DataFrame([{
                "timestamp": bucket,
                "news_volume_15m": int(news_data.get("news_volume_15m", 0)),
                "sentiment_score_15m": round(float(news_data.get("sentiment_score_15m", 0.0)), 4),
                "social_volume_15m": int(news_data.get("social_volume_15m", 0)),
            }])

            if self.NEWS_HISTORY_PATH.exists():
                existing = pd.read_parquet(self.NEWS_HISTORY_PATH)
                combined = pd.concat([existing, new_row], ignore_index=True)
                combined = combined.drop_duplicates(
                    subset=["timestamp"], keep="last"
                ).sort_values("timestamp").reset_index(drop=True)
            else:
                combined = new_row

            self._atomic_write_parquet(combined, self.NEWS_HISTORY_PATH)
            print(f"  [News] 已更新 news_sentiment_history_15m.parquet "
                  f"(vol={news_data.get('news_volume_15m', 0)}, "
                  f"sent={news_data.get('sentiment_score_15m', 0.0):.4f}, "
                  f"总 {len(combined)} 桶)")
        except Exception as e:
            print(f"  [WARN] News 历史文件写入失败: {e}")

    def collect_once(self) -> dict:
        """执行一次 15 分钟数据收集。"""
        now = datetime.now(timezone.utc)
        print(f"\n[{now.strftime('%Y-%m-%d %H:%M:%S UTC')}] 收集情绪数据...")

        # 1. CryptoCompare 新闻
        news = fetch_cryptocompare_news(lookback_minutes=20)
        print(f"  新闻量: {news['news_volume_15m']}, 情绪: {news['sentiment_score_15m']:.4f}")
        if news["top_headlines"]:
            for h in news["top_headlines"][:2]:
                print(f"    -> {h}")

        # 2. 尝试从 CFGI Webhook 获取（免费无限，15min 粒度）
        cfgi_data = fetch_cfgi_webhook()
        if cfgi_data and cfgi_data.get("fear_greed_index") is not None:
            print(f"  CFGI Webhook: FGI={cfgi_data['fear_greed_index']}, social={cfgi_data.get('social_volume_15m')}")
            # 覆盖新闻的 social_volume（CFGI 更准）
            if cfgi_data.get("social_volume_15m") is not None:
                news["social_volume_15m"] = cfgi_data["social_volume_15m"]

        # 3. Fear & Greed Index（CFGI webhook 优先，Alternative.me 兜底）
        fgi = (cfgi_data or {}).get("fear_greed_index")
        if fgi is None:
            fgi = fetch_fear_greed_latest()
        if fgi is not None:
            self._last_fgi = fgi
        elif self._last_fgi is not None:
            fgi = self._last_fgi  # 前向填充
        else:
            # 从历史中取最近值
            if len(self.df) > 0 and "fear_greed_index" in self.df.columns:
                last_valid = self.df["fear_greed_index"].dropna()
                fgi = int(last_valid.iloc[-1]) if len(last_valid) > 0 else 50
            else:
                fgi = 50  # 默认中性
        print(f"  FGI: {fgi}")

        # 3. 组装
        new_row = {
            "timestamp": now,
            "fear_greed_index": fgi,
            "news_volume_15m": news["news_volume_15m"],
            "sentiment_score_15m": news["sentiment_score_15m"],
            "social_volume_15m": news["social_volume_15m"],
        }

        # 4. 计算派生特征
        new_row = self._calculate_derived(new_row)

        # 5. 追加并保存
        self.df = pd.concat(
            [self.df, pd.DataFrame([new_row])], ignore_index=True
        )
        self.df.to_parquet(self.output_path, index=False)

        spike_marker = " [SPIKE!]" if new_row["sentiment_spike"] else ""
        print(
            f"  已保存 (总 {len(self.df)} 条) | "
            f"spike={new_row['sentiment_spike']}{spike_marker} | "
            f"change_rate={new_row['sentiment_change_rate']:.4f}"
        )

        # 6. 同步更新预测器使用的历史文件
        self._append_to_fgi_file(fgi)
        self._append_to_news_file(news)

        return new_row

    def run_loop(self, interval_minutes: int = 15):
        """持续运行（每 N 分钟一次）。"""
        print(f"\n启动情绪收集守护进程 (间隔 {interval_minutes} 分钟)")
        print(f"输出: {self.output_path}")
        print("-" * 50)

        while True:
            try:
                self.collect_once()
            except Exception as e:
                print(f"  [ERROR] {e}")
                import traceback
                traceback.print_exc()

            time.sleep(interval_minutes * 60)


# ───────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="实时情绪数据收集守护进程")
    p.add_argument(
        "--interval", type=int, default=15,
        help="收集间隔（分钟，默认 15）",
    )
    p.add_argument(
        "--output", type=str,
        default="data/sentiment/sentiment_realtime_15m.parquet",
        help="输出 parquet 文件路径",
    )
    p.add_argument(
        "--once", action="store_true",
        help="只收集一次然后退出（调试用）",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # 优雅退出
    def signal_handler(sig, frame):
        print("\n收到退出信号，正在关闭...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=" * 60)
    print("  情绪数据收集器")
    print("=" * 60)
    print(f"  数据源 1: CryptoCompare News (免费, 无 key)")
    print(f"  数据源 2: Alternative.me FGI  (免费, 无 key)")
    print(f"  NLP 引擎: {'VADER' if _vader else '简化关键词'}")
    print(f"  间隔:     {args.interval} 分钟")
    print(f"  输出:     {args.output}")
    print("=" * 60)

    collector = SentimentCollector(output_path=args.output)

    if args.once:
        collector.collect_once()
    else:
        collector.run_loop(interval_minutes=args.interval)


if __name__ == "__main__":
    main()
