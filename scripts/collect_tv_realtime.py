#!/usr/bin/env python3
"""
collect_tv_realtime.py — TradingView 宏观特征实时采集器

每 15 分钟从 TradingView 拉取最新数据，追加到 data/sentiment/tv_{tag}.parquet。
用于 Exp13/Exp14 模型的实时预测时的特征来源。

用法:
  # 前台运行
  python scripts/collect_tv_realtime.py

  # 后台运行（推荐）
  nohup python scripts/collect_tv_realtime.py > logs/tv_collector.log 2>&1 &

  # 使用自定义间隔
  python scripts/collect_tv_realtime.py --interval 900  # 900秒 = 15分钟

环境变量:
  TV_USERNAME — TradingView 用户名
  TV_PASSWORD — TradingView 密码
"""
import os
import sys
import time
import signal
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SENTIMENT_DIR = PROJECT_ROOT / "data" / "sentiment"

# 与 pull_tv_history.py 保持一致
TV_SYMBOLS = [
    # (symbol, exchange, interval_name, safe_tag)
    ("TOTAL",    "CRYPTOCAP", "4h",    "total"),
    ("OTHERS",   "CRYPTOCAP", "4h",    "others"),
    ("ETHBTC",   "BINANCE",   "4h",    "ethbtc"),
    ("DXY",      "TVC",       "4h",    "dxy"),
    ("TOTAL2",   "CRYPTOCAP", "daily", "total2"),
    ("TOTAL3",   "CRYPTOCAP", "daily", "total3"),
    ("OTHERS.D", "CRYPTOCAP", "daily", "others_d"),
    ("OTHERSBTC","CRYPTOCAP", "daily", "othersbtc"),
    ("USDT.D",   "CRYPTOCAP", "daily", "usdt_d"),
    ("BTC.D",    "CRYPTOCAP", "daily", "btc_d"),
]

DEFAULT_INTERVAL_SECONDS = 900  # 15 分钟

_running = True


def _signal_handler(sig, frame):
    global _running
    print(f"\n[{datetime.now():%H:%M:%S}] 收到信号 {sig}, 正在退出...")
    _running = False


def _get_tv_interval(name: str):
    from tvDatafeed import Interval
    return {
        "15m":   Interval.in_15_minute,
        "1h":    Interval.in_1_hour,
        "4h":    Interval.in_4_hour,
        "daily": Interval.in_daily,
    }[name]


def update_one_symbol(tv, symbol: str, exchange: str, interval_name: str,
                      tag: str) -> int:
    """拉取最新数据并追加到 parquet 文件。返回新增行数。"""
    interval = _get_tv_interval(interval_name)

    # 拉取最近 100 根 K 线（足够覆盖上次采集后的新数据）
    df = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=100)
    if df is None or len(df) == 0:
        return 0

    df = df.reset_index()
    df.rename(columns={"datetime": "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["timestamp_ms"] = df["timestamp"].astype("int64") // 10**6

    keep = ["timestamp", "open", "high", "low", "close", "volume", "timestamp_ms"]
    for c in keep:
        if c not in df.columns:
            if c == "volume":
                df["volume"] = 0.0
            elif c == "timestamp_ms":
                df["timestamp_ms"] = df["timestamp"].astype("int64") // 10**6
    df = df[[c for c in keep if c in df.columns]]

    out_path = SENTIMENT_DIR / f"tv_{tag}.parquet"

    if out_path.exists():
        existing = pd.read_parquet(out_path)
        existing["timestamp"] = pd.to_datetime(existing["timestamp"])

        # 只追加新数据（timestamp 不在已有数据中的）
        max_existing_ts = existing["timestamp"].max()
        new_rows = df[df["timestamp"] > max_existing_ts]

        if len(new_rows) > 0:
            combined = pd.concat([existing, new_rows], ignore_index=True)
            combined = combined.sort_values("timestamp").drop_duplicates(
                subset=["timestamp"], keep="last"
            ).reset_index(drop=True)
            combined.to_parquet(out_path, index=False)
            return len(new_rows)
        return 0
    else:
        df.to_parquet(out_path, index=False)
        return len(df)


def collect_cycle(tv) -> dict:
    """执行一次完整采集周期。返回 {tag: new_rows}。"""
    results = {}
    for symbol, exchange, interval_name, tag in TV_SYMBOLS:
        try:
            n_new = update_one_symbol(tv, symbol, exchange, interval_name, tag)
            results[tag] = n_new
        except Exception as e:
            print(f"  ❌ {symbol}: {e}")
            results[tag] = -1
        time.sleep(0.5)  # 避免限流
    return results


def main():
    parser = argparse.ArgumentParser(description="TradingView 宏观特征实时采集器")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SECONDS,
                        help=f"采集间隔（秒, 默认 {DEFAULT_INTERVAL_SECONDS}）")
    parser.add_argument("--username", default=os.environ.get("TV_USERNAME", ""),
                        help="TradingView 用户名")
    parser.add_argument("--password", default=os.environ.get("TV_PASSWORD", ""),
                        help="TradingView 密码")
    parser.add_argument("--once", action="store_true",
                        help="只采集一次然后退出")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    from tvDatafeed import TvDatafeed

    if args.username and args.password:
        tv = TvDatafeed(username=args.username, password=args.password)
    else:
        print("⚠️ 未提供 TV 账号, 使用匿名模式")
        tv = TvDatafeed()

    SENTIMENT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print(f"  TradingView 实时采集器")
    print(f"  采集间隔: {args.interval}s ({args.interval/60:.0f}min)")
    print(f"  符号数: {len(TV_SYMBOLS)}")
    print(f"  输出: {SENTIMENT_DIR}")
    print(f"{'=' * 60}")

    cycle = 0
    while _running:
        cycle += 1
        ts_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{ts_now}] 采集周期 #{cycle}...")

        results = collect_cycle(tv)
        total_new = sum(v for v in results.values() if v > 0)
        errors = sum(1 for v in results.values() if v < 0)

        updated_tags = [f"{t}(+{n})" for t, n in results.items() if n > 0]
        if updated_tags:
            print(f"  ✅ 新数据: {', '.join(updated_tags)}")
        else:
            print(f"  📌 无新数据")
        if errors:
            print(f"  ⚠️ {errors} 个符号出错")

        if args.once:
            print(f"\n  单次模式，退出")
            break

        # 等待下一次采集
        next_ts = datetime.now().strftime("%H:%M:%S")
        wait_until = (datetime.now().timestamp() + args.interval)
        next_at = datetime.fromtimestamp(wait_until).strftime("%H:%M:%S")
        print(f"  ⏳ 下次采集: {next_at} (等待 {args.interval}s)")

        # 分段 sleep 以响应信号
        elapsed = 0
        while _running and elapsed < args.interval:
            time.sleep(min(10, args.interval - elapsed))
            elapsed += 10

    print(f"\n[{datetime.now():%H:%M:%S}] TV 实时采集器已退出")


if __name__ == "__main__":
    main()
