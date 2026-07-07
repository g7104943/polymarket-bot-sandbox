#!/usr/bin/env python3
"""
pull_tv_history.py — 从 TradingView 拉取 10 个宏观特征符号的历史数据

符号列表 (CRYPTOCAP / BINANCE / TVC):
  TOTAL, TOTAL2, TOTAL3, OTHERS, OTHERS.D, OTHERSBTC, USDT.D, BTC.D, ETHBTC, DXY

数据覆盖策略:
  - 4h 数据可覆盖 2024-01 的: TOTAL, OTHERS, ETHBTC, DXY → 用 4h
  - 仅日线可覆盖 2024-01 的: TOTAL2, TOTAL3, OTHERS.D, OTHERSBTC, USDT.D, BTC.D → 用日线
  两者都用 merge_asof(backward) 对齐到 15m，宏观指标低频完全够用

输出: data/sentiment/tv_{symbol_clean}.parquet
"""
import os
import sys
import time
import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── TradingView 符号配置 ──────────────────────────────────────────────────
TV_SYMBOLS = [
    # (symbol, exchange, preferred_interval, safe_tag)
    # 4h 组 (2023-01 起)
    ("TOTAL",    "CRYPTOCAP", "4h",    "total"),
    ("OTHERS",   "CRYPTOCAP", "4h",    "others"),
    ("ETHBTC",   "BINANCE",   "4h",    "ethbtc"),
    ("DXY",      "TVC",       "4h",    "dxy"),
    # 日线组 (4h 历史不够, 用日线)
    ("TOTAL2",   "CRYPTOCAP", "daily", "total2"),
    ("TOTAL3",   "CRYPTOCAP", "daily", "total3"),
    ("OTHERS.D", "CRYPTOCAP", "daily", "others_d"),
    ("OTHERSBTC","CRYPTOCAP", "daily", "othersbtc"),
    ("USDT.D",   "CRYPTOCAP", "daily", "usdt_d"),
    ("BTC.D",    "CRYPTOCAP", "daily", "btc_d"),
]

SENTIMENT_DIR = PROJECT_ROOT / "data" / "sentiment"


def _get_interval(name: str):
    from tvDatafeed import Interval
    return {
        "15m":   Interval.in_15_minute,
        "1h":    Interval.in_1_hour,
        "4h":    Interval.in_4_hour,
        "daily": Interval.in_daily,
    }[name]


def pull_symbol(tv, symbol: str, exchange: str, interval_name: str,
                tag: str, min_date: str = "2024-01-01") -> pd.DataFrame | None:
    """拉取单个符号, 返回 DataFrame 或 None."""
    from tvDatafeed import Interval
    interval = _get_interval(interval_name)
    print(f"  拉取 {symbol} ({exchange}) @ {interval_name} ...", end=" ", flush=True)

    df = tv.get_hist(symbol=symbol, exchange=exchange, interval=interval, n_bars=50000)
    if df is None or len(df) == 0:
        print("❌ 无数据")
        return None

    # tvDatafeed 返回 datetime index, 转成 timestamp 列
    df = df.reset_index()
    df.rename(columns={"datetime": "timestamp"}, inplace=True)

    # 确保 timestamp 是 datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 过滤到 min_date 之后
    df = df[df["timestamp"] >= min_date].copy()

    # 只保留 OHLCV
    keep = ["timestamp", "open", "high", "low", "close", "volume"]
    for c in keep:
        if c not in df.columns:
            if c == "volume":
                df["volume"] = 0.0
            else:
                print(f"❌ 缺少列 {c}")
                return None
    df = df[keep].copy()

    # 添加 timestamp_ms (毫秒) 方便后续 merge_asof
    df["timestamp_ms"] = df["timestamp"].astype("int64") // 10**6

    # 添加 interval 元数据
    df.attrs["interval"] = interval_name
    df.attrs["symbol"] = symbol
    df.attrs["exchange"] = exchange

    print(f"✅ {len(df)} 行, {df['timestamp'].iloc[0].date()} ~ {df['timestamp'].iloc[-1].date()}")
    return df


def main():
    parser = argparse.ArgumentParser(description="拉取 TradingView 宏观特征历史数据")
    parser.add_argument("--username", default=os.environ.get("TV_USERNAME", ""),
                        help="TradingView 用户名 (或设 TV_USERNAME 环境变量)")
    parser.add_argument("--password", default=os.environ.get("TV_PASSWORD", ""),
                        help="TradingView 密码 (或设 TV_PASSWORD 环境变量)")
    parser.add_argument("--min-date", default="2024-01-01",
                        help="最早日期 (默认 2024-01-01)")
    parser.add_argument("--output-dir", default=str(SENTIMENT_DIR),
                        help="输出目录")
    args = parser.parse_args()

    from tvDatafeed import TvDatafeed

    if args.username and args.password:
        tv = TvDatafeed(username=args.username, password=args.password)
    else:
        print("⚠️  未提供 TV 账号, 使用匿名模式 (数据可能受限)")
        tv = TvDatafeed()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  TradingView 宏观特征历史拉取")
    print(f"  输出: {out_dir}")
    print(f"  最早: {args.min_date}")
    print(f"{'='*60}\n")

    results = {}
    for symbol, exchange, interval_name, tag in TV_SYMBOLS:
        df = pull_symbol(tv, symbol, exchange, interval_name, tag, args.min_date)
        if df is not None:
            out_path = out_dir / f"tv_{tag}.parquet"
            df.to_parquet(out_path, index=False)
            results[tag] = {
                "rows": len(df),
                "interval": interval_name,
                "start": str(df["timestamp"].iloc[0].date()),
                "end": str(df["timestamp"].iloc[-1].date()),
            }
        time.sleep(1)  # 避免限流

    # 汇总
    print(f"\n{'='*60}")
    print(f"  拉取完成: {len(results)}/{len(TV_SYMBOLS)} 个符号")
    print(f"{'='*60}")
    for tag, info in results.items():
        print(f"  {tag:12s}: {info['rows']:5d} 行 @ {info['interval']:5s}, "
              f"{info['start']} ~ {info['end']}")

    if len(results) < len(TV_SYMBOLS):
        failed = [t[3] for t in TV_SYMBOLS if t[3] not in results]
        print(f"\n  ❌ 失败: {failed}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
