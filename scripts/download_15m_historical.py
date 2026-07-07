#!/usr/bin/env python3
"""
下载 BTC/ETH 15 分钟 OHLCV 到 data/raw/，供 LSTM+XGBoost 混合模型使用。
默认拉取最近 7 年数据（训练+回测）。

用法:
  python scripts/download_15m_historical.py
  python scripts/download_15m_historical.py --years 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.python.data_fetcher import fetch_ohlcv, save_ohlcv

SYMBOLS_15M = ["BTC/USDT", "ETH/USDT"]


def main():
    ap = argparse.ArgumentParser(description="下载 BTC/ETH 15 分钟 K 线（默认 7 年）")
    ap.add_argument("--years", type=int, default=7, help="下载最近 N 年，默认 7")
    args = ap.parse_args()

    import time
    since_s = int(time.time()) - args.years * 365 * 24 * 3600
    since_ms = since_s * 1000

    for symbol in SYMBOLS_15M:
        print(f"Downloading {symbol} 15m (last {args.years} years)...")
        for attempt in range(1, 6):
            try:
                df = fetch_ohlcv(symbol, "15m", since=since_ms)
                if df.empty:
                    print(f"  -> 无数据")
                    break
                p = save_ohlcv(df, symbol, "15m")
                print(f"  -> {len(df)} rows -> {p}")
                ts_min = df["timestamp"].min()
                ts_max = df["timestamp"].max()
                print(f"  -> 时间范围: {ts_min} ~ {ts_max}")
                break
            except Exception as e:
                err = str(e).lower()
                if "429" in err or "too many requests" in err:
                    wait = 65
                    print(f"  -> 限频，{wait}s 后重试 (尝试 {attempt}/5)...")
                    time.sleep(wait)
                else:
                    print(f"  -> 失败: {e}")
                    raise
        else:
            raise RuntimeError(f"下载 {symbol} 15m 多次重试后仍失败")
        time.sleep(2)

    # 15m: 每年约 365*24*4 = 35040 根
    expected_per_year = 365 * 24 * 4
    print("Done. 校验: 每个 parquet 行数约", args.years * expected_per_year, "根/币。")


if __name__ == "__main__":
    main()
