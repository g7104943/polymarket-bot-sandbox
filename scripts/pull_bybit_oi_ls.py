#!/usr/bin/env python3
"""
从 Bybit v5 API 下载持仓量(OI) 和多空比(Long/Short Ratio) 历史数据。

Bybit 优势：
  - OI 和多空比均支持 15m 粒度
  - 历史数据保留远超 Binance 的 30 天限制（可拉 2+ 年）

数据保存格式与 Binance 脚本输出完全兼容，merge 函数无需改动。

用法:
    python scripts/pull_bybit_oi_ls.py                  # 默认 150 天
    python scripts/pull_bybit_oi_ls.py --days 180       # 自定义天数
    python scripts/pull_bybit_oi_ls.py --symbols BTCUSDT ETHUSDT

输出:
    data/sentiment/open_interest_15m.parquet    (覆盖 Binance 的 6 天数据)
    data/sentiment/long_short_ratio_15m.parquet (覆盖 Binance 的 6 天数据)
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "sentiment"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
BASE_URL = "https://api.bybit.com"

# ─── Bybit API 配置 ──────────────────────────────────────


def fetch_bybit_paginated(
    url: str,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = 200,
    max_retries: int = 3,
    interval_param_name: str = "intervalTime",
) -> list[dict]:
    """分页拉取 Bybit v5 API 数据，支持 cursor 翻页。

    interval_param_name: OI 用 "intervalTime"，account-ratio 用 "period"。
    """
    all_data = []
    cursor = ""
    page = 0

    while True:
        params = {
            "category": "linear",
            "symbol": symbol,
            interval_param_name: interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": limit,
        }
        if cursor:
            params["cursor"] = cursor

        data = None
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, params=params, timeout=30)
                resp.raise_for_status()
                body = resp.json()

                if body.get("retCode") != 0:
                    print(f"\n    API 错误: {body.get('retMsg', 'unknown')}")
                    return all_data

                data = body["result"]["list"]
                cursor = body["result"].get("nextPageCursor", "")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    print(f"\n    请求失败: {e}")
                    return all_data

        if not data:
            break

        all_data.extend(data)
        page += 1

        if page % 20 == 0:
            print(f"{len(all_data)}条..", end="", flush=True)

        if not cursor:
            break

        time.sleep(0.25)  # 限速保护

    return all_data


def download_open_interest(
    symbol: str, start_ms: int, end_ms: int
) -> pd.DataFrame:
    """下载 Bybit 持仓量数据并转换为兼容格式。"""
    url = f"{BASE_URL}/v5/market/open-interest"
    raw = fetch_bybit_paginated(url, symbol, "15min", start_ms, end_ms)

    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(raw)
    # Bybit 返回: openInterest (str), timestamp (str ms)
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    df["open_interest"] = pd.to_numeric(df["openInterest"])
    df["symbol"] = symbol
    df = df[["timestamp", "symbol", "open_interest"]]
    df = df.drop_duplicates(subset=["timestamp", "symbol"], keep="last")
    return df.sort_values("timestamp").reset_index(drop=True)


def download_long_short_ratio(
    symbol: str, start_ms: int, end_ms: int
) -> pd.DataFrame:
    """下载 Bybit 多空比数据并转换为兼容格式。

    Bybit 返回 buyRatio / sellRatio (买方/卖方比例)。
    转换为 long_short_ratio = buyRatio / sellRatio，与 Binance 格式一致。
    """
    url = f"{BASE_URL}/v5/market/account-ratio"
    # account-ratio 用 period 而非 intervalTime，最大 limit=500
    raw = fetch_bybit_paginated(
        url, symbol, "15min", start_ms, end_ms,
        limit=500, interval_param_name="period",
    )

    if not raw:
        return pd.DataFrame()

    df = pd.DataFrame(raw)
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    buy_ratio = pd.to_numeric(df["buyRatio"])
    sell_ratio = pd.to_numeric(df["sellRatio"])
    # long_short_ratio = 多头 / 空头
    df["long_short_ratio"] = buy_ratio / sell_ratio.replace(0, float("nan"))
    df["long_account"] = buy_ratio
    df["short_account"] = sell_ratio
    df["symbol"] = symbol
    df = df[["timestamp", "symbol", "long_short_ratio", "long_account", "short_account"]]
    df = df.drop_duplicates(subset=["timestamp", "symbol"], keep="last")
    return df.sort_values("timestamp").reset_index(drop=True)


def save_parquet(df: pd.DataFrame, output_path: Path, data_name: str):
    """保存到 parquet（直接覆盖，因为 Bybit 数据更完整）。"""
    if df.empty:
        print(f"  ⚠️ {data_name}: 无数据，跳过")
        return

    # 全局去重排序
    key_cols = ["timestamp", "symbol"]
    df = df.drop_duplicates(subset=key_cols, keep="last")
    df = df.sort_values(key_cols).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    kb = output_path.stat().st_size / 1024
    print(f"  💾 {output_path.name}: {len(df)} 条 ({kb:.0f}KB)")


def main():
    parser = argparse.ArgumentParser(
        description="从 Bybit 下载 OI + 多空比历史数据（15m）"
    )
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--days", type=int, default=150,
                        help="下载最近 N 天（默认 150 = 120天窗口 + 30天预热）")
    args = parser.parse_args()

    end_dt = datetime.utcnow()
    start_dt = end_dt - timedelta(days=args.days)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    print(f"{'=' * 60}")
    print(f"  Bybit 持仓量 + 多空比 下载器")
    print(f"{'=' * 60}")
    print(f"  币种:  {', '.join(args.symbols)}")
    print(f"  范围:  {start_dt.strftime('%Y-%m-%d')} → {end_dt.strftime('%Y-%m-%d')} ({args.days}天)")
    print(f"  粒度:  15min")
    print(f"{'=' * 60}\n")

    # ─── 持仓量 (Open Interest) ───
    print("📥 持仓量 (Open Interest):")
    oi_dfs = []
    for sym in args.symbols:
        print(f"  {sym}: ", end="", flush=True)
        df = download_open_interest(sym, start_ms, end_ms)
        print(f"→ {len(df)} 条")
        if not df.empty:
            oi_dfs.append(df)

    if oi_dfs:
        oi_all = pd.concat(oi_dfs, ignore_index=True)
        save_parquet(oi_all, OUTPUT_DIR / "open_interest_15m.parquet", "OI")
    print()

    # ─── 多空比 (Long/Short Ratio) ───
    print("📥 多空比 (Long/Short Ratio):")
    ls_dfs = []
    for sym in args.symbols:
        print(f"  {sym}: ", end="", flush=True)
        df = download_long_short_ratio(sym, start_ms, end_ms)
        print(f"→ {len(df)} 条")
        if not df.empty:
            ls_dfs.append(df)

    if ls_dfs:
        ls_all = pd.concat(ls_dfs, ignore_index=True)
        save_parquet(ls_all, OUTPUT_DIR / "long_short_ratio_15m.parquet", "LS")
    print()

    print(f"{'=' * 60}")
    print(f"  全部完成！数据兼容现有 merge 函数，无需改训练代码。")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
