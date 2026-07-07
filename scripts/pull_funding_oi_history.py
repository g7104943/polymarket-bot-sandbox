#!/usr/bin/env python3
"""
下载 Binance 永续合约的资金费率、持仓量、多空比、主动买卖比历史数据。

这些是加密货币量化中最重要的非价格数据源：
  - 资金费率（Funding Rate）：每 8h 一次，正值=多头付费给空头（看涨过度）
  - 持仓量（Open Interest）：15m 粒度，增加=新资金入场
  - 多空比（Long/Short Ratio）：账户多空比例
  - 主动买卖比（Taker Buy/Sell Ratio）：主动买入/卖出的比例

用法:
    # 下载全部（约 5 分钟）
    python scripts/pull_funding_oi_history.py

    # 只下载指定币种
    python scripts/pull_funding_oi_history.py --symbols BTCUSDT ETHUSDT

    # 指定时间范围
    python scripts/pull_funding_oi_history.py --start-date 2023-08-01

    # 5m 粒度（Exp13: BTC 5min 预测）
    python scripts/pull_funding_oi_history.py --period 5m --symbols BTCUSDT --start-date 2024-01-01

数据保存到:
    data/sentiment/funding_rate_history.parquet
    data/sentiment/open_interest_15m.parquet  (或 open_interest_5m.parquet)
    data/sentiment/long_short_ratio_15m.parquet  (或 long_short_ratio_5m.parquet)
    data/sentiment/taker_buy_sell_15m.parquet  (或 taker_buy_sell_5m.parquet)
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "sentiment"


# ─── Binance API ──────────────────────────────────────────

BASE_URL = "https://fapi.binance.com"

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]

# API 端点和配置
ENDPOINTS = {
    "funding_rate": {
        "url": "/fapi/v1/fundingRate",
        "limit": 1000,
        "period_param": None,  # 不需要 period 参数
        "time_field": "fundingTime",
        "desc": "资金费率",
    },
    "open_interest": {
        "url": "/futures/data/openInterestHist",
        "limit": 500,
        "period_param": "15m",
        "time_field": "timestamp",
        "desc": "持仓量",
    },
    "long_short_ratio": {
        "url": "/futures/data/globalLongShortAccountRatio",
        "limit": 500,
        "period_param": "15m",
        "time_field": "timestamp",
        "desc": "多空比",
    },
    "taker_buy_sell": {
        "url": "/futures/data/takerlongshortRatio",
        "limit": 500,
        "period_param": "15m",
        "time_field": "timestamp",
        "desc": "主动买卖比",
    },
}


def fetch_paginated(
    endpoint_key: str,
    symbol: str,
    start_ms: int,
    end_ms: int,
    max_retries: int = 3,
) -> list[dict]:
    """分页拉取 Binance API 数据。

    对 /futures/data/* 端点，Binance 要求 startTime ~ endTime 跨度不超过 30 天，
    因此先按 30 天窗口分段，每段内再用 limit 分页。
    """
    cfg = ENDPOINTS[endpoint_key]
    url = BASE_URL + cfg["url"]
    limit = cfg["limit"]
    all_records = []

    # /futures/data/* 端点最大时间跨度约 30 天（15m period），分段处理
    CHUNK_MS = 29 * 24 * 3600 * 1000  # 29 天（留 1 天余量）
    chunk_start = start_ms

    while chunk_start < end_ms:
        chunk_end = min(chunk_start + CHUNK_MS, end_ms)
        cursor = chunk_start

        while cursor < chunk_end:
            params = {
                "symbol": symbol,
                "startTime": cursor,
                "endTime": chunk_end,
                "limit": limit,
            }
            if cfg["period_param"]:
                params["period"] = cfg["period_param"]

            data = None
            for attempt in range(max_retries):
                try:
                    resp = requests.get(url, params=params, timeout=30)
                    if resp.status_code == 429:
                        wait = int(resp.headers.get("Retry-After", 10))
                        print(f"    限速，等待 {wait}s...")
                        time.sleep(wait)
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2 * (attempt + 1))
                    else:
                        print(f"    API 错误 ({symbol} {endpoint_key}): {e}")
                        data = []

            if not data:
                break

            all_records.extend(data)

            last_ts = int(data[-1].get(cfg["time_field"], data[-1].get("timestamp", 0)))
            if last_ts <= cursor:
                break
            cursor = last_ts + 1

            time.sleep(0.2)

        chunk_start = chunk_end + 1

    return all_records


def process_funding_rate(records: list[dict], symbol: str) -> pd.DataFrame:
    """处理资金费率数据。"""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_numeric(df["fundingTime"])
    df["funding_rate"] = pd.to_numeric(df["fundingRate"])
    df["symbol"] = symbol
    df = df[["timestamp", "symbol", "funding_rate"]]
    return df.drop_duplicates(subset=["timestamp", "symbol"]).sort_values("timestamp")


def process_open_interest(records: list[dict], symbol: str) -> pd.DataFrame:
    """处理持仓量数据。"""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    df["open_interest"] = pd.to_numeric(df["sumOpenInterest"])
    df["oi_value"] = pd.to_numeric(df["sumOpenInterestValue"])
    df["symbol"] = symbol
    df = df[["timestamp", "symbol", "open_interest", "oi_value"]]
    return df.drop_duplicates(subset=["timestamp", "symbol"]).sort_values("timestamp")


def process_long_short_ratio(records: list[dict], symbol: str) -> pd.DataFrame:
    """处理多空比数据。"""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    df["long_short_ratio"] = pd.to_numeric(df["longShortRatio"])
    df["long_account"] = pd.to_numeric(df["longAccount"])
    df["short_account"] = pd.to_numeric(df["shortAccount"])
    df["symbol"] = symbol
    df = df[["timestamp", "symbol", "long_short_ratio", "long_account", "short_account"]]
    return df.drop_duplicates(subset=["timestamp", "symbol"]).sort_values("timestamp")


def process_taker_buy_sell(records: list[dict], symbol: str) -> pd.DataFrame:
    """处理主动买卖比数据。"""
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    df["taker_buy_sell_ratio"] = pd.to_numeric(df["buySellRatio"])
    df["taker_buy_vol"] = pd.to_numeric(df["buyVol"])
    df["taker_sell_vol"] = pd.to_numeric(df["sellVol"])
    df["symbol"] = symbol
    df = df[["timestamp", "symbol", "taker_buy_sell_ratio", "taker_buy_vol", "taker_sell_vol"]]
    return df.drop_duplicates(subset=["timestamp", "symbol"]).sort_values("timestamp")


PROCESSORS = {
    "funding_rate": process_funding_rate,
    "open_interest": process_open_interest,
    "long_short_ratio": process_long_short_ratio,
    "taker_buy_sell": process_taker_buy_sell,
}

# 输出文件名（按 period 动态生成）
OUTPUT_FILES_15M = {
    "funding_rate": "funding_rate_history.parquet",
    "open_interest": "open_interest_15m.parquet",
    "long_short_ratio": "long_short_ratio_15m.parquet",
    "taker_buy_sell": "taker_buy_sell_15m.parquet",
}

OUTPUT_FILES_5M = {
    "funding_rate": "funding_rate_history.parquet",  # 资金费率不分 period
    "open_interest": "open_interest_5m.parquet",
    "long_short_ratio": "long_short_ratio_5m.parquet",
    "taker_buy_sell": "taker_buy_sell_5m.parquet",
}


def get_output_files(period: str = "15m") -> dict:
    """根据 period 返回对应的输出文件名映射。"""
    return OUTPUT_FILES_5M if period == "5m" else OUTPUT_FILES_15M


# 兼容旧代码的默认引用
OUTPUT_FILES = OUTPUT_FILES_15M


def download_one_type(
    endpoint_key: str,
    symbols: list[str],
    start_ms: int,
    end_ms: int,
    output_files: dict | None = None,
) -> pd.DataFrame:
    """下载一种数据类型的全部币种数据。"""
    cfg = ENDPOINTS[endpoint_key]
    processor = PROCESSORS[endpoint_key]
    all_dfs = []
    _output_files = output_files or OUTPUT_FILES

    for sym in symbols:
        print(f"  {sym}: ", end="", flush=True)

        # 检查已有数据，增量下载
        output_path = OUTPUT_DIR / _output_files[endpoint_key]
        actual_start = start_ms
        if output_path.exists():
            existing = pd.read_parquet(output_path)
            sym_data = existing[existing["symbol"] == sym] if "symbol" in existing.columns else existing
            if len(sym_data) > 0:
                max_ts = int(sym_data["timestamp"].max())
                if max_ts > actual_start:
                    actual_start = max_ts + 1
                    print(f"增量追加（已有到 {pd.Timestamp(max_ts, unit='ms').strftime('%Y-%m-%d')}）", end="", flush=True)

        records = fetch_paginated(endpoint_key, sym, actual_start, end_ms)
        df = processor(records, sym)
        print(f"→ {len(df)} 条", flush=True)

        if not df.empty:
            all_dfs.append(df)

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    return pd.DataFrame()


def save_with_merge(df: pd.DataFrame, output_path: Path):
    """保存数据，与已有数据合并。"""
    if df.empty:
        return

    if output_path.exists():
        existing = pd.read_parquet(output_path)
        df = pd.concat([existing, df], ignore_index=True)

    # 去重
    key_cols = ["timestamp", "symbol"] if "symbol" in df.columns else ["timestamp"]
    df = df.drop_duplicates(subset=key_cols, keep="last")
    df = df.sort_values(key_cols).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    kb = output_path.stat().st_size / 1024
    print(f"  💾 {output_path.name}: {len(df)} 条 ({kb:.0f}KB)")


def main():
    parser = argparse.ArgumentParser(description="下载 Binance 资金费率/持仓量/多空比历史数据")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS)
    parser.add_argument("--start-date", default="2023-08-01", help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", default=None, help="结束日期（默认今天）")
    parser.add_argument("--types", nargs="+", default=list(ENDPOINTS.keys()),
                        help="数据类型: funding_rate, open_interest, long_short_ratio, taker_buy_sell")
    parser.add_argument("--period", default="15m", choices=["5m", "15m"],
                        help="OI/LS/Taker 数据粒度 (默认 15m, Exp13 用 5m)")
    args = parser.parse_args()

    period = args.period
    output_files = get_output_files(period)

    start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end_date, "%Y-%m-%d") if args.end_date else datetime.utcnow()
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    # 动态覆盖 ENDPOINTS 中的 period_param
    if period != "15m":
        for key in ENDPOINTS:
            if ENDPOINTS[key]["period_param"] is not None:
                ENDPOINTS[key]["period_param"] = period

    print(f"{'=' * 60}")
    print(f"  Binance 永续合约数据下载")
    print(f"{'=' * 60}")
    print(f"  币种:  {', '.join(args.symbols)}")
    print(f"  范围:  {args.start_date} → {end_dt.strftime('%Y-%m-%d')}")
    print(f"  类型:  {', '.join(args.types)}")
    print(f"  粒度:  {period}")
    print(f"{'=' * 60}\n")

    # Binance /futures/data/* 端点只保留最近 ~30 天数据
    # 只有 /fapi/v1/fundingRate 有完整历史
    DATA_API_MAX_DAYS = 30
    data_api_start_ms = int((end_dt - timedelta(days=DATA_API_MAX_DAYS)).timestamp() * 1000)

    for data_type in args.types:
        if data_type not in ENDPOINTS:
            print(f"  未知类型: {data_type}，跳过")
            continue

        cfg = ENDPOINTS[data_type]
        # funding_rate 用用户指定的起始时间；其他端点强制用近 30 天
        if data_type == "funding_rate":
            effective_start = start_ms
        else:
            effective_start = max(start_ms, data_api_start_ms)
            print(f"  ⚠️ Binance {data_type} 仅保留最近 ~30 天数据")

        print(f"📥 {cfg['desc']}（{data_type}, period={cfg.get('period_param', 'N/A')}）:")

        df = download_one_type(data_type, args.symbols, effective_start, end_ms,
                               output_files=output_files)
        output_path = OUTPUT_DIR / output_files[data_type]
        save_with_merge(df, output_path)
        print()

    print(f"{'=' * 60}")
    print(f"  全部完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
