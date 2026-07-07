#!/usr/bin/env python3
"""
拉取 CFGI.io 历史 15 分钟恐贪指数（FGI）。

API 文档: cfgi.io/user#api (v2)
端点: https://cfgi.io/api/api_request_v2.php
计费: 1 credit = 1 field × 1 token × 1 value
      只拉 cfgi 字段 → 1 credit/row/token

用法:
  # 测试（100 free credits，拉 BTC 最近 100 条 15m）
  python scripts/pull_cfgi_history.py --test

  # 拉 BTC 1 年
  python scripts/pull_cfgi_history.py --symbols BTC --days 365

  # 拉 4 币种 1 年（约 140,160 credits ≈ $28）
  python scripts/pull_cfgi_history.py --symbols BTC,ETH,SOL,XRP --days 365

  # 自定义日期范围
  python scripts/pull_cfgi_history.py --symbols BTC --start 2025-02-07 --end 2026-02-07
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

API_URL = "https://cfgi.io/api/api_request_v2.php"

# CFGI 时区是 CET (UTC+1)，但我们统一转为 UTC 存储
CET_OFFSET = timedelta(hours=1)


def fetch_cfgi_batch(
    api_key: str,
    token: str,
    start: str,
    end: str,
    fields: str = "cfgi",
    max_retries: int = 3,
) -> tuple[list[dict], dict]:
    """
    拉取一批 CFGI 数据。

    Returns
    -------
    (data_list, headers_info)
    """
    params = {
        "api_key": api_key,
        "token": token,
        "period": 1,        # 15 分钟
        "fields": fields,
        "start": start,
        "end": end,
    }

    for attempt in range(max_retries):
        try:
            r = requests.get(API_URL, params=params, timeout=30)

            headers_info = {
                "credits_used": r.headers.get("X-Credits-Used", "?"),
                "credits_remaining": r.headers.get("X-Credits-Remaining", "?"),
                "fields_requested": r.headers.get("X-Fields-Requested", "?"),
                "tokens_requested": r.headers.get("X-Tokens-Requested", "?"),
            }

            if r.status_code == 200:
                data = r.json()
                if isinstance(data, list):
                    return data, headers_info
                elif isinstance(data, dict) and "error" in data:
                    print(f"  [ERROR] {data['error']}")
                    return [], headers_info
                else:
                    return [], headers_info

            elif r.status_code == 402:
                print(f"  [ERROR] 积分不足: {r.json()}")
                return [], headers_info

            elif r.status_code == 205:
                # 速率限制：每秒最多 1 次
                wait = 2 ** attempt
                print(f"  [WARN] 速率限制，等 {wait}s...")
                time.sleep(wait)
                continue

            else:
                print(f"  [ERROR] HTTP {r.status_code}: {r.text[:200]}")
                return [], headers_info

        except Exception as e:
            wait = 2 ** attempt
            if attempt < max_retries - 1:
                print(f"  [WARN] 请求失败 (重试 {attempt+1}): {type(e).__name__}")
                time.sleep(wait)
            else:
                print(f"  [ERROR] 请求失败: {e}")
                return [], {}

    return [], {}


def pull_symbol(
    api_key: str,
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    fields: str = "cfgi",
    batch_days: int = 12,  # 每批 12 天 ≈ 1152 行（< 1200 限制）
) -> pd.DataFrame:
    """
    按时间分批拉取一个币种的历史 CFGI 数据。
    """
    all_data = []
    current_start = start_date
    batch_num = 0
    total_credits = 0

    while current_start < end_date:
        current_end = min(current_start + timedelta(days=batch_days), end_date)

        start_str = current_start.strftime("%Y-%m-%dT%H:%M:%S")
        end_str = current_end.strftime("%Y-%m-%dT%H:%M:%S")

        data, headers = fetch_cfgi_batch(api_key, symbol, start_str, end_str, fields)

        credits_used = headers.get("credits_used", "?")
        credits_remaining = headers.get("credits_remaining", "?")

        if credits_used != "?":
            total_credits += int(credits_used)

        batch_num += 1
        if data:
            all_data.extend(data)
            print(
                f"  [{symbol}] 批次 {batch_num}: {current_start.strftime('%Y-%m-%d')} ~ "
                f"{current_end.strftime('%Y-%m-%d')} → {len(data)} 行 | "
                f"credits: -{credits_used} (剩 {credits_remaining})"
            )
        else:
            print(
                f"  [{symbol}] 批次 {batch_num}: {current_start.strftime('%Y-%m-%d')} ~ "
                f"{current_end.strftime('%Y-%m-%d')} → 0 行 | "
                f"credits 剩: {credits_remaining}"
            )
            # 如果积分不足就停止
            if credits_remaining != "?" and int(credits_remaining) <= 0:
                print(f"  [STOP] 积分耗尽，已拉 {len(all_data)} 行")
                break

        current_start = current_end

        # 每秒最多 1 次请求
        time.sleep(1.1)

    if not all_data:
        return pd.DataFrame()

    df = pd.DataFrame(all_data)

    # 解析时间戳（CET → UTC）
    df["timestamp"] = pd.to_datetime(df["date"]) - CET_OFFSET
    df["cfgi_15m"] = pd.to_numeric(df["cfgi"], errors="coerce")
    df["symbol"] = symbol

    # 保留需要的列
    keep_cols = ["timestamp", "symbol", "cfgi_15m"]
    # 如果请求了多个字段，也保留
    for col in df.columns:
        if col.startswith("data_") and col not in keep_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            keep_cols.append(col)
    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        keep_cols.append("price")

    df = df[[c for c in keep_cols if c in df.columns]]
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    df = df.reset_index(drop=True)

    print(f"  [{symbol}] 总计: {len(df)} 行, credits 消耗: {total_credits}")

    return df


def main():
    parser = argparse.ArgumentParser(description="拉取 CFGI.io 历史 15m 恐贪指数")
    parser.add_argument("--api-key", type=str, default="28347_0ce1_570a6f6871")
    parser.add_argument("--symbols", type=str, default="BTC",
                        help="逗号分隔的币种 (例: BTC,ETH,SOL,XRP)")
    parser.add_argument("--days", type=int, default=None, help="回溯天数")
    parser.add_argument("--start", type=str, default=None, help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="结束日期 YYYY-MM-DD")
    parser.add_argument("--fields", type=str, default="cfgi",
                        help="请求的字段 (默认只拉 cfgi，省 credits)")
    parser.add_argument("--output", type=str,
                        default="data/sentiment/cfgi_15m_history.parquet")
    parser.add_argument("--test", action="store_true",
                        help="测试模式：只拉 BTC 最近 1 天（约 96 credits）")
    args = parser.parse_args()

    # 解析日期
    now = datetime.now(timezone.utc)
    if args.test:
        symbols = ["BTC"]
        start_date = now - timedelta(days=1)
        end_date = now
        print("=== 测试模式：BTC 最近 1 天（约 96 credits） ===")
    elif args.start and args.end:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
        start_date = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end_date = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    elif args.days:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
        start_date = now - timedelta(days=args.days)
        end_date = now
    else:
        print("请指定 --days 或 --start/--end 或 --test")
        sys.exit(1)

    n_fields = len(args.fields.split(","))
    estimated_rows = int((end_date - start_date).total_seconds() / 900)  # 15min = 900s
    estimated_credits = estimated_rows * len(symbols) * n_fields
    estimated_cost = estimated_credits * 0.0002

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  CFGI.io 历史 15m FGI 拉取")
    print("=" * 60)
    print(f"  币种:     {', '.join(symbols)}")
    print(f"  时间:     {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
    print(f"  字段:     {args.fields} ({n_fields} 个)")
    print(f"  预估行数: {estimated_rows:,} × {len(symbols)} 币种 = {estimated_rows * len(symbols):,}")
    print(f"  预估消耗: {estimated_credits:,} credits ≈ ${estimated_cost:.2f}")
    print(f"  输出:     {args.output}")
    print("=" * 60)

    # 加载已有数据（增量追加，避免重复消耗 credits）
    existing = pd.DataFrame()
    if output_path.exists():
        existing = pd.read_parquet(output_path)
        print(f"\n  已有数据: {len(existing)} 行")
        for sym in existing["symbol"].unique():
            sub = existing[existing["symbol"] == sym]
            print(f"    {sym}: {len(sub)} 行, {sub['timestamp'].min()} ~ {sub['timestamp'].max()}")

    all_dfs = []
    for symbol in symbols:
        # 检查该币种已有数据，跳过已覆盖的时间段
        sym_start = start_date
        sym_end = end_date
        if not existing.empty and symbol in existing["symbol"].values:
            sym_existing = existing[existing["symbol"] == symbol]
            # 已有数据可能是 tz-naive，统一转为 tz-aware UTC
            _emin = pd.Timestamp(sym_existing["timestamp"].min())
            _emax = pd.Timestamp(sym_existing["timestamp"].max())
            existing_min = _emin.tz_localize("UTC") if _emin.tzinfo is None else _emin
            existing_max = _emax.tz_localize("UTC") if _emax.tzinfo is None else _emax

            # 如果已有数据完全覆盖请求范围 → 跳过
            if existing_min <= sym_start and existing_max >= sym_end - timedelta(hours=1):
                print(f"\n{symbol}: 已有数据完全覆盖，跳过")
                continue

            # 只拉缺失的前段（已有数据之前）和后段（已有数据之后）
            parts_to_pull = []
            if sym_start < existing_min - timedelta(hours=1):
                parts_to_pull.append((sym_start, existing_min - timedelta(minutes=15)))
            if sym_end > existing_max + timedelta(hours=1):
                parts_to_pull.append((existing_max + timedelta(minutes=15), sym_end))

            if not parts_to_pull:
                print(f"\n{symbol}: 已有数据基本覆盖，跳过")
                continue

            print(f"\n{symbol}: 增量拉取 {len(parts_to_pull)} 段（跳过已有 {len(sym_existing)} 行）")
            for p_start, p_end in parts_to_pull:
                days = (p_end - p_start).days
                print(f"  段: {p_start.strftime('%Y-%m-%d')} ~ {p_end.strftime('%Y-%m-%d')} ({days}天)")
                df = pull_symbol(args.api_key, symbol, p_start, p_end, args.fields)
                if not df.empty:
                    all_dfs.append(df)
        else:
            print(f"\n{symbol}: 全量拉取 {(sym_end - sym_start).days} 天")
            df = pull_symbol(args.api_key, symbol, sym_start, sym_end, args.fields)
            if not df.empty:
                all_dfs.append(df)

    if not all_dfs and existing.empty:
        print("\n[ERROR] 无数据拉取到")
        sys.exit(1)

    # 合并新数据 + 已有数据
    if all_dfs:
        new_data = pd.concat(all_dfs, ignore_index=True)
        if not existing.empty:
            combined = pd.concat([existing, new_data], ignore_index=True)
        else:
            combined = new_data
    else:
        combined = existing
        print("\n  无新数据需要拉取，已有数据不变")

    combined = combined.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    combined = combined.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
    combined.to_parquet(output_path, index=False)

    print(f"\n{'=' * 60}")
    print(f"  完成!")
    print(f"  总行数: {len(combined)}")
    for sym in symbols:
        sub = combined[combined["symbol"] == sym]
        if not sub.empty:
            print(f"    {sym}: {len(sub)} 行, "
                  f"{sub['timestamp'].min()} ~ {sub['timestamp'].max()}, "
                  f"FGI range: {sub['cfgi_15m'].min():.1f} ~ {sub['cfgi_15m'].max():.1f}")
    print(f"  已保存: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
