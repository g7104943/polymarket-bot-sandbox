#!/usr/bin/env python3
"""
下载 Polymarket 15m 预开盘概率历史数据。

对每个 15m bar，拉取该市场 开盘前 15 分钟 的概率走势（fidelity=1min），
然后计算 5 个特征：
  - logit_p:        最后一分钟概率的 logit 变换
  - delta_prob_1m:  最后 1 分钟概率变化
  - delta_prob_3m:  最后 3 分钟概率变化
  - delta_prob_5m:  最后 5 分钟概率变化
  - prob_slope_12m: 12 分钟线性回归斜率（概率变化速度）

数据来源:
  - Gamma API: 通过 slug 查找 token ID
  - CLOB API: prices-history 获取分钟级概率

用法:
  python scripts/pull_polymarket_prob.py                    # 全部资产
  python scripts/pull_polymarket_prob.py --asset BTC_USDT   # 单个资产
  python scripts/pull_polymarket_prob.py --workers 4        # 并发数

输出:
  data/sentiment/polymarket_prob_<ASSET>.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_SENTIMENT = PROJECT_ROOT / "data" / "sentiment"


def _atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com/prices-history"

ASSETS = ["BTC_USDT", "ETH_USDT", "SOL_USDT", "XRP_USDT"]
SLUG_MAP = {"BTC_USDT": "btc", "ETH_USDT": "eth", "SOL_USDT": "sol", "XRP_USDT": "xrp"}

# Polymarket 15m 市场最早约 2025-10-09
EARLIEST_TS = int(pd.Timestamp("2025-10-08", tz="UTC").value // 10**9)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Accept": "application/json",
}

# 请求重试
MAX_RETRIES = 3
RETRY_SLEEP = 1.0


def _get_15m_timestamps(asset: str) -> np.ndarray:
    """从 OHLCV 数据获取所有 15m 时间戳（Unix 秒），只保留 Polymarket 有市场的时段。"""
    path = DATA_RAW / f"{asset.lower()}_15m.parquet"
    if not path.exists():
        raise FileNotFoundError(f"未找到 OHLCV: {path}")
    df = pd.read_parquet(path)
    ts = df["timestamp"].values
    if ts.dtype == np.int64 and ts[0] > 1e12:
        ts = ts // 1000  # ms → s
    ts = ts[ts >= EARLIEST_TS]
    return ts.astype(np.int64)


def _fetch_one(slug_sym: str, ts: int, session: requests.Session) -> Optional[Dict]:
    """获取一个 15m bar 的预开盘概率并计算特征。"""
    slug = f"{slug_sym}-updown-15m-{ts}"

    # 1) Gamma: 获取 token ID
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(
                f"{GAMMA_API}/events/slug/{slug}",
                headers=HEADERS,
                timeout=15,
            )
            if r.status_code == 404:
                return None  # 市场不存在
            r.raise_for_status()
            event = r.json()
            break
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_SLEEP * (attempt + 1))
            else:
                return None

    markets = event.get("markets", [])
    if not markets:
        return None

    cids = markets[0].get("clobTokenIds")
    if isinstance(cids, str):
        if cids.startswith("["):
            token_id = json.loads(cids)[0]
        else:
            token_id = cids.split(",")[0].strip()
    elif isinstance(cids, list):
        token_id = str(cids[0])
    else:
        return None

    # 2) CLOB: 获取预开盘概率（candle 前 15 分钟）
    for attempt in range(MAX_RETRIES):
        try:
            r2 = session.get(
                CLOB_URL,
                params={
                    "market": token_id,
                    "startTs": ts - 900,  # 前 15 分钟
                    "endTs": ts,          # candle 开始时
                    "fidelity": 1,        # 1 分钟粒度
                },
                headers=HEADERS,
                timeout=30,
            )
            r2.raise_for_status()
            hist = r2.json().get("history", [])
            break
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_SLEEP * (attempt + 1))
            else:
                return None

    # 新市场早期允许只有 1-2 个点，差分/斜率可以为空，但 last prob 仍然有价值。
    if len(hist) < 1:
        return None  # 数据点不足

    # 3) 计算特征
    prices = [float(h["p"]) for h in hist]
    p_last = prices[-1]

    # logit 变换：log(p / (1-p))
    p_clamp = max(min(p_last, 0.999), 0.001)
    logit_p = np.log(p_clamp / (1 - p_clamp))

    # 概率差值
    delta_1m = p_last - prices[-2] if len(prices) >= 2 else np.nan
    delta_3m = p_last - prices[-4] if len(prices) >= 4 else np.nan
    delta_5m = p_last - prices[-6] if len(prices) >= 6 else np.nan

    # 线性回归斜率（用最近 12 个点或全部）
    n_slope = min(len(prices), 12)
    y = np.array(prices[-n_slope:])
    x = np.arange(n_slope, dtype=float)
    if n_slope >= 3:
        slope = np.polyfit(x, y, 1)[0]
    else:
        slope = np.nan

    return {
        "timestamp_s": ts,
        "logit_p": logit_p,
        "delta_prob_1m": delta_1m,
        "delta_prob_3m": delta_3m,
        "delta_prob_5m": delta_5m,
        "prob_slope_12m": slope,
        "raw_p_last": p_last,
        "n_points": len(prices),
    }


def _load_existing(asset: str) -> set:
    """加载已下载的时间戳以支持增量更新。"""
    out = DATA_SENTIMENT / f"polymarket_prob_{asset.lower()}.parquet"
    if out.exists():
        df = pd.read_parquet(out)
        return set(df["timestamp_s"].values)
    return set()


def download_asset(
    asset: str,
    workers: int = 2,
    batch_size: int = 200,
    recent_hours: Optional[int] = None,
) -> int:
    """下载单个资产的 Polymarket 概率特征。"""
    slug_sym = SLUG_MAP.get(asset)
    if not slug_sym:
        print(f"  ⚠️ 不支持资产: {asset}")
        return 0

    all_ts = _get_15m_timestamps(asset)
    if recent_hours is not None and recent_hours > 0:
        min_ts = int(time.time()) - int(recent_hours) * 3600
        all_ts = all_ts[all_ts >= min_ts]
    existing = _load_existing(asset)
    todo = [int(t) for t in all_ts if int(t) not in existing]

    print(f"\n{'='*60}")
    print(f"  [{asset}] 总 15m bar: {len(all_ts)}, 已有: {len(existing)}, 待下载: {len(todo)}")
    if not todo:
        print("  ✅ 已是最新")
        return 0

    session = requests.Session()
    results: List[Dict] = []
    errors = 0
    t0 = time.time()

    for batch_start in range(0, len(todo), batch_size):
        batch = todo[batch_start: batch_start + batch_size]

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_fetch_one, slug_sym, ts, session): ts
                for ts in batch
            }
            for fut in as_completed(futures):
                res = fut.result()
                if res is not None:
                    results.append(res)
                else:
                    errors += 1

        done = batch_start + len(batch)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(todo) - done) / rate if rate > 0 else 0
        print(
            f"  进度: {done}/{len(todo)} "
            f"({done/len(todo)*100:.1f}%), "
            f"成功: {len(results)}, 失败: {errors}, "
            f"速度: {rate:.1f}/s, "
            f"ETA: {eta/60:.1f}min"
        )

    # 合并已有数据和新数据
    if results:
        new_df = pd.DataFrame(results)
        out_path = DATA_SENTIMENT / f"polymarket_prob_{asset.lower()}.parquet"
        if out_path.exists():
            old_df = pd.read_parquet(out_path)
            combined = pd.concat([old_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["timestamp_s"], keep="last")
        else:
            combined = new_df
        combined = combined.sort_values("timestamp_s").reset_index(drop=True)
        _atomic_write_parquet(combined, out_path)
        print(f"  ✅ 写入 {len(combined)} 行 → {out_path}")

    return len(results)


def main():
    ap = argparse.ArgumentParser(description="下载 Polymarket 15m 预开盘概率特征")
    ap.add_argument("--asset", type=str, default=None,
                    help="单个资产，如 BTC_USDT（默认全部）")
    ap.add_argument("--workers", type=int, default=2,
                    help="并发线程数（默认 2，别太高以免被限流）")
    ap.add_argument("--batch-size", type=int, default=200,
                    help="每批处理数（默认 200）")
    ap.add_argument("--recent-hours", type=int, default=None,
                    help="仅回填最近 N 小时（例如 72）")
    args = ap.parse_args()

    assets = [args.asset] if args.asset else ASSETS
    total = 0
    t0 = time.time()

    for asset in assets:
        total += download_asset(
            asset,
            workers=args.workers,
            batch_size=args.batch_size,
            recent_hours=args.recent_hours,
        )

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  全部完成: 共下载 {total} 条, 耗时 {elapsed/60:.1f} 分钟")
    return 0


if __name__ == "__main__":
    exit(main())
