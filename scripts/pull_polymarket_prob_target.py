#!/usr/bin/env python3
"""
下载 Polymarket 15m **目标市场** 概率历史数据（Target Bar PM Probability）。

对每个 15m bar（时间戳 ts），获取 **下一个 bar**（ts+900）市场在 T-120s 时刻
（即当前 bar 收盘前 120 秒 = 预测触发时刻）的概率走势。

时序示例（预测 10:00-10:15）：
  当前 bar: 9:45-10:00 (ts = 9:45 的 Unix 秒)
  目标 bar: 10:00-10:15 (ts_target = ts + 900)
  预测时刻: 9:58:00 (T-120s)
  CLOB 查询: startTs=ts, endTs=ts+780 (9:45~9:58 的市场价格)

计算 5 个特征（前缀 target_）：
  - target_logit_p:        T-120s 时刻目标市场概率的 logit 变换
  - target_delta_prob_1m:  最后 1 分钟概率变化
  - target_delta_prob_3m:  最后 3 分钟概率变化
  - target_delta_prob_5m:  最后 5 分钟概率变化
  - target_prob_slope_12m: 12 分钟概率线性回归斜率

数据来源:
  - Gamma API: 通过 slug 查找 token ID
  - CLOB API: prices-history 获取分钟级概率

用法:
  python scripts/pull_polymarket_prob_target.py                    # 全部资产
  python scripts/pull_polymarket_prob_target.py --asset BTC_USDT   # 单个资产
  python scripts/pull_polymarket_prob_target.py --workers 2        # 并发数

输出:
  data/sentiment/polymarket_prob_target_<ASSET>.parquet
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from runtime_parquet_io import atomic_write_parquet

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_SENTIMENT = PROJECT_ROOT / "data" / "sentiment"
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com/prices-history"

ASSETS = ["BTC_USDT", "ETH_USDT", "XRP_USDT"]  # SOL 移除：AUC≈0.50，与 run_grid.py 对齐
SLUG_MAP = {"BTC_USDT": "btc", "ETH_USDT": "eth", "SOL_USDT": "sol", "XRP_USDT": "xrp"}

# Polymarket 15m 市场最早约 2025-10-09
EARLIEST_TS = int(pd.Timestamp("2025-10-08", tz="UTC").value // 10**9)

# T-120s: 预测在当前 bar 收盘前 120 秒触发
TRIGGER_BEFORE_CLOSE = 120

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Accept": "application/json",
}

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


def _fetch_one_target(slug_sym: str, current_bar_ts: int,
                      session: requests.Session) -> Optional[Dict]:
    """获取当前 bar 对应的 **目标市场**（下一个 bar）在 T-120s 时的概率特征。

    Args:
        slug_sym: Polymarket slug 前缀 (e.g. "btc")
        current_bar_ts: 当前 bar 开始的 Unix 秒时间戳
        session: requests 会话

    Returns:
        特征字典，timestamp_s 为 current_bar_ts（与 OHLCV 对齐）
    """
    target_bar_ts = current_bar_ts + 900  # 目标 bar（下一个 15m）
    slug = f"{slug_sym}-updown-15m-{target_bar_ts}"

    # 1) Gamma: 获取目标市场的 token ID
    event = None
    for attempt in range(MAX_RETRIES):
        try:
            r = session.get(
                f"{GAMMA_API}/events/slug/{slug}",
                headers=HEADERS,
                timeout=15,
            )
            if r.status_code == 404:
                return None  # 目标市场不存在
            r.raise_for_status()
            event = r.json()
            break
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_SLEEP * (attempt + 1))
            else:
                return None

    if event is None:
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

    # 2) CLOB: 获取目标市场在 T-120s 时刻之前的价格历史
    #    startTs = current_bar_ts (= target_bar_ts - 900)
    #    endTs   = current_bar_ts + 780 (= target_bar_ts - 120, 即 T-120s 时刻)
    end_ts = current_bar_ts + (900 - TRIGGER_BEFORE_CLOSE)  # = target_bar_ts - 120

    hist = None
    for attempt in range(MAX_RETRIES):
        try:
            r2 = session.get(
                CLOB_URL,
                params={
                    "market": token_id,
                    "startTs": current_bar_ts,
                    "endTs": end_ts,
                    "fidelity": 1,  # 1 分钟粒度
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

    # 目标市场刚开盘时经常只有 1-2 个点；只要有 last prob 就保留，
    # 缺失的 delta/slope 交给下游按 NaN 处理。
    if hist is None or len(hist) < 1:
        return None  # 数据点不足

    # 3) 计算特征（与 pull_polymarket_prob.py 相同逻辑，但列名加 target_ 前缀）
    prices = [float(h["p"]) for h in hist]
    p_last = prices[-1]

    # logit 变换
    p_clamp = max(min(p_last, 0.999), 0.001)
    logit_p = np.log(p_clamp / (1 - p_clamp))

    # 概率差值
    delta_1m = p_last - prices[-2] if len(prices) >= 2 else np.nan
    delta_3m = p_last - prices[-4] if len(prices) >= 4 else np.nan
    delta_5m = p_last - prices[-6] if len(prices) >= 6 else np.nan

    # 线性回归斜率
    n_slope = min(len(prices), 12)
    y = np.array(prices[-n_slope:])
    x = np.arange(n_slope, dtype=float)
    slope = np.polyfit(x, y, 1)[0] if n_slope >= 3 else np.nan

    return {
        "timestamp_s": current_bar_ts,  # 与 OHLCV 的当前 bar 对齐
        "target_logit_p": logit_p,
        "target_delta_prob_1m": delta_1m,
        "target_delta_prob_3m": delta_3m,
        "target_delta_prob_5m": delta_5m,
        "target_prob_slope_12m": slope,
        "target_raw_p_last": p_last,
        "target_n_points": len(prices),
    }


def _load_existing_target(asset: str) -> set:
    """加载已下载的时间戳以支持增量更新。"""
    out = DATA_SENTIMENT / f"polymarket_prob_target_{asset.lower()}.parquet"
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
    """下载单个资产的目标市场 Polymarket 概率特征。"""
    slug_sym = SLUG_MAP.get(asset)
    if not slug_sym:
        print(f"  ⚠️ 不支持资产: {asset}")
        return 0

    all_ts = _get_15m_timestamps(asset)
    if recent_hours is not None and recent_hours > 0:
        min_ts = int(time.time()) - int(recent_hours) * 3600
        all_ts = all_ts[all_ts >= min_ts]
    # 排除最后一个 bar（没有下一个 bar 作为目标）
    all_ts = all_ts[:-1] if len(all_ts) > 0 else all_ts

    existing = _load_existing_target(asset)
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
                pool.submit(_fetch_one_target, slug_sym, ts, session): ts
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
        out_path = DATA_SENTIMENT / f"polymarket_prob_target_{asset.lower()}.parquet"
        if out_path.exists():
            old_df = pd.read_parquet(out_path)
            combined = pd.concat([old_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["timestamp_s"], keep="last")
        else:
            combined = new_df
        combined = combined.sort_values("timestamp_s").reset_index(drop=True)
        atomic_write_parquet(combined, out_path)
        print(f"  ✅ 写入 {len(combined)} 行 → {out_path}")

    return len(results)


def main():
    ap = argparse.ArgumentParser(
        description="下载 Polymarket 15m 目标市场概率特征（Target Bar PM Probability）"
    )
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

    print("=" * 60)
    print("  Polymarket 目标市场概率回填")
    print(f"  T-{TRIGGER_BEFORE_CLOSE}s 模式: 获取下一个 bar 市场在预测时刻的概率")
    print(f"  资产: {', '.join(assets)}")
    print("=" * 60)

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
