#!/usr/bin/env python3
"""
拉取 Polymarket 15 分钟加密市场每个 bar 内的 1 分钟价格数据。
═══════════════════════════════════════════════════════════════
目的：为 Post-only Maker 回测提供 intra-bar low/high，准确判断挂单是否会成交。

数据源（均无需认证）：
  1. Gamma API → 通过 slug 查找 UP token 的 token_id
  2. CLOB prices-history → 用 token_id 拉取 bar 内 1 分钟级价格

输出：data/polymarket_1m_intrabar_eth_usdt.parquet
  列: bar_ts, p_open, p_close, p_low, p_high, n_points

用法:
  python3 scripts/pull_polymarket_1m_intrabar.py                    # ETH（默认）
  python3 scripts/pull_polymarket_1m_intrabar.py --asset BTC_USDT   # BTC
  python3 scripts/pull_polymarket_1m_intrabar.py --workers 3        # 3 线程
  python3 scripts/pull_polymarket_1m_intrabar.py --resume            # 断点续拉（跳过已有 bar）

约 12000 个 bar × 2 API 调用 = ~24000 请求，预计 1~2 小时。
支持增量续拉：已拉取的 bar 会跳过。
"""

import json
import time
import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_SENTIMENT = PROJECT_ROOT / "data" / "sentiment"
DATA_OUT = PROJECT_ROOT / "data"

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_URL = "https://clob.polymarket.com/prices-history"

SLUG_MAP = {"BTC_USDT": "btc", "ETH_USDT": "eth", "SOL_USDT": "sol", "XRP_USDT": "xrp"}
EARLIEST_TS = int(pd.Timestamp("2025-10-08", tz="UTC").value // 10**9)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Accept": "application/json",
}
MAX_RETRIES = 3
RETRY_SLEEP = 1.5


def get_bar_timestamps(asset: str) -> np.ndarray:
    """从已有 polymarket_prob 数据获取所有 bar 的时间戳。"""
    prob_path = DATA_SENTIMENT / f"polymarket_prob_{asset.lower()}.parquet"
    if prob_path.exists():
        df = pd.read_parquet(prob_path)
        return np.sort(df["timestamp_s"].values.astype(np.int64))

    # 回退到 OHLCV 时间戳
    ohlcv_path = DATA_RAW / f"{asset.lower()}_15m.parquet"
    if not ohlcv_path.exists():
        raise FileNotFoundError(f"未找到数据: {prob_path} 或 {ohlcv_path}")
    df = pd.read_parquet(ohlcv_path)
    ts = df["timestamp"].values
    if ts.dtype == np.int64 and ts[0] > 1e12:
        ts = ts // 1000
    return np.sort(ts[ts >= EARLIEST_TS].astype(np.int64))


def load_existing(asset: str) -> set:
    """加载已拉取的 bar 时间戳（增量续拉用）。"""
    out_path = DATA_OUT / f"polymarket_1m_intrabar_{asset.lower()}.parquet"
    if out_path.exists():
        df = pd.read_parquet(out_path)
        return set(df["bar_ts"].values)
    return set()


def fetch_one_bar(slug_sym: str, bar_ts: int, session: requests.Session) -> Optional[Dict]:
    """
    拉取一个 15 分钟 bar 内的 1 分钟价格数据。

    步骤:
      1. Gamma API → token_id (通过 slug: {slug_sym}-updown-15m-{bar_ts})
      2. CLOB prices-history → bar 内 1 分钟价格 (bar_ts ~ bar_ts+900, fidelity=1)
      3. 计算 p_open, p_close, p_low, p_high
    """
    slug = f"{slug_sym}-updown-15m-{bar_ts}"

    # 1) Gamma: 获取 token ID
    token_id = None
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

            if token_id:
                break
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_SLEEP * (attempt + 1))
            else:
                return None

    if not token_id:
        return None

    # 2) CLOB: bar 内 1 分钟价格 (bar 起始到结束)
    prices = []
    for attempt in range(MAX_RETRIES):
        try:
            r2 = session.get(
                CLOB_URL,
                params={
                    "market": token_id,
                    "startTs": bar_ts,       # bar 开始
                    "endTs": bar_ts + 900,   # bar 结束（15 分钟 = 900 秒）
                    "fidelity": 1,           # 1 分钟粒度
                },
                headers=HEADERS,
                timeout=30,
            )
            r2.raise_for_status()
            hist = r2.json().get("history", [])
            prices = [float(h["p"]) for h in hist if "p" in h]
            break
        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_SLEEP * (attempt + 1))
            else:
                return None

    if len(prices) < 2:
        return None  # 数据点不足

    return {
        "bar_ts": bar_ts,
        "p_open": prices[0],
        "p_close": prices[-1],
        "p_low": min(prices),
        "p_high": max(prices),
        "n_points": len(prices),
    }


def save_incremental(results: List[Dict], asset: str):
    """增量保存结果（合并已有数据）。"""
    out_path = DATA_OUT / f"polymarket_1m_intrabar_{asset.lower()}.parquet"
    new_df = pd.DataFrame(results)

    if out_path.exists():
        old_df = pd.read_parquet(out_path)
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["bar_ts"], keep="last")
    else:
        combined = new_df

    combined = combined.sort_values("bar_ts").reset_index(drop=True)
    combined.to_parquet(out_path, index=False)
    return len(combined)


def download_asset(asset: str, workers: int = 2, batch_size: int = 100):
    """拉取单个资产的全部 bar 内 1 分钟数据。"""
    slug_sym = SLUG_MAP.get(asset)
    if not slug_sym:
        print(f"  ⚠️ 不支持资产: {asset}")
        return 0

    all_ts = get_bar_timestamps(asset)
    existing = load_existing(asset)
    todo = [int(t) for t in all_ts if int(t) not in existing]

    print(f"\n{'='*70}")
    print(f"  [{asset}] 总 bar: {len(all_ts)}, 已有: {len(existing)}, 待拉取: {len(todo)}")
    if not todo:
        print("  ✅ 已是最新")
        return 0

    # 预估时间
    est_sec = len(todo) * 2 / max(workers, 1)  # ~2 API 调用/bar
    print(f"  预计耗时: {est_sec/60:.0f} 分钟 ({workers} 线程)")
    print(f"{'='*70}")

    session = requests.Session()
    results: List[Dict] = []
    errors = 0
    t0 = time.time()
    total_saved = len(existing)

    for batch_start in range(0, len(todo), batch_size):
        batch = todo[batch_start: batch_start + batch_size]

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(fetch_one_bar, slug_sym, ts, session): ts
                for ts in batch
            }
            for fut in as_completed(futures):
                res = fut.result()
                if res is not None:
                    results.append(res)
                else:
                    errors += 1

        # 每批保存一次（防止中途中断丢失数据）
        if results:
            total_saved = save_incremental(results, asset)
            results = []  # 清空已保存的

        done = batch_start + len(batch)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(todo) - done) / rate if rate > 0 else 0
        print(
            f"  进度: {done}/{len(todo)} "
            f"({done/len(todo)*100:.1f}%), "
            f"成功: {total_saved - len(existing)}, 失败: {errors}, "
            f"速度: {rate:.1f} bar/s, "
            f"ETA: {eta/60:.1f} min"
        )

    # 最终保存
    if results:
        total_saved = save_incremental(results, asset)

    print(f"\n  ✅ [{asset}] 完成: 总 {total_saved} bar 已保存")
    return total_saved - len(existing)


def main():
    ap = argparse.ArgumentParser(
        description="拉取 Polymarket 15m bar 内 1 分钟价格 (p_low/p_high/p_open/p_close)"
    )
    ap.add_argument("--asset", type=str, default="ETH_USDT",
                    help="资产，如 ETH_USDT / BTC_USDT（默认 ETH_USDT）")
    ap.add_argument("--workers", type=int, default=2,
                    help="并发线程数（默认 2，太高可能被限流）")
    ap.add_argument("--batch-size", type=int, default=100,
                    help="每批处理数（默认 100，每批自动保存）")
    args = ap.parse_args()

    t0 = time.time()
    new_count = download_asset(args.asset, workers=args.workers, batch_size=args.batch_size)
    elapsed = time.time() - t0

    print(f"\n{'='*70}")
    print(f"  全部完成: 新增 {new_count} bar, 耗时 {elapsed/60:.1f} 分钟")
    print(f"  输出: data/polymarket_1m_intrabar_{args.asset.lower()}.parquet")
    print(f"{'='*70}")


if __name__ == "__main__":
    sys.exit(main() or 0)
