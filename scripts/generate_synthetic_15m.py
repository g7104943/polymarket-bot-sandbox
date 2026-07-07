#!/usr/bin/env python3
"""
生成合成 15m OHLCV 数据用于跑通 LSTM+XGB 流程（无真实 API）。
保存到 data/raw/btc_usdt_15m.parquet、eth_usdt_15m.parquet。
约 3 年数据，可 2 年训练 + 1 年回测；或与 download_15m_historical 真实数据并存时会被覆盖。
写入通过 data_fetcher.save_ohlcv，与预测写入器共用文件锁，避免并发写损坏。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_RAW = PROJECT_ROOT / "data" / "raw"
BAR_MS = 15 * 60 * 1000


def generate_synthetic_15m(
    years: int = 7,
    seed: int = 42,
    start_price_btc: float = 30000.0,
    start_price_eth: float = 2000.0,
) -> tuple:
    """生成 years 年的 15m K 线（随机游走），返回 (btc_df, eth_df)。"""
    rng = np.random.default_rng(seed)
    n_bars = years * 365 * 24 * 4  # 15m bars per year
    t0 = 1704067200000  # 2024-01-01 00:00:00 UTC
    timestamps = t0 + np.arange(n_bars) * BAR_MS

    def one_asset(start_price: float, vol: float) -> pd.DataFrame:
        log_ret = rng.normal(0, vol, n_bars)
        close = start_price * np.exp(np.cumsum(log_ret))
        high = close * (1 + np.abs(rng.normal(0, 0.002, n_bars)))
        low = close * (1 - np.abs(rng.normal(0, 0.002, n_bars)))
        open_ = np.roll(close, 1)
        open_[0] = start_price
        volume = rng.lognormal(10, 1, n_bars).astype(np.float64)
        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": open_.clip(1e-6, None),
            "high": high.clip(1e-6, None),
            "low": low.clip(1e-6, None),
            "close": close.clip(1e-6, None),
            "volume": volume,
        })
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    btc = one_asset(start_price_btc, 0.0012)
    eth = one_asset(start_price_eth, 0.0015)
    return btc, eth


def main():
    import argparse
    from src.python.data_fetcher import save_ohlcv

    ap = argparse.ArgumentParser(description="生成合成 15m 数据")
    ap.add_argument("--years", type=int, default=7, help="生成年数，默认 7（与 5 年训练+2 年回测一致）")
    args = ap.parse_args()

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    btc_df, eth_df = generate_synthetic_15m(years=args.years)
    save_ohlcv(btc_df, "BTC/USDT", "15m")
    save_ohlcv(eth_df, "ETH/USDT", "15m")
    btc_path = DATA_RAW / "btc_usdt_15m.parquet"
    eth_path = DATA_RAW / "eth_usdt_15m.parquet"
    print("Generated:", btc_path, "rows:", len(btc_df))
    print("Generated:", eth_path, "rows:", len(eth_df))


if __name__ == "__main__":
    main()
