#!/usr/bin/env python3
"""
离线分析脚本：从历史 Bybit 成交数据（CSV 或 .data 文件）计算
4 个币种的 VPIN 分布统计量（mean / std / 85th percentile），
用于调参和设定阈值。

支持的输入格式：
  1. CSV（含 timestamp, side, size_usdt 列）
  2. Bybit 历史数据下载的 .data / .csv.gz 文件
     （每行一条成交: timestamp, symbol, side, size, price, ...）

用法：
  # 从 CSV
  python scripts/analyze_vpin_history.py --input data/trades/BTCUSDT_trades.csv

  # 从 Bybit 下载目录（自动扫描所有 .csv.gz / .data 文件）
  python scripts/analyze_vpin_history.py --input-dir data/bybit_trades/ --symbols BTCUSDT ETHUSDT

  # 自定义参数
  python scripts/analyze_vpin_history.py --input-dir data/bybit_trades/ \\
      --bucket-volume 3000 --lookback-buckets 100

输出：
  - 终端打印每个币种的 VPIN 统计表
  - 保存 CSV: vpin_statistics_{symbol}.csv（VPIN 时间序列）
  - 保存 JSON: vpin_thresholds.json（推荐阈值参数）
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# 确保能 import 项目模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.python.features.vpin_filter import CryptoVPINFilter


def parse_args():
    p = argparse.ArgumentParser(description="VPIN 历史分析")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--input", type=str, help="单个 CSV/gz 文件路径")
    g.add_argument("--input-dir", type=str, help="Bybit 下载数据目录")

    p.add_argument(
        "--symbols", nargs="+",
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
        help="要分析的交易对",
    )
    p.add_argument("--bucket-volume", type=float, default=5000.0,
                   help="每桶 USDT 成交额 (默认 5000)")
    p.add_argument("--lookback-buckets", type=int, default=80,
                   help="滚动窗口桶数 (默认 80)")
    p.add_argument("--output-dir", type=str, default="data/vpin_analysis",
                   help="输出目录")
    return p.parse_args()


def load_csv_trades(filepath: str, target_symbol: str = None) -> List[Dict]:
    """
    加载 CSV 格式的成交数据。

    支持的列名：
      方式 1: timestamp, side, size_usdt  (已计算好 USDT 金额)
      方式 2: timestamp, symbol, side, size/qty, price  (原始 Bybit 格式)
    """
    trades = []
    opener = gzip.open if filepath.endswith(".gz") else open

    with opener(filepath, "rt", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            # 无表头，尝试按位置解析 Bybit 格式
            f.seek(0)
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 5:
                    continue
                try:
                    ts = float(parts[0])
                    symbol = parts[1].strip()
                    side = parts[2].strip()
                    qty = float(parts[3])
                    price = float(parts[4])
                except (ValueError, IndexError):
                    continue

                if target_symbol and symbol != target_symbol:
                    continue

                trades.append({
                    "timestamp": ts,
                    "side": side,
                    "size_usdt": qty * price,
                })
        else:
            fields = set(reader.fieldnames)
            for row in reader:
                try:
                    # 方式 1: 已有 size_usdt
                    if "size_usdt" in fields:
                        ts = float(row.get("timestamp", 0))
                        side = row.get("side", "Buy")
                        size_usdt = float(row.get("size_usdt", 0))
                    # 方式 2: 需要自己算
                    elif "price" in fields:
                        ts = float(row.get("timestamp", row.get("T", 0)))
                        sym = row.get("symbol", row.get("s", ""))
                        if target_symbol and sym and sym != target_symbol:
                            continue
                        side = row.get("side", row.get("S", "Buy"))
                        qty = float(row.get("size", row.get("v", row.get("qty", 0))))
                        price = float(row.get("price", row.get("p", 0)))
                        size_usdt = qty * price
                    else:
                        continue

                    if size_usdt > 0:
                        trades.append({
                            "timestamp": ts,
                            "side": side,
                            "size_usdt": size_usdt,
                        })
                except (ValueError, KeyError):
                    continue

    return sorted(trades, key=lambda x: x["timestamp"])


def find_trade_files(input_dir: str, symbol: str) -> List[str]:
    """在目录中查找匹配的成交文件。"""
    d = Path(input_dir)
    patterns = [
        f"*{symbol}*trade*.csv",
        f"*{symbol}*trade*.csv.gz",
        f"*{symbol}*.data",
        f"*{symbol}*.data.gz",
        f"{symbol}*.csv",
        f"{symbol}*.csv.gz",
    ]
    files = []
    for pat in patterns:
        files.extend(d.glob(pat))
        files.extend(d.rglob(pat))

    return sorted(set(str(f) for f in files))


def load_bybit_data_file(filepath: str) -> List[Dict]:
    """
    加载 Bybit .data 文件（每行一个 JSON snapshot 或成交记录）。
    """
    trades = []
    opener = gzip.open if filepath.endswith(".gz") else open

    try:
        with opener(filepath, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # JSON 格式
                if line.startswith("{"):
                    try:
                        obj = json.loads(line)
                        # Bybit trade format
                        ts = float(obj.get("T", obj.get("timestamp", 0)))
                        side = obj.get("S", obj.get("side", "Buy"))
                        qty = float(obj.get("v", obj.get("size", obj.get("qty", 0))))
                        price = float(obj.get("p", obj.get("price", 0)))
                        if qty > 0 and price > 0:
                            trades.append({
                                "timestamp": ts / 1000 if ts > 1e12 else ts,
                                "side": side,
                                "size_usdt": qty * price,
                            })
                    except (json.JSONDecodeError, ValueError, KeyError):
                        continue
                else:
                    # CSV 行
                    parts = line.split(",")
                    if len(parts) >= 5:
                        try:
                            ts = float(parts[0])
                            side = parts[2].strip().strip('"')
                            qty = float(parts[3])
                            price = float(parts[4])
                            trades.append({
                                "timestamp": ts / 1000 if ts > 1e12 else ts,
                                "side": side,
                                "size_usdt": qty * price,
                            })
                        except (ValueError, IndexError):
                            continue
    except Exception as e:
        print(f"  [WARN] 无法读取 {filepath}: {e}")

    return sorted(trades, key=lambda x: x["timestamp"])


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_thresholds = {}

    for symbol in args.symbols:
        print(f"\n{'='*60}")
        print(f"  分析 {symbol}")
        print(f"{'='*60}")

        # 加载数据
        trades = []
        if args.input:
            print(f"  加载文件: {args.input}")
            if args.input.endswith(".data") or args.input.endswith(".data.gz"):
                trades = load_bybit_data_file(args.input)
            else:
                trades = load_csv_trades(args.input, target_symbol=symbol)
        else:
            files = find_trade_files(args.input_dir, symbol)
            if not files:
                print(f"  [WARN] 未找到 {symbol} 的成交文件，跳过")
                continue
            print(f"  找到 {len(files)} 个文件:")
            for fp in files[:10]:
                print(f"    {fp}")
            if len(files) > 10:
                print(f"    ... 还有 {len(files)-10} 个")

            for fp in files:
                if fp.endswith(".data") or fp.endswith(".data.gz"):
                    trades.extend(load_bybit_data_file(fp))
                else:
                    trades.extend(load_csv_trades(fp, target_symbol=symbol))

            trades.sort(key=lambda x: x["timestamp"])

        if not trades:
            print(f"  [WARN] {symbol} 无有效成交数据")
            continue

        print(f"  总成交笔数: {len(trades):,}")
        if trades:
            from datetime import datetime
            ts_min = trades[0]["timestamp"]
            ts_max = trades[-1]["timestamp"]
            print(f"  时间范围: {datetime.fromtimestamp(ts_min)} ~ {datetime.fromtimestamp(ts_max)}")

        # 计算 VPIN 序列
        print(f"  计算 VPIN (桶={args.bucket_volume}, 窗口={args.lookback_buckets})...")
        results = CryptoVPINFilter.compute_from_trades(
            trades,
            bucket_volume=args.bucket_volume,
            lookback_buckets=args.lookback_buckets,
        )

        if not results:
            print(f"  [WARN] 成交量不足以生成任何桶")
            continue

        vpin_values = [r["vpin"] for r in results if r["vpin"] > 0]
        print(f"  总桶数: {len(results):,}")
        print(f"  有效 VPIN 值: {len(vpin_values):,}")

        # 统计
        stats = CryptoVPINFilter.compute_statistics(vpin_values)

        print(f"\n  ┌─────────────────────────────────────┐")
        print(f"  │  {symbol} VPIN 统计                  │")
        print(f"  ├─────────────────────────────────────┤")
        print(f"  │  样本数:    {stats['count']:>10,}            │")
        print(f"  │  均值:      {stats['mean']:>10.6f}            │")
        print(f"  │  标准差:    {stats['std']:>10.6f}            │")
        print(f"  │  中位数:    {stats['median']:>10.6f}            │")
        print(f"  │  75th pct:  {stats['p75']:>10.6f}            │")
        print(f"  │  85th pct:  {stats['p85']:>10.6f}  ← 推荐相对阈值│")
        print(f"  │  90th pct:  {stats['p90']:>10.6f}            │")
        print(f"  │  95th pct:  {stats['p95']:>10.6f}            │")
        print(f"  │  最小值:    {stats['min']:>10.6f}            │")
        print(f"  │  最大值:    {stats['max']:>10.6f}            │")
        print(f"  └─────────────────────────────────────┘")

        # 推荐阈值
        recommended_abs = round(stats["mean"] + 1.5 * stats["std"], 4)
        recommended_rel = stats["p85"]
        print(f"\n  推荐绝对阈值: {recommended_abs} (mean + 1.5*std)")
        print(f"  推荐相对阈值: {recommended_rel} (85th percentile)")

        all_thresholds[symbol] = {
            "stats": stats,
            "recommended_absolute": recommended_abs,
            "recommended_relative": recommended_rel,
            "bucket_volume": args.bucket_volume,
            "lookback_buckets": args.lookback_buckets,
        }

        # 保存 VPIN 时间序列 CSV
        csv_path = out / f"vpin_timeseries_{symbol}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "vpin", "buy_fraction", "bucket_index"])
            writer.writeheader()
            writer.writerows(results)
        print(f"  已保存 VPIN 序列: {csv_path}")

    # 保存汇总 JSON
    if all_thresholds:
        json_path = out / "vpin_thresholds.json"
        with open(json_path, "w") as f:
            json.dump(all_thresholds, f, indent=2)
        print(f"\n{'='*60}")
        print(f"  汇总阈值已保存: {json_path}")
        print(f"{'='*60}")

        # 生成配置建议
        print("\n  建议的 collect_trades_vpin.py 启动参数:")
        # 取所有币种推荐阈值的最小值（保守）
        min_abs = min(t["recommended_absolute"] for t in all_thresholds.values())
        print(f"    python scripts/collect_trades_vpin.py \\")
        print(f"        --bucket-volume {args.bucket_volume} \\")
        print(f"        --lookback-buckets {args.lookback_buckets} \\")
        print(f"        --threshold {min_abs}")


if __name__ == "__main__":
    main()
