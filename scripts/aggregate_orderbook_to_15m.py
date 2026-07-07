#!/usr/bin/env python3
"""
Orderbook 特征聚合到 15m K 线（步骤 6.2）

将 10ms 级别的 L2 快照聚合到 15 分钟 bar，
每个特征计算 5 个统计量（mean/std/min/max/last），
输出 10 × 5 = 50 个新特征列。

用法：
    python scripts/aggregate_orderbook_to_15m.py \\
        --input data/orderbook/ \\
        --output data/processed/features_with_orderbook.parquet

输入：data/orderbook/<symbol>_<date>.parquet（或 .data 文件）
输出：含 OB 特征的 15m parquet
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.python.features.orderbook_features import (
    extract_orderbook_features,
    aggregate_ob_features_to_bar,
    get_ob_feature_names,
)


def parse_ob_snapshot(line: str) -> dict:
    """
    解析单行 Bybit orderbook snapshot。
    Bybit 格式：每行一个 JSON，含 bids/asks 数组。
    """
    try:
        data = json.loads(line.strip())
        return data
    except (json.JSONDecodeError, TypeError):
        return None


def process_ob_file(
    ob_path: Path,
    max_levels: int = 20,
) -> pd.DataFrame:
    """
    处理单个 orderbook 文件，提取特征并按 15m 聚合。

    返回 DataFrame，index 为 15m 时间戳，columns 为 50 个 OB 特征。
    """
    snapshots_by_bar = {}  # {bar_ts: [features_dict, ...]}

    with open(ob_path, "r") as f:
        for line in f:
            data = parse_ob_snapshot(line)
            if data is None:
                continue

            # 提取时间戳（毫秒或秒）
            ts = data.get("timestamp") or data.get("ts") or data.get("T")
            if ts is None:
                continue
            if isinstance(ts, str):
                try:
                    ts = pd.Timestamp(ts).timestamp()
                except Exception:
                    continue
            if ts > 1e12:
                ts = ts / 1000  # 毫秒转秒

            # 对齐到 15m bar 的起始时间戳
            bar_ts = int(ts // 900) * 900

            bids = data.get("bids") or data.get("b") or []
            asks = data.get("asks") or data.get("a") or []
            last_price = data.get("last_price") or data.get("lp")

            # 提取特征
            features = extract_orderbook_features(
                bids=bids, asks=asks,
                last_price=float(last_price) if last_price else None,
                max_levels=max_levels,
            )
            if bar_ts not in snapshots_by_bar:
                snapshots_by_bar[bar_ts] = []
            snapshots_by_bar[bar_ts].append(features)

    # 聚合到 bar
    rows = []
    for bar_ts in sorted(snapshots_by_bar.keys()):
        agg = aggregate_ob_features_to_bar(snapshots_by_bar[bar_ts])
        agg["timestamp"] = int(bar_ts * 1000)  # 转回毫秒
        agg["n_snapshots"] = len(snapshots_by_bar[bar_ts])
        rows.append(agg)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="聚合 Orderbook 特征到 15m")
    parser.add_argument("--input", type=str, default="data/orderbook",
                        help="orderbook 数据目录")
    parser.add_argument("--output", type=str, default="data/processed/features_with_orderbook.parquet",
                        help="输出 parquet 路径")
    parser.add_argument("--max-levels", type=int, default=20,
                        help="使用的最大档位数")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    # 查找所有 OB 数据文件（支持子目录 + zip）
    import zipfile, tempfile, shutil

    ob_files = (
        sorted(input_dir.glob("*.data"))
        + sorted(input_dir.glob("*.parquet"))
        + sorted(input_dir.glob("**/*.data"))
        + sorted(input_dir.glob("**/*.data.zip"))
    )
    # 去重
    ob_files = sorted(set(ob_files))
    if not ob_files:
        print(f"未找到 orderbook 数据文件: {input_dir}")
        print("请先运行: python scripts/download_bybit_orderbook.py")
        sys.exit(1)

    print(f"找到 {len(ob_files)} 个 OB 文件")
    all_dfs = []
    for f in ob_files:
        print(f"  处理: {f.name} ...", end=" ", flush=True)
        try:
            if f.suffix == ".zip":
                # 解压 zip 到临时文件再处理
                with zipfile.ZipFile(f, "r") as zf:
                    names = zf.namelist()
                    data_names = [n for n in names if n.endswith(".data")]
                    if not data_names:
                        print("zip 内无 .data 文件，跳过")
                        continue
                    with tempfile.TemporaryDirectory() as tmpdir:
                        zf.extract(data_names[0], tmpdir)
                        tmp_path = Path(tmpdir) / data_names[0]
                        df = process_ob_file(tmp_path, max_levels=args.max_levels)
            else:
                df = process_ob_file(f, max_levels=args.max_levels)
            if not df.empty:
                # 从文件名推断 symbol
                fname = f.name.upper()
                for sym in ("BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT"):
                    if sym in fname:
                        df["symbol"] = sym
                        break
                else:
                    df["symbol"] = f.parent.name.upper()
                print(f"{len(df)} bars")
                all_dfs.append(df)
            else:
                print("空")
        except Exception as e:
            print(f"错误: {e}")

    if not all_dfs:
        print("没有有效数据")
        sys.exit(1)

    result = pd.concat(all_dfs, ignore_index=True)
    result = result.sort_values("timestamp").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)

    feature_names = get_ob_feature_names()
    print(f"\n已写出: {output_path}")
    print(f"  {len(result)} bars, {len(feature_names)} OB 特征列")
    print(f"  特征示例: {feature_names[:5]} ...")


if __name__ == "__main__":
    main()
