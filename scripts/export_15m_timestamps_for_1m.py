#!/usr/bin/env python3
"""
从回测用的 15m parquet 导出每根 K 线的起始时间（Unix 秒），用于填写 polymarket_15m_up_token_ids.json，
保证 1m 数据与 15m 严格对齐、不错位。

用法（与回测同一时间窗口）:
  python scripts/export_15m_timestamps_for_1m.py --asset ETH_USDT --after-date 2025-01-03 --year-days 30
  或先算无泄露起始日再传:
  python scripts/export_15m_timestamps_for_1m.py --asset ETH_USDT --auto-after-date --year-days 30

输出: data/polymarket_15m_up_token_ids_{asset}_template.json（仅含 start_ts，token_id 留空待填）。
"""
import json
import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"


def main():
    ap = argparse.ArgumentParser(description="从 15m parquet 导出时间戳，生成 1m token 列表模板，保证与 15m 对齐")
    ap.add_argument("--asset", type=str, required=True, help="资产，如 ETH_USDT")
    ap.add_argument("--after-date", type=str, default=None, help="回测起始日 YYYY-MM-DD，与 backtest_gru_regime_report 一致")
    ap.add_argument("--auto-after-date", action="store_true", help="从 config+parquet 自动算无泄露起始日（与 --auto-after-date 一致）")
    ap.add_argument("--year-days", type=int, default=30, help="回测天数，默认 30")
    ap.add_argument("--data-src", type=str, default=None, help="15m 数据目录，默认 data/raw")
    args = ap.parse_args()

    data_src = Path(args.data_src) if args.data_src else DATA_RAW
    if not data_src.is_absolute():
        data_src = PROJECT_ROOT / data_src
    path_15m = data_src / f"{args.asset.lower()}_15m.parquet"
    if not path_15m.exists():
        print(f"❌ 未找到 {path_15m}")
        return 1

    df = pd.read_parquet(path_15m)
    if "timestamp" not in df.columns and "date" in df.columns:
        df["timestamp"] = pd.to_datetime(df["date"]).astype("int64") // 10**6
    df = df.sort_values("timestamp").reset_index(drop=True)

    after_date = args.after_date
    if args.auto_after_date:
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.backtest_gru_regime import get_no_leak_start_date
        after_date = get_no_leak_start_date(data_src, None, args.asset)
        print(f"  [--auto-after-date] 无泄露起始日: {after_date}")
    if not after_date:
        print("❌ 请指定 --after-date 或 --auto-after-date")
        return 1

    cutoff_ms = int(pd.Timestamp(after_date, tz="UTC").value // 10**6)
    end_date = (pd.Timestamp(after_date, tz="UTC") + pd.Timedelta(days=args.year_days)).strftime("%Y-%m-%d")
    end_ms = int(pd.Timestamp(end_date, tz="UTC").value // 10**6)
    df = df[(df["timestamp"] > cutoff_ms) & (df["timestamp"] <= end_ms)].copy()
    if df.empty:
        print("❌ 该时间窗口内无 15m 数据")
        return 1

    # 每根 15m 的起始时间（Unix 秒），与回测中 merged["timestamp"]//1000 一致
    start_ts_list = (df["timestamp"].values // 1000).astype(int).tolist()
    template = {args.asset: [{"start_ts": int(t), "token_id": ""} for t in start_ts_list]}

    out_path = PROJECT_ROOT / "data" / f"polymarket_15m_up_token_ids_{args.asset.lower()}_template.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2, ensure_ascii=False)
    print(f"✅ 已导出 {len(start_ts_list)} 个 15m 起始时间（与回测 {after_date} 起 {args.year_days} 天一致）→ {out_path}")
    print("   请将 token_id 填为 Polymarket 该 15m 周期 UP token 的 CLOB Token ID，另存为 polymarket_15m_up_token_ids.json 后运行 download_polymarket_1m.py")
    return 0


if __name__ == "__main__":
    exit(main())
