#!/usr/bin/env python3
"""
Orderbook 流水线 v2：按周批量 下载 → 聚合 → 删除 → 循环

每次下 7 天 (~1.4GB) → 聚合成 15m parquet (~14KB) → 删除 zip → 下一周。
峰值磁盘 ~1.5GB，最终输出 ~1MB。

用法:
    cd /Users/mac/polyfun
    python scripts/stream_ob_pipeline.py \
        --symbols BTCUSDT ETHUSDT XRPUSDT SOLUSDT \
        --start-date 2025-08-11 \
        --end-date 2026-02-06

    # 断点续跑：已聚合到 parquet 的天会自动跳过
    # Ctrl+C 安全中断，再跑同一条命令即可继续
"""

import argparse
import json
import os
import signal
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.python.features.orderbook_features import (
    extract_orderbook_features,
    aggregate_ob_features_to_bar,
    get_ob_feature_names,
)

# ─── Bybit API（和原始 scraper 相同的 headers）────────────

BYBIT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:143.0) Gecko/20100101 Firefox/143.0",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Referer": "https://www.bybit.com/derivatives/en/history-data",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "Connection": "keep-alive",
}

terminate = threading.Event()


def handle_exit(signum, frame):
    print("\n⏹ 收到中断信号，安全退出中...")
    terminate.set()


signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)


# ─── 日期工具 ──────────────────────────────────────────────

def week_ranges(start_str: str, end_str: str, step: int = 7):
    """按 step 天生成 (week_start, week_end) 序列"""
    s = datetime.strptime(start_str, "%Y-%m-%d")
    e = datetime.strptime(end_str, "%Y-%m-%d")
    while s <= e:
        we = min(s + timedelta(days=step - 1), e)
        yield s, we
        s = we + timedelta(days=1)


def day_strs_in_range(ws, we):
    """返回一周内每天的日期字符串列表"""
    days = []
    d = ws
    while d <= we:
        days.append(d.strftime("%Y-%m-%d"))
        d += timedelta(days=1)
    return days


# ─── 下载 ──────────────────────────────────────────────────

def download_file(url: str, save_path: str, max_retries: int = 5) -> bool:
    """下载单个文件（带重试）"""
    for attempt in range(max_retries):
        if terminate.is_set():
            return False
        try:
            with requests.get(url, headers=BYBIT_HEADERS, stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if terminate.is_set():
                            return False
                        if chunk:
                            f.write(chunk)
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                print(f"下载失败: {os.path.basename(save_path)} ({e})")
    return False


def download_week(symbol: str, ws: datetime, we: datetime, data_dir: Path) -> list[Path]:
    """下载一周的 OB 数据，返回成功的 zip 路径列表"""
    start_str = ws.strftime("%Y-%m-%d")
    end_str = we.strftime("%Y-%m-%d")

    url = (
        "https://www.bybit.com/x-api/quote/public/support/download/list-files"
        f"?bizType=contract&productId=orderbook&symbols={symbol}&interval=daily"
        f"&periods=&startDay={start_str}&endDay={end_str}"
    )

    file_list = []
    for api_attempt in range(5):
        try:
            r = requests.get(url, headers=BYBIT_HEADERS, timeout=60)
            r.raise_for_status()
            file_list = r.json().get("result", {}).get("list", [])
            break
        except Exception as e:
            if api_attempt < 4:
                wait = 5 * (api_attempt + 1)
                print(f"    API 重试 ({api_attempt+1}/5, 等{wait}s): {type(e).__name__}")
                time.sleep(wait)
            else:
                print(f"    API 错误（5次重试均失败）: {e}")
                return []

    if not file_list:
        return []

    sym_dir = data_dir / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {}
        for fi in file_list:
            save_path = sym_dir / fi["filename"]
            # 已存在且有效的跳过下载
            if save_path.exists() and is_valid_zip(save_path):
                downloaded.append(save_path)
                continue
            if save_path.exists():
                save_path.unlink()  # 删除损坏的
            futures[executor.submit(download_file, fi["url"], str(save_path))] = save_path

        for future in as_completed(futures):
            if terminate.is_set():
                break
            path = futures[future]
            try:
                if future.result() and path.exists() and is_valid_zip(path):
                    downloaded.append(path)
            except Exception:
                pass

    return downloaded


# ─── 聚合 ──────────────────────────────────────────────────

def is_valid_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path, "r") as zf:
            return zf.testzip() is None
    except Exception:
        return False


def aggregate_one_zip(zip_path: Path, symbol: str, max_levels: int = 20) -> pd.DataFrame:
    """解压 zip → 逐行解析 → 聚合到 15m bar"""
    snapshots_by_bar = {}

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            data_files = [n for n in zf.namelist() if n.endswith(".data")]
            if not data_files:
                return pd.DataFrame()
            with zf.open(data_files[0]) as f:
                for raw_line in f:
                    try:
                        line = raw_line.decode("utf-8", errors="ignore").strip()
                        if not line:
                            continue
                        data = json.loads(line)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue

                    ts = data.get("timestamp") or data.get("ts") or data.get("T")
                    if ts is None:
                        continue
                    if isinstance(ts, str):
                        try:
                            ts = pd.Timestamp(ts).timestamp()
                        except Exception:
                            continue
                    ts = float(ts)
                    if ts > 1e12:
                        ts = ts / 1000
                    bar_ts = int(ts // 900) * 900

                    bids = data.get("bids") or data.get("b") or []
                    asks = data.get("asks") or data.get("a") or []
                    last_price = data.get("last_price") or data.get("lp")

                    features = extract_orderbook_features(
                        bids=bids, asks=asks,
                        last_price=float(last_price) if last_price else None,
                        max_levels=max_levels,
                    )
                    if bar_ts not in snapshots_by_bar:
                        snapshots_by_bar[bar_ts] = []
                    snapshots_by_bar[bar_ts].append(features)
    except Exception as e:
        print(f"    解压/解析错误: {e}")
        return pd.DataFrame()

    rows = []
    for bar_ts in sorted(snapshots_by_bar.keys()):
        agg = aggregate_ob_features_to_bar(snapshots_by_bar[bar_ts])
        agg["timestamp"] = int(bar_ts * 1000)
        agg["n_snapshots"] = len(snapshots_by_bar[bar_ts])
        agg["symbol"] = symbol
        rows.append(agg)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ─── 主流水线 ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OB 流水线 v2：按周批量")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT"])
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--keep-raw", action="store_true", help="保留原始 zip")
    parser.add_argument("--no-ablation", action="store_true")
    args = parser.parse_args()

    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = PROJECT_ROOT / "freebies" / "ob_scraper" / "data"

    weeks = list(week_ranges(args.start_date, args.end_date))
    total_weeks = len(weeks) * len(args.symbols)
    total_days = sum((we - ws).days + 1 for ws, we in weeks) * len(args.symbols)

    print(f"{'='*60}")
    print(f"  OB 流水线 v2：按周批量 下载→聚合→删除")
    print(f"{'='*60}")
    print(f"  币种:     {', '.join(args.symbols)}")
    print(f"  日期:     {args.start_date} → {args.end_date}")
    print(f"  总计:     {total_days} 天 = {total_weeks} 周×币")
    print(f"  峰值磁盘: ~1.5GB（每次只保留 1 周 zip）")
    print(f"{'='*60}\n")

    week_idx = 0
    stats = {"downloaded": 0, "aggregated": 0, "skipped": 0, "failed": 0}

    for sym in args.symbols:
        if terminate.is_set():
            break

        parquet_path = out_dir / f"ob_15m_{sym}.parquet"

        # 加载已有聚合数据，获取已有日期集合
        existing_dates = set()
        existing_df = None
        if parquet_path.exists():
            existing_df = pd.read_parquet(parquet_path)
            if "timestamp" in existing_df.columns:
                for ts in existing_df["timestamp"].unique():
                    dt = pd.Timestamp(ts, unit="ms")
                    existing_dates.add(dt.strftime("%Y-%m-%d"))
            print(f"📂 {sym}: 已有 {len(existing_df)} bars（{len(existing_dates)} 天），增量追加\n")

        new_dfs = []

        for ws, we in weeks:
            if terminate.is_set():
                break
            week_idx += 1
            w_start = ws.strftime("%Y-%m-%d")
            w_end = we.strftime("%Y-%m-%d")
            n_days = (we - ws).days + 1

            # 检查这周是否全部已聚合
            week_days = day_strs_in_range(ws, we)
            needed = [d for d in week_days if d not in existing_dates]
            if not needed:
                stats["skipped"] += n_days
                continue

            print(f"  [{week_idx}/{total_weeks}] {sym} {w_start}~{w_end} ({n_days}天, 需下{len(needed)}天): ", end="", flush=True)

            # 1. 下载这周
            zip_files = download_week(sym, ws, we, data_dir)
            if terminate.is_set():
                break

            if not zip_files:
                print("无数据")
                stats["failed"] += n_days
                continue

            stats["downloaded"] += len(zip_files)
            print(f"下载{len(zip_files)}个 → ", end="", flush=True)

            # 2. 聚合每个 zip
            week_bars = 0
            for zp in zip_files:
                # 跳过已聚合的天
                date_in_name = zp.name[:10]  # "2026-01-07"
                if date_in_name in existing_dates:
                    if not args.keep_raw:
                        zp.unlink(missing_ok=True)
                    continue

                df = aggregate_one_zip(zp, sym)
                if not df.empty:
                    new_dfs.append(df)
                    week_bars += len(df)
                    stats["aggregated"] += 1

                # 3. 删除原始 zip
                if not args.keep_raw:
                    zp.unlink(missing_ok=True)

            print(f"聚合{week_bars}bars ✅", flush=True)

            # 每周保存一次（防止中断丢失）
            if new_dfs:
                all_parts = [existing_df] if existing_df is not None else []
                all_parts.extend(new_dfs)
                merged = pd.concat(all_parts, ignore_index=True)
                merged = merged.drop_duplicates(subset=["timestamp", "symbol"]).sort_values("timestamp").reset_index(drop=True)
                merged.to_parquet(parquet_path, index=False)
                # 更新 existing 以便后续跳过
                existing_df = merged
                for ts in merged["timestamp"].unique():
                    existing_dates.add(pd.Timestamp(ts, unit="ms").strftime("%Y-%m-%d"))
                new_dfs = []  # 已合入 existing_df

        # 最终保存
        if new_dfs:
            all_parts = [existing_df] if existing_df is not None else []
            all_parts.extend(new_dfs)
            merged = pd.concat(all_parts, ignore_index=True)
            merged = merged.drop_duplicates(subset=["timestamp", "symbol"]).sort_values("timestamp").reset_index(drop=True)
            merged.to_parquet(parquet_path, index=False)

        # 清理空目录
        sym_dir = data_dir / sym
        if not args.keep_raw and sym_dir.exists():
            remaining = list(sym_dir.iterdir())
            if not remaining:
                sym_dir.rmdir()

        # 报告
        if parquet_path.exists():
            final_df = pd.read_parquet(parquet_path)
            kb = parquet_path.stat().st_size / 1024
            print(f"\n  💾 {sym}: {len(final_df)} bars → {parquet_path.name} ({kb:.0f}KB)\n")

    # ─── 汇总 ───
    print(f"\n{'='*60}")
    print(f"  流水线完成{'（被中断）' if terminate.is_set() else ''}!")
    print(f"  下载: {stats['downloaded']} 文件")
    print(f"  聚合: {stats['aggregated']} 天")
    print(f"  跳过: {stats['skipped']} 天（已有）")
    print(f"  失败: {stats['failed']} 天")
    if terminate.is_set():
        print(f"  💡 再跑同一条命令即可断点续跑")
    print(f"{'='*60}")

    # 自动 ablation
    if not args.no_ablation and not terminate.is_set():
        sym_map = {"BTCUSDT": "BTC_USDT", "ETHUSDT": "ETH_USDT", "XRPUSDT": "XRP_USDT", "SOLUSDT": "SOL_USDT"}
        ablation_script = PROJECT_ROOT / "experiments" / "orderbook_ablation" / "run_ablation.py"
        if ablation_script.exists():
            print(f"\n{'='*60}")
            print(f"  自动运行 Ablation Study...")
            print(f"{'='*60}\n")
            import subprocess
            for sym in args.symbols:
                pq = out_dir / f"ob_15m_{sym}.parquet"
                asset = sym_map.get(sym, sym.replace("USDT", "_USDT"))
                if pq.exists():
                    print(f"  ▶ {asset} ablation...")
                    subprocess.run([sys.executable, str(ablation_script), "--ob-features", str(pq), "--asset", asset], cwd=str(PROJECT_ROOT))
                    print()


if __name__ == "__main__":
    main()
