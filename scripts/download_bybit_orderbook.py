#!/usr/bin/env python3
"""
Bybit L2 Orderbook 数据下载脚本（步骤 6.1）

封装 KatanaQuant/freebies 的 bybit_ob_scrape.py，批量下载 L2 深度快照。

前置条件：
    git clone https://github.com/KatanaQuant/freebies.git
    cd freebies/ob_scraper
    pip install playwright beautifulsoup4 tqdm
    playwright install chromium

用法：
    python scripts/download_bybit_orderbook.py --symbols BTCUSDT ETHUSDT XRPUSDT --start 2025-02-01 --end 2026-02-06

数据源：Bybit 官方 https://public.bybit.com/
数据类型：orderbook snapshots（每 10ms 一条，500 档 bids/asks）
"""

import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FREEBIES_DIR = PROJECT_ROOT / "freebies"
OB_SCRAPER_DIR = FREEBIES_DIR / "ob_scraper"
DATA_DIR = PROJECT_ROOT / "data" / "orderbook"


def check_prerequisites():
    """检查前置条件是否满足。"""
    if not OB_SCRAPER_DIR.exists():
        print(f"错误: 未找到 freebies 仓库: {FREEBIES_DIR}")
        print("请先运行:")
        print(f"  cd {PROJECT_ROOT}")
        print("  git clone https://github.com/KatanaQuant/freebies.git")
        print("  cd freebies/ob_scraper")
        print("  pip install playwright beautifulsoup4 tqdm")
        print("  playwright install chromium")
        return False

    scrape_script = OB_SCRAPER_DIR / "bybit_ob_scrape.py"
    if not scrape_script.exists():
        print(f"错误: 未找到下载脚本: {scrape_script}")
        return False

    return True


def download_symbol(symbol: str, start_date: str, end_date: str):
    """下载单个交易对的 orderbook 数据。"""
    scrape_script = OB_SCRAPER_DIR / "bybit_ob_scrape.py"
    cmd = [
        sys.executable, str(scrape_script),
        symbol,
        "--start-date", start_date,
        "--end-date", end_date,
    ]
    print(f"\n下载 {symbol}: {start_date} ~ {end_date}")
    print(f"  命令: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, cwd=str(OB_SCRAPER_DIR),
            capture_output=False, text=True,
        )
        if result.returncode == 0:
            print(f"  {symbol}: 下载完成")
        else:
            print(f"  {symbol}: 下载失败 (exit code {result.returncode})")
    except Exception as e:
        print(f"  {symbol}: 下载异常: {e}")


def main():
    parser = argparse.ArgumentParser(description="下载 Bybit L2 Orderbook 数据")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT", "XRPUSDT"],
                        help="交易对列表")
    parser.add_argument("--start", type=str, default="2025-02-01",
                        help="起始日期 (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=datetime.now().strftime("%Y-%m-%d"),
                        help="结束日期 (YYYY-MM-DD)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not check_prerequisites():
        print("\n提示: 如果不想用 KatanaQuant 脚本，也可手动从以下地址下载:")
        print("  https://public.bybit.com/")
        print("  https://www.bybit.com/derivatives/en/history-data")
        sys.exit(1)

    for symbol in args.symbols:
        download_symbol(symbol, args.start, args.end)

    print(f"\n所有下载完成。数据位于: {OB_SCRAPER_DIR}")
    print(f"请将下载的数据移动到: {DATA_DIR}")


if __name__ == "__main__":
    main()
