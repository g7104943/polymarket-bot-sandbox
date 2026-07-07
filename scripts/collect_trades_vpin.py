#!/usr/bin/env python3
"""
VPIN 实时守护进程：订阅 Bybit V5 publicTrade，
计算实时 VPIN 并写入 vpin_status.json 供 TypeScript 下单逻辑读取。

用法：
  python scripts/collect_trades_vpin.py                            # 默认 4 币种
  python scripts/collect_trades_vpin.py --symbols BTCUSDT ETHUSDT  # 仅 BTC+ETH
  python scripts/collect_trades_vpin.py --bucket-volume 3000       # 自定义桶大小
  python scripts/collect_trades_vpin.py --output polymarket/vpin_status.json

生成文件：
  vpin_status.json  — 每秒更新，TypeScript 端直接 fs.readFileSync 解析
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

# 确保能 import 项目模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.python.features.vpin_filter import CryptoVPINFilter

# ───────────────────────────────────────────────────
# Bybit WebSocket 映射
# ───────────────────────────────────────────────────

# Polymarket 预测用的 crypto → Bybit 永续合约 symbol
POLYMARKET_TO_BYBIT = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
}


def parse_args():
    p = argparse.ArgumentParser(description="VPIN 实时守护进程")
    p.add_argument(
        "--symbols", nargs="+",
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
        help="监控的 Bybit 交易对",
    )
    p.add_argument("--bucket-volume", type=float, default=5000.0,
                   help="每桶 USDT 成交额 (默认 5000)")
    p.add_argument("--lookback-buckets", type=int, default=80,
                   help="滚动窗口桶数 (默认 80)")
    p.add_argument("--threshold", type=float, default=0.62,
                   help="VPIN 绝对阈值 (默认 0.62)")
    p.add_argument("--cooldown", type=float, default=20.0,
                   help="持久高冷却分钟数 (默认 20)")
    p.add_argument("--output", type=str, default="polymarket/vpin_status.json",
                   help="输出 JSON 文件路径")
    p.add_argument("--dump-interval", type=float, default=1.0,
                   help="状态写入间隔秒数 (默认 1)")
    p.add_argument("--verbose", action="store_true", default=False,
                   help="打印详细日志")
    return p.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)

    print("=" * 60)
    print("  VPIN 实时守护进程")
    print("=" * 60)
    print(f"  币种:      {args.symbols}")
    print(f"  桶大小:    {args.bucket_volume} USDT")
    print(f"  窗口:      {args.lookback_buckets} 桶")
    print(f"  绝对阈值:  {args.threshold}")
    print(f"  冷却:      {args.cooldown} 分钟")
    print(f"  输出:      {output_path}")
    print("=" * 60)

    # 初始化 VPIN 过滤器
    vpin = CryptoVPINFilter(
        symbols=args.symbols,
        bucket_volume=args.bucket_volume,
        lookback_buckets=args.lookback_buckets,
        threshold_absolute=args.threshold,
        cooldown_minutes=args.cooldown,
        verbose=args.verbose,
    )

    # 尝试导入 pybit
    try:
        from pybit.unified_trading import WebSocket
    except ImportError:
        print("\n[ERROR] 需要安装 pybit:")
        print("  pip install pybit")
        sys.exit(1)

    # 优雅退出
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        print("\n[INFO] 收到退出信号，正在关闭...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # 成交回调
    trade_count = 0
    last_dump_time = 0.0

    def on_public_trade(msg):
        nonlocal trade_count, last_dump_time
        """
        Bybit publicTrade 消息格式:
        {
          "topic": "publicTrade.BTCUSDT",
          "type": "snapshot",
          "ts": 1672304486868,
          "data": [
            {
              "T": 1672304486865,   # timestamp ms
              "s": "BTCUSDT",       # symbol
              "S": "Buy",           # side
              "v": "0.001",         # quantity (coins)
              "p": "16578.50",      # price
              "L": "PlusTick",      # tick direction
              "i": "...",           # trade id
              "BT": false           # is block trade
            }, ...
          ]
        }
        """
        data = msg.get("data", [])
        for trade in data:
            symbol = trade.get("s", "")
            side = trade.get("S", "Buy")
            qty = float(trade.get("v", 0))
            price = float(trade.get("p", 0))
            size_usdt = qty * price

            vpin.on_trade(symbol, side, size_usdt)
            trade_count += 1

        # 定期写入状态
        now = time.time()
        if now - last_dump_time >= args.dump_interval:
            vpin.dump_status_json(output_path)
            last_dump_time = now

    # 连接 WebSocket
    print("\n[INFO] 正在连接 Bybit V5 Public WebSocket (linear)...")
    ws = WebSocket(testnet=False, channel_type="linear")

    for sym in args.symbols:
        print(f"[INFO] 订阅 publicTrade.{sym}")
        ws.trade_stream(symbol=sym, callback=on_public_trade)

    print("[INFO] WebSocket 已连接，等待成交数据...\n")

    # 主循环：定期打印状态
    status_interval = 30  # 每 30 秒打印一次摘要
    last_status_time = time.time()

    try:
        while running:
            time.sleep(1)

            # 确保定期写入（即使没有新 trade）
            now = time.time()
            if now - last_dump_time >= args.dump_interval * 2:
                vpin.dump_status_json(output_path)
                last_dump_time = now

            # 定期打印
            if now - last_status_time >= status_interval:
                last_status_time = now
                print(f"\n[STATUS] {time.strftime('%H:%M:%S')} | 总成交: {trade_count}")
                for sym in args.symbols:
                    status = vpin.get_status(sym)
                    is_safe, reason = vpin.should_trade(sym)
                    marker = "✅" if is_safe else "❌"
                    print(
                        f"  {marker} {sym}: VPIN={status['vpin']:.4f} | "
                        f"桶={status['total_buckets']} | "
                        f"当前桶填充={status['current_bucket_fill_pct']:.0f}% | "
                        f"{'可交易' if is_safe else reason}"
                    )
    except KeyboardInterrupt:
        pass
    finally:
        # 最终写入一次
        vpin.dump_status_json(output_path)
        print(f"\n[INFO] 已退出。最终状态已写入 {output_path}")


if __name__ == "__main__":
    main()
