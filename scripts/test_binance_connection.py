#!/usr/bin/env python3
"""
测试 Binance API 连接（用于排查「仅本地(网络失败)」）。
数据更新依赖 ccxt -> Binance，若连不上则预测器退回到仅本地数据。
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("=" * 60)
    print("  测试 Binance 连接（预测器拉取 K 线用）")
    print("=" * 60)

    try:
        import ccxt
        print("  ✓ ccxt 已安装")
    except ImportError as e:
        print(f"  ✗ ccxt 未安装: {e}")
        print("    解决: pip install ccxt  或  venv/bin/pip install ccxt")
        return 1

    try:
        ex = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})
        rows = ex.fetch_ohlcv("BTC/USDT", "15m", limit=5)
        if not rows:
            print("  ✗ 拉取 K 线返回空")
            return 1
        last = rows[-1]
        ts_ms, o, h, l, c, v = last
        from datetime import datetime, timezone
        ts_str = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        print(f"  ✓ 拉取成功: 最新一根 15m K 线 {ts_str}, 收盘 ${c:.2f}")
        print("  → 预测器应能「本地+实时更新」，若仍显示仅本地请查看预测器日志中的具体报错。")
    except Exception as e:
        print(f"  ✗ 连接失败: {type(e).__name__}: {e}")
        print("")
        print("  可能原因：")
        print("    - 无法访问 api.binance.com（防火墙/代理/地区限制）")
        print("    - DNS 解析失败、超时、SSL 证书问题")
        print("  解决：")
        print("    - 检查网络、代理；或定期手动更新 data/raw 后只用本地")
        return 1

    print("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
