#!/usr/bin/env python3
"""
API 连通性预检脚本 — 启动交易前验证所有免费数据源可用。

用法:
    python scripts/check_api_connectivity.py

退出码:
    0 = 全部通过
    1 = 有失败
"""
import sys
import time
import requests

TIMEOUT = 15

def test_api(name, url, params=None, headers=None, check_fn=None):
    """Test a single API endpoint."""
    try:
        r = requests.get(url, params=params, headers=headers or {}, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if check_fn and not check_fn(data):
            return False, "响应格式不符预期"
        return True, f"HTTP {r.status_code}"
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 60)
    print("  API 连通性预检（免费数据源）")
    print("=" * 60)
    print()
    
    results = []
    
    # 1. Binance REST - OHLCV 1m
    ok, msg = test_api(
        "Binance OHLCV 1m",
        "https://fapi.binance.com/fapi/v1/klines",
        params={"symbol": "BTCUSDT", "interval": "1m", "limit": 1},
        check_fn=lambda d: isinstance(d, list) and len(d) > 0
    )
    results.append(("Binance OHLCV 1m", ok, msg))
    
    # 2. Binance REST - Open Interest
    ok, msg = test_api(
        "Binance Open Interest",
        "https://fapi.binance.com/fapi/v1/openInterest",
        params={"symbol": "BTCUSDT"},
        check_fn=lambda d: "openInterest" in d
    )
    results.append(("Binance Open Interest", ok, msg))
    
    # 3. Binance REST - Long/Short Ratio
    ok, msg = test_api(
        "Binance Long/Short Ratio",
        "https://fapi.binance.com/futures/data/topLongShortAccountRatio",
        params={"symbol": "BTCUSDT", "period": "15m", "limit": 1},
        check_fn=lambda d: isinstance(d, list) and len(d) > 0 and "longShortRatio" in d[0]
    )
    results.append(("Binance Long/Short Ratio", ok, msg))
    
    # 4. Binance WebSocket - Funding Rate (test REST endpoint instead, WS needs persistent connection)
    ok, msg = test_api(
        "Binance Funding Rate",
        "https://fapi.binance.com/fapi/v1/premiumIndex",
        params={"symbol": "BTCUSDT"},
        check_fn=lambda d: "lastFundingRate" in d
    )
    results.append(("Binance Funding Rate", ok, msg))
    
    # 5. Bybit REST - Order Book
    ok, msg = test_api(
        "Bybit Order Book",
        "https://api.bybit.com/v5/market/orderbook",
        params={"category": "linear", "symbol": "BTCUSDT", "limit": 5},
        check_fn=lambda d: "result" in d and "b" in d["result"] and "a" in d["result"]
    )
    results.append(("Bybit Order Book", ok, msg))
    
    # 6. Polymarket Gamma API
    # Use a generic slug format - may 404 if no current market, that's ok for connectivity test
    ok, msg = test_api(
        "Polymarket Gamma API",
        "https://gamma-api.polymarket.com/events",
        params={"limit": 1},
        check_fn=lambda d: isinstance(d, list)
    )
    results.append(("Polymarket Gamma API", ok, msg))
    
    # 7. Polymarket CLOB API
    ok, msg = test_api(
        "Polymarket CLOB API",
        "https://clob.polymarket.com/sampling-markets",
        params={"next_cursor": "MA=="},
        check_fn=lambda d: isinstance(d, dict)
    )
    results.append(("Polymarket CLOB API", ok, msg))
    
    # 8. Alternative.me - FGI
    ok, msg = test_api(
        "Alternative.me FGI",
        "https://api.alternative.me/fng/",
        params={"limit": 1},
        check_fn=lambda d: "data" in d and len(d["data"]) > 0
    )
    results.append(("Alternative.me FGI", ok, msg))
    
    # 9. CryptoCompare - News
    ok, msg = test_api(
        "CryptoCompare News",
        "https://min-api.cryptocompare.com/data/v2/news/",
        params={"lang": "EN"},
        check_fn=lambda d: "Data" in d and isinstance(d["Data"], list)
    )
    results.append(("CryptoCompare News", ok, msg))
    
    # Print results
    passed = 0
    failed = 0
    for name, ok, msg in results:
        status = "✅ PASS" if ok else "❌ FAIL"
        print(f"  {status}  {name}: {msg}")
        if ok:
            passed += 1
        else:
            failed += 1
    
    print()
    print(f"  结果: {passed} 通过, {failed} 失败")
    print()
    
    if failed > 0:
        print("❌ 有 API 不可用，请修复后再启动交易")
        return 1
    
    print("✅ 所有 API 连通性检查通过")
    # Note: CFGI (cfgi.io) 为收费 API，不测试
    print("  注意: CFGI (cfgi.io) 为收费 API，未包含在本检查中")
    return 0

if __name__ == "__main__":
    sys.exit(main())
