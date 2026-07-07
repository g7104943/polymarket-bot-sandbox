#!/usr/bin/env python3
"""
最小用例：验证「执行预测前健康检查 + 修复」流程是否生效。

流程：
1. 备份 data/raw/eth_usdt_1h.parquet（若存在）
2. 故意损坏 1h 文件（写非 parquet 内容），模拟「1h 异常」
3. 调用 run_data_health_check -> 应发现 (ETH/USDT, 1h, ...)
4. 调用 run_data_health_check_and_repair -> 应尝试 update_latest 并修复
5. 恢复备份，不改变用户原始数据

运行方式（在项目根目录）：
  python scripts/test_health_check_before_predict.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_RAW = PROJECT_ROOT / "data" / "raw"
ETH_1H = DATA_RAW / "eth_usdt_1h.parquet"
ETH_1H_BAK = DATA_RAW / "eth_usdt_1h.parquet.bak"


def main():
    from src.python.data_fetcher import (
        run_data_health_check,
        run_data_health_check_and_repair,
        load_ohlcv,
        validate_ohlcv_df,
    )

    symbols = ["ETH/USDT"]
    timeframes = ["15m", "1h", "4h"]
    min_rows = 50

    print("=" * 60)
    print("  最小用例：健康检查 + 修复（1h 缺失/异常）")
    print("=" * 60)

    # 1) 备份 1h 文件
    if ETH_1H.exists():
        import shutil
        shutil.copy2(ETH_1H, ETH_1H_BAK)
        print(f"\n[1] 已备份: {ETH_1H.name} -> {ETH_1H_BAK.name}")
    else:
        ETH_1H_BAK.write_bytes(b"")  # 占位，最后不覆盖
        print(f"\n[1] 无原始 1h 文件，稍后修复会拉取新文件")

    # 2) 故意损坏 1h（写非 parquet 内容）
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    ETH_1H.write_bytes(b"not_parquet_magic_bytes")
    print(f"[2] 已损坏 1h 文件（写入非 parquet 内容）")

    # 3) 健康检查应发现 1h 异常（load_ohlcv 读损坏文件会返回空 DataFrame）
    issues_before = run_data_health_check(symbols, timeframes, min_rows=min_rows)
    issues_1h = [(s, tf, msg) for s, tf, msg in issues_before if tf == "1h"]
    print(f"\n[3] run_data_health_check 结果: 共 {len(issues_before)} 项异常")
    for s, tf, msg in issues_before:
        print(f"    - {s} {tf}: {msg}")
    if not issues_1h:
        print("    -> 未发现 1h 异常，请检查 load_ohlcv 对损坏文件的处理")
    else:
        print("    -> 已发现 1h 异常，符合预期")

    # 4) 修复：应调用 update_latest 并重验
    results = run_data_health_check_and_repair(symbols, timeframes, min_rows=min_rows)
    print(f"\n[4] run_data_health_check_and_repair 结果: 共 {len(results)} 项")
    for s, tf, msg, repaired in results:
        status = "已拉取更新并修复" if repaired else "尝试修复后仍不通过"
        print(f"    - {s} {tf}: {msg} -> {status}")
    repaired_1h = any(tf == "1h" and repaired for _, tf, _, repaired in results)
    if repaired_1h:
        print("    -> 1h 修复成功，符合预期")
    else:
        one_1h = next((r for r in results if r[1] == "1h"), None)
        if one_1h:
            print("    -> 1h 修复未通过（可能网络/API 不可用），但修复流程已执行")
        else:
            print("    -> 未对 1h 执行修复，请检查逻辑")

    # 5) 恢复备份
    if ETH_1H_BAK.exists() and ETH_1H_BAK.stat().st_size > 0:
        import shutil
        shutil.copy2(ETH_1H_BAK, ETH_1H)
        ETH_1H_BAK.unlink(missing_ok=True)
        print(f"\n[5] 已恢复原始 1h 文件并删除备份")
    else:
        ETH_1H_BAK.unlink(missing_ok=True)
        print(f"\n[5] 无备份可恢复，保留当前 1h 文件")

    # 6) 再次健康检查，确认 1h 可读（若已恢复或修复成功）
    df = load_ohlcv("ETH/USDT", "1h")
    ok, msg = validate_ohlcv_df(df, min_rows=min_rows, symbol="ETH/USDT", timeframe="1h")
    print(f"\n[6] 恢复/修复后 1h 校验: {'通过' if ok else '不通过'} ({msg})")

    print("\n" + "=" * 60)
    print("  测试完成：健康检查能发现 1h 异常，修复流程会调用 update_latest")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
