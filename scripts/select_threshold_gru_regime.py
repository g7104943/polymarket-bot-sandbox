#!/usr/bin/env python3
"""
GRU Regime 阈值扫描：对 models_best 回测扫一遍阈值，按胜率/交易数选最佳。
初始资金固定为 400 和 1000。

在项目根目录执行:
  python scripts/select_threshold_gru_regime.py
  python scripts/select_threshold_gru_regime.py --test-days 365 --min-trades 30
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest_gru_regime import (
    ASSETS,
    get_backtest_df_one_asset,
    run_trading_loop,
    get_device,
)

# 默认初始资金：400 和 1000
DEFAULT_CAPITALS = [400.0, 1000.0]


def sweep_thresholds(
    test_days: int = 365,
    capitals: List[float] = None,
    threshold_start: float = 0.55,
    threshold_end: float = 0.90,
    threshold_step: float = 0.01,
    min_trades: int = 50,
    bet_ratio: float = 0.05,
    data_src: Path = None,
    no_mps: bool = False,
    after_date: Optional[str] = None,
) -> Dict[str, Any]:
    """
    对 GRU Regime 各资产扫阈值，初始资金 400 和 1000，按胜率（且交易数>=min_trades）选最佳。
    after_date 为 YYYY-MM-DD 时仅用该日期之后的数据（防泄露）。

    Returns:
        {
            'best_per_asset_capital': { (asset, capital): { threshold, win_rate, trades, ... } },
            'all_results': [ { asset, capital, threshold, trades, win_rate, profit_pct, ... } ],
            'params': { test_days, capitals, threshold_range, min_trades }
        }
    """
    if capitals is None:
        capitals = DEFAULT_CAPITALS
    if data_src is None:
        data_src = PROJECT_ROOT / "data" / "raw"
    if not data_src.is_absolute():
        data_src = PROJECT_ROOT / data_src

    threshold_range = [
        round(threshold_start + i * threshold_step, 2)
        for i in range(int((threshold_end - threshold_start) / threshold_step) + 1)
    ]
    device = get_device(use_mps=not no_mps)

    # 每个资产只加载/推理一次，得到带 pred 的 df
    asset_dfs: Dict[str, Any] = {}
    for asset in ASSETS:
        print(f"📂 加载 {asset} 回测数据与预测...", flush=True)
        try:
            df = get_backtest_df_one_asset(asset, test_days, device, data_src, after_date=after_date)
            asset_dfs[asset] = df
        except Exception as e:
            print(f"   ❌ {asset}: {e}")
            continue

    if not asset_dfs:
        raise RuntimeError("没有可用的资产回测数据")

    all_results: List[Dict[str, Any]] = []
    best_per: Dict[tuple, Dict] = {}  # (asset, capital) -> best row

    for asset, df in asset_dfs.items():
        print(f"\n🔍 扫描 {asset} 阈值 {threshold_range[0]:.2f} - {threshold_range[-1]:.2f} (共 {len(threshold_range)} 个), 资金 {capitals}")
        for thr in threshold_range:
            for cap in capitals:
                res = run_trading_loop(
                    df,
                    initial_capital=cap,
                    bet_ratio=bet_ratio,
                    prob_threshold=thr,
                )
                row = {
                    "asset": asset,
                    "initial_capital": cap,
                    "threshold": thr,
                    "total_trades": res["total_trades"],
                    "wins": res["wins"],
                    "losses": res["losses"],
                    "win_rate": round(res["win_rate"] * 100, 2),
                    "profit_pct": round(res["profit_pct"], 2),
                    "final_capital": round(res["final_capital"], 2),
                }
                all_results.append(row)

                key = (asset, cap)
                if res["total_trades"] >= min_trades:
                    cur_best = best_per.get(key)
                    if cur_best is None or res["win_rate"] > cur_best["win_rate"]:
                        best_per[key] = {
                            "threshold": thr,
                            "win_rate": round(res["win_rate"] * 100, 2),
                            "total_trades": res["total_trades"],
                            "profit_pct": round(res["profit_pct"], 2),
                            "final_capital": round(res["final_capital"], 2),
                        }

            # 当前阈值下 400/1000 的简要结果
            rows_this = [r for r in all_results if r["asset"] == asset and r["threshold"] == thr]
            r400 = next((r for r in rows_this if r["initial_capital"] == 400), None)
            r1000 = next((r for r in rows_this if r["initial_capital"] == 1000), None)
            s400 = f"400: {r400['total_trades']}笔 胜率{r400['win_rate']}%" if r400 else ""
            s1000 = f"1000: {r1000['total_trades']}笔 胜率{r1000['win_rate']}%" if r1000 else ""
            best_400 = best_per.get((asset, 400.0))
            best_1000 = best_per.get((asset, 1000.0))
            mark = ""
            if best_400 and best_400["threshold"] == thr:
                mark = " ⭐400"
            if best_1000 and best_1000["threshold"] == thr:
                mark = (mark + " ⭐1000") if mark else " ⭐1000"
            print(f"   阈值 {thr:.2f}  {s400}  |  {s1000}{mark}")

    return {
        "best_per_asset_capital": {
            f"{a}_{int(c)}": v for (a, c), v in best_per.items()
        },
        "all_results": all_results,
        "params": {
            "test_days": test_days,
            "capitals": capitals,
            "threshold_range": threshold_range,
            "min_trades": min_trades,
            "bet_ratio": bet_ratio,
            "after_date": after_date,
        },
    }


def main():
    ap = argparse.ArgumentParser(description="GRU Regime 阈值扫描，按胜率/交易数选最佳（初始资金 400、1000）")
    ap.add_argument("--test-days", type=int, default=365, help="回测天数，默认 365")
    ap.add_argument("--capitals", type=float, nargs="+", default=DEFAULT_CAPITALS, help="初始资金列表，默认 400 1000")
    ap.add_argument("--min-trades", type=int, default=50, help="最小交易数要求，默认 50")
    ap.add_argument("--threshold-start", type=float, default=0.55, help="阈值起始")
    ap.add_argument("--threshold-end", type=float, default=0.90, help="阈值结束")
    ap.add_argument("--threshold-step", type=float, default=0.01, help="阈值步长")
    ap.add_argument("--bet-ratio", type=float, default=0.05, help="下注比例")
    ap.add_argument("--data-src", type=str, default=None, help="数据目录，默认 data/raw")
    ap.add_argument("--after-date", type=str, default=None, help="防泄露：仅用该日期之后的数据，格式 YYYY-MM-DD（需晚于训练/验证集结束日）")
    ap.add_argument("--no-mps", action="store_true", help="禁用 MPS")
    ap.add_argument("--out-dir", type=str, default=None, help="结果输出目录，默认 data/reports")
    args = ap.parse_args()

    data_src = Path(args.data_src) if args.data_src else PROJECT_ROOT / "data" / "raw"
    if not data_src.is_absolute():
        data_src = PROJECT_ROOT / data_src
    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "data" / "reports"
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir

    print("=" * 60)
    print("  GRU Regime 阈值扫描（胜率/交易数选最佳）")
    print("=" * 60)
    print(f"  回测天数: {args.test_days}")
    print(f"  初始资金: {args.capitals}")
    print(f"  最小交易数: {args.min_trades}")
    if args.after_date:
        print(f"  防泄露截止日: 仅用 > {args.after_date} 的数据")
    print("=" * 60)

    try:
        result = sweep_thresholds(
            test_days=args.test_days,
            capitals=args.capitals,
            threshold_start=args.threshold_start,
            threshold_end=args.threshold_end,
            threshold_step=args.threshold_step,
            min_trades=args.min_trades,
            bet_ratio=args.bet_ratio,
            data_src=data_src,
            no_mps=args.no_mps,
            after_date=args.after_date,
        )
    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 打印最佳阈值汇总
    print("\n" + "=" * 60)
    print("  最佳阈值（按资产 × 初始资金）")
    print("=" * 60)
    for key, v in result["best_per_asset_capital"].items():
        print(f"  {key}: 阈值 {v['threshold']:.2f}  胜率 {v['win_rate']}%  交易 {v['total_trades']} 笔  盈亏 {v['profit_pct']:+.2f}%")
    print("=" * 60)

    # 保存
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "gru_regime_optimal_threshold.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    csv_path = out_dir / "gru_regime_threshold_sweep.csv"
    import pandas as pd
    pd.DataFrame(result["all_results"]).to_csv(csv_path, index=False)
    print(f"\n📁 结果已保存: {json_path}")
    print(f"   CSV: {csv_path}")


if __name__ == "__main__":
    main()
