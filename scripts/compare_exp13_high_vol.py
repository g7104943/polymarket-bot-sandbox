#!/usr/bin/env python3
"""
Exp13（v5_production_tv）bp0450～bp0530 组合：回测「无高波过滤 vs 有高波过滤」对比。
与模拟盘 exp13 同条件（ETH 15m、初始 400、>6万固定3000/≤6万5%）。
不修改任何预测/模拟配置，不影响正在跑的模拟交易。
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest_simulation import run_backtest_for_pair
from scripts.backtest_gru_regime import get_no_leak_start_date, get_parquet_end_date
from src.python.predictor import load_predictor, _find_symbol_timeframe_model

SYMBOL = "ETH/USDT"
TIMEFRAME = "15m"
MODELS_DIR = PROJECT_ROOT / "data" / "models"
EXP13_DIR = "v5_production_tv"
INITIAL_CAP = 400.0
BET_RATIO = 0.05
# bp0450 ~ bp0530
BUY_PRICES = [0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53]
YEAR_DAYS = 365


def main():
    if not models_root.exists():
        print(f"Exp13 模型目录不存在: {models_root}")
        return
    model_dir = _find_symbol_timeframe_model(SYMBOL, TIMEFRAME, models_root=models_root)
    if not model_dir:
        print(f"未找到 {SYMBOL} {TIMEFRAME} 模型")
        return
    model, feats, meta = load_predictor(model_dir)
    feature_names = list(feats) if feats is not None else []

    data_src = PROJECT_ROOT / "data" / "raw"
    after_date = get_no_leak_start_date(data_src, None, "ETH_USDT")
    parquet_end = get_parquet_end_date(data_src, "ETH_USDT")
    after_ts = pd.Timestamp(after_date, tz="UTC")
    end_ts = pd.Timestamp(parquet_end, tz="UTC")
    actual_days = (end_ts - after_ts).days
    test_days = min(YEAR_DAYS, actual_days) + 30
    end_date = parquet_end if actual_days < YEAR_DAYS else (after_ts + pd.Timedelta(days=YEAR_DAYS)).strftime("%Y-%m-%d")

    print("Exp13 高波动策略对比 ({} ~ {}, 可用无泄漏 {} 天)".format(after_date, end_date, actual_days))
    print("  初始 {}u, 5% 下注, 买价 bp0450～bp0530".format(int(INITIAL_CAP)))
    print()

    rows = []
    for bp in BUY_PRICES:
        r0 = run_backtest_for_pair(
            SYMBOL, TIMEFRAME, model, feature_names,
            initial_capital=INITIAL_CAP, bet_ratio=BET_RATIO, test_days=test_days,
            prob_threshold=0.5, train_cutoff_date=after_date, end_date=end_date,
            order_price=bp, use_fixed_bet=False, use_smart_bet=True,
            enable_high_vol_filter=False,
        )
        if r0.get("error"):
            rows.append({"bp": bp, "err": r0["error"], "cap_no": None, "cap_yes": None})
            continue
        r1 = run_backtest_for_pair(
            SYMBOL, TIMEFRAME, model, feature_names,
            initial_capital=INITIAL_CAP, bet_ratio=BET_RATIO, test_days=test_days,
            prob_threshold=0.5, train_cutoff_date=after_date, end_date=end_date,
            order_price=bp, use_fixed_bet=False, use_smart_bet=True,
            enable_high_vol_filter=True,
        )
        if r1.get("error"):
            rows.append({"bp": bp, "err": r1["error"], "cap_no": r0["final_capital"], "cap_yes": None})
            continue
        n0, n1 = r0["total_trades"] or 0, r1["total_trades"] or 0
        wr0 = (r0["wins"] / n0 * 100) if n0 else 0
        wr1 = (r1["wins"] / n1 * 100) if n1 else 0
        rows.append({
            "bp": bp,
            "cap_no": r0["final_capital"],
            "n_no": n0,
            "wr_no": wr0,
            "cap_yes": r1["final_capital"],
            "n_yes": n1,
            "wr_yes": wr1,
            "yes_better": r1["final_capital"] > r0["final_capital"],
        })

    print("\n  bp    无高波_资金  无高波_笔数  无高波_胜率%  有高波_资金  有高波_笔数  有高波_胜率%  有高波更优")
    print("  " + "-" * 90)
    for r in rows:
        if r.get("err"):
            print("  {:.2f}  错误: {}".format(r["bp"], r["err"]))
            continue
        better = "是" if r["yes_better"] else "否"
        print("  {:.2f}  {:>10.2f}  {:>8}  {:>8.1f}  {:>10.2f}  {:>8}  {:>8.1f}  {}".format(
            r["bp"], r["cap_no"], r["n_no"], r["wr_no"], r["cap_yes"], r["n_yes"], r["wr_yes"], better))
    wins_yes = sum(1 for r in rows if r.get("yes_better"))
    print("  " + "-" * 90)
    print("  有高波更优: {}/{} 个 bp".format(wins_yes, len([r for r in rows if not r.get("err")])))


if __name__ == "__main__":
    main()
