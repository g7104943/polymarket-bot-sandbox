#!/usr/bin/env python3
"""
ETH 单币：GRU 回测「无高波过滤 vs 有高波过滤」对比。
与模拟盘 GRU_ETH_EK 同条件（0.54 有动态、400 起、>6万固定3000/≤6万5%）。
- 无泄漏数据 ≥365 天：用满 365 天单次回测。
- 无泄漏数据 <365 天：用满全部无泄漏天数单次回测（样本最多），再跑 2 个不重叠子窗口做一致性检查；
  若多窗一致则结论更可信；末尾提示「数据不足，建议满 365 天后再跑一次以决定是否实施高波策略」。
不读不写任何预测/模拟配置，不影响正在跑的模拟交易。
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest_gru_regime import (
    get_backtest_df_one_asset,
    get_no_leak_start_date,
    get_parquet_end_date,
    run_trading_loop,
    add_high_vol_skip_column,
    get_device,
)

ASSET = "ETH_USDT"
THRESHOLD = 0.54
ENABLE_DYNAMIC = True
INITIAL_CAP = 400.0
BET_RATIO = 0.05
ORDER_PRICE = 0.503
YEAR_DAYS = 365
MIN_DAYS_FOR_HALF = 30  # 子窗口至少 30 天


def main():
    data_src = PROJECT_ROOT / "data" / "raw"
    after_date = get_no_leak_start_date(data_src, None, ASSET)
    parquet_end = get_parquet_end_date(data_src, ASSET)
    after_ts = pd.Timestamp(after_date, tz="UTC")
    end_ts = pd.Timestamp(parquet_end, tz="UTC")
    actual_no_leak_days = (end_ts - after_ts).days
    device = get_device(use_mps=True)

    if actual_no_leak_days >= YEAR_DAYS:
        # 无泄漏满一年：单次 365 天回测
        end_date = (after_ts + pd.Timedelta(days=YEAR_DAYS)).strftime("%Y-%m-%d")
        df = get_backtest_df_one_asset(ASSET, test_days=YEAR_DAYS, device=device, data_src=data_src, after_date=after_date, end_date=end_date)
        if len(df) < 50:
            print("数据不足，跳过")
            return
        add_high_vol_skip_column(df)
        res_no = run_trading_loop(
            df, initial_capital=INITIAL_CAP, bet_ratio=BET_RATIO, prob_threshold=THRESHOLD,
            enable_dynamic_bet_ratio=ENABLE_DYNAMIC, order_price=ORDER_PRICE, stop_loss_exit_price=None,
            use_fixed_bet=False, use_smart_bet=True, enable_high_vol_filter=False,
        )
        res_yes = run_trading_loop(
            df, initial_capital=INITIAL_CAP, bet_ratio=BET_RATIO, prob_threshold=THRESHOLD,
            enable_dynamic_bet_ratio=ENABLE_DYNAMIC, order_price=ORDER_PRICE, stop_loss_exit_price=None,
            use_fixed_bet=False, use_smart_bet=True, enable_high_vol_filter=True,
        )
        results_no = [res_no]
        results_yes = [res_yes]
        print("回测区间: 无泄漏 {} 天 ({} ~ {})".format(YEAR_DAYS, after_date, end_date))
    else:
        # 无泄漏不足一年：用满全部无泄漏天数单次回测（样本最多），再跑 2 个不重叠子窗口看一致性
        if actual_no_leak_days < MIN_DAYS_FOR_HALF:
            print("无泄漏可用仅 {} 天，不足 {} 天，跳过".format(actual_no_leak_days, MIN_DAYS_FOR_HALF))
            return
        end_date = parquet_end
        df_full = get_backtest_df_one_asset(ASSET, test_days=actual_no_leak_days + 30, device=device, data_src=data_src, after_date=after_date, end_date=end_date)
        if len(df_full) < 50:
            print("数据不足，跳过")
            return
        add_high_vol_skip_column(df_full)
        res_no = run_trading_loop(
            df_full, initial_capital=INITIAL_CAP, bet_ratio=BET_RATIO, prob_threshold=THRESHOLD,
            enable_dynamic_bet_ratio=ENABLE_DYNAMIC, order_price=ORDER_PRICE, stop_loss_exit_price=None,
            use_fixed_bet=False, use_smart_bet=True, enable_high_vol_filter=False,
        )
        res_yes = run_trading_loop(
            df_full, initial_capital=INITIAL_CAP, bet_ratio=BET_RATIO, prob_threshold=THRESHOLD,
            enable_dynamic_bet_ratio=ENABLE_DYNAMIC, order_price=ORDER_PRICE, stop_loss_exit_price=None,
            use_fixed_bet=False, use_smart_bet=True, enable_high_vol_filter=True,
        )
        results_no = [res_no]
        results_yes = [res_yes]
        print("回测: 无泄漏可用 {} 天，用满全段 ({} ~ {})".format(actual_no_leak_days, after_date, parquet_end))

        # 2 个不重叠子窗口一致性检查
        half = actual_no_leak_days // 2
        if half >= MIN_DAYS_FOR_HALF:
            wins_yes = 0
            for i, (sname, sdays, edays) in enumerate([
                ("前半段", 0, half),
                ("后半段", half, actual_no_leak_days),
            ]):
                s_ts = after_ts + pd.Timedelta(days=sdays)
                e_ts = after_ts + pd.Timedelta(days=edays)
                es = s_ts.strftime("%Y-%m-%d")
                ee = e_ts.strftime("%Y-%m-%d")
                df_sub = get_backtest_df_one_asset(ASSET, test_days=edays - sdays + 30, device=device, data_src=data_src, after_date=es, end_date=ee)
                if len(df_sub) < 50:
                    continue
                add_high_vol_skip_column(df_sub)
                rn = run_trading_loop(df_sub, initial_capital=INITIAL_CAP, bet_ratio=BET_RATIO, prob_threshold=THRESHOLD,
                    enable_dynamic_bet_ratio=ENABLE_DYNAMIC, order_price=ORDER_PRICE, stop_loss_exit_price=None,
                    use_fixed_bet=False, use_smart_bet=True, enable_high_vol_filter=False)
                ry = run_trading_loop(df_sub, initial_capital=INITIAL_CAP, bet_ratio=BET_RATIO, prob_threshold=THRESHOLD,
                    enable_dynamic_bet_ratio=ENABLE_DYNAMIC, order_price=ORDER_PRICE, stop_loss_exit_price=None,
                    use_fixed_bet=False, use_smart_bet=True, enable_high_vol_filter=True)
                if ry["final_capital"] > rn["final_capital"]:
                    wins_yes += 1
                print("  {} ({}~{}d): 无高波={:.2f}  有高波={:.2f}  → 有高波更优={}".format(
                    sname, sdays, edays, rn["final_capital"], ry["final_capital"], "是" if ry["final_capital"] > rn["final_capital"] else "否"))
            print("  子窗一致性: 有高波更优 {}/2 窗".format(wins_yes))

    # 汇总
    cap_no = [r["final_capital"] for r in results_no]
    cap_yes = [r["final_capital"] for r in results_yes]
    n0 = sum(r["total_trades"] for r in results_no)
    n1 = sum(r["total_trades"] for r in results_yes)
    w0 = sum(r["wins"] for r in results_no)
    w1 = sum(r["wins"] for r in results_yes)
    avg_no = sum(cap_no) / len(cap_no)
    avg_yes = sum(cap_yes) / len(cap_yes)
    wr0 = (w0 / n0 * 100) if n0 else 0
    wr1 = (w1 / n1 * 100) if n1 else 0

    print("ETH GRU 0.54 有动态（与 GRU_ETH_EK 同条件）")
    print("  初始 {}u，<6万按当前资金 {:.0%} 下注（首笔约 {:.0f}u），买价 {}；手续费+滑点+买价下 50%% 胜率期望略负，资金下行属正常。".format(
        int(INITIAL_CAP), BET_RATIO, INITIAL_CAP * BET_RATIO, ORDER_PRICE))
    if len(results_no) == 1:
        print("  无高波过滤: 最终资金={:.2f}  笔数={}  胜率%={:.1f}".format(cap_no[0], n0, wr0))
        print("  有高波过滤: 最终资金={:.2f}  笔数={}  胜率%={:.1f}".format(cap_yes[0], n1, wr1))
    else:
        print("  无高波过滤: 平均最终资金={:.2f}  总笔数={}  胜率%={:.1f}".format(avg_no, n0, wr0))
        print("  有高波过滤: 平均最终资金={:.2f}  总笔数={}  胜率%={:.1f}".format(avg_yes, n1, wr1))
    if avg_yes > avg_no:
        print("  结论: 加高波过滤更优")
    else:
        print("  结论: 不加高波过滤更优或相当")
    if actual_no_leak_days < YEAR_DAYS:
        print("  ⚠ 当前无泄漏仅 {} 天，不足以最终确认；建议无泄漏满 365 天（或至少 6 个月）后再跑本脚本，若仍为「有高波更优」再考虑实施。".format(actual_no_leak_days))


if __name__ == "__main__":
    main()
