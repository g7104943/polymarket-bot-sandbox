#!/usr/bin/env python3
"""
22 组合回测：分两表输出，避免天数不一致导致排名不可比。

- 表1（105 天）：旧模型 + 对应超参（base_idx 0～4），用「旧模型最大无泄漏区间」= train_cutoff 起～数据末。
- 表2（365 天）：GRU + 对应超参（base_idx 5～12），用「GRU 最大无泄漏区间」= no_leak_start 起 365 天。

规则（与模拟/实盘一致）：
- 买入价固定 0.527，初始 400，≥6 万固定 3000，<6 万按配置比例（超参为档位 p1～p4）

用法（项目根目录）:
  python scripts/backtest_22_combos_unified.py
  python scripts/backtest_22_combos_unified.py --data-src data/raw --out-dir .
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# 复用 hyperparam_tune_13_combos 的常量与逻辑
from scripts.hyperparam_tune_13_combos import (
    B1,
    B2,
    B3,
    CAPITAL_THRESHOLD,
    FIXED_ORDER_PRICE_DEFAULT,
    INITIAL_CAPITAL,
    MAX_BET_CAP,
    filter_trades_by_date,
    filter_trades_by_entry,
    get_trades_gru,
    get_trades_old,
    get_trades_old_dual,
    run_equity_curve,
    simulate_combo_stats,
)
from scripts.backtest_gru_regime_report import OLD_POLYMARKET_COMBOS, SIM_13_GRU_COMBOS

DATA_RAW = PROJECT_ROOT / "data" / "raw"
REPORT_365_DAYS = 365

# 22 组合固定配置（与 启动三个版本 / 启动GRU新模型 / 启动超参组合.sh 一致）
# 1～13：非超参，delta=0，p1=p2=p3=p4=5（统一 5%）
# 14～22：超参，threshold 为脚本中的 PROB_THRESHOLD，delta=0 已含在阈值内，p1～p4 来自稳健表/启动脚本
COMBO_22_CONFIG: List[Dict[str, Any]] = [
    # 1～5 旧模型
    {"id": "logs_eth", "display": "ETH_92", "type": "old", "base_idx": 0, "th_up": 0.92, "th_down": None, "is_dual": False, "delta": 0, "p1": 5, "p2": 5, "p3": 5, "p4": 5},
    {"id": "logs_eth_10_90", "display": "ETH_10_90", "type": "old", "base_idx": 1, "th_up": 0.9, "th_down": 0.1, "is_dual": True, "delta": 0, "p1": 5, "p2": 5, "p3": 5, "p4": 5},
    {"id": "logs_btc", "display": "BTC_55", "type": "old", "base_idx": 2, "th_up": 0.55, "th_down": None, "is_dual": False, "delta": 0, "p1": 5, "p2": 5, "p3": 5, "p4": 5},
    {"id": "logs_xrp", "display": "XRP_53", "type": "old", "base_idx": 3, "th_up": 0.53, "th_down": None, "is_dual": False, "delta": 0, "p1": 5, "p2": 5, "p3": 5, "p4": 5},
    {"id": "logs_xrp_20_80", "display": "XRP_20_80", "type": "old", "base_idx": 4, "th_up": 0.8, "th_down": 0.2, "is_dual": True, "delta": 0, "p1": 5, "p2": 5, "p3": 5, "p4": 5},
    # 6～13 GRU
    {"id": "logs_gru_xrp_55", "display": "GRU_XRP_55", "type": "gru", "base_idx": 5, "th_up": 0.55, "th_down": None, "is_dual": False, "delta": 0, "p1": 5, "p2": 5, "p3": 5, "p4": 5},
    {"id": "logs_gru_btc_55", "display": "GRU_BTC_55", "type": "gru", "base_idx": 6, "th_up": 0.55, "th_down": None, "is_dual": False, "delta": 0, "p1": 5, "p2": 5, "p3": 5, "p4": 5},
    {"id": "logs_gru_eth_54", "display": "GRU_ETH_54", "type": "gru", "base_idx": 7, "th_up": 0.54, "th_down": None, "is_dual": False, "delta": 0, "p1": 5, "p2": 5, "p3": 5, "p4": 5},
    {"id": "logs_gru_sol_52", "display": "GRU_SOL_52", "type": "gru", "base_idx": 8, "th_up": 0.52, "th_down": None, "is_dual": False, "delta": 0, "p1": 5, "p2": 5, "p3": 5, "p4": 5},
    {"id": "logs_gru_eth_55_dyn", "display": "GRU_ETH_55_dyn", "type": "gru", "base_idx": 9, "th_up": 0.55, "th_down": None, "is_dual": False, "delta": 0, "p1": 5, "p2": 5, "p3": 5, "p4": 5},
    {"id": "logs_gru_eth_55_no1h4h", "display": "GRU_ETH_55_no1h4h", "type": "gru", "base_idx": 10, "th_up": 0.55, "th_down": None, "is_dual": False, "delta": 0, "p1": 5, "p2": 5, "p3": 5, "p4": 5},
    {"id": "logs_gru_btc_57_no1h4h", "display": "GRU_BTC_57_no1h4h", "type": "gru", "base_idx": 11, "th_up": 0.57, "th_down": None, "is_dual": False, "delta": 0, "p1": 5, "p2": 5, "p3": 5, "p4": 5},
    {"id": "logs_gru_xrp_53_no1h4h", "display": "GRU_XRP_53_no1h4h", "type": "gru", "base_idx": 12, "th_up": 0.53, "th_down": None, "is_dual": False, "delta": 0, "p1": 5, "p2": 5, "p3": 5, "p4": 5},
    # 14～22 超参（档位 p1～p4 与 启动超参组合.sh / 稳健表一致）
    {"id": "logs_eth_超参", "display": "超参_ETH", "type": "超参", "base_idx": 0, "th_up": 0.94, "th_down": None, "is_dual": False, "delta": 0, "p1": 2, "p2": 4, "p3": 2, "p4": 6},
    {"id": "logs_eth_10_90_超参", "display": "超参_ETH_10_90", "type": "超参", "base_idx": 1, "th_up": 0.95, "th_down": 0.05, "is_dual": True, "delta": 0, "p1": 6, "p2": 2, "p3": 4, "p4": 6},
    {"id": "logs_btc_超参", "display": "超参_BTC", "type": "超参", "base_idx": 2, "th_up": 0.60, "th_down": None, "is_dual": False, "delta": 0, "p1": 4, "p2": 10, "p3": 6, "p4": 2},
    {"id": "logs_gru_eth_54_超参", "display": "超参_GRU_ETH_54", "type": "超参", "base_idx": 7, "th_up": 0.56, "th_down": None, "is_dual": False, "delta": 0, "p1": 4, "p2": 10, "p3": 10, "p4": 2},
    {"id": "logs_gru_eth_55_dyn_超参", "display": "超参_GRU_ETH_55_dyn", "type": "超参", "base_idx": 9, "th_up": 0.57, "th_down": None, "is_dual": False, "delta": 0, "p1": 8, "p2": 10, "p3": 4, "p4": 2},
    {"id": "logs_gru_eth_55_no1h4h_超参", "display": "超参_GRU_ETH_55_no1h4h", "type": "超参", "base_idx": 10, "th_up": 0.57, "th_down": None, "is_dual": False, "delta": 0, "p1": 8, "p2": 10, "p3": 2, "p4": 6},
    {"id": "logs_gru_btc_57_no1h4h_超参", "display": "超参_GRU_BTC_57_no1h4h", "type": "超参", "base_idx": 11, "th_up": 0.59, "th_down": None, "is_dual": False, "delta": 0, "p1": 8, "p2": 2, "p3": 10, "p4": 10},
    {"id": "logs_gru_xrp_53_no1h4h_超参", "display": "超参_GRU_XRP_53_no1h4h", "type": "超参", "base_idx": 12, "th_up": 0.55, "th_down": None, "is_dual": False, "delta": 0, "p1": 4, "p2": 4, "p3": 2, "p4": 6},
    {"id": "logs_gru_sol_52_超参", "display": "超参_GRU_SOL_52", "type": "超参", "base_idx": 8, "th_up": 0.54, "th_down": None, "is_dual": False, "delta": 0, "p1": 2, "p2": 6, "p3": 10, "p4": 6},
]


def _get_old_model_train_cutoff_date(symbol: str, models_dir: Path) -> Optional[str]:
    """
    从旧模型 metadata 读取 train_cutoff_date，保证回测只用样本外数据。
    返回 YYYY-MM-DD，无则返回 None（沿用统一 no_leak_start）。
    """
    try:
        from src.python.predictor import _find_symbol_timeframe_model
        model_dir = _find_symbol_timeframe_model(symbol, "15m", models_root=models_dir)
        if not model_dir or not (model_dir / "metadata.json").exists():
            return None
        meta = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        t = meta.get("train_cutoff_date")
        if not t:
            return None
        # 解析为 YYYY-MM-DD
        dt = pd.to_datetime(t, utc=True)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def _build_base_combos() -> List[Dict[str, Any]]:
    """与 hyperparam_tune_13_combos 一致的 13 个基础组合（用于取交易序列）。"""
    combos = []
    for c in OLD_POLYMARKET_COMBOS:
        th_up = c.get("prob_threshold") or c.get("prob_threshold_up") or 0.5
        th_down = c.get("prob_threshold_down")
        is_dual = c.get("prob_threshold_up") is not None and c.get("prob_threshold_down") is not None
        combos.append({
            "id": c["name"],
            "type": "old",
            "symbol": c["symbol"],
            "models_dir": PROJECT_ROOT / c["models_dir"] if not Path(c["models_dir"]).is_absolute() else Path(c["models_dir"]),
            "threshold_up": th_up,
            "threshold_down": th_down,
            "is_dual": is_dual,
        })
    for c in SIM_13_GRU_COMBOS:
        combos.append({
            "id": c["combo_name"],
            "type": "gru",
            "asset": c["asset"],
            "threshold_up": c["threshold"],
            "threshold_down": None,
            "is_dual": False,
            "use_no1h4h": c.get("use_no1h4h", False),
        })
    return combos


def main():
    parser = argparse.ArgumentParser(description="22 组合统一回测：固定配置、0.527/400/6万3000、无泄漏 365 天、统一排名")
    parser.add_argument("--data-src", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--order-price", type=float, default=FIXED_ORDER_PRICE_DEFAULT)
    parser.add_argument("--advanced-rules", action="store_true",
                        help="同时用 5 层交易防护系统回测，生成对比 CSV")
    args = parser.parse_args()

    data_src = Path(args.data_src or DATA_RAW)
    if not data_src.is_absolute():
        data_src = PROJECT_ROOT / data_src
    out_dir = Path(args.out_dir or PROJECT_ROOT)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    fixed_price = getattr(args, "order_price", FIXED_ORDER_PRICE_DEFAULT)

    from scripts.backtest_gru_regime import get_no_leak_start_date, get_parquet_end_date, get_device

    no_leak_start = get_no_leak_start_date(data_src, None, "BTC_USDT")
    end_full = get_parquet_end_date(data_src, "BTC_USDT")
    end_dt = pd.Timestamp(end_full, tz="UTC")

    device = get_device(use_mps=True)
    models_best_no1h4h = PROJECT_ROOT / "experiments" / "gru_regime_v1" / "outputs" / "models_best_no1h4h"
    models_best = PROJECT_ROOT / "experiments" / "gru_regime_v1" / "outputs" / "models_best"
    base_combos = _build_base_combos()

    # 105 天区间：旧模型最大无泄漏（train_cutoff 起～数据末）
    latest_old_cutoff = no_leak_start
    for combo in base_combos:
        if combo["type"] == "old":
            tc = _get_old_model_train_cutoff_date(combo["symbol"], combo["models_dir"])
            if tc and tc > latest_old_cutoff:
                latest_old_cutoff = tc
    start_105 = latest_old_cutoff
    end_105_dt = min(pd.Timestamp(start_105, tz="UTC") + pd.Timedelta(days=REPORT_365_DAYS - 1), end_dt)
    end_105 = end_105_dt.strftime("%Y-%m-%d")
    days_105 = (end_105_dt - pd.Timestamp(start_105, tz="UTC")).days + 1

    # 365 天区间：GRU 最大无泄漏（no_leak_start 起 365 天）
    start_365 = no_leak_start
    end_365_dt = min(pd.Timestamp(start_365, tz="UTC") + pd.Timedelta(days=REPORT_365_DAYS - 1), end_dt)
    end_365 = end_365_dt.strftime("%Y-%m-%d")
    days_365 = (end_365_dt - pd.Timestamp(start_365, tz="UTC")).days + 1

    OLD_BASE_INDICES = (0, 1, 2, 3, 4)   # 5 旧 → 表1 含 5 旧 + 3 超参 = 8 个
    GRU_BASE_INDICES = (5, 6, 7, 8, 9, 10, 11, 12)  # 8 GRU → 表2 含 8 GRU + 6 超参 = 14 个
    ALL_BASE_INDICES = tuple(range(13))   # 表3 统一 105 天全部 22 个组合

    print("22 组合回测（分两表：105 天旧模型 / 365 天 GRU）")
    print("  规则: 买入价 {}  初始 {}   ≥6万固定3000  <6万按配置比例".format(fixed_price, INITIAL_CAPITAL))

    # 预加载：105 天区间只拉 base 0～4，365 天区间只拉 base 5～12
    trades_105_by_base: List[Optional[List[Dict[str, Any]]]] = [None] * 13
    trades_365_by_base: List[Optional[List[Dict[str, Any]]]] = [None] * 13

    for idx, combo in enumerate(base_combos):
        if combo["type"] == "old":
            if combo.get("is_dual"):
                raw = get_trades_old_dual(
                    combo["symbol"], combo["models_dir"], start_105, end_105,
                    combo["threshold_up"], combo.get("threshold_down") or 0.0, None, None,
                )
            else:
                raw = get_trades_old(combo["symbol"], combo["models_dir"], start_105, end_105)
            trades_105_by_base[idx] = raw
            print("  [105d base {}] {}  笔数: {}".format(idx + 1, combo["id"], len(raw)))
        else:
            models_ov = models_best_no1h4h if combo.get("use_no1h4h") and models_best_no1h4h.exists() else models_best
            all_t_365 = get_trades_gru(combo["asset"], start_365, end_365, data_src, models_ov, device)
            trades_365_by_base[idx] = filter_trades_by_date(all_t_365, start_365, end_365)
            all_t_105 = get_trades_gru(combo["asset"], start_105, end_105, data_src, models_ov, device)
            trades_105_by_base[idx] = filter_trades_by_date(all_t_105, start_105, end_105)
            print("  [365d base {}] {}  笔数: {}  |  [105d] 笔数: {}".format(
                idx + 1, combo["id"], len(trades_365_by_base[idx]), len(trades_105_by_base[idx])))

    def _build_rows(trades_by_base: List[Optional[List[Dict[str, Any]]]], config_filter) -> List[Dict[str, Any]]:
        rows = []
        for cfg in COMBO_22_CONFIG:
            if cfg["base_idx"] not in config_filter:
                continue
            raw = trades_by_base[cfg["base_idx"]]
            if not raw:
                rows.append({
                    "排名": None,
                    "组合": cfg["display"],
                    "组合id": cfg["id"],
                    "下注方式": "档位p1～p4" if cfg["type"] == "超参" else "5%",
                    "最终资金": None,
                    "胜率%": None,
                    "盈亏%": None,
                    "最大回撤%": None,
                    "交易次数": 0,
                    "最低/初始%": None,
                })
                continue
            filtered = filter_trades_by_entry(
                raw, cfg["th_up"], cfg.get("th_down"), cfg["delta"], cfg.get("is_dual", False)
            )
            b0 = int(round(cfg["delta"] * 100))
            st = simulate_combo_stats(
                filtered, cfg["th_up"], b0, B1, B2, B3,
                cfg["p1"], cfg["p2"], cfg["p3"], cfg["p4"], fixed_price,
            )
            pnl_pct = (st["final_capital"] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100.0
            rows.append({
                "组合": cfg["display"],
                "组合id": cfg["id"],
                "下注方式": "档位p1～p4" if cfg["type"] == "超参" else "5%",
                "最终资金": round(st["final_capital"], 2),
                "胜率%": round(st["win_rate_pct"], 1),
                "盈亏%": round(pnl_pct, 2),
                "最大回撤%": round(st["max_drawdown_pct"], 1),
                "交易次数": st["n_trades"],
                "最低/初始%": round(st["min_over_initial_pct"], 2),
            })
        return rows

    # 表1：105 天，旧模型 + 对应超参（base_idx 0～4）
    rows_105 = _build_rows(trades_105_by_base, OLD_BASE_INDICES)
    df_105 = pd.DataFrame(rows_105)
    df_105 = df_105.sort_values("最终资金", ascending=False, na_position="last").reset_index(drop=True)
    df_105.insert(0, "排名", range(1, len(df_105) + 1))
    path_105 = out_dir / "backtest_22_combos_105d_ranking.csv"
    df_105.to_csv(path_105, index=False, encoding="utf-8-sig")
    print("\n已写出: {} （{} 行，区间 {}～{}，{} 天）".format(
        path_105, len(df_105), start_105, end_105, days_105))

    # 表2：365 天，GRU + 对应超参（base_idx 5～12）
    rows_365 = _build_rows(trades_365_by_base, GRU_BASE_INDICES)
    df_365 = pd.DataFrame(rows_365)
    df_365 = df_365.sort_values("最终资金", ascending=False, na_position="last").reset_index(drop=True)
    df_365.insert(0, "排名", range(1, len(df_365) + 1))
    path_365 = out_dir / "backtest_22_combos_365d_ranking.csv"
    df_365.to_csv(path_365, index=False, encoding="utf-8-sig")
    print("已写出: {} （{} 行，区间 {}～{}，{} 天）".format(
        path_365, len(df_365), start_365, end_365, days_365))

    # 表3：统一 105 天无泄漏，22 个组合同一区间排名（与表1/表2 并存）
    rows_unified_105 = _build_rows(trades_105_by_base, ALL_BASE_INDICES)
    df_unified_105 = pd.DataFrame(rows_unified_105)
    df_unified_105 = df_unified_105.sort_values("最终资金", ascending=False, na_position="last").reset_index(drop=True)
    df_unified_105.insert(0, "排名", range(1, len(df_unified_105) + 1))
    path_unified_105 = out_dir / "backtest_22_combos_unified_105d_ranking.csv"
    df_unified_105.to_csv(path_unified_105, index=False, encoding="utf-8-sig")
    print("已写出: {} （22 行，统一区间 {}～{}，{} 天无泄漏）".format(
        path_unified_105, start_105, end_105, days_105))

    # ======= 5 层交易防护系统对比回测 =======
    if getattr(args, "advanced_rules", False):
        from src.python.trading_rules import (
            AdvancedEquityCurve, advanced_score, run_advanced_backtest,
            ADVANCED_FIXED_PARAMS,
        )

        # 默认参数（可从 hyperparam 搜索结果读取，此处用合理默认值）
        default_adv_params = {
            "min_edge": 0.02,
            "kelly_frac": 0.33,
            "roll_window": 20,
            "cold_threshold": 0.48,
            "dd_level1": 0.10,
            "dd_level2": 0.20,
            "dd_halt": 0.30,
            "max_capital_pct": 0.10,
        }
        default_adv_params.update(ADVANCED_FIXED_PARAMS)

        def _build_rows_advanced(
            trades_by_base: List[Optional[List[Dict[str, Any]]]],
            config_filter,
        ) -> List[Dict[str, Any]]:
            rows = []
            for cfg in COMBO_22_CONFIG:
                if cfg["base_idx"] not in config_filter:
                    continue
                raw = trades_by_base[cfg["base_idx"]]
                if not raw:
                    rows.append({
                        "组合": cfg["display"],
                        "组合id": cfg["id"],
                        "下注方式": "5层防护",
                        "最终资金": None,
                        "胜率%": None,
                        "盈亏%": None,
                        "最大回撤%": None,
                        "交易次数": 0,
                        "最低/初始%": None,
                    })
                    continue
                # 5 层系统不需要 filter_trades_by_entry，内部用 edge 过滤
                st = run_advanced_backtest(raw, cfg["th_up"], default_adv_params, fixed_price)
                pnl_pct = (st["final_capital"] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100.0
                rows.append({
                    "组合": cfg["display"],
                    "组合id": cfg["id"],
                    "下注方式": "5层防护",
                    "最终资金": round(st["final_capital"], 2),
                    "胜率%": round(st["win_rate_pct"], 1),
                    "盈亏%": round(pnl_pct, 2),
                    "最大回撤%": round(st["max_drawdown_pct"], 1),
                    "交易次数": st["n_trades"],
                    "最低/初始%": round(st["min_over_initial_pct"], 2),
                    "边际过滤": st["filtered_by_edge"],
                    "连败暂停": st["filtered_by_streak"],
                    "回撤熔断": st["halted_count"],
                })
            return rows

        print("\n--- 5 层交易防护系统对比回测 ---")
        print("  参数: min_edge={}, kelly_frac={}, dd_halt={}, roll_window={}".format(
            default_adv_params["min_edge"], default_adv_params["kelly_frac"],
            default_adv_params["dd_halt"], default_adv_params["roll_window"],
        ))

        # 统一 105 天：原始 vs 5 层
        rows_adv_105 = _build_rows_advanced(trades_105_by_base, ALL_BASE_INDICES)
        df_adv_105 = pd.DataFrame(rows_adv_105)
        df_adv_105 = df_adv_105.sort_values("最终资金", ascending=False, na_position="last").reset_index(drop=True)
        df_adv_105.insert(0, "排名", range(1, len(df_adv_105) + 1))

        # 合并对比表：原始 + 5 层并排
        df_orig = df_unified_105.copy()
        df_orig["规则"] = "原始"
        df_adv_copy = df_adv_105.copy()
        df_adv_copy["规则"] = "5层防护"
        df_compare = pd.concat([df_orig, df_adv_copy], ignore_index=True)
        df_compare = df_compare.sort_values(
            ["组合", "规则"], ascending=[True, True]
        ).reset_index(drop=True)
        df_compare.insert(0, "序号", range(1, len(df_compare) + 1))

        path_compare = out_dir / "backtest_22_combos_advanced_vs_original.csv"
        df_compare.to_csv(path_compare, index=False, encoding="utf-8-sig")
        print("已写出: {} （{} 行，原始 vs 5层 对比）".format(path_compare, len(df_compare)))

        # 单独的 5 层排名表
        path_adv = out_dir / "backtest_22_combos_unified_105d_advanced.csv"
        df_adv_105.to_csv(path_adv, index=False, encoding="utf-8-sig")
        print("已写出: {} （{} 行，5 层防护排名）".format(path_adv, len(df_adv_105)))


if __name__ == "__main__":
    main()
