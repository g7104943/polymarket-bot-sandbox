#!/usr/bin/env python3
"""
未泄露 365 天回测报告：旧模型 4 组合 + 新模型 GRU 无动态/有动态。
6 万初始、固定每笔 3000、0.503 价格；有动态=回撤>20%% 下注 1500、>10%% 下注 2250、否则 3000。
按最终资金从高到低排序的 CSV。

在项目根目录执行:
  python scripts/backtest_gru_regime_report.py --auto-after-date
  ./一键回测GRU_未泄露一年.sh
"""
import os
import sys
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# --no-1h4h：在导入 backtest_gru_regime 前设置，供其读取 MODELS_BEST 与 GRU_SKIP_MTF
if "--no-1h4h" in sys.argv:
    _no1h4h_best = PROJECT_ROOT / "experiments" / "gru_regime_v1" / "outputs" / "models_best_no1h4h"
    os.environ["GRU_MODELS_BEST"] = str(_no1h4h_best)
    os.environ["GRU_SKIP_MTF"] = "1"

from scripts.backtest_gru_regime import (
    ASSETS,
    get_backtest_df_one_asset,
    get_no_leak_start_date,
    run_trading_loop,
    get_device,
    add_high_vol_skip_column,
)

TIMEFRAME = "15m"
MODEL_LABEL = "gru_regime_best"
# 阈值 0.51～0.98，步长 0.01（默认全部资产统一）
THRESHOLD_RANGE = [round(0.51 + i * 0.01, 2) for i in range(48)]  # 0.51 .. 0.98
# 4 资产 GRU 无1h4h 专用：0.53～0.59（含 0.53 和 0.59）
THRESHOLD_RANGE_NO1H4H = [round(0.53 + i * 0.01, 2) for i in range(7)]  # 0.53 .. 0.59
REPORT_YEAR_DAYS = 365

# 与 启动三个版本.sh 五组旧模型一致：eth_92无动态、eth_10-90有动态、btc_55无动态、xrp_53无动态、xrp_20-80无动态
OLD_POLYMARKET_COMBOS = [
    {"name": "logs_eth", "models_dir": "data/models", "symbol": "ETH/USDT", "prob_threshold": 0.92, "prob_threshold_up": None, "prob_threshold_down": None, "enable_dynamic": False},
    {"name": "logs_eth_10_90", "models_dir": "data/models", "symbol": "ETH/USDT", "prob_threshold": None, "prob_threshold_up": 0.9, "prob_threshold_down": 0.1, "enable_dynamic": True},
    {"name": "logs_btc", "models_dir": "data/models_C", "symbol": "BTC/USDT", "prob_threshold": 0.55, "prob_threshold_up": None, "prob_threshold_down": None, "enable_dynamic": False},
    {"name": "logs_xrp", "models_dir": "data/models_C", "symbol": "XRP/USDT", "prob_threshold": 0.53, "prob_threshold_up": None, "prob_threshold_down": None, "enable_dynamic": False},
    {"name": "logs_xrp_20_80", "models_dir": "data/models_C", "symbol": "XRP/USDT", "prob_threshold": None, "prob_threshold_up": 0.8, "prob_threshold_down": 0.2, "enable_dynamic": False},
]

# 与 polymarket 五组 GRU 新模型一致（模拟盘 10 组合 = 5 旧 + 5 GRU）；新模型 GRU 全部无校准
GRU_SIM_COMBOS = [
    {"asset": "ETH_USDT", "threshold": 0.55, "enable_dynamic": False, "combo_name": "logs_gru_eth_55"},
    {"asset": "XRP_USDT", "threshold": 0.55, "enable_dynamic": False, "combo_name": "logs_gru_xrp_55"},
    {"asset": "BTC_USDT", "threshold": 0.55, "enable_dynamic": False, "combo_name": "logs_gru_btc_55"},
    {"asset": "ETH_USDT", "threshold": 0.54, "enable_dynamic": True, "combo_name": "logs_gru_eth_54"},
    {"asset": "SOL_USDT", "threshold": 0.52, "enable_dynamic": False, "combo_name": "logs_gru_sol_52"},
]

# 与 查看实时日志.sh 选项 1～13 一致：正在跑模拟的 13 个进程（5 旧 + 8 GRU），用于 --sim-13-combos 按金额排名
#
# 13 组合完整定义：
#   1) logs_eth         — 旧模型 ETH，阈值 0.92，无动态，data/models
#   2) logs_eth_10_90    — 旧模型 ETH，双阈值 0.1/0.9，有动态，data/models
#   3) logs_btc          — 旧模型 BTC，阈值 0.55，无动态，data/models_C
#   4) logs_xrp          — 旧模型 XRP，阈值 0.53，无动态，data/models_C
#   5) logs_xrp_20_80    — 旧模型 XRP，双阈值 0.2/0.8，无动态，data/models_C
#   6) logs_gru_xrp_55   — GRU XRP 0.55 无动态，有1h4h
#   7) logs_gru_btc_55   — GRU BTC 0.55 无动态，有1h4h
#   8) logs_gru_eth_54   — GRU ETH 0.54 有动态，有1h4h
#   9) logs_gru_sol_52   — GRU SOL 0.52 无动态，有1h4h
#  10) logs_gru_eth_55_dyn    — GRU ETH 0.55 有动态，有1h4h
#  11) logs_gru_eth_55_no1h4h — GRU ETH 0.55 无动态，无1h4h（models_best_no1h4h）
#  12) logs_gru_btc_57_no1h4h — GRU BTC 0.57 无动态，无1h4h（models_best_no1h4h）
#  13) logs_gru_xrp_53_no1h4h — GRU XRP 0.53 无动态，无1h4h（models_best_no1h4h）
#
# use_no1h4h=True 时用 models_best_no1h4h，否则用 models_best（有1h4h）
SIM_13_GRU_COMBOS = [
    {"asset": "XRP_USDT", "threshold": 0.55, "enable_dynamic": False, "combo_name": "logs_gru_xrp_55", "use_no1h4h": False},
    {"asset": "BTC_USDT", "threshold": 0.55, "enable_dynamic": False, "combo_name": "logs_gru_btc_55", "use_no1h4h": False},
    {"asset": "ETH_USDT", "threshold": 0.54, "enable_dynamic": True, "combo_name": "logs_gru_eth_54", "use_no1h4h": False},
    {"asset": "SOL_USDT", "threshold": 0.52, "enable_dynamic": False, "combo_name": "logs_gru_sol_52", "use_no1h4h": False},
    {"asset": "ETH_USDT", "threshold": 0.55, "enable_dynamic": True, "combo_name": "logs_gru_eth_55_dyn", "use_no1h4h": False},
    {"asset": "ETH_USDT", "threshold": 0.55, "enable_dynamic": False, "combo_name": "logs_gru_eth_55_no1h4h", "use_no1h4h": True},
    {"asset": "BTC_USDT", "threshold": 0.57, "enable_dynamic": False, "combo_name": "logs_gru_btc_57_no1h4h", "use_no1h4h": True},
    {"asset": "XRP_USDT", "threshold": 0.53, "enable_dynamic": False, "combo_name": "logs_gru_xrp_53_no1h4h", "use_no1h4h": True},
]
# 12 组合 = 5 旧 + 7 GRU（与 查看实时日志 1～12 一致，不含第 13 个 xrp_53_no1h4h）
SIM_12_GRU_COMBOS = SIM_13_GRU_COMBOS[:7]

# 所有模型组合统一输出列：排名…阈值、动态调整、有无止损、反转次数、校准、组合（组合为 gru_eth_0.54_无动态 格式）
RANKING_COL_ORDER = [
    "排名", "币种", "周期", "交易次数", "胜场", "败场", "胜率%",
    "最终资金", "盈亏%", "最大回撤%", "最低/初始%", "阈值", "动态调整", "有无止损", "反转次数", "校准", "组合",
]


def _model_short(model: str) -> str:
    """模型简称，用于组合显示 gru_eth_0.54_无动态"""
    if model == "gru_regime_best":
        return "gru"
    if model == "models_C":
        return "models_c"
    return "models"


def _to_ranking_format(df: pd.DataFrame) -> pd.DataFrame:
    """统一为排名 CSV 格式：动态调整=策略过滤，组合=model_symbol_thr_动态（有校准时追加 _校准），只保留 RANKING_COL_ORDER"""
    if df.empty:
        return df
    df = df.copy()
    if "策略过滤" in df.columns:
        df["动态调整"] = df["策略过滤"]
    if "模型" in df.columns and "币种" in df.columns and "阈值" in df.columns and "策略过滤" in df.columns:
        def _combo_row(r):
            base = f"{_model_short(str(r['模型']))}_{str(r['币种']).lower()}_{r['阈值']}_{r['策略过滤']}"
            if "校准" in df.columns and r.get("校准") and str(r["校准"]).strip():
                return f"{base}_{r['校准']}"
            return base
        df["组合"] = df.apply(_combo_row, axis=1)
    return df[[c for c in RANKING_COL_ORDER if c in df.columns]]


def asset_to_symbol(asset: str) -> str:
    """BTC_USDT -> BTC"""
    return asset.replace("_USDT", "")


def _run_high_vol_comparison_sim12(
    data_src: Path,
    device,
    after_date: str,
    end_date: Optional[str],
    year_days: int,
    initial_cap: float,
    ratio_005: float,
    order_price: float,
    _mtpd_kw: dict,
    df_ranking: pd.DataFrame,
    models_best_no1h4h: Optional[Path] = None,
) -> None:
    """对 7 个 GRU 组合跑「有高波动过滤」回测，与 df_ranking 中无过滤结果对比并打印表。不修改 df_ranking。"""
    comparison_rows = []
    for g in SIM_12_GRU_COMBOS:
        asset = g["asset"]
        combo_key = f"gru_{asset_to_symbol(asset).lower()}_{g['threshold']:.2f}_{'有动态' if g['enable_dynamic'] else '无动态'}"
        combo_key += "_无1h4h" if g.get("use_no1h4h") else "_有1h4h"
        no_filter = df_ranking[df_ranking["组合"] == combo_key]
        if no_filter.empty:
            continue
        no_row = no_filter.iloc[0]
        try:
            df = get_backtest_df_one_asset(
                asset, test_days=year_days, device=device, data_src=data_src,
                after_date=after_date, end_date=end_date,
                models_best_override=models_best_no1h4h if (g.get("use_no1h4h") and models_best_no1h4h and models_best_no1h4h.exists()) else None,
            )
        except Exception as e:
            print(f"   [高波对比] {combo_key}: 取 df 失败 {e}")
            continue
        if len(df) < 50:
            continue
        add_high_vol_skip_column(df)
        res = run_trading_loop(
            df, initial_capital=initial_cap, bet_ratio=ratio_005, prob_threshold=g["threshold"],
            enable_dynamic_bet_ratio=g["enable_dynamic"], order_price=order_price, stop_loss_exit_price=None,
            use_fixed_bet=False, use_smart_bet=True,
            enable_high_vol_filter=True,
            **_mtpd_kw,
        )
        n = res.get("total_trades") or 0
        wr = (res.get("wins") or 0) / n * 100 if n > 0 else 0.0
        comparison_rows.append({
            "组合": combo_key,
            "无过滤_最终资金": no_row["最终资金"],
            "无过滤_交易次数": int(no_row["交易次数"]),
            "无过滤_胜率%": no_row["胜率%"],
            "有过滤_最终资金": round(res["final_capital"], 2),
            "有过滤_交易次数": res.get("total_trades", 0),
            "有过滤_胜率%": round(wr, 1),
        })
    if not comparison_rows:
        print("\n⚠️ 高波动对比：未匹配到任何 GRU 组合（或全部失败），跳过打印")
        return
    comp_df = pd.DataFrame(comparison_rows)
    print("\n📊 7 GRU 组合：无高波过滤 vs 有高波过滤")
    print(comp_df.to_string(index=False))


def run_report(
    after_date: str,
    initial_capital: float = 60000.0,
    bet_ratio: float = 0.05,
    data_src: Path = None,
    no_mps: bool = False,
    year_days: int = REPORT_YEAR_DAYS,
    order_price: float = 0.5,
    stop_loss_exit_price: Optional[float] = None,
    asset_only: str = None,
    threshold_range: List[float] = None,

    add_400_5pct_profile: bool = False,
    run_all_11_combos: bool = False,
    capital_400_smart: bool = False,
    compare_stop_loss: bool = False,
    unified_eth: bool = False,
    unified_all: bool = False,
    with_old_combos: bool = False,
    max_trades_per_day: Optional[int] = None,
    run_sim_12_combos: bool = False,
    run_compare_high_vol: bool = False,
) -> pd.DataFrame:
    """
    6 万初始、固定每笔 3000、0.503 价格（默认）。order_price/stop_loss_exit_price 可指定固定买入价与败局止损回收价。
    run_sim_12_combos: 若 True，仅跑 查看实时日志.sh 里选项 1～12 对应的 12 个组合（5 旧 + 7 GRU），每组合一行，按最终资金降序排名。
    max_trades_per_day: 每日最大交易笔数，传入 0 表示不限制；None 表示用 backtest_gru_regime 默认 20。
    unified_eth: 仅 ETH，有1h4h 与 无1h4h 两套模型同条件（365 天、capital_400_smart）一起回测、统一排名，不做止损对比。
    unified_all: 10 组合（5 旧 + 5 GRU 有1h4h）+ 4 资产 GRU 无1h4h（0.53～0.59）同条件 365 天、无止损，统一排名。
    compare_stop_loss: 若 True 且 stop_loss_exit_price 有值，每个(阈值,动态)会跑有止损+无止损两条，CSV 列 有无止损。
    有动态：回撤>20%% 下注 1500，>10%% 下注 2250，否则 3000。
    仅使用 after_date 之后 year_days 天内的数据（严格未泄露一年）。
    asset_only: 若指定（如 SOL_USDT），仅回测该资产 GRU，使用 threshold_range，跳过旧模型 4 组合。
    threshold_range: 与 asset_only 配套使用，阈值列表（如 0.51～0.98 步长 0.01）。
    add_400_5pct_profile: 若 True 且 asset_only 时，在现有 6万/固定3000 基础上再跑一份 初始400、每笔=当前总资金×5%%（跟随总资金，如到2万则下1000）的无动态+有动态，并写入同一 CSV（列 资金设定 区分）。
    run_all_11_combos: 若 True，仅跑模拟盘 10 组合（5 旧 + 5 GRU），初始400、>6万3000/≤6万5%%，每(组合,动态)保留带/不带校准更优一条，共 20 个排名。
    capital_400_smart: 若 True 且 asset_only 时，单资产回测用「初始400、资金≥6万固定3000、<6万5%」规则（与 11 组合一致）。
    """
    if data_src is None:
        data_src = PROJECT_ROOT / "data" / "raw"
    if not data_src.is_absolute():
        data_src = PROJECT_ROOT / data_src

    _mtpd_kw = {"max_trades_per_day": max_trades_per_day} if max_trades_per_day is not None else {}

    device = get_device(use_mps=not no_mps)
    # 固定 365 天窗口：end_date = after_date + year_days，由加载层直接取 (after_date, end_date]
    end_date = (pd.Timestamp(after_date, tz="UTC") + pd.Timedelta(days=year_days)).strftime("%Y-%m-%d")

    rows: List[Dict[str, Any]] = []
    # _append_row 写入 模型、策略过滤、组合(原始名)；最后统一用 _to_ranking_format 转为 RANKING_COL_ORDER

    capital_label_default = "6万/固定3000" if (asset_only and add_400_5pct_profile) else ""
    if run_all_11_combos or run_sim_12_combos:
        capital_label_default = ""

    def _append_row(symbol_short: str, res: Dict, model: str, thr_str: str, filter_str: str, combo: str, capital_label: str = None, cal_label: str = None, stop_label: str = ""):
        if capital_label is None:
            capital_label = capital_label_default
        total_trades = res["total_trades"]
        wins = res["wins"]
        losses = res["losses"]
        win_rate_pct = round(wins / total_trades * 100, 1) if total_trades > 0 else 0.0
        max_dd = max(0, min(100, res.get("max_drawdown", 0)))
        min_over_init = max(0, min(1000, res.get("min_over_initial_pct", 100)))
        rows.append({
            "排名": None,
            "币种": symbol_short,
            "周期": TIMEFRAME,
            "交易次数": total_trades,
            "胜场": wins,
            "败场": losses,
            "胜率%": win_rate_pct,
            "最终资金": round(res["final_capital"], 2),
            "盈亏%": round(res["profit_pct"], 2),
            "最大回撤%": round(max_dd, 2),
            "最低/初始%": round(min_over_init, 2),
            "模型": model,
            "阈值": thr_str,
            "策略过滤": filter_str,
            "有无止损": stop_label if stop_label else "—",
            "反转次数": res.get("reversal_count", 0),
            "校准": cal_label if cal_label is not None else "",
            "资金设定": capital_label,
            "组合": combo,
        })

    # ========== 10 组合模式：5 旧 + 5 GRU，初始 400，贴近实盘（>6万固定3000/≤6万5%），总资金逐笔变化 ==========
    if run_all_11_combos:
        from scripts.backtest_simulation import run_backtest_for_pair
        from src.python.predictor import load_predictor, _find_symbol_timeframe_model
        initial_cap, ratio_005 = 400.0, 0.05  # 初始 400u，每笔按当前资金：>6万固定3000、≤6万按总资金5%
        smart_label = "初始400、>6万固定3000/≤6万5%"
        print("\n📊 10 组合回测（5 旧 + 5 GRU）→ 每(组合,动态)保留带/不带校准更优一条 → 20 个排名")
        print("   初始 400u；>6万固定3000/≤6万5%（有动态时按回撤减仓）；买入价 0.503，365 天")
        # 5 旧模型：每组合跑 4 遍（有动态/无动态 × 带校准/不带校准），取更优一条
        print("\n📊 旧模型 5 组合（每组合 有动态/无动态 × 带校准/不带校准 = 4 条，保留更优）…")
        for combo in OLD_POLYMARKET_COMBOS:
            md = combo["models_dir"]
            models_dir = (PROJECT_ROOT / md) if not Path(md).is_absolute() else Path(md)
            model_dir = _find_symbol_timeframe_model(combo["symbol"], TIMEFRAME, models_root=models_dir)
            if not model_dir:
                print(f"   ⚠️ 未找到 {combo['symbol']} {combo['name']}，跳过")
                continue
            try:
                model, feats, meta = load_predictor(model_dir)
            except Exception as e:
                print(f"   ❌ {combo['name']}: 加载失败 {e}")
                continue
            thr_str = f"{combo['prob_threshold']}" if combo.get("prob_threshold") is not None else f"{int((combo.get('prob_threshold_down') or 0)*100)}-{int((combo.get('prob_threshold_up') or 0)*100)}"
            sym_short = combo["symbol"].replace("/USDT", "")
            for enable_dyn in [True, False]:
                filter_str = "有动态" if enable_dyn else "无动态"
                for cal_label, cal, cal_m in [("带校准", meta.get("calibrator"), meta.get("calibration")), ("不带校准", None, None)]:
                    try:
                        res = run_backtest_for_pair(
                            combo["symbol"], TIMEFRAME, model, feats,
                            initial_capital=initial_cap, bet_ratio=ratio_005, test_days=year_days,
                            prob_threshold=combo.get("prob_threshold"),
                            prob_threshold_up=combo.get("prob_threshold_up"),
                            prob_threshold_down=combo.get("prob_threshold_down"),
                            train_cutoff_date=after_date, end_date=end_date,
                            enable_dynamic_bet_ratio=enable_dyn,
                            order_price=order_price, use_fixed_bet=False, use_smart_bet=True,
                            calibrator=cal, calibration_method=cal_m,
                        )
                    except Exception as e:
                        print(f"   ❌ {combo['name']} {filter_str} {cal_label}: {e}")
                        continue
                    if res.get("error"):
                        continue
                    if res.get("total_trades") == 0 and res.get("prob_min") is not None:
                        print(f"   ⚠️ {combo['name']} {filter_str} {cal_label}: 0 笔交易，回测期内 pred_prob 范围 {res['prob_min']:.2f}~{res['prob_max']:.2f}，未达双阈值")
                    _append_row(sym_short, res, models_dir.name, thr_str, filter_str, combo["name"], capital_label=smart_label, cal_label=cal_label)
        # 5 GRU：无校准器，每组合只跑 2 遍（有动态/无动态），不区分带/不带校准
        print("\n📊 GRU 新模型 5 组合（无校准器，每组合 有动态/无动态 = 2 条）…")
        _actual_range_printed = False
        for g in GRU_SIM_COMBOS:
            asset = g["asset"]
            try:
                df = get_backtest_df_one_asset(
                    asset, test_days=year_days, device=device, data_src=data_src,
                    after_date=after_date, end_date=end_date,
                )
            except Exception as e:
                print(f"   ❌ {g['combo_name']}: {e}")
                continue
            if len(df) < 50:
                continue
            if not _actual_range_printed and "date" in df.columns:
                dmin, dmax = df["date"].min(), df["date"].max()
                try:
                    delta = pd.Timestamp(dmax) - pd.Timestamp(dmin)
                    days_actual = int(delta.total_seconds() / 86400)
                except Exception:
                    days_actual = max(1, len(df) // 96)
                dmin_s = pd.Timestamp(dmin).strftime("%Y-%m-%d") if hasattr(pd.Timestamp(dmin), "strftime") else str(dmin)[:10]
                dmax_s = pd.Timestamp(dmax).strftime("%Y-%m-%d") if hasattr(pd.Timestamp(dmax), "strftime") else str(dmax)[:10]
                print(f"   📅 实际回测区间: {dmin_s} ~ {dmax_s}（约 {days_actual} 天）")
                _actual_range_printed = True
            symbol = asset_to_symbol(asset)
            thr_str = f"{g['threshold']:.2f}"
            for enable_dyn in [True, False]:
                filter_str = "有动态" if enable_dyn else "无动态"
                res = run_trading_loop(
                    df, initial_capital=initial_cap, bet_ratio=ratio_005, prob_threshold=g["threshold"],
                    enable_dynamic_bet_ratio=enable_dyn, order_price=order_price, stop_loss_exit_price=stop_loss_exit_price, use_fixed_bet=False, use_smart_bet=True,
                    **_mtpd_kw,
                )
                _append_row(symbol, res, MODEL_LABEL, thr_str, filter_str, g["combo_name"], capital_label=smart_label, cal_label="—")
        df_ranking = pd.DataFrame(rows)
        if not df_ranking.empty:
            # 每个(组合, 策略过滤)只保留一条：带/不带校准中最终资金更高的；同资金则保留不带校准（不依赖 groupby，保证 策略过滤/组合 列不丢）
            kept = []
            for (combo_key, filter_key), g in df_ranking.groupby(["组合", "策略过滤"]):
                if len(g) <= 1:
                    kept.append(g)
                    continue
                g = g.sort_values("最终资金", ascending=False)
                best_cap = g["最终资金"].iloc[0]
                candidates = g[g["最终资金"] == best_cap]
                no_cal = candidates[candidates["校准"] == "不带校准"]
                if not no_cal.empty:
                    kept.append(no_cal.head(1))
                else:
                    kept.append(candidates.head(1))
            df_ranking = pd.concat(kept, ignore_index=True)
            df_ranking = df_ranking.sort_values("最终资金", ascending=False).reset_index(drop=True)
            df_ranking["排名"] = df_ranking.index + 1
            df_ranking = _to_ranking_format(df_ranking)
        return df_ranking

    # ========== 模拟盘 12 组合（与 查看实时日志.sh 选项 1～12 一致）：5 旧 + 7 GRU，每组合一行，按最终资金降序排名 ==========
    if run_sim_12_combos:
        from scripts.backtest_simulation import run_backtest_for_pair
        from src.python.predictor import load_predictor, _find_symbol_timeframe_model
        models_best_no1h4h = PROJECT_ROOT / "experiments" / "gru_regime_v1" / "outputs" / "models_best_no1h4h"
        initial_cap, ratio_005 = 400.0, 0.05
        smart_label = "初始400、>6万固定3000/≤6万5%"
        print("\n📊 模拟盘 12 组合回测（与 查看实时日志 1～12 一致），按最终资金降序排名")
        print("   初始 400u；>6万固定3000/≤6万5%；买入价 {}，365 天，无止损/无交易上限".format(order_price))
        # 1) 5 旧模型：每组合一行（不带校准）
        print("\n📊 旧模型 5 组合…")
        for combo in OLD_POLYMARKET_COMBOS:
            md = combo["models_dir"]
            models_dir = (PROJECT_ROOT / md) if not Path(md).is_absolute() else Path(md)
            model_dir = _find_symbol_timeframe_model(combo["symbol"], TIMEFRAME, models_root=models_dir)
            if not model_dir:
                print(f"   ⚠️ 未找到 {combo['symbol']} {combo['name']}，跳过")
                continue
            try:
                model, feats, meta = load_predictor(model_dir)
            except Exception as e:
                print(f"   ❌ {combo['name']}: 加载失败 {e}")
                continue
            thr_str = f"{combo['prob_threshold']}" if combo.get("prob_threshold") is not None else f"{int((combo.get('prob_threshold_down') or 0)*100)}-{int((combo.get('prob_threshold_up') or 0)*100)}"
            sym_short = combo["symbol"].replace("/USDT", "")
            filter_str = "有动态" if combo["enable_dynamic"] else "无动态"
            try:
                res = run_backtest_for_pair(
                    combo["symbol"], TIMEFRAME, model, feats,
                    initial_capital=initial_cap, bet_ratio=ratio_005, test_days=year_days,
                    prob_threshold=combo.get("prob_threshold"),
                    prob_threshold_up=combo.get("prob_threshold_up"),
                    prob_threshold_down=combo.get("prob_threshold_down"),
                    train_cutoff_date=after_date, end_date=end_date,
                    enable_dynamic_bet_ratio=combo["enable_dynamic"],
                    order_price=order_price, use_fixed_bet=False, use_smart_bet=True,
                    calibrator=None, calibration_method=None,
                )
            except Exception as e:
                print(f"   ❌ {combo['name']} {filter_str}: {e}")
                continue
            if res.get("error"):
                continue
            _append_row(sym_short, res, models_dir.name, thr_str, filter_str, combo["name"], capital_label=smart_label, cal_label="—")
        # 2) 7 GRU（有1h4h 与 无1h4h 按 combo 指定）
        print("\n📊 GRU 7 组合（xrp_55, btc_55, eth_54, sol_52, eth_55_dyn, eth_55_no1h4h, btc_57_no1h4h）…")
        _actual_range_printed = False
        for g in SIM_12_GRU_COMBOS:
            asset = g["asset"]
            models_override = models_best_no1h4h if g.get("use_no1h4h") and models_best_no1h4h.exists() else None
            try:
                df = get_backtest_df_one_asset(
                    asset, test_days=year_days, device=device, data_src=data_src,
                    after_date=after_date, end_date=end_date,
                    models_best_override=models_override,
                )
            except Exception as e:
                print(f"   ❌ {g['combo_name']}: {e}")
                continue
            if len(df) < 50:
                continue
            if not _actual_range_printed and "date" in df.columns:
                dmin, dmax = df["date"].min(), df["date"].max()
                try:
                    delta = pd.Timestamp(dmax) - pd.Timestamp(dmin)
                    days_actual = int(delta.total_seconds() / 86400)
                except Exception:
                    days_actual = max(1, len(df) // 96)
                dmin_s = pd.Timestamp(dmin).strftime("%Y-%m-%d") if hasattr(pd.Timestamp(dmin), "strftime") else str(dmin)[:10]
                dmax_s = pd.Timestamp(dmax).strftime("%Y-%m-%d") if hasattr(pd.Timestamp(dmax), "strftime") else str(dmax)[:10]
                print(f"   📅 实际回测区间: {dmin_s} ~ {dmax_s}（约 {days_actual} 天）")
                _actual_range_printed = True
            symbol = asset_to_symbol(asset)
            thr_str = f"{g['threshold']:.2f}"
            filter_str = "有动态" if g["enable_dynamic"] else "无动态"
            res = run_trading_loop(
                df, initial_capital=initial_cap, bet_ratio=ratio_005, prob_threshold=g["threshold"],
                enable_dynamic_bet_ratio=g["enable_dynamic"], order_price=order_price, stop_loss_exit_price=None,
                use_fixed_bet=False, use_smart_bet=True,
                **_mtpd_kw,
            )
            _append_row(symbol, res, MODEL_LABEL, thr_str, filter_str, g["combo_name"], capital_label=smart_label, cal_label="无1h4h" if g.get("use_no1h4h") else "有1h4h")
        df_ranking = pd.DataFrame(rows)
        if not df_ranking.empty:
            df_ranking = df_ranking.sort_values("最终资金", ascending=False).reset_index(drop=True)
            df_ranking["排名"] = df_ranking.index + 1
            df_ranking = _to_ranking_format(df_ranking)
        if run_compare_high_vol and not df_ranking.empty:
            _run_high_vol_comparison_sim12(
                data_src=data_src, device=device, after_date=after_date, end_date=end_date,
                year_days=year_days, initial_cap=initial_cap, ratio_005=ratio_005,
                order_price=order_price, _mtpd_kw=_mtpd_kw, df_ranking=df_ranking,
                models_best_no1h4h=models_best_no1h4h,
            )
        return df_ranking

    # ========== 统一全部排名：10 组合（5 旧 + 5 GRU 有1h4h）+ 4 资产 GRU 无1h4h 0.53～0.59，同条件 365 天、无止损 ==========
    if unified_all:
        from scripts.backtest_simulation import run_backtest_for_pair
        from src.python.predictor import load_predictor, _find_symbol_timeframe_model
        cap = 400.0
        ratio = 0.05
        use_fixed = False
        use_smart = True
        models_best_no1h4h = PROJECT_ROOT / "experiments" / "gru_regime_v1" / "outputs" / "models_best_no1h4h"
        thr_range = THRESHOLD_RANGE
        smart_label = "初始400、>6万固定3000/≤6万5%"
        print("\n📊 统一全部排名：10 组合 + 4 资产 GRU 无1h4h（0.53～0.59），365 天、无止损…")
        # 1) 5 旧模型：每组合 有动态/无动态 × 带校准/不带校准，保留更优一条
        print("\n📊 旧模型 5 组合…")
        for combo in OLD_POLYMARKET_COMBOS:
            md = combo["models_dir"]
            models_dir = (PROJECT_ROOT / md) if not Path(md).is_absolute() else Path(md)
            model_dir = _find_symbol_timeframe_model(combo["symbol"], TIMEFRAME, models_root=models_dir)
            if not model_dir:
                print(f"   ⚠️ 未找到 {combo['symbol']} {combo['name']}，跳过")
                continue
            try:
                model, feats, meta = load_predictor(model_dir)
            except Exception as e:
                print(f"   ❌ {combo['name']}: 加载失败 {e}")
                continue
            thr_str = f"{combo['prob_threshold']}" if combo.get("prob_threshold") is not None else f"{int((combo.get('prob_threshold_down') or 0)*100)}-{int((combo.get('prob_threshold_up') or 0)*100)}"
            sym_short = combo["symbol"].replace("/USDT", "")
            for enable_dyn in [True, False]:
                filter_str = "有动态" if enable_dyn else "无动态"
                for cal_label, cal, cal_m in [("带校准", meta.get("calibrator"), meta.get("calibration")), ("不带校准", None, None)]:
                    try:
                        res = run_backtest_for_pair(
                            combo["symbol"], TIMEFRAME, model, feats,
                            initial_capital=cap, bet_ratio=ratio, test_days=year_days,
                            prob_threshold=combo.get("prob_threshold"),
                            prob_threshold_up=combo.get("prob_threshold_up"),
                            prob_threshold_down=combo.get("prob_threshold_down"),
                            train_cutoff_date=after_date, end_date=end_date,
                            enable_dynamic_bet_ratio=enable_dyn,
                            order_price=order_price, use_fixed_bet=False, use_smart_bet=True,
                            calibrator=cal, calibration_method=cal_m,
                        )
                    except Exception as e:
                        continue
                    if res.get("error"):
                        continue
                    _append_row(sym_short, res, models_dir.name, thr_str, filter_str, combo["name"], capital_label=smart_label, cal_label=cal_label, stop_label="—")
        # 2) 5 GRU 有1h4h 组合（默认 MODELS_BEST）
        print("\n📊 GRU 有1h4h 五组合…")
        _actual_range_printed = False
        for g in GRU_SIM_COMBOS:
            asset = g["asset"]
            try:
                df = get_backtest_df_one_asset(
                    asset, test_days=year_days, device=device, data_src=data_src,
                    after_date=after_date, end_date=end_date,
                )
            except Exception as e:
                print(f"   ⚠️ {g['combo_name']}: {e}")
                continue
            if len(df) < 50:
                continue
            if not _actual_range_printed and "date" in df.columns:
                dmin, dmax = df["date"].min(), df["date"].max()
                try:
                    delta = pd.Timestamp(dmax) - pd.Timestamp(dmin)
                    days_actual = int(delta.total_seconds() / 86400)
                except Exception:
                    days_actual = max(1, len(df) // 96)
                dmin_s = pd.Timestamp(dmin).strftime("%Y-%m-%d") if hasattr(pd.Timestamp(dmin), "strftime") else str(dmin)[:10]
                dmax_s = pd.Timestamp(dmax).strftime("%Y-%m-%d") if hasattr(pd.Timestamp(dmax), "strftime") else str(dmax)[:10]
                print(f"   📅 实际回测区间: {dmin_s} ~ {dmax_s}（约 {days_actual} 天）")
                _actual_range_printed = True
            symbol = asset_to_symbol(asset)
            thr_str = f"{g['threshold']:.2f}"
            for enable_dyn in [True, False]:
                filter_str = "有动态" if enable_dyn else "无动态"
                res = run_trading_loop(
                    df, initial_capital=cap, bet_ratio=ratio, prob_threshold=g["threshold"],
                    enable_dynamic_bet_ratio=enable_dyn, order_price=order_price, stop_loss_exit_price=None, use_fixed_bet=False, use_smart_bet=True,
                    **_mtpd_kw,
                )
                _append_row(symbol, res, MODEL_LABEL, thr_str, filter_str, g["combo_name"], capital_label=smart_label, cal_label="有1h4h", stop_label="—")
        # 3) 4 资产 GRU 无1h4h：0.53～0.59（含两端）× 无动态+有动态
        if models_best_no1h4h.exists():
            print("\n📊 GRU 无1h4h 四资产（0.53～0.59）…")
            for asset in ASSETS:
                try:
                    df = get_backtest_df_one_asset(
                        asset, test_days=year_days, device=device, data_src=data_src,
                        after_date=after_date, end_date=end_date, models_best_override=models_best_no1h4h,
                    )
                except Exception as e:
                    print(f"   ⚠️ {asset} 无1h4h: {e}")
                    continue
                if len(df) < 50:
                    continue
                symbol = asset_to_symbol(asset)
                for thr in THRESHOLD_RANGE_NO1H4H:
                    for enable_dyn, filter_str in [(False, "无动态"), (True, "有动态")]:
                        res = run_trading_loop(
                            df, initial_capital=cap, bet_ratio=ratio, prob_threshold=thr,
                            enable_dynamic_bet_ratio=enable_dyn, order_price=order_price, stop_loss_exit_price=None,
                            use_fixed_bet=use_fixed, use_smart_bet=use_smart,
                            **_mtpd_kw,
                        )
                        combo_name = f"gru_{symbol.lower()}_{thr:.2f}_{filter_str}_无1h4h"
                        _append_row(symbol, res, MODEL_LABEL, f"{thr:.2f}", filter_str, combo_name, capital_label=smart_label, cal_label="无1h4h", stop_label="—")
        else:
            print("\n   ⚠️ models_best_no1h4h 不存在，跳过 GRU 无1h4h 四资产")
        # 统一：旧模型每(组合,策略过滤)保留带/不带校准更优一条
        df_ranking = pd.DataFrame(rows)
        if not df_ranking.empty:
            kept = []
            for (combo_key, filter_key), g in df_ranking.groupby(["组合", "策略过滤"]):
                if len(g) <= 1:
                    kept.append(g)
                    continue
                g = g.sort_values("最终资金", ascending=False)
                best_cap = g["最终资金"].iloc[0]
                candidates = g[g["最终资金"] == best_cap]
                no_cal = candidates[candidates["校准"] == "不带校准"]
                if not no_cal.empty:
                    kept.append(no_cal.head(1))
                else:
                    kept.append(candidates.head(1))
            df_ranking = pd.concat(kept, ignore_index=True)
            df_ranking = df_ranking.sort_values("最终资金", ascending=False).reset_index(drop=True)
            df_ranking["排名"] = df_ranking.index + 1
            df_ranking = _to_ranking_format(df_ranking)
        return df_ranking

    # ========== 统一 ETH 排名：有1h4h + 无1h4h 同条件 365 天，无止损，统一按最终资金排序 ==========
    if unified_eth and asset_only and threshold_range is not None:
        cap = 400.0
        ratio = 0.05
        use_fixed = False
        use_smart = True
        models_best_no1h4h = PROJECT_ROOT / "experiments" / "gru_regime_v1" / "outputs" / "models_best_no1h4h"
        thr_range = threshold_range
        symbol = asset_to_symbol(asset_only)
        print("\n📊 统一 ETH 排名：有1h4h 与 无1h4h 同条件 365 天（初始400、≥6万固定3000、<6万5%%，无止损）…")
        _actual_range_printed = False
        for cal_label, models_override in [("有1h4h", None), ("无1h4h", models_best_no1h4h)]:
            if models_override is not None and not models_best_no1h4h.exists():
                print(f"   ⚠️ 跳过 {cal_label}：{models_best_no1h4h} 不存在")
                continue
            try:
                df = get_backtest_df_one_asset(
                    asset_only,
                    test_days=year_days,
                    device=device,
                    data_src=data_src,
                    after_date=after_date,
                    end_date=end_date,
                    models_best_override=models_override,
                )
            except Exception as e:
                print(f"   ❌ {cal_label}: {e}")
                continue
            if len(df) < 50:
                continue
            if not _actual_range_printed and "date" in df.columns:
                dmin, dmax = df["date"].min(), df["date"].max()
                try:
                    delta = pd.Timestamp(dmax) - pd.Timestamp(dmin)
                    days_actual = int(delta.total_seconds() / 86400)
                except Exception:
                    days_actual = max(1, len(df) // 96)
                dmin_s = pd.Timestamp(dmin).strftime("%Y-%m-%d") if hasattr(pd.Timestamp(dmin), "strftime") else str(dmin)[:10]
                dmax_s = pd.Timestamp(dmax).strftime("%Y-%m-%d") if hasattr(pd.Timestamp(dmax), "strftime") else str(dmax)[:10]
                print(f"   📅 实际回测区间: {dmin_s} ~ {dmax_s}（约 {days_actual} 天，{len(df)} 根K线）")
                _actual_range_printed = True
            for thr in thr_range:
                for enable_dyn, filter_str in [(False, "无动态"), (True, "有动态")]:
                    res = run_trading_loop(
                        df,
                        initial_capital=cap,
                        bet_ratio=ratio,
                        prob_threshold=thr,
                        enable_dynamic_bet_ratio=enable_dyn,
                        order_price=order_price,
                        stop_loss_exit_price=None,
                        use_fixed_bet=use_fixed,
                        use_smart_bet=use_smart,
                        **_mtpd_kw,
                    )
                    combo_name = f"gru_eth_{thr:.2f}_{filter_str}_{cal_label}"
                    _append_row(symbol, res, MODEL_LABEL, f"{thr:.2f}", filter_str, combo_name, capital_label="", cal_label=cal_label, stop_label="—")
        df_ranking = pd.DataFrame(rows)
        if not df_ranking.empty:
            df_ranking = df_ranking.sort_values("最终资金", ascending=False).reset_index(drop=True)
            df_ranking["排名"] = df_ranking.index + 1
            df_ranking = _to_ranking_format(df_ranking)
        return df_ranking

    # SOL 等单资产模式：只跑该资产 GRU；若未指定 --with-old-combos 则不跑旧 5
    if asset_only and threshold_range is not None:
        assets_gru = [asset_only]
        thr_range = threshold_range
        run_old_combos = with_old_combos  # 单资产时默认不跑旧 5，传 --with-old-combos 则跑
    else:
        assets_gru = ASSETS
        thr_range = THRESHOLD_RANGE
        run_old_combos = True  # 多资产 / 默认：始终跑旧 5 组合

    # 1) 旧模型 5 个（仅当非单资产模式时跑；all-11 已在上面 return）
    from scripts.backtest_simulation import run_backtest_for_pair
    from src.python.predictor import load_predictor, _find_symbol_timeframe_model
    if run_old_combos:
        print("\n📊 旧模型 5 组合（logs_eth / logs_eth_10_90 / logs_btc / logs_xrp / logs_xrp_20_80）…")
    for combo in (OLD_POLYMARKET_COMBOS if run_old_combos else []):
        md = combo["models_dir"]
        models_dir = (PROJECT_ROOT / md) if not Path(md).is_absolute() else Path(md)
        model_dir = _find_symbol_timeframe_model(combo["symbol"], TIMEFRAME, models_root=models_dir)
        if not model_dir:
            print(f"   ⚠️ 未找到 {combo['symbol']} {combo['name']}，跳过")
            continue
        try:
            model, feats, meta = load_predictor(model_dir)
        except Exception as e:
            print(f"   ❌ {combo['name']}: 加载失败 {e}")
            continue
        try:
            res = run_backtest_for_pair(
                combo["symbol"],
                TIMEFRAME,
                model,
                feats,
                initial_capital=initial_capital,
                bet_ratio=bet_ratio,
                test_days=year_days,
                prob_threshold=combo.get("prob_threshold"),
                prob_threshold_up=combo.get("prob_threshold_up"),
                prob_threshold_down=combo.get("prob_threshold_down"),
                train_cutoff_date=after_date,
                end_date=end_date,
                enable_dynamic_bet_ratio=combo["enable_dynamic"],
                order_price=order_price,
                use_fixed_bet=True,
            )
        except Exception as e:
            print(f"   ❌ {combo['name']}: {e}")
            continue
        if res.get("error"):
            continue
        thr_str = f"{combo['prob_threshold']}" if combo.get("prob_threshold") is not None else f"{int((combo.get('prob_threshold_down') or 0)*100)}-{int((combo.get('prob_threshold_up') or 0)*100)}"
        filter_str = "有动态" if combo["enable_dynamic"] else "无动态"
        _append_row(combo["symbol"].replace("/USDT", ""), res, models_dir.name, thr_str, filter_str, combo["name"], capital_label="")

    # 单资产 + capital_400_smart：初始400、≥6万固定3000、<6万5%
    cap = 400.0 if (asset_only and capital_400_smart) else initial_capital
    ratio = bet_ratio
    use_fixed = not (asset_only and capital_400_smart)
    use_smart = bool(asset_only and capital_400_smart)

    # 2) 新模型 GRU：无动态（资产 × 阈值）
    print(f"\n📊 新模型 GRU 无动态（{len(assets_gru)} 资产 × 阈值 {thr_range[0]:.2f}～{thr_range[-1]:.2f}）…")
    _actual_range_printed = False
    for asset in assets_gru:
        try:
            df = get_backtest_df_one_asset(
                asset,
                test_days=year_days,
                device=device,
                data_src=data_src,
                after_date=after_date,
                end_date=end_date,
            )
        except Exception as e:
            print(f"   ❌ {asset}: {e}")
            continue
        if len(df) < 50:
            continue
        if not _actual_range_printed and "date" in df.columns:
            dmin, dmax = df["date"].min(), df["date"].max()
            try:
                delta = pd.Timestamp(dmax) - pd.Timestamp(dmin)
                days_actual = int(delta.total_seconds() / 86400)
            except Exception:
                days_actual = max(1, len(df) // 96)
            dmin_s = pd.Timestamp(dmin).strftime("%Y-%m-%d") if hasattr(pd.Timestamp(dmin), "strftime") else str(dmin)[:10]
            dmax_s = pd.Timestamp(dmax).strftime("%Y-%m-%d") if hasattr(pd.Timestamp(dmax), "strftime") else str(dmax)[:10]
            print(f"   📅 实际回测区间: {dmin_s} ~ {dmax_s}（约 {days_actual} 天，{len(df)} 根K线）")
            if days_actual < year_days * 0.95:
                print(f"   ⚠️ 数据不足 {year_days} 天（parquet 仅到近期），改 365 天与 90 天会得到相同结果；补全数据后再跑 365 天可见差异。")
            _actual_range_printed = True
        symbol = asset_to_symbol(asset)
        for thr in thr_range:
            combo_name = f"{MODEL_LABEL}_{thr:.2f}_无动态"
            if compare_stop_loss and stop_loss_exit_price is not None:
                for sl_price, sl_label in [(stop_loss_exit_price, "有止损"), (None, "无止损")]:
                    res = run_trading_loop(
                        df,
                        initial_capital=cap,
                        bet_ratio=ratio,
                        prob_threshold=thr,
                        enable_dynamic_bet_ratio=False,
                        order_price=order_price,
                        stop_loss_exit_price=sl_price,
                        use_fixed_bet=use_fixed,
                        use_smart_bet=use_smart,
                        **_mtpd_kw,
                    )
                    _append_row(symbol, res, MODEL_LABEL, f"{thr:.2f}", "无动态", combo_name, stop_label=sl_label)
            else:
                res = run_trading_loop(
                    df,
                    initial_capital=cap,
                    bet_ratio=ratio,
                    prob_threshold=thr,
                    enable_dynamic_bet_ratio=False,
                    order_price=order_price,
                    stop_loss_exit_price=stop_loss_exit_price,
                    use_fixed_bet=use_fixed,
                    use_smart_bet=use_smart,
                    **_mtpd_kw,
                )
                _append_row(symbol, res, MODEL_LABEL, f"{thr:.2f}", "无动态", combo_name)

    # 3) 新模型 GRU：有动态（资产 × 阈值）
    print(f"📊 新模型 GRU 有动态（{len(assets_gru)} 资产 × 阈值 {thr_range[0]:.2f}～{thr_range[-1]:.2f}）…")
    for asset in assets_gru:
        try:
            df = get_backtest_df_one_asset(
                asset,
                test_days=year_days,
                device=device,
                data_src=data_src,
                after_date=after_date,
                end_date=end_date,
            )
        except Exception as e:
            continue
        if len(df) < 50:
            continue
        symbol = asset_to_symbol(asset)
        for thr in thr_range:
            combo_name = f"{MODEL_LABEL}_{thr:.2f}_有动态"
            if compare_stop_loss and stop_loss_exit_price is not None:
                for sl_price, sl_label in [(stop_loss_exit_price, "有止损"), (None, "无止损")]:
                    res = run_trading_loop(
                        df,
                        initial_capital=cap,
                        bet_ratio=ratio,
                        prob_threshold=thr,
                        enable_dynamic_bet_ratio=True,
                        order_price=order_price,
                        stop_loss_exit_price=sl_price,
                        use_fixed_bet=use_fixed,
                        use_smart_bet=use_smart,
                        **_mtpd_kw,
                    )
                    _append_row(symbol, res, MODEL_LABEL, f"{thr:.2f}", "有动态", combo_name, stop_label=sl_label)
            else:
                res = run_trading_loop(
                    df,
                    initial_capital=cap,
                    bet_ratio=ratio,
                    prob_threshold=thr,
                    enable_dynamic_bet_ratio=True,
                    order_price=order_price,
                    stop_loss_exit_price=stop_loss_exit_price,
                    use_fixed_bet=use_fixed,
                    use_smart_bet=use_smart,
                    **_mtpd_kw,
                )
                _append_row(symbol, res, MODEL_LABEL, f"{thr:.2f}", "有动态", combo_name)

    # 4) 单资产时可选：再跑一份 初始400、每笔5%资金比例、无动态+有动态
    if asset_only and add_400_5pct_profile:
        cap_400, ratio_005 = 400.0, 0.05
        print(f"\n📊 新模型 GRU 资金设定 400/5%比例（{len(assets_gru)} 资产 × 阈值 × 无动态+有动态）…")
        for asset in assets_gru:
            try:
                df = get_backtest_df_one_asset(
                    asset,
                    test_days=year_days,
                    device=device,
                    data_src=data_src,
                    after_date=after_date,
                    end_date=end_date,
                )
            except Exception as e:
                continue
            if len(df) < 50:
                continue
            symbol = asset_to_symbol(asset)
            for thr in thr_range:
                for enable_dyn, filter_str, combo_suffix in [(False, "无动态", "无动态"), (True, "有动态", "有动态")]:
                    res = run_trading_loop(
                        df,
                        initial_capital=cap_400,
                        bet_ratio=ratio_005,
                        prob_threshold=thr,
                        enable_dynamic_bet_ratio=enable_dyn,
                        order_price=order_price,
                        stop_loss_exit_price=stop_loss_exit_price,
                        use_fixed_bet=False,
                        **_mtpd_kw,
                    )
                    combo_name = f"{MODEL_LABEL}_{thr:.2f}_{combo_suffix}_400/5%"
                    _append_row(symbol, res, MODEL_LABEL, f"{thr:.2f}", filter_str, combo_name, capital_label="400/5%比例")

    df_ranking = pd.DataFrame(rows)
    if not df_ranking.empty:
        df_ranking = df_ranking.sort_values("最终资金", ascending=False).reset_index(drop=True)
        df_ranking["排名"] = df_ranking.index + 1
        df_ranking = _to_ranking_format(df_ranking)
    return df_ranking


def main():
    ap = argparse.ArgumentParser(
        description="GRU Regime 未泄露一年回测报告，输出与旧版一致的 CSV（币种/周期/交易次数/胜率/最终资金/盈亏/回撤/模型/阈值/策略过滤/组合）"
    )
    ap.add_argument("--after-date", type=str, default=None, help="防泄露：仅用该日期之后 365 天的数据，格式 YYYY-MM-DD；不填则需用 --auto-after-date")
    ap.add_argument("--auto-after-date", action="store_true", help="从 config+parquet 按训练时 LightGBM 时序划分自动算出无泄露起始日（测试集起点），再取该日起 365 天")
    ap.add_argument("--initial-capital", type=float, default=60000.0, help="初始资金，默认 60000")
    ap.add_argument("--bet-ratio", type=float, default=0.05, help="下注比例，默认 0.05")
    ap.add_argument("--order-price", type=float, default=0.503, help="固定下单价格，默认 0.503")
    ap.add_argument("--stop-loss-exit-price", type=float, default=None, help="败局时按该价卖出 UP 回收，如 0.07；不传则败局全损")
    ap.add_argument("--compare-stop-loss", action="store_true", help="单资产回测时同时跑有止损+无止损，CSV 增加列 有无止损，便于对比最终金额")
    ap.add_argument("--data-src", type=str, default=None, help="数据目录，默认 data/raw")
    ap.add_argument("--no-mps", action="store_true", help="禁用 MPS")
    ap.add_argument("--output", type=str, default=None, help="输出 CSV 路径，默认 data/reports/gru_regime_backtest_ranking.csv")
    ap.add_argument("--asset", type=str, default=None, help="仅回测该资产（如 SOL_USDT），阈值用 --threshold-start/end；不跑旧模型 4 组合")
    ap.add_argument("--threshold-start", type=float, default=0.51, help="阈值起始（与 --asset 配套），默认 0.51")
    ap.add_argument("--threshold-end", type=float, default=0.98, help="阈值结束（与 --asset 配套），默认 0.98")
    ap.add_argument("--threshold-step", type=float, default=0.01, help="阈值步长，默认 0.01")
    ap.add_argument("--add-400-5pct", action="store_true", help="单资产模式下再跑一份：初始400、每笔5%%资金比例、无动态+有动态，列 资金设定 区分")
    ap.add_argument("--capital-400-smart", action="store_true", help="单资产模式：初始400、资金≥6万固定3000、<6万按5%%（与 11 组合规则一致）")
    ap.add_argument("--all-11-combos", action="store_true", help="模拟盘 10 组合回测：5 旧 + 5 GRU，每(组合,动态)保留校准更优一条，共 20 排名，365 天，买入价 0.503")
    ap.add_argument("--sim-12-combos", action="store_true", help="模拟盘 12 组合回测：与 查看实时日志 1～12 一致（5 旧 + 7 GRU），每组合一行，按最终资金降序排名，输出 gru_regime_backtest_ranking_sim12.csv")
    ap.add_argument("--compare-high-vol", action="store_true", help="与 --sim-12-combos 同用：对 7 个 GRU 组合额外跑「有高波动过滤」回测并打印无过滤 vs 有过滤对比表")
    ap.add_argument("--no-1h4h", action="store_true", help="使用无 1h/4h 特征的 GRU 模型：GRU_MODELS_BEST=models_best_no1h4h、GRU_SKIP_MTF=1，输出 CSV 为 gru_regime_backtest_ranking_11combo_no1h4h.csv")
    ap.add_argument("--unified-eth", action="store_true", help="ETH 有1h4h 与 无1h4h 同条件 365 天一起回测、统一排名（初始400、≥6万固定3000、<6万5%%），无止损")
    ap.add_argument("--unified-all", action="store_true", help="10 组合（5 旧 + 5 GRU 有1h4h）+ 4 资产 GRU 无1h4h（0.53～0.59）同条件 365 天、无止损，统一排名（已含旧 5 组合）")
    ap.add_argument("--with-old-combos", action="store_true", help="单资产模式（--asset）时也跑旧 5 组合，与 GRU 一起排名；其他模式（unified-all / all-11-combos）默认已含旧 5")
    ap.add_argument("--year-days", type=int, default=365, help="回测天数：从 after_date 起取多少天数据，默认 365；试 90 天可传 --year-days 90")
    ap.add_argument("--recent-days", type=int, default=None, help="11 组合专用：用最近 N 天数据回测（与模拟时段对齐），如 90；指定后忽略 after_date/year-days，用今日往前 N 天")
    ap.add_argument("--no-csv", action="store_true", help="11 组合模式下不生成 CSV，仅打印排名到控制台")
    ap.add_argument("--max-trades-per-day", type=int, default=None, help="每日最大交易笔数，0=不限制；不传则用 backtest_gru_regime 默认 20。与档位回测对比时建议传 0")
    args = ap.parse_args()

    data_src = Path(args.data_src) if args.data_src else PROJECT_ROOT / "data" / "raw"
    if not data_src.is_absolute():
        data_src = PROJECT_ROOT / data_src
    out_path = Path(args.output) if args.output else PROJECT_ROOT / "data" / "reports" / "gru_regime_backtest_ranking.csv"
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    after_date = args.after_date
    asset_for_date = (args.asset.strip().upper() if getattr(args, "asset", None) else None) or "BTC_USDT"
    if getattr(args, "auto_after_date", False):
        if after_date:
            print("⚠️ 同时指定了 --after-date 与 --auto-after-date，使用 --auto-after-date 计算的日期")
        after_date = get_no_leak_start_date(data_src, None, asset_for_date)
        print(f"  [--auto-after-date] 从 config+parquet 算出无泄露起始日: {after_date} (资产: {asset_for_date})")
    # --all-11-combos / --sim-12-combos 未指定起始日时自动用无泄露起始日
    if not after_date and (getattr(args, "all_11_combos", False) or getattr(args, "sim_12_combos", False)):
        after_date = get_no_leak_start_date(data_src, None, asset_for_date)
        label = "--sim-12-combos" if getattr(args, "sim_12_combos", False) else "--all-11-combos"
        print(f"  [{label}] 未指定起始日，自动用无泄露起始日: {after_date} (资产: {asset_for_date})")
    if not after_date:
        print("❌ 请指定 --after-date YYYY-MM-DD 或使用 --auto-after-date")
        sys.exit(1)

    initial_capital = args.initial_capital
    bet_ratio = args.bet_ratio
    order_price = args.order_price
    stop_loss_exit_price = getattr(args, "stop_loss_exit_price", None)
    compare_stop_loss = getattr(args, "compare_stop_loss", False)
    asset_only = getattr(args, "asset", None) and args.asset.strip().upper() or None
    if asset_only and "/" in asset_only:
        asset_only = asset_only.replace("/", "_")  # SOL/USDT -> SOL_USDT
    thr_start = getattr(args, "threshold_start", 0.51)
    thr_end = getattr(args, "threshold_end", 0.98)
    thr_step = getattr(args, "threshold_step", 0.01)
    if asset_only:
        threshold_range = [round(thr_start + i * thr_step, 2) for i in range(int((thr_end - thr_start) / thr_step) + 1)]
        threshold_range = [t for t in threshold_range if thr_start <= t <= thr_end]
    else:
        threshold_range = None
    add_400_5pct = getattr(args, "add_400_5pct", False)
    capital_400_smart = getattr(args, "capital_400_smart", False)
    run_all_11_combos = getattr(args, "all_11_combos", False)
    run_sim_12_combos = getattr(args, "sim_12_combos", False)
    run_compare_high_vol = getattr(args, "compare_high_vol", False)
    no_1h4h = getattr(args, "no_1h4h", False)
    unified_eth = getattr(args, "unified_eth", False)
    unified_all = getattr(args, "unified_all", False)
    with_old_combos = getattr(args, "with_old_combos", False)
    no_csv = getattr(args, "no_csv", False)
    year_days = getattr(args, "year_days", 365)
    recent_days = getattr(args, "recent_days", None)
    max_trades_per_day = getattr(args, "max_trades_per_day", None)

    if unified_all:
        capital_400_smart = True
        compare_stop_loss = False
        year_days = 365
        if order_price == 0.503:
            order_price = 0.5275  # 无1h4h 回测默认买单 0.5275
    if unified_eth:
        asset_only = asset_only or "ETH_USDT"
        capital_400_smart = True
        compare_stop_loss = False
        year_days = 365
        if not threshold_range:
            threshold_range = [round(0.51 + i * 0.01, 2) for i in range(48)]

    if run_all_11_combos and recent_days is not None and recent_days > 0:
        # 与模拟时段对齐：用最近 N 天数据回测（XRP 20-80 等双阈值在近期可能有交易）
        now_utc = pd.Timestamp.now(tz="UTC")
        after_date = (now_utc - pd.Timedelta(days=recent_days)).strftime("%Y-%m-%d")
        year_days = recent_days
        print(f"  [11组合 --recent-days] 使用最近 {recent_days} 天数据: {after_date} 起（与模拟时段对齐）")

    print("=" * 60)
    if run_sim_12_combos:
        print("  模拟盘 12 组合回测（与 查看实时日志 1～12 一致），按最终资金降序排名")
        print(f"  防泄露起始日: {after_date}  回测天数: {year_days}  买入价: {order_price}")
    elif run_all_11_combos:
        print("  10 组合回测（5 旧 + 5 GRU），每(组合,动态)保留校准更优一条，共 20 个排名")
        if no_1h4h:
            print("  模型: 无 1h/4h 特征 (models_best_no1h4h, GRU_SKIP_MTF=1)")
        print(f"  防泄露起始日: {after_date}  回测天数: {year_days}  买入价: {order_price}")
    elif unified_all:
        print("  统一全部排名：10 组合 + 4 资产 GRU 无1h4h（0.53～0.59），365 天、无止损")
        print(f"  防泄露起始日: {after_date}  初始400、≥6万固定3000、<6万5%%  买入价: {order_price}")
    elif unified_eth:
        print("  ETH 统一排名：有1h4h + 无1h4h 同条件 365 天，无止损")
        print(f"  防泄露起始日: {after_date}  初始400、≥6万固定3000、<6万5%%  买入价: {order_price}")
    elif asset_only:
        print(f"  GRU Regime 未泄露一年回测报告（仅 {asset_only}，阈值 {thr_start:.2f}～{thr_end:.2f}）")
    else:
        print("  GRU Regime 未泄露一年回测报告（多资产 × 阈值 0.51～0.98）")
    print("=" * 60)
    print(f"  防泄露起始日: {after_date}（仅用该日起 {year_days} 天数据）")
    print(f"  初始资金: {initial_capital}  下注比例: {bet_ratio}  下单价格: {order_price}" + (f"  败局止损价: {stop_loss_exit_price}" if stop_loss_exit_price is not None else ""))
    if max_trades_per_day is not None and max_trades_per_day == 0:
        print("  每日交易上限: 不限制（--max-trades-per-day 0）")
    print("  说明: 6万/固定3000/0.503；有动态=回撤>20%% 下注1500、>10%% 下注2250")
    if capital_400_smart and asset_only:
        print("  资金规则: 初始400、≥6万固定3000、<6万5%%（与 11 组合一致）")
    if compare_stop_loss and asset_only:
        print("  有无止损对比: 每个(阈值,动态)跑 有止损 + 无止损 两条，列 有无止损")
    if add_400_5pct:
        print("  另加: 400/5%%比例 无动态+有动态（列 资金设定 区分）")
    print(f"  周期: {TIMEFRAME}  模型: {MODEL_LABEL}")
    if asset_only:
        print(f"  单资产: {asset_only}  阈值数: {len(threshold_range) if threshold_range else 0}")
    print("=" * 60)

    try:
        df = run_report(
            after_date=after_date,
            initial_capital=initial_capital,
            bet_ratio=bet_ratio,
            data_src=data_src,
            no_mps=args.no_mps,
            year_days=year_days,
            order_price=order_price,
            stop_loss_exit_price=stop_loss_exit_price,
            asset_only=asset_only,
            threshold_range=threshold_range,
            add_400_5pct_profile=add_400_5pct,
            run_all_11_combos=run_all_11_combos,
            capital_400_smart=capital_400_smart,
            compare_stop_loss=compare_stop_loss,
            unified_eth=unified_eth,
            unified_all=unified_all,
            with_old_combos=with_old_combos,
            max_trades_per_day=max_trades_per_day,
            run_sim_12_combos=run_sim_12_combos,
            run_compare_high_vol=run_compare_high_vol,
        )
    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if df.empty:
        print("❌ 无有效记录")
        sys.exit(1)

    if run_sim_12_combos:
        out_path = out_path.parent / "gru_regime_backtest_ranking_sim12.csv"
    elif run_all_11_combos and not no_csv:
        out_path = out_path.parent / ("gru_regime_backtest_ranking_11combo_no1h4h.csv" if no_1h4h else "gru_regime_backtest_ranking_11combo.csv")
    elif run_all_11_combos and no_csv:
        out_path = None
    elif unified_all and out_path.name == "gru_regime_backtest_ranking.csv":
        out_path = out_path.parent / "gru_regime_backtest_ranking_unified_all.csv"
    elif unified_eth and asset_only and out_path.name == "gru_regime_backtest_ranking.csv":
        out_path = out_path.parent / "gru_regime_backtest_ranking_eth_unified.csv"
    elif asset_only and out_path.name == "gru_regime_backtest_ranking.csv":
        out_path = out_path.parent / f"gru_regime_backtest_ranking_{asset_only.lower()}.csv"

    def _write_ranked(csv_df: pd.DataFrame, path: Optional[Path]) -> None:
        out = csv_df.sort_values("最终资金", ascending=False).reset_index(drop=True)
        out["排名"] = out.index + 1
        out = out[[c for c in RANKING_COL_ORDER if c in out.columns]]
        if path is not None:
            out.to_csv(path, index=False, encoding="utf-8-sig")
            print(f"\n📄 CSV 已保存: {path}")
        print(f"   共 {len(out)} 条记录（按最终资金排序）")
        print("\n🏆 Top 10:")
        print(out.head(10).to_string(index=False))

    if run_sim_12_combos or run_all_11_combos:
        _write_ranked(df, out_path)
    elif add_400_5pct and "资金设定" in df.columns:
        df_6w = df[df["资金设定"] == "6万/固定3000"].copy()
        df_400 = df[df["资金设定"] == "400/5%比例"].copy()
        if not df_6w.empty:
            _write_ranked(_to_ranking_format(df_6w), out_path)
        if not df_400.empty:
            out_400 = out_path.parent / (out_path.stem + "_400_5pct" + out_path.suffix)
            _write_ranked(_to_ranking_format(df_400), out_400)
        if df_6w.empty and df_400.empty:
            _write_ranked(df, out_path)
    else:
        _write_ranked(df, out_path)


if __name__ == "__main__":
    main()
