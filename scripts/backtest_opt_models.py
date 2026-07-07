#!/usr/bin/env python3
"""
新模型优化回测对比脚本：使用双阈值区间和单阈值原始配置对比 models 和 models_C

回测配置：
- 模型目录：data/models, data/models_C
- 币对：BTC, ETH, XRP, SOL
- 时间周期：15m
- 双阈值区间：
  * DOWN 从 0.1 到 0.5（步长 0.1），UP 从 0.6 到 0.9（步长 0.1）
  * 共 5 × 4 = 20 个区间组合（例如：0.1～0.6, 0.1～0.7, 0.1～0.8, 0.1～0.9, 0.2～0.6, ...）
- 单阈值（原始置信度，只保留排名前几的组合）：
  * ETH models + 0.92
  * ETH models_C + 0.55
  * XRP models + 0.92
  * BTC models_C + 0.55
  * XRP models_C + 0.53
- 下注方式：固定下注比例 5%
- 排名方式：按最终资金从高到低排序
- 生成 Excel 排名表格（与 polymarket 报告格式一致）
"""

import json
import sys
import os
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np

# 抑制 scikit-learn 版本不匹配警告
warnings.filterwarnings('ignore', message='.*InconsistentVersionWarning.*')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.backtest_simulation import run_full_backtest, generate_summary_table

# 配置
MODEL_DIRS = ["data/models", "data/models_C"]
SYMBOLS = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT"]
TIMEFRAME = "15m"

# 双阈值区间配置：(prob_threshold_up, prob_threshold_down)
# UP: 仅当 prob >= prob_threshold_up 时做多
# DOWN: 仅当 prob < prob_threshold_down 时做空
# 中间区间：no_trade
# 扩展配置：DOWN 从 0.1 到 0.5（步长 0.1），UP 从 0.6 到 0.9（步长 0.1）
THRESHOLD_RANGES = []
for down_thr in [0.1, 0.2, 0.3, 0.4, 0.5]:  # DOWN 阈值：0.1～0.5
    for up_thr in [0.6, 0.7, 0.8, 0.9]:    # UP 阈值：0.6～0.9
        THRESHOLD_RANGES.append((up_thr, down_thr))

# 策略层过滤配置：(require_ema_trend, min_atr_pct, min_volume_ratio)
# require_ema_trend: True=需要EMA趋势确认，False=不需要
# min_atr_pct: ATR百分比最小值（None=不限制）
# min_volume_ratio: 成交量比率最小值（None=不限制）
STRATEGY_FILTER_CONFIGS = [
    (False, None, None),  # 无过滤（baseline）
    (True, 0.3, 0.8),     # EMA+ATR+Volume（全过滤）
]

# 单阈值配置（原始置信度阈值）：(models_dir, prob_threshold, symbol)
# 只保留回测结果中排名前几的组合（按最终资金排序）
# 
# 重要区别说明：
# - **超参数（Hyperparameters）**：模型训练时的参数（如 learning_rate, num_leaves, max_depth）
#   * 由 Optuna 优化，保存在 config/lightgbm_params_optuna.json
#   * 这些参数影响模型的训练过程和预测能力
#   * 所有模型（models, models_C）都使用相同的 Optuna 优化超参数
#
# - **交易阈值（Trading Threshold）**：交易决策时的置信度阈值
#   * 0.92 表示只有模型预测置信度≥92%时才交易
#   * 这些阈值不是超参数，而是交易策略参数
#   * 用于控制交易频率和胜率（阈值越高，交易越少，但胜率可能更高）
#
# 以下配置只保留回测结果中排名前几的组合（根据终端输出）：
# 1. ETH   models    92  无过滤
# 2. ETH models_C    55  无过滤
# 3. XRP   models    92  无过滤
# 4. BTC models_C    55  无过滤
# 5. XRP models_C    53  无过滤
# 6. ETH models_C 45-75  无过滤（双阈值，已在 THRESHOLD_RANGES 中）
SINGLE_THRESHOLD_CONFIGS = [
    ("data/models", 0.92, "ETH/USDT"),      # ETH models 92
    ("data/models_C", 0.55, "ETH/USDT"),    # ETH models_C 55
    ("data/models", 0.92, "XRP/USDT"),      # XRP models 92
    ("data/models_C", 0.55, "BTC/USDT"),    # BTC models_C 55
    ("data/models_C", 0.53, "XRP/USDT"),    # XRP models_C 53
]

# 不再使用止损配置


def run_backtest_for_model_threshold_range(
    models_dir: str,
    prob_threshold_up: Optional[float] = None,
    prob_threshold_down: Optional[float] = None,
    prob_threshold: Optional[float] = None,
    initial_capital: float = 400.0,
    bet_ratio: float = 0.05,
    test_days: int = 90,
    n_jobs: int = 1,
    require_ema_trend: bool = False,
    min_atr_pct: Optional[float] = None,
    min_volume_ratio: Optional[float] = None,
    target_symbols: Optional[List[str]] = None,  # 只回测指定的币对（用于单阈值配置）
) -> Dict[str, Any]:
    """为单个模型目录和阈值配置运行回测（支持单阈值或双阈值）"""
    try:
        models_path = Path(PROJECT_ROOT / models_dir)
        if not models_path.exists():
            return {
                "models_dir": models_dir,
                "prob_threshold_up": prob_threshold_up,
                "prob_threshold_down": prob_threshold_down,
            "prob_threshold": prob_threshold,
            "require_ema_trend": require_ema_trend,
            "min_atr_pct": min_atr_pct,
            "min_volume_ratio": min_volume_ratio,
            "error": f"模型目录不存在: {models_dir}",
                "results": [],
            }
        
        filter_desc = []
        if require_ema_trend:
            filter_desc.append("EMA")
        if min_atr_pct is not None:
            filter_desc.append(f"ATR≥{min_atr_pct*100:.0f}%")
        if min_volume_ratio is not None:
            filter_desc.append(f"Vol≥{min_volume_ratio}")
        filter_str = "+".join(filter_desc) if filter_desc else "无过滤"
        
        # 判断是单阈值还是双阈值
        use_dual = prob_threshold_up is not None and prob_threshold_down is not None
        if use_dual:
            threshold_desc = f"双阈值: UP≥{prob_threshold_up*100:.0f}%, DOWN<{prob_threshold_down*100:.0f}%"
        else:
            threshold_desc = f"单阈值: {prob_threshold*100:.0f}%"
        
        print(f"\n{'='*70}")
        print(f"📊 回测: {models_dir} | {threshold_desc} | 策略过滤: {filter_str}")
        print(f"{'='*70}")
        
        backtest_data = run_full_backtest(
            initial_capital=initial_capital,
            bet_ratio=bet_ratio,
            test_days=test_days,
            prob_threshold=prob_threshold if not use_dual else None,
            prob_threshold_up=prob_threshold_up if use_dual else None,
            prob_threshold_down=prob_threshold_down if use_dual else None,
            models_dir=models_path,
            n_jobs=n_jobs,
            timeframes=[TIMEFRAME],
            require_ema_trend=require_ema_trend,
            min_atr_pct=min_atr_pct,
            min_volume_ratio=min_volume_ratio,
            symbols=target_symbols,  # 只回测指定的币对（None 表示回测所有币对）
        )
        
        results = backtest_data.get("results", [])
        # 只保留目标币对的结果
        target_symbols_list = target_symbols if target_symbols is not None else SYMBOLS
        filtered_results = [
            r for r in results 
            if r.get("symbol") in target_symbols_list and r.get("timeframe") == TIMEFRAME
        ]
        
        # 计算汇总统计（符合 polymarket 计算规则）
        total_trades = sum(r.get("total_trades", 0) for r in filtered_results)
        total_wins = sum(r.get("wins", 0) for r in filtered_results)
        total_losses = sum(r.get("losses", 0) for r in filtered_results)
        
        # 胜率：总胜场 / 总交易次数 × 100%
        overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
        
        # 平均盈亏%：按交易次数加权的平均盈亏百分比
        total_profit_weighted = 0
        total_weight = 0
        for r in filtered_results:
            trades = r.get("total_trades", 0)
            if trades > 0:
                profit_pct = r.get("profit_pct", 0)  # 已经是百分比
                total_profit_weighted += profit_pct * trades
                total_weight += trades
        avg_profit_pct = (total_profit_weighted / total_weight) if total_weight > 0 else 0
        
        # 最终资金总和：所有币种的最终资金之和
        final_capital_sum = sum(r.get("final_capital", initial_capital) for r in filtered_results)
        
        # 最大回撤%：所有币种中的最大回撤（backtest_simulation.py 返回的已经是百分比）
        max_drawdown_values = [r.get("max_drawdown", 0) for r in filtered_results]
        max_drawdown = max(max_drawdown_values) if max_drawdown_values else 0
        # 确保在合理范围内（0-100%）
        max_drawdown = max(0, min(100, max_drawdown))
        
        return {
            "models_dir": models_dir,
            "prob_threshold_up": prob_threshold_up,
            "prob_threshold_down": prob_threshold_down,
            "prob_threshold": prob_threshold,
            "require_ema_trend": require_ema_trend,
            "min_atr_pct": min_atr_pct,
            "min_volume_ratio": min_volume_ratio,
            "results": filtered_results,
            "summary": {
                "total_trades": total_trades,
                "total_wins": total_wins,
                "total_losses": total_losses,
                "overall_win_rate": overall_win_rate,
                "avg_profit_pct": avg_profit_pct,
                "final_capital_sum": final_capital_sum,
                "max_drawdown": max_drawdown,
            },
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "models_dir": models_dir,
            "prob_threshold_up": prob_threshold_up,
            "prob_threshold_down": prob_threshold_down,
            "prob_threshold": prob_threshold,
            "require_ema_trend": require_ema_trend,
            "min_atr_pct": min_atr_pct,
            "min_volume_ratio": min_volume_ratio,
            "error": str(e),
            "results": [],
        }


def generate_ranking_excel(
    all_results: List[Dict[str, Any]],
    output_path: Path,
    initial_capital: float,
):
    """生成 Excel 排名表格（与 polymarket 报告格式一致）"""
    
    # 创建排名表（所有组合的所有币种，按最终资金从高到低排序）
    ranking_rows = []
    
    for result in all_results:
        if "error" in result:
            continue
        
        models_dir = result["models_dir"]
        prob_threshold_up = result.get("prob_threshold_up")
        prob_threshold_down = result.get("prob_threshold_down")
        prob_threshold = result.get("prob_threshold")
        require_ema_trend = result.get("require_ema_trend", False)
        min_atr_pct = result.get("min_atr_pct")
        min_volume_ratio = result.get("min_volume_ratio")
        results = result.get("results", [])
        
        # 构建策略过滤描述
        filter_desc = []
        if require_ema_trend:
            filter_desc.append("EMA")
        if min_atr_pct is not None:
            filter_desc.append(f"ATR{int(min_atr_pct*100)}")
        if min_volume_ratio is not None:
            filter_desc.append(f"Vol{int(min_volume_ratio*10)}")
        filter_str = "+".join(filter_desc) if filter_desc else "无过滤"
        
        # 构建阈值描述
        use_dual = prob_threshold_up is not None and prob_threshold_down is not None
        if use_dual:
            threshold_str = f"{int(prob_threshold_down*100)}-{int(prob_threshold_up*100)}"
        else:
            threshold_str = f"{int(prob_threshold*100)}"
        
        for r in results:
            symbol = r.get("symbol", "").replace("/USDT", "")
            
            # 确保数值在合理范围内
            max_dd = r.get("max_drawdown", 0)
            max_dd = max(0, min(100, max_dd))  # 回撤在 0-100% 之间
            
            min_over_init = r.get("min_over_initial_pct", 100)
            min_over_init = max(0, min(1000, min_over_init))  # 最低/初始%在 0-1000% 之间
            
            # 盈亏%：backtest_simulation.py 返回的已经是百分比
            profit_pct = r.get("profit_pct", 0)
            
            # 胜率%：backtest_simulation.py 返回的是小数（0-1），需要转换为百分比
            win_rate = r.get("win_rate", 0) * 100 if r.get("win_rate", 0) < 1 else r.get("win_rate", 0)
            
            # 最终资金
            final_capital = r.get("final_capital", initial_capital)
            
            ranking_rows.append({
                "排名": None,  # 稍后排序后填充
                "币种": symbol,
                "周期": r.get("timeframe", TIMEFRAME),
                "交易次数": r.get("total_trades", 0),
                "胜场": r.get("wins", 0),
                "败场": r.get("losses", 0),
                "胜率%": round(win_rate, 1),
                "最终资金": round(final_capital, 2),
                "盈亏%": round(profit_pct, 2),
                "最大回撤%": round(max_dd, 2),
                "最低/初始%": round(min_over_init, 2),
                "模型": Path(models_dir).name,
                "阈值": threshold_str,
                "策略过滤": filter_str,
                "组合": f"{Path(models_dir).name}_{threshold_str}_{filter_str}",
            })
    
    df_ranking = pd.DataFrame(ranking_rows)
    
    # 按最终资金从高到低排序并填充排名
    if not df_ranking.empty:
        df_ranking = df_ranking.sort_values("最终资金", ascending=False).reset_index(drop=True)
        df_ranking["排名"] = df_ranking.index + 1
    
    # 创建汇总表（每个模型+阈值区间的汇总）
    summary_rows = []
    for result in all_results:
        if "error" in result:
            continue
        
        summary = result.get("summary", {})
        models_dir = result["models_dir"]
        prob_threshold_up = result.get("prob_threshold_up")
        prob_threshold_down = result.get("prob_threshold_down")
        prob_threshold = result.get("prob_threshold")
        require_ema_trend = result.get("require_ema_trend", False)
        min_atr_pct = result.get("min_atr_pct")
        min_volume_ratio = result.get("min_volume_ratio")
        
        # 构建阈值描述
        use_dual = prob_threshold_up is not None and prob_threshold_down is not None
        if use_dual:
            threshold_str = f"{int(prob_threshold_down*100)}-{int(prob_threshold_up*100)}"
        else:
            threshold_str = f"{int(prob_threshold*100)}"
        
        # 构建策略过滤描述
        filter_desc = []
        if require_ema_trend:
            filter_desc.append("EMA")
        if min_atr_pct is not None:
            filter_desc.append(f"ATR{int(min_atr_pct*100)}")
        if min_volume_ratio is not None:
            filter_desc.append(f"Vol{int(min_volume_ratio*10)}")
        filter_str = "+".join(filter_desc) if filter_desc else "无过滤"
        
        summary_rows.append({
            "排名": None,
            "组合": f"{Path(models_dir).name}_{threshold_str}_{filter_str}",
            "模型": Path(models_dir).name,
            "阈值": threshold_str,
            "策略过滤": filter_str,
            "总交易次数": summary.get("total_trades", 0),
            "总胜场": summary.get("total_wins", 0),
            "总败场": summary.get("total_losses", 0),
            "总体胜率%": round(summary.get("overall_win_rate", 0), 1),
            "平均盈亏%": round(summary.get("avg_profit_pct", 0), 2),
            "最终资金总和": round(summary.get("final_capital_sum", initial_capital * len(SYMBOLS)), 2),
            "最大回撤%": round(summary.get("max_drawdown", 0), 2),
        })
    
    df_summary = pd.DataFrame(summary_rows)
    
    # 对汇总表按最终资金总和从高到低排序并填充排名
    if not df_summary.empty:
        df_summary = df_summary.sort_values("最终资金总和", ascending=False).reset_index(drop=True)
        df_summary["排名"] = df_summary.index + 1
    
    # 尝试写入 Excel，如果 openpyxl 不可用则使用 CSV
    try:
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 排名表（所有币种详细结果）
            if not df_ranking.empty:
                df_ranking.to_excel(writer, sheet_name='排名表', index=False)
            
            # 组合汇总表
            if not df_summary.empty:
                df_summary.to_excel(writer, sheet_name='组合汇总', index=False)
        
        print(f"\n📊 Excel 报告: {output_path}")
        print(f"   - 排名表: {len(df_ranking)} 条记录")
        print(f"   - 组合汇总: {len(df_summary)} 条记录")
    except ImportError:
        # openpyxl 不可用，使用 CSV
        print(f"\n⚠️  openpyxl 未安装，使用 CSV 格式")
        output_path = output_path.with_suffix('.csv')
    
    # 同时保存 CSV（无论是否成功写入 Excel）
    if not df_ranking.empty:
        csv_path = output_path.with_suffix('.ranking.csv')
        df_ranking.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"📄 CSV 排名表: {csv_path}")
    
    if not df_summary.empty:
        csv_summary_path = output_path.with_suffix('.summary.csv')
        df_summary.to_csv(csv_summary_path, index=False, encoding='utf-8-sig')
        print(f"📄 CSV 汇总表: {csv_summary_path}")
    
    # 打印 Top 10
    if not df_ranking.empty:
        print(f"\n🏆 Top 10 排名:")
        print(df_ranking.head(10)[["排名", "币种", "模型", "阈值", "策略过滤", "最终资金", "盈亏%", "胜率%", "交易次数"]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description="新模型优化回测对比：models vs models_C（双阈值区间）")
    ap.add_argument("--initial-capital", type=float, default=400.0, help="初始资金，默认 400")
    ap.add_argument("--bet-ratio", type=float, default=0.05, help="下注比例，默认 0.05 (5%%)")
    ap.add_argument("--test-days", type=int, default=90, help="回测天数，默认 90")
    ap.add_argument("--n-jobs", type=int, default=2, help="并行进程数，默认 2")
    ap.add_argument("--output", type=str, default=None, help="输出 Excel 文件路径")
    
    args = ap.parse_args()
    
    # 输出文件路径
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = PROJECT_ROOT / "data" / "reports" / f"backtest_opt_models_{timestamp}.xlsx"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("📊 新模型优化回测对比：models vs models_C（双阈值区间 + 单阈值原始配置）")
    print("=" * 70)
    print(f"模型目录: {MODEL_DIRS}")
    print(f"币对: {SYMBOLS}")
    print(f"时间周期: {TIMEFRAME}")
    print(f"双阈值区间:")
    for i, (up, down) in enumerate(THRESHOLD_RANGES, 1):
        print(f"  {i}. {down:.2f}～{up:.2f} (UP≥{up*100:.0f}%, DOWN<{down*100:.0f}%)")
    print(f"策略过滤:")
    for i, (ema, atr, vol) in enumerate(STRATEGY_FILTER_CONFIGS, 1):
        filter_desc = []
        if ema:
            filter_desc.append("EMA")
        if atr is not None:
            filter_desc.append(f"ATR≥{atr*100:.0f}%")
        if vol is not None:
            filter_desc.append(f"Vol≥{vol}")
        filter_str = "+".join(filter_desc) if filter_desc else "无过滤"
        print(f"  {i}. {filter_str}")
    print(f"单阈值（原始置信度，只保留排名前几的组合）:")
    for i, (models_dir, threshold, symbol) in enumerate(SINGLE_THRESHOLD_CONFIGS, 1):
        print(f"  {i}. {Path(models_dir).name} + {threshold*100:.0f}% ({symbol})")
    print(f"下注方式: 固定下注比例 {args.bet_ratio * 100}%")
    print(f"初始资金: ${args.initial_capital}")
    print(f"下注比例: {args.bet_ratio * 100}%")
    print(f"回测天数: {args.test_days}")
    print(f"并行进程: {args.n_jobs}")
    print(f"输出文件: {output_path}")
    print("=" * 70)
    print()
    
    # 准备任务列表
    tasks = []
    
    # 1. 双阈值区间任务
    for models_dir in MODEL_DIRS:
        for prob_threshold_up, prob_threshold_down in THRESHOLD_RANGES:
            for require_ema_trend, min_atr_pct, min_volume_ratio in STRATEGY_FILTER_CONFIGS:
                tasks.append((models_dir, prob_threshold_up, prob_threshold_down, None,
                            require_ema_trend, min_atr_pct, min_volume_ratio, None))  # None 表示回测所有币对
    
    # 2. 单阈值任务（原始置信度配置，只回测特定币对）
    for models_dir, prob_threshold, target_symbol in SINGLE_THRESHOLD_CONFIGS:
        # 单阈值只测试无过滤的情况（baseline）
        tasks.append((models_dir, None, None, prob_threshold,
                    False, None, None, [target_symbol]))  # 只回测特定币对
    
    total_tasks = len(tasks)
    # 计算总记录数：双阈值任务回测所有币对，单阈值任务只回测特定币对
    dual_threshold_tasks = len(MODEL_DIRS) * len(THRESHOLD_RANGES) * len(STRATEGY_FILTER_CONFIGS)
    single_threshold_tasks = len(SINGLE_THRESHOLD_CONFIGS)
    dual_threshold_records = dual_threshold_tasks * len(SYMBOLS)  # 每个任务回测所有币对
    single_threshold_records = single_threshold_tasks  # 每个任务只回测1个币对
    total_records = dual_threshold_records + single_threshold_records
    
    print(f"📋 回测任务数: {total_tasks}")
    print(f"   - 双阈值区间: {dual_threshold_tasks} 个 (2 个模型 × {len(THRESHOLD_RANGES)} 个区间 × {len(STRATEGY_FILTER_CONFIGS)} 个策略过滤)")
    print(f"   - 单阈值（原始置信度，只保留排名前几的组合）: {single_threshold_tasks} 个")
    print(f"📊 排名表格记录数: {total_records}")
    print(f"   - 双阈值: {dual_threshold_records} 条 ({dual_threshold_tasks} 个任务 × {len(SYMBOLS)} 个币对)")
    print(f"   - 单阈值: {single_threshold_records} 条 ({single_threshold_tasks} 个任务 × 1 个币对/任务)")
    print()
    
    # 执行回测（并行）
    all_results = []
    completed = 0
    
    if args.n_jobs > 1:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            futures = {
                executor.submit(
                    run_backtest_for_model_threshold_range,
                    models_dir, prob_threshold_up, prob_threshold_down, prob_threshold,
                    args.initial_capital, args.bet_ratio, args.test_days, 1,  # 子任务内不再并行
                    require_ema_trend, min_atr_pct, min_volume_ratio, target_symbols
                ): (models_dir, prob_threshold_up, prob_threshold_down, prob_threshold, require_ema_trend, min_atr_pct, min_volume_ratio, target_symbols)
                for models_dir, prob_threshold_up, prob_threshold_down, prob_threshold, require_ema_trend, min_atr_pct, min_volume_ratio, target_symbols in tasks
            }
            
            for future in as_completed(futures):
                models_dir, prob_threshold_up, prob_threshold_down, prob_threshold, require_ema_trend, min_atr_pct, min_volume_ratio, target_symbols = futures[future]
                completed += 1
                try:
                    result = future.result()
                    all_results.append(result)
                    if "error" in result:
                        filter_desc = "+".join([d for d in ["EMA" if require_ema_trend else None, 
                                                           f"ATR{int(min_atr_pct*100)}" if min_atr_pct else None,
                                                           f"Vol{int(min_volume_ratio*10)}" if min_volume_ratio else None] if d])
                        filter_str = filter_desc if filter_desc else "无过滤"
                        if prob_threshold is not None:
                            threshold_str = f"{int(prob_threshold*100)}"
                        else:
                            threshold_str = f"{int(prob_threshold_down*100)}-{int(prob_threshold_up*100)}"
                        print(f"[{completed}/{total_tasks}] ❌ {models_dir} {threshold_str} {filter_str}: {result['error']}")
                    else:
                        summary = result.get("summary", {})
                        filter_desc = "+".join([d for d in ["EMA" if require_ema_trend else None, 
                                                           f"ATR{int(min_atr_pct*100)}" if min_atr_pct else None,
                                                           f"Vol{int(min_volume_ratio*10)}" if min_volume_ratio else None] if d])
                        filter_str = filter_desc if filter_desc else "无过滤"
                        if prob_threshold is not None:
                            threshold_str = f"{int(prob_threshold*100)}"
                        else:
                            threshold_str = f"{int(prob_threshold_down*100)}-{int(prob_threshold_up*100)}"
                        print(f"[{completed}/{total_tasks}] ✅ {models_dir} {threshold_str} {filter_str}: "
                              f"最终资金 ${summary.get('final_capital_sum', 0):.0f}, "
                              f"胜率 {summary.get('overall_win_rate', 0):.1f}%, "
                              f"交易 {summary.get('total_trades', 0)} 笔")
                except Exception as e:
                    filter_desc = "+".join([d for d in ["EMA" if require_ema_trend else None, 
                                                       f"ATR{int(min_atr_pct*100)}" if min_atr_pct else None,
                                                       f"Vol{int(min_volume_ratio*10)}" if min_volume_ratio else None] if d])
                    filter_str = filter_desc if filter_desc else "无过滤"
                    if prob_threshold is not None:
                        threshold_str = f"{int(prob_threshold*100)}"
                    else:
                        threshold_str = f"{int(prob_threshold_down*100)}-{int(prob_threshold_up*100)}"
                    print(f"[{completed}/{total_tasks}] ❌ {models_dir} {threshold_str} {filter_str}: {e}")
                    all_results.append({
                        "models_dir": models_dir,
                        "prob_threshold_up": prob_threshold_up,
                        "prob_threshold_down": prob_threshold_down,
                        "prob_threshold": prob_threshold,
                        "require_ema_trend": require_ema_trend,
                        "min_atr_pct": min_atr_pct,
                        "min_volume_ratio": min_volume_ratio,
                        "error": str(e),
                        "results": [],
                    })
    else:
        # 串行执行
        for models_dir, prob_threshold_up, prob_threshold_down in tasks:
            completed += 1
            result = run_backtest_for_model_threshold_range(
                models_dir, prob_threshold_up, prob_threshold_down,
                args.initial_capital, args.bet_ratio, args.test_days, 1
            )
            all_results.append(result)
            if "error" in result:
                print(f"[{completed}/{total_tasks}] ❌ {models_dir} {int(prob_threshold_down*100)}-{int(prob_threshold_up*100)}: {result['error']}")
            else:
                summary = result.get("summary", {})
                print(f"[{completed}/{total_tasks}] ✅ {models_dir} {int(prob_threshold_down*100)}-{int(prob_threshold_up*100)}: "
                      f"胜率 {summary.get('overall_win_rate', 0):.1f}%, "
                      f"交易 {summary.get('total_trades', 0)} 笔")
    
    print()
    print("=" * 70)
    print("📊 生成排名表格...")
    print("=" * 70)
    
    # 生成 Excel 报告
    generate_ranking_excel(all_results, output_path, args.initial_capital)
    
    print()
    print("=" * 70)
    print("✅ 回测完成！")
    print("=" * 70)
    print(f"📁 报告位置: {output_path}")
    print()


if __name__ == "__main__":
    main()
