#!/usr/bin/env python3
"""
基于胜率选择最优阈值：回测不同阈值，选择胜率最高且交易数足够的阈值

按《新模型优化》第③步：用胜率而不是 AUC 来选阈值。
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from scripts.backtest_simulation import run_backtest_for_pair
from src.python.predictor import load_predictor, _find_symbol_timeframe_model


def find_optimal_threshold(
    symbol: str,
    timeframe: str,
    models_dir: Path,
    threshold_range: List[float] = None,
    min_trades: int = 50,
    test_days: int = 90,
    initial_capital: float = 400.0,
    bet_ratio: float = 0.05,
) -> Dict:
    """
    回测不同阈值，找出胜率最高且交易数足够的阈值。
    
    Returns:
        {
            'best_threshold': float,
            'best_win_rate': float,
            'best_trades': int,
            'all_results': List[Dict]
        }
    """
    if threshold_range is None:
        # 默认：0.55 到 0.90，步长 0.01（共36个）
        threshold_range = [round(0.55 + i * 0.01, 2) for i in range(36)]
    
    # 查找模型
    model_dir = _find_symbol_timeframe_model(symbol, timeframe, models_root=models_dir)
    if not model_dir:
        raise ValueError(f"未找到 {symbol} {timeframe} 的模型")
    
    model, feats, meta = load_predictor(model_dir)
    
    print(f"📊 模型: {model_dir.name}")
    print(f"🔍 测试阈值范围: {threshold_range[0]:.2f} - {threshold_range[-1]:.2f} (共 {len(threshold_range)} 个)")
    print(f"📋 最小交易数: {min_trades}")
    print()
    
    results = []
    best_thr = None
    best_wr = 0.0
    best_trades = 0
    
    for thr in threshold_range:
        try:
            # 回测
            backtest_result = run_backtest_for_pair(
                symbol=symbol,
                timeframe=timeframe,
                model=model,
                feature_names=feats,
                initial_capital=initial_capital,
                bet_ratio=bet_ratio,
                test_days=test_days,
                prob_threshold=thr,
                train_cutoff_date=meta.get("train_cutoff_date"),
            )
            
            total_trades = backtest_result.get("total_trades", 0)
            wins = backtest_result.get("wins", 0)
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
            
            results.append({
                'threshold': thr,
                'trades': total_trades,
                'wins': wins,
                'losses': backtest_result.get("losses", 0),
                'win_rate': round(win_rate, 2),
                'profit_pct': backtest_result.get("profit_pct", 0),
                'final_capital': backtest_result.get("final_capital", initial_capital),
            })
            
            # 更新最佳阈值（胜率最高且交易数足够）
            if total_trades >= min_trades and win_rate > best_wr:
                best_thr = thr
                best_wr = win_rate
                best_trades = total_trades
            
            print(f"  阈值 {thr:.2f}: 交易 {total_trades} 笔, 胜率 {win_rate:.2f}%", end="")
            if thr == best_thr:
                print(" ⭐ 当前最佳")
            else:
                print()
                
        except Exception as e:
            print(f"  阈值 {thr:.2f}: 回测失败 - {e}")
            continue
    
    if best_thr is None:
        raise ValueError("未找到满足最小交易数的阈值")
    
    return {
        'best_threshold': best_thr,
        'best_win_rate': best_wr,
        'best_trades': best_trades,
        'all_results': results,
    }


def main():
    import argparse
    
    ap = argparse.ArgumentParser(description="基于胜率选择最优阈值")
    ap.add_argument("--symbol", required=True, help="交易对，如 ETH/USDT")
    ap.add_argument("--timeframe", default="15m", help="时间周期")
    ap.add_argument("--models-dir", type=str, default="data/models_C", help="模型目录")
    ap.add_argument("--min-trades", type=int, default=50, help="最小交易数要求")
    ap.add_argument("--test-days", type=int, default=90, help="回测天数")
    ap.add_argument("--threshold-start", type=float, default=0.55, help="阈值起始值")
    ap.add_argument("--threshold-end", type=float, default=0.90, help="阈值结束值")
    ap.add_argument("--threshold-step", type=float, default=0.01, help="阈值步长")
    
    args = ap.parse_args()
    
    models_dir = Path(args.models_dir)
    threshold_range = [round(args.threshold_start + i * args.threshold_step, 2) 
                      for i in range(int((args.threshold_end - args.threshold_start) / args.threshold_step) + 1)]
    
    try:
        result = find_optimal_threshold(
            args.symbol,
            args.timeframe,
            models_dir,
            threshold_range=threshold_range,
            min_trades=args.min_trades,
            test_days=args.test_days,
        )
        
        print()
        print("=" * 60)
        print("最优阈值选择结果")
        print("=" * 60)
        print(f"最佳阈值: {result['best_threshold']:.2f}")
        print(f"胜率: {result['best_win_rate']:.2f}%")
        print(f"交易数: {result['best_trades']} 笔")
        print()
        
        # 保存结果
        output_file = PROJECT_ROOT / "data" / "reports" / f"optimal_threshold_{args.symbol.replace('/', '_')}_{args.timeframe}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # 保存 CSV
        csv_file = output_file.with_suffix('.csv')
        pd.DataFrame(result['all_results']).to_csv(csv_file, index=False)
        
        print(f"📁 结果已保存:")
        print(f"   JSON: {output_file}")
        print(f"   CSV: {csv_file}")
        
    except Exception as e:
        print(f"❌ 失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
