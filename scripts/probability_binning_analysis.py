#!/usr/bin/env python3
"""
概率分桶分析：在验证集/回测集上分析不同概率区间的胜率

按《新模型优化》第①步：在验证集上画概率分桶表，找出胜率明显抬升的区间。
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from src.python.data_fetcher import load_ohlcv
from src.python.feature_engineering import build_features, prepare_train_data
from src.python.predictor import load_predictor, predict_one


def analyze_probability_bins(
    symbol: str,
    timeframe: str,
    model_dir: Path,
    bins: List[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """
    在验证集上分析不同概率区间的胜率。
    
    Returns:
        DataFrame with columns: bin_start, bin_end, trades, wins, win_rate
    """
    if bins is None:
        # 默认分桶：0.50-0.55, 0.55-0.60, ..., 0.80-1.00
        bins = [(0.50 + i*0.05, 0.50 + (i+1)*0.05) for i in range(6)]
        bins[-1] = (0.80, 1.00)  # 最后一个到1.0
    
    # 加载模型
    model, feats, meta = load_predictor(model_dir)
    calibrator = meta.get("calibrator")
    calibration_method = meta.get("calibration")
    
    # 加载数据
    df = load_ohlcv(symbol, timeframe)
    if df.empty or len(df) < 500:
        raise ValueError(f"数据不足: {symbol} {timeframe}")
    
    # 构建特征
    df_features = build_features(df)
    
    # 准备标签（下一根K线涨跌）
    X, y = prepare_train_data(df_features, lookahead=1)
    
    # 使用 holdout 数据（与训练时一致）
    train_cutoff_date = meta.get("train_cutoff_date")
    if train_cutoff_date:
        if isinstance(train_cutoff_date, str):
            cutoff_ts = pd.Timestamp(train_cutoff_date)
        else:
            cutoff_ts = pd.Timestamp(train_cutoff_date)
        if pd.api.types.is_datetime64_any_dtype(df_features.index):
            df_val = df_features[df_features.index >= cutoff_ts]
        else:
            # 如果没有时间索引，用最后20%作为验证集
            n = len(df_features)
            df_val = df_features.iloc[int(n * 0.8):]
    else:
        # 默认用最后20%
        n = len(df_features)
        df_val = df_features.iloc[int(n * 0.8):]
    
    X_val, y_val = prepare_train_data(df_val, lookahead=1)
    
    if len(X_val) < 100:
        raise ValueError(f"验证集数据不足: {len(X_val)} 条")
    
    # 预测所有验证集样本
    probs = []
    for idx in range(len(X_val)):
        # 构建到当前时刻的历史数据
        hist_end = df_val.index[X_val.index[idx]]
        hist_df = df_features[df_features.index <= hist_end]
        
        if len(hist_df) < 50:
            continue
        
        try:
            pred, prob = predict_one(
                model, feats, hist_df,
                calibrator=calibrator,
                calibration_method=calibration_method,
            )
            probs.append({
                'idx': idx,
                'prob_up': prob,
                'actual': y_val.iloc[idx],
                'pred': pred,
            })
        except Exception as e:
            continue
    
    if len(probs) == 0:
        raise ValueError("无法生成预测")
    
    probs_df = pd.DataFrame(probs)
    
    # 分桶统计
    results = []
    for bin_start, bin_end in bins:
        if bin_end == 1.0:
            mask = (probs_df['prob_up'] >= bin_start) & (probs_df['prob_up'] <= bin_end)
        else:
            mask = (probs_df['prob_up'] >= bin_start) & (probs_df['prob_up'] < bin_end)
        
        trades = probs_df[mask]
        if len(trades) == 0:
            results.append({
                'bin_start': bin_start,
                'bin_end': bin_end,
                'trades': 0,
                'wins': 0,
                'win_rate': 0.0,
            })
            continue
        
        # 计算胜率（预测方向与实际方向一致）
        wins = (trades['pred'] == trades['actual']).sum()
        win_rate = wins / len(trades) * 100
        
        results.append({
            'bin_start': bin_start,
            'bin_end': bin_end,
            'trades': len(trades),
            'wins': wins,
            'win_rate': round(win_rate, 2),
        })
    
    return pd.DataFrame(results)


def main():
    import argparse
    
    ap = argparse.ArgumentParser(description="概率分桶分析")
    ap.add_argument("--symbol", required=True, help="交易对，如 ETH/USDT")
    ap.add_argument("--timeframe", default="15m", help="时间周期")
    ap.add_argument("--models-dir", type=str, default="data/models_C", help="模型目录")
    
    args = ap.parse_args()
    
    models_dir = Path(args.models_dir)
    symbol_key = args.symbol.replace("/", "_")
    
    # 查找模型
    model_dirs = [d for d in models_dir.iterdir() 
                  if d.is_dir() and symbol_key in d.name and f"_{args.timeframe}_" in d.name]
    if not model_dirs:
        print(f"❌ 未找到 {args.symbol} {args.timeframe} 的模型")
        sys.exit(1)
    
    model_dir = max(model_dirs, key=lambda d: d.stat().st_mtime)
    print(f"📊 使用模型: {model_dir.name}\n")
    
    # 分析
    try:
        results = analyze_probability_bins(args.symbol, args.timeframe, model_dir)
        
        print("=" * 60)
        print(f"概率分桶分析 - {args.symbol} {args.timeframe}")
        print("=" * 60)
        print()
        print(f"{'概率区间':<15} {'交易数':<10} {'胜率':<10}")
        print("-" * 60)
        for _, row in results.iterrows():
            bin_str = f"{row['bin_start']:.2f}-{row['bin_end']:.2f}"
            print(f"{bin_str:<15} {row['trades']:<10} {row['win_rate']:.2f}%")
        print()
        
        # 找出胜率明显抬升的区间
        high_wr = results[results['win_rate'] >= 60]
        if len(high_wr) > 0:
            print("✅ 高胜率区间（≥60%）：")
            for _, row in high_wr.iterrows():
                print(f"   {row['bin_start']:.2f}-{row['bin_end']:.2f}: {row['win_rate']:.2f}% ({row['trades']} 笔)")
            print()
        
        # 保存结果
        output_file = PROJECT_ROOT / "data" / "reports" / f"prob_binning_{symbol_key}_{args.timeframe}.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_file, index=False)
        print(f"📁 结果已保存: {output_file}")
        
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
