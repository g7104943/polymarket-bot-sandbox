#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交易数据分析脚本
分析旧模型、Exp8和Exp16的交易数据
"""

import json
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd

BASE_DIR = Path(__file__).parent

def load_trades(json_path):
    """加载交易数据"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return []

def analyze_old_model(model_dir):
    """分析旧模型数据"""
    json_path = BASE_DIR / model_dir / "prediction_trades.json"
    if not json_path.exists():
        return None
    
    trades = load_trades(json_path)
    if not trades:
        return None
    
    total_trades = len(trades)
    settled_trades = [t for t in trades if t.get('result') in ['win', 'lose']]
    settled_count = len(settled_trades)
    
    win_count = len([t for t in settled_trades if t.get('result') == 'win'])
    win_rate = win_count / settled_count if settled_count > 0 else 0
    
    total_pnl = sum(t.get('pnl', 0) for t in trades if t.get('pnl') is not None)
    
    executed_trades = [t for t in trades if t.get('status') == 'executed']
    avg_token_price = sum(t.get('tokenPrice', 0) for t in executed_trades if t.get('tokenPrice')) / len(executed_trades) if executed_trades else 0
    
    return {
        'model': model_dir,
        'total_trades': total_trades,
        'settled_count': settled_count,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_token_price': avg_token_price
    }

def analyze_exp_model(model_dir, exp_name):
    """分析Exp8或Exp16模型数据，按币种和买价分组"""
    json_path = BASE_DIR / model_dir / "prediction_trades.json"
    if not json_path.exists():
        return []
    
    trades = load_trades(json_path)
    if not trades:
        return []
    
    # 从目录名提取买价
    bp = None
    if 'bp_dyn' in model_dir:
        # 动态买价，提取范围
        if '0450_0530' in model_dir:
            bp = 'dyn_0450_0530'
        elif '0480_0510' in model_dir:
            bp = 'dyn_0480_0510'
    else:
        # 固定买价，提取数字
        bp_match = model_dir.split('bp')[1] if 'bp' in model_dir else None
        if bp_match:
            bp = float(bp_match) / 1000  # 例如 bp0450 -> 0.45
    
    results = []
    
    # 按币种分组
    by_symbol = defaultdict(list)
    for trade in trades:
        symbol = trade.get('symbol', 'UNKNOWN')
        by_symbol[symbol].append(trade)
    
    for symbol, symbol_trades in by_symbol.items():
        total_trades = len(symbol_trades)
        settled_trades = [t for t in symbol_trades if t.get('result') in ['win', 'lose']]
        settled_count = len(settled_trades)
        
        win_count = len([t for t in settled_trades if t.get('result') == 'win'])
        win_rate = win_count / settled_count if settled_count > 0 else 0
        
        total_pnl = sum(t.get('pnl', 0) for t in symbol_trades if t.get('pnl') is not None)
        
        executed_trades = [t for t in symbol_trades if t.get('status') == 'executed']
        avg_token_price = sum(t.get('tokenPrice', 0) for t in executed_trades if t.get('tokenPrice')) / len(executed_trades) if executed_trades else 0
        
        results.append({
            'exp': exp_name,
            'model': model_dir,
            'buy_price': bp,
            'symbol': symbol,
            'total_trades': total_trades,
            'settled_count': settled_count,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_token_price': avg_token_price
        })
    
    return results

def analyze_bp_vs_performance(exp_name, symbol_filter=None):
    """分析买价 vs 成交数量 vs PnL的关系"""
    # 找到所有相关目录
    exp_dirs = []
    for item in BASE_DIR.iterdir():
        if item.is_dir() and f'logs_v5_{exp_name}_bp' in item.name and 'archive' not in str(item):
            exp_dirs.append(item.name)
    
    exp_dirs.sort()
    
    results = []
    
    for model_dir in exp_dirs:
        json_path = BASE_DIR / model_dir / "prediction_trades.json"
        if not json_path.exists():
            continue
        
        trades = load_trades(json_path)
        if not trades:
            continue
        
        # 提取买价
        bp = None
        if 'bp_dyn' in model_dir:
            continue  # 跳过动态买价，只分析固定买价
        else:
            bp_match = model_dir.split('bp')[1] if 'bp' in model_dir else None
            if bp_match:
                bp = float(bp_match) / 1000
        
        if bp is None:
            continue
        
        # 按币种分组
        by_symbol = defaultdict(list)
        for trade in trades:
            trade_symbol = trade.get('symbol', 'UNKNOWN')
            if symbol_filter and trade_symbol != symbol_filter:
                continue
            by_symbol[trade_symbol].append(trade)
        
        for symbol, symbol_trades in by_symbol.items():
            if symbol_filter and symbol != symbol_filter:
                continue
            
            total_trades = len(symbol_trades)
            settled_trades = [t for t in symbol_trades if t.get('result') in ['win', 'lose']]
            settled_count = len(settled_trades)
            
            win_count = len([t for t in settled_trades if t.get('result') == 'win'])
            win_rate = win_count / settled_count if settled_count > 0 else 0
            
            total_pnl = sum(t.get('pnl', 0) for t in symbol_trades if t.get('pnl') is not None)
            
            executed_trades = [t for t in symbol_trades if t.get('status') == 'executed']
            avg_token_price = sum(t.get('tokenPrice', 0) for t in executed_trades if t.get('tokenPrice')) / len(executed_trades) if executed_trades else 0
            
            # 统一 exp 名称格式
            exp_display = 'Exp8' if 'exp8' in exp_name.lower() else 'Exp16' if 'exp16' in exp_name.lower() else exp_name
            
            results.append({
                'exp': exp_display,
                'model': model_dir,
                'buy_price': bp,
                'symbol': symbol,
                'total_trades': total_trades,
                'settled_count': settled_count,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_token_price': avg_token_price
            })
    
    return results

def main():
    print("=" * 80)
    print("交易数据分析报告")
    print("=" * 80)
    
    # 1. 分析所有旧模型
    print("\n【1. 旧模型成交情况统计】")
    print("-" * 80)
    
    old_models = []
    for item in BASE_DIR.iterdir():
        if item.is_dir() and ('logs_gru_' in item.name or 'logs_lgb_' in item.name):
            if 'archive' not in str(item):
                old_models.append(item.name)
    
    old_models.sort()
    
    old_model_results = []
    for model_dir in old_models:
        result = analyze_old_model(model_dir)
        if result:
            old_model_results.append(result)
    
    if old_model_results:
        df_old = pd.DataFrame(old_model_results)
        print(df_old.to_string(index=False))
        print(f"\n总计: {len(old_model_results)} 个旧模型")
        print(f"总交易笔数: {df_old['total_trades'].sum()}")
        print(f"总已结算笔数: {df_old['settled_count'].sum()}")
        print(f"总PnL: {df_old['total_pnl'].sum():.2f}")
    
    # 2. 分析Exp8和Exp16的所有买价组合（按币种）
    print("\n\n【2. Exp8 和 Exp16 买价组合统计（按币种）】")
    print("-" * 80)
    
    exp8_dirs = [d.name for d in BASE_DIR.iterdir() if d.is_dir() and 'logs_v5_exp8_' in d.name and 'archive' not in str(d)]
    exp16_dirs = [d.name for d in BASE_DIR.iterdir() if d.is_dir() and 'logs_v5_exp16_' in d.name and 'archive' not in str(d)]
    
    exp8_results = []
    for model_dir in sorted(exp8_dirs):
        results = analyze_exp_model(model_dir, 'Exp8')
        exp8_results.extend(results)
    
    exp16_results = []
    for model_dir in sorted(exp16_dirs):
        results = analyze_exp_model(model_dir, 'Exp16')
        exp16_results.extend(results)
    
    all_exp_results = exp8_results + exp16_results
    
    if all_exp_results:
        df_exp = pd.DataFrame(all_exp_results)
        # 只显示BTC和XRP
        df_exp_filtered = df_exp[df_exp['symbol'].isin(['BTC', 'XRP'])].copy()
        df_exp_filtered = df_exp_filtered.sort_values(['exp', 'symbol', 'buy_price'])
        print(df_exp_filtered.to_string(index=False))
    
    # 3. 分析买价 vs 成交数量 vs PnL的关系
    print("\n\n【3. 买价 vs 成交数量 vs PnL 关系分析】")
    print("-" * 80)
    
    for symbol in ['BTC', 'XRP']:
        print(f"\n【{symbol}】")
        print("-" * 80)
        
        exp8_bp_results = analyze_bp_vs_performance('exp8', symbol)
        exp16_bp_results = analyze_bp_vs_performance('exp16', symbol)
        
        all_bp_results = exp8_bp_results + exp16_bp_results
        
        if all_bp_results:
            df_bp = pd.DataFrame(all_bp_results)
            df_bp = df_bp.sort_values(['exp', 'buy_price'])
            
            # 按实验分组显示
            for exp in ['Exp8', 'Exp16']:
                df_exp_bp = df_bp[df_bp['exp'] == exp].copy()
                if not df_exp_bp.empty:
                    print(f"\n{exp}:")
                    print(df_exp_bp[['buy_price', 'total_trades', 'settled_count', 'win_rate', 'total_pnl', 'avg_token_price']].to_string(index=False))
                    
                    # 找出最优买价
                    if len(df_exp_bp) > 0:
                        best_bp = df_exp_bp.loc[df_exp_bp['total_pnl'].idxmax()]
                        print(f"\n最优买价（PnL最大化）: {best_bp['buy_price']:.3f}")
                        print(f"  - 交易笔数: {best_bp['total_trades']}")
                        print(f"  - 胜率: {best_bp['win_rate']:.2%}")
                        print(f"  - 总PnL: {best_bp['total_pnl']:.2f}")
                        print(f"  - 平均成交价: {best_bp['avg_token_price']:.4f}")
            
            # 合并两个实验的数据，按买价汇总
            print(f"\n【{symbol} - Exp8 + Exp16 合并统计】")
            
            # 重新计算合并后的数据，确保胜率正确
            combined_data = []
            for bp in sorted(df_bp['buy_price'].unique()):
                bp_data = df_bp[df_bp['buy_price'] == bp]
                
                # 重新加载所有相关交易来计算准确的胜率
                total_wins = 0
                total_settled = 0
                total_trades_sum = 0
                total_pnl_sum = 0
                token_prices = []
                
                for _, row in bp_data.iterrows():
                    model_dir = row['model']
                    json_path = BASE_DIR / model_dir / "prediction_trades.json"
                    trades = load_trades(json_path)
                    
                    symbol_trades = [t for t in trades if t.get('symbol') == symbol]
                    total_trades_sum += len(symbol_trades)
                    
                    settled = [t for t in symbol_trades if t.get('result') in ['win', 'lose']]
                    total_settled += len(settled)
                    total_wins += len([t for t in settled if t.get('result') == 'win'])
                    
                    total_pnl_sum += sum(t.get('pnl', 0) for t in symbol_trades if t.get('pnl') is not None)
                    
                    executed = [t for t in symbol_trades if t.get('status') == 'executed']
                    token_prices.extend([t.get('tokenPrice') for t in executed if t.get('tokenPrice')])
                
                win_rate = total_wins / total_settled if total_settled > 0 else 0
                avg_token_price = sum(token_prices) / len(token_prices) if token_prices else 0
                
                combined_data.append({
                    'buy_price': bp,
                    'total_trades': total_trades_sum,
                    'settled_count': total_settled,
                    'win_rate': win_rate,
                    'total_pnl': total_pnl_sum,
                    'avg_token_price': avg_token_price
                })
            
            df_combined = pd.DataFrame(combined_data)
            
            df_combined = df_combined.sort_values('buy_price')
            print(df_combined[['buy_price', 'total_trades', 'settled_count', 'win_rate', 'total_pnl', 'avg_token_price']].to_string(index=False))
            
            # 找出合并后的最优买价
            if len(df_combined) > 0:
                best_bp_combined = df_combined.loc[df_combined['total_pnl'].idxmax()]
                print(f"\n合并后最优买价（PnL最大化）: {best_bp_combined['buy_price']:.3f}")
                print(f"  - 交易笔数: {best_bp_combined['total_trades']}")
                print(f"  - 胜率: {best_bp_combined['win_rate']:.2%}")
                print(f"  - 总PnL: {best_bp_combined['total_pnl']:.2f}")
                print(f"  - 平均成交价: {best_bp_combined['avg_token_price']:.4f}")

if __name__ == '__main__':
    main()
