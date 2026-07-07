#!/usr/bin/env python3
"""
分析 Exp8 和 Exp16 模型的 ETH 交易数据统计
"""
import json
from pathlib import Path
from collections import defaultdict
import pandas as pd

SCRIPT_DIR = Path(__file__).parent

def normalize_symbol(symbol):
    """标准化币种符号"""
    if not symbol:
        return ""
    return symbol.upper().replace('/USDT', '').strip()

def analyze_exp_trades(exp_name, bp_patterns):
    """分析指定实验的交易数据"""
    results = []
    
    for bp_pattern in bp_patterns:
        # 构建目录名
        if bp_pattern.startswith('bp_dyn'):
            d = f'logs_v5_{exp_name}_{bp_pattern}'
        else:
            d = f'logs_v5_{exp_name}_{bp_pattern}'
        
        p = SCRIPT_DIR / d / 'prediction_trades.json'
        
        if not p.exists():
            print(f"警告: 文件不存在 {p}")
            continue
        
        try:
            data = json.load(open(p))
        except Exception as e:
            print(f"错误: 无法读取 {p}: {e}")
            continue
        
        # 筛选 ETH 币种且已执行的交易
        eth_trades = [
            t for t in data 
            if t.get('status') == 'executed' 
            and normalize_symbol(t.get('symbol', '')) == 'ETH'
        ]
        
        if not eth_trades:
            print(f"警告: {d} 中没有 ETH 交易")
            continue
        
        # 统计指标
        num_trades = len(eth_trades)
        
        # 已结算事件数（不同的 marketSlug 中有 win/lose 结果的）
        settled_slugs = set()
        for t in eth_trades:
            result = t.get('result', '').lower()
            if result in ['win', 'lose']:
                slug = t.get('marketSlug', '')
                if slug:
                    settled_slugs.add(slug)
        num_settled_events = len(settled_slugs)
        
        # 胜率（基于已结算的交易）
        settled_trades = [t for t in eth_trades if t.get('result', '').lower() in ['win', 'lose']]
        if settled_trades:
            wins = sum(1 for t in settled_trades if t.get('result', '').lower() == 'win')
            win_rate = wins / len(settled_trades) * 100
        else:
            win_rate = 0.0
        
        # 总 PnL（只计算已结算的交易）
        total_pnl = sum(float(t.get('pnl', 0)) for t in settled_trades if t.get('pnl') is not None)
        
        # 平均 tokenPrice（所有已执行交易的实际成交价）
        prices = [float(t['tokenPrice']) for t in eth_trades if t.get('tokenPrice') is not None]
        avg_price = sum(prices) / len(prices) if prices else 0.0
        
        results.append({
            '买价组合': bp_pattern,
            '交易笔数': num_trades,
            '已结算事件数': num_settled_events,
            '胜率(%)': round(win_rate, 2),
            '总PnL': round(total_pnl, 2),
            '平均tokenPrice': round(avg_price, 4)
        })
        
        print(f"✓ {d}: {num_trades} 笔交易, {num_settled_events} 个已结算事件, 胜率 {win_rate:.2f}%")
    
    return results

def main():
    print("=" * 80)
    print("Exp8 和 Exp16 ETH 交易数据分析")
    print("=" * 80)
    
    # Exp8 的买价组合
    exp8_bps = [
        'bp0450', 'bp0460', 'bp0470', 'bp0480', 'bp0490', 
        'bp0500', 'bp0510', 'bp0520', 'bp0530',
        'bp_dyn_0450_0530', 'bp_dyn_0480_0510'
    ]
    
    # Exp16 的买价组合
    exp16_bps = [
        'bp0450', 'bp0460', 'bp0470', 'bp0480', 'bp0490', 
        'bp0500', 'bp0510', 'bp0520', 'bp0530',
        'bp_dyn_0450_0530', 'bp_dyn_0480_0510'
    ]
    
    print("\n" + "=" * 80)
    print("Exp8 分析")
    print("=" * 80)
    exp8_results = analyze_exp_trades('exp8', exp8_bps)
    
    print("\n" + "=" * 80)
    print("Exp16 分析")
    print("=" * 80)
    exp16_results = analyze_exp_trades('exp16', exp16_bps)
    
    # 创建 DataFrame 并输出
    print("\n" + "=" * 80)
    print("Exp8 统计结果")
    print("=" * 80)
    df_exp8 = pd.DataFrame(exp8_results)
    if not df_exp8.empty:
        print(df_exp8.to_string(index=False))
    else:
        print("无数据")
    
    print("\n" + "=" * 80)
    print("Exp16 统计结果")
    print("=" * 80)
    df_exp16 = pd.DataFrame(exp16_results)
    if not df_exp16.empty:
        print(df_exp16.to_string(index=False))
    else:
        print("无数据")
    
    # 保存到 CSV
    if not df_exp8.empty:
        df_exp8.to_csv(SCRIPT_DIR / 'exp8_eth_statistics.csv', index=False, encoding='utf-8-sig')
        print(f"\nExp8 结果已保存到: {SCRIPT_DIR / 'exp8_eth_statistics.csv'}")
    
    if not df_exp16.empty:
        df_exp16.to_csv(SCRIPT_DIR / 'exp16_eth_statistics.csv', index=False, encoding='utf-8-sig')
        print(f"Exp16 结果已保存到: {SCRIPT_DIR / 'exp16_eth_statistics.csv'}")
    
    # Dynamic 价格组合的平均成交价汇总
    print("\n" + "=" * 80)
    print("Dynamic 价格组合平均成交价汇总")
    print("=" * 80)
    dynamic_results = []
    
    for exp_name in ['exp8', 'exp16']:
        for bp_dyn in ['bp_dyn_0450_0530', 'bp_dyn_0480_0510']:
            d = f'logs_v5_{exp_name}_{bp_dyn}'
            p = SCRIPT_DIR / d / 'prediction_trades.json'
            
            if not p.exists():
                continue
            
            try:
                data = json.load(open(p))
            except Exception as e:
                continue
            
            eth_trades = [
                t for t in data 
                if t.get('status') == 'executed' 
                and normalize_symbol(t.get('symbol', '')) == 'ETH'
            ]
            
            if eth_trades:
                prices = [float(t['tokenPrice']) for t in eth_trades if t.get('tokenPrice') is not None]
                avg_price = sum(prices) / len(prices) if prices else 0.0
                dynamic_results.append({
                    '实验': exp_name.upper(),
                    '买价组合': bp_dyn,
                    '平均tokenPrice': round(avg_price, 4),
                    '交易笔数': len(eth_trades)
                })
    
    if dynamic_results:
        df_dynamic = pd.DataFrame(dynamic_results)
        print(df_dynamic.to_string(index=False))
        df_dynamic.to_csv(SCRIPT_DIR / 'dynamic_price_statistics.csv', index=False, encoding='utf-8-sig')
        print(f"\nDynamic 结果已保存到: {SCRIPT_DIR / 'dynamic_price_statistics.csv'}")
    else:
        print("无数据")

if __name__ == '__main__':
    main()
