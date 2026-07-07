#!/usr/bin/env python3
"""
生成旧模型的完整统计报告
"""

import json
import os
import glob
from pathlib import Path
import re
from collections import defaultdict

def extract_model_info(dir_name):
    """从目录名提取模型信息"""
    match = re.match(r'logs_(gru|lgb)_([a-z]+)_(\d+)(?:_(.+))?_超参', dir_name)
    if match:
        model_type = match.group(1).upper()
        symbol = match.group(2).upper()
        model_id = match.group(3)
        suffix = match.group(4) if match.group(4) else ""
        model_name = f"{model_type}_{symbol}_{model_id}"
        if suffix:
            model_name += f"_{suffix}"
        return model_name, symbol
    return None, None

def analyze_trades(file_path):
    """分析单个 prediction_trades.json 文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            trades = json.load(f)
    except Exception as e:
        return None
    
    if not trades:
        return None
    
    # 统计指标
    total_trades = len(trades)
    settled_trades = 0
    wins = 0
    loses = 0
    total_pnl = 0
    total_token_price = 0
    total_amount = 0
    token_prices = []
    limit_prices = []
    
    for trade in trades:
        # 检查是否已结算
        if trade.get('result') in ['win', 'lose']:
            settled_trades += 1
            if trade.get('result') == 'win':
                wins += 1
            elif trade.get('result') == 'lose':
                loses += 1
        
        # PnL
        if 'pnl' in trade:
            total_pnl += trade['pnl']
        
        # tokenPrice
        if 'tokenPrice' in trade:
            token_price = trade['tokenPrice']
            total_token_price += token_price
            token_prices.append(token_price)
        
        # amount
        if 'amount' in trade:
            total_amount += trade['amount']
        
        # limitPriceConfigured
        if 'limitPriceConfigured' in trade:
            limit_prices.append(trade['limitPriceConfigured'])
    
    # 计算平均值
    avg_token_price = total_token_price / len(token_prices) if token_prices else 0
    avg_amount = total_amount / total_trades if total_trades > 0 else 0
    win_rate = wins / settled_trades if settled_trades > 0 else 0
    
    # tokenPrice 统计
    if token_prices:
        min_token_price = min(token_prices)
        max_token_price = max(token_prices)
        # 计算中位数
        sorted_prices = sorted(token_prices)
        median_token_price = sorted_prices[len(sorted_prices) // 2]
    else:
        min_token_price = max_token_price = median_token_price = 0
    
    # 检查 limitPriceConfigured
    unique_limit_prices = list(set(limit_prices)) if limit_prices else []
    
    return {
        'total_trades': total_trades,
        'settled_trades': settled_trades,
        'wins': wins,
        'loses': loses,
        'win_rate': win_rate,
        'avg_token_price': avg_token_price,
        'min_token_price': min_token_price,
        'max_token_price': max_token_price,
        'median_token_price': median_token_price,
        'total_pnl': total_pnl,
        'avg_amount': avg_amount,
        'unique_limit_prices': unique_limit_prices,
        'token_prices': token_prices
    }

def main():
    base_dir = Path('/Users/mac/polyfun/polymarket')
    
    # 查找所有匹配的日志目录
    pattern1 = base_dir / 'logs_gru_*_超参'
    pattern2 = base_dir / 'logs_lgb_*_超参'
    
    model_dirs = []
    for pattern in [pattern1, pattern2]:
        model_dirs.extend(glob.glob(str(pattern)))
    
    # 排除 archive 目录中的
    model_dirs = [d for d in model_dirs if 'archive' not in d]
    
    # 分析每个模型
    results = []
    for model_dir in sorted(model_dirs):
        dir_name = os.path.basename(model_dir)
        model_name, symbol = extract_model_info(dir_name)
        
        if not model_name:
            continue
        
        trades_file = os.path.join(model_dir, 'prediction_trades.json')
        if not os.path.exists(trades_file):
            continue
        
        stats = analyze_trades(trades_file)
        if stats:
            results.append({
                'model_name': model_name,
                'symbol': symbol,
                'dir_name': dir_name,
                **stats
            })
    
    # 生成完整报告
    print("\n" + "=" * 180)
    print("旧模型完整统计报告".center(180))
    print("=" * 180)
    print()
    
    # 主表格
    print(f"{'模型名称':<30} {'币种':<6} {'总交易':<8} {'已结算':<8} {'胜率':<10} {'平均tokenPrice':<18} {'总PnL':<12} {'平均amount':<12}")
    print("-" * 180)
    
    for result in sorted(results, key=lambda x: (x['symbol'], x['model_name'])):
        model_name = result['model_name']
        symbol = result['symbol']
        total_trades = result['total_trades']
        settled_trades = result['settled_trades']
        win_rate = result['win_rate']
        avg_token_price = result['avg_token_price']
        total_pnl = result['total_pnl']
        avg_amount = result['avg_amount']
        
        win_rate_str = f"{win_rate:.2%}" if win_rate > 0 else "N/A"
        
        print(f"{model_name:<30} {symbol:<6} {total_trades:<8} {settled_trades:<8} {win_rate_str:<10} {avg_token_price:<18.4f} {total_pnl:<12.2f} {avg_amount:<12.2f}")
    
    print("=" * 180)
    print()
    
    # 详细分析：平均 tokenPrice 与 LIMIT_PRICE=0.51 的对比
    print("\n" + "=" * 180)
    print("详细分析：平均 tokenPrice 与 LIMIT_PRICE=0.51 的对比".center(180))
    print("=" * 180)
    print()
    
    print(f"{'模型名称':<30} {'币种':<6} {'平均tokenPrice':<18} {'最小':<10} {'最大':<10} {'中位数':<10} {'LIMIT_PRICE配置':<20} {'与0.51差异':<15}")
    print("-" * 180)
    
    for result in sorted(results, key=lambda x: (x['symbol'], x['model_name'])):
        model_name = result['model_name']
        symbol = result['symbol']
        avg_token_price = result['avg_token_price']
        min_token_price = result['min_token_price']
        max_token_price = result['max_token_price']
        median_token_price = result['median_token_price']
        limit_prices = result['unique_limit_prices']
        
        limit_price_str = ", ".join([f"{p:.2f}" for p in sorted(set(limit_prices))]) if limit_prices else "未配置"
        diff_from_051 = abs(avg_token_price - 0.51) if avg_token_price > 0 else None
        diff_str = f"{diff_from_051:.4f}" if diff_from_051 is not None else "N/A"
        
        print(f"{model_name:<30} {symbol:<6} {avg_token_price:<18.4f} {min_token_price:<10.4f} {max_token_price:<10.4f} {median_token_price:<10.4f} {limit_price_str:<20} {diff_str:<15}")
    
    print("=" * 180)
    print()
    
    # 重点结论
    print("\n" + "=" * 180)
    print("重点结论".center(180))
    print("=" * 180)
    print()
    
    # 统计平均 tokenPrice
    all_avg_prices = [r['avg_token_price'] for r in results if r['avg_token_price'] > 0]
    overall_avg = sum(all_avg_prices) / len(all_avg_prices) if all_avg_prices else 0
    
    print(f"1. 所有旧模型的平均 tokenPrice: {overall_avg:.4f}")
    print(f"2. 与 LIMIT_PRICE=0.51 的差异: {abs(overall_avg - 0.51):.4f}")
    
    if abs(overall_avg - 0.51) < 0.01:
        print("   ✓ 整体平均成交价非常接近 0.51，说明限价订单执行良好")
    elif abs(overall_avg - 0.51) < 0.05:
        print("   ⚠ 整体平均成交价接近 0.51，但存在一定偏差")
    else:
        print("   ✗ 整体平均成交价偏离 0.51 较远，需要检查限价订单执行情况")
    
    print()
    
    # 检查是否有 0.50 的配置
    has_050 = False
    for result in results:
        if 0.50 in result['unique_limit_prices']:
            has_050 = True
            print(f"3. ⚠ 发现模型 {result['model_name']} 使用 0.50 的 LIMIT_PRICE 配置")
    
    if not has_050:
        print("3. ✓ 所有旧模型都使用 0.51 的 LIMIT_PRICE 配置，未发现 0.50 的配置")
    
    print()
    
    # 按币种汇总
    print("4. 按币种汇总:")
    by_symbol = defaultdict(list)
    for result in results:
        by_symbol[result['symbol']].append(result)
    
    for symbol, models in sorted(by_symbol.items()):
        total_trades_all = sum(m['total_trades'] for m in models)
        total_settled_all = sum(m['settled_trades'] for m in models)
        total_pnl_all = sum(m['total_pnl'] for m in models)
        avg_token_prices = [m['avg_token_price'] for m in models if m['avg_token_price'] > 0]
        symbol_avg_price = sum(avg_token_prices) / len(avg_token_prices) if avg_token_prices else 0
        
        print(f"   {symbol}:")
        print(f"     - 模型数量: {len(models)}")
        print(f"     - 总交易笔数: {total_trades_all}")
        print(f"     - 总已结算笔数: {total_settled_all}")
        print(f"     - 总 PnL: {total_pnl_all:.2f}")
        print(f"     - 平均 tokenPrice: {symbol_avg_price:.4f}")
        print(f"     - 与 0.51 的差异: {abs(symbol_avg_price - 0.51):.4f}")
        print()

if __name__ == '__main__':
    main()
