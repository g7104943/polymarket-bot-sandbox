#!/usr/bin/env python3
"""
分析旧模型的 prediction_trades.json 数据
统计每个模型的关键指标
"""

import json
import os
import glob
from pathlib import Path
from collections import defaultdict
import re

def extract_model_info(dir_name):
    """从目录名提取模型信息"""
    # 匹配 logs_gru_*_超参 或 logs_lgb_*_超参
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
        print(f"Error reading {file_path}: {e}")
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
    
    # 检查 limitPriceConfigured
    unique_limit_prices = list(set(limit_prices)) if limit_prices else []
    
    return {
        'total_trades': total_trades,
        'settled_trades': settled_trades,
        'wins': wins,
        'loses': loses,
        'win_rate': win_rate,
        'avg_token_price': avg_token_price,
        'total_pnl': total_pnl,
        'avg_amount': avg_amount,
        'unique_limit_prices': unique_limit_prices,
        'token_prices': token_prices
    }

def find_startup_scripts(base_dir):
    """查找启动脚本中的 LIMIT_PRICE 设置"""
    limit_price_configs = {}
    
    # 查找所有可能的启动脚本
    script_patterns = [
        '*.sh',
        '*.py',
        '*.js',
        '*.ts',
        '*.env',
        '.env*'
    ]
    
    for pattern in script_patterns:
        for file_path in glob.glob(os.path.join(base_dir, '**', pattern), recursive=True):
            if 'node_modules' in file_path or '__pycache__' in file_path:
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # 查找 LIMIT_PRICE 相关配置
                matches = re.findall(r'LIMIT_PRICE\s*[=:]\s*([0-9.]+)', content, re.IGNORECASE)
                if matches:
                    limit_price_configs[file_path] = matches
            except:
                pass
    
    return limit_price_configs

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
    
    print(f"找到 {len(model_dirs)} 个模型目录\n")
    
    # 分析每个模型
    results = []
    for model_dir in sorted(model_dirs):
        dir_name = os.path.basename(model_dir)
        model_name, symbol = extract_model_info(dir_name)
        
        if not model_name:
            print(f"警告: 无法解析目录名: {dir_name}")
            continue
        
        trades_file = os.path.join(model_dir, 'prediction_trades.json')
        if not os.path.exists(trades_file):
            print(f"警告: 文件不存在: {trades_file}")
            continue
        
        stats = analyze_trades(trades_file)
        if stats:
            results.append({
                'model_name': model_name,
                'symbol': symbol,
                'dir_name': dir_name,
                **stats
            })
    
    # 打印统计表格
    print("=" * 150)
    print(f"{'模型名称':<30} {'币种':<6} {'总交易':<8} {'已结算':<8} {'胜率':<8} {'平均tokenPrice':<15} {'总PnL':<12} {'平均amount':<12} {'LIMIT_PRICE配置':<20}")
    print("=" * 150)
    
    for result in sorted(results, key=lambda x: (x['symbol'], x['model_name'])):
        model_name = result['model_name']
        symbol = result['symbol']
        total_trades = result['total_trades']
        settled_trades = result['settled_trades']
        win_rate = result['win_rate']
        avg_token_price = result['avg_token_price']
        total_pnl = result['total_pnl']
        avg_amount = result['avg_amount']
        limit_prices = result['unique_limit_prices']
        
        win_rate_str = f"{win_rate:.2%}" if win_rate > 0 else "N/A"
        limit_price_str = ", ".join([f"{p:.2f}" for p in sorted(set(limit_prices))]) if limit_prices else "N/A"
        
        print(f"{model_name:<30} {symbol:<6} {total_trades:<8} {settled_trades:<8} {win_rate_str:<8} {avg_token_price:<15.4f} {total_pnl:<12.2f} {avg_amount:<12.2f} {limit_price_str:<20}")
    
    print("=" * 150)
    print()
    
    # 重点分析平均 tokenPrice
    print("\n" + "=" * 80)
    print("重点分析：平均 tokenPrice 与 LIMIT_PRICE=0.51 的对比")
    print("=" * 80)
    print()
    
    for result in sorted(results, key=lambda x: (x['symbol'], x['model_name'])):
        model_name = result['model_name']
        symbol = result['symbol']
        avg_token_price = result['avg_token_price']
        limit_prices = result['unique_limit_prices']
        
        # 检查是否有 0.51 的配置
        has_051 = 0.51 in limit_prices if limit_prices else False
        has_050 = 0.50 in limit_prices if limit_prices else False
        
        diff_from_051 = abs(avg_token_price - 0.51) if avg_token_price > 0 else None
        
        print(f"{model_name} ({symbol}):")
        print(f"  平均 tokenPrice: {avg_token_price:.4f}")
        if limit_prices:
            print(f"  LIMIT_PRICE 配置: {', '.join([f'{p:.2f}' for p in sorted(set(limit_prices))])}")
        else:
            print(f"  LIMIT_PRICE 配置: 未找到（可能是旧数据）")
        
        if diff_from_051 is not None:
            if diff_from_051 < 0.01:
                print(f"  ✓ 平均成交价非常接近 0.51 (差异: {diff_from_051:.4f})")
            elif diff_from_051 < 0.05:
                print(f"  ⚠ 平均成交价接近 0.51 (差异: {diff_from_051:.4f})")
            else:
                print(f"  ✗ 平均成交价偏离 0.51 (差异: {diff_from_051:.4f})")
        
        if has_050:
            print(f"  ⚠ 发现使用 0.50 的配置！")
        
        print()
    
    # 查找启动脚本中的 LIMIT_PRICE 配置
    print("\n" + "=" * 80)
    print("查找启动脚本或环境变量中的 LIMIT_PRICE 配置")
    print("=" * 80)
    print()
    
    limit_price_configs = find_startup_scripts(base_dir)
    
    if limit_price_configs:
        for file_path, prices in limit_price_configs.items():
            rel_path = os.path.relpath(file_path, base_dir)
            unique_prices = sorted(set([float(p) for p in prices]))
            print(f"{rel_path}:")
            for price in unique_prices:
                print(f"  LIMIT_PRICE = {price:.2f}")
            print()
    else:
        print("未在启动脚本中找到 LIMIT_PRICE 配置")
        print("（可能通过环境变量或其他方式配置）")
    
    # 统计汇总
    print("\n" + "=" * 80)
    print("汇总统计")
    print("=" * 80)
    print()
    
    # 按币种统计
    by_symbol = defaultdict(list)
    for result in results:
        by_symbol[result['symbol']].append(result)
    
    for symbol, models in sorted(by_symbol.items()):
        print(f"{symbol} 币种:")
        total_models = len(models)
        total_trades_all = sum(m['total_trades'] for m in models)
        total_settled_all = sum(m['settled_trades'] for m in models)
        total_pnl_all = sum(m['total_pnl'] for m in models)
        avg_token_prices = [m['avg_token_price'] for m in models if m['avg_token_price'] > 0]
        overall_avg_token_price = sum(avg_token_prices) / len(avg_token_prices) if avg_token_prices else 0
        
        print(f"  模型数量: {total_models}")
        print(f"  总交易笔数: {total_trades_all}")
        print(f"  总已结算笔数: {total_settled_all}")
        print(f"  总 PnL: {total_pnl_all:.2f}")
        print(f"  平均 tokenPrice: {overall_avg_token_price:.4f}")
        print()

if __name__ == '__main__':
    main()
