#!/usr/bin/env python3
"""
检查是否有旧模型使用 0.50 的 LIMIT_PRICE 配置
"""

import json
import os
import glob
from pathlib import Path
import re

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

def check_limit_price_050(file_path):
    """检查文件中是否有 0.50 的 limitPriceConfigured"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            trades = json.load(f)
    except Exception as e:
        return None
    
    trades_with_050 = []
    for trade in trades:
        if 'limitPriceConfigured' in trade:
            limit_price = trade['limitPriceConfigured']
            if abs(limit_price - 0.50) < 0.001:  # 允许浮点数误差
                trades_with_050.append({
                    'id': trade.get('id'),
                    'timestamp': trade.get('timestamp'),
                    'tokenPrice': trade.get('tokenPrice'),
                    'limitPriceConfigured': limit_price,
                    'amount': trade.get('amount')
                })
    
    return trades_with_050

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
    
    print("检查是否有旧模型使用 0.50 的 LIMIT_PRICE 配置\n")
    print("=" * 100)
    
    found_050 = False
    
    for model_dir in sorted(model_dirs):
        dir_name = os.path.basename(model_dir)
        model_name, symbol = extract_model_info(dir_name)
        
        if not model_name:
            continue
        
        trades_file = os.path.join(model_dir, 'prediction_trades.json')
        if not os.path.exists(trades_file):
            continue
        
        trades_with_050 = check_limit_price_050(trades_file)
        
        if trades_with_050:
            found_050 = True
            print(f"\n{model_name} ({symbol}):")
            print(f"  找到 {len(trades_with_050)} 笔使用 0.50 的交易")
            print(f"  示例交易:")
            for i, trade in enumerate(trades_with_050[:5]):  # 只显示前5个
                print(f"    - ID: {trade['id']}")
                print(f"      时间: {trade['timestamp']}")
                print(f"      实际成交价: {trade['tokenPrice']}")
                print(f"      配置限价: {trade['limitPriceConfigured']}")
                print(f"      金额: {trade['amount']}")
            if len(trades_with_050) > 5:
                print(f"    ... 还有 {len(trades_with_050) - 5} 笔")
    
    if not found_050:
        print("\n✓ 未发现任何旧模型使用 0.50 的 LIMIT_PRICE 配置")
        print("  所有旧模型都使用 0.51 的配置")
    
    print("\n" + "=" * 100)
    
    # 检查启动脚本
    print("\n检查启动脚本中的 LIMIT_PRICE=0.50 配置:")
    print("-" * 100)
    
    script_files = []
    for root, dirs, files in os.walk(base_dir):
        # 跳过 node_modules
        if 'node_modules' in root:
            continue
        
        for file in files:
            if file.endswith(('.sh', '.py', '.js', '.ts', '.env')):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if re.search(r'LIMIT_PRICE\s*[=:]\s*0\.50\b', content, re.IGNORECASE):
                            script_files.append(file_path)
                except:
                    pass
    
    if script_files:
        print("找到以下文件包含 LIMIT_PRICE=0.50:")
        for file_path in script_files:
            rel_path = os.path.relpath(file_path, base_dir)
            print(f"  - {rel_path}")
    else:
        print("✓ 未在启动脚本中找到 LIMIT_PRICE=0.50 的配置")

if __name__ == '__main__':
    main()
