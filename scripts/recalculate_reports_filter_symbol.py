#!/usr/bin/env python3
"""
重新计算报告：过滤掉指定币种的交易，重新计算资金和统计

用法:
    python scripts/recalculate_reports_filter_symbol.py --log-dir logs_btc --filter-symbol ETH
    python scripts/recalculate_reports_filter_symbol.py --log-dir logs_xrp --filter-symbol ETH
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def recalculate_capital(trades: List[Dict], initial_capital: float = 400.0) -> Dict[str, Any]:
    """
    根据交易记录重新计算资金
    
    逻辑：
    - 初始资金：400
    - 下单时：扣除下注金额
    - 赢了：加上代币价值（代币数量 * $1）
    - 输了：不加回（已在下单时扣除）
    """
    capital = initial_capital
    peak_capital = initial_capital
    min_capital = initial_capital
    
    # 按时间排序
    sorted_trades = sorted(trades, key=lambda x: x.get('timestamp', ''))
    
    for trade in sorted_trades:
        amount = trade.get('amount', 0)
        result = trade.get('result')
        pnl = trade.get('pnl', 0)
        
        # 下单时扣除下注金额
        capital -= amount
        
        # 如果赢了，加上代币价值
        if result == 'win' and pnl is not None:
            # pnl = 代币价值 - 下注金额
            # 所以代币价值 = pnl + 下注金额
            capital += (pnl + amount)
        # 如果输了，不加回（已在下单时扣除）
        
        # 更新峰值和最低资金
        if capital > peak_capital:
            peak_capital = capital
        if capital < min_capital:
            min_capital = capital
    
    return {
        'current_capital': capital,
        'peak_capital': peak_capital,
        'min_capital': min_capital,
    }


def filter_trades_by_symbol(trades: List[Dict], filter_symbol: str) -> List[Dict]:
    """过滤掉指定币种的交易"""
    return [t for t in trades if t.get('symbol') != filter_symbol]


def calculate_statistics(trades: List[Dict]) -> Dict[str, Any]:
    """计算统计信息"""
    wins = 0
    losses = 0
    pending = 0
    total_pnl = 0
    total_confidence = 0
    up_count = 0
    down_count = 0
    
    by_symbol: Dict[str, Dict] = {}
    
    for trade in trades:
        symbol = trade.get('symbol', '')
        if symbol not in by_symbol:
            by_symbol[symbol] = {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'pending': 0,
                'pnl': 0,
                'total_confidence': 0,
                'up_count': 0,
                'down_count': 0,
            }
        
        s = by_symbol[symbol]
        s['trades'] += 1
        s['total_confidence'] += trade.get('confidence', 0)
        total_confidence += trade.get('confidence', 0)
        
        direction = trade.get('direction')
        if direction == 'UP':
            up_count += 1
            s['up_count'] += 1
        elif direction == 'DOWN':
            down_count += 1
            s['down_count'] += 1
        
        result = trade.get('result')
        if result == 'win':
            wins += 1
            s['wins'] += 1
        elif result == 'lose':
            losses += 1
            s['losses'] += 1
        else:
            pending += 1
            s['pending'] += 1
        
        pnl = trade.get('pnl', 0)
        if pnl is not None:
            total_pnl += pnl
            s['pnl'] += pnl
    
    # 计算平均置信度和胜率
    avg_confidence = (total_confidence / len(trades) * 100) if trades else 0
    completed = wins + losses
    win_rate = (wins / completed * 100) if completed > 0 else 0
    
    # 计算每个币种的统计
    by_symbol_stats = {}
    for symbol, s in by_symbol.items():
        symbol_completed = s['wins'] + s['losses']
        by_symbol_stats[symbol] = {
            'trades': s['trades'],
            'wins': s['wins'],
            'losses': s['losses'],
            'pending': s['pending'],
            'win_rate': (s['wins'] / symbol_completed * 100) if symbol_completed > 0 else 0,
            'pnl': round(s['pnl'], 2),
            'avg_confidence': (s['total_confidence'] / s['trades'] * 100) if s['trades'] > 0 else 0,
            'up_count': s['up_count'],
            'down_count': s['down_count'],
        }
    
    return {
        'total_trades': len(trades),
        'completed': completed,
        'wins': wins,
        'losses': losses,
        'pending': pending,
        'win_rate': round(win_rate, 1),
        'total_pnl': round(total_pnl, 2),
        'avg_confidence': round(avg_confidence, 1),
        'up_count': up_count,
        'down_count': down_count,
        'by_symbol': by_symbol_stats,
    }


def format_report_text(report: Dict[str, Any], capital_info: Dict[str, Any]) -> str:
    """格式化报告文本"""
    summary = report['summary']
    config = report.get('config', {})
    # 支持两种命名方式：bySymbol 和 by_symbol
    by_symbol = report.get('bySymbol') or report.get('by_symbol', {})
    trades = report.get('trades', [])
    
    lines = []
    lines.append("═" * 60)
    lines.append("  POLYMARKET 预测交易 - 总汇总报告（全部历史）")
    lines.append("═" * 60)
    lines.append("")
    lines.append(f"报告时间: {report['reportDate']}")
    lines.append(f"统计周期: {report['reportPeriod']['start']} ~ {report['reportPeriod']['end']} (从开始到现在的全部历史)")
    lines.append("")
    lines.append("─" * 60)
    lines.append("  综合统计")
    lines.append("─" * 60)
    lines.append(f"  总交易数:     {summary['totalTrades']}")
    lines.append(f"  下单方向:     Up: {summary['upCount']} 笔, Down: {summary['downCount']} 笔")
    lines.append(f"  已完成:       {summary['completed']}")
    lines.append(f"  胜利:         {summary['wins']}")
    lines.append(f"  失败:         {summary['losses']}")
    lines.append(f"  待结算:       {summary['pending']}")
    lines.append(f"  胜率:         {summary['winRate']}%")
    lines.append(f"  总盈亏:       ${summary['totalPnL']:+.2f}")
    lines.append(f"  平均置信度:   {summary['avgConfidence']}%")
    lines.append(f"  初始资金:     ${config['initialCapital']}")
    lines.append(f"  当前资金:     ${capital_info['current_capital']:.2f}")
    lines.append(f"  资金变化:     ${capital_info['current_capital'] - config['initialCapital']:+.2f}")
    lines.append("")
    lines.append("─" * 60)
    lines.append("  各币种统计")
    lines.append("─" * 60)
    
    if by_symbol:
        for symbol, stats in by_symbol.items():
            lines.append(f"  {symbol}:")
            lines.append(f"    交易数: {stats['trades']}, 胜率: {stats['win_rate']:.1f}%, 盈亏: ${stats['pnl']:+.2f}")
            lines.append(f"    下单: Up {stats['up_count']} 笔, Down {stats['down_count']} 笔 | 胜/负/待: {stats['wins']}/{stats['losses']}/{stats['pending']}, 平均置信度: {stats['avg_confidence']:.1f}%")
    
    lines.append("")
    lines.append("─" * 60)
    lines.append("  配置信息")
    lines.append("─" * 60)
    lines.append(f"  交易模式:     {config['tradingMode']}")
    lines.append(f"  初始资金:     ${config['initialCapital']}")
    lines.append(f"  下注比例:     {config['betSizePercent']}")
    lines.append(f"  置信度阈值:   {config['probThreshold']}")
    lines.append(f"  最小利润比:   {config['minProfitRatio']}")
    lines.append(f"  允许市场:     {', '.join(config['allowedMarkets'])}")
    lines.append("")
    lines.append("─" * 60)
    lines.append("  交易记录")
    lines.append("─" * 60)
    
    # 按时间排序显示交易记录
    sorted_trades = sorted(trades, key=lambda x: x.get('timestamp', ''))
    for trade in sorted_trades[-50:]:  # 只显示最后50笔
        timestamp = trade.get('timestamp', '')
        if timestamp:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            time_str = dt.strftime('%Y/%m/%d %H:%M:%S')
        else:
            time_str = timestamp
        
        symbol = trade.get('symbol', '')
        direction = '↑' if trade.get('direction') == 'UP' else '↓'
        confidence = trade.get('confidence', 0) * 100
        amount = trade.get('amount', 0)
        result = trade.get('result', 'pending')
        pnl = trade.get('pnl', 0)
        
        if result == 'win':
            result_icon = '✓'
            pnl_str = f"$+{pnl:.2f}" if pnl else "$+0.00"
        elif result == 'lose':
            result_icon = '✗'
            pnl_str = f"${pnl:.2f}" if pnl else "$0.00"
        else:
            result_icon = '?'
            pnl_str = '-'
        
        lines.append(f"  [{time_str}] {symbol} {direction} {confidence:.1f}% ${amount:.2f} → {result_icon} {pnl_str}")
    
    lines.append("")
    lines.append("═" * 60)
    lines.append("  报告结束")
    lines.append("═" * 60)
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="重新计算报告：过滤掉指定币种的交易")
    parser.add_argument('--log-dir', type=str, required=True, help='日志目录，如 logs_btc 或 logs_xrp')
    parser.add_argument('--filter-symbol', type=str, required=True, help='要过滤的币种，如 ETH')
    parser.add_argument('--initial-capital', type=float, default=400.0, help='初始资金，默认 400')
    
    args = parser.parse_args()
    
    log_dir = PROJECT_ROOT / "polymarket" / args.log_dir
    trades_file = log_dir / "prediction_trades.json"
    reports_dir = log_dir / "reports"
    
    if not trades_file.exists():
        print(f"❌ 文件不存在: {trades_file}")
        return
    
    print(f"📊 处理日志目录: {log_dir}")
    print(f"🔍 过滤币种: {args.filter_symbol}")
    print()
    
    # 读取交易记录
    with open(trades_file, 'r', encoding='utf-8') as f:
        all_trades = json.load(f)
    
    print(f"原始交易数: {len(all_trades)} 笔")
    
    # 过滤交易
    filtered_trades = filter_trades_by_symbol(all_trades, args.filter_symbol)
    filtered_count = len(all_trades) - len(filtered_trades)
    
    print(f"过滤掉 {args.filter_symbol}: {filtered_count} 笔")
    print(f"剩余交易数: {len(filtered_trades)} 笔")
    print()
    
    if not filtered_trades:
        print("⚠️  警告: 过滤后没有交易记录了")
        return
    
    # 重新计算资金
    print("💰 重新计算资金...")
    capital_info = recalculate_capital(filtered_trades, args.initial_capital)
    print(f"  初始资金: ${args.initial_capital}")
    print(f"  当前资金: ${capital_info['current_capital']:.2f}")
    print(f"  资金变化: ${capital_info['current_capital'] - args.initial_capital:+.2f}")
    print(f"  峰值资金: ${capital_info['peak_capital']:.2f}")
    print(f"  最低资金: ${capital_info['min_capital']:.2f}")
    print()
    
    # 计算统计
    print("📈 计算统计信息...")
    stats = calculate_statistics(filtered_trades)
    
    # 构建报告对象
    first_trade_time = filtered_trades[0].get('timestamp', '') if filtered_trades else datetime.now().isoformat()
    last_trade_time = filtered_trades[-1].get('timestamp', '') if filtered_trades else datetime.now().isoformat()
    
    # 读取原始报告以获取配置信息
    original_report_file = reports_dir / "report_summary.json"
    if original_report_file.exists():
        with open(original_report_file, 'r', encoding='utf-8') as f:
            original_report = json.load(f)
        config = original_report.get('config', {})
    else:
        config = {
            'tradingMode': 'simulation',
            'initialCapital': args.initial_capital,
            'betSizePercent': '5%',
            'probThreshold': '55%',
            'minProfitRatio': 0.4,
            'allowedMarkets': [s for s in ['BTC', 'ETH', 'XRP', 'SOL'] if s != args.filter_symbol],
        }
    
    report = {
        'reportType': 'summary',
        'reportDate': datetime.now().isoformat(),
        'reportPeriod': {
            'start': first_trade_time,
            'end': last_trade_time,
        },
        'summary': {
            'totalTrades': stats['total_trades'],
            'completedTrades': stats['completed'],
            'wins': stats['wins'],
            'losses': stats['losses'],
            'pending': stats['pending'],
            'upCount': stats['up_count'],
            'downCount': stats['down_count'],
            'winRate': stats['win_rate'],
            'totalPnL': stats['total_pnl'],
            'avgConfidence': stats['avg_confidence'],
            'currentCapital': capital_info['current_capital'],
            'initialCapital': args.initial_capital,
            'capitalChange': capital_info['current_capital'] - args.initial_capital,
        },
        'bySymbol': stats['by_symbol'],
        'config': config,
        'trades': filtered_trades,
    }
    
    # 确保 report 对象包含所有需要的字段
    report['summary'] = {
        **report['summary'],
        'completed': stats['completed'],
    }
    
    # 更新配置中的允许市场
    if args.filter_symbol in config.get('allowedMarkets', []):
        config['allowedMarkets'] = [m for m in config['allowedMarkets'] if m != args.filter_symbol]
    
    # 生成报告文本
    report_text = format_report_text(report, capital_info)
    
    # 保存报告
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存 JSON
    json_path = reports_dir / "report_summary.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON 报告已保存: {json_path}")
    
    # 保存文本
    txt_path = reports_dir / "report_summary.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"✅ 文本报告已保存: {txt_path}")
    
    print()
    print("=" * 60)
    print("✅ 重新计算完成！")
    print("=" * 60)
    print()
    print("统计摘要:")
    print(f"  总交易数: {stats['total_trades']}")
    print(f"  胜率: {stats['win_rate']:.1f}%")
    print(f"  总盈亏: ${stats['total_pnl']:+.2f}")
    print(f"  当前资金: ${capital_info['current_capital']:.2f}")
    print(f"  资金变化: ${capital_info['current_capital'] - args.initial_capital:+.2f}")


if __name__ == '__main__':
    main()
