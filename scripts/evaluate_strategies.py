#!/usr/bin/env python3
"""
策略评测与排名脚本
基于金融和数学理论，综合评估交易策略的表现

评估维度：
1. 收益指标：CAGR（年化复合增长率）
2. 风险指标：最大回撤（MaxDD）、波动率（Volatility）
3. 风险调整收益：Sharpe Ratio、Sortino Ratio、Calmar Ratio
4. 稳定性：胜率（WinRate）、盈亏比（Profit Factor）
5. 稳健性：回撤恢复时间、连续亏损次数

排名方法：多维度加权评分 + 风险调整
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 配置参数
# ============================================================

# 指标权重配置（可根据策略类型调整）
WEIGHTS = {
    'calmar': 0.30,      # Calmar Ratio：风险调整收益的核心指标
    'sharpe': 0.20,      # Sharpe Ratio：总风险调整收益
    'sortino': 0.15,     # Sortino Ratio：下行风险调整收益
    'mdd': 0.15,         # 最大回撤：风险控制能力
    'winrate': 0.10,     # 胜率：策略稳定性
    'profit_factor': 0.10, # 盈亏比：盈利效率
}

# 可选：添加其他指标权重
# 'volatility': 0.05,    # 波动率（越小越好）
# 'recovery_time': 0.05, # 回撤恢复时间（越小越好）

# ============================================================
# 辅助函数
# ============================================================

def minmax_normalize(series: pd.Series, reverse: bool = False) -> pd.Series:
    """
    最小-最大标准化（Min-Max Normalization）
    
    Args:
        series: 待标准化的序列
        reverse: 是否反向（越小越好时设为 True）
    
    Returns:
        标准化后的序列（0-1之间）
    """
    if series.max() == series.min():
        return pd.Series(0.5, index=series.index)
    
    normalized = (series - series.min()) / (series.max() - series.min() + 1e-9)
    return 1 - normalized if reverse else normalized


def zscore_normalize(series: pd.Series, reverse: bool = False) -> pd.Series:
    """
    Z-score 标准化（可选方法）
    适用于数据分布接近正态分布的情况
    """
    mean = series.mean()
    std = series.std()
    if std == 0:
        return pd.Series(0.5, index=series.index)
    
    z_scores = (series - mean) / (std + 1e-9)
    # 转换为 0-1 范围（使用 sigmoid）
    normalized = 1 / (1 + np.exp(-z_scores))
    return 1 - normalized if reverse else normalized


def calculate_additional_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算额外的金融指标
    支持多种列名变体（中英文）
    """
    df = df.copy()
    
    # 1. 识别 CAGR（年化收益率）列
    cagr_col = None
    for col in ['CAGR', '年化收益率', '年化收益', '盈亏%', '盈亏']:
        if col in df.columns:
            cagr_col = col
            break
    
    # 2. 识别 MaxDD（最大回撤）列
    mdd_col = None
    for col in ['MaxDD', '最大回撤%', '最大回撤', 'MaxDD (%)']:
        if col in df.columns:
            mdd_col = col
            break
    
    # 3. 计算 Calmar Ratio（如果未提供但有所需列）
    if 'Calmar' not in df.columns and cagr_col and mdd_col:
        df['Calmar'] = df[cagr_col] / (df[mdd_col] + 1e-9)
    
    # 4. 标准化列名（为了后续处理方便，创建标准列名的副本）
    if cagr_col and 'CAGR' not in df.columns:
        df['CAGR'] = df[cagr_col]
    if mdd_col and 'MaxDD' not in df.columns:
        df['MaxDD'] = df[mdd_col]
    
    # 5. 识别其他指标
    if 'Sharpe' not in df.columns:
        for col in ['Sharpe', 'Sharpe Ratio']:
            if col in df.columns:
                df['Sharpe'] = df[col]
                break
    
    if 'Sortino' not in df.columns:
        for col in ['Sortino', 'Sortino Ratio']:
            if col in df.columns:
                df['Sortino'] = df[col]
                break
    
    if 'WinRate' not in df.columns:
        for col in ['WinRate', '胜率%', '胜率', 'WinRate (%)']:
            if col in df.columns:
                df['WinRate'] = df[col]
                break
    
    return df


def find_column(df: pd.DataFrame, possible_names: list) -> Optional[str]:
    """查找列名（支持多种变体）"""
    for name in possible_names:
        if name in df.columns:
            return name
    return None

def calculate_composite_score(
    df: pd.DataFrame,
    weights: Dict[str, float],
    use_zscore: bool = False
) -> pd.Series:
    """
    计算综合评分
    支持中英文列名
    """
    score = pd.Series(0.0, index=df.index)
    normalize_func = zscore_normalize if use_zscore else minmax_normalize
    
    # Calmar Ratio（越大越好）
    calmar_col = find_column(df, ['Calmar', 'Calmar Ratio'])
    if calmar_col and 'calmar' in weights:
        score += weights['calmar'] * normalize_func(df[calmar_col])
    
    # Sharpe Ratio（越大越好）
    sharpe_col = find_column(df, ['Sharpe', 'Sharpe Ratio'])
    if sharpe_col and 'sharpe' in weights:
        score += weights['sharpe'] * normalize_func(df[sharpe_col])
    
    # Sortino Ratio（越大越好）
    sortino_col = find_column(df, ['Sortino', 'Sortino Ratio'])
    if sortino_col and 'sortino' in weights:
        score += weights['sortino'] * normalize_func(df[sortino_col])
    
    # MaxDD（越小越好，反向）- 支持中文列名
    mdd_col = find_column(df, ['MaxDD', 'MaxDD (%)', '最大回撤%', '最大回撤'])
    if mdd_col and 'mdd' in weights:
        score += weights['mdd'] * normalize_func(df[mdd_col], reverse=True)
    
    # WinRate（越大越好）- 支持中文列名
    winrate_col = find_column(df, ['WinRate', 'WinRate (%)', '胜率%', '胜率'])
    if winrate_col and 'winrate' in weights:
        score += weights['winrate'] * normalize_func(df[winrate_col])
    
    # Profit Factor（越大越好）
    pf_col = find_column(df, ['ProfitFactor', 'Profit_Factor', 'Profit Factor', '盈亏比'])
    if pf_col and 'profit_factor' in weights:
        score += weights['profit_factor'] * normalize_func(df[pf_col])
    
    # Volatility（越小越好，反向）
    vol_col = find_column(df, ['Volatility', 'Volatility (%)', '波动率'])
    if vol_col and 'volatility' in weights:
        score += weights['volatility'] * normalize_func(df[vol_col], reverse=True)
    
    return score


def rank_strategies(
    df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    method: str = 'weighted'
) -> pd.DataFrame:
    """
    策略排名
    
    Args:
        df: 包含策略数据的数据框
        weights: 权重字典（默认使用全局 WEIGHTS）
        method: 排名方法
            - 'weighted': 加权评分（默认）
            - 'calmar_only': 仅使用 Calmar Ratio
            - 'sharpe_only': 仅使用 Sharpe Ratio
            - 'pareto': 帕累托最优（多目标优化）
    
    Returns:
        排名后的数据框
    """
    if weights is None:
        weights = WEIGHTS
    
    df = df.copy()
    
    # 计算额外指标
    df = calculate_additional_metrics(df)
    
    if method == 'weighted':
        # 加权评分方法
        df['Score'] = calculate_composite_score(df, weights)
        df = df.sort_values('Score', ascending=False)
        
    elif method == 'calmar_only':
        # 仅使用 Calmar Ratio
        if 'Calmar' not in df.columns:
            df['Calmar'] = df['CAGR'] / (df['MaxDD'] + 1e-9)
        df['Score'] = df['Calmar']
        df = df.sort_values('Score', ascending=False)
        
    elif method == 'sharpe_only':
        # 仅使用 Sharpe Ratio
        if 'Sharpe' not in df.columns:
            raise ValueError("Sharpe Ratio 未在数据中找到")
        df['Score'] = df['Sharpe']
        df = df.sort_values('Score', ascending=False)
        
    elif method == 'pareto':
        # 帕累托最优方法（多目标优化）
        # 这里简化实现：找到在所有指标上都表现较好的策略
        # 实际应用中可以使用更复杂的帕累托前沿算法
        df['Score'] = (
            df['Calmar'].rank(ascending=False, pct=True) * weights.get('calmar', 0.3) +
            df['Sharpe'].rank(ascending=False, pct=True) * weights.get('sharpe', 0.2) +
            df['MaxDD'].rank(ascending=True, pct=True) * weights.get('mdd', 0.15) +
            df['WinRate'].rank(ascending=False, pct=True) * weights.get('winrate', 0.1)
        )
        df = df.sort_values('Score', ascending=False)
    
    # 添加排名
    df['Rank'] = range(1, len(df) + 1)
    
    return df


# ============================================================
# 主函数
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='策略评测与排名脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 从 CSV 文件读取并排名
  python evaluate_strategies.py --input summary.csv --output ranked_summary.csv
  
  # 使用自定义权重
  python evaluate_strategies.py --input summary.csv --calmar-weight 0.4 --sharpe-weight 0.3
  
  # 使用不同排名方法
  python evaluate_strategies.py --input summary.csv --method calmar_only
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='summary.csv',
        help='输入 CSV 文件路径（默认: summary.csv）'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='ranked_summary.csv',
        help='输出 CSV 文件路径（默认: ranked_summary.csv）'
    )
    
    parser.add_argument(
        '--method', '-m',
        type=str,
        choices=['weighted', 'calmar_only', 'sharpe_only', 'pareto'],
        default='weighted',
        help='排名方法（默认: weighted）'
    )
    
    parser.add_argument(
        '--calmar-weight',
        type=float,
        default=None,
        help='Calmar Ratio 权重（覆盖默认值）'
    )
    
    parser.add_argument(
        '--sharpe-weight',
        type=float,
        default=None,
        help='Sharpe Ratio 权重（覆盖默认值）'
    )
    
    parser.add_argument(
        '--use-zscore',
        action='store_true',
        help='使用 Z-score 标准化（默认使用 Min-Max）'
    )
    
    parser.add_argument(
        '--show-all',
        action='store_true',
        help='显示所有列（默认只显示关键指标）'
    )
    
    args = parser.parse_args()
    
    # 读取数据
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 错误：文件不存在: {input_path}")
        return
    
    print(f"📊 读取数据: {input_path}")
    df = pd.read_csv(input_path)
    print(f"   找到 {len(df)} 个策略\n")
    
    # 检查是否有可用于排名的列（不强制要求，会自动识别）
    print(f"📋 数据列: {list(df.columns)}")
    
    # 尝试识别关键指标
    cagr_col = find_column(df, ['CAGR', '年化收益率', '年化收益', '盈亏%', '盈亏'])
    mdd_col = find_column(df, ['MaxDD', 'MaxDD (%)', '最大回撤%', '最大回撤'])
    
    if not cagr_col or not mdd_col:
        print(f"\n⚠️  警告：未找到关键指标列")
        if not cagr_col:
            print(f"   - 未找到 CAGR/年化收益率/盈亏% 列")
        if not mdd_col:
            print(f"   - 未找到 MaxDD/最大回撤% 列")
        print(f"   脚本会尝试从现有列中计算指标，但可能影响排名准确性")
    else:
        print(f"   ✅ 找到关键指标: CAGR={cagr_col}, MaxDD={mdd_col}")
    
    # 更新权重（如果提供了自定义权重）
    weights = WEIGHTS.copy()
    if args.calmar_weight is not None:
        weights['calmar'] = args.calmar_weight
    if args.sharpe_weight is not None:
        weights['sharpe'] = args.sharpe_weight
    
    # 标准化权重（确保总和为 1）
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}
    
    print(f"📐 使用权重配置:")
    for key, value in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"   {key:15s}: {value:.2%}")
    print()
    
    # 排名
    print(f"🔍 使用排名方法: {args.method}")
    df_ranked = rank_strategies(df, weights=weights, method=args.method)
    
    # 显示结果
    print("\n" + "="*80)
    print("📈 策略排名结果")
    print("="*80)
    
    # 选择显示的列
    display_cols = ['Rank', 'name', 'Score']
    if 'name' not in df_ranked.columns and 'Name' in df_ranked.columns:
        display_cols[1] = 'Name'
    elif 'name' not in df_ranked.columns and 'Strategy' in df_ranked.columns:
        display_cols[1] = 'Strategy'
    
    # 添加关键指标
    key_metrics = ['CAGR', 'MaxDD', 'Sharpe', 'Calmar', 'WinRate']
    for metric in key_metrics:
        if metric in df_ranked.columns:
            display_cols.append(metric)
    
    if args.show_all:
        display_cols = df_ranked.columns.tolist()
    
    # 显示前 10 名
    print("\n🏆 Top 10 策略:")
    print(df_ranked[display_cols].head(10).to_string(index=False))
    
    # 保存结果
    output_path = Path(args.output)
    df_ranked.to_csv(output_path, index=False)
    print(f"\n✅ 结果已保存到: {output_path}")
    
    # 统计信息
    print("\n📊 统计信息:")
    print(f"   总策略数: {len(df_ranked)}")
    if 'Score' in df_ranked.columns:
        print(f"   最高分: {df_ranked['Score'].max():.4f}")
        print(f"   最低分: {df_ranked['Score'].min():.4f}")
        print(f"   平均分: {df_ranked['Score'].mean():.4f}")
    
    if 'Calmar' in df_ranked.columns:
        print(f"   最佳 Calmar: {df_ranked['Calmar'].max():.4f}")
    if 'Sharpe' in df_ranked.columns:
        print(f"   最佳 Sharpe: {df_ranked['Sharpe'].max():.4f}")


if __name__ == '__main__':
    main()
