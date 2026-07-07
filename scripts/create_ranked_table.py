#!/usr/bin/env python3
"""
创建排名表格脚本
将排名结果格式化为表格并保存到桌面（Excel 格式）
保持与原始表格相似的格式
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import sys

def create_ranked_table(
    input_file: str,
    output_file: str = None,
    format: str = 'excel'
) -> pd.DataFrame:
    """
    创建格式化的排名表格
    
    Args:
        input_file: 排名结果 CSV 文件
        output_file: 输出文件路径（默认保存到桌面）
        format: 输出格式 ('excel' 或 'csv')
    """
    # 读取排名结果
    df = pd.read_csv(input_file)
    
    # 确保有 Rank 列
    if 'Rank' not in df.columns and 'rank' not in df.columns:
        df['Rank'] = range(1, len(df) + 1)
    
    # 按照原始表格的列顺序排列（参考 summary.csv）
    # 原始顺序：排名,币种,周期,交易次数,胜场,败场,胜率%,最终资金,盈亏%,最大回撤%,最低/初始%,模型,阈值,策略过滤,组合,name
    
    # 定义原始表格的列顺序（只保留原始列，不显示重复的计算列）
    original_cols = [
        '排名',  # 排名列（会更新为新排名）
        '币种',
        '周期',
        '交易次数',
        '胜场',
        '败场',
        '胜率%',
        '最终资金',
        '盈亏%',
        '最大回撤%',
        '最低/初始%',
        '模型',
        '阈值',
        '策略过滤',
        '组合',
        'name',
    ]
    
    # 新增的计算列（添加到后面）
    new_cols = [
        'Score',  # 综合评分
        'Calmar',  # Calmar Ratio
    ]
    
    # 构建显示列顺序
    display_cols = []
    
    # 1. 按原始顺序添加原始列（排除重复的计算列）
    for col in original_cols:
        if col in df.columns:
            display_cols.append(col)
    
    # 2. 添加新增的计算列（如果存在）
    for col in new_cols:
        if col in df.columns and col not in display_cols:
            display_cols.append(col)
    
    # 3. 添加其他未列出的原始列（排除重复的计算列）
    exclude_cols = ['CAGR', 'MaxDD', 'WinRate', 'Sharpe', 'Sortino', 'Calmar', 'Score', 'Rank']
    for col in df.columns:
        if col not in display_cols and col not in exclude_cols:
            display_cols.append(col)
    
    # 创建显示用的数据框
    df_display = df[display_cols].copy()
    
    # 更新"排名"列为新的全局排名（使用Rank列的值）
    if 'Rank' in df.columns and '排名' in df_display.columns:
        # 直接用Rank列的值替换"排名"列
        df_display['排名'] = df['Rank'].values
    
    # 只格式化新增的计算列（Rank, Score, Calmar），保留原始列的原始格式
    if 'Rank' in df_display.columns:
        df_display['Rank'] = df_display['Rank'].apply(lambda x: f"{int(x)}" if pd.notna(x) else "")
    
    if 'Score' in df_display.columns:
        df_display['Score'] = df_display['Score'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    
    if 'Calmar' in df_display.columns:
        df_display['Calmar'] = df_display['Calmar'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    
    # 格式化"排名"列为整数（确保使用新的排名值）
    if '排名' in df_display.columns:
        # 如果Rank列存在，使用Rank的值
        if 'Rank' in df.columns:
            df_display['排名'] = df['Rank'].apply(lambda x: f"{int(x)}" if pd.notna(x) else "")
        else:
            # 否则使用当前值
            df_display['排名'] = df_display['排名'].apply(
                lambda x: f"{int(float(x))}" if pd.notna(x) and str(x) != '' and str(x) != 'nan' else ""
            )
    
    # 确定输出路径
    if output_file is None:
        desktop = Path.home() / "Desktop"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if format == 'excel':
            output_file = desktop / f"策略排名_{timestamp}.xlsx"
        else:
            output_file = desktop / f"策略排名_{timestamp}.csv"
    else:
        output_file = Path(output_file)
    
    # 保存文件
    if format == 'excel':
        # 使用 Excel 格式，添加格式设置
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df_display.to_excel(writer, sheet_name='策略排名', index=False)
            
            # 获取工作表
            worksheet = writer.sheets['策略排名']
            
            # 设置列宽（自动调整）
            for idx, col in enumerate(df_display.columns):
                max_length = max(
                    df_display[col].astype(str).map(len).max(),
                    len(str(col))
                ) + 2
                worksheet.column_dimensions[chr(65 + idx)].width = min(max_length, 50)
            
            # 冻结首行
            worksheet.freeze_panes = 'A2'
            
            # 设置首行样式（如果需要）
            from openpyxl.styles import Font, PatternFill, Alignment
            
            header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
            header_font = Font(bold=True, color="FFFFFF")
            
            for cell in worksheet[1]:
                cell.fill = header_fill
                cell.font = header_font
                cell.alignment = Alignment(horizontal='center', vertical='center')
        
        print(f"✅ Excel 表格已保存到: {output_file}")
    else:
        df_display.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"✅ CSV 表格已保存到: {output_file}")
    
    # 显示预览
    print(f"\n📊 排名表格预览（前 10 名）:")
    print("=" * 100)
    print(df_display.head(10).to_string(index=False))
    print("=" * 100)
    
    return df_display


def main():
    parser = argparse.ArgumentParser(
        description='创建格式化的排名表格',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 从排名结果创建 Excel 表格（保存到桌面）
  python create_ranked_table.py --input ranked_summary.csv
  
  # 指定输出文件
  python create_ranked_table.py --input ranked_summary.csv --output ~/Desktop/策略排名.xlsx
  
  # 创建 CSV 格式
  python create_ranked_table.py --input ranked_summary.csv --format csv
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='ranked_summary.csv',
        help='排名结果 CSV 文件（默认: ranked_summary.csv）'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='输出文件路径（默认: 桌面/策略排名_时间戳.xlsx）'
    )
    
    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['excel', 'csv'],
        default='excel',
        help='输出格式（默认: excel）'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 错误：文件不存在: {input_path}")
        print(f"💡 提示：请先运行排名脚本生成 ranked_summary.csv")
        sys.exit(1)
    
    try:
        df = create_ranked_table(
            input_file=str(input_path),
            output_file=args.output,
            format=args.format
        )
        
        print(f"\n✅ 完成！表格已创建，共 {len(df)} 行数据")
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
