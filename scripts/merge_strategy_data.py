#!/usr/bin/env python3
"""
合并策略数据脚本
将多个策略的评测数据合并为一个 CSV 文件，用于排名

支持格式：
- Excel (.xlsx, .xls)
- Numbers (.numbers) - 使用 numbers-parser 库直接读取
- CSV (.csv)

特点：保留所有原始列，不进行重命名
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys

# 尝试导入 numbers-parser（可选依赖）
try:
    from numbers_parser import Document
    HAS_NUMBERS_PARSER = True
except ImportError:
    HAS_NUMBERS_PARSER = False

def read_data_file(file_path: Path) -> pd.DataFrame:
    """读取数据文件（支持多种格式）"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    suffix = file_path.suffix.lower()
    
    try:
        if suffix in ['.xlsx', '.xls']:
            # 尝试读取第一个工作表
            df = pd.read_excel(file_path, sheet_name=0)
        elif suffix == '.numbers':
            # 尝试使用 numbers-parser 直接读取
            if HAS_NUMBERS_PARSER:
                try:
                    print(f"   📊 使用 numbers-parser 读取 .numbers 文件...")
                    doc = Document(str(file_path))
                    dfs = []
                    for sheet in doc.sheets:
                        for table in sheet.tables:
                            # 获取表头和数据
                            rows = table.rows(values_only=True)
                            if len(rows) == 0:
                                continue
                            # 第一行作为列名
                            columns = rows[0]
                            # 其余行作为数据
                            data = rows[1:] if len(rows) > 1 else []
                            df_table = pd.DataFrame(data, columns=columns)
                            dfs.append(df_table)
                    
                    if len(dfs) == 0:
                        raise ValueError("Numbers 文件中没有找到数据表")
                    elif len(dfs) == 1:
                        df = dfs[0]
                    else:
                        # 多个表时，合并或使用第一个
                        print(f"   ⚠️  找到 {len(dfs)} 个表，使用第一个表")
                        df = dfs[0]
                    
                    print(f"   ✅ 成功读取 {len(df)} 行数据")
                    return df
                except Exception as e:
                    print(f"   ❌ 读取失败: {e}")
                    print(f"   💡 提示: 如果遇到问题，可以手动导出为 CSV 格式")
                    raise ValueError(f"读取 .numbers 文件失败: {e}")
            else:
                # 没有安装 numbers-parser
                print(f"\n⚠️  检测到 .numbers 文件: {file_path.name}")
                print(f"   .numbers 文件需要 numbers-parser 库才能直接读取")
                print(f"\n📦 安装方法:")
                print(f"   pip install numbers-parser")
                print(f"\n   或者手动转换为 CSV:")
                print(f"   1. 在 Numbers 中打开文件")
                print(f"   2. 文件 → 导出为 → CSV")
                print(f"   3. 保存为: {file_path.stem}.csv")
                raise ImportError(
                    "需要安装 numbers-parser 库才能读取 .numbers 文件\n"
                    "安装命令: pip install numbers-parser"
                )
        elif suffix == '.csv':
            df = pd.read_csv(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {suffix}")
        
        return df
    except Exception as e:
        raise ValueError(f"读取文件失败: {e}")


def find_file(file_path: str) -> Path:
    """查找文件（支持相对路径、桌面、当前目录）"""
    path = Path(file_path)
    
    # 如果是绝对路径且存在，直接返回
    if path.is_absolute() and path.exists():
        return path
    
    # 尝试在当前目录
    if path.exists():
        return path
    
    # 尝试在桌面
    desktop_path = Path.home() / "Desktop" / path.name
    if desktop_path.exists():
        return desktop_path
    
    # 尝试在项目根目录
    project_root = Path(__file__).parent.parent
    project_path = project_root / path.name
    if project_path.exists():
        return project_path
    
    # 如果都不存在，返回原始路径（让 read_data_file 处理错误）
    return path


def merge_strategies(
    files: list,
    strategy_names: list = None,
    output: str = 'summary.csv'
) -> pd.DataFrame:
    """
    合并多个策略数据
    保留所有原始列，不进行重命名
    """
    all_dfs = []
    
    for i, file_path in enumerate(files):
        # 查找文件
        actual_path = find_file(file_path)
        print(f"📖 读取: {file_path}")
        
        # 如果文件不存在，显示所有尝试过的路径
        if not actual_path.exists():
            desktop_path = Path.home() / "Desktop" / Path(file_path).name
            current_dir = Path.cwd() / Path(file_path).name
            project_root = Path(__file__).parent.parent / Path(file_path).name
            
            print(f"❌ 错误: 文件不存在")
            print(f"   已尝试查找以下位置:")
            print(f"   - 当前目录: {current_dir}")
            print(f"   - 桌面: {desktop_path}")
            print(f"   - 项目根目录: {project_root}")
            print(f"   - 绝对路径: {Path(file_path).resolve() if Path(file_path).is_absolute() else 'N/A'}")
            print(f"\n💡 提示:")
            print(f"   - 请确认文件名是否正确（包括大小写和扩展名）")
            print(f"   - 可以使用完整路径: ~/Desktop/无封控.csv")
            print(f"   - 或者先切换到文件所在目录")
            raise FileNotFoundError(f"无法找到文件: {file_path}")
        
        if str(actual_path) != file_path:
            print(f"   → 找到文件: {actual_path}")
        df = read_data_file(actual_path)
        
        # 保留所有原始列，不进行标准化
        # 只添加策略名称列（如果不存在）
        
        # 添加策略名称
        if strategy_names and i < len(strategy_names):
            strategy_name = strategy_names[i]
        else:
            strategy_name = Path(file_path).stem
        
        # 如果数据有多行，为每行添加策略名称
        if 'name' not in df.columns:
            df['name'] = strategy_name
        else:
            # 如果已有 name 列，但想覆盖
            df['name'] = strategy_name
        
        all_dfs.append(df)
        print(f"   ✓ 找到 {len(df)} 行数据，{len(df.columns)} 列")
    
    # 合并
    print(f"\n🔗 合并 {len(all_dfs)} 个策略...")
    df_merged = pd.concat(all_dfs, ignore_index=True)
    
    # 显示所有列
    print(f"\n📋 数据列（共 {len(df_merged.columns)} 列）:")
    for col in df_merged.columns:
        print(f"   - {col}")
    
    # 提示可能需要的列（但不强制）
    suggested_cols = ['CAGR', 'MaxDD', 'Sharpe', 'Sortino', 'WinRate', '盈亏%', '最大回撤%', '胜率%', 'Calmar']
    found_cols = [col for col in suggested_cols if col in df_merged.columns]
    if found_cols:
        print(f"\n✅ 找到可用于排名的列: {found_cols}")
    else:
        print(f"\n⚠️  未找到常见的排名指标列")
        print(f"   排名脚本会尝试从现有列中识别指标")
    
    # 保存
    output_path = Path(output)
    df_merged.to_csv(output_path, index=False)
    print(f"\n✅ 已保存到: {output_path}")
    print(f"   总策略数: {len(df_merged)}")
    
    return df_merged


def main():
    parser = argparse.ArgumentParser(
        description='合并策略数据用于排名（保留所有原始列）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 合并两个策略文件（.numbers 格式）
  python merge_strategy_data.py \\
      --files ~/Desktop/无封控.numbers ~/Desktop/动态调整.numbers \\
      --names "无封控" "动态调整" \\
      --output summary.csv
  
  # 从 CSV 文件合并
  python merge_strategy_data.py \\
      --files strategy1.csv strategy2.csv \\
      --output summary.csv
        """
    )
    
    parser.add_argument(
        '--files', '-f',
        nargs='+',
        required=True,
        help='要合并的文件路径（支持 .xlsx, .xls, .csv, .numbers）'
    )
    
    parser.add_argument(
        '--names', '-n',
        nargs='+',
        default=None,
        help='策略名称列表（可选，默认使用文件名）'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='summary.csv',
        help='输出 CSV 文件路径（默认: summary.csv）'
    )
    
    args = parser.parse_args()
    
    # 验证文件数量
    if args.names and len(args.names) != len(args.files):
        print(f"⚠️  警告：策略名称数量 ({len(args.names)}) 与文件数量 ({len(args.files)}) 不匹配")
        print(f"   将使用文件名作为策略名称")
        args.names = None
    
    try:
        df = merge_strategies(
            files=args.files,
            strategy_names=args.names,
            output=args.output
        )
        
        print(f"\n📊 数据预览（前 5 行）:")
        print(df.head().to_string(index=False))
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
