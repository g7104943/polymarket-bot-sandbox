# 策略评测与排名脚本使用说明

## 功能特点

基于金融和数学理论的多维度策略评估系统：

1. **多指标综合评分**：Calmar Ratio、Sharpe Ratio、Sortino Ratio、最大回撤、胜率等
2. **灵活的权重配置**：可根据策略类型调整权重
3. **多种排名方法**：加权评分、单一指标、帕累托最优
4. **标准化处理**：Min-Max 或 Z-score 标准化

## 快速开始

### 1. 准备数据文件

将两个策略的数据合并到一个 CSV 文件中，包含以下列：

**必需列**：
- `name` 或 `Name`：策略名称
- `CAGR`：年化复合增长率（%）
- `MaxDD`：最大回撤（%）

**可选列**（如果有会提高评估准确性）：
- `Sharpe`：Sharpe Ratio
- `Sortino`：Sortino Ratio
- `WinRate`：胜率（%）
- `ProfitFactor` 或 `Profit_Factor`：盈亏比
- `Volatility`：波动率（%）

### 2. 基本使用

```bash
# 使用默认权重和排名方法
python scripts/evaluate_strategies.py --input summary.csv --output ranked_summary.csv
```

### 3. 自定义权重

```bash
# 更重视 Calmar Ratio 和 Sharpe Ratio
python scripts/evaluate_strategies.py \
    --input summary.csv \
    --calmar-weight 0.4 \
    --sharpe-weight 0.3 \
    --output ranked_summary.csv
```

### 4. 使用不同排名方法

```bash
# 仅使用 Calmar Ratio 排名
python scripts/evaluate_strategies.py \
    --input summary.csv \
    --method calmar_only \
    --output ranked_calmar.csv

# 仅使用 Sharpe Ratio 排名
python scripts/evaluate_strategies.py \
    --input summary.csv \
    --method sharpe_only \
    --output ranked_sharpe.csv
```

## 默认权重配置

```python
{
    'calmar': 0.30,      # Calmar Ratio：风险调整收益的核心指标
    'sharpe': 0.20,      # Sharpe Ratio：总风险调整收益
    'sortino': 0.15,     # Sortino Ratio：下行风险调整收益
    'mdd': 0.15,         # 最大回撤：风险控制能力
    'winrate': 0.10,     # 胜率：策略稳定性
    'profit_factor': 0.10, # 盈亏比：盈利效率
}
```

## 指标说明

### Calmar Ratio
- **公式**：CAGR / MaxDD
- **含义**：年化收益与最大回撤的比值，衡量风险调整后的收益
- **越高越好**：> 1 表示收益超过最大回撤

### Sharpe Ratio
- **含义**：超额收益与总风险的比值
- **越高越好**：> 1 为良好，> 2 为优秀

### Sortino Ratio
- **含义**：超额收益与下行风险的比值（只考虑负收益的波动）
- **越高越好**：通常比 Sharpe Ratio 更严格

### 最大回撤（MaxDD）
- **含义**：从峰值到谷值的最大跌幅
- **越小越好**：< 20% 为良好

## 示例：合并两个策略数据

```python
import pandas as pd

# 读取两个策略的数据
df_no_control = pd.read_excel("无风控.numbers", sheet_name="Sheet1")
df_dynamic = pd.read_excel("动态调整.numbers", sheet_name="Sheet1")

# 添加策略名称
df_no_control['name'] = '无风控'
df_dynamic['name'] = '动态调整'

# 合并
df_combined = pd.concat([df_no_control, df_dynamic], ignore_index=True)

# 保存
df_combined.to_csv("summary.csv", index=False)
```

## 输出说明

脚本会输出：
1. **排名结果**：按综合评分排序的策略列表
2. **Top 10 策略**：在终端显示前 10 名
3. **CSV 文件**：包含所有策略和排名的完整数据
4. **统计信息**：最高分、最低分、平均分等

## 高级选项

```bash
# 使用 Z-score 标准化（适用于数据分布接近正态分布）
python scripts/evaluate_strategies.py \
    --input summary.csv \
    --use-zscore \
    --output ranked_summary.csv

# 显示所有列
python scripts/evaluate_strategies.py \
    --input summary.csv \
    --show-all \
    --output ranked_summary.csv
```

## 权重调整建议

根据不同策略类型，可以调整权重：

### 保守型策略（重视风险控制）
```python
{
    'calmar': 0.25,
    'sharpe': 0.15,
    'sortino': 0.20,
    'mdd': 0.25,        # 更重视最大回撤
    'winrate': 0.10,
    'profit_factor': 0.05,
}
```

### 激进型策略（重视收益）
```python
{
    'calmar': 0.40,     # 更重视 Calmar
    'sharpe': 0.30,
    'sortino': 0.15,
    'mdd': 0.10,
    'winrate': 0.05,
    'profit_factor': 0.00,
}
```

### 平衡型策略（默认配置）
使用脚本中的默认权重即可。
