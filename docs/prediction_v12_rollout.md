# 15m 胜率/收益提升 V1.2 落地手册

## 1. 共存契约（已落地）
- 决策顺序（运行态过滤链）：
  - calibration -> threshold drift -> expectancy gate -> meta-label -> SHOCK -> comboPause -> UP/DOWN -> selector overlay -> should_trade -> executor cooldown
- 职责边界：
  - calibration：只改概率，不阻断、不改仓位。
  - threshold drift：只改阈值增量，不阻断、不改仓位。
  - expectancy：只负责 normal/degraded/blocked + 仓位缩放，不再承担阈值漂移。
  - meta-label：只决定单笔是否参与。
  - selector：只决定 trader+symbol 是否参与。
  - SHOCK/combo/cooldown：执行侧阻断/节流，不改预测概率。
- route 隔离：simulation 仅读 simulation 数据；live 仅读 live 数据。

## 2. 统一评估底座（离线）
- 脚本：`/Users/mac/Downloads/polyfun/scripts/ops/evaluate_prediction_v12_base.py`
- 默认输出：`/Users/mac/Downloads/polyfun/reports/prediction_v12_base_eval.json`
- 严格口径硬约束：
  - NetPnL 降幅 <= 5%
  - MDD 改善 >= 10%
  - WR drop <= 1pct
  - suppression <= 30%

示例：
```bash
python3 /Users/mac/Downloads/polyfun/scripts/ops/evaluate_prediction_v12_base.py
```

## 3. 审计与监测（全链路）
- 预测侧控制审计：
```bash
python3 /Users/mac/Downloads/polyfun/scripts/ops/audit_prediction_side_controls.py
```
- 总审计：
```bash
python3 /Users/mac/Downloads/polyfun/scripts/ops/audit_prediction_edge_total.py --output /Users/mac/Downloads/polyfun/reports/prediction_edge_total_audit_v12_latest.json
```
- 健康监测：
```bash
python3 /Users/mac/Downloads/polyfun/scripts/api_health_check.py --once
```

## 4. 受控重载（default + 70）
```bash
cd /Users/mac/Downloads/polyfun/polymarket && npm run build
cd /Users/mac/Downloads/polyfun && ./reload_launchctl_multi_trading.sh
cd /Users/mac/Downloads/polyfun && ./启动70组合.sh
```

## 5. 当前状态读取重点
- `api_health_check.py` 已显示：
  - DOWN/UP/SHOCK/COMBO/COOLDOWN
  - calibration / expectancy
  - threshold drift / meta-label / selector
  - cluster 分解
- `audit_prediction_edge_total.py` 已输出 runtime guard 新层覆盖：
  - `drift_cells`
  - `meta_cells`
  - `selector_cells`

## 6. 接入策略
- 新层仍按“一次一层”推进，不多层同时切 enforce。
- 不达标簇必须进入修复闭环：参数二次网格 -> 结构简化 -> 跨簇收缩 -> 替代方案。
