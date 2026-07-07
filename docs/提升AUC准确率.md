# 提升模型 AUC / 准确率：可行方向与优先级

当前 AUC ≈ 0.54～0.57，有提升空间。下面按**易做、性价比高**到**需改代码、难度大**排序，并给出具体改法和命令。

---

## 一、标签设计（改参数即可，优先试）

### 1. 增大 lookahead：1 → 3 或 5

- **现象**：`lookahead=1` 时，标签 = 下一根涨跌，单根噪音大，难学。
- **做法**：用「未来 3 根或 5 根的整体方向」做标签，更平滑、更稳定。
- **命令**：训练时加 `--lookahead 3` 或 `--lookahead 5`。

```bash
python -m src.python.model_trainer --per-symbol-timeframe --use-cv --holdout-days 90 --lookahead 3
```

- **注意**：预测逻辑上仍是「用最后一根特征 → 模型输出」，但**训练时的标签**变成了「未来 3 根/5 根涨跌」。和 Polymarket 单根 15m 的对应关系：可理解为「未来 3 根里多数向上则 UP」，需在回测/实盘时自己统一口径。

---

### 2. 涨跌幅阈值：过滤横盘

- **现象**：涨跌 < 0.05% 的「横盘」被标成 UP/DOWN，增加噪音。
- **做法**：只有涨跌幅 **超过阈值** 的才标为 1/0，其余样本训练时过滤（`build_labels` 会产出 NaN，`prepare_train_data` 会滤掉）。
- **命令**：例如 `--label-threshold 0.1` 表示 0.1%。

```bash
python -m src.python.model_trainer --per-symbol-timeframe --use-cv --holdout-days 90 --lookahead 1 --label-threshold 0.2
```

- ** trade-off**：样本变少，但标签更「干净」。0.05～0.2 都可试；0.1 是常用起点。

---

## 二、特征：减噪与增强

### 3. 按重要性做特征筛选（需加一小段脚本）

- **现象**：100+ 特征里，部分与标签几乎无关，反而增加过拟合。
- **做法**：训练完后用 `model.feature_importance()` 或 `lgb.plot_importance`，剔除 importance 最低的 20%～30%，只保留剩下的特征重新训练。
- **已实现**（`model_trainer` + `config/feature_allowlist.txt`）：
  1. **第一轮**：加 `--feature-drop-bottom-pct 0.3`，按 gain 剔掉后 30%，把保留的特征写入 `config/feature_allowlist.txt`；
  2. **第二轮**：加 `--use-feature-allowlist`，只使用该文件中的特征训练。
- **示例**（建议先跑 Optuna 超参，再第一轮、第二轮）：
  ```bash
  # 0. 先超参（可选，结果写入 lightgbm_params_optuna.json，后续会自动用）
  python -m src.python.model_trainer --optuna --optuna-trials 10000
  # 1. 第一轮：生成 feature_allowlist.txt
  python -m src.python.model_trainer --per-symbol-timeframe --use-cv --holdout-days 90 --lookahead 1 --label-threshold 0.2 --feature-drop-bottom-pct 0.3
  # 2. 第二轮：用 allowlist 训练
  python -m src.python.model_trainer --per-symbol-timeframe --use-cv --holdout-days 90 --lookahead 1 --label-threshold 0.2 --use-feature-allowlist
  ```

---

### 4. 多时间框架特征（需改特征与数据管线）

- **思路**：在 15m 上预测时，额外加入 1h、4h 的「状态」：例如 1h 的 `ema_10_20_diff`、4h 的 `rang_osc`，按时间对齐到 15m 的每一行。
- **做法**：对 1h、4h 的 parquet 做 `build_features`，再按 `timestamp` 重采样/向前填充到 15m 的 `date`，merge 到 15m 的 `build_features` 结果上，作为额外列。只加 5～10 个高层 summary 特征即可，避免过多。
- **收益**：趋势、结构信息更丰富，有利于 AUC；实现成本中等。

---

## 三、训练过程：防泄漏、更稳的验证

### 5. Purging / Embargo（需改 `TimeSeriesSplit` 使用方式）

- **问题**：`TimeSeriesSplit` 的 train/val 是紧挨着的。你有 `close_lag_1`～`close_lag_10` 和 `lookahead`：val 的第一行在时间上只比 train 最后一行晚 1 根，滞后与 lookahead 可能导致轻微的信息渗透。
- **做法**：在切分时做 **purging**：从 train 末尾删掉 `max( lookahead, max_lag )` 行（例如 10）；并在 val 开头做 **embargo**：删掉同样行数。这样 train 和 val 之间有一小段「空白」。
- **实现**：在 `model_trainer` 的 `train_with_cv` 里，对 `tscv.split(X)` 得到的 `train_idx, val_idx` 做二次裁剪：  
  `train_idx = train_idx[:- purge]`，`val_idx = val_idx[embargo:]`（purge/embargo 取 10～15 可作起点）。

---

### 6. 增加 CV 折数、早停与 Optuna

- **CV**：`train_with_cv(..., n_splits=10)` 可提高验证稳定性，看到更稳的 `mean_auc`/`std_auc`。
- **early_stopping_rounds**：50 可尝试调到 80～100，让树多学几轮，观察是否欠拟合（若 val 持续变差则不必加）。
- **Optuna**：在现有 `lightgbm_params_optuna.json` 基础上，再跑一轮 **更多 trials**（500～1000），或把 `n_estimators`、`num_leaves`、`max_depth` 的搜索范围略放大，往往能挤出一点 AUC。

```bash
python -m src.python.model_trainer --optuna --optuna-trials 500 --per-symbol-timeframe --use-cv --holdout-days 90 --lookahead 3
```

---

## 四、类别不平衡（若 UP/DOWN 明显不均）

### 7. `scale_pos_weight`

- **现象**：若 `y.mean()` 明显偏离 0.5（例如 0.35 或 0.65），模型会偏向多数类。
- **做法**：LightGBM 的 `scale_pos_weight = (1 - y.mean()) / y.mean()`，在 `train_one` / `train_with_cv` 里传入 `params` 或 `lgb.train` 的 `callbacks` 前设好。  
- **查看**：`metadata` 里的 `up_ratio_true` 若长期 < 0.45 或 > 0.55，可启用。

---

## 五、集成与使用方式（不改训练，只改用法）

### 8. 多模型投票 / 平均概率

- **做法**：同一 symbol+timeframe，训练 2～3 个不同随机种子或不同 `lookahead` 的模型；预测时对 `prob` 做平均，或用「多数投票」决定 UP/DOWN。  
- **实现**：在 `predictor` 或 `prediction_writer` 里，对同一 (symbol, tf) 加载多个 `model.joblib`，分别 `predict` 后 `prob_mean = probs.mean()`，再 `pred = 1 if prob_mean >= 0.5 else 0`。

---

### 9. 提高下单阈值，减少低置信度交易

- **做法**：你已有 `PROB_THRESHOLD`。可把 0.5 提到 **0.55～0.6**，只做「更自信」的预测，往往能提高实盘/回测的胜率，但交易次数会变少。  
- **配置**：`.env` 里 `PROB_THRESHOLD=0.58` 等，按回测曲线自己调。

---

## 六、推荐实施顺序

| 顺序 | 项目 | 类型 | 预期 |
|------|------|------|------|
| 1 | `--lookahead 3` 或 5 | 改命令行 | 标签更平滑，AUC 常能 +0.01～0.02 |
| 2 | `--label-threshold 0.1` | 改命令行 | 过滤横盘，减少噪音 |
| 3 | Optuna `--optuna-trials 500` | 改命令行 | 超参更优，再挤一点 AUC |
| 4 | 特征筛选（importance 剔尾） | 加小脚本 | 减过拟合，方差可能降 |
| 5 | Purging/Embargo | 改 `model_trainer` | 验证更干净，防轻微泄漏 |
| 6 | `scale_pos_weight` | 改 `model_trainer` | 仅在不平衡时明显 |
| 7 | 多时间框架特征 | 改 `feature_engineering` + 数据 | 潜力大，实现量也大 |
| 8 | 多模型集成 | 改 `predictor` / 预测流程 | 提升稳定性与胜率 |

**5. Purging/Embargo**（已实现）  
- `--purge-embargo 10`：train 末 / val 头各删 10 行，0=关闭  
- 默认 10，可与现有命令一起用  

**6. scale_pos_weight**（已实现）  
- 当 `y.mean()` 不在 [0.45, 0.55] 时自动设 `scale_pos_weight = (1-p)/p`  
- 默认开启，`--no-scale-pos-weight` 关闭  

**7. 多时间框架特征**（已实现）  
- `--add-multi-timeframe`：15m 时合并 1h、4h 的 `ema_10_20_diff`、`rang_osc`、`kdj_k`、`rsi`、`bb_position`（`mtf_1h_*`、`mtf_4h_*`）  
- 需有 `data/raw/{symbol}_1h.parquet`、`data/raw/{symbol}_4h.parquet`  

**8. 多模型集成**（已实现）  
- 预测时：`python -m src.python.prediction_writer --once --ensemble --ensemble-max 3`  
- 对同一 (symbol, tf) 的多个模型 prob 取平均后再判 UP/DOWN  

**示例：训练时同时开 5+6+7**

```bash
python -m src.python.model_trainer --per-symbol-timeframe --use-cv --holdout-days 90 --lookahead 1 --label-threshold 0.2 \
  --purge-embargo 10 --add-multi-timeframe
# 默认已开 scale_pos_weight；关闭用 --no-scale-pos-weight
```

---

## 七、一句命令：优先尝试组合

先用「标签 + 阈值 + 更多 Optuna」做一轮，看 AUC 与回测：

```bash
python -m src.python.model_trainer --optuna --optuna-trials 500 --per-symbol-timeframe --use-cv --holdout-days 90 --lookahead 3 --label-threshold 0.1
```

训练完成后，看各 `metadata.json` 的 `mean_auc`、`final_metrics.auc`，再跑 `scripts/backtest_simulation.py` 对比之前 `--lookahead 1`、无 `--label-threshold` 的结果。若 15m 的 AUC 能到 0.58～0.60，再考虑特征选择、purging 和多时间框架。
