# SOL GRU 完整流程（与 ETH/BTC/XRP 一致）

随机搜索跑完后，按下面三步：**导出最佳模型 → 回测选阈值 →（可选）启动预测+模拟交易**。

---

## 1. 导出最佳模型

在**实验目录**执行，把 SOL 的 best_experiment 复制到 `outputs/models_best/SOL_USDT/`：

```bash
cd /Users/mac/Downloads/polyfun/experiments/gru_regime_v1
python scripts/export_best_models.py
```

- 会从 `outputs/metrics/SOL_USDT/random_*_results.json` 读 `best_experiment`（当前是 run_id `20260201_094939`），复制：
  - `encoder.pt`
  - `lightgbm_with_embedding.joblib`
  - `encoder_config.json`
- 导出后路径：`experiments/gru_regime_v1/outputs/models_best/SOL_USDT/`

**查看当前最佳 run_id：**

```bash
cd experiments/gru_regime_v1
python scripts/show_best_run_ids.py
```

---

## 2. 回测（阈值 0.51～0.98）

在**项目根目录**执行，生成 SOL 回测排名 CSV：

```bash
cd /Users/mac/Downloads/polyfun
./一键回测GRU_SOL_0.51-0.98.sh
```

- 依赖：`experiments/gru_regime_v1/outputs/models_best/SOL_USDT/` 已存在，且 `data/raw/sol_usdt_15m.parquet` 存在。
- 输出：`data/reports/gru_regime_backtest_ranking_sol_usdt.csv`（不同阈值下的胜率、交易数、最终资金等，用于选阈值）。

等价命令：

```bash
python3 scripts/backtest_gru_regime_report.py \
  --auto-after-date \
  --asset SOL_USDT \
  --threshold-start 0.51 \
  --threshold-end 0.98 \
  --threshold-step 0.01 \
  --output data/reports/gru_regime_backtest_ranking_sol_usdt.csv
```

---

## 3. 启动 SOL 预测 + 模拟交易（可选）

选定阈值后，在**项目根目录**启动（与 GRU_ETH_55 同流程）：

```bash
cd /Users/mac/Downloads/polyfun
./启动GRU新模型_SOL.sh
```

- 会启动：Python 预测写入器（写 `polymarket/predictions_gru_sol.json`）+ npm 交易进程（读该文件，阈值 0.55，可改脚本里的 `PROB_THRESHOLD`）。
- 日志：`polymarket/logs_gru_sol/`（prediction_writer_stdout.log、trading_stdout.log 等）。

**注意**：Polymarket 需有 SOL 的 15m 市场；若没有，只能回测不能实盘/模拟。

---

## 流程小结

| 步骤 | 在哪执行 | 命令 |
|------|----------|------|
| 1. 导出最佳模型 | `experiments/gru_regime_v1` | `python scripts/export_best_models.py` |
| 2. 回测选阈值 | 项目根目录 | `./一键回测GRU_SOL_0.51-0.98.sh` |
| 3. 启动预测+交易 | 项目根目录 | `./启动GRU新模型_SOL.sh` |

与 ETH（GRU_ETH_55）一致：先导出 best → 回测看阈值 → 再决定是否启动实盘/模拟。
