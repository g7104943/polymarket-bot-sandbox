# 报告与 prediction_trades 数据流说明

## 数据来源

| 数据 | 来源 | 说明 |
|------|------|------|
| **report_summary.txt / report_summary.json** | 运行中进程内存 | `PredictionExecutor.generateSummaryReport()` 用 `this.logger.getAllEntries()` 生成，每小时整点写入 `logs_*/reports/` |
| **prediction_trades.json** | 同进程的 TradeLogger | 每次 `addEntry` / `updateEntry` / `settleTrade` 会 `save()` 写回文件；**进程启动时** 会先 `load()` 再 `cleanup()` 再 `save()` |

## 为何会出现「报告 211 笔 / $476，aggregate 196 笔 / $336」？

- **报告**：来自**上次写报告时**进程内存里的全部交易（例如 211 笔），写的是当时快照。
- **prediction_trades.json**：进程**每次启动**时，TradeLogger 会：
  1. 从文件 `load()` 全部记录；
  2. **cleanup()** 只保留「最近 10 天」的记录（`LOG_RETENTION_DAYS = 10`，见 `src/models/trade_log.ts`）；
  3. 若有删除则 **save()** 覆盖写回文件。

因此：**一旦进程重启**，文件会被截断为「仅最近 10 天」，例如 211 条变成 196 条；而磁盘上的 report_summary 仍是重启前最后一次写入的完整快照（211 笔、$476）。aggregate 若只读 prediction_trades.json，就会得到 196 笔和更少的资金。

## 当前约定

- **aggregate_avg_buy_price.py 的「按最终资金排名」**：优先使用 `reports/report_summary.json` 里的 `summary`（总笔数、胜/负、总盈亏、当前资金），与报告一致；若无报告或解析失败，再回退到 `prediction_trades.json`。
- **按币对汇总**（平均买入价等）：仍只读 `prediction_trades.json`（且只统计有 tokenPrice 的已执行单），因为报告里没有逐笔 tokenPrice。

## 如需报告与文件长期一致

可考虑（按需选做）：

1. **取消或放宽 10 天清理**：修改 `trade_log.ts` 的 `LOG_RETENTION_DAYS`，或改为只清理「超过 N 天」且「已结算」的记录，避免未结算被删。
2. **报告也以文件为准**：在写 report 前用 `this.logger.getAllEntries()` 与当前文件内容对齐（例如以文件为主、内存只保留文件 + 本次运行新增），需改 PredictionExecutor / TradeLogger 的协作方式。
