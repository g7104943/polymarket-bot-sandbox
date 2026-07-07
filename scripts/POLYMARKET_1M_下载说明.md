# Polymarket 1 分钟数据下载说明

回测「反转次数」需使用 Polymarket CLOB 的 1m 价格数据，见 [Historical Timeseries Data](https://docs.polymarket.com/developers/CLOB/timeseries)。

**数据来源与对齐（避免分叉/错位）：** 15m 数据来自项目现有 `data/raw/{asset}_15m.parquet`（交易所 K 线），回测 30 天由 `--after-date` + `--year-days 30` 决定；1m 数据仅用 Polymarket CLOB prices-history。**1m 的 15m_start_ts 必须与 15m parquet 每根 K 线起始时间（Unix 秒）一致**，否则会错位。**推荐一键脚本**：`scripts/fetch_eth_1m_polymarket.py` 自动用 Gamma API（按 slug 查询）取 Token ID 并拉取 1m，与 15m 严格对齐。

## API 端点

**GET** `https://clob.polymarket.com/prices-history`

| 参数 | 说明 |
|------|------|
| **market** | Token ID（必需） |
| **startTs** | 开始时间（Unix 秒） |
| **endTs** | 结束时间（Unix 秒） |
| **fidelity** | 分辨率，设为 **1** 表示 1 分钟 |
| **interval** | 可选：1m、1h、6h、1d、1w、max |

返回格式示例：

```json
{
  "history": [
    {"t": 1697875200, "p": 0.55},
    {"t": 1697875260, "p": 0.56}
  ]
}
```

## 如何获取 Token ID

### 方法 1：Gamma API

```bash
# 获取所有活跃市场
curl "https://gamma-api.polymarket.com/events?active=true&closed=false"

# 按 slug 查询特定市场
curl "https://gamma-api.polymarket.com/markets?slug=your-market-slug"
```

响应中的 **clobTokenIds** 字段包含 Token ID。

### 方法 2：CLOB Client

```javascript
const markets = await client.getMarkets();
markets.data.forEach(market => {
  market.tokens.forEach(token => {
    console.log(`${token.outcome}: ${token.token_id}`);
  });
});
```

每个市场的 `tokens` 数组里包含 `token_id`（UP/Yes 对应一个，DOWN/No 对应一个）。

### 方法 3：通过 Condition ID

```javascript
const market = await client.getMarket("CONDITION_ID");
// market.tokens[0].token_id = Yes token
// market.tokens[1].token_id = No token
```

Token ID 为长数字字符串，例如：  
`34097058504275310827233323421517291090691602969494795225921954353603704046623`

## 本仓库用法（保证 1m 与 15m 对齐）

### 推荐：一键拉取（Token ID 由 Gamma API 按 slug 获取）

与回测使用**相同**的 `--after-date` 和 `--year-days`（如 30 天）运行：

```bash
python scripts/fetch_eth_1m_polymarket.py --asset ETH_USDT --auto-after-date --year-days 30
```

脚本会：  
1）从 `data/raw/eth_usdt_15m.parquet` 取与回测一致的 15m 起始时间（Unix 秒）；  
2）对每个 15m 用 **Gamma API 按 slug 查询**：`GET https://gamma-api.polymarket.com/events/slug/eth-updown-15m-{15m_start_ts}`，从响应的 `markets[0].clobTokenIds` 取第一个为 UP token；  
3）对每个 (start_ts, token_id) 请求 **CLOB prices-history**：`GET https://clob.polymarket.com/prices-history?market=token_id&startTs=&endTs=&fidelity=1`；  
4）写入 `data/polymarket_1m_eth_usdt.parquet`（列：15m_start_ts, t_sec, p），与 15m 严格对齐。

回测脚本会自动读取该 parquet 计算「反转次数」；无该文件时反转次数为 0。

### 备选：手动填 Token ID 再拉 1m

1. 导出 15m 时间戳模板：`python scripts/export_15m_timestamps_for_1m.py --asset ETH_USDT --auto-after-date --year-days 30`  
2. 将模板中的 `token_id` 填为 Polymarket 该周期 UP 的 CLOB Token ID（见上文「如何获取 Token ID」），另存为 `data/polymarket_15m_up_token_ids.json`  
3. 拉取 1m：`python scripts/download_polymarket_1m.py --asset ETH_USDT`

文档还提供 [Google Colab Notebook](https://colab.research.google.com/drive/1s4TCOR4K7fRP7EwAH1YmOactMakx24Cs) 用于可视化此数据。

---

## 是否需要重跑一键训练脚本

- **若只想得到 30 天回测结果**：不必重跑训练。在项目根目录只重跑回测即可：
  ```bash
  python scripts/backtest_gru_regime_report.py \
    --no-1h4h --capital-400-smart --auto-after-date --asset ETH_USDT \
    --order-price 0.53 --stop-loss-exit-price 0.07 --compare-stop-loss \
    --threshold-start 0.51 --threshold-end 0.98 --year-days 30 \
    --output data/reports/gru_regime_backtest_ranking_eth_no1h4h.csv
  ```
- **若想完整重跑（含训练 + 导出 + 回测）**：执行 `./一键训练ETH_无1h4h_随机30.sh` 即可，回测阶段已使用 `--year-days 30`。
