# Polymarket 15m 模拟交易 — 数学与逻辑审计报告

## 一、Polymarket 二元市场基础力学

### 1.1 结算规则（官方）

- **价格 p**：代币市价 = 隐含概率（$0.50 = 50%）
- **买入**：花 $A，获得 `A/p` 份代币（扣手续费后为 `A/p * (1-fee)`）
- **赢**：每份兑 $1 → 收回 `A/p * (1-fee)` 美元，PnL = `A/p*(1-fee) - A`
- **输**：代币归零 → PnL = `-A`

**重要**：手续费在**买入时**从份额中扣除，输时**不再**额外扣费，亏损 = 本金。

### 1.2 赔率与 Edge

- **odds** = (1-p)/p：每 $1 下注若赢，净利（美元）
- **edge** = p_true × odds - (1 - p_true) = p×odds - q
- **Kelly** = (p×odds - q) / odds = edge / odds

与 Polymarket 二元结算一致，数学正确。

---

## 二、各模块公式审计

### 2.1 prediction_executor.ts ✅ 正确

| 公式 | 实现 | 判定 |
|------|------|------|
| odds | `(1-price)/price` | ✅ |
| edge | `conf * odds - (1-conf)` | ✅ |
| Kelly | `(conf*odds - (1-conf)) / odds` | ✅ |
| 赢 PnL | `tokensAfterFee - amount`，`tokensAfterFee = amount/price * (1-feeRate)` | ✅ |
| 输 PnL | `-amount` | ✅ |
| 手续费 | `0.25 * (p*(1-p))^2`（15m crypto） | ✅ |

**结算展示**（2026-02 核对）：赢局日志原显示「获得: X 代币 = $X」使用扣费前代币数，与实际入账（`tokensAfterFee`）不一致；资金与 PnL 计算一直正确。已改为显示「X 代币(扣费后) = $X」与 `tokensAfterFee` 一致。

**Polymarket 实际规则**：官方对 15m crypto 等市场收取 taker 费，有效费率约 1.56%@p=0.50。费用在**买入时从份额中扣除**（你拿到的是更少的代币），结算时每份兑 $1，故到账金额 = 扣费后代币数。因此「29.61 代币 = $29.61」不对（29.61 为扣费前代币数）；正确应为扣费后代币与金额一致，与当前实现一致。

**Edge/置信度过滤验算**：`edgeAtLimit = conf * (1-limitPrice)/limitPrice - (1-conf)`；limit=$0.50 时即 `2*conf - 1`。例：conf=50.55% → edge=1.1%；若 MIN_EDGE=1.66%，则 1.1% < 1.66% 正确跳过。置信度阈值在别处：预测侧「符合下单条件 (置信度≥阈值)」与执行侧 PROB_THRESHOLD 一致，属正常过滤。

**手续费不重复扣**：下单时只扣 `amount`（本金），不另扣手续费。赢局结算只加 `tokensAfterFee`（已含费后份额），无第二次扣费。输局仅记 pnl=-amount，资金已在下单时扣过。全流程费只体现在「赢时到账 = 扣费后」这一处，无重复扣费。

**全组合/全路径核对**（2026-02）：
- **prediction_executor.ts**：主结算、止损卖出、资金恢复（_recomputeCapitalFromHistory / _recomputePeakCapital）、报告重算资金，均使用 `tokensAfterFee = amount/price * (1 - polymarketTakerFeeRate(price))` 与赢 pnl = tokensAfterFee - amount；输 pnl = -amount。✅
- **scripts/settle_pending_trades.ts**：手动结算脚本原先赢局用 `pnl = tokensOwned - amount`（未扣费），已改为与 executor 一致：`feeRate = 0.25*(p*(1-p))^2`，`tokensAfterFee = tokensOwned*(1-feeRate)`，`pnl = tokensAfterFee - amount`。✅
- **postonly_tracker.ts**：影子订单按 Maker 0% 费 + rebate 计算，与主路径 Taker 结算分离，仅用于展示。✅
- **Edge 公式统一**：executePrediction（单笔）、executeBatchLimit（批量限价）均用 `edgeAtLimit = conf * (1-limitPrice)/limitPrice - (1-conf)`，与 MIN_EDGE 比较；阶梯路径仅用中位价做 Kelly 与展示，不做 edge 过滤。置信度过滤用 PROB_THRESHOLD（及双阈值）与 prediction_api 一致。✅

### 2.2 optimize_trading_rules.py ⚠️ 输局扣费错误

```python
# 当前实现
if correct:
    pnl = bet_amount * odds_this - bet_amount * settlement_cost  # 赢：正确
else:
    pnl = -(bet_amount + bet_amount * settlement_cost)           # 输：错误
```

**问题**：Polymarket 输时仅亏本金，不另扣手续费。`settlement_cost` 只应在赢局扣除。

**修正**：输局 `pnl = -bet_amount`。

### 2.3 hyperparam_tune_13_combos.py ⚠️ 同构错误

```python
if t.get("result") == "win":
    new_pnl = bet * (1.0/price - 1.0) - fee   # 正确
else:
    new_pnl = -bet - fee   # 错误：输时不应扣 fee
```

**修正**：输局 `new_pnl = -bet`。

### 2.4 rolling_backtest_limit_price.py ⚠️ 同构错误

```python
pnl_per = np.where(
    ic_filled == 1,
    bet * (1.0/fp_filled - 1.0 - TOTAL_COST),   # 赢：可接受
    -bet * (1.0 + TOTAL_COST),                   # 输：错误
)
```

**修正**：输局 `-bet`。

### 2.5 test_limit_order_dryrun.ts ⚠️ Edge 公式与主路径不一致

```typescript
// 测试用 totalCost、netP
const netP1 = conf1 - totalCost / 2;
const edge1 = netP1 * odds1 - (1 - netP1) - totalCost;
```

主流程 Edge 不含费率（`edge = conf*odds - (1-conf)`），测试文件混入了费用项，易误导。建议测试改为与主流程一致的 Edge 公式，或明确标注为「含费版本」。

### 2.6 run_grid.py 的 pnl 指标

```python
pnl = correct * 2 - 1 - 0.001  # 简化的方向正确性指标
```

用于 Optuna 的简化 PnL（正确/错误 + 固定成本），与 Polymarket 实际 PnL 不同，但仅用于超参搜索排序，可接受。若需与实盘更一致，应改用真实 PnL 公式。

---

## 三、逻辑与一致性检查

### 3.1 Edge 过滤价格 ✅（已修复）

- **executeLimitOrder**：Edge 使用 `limitPrice`，与限价单执行价一致 ✅
- **executePrediction**：Edge 使用 `limitPrice` ✅

### 3.2 Kelly 下注价格

- **calculateBetAmount**：Kelly 使用 `tokenPrice`（bestAsk 或成交价）
- 限价单场景下，成交价通常 ≤ limitPrice，用 bestAsk 作 Kelly 输入合理；若用 limitPrice 会更保守，当前做法可接受。

### 3.3 calcWeightedAvgFill

- 按 `price <= limitPrice` 从低到高消耗 asks
- `consumeUsd = min(ask.sizeUsd, remaining)`：按美元消耗
- `totalShares += consumeUsd / ask.price`：份额计算正确
- `avgPrice = totalCost / totalShares`：加权均价正确 ✅

### 3.4 normalizeAsks

- `sizeUsd = size * price`：若 `size` 为代币数量，则 `sizeUsd` 为美元
- Polymarket CLOB 通常返回 `size` 为份额，该转换合理 ✅

---

## 四、已修复汇总

| 文件 | 问题 | 修复 |
|------|------|------|
| optimize_trading_rules.py | 输局多加 settlement_cost | 输局 pnl = -bet_amount ✅ |
| optimize_trading_rules_3to7.py | 输局多加 total_cost | 输局 pnl = -bet_amount ✅ |
| hyperparam_tune_13_combos.py | 输局多加 fee | 输局 new_pnl = -bet ✅ |
| rolling_backtest_limit_price.py | 输局多加 TOTAL_COST | 输局 pnl = -bet ✅ |
| compare_fair.py | 输局多加 TOTAL_COST | 输局 pnl = -bet_amount ✅ |

**其他含类似逻辑的脚本**（optuna_gru_edge_kelly.py, grid_search_buy_price.py, backtest_tiered_bet_grid.py, backtest_simulation.py, src/python/trading_rules.py）建议后续统一审计。

---

## 五、总结

- **prediction_executor.ts**：公式与 Polymarket 15m 二元市场一致，实现正确。
- **optimizer / backtest**：输局多扣手续费，会低估回测收益；修复后与实盘结算更一致。
