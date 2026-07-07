# 可进一步提速的方向

在已完成「WebSocket 实时 best_ask + 3 秒监控」的基础上，以下方向可继续提升响应/执行速度。

---

## 与官方推荐方案对齐

- **流动性监控**：使用 WebSocket Market Channel（`wss://ws-subscriptions-clob.polymarket.com/ws/market`），延迟约 ~100ms；订阅 **book** 与 **price_change** 实时获取订单簿更新。
- **可买数量**：从 WebSocket **book** 消息解析；订单簿快照含各价位的可用数量（`asksSnapshot`），用于计算可用流动性和部分成交。
- **买入**：FOK/FAK 市价单（立即执行），使用 **createAndPostMarketOrder()** 一步完成，不再先调 getOrderBook 再下单。
  - 有 WS 深度且流动性 ≥ 下单金额时用 **FOK**（全部成交或取消）；否则用 **FAK**（部分成交，剩余加入流动性监控）。
- **最快路径**：WS 收到 book → 用 asksSnapshot 算价格与可买数量 → 直接 createAndPostMarketOrder（无额外 /book、/order 以外的 REST），符合「收到更新后立即 FOK/FAK 下单」。

---

## 请求频率（大概每秒几次）

- **WebSocket（我们→服务器）**：长连，不是按“每秒几次”的 REST。
  - **PING**：每 **10 秒** 1 次（保活）。
  - **订阅/更新**：每 **3 秒** 调用一次 `syncSubscribe()`，**仅当监控队列 token 列表变化**才真正发一条 `{ assets_ids, operation: 'subscribe' }`。
  - 合计：我们主动发到 WS 的约 **0.1～0.4 次/秒**（PING 0.1/秒 + 订阅更新最多约 0.33/秒且仅当列表变）。
- **REST 轮询**：
  - 高价监控：每 **3 秒** 一次（`getTokenPrice` 等，不调 /book）。
  - 低价监控：每 **3 秒** 一次（同上）。
  - 流动性监控：每 **10 秒** 一次（`getTokenPrice` + `getOrderBook`，兜底用）。
  - 有 WS **book** 触发时，对应那笔**不会**再发 REST /book，只有 POST 市价单。

---

## 限速与不下单前多查

- **CLOB /book**：1,500 请求/10秒（单）、500/10秒（批量）；**CLOB 通用** 9,000/10秒；**POST /order** 3,500/10秒（突发）、36,000/10分钟（持续）。
- **策略**：WS 触发且有 asksSnapshot 时**不再调用 getOrderBook/getTokenPrice**，仅 POST 市价单；流动性监控 REST 兜底由 3 秒改为 **10 秒**，减少 /book 占用；高价/低价监控仍 3 秒轮询（只做入队与价格判断，触发下单时若来自 WS book 则无额外 /book）。这样在不会被限速的前提下，下单前查询最少、成交路径最短。

---

## 逻辑复查（WS/asksSnapshot/FOK/FAK）

- **asksSnapshot 传递**：仅 WS **book** 消息带 `asksSnapshot`；**price_change** 不带。`tryTriggerPriceMonitorByTokenId` / `tryTriggerLowPriceMonitorByTokenId` 收到 `asksSnapshot` 时写入 `entry.asksSnapshot`；轮询路径（checkPriceMonitorQueue）不会设置 `entry.asksSnapshot`，故轮询触发时仍走 getOrderBook，逻辑正确。
- **executePrediction opts**：`opts.asksSnapshot` 有则传给 `simulateTrade` / `executeLiveTrade`；二者用 `useWsData = opts?.triggerPrice != null && opts?.asksSnapshot != null && opts.asksSnapshot.length > 0` 判断是否跳过 getOrderBook，与“有 book 才跳过”一致。
- **FOK/FAK**：`useFOK = useWsData && availableLiquidity >= betAmount`；FOK 被 Kill（0 成交）时 `actualAmount=0`、`remainingAmount=betAmount`，走“部分成交”分支并加入流动性监控；订单失败（`resp.success = false`）时由 executePrediction 将剩余金额加入流动性监控。两路都会继续补单，无漏单。
- **orderTypeLabel**：在 `if (resp.success)` 内、`if (isPartialFill)` 前定义，两分支均可访问，无作用域错误。

---

## 1. 入队时立即同步 WS 订阅（小改）

**现状**：新条目加入价格/低价监控队列后，要等下一个 3 秒周期才执行 `syncSubscribe()`，WS 才会订阅该 token。

**改进**：在 `prediction_executor.ts` 中，每次 `priceMonitorQueue.push()` / `lowPriceMonitorQueue.push()` 后，通知入口层（或通过回调）立即调用一次 `marketOrderbookWS.syncSubscribe()`，这样新市场可马上被 WS 订阅，best_ask 一到阈值即可触发，无需等 3 秒。

**实现要点**：执行器接收可选回调 `onMonitorQueueChanged?: () => void`，在 push 后调用；入口在创建 WS 时传入该回调，在回调里执行 `marketOrderbookWS.syncSubscribe()`。

---

## 2. 流动性监控也用 WS 订单簿（已实现）

**现状**：流动性监控若纯轮询会每 3 秒调 `getTokenPrice` + `getOrderBook`，易触及 /book 限速（1,500/10秒）。

**已实现**：WS **book** 为主：`getMonitorTokenIds()` 含 liquidity 队列 tokenId；`book` 消息带 `asksSnapshot`，`tryTriggerLiquidityMonitorByTokenId` 深度足够时立即补单；`simulateTrade` / `executeLiveTrade` 有 `opts.asksSnapshot` 时不再拉 API。REST 兜底改为 **10 秒** 一次，避免限速。

---

## 3. WS 触发下单时复用 best_ask（已实现）

**已实现**：`executePrediction` 在 `useTriggerPrice` 时不再走 `getTokenPrice` 的 fallback；`simulateTrade` / `executeLiveTrade` 支持 `opts.triggerPrice` 与 `opts.asksSnapshot`，WS 触发的价格/流动性监控补单直接复用 WS 数据，不再拉 API。

---

## 4. 扫描 predictions.json 间隔（可选）

**现状**：`SCAN_INTERVAL_MS = 1000`（1 秒）扫描一次 predictions.json。

**改进**：若希望「新预测出现后更快发现」，可改为 500ms；需权衡磁盘 I/O 与 Python 写入频率，通常 1 秒已够。

---

## 5. 止损检查间隔（可选）

**现状**：单笔止损每 30 秒检查一次（`stopLossIntervalMs = 30 * 1000`）。

**改进**：若希望更快止损，可改为 15 秒；频率提高会略增价格查询次数。

---

## 6. 报告/结算间隔（不建议再提速）

- 报告生成：每小时一次，与交易延迟无关，保持即可。
- 结算检查：每 1 分钟，与链上/API 节奏匹配，保持即可。
- LIVE 余额同步：每 10 秒，与链上区块时间匹配，保持即可。

---

## 检测脚本用法

```bash
# 仅静态检查（监控间隔、WS 端点、执行器方法等）
npm run verify-ws-monitor

# 含 WebSocket 连通性测试（需提供任意有效 tokenId，或 Node 22+ / 安装 ws）
VERIFY_WS_TOKEN_ID=某个市场的tokenId npm run verify-ws-monitor
```
