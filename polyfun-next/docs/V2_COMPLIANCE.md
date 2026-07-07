# Polymarket V2/pUSD 合规矩阵

依据官方文档：

- V2 Migration: https://docs.polymarket.com/v2-migration
- Orders: https://docs.polymarket.com/trading/orders/overview
- pUSD: https://docs.polymarket.com/concepts/pusd

## 固化要求

| 项 | 新系统要求 |
|---|---|
| 生产地址 | `https://clob.polymarket.com` |
| SDK | `py-clob-client-v2` 或同等 V2 签名实现 |
| 抵押币 | `pUSD 0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB` |
| V1 残留 | 不允许 V1 签名订单进入生产 |
| 本地订单状态 | 必须由 `getOrder/openOrders/activity` 确认 |
| claim | 以官网 positions/claimable 归零为成功，不信单一本地日志 |

## 关键语义

- 本地生成订单计划不等于官网挂单。
- 只有官网 `openOrders` 存在才叫“当前挂单中”。
- 官网 `CANCELED` 且成交为 0，只能叫“官方取消零成交”。
- 官网查不到的订单，必须叫 `OFFICIAL_MISSING`，不能叫 `created`。
