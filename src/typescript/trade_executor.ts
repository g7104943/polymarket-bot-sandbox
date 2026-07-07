/**
 * 交易执行：根据预测在 Polymarket 下注，支持 simulation/live，记录到 MongoDB
 */

import { getEnv, getEnvNumber, getEnvBool, Symbol, Timeframe } from "./utils";
import { resolveMarketId, placeOrder, OrderResult } from "./polymarket_client";
import { insertTrade } from "./database";
import { Prediction } from "./prediction_listener";

const FIXED_AMOUNT = "FIXED_AMOUNT";
const RATIO_AMOUNT = "RATIO_AMOUNT";
const USE_FIXED_AMOUNT = "USE_FIXED_AMOUNT";
const TRADING_MODE = "TRADING_MODE";

export async function executeFromPrediction(p: Prediction): Promise<{ ok: boolean; error?: string }> {
  const useFixed = getEnvBool(USE_FIXED_AMOUNT, true);
  const fixed = getEnvNumber(FIXED_AMOUNT, 100);
  const ratio = getEnvNumber(RATIO_AMOUNT, 0.05);
  const mode = (getEnv(TRADING_MODE, "simulation") === "live" ? "live" : "simulation") as "simulation" | "live";

  // 比例模式需查钱包 USDC 余额，此处暂用 fixed 作为 fallback
  const amount = useFixed ? fixed : fixed;
  if (amount <= 0) return { ok: false, error: "amount <= 0" };

  const sym = p.symbol.replace("/USDT", "").toUpperCase() as Symbol;
  const tf = p.timeframe as Timeframe;
  const marketId = resolveMarketId(sym, tf, p.direction);
  if (!marketId) {
    // 无市场映射时仅记录预测，不下单
    await insertTrade({
      symbol: p.symbol,
      timeframe: p.timeframe,
      direction: p.direction,
      confidence: p.confidence,
      amount,
      outcome: p.direction === "UP" ? "YES" : "NO",
      mode,
      marketId: undefined,
    });
    return { ok: true };
  }

  const outcome: "YES" | "NO" = p.direction === "UP" ? "YES" : "NO";
  const orderRes: OrderResult = await placeOrder(
    { marketId, outcome, amount },
    mode
  );

  await insertTrade({
    symbol: p.symbol,
    timeframe: p.timeframe,
    direction: p.direction,
    confidence: p.confidence,
    amount,
    outcome,
    marketId,
    txHash: orderRes.txHash,
    mode,
  });

  if (!orderRes.success) return { ok: false, error: orderRes.error };
  return { ok: true };
}
