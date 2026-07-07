/**
 * 主入口：启动预测监听 + 交易执行；根据 TRADING_MODE 做 simulation 或 live
 */

import "dotenv/config";
import { startPolling } from "./prediction_listener";
import { executeFromPrediction } from "./trade_executor";
import { getEnv } from "./utils";
import { closeDb } from "./database";

const TRADING_MODE = "TRADING_MODE";

async function main(): Promise<void> {
  const mode = getEnv(TRADING_MODE, "simulation");
  console.log(`[main] TRADING_MODE=${mode}`);

  const stop = await startPolling(async (p) => {
    console.log(`[predict] ${p.symbol} ${p.timeframe} -> ${p.direction} (${p.confidence})`);
    const r = await executeFromPrediction(p);
    if (!r.ok) console.error(`[trade] failed: ${r.error}`);
  }, 60_000);

  const shutdown = async () => {
    stop();
    await closeDb();
    process.exit(0);
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
