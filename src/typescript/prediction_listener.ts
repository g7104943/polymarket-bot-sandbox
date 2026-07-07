/**
 * 监听模型预测：轮询 Python 预测 API，或通过 WebSocket 接收预测，转交 trade_executor
 */

import axios from "axios";
import { getEnv, getEnvNumber, SYMBOLS, TIMEFRAMES, Symbol, Timeframe } from "./utils";

const DEFAULT_PORT = 8080;

export interface Prediction {
  symbol: string;
  timeframe: string;
  direction: "UP" | "DOWN";
  confidence: number;
}

export interface PredictionHandler {
  (p: Prediction): Promise<void>;
}

function getApiBase(): string {
  const port = getEnvNumber("MODEL_PREDICTION_PORT", DEFAULT_PORT);
  const host = process.env.MODEL_PREDICTION_HOST || "http://127.0.0.1";
  return `${host}:${port}`;
}

/**
 * 拉取 /predict 获取所有 symbol×timeframe 的预测
 */
export async function fetchPredictions(): Promise<Prediction[]> {
  const base = getApiBase();
  const url = `${base}/predict`;
  const { data } = await axios.get<{ predictions: Record<string, { direction: string; confidence: number; error?: string }> }>(url);
  const out: Prediction[] = [];
  for (const [key, v] of Object.entries(data.predictions || {})) {
    if (v.error || v.direction == null) continue;
    const [symbol, timeframe] = key.split("_");
    if (symbol && timeframe) {
      out.push({
        symbol,
        timeframe,
        direction: v.direction as "UP" | "DOWN",
        confidence: Number(v.confidence) || 0,
      });
    }
  }
  return out;
}

/**
 * 拉取 /predict/{symbol}?timeframe=xx
 */
export async function fetchPrediction(symbol: string, timeframe: string): Promise<Prediction | null> {
  const base = getApiBase();
  const s = symbol.replace("/USDT", "");
  const url = `${base}/predict/${s}?timeframe=${encodeURIComponent(timeframe)}`;
  try {
    const { data } = await axios.get<{ symbol: string; timeframe: string; direction: string; confidence: number }>(url);
    return {
      symbol: data.symbol,
      timeframe: data.timeframe,
      direction: data.direction as "UP" | "DOWN",
      confidence: Number(data.confidence) || 0,
    };
  } catch {
    return null;
  }
}

/**
 * 轮询 /predict，并按间隔调用 handler
 */
export async function startPolling(
  handler: PredictionHandler,
  intervalMs: number = 60_000
): Promise<() => void> {
  let stopped = false;
  const run = async (): Promise<void> => {
    while (!stopped) {
      try {
        const list = await fetchPredictions();
        for (const p of list) {
          await handler(p);
        }
      } catch (e) {
        console.error("[prediction_listener] fetch error:", e);
      }
      await new Promise((r) => setTimeout(r, intervalMs));
    }
  };
  run();
  return () => { stopped = true; };
}
