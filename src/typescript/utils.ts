/**
 * 工具函数
 */

export function getEnv(key: string, defaultValue?: string): string {
  const v = process.env[key];
  if (v != null && v !== "") return v;
  if (defaultValue !== undefined) return defaultValue;
  throw new Error(`Missing env: ${key}`);
}

export function getEnvNumber(key: string, defaultValue: number): number {
  const v = process.env[key];
  if (v == null || v === "") return defaultValue;
  const n = parseFloat(v);
  return isNaN(n) ? defaultValue : n;
}

export function getEnvBool(key: string, defaultValue: boolean): boolean {
  const v = process.env[key];
  if (v == null || v === "") return defaultValue;
  return /^(1|true|yes|on)$/i.test(v);
}

export function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

export type Symbol = "BTC" | "ETH" | "SOL" | "XRP";
export type Timeframe = "15m" | "1h" | "4h";

export const SYMBOLS: Symbol[] = ["BTC", "ETH", "SOL", "XRP"];
export const TIMEFRAMES: Timeframe[] = ["15m", "1h", "4h"];

export function toApiSymbol(s: string): string {
  const u = s.toUpperCase().replace("/USDT", "").replace("USDT", "");
  return ["BTC", "ETH", "SOL", "XRP"].includes(u) ? `${u}/USDT` : s;
}
