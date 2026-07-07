import { strict as assert } from 'assert';

export type FillStatus = 'filled' | 'partial_fill' | 'timeout_unfilled' | 'queued' | 'skipped';
export type Side = 'buy' | 'sell';

export interface RawBookLevel {
  price: string | number;
  size?: string | number;
  amount?: string | number;
  sizeInShares?: string | number;
}

export interface BookLevel {
  price: number;
  sizeShares: number;
  sizeUsd: number;
}

export interface QueueAwareFillResult {
  side: Side;
  fillStatus: FillStatus;
  requestedUsd: number;
  rawFillableUsd: number;
  filledUsd: number;
  remainingUsd: number;
  avgFillPrice: number;
  queueCompetitionUsd: number;
  effectiveFillableUsd: number;
}

function toFiniteNumber(value: unknown): number | null {
  const num = typeof value === 'number' ? value : parseFloat(String(value ?? ''));
  return Number.isFinite(num) ? num : null;
}

export function normalizeBookSide(levels: RawBookLevel[] | undefined | null): BookLevel[] {
  if (!Array.isArray(levels)) return [];
  const out: BookLevel[] = [];
  for (const level of levels) {
    const price = toFiniteNumber(level.price);
    const rawSize = toFiniteNumber(level.size ?? level.amount ?? level.sizeInShares ?? 0);
    if (price == null || rawSize == null || price <= 0 || rawSize <= 0) continue;
    out.push({
      price,
      sizeShares: rawSize,
      sizeUsd: rawSize * price,
    });
  }
  return out;
}

export function calcWeightedAvgBuyFill(asks: BookLevel[], limitPrice: number, requestUsd: number): { avgPrice: number; filledUsd: number } {
  const sorted = asks.filter((a) => a.price <= limitPrice && a.sizeUsd > 0).sort((a, b) => a.price - b.price);
  let remaining = requestUsd;
  let totalCost = 0;
  let totalShares = 0;
  for (const ask of sorted) {
    if (remaining <= 0) break;
    const consumeUsd = Math.min(ask.sizeUsd, remaining);
    totalCost += consumeUsd;
    totalShares += consumeUsd / ask.price;
    remaining -= consumeUsd;
  }
  const avgPrice = totalShares > 0 ? totalCost / totalShares : limitPrice;
  return { avgPrice: Math.min(Math.max(avgPrice, 0.01), limitPrice), filledUsd: Math.max(0, totalCost) };
}

export function calcWeightedAvgSellFill(bids: BookLevel[], minPrice: number, requestUsd: number): { avgPrice: number; filledUsd: number } {
  const sorted = bids.filter((b) => b.price >= minPrice && b.sizeUsd > 0).sort((a, b) => b.price - a.price);
  let remaining = requestUsd;
  let totalProceeds = 0;
  let totalShares = 0;
  for (const bid of sorted) {
    if (remaining <= 0) break;
    const consumeUsd = Math.min(bid.sizeUsd, remaining);
    totalProceeds += consumeUsd;
    totalShares += consumeUsd / bid.price;
    remaining -= consumeUsd;
  }
  const avgPrice = totalShares > 0 ? totalProceeds / totalShares : minPrice;
  return { avgPrice: Math.max(avgPrice, 0.01), filledUsd: Math.max(0, totalProceeds) };
}

export function estimateBidCompetitionUsd(bids: BookLevel[], minPrice: number): number {
  let total = 0;
  for (const bid of bids) {
    if (bid.price < minPrice) continue;
    total += bid.sizeUsd;
  }
  return total;
}

export function estimateAskCompetitionUsd(asks: BookLevel[], maxPrice: number): number {
  let total = 0;
  for (const ask of asks) {
    if (ask.price > maxPrice) continue;
    total += ask.sizeUsd;
  }
  return total;
}

export function simulateQueueAwareBuyFill(args: {
  asks: BookLevel[];
  bids?: BookLevel[];
  limitPrice: number;
  requestUsd: number;
  queueCompetitionUsd?: number;
}): QueueAwareFillResult {
  const { asks, bids = [], limitPrice, requestUsd } = args;
  const rawFill = calcWeightedAvgBuyFill(asks, limitPrice, requestUsd);
  const queueCompetitionUsd = args.queueCompetitionUsd ?? estimateBidCompetitionUsd(bids, limitPrice);
  const effectiveFillableUsd = Math.max(0, rawFill.filledUsd - queueCompetitionUsd);
  const filledUsd = Math.min(requestUsd, effectiveFillableUsd);
  return {
    side: 'buy',
    fillStatus: filledUsd >= requestUsd ? 'filled' : filledUsd > 0 ? 'partial_fill' : (rawFill.filledUsd > 0 ? 'queued' : 'timeout_unfilled'),
    requestedUsd: requestUsd,
    rawFillableUsd: rawFill.filledUsd,
    filledUsd,
    remainingUsd: Math.max(0, requestUsd - filledUsd),
    avgFillPrice: rawFill.avgPrice,
    queueCompetitionUsd,
    effectiveFillableUsd,
  };
}

export function simulateQueueAwareSellFill(args: {
  bids: BookLevel[];
  asks?: BookLevel[];
  minPrice: number;
  requestUsd: number;
  queueCompetitionUsd?: number;
}): QueueAwareFillResult {
  const { bids, asks = [], minPrice, requestUsd } = args;
  const rawFill = calcWeightedAvgSellFill(bids, minPrice, requestUsd);
  const queueCompetitionUsd = args.queueCompetitionUsd ?? estimateAskCompetitionUsd(asks, minPrice);
  const effectiveFillableUsd = Math.max(0, rawFill.filledUsd - queueCompetitionUsd);
  const filledUsd = Math.min(requestUsd, effectiveFillableUsd);
  return {
    side: 'sell',
    fillStatus: filledUsd >= requestUsd ? 'filled' : filledUsd > 0 ? 'partial_fill' : (rawFill.filledUsd > 0 ? 'queued' : 'timeout_unfilled'),
    requestedUsd: requestUsd,
    rawFillableUsd: rawFill.filledUsd,
    filledUsd,
    remainingUsd: Math.max(0, requestUsd - filledUsd),
    avgFillPrice: rawFill.avgPrice,
    queueCompetitionUsd,
    effectiveFillableUsd,
  };
}

export function topDepthUsd(levels: BookLevel[], count = 3): number {
  return levels.slice(0, count).reduce((sum, level) => sum + level.sizeUsd, 0);
}

export function bestPrice(levels: BookLevel[], side: Side): number | null {
  if (levels.length === 0) return null;
  return side === 'buy' ? Math.min(...levels.map((l) => l.price)) : Math.max(...levels.map((l) => l.price));
}
