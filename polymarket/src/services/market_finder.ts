/**
 * Polymarket 市场查找器
 * 使用事件 Slug 直接获取 BTC/ETH/SOL 二元市场
 * 
 * API 文档:
 * - Gamma API: https://docs.polymarket.com/developers/gamma-markets-api/fetch-markets-guide
 * - CLOB API: https://docs.polymarket.com/developers/CLOB/clients/methods-public
 */

import * as fs from 'fs';
import * as path from 'path';
import axios, { AxiosInstance } from 'axios';
import { HttpsProxyAgent } from 'https-proxy-agent';
import { PREDICTION_ENV } from '../config/prediction_env';

// ── 全局 axios 实例：显式代理 + keepAlive 连接池 ──
const proxyUrl = process.env.HTTPS_PROXY || process.env.https_proxy
              || process.env.HTTP_PROXY  || process.env.http_proxy;
const proxyAgent = proxyUrl ? new HttpsProxyAgent(proxyUrl, { keepAlive: true }) : undefined;

const axiosClient: AxiosInstance = axios.create({
    ...(proxyAgent ? { httpAgent: proxyAgent, httpsAgent: proxyAgent, proxy: false } : {}),
});

// Data API 基础 URL
const DATA_API_BASE = 'https://data-api.polymarket.com';

// ============================================================
// 市场数据缓存（Market Data Daemon）
// ============================================================

const CACHE_PATH = path.resolve(__dirname, '..', '..', 'market_data_cache.json');
const CACHE_STALE_SECONDS = 30;

interface DaemonCache {
    updated_at: number;
    markets: Record<string, {
        slug: string;
        conditionId: string;
        tokens: Array<{ tokenId: string; outcome: string }>;
        resolved: boolean;
        winner: string | null;
    }>;
    orderbooks: Record<string, {
        best_ask: number;
        best_bid: number;
        asks: Array<{ price: string; size: string }>;
        bids: Array<{ price: string; size: string }>;
        updated_at: number;
    }>;
    prices: Record<string, {
        buy: number;
        sell: number;
        updated_at: number;
    }>;
}

let _cachedData: DaemonCache | null = null;
let _cacheReadAt = 0;
const CACHE_MEM_TTL_MS = 1000;

function readMarketDataCache(): DaemonCache | null {
    const nowMs = Date.now();
    if (_cachedData && nowMs - _cacheReadAt < CACHE_MEM_TTL_MS) return _cachedData;
    try {
        const raw = fs.readFileSync(CACHE_PATH, 'utf-8');
        _cachedData = JSON.parse(raw) as DaemonCache;
        _cacheReadAt = nowMs;
        return _cachedData;
    } catch {
        return null;
    }
}

function isCacheFresh(cache: DaemonCache | null): boolean {
    if (!cache) return false;
    return (Math.floor(Date.now() / 1000) - cache.updated_at) < CACHE_STALE_SECONDS;
}

// ============================================================
// 类型定义
// ============================================================

export interface PolymarketMarket {
    conditionId: string;       // 市场条件 ID
    questionId?: string;       // 问题 ID
    slug: string;              // URL slug
    title: string;             // 市场标题
    description?: string;      // 市场描述
    outcomes: string[];        // 结果选项 ["Up", "Down"] 或 ["Yes", "No"]
    outcomePrices?: string[];  // 当前价格
    volume?: number;           // 交易量
    liquidity?: number;        // 流动性
    endDate?: string;          // 结束时间
    active: boolean;           // 是否活跃
    tokens: MarketToken[];     // 代币信息
}

export interface MarketToken {
    tokenId: string;           // 代币 ID (clobTokenId)
    outcome: string;           // 对应结果 ("Up" 或 "Down")
    price?: number;            // 当前价格
}

export interface MarketSearchResult {
    symbol: string;            // 加密货币符号
    market: PolymarketMarket | null;
    error?: string;
}

export interface MarketResult {
    resolved: boolean;         // 市场是否已结算
    winner: 'Up' | 'Down' | null;  // 赢家（Up 或 Down）
    closed: boolean;           // 市场是否已关闭
    slug: string;              // 市场 slug
    timestamp?: string;        // 结算时间
    error?: string;            // 错误信息
}

// ============================================================
// 常量
// ============================================================

// Polymarket API 端点
const GAMMA_API_BASE = 'https://gamma-api.polymarket.com';
const CLOB_API_BASE = 'https://clob.polymarket.com';

// 加密货币 slug 映射（Polymarket Up/Down 格式通常为 xxx-updown-{timeframe}-{ts}）
const SYMBOL_SLUG_MAP: Record<string, string> = {
    'BTC': 'btc',
    'ETH': 'eth',
    'SOL': 'sol',
    'XRP': 'xrp',
};

// 周期常量（秒）
const INTERVAL_15M_SECONDS = 15 * 60;
const INTERVAL_5M_SECONDS = 5 * 60;
const INTERVAL_1H_SECONDS = 60 * 60;

// ============================================================
// 时间戳计算
// ============================================================

/**
 * 获取下一个 15 分钟 K 线的 Unix 时间戳
 * 
 * 例如：
 * - 当前时间 11:13:00 -> 返回 11:15:00 的时间戳
 * - 当前时间 11:15:30 -> 返回 11:30:00 的时间戳
 * 
 * @param earlySeconds 提前秒数（用于计算目标市场）
 */
export const getNext15MinTimestamp = (earlySeconds: number = 0): number => {
    const now = Math.floor(Date.now() / 1000); // 当前 Unix 时间戳（秒）
    const adjustedNow = now + earlySeconds; // 加上提前时间
    
    // 计算下一个 15 分钟边界
    const remainder = adjustedNow % INTERVAL_15M_SECONDS;
    const nextBoundary = adjustedNow + (INTERVAL_15M_SECONDS - remainder);
    
    return nextBoundary;
};

/**
 * 获取下一个 5 分钟 K 线的 Unix 时间戳（Exp13 用）
 * 
 * @param earlySeconds 提前秒数
 */
export const getNext5MinTimestamp = (earlySeconds: number = 0): number => {
    const now = Math.floor(Date.now() / 1000);
    const adjustedNow = now + earlySeconds;
    const remainder = adjustedNow % INTERVAL_5M_SECONDS;
    const nextBoundary = adjustedNow + (INTERVAL_5M_SECONDS - remainder);
    return nextBoundary;
};

/**
 * 获取下一个 1 小时 K 线边界时间戳。
 */
export const getNext1HourTimestamp = (earlySeconds: number = 0): number => {
    const now = Math.floor(Date.now() / 1000);
    const adjustedNow = now + earlySeconds;
    const remainder = adjustedNow % INTERVAL_1H_SECONDS;
    const nextBoundary = adjustedNow + (INTERVAL_1H_SECONDS - remainder);
    return nextBoundary;
};

/**
 * 获取当前 15 分钟周期的开始时间戳
 */
export const getCurrent15MinTimestamp = (): number => {
    const now = Math.floor(Date.now() / 1000);
    const remainder = now % INTERVAL_15M_SECONDS;
    return now - remainder;
};

/**
 * 获取当前 5 分钟周期的开始时间戳
 */
export const getCurrent5MinTimestamp = (): number => {
    const now = Math.floor(Date.now() / 1000);
    const remainder = now % INTERVAL_5M_SECONDS;
    return now - remainder;
};

/**
 * 构建市场事件 slug
 * 
 * 格式: {symbol}-updown-{timeframe}-{timestamp}
 * 支持 15m、5m 和研究用 1h。若官网无对应 slug，会返回“未找到市场”，不会伪造市场。
 */
export const buildMarketSlug = (symbol: string, timestamp: number, timeframe: string = '15m'): string => {
    const slugSymbol = SYMBOL_SLUG_MAP[symbol.toUpperCase()] || symbol.toLowerCase();
    return `${slugSymbol}-updown-${timeframe}-${timestamp}`;
};

/**
 * 格式化时间戳为可读字符串
 */
const formatTimestamp = (timestamp: number): string => {
    return new Date(timestamp * 1000).toLocaleString('zh-CN', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false,
    });
};

// ============================================================
// Gamma API - 获取市场信息
// ============================================================

/**
 * 判断是否为网络错误（可重试）
 */
const isNetworkError = (error: unknown): boolean => {
    if (!axios.isAxiosError(error)) return false;
    
    const status = error.response?.status;
    // 404 是市场不存在，不需要重试
    if (status === 404) return false;
    // HTTP 5xx 服务器错误，应重试
    if (status && status >= 500 && status < 600) return true;
    // HTTP 429 限流，应重试（可能需要更长延迟，但暂时用相同策略）
    if (status === 429) return true;
    
    // 其他网络错误：连接断开、超时、DNS 错误等
    const msg = error.message?.toLowerCase() || '';
    return (
        error.code === 'ECONNRESET' ||
        error.code === 'ECONNREFUSED' ||
        error.code === 'ETIMEDOUT' ||
        error.code === 'ENOTFOUND' ||
        msg.includes('socket') ||
        msg.includes('network') ||
        msg.includes('timeout') ||
        msg.includes('disconnected') ||
        msg.includes('tls') ||
        !error.response  // 没有响应通常是网络问题
    );
};

/**
 * 通过事件 slug 获取市场信息（带网络重试）
 * 
 * API: GET https://gamma-api.polymarket.com/events/slug/{slug}
 * 
 * 重试策略：
 * - 网络错误时每 3 秒重试一次
 * - 日志每 1 分钟打印一次（避免刷屏）
 * - 最多重试 10 分钟
 * 
 * @param slug 市场 slug，如 "btc-updown-15m-1768881600"
 */
export const getMarketBySlug = async (slug: string): Promise<PolymarketMarket | null> => {
    const url = `${GAMMA_API_BASE}/events/slug/${slug}`;
    const RETRY_INTERVAL_MS = 3000;      // 每 3 秒重试
    const LOG_INTERVAL_MS = 60000;       // 日志每 1 分钟打印
    const MAX_RETRY_MS = 10 * 60 * 1000; // 最多重试 10 分钟
    
    const startTime = Date.now();
    let lastLogTime = 0;
    let retryCount = 0;
    
    while (true) {
        try {
            const response = await axiosClient.get(url, {
                timeout: PREDICTION_ENV.REQUEST_TIMEOUT_MS,
            });
            
            const event = response.data;
            
            if (!event || !event.markets || event.markets.length === 0) {
                console.log(`  [警告] 未找到市场: ${slug}`);
                return null;
            }
            
            // 获取第一个市场（15分钟市场通常只有一个）
            const market = event.markets[0];
            
            // 解析代币信息
            const tokens: MarketToken[] = [];
            
            // clobTokenIds 和 outcomes 可能是字符串或数组
            const tokenIds = typeof market.clobTokenIds === 'string' 
                ? JSON.parse(market.clobTokenIds) 
                : market.clobTokenIds || [];
                
            const outcomes = typeof market.outcomes === 'string'
                ? JSON.parse(market.outcomes)
                : market.outcomes || ['Up', 'Down'];
                
            const prices = typeof market.outcomePrices === 'string'
                ? JSON.parse(market.outcomePrices)
                : market.outcomePrices || [];
            
            for (let i = 0; i < tokenIds.length; i++) {
                tokens.push({
                    tokenId: tokenIds[i],
                    outcome: outcomes[i] || (i === 0 ? 'Up' : 'Down'),
                    price: parseFloat(prices[i]) || 0,
                });
            }
            
            // 如果之前有重试过，打印恢复日志
            if (retryCount > 0) {
                const elapsed = Math.round((Date.now() - startTime) / 1000);
                console.log(`  [恢复] 网络恢复，成功获取市场 (${slug})，共重试 ${retryCount} 次，耗时 ${elapsed} 秒`);
            }
            
            return {
                conditionId: market.conditionId || market.condition_id || '',
                questionId: market.questionId || market.question_id,
                slug: slug,
                title: market.question || market.title || event.title || '',
                description: market.description || event.description,
                outcomes: outcomes,
                outcomePrices: prices,
                volume: parseFloat(market.volume) || 0,
                liquidity: parseFloat(market.liquidity) || 0,
                endDate: market.endDate || market.end_date,
                active: market.active !== false,
                tokens,
            };
            
        } catch (error) {
            // 404 = 市场不存在，不重试
            if (axios.isAxiosError(error) && error.response?.status === 404) {
                console.log(`  [信息] 市场尚未创建: ${slug}`);
                return null;
            }
            
            const errMsg = axios.isAxiosError(error) ? error.message : String(error);
            
            // 检查是否为可重试的网络错误
            if (isNetworkError(error)) {
                const elapsed = Date.now() - startTime;
                
                // 超过最大重试时间，放弃
                if (elapsed >= MAX_RETRY_MS) {
                    console.error(`  [错误] 获取市场失败 (${slug}): 网络重试超时 (${Math.round(elapsed / 1000)}秒)，放弃`);
                    return null;
                }
                
                retryCount++;
                
                // 每 1 分钟打印一次日志（避免刷屏）
                const now = Date.now();
                if (now - lastLogTime >= LOG_INTERVAL_MS) {
                    lastLogTime = now;
                    const elapsedSec = Math.round(elapsed / 1000);
                    console.log(`  [重试] 网络错误 (${slug}): ${errMsg.slice(0, 50)}... | 已重试 ${retryCount} 次，耗时 ${elapsedSec} 秒，继续重试...`);
                }
                
                // 等待 3 秒后重试
                await new Promise(resolve => setTimeout(resolve, RETRY_INTERVAL_MS));
                continue;
            }
            
            // 非网络错误，直接返回失败
            console.error(`  [错误] 获取市场失败 (${slug}): ${errMsg}`);
            return null;
        }
    }
};

// ============================================================
// CLOB API - 获取实时价格
// ============================================================

/**
 * 通过 CLOB API 获取代币实时价格
 * 
 * API: GET https://clob.polymarket.com/price?token_id={tokenId}&side=BUY
 * 
 * @param tokenId 代币 ID
 * @param side 买入或卖出 ("BUY" | "SELL")，官方 API 要求大写
 */
export const getTokenPrice = async (
    tokenId: string,
    side: 'BUY' | 'SELL' = 'BUY'
): Promise<number | null> => {
    // ── 优先从 daemon 缓存读取 ──
    const dc = readMarketDataCache();
    if (isCacheFresh(dc)) {
        const cp = dc!.prices[tokenId];
        if (cp) return side === 'BUY' ? cp.buy : cp.sell;
    }

    // ── REST fallback ──
    const url = `${CLOB_API_BASE}/price`;
    
    try {
        const response = await axiosClient.get(url, {
            params: {
                token_id: tokenId,
                side: side.toUpperCase(),
            },
            timeout: PREDICTION_ENV.REQUEST_TIMEOUT_MS,
        });
        
        const price = parseFloat(response.data?.price);
        return isNaN(price) ? null : price;
        
    } catch (error) {
        if (axios.isAxiosError(error) && error.response?.status === 404) {
            return null;
        }
        const errMsg = axios.isAxiosError(error) ? error.message : String(error);
        console.error(`  [警告] 获取价格失败 (${tokenId.slice(0, 15)}...): ${errMsg}`);
        return null;
    }
};

/**
 * 获取订单簿信息
 * 
 * API: GET https://clob.polymarket.com/book?token_id={tokenId}
 * 
 * 注意：404 表示订单簿不存在（市场刚创建/即将结算），静默处理
 */
export const getOrderBook = async (tokenId: string): Promise<any | null> => {
    // ── 优先从 daemon 缓存读取 ──
    const dc = readMarketDataCache();
    if (isCacheFresh(dc)) {
        const ob = dc!.orderbooks[tokenId];
        if (ob) return { asks: ob.asks, bids: ob.bids };
    }

    // ── REST fallback ──
    const url = `${CLOB_API_BASE}/book`;
    
    try {
        const response = await axiosClient.get(url, {
            params: { token_id: tokenId },
            timeout: PREDICTION_ENV.REQUEST_TIMEOUT_MS,
        });
        
        return response.data;
        
    } catch (error) {
        if (axios.isAxiosError(error) && error.response?.status === 404) {
            return null;
        }
        const errMsg = axios.isAxiosError(error) ? error.message : String(error);
        console.error(`  [错误] 获取订单簿失败: ${errMsg}`);
        return null;
    }
};

/**
 * 更新市场代币的实时价格
 */
const updateMarketPrices = async (market: PolymarketMarket): Promise<PolymarketMarket> => {
    const updatedTokens: MarketToken[] = [];
    
    for (const token of market.tokens) {
        const price = await getTokenPrice(token.tokenId, 'BUY');
        updatedTokens.push({
            ...token,
            price: price ?? token.price,
        });
    }
    
    return {
        ...market,
        tokens: updatedTokens,
    };
};

// ============================================================
// 主要公共 API
// ============================================================

/**
 * 查找指定加密货币的二元市场
 * 
 * @param symbol 加密货币符号 (BTC, ETH, SOL)
 * @param timeframe 时间周期（15m/5m；1h 仅在官网存在对应市场时可用）
 * @param targetPeriodEndTs 目标 15M 的 slug ts（=周期开始，与 Polymarket 官网及 predictions.json 一致）；未提供时用 getNext15MinTimestamp 推算
 */
export const findMarketForSymbol = async (
    symbol: string,
    timeframe?: string,
    targetPeriodEndTs?: number
): Promise<MarketSearchResult> => {
    // 兼容 "BTC/USDT" 或 "BTC"，取出基础符号
    const base = symbol.includes('/') ? symbol.split('/')[0] : symbol;
    const normalizedSymbol = base.toUpperCase();
    const tf = timeframe || PREDICTION_ENV.TIMEFRAME;
    
    if (!SYMBOL_SLUG_MAP[normalizedSymbol]) {
        return {
            symbol: normalizedSymbol,
            market: null,
            error: `不支持的币种: ${symbol}。支持: ${Object.keys(SYMBOL_SLUG_MAP).join(', ')}`,
        };
    }
    
    if (tf !== '15m' && tf !== '5m' && tf !== '1h') {
        return {
            symbol: normalizedSymbol,
            market: null,
            error: `仅支持 15m/5m/1h 时间周期的直接查找`,
        };
    }
    
    const intervalSeconds = tf === '5m'
        ? INTERVAL_5M_SECONDS
        : tf === '1h'
            ? INTERVAL_1H_SECONDS
            : INTERVAL_15M_SECONDS;
    const nextTimestamp = typeof targetPeriodEndTs === 'number'
        ? targetPeriodEndTs
        : (tf === '5m'
            ? getNext5MinTimestamp(PREDICTION_ENV.EARLY_ENTRY_SECONDS || 0)
            : tf === '1h'
                ? getNext1HourTimestamp(PREDICTION_ENV.EARLY_ENTRY_SECONDS || 0)
            : getNext15MinTimestamp(PREDICTION_ENV.EARLY_ENTRY_SECONDS || 0));
    
    const slug = buildMarketSlug(normalizedSymbol, nextTimestamp, tf);

    // ── 优先从 daemon 缓存读取 ──
    const dc = readMarketDataCache();
    if (isCacheFresh(dc)) {
        const cm = dc!.markets[slug];
        if (cm && !cm.resolved) {
            const tokens: MarketToken[] = cm.tokens.map((t) => {
                const cp = dc!.prices[t.tokenId];
                return { tokenId: t.tokenId, outcome: t.outcome, price: cp?.buy ?? 0 };
            });
            return {
                symbol: normalizedSymbol,
                market: {
                    conditionId: cm.conditionId,
                    slug: cm.slug,
                    title: `${normalizedSymbol} Up/Down ${tf}`,
                    outcomes: cm.tokens.map((t) => t.outcome),
                    active: true,
                    tokens,
                },
            };
        }
    }

    // ── REST fallback ──
    console.log(`  [${normalizedSymbol}] 正在查找市场: ${slug}`);
    console.log(`  [${normalizedSymbol}] 目标时间: ${formatTimestamp(nextTimestamp)}`);
    
    let market = await getMarketBySlug(slug);
    
    if (!market) {
        const nextNextTimestamp = nextTimestamp + intervalSeconds;
        const nextSlug = buildMarketSlug(normalizedSymbol, nextNextTimestamp, tf);
        console.log(`  [${normalizedSymbol}] 尝试下一时段: ${nextSlug}`);
        market = await getMarketBySlug(nextSlug);
    }
    
    if (!market) {
        return {
            symbol: normalizedSymbol,
            market: null,
            error: `未找到 ${normalizedSymbol} ${tf} 市场`,
        };
    }
    
    market = await updateMarketPrices(market);
    
    return {
        symbol: normalizedSymbol,
        market,
    };
};

/**
 * 查找市场的对应 Polymarket 市场
 * @param targetPeriodEndTs 与 predictions 的 target_period_end_ts 一致（=周期开始/slug ts），用于按 ts 下单
 * @param symbolsOverride 若提供，只查这些 symbol（否则用 ALLOWED_MARKETS），如从 predictions 解析出的 list
 */
export const findAllMarkets = async (
    timeframe?: string,
    targetPeriodEndTs?: number,
    symbolsOverride?: string[]
): Promise<MarketSearchResult[]> => {
    const tf = timeframe || PREDICTION_ENV.TIMEFRAME;
    const symbols = symbolsOverride ?? PREDICTION_ENV.ALLOWED_MARKETS;
    
    console.log(`\n[市场查找器] 正在搜索 ${symbols.join(', ')} ${tf} 市场...`);
    if (typeof targetPeriodEndTs === 'number') {
        console.log(`[市场查找器] 目标周期(slug ts=周期开始): ${formatTimestamp(targetPeriodEndTs)} (slug: *-updown-${tf}-${targetPeriodEndTs})`);
    }
    
    const promises = symbols.map((s) => findMarketForSymbol(s, tf, targetPeriodEndTs));
    const results = await Promise.all(promises);
    
    return results;
};

/**
 * 根据预测方向获取要购买的代币
 * 
 * 对于 "Bitcoin Up or Down" 市场:
 * - UP 预测 -> 买 "Up" 代币
 * - DOWN 预测 -> 买 "Down" 代币
 * 
 * @param market Polymarket 市场
 * @param direction 预测方向 "UP" 或 "DOWN"
 */
export const getTokenForDirection = (
    market: PolymarketMarket,
    direction: 'UP' | 'DOWN'
): MarketToken | null => {
    if (!market.tokens || market.tokens.length < 2) {
        return null;
    }
    
    // 15 分钟市场的 outcomes 通常是 ["Up", "Down"]
    // 找到对应方向的代币
    const targetOutcome = direction === 'UP' ? 'Up' : 'Down';
    
    // 先尝试精确匹配
    let token = market.tokens.find((t) => 
        t.outcome.toLowerCase() === targetOutcome.toLowerCase()
    );
    
    // 如果没找到，可能是 Yes/No 格式（兼容旧市场）
    if (!token) {
        const title = market.title.toLowerCase();
        
        // 检查市场标题来确定映射
        if (title.includes('up or down')) {
            // 标准 up/down 市场
            // outcomes[0] = "Up", outcomes[1] = "Down"
            const index = direction === 'UP' ? 0 : 1;
            token = market.tokens[index];
        } else if (title.includes('higher') || title.includes('above')) {
            // "Will it be higher?" 类型
            token = market.tokens.find((t) => 
                t.outcome.toLowerCase() === (direction === 'UP' ? 'yes' : 'no')
            );
        } else {
            // 默认：第一个代币是 UP，第二个是 DOWN
            const index = direction === 'UP' ? 0 : 1;
            token = market.tokens[index];
        }
    }
    
    return token || null;
};

/**
 * 打印市场搜索结果
 */
export const printMarketResults = (results: MarketSearchResult[]): void => {
    console.log('\n[市场搜索结果]');
    console.log('-'.repeat(60));
    
    for (const r of results) {
        if (r.market) {
            console.log(`  [成功] ${r.symbol}: ${r.market.title}`);
            console.log(`       Slug: ${r.market.slug}`);
            console.log(`       Condition ID: ${r.market.conditionId.slice(0, 20)}...`);
            console.log(`       流动性: $${r.market.liquidity?.toFixed(2) || 'N/A'}`);
            
            for (const token of r.market.tokens) {
                const priceStr = token.price ? `$${token.price.toFixed(4)}` : 'N/A';
                const directionText = token.outcome === 'Up' ? '上涨' : '下跌';
                console.log(`       ${directionText}(${token.outcome}): ${token.tokenId.slice(0, 15)}... @ ${priceStr}`);
            }
        } else {
            console.log(`  [失败] ${r.symbol}: ${r.error || '未找到市场'}`);
        }
        console.log('');
    }
    
    console.log('-'.repeat(60));
};

/**
 * 获取市场详细信息的调试输出
 */
export const debugMarketInfo = async (symbol: string): Promise<void> => {
    console.log(`\n=== ${symbol} 市场调试信息 ===\n`);
    
    const earlySeconds = PREDICTION_ENV.EARLY_ENTRY_SECONDS || 0;
    const currentTs = getCurrent15MinTimestamp();
    const nextTs = getNext15MinTimestamp(earlySeconds);
    
    console.log(`当前15分钟周期开始: ${formatTimestamp(currentTs)}`);
    console.log(`下一个15分钟周期开始: ${formatTimestamp(nextTs)}`);
    console.log(`提前下单时间: ${earlySeconds} 秒`);
    console.log('');
    
    const slug = buildMarketSlug(symbol, nextTs);
    console.log(`构建的 slug: ${slug}`);
    console.log(`API URL: ${GAMMA_API_BASE}/events/slug/${slug}`);
    console.log('');
    
    const result = await findMarketForSymbol(symbol);
    
    if (result.market) {
        console.log('找到市场!');
        console.log(JSON.stringify(result.market, null, 2));
    } else {
        console.log('未找到市场:', result.error);
    }
};

// ============================================================
// 市场结算结果查询
// ============================================================

/**
 * 通过 CLOB API getMarket(conditionId) 获取结算结果（官方推荐）
 * 
 * GET https://clob.polymarket.com/markets/{conditionId}
 * 使用 market.closed 与 tokens[].winner 判断胜者，与官网/链上一致。
 * 
 * @param conditionId 市场 condition_id（0x...）
 * @param slugHint 仅用于 MarketResult.slug 展示，可选
 */
export const getMarketResultByConditionId = async (
    conditionId: string,
    slugHint?: string
): Promise<MarketResult> => {
    const result: MarketResult = {
        resolved: false,
        winner: null,
        closed: false,
        slug: slugHint || `condition:${conditionId.slice(0, 12)}...`,
    };
    if (!conditionId || !conditionId.startsWith('0x')) {
        result.error = '无效的 conditionId';
        return result;
    }

    // ── 优先从 daemon 缓存读取 ──
    const dc = readMarketDataCache();
    if (isCacheFresh(dc)) {
        for (const cm of Object.values(dc!.markets)) {
            if (cm.conditionId === conditionId && cm.resolved) {
                return { resolved: true, winner: cm.winner as 'Up' | 'Down' | null, closed: true, slug: cm.slug };
            }
        }
    }

    // ── REST fallback ──
    try {
        const url = `${CLOB_API_BASE}/markets/${conditionId}`;
        const res = await axiosClient.get(url, { timeout: PREDICTION_ENV.REQUEST_TIMEOUT_MS });
        const m = res.data;
        const closed = m?.closed === true;
        result.closed = closed;

        // 1) 优先用 tokens[].winner（官方：tokens.find(t => t.winner)）
        const tokens = m?.tokens || m?.assets || [];
        const winnerToken = tokens.find((t: any) => t.winner === true);
        if (winnerToken && (winnerToken.outcome != null || winnerToken.name != null)) {
            const o = String(winnerToken.outcome ?? winnerToken.name ?? '');
            result.resolved = true;
            result.winner = /^down$/i.test(o) ? 'Down' : 'Up';
            return result;
        }

        // 2) 顶层 winning_outcome（如 WebSocket market_resolved）
        const wo = m?.winning_outcome ?? m?.winningOutcome;
        if (closed && wo != null) {
            const o = String(wo);
            result.resolved = true;
            result.winner = /^down$/i.test(o) ? 'Down' : 'Up';
            return result;
        }

        // 3) CLOB 未标 winner 且 closed：无法从 CLOB 得到胜者，留给 Gamma 兜底
        if (closed && tokens.length >= 2) {
            result.error = 'CLOB 返回 closed 但无 tokens[].winner，将用 Gamma 兜底';
        }
        return result;
    } catch (e: any) {
        if (axios.isAxiosError(e) && e.response?.status === 404) {
            result.error = `CLOB 市场不存在: ${conditionId.slice(0, 16)}...`;
        } else {
            result.error = `CLOB 请求失败: ${e?.message || String(e)}`;
        }
        return result;
    }
};

/**
 * 通过 Data API 获取用户持仓
 * 
 * API: GET https://data-api.polymarket.com/positions?user={address}&market={conditionId}
 * 
 * @param userAddress 用户钱包地址
 * @param conditionId 可选，如果提供则只查询该市场的持仓（官方文档参数名为 market）
 * @returns 持仓列表，每个持仓包含 tokenId, size, value 等信息
 */
export const getUserPositions = async (
    userAddress: string,
    conditionId?: string
): Promise<any[]> => {
    const url = `${DATA_API_BASE}/positions`;
    
    try {
        const params: any = { user: userAddress };
        if (conditionId) {
            // 官方文档要求使用 'market' 作为参数名
            params.market = conditionId;
        }
        
        const response = await axiosClient.get(url, {
            params,
            timeout: PREDICTION_ENV.REQUEST_TIMEOUT_MS,
        });
        
        return response.data || [];
        
    } catch (error) {
        const errMsg = axios.isAxiosError(error) ? error.message : String(error);
        console.error(`  [错误] 获取持仓失败: ${errMsg}`);
        return [];
    }
};

/**
 * 获取用户在指定市场的持仓金额（USD）
 * 
 * @param userAddress 用户钱包地址
 * @param conditionId 市场 conditionId
 * @param tokenId 代币 tokenId（可选，如果提供则只查询该代币的持仓）
 * @returns 持仓金额（USD），如果查询失败返回 0
 */
export const getPositionValue = async (
    userAddress: string,
    conditionId: string,
    tokenId?: string
): Promise<number> => {
    try {
        const positions = await getUserPositions(userAddress, conditionId);
        
        if (!positions || positions.length === 0) {
            return 0;
        }
        
        // 如果提供了 tokenId，只计算该代币的持仓
        if (tokenId) {
            const position = positions.find((p: any) => p.tokenId === tokenId);
            return position ? parseFloat(position.value || '0') : 0;
        }
        
        // 否则计算该市场所有代币的持仓总和
        const totalValue = positions.reduce((sum: number, p: any) => {
            return sum + parseFloat(p.value || '0');
        }, 0);
        
        return totalValue;
        
    } catch (error) {
        console.error(`  [错误] 获取持仓金额失败: ${error}`);
        return 0;
    }
};

/**
 * 获取市场结算结果（Gamma API，按 slug）
 * 
 * 通过 Gamma API 查询市场是否已结算，以及哪个选项赢了
 * 
 * 判断逻辑（满足任一即视为可结算）：
 * - market/event 的 closed 或 archived 为 true
 * - outcomePrices 已呈 ["1","0"] 或 ["0","1"]（≥0.99 / ≤0.01）
 * 
 * 说明：Gamma 的 closed/archived 可能比 outcomePrices 更新滞后，官网已显示
 * 结算时 outcomePrices 往往先变为 1/0，故以 outcomePrices 为准做补充判断。
 * 
 * @param slug 市场 slug，如 "btc-updown-15m-1768881600"
 */
export const getMarketResult = async (slug: string): Promise<MarketResult> => {
    const url = `${GAMMA_API_BASE}/events/slug/${slug}`;
    
    const result: MarketResult = {
        resolved: false,
        winner: null,
        closed: false,
        slug: slug,
    };

    // ── 优先从 daemon 缓存读取 ──
    const dc = readMarketDataCache();
    if (isCacheFresh(dc)) {
        const cm = dc!.markets[slug];
        if (cm && cm.resolved) {
            return { resolved: true, winner: cm.winner as 'Up' | 'Down' | null, closed: true, slug };
        }
    }

    // ── REST fallback ──
    try {
        const response = await axiosClient.get(url, {
            timeout: PREDICTION_ENV.REQUEST_TIMEOUT_MS,
        });
        
        const event = response.data;
        
        if (!event || !event.markets || event.markets.length === 0) {
            result.error = `市场不存在: ${slug}`;
            return result;
        }
        
        const market = event.markets[0];
        
        // 是否已关闭：market 或 event 层任一为 closed/archived 即可
        result.closed = (
            market.closed === true || market.archived === true ||
            event.closed === true || event.archived === true
        );
        
        // 解析 outcomePrices 与 outcomes（无论 closed 与否都解析，用于下文判断）
        const prices = typeof market.outcomePrices === 'string'
            ? JSON.parse(market.outcomePrices)
            : market.outcomePrices || [];
        const outcomes = typeof market.outcomes === 'string'
            ? JSON.parse(market.outcomes)
            : market.outcomes || ['Up', 'Down'];
        
        // 用 outcomePrices 判断是否已结算（Gamma 可能先更新价格、后更新 closed）
        // 赢的选项 ≈1，输的 ≈0 即视为已结算
        if (prices.length >= 2) {
            const price0 = parseFloat(prices[0]) || 0;
            const price1 = parseFloat(prices[1]) || 0;
            if (price0 >= 0.99 && price1 <= 0.01) {
                result.resolved = true;
                result.winner = outcomes[0]?.toLowerCase() === 'up' ? 'Up' :
                    outcomes[0]?.toLowerCase() === 'down' ? 'Down' : 'Up';
            } else if (price1 >= 0.99 && price0 <= 0.01) {
                result.resolved = true;
                result.winner = outcomes[1]?.toLowerCase() === 'up' ? 'Up' :
                    outcomes[1]?.toLowerCase() === 'down' ? 'Down' : 'Down';
            }
        }
        
        result.timestamp = market.endDate || market.end_date || new Date().toISOString();
        return result;
        
    } catch (error) {
        if (axios.isAxiosError(error) && error.response?.status === 404) {
            result.error = `市场不存在: ${slug}`;
        } else {
            const errMsg = axios.isAxiosError(error) ? error.message : String(error);
            result.error = `获取市场结果失败: ${errMsg}`;
        }
        return result;
    }
};

/**
 * 批量获取多个市场的结算结果
 * 
 * @param slugs 市场 slug 数组
 */
export const getMultipleMarketResults = async (
    slugs: string[]
): Promise<Map<string, MarketResult>> => {
    const results = new Map<string, MarketResult>();
    
    // 并行请求所有市场结果
    const promises = slugs.map(async (slug) => {
        const result = await getMarketResult(slug);
        return { slug, result };
    });
    
    const responses = await Promise.all(promises);
    
    for (const { slug, result } of responses) {
        results.set(slug, result);
    }
    
    return results;
};
