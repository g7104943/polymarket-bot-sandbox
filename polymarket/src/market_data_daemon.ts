/**
 * Market Data Daemon — 单进程集中管理所有 Polymarket 市场数据
 *
 * 维护 1 个 WebSocket + 少量 REST 请求，将订单簿、价格、市场元数据
 * 写入本地 market_data_cache.json，供 264+ 个交易进程本地读取。
 *
 * 启动: npx ts-node src/market_data_daemon.ts
 * 停止: kill $(cat market_data_daemon.pid)
 *
 * 断开原因与已做措施:
 * 1) 已做：每次断开打日志 code/reason、固定 3s 重连、并增加卡死检测（WS 长时间无消息自动软重连）。
 * 2) 根因：Polymarket 要求连接后立即发订阅、每 10s 发 PING；本端已满足。若仍频繁断开，多为网络/代理不稳（code 1006、reason 空）。
 * 3) 根因缓解：若当前/下一周期无 token 导致未发订阅，服务端会关连接；已改为用「全部未结算 token」兜底发送订阅，避免因此被关。
 */

import * as fs from 'fs';
import * as path from 'path';
import { execSync } from 'child_process';

// ── 代理自动检测（与 prediction_index.ts 一致）──
if (!process.env.HTTPS_PROXY && !process.env.https_proxy) {
    try {
        const out = execSync(
            `python3 -c "import urllib.request; p=urllib.request.getproxies(); print(p.get('https','') or p.get('http',''))"`,
            { timeout: 3000, encoding: 'utf-8' },
        ).trim();
        if (out) {
            process.env.HTTPS_PROXY = out;
            process.env.HTTP_PROXY = out;
            console.log(`[Daemon] 检测到系统代理: ${out}`);
        }
    } catch { /* 无代理 */ }
}

import axios from 'axios';
import WS from 'ws';
import { HttpsProxyAgent } from 'https-proxy-agent';

// ── 常量 ──

const POLYMARKET_DIR = path.resolve(__dirname, '..');
const CACHE_FILE = path.join(POLYMARKET_DIR, 'market_data_cache.json');
const PID_FILE = path.join(POLYMARKET_DIR, 'market_data_daemon.pid');

const WS_URL = 'wss://ws-subscriptions-clob.polymarket.com/ws/market';
const GAMMA_API = 'https://gamma-api.polymarket.com';
const CLOB_API = 'https://clob.polymarket.com';
const TIMEOUT_MS = 10_000;
const INTERVAL_15M = 15 * 60;

const SYMBOLS = ['BTC', 'ETH', 'SOL', 'XRP'];
const SLUG_MAP: Record<string, string> = { BTC: 'btc', ETH: 'eth', SOL: 'sol', XRP: 'xrp' };

// ── 代理 axios ──

const proxyUrl = process.env.HTTPS_PROXY || process.env.https_proxy
              || process.env.HTTP_PROXY  || process.env.http_proxy;
const proxyAgent = proxyUrl ? new HttpsProxyAgent(proxyUrl, { keepAlive: true }) : undefined;
const http = axios.create({
    ...(proxyAgent ? { httpAgent: proxyAgent, httpsAgent: proxyAgent, proxy: false } : {}),
    timeout: TIMEOUT_MS,
});

// ── Cache 数据结构 ──

interface CacheMarket {
    slug: string;
    conditionId: string;
    tokens: Array<{ tokenId: string; outcome: string }>;
    resolved: boolean;
    winner: string | null;
}

interface CacheOrderbook {
    best_ask: number;
    best_bid: number;
    asks: Array<{ price: string; size: string }>;
    bids: Array<{ price: string; size: string }>;
    updated_at: number;
}

interface CachePrice {
    buy: number;
    sell: number;
    updated_at: number;
}

interface CacheData {
    updated_at: number;
    markets: Record<string, CacheMarket>;
    orderbooks: Record<string, CacheOrderbook>;
    prices: Record<string, CachePrice>;
}

const cache: CacheData = {
    updated_at: 0,
    markets: {},
    orderbooks: {},
    prices: {},
};

function pruneExpiredMarketCache(): void {
    const cutoff = now() - 30 * 60;
    const liveTokenIds = new Set<string>();
    for (const [slug, mkt] of Object.entries(cache.markets)) {
        const m = slug.match(/-(\d+)$/);
        if (m && parseInt(m[1]) + INTERVAL_15M < cutoff) {
            delete cache.markets[slug];
        } else {
            for (const t of mkt.tokens) liveTokenIds.add(t.tokenId);
        }
    }
    for (const tid of Object.keys(cache.orderbooks)) {
        if (!liveTokenIds.has(tid)) delete cache.orderbooks[tid];
    }
    for (const tid of Object.keys(cache.prices)) {
        if (!liveTokenIds.has(tid)) delete cache.prices[tid];
    }
}

function loadCacheFromDisk(): void {
    if (!fs.existsSync(CACHE_FILE)) {
        try {
            // 首次启动时写一个空壳缓存，避免上游因为“文件不存在”报错。
            fs.writeFileSync(CACHE_FILE, JSON.stringify(cache), 'utf-8');
        } catch (e: any) {
            console.warn(`[Daemon] 初始化缓存文件失败: ${e.message}`);
        }
        return;
    }
    try {
        const raw = fs.readFileSync(CACHE_FILE, 'utf-8');
        const parsed = JSON.parse(raw) as Partial<CacheData>;
        cache.updated_at = Number(parsed.updated_at || 0);
        cache.markets = (parsed.markets && typeof parsed.markets === 'object')
            ? parsed.markets as Record<string, CacheMarket>
            : {};
        cache.orderbooks = (parsed.orderbooks && typeof parsed.orderbooks === 'object')
            ? parsed.orderbooks as Record<string, CacheOrderbook>
            : {};
        cache.prices = (parsed.prices && typeof parsed.prices === 'object')
            ? parsed.prices as Record<string, CachePrice>
            : {};
        console.log(
            `[Daemon] 已加载历史缓存: markets=${Object.keys(cache.markets).length}, ` +
            `orderbooks=${Object.keys(cache.orderbooks).length}, prices=${Object.keys(cache.prices).length}`,
        );
    } catch (e: any) {
        console.warn(`[Daemon] 读取历史缓存失败，将以空缓存启动: ${e.message}`);
    }
}

// ── 辅助 ──

function now(): number { return Math.floor(Date.now() / 1000); }

function getBarTimestamps(): number[] {
    const n = now();
    const current = n - (n % INTERVAL_15M);
    return [current, current + INTERVAL_15M, current + 2 * INTERVAL_15M];
}

function buildSlug(sym: string, ts: number): string {
    return `${SLUG_MAP[sym] || sym.toLowerCase()}-updown-15m-${ts}`;
}

function allTokenIds(excludeResolved = false): string[] {
    const ids: string[] = [];
    for (const m of Object.values(cache.markets)) {
        if (excludeResolved && m.resolved) continue;
        for (const t of m.tokens) ids.push(t.tokenId);
    }
    return Array.from(new Set(ids));
}

let writeQueued = false;
function flushCache(): void {
    if (writeQueued) return;
    writeQueued = true;
    setImmediate(() => {
        writeQueued = false;
        cache.updated_at = now();
        // Use a per-process temp file to avoid cross-process rename races
        // when multiple daemon instances are accidentally running.
        const tmp = `${CACHE_FILE}.${process.pid}.${Date.now()}.tmp`;
        try {
            fs.mkdirSync(path.dirname(CACHE_FILE), { recursive: true });
            fs.writeFileSync(tmp, JSON.stringify(cache), 'utf-8');
            fs.renameSync(tmp, CACHE_FILE);
        } catch (e: any) {
            console.error(`[Daemon] 写缓存失败: ${e.message}`);
            try { if (fs.existsSync(tmp)) fs.unlinkSync(tmp); } catch {}
        }
    });
}

// ── 市场发现 ──

async function discoverMarkets(): Promise<void> {
    const timestamps = getBarTimestamps();
    let added = 0;
    let requestCount = 0;
    let networkErrCount = 0;
    for (const sym of SYMBOLS) {
        for (const ts of timestamps) {
            const slug = buildSlug(sym, ts);
            if (cache.markets[slug]) continue;
            requestCount++;
            try {
                const res = await http.get(`${GAMMA_API}/events/slug/${slug}`);
                const event = res.data;
                if (!event?.markets?.length) continue;
                const mkt = event.markets[0];
                const tokenIds = typeof mkt.clobTokenIds === 'string'
                    ? JSON.parse(mkt.clobTokenIds) : mkt.clobTokenIds || [];
                const outcomes = typeof mkt.outcomes === 'string'
                    ? JSON.parse(mkt.outcomes) : mkt.outcomes || ['Up', 'Down'];
                cache.markets[slug] = {
                    slug,
                    conditionId: mkt.conditionId || mkt.condition_id || '',
                    tokens: tokenIds.map((id: string, i: number) => ({
                        tokenId: id, outcome: outcomes[i] || (i === 0 ? 'Up' : 'Down'),
                    })),
                    resolved: false,
                    winner: null,
                };
                added++;
            } catch (e: any) {
                const isAxiosErr = axios.isAxiosError(e);
                const is404 = isAxiosErr && e.response?.status === 404;
                if (!is404) {
                    console.error(`[Daemon] 发现市场失败 ${slug}: ${e.message?.slice(0, 60)}`);
                    if (isAxiosErr && !e.response) networkErrCount++;
                }
            }
        }
    }
    // Gamma 全量网络失败时，不清理旧缓存，避免“VPN 抖动 -> 缓存被掏空”。
    if (requestCount > 0 && networkErrCount >= requestCount) {
        // Even if market discovery fully failed this round, we still prune
        // bars that are already locally expired. Otherwise stale historical
        // orderbooks linger in cache and health checks misclassify the daemon
        // as unhealthy for hours after a transient Gamma outage.
        pruneExpiredMarketCache();
        flushCache();
        console.warn(
            `[Daemon] 发现市场阶段网络异常 (${networkErrCount}/${requestCount})，` +
            '保留旧缓存并等待下一轮重试',
        );
        return;
    }
    pruneExpiredMarketCache();
    if (added > 0) {
        console.log(`[Daemon] 发现 ${added} 个新市场, 共 ${Object.keys(cache.markets).length} 个活跃`);
        wsUpdateSubscription();
    }
    flushCache();
}

// ── REST 订单簿拉取 ──

let lastWsMessageAt = 0;

function isWsHealthy(): boolean {
    return ws?.readyState === WS.OPEN && (Date.now() / 1000 - lastWsMessageAt) < 15;
}

async function fetchOrderbooks(): Promise<void> {
    const ids = allTokenIds(true);
    if (ids.length === 0) return;

    // WS 在线且活跃时，只拉 WS 未覆盖的 token（兜底），大幅减少 REST 流量
    const idsToFetch = isWsHealthy()
        ? ids.filter(id => !cache.orderbooks[id] || (now() - cache.orderbooks[id].updated_at) > 30)
        : ids;
    if (idsToFetch.length === 0) return;

    const batchSize = 4;
    for (let i = 0; i < idsToFetch.length; i += batchSize) {
        const batch = idsToFetch.slice(i, i + batchSize);
        await Promise.all(batch.map(async (tokenId) => {
            try {
                const res = await http.get(`${CLOB_API}/book`, { params: { token_id: tokenId } });
                const data = res.data;
                const asks: Array<{ price: string; size: string }> = data?.asks || [];
                const bids: Array<{ price: string; size: string }> = data?.bids || [];
                const bestAsk = asks.length > 0 ? parseFloat(asks[0].price) : 999;
                const bestBid = bids.length > 0 ? parseFloat(bids[0].price) : 0;
                cache.orderbooks[tokenId] = { best_ask: bestAsk, best_bid: bestBid, asks, bids, updated_at: now() };
                cache.prices[tokenId] = { buy: bestAsk, sell: bestBid, updated_at: now() };
            } catch { /* 静默 */ }
        }));
    }
    flushCache();
}

// ── 结算检查 ──

async function checkSettlement(): Promise<void> {
    for (const [slug, mkt] of Object.entries(cache.markets)) {
        if (mkt.resolved) continue;
        if (!mkt.conditionId || !mkt.conditionId.startsWith('0x')) continue;
        // 只检查已过期的市场
        const m = slug.match(/-(\d+)$/);
        if (!m) continue;
        const endTs = parseInt(m[1]) + INTERVAL_15M;
        if (now() < endTs + 60) continue; // 结算后等 60s
        try {
            const res = await http.get(`${CLOB_API}/markets/${mkt.conditionId}`);
            const d = res.data;
            if (d?.closed) {
                const tokens = d?.tokens || [];
                const winner = tokens.find((t: any) => t.winner === true);
                if (winner) {
                    const o = String(winner.outcome ?? winner.name ?? '');
                    mkt.resolved = true;
                    mkt.winner = /^down$/i.test(o) ? 'Down' : 'Up';
                }
            }
        } catch { /* 静默 */ }
    }
    flushCache();
}

// ── WebSocket ──

let ws: WS | null = null;
const WS_RECONNECT_DELAY_MS = 3000;  // 固定 3s 重连，不再退避累积
const WS_FLAP_COOLDOWN_MS = 30_000;  // 频繁断开时延长重连，避免重连风暴
const WS_STALE_RECONNECT_SECONDS = 45; // WS 连着但超过该秒数无消息，判定假连接并软重连
let wsClosing = false;
let wsPingTimer: ReturnType<typeof setInterval> | null = null;
let wsReconnectTimer: ReturnType<typeof setTimeout> | null = null;
let wsSubscribedIds: Set<string> = new Set();
let wsOpenedAt = 0;
const WS_STABLE_THRESHOLD_MS = 30_000;
// 频繁断开检测：短时间多次断开视为 VPN/网络不稳，进入保活重连模式
const WS_DISCONNECT_WINDOW_MS = 3 * 60 * 1000;  // 3 分钟
const WS_DISCONNECT_RESTART_THRESHOLD = 5;       // 3 分钟内 >=5 次则进入保活重连
const wsDisconnectTimes: number[] = [];         // 最近几次断开时间戳

function getWsAgent(): HttpsProxyAgent<string> | undefined {
    const p = process.env.HTTPS_PROXY || process.env.https_proxy
           || process.env.HTTP_PROXY  || process.env.http_proxy;
    return p ? new HttpsProxyAgent(p) : undefined;
}

function wsConnect(): void {
    if (wsClosing) return;
    if (wsReconnectTimer) {
        clearTimeout(wsReconnectTimer);
        wsReconnectTimer = null;
    }
    try {
        const agent = getWsAgent();
        ws = new WS(WS_URL, agent ? { agent } : undefined);
    } catch {
        wsScheduleReconnect();
        return;
    }

    ws.on('open', () => {
        wsOpenedAt = Date.now();
        // 新连接必须强制重新订阅；否则服务端可能在 ~10s 后因“未订阅”主动断开
        wsSubscribedIds = new Set();
        // 重连后重置消息时间，避免沿用旧时间戳导致 watchdog 误判“假连接”
        lastWsMessageAt = now();
        console.log('[Daemon] WebSocket 已连接');
        wsUpdateSubscription();
        wsPingTimer = setInterval(() => {
            if (ws?.readyState !== WS.OPEN) return;
            try {
                ws.send('PING');
            } catch (e: any) {
                console.warn(`[Daemon] WS PING 发送失败: ${e?.message || e}`);
            }
        }, 10_000);
    });

    ws.on('message', (raw: WS.Data) => {
        const str = raw.toString();
        if (str === 'PONG') return;
        try {
            const msg = JSON.parse(str);
            wsHandleMessage(msg);
        } catch { /* ignore */ }
    });

    ws.on('error', (err: Error) => {
        console.error('[Daemon] WebSocket error:', err.message || String(err));
    });

    ws.on('close', (code: number, reason: Buffer) => {
        ws = null;
        wsSubscribedIds = new Set();
        if (wsPingTimer) { clearInterval(wsPingTimer); wsPingTimer = null; }
        if (!wsClosing) {
            const now = Date.now();
            const aliveMs = wsOpenedAt > 0 ? (now - wsOpenedAt) : 0;
            const reasonStr = (reason && reason.length > 0) ? reason.toString() : '(无)';
            // 记录断开原因便于排查：code 1006=异常关闭(含网络/代理断), 1000=正常关闭
            console.log(`[Daemon] WebSocket 断开 code=${code} reason=${reasonStr} 连接存活=${(aliveMs/1000).toFixed(1)}s`);
            wsDisconnectTimes.push(now);
            const cutoff = now - WS_DISCONNECT_WINDOW_MS;
            while (wsDisconnectTimes.length > 0 && wsDisconnectTimes[0] < cutoff) wsDisconnectTimes.shift();
            if (wsDisconnectTimes.length >= WS_DISCONNECT_RESTART_THRESHOLD) {
                console.log(
                    `[Daemon] ${WS_DISCONNECT_WINDOW_MS/60000} 分钟内断开 ${wsDisconnectTimes.length} 次，` +
                    `判定为 VPN/网络不稳，进入保活重连模式（${WS_FLAP_COOLDOWN_MS/1000}s）`,
                );
                wsDisconnectTimes.length = 0;
                setTimeout(() => {
                    if (!wsClosing && !ws) wsConnect();
                }, WS_FLAP_COOLDOWN_MS);
                return;
            }
            console.log(`[Daemon] ${WS_RECONNECT_DELAY_MS/1000}s 后重连...`);
            wsScheduleReconnect();
        }
    });
}

function wsScheduleReconnect(delayMs = WS_RECONNECT_DELAY_MS): void {
    if (wsClosing) return;
    if (wsReconnectTimer) return;
    wsReconnectTimer = setTimeout(() => {
        wsReconnectTimer = null;
        if (!wsClosing) wsConnect();
    }, delayMs);
}

function wsState(): string {
    if (!ws) return 'null';
    switch (ws.readyState) {
        case WS.CONNECTING: return 'connecting';
        case WS.OPEN: return 'open';
        case WS.CLOSING: return 'closing';
        case WS.CLOSED: return 'closed';
        default: return `unknown(${String(ws.readyState)})`;
    }
}

function wsWatchdogTick(): void {
    if (wsClosing) return;
    if (!ws) {
        wsScheduleReconnect(1000);
        return;
    }
    if (ws.readyState !== WS.OPEN) return;
    const age = now() - lastWsMessageAt;
    if (age <= WS_STALE_RECONNECT_SECONDS) return;
    console.warn(`[Daemon] WS 疑似卡死: ${age}s 无消息，执行软重连`);
    try {
        ws.terminate();
    } catch {}
}

function heartbeatTick(): void {
    const cacheAge = cache.updated_at > 0 ? now() - cache.updated_at : -1;
    const wsMsgAge = lastWsMessageAt > 0 ? now() - lastWsMessageAt : -1;
    console.log(
        `[Daemon] 心跳 ws=${wsState()} ws_msg_age=${wsMsgAge}s cache_age=${cacheAge}s ` +
        `markets=${Object.keys(cache.markets).length} tokens=${allTokenIds().length}`,
    );
}

function wsActiveTokenIds(): string[] {
    // 只返回当前周期和下一周期的 token（即将交易的），跳过 2 个周期后的（省 WS 流量）
    const n = now();
    const currentBar = n - (n % INTERVAL_15M);
    const cutoff = currentBar + INTERVAL_15M;
    const ids: string[] = [];
    for (const [slug, m] of Object.entries(cache.markets)) {
        if (m.resolved) continue;
        const match = slug.match(/-(\d+)$/);
        if (match) {
            const barTs = parseInt(match[1]);
            if (barTs > cutoff) continue;
        }
        for (const t of m.tokens) ids.push(t.tokenId);
    }
    return Array.from(new Set(ids));
}

function wsUpdateSubscription(): void {
    let ids = wsActiveTokenIds();
    // 官方文档要求连接后立即发送订阅，否则服务端会关闭连接。若当前/下一周期无 token 则用全部未结算 token 兜底
    if (ids.length === 0) ids = allTokenIds(true);
    if (ids.length === 0 || ws?.readyState !== WS.OPEN) return;
    const isFirst = wsSubscribedIds.size === 0;
    if (!isFirst && ids.length === wsSubscribedIds.size && ids.every(id => wsSubscribedIds.has(id))) {
        return;
    }
    try {
        ws!.send(JSON.stringify(
            isFirst ? { assets_ids: ids, type: 'market' } : { assets_ids: ids, operation: 'subscribe' }
        ));
    } catch (e: any) {
        console.warn(`[Daemon] WS 订阅发送失败: ${e?.message || e}`);
        return;
    }
    wsSubscribedIds = new Set(ids);
    console.log(`[Daemon] WS 订阅 ${ids.length} 个 token`);
}

function wsHandleMessage(msg: Record<string, unknown>): void {
    lastWsMessageAt = now();
    const eventType = msg.event_type as string | undefined;

    if (eventType === 'book') {
        const tokenId = msg.asset_id as string | undefined;
        const asks = (msg.asks ?? msg.sells) as Array<{ price: string; size: string }> | undefined;
        const bids = msg.bids as Array<{ price: string; size: string }> | undefined;
        const hasBids = Array.isArray(bids) && bids.length > 0;
        if (tokenId && asks && asks.length > 0) {
            const bestAsk = parseFloat(asks[0].price);
            const bestBid = hasBids ? parseFloat(bids[0].price) : (cache.orderbooks[tokenId]?.best_bid ?? 0);
            // 若 WS book 未带 bids，保留已有缓存的 bids（用副本避免后续被改写），避免买方竞争深度被清空
            const existing = cache.orderbooks[tokenId];
            const bidsToStore = hasBids ? bids.slice(0, 20) : (existing?.bids?.length ? existing.bids.slice() : []);
            cache.orderbooks[tokenId] = {
                best_ask: bestAsk,
                best_bid: bestBid,
                asks: asks.slice(0, 20),
                bids: bidsToStore,
                updated_at: now(),
            };
            cache.prices[tokenId] = { buy: bestAsk, sell: bestBid, updated_at: now() };
            flushCache();
        }
        return;
    }

    if (eventType === 'price_change') {
        const changes = msg.price_changes as Array<{
            asset_id?: string; best_ask?: string; best_bid?: string;
        }> | undefined;
        if (!Array.isArray(changes)) return;
        for (const pc of changes) {
            const tokenId = pc.asset_id;
            if (!tokenId) continue;
            const existing = cache.orderbooks[tokenId];
            const bestAsk = pc.best_ask != null ? parseFloat(pc.best_ask) : existing?.best_ask ?? 999;
            const bestBid = pc.best_bid != null ? parseFloat(pc.best_bid) : existing?.best_bid ?? 0;
            if (existing) {
                existing.best_ask = bestAsk;
                existing.best_bid = bestBid;
                existing.updated_at = now();
                if (pc.best_ask != null && existing.asks.length > 0) {
                    existing.asks[0] = { ...existing.asks[0], price: pc.best_ask };
                }
            } else {
                cache.orderbooks[tokenId] = {
                    best_ask: bestAsk, best_bid: bestBid,
                    asks: pc.best_ask ? [{ price: pc.best_ask, size: '0' }] : [],
                    bids: pc.best_bid ? [{ price: pc.best_bid, size: '0' }] : [],
                    updated_at: now(),
                };
            }
            cache.prices[tokenId] = { buy: bestAsk, sell: bestBid, updated_at: now() };
        }
        flushCache();
        return;
    }
}

// ── 主循环 ──

async function main(): Promise<void> {
    console.log('═'.repeat(60));
    console.log('  Market Data Daemon 启动');
    console.log(`  缓存文件: ${CACHE_FILE}`);
    console.log(`  监控币种: ${SYMBOLS.join(', ')}`);
    console.log('═'.repeat(60));

    // 写入 PID
    fs.writeFileSync(PID_FILE, String(process.pid), 'utf-8');

    // 启动时优先复用历史缓存（VPN 抖动时保证读链路不断）。
    loadCacheFromDisk();

    // 优雅关闭
    const shutdown = () => {
        console.log('\n[Daemon] 关闭中...');
        wsClosing = true;
        if (wsPingTimer) clearInterval(wsPingTimer);
        if (ws) {
            try {
                ws.close(1000, 'daemon shutdown');
            } catch {}
        }
        try {
            const pidOnDisk = fs.existsSync(PID_FILE)
                ? fs.readFileSync(PID_FILE, 'utf-8').trim()
                : '';
            if (pidOnDisk === String(process.pid)) fs.unlinkSync(PID_FILE);
        } catch {}
        process.exit(0);
    };
    process.on('SIGTERM', shutdown);
    process.on('SIGINT', shutdown);

    // 初始市场发现
    console.log('[Daemon] 初始市场发现...');
    await discoverMarkets();

    // 启动 WebSocket
    wsConnect();

    // 首次 REST 拉取
    await fetchOrderbooks();
    console.log(`[Daemon] 初始化完成: ${Object.keys(cache.markets).length} 市场, ${allTokenIds().length} token`);

    // 定时任务
    setInterval(() => discoverMarkets().catch(console.error), 60_000);
    // REST 轮询: WS 在线时 30s 兜底, WS 断线时 5s 补偿
    setInterval(() => {
        const interval = isWsHealthy() ? 30_000 : 5_000;
        if ((Date.now() - _lastRestPoll) >= interval) {
            _lastRestPoll = Date.now();
            fetchOrderbooks().catch(console.error);
        }
    }, 5_000);
    setInterval(() => checkSettlement().catch(console.error), 60_000);
    setInterval(() => wsUpdateSubscription(), 15_000);
    setInterval(() => wsWatchdogTick(), 10_000);
    setInterval(() => heartbeatTick(), 60_000);
}

let _lastRestPoll = 0;

main().catch((e) => {
    console.error('[Daemon] 致命错误:', e);
    process.exit(1);
});
