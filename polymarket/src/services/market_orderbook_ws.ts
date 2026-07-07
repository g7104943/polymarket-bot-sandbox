/**
 * Polymarket 市场频道 WebSocket：订阅 agg_orderbook 实时最佳卖价（ask）与深度
 * 端点: wss://ws-subscriptions-clob.polymarket.com/ws/market
 * 消息: book（订单簿快照）、price_change（价格变动，含 best_ask）
 * 当 best_ask ≤ 阈值时由外部触发下单，避免轮询延迟
 *
 * 本地订单簿缓存：收到 book 时缓存 asks；收到 price_change 时用缓存的 asks 并更新 best_ask，
 * 始终向回调传入 asksSnapshot，避免触发后回退 REST 拉单导致价格已变（如 0.52→0.54）。
 */

import * as fs from 'fs';
import * as path from 'path';
import WS from 'ws';
import { HttpsProxyAgent } from 'https-proxy-agent';

const WS_URL = 'wss://ws-subscriptions-clob.polymarket.com/ws/market';
const DAEMON_CACHE_PATH = path.resolve(__dirname, '..', '..', 'market_data_cache.json');

function isDaemonActive(): boolean {
    try {
        const stat = fs.statSync(DAEMON_CACHE_PATH);
        return (Date.now() - stat.mtimeMs) < 30_000;
    } catch {
        return false;
    }
}

function getWsProxyAgent(): HttpsProxyAgent<string> | undefined {
    const proxy = process.env.HTTPS_PROXY || process.env.https_proxy
                || process.env.HTTP_PROXY || process.env.http_proxy;
    if (proxy) return new HttpsProxyAgent(proxy);
    return undefined;
}
const PING_INTERVAL_MS = 10_000;
const RECONNECT_DELAY_MS = 3_000;
const MAX_RECONNECT_DELAY_MS = 60_000;

export type OnBestAskCallback = (assetId: string, bestAsk: number, asksSnapshot?: Array<{ price: string; size: string }>, isFullBook?: boolean) => void;

export interface MarketOrderbookWSOptions {
    /** 获取当前需要订阅的 token ID 列表（如监控队列中的 tokenId） */
    getSubscribedTokenIds: () => string[];
    /** 当某 token 的 best_ask 更新时回调（可用于高价/低价监控触发） */
    onBestAsk: OnBestAskCallback;
    /** 是否静默（少打日志） */
    verbose?: boolean;
}

export class MarketOrderbookWS {
    private ws: WebSocket | null = null;
    private pingTimer: ReturnType<typeof setInterval> | null = null;
    private reconnectDelay = RECONNECT_DELAY_MS;
    private getSubscribedTokenIds: () => string[];
    private onBestAsk: OnBestAskCallback;
    private verbose: boolean;
    private lastSubscribedIds: Set<string> = new Set();
    private isClosing = false;
    /** 每 asset 最近一次 book 的 asks，用于 price_change 时仍能带 snapshot 触发，避免 REST 拉单延迟 */
    private lastAsksByAsset: Map<string, Array<{ price: string; size: string }>> = new Map();
    private daemonFallbackTimer: ReturnType<typeof setInterval> | null = null;
    private skippedByDaemon = false;

    constructor(options: MarketOrderbookWSOptions) {
        this.getSubscribedTokenIds = options.getSubscribedTokenIds;
        this.onBestAsk = options.onBestAsk;
        this.verbose = options.verbose ?? false;
    }

    start(): void {
        if (isDaemonActive()) {
            console.log('[WS] Market Data Daemon 活跃，跳过 per-process WebSocket');
            this.skippedByDaemon = true;
            if (!this.daemonFallbackTimer) {
                this.daemonFallbackTimer = setInterval(() => {
                    if (!isDaemonActive() && this.skippedByDaemon) {
                        console.log('[WS] Daemon 已失活，启动 per-process WebSocket 接管');
                        this.skippedByDaemon = false;
                        this.isClosing = false;
                        this.connect();
                    }
                }, 30_000);
            }
            return;
        }
        if (this.ws?.readyState === WebSocket.OPEN) return;
        this.isClosing = false;
        this.connect();
    }

    stop(): void {
        this.isClosing = true;
        if (this.daemonFallbackTimer) {
            clearInterval(this.daemonFallbackTimer);
            this.daemonFallbackTimer = null;
        }
        if (this.pingTimer) {
            clearInterval(this.pingTimer);
            this.pingTimer = null;
        }
        if (this.ws) {
            this.ws.close(1000, 'client stop');
            this.ws = null;
        }
    }

    private connect(): void {
        if (this.isClosing) return;
        try {
            const agent = getWsProxyAgent();
            this.ws = new WS(WS_URL, agent ? { agent } : undefined) as unknown as WebSocket;
        } catch (e) {
            if (this.verbose) console.error('[WS] connect error', e);
            this.scheduleReconnect();
            return;
        }

        this.ws.onopen = () => {
            this.reconnectDelay = RECONNECT_DELAY_MS;
            console.log('\n📡 市场频道 WebSocket 已连接');
            this.sendSubscribe();
            this.startPing();
        };

        this.ws.onmessage = (event: { data: unknown }) => {
            if (typeof event.data === 'string' && event.data === 'PONG') return;
            try {
                const msg = JSON.parse(event.data as string);
                this.handleMessage(msg);
            } catch {
                // ignore parse error
            }
        };

        this.ws.onerror = () => {
            // 错误细节在 onclose 处理
        };

        this.ws.onclose = () => {
            this.ws = null;
            if (this.pingTimer) {
                clearInterval(this.pingTimer);
                this.pingTimer = null;
            }
            if (!this.isClosing) {
                console.log('\n📡 市场频道 WebSocket 已断开，将重连...');
                this.scheduleReconnect();
            }
        };
    }

    private startPing(): void {
        this.pingTimer = setInterval(() => {
            if (this.ws?.readyState === WebSocket.OPEN) this.ws.send('PING');
        }, PING_INTERVAL_MS);
    }

    private scheduleReconnect(): void {
        if (this.isClosing) return;
        setTimeout(() => {
            if (this.isClosing) return;
            console.log('\n📡 市场频道 WebSocket 重连中...');
            this.connect();
        }, this.reconnectDelay);
        this.reconnectDelay = Math.min(MAX_RECONNECT_DELAY_MS, this.reconnectDelay * 1.5);
    }

    /** 发送初始订阅或更新订阅列表（assets_ids） */
    sendSubscribe(): void {
        const ids = this.getSubscribedTokenIds();
        const uniq = [...new Set(ids)].filter(Boolean);
        if (uniq.length === 0) {
            // 文档示例：连接时发送 type: "market" + assets_ids
            if (this.ws?.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ assets_ids: [], type: 'market' }));
            }
            this.lastSubscribedIds.clear();
            return;
        }
        if (this.ws?.readyState !== WebSocket.OPEN) return;
        // 首次连接用 type: "market"；后续用 operation: "subscribe"
        const isFirst = this.lastSubscribedIds.size === 0;
        this.ws.send(JSON.stringify(
            isFirst ? { assets_ids: uniq, type: 'market' } : { assets_ids: uniq, operation: 'subscribe' }
        ));
        this.lastSubscribedIds = new Set(uniq);
    }

    /** 由外部定时调用，同步监控队列中的 token 列表并更新订阅 */
    syncSubscribe(): void {
        const ids = this.getSubscribedTokenIds();
        const uniq = [...new Set(ids)].filter(Boolean);
        const set = new Set(uniq);
        const same = set.size === this.lastSubscribedIds.size && uniq.every((id) => this.lastSubscribedIds.has(id));
        if (!same && this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ assets_ids: uniq, operation: 'subscribe' }));
            this.lastSubscribedIds = set;
        }
    }

    private handleMessage(msg: Record<string, unknown>): void {
        const eventType = msg.event_type as string | undefined;
        if (eventType === 'book') {
            const assetId = msg.asset_id as string | undefined;
            const asks = (msg.asks ?? msg.sells) as Array<{ price: string; size: string }> | undefined;
            if (assetId && asks && asks.length > 0) {
                const asksCopy = asks.map((a) => ({ price: String(a.price), size: String(a.size) }));
                this.lastAsksByAsset.set(assetId, asksCopy);

                let bestAsk = Infinity;
                for (const ask of asksCopy) {
                    const p = parseFloat(ask.price);
                    if (Number.isFinite(p) && p < bestAsk) bestAsk = p;
                }
                if (Number.isFinite(bestAsk)) {
                    // 回调传独立副本，避免下游 trader 间共享引用导致串改
                    this.onBestAsk(assetId, bestAsk, asksCopy.map((a) => ({ ...a })), true);
                }
            }
            return;
        }
        if (eventType === 'price_change') {
            const priceChanges = msg.price_changes as Array<{
                asset_id?: string;
                best_ask?: string;
                best_bid?: string;
            }> | undefined;
            if (Array.isArray(priceChanges)) {
                for (const pc of priceChanges) {
                    const assetId = pc.asset_id;
                    const bestAskStr = pc.best_ask;
                    if (assetId && bestAskStr != null) {
                        const bestAsk = parseFloat(bestAskStr);
                        if (!Number.isFinite(bestAsk)) continue;
                        const cached = this.lastAsksByAsset.get(assetId);
                        if (cached && cached.length > 0) {
                            const snapshot = cached.map((level) => ({ ...level }));
                            snapshot[0] = { ...snapshot[0], price: bestAskStr };
                            this.onBestAsk(assetId, bestAsk, snapshot, false);
                        } else {
                            this.onBestAsk(assetId, bestAsk, undefined, false);
                        }
                    }
                }
            }
            return;
        }
    }
}
