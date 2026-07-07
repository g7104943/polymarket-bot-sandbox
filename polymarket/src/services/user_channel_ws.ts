/**
 * Polymarket User Channel WebSocket 客户端
 * 端点: wss://ws-subscriptions-clob.polymarket.com/ws/user
 *
 * 接收与当前 API key 关联的订单和成交事件，用于真实交易的实时订单追踪，
 * 替代之前的 REST 轮询查询成交。
 *
 * 事件类型:
 *   - trade: MATCHED → MINED → CONFIRMED / RETRYING → FAILED
 *   - order: PLACEMENT / UPDATE / CANCELLATION
 */

import WS from 'ws';
import { HttpsProxyAgent } from 'https-proxy-agent';

const USER_WS_URL = 'wss://ws-subscriptions-clob.polymarket.com/ws/user';
const PING_INTERVAL_MS = 10_000;
const RECONNECT_DELAY_MS = 3_000;
const MAX_RECONNECT_DELAY_MS = 60_000;

// ─── 事件数据结构 ──────────────────────────────────────────────

export interface TradeEvent {
    asset_id: string;
    event_type: 'trade';
    id: string;             // trade ID
    market: string;         // condition ID
    status: 'MATCHED' | 'MINED' | 'CONFIRMED' | 'RETRYING' | 'FAILED';
    side: 'BUY' | 'SELL';
    price: string;
    size: string;
    taker_order_id: string;
    maker_orders?: Array<{
        order_id: string;
        matched_amount: string;
        price: string;
        asset_id: string;
    }>;
    timestamp: string;
    matchtime?: string;
    outcome?: string;
    owner?: string;
    trade_owner?: string;
}

export interface OrderEvent {
    asset_id: string;
    event_type: 'order';
    id: string;             // order ID
    market: string;         // condition ID
    type: 'PLACEMENT' | 'UPDATE' | 'CANCELLATION';
    side: 'BUY' | 'SELL';
    price: string;
    original_size: string;
    size_matched: string;
    outcome?: string;
    owner?: string;
    timestamp: string;
}

export type OnTradeCallback = (event: TradeEvent) => void;
export type OnOrderCallback = (event: OrderEvent) => void;

// ─── 配置 ──────────────────────────────────────────────────────

export interface UserChannelWSOptions {
    apiKey: string;
    apiSecret: string;
    apiPassphrase: string;
    /** 需要监听的 condition ID 列表（动态） */
    getSubscribedMarkets: () => string[];
    onTrade: OnTradeCallback;
    onOrder: OnOrderCallback;
    verbose?: boolean;
}

// ─── 实现 ──────────────────────────────────────────────────────

function getWsProxyAgent(): HttpsProxyAgent<string> | undefined {
    const proxy = process.env.HTTPS_PROXY || process.env.https_proxy
                || process.env.HTTP_PROXY || process.env.http_proxy;
    if (proxy) return new HttpsProxyAgent(proxy);
    return undefined;
}

export class UserChannelWS {
    private ws: WebSocket | null = null;
    private pingTimer: ReturnType<typeof setInterval> | null = null;
    private reconnectDelay = RECONNECT_DELAY_MS;
    private isClosing = false;
    private lastSubscribedMarkets: Set<string> = new Set();

    private readonly apiKey: string;
    private readonly apiSecret: string;
    private readonly apiPassphrase: string;
    private readonly getSubscribedMarkets: () => string[];
    private readonly onTrade: OnTradeCallback;
    private readonly onOrder: OnOrderCallback;
    private readonly verbose: boolean;

    constructor(options: UserChannelWSOptions) {
        this.apiKey = options.apiKey;
        this.apiSecret = options.apiSecret;
        this.apiPassphrase = options.apiPassphrase;
        this.getSubscribedMarkets = options.getSubscribedMarkets;
        this.onTrade = options.onTrade;
        this.onOrder = options.onOrder;
        this.verbose = options.verbose ?? false;
    }

    start(): void {
        if (this.ws?.readyState === WebSocket.OPEN) return;
        this.isClosing = false;
        this.connect();
    }

    stop(): void {
        this.isClosing = true;
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
            this.ws = new WS(USER_WS_URL, agent ? { agent } : undefined) as unknown as WebSocket;
        } catch (e) {
            if (this.verbose) console.error('[UserWS] connect error', e);
            this.scheduleReconnect();
            return;
        }

        this.ws.onopen = () => {
            this.reconnectDelay = RECONNECT_DELAY_MS;
            console.log('\n📡 [UserWS] User Channel WebSocket 已连接');
            this.sendAuthSubscribe();
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

        this.ws.onerror = () => { /* handled in onclose */ };

        this.ws.onclose = () => {
            this.ws = null;
            if (this.pingTimer) {
                clearInterval(this.pingTimer);
                this.pingTimer = null;
            }
            if (!this.isClosing) {
                console.log('\n📡 [UserWS] 连接断开，将重连...');
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
            console.log('\n📡 [UserWS] 重连中...');
            this.connect();
        }, this.reconnectDelay);
        this.reconnectDelay = Math.min(MAX_RECONNECT_DELAY_MS, this.reconnectDelay * 1.5);
    }

    /** 连接后发送带认证的初始订阅 */
    private sendAuthSubscribe(): void {
        if (this.ws?.readyState !== WebSocket.OPEN) return;
        const markets = [...new Set(this.getSubscribedMarkets())].filter(Boolean);
        const msg = {
            auth: {
                apiKey: this.apiKey,
                secret: this.apiSecret,
                passphrase: this.apiPassphrase,
            },
            markets,
            type: 'user',
        };
        this.ws.send(JSON.stringify(msg));
        this.lastSubscribedMarkets = new Set(markets);
        if (this.verbose) {
            console.log(`[UserWS] 已发送认证订阅，markets: ${markets.length} 个`);
        }
    }

    /** 外部调用：动态更新订阅的 market 列表（不重连） */
    syncSubscribe(): void {
        const markets = [...new Set(this.getSubscribedMarkets())].filter(Boolean);
        const set = new Set(markets);
        const same = set.size === this.lastSubscribedMarkets.size
            && markets.every((m) => this.lastSubscribedMarkets.has(m));
        if (!same && this.ws?.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ markets, type: 'user' }));
            this.lastSubscribedMarkets = set;
            if (this.verbose) {
                console.log(`[UserWS] 订阅已更新，markets: ${markets.length} 个`);
            }
        }
    }

    private handleMessage(msg: Record<string, unknown>): void {
        const eventType = msg.event_type as string | undefined;

        if (eventType === 'trade') {
            try { this.onTrade(msg as unknown as TradeEvent); } catch { /* isolation */ }
            return;
        }

        if (eventType === 'order') {
            try { this.onOrder(msg as unknown as OrderEvent); } catch { /* isolation */ }
            return;
        }

        if (this.verbose && eventType) {
            console.log(`[UserWS] 未知事件类型: ${eventType}`);
        }
    }

    isConnected(): boolean {
        return this.ws?.readyState === WebSocket.OPEN;
    }
}
