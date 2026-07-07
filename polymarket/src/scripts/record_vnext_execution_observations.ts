import * as fs from 'fs';
import * as path from 'path';
import WS from 'ws';
import { HttpsProxyAgent } from 'https-proxy-agent';
import { normalizeBookSide, topDepthUsd } from '../services/compare_execution';

const PROJECT_ROOT = path.resolve(__dirname, '..', '..', '..');
const POLY_DIR = path.resolve(PROJECT_ROOT, 'polymarket');
const PROCESSED_DIR = path.resolve(PROJECT_ROOT, 'data', 'processed');
const RAW_LOG_DIRS: Record<string, string> = {
  BTC_USDT: path.join(POLY_DIR, 'logs_vnext_btc_entry_exit_v1_raw'),
  ETH_USDT: path.join(POLY_DIR, 'logs_vnext_eth_entry_exit_v1_raw'),
};
const ORDERBOOK_PATHS: Record<string, string> = {
  BTC_USDT: path.join(PROCESSED_DIR, 'vnext_execution_orderbook_btc_usdt.jsonl'),
  ETH_USDT: path.join(PROCESSED_DIR, 'vnext_execution_orderbook_eth_usdt.jsonl'),
};
const LIFECYCLE_PATHS: Record<string, string> = {
  BTC_USDT: path.join(PROCESSED_DIR, 'vnext_execution_lifecycle_btc_usdt.jsonl'),
  ETH_USDT: path.join(PROCESSED_DIR, 'vnext_execution_lifecycle_eth_usdt.jsonl'),
};
const WS_URL = 'wss://ws-subscriptions-clob.polymarket.com/ws/market';
const SCAN_MS = 10000;
const PING_MS = 10000;

type Asset = 'BTC_USDT' | 'ETH_USDT';

type ActiveTrade = {
  asset: Asset;
  symbol: string;
  tradeId: string;
  tokenId: string;
  marketSlug: string;
  status: string;
  result: string;
  pnl?: number;
};

function getWsProxyAgent(): HttpsProxyAgent<string> | undefined {
  const proxy = process.env.HTTPS_PROXY || process.env.https_proxy || process.env.HTTP_PROXY || process.env.http_proxy;
  return proxy ? new HttpsProxyAgent(proxy) : undefined;
}

function ensureParent(filePath: string): void {
  fs.mkdirSync(path.dirname(filePath), { recursive: true });
}

function appendJsonl(filePath: string, payload: Record<string, unknown>): void {
  ensureParent(filePath);
  fs.appendFileSync(filePath, JSON.stringify(payload) + '\n', 'utf8');
}

function loadTradeArray(filePath: string): Array<Record<string, unknown>> {
  try {
    const raw = fs.readFileSync(filePath, 'utf8');
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed.filter((row) => row && typeof row === 'object') as Array<Record<string, unknown>> : [];
  } catch {
    return [];
  }
}

function readActiveTrades(): Map<string, ActiveTrade> {
  const out = new Map<string, ActiveTrade>();
  for (const [asset, dir] of Object.entries(RAW_LOG_DIRS) as Array<[Asset, string]>) {
    const rows = loadTradeArray(path.join(dir, 'prediction_trades.simulation.json'));
    for (const row of rows) {
      const tradeId = String(row.id || '').trim();
      const tokenId = String(row.tokenId || '').trim();
      const marketSlug = String(row.marketSlug || '').trim();
      if (!tradeId || !tokenId || !marketSlug) continue;
      const result = String(row.result || '').toLowerCase();
      if (result === 'win' || result === 'lose') continue;
      out.set(tradeId, {
        asset,
        symbol: asset.split('_', 1)[0],
        tradeId,
        tokenId,
        marketSlug,
        status: String(row.status || ''),
        result: String(row.result || ''),
        pnl: row.pnl == null ? undefined : Number(row.pnl),
      });
    }
  }
  return out;
}

class ObservationRecorder {
  private ws: WS | null = null;
  private pingTimer: NodeJS.Timeout | null = null;
  private activeTrades = new Map<string, ActiveTrade>();
  private lifecycleState = new Map<string, string>();
  private subscribed = new Set<string>();

  start(): void {
    this.scanAndRecordLifecycle();
    setInterval(() => this.scanAndRecordLifecycle(), SCAN_MS);
    this.connect();
  }

  private connect(): void {
    const agent = getWsProxyAgent();
    this.ws = new WS(WS_URL, agent ? { agent } : undefined);
    this.ws.on('open', () => {
      this.syncSubscriptions();
      this.pingTimer = setInterval(() => {
        if (this.ws && this.ws.readyState === WS.OPEN) {
          this.ws.send('PING');
        }
      }, PING_MS);
    });
    this.ws.on('message', (buf) => this.handleMessage(String(buf)));
    this.ws.on('close', () => {
      if (this.pingTimer) clearInterval(this.pingTimer);
      this.pingTimer = null;
      this.ws = null;
      setTimeout(() => this.connect(), 3000);
    });
    this.ws.on('error', () => {
      // close handler will retry
    });
  }

  private scanAndRecordLifecycle(): void {
    const latest = readActiveTrades();
    for (const trade of latest.values()) {
      const statusKey = JSON.stringify({ status: trade.status, result: trade.result, pnl: trade.pnl ?? null });
      if (this.lifecycleState.get(trade.tradeId) !== statusKey) {
        this.lifecycleState.set(trade.tradeId, statusKey);
        appendJsonl(LIFECYCLE_PATHS[trade.asset], {
          generated_at: new Date().toISOString(),
          ts: new Date().toISOString(),
          asset: trade.asset,
          symbol: trade.symbol,
          trade_id: trade.tradeId,
          token_id: trade.tokenId,
          market_slug: trade.marketSlug,
          status: trade.status,
          result: trade.result,
          pnl: trade.pnl ?? null,
          fill_status: trade.status.toLowerCase() === 'executed' ? 'filled' : 'queued',
          partial_fill_ratio: trade.status.toLowerCase() === 'executed' ? 1.0 : 0.0,
          queue_wait_seconds: 0.0,
          entry_fill_mode: trade.status.toLowerCase() === 'executed' ? 'bootstrap_taker' : 'queued',
          exit_fill_mode: 'observed_only',
        });
      }
    }
    this.activeTrades = latest;
    this.syncSubscriptions();
  }

  private syncSubscriptions(): void {
    if (!this.ws || this.ws.readyState !== WS.OPEN) return;
    const tokenIds = Array.from(new Set(Array.from(this.activeTrades.values()).map((trade) => trade.tokenId))).filter(Boolean);
    const next = new Set(tokenIds);
    const same = next.size === this.subscribed.size && Array.from(next).every((id) => this.subscribed.has(id));
    if (same) return;
    this.ws.send(JSON.stringify(this.subscribed.size === 0 ? { assets_ids: tokenIds, type: 'market' } : { assets_ids: tokenIds, operation: 'subscribe' }));
    this.subscribed = next;
  }

  private tradeByToken(tokenId: string): ActiveTrade[] {
    return Array.from(this.activeTrades.values()).filter((trade) => trade.tokenId === tokenId);
  }

  private handleMessage(raw: string): void {
    if (raw === 'PONG') return;
    let msg: any;
    try {
      msg = JSON.parse(raw);
    } catch {
      return;
    }
    const eventType = String(msg.event_type || '');
    if (eventType !== 'book' && eventType !== 'price_change') return;
    if (eventType === 'book') {
      const tokenId = String(msg.asset_id || '');
      if (!tokenId) return;
      const asks = normalizeBookSide(msg.asks || msg.sells || []);
      const bids = normalizeBookSide(msg.bids || msg.buys || []);
      const bestAsk = asks.length > 0 ? Math.min(...asks.map((level) => level.price)) : null;
      const bestBid = bids.length > 0 ? Math.max(...bids.map((level) => level.price)) : null;
      const spread = bestAsk != null && bestBid != null ? Math.max(0, bestAsk - bestBid) : null;
      for (const trade of this.tradeByToken(tokenId)) {
        appendJsonl(ORDERBOOK_PATHS[trade.asset], {
          generated_at: new Date().toISOString(),
          ts: new Date().toISOString(),
          asset: trade.asset,
          symbol: trade.symbol,
          trade_id: trade.tradeId,
          token_id: tokenId,
          market_slug: trade.marketSlug,
          event_type: 'book',
          best_ask: bestAsk,
          best_bid: bestBid,
          spread,
          ask_depth_top3: topDepthUsd(asks, 3),
          bid_depth_top3: topDepthUsd(bids, 3),
          asks_top: asks.slice(0, 3),
          bids_top: bids.slice(0, 3),
        });
      }
      return;
    }
    const changes = Array.isArray(msg.price_changes) ? msg.price_changes : [];
    for (const change of changes) {
      const tokenId = String(change.asset_id || '');
      if (!tokenId) continue;
      const bestAsk = change.best_ask == null ? null : Number(change.best_ask);
      const bestBid = change.best_bid == null ? null : Number(change.best_bid);
      const spread = bestAsk != null && bestBid != null ? Math.max(0, bestAsk - bestBid) : null;
      for (const trade of this.tradeByToken(tokenId)) {
        appendJsonl(ORDERBOOK_PATHS[trade.asset], {
          generated_at: new Date().toISOString(),
          ts: new Date().toISOString(),
          asset: trade.asset,
          symbol: trade.symbol,
          trade_id: trade.tradeId,
          token_id: tokenId,
          market_slug: trade.marketSlug,
          event_type: 'price_change',
          best_ask: bestAsk,
          best_bid: bestBid,
          spread,
          ask_depth_top3: null,
          bid_depth_top3: null,
          asks_top: [],
          bids_top: [],
        });
      }
    }
  }
}

new ObservationRecorder().start();
