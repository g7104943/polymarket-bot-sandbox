/**
 * 预测执行器
 * 核心逻辑：获取 Python 模型预测，查找对应的 Polymarket 市场，执行交易
 */

import * as fs from 'fs';
import * as path from 'path';
import { AssetType, Chain, ClobClient, OrderType, Side, SignatureTypeV2 } from '@polymarket/clob-client-v2';
import axios from 'axios';
import { ethers } from 'ethers';
import { PREDICTION_ENV, PredictionEnvConfig, createPredictionEnv, TradingMode, printPredictionConfig } from '../config/prediction_env';
import { 
    PredictionResult, 
    getHighConfidencePredictions, 
    checkPredictionApiHealth,
    printPredictions,
    getPredictionFilePath,
    readLocalPredictionsRaw,
    parseLocalPredictions,
} from '../utils/prediction_api';
import { 
    findAllMarkets, 
    MarketSearchResult, 
    getTokenForDirection,
    printMarketResults,
    getMarketResult,
    getMarketResultByConditionId,
    getTokenPrice,
    getPositionValue,
    getOrderBook,
} from './market_finder';
import { TradeLogger, getTradeLogger, createTradeLogger, TradeLogEntry, TradeLogModeScope } from '../models/trade_log';
import { IOrderPoster, SingleWalletOrderPoster } from './order_poster';
import { redeemPositionsGasless } from './claim_relayer';
import { BookLevel, normalizeBookSide, simulateQueueAwareBuyFill } from './compare_execution';

// ============================================================
// 常量
// ============================================================

const MIN_ORDER_SIZE_USD = 1.0;  // Polymarket 最小订单金额
const DEFAULT_EXCHANGE_MIN_SHARES = 5.0;  // 交易所返回的当前最小 shares 约束（观察到为 5）
const POLYMARKET_CTF_EXCHANGE_V2 = '0xE111180000d2663C0091e4f400237545B87B996B'.toLowerCase();
const POLYMARKET_CTF_EXCHANGE_V1_LEGACY = '0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E'.toLowerCase();
const POLYMARKET_EXCHANGE_SPENDER = POLYMARKET_CTF_EXCHANGE_V2;
const POLYMARKET_PUSD_COLLATERAL = '0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB';
const ERC20_BALANCE_ALLOWANCE_ABI = [
    'function balanceOf(address) view returns (uint256)',
    'function allowance(address,address) view returns (uint256)',
];
const RETRY_LIMIT = 30;
const RETRY_INTERVAL_MS = 2000;  // 2秒重试间隔
/** 实盘及内部金额统一保留小数位数（USDC 最多 2 位） */
const USD_DECIMALS = 2;
const roundUsd = (x: number): number => (Number.isFinite(x) ? Math.round(x * 100) / 100 : 0);

/** 从 market slug（如 sol-updown-15m-1770040800）解析周期开始时间戳 */
function parsePeriodStartFromSlug(slug: string | undefined): number | null {
    if (!slug || typeof slug !== 'string') return null;
    const m = slug.match(/-(\d+)$/);
    return m ? parseInt(m[1], 10) : null;
}

/** 当前北京时间，24小时制，如 "13:28" */
function nowBeijing(): string {
    return new Date().toLocaleTimeString('zh-CN', { timeZone: 'Asia/Shanghai', hour: '2-digit', minute: '2-digit', hour12: false });
}

function normalizeSymbol(raw: unknown): string {
    return String(raw ?? '')
        .replace(/\/.*$/, '')
        .trim()
        .toUpperCase();
}

/** 格式化为北京时间周期，如 22:00～22:15 北京时间（periodStartTs = 周期开始，periodSec = 窗口秒数） */
function formatPeriodBeijing(periodStartTs: number, periodSec: number = 900): string {
    const start = new Date(periodStartTs * 1000);
    const end = new Date((periodStartTs + periodSec) * 1000);
    const fmt = (d: Date) => d.toLocaleTimeString('zh-CN', { timeZone: 'Asia/Shanghai', hour: '2-digit', minute: '2-digit', hour12: false });
    return `${fmt(start)}～${fmt(end)} 北京时间`;
}

// ============================================================
// 类型定义
// ============================================================

interface ExecutionResult {
    symbol: string;
    success: boolean;
    /** 因本市场已由监控队列或前序流程成交而跳过，不算失败 */
    skippedReason?: 'already_bought';
    tradeLogId?: string;
    amount?: number;
    actualAmount?: number;  // 实际成交金额（部分成交时与 amount 不同）
    direction?: 'UP' | 'DOWN';
    error?: string;
    /** 周期显示（如 11:45～12:00 北京时间），用于本轮摘要失败行统一带周期 */
    periodDisplay?: string;
    /** 是否有未成交部分进入持续监控 */
    pendingMonitor?: boolean;
}

interface ExecutorState {
    isRunning: boolean;
    currentCapital: number;
    todayTradeCount: number;
    lastExecutionTime: Date | null;
    marketCache: Map<string, MarketSearchResult>;
    // 风控状态
    peakCapital: number;
    consecutiveLosses: number;
    formalConsecutiveLosses: number;
    provisionalConsecutiveLosses: number;
    tradingPaused: boolean;
    pauseReason: string | null;
    pausedKBarsRemaining: number;  // 剩余暂停的 K 线数（触发止损后使用）
    lastLossTs: number;            // 上次亏损交易的时间戳（ms），用于日志
    everReachedCap: boolean;       // 资金是否曾到过 $60K（流动性封顶后保守模式）
    drawdownHaltStartTime: number | null;  // 熔断开始时间戳(ms)，null=未熔断
}

/** GTD 真实订单追踪（由 UserChannelWS 回调驱动） */
interface LiveOrderTracker {
    orderId: string;
    conditionId: string;
    symbol: string;
    side: 'BUY' | 'SELL';
    price: number;
    sizeOrdered: number;
    amountUsd: number;
    sizeFilled: number;
    filledUsd: number;
    logEntryId: string;
    status: 'pending' | 'partial' | 'filled' | 'cancelled' | 'failed';
    createdAt: number;
    expirationSec: number;
    /** 用于阻塞等待 WS 确认的 resolve 回调 */
    resolve?: (result: { amountUsd: number; tokens: number; source: string }) => void;
}

interface StaleLiveOrderCandidate {
    entry: TradeLogEntry;
    orderId: string;
    limitPrice: number;
    orderAgeSec: number;
    timeToExpirySec: number;
    currentPrice: number;
    priceDriftTicks: number;
    cancelReasonKind?: 'stale' | 'late_fill_timeout';
    lateFillCancelMinutes?: number;
}

interface LiveSettlementOverride {
    result: 'win' | 'lose';
    totalPnL?: number;
    settledPnL?: number;
    source?: string;
}

interface PublicActivityTrade {
    conditionId: string;
    marketSlug: string;
    tokenId: string;
    tokenOutcome: string;
    transactionHash: string;
    timestampIso: string;
    timestampSec: number;
    usdcSize: number;
    size: number;
    price: number;
}

interface ExchangeExecutionConstraint {
    tokenId: string;
    marketSlug: string;
    minOrderSizeShares: number;
    tickSize: number | null;
    retrievedAt: string;
    source: 'official_book' | 'last_live_reject' | 'default_fallback';
}

interface FixedOrderExecutionPlan {
    requestedAmountUsd: number;
    finalAmountUsd: number;
    requestedShares: number;
    finalShares: number;
    uplifted: boolean;
    upliftedNotionalUsd: number;
    constraint: ExchangeExecutionConstraint;
    executable: boolean;
    gt5PreservationStatus: 'exact_preservation' | 'not_applicable' | 'contract_violation';
    compressionReason?: 'min_shares_floor' | 'unexpected_floor_violation';
    failureReason?: string;
}

interface DynamicRungExecutionPlan {
    rungIndex: number;
    rungPrice: number;
    originalBetUsd: number;
    requestedShares: number;
    minExecutableNotionalUsd: number;
    redistributedNotionalUsd: number;
    adjustedBetUsd: number;
    adjustedShares: number;
    uplifted: boolean;
    kept: boolean;
    gt5PreservationStatus: 'exact_preservation' | 'compressed_but_not_floored' | 'not_applicable' | 'contract_violation';
    compressionReason?: string;
}

interface DynamicOrderExecutionPlan {
    constraint: ExchangeExecutionConstraint;
    attemptedRungs: number;
    upliftedDynamicRungs: number;
    keptDynamicRungs: number;
    droppedDynamicRungs: number;
    selectedRungs: DynamicRungExecutionPlan[];
    allRungs: DynamicRungExecutionPlan[];
    requestedBudgetUsd: number;
    totalSelectedNotionalUsd: number;
    totalUpliftedNotionalUsd: number;
    executionCompressionUsd: number;
    executionCompressionReason?: string;
    decision: 'dynamic_min_shares_satisfied' | 'compressed_dynamic_rungs' | 'skipped_precheck_min_shares_budget_insufficient';
    failureReason?: string;
}

interface RequestedNotionalResolution {
    gateAmountUsd: number;
    targetNotionalUsd: number;
    requestedNotionalUsd: number;
    requestLayerCompressionUsd: number;
    requestCompressionMode: string;
    sizingMode: string;
    bankrollPct: number;
    confidenceBandScale: number;
    effectiveSizingPct: number;
    currentCapital: number;
    runtimeSizingCapPct?: number;
    runtimeSizingCapReason?: string;
}

interface ExecutionV2Profile {
    version: string;
    shadowOnly: boolean;
    chaseEnabled: boolean;
    chaseRounds: number;
    chaseWaitSec: number;
    firstChaseWaitSec: number;
    nextChaseGapSec: number;
    chaseTickLift: number;
    chaseAbsoluteMaxPrice: number;
    chaseMinConfidenceBand: number;
    chaseMinTimeToExpirySec: number;
    directionContinuationTicks: number;
    directionContinuationWindowSec: number;
    finalAggressiveWindowSec: number;
    finalAggressiveTickLift: number;
    finalAggressiveMaxPrice: number;
    cancelOnAdverseMoveTicks: number;
    cancelOnSignalInvalidation: boolean;
    cancelOnQueueStall: boolean;
    queueStallWindowSec: number;
    staleCancelMinAgeSec: number;
    staleCancelMinTimeToExpirySec: number;
    lateFillCancelMinutes: number;
}

type ExecutionV2State =
    | 'initial_ladder_posted'
    | 'waiting_fill'
    | 'chase_round_n'
    | 'final_aggressive_window'
    | 'cancelled_on_invalidation'
    | 'cancelled_on_stale_queue'
    | 'cancelled_pre_expiry_timeout'
    | 'expired_unfilled'
    | 'fully_filled';

interface ExecutionV2EventRecord {
    timestamp: string;
    symbol: string;
    direction: 'UP' | 'DOWN';
    marketSlug?: string;
    conditionId?: string;
    mode: TradeLogModeScope;
    eventType:
        | 'initial_ladder_posted'
        | 'request_notional_finalized'
        | 'posted_notional_reallocated'
        | 'compression_bypassed'
        | 'chase_reprice_posted'
        | 'chase_reprice_filled'
        | 'final_aggressive_posted'
        | 'cancelled_on_direction_invalidation'
        | 'cancelled_on_adverse_move'
        | 'cancelled_on_stale_queue'
        | 'cancelled_pre_expiry_timeout'
        | 'expired_unfilled'
        | 'fully_filled';
    shadowOnly: boolean;
    targetNotionalUsd?: number;
    requestedAmountUsd?: number;
    postedNotionalUsd?: number;
    filledNotionalUsd?: number;
    remainingAmountUsd?: number;
    initialPrice?: number;
    currentPrice?: number;
    currentMaxPrice?: number;
    chaseRound?: number;
    chaseTickLift?: number;
    finalAggressiveWindowSec?: number;
    finalAggressiveMaxPrice?: number;
    sizingMode?: string;
    bankrollPct?: number;
    confidenceBandScale?: number;
    effectiveSizingPct?: number;
    currentCapital?: number;
    reason?: string;
}

interface PriceMonitorEntry {
    prediction: PredictionResult;
    marketResult: MarketSearchResult;
    targetPeriodEndTs: number;  // 订单开始时间（K线周期开始）
    monitorStartTime: Date;      // 开始监控的时间
    monitorEndTime: Date;        // 监控结束时间（订单开始前5分钟）
    initialPrice: number;        // 初始价格
    triggerPrice?: number;        // 触发时的价格（满足条件时的价格）
    isProcessing?: boolean;      // 是否正在处理中（避免重复触发）
    /** WebSocket book 消息的订单簿快照，有则下单时跳过 REST getOrderBook（~100ms 提速） */
    asksSnapshot?: Array<{ price: string; size: string }>;
    /** 提前签名订单（监听到下单最快框架：触发时只 POST 不签名）；多钱包时可为多笔 */
    preSignedOrder?: unknown;
    executionProfile?: ExecutionV2Profile;
    executionState?: ExecutionV2State;
    requestedAmountUsd?: number;
    remainingAmountUsd?: number;
    currentMaxPrice?: number;
    highestObservedPrice?: number;
    chaseRound?: number;
    lastChaseAt?: number;
    initialLimitPrice?: number;
}

interface LiquidityMonitorEntry {
    prediction: PredictionResult;
    marketResult: MarketSearchResult;
    targetPeriodEndTs: number;  // 订单开始时间（K线周期开始）
    monitorStartTime: Date;      // 开始监控的时间
    monitorEndTime: Date;        // 监控结束时间（订单开始前5分钟）
    remainingAmount: number;     // 剩余需要买入的金额
    originalAmount: number;      // 原始下单金额
    logEntryId: string;          // 关联的交易日志ID
    totalBoughtAmount: number;   // 已买入的总金额（用于避免买多）
    minPrice: number;            // 最低买入价格（0.25）
    maxPrice: number;            // 最高买入价格（与 this.env.MAX_TOKEN_PRICE 一致，默认 0.54）
    executionProfile?: ExecutionV2Profile;
    executionState?: ExecutionV2State;
    highestObservedPrice?: number;
    chaseRound?: number;
    lastChaseAt?: number;
    initialLimitPrice?: number;
}

// ─── 两阶段限价单: Phase 1 订单跟踪 ───────────────────
interface PhaseOrderEntry {
    /** Phase 1 限价单的 orderID（用于取消） */
    orderID: string;
    /** 关联的 tokenID */
    tokenID: string;
    /** 目标周期 */
    targetPeriodEndTs: number;
    /** 币对 symbol（如 "BTC"） */
    symbol: string;
    /** Phase 1 预测方向 */
    direction: 'UP' | 'DOWN';
    /** 实际成交金额（$0 = 未成交/挂单中） */
    amountUsd: number;
    /** 计划下单金额（用于 Phase 2 计算剩余仓位） */
    plannedAmountUsd: number;
    /** 限价 */
    limitPrice: number;
    /** Phase 1 置信度 */
    confidence: number;
    /** 创建时间 */
    createdAt: Date;
    /** 是否已成交（模拟模式用订单簿判断，LIVE 模式用 API 查询） */
    filled: boolean;
}

// ─── 模拟限价单持续监控：挂单池 ───────────────────────────
/**
 * 模拟限价单挂单：下单时 best_ask > limit 的订单，资金冻结，
 * 每 5s 轮询订单簿，一旦 best_ask <= limit 即成交；
 * 市场周期结束（targetPeriodEndTs + PERIOD_SECONDS）后自动取消退款。
 */
interface PendingSimOrder {
    /** 唯一 ID */
    id: string;
    /** 币对 symbol (如 "BTC") */
    symbol: string;
    /** 预测方向 */
    direction: 'UP' | 'DOWN';
    /** 冻结金额（下注金额） */
    frozenAmount: number;
    /** 限价 */
    limitPrice: number;
    /** token ID（用于查订单簿） */
    tokenId: string;
    /** token outcome（Up/Down，用于记录交易） */
    tokenOutcome: string;
    /** 置信度 */
    confidence: number;
    /** 市场信息（用于记录交易） */
    marketTitle: string;
    conditionId: string;
    marketSlug: string;
    /** 目标周期开始时间戳（秒） */
    targetPeriodEndTs: number;
    /** 挂单创建时间 */
    createdAt: Date;
    /** 订单到期时间：targetPeriodEndTs + PERIOD_SECONDS（周期结束） */
    expiresAt: Date;
    /** 建单时订单簿最优卖价 */
    bestAskAtEntry?: number;
    /** 建单时同价及更优买单竞争深度（USD） */
    queueCompetitionUsdAtEntry?: number;
    /** 建单时扣减竞争后的可成交金额（haircut 前） */
    effectiveFillableUsdAtEntry?: number;
    /** 建单时基于订单簿推算的实际加权成交均价 */
    avgActualFillPriceAtEntry?: number;
    /** 是否正在检查中（防止并发） */
    checking: boolean;
}

type PersistedPendingSimOrder = Omit<PendingSimOrder, 'createdAt' | 'expiresAt' | 'checking'> & {
    createdAt: string;
    expiresAt: string;
};

type PendingOrderLedgerEventType =
    | 'created'
    | 'partial_fill'
    | 'filled'
    | 'expired_refunded'
    | 'cancelled'
    | 'official_canceled_zero_fill'
    | 'official_canceled_partial_fill';

interface PendingOrderLedgerEvent {
    timestamp: string;
    event: PendingOrderLedgerEventType;
    orderId: string;
    symbol: string;
    direction: 'UP' | 'DOWN';
    amountUsd: number;
    remainingFrozenAmount: number;
    limitPrice: number;
    tokenId: string;
    tokenOutcome: string;
    confidence: number;
    marketTitle: string;
    conditionId: string;
    marketSlug: string;
    targetPeriodEndTs: number;
    createdAt: string;
    expiresAt: string;
    bestAsk?: number;
    avgFillPrice?: number;
    queueCompetitionUsdAtEntry?: number;
    effectiveFillableUsdAtEntry?: number;
    terminalTimestampSource?: string;
    officialAcceptanceStatus?: string;
    officialAcceptanceSource?: string;
    officialAcceptanceAt?: string;
}

type OfficialOrderLifecycleState = {
    status: string;
    matchedSize: number;
    raw?: any;
    reason?: string;
};

/** 模拟挂单监控间隔（毫秒） */
const PENDING_SIM_MONITOR_INTERVAL_MS = 5000;
const SIM_QUEUE_HAIRCUT = 0.95;

// ============================================================
// 辅助函数
// ============================================================

/**
 * 获取交易模式的中文描述
 */
const getTradingModeText = (mode: TradingMode): string => {
    switch (mode) {
        case TradingMode.BACKTEST:
            return '回测模式';
        case TradingMode.SIMULATION:
            return '模拟交易';
        case TradingMode.LIVE:
            return '真实交易';
        default:
            return mode;
    }
};

/**
 * 解析错误原因
 */
const parseErrorReason = (error: unknown): string => {
    const msg = extractErrorText(error);
    const lower = msg.toLowerCase();
    
    if (lower.includes('401') || lower.includes('403') || lower.includes('unauthorized') || lower.includes('invalid api key') || lower.includes('invalid signature')) {
        return '认证失败';
    }
    if (lower.includes('502') || lower.includes('bad gateway')) {
        return '网关 502';
    }
    if (lower.includes('allowance') || msg.includes('授权')) {
        return 'pUSD 授权不足';
    }
    if (msg.includes('timeout') || msg.includes('ETIMEDOUT')) {
        return '网络请求超时';
    }
    if (msg.includes('ECONNREFUSED')) {
        return '连接被拒绝，服务可能未启动';
    }
    if (lower.includes('insufficient') || lower.includes('balance')) {
        return '余额不足';
    }
    if (lower.includes('rate limit')) {
        return 'API 请求频率限制';
    }
    
    return msg;
};

type LiveOrderPreflightStatus = 'ok' | 'allowance_zero' | 'balance_insufficient' | 'gateway_502' | 'auth_failure' | 'order_submit_failure';

interface LiveOrderPreflightResult {
    ok: boolean;
    status: LiveOrderPreflightStatus;
    reason: string;
    balanceUsd?: number;
    allowanceRaw?: string | null;
}

function extractErrorText(error: unknown): string {
    if (typeof error === 'string') return error;
    const anyErr = error as any;
    const parts: string[] = [];
    if (anyErr?.message) parts.push(String(anyErr.message));
    if (anyErr?.status) parts.push(`status=${String(anyErr.status)}`);
    if (anyErr?.code) parts.push(`code=${String(anyErr.code)}`);
    if (anyErr?.response?.status) parts.push(`response_status=${String(anyErr.response.status)}`);
    if (anyErr?.response?.data !== undefined) {
        try {
            parts.push(`response_data=${JSON.stringify(anyErr.response.data)}`);
        } catch {
            parts.push(`response_data=${String(anyErr.response.data)}`);
        }
    }
    if (anyErr?.data !== undefined) {
        try {
            parts.push(`data=${JSON.stringify(anyErr.data)}`);
        } catch {
            parts.push(`data=${String(anyErr.data)}`);
        }
    }
    if (parts.length > 0) return parts.join(' | ');
    try {
        return JSON.stringify(error);
    } catch {
        return String(error);
    }
}

function classifyLiveSubmitIssue(raw: unknown): { status: LiveOrderPreflightStatus; reason: string } {
    const text = extractErrorText(raw);
    const lower = text.toLowerCase();
    if (lower.includes('invalid expiration value') || text.includes('过期时间无效')) {
        return { status: 'order_submit_failure', reason: 'GTD 过期时间无效' };
    }
    if (lower.includes('401') || lower.includes('403') || lower.includes('unauthorized') || lower.includes('invalid api key') || lower.includes('invalid signature') || text.includes('认证失败')) {
        return { status: 'auth_failure', reason: '认证失败' };
    }
    if (lower.includes('502') || lower.includes('bad gateway') || lower.includes('/balance-allowance') || lower.includes('/auth/derive-api-key') || text.includes('网关 502')) {
        return { status: 'gateway_502', reason: '网关 502' };
    }
    if (lower.includes('allowance') || text.includes('授权')) {
        return { status: 'allowance_zero', reason: 'pUSD 授权不足' };
    }
    if (lower.includes('insufficient') || lower.includes('balance') || text.includes('余额不足')) {
        return { status: 'balance_insufficient', reason: '余额不足' };
    }
    return { status: 'order_submit_failure', reason: parseErrorReason(raw) };
}

/**
 * Polymarket 15分钟加密货币市场 taker 手续费计算
 *
 * 公式: fee = C × p × feeRate × (p × (1 - p))^exponent
 * 15-min crypto: feeRate = 0.25, exponent = 2
 * 有效费率 = feeRate × (p × (1 - p))^exponent
 *
 * 价格 $0.45 → 1.53%, $0.50 → 1.56%, $0.53 → 1.55%
 * Maker 订单不收费，但我们的 GTC 限价单在 best_ask <= limit_price 时为 taker 成交
 *
 * @param p  成交价（0~1 之间的代币价格）
 * @returns  有效费率（例如 0.0156 表示 1.56%）
 */
function polymarketTakerFeeRate(p: number): number {
    if (p <= 0 || p >= 1) return 0;
    return 0.25 * Math.pow(p * (1 - p), 2);
}

/**
 * 费率说明：
 * - 实际 taker 手续费由 polymarketTakerFeeRate(p) 在结算时计算（~1.56%@p=0.50）
 * - Edge/Kelly 不再扣除费率，保持纯数学公式：edge = p*odds - q, kelly = (p*odds - q) / odds
 * - 原因：(1) 费率可能变动 (2) Optuna 超参已用错误费率优化，匹配无意义 (3) 费率只影响结算PnL，不影响方向判断
 */

interface QueueAwareBookSnapshot {
    bestAskPrice: number;
    asks: BookLevel[];
    bids: BookLevel[];
    hadBidsData: boolean;
}

interface QueueAwareBuyFillSnapshot {
    bestAskPrice: number;
    avgFillPrice: number;
    rawFillableUsd: number;
    queueCompetitionUsd: number;
    effectiveFillableUsd: number;
    haircutAdjustedFillableUsd: number;
    hadBidsData: boolean;
}

function buildQueueAwareBookSnapshot(orderBook: any): QueueAwareBookSnapshot {
    const asks = normalizeBookSide(orderBook?.asks);
    const bids = normalizeBookSide(orderBook?.bids);
    let bestAskPrice = 999;
    for (const ask of asks) {
        if (ask.price < bestAskPrice) bestAskPrice = ask.price;
    }
    return {
        bestAskPrice,
        asks,
        bids,
        hadBidsData: bids.length > 0,
    };
}

function simulateQueueAwareBuyFillWithHaircut(
    snapshot: QueueAwareBookSnapshot,
    limitPrice: number,
    requestUsd: number,
): QueueAwareBuyFillSnapshot {
    const fillResult = simulateQueueAwareBuyFill({
        asks: snapshot.asks,
        bids: snapshot.bids,
        limitPrice,
        requestUsd,
    });
    const haircutAdjustedFillableUsd = fillResult.filledUsd > 0
        ? roundUsd(fillResult.filledUsd * SIM_QUEUE_HAIRCUT)
        : 0;
    return {
        bestAskPrice: snapshot.bestAskPrice,
        avgFillPrice: fillResult.avgFillPrice,
        rawFillableUsd: roundUsd(fillResult.rawFillableUsd),
        queueCompetitionUsd: roundUsd(fillResult.queueCompetitionUsd),
        effectiveFillableUsd: roundUsd(fillResult.effectiveFillableUsd),
        haircutAdjustedFillableUsd,
        hadBidsData: snapshot.hadBidsData,
    };
}

function consumeAskBookLiquidity(asks: BookLevel[], limitPrice: number, consumeUsd: number): void {
    let remaining = Math.max(0, consumeUsd);
    const sorted = asks
        .map((ask, idx) => ({ ask, idx }))
        .filter(({ ask }) => ask.price <= limitPrice && ask.sizeUsd > 0)
        .sort((a, b) => a.ask.price - b.ask.price);
    for (const { ask } of sorted) {
        if (remaining <= 0) break;
        const consume = Math.min(ask.sizeUsd, remaining);
        ask.sizeUsd = Math.max(0, ask.sizeUsd - consume);
        ask.sizeShares = ask.price > 0 ? ask.sizeUsd / ask.price : 0;
        remaining -= consume;
    }
}

// ============================================================
// PredictionExecutor 类
// ============================================================

export class PredictionExecutor {
    /** 注入的环境配置（多 trader 各自独立；单进程模式默认读 PREDICTION_ENV） */
    public readonly env: PredictionEnvConfig;
    public readonly modeScope: TradeLogModeScope;
    /** multi 模式下从外部传入的 logsDir（覆盖 process.env.LOGS_DIR） */
    public readonly logsDir: string;
    /** logsDir 的绝对路径（带 cwd 前缀），统一用于文件操作 */
    public readonly logsDirFull: string;
    private clobClient: ClobClient | null = null;
    /** 下单提交器：单钱包现在，多钱包并发后续可替换为 MultiWalletOrderPoster */
    private orderPoster: IOrderPoster | null = null;
    /** CLOB V2 要求持续 heartbeat；否则开放订单会被官方自动取消 */
    private clobHeartbeatId = '';
    private clobHeartbeatTimer: ReturnType<typeof setInterval> | null = null;
    private clobHeartbeatInFlight: Promise<void> | null = null;
    private clobLastHeartbeatAt = 0;
    private clobLastHeartbeatError: string | null = null;
    private static clobHeartbeatQueue: Promise<void> = Promise.resolve();
    private static clobLastGlobalHeartbeatPostAt = 0;
    private static readonly CLOB_HEARTBEAT_GLOBAL_GAP_MS = 2500;
    private logger: TradeLogger;
    private state: ExecutorState;
    private priceMonitorQueue: PriceMonitorEntry[] = [];  // 价格监控队列（高价）
    private lowPriceMonitorQueue: PriceMonitorEntry[] = [];  // 价格监控队列（低价，等待回升）
    private liquidityMonitorQueue: LiquidityMonitorEntry[] = [];  // 流动性监控队列（用于部分成交后继续买入）
    /** 两阶段限价单: Phase 1 未成交/部分成交的订单跟踪（key = targetPeriodEndTs + symbol） */
    private phaseOrderTracker: Map<string, PhaseOrderEntry> = new Map();
    private get MAX_PRICE_THRESHOLD(): number { return this.env.MAX_TOKEN_PRICE; }
    /** 周期时长（秒）：15m=900, 5m=300，根据 TIMEFRAME 自动适配（static 版保留向后兼容） */
    static get PERIOD_SECONDS(): number {
        const tf = PREDICTION_ENV.TIMEFRAME;
        if (tf === '5m') return 300;
        if (tf === '1h') return 3600;
        if (tf === '4h') return 14400;
        return 900;
    }
    /** 实例版 PERIOD_SECONDS（多 trader 模式使用各自 env.TIMEFRAME） */
    get periodSeconds(): number {
        const tf = this.env.TIMEFRAME;
        if (tf === '5m') return 300;
        if (tf === '1h') return 3600;
        if (tf === '4h') return 14400;
        return 900;
    }
    /** 使用实例周期格式化北京时间区间 */
    private _fmtPeriod(periodStartTs: number): string {
        return formatPeriodBeijing(periodStartTs, this.periodSeconds);
    }
    private get MONITOR_WINDOW_MINUTES(): number {
        return this.env.TIMEFRAME === '5m' ? 2 : 5;
    }
    /** 最后5分钟市价成交价格带：前10分钟未达 [MIN, MAX] 时，在周期结束前5分钟内若价格在此区间则市价成交，避免最终未买；上限 0.65 防止高价 adverse selection */
    private static readonly LAST5MIN_MIN_PRICE = 0.3;
    private static readonly LAST5MIN_MAX_PRICE = 0.65;
    // 防重复下单锁：追踪正在处理的 conditionId，防止并发时同一市场被重复下单
    private processingConditionIds: Set<string> = new Set();
    /** 模拟限价单挂单池：资金已冻结，等待订单簿价格满足条件 */
    private pendingSimOrders: PendingSimOrder[] = [];
    /** 模拟挂单监控定时器 */
    private pendingSimMonitorTimer: ReturnType<typeof setInterval> | null = null;
    // ─── Per-Coin Capital Isolation ──────────────────────────────
    private readonly perCoinEnabled: boolean;
    private capitalByCoin: Map<string, number>;
    private peakCapitalByCoin: Map<string, number>;

    // ─── Per-Coin Cooldown（按币种独立冷却）──────────────────────
    private cooldownByCoin: Map<string, number>;
    // 新版冷却：按 symbol 或 symbol+direction 作用域（默认 symbol+direction）
    private cooldownByCoinDir: Map<string, number>;

    // ─── GTD 真实订单追踪（由 UserChannelWS 回调驱动）──────────
    private pendingLiveOrders: Map<string, LiveOrderTracker> = new Map();
    /** 交易所最小可执行 shares（从真实拒单动态学习，缺省按当前观测值 5） */
    private liveExchangeMinShares = DEFAULT_EXCHANGE_MIN_SHARES;
    /** token 级执行约束缓存：官方 /book 优先，其次真实拒单学习，再次默认回退。 */
    private executionConstraintByToken: Map<string, ExchangeExecutionConstraint> = new Map();
    /** 已输出过「止损已弃用」提示，避免周期日志刷屏 */
    private stopLossDeprecatedLogged = false;
    /** 运行时仓位缩放（由编排器按数据质量动态注入）。1.0=正常，0=停单。 */
    private runtimeBetScale = 1.0;
    /** 运行时按 symbol 仓位缩放（极端行情等附加层使用）。 */
    private runtimeSymbolScaleBySymbol: Map<string, number> = new Map();
    /** 运行时 DOWN 风控缩放（按 symbol 注入，仅 DOWN 生效）。 */
    private runtimeDownRiskScaleBySymbol: Map<string, number> = new Map();
    /** 运行时 DOWN 风控封禁（按 symbol 注入，仅 DOWN 生效）。 */
    private runtimeDownRiskBlockedBySymbol: Map<string, boolean> = new Map();
    /** 运行时 UP 风控缩放（按 symbol 注入，仅 UP 生效）。 */
    private runtimeUpRiskScaleBySymbol: Map<string, number> = new Map();
    /** 运行时 UP 风控封禁（按 symbol 注入，仅 UP 生效）。 */
    private runtimeUpRiskBlockedBySymbol: Map<string, boolean> = new Map();
    /** 运行时期望值启停缩放（按 symbol+direction 注入，degraded 时降仓，不减单）。 */
    private runtimeExpectancyScaleBySymbolDir: Map<string, number> = new Map();
    /** 运行时期望值启停封禁（按 symbol+direction 注入，仅极端 blocked 生效）。 */
    private runtimeExpectancyBlockedBySymbolDir: Map<string, boolean> = new Map();
    /** 运行时高置信同方向连错轻缩仓（按 symbol+direction 注入）。 */
    private runtimeDirectionLossScaleBySymbolDir: Map<string, number> = new Map();
    /** 运行时回撤加速度观察态轻缩仓（按 symbol 注入）。 */
    private runtimeDrawdownAccelerationScaleBySymbol: Map<string, number> = new Map();
    /** 方向保守仓位滞后状态（按 symbol 持久化）；true=当前仍在保守区。 */
    private runtimeDirectionalConservativeBySymbol: Map<string, boolean> = new Map();
    /** 已经成功 claim 的 conditionId（从 redemption_log.jsonl 启动加载 + 运行时更新） */
    private claimedSuccessConditionIds: Set<string> = new Set();
    /** 正在提交中的 claim conditionId，避免同一轮或并发重复请求。 */
    private claimInFlightConditionIds: Set<string> = new Set();
    /** signal_outcomes 去重缓存，避免同 conditionId+symbol+mode 重复写入 */
    private signalOutcomeKeys: Set<string> = new Set();
    private signalOutcomeKeysLoaded = false;
    /** 冷却拦截机会去重缓存，避免同一机会重复记账。 */
    private blockedOpportunityLogKeys: Set<string> = new Set();
    /** LIVE 下单后延迟对账互斥，避免多个回填定时器同时重写账本。 */
    private liveFollowUpReconciliationRunning = false;
    /** LIVE 下单前只做快路径；深度公共 activity 对账转后台，避免卡住自动下单。 */
    private livePreOrderDeepReconcileLastStartedAt = 0;
    /** LIVE 下单前的 pending 订单对账也不能同步阻塞当前 15m 下单窗口。 */
    private livePreOrderPendingRefreshRunning = false;

    /** Edge 跳过统计日志写入 */
    private _logEdgeSkip(data: {
        symbol: string; confidence: number; bestAsk: number;
        edge: number; minEdge: number; limitPrice?: number;
        conditionId?: string; reason: string;
    }): void {
        try {
            const logPath = path.join(this.logsDirFull, 'edge_skip_log.jsonl');
            const entry = { timestamp: new Date().toISOString(), ...data };
            fs.appendFileSync(logPath, JSON.stringify(entry) + '\n');
        } catch { /* 日志写入失败不影响交易 */ }
    }

    private appendSlot01GentleFilterBlockedOpportunity(args: {
        prediction: PredictionResult;
        marketResult: MarketSearchResult;
        symbol: string;
        direction: 'UP' | 'DOWN';
        confidence: number;
        roundedConfidence: number;
        confidenceMin: number;
        confidenceMax: number;
        currentCapital: number;
        targetPeriodEndTs?: number;
    }): void {
        try {
            const logPath = path.join(this.logsDirFull, `gentle_filter_blocked_opportunities.${this.modeScope}.jsonl`);
            const market = args.marketResult.market as any;
            const payload = {
                timestamp: new Date().toISOString(),
                filterName: 'slot01_gentle_confidence_interval',
                filterReason: '温和信心区间过滤',
                executionDecision: 'slot01_gentle_filter_blocked_before_order',
                symbol: args.symbol,
                direction: args.direction,
                confidence: args.confidence,
                roundedConfidence: args.roundedConfidence,
                confidenceMin: args.confidenceMin,
                confidenceMax: args.confidenceMax,
                currentCapital: roundUsd(args.currentCapital),
                targetPeriodEndTs: args.targetPeriodEndTs,
                marketTitle: market?.title,
                marketSlug: market?.slug,
                conditionId: market?.conditionId,
                explanationChinese: '该机会命中 slot1 ETH 温和信心区间过滤门，已在创建订单前跳过；模型、阈值、冷却、金额、时点均未改变。',
            };
            fs.appendFileSync(logPath, `${JSON.stringify(payload)}\n`, 'utf-8');
        } catch {
            /* 过滤证据日志写入失败不影响主流程 */
        }
    }

    private getExecutionV2EventsPath(): string {
        return path.join(this.logsDirFull, `execution_v2_events.${this.modeScope}.jsonl`);
    }

    private roundTokenPriceTick(price: number): number {
        return Number.isFinite(price) ? Math.round(price * 100) / 100 : 0;
    }

    private buildExecutionV2Profile(): ExecutionV2Profile {
        const legacyChaseWaitSec = Math.max(0, Number(this.env.CHASE_WAIT_SEC) || 0);
        return {
            version: String(this.env.EXECUTION_PROFILE_VERSION || 'v1'),
            shadowOnly: Boolean(this.env.EXECUTION_V2_SHADOW_ONLY),
            chaseEnabled: Boolean(this.env.CHASE_ENABLED),
            chaseRounds: Math.max(0, Number(this.env.CHASE_ROUNDS) || 0),
            chaseWaitSec: legacyChaseWaitSec,
            firstChaseWaitSec: Math.max(0, Number(this.env.FIRST_CHASE_WAIT_SEC) || legacyChaseWaitSec || 35),
            nextChaseGapSec: Math.max(0, Number(this.env.NEXT_CHASE_GAP_SEC) || 20),
            chaseTickLift: Math.max(0, Number(this.env.CHASE_TICK_LIFT) || 0),
            chaseAbsoluteMaxPrice: Math.max(0, Number(this.env.CHASE_ABSOLUTE_MAX_PRICE) || this.MAX_PRICE_THRESHOLD),
            chaseMinConfidenceBand: Math.max(0, Math.min(1, Number(this.env.CHASE_MIN_CONFIDENCE_BAND) || 0)),
            chaseMinTimeToExpirySec: Math.max(0, Number(this.env.CHASE_MIN_TIME_TO_EXPIRY_SEC) || 0),
            directionContinuationTicks: Math.max(0, Number(this.env.DIRECTION_CONTINUATION_TICKS) || 0),
            directionContinuationWindowSec: Math.max(0, Number(this.env.DIRECTION_CONTINUATION_WINDOW_SEC) || 0),
            finalAggressiveWindowSec: Math.max(0, Number(this.env.FINAL_AGGRESSIVE_WINDOW_SEC) || 0),
            finalAggressiveTickLift: Math.max(0, Number(this.env.FINAL_AGGRESSIVE_TICK_LIFT) || 0),
            finalAggressiveMaxPrice: Math.max(0, Number(this.env.FINAL_AGGRESSIVE_MAX_PRICE) || PredictionExecutor.LAST5MIN_MAX_PRICE),
            cancelOnAdverseMoveTicks: Math.max(0, Number(this.env.CANCEL_ON_ADVERSE_MOVE_TICKS) || 0),
            cancelOnSignalInvalidation: Boolean(this.env.CANCEL_ON_SIGNAL_INVALIDATION),
            cancelOnQueueStall: Boolean(this.env.CANCEL_ON_QUEUE_STALL),
            queueStallWindowSec: Math.max(0, Number(this.env.QUEUE_STALL_WINDOW_SEC) || 0),
            staleCancelMinAgeSec: Math.max(0, Number(this.env.STALE_CANCEL_MIN_AGE_SEC) || 0),
            staleCancelMinTimeToExpirySec: Math.max(0, Number(this.env.STALE_CANCEL_MIN_TIME_TO_EXPIRY_SEC) || 0),
            lateFillCancelMinutes: Math.max(0, Number((this.env as any).LATE_FILL_CANCEL_MINUTES) || 0),
        };
    }

    private executionV2Enabled(profile?: ExecutionV2Profile | null): boolean {
        const p = profile || this.buildExecutionV2Profile();
        return p.version === 'v2' || p.chaseEnabled;
    }

    private appendExecutionV2Event(record: ExecutionV2EventRecord): void {
        try {
            fs.appendFileSync(this.getExecutionV2EventsPath(), `${JSON.stringify(record)}\n`, 'utf-8');
        } catch {
            /* 事件日志写入失败不影响交易 */
        }
    }

    private maybeUpdateExecutionProfileOnLog(
        logEntryId: string | undefined,
        profile: ExecutionV2Profile | undefined,
        updates?: Partial<TradeLogEntry>,
    ): void {
        if (!logEntryId || !profile) return;
        this.logger.updateEntry(logEntryId, {
            executionProfileVersion: profile.version,
            chaseRound: Number(updates?.chaseRound ?? 0) || undefined,
            chaseTickLift: Number(updates?.chaseTickLift ?? profile.chaseTickLift) || undefined,
            chaseWaitSec: Number(updates?.chaseWaitSec ?? profile.chaseWaitSec) || undefined,
            firstChaseWaitSec: Number(updates?.firstChaseWaitSec ?? profile.firstChaseWaitSec) || undefined,
            nextChaseGapSec: Number(updates?.nextChaseGapSec ?? profile.nextChaseGapSec) || undefined,
            chaseAbsoluteMaxPrice: Number(updates?.chaseAbsoluteMaxPrice ?? profile.chaseAbsoluteMaxPrice) || undefined,
            finalAggressiveWindowSec: Number(updates?.finalAggressiveWindowSec ?? profile.finalAggressiveWindowSec) || undefined,
            finalAggressiveMaxPrice: Number(updates?.finalAggressiveMaxPrice ?? profile.finalAggressiveMaxPrice) || undefined,
            cancelOnAdverseMoveTicks: Number(updates?.cancelOnAdverseMoveTicks ?? profile.cancelOnAdverseMoveTicks) || undefined,
            cancelOnSignalInvalidation: updates?.cancelOnSignalInvalidation ?? profile.cancelOnSignalInvalidation,
            cancelOnQueueStall: updates?.cancelOnQueueStall ?? profile.cancelOnQueueStall,
            queueStallWindowSec: Number(updates?.queueStallWindowSec ?? profile.queueStallWindowSec) || undefined,
            staleCancelMinAgeSec: Number(updates?.staleCancelMinAgeSec ?? profile.staleCancelMinAgeSec) || undefined,
            staleCancelMinTimeToExpirySec: Number(updates?.staleCancelMinTimeToExpirySec ?? profile.staleCancelMinTimeToExpirySec) || undefined,
            lateFillCancelMinutes: Number((updates as any)?.lateFillCancelMinutes ?? profile.lateFillCancelMinutes) || undefined,
            executionState: updates?.executionState,
            chaseTriggered: updates?.chaseTriggered,
            finalAggressiveTriggered: updates?.finalAggressiveTriggered,
        });
    }

    private appendExecutionV2SettlementEventFromLog(
        logEntryId: string | undefined,
        eventType: 'chase_reprice_filled' | 'fully_filled',
        reason: string,
    ): void {
        if (!logEntryId) return;
        const row = this.logger.getEntry(logEntryId);
        if (!row) return;
        this.appendExecutionV2Event({
            timestamp: new Date().toISOString(),
            symbol: row.symbol,
            direction: row.direction,
            marketSlug: row.marketSlug,
            conditionId: row.conditionId,
            mode: this.env.TRADING_MODE,
            eventType,
            shadowOnly: false,
            requestedAmountUsd: row.requestedAmountUsd ?? row.amount,
            remainingAmountUsd: row.pendingAmountUsd ?? 0,
            initialPrice: row.limitPriceConfigured ?? row.tokenPrice,
            currentPrice: row.avgActualFillPrice ?? row.tokenPrice,
            currentMaxPrice: row.finalAggressiveMaxPrice ?? row.chaseAbsoluteMaxPrice ?? row.limitPriceConfigured ?? row.tokenPrice,
            chaseRound: row.chaseRound,
            chaseTickLift: row.chaseTickLift,
            finalAggressiveWindowSec: row.finalAggressiveWindowSec,
            finalAggressiveMaxPrice: row.finalAggressiveMaxPrice,
            reason,
        });
    }

    private isLatestSignalStillValid(prediction: PredictionResult, targetPeriodEndTs: number): boolean | null {
        try {
            const raw = readLocalPredictionsRaw();
            if (!raw || Number(raw.target_period_end_ts || 0) !== Number(targetPeriodEndTs || 0)) {
                return null;
            }
            const parsed = parseLocalPredictions(raw.data, this.env.TIMEFRAME, [prediction.symbol]);
            const latest = parsed.find((row) => row.symbol === prediction.symbol);
            if (!latest) return null;
            if (!latest.direction || latest.direction !== prediction.direction) return false;
            if (latest.shouldTrade === false) return false;
            return true;
        } catch {
            return null;
        }
    }

    private evaluateExecutionV2Adjustment(params: {
        profile?: ExecutionV2Profile;
        prediction: PredictionResult;
        targetPeriodEndTs: number;
        currentPrice: number;
        currentMaxPrice: number;
        highestObservedPrice?: number;
        initialLimitPrice?: number;
        chaseRound?: number;
        lastChaseAt?: number;
        monitorStartMs?: number;
        timeToExpirySec: number;
        nowMs: number;
    }): {
        shouldPost: boolean;
        shouldCancel: boolean;
        eventType?: ExecutionV2EventRecord['eventType'];
        nextState?: ExecutionV2State;
        nextMaxPrice?: number;
        nextHighestObservedPrice?: number;
        nextChaseRound?: number;
        reason?: string;
    } {
        const profile = params.profile;
        const currentMaxPrice = this.roundTokenPriceTick(params.currentMaxPrice);
        const highestObservedPrice = Math.max(
            Number(params.highestObservedPrice ?? params.initialLimitPrice ?? params.currentPrice) || params.currentPrice,
            params.currentPrice,
        );
        const latestSignalStillValid = profile?.cancelOnSignalInvalidation
            ? this.isLatestSignalStillValid(params.prediction, params.targetPeriodEndTs)
            : null;
        if (!profile || !this.executionV2Enabled(profile)) {
            return {
                shouldPost: params.currentPrice >= this.env.MIN_TOKEN_PRICE && params.currentPrice <= currentMaxPrice,
                shouldCancel: false,
                nextHighestObservedPrice: highestObservedPrice,
            };
        }

        if (profile.cancelOnSignalInvalidation && latestSignalStillValid === false) {
            return {
                shouldPost: false,
                shouldCancel: true,
                eventType: 'cancelled_on_direction_invalidation',
                nextState: 'cancelled_on_invalidation',
                nextHighestObservedPrice: highestObservedPrice,
                reason: 'latest_local_signal_invalidated',
            };
        }

        const tickSize = 0.01;
        const adverseTicks = profile.cancelOnAdverseMoveTicks;
        const adverseMove = adverseTicks > 0 && (highestObservedPrice - params.currentPrice) >= (adverseTicks * tickSize);
        if (adverseMove) {
            return {
                shouldPost: false,
                shouldCancel: true,
                eventType: 'cancelled_on_adverse_move',
                nextState: 'cancelled_on_invalidation',
                nextHighestObservedPrice: highestObservedPrice,
                reason: `price_retraced_${adverseTicks}t_after_favorable_move`,
            };
        }

        if (params.currentPrice >= this.env.MIN_TOKEN_PRICE && params.currentPrice <= currentMaxPrice) {
            return {
                shouldPost: true,
                shouldCancel: false,
                nextHighestObservedPrice: highestObservedPrice,
            };
        }

        if (params.timeToExpirySec <= 0) {
            return {
                shouldPost: false,
                shouldCancel: true,
                eventType: 'expired_unfilled',
                nextState: 'expired_unfilled',
                nextHighestObservedPrice: highestObservedPrice,
                reason: 'expired_before_fill',
            };
        }

        const confidence = Number(params.prediction.confidence || 0);
        const chaseRound = Math.max(0, Number(params.chaseRound) || 0);
        const lastChaseAt = Math.max(0, Number(params.lastChaseAt) || 0);
        const monitorStartMs = Math.max(0, Number(params.monitorStartMs) || 0);
        const firstWaitSec = Math.max(0, Number(profile.firstChaseWaitSec || profile.chaseWaitSec || 0));
        const nextGapSec = Math.max(0, Number(profile.nextChaseGapSec || profile.chaseWaitSec || 0));
        const waitAnchorMs = lastChaseAt > 0 ? lastChaseAt : monitorStartMs;
        const requiredWaitSec = chaseRound > 0 ? nextGapSec : firstWaitSec;
        const waitOk = waitAnchorMs <= 0 || ((params.nowMs - waitAnchorMs) / 1000.0) >= requiredWaitSec;
        const totalAgeSec = monitorStartMs > 0 ? ((params.nowMs - monitorStartMs) / 1000.0) : 0;
        const continuationOk = params.currentPrice >= ((params.highestObservedPrice ?? params.initialLimitPrice ?? params.currentMaxPrice) + (profile.directionContinuationTicks * tickSize) - 1e-9);
        const queueStallTriggered = profile.cancelOnQueueStall
            && profile.queueStallWindowSec > 0
            && totalAgeSec >= Math.max(profile.staleCancelMinAgeSec, profile.queueStallWindowSec)
            && params.timeToExpirySec >= profile.staleCancelMinTimeToExpirySec
            && params.currentPrice > currentMaxPrice + 1e-9
            && highestObservedPrice <= currentMaxPrice + tickSize + 1e-9;
        if (queueStallTriggered) {
            return {
                shouldPost: false,
                shouldCancel: true,
                eventType: 'cancelled_on_stale_queue',
                nextState: 'cancelled_on_stale_queue',
                nextHighestObservedPrice: highestObservedPrice,
                reason: 'queue_stalled_without_favorable_progress',
            };
        }
        const canEnterFinal = profile.finalAggressiveWindowSec > 0
            && params.timeToExpirySec <= profile.finalAggressiveWindowSec
            && params.currentPrice <= profile.finalAggressiveMaxPrice + 1e-9;
        if (canEnterFinal) {
            const lifted = this.roundTokenPriceTick(Math.min(
                profile.finalAggressiveMaxPrice,
                Math.max(
                    params.currentPrice,
                    currentMaxPrice + (profile.finalAggressiveTickLift * tickSize),
                ),
            ));
            return {
                shouldPost: params.currentPrice <= lifted + 1e-9,
                shouldCancel: false,
                eventType: 'final_aggressive_posted',
                nextState: 'final_aggressive_window',
                nextMaxPrice: lifted,
                nextHighestObservedPrice: highestObservedPrice,
                nextChaseRound: chaseRound,
                reason: 'entered_final_aggressive_window',
            };
        }
        const preExpiryTimeoutTriggered = profile.staleCancelMinTimeToExpirySec > 0
            && params.timeToExpirySec <= profile.staleCancelMinTimeToExpirySec
            && params.currentPrice > profile.finalAggressiveMaxPrice + 1e-9;
        if (preExpiryTimeoutTriggered) {
            return {
                shouldPost: false,
                shouldCancel: true,
                eventType: 'cancelled_pre_expiry_timeout',
                nextState: 'cancelled_pre_expiry_timeout',
                nextHighestObservedPrice: highestObservedPrice,
                reason: 'pre_expiry_timeout_without_aggressive_window_entry',
            };
        }

        const canChase = profile.chaseEnabled
            && chaseRound < profile.chaseRounds
            && waitOk
            && confidence >= profile.chaseMinConfidenceBand
            && params.timeToExpirySec >= profile.chaseMinTimeToExpirySec
            && continuationOk
            && params.currentPrice <= profile.chaseAbsoluteMaxPrice + 1e-9;
        if (!canChase) {
            return {
                shouldPost: false,
                shouldCancel: false,
                nextHighestObservedPrice: highestObservedPrice,
            };
        }

        const lifted = this.roundTokenPriceTick(Math.min(
            profile.chaseAbsoluteMaxPrice,
            Math.max(
                params.currentPrice,
                currentMaxPrice + (profile.chaseTickLift * tickSize),
            ),
        ));
        return {
            shouldPost: params.currentPrice <= lifted + 1e-9,
            shouldCancel: false,
            eventType: 'chase_reprice_posted',
            nextState: 'chase_round_n',
            nextMaxPrice: lifted,
            nextHighestObservedPrice: highestObservedPrice,
            nextChaseRound: chaseRound + 1,
            reason: 'favorable_move_chase_reprice',
        };
    }

    private applyExecutionV2MonitorDecision(
        entry: {
            prediction: PredictionResult;
            marketResult: MarketSearchResult;
            monitorStartTime: Date;
            targetPeriodEndTs: number;
            executionProfile?: ExecutionV2Profile;
            executionState?: ExecutionV2State;
            highestObservedPrice?: number;
            initialLimitPrice?: number;
            currentMaxPrice?: number;
            maxPrice?: number;
            requestedAmountUsd?: number;
            remainingAmountUsd?: number;
            remainingAmount?: number;
            chaseRound?: number;
            lastChaseAt?: number;
            logEntryId?: string;
        },
        currentPrice: number,
        now: Date,
    ): {
        shouldExecute: boolean;
        shouldCancel: boolean;
        effectiveMaxPrice: number;
        reason?: string;
    } {
        const profile = entry.executionProfile;
        const baseMaxPrice = this.roundTokenPriceTick(
            Number(entry.currentMaxPrice ?? entry.maxPrice ?? this.MAX_PRICE_THRESHOLD) || this.MAX_PRICE_THRESHOLD,
        );
        const nowMs = now.getTime();
        const periodEndMs = (entry.targetPeriodEndTs + this.periodSeconds) * 1000;
        const timeToExpirySec = Math.max(0, Math.floor((periodEndMs - nowMs) / 1000));
        const decision = this.evaluateExecutionV2Adjustment({
            profile,
            prediction: entry.prediction,
            targetPeriodEndTs: entry.targetPeriodEndTs,
            currentPrice,
            currentMaxPrice: baseMaxPrice,
            highestObservedPrice: entry.highestObservedPrice,
            initialLimitPrice: entry.initialLimitPrice,
            chaseRound: entry.chaseRound,
            lastChaseAt: entry.lastChaseAt,
            monitorStartMs: entry.monitorStartTime.getTime(),
            timeToExpirySec,
            nowMs,
        });

        const shadowOnly = Boolean(profile?.shadowOnly);
        if (decision.nextHighestObservedPrice != null) {
            entry.highestObservedPrice = decision.nextHighestObservedPrice;
        }
        if (decision.nextChaseRound != null && decision.nextChaseRound !== entry.chaseRound) {
            entry.chaseRound = decision.nextChaseRound;
            entry.lastChaseAt = nowMs;
        }
        if (decision.nextState && (!shadowOnly || decision.nextState === 'chase_round_n' || decision.nextState === 'final_aggressive_window')) {
            entry.executionState = decision.nextState;
        }

        let effectiveMaxPrice = baseMaxPrice;
        if (decision.nextMaxPrice != null && !shadowOnly) {
            effectiveMaxPrice = this.roundTokenPriceTick(decision.nextMaxPrice);
            entry.currentMaxPrice = effectiveMaxPrice;
            if (Object.prototype.hasOwnProperty.call(entry, 'maxPrice')) {
                entry.maxPrice = effectiveMaxPrice;
            }
        }

        if (decision.eventType) {
            this.appendExecutionV2Event({
                timestamp: now.toISOString(),
                symbol: entry.prediction.symbol.replace('/USDT', ''),
                direction: entry.prediction.direction || 'UP',
                marketSlug: entry.marketResult.market?.slug,
                conditionId: entry.marketResult.market?.conditionId,
                mode: this.env.TRADING_MODE,
                eventType: decision.eventType,
                shadowOnly,
                requestedAmountUsd: entry.requestedAmountUsd ?? entry.remainingAmountUsd ?? entry.remainingAmount,
                remainingAmountUsd: entry.remainingAmountUsd ?? entry.remainingAmount,
                initialPrice: entry.initialLimitPrice,
                currentPrice,
                currentMaxPrice: decision.nextMaxPrice ?? baseMaxPrice,
                chaseRound: entry.chaseRound,
                chaseTickLift: profile?.chaseTickLift,
                finalAggressiveWindowSec: profile?.finalAggressiveWindowSec,
                finalAggressiveMaxPrice: profile?.finalAggressiveMaxPrice,
                reason: decision.reason,
            });
            this.maybeUpdateExecutionProfileOnLog(entry.logEntryId, profile, {
                executionState: shadowOnly && decision.shouldCancel ? 'waiting_fill' : entry.executionState,
                chaseRound: entry.chaseRound,
                firstChaseWaitSec: profile?.firstChaseWaitSec,
                nextChaseGapSec: profile?.nextChaseGapSec,
                chaseTriggered: decision.eventType === 'chase_reprice_posted' ? true : undefined,
                finalAggressiveTriggered: decision.eventType === 'final_aggressive_posted' ? true : undefined,
                cancelOnQueueStall: profile?.cancelOnQueueStall,
                queueStallWindowSec: profile?.queueStallWindowSec,
                staleCancelMinAgeSec: profile?.staleCancelMinAgeSec,
                staleCancelMinTimeToExpirySec: profile?.staleCancelMinTimeToExpirySec,
            });
        }

        const shouldCancel = Boolean(decision.shouldCancel && !shadowOnly);
        const shouldExecute = Boolean(
            decision.shouldPost &&
            currentPrice >= this.env.MIN_TOKEN_PRICE - 1e-9 &&
            currentPrice <= effectiveMaxPrice + 1e-9,
        );
        return {
            shouldExecute,
            shouldCancel,
            effectiveMaxPrice,
            reason: decision.reason,
        };
    }


    private getDirectionalMinEdge(direction: 'UP' | 'DOWN' | null | undefined): number {
        const fallback = Number.isFinite(Number(this.env.MIN_EDGE)) ? Number(this.env.MIN_EDGE) : 0;
        if (direction === 'UP') {
            const up = Number(this.env.MIN_EDGE_UP);
            return Number.isFinite(up) ? up : fallback;
        }
        if (direction === 'DOWN') {
            const down = Number(this.env.MIN_EDGE_DOWN);
            return Number.isFinite(down) ? down : fallback;
        }
        return fallback;
    }

    private getDirectionalConsensusThreshold(direction: 'UP' | 'DOWN' | null | undefined): number | null {
        const raw = direction === 'UP'
            ? this.env.CONSENSUS_THRESHOLD_UP
            : direction === 'DOWN'
                ? this.env.CONSENSUS_THRESHOLD_DOWN
                : undefined;
        const value = Number(raw);
        return Number.isFinite(value) && value > 0 && value < 1 ? value : null;
    }

    private getPredictionConsensusScore(prediction: PredictionResult): number | null {
        const value = Number(prediction.ensembleMeta?.consensusScore);
        return Number.isFinite(value) ? value : null;
    }

    private getConsensusGateFailure(
        prediction: PredictionResult,
        direction: 'UP' | 'DOWN' | null | undefined,
    ): { threshold: number; score: number } | null {
        const threshold = this.getDirectionalConsensusThreshold(direction);
        if (threshold == null) return null;
        const score = this.getPredictionConsensusScore(prediction);
        if (score == null) return null;
        if (score + 1e-9 >= threshold) return null;
        return { threshold, score };
    }

    private getBlockedOpportunityLogPath(): string {
        return path.join(this.logsDirFull, 'direction_loss_cooldown_blocked_opportunities.jsonl');
    }

    private appendCooldownBlockedOpportunity(args: {
        prediction: PredictionResult;
        marketResult?: MarketSearchResult | null;
        symbol: string;
        direction: 'UP' | 'DOWN';
        targetPeriodEndTs?: number;
        remainingBarsBefore: number;
        remainingBarsAfter: number;
        decrementedThisRound: boolean;
        reasonCode: 'cooldown_bars_blocked' | 'cooldown_pending_resolution_guard';
    }): void {
        try {
            const marketSlug = String(args.marketResult?.market?.slug || '');
            const conditionId = String(args.marketResult?.market?.conditionId || '');
            const targetTs = Number.isFinite(Number(args.targetPeriodEndTs)) ? Number(args.targetPeriodEndTs) : null;
            const periodStartTs = parsePeriodStartFromSlug(marketSlug);
            const key = [
                this.modeScope,
                this.logsDir,
                args.symbol,
                args.direction,
                marketSlug || 'no_market_slug',
                conditionId || 'no_condition_id',
                targetTs ?? 'no_target_ts',
                args.prediction.timestamp || 'no_prediction_ts',
                args.reasonCode,
            ].join('::');
            if (this.blockedOpportunityLogKeys.has(key)) return;
            this.blockedOpportunityLogKeys.add(key);
            const logPath = this.getBlockedOpportunityLogPath();
            const logDir = path.dirname(logPath);
            if (!fs.existsSync(logDir)) fs.mkdirSync(logDir, { recursive: true });
            const payload = {
                loggedAt: new Date().toISOString(),
                key,
                opportunityType: args.reasonCode,
                counterfactualUse: 'no_cooldown_replay_candidate',
                traderLogsDir: this.logsDir,
                modeScope: this.modeScope,
                symbol: args.symbol,
                direction: args.direction,
                marketSlug: marketSlug || null,
                conditionId: conditionId || null,
                targetPeriodEndTs: targetTs,
                periodStartTs,
                predictionTimestamp: args.prediction.timestamp || null,
                confidence: Number.isFinite(Number(args.prediction.confidence)) ? Number(args.prediction.confidence) : null,
                rawConfidence: Number.isFinite(Number(args.prediction.rawConfidence)) ? Number(args.prediction.rawConfidence) : null,
                calibratedConfidence: Number.isFinite(Number(args.prediction.calibratedConfidence)) ? Number(args.prediction.calibratedConfidence) : null,
                remainingBarsBefore: Math.max(0, Number(args.remainingBarsBefore) || 0),
                remainingBarsAfter: Math.max(0, Number(args.remainingBarsAfter) || 0),
                decrementedThisRound: Boolean(args.decrementedThisRound),
                cooldownScope: this.getCooldownScope(),
                cooldownKey: this.buildCooldownKey(args.symbol, args.direction),
                cooldownScopeLabel: this.getCooldownScope() === 'symbol' ? 'total_chain_by_symbol' : 'symbol_direction',
                explanationChinese: '该机会因 30 分钟冷却（cooldown_bars）被挡掉，用于后续无冷却反事实回放；这里只记证据，不直接造收益。',
            };
            fs.appendFileSync(logPath, JSON.stringify(payload) + '\n', 'utf-8');
        } catch {
            /* 机会账本写入失败不影响真实交易 */
        }
    }

    constructor(injectedEnv?: PredictionEnvConfig, injectedLogsDir?: string) {
        this.env = injectedEnv ?? PREDICTION_ENV;
        this.modeScope = this.env.TRADING_MODE as TradeLogModeScope;
        this.logsDir = injectedLogsDir ?? (process.env.LOGS_DIR || 'logs');
        this.logsDirFull = path.join(process.cwd(), this.logsDir);
        this.logger = injectedEnv
            ? createTradeLogger(this.logsDirFull, this.modeScope)
            : getTradeLogger(this.logsDirFull, this.modeScope);
        this.state = {
            isRunning: false,
            currentCapital: this.env.INITIAL_CAPITAL,
            todayTradeCount: 0,
            lastExecutionTime: null,
            marketCache: new Map(),
            peakCapital: this.env.INITIAL_CAPITAL,
            consecutiveLosses: 0,
            formalConsecutiveLosses: 0,
            provisionalConsecutiveLosses: 0,
            tradingPaused: false,
            pauseReason: null,
            pausedKBarsRemaining: 0,
            lastLossTs: 0,
            everReachedCap: false,
            drawdownHaltStartTime: null,
        };

        this.perCoinEnabled = this.env.PER_COIN_CAPITAL;
        this.capitalByCoin = new Map();
        this.peakCapitalByCoin = new Map();
        this.cooldownByCoin = new Map();
        this.cooldownByCoinDir = new Map();
        if (this.perCoinEnabled) {
            const numCoins = this.env.ALLOWED_MARKETS.length;
            const perCoinInit = this.env.INITIAL_CAPITAL / numCoins;
            for (const sym of this.env.ALLOWED_MARKETS) {
                this.capitalByCoin.set(sym, perCoinInit);
                this.peakCapitalByCoin.set(sym, perCoinInit);
            }
        }
        this.loadClaimedSuccessCache();
    }

    private getPendingSimOrdersFilePath(): string {
        return path.join(this.logsDirFull, `pending_sim_orders.${this.modeScope}.json`);
    }

    private getPendingOrderLedgerPath(): string {
        return path.join(this.logsDirFull, `pending_order_ledger.${this.modeScope}.jsonl`);
    }

    private serializePendingSimOrders(): PersistedPendingSimOrder[] {
        return this.pendingSimOrders.map((order) => ({
            id: order.id,
            symbol: order.symbol,
            direction: order.direction,
            frozenAmount: order.frozenAmount,
            limitPrice: order.limitPrice,
            tokenId: order.tokenId,
            tokenOutcome: order.tokenOutcome,
            confidence: order.confidence,
            marketTitle: order.marketTitle,
            conditionId: order.conditionId,
            marketSlug: order.marketSlug,
            targetPeriodEndTs: order.targetPeriodEndTs,
            createdAt: order.createdAt.toISOString(),
            expiresAt: order.expiresAt.toISOString(),
            bestAskAtEntry: order.bestAskAtEntry,
            queueCompetitionUsdAtEntry: order.queueCompetitionUsdAtEntry,
            effectiveFillableUsdAtEntry: order.effectiveFillableUsdAtEntry,
            avgActualFillPriceAtEntry: order.avgActualFillPriceAtEntry,
        }));
    }

    private persistPendingSimOrders(): void {
        try {
            const filePath = this.getPendingSimOrdersFilePath();
            const tmpPath = `${filePath}.tmp`;
            fs.writeFileSync(tmpPath, JSON.stringify(this.serializePendingSimOrders(), null, 2), 'utf-8');
            fs.renameSync(tmpPath, filePath);
        } catch (error) {
            console.error(`  [挂单持久化] 保存失败: ${error instanceof Error ? error.message : error}`);
        }
    }

    private appendPendingOrderLedgerEvent(event: Omit<PendingOrderLedgerEvent, 'timestamp'>): void {
        try {
            const roundedAmount = roundUsd(event.amountUsd);
            if ((event.event === 'filled' || event.event === 'partial_fill') && roundedAmount <= 0) {
                // V2 order/user-channel updates can report a cancellation/update with size_matched=0.
                // Never write that as a fill; it pollutes PnL, cooldown, claim and monitor truth.
                console.warn(`  [挂单账本] 忽略 0U ${event.event} 事件: ${event.orderId}`);
                return;
            }
            const payload: PendingOrderLedgerEvent = {
                timestamp: new Date().toISOString(),
                ...event,
                amountUsd: roundedAmount,
                remainingFrozenAmount: roundUsd(event.remainingFrozenAmount),
                bestAsk: event.bestAsk != null ? Math.round(Number(event.bestAsk) * 1_000_000) / 1_000_000 : event.bestAsk,
                avgFillPrice: event.avgFillPrice != null ? Math.round(Number(event.avgFillPrice) * 1_000_000) / 1_000_000 : event.avgFillPrice,
                queueCompetitionUsdAtEntry: event.queueCompetitionUsdAtEntry != null ? roundUsd(event.queueCompetitionUsdAtEntry) : event.queueCompetitionUsdAtEntry,
                effectiveFillableUsdAtEntry: event.effectiveFillableUsdAtEntry != null ? roundUsd(event.effectiveFillableUsdAtEntry) : event.effectiveFillableUsdAtEntry,
            };
            fs.appendFileSync(this.getPendingOrderLedgerPath(), JSON.stringify(payload) + '\n', 'utf-8');
        } catch (error) {
            console.error(`  [挂单账本] 写入失败: ${error instanceof Error ? error.message : error}`);
        }
    }

    private queuePendingSimOrder(order: PendingSimOrder): void {
        this.pendingSimOrders.push(order);
        this.persistPendingSimOrders();
        this.appendPendingOrderLedgerEvent({
            event: 'created',
            orderId: order.id,
            symbol: order.symbol,
            direction: order.direction,
            amountUsd: order.frozenAmount,
            remainingFrozenAmount: order.frozenAmount,
            limitPrice: order.limitPrice,
            tokenId: order.tokenId,
            tokenOutcome: order.tokenOutcome,
            confidence: order.confidence,
            marketTitle: order.marketTitle,
            conditionId: order.conditionId,
            marketSlug: order.marketSlug,
            targetPeriodEndTs: order.targetPeriodEndTs,
            createdAt: order.createdAt.toISOString(),
            expiresAt: order.expiresAt.toISOString(),
            bestAsk: order.bestAskAtEntry,
            avgFillPrice: order.avgActualFillPriceAtEntry,
            queueCompetitionUsdAtEntry: order.queueCompetitionUsdAtEntry,
            effectiveFillableUsdAtEntry: order.effectiveFillableUsdAtEntry,
        });
    }

    private loadPendingSimOrders(): void {
        const filePath = this.getPendingSimOrdersFilePath();
        if (!fs.existsSync(filePath)) return;
        try {
            const raw = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
            if (!Array.isArray(raw)) return;
            const now = Date.now();
            const restored: PendingSimOrder[] = [];
            for (const item of raw as PersistedPendingSimOrder[]) {
                const createdAt = new Date(item.createdAt);
                const expiresAt = new Date(item.expiresAt);
                if (!Number.isFinite(createdAt.getTime()) || !Number.isFinite(expiresAt.getTime())) continue;
                if (expiresAt.getTime() <= now) continue;
                restored.push({
                    ...item,
                    createdAt,
                    expiresAt,
                    checking: false,
                });
            }
            this.pendingSimOrders = restored;
            if (restored.length !== raw.length) {
                this.persistPendingSimOrders();
            }
            if (this.pendingSimOrders.length > 0) {
                const totalFrozen = roundUsd(this.pendingSimOrders.reduce((sum, order) => sum + order.frozenAmount, 0));
                console.log(`  📦 恢复模拟挂单 ${this.pendingSimOrders.length} 笔, 冻结资金 $${totalFrozen.toFixed(2)}`);
            }
        } catch (error) {
            console.error(`  [挂单持久化] 读取失败: ${error instanceof Error ? error.message : error}`);
            this.pendingSimOrders = [];
        }
    }
    
    /** 设置运行时仓位缩放（0~1）。 */
    setRuntimeBetScale(scale: number): void {
        if (!Number.isFinite(scale)) {
            this.runtimeBetScale = 1.0;
            return;
        }
        this.runtimeBetScale = Math.min(1, Math.max(0, Number(scale)));
    }

    /** 只读快照：当前可用资金（LIVE 为最近一次链上同步值）。 */
    getCurrentCapitalSnapshot(): number {
        return roundUsd(this.state.currentCapital);
    }

    /** 只读快照：按当前运行时缩放与风控状态预估本次下注金额。 */
    previewBetAmount(
        confidence?: number,
        tokenPrice?: number,
        symbol?: string,
        direction?: 'UP' | 'DOWN' | null,
    ): number {
        return roundUsd(this.calculateBetAmount(confidence, tokenPrice, symbol, direction));
    }

    private getRequestCompressionMode(): string {
        return String((this.env as any).REQUEST_COMPRESSION_MODE || 'legacy').trim().toLowerCase() || 'legacy';
    }

    private getExecutionCompressionMode(): string {
        return String((this.env as any).EXECUTION_COMPRESSION_MODE || 'legacy').trim().toLowerCase() || 'legacy';
    }

    private getSizingMode(): string {
        return String((this.env as any).SIZING_MODE || 'percent_cap').trim().toLowerCase() || 'percent_cap';
    }

    private getFixedTargetNotionalUsd(): number {
        const raw = Number((this.env as any).FIXED_TARGET_NOTIONAL_USD);
        return Number.isFinite(raw) && raw > 0 ? roundUsd(raw) : 0;
    }

    private getBankrollPct(): number {
        const raw = Number((this.env as any).BANKROLL_PCT);
        if (Number.isFinite(raw) && raw >= 0) return raw;
        const fallback = Number(this.env.BET_PCT_NORMAL);
        return Number.isFinite(fallback) && fallback >= 0 ? fallback : 0;
    }

    private getCapitalRiskCapPct(): number {
        const raw = Number((this.env as any).CAPITAL_RISK_CAP_PCT);
        if (Number.isFinite(raw) && raw >= 0) return Math.min(1, Math.max(0, raw));
        return 0.95;
    }

    private getCapitalRiskUpperLimit(availableCapital: number): number {
        return roundUsd(Math.max(0, availableCapital * this.getCapitalRiskCapPct()));
    }

    private getRuntimeRiskScalingMode(): string {
        return String((this.env as any).RUNTIME_RISK_SCALING_MODE || 'legacy').trim().toLowerCase() || 'legacy';
    }

    private getSymbolDirectionKey(symbol?: string, direction?: 'UP' | 'DOWN' | null): string | null {
        const sym = normalizeSymbol(symbol);
        const dir = this.normalizeDirection(direction);
        if (!sym || !dir) return null;
        return `${sym}_${dir}`;
    }

    private getDirectionalNumericOverride(
        rawMap: unknown,
        symbol?: string,
        direction?: 'UP' | 'DOWN' | null,
    ): number | null {
        const key = this.getSymbolDirectionKey(symbol, direction);
        if (!key || !rawMap || typeof rawMap !== 'object' || Array.isArray(rawMap)) return null;
        const value = (rawMap as Record<string, unknown>)[key];
        const parsed = Number(value);
        return Number.isFinite(parsed) ? parsed : null;
    }

    private getDirectionalPriceRangeOverride(
        rawMap: unknown,
        symbol?: string,
        direction?: 'UP' | 'DOWN' | null,
    ): [number, number] | null {
        const key = this.getSymbolDirectionKey(symbol, direction);
        if (!key || !rawMap || typeof rawMap !== 'object' || Array.isArray(rawMap)) return null;
        const value = (rawMap as Record<string, unknown>)[key];
        if (!Array.isArray(value) || value.length < 2) return null;
        const lower = Number(value[0]);
        const upper = Number(value[1]);
        if (!Number.isFinite(lower) || !Number.isFinite(upper) || upper < lower) return null;
        return [lower, upper];
    }

    private getDirectionalBetPctNormal(symbol?: string, direction?: 'UP' | 'DOWN' | null): number {
        const directional = this.getDirectionalNumericOverride((this.env as any).BET_PCT_NORMAL_BY_SYMBOL_DIRECTION, symbol, direction);
        if (Number.isFinite(directional as number) && Number(directional) >= 0) return Number(directional);
        const fallback = Number(this.env.BET_PCT_NORMAL);
        return Number.isFinite(fallback) && fallback >= 0 ? fallback : 0;
    }

    private getDirectionalBetPctConservative(symbol?: string, direction?: 'UP' | 'DOWN' | null): number {
        const directional = this.getDirectionalNumericOverride((this.env as any).BET_PCT_CONSERVATIVE_BY_SYMBOL_DIRECTION, symbol, direction);
        if (Number.isFinite(directional as number) && Number(directional) >= 0) return Number(directional);
        const fallback = Number(this.env.BET_PCT_CONSERVATIVE);
        return Number.isFinite(fallback) && fallback >= 0 ? fallback : 0;
    }

    private getDirectionalBetTargetCapUsd(confidence?: number, tokenPrice?: number | null, symbol?: string, direction?: 'UP' | 'DOWN' | null): number {
        const directional = this.getDirectionalNumericOverride((this.env as any).BET_TARGET_CAP_USD_BY_SYMBOL_DIRECTION, symbol, direction);
        if (Number.isFinite(directional as number) && Number(directional) > 0) {
            return Math.max(0, Number(directional));
        }
        return this.resolveBetTargetCapUsd(confidence, tokenPrice, symbol, direction);
    }

    private getDirectionalSizingSource(symbol?: string, direction?: 'UP' | 'DOWN' | null): string {
        const hasDirectionalPct =
            this.getDirectionalNumericOverride((this.env as any).BET_PCT_NORMAL_BY_SYMBOL_DIRECTION, symbol, direction) !== null
            || this.getDirectionalNumericOverride((this.env as any).BET_PCT_CONSERVATIVE_BY_SYMBOL_DIRECTION, symbol, direction) !== null
            || this.getDirectionalNumericOverride((this.env as any).BET_TARGET_CAP_USD_BY_SYMBOL_DIRECTION, symbol, direction) !== null;
        return hasDirectionalPct ? 'directional_override' : 'global_fallback';
    }

    private getDirectionalRuntimeBetPct(symbol?: string, direction?: 'UP' | 'DOWN' | null): number {
        if (this.getRuntimeRiskScalingMode() === 'cap_to_directional_conservative_with_hysteresis') {
            return this.isDirectionalConservativeActive(symbol)
                ? this.getDirectionalBetPctConservative(symbol, direction)
                : this.getDirectionalBetPctNormal(symbol, direction);
        }
        return this.getDirectionalBetPctNormal(symbol, direction);
    }

    private getRuntimeRiskScalingEnterDrawdownPct(): number {
        const raw = Number((this.env as any).RUNTIME_RISK_SCALING_ENTER_DRAWDOWN_PCT);
        return Number.isFinite(raw) ? Math.max(0, Math.min(0.95, raw)) : 0.20;
    }

    private getRuntimeRiskScalingRecoverDrawdownPct(): number {
        const raw = Number((this.env as any).RUNTIME_RISK_SCALING_RECOVER_DRAWDOWN_PCT);
        return Number.isFinite(raw) ? Math.max(0, Math.min(0.95, raw)) : 0.18;
    }

    private isDirectionalConservativeActive(symbol?: string): boolean {
        const sym = normalizeSymbol(symbol);
        if (!sym) return false;
        const mode = this.getRuntimeRiskScalingMode();
        if (mode !== 'cap_to_directional_conservative_with_hysteresis') {
            return false;
        }
        const coinCap = Math.max(0, this.getCoinCapital(sym) || 0);
        const coinPeak = Math.max(0, this.getCoinPeakCapital(sym) || 0);
        const currentDrawdown = coinPeak > 0 ? (coinPeak - coinCap) / coinPeak : 0;
        const enterPct = this.getRuntimeRiskScalingEnterDrawdownPct();
        const recoverPct = Math.min(enterPct, this.getRuntimeRiskScalingRecoverDrawdownPct());
        const prev = this.runtimeDirectionalConservativeBySymbol.get(sym) ?? false;
        const next = prev ? currentDrawdown >= recoverPct : currentDrawdown >= enterPct;
        if (prev !== next) {
            this.runtimeDirectionalConservativeBySymbol.set(sym, next);
            this._saveHaltState();
        } else if (!this.runtimeDirectionalConservativeBySymbol.has(sym)) {
            this.runtimeDirectionalConservativeBySymbol.set(sym, next);
        }
        return next;
    }

    private resolveDirectionalLowPriceRange(symbol?: string, direction?: 'UP' | 'DOWN' | null): [number, number] | null {
        const directional = this.getDirectionalPriceRangeOverride((this.env as any).LOWPRICE_SELECTED_BUY_PRICE_RANGE_BY_SYMBOL_DIRECTION, symbol, direction);
        if (directional) return directional;
        const ladder = String(this.env.LIMIT_PRICE_LADDER || '').trim();
        if (ladder) {
            const values = ladder.split(',').map((item) => Number(item.trim())).filter((value) => Number.isFinite(value));
            if (values.length > 0) {
                return [Math.min(...values), Math.max(...values)];
            }
        }
        return null;
    }

    private resolveDirectionalLowPriceRungCount(symbol?: string, direction?: 'UP' | 'DOWN' | null): number | null {
        const directional = this.getDirectionalNumericOverride((this.env as any).LOWPRICE_RUNG_COUNT_BY_SYMBOL_DIRECTION, symbol, direction);
        if (Number.isFinite(directional as number) && Number(directional) >= 1) {
            return Math.max(1, Math.round(Number(directional)));
        }
        const ladder = String(this.env.LIMIT_PRICE_LADDER || '').trim();
        if (!ladder) return null;
        const values = ladder.split(',').map((item) => Number(item.trim())).filter((value) => Number.isFinite(value));
        return values.length > 0 ? values.length : null;
    }

    private buildDirectionalVirtualLadderPrices(symbol?: string, direction?: 'UP' | 'DOWN' | null): number[] {
        const priceRange = this.resolveDirectionalLowPriceRange(symbol, direction);
        const rungCount = this.resolveDirectionalLowPriceRungCount(symbol, direction);
        if (!priceRange || !rungCount || rungCount <= 0) return [];
        if (rungCount === 1) return [this.roundTokenPriceTick(priceRange[0])];
        const [lower, upper] = priceRange;
        const step = (upper - lower) / Math.max(1, rungCount - 1);
        const prices: number[] = [];
        for (let idx = 0; idx < rungCount; idx++) {
            prices.push(this.roundTokenPriceTick(lower + step * idx));
        }
        return prices;
    }

    private calculateTargetNotionalUsd(
        confidence?: number,
        tokenPrice?: number,
        symbol?: string,
        direction?: 'UP' | 'DOWN' | null,
    ): number {
        const sizingMode = this.getSizingMode();
        if (sizingMode === 'fixed_usd') {
            const fixedTarget = this.getFixedTargetNotionalUsd();
            return fixedTarget >= MIN_ORDER_SIZE_USD ? fixedTarget : 0;
        }
        const coinCap = Math.max(0, this.getCoinCapital(symbol) || 0);
        if (coinCap <= 0) return 0;
        const sizingTrace = this.buildSizingTrace(confidence, symbol, direction);
        const targetPct = sizingMode === 'symmetric_dynamic_pct'
            ? sizingTrace.effectiveSizingPct
            : sizingTrace.bankrollPct;
        let targetAmount = coinCap * targetPct;
        const betTargetCapUsd = this.getDirectionalBetTargetCapUsd(confidence, tokenPrice, symbol, direction);
        if (betTargetCapUsd > 0 && targetAmount > betTargetCapUsd) {
            targetAmount = betTargetCapUsd;
        }
        const capitalRiskUpperLimit = this.getCapitalRiskUpperLimit(coinCap);
        if (targetAmount > capitalRiskUpperLimit) targetAmount = capitalRiskUpperLimit;
        if (targetAmount < MIN_ORDER_SIZE_USD) return 0;
        return roundUsd(targetAmount);
    }

    private resolveRequestedNotionalUsd(
        confidence?: number,
        tokenPrice?: number,
        symbol?: string,
        direction?: 'UP' | 'DOWN' | null,
        prediction?: PredictionResult | null,
    ): RequestedNotionalResolution {
        const sizingMode = this.getSizingMode();
        const sizingTrace = this.buildSizingTrace(confidence, symbol, direction);
        const gateAmountUsd = roundUsd(this.calculateBetAmount(confidence, tokenPrice, symbol, direction));
        const targetNotionalUsd = this.calculateTargetNotionalUsd(confidence, tokenPrice, symbol, direction);
        const requestCompressionMode = this.getRequestCompressionMode();
        let requestedNotionalUsd = requestCompressionMode === 'off' && targetNotionalUsd > 0
            ? targetNotionalUsd
            : gateAmountUsd;
        if (sizingMode === 'fixed_usd' && targetNotionalUsd > 0) {
            const availableCapital = Math.max(0, this.getCoinCapital(symbol) || 0);
            requestedNotionalUsd = availableCapital > 0
                ? Math.min(targetNotionalUsd, availableCapital)
                : 0;
        } else if (sizingMode === 'symmetric_dynamic_pct' && targetNotionalUsd > 0) {
            const availableCapital = Math.max(0, this.getCoinCapital(symbol) || 0);
            const capitalRiskUpperLimit = this.getCapitalRiskUpperLimit(availableCapital);
            requestedNotionalUsd = availableCapital > 0
                ? Math.min(targetNotionalUsd, capitalRiskUpperLimit > 0 ? capitalRiskUpperLimit : availableCapital)
                : 0;
        }
        const runtimeSizingCapPct = Number.isFinite(Number(prediction?.runtimeSizingCapPct))
            ? Math.max(0, Number(prediction?.runtimeSizingCapPct))
            : Number.NaN;
        let adjustedGateAmountUsd = gateAmountUsd;
        let adjustedTargetNotionalUsd = targetNotionalUsd;
        let adjustedRequestedNotionalUsd = roundUsd(requestedNotionalUsd);
        let adjustedEffectiveSizingPct = sizingTrace.effectiveSizingPct;
        let adjustedBandScale = sizingTrace.confidenceBandScale;
        if (Number.isFinite(runtimeSizingCapPct) && runtimeSizingCapPct > 0 && sizingTrace.currentCapital > 0) {
            const cappedPct = Math.min(Math.max(0, adjustedEffectiveSizingPct), runtimeSizingCapPct);
            const cappedUsd = roundUsd(sizingTrace.currentCapital * cappedPct);
            adjustedGateAmountUsd = roundUsd(Math.min(adjustedGateAmountUsd, cappedUsd));
            adjustedTargetNotionalUsd = roundUsd(Math.min(adjustedTargetNotionalUsd, cappedUsd));
            adjustedRequestedNotionalUsd = roundUsd(Math.min(adjustedRequestedNotionalUsd, cappedUsd));
            adjustedEffectiveSizingPct = cappedPct;
            adjustedBandScale = sizingTrace.bankrollPct > 0 ? cappedPct / sizingTrace.bankrollPct : sizingTrace.confidenceBandScale;
        }
        return {
            gateAmountUsd: adjustedGateAmountUsd,
            targetNotionalUsd: adjustedTargetNotionalUsd,
            requestedNotionalUsd: adjustedRequestedNotionalUsd,
            requestLayerCompressionUsd: roundUsd(Math.max(0, adjustedTargetNotionalUsd - adjustedRequestedNotionalUsd)),
            requestCompressionMode,
            sizingMode: sizingTrace.sizingMode,
            bankrollPct: sizingTrace.bankrollPct,
            confidenceBandScale: adjustedBandScale,
            effectiveSizingPct: adjustedEffectiveSizingPct,
            currentCapital: sizingTrace.currentCapital,
            runtimeSizingCapPct: Number.isFinite(runtimeSizingCapPct) ? runtimeSizingCapPct : undefined,
            runtimeSizingCapReason: prediction?.runtimeSizingCapReason,
        };
    }

    /** 设置按 symbol 的运行时仓位缩放（0~1）。 */
    setRuntimeSymbolScale(symbol: string, scale: number): void {
        const key = normalizeSymbol(symbol);
        if (!key) return;
        if (!Number.isFinite(scale)) {
            this.runtimeSymbolScaleBySymbol.set(key, 1);
            return;
        }
        this.runtimeSymbolScaleBySymbol.set(key, Math.min(1, Math.max(0, Number(scale))));
    }

    /** 设置运行时 DOWN 风控缩放（按 symbol，0~1，仅 DOWN 生效）。 */
    setRuntimeDownRiskScale(symbol: string, scale: number): void {
        const key = normalizeSymbol(symbol);
        if (!key) return;
        if (!Number.isFinite(scale)) {
            this.runtimeDownRiskScaleBySymbol.set(key, 1);
            return;
        }
        this.runtimeDownRiskScaleBySymbol.set(key, Math.min(1, Math.max(0, Number(scale))));
    }

    /** 设置运行时 DOWN 风控封禁（按 symbol，仅 DOWN 生效）。 */
    setRuntimeDownRiskBlocked(symbol: string, blocked: boolean): void {
        const key = normalizeSymbol(symbol);
        if (!key) return;
        this.runtimeDownRiskBlockedBySymbol.set(key, Boolean(blocked));
    }

    /** 设置运行时 UP 风控缩放（按 symbol，0~1，仅 UP 生效）。 */
    setRuntimeUpRiskScale(symbol: string, scale: number): void {
        const key = normalizeSymbol(symbol);
        if (!key) return;
        if (!Number.isFinite(scale)) {
            this.runtimeUpRiskScaleBySymbol.set(key, 1);
            return;
        }
        this.runtimeUpRiskScaleBySymbol.set(key, Math.min(1, Math.max(0, Number(scale))));
    }

    /** 设置运行时 UP 风控封禁（按 symbol，仅 UP 生效）。 */
    setRuntimeUpRiskBlocked(symbol: string, blocked: boolean): void {
        const key = normalizeSymbol(symbol);
        if (!key) return;
        this.runtimeUpRiskBlockedBySymbol.set(key, Boolean(blocked));
    }

    private buildRuntimeSymbolDirectionKey(symbol: string, direction: 'UP' | 'DOWN'): string {
        const sym = normalizeSymbol(symbol);
        return sym ? `${sym}::${direction}` : '';
    }

    /** 设置运行时期望值启停缩放（按 symbol+direction 注入）。 */
    setRuntimeExpectancyScale(symbol: string, direction: 'UP' | 'DOWN', scale: number): void {
        const key = this.buildRuntimeSymbolDirectionKey(symbol, direction);
        if (!key) return;
        if (!Number.isFinite(scale)) {
            this.runtimeExpectancyScaleBySymbolDir.set(key, 1);
            return;
        }
        this.runtimeExpectancyScaleBySymbolDir.set(key, Math.min(1, Math.max(0, Number(scale))));
    }

    /** 设置运行时期望值启停封禁（按 symbol+direction 注入）。 */
    setRuntimeExpectancyBlocked(symbol: string, direction: 'UP' | 'DOWN', blocked: boolean): void {
        const key = this.buildRuntimeSymbolDirectionKey(symbol, direction);
        if (!key) return;
        this.runtimeExpectancyBlockedBySymbolDir.set(key, Boolean(blocked));
    }

    /** 设置运行时高置信同方向连错轻缩仓（按 symbol+direction 注入）。 */
    setRuntimeDirectionLossScale(symbol: string, direction: 'UP' | 'DOWN', scale: number): void {
        const key = this.buildRuntimeSymbolDirectionKey(symbol, direction);
        if (!key) return;
        if (!Number.isFinite(scale)) {
            this.runtimeDirectionLossScaleBySymbolDir.set(key, 1);
            return;
        }
        this.runtimeDirectionLossScaleBySymbolDir.set(key, Math.min(1, Math.max(0, Number(scale))));
    }

    /** 设置运行时回撤加速度观察态轻缩仓（按 symbol 注入）。 */
    setRuntimeDrawdownAccelerationScale(symbol: string, scale: number): void {
        const key = normalizeSymbol(symbol);
        if (!key) return;
        if (!Number.isFinite(scale)) {
            this.runtimeDrawdownAccelerationScaleBySymbol.set(key, 1);
            return;
        }
        this.runtimeDrawdownAccelerationScaleBySymbol.set(key, Math.min(1, Math.max(0, Number(scale))));
    }

    private normalizeDirection(direction?: string | null): 'UP' | 'DOWN' | null {
        const d = String(direction || '').toUpperCase();
        if (d === 'UP' || d === 'DOWN') return d;
        return null;
    }

    private getCooldownScope(): 'symbol' | 'symbol_direction' {
        return this.env.COOLDOWN_SCOPE === 'symbol_direction' ? 'symbol_direction' : 'symbol';
    }

    private buildCooldownKey(symbol: string, direction?: string | null): string {
        const sym = normalizeSymbol(symbol) || String(symbol || '').toUpperCase();
        if (this.getCooldownScope() === 'symbol') return sym;
        const dir = this.normalizeDirection(direction) || 'UNKNOWN';
        return `${sym}::${dir}`;
    }

    private formatCooldownLabel(symbol: string, direction?: string | null): string {
        const sym = normalizeSymbol(symbol) || String(symbol || '').toUpperCase();
        if (this.getCooldownScope() === 'symbol') return sym;
        const dir = this.normalizeDirection(direction);
        return dir ? `${sym}-${dir}` : sym;
    }

    private getCooldownRemaining(symbol: string, direction?: string | null): number {
        const key = this.buildCooldownKey(symbol, direction);
        const current = this.cooldownByCoinDir.get(key);
        if (Number.isFinite(current as number)) {
            return Math.max(0, Number(current));
        }
        const sym = normalizeSymbol(symbol) || String(symbol || '').toUpperCase();
        // 兼容总链冷却：历史/报告链会把总链冷却写成 "ETH" / "BTC"。
        // 即使某个运行环境误落回 symbol_direction，也不能因此绕开已有总链冷却。
        const totalChainInDirMap = this.cooldownByCoinDir.get(sym);
        if (Number.isFinite(totalChainInDirMap as number)) {
            return Math.max(0, Number(totalChainInDirMap));
        }
        const totalChainInLegacyMap = this.cooldownByCoin.get(sym);
        if (Number.isFinite(totalChainInLegacyMap as number)) {
            return Math.max(0, Number(totalChainInLegacyMap));
        }
        // 兼容迁移：旧版按 symbol 的冷却值回填到新映射
        if (this.getCooldownScope() === 'symbol') {
            const legacy = this.cooldownByCoin.get(sym);
            if (Number.isFinite(legacy as number)) {
                const v = Math.max(0, Number(legacy));
                this.cooldownByCoinDir.set(key, v);
                return v;
            }
        }
        return 0;
    }

    private setCooldownRemaining(symbol: string, direction: string | null | undefined, remainingBars: number): void {
        const key = this.buildCooldownKey(symbol, direction);
        const v = Math.max(0, Math.floor(Number(remainingBars) || 0));
        this.cooldownByCoinDir.set(key, v);
        const sym = normalizeSymbol(symbol) || String(symbol || '').toUpperCase();
        if (this.getCooldownScope() === 'symbol') {
            // 仅用于兼容：保留旧 map 同步值
            this.cooldownByCoin.set(sym, v);
        }
        this._saveHaltState();
    }

    private getCooldownAfterLossesNum(): number {
        return Math.max(1, Number((this.env as any).COOLDOWN_AFTER_LOSSES) || 1);
    }

    private getCooldownBarsNum(): number {
        return Math.max(0, Number(this.env.COOLDOWN_BARS) || 0);
    }

    private hasExecutableFill(entry: TradeLogEntry): boolean {
        return Number(entry.amount ?? 0) > 0 || Number(entry.matchedShares ?? 0) > 0;
    }

    private isCooldownEligibleEntry(entry: TradeLogEntry): boolean {
        if (!entry || !entry.symbol) return false;
        if (!this.env.ALLOWED_MARKETS.includes(entry.symbol)) return false;
        const settledLikeExecuted =
            entry.status === 'executed' ||
            ((entry.status === 'failed' || entry.status === 'pending') && this.hasExecutableFill(entry));
        return settledLikeExecuted;
    }

    private getEffectiveTradeResult(entry: TradeLogEntry, includeProvisional: boolean): 'win' | 'lose' | null {
        if (entry.result === 'win' || entry.result === 'lose') return entry.result;
        if (includeProvisional && (entry.provisionalResult === 'win' || entry.provisionalResult === 'lose')) {
            return entry.provisionalResult;
        }
        return null;
    }

    private buildCooldownConditionToSlugAlias(entries: TradeLogEntry[]): Map<string, string> {
        const aliases = new Map<string, string>();
        for (const entry of entries) {
            const conditionId = String(entry.conditionId || '').trim().toLowerCase();
            const marketSlug = String(entry.marketSlug || '').trim().toLowerCase();
            if (conditionId && marketSlug && !aliases.has(conditionId)) {
                aliases.set(conditionId, marketSlug);
            }
        }
        return aliases;
    }

    private buildCooldownLogicalMarketKey(entry: TradeLogEntry, conditionToSlug?: Map<string, string>): string {
        const symbol = normalizeSymbol(entry.symbol) || entry.symbol;
        const conditionId = String(entry.conditionId || '').trim().toLowerCase();
        const marketSlug = String(entry.marketSlug || '').trim().toLowerCase();
        const aliasedSlug = conditionId ? conditionToSlug?.get(conditionId) || null : null;
        const marketKey = marketSlug || aliasedSlug || conditionId || String(entry.id || '').trim();
        return `${symbol}::${marketKey || entry.id}`;
    }

    private getLogicalCooldownMarketEntryIds(entry: TradeLogEntry): string[] {
        const allEntries = this.logger.getAllEntries();
        const aliases = this.buildCooldownConditionToSlugAlias(allEntries);
        const targetKey = this.buildCooldownLogicalMarketKey(entry, aliases);
        return allEntries
            .filter((item) => this.buildCooldownLogicalMarketKey(item, aliases) === targetKey)
            .map((item) => item.id);
    }

    private updateLogicalCooldownMarketEntries(entry: TradeLogEntry, updates: Partial<TradeLogEntry>): number {
        const ids = this.getLogicalCooldownMarketEntryIds(entry);
        if (ids.length === 0) return 0;
        return this.logger.updateEntries(ids, updates);
    }

    private repairLogicalCooldownMetadataFromLedger(): { groupsTouched: number; entriesUpdated: number } {
        const cooldownAfterLossesNum = this.getCooldownAfterLossesNum();
        const allEntries = this.logger.getAllEntries();
        const aliases = this.buildCooldownConditionToSlugAlias(allEntries);
        const groups = new Map<string, TradeLogEntry[]>();
        for (const entry of allEntries) {
            const key = this.buildCooldownLogicalMarketKey(entry, aliases);
            if (!groups.has(key)) groups.set(key, []);
            groups.get(key)!.push(entry);
        }

        let groupsTouched = 0;
        let entriesUpdated = 0;
        for (const groupEntries of groups.values()) {
            if (groupEntries.length <= 1) continue;
            const probe = groupEntries[0];
            const pickResult = (field: 'result' | 'formalResult' | 'provisionalResult'): 'win' | 'lose' | null => {
                const wins = groupEntries.some((entry) => entry[field] === 'win');
                if (wins) return 'win';
                const losses = groupEntries.some((entry) => entry[field] === 'lose');
                return losses ? 'lose' : null;
            };
            const canonicalResult = pickResult('result');
            const canonicalFormalResult = pickResult('formalResult') || canonicalResult;
            const canonicalProvisionalResult = pickResult('provisionalResult') || canonicalFormalResult || canonicalResult;
            const canonicalStage = (
                groupEntries.some((entry) => entry.cooldownTriggeredStage === 'formal')
                    ? 'formal'
                    : groupEntries.some((entry) => entry.cooldownTriggeredStage === 'formal_expired_backfill')
                        ? 'formal_expired_backfill'
                        : groupEntries.some((entry) => entry.cooldownTriggeredStage === 'provisional')
                            ? 'provisional'
                            : null
            ) as TradeLogEntry['cooldownTriggeredStage'] | null;
            const canonicalScope = (
                groupEntries.find((entry) => entry.cooldownScope === 'symbol' || entry.cooldownScope === 'symbol_direction')?.cooldownScope
                || this.getCooldownScope()
            ) as TradeLogEntry['cooldownScope'] | undefined;
            const canonicalKey = groupEntries.find((entry) => typeof entry.cooldownKey === 'string' && entry.cooldownKey.trim().length > 0)?.cooldownKey
                || this.buildCooldownKey(probe.symbol, probe.direction);
            const latestProvisionalEvaluatedAt = groupEntries
                .map((entry) => entry.provisionalEvaluatedAt)
                .filter((value): value is string => typeof value === 'string' && value.trim().length > 0)
                .sort()
                .slice(-1)[0];

            const nums = (field: 'formalConsecutiveLosses' | 'provisionalConsecutiveLosses') =>
                groupEntries
                    .map((entry) => Number(entry[field]))
                    .filter((value) => Number.isFinite(value));
            const formalNums = nums('formalConsecutiveLosses');
            const provisionalNums = nums('provisionalConsecutiveLosses');

            let canonicalFormalConsecutiveLosses: number | null = null;
            if (canonicalFormalResult === 'win' || canonicalResult === 'win') {
                canonicalFormalConsecutiveLosses = 0;
            } else if (canonicalStage === 'formal') {
                canonicalFormalConsecutiveLosses = 0;
            } else if (canonicalStage === 'formal_expired_backfill') {
                canonicalFormalConsecutiveLosses = cooldownAfterLossesNum;
            } else if (formalNums.length > 0) {
                canonicalFormalConsecutiveLosses = Math.max(...formalNums);
            }

            let canonicalProvisionalConsecutiveLosses: number | null = null;
            if (canonicalProvisionalResult === 'win' || canonicalFormalResult === 'win' || canonicalResult === 'win') {
                canonicalProvisionalConsecutiveLosses = 0;
            } else if (canonicalStage === 'provisional') {
                canonicalProvisionalConsecutiveLosses = cooldownAfterLossesNum;
            } else if (provisionalNums.length > 0) {
                canonicalProvisionalConsecutiveLosses = Math.max(...provisionalNums);
            }

            const sharedUpdates: Partial<TradeLogEntry> = {};
            if (canonicalResult) sharedUpdates.result = canonicalResult;
            if (canonicalFormalResult) sharedUpdates.formalResult = canonicalFormalResult;
            if (canonicalProvisionalResult) sharedUpdates.provisionalResult = canonicalProvisionalResult;
            if (canonicalStage) sharedUpdates.cooldownTriggeredStage = canonicalStage;
            if (canonicalScope) sharedUpdates.cooldownScope = canonicalScope;
            if (canonicalKey) sharedUpdates.cooldownKey = canonicalKey;
            if (latestProvisionalEvaluatedAt) sharedUpdates.provisionalEvaluatedAt = latestProvisionalEvaluatedAt;
            if (canonicalFormalConsecutiveLosses != null) sharedUpdates.formalConsecutiveLosses = canonicalFormalConsecutiveLosses;
            if (canonicalProvisionalConsecutiveLosses != null) sharedUpdates.provisionalConsecutiveLosses = canonicalProvisionalConsecutiveLosses;

            const needsRepair = groupEntries.some((entry) => {
                if (sharedUpdates.result && entry.result !== sharedUpdates.result) return true;
                if (sharedUpdates.formalResult && entry.formalResult !== sharedUpdates.formalResult) return true;
                if (sharedUpdates.provisionalResult && entry.provisionalResult !== sharedUpdates.provisionalResult) return true;
                if (sharedUpdates.cooldownTriggeredStage && entry.cooldownTriggeredStage !== sharedUpdates.cooldownTriggeredStage) return true;
                if (sharedUpdates.cooldownScope && entry.cooldownScope !== sharedUpdates.cooldownScope) return true;
                if (sharedUpdates.cooldownKey && entry.cooldownKey !== sharedUpdates.cooldownKey) return true;
                if (
                    sharedUpdates.formalConsecutiveLosses != null
                    && Number(entry.formalConsecutiveLosses) !== Number(sharedUpdates.formalConsecutiveLosses)
                ) return true;
                if (
                    sharedUpdates.provisionalConsecutiveLosses != null
                    && Number(entry.provisionalConsecutiveLosses) !== Number(sharedUpdates.provisionalConsecutiveLosses)
                ) return true;
                return false;
            });
            if (!needsRepair) continue;
            groupsTouched += 1;
            entriesUpdated += this.logger.updateEntries(groupEntries.map((item) => item.id), sharedUpdates);
        }

        if (entriesUpdated > 0) {
            this.refreshCooldownLossStateFromHistory();
        }
        return { groupsTouched, entriesUpdated };
    }

    private hasPotentialLiveExposure(entry: TradeLogEntry): boolean {
        return (
            this.hasExecutableFill(entry) ||
            Number(entry.pendingAmountUsd ?? 0) > 0 ||
            Number(entry.requestedAmountUsd ?? 0) > 0 ||
            Boolean(String(entry.orderId || entry.txHash || '').trim())
        );
    }

    private getCooldownPendingResolutionMaxAgeSec(): number {
        // Pending-resolution guard is only meant to protect the immediate
        // post-loss cooldown window. After that window has passed, an old
        // unresolved partial fill is a reconciliation/reporting problem, not a
        // reason to block new live entries forever.
        const cooldownBarsNum = this.getCooldownBarsNum();
        return Math.max(this.periodSeconds + 120, (cooldownBarsNum + 1) * this.periodSeconds + 120);
    }

    private isStaleCooldownPendingResolution(entry: TradeLogEntry): boolean {
        const parsedPeriodStartTs = parsePeriodStartFromSlug(entry.marketSlug);
        const periodStartTs = Number.isFinite(Number(parsedPeriodStartTs)) ? Number(parsedPeriodStartTs) : 0;
        if (periodStartTs <= 0) return false;
        const periodEndTs = periodStartTs + this.periodSeconds;
        const nowSec = Math.floor(Date.now() / 1000);
        if (nowSec <= periodEndTs + this.getCooldownPendingResolutionMaxAgeSec()) return false;
        return this.getEffectiveTradeResult(entry, true) !== 'win' && this.getEffectiveTradeResult(entry, true) !== 'lose';
    }

    private hasCooldownPendingResolutionExposure(entry: TradeLogEntry): boolean {
        if (!entry || entry.stoppedOut) return false;
        if (entry.result === 'win' || entry.result === 'lose') return false;
        if (entry.provisionalResult === 'win' || entry.provisionalResult === 'lose') return false;
        if (this.isStaleCooldownPendingResolution(entry)) return false;

        // 已有真实成交金额/份额，市场结束后必须先预结算，避免第 4 单漏挡。
        if (this.hasExecutableFill(entry)) return true;

        const parsedPeriodStartTs = parsePeriodStartFromSlug(entry.marketSlug);
        const periodStartTs = Number.isFinite(Number(parsedPeriodStartTs)) ? Number(parsedPeriodStartTs) : 0;
        const periodEndTs = periodStartTs > 0 ? periodStartTs + this.periodSeconds : 0;
        const nowSec = Date.now() / 1000;
        if (
            periodEndTs > 0 &&
            nowSec > periodEndTs + 60 &&
            Number(entry.amount ?? 0) <= 0 &&
            Number(entry.matchedShares ?? 0) <= 0 &&
            Number(entry.filledNotionalUsd ?? 0) <= 0
        ) {
            // 只有“有成交但结果未入账”的市场才应卡住第 4 单。
            // 0U 挂单在市场结束后已经不可能再成交，不能长期误当成待确认敞口。
            return false;
        }

        const filledExposure =
            Number(entry.amount ?? 0) > 0 ||
            Number(entry.matchedShares ?? 0) > 0 ||
            Number(entry.filledNotionalUsd ?? 0) > 0;
        const effectiveResult = this.getEffectiveTradeResult(entry, true);
        if (
            periodEndTs > 0 &&
            nowSec >= periodEndTs &&
            filledExposure &&
            effectiveResult !== 'win' &&
            effectiveResult !== 'lose'
        ) {
            // A filled market whose 15m window already ended can be the 3rd
            // objective loss before formal settlement lands. Treat it as a
            // pending cooldown resolution so the next bar cannot become a 4th
            // live entry while the result is still being reconciled.
            return true;
        }

        const liveOrderStatus = String(entry.liveOrderStatus || '').trim();
        const openOrderState = (
            entry.status === 'pending' ||
            liveOrderStatus === 'posted_pending' ||
            liveOrderStatus === 'partially_filled'
        );
        if (!openOrderState) return false;

        // 只把仍可能成交/部分成交的挂单视为待确认敞口。
        // 已经被公开 activity 证明无成交并写成 expired/rejected/cancelled 的 0U 记录，
        // 不能继续卡住总链冷却。
        return (
            Number(entry.pendingAmountUsd ?? 0) > 0 ||
            Number(entry.matchedShares ?? 0) > 0 ||
            Number(entry.amount ?? 0) > 0 ||
            Boolean(String(entry.orderId || entry.txHash || '').trim())
        );
    }

    private shouldBlockForPendingCooldownResolution(symbol: string): {
        blocked: boolean;
        label: string;
        reasonCode: 'cooldown_bars_blocked' | 'cooldown_pending_resolution_guard';
        unresolvedMarketSlug?: string;
        consecutiveBeforeUnknown?: number;
    } {
        const cooldownAfterLossesNum = this.getCooldownAfterLossesNum();
        const cooldownBarsNum = this.getCooldownBarsNum();
        if (cooldownBarsNum <= 0 || cooldownAfterLossesNum <= 1) {
            return { blocked: false, label: this.formatCooldownLabel(symbol), reasonCode: 'cooldown_pending_resolution_guard' };
        }

        const sym = normalizeSymbol(symbol) || String(symbol || '').toUpperCase();
        const consecutive = this.computeCooldownLossState(true).consecutive;
        if (consecutive < cooldownAfterLossesNum - 1) {
            return { blocked: false, label: this.formatCooldownLabel(symbol), reasonCode: 'cooldown_pending_resolution_guard', consecutiveBeforeUnknown: consecutive };
        }

        const entries = this.logger
            .getAllEntries()
            .filter((entry) => (normalizeSymbol(entry.symbol) || entry.symbol) === sym)
            .filter((entry) => (entry.marketSlug || entry.conditionId) && this.hasCooldownPendingResolutionExposure(entry))
            .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

        const groups = new Map<string, {
            entries: TradeLogEntry[];
            timestampMs: number;
            result: 'win' | 'lose' | null;
            marketSlug?: string;
        }>();
        for (const entry of entries) {
            const key = this.buildCooldownLogicalMarketKey(entry);
            const ts = new Date(entry.timestamp).getTime();
            let group = groups.get(key);
            if (!group) {
                group = { entries: [], timestampMs: Number.isFinite(ts) ? ts : 0, result: null, marketSlug: entry.marketSlug };
                groups.set(key, group);
            }
            group.entries.push(entry);
            if (!group.marketSlug && entry.marketSlug) group.marketSlug = entry.marketSlug;
            if (Number.isFinite(ts) && (group.timestampMs <= 0 || ts < group.timestampMs)) {
                group.timestampMs = ts;
            }
            const result = this.getEffectiveTradeResult(entry, true);
            if (result === 'win') {
                group.result = 'win';
            } else if (result === 'lose' && group.result !== 'win') {
                group.result = 'lose';
            }
        }

        const ordered = Array.from(groups.values()).sort((a, b) => a.timestampMs - b.timestampMs);
        const last = ordered[ordered.length - 1];
        if (!last || last.result) {
            return { blocked: false, label: this.formatCooldownLabel(symbol), reasonCode: 'cooldown_pending_resolution_guard', consecutiveBeforeUnknown: consecutive };
        }

        const marketStartTs = parsePeriodStartFromSlug(last.marketSlug);
        const nowSec = Math.floor(Date.now() / 1000);
        const ended = marketStartTs != null && nowSec >= marketStartTs + this.periodSeconds;
        if (!ended) {
            return { blocked: false, label: this.formatCooldownLabel(symbol), reasonCode: 'cooldown_pending_resolution_guard', unresolvedMarketSlug: last.marketSlug, consecutiveBeforeUnknown: consecutive };
        }

        return {
            blocked: true,
            label: this.formatCooldownLabel(symbol),
            reasonCode: 'cooldown_pending_resolution_guard',
            unresolvedMarketSlug: last.marketSlug,
            consecutiveBeforeUnknown: consecutive,
        };
    }

    private getPersistedCooldownWindowBlock(
        symbol: string,
        direction?: string | null,
        targetPeriodStartTs?: number,
    ): {
        blocked: boolean;
        label: string;
        triggerMarketSlug?: string;
        triggerStage?: string;
        remainingBarsBefore: number;
        remainingBarsAfter: number;
        error?: string;
    } {
        const snapshot = this.peekPersistedCooldownWindowBlock(symbol, direction, targetPeriodStartTs);
        if (snapshot.blocked) {
            this.setCooldownRemaining(symbol, direction, snapshot.remainingBarsAfter);
        }
        return snapshot;
    }

    private peekPersistedCooldownWindowBlock(
        symbol: string,
        direction?: string | null,
        targetPeriodStartTs?: number,
    ): {
        blocked: boolean;
        label: string;
        triggerMarketSlug?: string;
        triggerStage?: string;
        remainingBarsBefore: number;
        remainingBarsAfter: number;
        error?: string;
    } {
        const cooldownBarsNum = this.getCooldownBarsNum();
        if (cooldownBarsNum <= 0 || !targetPeriodStartTs || !Number.isFinite(targetPeriodStartTs)) {
            return { blocked: false, label: this.formatCooldownLabel(symbol, direction), remainingBarsBefore: 0, remainingBarsAfter: 0 };
        }

        const sym = normalizeSymbol(symbol) || String(symbol || '').toUpperCase();
        const currentStartTs = Math.floor(Number(targetPeriodStartTs) > 1e12 ? Number(targetPeriodStartTs) / 1000 : Number(targetPeriodStartTs));
        const candidates = this.logger
            .getAllEntries()
            .filter((entry) => (normalizeSymbol(entry.symbol) || entry.symbol) === sym)
            .filter((entry) => entry.cooldownTriggeredStage === 'formal' || entry.cooldownTriggeredStage === 'provisional')
            .filter((entry) => {
                if (this.getCooldownScope() === 'symbol') return true;
                return this.buildCooldownKey(entry.symbol, entry.direction) === this.buildCooldownKey(symbol, direction);
            })
            .map((entry) => {
                const triggerStartTs = parsePeriodStartFromSlug(entry.marketSlug);
                return { entry, triggerStartTs };
            })
            .filter((item): item is { entry: TradeLogEntry; triggerStartTs: number } => item.triggerStartTs != null && Number.isFinite(item.triggerStartTs))
            .sort((a, b) => b.triggerStartTs - a.triggerStartTs);

        for (const item of candidates) {
            if (currentStartTs <= item.triggerStartTs) continue;
            const barsAfterTrigger = Math.floor((currentStartTs - item.triggerStartTs) / this.periodSeconds);
            if (barsAfterTrigger >= 1 && barsAfterTrigger <= cooldownBarsNum) {
                const remainingBarsBefore = Math.max(1, cooldownBarsNum - barsAfterTrigger + 1);
                const remainingBarsAfter = Math.max(0, remainingBarsBefore - 1);
                const label = this.formatCooldownLabel(symbol, direction);
                return {
                    blocked: true,
                    label,
                    triggerMarketSlug: item.entry.marketSlug,
                    triggerStage: item.entry.cooldownTriggeredStage,
                    remainingBarsBefore,
                    remainingBarsAfter,
                    error: `冷却中: ${label} 由 ${item.entry.marketSlug || '上一亏损市场'} 触发，当前仍处于后续第 ${barsAfterTrigger}/${cooldownBarsNum} 根冷却窗口`,
                };
            }
        }

        return { blocked: false, label: this.formatCooldownLabel(symbol, direction), remainingBarsBefore: 0, remainingBarsAfter: 0 };
    }

    public getCooldownDecisionDebugSnapshot(
        symbol: string,
        direction?: string | null,
        targetPeriodStartTs?: number,
    ): {
        cooldownScope: 'symbol' | 'symbol_direction';
        cooldownKey: string;
        cooldownScopeLabel: 'total_chain_by_symbol' | 'symbol_direction';
        formalConsecutiveLosses: number;
        provisionalConsecutiveLosses: number;
        effectiveConsecutiveLosses: number;
        cooldownRemainingBars: number;
        pendingResolutionBlocked: boolean;
        pendingResolutionMarketSlug?: string;
        pendingResolutionConsecutiveBeforeUnknown?: number;
        persistedWindowBlocked: boolean;
        persistedTriggerMarketSlug?: string;
        persistedTriggerStage?: string;
        persistedRemainingBarsBefore: number;
        persistedRemainingBarsAfter: number;
        persistedError?: string;
    } {
        this.refreshCooldownLossStateFromHistory();
        const scope = this.getCooldownScope();
        const cooldownKey = this.buildCooldownKey(symbol, direction);
        const pendingResolutionGuard = this.shouldBlockForPendingCooldownResolution(symbol);
        const persistedWindowBlock = this.peekPersistedCooldownWindowBlock(symbol, direction, targetPeriodStartTs);
        return {
            cooldownScope: scope,
            cooldownKey,
            cooldownScopeLabel: scope === 'symbol' ? 'total_chain_by_symbol' : 'symbol_direction',
            formalConsecutiveLosses: this.state.formalConsecutiveLosses,
            provisionalConsecutiveLosses: this.state.provisionalConsecutiveLosses,
            effectiveConsecutiveLosses: this.state.consecutiveLosses,
            cooldownRemainingBars: this.getCooldownRemaining(symbol, direction),
            pendingResolutionBlocked: pendingResolutionGuard.blocked,
            pendingResolutionMarketSlug: pendingResolutionGuard.unresolvedMarketSlug,
            pendingResolutionConsecutiveBeforeUnknown: pendingResolutionGuard.consecutiveBeforeUnknown,
            persistedWindowBlocked: persistedWindowBlock.blocked,
            persistedTriggerMarketSlug: persistedWindowBlock.triggerMarketSlug,
            persistedTriggerStage: persistedWindowBlock.triggerStage,
            persistedRemainingBarsBefore: persistedWindowBlock.remainingBarsBefore,
            persistedRemainingBarsAfter: persistedWindowBlock.remainingBarsAfter,
            persistedError: persistedWindowBlock.error,
        };
    }

    private evaluateHardCooldownSubmissionBlock(symbol: string, direction?: string | null, targetPeriodStartTs?: number): {
        blocked: boolean;
        label: string;
        reasonCode: 'cooldown_bars_blocked' | 'cooldown_pending_resolution_guard';
        remainingBarsBefore: number;
        remainingBarsAfter: number;
        unresolvedMarketSlug?: string;
        consecutiveBeforeUnknown?: number;
        error?: string;
    } {
        const pendingResolutionGuard = this.shouldBlockForPendingCooldownResolution(symbol);
        if (pendingResolutionGuard.blocked) {
            return {
                blocked: true,
                label: pendingResolutionGuard.label,
                reasonCode: pendingResolutionGuard.reasonCode,
                remainingBarsBefore: 0,
                remainingBarsAfter: 0,
                unresolvedMarketSlug: pendingResolutionGuard.unresolvedMarketSlug,
                consecutiveBeforeUnknown: pendingResolutionGuard.consecutiveBeforeUnknown,
                error: `冷却待确认: ${pendingResolutionGuard.label} 已有 ${pendingResolutionGuard.consecutiveBeforeUnknown} 连输，上一根 ${pendingResolutionGuard.unresolvedMarketSlug || ''} 已结束但结果尚未入账，先挡本轮`,
            };
        }
        const persistedWindowBlock = this.getPersistedCooldownWindowBlock(symbol, direction, targetPeriodStartTs);
        if (persistedWindowBlock.blocked) {
            this.setCooldownRemaining(symbol, direction, persistedWindowBlock.remainingBarsAfter);
            return {
                blocked: true,
                label: persistedWindowBlock.label,
                reasonCode: 'cooldown_bars_blocked',
                remainingBarsBefore: persistedWindowBlock.remainingBarsBefore,
                remainingBarsAfter: persistedWindowBlock.remainingBarsAfter,
                error: persistedWindowBlock.error,
            };
        }
        const remaining = this.getCooldownRemaining(symbol, direction);
        if (remaining > 0) {
            const label = this.formatCooldownLabel(symbol, direction);
            return {
                blocked: true,
                label,
                reasonCode: 'cooldown_bars_blocked',
                remainingBarsBefore: remaining,
                remainingBarsAfter: remaining,
                error: `冷却中: ${label} 剩余 ${remaining} 个周期`,
            };
        }
        return {
            blocked: false,
            label: this.formatCooldownLabel(symbol, direction),
            reasonCode: 'cooldown_bars_blocked',
            remainingBarsBefore: 0,
            remainingBarsAfter: 0,
        };
    }

    private computeCooldownLossState(includeProvisional: boolean): {
        consecutive: number;
        triggerCandidates: TradeLogEntry[];
    } {
        const cooldownAfterLossesNum = this.getCooldownAfterLossesNum();
        const cooldownBarsNum = this.getCooldownBarsNum();
        const entries = this.logger
            .getAllEntries()
            .filter((entry) => this.isCooldownEligibleEntry(entry))
            .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
        const aliases = this.buildCooldownConditionToSlugAlias(entries);
        const logicalGroups = new Map<string, {
            entries: TradeLogEntry[];
            timestampMs: number;
            result: 'win' | 'lose' | null;
            cooldownAlreadyTriggered: boolean;
        }>();
        for (const entry of entries) {
            const groupKey = this.buildCooldownLogicalMarketKey(entry, aliases);
            const ts = new Date(entry.timestamp).getTime();
            let group = logicalGroups.get(groupKey);
            if (!group) {
                group = {
                    entries: [],
                    timestampMs: Number.isFinite(ts) ? ts : 0,
                    result: null,
                    cooldownAlreadyTriggered: false,
                };
                logicalGroups.set(groupKey, group);
            }
            group.entries.push(entry);
            if (Number.isFinite(ts) && (group.timestampMs <= 0 || ts < group.timestampMs)) {
                group.timestampMs = ts;
            }
            const effective = this.getEffectiveTradeResult(entry, includeProvisional);
            if (effective === 'win') {
                group.result = 'win';
            } else if (effective === 'lose' && group.result !== 'win') {
                group.result = 'lose';
            }
            if (entry.cooldownTriggeredStage) {
                group.cooldownAlreadyTriggered = true;
            }
        }
        const groups = Array.from(logicalGroups.values())
            .sort((a, b) => a.timestampMs - b.timestampMs);
        let consecutive = 0;
        const triggerCandidates: TradeLogEntry[] = [];
        for (const group of groups) {
            if (group.result === 'win') {
                consecutive = 0;
                continue;
            }
            if (group.result !== 'lose') {
                continue;
            }
            consecutive += 1;
            if (group.cooldownAlreadyTriggered && consecutive >= cooldownAfterLossesNum) {
                // A historical trigger marker is only an authoritative boundary
                // if this logical market really reaches the configured loss
                // threshold. Older backfills could mark an earlier split-fill
                // market too soon; treating that as a boundary lets the true
                // third loss leak through the next bar.
                consecutive = 0;
                continue;
            }
            if (cooldownBarsNum > 0 && consecutive >= cooldownAfterLossesNum) {
                const ordered = group.entries
                    .slice()
                    .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime());
                triggerCandidates.push(ordered[0]);
                consecutive = 0;
            }
        }
        return { consecutive, triggerCandidates };
    }

    private refreshCooldownLossStateFromHistory(): void {
        const formal = this.computeCooldownLossState(false);
        const provisional = this.computeCooldownLossState(true);
        this.state.formalConsecutiveLosses = formal.consecutive;
        this.state.provisionalConsecutiveLosses = provisional.consecutive;
        this.state.consecutiveLosses = Math.max(formal.consecutive, provisional.consecutive);
    }

    private async prepareLiveCooldownStateBeforeOrder(): Promise<void> {
        if (this.env.TRADING_MODE !== TradingMode.LIVE) return;

        const startedAt = Date.now();
        console.log('  [LIVE 下单前] 快速刷新账本/冷却状态...');
        // 下单前不能同步扫所有旧 pending 订单：V2 cutover 后历史 pending 很多，
        // 逐笔 getOrder 会把当前 15m 窗口拖到过期。这里先用本地账本重锚冷却；
        // 官方订单/成交深对账放后台补齐。
        if (!this.livePreOrderPendingRefreshRunning) {
            this.livePreOrderPendingRefreshRunning = true;
            void (async () => {
                try {
                    const pendingReconciliation = await this.reconcilePendingLiveTrades();
                    if (pendingReconciliation.reconciledCount > 0 || pendingReconciliation.pendingAfter > 0) {
                        console.log(
                            `  [LIVE 下单前] 后台 pending 对账完成: pending_reconciled=${pendingReconciliation.reconciledCount}, `
                            + `pending_after=${pendingReconciliation.pendingAfter}`,
                        );
                    }
                } catch (error) {
                    console.log(`  [LIVE 下单前] 后台 pending 对账失败: ${error instanceof Error ? error.message : String(error)}`);
                } finally {
                    this.livePreOrderPendingRefreshRunning = false;
                }
            })();
        }
        this.refreshCooldownLossStateFromHistory();
        await this.settleHistory({ skipReportRefresh: true });
        await this.reconcileProvisionalCooldownState();
        this.repairLogicalCooldownMetadataFromLedger();
        this.refreshCooldownLossStateFromHistory();

        const nowMs = Date.now();
        const deepReconcileMinGapMs = 60_000;
        if (
            nowMs - this.livePreOrderDeepReconcileLastStartedAt >= deepReconcileMinGapMs
            && !this.liveFollowUpReconciliationRunning
        ) {
            this.livePreOrderDeepReconcileLastStartedAt = nowMs;
            console.log('  [LIVE 下单前] 深度官网 activity 对账转后台执行，不阻塞本轮下单');
            void this.runLiveStateReconciliation('pre_order_background').catch((error) => {
                console.log(`  [LIVE 下单前] 后台深度对账失败: ${error instanceof Error ? error.message : String(error)}`);
            });
        }
        console.log(`  [LIVE 下单前] 快速冷却状态刷新完成 (${Date.now() - startedAt}ms)`);
    }

    private async reconcileLiveStateAfterOrder(context: string): Promise<void> {
        if (this.env.TRADING_MODE !== TradingMode.LIVE) return;
        // Public activity can lag the order response by a few seconds. A short
        // post-order reconciliation prevents filled orders from lingering as
        // fake pending state until the next report cycle.
        await new Promise((resolve) => setTimeout(resolve, 1500));
        await this.runLiveStateReconciliation(`${context}:post_1_5s`);
        this.scheduleLiveStateFollowUpReconciliations(context);
    }

    private scheduleLiveStateFollowUpReconciliations(context: string): void {
        if (this.env.TRADING_MODE !== TradingMode.LIVE) return;
        for (const delayMs of [8000, 25000, 60000]) {
            const timer = setTimeout(() => {
                void this.runLiveStateReconciliation(`${context}:post_${Math.round(delayMs / 1000)}s`);
            }, delayMs);
            const maybeUnref = (timer as any).unref;
            if (typeof maybeUnref === 'function') maybeUnref.call(timer);
        }
    }

    private async runLiveStateReconciliation(context: string): Promise<void> {
        if (this.env.TRADING_MODE !== TradingMode.LIVE) return;
        if (this.liveFollowUpReconciliationRunning) return;
        this.liveFollowUpReconciliationRunning = true;
        try {
        const pendingReconciliation = await this.reconcilePendingLiveTrades();
        const executedReconciliation = await this.reconcileExecutedLiveTradesWithPublicActivity();
        const ledgerRepair = this.rebuildLivePendingOrderLedgerFromTrades();
        this.refreshCooldownLossStateFromHistory();
        await this.reconcileProvisionalCooldownState();
        this.repairLogicalCooldownMetadataFromLedger();
        this.refreshCooldownLossStateFromHistory();
        const changed = pendingReconciliation.reconciledCount > 0
            || executedReconciliation.reconciledCount > 0
            || ledgerRepair.rewritten;
        if (changed) {
            console.log(
                `  🔄 LIVE 下单后成交回填(${context}): pending_reconciled=${pendingReconciliation.reconciledCount}, `
                + `executed_reconciled=${executedReconciliation.reconciledCount}, pending_after=${pendingReconciliation.pendingAfter}, `
                + `ledger_events=${ledgerRepair.eventCount}`,
            );
        }
        } finally {
            this.liveFollowUpReconciliationRunning = false;
        }
    }

    private formalCooldownWindowExpired(entry: TradeLogEntry, cooldownBarsNum: number): boolean {
        const marketStartTs = parsePeriodStartFromSlug(entry.marketSlug);
        if (marketStartTs == null) return false;
        const nowSec = Math.floor(Date.now() / 1000);
        const cooldownWindowEndTs = marketStartTs + this.periodSeconds * (1 + Math.max(0, cooldownBarsNum));
        return nowSec >= cooldownWindowEndTs;
    }

    private async reconcileProvisionalCooldownState(): Promise<{ evaluated: number; activated: number }> {
        const comboName = this.getComboDisplayName();
        let evaluated = 0;
        let activated = 0;
        const unresolvedExecutableEntries = this.logger
            .getAllEntries()
            .filter((entry) => this.isCooldownEligibleEntry(entry))
            .filter((entry) => (entry.marketSlug || entry.conditionId))
            .filter((entry) => entry.result !== 'win' && entry.result !== 'lose')
            .filter((entry) => entry.provisionalResult !== 'win' && entry.provisionalResult !== 'lose')
            .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

        const groups = new Map<string, TradeLogEntry[]>();
        for (const entry of unresolvedExecutableEntries) {
            const key = this.buildCooldownLogicalMarketKey(entry);
            if (!groups.has(key)) groups.set(key, []);
            groups.get(key)!.push(entry);
        }

        for (const groupEntries of groups.values()) {
            const probe = groupEntries.find((entry) => entry.conditionId || entry.marketSlug);
            if (!probe) continue;

            const marketStartTs = parsePeriodStartFromSlug(probe.marketSlug);
            const nowSec = Math.floor(Date.now() / 1000);
            if (marketStartTs != null && nowSec < marketStartTs + this.periodSeconds) {
                continue;
            }

            let marketResult: Awaited<ReturnType<typeof getMarketResultByConditionId>>;
            if (probe.conditionId) {
                marketResult = await getMarketResultByConditionId(probe.conditionId, probe.marketSlug);
                if (!marketResult.resolved && probe.marketSlug) {
                    marketResult = await getMarketResult(probe.marketSlug);
                }
            } else {
                marketResult = await getMarketResult(probe.marketSlug!);
            }
            if (!marketResult?.resolved || marketResult.winner == null) {
                continue;
            }

            const nowIso = new Date().toISOString();
            for (const trade of groupEntries) {
                const tokenOutcome = trade.tokenOutcome || (trade.direction === 'UP' ? 'Up' : 'Down');
                const provisionalResult: 'win' | 'lose' = marketResult.winner === tokenOutcome ? 'win' : 'lose';
                const current = this.logger.getEntry(trade.id);
                if (!current) continue;
                if (current.provisionalResult !== provisionalResult || !current.provisionalEvaluatedAt) {
                    this.logger.updateEntry(trade.id, {
                        provisionalResult,
                        provisionalEvaluatedAt: nowIso,
                    });
                }
                evaluated += 1;
            }
            this.updateLogicalCooldownMarketEntries(probe, {
                provisionalResult: marketResult.winner === (probe.tokenOutcome || (probe.direction === 'UP' ? 'Up' : 'Down')) ? 'win' : 'lose',
                provisionalEvaluatedAt: nowIso,
            });
        }

        const provisionalState = this.computeCooldownLossState(true);
        const cooldownBarsNum = this.getCooldownBarsNum();
        const cooldownAfterLossesNum = this.getCooldownAfterLossesNum();
        for (const entry of provisionalState.triggerCandidates) {
            if (entry.result === 'win') continue;
            if (entry.result === 'lose') {
                if (this.formalCooldownWindowExpired(entry, cooldownBarsNum)) {
                this.logger.updateEntry(entry.id, {
                    cooldownTriggeredStage: 'formal_expired_backfill',
                    cooldownScope: this.getCooldownScope(),
                    cooldownKey: this.buildCooldownKey(entry.symbol, entry.direction),
                    formalConsecutiveLosses: cooldownAfterLossesNum,
                    provisionalConsecutiveLosses: this.state.provisionalConsecutiveLosses,
                });
                this.updateLogicalCooldownMarketEntries(entry, {
                    cooldownTriggeredStage: 'formal_expired_backfill',
                    cooldownScope: this.getCooldownScope(),
                    cooldownKey: this.buildCooldownKey(entry.symbol, entry.direction),
                    formalConsecutiveLosses: cooldownAfterLossesNum,
                    provisionalConsecutiveLosses: this.state.provisionalConsecutiveLosses,
                    formalResult: 'lose',
                    provisionalResult: entry.provisionalResult === 'win' ? 'win' : 'lose',
                });
                const label = this.formatCooldownLabel(entry.symbol, entry.direction);
                console.log(`  🧊 [${comboName}] 历史正式亏损冷却补标记: ${label}（窗口已过，不对当前再触发冷却）`);
                continue;
            }
            this.setCooldownRemaining(entry.symbol, entry.direction, cooldownBarsNum);
                this.logger.updateEntry(entry.id, {
                    cooldownTriggeredStage: 'formal',
                    cooldownScope: this.getCooldownScope(),
                    cooldownKey: this.buildCooldownKey(entry.symbol, entry.direction),
                    formalConsecutiveLosses: cooldownAfterLossesNum,
                    provisionalConsecutiveLosses: this.state.provisionalConsecutiveLosses,
                });
                this.updateLogicalCooldownMarketEntries(entry, {
                    cooldownTriggeredStage: 'formal',
                    cooldownScope: this.getCooldownScope(),
                    cooldownKey: this.buildCooldownKey(entry.symbol, entry.direction),
                    formalConsecutiveLosses: cooldownAfterLossesNum,
                    provisionalConsecutiveLosses: this.state.provisionalConsecutiveLosses,
                    formalResult: 'lose',
                    provisionalResult: entry.provisionalResult === 'win' ? 'win' : 'lose',
                });
                activated += 1;
                const label = this.formatCooldownLabel(entry.symbol, entry.direction);
                console.log(`  🧊 [${comboName}] 历史正式亏损冷却补触发: ${label}（已结算连输达到 ${cooldownAfterLossesNum} 次，但原结算路径未写入冷却）跳过 ${cooldownBarsNum} 个周期 (${cooldownBarsNum * 15}分钟)`);
                continue;
            }
            if (entry.provisionalResult !== 'lose') continue;
            this.setCooldownRemaining(entry.symbol, entry.direction, cooldownBarsNum);
            this.logger.updateEntry(entry.id, {
                cooldownTriggeredStage: 'provisional',
                cooldownScope: this.getCooldownScope(),
                cooldownKey: this.buildCooldownKey(entry.symbol, entry.direction),
                provisionalEvaluatedAt: new Date().toISOString(),
                provisionalConsecutiveLosses: cooldownAfterLossesNum,
                formalConsecutiveLosses: this.state.formalConsecutiveLosses,
            });
            this.updateLogicalCooldownMarketEntries(entry, {
                cooldownTriggeredStage: 'provisional',
                cooldownScope: this.getCooldownScope(),
                cooldownKey: this.buildCooldownKey(entry.symbol, entry.direction),
                provisionalEvaluatedAt: new Date().toISOString(),
                provisionalConsecutiveLosses: cooldownAfterLossesNum,
                formalConsecutiveLosses: this.state.formalConsecutiveLosses,
                provisionalResult: 'lose',
            });
            activated += 1;
            const label = this.formatCooldownLabel(entry.symbol, entry.direction);
            console.log(`  🧊 [${comboName}] 预结算冷却已激活: ${label}（市场已结束且确认连续亏损达到 ${cooldownAfterLossesNum} 次）跳过 ${cooldownBarsNum} 个周期 (${cooldownBarsNum * 15}分钟)`);
        }
        this.repairLogicalCooldownMetadataFromLedger();
        this.refreshCooldownLossStateFromHistory();
        return { evaluated, activated };
    }

    public getCooldownSnapshot(): Array<{
        key: string;
        symbol: string;
        direction: string | null;
        remainingBars: number;
        scope: 'symbol' | 'symbol_direction';
    }> {
        const scope = this.getCooldownScope();
        const out: Array<{
            key: string;
            symbol: string;
            direction: string | null;
            remainingBars: number;
            scope: 'symbol' | 'symbol_direction';
        }> = [];
        for (const [key, raw] of this.cooldownByCoinDir.entries()) {
            const remaining = Math.max(0, Number(raw) || 0);
            if (remaining <= 0) continue;
            let symbol = key;
            let direction: string | null = null;
            if (scope === 'symbol_direction') {
                const parts = key.split('::');
                symbol = parts[0] || key;
                direction = parts[1] || null;
            }
            out.push({ key, symbol, direction, remainingBars: remaining, scope });
        }
        return out;
    }

    public getCurrentSymbolCapital(symbol: string): number {
        return this.getCoinCapital(symbol);
    }

    // ─── Per-Coin Capital Helpers ────────────────────────────────

    /** 获取某币种可用资金；未启用隔离时返回总资金 */
    private getCoinCapital(symbol?: string): number {
        if (!this.perCoinEnabled || !symbol) return this.state.currentCapital;
        return this.capitalByCoin.get(symbol) ?? 0;
    }

    /** 调整某币种资金（delta 可正可负），自动同步总资金 */
    private adjustCoinCapital(symbol: string, delta: number): void {
        if (this.perCoinEnabled) {
            const old = this.capitalByCoin.get(symbol) ?? 0;
            this.capitalByCoin.set(symbol, roundUsd(old + delta));
            this._syncTotalCapital();
        } else {
            this.state.currentCapital = roundUsd(this.state.currentCapital + delta);
        }
    }

    /** 将某币种资金设置为绝对值，自动同步总资金 */
    private setCoinCapital(symbol: string, value: number): void {
        if (this.perCoinEnabled) {
            this.capitalByCoin.set(symbol, roundUsd(value));
            this._syncTotalCapital();
        } else {
            this.state.currentCapital = roundUsd(value);
        }
    }

    /** 同步 state.currentCapital = sum(capitalByCoin) */
    private _syncTotalCapital(): void {
        let total = 0;
        for (const v of this.capitalByCoin.values()) total += v;
        this.state.currentCapital = roundUsd(total);
    }

    /** 获取某币种的峰值资金；未启用隔离时返回总峰值 */
    private getCoinPeakCapital(symbol?: string): number {
        if (!this.perCoinEnabled || !symbol) return this.state.peakCapital;
        return this.peakCapitalByCoin.get(symbol) ?? 0;
    }

    /** 若某币种当前资金 > 其峰值，则更新峰值；自动同步总峰值 */
    private updateCoinPeakIfNeeded(symbol: string): void {
        if (this.perCoinEnabled) {
            const cap = this.capitalByCoin.get(symbol) ?? 0;
            const peak = this.peakCapitalByCoin.get(symbol) ?? 0;
            if (cap > peak) this.peakCapitalByCoin.set(symbol, roundUsd(cap));
            let totalPeak = 0;
            for (const v of this.peakCapitalByCoin.values()) totalPeak += v;
            this.state.peakCapital = roundUsd(totalPeak);
        } else {
            if (this.state.currentCapital > this.state.peakCapital) {
                this.state.peakCapital = roundUsd(this.state.currentCapital);
            }
        }
    }

    /** 获取某币种的回撤比例（用于下注比例调整） */
    private getCoinDrawdown(symbol?: string): number {
        const cap = this.getCoinCapital(symbol);
        const peak = this.getCoinPeakCapital(symbol);
        return peak > 0 ? (peak - cap) / peak : 0;
    }

    /** 显示 per-coin 资金明细（初始化/结算后打印） */
    private printPerCoinCapital(): void {
        if (!this.perCoinEnabled) return;
        const parts: string[] = [];
        for (const [sym, cap] of this.capitalByCoin) {
            const peak = this.peakCapitalByCoin.get(sym) ?? cap;
            const dd = peak > 0 ? ((peak - cap) / peak * 100).toFixed(1) : '0.0';
            parts.push(`${sym}=$${cap.toFixed(2)}(peak=$${peak.toFixed(2)},DD=${dd}%)`);
        }
        console.log(`  💰 [按币种] ${parts.join(' | ')}`);
    }

    /**
     * 初始化执行器
     */
    async initialize(): Promise<boolean> {
        console.log('\n🚀 正在初始化预测执行器...\n');
        
        // 打印配置
        printPredictionConfig(this.env, this.logsDir);
        
        // 检查预测数据源（仅警告，不阻止启动）
        console.log('🔌 正在检查预测数据源...');
        const apiHealthy = await checkPredictionApiHealth();
        
        if (!apiHealthy) {
            console.log('⚠️  预测数据当前不可用（可能已过期或尚未生成）');
            console.log('   系统将继续启动，等待下一次预测生成');
            console.log('   目标文件: ' + getPredictionFilePath());
            console.log('   请确保 Python 预测写入器正在运行:');
            console.log('   默认: python -m src.python.prediction_writer');
            if (process.env.PREDICTION_SUFFIX === '_A') {
                console.log('   A 套: python -m src.python.prediction_writer --models-dir data/models_A --output polymarket/predictions_A.json');
            }
            console.log('');
        } else {
            console.log('✅ 预测数据源可用\n');
        }
        
        // 对于真实交易模式，初始化 CLOB 客户端与下单提交器（单钱包；多钱包后续可替换 orderPoster）
        if (this.env.TRADING_MODE === TradingMode.LIVE) {
            try {
                console.log('🔑 正在初始化 Polymarket CLOB 客户端...');
                this.clobClient = await this.createClobClient();
                this.orderPoster = this.clobClient ? new SingleWalletOrderPoster(this.clobClient) : null;
                this.startClobHeartbeat();
                console.log('✅ CLOB 客户端初始化成功\n');
            } catch (error) {
                console.error('❌ CLOB 客户端初始化失败:', parseErrorReason(error));
                console.error(
                    '  live_init_context=',
                    JSON.stringify({
                        proxyWalletPresent: Boolean(String(this.env.PROXY_WALLET || '').trim()),
                        privateKeyPresent: Boolean(String(this.env.PRIVATE_KEY || '').trim()),
                        polyApiCredsPresent: Boolean(
                            String(this.env.POLY_API_KEY || '').trim()
                            && String(this.env.POLY_API_SECRET || '').trim()
                            && String(this.env.POLY_API_PASSPHRASE || '').trim(),
                        ),
                        builderCredsPresent: Boolean(
                            String(this.env.BUILDER_API_KEY || '').trim()
                            && String(this.env.BUILDER_API_SECRET || '').trim()
                            && String(this.env.BUILDER_API_PASSPHRASE || '').trim(),
                        ),
                        rpcUrlPresent: Boolean(String(this.env.RPC_URL || '').trim()),
                        clobHttpUrl: this.env.CLOB_HTTP_URL,
                    }),
                );
                if (error instanceof Error && error.stack) {
                    console.error(error.stack);
                } else {
                    console.error(error);
                }
                return false;
            }
        } else if (this.env.TRADING_MODE === TradingMode.SIMULATION) {
            // 模拟模式：尝试初始化 CLOB 客户端以获取实时订单簿（最大程度仿真实交易）
            // 如果失败，会回退到价格 API（公开接口，不需要认证）
            try {
                console.log('🔑 正在初始化 CLOB 客户端（模拟模式，用于获取实时价格）...');
                this.clobClient = await this.createClobClient();
                this.orderPoster = this.clobClient ? new SingleWalletOrderPoster(this.clobClient) : null;
                console.log('✅ CLOB 客户端初始化成功（将使用实时订单簿价格）\n');
            } catch (error) {
                // 模拟模式下 CLOB 客户端初始化失败不影响运行，会使用价格 API
                console.log(`ℹ️  模拟模式 - CLOB 客户端初始化失败，将使用价格 API 获取实时价格\n`);
            }
        } else {
            console.log(`ℹ️  ${getTradingModeText(this.env.TRADING_MODE)} - 无需 CLOB 客户端\n`);
        }
        
        // 预加载市场缓存
        console.log(`🔍 正在搜索 Polymarket ${this.env.TIMEFRAME} 市场...`);
        const markets = await findAllMarkets(this.env.TIMEFRAME, undefined, this.env.ALLOWED_MARKETS);
        for (const result of markets) {
            this.state.marketCache.set(result.symbol, result);
        }
        printMarketResults(markets);
        
        // 加载今日交易数量
        this.state.todayTradeCount = this.logger.getTodayTradeCount();
        console.log(`📊 今日已交易: ${this.state.todayTradeCount}/${this.env.MAX_TRADES_PER_SESSION === 0 ? '不限' : this.env.MAX_TRADES_PER_SESSION} 笔\n`);
        
        // 先恢复未到期的模拟挂单，再从交易历史恢复资金。
        // 这样重启后 pending/GTC 冻结资金仍会占用，避免挂单丢失后资金虚增。
        if (this.env.TRADING_MODE !== TradingMode.LIVE) {
            this.loadPendingSimOrders();
        }

        // 从交易历史恢复 currentCapital（进程重启后避免重复加「赢」、漏算「输」）
        if (this.env.TRADING_MODE !== TradingMode.LIVE) {
            this._recomputeCapitalFromHistory();
            if (this.logger.getAllEntries().length === 0) {
                console.log(`  💵 无历史记录，从初始资金开始: $${this.state.currentCapital.toFixed(2)}\n`);
            }
            if (this.env.PROXY_WALLET && this.env.PROXY_WALLET.length > 10) {
                console.log(`  ℹ️  PROXY_WALLET 已配置 (${this.env.PROXY_WALLET.slice(0, 10)}...)，但当前为【模拟模式】— 不会向 Polymarket 发真实订单。\n`);
            }
            if (this.pendingSimOrders.length > 0) {
                this.startPendingSimMonitor();
            }
        } else if (this.clobClient && this.env.PROXY_WALLET) {
            // LIVE 模式：从链上读取真实 pUSD 余额，用于 5%/2.5% 等下注比例
            await this.refreshLiveCapital();
            console.log(`  💵 实盘资金(链上): $${this.state.currentCapital.toFixed(2)}（每 10 秒同步）\n`);
            // 启动快路径不能等待逐笔官网订单对账。V2 cutover 后历史 pending 很多，
            // 同步 getOrder 会把 worker 卡在初始化，导致自动交易完全不扫描。
            // 这里先用本地账本恢复冷却/资金，官网 pending 深对账后台补齐。
            void (async () => {
                try {
                    const pendingReconciliation = await this.reconcilePendingLiveTrades();
                    if (pendingReconciliation.reconciledCount > 0 || pendingReconciliation.pendingAfter > 0) {
                        console.log(
                            `  [LIVE 启动] 后台 pending 对账完成: pending_reconciled=${pendingReconciliation.reconciledCount}, `
                            + `pending_after=${pendingReconciliation.pendingAfter}`,
                        );
                    }
                } catch (error) {
                    console.log(`  [LIVE 启动] 后台 pending 对账失败: ${error instanceof Error ? error.message : String(error)}`);
                }
            })();
            const ledgerRepair = this.rebuildLivePendingOrderLedgerFromTrades();
            const settlement = await this.settleHistory({ skipReportRefresh: true });
            await this.reconcileProvisionalCooldownState();
            const cooldownRepair = this.repairLogicalCooldownMetadataFromLedger();
            this.refreshCooldownLossStateFromHistory();
            void this.runLiveStateReconciliation('startup_background').catch((error) => {
                console.log(`  [LIVE 启动] 后台深度官网对账失败: ${error instanceof Error ? error.message : String(error)}`);
            });
            if (
                ledgerRepair.rewritten ||
                settlement.settledCount > 0 ||
                cooldownRepair.entriesUpdated > 0
            ) {
                console.log(
                    `  🔄 LIVE 启动快速账本/冷却恢复完成: pending_reconciled=background, `
                    + `executed_reconciled=background, pending_after=background, `
                    + `ledger_events=${ledgerRepair.eventCount}, settled=${settlement.settledCount}, `
                    + `cooldown_groups_touched=${cooldownRepair.groupsTouched}, cooldown_entries_updated=${cooldownRepair.entriesUpdated}`,
                );
            } else {
                console.log(
                    `  🔎 LIVE 启动快速账本/冷却检查: pending_reconciled=background, `
                    + `executed_reconciled=background, pending_after=background, `
                    + `ledger_rewritten=${ledgerRepair.rewritten}, settled=0, cooldown_entries_updated=0`,
                );
            }
        }
        
        // 恢复峰值资金（从历史记录中找到最高点）。
        // 新的非 Kelly 运行时风险缩放也依赖 peakCapital；不能只在旧风控/Kelly 时恢复。
        const runtimeRiskScalingModeForPeak = this.getRuntimeRiskScalingMode();
        const needsRuntimeRiskPeak =
            runtimeRiskScalingModeForPeak === 'cap_to_conservative_above_15dd'
            || runtimeRiskScalingModeForPeak === 'cap_to_directional_conservative_with_hysteresis';
        if (this.env.RISK_CONTROL_ENABLED) {
            this._recomputePeakCapital();
            this._loadHaltState();
            console.log(`🛡️  本组合动态风控已启用 (LOGS_DIR=${this.logsDir})：回撤>20% 下注减半，>10% 减 25%\n`);
        } else if (this.env.KELLY_ENABLED || needsRuntimeRiskPeak) {
            // Kelly 和非 Kelly 的运行时缩放都需要 peakCapital 与持久化状态。
            this._recomputePeakCapital();
            this._loadHaltState();
        }
        
        return true;
    }
    
    /**
     * 根据交易日志重算 currentCapital，避免进程重启后只加「赢」、漏掉「下注扣除」导致资金偏大。
     * 规则：每笔已执行订单当时已扣 amount；若赢则加 amount/tokenPrice；若止损卖出则加 proceeds = amount + pnl。
     * 只计算允许的市场中的交易（ALLOWED_MARKETS）。
     */
    private _recomputeCapitalFromHistory(): void {
        const allowedSymbols = new Set(this.env.ALLOWED_MARKETS);
        const entries = this.logger
            .getAllEntries()
            .filter(e => e.status === 'executed' && allowedSymbols.has(e.symbol))
            .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
        const pendingFrozenBySymbol = new Map<string, number>();
        for (const order of this.pendingSimOrders) {
            const current = pendingFrozenBySymbol.get(order.symbol) ?? 0;
            pendingFrozenBySymbol.set(order.symbol, roundUsd(current + order.frozenAmount));
        }
        const hasAnyPending = this.pendingSimOrders.length > 0;

        /** 处理单笔交易对资金的影响 */
        const applyTrade = (cap: number, e: any): number => {
            cap -= e.amount;
            if (e.result === 'win') {
                if (e.tokenPrice) {
                    const feeR = polymarketTakerFeeRate(e.tokenPrice);
                    cap += (e.amount / e.tokenPrice) * (1 - feeR);
                } else if (e.pnl != null) {
                    cap += e.amount + e.pnl;
                } else {
                    console.warn(`  ⚠️ _recomputeCapitalFromHistory: 交易 ${e.id ?? '?'} 缺少 tokenPrice 和 pnl，使用兜底 buyPrice=0.5 计算`);
                    const feeR = polymarketTakerFeeRate(0.5);
                    cap += (e.amount / 0.5) * (1 - feeR);
                }
            } else if (e.result === 'lose' && e.stoppedOut && e.pnl != null) {
                cap += e.amount + e.pnl;
            }
            return cap;
        };

        if (this.perCoinEnabled) {
            // ─── 按币种独立计算资金 ───
            const numCoins = this.env.ALLOWED_MARKETS.length;
            const perCoinInit = this.env.INITIAL_CAPITAL / numCoins;
            for (const sym of this.env.ALLOWED_MARKETS) {
                let coinCap = perCoinInit;
                const coinEntries = entries.filter(e => e.symbol === sym);
                for (const e of coinEntries) {
                    coinCap = applyTrade(coinCap, e);
                }
                const pendingFrozen = pendingFrozenBySymbol.get(sym) ?? 0;
                if (pendingFrozen > 0) {
                    coinCap = roundUsd(coinCap - pendingFrozen);
                }
                // report 兜底（按币种）
                const reportCoinCap = this._tryLoadCoinCapitalFromReport(sym, perCoinInit);
                // 历史交易存在时，以交易日志重算为准。旧 report 可能因为重启时机
                // 或未结算持仓而放大 currentCapital，不能反过来覆盖交易真相。
                if (reportCoinCap !== null && coinEntries.length === 0 && pendingFrozen <= 0) {
                    coinCap = reportCoinCap;
                }
                this.capitalByCoin.set(sym, roundUsd(Math.max(0, coinCap)));
            }
            this._syncTotalCapital();
            console.log(`  💵 从历史交易恢复资金(当前): $${this.state.currentCapital.toFixed(2)} (初始: $${this.env.INITIAL_CAPITAL.toFixed(2)}, 按币种隔离)`);
            this.printPerCoinCapital();
        } else {
            // ─── 原有逻辑：总资金 ───
            let cap = this.env.INITIAL_CAPITAL;
            for (const e of entries) {
                cap = applyTrade(cap, e);
            }
            const totalPendingFrozen = roundUsd(this.pendingSimOrders.reduce((sum, order) => sum + order.frozenAmount, 0));
            if (totalPendingFrozen > 0) {
                cap = roundUsd(cap - totalPendingFrozen);
            }
            const reportCap = this._tryLoadCapitalFromReport(allowedSymbols);
            // 与 per-coin 口径一致：有历史交易时只信交易日志，不让旧 report 覆盖。
            if (reportCap !== null && entries.length === 0 && !hasAnyPending) {
                cap = reportCap;
                console.log(`  💵 trades 为空，从 ${path.basename(this.getReportSummaryModePath())} 恢复资金: $${cap.toFixed(2)}`);
            }
            if (cap <= 0) {
                console.log(`\n⚠️  警告: 资金已归零 (计算值: $${cap.toFixed(2)})，将暂停交易（与优化器 capital<=0 break 一致）`);
            }
            this.state.currentCapital = roundUsd(Math.max(0, cap));
            console.log(`  💵 从历史交易恢复资金(当前): $${this.state.currentCapital.toFixed(2)} (初始: $${this.env.INITIAL_CAPITAL.toFixed(2)})\n`);
        }
    }

    private getReportSummaryModePath(mode: TradeLogModeScope = this.modeScope): string {
        return path.join(path.dirname(this.logger.getLogFilePath()), 'reports', `report_summary.${mode}.json`);
    }

    private readReportSummaryForCurrentMode(): any | null {
        const modePath = this.getReportSummaryModePath();
        try {
            if (fs.existsSync(modePath)) {
                return JSON.parse(fs.readFileSync(modePath, 'utf-8'));
            }
        } catch {
            // ignore and fallback to null
        }
        return null;
    }

    /** 从 report_summary.<mode>.json 的 bySymbol 计算 allowed markets 的总资金 */
    private _tryLoadCapitalFromReport(allowedSymbols: Set<string>): number | null {
        try {
            const data = this.readReportSummaryForCurrentMode();
            if (!data) return null;
            const bySymbol = data?.bySymbol;
            if (!bySymbol || typeof bySymbol !== 'object') return null;
            let totalPnl = 0;
            allowedSymbols.forEach((sym: string) => {
                const symData = bySymbol[sym];
                if (symData && typeof symData.pnl === 'number') {
                    totalPnl += symData.pnl;
                }
            });
            return this.env.INITIAL_CAPITAL + totalPnl;
        } catch {
            return null;
        }
    }

    /** 从 report_summary.<mode>.json 读取单币种资金（per-coin 模式兜底） */
    private _tryLoadCoinCapitalFromReport(symbol: string, perCoinInit: number): number | null {
        try {
            const data = this.readReportSummaryForCurrentMode();
            if (!data) return null;
            const symData = data?.bySymbol?.[symbol];
            if (!symData || typeof symData.pnl !== 'number') return null;
            return perCoinInit + symData.pnl;
        } catch {
            return null;
        }
    }
    
    /**
     * 从交易历史恢复峰值资金和连续亏损计数
     * 只计算允许的市场中的交易（ALLOWED_MARKETS）。
     */
    private _recomputePeakCapital(): void {
        const allowedSymbols = new Set(this.env.ALLOWED_MARKETS);
        const entries = this.logger.getAllEntries()
            .filter(e => e.status === 'executed' && allowedSymbols.has(e.symbol))
            .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

        /** 处理单笔交易对资金的影响 */
        const applyTrade = (cap: number, e: any): number => {
            cap -= e.amount;
            if (e.result === 'win') {
                if (e.tokenPrice) {
                    const feeR = polymarketTakerFeeRate(e.tokenPrice);
                    cap += (e.amount / e.tokenPrice) * (1 - feeR);
                } else if (e.pnl != null) {
                    cap += e.amount + e.pnl;
                } else {
                    console.warn(`  ⚠️ _recomputePeakCapital: 交易 ${e.id ?? '?'} 缺少 tokenPrice 和 pnl，使用兜底 buyPrice=0.5 计算`);
                    const feeR = polymarketTakerFeeRate(0.5);
                    cap += (e.amount / 0.5) * (1 - feeR);
                }
            } else if (e.result === 'lose') {
                if (e.stoppedOut && e.pnl != null) cap += e.amount + e.pnl;
            }
            return cap;
        };

        const baseCapital = (
            this.env.TRADING_MODE === TradingMode.LIVE && entries.length === 0
                ? this.state.currentCapital
                : this.env.INITIAL_CAPITAL
        );

        if (this.perCoinEnabled) {
            const numCoins = this.env.ALLOWED_MARKETS.length;
            const perCoinInit = baseCapital / numCoins;
            let totalPeak = 0;
            for (const sym of this.env.ALLOWED_MARKETS) {
                let coinCap = perCoinInit;
                let coinPeak = perCoinInit;
                const coinEntries = entries.filter(e => e.symbol === sym);
                for (const e of coinEntries) {
                    coinCap = applyTrade(coinCap, e);
                    if (coinCap > coinPeak) coinPeak = coinCap;
                }
                this.peakCapitalByCoin.set(sym, roundUsd(Math.max(coinPeak, perCoinInit)));
                totalPeak += this.peakCapitalByCoin.get(sym)!;
            }
            this.state.peakCapital = roundUsd(totalPeak);
        } else {
            let cap = baseCapital;
            let peak = cap;
            for (const e of entries) {
                cap = applyTrade(cap, e);
                if (cap > peak) peak = cap;
            }
            this.state.peakCapital = roundUsd(Math.max(peak, baseCapital));
        }

        this.refreshCooldownLossStateFromHistory();
        console.log(`  📈 恢复峰值资金(历史最高): $${this.state.peakCapital.toFixed(2)}, 连续亏损: ${this.state.consecutiveLosses}\n`);
        if (this.perCoinEnabled) this.printPerCoinCapital();
    }

    private _saveHaltState(): void {
        try {
            const statePath = path.join(this.logsDirFull, 'halt_state.json');
            const state = {
                drawdownHaltStartTime: this.state.drawdownHaltStartTime,
                runtimeDirectionalConservativeBySymbol: Object.fromEntries(this.runtimeDirectionalConservativeBySymbol.entries()),
                cooldownByCoinDir: Object.fromEntries(this.cooldownByCoinDir.entries()),
                cooldownByCoin: Object.fromEntries(this.cooldownByCoin.entries()),
                savedAt: Date.now(),
            };
            const tmp = statePath + '.tmp';
            fs.writeFileSync(tmp, JSON.stringify(state, null, 2));
            fs.renameSync(tmp, statePath);
        } catch (e) {
            // 静默失败，不影响交易流程
        }
    }

    private _persistHistoricalHaltRecord(startedAt: number, reason: string): void {
        try {
            const statePath = path.join(this.logsDirFull, 'halt_state.json');
            const state = {
                drawdownHaltStartTime: null,
                historicalDrawdownHaltStartTime: startedAt,
                historicalDrawdownHaltClearedAt: Date.now(),
                historicalDrawdownHaltClearReason: reason,
                runtimeDirectionalConservativeBySymbol: Object.fromEntries(this.runtimeDirectionalConservativeBySymbol.entries()),
                cooldownByCoinDir: Object.fromEntries(this.cooldownByCoinDir.entries()),
                cooldownByCoin: Object.fromEntries(this.cooldownByCoin.entries()),
                savedAt: Date.now(),
            };
            const tmp = statePath + '.tmp';
            fs.writeFileSync(tmp, JSON.stringify(state, null, 2));
            fs.renameSync(tmp, statePath);
        } catch {
            // 静默失败，不影响交易流程
        }
    }

    private _hasPostHaltDataPlaneEvidence(startedAt: number): boolean {
        const startedIsoMs = startedAt;
        const hasTsAfter = (raw: unknown): boolean => {
            const text = String(raw ?? '').trim();
            if (!text) return false;
            const normalized = text.endsWith('Z') ? text : text.replace(' ', 'T');
            const ts = Date.parse(normalized);
            return Number.isFinite(ts) && ts >= startedIsoMs;
        };

        try {
            const summaryPath = path.join(this.logsDirFull, 'reports', 'report_summary.live.json');
            if (fs.existsSync(summaryPath)) {
                const summary = JSON.parse(fs.readFileSync(summaryPath, 'utf-8'));
                const latestAttemptAt = summary?.summary?.lastAttemptAt ?? summary?.lastAttemptAt;
                const latestAttemptResult = String(summary?.summary?.lastAttemptResult ?? summary?.lastAttemptResult ?? '').trim();
                if (latestAttemptResult && hasTsAfter(latestAttemptAt)) return true;
            }
        } catch {
            // ignore
        }

        try {
            const tradePath = path.join(this.logsDirFull, 'prediction_trades.live.json');
            if (fs.existsSync(tradePath)) {
                const rows = JSON.parse(fs.readFileSync(tradePath, 'utf-8'));
                if (Array.isArray(rows)) {
                    for (let i = rows.length - 1; i >= 0; i -= 1) {
                        const row = rows[i] ?? {};
                        if (hasTsAfter(row.timestamp ?? row.createdAt ?? row.updatedAt)) return true;
                    }
                }
            }
        } catch {
            // ignore
        }

        try {
            const eventPath = path.join(this.logsDirFull, 'execution_v2_events.live.jsonl');
            if (fs.existsSync(eventPath)) {
                const lines = fs.readFileSync(eventPath, 'utf-8').trim().split('\n');
                for (let i = lines.length - 1; i >= 0; i -= 1) {
                    const line = lines[i]?.trim();
                    if (!line) continue;
                    try {
                        const row = JSON.parse(line);
                        if (hasTsAfter(row.timestamp ?? row.createdAt)) return true;
                    } catch {
                        continue;
                    }
                }
            }
        } catch {
            // ignore
        }

        return false;
    }

    private _loadHaltState(): void {
        try {
            const statePath = path.join(this.logsDirFull, 'halt_state.json');
            if (fs.existsSync(statePath)) {
                const data = JSON.parse(fs.readFileSync(statePath, 'utf-8'));
                const runtimeDirectionalState = data.runtimeDirectionalConservativeBySymbol;
                if (runtimeDirectionalState && typeof runtimeDirectionalState === 'object' && !Array.isArray(runtimeDirectionalState)) {
                    for (const [symRaw, activeRaw] of Object.entries(runtimeDirectionalState as Record<string, unknown>)) {
                        const sym = normalizeSymbol(symRaw);
                        if (!sym) continue;
                        this.runtimeDirectionalConservativeBySymbol.set(sym, Boolean(activeRaw));
                    }
                }
                const cooldownState = data.cooldownByCoinDir;
                if (cooldownState && typeof cooldownState === 'object' && !Array.isArray(cooldownState)) {
                    for (const [keyRaw, valueRaw] of Object.entries(cooldownState as Record<string, unknown>)) {
                        const key = String(keyRaw || '').trim();
                        const value = Math.max(0, Math.floor(Number(valueRaw) || 0));
                        if (key && value > 0) {
                            this.cooldownByCoinDir.set(key, value);
                        }
                    }
                }
                const legacyCooldownState = data.cooldownByCoin;
                if (legacyCooldownState && typeof legacyCooldownState === 'object' && !Array.isArray(legacyCooldownState)) {
                    for (const [symRaw, valueRaw] of Object.entries(legacyCooldownState as Record<string, unknown>)) {
                        const sym = normalizeSymbol(symRaw);
                        const value = Math.max(0, Math.floor(Number(valueRaw) || 0));
                        if (sym && value > 0) {
                            this.cooldownByCoin.set(sym, value);
                        }
                    }
                }
                const startedAt = Number(data.drawdownHaltStartTime || 0);
                if (startedAt > 0) {
                    const now = Date.now();
                    const untilTs = startedAt + this.getDrawdownHaltDurationMs();
                    if (now < untilTs) {
                        if (this._hasPostHaltDataPlaneEvidence(startedAt)) {
                            this.state.drawdownHaltStartTime = null;
                            if (this.state.pauseReason === 'drawdown_halt') {
                                this.state.tradingPaused = false;
                                this.state.pauseReason = null;
                            }
                            this._persistHistoricalHaltRecord(startedAt, 'stale_file_post_halt_activity');
                            console.log(`  ✅ 检测到 halt 之后已有真实 trade/execution 证据，清除陈旧回撤熔断记录`);
                            return;
                        }
                        this.state.drawdownHaltStartTime = startedAt;
                        this.state.tradingPaused = true;
                        this.state.pauseReason = 'drawdown_halt';
                        const remainMin = Math.max(0, Math.ceil((untilTs - now) / 60000));
                        console.log(`  🛑 恢复回撤熔断状态，剩余约 ${remainMin} 分钟`);
                    } else {
                        this.clearDrawdownHaltState(true);
                        console.log(`  ✅ 已清除过期的回撤熔断状态，并按当前资金重置峰值`);
                    }
                }
            }
        } catch (e) {
            // 静默失败
        }
    }

    private getRedemptionLogPath(): string {
        return path.join(this.logsDirFull, 'redemption_log.jsonl');
    }

    private normalizeClaimConditionId(conditionId: string | null | undefined): string {
        const raw = String(conditionId || '').trim().toLowerCase();
        if (!raw) return '';
        return raw.startsWith('0x') ? raw : `0x${raw}`;
    }

    private appendRedemptionLog(entry: Record<string, unknown>): void {
        try {
            const logPath = this.getRedemptionLogPath();
            fs.appendFileSync(logPath, JSON.stringify({ timestamp: new Date().toISOString(), ...entry }) + '\n');
        } catch {
            // 日志失败不影响主流程
        }
    }

    private loadClaimedSuccessCache(): void {
        try {
            const logPath = this.getRedemptionLogPath();
            if (!fs.existsSync(logPath)) return;
            const lines = fs.readFileSync(logPath, 'utf-8').split('\n');
            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed) continue;
                try {
                    const row = JSON.parse(trimmed);
                    const conditionId = this.normalizeClaimConditionId(row?.conditionId ? String(row.conditionId) : '');
                    const status = String(row?.status || '').trim().toLowerCase();
                    if (!conditionId) continue;
                    if (status.startsWith('success')) {
                        this.claimedSuccessConditionIds.add(conditionId);
                    }
                } catch {
                    // ignore malformed line
                }
            }
        } catch {
            // ignore
        }
    }

    private hasSuccessfulClaim(conditionId: string): boolean {
        const normalized = this.normalizeClaimConditionId(conditionId);
        return normalized ? this.claimedSuccessConditionIds.has(normalized) : false;
    }

    private markClaimSuccess(conditionId: string): void {
        const normalized = this.normalizeClaimConditionId(conditionId);
        if (normalized) {
            this.claimedSuccessConditionIds.add(normalized);
        }
    }

    private getSignalOutcomesPath(): string {
        return path.join(this.logsDirFull, 'signal_outcomes.jsonl');
    }

    private buildSignalOutcomeKey(
        conditionId: string | undefined,
        marketSlug: string | undefined,
        symbol: string,
        mode: string,
    ): string {
        if (conditionId) return `${conditionId}|${symbol}|${mode}`;
        if (marketSlug) return `${marketSlug}|${symbol}|${mode}`;
        return `${symbol}|${mode}|unknown`;
    }

    private loadSignalOutcomeKeysIfNeeded(): void {
        if (this.signalOutcomeKeysLoaded) return;
        this.signalOutcomeKeysLoaded = true;
        try {
            const fp = this.getSignalOutcomesPath();
            if (!fs.existsSync(fp)) return;
            const lines = fs.readFileSync(fp, 'utf-8').split('\n');
            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed) continue;
                try {
                    const row = JSON.parse(trimmed);
                    const key = this.buildSignalOutcomeKey(
                        row?.conditionId ? String(row.conditionId) : undefined,
                        row?.marketSlug ? String(row.marketSlug) : undefined,
                        String(row?.symbol || ''),
                        String(row?.mode || ''),
                    );
                    this.signalOutcomeKeys.add(key);
                } catch {
                    // ignore malformed line
                }
            }
        } catch {
            // ignore
        }
    }

    private appendSignalOutcome(trade: TradeLogEntry, won: boolean): void {
        try {
            this.loadSignalOutcomeKeysIfNeeded();
            const key = this.buildSignalOutcomeKey(trade.conditionId, trade.marketSlug, trade.symbol, trade.mode);
            if (this.signalOutcomeKeys.has(key)) return;

            const payload = {
                timestamp: new Date().toISOString(),
                conditionId: trade.conditionId || null,
                marketSlug: trade.marketSlug || null,
                symbol: trade.symbol,
                direction: trade.direction,
                isCorrect: won,
                mode: trade.mode,
                logsDir: this.logsDir,
                comboName: this.getComboDisplayName(),
            };
            fs.appendFileSync(this.getSignalOutcomesPath(), JSON.stringify(payload) + '\n');
            this.signalOutcomeKeys.add(key);
        } catch {
            // ignore
        }
    }
    
    /**
     * LIVE 模式：从 CLOB 读取钱包真实 pUSD 余额并更新 currentCapital，用于正确计算 5%/2.5% 等下注比例。
     * 外部 claim 后余额会更新，轮询即可与模拟交易一致。
     */
    async refreshLiveCapital(): Promise<void> {
        if (this.env.TRADING_MODE !== TradingMode.LIVE || !this.clobClient || !this.env.PROXY_WALLET) {
            return;
        }
        try {
            const res = await this.getLiveCollateralBalanceAllowance();
            const bal = res.balanceRaw != null ? String(res.balanceRaw) : '';
            if (!bal) return;
            const balanceUsd = parseInt(bal, 10) / 1e6;
            if (!Number.isFinite(balanceUsd) || balanceUsd < 0) return;
            this.state.currentCapital = roundUsd(balanceUsd);
            if (balanceUsd > this.state.peakCapital) {
                this.state.peakCapital = roundUsd(balanceUsd);
            }
            // PER_COIN_CAPITAL 模式：按 allowed_markets 均分钱包余额
            if (this.perCoinEnabled && this.env.ALLOWED_MARKETS.length > 0) {
                const perCoin = roundUsd(balanceUsd / this.env.ALLOWED_MARKETS.length);
                for (const sym of this.env.ALLOWED_MARKETS) {
                    this.capitalByCoin.set(sym, perCoin);
                    const prevPeak = this.peakCapitalByCoin.get(sym) ?? 0;
                    if (perCoin > prevPeak) {
                        this.peakCapitalByCoin.set(sym, perCoin);
                    }
                }
            }
        } catch {
            // 静默失败，避免轮询刷屏；下次 10 秒后再试
        }
    }

    private extractExchangeAllowanceRaw(balanceAllowance: any): string | null {
        if (balanceAllowance?.allowance != null) {
            return String(balanceAllowance.allowance).trim() || '0';
        }
        const allowances = balanceAllowance?.allowances;
        if (!allowances || typeof allowances !== 'object') return null;
        for (const [spender, value] of Object.entries(allowances as Record<string, unknown>)) {
            if (String(spender || '').trim().toLowerCase() === POLYMARKET_EXCHANGE_SPENDER) {
                return String(value ?? '').trim() || '0';
            }
        }
        return null;
    }

    private formatKnownExchangeAllowances(balanceAllowance: any): string {
        if (balanceAllowance?.allowance != null) {
            return `v2_allowance=${String(balanceAllowance.allowance)}`;
        }
        const allowances = balanceAllowance?.allowances;
        if (!allowances || typeof allowances !== 'object') return 'allowances_unavailable';
        const bySpender = allowances as Record<string, unknown>;
        const get = (spender: string): string => {
            const direct = bySpender[spender];
            if (direct != null) return String(direct);
            for (const [key, value] of Object.entries(bySpender)) {
                if (String(key).toLowerCase() === spender) return String(value);
            }
            return 'missing';
        };
        return `v2=${get(POLYMARKET_CTF_EXCHANGE_V2)}, legacy_v1=${get(POLYMARKET_CTF_EXCHANGE_V1_LEGACY)}`;
    }

    private async readOnchainPusdBalanceAllowance(): Promise<{ balanceRaw: string | null; allowanceRaw: string | null; source: string; error?: string }> {
        if (!this.env.RPC_URL || !this.env.PROXY_WALLET) {
            return { balanceRaw: null, allowanceRaw: null, source: 'onchain_unavailable', error: 'missing RPC_URL or PROXY_WALLET' };
        }
        try {
            const provider = new ethers.providers.JsonRpcProvider(this.env.RPC_URL);
            const erc20 = new ethers.Contract(POLYMARKET_PUSD_COLLATERAL, ERC20_BALANCE_ALLOWANCE_ABI, provider);
            const [balance, allowance] = await Promise.all([
                erc20.balanceOf(this.env.PROXY_WALLET),
                erc20.allowance(this.env.PROXY_WALLET, POLYMARKET_CTF_EXCHANGE_V2),
            ]);
            return {
                balanceRaw: balance?.toString?.() ?? null,
                allowanceRaw: allowance?.toString?.() ?? null,
                source: 'onchain_pusd',
            };
        } catch (error) {
            return { balanceRaw: null, allowanceRaw: null, source: 'onchain_error', error: extractErrorText(error) };
        }
    }

    private async getLiveCollateralBalanceAllowance(): Promise<{
        balanceRaw: string | null;
        allowanceRaw: string | null;
        balanceSource: string;
        allowanceSource: string;
        knownAllowances: string;
        clobRaw?: any;
        onchain?: { balanceRaw: string | null; allowanceRaw: string | null; source: string; error?: string };
    }> {
        const balanceAllowance = await this.clobClient!.getBalanceAllowance({ asset_type: AssetType.COLLATERAL });
        const clobBalanceRaw = balanceAllowance?.balance != null ? String(balanceAllowance.balance) : null;
        const clobAllowanceRaw = this.extractExchangeAllowanceRaw(balanceAllowance);
        const knownAllowances = this.formatKnownExchangeAllowances(balanceAllowance);
        const clobBalancePositive = !!clobBalanceRaw && /^\d+$/.test(clobBalanceRaw) && BigInt(clobBalanceRaw) > BigInt(0);
        const clobAllowancePositive = !!clobAllowanceRaw && /^\d+$/.test(clobAllowanceRaw) && BigInt(clobAllowanceRaw) > BigInt(0);
        if (clobBalancePositive && clobAllowancePositive) {
            return {
                balanceRaw: clobBalanceRaw,
                allowanceRaw: clobAllowanceRaw,
                balanceSource: 'clob_getBalanceAllowance',
                allowanceSource: 'clob_getBalanceAllowance',
                knownAllowances,
                clobRaw: balanceAllowance,
            };
        }

        // V2 proxy wallets can trade successfully while the CLOB balance-allowance endpoint
        // returns zero/legacy-shaped data. Onchain pUSD is the authoritative fallback.
        const onchain = await this.readOnchainPusdBalanceAllowance();
        const onchainBalancePositive = !!onchain.balanceRaw && /^\d+$/.test(onchain.balanceRaw) && BigInt(onchain.balanceRaw) > BigInt(0);
        const onchainAllowancePositive = !!onchain.allowanceRaw && /^\d+$/.test(onchain.allowanceRaw) && BigInt(onchain.allowanceRaw) > BigInt(0);
        return {
            balanceRaw: onchainBalancePositive ? onchain.balanceRaw : clobBalanceRaw,
            allowanceRaw: onchainAllowancePositive ? onchain.allowanceRaw : clobAllowanceRaw,
            balanceSource: onchainBalancePositive ? 'onchain_pusd_fallback' : 'clob_getBalanceAllowance',
            allowanceSource: onchainAllowancePositive ? 'onchain_pusd_fallback' : 'clob_getBalanceAllowance',
            knownAllowances: `${knownAllowances}; onchain_balance=${onchain.balanceRaw ?? 'missing'}; onchain_allowance=${onchain.allowanceRaw ?? 'missing'}${onchain.error ? `; onchain_error=${onchain.error}` : ''}`,
            clobRaw: balanceAllowance,
            onchain,
        };
    }

    private async preflightLiveOrder(requiredAmountUsd: number): Promise<LiveOrderPreflightResult> {
        if (this.env.TRADING_MODE !== TradingMode.LIVE || !this.clobClient || !this.env.PROXY_WALLET) {
            return { ok: false, status: 'order_submit_failure', reason: 'LIVE 预检缺少 clobClient 或 PROXY_WALLET' };
        }
        try {
            const balanceAllowance = await this.getLiveCollateralBalanceAllowance();
            const balanceRaw = balanceAllowance.balanceRaw != null ? String(balanceAllowance.balanceRaw) : '';
            const balanceUsd = balanceRaw ? parseInt(balanceRaw, 10) / 1e6 : NaN;
            const allowanceRaw = balanceAllowance.allowanceRaw;
            const knownAllowances = balanceAllowance.knownAllowances;
            const allowanceUnits = allowanceRaw ? Number(allowanceRaw) / 1e6 : 0;
            const requiredAllowanceRaw = BigInt(Math.max(0, Math.ceil(requiredAmountUsd * 1e6)));
            const allowanceRawValue = allowanceRaw && /^\d+$/.test(allowanceRaw) ? BigInt(allowanceRaw) : BigInt(0);
            if (!Number.isFinite(balanceUsd)) {
                return { ok: false, status: 'order_submit_failure', reason: '余额查询返回无效值', allowanceRaw };
            }
            if (!Number.isFinite(allowanceUnits) || allowanceUnits <= 0) {
                return {
                    ok: false,
                    status: 'allowance_zero',
                    reason: `pUSD V2 交易合约授权不足 (spender=${POLYMARKET_CTF_EXCHANGE_V2}, allowance=${allowanceRaw || '0'}, balanceSource=${balanceAllowance.balanceSource}, allowanceSource=${balanceAllowance.allowanceSource}, ${knownAllowances})`,
                    balanceUsd,
                    allowanceRaw,
                };
            }
            if (allowanceRawValue < requiredAllowanceRaw) {
                return {
                    ok: false,
                    status: 'allowance_zero',
                    reason: `pUSD V2 交易合约授权不足 (spender=${POLYMARKET_CTF_EXCHANGE_V2}, allowance=${allowanceRaw || '0'} < required=${requiredAllowanceRaw.toString()}, balanceSource=${balanceAllowance.balanceSource}, allowanceSource=${balanceAllowance.allowanceSource}, ${knownAllowances})`,
                    balanceUsd,
                    allowanceRaw,
                };
            }
            if (balanceUsd + 1e-9 < requiredAmountUsd) {
                return {
                    ok: false,
                    status: 'balance_insufficient',
                    reason: `余额不足 (balance=${balanceUsd.toFixed(6)} < required=${requiredAmountUsd.toFixed(6)})`,
                    balanceUsd,
                    allowanceRaw,
                };
            }
            return {
                ok: true,
                status: 'ok',
                reason: `live_preflight_ok balance=${balanceUsd.toFixed(6)} spender=${POLYMARKET_CTF_EXCHANGE_V2} allowance=${allowanceRaw} balanceSource=${balanceAllowance.balanceSource} allowanceSource=${balanceAllowance.allowanceSource}`,
                balanceUsd,
                allowanceRaw,
            };
        } catch (error) {
            const classified = classifyLiveSubmitIssue(error);
            return {
                ok: false,
                status: classified.status,
                reason: classified.reason,
            };
        }
    }

    private shouldRetryLiveSubmitIssue(status: LiveOrderPreflightStatus): boolean {
        return status === 'gateway_502' || status === 'order_submit_failure';
    }

    private extractPostedOrderId(resp: any): string | null {
        const candidates = [
            resp?.orderID,
            resp?.orderId,
            resp?.order_id,
            resp?.id,
            resp?.order?.id,
            resp?.order?.orderID,
            resp?.order?.orderId,
            resp?.data?.orderID,
            resp?.data?.orderId,
            resp?.data?.order_id,
            resp?.data?.id,
        ];
        for (const candidate of candidates) {
            const text = String(candidate ?? '').trim();
            if (text) return text;
        }
        return null;
    }

    private summarizePostOrderResponse(resp: any): string {
        try {
            return JSON.stringify(resp).slice(0, 500);
        } catch {
            return String(resp ?? '').slice(0, 500);
        }
    }

    private extractOfficialMatchedSize(order: any): number {
        const candidates = [
            order?.size_matched,
            order?.sizeMatched,
            order?.matched_size,
            order?.matchedSize,
            order?.filled,
            order?.filled_size,
            order?.filledSize,
            order?.matched_amount,
            order?.matchedAmount,
        ];
        for (const candidate of candidates) {
            const value = Number(candidate);
            if (Number.isFinite(value)) return value;
        }
        return 0;
    }

    private classifyOfficialOrderState(order: any): { accepted: boolean; status: string; matchedSize: number; reason?: string } {
        const status = String(order?.status ?? order?.rawStatus ?? '').trim().toUpperCase() || 'UNKNOWN';
        const matchedSize = this.extractOfficialMatchedSize(order);
        if (['CANCELED', 'CANCELLED', 'EXPIRED', 'REJECTED', 'FAILED'].includes(status) && matchedSize <= 0) {
            return {
                accepted: false,
                status,
                matchedSize,
                reason: `official_${status.toLowerCase()}_zero_fill`,
            };
        }
        return { accepted: true, status, matchedSize };
    }

    private extractOfficialOrderPrice(order: any, fallbackPrice: number): number {
        const candidates = [
            order?.price,
            order?.order?.price,
            order?.limit_price,
            order?.limitPrice,
            order?.original_price,
            order?.originalPrice,
            fallbackPrice,
        ];
        for (const candidate of candidates) {
            const value = Number(candidate);
            if (Number.isFinite(value) && value > 0) return value;
        }
        return fallbackPrice;
    }

    private buildOfficialImmediateFill(
        order: any,
        fallbackPrice: number,
        requestedShares: number,
        requestedAmountUsd: number,
    ): { amountUsd: number; tokens: number; price: number; status: 'filled' | 'partially_filled'; source: string } | null {
        if (!order) return null;
        const official = this.classifyOfficialOrderState(order);
        const matchedShares = Number(official.matchedSize || 0);
        if (!Number.isFinite(matchedShares) || matchedShares <= 0) return null;
        const fillPrice = this.extractOfficialOrderPrice(order, fallbackPrice);
        const amountUsd = roundUsd(matchedShares * fillPrice);
        if (!(amountUsd > 0)) return null;
        const isPartial = requestedShares > 0
            ? matchedShares + 0.000001 < requestedShares
            : amountUsd + 0.01 < requestedAmountUsd;
        return {
            amountUsd,
            tokens: matchedShares,
            price: fillPrice,
            status: isPartial ? 'partially_filled' : 'filled',
            source: `official_get_order_${String(official.status || 'matched').toLowerCase()}`,
        };
    }

    private isOfficialTerminalZeroFill(state: OfficialOrderLifecycleState): boolean {
        const status = String(state.status || '').toUpperCase();
        return state.matchedSize <= 0 && ['CANCELED', 'CANCELLED', 'EXPIRED', 'REJECTED', 'FAILED'].includes(status);
    }

    private isOfficialTerminalWithFill(state: OfficialOrderLifecycleState): boolean {
        const status = String(state.status || '').toUpperCase();
        return state.matchedSize > 0 && ['CANCELED', 'CANCELLED', 'EXPIRED', 'MATCHED', 'FILLED'].includes(status);
    }

    private async queryOfficialOrderLifecycle(orderId: string): Promise<OfficialOrderLifecycleState | null> {
        if (!this.clobClient || typeof this.clobClient.getOrder !== 'function') return null;
        try {
            const order = await this.clobClient.getOrder(orderId);
            if (!order || (order as any)?.error) return null;
            const state = this.classifyOfficialOrderState(order as any);
            return {
                status: state.status,
                matchedSize: state.matchedSize,
                raw: order,
                reason: state.reason,
            };
        } catch (error) {
            return {
                status: 'UNKNOWN',
                matchedSize: 0,
                reason: `getOrder_error:${parseErrorReason(error)}`,
            };
        }
    }

    private markOfficialTerminalZeroFill(entry: TradeLogEntry, orderId: string, state: OfficialOrderLifecycleState): void {
        const reason = state.reason || `official_${String(state.status || 'canceled').toLowerCase()}_zero_fill`;
        this.pendingLiveOrders.delete(orderId);
        this.logger.updateEntry(entry.id, {
            status: 'failed',
            orderId,
            txHash: orderId,
            amount: 0,
            pendingAmountUsd: 0,
            matchedShares: 0,
            liveOrderStatus: 'official_canceled_zero_fill',
            fillSource: reason,
            lastNonExecutionReason: reason,
            error: reason,
            stalePolicyDecision: 'official_terminal_zero_fill',
            staleCancelReason: reason,
            familyRuleVersion: this.getFamilyRuleVersion() || entry.familyRuleVersion,
            lowpriceFamilyId: this.getLowpriceFamilyId() || entry.lowpriceFamilyId,
        });
        this.appendPendingOrderLedgerEvent({
            event: 'official_canceled_zero_fill',
            orderId,
            symbol: entry.symbol,
            direction: entry.direction,
            amountUsd: 0,
            remainingFrozenAmount: 0,
            limitPrice: Number(entry.limitPriceConfigured ?? entry.tokenPrice ?? 0),
            tokenId: String(entry.tokenId || ''),
            tokenOutcome: String(entry.tokenOutcome || ''),
            confidence: Number(entry.confidence || 0),
            marketTitle: String(entry.marketTitle || ''),
            conditionId: String(entry.conditionId || ''),
            marketSlug: String(entry.marketSlug || ''),
            targetPeriodEndTs: parsePeriodStartFromSlug(entry.marketSlug) || 0,
            createdAt: String(entry.timestamp || new Date().toISOString()),
            expiresAt: new Date().toISOString(),
            terminalTimestampSource: reason,
        });
    }

    private markOfficialTerminalWithFill(entry: TradeLogEntry, orderId: string, state: OfficialOrderLifecycleState): void {
        const fill = this.buildOfficialImmediateFill(
            state.raw,
            Number(entry.limitPriceConfigured ?? entry.tokenPrice ?? 0),
            Number(entry.requestedShares ?? 0),
            Number(entry.requestedAmountUsd ?? entry.amount ?? 0),
        );
        if (!fill) return;
        this.pendingLiveOrders.delete(orderId);
        const status = String(state.status || '').toUpperCase();
        const canceledWithFill = status === 'CANCELED' || status === 'CANCELLED' || status === 'EXPIRED';
        const ledgerEvent: PendingOrderLedgerEventType = canceledWithFill
            ? 'official_canceled_partial_fill'
            : (fill.status === 'partially_filled' ? 'partial_fill' : 'filled');
        this.logger.updateEntry(entry.id, {
            status: 'executed',
            orderId,
            txHash: orderId,
            amount: fill.amountUsd,
            pendingAmountUsd: 0,
            matchedShares: fill.tokens,
            tokenPrice: fill.price,
            avgActualFillPrice: fill.price,
            liveOrderStatus: canceledWithFill
                ? 'official_canceled_partial_fill'
                : fill.status,
            fillSource: fill.source,
            stalePolicyDecision: 'official_terminal_with_fill',
            familyRuleVersion: this.getFamilyRuleVersion() || entry.familyRuleVersion,
            lowpriceFamilyId: this.getLowpriceFamilyId() || entry.lowpriceFamilyId,
        });
        this.appendPendingOrderLedgerEvent({
            event: ledgerEvent,
            orderId,
            symbol: entry.symbol,
            direction: entry.direction,
            amountUsd: fill.amountUsd,
            remainingFrozenAmount: 0,
            limitPrice: fill.price,
            tokenId: String(entry.tokenId || ''),
            tokenOutcome: String(entry.tokenOutcome || ''),
            confidence: Number(entry.confidence || 0),
            marketTitle: String(entry.marketTitle || ''),
            conditionId: String(entry.conditionId || ''),
            marketSlug: String(entry.marketSlug || ''),
            targetPeriodEndTs: parsePeriodStartFromSlug(entry.marketSlug) || 0,
            createdAt: String(entry.timestamp || new Date().toISOString()),
            expiresAt: new Date().toISOString(),
            avgFillPrice: fill.price,
            terminalTimestampSource: fill.source,
        });
    }

    private getPostOrderImmediateError(resp: any): string | null {
        if (!resp) return 'postOrder 返回空响应';
        if (resp?.success === false) return parseErrorReason(resp);
        const errorText = String(resp?.errorMsg ?? resp?.error ?? resp?.message ?? '').trim();
        if (errorText) return errorText;
        const statusText = String(resp?.status ?? resp?.rawStatus ?? resp?.order?.status ?? '').trim().toLowerCase();
        if (statusText && ['error', 'failed', 'failure', 'rejected', 'reject'].some((bad) => statusText.includes(bad))) {
            return `postOrder status=${statusText}`;
        }
        return null;
    }

    /**
     * V2 postOrder 返回 orderID 不等于官网已收单。必须用官方 getOrder/openOrders 二次确认，
     * 否则会把“本地构造成功/提交异常”误写成真实挂单。
     */
    private async verifyOfficialOrderAcceptance(
        orderId: string,
        tokenId?: string,
        attempts = 4
    ): Promise<{ accepted: boolean; status?: string; source?: string; reason?: string; raw?: any }> {
        let lastReason = 'official_order_not_seen';
        for (let attempt = 0; attempt < attempts; attempt++) {
            if (attempt > 0) {
                await new Promise((resolve) => setTimeout(resolve, 750 * attempt));
            }
            try {
                if (this.clobClient && typeof this.clobClient.getOrder === 'function') {
                    const order = await this.clobClient.getOrder(orderId);
                    const orderAny = order as any;
                    if (orderAny && !orderAny.error) {
                        const state = this.classifyOfficialOrderState(orderAny);
                        if (state.accepted) {
                            return { accepted: true, status: state.status, source: 'getOrder', raw: order };
                        }
                        lastReason = `${state.reason}:status=${state.status}:matched=${state.matchedSize}:${this.summarizePostOrderResponse(order)}`;
                        continue;
                    }
                    lastReason = `getOrder_empty:${this.summarizePostOrderResponse(order)}`;
                }
            } catch (error) {
                lastReason = `getOrder_error:${parseErrorReason(error)}`;
            }

            if (tokenId && this.clobClient && typeof (this.clobClient as any).getOpenOrders === 'function') {
                try {
                    const openOrders = await (this.clobClient as any).getOpenOrders({ asset_id: tokenId });
                    const rows = Array.isArray(openOrders) ? openOrders : [];
                    const match = rows.find((row: any) => String(row?.id ?? row?.orderID ?? row?.orderId ?? '').toLowerCase() === orderId.toLowerCase());
                    if (match) {
                        const status = String(match.status ?? '').trim() || 'LIVE';
                        return { accepted: true, status, source: 'openOrders', raw: match };
                    }
                    lastReason = `openOrders_no_match:${rows.length}`;
                } catch (error) {
                    lastReason = `openOrders_error:${parseErrorReason(error)}`;
                }
            }
        }
        return { accepted: false, reason: lastReason };
    }

    private computeLiveGtdExpiration(targetPeriodEndTs: number): number | null {
        const nowSec = Math.floor(Date.now() / 1000);
        const marketSafeExpiration = Math.floor(targetPeriodEndTs + this.periodSeconds - 60);
        const lateFillCancelMinutes = Math.max(0, Number((this.env as any).LATE_FILL_CANCEL_MINUTES) || 0);
        const lateFillExpiration = lateFillCancelMinutes > 0
            ? nowSec + Math.floor(lateFillCancelMinutes * 60)
            : marketSafeExpiration;
        const desiredExpiration = Math.min(marketSafeExpiration, lateFillExpiration);
        const minAcceptedExpiration = nowSec + 61;
        if (desiredExpiration <= minAcceptedExpiration) return null;
        return desiredExpiration;
    }

    private buildLiveGtdExpiredReason(targetPeriodEndTs: number): string {
        const nowSec = Math.floor(Date.now() / 1000);
        const marketSafeExpiration = Math.floor(targetPeriodEndTs + this.periodSeconds - 60);
        const lateFillCancelMinutes = Math.max(0, Number((this.env as any).LATE_FILL_CANCEL_MINUTES) || 0);
        const lateFillExpiration = lateFillCancelMinutes > 0
            ? nowSec + Math.floor(lateFillCancelMinutes * 60)
            : marketSafeExpiration;
        const desiredExpiration = Math.min(marketSafeExpiration, lateFillExpiration);
        return `GTD 过期窗口不足: expiration=${desiredExpiration}, now=${nowSec}, lateFillCancelMinutes=${lateFillCancelMinutes}`;
    }
    
    /**
     * LIVE 模式：赎回已结算市场的胜出 token，将 winning token 兑换为 pUSD。
     * CTF 合约: 0x4D97DCd97eC945f40cF65F87097ACe5EA0476045 (Polygon)
     * 调用 redeemPositions(pUSD, bytes32(0), conditionId, [1, 2])
     */
    async redeemWinningTokens(conditionId: string): Promise<boolean> {
        if (this.env.TRADING_MODE !== TradingMode.LIVE || !this.env.PROXY_WALLET || !this.env.PRIVATE_KEY) {
            return false;
        }
        const normalizedConditionId = this.normalizeClaimConditionId(conditionId);
        if (!normalizedConditionId) {
            return false;
        }
        if (this.hasSuccessfulClaim(normalizedConditionId)) {
            return true;
        }
        if (this.claimInFlightConditionIds.has(normalizedConditionId)) {
            return true;
        }
        this.claimInFlightConditionIds.add(normalizedConditionId);
        const CTF_ADDRESS = '0x4D97DCd97eC945f40cF65F87097ACe5EA0476045';
        const PUSD_ADDRESS = '0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB';
        const REDEEM_ABI = ['function redeemPositions(address collateralToken, bytes32 parentCollectionId, bytes32 conditionId, uint256[] indexSets)'];
        const ZERO_BYTES32 = '0x' + '0'.repeat(64);
        const condBytes32 = normalizedConditionId;

        try {
            if (this.env.CLAIM_GASLESS_ENABLED) {
                const gasless = await redeemPositionsGasless(condBytes32, this.env);
                if (gasless.ok) {
                    console.log(`  ✅ Gasless 赎回成功${gasless.txHash ? `: ${gasless.txHash}` : ''}`);
                    this.markClaimSuccess(normalizedConditionId);
                    this.appendRedemptionLog({ conditionId: normalizedConditionId, txHash: gasless.txHash || '', status: 'success_gasless' });
                    await this.refreshLiveCapital();
                    return true;
                }

                const err = gasless.error || 'unknown gasless redeem error';
                if (this.env.CLAIM_GASLESS_STRICT) {
                    console.error(`  ❌ Gasless 赎回失败(strict, 不回退链上) (${normalizedConditionId.slice(0, 18)}...): ${err}`);
                    this.appendRedemptionLog({ conditionId: normalizedConditionId, status: 'failed_gasless_strict', error: err });
                    return false;
                }
                if (!this.env.CLAIM_ALLOW_ONCHAIN_FALLBACK) {
                    console.error(`  ❌ Gasless 赎回失败且未允许链上回退 (${normalizedConditionId.slice(0, 18)}...): ${err}`);
                    this.appendRedemptionLog({ conditionId: normalizedConditionId, status: 'failed_gasless_no_fallback', error: err });
                    return false;
                }
                console.warn(`  ⚠️ Gasless 赎回失败，回退链上 redeem (${normalizedConditionId.slice(0, 18)}...): ${err}`);
            }

            const provider = new ethers.providers.JsonRpcProvider(this.env.RPC_URL);
            const wallet = new ethers.Wallet(this.env.PRIVATE_KEY, provider);
            const ctf = new ethers.Contract(CTF_ADDRESS, REDEEM_ABI, wallet);

            const tx = await ctf.redeemPositions(PUSD_ADDRESS, ZERO_BYTES32, condBytes32, [1, 2]);
            console.log(`  🔄 赎回交易已提交: ${tx.hash}`);
            const receipt = await tx.wait();
            console.log(`  ✅ 赎回成功 (gas: ${receipt.gasUsed.toString()})`);

            this.markClaimSuccess(normalizedConditionId);
            this.appendRedemptionLog({
                conditionId: normalizedConditionId,
                txHash: tx.hash,
                gasUsed: receipt.gasUsed.toString(),
                status: this.env.CLAIM_GASLESS_ENABLED ? 'success_onchain_fallback' : 'success_onchain',
            });

            await this.refreshLiveCapital();
            return true;
        } catch (e: any) {
            console.error(`  ❌ 赎回失败 (${normalizedConditionId.slice(0, 18)}...): ${e.message || e}`);
            this.appendRedemptionLog({ conditionId: normalizedConditionId, status: 'failed', error: String(e.message || e) });
            return false;
        } finally {
            this.claimInFlightConditionIds.delete(normalizedConditionId);
        }
    }

    /**
     * 对已经结算为 win 但尚未 claim 成功的 live 交易做重试。
     * 说明：settleTrade 后该笔交易不再属于 pending，若本轮 claim 失败，需要独立补偿重试。
     */
    private async retryUnclaimedWinningClaims(claimAttemptedThisRound: Set<string>): Promise<void> {
        if (this.env.TRADING_MODE !== TradingMode.LIVE || !this.env.LIVE_CLAIM_SUBMIT_ENABLED) return;
        const seen = new Set<string>();
        const all = this.logger.getAllEntries();
        for (const trade of all) {
            if (trade.mode !== 'live') continue;
            if (trade.status !== 'executed') continue;
            if (trade.result !== 'win') continue;
            const conditionId = trade.conditionId;
            if (!conditionId) continue;
            if (seen.has(conditionId)) continue;
            seen.add(conditionId);
            if (this.hasSuccessfulClaim(conditionId)) continue;
            if (claimAttemptedThisRound.has(conditionId)) continue;
            claimAttemptedThisRound.add(conditionId);
            try {
                const ok = await this.redeemWinningTokens(conditionId);
                if (!ok) {
                    console.warn(`  ⚠️ claim 补偿重试仍失败 (${conditionId.slice(0, 18)}...)`);
                }
            } catch (e) {
                console.error(`  ⚠️ claim 补偿重试异常 (${conditionId.slice(0, 18)}...): ${e}`);
            }
        }
    }

    /**
     * 创建 CLOB 客户端
     */
    private async createClobClient(): Promise<ClobClient> {
        const withTimeout = async <T>(promise: Promise<T>, ms: number, label: string): Promise<T> => Promise.race([
            promise,
            new Promise<T>((_, reject) => {
                setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
            }),
        ]);
        const normalizeCreds = (candidate: any): { key: string; secret: string; passphrase: string } | null => {
            const key = String(candidate?.key ?? '').trim();
            const secret = String(candidate?.secret ?? '').trim();
            const passphrase = String(candidate?.passphrase ?? '').trim();
            if (!key || !secret || !passphrase) return null;
            return { key, secret, passphrase };
        };
        const chain = Chain.POLYGON;
        const host = this.env.CLOB_HTTP_URL;
        const wallet = new ethers.Wallet(this.env.PRIVATE_KEY);

        // 按官方 proxy wallet / funder 语义区分:
        // - signer 地址 == PROXY_WALLET: 直连 EOA
        // - PROXY_WALLET 是合约: Gnosis Safe
        // - PROXY_WALLET 与 signer 不同且非合约: Polymarket proxy wallet
        const provider = new ethers.providers.JsonRpcProvider(this.env.RPC_URL);
        const signerAddress = wallet.address.toLowerCase();
        const proxyWallet = String(this.env.PROXY_WALLET || '').trim();
        const normalizedProxyWallet = proxyWallet.toLowerCase();
        const hasDistinctProxyWallet = Boolean(proxyWallet) && normalizedProxyWallet !== signerAddress;
        const code = proxyWallet ? await provider.getCode(proxyWallet) : '0x';
        const isGnosisSafe = hasDistinctProxyWallet && code !== '0x';
        const signatureType = hasDistinctProxyWallet
            ? (isGnosisSafe ? SignatureTypeV2.POLY_GNOSIS_SAFE : SignatureTypeV2.POLY_PROXY)
            : SignatureTypeV2.EOA;
        const funder = hasDistinctProxyWallet ? proxyWallet : undefined;

        const clobClient = new ClobClient({
            host,
            chain,
            signer: wallet,
            signatureType,
            funderAddress: funder,
        });
        
        // L2 凭证：未配置时用 L1 的 derive→失败则 create 取 key/secret/passphrase（与 test-clob-auth 一致）
        let creds = normalizeCreds(
            this.env.POLY_API_KEY && this.env.POLY_API_SECRET && this.env.POLY_API_PASSPHRASE
                ? {
                key: this.env.POLY_API_KEY,
                secret: this.env.POLY_API_SECRET,
                passphrase: this.env.POLY_API_PASSPHRASE,
            }
                : null,
        );
        if (!creds) {
            let deriveError: unknown = null;
            let createError: unknown = null;
            try {
                creds = normalizeCreds(await withTimeout(clobClient.deriveApiKey(), 12000, 'deriveApiKey'));
                if (!creds) {
                    deriveError = new Error('deriveApiKey returned incomplete credentials');
                }
            } catch (error) {
                deriveError = error;
            }
            if (!creds) {
                try {
                    creds = normalizeCreds(await withTimeout(clobClient.createApiKey(), 12000, 'createApiKey'));
                    if (!creds) {
                        createError = new Error('createApiKey returned incomplete credentials');
                    }
                } catch (error) {
                    createError = error;
                }
            }
            if (!creds) {
                const reasons = [deriveError, createError]
                    .filter(Boolean)
                    .map((item) => parseErrorReason(item))
                    .filter((item) => String(item || '').trim());
                throw new Error(
                    `L2 credentials unavailable for live CLOB auth: ${reasons.join(' | ') || 'derive/create returned no usable credentials'}. `
                    + 'Provide complete POLY_API_* in keychain for this slot or repair the CLOB derive/create path.',
                );
            }
        }

        // 运行时自动拿到的凭证回写到 env，供 UserChannelWS 启动判断使用
        if (creds) {
            this.env.POLY_API_KEY = creds.key;
            this.env.POLY_API_SECRET = creds.secret;
            this.env.POLY_API_PASSPHRASE = creds.passphrase;
        }
        
        return new ClobClient({
            host,
            chain,
            signer: wallet,
            creds,
            signatureType,
            funderAddress: funder,
        });
    }

    private extractClobHeartbeatId(payload: unknown): string {
        const obj = payload as any;
        const candidates = [
            obj?.heartbeat_id,
            obj?.heartbeatId,
            obj?.data?.heartbeat_id,
            obj?.data?.heartbeatId,
            obj?.error?.heartbeat_id,
            obj?.error?.heartbeatId,
            obj?.error?.data?.heartbeat_id,
            obj?.error?.data?.heartbeatId,
            obj?.response?.data?.heartbeat_id,
            obj?.response?.data?.heartbeatId,
            obj?.response?.data?.data?.heartbeat_id,
            obj?.response?.data?.data?.heartbeatId,
        ];
        for (const candidate of candidates) {
            const text = String(candidate ?? '').trim();
            if (text) return text;
        }
        const raw = extractErrorText(payload);
        const quotedMatch = raw.match(/["']heartbeat_id["']\s*:\s*["']([^"']+)["']/i);
        if (quotedMatch?.[1]) return quotedMatch[1].trim();
        const looseMatch = raw.match(/heartbeat[_-]?id[^A-Za-z0-9-]+([0-9a-f]{8}-[0-9a-f-]{27,})/i);
        if (looseMatch?.[1]) return looseMatch[1].trim();
        return '';
    }

    private isClobHeartbeatRejected(payload: unknown): boolean {
        const obj = payload as any;
        const status = Number(obj?.status || obj?.response?.status || 0);
        const raw = extractErrorText(payload).toLowerCase();
        return (
            status >= 400
            || Boolean(obj?.error)
            || Boolean(obj?.error_msg)
            || raw.includes('invalid heartbeat id')
            || raw.includes('error_msg')
        );
    }

    private isInvalidClobHeartbeatId(error: unknown): boolean {
        const text = extractErrorText(error).toLowerCase();
        return text.includes('invalid heartbeat id') || text.includes('heartbeat_id') || Boolean(this.extractClobHeartbeatId(error));
    }

    private static sleepMs(ms: number): Promise<void> {
        return new Promise((resolve) => setTimeout(resolve, Math.max(0, ms)));
    }

    private async postClobHeartbeatQueued(heartbeatId: string): Promise<string> {
        let release: () => void = () => undefined;
        const previous = PredictionExecutor.clobHeartbeatQueue.catch(() => undefined);
        PredictionExecutor.clobHeartbeatQueue = new Promise<void>((resolve) => {
            release = resolve;
        });
        await previous;
        try {
            const elapsed = Date.now() - PredictionExecutor.clobLastGlobalHeartbeatPostAt;
            const waitMs = PredictionExecutor.CLOB_HEARTBEAT_GLOBAL_GAP_MS - elapsed;
            if (waitMs > 0) {
                await PredictionExecutor.sleepMs(waitMs);
            }
            PredictionExecutor.clobLastGlobalHeartbeatPostAt = Date.now();
            return await this.postClobHeartbeatWithId(heartbeatId);
        } finally {
            release();
        }
    }

    private async postClobHeartbeatWithId(heartbeatId: string): Promise<string> {
        let currentHeartbeatId = heartbeatId;
        let lastError: unknown = null;
        for (let attempt = 1; attempt <= 5; attempt += 1) {
            try {
                const resp: any = await (this.clobClient as any).postHeartbeat(currentHeartbeatId);
                if (resp == null) {
                    throw new Error('CLOB heartbeat returned empty response');
                }
                const serverHeartbeatId = this.extractClobHeartbeatId(resp);
                if (serverHeartbeatId) this.clobHeartbeatId = serverHeartbeatId;
                if (this.isClobHeartbeatRejected(resp)) {
                    const reason = parseErrorReason(resp);
                    if (
                        this.isInvalidClobHeartbeatId(resp)
                        && serverHeartbeatId
                        && serverHeartbeatId !== currentHeartbeatId
                        && attempt < 5
                    ) {
                        currentHeartbeatId = serverHeartbeatId;
                        await PredictionExecutor.sleepMs(750);
                        continue;
                    }
                    throw new Error(`CLOB heartbeat rejected: ${reason}`);
                }
                this.clobLastHeartbeatAt = Date.now();
                this.clobLastHeartbeatError = null;
                return this.clobHeartbeatId || currentHeartbeatId;
            } catch (error) {
                lastError = error;
                const serverHeartbeatId = this.extractClobHeartbeatId(error);
                if (serverHeartbeatId) {
                    this.clobHeartbeatId = serverHeartbeatId;
                    currentHeartbeatId = serverHeartbeatId;
                }
                if (!this.isInvalidClobHeartbeatId(error) || attempt >= 5) {
                    break;
                }
                // V2 heartbeat returns the next usable id in 400 responses. Retry it
                // while holding the global heartbeat queue so the two wallets do not
                // invalidate each other's session cadence.
                await PredictionExecutor.sleepMs(750);
            }
        }
        throw lastError instanceof Error ? lastError : new Error(`CLOB heartbeat rejected: ${parseErrorReason(lastError)}`);
    }

    private async sendClobHeartbeat(resetId = false): Promise<void> {
        if (!this.clobClient || this.env.TRADING_MODE !== TradingMode.LIVE) return;
        if (this.clobHeartbeatInFlight) return this.clobHeartbeatInFlight;
        this.clobHeartbeatInFlight = (async () => {
            const heartbeatId = resetId ? '' : this.clobHeartbeatId;
            try {
                await this.postClobHeartbeatQueued(heartbeatId);
            } catch (error) {
                const reason = parseErrorReason(error);
                this.clobLastHeartbeatError = reason;
                const serverHeartbeatId = this.extractClobHeartbeatId(error);
                if (serverHeartbeatId) this.clobHeartbeatId = serverHeartbeatId;
                throw error;
            } finally {
                this.clobHeartbeatInFlight = null;
            }
        })();
        return this.clobHeartbeatInFlight;
    }

    private isClobHeartbeatEnabled(): boolean {
        // CLOB V2 requires a live heartbeat for resting orders. Official docs say
        // open orders are cancelled when a valid heartbeat is not received in time.
        return true;
    }

    private startClobHeartbeat(): void {
        if (!this.clobClient || this.env.TRADING_MODE !== TradingMode.LIVE) return;
        if (!this.isClobHeartbeatEnabled()) return;
        if (this.clobHeartbeatTimer) return;
        const envAny = this.env as any;
        const walletKey = String(this.env.PROXY_WALLET || envAny.FUNDER_ADDRESS || envAny.TRADER_NAME || '');
        const staggerMs = Array.from(walletKey).reduce((acc, ch) => acc + ch.charCodeAt(0), 0) % 2500;
        const initialTimer = setTimeout(() => {
            // First V2 heartbeat must be sent with an empty id. Reusing a stale id
            // starts the session in a 400 "Invalid Heartbeat ID" loop.
            void this.sendClobHeartbeat(true).catch((error) => {
                console.warn(`  ⚠️ CLOB V2 heartbeat 初始化失败: ${parseErrorReason(error)}`);
            });
        }, staggerMs);
        (initialTimer as any).unref?.();
        this.clobHeartbeatTimer = setInterval(() => {
            void this.sendClobHeartbeat().catch((error) => {
                console.warn(`  ⚠️ CLOB V2 heartbeat 失败: ${parseErrorReason(error)}`);
            });
        }, 5000);
        (this.clobHeartbeatTimer as any).unref?.();
    }

    private async ensureClobHeartbeatFresh(): Promise<void> {
        if (!this.clobClient || this.env.TRADING_MODE !== TradingMode.LIVE) return;
        if (!this.isClobHeartbeatEnabled()) return;
        const ageMs = this.clobLastHeartbeatAt > 0 ? Date.now() - this.clobLastHeartbeatAt : Number.POSITIVE_INFINITY;
        if (ageMs <= 8000) return;
        try {
            await this.sendClobHeartbeat();
        } catch (error) {
            try {
                await this.sendClobHeartbeat(true);
            } catch (retryError) {
                const reason = parseErrorReason(retryError || error);
                this.clobLastHeartbeatError = reason;
                throw new Error(`CLOB V2 heartbeat preflight failed: ${reason}`);
            }
        }
    }
    
    /**
     * 按置信度档位取比例（与 hyperparam_tune 一致：base=PROB_THRESHOLD，档1～4 上界 +B1,+B2,+B3 百分点）
     */
    /** 四舍五入到 4 位小数，避免档位边界因浮点误差选错档 */
    private static roundProb(x: number): number {
        return Math.round(x * 10000) / 10000;
    }

    private evaluateSlot01GentleFilter(
        symbol: string,
        direction: 'UP' | 'DOWN',
        confidence: number,
    ): { blocked: boolean; roundedConfidence: number; confidenceMin: number; confidenceMax: number; directionMode: string } {
        const roundedConfidence = PredictionExecutor.roundProb(Number(confidence));
        const confidenceMin = PredictionExecutor.roundProb(Number(this.env.SLOT01_GENTLE_FILTER_CONFIDENCE_MIN) || 0);
        const confidenceMax = PredictionExecutor.roundProb(Number(this.env.SLOT01_GENTLE_FILTER_CONFIDENCE_MAX) || 0);
        const directionMode = String(this.env.SLOT01_GENTLE_FILTER_DIRECTION || 'all').trim().toLowerCase() || 'all';
        if (!this.env.SLOT01_GENTLE_FILTER_ENABLED) {
            return { blocked: false, roundedConfidence, confidenceMin, confidenceMax, directionMode };
        }
        if (String(symbol || '').trim().toUpperCase() !== 'ETH') {
            return { blocked: false, roundedConfidence, confidenceMin, confidenceMax, directionMode };
        }
        if (!(confidenceMax > confidenceMin)) {
            return { blocked: false, roundedConfidence, confidenceMin, confidenceMax, directionMode };
        }
        const dir = String(direction || '').trim().toLowerCase();
        if (directionMode !== 'all' && directionMode !== dir) {
            return { blocked: false, roundedConfidence, confidenceMin, confidenceMax, directionMode };
        }
        return {
            blocked: roundedConfidence >= confidenceMin && roundedConfidence < confidenceMax,
            roundedConfidence,
            confidenceMin,
            confidenceMax,
            directionMode,
        };
    }

    private getTierPercent(confidence: number): number | null {
        const p1 = this.env.TIER_P1_PCT;
        const p2 = this.env.TIER_P2_PCT;
        const p3 = this.env.TIER_P3_PCT;
        const p4 = this.env.TIER_P4_PCT;
        if (p1 == null || p2 == null || p3 == null || p4 == null) return null;
        const base = this.env.PROB_THRESHOLD ?? 0.5;
        const b1 = (this.env.TIER_B1 ?? 2) / 100;
        const b2 = (this.env.TIER_B2 ?? 4) / 100;
        const b3 = (this.env.TIER_B3 ?? 6) / 100;
        const c1 = PredictionExecutor.roundProb(base + b1);
        const c2 = PredictionExecutor.roundProb(base + b2);
        const c3 = PredictionExecutor.roundProb(base + b3);
        const c = PredictionExecutor.roundProb(confidence);
        if (c <= c1) return p1;
        if (c <= c2) return p2;
        if (c <= c3) return p3;
        return p4;
    }

    private getConfidenceBandScale(confidence: number, symbol?: string, direction?: 'UP' | 'DOWN' | null): number {
        const envCfg = (this.env as PredictionEnvConfig & {
            CONFIDENCE_SCALE_BANDS_BY_SYMBOL?: Record<string, Array<{ min: number; max: number; mult: number }>>;
            CONFIDENCE_SCALE_BANDS_BY_SYMBOL_DIRECTION?: Record<string, Array<{ min: number; max: number; mult: number }>>;
        });
        const bySymbol = envCfg.CONFIDENCE_SCALE_BANDS_BY_SYMBOL;
        const bySymbolDirection = envCfg.CONFIDENCE_SCALE_BANDS_BY_SYMBOL_DIRECTION;
        const key = (symbol || '').trim().toUpperCase();
        const c = PredictionExecutor.roundProb(confidence);
        const keys: string[] = [];
        if (key && direction) keys.push(`${key}_${String(direction).toUpperCase()}`);
        if (key) keys.push(key);
        for (const candidateKey of keys) {
            const bands = (bySymbolDirection && Array.isArray(bySymbolDirection[candidateKey]))
                ? bySymbolDirection[candidateKey]
                : (bySymbol && Array.isArray(bySymbol[candidateKey]) ? bySymbol[candidateKey] : undefined);
            if (!bands) continue;
            const hitBands: Array<{ min: number; max: number; mult: number; width: number }> = [];
            for (const band of bands) {
                const min = PredictionExecutor.roundProb(Number(band.min));
                const max = PredictionExecutor.roundProb(Number(band.max));
                const mult = Number(band.mult);
                if (!Number.isFinite(min) || !Number.isFinite(max) || !Number.isFinite(mult) || !(max > min)) continue;
                if (c >= min && c < max) {
                    hitBands.push({ min, max, mult, width: max - min });
                }
            }
            if (hitBands.length > 0) {
                // 重叠区间时：最窄区间优先；若宽度相同，则更高 min 优先（更高置信细分优先）。
                hitBands.sort((a, b) => {
                    if (Math.abs(a.width - b.width) > 1e-9) return a.width - b.width;
                    if (Math.abs(a.min - b.min) > 1e-9) return b.min - a.min;
                    return b.mult - a.mult;
                });
                return Math.max(0, hitBands[0].mult);
            }
        }
        return 1;
    }

    private buildSizingTrace(
        confidence?: number,
        symbol?: string,
        direction?: 'UP' | 'DOWN' | null,
    ): {
        sizingMode: string;
        bankrollPct: number;
        confidenceBandScale: number;
        effectiveSizingPct: number;
        currentCapital: number;
    } {
        const sizingMode = this.getSizingMode();
        const currentCapital = roundUsd(Math.max(0, this.getCoinCapital(symbol) || 0));
        const bankrollPct = sizingMode === 'symmetric_dynamic_pct'
            ? this.getBankrollPct()
            : this.getDirectionalRuntimeBetPct(symbol, direction);
        const confidenceBandScale = confidence != null
            ? this.getConfidenceBandScale(confidence, symbol, direction)
            : 1;
        const effectiveSizingPct = sizingMode === 'symmetric_dynamic_pct'
            ? Math.max(0, bankrollPct * confidenceBandScale)
            : Math.max(0, bankrollPct);
        return {
            sizingMode,
            bankrollPct,
            confidenceBandScale,
            effectiveSizingPct,
            currentCapital,
        };
    }

    private computeDirectionalEdge(confidence: number, tokenPrice?: number | null): number | null {
        const price = Number(tokenPrice);
        if (!Number.isFinite(price) || !(price > 0) || !(price < 1)) return null;
        const odds = (1.0 - price) / price;
        if (!(odds > 0)) return null;
        return confidence * odds - (1.0 - confidence);
    }

    private resolveBetSizingRule(confidence: number, tokenPrice?: number | null, symbol?: string, direction?: 'UP' | 'DOWN' | null): { mult: number; cap: number | null } {
        const envCfg = (this.env as PredictionEnvConfig & {
            BET_SIZING_RULES_BY_SYMBOL_DIRECTION?: Record<string, Array<{ minProb: number; maxProb: number; minEdge?: number; maxEdge?: number; mult?: number; cap?: number }>>;
        });
        const rulesMap = envCfg.BET_SIZING_RULES_BY_SYMBOL_DIRECTION;
        const key = (symbol || '').trim().toUpperCase();
        if (!rulesMap || !key || !direction) {
            return { mult: 1, cap: null };
        }
        const rules = rulesMap[`${key}_${String(direction).toUpperCase()}`];
        if (!Array.isArray(rules) || rules.length <= 0) {
            return { mult: 1, cap: null };
        }
        const c = PredictionExecutor.roundProb(confidence);
        const edge = this.computeDirectionalEdge(confidence, tokenPrice);
        const hits: Array<{ mult: number; cap: number | null; width: number; minProb: number }> = [];
        for (const rawRule of rules) {
            const minProb = PredictionExecutor.roundProb(Number(rawRule.minProb));
            const maxProb = PredictionExecutor.roundProb(Number(rawRule.maxProb));
            if (!Number.isFinite(minProb) || !Number.isFinite(maxProb) || !(maxProb > minProb)) continue;
            if (!(c >= minProb && c < maxProb)) continue;
            const minEdge = rawRule.minEdge != null ? Number(rawRule.minEdge) : null;
            const maxEdge = rawRule.maxEdge != null ? Number(rawRule.maxEdge) : null;
            if (edge != null) {
                if (Number.isFinite(minEdge as number) && edge < Number(minEdge)) continue;
                if (Number.isFinite(maxEdge as number) && edge >= Number(maxEdge)) continue;
            }
            const mult = Number.isFinite(Number(rawRule.mult)) ? Math.max(0, Number(rawRule.mult)) : 1;
            const cap = Number.isFinite(Number(rawRule.cap)) ? Math.max(0, Number(rawRule.cap)) : null;
            hits.push({ mult, cap, width: maxProb - minProb, minProb });
        }
        if (hits.length <= 0) return { mult: 1, cap: null };
        hits.sort((a, b) => {
            if (Math.abs(a.width - b.width) > 1e-9) return a.width - b.width;
            if (Math.abs(a.minProb - b.minProb) > 1e-9) return b.minProb - a.minProb;
            return (b.cap ?? -1) - (a.cap ?? -1);
        });
        return { mult: hits[0].mult, cap: hits[0].cap };
    }

    private resolveBetTargetCapUsd(confidence?: number, tokenPrice?: number | null, symbol?: string, direction?: 'UP' | 'DOWN' | null): number {
        const fallbackCap = Number.isFinite(Number(this.env.BET_TARGET_CAP_USD))
            ? Math.max(0, Number(this.env.BET_TARGET_CAP_USD))
            : 0;
        if (confidence == null || !symbol || !direction) return fallbackCap;
        const sizingRule = this.resolveBetSizingRule(confidence, tokenPrice, symbol, direction);
        if (Number.isFinite(Number(sizingRule.cap)) && Number(sizingRule.cap) > 0) {
            return Number(sizingRule.cap);
        }
        return fallbackCap;
    }

    /**
     * 计算下注金额（含档位 p1～p4、动态风控调整）
     * @param confidence 当前预测置信度（0-1）；传入且启用档位时按档位取比例，否则用 BET_SIZE_PERCENT
     */
    private _kellyDiagPrinted = false;  // 首次 Kelly 诊断已打印
    /** 逆向选择累计统计（跟踪成交 vs 未成交订单的置信度差异） */
    private _adverseSelectionStats: {
        totalFilled: number; totalUnfilled: number;
        sumConfFilled: number; sumConfUnfilled: number;
    } | null = null;

    private calculateBetAmount(confidence?: number, tokenPrice?: number, symbol?: string, direction?: 'UP' | 'DOWN' | null): number {
        const coinCap = this.getCoinCapital(symbol);
        const coinPeak = this.getCoinPeakCapital(symbol);
        const sizingMode = this.getSizingMode();
        let runtimeScale = Math.min(1, Math.max(0, this.runtimeBetScale));
        const symbolKey = normalizeSymbol(symbol);
        if (symbolKey) {
            const symbolScale = this.runtimeSymbolScaleBySymbol.get(symbolKey);
            if (Number.isFinite(symbolScale as number)) {
                runtimeScale *= Math.min(1, Math.max(0, Number(symbolScale)));
            }
        }
        if (direction === 'DOWN' && symbolKey) {
            const downBlocked = this.runtimeDownRiskBlockedBySymbol.get(symbolKey);
            if (downBlocked) return 0;
            const downScale = this.runtimeDownRiskScaleBySymbol.get(symbolKey);
            if (Number.isFinite(downScale as number)) {
                runtimeScale *= Math.min(1, Math.max(0, Number(downScale)));
            }
        }
        if (direction === 'UP' && symbolKey) {
            const upBlocked = this.runtimeUpRiskBlockedBySymbol.get(symbolKey);
            if (upBlocked) return 0;
            const upScale = this.runtimeUpRiskScaleBySymbol.get(symbolKey);
            if (Number.isFinite(upScale as number)) {
                runtimeScale *= Math.min(1, Math.max(0, Number(upScale)));
            }
        }
        if (direction && symbolKey) {
            const expectancyKey = this.buildRuntimeSymbolDirectionKey(symbolKey, direction);
            const expectancyBlocked = this.runtimeExpectancyBlockedBySymbolDir.get(expectancyKey);
            if (expectancyBlocked) return 0;
            const expectancyScale = this.runtimeExpectancyScaleBySymbolDir.get(expectancyKey);
            if (Number.isFinite(expectancyScale as number)) {
                runtimeScale *= Math.min(1, Math.max(0, Number(expectancyScale)));
            }
            const directionLossScale = this.runtimeDirectionLossScaleBySymbolDir.get(expectancyKey);
            if (Number.isFinite(directionLossScale as number)) {
                runtimeScale *= Math.min(1, Math.max(0, Number(directionLossScale)));
            }
        }
        if (symbolKey) {
            const drawdownAccelerationScale = this.runtimeDrawdownAccelerationScaleBySymbol.get(symbolKey);
            if (Number.isFinite(drawdownAccelerationScale as number)) {
                runtimeScale *= Math.min(1, Math.max(0, Number(drawdownAccelerationScale)));
            }
        }

        if (coinCap <= 0) {
            console.log(`  ⚠️  ${symbol ? symbol + ' ' : ''}资金已归零，暂停交易`);
            return 0;
        }

        const baseDirectionalPercent = sizingMode === 'symmetric_dynamic_pct'
            ? this.getBankrollPct() * 100
            : this.getDirectionalBetPctNormal(symbol, direction) * 100;
        let percent = baseDirectionalPercent > 0 ? baseDirectionalPercent : this.env.BET_SIZE_PERCENT;

        // ─── Kelly + 优化器 tier 乘数 （KELLY_ENABLED=true 时生效）──
        if (this.env.KELLY_ENABLED && confidence != null && tokenPrice != null && tokenPrice > 0 && tokenPrice < 1) {
            const odds = (1.0 - tokenPrice) / tokenPrice;
            // 纯 Kelly 公式（不含费率）：f* = (p*b - q) / b
            const kellyF = odds > 0 ? Math.max(0, (confidence * odds - (1 - confidence)) / odds) : 0;

            // bet_ratio = kelly_f × kelly_frac（与优化器 _simulate_trading_core 一致）
            let betRatio = kellyF * this.env.KELLY_FRAC;

            // 置信度分档乘数（与优化器 tier1_mult/tier2_mult/tier3_mult 一致）
            let tierLabel = '';
            if (confidence < this.env.CONF_TIER1_BOUND) {
                betRatio *= this.env.TIER1_MULT;
                tierLabel = `T1(x${this.env.TIER1_MULT})`;
            } else if (confidence < this.env.CONF_TIER2_BOUND) {
                betRatio *= this.env.TIER2_MULT;
                tierLabel = `T2(x${this.env.TIER2_MULT})`;
            } else {
                betRatio *= this.env.TIER3_MULT;
                tierLabel = `T3(x${this.env.TIER3_MULT})`;
            }
            const bandScale = this.getConfidenceBandScale(confidence, symbol, direction);
            if (bandScale !== 1) {
                betRatio *= bandScale;
                tierLabel += `|Band(x${bandScale})`;
            }
            const sizingRule = this.resolveBetSizingRule(confidence, tokenPrice, symbol, direction);
            if (sizingRule.mult !== 1) {
                betRatio *= sizingRule.mult;
                tierLabel += `|Shape(x${sizingRule.mult})`;
            }

            const betRatioBeforeCap = betRatio;
            // 三阶段仓位（与优化器 _simulate_trading_core 一致）
            const directionalConservativeActive = this.isDirectionalConservativeActive(symbol);
            if (this.state.everReachedCap || directionalConservativeActive) {
                // Phase 3: 曾到过 $60K 后跌回 → 保守仓位
                betRatio = Math.min(betRatio, this.getDirectionalBetPctConservative(symbol, direction));
            } else {
                // Phase 1: 从未到过 $60K → 正常仓位
                betRatio = Math.min(betRatio, this.getDirectionalBetPctNormal(symbol, direction));
            }

            percent = betRatio * 100;  // 转为百分比

            // 首次调用时输出完整 Kelly 诊断
            if (!this._kellyDiagPrinted) {
                this._kellyDiagPrinted = true;
                console.log(`\n  [Kelly诊断] KELLY_ENABLED=true, conf=${confidence.toFixed(4)}, tokenPrice=${tokenPrice}`);
                console.log(`    odds=${odds.toFixed(4)}, kellyF=${kellyF.toFixed(4)}, frac=${this.env.KELLY_FRAC}, ${tierLabel}`);
                console.log(`    betRatio=${betRatioBeforeCap.toFixed(5)} → cap=${betRatio.toFixed(5)} (normal=${this.getDirectionalBetPctNormal(symbol, direction)}, conservative=${this.getDirectionalBetPctConservative(symbol, direction)})`);
                console.log(`    final_percent=${percent.toFixed(3)}%, capital=$${coinCap.toFixed(2)}, amount=$${(coinCap * percent / 100).toFixed(2)}`);
            }
        } else {
            // ─── 旧逻辑：固定百分比 / 档位 ──────────────────────
            if (!this._kellyDiagPrinted && this.env.KELLY_ENABLED) {
                this._kellyDiagPrinted = true;
                console.log(`\n  [Kelly诊断] KELLY_ENABLED=true 但条件不满足: conf=${confidence}, tokenPrice=${tokenPrice}`);
                console.log(`    回退到固定 BET_SIZE_PERCENT=${this.env.BET_SIZE_PERCENT}%`);
            }
            if (confidence != null) {
                if (sizingMode === 'symmetric_dynamic_pct') {
                    const bandScale = this.getConfidenceBandScale(confidence, symbol, direction);
                    if (bandScale !== 1) percent *= bandScale;
                } else {
                    const tierPercent = this.getTierPercent(confidence);
                    if (tierPercent != null) percent = tierPercent;
                    const bandScale = this.getConfidenceBandScale(confidence, symbol, direction);
                    if (bandScale !== 1) percent *= bandScale;
                }
            }
        }

        const runtimeRiskScalingMode = this.getRuntimeRiskScalingMode();
        if (!this.env.KELLY_ENABLED && runtimeRiskScalingMode === 'cap_to_conservative_above_15dd' && coinPeak > 0) {
            const currentDrawdown = coinPeak > 0
                ? (coinPeak - coinCap) / coinPeak
                : 0;
            const runtimeCap = currentDrawdown >= 0.15
                ? this.getDirectionalBetPctConservative(symbol, direction)
                : this.getDirectionalBetPctNormal(symbol, direction);
            percent = Math.min(percent, runtimeCap * 100);
        }
        if (!this.env.KELLY_ENABLED && runtimeRiskScalingMode === 'cap_to_directional_conservative_with_hysteresis' && coinPeak > 0) {
            const runtimeCap = this.isDirectionalConservativeActive(symbol)
                ? this.getDirectionalBetPctConservative(symbol, direction)
                : this.getDirectionalBetPctNormal(symbol, direction);
            percent = Math.min(percent, runtimeCap * 100);
        }

        // 动态调整下注比例（根据回撤情况, 仅旧模型用; Kelly 模式由优化器参数控制）
        if (!this.env.KELLY_ENABLED && runtimeRiskScalingMode === 'legacy' && this.env.RISK_CONTROL_ENABLED && coinPeak > 0) {
            const currentDrawdown = coinPeak > 0
                ? (coinPeak - coinCap) / coinPeak
                : 0;
            if (currentDrawdown > 0.2) {
                percent = percent * 0.5;
                console.log(`\n🛡️  动态风控: 回撤 ${(currentDrawdown * 100).toFixed(1)}% > 20%，下注比例减半 → ${percent.toFixed(2)}%`);
            } else if (currentDrawdown > 0.1) {
                percent = percent * 0.75;
                console.log(`\n🛡️  动态风控: 回撤 ${(currentDrawdown * 100).toFixed(1)}% > 10%，下注比例减 25% → ${percent.toFixed(2)}%`);
            }
        }

        // 流动性封顶: >= $60K → 固定 $3K（与优化器一致）
        if (coinCap >= 60000) {
            if (!this.state.everReachedCap) {
                this.state.everReachedCap = true;
            }
            let largeCapAmount = roundUsd(3000 * runtimeScale);
            const betTargetCapUsd = this.resolveBetTargetCapUsd(confidence, tokenPrice, symbol, direction);
            if (betTargetCapUsd > 0 && largeCapAmount > betTargetCapUsd) {
                largeCapAmount = roundUsd(betTargetCapUsd);
            }
            return largeCapAmount;
        }

        // Kelly 计算出零下注（负 edge）时，不强制最低金额，让调用方跳过
        if (this.env.KELLY_ENABLED && percent <= 0) {
            return 0;
        }

        let amount = coinCap * (percent / 100);
        amount *= runtimeScale;

        if (sizingMode !== 'symmetric_dynamic_pct') {
            const effectiveBetPctCap = coinCap * (
                this.state.everReachedCap || this.isDirectionalConservativeActive(symbol)
                    ? this.getDirectionalBetPctConservative(symbol, direction)
                    : this.getDirectionalBetPctNormal(symbol, direction)
            );
            if (amount > effectiveBetPctCap) amount = effectiveBetPctCap;
        }

        // 不超过可用资金 95%（与优化器一致）
        const capitalRiskUpperLimit = this.getCapitalRiskUpperLimit(coinCap);
        if (amount > capitalRiskUpperLimit) amount = capitalRiskUpperLimit;

        const betTargetCapUsd = this.getDirectionalBetTargetCapUsd(confidence, tokenPrice, symbol, direction);
        if (betTargetCapUsd > 0 && amount > betTargetCapUsd) {
            amount = betTargetCapUsd;
        }

        // 资金不足时不强制最低金额（避免 $0.25 被强制为 $1 超过可用资金）
        // 与优化器一致: if bet_amount < 1.0: continue
        if (amount < MIN_ORDER_SIZE_USD) {
            return 0;
        }

        return roundUsd(amount);
    }
    
    /** 当前 live rollout 策略已全局禁用绝对止损底线。 */
    private isAbsoluteStopPolicyDisabled(): boolean {
        return true;
    }

    /** 按币种时的单币绝对止损线（当前仅为兼容健康/审计字段保留）。 */
    private getPerCoinAbsoluteFloor(): number {
        if (this.isAbsoluteStopPolicyDisabled()) return 0;
        if (!this.perCoinEnabled || !this.env.ALLOWED_MARKETS?.length) return 0;
        const perCoinInit = this.env.INITIAL_CAPITAL / this.env.ALLOWED_MARKETS.length;
        return perCoinInit * 0.30;
    }

    /** 当前策略已禁用绝对止损，因此不再按币种触发永久停止。 */
    private isSymbolAbsoluteStopped(symbol: string): boolean {
        if (this.isAbsoluteStopPolicyDisabled()) return false;
        if (!this.perCoinEnabled) return false;
        const floor = this.getPerCoinAbsoluteFloor();
        return (this.getCoinCapital(symbol) ?? 0) < floor;
    }

    /** 回撤熔断暂停时长：24 小时 */
    private getDrawdownHaltDurationMs(): number {
        return 24 * 60 * 60 * 1000;
    }

    /**
     * 清除回撤熔断状态，并按当前资金重置峰值，避免恢复后立即被旧峰值再次触发。
     */
    private clearDrawdownHaltState(resetPeak: boolean = false): void {
        this.state.drawdownHaltStartTime = null;
        if (this.state.pauseReason === 'drawdown_halt') {
            this.state.tradingPaused = false;
            this.state.pauseReason = null;
        }
        if (resetPeak) {
            if (this.perCoinEnabled) {
                let totalPeak = 0;
                for (const sym of this.env.ALLOWED_MARKETS || []) {
                    const cap = roundUsd(this.getCoinCapital(sym));
                    this.peakCapitalByCoin.set(sym, cap);
                    totalPeak += cap;
                }
                this.state.peakCapital = roundUsd(totalPeak || this.state.currentCapital);
            } else {
                this.state.peakCapital = roundUsd(this.state.currentCapital);
            }
        }
        this._saveHaltState();
    }

    /**
     * 检查风险控制条件
     * @returns { shouldPause: boolean, reason: string | null }
     */
    private checkRiskControls(): { shouldPause: boolean; reason: string | null } {
        if (
            this.env.TRADING_MODE !== TradingMode.LIVE
            && Boolean((this.env as PredictionEnvConfig & { SIMULATION_BLOCKING_GUARDS_DISABLED?: boolean }).SIMULATION_BLOCKING_GUARDS_DISABLED)
        ) {
            if (this.state.drawdownHaltStartTime !== null || this.state.tradingPaused) {
                this.state.drawdownHaltStartTime = null;
                this.state.tradingPaused = false;
                this.state.pauseReason = null;
                this._saveHaltState();
            }
            return { shouldPause: false, reason: null };
        }
        const drawdownHaltEnabled = Boolean(this.env.RISK_CONTROL_ENABLED) && Number(this.env.DRAWDOWN_HALT || 0) > 0;
        // ─── 绝对止损已按当前 rollout 策略全局关闭；其余 drawdown/cooldown 等风控保持不变 ──
        if (!this.isAbsoluteStopPolicyDisabled() && !this.perCoinEnabled) {
            const absoluteFloor = this.env.INITIAL_CAPITAL * 0.30;
            if (this.state.currentCapital < absoluteFloor) {
                if (!this.state.tradingPaused || this.state.pauseReason !== 'absolute_stop') {
                    this.state.tradingPaused = true;
                    this.state.pauseReason = 'absolute_stop';
                    this._saveHaltState();
                    console.log(`\n🛑 绝对止损触发: 资金 $${this.state.currentCapital.toFixed(2)} < 初始资金 30% ($${absoluteFloor.toFixed(2)})，永久暂停交易`);
                }
                return {
                    shouldPause: true,
                    reason: `绝对止损: 资金 $${this.state.currentCapital.toFixed(2)} < $${absoluteFloor.toFixed(2)} (初始${this.env.INITIAL_CAPITAL}的30%)`
                };
            }
        }

        // ─── 回撤熔断：达到阈值后暂停 24 小时，冷却结束后重置峰值恢复 ──
        // 注意：冷却结束后 clearDrawdownHaltState(resetPeak=true) 会把 peak 重置到当前资金。
        // 这里不能再强行抬回 INITIAL_CAPITAL，否则会在恢复后立刻被旧峰值语义再次触发。
        if (drawdownHaltEnabled) {
            const peakCapital = Math.max(this.state.peakCapital, this.state.currentCapital, 0);
            const drawdownNow = peakCapital > 0
                ? (peakCapital - this.state.currentCapital) / peakCapital
                : 0;
            if (this.state.drawdownHaltStartTime !== null) {
                const untilTs = this.state.drawdownHaltStartTime + this.getDrawdownHaltDurationMs();
                const remainMs = untilTs - Date.now();
                if (remainMs > 0) {
                    this.state.tradingPaused = true;
                    this.state.pauseReason = 'drawdown_halt';
                    return {
                        shouldPause: true,
                        reason: `回撤熔断中，剩余约 ${Math.max(1, Math.ceil(remainMs / 60000))} 分钟`,
                    };
                }
                this.clearDrawdownHaltState(true);
                console.log(`\n✅ 回撤熔断冷却结束，已按当前资金 $${this.state.currentCapital.toFixed(2)} 重置峰值并恢复交易`);
            } else if (drawdownNow >= this.env.DRAWDOWN_HALT) {
                this.state.drawdownHaltStartTime = Date.now();
                this.state.tradingPaused = true;
                this.state.pauseReason = 'drawdown_halt';
                this._saveHaltState();
                return {
                    shouldPause: true,
                    reason: `回撤熔断: 当前回撤 ${(drawdownNow * 100).toFixed(1)}% >= ${(this.env.DRAWDOWN_HALT * 100).toFixed(1)}%`,
                };
            }
        } else if (this.state.drawdownHaltStartTime !== null || this.state.pauseReason === 'drawdown_halt') {
            this.clearDrawdownHaltState(true);
            console.log(`\n✅ 回撤熔断未启用，已清除陈旧熔断状态`);
        }

        // ─── 旧版风控（仅 RISK_CONTROL_ENABLED=true 时生效）─────
        if (!this.env.RISK_CONTROL_ENABLED) {
            return { shouldPause: false, reason: null };
        }
        
        // 如果正在暂停中（K 线计数），每根 K 线减 1
        if (this.state.pausedKBarsRemaining > 0) {
            this.state.pausedKBarsRemaining--;
            if (this.state.pausedKBarsRemaining <= 0) {
                // 暂停时间到，恢复交易
                this.state.tradingPaused = false;
                this.state.pauseReason = null;
                console.log(`\n✅ 风险控制: 暂停时间到，交易已恢复`);
                return { shouldPause: false, reason: null };
            }
            // 仍在暂停中
            return { shouldPause: true, reason: this.state.pauseReason || `暂停中，剩余 ${this.state.pausedKBarsRemaining} 根 K 线` };
        }
        
        // 如果已经暂停（但 pausedKBarsRemaining 为 0，可能是其他原因暂停的），检查是否可以恢复
        if (this.state.tradingPaused) {
            return { shouldPause: true, reason: this.state.pauseReason };
        }
        
        // 已去掉的是旧版全局连续亏损暂停；同方向运行时冷却由 multi_prediction_index 注入为硬拦截/轻减仓。
        
        return { shouldPause: false, reason: null };
    }
    
    /**
     * 执行单个预测交易
     */
    private async executePrediction(
        prediction: PredictionResult,
        marketResult: MarketSearchResult,
        opts?: {
            targetPeriodEndTs?: number;
            triggerPrice?: number;
            asksSnapshot?: Array<{ price: string; size: string }>;
            preSignedOrder?: unknown;
            cooldownDecrementedThisRound?: Set<string>;
            minPrice?: number;
            maxPrice?: number;
        }
    ): Promise<ExecutionResult> {
        const symbol = prediction.symbol.replace('/USDT', '');
        const result: ExecutionResult = {
            symbol,
            success: false,
        };
        
        // 检查市场是否存在
        if (!marketResult.market) {
            result.error = marketResult.error || `未找到 ${symbol} 的 ${this.env.TIMEFRAME} 市场`;
            console.log(`\n❌ 跳过 ${symbol}: ${result.error}`);
            return result;
        }

        // 检查该市场是否已下过单（含重启后、或监控队列已成交：不重复买入同一 conditionId）
        if (this.logger.hasAlreadyBought(marketResult.market.conditionId)) {
            result.error = '该市场已由本周期或监控队列成交，本轮跳过（避免重复买入）';
            result.skippedReason = 'already_bought';
            console.log(`\n⏭️ 跳过 ${symbol}: ${result.error}`);
            return result;
        }
        
        // 【防重复锁】检查该市场是否正在处理中（防止并发时重复下单）
        const conditionId = marketResult.market.conditionId;
        if (this.processingConditionIds.has(conditionId)) {
            result.error = '该市场正在处理中，跳过（防止并发重复下单）';
            console.log(`\n⏳ 跳过 ${symbol}: ${result.error}`);
            return result;
        }
        // 加锁：标记该市场正在处理
        this.processingConditionIds.add(conditionId);
        
        try {
        // 检查预测方向
        if (!prediction.direction) {
            result.error = '无预测方向';
            console.log(`\n❌ 跳过 ${symbol}: ${result.error}`);
            return result;
        }
        
        // 检查风险控制
        const riskCheck = this.checkRiskControls();
        if (riskCheck.shouldPause) {
            result.error = `风险控制: ${riskCheck.reason || '交易已暂停'}`;
            console.log(`\n🛡️  跳过 ${symbol}: ${result.error}`);
            return result;
        }
        // 按币种绝对止损当前已全局禁用，其余运行时 guards 保持原样。
        if (this.isSymbolAbsoluteStopped(symbol)) {
            const floor = this.getPerCoinAbsoluteFloor();
            result.error = `绝对止损: ${symbol} 资金低于该币种初始30% ($${floor.toFixed(2)})`;
            console.log(`\n🛑 跳过 ${symbol}: ${result.error} (当前 $${(this.getCoinCapital(symbol) ?? 0).toFixed(2)})`);
            return result;
        }
        
        // 检查今日交易次数（0=不限制）
        if (this.env.MAX_TRADES_PER_SESSION > 0 && this.state.todayTradeCount >= this.env.MAX_TRADES_PER_SESSION) {
            result.error = `已达到今日最大交易次数 (${this.env.MAX_TRADES_PER_SESSION})`;
            console.log(`\n❌ 跳过 ${symbol}: ${result.error}`);
            return result;
        }

        // ─── Cooldown 冷却检查（默认按币种+方向独立，可降级为按币种）──────────
        {
            await this.prepareLiveCooldownStateBeforeOrder();
            const cooldownKey = this.buildCooldownKey(symbol, prediction.direction);
            const cooldownLabel = this.formatCooldownLabel(symbol, prediction.direction);
            const pendingResolutionGuard = this.shouldBlockForPendingCooldownResolution(symbol);
            if (pendingResolutionGuard.blocked) {
                result.error = `冷却待确认: ${pendingResolutionGuard.label} 已有 ${pendingResolutionGuard.consecutiveBeforeUnknown} 连输，上一根 ${pendingResolutionGuard.unresolvedMarketSlug || ''} 已结束但结果尚未入账，先挡本轮`;
                console.log(`\n🧊 跳过 ${symbol}: ${result.error}`);
                this.appendCooldownBlockedOpportunity({
                    prediction,
                    marketResult,
                    symbol,
                    direction: prediction.direction,
                    targetPeriodEndTs: opts?.targetPeriodEndTs,
                    remainingBarsBefore: 0,
                    remainingBarsAfter: 0,
                    decrementedThisRound: false,
                    reasonCode: pendingResolutionGuard.reasonCode,
                });
                this.processingConditionIds.delete(conditionId);
                return result;
            }
            const persistedWindowGuard = this.getPersistedCooldownWindowBlock(symbol, prediction.direction, opts?.targetPeriodEndTs);
            if (persistedWindowGuard.blocked) {
                this.setCooldownRemaining(symbol, prediction.direction, persistedWindowGuard.remainingBarsAfter);
                result.error = persistedWindowGuard.error || `冷却中: ${persistedWindowGuard.label} 剩余 ${persistedWindowGuard.remainingBarsAfter} 个周期`;
                console.log(`\n🧊 跳过 ${symbol}: ${result.error}`);
                this.appendCooldownBlockedOpportunity({
                    prediction,
                    marketResult,
                    symbol,
                    direction: prediction.direction,
                    targetPeriodEndTs: opts?.targetPeriodEndTs,
                    remainingBarsBefore: persistedWindowGuard.remainingBarsBefore,
                    remainingBarsAfter: persistedWindowGuard.remainingBarsAfter,
                    decrementedThisRound: true,
                    reasonCode: 'cooldown_bars_blocked',
                });
                this.processingConditionIds.delete(conditionId);
                return result;
            }
            const cd = this.getCooldownRemaining(symbol, prediction.direction);
            if (cd > 0) {
                const decrementedSet = opts?.cooldownDecrementedThisRound;
                if (decrementedSet?.has(cooldownKey)) {
                    result.error = `冷却中: ${cooldownLabel} 剩余 ${cd} 个周期 (本轮已自减过，不再扣)`;
                    console.log(`\n🧊 跳过 ${symbol}: ${result.error}`);
                    this.appendCooldownBlockedOpportunity({
                        prediction,
                        marketResult,
                        symbol,
                        direction: prediction.direction,
                        targetPeriodEndTs: opts?.targetPeriodEndTs,
                        remainingBarsBefore: cd,
                        remainingBarsAfter: cd,
                        decrementedThisRound: true,
                        reasonCode: 'cooldown_bars_blocked',
                    });
                } else {
                    const nextCd = Math.max(0, Number(cd) - 1);
                    this.setCooldownRemaining(symbol, prediction.direction, nextCd);
                    if (decrementedSet) decrementedSet.add(cooldownKey);
                    result.error = `冷却中: ${cooldownLabel} 剩余 ${nextCd} 个周期 (cooldown_bars=${this.env.COOLDOWN_BARS})`;
                    console.log(`\n🧊 跳过 ${symbol}: ${result.error}`);
                    this.appendCooldownBlockedOpportunity({
                        prediction,
                        marketResult,
                        symbol,
                        direction: prediction.direction,
                        targetPeriodEndTs: opts?.targetPeriodEndTs,
                        remainingBarsBefore: cd,
                        remainingBarsAfter: nextCd,
                        decrementedThisRound: false,
                        reasonCode: 'cooldown_bars_blocked',
                    });
                }
                this.processingConditionIds.delete(conditionId);
                return result;
            }
        }

        const gentleFilter = this.evaluateSlot01GentleFilter(symbol, prediction.direction, prediction.confidence);
        if (gentleFilter.blocked) {
            const currentCapital = this.getCoinCapital(symbol) ?? this.state.currentCapital;
            result.error = `温和信心区间过滤: ${symbol} ${prediction.direction} 信心值 ${gentleFilter.roundedConfidence.toFixed(4)} 命中 [${gentleFilter.confidenceMin.toFixed(4)}, ${gentleFilter.confidenceMax.toFixed(4)})，创建订单前跳过`;
            console.log(`\n🧩 跳过 ${symbol}: ${result.error}`);
            this.appendSlot01GentleFilterBlockedOpportunity({
                prediction,
                marketResult,
                symbol,
                direction: prediction.direction,
                confidence: prediction.confidence,
                roundedConfidence: gentleFilter.roundedConfidence,
                confidenceMin: gentleFilter.confidenceMin,
                confidenceMax: gentleFilter.confidenceMax,
                currentCapital,
                targetPeriodEndTs: opts?.targetPeriodEndTs,
            });
            this.processingConditionIds.delete(conditionId);
            return result;
        }
        
        // betAmount 在获取 tokenPrice 后再计算（Kelly 需要实际价格）
        // 先获取代币信息和价格
        
        // 获取要购买的代币
        const token = getTokenForDirection(marketResult.market, prediction.direction);
        if (!token) {
            result.error = '无法确定目标代币';
            console.log(`\n❌ 跳过 ${symbol}: ${result.error}`);
            this.processingConditionIds.delete(conditionId); // 释放锁
            return result;
        }
        
        // 真实交易模式：重启后先查实际持仓，若已买够则跳过（避免重复下单）
        if (this.env.TRADING_MODE === TradingMode.LIVE && this.env.PROXY_WALLET && token.tokenId) {
            try {
                const positionValue = await getPositionValue(
                    this.env.PROXY_WALLET,
                    conditionId,
                    token.tokenId
                );
                // 若已持有金额 >= 预估下注金额的 90%，视为已买够（此处用简单百分比估算，精确值在 tokenPrice 确定后计算）
                const estBet = this.getCoinCapital(symbol) * (this.env.BET_SIZE_PERCENT / 100);
                if (positionValue >= estBet * 0.9) {
                    result.error = `该市场已持有足够仓位 (约 $${positionValue.toFixed(2)} >= 预估 $${estBet.toFixed(2)})，跳过`;
                    result.skippedReason = 'already_bought';
                    console.log(`\n⏭️ 跳过 ${symbol}: ${result.error}`);
                    this.processingConditionIds.delete(conditionId);
                    return result;
                }
            } catch (error) {
                // 查询持仓失败时：若本地日志近期有该市场 executed 记录，视为已买，跳过（防 Positions 未及时更新或 API 故障导致重复买）
                if (this.logger.hasRecentExecutedInLog(conditionId, 30)) {
                    result.error = 'Data API 查询持仓失败，且本地日志 30 分钟内已有该市场成交记录，跳过以防重复买入';
                    result.skippedReason = 'already_bought';
                    console.log(`\n⏭️ 跳过 ${symbol}: ${result.error}`);
                    this.processingConditionIds.delete(conditionId);
                    return result;
                }
                // 否则不阻塞下单，继续走正常流程
            }
        }
        
        // 获取实时价格（最大程度仿真实交易）
        // 优先使用订单簿（如果有 CLOB 客户端），否则使用价格 API
        let tokenPrice = token.price || 0.5; // 默认价格（降级处理）
        const useTriggerPrice = opts?.triggerPrice != null;
        const effectiveMaxPrice = opts?.maxPrice ?? this.MAX_PRICE_THRESHOLD;
        
        try {
            if (useTriggerPrice && opts?.triggerPrice != null) {
                tokenPrice = opts.triggerPrice;
            } else {
                // 方法1: 获取订单簿最佳卖价（最准确）- 用静默版本避免 SDK 打印 404 日志
                if (token.tokenId) {
                    const orderBook = await getOrderBook(token.tokenId);
                    if (orderBook && orderBook.asks && orderBook.asks.length > 0) {
                        // 获取最佳卖价（最低价格）
                        const bestAsk = orderBook.asks.reduce((min: any, ask: any) => 
                            parseFloat(ask.price) < parseFloat(min.price) ? ask : min
                        , orderBook.asks[0]);
                        tokenPrice = parseFloat(bestAsk.price);
                    }
                }
                
                // 方法2: 使用价格 API（公开接口，不需要认证）；有 triggerPrice 时不再拉价，复用 WS 价格
                if (!useTriggerPrice && token.tokenId && tokenPrice === (token.price || 0.5)) {
                    const realTimePrice = await getTokenPrice(token.tokenId, 'BUY');
                    if (realTimePrice !== null) {
                        tokenPrice = realTimePrice;
                    }
                }
            }
        } catch (error) {
            // 如果获取实时价格失败，使用默认价格（token.price 或 0.5）
            // 不影响交易流程
        }

        // ─── Edge/Kelly 实时计算 ──────────────
        const buyPrice = tokenPrice;
        const conf = prediction.confidence;
        const q = 1.0 - conf;
        // Edge 过滤用 limitPrice（实际执行价），Kelly 用 bestAsk（实时市场价）
        const limitPriceForEdge = this.env.LIMIT_PRICE;
        const edgeOdds = limitPriceForEdge > 0 && limitPriceForEdge < 1 ? (1.0 - limitPriceForEdge) / limitPriceForEdge : 0;
        const edgeAtLimit = conf * edgeOdds - q;
        const realOdds = buyPrice > 0 && buyPrice < 1 ? (1.0 - buyPrice) / buyPrice : 0;
        const realEdge = conf * realOdds - q;
        const kellyFraction = realOdds > 0 ? Math.max(0, (conf * realOdds - q) / realOdds) : 0;
        const directionalMinEdge = this.getDirectionalMinEdge(prediction.direction);
        const requestedNotional = this.resolveRequestedNotionalUsd(prediction.confidence, tokenPrice, symbol, prediction.direction, prediction);
        const requestedBetAmount = requestedNotional.requestedNotionalUsd;
        const consensusGate = this.getConsensusGateFailure(prediction, prediction.direction);
        if (consensusGate) {
            result.error = `共识分不足: score=${(consensusGate.score * 100).toFixed(2)}% < threshold=${(consensusGate.threshold * 100).toFixed(2)}%`;
                console.log(`\n⏭️ 跳过 ${symbol}: ${result.error}`);
            this.addExecutionSkippedEntry({
                symbol,
                direction: prediction.direction,
                confidence: prediction.confidence,
                reason: result.error,
                executionDecision: 'consensus_threshold_blocked',
                requestedAmountUsd: requestedBetAmount,
                targetNotionalUsd: requestedNotional.targetNotionalUsd,
                requestedNotionalUsd: requestedNotional.requestedNotionalUsd,
                postedNotionalUsd: 0,
                filledNotionalUsd: 0,
                requestCompressionMode: requestedNotional.requestCompressionMode,
                executionCompressionMode: this.getExecutionCompressionMode(),
                requestLayerCompressionUsd: requestedNotional.requestLayerCompressionUsd,
                executionLayerCompressionUsd: 0,
                executionLayerCompressionReason: 'not_posted',
                requestedShares: 0,
                originalRequestedShares: 0,
                market: {
                    title: marketResult.market.title,
                    conditionId: marketResult.market.conditionId,
                    slug: marketResult.market.slug,
                },
                token,
            });
            this.processingConditionIds.delete(conditionId);
            return result;
        }

        if (requestedBetAmount <= 0) {
            result.error = '下注金额不足';
            console.log(`\n❌ 跳过 ${symbol}: ${result.error}`);
            this.addExecutionSkippedEntry({
                symbol,
                direction: prediction.direction,
                confidence: prediction.confidence,
                reason: result.error,
                executionDecision: 'requested_notional_zero_after_gating',
                requestedAmountUsd: requestedBetAmount,
                targetNotionalUsd: requestedNotional.targetNotionalUsd,
                requestedNotionalUsd: requestedNotional.requestedNotionalUsd,
                postedNotionalUsd: 0,
                filledNotionalUsd: 0,
                requestCompressionMode: requestedNotional.requestCompressionMode,
                executionCompressionMode: this.getExecutionCompressionMode(),
                requestLayerCompressionUsd: requestedNotional.requestLayerCompressionUsd,
                executionLayerCompressionUsd: 0,
                executionLayerCompressionReason: 'not_posted',
                requestedShares: 0,
                originalRequestedShares: 0,
                market: {
                    title: marketResult.market.title,
                    conditionId: marketResult.market.conditionId,
                    slug: marketResult.market.slug,
                },
                token,
            });
            this.processingConditionIds.delete(conditionId);
            return result;
        }

        // Edge 过滤（用 limitPrice 而非 bestAsk，因为限价单实际执行价是 limitPrice）
        if (this.env.EDGE_FILTER_ENABLED && !useTriggerPrice) {
            if (edgeAtLimit < 0) {
                result.error = `边际为负: edge@limit=${(edgeAtLimit * 100).toFixed(2)}% (limit$${limitPriceForEdge.toFixed(3)}, bestAsk$${buyPrice.toFixed(3)}, 置信${(conf * 100).toFixed(1)}%)`;
                console.log(`\n⏭️ 跳过 ${symbol}: ${result.error}`);
                this._logEdgeSkip({ symbol, confidence: conf, bestAsk: buyPrice, edge: edgeAtLimit, minEdge: directionalMinEdge, conditionId, reason: 'negative_edge' });
                this.processingConditionIds.delete(conditionId);
                return result;
            }
            if (edgeAtLimit < directionalMinEdge) {
                result.error = `边际不足: edge@limit=${(edgeAtLimit * 100).toFixed(2)}% < MIN_EDGE_${prediction.direction}=${(directionalMinEdge * 100).toFixed(2)}% (limit$${limitPriceForEdge.toFixed(3)}, bestAsk$${buyPrice.toFixed(3)}, 置信${(conf * 100).toFixed(1)}%)`;
                console.log(`\n⏭️ 跳过 ${symbol}: ${result.error}`);
                this._logEdgeSkip({ symbol, confidence: conf, bestAsk: buyPrice, edge: edgeAtLimit, minEdge: directionalMinEdge, conditionId, reason: 'insufficient_edge' });
                this.processingConditionIds.delete(conditionId);
                return result;
            }
        }

        // 计算下注金额（传入置信度 + 当前代币价格，Kelly 模式需要后者）
        const coinCapForCheck = this.getCoinCapital(symbol);
        const riskUpperLimit = this.getSizingMode() === 'fixed_usd'
            ? roundUsd(Math.max(0, coinCapForCheck))
            : this.getCapitalRiskUpperLimit(Math.max(0, coinCapForCheck));
        const fixedConstraint = await this.getExecutionConstraint(token.tokenId, marketResult.market.slug);
        const fixedPlan = this.buildFixedOrderExecutionPlan({
            requestedAmountUsd: requestedBetAmount,
            price: tokenPrice,
            maxBudgetUsd: riskUpperLimit,
            constraint: fixedConstraint,
        });
        let betAmount = fixedPlan.finalAmountUsd;
        const betPercent = (betAmount / (coinCapForCheck || 1) * 100).toFixed(1);
        
        // 检查资金是否充足
        if (requestedBetAmount > coinCapForCheck) {
            result.error = `资金不足: 需要 $${requestedBetAmount.toFixed(2)}, 当前 $${coinCapForCheck.toFixed(2)}`;
            console.log(`\n❌ 跳过 ${symbol}: ${result.error}`);
            this.processingConditionIds.delete(conditionId);
            return result;
        }
        if (!fixedPlan.executable) {
            result.error = fixedPlan.failureReason || '固定价订单 uplift 后超过预算/风控上限';
            console.log(`\n❌ 跳过 ${symbol}: ${result.error}`);
            this.addExecutionSkippedEntry({
                symbol,
                direction: prediction.direction,
                confidence: prediction.confidence,
                reason: result.error,
                executionDecision: 'uplifted_fixed_order_to_min_shares',
                requestedAmountUsd: requestedBetAmount,
                targetNotionalUsd: requestedNotional.targetNotionalUsd,
                requestedNotionalUsd: requestedNotional.requestedNotionalUsd,
                postedNotionalUsd: 0,
                filledNotionalUsd: 0,
                requestCompressionMode: requestedNotional.requestCompressionMode,
                executionCompressionMode: this.getExecutionCompressionMode(),
                requestLayerCompressionUsd: requestedNotional.requestLayerCompressionUsd,
                executionLayerCompressionUsd: roundUsd(Math.max(0, requestedBetAmount - fixedPlan.finalAmountUsd)),
                executionLayerCompressionReason: fixedPlan.compressionReason || 'fixed_order_budget_guard',
                requestedShares: fixedPlan.requestedShares,
                originalRequestedShares: fixedPlan.requestedShares,
                market: {
                    title: marketResult.market.title,
                    conditionId: marketResult.market.conditionId,
                    slug: marketResult.market.slug,
                },
                token,
                exchangeMinOrderSizeShares: fixedConstraint.minOrderSizeShares,
                upliftedFixedOrders: fixedPlan.uplifted ? 1 : 0,
                upliftedNotionalUsd: fixedPlan.upliftedNotionalUsd,
                gt5PreservationStatus: fixedPlan.gt5PreservationStatus,
                compressionReason: fixedPlan.compressionReason,
            });
            this.processingConditionIds.delete(conditionId);
            return result;
        }

        this.appendExecutionV2Event({
            timestamp: new Date().toISOString(),
            symbol,
            direction: prediction.direction,
            marketSlug: marketResult.market.slug,
            conditionId,
            mode: this.env.TRADING_MODE,
            eventType: requestedNotional.requestCompressionMode === 'off' ? 'compression_bypassed' : 'request_notional_finalized',
            shadowOnly: false,
            targetNotionalUsd: requestedNotional.targetNotionalUsd,
            requestedAmountUsd: requestedNotional.requestedNotionalUsd,
            postedNotionalUsd: fixedPlan.finalAmountUsd,
            filledNotionalUsd: 0,
            sizingMode: requestedNotional.sizingMode,
            bankrollPct: requestedNotional.bankrollPct,
            confidenceBandScale: requestedNotional.confidenceBandScale,
            effectiveSizingPct: requestedNotional.effectiveSizingPct,
            currentCapital: requestedNotional.currentCapital,
            initialPrice: tokenPrice,
            currentPrice: tokenPrice,
            currentMaxPrice: effectiveMaxPrice,
            reason: `requestCompressionMode=${requestedNotional.requestCompressionMode};gateAmountUsd=${requestedNotional.gateAmountUsd.toFixed(2)}`,
        });

        const executionProfile = this.buildExecutionV2Profile();

        // 代币价过低视为市场已近乎定价该结果（如 Down@0.02=市场认为约 98% 涨）
        // 改进：不直接跳过，加入低价监控队列，等待价格回升到阈值以上再买入
        if (!useTriggerPrice && tokenPrice < this.env.MIN_TOKEN_PRICE) {
            // 计算监控时间窗口（与高价监控类似）
            const targetPeriodStart = opts?.targetPeriodEndTs || (Date.now() / 1000);
            const periodEndTime = targetPeriodStart + this.periodSeconds; // 订单结束时间
            const monitorEndTime = periodEndTime - (this.MONITOR_WINDOW_MINUTES * 60); // 结束前5分钟
            const monitorEndDate = new Date(monitorEndTime * 1000);
            const monitorEndStr = monitorEndDate.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', hour12: false });
            const now = new Date();
            
            // 如果已经过了监控窗口，直接跳过
            if (now.getTime() / 1000 > monitorEndTime) {
                result.error = `市场已近乎定价该结果 (${token.outcome} @ $${tokenPrice.toFixed(4)} < $${this.env.MIN_TOKEN_PRICE})，且监控窗口已过期，跳过`;
                console.log(`\n❌ 跳过 ${symbol}: ${result.error}`);
                this.processingConditionIds.delete(conditionId); // 释放锁
                return result;
            }
            
            // 如果该市场已在低价监控队列中（同一 conditionId + direction），不重复加入
            const existsInLow = this.lowPriceMonitorQueue.some((entry) =>
                entry.marketResult.market?.conditionId === conditionId &&
                entry.prediction.direction === prediction.direction &&
                entry.monitorEndTime.getTime() >= now.getTime()
            );
            if (!existsInLow) {
                const entry: PriceMonitorEntry = {
                    prediction,
                    marketResult,
                    targetPeriodEndTs: targetPeriodStart,
                    monitorStartTime: now,
                    monitorEndTime: monitorEndDate,
                    initialPrice: tokenPrice,
                    initialLimitPrice: tokenPrice,
                    executionProfile,
                    executionState: 'initial_ladder_posted',
                    requestedAmountUsd: requestedBetAmount,
                    remainingAmountUsd: requestedBetAmount,
                    currentMaxPrice: effectiveMaxPrice,
                    highestObservedPrice: tokenPrice,
                    chaseRound: 0,
                };
                this.appendExecutionV2Event({
                    timestamp: now.toISOString(),
                    symbol,
                    direction: prediction.direction,
                    marketSlug: marketResult.market?.slug,
                    conditionId: marketResult.market?.conditionId,
                    mode: this.env.TRADING_MODE,
                    eventType: 'initial_ladder_posted',
                    shadowOnly: Boolean(executionProfile.shadowOnly),
                    requestedAmountUsd: requestedBetAmount,
                    remainingAmountUsd: requestedBetAmount,
                    sizingMode: requestedNotional.sizingMode,
                    bankrollPct: requestedNotional.bankrollPct,
                    confidenceBandScale: requestedNotional.confidenceBandScale,
                    effectiveSizingPct: requestedNotional.effectiveSizingPct,
                    currentCapital: requestedNotional.currentCapital,
                    initialPrice: tokenPrice,
                    currentMaxPrice: effectiveMaxPrice,
                });
                this.lowPriceMonitorQueue.push(entry);
                if (this.env.TRADING_MODE === TradingMode.LIVE) this.schedulePreSignForPriceMonitorEntry(entry);
                const frameNote = this.env.TRADING_MODE === TradingMode.LIVE ? '防夹框架: LIVE 将预签，触发时只 POST' : '模拟模式: 不预签，触发时按原流程';
                console.log(`\n🔍 ${symbol}: 价格过低 $${tokenPrice.toFixed(4)} < $${this.env.MIN_TOKEN_PRICE}，加入低价监控队列等待回升 | ${frameNote}`);
            }
            
            result.error = `价格过低，已加入低价监控队列: $${tokenPrice.toFixed(4)} < $${this.env.MIN_TOKEN_PRICE}`;
            result.periodDisplay = this._fmtPeriod(targetPeriodStart);
            this.processingConditionIds.delete(conditionId); // 释放锁
            return result;
        }

        // 价格监控逻辑：价格 > MAX_TOKEN_PRICE 时一律加入监控队列，可下单截止前若价格回落到 [MIN, MAX] 则买入
        if (!useTriggerPrice && tokenPrice > this.MAX_PRICE_THRESHOLD) {
            // 计算监控时间窗口：可下单截止 = 周期结束前5分钟
            const targetPeriodStart = opts?.targetPeriodEndTs ?? (Date.now() / 1000);
            const periodEndTime = targetPeriodStart + this.periodSeconds; // 周期结束时间
            const monitorEndTime = periodEndTime - (this.MONITOR_WINDOW_MINUTES * 60); // 可下单截止（周期结束前5分钟）
            const monitorEndDate = new Date(monitorEndTime * 1000);
            const monitorEndStr = monitorEndDate.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', hour12: false });
            const now = new Date();
            
            // 一律加入监控队列（不因“已过可下单截止”而跳过）；队列检查器会在截止后静默移除
            const exists = this.priceMonitorQueue.some((entry) =>
                entry.marketResult.market?.conditionId === conditionId &&
                entry.prediction.direction === prediction.direction &&
                entry.monitorEndTime.getTime() >= now.getTime()
            );
            if (!exists) {
                const entry: PriceMonitorEntry = {
                    prediction,
                    marketResult,
                    targetPeriodEndTs: targetPeriodStart,
                    monitorStartTime: now,
                    monitorEndTime: monitorEndDate,
                    initialPrice: tokenPrice,
                    initialLimitPrice: tokenPrice,
                    executionProfile,
                    executionState: 'initial_ladder_posted',
                    requestedAmountUsd: requestedBetAmount,
                    remainingAmountUsd: requestedBetAmount,
                    currentMaxPrice: effectiveMaxPrice,
                    highestObservedPrice: tokenPrice,
                    chaseRound: 0,
                };
                this.appendExecutionV2Event({
                    timestamp: now.toISOString(),
                    symbol,
                    direction: prediction.direction,
                    marketSlug: marketResult.market?.slug,
                    conditionId: marketResult.market?.conditionId,
                    mode: this.env.TRADING_MODE,
                    eventType: 'initial_ladder_posted',
                    shadowOnly: Boolean(executionProfile.shadowOnly),
                    requestedAmountUsd: requestedBetAmount,
                    remainingAmountUsd: requestedBetAmount,
                    sizingMode: requestedNotional.sizingMode,
                    bankrollPct: requestedNotional.bankrollPct,
                    confidenceBandScale: requestedNotional.confidenceBandScale,
                    effectiveSizingPct: requestedNotional.effectiveSizingPct,
                    currentCapital: requestedNotional.currentCapital,
                    initialPrice: tokenPrice,
                    currentMaxPrice: effectiveMaxPrice,
                });
                this.priceMonitorQueue.push(entry);
                if (this.env.TRADING_MODE === TradingMode.LIVE) this.schedulePreSignForPriceMonitorEntry(entry);
                const periodStr = this._fmtPeriod(targetPeriodStart);
                const frameNote = this.env.TRADING_MODE === TradingMode.LIVE ? '防夹框架: LIVE 将预签，触发时只 POST' : '模拟模式: 不预签，触发时按原流程';
                console.log(`\n🔍 ${symbol}: 周期 ${periodStr} | 价格过高 $${tokenPrice.toFixed(4)} > $${this.MAX_PRICE_THRESHOLD}，已加入高价监控队列，${monitorEndStr} 前若回落至 [$${this.env.MIN_TOKEN_PRICE}, $${this.MAX_PRICE_THRESHOLD}] 则买入 | ${frameNote}`);
            }
            
            result.error = `价格过高，已加入监控队列: $${tokenPrice.toFixed(4)} > $${this.MAX_PRICE_THRESHOLD}（${monitorEndStr} 前价格回落则买入）`;
            result.periodDisplay = this._fmtPeriod(targetPeriodStart);
            this.processingConditionIds.delete(conditionId); // 释放锁
            return result;
        }
        if (useTriggerPrice && tokenPrice > effectiveMaxPrice) {
            // 最后5分钟市价带：允许 [0.3, 0.65] 内成交，防止高价 adverse selection
            if (tokenPrice < PredictionExecutor.LAST5MIN_MIN_PRICE || tokenPrice > PredictionExecutor.LAST5MIN_MAX_PRICE) {
                result.error = `触发价仍高于阈值: $${tokenPrice.toFixed(4)} > $${effectiveMaxPrice.toFixed(4)}`;
                this.processingConditionIds.delete(conditionId); // 释放锁
                return result;
            }
            // 在 [0.3, 0.65] 内，继续下单
        }

        // 价格在 [MIN_TOKEN_PRICE, MAX_TOKEN_PRICE] 或最后5分钟市价带内，立即下单
        const expectedTokens = betAmount / tokenPrice;
        const fixedSizingTrace = this.buildSizingTrace(prediction.confidence, symbol, prediction.direction);
        
        // 创建交易日志条目
        const logEntry = this.logger.addEntry({
            symbol,
            direction: prediction.direction,
            confidence: prediction.confidence,
            amount: betAmount,
            mode: this.env.TRADING_MODE,
            status: 'pending',
            marketTitle: marketResult.market.title,
            conditionId: marketResult.market.conditionId,
            tokenId: token.tokenId,
            // 结算相关字段
            marketSlug: marketResult.market.slug,
            tokenOutcome: token.outcome,  // "Up" 或 "Down"
            tokenPrice: tokenPrice,
            familyRuleVersion: this.getFamilyRuleVersion() || undefined,
            lowpriceFamilyId: this.getLowpriceFamilyId() || undefined,
            stalePolicyDecision: this.getStaleOrderPolicy() === 'off' ? 'off' : 'kept',
            exchangeMinOrderSizeShares: fixedConstraint.minOrderSizeShares,
            upliftedFixedOrders: fixedPlan.uplifted ? 1 : 0,
            upliftedNotionalUsd: fixedPlan.upliftedNotionalUsd,
            executionDecision: fixedPlan.uplifted ? 'uplifted_fixed_order_to_min_shares' : 'posted_pending',
            executionProfileVersion: executionProfile.version,
            sizingMode: fixedSizingTrace.sizingMode,
            bankrollPct: fixedSizingTrace.bankrollPct,
            confidenceBandScale: fixedSizingTrace.confidenceBandScale,
            effectiveSizingPct: fixedSizingTrace.effectiveSizingPct,
            currentCapital: fixedSizingTrace.currentCapital,
        });
        this.maybeUpdateExecutionProfileOnLog(logEntry.id, executionProfile, {
            executionState: 'waiting_fill',
        });
        
        result.tradeLogId = logEntry.id;
        result.amount = betAmount;
        result.direction = prediction.direction;
        
        // 打印详细交易信息
        const directionText = prediction.direction === 'UP' ? '📈 上涨' : '📉 下跌';
        const dualThreshold = this.env.PROB_THRESHOLD_UP != null && this.env.PROB_THRESHOLD_DOWN != null;
        const effectiveThreshold = Number(prediction.effectiveThreshold);
        const thresholdPct = dualThreshold 
            ? (prediction.direction === 'UP' 
                ? `${(this.env.PROB_THRESHOLD_UP! * 100).toFixed(0)}%` 
                : `≥${((1 - this.env.PROB_THRESHOLD_DOWN!) * 100).toFixed(0)}% (prob<${(this.env.PROB_THRESHOLD_DOWN! * 100).toFixed(0)}%)`)
            : `${((Number.isFinite(effectiveThreshold) ? effectiveThreshold : this.env.PROB_THRESHOLD) * 100).toFixed(0)}%`;
        const confidencePct = (prediction.confidence * 100).toFixed(1);
        
        const marketSlug = marketResult.market?.slug || '';
        // conditionId 已在前面声明（用于防重复锁），这里直接使用
        const marketLabel = symbol + '-15M' + (marketSlug ? ' (' + marketSlug + ')' : '');
        // 标题 slug、conditionId、真实下单 同源：均来自本次 marketResult.market，与 CLOB 使用的 tokenId/conditionId 一致
        const tradeNumber = this.state.todayTradeCount + 1; // 今日第几笔交易
        const periodStartTs = opts?.targetPeriodEndTs ?? parsePeriodStartFromSlug(marketSlug);
        console.log(`\n┌${'─'.repeat(58)}┐`);
        console.log(`│ 交易 #${tradeNumber} ${(marketSlug || logEntry.id.slice(-8)).padEnd(47)}│`);
        console.log(`├${'─'.repeat(58)}┤`);
        console.log(`│ 市场:       ${marketLabel.padEnd(45)}│`);
        // 统一显示周期（模拟/真实交易日志默认功能，无 ts 时显示未知）
        const periodDisplay = periodStartTs != null ? this._fmtPeriod(periodStartTs) : '(未知)';
        console.log(`│ 周期:       ${periodDisplay.padEnd(45)}│`);
        if (conditionId) {
            const cidShort = conditionId.length >= 18 ? `${conditionId.slice(0, 10)}…${conditionId.slice(-8)}` : conditionId;
            console.log(`│ conditionId: ${('' + cidShort + ' (与 CLOB 下单同源，可对单)').padEnd(45)}│`);
        }
        console.log(`│ 北京时间:   ${nowBeijing().padEnd(45)}│`);
        console.log(`│ 预测方向:   ${directionText.padEnd(45)}│`);
        console.log(`│ 置信度:     ${(confidencePct + '% (阈值: ' + thresholdPct + '%)').padEnd(45)}│`);
        const entryReason = fixedPlan.uplifted
            ? `固定价单按官方 min shares=${fixedConstraint.minOrderSizeShares.toFixed(2)} 强提`
            : '最终执行判定通过，允许下单';
        console.log(`│ 成交原因:   ${entryReason.padEnd(45)}│`);
        console.log(`│ 下注金额:   ${('$' + betAmount.toFixed(2) + ' (占资金 ' + betPercent + '%, 原$' + requestedBetAmount.toFixed(2) + ')').padEnd(45)}│`);
        console.log(`│ 代币价格:   ${('$' + tokenPrice.toFixed(4) + ' (赔率 ' + realOdds.toFixed(3) + ')').padEnd(45)}│`);
        console.log(`│ Edge/Kelly: ${('edge=' + (realEdge * 100).toFixed(2) + '% kelly=' + (kellyFraction * 100).toFixed(1) + '%').padEnd(45)}│`);
        console.log(`│ 预计获得:   ${('~' + fixedPlan.finalShares.toFixed(2) + ' 个代币').padEnd(45)}│`);
        console.log(`│ 交易模式:   ${getTradingModeText(this.env.TRADING_MODE).padEnd(45)}│`);
        
        try {
            // 根据模式执行
            switch (this.env.TRADING_MODE) {
                case TradingMode.BACKTEST:
                    // 回测模式：仅记录，不调用任何 API
                    console.log(`│ 状态:       ${'✅ 已记录 (回测模式)'.padEnd(45)}│`);
                    this.logger.updateEntry(logEntry.id, { status: 'executed' });
                    result.success = true;
                    result.actualAmount = roundUsd(betAmount); // 回测模式假设全部成交
                    break;
                    
                case TradingMode.SIMULATION:
                    // 模拟模式：调用 API 获取订单簿和流动性，但不实际下单
                    // 最大程度仿真实交易：检查流动性，模拟部分成交，剩余部分加入流动性监控队列
                    const simResult = await this.simulateTrade(marketResult, token, betAmount, logEntry, prediction, {
                        ...opts,
                        executionProfile,
                    });
                    result.success = true;
                    result.actualAmount = roundUsd(simResult.actualAmount); // 记录实际成交金额
                    break;
                    
                case TradingMode.LIVE:
                    // 真实交易模式：实际下单
                    if (this.clobClient) {
                        const txResult = await this.executeLiveTrade(
                            marketResult,
                            token,
                            betAmount,
                            logEntry,
                            prediction,  // 传递 prediction 用于流动性监控
                            {
                                ...opts,
                                executionProfile,
                            }         // 传递 opts 用于流动性监控
                        );
                        result.success = txResult.success;
                        if (txResult.txHash) {
                            result.tradeLogId = txResult.txHash;
                        }
                        
                        // 计算实际成交金额（用于资金更新），实盘金额统一 2 位小数
                        const liveActualAmount = txResult.success 
                            ? betAmount - (txResult.remainingAmount || 0)
                            : 0;
                        result.actualAmount = roundUsd(liveActualAmount);
                        
                        // 如果下单失败且是真实交易，将剩余金额加入流动性监控队列
                        if (!txResult.success && !txResult.blockedByCooldown && betAmount >= MIN_ORDER_SIZE_USD) {
                            const targetPeriodStart = opts?.targetPeriodEndTs || (Date.now() / 1000);
                            // targetPeriodStart 是订单开始时间（K线周期开始），订单结束时间是开始时间 + PERIOD_SECONDS
                            const periodEndTime = targetPeriodStart + this.periodSeconds; // 订单结束时间
                            const monitorEndTime = periodEndTime - (this.MONITOR_WINDOW_MINUTES * 60); // 订单结束前5分钟
                            const monitorEndDate = new Date(monitorEndTime * 1000);
            const monitorEndStr = monitorEndDate.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', hour12: false });
                            const now = new Date();
                            
                            // 如果还在监控窗口内，加入流动性监控队列
                            if (now.getTime() / 1000 < monitorEndTime) {
                                // 计算已买入金额（从交易日志中查找同一conditionId的所有已执行交易）
                                const allEntries = this.logger.getAllEntries();
                                const sameMarketEntries = allEntries.filter(e => 
                                    e.conditionId === marketResult.market?.conditionId && 
                                    e.status === 'executed' &&
                                    e.id !== logEntry.id
                                );
                                const totalBoughtAmount = sameMarketEntries.reduce((sum, e) => sum + (e.amount || 0), 0);
                                
                            this.liquidityMonitorQueue.push({
                                prediction,
                                marketResult,
                                targetPeriodEndTs: targetPeriodStart,
                                monitorStartTime: now,
                                monitorEndTime: monitorEndDate,
                                remainingAmount: roundUsd(betAmount),
                                originalAmount: roundUsd(betAmount),
                                logEntryId: logEntry.id,
                                totalBoughtAmount: roundUsd(totalBoughtAmount),
                                minPrice: 0.25,
                                maxPrice: effectiveMaxPrice,
                                executionProfile,
                                executionState: 'waiting_fill',
                                highestObservedPrice: tokenPrice,
                                chaseRound: 0,
                                initialLimitPrice: tokenPrice,
                            });
                                console.log(`│ 监控剩余:   ${('$' + betAmount.toFixed(2) + ' 已加入流动性监控队列 (价格: 0.25-' + this.env.MAX_TOKEN_PRICE.toFixed(2) + ')').padEnd(45)}│`);
                            }
                        }
                        if (!txResult.success && txResult.blockedByCooldown) {
                            result.error = txResult.blockReason || '冷却阻断';
                        }
                    } else {
                        throw new Error('CLOB 客户端未初始化');
                    }
                    break;
            }
            
            // 更新资金（使用实际成交金额，而非计划下注金额）
            // LIVE 模式：currentCapital 由每 10 秒 refreshLiveCapital() 从链上同步，此处仅做即时扣减显示
            const actualDeduction = roundUsd(result.actualAmount ?? betAmount);
            if (actualDeduction > 0) {
                this.adjustCoinCapital(symbol, -actualDeduction);
                console.log(`│ 剩余资金:   ${('$' + this.state.currentCapital.toFixed(2) + (this.perCoinEnabled ? ` (${symbol}=$${this.getCoinCapital(symbol).toFixed(2)})` : '')).padEnd(45)}│`);
                
                this.updateCoinPeakIfNeeded(symbol);
            }
            
            // 更新今日交易计数
            this.state.todayTradeCount++;
            
        } catch (error) {
            result.error = parseErrorReason(error);
            console.log(`│ 状态:       ${'❌ 失败'.padEnd(45)}│`);
            console.log(`│ 错误原因:   ${result.error.slice(0, 45).padEnd(45)}│`);
            this.logger.updateEntry(logEntry.id, { 
                status: 'failed', 
                error: result.error 
            });
        }
        
        console.log(`└${'─'.repeat(58)}┘`);
        
        return result;
        } finally {
            // 释放锁（try-finally 确保异常时也能释放）
            this.processingConditionIds.delete(conditionId);
        }
    }
    
    /**
     * 模拟交易（不实际下单）
     * 为了最大程度仿真实交易，使用实时价格和流动性检查（从订单簿获取）
     * @returns 实际成交金额（用于资金扣除）
     */
    private async simulateTrade(
        marketResult: MarketSearchResult,
        token: { tokenId: string; outcome: string; price?: number },
        betAmount: number,
        logEntry: TradeLogEntry,
        prediction?: PredictionResult,
        opts?: {
            targetPeriodEndTs?: number;
            minPrice?: number;
            maxPrice?: number;
            triggerPrice?: number;
            asksSnapshot?: Array<{ price: string; size: string }>;
            executionProfile?: ExecutionV2Profile;
        }
    ): Promise<{ actualAmount: number }> {
        const minPrice = opts?.minPrice || 0.25;
        const maxPrice = opts?.maxPrice || this.MAX_PRICE_THRESHOLD;
        let price = token.price || 0.5;
        let availableLiquidity = 0;
        const useWsData = opts?.triggerPrice != null && opts?.asksSnapshot != null && opts.asksSnapshot.length > 0;

        if (useWsData) {
            // 复用 WS 推送的 best_ask 和订单簿，不再拉取 API
            price = opts.triggerPrice!;
            for (const ask of opts.asksSnapshot!) {
                const askPrice = parseFloat(ask.price);
                if (askPrice < minPrice || askPrice > maxPrice) continue;
                const askSize = parseFloat(ask.size || '0');
                availableLiquidity += askSize * askPrice;
            }
        } else {
            try {
                if (token.tokenId) {
                    const orderBook = await getOrderBook(token.tokenId);
                    if (orderBook && orderBook.asks && orderBook.asks.length > 0) {
                        const bestAsk = orderBook.asks.reduce((min: any, ask: any) =>
                            parseFloat(ask.price) < parseFloat(min.price) ? ask : min
                        , orderBook.asks[0]);
                        price = parseFloat(bestAsk.price);
                        try {
                            for (const ask of orderBook.asks) {
                                const askPrice = parseFloat(ask.price);
                                if (askPrice < minPrice || askPrice > maxPrice) continue;
                                const askSize = parseFloat((ask as any).size || (ask as any).amount || '0');
                                availableLiquidity += askSize * askPrice;
                            }
                        } catch (error) { /* ignore */ }
                    }
                }
                if (token.tokenId && price === (token.price || 0.5)) {
                    const realTimePrice = await getTokenPrice(token.tokenId, 'BUY');
                    if (realTimePrice !== null) price = realTimePrice;
                }
            } catch (error) {
                // 获取失败时使用默认价格
            }
        }
        
        // 更新日志中的价格
        this.logger.updateEntry(logEntry.id, { 
            tokenPrice: price 
        });
        
        // 模拟流动性检查（最大程度仿真实交易）
        // 如果流动性不足，模拟部分成交，剩余部分加入流动性监控队列
        let actualAmount = betAmount; // 实际成交金额
        let remainingAmount = 0; // 剩余金额
        
        if (availableLiquidity > 0 && availableLiquidity < betAmount) {
            // 流动性不足，模拟部分成交；金额统一 2 位小数
            actualAmount = roundUsd(availableLiquidity * 0.95); // 95% 流动性，留5%缓冲
            remainingAmount = roundUsd(betAmount - actualAmount);
            
            // 如果剩余金额 >= 最小订单金额，加入流动性监控队列（模拟模式也支持）
            if (remainingAmount >= MIN_ORDER_SIZE_USD && prediction && marketResult.market) {
                const targetPeriodStart = opts?.targetPeriodEndTs || (Date.now() / 1000);
                const periodEndTime = targetPeriodStart + this.periodSeconds; // 订单结束时间
                const monitorEndTime = periodEndTime - (this.MONITOR_WINDOW_MINUTES * 60); // 订单结束前5分钟
                const monitorEndDate = new Date(monitorEndTime * 1000);
            const monitorEndStr = monitorEndDate.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', hour12: false });
                const now = new Date();
                
                // 如果还在监控窗口内，加入流动性监控队列
                if (now.getTime() / 1000 < monitorEndTime) {
                    const allEntries = this.logger.getAllEntries();
                    const sameMarketEntries = allEntries.filter(e => 
                        e.conditionId === marketResult.market?.conditionId && 
                        e.status === 'executed' &&
                        e.id !== logEntry.id
                    );
                    const totalBoughtAmount = roundUsd(sameMarketEntries.reduce((sum, e) => sum + (e.amount || 0), 0) + actualAmount);
                    
                    this.liquidityMonitorQueue.push({
                        prediction,
                        marketResult,
                        targetPeriodEndTs: targetPeriodStart,
                        monitorStartTime: now,
                        monitorEndTime: monitorEndDate,
                        remainingAmount,
                        originalAmount: roundUsd(betAmount),
                        logEntryId: logEntry.id,
                        totalBoughtAmount,
                        minPrice: opts?.minPrice || 0.25,
                        maxPrice: opts?.maxPrice || this.MAX_PRICE_THRESHOLD,
                        executionProfile: opts?.executionProfile,
                        executionState: 'waiting_fill',
                        highestObservedPrice: price,
                        chaseRound: 0,
                        initialLimitPrice: price,
                    });
                    this.maybeUpdateExecutionProfileOnLog(logEntry.id, opts?.executionProfile, {
                        executionState: 'waiting_fill',
                    });
                }
            }
        }
        
        const actualTokens = actualAmount / price;
        const expectedTokens = betAmount / price;
        
        console.log(`│ 代币选择:   ${(token.outcome + ' @ $' + price.toFixed(4)).padEnd(45)}│`);
        if (remainingAmount > 0) {
            // 部分成交
            const fillPercent = ((actualAmount / betAmount) * 100).toFixed(1);
            console.log(`│ 预计获得:   ${('~' + expectedTokens.toFixed(2) + ' 个代币').padEnd(45)}│`);
            console.log(`│ 状态:       ${('⚠️  模拟部分成交 (' + fillPercent + '%)').padEnd(45)}│`);
            console.log(`│ 成交金额:   ${('$' + actualAmount.toFixed(2) + ' / $' + betAmount.toFixed(2)).padEnd(45)}│`);
            console.log(`│ 成交数量:   ${('~' + actualTokens.toFixed(2) + ' 个代币').padEnd(45)}│`);
            console.log(`│ 未成交:     ${('$' + remainingAmount.toFixed(2) + ' (已加入流动性监控)').padEnd(45)}│`);
            
            // 更新日志为实际成交金额（2 位小数）
            this.logger.updateEntry(logEntry.id, { 
                status: 'executed',
                amount: actualAmount,
            });
        } else {
            // 全部成交
            console.log(`│ 预计获得:   ${('~' + expectedTokens.toFixed(2) + ' 个代币').padEnd(45)}│`);
            console.log(`│ 状态:       ${'✅ 模拟成交 (未实际下单，使用实时价格)'.padEnd(45)}│`);
            
            this.logger.updateEntry(logEntry.id, { status: 'executed', amount: roundUsd(betAmount) });
        }
        
        return { actualAmount };
    }
    
    /**
     * 执行真实交易
     * @param prediction 预测结果（可选，用于流动性监控）
     * @param opts 选项（可选，用于流动性监控）
     */
    /**
     * 按订单 ID 查询该订单作为 taker 的实际成交（市价单我们是 taker，不能用 maker_address 查）
     * 支持一笔订单多笔成交的汇总
     */
    private async fetchTradesByOrderId(conditionId: string | undefined, orderID: string): Promise<{ amountUsd: number; tokens: number }> {
        if (!conditionId || !this.clobClient || typeof this.clobClient.getTrades !== 'function') return { amountUsd: 0, tokens: 0 };
        const trades = await this.clobClient.getTrades({ market: conditionId });
        let amountUsd = 0;
        let tokens = 0;
        for (const t of trades || []) {
            if (t?.taker_order_id === orderID) {
                const sz = parseFloat(t.size || '0');
                const pr = parseFloat(t.price || '0');
                if (Number.isFinite(sz) && Number.isFinite(pr) && sz > 0 && pr > 0) {
                    amountUsd += sz * pr;
                    tokens += sz;
                }
                continue;
            }
            const makerOrders = Array.isArray((t as any)?.maker_orders) ? (t as any).maker_orders : [];
            for (const maker of makerOrders) {
                if (String(maker?.order_id || '') !== orderID) continue;
                const sz = parseFloat(String(maker?.matched_amount || t.size || '0'));
                const pr = parseFloat(String(maker?.price || t.price || '0'));
                if (Number.isFinite(sz) && Number.isFinite(pr) && sz > 0 && pr > 0) {
                    amountUsd += sz * pr;
                    tokens += sz;
                }
            }
        }
        return { amountUsd: roundUsd(amountUsd), tokens };
    }

    /**
     * 查询订单实际成交（先 getOrder，无结果再 getTrades）。
     * 返回 amountUsd >= 0 表示成功查到；-1 表示暂未查到（可重试）。
     */
    private async queryActualFill(
        orderID: string | undefined,
        conditionId: string | undefined,
        price: number,
        betAmount: number
    ): Promise<{ amountUsd: number; tokens: number; source: string }> {
        if (!orderID || !this.clobClient) return { amountUsd: -1, tokens: 0, source: '' };
        if (price <= 0 || !Number.isFinite(price)) return { amountUsd: -1, tokens: 0, source: '' }; // 避免除零
        try {
            if (typeof this.clobClient.getOrder === 'function') {
                const order = await this.clobClient.getOrder(orderID);
                if (order && order.size_matched !== undefined && order.size_matched !== null && order.size_matched !== '') {
                    const sizeMatched = parseFloat(String(order.size_matched));
                    const orderAny = order as any;
                    const officialStatus = String(orderAny.status || orderAny.order_status || orderAny.state || '').trim().toUpperCase();
                    if (
                        Number.isFinite(sizeMatched) &&
                        sizeMatched <= 0 &&
                        (officialStatus === 'CANCELED' || officialStatus === 'CANCELLED')
                    ) {
                        return {
                            amountUsd: 0,
                            tokens: 0,
                            source: 'official_canceled_zero_fill',
                        };
                    }
                    if (Number.isFinite(sizeMatched) && sizeMatched > 0) {
                        // CLOB V2 getOrder 会返回订单实际价格；有 size_matched + price 时可直接回填。
                        // 旧逻辑只返回 size_only，导致官网 MATCHED 后本地长期停在 posted_pending。
                        const officialPrice = this.extractOfficialOrderPrice(order, price);
                        const amountUsd = roundUsd(sizeMatched * officialPrice);
                        if (amountUsd > 0) {
                            return {
                                amountUsd,
                                tokens: sizeMatched,
                                source: `getOrder_${officialStatus ? officialStatus.toLowerCase() : 'matched'}`,
                            };
                        }
                    }
                    if (Number.isFinite(sizeMatched) && sizeMatched === 0) {
                        return {
                            amountUsd: -1,
                            tokens: 0,
                            source: 'getOrder_open_zero_fill',
                        };
                    }
                }
            }
        } catch (_) { /* 继续用 getTrades */ }
        try {
            if (typeof this.clobClient.getTrades === 'function' && conditionId) {
                const { amountUsd, tokens } = await this.fetchTradesByOrderId(conditionId, orderID);
                // getTrades 返回 0 可能是延迟未写入，让调用方重试
                if (amountUsd > 0) return { amountUsd, tokens, source: 'getTrades' };
            }
        } catch (_) { /* 忽略 */ }
        return { amountUsd: -1, tokens: 0, source: '' };
    }

    private buildLiveTradeGroupKey(conditionId: string | undefined, marketSlug: string | undefined): string {
        return `${String(conditionId || '').trim()}::${String(marketSlug || '').trim()}`;
    }

    private buildLiveTradeConditionKey(conditionId: string | undefined): string {
        return String(conditionId || '').trim().toLowerCase();
    }

    private normalizeOutcomeName(value: unknown): string {
        const text = String(value ?? '').trim().toLowerCase();
        if (text === 'up' || text === 'yes') return 'up';
        if (text === 'down' || text === 'no') return 'down';
        return text;
    }

    private expectedOutcomeForEntry(entry: Pick<TradeLogEntry, 'tokenOutcome' | 'direction'>): string {
        return this.normalizeOutcomeName(entry.tokenOutcome || (entry.direction === 'UP' ? 'Up' : entry.direction === 'DOWN' ? 'Down' : ''));
    }

    private publicActivityMatchesEntry(trade: PublicActivityTrade, entry: Pick<TradeLogEntry, 'conditionId' | 'marketSlug' | 'tokenOutcome' | 'direction'>): boolean {
        const tradeCondition = this.buildLiveTradeConditionKey(trade.conditionId);
        const entryCondition = this.buildLiveTradeConditionKey(entry.conditionId);
        const tradeSlug = String(trade.marketSlug || '').trim();
        const entrySlug = String(entry.marketSlug || '').trim();
        const sameMarket = (
            (tradeCondition && entryCondition && tradeCondition === entryCondition) ||
            (tradeSlug && entrySlug && tradeSlug === entrySlug)
        );
        if (!sameMarket) return false;

        const expectedOutcome = this.expectedOutcomeForEntry(entry);
        const actualOutcome = this.normalizeOutcomeName(trade.tokenOutcome);
        return !expectedOutcome || !actualOutcome || expectedOutcome === actualOutcome;
    }

    private getPublicTradesForLocalRows(
        localRows: Pick<TradeLogEntry, 'conditionId' | 'marketSlug' | 'tokenOutcome' | 'direction'>[],
        publicTradesByGroup: Map<string, PublicActivityTrade[]>,
        publicTradesByCondition: Map<string, PublicActivityTrade[]>,
        groupKey?: string,
    ): PublicActivityTrade[] {
        const candidates: PublicActivityTrade[] = [];
        const seen = new Set<string>();
        const add = (row: PublicActivityTrade) => {
            const key = `${row.transactionHash || ''}::${row.timestampSec || ''}::${row.usdcSize || ''}::${row.tokenId || ''}`;
            if (seen.has(key)) return;
            seen.add(key);
            candidates.push(row);
        };

        if (groupKey) {
            for (const row of publicTradesByGroup.get(groupKey) || []) add(row);
        }
        for (const localRow of localRows) {
            const conditionKey = this.buildLiveTradeConditionKey(localRow.conditionId);
            if (!conditionKey) continue;
            for (const row of publicTradesByCondition.get(conditionKey) || []) add(row);
        }

        return candidates.filter((trade) => localRows.some((entry) => this.publicActivityMatchesEntry(trade, entry)));
    }

    private async fetchPublicActivityTrades(sinceTimestampSec?: number): Promise<PublicActivityTrade[]> {
        const proxyWallet = String(this.env.PROXY_WALLET || '').trim().toLowerCase();
        if (!proxyWallet) return [];
        try {
            const rows: any[] = [];
            const limit = 500;
            let offset = 0;
            while (true) {
                let response: any;
                try {
                    response = await axios.get('https://data-api.polymarket.com/activity', {
                        params: { user: proxyWallet, limit, offset },
                        timeout: Math.max(this.env.REQUEST_TIMEOUT_MS, 5000),
                        headers: { 'User-Agent': 'polyfun-live-reconcile/1.0' },
                    });
                } catch (error) {
                    if (rows.length > 0) {
                        console.log(`  [LIVE 对账] 公共 activity API 翻页在 offset=${offset} 中止，保留已读取 ${rows.length} 行: ${error instanceof Error ? error.message : String(error)}`);
                        break;
                    }
                    throw error;
                }
                const batch = Array.isArray(response.data) ? response.data : [];
                rows.push(...batch);
                if (batch.length < limit) break;
                offset += limit;
                if (offset > 5000) {
                    console.log(`  [LIVE 对账] 公共 activity API 超过最大翻页 offset=${offset}，保留已读取 ${rows.length} 行`);
                    break;
                }
            }
            return rows
                .filter((row: any) => String(row?.type || '').toUpperCase() === 'TRADE')
                .filter((row: any) => String(row?.side || '').toUpperCase() === 'BUY')
                .filter((row: any) => String(row?.proxyWallet || '').trim().toLowerCase() === proxyWallet)
                .map((row: any) => {
                    const timestampSec = Number(row?.timestamp || 0);
                    return {
                        conditionId: String(row?.conditionId || '').trim(),
                        marketSlug: String(row?.slug || row?.eventSlug || '').trim(),
                        tokenId: String(row?.asset || '').trim(),
                        tokenOutcome: String(row?.outcome || '').trim(),
                        transactionHash: String(row?.transactionHash || '').trim(),
                        timestampIso: Number.isFinite(timestampSec) && timestampSec > 0
                            ? new Date(timestampSec * 1000).toISOString()
                            : new Date().toISOString(),
                        timestampSec,
                        usdcSize: Number(row?.usdcSize || 0),
                        size: Number(row?.size || 0),
                        price: Number(row?.price || 0),
                    } as PublicActivityTrade;
                })
                .filter((row) => row.conditionId && row.marketSlug && row.transactionHash)
                .filter((row) => sinceTimestampSec == null || row.timestampSec >= sinceTimestampSec)
                .sort((a, b) => a.timestampSec - b.timestampSec || a.transactionHash.localeCompare(b.transactionHash));
        } catch (error) {
            console.log(`  [LIVE 对账] 公共 activity API 读取失败: ${error instanceof Error ? error.message : String(error)}`);
            return [];
        }
    }

    private annotateLiveReportLastAttemptFromCanonicalRows(report: any, trades: any[], publicTrades: PublicActivityTrade[]): void {
        const allowedSymbols = new Set((this.env.ALLOWED_MARKETS || []).map((sym) => String(sym).trim().toUpperCase()));
        const localRows = trades
            .filter((row) => allowedSymbols.has(String(row?.symbol || '').trim().toUpperCase()))
            .filter((row) => row?.timestamp)
            .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
        const latest = localRows[localRows.length - 1];
        if (!latest || !report?.summary) return;

        const logicalKey = latest.conditionId || latest.id;
        const group = localRows.filter((row) => (row.conditionId || row.id) === logicalKey);
        const groupPublicRows = publicTrades
            .filter((trade) => group.some((row) => this.publicActivityMatchesEntry(trade, row)))
            .sort((a, b) => a.timestampSec - b.timestampSec || a.transactionHash.localeCompare(b.transactionHash));
        const publicFillUsd = roundUsd(groupPublicRows.reduce((sum, row) => sum + (Number(row.usdcSize) || 0), 0));
        const latestLocalTs = group
            .map((row) => new Date(row.timestamp).getTime())
            .filter((value) => Number.isFinite(value))
            .sort((a, b) => a - b)
            .slice(-1)[0];
        const latestPublicTs = groupPublicRows
            .map((row) => row.timestampSec * 1000)
            .filter((value) => Number.isFinite(value))
            .sort((a, b) => a - b)
            .slice(-1)[0];

        report.summary.lastAttemptAt = new Date(Math.max(latestLocalTs || 0, latestPublicTs || 0)).toISOString();
        report.summary.lastAttemptMarketSlug = latest.marketSlug || undefined;
        report.summary.lastAttemptConditionId = latest.conditionId || undefined;
        report.summary.lastAttemptDirection = latest.direction || undefined;
        report.summary.lastAttemptPublicFillUsd = publicFillUsd;
        report.summary.lastAttemptSource = 'canonical_local_trade_group_with_public_activity_crosscheck';

        if (publicFillUsd > 0 || group.some((row) => row.status === 'executed')) {
            report.summary.lastAttemptResult = publicFillUsd > 0
                ? 'filled_visible_in_public_activity'
                : 'filled_local_log';
            report.summary.lastNonExecutionReason = undefined;
            return;
        }
        if (group.some((row) => row.status === 'pending')) {
            report.summary.lastAttemptResult = 'posted_pending';
            report.summary.lastNonExecutionReason = undefined;
            return;
        }

        const lastReason = [...group]
            .reverse()
            .map((row) => String(row.lastNonExecutionReason || row.lastRejectionReason || row.error || '').trim())
            .find((value) => value) || 'not_executed';
        report.summary.lastAttemptResult = 'not_filled';
        report.summary.lastNonExecutionReason = lastReason;
    }

    private extractMinimumSharesRequirement(reason: string | undefined): number | null {
        const text = String(reason || '').trim();
        if (!text) return null;
        const match = text.match(/minimum:\s*([0-9]+(?:\.[0-9]+)?)/i);
        if (!match) return null;
        const value = Number(match[1]);
        return Number.isFinite(value) && value > 0 ? value : null;
    }

    private rememberLiveExchangeConstraints(
        reason: string | undefined,
        tokenId?: string,
        marketSlug?: string,
    ): void {
        const minShares = this.extractMinimumSharesRequirement(reason);
        if (minShares != null) {
            this.liveExchangeMinShares = Math.max(this.liveExchangeMinShares, minShares);
            if (tokenId) {
                const current = this.executionConstraintByToken.get(tokenId);
                const nextMinShares = current?.minOrderSizeShares && current.minOrderSizeShares > minShares
                    ? current.minOrderSizeShares
                    : minShares;
                this.executionConstraintByToken.set(tokenId, {
                    tokenId,
                    marketSlug: String(marketSlug || current?.marketSlug || '').trim(),
                    minOrderSizeShares: nextMinShares,
                    tickSize: current?.tickSize ?? null,
                    retrievedAt: new Date().toISOString(),
                    source: 'last_live_reject',
                });
            }
        }
    }

    private isNonRetryableLiveReject(reason: string | undefined): boolean {
        const text = String(reason || '').toLowerCase();
        if (!text) return false;
        return (
            text.includes('size lower than minimum') ||
            text.includes('trading restricted in your region') ||
            text.includes('restricted in your region') ||
            text.includes('market not accessible in your region')
        );
    }

    private getLiveExchangeMinShares(): number {
        return Number.isFinite(this.liveExchangeMinShares) && this.liveExchangeMinShares > 0
            ? this.liveExchangeMinShares
            : DEFAULT_EXCHANGE_MIN_SHARES;
    }

    private buildFallbackExecutionConstraint(tokenId?: string, marketSlug?: string): ExchangeExecutionConstraint {
        return {
            tokenId: String(tokenId || '').trim(),
            marketSlug: String(marketSlug || '').trim(),
            minOrderSizeShares: this.getLiveExchangeMinShares(),
            tickSize: null,
            retrievedAt: new Date().toISOString(),
            source: 'default_fallback',
        };
    }

    private async getExecutionConstraint(tokenId?: string, marketSlug?: string): Promise<ExchangeExecutionConstraint> {
        const normalizedTokenId = String(tokenId || '').trim();
        const normalizedMarketSlug = String(marketSlug || '').trim();
        const cached = normalizedTokenId ? this.executionConstraintByToken.get(normalizedTokenId) : null;
        try {
            let book: any = null;
            if (this.clobClient && normalizedTokenId) {
                try {
                    book = await this.clobClient.getOrderBook(normalizedTokenId as any);
                } catch {
                    book = null;
                }
            }
            if (!book && normalizedTokenId) {
                book = await getOrderBook(normalizedTokenId);
            }
            const minOrderSizeRaw = Number(book?.min_order_size ?? book?.minOrderSize ?? book?.minOrderSizeShares);
            const tickSizeRaw = Number(book?.tick_size ?? book?.tickSize);
            if (Number.isFinite(minOrderSizeRaw) && minOrderSizeRaw > 0) {
                const constraint: ExchangeExecutionConstraint = {
                    tokenId: normalizedTokenId,
                    marketSlug: normalizedMarketSlug,
                    minOrderSizeShares: minOrderSizeRaw,
                    tickSize: Number.isFinite(tickSizeRaw) && tickSizeRaw > 0 ? tickSizeRaw : null,
                    retrievedAt: new Date().toISOString(),
                    source: 'official_book',
                };
                this.liveExchangeMinShares = Math.max(this.liveExchangeMinShares, constraint.minOrderSizeShares);
                if (normalizedTokenId) {
                    this.executionConstraintByToken.set(normalizedTokenId, constraint);
                }
                return constraint;
            }
        } catch {
            // 回退到缓存/拒单学习/默认值
        }
        if (cached) {
            return {
                ...cached,
                marketSlug: normalizedMarketSlug || cached.marketSlug,
            };
        }
        return this.buildFallbackExecutionConstraint(normalizedTokenId, normalizedMarketSlug);
    }

    private getLowpriceFamilyId(): string {
        return String(this.env.LOWPRICE_FAMILY_ID || '').trim();
    }

    private getFamilyRuleVersion(): string {
        return String(this.env.FAMILY_RULE_VERSION || '').trim();
    }

    private getStaleOrderPolicy(): string {
        return String(this.env.STALE_ORDER_POLICY || 'off').trim().toLowerCase();
    }

    private buildStaleCancelReason(candidate: StaleLiveOrderCandidate): string {
        if (candidate.cancelReasonKind === 'late_fill_timeout') {
            return [
                'late_fill_timeout_cancelled',
                `age=${candidate.orderAgeSec}s`,
                `threshold=${candidate.lateFillCancelMinutes ?? 5}m`,
                `tte=${candidate.timeToExpirySec}s`,
                `limit=$${candidate.limitPrice.toFixed(2)}`,
            ].join(' | ');
        }
        return [
            'stale_order_cancelled',
            `age=${candidate.orderAgeSec}s`,
            `drift_ticks=${candidate.priceDriftTicks}`,
            `tte=${candidate.timeToExpirySec}s`,
            `limit=$${candidate.limitPrice.toFixed(2)}`,
            `current=$${candidate.currentPrice.toFixed(2)}`,
        ].join(' | ');
    }

    private computeOrderShares(amountUsd: number, price: number): number {
        if (!Number.isFinite(amountUsd) || amountUsd <= 0 || !Number.isFinite(price) || price <= 0) return 0;
        return Math.floor((amountUsd / price) * 100) / 100;
    }

    private classifyFixedGt5Preservation(
        requestedShares: number,
        finalShares: number,
        uplifted: boolean,
    ): Pick<FixedOrderExecutionPlan, 'gt5PreservationStatus' | 'compressionReason'> {
        if (requestedShares > 5 + 1e-9) {
            if (finalShares + 1e-9 < requestedShares) {
                return {
                    gt5PreservationStatus: 'contract_violation',
                    compressionReason: 'unexpected_floor_violation',
                };
            }
            return { gt5PreservationStatus: 'exact_preservation' };
        }
        return {
            gt5PreservationStatus: 'not_applicable',
            compressionReason: uplifted ? 'min_shares_floor' : undefined,
        };
    }

    private classifyDynamicGt5Preservation(
        requestedShares: number,
        finalShares: number,
        uplifted: boolean,
    ): Pick<DynamicRungExecutionPlan, 'gt5PreservationStatus' | 'compressionReason'> {
        if (requestedShares > 5 + 1e-9) {
            if (finalShares + 1e-9 < 5) {
                return {
                    gt5PreservationStatus: 'contract_violation',
                    compressionReason: 'unexpected_floor_violation',
                };
            }
            if (finalShares + 1e-9 < requestedShares) {
                return {
                    gt5PreservationStatus: 'compressed_but_not_floored',
                    compressionReason: 'budget_compression',
                };
            }
            return { gt5PreservationStatus: 'exact_preservation' };
        }
        return {
            gt5PreservationStatus: 'not_applicable',
            compressionReason: uplifted ? 'min_shares_floor' : undefined,
        };
    }

    private computeMinExecutableNotional(price: number, minShares: number): number {
        if (!Number.isFinite(price) || price <= 0 || !Number.isFinite(minShares) || minShares <= 0) return 0;
        let cents = Math.max(1, Math.ceil((price * minShares - 1e-9) * 100));
        let amountUsd = cents / 100;
        let shares = this.computeOrderShares(amountUsd, price);
        while (shares + 1e-9 < minShares) {
            cents += 1;
            amountUsd = cents / 100;
            shares = this.computeOrderShares(amountUsd, price);
        }
        return roundUsd(amountUsd);
    }

    private precheckSharesAgainstConstraint(
        shares: number,
        constraint: ExchangeExecutionConstraint,
    ): { ok: boolean; reason?: string } {
        const minShares = Number(constraint.minOrderSizeShares) || DEFAULT_EXCHANGE_MIN_SHARES;
        if (!Number.isFinite(shares) || shares <= 0) {
            return { ok: false, reason: 'shares 无效，无法提交真实订单' };
        }
        if (shares + 1e-9 < minShares) {
            return {
                ok: false,
                reason: `Size (${shares.toFixed(2)}) lower than the minimum: ${minShares.toFixed(2)}`,
            };
        }
        return { ok: true };
    }

    private precheckLiveShares(shares: number): { ok: boolean; reason?: string } {
        return this.precheckSharesAgainstConstraint(shares, this.buildFallbackExecutionConstraint());
    }

    private buildFixedOrderExecutionPlan(params: {
        requestedAmountUsd: number;
        price: number;
        maxBudgetUsd: number;
        constraint: ExchangeExecutionConstraint;
    }): FixedOrderExecutionPlan {
        const requestedAmountUsd = roundUsd(params.requestedAmountUsd);
        const requestedShares = this.computeOrderShares(requestedAmountUsd, params.price);
        const minExecutableNotional = this.computeMinExecutableNotional(params.price, params.constraint.minOrderSizeShares);
        const finalAmountUsd = roundUsd(Math.max(requestedAmountUsd, minExecutableNotional));
        const finalShares = this.computeOrderShares(finalAmountUsd, params.price);
        const uplifted = finalAmountUsd > requestedAmountUsd + 1e-9;
        const gt5Meta = this.classifyFixedGt5Preservation(requestedShares, finalShares, uplifted);
        if (finalAmountUsd > roundUsd(params.maxBudgetUsd) + 1e-9) {
            return {
                requestedAmountUsd,
                finalAmountUsd,
                requestedShares,
                finalShares,
                uplifted,
                upliftedNotionalUsd: roundUsd(Math.max(0, finalAmountUsd - requestedAmountUsd)),
                constraint: params.constraint,
                executable: false,
                gt5PreservationStatus: gt5Meta.gt5PreservationStatus,
                compressionReason: gt5Meta.compressionReason,
                failureReason: `fixed_order_min_shares_requires_$${finalAmountUsd.toFixed(2)}_but_budget_cap_is_$${roundUsd(params.maxBudgetUsd).toFixed(2)}`,
            };
        }
        if (gt5Meta.gt5PreservationStatus === 'contract_violation') {
            console.warn(`[gt5-contract] fixed order requested ${requestedShares.toFixed(2)} shares but final shares regressed to ${finalShares.toFixed(2)}`);
        }
        return {
            requestedAmountUsd,
            finalAmountUsd,
            requestedShares,
            finalShares,
            uplifted,
            upliftedNotionalUsd: roundUsd(Math.max(0, finalAmountUsd - requestedAmountUsd)),
            constraint: params.constraint,
            executable: true,
            gt5PreservationStatus: gt5Meta.gt5PreservationStatus,
            compressionReason: gt5Meta.compressionReason,
        };
    }

    private buildDynamicOrderExecutionPlan(params: {
        rungPrices: number[];
        rungBets: number[];
        maxBudgetUsd: number;
        constraint: ExchangeExecutionConstraint;
        executionCompressionMode?: string;
    }): DynamicOrderExecutionPlan {
        const allRungs: DynamicRungExecutionPlan[] = params.rungPrices.map((rungPrice, rungIndex) => {
            const originalBetUsd = roundUsd(params.rungBets[rungIndex] || 0);
            const requestedShares = this.computeOrderShares(originalBetUsd, rungPrice);
            const minExecutableNotionalUsd = this.computeMinExecutableNotional(rungPrice, params.constraint.minOrderSizeShares);
            const adjustedShares = this.computeOrderShares(minExecutableNotionalUsd, rungPrice);
            const uplifted = minExecutableNotionalUsd > originalBetUsd + 1e-9;
            const gt5Meta = this.classifyDynamicGt5Preservation(requestedShares, adjustedShares, uplifted);
            return {
                rungIndex,
                rungPrice,
                originalBetUsd,
                requestedShares,
                minExecutableNotionalUsd,
                redistributedNotionalUsd: 0,
                adjustedBetUsd: minExecutableNotionalUsd,
                adjustedShares,
                uplifted,
                kept: false,
                gt5PreservationStatus: gt5Meta.gt5PreservationStatus,
                compressionReason: gt5Meta.compressionReason,
            };
        });

        const requestedBudgetUsd = roundUsd(allRungs.reduce((sum, rung) => sum + rung.originalBetUsd, 0));
        const budgetCap = roundUsd(Math.max(0, Math.min(params.maxBudgetUsd, requestedBudgetUsd)));
        const budgetCapCents = Math.max(0, Math.round(budgetCap * 100));
        const byLowEnd = [...allRungs].sort((a, b) => a.rungPrice - b.rungPrice || a.rungIndex - b.rungIndex);
        let runningMinCents = 0;
        const keptLowPrefix: DynamicRungExecutionPlan[] = [];
        for (const rung of byLowEnd) {
            const rungMinCents = Math.round(rung.minExecutableNotionalUsd * 100);
            if (runningMinCents + rungMinCents <= budgetCapCents) {
                keptLowPrefix.push(rung);
                runningMinCents += rungMinCents;
            } else {
                break;
            }
        }

        const keepIdx = new Set<number>(keptLowPrefix.map((rung) => rung.rungIndex));
        const selectedRungs: DynamicRungExecutionPlan[] = allRungs
            .map((rung) => ({ ...rung, kept: keepIdx.has(rung.rungIndex) }))
            .filter((rung) => rung.kept);

        if (selectedRungs.length > 0) {
            const selectedLowOrder = [...selectedRungs].sort((a, b) => a.rungPrice - b.rungPrice || a.rungIndex - b.rungIndex);
            const remainingCents = Math.max(0, budgetCapCents - runningMinCents);
            const lateCompletionEnabled = String(params.executionCompressionMode || '').trim().toLowerCase() === 'reallocate_plus_late_completion';
            const selectedWeights = selectedLowOrder.map((rung, idx) => {
                const baseWeight = Math.max(1, Math.round(rung.originalBetUsd * 100));
                if (!lateCompletionEnabled || selectedLowOrder.length <= 1) {
                    return baseWeight;
                }
                const relativePosition = idx / Math.max(1, selectedLowOrder.length - 1);
                const aggressiveness = 1 + (0.35 * relativePosition);
                return Math.max(1, Math.round(baseWeight * aggressiveness));
            });
            const totalWeight = selectedWeights.reduce((sum, value) => sum + value, 0);
            const effectiveWeights = totalWeight > 0
                ? selectedWeights
                : selectedLowOrder.map(() => 1);
            const effectiveWeightTotal = effectiveWeights.reduce((sum, value) => sum + value, 0);
            const baseRedistributedCents = selectedLowOrder.map((_, idx) => Math.floor((remainingCents * effectiveWeights[idx]) / effectiveWeightTotal));
            let distributedCents = baseRedistributedCents.reduce((sum, value) => sum + value, 0);
            let cursor = 0;
            while (distributedCents < remainingCents && selectedLowOrder.length > 0) {
                baseRedistributedCents[cursor % selectedLowOrder.length] += 1;
                distributedCents += 1;
                cursor += 1;
            }

            for (let idx = 0; idx < selectedLowOrder.length; idx++) {
                const redistributedNotionalUsd = roundUsd(baseRedistributedCents[idx] / 100);
                const adjustedBetUsd = roundUsd(selectedLowOrder[idx].minExecutableNotionalUsd + redistributedNotionalUsd);
                const adjustedShares = this.computeOrderShares(adjustedBetUsd, selectedLowOrder[idx].rungPrice);
                const uplifted = adjustedBetUsd > selectedLowOrder[idx].originalBetUsd + 1e-9;
                const gt5Meta = this.classifyDynamicGt5Preservation(
                    selectedLowOrder[idx].requestedShares,
                    adjustedShares,
                    uplifted,
                );
                if (gt5Meta.gt5PreservationStatus === 'contract_violation') {
                    console.warn(
                        `[gt5-contract] dynamic rung ${selectedLowOrder[idx].rungIndex} requested ${selectedLowOrder[idx].requestedShares.toFixed(2)} shares but final shares regressed to ${adjustedShares.toFixed(2)}`,
                    );
                }
                selectedLowOrder[idx] = {
                    ...selectedLowOrder[idx],
                    redistributedNotionalUsd,
                    adjustedBetUsd,
                    adjustedShares,
                    uplifted,
                    gt5PreservationStatus: gt5Meta.gt5PreservationStatus,
                    compressionReason: gt5Meta.compressionReason,
                };
            }

            const updatedByIndex = new Map<number, DynamicRungExecutionPlan>(selectedLowOrder.map((rung) => [rung.rungIndex, rung]));
            for (let idx = 0; idx < selectedRungs.length; idx++) {
                const updated = updatedByIndex.get(selectedRungs[idx].rungIndex);
                if (updated) selectedRungs[idx] = updated;
            }
        }

        const selectedByIndex = new Map<number, DynamicRungExecutionPlan>(selectedRungs.map((rung) => [rung.rungIndex, rung]));
        const finalizedAllRungs = allRungs.map((rung) => selectedByIndex.get(rung.rungIndex) || rung);

        const upliftedDynamicRungs = selectedRungs.filter((rung) => rung.uplifted).length;
        const keptDynamicRungs = selectedRungs.length;
        const droppedDynamicRungs = allRungs.length - keptDynamicRungs;
        const totalSelectedNotionalUsd = roundUsd(selectedRungs.reduce((sum, rung) => sum + rung.adjustedBetUsd, 0));
        const totalUpliftedNotionalUsd = roundUsd(
            selectedRungs.reduce((sum, rung) => sum + Math.max(0, rung.adjustedBetUsd - rung.originalBetUsd), 0),
        );
        const decision = keptDynamicRungs === 0
            ? 'skipped_precheck_min_shares_budget_insufficient'
            : droppedDynamicRungs > 0
                ? 'compressed_dynamic_rungs'
                : 'dynamic_min_shares_satisfied';
        const failureReason = keptDynamicRungs === 0
            ? `dynamic_low_end_prefix_min_shares_requires_$${(keptLowPrefix[0]?.minExecutableNotionalUsd ?? byLowEnd[0]?.minExecutableNotionalUsd ?? 0).toFixed(2)}_but_budget_cap_is_$${budgetCap.toFixed(2)}`
            : undefined;

        return {
            constraint: params.constraint,
            attemptedRungs: allRungs.length,
            upliftedDynamicRungs,
            keptDynamicRungs,
            droppedDynamicRungs,
            selectedRungs,
            allRungs: finalizedAllRungs,
            requestedBudgetUsd,
            totalSelectedNotionalUsd,
            totalUpliftedNotionalUsd,
            executionCompressionUsd: roundUsd(Math.max(0, requestedBudgetUsd - totalSelectedNotionalUsd)),
            executionCompressionReason: keptDynamicRungs === 0
                ? 'no_executable_rungs_after_min_share_check'
                : String(params.executionCompressionMode || 'reallocate_without_notional_loss').trim().toLowerCase() || 'reallocate_without_notional_loss',
            decision,
            failureReason,
        };
    }

    private collapseSelectedRungsByPrice(selectedRungs: DynamicRungExecutionPlan[]): DynamicRungExecutionPlan[] {
        const priority: Record<string, number> = {
            contract_violation: 3,
            compressed_but_not_floored: 2,
            exact_preservation: 1,
            not_applicable: 0,
        };
        const grouped = new Map<string, DynamicRungExecutionPlan>();
        for (const rung of selectedRungs) {
            const key = this.roundTokenPriceTick(rung.rungPrice).toFixed(2);
            const existing = grouped.get(key);
            if (!existing) {
                grouped.set(key, { ...rung, rungPrice: Number(key) });
                continue;
            }
            const gt5PreservationStatus = priority[rung.gt5PreservationStatus] > priority[existing.gt5PreservationStatus]
                ? rung.gt5PreservationStatus
                : existing.gt5PreservationStatus;
            const compressionReasonSet = new Set<string>(
                [existing.compressionReason, rung.compressionReason]
                    .map((value) => String(value || '').trim())
                    .filter(Boolean),
            );
            grouped.set(key, {
                ...existing,
                originalBetUsd: roundUsd(existing.originalBetUsd + rung.originalBetUsd),
                requestedShares: existing.requestedShares + rung.requestedShares,
                minExecutableNotionalUsd: roundUsd(existing.minExecutableNotionalUsd + rung.minExecutableNotionalUsd),
                redistributedNotionalUsd: roundUsd(existing.redistributedNotionalUsd + rung.redistributedNotionalUsd),
                adjustedBetUsd: roundUsd(existing.adjustedBetUsd + rung.adjustedBetUsd),
                adjustedShares: existing.adjustedShares + rung.adjustedShares,
                uplifted: existing.uplifted || rung.uplifted,
                gt5PreservationStatus,
                compressionReason: compressionReasonSet.size > 0 ? Array.from(compressionReasonSet).join('+') : undefined,
            });
        }
        return [...grouped.values()].sort((a, b) => a.rungPrice - b.rungPrice || a.rungIndex - b.rungIndex);
    }

    private addExecutionSkippedEntry(params: {
        symbol: string;
        direction: 'UP' | 'DOWN';
        confidence: number;
        reason: string;
        executionDecision: string;
        requestedAmountUsd: number;
        targetNotionalUsd?: number;
        requestedNotionalUsd?: number;
        postedNotionalUsd?: number;
        filledNotionalUsd?: number;
        requestCompressionMode?: string;
        executionCompressionMode?: string;
        requestLayerCompressionUsd?: number;
        executionLayerCompressionUsd?: number;
        executionLayerCompressionReason?: string;
        requestedShares?: number;
        originalRequestedShares?: number;
        market: { title: string; conditionId: string; slug: string };
        token: { tokenId: string; outcome: string };
        exchangeMinOrderSizeShares?: number;
        attemptedRungs?: number;
        compressedRungs?: number;
        postedRungs?: number;
        filledRungs?: number;
        exchangeRejectedRungs?: number;
        upliftedDynamicRungs?: number;
        keptDynamicRungs?: number;
        droppedDynamicRungs?: number;
        minExecutableNotionalUsd?: number;
        redistributedNotionalUsd?: number;
        upliftedFixedOrders?: number;
        upliftedNotionalUsd?: number;
        gt5PreservationStatus?: string;
        compressionReason?: string;
        virtualRungCount?: number;
        materializedLimitPriceLadder?: number[];
        collapsedSamePriceRungs?: number;
        gt5AutoReduceTriggered?: boolean;
        runtimeRiskScalingState?: string;
        directionalSizingSource?: string;
        sizingMode?: string;
        bankrollPct?: number;
        confidenceBandScale?: number;
        effectiveSizingPct?: number;
        currentCapital?: number;
    }): void {
        const sizingTrace = this.buildSizingTrace(params.confidence, params.symbol, params.direction);
        const reasonText = extractErrorText(params.reason);
        const executionLayerCompressionReason = params.executionLayerCompressionReason == null
            ? undefined
            : extractErrorText(params.executionLayerCompressionReason);
        const sizingMode = params.sizingMode ?? sizingTrace.sizingMode;
        const bankrollPct = Number.isFinite(Number(params.bankrollPct))
            ? Number(params.bankrollPct)
            : sizingTrace.bankrollPct;
        const confidenceBandScale = Number.isFinite(Number(params.confidenceBandScale))
            ? Number(params.confidenceBandScale)
            : sizingTrace.confidenceBandScale;
        const effectiveSizingPct = Number.isFinite(Number(params.effectiveSizingPct))
            ? Number(params.effectiveSizingPct)
            : sizingTrace.effectiveSizingPct;
        const currentCapital = Number.isFinite(Number(params.currentCapital))
            ? roundUsd(Number(params.currentCapital))
            : sizingTrace.currentCapital;
        this.logger.addEntry({
            symbol: params.symbol,
            direction: params.direction,
            confidence: params.confidence,
            amount: 0,
            mode: this.env.TRADING_MODE,
            status: 'failed',
            marketTitle: params.market.title,
            conditionId: params.market.conditionId,
            tokenId: params.token.tokenId,
            marketSlug: params.market.slug,
            tokenOutcome: params.token.outcome,
            requestedAmountUsd: roundUsd(params.requestedAmountUsd),
            targetNotionalUsd: Number.isFinite(Number(params.targetNotionalUsd)) ? roundUsd(Number(params.targetNotionalUsd)) : undefined,
            requestedNotionalUsd: Number.isFinite(Number(params.requestedNotionalUsd)) ? roundUsd(Number(params.requestedNotionalUsd)) : undefined,
            postedNotionalUsd: Number.isFinite(Number(params.postedNotionalUsd)) ? roundUsd(Number(params.postedNotionalUsd)) : undefined,
            filledNotionalUsd: Number.isFinite(Number(params.filledNotionalUsd)) ? roundUsd(Number(params.filledNotionalUsd)) : undefined,
            requestCompressionMode: params.requestCompressionMode,
            executionCompressionMode: params.executionCompressionMode,
            requestLayerCompressionUsd: Number.isFinite(Number(params.requestLayerCompressionUsd)) ? roundUsd(Number(params.requestLayerCompressionUsd)) : undefined,
            executionLayerCompressionUsd: Number.isFinite(Number(params.executionLayerCompressionUsd)) ? roundUsd(Number(params.executionLayerCompressionUsd)) : undefined,
            executionLayerCompressionReason,
            pendingAmountUsd: 0,
            requestedShares: Number.isFinite(Number(params.requestedShares)) ? Number(params.requestedShares) : 0,
            originalRequestedShares: Number.isFinite(Number(params.originalRequestedShares))
                ? Number(params.originalRequestedShares)
                : Number.isFinite(Number(params.requestedShares))
                    ? Number(params.requestedShares)
                    : 0,
            matchedShares: 0,
            error: reasonText,
            lastRejectionReason: reasonText,
            familyRuleVersion: this.getFamilyRuleVersion() || undefined,
            lowpriceFamilyId: this.getLowpriceFamilyId() || undefined,
            stalePolicyDecision: this.getStaleOrderPolicy() === 'off' ? 'off' : 'kept',
            exchangeMinOrderSizeShares: params.exchangeMinOrderSizeShares,
            attemptedRungs: params.attemptedRungs,
            compressedRungs: params.compressedRungs ?? params.droppedDynamicRungs,
            postedRungs: params.postedRungs,
            filledRungs: params.filledRungs,
            exchangeRejectedRungs: params.exchangeRejectedRungs,
            upliftedDynamicRungs: params.upliftedDynamicRungs,
            keptDynamicRungs: params.keptDynamicRungs,
            droppedDynamicRungs: params.droppedDynamicRungs,
            minExecutableNotionalUsd: params.minExecutableNotionalUsd,
            redistributedNotionalUsd: params.redistributedNotionalUsd,
            upliftedFixedOrders: params.upliftedFixedOrders,
            upliftedNotionalUsd: params.upliftedNotionalUsd,
            executionDecision: params.executionDecision,
            gt5PreservationStatus: params.gt5PreservationStatus,
            compressionReason: params.compressionReason,
            virtualRungCount: params.virtualRungCount,
            materializedLimitPriceLadder: params.materializedLimitPriceLadder,
            collapsedSamePriceRungs: params.collapsedSamePriceRungs,
            gt5AutoReduceTriggered: params.gt5AutoReduceTriggered,
            runtimeRiskScalingState: params.runtimeRiskScalingState,
            directionalSizingSource: params.directionalSizingSource,
            sizingMode,
            bankrollPct,
            confidenceBandScale,
            effectiveSizingPct,
            currentCapital,
            lastNonExecutionReason: reasonText,
        });
    }

    private markLiveTradeRejected(
        logEntry: TradeLogEntry,
        reason: string,
        requestedAmountUsd: number,
        requestedShares: number,
        tokenId?: string,
        marketSlug?: string,
    ): void {
        this.rememberLiveExchangeConstraints(reason, tokenId, marketSlug);
        const constraint = tokenId
            ? (this.executionConstraintByToken.get(tokenId) || this.buildFallbackExecutionConstraint(tokenId, marketSlug))
            : this.buildFallbackExecutionConstraint(undefined, marketSlug);
        this.logger.updateEntry(logEntry.id, {
            status: 'failed',
            error: reason,
            amount: 0,
            requestedAmountUsd: roundUsd(requestedAmountUsd),
            pendingAmountUsd: 0,
            requestedShares,
            originalRequestedShares: logEntry.originalRequestedShares ?? requestedShares,
            matchedShares: 0,
            liveOrderStatus: 'rejected',
            lastRejectionReason: reason,
            familyRuleVersion: this.getFamilyRuleVersion() || logEntry.familyRuleVersion,
            lowpriceFamilyId: this.getLowpriceFamilyId() || logEntry.lowpriceFamilyId,
            stalePolicyDecision: this.getStaleOrderPolicy() === 'off' ? 'off' : 'kept',
            exchangeMinOrderSizeShares: constraint.minOrderSizeShares,
            exchangeRejectedRungs: 1,
            executionDecision: 'exchange_rejected',
            gt5PreservationStatus: logEntry.gt5PreservationStatus,
            compressionReason: logEntry.compressionReason,
            lastNonExecutionReason: reason,
        });
    }

    private addRejectedLiveRungEntry(params: {
        symbol: string;
        direction: 'UP' | 'DOWN';
        confidence: number;
        reason: string;
        requestedAmountUsd: number;
        requestedShares: number;
        rungPrice: number;
        market: { title: string; conditionId: string; slug: string };
        token: { tokenId: string; outcome: string };
    }): void {
        this.rememberLiveExchangeConstraints(params.reason, params.token.tokenId, params.market.slug);
        const constraint = this.executionConstraintByToken.get(params.token.tokenId)
            || this.buildFallbackExecutionConstraint(params.token.tokenId, params.market.slug);
        this.logger.addEntry({
            symbol: params.symbol,
            direction: params.direction,
            confidence: params.confidence,
            amount: 0,
            mode: this.env.TRADING_MODE,
            status: 'failed',
            marketTitle: params.market.title,
            conditionId: params.market.conditionId,
            tokenId: params.token.tokenId,
            marketSlug: params.market.slug,
            tokenOutcome: params.token.outcome,
            tokenPrice: params.rungPrice,
            limitPriceConfigured: params.rungPrice,
            requestedAmountUsd: roundUsd(params.requestedAmountUsd),
            pendingAmountUsd: 0,
            requestedShares: params.requestedShares,
            matchedShares: 0,
            liveOrderStatus: 'rejected',
            lastRejectionReason: params.reason,
            familyRuleVersion: this.getFamilyRuleVersion() || undefined,
            lowpriceFamilyId: this.getLowpriceFamilyId() || undefined,
            stalePolicyDecision: this.getStaleOrderPolicy() === 'off' ? 'off' : 'kept',
            error: params.reason,
            exchangeMinOrderSizeShares: constraint.minOrderSizeShares,
            exchangeRejectedRungs: 1,
            executionDecision: 'exchange_rejected',
            lastNonExecutionReason: params.reason,
        });
    }

    async checkStaleLiveOrderPolicy(): Promise<{ scanned: number; cancelled: number }> {
        if (this.env.TRADING_MODE !== TradingMode.LIVE || !this.clobClient) {
            return { scanned: 0, cancelled: 0 };
        }
        const stalePolicyEnabled = this.getStaleOrderPolicy() === 'cancel_only_when_stale';
        const lateFillCancelMinutes = Math.max(0, Number((this.env as any).LATE_FILL_CANCEL_MINUTES) || 0);
        const lateFillCancelSec = lateFillCancelMinutes > 0 ? lateFillCancelMinutes * 60 : 0;
        if (!stalePolicyEnabled && lateFillCancelSec <= 0) {
            return { scanned: 0, cancelled: 0 };
        }

        const pendingEntries = this.logger.getAllEntries().filter((entry) => {
            const orderId = String(entry.orderId || entry.txHash || '').trim();
            if (!orderId || !entry.tokenId) return false;
            if (entry.mode !== TradingMode.LIVE) return false;
            const liveStatus = String(entry.liveOrderStatus || '').trim().toLowerCase();
            const staleDecision = String(entry.stalePolicyDecision || '').trim().toLowerCase();
            const isTerminalLocalState = [
                'cancelled',
                'official_canceled_zero_fill',
                'official_canceled_partial_fill',
                'filled',
                'rejected',
                'expired',
            ].includes(liveStatus);
            const alreadyTerminalByPolicy = [
                'cancelled',
                'late_fill_timeout_cancelled',
                'official_terminal_zero_fill',
                'official_terminal_with_fill',
            ].includes(staleDecision);
            if (isTerminalLocalState || alreadyTerminalByPolicy) return false;
            const pendingAmount = Number(entry.pendingAmountUsd ?? 0);
            return (
                entry.status === 'pending' ||
                pendingAmount > 0 ||
                liveStatus === 'posted_pending' ||
                liveStatus === 'official_open' ||
                liveStatus === 'partially_filled'
            );
        }).sort((a, b) => {
            const ta = Number.isFinite(new Date(a.timestamp).getTime()) ? new Date(a.timestamp).getTime() : 0;
            const tb = Number.isFinite(new Date(b.timestamp).getTime()) ? new Date(b.timestamp).getTime() : 0;
            return tb - ta;
        });
        if (pendingEntries.length === 0) {
            return { scanned: 0, cancelled: 0 };
        }

        const nowMs = Date.now();
        const nowSec = Math.floor(nowMs / 1000);
        const minAgeSec = Math.max(0, Number(this.env.STALE_CANCEL_MIN_AGE_SEC || 0));
        const minDriftTicks = Math.max(0, Number(this.env.STALE_CANCEL_MIN_DRIFT_TICKS || 0));
        const minTimeToExpirySec = Math.max(0, Number(this.env.STALE_CANCEL_MIN_TIME_TO_EXPIRY_SEC || 0));
        const priceCache = new Map<string, number | null>();
        let cancelled = 0;

        for (const entry of pendingEntries) {
            const orderId = String(entry.orderId || entry.txHash || '').trim();
            if (!orderId || !entry.tokenId) continue;

            const createdAtMs = Number.isFinite(new Date(entry.timestamp).getTime())
                ? new Date(entry.timestamp).getTime()
                : nowMs;
            const orderAgeSec = Math.max(0, Math.floor((nowMs - createdAtMs) / 1000));
            const marketStartTs = parsePeriodStartFromSlug(entry.marketSlug);
            const fallbackExpirySec = marketStartTs != null
                ? Math.floor(marketStartTs + this.periodSeconds - 60)
                : Math.floor(createdAtMs / 1000) + this.periodSeconds;
            const tracker = this.pendingLiveOrders.get(orderId);
            const expirationSec = tracker?.expirationSec || fallbackExpirySec;
            const timeToExpirySec = Math.max(0, expirationSec - nowSec);
            const lateFillTimeoutTriggered = lateFillCancelSec > 0 && orderAgeSec >= lateFillCancelSec;
            if (!lateFillTimeoutTriggered) {
                if (!stalePolicyEnabled) continue;
                if (orderAgeSec < minAgeSec || timeToExpirySec < minTimeToExpirySec) continue;
            }

            const limitPrice = Number(entry.limitPriceConfigured ?? entry.tokenPrice ?? 0);
            if (!Number.isFinite(limitPrice) || limitPrice <= 0) continue;

            if (lateFillTimeoutTriggered) {
                const candidate: StaleLiveOrderCandidate = {
                    entry,
                    orderId,
                    limitPrice,
                    orderAgeSec,
                    timeToExpirySec,
                    currentPrice: Number(entry.bestAskAtEntry ?? limitPrice) || limitPrice,
                    priceDriftTicks: 0,
                    cancelReasonKind: 'late_fill_timeout',
                    lateFillCancelMinutes,
                };
                const didCancel = await this.cancelStaleLiveOrder(candidate);
                if (didCancel) cancelled += 1;
                continue;
            }

            let currentPrice = priceCache.get(entry.tokenId);
            if (typeof currentPrice === 'undefined') {
                try {
                    currentPrice = await getTokenPrice(entry.tokenId, 'BUY');
                } catch {
                    currentPrice = null;
                }
                priceCache.set(entry.tokenId, currentPrice ?? null);
            }
            if (currentPrice == null || !Number.isFinite(currentPrice) || currentPrice <= 0) continue;

            const priceDriftTicks = Math.max(0, Math.floor(((limitPrice - currentPrice) / 0.01) + 1e-9));
            if (priceDriftTicks < minDriftTicks) continue;

            const candidate: StaleLiveOrderCandidate = {
                entry,
                orderId,
                limitPrice,
                orderAgeSec,
                timeToExpirySec,
                currentPrice,
                priceDriftTicks,
            };
            const didCancel = await this.cancelStaleLiveOrder(candidate);
            if (didCancel) cancelled += 1;
        }

        return { scanned: pendingEntries.length, cancelled };
    }

    private async cancelStaleLiveOrder(candidate: StaleLiveOrderCandidate): Promise<boolean> {
        if (!this.clobClient) return false;
        const officialState = await this.queryOfficialOrderLifecycle(candidate.orderId);
        if (officialState) {
            if (this.isOfficialTerminalZeroFill(officialState)) {
                this.markOfficialTerminalZeroFill(candidate.entry, candidate.orderId, officialState);
                console.log(`  🧹 skip cancel ${candidate.entry.symbol} ${candidate.orderId.slice(0, 12)}... official=${officialState.status} zero-fill`);
                return false;
            }
            if (this.isOfficialTerminalWithFill(officialState)) {
                this.markOfficialTerminalWithFill(candidate.entry, candidate.orderId, officialState);
                console.log(`  🧹 skip cancel ${candidate.entry.symbol} ${candidate.orderId.slice(0, 12)}... official=${officialState.status} matched=${officialState.matchedSize}`);
                return false;
            }
        }
        try {
            await this.clobClient.cancelOrder({ orderID: candidate.orderId });
        } catch (error) {
            const reason = parseErrorReason(error);
            const normalized = reason.toLowerCase();
            if (
                normalized.includes('filled') ||
                normalized.includes('matched') ||
                normalized.includes('executed')
            ) {
                return false;
            }
            if (
                normalized.includes('already canceled') ||
                normalized.includes('already cancelled')
            ) {
                // treat as terminal cancel state
                this.markOfficialTerminalZeroFill(candidate.entry, candidate.orderId, {
                    status: 'CANCELED',
                    matchedSize: 0,
                    reason: 'official_already_canceled_zero_fill',
                });
                return false;
            } else {
                console.log(`  ⚠️ stale cancel 失败 ${candidate.orderId}: ${reason}`);
                return false;
            }
        }

        const reason = this.buildStaleCancelReason(candidate);
        const tracker = this.pendingLiveOrders.get(candidate.orderId);
        if (tracker?.resolve) {
            tracker.resolve({
                amountUsd: roundUsd(tracker.filledUsd || 0),
                tokens: Number(tracker.sizeFilled || 0),
                source: 'STALE_CANCELLED',
            });
        }
        this.pendingLiveOrders.delete(candidate.orderId);
        this.logger.updateEntry(candidate.entry.id, {
            status: 'failed',
            error: reason,
            pendingAmountUsd: 0,
            liveOrderStatus: 'cancelled',
            lastRejectionReason: reason,
            familyRuleVersion: this.getFamilyRuleVersion() || candidate.entry.familyRuleVersion,
            lowpriceFamilyId: this.getLowpriceFamilyId() || candidate.entry.lowpriceFamilyId,
            stalePolicyDecision: candidate.cancelReasonKind === 'late_fill_timeout'
                ? 'late_fill_timeout_cancelled'
                : 'cancelled',
            staleCancelReason: reason,
            lateFillCancelMinutes: candidate.cancelReasonKind === 'late_fill_timeout'
                ? candidate.lateFillCancelMinutes
                : candidate.entry.lateFillCancelMinutes,
        });
        this.appendPendingOrderLedgerEvent({
            event: 'cancelled',
            orderId: candidate.orderId,
            symbol: candidate.entry.symbol,
            direction: candidate.entry.direction,
            amountUsd: 0,
            remainingFrozenAmount: Number(candidate.entry.pendingAmountUsd ?? candidate.entry.requestedAmountUsd ?? candidate.entry.amount ?? 0) || 0,
            limitPrice: Number(candidate.entry.limitPriceConfigured || candidate.entry.tokenPrice || candidate.limitPrice || 0),
            tokenId: candidate.entry.tokenId || '',
            tokenOutcome: candidate.entry.tokenOutcome || (candidate.entry.direction === 'UP' ? 'Up' : 'Down'),
            confidence: candidate.entry.confidence,
            marketTitle: candidate.entry.marketTitle || '',
            conditionId: candidate.entry.conditionId || '',
            marketSlug: candidate.entry.marketSlug || '',
            targetPeriodEndTs: parsePeriodStartFromSlug(candidate.entry.marketSlug) || 0,
            createdAt: candidate.entry.timestamp,
            expiresAt: new Date().toISOString(),
            bestAsk: candidate.entry.bestAskAtEntry ?? candidate.entry.tokenPrice,
            avgFillPrice: undefined,
            queueCompetitionUsdAtEntry: candidate.entry.queueCompetitionUsdAtEntry,
            effectiveFillableUsdAtEntry: candidate.entry.effectiveFillableUsdAtEntry,
        });
        const label = candidate.cancelReasonKind === 'late_fill_timeout' ? 'late-fill cancel' : 'stale cancel';
        console.log(`  🧹 ${label} ${candidate.entry.symbol} ${candidate.orderId.slice(0, 12)}... limit=$${candidate.limitPrice.toFixed(2)} current=$${candidate.currentPrice.toFixed(2)} drift=${candidate.priceDriftTicks}t age=${candidate.orderAgeSec}s`);
        return true;
    }

    private appendLiveFillLedgerEvent(
        entry: TradeLogEntry,
        fillAmountUsd: number,
        remainingPendingUsd: number,
        fillPrice: number | null,
        event: PendingOrderLedgerEventType,
    ): void {
        const orderId = String(entry.orderId || entry.txHash || entry.id);
        if (!orderId || !entry.conditionId || !entry.tokenId || !entry.marketSlug || !entry.tokenOutcome || !entry.marketTitle) {
            return;
        }
        this.appendPendingOrderLedgerEvent({
            event,
            orderId,
            symbol: entry.symbol,
            direction: entry.direction,
            amountUsd: fillAmountUsd,
            remainingFrozenAmount: remainingPendingUsd,
            limitPrice: Number(entry.limitPriceConfigured || entry.tokenPrice || fillPrice || 0),
            tokenId: entry.tokenId,
            tokenOutcome: entry.tokenOutcome,
            confidence: entry.confidence,
            marketTitle: entry.marketTitle,
            conditionId: entry.conditionId,
            marketSlug: entry.marketSlug,
            targetPeriodEndTs: parsePeriodStartFromSlug(entry.marketSlug) || 0,
            createdAt: entry.timestamp,
            expiresAt: entry.timestamp,
            bestAsk: entry.bestAskAtEntry ?? entry.tokenPrice,
            avgFillPrice: fillPrice ?? entry.avgActualFillPrice ?? entry.tokenPrice,
            queueCompetitionUsdAtEntry: entry.queueCompetitionUsdAtEntry,
            effectiveFillableUsdAtEntry: entry.effectiveFillableUsdAtEntry,
        });
    }

    private async reconcilePendingLiveTrades(): Promise<{ reconciledCount: number; externalTradeCount: number; pendingAfter: number }> {
        if (this.env.TRADING_MODE !== TradingMode.LIVE) {
            return { reconciledCount: 0, externalTradeCount: 0, pendingAfter: 0 };
        }
        const currentEntries = this.logger.getAllEntries();
        const pendingEntries = currentEntries
            .filter((entry) => (entry.status === 'pending' || entry.status === 'failed') && ((entry.conditionId && entry.conditionId.trim()) || (entry.marketSlug && entry.marketSlug.trim())))
            .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
        if (pendingEntries.length === 0) {
            return { reconciledCount: 0, externalTradeCount: 0, pendingAfter: 0 };
        }

        const publicTrades = await this.fetchPublicActivityTrades();
        const executedTxHashes = new Set(
            currentEntries
                .filter((entry) => entry.status === 'executed')
                .map((entry) => String(entry.transactionHash || entry.txHash || '').trim().toLowerCase())
                .filter(Boolean),
        );

        const publicTradesByGroup = new Map<string, PublicActivityTrade[]>();
        const publicTradesByCondition = new Map<string, PublicActivityTrade[]>();
        for (const trade of publicTrades) {
            const key = this.buildLiveTradeGroupKey(trade.conditionId, trade.marketSlug);
            if (!publicTradesByGroup.has(key)) publicTradesByGroup.set(key, []);
            publicTradesByGroup.get(key)!.push(trade);
            const conditionKey = this.buildLiveTradeConditionKey(trade.conditionId);
            if (conditionKey) {
                if (!publicTradesByCondition.has(conditionKey)) publicTradesByCondition.set(conditionKey, []);
                publicTradesByCondition.get(conditionKey)!.push(trade);
            }
        }

        const pendingByGroup = new Map<string, TradeLogEntry[]>();
        for (const entry of pendingEntries) {
            const key = this.buildLiveTradeGroupKey(entry.conditionId, entry.marketSlug);
            if (!pendingByGroup.has(key)) pendingByGroup.set(key, []);
            pendingByGroup.get(key)!.push(entry);
        }

        let reconciledCount = 0;
        for (const [groupKey, groupPendingEntries] of pendingByGroup.entries()) {
            const groupPublicTrades = this.getPublicTradesForLocalRows(
                groupPendingEntries,
                publicTradesByGroup,
                publicTradesByCondition,
                groupKey,
            )
                .filter((trade) => !executedTxHashes.has(trade.transactionHash.toLowerCase()))
                .sort((a, b) => a.timestampSec - b.timestampSec || a.transactionHash.localeCompare(b.transactionHash));
            let publicIndex = 0;
            const reconciledPendingIds = new Set<string>();
            for (const pendingEntry of groupPendingEntries) {
                const orderId = String(pendingEntry.orderId || pendingEntry.txHash || '').trim();
                const requestedAmountUsd = Number(
                    pendingEntry.requestedAmountUsd ?? pendingEntry.pendingAmountUsd ?? pendingEntry.amount ?? 0,
                );
                const requestedShares = Number(pendingEntry.requestedShares ?? 0);
                const priceHint = Number(pendingEntry.limitPriceConfigured || pendingEntry.tokenPrice || 0);

                let resolvedAmountUsd = -1;
                let resolvedTokens = 0;
                let fillSource = '';
                let fillPrice: number | null = null;
                let fillTimestamp = pendingEntry.timestamp;
                let transactionHash = '';

                if (orderId) {
                    const fill = await this.queryActualFill(orderId, pendingEntry.conditionId, priceHint > 0 ? priceHint : 1, requestedAmountUsd);
                    if (fill.source === 'official_canceled_zero_fill') {
                        this.logger.updateEntry(pendingEntry.id, {
                            status: 'failed',
                            amount: 0,
                            pendingAmountUsd: 0,
                            matchedShares: 0,
                            liveOrderStatus: 'official_canceled_zero_fill',
                            fillSource: fill.source,
                            lastRejectionReason: 'official_canceled_zero_fill',
                            lastNonExecutionReason: 'official_canceled_zero_fill',
                            error: 'official_canceled_zero_fill',
                        });
                        this.appendPendingOrderLedgerEvent({
                            event: 'official_canceled_zero_fill',
                            orderId: orderId || pendingEntry.orderId || '',
                            symbol: pendingEntry.symbol,
                            direction: pendingEntry.direction,
                            amountUsd: 0,
                            remainingFrozenAmount: 0,
                            limitPrice: priceHint > 0 ? priceHint : Number(pendingEntry.tokenPrice || 0),
                            tokenId: pendingEntry.tokenId || '',
                            tokenOutcome: pendingEntry.tokenOutcome || '',
                            confidence: Number(pendingEntry.confidence || 0),
                            marketTitle: pendingEntry.marketTitle || '',
                            conditionId: pendingEntry.conditionId || '',
                            marketSlug: pendingEntry.marketSlug || '',
                            targetPeriodEndTs: parsePeriodStartFromSlug(pendingEntry.marketSlug) || 0,
                            createdAt: pendingEntry.timestamp,
                            expiresAt: String((pendingEntry as any).expiresAt || pendingEntry.timestamp),
                            avgFillPrice: undefined,
                            terminalTimestampSource: 'official_get_order_canceled',
                        });
                        reconciledPendingIds.add(pendingEntry.id);
                        reconciledCount += 1;
                        continue;
                    }
                    if (fill.amountUsd >= 0) {
                        resolvedAmountUsd = fill.amountUsd;
                        resolvedTokens = fill.tokens;
                        fillSource = fill.source;
                    }
                }

                let matchedPublicTrade: PublicActivityTrade | undefined;
                while (publicIndex < groupPublicTrades.length) {
                    const candidate = groupPublicTrades[publicIndex++];
                    if (executedTxHashes.has(candidate.transactionHash.toLowerCase())) continue;
                    matchedPublicTrade = candidate;
                    transactionHash = candidate.transactionHash;
                    fillPrice = candidate.price > 0 ? candidate.price : fillPrice;
                    fillTimestamp = candidate.timestampIso || fillTimestamp;
                    // 公共 activity 是 live 成交金额/价格的更高优先级事实源。
                    // 不能保留 getOrder * 本地挂单价 的近似值，否则会把更优成交价放大成 intended notional。
                    resolvedAmountUsd = roundUsd(candidate.usdcSize);
                    resolvedTokens = candidate.size;
                    fillSource = 'public_activity_api';
                    break;
                }

                if (!(resolvedAmountUsd >= 0)) {
                    continue;
                }

                const isPartialFill = requestedShares > 0 && resolvedTokens > 0 && resolvedTokens + 0.01 < requestedShares;
                const remainingPendingUsd = isPartialFill && requestedAmountUsd > 0
                    ? Math.max(0, roundUsd(requestedAmountUsd - resolvedAmountUsd))
                    : 0;

                this.logger.updateEntry(pendingEntry.id, {
                    timestamp: fillTimestamp,
                    status: 'executed',
                    amount: roundUsd(Math.max(0, resolvedAmountUsd)),
                    orderId: orderId || pendingEntry.orderId,
                    txHash: transactionHash || pendingEntry.txHash,
                    transactionHash: transactionHash || pendingEntry.transactionHash,
                    tokenId: matchedPublicTrade?.tokenId || pendingEntry.tokenId,
                    tokenOutcome: matchedPublicTrade?.tokenOutcome || pendingEntry.tokenOutcome,
                    tokenPrice: fillPrice ?? pendingEntry.tokenPrice,
                    avgActualFillPrice: fillPrice ?? pendingEntry.avgActualFillPrice ?? pendingEntry.tokenPrice,
                    requestedAmountUsd: requestedAmountUsd || pendingEntry.requestedAmountUsd,
                    pendingAmountUsd: remainingPendingUsd,
                    requestedShares: requestedShares || pendingEntry.requestedShares,
                    matchedShares: resolvedTokens > 0 ? resolvedTokens : pendingEntry.matchedShares,
                    liveOrderStatus: isPartialFill ? 'partially_filled' : 'filled',
                    fillSource: fillSource || pendingEntry.fillSource,
                    lastRejectionReason: undefined,
                    lastNonExecutionReason: undefined,
                    error: undefined,
                    result: pendingEntry.result || 'pending',
                });
                this.appendExecutionV2SettlementEventFromLog(
                    pendingEntry.id,
                    isPartialFill ? 'chase_reprice_filled' : 'fully_filled',
                    isPartialFill ? 'reconciled_partial_fill' : 'reconciled_full_fill',
                );
                this.appendLiveFillLedgerEvent(
                    pendingEntry,
                    roundUsd(Math.max(0, resolvedAmountUsd)),
                    remainingPendingUsd,
                    fillPrice,
                    isPartialFill ? 'partial_fill' : 'filled',
                );
                reconciledPendingIds.add(pendingEntry.id);
                if (transactionHash) {
                    executedTxHashes.add(transactionHash.toLowerCase());
                }
                reconciledCount += 1;
            }

            const unresolvedPendingEntries = groupPendingEntries.filter((entry) => !reconciledPendingIds.has(entry.id));
            if (unresolvedPendingEntries.length > 0) {
                const latePublicTrades = this.getPublicTradesForLocalRows(
                    unresolvedPendingEntries,
                    publicTradesByGroup,
                    publicTradesByCondition,
                    groupKey,
                ).filter((trade) => !executedTxHashes.has(trade.transactionHash.toLowerCase()));
                if (latePublicTrades.length > 0) {
                    // 如果官方 activity 已经能看到同市场同方向成交，就必须直接回写。
                    // 旧逻辑这里只是 continue，导致真钱已成交但本地仍显示 posted_pending。
                    const sortedLateTrades = latePublicTrades
                        .slice()
                        .sort((a, b) => a.timestampSec - b.timestampSec || a.transactionHash.localeCompare(b.transactionHash));
                    for (let idx = 0; idx < sortedLateTrades.length; idx++) {
                        const publicRow = sortedLateTrades[idx];
                        const targetEntry = unresolvedPendingEntries[Math.min(idx, unresolvedPendingEntries.length - 1)];
                        const nextAmount = roundUsd(publicRow.usdcSize);
                        const nextPrice = publicRow.price > 0
                            ? publicRow.price
                            : Number(targetEntry.avgActualFillPrice ?? targetEntry.tokenPrice ?? 0);
                        const nextShares = Number(publicRow.size || targetEntry.matchedShares || 0);
                        const nextTx = String(publicRow.transactionHash || targetEntry.transactionHash || targetEntry.txHash || '').trim();
                        if (idx < unresolvedPendingEntries.length) {
                            this.logger.updateEntry(targetEntry.id, {
                                timestamp: publicRow.timestampIso || targetEntry.timestamp,
                                status: 'executed',
                                amount: nextAmount,
                                orderId: targetEntry.orderId,
                                txHash: nextTx || targetEntry.txHash,
                                transactionHash: nextTx || targetEntry.transactionHash,
                                tokenId: publicRow.tokenId || targetEntry.tokenId,
                                tokenOutcome: publicRow.tokenOutcome || targetEntry.tokenOutcome,
                                tokenPrice: nextPrice > 0 ? nextPrice : targetEntry.tokenPrice,
                                avgActualFillPrice: nextPrice > 0 ? nextPrice : targetEntry.avgActualFillPrice ?? targetEntry.tokenPrice,
                                requestedAmountUsd: Number(targetEntry.requestedAmountUsd ?? nextAmount),
                                pendingAmountUsd: 0,
                                requestedShares: Number(targetEntry.requestedShares ?? nextShares),
                                matchedShares: nextShares > 0 ? nextShares : targetEntry.matchedShares,
                                liveOrderStatus: 'filled',
                                fillSource: 'public_activity_api',
                                lastRejectionReason: undefined,
                                lastNonExecutionReason: undefined,
                                error: undefined,
                                result: targetEntry.result || 'pending',
                            });
                            this.appendLiveFillLedgerEvent(
                                targetEntry,
                                nextAmount,
                                0,
                                nextPrice > 0 ? nextPrice : null,
                                'filled',
                            );
                        } else {
                            this.logger.addEntry({
                                symbol: targetEntry.symbol,
                                direction: targetEntry.direction,
                                confidence: Number(targetEntry.confidence || 0),
                                amount: nextAmount,
                                mode: this.modeScope,
                                status: 'executed',
                                marketTitle: targetEntry.marketTitle,
                                conditionId: publicRow.conditionId || targetEntry.conditionId,
                                tokenId: publicRow.tokenId || targetEntry.tokenId,
                                txHash: nextTx,
                                orderId: targetEntry.orderId,
                                transactionHash: nextTx,
                                result: targetEntry.result || 'pending',
                                marketSlug: publicRow.marketSlug || targetEntry.marketSlug,
                                tokenOutcome: publicRow.tokenOutcome || targetEntry.tokenOutcome,
                                tokenPrice: nextPrice > 0 ? nextPrice : targetEntry.tokenPrice,
                                avgActualFillPrice: nextPrice > 0 ? nextPrice : targetEntry.avgActualFillPrice ?? targetEntry.tokenPrice,
                                limitPriceConfigured: targetEntry.limitPriceConfigured,
                                requestedAmountUsd: nextAmount,
                                pendingAmountUsd: 0,
                                requestedShares: nextShares,
                                matchedShares: nextShares,
                                liveOrderStatus: 'filled',
                                fillSource: 'public_activity_api',
                                familyRuleVersion: targetEntry.familyRuleVersion,
                                lowpriceFamilyId: targetEntry.lowpriceFamilyId,
                                stalePolicyDecision: targetEntry.stalePolicyDecision,
                                exchangeMinOrderSizeShares: targetEntry.exchangeMinOrderSizeShares,
                                attemptedRungs: targetEntry.attemptedRungs,
                                compressedRungs: targetEntry.compressedRungs,
                                postedRungs: targetEntry.postedRungs,
                                filledRungs: targetEntry.filledRungs,
                                exchangeRejectedRungs: targetEntry.exchangeRejectedRungs,
                                upliftedDynamicRungs: targetEntry.upliftedDynamicRungs,
                                keptDynamicRungs: targetEntry.keptDynamicRungs,
                                droppedDynamicRungs: targetEntry.droppedDynamicRungs,
                                minExecutableNotionalUsd: targetEntry.minExecutableNotionalUsd,
                                redistributedNotionalUsd: targetEntry.redistributedNotionalUsd,
                                upliftedNotionalUsd: targetEntry.upliftedNotionalUsd,
                                executionDecision: targetEntry.executionDecision,
                                gt5PreservationStatus: targetEntry.gt5PreservationStatus,
                                compressionReason: targetEntry.compressionReason,
                                bestAskAtEntry: targetEntry.bestAskAtEntry,
                                queueCompetitionUsdAtEntry: targetEntry.queueCompetitionUsdAtEntry,
                                effectiveFillableUsdAtEntry: targetEntry.effectiveFillableUsdAtEntry,
                            });
                        }
                        if (nextTx) {
                            executedTxHashes.add(nextTx.toLowerCase());
                        }
                        reconciledCount += 1;
                    }
                    continue;
                }
                const probe = unresolvedPendingEntries[0];
                let marketResult: Awaited<ReturnType<typeof getMarketResultByConditionId>> | null = null;
                try {
                    if (probe.conditionId) {
                        marketResult = await getMarketResultByConditionId(probe.conditionId, probe.marketSlug);
                        if (!marketResult.resolved && probe.marketSlug) {
                            marketResult = await getMarketResult(probe.marketSlug);
                        }
                    } else if (probe.marketSlug) {
                        marketResult = await getMarketResult(probe.marketSlug);
                    }
                } catch {
                    marketResult = null;
                }
                if (marketResult?.resolved) {
                    for (const unresolvedEntry of unresolvedPendingEntries) {
                        this.logger.updateEntry(unresolvedEntry.id, {
                            status: 'failed',
                            liveOrderStatus: 'expired',
                            pendingAmountUsd: 0,
                            amount: 0,
                            matchedShares: 0,
                            fillSource: unresolvedEntry.fillSource || 'resolved_without_public_fill',
                            lastRejectionReason: 'market_resolved_without_public_fill',
                            lastNonExecutionReason: 'market_resolved_without_public_fill',
                            error: unresolvedEntry.error || 'market_resolved_without_public_fill',
                        });
                        reconciledCount += 1;
                    }
                }
            }
        }

        const pendingAfter = this.logger.getAllEntries().filter((entry) => entry.status === 'pending').length;
        return { reconciledCount, externalTradeCount: publicTrades.length, pendingAfter };
    }

    private async reconcileExecutedLiveTradesWithPublicActivity(): Promise<{ reconciledCount: number; externalTradeCount: number }> {
        if (this.env.TRADING_MODE !== TradingMode.LIVE) {
            return { reconciledCount: 0, externalTradeCount: 0 };
        }
        const currentEntries = this.logger.getAllEntries();
        const localFillCandidateEntries = currentEntries
            .filter((entry) => ['executed', 'pending', 'failed'].includes(String(entry.status || '')))
            .filter((entry) => ((entry.conditionId && entry.conditionId.trim()) || (entry.marketSlug && entry.marketSlug.trim())))
            .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
        if (localFillCandidateEntries.length === 0) {
            return { reconciledCount: 0, externalTradeCount: 0 };
        }

        const publicTrades = await this.fetchPublicActivityTrades();
        const publicTradesByGroup = new Map<string, PublicActivityTrade[]>();
        const publicTradesByCondition = new Map<string, PublicActivityTrade[]>();
        for (const trade of publicTrades) {
            const key = this.buildLiveTradeGroupKey(trade.conditionId, trade.marketSlug);
            if (!publicTradesByGroup.has(key)) publicTradesByGroup.set(key, []);
            publicTradesByGroup.get(key)!.push(trade);
            const conditionKey = this.buildLiveTradeConditionKey(trade.conditionId);
            if (conditionKey) {
                if (!publicTradesByCondition.has(conditionKey)) publicTradesByCondition.set(conditionKey, []);
                publicTradesByCondition.get(conditionKey)!.push(trade);
            }
        }

        const executedByGroup = new Map<string, TradeLogEntry[]>();
        for (const entry of localFillCandidateEntries) {
            const key = this.buildLiveTradeGroupKey(entry.conditionId, entry.marketSlug);
            if (!executedByGroup.has(key)) executedByGroup.set(key, []);
            executedByGroup.get(key)!.push(entry);
        }

        let reconciledCount = 0;
        for (const [groupKey, localRows] of executedByGroup.entries()) {
            const publicRows = this.getPublicTradesForLocalRows(
                localRows,
                publicTradesByGroup,
                publicTradesByCondition,
                groupKey,
            )
                .slice()
                .sort((a, b) => a.timestampSec - b.timestampSec || a.transactionHash.localeCompare(b.transactionHash));
            if (publicRows.length === 0) continue;

            const assignments = new Map<string, PublicActivityTrade>();
            const usedTx = new Set<string>();

            for (const localRow of localRows) {
                const localTx = String(localRow.transactionHash || localRow.txHash || '').trim().toLowerCase();
                if (!localTx) continue;
                const matched = publicRows.find((row) => row.transactionHash.toLowerCase() === localTx);
                if (matched) {
                    assignments.set(localRow.id, matched);
                    usedTx.add(matched.transactionHash.toLowerCase());
                }
            }

            const unmatchedPublicRows = publicRows.filter((row) => !usedTx.has(row.transactionHash.toLowerCase()));
            const unmatchedLocalRows = localRows
                .filter((row) => !assignments.has(row.id))
                .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

            for (let idx = 0; idx < unmatchedLocalRows.length && idx < unmatchedPublicRows.length; idx++) {
                const matchedRow = unmatchedPublicRows[idx];
                assignments.set(unmatchedLocalRows[idx].id, matchedRow);
                usedTx.add(matchedRow.transactionHash.toLowerCase());
            }

            for (const localRow of localRows) {
                const publicRow = assignments.get(localRow.id);
                if (!publicRow) continue;

                const nextAmount = roundUsd(publicRow.usdcSize);
                const nextPrice = publicRow.price > 0 ? publicRow.price : Number(localRow.avgActualFillPrice ?? localRow.tokenPrice ?? 0);
                const nextShares = Number(publicRow.size || localRow.matchedShares || 0);
                const nextTx = String(publicRow.transactionHash || localRow.transactionHash || localRow.txHash || '').trim();
                const currentAmount = roundUsd(Number(localRow.amount || 0));
                const currentPrice = Number(localRow.avgActualFillPrice ?? localRow.tokenPrice ?? 0);
                const currentShares = Number(localRow.matchedShares || 0);
                const currentTx = String(localRow.transactionHash || localRow.txHash || '').trim();

                const changed = (
                    Math.abs(currentAmount - nextAmount) > 0.0001 ||
                    Math.abs(currentPrice - nextPrice) > 0.000001 ||
                    Math.abs(currentShares - nextShares) > 0.0001 ||
                    currentTx.toLowerCase() !== nextTx.toLowerCase()
                );
                if (!changed) continue;

                this.logger.updateEntry(localRow.id, {
                    timestamp: publicRow.timestampIso || localRow.timestamp,
                    status: 'executed',
                    amount: nextAmount,
                    transactionHash: nextTx || localRow.transactionHash,
                    txHash: nextTx || localRow.txHash,
                    tokenId: publicRow.tokenId || localRow.tokenId,
                    tokenOutcome: publicRow.tokenOutcome || localRow.tokenOutcome,
                    tokenPrice: nextPrice > 0 ? nextPrice : localRow.tokenPrice,
                    avgActualFillPrice: nextPrice > 0 ? nextPrice : localRow.avgActualFillPrice ?? localRow.tokenPrice,
                    matchedShares: nextShares > 0 ? nextShares : localRow.matchedShares,
                    pendingAmountUsd: 0,
                    liveOrderStatus: 'filled',
                    fillSource: 'public_activity_api',
                    lastRejectionReason: undefined,
                    lastNonExecutionReason: undefined,
                    error: undefined,
                });
                reconciledCount += 1;
            }

            const assignedTxHashes = new Set(
                Array.from(assignments.values()).map((row) => row.transactionHash.toLowerCase()).filter(Boolean),
            );
            const stillUnmatchedPublicRows = publicRows.filter((row) => !assignedTxHashes.has(row.transactionHash.toLowerCase()));
            if (stillUnmatchedPublicRows.length > 0) {
                const template = localRows[localRows.length - 1];
                for (const publicRow of stillUnmatchedPublicRows) {
                    const nextAmount = roundUsd(publicRow.usdcSize);
                    const nextPrice = publicRow.price > 0
                        ? publicRow.price
                        : Number(template.avgActualFillPrice ?? template.tokenPrice ?? 0);
                    const nextShares = Number(publicRow.size || template.matchedShares || 0);
                    this.logger.addEntry({
                        symbol: template.symbol,
                        direction: template.direction,
                        confidence: Number(template.confidence || 0),
                        amount: nextAmount,
                        mode: this.modeScope,
                        status: 'executed',
                        marketTitle: template.marketTitle,
                        conditionId: publicRow.conditionId || template.conditionId,
                        tokenId: publicRow.tokenId || template.tokenId,
                        txHash: publicRow.transactionHash,
                        orderId: template.orderId,
                        transactionHash: publicRow.transactionHash,
                        error: undefined,
                        result: template.result || 'pending',
                        pnl: Number(template.pnl || 0),
                        marketSlug: publicRow.marketSlug || template.marketSlug,
                        tokenOutcome: publicRow.tokenOutcome || template.tokenOutcome,
                        tokenPrice: nextPrice > 0 ? nextPrice : template.tokenPrice,
                        avgActualFillPrice: nextPrice > 0 ? nextPrice : template.avgActualFillPrice ?? template.tokenPrice,
                        limitPriceConfigured: template.limitPriceConfigured,
                        requestedAmountUsd: Number(template.requestedAmountUsd ?? nextAmount),
                        pendingAmountUsd: 0,
                        requestedShares: Number(template.requestedShares ?? nextShares),
                        originalRequestedShares: template.originalRequestedShares,
                        matchedShares: nextShares > 0 ? nextShares : template.matchedShares,
                        liveOrderStatus: 'filled',
                        fillSource: 'public_activity_api',
                        lastRejectionReason: undefined,
                        familyRuleVersion: template.familyRuleVersion,
                        lowpriceFamilyId: template.lowpriceFamilyId,
                        stalePolicyDecision: template.stalePolicyDecision,
                        staleCancelReason: template.staleCancelReason,
                        exchangeMinOrderSizeShares: template.exchangeMinOrderSizeShares,
                        attemptedRungs: template.attemptedRungs,
                        compressedRungs: template.compressedRungs,
                        postedRungs: template.postedRungs,
                        filledRungs: template.filledRungs,
                        exchangeRejectedRungs: template.exchangeRejectedRungs,
                        upliftedDynamicRungs: template.upliftedDynamicRungs,
                        keptDynamicRungs: template.keptDynamicRungs,
                        droppedDynamicRungs: template.droppedDynamicRungs,
                        minExecutableNotionalUsd: template.minExecutableNotionalUsd,
                        redistributedNotionalUsd: template.redistributedNotionalUsd,
                        upliftedFixedOrders: template.upliftedFixedOrders,
                        upliftedNotionalUsd: template.upliftedNotionalUsd,
                        executionDecision: template.executionDecision,
                        gt5PreservationStatus: template.gt5PreservationStatus,
                        compressionReason: template.compressionReason,
                        lastNonExecutionReason: undefined,
                        bestAskAtEntry: template.bestAskAtEntry,
                        queueCompetitionUsdAtEntry: template.queueCompetitionUsdAtEntry,
                        effectiveFillableUsdAtEntry: template.effectiveFillableUsdAtEntry,
                    });
                    reconciledCount += 1;
                }
            }
        }

        return { reconciledCount, externalTradeCount: publicTrades.length };
    }

    private rebuildLivePendingOrderLedgerFromTrades(): { rewritten: boolean; eventCount: number } {
        if (this.env.TRADING_MODE !== TradingMode.LIVE) {
            return { rewritten: false, eventCount: 0 };
        }
        const ledgerPath = this.getPendingOrderLedgerPath();
        const existingRows: PendingOrderLedgerEvent[] = [];
        if (fs.existsSync(ledgerPath)) {
            try {
                const raw = fs.readFileSync(ledgerPath, 'utf-8');
                for (const line of raw.split('\n')) {
                    const trimmed = line.trim();
                    if (!trimmed) continue;
                    const parsed = JSON.parse(trimmed);
                    if (parsed && typeof parsed === 'object') {
                        existingRows.push(parsed as PendingOrderLedgerEvent);
                    }
                }
            } catch (error) {
                console.error(`  [挂单账本] 读取现有 ledger 失败，将按当前交易日志重建: ${error instanceof Error ? error.message : error}`);
            }
        }

        const createdByOrder = new Map<string, PendingOrderLedgerEvent>();
        for (const row of existingRows) {
            const orderId = String(row.orderId || '').trim();
            if (!orderId || row.event !== 'created') continue;
            const existing = createdByOrder.get(orderId);
            if (!existing || new Date(row.timestamp).getTime() < new Date(existing.timestamp).getTime()) {
                createdByOrder.set(orderId, row);
            }
        }

        const trades = this.logger.getAllEntries()
            .filter((entry) => (entry.mode === 'live' || this.modeScope === 'live') && String(entry.orderId || entry.txHash || '').trim())
            .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());

        const rowsOut: PendingOrderLedgerEvent[] = [];
        const seenCreated = new Set<string>();
        const seenFillEvents = new Set<string>();

        for (const trade of trades) {
            const orderId = String(trade.orderId || trade.txHash || trade.id).trim();
            if (!orderId) continue;

            let created = createdByOrder.get(orderId);
            const createdAmountUsd = roundUsd(Number(trade.requestedAmountUsd ?? trade.pendingAmountUsd ?? trade.amount ?? 0));
            const periodStartFromTrade = parsePeriodStartFromSlug(trade.marketSlug) || 0;
            const expectedExpiryIso = periodStartFromTrade > 0
                ? new Date((periodStartFromTrade + this.periodSeconds - 60) * 1000).toISOString()
                : trade.timestamp;
            if (!created) {
                created = {
                    timestamp: trade.timestamp,
                    event: 'created',
                    orderId,
                    symbol: trade.symbol,
                    direction: trade.direction,
                    amountUsd: createdAmountUsd,
                    remainingFrozenAmount: createdAmountUsd,
                    limitPrice: Number(trade.limitPriceConfigured ?? trade.tokenPrice ?? 0),
                    tokenId: String(trade.tokenId || ''),
                    tokenOutcome: String(trade.tokenOutcome || ''),
                    confidence: Number(trade.confidence || 0),
                    marketTitle: String(trade.marketTitle || ''),
                    conditionId: String(trade.conditionId || ''),
                    marketSlug: String(trade.marketSlug || ''),
                    targetPeriodEndTs: periodStartFromTrade,
                    createdAt: trade.timestamp,
                    expiresAt: expectedExpiryIso,
                };
            } else if (
                createdAmountUsd > 0 &&
                (Number(created.remainingFrozenAmount || 0) <= 0 || created.expiresAt === created.timestamp || created.expiresAt === created.createdAt)
            ) {
                // 旧重建链曾把“已过期未成交”的 created 行写成 remaining=0、
                // expiresAt=createdAt，导致看起来像“创建即不存在”。这里按交易日志恢复建单原貌。
                created = {
                    ...created,
                    amountUsd: createdAmountUsd,
                    remainingFrozenAmount: createdAmountUsd,
                    expiresAt: expectedExpiryIso,
                };
            }
            if (!seenCreated.has(orderId)) {
                rowsOut.push(created);
                seenCreated.add(orderId);
            }

            if (trade.status !== 'executed') {
                const status = String(trade.liveOrderStatus || '').trim();
                const reason = String(trade.lastNonExecutionReason || trade.lastRejectionReason || trade.error || '').trim();
                if (status === 'expired' || status === 'official_canceled_zero_fill' || reason === 'market_resolved_without_public_fill') {
                    const fillKey = `${orderId}:expired_refunded`;
                    if (!seenFillEvents.has(fillKey)) {
                        seenFillEvents.add(fillKey);
                        const createdTs = new Date(created.timestamp || created.createdAt || trade.timestamp).getTime();
                        const tradeTerminalTs = new Date(String(trade.settledAt || '')).getTime();
                        const expiresTs = new Date(String(created.expiresAt || '')).getTime();
                        const statusTs = new Date(String((trade as any).officialUpdatedAt || (trade as any).officialCanceledAt || '')).getTime();
                        let terminalTimestampSource = '';
                        let terminalTs = NaN;
                        if (Number.isFinite(statusTs)) {
                            terminalTs = statusTs;
                            terminalTimestampSource = 'official_status_time';
                        } else if (Number.isFinite(tradeTerminalTs)) {
                            terminalTs = tradeTerminalTs;
                            terminalTimestampSource = 'trade_settled_time';
                        } else if (Number.isFinite(expiresTs) && (!Number.isFinite(createdTs) || expiresTs > createdTs)) {
                            terminalTs = expiresTs;
                            terminalTimestampSource = 'order_expires_at';
                        }
                        if (!Number.isFinite(terminalTs) || (Number.isFinite(createdTs) && terminalTs <= createdTs)) {
                            // 没有官方终态/过期时间时，不能再用 created+1ms 伪造“立刻过期”。
                            // 官网当前挂单为空，只能说明“现在没有开放订单”，不能倒推创建瞬间已过期。
                            continue;
                        }
                        rowsOut.push({
                            timestamp: new Date(terminalTs).toISOString(),
                            event: 'expired_refunded',
                            orderId,
                            symbol: trade.symbol,
                            direction: trade.direction,
                            amountUsd: roundUsd(Number(trade.requestedAmountUsd ?? created.amountUsd ?? 0)),
                            remainingFrozenAmount: 0,
                            limitPrice: Number(trade.limitPriceConfigured ?? trade.tokenPrice ?? 0),
                            tokenId: String(trade.tokenId || ''),
                            tokenOutcome: String(trade.tokenOutcome || ''),
                            confidence: Number(trade.confidence || 0),
                            marketTitle: String(trade.marketTitle || ''),
                            conditionId: String(trade.conditionId || ''),
                            marketSlug: String(trade.marketSlug || ''),
                            targetPeriodEndTs: parsePeriodStartFromSlug(trade.marketSlug) || 0,
                            createdAt: created.createdAt,
                            expiresAt: created.expiresAt,
                            bestAsk: created.bestAsk,
                            avgFillPrice: Number(trade.avgActualFillPrice ?? trade.tokenPrice ?? 0) || undefined,
                            queueCompetitionUsdAtEntry: created.queueCompetitionUsdAtEntry,
                            effectiveFillableUsdAtEntry: created.effectiveFillableUsdAtEntry,
                            terminalTimestampSource,
                        });
                    }
                }
                continue;
            }
            const eventType: PendingOrderLedgerEventType = Number(trade.pendingAmountUsd || 0) > 0.0001 ? 'partial_fill' : 'filled';
            const fillKey = `${orderId}:${eventType}`;
            if (seenFillEvents.has(fillKey)) continue;
            seenFillEvents.add(fillKey);
            rowsOut.push({
                timestamp: String(trade.settledAt || trade.timestamp),
                event: eventType,
                orderId,
                symbol: trade.symbol,
                direction: trade.direction,
                amountUsd: roundUsd(Number(trade.amount || 0)),
                remainingFrozenAmount: roundUsd(Number(trade.pendingAmountUsd ?? 0)),
                limitPrice: Number(trade.limitPriceConfigured ?? trade.tokenPrice ?? trade.avgActualFillPrice ?? 0),
                tokenId: String(trade.tokenId || ''),
                tokenOutcome: String(trade.tokenOutcome || ''),
                confidence: Number(trade.confidence || 0),
                marketTitle: String(trade.marketTitle || ''),
                conditionId: String(trade.conditionId || ''),
                marketSlug: String(trade.marketSlug || ''),
                targetPeriodEndTs: parsePeriodStartFromSlug(trade.marketSlug) || 0,
                createdAt: created.createdAt,
                expiresAt: created.expiresAt,
                bestAsk: created.bestAsk,
                avgFillPrice: Number(trade.avgActualFillPrice ?? trade.tokenPrice ?? 0) || undefined,
                queueCompetitionUsdAtEntry: created.queueCompetitionUsdAtEntry,
                effectiveFillableUsdAtEntry: created.effectiveFillableUsdAtEntry,
            });
        }

        rowsOut.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
        try {
            const payload = rowsOut.map((row) => JSON.stringify(row)).join('\n');
            fs.writeFileSync(ledgerPath, payload ? `${payload}\n` : '', 'utf-8');
            return { rewritten: true, eventCount: rowsOut.length };
        } catch (error) {
            console.error(`  [挂单账本] 重建失败: ${error instanceof Error ? error.message : error}`);
            return { rewritten: false, eventCount: 0 };
        }
    }

    private async executeLiveTrade(
        marketResult: MarketSearchResult,
        token: { tokenId: string; outcome: string; price?: number },
        betAmount: number,
        logEntry: TradeLogEntry,
        prediction?: PredictionResult,
        opts?: {
            targetPeriodEndTs?: number;
            minPrice?: number;
            maxPrice?: number;
            triggerPrice?: number;
            asksSnapshot?: Array<{ price: string; size: string }>;
            preSignedOrder?: unknown;
            executionProfile?: ExecutionV2Profile;
        }
    ): Promise<{ success: boolean; txHash?: string; remainingAmount?: number; blockedByCooldown?: boolean; blockReason?: string }> {
        if (this.env.TRADING_MODE !== TradingMode.LIVE) {
            console.error('\n❌ [防护] executeLiveTrade 被调用但 TRADING_MODE 不是 LIVE，拒绝真实下单。当前模式:', this.env.TRADING_MODE);
            return { success: false };
        }
        if (!this.clobClient || !marketResult.market) {
            return { success: false };
        }
        const symbol = normalizeSymbol(logEntry.symbol || prediction?.symbol.replace('/USDT', '') || marketResult.symbol) || logEntry.symbol || marketResult.symbol;
        const direction = this.normalizeDirection(prediction?.direction || logEntry.direction);
        const marketSlugForCooldown = logEntry.marketSlug || marketResult.market?.slug;
        const targetPeriodStartForCooldown = opts?.targetPeriodEndTs ?? parsePeriodStartFromSlug(marketSlugForCooldown) ?? undefined;
        await this.prepareLiveCooldownStateBeforeOrder();
        const cooldownBlock = this.evaluateHardCooldownSubmissionBlock(symbol, direction, targetPeriodStartForCooldown);
        if (cooldownBlock.blocked) {
            const reason = cooldownBlock.error || `冷却阻断: ${cooldownBlock.label}`;
            console.log(`│ 🧊 ${reason.slice(0, 52).padEnd(52)}│`);
            this.logger.updateEntry(logEntry.id, {
                status: 'failed',
                error: reason,
                amount: 0,
                pendingAmountUsd: 0,
                executionDecision: 'cooldown_blocked_before_submit',
                lastNonExecutionReason: reason,
                cooldownScope: this.getCooldownScope(),
                cooldownKey: this.buildCooldownKey(symbol, direction),
            });
            if (prediction && direction && marketResult.market) {
                this.appendCooldownBlockedOpportunity({
                    prediction,
                    marketResult,
                    symbol,
                    direction,
                    targetPeriodEndTs: targetPeriodStartForCooldown,
                    remainingBarsBefore: cooldownBlock.remainingBarsBefore,
                    remainingBarsAfter: cooldownBlock.remainingBarsAfter,
                    decrementedThisRound: true,
                    reasonCode: cooldownBlock.reasonCode,
                });
            }
            return { success: false, remainingAmount: betAmount, blockedByCooldown: true, blockReason: reason };
        }
        const minPrice = opts?.minPrice || 0.25;
        const maxPrice = opts?.maxPrice || this.MAX_PRICE_THRESHOLD;
        const useWsData = opts?.triggerPrice != null && opts?.asksSnapshot != null && opts.asksSnapshot.length > 0;

        if (this.env.TRADING_MODE === TradingMode.LIVE && opts?.targetPeriodEndTs) {
            const earlyExpiration = this.computeLiveGtdExpiration(opts.targetPeriodEndTs);
            if (!earlyExpiration) {
                const reason = this.buildLiveGtdExpiredReason(opts.targetPeriodEndTs);
                console.log(`│ ⚠️  ${reason.slice(0, 55).padEnd(55)}│`);
                this.markLiveTradeRejected(logEntry, reason, betAmount, 0, token.tokenId, marketResult.market?.slug);
                return { success: false, remainingAmount: betAmount };
            }
        }

        let price: number | undefined;
        let availableLiquidity = 0;

        if (useWsData) {
            // 复用 WS 推送的 best_ask 和订单簿，不再拉取 API
            price = opts.triggerPrice!;
            for (const ask of opts.asksSnapshot!) {
                const askPrice = parseFloat(ask.price);
                if (askPrice < minPrice || askPrice > maxPrice) continue;
                const askSize = parseFloat(ask.size || '0');
                availableLiquidity += askSize * askPrice;
            }
        } else {
            let retry = 0;
            const timeoutMs = this.env.REQUEST_TIMEOUT_MS;
            while (retry < RETRY_LIMIT) {
                try {
                    console.log(`│ 正在获取订单簿...`.padEnd(59) + '│');
                    const orderBook = await this.clobClient.getOrderBook(token.tokenId);
                    if (!orderBook.asks || orderBook.asks.length === 0) {
                        console.log(`│ ⚠️  订单簿为空，无卖单可用`.padEnd(59) + '│');
                        return { success: false };
                    }
                    const bestAsk = orderBook.asks.reduce((min, ask) =>
                        parseFloat(ask.price) < parseFloat(min.price) ? ask : min
                    , orderBook.asks[0]);
                    price = parseFloat(bestAsk.price);
                    try {
                        for (const ask of orderBook.asks) {
                            const askPrice = parseFloat(ask.price);
                            const askSize = parseFloat((ask as any).size || (ask as any).amount || '0');
                            availableLiquidity += askSize * askPrice;
                        }
                    } catch (error) {
                        console.log(`│ ⚠️  无法计算订单簿流动性，继续尝试下单`.padEnd(59) + '│');
                    }
                    break;
                } catch (error) {
                    retry++;
                    if (retry >= RETRY_LIMIT) return { success: false };
                    await new Promise((r) => setTimeout(r, RETRY_INTERVAL_MS));
                }
            }
        }

        if (typeof price === 'undefined') {
            return { success: false };
        }
        
        let retry = 0;
        let preSignedOrder = opts?.preSignedOrder;
        if (preSignedOrder && maxPrice > this.MAX_PRICE_THRESHOLD + 1e-9 && price > this.MAX_PRICE_THRESHOLD + 1e-9) {
            preSignedOrder = undefined;
        }
        const timeoutMs = this.env.REQUEST_TIMEOUT_MS;
        let liveOrderShares = 0;
        
        while (retry < RETRY_LIMIT) {
            try {
                
                if (price < this.env.MIN_TOKEN_PRICE) {
                    const msg = `市场已近乎定价该结果 (卖价 $${price.toFixed(4)} < $${this.env.MIN_TOKEN_PRICE})，跳过`;
                    console.log(`│ ⚠️  ${msg.slice(0, 55).padEnd(55)}│`);
                    this.logger.updateEntry(logEntry.id, { status: 'failed', error: msg });
                    return { success: false };
                }

                if (price < minPrice || price > maxPrice) {
                    const msg = `价格不在允许范围: $${price.toFixed(4)} (要求: $${minPrice} ~ $${maxPrice})`;
                    console.log(`│ ⚠️  ${msg.slice(0, 55).padEnd(55)}│`);
                    this.logger.updateEntry(logEntry.id, { status: 'failed', error: msg });
                    return { success: false, remainingAmount: betAmount };
                }
                
                // GTD 限价单：到期自动取消，无需手动管理；UserChannelWS 实时推送成交
                const periodStartSec = opts?.targetPeriodEndTs || (Date.now() / 1000);
                const expirationSec = this.computeLiveGtdExpiration(periodStartSec);
                if (!expirationSec) {
                    const reason = this.buildLiveGtdExpiredReason(periodStartSec);
                    console.log(`│ ⚠️  ${reason.slice(0, 55).padEnd(55)}│`);
                    this.markLiveTradeRejected(logEntry, reason, betAmount, liveOrderShares || 0, token.tokenId, marketResult.market?.slug);
                    return { success: false, remainingAmount: betAmount };
                }
                const shares = preSignedOrder
                    ? Math.floor((betAmount / this.MAX_PRICE_THRESHOLD) * 100) / 100
                    : Math.floor((betAmount / price) * 100) / 100;
                liveOrderShares = shares;
                const priceForSpend = preSignedOrder ? this.MAX_PRICE_THRESHOLD : price;
                const shareCheck = this.precheckSharesAgainstConstraint(
                    shares,
                    this.executionConstraintByToken.get(token.tokenId) || this.buildFallbackExecutionConstraint(token.tokenId, marketResult.market?.slug),
                );
                if (!shareCheck.ok) {
                    const reason = String(shareCheck.reason || '真实订单 shares 不满足交易所最小限制');
                    console.log(`│ ⚠️  ${reason.slice(0, 55).padEnd(55)}│`);
                    this.markLiveTradeRejected(logEntry, reason, betAmount, shares, token.tokenId, marketResult.market?.slug);
                    return { success: false, remainingAmount: betAmount };
                }

                const userOrder = {
                    tokenID: token.tokenId,
                    price,
                    size: shares,
                    side: Side.BUY,
                    expiration: expirationSec,
                };
                
                const tokens = shares;
                const expDate = new Date(expirationSec * 1000);
                const expStr = expDate.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false });
                
                console.log(`│ 最佳卖价:   ${('$' + price.toFixed(4)).padEnd(45)}│`);
                console.log(`│ 价格限制:   ${('$' + minPrice.toFixed(2) + ' ~ $' + maxPrice.toFixed(2)).padEnd(45)}│`);
                console.log(`│ 预计获得:   ${('~' + tokens.toFixed(2) + ' 个代币 (' + shares + ' shares)').padEnd(45)}│`);
                if (preSignedOrder) {
                    console.log(`│ 预签价格:   ${('$' + this.MAX_PRICE_THRESHOLD.toFixed(2) + ' (预签单触发直发)').padEnd(45)}│`);
                }
                console.log(`│ 订单类型:   ${'GTD 限价单 (到期自动取消)'.padEnd(45)}│`);
                const expirationLabel = Math.max(0, Number((this.env as any).LATE_FILL_CANCEL_MINUTES) || 0) > 0
                    ? `${expStr} (满${Math.max(0, Number((this.env as any).LATE_FILL_CANCEL_MINUTES) || 0)}分钟或市场安全到期)`
                    : `${expStr} (市场结束前1分钟)`;
                console.log(`│ 过期时间:   ${expirationLabel.padEnd(45)}│`);
                console.log(`│ 正在提交订单...`.padEnd(59) + '│');
                
                // 二次防护：提交前再次确认模式，防止误发真实订单
                if (this.env.TRADING_MODE !== TradingMode.LIVE) {
                    console.error('\n❌ [防护] 提交前检测到 TRADING_MODE 非 LIVE，取消提交。');
                    return { success: false };
                }

                const preflight = await this.preflightLiveOrder(betAmount);
                if (!preflight.ok) {
                    retry++;
                    const reason = preflight.reason;
                    this.rememberLiveExchangeConstraints(reason, token.tokenId, marketResult.market?.slug);
                    if (!this.shouldRetryLiveSubmitIssue(preflight.status)) {
                        console.log(`│ ❌ LIVE 预检失败: ${reason.slice(0, 37).padEnd(37)}│`);
                        this.markLiveTradeRejected(logEntry, reason, betAmount, liveOrderShares, token.tokenId, marketResult.market?.slug);
                        return { success: false, remainingAmount: betAmount };
                    }
                    console.log(`│ ⚠️ LIVE 预检失败 ${retry}/${RETRY_LIMIT}: ${reason.slice(0, 26)}│`);
                    if (retry >= RETRY_LIMIT) {
                        this.markLiveTradeRejected(logEntry, reason, betAmount, liveOrderShares, token.tokenId, marketResult.market?.slug);
                        return { success: false, remainingAmount: betAmount };
                    }
                    await new Promise((r) => setTimeout(r, RETRY_INTERVAL_MS));
                    continue;
                }
                
                let resp: any;
                try {
                    await this.ensureClobHeartbeatFresh();
                    if (preSignedOrder) {
                        if (this.orderPoster) {
                            resp = await this.orderPoster.postOrder(preSignedOrder, OrderType.GTD);
                        } else {
                            resp = await this.clobClient.postOrder(preSignedOrder as any, OrderType.GTD);
                        }
                    } else {
                        const signedOrder = await this.clobClient.createOrder(userOrder, { tickSize: '0.01' });
                        if (this.orderPoster) {
                            resp = await this.orderPoster.postOrder(signedOrder, OrderType.GTD);
                        } else {
                            resp = await this.clobClient.postOrder(signedOrder, OrderType.GTD);
                        }
                    }
                } catch (orderError) {
                    // 预签订单可能因过期/签名无效失败：回退为触发时签名
                    if (preSignedOrder) preSignedOrder = undefined;
                    retry++;
                    const reason = parseErrorReason(orderError);
                    this.rememberLiveExchangeConstraints(reason, token.tokenId, marketResult.market?.slug);
                    if (this.isNonRetryableLiveReject(reason)) {
                        console.log(`│ ❌ 非重试拒单: ${reason.slice(0, 39).padEnd(39)}│`);
                        this.markLiveTradeRejected(logEntry, reason, betAmount, liveOrderShares, token.tokenId, marketResult.market?.slug);
                        return { success: false, remainingAmount: betAmount };
                    }
                    console.log(`│ 尝试 ${retry}/${RETRY_LIMIT} 签名/提交错误: ${reason.slice(0, 30)}│`);
                    if (retry >= RETRY_LIMIT) break;
                    await new Promise((r) => setTimeout(r, RETRY_INTERVAL_MS));
                    continue;
                }
                
                const immediatePostError = this.getPostOrderImmediateError(resp);
                if (immediatePostError) {
                    this.rememberLiveExchangeConstraints(immediatePostError, token.tokenId, marketResult.market?.slug);
                    console.log(`│ ❌ LIVE 提交被交易所拒绝: ${immediatePostError.slice(0, 34).padEnd(34)}│`);
                    this.markLiveTradeRejected(logEntry, immediatePostError, betAmount, liveOrderShares, token.tokenId, marketResult.market?.slug);
                    return { success: false, remainingAmount: betAmount };
                }
                const postedOrderId = this.extractPostedOrderId(resp);
                if (postedOrderId) {
                    const orderID = postedOrderId;
                    const officialAcceptance = await this.verifyOfficialOrderAcceptance(orderID, token.tokenId);
                    if (!officialAcceptance.accepted) {
                        const reason = `官网未确认自动订单收单: orderID=${orderID} reason=${officialAcceptance.reason || 'unknown'} post=${this.summarizePostOrderResponse(resp)}`;
                        this.rememberLiveExchangeConstraints(reason, token.tokenId, marketResult.market?.slug);
                        console.log(`│ ❌ ${reason.slice(0, 51).padEnd(51)}│`);
                        this.markLiveTradeRejected(logEntry, reason, betAmount, liveOrderShares, token.tokenId, marketResult.market?.slug);
                        return { success: false, remainingAmount: betAmount };
                    }
                    const conditionId = marketResult.market?.conditionId || '';
                    this.appendPendingOrderLedgerEvent({
                        event: 'created',
                        orderId: orderID,
                        symbol: logEntry.symbol || symbol,
                        direction: logEntry.direction,
                        amountUsd: betAmount,
                        remainingFrozenAmount: betAmount,
                        limitPrice: priceForSpend,
                        tokenId: token.tokenId,
                        tokenOutcome: token.outcome,
                        confidence: logEntry.confidence,
                        marketTitle: marketResult.market?.title || logEntry.marketTitle || '',
                        conditionId,
                        marketSlug: marketResult.market?.slug || logEntry.marketSlug || '',
                        targetPeriodEndTs: opts?.targetPeriodEndTs || parsePeriodStartFromSlug(marketResult.market?.slug) || 0,
                        createdAt: new Date().toISOString(),
                        expiresAt: new Date(expirationSec * 1000).toISOString(),
                        bestAsk: price,
                        avgFillPrice: price,
                        queueCompetitionUsdAtEntry: logEntry.queueCompetitionUsdAtEntry,
                        effectiveFillableUsdAtEntry: logEntry.effectiveFillableUsdAtEntry,
                        officialAcceptanceStatus: officialAcceptance.status,
                        officialAcceptanceSource: officialAcceptance.source,
                        officialAcceptanceAt: new Date().toISOString(),
                    });

                    const wsWaitPromise = new Promise<{ amountUsd: number; tokens: number; source: string }>((resolve) => {
                        this.pendingLiveOrders.set(orderID, {
                            orderId: orderID,
                            conditionId,
                            symbol: logEntry.symbol || '',
                            side: 'BUY',
                            price: priceForSpend,
                            sizeOrdered: shares,
                            amountUsd: betAmount,
                            sizeFilled: 0,
                            filledUsd: 0,
                            logEntryId: logEntry.id,
                            status: 'pending',
                            createdAt: Date.now(),
                            expirationSec,
                            resolve,
                        });
                    });

                    const WS_CONFIRM_TIMEOUT_MS = 10_000;
                    let fillResult: { amountUsd: number; tokens: number; source: string };
                    const timeoutPromise = new Promise<{ amountUsd: number; tokens: number; source: string }>((resolve) => {
                        setTimeout(() => resolve({ amountUsd: -1, tokens: 0, source: '' }), WS_CONFIRM_TIMEOUT_MS);
                    });

                    fillResult = await Promise.race([wsWaitPromise, timeoutPromise]);

                    if (fillResult.amountUsd < 0) {
                        console.log(`│ ⚠️  WS 确认超时(${WS_CONFIRM_TIMEOUT_MS / 1000}s)，回退REST查询...`.padEnd(59) + '│');
                        const restFill = await this.queryActualFill(orderID, conditionId, price, betAmount);
                        if (restFill.amountUsd >= 0) {
                            fillResult = { amountUsd: restFill.amountUsd, tokens: restFill.tokens, source: 'REST_fallback' };
                        } else {
                            fillResult = { amountUsd: 0, tokens: 0, source: '' };
                        }
                    }

                    this.pendingLiveOrders.delete(orderID);

                    const actualAmount = roundUsd(fillResult.amountUsd);
                    const actualTokens = fillResult.tokens;
                    const remainingAmount = roundUsd(betAmount - actualAmount);

                    if (fillResult.source === 'official_canceled_zero_fill') {
                        const reason = '官方已取消零成交';
                        console.log(`│ ⚠️  ${reason}，官网无当前开放挂单`.padEnd(59) + '│');
                        this.logger.updateEntry(logEntry.id, {
                            status: 'failed',
                            orderId: orderID,
                            txHash: orderID,
                            amount: 0,
                            requestedAmountUsd: roundUsd(betAmount),
                            pendingAmountUsd: 0,
                            requestedShares: shares,
                            matchedShares: 0,
                            liveOrderStatus: 'official_canceled_zero_fill',
                            fillSource: fillResult.source,
                            lastNonExecutionReason: reason,
                            error: reason,
                            familyRuleVersion: this.getFamilyRuleVersion() || logEntry.familyRuleVersion,
                            lowpriceFamilyId: this.getLowpriceFamilyId() || logEntry.lowpriceFamilyId,
                            stalePolicyDecision: this.getStaleOrderPolicy() === 'off' ? 'off' : 'kept',
                        });
                        this.appendPendingOrderLedgerEvent({
                            event: 'official_canceled_zero_fill',
                            orderId: orderID,
                            symbol: logEntry.symbol || symbol,
                            direction: logEntry.direction,
                            amountUsd: 0,
                            remainingFrozenAmount: 0,
                            limitPrice: priceForSpend,
                            tokenId: token.tokenId,
                            tokenOutcome: token.outcome,
                            confidence: logEntry.confidence,
                            marketTitle: marketResult.market?.title || logEntry.marketTitle || '',
                            conditionId,
                            marketSlug: marketResult.market?.slug || logEntry.marketSlug || '',
                            targetPeriodEndTs: opts?.targetPeriodEndTs || parsePeriodStartFromSlug(marketResult.market?.slug) || 0,
                            createdAt: new Date().toISOString(),
                            expiresAt: new Date(expirationSec * 1000).toISOString(),
                            bestAsk: price,
                            avgFillPrice: undefined,
                            queueCompetitionUsdAtEntry: logEntry.queueCompetitionUsdAtEntry,
                            effectiveFillableUsdAtEntry: logEntry.effectiveFillableUsdAtEntry,
                            terminalTimestampSource: 'official_get_order_canceled',
                        });
                        return { success: false, txHash: orderID, remainingAmount: betAmount };
                    }

                    if (actualAmount <= 0 && (
                        fillResult.source === 'WS_CANCELLED' ||
                        fillResult.source === 'WS_CANCELLED_ZERO_FILL' ||
                        fillResult.source === 'STALE_CANCELLED'
                    )) {
                        const reason = fillResult.source === 'STALE_CANCELLED'
                            ? '本地超时取消零成交'
                            : '官方推送取消零成交';
                        console.log(`│ ⚠️  ${reason}，不得记为成交`.padEnd(59) + '│');
                        this.logger.updateEntry(logEntry.id, {
                            status: 'failed',
                            orderId: orderID,
                            txHash: orderID,
                            amount: 0,
                            requestedAmountUsd: roundUsd(betAmount),
                            pendingAmountUsd: 0,
                            requestedShares: shares,
                            matchedShares: 0,
                            liveOrderStatus: 'cancelled',
                            fillSource: fillResult.source,
                            lastNonExecutionReason: reason,
                            error: reason,
                            familyRuleVersion: this.getFamilyRuleVersion() || logEntry.familyRuleVersion,
                            lowpriceFamilyId: this.getLowpriceFamilyId() || logEntry.lowpriceFamilyId,
                            stalePolicyDecision: fillResult.source === 'STALE_CANCELLED' ? 'late_fill_timeout_cancelled' : logEntry.stalePolicyDecision,
                        });
                        this.appendPendingOrderLedgerEvent({
                            event: 'cancelled',
                            orderId: orderID,
                            symbol: logEntry.symbol || symbol,
                            direction: logEntry.direction,
                            amountUsd: 0,
                            remainingFrozenAmount: 0,
                            limitPrice: priceForSpend,
                            tokenId: token.tokenId,
                            tokenOutcome: token.outcome,
                            confidence: logEntry.confidence,
                            marketTitle: marketResult.market?.title || logEntry.marketTitle || '',
                            conditionId,
                            marketSlug: marketResult.market?.slug || logEntry.marketSlug || '',
                            targetPeriodEndTs: opts?.targetPeriodEndTs || parsePeriodStartFromSlug(marketResult.market?.slug) || 0,
                            createdAt: new Date().toISOString(),
                            expiresAt: new Date(expirationSec * 1000).toISOString(),
                            bestAsk: price,
                            avgFillPrice: undefined,
                            queueCompetitionUsdAtEntry: logEntry.queueCompetitionUsdAtEntry,
                            effectiveFillableUsdAtEntry: logEntry.effectiveFillableUsdAtEntry,
                            terminalTimestampSource: fillResult.source,
                        });
                        return { success: false, txHash: orderID, remainingAmount: betAmount };
                    }

                    if (!fillResult.source) {
                        console.log(`│ ⚠️  暂未确认成交，GTD 订单挂单中 (过期 ${expStr})`.padEnd(59) + '│');
                        console.log(`│ 订单ID:     ${orderID.slice(0, 45).padEnd(45)}│`);
                        this.logger.updateEntry(logEntry.id, {
                            status: 'pending',
                            orderId: orderID,
                            txHash: orderID,
                            amount: 0,
                            requestedAmountUsd: roundUsd(betAmount),
                            pendingAmountUsd: roundUsd(betAmount),
                            requestedShares: shares,
                            matchedShares: 0,
                            liveOrderStatus: 'posted_pending',
                            lateFillCancelMinutes: Math.max(0, Number((this.env as any).LATE_FILL_CANCEL_MINUTES) || 0) || undefined,
                            familyRuleVersion: this.getFamilyRuleVersion() || logEntry.familyRuleVersion,
                            lowpriceFamilyId: this.getLowpriceFamilyId() || logEntry.lowpriceFamilyId,
                            stalePolicyDecision: this.getStaleOrderPolicy() === 'off' ? 'off' : 'kept',
                        });
                        return { success: true, txHash: orderID, remainingAmount: betAmount };
                    }
                    
                    const isPartialFill = remainingAmount > 0.01;
                    if (isPartialFill) {
                        const fillPercent = (actualAmount / betAmount * 100).toFixed(1);
                        console.log(`│ 状态:       ${('⚠️  GTD部分成交 (' + fillPercent + '%)').padEnd(45)}│`);
                        console.log(`│ 订单ID:     ${orderID.slice(0, 45).padEnd(45)}│`);
                        console.log(`│ 成交金额:   ${('$' + actualAmount.toFixed(2) + ' / $' + betAmount.toFixed(2)).padEnd(45)}│`);
                        console.log(`│ 成交价格:   ${('$' + price.toFixed(4)).padEnd(45)}│`);
                        console.log(`│ 成交数量:   ${('~' + actualTokens.toFixed(2) + ' 个代币').padEnd(45)}│`);
                        console.log(`│ 未成交:     ${('$' + remainingAmount.toFixed(2) + ' (GTD挂单等待至 ' + expStr + ')').padEnd(45)}│`);
                        console.log(`│ 确认来源:   ${fillResult.source.padEnd(45)}│`);
                        
                        this.logger.updateEntry(logEntry.id, { 
                            status: 'executed',
                            orderId: orderID,
                            txHash: orderID,
                            amount: actualAmount,
                            requestedAmountUsd: roundUsd(betAmount),
                            pendingAmountUsd: remainingAmount,
                            requestedShares: shares,
                            matchedShares: actualTokens,
                            liveOrderStatus: 'partially_filled',
                            fillSource: fillResult.source,
                            familyRuleVersion: this.getFamilyRuleVersion() || logEntry.familyRuleVersion,
                            lowpriceFamilyId: this.getLowpriceFamilyId() || logEntry.lowpriceFamilyId,
                            stalePolicyDecision: this.getStaleOrderPolicy() === 'off' ? 'off' : 'kept',
                        });
                        this.appendExecutionV2SettlementEventFromLog(logEntry.id, 'chase_reprice_filled', 'live_partial_fill_confirmed');
                        
                        return { success: true, txHash: orderID, remainingAmount };
                    } else {
                        console.log(`│ 状态:       ${('✅ GTD全部成交').padEnd(45)}│`);
                        console.log(`│ 订单ID:     ${orderID.slice(0, 45).padEnd(45)}│`);
                        console.log(`│ 成交金额:   ${('$' + actualAmount.toFixed(2)).padEnd(45)}│`);
                        console.log(`│ 成交价格:   ${('$' + price.toFixed(4)).padEnd(45)}│`);
                        console.log(`│ 成交数量:   ${('~' + actualTokens.toFixed(2) + ' 个代币').padEnd(45)}│`);
                        console.log(`│ 确认来源:   ${fillResult.source.padEnd(45)}│`);
                        
                        this.logger.updateEntry(logEntry.id, { 
                            status: 'executed',
                            orderId: orderID,
                            txHash: orderID,
                            amount: actualAmount,
                            requestedAmountUsd: roundUsd(betAmount),
                            pendingAmountUsd: 0,
                            requestedShares: shares,
                            matchedShares: actualTokens,
                            liveOrderStatus: 'filled',
                            fillSource: fillResult.source,
                            familyRuleVersion: this.getFamilyRuleVersion() || logEntry.familyRuleVersion,
                            lowpriceFamilyId: this.getLowpriceFamilyId() || logEntry.lowpriceFamilyId,
                            stalePolicyDecision: this.getStaleOrderPolicy() === 'off' ? 'off' : 'kept',
                        });
                        this.appendExecutionV2SettlementEventFromLog(logEntry.id, 'fully_filled', 'live_full_fill_confirmed');
                        
                        return { success: true, txHash: orderID, remainingAmount: 0 };
                    }
                } else {
                    const classified = classifyLiveSubmitIssue(resp);
                    const reason = classified.reason;
                    this.rememberLiveExchangeConstraints(reason, token.tokenId, marketResult.market?.slug);
                    console.log(`│ ❌ LIVE 提交失败: ${reason.slice(0, 40).padEnd(40)}│`);
                    this.markLiveTradeRejected(logEntry, reason, betAmount, liveOrderShares, token.tokenId, marketResult.market?.slug);
                    return { success: false, remainingAmount: betAmount };
                }
            } catch (error) {
                    retry++;
                    const reason = parseErrorReason(error);
                    this.rememberLiveExchangeConstraints(reason, token.tokenId, marketResult.market?.slug);
                    if (this.isNonRetryableLiveReject(reason)) {
                        console.log(`│ ❌ 非重试拒单: ${reason.slice(0, 39).padEnd(39)}│`);
                        this.markLiveTradeRejected(logEntry, reason, betAmount, liveOrderShares, token.tokenId, marketResult.market?.slug);
                        return { success: false, remainingAmount: betAmount };
                    }
                    if (reason.includes('超时')) {
                        console.log(`│ ⏱️  请求超时 (${timeoutMs/1000}秒)，正在重试...`.padEnd(59) + '│');
                    } else {
                    console.log(`│ 尝试 ${retry}/${RETRY_LIMIT} 错误: ${reason.slice(0, 35).padEnd(40)}│`);
                }
            }
        }
        
        const finalError = `重试 ${RETRY_LIMIT} 次后失败`;
        console.log(`│ 状态:       ${'❌ 失败 (已重试3次)'.padEnd(45)}│`);
        
        this.logger.updateEntry(logEntry.id, { 
            status: 'failed',
            error: finalError,
        });
        
        return { success: false };
    }

    /**
     * UserChannelWS trade 回调：实时接收成交事件，更新 pendingLiveOrders
     */
    onWsTrade(event: { id: string; taker_order_id: string; status: string; price: string; size: string; maker_orders?: Array<{ order_id: string; matched_amount: string; price: string }> }): void {
        const takerOrderId = event.taker_order_id;
        const tracker = this.pendingLiveOrders.get(takerOrderId);
        if (!tracker) {
            for (const [orderId, t] of this.pendingLiveOrders) {
                const makerMatch = event.maker_orders?.find(m => m.order_id === orderId);
                if (makerMatch) {
                    const matchedTokens = parseFloat(makerMatch.matched_amount || '0');
                    const matchedPrice = parseFloat(makerMatch.price || event.price || '0');
                    t.sizeFilled += matchedTokens;
                    t.filledUsd += roundUsd(matchedTokens * matchedPrice);
                    if (event.status === 'CONFIRMED' || event.status === 'MATCHED') {
                        t.status = t.sizeFilled >= t.sizeOrdered ? 'filled' : 'partial';
                        if (t.resolve) {
                            t.resolve({ amountUsd: t.filledUsd, tokens: t.sizeFilled, source: `WS_${event.status}` });
                            t.resolve = undefined;
                        }
                    } else if (event.status === 'FAILED') {
                        t.status = 'failed';
                        if (t.resolve) {
                            t.resolve({ amountUsd: 0, tokens: 0, source: 'WS_FAILED' });
                            t.resolve = undefined;
                        }
                    }
                    break;
                }
            }
            return;
        }

        const sz = parseFloat(event.size || '0');
        const pr = parseFloat(event.price || '0');
        if (event.status === 'MATCHED' || event.status === 'CONFIRMED') {
            tracker.sizeFilled += sz;
            tracker.filledUsd += roundUsd(sz * pr);
            tracker.status = tracker.sizeFilled >= tracker.sizeOrdered ? 'filled' : 'partial';
            if (tracker.resolve) {
                tracker.resolve({ amountUsd: tracker.filledUsd, tokens: tracker.sizeFilled, source: `WS_${event.status}` });
                tracker.resolve = undefined;
            }
        } else if (event.status === 'FAILED') {
            tracker.status = 'failed';
            if (tracker.resolve) {
                tracker.resolve({ amountUsd: 0, tokens: 0, source: 'WS_FAILED' });
                tracker.resolve = undefined;
            }
        }
    }

    /**
     * UserChannelWS order 回调：接收订单状态更新
     */
    onWsOrder(event: { id: string; type: string; size_matched: string; original_size: string; price: string }): void {
        const tracker = this.pendingLiveOrders.get(event.id);
        if (!tracker) return;

        if (event.type === 'CANCELLATION') {
            tracker.status = 'cancelled';
            const matchedTokens = parseFloat(event.size_matched || '0');
            const pr = parseFloat(event.price || String(tracker.price));
            if (tracker.resolve) {
                tracker.resolve({
                    amountUsd: roundUsd(matchedTokens * pr),
                    tokens: matchedTokens,
                    source: matchedTokens > 0 ? 'WS_CANCELLED' : 'WS_CANCELLED_ZERO_FILL',
                });
                tracker.resolve = undefined;
            }
        } else if (event.type === 'UPDATE') {
            const matchedTokens = parseFloat(event.size_matched || '0');
            if (matchedTokens > tracker.sizeFilled) {
                tracker.sizeFilled = matchedTokens;
                tracker.status = 'partial';
            }
        }
    }

    /** 获取当前需要 UserChannelWS 监听的 conditionId 列表（公开方法，供 index 调用） */
    getLiveOrderConditionIds(): string[] {
        const ids = new Set<string>();
        for (const t of this.pendingLiveOrders.values()) {
            if (t.conditionId) ids.add(t.conditionId);
        }
        const pending = this.logger.getPendingTrades();
        for (const t of pending) {
            if (t.conditionId) ids.add(t.conditionId);
        }
        return [...ids];
    }
    
    // ============================================================
    // 结算相关方法
    // ============================================================
    
    /**
     * 单笔止损检查：当代币价格从买入价下跌达到 STOP_LOSS_PCT 时，立即卖出止损。
     * 仅模拟模式支持；live 模式暂不执行实际卖出。
     */
    async checkPerTradeStopLoss(): Promise<{ stoppedCount: number }> {
        // 历史遗留参数：当前策略链路已不使用自动止损执行（模拟/实盘均禁用）
        if (this.env.STOP_LOSS_PCT != null && this.env.STOP_LOSS_PCT > 0 && !this.stopLossDeprecatedLogged) {
            console.log(`ℹ️  STOP_LOSS_PCT=${this.env.STOP_LOSS_PCT} 已配置，但当前版本不执行自动止损卖出（参数仅保留兼容）。`);
            this.stopLossDeprecatedLogged = true;
        }
        return { stoppedCount: 0 };
    }
    
    /**
     * 结算历史交易
     * 
     * 实盘与模拟共用同一套逻辑：通过 API（getMarketResultByConditionId / getMarketResult）读取市场结果，
     * 判断 win/lose、计算 pnl、写入交易日志（settleTrade）。实盘也能和模拟一样判断「结果是否正确」。
     * 唯一差异：仅模拟/回测在此处更新 currentCapital；实盘资金由 refreshLiveCapital 从链上同步。
     * 
     * 【真实交易结算如何计算】
     * - 链上：Polymarket 协议在周期结束后自动解析结果（Oracle 上链），赢家代币按 $1 赎回，输家归零。
     * - 脚本：通过 CLOB/Gamma API 读取市场 resolved 与 winner（Up/Down），与本地记录的 tokenOutcome 比对，
     *   一致则记为赢，pnl = (下注/买入价)*1 - 下注；否则记为输，pnl = -下注。仅更新本地日志与报告；
     *   真实余额以链上/钱包为准，由 refreshLiveCapital 同步，不在此处改写。
     * 
     * @returns { settledCount, totalPnL } 本轮回结单笔数、结单盈亏合计（正赚负赔）
     */
    async settleHistory(opts?: { skipReportRefresh?: boolean }): Promise<{ settledCount: number; totalPnL: number }> {
        // Rebuild loss streak from the persisted ledger before settling newly
        // resolved trades. This makes cooldown robust after process restarts and
        // when direct ladder execution bypasses executeRound().
        this.refreshCooldownLossStateFromHistory();
        const pendingTrades = this.logger.getPendingTrades();
        /** 本轮结算内，live claim 按 conditionId 最多触发一次 */
        const claimAttemptedThisRound = new Set<string>();
        const claimTasks: Promise<void>[] = [];
        if (pendingTrades.length === 0) {
            if (this.env.LIVE_CLAIM_SUBMIT_ENABLED) {
                await this.retryUnclaimedWinningClaims(claimAttemptedThisRound);
            }
            return { settledCount: 0, totalPnL: 0 };
        }
        const comboName = this.getComboDisplayName();
        const pendingLogName = path.basename(this.logger.getLogFilePath());
        console.log('\n' + '═'.repeat(60));
        console.log(`  📊 [${comboName}] 结算检查 - 发现 ${pendingTrades.length} 笔待结算交易（来自本组合目录 ${this.logsDir}/${pendingLogName}）`);
        console.log('═'.repeat(60));
        
        let settledCount = 0;
        let totalPnL = 0;
        let wins = 0;
        let losses = 0;
        /** 冷却按「每个市场+方向」只触发一次：同一 conditionId 多笔成交只算一次该方向亏损 */
        const cooldownActivatedForMarket = new Set<string>();
        /** 连输状态按逻辑市场只推进一次；同一市场多笔成交不能重复累计连输。 */
        const cooldownStateAppliedForMarket = new Set<string>();
        const pendingTradeIds = new Set(pendingTrades.map((trade) => trade.id));
        for (const entry of this.logger.getAllEntries()) {
            if (!this.isCooldownEligibleEntry(entry)) continue;
            if (pendingTradeIds.has(entry.id)) continue;
            const alreadyCountedResult =
                entry.formalResult === 'win' ||
                entry.formalResult === 'lose' ||
                entry.result === 'win' ||
                entry.result === 'lose' ||
                Boolean(entry.cooldownTriggeredStage);
            if (!alreadyCountedResult) continue;
            cooldownStateAppliedForMarket.add(this.buildCooldownLogicalMarketKey(entry));
        }
        /** 本批次内该币种+方向是否已有亏损市场（避免同批内先赢后亏时错误重置冷却） */
        const symbolDirectionHadLossInBatch = new Set<string>();
        /** 每笔交易的市场结果缓存，避免重复请求，并用于预计算 symbolHadLossInBatch */
        const resultCache = new Map<string, Awaited<ReturnType<typeof getMarketResultByConditionId>>>();
        for (const t of pendingTrades) {
            if (!t.conditionId && !t.marketSlug) continue;
            let res: Awaited<ReturnType<typeof getMarketResultByConditionId>>;
            if (t.conditionId) {
                res = await getMarketResultByConditionId(t.conditionId, t.marketSlug);
                if (!res.resolved && t.marketSlug) {
                    res = await getMarketResult(t.marketSlug);
                    if (res.resolved) {
                        console.warn(`  ⚠️ [${comboName}] ${t.symbol}-15M 结算: CLOB 未解析，使用 Gamma 兜底已解析 (${t.marketSlug})`);
                    }
                }
            } else {
                res = await getMarketResult(t.marketSlug!);
            }
            resultCache.set(t.id, res);
                if (res.resolved && res.winner != null) {
                    const tokenOutcome = t.tokenOutcome || (t.direction === 'UP' ? 'Up' : 'Down');
                    if (res.winner !== tokenOutcome) {
                        symbolDirectionHadLossInBatch.add(this.buildCooldownKey(t.symbol, t.direction));
                    }
                }
            }

        for (const trade of pendingTrades) {
            if (!trade.conditionId && !trade.marketSlug) {
                continue;
            }
            const marketResult = resultCache.get(trade.id)!;
            if (!marketResult) continue;
            if (!marketResult.resolved) {
                // 市场尚未结算，跳过。括号内为该笔订单对应的 market slug，便于与 Polymarket 对单
                const slugHint = trade.marketSlug ? ` (${trade.marketSlug})` : (trade.conditionId ? ` (${trade.conditionId.slice(0, 18)}...)` : '');
                console.log(`  ⏳ [${comboName}] ${trade.symbol}-15M${slugHint}: 市场尚未结算`);
                continue;
            }
            
            // 校验 winner 字段：已 resolved 但 winner 为 null 可能是 API 返回异常
            if (marketResult.winner === null || marketResult.winner === undefined) {
                const slugHint = trade.marketSlug ? ` (${trade.marketSlug})` : '';
                console.log(`  ⚠️  [${comboName}] ${trade.symbol}-15M${slugHint}: 市场已结算但 winner 为空，跳过（API 异常？）`);
                continue;
            }
            
            // 判断是否赢了
            // 用户买的是 tokenOutcome（Up 或 Down），如果市场 winner 与之匹配则赢
            const tokenOutcome = trade.tokenOutcome || (trade.direction === 'UP' ? 'Up' : 'Down');
            const won = marketResult.winner === tokenOutcome;
            const logicalCooldownMarketKey = this.buildCooldownLogicalMarketKey(trade);
            const applyCooldownStateForThisMarket = !cooldownStateAppliedForMarket.has(logicalCooldownMarketKey);
            if (applyCooldownStateForThisMarket) {
                cooldownStateAppliedForMarket.add(logicalCooldownMarketKey);
            }
            
            // 计算盈亏
            // Polymarket 规则：
            // - 赢：获得 (下注金额 / 买入价格) 个代币，每个代币值 $1
            // - 输：损失全部下注金额
            // 校验 tokenPrice：为 0 或缺失时记录警告并使用默认值
            let buyPrice = trade.tokenPrice;
            if (!buyPrice || buyPrice <= 0) {
                const slugHint = trade.marketSlug ? ` (${trade.marketSlug})` : '';
                console.log(`  ⚠️  [${comboName}] ${trade.symbol}-15M${slugHint}: tokenPrice 为 ${trade.tokenPrice}，使用默认值 0.5 计算 PnL`);
                buyPrice = 0.5;
            }
            const tokensOwned = trade.amount / buyPrice;
            // 优先使用真实成交返回的 matchedShares；这已经是实际拿到的 shares，
            // 比基于 taker fee 公式的近似推导更接近官方成交与结算口径。
            const matchedShares = Number((trade as any).matchedShares);
            // Polymarket taker 手续费：在买入时以 shares 形式扣除，减少实际持有代币数。
            // 当没有 matchedShares 时，仍保留旧的 fee-rate 近似作为回退。
            const feeRate = polymarketTakerFeeRate(buyPrice);
            const tokensAfterFee = Number.isFinite(matchedShares) && matchedShares > 0
                ? matchedShares
                : tokensOwned * (1 - feeRate);
            
            let pnl: number;
            if (won) {
                // 赢了：获得代币数量(优先真实 matchedShares，否则用近似扣费后代币数) * $1 - 原始下注金额
                pnl = tokensAfterFee - trade.amount;
                wins++;
                
                // 更新资金（仅模拟/回测模式）
                if (trade.mode !== 'live') {
                    this.adjustCoinCapital(trade.symbol, tokensAfterFee);
                    this.updateCoinPeakIfNeeded(trade.symbol);
                }
                if (applyCooldownStateForThisMarket) {
                    this.state.formalConsecutiveLosses = 0;
                }
                const winCooldownUpdates: Partial<TradeLogEntry> = {
                    result: 'win',
                    formalResult: 'win',
                    formalConsecutiveLosses: this.state.formalConsecutiveLosses,
                    provisionalConsecutiveLosses: 0,
                    cooldownScope: this.getCooldownScope(),
                    cooldownKey: this.buildCooldownKey(trade.symbol, trade.direction),
                };
                this.logger.updateEntry(trade.id, winCooldownUpdates);
                if (applyCooldownStateForThisMarket) {
                    this.updateLogicalCooldownMarketEntries(trade, winCooldownUpdates);
                }
                // LIVE 模式：自动赎回胜出 token
                if (trade.mode === 'live' && this.env.LIVE_CLAIM_SUBMIT_ENABLED && trade.conditionId) {
                    const conditionId = trade.conditionId;
                    if (!claimAttemptedThisRound.has(conditionId) && !this.hasSuccessfulClaim(conditionId)) {
                        claimAttemptedThisRound.add(conditionId);
                        claimTasks.push(
                            this.redeemWinningTokens(conditionId)
                                .then(() => undefined)
                                .catch(e => {
                                    console.error(`  ⚠️ 自动赎回异常 (${conditionId.slice(0, 18)}...): ${e}`);
                                })
                        );
                    }
                }
                // 胜场重置该币种+方向冷却（若本批次同方向已有亏损，则不重置）
                if (applyCooldownStateForThisMarket) {
                    const cooldownKey = this.buildCooldownKey(trade.symbol, trade.direction);
                    if (!symbolDirectionHadLossInBatch.has(cooldownKey)) {
                        this.setCooldownRemaining(trade.symbol, trade.direction, 0);
                    }
                }
            } else {
                // 输了：损失全部下注金额（已在下单时扣除）
                pnl = -trade.amount;
                losses++;
                // symbolHadLossInBatch 已在预取阶段按结算结果填好，此处不再重复 add

                const cooldownStage = this.logger.getEntry(trade.id)?.cooldownTriggeredStage;
                this.state.lastLossTs = Date.now();
                if (!applyCooldownStateForThisMarket) {
                    this.logger.updateEntry(trade.id, {
                        result: 'lose',
                        formalResult: 'lose',
                        formalConsecutiveLosses: this.state.formalConsecutiveLosses,
                        provisionalConsecutiveLosses: this.state.provisionalConsecutiveLosses,
                        cooldownScope: this.getCooldownScope(),
                        cooldownKey: this.buildCooldownKey(trade.symbol, trade.direction),
                    });
                } else if (cooldownStage === 'provisional') {
                    this.state.formalConsecutiveLosses = 0;
                    const provisionalToFormalUpdates: Partial<TradeLogEntry> = {
                        result: 'lose',
                        formalResult: 'lose',
                        formalConsecutiveLosses: this.state.formalConsecutiveLosses,
                        provisionalConsecutiveLosses: this.state.provisionalConsecutiveLosses,
                        cooldownScope: this.getCooldownScope(),
                        cooldownKey: this.buildCooldownKey(trade.symbol, trade.direction),
                    };
                    this.logger.updateEntry(trade.id, provisionalToFormalUpdates);
                    this.updateLogicalCooldownMarketEntries(trade, provisionalToFormalUpdates);
                    console.log(`  🧊 [${comboName}] ${trade.symbol}-15M 亏损已在预结算阶段触发冷却，正式结算仅接管状态，不重复累计连输`);
                } else {
                    this.state.formalConsecutiveLosses++;
                    const lossCooldownUpdates: Partial<TradeLogEntry> = {
                        result: 'lose',
                        formalResult: 'lose',
                        formalConsecutiveLosses: this.state.formalConsecutiveLosses,
                        provisionalConsecutiveLosses: this.state.provisionalConsecutiveLosses,
                        cooldownScope: this.getCooldownScope(),
                        cooldownKey: this.buildCooldownKey(trade.symbol, trade.direction),
                    };
                    this.logger.updateEntry(trade.id, lossCooldownUpdates);
                    this.updateLogicalCooldownMarketEntries(trade, lossCooldownUpdates);
                    // 冷却：支持“连续亏损达到阈值后，触发 N 根 K 线冷却”。
                    // 若未显式配置 COOLDOWN_AFTER_LOSSES，则默认保持历史行为（1 次亏损即触发）。
                    const dir = this.normalizeDirection(trade.direction) || 'UNKNOWN';
                    const marketKey = `${trade.symbol}:${dir}:${trade.conditionId || trade.marketSlug || ''}`;
                    const cooldownAfterLossesNum = this.getCooldownAfterLossesNum();
                    const cooldownBarsNum = this.getCooldownBarsNum();
                    if (
                        cooldownBarsNum > 0
                        && this.state.formalConsecutiveLosses >= cooldownAfterLossesNum
                        && !cooldownActivatedForMarket.has(marketKey)
                    ) {
                        cooldownActivatedForMarket.add(marketKey);
                        this.setCooldownRemaining(trade.symbol, trade.direction, cooldownBarsNum);
                        this.state.formalConsecutiveLosses = 0;
                        const cdLabel = this.formatCooldownLabel(trade.symbol, trade.direction);
                        console.log(`  🧊 冷却已激活: ${cdLabel}（连续亏损达到 ${cooldownAfterLossesNum} 次）跳过 ${cooldownBarsNum} 个周期 (${cooldownBarsNum * 15}分钟)`);
                        this.logger.updateEntry(trade.id, {
                            cooldownTriggeredStage: 'formal',
                            cooldownScope: this.getCooldownScope(),
                            cooldownKey: this.buildCooldownKey(trade.symbol, trade.direction),
                            formalConsecutiveLosses: this.state.formalConsecutiveLosses,
                        });
                        this.updateLogicalCooldownMarketEntries(trade, {
                            cooldownTriggeredStage: 'formal',
                            cooldownScope: this.getCooldownScope(),
                            cooldownKey: this.buildCooldownKey(trade.symbol, trade.direction),
                            formalConsecutiveLosses: this.state.formalConsecutiveLosses,
                            result: 'lose',
                            formalResult: 'lose',
                        });
                    }
                }
            }
            
            totalPnL += pnl;
            
            // 更新交易记录
            this.logger.settleTrade(trade.id, won, pnl);
            this.appendSignalOutcome(trade, won);
            settledCount++;
            
            // 打印结算详情
            const dirIcon = trade.direction === 'UP' ? '📈' : '📉';
            const dirText = trade.direction === 'UP' ? '上涨' : '下跌';
            const resultIcon = won ? '✅' : '❌';
            const resultText = won ? '正确' : '错误';
            const marketWinnerText = marketResult.winner === 'Up' ? '上涨' : '下跌';
            const pnlSign = pnl >= 0 ? '+' : '';
            
            const marketLabel = trade.symbol + '-15M' + (trade.marketSlug ? ' (' + trade.marketSlug + ')' : '');
            const settlePeriodTs = parsePeriodStartFromSlug(trade.marketSlug);
            console.log(`\n┌${'─'.repeat(50)}┐`);
            console.log(`│ 结算 #${trade.id.slice(-8).padEnd(42)}│`);
            console.log(`├${'─'.repeat(50)}┤`);
            console.log(`│ 市场:   ${marketLabel.padEnd(40)}│`);
            if (settlePeriodTs != null) {
                console.log(`│ 周期:   ${(this._fmtPeriod(settlePeriodTs)).padEnd(40)}│`);
            }
            console.log(`│ 预测:   ${(dirIcon + ' ' + dirText).padEnd(40)}│`);
            console.log(`│ 结果:   ${(resultIcon + ' ' + resultText + '！市场' + marketWinnerText).padEnd(40)}│`);
            const settledTradeAmount = Number.isFinite(Number(trade.amount)) ? Number(trade.amount) : 0;
            console.log(`│ 下注:   ${('$' + settledTradeAmount.toFixed(2) + ' @ $' + buyPrice.toFixed(4)).padEnd(40)}│`);
            
            if (won) {
                // 实际入账为扣费后代币数 tokensAfterFee（与 pnl、adjustCoinCapital 一致）
                console.log(`│ 获得:   ${(tokensAfterFee.toFixed(2) + ' 代币(扣费后) = $' + tokensAfterFee.toFixed(2)).padEnd(40)}│`);
            } else {
                // 亏损：下注归零，资金已在下单时扣除，此处仅记录盈亏并更新日志
                console.log(`│ 亏损:   ${('$' + settledTradeAmount.toFixed(2) + ' 归零 (已在下单时从资金扣除)').padEnd(40)}│`);
            }
            
            console.log(`│ 盈亏:   ${(pnlSign + '$' + pnl.toFixed(2)).padEnd(40)}│`);
            console.log(`└${'─'.repeat(50)}┘`);
        }
        
        if (claimTasks.length > 0) {
            await Promise.allSettled(claimTasks);
        }
        this.repairLogicalCooldownMetadataFromLedger();
        this.refreshCooldownLossStateFromHistory();
        if (this.env.LIVE_CLAIM_SUBMIT_ENABLED) {
            await this.retryUnclaimedWinningClaims(claimAttemptedThisRound);
        }

        // 打印结算摘要
        if (settledCount > 0 && !opts?.skipReportRefresh) {
            const pnlSign = totalPnL >= 0 ? '+' : '';
            console.log('\n' + '─'.repeat(60));
            console.log(`  💰 结算完成: ${wins}胜 ${losses}负, 盈亏 ${pnlSign}$${totalPnL.toFixed(2)}`);
            console.log(`  💵 当前资金: $${this.state.currentCapital.toFixed(2)}`);
            console.log('─'.repeat(60) + '\n');
            
            // 实时更新报告
            try {
                await this.generateReports();
            } catch (e) {
                console.error('实时报告生成失败:', e);
            }
        }
        
        return { settledCount, totalPnL };
    }
    
    // ════════════════════════════════════════════════════════════
    // 两阶段限价单方法 (Two-Phase Limit Order Strategy)
    // ════════════════════════════════════════════════════════════

    /**
     * Phase 1: 用 GTC 限价单下注 50% 仓位 @limit_price
     * 返回 orderID 列表用于后续跟踪
     */
    async executePhase1(opts: {
        targetPeriodEndTs: number;
        predictions: PredictionResult[];
        limitPrice: number;
        betFraction: number;
    }): Promise<ExecutionResult[]> {
        const { targetPeriodEndTs, predictions, limitPrice, betFraction } = opts;
        const results: ExecutionResult[] = [];

        console.log('\n' + '═'.repeat(60));
        console.log(`  🔵 Phase 1 限价单 - ${new Date().toLocaleString('zh-CN')}`);
        const rangeBeijing = this._fmtPeriod(targetPeriodEndTs);
        console.log(`  📌 目标周期: ${rangeBeijing}`);
        console.log(`  💰 限价: $${limitPrice}  仓位比: ${(betFraction * 100).toFixed(0)}%`);
        console.log('═'.repeat(60));

        printPredictions(predictions);

        const symbolsToFetch = [...new Set(predictions.map((p) => p.symbol.replace(/\/.*$/, '')))];
        const markets = await findAllMarkets(this.env.TIMEFRAME, targetPeriodEndTs, symbolsToFetch);
        for (const result of markets) {
            this.state.marketCache.set(result.symbol, result);
        }

        for (const prediction of predictions) {
            const symbol = prediction.symbol.replace('/USDT', '');
            const marketResult = this.state.marketCache.get(symbol);

            if (!marketResult || !marketResult.market) {
                console.log(`  ❌ ${symbol}: 市场数据未缓存`);
                results.push({ symbol, success: false, error: '市场数据未缓存' });
                continue;
            }

            // 计算下注金额（= 总仓位 × betFraction）
            const totalBet = this.calculateBetAmount(prediction.confidence, undefined, symbol, prediction.direction);
            const requestedPhaseBet = roundUsd(totalBet * betFraction);

            const direction = prediction.direction!;
            const token = marketResult.market ? getTokenForDirection(marketResult.market, direction) : null;
            if (!token) {
                console.log(`  ❌ ${symbol}: 无法获取 ${direction} token`);
                results.push({ symbol, success: false, error: 'token not found' });
                continue;
            }

            const phaseConstraint = await this.getExecutionConstraint(token.tokenId, marketResult.market.slug);
            const availableCapital = Math.max(0, this.getCoinCapital(symbol) || 0);
            const riskUpperLimit = this.getCapitalRiskUpperLimit(availableCapital);
            const phasePlan = this.buildFixedOrderExecutionPlan({
                requestedAmountUsd: requestedPhaseBet,
                price: limitPrice,
                maxBudgetUsd: Math.min(availableCapital, riskUpperLimit),
                constraint: phaseConstraint,
            });
            if (!phasePlan.executable) {
                const reason = phasePlan.failureReason || 'Phase 1 固定价订单 uplift 后超过资金/风控上限';
                console.log(`  ⏭️ ${symbol}: ${reason}`);
                this.addExecutionSkippedEntry({
                    symbol,
                    direction,
                    confidence: prediction.confidence,
                    reason,
                    executionDecision: 'uplifted_fixed_order_to_min_shares',
                    requestedAmountUsd: requestedPhaseBet,
                    requestedShares: phasePlan.requestedShares,
                    originalRequestedShares: phasePlan.requestedShares,
                    market: {
                        title: marketResult.market.title,
                        conditionId: marketResult.market.conditionId,
                        slug: marketResult.market.slug,
                    },
                    token,
                    exchangeMinOrderSizeShares: phaseConstraint.minOrderSizeShares,
                    upliftedFixedOrders: phasePlan.uplifted ? 1 : 0,
                    upliftedNotionalUsd: phasePlan.upliftedNotionalUsd,
                    gt5PreservationStatus: phasePlan.gt5PreservationStatus,
                    compressionReason: phasePlan.compressionReason,
                });
                results.push({ symbol, success: false, error: reason });
                continue;
            }

            const phaseBet = phasePlan.finalAmountUsd;
            const shares = phasePlan.finalShares;

            console.log(`\n  📝 ${symbol} Phase 1 限价单:`);
            console.log(`     方向: ${direction}  金额: $${phaseBet.toFixed(2)} (原始$${requestedPhaseBet.toFixed(2)})  限价: $${limitPrice}`);
            console.log(`     预计份额: ${shares.toFixed(2)} shares (min=${phaseConstraint.minOrderSizeShares.toFixed(2)})`);

            if (this.env.TRADING_MODE === TradingMode.SIMULATION) {
                // ─── 模拟模式: 用 Poly 订单簿判断限价单是否能成交 ───
                let bookSnapshot: QueueAwareBookSnapshot = {
                    bestAskPrice: 999,
                    asks: [],
                    bids: [],
                    hadBidsData: false,
                };

                try {
                    if (token.tokenId) {
                        const orderBook = await getOrderBook(token.tokenId);
                        bookSnapshot = buildQueueAwareBookSnapshot(orderBook);
                    }
                } catch { /* 获取失败时假设不成交 */ }

                const simulatedFill = simulateQueueAwareBuyFillWithHaircut(bookSnapshot, limitPrice, phaseBet);
                const fillableAmount = simulatedFill.haircutAdjustedFillableUsd;
                const avgFillPrice = simulatedFill.avgFillPrice;
                const bestAskPrice = simulatedFill.bestAskPrice;
                const queueCompetitionUsd = simulatedFill.queueCompetitionUsd;
                const effectiveFillableUsd = simulatedFill.effectiveFillableUsd;

                const fakeOrderID = `sim_phase1_${targetPeriodEndTs}_${symbol}_${Date.now()}`;
                const trackerKey = `${targetPeriodEndTs}_${symbol}`;

                if (bestAskPrice <= limitPrice && fillableAmount > 0) {
                    // ─── 限价单能成交（best_ask <= limit_price 且有流动性）───
                    const actualFill = Math.min(phaseBet, roundUsd(fillableAmount * 0.95));
                    
                    // 记入 trade log（关键：确保结算时能找到这笔交易）
                    this.logger.addEntry({
                        symbol,
                        direction,
                        confidence: prediction.confidence,
                        amount: actualFill,
                        mode: this.env.TRADING_MODE,
                        status: 'executed',
                        marketTitle: marketResult.market.title,
                        conditionId: marketResult.market.conditionId,
                        tokenId: token.tokenId,
                        marketSlug: marketResult.market.slug,
                        tokenOutcome: token.outcome,
                        tokenPrice: avgFillPrice,             // 实际加权均价（非限价）
                        avgActualFillPrice: avgFillPrice,
                        limitPriceConfigured: limitPrice,     // 审计：配置限价
                        bestAskAtEntry: bestAskPrice,
                        queueCompetitionUsdAtEntry: queueCompetitionUsd,
                        effectiveFillableUsdAtEntry: effectiveFillableUsd,
                        requestedAmountUsd: phaseBet,
                        requestedShares: phasePlan.finalShares,
                        originalRequestedShares: phasePlan.requestedShares,
                        exchangeMinOrderSizeShares: phaseConstraint.minOrderSizeShares,
                        upliftedFixedOrders: phasePlan.uplifted ? 1 : 0,
                        upliftedNotionalUsd: phasePlan.upliftedNotionalUsd,
                        executionDecision: phasePlan.uplifted ? 'uplifted_fixed_order_to_min_shares' : 'executed',
                        gt5PreservationStatus: phasePlan.gt5PreservationStatus,
                        compressionReason: phasePlan.compressionReason,
                    });

                    // 从资金中扣除（冻结）
                    this.adjustCoinCapital(symbol, -actualFill);

                    this.phaseOrderTracker.set(trackerKey, {
                        orderID: fakeOrderID,
                        tokenID: token.tokenId,
                        targetPeriodEndTs,
                        symbol,
                        direction,
                        amountUsd: actualFill,
                        plannedAmountUsd: phaseBet,
                        limitPrice,
                        confidence: prediction.confidence,
                        createdAt: new Date(),
                        filled: true,
                    });
                    console.log(`     ✅ [模拟] 限价单成交 (best_ask=$${bestAskPrice.toFixed(3)}, 均价=$${avgFillPrice.toFixed(4)}, 限价=$${limitPrice}, 排队竞争=$${queueCompetitionUsd.toFixed(2)})`);
                    console.log(`     💰 成交 $${actualFill.toFixed(2)}/${phaseBet.toFixed(2)}, 剩余资金 $${this.state.currentCapital.toFixed(2)}`);
                    results.push({ symbol, success: true, amount: actualFill, direction });
                } else {
                    // ─── 限价单不能成交（best_ask > limit_price）───
                    // 仍挂出（GTC 等待），但不扣资金、不记 trade log
                    this.phaseOrderTracker.set(trackerKey, {
                        orderID: fakeOrderID,
                        tokenID: token.tokenId,
                        targetPeriodEndTs,
                        symbol,
                        direction,
                        amountUsd: 0,  // 未成交，实际金额为 0
                        plannedAmountUsd: phaseBet,
                        limitPrice,
                        confidence: prediction.confidence,
                        createdAt: new Date(),
                        filled: false,
                    });
                    console.log(`     ⏸️ [模拟] 限价单已挂出但未成交 (best_ask=$${bestAskPrice.toFixed(3)} > limit=$${limitPrice})`);
                    console.log(`     ℹ️ 资金不冻结，等 Phase 2 时若方向一致将按市价下单`);
                    results.push({ symbol, success: true, amount: 0, direction });
                }
            } else if (this.env.TRADING_MODE === TradingMode.LIVE && this.clobClient) {
                // 真实模式: 用 createOrder + postOrder GTC（带重试）
                for (let orderRetry = 0; orderRetry < RETRY_LIMIT; orderRetry++) {
                    try {
                        const preflight = await this.preflightLiveOrder(phaseBet);
                        if (!preflight.ok) {
                            const reason = preflight.reason;
                            this.rememberLiveExchangeConstraints(reason, token.tokenId, marketResult.market.slug);
                            if (this.shouldRetryLiveSubmitIssue(preflight.status) && orderRetry < RETRY_LIMIT - 1) {
                                console.log(`     ⚠️ LIVE 预检失败，${RETRY_INTERVAL_MS/1000}s 后重试 [${orderRetry+1}/${RETRY_LIMIT}]`);
                                await new Promise(r => setTimeout(r, RETRY_INTERVAL_MS));
                                continue;
                            }
                            results.push({ symbol, success: false, error: reason });
                            break;
                        }
                        const gtdExpiration = this.computeLiveGtdExpiration(targetPeriodEndTs);
                        if (!gtdExpiration) {
                            const reason = this.buildLiveGtdExpiredReason(targetPeriodEndTs);
                            results.push({ symbol, success: false, error: reason });
                            break;
                        }
                        const userOrder = {
                            tokenID: token.tokenId,
                            price: limitPrice,
                            size: shares,
                            side: Side.BUY as Side,
                            expiration: gtdExpiration,
                        };
                        await this.ensureClobHeartbeatFresh();
                        const signedOrder = await this.clobClient.createOrder(userOrder, { tickSize: '0.01' });
                        const resp = await this.clobClient.postOrder(signedOrder, OrderType.GTD);

                        const immediatePostError = this.getPostOrderImmediateError(resp);
                        if (immediatePostError) {
                            this.rememberLiveExchangeConstraints(immediatePostError, token.tokenId, marketResult.market.slug);
                            results.push({ symbol, success: false, error: immediatePostError });
                            break;
                        }
                        const postedOrderId = this.extractPostedOrderId(resp);
                        if (postedOrderId) {
                            const orderID = postedOrderId;
                            const officialAcceptance = await this.verifyOfficialOrderAcceptance(orderID, token.tokenId);
                            if (!officialAcceptance.accepted) {
                                const reason = `官网未确认自动订单收单: orderID=${orderID} reason=${officialAcceptance.reason || 'unknown'} post=${this.summarizePostOrderResponse(resp)}`;
                                this.rememberLiveExchangeConstraints(reason, token.tokenId, marketResult.market.slug);
                                results.push({ symbol, success: false, error: reason });
                                break;
                            }
                            const trackerKey = `${targetPeriodEndTs}_${symbol}`;
                            this.phaseOrderTracker.set(trackerKey, {
                                orderID,
                                tokenID: token.tokenId,
                                targetPeriodEndTs,
                                symbol,
                                direction,
                                amountUsd: 0,  // LIVE 模式挂单后不确定是否成交，Phase 2 会查询
                                plannedAmountUsd: phaseBet,
                                limitPrice,
                                confidence: prediction.confidence,
                                createdAt: new Date(),
                                filled: false,  // 初始状态，Phase 2 查询后更新
                            });
                            console.log(`     ✅ GTD 限价单已挂出 (orderID: ${orderID})`);
                            results.push({ symbol, success: true, amount: phaseBet, direction });
                        } else {
                            const classified = classifyLiveSubmitIssue(resp);
                            console.log(`     ❌ 限价单提交失败: ${classified.reason}`);
                            results.push({ symbol, success: false, error: classified.reason });
                        }
                        break;  // success
                    } catch (orderErr) {
                        if (orderRetry < RETRY_LIMIT - 1) {
                            console.log(`     ⚠️ 限价单下单失败，${RETRY_INTERVAL_MS/1000}s 后重试 [${orderRetry+1}/${RETRY_LIMIT}]`);
                            await new Promise(r => setTimeout(r, RETRY_INTERVAL_MS));
                        } else {
                            const msg = parseErrorReason(orderErr);
                            console.log(`     ❌ 限价单 ${RETRY_LIMIT} 次重试均失败: ${msg}`);
                            results.push({ symbol, success: false, error: msg });
                        }
                    }
                }
            } else if (this.env.TRADING_MODE === TradingMode.LIVE) {
                console.error(`     ❌ [LIVE 防护] clobClient 未初始化，拒绝下单`);
                results.push({ symbol, success: false, error: 'clobClient 未初始化' });
            } else {
                console.log(`     📋 [回测] 记录 Phase 1 限价单`);
                results.push({ symbol, success: true, amount: phaseBet, direction });
            }
        }

        const successful = results.filter((r) => r.success).length;
        console.log(`\n  📋 Phase 1 摘要: ${successful}/${results.length} 笔限价单已挂出`);
        console.log('─'.repeat(60) + '\n');

        return results;
    }

    /**
     * Phase 2: 确认/取消/加仓
     * - 方向一致: 用剩余仓位下限价单 @best_ask + premium，并保留 Phase 1 单
     * - 方向反转: 取消 Phase 1 未成交的限价单
     */
    async executePhase2(opts: {
        targetPeriodEndTs: number;
        predictions: PredictionResult[];
        betFraction: number;
    }): Promise<ExecutionResult[]> {
        const { targetPeriodEndTs, predictions, betFraction } = opts;
        const results: ExecutionResult[] = [];

        console.log('\n' + '═'.repeat(60));
        console.log(`  🟢 Phase 2 确认/取消 - ${new Date().toLocaleString('zh-CN')}`);
        const rangeBeijing = this._fmtPeriod(targetPeriodEndTs);
        console.log(`  📌 目标周期: ${rangeBeijing}`);
        console.log(`  💰 剩余仓位比: ${(betFraction * 100).toFixed(0)}%`);
        console.log('═'.repeat(60));

        printPredictions(predictions);

        const symbolsToFetch = [...new Set(predictions.map((p) => p.symbol.replace(/\/.*$/, '')))];
        const markets = await findAllMarkets(this.env.TIMEFRAME, targetPeriodEndTs, symbolsToFetch);
        for (const result of markets) {
            this.state.marketCache.set(result.symbol, result);
        }

        for (const prediction of predictions) {
            const symbol = prediction.symbol.replace('/USDT', '');
            const trackerKey = `${targetPeriodEndTs}_${symbol}`;
            const phase1Order = this.phaseOrderTracker.get(trackerKey);

            if (!phase1Order) {
                console.log(`  ℹ️ ${symbol}: 无 Phase 1 订单记录，按正常流程下单`);
                // 没有 Phase 1 记录，作为新单处理
                const marketResult = this.state.marketCache.get(symbol);
                if (marketResult) {
                    const result = await this.executePrediction(prediction, marketResult, { targetPeriodEndTs });
                    results.push(result);
                }
                continue;
            }

            const phase1Direction = phase1Order.direction;
            const phase2Direction = prediction.direction;

            if (phase2Direction === phase1Direction) {
                // ─── 方向一致: 保留 Phase 1 + Phase 2 下剩余仓位 ──────
                const p1Filled = phase1Order.filled;
                const p1Amount = phase1Order.amountUsd;
                console.log(`\n  ✅ ${symbol}: Phase 2 确认方向 ${phase2Direction} (与 Phase 1 一致)`);
                console.log(`     Phase 1 状态: ${p1Filled ? `已成交 $${p1Amount.toFixed(2)}` : '未成交(挂单中)'} (orderID: ${phase1Order.orderID})`);

                // Phase 2: 下剩余仓位
                // 如果 Phase 1 已成交且扣了资金，executePrediction 基于 currentCapital 自然会算更少
                // 如果 Phase 1 未成交，资金没扣过，executePrediction 按完整资金计算
                const marketResult = this.state.marketCache.get(symbol);
                if (marketResult) {
                    const result = await this.executePrediction(prediction, marketResult, { targetPeriodEndTs });
                    results.push(result);
                }

                // Phase 1 订单保持不动（GTC 继续挂单 / 已成交保留）
            } else {
                // ─── 方向反转: 取消 Phase 1 ──────────────────
                const p1Filled = phase1Order.filled;
                const p1Amount = phase1Order.amountUsd;
                console.log(`\n  🔄 ${symbol}: Phase 2 方向反转 ${phase1Direction} → ${phase2Direction || 'SKIP'}`);
                console.log(`     取消 Phase 1 限价单 (orderID: ${phase1Order.orderID})`);
                console.log(`     Phase 1 状态: ${p1Filled ? `已成交 $${p1Amount.toFixed(2)} (无法撤回)` : '未成交(将取消)'}`);

                if (!p1Filled) {
                    // 未成交: 取消限价单，无需退资金（因为没扣过）
                    await this.cancelPhaseOrder(phase1Order);
                } else {
                    // 已成交: 真实交易中无法撤回已成交部分
                    // 模拟模式: 已成交的钱已经扣了，方向反转意味着这部分将按原方向结算（可能亏损）
                    console.log(`     ⚠️ Phase 1 已成交部分 $${p1Amount.toFixed(2)} 无法撤回，将按原方向 ${phase1Direction} 结算`);
                }
                this.phaseOrderTracker.delete(trackerKey);

                // 如果 Phase 2 有新方向且达到阈值，下新单
                if (phase2Direction && prediction.confidence >= this.env.PHASE2_MIN_CONFIDENCE) {
                    const marketResult = this.state.marketCache.get(symbol);
                    if (marketResult) {
                        const result = await this.executePrediction(prediction, marketResult, { targetPeriodEndTs });
                        results.push(result);
                    }
                } else {
                    console.log(`     ⏭️ Phase 2 置信度不足或无方向，不下新单`);
                    results.push({ symbol, success: false, error: 'Phase 2 reversed, confidence too low' });
                }
            }
        }

        // 清理过期的 Phase 1 跟踪
        this.cleanupPhaseOrders();

        const successful = results.filter((r) => r.success).length;
        console.log(`\n  📋 Phase 2 摘要: ${successful}/${results.length} 笔处理完成`);
        console.log('─'.repeat(60) + '\n');

        return results;
    }

    /**
     * 取消一个 Phase 1 的 GTC 限价单
     */
    private async cancelPhaseOrder(entry: PhaseOrderEntry): Promise<boolean> {
        if (this.env.TRADING_MODE === TradingMode.SIMULATION) {
            console.log(`     🗑️ [模拟] 取消 Phase 1 限价单 ${entry.orderID}`);
            return true;
        }

        if (this.env.TRADING_MODE === TradingMode.LIVE && this.clobClient) {
            try {
                await this.clobClient.cancelOrder({ orderID: entry.orderID });
                console.log(`     🗑️ 已取消 Phase 1 限价单 ${entry.orderID}`);
                return true;
            } catch (error) {
                const msg = parseErrorReason(error);
                // 已成交或已取消的订单取消会失败，这是正常的
                console.log(`     ⚠️ 取消失败 (可能已成交): ${msg}`);
                return false;
            }
        }

        return false;
    }

    /**
     * 清理过期的 Phase 1 订单跟踪（超过 15 分钟的周期）
     */
    private cleanupPhaseOrders(): void {
        const now = Math.floor(Date.now() / 1000);
        const keysToDelete: string[] = [];
        for (const [key, entry] of this.phaseOrderTracker.entries()) {
            // 周期结束 = targetPeriodEndTs + PERIOD_SECONDS
            if (now > entry.targetPeriodEndTs + this.periodSeconds + 60) {
                keysToDelete.push(key);
            }
        }
        for (const key of keysToDelete) {
            this.phaseOrderTracker.delete(key);
        }
        if (keysToDelete.length > 0) {
            console.log(`  🧹 清理 ${keysToDelete.length} 个过期 Phase 1 订单跟踪`);
        }
    }

    /**
     * 获取 Phase 1 订单跟踪器（供 prediction_index.ts 使用）
     */
    getPhaseOrderTracker(): Map<string, PhaseOrderEntry> {
        return this.phaseOrderTracker;
    }

    // ════════════════════════════════════════════════════════════
    // 单限价单方法 (Single Limit Order — 替代旧双腿下单)
    // ════════════════════════════════════════════════════════════

    /**
     * 单限价单: 在指定限价下单。
     *
     * 模拟成交精确流程（完美模拟真实 Polymarket）：
     * 1. 下单时: 冻结 betAmount 资金
     * 2. 即时成交检查: 查 orderbook，asks <= limitPrice 的全部成交
     *    - 完全成交: 扣 filledAmount
     *    - 部分成交: 扣 filledAmount，剩余保持冻结
     *    - 无法成交: 全额冻结，挂 GTC
     * 3. 结单时（下一个预测周期或超时 15min）:
     *    - 取消剩余未成交部分
     *    - 精确退还: returnAmount = frozenAmount - totalFilledAmount
     *    - 更新资金: currentCapital += returnAmount
     * 4. 盈亏计算: 只基于实际成交量计算，未成交部分不算
     */
    async executeLimitOrder(opts: {
        targetPeriodEndTs: number;
        predictions: PredictionResult[];
        limitPrice: number;
    }): Promise<ExecutionResult[]> {
        const { targetPeriodEndTs, predictions, limitPrice } = opts;
        const results: ExecutionResult[] = [];
        await this.prepareLiveCooldownStateBeforeOrder();

        console.log('\n' + '═'.repeat(60));
        console.log(`  📝 单限价单 @$${limitPrice.toFixed(2)} - ${new Date().toLocaleString('zh-CN')}`);
        const rangeBeijing = this._fmtPeriod(targetPeriodEndTs);
        console.log(`  📌 目标周期: ${rangeBeijing}`);
        console.log('═'.repeat(60));

        printPredictions(predictions);

        const symbolsToFetch = [...new Set(predictions.map((p) => p.symbol.replace(/\/.*$/, '')))];
        const markets = await findAllMarkets(this.env.TIMEFRAME, targetPeriodEndTs, symbolsToFetch);
        for (const result of markets) {
            this.state.marketCache.set(result.symbol, result);
        }
        /** 本轮已对冷却做过自减的 key（按配置可为 symbol 或 symbol+direction） */
        const cooldownDecrementedThisRound = new Set<string>();

        for (const prediction of predictions) {
            const symbol = prediction.symbol.replace('/USDT', '');
            const marketResult = this.state.marketCache.get(symbol);
            const direction = prediction.direction!;

            if (!marketResult || !marketResult.market) {
                console.log(`  ❌ ${symbol}: 市场数据未缓存`);
                results.push({ symbol, success: false, error: '市场数据未缓存' });
                continue;
            }

            const token = getTokenForDirection(marketResult.market, direction);
            if (!token) {
                console.log(`  ❌ ${symbol}: 无法获取 ${direction} token`);
                results.push({ symbol, success: false, error: 'token not found' });
                continue;
            }

            // ─── 风控检查（回撤熔断 + 冷却），与优化器/回测对齐 ───
            const riskCheck = this.checkRiskControls();
            if (riskCheck.shouldPause) {
                console.log(`\n🛡️  跳过 ${symbol}: 风险控制: ${riskCheck.reason || '交易已暂停'}`);
                results.push({ symbol, success: false, error: `风险控制: ${riskCheck.reason}` });
                continue;
            }
            // 按币种绝对止损当前已全局禁用，其余运行时 guards 保持原样。
            if (this.isSymbolAbsoluteStopped(symbol)) {
                const floor = this.getPerCoinAbsoluteFloor();
                console.log(`\n🛑 跳过 ${symbol}: 绝对止损 (该币种资金 $${(this.getCoinCapital(symbol) ?? 0).toFixed(2)} < $${floor.toFixed(2)})`);
                results.push({ symbol, success: false, error: `绝对止损: ${symbol} 资金低于该币种初始30%` });
                continue;
            }

            // ─── Cooldown 冷却检查（默认按币种+方向独立）───
            {
                const cooldownKey = this.buildCooldownKey(symbol, direction);
                const cooldownLabel = this.formatCooldownLabel(symbol, direction);
            const pendingResolutionGuard = this.shouldBlockForPendingCooldownResolution(symbol);
            if (pendingResolutionGuard.blocked) {
                console.log(`\n🧊 跳过 ${symbol}: 冷却待确认: ${pendingResolutionGuard.label} 已有 ${pendingResolutionGuard.consecutiveBeforeUnknown} 连输，上一根 ${pendingResolutionGuard.unresolvedMarketSlug || ''} 已结束但结果尚未入账`);
                results.push({ symbol, success: false, error: `冷却待确认 (${pendingResolutionGuard.label})` });
                this.appendCooldownBlockedOpportunity({
                        prediction,
                        marketResult,
                        symbol,
                        direction: direction!,
                        targetPeriodEndTs,
                        remainingBarsBefore: 0,
                        remainingBarsAfter: 0,
                        decrementedThisRound: false,
                        reasonCode: pendingResolutionGuard.reasonCode,
                });
                continue;
            }
            const persistedWindowGuard = this.getPersistedCooldownWindowBlock(symbol, direction, targetPeriodEndTs);
            if (persistedWindowGuard.blocked) {
                this.setCooldownRemaining(symbol, direction, persistedWindowGuard.remainingBarsAfter);
                console.log(`\n🧊 跳过 ${symbol}: ${persistedWindowGuard.error || `冷却中: ${persistedWindowGuard.label}`}`);
                results.push({ symbol, success: false, error: persistedWindowGuard.error || `冷却中 (${persistedWindowGuard.label})` });
                this.appendCooldownBlockedOpportunity({
                    prediction,
                    marketResult,
                    symbol,
                    direction: direction!,
                    targetPeriodEndTs,
                    remainingBarsBefore: persistedWindowGuard.remainingBarsBefore,
                    remainingBarsAfter: persistedWindowGuard.remainingBarsAfter,
                    decrementedThisRound: true,
                    reasonCode: 'cooldown_bars_blocked',
                });
                continue;
            }
            const cd = this.getCooldownRemaining(symbol, direction);
            if (cd > 0) {
                if (cooldownDecrementedThisRound.has(cooldownKey)) {
                    console.log(`\n🧊 跳过 ${symbol}: 冷却中: ${cooldownLabel} 剩余 ${cd} 个周期 (本轮已自减过)`);
                    results.push({ symbol, success: false, error: `冷却中 (${cooldownLabel}剩${cd}个周期)` });
                        this.appendCooldownBlockedOpportunity({
                            prediction,
                            marketResult,
                            symbol,
                            direction: direction!,
                            targetPeriodEndTs,
                            remainingBarsBefore: cd,
                            remainingBarsAfter: cd,
                            decrementedThisRound: true,
                            reasonCode: 'cooldown_bars_blocked',
                        });
                    } else {
                        const nextCd = Math.max(0, Number(cd) - 1);
                        this.setCooldownRemaining(symbol, direction, nextCd);
                        cooldownDecrementedThisRound.add(cooldownKey);
                        console.log(`\n🧊 跳过 ${symbol}: 冷却中: ${cooldownLabel} 剩余 ${nextCd} 个周期 (cooldown_bars=${this.env.COOLDOWN_BARS})`);
                        results.push({ symbol, success: false, error: `冷却中 (${cooldownLabel}剩${nextCd}个周期)` });
                        this.appendCooldownBlockedOpportunity({
                            prediction,
                            marketResult,
                            symbol,
                            direction: direction!,
                            targetPeriodEndTs,
                            remainingBarsBefore: cd,
                            remainingBarsAfter: nextCd,
                            decrementedThisRound: false,
                            reasonCode: 'cooldown_bars_blocked',
                        });
                    }
                    continue;
                }
            }

            // ─── Edge 过滤：限价单用 limitPrice 算 edge（与 executePrediction 一致，避免 bestAsk 高时误拦）───
            let earlyBookSnapshot: QueueAwareBookSnapshot = {
                bestAskPrice: 999,
                asks: [],
                bids: [],
                hadBidsData: false,
            };
            try {
                if (token.tokenId) {
                    const ob = await getOrderBook(token.tokenId);
                    earlyBookSnapshot = buildQueueAwareBookSnapshot(ob);
                }
            } catch { /* orderbook 获取失败时跳过 edge 检查，继续用 limitPrice */ }
            const earlyBestAsk = earlyBookSnapshot.bestAskPrice;

            const directionalMinEdge = this.getDirectionalMinEdge(direction);
            const requestedNotional = this.resolveRequestedNotionalUsd(prediction.confidence, limitPrice, symbol, prediction.direction, prediction);
            const requestedTotalBet = requestedNotional.requestedNotionalUsd;
            const consensusGate = this.getConsensusGateFailure(prediction, direction);
            if (consensusGate) {
                const reason = `共识分不足: score=${(consensusGate.score * 100).toFixed(2)}% < threshold=${(consensusGate.threshold * 100).toFixed(2)}%`;
                console.log(`  ⏭️ ${symbol}: ${reason}`);
                this.addExecutionSkippedEntry({
                    symbol,
                    direction,
                    confidence: prediction.confidence,
                    reason,
                    executionDecision: 'consensus_threshold_blocked',
                    requestedAmountUsd: requestedTotalBet,
                    requestedShares: 0,
                    originalRequestedShares: 0,
                    market: {
                        title: marketResult.market.title,
                        conditionId: marketResult.market.conditionId,
                        slug: marketResult.market.slug,
                    },
                    token,
                });
                results.push({ symbol, success: false, error: reason });
                continue;
            }
            if (this.env.EDGE_FILTER_ENABLED && limitPrice > 0 && limitPrice < 1) {
                const edgeOddsForLimit = (1.0 - limitPrice) / limitPrice;
                const edgeConf = prediction.confidence ?? 0;
                const edgeAtLimit = edgeConf * edgeOddsForLimit - (1.0 - edgeConf);
                if (edgeAtLimit < 0) {
                    console.log(`  ⏭️ ${symbol}: 边际为负 edge@limit=${(edgeAtLimit * 100).toFixed(2)}% (limit$${limitPrice}, bestAsk$${earlyBestAsk.toFixed(3)}, conf=${(edgeConf * 100).toFixed(1)}%)`);
                    this._logEdgeSkip({ symbol, confidence: edgeConf, bestAsk: earlyBestAsk, edge: edgeAtLimit, minEdge: directionalMinEdge, limitPrice, reason: 'negative_edge' });
                    results.push({ symbol, success: false, error: `边际为负 (${(edgeAtLimit * 100).toFixed(1)}%)` });
                    continue;
                }
                if (edgeAtLimit < directionalMinEdge) {
                    console.log(`  ⏭️ ${symbol}: edge不足 ${(edgeAtLimit * 100).toFixed(2)}% < ${(directionalMinEdge * 100).toFixed(2)}% (limit$${limitPrice}, bestAsk$${earlyBestAsk.toFixed(3)}, conf=${(edgeConf * 100).toFixed(1)}%)`);
                    this._logEdgeSkip({ symbol, confidence: edgeConf, bestAsk: earlyBestAsk, edge: edgeAtLimit, minEdge: directionalMinEdge, limitPrice, reason: 'insufficient_edge' });
                    results.push({ symbol, success: false, error: `edge不足 (${(edgeAtLimit * 100).toFixed(1)}% < ${(directionalMinEdge * 100).toFixed(1)}%)` });
                    continue;
                }
            }

            const availableCapital = Math.max(0, this.getCoinCapital(symbol) || 0);
            const riskUpperLimit = this.getCapitalRiskUpperLimit(availableCapital);
            const fixedConstraint = await this.getExecutionConstraint(token.tokenId, marketResult.market.slug);
            const fixedPlan = this.buildFixedOrderExecutionPlan({
                requestedAmountUsd: requestedTotalBet,
                price: limitPrice,
                maxBudgetUsd: Math.min(availableCapital, riskUpperLimit),
                constraint: fixedConstraint,
            });
            if (!fixedPlan.executable) {
                const reason = fixedPlan.failureReason || '固定价订单 uplift 后超过资金/风控上限';
                console.log(`  ⏭️ ${symbol}: ${reason}`);
                this.addExecutionSkippedEntry({
                    symbol,
                    direction,
                    confidence: prediction.confidence,
                    reason,
                    executionDecision: 'uplifted_fixed_order_to_min_shares',
                    requestedAmountUsd: requestedTotalBet,
                    requestedShares: fixedPlan.requestedShares,
                    originalRequestedShares: fixedPlan.requestedShares,
                    market: {
                        title: marketResult.market.title,
                        conditionId: marketResult.market.conditionId,
                        slug: marketResult.market.slug,
                    },
                    token,
                    exchangeMinOrderSizeShares: fixedConstraint.minOrderSizeShares,
                    upliftedFixedOrders: fixedPlan.uplifted ? 1 : 0,
                    upliftedNotionalUsd: fixedPlan.upliftedNotionalUsd,
                    gt5PreservationStatus: fixedPlan.gt5PreservationStatus,
                    compressionReason: fixedPlan.compressionReason,
                });
                results.push({ symbol, success: false, error: reason });
                continue;
            }

            const totalBet = fixedPlan.finalAmountUsd;
            const shares = fixedPlan.finalShares;

            // ─── 计算 Edge/Kelly（日志展示用，同时显示 limitPrice 和 bestAsk 的 edge）───
            const buyPrice = limitPrice;
            const realOdds = buyPrice > 0 && buyPrice < 1 ? (1.0 - buyPrice) / buyPrice : 0;
            const _conf = prediction.confidence ?? 0;
            const _q = 1.0 - _conf;
            const realEdge = _conf * realOdds - _q;
            const kellyFraction = realOdds > 0 ? Math.max(0, (_conf * realOdds - _q) / realOdds) : 0;
            const marketEdgeForLog = earlyBestAsk < 1 ? _conf * ((1.0 - earlyBestAsk) / earlyBestAsk) - _q : realEdge;
            const coinCapForDisplay = this.getCoinCapital(symbol);
            const betPercent = (totalBet / (coinCapForDisplay || 1) * 100).toFixed(1);
            const confidencePct = (_conf * 100).toFixed(1);
            const effectiveThreshold = Number(prediction.effectiveThreshold);
            const thresholdPct = ((Number.isFinite(effectiveThreshold) ? effectiveThreshold : this.env.PROB_THRESHOLD) * 100).toFixed(0);
            const directionIcon = direction === 'UP' ? '📈 UP' : '📉 DOWN';
            const modeText = this.env.TRADING_MODE === TradingMode.SIMULATION ? '模拟交易'
                : this.env.TRADING_MODE === TradingMode.LIVE ? '真实交易' : '回测';
            const slugShort = marketResult.market.slug
                ? (marketResult.market.slug.length > 35 ? marketResult.market.slug.slice(0, 35) + '...' : marketResult.market.slug)
                : '';

            // 详情框（与旧模型 executePrediction 风格对齐）
            console.log(`\n┌${'─'.repeat(58)}┐`);
            console.log(`│ ${(symbol + ' ' + directionIcon + ' @$' + limitPrice.toFixed(2) + ' [' + slugShort + ']').padEnd(57)}│`);
            console.log(`├${'─'.repeat(58)}┤`);
            console.log(`│ 北京时间:   ${nowBeijing().padEnd(45)}│`);
            console.log(`│ 置信度:     ${(confidencePct + '% (阈值: ' + thresholdPct + '%)').padEnd(45)}│`);
            console.log(`│ Edge/Kelly: ${('edge=' + (realEdge * 100).toFixed(2) + '% kelly=' + (kellyFraction * 100).toFixed(1) + '% mktEdge=' + (marketEdgeForLog * 100).toFixed(2) + '%').padEnd(45)}│`);
            if (earlyBestAsk < 1) console.log(`│ BestAsk:    ${('$' + earlyBestAsk.toFixed(3) + (this.env.EDGE_FILTER_ENABLED ? ' (minEdge=' + (this.env.MIN_EDGE * 100).toFixed(1) + '%)' : '')).padEnd(45)}│`);
            const executionDecisionText = fixedPlan.uplifted
                ? `固定价单 uplift 至 ${fixedConstraint.minOrderSizeShares.toFixed(2)} shares`
                : '最终执行判定通过';
            console.log(`│ 执行判定:   ${executionDecisionText.padEnd(45)}│`);
            console.log(`│ 下注金额:   ${('$' + totalBet.toFixed(2) + ' (占资金 ' + betPercent + '%, 原$' + requestedTotalBet.toFixed(2) + ')').padEnd(45)}│`);
            console.log(`│ 限价/赔率:  ${('$' + limitPrice.toFixed(2) + ' (赔率 ' + realOdds.toFixed(3) + ')').padEnd(45)}│`);
            console.log(`│ 预计份额:   ${('~' + shares.toFixed(2) + ' 个代币 (min=' + fixedConstraint.minOrderSizeShares.toFixed(2) + ')').padEnd(45)}│`);
            console.log(`│ 交易模式:   ${modeText.padEnd(45)}│`);
            console.log(`└${'─'.repeat(58)}┘`);

            if (this.env.TRADING_MODE === TradingMode.SIMULATION) {
                // ─── 模拟模式: 精确模拟限价单成交 ───
                // Step 1: 冻结资金
                const frozenAmount = totalBet;
                this.adjustCoinCapital(symbol, -frozenAmount);

                // 回撤熔断已移除（下注后检测点）

                // Step 2: 复用 edge 检查阶段已获取的 orderbook 数据（避免重复 API 调用）
                const simulatedFill = simulateQueueAwareBuyFillWithHaircut(earlyBookSnapshot, limitPrice, frozenAmount);
                const bestAskPrice = simulatedFill.bestAskPrice;
                const fillableAmount = simulatedFill.haircutAdjustedFillableUsd;
                const avgFillPrice = simulatedFill.avgFillPrice;
                const queueCompetitionUsd = simulatedFill.queueCompetitionUsd;
                const effectiveFillableUsd = simulatedFill.effectiveFillableUsd;

                let filledAmount = 0;
                let unfilled = frozenAmount;
                let status: 'filled' | 'partial' | 'pending';

                if (bestAskPrice <= limitPrice && fillableAmount >= frozenAmount) {
                    // 完全成交
                    filledAmount = frozenAmount;
                    unfilled = 0;
                    status = 'filled';
                    console.log(`     ✅ [模拟] 完全成交 (best_ask=$${bestAskPrice.toFixed(3)}, 均价=$${avgFillPrice.toFixed(4)}, 限价=$${limitPrice}, 排队竞争=$${queueCompetitionUsd.toFixed(2)})`);
                } else if (bestAskPrice <= limitPrice && fillableAmount > 0) {
                    // 部分成交
                    filledAmount = Math.min(fillableAmount, frozenAmount);
                    unfilled = roundUsd(frozenAmount - filledAmount);
                    status = 'partial';
                    const pct = ((filledAmount / frozenAmount) * 100).toFixed(1);
                    console.log(`     ⚠️ [模拟] 部分成交 ${pct}% (best_ask=$${bestAskPrice.toFixed(3)}, 均价=$${avgFillPrice.toFixed(4)}, 有效流动性=$${fillableAmount.toFixed(2)}, 排队竞争=$${queueCompetitionUsd.toFixed(2)})`);
                } else {
                    // 无法成交 — 挂 GTC，资金保持冻结，加入持续监控池
                    status = 'pending';
                    console.log(`     ⏳ [模拟] 挂单中 (best_ask=$${bestAskPrice.toFixed(3)} > limit=$${limitPrice})`);
                }

                // Step 3: 处理成交/未成交资金
                if (status === 'filled') {
                    // 完全成交：资金已扣，无需退还
                    console.log(`     💰 成交 $${filledAmount.toFixed(2)}, 剩余资金 $${this.state.currentCapital.toFixed(2)}`);
                } else if (status === 'partial') {
                    // 部分成交：退还未成交部分的同时，将未成交部分加入监控池
                    // 注：部分成交的未成交部分也放入挂单池继续等待
                    const pendingId = `sim_${targetPeriodEndTs}_${symbol}_${Date.now()}`;
                    const createdAt = new Date();
                    const expiresAt = new Date((targetPeriodEndTs + this.periodSeconds) * 1000); // 周期结束
                    this.queuePendingSimOrder({
                        id: pendingId,
                        symbol,
                        direction,
                        frozenAmount: unfilled,
                        limitPrice,
                        tokenId: token.tokenId,
                        tokenOutcome: token.outcome,
                        confidence: prediction.confidence,
                        marketTitle: marketResult.market.title,
                        conditionId: marketResult.market.conditionId,
                        marketSlug: marketResult.market.slug,
                        targetPeriodEndTs,
                        createdAt,
                        expiresAt,
                        bestAskAtEntry: bestAskPrice,
                        queueCompetitionUsdAtEntry: queueCompetitionUsd,
                        effectiveFillableUsdAtEntry: effectiveFillableUsd,
                        avgActualFillPriceAtEntry: avgFillPrice,
                        checking: false,
                    });
                    console.log(`     💰 部分成交 $${filledAmount.toFixed(2)}, 未成交 $${unfilled.toFixed(2)} 冻结挂单中`);
                    console.log(`     📡 加入持续监控 (每${PENDING_SIM_MONITOR_INTERVAL_MS / 1000}s查订单簿, 到期: ${expiresAt.toLocaleTimeString('zh-CN')})`);
                    this.startPendingSimMonitor();
                } else {
                    // 完全未成交：全额冻结，加入监控池
                    const pendingId = `sim_${targetPeriodEndTs}_${symbol}_${Date.now()}`;
                    const createdAt = new Date();
                    const expiresAt = new Date((targetPeriodEndTs + this.periodSeconds) * 1000); // 周期结束
                    this.queuePendingSimOrder({
                        id: pendingId,
                        symbol,
                        direction,
                        frozenAmount,
                        limitPrice,
                        tokenId: token.tokenId,
                        tokenOutcome: token.outcome,
                        confidence: prediction.confidence,
                        marketTitle: marketResult.market.title,
                        conditionId: marketResult.market.conditionId,
                        marketSlug: marketResult.market.slug,
                        targetPeriodEndTs,
                        createdAt,
                        expiresAt,
                        bestAskAtEntry: bestAskPrice,
                        queueCompetitionUsdAtEntry: queueCompetitionUsd,
                        effectiveFillableUsdAtEntry: effectiveFillableUsd,
                        avgActualFillPriceAtEntry: avgFillPrice,
                        checking: false,
                    });
                    console.log(`     💰 资金 $${frozenAmount.toFixed(2)} 冻结挂单中, 剩余可用 $${this.state.currentCapital.toFixed(2)}`);
                    console.log(`     📡 加入持续监控 (每${PENDING_SIM_MONITOR_INTERVAL_MS / 1000}s查订单簿, 到期: ${expiresAt.toLocaleTimeString('zh-CN')})`);
                    this.startPendingSimMonitor();
                }

                // 记录日志（只记录实际成交部分）
                if (filledAmount > 0) {
                    this.logger.addEntry({
                        symbol,
                        direction,
                        confidence: prediction.confidence,
                        amount: filledAmount,
                        mode: this.env.TRADING_MODE,
                        status: 'executed',
                        marketTitle: marketResult.market.title,
                        conditionId: marketResult.market.conditionId,
                        tokenId: token.tokenId,
                        marketSlug: marketResult.market.slug,
                        tokenOutcome: token.outcome,
                        tokenPrice: avgFillPrice,             // 实际加权均价（非限价）
                        avgActualFillPrice: avgFillPrice,
                        limitPriceConfigured: limitPrice,     // 审计：配置限价
                        bestAskAtEntry: bestAskPrice,
                        queueCompetitionUsdAtEntry: queueCompetitionUsd,
                        effectiveFillableUsdAtEntry: effectiveFillableUsd,
                        exchangeMinOrderSizeShares: fixedConstraint.minOrderSizeShares,
                        upliftedFixedOrders: fixedPlan.uplifted ? 1 : 0,
                        upliftedNotionalUsd: fixedPlan.upliftedNotionalUsd,
                        executionDecision: fixedPlan.uplifted ? 'uplifted_fixed_order_to_min_shares' : 'executed',
                        requestedAmountUsd: totalBet,
                        requestedShares: fixedPlan.finalShares,
                        originalRequestedShares: fixedPlan.requestedShares,
                        gt5PreservationStatus: fixedPlan.gt5PreservationStatus,
                        compressionReason: fixedPlan.compressionReason,
                    });
                }

                results.push({
                    symbol,
                    success: filledAmount > 0,
                    amount: filledAmount,
                    direction,
                    pendingMonitor: status === 'pending' || (status === 'partial' && unfilled > 0),
                });
            } else if (this.env.TRADING_MODE === TradingMode.LIVE && this.clobClient) {
                for (let orderRetry = 0; orderRetry < RETRY_LIMIT; orderRetry++) {
                    try {
                        const gtdExpiration = this.computeLiveGtdExpiration(targetPeriodEndTs);
                        if (!gtdExpiration) {
                            const reason = this.buildLiveGtdExpiredReason(targetPeriodEndTs);
                            results.push({ symbol, success: false, error: reason });
                            break;
                        }
                        const userOrder = {
                            tokenID: token.tokenId,
                            price: limitPrice,
                            size: shares,
                            side: Side.BUY,
                            expiration: gtdExpiration,
                        };
                        await this.ensureClobHeartbeatFresh();
                        const signedOrder = await this.clobClient.createOrder(userOrder, { tickSize: '0.01' });
                        const resp = await this.clobClient.postOrder(signedOrder, OrderType.GTD);

                        const immediatePostError = this.getPostOrderImmediateError(resp);
                        if (immediatePostError) {
                            this.rememberLiveExchangeConstraints(immediatePostError, token.tokenId, marketResult.market.slug);
                            this.addExecutionSkippedEntry({
                                symbol,
                                direction,
                                confidence: prediction.confidence,
                                reason: immediatePostError,
                                executionDecision: 'exchange_rejected',
                                requestedAmountUsd: totalBet,
                                requestedShares: shares,
                                originalRequestedShares: fixedPlan.requestedShares,
                                market: {
                                    title: marketResult.market.title,
                                    conditionId: marketResult.market.conditionId,
                                    slug: marketResult.market.slug,
                                },
                                token,
                                exchangeMinOrderSizeShares: fixedConstraint.minOrderSizeShares,
                                exchangeRejectedRungs: 1,
                                upliftedFixedOrders: fixedPlan.uplifted ? 1 : 0,
                                upliftedNotionalUsd: fixedPlan.upliftedNotionalUsd,
                                gt5PreservationStatus: fixedPlan.gt5PreservationStatus,
                                compressionReason: fixedPlan.compressionReason,
                            });
                            results.push({ symbol, success: false, error: immediatePostError });
                            break;
                        }
                        const postedOrderId = this.extractPostedOrderId(resp);
                        if (postedOrderId) {
                            const orderID = postedOrderId;
                            const officialAcceptance = await this.verifyOfficialOrderAcceptance(orderID, token.tokenId);
                            if (!officialAcceptance.accepted) {
                                const reason = `官网未确认自动订单收单: orderID=${orderID} reason=${officialAcceptance.reason || 'unknown'} post=${this.summarizePostOrderResponse(resp)}`;
                                this.rememberLiveExchangeConstraints(reason, token.tokenId, marketResult.market.slug);
                                this.addExecutionSkippedEntry({
                                    symbol,
                                    direction,
                                    confidence: prediction.confidence,
                                    reason,
                                    executionDecision: 'exchange_rejected',
                                    requestedAmountUsd: totalBet,
                                    requestedShares: shares,
                                    originalRequestedShares: fixedPlan.requestedShares,
                                    market: {
                                        title: marketResult.market.title,
                                        conditionId: marketResult.market.conditionId,
                                        slug: marketResult.market.slug,
                                    },
                                    token,
                                    exchangeMinOrderSizeShares: fixedConstraint.minOrderSizeShares,
                                    exchangeRejectedRungs: 1,
                                    upliftedFixedOrders: fixedPlan.uplifted ? 1 : 0,
                                    upliftedNotionalUsd: fixedPlan.upliftedNotionalUsd,
                                    gt5PreservationStatus: fixedPlan.gt5PreservationStatus,
                                    compressionReason: fixedPlan.compressionReason,
                                });
                                results.push({ symbol, success: false, error: reason });
                                break;
                            }
                            const immediateFill = this.buildOfficialImmediateFill(
                                officialAcceptance.raw,
                                limitPrice,
                                shares,
                                totalBet,
                            );
                            const createdAtIso = new Date().toISOString();
                            if (immediateFill) {
                                console.log(`     ✅ GTD 限价单官网已成交 (orderID: ${orderID}, $${immediateFill.amountUsd.toFixed(2)}, ${immediateFill.source})`);
                            } else {
                                console.log(`     ✅ GTD 限价单已挂出 (orderID: ${orderID})`);
                            }

                            // LIVE 语义：官网 MATCHED/FILLED 必须立即记成交；只有 OPEN/LIVE 才记 pending。
                            const liveEntry = this.logger.addEntry({
                                symbol,
                                direction,
                                confidence: prediction.confidence,
                                amount: immediateFill ? immediateFill.amountUsd : 0,
                                mode: this.env.TRADING_MODE,
                                status: immediateFill ? 'executed' : 'pending',
                                marketTitle: marketResult.market.title,
                                conditionId: marketResult.market.conditionId,
                                tokenId: token.tokenId,
                                marketSlug: marketResult.market.slug,
                                tokenOutcome: token.outcome,
                                tokenPrice: limitPrice,
                                limitPriceConfigured: limitPrice,
                                orderId: orderID,
                                txHash: orderID,
                                requestedAmountUsd: totalBet,
                                pendingAmountUsd: immediateFill?.status === 'partially_filled'
                                    ? Math.max(0, roundUsd(totalBet - immediateFill.amountUsd))
                                    : (immediateFill ? 0 : totalBet),
                                requestedShares: shares,
                                originalRequestedShares: fixedPlan.requestedShares,
                                matchedShares: immediateFill ? immediateFill.tokens : 0,
                                avgActualFillPrice: immediateFill?.price,
                                fillSource: immediateFill?.source,
                                liveOrderStatus: immediateFill ? immediateFill.status : 'posted_pending',
                                exchangeMinOrderSizeShares: fixedConstraint.minOrderSizeShares,
                                upliftedFixedOrders: fixedPlan.uplifted ? 1 : 0,
                                upliftedNotionalUsd: fixedPlan.upliftedNotionalUsd,
                                executionDecision: immediateFill
                                    ? (fixedPlan.uplifted ? 'uplifted_fixed_order_immediate_fill' : 'official_immediate_fill')
                                    : (fixedPlan.uplifted ? 'uplifted_fixed_order_to_min_shares' : 'posted_pending'),
                                gt5PreservationStatus: fixedPlan.gt5PreservationStatus,
                                compressionReason: fixedPlan.compressionReason,
                            });
                            this.appendPendingOrderLedgerEvent({
                                event: 'created',
                                orderId: orderID,
                                symbol,
                                direction,
                                amountUsd: totalBet,
                                remainingFrozenAmount: totalBet,
                                limitPrice,
                                tokenId: token.tokenId,
                                tokenOutcome: token.outcome,
                                confidence: prediction.confidence,
                                marketTitle: marketResult.market.title,
                                conditionId: marketResult.market.conditionId,
                                marketSlug: marketResult.market.slug,
                                targetPeriodEndTs,
                                createdAt: createdAtIso,
                                expiresAt: new Date(gtdExpiration * 1000).toISOString(),
                                bestAsk: token.price,
                                officialAcceptanceStatus: officialAcceptance.status,
                                officialAcceptanceSource: officialAcceptance.source,
                                officialAcceptanceAt: createdAtIso,
                            });
                            if (immediateFill) {
                                this.appendPendingOrderLedgerEvent({
                                    event: immediateFill.status === 'partially_filled' ? 'partial_fill' : 'filled',
                                    orderId: orderID,
                                    symbol,
                                    direction,
                                    amountUsd: immediateFill.amountUsd,
                                    remainingFrozenAmount: immediateFill.status === 'partially_filled'
                                        ? Math.max(0, roundUsd(totalBet - immediateFill.amountUsd))
                                        : 0,
                                    limitPrice,
                                    tokenId: token.tokenId,
                                    tokenOutcome: token.outcome,
                                    confidence: prediction.confidence,
                                    marketTitle: marketResult.market.title,
                                    conditionId: marketResult.market.conditionId,
                                    marketSlug: marketResult.market.slug,
                                    targetPeriodEndTs,
                                    createdAt: createdAtIso,
                                    expiresAt: new Date(gtdExpiration * 1000).toISOString(),
                                    bestAsk: token.price,
                                    avgFillPrice: immediateFill.price,
                                    officialAcceptanceStatus: officialAcceptance.status,
                                    officialAcceptanceSource: officialAcceptance.source,
                                    officialAcceptanceAt: createdAtIso,
                                });
                                this.appendExecutionV2SettlementEventFromLog(
                                    liveEntry.id,
                                    immediateFill.status === 'partially_filled' ? 'chase_reprice_filled' : 'fully_filled',
                                    immediateFill.status === 'partially_filled' ? 'official_immediate_partial_fill' : 'official_immediate_full_fill',
                                );
                            }
                            console.log(`     📝 LIVE ${immediateFill ? '成交' : '挂单'}已记录日志 (${immediateFill ? immediateFill.status : 'pending'}, orderID=${orderID})`);

                            results.push({
                                symbol,
                                success: Boolean(immediateFill),
                                amount: immediateFill?.amountUsd || 0,
                                direction,
                                pendingMonitor: !immediateFill,
                            });
                        } else {
                            const classified = classifyLiveSubmitIssue(resp);
                            const reason = classified.reason;
                            console.log(`     ❌ 限价单提交失败: ${reason}`);
                            this.addExecutionSkippedEntry({
                                symbol,
                                direction,
                                confidence: prediction.confidence,
                                reason,
                                executionDecision: classified.status === 'allowance_zero' || classified.status === 'balance_insufficient' ? 'exchange_rejected' : 'fixed_order_exception',
                                requestedAmountUsd: totalBet,
                                requestedShares: shares,
                                originalRequestedShares: fixedPlan.requestedShares,
                                market: {
                                    title: marketResult.market.title,
                                    conditionId: marketResult.market.conditionId,
                                    slug: marketResult.market.slug,
                                },
                                token,
                                exchangeMinOrderSizeShares: fixedConstraint.minOrderSizeShares,
                                exchangeRejectedRungs: classified.status === 'allowance_zero' || classified.status === 'balance_insufficient' ? 1 : 0,
                                upliftedFixedOrders: fixedPlan.uplifted ? 1 : 0,
                                upliftedNotionalUsd: fixedPlan.upliftedNotionalUsd,
                                gt5PreservationStatus: fixedPlan.gt5PreservationStatus,
                                compressionReason: fixedPlan.compressionReason,
                            });
                            results.push({ symbol, success: false, error: reason });
                        }
                        break;  // success
                    } catch (orderErr) {
                        if (orderRetry < RETRY_LIMIT - 1) {
                            console.log(`     ⚠️ 限价单下单失败，${RETRY_INTERVAL_MS/1000}s 后重试 [${orderRetry+1}/${RETRY_LIMIT}]`);
                            await new Promise(r => setTimeout(r, RETRY_INTERVAL_MS));
                        } else {
                            const msg = orderErr instanceof Error ? orderErr.message : String(orderErr);
                            this.rememberLiveExchangeConstraints(msg, token.tokenId, marketResult.market.slug);
                            console.log(`     ❌ 限价单 ${RETRY_LIMIT} 次重试均失败: ${msg}`);
                            this.addExecutionSkippedEntry({
                                symbol,
                                direction,
                                confidence: prediction.confidence,
                                reason: msg,
                                executionDecision: this.isNonRetryableLiveReject(msg) ? 'exchange_rejected' : 'fixed_order_exception',
                                requestedAmountUsd: totalBet,
                                requestedShares: shares,
                                originalRequestedShares: fixedPlan.requestedShares,
                                market: {
                                    title: marketResult.market.title,
                                    conditionId: marketResult.market.conditionId,
                                    slug: marketResult.market.slug,
                                },
                                token,
                                exchangeMinOrderSizeShares: fixedConstraint.minOrderSizeShares,
                                exchangeRejectedRungs: this.isNonRetryableLiveReject(msg) ? 1 : 0,
                                upliftedFixedOrders: fixedPlan.uplifted ? 1 : 0,
                                upliftedNotionalUsd: fixedPlan.upliftedNotionalUsd,
                                gt5PreservationStatus: fixedPlan.gt5PreservationStatus,
                                compressionReason: fixedPlan.compressionReason,
                            });
                            results.push({ symbol, success: false, error: msg });
                        }
                    }
                }
            } else if (this.env.TRADING_MODE === TradingMode.LIVE) {
                console.error(`     ❌ [LIVE 防护] clobClient 未初始化，拒绝下单`);
                results.push({ symbol, success: false, error: 'clobClient 未初始化' });
            } else {
                console.log(`     📋 [回测] 记录限价单`);
                results.push({ symbol, success: true, amount: totalBet, direction });
            }
        }

        // 打印摘要（区分"监控中(挂单)"和真正的"失败"）
        const successful = results.filter((r) => r.success).length;
        const monitoring = results.filter((r) => r.pendingMonitor).length;
        const realFailed = results.filter((r) => !r.success && !r.pendingMonitor).length;
        const totalFilled = results.filter(r => r.success).reduce((s, r) => s + (r.amount || 0), 0);
        const parts: string[] = [`成交 ${successful} 笔`];
        if (monitoring > 0) {
            if (this.env.TRADING_MODE === TradingMode.SIMULATION) {
                const pendingFrozen = this.pendingSimOrders.reduce((s, o) => s + o.frozenAmount, 0);
                parts.push(`挂单监控中 ${monitoring} 笔 (冻结$${pendingFrozen.toFixed(2)})`);
            } else if (this.env.TRADING_MODE === TradingMode.LIVE) {
                parts.push(`挂单待成交 ${monitoring} 笔`);
            } else {
                parts.push(`挂单中 ${monitoring} 笔`);
            }
        }
        if (realFailed > 0) parts.push(`失败 ${realFailed} 笔`);
        console.log('\n' + '─'.repeat(60));
        console.log(`  📋 限价单摘要: ${parts.join(', ')}, 成交额 $${totalFilled.toFixed(2)}, 可用资金 $${this.state.currentCapital.toFixed(2)}`);

        // ─── 逆向选择统计（Adverse Selection Monitor）───
        // 追踪成交 vs 未成交订单的置信度差异，用于后续分析
        if (predictions.length > 0) {
            const filledConfs: number[] = [];
            const unfilledConfs: number[] = [];
            for (let i = 0; i < results.length && i < predictions.length; i++) {
                const conf = predictions[i]?.confidence ?? 0;
                if (results[i].success && (results[i].amount ?? 0) > 0) {
                    filledConfs.push(conf);
                } else if (results[i].pendingMonitor || results[i].error === '挂单未成交') {
                    unfilledConfs.push(conf);
                }
            }
            const avgFilled = filledConfs.length > 0
                ? (filledConfs.reduce((a, b) => a + b, 0) / filledConfs.length * 100).toFixed(1)
                : '-';
            const avgUnfilled = unfilledConfs.length > 0
                ? (unfilledConfs.reduce((a, b) => a + b, 0) / unfilledConfs.length * 100).toFixed(1)
                : '-';
            const fillRate = predictions.length > 0
                ? ((filledConfs.length / predictions.length) * 100).toFixed(0)
                : '0';
            console.log(`  📊 逆向选择监控: 成交率 ${fillRate}% | 成交均置信 ${avgFilled}% | 挂单监控中均置信 ${avgUnfilled}%`);

            // 累计统计（持久化在内存中，随进程运行积累）
            if (!this._adverseSelectionStats) {
                this._adverseSelectionStats = { totalFilled: 0, totalUnfilled: 0, sumConfFilled: 0, sumConfUnfilled: 0 };
            }
            this._adverseSelectionStats.totalFilled += filledConfs.length;
            this._adverseSelectionStats.totalUnfilled += unfilledConfs.length;
            this._adverseSelectionStats.sumConfFilled += filledConfs.reduce((a, b) => a + b, 0);
            this._adverseSelectionStats.sumConfUnfilled += unfilledConfs.reduce((a, b) => a + b, 0);
            const stats = this._adverseSelectionStats;
            const cumFillRate = (stats.totalFilled + stats.totalUnfilled) > 0
                ? ((stats.totalFilled / (stats.totalFilled + stats.totalUnfilled)) * 100).toFixed(1)
                : '0';
            const cumAvgFilled = stats.totalFilled > 0
                ? ((stats.sumConfFilled / stats.totalFilled) * 100).toFixed(1)
                : '-';
            const cumAvgUnfilled = stats.totalUnfilled > 0
                ? ((stats.sumConfUnfilled / stats.totalUnfilled) * 100).toFixed(1)
                : '-';
            console.log(`  📊 累计统计: 成交率 ${cumFillRate}% (${stats.totalFilled}/${stats.totalFilled + stats.totalUnfilled}) | 成交均置信 ${cumAvgFilled}% | 未成交均置信 ${cumAvgUnfilled}%`);
        }
        console.log('─'.repeat(60) + '\n');

        await this.reconcileLiveStateAfterOrder('limit_order');

        return results;
    }

    // 阶梯限价单方法 (Ladder Limit Order — 动态模型用)
    // ════════════════════════════════════════════════════════════

    /**
     * 9 档阶梯限价单: 将总金额均分 9 份，在 $0.45~$0.53 各下一笔限价单。
     * 用于动态模型（v5_dyn），同时覆盖所有买价区间。
     *
     * 各价位独立检查 orderbook depth，独立计算成交。
     */
    async executeLadderOrder(opts: {
        targetPeriodEndTs: number;
        predictions: PredictionResult[];
        ladderPrices: number[];
        sizingReferencePrice?: number;
        minTotalAmountUsd?: number;
    }): Promise<ExecutionResult[]> {
        const { targetPeriodEndTs, predictions, ladderPrices, sizingReferencePrice, minTotalAmountUsd } = opts;
        const results: ExecutionResult[] = [];
        await this.prepareLiveCooldownStateBeforeOrder();
        const hasDirectionalLadderConfig =
            Object.keys(((this.env as any).LOWPRICE_SELECTED_BUY_PRICE_RANGE_BY_SYMBOL_DIRECTION || {}) as Record<string, unknown>).length > 0
            || Object.keys(((this.env as any).LOWPRICE_RUNG_COUNT_BY_SYMBOL_DIRECTION || {}) as Record<string, unknown>).length > 0;

        // 防御：没有任何统一或方向梯子时直接返回
        if ((!ladderPrices || ladderPrices.length === 0) && !hasDirectionalLadderConfig) {
            console.log('\n⚠️  阶梯限价单: ladderPrices 为空，跳过');
            return results;
        }

        console.log('\n' + '═'.repeat(60));
        console.log(`  🪜 阶梯限价单 (${hasDirectionalLadderConfig ? '方向梯子' : `${ladderPrices.length} 档`}) - ${new Date().toLocaleString('zh-CN')}`);
        const rangeBeijing = this._fmtPeriod(targetPeriodEndTs);
        console.log(`  📌 目标周期: ${rangeBeijing}`);
        if (ladderPrices && ladderPrices.length > 0) {
            console.log(`  💰 回退价格档: ${ladderPrices.map(p => '$' + p.toFixed(2)).join(', ')}`);
        } else {
            console.log(`  💰 价格档: 将按方向梯子实时展开`);
        }
        console.log('═'.repeat(60));

        printPredictions(predictions);

        const symbolsToFetch = [...new Set(predictions.map((p) => p.symbol.replace(/\/.*$/, '')))];
        const markets = await findAllMarkets(this.env.TIMEFRAME, targetPeriodEndTs, symbolsToFetch);
        for (const result of markets) {
            this.state.marketCache.set(result.symbol, result);
        }
        /** 本轮已对冷却做过自减的 key（按配置可为 symbol 或 symbol+direction） */
        const cooldownDecrementedThisRound = new Set<string>();

        for (const prediction of predictions) {
            const symbol = prediction.symbol.replace('/USDT', '');
            const marketResult = this.state.marketCache.get(symbol);
            const direction = prediction.direction!;

            if (!marketResult || !marketResult.market) {
                console.log(`  ❌ ${symbol}: 市场数据未缓存`);
                results.push({ symbol, success: false, error: '市场数据未缓存' });
                continue;
            }

            // ─── 风控检查（回撤熔断 + 冷却），与优化器/回测对齐 ───
            const riskCheck = this.checkRiskControls();
            if (riskCheck.shouldPause) {
                console.log(`\n🛡️  跳过 ${symbol}: 风险控制: ${riskCheck.reason || '交易已暂停'}`);
                results.push({ symbol, success: false, error: `风险控制: ${riskCheck.reason}` });
                continue;
            }
            // 按币种绝对止损当前已全局禁用，其余运行时 guards 保持原样。
            if (this.isSymbolAbsoluteStopped(symbol)) {
                const floor = this.getPerCoinAbsoluteFloor();
                console.log(`\n🛑 跳过 ${symbol}: 绝对止损 (该币种资金 $${(this.getCoinCapital(symbol) ?? 0).toFixed(2)} < $${floor.toFixed(2)})`);
                results.push({ symbol, success: false, error: `绝对止损: ${symbol} 资金低于该币种初始30%` });
                continue;
            }

            // ─── Cooldown 冷却检查（默认按币种+方向独立）───
            {
                const cooldownKey = this.buildCooldownKey(symbol, direction);
                const cooldownLabel = this.formatCooldownLabel(symbol, direction);
            const pendingResolutionGuard = this.shouldBlockForPendingCooldownResolution(symbol);
            if (pendingResolutionGuard.blocked) {
                console.log(`\n🧊 跳过 ${symbol}: 冷却待确认: ${pendingResolutionGuard.label} 已有 ${pendingResolutionGuard.consecutiveBeforeUnknown} 连输，上一根 ${pendingResolutionGuard.unresolvedMarketSlug || ''} 已结束但结果尚未入账`);
                results.push({ symbol, success: false, error: `冷却待确认 (${pendingResolutionGuard.label})` });
                this.appendCooldownBlockedOpportunity({
                        prediction,
                        marketResult,
                        symbol,
                        direction: direction!,
                        targetPeriodEndTs,
                        remainingBarsBefore: 0,
                        remainingBarsAfter: 0,
                        decrementedThisRound: false,
                        reasonCode: pendingResolutionGuard.reasonCode,
                });
                continue;
            }
            const persistedWindowGuard = this.getPersistedCooldownWindowBlock(symbol, direction, targetPeriodEndTs);
            if (persistedWindowGuard.blocked) {
                this.setCooldownRemaining(symbol, direction, persistedWindowGuard.remainingBarsAfter);
                console.log(`\n🧊 跳过 ${symbol}: ${persistedWindowGuard.error || `冷却中: ${persistedWindowGuard.label}`}`);
                results.push({ symbol, success: false, error: persistedWindowGuard.error || `冷却中 (${persistedWindowGuard.label})` });
                this.appendCooldownBlockedOpportunity({
                    prediction,
                    marketResult,
                    symbol,
                    direction: direction!,
                    targetPeriodEndTs,
                    remainingBarsBefore: persistedWindowGuard.remainingBarsBefore,
                    remainingBarsAfter: persistedWindowGuard.remainingBarsAfter,
                    decrementedThisRound: true,
                    reasonCode: 'cooldown_bars_blocked',
                });
                continue;
            }
            const cd = this.getCooldownRemaining(symbol, direction);
            if (cd > 0) {
                if (cooldownDecrementedThisRound.has(cooldownKey)) {
                    console.log(`\n🧊 跳过 ${symbol}: 冷却中: ${cooldownLabel} 剩余 ${cd} 个周期 (本轮已自减过)`);
                    results.push({ symbol, success: false, error: `冷却中 (${cooldownLabel}剩${cd}个周期)` });
                        this.appendCooldownBlockedOpportunity({
                            prediction,
                            marketResult,
                            symbol,
                            direction: direction!,
                            targetPeriodEndTs,
                            remainingBarsBefore: cd,
                            remainingBarsAfter: cd,
                            decrementedThisRound: true,
                            reasonCode: 'cooldown_bars_blocked',
                        });
                    } else {
                        const nextCd = Math.max(0, Number(cd) - 1);
                        this.setCooldownRemaining(symbol, direction, nextCd);
                        cooldownDecrementedThisRound.add(cooldownKey);
                        console.log(`\n🧊 跳过 ${symbol}: 冷却中: ${cooldownLabel} 剩余 ${nextCd} 个周期 (cooldown_bars=${this.env.COOLDOWN_BARS})`);
                        results.push({ symbol, success: false, error: `冷却中 (${cooldownLabel}剩${nextCd}个周期)` });
                        this.appendCooldownBlockedOpportunity({
                            prediction,
                            marketResult,
                            symbol,
                            direction: direction!,
                            targetPeriodEndTs,
                            remainingBarsBefore: cd,
                            remainingBarsAfter: nextCd,
                            decrementedThisRound: false,
                            reasonCode: 'cooldown_bars_blocked',
                        });
                    }
                    continue;
                }
            }

            const directionalVirtualLadder = this.buildDirectionalVirtualLadderPrices(symbol, direction);
            const effectiveLadderPrices = directionalVirtualLadder.length > 0 ? directionalVirtualLadder : ladderPrices;
            if (!effectiveLadderPrices || effectiveLadderPrices.length === 0) {
                const reason = '方向梯子未物化，且无可回退统一梯子';
                console.log(`  ⏭️ ${symbol}: ${reason}`);
                results.push({ symbol, success: false, error: reason });
                continue;
            }
            const nRungs = effectiveLadderPrices.length;
            const minLadderPrice = Math.min(...effectiveLadderPrices);
            const maxLadderPrice = Math.max(...effectiveLadderPrices);
            const midPrice = effectiveLadderPrices[Math.floor(effectiveLadderPrices.length / 2)];
            const sourceSizingPrice = Number.isFinite(sizingReferencePrice as number) ? Number(sizingReferencePrice) : midPrice;
            const requestedNotional = this.resolveRequestedNotionalUsd(prediction.confidence, sourceSizingPrice, symbol, direction, prediction);
            const sourceBet = requestedNotional.requestedNotionalUsd;
            const minDynamicTotal = Number.isFinite(minTotalAmountUsd as number) ? Math.max(0, Number(minTotalAmountUsd)) : 0;
            const betTargetCapUsd = this.getDirectionalBetTargetCapUsd(prediction.confidence, sourceSizingPrice, symbol, direction);
            const targetNotionalUsd = roundUsd(Math.max(requestedNotional.targetNotionalUsd, minDynamicTotal));
            let targetTotalBet = Math.max(sourceBet, minDynamicTotal);
            if (betTargetCapUsd > 0 && targetTotalBet > betTargetCapUsd) {
                targetTotalBet = betTargetCapUsd;
            }
            const availableCapital = Math.max(0, this.getCoinCapital(symbol) || 0);
            const riskUpperLimit = this.getSizingMode() === 'fixed_usd'
                ? roundUsd(availableCapital)
                : this.getCapitalRiskUpperLimit(availableCapital);
            const effectiveTotalBet = roundUsd(Math.min(targetTotalBet, availableCapital, riskUpperLimit));
            const requestLayerCompressionUsd = roundUsd(Math.max(0, targetNotionalUsd - effectiveTotalBet));
            if (effectiveTotalBet < Math.max(MIN_ORDER_SIZE_USD, minDynamicTotal || 0)) {
                console.log(`  ⏭️ ${symbol}: 动态总金额 $${effectiveTotalBet.toFixed(2)} < 最小执行阈值 $${Math.max(MIN_ORDER_SIZE_USD, minDynamicTotal || 0).toFixed(2)}`);
                results.push({ symbol, success: false, error: '金额不足' });
                continue;
            }

            const token = getTokenForDirection(marketResult.market, direction!);
            if (!token) {
                console.log(`  ❌ ${symbol}: 无法获取 ${direction} token`);
                results.push({ symbol, success: false, error: 'token not found' });
                continue;
            }

            const totalCents = Math.round(effectiveTotalBet * 100);
            const baseCents = Math.floor(totalCents / nRungs);
            const remainderCents = totalCents % nRungs;
            const rungBets = effectiveLadderPrices.map((_, idx) => roundUsd((baseCents + (idx < remainderCents ? 1 : 0)) / 100));
            const dynamicConstraint = await this.getExecutionConstraint(token.tokenId, marketResult.market.slug);
            const executionCompressionMode = this.getExecutionCompressionMode();
            const dynamicPlan = this.buildDynamicOrderExecutionPlan({
                rungPrices: effectiveLadderPrices,
                rungBets,
                maxBudgetUsd: effectiveTotalBet,
                constraint: dynamicConstraint,
                executionCompressionMode,
            });
            const selectedRungs = dynamicPlan.selectedRungs;
            const postedRungs = this.collapseSelectedRungsByPrice(selectedRungs);
            const collapsedSamePriceRungs = Math.max(0, selectedRungs.length - postedRungs.length);
            const materializedLimitPriceLadder = postedRungs.map((rung) => this.roundTokenPriceTick(rung.rungPrice));
            const gt5AutoReduceTriggered = dynamicPlan.keptDynamicRungs < dynamicPlan.attemptedRungs;
            const runtimeRiskScalingState = this.getRuntimeRiskScalingMode() === 'cap_to_directional_conservative_with_hysteresis'
                ? (this.isDirectionalConservativeActive(symbol) ? 'conservative' : 'normal')
                : (this.getRuntimeRiskScalingMode() || 'off');
            const directionalSizingSource = this.getDirectionalSizingSource(symbol, direction);

            this.appendExecutionV2Event({
                timestamp: new Date().toISOString(),
                symbol,
                direction,
                marketSlug: marketResult.market.slug,
                conditionId: marketResult.market.conditionId,
                mode: this.env.TRADING_MODE,
                eventType: requestedNotional.requestCompressionMode === 'off' ? 'compression_bypassed' : 'request_notional_finalized',
                shadowOnly: false,
                targetNotionalUsd,
                requestedAmountUsd: effectiveTotalBet,
                postedNotionalUsd: dynamicPlan.totalSelectedNotionalUsd,
                filledNotionalUsd: 0,
                sizingMode: requestedNotional.sizingMode,
                bankrollPct: requestedNotional.bankrollPct,
                confidenceBandScale: requestedNotional.confidenceBandScale,
                effectiveSizingPct: requestedNotional.effectiveSizingPct,
                currentCapital: requestedNotional.currentCapital,
                initialPrice: sourceSizingPrice,
                currentPrice: midPrice,
                currentMaxPrice: maxLadderPrice,
                reason: `requestCompressionMode=${requestedNotional.requestCompressionMode};gateAmountUsd=${requestedNotional.gateAmountUsd.toFixed(2)};virtualRungs=${nRungs};postedRungs=${postedRungs.length};collapsedSamePriceRungs=${collapsedSamePriceRungs}`,
            });
            if (dynamicPlan.keptDynamicRungs > 0) {
                this.appendExecutionV2Event({
                    timestamp: new Date().toISOString(),
                    symbol,
                    direction,
                    marketSlug: marketResult.market.slug,
                    conditionId: marketResult.market.conditionId,
                    mode: this.env.TRADING_MODE,
                    eventType: 'posted_notional_reallocated',
                    shadowOnly: false,
                    targetNotionalUsd,
                    requestedAmountUsd: effectiveTotalBet,
                    postedNotionalUsd: dynamicPlan.totalSelectedNotionalUsd,
                    filledNotionalUsd: 0,
                    sizingMode: requestedNotional.sizingMode,
                    bankrollPct: requestedNotional.bankrollPct,
                    confidenceBandScale: requestedNotional.confidenceBandScale,
                    effectiveSizingPct: requestedNotional.effectiveSizingPct,
                    currentCapital: requestedNotional.currentCapital,
                    initialPrice: minLadderPrice,
                    currentPrice: postedRungs.length > 0 ? postedRungs[postedRungs.length - 1].rungPrice : minLadderPrice,
                    currentMaxPrice: maxLadderPrice,
                    reason: `executionCompressionMode=${executionCompressionMode};kept=${dynamicPlan.keptDynamicRungs}/${dynamicPlan.attemptedRungs};posted=${postedRungs.length};collapsedSamePriceRungs=${collapsedSamePriceRungs};executionCompressionUsd=${dynamicPlan.executionCompressionUsd.toFixed(2)}`,
                });
            }

            // ─── 计算 Edge/Kelly（用中位价，与 executePrediction 一致，仅用于日志）───
            const _ladderOdds = midPrice > 0 && midPrice < 1 ? (1.0 - midPrice) / midPrice : 0;
            const _ladderConf = prediction.confidence ?? 0;
            const _ladderQ = 1.0 - _ladderConf;
            const _ladderEdge = _ladderConf * _ladderOdds - _ladderQ;
            const _ladderKelly = _ladderOdds > 0 ? Math.max(0, (_ladderConf * _ladderOdds - _ladderQ) / _ladderOdds) : 0;
            const _ladderBetPct = (effectiveTotalBet / (this.getCoinCapital(symbol) || 1) * 100).toFixed(1);
            const _ladderConfPct = (_ladderConf * 100).toFixed(1);
            const _ladderThreshPct = (this.env.PROB_THRESHOLD * 100).toFixed(0);
            const _ladderDirIcon = direction === 'UP' ? '📈 UP' : '📉 DOWN';
            const _ladderMode = this.env.TRADING_MODE === TradingMode.SIMULATION ? '模拟交易'
                : this.env.TRADING_MODE === TradingMode.LIVE ? '真实交易' : '回测';
            const _ladderSlug = marketResult.market.slug
                ? (marketResult.market.slug.length > 35 ? marketResult.market.slug.slice(0, 35) + '...' : marketResult.market.slug)
                : '';

            // 详情框（与旧模型 executePrediction 风格对齐）
            console.log(`\n┌${'─'.repeat(58)}┐`);
            console.log(`│ ${(symbol + ' ' + _ladderDirIcon + ' @$' + midPrice.toFixed(2) + '(中位) [' + _ladderSlug + ']').padEnd(57)}│`);
            console.log(`├${'─'.repeat(58)}┤`);
            console.log(`│ 北京时间:   ${nowBeijing().padEnd(45)}│`);
            console.log(`│ 置信度:     ${(_ladderConfPct + '% (阈值: ' + _ladderThreshPct + '%)').padEnd(45)}│`);
            console.log(`│ Edge/Kelly: ${('edge=' + (_ladderEdge * 100).toFixed(2) + '% kelly=' + (_ladderKelly * 100).toFixed(1) + '%').padEnd(45)}│`);
            console.log(`│ 总下注:     ${('$' + effectiveTotalBet.toFixed(2) + ' (占资金 ' + _ladderBetPct + '%, 源$' + sourceBet.toFixed(2) + ')').padEnd(45)}│`);
            console.log(`│ Sizing价:   ${('$' + sourceSizingPrice.toFixed(2) + ' (min_total=$' + minDynamicTotal.toFixed(2) + ')').padEnd(45)}│`);
            console.log(`│ 价格范围:   ${('$' + minLadderPrice.toFixed(2) + '~$' + maxLadderPrice.toFixed(2) + ` (虚拟${nRungs}档/真实${postedRungs.length}档)`).padEnd(45)}│`);
            console.log(`│ 执行约束:   ${('minShares=' + dynamicConstraint.minOrderSizeShares.toFixed(2) + ' uplift=' + dynamicPlan.upliftedDynamicRungs + ' kept=' + dynamicPlan.keptDynamicRungs + '/' + dynamicPlan.attemptedRungs + ' 合并=' + collapsedSamePriceRungs).padEnd(45)}│`);
            console.log(`│ 交易模式:   ${_ladderMode.padEnd(45)}│`);
            console.log(`└${'─'.repeat(58)}┘`);

            if (dynamicPlan.keptDynamicRungs === 0) {
                const reason = dynamicPlan.failureReason || '动态阶梯单 uplift 后预算不足，未保留任何可执行档';
                console.log(`  ⏭️ ${symbol}: ${reason}`);
                this.addExecutionSkippedEntry({
                    symbol,
                    direction,
                    confidence: prediction.confidence,
                    reason,
                    executionDecision: dynamicPlan.decision,
                    requestedAmountUsd: effectiveTotalBet,
                    targetNotionalUsd,
                    requestedNotionalUsd: effectiveTotalBet,
                    postedNotionalUsd: dynamicPlan.totalSelectedNotionalUsd,
                    filledNotionalUsd: 0,
                    requestCompressionMode: requestedNotional.requestCompressionMode,
                    executionCompressionMode,
                    requestLayerCompressionUsd,
                    executionLayerCompressionUsd: dynamicPlan.executionCompressionUsd,
                    executionLayerCompressionReason: dynamicPlan.executionCompressionReason || dynamicPlan.failureReason,
                    sizingMode: requestedNotional.sizingMode,
                    bankrollPct: requestedNotional.bankrollPct,
                    confidenceBandScale: requestedNotional.confidenceBandScale,
                    effectiveSizingPct: requestedNotional.effectiveSizingPct,
                    currentCapital: requestedNotional.currentCapital,
                    requestedShares: 0,
                    market: {
                        title: marketResult.market.title,
                        conditionId: marketResult.market.conditionId,
                        slug: marketResult.market.slug,
                    },
                    token,
                    exchangeMinOrderSizeShares: dynamicConstraint.minOrderSizeShares,
                    attemptedRungs: dynamicPlan.attemptedRungs,
                    compressedRungs: dynamicPlan.droppedDynamicRungs,
                    postedRungs: 0,
                    filledRungs: 0,
                    exchangeRejectedRungs: 0,
                    upliftedDynamicRungs: dynamicPlan.upliftedDynamicRungs,
                    keptDynamicRungs: dynamicPlan.keptDynamicRungs,
                    droppedDynamicRungs: dynamicPlan.droppedDynamicRungs,
                    upliftedNotionalUsd: dynamicPlan.totalUpliftedNotionalUsd,
                    virtualRungCount: nRungs,
                    materializedLimitPriceLadder,
                    collapsedSamePriceRungs,
                    gt5AutoReduceTriggered,
                    runtimeRiskScalingState,
                    directionalSizingSource,
                });
                results.push({ symbol, success: false, error: reason });
                continue;
            }

            let totalFilled = 0;
            let totalFrozen = 0;
            let totalPendingMonitorAmount = 0;
            let rungResults: string[] = [];

            if (this.env.TRADING_MODE === TradingMode.LIVE && !this.computeLiveGtdExpiration(targetPeriodEndTs)) {
                const reason = this.buildLiveGtdExpiredReason(targetPeriodEndTs);
                console.log(`  ⏭️ ${symbol}: ${reason}`);
                this.addExecutionSkippedEntry({
                    symbol,
                    direction,
                    confidence: prediction.confidence,
                    reason,
                    executionDecision: 'market_expired_before_submit',
                    requestedAmountUsd: effectiveTotalBet,
                    targetNotionalUsd,
                    requestedNotionalUsd: effectiveTotalBet,
                    postedNotionalUsd: 0,
                    filledNotionalUsd: 0,
                    requestCompressionMode: requestedNotional.requestCompressionMode,
                    executionCompressionMode,
                    requestLayerCompressionUsd,
                    executionLayerCompressionUsd: dynamicPlan.executionCompressionUsd,
                    executionLayerCompressionReason: reason,
                    requestedShares: 0,
                    market: {
                        title: marketResult.market.title,
                        conditionId: marketResult.market.conditionId,
                        slug: marketResult.market.slug,
                    },
                    token,
                    exchangeMinOrderSizeShares: dynamicConstraint.minOrderSizeShares,
                    attemptedRungs: dynamicPlan.attemptedRungs,
                    compressedRungs: dynamicPlan.droppedDynamicRungs,
                    postedRungs: 0,
                    filledRungs: 0,
                    exchangeRejectedRungs: 0,
                    upliftedDynamicRungs: dynamicPlan.upliftedDynamicRungs,
                    keptDynamicRungs: dynamicPlan.keptDynamicRungs,
                    droppedDynamicRungs: dynamicPlan.droppedDynamicRungs,
                    minExecutableNotionalUsd: dynamicPlan.selectedRungs[0]?.minExecutableNotionalUsd,
                    upliftedNotionalUsd: dynamicPlan.totalUpliftedNotionalUsd,
                    gt5PreservationStatus: dynamicPlan.selectedRungs[0]?.gt5PreservationStatus,
                    compressionReason: dynamicPlan.executionCompressionReason,
                    virtualRungCount: nRungs,
                    materializedLimitPriceLadder,
                    collapsedSamePriceRungs,
                    gt5AutoReduceTriggered,
                    runtimeRiskScalingState,
                    directionalSizingSource,
                    sizingMode: requestedNotional.sizingMode,
                    bankrollPct: requestedNotional.bankrollPct,
                    confidenceBandScale: requestedNotional.confidenceBandScale,
                    effectiveSizingPct: requestedNotional.effectiveSizingPct,
                    currentCapital: requestedNotional.currentCapital,
                });
                results.push({ symbol, success: false, error: reason });
                continue;
            }

            // 获取一次 orderbook（所有档共享）
            let ladderBookSnapshot: QueueAwareBookSnapshot = {
                bestAskPrice: 999,
                asks: [],
                bids: [],
                hadBidsData: false,
            };
            try {
                if (token.tokenId) {
                    const orderBook = await getOrderBook(token.tokenId);
                    ladderBookSnapshot = buildQueueAwareBookSnapshot(orderBook);
                }
            } catch { /* 获取失败 */ }
            if (
                Number.isFinite(ladderBookSnapshot.bestAskPrice) &&
                ladderBookSnapshot.bestAskPrice > 0 &&
                ladderBookSnapshot.bestAskPrice < this.env.MIN_TOKEN_PRICE
            ) {
                const reason = `市场已近乎定价该结果 (卖价 $${ladderBookSnapshot.bestAskPrice.toFixed(4)} < $${this.env.MIN_TOKEN_PRICE})，跳过动态阶梯单`;
                console.log(`  ⏭️ ${symbol}: ${reason}`);
                this.addExecutionSkippedEntry({
                    symbol,
                    direction,
                    confidence: prediction.confidence,
                    reason,
                    executionDecision: 'market_price_below_min_floor',
                    requestedAmountUsd: effectiveTotalBet,
                    targetNotionalUsd,
                    requestedNotionalUsd: effectiveTotalBet,
                    postedNotionalUsd: 0,
                    filledNotionalUsd: 0,
                    requestCompressionMode: requestedNotional.requestCompressionMode,
                    executionCompressionMode,
                    requestLayerCompressionUsd,
                    executionLayerCompressionUsd: dynamicPlan.executionCompressionUsd,
                    executionLayerCompressionReason: 'market_price_below_min_floor',
                    requestedShares: 0,
                    market: {
                        title: marketResult.market.title,
                        conditionId: marketResult.market.conditionId,
                        slug: marketResult.market.slug,
                    },
                    token,
                    exchangeMinOrderSizeShares: dynamicConstraint.minOrderSizeShares,
                    attemptedRungs: dynamicPlan.attemptedRungs,
                    compressedRungs: dynamicPlan.droppedDynamicRungs,
                    postedRungs: 0,
                    filledRungs: 0,
                    exchangeRejectedRungs: 0,
                    upliftedDynamicRungs: dynamicPlan.upliftedDynamicRungs,
                    keptDynamicRungs: dynamicPlan.keptDynamicRungs,
                    droppedDynamicRungs: dynamicPlan.droppedDynamicRungs,
                    upliftedNotionalUsd: dynamicPlan.totalUpliftedNotionalUsd,
                    virtualRungCount: nRungs,
                    materializedLimitPriceLadder,
                    collapsedSamePriceRungs,
                    gt5AutoReduceTriggered,
                    runtimeRiskScalingState,
                    directionalSizingSource,
                });
                results.push({ symbol, success: false, error: reason });
                continue;
            }

            for (const rungPlan of dynamicPlan.allRungs) {
                if (rungPlan.kept) {
                    continue;
                }
                const rungPrice = rungPlan.rungPrice;
                rungResults.push(`$${rungPrice.toFixed(2)}: 动态压缩裁剪(低价端优先, 最低可执行$${rungPlan.minExecutableNotionalUsd.toFixed(2)})`);
            }

            for (const rungPlan of postedRungs) {
                const rungPrice = rungPlan.rungPrice;
                const rungBet = rungPlan.adjustedBetUsd;
                const rungShares = rungPlan.adjustedShares;

                if (this.env.TRADING_MODE === TradingMode.SIMULATION) {
                    this.adjustCoinCapital(symbol, -rungBet);
                    totalFrozen += rungBet;

                    const rungFill = simulateQueueAwareBuyFillWithHaircut(ladderBookSnapshot, rungPrice, rungBet);
                    const fillable = rungFill.haircutAdjustedFillableUsd;
                    const rungAvgPrice = rungFill.avgFillPrice;
                    const bestAsk = rungFill.bestAskPrice;
                    const queueCompetitionUsd = rungFill.queueCompetitionUsd;
                    const effectiveFillableUsd = rungFill.effectiveFillableUsd;

                    let filled = 0;
                    if (bestAsk <= rungPrice && fillable >= rungBet) {
                        filled = rungBet;
                        rungResults.push(`$${rungPrice.toFixed(2)}: 全部成交 $${filled.toFixed(2)} (均价$${rungAvgPrice.toFixed(4)}, 排队竞争$${queueCompetitionUsd.toFixed(2)})`);
                    } else if (bestAsk <= rungPrice && fillable > 0) {
                        filled = Math.min(fillable, rungBet);
                        const pct = ((filled / rungBet) * 100).toFixed(0);
                        rungResults.push(`$${rungPrice.toFixed(2)}: 部分成交 ${pct}% ($${filled.toFixed(2)}, 均价$${rungAvgPrice.toFixed(4)}, 排队竞争$${queueCompetitionUsd.toFixed(2)})`);
                    } else {
                        rungResults.push(`$${rungPrice.toFixed(2)}: 挂单监控中`);
                    }

                    if (filled > 0) {
                        consumeAskBookLiquidity(ladderBookSnapshot.asks, rungPrice, filled);
                    }

                    const rungUnfilled = roundUsd(rungBet - filled);
                    if (rungUnfilled >= MIN_ORDER_SIZE_USD) {
                        const pendingId = `sim_ladder_${targetPeriodEndTs}_${symbol}_${rungPrice}_${Date.now()}`;
                        const createdAt = new Date();
                        const expiresAt = new Date((targetPeriodEndTs + this.periodSeconds) * 1000);
                        this.queuePendingSimOrder({
                            id: pendingId,
                            symbol,
                            direction,
                            frozenAmount: rungUnfilled,
                            limitPrice: rungPrice,
                            tokenId: token.tokenId,
                            tokenOutcome: token.outcome,
                            confidence: prediction.confidence,
                            marketTitle: marketResult.market.title,
                            conditionId: marketResult.market.conditionId,
                            marketSlug: marketResult.market.slug,
                            targetPeriodEndTs,
                            createdAt,
                            expiresAt,
                            bestAskAtEntry: bestAsk,
                            queueCompetitionUsdAtEntry: queueCompetitionUsd,
                            effectiveFillableUsdAtEntry: effectiveFillableUsd,
                            avgActualFillPriceAtEntry: rungAvgPrice,
                            checking: false,
                        });
                        totalPendingMonitorAmount += rungUnfilled;
                        this.startPendingSimMonitor();
                    } else if (rungUnfilled > 0) {
                        this.adjustCoinCapital(symbol, rungUnfilled);
                    }
                    totalFilled += filled;

                    if (filled > 0) {
                        this.logger.addEntry({
                            symbol,
                            direction,
                            confidence: prediction.confidence,
                            amount: filled,
                            mode: this.env.TRADING_MODE,
                            status: 'executed',
                            marketTitle: marketResult.market.title,
                            conditionId: marketResult.market.conditionId,
                            tokenId: token.tokenId,
                            marketSlug: marketResult.market.slug,
                            tokenOutcome: token.outcome,
                            tokenPrice: rungAvgPrice,
                            avgActualFillPrice: rungAvgPrice,
                            limitPriceConfigured: rungPrice,
                            bestAskAtEntry: bestAsk,
                            queueCompetitionUsdAtEntry: queueCompetitionUsd,
                            effectiveFillableUsdAtEntry: effectiveFillableUsd,
                            requestCompressionMode: requestedNotional.requestCompressionMode,
                            executionCompressionMode,
                            targetNotionalUsd,
                            requestedNotionalUsd: effectiveTotalBet,
                            postedNotionalUsd: dynamicPlan.totalSelectedNotionalUsd,
                            filledNotionalUsd: filled,
                            requestLayerCompressionUsd,
                            executionLayerCompressionUsd: dynamicPlan.executionCompressionUsd,
                            executionLayerCompressionReason: dynamicPlan.executionCompressionReason,
                            exchangeMinOrderSizeShares: dynamicConstraint.minOrderSizeShares,
                            attemptedRungs: dynamicPlan.attemptedRungs,
                            compressedRungs: dynamicPlan.droppedDynamicRungs,
                            postedRungs: postedRungs.length,
                            filledRungs: 1,
                            exchangeRejectedRungs: 0,
                            upliftedDynamicRungs: dynamicPlan.upliftedDynamicRungs,
                            keptDynamicRungs: dynamicPlan.keptDynamicRungs,
                            droppedDynamicRungs: dynamicPlan.droppedDynamicRungs,
                            minExecutableNotionalUsd: rungPlan.minExecutableNotionalUsd,
                            redistributedNotionalUsd: rungPlan.redistributedNotionalUsd,
                            upliftedNotionalUsd: dynamicPlan.totalUpliftedNotionalUsd,
                            executionDecision: dynamicPlan.decision,
                            requestedAmountUsd: rungBet,
                            requestedShares: rungShares,
                            originalRequestedShares: rungPlan.requestedShares,
                            gt5PreservationStatus: rungPlan.gt5PreservationStatus,
                            compressionReason: rungPlan.compressionReason,
                            virtualRungCount: nRungs,
                            materializedLimitPriceLadder,
                            collapsedSamePriceRungs,
                            gt5AutoReduceTriggered,
                            runtimeRiskScalingState,
                            directionalSizingSource,
                            sizingMode: requestedNotional.sizingMode,
                            bankrollPct: requestedNotional.bankrollPct,
                            confidenceBandScale: requestedNotional.confidenceBandScale,
                            effectiveSizingPct: requestedNotional.effectiveSizingPct,
                            currentCapital: requestedNotional.currentCapital,
                        });
                    }
                } else if (this.env.TRADING_MODE === TradingMode.LIVE && this.clobClient) {
                    const shareCheck = this.precheckSharesAgainstConstraint(rungShares, dynamicConstraint);
                    if (!shareCheck.ok) {
                        rungResults.push(`$${rungPrice.toFixed(2)}: 本地过滤(${shareCheck.reason})`);
                        continue;
                    }
                    for (let orderRetry = 0; orderRetry < RETRY_LIMIT; orderRetry++) {
                        try {
                            const preflight = await this.preflightLiveOrder(rungBet);
                            if (!preflight.ok) {
                                const reason = preflight.reason;
                                this.rememberLiveExchangeConstraints(reason, token.tokenId, marketResult.market.slug);
                                if (this.shouldRetryLiveSubmitIssue(preflight.status) && orderRetry < RETRY_LIMIT - 1) {
                                    rungResults.push(`$${rungPrice.toFixed(2)}: 预检失败重试(${reason})`);
                                    await new Promise(r => setTimeout(r, RETRY_INTERVAL_MS));
                                    continue;
                                }
                                rungResults.push(`$${rungPrice.toFixed(2)}: 预检失败(${reason})`);
                                this.addExecutionSkippedEntry({
                                    symbol,
                                    direction,
                                    confidence: prediction.confidence,
                                    reason,
                                    executionDecision: preflight.status === 'allowance_zero' || preflight.status === 'balance_insufficient' ? 'exchange_rejected' : 'fixed_order_exception',
                                    requestedAmountUsd: rungBet,
                                    requestedShares: rungShares,
                                    originalRequestedShares: rungPlan.requestedShares,
                                    market: {
                                        title: marketResult.market.title,
                                        conditionId: marketResult.market.conditionId,
                                        slug: marketResult.market.slug,
                                    },
                                    token,
                                    exchangeMinOrderSizeShares: dynamicConstraint.minOrderSizeShares,
                                    attemptedRungs: dynamicPlan.attemptedRungs,
                                    compressedRungs: dynamicPlan.droppedDynamicRungs,
                                    postedRungs: 0,
                                    filledRungs: 0,
                                    exchangeRejectedRungs: preflight.status === 'allowance_zero' || preflight.status === 'balance_insufficient' ? 1 : 0,
                                    upliftedDynamicRungs: dynamicPlan.upliftedDynamicRungs,
                                    keptDynamicRungs: dynamicPlan.keptDynamicRungs,
                                    droppedDynamicRungs: dynamicPlan.droppedDynamicRungs,
                                    minExecutableNotionalUsd: rungPlan.minExecutableNotionalUsd,
                                    redistributedNotionalUsd: rungPlan.redistributedNotionalUsd,
                                    upliftedNotionalUsd: dynamicPlan.totalUpliftedNotionalUsd,
                                    gt5PreservationStatus: rungPlan.gt5PreservationStatus,
                                    compressionReason: rungPlan.compressionReason,
                                    virtualRungCount: nRungs,
                                    materializedLimitPriceLadder,
                                    collapsedSamePriceRungs,
                                    gt5AutoReduceTriggered,
                                    runtimeRiskScalingState,
                                    directionalSizingSource,
                                });
                                break;
                            }
                            const gtdExpiration = this.computeLiveGtdExpiration(targetPeriodEndTs);
                            if (!gtdExpiration) {
                                const reason = this.buildLiveGtdExpiredReason(targetPeriodEndTs);
                                rungResults.push(`$${rungPrice.toFixed(2)}: ${reason}`);
                                continue;
                            }
                            const userOrder = {
                                tokenID: token.tokenId,
                                price: rungPrice,
                                size: rungShares,
                                side: Side.BUY,
                                expiration: gtdExpiration,
                            };
                            await this.ensureClobHeartbeatFresh();
                            const signedOrder = await this.clobClient.createOrder(userOrder, { tickSize: '0.01' });
                            const resp = await this.clobClient.postOrder(signedOrder, OrderType.GTD);
                            const immediatePostError = this.getPostOrderImmediateError(resp);
                            if (immediatePostError) {
                                this.rememberLiveExchangeConstraints(immediatePostError, token.tokenId, marketResult.market.slug);
                                rungResults.push(`$${rungPrice.toFixed(2)}: 拒单(${immediatePostError})`);
                                this.addExecutionSkippedEntry({
                                    symbol,
                                    direction,
                                    confidence: prediction.confidence,
                                    reason: immediatePostError,
                                    executionDecision: 'exchange_rejected',
                                    requestedAmountUsd: rungBet,
                                    targetNotionalUsd,
                                    requestedNotionalUsd: effectiveTotalBet,
                                    postedNotionalUsd: 0,
                                    filledNotionalUsd: 0,
                                    requestCompressionMode: requestedNotional.requestCompressionMode,
                                    executionCompressionMode,
                                    requestLayerCompressionUsd,
                                    executionLayerCompressionUsd: dynamicPlan.executionCompressionUsd,
                                    executionLayerCompressionReason: immediatePostError,
                                    requestedShares: rungShares,
                                    originalRequestedShares: rungPlan.requestedShares,
                                    market: {
                                        title: marketResult.market.title,
                                        conditionId: marketResult.market.conditionId,
                                        slug: marketResult.market.slug,
                                    },
                                    token,
                                    exchangeMinOrderSizeShares: dynamicConstraint.minOrderSizeShares,
                                    attemptedRungs: dynamicPlan.attemptedRungs,
                                    compressedRungs: dynamicPlan.droppedDynamicRungs,
                                    postedRungs: 0,
                                    filledRungs: 0,
                                    exchangeRejectedRungs: 1,
                                    upliftedDynamicRungs: dynamicPlan.upliftedDynamicRungs,
                                    keptDynamicRungs: dynamicPlan.keptDynamicRungs,
                                    droppedDynamicRungs: dynamicPlan.droppedDynamicRungs,
                                    minExecutableNotionalUsd: rungPlan.minExecutableNotionalUsd,
                                    redistributedNotionalUsd: rungPlan.redistributedNotionalUsd,
                                    upliftedNotionalUsd: dynamicPlan.totalUpliftedNotionalUsd,
                                    gt5PreservationStatus: rungPlan.gt5PreservationStatus,
                                    compressionReason: rungPlan.compressionReason,
                                    virtualRungCount: nRungs,
                                    materializedLimitPriceLadder,
                                    collapsedSamePriceRungs,
                                    gt5AutoReduceTriggered,
                                    runtimeRiskScalingState,
                                    directionalSizingSource,
                                });
                                break;
                            }
                            const postedOrderId = this.extractPostedOrderId(resp);
                            if (postedOrderId) {
                                const orderID = postedOrderId;
                                const officialAcceptance = await this.verifyOfficialOrderAcceptance(orderID, token.tokenId);
                                if (!officialAcceptance.accepted) {
                                    const reason = `官网未确认自动订单收单: orderID=${orderID} reason=${officialAcceptance.reason || 'unknown'} post=${this.summarizePostOrderResponse(resp)}`;
                                    this.rememberLiveExchangeConstraints(reason, token.tokenId, marketResult.market.slug);
                                    rungResults.push(`$${rungPrice.toFixed(2)}: 拒单(${reason})`);
                                    this.addExecutionSkippedEntry({
                                        symbol,
                                        direction,
                                        confidence: prediction.confidence,
                                        reason,
                                        executionDecision: 'exchange_rejected',
                                        requestedAmountUsd: rungBet,
                                        targetNotionalUsd,
                                        requestedNotionalUsd: effectiveTotalBet,
                                        postedNotionalUsd: 0,
                                        filledNotionalUsd: 0,
                                        requestCompressionMode: requestedNotional.requestCompressionMode,
                                        executionCompressionMode,
                                        requestLayerCompressionUsd,
                                        executionLayerCompressionUsd: dynamicPlan.executionCompressionUsd,
                                        executionLayerCompressionReason: reason,
                                        requestedShares: rungShares,
                                        originalRequestedShares: rungPlan.requestedShares,
                                        market: {
                                            title: marketResult.market.title,
                                            conditionId: marketResult.market.conditionId,
                                            slug: marketResult.market.slug,
                                        },
                                        token,
                                        exchangeMinOrderSizeShares: dynamicConstraint.minOrderSizeShares,
                                        attemptedRungs: dynamicPlan.attemptedRungs,
                                        compressedRungs: dynamicPlan.droppedDynamicRungs,
                                        postedRungs: 0,
                                        filledRungs: 0,
                                        exchangeRejectedRungs: 1,
                                        upliftedDynamicRungs: dynamicPlan.upliftedDynamicRungs,
                                        keptDynamicRungs: dynamicPlan.keptDynamicRungs,
                                        droppedDynamicRungs: dynamicPlan.droppedDynamicRungs,
                                        minExecutableNotionalUsd: rungPlan.minExecutableNotionalUsd,
                                        redistributedNotionalUsd: rungPlan.redistributedNotionalUsd,
                                        upliftedNotionalUsd: dynamicPlan.totalUpliftedNotionalUsd,
                                        gt5PreservationStatus: rungPlan.gt5PreservationStatus,
                                        compressionReason: rungPlan.compressionReason,
                                        virtualRungCount: nRungs,
                                        materializedLimitPriceLadder,
                                        collapsedSamePriceRungs,
                                        gt5AutoReduceTriggered,
                                        runtimeRiskScalingState,
                                        directionalSizingSource,
                                    });
                                    break;
                                }
                                const immediateFill = this.buildOfficialImmediateFill(
                                    officialAcceptance.raw,
                                    rungPrice,
                                    rungShares,
                                    rungBet,
                                );
                                const createdAtIso = new Date().toISOString();
                                if (immediateFill) {
                                    rungResults.push(`$${rungPrice.toFixed(2)}: 官网已成交 $${immediateFill.amountUsd.toFixed(2)} (${immediateFill.source})`);
                                    totalFilled += immediateFill.amountUsd;
                                } else {
                                    rungResults.push(`$${rungPrice.toFixed(2)}: 已挂出待成交`);
                                    totalPendingMonitorAmount += rungBet;
                                }
                                const liveEntry = this.logger.addEntry({
                                    symbol,
                                    direction,
                                    confidence: prediction.confidence,
                                    amount: immediateFill ? immediateFill.amountUsd : 0,
                                    mode: this.env.TRADING_MODE,
                                    status: immediateFill ? 'executed' : 'pending',
                                    marketTitle: marketResult.market.title,
                                    conditionId: marketResult.market.conditionId,
                                    tokenId: token.tokenId,
                                    marketSlug: marketResult.market.slug,
                                    tokenOutcome: token.outcome,
                                    tokenPrice: rungPrice,
                                    limitPriceConfigured: rungPrice,
                                    orderId: orderID,
                                    txHash: orderID,
                                    requestedAmountUsd: rungBet,
                                    targetNotionalUsd,
                                    requestedNotionalUsd: effectiveTotalBet,
                                    postedNotionalUsd: dynamicPlan.totalSelectedNotionalUsd,
                                    filledNotionalUsd: immediateFill ? immediateFill.amountUsd : 0,
                                    requestCompressionMode: requestedNotional.requestCompressionMode,
                                    executionCompressionMode,
                                    requestLayerCompressionUsd,
                                    executionLayerCompressionUsd: dynamicPlan.executionCompressionUsd,
                                    executionLayerCompressionReason: dynamicPlan.executionCompressionReason,
                                    pendingAmountUsd: immediateFill?.status === 'partially_filled'
                                        ? Math.max(0, roundUsd(rungBet - immediateFill.amountUsd))
                                        : (immediateFill ? 0 : rungBet),
                                    requestedShares: rungShares,
                                    originalRequestedShares: rungPlan.requestedShares,
                                    matchedShares: immediateFill ? immediateFill.tokens : 0,
                                    avgActualFillPrice: immediateFill?.price,
                                    fillSource: immediateFill?.source,
                                    liveOrderStatus: immediateFill ? immediateFill.status : 'posted_pending',
                                    familyRuleVersion: this.getFamilyRuleVersion() || undefined,
                                    lowpriceFamilyId: this.getLowpriceFamilyId() || undefined,
                                    stalePolicyDecision: this.getStaleOrderPolicy() === 'off' ? 'off' : 'kept',
                                    exchangeMinOrderSizeShares: dynamicConstraint.minOrderSizeShares,
                                    attemptedRungs: dynamicPlan.attemptedRungs,
                                    compressedRungs: dynamicPlan.droppedDynamicRungs,
                                    postedRungs: postedRungs.length,
                                    filledRungs: immediateFill ? 1 : 0,
                                    exchangeRejectedRungs: 0,
                                    upliftedDynamicRungs: dynamicPlan.upliftedDynamicRungs,
                                    keptDynamicRungs: dynamicPlan.keptDynamicRungs,
                                    droppedDynamicRungs: dynamicPlan.droppedDynamicRungs,
                                    minExecutableNotionalUsd: rungPlan.minExecutableNotionalUsd,
                                    redistributedNotionalUsd: rungPlan.redistributedNotionalUsd,
                                    upliftedNotionalUsd: dynamicPlan.totalUpliftedNotionalUsd,
                                    executionDecision: dynamicPlan.decision,
                                    gt5PreservationStatus: rungPlan.gt5PreservationStatus,
                                    compressionReason: rungPlan.compressionReason,
                                    virtualRungCount: nRungs,
                                    materializedLimitPriceLadder,
                                    collapsedSamePriceRungs,
                                    gt5AutoReduceTriggered,
                                    runtimeRiskScalingState,
                                    directionalSizingSource,
                                    sizingMode: requestedNotional.sizingMode,
                                    bankrollPct: requestedNotional.bankrollPct,
                                    confidenceBandScale: requestedNotional.confidenceBandScale,
                                    effectiveSizingPct: requestedNotional.effectiveSizingPct,
                                    currentCapital: requestedNotional.currentCapital,
                                });
                                this.appendPendingOrderLedgerEvent({
                                    event: 'created',
                                    orderId: orderID,
                                    symbol,
                                    direction,
                                    amountUsd: rungBet,
                                    remainingFrozenAmount: rungBet,
                                    limitPrice: rungPrice,
                                    tokenId: token.tokenId,
                                    tokenOutcome: token.outcome,
                                    confidence: prediction.confidence,
                                    marketTitle: marketResult.market.title,
                                    conditionId: marketResult.market.conditionId,
                                    marketSlug: marketResult.market.slug,
                                    targetPeriodEndTs,
                                    createdAt: createdAtIso,
                                    expiresAt: new Date(gtdExpiration * 1000).toISOString(),
                                    bestAsk: token.price,
                                    officialAcceptanceStatus: officialAcceptance.status,
                                    officialAcceptanceSource: officialAcceptance.source,
                                    officialAcceptanceAt: createdAtIso,
                                });
                                if (immediateFill) {
                                    this.appendPendingOrderLedgerEvent({
                                        event: immediateFill.status === 'partially_filled' ? 'partial_fill' : 'filled',
                                        orderId: orderID,
                                        symbol,
                                        direction,
                                        amountUsd: immediateFill.amountUsd,
                                        remainingFrozenAmount: immediateFill.status === 'partially_filled'
                                            ? Math.max(0, roundUsd(rungBet - immediateFill.amountUsd))
                                            : 0,
                                        limitPrice: rungPrice,
                                        tokenId: token.tokenId,
                                        tokenOutcome: token.outcome,
                                        confidence: prediction.confidence,
                                        marketTitle: marketResult.market.title,
                                        conditionId: marketResult.market.conditionId,
                                        marketSlug: marketResult.market.slug,
                                        targetPeriodEndTs,
                                        createdAt: createdAtIso,
                                        expiresAt: new Date(gtdExpiration * 1000).toISOString(),
                                        bestAsk: token.price,
                                        avgFillPrice: immediateFill.price,
                                        officialAcceptanceStatus: officialAcceptance.status,
                                        officialAcceptanceSource: officialAcceptance.source,
                                        officialAcceptanceAt: createdAtIso,
                                    });
                                    this.appendExecutionV2SettlementEventFromLog(
                                        liveEntry.id,
                                        immediateFill.status === 'partially_filled' ? 'chase_reprice_filled' : 'fully_filled',
                                        immediateFill.status === 'partially_filled' ? 'official_immediate_partial_fill' : 'official_immediate_full_fill',
                                    );
                                }
                            } else {
                                const classified = classifyLiveSubmitIssue(resp);
                                const reason = classified.reason;
                                rungResults.push(`$${rungPrice.toFixed(2)}: 失败(${reason})`);
                                this.addExecutionSkippedEntry({
                                    symbol,
                                    direction,
                                    confidence: prediction.confidence,
                                    reason,
                                    executionDecision: classified.status === 'allowance_zero' || classified.status === 'balance_insufficient' ? 'exchange_rejected' : 'fixed_order_exception',
                                    requestedAmountUsd: rungBet,
                                    targetNotionalUsd,
                                    requestedNotionalUsd: effectiveTotalBet,
                                    postedNotionalUsd: 0,
                                    filledNotionalUsd: 0,
                                    requestCompressionMode: requestedNotional.requestCompressionMode,
                                    executionCompressionMode,
                                    requestLayerCompressionUsd,
                                    executionLayerCompressionUsd: dynamicPlan.executionCompressionUsd,
                                    executionLayerCompressionReason: reason,
                                    requestedShares: rungShares,
                                    originalRequestedShares: rungPlan.requestedShares,
                                    market: {
                                        title: marketResult.market.title,
                                        conditionId: marketResult.market.conditionId,
                                        slug: marketResult.market.slug,
                                    },
                                    token,
                                    exchangeMinOrderSizeShares: dynamicConstraint.minOrderSizeShares,
                                    attemptedRungs: dynamicPlan.attemptedRungs,
                                    compressedRungs: dynamicPlan.droppedDynamicRungs,
                                    postedRungs: 0,
                                    filledRungs: 0,
                                    exchangeRejectedRungs: classified.status === 'allowance_zero' || classified.status === 'balance_insufficient' ? 1 : 0,
                                    upliftedDynamicRungs: dynamicPlan.upliftedDynamicRungs,
                                    keptDynamicRungs: dynamicPlan.keptDynamicRungs,
                                    droppedDynamicRungs: dynamicPlan.droppedDynamicRungs,
                                    minExecutableNotionalUsd: rungPlan.minExecutableNotionalUsd,
                                    redistributedNotionalUsd: rungPlan.redistributedNotionalUsd,
                                    upliftedNotionalUsd: dynamicPlan.totalUpliftedNotionalUsd,
                                    gt5PreservationStatus: rungPlan.gt5PreservationStatus,
                                    compressionReason: rungPlan.compressionReason,
                                    virtualRungCount: nRungs,
                                    materializedLimitPriceLadder,
                                    collapsedSamePriceRungs,
                                    gt5AutoReduceTriggered,
                                    runtimeRiskScalingState,
                                    directionalSizingSource,
                                });
                            }
                            break;
                        } catch (orderErr) {
                            const msg = parseErrorReason(orderErr);
                            this.rememberLiveExchangeConstraints(msg, token.tokenId, marketResult.market.slug);
                            if (this.isNonRetryableLiveReject(msg)) {
                                rungResults.push(`$${rungPrice.toFixed(2)}: 拒单(${msg})`);
                                this.addExecutionSkippedEntry({
                                    symbol,
                                    direction,
                                    confidence: prediction.confidence,
                                    reason: msg,
                                    executionDecision: 'exchange_rejected',
                                    requestedAmountUsd: rungBet,
                                    targetNotionalUsd,
                                    requestedNotionalUsd: effectiveTotalBet,
                                    postedNotionalUsd: 0,
                                    filledNotionalUsd: 0,
                                    requestCompressionMode: requestedNotional.requestCompressionMode,
                                    executionCompressionMode,
                                    requestLayerCompressionUsd,
                                    executionLayerCompressionUsd: dynamicPlan.executionCompressionUsd,
                                    executionLayerCompressionReason: msg,
                                    requestedShares: rungShares,
                                    originalRequestedShares: rungPlan.requestedShares,
                                    market: {
                                        title: marketResult.market.title,
                                        conditionId: marketResult.market.conditionId,
                                        slug: marketResult.market.slug,
                                    },
                                    token,
                                    exchangeMinOrderSizeShares: dynamicConstraint.minOrderSizeShares,
                                    attemptedRungs: dynamicPlan.attemptedRungs,
                                    compressedRungs: dynamicPlan.droppedDynamicRungs,
                                    postedRungs: 0,
                                    filledRungs: 0,
                                    exchangeRejectedRungs: 1,
                                    upliftedDynamicRungs: dynamicPlan.upliftedDynamicRungs,
                                    keptDynamicRungs: dynamicPlan.keptDynamicRungs,
                                    droppedDynamicRungs: dynamicPlan.droppedDynamicRungs,
                                    minExecutableNotionalUsd: rungPlan.minExecutableNotionalUsd,
                                    redistributedNotionalUsd: rungPlan.redistributedNotionalUsd,
                                    upliftedNotionalUsd: dynamicPlan.totalUpliftedNotionalUsd,
                                    gt5PreservationStatus: rungPlan.gt5PreservationStatus,
                                    compressionReason: rungPlan.compressionReason,
                                    virtualRungCount: nRungs,
                                    materializedLimitPriceLadder,
                                    collapsedSamePriceRungs,
                                    gt5AutoReduceTriggered,
                                    runtimeRiskScalingState,
                                    directionalSizingSource,
                                });
                                break;
                            }
                            if (orderRetry < RETRY_LIMIT - 1) {
                                console.log(`     ⚠️ 限价单下单失败，${RETRY_INTERVAL_MS/1000}s 后重试 [${orderRetry+1}/${RETRY_LIMIT}]`);
                                await new Promise(r => setTimeout(r, RETRY_INTERVAL_MS));
                            } else {
                                rungResults.push(`$${rungPrice.toFixed(2)}: 异常(${msg})`);
                                this.addExecutionSkippedEntry({
                                    symbol,
                                    direction,
                                    confidence: prediction.confidence,
                                    reason: msg,
                                    executionDecision: 'dynamic_order_exception',
                                    requestedAmountUsd: rungBet,
                                    targetNotionalUsd,
                                    requestedNotionalUsd: effectiveTotalBet,
                                    postedNotionalUsd: 0,
                                    filledNotionalUsd: 0,
                                    requestCompressionMode: requestedNotional.requestCompressionMode,
                                    executionCompressionMode,
                                    requestLayerCompressionUsd,
                                    executionLayerCompressionUsd: dynamicPlan.executionCompressionUsd,
                                    executionLayerCompressionReason: msg,
                                    requestedShares: rungShares,
                                    originalRequestedShares: rungPlan.requestedShares,
                                    market: {
                                        title: marketResult.market.title,
                                        conditionId: marketResult.market.conditionId,
                                        slug: marketResult.market.slug,
                                    },
                                    token,
                                    exchangeMinOrderSizeShares: dynamicConstraint.minOrderSizeShares,
                                    attemptedRungs: dynamicPlan.attemptedRungs,
                                    compressedRungs: dynamicPlan.droppedDynamicRungs,
                                    postedRungs: 0,
                                    filledRungs: 0,
                                    exchangeRejectedRungs: 0,
                                    upliftedDynamicRungs: dynamicPlan.upliftedDynamicRungs,
                                    keptDynamicRungs: dynamicPlan.keptDynamicRungs,
                                    droppedDynamicRungs: dynamicPlan.droppedDynamicRungs,
                                    minExecutableNotionalUsd: rungPlan.minExecutableNotionalUsd,
                                    redistributedNotionalUsd: rungPlan.redistributedNotionalUsd,
                                    upliftedNotionalUsd: dynamicPlan.totalUpliftedNotionalUsd,
                                    gt5PreservationStatus: rungPlan.gt5PreservationStatus,
                                    compressionReason: rungPlan.compressionReason,
                                    virtualRungCount: nRungs,
                                    materializedLimitPriceLadder,
                                    collapsedSamePriceRungs,
                                    gt5AutoReduceTriggered,
                                    runtimeRiskScalingState,
                                    directionalSizingSource,
                                });
                            }
                        }
                    }
                } else if (this.env.TRADING_MODE === TradingMode.LIVE) {
                    console.error(`     ❌ [LIVE 防护] clobClient 未初始化，拒绝下单`);
                    rungResults.push(`$${rungPrice.toFixed(2)}: LIVE clobClient 缺失`);
                } else {
                    rungResults.push(`$${rungPrice.toFixed(2)}: [回测] 记录`);
                    totalFilled += rungBet;
                }
            }

            // 汇总日志
            const fillRateBase = dynamicPlan.totalSelectedNotionalUsd > 0 ? dynamicPlan.totalSelectedNotionalUsd : effectiveTotalBet;
            const fillRate = fillRateBase > 0 ? ((totalFilled / fillRateBase) * 100).toFixed(1) : '0.0';
            console.log(`\n  🪜 ${symbol} 阶梯结果 (已确认成交率 ${fillRate}%):`);
            for (const r of rungResults) {
                console.log(`     ${r}`);
            }
            if (totalPendingMonitorAmount > 0) {
                console.log(`     挂单待成交: $${totalPendingMonitorAmount.toFixed(2)} / $${dynamicPlan.totalSelectedNotionalUsd.toFixed(2)}`);
            }
            console.log(`     已确认成交: $${totalFilled.toFixed(2)} / $${dynamicPlan.totalSelectedNotionalUsd.toFixed(2)}, 虚拟档${nRungs} -> 真实档${postedRungs.length}, 资金: $${this.state.currentCapital.toFixed(2)}`);

            const hadExchangeReject = rungResults.some((r) => r.includes('拒单(') || r.includes('失败(') || r.includes('异常('));
            const detailedRejectReasons = rungResults
                .filter((r) => r.includes('拒单(') || r.includes('失败(') || r.includes('异常(') || r.includes('预检失败('))
                .slice(0, 3)
                .join(' | ');
            const noFillReasons = rungResults
                .filter((r) => r.includes('挂单监控中') || r.includes('已挂出待成交') || r.includes('本地过滤('))
                .slice(0, 3)
                .join(' | ');
            results.push({
                symbol,
                success: totalFilled > 0,
                amount: totalFilled,
                direction,
                pendingMonitor: totalPendingMonitorAmount > 0,
                error: totalFilled === 0
                    ? (
                        totalPendingMonitorAmount > 0
                            ? undefined
                            : (
                                hadExchangeReject
                                    ? (detailedRejectReasons ? `全部被交易所拒绝: ${detailedRejectReasons}` : '全部被交易所拒绝')
                                    : (noFillReasons ? `全部未成交: ${noFillReasons}` : '全部未成交')
                            )
                    )
                    : undefined,
            });
        }

        // 打印摘要（区分"未成交(挂单)" / "全部被拒" / 真正的"失败"）
        const successful = results.filter((r) => r.success).length;
        const ladderMonitoring = results.filter((r) => r.pendingMonitor).length;
        const ladderUnfilled = results.filter((r) => !r.success && String(r.error || '').includes('全部未成交')).length;
        const ladderRejected = results.filter((r) => !r.success && String(r.error || '').includes('全部被交易所拒绝')).length;
        const ladderRealFailed = results.filter((r) => {
            const error = String(r.error || '');
            return !r.success
                && !r.pendingMonitor
                && !error.includes('全部未成交')
                && !error.includes('全部被交易所拒绝')
                && !error.includes('金额不足');
        }).length;
        const totalFilledAll = results.filter(r => r.success).reduce((s, r) => s + (r.amount || 0), 0);
        const ladderParts: string[] = [`成交 ${successful} 笔`];
        if (ladderMonitoring > 0) ladderParts.push(`挂单待成交 ${ladderMonitoring} 笔`);
        if (ladderUnfilled > 0) ladderParts.push(`未成交(挂单) ${ladderUnfilled} 笔`);
        if (ladderRejected > 0) ladderParts.push(`交易所拒绝 ${ladderRejected} 笔`);
        if (ladderRealFailed > 0) ladderParts.push(`失败 ${ladderRealFailed} 笔`);
        console.log('\n' + '─'.repeat(60));
        console.log(`  📋 阶梯限价单摘要: ${ladderParts.join(', ')}, 成交额 $${totalFilledAll.toFixed(2)}, 资金 $${this.state.currentCapital.toFixed(2)}`);

        // ─── 逆向选择统计（Adverse Selection Monitor）───
        if (predictions.length > 0) {
            const filledConfs: number[] = [];
            const unfilledConfs: number[] = [];
            for (let i = 0; i < results.length && i < predictions.length; i++) {
                const conf = predictions[i]?.confidence ?? 0;
                if (results[i].success && (results[i].amount ?? 0) > 0) {
                    filledConfs.push(conf);
                } else if (String(results[i].error || '').includes('全部未成交')) {
                    unfilledConfs.push(conf);
                }
            }
            const avgFilled = filledConfs.length > 0
                ? (filledConfs.reduce((a, b) => a + b, 0) / filledConfs.length * 100).toFixed(1)
                : '-';
            const avgUnfilled = unfilledConfs.length > 0
                ? (unfilledConfs.reduce((a, b) => a + b, 0) / unfilledConfs.length * 100).toFixed(1)
                : '-';
            const fillRate = predictions.length > 0
                ? ((filledConfs.length / predictions.length) * 100).toFixed(0)
                : '0';
            console.log(`  📊 逆向选择监控: 成交率 ${fillRate}% | 成交均置信 ${avgFilled}% | 未成交均置信 ${avgUnfilled}%`);

            // 累计统计
            if (!this._adverseSelectionStats) {
                this._adverseSelectionStats = { totalFilled: 0, totalUnfilled: 0, sumConfFilled: 0, sumConfUnfilled: 0 };
            }
            this._adverseSelectionStats.totalFilled += filledConfs.length;
            this._adverseSelectionStats.totalUnfilled += unfilledConfs.length;
            this._adverseSelectionStats.sumConfFilled += filledConfs.reduce((a, b) => a + b, 0);
            this._adverseSelectionStats.sumConfUnfilled += unfilledConfs.reduce((a, b) => a + b, 0);
            const stats = this._adverseSelectionStats;
            const cumFillRate = (stats.totalFilled + stats.totalUnfilled) > 0
                ? ((stats.totalFilled / (stats.totalFilled + stats.totalUnfilled)) * 100).toFixed(1)
                : '0';
            const cumAvgFilled = stats.totalFilled > 0
                ? ((stats.sumConfFilled / stats.totalFilled) * 100).toFixed(1)
                : '-';
            const cumAvgUnfilled = stats.totalUnfilled > 0
                ? ((stats.sumConfUnfilled / stats.totalUnfilled) * 100).toFixed(1)
                : '-';
            console.log(`  📊 累计统计: 成交率 ${cumFillRate}% (${stats.totalFilled}/${stats.totalFilled + stats.totalUnfilled}) | 成交均置信 ${cumAvgFilled}% | 未成交均置信 ${cumAvgUnfilled}%`);
        }
        console.log('─'.repeat(60) + '\n');

        await this.reconcileLiveStateAfterOrder('ladder_order');

        return results;
    }

    // ─── 模拟限价单持续监控 ───────────────────────────────────

    /**
     * 启动挂单监控定时器（如果尚未启动）。
     * 每 5s 检查一次所有 pendingSimOrders：
     *   - 到期？退还冻结资金，移除
     *   - 查订单簿，best_ask <= limitPrice？成交，记录交易
     */
    startPendingSimMonitor(): void {
        if (this.pendingSimMonitorTimer) return;  // 已在运行
        this.pendingSimMonitorTimer = setInterval(() => {
            this._checkPendingSimOrders().catch(err => {
                console.error(`  [挂单监控] 异常: ${err instanceof Error ? err.message : err}`);
            });
        }, PENDING_SIM_MONITOR_INTERVAL_MS);
    }

    /**
     * 停止挂单监控定时器
     */
    private stopPendingSimMonitor(): void {
        if (this.pendingSimMonitorTimer) {
            clearInterval(this.pendingSimMonitorTimer);
            this.pendingSimMonitorTimer = null;
        }
    }

    /**
     * 获取当前挂单数量（供外部查询）
     */
    getPendingSimOrderCount(): number {
        return this.pendingSimOrders.length;
    }

    /**
     * 检查所有待成交的模拟限价单
     */
    private async _checkPendingSimOrders(): Promise<void> {
        if (this.pendingSimOrders.length === 0) {
            this.stopPendingSimMonitor();
            return;
        }

        const now = new Date();
        const toRemove: string[] = [];
        let changed = false;

        for (const order of this.pendingSimOrders) {
            if (order.checking) continue;
            order.checking = true;

            try {
                // ─── 到期检查：市场周期已结束，取消挂单退还资金 ───
                if (now >= order.expiresAt) {
                    this.adjustCoinCapital(order.symbol, order.frozenAmount);
                    this.appendPendingOrderLedgerEvent({
                        event: 'expired_refunded',
                        orderId: order.id,
                        symbol: order.symbol,
                        direction: order.direction,
                        amountUsd: order.frozenAmount,
                        remainingFrozenAmount: 0,
                        limitPrice: order.limitPrice,
                        tokenId: order.tokenId,
                        tokenOutcome: order.tokenOutcome,
                        confidence: order.confidence,
                        marketTitle: order.marketTitle,
                        conditionId: order.conditionId,
                        marketSlug: order.marketSlug,
                        targetPeriodEndTs: order.targetPeriodEndTs,
                        createdAt: order.createdAt.toISOString(),
                        expiresAt: order.expiresAt.toISOString(),
                    });
                    console.log(`  📡 [挂单到期] ${order.symbol} @$${order.limitPrice.toFixed(2)}: 退还 $${order.frozenAmount.toFixed(2)}, 资金 $${this.state.currentCapital.toFixed(2)}`);
                    toRemove.push(order.id);
                    changed = true;
                    continue;
                }

                // ─── 查订单簿（asks + bids）───
                let bookSnapshot: QueueAwareBookSnapshot = {
                    bestAskPrice: 999,
                    asks: [],
                    bids: [],
                    hadBidsData: false,
                };

                try {
                    const orderBook = await getOrderBook(order.tokenId);
                    bookSnapshot = buildQueueAwareBookSnapshot(orderBook);
                } catch { /* 网络失败，下次再检查 */ }

                const simulatedFill = simulateQueueAwareBuyFillWithHaircut(bookSnapshot, order.limitPrice, order.frozenAmount);
                const bestAskPrice = simulatedFill.bestAskPrice;
                const avgFillPrice = simulatedFill.avgFillPrice;
                const bidDepthAtLimit = simulatedFill.queueCompetitionUsd;
                const fillableAmount = simulatedFill.haircutAdjustedFillableUsd;
                const effectiveFillable = simulatedFill.haircutAdjustedFillableUsd;
                const effectiveFillableUsdAtEntry = simulatedFill.effectiveFillableUsd;
                const bidDepthNote = bidDepthAtLimit === 0 && !simulatedFill.hadBidsData ? ' (无买单深度数据)' : '';

                // ─── 成交判定（使用扣除竞争后的有效流动性）───
                if (bestAskPrice <= order.limitPrice) {
                    let filledAmount: number;
                    if (effectiveFillable >= order.frozenAmount) {
                        // 完全成交
                        filledAmount = order.frozenAmount;
                        this.appendPendingOrderLedgerEvent({
                            event: 'filled',
                            orderId: order.id,
                            symbol: order.symbol,
                            direction: order.direction,
                            amountUsd: filledAmount,
                            remainingFrozenAmount: 0,
                            limitPrice: order.limitPrice,
                            tokenId: order.tokenId,
                            tokenOutcome: order.tokenOutcome,
                            confidence: order.confidence,
                            marketTitle: order.marketTitle,
                            conditionId: order.conditionId,
                            marketSlug: order.marketSlug,
                            targetPeriodEndTs: order.targetPeriodEndTs,
                            createdAt: order.createdAt.toISOString(),
                            expiresAt: order.expiresAt.toISOString(),
                            bestAsk: bestAskPrice,
                            avgFillPrice: avgFillPrice,
                            queueCompetitionUsdAtEntry: bidDepthAtLimit,
                            effectiveFillableUsdAtEntry,
                        });
                        console.log(`  📡 [挂单成交] ${order.symbol} @$${order.limitPrice.toFixed(2)}: 完全成交 $${filledAmount.toFixed(2)} (best_ask=$${bestAskPrice.toFixed(3)}, 均价=$${avgFillPrice.toFixed(4)}, 买方竞争=$${bidDepthAtLimit.toFixed(0)}${bidDepthNote})`);
                        toRemove.push(order.id);
                        changed = true;
                    } else if (effectiveFillable > 0) {
                        // 部分成交
                        filledAmount = effectiveFillable;
                        order.frozenAmount = roundUsd(order.frozenAmount - filledAmount);
                        this.appendPendingOrderLedgerEvent({
                            event: 'partial_fill',
                            orderId: order.id,
                            symbol: order.symbol,
                            direction: order.direction,
                            amountUsd: filledAmount,
                            remainingFrozenAmount: order.frozenAmount,
                            limitPrice: order.limitPrice,
                            tokenId: order.tokenId,
                            tokenOutcome: order.tokenOutcome,
                            confidence: order.confidence,
                            marketTitle: order.marketTitle,
                            conditionId: order.conditionId,
                            marketSlug: order.marketSlug,
                            targetPeriodEndTs: order.targetPeriodEndTs,
                            createdAt: order.createdAt.toISOString(),
                            expiresAt: order.expiresAt.toISOString(),
                            bestAsk: bestAskPrice,
                            avgFillPrice: avgFillPrice,
                            queueCompetitionUsdAtEntry: bidDepthAtLimit,
                            effectiveFillableUsdAtEntry,
                        });
                        changed = true;
                        const pct = ((filledAmount / (filledAmount + order.frozenAmount)) * 100).toFixed(0);
                        console.log(`  📡 [挂单部分成交] ${order.symbol} @$${order.limitPrice.toFixed(2)}: 成交 $${filledAmount.toFixed(2)} (${pct}%), 剩余冻结 $${order.frozenAmount.toFixed(2)} (买方竞争=$${bidDepthAtLimit.toFixed(0)}${bidDepthNote})`);
                        // 不移除，继续监控剩余部分
                    } else {
                        filledAmount = 0;
                        // 有卖方流动性但买方排队太深，暂不成交
                        if (fillableAmount > 0 && bidDepthAtLimit > 0) {
                            console.log(`  📡 [挂单排队] ${order.symbol} @$${order.limitPrice.toFixed(2)}: 卖方流动性 $${fillableAmount.toFixed(0)} < 买方竞争 $${bidDepthAtLimit.toFixed(0)}, 继续等待`);
                        }
                    }

                    if (filledAmount > 0) {
                        // 记录交易
                        const pendingLogEntry = this.logger.addEntry({
                            symbol: order.symbol,
                            direction: order.direction,
                            confidence: order.confidence,
                            amount: filledAmount,
                            mode: this.env.TRADING_MODE,
                            status: 'executed',
                            marketTitle: order.marketTitle,
                            conditionId: order.conditionId,
                            tokenId: order.tokenId,
                            marketSlug: order.marketSlug,
                            tokenOutcome: order.tokenOutcome,
                            tokenPrice: avgFillPrice,             // 实际加权均价（非限价）
                            avgActualFillPrice: avgFillPrice,
                            limitPriceConfigured: order.limitPrice, // 审计：配置限价
                            bestAskAtEntry: bestAskPrice,
                            queueCompetitionUsdAtEntry: bidDepthAtLimit,
                            effectiveFillableUsdAtEntry,
                        });
                        console.log(`     💰 交易已记录, 资金: $${this.state.currentCapital.toFixed(2)}`);
                    }
                }
                // best_ask > limitPrice → 继续等待
            } finally {
                order.checking = false;
            }
        }

        // 移除已处理的订单
        if (toRemove.length > 0) {
            this.pendingSimOrders = this.pendingSimOrders.filter(o => !toRemove.includes(o.id));
            changed = true;
            if (this.pendingSimOrders.length === 0) {
                this.stopPendingSimMonitor();
                console.log(`  📡 [挂单监控] 所有挂单已处理完毕，监控已停止`);
            } else {
                console.log(`  📡 [挂单监控] 剩余 ${this.pendingSimOrders.length} 笔挂单监控中`);
            }
        }

        if (changed) {
            this.persistPendingSimOrders();
        }
    }

    /**
     * 执行一轮预测交易
     * @param opts.targetPeriodEndTs 目标 15M 的 slug ts（=周期开始，与 predictions.json、Polymarket 官网一致），按此 ts 下单
     * @param opts.predictions 已解析且过滤好的预测（扫描模式下传入，避免再读文件）
     */
    async executeRound(opts?: { targetPeriodEndTs?: number; predictions?: PredictionResult[] }): Promise<ExecutionResult[]> {
        const results: ExecutionResult[] = [];
        if (this.env.TRADING_MODE === TradingMode.LIVE) {
            await this.prepareLiveCooldownStateBeforeOrder();
        } else {
            await this.settleHistory();
        }
        if (this.env.TRADING_MODE !== TradingMode.LIVE) {
            // 用历史交易+挂单冻结重锚资金，避免某些旧执行分支的 state 漂移继续影响后续下注比例。
            this._recomputeCapitalFromHistory();
        }
        this.state.todayTradeCount = this.logger.getTodayTradeCount();
        
        console.log('\n' + '═'.repeat(60));
        console.log(`  🎯 预测交易执行 - ${new Date().toLocaleString('zh-CN')}`);
        if (typeof opts?.targetPeriodEndTs === 'number') {
            const rangeBeijing = this._fmtPeriod(opts.targetPeriodEndTs);
            const tf = this.env.TIMEFRAME;
            console.log(`  📌 目标周期: ${rangeBeijing} (slug: *-updown-${tf}-${opts.targetPeriodEndTs})`);
        }
        console.log('═'.repeat(60));
        
        const predictions: PredictionResult[] = Array.isArray(opts?.predictions)
            ? opts.predictions
            : await getHighConfidencePredictions();
        
        if (predictions.length === 0) {
            return results;
        }
        
        printPredictions(predictions);
        
        const symbolsToFetch = [...new Set(predictions.map((p) => p.symbol.replace(/\/.*$/, '')))];
        const tf = this.env.TIMEFRAME;
        console.log(`🔍 查找市场 (按 target_period_end_ts 构建 slug: *-updown-${tf}-{ts})...`);
        const markets = await findAllMarkets(this.env.TIMEFRAME, opts?.targetPeriodEndTs, symbolsToFetch);
        for (const result of markets) {
            this.state.marketCache.set(result.symbol, result);
        }
        
        const tradesToExecute = predictions.length;
        console.log(`\n📈 执行 ${tradesToExecute} 笔交易...\n`);
        /** 本轮已对冷却做过自减的 symbol，避免同一周期内同一 symbol 出现多次时重复自减 */
        const cooldownDecrementedThisRound = new Set<string>();

        for (let i = 0; i < tradesToExecute; i++) {
            const prediction = predictions[i];
            const symbol = prediction.symbol.replace('/USDT', '');
            const marketResult = this.state.marketCache.get(symbol);
            
            if (!marketResult) {
                console.log(`\n❌ ${symbol}: 市场数据未缓存`);
                results.push({
                    symbol,
                    success: false,
                    error: '市场数据未缓存',
                });
                continue;
            }
            
            const result = await this.executePrediction(prediction, marketResult, { targetPeriodEndTs: opts?.targetPeriodEndTs, cooldownDecrementedThisRound });
            results.push(result);
            
            // 短暂延迟避免速率限制
            if (i < tradesToExecute - 1) {
                console.log('\n⏳ 等待 1 秒后继续...');
                await new Promise((resolve) => setTimeout(resolve, 1000));
            }
        }
        
        // 打印摘要
        const successful = results.filter((r) => r.success).length;
        const skipped = results.filter((r) => r.skippedReason === 'already_bought').length;
        const failed = results.filter((r) => !r.success && r.skippedReason !== 'already_bought').length;
        
        console.log('\n' + '─'.repeat(60));
        const periodLabel = typeof opts?.targetPeriodEndTs === 'number' ? this._fmtPeriod(opts.targetPeriodEndTs) : '';
        const summaryParts = [`成功 ${successful} 笔`];
        if (skipped > 0) summaryParts.push(`跳过(已成交) ${skipped} 笔`);
        if (failed > 0) summaryParts.push(`失败 ${failed} 笔`);
        console.log(`  📋 本轮摘要${periodLabel ? ` [${periodLabel}]` : ''}: ${summaryParts.join(', ')}, 资金 $${this.state.currentCapital.toFixed(2)}`);
        // 若有跳过(已成交)，单独一行说明
        if (skipped > 0) {
            const skippedResults = results.filter((r) => r.skippedReason === 'already_bought');
            for (const r of skippedResults) {
                console.log(`  ⏭️ ${r.symbol}: ${r.error || '已成交'}`);
            }
        }
        // 若有失败，打印失败原因便于排查（带周期便于对照）
        if (failed > 0) {
            const failedResults = results.filter((r) => !r.success && r.skippedReason !== 'already_bought');
            for (const r of failedResults) {
                const periodPart = r.periodDisplay ? ` [${r.periodDisplay}]` : '';
                console.log(`  ❌ ${r.symbol}${periodPart}: ${r.error || '未知原因'}`);
            }
        }
        console.log('─'.repeat(60) + '\n');

        await this.reconcileLiveStateAfterOrder('execute_round');
        
        this.state.lastExecutionTime = new Date();
        
        return results;
    }
    
    /**
     * 检查价格监控队列，如果价格降到阈值以下则下单
     * 静默检查，只在关键事件时显示日志（触发下单、超时、API重连）
     */
    async checkPriceMonitorQueue(): Promise<void> {
        if (this.priceMonitorQueue.length === 0) return;
        
        const now = new Date();
        const entriesToRemove: number[] = [];
        const entriesToExecute: PriceMonitorEntry[] = [];
        let apiErrorCount = 0;
        
        for (let i = 0; i < this.priceMonitorQueue.length; i++) {
            const entry = this.priceMonitorQueue[i];
            const periodEndTime = new Date((entry.targetPeriodEndTs + this.periodSeconds) * 1000);

            // 已过周期结束：静默移除（真正过期）
            if (now >= periodEndTime) {
                entriesToRemove.push(i);
                continue;
            }
            // 已过「可下单截止」但仍在最后5分钟内：不在此处 continue，到下面获取价格后按「最后5分钟市价带」处理

            // 检查是否已经在处理中（避免重复触发）
            if (entry.isProcessing) {
                // 已经在处理中，跳过（避免重复触发）
                continue;
            }
            
            // 检查是否已下过单（包括成功、失败和pending的）
            if (entry.marketResult.market) {
                const conditionId = entry.marketResult.market.conditionId;
                // 检查是否已成功下单
                if (this.logger.hasAlreadyBought(conditionId)) {
                    entriesToRemove.push(i);
                    continue;
                }
                // 检查是否已有该市场的交易记录（包括成功、失败和pending的），避免重复尝试
                const allEntries = this.logger.getAllEntries();
                const monitorStartTimeMs = entry.monitorStartTime.getTime();
                const hasTradeForMarket = allEntries.some(e => 
                    e.conditionId === conditionId && 
                    (e.status === 'executed' || e.status === 'failed' || e.status === 'pending') && 
                    new Date(e.timestamp).getTime() >= monitorStartTimeMs - 60000 // 1分钟内的记录
                );
                if (hasTradeForMarket) {
                    entriesToRemove.push(i);
                    continue;
                }
            }
            
            // 获取当前价格
            if (!entry.prediction.direction) {
                entriesToRemove.push(i);
                continue;
            }
            const token = getTokenForDirection(entry.marketResult.market!, entry.prediction.direction);
            if (!token || !token.tokenId) {
                // token 不存在是数据问题，移除条目避免永久占用队列
                entriesToRemove.push(i);
                continue;
            }
            
            let currentPrice: number | null = null;
            let apiError = false;
            
            try {
                currentPrice = await getTokenPrice(token.tokenId, 'BUY');
                if (currentPrice === null) {
                    apiError = true;
                    apiErrorCount++;
                }
            } catch (error) {
                apiError = true;
                apiErrorCount++;
            }
            
            // API错误时，跳过本次检查，等待下次重试（临时性错误不移除条目）
            if (apiError) {
                continue;
            }

            if (currentPrice === null) continue;
            const executionDecision = this.applyExecutionV2MonitorDecision(entry, currentPrice, now);
            if (executionDecision.shouldCancel) {
                entriesToRemove.push(i);
                continue;
            }
            const legacyFinalAggressive = now >= entry.monitorEndTime
                && currentPrice >= PredictionExecutor.LAST5MIN_MIN_PRICE
                && currentPrice <= PredictionExecutor.LAST5MIN_MAX_PRICE;
            const effectiveExecutionMaxPrice = legacyFinalAggressive
                ? Math.max(executionDecision.effectiveMaxPrice, PredictionExecutor.LAST5MIN_MAX_PRICE)
                : executionDecision.effectiveMaxPrice;

            // 最后5分钟市价成交：已过可下单截止(monitorEndTime)但未过周期结束
            if (now >= entry.monitorEndTime) {
                if (executionDecision.shouldExecute || legacyFinalAggressive) {
                    if (entry.marketResult.market) {
                        const conditionId = entry.marketResult.market.conditionId;
                        if (this.logger.hasAlreadyBought(conditionId)) {
                            entriesToRemove.push(i);
                            continue;
                        }
                        // 最后5分钟：仅「已成交」视为已买，failed/pending 仍允许重试
                        const allEntries = this.logger.getAllEntries();
                        const monitorStartTimeMs = entry.monitorStartTime.getTime();
                        const hasExecutedForMarket = allEntries.some(e =>
                            e.conditionId === conditionId && e.status === 'executed' &&
                            new Date(e.timestamp).getTime() >= monitorStartTimeMs - 60000
                        );
                        if (hasExecutedForMarket) {
                            entriesToRemove.push(i);
                            continue;
                        }
                    }
                    entry.isProcessing = true;
                    entry.triggerPrice = currentPrice;
                    entry.currentMaxPrice = effectiveExecutionMaxPrice;
                    entriesToRemove.push(i);
                    entriesToExecute.push(entry);
                    const symbol = entry.prediction.symbol.replace('/USDT', '');
                    const periodStr = this._fmtPeriod(entry.targetPeriodEndTs);
                    console.log(`\n✅ ${symbol} 最后窗口触发 [${periodStr}]: $${entry.initialPrice.toFixed(4)} (进入监控) → $${currentPrice.toFixed(4)} (当前) ≤ $${effectiveExecutionMaxPrice.toFixed(2)}，立即下单`);
                } else {
                    // 未触发时打一次节流日志（每条目每分钟最多一条），便于排查。轮询路径下 API 返回 null 时会在上面 continue，此处多为「价格不在允许范围」
                    const symbol = entry.prediction.symbol.replace('/USDT', '');
                    const periodStr = this._fmtPeriod(entry.targetPeriodEndTs);
                    const lastLog = (entry as any).last5MinLogTime as number | undefined;
                    if (lastLog == null || now.getTime() - lastLog >= 60000) {
                        (entry as any).last5MinLogTime = now.getTime();
                        if (currentPrice == null) {
                            console.log(`\n⏳ ${symbol} 最后5分钟 [${periodStr}]: 无报价，本次不触发（等下一轮询/WS）`);
                        } else {
                            console.log(`\n⏳ ${symbol} 最后5分钟 [${periodStr}]: 当前价 $${currentPrice.toFixed(4)} 不在 [${PredictionExecutor.LAST5MIN_MIN_PRICE.toFixed(2)}, ${PredictionExecutor.LAST5MIN_MAX_PRICE.toFixed(2)}]，不触发`);
                        }
                    }
                }
                continue;
            }

            // 检查价格是否进入允许范围 / 追价上界
            if (executionDecision.shouldExecute) {
                    // 检查是否已经在处理中（避免重复触发）
                    if (entry.isProcessing) {
                        // 已经在处理中，跳过（避免重复触发）
                        continue;
                    }
                    
                    // 检查是否已有该市场的交易记录（包括成功、失败和pending），避免重复尝试
                    if (entry.marketResult.market) {
                        const conditionId = entry.marketResult.market.conditionId;
                        const allEntries = this.logger.getAllEntries();
                        const monitorStartTimeMs = entry.monitorStartTime.getTime();
                        const hasTradeForMarket = allEntries.some(e => 
                            e.conditionId === conditionId && 
                            (e.status === 'executed' || e.status === 'failed' || e.status === 'pending') && 
                            new Date(e.timestamp).getTime() >= monitorStartTimeMs - 60000 // 1分钟内的记录
                        );
                        if (hasTradeForMarket) {
                            // 已有交易记录，移除条目
                            entriesToRemove.push(i);
                            continue;
                        }
                    }
                    
                    // 检查是否已下过单（避免重复下单）
                    if (entry.marketResult.market) {
                        const conditionId = entry.marketResult.market.conditionId;
                        if (this.logger.hasAlreadyBought(conditionId)) {
                            // 已下过单，移除条目
                            entriesToRemove.push(i);
                            continue;
                        }
                    }
                    
                    // 标记为正在处理，避免重复触发
                    entry.isProcessing = true;
                    // 保存触发价格，避免重新获取
                    entry.triggerPrice = currentPrice;
                    entry.currentMaxPrice = effectiveExecutionMaxPrice;
                    // 立即移除条目，避免重复触发（即使下单失败也不再监控）
                    entriesToRemove.push(i);
                    entriesToExecute.push(entry);
                    const symbol = entry.prediction.symbol.replace('/USDT', '');
                    const periodStr = this._fmtPeriod(entry.targetPeriodEndTs);
                    console.log(`\n✅ ${symbol} 价格监控触发 [${periodStr}]: $${entry.initialPrice.toFixed(4)} (进入监控) → $${currentPrice.toFixed(4)} (当前) ≤ $${effectiveExecutionMaxPrice.toFixed(2)}，满足条件，立即下单`);
            }
        }
        
        // 如果API错误，显示重连日志（不限次数）
        if (apiErrorCount > 0) {
            console.log(`\n🔄 价格监控API错误，${apiErrorCount} 个请求失败，将重试（使用实时价格）`);
        }
        
        // 先移除已处理或过期的条目（从后往前删除，避免索引错乱）
        // 注意：必须在执行订单前移除，避免重复触发
        for (let i = entriesToRemove.length - 1; i >= 0; i--) {
            this.priceMonitorQueue.splice(entriesToRemove[i], 1);
        }
        
        // 执行满足条件的订单（带重试机制）
        for (const entry of entriesToExecute) {
            const symbol = entry.prediction.symbol.replace('/USDT', '');
            let lastError: string | null = null;
            let success = false;
            // 使用保存的触发价格，如果没有则尝试获取（降级处理）
            let triggerPrice: number | null = entry.triggerPrice ?? null;
            
            // 如果没有保存的触发价格，尝试获取（降级处理）
            if (triggerPrice === null) {
                try {
                    if (entry.marketResult.market && entry.prediction.direction) {
                        const token = getTokenForDirection(entry.marketResult.market, entry.prediction.direction);
                        if (token && token.tokenId) {
                            triggerPrice = await getTokenPrice(token.tokenId, 'BUY');
                        }
                    }
                } catch (error) {
                    // 忽略价格获取错误
                }
            }
            
            // 重试3次（有预签单时触发即 POST，最快路径）
            for (let retry = 0; retry < 3; retry++) {
                try {
                    const result = await this.executePrediction(entry.prediction, entry.marketResult, {
                        targetPeriodEndTs: entry.targetPeriodEndTs,
                        triggerPrice: triggerPrice ?? undefined,
                        asksSnapshot: entry.asksSnapshot,
                        preSignedOrder: entry.preSignedOrder,
                        maxPrice: entry.currentMaxPrice,
                    });
                    
                    if (result.success) {
                        success = true;
                        break; // 成功，退出重试循环
                    } else {
                        lastError = result.error || '未知错误';
                        if (retry < 2) {
                            // 不是最后一次重试，等待1秒后重试
                            await new Promise(resolve => setTimeout(resolve, 1000));
                        }
                    }
                } catch (error) {
                    lastError = error instanceof Error ? error.message : String(error);
                    if (retry < 2) {
                        // 不是最后一次重试，等待1秒后重试
                        await new Promise(resolve => setTimeout(resolve, 1000));
                    }
                }
            }
            
            // 打印最终结果
            if (!success) {
                const periodStr = this._fmtPeriod(entry.targetPeriodEndTs);
                console.log(`\n${'═'.repeat(60)}`);
                console.log(`  ❌ ${symbol} 价格监控下单失败报告 [${periodStr}]`);
                console.log(`${'═'.repeat(60)}`);
                console.log(`  周期:       ${periodStr}`);
                console.log(`  初始价格:   $${entry.initialPrice.toFixed(4)}`);
                console.log(`  触发价格:   $${triggerPrice !== null ? triggerPrice.toFixed(4) : 'N/A'}`);
                console.log(`  重试次数:   3 次`);
                console.log(`  最终状态:   失败`);
                const displayError = lastError && lastError.startsWith('价格监控中:')
                    ? '触发后重新校验价格超阈值，已取消'
                    : (lastError || '未知错误');
                console.log(`  错误原因:   ${displayError}`);
                console.log(`  市场:       ${entry.marketResult.market?.title || 'N/A'}`);
                console.log(`  条件ID:     ${entry.marketResult.market?.conditionId?.slice(0, 20) || 'N/A'}...`);
                console.log(`${'═'.repeat(60)}\n`);
            }
        }
    }

    /**
     * 监控入队后异步预签 GTD 限价单（仅 LIVE）：触发时优先 POST 预签单，失败回退触发时签名。
     * 多钱包时后续可扩展为预签多笔并存入 entry.preSignedOrders。
     */
    private schedulePreSignForPriceMonitorEntry(entry: PriceMonitorEntry): void {
        if (!this.clobClient || this.env.TRADING_MODE !== TradingMode.LIVE) return;
        const token = entry.marketResult.market && entry.prediction.direction
            ? getTokenForDirection(entry.marketResult.market, entry.prediction.direction)
            : null;
        if (!token?.tokenId) return;
        const symbol = entry.prediction.symbol.replace('/USDT', '');
        const betAmount = this.calculateBetAmount(entry.prediction.confidence, undefined, symbol, entry.prediction.direction);
        const periodStartSec = entry.targetPeriodEndTs || (Date.now() / 1000);
        const expirationSec = this.computeLiveGtdExpiration(periodStartSec);
        const shares = Math.floor((betAmount / this.MAX_PRICE_THRESHOLD) * 100) / 100;
        if (!Number.isFinite(shares) || shares <= 0) return;
        if (!expirationSec) return;
        const userOrder = {
            tokenID: token.tokenId,
            price: this.MAX_PRICE_THRESHOLD,
            size: shares,
            side: Side.BUY,
            expiration: expirationSec,
        };
        this.clobClient.createOrder(userOrder, { tickSize: '0.01' })
            .then((signed) => { entry.preSignedOrder = signed; })
            .catch(() => { entry.preSignedOrder = undefined; });
    }

    /**
     * 供 WebSocket 市场频道使用：返回当前价格/低价/流动性监控队列中所有 tokenId，用于订阅 agg_orderbook
     */
    getMonitorTokenIds(): string[] {
        const ids: string[] = [];
        for (const entry of this.priceMonitorQueue) {
            if (entry.marketResult.market && entry.prediction.direction) {
                const token = getTokenForDirection(entry.marketResult.market, entry.prediction.direction);
                if (token?.tokenId) ids.push(token.tokenId);
            }
        }
        for (const entry of this.lowPriceMonitorQueue) {
            if (entry.marketResult.market && entry.prediction.direction) {
                const token = getTokenForDirection(entry.marketResult.market, entry.prediction.direction);
                if (token?.tokenId) ids.push(token.tokenId);
            }
        }
        for (const entry of this.liquidityMonitorQueue) {
            if (entry.marketResult.market && entry.prediction.direction) {
                const token = getTokenForDirection(entry.marketResult.market, entry.prediction.direction);
                if (token?.tokenId) ids.push(token.tokenId);
            }
        }
        // 模拟限价挂单池: 挂单期间需要 WS 订阅以便影子追踪器检测穿价
        for (const order of this.pendingSimOrders) {
            if (order.tokenId) ids.push(order.tokenId);
        }
        return ids;
    }
    
    /**
     * WebSocket 收到 best_ask 更新时调用：若该 token 在高价监控队列且 best_ask ≤ 阈值，立即触发下单
     * @param asksSnapshot 可选，来自 WS book 消息时传入，下单时跳过 REST getOrderBook（提速）
     */
    async tryTriggerPriceMonitorByTokenId(assetId: string, bestAsk: number, asksSnapshot?: Array<{ price: string; size: string }>): Promise<void> {
        const now = new Date();
        let idx = -1;
        let entry: PriceMonitorEntry | null = null;
        for (let i = 0; i < this.priceMonitorQueue.length; i++) {
            const e = this.priceMonitorQueue[i];
            if (!e.marketResult.market || !e.prediction.direction) continue;
            const token = getTokenForDirection(e.marketResult.market, e.prediction.direction);
            if (token?.tokenId === assetId) {
                const periodEndTime = new Date((e.targetPeriodEndTs + this.periodSeconds) * 1000);
                if (now >= periodEndTime) {
                    this.priceMonitorQueue.splice(i, 1);
                    return;
                }
                if (now >= e.monitorEndTime) {
                    // 最后5分钟：仅当 bestAsk 在 [0.3, 0.65] 时触发市价成交；仅「已成交」视为已买，failed/pending 仍可重试
                    if (bestAsk >= PredictionExecutor.LAST5MIN_MIN_PRICE && bestAsk <= PredictionExecutor.LAST5MIN_MAX_PRICE) {
                        const conditionId = e.marketResult.market.conditionId;
                        if (this.logger.hasAlreadyBought(conditionId)) {
                            this.priceMonitorQueue.splice(i, 1);
                            return;
                        }
                        const allEntries = this.logger.getAllEntries();
                        const monitorStartTimeMs = e.monitorStartTime.getTime();
                        const hasExecutedForMarket = allEntries.some(ev =>
                            ev.conditionId === conditionId && ev.status === 'executed' &&
                            new Date(ev.timestamp).getTime() >= monitorStartTimeMs - 60000
                        );
                        if (hasExecutedForMarket) {
                            this.priceMonitorQueue.splice(i, 1);
                            return;
                        }
                        idx = i;
                        entry = e;
                        entry.triggerPrice = bestAsk;
                        if (asksSnapshot && asksSnapshot.length > 0) entry.asksSnapshot = asksSnapshot;
                        const symbol = e.prediction.symbol.replace('/USDT', '');
                        const periodStr = this._fmtPeriod(e.targetPeriodEndTs);
                        console.log(`\n✅ ${symbol} 最后5分钟市价成交(WS) [${periodStr}]: $${e.initialPrice.toFixed(4)} (进入监控) → $${bestAsk.toFixed(4)} (当前) 在 [${PredictionExecutor.LAST5MIN_MIN_PRICE.toFixed(2)}, ${PredictionExecutor.LAST5MIN_MAX_PRICE.toFixed(2)}]，立即下单`);
                        break; // 交给下方 splice + executeOnePriceMonitorEntry
                    }
                    return; // 最后5分钟内价格不在 [0.3, 0.65]，不移除，等下一 tick 再试
                }
                if (bestAsk > this.MAX_PRICE_THRESHOLD || bestAsk < this.env.MIN_TOKEN_PRICE) return;
                if (e.isProcessing) return;
                const conditionId = e.marketResult.market.conditionId;
                if (this.logger.hasAlreadyBought(conditionId)) {
                    this.priceMonitorQueue.splice(i, 1);
                    return;
                }
                const allEntries = this.logger.getAllEntries();
                const monitorStartTimeMs = e.monitorStartTime.getTime();
                const hasTradeForMarket = allEntries.some(ev =>
                    ev.conditionId === conditionId &&
                    (ev.status === 'executed' || ev.status === 'failed' || ev.status === 'pending') &&
                    new Date(ev.timestamp).getTime() >= monitorStartTimeMs - 60000
                );
                if (hasTradeForMarket) {
                    this.priceMonitorQueue.splice(i, 1);
                    return;
                }
                idx = i;
                entry = e;
                break;
            }
        }
        if (idx < 0 || !entry) return;
        const executionDecision = this.applyExecutionV2MonitorDecision(entry, bestAsk, now);
        if (executionDecision.shouldCancel) {
            this.priceMonitorQueue.splice(idx, 1);
            return;
        }
        const legacyFinalAggressive = now >= entry.monitorEndTime
            && bestAsk >= PredictionExecutor.LAST5MIN_MIN_PRICE
            && bestAsk <= PredictionExecutor.LAST5MIN_MAX_PRICE;
        if (!executionDecision.shouldExecute && !legacyFinalAggressive) return;
        entry.isProcessing = true;
        entry.triggerPrice = entry.triggerPrice ?? bestAsk;
        entry.currentMaxPrice = legacyFinalAggressive
            ? Math.max(executionDecision.effectiveMaxPrice, PredictionExecutor.LAST5MIN_MAX_PRICE)
            : executionDecision.effectiveMaxPrice;
        if (asksSnapshot && asksSnapshot.length > 0) entry.asksSnapshot = asksSnapshot;
        this.priceMonitorQueue.splice(idx, 1);
        const symbol = entry.prediction.symbol.replace('/USDT', '');
        const periodStr = this._fmtPeriod(entry.targetPeriodEndTs);
        if (now < entry.monitorEndTime) {
            console.log(`\n✅ ${symbol} 价格监控触发(WS) [${periodStr}]: $${entry.initialPrice.toFixed(4)} (进入监控) → $${bestAsk.toFixed(4)} (当前) ≤ $${entry.currentMaxPrice?.toFixed(2)}，立即下单`);
        }
        await this.executeOnePriceMonitorEntry(entry);
    }
    
    /**
     * WebSocket 收到 best_ask 更新时调用：若该 token 在低价监控队列且 best_ask 在 [MIN, MAX] 内，立即触发下单
     * @param asksSnapshot 可选，来自 WS book 消息时传入，下单时跳过 REST getOrderBook（提速）
     */
    async tryTriggerLowPriceMonitorByTokenId(assetId: string, bestAsk: number, asksSnapshot?: Array<{ price: string; size: string }>): Promise<void> {
        const now = new Date();
        let idx = -1;
        let entry: PriceMonitorEntry | null = null;
        for (let i = 0; i < this.lowPriceMonitorQueue.length; i++) {
            const e = this.lowPriceMonitorQueue[i];
            if (!e.marketResult.market || !e.prediction.direction) continue;
            const token = getTokenForDirection(e.marketResult.market, e.prediction.direction);
            if (token?.tokenId === assetId) {
                const periodEndTime = new Date((e.targetPeriodEndTs + this.periodSeconds) * 1000);
                if (now >= periodEndTime) {
                    this.lowPriceMonitorQueue.splice(i, 1);
                    return;
                }
                const executionDecision = this.applyExecutionV2MonitorDecision(e, bestAsk, now);
                if (executionDecision.shouldCancel) {
                    this.lowPriceMonitorQueue.splice(i, 1);
                    return;
                }
                const legacyFinalAggressive = now >= e.monitorEndTime
                    && bestAsk >= PredictionExecutor.LAST5MIN_MIN_PRICE
                    && bestAsk <= PredictionExecutor.LAST5MIN_MAX_PRICE;
                if (now >= e.monitorEndTime) {
                    if (executionDecision.shouldExecute || legacyFinalAggressive) {
                        const conditionId = e.marketResult.market.conditionId;
                        if (this.logger.hasAlreadyBought(conditionId)) {
                            this.lowPriceMonitorQueue.splice(i, 1);
                            return;
                        }
                        const allEntries = this.logger.getAllEntries();
                        const monitorStartTimeMs = e.monitorStartTime.getTime();
                        const hasExecutedForMarket = allEntries.some(ev =>
                            ev.conditionId === conditionId && ev.status === 'executed' &&
                            new Date(ev.timestamp).getTime() >= monitorStartTimeMs - 60000
                        );
                        if (hasExecutedForMarket) {
                            this.lowPriceMonitorQueue.splice(i, 1);
                            return;
                        }
                        idx = i;
                        entry = e;
                        entry.triggerPrice = bestAsk;
                        entry.currentMaxPrice = legacyFinalAggressive
                            ? Math.max(executionDecision.effectiveMaxPrice, PredictionExecutor.LAST5MIN_MAX_PRICE)
                            : executionDecision.effectiveMaxPrice;
                        if (asksSnapshot && asksSnapshot.length > 0) entry.asksSnapshot = asksSnapshot;
                        const symbol = e.prediction.symbol.replace('/USDT', '');
                        const periodStr = this._fmtPeriod(e.targetPeriodEndTs);
                        console.log(`\n✅ ${symbol} 最后5分钟市价成交(低价/WS) [${periodStr}]: $${e.initialPrice.toFixed(4)} (进入监控) → $${bestAsk.toFixed(4)} (当前) 在 [${PredictionExecutor.LAST5MIN_MIN_PRICE.toFixed(2)}, ${PredictionExecutor.LAST5MIN_MAX_PRICE.toFixed(2)}]，立即下单`);
                        break;
                    }
                    return;
                }
                if (!executionDecision.shouldExecute) return;
                if (e.isProcessing) return;
                const conditionId = e.marketResult.market.conditionId;
                if (this.logger.hasAlreadyBought(conditionId)) {
                    this.lowPriceMonitorQueue.splice(i, 1);
                    return;
                }
                const allEntries = this.logger.getAllEntries();
                const monitorStartTimeMs = e.monitorStartTime.getTime();
                const hasTradeForMarket = allEntries.some(ev =>
                    ev.conditionId === conditionId &&
                    (ev.status === 'executed' || ev.status === 'failed' || ev.status === 'pending') &&
                    new Date(ev.timestamp).getTime() >= monitorStartTimeMs - 60000
                );
                if (hasTradeForMarket) {
                    this.lowPriceMonitorQueue.splice(i, 1);
                    return;
                }
                idx = i;
                entry = e;
                entry.currentMaxPrice = executionDecision.effectiveMaxPrice;
                break;
            }
        }
        if (idx < 0 || !entry) return;
        entry.isProcessing = true;
        entry.triggerPrice = entry.triggerPrice ?? bestAsk;
        if (asksSnapshot && asksSnapshot.length > 0) entry.asksSnapshot = asksSnapshot;
        this.lowPriceMonitorQueue.splice(idx, 1);
        const symbol = entry.prediction.symbol.replace('/USDT', '');
        const periodStr = this._fmtPeriod(entry.targetPeriodEndTs);
        if (now < entry.monitorEndTime) {
            console.log(`\n✅ ${symbol} 低价监控触发(WS) [${periodStr}]: $${entry.initialPrice.toFixed(4)} (进入监控) → $${bestAsk.toFixed(4)} (当前) ≥ $${this.env.MIN_TOKEN_PRICE.toFixed(2)} (阈值)，立即下单`);
        }
        await this.executeOneLowPriceMonitorEntry(entry);
    }

    /** 执行单条高价监控条目（重试 3 次） */
    private async executeOnePriceMonitorEntry(entry: PriceMonitorEntry): Promise<void> {
        const symbol = entry.prediction.symbol.replace('/USDT', '');
        let lastError: string | null = null;
        let triggerPrice: number | null = entry.triggerPrice ?? null;
        if (triggerPrice === null && entry.marketResult.market && entry.prediction.direction) {
            try {
                const token = getTokenForDirection(entry.marketResult.market, entry.prediction.direction);
                if (token?.tokenId) triggerPrice = await getTokenPrice(token.tokenId, 'BUY');
            } catch { /* ignore */ }
        }
        for (let retry = 0; retry < 3; retry++) {
            try {
                const result = await this.executePrediction(entry.prediction, entry.marketResult, {
                    targetPeriodEndTs: entry.targetPeriodEndTs,
                    triggerPrice: triggerPrice ?? undefined,
                    asksSnapshot: entry.asksSnapshot,
                    preSignedOrder: entry.preSignedOrder,
                    maxPrice: entry.currentMaxPrice,
                });
                if (result.success) return;
                lastError = result.error || '未知错误';
            } catch (error) {
                lastError = error instanceof Error ? error.message : String(error);
            }
            if (retry < 2) await new Promise((r) => setTimeout(r, 1000));
        }
        const periodStr = this._fmtPeriod(entry.targetPeriodEndTs);
        console.log(`\n${'═'.repeat(60)}\n  ❌ ${symbol} 价格监控下单失败(WS) [${periodStr}]\n  错误: ${lastError || '未知'}\n${'═'.repeat(60)}\n`);
    }
    
    /** 执行单条低价监控条目（重试 3 次；有预签单时触发即 POST） */
    private async executeOneLowPriceMonitorEntry(entry: PriceMonitorEntry): Promise<void> {
        const symbol = entry.prediction.symbol.replace('/USDT', '');
        let lastError: string | null = null;
        let triggerPrice: number | null = entry.triggerPrice ?? null;
        if (triggerPrice === null && entry.marketResult.market && entry.prediction.direction) {
            try {
                const token = getTokenForDirection(entry.marketResult.market, entry.prediction.direction);
                if (token?.tokenId) triggerPrice = await getTokenPrice(token.tokenId, 'BUY');
            } catch { /* ignore */ }
        }
        for (let retry = 0; retry < 3; retry++) {
            try {
                const result = await this.executePrediction(entry.prediction, entry.marketResult, {
                    targetPeriodEndTs: entry.targetPeriodEndTs,
                    triggerPrice: triggerPrice ?? undefined,
                    asksSnapshot: entry.asksSnapshot,
                    preSignedOrder: entry.preSignedOrder,
                    maxPrice: entry.currentMaxPrice,
                });
                if (result.success) return;
                lastError = result.error || '未知错误';
            } catch (error) {
                lastError = error instanceof Error ? error.message : String(error);
            }
            if (retry < 2) await new Promise((r) => setTimeout(r, 1000));
        }
        const periodStr = this._fmtPeriod(entry.targetPeriodEndTs);
        console.log(`\n${'═'.repeat(60)}\n  ❌ ${symbol} 低价监控下单失败(WS) [${periodStr}]\n  错误: ${lastError || '未知'}\n${'═'.repeat(60)}\n`);
    }
    
    /**
     * WebSocket 收到 book 消息（含 asksSnapshot）时调用：若该 token 在流动性监控队列且深度足够，立即触发补单
     */
    async tryTriggerLiquidityMonitorByTokenId(
        assetId: string,
        bestAsk: number,
        asksSnapshot: Array<{ price: string; size: string }>
    ): Promise<void> {
        const now = new Date();
        let idx = -1;
        let entry: LiquidityMonitorEntry | null = null;
        let availableLiquidity = 0;
        for (let i = 0; i < this.liquidityMonitorQueue.length; i++) {
            const e = this.liquidityMonitorQueue[i];
            if (!e.marketResult.market || !e.prediction.direction) continue;
            const token = getTokenForDirection(e.marketResult.market, e.prediction.direction);
            if (token?.tokenId !== assetId) continue;
            if (now > e.monitorEndTime) {
                this.liquidityMonitorQueue.splice(i, 1);
                return;
            }
            const executionDecision = this.applyExecutionV2MonitorDecision(e, bestAsk, now);
            if (executionDecision.shouldCancel) {
                this.liquidityMonitorQueue.splice(i, 1);
                return;
            }
            if (!executionDecision.shouldExecute) return;
            const emin = e.minPrice ?? 0.25;
            const emax = executionDecision.effectiveMaxPrice;
            e.maxPrice = emax;
            availableLiquidity = 0;
            for (const ask of asksSnapshot) {
                const askPrice = parseFloat(ask.price);
                if (askPrice < emin || askPrice > emax) continue;
                const askSize = parseFloat(ask.size || '0');
                availableLiquidity += askSize * askPrice;
            }
            if (availableLiquidity <= 0) return;
            idx = i;
            entry = e;
            break;
        }
        if (idx < 0 || !entry) return;
        this.liquidityMonitorQueue.splice(idx, 1);
        const symbol = entry.prediction.symbol.replace('/USDT', '');
        const periodStr = this._fmtPeriod(entry.targetPeriodEndTs);
        console.log(`\n✅ ${symbol} 流动性监控触发(WS) [${periodStr}]: 价格 $${bestAsk.toFixed(4)}, 可用 $${availableLiquidity.toFixed(2)}，允许上界 $${entry.maxPrice?.toFixed(2)}，立即补单`);
        await this.executeOneLiquidityMonitorEntry(entry, bestAsk, asksSnapshot);
    }

    /** 执行单条流动性监控条目（复用 WS 的 best_ask 与 asksSnapshot，不再拉 API） */
    private async executeOneLiquidityMonitorEntry(
        entry: LiquidityMonitorEntry,
        triggerPrice: number,
        asksSnapshot: Array<{ price: string; size: string }>
    ): Promise<void> {
        const symbol = entry.prediction.symbol.replace('/USDT', '');
        if (entry.marketResult.market) {
            const conditionId = entry.marketResult.market.conditionId;
            const allEntries = this.logger.getAllEntries();
            const hasCompleteTrade = allEntries.some(e =>
                e.conditionId === conditionId &&
                e.status === 'executed' &&
                e.id !== entry.logEntryId &&
                Math.abs((e.amount || 0) - entry.originalAmount) < 0.01
            );
            if (hasCompleteTrade) {
                console.log(`\n✅ ${symbol} 流动性监控(WS)：已有完整交易，跳过`);
                return;
            }
        }
        let currentBoughtAmount = 0;
        if (this.env.TRADING_MODE === TradingMode.LIVE && this.env.PROXY_WALLET && entry.marketResult.market?.conditionId && entry.prediction.direction) {
            try {
                const token = getTokenForDirection(entry.marketResult.market, entry.prediction.direction);
                if (token?.tokenId) {
                    currentBoughtAmount = await getPositionValue(this.env.PROXY_WALLET, entry.marketResult.market.conditionId, token.tokenId);
                }
            } catch {
                const allEntries = this.logger.getAllEntries();
                currentBoughtAmount = allEntries
                    .filter(e => e.conditionId === entry.marketResult.market?.conditionId && e.direction === entry.prediction.direction && e.status === 'executed')
                    .reduce((sum, e) => sum + (e.amount || 0), 0);
            }
        } else {
            const allEntries = this.logger.getAllEntries();
            currentBoughtAmount = allEntries
                .filter(e => e.conditionId === entry.marketResult.market?.conditionId && e.direction === entry.prediction.direction && e.status === 'executed')
                .reduce((sum, e) => sum + (e.amount || 0), 0);
        }
        if (currentBoughtAmount >= entry.originalAmount) {
            console.log(`\n✅ ${symbol} 流动性监控(WS)：本单累计已买够，跳过`);
            return;
        }
        const stillNeeded = entry.originalAmount - currentBoughtAmount;
        const buyAmount = Math.min(entry.remainingAmount, stillNeeded);
        if (buyAmount < MIN_ORDER_SIZE_USD) {
            console.log(`\n✅ ${symbol} 流动性监控(WS)：剩余金额不足最小订单，跳过`);
            return;
        }
        if (!entry.prediction.direction) return;
        const token = getTokenForDirection(entry.marketResult.market!, entry.prediction.direction);
        if (!token) return;
        const continueLogEntry = this.logger.addEntry({
            symbol: entry.prediction.symbol.replace('/USDT', ''),
            direction: entry.prediction.direction,
            confidence: entry.prediction.confidence,
            amount: buyAmount,
            mode: this.env.TRADING_MODE,
            status: 'pending',
            marketTitle: entry.marketResult.market?.title || '',
            conditionId: entry.marketResult.market?.conditionId,
            tokenId: token.tokenId,
            marketSlug: entry.marketResult.market?.slug,
        });
        const opts = {
            targetPeriodEndTs: entry.targetPeriodEndTs,
            minPrice: entry.minPrice,
            maxPrice: entry.maxPrice,
            triggerPrice,
            asksSnapshot,
            executionProfile: entry.executionProfile,
        };
        let actualBoughtAmount = 0;
        if (this.env.TRADING_MODE === TradingMode.LIVE) {
            const result = await this.executeLiveTrade(entry.marketResult, token, buyAmount, continueLogEntry, entry.prediction, opts);
            actualBoughtAmount = result.success ? roundUsd(buyAmount - (result.remainingAmount || 0)) : 0;
            if (!result.success && !result.blockedByCooldown) {
                this.logger.updateEntry(continueLogEntry.id, { status: 'failed', error: '流动性不足或下单失败' });
            }
        } else {
            const simResult = await this.simulateTrade(entry.marketResult, token, buyAmount, continueLogEntry, entry.prediction, opts);
            actualBoughtAmount = roundUsd(simResult.actualAmount);
            console.log(`\nℹ️  ${symbol} 流动性监控(WS)：模拟补单 $${actualBoughtAmount.toFixed(2)}`);
        }
        if (actualBoughtAmount > 0) {
            const newTotal = roundUsd(currentBoughtAmount + actualBoughtAmount);
            entry.totalBoughtAmount = newTotal;
            entry.remainingAmount = roundUsd(Math.max(0, entry.originalAmount - newTotal));
            if (entry.remainingAmount >= MIN_ORDER_SIZE_USD) {
                this.liquidityMonitorQueue.push(entry);
            }
        }
    }

    /**
     * 检查低价监控队列，如果价格回升到 MIN_TOKEN_PRICE 以上则触发买入
     * 静默检查，只在关键事件时显示日志（触发下单、超时、API重连）
     */
    async checkLowPriceMonitorQueue(): Promise<void> {
        if (this.lowPriceMonitorQueue.length === 0) return;
        
        const now = new Date();
        const entriesToRemove: number[] = [];
        const entriesToExecute: PriceMonitorEntry[] = [];
        let apiErrorCount = 0;
        
        for (let i = 0; i < this.lowPriceMonitorQueue.length; i++) {
            const entry = this.lowPriceMonitorQueue[i];
            const periodEndTime = new Date((entry.targetPeriodEndTs + this.periodSeconds) * 1000);

            // 已过周期结束：静默移除
            if (now >= periodEndTime) {
                entriesToRemove.push(i);
                continue;
            }

            // 检查是否已经在处理中
            if (entry.isProcessing) {
                continue;
            }

            // 检查是否已下过单
            if (entry.marketResult.market) {
                const conditionId = entry.marketResult.market.conditionId;
                if (this.logger.hasAlreadyBought(conditionId)) {
                    entriesToRemove.push(i);
                    continue;
                }
                // 检查是否已有该市场的交易记录
                const allEntries = this.logger.getAllEntries();
                const monitorStartTimeMs = entry.monitorStartTime.getTime();
                const hasTradeForMarket = allEntries.some(e => 
                    e.conditionId === conditionId && 
                    (e.status === 'executed' || e.status === 'failed' || e.status === 'pending') && 
                    new Date(e.timestamp).getTime() >= monitorStartTimeMs - 60000
                );
                if (hasTradeForMarket) {
                    entriesToRemove.push(i);
                    continue;
                }
            }
            
            // 获取当前价格
            if (!entry.prediction.direction) {
                entriesToRemove.push(i);
                continue;
            }
            const token = getTokenForDirection(entry.marketResult.market!, entry.prediction.direction);
            if (!token || !token.tokenId) {
                // token 不存在是数据问题，移除条目避免永久占用队列
                entriesToRemove.push(i);
                continue;
            }
            
            let currentPrice: number | null = null;
            let apiError = false;
            
            try {
                currentPrice = await getTokenPrice(token.tokenId, 'BUY');
                if (currentPrice === null) {
                    apiError = true;
                    apiErrorCount++;
                }
            } catch (error) {
                apiError = true;
                apiErrorCount++;
            }
            
            // API错误时，跳过本次检查，等待下次重试（临时性错误不移除条目）
            if (apiError) {
                continue;
            }

            if (currentPrice === null) continue;
            const executionDecision = this.applyExecutionV2MonitorDecision(entry, currentPrice, now);
            if (executionDecision.shouldCancel) {
                entriesToRemove.push(i);
                continue;
            }
            const legacyFinalAggressive = now >= entry.monitorEndTime
                && currentPrice >= PredictionExecutor.LAST5MIN_MIN_PRICE
                && currentPrice <= PredictionExecutor.LAST5MIN_MAX_PRICE;
            const effectiveExecutionMaxPrice = legacyFinalAggressive
                ? Math.max(executionDecision.effectiveMaxPrice, PredictionExecutor.LAST5MIN_MAX_PRICE)
                : executionDecision.effectiveMaxPrice;

            // 最后5分钟市价成交：已过可下单截止但未过周期结束
            if (now >= entry.monitorEndTime) {
                if (executionDecision.shouldExecute || legacyFinalAggressive) {
                    if (entry.marketResult.market) {
                        const conditionId = entry.marketResult.market.conditionId;
                        if (this.logger.hasAlreadyBought(conditionId)) {
                            entriesToRemove.push(i);
                            continue;
                        }
                        const allEntries = this.logger.getAllEntries();
                        const monitorStartTimeMs = entry.monitorStartTime.getTime();
                        const hasExecutedForMarket = allEntries.some(e =>
                            e.conditionId === conditionId && e.status === 'executed' &&
                            new Date(e.timestamp).getTime() >= monitorStartTimeMs - 60000
                        );
                        if (hasExecutedForMarket) {
                            entriesToRemove.push(i);
                            continue;
                        }
                    }
                    entry.isProcessing = true;
                    entry.triggerPrice = currentPrice;
                    entry.currentMaxPrice = effectiveExecutionMaxPrice;
                    entriesToRemove.push(i);
                    entriesToExecute.push(entry);
                    const symbol = entry.prediction.symbol.replace('/USDT', '');
                    const periodStr = this._fmtPeriod(entry.targetPeriodEndTs);
                    console.log(`\n✅ ${symbol} 最后5分钟市价成交(低价) [${periodStr}]: $${entry.initialPrice.toFixed(4)} (进入监控) → $${currentPrice.toFixed(4)} (当前) 在 [${PredictionExecutor.LAST5MIN_MIN_PRICE.toFixed(2)}, ${PredictionExecutor.LAST5MIN_MAX_PRICE.toFixed(2)}]，立即下单`);
                }
                continue;
            }

            // 检查价格是否回升到允许范围 / 追价上界
            if (executionDecision.shouldExecute) {
                    // 检查是否已经在处理中
                    if (entry.isProcessing) {
                        continue;
                    }
                    
                    // 再次检查是否已有交易记录
                    if (entry.marketResult.market) {
                        const conditionId = entry.marketResult.market.conditionId;
                        const allEntries = this.logger.getAllEntries();
                        const monitorStartTimeMs = entry.monitorStartTime.getTime();
                        const hasTradeForMarket = allEntries.some(e => 
                            e.conditionId === conditionId && 
                            (e.status === 'executed' || e.status === 'failed' || e.status === 'pending') && 
                            new Date(e.timestamp).getTime() >= monitorStartTimeMs - 60000
                        );
                        if (hasTradeForMarket) {
                            entriesToRemove.push(i);
                            continue;
                        }
                    }
                    
                    // 检查是否已下过单
                    if (entry.marketResult.market) {
                        const conditionId = entry.marketResult.market.conditionId;
                        if (this.logger.hasAlreadyBought(conditionId)) {
                            entriesToRemove.push(i);
                            continue;
                        }
                    }
                    
                    // 标记为正在处理
                    entry.isProcessing = true;
                    entry.triggerPrice = currentPrice;
                    entry.currentMaxPrice = effectiveExecutionMaxPrice;
                    entriesToRemove.push(i);
                    entriesToExecute.push(entry);
                    const symbol = entry.prediction.symbol.replace('/USDT', '');
                    const periodStr = this._fmtPeriod(entry.targetPeriodEndTs);
                    console.log(`\n✅ ${symbol} 低价监控触发 [${periodStr}]: $${entry.initialPrice.toFixed(4)} (进入监控) → $${currentPrice.toFixed(4)} (当前)，允许价格上界 $${effectiveExecutionMaxPrice.toFixed(2)}，立即下单`);
            }
        }
        
        if (apiErrorCount > 0) {
            console.log(`\n🔄 低价监控API错误，${apiErrorCount} 个请求失败，将重试`);
        }
        
        // 移除已处理或过期的条目
        for (let i = entriesToRemove.length - 1; i >= 0; i--) {
            this.lowPriceMonitorQueue.splice(entriesToRemove[i], 1);
        }
        
        // 执行满足条件的订单
        for (const entry of entriesToExecute) {
            const symbol = entry.prediction.symbol.replace('/USDT', '');
            let lastError: string | null = null;
            let success = false;
            let triggerPrice: number | null = entry.triggerPrice ?? null;
            
            if (triggerPrice === null) {
                try {
                    if (entry.marketResult.market && entry.prediction.direction) {
                        const token = getTokenForDirection(entry.marketResult.market, entry.prediction.direction);
                        if (token && token.tokenId) {
                            triggerPrice = await getTokenPrice(token.tokenId, 'BUY');
                        }
                    }
                } catch (error) {
                    // 忽略
                }
            }
            
            // 重试3次（有预签单时触发即 POST）
            for (let retry = 0; retry < 3; retry++) {
                try {
                    const result = await this.executePrediction(entry.prediction, entry.marketResult, { 
                        targetPeriodEndTs: entry.targetPeriodEndTs,
                        triggerPrice: triggerPrice ?? undefined,
                        asksSnapshot: entry.asksSnapshot,
                        preSignedOrder: entry.preSignedOrder,
                        maxPrice: entry.currentMaxPrice,
                    });
                    
                    if (result.success) {
                        success = true;
                        break;
                    } else {
                        lastError = result.error || '未知错误';
                        if (retry < 2) {
                            await new Promise(resolve => setTimeout(resolve, 1000));
                        }
                    }
                } catch (error) {
                    lastError = error instanceof Error ? error.message : String(error);
                    if (retry < 2) {
                        await new Promise(resolve => setTimeout(resolve, 1000));
                    }
                }
            }
            
            if (!success) {
                const periodStr = this._fmtPeriod(entry.targetPeriodEndTs);
                console.log(`\n${'═'.repeat(60)}`);
                console.log(`  ❌ ${symbol} 低价监控下单失败报告 [${periodStr}]`);
                console.log(`${'═'.repeat(60)}`);
                console.log(`  周期:       ${periodStr}`);
                console.log(`  初始价格:   $${entry.initialPrice.toFixed(4)}`);
                console.log(`  触发价格:   $${triggerPrice !== null ? triggerPrice.toFixed(4) : 'N/A'}`);
                console.log(`  重试次数:   3 次`);
                console.log(`  最终状态:   失败`);
                console.log(`  错误原因:   ${lastError || '未知错误'}`);
                console.log(`  市场:       ${entry.marketResult.market?.title || 'N/A'}`);
                console.log(`  条件ID:     ${entry.marketResult.market?.conditionId?.slice(0, 20) || 'N/A'}...`);
                console.log(`${'═'.repeat(60)}\n`);
            }
        }
    }
    
    /**
     * 检查流动性监控队列，如果订单簿有足够流动性则继续买入剩余部分
     * 静默检查，只在关键事件时显示日志（触发买入、超时、API重连）
     */
    async checkLiquidityMonitorQueue(): Promise<void> {
        if (this.liquidityMonitorQueue.length === 0) return;
        
        const now = new Date();
        const entriesToRemove: number[] = [];
        const entriesToExecute: LiquidityMonitorEntry[] = [];
        let apiErrorCount = 0;
        
        for (let i = 0; i < this.liquidityMonitorQueue.length; i++) {
            const entry = this.liquidityMonitorQueue[i];
            
            // 检查是否过了监控窗口
            if (now > entry.monitorEndTime) {
                entriesToRemove.push(i);
                continue;
            }
            
            // 检查是否已经买够了（通过已买入金额判断，而不是简单的 hasAlreadyBought）
            // 注意：hasAlreadyBought 只检查是否有 executed 记录，不适合多次买入场景
            // 我们应该检查已买入金额是否 >= 目标金额
            // 这个检查会在后面执行，这里先跳过 hasAlreadyBought 检查
            
            // 获取当前价格和订单簿
            if (!entry.prediction.direction) {
                entriesToRemove.push(i);
                continue;
            }
            
            const token = getTokenForDirection(entry.marketResult.market!, entry.prediction.direction);
            if (!token || !token.tokenId) {
                continue;
            }
            
            let currentPrice: number | null = null;
            let orderBook: any = null;
            let apiError = false;
            
            try {
                // 获取当前价格
                currentPrice = await getTokenPrice(token.tokenId, 'BUY');
                if (currentPrice === null) {
                    apiError = true;
                    apiErrorCount++;
                }
                
                // 获取订单簿 - 用静默版本避免 SDK 打印 404 日志
                if (!apiError) {
                    orderBook = await getOrderBook(token.tokenId);
                    if (!orderBook || !orderBook.asks || orderBook.asks.length === 0) {
                        continue; // 订单簿为空或不存在，等待下次检查
                    }
                }
            } catch (error) {
                apiError = true;
                apiErrorCount++;
            }
            
            // API错误时，跳过本次检查，等待下次重试
            if (apiError || !orderBook) {
                continue;
            }
            
            if (currentPrice === null) {
                continue;
            }
            const executionDecision = this.applyExecutionV2MonitorDecision(entry, currentPrice, now);
            if (executionDecision.shouldCancel) {
                entriesToRemove.push(i);
                continue;
            }
            // 检查价格是否在允许范围内
            const minPrice = entry.minPrice || 0.25;
            const maxPrice = executionDecision.effectiveMaxPrice;
            entry.maxPrice = maxPrice;
            if (!executionDecision.shouldExecute) {
                continue; // 价格不在允许范围，等待下次检查
            }
            
            // 计算订单簿可用流动性（只计算价格在范围内的部分）
            // 根据 CLOB API 文档，orderbook.asks 中的 size 是代币数量
            // 需要转换为 USDC 金额：size * price
            let availableLiquidity = 0;
            try {
                for (const ask of orderBook.asks) {
                    const askPrice = parseFloat(ask.price);
                    if (askPrice < minPrice || askPrice > maxPrice) continue; // 只计算价格范围内的
                    const askSize = parseFloat((ask as any).size || (ask as any).amount || '0');
                    availableLiquidity += askSize * askPrice; // 代币数量 * 价格 = USDC 金额
                }
            } catch (error) {
                continue; // 无法计算流动性，等待下次检查
            }
            
            // 检查是否有足够的流动性买入剩余部分（只要有流动性就买入，可以部分买入）
            const requiredAmount = entry.remainingAmount;
            if (availableLiquidity > 0) {
                // 有流动性时只加入执行列表，不在此处从队列移除；执行后再根据是否买够决定是否移除，避免部分成交后漏买
                entriesToExecute.push(entry);
                const symbol = entry.prediction.symbol.replace('/USDT', '');
                const periodStr = this._fmtPeriod(entry.targetPeriodEndTs);
                const buyAmount = Math.min(requiredAmount, availableLiquidity * 0.95); // 95% 流动性，留5%缓冲
                console.log(`\n✅ ${symbol} 流动性监控触发 [${periodStr}]: 价格 $${currentPrice.toFixed(4)}, 可用 $${availableLiquidity.toFixed(2)}, 将买入 $${buyAmount.toFixed(2)}`);
            }
        }
        
        // 如果API错误，显示重连日志（不限次数）
        if (apiErrorCount > 0) {
            console.log(`\n🔄 流动性监控API错误，${apiErrorCount} 个请求失败，将重试`);
        }
        
        // 仅移除已过期或无效的条目（从后往前删除，避免索引错乱）；有流动性的条目留在队列中，执行后再按是否买够移除
        for (let i = entriesToRemove.length - 1; i >= 0; i--) {
            this.liquidityMonitorQueue.splice(entriesToRemove[i], 1);
        }
        
        // 执行继续买入（支持多次买入，直到金额够或时间过了）
        const entriesCompletedOrSkipped: LiquidityMonitorEntry[] = []; // 本轮因已买够/已有完整交易/剩余不足而跳过，需从队列移除
        for (const entry of entriesToExecute) {
            const symbol = entry.prediction.symbol.replace('/USDT', '');
            
            // 检查是否已下过单（避免重复下单）
            // 注意：流动性监控是继续买入剩余部分，所以需要检查是否已经买够了，而不是简单的 hasAlreadyBought
            // 但如果已经有完整的 executed 记录（不是部分成交），说明已经处理过了，应该跳过
            if (entry.marketResult.market) {
                const conditionId = entry.marketResult.market.conditionId;
                // 检查是否有完整的 executed 记录（不是从流动性监控队列来的）
                const allEntries = this.logger.getAllEntries();
                const hasCompleteTrade = allEntries.some(e => 
                    e.conditionId === conditionId && 
                    e.status === 'executed' &&
                    e.id !== entry.logEntryId && // 排除原始日志条目（可能是部分成交）
                    Math.abs((e.amount || 0) - entry.originalAmount) < 0.01 // 金额接近原始金额，说明是完整交易
                );
                if (hasCompleteTrade) {
                    console.log(`\n✅ ${symbol} 流动性监控：已有完整交易，停止监控`);
                    entriesCompletedOrSkipped.push(entry);
                    continue;
                }
            }
            
            // 检查已买入金额（优先使用 Data API 查询实际持仓，如果失败则使用交易日志）
            // 注意：只计算与当前 entry 相关的交易（同一 conditionId 和 direction）
            let currentBoughtAmount = 0;
            if (this.env.TRADING_MODE === TradingMode.LIVE && this.env.PROXY_WALLET && entry.marketResult.market?.conditionId && entry.prediction.direction) {
                try {
                    // 使用 Data API 查询实际持仓
                    const token = getTokenForDirection(entry.marketResult.market, entry.prediction.direction);
                    if (token?.tokenId) {
                        const positionValue = await getPositionValue(
                            this.env.PROXY_WALLET,
                            entry.marketResult.market.conditionId,
                            token.tokenId
                        );
                        currentBoughtAmount = positionValue;
                    }
                } catch (error) {
                    // API 查询失败，回退到交易日志
                    // 只计算与当前 entry 相关的交易（同一 conditionId 和 direction，且是 executed 状态）
                    const allEntries = this.logger.getAllEntries();
                    const sameMarketEntries = allEntries.filter(e => 
                        e.conditionId === entry.marketResult.market?.conditionId && 
                        e.direction === entry.prediction.direction &&
                        e.status === 'executed'
                    );
                    currentBoughtAmount = sameMarketEntries.reduce((sum, e) => sum + (e.amount || 0), 0);
                }
            } else {
                // 模拟模式或回测模式，使用交易日志
                // 只计算与当前 entry 相关的交易（同一 conditionId 和 direction，且是 executed 状态）
                const allEntries = this.logger.getAllEntries();
                const sameMarketEntries = allEntries.filter(e => 
                    e.conditionId === entry.marketResult.market?.conditionId && 
                    e.direction === entry.prediction.direction &&
                    e.status === 'executed'
                );
                currentBoughtAmount = sameMarketEntries.reduce((sum, e) => sum + (e.amount || 0), 0);
            }
            
            // 检查是否已经买够了（避免买多）- 这是正确的检查方式
            if (currentBoughtAmount >= entry.originalAmount) {
                // 本单累计可能大于此条目标（补单部分成交时会新增一条 originalAmount=当次补单金额 的条目）
                const msg = currentBoughtAmount === entry.originalAmount
                    ? `\n✅ ${symbol} 流动性监控：本单累计已买入 $${currentBoughtAmount.toFixed(2)}，达到目标，停止买入`
                    : `\n✅ ${symbol} 流动性监控：本单累计已买入 $${currentBoughtAmount.toFixed(2)}，此条目标 $${entry.originalAmount.toFixed(2)} 已达成，停止买入`;
                console.log(msg);
                entriesCompletedOrSkipped.push(entry);
                continue;
            }
            
            // 计算本次需要买入的金额（不超过剩余金额）
            const stillNeeded = entry.originalAmount - currentBoughtAmount;
            const buyAmount = Math.min(entry.remainingAmount, stillNeeded);
            
            if (buyAmount < MIN_ORDER_SIZE_USD) {
                console.log(`\n✅ ${symbol} 流动性监控：剩余金额 $${buyAmount.toFixed(2)} 小于最小订单金额，停止买入`);
                entriesCompletedOrSkipped.push(entry);
                continue;
            }
            
            // 简化逻辑：不重试，失败了等下次监控周期再尝试
            // 防超买：只用 currentBoughtAmount 判断，不依赖 hasAlreadyBought
            
            try {
                // 获取 token 信息（LIVE 和模拟都需要）
                if (!entry.prediction.direction) {
                    console.log(`\n⚠️  ${symbol} 流动性监控：预测方向为空，跳过并移出队列`);
                    entriesCompletedOrSkipped.push(entry);
                    continue;
                }
                const token = getTokenForDirection(entry.marketResult.market!, entry.prediction.direction);
                if (!token) {
                    console.log(`\n⚠️  ${symbol} 流动性监控：无法确定目标代币，跳过并移出队列`);
                    entriesCompletedOrSkipped.push(entry);
                    continue;
                }
                
                // 创建新的交易日志条目用于本次买入
                const continueLogEntry = this.logger.addEntry({
                    symbol: entry.prediction.symbol.replace('/USDT', ''),
                    direction: entry.prediction.direction,
                    confidence: entry.prediction.confidence,
                    amount: buyAmount,
                    mode: this.env.TRADING_MODE,
                    status: 'pending',
                    marketTitle: entry.marketResult.market?.title || '',
                    conditionId: entry.marketResult.market?.conditionId,
                    tokenId: token.tokenId,
                    marketSlug: entry.marketResult.market?.slug,
                });
                
                // 【方案 B】根据模式选择调用 executeLiveTrade 或 simulateTrade
                let actualBoughtAmount = 0;
                let tradeSuccess = false;
                
                if (this.env.TRADING_MODE === TradingMode.LIVE) {
                    // LIVE 模式：调用 executeLiveTrade 真实下单
                    const result = await this.executeLiveTrade(
                        entry.marketResult,
                        token,
                        buyAmount,
                        continueLogEntry,
                        entry.prediction,
                        { 
                            targetPeriodEndTs: entry.targetPeriodEndTs,
                            minPrice: entry.minPrice,  // 0.25
                            maxPrice: entry.maxPrice
                        }
                    );
                    tradeSuccess = result.success;
                    actualBoughtAmount = tradeSuccess ? roundUsd(buyAmount - (result.remainingAmount || 0)) : 0;
                    
                    if (!tradeSuccess && !result.blockedByCooldown) {
                        // 买入失败，更新日志状态
                        this.logger.updateEntry(continueLogEntry.id, { status: 'failed', error: '流动性不足或下单失败' });
                    }
                } else {
                    // 模拟/回测模式：调用 simulateTrade 模拟下单
                    // 注意：simulateTrade 内部会处理部分成交、更新日志、加入流动性监控队列等
                    const simResult = await this.simulateTrade(
                        entry.marketResult,
                        token,
                        buyAmount,
                        continueLogEntry,
                        entry.prediction,
                        { 
                            targetPeriodEndTs: entry.targetPeriodEndTs,
                            minPrice: entry.minPrice,
                            maxPrice: entry.maxPrice
                        }
                    );
                    // simulateTrade 始终视为成功（它内部已处理好日志更新）
                    tradeSuccess = true;
                    actualBoughtAmount = roundUsd(simResult.actualAmount);
                    
                    console.log(`\nℹ️  ${symbol} 流动性监控（${getTradingModeText(this.env.TRADING_MODE)}）：模拟补单 $${actualBoughtAmount.toFixed(2)}`);
                }
                
                if (tradeSuccess && actualBoughtAmount > 0) {
                    // 更新已买入金额（统一使用 totalBoughtAmount，金额最多 2 位小数）
                    const newTotal = roundUsd(currentBoughtAmount + actualBoughtAmount);
                    entry.totalBoughtAmount = newTotal;
                    entry.remainingAmount = roundUsd(Math.max(0, entry.originalAmount - newTotal));
                    
                    // 【资金扣除】模拟/回测模式下，补单成功后需要扣除资金（与主流程一致）
                    if (this.env.TRADING_MODE !== TradingMode.LIVE) {
                        this.adjustCoinCapital(symbol, -actualBoughtAmount);
                        this.updateCoinPeakIfNeeded(symbol);
                        // 回撤熔断已移除（流动性补单后检测点）
                    }
                    
                    if (entry.totalBoughtAmount >= entry.originalAmount) {
                        console.log(`\n✅ ${symbol} 流动性监控：买入 $${actualBoughtAmount.toFixed(2)}，累计 $${entry.totalBoughtAmount.toFixed(2)}，达到目标 $${entry.originalAmount.toFixed(2)}，剩余资金 $${this.state.currentCapital.toFixed(2)}`);
                    } else if (entry.remainingAmount > 0) {
                        console.log(`\n⚠️  ${symbol} 流动性监控：买入 $${actualBoughtAmount.toFixed(2)}，累计 $${entry.totalBoughtAmount.toFixed(2)} / $${entry.originalAmount.toFixed(2)}，继续监控，剩余资金 $${this.state.currentCapital.toFixed(2)}`);
                    }
                } else if (!tradeSuccess) {
                    // 买入失败，下次监控周期会再尝试
                    console.log(`\n⚠️  ${symbol} 流动性监控：买入失败，将在下次监控周期重试`);
                }
            } catch (error) {
                const errMsg = error instanceof Error ? error.message : String(error);
                console.log(`\n⚠️  ${symbol} 流动性监控：异常 - ${errMsg}，将在下次监控周期重试`);
            }
        }
        // 执行后：从队列中移除已达成目标、已跳过（已有完整交易/已买够/剩余不足）的条目，其余继续监控（防漏买、防买超）
        if (entriesToExecute.length > 0 || entriesCompletedOrSkipped.length > 0) {
            this.liquidityMonitorQueue = this.liquidityMonitorQueue.filter(e => {
                if (entriesCompletedOrSkipped.includes(e)) return false;
                const wasExecuted = entriesToExecute.includes(e);
                if (!wasExecuted) return true;
                return e.totalBoughtAmount < e.originalAmount && e.remainingAmount >= MIN_ORDER_SIZE_USD;
            });
        }
    }
    
    /**
     * 启动定时执行
     */
    startScheduler(intervalMs: number = 15 * 60 * 1000): void {
        if (this.state.isRunning) {
            console.log('⚠️  调度器已在运行中');
            return;
        }
        
        this.state.isRunning = true;
        
        console.log(`\n🕐 调度器已启动 (间隔: ${intervalMs / 60000} 分钟)`);
        console.log('   按 Ctrl+C 停止\n');
        
        // 立即执行一次
        this.executeRound();
        
        // 设置定时任务
        const interval = setInterval(async () => {
            if (!this.state.isRunning) {
                clearInterval(interval);
                return;
            }
            
            await this.executeRound();
        }, intervalMs);
        
        // 处理关闭信号
        const cleanup = () => {
            console.log('\n\n🛑 正在关闭...');
            this.state.isRunning = false;
            clearInterval(interval);
            
            // 打印最终摘要
            this.logger.printSummary();
            this.logger.printRecentTrades(5);
            
            process.exit(0);
        };
        
        process.on('SIGINT', cleanup);
        process.on('SIGTERM', cleanup);
    }
    
    /**
     * 停止调度器
     */
    stop(): void {
        this.state.isRunning = false;
        this.priceMonitorQueue = [];  // 清空价格监控队列（高价）
        this.lowPriceMonitorQueue = [];  // 清空价格监控队列（低价）
        this.liquidityMonitorQueue = [];  // 清空流动性监控队列
        console.log('🛑 执行器已停止');
    }
    
    /**
     * 获取当前状态
     */
    getState(): ExecutorState {
        return { ...this.state };
    }
    
    /**
     * 打印状态摘要
     */
    printStatus(): void {
        console.log('\n📊 执行器状态:');
        console.log('─'.repeat(40));
        console.log(`  运行状态:   ${this.state.isRunning ? '✅ 运行中' : '❌ 已停止'}`);
        console.log(`  当前资金:   $${this.state.currentCapital.toFixed(2)}`);
        if (this.perCoinEnabled) this.printPerCoinCapital();
        if (this.env.RISK_CONTROL_ENABLED) {
            const drawdown = this.state.peakCapital > 0 
                ? ((this.state.peakCapital - this.state.currentCapital) / this.state.peakCapital * 100).toFixed(1)
                : '0.0';
            console.log(`  峰值资金:   $${this.state.peakCapital.toFixed(2)} (回撤: ${drawdown}%)`);
            console.log(`  连续亏损:   ${this.state.consecutiveLosses} 次`);
            if (this.state.tradingPaused) {
                const pauseInfo = this.state.pausedKBarsRemaining > 0 
                    ? `${this.state.pauseReason || '风控触发'} (剩余 ${this.state.pausedKBarsRemaining} 根 K 线)`
                    : (this.state.pauseReason || '风控触发');
                console.log(`  交易状态:   ⛔ 已暂停 (${pauseInfo})`);
            }
        }
        console.log(`  今日交易:   ${this.state.todayTradeCount}/${this.env.MAX_TRADES_PER_SESSION === 0 ? '不限' : this.env.MAX_TRADES_PER_SESSION} 笔`);
        // v5 新功能状态
        if (this.env.EDGE_FILTER_ENABLED) console.log(`  Edge过滤:   ✅ 开启（负edge跳过）`);
        if (this.env.KELLY_ENABLED) console.log(`  Kelly仓位:  ✅ 开启 (frac=${this.env.KELLY_FRAC})`);
        if (this.env.COOLDOWN_BARS > 0) {
            const scopeText = this.getCooldownScope() === 'symbol' ? '按币种' : '按币种+方向';
            console.log(`  冷却机制:   ✅ ${this.env.COOLDOWN_BARS} 根K线 (${scopeText})`);
        }
        console.log(`  缓存市场:   ${this.state.marketCache.size} 个`);
        console.log(`  上次执行:   ${this.state.lastExecutionTime?.toLocaleString('zh-CN') || '从未'}`);
        console.log('─'.repeat(40) + '\n');
        
        this.logger.printSummary();
    }
    
    // ============================================================
    // 报告生成：每日 + 总汇总
    // - 每日：当天 0 点至今合并，report_daily_YYYYMMDD，每天一份不覆盖
    // - 总汇总：从开始到现在的所有历史交易，report_summary，每小时更新一次
    //   每个日志目录（logs_btc/eth/xrp）都有独立的总汇总报告
    // ============================================================

    /**
     * 从交易列表构建报告对象
     * @param allTradesForCapital 用于计算起始资金的所有交易（对于每日报告，需要包含今天之前的所有交易）
     */
    private buildReportFromTrades(
        recentTrades: any[],
        periodStart: Date,
        periodEnd: Date,
        reportType: 'daily' | 'summary',
        allTradesForCapital?: any[]
    ): object {
        // 过滤掉不在允许市场列表中的交易
        const allowedSymbols = new Set(this.env.ALLOWED_MARKETS);
        const filteredTrades = recentTrades.filter(trade => allowedSymbols.has(trade.symbol));
        const liveSettlementOverrides = this.loadLiveSettlementOverrides();
        const executedTrades = filteredTrades.filter((trade) => trade.status === 'executed');
        const pendingOrderTrades = filteredTrades.filter((trade) => trade.status === 'pending');
        const staleCancelledTrades = filteredTrades.filter((trade) => trade.liveOrderStatus === 'cancelled' || trade.stalePolicyDecision === 'cancelled');
        const latestFamilyRuleVersion = [...filteredTrades]
            .reverse()
            .map((trade) => String(trade.familyRuleVersion || '').trim())
            .find((value) => value) || '';
        const latestLowpriceFamilyId = [...filteredTrades]
            .reverse()
            .map((trade) => String(trade.lowpriceFamilyId || '').trim())
            .find((value) => value) || '';
        const effectiveWinPayout = (trade: any): number | null => {
            const matchedShares = Number(trade.matchedShares);
            if (Number.isFinite(matchedShares) && matchedShares > 0) {
                return matchedShares;
            }
            const tokenPrice = Number(trade.tokenPrice);
            if (Number.isFinite(tokenPrice) && tokenPrice > 0) {
                const feeRate = polymarketTakerFeeRate(tokenPrice);
                return (Number(trade.amount) || 0) / tokenPrice * (1 - feeRate);
            }
            return null;
        };
        const effectiveTradePnl = (trade: any): number => {
            if (trade.status !== 'executed') return 0;
            if (trade.result === 'win') {
                const payout = effectiveWinPayout(trade);
                if (payout != null) return payout - (Number(trade.amount) || 0);
            }
            if (trade.result === 'lose') {
                if (trade.stoppedOut && trade.pnl != null) return Number(trade.pnl) || 0;
                return -(Number(trade.amount) || 0);
            }
            return Number(trade.pnl) || 0;
        };
        const applyTradeToCapital = (capital: number, trade: any): number => {
            if (trade.status !== 'executed') return capital;
            return capital + effectiveTradePnl(trade);
        };
        const round2 = (value: number): number => Math.round(value * 100) / 100;
        const intOrZero = (value: unknown): number => {
            const parsed = Number(value);
            return Number.isFinite(parsed) ? Math.trunc(parsed) : 0;
        };
        const latestText = (group: any[], ...keys: string[]): string | null => {
            for (const row of [...group].reverse()) {
                for (const key of keys) {
                    const value = String(row?.[key] ?? '').trim();
                    if (value) return value;
                }
            }
            return null;
        };
        const latestNumber = (group: any[], ...keys: string[]): number | null => {
            for (const row of [...group].reverse()) {
                for (const key of keys) {
                    const value = Number(row?.[key]);
                    if (Number.isFinite(value)) return value;
                }
            }
            return null;
        };
        const maxNumber = (group: any[], ...keys: string[]): number => {
            let maxValue = 0;
            for (const row of group) {
                for (const key of keys) {
                    const value = Number(row?.[key]);
                    if (Number.isFinite(value)) maxValue = Math.max(maxValue, value);
                }
            }
            return maxValue;
        };
        const parseMinSharesFromText = (value: string | null | undefined): number => {
            const text = String(value || '').trim();
            if (!text) return 0;
            const minimumMatch = text.match(/minimum:\s*([0-9]+(?:\.[0-9]+)?)/i);
            if (minimumMatch) {
                const parsed = Number(minimumMatch[1]);
                if (Number.isFinite(parsed) && parsed > 0) return parsed;
            }
            const minSharesMatch = text.match(/min[_\s-]*shares(?:=|:)\s*([0-9]+(?:\.[0-9]+)?)/i);
            if (minSharesMatch) {
                const parsed = Number(minSharesMatch[1]);
                if (Number.isFinite(parsed) && parsed > 0) return parsed;
            }
            return 0;
        };
        const preferByPriority = (
            group: any[],
            key: string,
            priority: Record<string, number>,
        ): string | null => {
            let bestValue: string | null = null;
            let bestScore = -1;
            for (const row of group) {
                const value = String(row?.[key] ?? '').trim();
                if (!value) continue;
                const score = priority[value] ?? -1;
                if (score > bestScore) {
                    bestScore = score;
                    bestValue = value;
                }
            }
            return bestValue;
        };
        const gt5PreservationPriority: Record<string, number> = {
            contract_violation: 4,
            compressed_but_not_floored: 3,
            exact_preservation: 2,
            not_applicable: 1,
        };
        const compressionReasonPriority: Record<string, number> = {
            unexpected_floor_violation: 3,
            budget_compression: 2,
            min_shares_floor: 1,
        };

        // 逻辑订单：同一 conditionId 的多笔成交（部分成交/补单）算作一笔订单，胜率按逻辑订单数计算
        const logicalKey = (t: { conditionId?: string; id: string }) => t.conditionId || t.id;
        const groupsByCondition = new Map<string, typeof filteredTrades>();
        for (const trade of filteredTrades) {
            const key = logicalKey(trade);
            if (!groupsByCondition.has(key)) groupsByCondition.set(key, []);
            groupsByCondition.get(key)!.push(trade);
        }
        const logicalOrders = Array.from(groupsByCondition.values());

        let wins = 0, losses = 0, pending = 0, totalPnL = 0, totalConfidence = 0;
        let totalExecutedNotional = 0;
        let totalPendingNotional = 0;
        let upCount = 0, downCount = 0;
        let summaryAttemptedRungs = 0;
        let summaryPostedRungs = 0;
        let summaryFilledRungs = 0;
        let summaryExchangeRejectedRungs = 0;
        let summaryUpliftedDynamicRungs = 0;
        let summaryKeptDynamicRungs = 0;
        let summaryDroppedDynamicRungs = 0;
        let summaryUpliftedFixedOrders = 0;
        let summaryUpliftedNotionalUsd = 0;
        let summarySkippedLogicalOrders = 0;
        const bySymbol: Record<string, { trades: number; wins: number; losses: number; pending: number; pnl: number; avgConfidence: number; totalConfidence: number; confidenceSampleCount: number; upCount: number; downCount: number; pendingOrderCount: number; pendingNotional: number; executedNotional: number }> = {};
        const bySymbolFillCount: Record<string, number> = {};
        for (const trade of filteredTrades) {
            if (!bySymbol[trade.symbol]) {
                bySymbol[trade.symbol] = { trades: 0, wins: 0, losses: 0, pending: 0, pnl: 0, avgConfidence: 0, totalConfidence: 0, confidenceSampleCount: 0, upCount: 0, downCount: 0, pendingOrderCount: 0, pendingNotional: 0, executedNotional: 0 };
            }
            const s = bySymbol[trade.symbol];
            s.totalConfidence += trade.confidence;
            s.confidenceSampleCount += 1;
            totalConfidence += trade.confidence;
            if (trade.status === 'executed') {
                bySymbolFillCount[trade.symbol] = (bySymbolFillCount[trade.symbol] || 0) + 1;
                const executedAmount = Number(trade.amount) || 0;
                totalExecutedNotional += executedAmount;
                s.executedNotional += executedAmount;
                const effectivePnl = effectiveTradePnl(trade); totalPnL += effectivePnl; s.pnl += effectivePnl;
            } else if (trade.status === 'pending') {
                const pendingAmount = Number(trade.pendingAmountUsd ?? trade.requestedAmountUsd ?? trade.amount) || 0;
                totalPendingNotional += pendingAmount;
                s.pendingOrderCount += 1;
                s.pendingNotional += pendingAmount;
            }
        }
        const allLogicalTradeRows = logicalOrders.map((group) => {
            const first = group[0];
            const executedGroup = group.filter((row) => row.status === 'executed');
            const pendingGroup = group.filter((row) => row.status === 'pending');
            const failedGroup = group.filter((row) => row.status === 'failed');
            const hasExecuted = executedGroup.length > 0;
            const hasPendingOrders = pendingGroup.length > 0;
            const groupType = hasExecuted
                ? 'executed_logical'
                : hasPendingOrders
                    ? 'open_order_only'
                    : 'skipped';
            let result = hasExecuted
                ? (
                    group.some(t => t.result === 'win')
                        ? 'win'
                        : group.some(t => t.result === 'lose')
                            ? 'lose'
                            : 'pending'
                )
                : (hasPendingOrders ? 'open_orders' : 'skipped');
            const conditionId = first.conditionId || null;
            const liveMode = group.some((row) => row.mode === 'live');
            const settlementOverride = liveMode && conditionId
                ? liveSettlementOverrides.get(String(conditionId).trim().toLowerCase())
                : undefined;
            if (settlementOverride?.result) {
                result = settlementOverride.result;
            }
            const totalAmount = executedGroup.reduce((sum, row) => sum + (Number(row.amount) || 0), 0);
            const pendingNotional = pendingGroup.reduce((sum, row) => sum + (Number(row.pendingAmountUsd ?? row.requestedAmountUsd ?? row.amount) || 0), 0);
            const rawTotalGroupPnl = executedGroup.reduce((sum, row) => sum + effectiveTradePnl(row), 0);
            // 只允许 live settlement override 覆盖结果，不再覆盖逻辑单 PnL。
            // 历史上旧 summary/logical ledger 会把 claimed / claim_ready 口径写成 0，
            // 如果这里继续用 override 的 totalPnl/settledPnl，会把真实赢单的盈亏抹掉。
            const totalGroupPnl = rawTotalGroupPnl;
            const settledGroupPnl = totalGroupPnl;
            const totalGroupConfidence = group.reduce((sum, row) => sum + (Number(row.confidence) || 0), 0);
            const attemptedRungs = Math.max(maxNumber(group, 'attemptedRungs'), group.length);
            const postedRungs = Math.max(maxNumber(group, 'postedRungs'), executedGroup.length + pendingGroup.length);
            const filledRungs = Math.max(maxNumber(group, 'filledRungs', 'fillCount'), executedGroup.length);
            const exchangeRejectedRungs = Math.max(maxNumber(group, 'exchangeRejectedRungs'), failedGroup.filter((row) => String(row.executionDecision || '') === 'exchange_rejected').length);
            const upliftedDynamicRungs = maxNumber(group, 'upliftedDynamicRungs');
            const keptDynamicRungs = Math.max(maxNumber(group, 'keptDynamicRungs'), postedRungs);
            const droppedDynamicRungs = maxNumber(group, 'droppedDynamicRungs');
            const compressedRungs = Math.max(maxNumber(group, 'compressedRungs'), droppedDynamicRungs);
            const upliftedFixedOrders = maxNumber(group, 'upliftedFixedOrders');
            const upliftedNotionalUsd = round2(maxNumber(group, 'upliftedNotionalUsd'));
            const rejectionReason = latestText(group, 'lastNonExecutionReason', 'lastRejectionReason', 'error');
            const exchangeMinOrderSizeShares = latestNumber(group, 'exchangeMinOrderSizeShares')
                || parseMinSharesFromText(rejectionReason)
                || 0;
            const gt5PreservationStatus = preferByPriority(group, 'gt5PreservationStatus', gt5PreservationPriority);
            const compressionReason = preferByPriority(group, 'compressionReason', compressionReasonPriority);
            const tokenPriceRows = group
                .map((row) => Number(row.avgActualFillPrice ?? row.tokenPrice))
                .filter((value) => Number.isFinite(value) && value > 0);
            const preferredOutcome = String(first.tokenOutcome || (first.direction === 'UP' ? 'Up' : 'Down') || '').trim() || null;
            const winningOutcome = (result === 'pending' || result === 'skipped' || result === 'open_orders')
                ? null
                : (
                    result === 'win'
                        ? preferredOutcome
                        : preferredOutcome === 'Up'
                            ? 'Down'
                            : preferredOutcome === 'Down'
                                ? 'Up'
                                : null
                );
            const resolutionStatus = hasExecuted
                ? (result === 'pending' ? 'pending_resolution' : 'resolved')
                : hasPendingOrders
                    ? 'open_orders_pending_fill'
                    : 'not_executed';
            const settlementStatus = hasExecuted
                ? (result === 'pending' ? 'pending' : 'settled')
                : hasPendingOrders
                    ? 'open_orders_pending_fill'
                    : 'not_executed';
            const claimStatus = !liveMode
                ? 'not_applicable'
                : !hasExecuted
                    ? (hasPendingOrders ? 'awaiting_fill' : 'not_applicable')
                    : result === 'win'
                    ? (conditionId && this.hasSuccessfulClaim(conditionId) ? 'claimed' : 'claim_ready')
                    : result === 'lose'
                        ? 'not_applicable'
                        : 'awaiting_resolution';
            const executionDecision = latestText(group, 'executionDecision') || (groupType === 'skipped' ? 'skipped' : null);
            return {
                logicalTradeId: logicalKey(first),
                groupType,
                conditionId,
                marketSlug: first.marketSlug || null,
                firstFillId: first.id,
                symbol: first.symbol,
                direction: first.direction,
                fillCount: executedGroup.length,
                pendingOrderCount: pendingGroup.length,
                rejectedRungCount: exchangeRejectedRungs,
                attemptedRungs,
                postedRungs,
                filledRungs,
                pendingRungs: pendingGroup.length,
                exchangeRejectedRungs,
                upliftedDynamicRungs,
                keptDynamicRungs,
                compressedRungs,
                droppedDynamicRungs,
                upliftedFixedOrders,
                upliftedNotionalUsd,
                exchangeMinOrderSizeShares,
                executionDecision,
                gt5PreservationStatus,
                compressionReason,
                lastNonExecutionReason: rejectionReason,
                totalAmount: Math.round(totalAmount * 100) / 100,
                pendingNotional: Math.round(pendingNotional * 100) / 100,
                totalPnL: Math.round(totalGroupPnl * 100) / 100,
                settledPnL: Math.round((result === 'pending' || result === 'skipped' ? 0 : settledGroupPnl) * 100) / 100,
                avgConfidence: group.length > 0 ? Math.round((totalGroupConfidence / group.length) * 1000) / 10 : 0,
                avgTokenPrice: tokenPriceRows.length > 0 ? Math.round((tokenPriceRows.reduce((sum, value) => sum + value, 0) / tokenPriceRows.length) * 10000) / 10000 : null,
                minTokenPrice: tokenPriceRows.length > 0 ? Math.min(...tokenPriceRows) : null,
                maxTokenPrice: tokenPriceRows.length > 0 ? Math.max(...tokenPriceRows) : null,
                result,
                status: result,
                resolutionStatus,
                settlementStatus,
                claimStatus,
                settlementOverrideSource: settlementOverride?.source,
                winningOutcome,
                lastRejectionReason: rejectionReason,
                firstTimestamp: group
                    .map((row) => row.timestamp)
                    .filter((value) => typeof value === 'string' && value)
                    .sort()[0] || first.timestamp,
                lastTimestamp: group
                    .map((row) => row.timestamp)
                    .filter((value) => typeof value === 'string' && value)
                    .sort()
                    .slice(-1)[0] || first.timestamp,
            };
        });
        const logicalTradeRows = allLogicalTradeRows.filter((row) => row.groupType === 'executed_logical');
        const openOrderOnlyRows = allLogicalTradeRows.filter((row) => row.groupType === 'open_order_only');
        const logicalRowById = new Map(allLogicalTradeRows.map((row) => [String(row.logicalTradeId), row]));
        for (const group of logicalOrders) {
            const first = group[0];
            const row = logicalRowById.get(String(logicalKey(first)));
            const result = row?.result || 'skipped';
            if (row) {
                summaryAttemptedRungs += intOrZero(row.attemptedRungs);
                summaryPostedRungs += intOrZero(row.postedRungs);
                summaryFilledRungs += intOrZero(row.filledRungs);
                summaryExchangeRejectedRungs += intOrZero(row.exchangeRejectedRungs);
                summaryUpliftedDynamicRungs += intOrZero(row.upliftedDynamicRungs);
                summaryKeptDynamicRungs += intOrZero(row.keptDynamicRungs);
                summaryDroppedDynamicRungs += intOrZero(row.droppedDynamicRungs);
                summaryUpliftedFixedOrders += intOrZero(row.upliftedFixedOrders);
                summaryUpliftedNotionalUsd = round2(summaryUpliftedNotionalUsd + Number(row.upliftedNotionalUsd || 0));
            }
            if (row?.groupType === 'skipped') {
                summarySkippedLogicalOrders += 1;
                continue;
            }
            if (row?.groupType === 'open_order_only') {
                continue;
            }
            if (result === 'win') { wins++; } else if (result === 'lose') { losses++; } else { pending++; }
            if (first.direction === 'UP') { upCount++; } else { downCount++; }
            const sym = first.symbol;
            if (!bySymbol[sym]) bySymbol[sym] = { trades: 0, wins: 0, losses: 0, pending: 0, pnl: 0, avgConfidence: 0, totalConfidence: 0, confidenceSampleCount: 0, upCount: 0, downCount: 0, pendingOrderCount: 0, pendingNotional: 0, executedNotional: 0 };
            bySymbol[sym].trades++;
            if (result === 'win') bySymbol[sym].wins++;
            else if (result === 'lose') bySymbol[sym].losses++;
            else bySymbol[sym].pending++;
            if (first.direction === 'UP') bySymbol[sym].upCount++; else bySymbol[sym].downCount++;
        }
        for (const k of Object.keys(bySymbol)) {
            const s = bySymbol[k];
            const confidenceCount = s.confidenceSampleCount || 0;
            s.avgConfidence = confidenceCount > 0 ? s.totalConfidence / confidenceCount : 0;
        }
        // 过滤 bySymbol，只保留允许的市场
        const filteredBySymbol: Record<string, any> = {};
        for (const symbol of this.env.ALLOWED_MARKETS) {
            if (bySymbol[symbol]) {
                filteredBySymbol[symbol] = bySymbol[symbol];
            }
        }
        const completed = wins + losses;
        const winRate = completed > 0 ? wins / completed : 0;
        const totalLogicalOrders = logicalTradeRows.length;
        const logicalSettledRows = logicalTradeRows.filter((row) => {
            const result = String(row.result || '').trim().toLowerCase();
            return result === 'win' || result === 'lose';
        });
        const logicalSettledPnl = logicalSettledRows.reduce((sum, row) => sum + (Number(row.totalPnL) || 0), 0);
        const logicalSettledPnlBySymbol = new Map<string, number>();
        for (const row of logicalSettledRows) {
            const symbol = String(row.symbol || '').trim().toUpperCase() || 'UNKNOWN';
            logicalSettledPnlBySymbol.set(
                symbol,
                (logicalSettledPnlBySymbol.get(symbol) || 0) + (Number(row.totalPnL) || 0),
            );
        }
        const avgConf = filteredTrades.length > 0 ? totalConfidence / filteredTrades.length : 0;
        const latestExecutionDecision = [...allLogicalTradeRows]
            .reverse()
            .map((row) => String(row.executionDecision || '').trim())
            .find((value) => value) || '';
        const latestNonExecutionReason = [...allLogicalTradeRows]
            .reverse()
            .map((row) => String(row.lastNonExecutionReason || row.lastRejectionReason || '').trim())
            .find((value) => value) || '';
        const latestExchangeMinOrderSizeShares = [...allLogicalTradeRows]
            .reverse()
            .map((row) => Number(row.exchangeMinOrderSizeShares))
            .find((value) => Number.isFinite(value) && value > 0) || 0;
        const latestGt5PreservationStatus = [...allLogicalTradeRows]
            .reverse()
            .map((row) => String(row.gt5PreservationStatus || '').trim())
            .find((value) => value) || '';
        const latestCompressionReason = [...allLogicalTradeRows]
            .reverse()
            .map((row) => String(row.compressionReason || '').trim())
            .find((value) => value) || '';
        const inferredGt5PreservationStatus = latestGt5PreservationStatus
            || (latestExecutionDecision === 'compressed_dynamic_rungs' ? 'compressed_but_not_floored' : '')
            || (latestExecutionDecision === 'uplifted_fixed_order_to_min_shares' ? 'not_applicable' : '')
            || '';
        const inferredCompressionReason = latestCompressionReason
            || (latestExecutionDecision === 'compressed_dynamic_rungs' ? 'budget_compression' : '')
            || (latestExecutionDecision === 'uplifted_fixed_order_to_min_shares' ? 'min_shares_floor' : '')
            || '';
        const inferredExchangeMinOrderSizeShares = latestExchangeMinOrderSizeShares
            || parseMinSharesFromText(latestNonExecutionReason)
            || this.getLiveExchangeMinShares()
            || 0;

        // 计算每日胜率（按日期分组，同一 conditionId 在同一天只算一笔）
        const dailyStats: Record<string, { trades: number; wins: number; losses: number; fillCount: number; winRate: number; pnl: number; fillPnL: number; logicalTradePnl: number }> = {};
        const dailySeenCondition = new Map<string, Set<string>>();
        for (const trade of filteredTrades) {
            if (trade.status !== 'executed' || !trade.result || trade.result === 'pending') continue;
            const tradeDate = new Date(trade.timestamp);
            const year = tradeDate.getFullYear();
            const month = String(tradeDate.getMonth() + 1).padStart(2, '0');
            const day = String(tradeDate.getDate()).padStart(2, '0');
            const dateKey = `${year}-${month}-${day}`;
            if (!dailyStats[dateKey]) {
                dailyStats[dateKey] = { trades: 0, wins: 0, losses: 0, fillCount: 0, winRate: 0, pnl: 0, fillPnL: 0, logicalTradePnl: 0 };
            }
            const dayStats = dailyStats[dateKey];
            dayStats.fillCount++;
            const key = logicalKey(trade);
            if (!dailySeenCondition.has(dateKey)) dailySeenCondition.set(dateKey, new Set());
            if (!dailySeenCondition.get(dateKey)!.has(key)) {
                dailySeenCondition.get(dateKey)!.add(key);
                dayStats.trades++;
                if (trade.result === 'win') dayStats.wins++; else if (trade.result === 'lose') dayStats.losses++;
            }
            const effectivePnl = effectiveTradePnl(trade);
            dayStats.pnl += effectivePnl;
            dayStats.fillPnL += effectivePnl;
            dayStats.logicalTradePnl += effectivePnl;
        }
        // 计算每日胜率
        for (const dateKey of Object.keys(dailyStats)) {
            const dayStats = dailyStats[dateKey];
            dayStats.winRate = (dayStats.wins + dayStats.losses) > 0 ? Math.round((dayStats.wins / (dayStats.wins + dayStats.losses)) * 1000) / 10 : 0;
            dayStats.pnl = Math.round(dayStats.pnl * 100) / 100;
            dayStats.fillPnL = Math.round(dayStats.fillPnL * 100) / 100;
            dayStats.logicalTradePnl = Math.round(dayStats.logicalTradePnl * 100) / 100;
        }
        
        // 基于交易历史重新计算资金，确保与交易记录一致
        // 对于每日报告，需要先计算今天开始时的资金（从所有历史交易），然后加上今天的交易
        // 对于总汇总报告，从初始资金开始计算所有交易
        let computedCapital = this.env.INITIAL_CAPITAL;
        let startingCapital = this.env.INITIAL_CAPITAL;
        
        // 如果是每日报告，需要先计算今天开始时的资金
        if (reportType === 'daily' && allTradesForCapital) {
            const allowedSymbolsForCapital = new Set(this.env.ALLOWED_MARKETS);
            const allFilteredTrades = allTradesForCapital.filter(trade => allowedSymbolsForCapital.has(trade.symbol));
            const sortedAllTrades = [...allFilteredTrades].sort((a, b) => 
                new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
            );
            
            // 计算今天开始时的资金（处理今天之前的所有交易）
            for (const trade of sortedAllTrades) {
                const tradeDate = new Date(trade.timestamp);
                if (tradeDate >= periodStart) break; // 到达今天，停止
                startingCapital = applyTradeToCapital(startingCapital, trade);
            }
            startingCapital = Math.max(0, startingCapital);
            computedCapital = startingCapital; // 从今天开始时的资金开始
        }
        
        // 处理报告期间的交易（对于每日报告是今天的交易，对于总汇总报告是所有交易）
        const sortedTrades = [...filteredTrades].sort((a, b) => 
            new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
        );
        
        for (const trade of sortedTrades) {
            computedCapital = applyTradeToCapital(computedCapital, trade);
        }
        computedCapital = Math.max(0, computedCapital);

        const perSymbolCapital: Record<string, { initialCapital: number; currentCapital: number; capitalChange: number; peakCapital: number }> = {};
        if (this.perCoinEnabled) {
            const allowedList = Array.isArray(this.env.ALLOWED_MARKETS) ? this.env.ALLOWED_MARKETS : [];
            const symbolCount = Math.max(1, allowedList.length);
            const initialPerCoin = this.env.INITIAL_CAPITAL / symbolCount;
            const capitalSourceTrades = reportType === 'daily' && allTradesForCapital ? allTradesForCapital : filteredTrades;
            for (const sym of allowedList) {
                const symbolAllTrades = capitalSourceTrades
                    .filter(trade => trade.symbol === sym && allowedSymbols.has(trade.symbol))
                    .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
                const symbolRecentTrades = filteredTrades
                    .filter(trade => trade.symbol === sym)
                    .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
                let symbolStartingCapital = initialPerCoin;
                let symbolPeakCapital = initialPerCoin;
                if (reportType === 'daily' && allTradesForCapital) {
                    for (const trade of symbolAllTrades) {
                        const tradeDate = new Date(trade.timestamp);
                        if (tradeDate >= periodStart) break;
                        symbolStartingCapital = applyTradeToCapital(symbolStartingCapital, trade);
                        symbolPeakCapital = Math.max(symbolPeakCapital, symbolStartingCapital);
                    }
                }
                symbolStartingCapital = Math.max(0, symbolStartingCapital);
                let symbolCurrentCapital = symbolStartingCapital;
                symbolPeakCapital = Math.max(symbolPeakCapital, symbolStartingCapital);
                for (const trade of symbolRecentTrades) {
                    symbolCurrentCapital = applyTradeToCapital(symbolCurrentCapital, trade);
                    symbolPeakCapital = Math.max(symbolPeakCapital, symbolCurrentCapital);
                }
                symbolCurrentCapital = Math.max(0, symbolCurrentCapital);
                perSymbolCapital[sym] = {
                    initialCapital: round2(reportType === 'daily' ? symbolStartingCapital : initialPerCoin),
                    currentCapital: round2(symbolCurrentCapital),
                    capitalChange: round2(symbolCurrentCapital - (reportType === 'daily' ? symbolStartingCapital : initialPerCoin)),
                    peakCapital: round2(symbolPeakCapital),
                };
            }
        }
        
        return {
            reportType,
            reportDate: periodEnd.toISOString(),
            reportPeriod: { start: periodStart.toISOString(), end: periodEnd.toISOString() },
            summary: {
                primaryTradeUnit: 'logical_order',
                reportingContract: 'summary_counts_use_logical_orders_executed_fills_and_pending_orders_separately',
                totalTrades: totalLogicalOrders,
                totalLogicalTrades: totalLogicalOrders,
                skippedLogicalTrades: summarySkippedLogicalOrders,
                openOrderLogicalGroups: openOrderOnlyRows.length,
                totalFills: executedTrades.length,
                fillCount: executedTrades.length,
                completedTrades: completed,
                wins, losses, pending,
                upCount, downCount,
                winRate: Math.round(winRate * 1000) / 10,
                totalPnL: Math.round(logicalSettledPnl * 100) / 100,
                logicalTradePnL: Math.round(logicalSettledPnl * 100) / 100,
                fillPnL: Math.round(totalPnL * 100) / 100,
                settledPnL: Math.round(logicalSettledPnl * 100) / 100,
                executedNotional: Math.round(totalExecutedNotional * 100) / 100,
                exchangeMinOrderSizeShares: inferredExchangeMinOrderSizeShares || undefined,
                attemptedRungs: summaryAttemptedRungs,
                postedRungs: summaryPostedRungs,
                filledRungs: summaryFilledRungs,
                exchangeRejectedRungs: summaryExchangeRejectedRungs,
                upliftedDynamicRungs: summaryUpliftedDynamicRungs,
                keptDynamicRungs: summaryKeptDynamicRungs,
                compressedRungs: summaryDroppedDynamicRungs,
                droppedDynamicRungs: summaryDroppedDynamicRungs,
                upliftedFixedOrders: summaryUpliftedFixedOrders,
                upliftedNotionalUsd: round2(summaryUpliftedNotionalUsd),
                executionDecision: latestExecutionDecision || undefined,
                gt5PreservationStatus: inferredGt5PreservationStatus || undefined,
                compressionReason: inferredCompressionReason || undefined,
                lastNonExecutionReason: latestNonExecutionReason || undefined,
                pendingRungs: pendingOrderTrades.length,
                pendingOrderCount: pendingOrderTrades.length,
                pendingNotional: Math.round(totalPendingNotional * 100) / 100,
                settledLogicalTrades: completed,
                unsettledLogicalTrades: pending,
                claimReadyLogicalTrades: logicalTradeRows.filter((row) => row.claimStatus === 'claim_ready').length,
                claimedLogicalTrades: logicalTradeRows.filter((row) => row.claimStatus === 'claimed').length,
                resolutionStatus: pending > 0 ? 'partially_resolved' : 'resolved',
                settlementStatus: pending > 0 ? 'partially_settled' : 'settled',
                claimStatus: logicalTradeRows.some((row) => row.claimStatus === 'claim_ready')
                    ? 'claim_ready'
                    : logicalTradeRows.some((row) => row.claimStatus === 'claimed')
                        ? 'claimed'
                        : 'not_applicable',
                familyRuleVersion: latestFamilyRuleVersion || this.getFamilyRuleVersion() || undefined,
                lowpriceFamilyId: latestLowpriceFamilyId || this.getLowpriceFamilyId() || undefined,
                staleOrderPolicy: this.getStaleOrderPolicy(),
                staleCancelledRungs: staleCancelledTrades.length,
                staleCancelledNotional: Math.round(staleCancelledTrades.reduce((sum, trade) => sum + (Number(trade.requestedAmountUsd ?? trade.pendingAmountUsd ?? trade.amount) || 0), 0) * 100) / 100,
                avgConfidence: Math.round(avgConf * 1000) / 10,
                currentCapital: Math.round(computedCapital * 100) / 100,
                initialCapital: reportType === 'daily' ? Math.round(startingCapital * 100) / 100 : this.env.INITIAL_CAPITAL,
                capitalChange: reportType === 'daily' 
                    ? Math.round((computedCapital - startingCapital) * 100) / 100  // 每日报告：相对于今天开始时的资金
                    : Math.round((computedCapital - this.env.INITIAL_CAPITAL) * 100) / 100,  // 总汇总：相对于初始资金
            },
            bySymbol: Object.fromEntries(
                Object.entries(filteredBySymbol).map(([sym, s]) => [
                    sym,
                    {
                        trades: s.trades, wins: s.wins, losses: s.losses, pending: s.pending,
                        primaryTradeUnit: 'logical_order',
                        logicalTradeCount: s.trades,
                        fillCount: bySymbolFillCount[sym] || 0,
                        pendingOrderCount: s.pendingOrderCount,
                        pendingNotional: Math.round(s.pendingNotional * 100) / 100,
                        executedNotional: Math.round(s.executedNotional * 100) / 100,
                        upCount: s.upCount, downCount: s.downCount,
                        winRate: (s.wins + s.losses) > 0 ? Math.round(s.wins / (s.wins + s.losses) * 1000) / 10 : 0,
                        pnl: Math.round(s.pnl * 100) / 100,
                        logicalTradePnl: Math.round((logicalSettledPnlBySymbol.get(sym) || 0) * 100) / 100,
                        fillPnL: Math.round(s.pnl * 100) / 100,
                        avgConfidence: Math.round(s.avgConfidence * 1000) / 10,
                        ...(perSymbolCapital[sym] || {}),
                    },
                ])
            ),
            config: {
                tradingMode: this.env.TRADING_MODE,
                initialCapital: this.env.INITIAL_CAPITAL,
                perCoinCapital: this.perCoinEnabled ? Object.fromEntries(this.capitalByCoin) : undefined,
                perCoinPeakCapital: this.perCoinEnabled ? Object.fromEntries(this.peakCapitalByCoin) : undefined,
                betSizePercent: (() => {
                    if (this.env.KELLY_ENABLED) {
                        return `Kelly模式 (frac=${this.env.KELLY_FRAC}, normal=${(this.env.BET_PCT_NORMAL * 100).toFixed(1)}%, conservative=${(this.env.BET_PCT_CONSERVATIVE * 100).toFixed(1)}%)`;
                    }
                    const tiered = this.env.TIER_P1_PCT != null && this.env.TIER_P2_PCT != null
                        && this.env.TIER_P3_PCT != null && this.env.TIER_P4_PCT != null;
                    if (tiered) {
                        return `档位 p1～p4: ${this.env.TIER_P1_PCT}%,${this.env.TIER_P2_PCT}%,${this.env.TIER_P3_PCT}%,${this.env.TIER_P4_PCT}% (B1=${this.env.TIER_B1},B2=${this.env.TIER_B2},B3=${this.env.TIER_B3})`;
                    }
                    return `${this.env.BET_SIZE_PERCENT}%`;
                })(),
                probThreshold: (() => {
                    const dual = this.env.PROB_THRESHOLD_UP != null && this.env.PROB_THRESHOLD_DOWN != null;
                    if (dual) {
                        return `UP≥${Math.round(this.env.PROB_THRESHOLD_UP! * 100)}%, DOWN≥${Math.round((1 - this.env.PROB_THRESHOLD_DOWN!) * 100)}% (即prob<${Math.round(this.env.PROB_THRESHOLD_DOWN! * 100)}%)`;
                    }
                    return `${Math.round(this.env.PROB_THRESHOLD * 100)}%`;
                })(),
                minProfitRatio: this.env.MIN_PROFIT_RATIO,
                allowedMarkets: this.env.ALLOWED_MARKETS,
                lowpriceFamilyId: this.getLowpriceFamilyId() || undefined,
                familyRuleVersion: this.getFamilyRuleVersion() || undefined,
                staleOrderPolicy: this.getStaleOrderPolicy(),
                staleCancelMinAgeSec: this.env.STALE_CANCEL_MIN_AGE_SEC,
                staleCancelMinDriftTicks: this.env.STALE_CANCEL_MIN_DRIFT_TICKS,
                staleCancelMinTimeToExpirySec: this.env.STALE_CANCEL_MIN_TIME_TO_EXPIRY_SEC,
            },
            dailyStats: dailyStats,  // 每日胜率统计
            reportingContract: {
                primaryTradeUnit: 'logical_order',
                tradesArrayUnit: 'fill',
                summaryCountsUse: 'logical_orders',
                fillCountUse: 'executed_fills_only',
                pendingUse: 'executed_logical_orders_only_for_pending_count_with_open_orders_tracked_separately',
                pnlUse: 'summary_total_and_logicalTradePnl_use_settled_logical_orders_fillPnL_uses_executed_fills',
                settlementStages: 'fill -> resolved -> settled -> claim_ready/claimed',
                note: '主统计只把含 executed fills 的逻辑单计入 wins/losses/pending；open posted orders 与 failed/skipped groups 单独统计；PnL 只在 settled 后进入 realized 口径。',
            },
            logicalTrades: logicalTradeRows,
            trades: filteredTrades.map((t) => ({
                id: t.id, timestamp: t.timestamp, symbol: t.symbol, direction: t.direction,
                confidence: t.confidence,
                status: t.status,
                amount: t.amount,
                pendingAmountUsd: t.pendingAmountUsd ?? t.requestedAmountUsd ?? 0,
                result: t.result || 'pending',
                pnl: t.pnl,
                familyRuleVersion: t.familyRuleVersion || null,
                lowpriceFamilyId: t.lowpriceFamilyId || null,
                stalePolicyDecision: t.stalePolicyDecision || null,
                staleCancelReason: t.staleCancelReason || null,
            })),
        };
    }

    private getMergedTradesFilePath(): string {
        return path.join(this.logsDirFull, 'prediction_trades.json');
    }

    private getModeTradesFilePath(): string {
        return this.logger.getLogFilePath();
    }

    private readTradesFromFile(filePath: string): TradeLogEntry[] {
        try {
            if (!fs.existsSync(filePath)) return [];
            const raw = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
            if (!Array.isArray(raw)) return [];
            return raw as TradeLogEntry[];
        } catch {
            return [];
        }
    }

    private loadLiveSettlementOverrides(): Map<string, LiveSettlementOverride> {
        const overrides = new Map<string, LiveSettlementOverride>();
        if (this.env.TRADING_MODE !== TradingMode.LIVE) return overrides;
        const reportsDir = path.join(this.logsDirFull, 'reports');
        const candidates = [
            path.join(reportsDir, `report_summary.${this.modeScope}.json`),
            path.join(reportsDir, 'report_summary.json'),
        ];
        for (const filePath of candidates) {
            try {
                if (!fs.existsSync(filePath)) continue;
                const payload = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
                const rows = Array.isArray(payload?.logicalTrades) ? payload.logicalTrades : [];
                for (const row of rows) {
                    const conditionId = String(row?.conditionId || row?.logicalTradeId || '').trim().toLowerCase();
                    const result = String(row?.result || '').trim().toLowerCase();
                    const resolved = ['resolved', 'settled'].includes(String(row?.resolutionStatus || '').trim().toLowerCase())
                        || ['resolved', 'settled'].includes(String(row?.settlementStatus || '').trim().toLowerCase());
                    if (!conditionId || (result !== 'win' && result !== 'lose') || !resolved) continue;
                    const totalPnL = Number(row?.totalPnL);
                    const settledPnL = Number(row?.settledPnL);
                    overrides.set(conditionId, {
                        result: result as 'win' | 'lose',
                        totalPnL: Number.isFinite(totalPnL) ? totalPnL : undefined,
                        settledPnL: Number.isFinite(settledPnL) ? settledPnL : undefined,
                        source: String(payload?.writerOwner || 'previous_live_summary'),
                    });
                }
                if (overrides.size > 0) return overrides;
            } catch {
                continue;
            }
        }
        return overrides;
    }

    private validateReportPayload(report: any, fileBase: string): void {
        const reportDate = String(report?.reportDate || '').trim();
        const periodEnd = String(report?.reportPeriod?.end || '').trim();
        if (!reportDate || !periodEnd || reportDate !== periodEnd) {
            throw new Error(`invalid report payload for ${fileBase}: reportDate must equal reportPeriod.end`);
        }
    }

    private writeReportArtifactsAtomically(basePath: string, report: any, tradesFileName: string): { jsonPath: string; txtPath: string } {
        this.validateReportPayload(report, path.basename(basePath));
        const jsonPath = `${basePath}.json`;
        const txtPath = `${basePath}.txt`;
        const stamp = new Date();
        const suffix = `tmp-${process.pid}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
        const jsonTmpPath = `${jsonPath}.${suffix}`;
        const txtTmpPath = `${txtPath}.${suffix}`;
        try {
            fs.writeFileSync(jsonTmpPath, JSON.stringify(report, null, 2), 'utf-8');
            fs.writeFileSync(txtTmpPath, this.formatReportToText(report, tradesFileName), 'utf-8');
            // 先替换 txt，再替换 canonical json，尽量减少消费者读到“新 json / 旧 txt”的窗口。
            fs.renameSync(txtTmpPath, txtPath);
            fs.renameSync(jsonTmpPath, jsonPath);
            fs.utimesSync(txtPath, stamp, stamp);
            fs.utimesSync(jsonPath, stamp, stamp);
        } catch (error) {
            try {
                if (fs.existsSync(jsonTmpPath)) fs.unlinkSync(jsonTmpPath);
            } catch {}
            try {
                if (fs.existsSync(txtTmpPath)) fs.unlinkSync(txtTmpPath);
            } catch {}
            throw error;
        }
        return { jsonPath, txtPath };
    }

    private annotateRuntimeReportMetadata(report: any, kind: 'summary' | 'daily', modeScope: string): any {
        const stamp = new Date();
        const generatedAt = stamp.toISOString();
        report.writerOwner = 'prediction_executor_runtime_reports';
        report.generatedByRuntimeReportAt = generatedAt;
        report.sourceInputsUpdatedAt = generatedAt;
        report.generationEpoch = stamp.getTime();
        report.runtimeWritePolicy = this.env.TRADING_MODE === TradingMode.LIVE
            ? 'runtime_live_generate_reports'
            : `runtime_${modeScope}_generate_reports`;
        report.summaryContractVersion = 'runtime_summary_v2';
        report.reportKind = kind;
        return report;
    }

    /**
     * 生成总汇总报告：从开始到现在的所有历史交易，写 report_summary，每小时更新一次
     * 这是每个日志目录（logs_btc/eth/xrp）的完整历史汇总，不是每日报告的汇总
     */
    async generateSummaryReport(): Promise<string | null> {
        const now = new Date();
        const reportsDir = path.join(this.logsDirFull, 'reports');
        if (!fs.existsSync(reportsDir)) fs.mkdirSync(reportsDir, { recursive: true });
        const allTrades = this.readTradesFromFile(this.getModeTradesFilePath());
        // 总汇总报告：从最早交易开始到现在的所有交易（全部历史）
        const firstTrade = allTrades.length > 0 ? new Date(allTrades[0].timestamp) : now;
        const report = this.buildReportFromTrades(allTrades, firstTrade, now, 'summary') as any;
        // LIVE 模式：真钱主表必须以官方可用现金余额为展示口径。
        // 本地逐单推导资金只作策略诊断，不能抬高钱包盈亏，否则会和官网资金打架。
        if (this.env.TRADING_MODE === TradingMode.LIVE) {
            const computedCurrentCapital = Math.round(Number(report.summary.currentCapital || 0) * 100) / 100;
            const walletCurrentCapital = Math.round(this.state.currentCapital * 100) / 100;
            const displayCapital = walletCurrentCapital;
            report.summary.walletCurrentCapital = walletCurrentCapital;
            report.summary.computedCurrentCapital = computedCurrentCapital;
            report.summary.capitalSource = 'live_wallet_balance_cash_api';
            report.summary.currentCapital = displayCapital;
            report.summary.capitalChange = Math.round((displayCapital - this.env.INITIAL_CAPITAL) * 100) / 100;
            report.summary.strategyLogicalPnlUsd = Math.round(Number(report.summary.totalPnL || 0) * 100) / 100;
            const publicTrades = await this.fetchPublicActivityTrades();
            this.annotateLiveReportLastAttemptFromCanonicalRows(report, allTrades, publicTrades);
        }
        this.annotateRuntimeReportMetadata(report, 'summary', this.modeScope);
        const modeBase = `report_summary.${this.modeScope}`;
        const modePaths = this.writeReportArtifactsAtomically(
            path.join(reportsDir, modeBase),
            report,
            path.basename(this.logger.getLogFilePath()),
        );

        const mergedTrades = this.readTradesFromFile(this.getMergedTradesFilePath());
        const mergedFirstTrade = mergedTrades.length > 0 ? new Date(mergedTrades[0].timestamp) : now;
        const mergedReport = this.env.TRADING_MODE === TradingMode.LIVE
            ? JSON.parse(JSON.stringify(report))
            : this.buildReportFromTrades(mergedTrades, mergedFirstTrade, now, 'summary') as any;
        this.annotateRuntimeReportMetadata(mergedReport, 'summary', 'merged');
        const mergedBase = 'report_summary';
        this.writeReportArtifactsAtomically(
            path.join(reportsDir, mergedBase),
            mergedReport,
            'prediction_trades.json',
        );

        const comboName = this.getComboDisplayName();
        console.log('\n' + '═'.repeat(60));
        console.log(`  📊 总汇总报告 (从开始到现在的全部历史) - ${now.toLocaleString('zh-CN')}`);
        console.log('═'.repeat(60));
        console.log(`  模型/组合:  ${comboName}`);
        console.log(`  币对:       ${report.config.allowedMarkets.map((m: string) => `${m}/USDT`).join(', ')}`);
        console.log(`  统计周期:   ${firstTrade.toLocaleString('zh-CN')} ~ ${now.toLocaleString('zh-CN')}`);
        console.log(`  总交易:     ${report.summary.totalTrades} 笔 | 胜/负/待: ${report.summary.wins}/${report.summary.losses}/${report.summary.pending}`);
        console.log(`  胜率:       ${report.summary.winRate}% | 总盈亏: $${report.summary.totalPnL >= 0 ? '+' : ''}${report.summary.totalPnL} | 资金: $${report.summary.currentCapital}`);
        console.log(`  📁 ${modeBase}.json / ${modeBase}.txt (mode)`);
        console.log(`  📁 ${mergedBase}.json / ${mergedBase}.txt (compat merged)`);
        console.log('═'.repeat(60) + '\n');
        return modePaths.jsonPath;
    }

    /**
     * 生成每日报告：当天 0 点至今的交易合并，写 report_daily_YYYYMMDD，每天一份不覆盖
     */
    async generateDailyReport(): Promise<string | null> {
        const now = new Date();
        const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate(), 0, 0, 0, 0);
        const reportsDir = path.join(this.logsDirFull, 'reports');
        if (!fs.existsSync(reportsDir)) fs.mkdirSync(reportsDir, { recursive: true });
        const allTrades = this.readTradesFromFile(this.getModeTradesFilePath());
        const recentTrades = allTrades.filter((t) => new Date(t.timestamp) >= startOfToday);
        // 对于每日报告，传入所有交易用于计算今天开始时的资金
        const report = this.buildReportFromTrades(recentTrades, startOfToday, now, 'daily', allTrades) as any;
        // LIVE 模式：每日真钱展示同样使用官方可用现金余额。
        if (this.env.TRADING_MODE === TradingMode.LIVE) {
            const computedCurrentCapital = Math.round(Number(report.summary.currentCapital || 0) * 100) / 100;
            const walletCurrentCapital = Math.round(this.state.currentCapital * 100) / 100;
            const displayCapital = walletCurrentCapital;
            const dailyStartCap = report.summary.initialCapital as number;
            report.summary.walletCurrentCapital = walletCurrentCapital;
            report.summary.computedCurrentCapital = computedCurrentCapital;
            report.summary.capitalSource = 'live_wallet_balance_cash_api';
            report.summary.currentCapital = displayCapital;
            report.summary.capitalChange = Math.round((displayCapital - dailyStartCap) * 100) / 100;
            report.summary.strategyLogicalPnlUsd = Math.round(Number(report.summary.totalPnL || 0) * 100) / 100;
            const publicTrades = await this.fetchPublicActivityTrades();
            this.annotateLiveReportLastAttemptFromCanonicalRows(report, recentTrades, publicTrades);
        }
        this.annotateRuntimeReportMetadata(report, 'daily', this.modeScope);
        const dateStr = `${now.getFullYear()}${String(now.getMonth() + 1).padStart(2, '0')}${String(now.getDate()).padStart(2, '0')}`;
        const modeBase = `report_daily_${dateStr}.${this.modeScope}`;
        const modePaths = this.writeReportArtifactsAtomically(
            path.join(reportsDir, modeBase),
            report,
            path.basename(this.logger.getLogFilePath()),
        );

        const mergedTrades = this.readTradesFromFile(this.getMergedTradesFilePath());
        const mergedRecentTrades = mergedTrades.filter((t) => new Date(t.timestamp) >= startOfToday);
        const mergedDailyReport = this.env.TRADING_MODE === TradingMode.LIVE
            ? JSON.parse(JSON.stringify(report))
            : this.buildReportFromTrades(mergedRecentTrades, startOfToday, now, 'daily', mergedTrades) as any;
        this.annotateRuntimeReportMetadata(mergedDailyReport, 'daily', 'merged');
        const mergedBase = `report_daily_${dateStr}`;
        this.writeReportArtifactsAtomically(
            path.join(reportsDir, mergedBase),
            mergedDailyReport,
            'prediction_trades.json',
        );
        const comboName = this.getComboDisplayName();
        console.log('\n' + '═'.repeat(60));
        console.log(`  📊 每日报告 (当天汇总) - ${now.toLocaleString('zh-CN')}`);
        console.log('═'.repeat(60));
        console.log(`  模型/组合:  ${comboName}`);
        console.log(`  币对:       ${report.config.allowedMarkets.map((m: string) => `${m}/USDT`).join(', ')}`);
        console.log(`  总交易:     ${report.summary.totalTrades} 笔 | 胜/负/待: ${report.summary.wins}/${report.summary.losses}/${report.summary.pending}`);
        console.log(`  胜率:       ${report.summary.winRate}% | 总盈亏: $${report.summary.totalPnL >= 0 ? '+' : ''}${report.summary.totalPnL} | 资金: $${report.summary.currentCapital}`);
        console.log(`  📁 ${modeBase}.json / ${modeBase}.txt (mode)`);
        console.log(`  📁 ${mergedBase}.json / ${mergedBase}.txt (compat merged)`);
        console.log('═'.repeat(60) + '\n');
        return modePaths.jsonPath;
    }

    /**
     * 同时生成每日报告和汇总报告（调度入口）
     */
    async generateReports(): Promise<void> {
        console.log('\n' + '═'.repeat(60));
        console.log(`  📊 生成报告 - ${new Date().toLocaleString('zh-CN')}`);
        console.log('═'.repeat(60));
        if (this.env.TRADING_MODE === TradingMode.LIVE) {
            const reconciliation = await this.reconcilePendingLiveTrades();
            if (reconciliation.reconciledCount > 0) {
                console.log(`  🔄 LIVE 成交回写已对齐: reconciled=${reconciliation.reconciledCount}, pending_after=${reconciliation.pendingAfter}, external_trades=${reconciliation.externalTradeCount}`);
            }
            const executedReconciliation = await this.reconcileExecutedLiveTradesWithPublicActivity();
            if (executedReconciliation.reconciledCount > 0) {
                console.log(`  🔁 LIVE 已执行成交已按公共 activity 纠偏: reconciled=${executedReconciliation.reconciledCount}, external_trades=${executedReconciliation.externalTradeCount}`);
            }
            const ledgerRepair = this.rebuildLivePendingOrderLedgerFromTrades();
            if (ledgerRepair.rewritten) {
                console.log(`  🧾 LIVE 挂单账本已重建: events=${ledgerRepair.eventCount}`);
            }
        }
        const settlement = await this.settleHistory({ skipReportRefresh: true });
        if (settlement.settledCount > 0) {
            console.log(`  ✅ 报告前结算推进完成: settled=${settlement.settledCount}, pnl=${settlement.totalPnL >= 0 ? '+' : ''}$${settlement.totalPnL.toFixed(2)}`);
        }
        if (this.env.TRADING_MODE !== TradingMode.LIVE) {
            // 报告前再次重锚，保证 mode report 与交易日志/ledger 金额路径一致。
            this._recomputeCapitalFromHistory();
        }
        await this.generateDailyReport();
        await this.generateSummaryReport();
    }
    
    /**
     * 格式化报告为可读文本
     */
    /**
     * 从 LOGS_DIR 得到展示用组合名，并附带真实传递的超参信息。
     * 例：超参_GRU_BTC_57_NO1H4H(Δ59,p=8/2/10/10)
     * 非超参：ETH_92，XRP_20_80
     */
    getComboDisplayName(): string {
        const raw = this.logsDir.replace(/^logs_?/, '');
        const parts = raw.split('_').filter(Boolean);
        const base = parts.map((p) => (/^[a-z]+$/i.test(p) ? p.toUpperCase() : p)).join('_') || raw;
        const hasDigit = /\d/.test(base);
        const dualThreshold = this.env.PROB_THRESHOLD_UP != null && this.env.PROB_THRESHOLD_DOWN != null;

        // 如果是超参组合，附加真实传入的阈值和档位信息
        const isTuned = raw.includes('超参');
        if (isTuned) {
            const tiered = this.env.TIER_P1_PCT != null && this.env.TIER_P2_PCT != null
                && this.env.TIER_P3_PCT != null && this.env.TIER_P4_PCT != null;
            const thrPart = dualThreshold
                ? `UP${Math.round(this.env.PROB_THRESHOLD_UP! * 100)}/DN${Math.round((1 - this.env.PROB_THRESHOLD_DOWN!) * 100)}`
                : `Δ${Math.round(this.env.PROB_THRESHOLD * 100)}`;
            const tierPart = tiered
                ? `,p=${this.env.TIER_P1_PCT}/${this.env.TIER_P2_PCT}/${this.env.TIER_P3_PCT}/${this.env.TIER_P4_PCT}`
                : '';
            return `超参_${base.replace(/_超参/i, '')}(${thrPart}${tierPart})`;
        }

        if (!hasDigit && !dualThreshold) {
            const pct = Math.round(this.env.PROB_THRESHOLD * 100);
            return `${base}_${pct}`;
        }
        return base;
    }

    private formatReportToText(report: any, tradesFileNameOverride?: string): string {
        const lines: string[] = [];
        const tradesFileName = tradesFileNameOverride || path.basename(this.logger.getLogFilePath());
        const titleMap: Record<string, string> = {
            'daily': '每日报告',
            'summary': '总汇总报告（全部历史）'
        };
        const title = titleMap[report.reportType] || '报告';
        const comboName = this.getComboDisplayName();
        lines.push('═'.repeat(60));
        lines.push(`  POLYMARKET 预测交易 - ${title}`);
        lines.push('═'.repeat(60));
        lines.push('');
        lines.push(`报告时间: ${report.reportDate}`);
        lines.push(`模型/组合: ${comboName}`);
        if (report.reportType === 'summary') {
            lines.push(`统计周期: ${report.reportPeriod.start} ~ ${report.reportPeriod.end} (从开始到现在的全部历史)`);
        } else {
            lines.push(`统计周期: ${report.reportPeriod.start} ~ ${report.reportPeriod.end}`);
        }
        lines.push('');
        
        lines.push('─'.repeat(60));
        lines.push('  综合统计');
        lines.push('─'.repeat(60));
        lines.push(`  币对:         ${report.config.allowedMarkets.map((m: string) => `${m}/USDT`).join(', ')}`);
        lines.push(`  总逻辑单:     ${report.summary.totalTrades}`);
        lines.push(`  总成交fills:  ${report.summary.totalFills ?? report.summary.fillCount ?? (Array.isArray(report.trades) ? report.trades.length : 0)}`);
        lines.push(`  下单方向:     Up: ${report.summary.upCount ?? 0} 笔, Down: ${report.summary.downCount ?? 0} 笔`);
        lines.push(`  已完成:       ${report.summary.completedTrades}`);
        lines.push(`  胜利:         ${report.summary.wins}`);
        lines.push(`  失败:         ${report.summary.losses}`);
        lines.push(`  待结算:       ${report.summary.pending}（来自本组合目录 ${this.logsDir}/${tradesFileName}，已执行等待市场出结果）`);
        lines.push(`  已结算逻辑单: ${report.summary.settledLogicalTrades ?? report.summary.completedTrades ?? 0}`);
        lines.push(`  待 Claim:     ${report.summary.claimReadyLogicalTrades ?? 0} | 已 Claim: ${report.summary.claimedLogicalTrades ?? 0}`);
        lines.push(`  结算阶段:     ${report.summary.resolutionStatus ?? '-'} / ${report.summary.settlementStatus ?? '-'} / ${report.summary.claimStatus ?? '-'}`);
        lines.push(`  主口径说明:   交易数/胜负按逻辑单；PnL 为全部 fills 聚合净值；下方另列逻辑单摘要与分笔成交明细`);
        if (report.summary.pending > 0 && report.logicalTrades && Array.isArray(report.logicalTrades)) {
            const pendingList = report.logicalTrades.filter((t: any) => t.result === 'pending' || (t.result !== 'win' && t.result !== 'lose'));
            for (const t of pendingList) {
                const time = new Date(t.lastTimestamp || t.firstTimestamp || t.timestamp).toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false });
                const dir = t.direction === 'UP' ? 'UP' : 'DOWN';
                lines.push(`    其中: ${time} ${t.symbol} ${dir} ${(t.totalAmount ?? t.amount ?? 0).toFixed(0)} USDC`);
            }
        }
        lines.push(`  胜率:         ${report.summary.winRate}%`);
        lines.push(`  总盈亏:       $${report.summary.totalPnL}`);
        lines.push(`  平均置信度:   ${report.summary.avgConfidence}%`);
        lines.push(`  初始资金:     $${report.summary.initialCapital}`);
        lines.push(`  当前资金:     $${report.summary.currentCapital}`);
        lines.push(`  资金变化:     $${report.summary.capitalChange >= 0 ? '+' : ''}${report.summary.capitalChange}`);
        lines.push(`  （资金与盈亏均根据本组合目录 ${this.logsDir}/${tradesFileName} 计算，与其它组合/预测共享无关）`);
        lines.push('');
        
        lines.push('─'.repeat(60));
        lines.push('  各币种统计');
        lines.push('─'.repeat(60));
        for (const [symbol, stats] of Object.entries(report.bySymbol) as [string, any][]) {
            lines.push(`  ${symbol}:`);
            lines.push(`    逻辑单: ${stats.trades}, fills: ${stats.fillCount ?? 0}, 胜率: ${stats.winRate}%, 盈亏: $${stats.pnl}`);
            lines.push(`    下单: Up ${stats.upCount ?? 0} 笔, Down ${stats.downCount ?? 0} 笔 | 胜/负/待: ${stats.wins}/${stats.losses}/${stats.pending}, 平均置信度: ${stats.avgConfidence}%`);
        }
        lines.push('');

        if (report.logicalTrades && Array.isArray(report.logicalTrades) && report.logicalTrades.length > 0) {
            lines.push('─'.repeat(60));
            lines.push('  逻辑单摘要');
            lines.push('─'.repeat(60));
            for (const trade of report.logicalTrades) {
                const time = new Date(trade.lastTimestamp || trade.firstTimestamp || trade.timestamp).toLocaleString('zh-CN');
                const dir = trade.direction === 'UP' ? '↑' : '↓';
                const result = trade.result === 'win' ? '✓' : trade.result === 'lose' ? '✗' : '?';
                const pnl = trade.totalPnL !== undefined ? `$${trade.totalPnL >= 0 ? '+' : ''}${Number(trade.totalPnL).toFixed(2)}` : '-';
                lines.push(`  [${time}] ${trade.symbol} ${dir} fills=${trade.fillCount ?? 0} pending=${trade.pendingOrderCount ?? 0} $${Number(trade.totalAmount ?? 0).toFixed(2)} → ${result} ${pnl} (${trade.resolutionStatus ?? '-'} / ${trade.settlementStatus ?? '-'} / ${trade.claimStatus ?? '-'})`);
            }
            lines.push('');
        }
        
        // 每日胜率统计
        if (report.dailyStats && Object.keys(report.dailyStats).length > 0) {
            lines.push('─'.repeat(60));
            lines.push('  每日胜率统计');
            lines.push('─'.repeat(60));
            const sortedDates = Object.keys(report.dailyStats).sort();
            for (const dateKey of sortedDates) {
                const day = report.dailyStats[dateKey];
                const dateObj = new Date(dateKey + 'T00:00:00');
                const dateStr = dateObj.toLocaleDateString('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit' });
                lines.push(`  ${dateStr}:`);
                lines.push(`    逻辑单: ${day.trades}, fills: ${day.fillCount ?? 0}, 胜率: ${day.winRate}%, 盈亏: $${day.pnl >= 0 ? '+' : ''}${day.pnl}`);
            }
            lines.push('');
        }
        
        lines.push('─'.repeat(60));
        lines.push('  配置信息');
        lines.push('─'.repeat(60));
        lines.push(`  交易模式:     ${report.config.tradingMode}`);
        lines.push(`  初始资金:     $${report.config.initialCapital}`);
        lines.push(`  下注比例:     ${report.config.betSizePercent}`);
        lines.push(`  置信度阈值:   ${report.config.probThreshold}`);
        lines.push(`  最小利润比:   ${report.config.minProfitRatio ?? 0.4}`);
        lines.push(`  允许市场:     ${report.config.allowedMarkets.join(', ')}`);
        lines.push('');
        
        lines.push('─'.repeat(60));
        lines.push('  分笔成交记录');
        lines.push('─'.repeat(60));
        for (const trade of report.trades) {
            const time = new Date(trade.timestamp).toLocaleString('zh-CN');
            const dir = trade.direction === 'UP' ? '↑' : '↓';
            const result = trade.result === 'win' ? '✓' : trade.result === 'lose' ? '✗' : '?';
            const pnlValue = Number(trade?.pnl);
            const confValue = Number(trade?.confidence);
            const amountValue = Number(trade?.amount);
            const pnl = Number.isFinite(pnlValue) ? `$${pnlValue >= 0 ? '+' : ''}${pnlValue.toFixed(2)}` : '-';
            const conf = Number.isFinite(confValue) ? `${(confValue * 100).toFixed(1)}%` : '-';
            const amount = Number.isFinite(amountValue) ? `$${amountValue.toFixed(2)}` : '-';
            lines.push(`  [${time}] ${trade.symbol} ${dir} ${conf} ${amount} → ${result} ${pnl}`);
        }
        lines.push('');
        
        lines.push('═'.repeat(60));
        lines.push('  报告结束');
        lines.push('═'.repeat(60));
        
        return lines.join('\n');
    }
}

// ============================================================
// 默认导出
// ============================================================

export default PredictionExecutor;
