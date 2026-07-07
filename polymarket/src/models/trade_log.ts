/**
 * 简化交易日志
 * 记录预测交易的执行情况，保留全部历史数据
 */

import * as fs from 'fs';
import * as path from 'path';

// ============================================================
// 类型定义
// ============================================================

export interface TradeLogEntry {
    id: string;                    // 唯一 ID
    timestamp: string;             // ISO 时间戳
    symbol: string;                // 标的（BTC, ETH, SOL）
    direction: 'UP' | 'DOWN';      // 预测方向
    confidence: number;            // 置信度
    amount: number;                // 下注金额（USDC）
    mode: 'backtest' | 'simulation' | 'live';  // 交易模式
    status: 'pending' | 'executed' | 'failed'; // 状态
    marketTitle?: string;          // Polymarket 市场标题
    conditionId?: string;          // 市场 condition ID
    tokenId?: string;              // 购买的 token ID
    txHash?: string;               // 交易哈希（仅 live 模式）
    orderId?: string;              // CLOB order id（live GTD 挂单/成交跟踪）
    transactionHash?: string;      // 链上成交 tx hash（与 orderId 分离，避免挂单/成交混淆）
    error?: string;                // 错误信息
    result?: 'win' | 'lose' | 'pending';  // 结果
    provisionalResult?: 'win' | 'lose';   // 市场已结束且输赢客观确定后的预结算结果
    provisionalEvaluatedAt?: string;      // 预结算结果最后一次确认时间
    cooldownTriggeredStage?: 'provisional' | 'formal' | 'formal_expired_backfill'; // 该笔是否已作为冷却触发单
    cooldownScope?: 'symbol' | 'symbol_direction'; // 冷却作用域：总链按币种，或按币种+方向
    cooldownKey?: string;              // 实际写入冷却表的 key，便于审计
    formalResult?: 'win' | 'lose';     // 正式结算结果镜像，避免和预结算口径混淆
    formalConsecutiveLosses?: number;  // 该笔处理后的正式连输数
    provisionalConsecutiveLosses?: number; // 该笔处理后的预结算连输数
    pnl?: number;                  // 盈亏（USDC）

    // 结算相关字段
    marketSlug?: string;           // 市场 slug（用于查询结算结果）
    tokenOutcome?: string;         // 购买的代币结果 (Up/Down)
    tokenPrice?: number;           // 购买时的代币价格（模拟时为加权平均成交价）
    avgActualFillPrice?: number;   // 实际加权成交均价（与 tokenPrice 等价，单独持久化便于审计）
    limitPriceConfigured?: number; // 配置的限价（审计用：对比 tokenPrice 与原始限价）
    requestedAmountUsd?: number;   // 挂单/计划下单的目标金额（USDC）
    pendingAmountUsd?: number;     // 仍在挂单中的未成交金额（USDC）
    requestedShares?: number;      // 计划下单的 token 数量
    originalRequestedShares?: number; // 执行约束前模型原始请求的 token 数量（用于 >5 shares 保真审计）
    matchedShares?: number;        // 已确认成交的 token 数量
    liveOrderStatus?: 'posted_pending' | 'official_open' | 'official_canceled_zero_fill' | 'official_canceled_partial_fill' | 'official_missing_after_post' | 'partially_filled' | 'filled' | 'rejected' | 'expired' | 'cancelled'; // live 订单生命周期
    fillSource?: string;           // live 成交确认来源（WS/REST/public_activity）
    lastRejectionReason?: string;  // 最近一次拒单原因（如 region/min size）
    familyRuleVersion?: string;    // lowprice family rule version（便于 compare/live 激活审计）
    lowpriceFamilyId?: string;     // lowprice family id
    stalePolicyDecision?: string;  // stale-order policy 结果（off/cancelled/kept）
    staleCancelReason?: string;    // stale-order 触发原因
    exchangeMinOrderSizeShares?: number; // 交易所最小下单 shares（官方 book / live reject 学到的约束）
    attemptedRungs?: number;       // 本批原始计划档数
    compressedRungs?: number;      // 动态梯子因预算约束被压缩掉的档数（兼容 = droppedDynamicRungs）
    postedRungs?: number;          // 本批实际已提交档数
    filledRungs?: number;          // 本批实际成交档数
    exchangeRejectedRungs?: number; // 本批被交易所拒绝的档数
    upliftedDynamicRungs?: number; // 动态梯子中被强提到 min shares 的档数
    keptDynamicRungs?: number;     // 动态梯子最终保留并进入执行的档数
    droppedDynamicRungs?: number;  // 动态梯子因预算/风控被裁掉的档数
    minExecutableNotionalUsd?: number; // 动态梯子当前档满足 min shares 的最低金额
    redistributedNotionalUsd?: number; // 动态梯子在最低金额之上分回的剩余预算
    upliftedFixedOrders?: number;  // 固定价单是否被强提到 min shares（0/1）
    upliftedNotionalUsd?: number;  // 因 min shares uplift 额外增加的金额
    executionDecision?: string;    // 最终执行判定（阈值/风控/min-shares/uplift/post/executed 等）
    requestCompressionMode?: string; // 请求层压缩模式
    executionCompressionMode?: string; // 执行层压缩模式
    targetNotionalUsd?: number;    // 该笔单 sizing 合同目标金额
    requestedNotionalUsd?: number; // 请求层最终金额
    postedNotionalUsd?: number;    // 实际挂单总金额
    filledNotionalUsd?: number;    // 实际成交总金额
    requestLayerCompressionUsd?: number; // 第一层压缩金额
    executionLayerCompressionUsd?: number; // 第二层压缩金额
    executionLayerCompressionReason?: string; // 第二层压缩或重分配原因
    executionProfileVersion?: string; // 执行规则版本（v1/v2）
    executionState?: string;       // Execution v2 状态机阶段
    chaseRound?: number;           // 追价轮次（0=未追价）
    chaseTickLift?: number;        // 当前追价上移 tick 数
    chaseWaitSec?: number;         // 当前追价等待秒数
    firstChaseWaitSec?: number;    // 首次追价等待秒数
    nextChaseGapSec?: number;      // 后续追价值之间的最小间隔
    chaseTriggered?: boolean;      // 是否触发过追价
    finalAggressiveTriggered?: boolean; // 是否触发过最后窗口积极成交
    chaseAbsoluteMaxPrice?: number; // 追价绝对价格上界
    finalAggressiveWindowSec?: number; // 最后窗口秒数
    finalAggressiveMaxPrice?: number; // 最后窗口最高价格
    cancelOnAdverseMoveTicks?: number; // 反向 drift 取消阈值
    cancelOnSignalInvalidation?: boolean; // 方向失效即取消
    cancelOnQueueStall?: boolean;  // 排队停滞时是否撤单
    queueStallWindowSec?: number;  // 排队停滞判定窗口
    staleCancelMinAgeSec?: number; // 旧挂单最小年龄门槛
    staleCancelMinTimeToExpirySec?: number; // 临近到期前的最小撤单窗口
    lateFillCancelMinutes?: number; // 挂单满 N 分钟后取消未成交剩余部分
    gt5PreservationStatus?: string; // >5 shares 合同状态（exact_preservation/compressed_but_not_floored/not_applicable）
    compressionReason?: string;    // shares 变化原因（min_shares_floor/budget_compression）
    virtualRungCount?: number;     // 研究虚拟档数（方向梯子投影前）
    materializedLimitPriceLadder?: number[]; // 最终真实可执行价格梯子（投影/合并后）
    collapsedSamePriceRungs?: number; // 同价 surviving 档被合并掉的数量
    gt5AutoReduceTriggered?: boolean; // 是否发生 >5 shares 约束下的自动减档加厚
    runtimeRiskScalingState?: string; // 当前使用常规或保守比例的运行时状态
    directionalSizingSource?: string; // 当前金额来源：方向字段或旧全局字段
    sizingMode?: string;              // 当前金额模式（fixed_usd / symmetric_dynamic_pct / percent_cap）
    bankrollPct?: number;             // 当前基础百分比（小数，如 0.03）
    confidenceBandScale?: number;     // 当前命中的信心分档倍数
    effectiveSizingPct?: number;      // 实际生效总资金百分比（小数，如 0.07）
    currentCapital?: number;          // 计算该笔金额时使用的 currentCapital
    lastNonExecutionReason?: string; // 最近一次未执行原因（统一给 summary/monitor 消费）
    bestAskAtEntry?: number;       // 入场时订单簿最优卖价
    queueCompetitionUsdAtEntry?: number;   // 入场时同价及更优买单竞争深度（USD）
    effectiveFillableUsdAtEntry?: number;  // 扣除竞争后的可成交金额（haircut 前）
    settledAt?: string;            // 结算时间
    stoppedOut?: boolean;          // 是否为止损卖出（单笔价格下跌触发）
}

export type TradeLogModeScope = TradeLogEntry['mode'];

export interface TradeLogSummary {
    primaryTradeUnit: 'logical_order';
    totalTrades: number;
    totalFills: number;
    todayTrades: number;
    wins: number;
    losses: number;
    pending: number;
    totalPnL: number;
    fillPnL: number;
    winRate: number;
    bySymbol: Record<string, {
        trades: number;
        fillCount: number;
        wins: number;
        losses: number;  // 该符号的亏损次数
        winRate: number;
        pnl: number;
    }>;
}

// ============================================================
// 常量
// ============================================================

const LOG_RETENTION_DAYS = 99999;
const MERGED_LOG_FILE_NAME = 'prediction_trades.json';
const MODE_LOG_FILE_NAMES: Record<TradeLogModeScope, string> = {
    simulation: 'prediction_trades.simulation.json',
    live: 'prediction_trades.live.json',
    backtest: 'prediction_trades.backtest.json',
};
const MODE_SCOPES: TradeLogModeScope[] = ['simulation', 'live', 'backtest'];

// ============================================================
// 辅助函数
// ============================================================

function normalizeModeScope(modeScope?: string): TradeLogModeScope {
    const normalized = String(modeScope || 'simulation').toLowerCase();
    if (normalized === 'simulation' || normalized === 'live' || normalized === 'backtest') {
        return normalized;
    }
    return 'simulation';
}

function parseTradeList(raw: string): TradeLogEntry[] {
    try {
        const parsed = JSON.parse(raw);
        return Array.isArray(parsed) ? (parsed as TradeLogEntry[]) : [];
    } catch {
        return [];
    }
}

// ============================================================
// TradeLogger 类
// ============================================================

export class TradeLogger {
    private readonly modeScope: TradeLogModeScope;
    private readonly logDir: string;
    private readonly mergedLogFilePath: string;
    private logFilePath: string;
    private entries: TradeLogEntry[] = [];
    /** 本进程本次运行内已成交的 conditionId（重启后清空，避免用历史文件阻止本轮下单） */
    private executedInThisSession: Set<string> = new Set();

    constructor(logDir?: string, modeScope: TradeLogModeScope = 'simulation') {
        const dir = logDir || path.join(process.cwd(), 'logs');
        this.modeScope = normalizeModeScope(modeScope);
        this.logDir = dir;

        // 确保日志目录存在
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }

        // 确保报告子目录存在
        const reportsDir = path.join(dir, 'reports');
        if (!fs.existsSync(reportsDir)) {
            fs.mkdirSync(reportsDir, { recursive: true });
        }

        this.logFilePath = path.join(dir, MODE_LOG_FILE_NAMES[this.modeScope]);
        this.mergedLogFilePath = path.join(dir, MERGED_LOG_FILE_NAME);
        this.ensureModeScopedFiles();
        this.load();
        this.cleanup();
    }

    /**
     * 返回当前 logger 负责的 mode scope。
     */
    getModeScope(): TradeLogModeScope {
        return this.modeScope;
    }

    /**
     * 从文件加载日志（仅当前 mode scope）。
     */
    private load(): void {
        try {
            this.entries = this.readEntriesFromPath(this.logFilePath).map((entry) => ({
                ...entry,
                mode: normalizeModeScope(entry.mode || this.modeScope),
            }));
        } catch (error) {
            console.error('  [日志] 加载交易日志失败:', error);
            this.entries = [];
        }
    }

    /**
     * 如果只有旧的 prediction_trades.json，首次启动时自动拆分为三模式文件。
     * 然后重建兼容聚合文件 prediction_trades.json。
     */
    private ensureModeScopedFiles(): void {
        const hasAnyModeFile = MODE_SCOPES.some((mode) => {
            const fp = this.modeLogPath(mode);
            return fs.existsSync(fp);
        });

        if (hasAnyModeFile) {
            this.rebuildMergedLogFromModeFiles();
            return;
        }

        if (!fs.existsSync(this.mergedLogFilePath)) {
            return;
        }

        try {
            const legacyEntries = this.readEntriesFromPath(this.mergedLogFilePath);
            const grouped: Record<TradeLogModeScope, TradeLogEntry[]> = {
                simulation: [],
                live: [],
                backtest: [],
            };

            for (const entry of legacyEntries) {
                const mode = normalizeModeScope(entry.mode);
                grouped[mode].push({ ...entry, mode });
            }

            for (const mode of MODE_SCOPES) {
                this.writeEntriesToPath(this.modeLogPath(mode), grouped[mode]);
            }
            this.rebuildMergedLogFromModeFiles();
        } catch (error) {
            console.error('  [日志] 旧格式迁移失败，继续使用空模式文件:', error);
            for (const mode of MODE_SCOPES) {
                const fp = this.modeLogPath(mode);
                if (!fs.existsSync(fp)) {
                    this.writeEntriesToPath(fp, []);
                }
            }
            this.rebuildMergedLogFromModeFiles();
        }
    }

    private modeLogPath(mode: TradeLogModeScope): string {
        return path.join(this.logDir, MODE_LOG_FILE_NAMES[mode]);
    }

    private readEntriesFromPath(filePath: string): TradeLogEntry[] {
        if (!fs.existsSync(filePath)) {
            return [];
        }
        const raw = fs.readFileSync(filePath, 'utf-8');
        return parseTradeList(raw);
    }

    private writeEntriesToPath(filePath: string, entries: TradeLogEntry[]): void {
        fs.writeFileSync(filePath, JSON.stringify(entries, null, 2), 'utf-8');
    }

    private sortEntriesByTimestamp(entries: TradeLogEntry[]): TradeLogEntry[] {
        return [...entries].sort((a, b) => {
            const ta = new Date(a.timestamp).getTime();
            const tb = new Date(b.timestamp).getTime();
            if (ta !== tb) return ta - tb;
            return (a.id || '').localeCompare(b.id || '');
        });
    }

    private readEntriesByMode(mode: TradeLogModeScope): TradeLogEntry[] {
        return this.readEntriesFromPath(this.modeLogPath(mode));
    }

    private rebuildMergedLogFromModeFiles(): void {
        try {
            const merged = MODE_SCOPES.flatMap((mode) => this.readEntriesByMode(mode));
            const sorted = this.sortEntriesByTimestamp(
                merged.map((entry) => ({ ...entry, mode: normalizeModeScope(entry.mode) }))
            );
            this.writeEntriesToPath(this.mergedLogFilePath, sorted);
        } catch (error) {
            console.error('  [日志] 重建兼容聚合日志失败:', error);
        }
    }

    /**
     * 保存日志到文件（当前 mode 文件 + 兼容聚合文件）。
     */
    private save(): void {
        try {
            this.writeEntriesToPath(this.logFilePath, this.entries);
            this.rebuildMergedLogFromModeFiles();
        } catch (error) {
            console.error('  [日志] 保存交易日志失败:', error);
        }
    }

    /**
     * 清理过期记录（保留天数 = LOG_RETENTION_DAYS，设 99999 等于永不清理）
     */
    private cleanup(): void {
        const cutoffDate = new Date();
        cutoffDate.setDate(cutoffDate.getDate() - LOG_RETENTION_DAYS);

        const beforeCount = this.entries.length;
        this.entries = this.entries.filter((entry) => {
            const entryDate = new Date(entry.timestamp);
            return entryDate >= cutoffDate;
        });

        const removedCount = beforeCount - this.entries.length;
        if (removedCount > 0) {
            console.log(`  [日志] 已清理 ${removedCount} 条过期记录 (超过 ${LOG_RETENTION_DAYS} 天)`);
            this.save();
        }
    }

    /**
     * 生成唯一 ID
     */
    private generateId(): string {
        return `trade_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    }

    /**
     * 添加新的交易记录
     */
    addEntry(entry: Omit<TradeLogEntry, 'id' | 'timestamp'>): TradeLogEntry {
        const fullEntry: TradeLogEntry = {
            id: this.generateId(),
            timestamp: new Date().toISOString(),
            ...entry,
            mode: this.modeScope,
        };

        this.entries.push(fullEntry);
        this.save();

        return fullEntry;
    }

    /**
     * 更新交易记录
     */
    updateEntry(id: string, updates: Partial<TradeLogEntry>): TradeLogEntry | null {
        const index = this.entries.findIndex((e) => e.id === id);

        if (index === -1) {
            return null;
        }

        this.entries[index] = {
            ...this.entries[index],
            ...updates,
            mode: this.modeScope,
        };

        // 本进程内标记为已成交的，加入 executedInThisSession，用于 hasAlreadyBought 只认“本轮”
        if (updates.status === 'executed') {
            // 优先使用 updates 中的 conditionId（如有），确保同时更新 status 和 conditionId 时不会遗漏
            const conditionId = updates.conditionId || this.entries[index].conditionId;
            if (conditionId) {
                this.executedInThisSession.add(conditionId);
            }
        }

        this.save();
        return this.entries[index];
    }

    /**
     * 批量更新多条交易记录，并只落盘一次。
     * 返回实际更新的记录数。
     */
    updateEntries(ids: string[], updates: Partial<TradeLogEntry>): number {
        if (!Array.isArray(ids) || ids.length === 0) {
            return 0;
        }

        const idSet = new Set(ids.filter((id) => typeof id === 'string' && id.trim().length > 0));
        if (idSet.size === 0) {
            return 0;
        }

        let updated = 0;
        for (let i = 0; i < this.entries.length; i++) {
            const entry = this.entries[i];
            if (!idSet.has(entry.id)) continue;
            this.entries[i] = {
                ...entry,
                ...updates,
                mode: this.modeScope,
            };
            const nextStatus = updates.status ?? entry.status;
            const nextConditionId = updates.conditionId || entry.conditionId;
            if (nextStatus === 'executed' && nextConditionId) {
                this.executedInThisSession.add(nextConditionId);
            }
            updated += 1;
        }

        if (updated > 0) {
            this.save();
        }
        return updated;
    }

    /**
     * 获取指定 ID 的记录
     */
    getEntry(id: string): TradeLogEntry | undefined {
        return this.entries.find((e) => e.id === id);
    }

    /**
     * 获取所有记录（当前 mode scope）
     */
    getAllEntries(): TradeLogEntry[] {
        return [...this.entries];
    }

    /**
     * 获取日志文件路径（当前 mode scope 文件）
     */
    getLogFilePath(): string {
        return this.logFilePath;
    }

    /**
     * 获取今日交易记录
     */
    getTodayEntries(): TradeLogEntry[] {
        const today = new Date();
        today.setHours(0, 0, 0, 0);

        return this.entries.filter((entry) => {
            const entryDate = new Date(entry.timestamp);
            entryDate.setHours(0, 0, 0, 0);
            return entryDate.getTime() === today.getTime();
        });
    }

    /**
     * 获取今日交易数量
     */
    getTodayTradeCount(): number {
        return this.getTodayEntries().length;
    }

    /**
     * 获取最近 N 条记录
     */
    getRecentEntries(count: number = 10): TradeLogEntry[] {
        return this.entries.slice(-count);
    }

    /**
     * 是否已对该市场下过单：
     * 1. 先查本进程 session 内存集合（最快，无 I/O）
     * 2. 再查文件中近 20min 的 executed 记录（防重启后同一 15min 周期重复下单）
     * 20 分钟 > 15 分钟周期，覆盖完整周期 + 少量缓冲。
     * 超过 20 分钟的旧记录不阻塞新周期交易（Polymarket conditionId 每 15min 不同）。
     */
    hasAlreadyBought(conditionId: string): boolean {
        // 1. 先查本进程 session 内存集合（最快）
        if (this.executedInThisSession.has(conditionId)) return true;
        // 2. 防重启后同周期重复下单：检查文件中近 20 分钟内是否已有该 conditionId 的成交
        //    20 分钟 > 15 分钟周期，覆盖整个周期 + 少量缓冲
        const cutoff = Date.now() - 20 * 60 * 1000;
        return this.entries.some(
            (e) =>
                e.conditionId === conditionId &&
                e.status === 'executed' &&
                new Date(e.timestamp).getTime() >= cutoff
        );
    }

    /**
     * 日志文件中是否在近期对该市场有过 executed 记录（用于 Data API 失败时防重复买）
     * @param conditionId 市场 conditionId
     * @param withinMinutes 时间窗口（分钟），默认 30
     */
    hasRecentExecutedInLog(conditionId: string, withinMinutes: number = 30): boolean {
        const cutoff = Date.now() - withinMinutes * 60 * 1000;
        return this.entries.some(
            (e) =>
                e.conditionId === conditionId &&
                e.status === 'executed' &&
                new Date(e.timestamp).getTime() >= cutoff
        );
    }

    /**
     * 获取待结算的交易（已执行但未有结果的交易）
     * 条件：
     * - status === 'executed'
     * - 或 live 过程中虽然被标成 failed，但已经有真实成交金额/份额
     *   （例如订单路径返回失败，但 public activity / matchedShares 证明这笔其实成交了）
     * 注意：排除已止损卖出的交易（stoppedOut === true），避免重复结算
     */
    getPendingTrades(): TradeLogEntry[] {
        return this.entries.filter((entry) => {
            const hasExecutableFill =
                Number(entry.amount ?? 0) > 0 ||
                Number(entry.matchedShares ?? 0) > 0;
            const isSettledLikeExecuted =
                entry.status === 'executed' ||
                ((entry.status === 'failed' || entry.status === 'pending') && hasExecutableFill);
            return (
                isSettledLikeExecuted &&
                (entry.marketSlug || entry.conditionId) &&  // 至少需 slug 或 conditionId 才能查询结果
                (entry.result === undefined || entry.result === 'pending') &&
                !entry.stoppedOut  // 排除已止损卖出的交易
            );
        });
    }

    /**
     * 结算交易
     *
     * @param id 交易 ID
     * @param won 是否赢了
     * @param pnl 盈亏金额（正为盈利，负为亏损）
     * @param opts 可选，如 stoppedOut 表示为止损卖出
     */
    settleTrade(id: string, won: boolean, pnl: number, opts?: { stoppedOut?: boolean }): TradeLogEntry | null {
        const index = this.entries.findIndex((e) => e.id === id);

        if (index === -1) {
            return null;
        }

        const updates: Partial<TradeLogEntry> = {
            result: won ? 'win' : 'lose',
            pnl: pnl,
            settledAt: new Date().toISOString(),
        };
        if (opts?.stoppedOut) {
            updates.stoppedOut = true;
        }

        this.entries[index] = {
            ...this.entries[index],
            ...updates,
            mode: this.modeScope,
        };

        this.save();
        return this.entries[index];
    }

    /**
     * 获取日志摘要统计
     * 胜率按逻辑订单计算：同一 conditionId 的多笔成交（部分成交/补单）算一笔订单
     */
    getSummary(): TradeLogSummary {
        const todayEntries = this.getTodayEntries();
        const logicalKey = (e: TradeLogEntry) => e.conditionId || e.id;
        const groups = new Map<string, TradeLogEntry[]>();
        for (const entry of this.entries) {
            const key = logicalKey(entry);
            if (!groups.has(key)) groups.set(key, []);
            groups.get(key)!.push(entry);
        }

        let wins = 0, losses = 0, pending = 0, totalPnL = 0;
        const bySymbol: Record<string, { trades: number; fillCount: number; wins: number; losses: number; pnl: number }> = {};

        for (const entry of this.entries) {
            if (entry.pnl !== undefined) {
                totalPnL += entry.pnl;
                const sym = entry.symbol;
                if (!bySymbol[sym]) bySymbol[sym] = { trades: 0, fillCount: 0, wins: 0, losses: 0, pnl: 0 };
                bySymbol[sym].fillCount++;
                bySymbol[sym].pnl += entry.pnl;
            }
        }
        for (const group of groups.values()) {
            const first = group[0];
            const result = group.some(e => e.result === 'win') ? 'win' : group.some(e => e.result === 'lose') ? 'lose' : 'pending';
            if (result === 'win') wins++; else if (result === 'lose') losses++; else pending++;
            const sym = first.symbol;
            if (!bySymbol[sym]) bySymbol[sym] = { trades: 0, fillCount: 0, wins: 0, losses: 0, pnl: 0 };
            bySymbol[sym].trades++;
            if (result === 'win') bySymbol[sym].wins++; else if (result === 'lose') bySymbol[sym].losses++;
        }

        const completedTrades = wins + losses;
        const winRate = completedTrades > 0 ? wins / completedTrades : 0;
        const bySymbolWithRate: Record<string, any> = {};
        for (const [symbol, stats] of Object.entries(bySymbol)) {
            const symbolCompleted = stats.wins + stats.losses;
            bySymbolWithRate[symbol] = {
                ...stats,
                winRate: symbolCompleted > 0 ? stats.wins / symbolCompleted : 0,
            };
        }

        return {
            primaryTradeUnit: 'logical_order',
            totalTrades: groups.size,
            totalFills: this.entries.length,
            todayTrades: todayEntries.length,
            wins,
            losses,
            pending,
            totalPnL,
            fillPnL: totalPnL,
            winRate,
            bySymbol: bySymbolWithRate,
        };
    }

    /**
     * 打印日志摘要
     */
    printSummary(): void {
        const summary = this.getSummary();

        console.log(`\n📋 交易日志摘要 (mode=${this.modeScope}, 全部历史):`);
        console.log('═'.repeat(50));

        console.log(`  总逻辑单:   ${summary.totalTrades}`);
        console.log(`  总成交fills:${summary.totalFills}`);
        console.log(`  今日交易:   ${summary.todayTrades}`);
        console.log(`  盈利次数:   ${summary.wins}`);
        console.log(`  亏损次数:   ${summary.losses}`);
        console.log(`  待结算:     ${summary.pending}`);
        console.log(`  胜率:       ${(summary.winRate * 100).toFixed(1)}%`);
        console.log(`  总盈亏:     $${summary.totalPnL.toFixed(2)}`);

        if (Object.keys(summary.bySymbol).length > 0) {
            console.log('\n  按币种统计:');
            for (const [symbol, stats] of Object.entries(summary.bySymbol)) {
                console.log(`    ${symbol}: 逻辑单 ${stats.trades} 笔, fills ${stats.fillCount} 笔, 胜率 ${(stats.winRate * 100).toFixed(1)}%, 盈亏 $${stats.pnl.toFixed(2)}`);
            }
        }

        console.log('═'.repeat(50) + '\n');
    }

    /**
     * 打印最近的交易记录
     */
    printRecentTrades(count: number = 5): void {
        const recent = this.getRecentEntries(count);

        console.log(`\n📜 最近 ${count} 笔交易:`);
        console.log('─'.repeat(60));

        if (recent.length === 0) {
            console.log('  暂无交易记录。');
        } else {
            for (const entry of recent) {
                const time = new Date(entry.timestamp).toLocaleString('zh-CN');
                const dirIcon = entry.direction === 'UP' ? '📈' : '📉';
                const dirText = entry.direction === 'UP' ? '上涨' : '下跌';
                const resultIcon = entry.result === 'win' ? '✅'
                    : entry.result === 'lose' ? '❌' : '⏳';
                const resultText = entry.result === 'win' ? '盈利'
                    : entry.result === 'lose' ? '亏损' : '待结算';

                const modeText = entry.mode === 'live' ? '真实'
                    : entry.mode === 'simulation' ? '模拟' : '回测';

                const statusText = entry.status === 'executed' ? '已执行'
                    : entry.status === 'failed' ? '失败' : '待处理';

                console.log(`  ${time}`);
                console.log(`    ${entry.symbol} ${dirIcon} ${dirText} | $${entry.amount.toFixed(2)} | ${modeText}模式`);
                console.log(`    状态: ${statusText} | 结果: ${resultIcon} ${resultText}`);
                if (entry.pnl !== undefined) {
                    console.log(`    盈亏: $${entry.pnl.toFixed(2)}`);
                }
                console.log('');
            }
        }

        console.log('─'.repeat(60) + '\n');
    }
}

// ============================================================
// 单例导出（兼容单进程模式）
// ============================================================

let _instance: TradeLogger | null = null;
const _instanceByLogDirMode = new Map<string, TradeLogger>();

function normalizeLogDir(logDir?: string): string {
    const dir = logDir || path.join(process.cwd(), 'logs');
    return path.resolve(dir);
}

function buildInstanceKey(logDir?: string, modeScope?: TradeLogModeScope): string {
    return `${normalizeLogDir(logDir)}::${normalizeModeScope(modeScope)}`;
}

export const getTradeLogger = (logDir?: string, modeScope: TradeLogModeScope = 'simulation'): TradeLogger => {
    const normalizedMode = normalizeModeScope(modeScope);
    if (!logDir && normalizedMode === 'simulation') {
        if (!_instance) {
            _instance = new TradeLogger(logDir, normalizedMode);
        }
        return _instance;
    }

    const key = buildInstanceKey(logDir, normalizedMode);
    const existing = _instanceByLogDirMode.get(key);
    if (existing) return existing;
    const created = new TradeLogger(logDir, normalizedMode);
    _instanceByLogDirMode.set(key, created);
    return created;
};

/**
 * 多 trader 模式：按 logsDir + modeScope 复用同一实例，避免同目录同模式多实例互相覆盖写入。
 * 不同 logsDir 或不同 modeScope 仍完全隔离。
 */
export const createTradeLogger = (logDir: string, modeScope: TradeLogModeScope = 'simulation'): TradeLogger => {
    const normalizedMode = normalizeModeScope(modeScope);
    const key = buildInstanceKey(logDir, normalizedMode);
    const existing = _instanceByLogDirMode.get(key);
    if (existing) return existing;
    const created = new TradeLogger(logDir, normalizedMode);
    _instanceByLogDirMode.set(key, created);
    return created;
};

export default TradeLogger;
