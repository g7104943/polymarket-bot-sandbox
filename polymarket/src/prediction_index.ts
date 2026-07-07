/**
 * Polymarket 预测交易入口点
 * 
 * 功能：
 * - 每秒扫描 predictions.json，发现新预测且目标 15M 未结算则下单
 * - 按 target_period_end_ts 与 Python 对齐，Python 慢 1～2 分钟也能在结算前下单
 * - 已执行周期持久化到 LOGS_DIR/executed_target_ts.json，重启后不重复下单
 * - 优雅关闭处理
 * 
 * 使用方法：
 *   npm run predict-trade          # 模拟模式
 *   npm run predict-trade:live     # 真实交易模式
 */

import * as path from 'path';
import * as fs from 'fs';
import { execSync } from 'child_process';

// ── 系统代理自动检测 (macOS Clash/V2Ray) ──────────────────────
// Node.js 不读 macOS 系统代理配置，导致 axios 直连 Polymarket 超时。
// 在 import axios 的模块被加载之前设置 env，确保 axios 走代理。
if (!process.env.HTTPS_PROXY && !process.env.https_proxy) {
    try {
        const out = execSync(
            `python3 -c "import urllib.request; p=urllib.request.getproxies(); print(p.get('https','') or p.get('http',''))"`,
            { timeout: 3000, encoding: 'utf-8' },
        ).trim();
        if (out) {
            process.env.HTTPS_PROXY = out;
            process.env.HTTP_PROXY = out;
            console.log(`[Proxy] 检测到系统代理: ${out}`);
        }
    } catch { /* 无代理或检测失败，继续直连 */ }
}

import PredictionExecutor from './services/prediction_executor';
import { MarketOrderbookWS } from './services/market_orderbook_ws';
import { UserChannelWS } from './services/user_channel_ws';
import { PREDICTION_ENV, TradingMode } from './config/prediction_env';
import { readLocalPredictionsRaw, parseLocalPredictions, thresholdMet } from './utils/prediction_api';

// ============================================================
// 常量
// ============================================================

/** 扫描 predictions.json 的间隔（毫秒） */
const SCAN_INTERVAL_MS = 1000;

// 结算检查间隔（毫秒）- 每1分钟检查一次
const SETTLEMENT_CHECK_INTERVAL = 1 * 60 * 1000;

// 报告生成间隔：每小时（生成每日报告 + 更新汇总报告）
const REPORT_INTERVAL_MS = 60 * 60 * 1000;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** 已执行周期持久化文件名（放在 LOGS_DIR 下，每组合一份） */
const EXECUTED_PERIODS_FILE = 'executed_target_ts.json';
/** 持久化中保留的 ts 最老时间：仅保留最近 7 天，避免文件无限增长 */
const EXECUTED_PERIODS_RETENTION_SEC = 7 * 24 * 3600;

const getExecutedPeriodsFilePath = (): string =>
    path.join(process.cwd(), process.env.LOGS_DIR || 'logs', EXECUTED_PERIODS_FILE);

/**
 * 从磁盘加载已执行过的 target_period_end_ts，重启后避免对同一周期重复下单
 */
const loadExecutedTargetTss = (): Set<number> => {
    const filePath = getExecutedPeriodsFilePath();
    try {
        const data = fs.readFileSync(filePath, 'utf8');
        const arr = JSON.parse(data);
        const list = Array.isArray(arr) ? arr : [];
        const now = Math.floor(Date.now() / 1000);
        const minTs = now - EXECUTED_PERIODS_RETENTION_SEC;
        const pruned = list.filter((t: number) => t >= minTs);
        return new Set<number>(pruned);
    } catch {
        return new Set<number>();
    }
};

/**
 * 将当前已执行周期集合写回磁盘（写入前会剔除超过保留期的 ts）
 */
const saveExecutedTargetTss = (set: Set<number>): void => {
    const filePath = getExecutedPeriodsFilePath();
    const dir = path.dirname(filePath);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    const now = Math.floor(Date.now() / 1000);
    const minTs = now - EXECUTED_PERIODS_RETENTION_SEC;
    const arr = Array.from(set).filter((t) => t >= minTs).sort((a, b) => a - b);
    fs.writeFileSync(filePath, JSON.stringify(arr, null, 0), 'utf8');
};

type RoundResultLike = {
    success?: boolean;
    pendingMonitor?: boolean;
    skippedReason?: string;
};

function shouldConsumeTargetTs(results: RoundResultLike[] | undefined): boolean {
    if (!Array.isArray(results) || results.length === 0) return false;
    return results.some((r) => Boolean(r.success) || Boolean(r.pendingMonitor) || r.skippedReason === 'already_bought');
}

/**
 * 打印启动横幅
 */
const printBanner = (): void => {
    const version = '1.0.0';
    const banner = `
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║        🎯 POLYMARKET 预测交易系统 v${version}                    ║
║                                                                ║
║        基于 AI 的加密货币价格预测交易                           ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
`;
    console.log(banner);
};

/**
 * 打印模式警告
 */
const printModeWarning = (): void => {
    const mode = PREDICTION_ENV.TRADING_MODE;
    
    if (mode === TradingMode.LIVE) {
        console.log('\n' + '⚠️ '.repeat(20));
        console.log('');
        console.log('  🔴 真实交易模式已激活 🔴');
        console.log('');
        console.log('  将使用真实资金进行交易！');
        console.log('  请确保您了解相关风险。');
        console.log('');
        console.log('  10秒内按 Ctrl+C 可取消...');
        console.log('');
        console.log('⚠️ '.repeat(20) + '\n');
    } else if (mode === TradingMode.SIMULATION) {
        console.log('\n📋 模拟交易模式');
        console.log('   交易将被模拟，但不会在 Polymarket 上执行真实订单。');
        console.log('   适合测试策略和验证系统。\n');
    } else {
        console.log('\n📋 回测模式');
        console.log('   仅记录交易，不调用任何外部 API。');
        console.log('   用于历史数据回测。\n');
    }
};

// ============================================================
// 主函数
// ============================================================

const main = async (): Promise<void> => {
    // 打印横幅
    printBanner();
    
    // 打印模式警告
    printModeWarning();
    
    // 对于真实交易模式，等待用户确认
    if (PREDICTION_ENV.TRADING_MODE === TradingMode.LIVE) {
        console.log('10秒后开始...');
        await new Promise((resolve) => setTimeout(resolve, 10000));
    }
    
    // 创建执行器
    const executor = new PredictionExecutor();
    
    // 初始化
    console.log('\n🔧 正在初始化系统...\n');
    const initialized = await executor.initialize();
    
    if (!initialized) {
        console.error('\n❌ 初始化失败，程序退出。\n');
        process.exit(1);
    }
    
    const logsDir = process.env.LOGS_DIR || 'logs';
    const comboName = executor.getComboDisplayName();
    console.log(`📌 本进程 模型/组合: ${comboName}  (LOGS_DIR=${logsDir})\n`);
    
    const executedTargetTss = loadExecutedTargetTss();
    if (executedTargetTss.size > 0) {
        console.log(`  [已执行周期] 从磁盘恢复 ${executedTargetTss.size} 个周期，避免重复下单`);
    }
    
    console.log(`\n🔄 扫描 predictions.json (每 ${SCAN_INTERVAL_MS / 1000} 秒)`);
    const dualThreshold = PREDICTION_ENV.PROB_THRESHOLD_UP != null && PREDICTION_ENV.PROB_THRESHOLD_DOWN != null;
    const thresholdDesc = dualThreshold
        ? `UP≥${(PREDICTION_ENV.PROB_THRESHOLD_UP! * 100).toFixed(0)}%, DOWN>${((1 - PREDICTION_ENV.PROB_THRESHOLD_DOWN!) * 100).toFixed(0)}% (即prob<${(PREDICTION_ENV.PROB_THRESHOLD_DOWN! * 100).toFixed(0)}%)`
        : `≥${(PREDICTION_ENV.PROB_THRESHOLD * 100).toFixed(0)}%`;
    console.log(`   只做 2 件事：1) 读取 2) 按 target_period_end_ts 对应市场下单，confidence${thresholdDesc} 则下单\n`);
    
    let isShuttingDown = false;
    
    const shutdown = async () => {
        if (isShuttingDown) return;
        isShuttingDown = true;
        console.log('\n\n🛑 正在优雅关闭...\n');
        marketOrderbookWS.stop();
        if (userChannelWS) userChannelWS.stop();
        executor.stop();
        try {
            console.log('📊 生成最终报告...');
            await executor.generateReports();
        } catch (error) {
            console.error('最终报告生成失败:', error);
        }
        executor.printStatus();
        console.log('👋 再见！\n');
        process.exit(0);
    };
    
    process.on('SIGINT', shutdown);
    process.on('SIGTERM', shutdown);
    
    console.log(`📊 [${comboName}] 结算检查器 (间隔: 1分钟)`);
    setInterval(async () => {
        if (isShuttingDown) return;
        try { await executor.settleHistory(); } catch (e) { console.error(`[${comboName}] 结算检查失败:`, e); }
    }, SETTLEMENT_CHECK_INTERVAL);
    
    // LIVE 模式：每 10 秒从链上同步真实 pUSD 余额到 currentCapital，用于 5%/2.5% 等下注比例
    if (PREDICTION_ENV.TRADING_MODE === TradingMode.LIVE) {
        setInterval(async () => {
            if (isShuttingDown) return;
            try { await executor.refreshLiveCapital(); } catch { /* 静默 */ }
        }, 10 * 1000);
    }
    
    // 价格监控：轮询兜底 + WebSocket 实时 best_ask 触发（高价/低价看 best_ask 是否到阈值）
    const MONITOR_INTERVAL_MS = 1 * 1000;       // 高价/低价监控轮询间隔（1 秒；主触发仍为 WS best_ask）
    const LIQUIDITY_MONITOR_INTERVAL_MS = 10 * 1000;  // 流动性监控轮询间隔（WS 为主，REST 兜底降频避免 /book 限速）
    
    let marketOrderbookWS: MarketOrderbookWS = new MarketOrderbookWS({
        getSubscribedTokenIds: () => executor.getMonitorTokenIds(),
        onBestAsk: (assetId, bestAsk, asksSnapshot, isFullBook) => {
            // 有 asksSnapshot（book 消息）时传入，下单时跳过 REST getOrderBook，提速 ~100ms
            void executor.tryTriggerPriceMonitorByTokenId(assetId, bestAsk, asksSnapshot);
            void executor.tryTriggerLowPriceMonitorByTokenId(assetId, bestAsk, asksSnapshot);
            if (asksSnapshot && asksSnapshot.length > 0) {
                void executor.tryTriggerLiquidityMonitorByTokenId(assetId, bestAsk, asksSnapshot);
            }
        },
        verbose: false,
    });
    marketOrderbookWS.start();
    console.log(`📡 市场频道 WebSocket 已启动 (best_ask 实时触发，价格监控兜底间隔 ${MONITOR_INTERVAL_MS / 1000} 秒)\n`);

    // ── User Channel WebSocket（仅 LIVE 模式）───
    let userChannelWS: UserChannelWS | null = null;
    if (PREDICTION_ENV.TRADING_MODE === TradingMode.LIVE) {
        if (PREDICTION_ENV.POLY_API_KEY && PREDICTION_ENV.POLY_API_SECRET && PREDICTION_ENV.POLY_API_PASSPHRASE) {
            userChannelWS = new UserChannelWS({
                apiKey: PREDICTION_ENV.POLY_API_KEY,
                apiSecret: PREDICTION_ENV.POLY_API_SECRET,
                apiPassphrase: PREDICTION_ENV.POLY_API_PASSPHRASE,
                getSubscribedMarkets: () => executor.getLiveOrderConditionIds(),
                onTrade: (event) => { try { executor.onWsTrade(event as any); } catch { /* isolation */ } },
                onOrder: (event) => { try { executor.onWsOrder(event as any); } catch { /* isolation */ } },
                verbose: false,
            });
            userChannelWS.start();
            console.log(`📡 User Channel WebSocket 已启动 (LIVE 模式)\n`);
        } else {
            console.log('⚠️ LIVE 模式未拿到完整 POLY_API_* 凭证，User Channel WebSocket 未启动（将退回 REST 成交查询）');
        }
    }

    // 高价监控：轮询兜底 + WS 触发。当代币价格 > MAX_TOKEN_PRICE 时加入队列，等 best_ask ≤ MAX 再下单
    setInterval(async () => {
        if (isShuttingDown) return;
        marketOrderbookWS.syncSubscribe();
        if (userChannelWS) userChannelWS.syncSubscribe();
        try { 
            await executor.checkPriceMonitorQueue(); 
        } catch (e) { 
            if (e instanceof Error && (e.message.includes('timeout') || e.message.includes('ECONNREFUSED') || e.message.includes('network'))) {
                console.log(`\n🔄 高价监控API连接错误，正在重试...`);
            }
        }
    }, MONITOR_INTERVAL_MS);
    
    // 低价监控：轮询兜底 + WS 触发。当代币价格 < MIN_TOKEN_PRICE 时加入队列，等 best_ask ≥ MIN 再下单
    setInterval(async () => {
        if (isShuttingDown) return;
        try { 
            await executor.checkLowPriceMonitorQueue(); 
        } catch (e) { 
            if (e instanceof Error && (e.message.includes('timeout') || e.message.includes('ECONNREFUSED') || e.message.includes('network'))) {
                console.log(`\n🔄 低价监控API连接错误，正在重试...`);
            }
        }
    }, MONITOR_INTERVAL_MS);

    setInterval(async () => {
        if (isShuttingDown) return;
        try {
            await executor.checkStaleLiveOrderPolicy();
        } catch (e) {
            if (e instanceof Error && (e.message.includes('timeout') || e.message.includes('ECONNREFUSED') || e.message.includes('network'))) {
                console.log(`\n🔄 stale 挂单巡检 API 连接错误，正在重试...`);
            }
        }
    }, MONITOR_INTERVAL_MS);
    
    // 流动性监控：WS book 实时触发为主；REST 兜底每 10 秒（避免 /book 150 请求/10秒 限速）
    setInterval(async () => {
        if (isShuttingDown) return;
        try { 
            await executor.checkLiquidityMonitorQueue(); 
        } catch (e) { 
            if (e instanceof Error && (e.message.includes('timeout') || e.message.includes('ECONNREFUSED') || e.message.includes('network'))) {
                console.log(`\n🔄 流动性监控API连接错误，正在重试...`);
            }
        }
    }, LIQUIDITY_MONITOR_INTERVAL_MS);
    
    const stopLossIntervalMs = 30 * 1000;
    if (PREDICTION_ENV.STOP_LOSS_PCT != null && PREDICTION_ENV.STOP_LOSS_PCT > 0) {
        console.log(`🛡️  单笔止损参数已配置 (每 ${stopLossIntervalMs / 1000} 秒巡检): 当前版本不执行自动卖出，仅保留兼容`);
        setInterval(async () => {
            if (isShuttingDown) return;
            try { await executor.checkPerTradeStopLoss(); } catch (e) { /* ignore */ }
        }, stopLossIntervalMs);
    }
    
    console.log(`📋 报告: 每小时生成 (每日报告按天合并不覆盖，总汇总报告每小时更新)\n`);
    const firstReportDelay = REPORT_INTERVAL_MS;
    const scheduleReport = () => {
        setInterval(async () => {
            if (isShuttingDown) return;
            try { await executor.generateReports(); } catch (e) { console.error('报告生成失败:', e); }
        }, REPORT_INTERVAL_MS);
    };
    setTimeout(async () => {
        if (!isShuttingDown) {
            try { await executor.generateReports(); } catch (e) { console.error('首次报告失败:', e); }
            scheduleReport();
        }
    }, firstReportDelay);
    
    // ─── 两阶段限价单: 已处理的 (ts, phase) 组合 ────────────
    // executedTargetTss 按 ts 去重（单阶段兼容），两阶段需按 ts+phase 去重
    const executedPhaseKeys = new Set<string>();
    /** 最近读到的文件时间戳（防同一文件重复处理） */
    let lastPredictionTimestamp = '';

    const twoPhaseEnabled = PREDICTION_ENV.TWO_PHASE_ENABLED;
    const limitPrice = PREDICTION_ENV.LIMIT_PRICE;
    const ladderPricesStr = PREDICTION_ENV.LIMIT_PRICE_LADDER;
    const directionalLadderMode =
        Object.keys(((PREDICTION_ENV as any).LOWPRICE_SELECTED_BUY_PRICE_RANGE_BY_SYMBOL_DIRECTION || {}) as Record<string, unknown>).length > 0
        || Object.keys(((PREDICTION_ENV as any).LOWPRICE_RUNG_COUNT_BY_SYMBOL_DIRECTION || {}) as Record<string, unknown>).length > 0;
    const isLadderMode = directionalLadderMode || ladderPricesStr.length > 0;
    const ladderPrices = isLadderMode
        ? ladderPricesStr.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n))
        : [];
    if (twoPhaseEnabled) {
        console.log(`\n🔄 两阶段限价单模式已启用`);
        console.log(`   Phase 1: 限价 $${PREDICTION_ENV.PHASE1_LIMIT_PRICE}, 仓位 ${(PREDICTION_ENV.PHASE1_BET_FRACTION * 100).toFixed(0)}%`);
        console.log(`   Phase 2: 确认/取消/加仓`);
        console.log(`   Phase 3: 扫单 (max $${PREDICTION_ENV.MAX_SWEEP_PRICE})\n`);
    }
    if (isLadderMode) {
        if (directionalLadderMode) {
            console.log(`\n🪜 阶梯限价单模式: 方向梯子主导，统一梯子仅作回退\n`);
        } else {
            console.log(`\n🪜 阶梯限价单模式 (${ladderPrices.length} 档): ${ladderPrices.map(p => '$' + p.toFixed(2)).join(', ')}\n`);
        }
    } else {
        console.log(`\n📝 单限价单模式: $${limitPrice.toFixed(2)}\n`);
    }

    // 扫描循环：读 predictions.json → 按 target_period_end_ts + phase 路由执行
    while (!isShuttingDown) {
        const raw = readLocalPredictionsRaw();
        if (!raw) {
            await sleep(SCAN_INTERVAL_MS);
            continue;
        }

        const ts = raw.target_period_end_ts;
        const phase = raw.phase;  // 0=单阶段, 1=Phase1, 2=Phase2
        const fileTimestamp = raw.data.timestamp;
        const decisionMinute = Math.max(0, Number((PREDICTION_ENV as any).DECISION_MINUTE || process.env.DECISION_MINUTE || 0));
        const decisionReadyTs = Number(ts) + decisionMinute * 60;

        // ─── 限价单模式（单限价 / 阶梯限价）────────────────────
        if (phase === 1 && !twoPhaseEnabled) {
            if (decisionMinute > 0 && Math.floor(Date.now() / 1000) < decisionReadyTs) {
                await sleep(SCAN_INTERVAL_MS);
                continue;
            }
            const limitKey = `${ts}_limit_${fileTimestamp}`;
            if (executedPhaseKeys.has(limitKey)) {
                await sleep(SCAN_INTERVAL_MS);
                continue;
            }
            if (fileTimestamp === lastPredictionTimestamp) {
                await sleep(SCAN_INTERVAL_MS);
                continue;
            }

            const parsed = parseLocalPredictions(raw.data, PREDICTION_ENV.TIMEFRAME, PREDICTION_ENV.ALLOWED_MARKETS);
            const toOrder = parsed.filter((p) => p.direction != null && p.shouldTrade !== false && thresholdMet(p));
            const skippedShouldTrade = parsed.filter((p) => p.direction != null && thresholdMet(p) && p.shouldTrade === false);
            for (const p of skippedShouldTrade) {
                console.log(`  [扫描] 跳过 ${p.symbol}: trade_decision.should_trade=false (融合 SKIP)`);
            }

            lastPredictionTimestamp = fileTimestamp;
            executedPhaseKeys.add(limitKey);

            if (toOrder.length === 0) {
                console.log(`  [扫描] 最终执行: 无可下单预测 ts=${ts} (threshold/trade_decision 未通过)`);
                await sleep(SCAN_INTERVAL_MS);
                continue;
            }

            const range = `${new Date(ts * 1000).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}–${new Date((ts + PredictionExecutor.PERIOD_SECONDS) * 1000).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}`;
            console.log(`  [扫描] 最终执行通过 ts=${ts} (${range}): ${toOrder.length} 条, limitPrice=$${limitPrice.toFixed(2)}`);
            for (const p of toOrder) {
                console.log(`    - ${p.symbol} limitPrice=$${limitPrice.toFixed(2)} bestAsk=—`);
            }

            try {
                if (raw.adaptive_pricing) {
                    // ─── 自适应定价（Exp13 5m）：用 best_ask 实时下单 ───
                    console.log(`  [扫描] 自适应定价 ts=${ts} (${range}): ${toOrder.length} 条预测, 用 best_ask 实时限价`);
                    const execResults = await executor.executeRound({ targetPeriodEndTs: ts, predictions: toOrder });
                    if (shouldConsumeTargetTs(execResults)) {
                        executedTargetTss.add(ts);
                        saveExecutedTargetTss(executedTargetTss);
                    } else {
                        console.warn(`  [扫描] ts=${ts} 本轮未成交/未挂单，不标记已执行，等待下一次预测更新时间重试`);
                    }
                } else if (isLadderMode) {
                    console.log(`  [扫描] 阶梯限价单 ts=${ts} (${range}): ${toOrder.length} 条预测, ${directionalLadderMode ? '方向梯子主导' : `${ladderPrices.length} 档`}`);
                    const execResults = await executor.executeLadderOrder({
                        targetPeriodEndTs: ts,
                        predictions: toOrder,
                        ladderPrices,
                    });
                    if (shouldConsumeTargetTs(execResults)) {
                        executedTargetTss.add(ts);
                        saveExecutedTargetTss(executedTargetTss);
                    } else {
                        console.warn(`  [扫描] ts=${ts} 本轮未成交/未挂单，不标记已执行，等待下一次预测更新时间重试`);
                    }
                } else {
                    console.log(`  [扫描] 单限价单 ts=${ts} (${range}): ${toOrder.length} 条预测, @$${limitPrice.toFixed(2)}`);
                    const execResults = await executor.executeLimitOrder({
                        targetPeriodEndTs: ts,
                        predictions: toOrder,
                        limitPrice,
                    });
                    if (shouldConsumeTargetTs(execResults)) {
                        executedTargetTss.add(ts);
                        saveExecutedTargetTss(executedTargetTss);
                    } else {
                        console.warn(`  [扫描] ts=${ts} 本轮未成交/未挂单，不标记已执行，等待下一次预测更新时间重试`);
                    }
                }
            } catch (e) {
                console.error(`  [扫描] 限价单执行失败:`, e);
            }
        }
        // ─── 两阶段去重: 同一 (ts, phase, timestamp) 只处理一次 ────
        else if (twoPhaseEnabled && phase > 0) {
            if (decisionMinute > 0 && Math.floor(Date.now() / 1000) < decisionReadyTs) {
                await sleep(SCAN_INTERVAL_MS);
                continue;
            }
            const phaseKey = `${ts}_phase${phase}_${fileTimestamp}`;
            if (executedPhaseKeys.has(phaseKey)) {
                await sleep(SCAN_INTERVAL_MS);
                continue;
            }
            // 防止同一文件反复触发（文件未更新时 timestamp 不变）
            if (fileTimestamp === lastPredictionTimestamp) {
                await sleep(SCAN_INTERVAL_MS);
                continue;
            }

            const parsed = parseLocalPredictions(raw.data, PREDICTION_ENV.TIMEFRAME, PREDICTION_ENV.ALLOWED_MARKETS);
            const toOrder = parsed.filter((p) => p.direction != null && p.shouldTrade !== false && thresholdMet(p));
            const skippedShouldTrade = parsed.filter((p) => p.direction != null && thresholdMet(p) && p.shouldTrade === false);
            for (const p of skippedShouldTrade) {
                console.log(`  [扫描] 跳过 ${p.symbol}: trade_decision.should_trade=false (融合 SKIP)`);
            }

            lastPredictionTimestamp = fileTimestamp;
            executedPhaseKeys.add(phaseKey);

            if (toOrder.length === 0) {
                console.log(`  [扫描] 最终执行: 无可下单预测 Phase ${phase} ts=${ts} (threshold/trade_decision 未通过)`);
                await sleep(SCAN_INTERVAL_MS);
                continue;
            }

            const range = `${new Date(ts * 1000).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}–${new Date((ts + PredictionExecutor.PERIOD_SECONDS) * 1000).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}`;
            console.log(`  [扫描] 最终执行通过 Phase ${phase} ts=${ts} (${range}): ${toOrder.length} 条`);
            for (const p of toOrder) {
                console.log(`    - ${p.symbol} limitPrice=$${(raw.limit_price ?? PREDICTION_ENV.PHASE1_LIMIT_PRICE ?? 0).toFixed(2)} bestAsk=—`);
            }

            try {
                if (phase === 1) {
                    // ─── Phase 1: GTC 限价单 ──────────────────
                    console.log(`  [扫描] Phase 1 ts=${ts} (${range}): ${toOrder.length} 条预测，下限价单 @$${raw.limit_price}`);
                    const execResults = await executor.executePhase1({
                        targetPeriodEndTs: ts,
                        predictions: toOrder,
                        limitPrice: raw.limit_price || PREDICTION_ENV.PHASE1_LIMIT_PRICE,
                        betFraction: raw.bet_fraction_this_phase || PREDICTION_ENV.PHASE1_BET_FRACTION,
                    });
                    if (shouldConsumeTargetTs(execResults)) {
                        executedTargetTss.add(ts);
                        saveExecutedTargetTss(executedTargetTss);
                    } else {
                        console.warn(`  [扫描] Phase 1 ts=${ts} 本轮未成交/未挂单，不标记已执行，等待下一次预测更新时间重试`);
                    }
                } else if (phase === 2) {
                    // ─── Phase 2: 确认/取消/加仓 ──────────────
                    console.log(`  [扫描] Phase 2 ts=${ts} (${range}): ${toOrder.length} 条预测，确认/取消`);
                    const execResults = await executor.executePhase2({
                        targetPeriodEndTs: ts,
                        predictions: toOrder,
                        betFraction: raw.bet_fraction_this_phase || (1 - PREDICTION_ENV.PHASE1_BET_FRACTION),
                    });
                    // Phase 2 完成后，若有成交/挂单/已持仓跳过才标记该 ts 已执行
                    if (shouldConsumeTargetTs(execResults)) {
                        executedTargetTss.add(ts);
                        saveExecutedTargetTss(executedTargetTss);
                    } else {
                        console.warn(`  [扫描] Phase 2 ts=${ts} 本轮未成交/未挂单，不标记已执行，等待下一次预测更新时间重试`);
                    }
                }
            } catch (e) {
                console.error(`  [扫描] Phase ${phase} 执行失败:`, e);
            }
        } else {
            // ─── 单阶段模式（phase=0 或未启用两阶段/双腿）──────────
            if (executedTargetTss.has(ts)) {
                await sleep(SCAN_INTERVAL_MS);
                continue;
            }
            if (fileTimestamp === lastPredictionTimestamp) {
                await sleep(SCAN_INTERVAL_MS);
                continue;
            }

            lastPredictionTimestamp = fileTimestamp;

            const parsed = parseLocalPredictions(raw.data, PREDICTION_ENV.TIMEFRAME, PREDICTION_ENV.ALLOWED_MARKETS);
            const toOrder = parsed.filter((p) => p.direction != null && p.shouldTrade !== false && thresholdMet(p));
            const skippedShouldTrade = parsed.filter((p) => p.direction != null && thresholdMet(p) && p.shouldTrade === false);
            for (const p of skippedShouldTrade) {
                console.log(`  [扫描] 跳过 ${p.symbol}: trade_decision.should_trade=false (融合 SKIP)`);
            }

            if (toOrder.length === 0) {
                console.log(`  [扫描] 最终执行: 无可下单预测 ts=${ts} (threshold/trade_decision 未通过)`);
                await sleep(SCAN_INTERVAL_MS);
                continue;
            }

            const range = `${new Date(ts * 1000).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}–${new Date((ts + PredictionExecutor.PERIOD_SECONDS) * 1000).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}`;
            const useLimitOrderMode = process.env.LIMIT_PRICE != null;
            const limitPriceStr = useLimitOrderMode ? `limitPrice=$${limitPrice.toFixed(2)}` : 'limitPrice=—';
            console.log(`  [扫描] 最终执行通过 ts=${ts} (${range}): ${toOrder.length} 条, ${limitPriceStr}`);
            for (const p of toOrder) {
                console.log(`    - ${p.symbol} limitPrice=$${useLimitOrderMode ? limitPrice.toFixed(2) : '—'} bestAsk=—`);
            }
            try {
                if (useLimitOrderMode) {
                    const execResults = await executor.executeLimitOrder({
                        targetPeriodEndTs: ts,
                        predictions: toOrder,
                        limitPrice,
                    });
                    if (shouldConsumeTargetTs(execResults)) {
                        executedTargetTss.add(ts);
                        saveExecutedTargetTss(executedTargetTss);
                    } else {
                        console.warn(`  [扫描] ts=${ts} 本轮未成交/未挂单，不标记已执行，等待下一次预测更新时间重试`);
                    }
                } else {
                    const execResults = await executor.executeRound({ targetPeriodEndTs: ts, predictions: toOrder });
                    if (shouldConsumeTargetTs(execResults)) {
                        executedTargetTss.add(ts);
                        saveExecutedTargetTss(executedTargetTss);
                    } else {
                        console.warn(`  [扫描] ts=${ts} 本轮未成交/未挂单，不标记已执行，等待下一次预测更新时间重试`);
                    }
                }
            } catch (e) {
                console.error('  [扫描] 执行失败:', e);
            }
        }
        await sleep(SCAN_INTERVAL_MS);
    }
};

// ============================================================
// 运行
// ============================================================

main().catch((error) => {
    console.error('\n❌ 致命错误:', error);
    process.exit(1);
});
