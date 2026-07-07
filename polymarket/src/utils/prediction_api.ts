/**
 * 预测读取器
 * 优先从本地 JSON 文件读取预测结果（更快），备选 HTTP API
 */

import * as fs from 'fs';
import * as path from 'path';
import axios from 'axios';
import { HttpsProxyAgent } from 'https-proxy-agent';
import { PREDICTION_ENV, PredictionEnvConfig } from '../config/prediction_env';

const _proxyUrl = process.env.HTTPS_PROXY || process.env.https_proxy
               || process.env.HTTP_PROXY  || process.env.http_proxy;
const _proxyAgent = _proxyUrl ? new HttpsProxyAgent(_proxyUrl, { keepAlive: true }) : undefined;
const axiosClient = axios.create({
    ...(_proxyAgent ? { httpAgent: _proxyAgent, httpsAgent: _proxyAgent, proxy: false } : {}),
});

// ============================================================
// 类型定义
// ============================================================

export interface PredictionResult {
    symbol: string;        // 交易对，如 "BTC/USDT"
    timeframe: string;     // 时间周期，如 "15m"
    direction: 'UP' | 'DOWN' | null;  // 预测方向
    confidence: number;    // 置信度 (0-1)
    effectiveThreshold?: number; // 当前实际生效阈值（含基础阈值覆盖与门控增量）
    rawConfidence?: number; // 原始置信度（供校准/审计使用）
    calibratedConfidence?: number; // 校准后的置信度（仅 enforce 时用于阈值判断）
    calibrationMeta?: {
        mode?: 'off' | 'shadow' | 'enforce';
        method?: 'identity' | 'temperature' | 'sigmoid' | 'isotonic';
        sampleCount?: number;
        lookbackHours?: number;
        brier?: number;
        ece?: number;
        calibratedConfidence?: number;
        rawConfidence?: number;
        reasonCode?: string;
        sourceMode?: 'simulation' | 'live';
    };
    expectancyMeta?: {
        mode?: 'off' | 'shadow' | 'enforce';
        status?: 'normal' | 'degraded' | 'blocked';
        sampleCount?: number;
        lookbackHours?: number;
        wrPost?: number;
        pnlPerTrade?: number;
        degradedBetScale?: number;
        degradedExtraDelta?: number;
        reasonCode?: string;
        sourceMode?: 'simulation' | 'live';
    };
    thresholdDriftMeta?: {
        mode?: 'off' | 'shadow' | 'enforce';
        status?: 'normal' | 'drifted';
        sampleCount?: number;
        lookbackHours?: number;
        wrPost?: number;
        pnlPerTrade?: number;
        thresholdDelta?: number;
        reasonCode?: string;
        sourceMode?: 'simulation' | 'live';
    };
    metaLabelMeta?: {
        mode?: 'off' | 'shadow' | 'enforce';
        takeTrade?: boolean;
        confidence?: number;
        reasonCode?: string;
    };
    selectorMeta?: {
        mode?: 'off' | 'shadow' | 'enforce';
        eligible?: boolean;
        status?: string;
        reasonCode?: string;
    };
    selectionMeta?: {
        applicable?: boolean;
        action?: 'allow' | 'abstain';
        reasonCode?: string;
        marketQualityScore?: number;
        evidence?: {
            qualityMode?: 'ok' | 'soft_degraded' | 'hard_degraded' | 'halt';
            expectancyStatus?: 'normal' | 'degraded' | 'blocked';
            expectancySampleCount?: number;
            expectancyPnlPerTrade?: number;
            uncertaintyStatus?: 'normal' | 'degraded' | 'blocked';
            uncertaintyDispersion?: number | null;
            uncertaintyEffectiveSourceCount?: number | null;
            uncertaintyIntervalWidth?: number | null;
            regimeDirectionPolicy?: 'BOTH' | 'UP' | 'DOWN' | 'NONE';
            regimeDisagreementCount?: number;
            changePointProb?: number | null;
            confidence?: number | null;
            effectiveThreshold?: number | null;
            reasons?: string[];
        };
    };
    uncertaintyMeta?: {
        mode?: 'off' | 'shadow' | 'enforce';
        status?: 'normal' | 'degraded' | 'blocked';
        dispersionScore?: number;
        effectiveSourceCount?: number;
        changePointProb?: number;
        intervalWidth?: number;
        intervalWidthMax?: number;
        intervalSource?: 'proxy';
        degradedExtraDelta?: number;
        reasonCode?: string;
    };
    runtimeSizingCapPct?: number;
    runtimeSizingCapReason?: string;
    slot02Aux1hActionCode?: number | null;
    slot02Aux1hSoftCapPct?: number | null;
    slot02Aux1hWeakSupportActive?: boolean | null;
    slot02Aux1hMainConfidenceBucket?: number | null;
    slot02Aux1hVolValue?: number | null;
    slot02Aux1hVolBucket?: number | null;
    regimeMeta?: {
        mode?: 'off' | 'shadow' | 'enforce';
        active?: boolean;
        directionPolicy?: 'BOTH' | 'UP' | 'DOWN' | 'NONE';
        policyMode?: 'one_sided' | 'neutral';
        policyConfidence?: number;
        policyReason?: string;
        policyDisagreementFlags?: string[];
        thresholdDelta?: number;
        regimeId?: string;
        reasonCode?: string;
        sourceMode?: 'simulation' | 'live';
        changePointProb?: number;
        changePointDirection?: 'UP' | 'DOWN' | 'NONE';
        changePointScoreUp?: number;
        changePointScoreDown?: number;
    };
    ensembleMeta?: {
        consensusScore?: number;
        dispersionScore?: number;
        effectiveSourceCount?: number;
        consensusBlocked?: boolean;
        reasonCode?: string | null;
    };
    timestamp?: string;    // 预测时间
    error?: string;        // 错误信息
    /** 融合侧 trade_decision.should_trade；未设置时视为可交易（兼容旧格式） */
    shouldTrade?: boolean;
}

export interface PredictionFile {
    timestamp: string;
    /** 目标周期的 Polymarket slug ts（= 周期开始，与官网一致）。*/
    target_period_end_ts?: number;
    /** 两阶段限价单: 阶段编号 (0=单阶段兼容, 1=Phase1, 2=Phase2) */
    phase?: number;
    /** 建议限价 (如 0.50) */
    limit_price?: number;
    /** 自适应定价: true 时 TS 端用 best_ask 作为限价（Exp13 5m 用） */
    adaptive_pricing?: boolean;
    /** 本阶段投入总仓位的比例 (如 0.50 = 50%) */
    bet_fraction_this_phase?: number;
    /** 扫单最高价 */
    max_sweep_price?: number;
    model_version?: string;
    predictions: Record<string, {
        symbol: string;
        timeframe: string;
        direction: 'UP' | 'DOWN' | null;
        confidence: number;
        timestamp?: string;
        error?: string;
        /** 融合/规则输出；含 should_trade 时 Node 与其对齐 */
        details?: {
            trade_decision?: {
                should_trade?: boolean;
                skip_reason?: string | null;
            };
            calibration?: {
                method?: 'identity' | 'temperature' | 'sigmoid' | 'isotonic';
                sample_count?: number;
                brier?: number;
                ece?: number;
                source_mode?: 'simulation' | 'live';
            };
            expectancy_gate?: {
                status?: 'normal' | 'degraded' | 'blocked';
                sample_count?: number;
                wr_post?: number;
                pnl_per_trade?: number;
                degraded_bet_scale?: number;
                source_mode?: 'simulation' | 'live';
            };
            threshold_drift?: {
                status?: 'normal' | 'drifted';
                sample_count?: number;
                wr_post?: number;
                pnl_per_trade?: number;
                threshold_delta?: number;
                source_mode?: 'simulation' | 'live';
            };
            meta_label?: {
                take_trade?: boolean;
                confidence?: number;
                reason_code?: string | null;
            };
            selector_overlay?: {
                eligible?: boolean;
                status?: string;
                reason_code?: string | null;
            };
            regime_gate?: {
                active?: boolean;
                direction_policy?: 'BOTH' | 'UP' | 'DOWN' | 'NONE';
                threshold_delta?: number;
                regime_id?: string;
                reason_code?: string | null;
                source_mode?: 'simulation' | 'live';
                change_point_prob?: number;
                change_point_direction?: 'UP' | 'DOWN' | 'NONE';
                change_point_score_up?: number;
                change_point_score_down?: number;
            };
            ensemble_consensus?: {
                consensus_score?: number;
                dispersion_score?: number;
                effective_source_count?: number;
                consensus_blocked?: boolean;
                reason_code?: string | null;
            };
        };
    }>;
}

/** 扫描模式：读到的原始预测 + 目标周期结束时间戳 + 阶段信息 */
export interface PredictionRaw {
    data: PredictionFile;
    target_period_end_ts: number;
    /** 阶段编号 (0=单阶段兼容, 1=Phase1, 2=Phase2) */
    phase: number;
    /** 建议限价 */
    limit_price: number;
    /** 本阶段投入总仓位的比例 */
    bet_fraction_this_phase: number;
    /** 扫单最高价 */
    max_sweep_price: number;
    /** 自适应定价: true 时用 best_ask 作为限价（Exp13 5m 用） */
    adaptive_pricing: boolean;
}

// ============================================================
// 阈值逻辑（与训练时一致，见 docs/概率与阈值逻辑.md）
// ============================================================

/** 四舍五入到 4 位小数，避免 0.57 与 0.56999999999999 等浮点误差导致边界漏单 */
const roundProb = (x: number): number => Math.round(x * 10000) / 10000;

export const effectiveConfidence = (p: PredictionResult): number => {
    const calibrated = Number(p.calibratedConfidence);
    if (Number.isFinite(calibrated) && calibrated >= 0 && calibrated <= 1) {
        return calibrated;
    }
    return Number(p.confidence) || 0;
};

/**
 * 判断预测是否达到下单阈值。
 * 单阈值：confidence >= PROB_THRESHOLD。
 * 双阈值：UP 时 confidence >= PROB_THRESHOLD_UP；DOWN 时 confidence > (1 - PROB_THRESHOLD_DOWN)，即 P(UP) < down，与训练时 P(UP) 语义一致。
 * 比较前对置信度与阈值做四舍五入，避免浮点误差导致本应下单的边界被误判为不下单。
 */
export const thresholdMet = (p: PredictionResult): boolean => {
    if (p.direction == null) return false;
    const confidence = effectiveConfidence(p);
    const dual = PREDICTION_ENV.PROB_THRESHOLD_UP != null && PREDICTION_ENV.PROB_THRESHOLD_DOWN != null;
    if (dual) {
        if (p.direction === 'UP') return roundProb(confidence) >= roundProb(PREDICTION_ENV.PROB_THRESHOLD_UP!);
        const bound = 1 - PREDICTION_ENV.PROB_THRESHOLD_DOWN!;
        return roundProb(confidence) > roundProb(bound);
    }
    return roundProb(confidence) >= roundProb(PREDICTION_ENV.PROB_THRESHOLD);
};

/**
 * 多 trader 模式：使用注入的 env 判断阈值（不读全局 PREDICTION_ENV）。
 */
export const thresholdMetWith = (p: PredictionResult, env: PredictionEnvConfig): boolean => {
    if (p.direction == null) return false;
    const confidence = effectiveConfidence(p);
    const dual = env.PROB_THRESHOLD_UP != null && env.PROB_THRESHOLD_DOWN != null;
    if (dual) {
        if (p.direction === 'UP') return roundProb(confidence) >= roundProb(env.PROB_THRESHOLD_UP!);
        const bound = 1 - env.PROB_THRESHOLD_DOWN!;
        return roundProb(confidence) > roundProb(bound);
    }
    return roundProb(confidence) >= roundProb(env.PROB_THRESHOLD);
};

// ============================================================
// 常量
// ============================================================

// 本地预测文件路径（与 Python prediction_writer 一致：项目根/polymarket/predictions.json）
// PREDICTION_SUFFIX: 多套模型并行时区分，如 _A => predictions_A.json；LOGS_DIR 搭配 logs_A 等
const _suffix = process.env.PREDICTION_SUFFIX || '';
const PREDICTION_FILE = path.resolve(__dirname, '..', '..', `predictions${_suffix}.json`);

// 预测文件最大有效期（秒）- 超过则认为过期
const MAX_PREDICTION_AGE_SECONDS = 120; // 2 分钟

// 等待预测文件更新的最大重试次数和间隔
const PREDICTION_RETRY_COUNT = 3;
const PREDICTION_RETRY_INTERVAL_MS = 5000; // 5 秒

// ============================================================
// 本地文件读取（优先）
// ============================================================

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

let _warnedOldFormat = false;
const _warnedOldFormatPaths = new Set<string>();

/**
 * 扫描模式：同步读取预测文件，不做时间有效期校验
 * 若缺少 target_period_end_ts（旧格式）返回 null
 */
export const readLocalPredictionsRaw = (): PredictionRaw | null => {
    try {
        if (!fs.existsSync(PREDICTION_FILE)) return null;
        const content = fs.readFileSync(PREDICTION_FILE, 'utf-8');
        const data: PredictionFile = JSON.parse(content);
        const ts = data.target_period_end_ts;
        if (typeof ts !== 'number') {
            if (!_warnedOldFormat) {
                _warnedOldFormat = true;
                console.warn('  [预测] 文件缺少 target_period_end_ts，请用最新版运行: python -m src.python.prediction_writer --once');
            }
            return null;
        }
        return {
            data,
            target_period_end_ts: ts,
            phase: typeof data.phase === 'number' ? data.phase : 0,
            limit_price: typeof data.limit_price === 'number' ? data.limit_price : 0,
            bet_fraction_this_phase: typeof data.bet_fraction_this_phase === 'number' ? data.bet_fraction_this_phase : 1.0,
            max_sweep_price: typeof data.max_sweep_price === 'number' ? data.max_sweep_price : 0.54,
            adaptive_pricing: data.adaptive_pricing === true,
        };
    } catch {
        return null;
    }
};

/**
 * 多 trader 模式：从指定路径读取预测文件（不依赖模块级 PREDICTION_FILE）。
 * 每个 trader 传入自己的 predictions_xxx.json 路径。
 */
export const readLocalPredictionsRawFromFile = (filePath: string): PredictionRaw | null => {
    try {
        if (!fs.existsSync(filePath)) return null;
        const content = fs.readFileSync(filePath, 'utf-8');
        const data: PredictionFile = JSON.parse(content);
        const ts = data.target_period_end_ts;
        if (typeof ts !== 'number') {
            if (!_warnedOldFormatPaths.has(filePath)) {
                _warnedOldFormatPaths.add(filePath);
                console.warn(`  [预测] ${path.basename(filePath)} 缺少 target_period_end_ts`);
            }
            return null;
        }
        return {
            data,
            target_period_end_ts: ts,
            phase: typeof data.phase === 'number' ? data.phase : 0,
            limit_price: typeof data.limit_price === 'number' ? data.limit_price : 0,
            bet_fraction_this_phase: typeof data.bet_fraction_this_phase === 'number' ? data.bet_fraction_this_phase : 1.0,
            max_sweep_price: typeof data.max_sweep_price === 'number' ? data.max_sweep_price : 0.54,
            adaptive_pricing: data.adaptive_pricing === true,
        };
    } catch {
        return null;
    }
};

/** 目标周期是否仍未结算。target_period_end_ts 实际为周期开始(slug ts)，周期结束=ts+PERIOD_SECONDS */
export const isPredictionPeriodOpen = (targetPeriodEndTs: number, periodSeconds: number = 900): boolean =>
    Math.floor(Date.now() / 1000) < targetPeriodEndTs + periodSeconds;

/**
 * 从本地 JSON 文件读取预测结果（单次尝试）
 */
const readLocalPredictionsOnce = (): { data: PredictionFile | null; expired: boolean; ageSeconds: number } => {
    try {
        if (!fs.existsSync(PREDICTION_FILE)) {
            console.log('  [预测] 本地预测文件不存在:', PREDICTION_FILE);
            return { data: null, expired: false, ageSeconds: 0 };
        }
        
        const content = fs.readFileSync(PREDICTION_FILE, 'utf-8');
        const data: PredictionFile = JSON.parse(content);
        
        // 检查预测是否过期
        const predictionTime = new Date(data.timestamp).getTime();
        const now = Date.now();
        const ageSeconds = (now - predictionTime) / 1000;
        
        if (ageSeconds > MAX_PREDICTION_AGE_SECONDS) {
            return { data: null, expired: true, ageSeconds };
        }
        
        return { data, expired: false, ageSeconds };
        
    } catch (error) {
        const errMsg = error instanceof Error ? error.message : String(error);
        console.error('  [预测] 读取预测文件失败:', errMsg);
        return { data: null, expired: false, ageSeconds: 0 };
    }
};

/**
 * 从本地 JSON 文件读取预测结果（带重试机制）
 * 如果文件过期，会等待几秒后重试（等待 Python 生成新预测）
 */
const readLocalPredictions = async (): Promise<PredictionFile | null> => {
    for (let attempt = 0; attempt <= PREDICTION_RETRY_COUNT; attempt++) {
        const { data, expired, ageSeconds } = readLocalPredictionsOnce();
        
        if (data) {
            console.log(`  [预测] 读取本地预测成功 (${Math.round(ageSeconds)}秒前生成)`);
            return data;
        }
        
        if (expired && attempt < PREDICTION_RETRY_COUNT) {
            console.log(`  [预测] 预测文件已过期 (${Math.round(ageSeconds)}秒前生成)`);
            console.log(`         文件: ${PREDICTION_FILE}`);
            console.log(`         等待 Python 生成新预测... (重试 ${attempt + 1}/${PREDICTION_RETRY_COUNT})`);
            await sleep(PREDICTION_RETRY_INTERVAL_MS);
            continue;
        }
        
        if (expired) {
            console.log(`  [预测] 预测文件已过期 (${Math.round(ageSeconds)}秒前生成，超过${MAX_PREDICTION_AGE_SECONDS}秒有效期)`);
            console.log(`         文件: ${PREDICTION_FILE}`);
            console.log(`         重试已用尽，请确保 Python 预测写入器正在运行`);
        }
        
        break;
    }
    
    return null;
};

/**
 * 解析本地预测文件为标准格式
 * @param allowedSymbols 若提供，仅保留在此列表中的 symbol（如 PREDICTION_ENV.ALLOWED_MARKETS，排除 XRP 等）
 */
export const parseLocalPredictions = (
    data: PredictionFile,
    timeframe?: string,
    allowedSymbols?: string[]
): PredictionResult[] => {
    const tf = timeframe || PREDICTION_ENV.TIMEFRAME;
    const results: PredictionResult[] = [];
    const set = allowedSymbols ? new Set(allowedSymbols) : null;
    
    for (const [key, pred] of Object.entries(data.predictions)) {
        if (pred.timeframe !== tf) continue;
        // 兼容 ALLOWED_MARKETS 为 "BTC/USDT" 或 "BTC"
        if (set) {
            const base = pred.symbol.replace(/\/.*$/, '');
            if (!set.has(pred.symbol) && !set.has(base)) continue;
        }
        
        results.push({
            symbol: pred.symbol,
            timeframe: pred.timeframe,
            direction: pred.direction,
            confidence: pred.confidence,
            rawConfidence: pred.confidence,
            calibratedConfidence: pred.details?.calibration && typeof pred.details.calibration === 'object'
                ? pred.confidence
                : undefined,
            calibrationMeta: pred.details?.calibration
                ? {
                    method: pred.details.calibration.method,
                    sampleCount: pred.details.calibration.sample_count,
                    brier: pred.details.calibration.brier,
                    ece: pred.details.calibration.ece,
                    sourceMode: pred.details.calibration.source_mode,
                }
                : undefined,
            expectancyMeta: pred.details?.expectancy_gate
                ? {
                    status: pred.details.expectancy_gate.status,
                    sampleCount: pred.details.expectancy_gate.sample_count,
                    wrPost: pred.details.expectancy_gate.wr_post,
                    pnlPerTrade: pred.details.expectancy_gate.pnl_per_trade,
                    degradedBetScale: pred.details.expectancy_gate.degraded_bet_scale,
                    sourceMode: pred.details.expectancy_gate.source_mode,
                }
                : undefined,
            thresholdDriftMeta: pred.details?.threshold_drift
                ? {
                    status: pred.details.threshold_drift.status,
                    sampleCount: pred.details.threshold_drift.sample_count,
                    wrPost: pred.details.threshold_drift.wr_post,
                    pnlPerTrade: pred.details.threshold_drift.pnl_per_trade,
                    thresholdDelta: pred.details.threshold_drift.threshold_delta,
                    sourceMode: pred.details.threshold_drift.source_mode,
                }
                : undefined,
            metaLabelMeta: pred.details?.meta_label
                ? {
                    takeTrade: pred.details.meta_label.take_trade,
                    confidence: pred.details.meta_label.confidence,
                    reasonCode: pred.details.meta_label.reason_code || undefined,
                }
                : undefined,
            selectorMeta: pred.details?.selector_overlay
                ? {
                    eligible: pred.details.selector_overlay.eligible,
                    status: pred.details.selector_overlay.status,
                    reasonCode: pred.details.selector_overlay.reason_code || undefined,
                }
                : undefined,
            regimeMeta: pred.details?.regime_gate
                ? {
                    active: pred.details.regime_gate.active,
                    directionPolicy: pred.details.regime_gate.direction_policy,
                    thresholdDelta: pred.details.regime_gate.threshold_delta,
                    regimeId: pred.details.regime_gate.regime_id,
                    reasonCode: pred.details.regime_gate.reason_code || undefined,
                    sourceMode: pred.details.regime_gate.source_mode,
                    changePointProb: pred.details.regime_gate.change_point_prob,
                    changePointDirection: pred.details.regime_gate.change_point_direction,
                    changePointScoreUp: pred.details.regime_gate.change_point_score_up,
                    changePointScoreDown: pred.details.regime_gate.change_point_score_down,
                }
                : undefined,
            ensembleMeta: pred.details?.ensemble_consensus
                ? {
                    consensusScore: pred.details.ensemble_consensus.consensus_score,
                    dispersionScore: pred.details.ensemble_consensus.dispersion_score,
                    effectiveSourceCount: pred.details.ensemble_consensus.effective_source_count,
                    consensusBlocked: pred.details.ensemble_consensus.consensus_blocked,
                    reasonCode: pred.details.ensemble_consensus.reason_code,
                }
                : undefined,
            timestamp: pred.timestamp,
            error: pred.error,
            shouldTrade: (data.model_version && String(data.model_version).includes('ensemble'))
                ? (pred.details?.trade_decision?.should_trade === true)
                : (pred.details?.trade_decision?.should_trade !== false),
        });
    }
    
    return results;
};

// ============================================================
// HTTP API 备选方案
// ============================================================

/**
 * 从 HTTP API 获取预测（备选）
 */
const fetchFromApi = async (symbol: string, timeframe: string): Promise<PredictionResult> => {
    const normalizedSymbol = symbol.toUpperCase().replace('/USDT', '');
    const url = `${PREDICTION_ENV.PREDICTION_API_URL}/predict/${normalizedSymbol}?timeframe=${timeframe}`;
    
    try {
        console.log(`  [API] 正在请求: ${url}`);
        const response = await axiosClient.get(url, {
            timeout: PREDICTION_ENV.REQUEST_TIMEOUT_MS,
        });
        
        const data = response.data;
        console.log(`  [API] ${normalizedSymbol} 预测成功`);
        
        return {
            symbol: data.symbol || `${normalizedSymbol}/USDT`,
            timeframe: data.timeframe || timeframe,
            direction: data.direction,
            confidence: data.confidence || 0,
        };
    } catch (error) {
        let errMsg = '未知错误';
        
        if (axios.isAxiosError(error)) {
            if (error.code === 'ECONNABORTED') {
                errMsg = `请求超时 (${PREDICTION_ENV.REQUEST_TIMEOUT_MS/1000}秒)`;
            } else if (error.code === 'ECONNREFUSED') {
                errMsg = '连接被拒绝，API服务可能未启动';
            } else {
                errMsg = error.message;
            }
        }
        
        console.error(`  [API] ${normalizedSymbol} 预测失败: ${errMsg}`);
        
        return {
            symbol: `${normalizedSymbol}/USDT`,
            timeframe,
            direction: null,
            confidence: 0,
            error: errMsg,
        };
    }
};

// ============================================================
// 公共 API
// ============================================================

/**
 * 检查预测源是否可用（用于初始化，不做 120 秒过期判断）
 * 有 target_period_end_ts 即认为可用；否则说明一句，扫描会继续等待
 */
export const checkPredictionApiHealth = async (): Promise<boolean> => {
    const raw = readLocalPredictionsRaw();
    if (raw) {
        console.log('  [预测] 本地预测文件可用 (含 target_period_end_ts)');
        return true;
    }
    console.log('  [预测] 预测文件尚未就绪或缺少 target_period_end_ts，扫描将等待 (目标: ' + path.basename(PREDICTION_FILE) + ')');
    return false;
};

/**
 * 获取单个交易对的预测
 */
export const getPrediction = async (
    symbol: string,
    timeframe?: string
): Promise<PredictionResult> => {
    const tf = timeframe || PREDICTION_ENV.TIMEFRAME;
    const normalizedSymbol = symbol.toUpperCase();
    const fullSymbol = normalizedSymbol.includes('/') ? normalizedSymbol : `${normalizedSymbol}/USDT`;
    
    // 优先从本地文件读取（带重试机制）
    const localData = await readLocalPredictions();
    if (localData) {
        const key = `${fullSymbol.replace('/', '_')}_${tf}`;
        const pred = localData.predictions[key];
        
        if (pred) {
            return {
                symbol: pred.symbol,
                timeframe: pred.timeframe,
                direction: pred.direction,
                confidence: pred.confidence,
                timestamp: pred.timestamp,
                error: pred.error,
            };
        }
    }
    
    // 备选：HTTP API
    console.log('  [预测] 本地文件无此预测，尝试 HTTP API...');
    return fetchFromApi(fullSymbol, tf);
};

/**
 * 获取所有允许市场的预测
 */
export const getAllPredictions = async (
    timeframe?: string
): Promise<PredictionResult[]> => {
    const tf = timeframe || PREDICTION_ENV.TIMEFRAME;
    
    // 优先从本地文件读取（带重试机制）
    const localData = await readLocalPredictions();
    if (localData) {
        const results = parseLocalPredictions(localData, tf, PREDICTION_ENV.ALLOWED_MARKETS);
        console.log(`  [预测] 从本地文件读取了 ${results.length} 个预测 (允许市场: ${PREDICTION_ENV.ALLOWED_MARKETS.join(', ')})`);
        return results;
    }
    
    // 备选：逐个调用 HTTP API
    console.log('  [预测] 本地文件不可用（重试后仍过期），改用 HTTP API...');
    const markets = PREDICTION_ENV.ALLOWED_MARKETS;
    const predictions: PredictionResult[] = [];
    
    for (const symbol of markets) {
        const result = await fetchFromApi(symbol, tf);
        predictions.push(result);
        await sleep(100); // 避免速率限制
    }
    
    console.log(`  [预测] 从 HTTP API 获取了 ${predictions.length} 个预测`);
    return predictions;
};

/**
 * 获取高置信度的预测
 */
export const getHighConfidencePredictions = async (
    timeframe?: string,
    threshold?: number
): Promise<PredictionResult[]> => {
    const allPredictions = await getAllPredictions(timeframe);
    const useCustomThreshold = threshold != null;
    const meetsThreshold = (p: PredictionResult): boolean =>
        useCustomThreshold ? roundProb(effectiveConfidence(p)) >= roundProb(threshold!) : thresholdMet(p);

    const highConfidence = allPredictions.filter((p) => {
        return !p.error && p.direction !== null && meetsThreshold(p);
    });
    
    const dualThreshold = PREDICTION_ENV.PROB_THRESHOLD_UP != null && PREDICTION_ENV.PROB_THRESHOLD_DOWN != null;
    let thresholdDesc: string;
    if (threshold != null) {
        thresholdDesc = `≥${(threshold * 100).toFixed(0)}%`;
    } else if (dualThreshold) {
        thresholdDesc = `UP≥${(PREDICTION_ENV.PROB_THRESHOLD_UP! * 100).toFixed(0)}%, DOWN>${((1 - PREDICTION_ENV.PROB_THRESHOLD_DOWN!) * 100).toFixed(0)}% (即prob<${(PREDICTION_ENV.PROB_THRESHOLD_DOWN! * 100).toFixed(0)}%)`;
    } else {
        thresholdDesc = `≥${(PREDICTION_ENV.PROB_THRESHOLD * 100).toFixed(0)}%`;
    }
    console.log(`  [预测] 高置信度预测: ${highConfidence.length}/${allPredictions.length} (置信度${thresholdDesc})`);
    
    return highConfidence;
};

/**
 * 格式化预测结果用于日志输出
 */
export const formatPrediction = (p: PredictionResult): string => {
    if (p.error) {
        return `${p.symbol} (${p.timeframe}): 错误 - ${p.error}`;
    }
    
    const directionIcon = p.direction === 'UP' ? '📈' : '📉';
    const directionText = p.direction === 'UP' ? '上涨' : '下跌';
    const confidencePercent = (effectiveConfidence(p) * 100).toFixed(1);
    const symbolShort = p.symbol.replace('/USDT', '');
    
    return `${symbolShort}-${p.timeframe}: ${directionIcon} ${directionText} (置信度: ${confidencePercent}%)`;
};

/**
 * 打印所有预测结果
 */
export const printPredictions = (predictions: PredictionResult[]): void => {
    console.log('\n[预测结果]');
    console.log('-'.repeat(50));

    for (const p of predictions) {
        const met = thresholdMet(p);
        const statusIcon = met ? '✅' : '⏸️';
        const statusText = met ? '基础阈值通过（非最终执行结论）' : '基础阈值未过（非最终执行结论）';
        
        console.log(`  ${statusIcon} ${formatPrediction(p)}`);
        if (!p.error) {
            console.log(`      状态: ${statusText}`);
        }
    }
    
    console.log('-'.repeat(50));
    
    const highConfidence = predictions.filter((p) => !p.error && thresholdMet(p));
    const dualThreshold = PREDICTION_ENV.PROB_THRESHOLD_UP != null && PREDICTION_ENV.PROB_THRESHOLD_DOWN != null;
    let thresholdDesc: string;
    if (dualThreshold) {
        thresholdDesc = `UP≥${(PREDICTION_ENV.PROB_THRESHOLD_UP! * 100).toFixed(0)}%, DOWN>${((1 - PREDICTION_ENV.PROB_THRESHOLD_DOWN!) * 100).toFixed(0)}% (即prob<${(PREDICTION_ENV.PROB_THRESHOLD_DOWN! * 100).toFixed(0)}%)`;
    } else {
        thresholdDesc = `≥${(PREDICTION_ENV.PROB_THRESHOLD * 100).toFixed(0)}%`;
    }
    console.log(`  总计: ${predictions.length} 个预测`);
    console.log(`  全局预测摘要(基础阈值${thresholdDesc}，不代表各 trader 最终执行/下单结论): ${highConfidence.length} 个`);
    console.log('');
};

/**
 * 获取预测文件路径。
 * - 传入 predictionSuffix 时优先使用该后缀（多 trader 并行场景）
 * - 不传时回退到模块级默认后缀（兼容旧调用）
 */
export const getPredictionFilePath = (predictionSuffix?: string): string => {
    const suffix = predictionSuffix ?? process.env.PREDICTION_SUFFIX ?? _suffix;
    return path.resolve(__dirname, '..', '..', `predictions${suffix}.json`);
};
