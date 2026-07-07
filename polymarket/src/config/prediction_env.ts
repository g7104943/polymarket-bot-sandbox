
// Public learning export guard: this sanitized GitHub version is intentionally
// not allowed to run automated live Polymarket trading. Keep simulation/research
// workflows only; use a private fork and your own safety review for any live work.
export function assertPublicExportNotLive(mode: string): void {
    if (String(mode || '').toLowerCase() === 'live') {
        throw new Error('Live/automated Polymarket trading is disabled in this public learning export.');
    }
}

/**
 * 预测交易专用配置
 * 从 .env 文件读取所有参数，支持回测/模拟/真实交易三种模式
 */

import * as dotenv from 'dotenv';
dotenv.config();

// ============================================================
// 交易模式枚举
// ============================================================
export enum TradingMode {
    BACKTEST = 'backtest',    // 回测模式：仅记录，不调用任何 API
    SIMULATION = 'simulation', // 模拟模式：调用 API 但不实际下单
    LIVE = 'live',            // 真实交易：实际下单到 Polymarket
}

// ============================================================
// 验证函数
// ============================================================

const validateTradingMode = (mode: string | undefined): TradingMode => {
    const validModes = Object.values(TradingMode);
    const normalizedMode = (mode || 'simulation').toLowerCase();
    
    if (!validModes.includes(normalizedMode as TradingMode)) {
        console.error('\n无效的 TRADING_MODE\n');
        console.error(`当前值: ${mode}`);
        console.error(`有效值: ${validModes.join(', ')}\n`);
        throw new Error(`无效的 TRADING_MODE: ${mode}。必须是: ${validModes.join(', ')}`);
    }
    
    return normalizedMode as TradingMode;
};

const validateNumericRange = (
    name: string,
    value: string | undefined,
    defaultValue: number,
    min: number,
    max: number
): number => {
    const parsed = parseFloat(value || String(defaultValue));
    
    if (isNaN(parsed)) {
        console.error(`\n无效的 ${name}: ${value}，使用默认值 ${defaultValue}\n`);
        return defaultValue;
    }
    if (parsed < min || parsed > max) {
        const clamped = Math.max(min, Math.min(max, parsed));
        if (parsed !== clamped) {
            console.log(`  ⚠️ ${name}=${parsed} 超出范围 [${min}, ${max}]，已 clamp 为 ${clamped}`);
        }
        return clamped;
    }
    
    return parsed;
};

const validatePositiveInt = (
    name: string,
    value: string | undefined,
    defaultValue: number
): number => {
    const parsed = parseInt(value || String(defaultValue), 10);
    
    if (isNaN(parsed) || parsed < 1) {
        console.error(`\n无效的 ${name}\n`);
        console.error(`当前值: ${value}`);
        console.error(`必须是正整数\n`);
        throw new Error(`无效的 ${name}: ${value}。必须是正整数。`);
    }
    
    return parsed;
};

/** 非负整数，0 表示不限制（用于每日最大交易次数等） */
const validateNonNegativeInt = (
    name: string,
    value: string | undefined,
    defaultValue: number
): number => {
    const parsed = parseInt(value || String(defaultValue), 10);
    if (isNaN(parsed) || parsed < 0) {
        console.error(`\n无效的 ${name}\n`);
        console.error(`当前值: ${value}`);
        console.error(`必须为非负整数，0 表示不限制\n`);
        throw new Error(`无效的 ${name}: ${value}。必须为非负整数，0=不限制。`);
    }
    return parsed;
};

const validateUrl = (name: string, url: string | undefined): string => {
    if (!url) {
        throw new Error(`缺少必需的配置: ${name}`);
    }
    
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
        console.error(`\n无效的 ${name}\n`);
        console.error(`当前值: ${url}`);
        console.error(`必须以 http:// 或 https:// 开头\n`);
        throw new Error(`无效的 ${name}: ${url}。必须是有效的 HTTP/HTTPS URL。`);
    }
    
    return url;
};

const parseAllowedMarkets = (input: string | undefined): string[] => {
    const defaultMarkets = ['BTC', 'ETH', 'SOL'];
    
    if (!input) {
        return defaultMarkets;
    }
    
    const markets = input
        .split(',')
        .map((m) => m.trim().toUpperCase())
        .filter((m) => m.length > 0);
    
    if (markets.length === 0) {
        return defaultMarkets;
    }
    
    return markets;
};

const validateTimeframe = (tf: string | undefined): string => {
    const validTimeframes = ['5m', '15m', '1h', '4h'];
    const normalized = (tf || '15m').toLowerCase();
    
    if (!validTimeframes.includes(normalized)) {
        console.error('\n无效的 TIMEFRAME\n');
        console.error(`当前值: ${tf}`);
        console.error(`有效值: ${validTimeframes.join(', ')}\n`);
        throw new Error(`无效的 TIMEFRAME: ${tf}。必须是: ${validTimeframes.join(', ')}`);
    }
    
    return normalized;
};

const validateCooldownScope = (scope: string | undefined): 'symbol' | 'symbol_direction' => {
    const raw = (scope || 'symbol').toLowerCase().trim();
    if (raw === 'symbol' || raw === 'symbol_direction') return raw;
    console.log(`  ⚠️ COOLDOWN_SCOPE=${scope} 无效，回退为 symbol`);
    return 'symbol';
};

const parseProbabilityMap = (raw: string | undefined): Record<string, number> => {
    const out: Record<string, number> = {};
    if (!raw) return out;
    for (const chunk of raw.split(',')) {
        const part = chunk.trim();
        if (!part) continue;
        const [symRaw, valRaw] = part.split(':', 2);
        const sym = String(symRaw || '').trim().toUpperCase();
        if (!sym || !valRaw) continue;
        const val = validateNumericRange(`PROB_THRESHOLD_BY_SYMBOL(${sym})`, valRaw, 0.6, 0.5, 0.99);
        out[sym] = val;
    }
    return out;
};

type ConfidenceScaleBand = { min: number; max: number; mult: number };
type BetSizingRule = {
    minProb: number;
    maxProb: number;
    minEdge?: number;
    maxEdge?: number;
    mult?: number;
    cap?: number;
};
type PriceRangeTuple = [number, number];

const parseConfidenceScaleBandMap = (
    raw: string | undefined,
    label: string,
): Record<string, ConfidenceScaleBand[]> => {
    const out: Record<string, ConfidenceScaleBand[]> = {};
    if (!raw) return out;
    let parsed: unknown;
    try {
        parsed = JSON.parse(raw);
    } catch {
        console.log(`  ⚠️ ${label} 不是合法 JSON，已忽略`);
        return out;
    }
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return out;
    for (const [symRaw, bandsRaw] of Object.entries(parsed as Record<string, unknown>)) {
        const sym = String(symRaw || '').trim().toUpperCase();
        if (!sym || !Array.isArray(bandsRaw)) continue;
        const bands: ConfidenceScaleBand[] = [];
        for (const bandRaw of bandsRaw) {
            if (!bandRaw || typeof bandRaw !== 'object' || Array.isArray(bandRaw)) continue;
            const band = bandRaw as Record<string, unknown>;
            const min = validateNumericRange(`${label}(${sym}).min`, String(band.min ?? ''), 0.5, 0.0, 0.99);
            const max = validateNumericRange(`${label}(${sym}).max`, String(band.max ?? ''), 1.0, 0.01, 1.0);
            const mult = validateNumericRange(`${label}(${sym}).mult`, String(band.mult ?? ''), 1.0, 0.0, 10.0);
            if (max > min) bands.push({ min, max, mult });
        }
        if (bands.length > 0) out[sym] = bands;
    }
    return out;
};

const parseBetSizingRuleMap = (
    raw: string | undefined,
    label: string,
): Record<string, BetSizingRule[]> => {
    const out: Record<string, BetSizingRule[]> = {};
    if (!raw) return out;
    let parsed: unknown;
    try {
        parsed = JSON.parse(raw);
    } catch {
        console.log(`  ⚠️ ${label} 不是合法 JSON，已忽略`);
        return out;
    }
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return out;
    for (const [symDirRaw, rulesRaw] of Object.entries(parsed as Record<string, unknown>)) {
        const symDir = String(symDirRaw || '').trim().toUpperCase();
        if (!symDir || !Array.isArray(rulesRaw)) continue;
        const rules: BetSizingRule[] = [];
        for (const ruleRaw of rulesRaw) {
            if (!ruleRaw || typeof ruleRaw !== 'object' || Array.isArray(ruleRaw)) continue;
            const rule = ruleRaw as Record<string, unknown>;
            const minProb = validateNumericRange(`${label}(${symDir}).minProb`, String(rule.minProb ?? ''), 0.5, 0.0, 0.99);
            const maxProb = validateNumericRange(`${label}(${symDir}).maxProb`, String(rule.maxProb ?? ''), 1.0, 0.01, 1.0);
            if (!(maxProb > minProb)) continue;
            const parsedRule: BetSizingRule = { minProb, maxProb };
            if (rule.minEdge != null && rule.minEdge !== '') {
                parsedRule.minEdge = validateNumericRange(`${label}(${symDir}).minEdge`, String(rule.minEdge), 0.0, -1.0, 1.0);
            }
            if (rule.maxEdge != null && rule.maxEdge !== '') {
                parsedRule.maxEdge = validateNumericRange(`${label}(${symDir}).maxEdge`, String(rule.maxEdge), 1.0, -1.0, 1.0);
            }
            if (rule.mult != null && rule.mult !== '') {
                parsedRule.mult = validateNumericRange(`${label}(${symDir}).mult`, String(rule.mult), 1.0, 0.0, 5.0);
            }
            if (rule.cap != null && rule.cap !== '') {
                parsedRule.cap = validateNumericRange(`${label}(${symDir}).cap`, String(rule.cap), 60.0, 0.0, 100000.0);
            }
            rules.push(parsedRule);
        }
        if (rules.length > 0) out[symDir] = rules;
    }
    return out;
};

const parseNumericMap = (
    raw: string | undefined,
    label: string,
    defaultValue: number,
    min: number,
    max: number,
): Record<string, number> => {
    const out: Record<string, number> = {};
    if (!raw) return out;
    let parsed: unknown;
    try {
        parsed = JSON.parse(raw);
    } catch {
        console.log(`  ⚠️ ${label} 不是合法 JSON，已忽略`);
        return out;
    }
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return out;
    for (const [keyRaw, valueRaw] of Object.entries(parsed as Record<string, unknown>)) {
        const key = String(keyRaw || '').trim().toUpperCase();
        if (!key || valueRaw == null || valueRaw === '') continue;
        out[key] = validateNumericRange(`${label}(${key})`, String(valueRaw), defaultValue, min, max);
    }
    return out;
};

const parsePositiveIntMap = (
    raw: string | undefined,
    label: string,
    defaultValue: number,
    min: number,
    max: number,
): Record<string, number> => {
    const out: Record<string, number> = {};
    if (!raw) return out;
    let parsed: unknown;
    try {
        parsed = JSON.parse(raw);
    } catch {
        console.log(`  ⚠️ ${label} 不是合法 JSON，已忽略`);
        return out;
    }
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return out;
    for (const [keyRaw, valueRaw] of Object.entries(parsed as Record<string, unknown>)) {
        const key = String(keyRaw || '').trim().toUpperCase();
        if (!key || valueRaw == null || valueRaw === '') continue;
        const parsedValue = validatePositiveInt(`${label}(${key})`, String(valueRaw), defaultValue);
        out[key] = Math.max(min, Math.min(max, parsedValue));
    }
    return out;
};

const parsePriceRangeMap = (
    raw: string | undefined,
    label: string,
): Record<string, PriceRangeTuple> => {
    const out: Record<string, PriceRangeTuple> = {};
    if (!raw) return out;
    let parsed: unknown;
    try {
        parsed = JSON.parse(raw);
    } catch {
        console.log(`  ⚠️ ${label} 不是合法 JSON，已忽略`);
        return out;
    }
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return out;
    for (const [keyRaw, valueRaw] of Object.entries(parsed as Record<string, unknown>)) {
        const key = String(keyRaw || '').trim().toUpperCase();
        if (!key || !Array.isArray(valueRaw) || valueRaw.length < 2) continue;
        const lower = validateNumericRange(`${label}(${key}).lower`, String(valueRaw[0] ?? ''), 0.30, 0.01, 0.99);
        const upper = validateNumericRange(`${label}(${key}).upper`, String(valueRaw[1] ?? ''), 0.40, 0.01, 0.99);
        if (!(upper >= lower)) continue;
        out[key] = [lower, upper];
    }
    return out;
};

// ============================================================
// 预测交易配置导出
// ============================================================

type EnvRecord = Record<string, string | undefined>;

/**
 * 从环境变量字典构建 PredictionEnvConfig。
 * 多进程合并模式下，每个 trader 传入独立的 envVars（不读 process.env）；
 * 单进程兼容模式下，不传参默认读 process.env。
 */
export function createPredictionEnv(envVars?: EnvRecord) {
    const e: EnvRecord = envVars ?? (process.env as EnvRecord);
    const tradingMode = validateTradingMode(e.TRADING_MODE);
    assertPublicExportNotLive(tradingMode);
    const riskEnabled = (e.RISK_CONTROL_ENABLED || 'false').toLowerCase() === 'true';
    const rawBetPctNormal = validateNumericRange('BET_PCT_NORMAL', e.BET_PCT_NORMAL, 0.05, 0.01, 0.50);
    const rawBetPctConservative = validateNumericRange('BET_PCT_CONSERVATIVE', e.BET_PCT_CONSERVATIVE, 0.03, 0.01, 0.20);
    const simulationBetPctNormalCap = 0.05;
    const simulationBetPctConservativeCap = 0.03;
    const effectiveBetPctNormal = tradingMode === TradingMode.LIVE
        ? rawBetPctNormal
        : Math.min(rawBetPctNormal, simulationBetPctNormalCap);
    const effectiveBetPctConservative = tradingMode === TradingMode.LIVE
        ? rawBetPctConservative
        : Math.min(rawBetPctConservative, simulationBetPctConservativeCap);
    if (tradingMode !== TradingMode.LIVE) {
        if (rawBetPctNormal > effectiveBetPctNormal + 1e-12) {
            console.log(`  ⚠️ BET_PCT_NORMAL=${rawBetPctNormal} 在 ${tradingMode} 模式下已硬截断为 ${effectiveBetPctNormal}`);
        }
        if (rawBetPctConservative > effectiveBetPctConservative + 1e-12) {
            console.log(`  ⚠️ BET_PCT_CONSERVATIVE=${rawBetPctConservative} 在 ${tradingMode} 模式下已硬截断为 ${effectiveBetPctConservative}`);
        }
    }
    return {
        TRADING_MODE: tradingMode,
        SIMULATION_BLOCKING_GUARDS_DISABLED: (
            e.SIMULATION_BLOCKING_GUARDS_DISABLED || 'false'
        ).toLowerCase() === 'true',
        MAX_TRADES_PER_SESSION: validateNonNegativeInt('MAX_TRADES_PER_SESSION', e.MAX_TRADES_PER_SESSION, 20),
        INITIAL_CAPITAL: validateNumericRange('INITIAL_CAPITAL', e.INITIAL_CAPITAL, 400.0, 1.0, 1000000.0),
        PER_COIN_CAPITAL: (e.PER_COIN_CAPITAL || '').toLowerCase() === 'true',
        BET_SIZE_PERCENT: validateNumericRange('BET_SIZE_PERCENT', e.BET_SIZE_PERCENT, 5.0, 0.1, 50.0),
        TIER_P1_PCT: e.TIER_P1_PCT != null && e.TIER_P1_PCT !== ''
            ? validateNumericRange('TIER_P1_PCT', e.TIER_P1_PCT, 2, 0.1, 50) : undefined,
        TIER_P2_PCT: e.TIER_P2_PCT != null && e.TIER_P2_PCT !== ''
            ? validateNumericRange('TIER_P2_PCT', e.TIER_P2_PCT, 4, 0.1, 50) : undefined,
        TIER_P3_PCT: e.TIER_P3_PCT != null && e.TIER_P3_PCT !== ''
            ? validateNumericRange('TIER_P3_PCT', e.TIER_P3_PCT, 6, 0.1, 50) : undefined,
        TIER_P4_PCT: e.TIER_P4_PCT != null && e.TIER_P4_PCT !== ''
            ? validateNumericRange('TIER_P4_PCT', e.TIER_P4_PCT, 8, 0.1, 50) : undefined,
        TIER_B1: e.TIER_B1 != null && e.TIER_B1 !== ''
            ? validateNumericRange('TIER_B1', e.TIER_B1, 2, 0, 20) : 2,
        TIER_B2: e.TIER_B2 != null && e.TIER_B2 !== ''
            ? validateNumericRange('TIER_B2', e.TIER_B2, 4, 0, 20) : 4,
        TIER_B3: e.TIER_B3 != null && e.TIER_B3 !== ''
            ? validateNumericRange('TIER_B3', e.TIER_B3, 6, 0, 20) : 6,
        PROB_THRESHOLD: validateNumericRange('PROB_THRESHOLD', e.PROB_THRESHOLD, 0.6, 0.5, 0.99),
        PROB_THRESHOLD_BY_SYMBOL: parseProbabilityMap(e.PROB_THRESHOLD_BY_SYMBOL),
        CONFIDENCE_SCALE_BANDS_BY_SYMBOL: parseConfidenceScaleBandMap(
            e.CONFIDENCE_SCALE_BANDS_BY_SYMBOL,
            'CONFIDENCE_SCALE_BANDS_BY_SYMBOL',
        ),
        CONFIDENCE_SCALE_BANDS_BY_SYMBOL_DIRECTION: parseConfidenceScaleBandMap(
            e.CONFIDENCE_SCALE_BANDS_BY_SYMBOL_DIRECTION,
            'CONFIDENCE_SCALE_BANDS_BY_SYMBOL_DIRECTION',
        ),
        SLOT01_GENTLE_FILTER_ENABLED: (e.SLOT01_GENTLE_FILTER_ENABLED || 'false').toLowerCase() === 'true',
        SLOT01_GENTLE_FILTER_CONFIDENCE_MIN: e.SLOT01_GENTLE_FILTER_CONFIDENCE_MIN != null && e.SLOT01_GENTLE_FILTER_CONFIDENCE_MIN !== ''
            ? validateNumericRange('SLOT01_GENTLE_FILTER_CONFIDENCE_MIN', e.SLOT01_GENTLE_FILTER_CONFIDENCE_MIN, 0, 0, 1)
            : 0,
        SLOT01_GENTLE_FILTER_CONFIDENCE_MAX: e.SLOT01_GENTLE_FILTER_CONFIDENCE_MAX != null && e.SLOT01_GENTLE_FILTER_CONFIDENCE_MAX !== ''
            ? validateNumericRange('SLOT01_GENTLE_FILTER_CONFIDENCE_MAX', e.SLOT01_GENTLE_FILTER_CONFIDENCE_MAX, 0, 0, 1)
            : 0,
        SLOT01_GENTLE_FILTER_DIRECTION: String(e.SLOT01_GENTLE_FILTER_DIRECTION || 'all').trim().toLowerCase() || 'all',
        BET_SIZING_RULES_BY_SYMBOL_DIRECTION: parseBetSizingRuleMap(
            e.BET_SIZING_RULES_BY_SYMBOL_DIRECTION,
            'BET_SIZING_RULES_BY_SYMBOL_DIRECTION',
        ),
        BET_PCT_NORMAL_BY_SYMBOL_DIRECTION: parseNumericMap(
            e.BET_PCT_NORMAL_BY_SYMBOL_DIRECTION,
            'BET_PCT_NORMAL_BY_SYMBOL_DIRECTION',
            effectiveBetPctNormal,
            0.0,
            1.0,
        ),
        BET_PCT_CONSERVATIVE_BY_SYMBOL_DIRECTION: parseNumericMap(
            e.BET_PCT_CONSERVATIVE_BY_SYMBOL_DIRECTION,
            'BET_PCT_CONSERVATIVE_BY_SYMBOL_DIRECTION',
            effectiveBetPctConservative,
            0.0,
            1.0,
        ),
        BET_TARGET_CAP_USD_BY_SYMBOL_DIRECTION: parseNumericMap(
            e.BET_TARGET_CAP_USD_BY_SYMBOL_DIRECTION,
            'BET_TARGET_CAP_USD_BY_SYMBOL_DIRECTION',
            60.0,
            0.0,
            100000.0,
        ),
        LOWPRICE_SELECTED_BUY_PRICE_RANGE_BY_SYMBOL_DIRECTION: parsePriceRangeMap(
            e.LOWPRICE_SELECTED_BUY_PRICE_RANGE_BY_SYMBOL_DIRECTION,
            'LOWPRICE_SELECTED_BUY_PRICE_RANGE_BY_SYMBOL_DIRECTION',
        ),
        LOWPRICE_RUNG_COUNT_BY_SYMBOL_DIRECTION: parsePositiveIntMap(
            e.LOWPRICE_RUNG_COUNT_BY_SYMBOL_DIRECTION,
            'LOWPRICE_RUNG_COUNT_BY_SYMBOL_DIRECTION',
            9,
            1,
            100,
        ),
        PROB_THRESHOLD_UP: e.PROB_THRESHOLD_UP != null && e.PROB_THRESHOLD_UP !== ''
            ? validateNumericRange('PROB_THRESHOLD_UP', e.PROB_THRESHOLD_UP, 0.65, 0.5, 0.99) : undefined,
        PROB_THRESHOLD_DOWN: e.PROB_THRESHOLD_DOWN != null && e.PROB_THRESHOLD_DOWN !== ''
            ? validateNumericRange('PROB_THRESHOLD_DOWN', e.PROB_THRESHOLD_DOWN, 0.35, 0.01, 0.5) : undefined,
        CONSENSUS_THRESHOLD_UP: e.CONSENSUS_THRESHOLD_UP != null && e.CONSENSUS_THRESHOLD_UP !== ''
            ? validateNumericRange('CONSENSUS_THRESHOLD_UP', e.CONSENSUS_THRESHOLD_UP, 0.65, 0.5, 0.99) : undefined,
        CONSENSUS_THRESHOLD_DOWN: e.CONSENSUS_THRESHOLD_DOWN != null && e.CONSENSUS_THRESHOLD_DOWN !== ''
            ? validateNumericRange('CONSENSUS_THRESHOLD_DOWN', e.CONSENSUS_THRESHOLD_DOWN, 0.72, 0.5, 0.99) : undefined,
        MIN_PROFIT_RATIO: validateNumericRange('MIN_PROFIT_RATIO', e.MIN_PROFIT_RATIO, 0.4, 0, 5),
        MIN_TOKEN_PRICE: validateNumericRange('MIN_TOKEN_PRICE', e.MIN_TOKEN_PRICE, 0.05, 0.01, 0.5),
        MAX_TOKEN_PRICE: validateNumericRange('MAX_TOKEN_PRICE', e.MAX_TOKEN_PRICE, 0.54, 0.01, 0.99),
        PREDICTION_API_URL: e.PREDICTION_API_URL || 'http://localhost:8080',
        ALLOWED_MARKETS: parseAllowedMarkets(e.ALLOWED_MARKETS),
        TIMEFRAME: validateTimeframe(e.TIMEFRAME),
        EARLY_ENTRY_SECONDS: parseInt(e.EARLY_ENTRY_SECONDS || '120', 10) || 120,
        PROXY_WALLET: e.PROXY_WALLET || '',
        PRIVATE_KEY: e.PRIVATE_KEY || '',
        POLY_API_KEY: e.POLY_API_KEY || e.CLOB_API_KEY || '',
        POLY_API_SECRET: e.POLY_API_SECRET || e.CLOB_API_SECRET || '',
        POLY_API_PASSPHRASE: e.POLY_API_PASSPHRASE || e.POLY_PASSPHRASE || e.CLOB_API_PASSPHRASE || '',
        BUILDER_API_KEY: e.BUILDER_API_KEY || '',
        BUILDER_API_SECRET: e.BUILDER_API_SECRET || '',
        BUILDER_API_PASSPHRASE: e.BUILDER_API_PASSPHRASE || '',
        RELAYER_API_KEY: e.RELAYER_API_KEY || '',
        RELAYER_API_KEY_ADDRESS: e.RELAYER_API_KEY_ADDRESS || '',
        CLAIM_GASLESS_ENABLED: (
            e.CLAIM_GASLESS_ENABLED
            || (String(e.TRADING_MODE || '').toLowerCase() === 'live' ? 'true' : 'false')
        ).toLowerCase() === 'true',
        LIVE_CLAIM_SUBMIT_ENABLED: (e.LIVE_CLAIM_SUBMIT_ENABLED || 'false').toLowerCase() === 'true',
        CLAIM_GASLESS_STRICT: (e.CLAIM_GASLESS_STRICT || 'true').toLowerCase() === 'true',
        CLAIM_ALLOW_ONCHAIN_FALLBACK: (e.CLAIM_ALLOW_ONCHAIN_FALLBACK || 'false').toLowerCase() === 'true',
        RELAYER_BASE_URL: e.RELAYER_BASE_URL || 'https://relayer-v2.polymarket.com/',
        CLOB_HTTP_URL: e.CLOB_HTTP_URL || 'https://clob.polymarket.com/',
        CLOB_WS_URL: e.CLOB_WS_URL || 'wss://ws-subscriptions-clob.polymarket.com/ws',
        RPC_URL: e.RPC_URL || '',
        USDC_CONTRACT_ADDRESS: e.USDC_CONTRACT_ADDRESS || '0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB',
        PHASE1_LIMIT_PRICE: validateNumericRange('PHASE1_LIMIT_PRICE', e.PHASE1_LIMIT_PRICE, 0.50, 0.01, 0.99),
        PHASE2_LIMIT_PREMIUM: validateNumericRange('PHASE2_LIMIT_PREMIUM', e.PHASE2_LIMIT_PREMIUM, 0.02, 0.0, 0.10),
        PHASE1_BET_FRACTION: validateNumericRange('PHASE1_BET_FRACTION', e.PHASE1_BET_FRACTION, 0.50, 0.1, 1.0),
        MAX_SWEEP_PRICE: validateNumericRange('MAX_SWEEP_PRICE', e.MAX_SWEEP_PRICE, 0.54, 0.01, 0.99),
        PHASE1_MIN_CONFIDENCE: validateNumericRange('PHASE1_MIN_CONFIDENCE', e.PHASE1_MIN_CONFIDENCE, 0.54, 0.5, 0.99),
        PHASE2_MIN_CONFIDENCE: validateNumericRange('PHASE2_MIN_CONFIDENCE', e.PHASE2_MIN_CONFIDENCE, 0.52, 0.5, 0.99),
        TWO_PHASE_ENABLED: (e.TWO_PHASE_ENABLED || 'false').toLowerCase() === 'true',
        LIMIT_PRICE: validateNumericRange('LIMIT_PRICE', e.LIMIT_PRICE, 0.50, 0.01, 0.99),
        LIMIT_PRICE_LADDER: e.LIMIT_PRICE_LADDER || '',
        LOWPRICE_FAMILY_ID: e.LOWPRICE_FAMILY_ID || '',
        FAMILY_RULE_VERSION: e.FAMILY_RULE_VERSION || '',
        STALE_ORDER_POLICY: (e.STALE_ORDER_POLICY || 'off').toLowerCase(),
        STALE_CANCEL_MIN_AGE_SEC: validateNonNegativeInt('STALE_CANCEL_MIN_AGE_SEC', e.STALE_CANCEL_MIN_AGE_SEC, 0),
        STALE_CANCEL_MIN_DRIFT_TICKS: validateNonNegativeInt('STALE_CANCEL_MIN_DRIFT_TICKS', e.STALE_CANCEL_MIN_DRIFT_TICKS, 0),
        STALE_CANCEL_MIN_TIME_TO_EXPIRY_SEC: validateNonNegativeInt('STALE_CANCEL_MIN_TIME_TO_EXPIRY_SEC', e.STALE_CANCEL_MIN_TIME_TO_EXPIRY_SEC, 0),
        LATE_FILL_CANCEL_MINUTES: validateNonNegativeInt('LATE_FILL_CANCEL_MINUTES', e.LATE_FILL_CANCEL_MINUTES, 0),
        EDGE_FILTER_ENABLED: (e.EDGE_FILTER_ENABLED || 'false').toLowerCase() === 'true',
        MIN_EDGE: validateNumericRange('MIN_EDGE', e.MIN_EDGE, 0.02, 0.001, 0.20),
        MIN_EDGE_UP: e.MIN_EDGE_UP != null && e.MIN_EDGE_UP !== ''
            ? validateNumericRange('MIN_EDGE_UP', e.MIN_EDGE_UP, 0.01638, 0.001, 0.20) : undefined,
        MIN_EDGE_DOWN: e.MIN_EDGE_DOWN != null && e.MIN_EDGE_DOWN !== ''
            ? validateNumericRange('MIN_EDGE_DOWN', e.MIN_EDGE_DOWN, 0.028, 0.001, 0.20) : undefined,
        KELLY_ENABLED: (e.KELLY_ENABLED || 'false').toLowerCase() === 'true',
        KELLY_FRAC: validateNumericRange('KELLY_FRAC', e.KELLY_FRAC, 0.33, 0.01, 2.0),
        BANKROLL_PCT: e.BANKROLL_PCT != null && e.BANKROLL_PCT !== ''
            ? validateNumericRange('BANKROLL_PCT', e.BANKROLL_PCT, effectiveBetPctNormal, 0.0, 1.0) : effectiveBetPctNormal,
        BET_PCT_NORMAL: effectiveBetPctNormal,
        BET_PCT_CONSERVATIVE: effectiveBetPctConservative,
        CAPITAL_RISK_CAP_PCT: e.CAPITAL_RISK_CAP_PCT != null && e.CAPITAL_RISK_CAP_PCT !== ''
            ? validateNumericRange('CAPITAL_RISK_CAP_PCT', e.CAPITAL_RISK_CAP_PCT, 0.95, 0.0, 1.0) : 0.95,
        BET_TARGET_CAP_USD: e.BET_TARGET_CAP_USD != null && e.BET_TARGET_CAP_USD !== ''
            ? validateNumericRange('BET_TARGET_CAP_USD', e.BET_TARGET_CAP_USD, 60.0, 0.0, 100000.0) : undefined,
        RUNTIME_RISK_SCALING_MODE: String(e.RUNTIME_RISK_SCALING_MODE || 'legacy').trim().toLowerCase() || 'legacy',
        RUNTIME_RISK_SCALING_ENTER_DRAWDOWN_PCT: e.RUNTIME_RISK_SCALING_ENTER_DRAWDOWN_PCT != null && e.RUNTIME_RISK_SCALING_ENTER_DRAWDOWN_PCT !== ''
            ? validateNumericRange('RUNTIME_RISK_SCALING_ENTER_DRAWDOWN_PCT', e.RUNTIME_RISK_SCALING_ENTER_DRAWDOWN_PCT, 0.20, 0.0, 0.95) : undefined,
        RUNTIME_RISK_SCALING_RECOVER_DRAWDOWN_PCT: e.RUNTIME_RISK_SCALING_RECOVER_DRAWDOWN_PCT != null && e.RUNTIME_RISK_SCALING_RECOVER_DRAWDOWN_PCT !== ''
            ? validateNumericRange('RUNTIME_RISK_SCALING_RECOVER_DRAWDOWN_PCT', e.RUNTIME_RISK_SCALING_RECOVER_DRAWDOWN_PCT, 0.18, 0.0, 0.95) : undefined,
        SIZING_MODE: String(e.SIZING_MODE || 'percent_cap').trim().toLowerCase() || 'percent_cap',
        FIXED_TARGET_NOTIONAL_USD: e.FIXED_TARGET_NOTIONAL_USD != null && e.FIXED_TARGET_NOTIONAL_USD !== ''
            ? validateNumericRange('FIXED_TARGET_NOTIONAL_USD', e.FIXED_TARGET_NOTIONAL_USD, 50.0, 0.0, 100000.0) : undefined,
        REQUEST_COMPRESSION_MODE: String(e.REQUEST_COMPRESSION_MODE || 'legacy').trim().toLowerCase() || 'legacy',
        EXECUTION_COMPRESSION_MODE: String(e.EXECUTION_COMPRESSION_MODE || 'legacy').trim().toLowerCase() || 'legacy',
        EXECUTION_PROFILE_VERSION: String(e.EXECUTION_PROFILE_VERSION || 'v1').trim() || 'v1',
        EXECUTION_V2_SHADOW_ONLY: (e.EXECUTION_V2_SHADOW_ONLY || 'false').toLowerCase() === 'true',
        CHASE_ENABLED: (e.CHASE_ENABLED || 'false').toLowerCase() === 'true',
        CHASE_ROUNDS: validateNonNegativeInt('CHASE_ROUNDS', e.CHASE_ROUNDS, 1),
        CHASE_WAIT_SEC: validateNonNegativeInt('CHASE_WAIT_SEC', e.CHASE_WAIT_SEC, 5),
        FIRST_CHASE_WAIT_SEC: validateNonNegativeInt('FIRST_CHASE_WAIT_SEC', e.FIRST_CHASE_WAIT_SEC, Math.max(0, parseInt(e.CHASE_WAIT_SEC || '35', 10) || 35)),
        NEXT_CHASE_GAP_SEC: validateNonNegativeInt('NEXT_CHASE_GAP_SEC', e.NEXT_CHASE_GAP_SEC, 20),
        CHASE_TICK_LIFT: validateNonNegativeInt('CHASE_TICK_LIFT', e.CHASE_TICK_LIFT, 3),
        CHASE_ABSOLUTE_MAX_PRICE: validateNumericRange('CHASE_ABSOLUTE_MAX_PRICE', e.CHASE_ABSOLUTE_MAX_PRICE, 0.50, 0.01, 0.99),
        CHASE_MIN_CONFIDENCE_BAND: validateNumericRange('CHASE_MIN_CONFIDENCE_BAND', e.CHASE_MIN_CONFIDENCE_BAND, 0.80, 0.0, 1.0),
        CHASE_MIN_TIME_TO_EXPIRY_SEC: validateNonNegativeInt('CHASE_MIN_TIME_TO_EXPIRY_SEC', e.CHASE_MIN_TIME_TO_EXPIRY_SEC, 120),
        DIRECTION_CONTINUATION_TICKS: validateNonNegativeInt('DIRECTION_CONTINUATION_TICKS', e.DIRECTION_CONTINUATION_TICKS, 1),
        DIRECTION_CONTINUATION_WINDOW_SEC: validateNonNegativeInt('DIRECTION_CONTINUATION_WINDOW_SEC', e.DIRECTION_CONTINUATION_WINDOW_SEC, 10),
        FINAL_AGGRESSIVE_WINDOW_SEC: validateNonNegativeInt('FINAL_AGGRESSIVE_WINDOW_SEC', e.FINAL_AGGRESSIVE_WINDOW_SEC, 120),
        FINAL_AGGRESSIVE_TICK_LIFT: validateNonNegativeInt('FINAL_AGGRESSIVE_TICK_LIFT', e.FINAL_AGGRESSIVE_TICK_LIFT, 2),
        FINAL_AGGRESSIVE_MAX_PRICE: validateNumericRange('FINAL_AGGRESSIVE_MAX_PRICE', e.FINAL_AGGRESSIVE_MAX_PRICE, 0.55, 0.01, 0.99),
        CANCEL_ON_ADVERSE_MOVE_TICKS: validateNonNegativeInt('CANCEL_ON_ADVERSE_MOVE_TICKS', e.CANCEL_ON_ADVERSE_MOVE_TICKS, 2),
        CANCEL_ON_SIGNAL_INVALIDATION: (e.CANCEL_ON_SIGNAL_INVALIDATION || 'true').toLowerCase() === 'true',
        CANCEL_ON_QUEUE_STALL: (e.CANCEL_ON_QUEUE_STALL || 'false').toLowerCase() === 'true',
        QUEUE_STALL_WINDOW_SEC: validateNonNegativeInt('QUEUE_STALL_WINDOW_SEC', e.QUEUE_STALL_WINDOW_SEC, 35),
        CONF_TIER1_BOUND: validateNumericRange('CONF_TIER1_BOUND', e.CONF_TIER1_BOUND, 0.52, 0.50, 0.60),
        CONF_TIER2_BOUND: validateNumericRange('CONF_TIER2_BOUND', e.CONF_TIER2_BOUND, 0.56, 0.50, 0.70),
        TIER1_MULT: validateNumericRange('TIER1_MULT', e.TIER1_MULT, 0.30, 0.01, 2.0),
        TIER2_MULT: validateNumericRange('TIER2_MULT', e.TIER2_MULT, 0.60, 0.01, 2.0),
        TIER3_MULT: validateNumericRange('TIER3_MULT', e.TIER3_MULT, 1.00, 0.01, 3.0),
        DRAWDOWN_HALT: validateNumericRange('DRAWDOWN_HALT', e.DRAWDOWN_HALT, 0.30, 0.05, 0.80),
        COOLDOWN_AFTER_LOSSES: validatePositiveInt('COOLDOWN_AFTER_LOSSES', e.COOLDOWN_AFTER_LOSSES, 1),
        COOLDOWN_BARS: validateNonNegativeInt('COOLDOWN_BARS', e.COOLDOWN_BARS, 0),
        COOLDOWN_SCOPE: validateCooldownScope(e.COOLDOWN_SCOPE),
        NUM_TRADING_ASSETS: validateNonNegativeInt('NUM_TRADING_ASSETS', e.NUM_TRADING_ASSETS, 3),
        REQUEST_TIMEOUT_MS: parseInt(e.REQUEST_TIMEOUT_MS || '10000', 10) || 10000,
        NETWORK_RETRY_LIMIT: parseInt(e.NETWORK_RETRY_LIMIT || '3', 10) || 3,
        RISK_CONTROL_ENABLED: riskEnabled,
        STOP_LOSS_PCT: e.STOP_LOSS_PCT != null && e.STOP_LOSS_PCT !== ''
            ? validateNumericRange('STOP_LOSS_PCT', e.STOP_LOSS_PCT, 0.3, 0, 0.9) : undefined,
        CONSECUTIVE_LOSS_LIMIT: riskEnabled && e.CONSECUTIVE_LOSS_LIMIT
            ? validatePositiveInt('CONSECUTIVE_LOSS_LIMIT', e.CONSECUTIVE_LOSS_LIMIT, 5) : undefined,
        RISK_PAUSE_K_BARS: riskEnabled && e.RISK_PAUSE_K_BARS
            ? validatePositiveInt('RISK_PAUSE_K_BARS', e.RISK_PAUSE_K_BARS, 8) : undefined,
    };
}

/** 配置对象的类型（供 PredictionExecutor 等作为 this.env 的类型声明） */
export type PredictionEnvConfig = ReturnType<typeof createPredictionEnv>;

/** 默认全局实例（从 process.env 构建，兼容单进程模式） */
export const PREDICTION_ENV: PredictionEnvConfig = createPredictionEnv();

// ============================================================
// 配置验证和打印
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

export const validatePredictionConfig = (env: PredictionEnvConfig = PREDICTION_ENV): void => {
    if (env.BET_SIZE_PERCENT <= 0 || env.BET_SIZE_PERCENT > 50) {
        throw new Error(
            `BET_SIZE_PERCENT (${env.BET_SIZE_PERCENT}) 必须在 0.1% - 50% 之间`
        );
    }
    
    if (env.TRADING_MODE === TradingMode.LIVE) {
        assertPublicExportNotLive(env.TRADING_MODE);
        if (!env.PROXY_WALLET || !env.PRIVATE_KEY) {
            console.error('\n真实交易模式需要钱包配置\n');
            console.error('缺少: .env 文件中的 PROXY_WALLET 和/或 PRIVATE_KEY\n');
            throw new Error('真实交易模式需要 PROXY_WALLET 和 PRIVATE_KEY');
        }
        
        if (!env.RPC_URL) {
            console.error('\n真实交易模式需要 RPC_URL\n');
            throw new Error('真实交易模式需要 RPC_URL');
        }

        if (env.CLAIM_GASLESS_ENABLED) {
            if (!env.BUILDER_API_KEY || !env.BUILDER_API_SECRET || !env.BUILDER_API_PASSPHRASE) {
                console.warn('\n⚠️ CLAIM_GASLESS_ENABLED=true，但 BUILDER_API_* 不完整。将无法使用 relayer 免 gas claim。');
            }
        }
    }
};

export const printPredictionConfig = (env: PredictionEnvConfig = PREDICTION_ENV, logsDir?: string): void => {
    const modeColors: Record<TradingMode, string> = {
        [TradingMode.BACKTEST]: '\x1b[33m',  // Yellow
        [TradingMode.SIMULATION]: '\x1b[36m', // Cyan
        [TradingMode.LIVE]: '\x1b[31m',       // Red
    };
    const reset = '\x1b[0m';
    const color = modeColors[env.TRADING_MODE];
    const modeText = getTradingModeText(env.TRADING_MODE);
    
    console.log('\n' + '═'.repeat(60));
    console.log('  📊 预测交易配置');
    console.log('═'.repeat(60) + '\n');
    
    const ld = logsDir ?? process.env.LOGS_DIR ?? 'logs';
    const isTuned = ld.includes('超参');
    console.log(`  日志目录:     ${ld}${isTuned ? ' 【超参组合】' : ''}`);
    console.log(`  交易模式:     ${color}${modeText.toUpperCase()}${reset}`);
    console.log(`  每日最大交易: ${env.MAX_TRADES_PER_SESSION === 0 ? '不限' : env.MAX_TRADES_PER_SESSION + ' 笔'}`);
    console.log(`  初始资金:     $${env.INITIAL_CAPITAL.toFixed(2)}${env.PER_COIN_CAPITAL ? ` (按币种隔离: ${env.ALLOWED_MARKETS.length}币×$${(env.INITIAL_CAPITAL / env.ALLOWED_MARKETS.length).toFixed(0)})` : ''}`);
    const tiered = env.TIER_P1_PCT != null && env.TIER_P2_PCT != null && env.TIER_P3_PCT != null && env.TIER_P4_PCT != null;
    if (env.KELLY_ENABLED) {
        console.log(`  下注模式:     Kelly 动态仓位`);
        console.log(`    KELLY_FRAC=${env.KELLY_FRAC}, BET_PCT_NORMAL=${(env.BET_PCT_NORMAL * 100).toFixed(2)}%, BET_PCT_CONSERVATIVE=${(env.BET_PCT_CONSERVATIVE * 100).toFixed(2)}%`);
        console.log(`    分档: T1(<${env.CONF_TIER1_BOUND})x${env.TIER1_MULT}, T2(<${env.CONF_TIER2_BOUND})x${env.TIER2_MULT}, T3(>=${env.CONF_TIER2_BOUND})x${env.TIER3_MULT}`);
        const bandMap = (env as PredictionEnvConfig & { CONFIDENCE_SCALE_BANDS_BY_SYMBOL?: Record<string, ConfidenceScaleBand[]> }).CONFIDENCE_SCALE_BANDS_BY_SYMBOL || {};
        if (Object.keys(bandMap).length > 0) {
            const bandText = Object.entries(bandMap)
                .map(([sym, bands]) => `${sym}:${bands.map((b) => `[${b.min},${b.max})x${b.mult}`).join('|')}`)
                .join(', ');
            console.log(`    口袋缩放: ${bandText}`);
        }
        const bandDirMap = (env as PredictionEnvConfig & { CONFIDENCE_SCALE_BANDS_BY_SYMBOL_DIRECTION?: Record<string, ConfidenceScaleBand[]> }).CONFIDENCE_SCALE_BANDS_BY_SYMBOL_DIRECTION || {};
        if (Object.keys(bandDirMap).length > 0) {
            const bandText = Object.entries(bandDirMap)
                .map(([symDir, bands]) => `${symDir}:${bands.map((b) => `[${b.min},${b.max})x${b.mult}`).join('|')}`)
                .join(', ');
            console.log(`    方向口袋缩放: ${bandText}`);
        }
        const sizingRuleMap = (env as PredictionEnvConfig & { BET_SIZING_RULES_BY_SYMBOL_DIRECTION?: Record<string, BetSizingRule[]> }).BET_SIZING_RULES_BY_SYMBOL_DIRECTION || {};
        if (Object.keys(sizingRuleMap).length > 0) {
            const ruleText = Object.entries(sizingRuleMap)
                .map(([symDir, rules]) => `${symDir}:${rules.map((r) => `[${r.minProb},${r.maxProb}) edge(${r.minEdge ?? '-inf'},${r.maxEdge ?? '+inf'}) x${r.mult ?? 1} cap=${r.cap ?? '-'}`).join('|')}`)
                .join(', ');
            console.log(`    方向分层金额: ${ruleText}`);
        }
        console.log(`    Edge过滤: ${env.EDGE_FILTER_ENABLED ? '开启' : '关闭'} (MIN_EDGE=${env.MIN_EDGE})`);
        console.log(`    冷却: ${env.COOLDOWN_AFTER_LOSSES} 连输后跳过 ${env.COOLDOWN_BARS} 根K线 (可与回撤熔断协同；绝对止损底线已按当前策略关闭)`);
    } else if (tiered) {
        console.log(`  下注比例:     档位 p1～p4: ${env.TIER_P1_PCT}%,${env.TIER_P2_PCT}%,${env.TIER_P3_PCT}%,${env.TIER_P4_PCT}% (B1=${env.TIER_B1},B2=${env.TIER_B2},B3=${env.TIER_B3})`);
    } else {
        console.log(`  下注比例:     ${env.BET_SIZE_PERCENT}%`);
    }
    const dualThreshold = env.PROB_THRESHOLD_UP != null && env.PROB_THRESHOLD_DOWN != null;
    if (dualThreshold) {
        console.log(`  置信度阈值:   UP≥${(env.PROB_THRESHOLD_UP! * 100).toFixed(0)}%, DOWN≥${((1 - env.PROB_THRESHOLD_DOWN!) * 100).toFixed(0)}% (即prob<${(env.PROB_THRESHOLD_DOWN! * 100).toFixed(0)}%) (双阈值模式)`);
    } else {
        console.log(`  置信度阈值:   ${(env.PROB_THRESHOLD * 100).toFixed(0)}% (押注方向的概率，≥ 才下单)`);
    }
    console.log(`  最小利润比:   ${env.MIN_PROFIT_RATIO} (可能利润 (1/价)-1 ≥ 此值才下单，避免 $1 等无利可图)`);
    console.log(`  最低代币价:   $${env.MIN_TOKEN_PRICE} (买入价低于此视为市场已近乎定价，跳过)`);
    console.log(`  最高代币价:   $${env.MAX_TOKEN_PRICE} (买入价超过此进入监控，回落再买；所有组合统一)`);
    console.log(`  交易市场:     ${env.ALLOWED_MARKETS.join(', ')}`);
    console.log(`  时间周期:     ${env.TIMEFRAME}`);
    console.log(`  提前下单:     ${env.EARLY_ENTRY_SECONDS} 秒`);
    
    if (env.TWO_PHASE_ENABLED) {
        console.log(`\n  🔄 两阶段限价单: 已启用`);
        console.log(`    Phase 1 限价: $${env.PHASE1_LIMIT_PRICE} (仓位 ${(env.PHASE1_BET_FRACTION * 100).toFixed(0)}%, 最低置信度 ${(env.PHASE1_MIN_CONFIDENCE * 100).toFixed(0)}%)`);
        console.log(`    Phase 2 溢价: $${env.PHASE2_LIMIT_PREMIUM} (最低置信度 ${(env.PHASE2_MIN_CONFIDENCE * 100).toFixed(0)}%)`);
        console.log(`    扫单上限:     $${env.MAX_SWEEP_PRICE}`);
    } else if (env.LIMIT_PRICE_LADDER) {
        const ladderPrices = env.LIMIT_PRICE_LADDER.split(',').map(s => s.trim());
        console.log(`\n  🪜 阶梯限价单: 已启用 (${ladderPrices.length} 档)`);
        console.log(`    价格: ${ladderPrices.map(p => '$' + p).join(', ')}`);
    } else {
        console.log(`\n  📝 单限价单: $${env.LIMIT_PRICE.toFixed(2)}`);
    }

    if (env.LOWPRICE_FAMILY_ID || env.FAMILY_RULE_VERSION) {
        console.log(`  lowprice family: ${env.LOWPRICE_FAMILY_ID || '-'} | rule version: ${env.FAMILY_RULE_VERSION || '-'}`);
    }
    if (env.STALE_ORDER_POLICY && env.STALE_ORDER_POLICY !== 'off') {
        console.log(`  stale 挂单策略: ${env.STALE_ORDER_POLICY} (age≥${env.STALE_CANCEL_MIN_AGE_SEC}s, drift≥${env.STALE_CANCEL_MIN_DRIFT_TICKS} ticks, tte≥${env.STALE_CANCEL_MIN_TIME_TO_EXPIRY_SEC}s)`);
    }
    if (env.EXECUTION_PROFILE_VERSION === 'v2' || env.CHASE_ENABLED) {
        console.log(`  执行配置:     ${env.EXECUTION_PROFILE_VERSION}${env.EXECUTION_V2_SHADOW_ONLY ? ' (shadow_only)' : ''}`);
        console.log(`    追价: enabled=${env.CHASE_ENABLED} rounds=${env.CHASE_ROUNDS} first_wait=${env.FIRST_CHASE_WAIT_SEC}s next_gap=${env.NEXT_CHASE_GAP_SEC}s lift=${env.CHASE_TICK_LIFT}t max=$${env.CHASE_ABSOLUTE_MAX_PRICE.toFixed(2)}`);
        console.log(`    强信号门: conf≥${(env.CHASE_MIN_CONFIDENCE_BAND * 100).toFixed(0)}% tte≥${env.CHASE_MIN_TIME_TO_EXPIRY_SEC}s continuation=${env.DIRECTION_CONTINUATION_TICKS}t/${env.DIRECTION_CONTINUATION_WINDOW_SEC}s`);
        console.log(`    最后窗口: ${env.FINAL_AGGRESSIVE_WINDOW_SEC}s lift=${env.FINAL_AGGRESSIVE_TICK_LIFT}t max=$${env.FINAL_AGGRESSIVE_MAX_PRICE.toFixed(2)}`);
        console.log(`    取消条件: adverse_move=${env.CANCEL_ON_ADVERSE_MOVE_TICKS}t signal_invalidation=${env.CANCEL_ON_SIGNAL_INVALIDATION} queue_stall=${env.CANCEL_ON_QUEUE_STALL}/${env.QUEUE_STALL_WINDOW_SEC}s`);
    }

    const drawdownHaltEnabled = env.RISK_CONTROL_ENABLED && env.DRAWDOWN_HALT > 0;
    const runtimeGuardEnabled = env.KELLY_ENABLED || env.COOLDOWN_BARS > 0 || drawdownHaltEnabled;
    if (env.RISK_CONTROL_ENABLED) {
        console.log(`\n  🛡️  风险控制: 已启用（仅动态下注，无连续亏损暂停）`);
        if (env.STOP_LOSS_PCT != null && env.STOP_LOSS_PCT > 0) {
            console.log(`    单笔止损参数: 已配置 ${(env.STOP_LOSS_PCT * 100).toFixed(0)}%（当前版本不执行自动卖出，仅保留兼容）`);
        }
        console.log(`    动态下注: 回撤 > 20% 时减半，> 10% 时减少 25%`);
    } else if (runtimeGuardEnabled) {
        const enabledGuards = [
            env.KELLY_ENABLED ? 'Kelly' : null,
            drawdownHaltEnabled ? '回撤熔断' : null,
            env.COOLDOWN_BARS > 0 ? '冷却' : null,
        ].filter(Boolean).join('/');
        console.log(`\n  🛡️  运行时风控: 已启用（${enabledGuards} 生效；旧版 RISK_CONTROL_ENABLED 未启用）`);
        if (env.KELLY_ENABLED) {
            console.log(`    Kelly仓位: 已启用 (frac=${env.KELLY_FRAC})`);
        }
        if (drawdownHaltEnabled) {
            console.log(`    回撤熔断: ${(env.DRAWDOWN_HALT * 100).toFixed(1)}%`);
        }
        if (env.COOLDOWN_BARS > 0) {
            console.log(`    冷却机制: ${env.COOLDOWN_AFTER_LOSSES} 连输后跳过 ${env.COOLDOWN_BARS} 根K线`);
        }
        console.log(`    绝对止损底线: 已按当前 rollout 策略全局关闭`);
    } else {
        if (env.STOP_LOSS_PCT != null && env.STOP_LOSS_PCT > 0) {
            console.log(`\n  🛡️  单笔止损参数: 已配置 ${(env.STOP_LOSS_PCT * 100).toFixed(0)}%（当前版本不执行自动卖出，仅保留兼容）`);
        } else {
            console.log(`\n  ⚠️  风险控制: 未启用（建议设置 RISK_CONTROL_ENABLED=true 或 STOP_LOSS_PCT）`);
        }
    }
    
    if (env.TRADING_MODE === TradingMode.LIVE) {
        console.log(`\n  ⚠️  真实交易模式 - 将使用真实资金！`);
        console.log(`  钱包地址:     ${env.PROXY_WALLET.slice(0, 10)}...`);
        const claimMode = env.CLAIM_GASLESS_ENABLED
            ? `gasless(${env.CLAIM_GASLESS_STRICT ? 'strict' : 'non_strict'}${env.CLAIM_ALLOW_ONCHAIN_FALLBACK ? '+onchain_fallback' : ''})`
            : 'onchain_only';
        console.log(`  结算赎回:     ${claimMode}`);
        console.log(`  Claim 提交权: ${env.LIVE_CLAIM_SUBMIT_ENABLED ? '交易主循环' : '独立 claim daemon'}`);
    }
    
    console.log('\n' + '═'.repeat(60) + '\n');
};

// 运行验证
try {
    validatePredictionConfig();
} catch (error) {
    // 在模块加载时不抛出错误，让调用方决定如何处理
    console.error(`配置警告: ${error instanceof Error ? error.message : String(error)}`);
}
