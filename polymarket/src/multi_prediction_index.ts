/**
 * 多 Trader 编排器 — 一个 Node.js 进程运行多个 PredictionExecutor 实例
 *
 * 用法:
 *   npx ts-node src/multi_prediction_index.ts --group v5_exp10
 *   npx ts-node src/multi_prediction_index.ts --group gru_all
 *
 * 设计要点（对应风险 A-I）:
 *   - 每个 trader 使用独立的 PredictionEnvConfig（风险 A/C）
 *   - 每个 trader 独立 executedTargetTss + 持久化路径（风险 B）
 *   - Promise.all 并行扫描 + findAllMarkets 缓存（风险 D/I）
 *   - per-trader try-catch，单个崩溃不影响其他（风险 E）
 *   - 不调用 startScheduler()，信号处理只在此处注册（风险 G/H）
 *   - 只使用参数化版本的预测文件读取和 thresholdMet（风险 A/F）
 */

import * as path from 'path';
import * as fs from 'fs';
import { execSync } from 'child_process';

// ── 系统代理自动检测（与 prediction_index.ts 一致）──
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
    } catch { /* 无代理或检测失败 */ }
}

import PredictionExecutor from './services/prediction_executor';
import { MarketOrderbookWS } from './services/market_orderbook_ws';
import { UserChannelWS } from './services/user_channel_ws';
import { createPredictionEnv, PredictionEnvConfig, TradingMode } from './config/prediction_env';
import {
    readLocalPredictionsRawFromFile,
    parseLocalPredictions,
    PredictionResult,
    effectiveConfidence,
    thresholdMetWith,
} from './utils/prediction_api';

// ============================================================
// 类型定义
// ============================================================

interface TraderConfig {
    name: string;
    group: string;
    profile?: 'default' | '70';
    runtimeBaseName?: string;
    runtimeBaseLogsDir?: string;
    runtimeSymbol?: string;
    runtimeSplitBySymbol?: boolean;
    runtimeIndependentCellCapital?: boolean;
    runtimeSeedCapitalFromReports?: boolean;
    predictionSuffix: string;
    predictionFallbackSuffix?: string;
    predictionPath?: string;
    predictionFallbackPath?: string;
    predictionSuffixBySymbol?: Record<string, string>;
    predictionPathBySymbol?: Record<string, string>;
    logsDir: string;
    timeframe?: string;
    runnerManaged?: boolean;
    simulationSource?: string;
    predictionSourceStatus?: string;
    nonRunnableReason?: string;
    // 阈值和交易规则可来自 JSON 文件或直接配置
    rulesJsonPath?: string;
    probThreshold?: number;
    probThresholdBySymbol?: Record<string, number>;
    confidenceScaleBandsBySymbol?: Record<string, Array<{ min: number; max: number; mult: number }>>;
    confidenceScaleBandsBySymbolDirection?: Record<string, Array<{ min: number; max: number; mult: number }>>;
    slot01GentleFilterEnabled?: boolean;
    slot01GentleFilterConfidenceMin?: number;
    slot01GentleFilterConfidenceMax?: number;
    slot01GentleFilterDirection?: 'all' | 'UP' | 'DOWN' | string;
    slot02Aux1hEnabled?: boolean;
    slot02Aux1hPredictionPath?: string;
    slot02Aux1hMode?: 'hard_veto' | 'hard_veto_with_confidence' | 'soft_to_4' | 'soft_to_3' | 'support_quality_v1' | string;
    slot02Aux1hStrengthThreshold?: number;
    slot02Aux1hUpStrengthThreshold?: number;
    slot02Aux1hDownStrengthThreshold?: number;
    slot02Aux1hMainConfidenceThreshold?: number;
    slot02Aux1hFreshnessSec?: number;
    slot02Aux1hWeakSupportThreshold?: number;
    slot02Aux1hLowConfidenceThreshold?: number;
    slot02Aux1hHighConfidenceThreshold?: number;
    slot02Aux1hLowVolThreshold?: number;
    slot02Aux1hHighVolThreshold?: number;
    slot02Aux1hWeakSupportMinVolBucket?: number;
    slot02Aux1hReverseActions?: [number, number, number] | number[];
    slot02Aux1hWeakSupportActions?: [number, number, number] | number[];
    betSizingRulesBySymbolDirection?: Record<string, Array<{ minProb: number; maxProb: number; minEdge?: number; maxEdge?: number; mult?: number; cap?: number }>>;
    betPctNormalBySymbolDirection?: Record<string, number>;
    betPctConservativeBySymbolDirection?: Record<string, number>;
    betTargetCapUsdBySymbolDirection?: Record<string, number>;
    lowPriceSelectedBuyPriceRangeBySymbolDirection?: Record<string, [number, number]>;
    lowPriceRungCountBySymbolDirection?: Record<string, number>;
    core10ExperimentalTuningBySymbol?: Record<string, boolean | { active?: boolean; sources?: string[]; updatedAt?: string }>;
    core10ExperimentalLiveApprovalBySymbol?: Record<string, boolean>;
    probThresholdUp?: number;
    probThresholdDown?: number;
    consensusThresholdUp?: number;
    consensusThresholdDown?: number;
    minEdgeUp?: number;
    minEdgeDown?: number;
    downThresholdDelta?: number;
    limitPrice?: number;
    usePredictionLimitPrice?: boolean;
    limitPriceLadder?: string;
    lowPriceBypassMarketSelection?: boolean;
    lowPriceBypassThresholdBase?: boolean;
    lowPriceSourceResolvedBuyPrice?: number;
    lowPriceSourceResolvedBuyPriceRange?: number[];
    lowPriceSourceSizingReferencePrice?: number;
    lowPriceDynamicMinTotalAmountUsd?: number;
    lowPriceFinalistKey?: string;
    lowpriceFamilyId?: string;
    familyRuleVersion?: string;
    staleOrderPolicy?: 'off' | 'cancel_only_when_stale';
    staleCancelMinAgeSec?: number;
    staleCancelMinDriftTicks?: number;
    staleCancelMinTimeToExpirySec?: number;
    lateFillCancelMinutes?: number;
    maxTokenPrice?: number;
    allowedMarkets: string;
    initialCapital: number;
    liveSelectedCapitalBySymbol?: Record<string, number>;
    liveSelectedCapitalReasonBySymbol?: Record<string, string>;
    liveSelectedRuntimeOverrideBySymbol?: Record<string, Partial<TraderConfig>>;
    liveSelectedRuntimeOverrideSourceBySymbol?: Record<string, string>;
    liveSelectedRuntimeOverrideScopeBySymbol?: Record<string, string>;
    liveSelectedRuntimeOverrideActiveSource?: string;
    liveSelectedRuntimeOverrideActiveScope?: string;
    perCoinCapital?: boolean;
    riskControlEnabled: boolean;
    simulationBlockingGuardsDisabled?: boolean;
    edgeFilterEnabled?: boolean;
    kellyEnabled?: boolean;
    betSizePercent?: number;
    cooldownAfterLosses?: number;
    cooldownBars?: number;
    cooldownScope?: 'symbol' | 'symbol_direction';
    numTradingAssets?: number;
    minEdge?: number;
    kellyFrac?: number;
    bankrollPct?: number;
    betPctNormal?: number;
    betPctConservative?: number;
    capitalRiskCapPct?: number;
    betTargetCapUsd?: number;
    runtimeRiskScalingMode?: string;
    runtimeRiskScalingEnterDrawdownPct?: number;
    runtimeRiskScalingRecoverDrawdownPct?: number;
    sizingMode?: string;
    fixedTargetNotionalUsd?: number;
    requestCompressionMode?: string;
    executionCompressionMode?: string;
    directionLossCooldownMode?: 'off' | 'shadow' | 'enforce';
    directionLossCooldownModeBySymbolDirection?: Record<string, 'off' | 'shadow' | 'enforce'>;
    decisionMinute?: number;
    forcedDirectionMode?: string;
    featureVersion?: string;
    executionProfileVersion?: string;
    executionV2ShadowOnly?: boolean;
    chaseEnabled?: boolean;
    chaseRounds?: number;
    chaseWaitSec?: number;
    firstChaseWaitSec?: number;
    nextChaseGapSec?: number;
    chaseTickLift?: number;
    chaseAbsoluteMaxPrice?: number;
    chaseMinConfidenceBand?: number;
    chaseMinTimeToExpirySec?: number;
    directionContinuationTicks?: number;
    directionContinuationWindowSec?: number;
    finalAggressiveWindowSec?: number;
    finalAggressiveTickLift?: number;
    finalAggressiveMaxPrice?: number;
    cancelOnAdverseMoveTicks?: number;
    cancelOnSignalInvalidation?: boolean;
    cancelOnQueueStall?: boolean;
    queueStallWindowSec?: number;
    confTier1Bound?: number;
    confTier2Bound?: number;
    tier1Mult?: number;
    tier2Mult?: number;
    tier3Mult?: number;
    drawdownHalt?: number;
    tierP1Pct?: number;
    tierP2Pct?: number;
    tierP3Pct?: number;
    tierP4Pct?: number;
    downBreakerEnabled?: boolean;
    downBreakerLookbackHours?: number;
    downBreakerMinTrades?: number;
    downBreakerMinWinrate?: number;
    downBreakerMinPnlPerTrade?: number;
    downBreakerHoldBars?: number;
    downBreakerCheckSeconds?: number;
    downRiskEngine?: 'v1' | 'v2';
    downRiskMode?: 'off' | 'shadow' | 'enforce';
    downRiskModeBySymbol?: Record<string, 'off' | 'shadow' | 'enforce'>;
    downRiskLookbackHoursMain?: number;
    downRiskLookbackHoursCalib?: number;
    downRiskMinTradesSoft?: number;
    downRiskMinTradesHard?: number;
    downRiskLookbackHoursFast?: number;
    downRiskMinTradesFast?: number;
    downRiskWrFastBlocked?: number;
    downRiskPnlFastBlocked?: number;
    downRiskWrSoft?: number;
    downRiskWrHard?: number;
    downRiskPnlSoft?: number;
    downRiskPnlHard?: number;
    downRiskSoftExtraDelta?: number;
    downRiskSoftBetScale?: number;
    downRiskHardHoldBars?: number;
    downRiskReleaseChecks?: number;
    downRiskReleaseWr?: number;
    downRiskReleasePnl?: number;
    downRiskReleaseConfirmHours?: number;
    downRiskCheckSeconds?: number;
    downRiskStatsMode?: 'simulation' | 'live' | 'route';
    downRiskBySymbol?: Record<string, Partial<{
        downRiskLookbackHoursMain: number;
        downRiskLookbackHoursCalib: number;
        downRiskMinTradesSoft: number;
        downRiskMinTradesHard: number;
        downRiskLookbackHoursFast: number;
        downRiskMinTradesFast: number;
        downRiskWrFastBlocked: number;
        downRiskPnlFastBlocked: number;
        downRiskWrSoft: number;
        downRiskWrHard: number;
        downRiskPnlSoft: number;
        downRiskPnlHard: number;
        downRiskSoftExtraDelta: number;
        downRiskSoftBetScale: number;
        downRiskHardHoldBars: number;
        downRiskReleaseChecks: number;
        downRiskReleaseWr: number;
        downRiskReleasePnl: number;
        downRiskReleaseConfirmHours: number;
    }>>;
    upRiskEngine?: 'v1' | 'v2';
    upRiskMode?: 'off' | 'shadow' | 'enforce';
    upRiskModeBySymbol?: Record<string, 'off' | 'shadow' | 'enforce'>;
    drawdownAccelerationMode?: 'off' | 'shadow' | 'enforce';
    drawdownAccelerationModeBySymbol?: Record<string, 'off' | 'shadow' | 'enforce'>;
    upRiskLookbackHoursMain?: number;
    upRiskLookbackHoursCalib?: number;
    upRiskMinTradesSoft?: number;
    upRiskWrSoft?: number;
    upRiskPnlSoft?: number;
    upRiskSoftExtraDelta?: number;
    upRiskSoftBetScale?: number;
    upRiskReleaseChecks?: number;
    upRiskReleaseWr?: number;
    upRiskReleasePnl?: number;
    upRiskCheckSeconds?: number;
    upRiskStatsMode?: 'simulation' | 'live' | 'route';
    upRiskBySymbol?: Record<string, Partial<{
        upRiskLookbackHoursMain: number;
        upRiskLookbackHoursCalib: number;
        upRiskMinTradesSoft: number;
        upRiskWrSoft: number;
        upRiskPnlSoft: number;
        upRiskSoftExtraDelta: number;
        upRiskSoftBetScale: number;
        upRiskReleaseChecks: number;
        upRiskReleaseWr: number;
        upRiskReleasePnl: number;
    }>>;
    shockRiskEngine?: 'v1' | 'v2';
    shockRiskMode?: 'off' | 'shadow' | 'enforce';
    shockRiskModeBySymbol?: Record<string, 'off' | 'shadow' | 'enforce'>;
    mainlineExtremeMode?: 'off' | 'shadow' | 'enforce';
    mainlineExtremeModeBySymbol?: Record<string, 'off' | 'shadow' | 'enforce'>;
    shockRiskStatsMode?: 'simulation' | 'live' | 'route';
    shockRiskWindowMinutes?: number;
    shockRiskMinTrades?: number;
    shockRiskWrPost?: number;
    shockRiskPnlPerTrade?: number;
    shockRiskVolPercentile?: number;
    shockRiskSevereVolRatio?: number;
    shockRiskHoldHours?: number;
    shockRiskSevereHoldHours?: number;
    shockRiskReleaseChecks?: number;
    shockRiskReleaseWr?: number;
    shockRiskReleasePnl?: number;
    shockRiskCheckSeconds?: number;
    shockRiskBySymbol?: Record<string, Partial<{
        shockRiskWindowMinutes: number;
        shockRiskMinTrades: number;
        shockRiskWrPost: number;
        shockRiskPnlPerTrade: number;
        shockRiskVolPercentile: number;
        shockRiskSevereVolRatio: number;
        shockRiskHoldHours: number;
        shockRiskSevereHoldHours: number;
        shockRiskReleaseChecks: number;
        shockRiskReleaseWr: number;
        shockRiskReleasePnl: number;
    }>>;
    comboPauseEnabled?: boolean;
    comboPauseMode?: 'off' | 'shadow' | 'enforce';
    comboPauseModeBySymbol?: Record<string, 'off' | 'shadow' | 'enforce'>;
    comboPauseModeBySymbolDirection?: Record<string, 'off' | 'shadow' | 'enforce'>;
    comboPauseStatsMode?: 'simulation' | 'live' | 'route';
    comboPauseDrawdown2h?: number;
    comboPauseHoldMinutes?: number;
    comboPauseCheckSeconds?: number;
    comboPauseBySymbol?: Record<string, Partial<{
        comboPauseDrawdown2h: number;
        comboPauseHoldMinutes: number;
    }>>;
    comboPauseBySymbolDirection?: Record<string, Partial<{
        comboPauseDrawdown2h: number;
        comboPauseHoldMinutes: number;
    }>>;
    calibrationMode?: 'off' | 'shadow' | 'enforce';
    calibrationModeBySymbolDirection?: Record<string, 'off' | 'shadow' | 'enforce'>;
    calibrationStatsMode?: 'simulation' | 'live' | 'route';
    calibrationLookbackHours?: number;
    calibrationMinSamples?: number;
    calibrationCheckSeconds?: number;
    calibrationBySymbolDirection?: Record<string, Partial<{
        calibrationLookbackHours: number;
        calibrationMinSamples: number;
        calibrationMethod: 'identity' | 'temperature' | 'sigmoid' | 'isotonic';
        calibrationTemperature: number;
        calibrationAlpha: number;
        calibrationBeta: number;
        calibrationIsotonicX: number[];
        calibrationIsotonicY: number[];
    }>>;
    expectancyGateMode?: 'off' | 'shadow' | 'enforce';
    expectancyGateModeBySymbolDirection?: Record<string, 'off' | 'shadow' | 'enforce'>;
    expectancyGateStatsMode?: 'simulation' | 'live' | 'route';
    expectancyGateLookbackHours?: number;
    expectancyGateLookbackHoursFast?: number;
    expectancyGateMinTrades?: number;
    expectancyGateMinTradesBlocked?: number;
    expectancyGateMinTradesFast?: number;
    expectancyGateWrDegraded?: number;
    expectancyGatePnlDegraded?: number;
    expectancyGateDegradedBetScale?: number;
    expectancyGateDegradedExtraDelta?: number;
    expectancyGateWrBlocked?: number;
    expectancyGatePnlBlocked?: number;
    expectancyGateWrFastBlocked?: number;
    expectancyGatePnlFastBlocked?: number;
    expectancyGateBlockedEnterChecks?: number;
    expectancyGateReleaseChecks?: number;
    expectancyGateReleaseWr?: number;
    expectancyGateReleasePnl?: number;
    expectancyGateCheckSeconds?: number;
    expectancyGateBySymbolDirection?: Record<string, Partial<{
        expectancyGateLookbackHours: number;
        expectancyGateLookbackHoursFast: number;
        expectancyGateMinTrades: number;
        expectancyGateMinTradesBlocked: number;
        expectancyGateMinTradesFast: number;
        expectancyGateWrDegraded: number;
        expectancyGatePnlDegraded: number;
        expectancyGateDegradedBetScale: number;
        expectancyGateDegradedExtraDelta: number;
        expectancyGateWrBlocked: number;
        expectancyGatePnlBlocked: number;
        expectancyGateWrFastBlocked: number;
        expectancyGatePnlFastBlocked: number;
        expectancyGateBlockedEnterChecks: number;
        expectancyGateReleaseChecks: number;
        expectancyGateReleaseWr: number;
        expectancyGateReleasePnl: number;
        expectancyGateManualAction: 'normal' | 'degraded' | 'blocked';
        expectancyGateManualDegradedBetScale: number;
        expectancyGateManualDegradedExtraDelta: number;
    }>>;
    uncertaintyGateMode?: 'off' | 'shadow' | 'enforce';
    uncertaintyGateModeBySymbolDirection?: Record<string, 'off' | 'shadow' | 'enforce'>;
    uncertaintyGateBySymbolDirection?: Record<string, Partial<{
        uncertaintyDispersionMax: number;
        uncertaintyHighVolDispersionMax: number;
        uncertaintyIntervalWidthMax: number;
        uncertaintyHighVolIntervalWidthMax: number;
        uncertaintyIntervalZScore: number;
        uncertaintyIntervalDispersionBlend: number;
        uncertaintyEffectiveSourceCountMin: number;
        uncertaintyChangePointProbMax: number;
        uncertaintyCountertrendChangePointProbMax: number;
        uncertaintyDegradedExtraDelta: number;
        uncertaintyManualAction: 'normal' | 'degraded' | 'blocked';
    }>>;
    thresholdDriftMode?: 'off' | 'shadow' | 'enforce';
    thresholdDriftModeBySymbolDirection?: Record<string, 'off' | 'shadow' | 'enforce'>;
    thresholdDriftStatsMode?: 'simulation' | 'live' | 'route';
    thresholdDriftLookbackHours?: number;
    thresholdDriftMinTrades?: number;
    thresholdDriftWrDegraded?: number;
    thresholdDriftPnlDegraded?: number;
    thresholdDriftDeltaStep?: number;
    thresholdDriftDeltaMax?: number;
    thresholdDriftReleaseChecks?: number;
    thresholdDriftReleaseWr?: number;
    thresholdDriftReleasePnl?: number;
    thresholdDriftCheckSeconds?: number;
    thresholdDriftBySymbolDirection?: Record<string, Partial<{
        thresholdDriftLookbackHours: number;
        thresholdDriftMinTrades: number;
        thresholdDriftWrDegraded: number;
        thresholdDriftPnlDegraded: number;
        thresholdDriftDeltaStep: number;
        thresholdDriftDeltaMax: number;
        thresholdDriftReleaseChecks: number;
        thresholdDriftReleaseWr: number;
        thresholdDriftReleasePnl: number;
    }>>;
    metaLabelMode?: 'off' | 'shadow' | 'enforce';
    metaLabelModeBySymbolDirection?: Record<string, 'off' | 'shadow' | 'enforce'>;
    selectorMode?: 'off' | 'shadow' | 'enforce';
    selectorModeBySymbol?: Record<string, 'off' | 'shadow' | 'enforce'>;
    selectorEligibleBySymbol?: Record<string, boolean>;
    selectorScopeBySymbol?: Record<string, 'core' | 'shadow'>;
    selectorStatsMode?: 'simulation' | 'live' | 'route';
    selectorCheckSeconds?: number;
    selectorDemoteStatuses?: string[];
    regimeMode?: 'off' | 'shadow' | 'enforce';
    regimeModeBySymbol?: Record<string, 'off' | 'shadow' | 'enforce'>;
    regimeStatsMode?: 'simulation' | 'live' | 'route';
    regimeCheckSeconds?: number;
    appliedTradingMode?: TradingModeValue;
    privateKeyEnv?: string;
    proxyWalletEnv?: string;
    polyApiKeyEnv?: string;
    polyApiSecretEnv?: string;
    polyApiPassphraseEnv?: string;
    builderApiKeyEnv?: string;
    builderApiSecretEnv?: string;
    builderApiPassphraseEnv?: string;
    symbolModeOverrides?: Record<string, 'simulation' | 'live'>;
    walletSlotBySymbol?: Record<string, string>;
}

type TradingModeValue = 'simulation' | 'live';

interface ModeRegistryEntry {
    traderName: string;
    symbol: string;
    mode: TradingModeValue;
    walletSlot?: string;
    enabled?: boolean;
}

interface ModeRegistryProfile {
    entries: ModeRegistryEntry[];
}

interface ModeRegistryFile {
    version?: number;
    effectiveFrom?: string;
    profiles?: Record<string, ModeRegistryProfile>;
}

interface WalletSlot {
    privateKeyEnv?: string;
    proxyWalletEnv?: string;
    polyApiKeyEnv?: string;
    polyApiSecretEnv?: string;
    polyApiPassphraseEnv?: string;
    builderApiKeyEnv?: string;
    builderApiSecretEnv?: string;
    builderApiPassphraseEnv?: string;
}

type WalletSlotsFile = Record<string, WalletSlot>;

interface EffectiveModeEntry {
    traderName: string;
    symbol: string;
    desiredMode: TradingModeValue;
    appliedMode: TradingModeValue;
    walletSlot?: string;
    enabled: boolean;
    reason: string;
}

interface ModeRegistryRuntimeState {
    enabled: boolean;
    validateOnly: boolean;
    failOpen: boolean;
    profile: 'default' | '70';
    registryFile: string;
    walletSlotsFile: string;
    effectiveFrom?: string;
    activeByTime: boolean;
    warnings: string[];
    errors: string[];
    entries: EffectiveModeEntry[];
    walletSlots: WalletSlotsFile;
}

interface DownBreakerStats {
    checkedAtMs: number;
    lookbackHours: number;
    trades: number;
    wins: number;
    winRate: number;
    pnl: number;
    pnlPerTrade: number;
    triggered: boolean;
    reason: string;
}

type DownRiskEngine = 'v1' | 'v2';
type DownRiskMode = 'off' | 'shadow' | 'enforce';
type DownRiskTier = 'normal' | 'soft' | 'hard';
type DownRiskStatsMode = 'simulation' | 'live' | 'route';
type DownRiskRouteMode = 'simulation' | 'live';
type DownRiskCellKey = string;
type UpRiskEngine = 'v1' | 'v2';
type UpRiskMode = 'off' | 'shadow' | 'enforce';
type DrawdownAccelerationMode = 'off' | 'shadow' | 'enforce';
type UpRiskTier = 'normal' | 'soft';
type UpRiskStatsMode = 'simulation' | 'live' | 'route';
type UpRiskRouteMode = 'simulation' | 'live';
type UpRiskCellKey = string;
type ShockRiskEngine = 'v1' | 'v2';
type ShockRiskMode = 'off' | 'shadow' | 'enforce';
type MainlineExtremeMode = 'off' | 'shadow' | 'enforce';
type ShockRiskStatsMode = 'simulation' | 'live' | 'route';
type ShockRiskRouteMode = 'simulation' | 'live';
type ShockRiskCellKey = string;
type ComboPauseMode = 'off' | 'shadow' | 'enforce';
type ComboPauseStatsMode = 'simulation' | 'live' | 'route';
type ComboPauseRouteMode = 'simulation' | 'live';
type ComboPauseCellKey = string;
type DirectionLossCooldownMode = 'off' | 'shadow' | 'enforce';
type CalibrationMode = 'off' | 'shadow' | 'enforce';
type CalibrationStatsMode = 'simulation' | 'live' | 'route';
type CalibrationRouteMode = 'simulation' | 'live';
type CalibrationMethod = 'identity' | 'temperature' | 'sigmoid' | 'isotonic';
type CalibrationCellKey = string;
type ExpectancyGateMode = 'off' | 'shadow' | 'enforce';
type ExpectancyGateStatsMode = 'simulation' | 'live' | 'route';
type ExpectancyGateRouteMode = 'simulation' | 'live';
type ExpectancyGateStatus = 'normal' | 'degraded' | 'blocked';
type ExpectancyGateCellKey = string;
type ThresholdDriftMode = 'off' | 'shadow' | 'enforce';
type ThresholdDriftStatsMode = 'simulation' | 'live' | 'route';
type ThresholdDriftRouteMode = 'simulation' | 'live';
type ThresholdDriftStatus = 'normal' | 'drifted';
type ThresholdDriftCellKey = string;
type MetaLabelMode = 'off' | 'shadow' | 'enforce';
type MetaLabelCellKey = string;
type SelectorMode = 'off' | 'shadow' | 'enforce';
type SelectorStatsMode = 'simulation' | 'live' | 'route';
type SelectorRouteMode = 'simulation' | 'live';
type SelectorCellKey = string;
type RegimeMode = 'off' | 'shadow' | 'enforce';
type RegimeStatsMode = 'simulation' | 'live' | 'route';
type RegimeRouteMode = 'simulation' | 'live';
type ModeBySymbol<T extends string> = Record<string, T> | undefined;

interface DownRiskWindowStats {
    lookbackHours: number;
    trades: number;
    wins: number;
    losses: number;
    winRate: number;
    wrPost: number;
    pnl: number;
    pnlPerTrade: number;
}

interface DownRiskCellState {
    key: DownRiskCellKey;
    profile: 'default' | '70';
    traderName: string;
    symbol: string;
    mode: DownRiskRouteMode;
    tier: DownRiskTier;
    hardUntilTs: number;
    releasePassStreak: number;
    checkedAtMs: number;
    lastAction: string;
    reasonCode: string;
    triggerLane: 'main' | 'fast' | 'cross_layer' | 'normal';
    enteredBy: 'down' | 'shock' | 'combo' | 'expect' | 'init' | 'normal';
    lastTransitionAtMs: number;
    blocked: boolean;
    downScale: number;
    extraDelta: number;
    releaseWindowMainPass: boolean;
    releaseWindowConfirmPass: boolean;
    sourceMode: DownRiskRouteMode;
    main: DownRiskWindowStats;
    fast: DownRiskWindowStats;
    calib: DownRiskWindowStats;
    releaseConfirm: DownRiskWindowStats;
}

interface UpRiskWindowStats {
    lookbackHours: number;
    trades: number;
    wins: number;
    losses: number;
    winRate: number;
    wrPost: number;
    pnl: number;
    pnlPerTrade: number;
}

interface UpRiskCellState {
    key: UpRiskCellKey;
    profile: 'default' | '70';
    traderName: string;
    symbol: string;
    mode: UpRiskRouteMode;
    tier: UpRiskTier;
    releasePassStreak: number;
    checkedAtMs: number;
    lastAction: string;
    reasonCode: string;
    blocked: boolean;
    upScale: number;
    extraDelta: number;
    sourceMode: UpRiskRouteMode;
    main: UpRiskWindowStats;
    calib: UpRiskWindowStats;
}

interface ShockRiskWindowStats {
    lookbackMinutes: number;
    trades: number;
    wins: number;
    losses: number;
    winRate: number;
    wrPost: number;
    pnl: number;
    pnlPerTrade: number;
    avgAbsPnl: number;
}

interface ShockRiskCellState {
    key: ShockRiskCellKey;
    profile: 'default' | '70';
    traderName: string;
    symbol: string;
    direction: 'UP' | 'DOWN';
    mode: ShockRiskRouteMode;
    active: boolean;
    activeUntilTs: number;
    releasePassStreak: number;
    checkedAtMs: number;
    lastAction: string;
    reasonCode: string;
    triggerLane: 'normal' | 'severe' | 'release';
    lastTransitionAtMs: number;
    blocked: boolean;
    sourceMode: ShockRiskRouteMode;
    vol1h: number;
    vol4h: number;
    vol24h: number;
    volRatio: number;
    volPctl: number;
    main: ShockRiskWindowStats;
}

interface ComboPauseCellState {
    key: ComboPauseCellKey;
    profile: 'default' | '70';
    traderName: string;
    symbol: string;
    direction: 'UP' | 'DOWN' | 'BOTH';
    mode: ComboPauseRouteMode;
    active: boolean;
    activeUntilTs: number;
    checkedAtMs: number;
    lastAction: string;
    reasonCode: string;
    triggerLane: 'symbol' | 'symbol_direction' | 'release';
    lastTransitionAtMs: number;
    blocked: boolean;
    sourceMode: ComboPauseRouteMode;
    drawdown2h: number;
    pnl2h: number;
    baseCapital: number;
    sampleCount: number;
}

interface CalibrationWindowStats {
    lookbackHours: number;
    sampleCount: number;
    wins: number;
    losses: number;
    winRate: number;
    avgConfidence: number;
    brier: number;
    ece: number;
    calibrationGap: number;
}

interface CalibrationCellState {
    key: CalibrationCellKey;
    profile: 'default' | '70';
    traderName: string;
    symbol: string;
    direction: 'UP' | 'DOWN';
    mode: CalibrationRouteMode;
    sourceMode: CalibrationRouteMode;
    method: CalibrationMethod;
    checkedAtMs: number;
    lastAction: string;
    reasonCode: string;
    active: boolean;
    minSamples: number;
    lastConfidence: number;
    calibratedConfidence: number;
    stats: CalibrationWindowStats;
}

interface ExpectancyGateWindowStats {
    lookbackHours: number;
    trades: number;
    wins: number;
    losses: number;
    winRate: number;
    wrPost: number;
    pnl: number;
    pnlPerTrade: number;
    netPnL: number;
}

interface ExpectancyGateCellState {
    key: ExpectancyGateCellKey;
    profile: 'default' | '70';
    traderName: string;
    symbol: string;
    direction: 'UP' | 'DOWN';
    mode: ExpectancyGateRouteMode;
    sourceMode: ExpectancyGateRouteMode;
    checkedAtMs: number;
    lastAction: string;
    reasonCode: string;
    triggerLane: 'main' | 'fast' | 'normal' | 'manual';
    lastTransitionAtMs: number;
    status: ExpectancyGateStatus;
    blocked: boolean;
    degradedBetScale: number;
    degradedExtraDelta: number;
    activeUntilTs: number;
    blockedEnterPassStreak: number;
    releasePassStreak: number;
    stats: ExpectancyGateWindowStats;
}

interface ThresholdDriftWindowStats {
    lookbackHours: number;
    trades: number;
    wins: number;
    losses: number;
    winRate: number;
    wrPost: number;
    pnl: number;
    pnlPerTrade: number;
}

interface ThresholdDriftCellState {
    key: ThresholdDriftCellKey;
    profile: 'default' | '70';
    traderName: string;
    symbol: string;
    direction: 'UP' | 'DOWN';
    mode: ThresholdDriftRouteMode;
    sourceMode: ThresholdDriftRouteMode;
    checkedAtMs: number;
    lastAction: string;
    reasonCode: string;
    status: ThresholdDriftStatus;
    active: boolean;
    thresholdDelta: number;
    releasePassStreak: number;
    stats: ThresholdDriftWindowStats;
}

interface MetaLabelCellState {
    key: MetaLabelCellKey;
    profile: 'default' | '70';
    traderName: string;
    symbol: string;
    direction: 'UP' | 'DOWN';
    mode: MetaLabelMode;
    checkedAtMs: number;
    lastAction: string;
    reasonCode: string;
    takeTrade: boolean;
    metaConfidence: number | null;
    evaluated: number;
    passed: number;
    blocked: number;
}

interface SelectorCellState {
    key: SelectorCellKey;
    profile: 'default' | '70';
    traderName: string;
    symbol: string;
    mode: SelectorRouteMode;
    sourceMode: SelectorRouteMode;
    checkedAtMs: number;
    lastAction: string;
    reasonCode: string;
    status: string;
    eligible: boolean;
    scope: 'core' | 'shadow' | 'unknown';
}

interface RegimeGateSymbolState {
    symbol: string;
    profile: 'default' | '70';
    mode: RegimeMode;
    sourceMode: RegimeRouteMode;
    effectiveMode: RegimeMode;
    active: boolean;
    directionPolicy: 'BOTH' | 'UP' | 'DOWN' | 'NONE';
    policyMode: 'one_sided' | 'neutral';
    policyConfidence: number;
    policyReason: string;
    policyDisagreementFlags: string[];
    thresholdDelta: number;
    regimeId: string;
    reasonCode: string;
    checkedAtMs: number;
    changePointProb?: number;
    changePointDirection?: 'UP' | 'DOWN' | 'NONE';
    changePointScoreUp?: number;
    changePointScoreDown?: number;
}

interface MainlineExtremeRuntimeState {
    symbol: string;
    profile: 'default' | '70';
    active: boolean;
    laneState: string;
    regimeId: string;
    directionPolicy: 'BOTH' | 'UP' | 'DOWN' | 'NONE';
    policyMode: 'one_sided' | 'neutral';
    policyConfidence: number;
    thresholdExtraDelta: number;
    betScale: number;
    reasonCode: string;
    checkedAtMs: number;
}

interface DirectionLossCooldownState {
    key: string;
    symbol: string;
    direction: 'UP' | 'DOWN';
    mode?: DirectionLossCooldownMode;
    blocked: boolean;
    evidenceFresh: boolean;
    softScaled: boolean;
    softOverlapWithExtreme: boolean;
    softScale: number;
    softSeverity: 'none' | 'base' | 'mid' | 'strong';
    streakLosses: number;
    recentTrades: number;
    recentPnl: number;
    recentHighConfidenceLosses: number;
    recentAvgLossConfidence: number;
    untilTs: number;
    reasonCode: string;
    softReasonCode: string;
    checkedAtMs: number;
}

interface DrawdownAccelerationState {
    symbol: string;
    mode?: DrawdownAccelerationMode;
    blocked: boolean;
    evidenceFresh: boolean;
    softScaled: boolean;
    softScale: number;
    recentTrades: number;
    recentLosses: number;
    recentPnl: number;
    untilTs: number;
    reasonCode: string;
    softReasonCode: string;
    checkedAtMs: number;
}

interface RuntimeThresholdDeltaEntry {
    layer: string;
    delta: number;
    active: boolean;
    reasonCode?: string;
}

interface RuntimeScaleEntry {
    layer: string;
    scale: number;
    active: boolean;
    reasonCode?: string;
}

interface MarketSelectionEvidence {
    qualityMode: GuardMode;
    expectancyStatus: 'normal' | 'degraded' | 'blocked';
    expectancySampleCount: number;
    expectancyPnlPerTrade: number;
    uncertaintyStatus: 'normal' | 'degraded' | 'blocked';
    uncertaintyDispersion: number | null;
    uncertaintyEffectiveSourceCount: number | null;
    uncertaintyIntervalWidth: number | null;
    regimeDirectionPolicy: 'BOTH' | 'UP' | 'DOWN' | 'NONE';
    regimeDisagreementCount: number;
    changePointProb: number | null;
    confidence: number | null;
    effectiveThreshold: number | null;
    reasons: string[];
}

interface MarketSelectionDecision {
    applicable: boolean;
    action: 'allow' | 'abstain';
    reasonCode: string;
    marketQualityScore: number;
    evidence: MarketSelectionEvidence;
}

interface ExecutionDecisionSnapshot {
    key: string;
    profile: string;
    traderName: string;
    symbol: string;
    direction: 'UP' | 'DOWN';
    decisionStatus: 'passed' | 'blocked' | 'abstained';
    primaryBlockReason: string | null;
    runtimePrimaryBlockReason: string | null;
    thresholdBlockReason: string | null;
    selectorMode: 'off' | 'shadow' | 'enforce';
    selectorEligible: boolean;
    selectorScope: 'core' | 'shadow' | 'unknown';
    thresholdDeltaStack: RuntimeThresholdDeltaEntry[];
    scaleStack: RuntimeScaleEntry[];
    finalRuntimeScale: number;
    gatesPassed: string[];
    effectiveThreshold: number;
    confidence: number | null;
    rawConfidence: number | null;
    calibratedConfidence: number | null;
    shouldTrade: boolean | null;
    selectionAction: 'allow' | 'abstain';
    selectionApplicable: boolean;
    selectionReasonCode: string;
    selectionRouteApplied: boolean;
    selectionRouteProfileScope: 'default_only' | 'skipped_non_default';
    selectionRouteAction: 'allow' | 'abstain' | 'skip';
    selectionRouteReasonCode: string;
    marketQualityScore: number | null;
    selectionEvidence: MarketSelectionEvidence | null;
    uncertaintyGateStatus: 'normal' | 'degraded' | 'blocked';
    uncertaintyGateReasonCode: string;
    runtime_primary_block_reason: string | null;
    threshold_block_reason: string | null;
    selection_route_applied: boolean;
    selection_route_profile_scope: 'default_only' | 'skipped_non_default';
    selection_route_action: 'allow' | 'abstain' | 'skip';
    selection_route_reason_code: string;
    uncertainty_gate_status: 'normal' | 'degraded' | 'blocked';
    uncertainty_gate_reason_code: string;
    finalDecision?: 'passed' | 'blocked' | 'abstained';
    noTradeReason?: string | null;
    currentCapital?: number | null;
    cooldownScope?: 'symbol' | 'symbol_direction' | null;
    cooldownKey?: string | null;
    cooldownScopeLabel?: 'total_chain_by_symbol' | 'symbol_direction' | null;
    cooldownRemainingBars?: number | null;
    formalConsecutiveLosses?: number | null;
    provisionalConsecutiveLosses?: number | null;
    effectiveConsecutiveLosses?: number | null;
    cooldownPendingResolutionBlocked?: boolean;
    cooldownPendingResolutionMarketSlug?: string | null;
    cooldownPendingResolutionConsecutiveBeforeUnknown?: number | null;
    cooldownPersistedWindowBlocked?: boolean;
    cooldownTriggerMarketSlug?: string | null;
    cooldownTriggerStage?: 'formal' | 'provisional' | null;
    cooldownPersistedRemainingBarsBefore?: number | null;
    cooldownPersistedRemainingBarsAfter?: number | null;
    cooldownPersistedError?: string | null;
    slot02Aux1hEnabled?: boolean;
    slot02Aux1hMode?: string | null;
    slot02Aux1hDirection?: 'UP' | 'DOWN' | 'NEUTRAL' | null;
    slot02Aux1hConfidence?: number | null;
    slot02Aux1hStrength?: number | null;
    slot02Aux1hFreshnessSec?: number | null;
    slot02Aux1hBlocked?: boolean;
    slot02Aux1hReason?: string | null;
    slot02Aux1hModelVersion?: string | null;
    slot02Aux1hUpStrengthThreshold?: number | null;
    slot02Aux1hDownStrengthThreshold?: number | null;
    slot02Aux1hMainConfidenceThreshold?: number | null;
    slot02Aux1hMainConfidenceBelowThreshold?: boolean | null;
    slot02Aux1hActionCode?: number | null;
    slot02Aux1hSoftCapPct?: number | null;
    slot02Aux1hWeakSupportActive?: boolean | null;
    slot02Aux1hMainConfidenceBucket?: number | null;
    slot02Aux1hVolValue?: number | null;
    slot02Aux1hVolBucket?: number | null;
    checkedAtMs: number;
    timestamp?: string;
}

interface TraderInstance {
    config: TraderConfig;
    env: PredictionEnvConfig;
    executor: PredictionExecutor;
    executedTargetTss: Set<number>;
    executedPhaseKeys: Map<string, number>;
    lastPredictionTimestampByTs: Map<number, string>;
    predictionFilePath: string;
    predictionFallbackFilePath?: string;
    lastPredictionPathDecision?: string;
    consecutiveErrors: number;
    paused: boolean;
    downBreakerUntilTs: number;
    downBreakerLastCheckMs: number;
    downBreakerLastFileMtimeMs: number;
    downBreakerStats: DownBreakerStats;
    downRiskCells: Map<DownRiskCellKey, DownRiskCellState>;
    downRiskLastCheckMsByMode: Record<DownRiskRouteMode, number>;
    downRiskLastFileMtimeMsByMode: Record<DownRiskRouteMode, number>;
    downRiskRuntimeScaleBySymbol: Record<string, number>;
    downRiskRuntimeBlockedBySymbol: Record<string, boolean>;
    downRiskRuntimeExtraDeltaBySymbol: Record<string, number>;
    upRiskCells: Map<UpRiskCellKey, UpRiskCellState>;
    upRiskLastCheckMsByMode: Record<UpRiskRouteMode, number>;
    upRiskLastFileMtimeMsByMode: Record<UpRiskRouteMode, number>;
    upRiskRuntimeScaleBySymbol: Record<string, number>;
    upRiskRuntimeBlockedBySymbol: Record<string, boolean>;
    upRiskRuntimeExtraDeltaBySymbol: Record<string, number>;
    shockRiskCells: Map<ShockRiskCellKey, ShockRiskCellState>;
    shockRiskLastCheckMsByMode: Record<ShockRiskRouteMode, number>;
    shockRiskLastFileMtimeMsByMode: Record<ShockRiskRouteMode, number>;
    shockRiskRuntimeBlockedBySymbolDir: Record<string, boolean>;
    comboPauseCells: Map<ComboPauseCellKey, ComboPauseCellState>;
    comboPauseLastCheckMsByMode: Record<ComboPauseRouteMode, number>;
    comboPauseLastFileMtimeMsByMode: Record<ComboPauseRouteMode, number>;
    comboPauseRuntimeBlockedBySymbol: Record<string, boolean>;
    comboPauseRuntimeBlockedBySymbolDir: Record<string, boolean>;
    calibrationCells: Map<CalibrationCellKey, CalibrationCellState>;
    calibrationLastCheckMsByMode: Record<CalibrationRouteMode, number>;
    calibrationLastFileMtimeMsByMode: Record<CalibrationRouteMode, number>;
    expectancyGateCells: Map<ExpectancyGateCellKey, ExpectancyGateCellState>;
    expectancyGateLastCheckMsByMode: Record<ExpectancyGateRouteMode, number>;
    expectancyGateLastFileMtimeMsByMode: Record<ExpectancyGateRouteMode, number>;
    expectancyGateRuntimeBlockedBySymbolDir: Record<string, boolean>;
    expectancyGateRuntimeScaleBySymbolDir: Record<string, number>;
    expectancyGateRuntimeExtraDeltaBySymbolDir: Record<string, number>;
    thresholdDriftCells: Map<ThresholdDriftCellKey, ThresholdDriftCellState>;
    thresholdDriftLastCheckMsByMode: Record<ThresholdDriftRouteMode, number>;
    thresholdDriftLastFileMtimeMsByMode: Record<ThresholdDriftRouteMode, number>;
    thresholdDriftRuntimeExtraDeltaBySymbolDir: Record<string, number>;
    metaLabelCells: Map<MetaLabelCellKey, MetaLabelCellState>;
    selectorCells: Map<SelectorCellKey, SelectorCellState>;
    selectorLastCheckMsByMode: Record<SelectorRouteMode, number>;
    selectorRuntimeEligibleBySymbol: Record<string, boolean>;
    regimeLastCheckMsByMode: Record<RegimeRouteMode, number>;
    regimeLastFileMtimeMsByMode: Record<RegimeRouteMode, number>;
    regimeRuntimeBySymbol: Record<string, RegimeGateSymbolState>;
    mainlineExtremeRuntimeBySymbol: Record<string, MainlineExtremeRuntimeState>;
    mainlineExtremeRuntimeScaleBySymbol: Record<string, number>;
    mainlineExtremeRuntimeExtraDeltaBySymbolDir: Record<string, number>;
    directionLossCooldownBySymbolDir: Record<string, DirectionLossCooldownState>;
    directionLossCooldownBlockedBySymbolDir: Record<string, boolean>;
    directionLossCooldownSoftScaleBySymbolDir: Record<string, number>;
    directionLossCooldownLastTriggerKeyBySymbolDir: Record<string, string>;
    drawdownAccelerationBySymbol: Record<string, DrawdownAccelerationState>;
    drawdownAccelerationBlockedBySymbol: Record<string, boolean>;
    drawdownAccelerationSoftScaleBySymbol: Record<string, number>;
    drawdownAccelerationLastTriggerKeyBySymbol: Record<string, string>;
    guardEvidence: GuardEvidenceStats;
    executionDecisionBySymbolDir: Record<string, ExecutionDecisionSnapshot>;
    blockedOpportunityLogKeys: Set<string>;
}

type GuardMode = 'ok' | 'soft_degraded' | 'hard_degraded' | 'halt';
type DataQualityTriggerClass = 'ok' | 'api_degraded' | 'stale_data' | 'mixed';

interface DataQualityState {
    checkedAtMs: number;
    mode: GuardMode;
    severity: GuardMode;
    triggerClass: DataQualityTriggerClass;
    staleSources: string[];
    criticalStale: string[];
    confidencePenalty: number;
    betScale: number;
    apiErrorRate: number;
    apiErrorSources: string[];
    keySourceErrors: string[];
    nonKeySourceErrors: string[];
    secondarySourceErrors: string[];
    observationCounts: Record<string, number>;
    consecutiveBadWindows: Record<string, number>;
    consecutiveGoodWindows: Record<string, number>;
    reason: string;
}

// ============================================================
// 常量
// ============================================================

const SCAN_INTERVAL_MS = 1000;
const SETTLEMENT_CHECK_INTERVAL = 60 * 1000;
const REPORT_INTERVAL_MS = 60 * 60 * 1000;
const MONITOR_INTERVAL_MS = 1000;
const LIQUIDITY_MONITOR_INTERVAL_MS = 10 * 1000;
const CONFIG_RELOAD_CHECK_INTERVAL_MS = 15 * 1000;

const EXECUTED_PERIODS_FILE = 'executed_target_ts.json';
const EXECUTED_PERIODS_RETENTION_SEC = 7 * 24 * 3600;
const MAX_CONSECUTIVE_ERRORS = 10;
const PERIOD_SECONDS = 900;
const RUNTIME_GUARD_STATUS_DIR = path.join(process.cwd(), 'logs');
const LEGACY_RUNTIME_GUARD_STATUS_FILE = path.join(RUNTIME_GUARD_STATUS_DIR, 'runtime_trade_guards.json');
const WRITE_LEGACY_RUNTIME_GUARD = String(process.env.WRITE_LEGACY_RUNTIME_GUARD ?? '').toLowerCase() === 'true';
const MAINLINE_EXTREME_STATE_FILE = path.join(process.cwd(), '..', 'reports', 'coremain_extreme_regime_state_latest.json');
const MAINLINE_EXTREME_REFRESH_MS = Number(process.env.MAINLINE_EXTREME_REFRESH_MS ?? '30000');
const MAINLINE_EXTREME_BASE_BET_SCALE = Number(process.env.MAINLINE_EXTREME_BASE_BET_SCALE ?? '0.88');
const MAINLINE_EXTREME_RALLY_BET_SCALE = Number(process.env.MAINLINE_EXTREME_RALLY_BET_SCALE ?? '0.82');
const MAINLINE_EXTREME_HARD_BET_SCALE = Number(process.env.MAINLINE_EXTREME_HARD_BET_SCALE ?? '0.68');
const MAINLINE_EXTREME_RALLY_COUNTERTREND_DELTA = Number(process.env.MAINLINE_EXTREME_RALLY_COUNTERTREND_DELTA ?? '0.03');
const MAINLINE_EXTREME_HARD_COUNTERTREND_DELTA = Number(process.env.MAINLINE_EXTREME_HARD_COUNTERTREND_DELTA ?? '0.06');
const MAINLINE_DIRECTION_LOSS_COOLDOWN_LOOKBACK_TRADES = Number(process.env.MAINLINE_DIRECTION_LOSS_COOLDOWN_LOOKBACK_TRADES ?? '4');
const MAINLINE_DIRECTION_LOSS_COOLDOWN_TRIGGER = Number(process.env.MAINLINE_DIRECTION_LOSS_COOLDOWN_TRIGGER ?? '2');
const MAINLINE_DIRECTION_LOSS_COOLDOWN_MINUTES = Number(process.env.MAINLINE_DIRECTION_LOSS_COOLDOWN_MINUTES ?? '60');
const MAINLINE_DIRECTION_LOSS_COOLDOWN_MAX_AGE_MINUTES = Number(process.env.MAINLINE_DIRECTION_LOSS_COOLDOWN_MAX_AGE_MINUTES ?? '240');
const MAINLINE_DIRECTION_LOSS_SOFT_MIN_HIGH_CONF_LOSSES = Number(process.env.MAINLINE_DIRECTION_LOSS_SOFT_MIN_HIGH_CONF_LOSSES ?? '2');
const MAINLINE_DIRECTION_LOSS_SOFT_MIN_CONFIDENCE = Number(process.env.MAINLINE_DIRECTION_LOSS_SOFT_MIN_CONFIDENCE ?? '0.60');
const MAINLINE_DIRECTION_LOSS_SOFT_BET_SCALE = Number(process.env.MAINLINE_DIRECTION_LOSS_SOFT_BET_SCALE ?? '0.90');
const MAINLINE_DIRECTION_LOSS_SOFT_MID_MIN_STREAK = Number(process.env.MAINLINE_DIRECTION_LOSS_SOFT_MID_MIN_STREAK ?? '2');
const MAINLINE_DIRECTION_LOSS_SOFT_MID_MIN_HIGH_CONF_LOSSES = Number(process.env.MAINLINE_DIRECTION_LOSS_SOFT_MID_MIN_HIGH_CONF_LOSSES ?? '3');
const MAINLINE_DIRECTION_LOSS_SOFT_MID_MIN_AVG_CONFIDENCE = Number(process.env.MAINLINE_DIRECTION_LOSS_SOFT_MID_MIN_AVG_CONFIDENCE ?? '0.70');
const MAINLINE_DIRECTION_LOSS_SOFT_MID_BET_SCALE = Number(process.env.MAINLINE_DIRECTION_LOSS_SOFT_MID_BET_SCALE ?? '0.86');
const MAINLINE_DIRECTION_LOSS_SOFT_STRONG_MIN_STREAK = Number(process.env.MAINLINE_DIRECTION_LOSS_SOFT_STRONG_MIN_STREAK ?? '3');
const MAINLINE_DIRECTION_LOSS_SOFT_STRONG_MIN_HIGH_CONF_LOSSES = Number(process.env.MAINLINE_DIRECTION_LOSS_SOFT_STRONG_MIN_HIGH_CONF_LOSSES ?? '4');
const MAINLINE_DIRECTION_LOSS_SOFT_STRONG_MIN_AVG_CONFIDENCE = Number(process.env.MAINLINE_DIRECTION_LOSS_SOFT_STRONG_MIN_AVG_CONFIDENCE ?? '0.68');
const MAINLINE_DIRECTION_LOSS_SOFT_STRONG_BET_SCALE = Number(process.env.MAINLINE_DIRECTION_LOSS_SOFT_STRONG_BET_SCALE ?? '0.82');
const MAINLINE_DRAWDOWN_ACCEL_LOOKBACK_TRADES = Number(process.env.MAINLINE_DRAWDOWN_ACCEL_LOOKBACK_TRADES ?? '6');
const MAINLINE_DRAWDOWN_ACCEL_MIN_TRADES = Number(process.env.MAINLINE_DRAWDOWN_ACCEL_MIN_TRADES ?? '4');
const MAINLINE_DRAWDOWN_ACCEL_PCT = Number(process.env.MAINLINE_DRAWDOWN_ACCEL_PCT ?? '0.18');
const MAINLINE_DRAWDOWN_ACCEL_MINUTES = Number(process.env.MAINLINE_DRAWDOWN_ACCEL_MINUTES ?? '75');
const MAINLINE_DRAWDOWN_ACCEL_MAX_AGE_MINUTES = Number(process.env.MAINLINE_DRAWDOWN_ACCEL_MAX_AGE_MINUTES ?? '360');
const MAINLINE_DRAWDOWN_ACCEL_SOFT_MIN_RECENT_LOSSES = Number(process.env.MAINLINE_DRAWDOWN_ACCEL_SOFT_MIN_RECENT_LOSSES ?? '2');
const MAINLINE_DRAWDOWN_ACCEL_SOFT_EARLY_MIN_TRADES = Number(process.env.MAINLINE_DRAWDOWN_ACCEL_SOFT_EARLY_MIN_TRADES ?? '3');
const MAINLINE_DRAWDOWN_ACCEL_SOFT_PNL_FACTOR = Number(process.env.MAINLINE_DRAWDOWN_ACCEL_SOFT_PNL_FACTOR ?? '0.60');
const MAINLINE_DRAWDOWN_ACCEL_SOFT_BET_SCALE = Number(process.env.MAINLINE_DRAWDOWN_ACCEL_SOFT_BET_SCALE ?? '0.92');

// 方向阈值默认策略：在保持 UP 阈值不变时，对 DOWN 额外加严
const DEFAULT_DOWN_THRESHOLD_DELTA = Number(process.env.DOWN_THRESHOLD_DELTA_DEFAULT ?? '0.04');
const GROUP_DOWN_DELTA_MAP: Record<string, number> = {
    v5_exp10: 0.09,
    v5_exp11: 0.09,
    v5_exp13: 0.06,
    v5_exp14: 0.04,
    v5_exp15: 0.06,
    v5_exp16: 0.02,
    v5_exp17: 0.05,
    ensemble: 0.08,
    gru_all: 0.09,
};

// 数据质量门控（P1：外部源异常时降级）
const DATA_QUALITY_FILE = path.join(process.cwd(), '..', 'data', 'data_ready.json');
const DATA_QUALITY_REFRESH_MS = Number(process.env.DATA_QUALITY_REFRESH_MS ?? '30000');
const DATA_QUALITY_MAX_AGE_SEC = Number(process.env.DATA_QUALITY_MAX_AGE_SEC ?? '1200');
const DATA_QUALITY_DEGRADE_STALE_COUNT = Number(process.env.DATA_QUALITY_DEGRADE_STALE_COUNT ?? '1');
const DATA_QUALITY_HALT_STALE_COUNT = Number(process.env.DATA_QUALITY_HALT_STALE_COUNT ?? '3');
const DATA_QUALITY_CONF_PENALTY_PER_STALE = Number(process.env.DATA_QUALITY_CONF_PENALTY_PER_STALE ?? '0.01');
const DATA_QUALITY_CONF_PENALTY_MAX = Number(process.env.DATA_QUALITY_CONF_PENALTY_MAX ?? '0.04');
const DATA_QUALITY_API_ERROR_DEGRADE = Number(process.env.DATA_QUALITY_API_ERROR_DEGRADE ?? '0.12');
const DATA_QUALITY_API_ERROR_HALT = Number(process.env.DATA_QUALITY_API_ERROR_HALT ?? '0.30');
const DATA_QUALITY_API_MIN_OBS = Number(process.env.DATA_QUALITY_API_MIN_OBS ?? '30');
const DATA_QUALITY_API_MIN_OBS_SOFT = Number(process.env.DATA_QUALITY_API_MIN_OBS_SOFT ?? '20');
const DATA_QUALITY_SOFT_SOURCE_BAD_WINDOWS = Number(process.env.DATA_QUALITY_SOFT_SOURCE_BAD_WINDOWS ?? '2');
const DATA_QUALITY_HARD_SOURCE_BAD_WINDOWS = Number(process.env.DATA_QUALITY_HARD_SOURCE_BAD_WINDOWS ?? '4');
const DATA_QUALITY_SOURCE_GOOD_WINDOWS_TO_RECOVER = Number(process.env.DATA_QUALITY_SOURCE_GOOD_WINDOWS_TO_RECOVER ?? '3');
const DATA_QUALITY_KEY_API_ERROR_SOFT = Number(process.env.DATA_QUALITY_KEY_API_ERROR_SOFT ?? '0.04');
const DATA_QUALITY_KEY_API_ERROR_HARD = Number(process.env.DATA_QUALITY_KEY_API_ERROR_HARD ?? '0.08');
const DATA_QUALITY_SECONDARY_API_ERROR_SOFT = Number(process.env.DATA_QUALITY_SECONDARY_API_ERROR_SOFT ?? '0.05');
const DATA_QUALITY_SECONDARY_API_ERROR_HARD = Number(process.env.DATA_QUALITY_SECONDARY_API_ERROR_HARD ?? '0.12');
const DATA_QUALITY_NON_KEY_API_ERROR_SOFT = Number(process.env.DATA_QUALITY_NON_KEY_API_ERROR_SOFT ?? '0.08');
const DATA_QUALITY_NON_KEY_API_ERROR_HARD = Number(process.env.DATA_QUALITY_NON_KEY_API_ERROR_HARD ?? '0.20');
const DATA_QUALITY_BET_SCALE_SOFT = Number(process.env.DATA_QUALITY_BET_SCALE_SOFT ?? '0.85');
const DATA_QUALITY_BET_SCALE_DEGRADED = Number(process.env.DATA_QUALITY_BET_SCALE_DEGRADED ?? '0.60');
const DATA_QUALITY_BET_SCALE_HALT = Number(process.env.DATA_QUALITY_BET_SCALE_HALT ?? '0.00');
const DATA_QUALITY_RECOVERY_BARS = Number(process.env.DATA_QUALITY_RECOVERY_BARS ?? '6');
const DATA_QUALITY_RECOVERY_STABLE_CHECKS = Number(process.env.DATA_QUALITY_RECOVERY_STABLE_CHECKS ?? '2');
const DATA_QUALITY_MAX_AGE_BY_SOURCE_SEC: Record<string, number> = {
    open_interest: Number(process.env.DATA_QUALITY_MAX_AGE_OPEN_INTEREST_SEC ?? '600'),
    long_short_ratio: Number(process.env.DATA_QUALITY_MAX_AGE_LONG_SHORT_RATIO_SEC ?? '600'),
    polymarket_prob: Number(process.env.DATA_QUALITY_MAX_AGE_PM_PROB_SEC ?? '1800'),
    polymarket_prob_target: Number(process.env.DATA_QUALITY_MAX_AGE_PM_TARGET_SEC ?? '1800'),
    funding_rate: Number(process.env.DATA_QUALITY_MAX_AGE_FUNDING_RATE_SEC ?? '900'),
    ob_realtime: Number(process.env.DATA_QUALITY_MAX_AGE_OB_REALTIME_SEC ?? '900'),
};
const DATA_QUALITY_CRITICAL_SOURCES = new Set<string>([
    'polymarket_prob_target',
    'funding_rate',
]);
const DATA_QUALITY_MTIME_FALLBACK_FILES: Record<string, string[]> = {
    polymarket_prob: [
        path.join(process.cwd(), '..', 'data', 'sentiment', 'polymarket_prob_btc_usdt.parquet'),
        path.join(process.cwd(), '..', 'data', 'sentiment', 'polymarket_prob_eth_usdt.parquet'),
    ],
    polymarket_prob_target: [
        path.join(process.cwd(), '..', 'data', 'sentiment', 'polymarket_prob_target_btc_usdt.parquet'),
        path.join(process.cwd(), '..', 'data', 'sentiment', 'polymarket_prob_target_eth_usdt.parquet'),
    ],
};
const DOWN_BREAKER_LOOKBACK_HOURS_DEFAULT = Number(process.env.DOWN_BREAKER_LOOKBACK_HOURS ?? '8');
const DOWN_BREAKER_MIN_TRADES_DEFAULT = Number(process.env.DOWN_BREAKER_MIN_TRADES ?? '30');
const DOWN_BREAKER_MIN_WINRATE_DEFAULT = Number(process.env.DOWN_BREAKER_MIN_WINRATE ?? '0.40');
const DOWN_BREAKER_MIN_PNL_PER_TRADE_DEFAULT = Number(process.env.DOWN_BREAKER_MIN_PNL_PER_TRADE ?? '-0.20');
const DOWN_BREAKER_HOLD_BARS_DEFAULT = Number(process.env.DOWN_BREAKER_HOLD_BARS ?? '8');
const DOWN_BREAKER_CHECK_SECONDS_DEFAULT = Number(process.env.DOWN_BREAKER_CHECK_SECONDS ?? '60');
// v1 熔断已退役，默认统一走 v2；即使环境变量误设 v1，也按 v2 处理。
const DOWN_RISK_ENGINE_DEFAULT: DownRiskEngine = 'v2';
const DOWN_RISK_MODE_DEFAULT = ((): DownRiskMode => {
    const raw = String(process.env.DOWN_RISK_MODE ?? 'off').toLowerCase();
    if (raw === 'shadow' || raw === 'enforce') return raw;
    return 'off';
})();
const DOWN_RISK_LOOKBACK_MAIN_HOURS_DEFAULT = Number(process.env.DOWN_RISK_LOOKBACK_MAIN_HOURS ?? '6');
const DOWN_RISK_LOOKBACK_CALIB_HOURS_DEFAULT = Number(process.env.DOWN_RISK_LOOKBACK_CALIB_HOURS ?? '24');
const DOWN_RISK_LOOKBACK_FAST_HOURS_DEFAULT = Number(process.env.DOWN_RISK_LOOKBACK_FAST_HOURS ?? '2');
const DOWN_RISK_MIN_TRADES_SOFT_DEFAULT = Number(process.env.DOWN_RISK_MIN_TRADES_SOFT ?? '8');
const DOWN_RISK_MIN_TRADES_HARD_DEFAULT = Number(process.env.DOWN_RISK_MIN_TRADES_HARD ?? '12');
const DOWN_RISK_MIN_TRADES_FAST_DEFAULT = Number(process.env.DOWN_RISK_MIN_TRADES_FAST ?? '4');
const DOWN_RISK_WR_SOFT_DEFAULT = Number(process.env.DOWN_RISK_WR_SOFT ?? '0.42');
const DOWN_RISK_WR_HARD_DEFAULT = Number(process.env.DOWN_RISK_WR_HARD ?? '0.35');
const DOWN_RISK_WR_FAST_BLOCKED_DEFAULT = Number(process.env.DOWN_RISK_WR_FAST_BLOCKED ?? '0.25');
const DOWN_RISK_PNL_SOFT_DEFAULT = Number(process.env.DOWN_RISK_PNL_SOFT ?? '-0.12');
const DOWN_RISK_PNL_HARD_DEFAULT = Number(process.env.DOWN_RISK_PNL_HARD ?? '-0.25');
const DOWN_RISK_PNL_FAST_BLOCKED_DEFAULT = Number(process.env.DOWN_RISK_PNL_FAST_BLOCKED ?? '-0.03');
const DOWN_RISK_SOFT_EXTRA_DELTA_DEFAULT = Number(process.env.DOWN_RISK_SOFT_EXTRA_DELTA ?? '0.02');
const DOWN_RISK_SOFT_BET_SCALE_DEFAULT = Number(process.env.DOWN_RISK_SOFT_BET_SCALE ?? '0.60');
const DOWN_RISK_HARD_HOLD_BARS_DEFAULT = Number(process.env.DOWN_RISK_HARD_HOLD_BARS ?? '4');
const DOWN_RISK_HARD_HOLD_BARS_PROFILE70_DEFAULT = Number(process.env.DOWN_RISK_HARD_HOLD_BARS_PROFILE70 ?? '6');
const DOWN_RISK_RELEASE_CHECKS_DEFAULT = Number(process.env.DOWN_RISK_RELEASE_CHECKS ?? '3');
const DOWN_RISK_RELEASE_WR_DEFAULT = Number(process.env.DOWN_RISK_RELEASE_WR ?? '0.45');
const DOWN_RISK_RELEASE_PNL_DEFAULT = Number(process.env.DOWN_RISK_RELEASE_PNL ?? '-0.05');
const DOWN_RISK_RELEASE_CONFIRM_HOURS_DEFAULT = Number(process.env.DOWN_RISK_RELEASE_CONFIRM_HOURS ?? '6');
const DOWN_RISK_STATS_MODE_DEFAULT: DownRiskStatsMode = ((): DownRiskStatsMode => {
    const raw = String(process.env.DOWN_RISK_STATS_MODE ?? 'route').toLowerCase();
    if (raw === 'simulation' || raw === 'live' || raw === 'route') return raw;
    return 'route';
})();
const DOWN_RISK_CHECK_SECONDS_DEFAULT = Number(process.env.DOWN_RISK_CHECK_SECONDS ?? String(DOWN_BREAKER_CHECK_SECONDS_DEFAULT));
const DOWN_RISK_TARGET_SYMBOLS = new Set(['BTC', 'ETH']);
const UP_RISK_ENGINE_DEFAULT = String(process.env.UP_RISK_ENGINE ?? 'v1').toLowerCase() === 'v2' ? 'v2' : 'v1';
const UP_RISK_MODE_DEFAULT = ((): UpRiskMode => {
    const raw = String(process.env.UP_RISK_MODE ?? 'off').toLowerCase();
    if (raw === 'shadow' || raw === 'enforce') return raw;
    return 'off';
})();
const UP_RISK_LOOKBACK_MAIN_HOURS_DEFAULT = Number(process.env.UP_RISK_LOOKBACK_MAIN_HOURS ?? '6');
const UP_RISK_LOOKBACK_CALIB_HOURS_DEFAULT = Number(process.env.UP_RISK_LOOKBACK_CALIB_HOURS ?? '24');
const UP_RISK_MIN_TRADES_SOFT_DEFAULT = Number(process.env.UP_RISK_MIN_TRADES_SOFT ?? '8');
const UP_RISK_WR_SOFT_DEFAULT = Number(process.env.UP_RISK_WR_SOFT ?? '0.42');
const UP_RISK_PNL_SOFT_DEFAULT = Number(process.env.UP_RISK_PNL_SOFT ?? '-0.12');
const UP_RISK_SOFT_EXTRA_DELTA_DEFAULT = Number(process.env.UP_RISK_SOFT_EXTRA_DELTA ?? '0.02');
const UP_RISK_SOFT_BET_SCALE_DEFAULT = Number(process.env.UP_RISK_SOFT_BET_SCALE ?? '0.60');
const UP_RISK_RELEASE_CHECKS_DEFAULT = Number(process.env.UP_RISK_RELEASE_CHECKS ?? '3');
const UP_RISK_RELEASE_WR_DEFAULT = Number(process.env.UP_RISK_RELEASE_WR ?? '0.45');
const UP_RISK_RELEASE_PNL_DEFAULT = Number(process.env.UP_RISK_RELEASE_PNL ?? '-0.05');
const UP_RISK_STATS_MODE_DEFAULT: UpRiskStatsMode = ((): UpRiskStatsMode => {
    const raw = String(process.env.UP_RISK_STATS_MODE ?? 'route').toLowerCase();
    if (raw === 'simulation' || raw === 'live' || raw === 'route') return raw;
    return 'route';
})();
const UP_RISK_CHECK_SECONDS_DEFAULT = Number(process.env.UP_RISK_CHECK_SECONDS ?? String(DOWN_BREAKER_CHECK_SECONDS_DEFAULT));
const UP_RISK_TARGET_SYMBOLS = new Set(['BTC', 'ETH']);
const SHOCK_RISK_ENGINE_DEFAULT = String(process.env.SHOCK_RISK_ENGINE ?? 'v2').toLowerCase() === 'v1' ? 'v1' : 'v2';
const SHOCK_RISK_MODE_DEFAULT = ((): ShockRiskMode => {
    const raw = String(process.env.SHOCK_RISK_MODE ?? 'shadow').toLowerCase();
    if (raw === 'off' || raw === 'enforce') return raw;
    return 'shadow';
})();
const SHOCK_RISK_STATS_MODE_DEFAULT: ShockRiskStatsMode = ((): ShockRiskStatsMode => {
    const raw = String(process.env.SHOCK_RISK_STATS_MODE ?? 'route').toLowerCase();
    if (raw === 'simulation' || raw === 'live' || raw === 'route') return raw;
    return 'route';
})();
const SHOCK_RISK_WINDOW_MINUTES_DEFAULT = Number(process.env.SHOCK_RISK_WINDOW_MINUTES ?? '90');
const SHOCK_RISK_MIN_TRADES_DEFAULT = Number(process.env.SHOCK_RISK_MIN_TRADES ?? '25');
const SHOCK_RISK_WR_POST_DEFAULT = Number(process.env.SHOCK_RISK_WR_POST ?? '0.35');
const SHOCK_RISK_PNL_PER_TRADE_DEFAULT = Number(process.env.SHOCK_RISK_PNL_PER_TRADE ?? '-0.08');
const SHOCK_RISK_VOL_PERCENTILE_DEFAULT = Number(process.env.SHOCK_RISK_VOL_PERCENTILE ?? '85');
const SHOCK_RISK_SEVERE_VOL_RATIO_DEFAULT = Number(process.env.SHOCK_RISK_SEVERE_VOL_RATIO ?? '2.2');
const SHOCK_RISK_HOLD_HOURS_DEFAULT = Number(process.env.SHOCK_RISK_HOLD_HOURS ?? '12');
const SHOCK_RISK_SEVERE_HOLD_HOURS_DEFAULT = Number(process.env.SHOCK_RISK_SEVERE_HOLD_HOURS ?? '18');
const SHOCK_RISK_RELEASE_CHECKS_DEFAULT = Number(process.env.SHOCK_RISK_RELEASE_CHECKS ?? '3');
const SHOCK_RISK_RELEASE_WR_DEFAULT = Number(process.env.SHOCK_RISK_RELEASE_WR ?? '0.45');
const SHOCK_RISK_RELEASE_PNL_DEFAULT = Number(process.env.SHOCK_RISK_RELEASE_PNL ?? '-0.05');
const SHOCK_RISK_CHECK_SECONDS_DEFAULT = Number(process.env.SHOCK_RISK_CHECK_SECONDS ?? '60');
const SHOCK_RISK_VOL_BASE_LOOKBACK_HOURS = Number(process.env.SHOCK_RISK_VOL_BASE_LOOKBACK_HOURS ?? '72');
const SHOCK_RISK_TARGET_SYMBOLS = new Set(['BTC', 'ETH']);
const COMBO_PAUSE_ENABLED_DEFAULT = parseBoolEnv(process.env.COMBO_PAUSE_ENABLED, true);
const COMBO_PAUSE_MODE_DEFAULT = ((): ComboPauseMode => {
    const raw = String(process.env.COMBO_PAUSE_MODE ?? 'shadow').toLowerCase();
    if (raw === 'off' || raw === 'enforce') return raw;
    return 'shadow';
})();
const COMBO_PAUSE_STATS_MODE_DEFAULT: ComboPauseStatsMode = ((): ComboPauseStatsMode => {
    const raw = String(process.env.COMBO_PAUSE_STATS_MODE ?? 'route').toLowerCase();
    if (raw === 'simulation' || raw === 'live' || raw === 'route') return raw;
    return 'route';
})();
const COMBO_PAUSE_DRAWDOWN_2H_DEFAULT = Number(process.env.COMBO_PAUSE_DRAWDOWN_2H ?? '0.12');
const COMBO_PAUSE_HOLD_MINUTES_DEFAULT = Number(process.env.COMBO_PAUSE_HOLD_MINUTES ?? '120');
const COMBO_PAUSE_CHECK_SECONDS_DEFAULT = Number(process.env.COMBO_PAUSE_CHECK_SECONDS ?? '60');
const COMBO_PAUSE_TARGET_SYMBOLS = new Set(['BTC', 'ETH']);
const CALIBRATION_MODE_DEFAULT = ((): CalibrationMode => {
    const raw = String(process.env.CALIBRATION_MODE ?? 'off').toLowerCase();
    if (raw === 'shadow' || raw === 'enforce') return raw;
    return 'off';
})();
const CALIBRATION_STATS_MODE_DEFAULT: CalibrationStatsMode = ((): CalibrationStatsMode => {
    const raw = String(process.env.CALIBRATION_STATS_MODE ?? 'route').toLowerCase();
    if (raw === 'simulation' || raw === 'live' || raw === 'route') return raw;
    return 'route';
})();
const CALIBRATION_LOOKBACK_HOURS_DEFAULT = Number(process.env.CALIBRATION_LOOKBACK_HOURS ?? '72');
const CALIBRATION_MIN_SAMPLES_DEFAULT = Number(process.env.CALIBRATION_MIN_SAMPLES ?? '24');
const CALIBRATION_CHECK_SECONDS_DEFAULT = Number(process.env.CALIBRATION_CHECK_SECONDS ?? '60');
const CALIBRATION_METHOD_DEFAULT: CalibrationMethod = ((): CalibrationMethod => {
    const raw = String(process.env.CALIBRATION_METHOD ?? 'identity').toLowerCase();
    if (raw === 'temperature' || raw === 'sigmoid' || raw === 'isotonic') return raw;
    return 'identity';
})();
const CALIBRATION_TEMPERATURE_DEFAULT = Number(process.env.CALIBRATION_TEMPERATURE ?? '1.0');
const CALIBRATION_ALPHA_DEFAULT = Number(process.env.CALIBRATION_ALPHA ?? '0.0');
const CALIBRATION_BETA_DEFAULT = Number(process.env.CALIBRATION_BETA ?? '1.0');
const CALIBRATION_TARGET_SYMBOLS = new Set(['BTC', 'ETH']);
const EXPECTANCY_GATE_MODE_DEFAULT = ((): ExpectancyGateMode => {
    const raw = String(process.env.EXPECTANCY_GATE_MODE ?? 'off').toLowerCase();
    if (raw === 'shadow' || raw === 'enforce') return raw;
    return 'off';
})();
const EXPECTANCY_GATE_STATS_MODE_DEFAULT: ExpectancyGateStatsMode = ((): ExpectancyGateStatsMode => {
    const raw = String(process.env.EXPECTANCY_GATE_STATS_MODE ?? 'route').toLowerCase();
    if (raw === 'simulation' || raw === 'live' || raw === 'route') return raw;
    return 'route';
})();
const EXPECTANCY_GATE_LOOKBACK_HOURS_DEFAULT = Number(process.env.EXPECTANCY_GATE_LOOKBACK_HOURS ?? '72');
const EXPECTANCY_GATE_LOOKBACK_FAST_HOURS_DEFAULT = Number(process.env.EXPECTANCY_GATE_LOOKBACK_FAST_HOURS ?? '12');
const EXPECTANCY_GATE_MIN_TRADES_DEFAULT = Number(process.env.EXPECTANCY_GATE_MIN_TRADES ?? '12');
const EXPECTANCY_GATE_MIN_TRADES_BLOCKED_DEFAULT = Number(process.env.EXPECTANCY_GATE_MIN_TRADES_BLOCKED ?? '18');
const EXPECTANCY_GATE_MIN_TRADES_FAST_DEFAULT = Number(process.env.EXPECTANCY_GATE_MIN_TRADES_FAST ?? '4');
const EXPECTANCY_GATE_WR_DEGRADED_DEFAULT = Number(process.env.EXPECTANCY_GATE_WR_DEGRADED ?? '0.48');
const EXPECTANCY_GATE_PNL_DEGRADED_DEFAULT = Number(process.env.EXPECTANCY_GATE_PNL_DEGRADED ?? '-0.02');
const EXPECTANCY_GATE_DEGRADED_BET_SCALE_DEFAULT = Number(process.env.EXPECTANCY_GATE_DEGRADED_BET_SCALE ?? '0.75');
const EXPECTANCY_GATE_DEGRADED_EXTRA_DELTA_DEFAULT = Number(process.env.EXPECTANCY_GATE_DEGRADED_EXTRA_DELTA ?? '0.02');
const EXPECTANCY_GATE_WR_BLOCKED_DEFAULT = Number(process.env.EXPECTANCY_GATE_WR_BLOCKED ?? '0.42');
const EXPECTANCY_GATE_PNL_BLOCKED_DEFAULT = Number(process.env.EXPECTANCY_GATE_PNL_BLOCKED ?? '-0.05');
const EXPECTANCY_GATE_WR_FAST_BLOCKED_DEFAULT = Number(process.env.EXPECTANCY_GATE_WR_FAST_BLOCKED ?? '0.30');
const EXPECTANCY_GATE_PNL_FAST_BLOCKED_DEFAULT = Number(process.env.EXPECTANCY_GATE_PNL_FAST_BLOCKED ?? '-0.06');
const EXPECTANCY_GATE_BLOCKED_ENTER_CHECKS_DEFAULT = Number(process.env.EXPECTANCY_GATE_BLOCKED_ENTER_CHECKS ?? '2');
const EXPECTANCY_GATE_RELEASE_CHECKS_DEFAULT = Number(process.env.EXPECTANCY_GATE_RELEASE_CHECKS ?? '2');
const EXPECTANCY_GATE_RELEASE_WR_DEFAULT = Number(process.env.EXPECTANCY_GATE_RELEASE_WR ?? '0.50');
const EXPECTANCY_GATE_RELEASE_PNL_DEFAULT = Number(process.env.EXPECTANCY_GATE_RELEASE_PNL ?? '0.0');
const EXPECTANCY_GATE_CHECK_SECONDS_DEFAULT = Number(process.env.EXPECTANCY_GATE_CHECK_SECONDS ?? '60');
const EXPECTANCY_GATE_TARGET_SYMBOLS = new Set(['BTC', 'ETH']);
const UNCERTAINTY_GATE_MODE_DEFAULT = ((): 'off' | 'shadow' | 'enforce' => {
    const raw = String(process.env.UNCERTAINTY_GATE_MODE_DEFAULT ?? 'off').toLowerCase();
    if (raw === 'shadow' || raw === 'enforce') return raw;
    return 'off';
})();
const UNCERTAINTY_GATE_DISPERSION_MAX_DEFAULT = Number(process.env.UNCERTAINTY_GATE_DISPERSION_MAX ?? '0.18');
const UNCERTAINTY_GATE_HIGH_VOL_DISPERSION_MAX_DEFAULT = Number(process.env.UNCERTAINTY_GATE_HIGH_VOL_DISPERSION_MAX ?? '0.12');
const UNCERTAINTY_GATE_INTERVAL_WIDTH_MAX_DEFAULT = Number(process.env.UNCERTAINTY_GATE_INTERVAL_WIDTH_MAX ?? '0.32');
const UNCERTAINTY_GATE_HIGH_VOL_INTERVAL_WIDTH_MAX_DEFAULT = Number(process.env.UNCERTAINTY_GATE_HIGH_VOL_INTERVAL_WIDTH_MAX ?? '0.26');
const UNCERTAINTY_GATE_INTERVAL_Z_SCORE_DEFAULT = Number(process.env.UNCERTAINTY_GATE_INTERVAL_Z_SCORE ?? '1.35');
const UNCERTAINTY_GATE_INTERVAL_DISPERSION_BLEND_DEFAULT = Number(process.env.UNCERTAINTY_GATE_INTERVAL_DISPERSION_BLEND ?? '0.85');
const UNCERTAINTY_GATE_EFFECTIVE_SOURCE_COUNT_MIN_DEFAULT = Number(process.env.UNCERTAINTY_GATE_EFFECTIVE_SOURCE_COUNT_MIN ?? '2');
const UNCERTAINTY_GATE_CHANGE_POINT_PROB_MAX_DEFAULT = Number(process.env.UNCERTAINTY_GATE_CHANGE_POINT_PROB_MAX ?? '0.82');
const UNCERTAINTY_GATE_COUNTERTREND_CHANGE_POINT_PROB_MAX_DEFAULT = Number(process.env.UNCERTAINTY_GATE_COUNTERTREND_CHANGE_POINT_PROB_MAX ?? '0.60');
const UNCERTAINTY_GATE_DEGRADED_EXTRA_DELTA_DEFAULT = Number(process.env.UNCERTAINTY_GATE_DEGRADED_EXTRA_DELTA ?? '0.03');
const THRESHOLD_DRIFT_MODE_DEFAULT = ((): ThresholdDriftMode => {
    const raw = String(process.env.THRESHOLD_DRIFT_MODE ?? 'off').toLowerCase();
    if (raw === 'shadow' || raw === 'enforce') return raw;
    return 'off';
})();
const THRESHOLD_DRIFT_STATS_MODE_DEFAULT: ThresholdDriftStatsMode = ((): ThresholdDriftStatsMode => {
    const raw = String(process.env.THRESHOLD_DRIFT_STATS_MODE ?? 'route').toLowerCase();
    if (raw === 'simulation' || raw === 'live' || raw === 'route') return raw;
    return 'route';
})();
const THRESHOLD_DRIFT_LOOKBACK_HOURS_DEFAULT = Number(process.env.THRESHOLD_DRIFT_LOOKBACK_HOURS ?? '72');
const THRESHOLD_DRIFT_MIN_TRADES_DEFAULT = Number(process.env.THRESHOLD_DRIFT_MIN_TRADES ?? '12');
const THRESHOLD_DRIFT_WR_DEGRADED_DEFAULT = Number(process.env.THRESHOLD_DRIFT_WR_DEGRADED ?? '0.48');
const THRESHOLD_DRIFT_PNL_DEGRADED_DEFAULT = Number(process.env.THRESHOLD_DRIFT_PNL_DEGRADED ?? '-0.02');
const THRESHOLD_DRIFT_DELTA_STEP_DEFAULT = Number(process.env.THRESHOLD_DRIFT_DELTA_STEP ?? '0.01');
const THRESHOLD_DRIFT_DELTA_MAX_DEFAULT = Number(process.env.THRESHOLD_DRIFT_DELTA_MAX ?? '0.06');
const THRESHOLD_DRIFT_RELEASE_CHECKS_DEFAULT = Number(process.env.THRESHOLD_DRIFT_RELEASE_CHECKS ?? '2');
const THRESHOLD_DRIFT_RELEASE_WR_DEFAULT = Number(process.env.THRESHOLD_DRIFT_RELEASE_WR ?? '0.50');
const THRESHOLD_DRIFT_RELEASE_PNL_DEFAULT = Number(process.env.THRESHOLD_DRIFT_RELEASE_PNL ?? '0.0');
const THRESHOLD_DRIFT_CHECK_SECONDS_DEFAULT = Number(process.env.THRESHOLD_DRIFT_CHECK_SECONDS ?? '60');
const THRESHOLD_DRIFT_TARGET_SYMBOLS = new Set(['BTC', 'ETH']);
const META_LABEL_MODE_DEFAULT = ((): MetaLabelMode => {
    const raw = String(process.env.META_LABEL_MODE ?? 'off').toLowerCase();
    if (raw === 'shadow' || raw === 'enforce') return raw;
    return 'off';
})();
const SELECTOR_MODE_DEFAULT = ((): SelectorMode => {
    const raw = String(process.env.SELECTOR_MODE ?? 'off').toLowerCase();
    if (raw === 'shadow' || raw === 'enforce') return raw;
    return 'off';
})();
const SELECTOR_STATS_MODE_DEFAULT: SelectorStatsMode = ((): SelectorStatsMode => {
    const raw = String(process.env.SELECTOR_STATS_MODE ?? 'route').toLowerCase();
    if (raw === 'simulation' || raw === 'live' || raw === 'route') return raw;
    return 'route';
})();
const SELECTOR_CHECK_SECONDS_DEFAULT = Number(process.env.SELECTOR_CHECK_SECONDS ?? '60');
const SELECTOR_TARGET_SYMBOLS = new Set(['BTC', 'ETH']);
const REGIME_MODE_DEFAULT = ((): RegimeMode => {
    const raw = String(process.env.REGIME_MODE ?? 'off').toLowerCase();
    if (raw === 'shadow' || raw === 'enforce') return raw;
    return 'off';
})();
const REGIME_STATS_MODE_DEFAULT: RegimeStatsMode = ((): RegimeStatsMode => {
    const raw = String(process.env.REGIME_STATS_MODE ?? 'route').toLowerCase();
    if (raw === 'simulation' || raw === 'live' || raw === 'route') return raw;
    return 'route';
})();
const REGIME_CHECK_SECONDS_DEFAULT = Number(process.env.REGIME_CHECK_SECONDS ?? '60');
const REGIME_RUNTIME_DIR = path.join(process.cwd(), 'logs', 'runtime');
const REGIME_TARGET_SYMBOLS = new Set(['BTC', 'ETH']);

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
const MODE_EFFECTIVE_DIR = path.join(process.cwd(), 'logs', 'runtime');
const MODE_REGISTRY_REQUIRED_SLOT_FIELDS = [
    'privateKeyEnv',
    'proxyWalletEnv',
] as const;
const MODE_REGISTRY_OPTIONAL_SLOT_FIELDS = [
    'polyApiKeyEnv',
    'polyApiSecretEnv',
    'polyApiPassphraseEnv',
    'builderApiKeyEnv',
    'builderApiSecretEnv',
    'builderApiPassphraseEnv',
] as const;

function parseBoolEnv(value: string | undefined, defaultValue: boolean): boolean {
    if (value == null || value === '') return defaultValue;
    return ['1', 'true', 'yes', 'on'].includes(value.toLowerCase());
}

function safeGroupName(value: string): string {
    return value.replace(/[^a-zA-Z0-9_-]/g, '_');
}

function detectProfile(groupName: string, configPath: string): 'default' | '70' {
    if (groupName.endsWith('_70')) return '70';
    if (path.basename(configPath).includes('_70')) return '70';
    return 'default';
}

function inferProfileFromGroup(groupName: string): 'default' | '70' {
    return groupName.endsWith('_70') ? '70' : 'default';
}

function parseAllowedSymbols(raw: string): string[] {
    return raw
        .split(',')
        .map((s) => s.trim().toUpperCase())
        .filter((s) => s.length > 0);
}

function parseIsoDateSafe(value: string | undefined): number | null {
    if (!value) return null;
    const ms = Date.parse(value);
    if (!Number.isFinite(ms)) return null;
    return ms;
}

function createDefaultModeEntries(configs: TraderConfig[]): EffectiveModeEntry[] {
    const out: EffectiveModeEntry[] = [];
    for (const cfg of configs) {
        const symbols = parseAllowedSymbols(cfg.allowedMarkets || '');
        for (const symbol of symbols) {
            out.push({
                traderName: cfg.name,
                symbol,
                desiredMode: 'simulation',
                appliedMode: 'simulation',
                enabled: true,
                reason: 'default_simulation',
            });
        }
    }
    return out;
}

function resolveModeRegistryRuntimeState(
    groupName: string,
    configPath: string,
    groupConfigs: TraderConfig[],
): ModeRegistryRuntimeState {
    const enabled = parseBoolEnv(process.env.MODE_REGISTRY_ENABLED, false);
    const validateOnly = parseBoolEnv(process.env.MODE_REGISTRY_VALIDATE_ONLY, true);
    const failOpen = parseBoolEnv(process.env.MODE_REGISTRY_FAIL_OPEN, true);
    const profile = detectProfile(groupName, configPath);
    const registryFile = path.resolve(process.env.MODE_REGISTRY_FILE || path.join(process.cwd(), 'trading_mode_registry.json'));
    const walletSlotsFile = path.resolve(process.env.WALLET_SLOTS_FILE || path.join(process.cwd(), 'wallet_slots.json'));
    const warnings: string[] = [];
    const errors: string[] = [];
    const entries = createDefaultModeEntries(groupConfigs);

    const shouldValidate = enabled || validateOnly;
    if (!shouldValidate) {
        return {
            enabled,
            validateOnly,
            failOpen,
            profile,
            registryFile,
            walletSlotsFile,
            activeByTime: false,
            warnings,
            errors,
            entries: entries.map((e) => ({ ...e, reason: 'mode_registry_disabled' })),
            walletSlots: {},
        };
    }

    let registryRaw: string;
    let slotsRaw: string;
    try {
        registryRaw = fs.readFileSync(registryFile, 'utf8');
    } catch (err) {
        errors.push(`mode registry file read failed: ${registryFile} (${String(err)})`);
        registryRaw = '{}';
    }
    try {
        slotsRaw = fs.readFileSync(walletSlotsFile, 'utf8');
    } catch (err) {
        errors.push(`wallet slots file read failed: ${walletSlotsFile} (${String(err)})`);
        slotsRaw = '{}';
    }

    let registry: ModeRegistryFile = {};
    let slots: WalletSlotsFile = {};
    try {
        registry = JSON.parse(registryRaw) as ModeRegistryFile;
    } catch (err) {
        errors.push(`mode registry JSON parse failed: ${String(err)}`);
    }
    try {
        slots = JSON.parse(slotsRaw) as WalletSlotsFile;
    } catch (err) {
        errors.push(`wallet slots JSON parse failed: ${String(err)}`);
    }

    const globalLiveSlotOwners = new Map<string, Set<string>>();
    const rawProfiles = registry.profiles && typeof registry.profiles === 'object'
        ? registry.profiles
        : {};
    for (const [rawProfile, rawProfileObj] of Object.entries(rawProfiles)) {
        const rawProfileEntries = Array.isArray((rawProfileObj as any)?.entries)
            ? (rawProfileObj as any).entries
            : [];
        rawProfileEntries.forEach((entry: any) => {
            const traderName = String(entry?.traderName || '').trim();
            const symbol = String(entry?.symbol || '').trim().toUpperCase();
            const mode = String(entry?.mode || '').trim().toLowerCase();
            const enabledEntry = entry?.enabled !== false;
            const walletSlot = String(entry?.walletSlot || '').trim();
            if (!traderName || !symbol || mode !== 'live' || !enabledEntry || !walletSlot) return;
            const ownerKey = `${String(rawProfile || '').trim()}:${traderName}::${symbol}`;
            const ownerSet = globalLiveSlotOwners.get(walletSlot) || new Set<string>();
            ownerSet.add(ownerKey);
            globalLiveSlotOwners.set(walletSlot, ownerSet);
        });
    }
    for (const [walletSlot, owners] of globalLiveSlotOwners.entries()) {
        const ownerList = [...owners].sort();
        if (ownerList.length > 1) {
            errors.push(`walletSlot reused by multiple live entries across profiles: ${walletSlot} used by ${ownerList.join(', ')}`);
        }
    }

    const effectiveFrom = typeof registry.effectiveFrom === 'string' ? registry.effectiveFrom : undefined;
    const effectiveMs = parseIsoDateSafe(effectiveFrom);
    if (effectiveFrom && effectiveMs == null) {
        // fail-closed: 时间格式错时绝不放行 live
        errors.push(`invalid effectiveFrom format: ${effectiveFrom}`);
    }
    const nowMs = Date.now();
    const activeByTime = effectiveMs == null
        ? (effectiveFrom ? false : true)
        : nowMs >= effectiveMs;

    const profileObj = registry.profiles?.[profile];
    const rawEntries = profileObj?.entries;
    if (!Array.isArray(rawEntries)) {
        warnings.push(`profiles.${profile}.entries missing or not array, fallback to simulation`);
    }

    const validPairs = new Map<string, { traderName: string; symbol: string }>();
    for (const e of entries) {
        validPairs.set(`${e.traderName}::${e.symbol}`, { traderName: e.traderName, symbol: e.symbol });
    }
    const modeByPair = new Map<string, ModeRegistryEntry>();
    const liveSlotOwner = new Map<string, string>();

    if (Array.isArray(rawEntries)) {
        rawEntries.forEach((entry, index) => {
            const traderName = String(entry?.traderName || '').trim();
            const symbol = String(entry?.symbol || '').trim().toUpperCase();
            const mode = String(entry?.mode || '').trim().toLowerCase();
            const enabledEntry = entry?.enabled !== false;
            const walletSlot = String(entry?.walletSlot || '').trim();
            if (!traderName || !symbol) {
                errors.push(`entry[${index}] missing traderName/symbol`);
                return;
            }
            if (mode !== 'simulation' && mode !== 'live') {
                errors.push(`entry[${index}] invalid mode=${mode}`);
                return;
            }
            const key = `${traderName}::${symbol}`;
            if (modeByPair.has(key)) {
                errors.push(`duplicate entry: ${key}`);
                return;
            }
            if (!validPairs.has(key)) {
                // 允许同 profile 里保留其它 inactive/simulation 条目，避免当前 group 的日志被历史遗留噪音刷屏。
                if (mode === 'live' && enabledEntry) {
                    warnings.push(`entry[${index}] not in active group/profile configs: ${key}`);
                }
                return;
            }
            if (mode === 'live' && enabledEntry) {
                if (!walletSlot) {
                    errors.push(`entry[${index}] live mode missing walletSlot for ${key}`);
                } else {
                    const owner = liveSlotOwner.get(walletSlot);
                    if (owner && owner !== key) {
                        errors.push(`walletSlot reused by multiple live entries: ${walletSlot} used by ${owner} and ${key}`);
                    } else {
                        liveSlotOwner.set(walletSlot, key);
                    }
                    const slot = slots[walletSlot];
                    if (!slot || typeof slot !== 'object') {
                        errors.push(`entry[${index}] walletSlot not found: ${walletSlot}`);
                    } else {
                        for (const field of MODE_REGISTRY_REQUIRED_SLOT_FIELDS) {
                            const value = slot[field];
                            if (typeof value !== 'string' || value.trim().length === 0) {
                                errors.push(`walletSlot ${walletSlot} missing required field ${field}`);
                            }
                        }
                        for (const field of MODE_REGISTRY_OPTIONAL_SLOT_FIELDS) {
                            const value = slot[field];
                            if (typeof value !== 'string' || value.trim().length === 0) {
                                warnings.push(`walletSlot ${walletSlot} missing optional field ${field}`);
                            }
                        }
                    }
                }
            }
            modeByPair.set(key, {
                traderName,
                symbol,
                mode: mode as TradingModeValue,
                walletSlot: walletSlot || undefined,
                enabled: enabledEntry,
            });
        });
    }

    const hasBlocking = errors.length > 0;
    const forceSimulation = hasBlocking || !enabled || !activeByTime || validateOnly;
    const forcedReason = hasBlocking
        ? 'validation_error_fallback_to_simulation'
        : (!enabled
            ? 'mode_registry_disabled'
            : (!activeByTime
                ? 'effective_from_not_reached'
                : (validateOnly ? 'validate_only' : 'registry_applied')));

    const mergedEntries: EffectiveModeEntry[] = entries.map((base): EffectiveModeEntry => {
        const key = `${base.traderName}::${base.symbol}`;
        const override = modeByPair.get(key);
        if (!override || !override.enabled) {
            return {
                ...base,
                desiredMode: 'simulation' as TradingModeValue,
                appliedMode: 'simulation' as TradingModeValue,
                enabled: override?.enabled ?? true,
                reason: override ? 'entry_disabled' : forcedReason,
            };
        }
        const desiredMode = override.mode;
        const appliedMode: TradingModeValue = forceSimulation ? 'simulation' : desiredMode;
        return {
            ...base,
            desiredMode,
            appliedMode,
            walletSlot: override.walletSlot,
            enabled: true,
            reason: forceSimulation
                ? forcedReason
                : (desiredMode === 'live' ? 'live_effective' : 'simulation_effective'),
        };
    });

    return {
        enabled,
        validateOnly,
        failOpen,
        profile,
        registryFile,
        walletSlotsFile,
        effectiveFrom,
        activeByTime,
        warnings,
        errors,
        entries: mergedEntries,
        walletSlots: slots,
    };
}

function applyWalletSlotEnv(cfg: TraderConfig, slot?: WalletSlot): TraderConfig {
    if (!slot) return cfg;
    return {
        ...cfg,
        privateKeyEnv: slot.privateKeyEnv || cfg.privateKeyEnv,
        proxyWalletEnv: slot.proxyWalletEnv || cfg.proxyWalletEnv,
        polyApiKeyEnv: slot.polyApiKeyEnv || cfg.polyApiKeyEnv,
        polyApiSecretEnv: slot.polyApiSecretEnv || cfg.polyApiSecretEnv,
        polyApiPassphraseEnv: slot.polyApiPassphraseEnv || cfg.polyApiPassphraseEnv,
        builderApiKeyEnv: slot.builderApiKeyEnv || cfg.builderApiKeyEnv,
        builderApiSecretEnv: slot.builderApiSecretEnv || cfg.builderApiSecretEnv,
        builderApiPassphraseEnv: slot.builderApiPassphraseEnv || cfg.builderApiPassphraseEnv,
    };
}

interface ResolvedSymbolRoute {
    symbol: string;
    mode: TradingModeValue;
    slotName?: string;
}

function isCore10ExperimentalLiveApproved(
    cfg: TraderConfig,
    symbol: string,
): boolean {
    const tuningMap = cfg.core10ExperimentalTuningBySymbol;
    const approvalMap = cfg.core10ExperimentalLiveApprovalBySymbol;
    const rawNode = tuningMap ? tuningMap[symbol] : undefined;
    let tuned = false;
    if (typeof rawNode === 'boolean') {
        tuned = rawNode;
    } else if (rawNode && typeof rawNode === 'object') {
        tuned = rawNode.active !== false;
    }
    if (!tuned) return true;
    return Boolean(approvalMap && approvalMap[symbol]);
}

function resolveSymbolRoutes(
    cfg: TraderConfig,
    modeState: ModeRegistryRuntimeState,
): {
    routes: ResolvedSymbolRoute[];
    warnings: string[];
} {
    const warnings: string[] = [];
    const symbols = parseAllowedSymbols(cfg.allowedMarkets || '');
    const modeBySymbol = cfg.symbolModeOverrides || {};
    const slotBySymbol = cfg.walletSlotBySymbol || {};
    const routes: ResolvedSymbolRoute[] = [];

    for (const symbol of symbols) {
        const desiredMode = modeBySymbol[symbol] === 'live' ? 'live' : 'simulation';
        if (desiredMode !== 'live') {
            routes.push({ symbol, mode: 'simulation' });
            continue;
        }
        if (!isCore10ExperimentalLiveApproved(cfg, symbol)) {
            warnings.push(`trader=${cfg.name} symbol=${symbol} 存在未审批的 Core10 experimental tuning，已回落 simulation`);
            routes.push({ symbol, mode: 'simulation' });
            continue;
        }
        const slotName = slotBySymbol[symbol];
        if (!slotName) {
            warnings.push(`trader=${cfg.name} symbol=${symbol} live 缺少 walletSlot，已回落 simulation`);
            routes.push({ symbol, mode: 'simulation' });
            continue;
        }
        if (!modeState.walletSlots[slotName]) {
            warnings.push(`trader=${cfg.name} symbol=${symbol} walletSlot=${slotName} 未定义，已回落 simulation`);
            routes.push({ symbol, mode: 'simulation' });
            continue;
        }
        routes.push({ symbol, mode: 'live', slotName });
    }

    return { routes, warnings };
}

const REPORT_SUMMARY_CANDIDATES = [
    'report_summary.simulation.json',
];

const LIVE_REPORT_SUMMARY_CANDIDATES = [
    'report_summary.live.json',
    'report_summary.json',
];

function readJsonFileSafe(filePath: string): any | null {
    try {
        if (!fs.existsSync(filePath)) return null;
        return JSON.parse(fs.readFileSync(filePath, 'utf8'));
    } catch {
        return null;
    }
}

function readSeedCapitalForSymbol(
    logsDirCandidates: string[],
    symbol: string,
): { capital: number; source: string } | null {
    const normalizedSymbol = String(symbol || '').trim().toUpperCase();
    if (!normalizedSymbol) return null;

    for (const logsDir of logsDirCandidates) {
        const reportsDir = path.join(process.cwd(), logsDir, 'reports');
        for (const reportName of REPORT_SUMMARY_CANDIDATES) {
            const payload = readJsonFileSafe(path.join(reportsDir, reportName));
            if (!payload || typeof payload !== 'object') continue;
            const bySymbol = payload.bySymbol && typeof payload.bySymbol === 'object' ? payload.bySymbol : {};
            const symStats = bySymbol[normalizedSymbol];
            const symCurrentCapital = typeof symStats?.currentCapital === 'number'
                ? symStats.currentCapital
                : null;
            if (symCurrentCapital != null && Number.isFinite(symCurrentCapital)) {
                return {
                    capital: Math.round(symCurrentCapital * 10000) / 10000,
                    source: `${logsDir}/reports/${reportName}:bySymbol.${normalizedSymbol}.currentCapital`,
                };
            }

            const perCoinCapital = payload.config?.perCoinCapital;
            const coinCapital = typeof perCoinCapital?.[normalizedSymbol] === 'number'
                ? perCoinCapital[normalizedSymbol]
                : null;
            if (coinCapital != null && Number.isFinite(coinCapital)) {
                return {
                    capital: Math.round(coinCapital * 10000) / 10000,
                    source: `${logsDir}/reports/${reportName}:config.perCoinCapital.${normalizedSymbol}`,
                };
            }

            const bySymbolKeys = Object.keys(bySymbol || {});
            if (bySymbolKeys.length === 1 && bySymbolKeys[0] === normalizedSymbol) {
                const summaryCapital = typeof payload.summary?.currentCapital === 'number'
                    ? payload.summary.currentCapital
                    : null;
                if (summaryCapital != null && Number.isFinite(summaryCapital)) {
                    return {
                        capital: Math.round(summaryCapital * 10000) / 10000,
                        source: `${logsDir}/reports/${reportName}:summary.currentCapital`,
                    };
                }
            }
        }
    }

    return null;
}

function readRuntimeCurrentCapitalForDecision(
    logsDir: string,
    tradingMode: string | null | undefined,
    symbol: string | null | undefined,
    executorCapitalFallback?: number | null,
): number | null {
    const normalizedMode = String(tradingMode || '').trim().toLowerCase();
    const normalizedSymbol = String(symbol || '').trim().toUpperCase();
    if (normalizedMode === 'live') {
        const reportsDir = path.join(process.cwd(), logsDir, 'reports');
        for (const reportName of LIVE_REPORT_SUMMARY_CANDIDATES) {
            const payload = readJsonFileSafe(path.join(reportsDir, reportName));
            if (!payload || typeof payload !== 'object') continue;
            const bySymbol = payload.bySymbol && typeof payload.bySymbol === 'object' ? payload.bySymbol : {};
            const symStats = normalizedSymbol ? bySymbol[normalizedSymbol] : null;
            const bySymbolCapital = typeof symStats?.currentCapital === 'number'
                ? symStats.currentCapital
                : typeof symStats?.walletCurrentCapital === 'number'
                    ? symStats.walletCurrentCapital
                    : null;
            if (bySymbolCapital != null && Number.isFinite(bySymbolCapital)) {
                return Math.round(bySymbolCapital * 10000) / 10000;
            }
            const summary = payload.summary && typeof payload.summary === 'object' ? payload.summary : {};
            const summaryCandidates = [
                summary.walletCurrentCapital,
                summary.currentCapital,
                summary.officialWalletPortfolioCapitalUsd,
                summary.officialWalletCashBalanceUsd,
            ];
            for (const candidate of summaryCandidates) {
                if (typeof candidate === 'number' && Number.isFinite(candidate)) {
                    return Math.round(candidate * 10000) / 10000;
                }
            }
        }
    }
    if (typeof executorCapitalFallback === 'number' && Number.isFinite(executorCapitalFallback)) {
        return Math.round(executorCapitalFallback * 10000) / 10000;
    }
    return null;
}

function expandRuntimeTraderConfigs(
    cfg: TraderConfig,
    modeState: ModeRegistryRuntimeState,
): {
    configs: TraderConfig[];
    warnings: string[];
} {
    const { routes, warnings } = resolveSymbolRoutes(cfg, modeState);
    const totalSymbolCount = Math.max(1, parseAllowedSymbols(cfg.allowedMarkets || '').length);
    if (routes.length === 0) {
        return {
            configs: [{
                ...cfg,
                appliedTradingMode: 'simulation',
            }],
            warnings: [
                ...warnings,
                `trader=${cfg.name} allowedMarkets 为空，按 simulation 启动`,
            ],
        };
    }

    const forceSplitBySymbol = cfg.runtimeSplitBySymbol === true;
    const independentCellCapital = cfg.runtimeIndependentCellCapital === true;
    const seedCapitalFromReports = cfg.runtimeSeedCapitalFromReports === true;
    const grouped = new Map<string, { mode: TradingModeValue; slotName?: string; symbols: string[] }>();
    for (const route of routes) {
        const splitKeySuffix = forceSplitBySymbol ? `::${route.symbol}` : '';
        const key = route.mode === 'live'
            ? `live::${route.slotName || ''}${splitKeySuffix}`
            : `simulation${splitKeySuffix}`;
        const existing = grouped.get(key);
        if (existing) {
            existing.symbols.push(route.symbol);
        } else {
            grouped.set(key, { mode: route.mode, slotName: route.slotName, symbols: [route.symbol] });
        }
    }

    const groups = [...grouped.values()].map((g) => ({
        ...g,
        symbols: [...new Set(g.symbols)],
    }));

    const shouldSplit = forceSplitBySymbol || groups.length > 1;
    if (shouldSplit) {
        const parts = groups.map((g) => `${g.symbols.join(',')}=${g.mode}${g.slotName ? `(${g.slotName})` : ''}`);
        warnings.push(`trader=${cfg.name} runtime split 生效: ${parts.join(' | ')}`);
    }

    const configs: TraderConfig[] = groups.map((group) => {
        const slot = group.slotName ? modeState.walletSlots[group.slotName] : undefined;
        const nameSuffix = `${group.mode}_${group.symbols.map((s) => s.toLowerCase()).join('_')}`;
        const runtimeName = shouldSplit ? `${cfg.name}__${nameSuffix}` : cfg.name;
        const runtimeLogsDir = shouldSplit ? `${cfg.logsDir}__${nameSuffix}` : cfg.logsDir;
        const splitRatio = shouldSplit
            ? (independentCellCapital ? 1 : (group.symbols.length / totalSymbolCount))
            : 1;
        let runtimeInitialCapital = Math.round((cfg.initialCapital * splitRatio) * 10000) / 10000;
        let capitalSeedSource = shouldSplit
            ? (independentCellCapital ? 'independent_cell_capital' : 'split_ratio')
            : 'base_initial_capital';
        if (group.mode === 'live' && group.symbols.length === 1) {
            const symbol = group.symbols[0].toUpperCase();
            const capitalOverride = Number(cfg.liveSelectedCapitalBySymbol?.[symbol] ?? 0);
            if (Number.isFinite(capitalOverride) && capitalOverride > 0) {
                runtimeInitialCapital = Math.round(capitalOverride * 10000) / 10000;
                capitalSeedSource = String(
                    cfg.liveSelectedCapitalReasonBySymbol?.[symbol]
                    || 'live_selected_display_reference_capital'
                );
            }
        }

        if (seedCapitalFromReports && group.mode === 'live') {
            warnings.push(
                `trader=${cfg.name} split=${runtimeName} 检测到 runtimeSeedCapitalFromReports=true 且 mode=live，已强制禁用历史资金继承（使用配置初始资金）`,
            );
        }

        if (seedCapitalFromReports && group.mode !== 'live' && group.symbols.length === 1) {
            const seed = readSeedCapitalForSymbol(
                [runtimeLogsDir, cfg.logsDir],
                group.symbols[0],
            );
            if (seed) {
                runtimeInitialCapital = seed.capital;
                capitalSeedSource = seed.source;
            }
        }

        const baseCfg: TraderConfig = {
            ...cfg,
            name: runtimeName,
            runtimeBaseName: cfg.runtimeBaseName || cfg.name,
            runtimeBaseLogsDir: cfg.runtimeBaseLogsDir || cfg.logsDir,
            runtimeSymbol: group.symbols.length === 1 ? group.symbols[0] : undefined,
            allowedMarkets: group.symbols.join(','),
            appliedTradingMode: group.mode,
            initialCapital: runtimeInitialCapital,
            logsDir: runtimeLogsDir,
        };
        if (group.symbols.length === 1) {
            const symbol = group.symbols[0];
            const suffixBySymbol = cfg.predictionSuffixBySymbol || {};
            const symbolSuffix = suffixBySymbol[symbol];
            if (typeof symbolSuffix === 'string' && symbolSuffix.trim()) {
                baseCfg.predictionFallbackSuffix = baseCfg.predictionSuffix;
                baseCfg.predictionSuffix = symbolSuffix.trim();
            }
            const pathBySymbol = cfg.predictionPathBySymbol || {};
            const symbolPath = pathBySymbol[symbol];
            if (typeof symbolPath === 'string' && symbolPath.trim()) {
                baseCfg.predictionFallbackPath = baseCfg.predictionPath;
                baseCfg.predictionPath = symbolPath.trim();
            }
        }
        if (group.mode === 'live' && group.symbols.length === 1) {
            const symbol = group.symbols[0].toUpperCase();
            const runtimeOverride = cfg.liveSelectedRuntimeOverrideBySymbol?.[symbol];
            if (runtimeOverride && Object.keys(runtimeOverride).length > 0) {
                Object.assign(baseCfg, _cloneJsonValue(runtimeOverride));
                baseCfg.liveSelectedRuntimeOverrideActiveSource = String(
                    cfg.liveSelectedRuntimeOverrideSourceBySymbol?.[symbol]
                    || 'live_selected_runtime_override'
                ).trim() || 'live_selected_runtime_override';
                baseCfg.liveSelectedRuntimeOverrideActiveScope = String(
                    cfg.liveSelectedRuntimeOverrideScopeBySymbol?.[symbol]
                    || 'cell_specific_runtime_override'
                ).trim() || 'cell_specific_runtime_override';
                warnings.push(
                    `trader=${cfg.name} split=${runtimeName} runtime override applied: symbol=${symbol} source=${baseCfg.liveSelectedRuntimeOverrideActiveSource} scope=${baseCfg.liveSelectedRuntimeOverrideActiveScope}`,
                );
            }
        }
        const withSlot = sanitizeCanonicalLiveSlotConfig(applyWalletSlotEnv(baseCfg, slot));
        if (shouldSplit) {
            warnings.push(
                `trader=${cfg.name} split=${runtimeName} 初始资金: ${cfg.initialCapital} -> ${runtimeInitialCapital} (source=${capitalSeedSource}), logsDir=${runtimeLogsDir}`,
            );
        }
        return withSlot;
    });

    return { configs, warnings };
}

function writeModeEffectiveSnapshot(
    groupName: string,
    state: ModeRegistryRuntimeState,
): void {
    try {
        const safeGroup = safeGroupName(groupName);
        const target = path.join(MODE_EFFECTIVE_DIR, `trading_mode_effective_${safeGroup}.json`);
        if (!fs.existsSync(MODE_EFFECTIVE_DIR)) fs.mkdirSync(MODE_EFFECTIVE_DIR, { recursive: true });
        const payload = {
            generatedAt: new Date().toISOString(),
            group: groupName,
            profile: state.profile,
            flags: {
                enabled: state.enabled,
                validateOnly: state.validateOnly,
                failOpen: state.failOpen,
                activeByTime: state.activeByTime,
            },
            files: {
                registryFile: state.registryFile,
                walletSlotsFile: state.walletSlotsFile,
            },
            effectiveFrom: state.effectiveFrom || null,
            warnings: state.warnings,
            errors: state.errors,
            summary: {
                total: state.entries.length,
                desiredLive: state.entries.filter((e) => e.desiredMode === 'live' && e.enabled).length,
                appliedLive: state.entries.filter((e) => e.appliedMode === 'live' && e.enabled).length,
            },
            entries: state.entries,
        };
        fs.writeFileSync(target, JSON.stringify(payload, null, 2), 'utf8');
    } catch (err) {
        console.warn(`  ⚠️ mode effective snapshot write failed: ${String(err)}`);
    }
}

// ============================================================
// 辅助函数
// ============================================================

function loadExecutedTargetTss(logsDir: string): Set<number> {
    const filePath = path.join(process.cwd(), logsDir, EXECUTED_PERIODS_FILE);
    try {
        const data = fs.readFileSync(filePath, 'utf8');
        const arr = JSON.parse(data);
        const list = Array.isArray(arr) ? arr : [];
        const now = Math.floor(Date.now() / 1000);
        const minTs = now - EXECUTED_PERIODS_RETENTION_SEC;
        return new Set<number>(list.filter((t: number) => t >= minTs));
    } catch {
        return new Set<number>();
    }
}

function saveExecutedTargetTss(logsDir: string, set: Set<number>): void {
    const filePath = path.join(process.cwd(), logsDir, EXECUTED_PERIODS_FILE);
    const dir = path.dirname(filePath);
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    const now = Math.floor(Date.now() / 1000);
    const minTs = now - EXECUTED_PERIODS_RETENTION_SEC;
    const arr = Array.from(set).filter((t) => t >= minTs).sort((a, b) => a - b);
    fs.writeFileSync(filePath, JSON.stringify(arr, null, 0), 'utf8');
}

function appendDecisionFunnelLog(logsDir: string, row: Record<string, unknown>): void {
    try {
        const filePath = path.join(process.cwd(), logsDir, 'decision_funnel.jsonl');
        const dir = path.dirname(filePath);
        if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
        fs.appendFileSync(filePath, JSON.stringify(row, null, 0) + '\n', 'utf8');
    } catch {
        // best effort only
    }
}

function parsePeriodStartFromSlug(slug: string | null | undefined): number | null {
    if (!slug) return null;
    const m = String(slug).match(/-(\d{8,})$/);
    if (!m) return null;
    const raw = Number(m[1]);
    if (!Number.isFinite(raw) || raw <= 0) return null;
    return raw > 1e12 ? Math.floor(raw / 1000) : raw;
}

function appendSlot01GentleFilterBlockedLog(
    logsDir: string,
    modeScope: string,
    row: Record<string, unknown>,
): void {
    try {
        const safeScope = String(modeScope || 'live').trim().toLowerCase() || 'live';
        const filePath = path.join(process.cwd(), logsDir, `gentle_filter_blocked_opportunities.${safeScope}.jsonl`);
        const dir = path.dirname(filePath);
        if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
        fs.appendFileSync(filePath, JSON.stringify(row, null, 0) + '\n', 'utf8');
    } catch {
        // best effort only
    }
}

function appendSlot02Aux1hBlockedLog(
    logsDir: string,
    modeScope: string,
    row: Record<string, unknown>,
): void {
    try {
        const safeScope = String(modeScope || 'live').trim().toLowerCase() || 'live';
        const filePath = path.join(process.cwd(), logsDir, `slot02_aux_1h_blocked_opportunities.${safeScope}.jsonl`);
        const dir = path.dirname(filePath);
        if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
        fs.appendFileSync(filePath, JSON.stringify(row, null, 0) + '\n', 'utf8');
    } catch {
        // best effort only
    }
}

function appendSlot02Aux1hHandledLog(
    logsDir: string,
    modeScope: string,
    row: Record<string, unknown>,
): void {
    try {
        const safeScope = String(modeScope || 'live').trim().toLowerCase() || 'live';
        const filePath = path.join(process.cwd(), logsDir, `slot02_aux_1h_handled_opportunities.${safeScope}.jsonl`);
        const dir = path.dirname(filePath);
        if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
        fs.appendFileSync(filePath, JSON.stringify(row, null, 0) + '\n', 'utf8');
    } catch {
        // best effort only
    }
}

const SLOT01_LIVE_TRADERS = new Set([
    'v5_exp10_bp_dyn_0300_0440_eth',
    'slot01_base_forcedtrade_eth_sim400',
]);
const SLOT02_LIVE_TRADERS = new Set([
    'slot02_newmethod_btc_sim400',
]);

function isCanonicalLiveSlotConfig(cfg: TraderConfig, slot: 'slot01' | 'slot02'): boolean {
    const name = String(cfg.name || '').trim();
    const runtimeBaseName = String(cfg.runtimeBaseName || '').trim();
    const targets = slot === 'slot01' ? SLOT01_LIVE_TRADERS : SLOT02_LIVE_TRADERS;
    return targets.has(name) || targets.has(runtimeBaseName);
}

function sanitizeCanonicalLiveSlotConfig(cfg: TraderConfig): TraderConfig {
    const sanitized: TraderConfig = { ...cfg };
    const clearLegacySizing = () => {
        sanitized.kellyEnabled = false;
        sanitized.kellyFrac = undefined;
        sanitized.confTier1Bound = undefined;
        sanitized.confTier2Bound = undefined;
        sanitized.tier1Mult = undefined;
        sanitized.tier2Mult = undefined;
        sanitized.tier3Mult = undefined;
        sanitized.tierP1Pct = undefined;
        sanitized.tierP2Pct = undefined;
        sanitized.tierP3Pct = undefined;
        sanitized.tierP4Pct = undefined;
        sanitized.fixedTargetNotionalUsd = undefined;
        sanitized.betSizePercent = undefined;
        sanitized.confidenceScaleBandsBySymbol = {};
        sanitized.betSizingRulesBySymbolDirection = {};
        sanitized.betPctNormalBySymbolDirection = {};
        sanitized.betPctConservativeBySymbolDirection = {};
        sanitized.betTargetCapUsdBySymbolDirection = {};
    };

    if (isCanonicalLiveSlotConfig(sanitized, 'slot01')) {
        clearLegacySizing();
        sanitized.sizingMode = sanitized.sizingMode || 'percent_cap';
        sanitized.capitalRiskCapPct = sanitized.capitalRiskCapPct ?? 1.0;
        sanitized.runtimeRiskScalingMode = sanitized.runtimeRiskScalingMode || 'cap_to_directional_conservative_with_hysteresis';
        sanitized.runtimeRiskScalingEnterDrawdownPct = sanitized.runtimeRiskScalingEnterDrawdownPct ?? 0.20;
        sanitized.runtimeRiskScalingRecoverDrawdownPct = sanitized.runtimeRiskScalingRecoverDrawdownPct ?? 0.18;
        sanitized.requestCompressionMode = sanitized.requestCompressionMode || 'off';
        sanitized.executionCompressionMode = sanitized.executionCompressionMode || 'reallocate_plus_late_completion';
    }

    if (isCanonicalLiveSlotConfig(sanitized, 'slot02')) {
        clearLegacySizing();
        sanitized.sizingMode = sanitized.sizingMode || 'symmetric_dynamic_pct';
        sanitized.betSizePercent = sanitized.betSizePercent ?? 5.0;
        sanitized.bankrollPct = sanitized.bankrollPct ?? 0.05;
        sanitized.betPctNormal = sanitized.betPctNormal ?? 0.05;
        sanitized.betPctConservative = sanitized.betPctConservative ?? 0.05;
        sanitized.lowPriceDynamicMinTotalAmountUsd = sanitized.lowPriceDynamicMinTotalAmountUsd ?? 0.0;
        sanitized.betTargetCapUsd = sanitized.betTargetCapUsd ?? 0.0;
        sanitized.capitalRiskCapPct = 1.0;
        sanitized.runtimeRiskScalingMode = 'off';
        sanitized.requestCompressionMode = 'off';
        sanitized.executionCompressionMode = 'reallocate_plus_late_completion';
    }

    return sanitized;
}

function applyCanonicalLiveEnvOverrides(
    cfg: TraderConfig,
    base: Record<string, string | undefined>,
): Record<string, string | undefined> {
    const out = { ...base };
    const clearKeys = [
        'KELLY_FRAC',
        'FIXED_TARGET_NOTIONAL_USD',
        'BET_SIZE_PERCENT',
        'CAPITAL_RISK_CAP_PCT',
        'CONFIDENCE_SCALE_BANDS_BY_SYMBOL',
        'CONFIDENCE_SCALE_BANDS_BY_SYMBOL_DIRECTION',
        'BET_SIZING_RULES_BY_SYMBOL_DIRECTION',
        'BET_PCT_NORMAL_BY_SYMBOL_DIRECTION',
        'BET_PCT_CONSERVATIVE_BY_SYMBOL_DIRECTION',
        'BET_TARGET_CAP_USD_BY_SYMBOL_DIRECTION',
        'CONF_TIER1_BOUND',
        'CONF_TIER2_BOUND',
        'TIER1_MULT',
        'TIER2_MULT',
        'TIER3_MULT',
        'TIER_P1_PCT',
        'TIER_P2_PCT',
        'TIER_P3_PCT',
        'TIER_P4_PCT',
    ];

    const clearLegacySizingEnv = (preserveConfidenceBands: boolean) => {
        out.KELLY_ENABLED = 'false';
        for (const key of clearKeys) {
            if (preserveConfidenceBands && (key === 'CONFIDENCE_SCALE_BANDS_BY_SYMBOL' || key === 'CONFIDENCE_SCALE_BANDS_BY_SYMBOL_DIRECTION')) continue;
            out[key] = '';
        }
    };

    if (isCanonicalLiveSlotConfig(cfg, 'slot01') || isCanonicalLiveSlotConfig(cfg, 'slot02')) {
        clearLegacySizingEnv(String(cfg.sizingMode || '') === 'symmetric_dynamic_pct');
    }

    return out;
}

function buildEnvRecord(cfg: TraderConfig): Record<string, string | undefined> {
    const baseProbThreshold = cfg.probThreshold ?? 0.55;
    const downDelta = Number(
        cfg.downThresholdDelta
        ?? process.env.DOWN_THRESHOLD_DELTA
        ?? GROUP_DOWN_DELTA_MAP[cfg.group]
        ?? DEFAULT_DOWN_THRESHOLD_DELTA,
    );
    const probUp = cfg.probThresholdUp ?? baseProbThreshold;
    const probDown = cfg.probThresholdDown
        ?? Math.max(0.01, Math.min(0.50, 1 - baseProbThreshold - Math.max(0, downDelta)));

    const base: Record<string, string | undefined> = {
        ...process.env,
        TRADING_MODE: cfg.appliedTradingMode || process.env.TRADING_MODE || 'simulation',
        PREDICTION_SUFFIX: cfg.predictionSuffix,
        PREDICTION_PATH: cfg.predictionPath,
        PREDICTION_FALLBACK_PATH: cfg.predictionFallbackPath,
        LOGS_DIR: cfg.logsDir,
        TIMEFRAME: cfg.timeframe,
        PROB_THRESHOLD: String(baseProbThreshold),
        PROB_THRESHOLD_UP: String(probUp),
        PROB_THRESHOLD_DOWN: String(probDown),
        ALLOWED_MARKETS: cfg.allowedMarkets,
        INITIAL_CAPITAL: String(cfg.initialCapital),
        PER_COIN_CAPITAL: cfg.perCoinCapital ? 'true' : 'false',
        RISK_CONTROL_ENABLED: cfg.riskControlEnabled ? 'true' : 'false',
        SIMULATION_BLOCKING_GUARDS_DISABLED: cfg.simulationBlockingGuardsDisabled ? 'true' : 'false',
    };

    if (cfg.probThresholdBySymbol && Object.keys(cfg.probThresholdBySymbol).length > 0) {
        base.PROB_THRESHOLD_BY_SYMBOL = Object.entries(cfg.probThresholdBySymbol)
            .map(([sym, val]) => `${String(sym).toUpperCase()}:${Number(val)}`)
            .join(',');
    }
    if (cfg.liveSelectedRuntimeOverrideActiveSource) {
        base.LIVE_SELECTED_RUNTIME_OVERRIDE_ACTIVE = 'true';
        base.LIVE_SELECTED_RUNTIME_OVERRIDE_SOURCE = String(cfg.liveSelectedRuntimeOverrideActiveSource);
    }
    if (cfg.liveSelectedRuntimeOverrideActiveScope) {
        base.LIVE_SELECTED_RUNTIME_OVERRIDE_SCOPE = String(cfg.liveSelectedRuntimeOverrideActiveScope);
    }
    if (cfg.confidenceScaleBandsBySymbol && Object.keys(cfg.confidenceScaleBandsBySymbol).length > 0) {
        base.CONFIDENCE_SCALE_BANDS_BY_SYMBOL = JSON.stringify(cfg.confidenceScaleBandsBySymbol);
    }
    if (cfg.confidenceScaleBandsBySymbolDirection && Object.keys(cfg.confidenceScaleBandsBySymbolDirection).length > 0) {
        base.CONFIDENCE_SCALE_BANDS_BY_SYMBOL_DIRECTION = JSON.stringify(cfg.confidenceScaleBandsBySymbolDirection);
    }
    if (cfg.slot01GentleFilterEnabled != null) {
        base.SLOT01_GENTLE_FILTER_ENABLED = cfg.slot01GentleFilterEnabled ? 'true' : 'false';
    }
    if (cfg.slot01GentleFilterConfidenceMin != null) {
        base.SLOT01_GENTLE_FILTER_CONFIDENCE_MIN = String(cfg.slot01GentleFilterConfidenceMin);
    }
    if (cfg.slot01GentleFilterConfidenceMax != null) {
        base.SLOT01_GENTLE_FILTER_CONFIDENCE_MAX = String(cfg.slot01GentleFilterConfidenceMax);
    }
    if (cfg.slot01GentleFilterDirection) {
        base.SLOT01_GENTLE_FILTER_DIRECTION = String(cfg.slot01GentleFilterDirection);
    }
    if (cfg.slot02Aux1hEnabled != null) {
        base.SLOT02_AUX_1H_ENABLED = cfg.slot02Aux1hEnabled ? 'true' : 'false';
    }
    if (cfg.slot02Aux1hPredictionPath) {
        base.SLOT02_AUX_1H_PREDICTION_PATH = String(cfg.slot02Aux1hPredictionPath);
    }
    if (cfg.slot02Aux1hMode) {
        base.SLOT02_AUX_1H_MODE = String(cfg.slot02Aux1hMode);
    }
    if (cfg.slot02Aux1hStrengthThreshold != null) {
        base.SLOT02_AUX_1H_STRENGTH_THRESHOLD = String(cfg.slot02Aux1hStrengthThreshold);
    }
    if (cfg.slot02Aux1hUpStrengthThreshold != null) {
        base.SLOT02_AUX_1H_UP_STRENGTH_THRESHOLD = String(cfg.slot02Aux1hUpStrengthThreshold);
    }
    if (cfg.slot02Aux1hDownStrengthThreshold != null) {
        base.SLOT02_AUX_1H_DOWN_STRENGTH_THRESHOLD = String(cfg.slot02Aux1hDownStrengthThreshold);
    }
    if (cfg.slot02Aux1hMainConfidenceThreshold != null) {
        base.SLOT02_AUX_1H_MAIN_CONFIDENCE_THRESHOLD = String(cfg.slot02Aux1hMainConfidenceThreshold);
    }
    if (cfg.slot02Aux1hWeakSupportThreshold != null) {
        base.SLOT02_AUX_1H_WEAK_SUPPORT_THRESHOLD = String(cfg.slot02Aux1hWeakSupportThreshold);
    }
    if (cfg.slot02Aux1hLowConfidenceThreshold != null) {
        base.SLOT02_AUX_1H_LOW_CONFIDENCE_THRESHOLD = String(cfg.slot02Aux1hLowConfidenceThreshold);
    }
    if (cfg.slot02Aux1hHighConfidenceThreshold != null) {
        base.SLOT02_AUX_1H_HIGH_CONFIDENCE_THRESHOLD = String(cfg.slot02Aux1hHighConfidenceThreshold);
    }
    if (cfg.slot02Aux1hLowVolThreshold != null) {
        base.SLOT02_AUX_1H_LOW_VOL_THRESHOLD = String(cfg.slot02Aux1hLowVolThreshold);
    }
    if (cfg.slot02Aux1hHighVolThreshold != null) {
        base.SLOT02_AUX_1H_HIGH_VOL_THRESHOLD = String(cfg.slot02Aux1hHighVolThreshold);
    }
    if (cfg.slot02Aux1hWeakSupportMinVolBucket != null) {
        base.SLOT02_AUX_1H_WEAK_SUPPORT_MIN_VOL_BUCKET = String(cfg.slot02Aux1hWeakSupportMinVolBucket);
    }
    if (cfg.slot02Aux1hReverseActions && cfg.slot02Aux1hReverseActions.length > 0) {
        base.SLOT02_AUX_1H_REVERSE_ACTIONS = JSON.stringify(cfg.slot02Aux1hReverseActions);
    }
    if (cfg.slot02Aux1hWeakSupportActions && cfg.slot02Aux1hWeakSupportActions.length > 0) {
        base.SLOT02_AUX_1H_WEAK_SUPPORT_ACTIONS = JSON.stringify(cfg.slot02Aux1hWeakSupportActions);
    }
    if (cfg.slot02Aux1hFreshnessSec != null) {
        base.SLOT02_AUX_1H_FRESHNESS_SEC = String(cfg.slot02Aux1hFreshnessSec);
    }
    if (cfg.betSizingRulesBySymbolDirection && Object.keys(cfg.betSizingRulesBySymbolDirection).length > 0) {
        base.BET_SIZING_RULES_BY_SYMBOL_DIRECTION = JSON.stringify(cfg.betSizingRulesBySymbolDirection);
    }
    if (cfg.betPctNormalBySymbolDirection && Object.keys(cfg.betPctNormalBySymbolDirection).length > 0) {
        base.BET_PCT_NORMAL_BY_SYMBOL_DIRECTION = JSON.stringify(cfg.betPctNormalBySymbolDirection);
    }
    if (cfg.betPctConservativeBySymbolDirection && Object.keys(cfg.betPctConservativeBySymbolDirection).length > 0) {
        base.BET_PCT_CONSERVATIVE_BY_SYMBOL_DIRECTION = JSON.stringify(cfg.betPctConservativeBySymbolDirection);
    }
    if (cfg.betTargetCapUsdBySymbolDirection && Object.keys(cfg.betTargetCapUsdBySymbolDirection).length > 0) {
        base.BET_TARGET_CAP_USD_BY_SYMBOL_DIRECTION = JSON.stringify(cfg.betTargetCapUsdBySymbolDirection);
    }
    if (cfg.lowPriceSelectedBuyPriceRangeBySymbolDirection && Object.keys(cfg.lowPriceSelectedBuyPriceRangeBySymbolDirection).length > 0) {
        base.LOWPRICE_SELECTED_BUY_PRICE_RANGE_BY_SYMBOL_DIRECTION = JSON.stringify(cfg.lowPriceSelectedBuyPriceRangeBySymbolDirection);
    }
    if (cfg.lowPriceRungCountBySymbolDirection && Object.keys(cfg.lowPriceRungCountBySymbolDirection).length > 0) {
        base.LOWPRICE_RUNG_COUNT_BY_SYMBOL_DIRECTION = JSON.stringify(cfg.lowPriceRungCountBySymbolDirection);
    }

    if (cfg.limitPrice != null) base.LIMIT_PRICE = String(cfg.limitPrice);
    if (cfg.limitPriceLadder) base.LIMIT_PRICE_LADDER = cfg.limitPriceLadder;
    if (cfg.lowpriceFamilyId) base.LOWPRICE_FAMILY_ID = cfg.lowpriceFamilyId;
    if (cfg.familyRuleVersion) base.FAMILY_RULE_VERSION = cfg.familyRuleVersion;
    if (cfg.staleOrderPolicy) base.STALE_ORDER_POLICY = cfg.staleOrderPolicy;
    if (cfg.staleCancelMinAgeSec != null) base.STALE_CANCEL_MIN_AGE_SEC = String(cfg.staleCancelMinAgeSec);
    if (cfg.staleCancelMinDriftTicks != null) base.STALE_CANCEL_MIN_DRIFT_TICKS = String(cfg.staleCancelMinDriftTicks);
    if (cfg.staleCancelMinTimeToExpirySec != null) base.STALE_CANCEL_MIN_TIME_TO_EXPIRY_SEC = String(cfg.staleCancelMinTimeToExpirySec);
    if (cfg.lateFillCancelMinutes != null) base.LATE_FILL_CANCEL_MINUTES = String(cfg.lateFillCancelMinutes);
    if (cfg.maxTokenPrice != null) base.MAX_TOKEN_PRICE = String(cfg.maxTokenPrice);
    if (cfg.edgeFilterEnabled) base.EDGE_FILTER_ENABLED = 'true';
    if (cfg.kellyEnabled) base.KELLY_ENABLED = 'true';
    if (cfg.betSizePercent != null) base.BET_SIZE_PERCENT = String(cfg.betSizePercent);
    if (cfg.cooldownAfterLosses != null) base.COOLDOWN_AFTER_LOSSES = String(cfg.cooldownAfterLosses);
    if (cfg.cooldownBars != null) base.COOLDOWN_BARS = String(cfg.cooldownBars);
    if (cfg.cooldownScope) base.COOLDOWN_SCOPE = cfg.cooldownScope;
    if (cfg.numTradingAssets != null) base.NUM_TRADING_ASSETS = String(cfg.numTradingAssets);
    if (cfg.minEdge != null) base.MIN_EDGE = String(cfg.minEdge);
    if (cfg.minEdgeUp != null) base.MIN_EDGE_UP = String(cfg.minEdgeUp);
    if (cfg.minEdgeDown != null) base.MIN_EDGE_DOWN = String(cfg.minEdgeDown);
    if (cfg.consensusThresholdUp != null) base.CONSENSUS_THRESHOLD_UP = String(cfg.consensusThresholdUp);
    if (cfg.consensusThresholdDown != null) base.CONSENSUS_THRESHOLD_DOWN = String(cfg.consensusThresholdDown);
    if (cfg.kellyFrac != null) base.KELLY_FRAC = String(cfg.kellyFrac);
    if (cfg.bankrollPct != null) base.BANKROLL_PCT = String(cfg.bankrollPct);
    if (cfg.betPctNormal != null) base.BET_PCT_NORMAL = String(cfg.betPctNormal);
    if (cfg.betPctConservative != null) base.BET_PCT_CONSERVATIVE = String(cfg.betPctConservative);
    if (cfg.capitalRiskCapPct != null) base.CAPITAL_RISK_CAP_PCT = String(cfg.capitalRiskCapPct);
    if (cfg.betTargetCapUsd != null) base.BET_TARGET_CAP_USD = String(cfg.betTargetCapUsd);
    if (cfg.runtimeRiskScalingMode) base.RUNTIME_RISK_SCALING_MODE = String(cfg.runtimeRiskScalingMode);
    if (cfg.runtimeRiskScalingEnterDrawdownPct != null) {
        base.RUNTIME_RISK_SCALING_ENTER_DRAWDOWN_PCT = String(cfg.runtimeRiskScalingEnterDrawdownPct);
    }
    if (cfg.runtimeRiskScalingRecoverDrawdownPct != null) {
        base.RUNTIME_RISK_SCALING_RECOVER_DRAWDOWN_PCT = String(cfg.runtimeRiskScalingRecoverDrawdownPct);
    }
    if (cfg.sizingMode) base.SIZING_MODE = String(cfg.sizingMode);
    if (cfg.fixedTargetNotionalUsd != null) base.FIXED_TARGET_NOTIONAL_USD = String(cfg.fixedTargetNotionalUsd);
    if (cfg.requestCompressionMode) base.REQUEST_COMPRESSION_MODE = String(cfg.requestCompressionMode);
    if (cfg.executionCompressionMode) base.EXECUTION_COMPRESSION_MODE = String(cfg.executionCompressionMode);
    if (cfg.executionProfileVersion) base.EXECUTION_PROFILE_VERSION = cfg.executionProfileVersion;
    if (cfg.executionV2ShadowOnly != null) base.EXECUTION_V2_SHADOW_ONLY = cfg.executionV2ShadowOnly ? 'true' : 'false';
    if (cfg.chaseEnabled != null) base.CHASE_ENABLED = cfg.chaseEnabled ? 'true' : 'false';
    if (cfg.chaseRounds != null) base.CHASE_ROUNDS = String(cfg.chaseRounds);
    if (cfg.chaseWaitSec != null) base.CHASE_WAIT_SEC = String(cfg.chaseWaitSec);
    if (cfg.firstChaseWaitSec != null) base.FIRST_CHASE_WAIT_SEC = String(cfg.firstChaseWaitSec);
    if (cfg.nextChaseGapSec != null) base.NEXT_CHASE_GAP_SEC = String(cfg.nextChaseGapSec);
    if (cfg.chaseTickLift != null) base.CHASE_TICK_LIFT = String(cfg.chaseTickLift);
    if (cfg.chaseAbsoluteMaxPrice != null) base.CHASE_ABSOLUTE_MAX_PRICE = String(cfg.chaseAbsoluteMaxPrice);
    if (cfg.chaseMinConfidenceBand != null) base.CHASE_MIN_CONFIDENCE_BAND = String(cfg.chaseMinConfidenceBand);
    if (cfg.chaseMinTimeToExpirySec != null) base.CHASE_MIN_TIME_TO_EXPIRY_SEC = String(cfg.chaseMinTimeToExpirySec);
    if (cfg.directionContinuationTicks != null) base.DIRECTION_CONTINUATION_TICKS = String(cfg.directionContinuationTicks);
    if (cfg.directionContinuationWindowSec != null) base.DIRECTION_CONTINUATION_WINDOW_SEC = String(cfg.directionContinuationWindowSec);
    if (cfg.finalAggressiveWindowSec != null) base.FINAL_AGGRESSIVE_WINDOW_SEC = String(cfg.finalAggressiveWindowSec);
    if (cfg.finalAggressiveTickLift != null) base.FINAL_AGGRESSIVE_TICK_LIFT = String(cfg.finalAggressiveTickLift);
    if (cfg.finalAggressiveMaxPrice != null) base.FINAL_AGGRESSIVE_MAX_PRICE = String(cfg.finalAggressiveMaxPrice);
    if (cfg.cancelOnAdverseMoveTicks != null) base.CANCEL_ON_ADVERSE_MOVE_TICKS = String(cfg.cancelOnAdverseMoveTicks);
    if (cfg.cancelOnSignalInvalidation != null) base.CANCEL_ON_SIGNAL_INVALIDATION = cfg.cancelOnSignalInvalidation ? 'true' : 'false';
    if (cfg.cancelOnQueueStall != null) base.CANCEL_ON_QUEUE_STALL = cfg.cancelOnQueueStall ? 'true' : 'false';
    if (cfg.queueStallWindowSec != null) base.QUEUE_STALL_WINDOW_SEC = String(cfg.queueStallWindowSec);
    if (cfg.confTier1Bound != null) base.CONF_TIER1_BOUND = String(cfg.confTier1Bound);
    if (cfg.confTier2Bound != null) base.CONF_TIER2_BOUND = String(cfg.confTier2Bound);
    if (cfg.tier1Mult != null) base.TIER1_MULT = String(cfg.tier1Mult);
    if (cfg.tier2Mult != null) base.TIER2_MULT = String(cfg.tier2Mult);
    if (cfg.tier3Mult != null) base.TIER3_MULT = String(cfg.tier3Mult);
    if (cfg.drawdownHalt != null) base.DRAWDOWN_HALT = String(cfg.drawdownHalt);
    if (cfg.tierP1Pct != null) base.TIER_P1_PCT = String(cfg.tierP1Pct);
    if (cfg.tierP2Pct != null) base.TIER_P2_PCT = String(cfg.tierP2Pct);
    if (cfg.tierP3Pct != null) base.TIER_P3_PCT = String(cfg.tierP3Pct);
    if (cfg.tierP4Pct != null) base.TIER_P4_PCT = String(cfg.tierP4Pct);

    // Slot env should only override when the slot value really exists.
    // Otherwise keep the already loaded base env and do not blank live creds.
    const applySlotOverride = (targetKey: string, envName?: string) => {
        if (!envName) return;
        const value = process.env[envName];
        if (value != null && String(value).trim() !== '') {
            base[targetKey] = value;
        }
    };
    applySlotOverride('PRIVATE_KEY', cfg.privateKeyEnv);
    applySlotOverride('PROXY_WALLET', cfg.proxyWalletEnv);
    applySlotOverride('POLY_API_KEY', cfg.polyApiKeyEnv);
    applySlotOverride('POLY_API_SECRET', cfg.polyApiSecretEnv);
    applySlotOverride('POLY_API_PASSPHRASE', cfg.polyApiPassphraseEnv);
    applySlotOverride('BUILDER_API_KEY', cfg.builderApiKeyEnv);
    applySlotOverride('BUILDER_API_SECRET', cfg.builderApiSecretEnv);
    applySlotOverride('BUILDER_API_PASSPHRASE', cfg.builderApiPassphraseEnv);
    applySlotOverride('RELAYER_API_KEY', (cfg as any).relayerApiKeyEnv);
    applySlotOverride('RELAYER_API_KEY_ADDRESS', (cfg as any).relayerApiKeyAddressEnv);

    return applyCanonicalLiveEnvOverrides(cfg, base);
}

function getPredictionFilePath(suffix: string): string {
    return path.resolve(__dirname, '..', `predictions${suffix}.json`);
}

function resolveConfiguredPredictionPath(cfg: TraderConfig): string {
    const explicit = String(cfg.predictionPath || '').trim();
    if (explicit) return explicit;
    return getPredictionFilePath(cfg.predictionSuffix);
}

function resolveConfiguredPredictionFallbackPath(cfg: TraderConfig, primaryPath: string): string | undefined {
    const explicit = String(cfg.predictionFallbackPath || '').trim();
    if (explicit) return explicit;
    if (cfg.predictionFallbackSuffix) {
        const fallback = getPredictionFilePath(cfg.predictionFallbackSuffix);
        return fallback === primaryPath ? undefined : fallback;
    }
    return undefined;
}

const SYMBOL_OVERRIDE_STALE_SEC = Math.max(
    60,
    Number.isFinite(Number(process.env.SYMBOL_OVERRIDE_STALE_SEC))
        ? Number(process.env.SYMBOL_OVERRIDE_STALE_SEC)
        : 1800,
);

function resolvePredictionReadTarget(
    primaryPath: string,
    fallbackPath?: string,
) : { filePath: string | null; decision: string } {
    const primaryMtimeMs = statFileMtimeMs(primaryPath);
    const primaryAgeSec = primaryMtimeMs > 0 ? (Date.now() - primaryMtimeMs) / 1000 : Number.POSITIVE_INFINITY;
    if (!fallbackPath || fallbackPath === primaryPath) {
        if (primaryAgeSec <= SYMBOL_OVERRIDE_STALE_SEC) {
            return { filePath: primaryPath, decision: 'primary_only' };
        }
        return { filePath: null, decision: primaryMtimeMs > 0 ? 'primary_only_stale' : 'primary_only_missing' };
    }
    if (primaryMtimeMs > 0 && primaryAgeSec <= SYMBOL_OVERRIDE_STALE_SEC) {
        return { filePath: primaryPath, decision: 'primary_fresh' };
    }
    const fallbackMtimeMs = statFileMtimeMs(fallbackPath);
    if (fallbackMtimeMs > 0) {
        const fallbackAgeSec = (Date.now() - fallbackMtimeMs) / 1000;
        if (fallbackAgeSec <= SYMBOL_OVERRIDE_STALE_SEC) {
            return {
                filePath: fallbackPath,
                decision: primaryMtimeMs > 0 ? 'fallback_primary_stale' : 'fallback_primary_missing',
            };
        }
    }
    return {
        filePath: null,
        decision: 'both_stale_or_missing',
    };
}

function deriveCore70AliasBasePredictionPath(primaryPath: string): string | null {
    const base = path.basename(primaryPath);
    const match = base.match(/^predictions_core10_(.+)_70\.json$/i);
    if (!match) return null;
    return path.join(path.dirname(primaryPath), `predictions_${match[1]}.json`);
}

function primaryPathLooksLikeAlias(primaryPath: string, traderName: string): boolean {
    const base = path.basename(primaryPath).toLowerCase();
    const trader = String(traderName || '').trim().toLowerCase();
    return !!trader && base.includes(trader);
}

function hasPredictionPayloadEntries(raw: any): boolean {
    const predictions = raw?.data?.predictions;
    return !!predictions && typeof predictions === 'object' && Object.keys(predictions).length > 0;
}

function predictionPayloadTargetTs(raw: any): number {
    const value = Number(raw?.target_period_end_ts ?? raw?.data?.target_period_end_ts ?? 0);
    return Number.isFinite(value) ? value : 0;
}

function predictionPayloadTimestampMs(raw: any): number {
    const text = String(raw?.data?.timestamp || raw?.timestamp || '').trim();
    if (!text) return 0;
    const value = Date.parse(text);
    return Number.isFinite(value) ? value : 0;
}

function isCanonicalFilterEmptyFresh(raw: any): boolean {
    const predictionCount = Number(
        raw?.data?.predictionCount
        ?? raw?.data?.source_prediction_count
        ?? Object.keys(raw?.data?.predictions || {}).length,
    );
    return Boolean(
        raw?.data
        && raw.data.source_stale !== true
        && raw.data.filtered_empty_due_to_threshold === true
        && predictionCount >= 0,
    );
}

function shouldFallbackFromCanonicalAliasToBase(primaryRaw: any, aliasBaseRaw: any): boolean {
    if (isCanonicalFilterEmptyFresh(primaryRaw)) {
        return false;
    }
    if (!aliasBaseRaw?.data || aliasBaseRaw.data.source_stale === true || !hasPredictionPayloadEntries(aliasBaseRaw)) {
        return false;
    }
    if (primaryRaw?.data?.source_stale === true) {
        return true;
    }
    const primaryTargetTs = predictionPayloadTargetTs(primaryRaw);
    const aliasBaseTargetTs = predictionPayloadTargetTs(aliasBaseRaw);
    if (aliasBaseTargetTs > 0 && (primaryTargetTs <= 0 || aliasBaseTargetTs > primaryTargetTs)) {
        return true;
    }
    const primaryTimestampMs = predictionPayloadTimestampMs(primaryRaw);
    const aliasBaseTimestampMs = predictionPayloadTimestampMs(aliasBaseRaw);
    return aliasBaseTimestampMs > 0 && primaryTimestampMs > 0 && aliasBaseTimestampMs > primaryTimestampMs + 1000;
}

function roundProb(x: number): number {
    return Math.round(x * 10000) / 10000;
}

function normalizeBaseSymbol(symbol: string): string {
    return String(symbol || '').toUpperCase().replace(/\/.*$/, '');
}

function evaluateSlot01GentleFilterAtDecisionLayer(
    cfg: TraderConfig,
    env: PredictionEnvConfig,
    symbol: string,
    direction: 'UP' | 'DOWN',
    confidence: number,
): {
    blocked: boolean;
    roundedConfidence: number;
    confidenceMin: number;
    confidenceMax: number;
    directionMode: string;
} {
    const roundedConfidence = roundProb(Number(confidence));
    const confidenceMin = roundProb(Number(
        cfg.slot01GentleFilterConfidenceMin
        ?? env.SLOT01_GENTLE_FILTER_CONFIDENCE_MIN
        ?? 0,
    ));
    const confidenceMax = roundProb(Number(
        cfg.slot01GentleFilterConfidenceMax
        ?? env.SLOT01_GENTLE_FILTER_CONFIDENCE_MAX
        ?? 0,
    ));
    const directionMode = String(
        cfg.slot01GentleFilterDirection
        ?? env.SLOT01_GENTLE_FILTER_DIRECTION
        ?? 'all',
    ).trim().toLowerCase() || 'all';
    const enabled = Boolean(
        cfg.slot01GentleFilterEnabled
        ?? env.SLOT01_GENTLE_FILTER_ENABLED,
    );
    if (!enabled) return { blocked: false, roundedConfidence, confidenceMin, confidenceMax, directionMode };
    if (String(cfg.name || '') !== 'slot01_base_forcedtrade_eth_sim400') {
        return { blocked: false, roundedConfidence, confidenceMin, confidenceMax, directionMode };
    }
    if (normalizeBaseSymbol(symbol) !== 'ETH') {
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

function evaluateSlot02Aux1hAtDecisionLayer(
    cfg: TraderConfig,
    env: PredictionEnvConfig,
    symbol: string,
    direction: 'UP' | 'DOWN',
    mainConfidence: number | null,
): {
    enabled: boolean;
    blocked: boolean;
    reason: string | null;
    mode: string | null;
    auxDirection: 'UP' | 'DOWN' | 'NEUTRAL' | null;
    confidence: number | null;
    strength: number | null;
    freshnessSec: number | null;
    modelVersion: string | null;
    upStrengthThreshold: number | null;
    downStrengthThreshold: number | null;
    mainConfidenceThreshold: number | null;
    mainConfidenceBelowThreshold: boolean | null;
    actionCode: number | null;
    softCapPct: number | null;
    weakSupportActive: boolean | null;
    mainConfidenceBucket: number | null;
    volValue: number | null;
    volBucket: number | null;
} {
    const envCfg = env as PredictionEnvConfig & {
        SLOT02_AUX_1H_ENABLED?: boolean | string;
        SLOT02_AUX_1H_PREDICTION_PATH?: string;
        SLOT02_AUX_1H_MODE?: string;
        SLOT02_AUX_1H_STRENGTH_THRESHOLD?: number | string;
        SLOT02_AUX_1H_UP_STRENGTH_THRESHOLD?: number | string;
        SLOT02_AUX_1H_DOWN_STRENGTH_THRESHOLD?: number | string;
        SLOT02_AUX_1H_MAIN_CONFIDENCE_THRESHOLD?: number | string;
        SLOT02_AUX_1H_FRESHNESS_SEC?: number | string;
        SLOT02_AUX_1H_WEAK_SUPPORT_THRESHOLD?: number | string;
        SLOT02_AUX_1H_LOW_CONFIDENCE_THRESHOLD?: number | string;
        SLOT02_AUX_1H_HIGH_CONFIDENCE_THRESHOLD?: number | string;
        SLOT02_AUX_1H_LOW_VOL_THRESHOLD?: number | string;
        SLOT02_AUX_1H_HIGH_VOL_THRESHOLD?: number | string;
        SLOT02_AUX_1H_WEAK_SUPPORT_MIN_VOL_BUCKET?: number | string;
        SLOT02_AUX_1H_REVERSE_ACTIONS?: string;
        SLOT02_AUX_1H_WEAK_SUPPORT_ACTIONS?: string;
    };
    const enabled = Boolean(cfg.slot02Aux1hEnabled ?? envCfg.SLOT02_AUX_1H_ENABLED);
    const resolvedMode = String(cfg.slot02Aux1hMode ?? envCfg.SLOT02_AUX_1H_MODE ?? 'hard_veto').trim().toLowerCase() || 'hard_veto';
    const emptyResponse = {
        enabled: false,
        blocked: false,
        reason: null,
        mode: null,
        auxDirection: null,
        confidence: null,
        strength: null,
        freshnessSec: null,
        modelVersion: null,
        upStrengthThreshold: null,
        downStrengthThreshold: null,
        mainConfidenceThreshold: null,
        mainConfidenceBelowThreshold: null,
        actionCode: null,
        softCapPct: null,
        weakSupportActive: null,
        mainConfidenceBucket: null,
        volValue: null,
        volBucket: null,
    };
    if (!enabled) {
        return emptyResponse;
    }
    if (String(cfg.name || '') !== 'slot02_newmethod_btc_sim400' || normalizeBaseSymbol(symbol) !== 'BTC') {
        return emptyResponse;
    }
    const mode = resolvedMode;
    if (mode !== 'hard_veto' && mode !== 'hard_veto_with_confidence' && mode !== 'support_quality_v1') {
        return { ...emptyResponse, enabled: true, blocked: true, reason: 'slot02_aux_1h_mode_unsupported' };
    }
    const predictionPath = String(cfg.slot02Aux1hPredictionPath ?? envCfg.SLOT02_AUX_1H_PREDICTION_PATH ?? '').trim();
    if (!predictionPath) {
        return { ...emptyResponse, enabled: true, blocked: true, reason: 'slot02_aux_1h_signal_unavailable' };
    }
    const payload = readJsonFileSafe(predictionPath);
    const pred = payload?.predictions?.BTC_USDT_aux_1h;
    if (!pred || typeof pred !== 'object') {
        return { ...emptyResponse, enabled: true, blocked: true, reason: 'slot02_aux_1h_signal_unavailable' };
    }
    const freshnessLimitSec = Math.max(60, Number(cfg.slot02Aux1hFreshnessSec ?? envCfg.SLOT02_AUX_1H_FRESHNESS_SEC ?? 5400));
    const tsRaw = payload?.generatedAt || pred?.timestamp || pred?.details?.feature_timestamp || null;
    const tsMs = tsRaw ? Date.parse(String(tsRaw)) : Number.NaN;
    const freshnessSec = Number.isFinite(tsMs) ? Math.max(0, (Date.now() - tsMs) / 1000) : null;
    if (freshnessSec == null || freshnessSec > freshnessLimitSec) {
        return {
            ...emptyResponse,
            enabled: true,
            blocked: true,
            reason: 'slot02_aux_1h_signal_stale',
            confidence: Number.isFinite(Number(pred?.confidence)) ? Number(pred.confidence) : null,
            strength: Number.isFinite(Number(pred?.details?.strength)) ? Number(pred.details.strength) : null,
            freshnessSec,
            modelVersion: String(payload?.modelVersion || pred?.details?.modelVersion || pred?.modelVersion || ''),
        };
    }
    const auxDirectionRaw = String(pred?.direction || '').trim().toUpperCase();
    const auxDirection = auxDirectionRaw === 'UP' || auxDirectionRaw === 'DOWN' || auxDirectionRaw === 'NEUTRAL'
        ? auxDirectionRaw as 'UP' | 'DOWN' | 'NEUTRAL'
        : 'NEUTRAL';
    const confidence = Number.isFinite(Number(pred?.confidence)) ? Number(pred.confidence) : null;
    const strength = Number.isFinite(Number(pred?.details?.strength))
        ? Number(pred.details.strength)
        : (
            Number.isFinite(Number(pred?.details?.upProb)) && Number.isFinite(Number(pred?.details?.downProb))
                ? Math.abs(Number(pred.details.upProb) - Number(pred.details.downProb))
                : null
        );
    const opposite = auxDirection !== 'NEUTRAL' && auxDirection !== direction;
    const defaultStrengthThreshold = Math.max(0, Number(cfg.slot02Aux1hStrengthThreshold ?? envCfg.SLOT02_AUX_1H_STRENGTH_THRESHOLD ?? 0));
    const upStrengthThreshold = Math.max(
        0,
        Number(
            cfg.slot02Aux1hUpStrengthThreshold
            ?? envCfg.SLOT02_AUX_1H_UP_STRENGTH_THRESHOLD
            ?? defaultStrengthThreshold,
        ),
    );
    const downStrengthThreshold = Math.max(
        0,
        Number(
            cfg.slot02Aux1hDownStrengthThreshold
            ?? envCfg.SLOT02_AUX_1H_DOWN_STRENGTH_THRESHOLD
            ?? defaultStrengthThreshold,
        ),
    );
    const requiredStrengthThreshold = direction === 'UP' ? upStrengthThreshold : downStrengthThreshold;
    const mainConfidenceThreshold = Number(
        cfg.slot02Aux1hMainConfidenceThreshold
        ?? envCfg.SLOT02_AUX_1H_MAIN_CONFIDENCE_THRESHOLD
        ?? Number.NaN,
    );
    const mainConfidenceBelowThreshold = Number.isFinite(Number(mainConfidence))
        && Number.isFinite(mainConfidenceThreshold)
        && Number(mainConfidence) < mainConfidenceThreshold;
    const strengthPassed = Number(strength ?? 0) >= requiredStrengthThreshold;
    let blocked = false;
    let reason: string | null = null;
    let actionCode: number | null = null;
    let softCapPct: number | null = null;
    let weakSupportActive: boolean | null = null;
    let mainConfidenceBucket: number | null = null;
    const volValue = Number.isFinite(Number(pred?.details?.h1VolZ)) ? Number(pred.details.h1VolZ) : null;
    let volBucket: number | null = null;
    if (mode === 'support_quality_v1') {
        if (!Number.isFinite(Number(mainConfidence))) {
            blocked = true;
            reason = 'slot02_aux_1h_main_confidence_unavailable';
        } else {
            const weakSupportThreshold = Math.max(
                0,
                Number(
                    cfg.slot02Aux1hWeakSupportThreshold
                    ?? envCfg.SLOT02_AUX_1H_WEAK_SUPPORT_THRESHOLD
                    ?? Number.NaN,
                ),
            );
            const lowConfidenceThreshold = Number(
                cfg.slot02Aux1hLowConfidenceThreshold
                ?? envCfg.SLOT02_AUX_1H_LOW_CONFIDENCE_THRESHOLD
                ?? Number.NaN,
            );
            const highConfidenceThreshold = Number(
                cfg.slot02Aux1hHighConfidenceThreshold
                ?? envCfg.SLOT02_AUX_1H_HIGH_CONFIDENCE_THRESHOLD
                ?? Number.NaN,
            );
            const lowVolThreshold = Number(
                cfg.slot02Aux1hLowVolThreshold
                ?? envCfg.SLOT02_AUX_1H_LOW_VOL_THRESHOLD
                ?? Number.NaN,
            );
            const highVolThreshold = Number(
                cfg.slot02Aux1hHighVolThreshold
                ?? envCfg.SLOT02_AUX_1H_HIGH_VOL_THRESHOLD
                ?? Number.NaN,
            );
            const weakSupportMinVolBucket = Math.max(
                0,
                Math.round(
                    Number(
                        cfg.slot02Aux1hWeakSupportMinVolBucket
                        ?? envCfg.SLOT02_AUX_1H_WEAK_SUPPORT_MIN_VOL_BUCKET
                        ?? 2,
                    ),
                ),
            );
            const parseActionTuple = (value: unknown, fallback: [number, number, number]): [number, number, number] => {
                const raw = Array.isArray(value)
                    ? value
                    : (typeof value === 'string' && value.trim()
                        ? (() => {
                            try {
                                const parsed = JSON.parse(value);
                                return Array.isArray(parsed) ? parsed : [];
                            } catch {
                                return [];
                            }
                        })()
                        : []);
                const nums = raw.map((item) => Math.trunc(Number(item))).filter((item) => Number.isFinite(item));
                return nums.length >= 3 ? [nums[0], nums[1], nums[2]] : fallback;
            };
            const reverseActions = parseActionTuple(
                cfg.slot02Aux1hReverseActions ?? envCfg.SLOT02_AUX_1H_REVERSE_ACTIONS,
                [-1, 4, 0],
            );
            const weakSupportActions = parseActionTuple(
                cfg.slot02Aux1hWeakSupportActions ?? envCfg.SLOT02_AUX_1H_WEAK_SUPPORT_ACTIONS,
                [4, 0, 0],
            );
            const mainConfNum = Number(mainConfidence);
            mainConfidenceBucket = Number.isFinite(lowConfidenceThreshold) && Number.isFinite(highConfidenceThreshold)
                ? (mainConfNum < lowConfidenceThreshold ? 0 : (mainConfNum < highConfidenceThreshold ? 1 : 2))
                : null;
            if (volValue != null && Number.isFinite(volValue) && Number.isFinite(lowVolThreshold) && Number.isFinite(highVolThreshold)) {
                volBucket = volValue < lowVolThreshold ? 0 : (volValue < highVolThreshold ? 1 : 2);
            }
            const reverseOppose = opposite && strengthPassed;
            weakSupportActive = (
                !reverseOppose
                && volBucket != null
                && volBucket >= weakSupportMinVolBucket
                && (
                    auxDirection === 'NEUTRAL'
                    || (auxDirection === direction && Number(strength ?? 0) < weakSupportThreshold)
                    || (
                        auxDirection !== direction
                        && Number(strength ?? 0) < requiredStrengthThreshold
                    )
                )
            );
            if (mainConfidenceBucket != null) {
                if (reverseOppose) {
                    actionCode = reverseActions[mainConfidenceBucket] ?? 0;
                } else if (weakSupportActive) {
                    actionCode = weakSupportActions[mainConfidenceBucket] ?? 0;
                }
            }
            if (actionCode === -1) {
                blocked = true;
                reason = 'slot02_aux_1h_support_quality_veto_blocked';
            } else if (actionCode === 4) {
                softCapPct = 0.04;
                reason = 'slot02_aux_1h_support_quality_soft_to_4';
            } else if (actionCode === 3) {
                softCapPct = 0.03;
                reason = 'slot02_aux_1h_support_quality_soft_to_3';
            }
        }
    } else if (mode === 'hard_veto_with_confidence') {
        if (!Number.isFinite(Number(mainConfidence))) {
            blocked = true;
            reason = 'slot02_aux_1h_main_confidence_unavailable';
        } else {
            blocked = opposite && strengthPassed && mainConfidenceBelowThreshold;
            reason = blocked ? 'slot02_aux_1h_confidence_veto_blocked' : null;
        }
    } else {
        blocked = opposite && strengthPassed;
        reason = blocked ? 'slot02_aux_1h_veto_blocked' : null;
    }
    return {
        enabled: true,
        blocked,
        reason,
        mode,
        auxDirection,
        confidence,
        strength,
        freshnessSec,
        modelVersion: String(payload?.modelVersion || pred?.details?.modelVersion || pred?.modelVersion || ''),
        upStrengthThreshold,
        downStrengthThreshold,
        mainConfidenceThreshold: Number.isFinite(mainConfidenceThreshold) ? mainConfidenceThreshold : null,
        mainConfidenceBelowThreshold: Number.isFinite(Number(mainConfidence)) && Number.isFinite(mainConfidenceThreshold)
            ? mainConfidenceBelowThreshold
            : null,
        actionCode,
        softCapPct,
        weakSupportActive,
        mainConfidenceBucket,
        volValue,
        volBucket,
    };
}

function getBaseProbThreshold(p: PredictionResult, env: PredictionEnvConfig): number {
    const bySymbol = (env as PredictionEnvConfig & { PROB_THRESHOLD_BY_SYMBOL?: Record<string, number> }).PROB_THRESHOLD_BY_SYMBOL;
    const sym = normalizeBaseSymbol(p.symbol);
    const fromMap = bySymbol && Number.isFinite(Number(bySymbol[sym])) ? Number(bySymbol[sym]) : Number.NaN;
    return Number.isFinite(fromMap) ? fromMap : env.PROB_THRESHOLD;
}

function getDirectionThreshold(p: PredictionResult, env: PredictionEnvConfig): number {
    const baseProbThreshold = getBaseProbThreshold(p, env);
    if (p.direction === 'UP') {
        return env.PROB_THRESHOLD_UP ?? baseProbThreshold;
    }
    const downBound = env.PROB_THRESHOLD_DOWN != null
        ? (1 - env.PROB_THRESHOLD_DOWN)
        : baseProbThreshold;
    return downBound;
}

function thresholdMetWithQuality(
    p: PredictionResult,
    env: PredictionEnvConfig,
    confidencePenalty: number,
    extraDelta = 0,
): boolean {
    if (!p.direction) return false;
    const base = getDirectionThreshold(p, env);
    const required = Math.min(0.999, Math.max(0, base + confidencePenalty + extraDelta));
    return roundProb(effectiveConfidence(p)) >= roundProb(required);
}

function qualityModeSeverity(mode: GuardMode): number {
    if (mode === 'halt') return 3;
    if (mode === 'hard_degraded') return 2;
    if (mode === 'soft_degraded') return 1;
    return 0;
}

function classifyQualitySource(source: string): 'key' | 'secondary' | 'non_key' {
    const src = String(source || '').trim().toLowerCase();
    if ([
        'ob_realtime',
        'bybit_orderbook',
        'polymarket_clob_book',
        'polymarket_clob_history',
        'polymarket_prob_target',
    ].includes(src)) {
        return 'key';
    }
    if (['polymarket_gamma', 'polymarket_prob'].includes(src)) {
        return 'secondary';
    }
    return 'non_key';
}

function apiRateSoftThreshold(sourceClass: 'key' | 'secondary' | 'non_key'): number {
    if (sourceClass === 'key') return DATA_QUALITY_KEY_API_ERROR_SOFT;
    if (sourceClass === 'secondary') return DATA_QUALITY_SECONDARY_API_ERROR_SOFT;
    return DATA_QUALITY_NON_KEY_API_ERROR_SOFT;
}

function apiRateHardThreshold(sourceClass: 'key' | 'secondary' | 'non_key'): number {
    if (sourceClass === 'key') return DATA_QUALITY_KEY_API_ERROR_HARD;
    if (sourceClass === 'secondary') return DATA_QUALITY_SECONDARY_API_ERROR_HARD;
    return DATA_QUALITY_NON_KEY_API_ERROR_HARD;
}

function latestMtimeSec(paths: string[]): number | null {
    let best = 0;
    for (const p of paths) {
        try {
            const st = fs.statSync(p);
            const ts = st.mtimeMs / 1000;
            if (Number.isFinite(ts) && ts > best) best = ts;
        } catch {
            // ignore
        }
    }
    return best > 0 ? best : null;
}

const _qualityCache: {
    state: DataQualityState;
    updatedAtMs: number;
    holdUntilSec: number;
    stableOkChecks: number;
} = {
    state: {
        checkedAtMs: 0,
        mode: 'hard_degraded',
        severity: 'hard_degraded',
        triggerClass: 'api_degraded',
        staleSources: [],
        criticalStale: [],
        confidencePenalty: 0.02,
        betScale: Math.min(1, Math.max(0, DATA_QUALITY_BET_SCALE_DEGRADED)),
        apiErrorRate: 0,
        apiErrorSources: [],
        keySourceErrors: [],
        nonKeySourceErrors: [],
        secondarySourceErrors: [],
        observationCounts: {},
        consecutiveBadWindows: {},
        consecutiveGoodWindows: {},
        reason: 'init',
    },
    updatedAtMs: 0,
    holdUntilSec: 0,
    stableOkChecks: 0,
};

function readDataQualityState(nowMs: number): DataQualityState {
    if (nowMs - _qualityCache.updatedAtMs < DATA_QUALITY_REFRESH_MS) {
        return _qualityCache.state;
    }

    const nextState: DataQualityState = {
        checkedAtMs: nowMs,
        mode: 'ok',
        severity: 'ok',
        triggerClass: 'ok',
        staleSources: [],
        criticalStale: [],
        confidencePenalty: 0,
        betScale: 1,
        apiErrorRate: 0,
        apiErrorSources: [],
        keySourceErrors: [],
        nonKeySourceErrors: [],
        secondarySourceErrors: [],
        observationCounts: {},
        consecutiveBadWindows: {},
        consecutiveGoodWindows: {},
        reason: 'ok',
    };

    try {
        if (!fs.existsSync(DATA_QUALITY_FILE)) {
            nextState.mode = 'hard_degraded';
            nextState.severity = 'hard_degraded';
            nextState.triggerClass = 'stale_data';
            nextState.confidencePenalty = 0.03;
            nextState.betScale = Math.min(1, Math.max(0, DATA_QUALITY_BET_SCALE_DEGRADED));
            nextState.reason = `missing:${path.basename(DATA_QUALITY_FILE)}`;
            const nowSecMissing = Math.floor(nowMs / 1000);
            _qualityCache.holdUntilSec = nowSecMissing + Math.max(1, DATA_QUALITY_RECOVERY_BARS) * PERIOD_SECONDS;
            _qualityCache.stableOkChecks = 0;
            _qualityCache.state = nextState;
            _qualityCache.updatedAtMs = nowMs;
            return nextState;
        }

        const raw = JSON.parse(fs.readFileSync(DATA_QUALITY_FILE, 'utf8')) as Record<string, unknown>;
        const nowSec = nowMs / 1000;
        const sources = [
            'open_interest',
            'long_short_ratio',
            'polymarket_prob',
            'polymarket_prob_target',
            'funding_rate',
            'ob_realtime',
        ];

        for (const src of sources) {
            const tsVal = raw[src];
            const ts = typeof tsVal === 'number' ? tsVal : Number.NaN;
            const maxAgeSec = Number.isFinite(DATA_QUALITY_MAX_AGE_BY_SOURCE_SEC[src])
                ? DATA_QUALITY_MAX_AGE_BY_SOURCE_SEC[src]
                : DATA_QUALITY_MAX_AGE_SEC;
            let fresh = false;
            if (Number.isFinite(ts)) {
                const ageSec = nowSec - ts;
                if (ageSec <= maxAgeSec) fresh = true;
            }
            // 兜底：某些源会由 writer 写 parquet 而非采集器更新 data_ready。
            if (!fresh) {
                const fb = latestMtimeSec(DATA_QUALITY_MTIME_FALLBACK_FILES[src] || []);
                if (fb != null) {
                    const ageSec = nowSec - fb;
                    if (ageSec <= maxAgeSec) fresh = true;
                }
            }
            if (!fresh) {
                nextState.staleSources.push(src);
                if (DATA_QUALITY_CRITICAL_SOURCES.has(src)) nextState.criticalStale.push(src);
            }
        }

        const apiRates = (raw.api_error_rates && typeof raw.api_error_rates === 'object')
            ? (raw.api_error_rates as Record<string, unknown>)
            : {};
        const apiObs = (raw.api_error_observations && typeof raw.api_error_observations === 'object')
            ? (raw.api_error_observations as Record<string, unknown>)
            : {};
        const apiBadWindows = (raw.api_consecutive_bad_windows && typeof raw.api_consecutive_bad_windows === 'object')
            ? (raw.api_consecutive_bad_windows as Record<string, unknown>)
            : {};
        const apiGoodWindows = (raw.api_consecutive_good_windows && typeof raw.api_consecutive_good_windows === 'object')
            ? (raw.api_consecutive_good_windows as Record<string, unknown>)
            : {};
        const apiErrorSources: string[] = [];
        const softSources: string[] = [];
        const hardSources: string[] = [];
        for (const [src, rateVal] of Object.entries(apiRates)) {
            const rate = Number(rateVal);
            const obs = Number(apiObs[src]);
            const badWindows = Number(apiBadWindows[src]);
            const goodWindows = Number(apiGoodWindows[src]);
            if (!Number.isFinite(rate) || !Number.isFinite(obs)) continue;
            nextState.observationCounts[src] = Math.max(0, Math.trunc(obs));
            if (Number.isFinite(badWindows) && badWindows >= 0) {
                nextState.consecutiveBadWindows[src] = Math.max(0, Math.trunc(badWindows));
            }
            if (Number.isFinite(goodWindows) && goodWindows >= 0) {
                nextState.consecutiveGoodWindows[src] = Math.max(0, Math.trunc(goodWindows));
            }
            if (obs < Math.max(1, DATA_QUALITY_API_MIN_OBS_SOFT)) continue;
            const sourceClass = classifyQualitySource(src);
            const softThreshold = apiRateSoftThreshold(sourceClass);
            const hardThreshold = apiRateHardThreshold(sourceClass);
            const badCount = Math.max(0, Math.trunc(Number.isFinite(badWindows) ? badWindows : 0));
            if (rate >= hardThreshold && badCount >= Math.max(1, DATA_QUALITY_HARD_SOURCE_BAD_WINDOWS)) {
                hardSources.push(src);
                apiErrorSources.push(src);
            } else if (rate >= softThreshold && badCount >= Math.max(1, DATA_QUALITY_SOFT_SOURCE_BAD_WINDOWS)) {
                softSources.push(src);
                apiErrorSources.push(src);
            }
        }
        nextState.apiErrorSources = apiErrorSources.sort();
        nextState.keySourceErrors = nextState.apiErrorSources.filter((src) => classifyQualitySource(src) === 'key');
        nextState.secondarySourceErrors = nextState.apiErrorSources.filter((src) => classifyQualitySource(src) === 'secondary');
        nextState.nonKeySourceErrors = nextState.apiErrorSources.filter((src) => classifyQualitySource(src) === 'non_key');
        const globalApiRate = Number(raw.api_error_rate);
        if (Number.isFinite(globalApiRate) && globalApiRate >= 0) {
            nextState.apiErrorRate = globalApiRate;
        } else if (nextState.apiErrorSources.length > 0) {
            const vals = nextState.apiErrorSources
                .map((s) => Number(apiRates[s]))
                .filter((v) => Number.isFinite(v));
            nextState.apiErrorRate = vals.length > 0 ? vals.reduce((a, b) => a + b, 0) / vals.length : 0;
        }

        const keyStale = nextState.staleSources.filter((src) => DATA_QUALITY_CRITICAL_SOURCES.has(src));
        const hardApiCount = hardSources.length;
        const softApiCount = softSources.length;
        const keyHardApiCount = hardSources.filter((src) => classifyQualitySource(src) === 'key').length;
        const secondaryHardApiCount = hardSources.filter((src) => classifyQualitySource(src) === 'secondary').length;
        const mixedSoftCount = softSources.filter((src) => classifyQualitySource(src) !== 'non_key').length;
        const hasApiHalt = nextState.apiErrorRate >= DATA_QUALITY_API_ERROR_HALT || keyHardApiCount >= 2 || hardApiCount >= 3;

        if (nextState.criticalStale.length > 1 || nextState.staleSources.length >= DATA_QUALITY_HALT_STALE_COUNT) {
            nextState.mode = 'halt';
            nextState.severity = 'halt';
            nextState.triggerClass = 'stale_data';
            nextState.confidencePenalty = 1.0;
            nextState.betScale = Math.min(1, Math.max(0, DATA_QUALITY_BET_SCALE_HALT));
            nextState.reason = `critical:${nextState.criticalStale.join(',') || nextState.staleSources.join(',')}`;
        } else if (hasApiHalt) {
            nextState.mode = 'halt';
            nextState.severity = 'halt';
            nextState.triggerClass = 'api_degraded';
            nextState.confidencePenalty = 1.0;
            nextState.betScale = Math.min(1, Math.max(0, DATA_QUALITY_BET_SCALE_HALT));
            nextState.reason = `api_halt:rate=${nextState.apiErrorRate.toFixed(3)}`;
        } else if (keyStale.length > 0 || nextState.staleSources.length >= 2 || keyHardApiCount >= 1 || secondaryHardApiCount >= 2) {
            nextState.mode = 'hard_degraded';
            nextState.severity = 'hard_degraded';
            nextState.triggerClass = nextState.staleSources.length > 0 && hardApiCount > 0 ? 'mixed' : (nextState.staleSources.length > 0 ? 'stale_data' : 'api_degraded');
            nextState.confidencePenalty = Math.min(
                DATA_QUALITY_CONF_PENALTY_MAX,
                Math.max(
                    nextState.staleSources.length * DATA_QUALITY_CONF_PENALTY_PER_STALE,
                    Math.max(DATA_QUALITY_CONF_PENALTY_PER_STALE, nextState.apiErrorRate * DATA_QUALITY_CONF_PENALTY_MAX * 2),
                ),
            );
            nextState.betScale = Math.min(1, Math.max(0, DATA_QUALITY_BET_SCALE_DEGRADED));
            nextState.reason = nextState.staleSources.length > 0
                ? `hard_stale:${nextState.staleSources.join(',')}`
                : `api_hard:rate=${nextState.apiErrorRate.toFixed(3)} src=${nextState.apiErrorSources.join(',')}`;
        } else if (nextState.staleSources.length >= DATA_QUALITY_DEGRADE_STALE_COUNT || softApiCount > 0 || mixedSoftCount > 1) {
            nextState.mode = 'soft_degraded';
            nextState.severity = 'soft_degraded';
            nextState.triggerClass = nextState.staleSources.length > 0 && softApiCount > 0 ? 'mixed' : (nextState.staleSources.length > 0 ? 'stale_data' : 'api_degraded');
            nextState.confidencePenalty = Math.min(
                DATA_QUALITY_CONF_PENALTY_MAX / 2,
                Math.max(
                    DATA_QUALITY_CONF_PENALTY_PER_STALE / 2,
                    nextState.apiErrorRate * DATA_QUALITY_CONF_PENALTY_MAX,
                ),
            );
            nextState.betScale = Math.min(1, Math.max(0, DATA_QUALITY_BET_SCALE_SOFT));
            nextState.reason = nextState.staleSources.length > 0
                ? `stale_soft:${nextState.staleSources.join(',')}`
                : `api_soft:rate=${nextState.apiErrorRate.toFixed(3)} src=${nextState.apiErrorSources.join(',')}`;
        }
    } catch (err) {
        nextState.mode = 'hard_degraded';
        nextState.severity = 'hard_degraded';
        nextState.triggerClass = 'api_degraded';
        nextState.confidencePenalty = 0.03;
        nextState.betScale = Math.min(1, Math.max(0, DATA_QUALITY_BET_SCALE_DEGRADED));
        nextState.reason = `parse_error:${String(err).slice(0, 80)}`;
    }

    const prev = _qualityCache.state;
    const prevSev = qualityModeSeverity(prev.mode);
    const nextSev = qualityModeSeverity(nextState.mode);
    const nowSec = Math.floor(nowMs / 1000);
    let effective = nextState;

    if (nextSev > prevSev) {
        _qualityCache.holdUntilSec = nowSec + Math.max(1, DATA_QUALITY_RECOVERY_BARS) * PERIOD_SECONDS;
        _qualityCache.stableOkChecks = 0;
    } else if (nextSev < prevSev) {
        const prevTriggerClass = String(prev.triggerClass || 'ok');
        const prevReason = String(prev.reason || '');
        const shouldHold = nowSec < _qualityCache.holdUntilSec && (prevTriggerClass === 'api_degraded' || prevTriggerClass === 'mixed');
        if (shouldHold) {
            effective = {
                ...nextState,
                // 恢复保持期只冻结门控强度，不冻结源状态快照，避免 stale/api 诊断信息长期滞后
                mode: prev.mode,
                severity: prev.severity,
                triggerClass: prev.triggerClass,
                confidencePenalty: prev.confidencePenalty,
                betScale: prev.betScale,
                checkedAtMs: nowMs,
                reason: `${nextState.reason}|recovery_hold_until=${new Date(_qualityCache.holdUntilSec * 1000).toISOString()}`,
            };
        } else if (nextSev === 0 && DATA_QUALITY_RECOVERY_STABLE_CHECKS > 1 && (prevTriggerClass === 'api_degraded' || prevTriggerClass === 'mixed')) {
            _qualityCache.stableOkChecks += 1;
            if (_qualityCache.stableOkChecks < DATA_QUALITY_RECOVERY_STABLE_CHECKS) {
                effective = {
                    ...nextState,
                    mode: prev.mode,
                    severity: prev.severity,
                    triggerClass: prev.triggerClass,
                    confidencePenalty: prev.confidencePenalty,
                    betScale: prev.betScale,
                    checkedAtMs: nowMs,
                    reason: `${nextState.reason}|stabilizing=${_qualityCache.stableOkChecks}/${DATA_QUALITY_RECOVERY_STABLE_CHECKS}`,
                };
            } else {
                _qualityCache.stableOkChecks = 0;
            }
        } else {
            _qualityCache.stableOkChecks = 0;
        }
    } else if (nextSev === 0) {
        _qualityCache.stableOkChecks = 0;
    }

    _qualityCache.state = effective;
    _qualityCache.updatedAtMs = nowMs;
    return effective;
}

function extractNumericPnl(entry: Record<string, unknown>): number {
    const sim = entry.simulatedPnl;
    if (typeof sim === 'number' && Number.isFinite(sim)) return sim;
    const pnl = entry.pnl;
    if (typeof pnl === 'number' && Number.isFinite(pnl)) return pnl;
    return 0;
}

function normalizeSymbol(raw: unknown): string {
    return String(raw ?? '')
        .replace(/\/.*$/, '')
        .trim()
        .toUpperCase();
}

function normalizeDirection(raw: unknown): 'UP' | 'DOWN' | null {
    const d = String(raw ?? '').trim().toUpperCase();
    if (d === 'UP' || d === 'DOWN') return d;
    return null;
}

function symbolDirectionConfigKey(symbol: string, direction: 'UP' | 'DOWN'): string {
    return `${String(symbol || '').toUpperCase()}_${direction}`;
}

function symbolDirectionRuntimeKey(symbol: string, direction: 'UP' | 'DOWN'): string {
    return `${String(symbol || '').toUpperCase()}::${direction}`;
}

function parseSettledMs(entry: Record<string, unknown>): number | null {
    const settledAt = String((entry.settledAt as string) || (entry.timestamp as string) || '');
    const ms = Date.parse(settledAt);
    return Number.isFinite(ms) ? ms : null;
}

function downRiskSampleKey(
    entry: Record<string, unknown>,
    symbol: string,
    mode: DownRiskRouteMode,
): string {
    const conditionId = String(entry.conditionId || '').trim();
    if (conditionId) {
        return `${conditionId}::${symbol}::DOWN::${mode}`;
    }
    const marketSlug = String(entry.marketSlug || '').trim();
    const settledAt = String((entry.settledAt as string) || (entry.timestamp as string) || '').trim();
    const entryId = String(entry.id || '').trim();
    return `${marketSlug}::${settledAt}::${entryId}::${symbol}::DOWN::${mode}`;
}

function upRiskSampleKey(
    entry: Record<string, unknown>,
    symbol: string,
    mode: UpRiskRouteMode,
): string {
    const conditionId = String(entry.conditionId || '').trim();
    if (conditionId) {
        return `${conditionId}::${symbol}::UP::${mode}`;
    }
    const marketSlug = String(entry.marketSlug || '').trim();
    const settledAt = String((entry.settledAt as string) || (entry.timestamp as string) || '').trim();
    const entryId = String(entry.id || '').trim();
    return `${marketSlug}::${settledAt}::${entryId}::${symbol}::UP::${mode}`;
}

function shockRiskSampleKey(
    entry: Record<string, unknown>,
    symbol: string,
    direction: 'UP' | 'DOWN',
    mode: ShockRiskRouteMode,
): string {
    const conditionId = String(entry.conditionId || '').trim();
    if (conditionId) {
        return `${conditionId}::${symbol}::${direction}::${mode}`;
    }
    const marketSlug = String(entry.marketSlug || '').trim();
    const settledAt = String((entry.settledAt as string) || (entry.timestamp as string) || '').trim();
    const entryId = String(entry.id || '').trim();
    return `${marketSlug}::${settledAt}::${entryId}::${symbol}::${direction}::${mode}`;
}

function comboPauseSampleKey(
    entry: Record<string, unknown>,
    symbol: string,
    mode: ComboPauseRouteMode,
): string {
    const direction = normalizeDirection(entry.direction) || 'UNKNOWN';
    const conditionId = String(entry.conditionId || '').trim();
    if (conditionId) {
        return `${conditionId}::${symbol}::${direction}::${mode}`;
    }
    const marketSlug = String(entry.marketSlug || '').trim();
    const settledAt = String((entry.settledAt as string) || (entry.timestamp as string) || '').trim();
    const entryId = String(entry.id || '').trim();
    return `${marketSlug}::${settledAt}::${entryId}::${symbol}::${direction}::${mode}`;
}

function calibrationSampleKey(
    entry: Record<string, unknown>,
    symbol: string,
    direction: 'UP' | 'DOWN',
    mode: CalibrationRouteMode,
): string {
    const conditionId = String(entry.conditionId || '').trim();
    if (conditionId) return `${conditionId}::${symbol}::${direction}::${mode}`;
    const marketSlug = String(entry.marketSlug || '').trim();
    const settledAt = String((entry.settledAt as string) || (entry.timestamp as string) || '').trim();
    const entryId = String(entry.id || '').trim();
    return `${marketSlug}::${settledAt}::${entryId}::${symbol}::${direction}::${mode}`;
}

function expectancyGateSampleKey(
    entry: Record<string, unknown>,
    symbol: string,
    direction: 'UP' | 'DOWN',
    mode: ExpectancyGateRouteMode,
): string {
    return calibrationSampleKey(entry, symbol, direction, mode);
}

function resolveDownRiskEngine(cfg: TraderConfig): DownRiskEngine {
    return cfg.downRiskEngine === 'v2' ? 'v2' : DOWN_RISK_ENGINE_DEFAULT;
}

function resolveSymbolScopedMode<T extends string>(
    symbol: string | undefined,
    bySymbol: ModeBySymbol<T>,
): T | undefined {
    if (!symbol || !bySymbol) return undefined;
    const upper = symbol.toUpperCase();
    const lower = symbol.toLowerCase();
    const exact = bySymbol[symbol];
    if (exact != null) return exact;
    const upperHit = bySymbol[upper];
    if (upperHit != null) return upperHit;
    const lowerHit = bySymbol[lower];
    if (lowerHit != null) return lowerHit;
    return undefined;
}

function resolveDownRiskMode(cfg: TraderConfig, symbol?: string): DownRiskMode {
    const scoped = resolveSymbolScopedMode<DownRiskMode>(symbol, cfg.downRiskModeBySymbol);
    const raw = scoped || cfg.downRiskMode || DOWN_RISK_MODE_DEFAULT;
    if (raw === 'shadow' || raw === 'enforce') return raw;
    return 'off';
}

function resolveDownRiskStatsMode(cfg: TraderConfig): DownRiskStatsMode {
    // V2.2 统一要求 route 隔离，避免显式 simulation/live 配置造成跨模式误读。
    if (resolveDownRiskEngine(cfg) === 'v2') return 'route';
    const raw = cfg.downRiskStatsMode || DOWN_RISK_STATS_MODE_DEFAULT;
    if (raw === 'simulation' || raw === 'live' || raw === 'route') return raw;
    return 'route';
}

function resolveDownRiskRouteMode(t: TraderInstance): DownRiskRouteMode {
    const statsMode = resolveDownRiskStatsMode(t.config);
    if (statsMode === 'simulation' || statsMode === 'live') {
        return statsMode;
    }
    return t.config.appliedTradingMode === 'live' ? 'live' : 'simulation';
}

function getDownRiskSymbolOverrides(t: TraderInstance, symbol: string): Record<string, unknown> {
    const all = t.config.downRiskBySymbol || {};
    const upper = symbol.toUpperCase();
    const lower = symbol.toLowerCase();
    return (all[upper] as Record<string, unknown>) || (all[lower] as Record<string, unknown>) || {};
}

function isDownRiskTargetSymbol(symbol: string): boolean {
    return DOWN_RISK_TARGET_SYMBOLS.has(String(symbol || '').toUpperCase());
}

function getDownRiskNumber(
    t: TraderInstance,
    symbol: string,
    key: string,
    fallback: number,
): number {
    const overrides = getDownRiskSymbolOverrides(t, symbol);
    const val = overrides[key];
    const n = Number(val);
    return Number.isFinite(n) ? n : fallback;
}

function resolveUpRiskEngine(cfg: TraderConfig): UpRiskEngine {
    return cfg.upRiskEngine === 'v2' ? 'v2' : UP_RISK_ENGINE_DEFAULT;
}

function resolveUpRiskMode(cfg: TraderConfig, symbol?: string): UpRiskMode {
    const scoped = resolveSymbolScopedMode<UpRiskMode>(symbol, cfg.upRiskModeBySymbol);
    const raw = scoped || cfg.upRiskMode || UP_RISK_MODE_DEFAULT;
    if (raw === 'shadow' || raw === 'enforce') return raw;
    return 'off';
}

function resolveDrawdownAccelerationMode(cfg: TraderConfig, symbol?: string): DrawdownAccelerationMode {
    const scoped = resolveSymbolScopedMode<DrawdownAccelerationMode>(symbol, cfg.drawdownAccelerationModeBySymbol);
    const raw = scoped || cfg.drawdownAccelerationMode || 'enforce';
    if (raw === 'off' || raw === 'shadow' || raw === 'enforce') return raw;
    return 'enforce';
}

function resolveUpRiskStatsMode(cfg: TraderConfig): UpRiskStatsMode {
    if (resolveUpRiskEngine(cfg) === 'v2') return 'route';
    const raw = cfg.upRiskStatsMode || UP_RISK_STATS_MODE_DEFAULT;
    if (raw === 'simulation' || raw === 'live' || raw === 'route') return raw;
    return 'route';
}

function resolveUpRiskRouteMode(t: TraderInstance): UpRiskRouteMode {
    const statsMode = resolveUpRiskStatsMode(t.config);
    if (statsMode === 'simulation' || statsMode === 'live') {
        return statsMode;
    }
    return t.config.appliedTradingMode === 'live' ? 'live' : 'simulation';
}

function getUpRiskSymbolOverrides(t: TraderInstance, symbol: string): Record<string, unknown> {
    const all = t.config.upRiskBySymbol || {};
    const upper = symbol.toUpperCase();
    const lower = symbol.toLowerCase();
    return (all[upper] as Record<string, unknown>) || (all[lower] as Record<string, unknown>) || {};
}

function isUpRiskTargetSymbol(symbol: string): boolean {
    return UP_RISK_TARGET_SYMBOLS.has(String(symbol || '').toUpperCase());
}

function getUpRiskNumber(
    t: TraderInstance,
    symbol: string,
    key: string,
    fallback: number,
): number {
    const overrides = getUpRiskSymbolOverrides(t, symbol);
    const val = overrides[key];
    const n = Number(val);
    return Number.isFinite(n) ? n : fallback;
}

function resolveShockRiskEngine(cfg: TraderConfig): ShockRiskEngine {
    return cfg.shockRiskEngine === 'v1' ? 'v1' : SHOCK_RISK_ENGINE_DEFAULT;
}

function resolveShockRiskMode(cfg: TraderConfig, symbol?: string): ShockRiskMode {
    const scoped = resolveSymbolScopedMode<ShockRiskMode>(symbol, cfg.shockRiskModeBySymbol);
    const raw = scoped || cfg.shockRiskMode || SHOCK_RISK_MODE_DEFAULT;
    if (raw === 'off' || raw === 'enforce') return raw;
    return 'shadow';
}

function resolveMainlineExtremeMode(cfg: TraderConfig, symbol?: string): MainlineExtremeMode {
    const scoped = resolveSymbolScopedMode<MainlineExtremeMode>(symbol, cfg.mainlineExtremeModeBySymbol);
    const raw = scoped || cfg.mainlineExtremeMode || 'enforce';
    if (raw === 'off' || raw === 'enforce') return raw;
    return 'shadow';
}

function resolveShockRiskStatsMode(cfg: TraderConfig): ShockRiskStatsMode {
    if (resolveShockRiskEngine(cfg) === 'v2') return 'route';
    const raw = cfg.shockRiskStatsMode || SHOCK_RISK_STATS_MODE_DEFAULT;
    if (raw === 'simulation' || raw === 'live' || raw === 'route') return raw;
    return 'route';
}

function resolveShockRiskRouteMode(t: TraderInstance): ShockRiskRouteMode {
    const statsMode = resolveShockRiskStatsMode(t.config);
    if (statsMode === 'simulation' || statsMode === 'live') {
        return statsMode;
    }
    return t.config.appliedTradingMode === 'live' ? 'live' : 'simulation';
}

function getShockRiskSymbolOverrides(t: TraderInstance, symbol: string): Record<string, unknown> {
    const all = t.config.shockRiskBySymbol || {};
    const upper = symbol.toUpperCase();
    const lower = symbol.toLowerCase();
    return (all[upper] as Record<string, unknown>) || (all[lower] as Record<string, unknown>) || {};
}

function isShockRiskTargetSymbol(symbol: string): boolean {
    return SHOCK_RISK_TARGET_SYMBOLS.has(String(symbol || '').toUpperCase());
}

function getShockRiskNumber(
    t: TraderInstance,
    symbol: string,
    key: string,
    fallback: number,
): number {
    const overrides = getShockRiskSymbolOverrides(t, symbol);
    const val = overrides[key];
    const n = Number(val);
    return Number.isFinite(n) ? n : fallback;
}

function resolveComboPauseEnabled(cfg: TraderConfig): boolean {
    if (typeof cfg.comboPauseEnabled === 'boolean') return cfg.comboPauseEnabled;
    return COMBO_PAUSE_ENABLED_DEFAULT;
}

function resolveComboPauseMode(
    cfg: TraderConfig,
    symbol?: string,
    direction: 'UP' | 'DOWN' | 'BOTH' = 'BOTH',
): ComboPauseMode {
    const scoped = direction === 'UP' || direction === 'DOWN'
        ? resolveSymbolDirectionScopedMode(symbol, direction, cfg.comboPauseModeBySymbolDirection)
        : undefined;
    const symbolScoped = resolveSymbolScopedMode<ComboPauseMode>(symbol, cfg.comboPauseModeBySymbol);
    const raw = scoped || symbolScoped || cfg.comboPauseMode || COMBO_PAUSE_MODE_DEFAULT;
    if (raw === 'off' || raw === 'enforce') return raw;
    return 'shadow';
}

function getComboPauseDirectionsForSymbol(
    cfg: TraderConfig,
    profile: 'default' | '70',
    symbol: string,
): Array<'UP' | 'DOWN' | 'BOTH'> {
    const upperSymbol = String(symbol || '').toUpperCase();
    const directionalKeys = new Set<'UP' | 'DOWN'>();
    const modeMap = cfg.comboPauseModeBySymbolDirection || {};
    const paramMap = cfg.comboPauseBySymbolDirection || {};
    for (const rawKey of Object.keys(modeMap).concat(Object.keys(paramMap))) {
        const key = String(rawKey || '').toUpperCase().replace('::', '_');
        if (key === `${upperSymbol}_UP`) directionalKeys.add('UP');
        if (key === `${upperSymbol}_DOWN`) directionalKeys.add('DOWN');
    }
    if (directionalKeys.size > 0) {
        const out: Array<'UP' | 'DOWN'> = [];
        if (directionalKeys.has('UP')) out.push('UP');
        if (directionalKeys.has('DOWN')) out.push('DOWN');
        return out;
    }
    if (profile === '70' && upperSymbol === 'BTC') {
        return ['DOWN'];
    }
    return ['BOTH'];
}

function resolveComboPauseModeCandidates(
    cfg: TraderConfig,
    symbol: string,
): Array<'UP' | 'DOWN' | 'BOTH'> {
    const directions = getComboPauseDirectionsForSymbol(cfg, (cfg.profile ?? 'default'), symbol);
    return directions;
}

function resolveComboPauseModeByCell(
    cfg: TraderConfig,
    symbol: string,
    direction: 'UP' | 'DOWN' | 'BOTH',
): ComboPauseMode {
    return resolveComboPauseMode(cfg, symbol, direction);
}

function resolveDirectionLossCooldownMode(
    cfg: TraderConfig,
    symbol: string,
    direction: 'UP' | 'DOWN',
): DirectionLossCooldownMode {
    const scoped = resolveSymbolDirectionScopedMode(symbol, direction, cfg.directionLossCooldownModeBySymbolDirection);
    const raw = scoped || cfg.directionLossCooldownMode || 'enforce';
    if (raw === 'off' || raw === 'enforce') return raw;
    return 'shadow';
}

function getComboPauseSymbolOverrides(t: TraderInstance, symbol: string): Record<string, unknown> {
    const all = t.config.comboPauseBySymbol || {};
    const upper = symbol.toUpperCase();
    const lower = symbol.toLowerCase();
    return (all[upper] as Record<string, unknown>) || (all[lower] as Record<string, unknown>) || {};
}

function getComboPauseSymbolDirectionOverrides(
    t: TraderInstance,
    symbol: string,
    direction: 'UP' | 'DOWN' | 'BOTH',
): Record<string, unknown> {
    if (direction !== 'UP' && direction !== 'DOWN') return {};
    const all = t.config.comboPauseBySymbolDirection || {};
    const keys = [
        symbolDirectionConfigKey(symbol, direction),
        symbolDirectionConfigKey(symbol.toUpperCase(), direction),
        symbolDirectionConfigKey(symbol.toLowerCase(), direction),
        symbolDirectionRuntimeKey(symbol, direction),
        symbolDirectionRuntimeKey(symbol.toUpperCase(), direction),
        symbolDirectionRuntimeKey(symbol.toLowerCase(), direction),
    ];
    for (const key of keys) {
        const hit = all[key];
        if (hit && typeof hit === 'object') return hit as Record<string, unknown>;
    }
    return {};
}

function getComboPauseNumber(
    t: TraderInstance,
    symbol: string,
    direction: 'UP' | 'DOWN' | 'BOTH',
    key: string,
    fallback: number,
): number {
    const directionOverrides = getComboPauseSymbolDirectionOverrides(t, symbol, direction);
    const directionVal = directionOverrides[key];
    const directionNum = Number(directionVal);
    if (Number.isFinite(directionNum)) return directionNum;
    const overrides = getComboPauseSymbolOverrides(t, symbol);
    const val = overrides[key];
    const n = Number(val);
    return Number.isFinite(n) ? n : fallback;
}

function isComboPauseTargetSymbol(symbol: string): boolean {
    return COMBO_PAUSE_TARGET_SYMBOLS.has(String(symbol || '').toUpperCase());
}

function isAcuteDownPriorityCluster(profile: 'default' | '70', symbol: string): boolean {
    const upper = String(symbol || '').toUpperCase();
    return (profile === '70' && (upper === 'BTC' || upper === 'ETH')) || (profile === 'default' && upper === 'ETH');
}

function isAcuteExpectancyPriorityCluster(
    profile: 'default' | '70',
    symbol: string,
    direction: 'UP' | 'DOWN',
): boolean {
    if (direction !== 'DOWN') return false;
    return isAcuteDownPriorityCluster(profile, symbol);
}

function isDirectionalComboPriorityCluster(
    profile: 'default' | '70',
    symbol: string,
    direction: 'UP' | 'DOWN',
): boolean {
    return profile === '70' && String(symbol || '').toUpperCase() === 'BTC' && direction === 'DOWN';
}

function getDownCrossLayerTriggerCellKeys(
    profile: 'default' | '70',
    traderName: string,
    symbol: string,
    routeMode: 'simulation' | 'live',
): { expectKey: string; shockKey: string; comboKeys: string[] } {
    const upperSymbol = symbol.toUpperCase();
    return {
        expectKey: expectancyGateCellKey(profile, traderName, upperSymbol, 'DOWN', routeMode),
        shockKey: shockRiskCellKey(profile, traderName, upperSymbol, 'DOWN', routeMode),
        comboKeys: [
            comboPauseCellKey(profile, traderName, upperSymbol, 'DOWN', routeMode),
            comboPauseCellKey(profile, traderName, upperSymbol, 'BOTH', routeMode),
        ],
    };
}

function resolveDownTriggerRouteMode(routeMode: DownRiskRouteMode): 'simulation' | 'live' {
    return routeMode === 'live' ? 'live' : 'simulation';
}

function resolveComboPauseStatsMode(cfg: TraderConfig): ComboPauseStatsMode {
    const raw = cfg.comboPauseStatsMode || COMBO_PAUSE_STATS_MODE_DEFAULT;
    if (raw === 'simulation' || raw === 'live' || raw === 'route') return raw;
    return 'route';
}

function resolveComboPauseRouteMode(t: TraderInstance): ComboPauseRouteMode {
    const statsMode = resolveComboPauseStatsMode(t.config);
    if (statsMode === 'simulation' || statsMode === 'live') {
        return statsMode;
    }
    return t.config.appliedTradingMode === 'live' ? 'live' : 'simulation';
}

function getComboPauseSymbolOverridesLegacy(t: TraderInstance, symbol: string): Record<string, unknown> {
    return getComboPauseSymbolOverrides(t, symbol);
}

function isComboPauseTargetSymbolLegacy(symbol: string): boolean {
    return isComboPauseTargetSymbol(symbol);
}

function getComboPauseNumberLegacy(
    t: TraderInstance,
    symbol: string,
    key: string,
    fallback: number,
): number {
    return getComboPauseNumber(t, symbol, 'BOTH', key, fallback);
}

// Legacy wrappers kept for nearby code paths that still call the old symbol-only helpers.
// They intentionally resolve to BOTH so existing symbol-level behavior remains stable.
function resolveComboPauseModeLegacy(cfg: TraderConfig, symbol?: string): ComboPauseMode {
    return resolveComboPauseMode(cfg, symbol, 'BOTH');
}

function getComboPauseNumberLegacyByDirection(
    t: TraderInstance,
    symbol: string,
    direction: 'UP' | 'DOWN' | 'BOTH',
    key: string,
    fallback: number,
): number {
    return getComboPauseNumber(t, symbol, direction, key, fallback);
}

// keep backward references below compiling while we gradually move callers
const resolveComboPauseModeSymbolOnly = resolveComboPauseModeLegacy;
const getComboPauseNumberSymbolOnly = getComboPauseNumberLegacy;

function resolveSymbolDirectionScopedMode<T extends string>(
    symbol: string | undefined,
    direction: 'UP' | 'DOWN',
    bySymbolDirection: Record<string, T> | undefined,
): T | undefined {
    if (!symbol || !bySymbolDirection) return undefined;
    const candidates = [
        symbolDirectionConfigKey(symbol, direction),
        symbolDirectionConfigKey(symbol.toUpperCase(), direction),
        symbolDirectionConfigKey(symbol.toLowerCase(), direction),
        symbolDirectionRuntimeKey(symbol, direction),
        symbolDirectionRuntimeKey(symbol.toUpperCase(), direction),
        symbolDirectionRuntimeKey(symbol.toLowerCase(), direction),
    ];
    for (const key of candidates) {
        const hit = bySymbolDirection[key];
        if (hit != null) return hit;
    }
    return undefined;
}

function resolveCalibrationMode(
    cfg: TraderConfig,
    symbol: string,
    direction: 'UP' | 'DOWN',
): CalibrationMode {
    const scoped = resolveSymbolDirectionScopedMode(symbol, direction, cfg.calibrationModeBySymbolDirection);
    const raw = scoped || cfg.calibrationMode || CALIBRATION_MODE_DEFAULT;
    if (raw === 'shadow' || raw === 'enforce') return raw;
    return 'off';
}

function resolveCalibrationStatsMode(cfg: TraderConfig): CalibrationStatsMode {
    const raw = cfg.calibrationStatsMode || CALIBRATION_STATS_MODE_DEFAULT;
    if (raw === 'simulation' || raw === 'live' || raw === 'route') return raw;
    return 'route';
}

function resolveCalibrationRouteMode(t: TraderInstance): CalibrationRouteMode {
    const statsMode = resolveCalibrationStatsMode(t.config);
    if (statsMode === 'simulation' || statsMode === 'live') return statsMode;
    return t.config.appliedTradingMode === 'live' ? 'live' : 'simulation';
}

function getCalibrationSymbolDirectionOverrides(
    t: TraderInstance,
    symbol: string,
    direction: 'UP' | 'DOWN',
): Record<string, unknown> {
    const all = t.config.calibrationBySymbolDirection || {};
    const keys = [
        symbolDirectionConfigKey(symbol, direction),
        symbolDirectionConfigKey(symbol.toUpperCase(), direction),
        symbolDirectionConfigKey(symbol.toLowerCase(), direction),
        symbolDirectionRuntimeKey(symbol, direction),
        symbolDirectionRuntimeKey(symbol.toUpperCase(), direction),
        symbolDirectionRuntimeKey(symbol.toLowerCase(), direction),
    ];
    for (const key of keys) {
        const hit = all[key];
        if (hit && typeof hit === 'object') return hit as Record<string, unknown>;
    }
    return {};
}

function getCalibrationNumber(
    t: TraderInstance,
    symbol: string,
    direction: 'UP' | 'DOWN',
    key: string,
    fallback: number,
): number {
    const overrides = getCalibrationSymbolDirectionOverrides(t, symbol, direction);
    const value = overrides[key];
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
}

function getCalibrationMethod(
    t: TraderInstance,
    symbol: string,
    direction: 'UP' | 'DOWN',
): CalibrationMethod {
    const overrides = getCalibrationSymbolDirectionOverrides(t, symbol, direction);
    const raw = String(overrides.calibrationMethod || CALIBRATION_METHOD_DEFAULT).toLowerCase();
    if (raw === 'temperature' || raw === 'sigmoid' || raw === 'isotonic') return raw;
    return 'identity';
}

function getCalibrationArray(
    t: TraderInstance,
    symbol: string,
    direction: 'UP' | 'DOWN',
    key: 'calibrationIsotonicX' | 'calibrationIsotonicY',
): number[] {
    const overrides = getCalibrationSymbolDirectionOverrides(t, symbol, direction);
    const value = overrides[key];
    if (!Array.isArray(value)) return [];
    return value
        .map((x) => Number(x))
        .filter((x) => Number.isFinite(x));
}

function isCalibrationTargetSymbol(symbol: string): boolean {
    return CALIBRATION_TARGET_SYMBOLS.has(String(symbol || '').toUpperCase());
}

function resolveExpectancyGateMode(
    cfg: TraderConfig,
    symbol: string,
    direction: 'UP' | 'DOWN',
): ExpectancyGateMode {
    const scoped = resolveSymbolDirectionScopedMode(symbol, direction, cfg.expectancyGateModeBySymbolDirection);
    const raw = scoped || cfg.expectancyGateMode || EXPECTANCY_GATE_MODE_DEFAULT;
    if (raw === 'shadow' || raw === 'enforce') return raw;
    return 'off';
}

function resolveExpectancyGateStatsMode(cfg: TraderConfig): ExpectancyGateStatsMode {
    const raw = cfg.expectancyGateStatsMode || EXPECTANCY_GATE_STATS_MODE_DEFAULT;
    if (raw === 'simulation' || raw === 'live' || raw === 'route') return raw;
    return 'route';
}

function resolveExpectancyGateRouteMode(t: TraderInstance): ExpectancyGateRouteMode {
    const statsMode = resolveExpectancyGateStatsMode(t.config);
    if (statsMode === 'simulation' || statsMode === 'live') return statsMode;
    return t.config.appliedTradingMode === 'live' ? 'live' : 'simulation';
}

function getExpectancyGateSymbolDirectionOverrides(
    t: TraderInstance,
    symbol: string,
    direction: 'UP' | 'DOWN',
): Record<string, unknown> {
    const all = t.config.expectancyGateBySymbolDirection || {};
    const keys = [
        symbolDirectionConfigKey(symbol, direction),
        symbolDirectionConfigKey(symbol.toUpperCase(), direction),
        symbolDirectionConfigKey(symbol.toLowerCase(), direction),
        symbolDirectionRuntimeKey(symbol, direction),
        symbolDirectionRuntimeKey(symbol.toUpperCase(), direction),
        symbolDirectionRuntimeKey(symbol.toLowerCase(), direction),
    ];
    for (const key of keys) {
        const hit = all[key];
        if (hit && typeof hit === 'object') return hit as Record<string, unknown>;
    }
    return {};
}

function getExpectancyGateNumber(
    t: TraderInstance,
    symbol: string,
    direction: 'UP' | 'DOWN',
    key: string,
    fallback: number,
): number {
    const overrides = getExpectancyGateSymbolDirectionOverrides(t, symbol, direction);
    const value = overrides[key];
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
}

function getExpectancyGateString(
    t: TraderInstance,
    symbol: string,
    direction: 'UP' | 'DOWN',
    key: string,
): string | undefined {
    const overrides = getExpectancyGateSymbolDirectionOverrides(t, symbol, direction);
    const value = overrides[key];
    if (value == null) return undefined;
    const s = String(value).trim();
    return s ? s : undefined;
}

function getExpectancyGateManualAction(
    t: TraderInstance,
    symbol: string,
    direction: 'UP' | 'DOWN',
): 'normal' | 'degraded' | 'blocked' {
    const raw = (getExpectancyGateString(t, symbol, direction, 'expectancyGateManualAction') || '').toLowerCase();
    if (raw === 'blocked' || raw === 'degraded') return raw;
    return 'normal';
}

function isExpectancyGateTargetSymbol(symbol: string): boolean {
    return EXPECTANCY_GATE_TARGET_SYMBOLS.has(String(symbol || '').toUpperCase());
}

function resolveUncertaintyGateMode(
    cfg: TraderConfig,
    symbol: string,
    direction: 'UP' | 'DOWN',
): 'off' | 'shadow' | 'enforce' {
    const scoped = resolveSymbolDirectionScopedMode(symbol, direction, cfg.uncertaintyGateModeBySymbolDirection);
    const raw = scoped || cfg.uncertaintyGateMode || UNCERTAINTY_GATE_MODE_DEFAULT;
    if (raw === 'shadow' || raw === 'enforce') return raw;
    return 'off';
}

function getUncertaintyGateSymbolDirectionOverrides(
    t: TraderInstance,
    symbol: string,
    direction: 'UP' | 'DOWN',
): Record<string, unknown> {
    const all = t.config.uncertaintyGateBySymbolDirection || {};
    const keys = [
        symbolDirectionConfigKey(symbol, direction),
        symbolDirectionConfigKey(symbol.toUpperCase(), direction),
        symbolDirectionConfigKey(symbol.toLowerCase(), direction),
        symbolDirectionRuntimeKey(symbol, direction),
        symbolDirectionRuntimeKey(symbol.toUpperCase(), direction),
        symbolDirectionRuntimeKey(symbol.toLowerCase(), direction),
    ];
    for (const key of keys) {
        const hit = all[key];
        if (hit && typeof hit === 'object') return hit as Record<string, unknown>;
    }
    return {};
}

function getUncertaintyGateNumber(
    t: TraderInstance,
    symbol: string,
    direction: 'UP' | 'DOWN',
    key: string,
    fallback: number,
): number {
    const overrides = getUncertaintyGateSymbolDirectionOverrides(t, symbol, direction);
    const value = overrides[key];
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
}

function getUncertaintyGateManualAction(
    t: TraderInstance,
    symbol: string,
    direction: 'UP' | 'DOWN',
): 'normal' | 'degraded' | 'blocked' {
    const overrides = getUncertaintyGateSymbolDirectionOverrides(t, symbol, direction);
    const raw = String(overrides.uncertaintyManualAction ?? '').trim().toLowerCase();
    if (raw === 'blocked' || raw === 'degraded') return raw;
    return 'normal';
}

function resolveThresholdDriftMode(
    cfg: TraderConfig,
    symbol: string,
    direction: 'UP' | 'DOWN',
): ThresholdDriftMode {
    const scoped = resolveSymbolDirectionScopedMode(symbol, direction, cfg.thresholdDriftModeBySymbolDirection);
    const raw = scoped || cfg.thresholdDriftMode || THRESHOLD_DRIFT_MODE_DEFAULT;
    if (raw === 'shadow' || raw === 'enforce') return raw;
    return 'off';
}

function resolveThresholdDriftStatsMode(cfg: TraderConfig): ThresholdDriftStatsMode {
    const raw = cfg.thresholdDriftStatsMode || THRESHOLD_DRIFT_STATS_MODE_DEFAULT;
    if (raw === 'simulation' || raw === 'live' || raw === 'route') return raw;
    return 'route';
}

function resolveThresholdDriftRouteMode(t: TraderInstance): ThresholdDriftRouteMode {
    const statsMode = resolveThresholdDriftStatsMode(t.config);
    if (statsMode === 'simulation' || statsMode === 'live') return statsMode;
    return t.config.appliedTradingMode === 'live' ? 'live' : 'simulation';
}

function getThresholdDriftSymbolDirectionOverrides(
    t: TraderInstance,
    symbol: string,
    direction: 'UP' | 'DOWN',
): Record<string, unknown> {
    const all = t.config.thresholdDriftBySymbolDirection || {};
    const keys = [
        symbolDirectionConfigKey(symbol, direction),
        symbolDirectionConfigKey(symbol.toUpperCase(), direction),
        symbolDirectionConfigKey(symbol.toLowerCase(), direction),
        symbolDirectionRuntimeKey(symbol, direction),
        symbolDirectionRuntimeKey(symbol.toUpperCase(), direction),
        symbolDirectionRuntimeKey(symbol.toLowerCase(), direction),
    ];
    for (const key of keys) {
        const hit = all[key];
        if (hit && typeof hit === 'object') return hit as Record<string, unknown>;
    }
    return {};
}

function getThresholdDriftNumber(
    t: TraderInstance,
    symbol: string,
    direction: 'UP' | 'DOWN',
    key: string,
    fallback: number,
): number {
    const overrides = getThresholdDriftSymbolDirectionOverrides(t, symbol, direction);
    const value = overrides[key];
    const n = Number(value);
    return Number.isFinite(n) ? n : fallback;
}

function isThresholdDriftTargetSymbol(symbol: string): boolean {
    return THRESHOLD_DRIFT_TARGET_SYMBOLS.has(String(symbol || '').toUpperCase());
}

function resolveMetaLabelMode(
    cfg: TraderConfig,
    symbol: string,
    direction: 'UP' | 'DOWN',
): MetaLabelMode {
    const scoped = resolveSymbolDirectionScopedMode(symbol, direction, cfg.metaLabelModeBySymbolDirection);
    const raw = scoped || cfg.metaLabelMode || META_LABEL_MODE_DEFAULT;
    if (raw === 'shadow' || raw === 'enforce') return raw;
    return 'off';
}

function resolveSelectorMode(cfg: TraderConfig, symbol: string): SelectorMode {
    const scoped = resolveSymbolScopedMode<SelectorMode>(symbol, cfg.selectorModeBySymbol);
    const raw = scoped || cfg.selectorMode || SELECTOR_MODE_DEFAULT;
    if (raw === 'shadow' || raw === 'enforce') return raw;
    return 'off';
}

function resolveSelectorStatsMode(cfg: TraderConfig): SelectorStatsMode {
    const raw = cfg.selectorStatsMode || SELECTOR_STATS_MODE_DEFAULT;
    if (raw === 'simulation' || raw === 'live' || raw === 'route') return raw;
    return 'route';
}

function resolveSelectorRouteMode(t: TraderInstance): SelectorRouteMode {
    const statsMode = resolveSelectorStatsMode(t.config);
    if (statsMode === 'simulation' || statsMode === 'live') return statsMode;
    return t.config.appliedTradingMode === 'live' ? 'live' : 'simulation';
}

function isSelectorTargetSymbol(symbol: string): boolean {
    return SELECTOR_TARGET_SYMBOLS.has(String(symbol || '').toUpperCase());
}

function selectorNotApplicableState(
    t: TraderInstance,
    symbol: string,
    mode: SelectorRouteMode,
): SelectorCellState {
    return {
        key: selectorCellKey(t.config.profile ?? 'default', t.config.name, symbol, mode),
        profile: t.config.profile ?? 'default',
        traderName: t.config.name,
        symbol: symbol.toUpperCase(),
        mode,
        sourceMode: mode,
        checkedAtMs: 0,
        lastAction: 'n_a_by_design',
        reasonCode: 'symbol_not_target',
        status: 'n_a_by_design',
        eligible: true,
        scope: 'unknown',
    };
}

function resolveRegimeMode(cfg: TraderConfig, symbol: string): RegimeMode {
    const scoped = resolveSymbolScopedMode<RegimeMode>(symbol, cfg.regimeModeBySymbol);
    const raw = scoped || cfg.regimeMode || REGIME_MODE_DEFAULT;
    if (raw === 'shadow' || raw === 'enforce') return raw;
    return 'off';
}

function resolveRegimeStatsMode(cfg: TraderConfig): RegimeStatsMode {
    const raw = cfg.regimeStatsMode || REGIME_STATS_MODE_DEFAULT;
    if (raw === 'simulation' || raw === 'live' || raw === 'route') return raw;
    return 'route';
}

function resolveRegimeRouteMode(t: TraderInstance): RegimeRouteMode {
    const statsMode = resolveRegimeStatsMode(t.config);
    if (statsMode === 'simulation' || statsMode === 'live') return statsMode;
    return t.config.appliedTradingMode === 'live' ? 'live' : 'simulation';
}

function isRegimeTargetSymbol(symbol: string): boolean {
    return REGIME_TARGET_SYMBOLS.has(String(symbol || '').toUpperCase());
}

function upRiskCellKey(
    profile: 'default' | '70',
    traderName: string,
    symbol: string,
    mode: UpRiskRouteMode,
): UpRiskCellKey {
    return `${profile}::${traderName}::${symbol.toUpperCase()}::${mode}`;
}

function getOrInitUpRiskCell(
    t: TraderInstance,
    symbol: string,
    mode: UpRiskRouteMode,
): UpRiskCellState {
    const profile = t.config.profile ?? 'default';
    const key = upRiskCellKey(profile, t.config.name, symbol, mode);
    const existing = t.upRiskCells.get(key);
    if (existing) return existing;
    const lookbackMain = Math.max(1, Number(t.config.upRiskLookbackHoursMain ?? UP_RISK_LOOKBACK_MAIN_HOURS_DEFAULT));
    const lookbackCalib = Math.max(lookbackMain, Number(t.config.upRiskLookbackHoursCalib ?? UP_RISK_LOOKBACK_CALIB_HOURS_DEFAULT));
    const initWindow = (lookbackHours: number): UpRiskWindowStats => ({
        lookbackHours,
        trades: 0,
        wins: 0,
        losses: 0,
        winRate: 1,
        wrPost: 0.5,
        pnl: 0,
        pnlPerTrade: 0,
    });
    const state: UpRiskCellState = {
        key,
        profile,
        traderName: t.config.name,
        symbol: symbol.toUpperCase(),
        mode,
        tier: 'normal',
        releasePassStreak: 0,
        checkedAtMs: 0,
        lastAction: 'init',
        reasonCode: 'init',
        blocked: false,
        upScale: 1,
        extraDelta: 0,
        sourceMode: mode,
        main: initWindow(lookbackMain),
        calib: initWindow(lookbackCalib),
    };
    t.upRiskCells.set(key, state);
    return state;
}

function downRiskCellKey(
    profile: 'default' | '70',
    traderName: string,
    symbol: string,
    mode: DownRiskRouteMode,
): DownRiskCellKey {
    return `${profile}::${traderName}::${symbol.toUpperCase()}::${mode}`;
}

function getOrInitDownRiskCell(
    t: TraderInstance,
    symbol: string,
    mode: DownRiskRouteMode,
): DownRiskCellState {
    const profile = t.config.profile ?? 'default';
    const key = downRiskCellKey(profile, t.config.name, symbol, mode);
    const existing = t.downRiskCells.get(key);
    if (existing) return existing;
    const lookbackMain = Math.max(1, Number(t.config.downRiskLookbackHoursMain ?? DOWN_RISK_LOOKBACK_MAIN_HOURS_DEFAULT));
    const lookbackFast = Math.max(1, Number(t.config.downRiskLookbackHoursFast ?? DOWN_RISK_LOOKBACK_FAST_HOURS_DEFAULT));
    const lookbackCalib = Math.max(lookbackMain, Number(t.config.downRiskLookbackHoursCalib ?? DOWN_RISK_LOOKBACK_CALIB_HOURS_DEFAULT));
    const releaseConfirmHours = Math.max(1, Number(t.config.downRiskReleaseConfirmHours ?? DOWN_RISK_RELEASE_CONFIRM_HOURS_DEFAULT));
    const initWindow = (lookbackHours: number): DownRiskWindowStats => ({
        lookbackHours,
        trades: 0,
        wins: 0,
        losses: 0,
        winRate: 1,
        wrPost: 0.5,
        pnl: 0,
        pnlPerTrade: 0,
    });
    const state: DownRiskCellState = {
        key,
        profile,
        traderName: t.config.name,
        symbol: symbol.toUpperCase(),
        mode,
        tier: 'normal',
        hardUntilTs: 0,
        releasePassStreak: 0,
        checkedAtMs: 0,
        lastAction: 'init',
        reasonCode: 'init',
        triggerLane: 'normal',
        enteredBy: 'init',
        lastTransitionAtMs: 0,
        blocked: false,
        downScale: 1,
        extraDelta: 0,
        releaseWindowMainPass: false,
        releaseWindowConfirmPass: false,
        sourceMode: mode,
        main: initWindow(lookbackMain),
        fast: initWindow(lookbackFast),
        calib: initWindow(lookbackCalib),
        releaseConfirm: initWindow(releaseConfirmHours),
    };
    t.downRiskCells.set(key, state);
    return state;
}

function shockRiskCellKey(
    profile: 'default' | '70',
    traderName: string,
    symbol: string,
    direction: 'UP' | 'DOWN',
    mode: ShockRiskRouteMode,
): ShockRiskCellKey {
    return `${profile}::${traderName}::${symbol.toUpperCase()}::${direction}::${mode}`;
}

function getOrInitShockRiskCell(
    t: TraderInstance,
    symbol: string,
    direction: 'UP' | 'DOWN',
    mode: ShockRiskRouteMode,
): ShockRiskCellState {
    const profile = t.config.profile ?? 'default';
    const key = shockRiskCellKey(profile, t.config.name, symbol, direction, mode);
    const existing = t.shockRiskCells.get(key);
    if (existing) return existing;
    const initWindow = (lookbackMinutes: number): ShockRiskWindowStats => ({
        lookbackMinutes,
        trades: 0,
        wins: 0,
        losses: 0,
        winRate: 1,
        wrPost: 0.5,
        pnl: 0,
        pnlPerTrade: 0,
        avgAbsPnl: 0,
    });
    const state: ShockRiskCellState = {
        key,
        profile,
        traderName: t.config.name,
        symbol: symbol.toUpperCase(),
        direction,
        mode,
        active: false,
        activeUntilTs: 0,
        releasePassStreak: 0,
        checkedAtMs: 0,
        lastAction: 'init',
        reasonCode: 'init',
        triggerLane: 'normal',
        lastTransitionAtMs: 0,
        blocked: false,
        sourceMode: mode,
        vol1h: 0,
        vol4h: 0,
        vol24h: 0,
        volRatio: 0,
        volPctl: 0,
        main: initWindow(Math.max(1, Number(t.config.shockRiskWindowMinutes ?? SHOCK_RISK_WINDOW_MINUTES_DEFAULT))),
    };
    t.shockRiskCells.set(key, state);
    return state;
}

function comboPauseCellKey(
    profile: 'default' | '70',
    traderName: string,
    symbol: string,
    direction: 'UP' | 'DOWN' | 'BOTH',
    mode: ComboPauseRouteMode,
): ComboPauseCellKey {
    return `${profile}::${traderName}::${symbol.toUpperCase()}::${direction}::${mode}`;
}

function getOrInitComboPauseCell(
    t: TraderInstance,
    symbol: string,
    direction: 'UP' | 'DOWN' | 'BOTH',
    mode: ComboPauseRouteMode,
): ComboPauseCellState {
    const profile = t.config.profile ?? 'default';
    const key = comboPauseCellKey(profile, t.config.name, symbol, direction, mode);
    const existing = t.comboPauseCells.get(key);
    if (existing) return existing;
    const state: ComboPauseCellState = {
        key,
        profile,
        traderName: t.config.name,
        symbol: symbol.toUpperCase(),
        direction,
        mode,
        active: false,
        activeUntilTs: 0,
        checkedAtMs: 0,
        lastAction: 'init',
        reasonCode: 'init',
        triggerLane: direction === 'BOTH' ? 'symbol' : 'symbol_direction',
        lastTransitionAtMs: 0,
        blocked: false,
        sourceMode: mode,
        drawdown2h: 0,
        pnl2h: 0,
        baseCapital: 0,
        sampleCount: 0,
    };
    t.comboPauseCells.set(key, state);
    return state;
}

function calibrationCellKey(
    profile: 'default' | '70',
    traderName: string,
    symbol: string,
    direction: 'UP' | 'DOWN',
    mode: CalibrationRouteMode,
): CalibrationCellKey {
    return `${profile}::${traderName}::${symbol.toUpperCase()}::${direction}::${mode}`;
}

function getOrInitCalibrationCell(
    t: TraderInstance,
    symbol: string,
    direction: 'UP' | 'DOWN',
    mode: CalibrationRouteMode,
): CalibrationCellState {
    const profile = t.config.profile ?? 'default';
    const key = calibrationCellKey(profile, t.config.name, symbol, direction, mode);
    const existing = t.calibrationCells.get(key);
    if (existing) return existing;
    const lookbackHours = Math.max(1, Number(t.config.calibrationLookbackHours ?? CALIBRATION_LOOKBACK_HOURS_DEFAULT));
    const state: CalibrationCellState = {
        key,
        profile,
        traderName: t.config.name,
        symbol: symbol.toUpperCase(),
        direction,
        mode,
        sourceMode: mode,
        method: getCalibrationMethod(t, symbol, direction),
        checkedAtMs: 0,
        lastAction: 'init',
        reasonCode: 'init',
        active: false,
        minSamples: Math.max(1, Number(t.config.calibrationMinSamples ?? CALIBRATION_MIN_SAMPLES_DEFAULT)),
        lastConfidence: 0,
        calibratedConfidence: 0,
        stats: {
            lookbackHours,
            sampleCount: 0,
            wins: 0,
            losses: 0,
            winRate: 1,
            avgConfidence: 0,
            brier: 0,
            ece: 0,
            calibrationGap: 0,
        },
    };
    t.calibrationCells.set(key, state);
    return state;
}

function expectancyGateCellKey(
    profile: 'default' | '70',
    traderName: string,
    symbol: string,
    direction: 'UP' | 'DOWN',
    mode: ExpectancyGateRouteMode,
): ExpectancyGateCellKey {
    return `${profile}::${traderName}::${symbol.toUpperCase()}::${direction}::${mode}`;
}

function getOrInitExpectancyGateCell(
    t: TraderInstance,
    symbol: string,
    direction: 'UP' | 'DOWN',
    mode: ExpectancyGateRouteMode,
): ExpectancyGateCellState {
    const profile = t.config.profile ?? 'default';
    const key = expectancyGateCellKey(profile, t.config.name, symbol, direction, mode);
    const existing = t.expectancyGateCells.get(key);
    if (existing) return existing;
    const lookbackHours = Math.max(1, Number(t.config.expectancyGateLookbackHours ?? EXPECTANCY_GATE_LOOKBACK_HOURS_DEFAULT));
    const state: ExpectancyGateCellState = {
        key,
        profile,
        traderName: t.config.name,
        symbol: symbol.toUpperCase(),
        direction,
        mode,
        sourceMode: mode,
        checkedAtMs: 0,
        lastAction: 'init',
        reasonCode: 'init',
        triggerLane: 'normal',
        lastTransitionAtMs: 0,
        status: 'normal',
        blocked: false,
        degradedBetScale: 1,
        degradedExtraDelta: 0,
        activeUntilTs: 0,
        blockedEnterPassStreak: 0,
        releasePassStreak: 0,
        stats: {
            lookbackHours,
            trades: 0,
            wins: 0,
            losses: 0,
            winRate: 1,
            wrPost: 0.5,
            pnl: 0,
            pnlPerTrade: 0,
            netPnL: 0,
        },
    };
    t.expectancyGateCells.set(key, state);
    return state;
}

function thresholdDriftCellKey(
    profile: 'default' | '70',
    traderName: string,
    symbol: string,
    direction: 'UP' | 'DOWN',
    mode: ThresholdDriftRouteMode,
): ThresholdDriftCellKey {
    return `${profile}::${traderName}::${symbol.toUpperCase()}::${direction}::${mode}`;
}

function getOrInitThresholdDriftCell(
    t: TraderInstance,
    symbol: string,
    direction: 'UP' | 'DOWN',
    mode: ThresholdDriftRouteMode,
): ThresholdDriftCellState {
    const profile = t.config.profile ?? 'default';
    const key = thresholdDriftCellKey(profile, t.config.name, symbol, direction, mode);
    const existing = t.thresholdDriftCells.get(key);
    if (existing) return existing;
    const lookbackHours = Math.max(1, Number(t.config.thresholdDriftLookbackHours ?? THRESHOLD_DRIFT_LOOKBACK_HOURS_DEFAULT));
    const state: ThresholdDriftCellState = {
        key,
        profile,
        traderName: t.config.name,
        symbol: symbol.toUpperCase(),
        direction,
        mode,
        sourceMode: mode,
        checkedAtMs: 0,
        lastAction: 'init',
        reasonCode: 'init',
        status: 'normal',
        active: false,
        thresholdDelta: 0,
        releasePassStreak: 0,
        stats: {
            lookbackHours,
            trades: 0,
            wins: 0,
            losses: 0,
            winRate: 1,
            wrPost: 0.5,
            pnl: 0,
            pnlPerTrade: 0,
        },
    };
    t.thresholdDriftCells.set(key, state);
    return state;
}

function metaLabelCellKey(
    profile: 'default' | '70',
    traderName: string,
    symbol: string,
    direction: 'UP' | 'DOWN',
): MetaLabelCellKey {
    return `${profile}::${traderName}::${symbol.toUpperCase()}::${direction}`;
}

function getOrInitMetaLabelCell(
    t: TraderInstance,
    symbol: string,
    direction: 'UP' | 'DOWN',
): MetaLabelCellState {
    const profile = t.config.profile ?? 'default';
    const key = metaLabelCellKey(profile, t.config.name, symbol, direction);
    const existing = t.metaLabelCells.get(key);
    if (existing) return existing;
    const state: MetaLabelCellState = {
        key,
        profile,
        traderName: t.config.name,
        symbol: symbol.toUpperCase(),
        direction,
        mode: resolveMetaLabelMode(t.config, symbol, direction),
        checkedAtMs: 0,
        lastAction: 'init',
        reasonCode: 'init',
        takeTrade: true,
        metaConfidence: null,
        evaluated: 0,
        passed: 0,
        blocked: 0,
    };
    t.metaLabelCells.set(key, state);
    return state;
}

function selectorCellKey(
    profile: 'default' | '70',
    traderName: string,
    symbol: string,
    mode: SelectorRouteMode,
): SelectorCellKey {
    return `${profile}::${traderName}::${symbol.toUpperCase()}::${mode}`;
}

function getOrInitSelectorCell(
    t: TraderInstance,
    symbol: string,
    mode: SelectorRouteMode,
): SelectorCellState {
    const profile = t.config.profile ?? 'default';
    const key = selectorCellKey(profile, t.config.name, symbol, mode);
    const existing = t.selectorCells.get(key);
    if (existing) return existing;
    const state: SelectorCellState = {
        key,
        profile,
        traderName: t.config.name,
        symbol: symbol.toUpperCase(),
        mode,
        sourceMode: mode,
        checkedAtMs: 0,
        lastAction: 'init',
        reasonCode: 'init',
        status: 'unknown',
        eligible: true,
        scope: 'unknown',
    };
    t.selectorCells.set(key, state);
    return state;
}

function calcThresholdDriftWindowStats(
    entries: Record<string, unknown>[],
    symbol: string,
    direction: 'UP' | 'DOWN',
    mode: ThresholdDriftRouteMode,
    cutoffMs: number,
    lookbackHours: number,
): ThresholdDriftWindowStats {
    const stats = calcExpectancyGateWindowStats(entries, symbol, direction, mode, cutoffMs, lookbackHours);
    return {
        lookbackHours: stats.lookbackHours,
        trades: stats.trades,
        wins: stats.wins,
        losses: stats.losses,
        winRate: stats.winRate,
        wrPost: stats.wrPost,
        pnl: stats.pnl,
        pnlPerTrade: stats.pnlPerTrade,
    };
}

function calcCalibrationWindowStats(
    entries: Record<string, unknown>[],
    symbol: string,
    direction: 'UP' | 'DOWN',
    mode: CalibrationRouteMode,
    cutoffMs: number,
    lookbackHours: number,
): CalibrationWindowStats {
    const seen = new Set<string>();
    let sampleCount = 0;
    let wins = 0;
    let losses = 0;
    let confidenceSum = 0;
    let brierSum = 0;
    let calibrationGapSum = 0;
    for (let i = entries.length - 1; i >= 0; i -= 1) {
        const e = entries[i];
        if (normalizeDirection(e.direction) !== direction) continue;
        if (normalizeSymbol(e.symbol) !== symbol) continue;
        const result = String(e.result || '').toLowerCase();
        if (result !== 'win' && result !== 'lose') continue;
        const settledMs = parseSettledMs(e);
        if (settledMs == null || settledMs < cutoffMs) continue;
        const uniq = calibrationSampleKey(e, symbol, direction, mode);
        if (seen.has(uniq)) continue;
        seen.add(uniq);
        const confidence = Number(e.calibratedConfidence ?? e.confidence);
        if (!Number.isFinite(confidence)) continue;
        const conf = Math.max(0, Math.min(1, confidence));
        const target = result === 'win' ? 1 : 0;
        sampleCount += 1;
        if (result === 'win') wins += 1;
        else losses += 1;
        confidenceSum += conf;
        brierSum += (conf - target) ** 2;
        calibrationGapSum += Math.abs(conf - target);
    }
    const winRate = sampleCount > 0 ? wins / sampleCount : 1;
    const avgConfidence = sampleCount > 0 ? confidenceSum / sampleCount : 0;
    const brier = sampleCount > 0 ? brierSum / sampleCount : 0;
    const calibrationGap = sampleCount > 0 ? calibrationGapSum / sampleCount : 0;
    const ece = Math.abs(avgConfidence - winRate);
    return {
        lookbackHours,
        sampleCount,
        wins,
        losses,
        winRate,
        avgConfidence,
        brier,
        ece,
        calibrationGap,
    };
}

function calcExpectancyGateWindowStats(
    entries: Record<string, unknown>[],
    symbol: string,
    direction: 'UP' | 'DOWN',
    mode: ExpectancyGateRouteMode,
    cutoffMs: number,
    lookbackHours: number,
): ExpectancyGateWindowStats {
    const seen = new Set<string>();
    let trades = 0;
    let wins = 0;
    let losses = 0;
    let pnl = 0;
    for (let i = entries.length - 1; i >= 0; i -= 1) {
        const e = entries[i];
        if (normalizeDirection(e.direction) !== direction) continue;
        if (normalizeSymbol(e.symbol) !== symbol) continue;
        const result = String(e.result || '').toLowerCase();
        if (result !== 'win' && result !== 'lose') continue;
        const settledMs = parseSettledMs(e);
        if (settledMs == null || settledMs < cutoffMs) continue;
        const uniq = expectancyGateSampleKey(e, symbol, direction, mode);
        if (seen.has(uniq)) continue;
        seen.add(uniq);
        trades += 1;
        if (result === 'win') wins += 1;
        else losses += 1;
        pnl += extractNumericPnl(e);
    }
    const winRate = trades > 0 ? wins / trades : 1;
    const wrPost = (wins + 2) / (trades + 4);
    const pnlPerTrade = trades > 0 ? pnl / trades : 0;
    return {
        lookbackHours,
        trades,
        wins,
        losses,
        winRate,
        wrPost,
        pnl,
        pnlPerTrade,
        netPnL: pnl,
    };
}

function applyCalibrationMethodToConfidence(
    confidence: number,
    t: TraderInstance,
    symbol: string,
    direction: 'UP' | 'DOWN',
): number {
    const method = getCalibrationMethod(t, symbol, direction);
    const x = Math.max(1e-6, Math.min(1 - 1e-6, confidence));
    if (method === 'temperature') {
        const temperature = Math.max(0.05, getCalibrationNumber(t, symbol, direction, 'calibrationTemperature', CALIBRATION_TEMPERATURE_DEFAULT));
        const logit = Math.log(x / (1 - x));
        return 1 / (1 + Math.exp(-logit / temperature));
    }
    if (method === 'sigmoid') {
        const alpha = getCalibrationNumber(t, symbol, direction, 'calibrationAlpha', CALIBRATION_ALPHA_DEFAULT);
        const beta = getCalibrationNumber(t, symbol, direction, 'calibrationBeta', CALIBRATION_BETA_DEFAULT);
        const logit = Math.log(x / (1 - x));
        return 1 / (1 + Math.exp(-(alpha + beta * logit)));
    }
    if (method === 'isotonic') {
        const xs = getCalibrationArray(t, symbol, direction, 'calibrationIsotonicX');
        const ys = getCalibrationArray(t, symbol, direction, 'calibrationIsotonicY');
        if (xs.length >= 2 && xs.length === ys.length) {
            const sorted = xs.map((vx, idx) => ({ x: vx, y: ys[idx] }))
                .sort((a, b) => a.x - b.x);
            if (x <= sorted[0].x) return Math.max(0, Math.min(1, sorted[0].y));
            if (x >= sorted[sorted.length - 1].x) return Math.max(0, Math.min(1, sorted[sorted.length - 1].y));
            for (let i = 1; i < sorted.length; i += 1) {
                const left = sorted[i - 1];
                const right = sorted[i];
                if (x <= right.x) {
                    const span = Math.max(1e-9, right.x - left.x);
                    const w = (x - left.x) / span;
                    return Math.max(0, Math.min(1, left.y + (right.y - left.y) * w));
                }
            }
        }
    }
    return x;
}

function readTradeEntriesForRoute(
    logsDir: string,
    routeMode: 'simulation' | 'live',
): {
    ok: boolean;
    entries: Record<string, unknown>[];
    mtimeMs: number;
    reason: string;
} {
    const tradesPath = path.join(process.cwd(), logsDir, `prediction_trades.${routeMode}.json`);
    if (!fs.existsSync(tradesPath)) {
        return { ok: false, entries: [], mtimeMs: 0, reason: 'no_data' };
    }
    let mtimeMs = 0;
    try {
        mtimeMs = fs.statSync(tradesPath).mtimeMs;
    } catch {
        return { ok: false, entries: [], mtimeMs: 0, reason: 'stat_error' };
    }
    try {
        const raw = JSON.parse(fs.readFileSync(tradesPath, 'utf8')) as unknown;
        if (!Array.isArray(raw)) {
            return { ok: false, entries: [], mtimeMs, reason: 'invalid_json_shape' };
        }
        return { ok: true, entries: raw as Record<string, unknown>[], mtimeMs, reason: 'ok' };
    } catch {
        return { ok: false, entries: [], mtimeMs, reason: 'parse_error' };
    }
}

const _mainlineExtremeCache: {
    lastCheckMs: number;
    lastMtimeMs: number;
    rowsBySymbol: Record<string, Record<string, unknown>>;
} = {
    lastCheckMs: 0,
    lastMtimeMs: 0,
    rowsBySymbol: {},
};

function readMainlineExtremeRows(nowMs: number): Record<string, Record<string, unknown>> {
    if (nowMs - _mainlineExtremeCache.lastCheckMs < MAINLINE_EXTREME_REFRESH_MS) {
        return _mainlineExtremeCache.rowsBySymbol;
    }
    _mainlineExtremeCache.lastCheckMs = nowMs;
    let mtimeMs = 0;
    try {
        mtimeMs = fs.existsSync(MAINLINE_EXTREME_STATE_FILE) ? fs.statSync(MAINLINE_EXTREME_STATE_FILE).mtimeMs : 0;
    } catch {
        _mainlineExtremeCache.rowsBySymbol = {};
        _mainlineExtremeCache.lastMtimeMs = 0;
        return _mainlineExtremeCache.rowsBySymbol;
    }
    if (mtimeMs <= 0) {
        _mainlineExtremeCache.rowsBySymbol = {};
        _mainlineExtremeCache.lastMtimeMs = 0;
        return _mainlineExtremeCache.rowsBySymbol;
    }
    if (mtimeMs === _mainlineExtremeCache.lastMtimeMs) {
        return _mainlineExtremeCache.rowsBySymbol;
    }
    try {
        const payload = JSON.parse(fs.readFileSync(MAINLINE_EXTREME_STATE_FILE, 'utf8')) as Record<string, unknown>;
        const rows = Array.isArray(payload.rows) ? payload.rows : [];
        const rowsBySymbol: Record<string, Record<string, unknown>> = {};
        for (const row of rows) {
            if (!row || typeof row !== 'object') continue;
            const symbol = normalizeSymbol((row as Record<string, unknown>).symbol);
            if (!symbol || rowsBySymbol[symbol]) continue;
            rowsBySymbol[symbol] = row as Record<string, unknown>;
        }
        _mainlineExtremeCache.rowsBySymbol = rowsBySymbol;
        _mainlineExtremeCache.lastMtimeMs = mtimeMs;
    } catch {
        _mainlineExtremeCache.rowsBySymbol = {};
        _mainlineExtremeCache.lastMtimeMs = mtimeMs;
    }
    return _mainlineExtremeCache.rowsBySymbol;
}

function tradeSampleKey(entry: Record<string, unknown>, symbol: string, direction?: 'UP' | 'DOWN' | null): string {
    const conditionId = String(entry.conditionId || '').trim();
    const resolvedDirection = direction || normalizeDirection(entry.direction);
    if (conditionId) {
        return `${conditionId}::${symbol}::${resolvedDirection || 'ANY'}`;
    }
    const marketSlug = String(entry.marketSlug || '').trim();
    const settledAt = String((entry.settledAt as string) || (entry.timestamp as string) || '').trim();
    const entryId = String(entry.id || '').trim();
    return `${marketSlug}::${settledAt}::${entryId}::${symbol}::${resolvedDirection || 'ANY'}`;
}

type RecentSettledTrade = {
    key: string;
    symbol: string;
    direction: 'UP' | 'DOWN';
    result: 'win' | 'lose';
    pnl: number;
    confidence: number;
    settledMs: number;
};

function collectRecentSettledTrades(
    entries: Record<string, unknown>[],
    symbol: string,
    direction: 'UP' | 'DOWN' | null,
    limit: number,
): RecentSettledTrade[] {
    const out: RecentSettledTrade[] = [];
    const seen = new Set<string>();
    for (let i = entries.length - 1; i >= 0; i -= 1) {
        const e = entries[i];
        const entryDirection = normalizeDirection(e.direction);
        if (!entryDirection) continue;
        if (direction && entryDirection !== direction) continue;
        if (normalizeSymbol(e.symbol) !== symbol) continue;
        const result = String(e.result || '').toLowerCase();
        if (result !== 'win' && result !== 'lose') continue;
        const settledMs = parseSettledMs(e);
        if (settledMs == null) continue;
        const key = tradeSampleKey(e, symbol, entryDirection);
        if (seen.has(key)) continue;
        seen.add(key);
        out.push({
            key,
            symbol,
            direction: entryDirection,
            result,
            pnl: extractNumericPnl(e),
            confidence: Number.isFinite(Number(e.confidence)) ? Number(e.confidence) : 0,
            settledMs,
        });
        if (out.length >= limit) break;
    }
    return out;
}

function maybeUpdateMainlineExtremeRuntime(t: TraderInstance, nowMs: number): void {
    const profile = t.config.profile ?? inferProfileFromGroup(t.config.group);
    const rowsBySymbol = readMainlineExtremeRows(nowMs);
    for (const symbol of parseAllowedSymbols(t.config.allowedMarkets || '')) {
        const mode = resolveMainlineExtremeMode(t.config, symbol);
        const row = rowsBySymbol[symbol] || {};
        const observedActive = Boolean(row.active);
        const active = mode === 'enforce' && observedActive;
        const regimeId = String(row.regimeId || '');
        const laneState = String(row.laneState || '');
        const directionPolicyRaw = String(row.directionPolicy || 'BOTH').toUpperCase();
        const directionPolicy = directionPolicyRaw === 'UP' || directionPolicyRaw === 'DOWN' || directionPolicyRaw === 'NONE'
            ? directionPolicyRaw
            : 'BOTH';
        const hardExtreme = regimeId === 'extreme_up' || regimeId === 'extreme_down';
        const rallyLike = laneState === 'rally' || regimeId === 'rally_up' || regimeId === 'rally_down';
        const baseExtremeLike = laneState === 'base' || laneState === 'high_vol_chop';
        const thresholdExtraDelta = active
            ? (hardExtreme ? MAINLINE_EXTREME_HARD_COUNTERTREND_DELTA : (rallyLike ? MAINLINE_EXTREME_RALLY_COUNTERTREND_DELTA : 0))
            : 0;
        const betScale = active
            ? (
                hardExtreme
                    ? MAINLINE_EXTREME_HARD_BET_SCALE
                    : (rallyLike
                        ? MAINLINE_EXTREME_RALLY_BET_SCALE
                        : (baseExtremeLike ? MAINLINE_EXTREME_BASE_BET_SCALE : 1))
            )
            : 1;
        const state: MainlineExtremeRuntimeState = {
            symbol,
            profile,
            active,
            laneState,
            regimeId,
            directionPolicy,
            policyMode: String(row.policyMode || '').toLowerCase() === 'one_sided' ? 'one_sided' : 'neutral',
            policyConfidence: Number.isFinite(Number(row.policyConfidence)) ? Number(row.policyConfidence) : 0,
            thresholdExtraDelta,
            betScale,
            reasonCode: (
                mode === 'off'
                    ? 'mainline_extreme_mode_off'
                    : mode === 'shadow'
                        ? 'mainline_extreme_mode_shadow'
                        : String(row.reasonCode || (active ? 'extreme_or_rally_active' : 'ok'))
            ),
            checkedAtMs: nowMs,
        };
        t.mainlineExtremeRuntimeBySymbol[symbol] = state;
        t.mainlineExtremeRuntimeScaleBySymbol[symbol] = betScale;
        for (const direction of ['UP', 'DOWN'] as const) {
            const key = symbolDirectionRuntimeKey(symbol, direction);
            const countertrend = directionPolicy === 'UP' || directionPolicy === 'DOWN'
                ? direction !== directionPolicy
                : false;
            t.mainlineExtremeRuntimeExtraDeltaBySymbolDir[key] = active && countertrend ? thresholdExtraDelta : 0;
        }
    }
}

function maybeUpdateDirectionLossCooldown(t: TraderInstance, nowMs: number): void {
    const routeMode = resolveComboPauseRouteMode(t);
    const tradesData = readTradeEntriesForRoute(t.config.logsDir, routeMode);
    const entries = tradesData.ok ? tradesData.entries : [];
    const nowSec = Math.floor(nowMs / 1000);
    const maxAgeMs = MAINLINE_DIRECTION_LOSS_COOLDOWN_MAX_AGE_MINUTES * 60 * 1000;
    const profile = t.config.profile ?? 'default';
    const shadowBlockSoftScale = 0.70;
    for (const symbol of parseAllowedSymbols(t.config.allowedMarkets || '')) {
        const extremeState = t.mainlineExtremeRuntimeBySymbol[symbol];
        const extremeActive = Boolean(extremeState?.active);
        for (const direction of ['UP', 'DOWN'] as const) {
            const key = symbolDirectionRuntimeKey(symbol, direction);
            const mode = resolveDirectionLossCooldownMode(t.config, symbol, direction);
            if (mode === 'off') {
                t.directionLossCooldownBlockedBySymbolDir[key] = false;
                t.directionLossCooldownSoftScaleBySymbolDir[key] = 1;
                t.directionLossCooldownBySymbolDir[key] = {
                    key,
                    symbol,
                    direction,
                    mode,
                    blocked: false,
                    evidenceFresh: false,
                    softScaled: false,
                    softOverlapWithExtreme: false,
                    softScale: 1,
                    softSeverity: 'none',
                    streakLosses: 0,
                    recentTrades: 0,
                    recentPnl: 0,
                    recentHighConfidenceLosses: 0,
                    recentAvgLossConfidence: 0,
                    untilTs: 0,
                    reasonCode: 'off_mode',
                    softReasonCode: 'off_mode',
                    checkedAtMs: nowMs,
                };
                continue;
            }
            const recent = collectRecentSettledTrades(entries, symbol, direction, Math.max(1, MAINLINE_DIRECTION_LOSS_COOLDOWN_LOOKBACK_TRADES));
            let streakLosses = 0;
            let recentPnl = 0;
            for (const trade of recent) {
                recentPnl += trade.pnl;
                if (trade.result === 'lose') streakLosses += 1;
                else break;
            }
            const recentLosses = recent.filter((trade) => trade.result === 'lose');
            const recentHighConfidenceLosses = recentLosses.filter((trade) => trade.confidence >= MAINLINE_DIRECTION_LOSS_SOFT_MIN_CONFIDENCE).length;
            const recentAvgLossConfidence = recentLosses.length > 0
                ? recentLosses.reduce((sum, trade) => sum + trade.confidence, 0) / recentLosses.length
                : 0;
            const latestSettledMs = recent.length > 0 ? recent[0].settledMs : 0;
            const evidenceFresh = latestSettledMs > 0 && (nowMs - latestSettledMs) <= maxAgeMs;
            const activeTrigger = evidenceFresh && streakLosses >= MAINLINE_DIRECTION_LOSS_COOLDOWN_TRIGGER;
            const triggerKey = recent.length > 0 ? `${recent[0].key}::${streakLosses}` : '';
            const currentUntilTs = Number(t.directionLossCooldownBySymbolDir[key]?.untilTs || 0);
            const lastTriggerKey = String(t.directionLossCooldownLastTriggerKeyBySymbolDir[key] || '');
            let untilTs = currentUntilTs;
            if (activeTrigger && nowSec >= currentUntilTs && triggerKey && triggerKey !== lastTriggerKey) {
                untilTs = nowSec + (MAINLINE_DIRECTION_LOSS_COOLDOWN_MINUTES * 60);
                t.directionLossCooldownLastTriggerKeyBySymbolDir[key] = triggerKey;
            }
            const wouldBlock = untilTs > nowSec;
            const blocked = mode === 'enforce' && wouldBlock;
            t.directionLossCooldownBlockedBySymbolDir[key] = blocked;
            const baseSoftEligible = (
                (!wouldBlock || mode === 'shadow')
                && evidenceFresh
                && streakLosses >= 1
                && recentPnl < 0
                && recentHighConfidenceLosses >= MAINLINE_DIRECTION_LOSS_SOFT_MIN_HIGH_CONF_LOSSES
            );
            const midSoftEligible = (
                baseSoftEligible
                && streakLosses >= MAINLINE_DIRECTION_LOSS_SOFT_MID_MIN_STREAK
                && recentHighConfidenceLosses >= MAINLINE_DIRECTION_LOSS_SOFT_MID_MIN_HIGH_CONF_LOSSES
                && recentAvgLossConfidence >= MAINLINE_DIRECTION_LOSS_SOFT_MID_MIN_AVG_CONFIDENCE
            );
            const strongSoftEligible = (
                baseSoftEligible
                && streakLosses >= MAINLINE_DIRECTION_LOSS_SOFT_STRONG_MIN_STREAK
                && recentHighConfidenceLosses >= MAINLINE_DIRECTION_LOSS_SOFT_STRONG_MIN_HIGH_CONF_LOSSES
                && recentAvgLossConfidence >= MAINLINE_DIRECTION_LOSS_SOFT_STRONG_MIN_AVG_CONFIDENCE
            );
            const softOverlapWithExtreme = (
                profile === 'default'
                && extremeActive
                && (midSoftEligible || strongSoftEligible)
            );
            const baseRuntimeSoft = baseSoftEligible && (!extremeActive || softOverlapWithExtreme);
            const shadowCooldownSoft = mode === 'shadow' && wouldBlock;
            const softScaled = shadowCooldownSoft || baseRuntimeSoft;
            const midSoftScaled = !shadowCooldownSoft && baseRuntimeSoft && midSoftEligible;
            const strongSoftScaled = !shadowCooldownSoft && baseRuntimeSoft && strongSoftEligible;
            let softScale = strongSoftScaled
                ? MAINLINE_DIRECTION_LOSS_SOFT_STRONG_BET_SCALE
                : (midSoftScaled ? MAINLINE_DIRECTION_LOSS_SOFT_MID_BET_SCALE : (baseRuntimeSoft ? MAINLINE_DIRECTION_LOSS_SOFT_BET_SCALE : 1));
            if (shadowCooldownSoft) softScale = Math.min(softScale, shadowBlockSoftScale);
            const softSeverity: 'none' | 'base' | 'mid' | 'strong' = shadowCooldownSoft ? 'strong' : (strongSoftScaled ? 'strong' : (midSoftScaled ? 'mid' : (softScaled ? 'base' : 'none')));
            t.directionLossCooldownSoftScaleBySymbolDir[key] = softScale;
            t.directionLossCooldownBySymbolDir[key] = {
                key,
                symbol,
                direction,
                mode,
                blocked,
                evidenceFresh,
                softScaled,
                softOverlapWithExtreme,
                softScale,
                softSeverity,
                streakLosses,
                recentTrades: recent.length,
                recentPnl,
                recentHighConfidenceLosses,
                recentAvgLossConfidence,
                untilTs: mode === 'shadow' || mode === 'enforce' ? untilTs : 0,
                reasonCode: blocked
                    ? 'same_direction_consecutive_losses'
                    : (shadowCooldownSoft ? 'shadow_same_direction_consecutive_losses' : 'ok'),
                softReasonCode: shadowCooldownSoft
                    ? 'shadow_direction_loss_cooldown_soft_block'
                    : strongSoftScaled
                    ? (softOverlapWithExtreme
                        ? 'high_conf_same_direction_loss_soft_scale_strong_extreme_stack'
                        : 'high_conf_same_direction_loss_soft_scale_strong')
                    : (midSoftScaled
                        ? (softOverlapWithExtreme
                            ? 'high_conf_same_direction_loss_soft_scale_mid_extreme_stack'
                            : 'high_conf_same_direction_loss_soft_scale_mid')
                        : (softScaled ? 'high_conf_same_direction_loss_soft_scale' : 'ok')),
                checkedAtMs: nowMs,
            };
        }
    }
}

function maybeUpdateDrawdownAcceleration(t: TraderInstance, nowMs: number): void {
    const routeMode = resolveComboPauseRouteMode(t);
    const tradesData = readTradeEntriesForRoute(t.config.logsDir, routeMode);
    const entries = tradesData.ok ? tradesData.entries : [];
    const nowSec = Math.floor(nowMs / 1000);
    const maxAgeMs = MAINLINE_DRAWDOWN_ACCEL_MAX_AGE_MINUTES * 60 * 1000;
    const baseCapital = Math.max(1, resolveComboPauseBaseCapital(t));
    const pnlThreshold = -Math.abs(baseCapital * MAINLINE_DRAWDOWN_ACCEL_PCT);
    for (const symbol of parseAllowedSymbols(t.config.allowedMarkets || '')) {
        const drawdownAccelerationMode = resolveDrawdownAccelerationMode(t.config, symbol);
        if (drawdownAccelerationMode === 'off') {
            t.drawdownAccelerationBlockedBySymbol[symbol] = false;
            t.drawdownAccelerationSoftScaleBySymbol[symbol] = 1;
            t.drawdownAccelerationBySymbol[symbol] = {
                symbol,
                mode: drawdownAccelerationMode,
                blocked: false,
                evidenceFresh: false,
                softScaled: false,
                softScale: 1,
                recentTrades: 0,
                recentLosses: 0,
                recentPnl: 0,
                untilTs: 0,
                reasonCode: 'off_mode',
                softReasonCode: 'off_mode',
                checkedAtMs: nowMs,
            };
            continue;
        }
        const recent = collectRecentSettledTrades(entries, symbol, null, Math.max(1, MAINLINE_DRAWDOWN_ACCEL_LOOKBACK_TRADES));
        const recentPnl = recent.reduce((sum, trade) => sum + trade.pnl, 0);
        const recentLosses = recent.filter((trade) => trade.result === 'lose').length;
        const latestSettledMs = recent.length > 0 ? recent[0].settledMs : 0;
        const evidenceFresh = latestSettledMs > 0 && (nowMs - latestSettledMs) <= maxAgeMs;
        const activeTrigger = (
            evidenceFresh
            && recent.length >= MAINLINE_DRAWDOWN_ACCEL_MIN_TRADES
            && recentLosses >= Math.max(2, MAINLINE_DRAWDOWN_ACCEL_MIN_TRADES - 1)
            && recentPnl <= pnlThreshold
        );
        const softLossClusterScaled = (
            !activeTrigger
            && evidenceFresh
            && recent.length >= MAINLINE_DRAWDOWN_ACCEL_MIN_TRADES
            && recentLosses >= Math.max(MAINLINE_DRAWDOWN_ACCEL_SOFT_MIN_RECENT_LOSSES + 1, MAINLINE_DRAWDOWN_ACCEL_MIN_TRADES - 1)
            && recentPnl < 0
        );
        const softEarlyScaled = (
            !activeTrigger
            && evidenceFresh
            && recent.length >= MAINLINE_DRAWDOWN_ACCEL_SOFT_EARLY_MIN_TRADES
            && recentLosses >= MAINLINE_DRAWDOWN_ACCEL_SOFT_MIN_RECENT_LOSSES
            && recentPnl < 0
        );
        const softScaled = (
            !activeTrigger
            && evidenceFresh
            && recent.length >= Math.min(MAINLINE_DRAWDOWN_ACCEL_MIN_TRADES, MAINLINE_DRAWDOWN_ACCEL_SOFT_EARLY_MIN_TRADES)
            && recentLosses >= MAINLINE_DRAWDOWN_ACCEL_SOFT_MIN_RECENT_LOSSES
            && (
                recentPnl <= (pnlThreshold * MAINLINE_DRAWDOWN_ACCEL_SOFT_PNL_FACTOR)
                || softEarlyScaled
                || softLossClusterScaled
            )
        );
        const triggerKey = recent.length > 0 ? `${recent[0].key}::${recentLosses}::${Math.round(recentPnl)}` : '';
        const currentUntilTs = Number(t.drawdownAccelerationBySymbol[symbol]?.untilTs || 0);
        const lastTriggerKey = String(t.drawdownAccelerationLastTriggerKeyBySymbol[symbol] || '');
        let untilTs = currentUntilTs;
        if (activeTrigger && nowSec >= currentUntilTs && triggerKey && triggerKey !== lastTriggerKey) {
            untilTs = nowSec + (MAINLINE_DRAWDOWN_ACCEL_MINUTES * 60);
            t.drawdownAccelerationLastTriggerKeyBySymbol[symbol] = triggerKey;
        }
        const wouldBlock = untilTs > nowSec;
        const modeEnforced = drawdownAccelerationMode === 'enforce';
        const blocked = modeEnforced && wouldBlock;
        const activeSoftScale = modeEnforced && !blocked && softScaled;
        t.drawdownAccelerationBlockedBySymbol[symbol] = blocked;
        t.drawdownAccelerationSoftScaleBySymbol[symbol] = activeSoftScale ? MAINLINE_DRAWDOWN_ACCEL_SOFT_BET_SCALE : 1;
        const softReasonCode = blocked
            ? 'ok'
            : (
                softScaled
                    ? (
                        softLossClusterScaled
                            ? 'drawdown_acceleration_watch_soft_scale_loss_cluster'
                            : (
                                softEarlyScaled
                                    ? 'drawdown_acceleration_watch_soft_scale_early'
                                    : 'drawdown_acceleration_watch_soft_scale'
                            )
                    )
                    : 'ok'
            );
        t.drawdownAccelerationBySymbol[symbol] = {
            symbol,
            mode: drawdownAccelerationMode,
            blocked,
            evidenceFresh,
            softScaled: activeSoftScale,
            softScale: activeSoftScale ? MAINLINE_DRAWDOWN_ACCEL_SOFT_BET_SCALE : 1,
            recentTrades: recent.length,
            recentLosses,
            recentPnl,
            untilTs: modeEnforced ? untilTs : 0,
            reasonCode: blocked ? 'drawdown_acceleration_pause' : (drawdownAccelerationMode === 'shadow' && wouldBlock ? 'shadow_drawdown_acceleration_pause' : 'ok'),
            softReasonCode: drawdownAccelerationMode === 'shadow' && softScaled ? `shadow_${softReasonCode}` : softReasonCode,
            checkedAtMs: nowMs,
        };
    }
}

function calcDownWindowStats(
    entries: Record<string, unknown>[],
    symbol: string,
    mode: DownRiskRouteMode,
    cutoffMs: number,
    lookbackHours: number,
): DownRiskWindowStats {
    let trades = 0;
    let wins = 0;
    let losses = 0;
    let pnl = 0;
    const seen = new Set<string>();
    for (let i = entries.length - 1; i >= 0; i -= 1) {
        const e = entries[i];
        const direction = String(e.direction || '').toUpperCase();
        if (direction !== 'DOWN') continue;
        if (normalizeSymbol(e.symbol) !== symbol) continue;
        const result = String(e.result || '').toLowerCase();
        if (result !== 'win' && result !== 'lose') continue;
        const settledMs = parseSettledMs(e);
        if (settledMs == null || settledMs < cutoffMs) continue;
        const uniq = downRiskSampleKey(e, symbol, mode);
        if (seen.has(uniq)) continue;
        seen.add(uniq);
        trades += 1;
        if (result === 'win') wins += 1;
        else losses += 1;
        pnl += extractNumericPnl(e);
    }
    const winRate = trades > 0 ? wins / trades : 1;
    const pnlPerTrade = trades > 0 ? pnl / trades : 0;
    const wrPost = (wins + 2) / (trades + 4);
    return {
        lookbackHours,
        trades,
        wins,
        losses,
        winRate,
        wrPost,
        pnl,
        pnlPerTrade,
    };
}

function calcShockWindowStats(
    entries: Record<string, unknown>[],
    symbol: string,
    direction: 'UP' | 'DOWN',
    mode: ShockRiskRouteMode,
    cutoffMs: number,
    lookbackMinutes: number,
): ShockRiskWindowStats {
    let trades = 0;
    let wins = 0;
    let losses = 0;
    let pnl = 0;
    let absPnlSum = 0;
    const seen = new Set<string>();
    for (let i = entries.length - 1; i >= 0; i -= 1) {
        const e = entries[i];
        if (normalizeDirection(e.direction) !== direction) continue;
        if (normalizeSymbol(e.symbol) !== symbol) continue;
        const result = String(e.result || '').toLowerCase();
        if (result !== 'win' && result !== 'lose') continue;
        const settledMs = parseSettledMs(e);
        if (settledMs == null || settledMs < cutoffMs) continue;
        const uniq = shockRiskSampleKey(e, symbol, direction, mode);
        if (seen.has(uniq)) continue;
        seen.add(uniq);
        trades += 1;
        if (result === 'win') wins += 1;
        else losses += 1;
        const tradePnl = extractNumericPnl(e);
        pnl += tradePnl;
        absPnlSum += Math.abs(tradePnl);
    }
    const winRate = trades > 0 ? wins / trades : 1;
    const pnlPerTrade = trades > 0 ? pnl / trades : 0;
    const wrPost = (wins + 2) / (trades + 4);
    const avgAbsPnl = trades > 0 ? absPnlSum / trades : 0;
    return {
        lookbackMinutes,
        trades,
        wins,
        losses,
        winRate,
        wrPost,
        pnl,
        pnlPerTrade,
        avgAbsPnl,
    };
}

function collectShockAbsPnlSeries(
    entries: Record<string, unknown>[],
    symbol: string,
    direction: 'UP' | 'DOWN',
    mode: ShockRiskRouteMode,
    cutoffMs: number,
): number[] {
    const out: number[] = [];
    const seen = new Set<string>();
    for (let i = entries.length - 1; i >= 0; i -= 1) {
        const e = entries[i];
        if (normalizeDirection(e.direction) !== direction) continue;
        if (normalizeSymbol(e.symbol) !== symbol) continue;
        const result = String(e.result || '').toLowerCase();
        if (result !== 'win' && result !== 'lose') continue;
        const settledMs = parseSettledMs(e);
        if (settledMs == null || settledMs < cutoffMs) continue;
        const uniq = shockRiskSampleKey(e, symbol, direction, mode);
        if (seen.has(uniq)) continue;
        seen.add(uniq);
        out.push(Math.abs(extractNumericPnl(e)));
    }
    return out;
}

function quantileFromSorted(sortedValues: number[], q: number): number {
    if (!sortedValues.length) return 0;
    const qq = Math.max(0, Math.min(1, q));
    const idx = Math.floor((sortedValues.length - 1) * qq);
    return sortedValues[Math.max(0, Math.min(sortedValues.length - 1, idx))];
}

function resolveComboPauseBaseCapital(t: TraderInstance): number {
    const total = Number(t.config.initialCapital || 0);
    if (!Number.isFinite(total) || total <= 0) return 0;
    if (t.config.perCoinCapital) {
        const n = Math.max(1, Number(t.config.numTradingAssets || parseAllowedSymbols(t.config.allowedMarkets || '').length || 1));
        return total / n;
    }
    return total;
}

function calcComboPauseWindow(
    entries: Record<string, unknown>[],
    symbol: string,
    mode: ComboPauseRouteMode,
    cutoffMs: number,
    direction: 'UP' | 'DOWN' | 'BOTH' = 'BOTH',
): { pnl: number; sampleCount: number } {
    let pnl = 0;
    let sampleCount = 0;
    const seen = new Set<string>();
    for (let i = entries.length - 1; i >= 0; i -= 1) {
        const e = entries[i];
        const entryDirection = normalizeDirection(e.direction);
        if (!entryDirection) continue;
        if (direction !== 'BOTH' && entryDirection !== direction) continue;
        if (normalizeSymbol(e.symbol) !== symbol) continue;
        const result = String(e.result || '').toLowerCase();
        if (result !== 'win' && result !== 'lose') continue;
        const settledMs = parseSettledMs(e);
        if (settledMs == null || settledMs < cutoffMs) continue;
        const uniq = comboPauseSampleKey(e, symbol, mode);
        if (seen.has(uniq)) continue;
        seen.add(uniq);
        sampleCount += 1;
        pnl += extractNumericPnl(e);
    }
    return { pnl, sampleCount };
}

function calcUpWindowStats(
    entries: Record<string, unknown>[],
    symbol: string,
    mode: UpRiskRouteMode,
    cutoffMs: number,
    lookbackHours: number,
): UpRiskWindowStats {
    let trades = 0;
    let wins = 0;
    let losses = 0;
    let pnl = 0;
    const seen = new Set<string>();
    for (let i = entries.length - 1; i >= 0; i -= 1) {
        const e = entries[i];
        const direction = String(e.direction || '').toUpperCase();
        if (direction !== 'UP') continue;
        if (normalizeSymbol(e.symbol) !== symbol) continue;
        const result = String(e.result || '').toLowerCase();
        if (result !== 'win' && result !== 'lose') continue;
        const settledMs = parseSettledMs(e);
        if (settledMs == null || settledMs < cutoffMs) continue;
        const uniq = upRiskSampleKey(e, symbol, mode);
        if (seen.has(uniq)) continue;
        seen.add(uniq);
        trades += 1;
        if (result === 'win') wins += 1;
        else losses += 1;
        pnl += extractNumericPnl(e);
    }
    const winRate = trades > 0 ? wins / trades : 1;
    const pnlPerTrade = trades > 0 ? pnl / trades : 0;
    const wrPost = (wins + 2) / (trades + 4);
    return {
        lookbackHours,
        trades,
        wins,
        losses,
        winRate,
        wrPost,
        pnl,
        pnlPerTrade,
    };
}

function maybeUpdateDownBreakerV1(t: TraderInstance, nowMs: number): void {
    // v1 已退役：不再读取兼容合并账本，也不再产生阻断动作。
    t.downBreakerUntilTs = 0;
    t.downBreakerLastCheckMs = nowMs;
    t.downBreakerStats = {
        checkedAtMs: nowMs,
        lookbackHours: Number(t.config.downBreakerLookbackHours ?? DOWN_BREAKER_LOOKBACK_HOURS_DEFAULT),
        trades: 0,
        wins: 0,
        winRate: 0,
        pnl: 0,
        pnlPerTrade: 0,
        triggered: false,
        reason: 'deprecated_v1_disabled',
    };
}

function maybeUpdateDownRiskV2(t: TraderInstance, nowMs: number): void {
    const routeMode = resolveDownRiskRouteMode(t);
    const symbols = parseAllowedSymbols(t.config.allowedMarkets || '');
    const targetSymbols = symbols.filter((s) => isDownRiskTargetSymbol(s));
    const symbolModes = new Map<string, DownRiskMode>();
    let hasActiveMode = false;
    let hasEnforceMode = false;
    for (const symbol of targetSymbols) {
        const mode = resolveDownRiskMode(t.config, symbol);
        symbolModes.set(symbol, mode);
        if (mode !== 'off') hasActiveMode = true;
        if (mode === 'enforce') hasEnforceMode = true;
    }
    for (const symbol of symbols) {
        t.downRiskRuntimeBlockedBySymbol[symbol] = false;
        t.downRiskRuntimeScaleBySymbol[symbol] = 1;
        t.downRiskRuntimeExtraDeltaBySymbol[symbol] = 0;
    }
    if (targetSymbols.length === 0) {
        return;
    }
    const checkEverySec = Math.max(10, Number(t.config.downRiskCheckSeconds ?? DOWN_RISK_CHECK_SECONDS_DEFAULT));
    const lastCheckMs = t.downRiskLastCheckMsByMode[routeMode] || 0;
    if (nowMs - lastCheckMs < checkEverySec * 1000) return;
    t.downRiskLastCheckMsByMode[routeMode] = nowMs;

    const baseLookbackMainHours = Math.max(1, Number(t.config.downRiskLookbackHoursMain ?? DOWN_RISK_LOOKBACK_MAIN_HOURS_DEFAULT));
    const baseLookbackCalibHours = Math.max(baseLookbackMainHours, Number(t.config.downRiskLookbackHoursCalib ?? DOWN_RISK_LOOKBACK_CALIB_HOURS_DEFAULT));
    const baseLookbackFastHours = Math.max(1, Number(t.config.downRiskLookbackHoursFast ?? DOWN_RISK_LOOKBACK_FAST_HOURS_DEFAULT));
    const baseMinTradesSoft = Math.max(1, Number(t.config.downRiskMinTradesSoft ?? DOWN_RISK_MIN_TRADES_SOFT_DEFAULT));
    const baseMinTradesHard = Math.max(baseMinTradesSoft, Number(t.config.downRiskMinTradesHard ?? DOWN_RISK_MIN_TRADES_HARD_DEFAULT));
    const baseMinTradesFast = Math.max(1, Number(t.config.downRiskMinTradesFast ?? DOWN_RISK_MIN_TRADES_FAST_DEFAULT));
    const baseWrSoft = Math.max(0, Math.min(1, Number(t.config.downRiskWrSoft ?? DOWN_RISK_WR_SOFT_DEFAULT)));
    const baseWrHard = Math.max(0, Math.min(1, Number(t.config.downRiskWrHard ?? DOWN_RISK_WR_HARD_DEFAULT)));
    const baseWrFastBlocked = Math.max(0, Math.min(1, Number(t.config.downRiskWrFastBlocked ?? DOWN_RISK_WR_FAST_BLOCKED_DEFAULT)));
    const basePnlSoft = Number(t.config.downRiskPnlSoft ?? DOWN_RISK_PNL_SOFT_DEFAULT);
    const basePnlHard = Number(t.config.downRiskPnlHard ?? DOWN_RISK_PNL_HARD_DEFAULT);
    const basePnlFastBlocked = Number(t.config.downRiskPnlFastBlocked ?? DOWN_RISK_PNL_FAST_BLOCKED_DEFAULT);
    const baseSoftExtraDelta = Math.max(0, Number(t.config.downRiskSoftExtraDelta ?? DOWN_RISK_SOFT_EXTRA_DELTA_DEFAULT));
    const baseSoftBetScale = Math.max(0, Math.min(1, Number(t.config.downRiskSoftBetScale ?? DOWN_RISK_SOFT_BET_SCALE_DEFAULT)));
    const baseHardHoldBars = Math.max(
        1,
        Number(
            t.config.downRiskHardHoldBars
            ?? (t.config.profile === '70' ? DOWN_RISK_HARD_HOLD_BARS_PROFILE70_DEFAULT : DOWN_RISK_HARD_HOLD_BARS_DEFAULT),
        ),
    );
    const baseReleaseChecks = Math.max(1, Number(t.config.downRiskReleaseChecks ?? DOWN_RISK_RELEASE_CHECKS_DEFAULT));
    const baseReleaseWr = Math.max(0, Math.min(1, Number(t.config.downRiskReleaseWr ?? DOWN_RISK_RELEASE_WR_DEFAULT)));
    const baseReleasePnl = Number(t.config.downRiskReleasePnl ?? DOWN_RISK_RELEASE_PNL_DEFAULT);
    const baseReleaseConfirmHours = Math.max(1, Number(t.config.downRiskReleaseConfirmHours ?? DOWN_RISK_RELEASE_CONFIRM_HOURS_DEFAULT));
    const nowSec = Math.floor(nowMs / 1000);
    const profile = t.config.profile ?? 'default';
    const triggerRouteMode = resolveDownTriggerRouteMode(routeMode);

    if (!hasActiveMode) {
        for (const symbol of targetSymbols) {
            const cell = getOrInitDownRiskCell(t, symbol, routeMode);
            cell.tier = 'normal';
            cell.blocked = false;
            cell.downScale = 1;
            cell.extraDelta = 0;
            cell.checkedAtMs = nowMs;
            cell.lastAction = 'off';
            cell.reasonCode = 'off_mode';
            cell.triggerLane = 'normal';
            cell.enteredBy = 'normal';
            cell.lastTransitionAtMs = nowMs;
            cell.releaseWindowMainPass = false;
            cell.releaseWindowConfirmPass = false;
            cell.sourceMode = routeMode;
            t.downRiskRuntimeBlockedBySymbol[symbol] = false;
            t.downRiskRuntimeScaleBySymbol[symbol] = 1;
            t.downRiskRuntimeExtraDeltaBySymbol[symbol] = 0;
        }
        return;
    }

    const tradesPath = path.join(process.cwd(), t.config.logsDir, `prediction_trades.${routeMode}.json`);
    if (!fs.existsSync(tradesPath)) {
        for (const symbol of targetSymbols) {
            const cell = getOrInitDownRiskCell(t, symbol, routeMode);
            const symbolMode = symbolModes.get(symbol) || 'off';
            cell.checkedAtMs = nowMs;
            if (symbolMode === 'off') {
                cell.tier = 'normal';
                cell.lastAction = 'off';
                cell.reasonCode = 'off_mode';
            } else {
                cell.lastAction = 'no_data_file';
                cell.reasonCode = 'no_data';
            }
            cell.sourceMode = routeMode;
            t.downRiskRuntimeBlockedBySymbol[symbol] = false;
            t.downRiskRuntimeScaleBySymbol[symbol] = 1;
            t.downRiskRuntimeExtraDeltaBySymbol[symbol] = 0;
        }
        return;
    }
    const stat = fs.statSync(tradesPath);
    const mtimeMs = stat.mtimeMs;
    if (mtimeMs === t.downRiskLastFileMtimeMsByMode[routeMode] && !hasEnforceMode) {
        for (const symbol of targetSymbols) {
            const cell = getOrInitDownRiskCell(t, symbol, routeMode);
            const symbolMode = symbolModes.get(symbol) || 'off';
            cell.checkedAtMs = nowMs;
            if (symbolMode === 'off') {
                cell.tier = 'normal';
                cell.lastAction = 'off';
                cell.reasonCode = 'off_mode';
            } else {
                cell.lastAction = 'shadow_no_new_data';
                cell.reasonCode = 'no_new_data';
            }
            cell.sourceMode = routeMode;
            t.downRiskRuntimeBlockedBySymbol[symbol] = false;
            t.downRiskRuntimeScaleBySymbol[symbol] = 1;
            t.downRiskRuntimeExtraDeltaBySymbol[symbol] = 0;
        }
        return;
    }
    t.downRiskLastFileMtimeMsByMode[routeMode] = mtimeMs;

    let rawEntries: Record<string, unknown>[] = [];
    try {
        const raw = JSON.parse(fs.readFileSync(tradesPath, 'utf8')) as unknown;
        if (!Array.isArray(raw)) return;
        rawEntries = raw as Record<string, unknown>[];
    } catch {
        for (const symbol of targetSymbols) {
            const cell = getOrInitDownRiskCell(t, symbol, routeMode);
            const symbolMode = symbolModes.get(symbol) || 'off';
            cell.checkedAtMs = nowMs;
            if (symbolMode === 'off') {
                cell.tier = 'normal';
                cell.lastAction = 'off';
                cell.reasonCode = 'off_mode';
            } else {
                cell.lastAction = 'parse_error';
                cell.reasonCode = 'parse_error';
            }
            cell.sourceMode = routeMode;
            t.downRiskRuntimeBlockedBySymbol[symbol] = false;
            t.downRiskRuntimeScaleBySymbol[symbol] = 1;
            t.downRiskRuntimeExtraDeltaBySymbol[symbol] = 0;
        }
        return;
    }

    for (const symbol of targetSymbols) {
        const symbolMode = symbolModes.get(symbol) || 'off';
        const isEnforce = symbolMode === 'enforce';
        const isShadow = symbolMode === 'shadow';
        if (symbolMode === 'off') {
            const cell = getOrInitDownRiskCell(t, symbol, routeMode);
            cell.tier = 'normal';
            cell.blocked = false;
            cell.downScale = 1;
            cell.extraDelta = 0;
            cell.checkedAtMs = nowMs;
            cell.lastAction = 'off';
            cell.reasonCode = 'off_mode';
            cell.triggerLane = 'normal';
            cell.enteredBy = 'normal';
            cell.lastTransitionAtMs = nowMs;
            cell.releaseWindowMainPass = false;
            cell.releaseWindowConfirmPass = false;
            cell.sourceMode = routeMode;
            t.downRiskRuntimeBlockedBySymbol[symbol] = false;
            t.downRiskRuntimeScaleBySymbol[symbol] = 1;
            t.downRiskRuntimeExtraDeltaBySymbol[symbol] = 0;
            continue;
        }
        const lookbackMainHours = Math.max(1, getDownRiskNumber(t, symbol, 'downRiskLookbackHoursMain', baseLookbackMainHours));
        const lookbackCalibHours = Math.max(lookbackMainHours, getDownRiskNumber(t, symbol, 'downRiskLookbackHoursCalib', baseLookbackCalibHours));
        const lookbackFastHours = Math.max(1, getDownRiskNumber(t, symbol, 'downRiskLookbackHoursFast', baseLookbackFastHours));
        const minTradesSoft = Math.max(1, getDownRiskNumber(t, symbol, 'downRiskMinTradesSoft', baseMinTradesSoft));
        const minTradesHard = Math.max(minTradesSoft, getDownRiskNumber(t, symbol, 'downRiskMinTradesHard', baseMinTradesHard));
        const minTradesFast = Math.max(1, getDownRiskNumber(t, symbol, 'downRiskMinTradesFast', baseMinTradesFast));
        const wrSoft = Math.max(0, Math.min(1, getDownRiskNumber(t, symbol, 'downRiskWrSoft', baseWrSoft)));
        const wrHard = Math.max(0, Math.min(1, getDownRiskNumber(t, symbol, 'downRiskWrHard', baseWrHard)));
        const wrFastBlocked = Math.max(0, Math.min(1, getDownRiskNumber(t, symbol, 'downRiskWrFastBlocked', baseWrFastBlocked)));
        const pnlSoft = getDownRiskNumber(t, symbol, 'downRiskPnlSoft', basePnlSoft);
        const pnlHard = getDownRiskNumber(t, symbol, 'downRiskPnlHard', basePnlHard);
        const pnlFastBlocked = getDownRiskNumber(t, symbol, 'downRiskPnlFastBlocked', basePnlFastBlocked);
        const baseSymbolSoftExtraDelta = Math.max(0, getDownRiskNumber(t, symbol, 'downRiskSoftExtraDelta', baseSoftExtraDelta));
        const baseSymbolSoftBetScale = Math.max(0, Math.min(1, getDownRiskNumber(t, symbol, 'downRiskSoftBetScale', baseSoftBetScale)));
        const hardHoldBars = Math.max(1, Math.round(getDownRiskNumber(t, symbol, 'downRiskHardHoldBars', baseHardHoldBars)));
        const releaseChecks = Math.max(1, Math.round(getDownRiskNumber(t, symbol, 'downRiskReleaseChecks', baseReleaseChecks)));
        const releaseWr = Math.max(0, Math.min(1, getDownRiskNumber(t, symbol, 'downRiskReleaseWr', baseReleaseWr)));
        const releasePnl = getDownRiskNumber(t, symbol, 'downRiskReleasePnl', baseReleasePnl);
        const releaseConfirmHours = Math.max(1, getDownRiskNumber(t, symbol, 'downRiskReleaseConfirmHours', baseReleaseConfirmHours));
        const priorityCluster = isAcuteDownPriorityCluster(profile, symbol);
        const softExtraDelta = priorityCluster ? Math.max(baseSymbolSoftExtraDelta + 0.01, baseSymbolSoftExtraDelta * 1.5) : baseSymbolSoftExtraDelta;
        const softBetScale = priorityCluster ? Math.min(baseSymbolSoftBetScale, 0.30) : baseSymbolSoftBetScale;
        const holdSecs = hardHoldBars * PERIOD_SECONDS;

        const cell = getOrInitDownRiskCell(t, symbol, routeMode);
        const mainCutoffMs = nowMs - lookbackMainHours * 3600 * 1000;
        const fastCutoffMs = nowMs - lookbackFastHours * 3600 * 1000;
        const calibCutoffMs = nowMs - lookbackCalibHours * 3600 * 1000;
        const releaseConfirmCutoffMs = nowMs - releaseConfirmHours * 3600 * 1000;
        const mainStats = calcDownWindowStats(rawEntries, symbol, routeMode, mainCutoffMs, lookbackMainHours);
        const fastStats = calcDownWindowStats(rawEntries, symbol, routeMode, fastCutoffMs, lookbackFastHours);
        const calibStats = calcDownWindowStats(rawEntries, symbol, routeMode, calibCutoffMs, lookbackCalibHours);
        const releaseConfirmStats = calcDownWindowStats(rawEntries, symbol, routeMode, releaseConfirmCutoffMs, releaseConfirmHours);

        cell.main = mainStats;
        cell.fast = fastStats;
        cell.calib = calibStats;
        cell.releaseConfirm = releaseConfirmStats;
        cell.checkedAtMs = nowMs;
        cell.sourceMode = routeMode;

        const softTriggered = mainStats.trades >= minTradesSoft
            && (mainStats.wrPost < wrSoft || mainStats.pnlPerTrade < pnlSoft);
        const hardTriggered = mainStats.trades >= minTradesHard
            && (mainStats.wrPost < wrHard && mainStats.pnlPerTrade < pnlHard);
        const fastHardTriggered = priorityCluster
            && fastStats.trades >= minTradesFast
            && (fastStats.wrPost < wrFastBlocked || fastStats.pnlPerTrade < pnlFastBlocked);

        let nextTier: DownRiskTier = hardTriggered ? 'hard' : (softTriggered ? 'soft' : 'normal');
        let triggerLane: DownRiskCellState['triggerLane'] = nextTier === 'normal' ? 'normal' : 'main';
        let enteredBy: DownRiskCellState['enteredBy'] = nextTier === 'normal' ? 'normal' : 'down';
        if (fastHardTriggered) {
            nextTier = 'hard';
            triggerLane = 'fast';
            enteredBy = 'down';
        }

        const crossLayerKeys = getDownCrossLayerTriggerCellKeys(profile, t.config.name, symbol, triggerRouteMode);
        const expectCell = t.expectancyGateCells.get(crossLayerKeys.expectKey);
        const shockCell = t.shockRiskCells.get(crossLayerKeys.shockKey);
        const comboCell = crossLayerKeys.comboKeys
            .map((key) => t.comboPauseCells.get(key))
            .find((value) => value != null);
        const shockActive = Boolean(shockCell?.active && nowSec < Number(shockCell?.activeUntilTs || 0));
        const comboActive = Boolean(comboCell?.active && nowSec < Number(comboCell?.activeUntilTs || 0));
        const expectStatus = String(expectCell?.status || 'normal').toLowerCase();
        const expectPressure = expectStatus === 'blocked' || expectStatus === 'degraded';
        const crossLayerHardTriggered = (softTriggered && (shockActive || comboActive))
            || (expectStatus === 'blocked' && shockActive);
        if (crossLayerHardTriggered) {
            nextTier = 'hard';
            triggerLane = 'cross_layer';
            enteredBy = comboActive ? 'combo' : (shockActive ? (expectStatus === 'blocked' ? 'expect' : 'shock') : 'down');
        }

        const calibCanDownshift = !shockActive
            && !comboActive
            && !expectPressure
            && calibStats.wrPost >= 0.50
            && calibStats.pnlPerTrade >= 0;
        if (calibCanDownshift) {
            if (nextTier === 'hard') nextTier = 'soft';
            else if (nextTier === 'soft') nextTier = 'normal';
            if (nextTier === 'normal') {
                triggerLane = 'normal';
                enteredBy = 'normal';
            } else {
                triggerLane = 'main';
                enteredBy = 'down';
            }
        }

        if (cell.tier !== 'hard' && nextTier === 'hard') {
            cell.hardUntilTs = nowSec + holdSecs;
            cell.releasePassStreak = 0;
            cell.triggerLane = triggerLane;
            cell.enteredBy = enteredBy;
            cell.lastTransitionAtMs = nowMs;
            if (triggerLane === 'fast') {
                cell.lastAction = `enter_hard_fast(wr2h=${fastStats.wrPost.toFixed(3)} pnl2h=${fastStats.pnlPerTrade.toFixed(3)})`;
                cell.reasonCode = 'hard_trigger_fast';
            } else if (triggerLane === 'cross_layer') {
                cell.lastAction = `enter_hard_cross_layer(shock=${shockActive ? 1 : 0},combo=${comboActive ? 1 : 0},expect=${expectStatus})`;
                cell.reasonCode = 'hard_cross_layer';
            } else {
                cell.lastAction = `enter_hard(main wrPost=${mainStats.wrPost.toFixed(3)} pnl=${mainStats.pnlPerTrade.toFixed(3)})`;
                cell.reasonCode = 'hard_trigger';
            }
        }

        if (cell.tier === 'hard') {
            if (nowSec < cell.hardUntilTs) {
                nextTier = 'hard';
            } else {
                const releasePassMain = mainStats.wrPost >= releaseWr && mainStats.pnlPerTrade >= releasePnl;
                const releasePassConfirm = releaseConfirmStats.wrPost >= releaseWr && releaseConfirmStats.pnlPerTrade >= releasePnl;
                cell.releaseWindowMainPass = releasePassMain;
                cell.releaseWindowConfirmPass = releasePassConfirm;
                if (releasePassMain && releasePassConfirm) {
                    cell.releasePassStreak += 1;
                    if (cell.releasePassStreak >= releaseChecks) {
                        nextTier = softTriggered ? 'soft' : 'normal';
                        cell.lastAction = `release_hard(streak=${cell.releasePassStreak})`;
                        cell.reasonCode = 'hard_release';
                        cell.releasePassStreak = 0;
                        cell.hardUntilTs = 0;
                        cell.lastTransitionAtMs = nowMs;
                        cell.triggerLane = nextTier === 'normal' ? 'normal' : 'main';
                        cell.enteredBy = nextTier === 'normal' ? 'normal' : 'down';
                    } else {
                        nextTier = 'hard';
                        cell.lastAction = `hard_wait_release(streak=${cell.releasePassStreak}/${releaseChecks})`;
                        cell.reasonCode = 'hard_release_pending';
                    }
                } else {
                    cell.releasePassStreak = 0;
                    nextTier = 'hard';
                    cell.hardUntilTs = nowSec + holdSecs;
                    cell.lastAction = `hard_extend(release main=${releasePassMain ? 1 : 0} confirm=${releasePassConfirm ? 1 : 0} wr=${mainStats.wrPost.toFixed(3)} pnl=${mainStats.pnlPerTrade.toFixed(3)})`;
                    cell.reasonCode = 'hard_extend';
                }
            }
        } else {
            cell.releaseWindowMainPass = false;
            cell.releaseWindowConfirmPass = false;
        }

        cell.tier = nextTier;
        const enforceBlocked = isEnforce && nextTier === 'hard' && nowSec < cell.hardUntilTs;
        const enforceScale = isEnforce && nextTier === 'soft' ? softBetScale : 1;
        const enforceDelta = isEnforce && nextTier === 'soft' ? softExtraDelta : 0;
        cell.blocked = enforceBlocked;
        cell.downScale = enforceScale;
        cell.extraDelta = enforceDelta;

        if (isShadow) {
            cell.lastAction = `shadow tier=${nextTier} wrPost6=${mainStats.wrPost.toFixed(3)} pnl6=${mainStats.pnlPerTrade.toFixed(3)} wrPost24=${calibStats.wrPost.toFixed(3)} pnl24=${calibStats.pnlPerTrade.toFixed(3)}`;
            cell.reasonCode = 'shadow_only';
        } else if (!isEnforce) {
            cell.lastAction = `off tier=${nextTier}`;
            cell.reasonCode = 'off_mode';
        } else if (nextTier === 'soft') {
            cell.lastAction = `soft(scale=${softBetScale.toFixed(2)},delta=+${softExtraDelta.toFixed(3)})`;
            cell.reasonCode = 'soft_trigger';
        } else if (nextTier === 'normal') {
            if (cell.reasonCode !== 'hard_release') {
                cell.reasonCode = 'normal';
                cell.lastAction = 'normal';
                cell.triggerLane = 'normal';
                cell.enteredBy = 'normal';
            }
        } else if (nextTier === 'hard' && cell.reasonCode === 'init') {
            cell.reasonCode = 'hard_hold';
        }

        t.downRiskRuntimeBlockedBySymbol[symbol] = enforceBlocked;
        t.downRiskRuntimeScaleBySymbol[symbol] = enforceScale;
        t.downRiskRuntimeExtraDeltaBySymbol[symbol] = enforceDelta;
    }
}

function maybeUpdateDownRisk(t: TraderInstance, nowMs: number): void {
    const engine = resolveDownRiskEngine(t.config);
    if (engine === 'v2') {
        maybeUpdateDownRiskV2(t, nowMs);
        return;
    }
    const symbols = parseAllowedSymbols(t.config.allowedMarkets || '');
    for (const symbol of symbols) {
        t.downRiskRuntimeBlockedBySymbol[symbol] = false;
        t.downRiskRuntimeScaleBySymbol[symbol] = 1;
        t.downRiskRuntimeExtraDeltaBySymbol[symbol] = 0;
    }
    maybeUpdateDownBreakerV1(t, nowMs);
}

function maybeUpdateUpRiskV2(t: TraderInstance, nowMs: number): void {
    const routeMode = resolveUpRiskRouteMode(t);
    const symbols = parseAllowedSymbols(t.config.allowedMarkets || '');
    const targetSymbols = symbols.filter((s) => isUpRiskTargetSymbol(s));
    const symbolModes = new Map<string, UpRiskMode>();
    let hasActiveMode = false;
    let hasEnforceMode = false;
    for (const symbol of targetSymbols) {
        const mode = resolveUpRiskMode(t.config, symbol);
        symbolModes.set(symbol, mode);
        if (mode !== 'off') hasActiveMode = true;
        if (mode === 'enforce') hasEnforceMode = true;
    }
    for (const symbol of symbols) {
        t.upRiskRuntimeBlockedBySymbol[symbol] = false;
        t.upRiskRuntimeScaleBySymbol[symbol] = 1;
        t.upRiskRuntimeExtraDeltaBySymbol[symbol] = 0;
    }
    if (targetSymbols.length === 0) {
        return;
    }
    const checkEverySec = Math.max(10, Number(t.config.upRiskCheckSeconds ?? UP_RISK_CHECK_SECONDS_DEFAULT));
    const lastCheckMs = t.upRiskLastCheckMsByMode[routeMode] || 0;
    if (nowMs - lastCheckMs < checkEverySec * 1000) return;
    t.upRiskLastCheckMsByMode[routeMode] = nowMs;

    const baseLookbackMainHours = Math.max(1, Number(t.config.upRiskLookbackHoursMain ?? UP_RISK_LOOKBACK_MAIN_HOURS_DEFAULT));
    const baseLookbackCalibHours = Math.max(baseLookbackMainHours, Number(t.config.upRiskLookbackHoursCalib ?? UP_RISK_LOOKBACK_CALIB_HOURS_DEFAULT));
    const baseMinTradesSoft = Math.max(1, Number(t.config.upRiskMinTradesSoft ?? UP_RISK_MIN_TRADES_SOFT_DEFAULT));
    const baseWrSoft = Math.max(0, Math.min(1, Number(t.config.upRiskWrSoft ?? UP_RISK_WR_SOFT_DEFAULT)));
    const basePnlSoft = Number(t.config.upRiskPnlSoft ?? UP_RISK_PNL_SOFT_DEFAULT);
    const baseSoftExtraDelta = Math.max(0, Number(t.config.upRiskSoftExtraDelta ?? UP_RISK_SOFT_EXTRA_DELTA_DEFAULT));
    const baseSoftBetScale = Math.max(0, Math.min(1, Number(t.config.upRiskSoftBetScale ?? UP_RISK_SOFT_BET_SCALE_DEFAULT)));
    const baseReleaseChecks = Math.max(1, Number(t.config.upRiskReleaseChecks ?? UP_RISK_RELEASE_CHECKS_DEFAULT));
    const baseReleaseWr = Math.max(0, Math.min(1, Number(t.config.upRiskReleaseWr ?? UP_RISK_RELEASE_WR_DEFAULT)));
    const baseReleasePnl = Number(t.config.upRiskReleasePnl ?? UP_RISK_RELEASE_PNL_DEFAULT);

    if (!hasActiveMode) {
        for (const symbol of targetSymbols) {
            const cell = getOrInitUpRiskCell(t, symbol, routeMode);
            cell.tier = 'normal';
            cell.blocked = false;
            cell.upScale = 1;
            cell.extraDelta = 0;
            cell.checkedAtMs = nowMs;
            cell.lastAction = 'off';
            cell.reasonCode = 'off_mode';
            cell.sourceMode = routeMode;
            t.upRiskRuntimeBlockedBySymbol[symbol] = false;
            t.upRiskRuntimeScaleBySymbol[symbol] = 1;
            t.upRiskRuntimeExtraDeltaBySymbol[symbol] = 0;
        }
        return;
    }

    const tradesPath = path.join(process.cwd(), t.config.logsDir, `prediction_trades.${routeMode}.json`);
    if (!fs.existsSync(tradesPath)) {
        for (const symbol of targetSymbols) {
            const cell = getOrInitUpRiskCell(t, symbol, routeMode);
            const symbolMode = symbolModes.get(symbol) || 'off';
            cell.checkedAtMs = nowMs;
            if (symbolMode === 'off') {
                cell.tier = 'normal';
                cell.lastAction = 'off';
                cell.reasonCode = 'off_mode';
            } else {
                cell.lastAction = 'no_data_file';
                cell.reasonCode = 'no_data';
            }
            cell.sourceMode = routeMode;
            t.upRiskRuntimeBlockedBySymbol[symbol] = false;
            t.upRiskRuntimeScaleBySymbol[symbol] = 1;
            t.upRiskRuntimeExtraDeltaBySymbol[symbol] = 0;
        }
        return;
    }

    const stat = fs.statSync(tradesPath);
    const mtimeMs = stat.mtimeMs;
    if (mtimeMs === t.upRiskLastFileMtimeMsByMode[routeMode] && !hasEnforceMode) {
        for (const symbol of targetSymbols) {
            const cell = getOrInitUpRiskCell(t, symbol, routeMode);
            const symbolMode = symbolModes.get(symbol) || 'off';
            cell.checkedAtMs = nowMs;
            if (symbolMode === 'off') {
                cell.tier = 'normal';
                cell.lastAction = 'off';
                cell.reasonCode = 'off_mode';
            } else {
                cell.lastAction = 'shadow_no_new_data';
                cell.reasonCode = 'no_new_data';
            }
            cell.sourceMode = routeMode;
            t.upRiskRuntimeBlockedBySymbol[symbol] = false;
            t.upRiskRuntimeScaleBySymbol[symbol] = 1;
            t.upRiskRuntimeExtraDeltaBySymbol[symbol] = 0;
        }
        return;
    }
    t.upRiskLastFileMtimeMsByMode[routeMode] = mtimeMs;

    let rawEntries: Record<string, unknown>[] = [];
    try {
        const raw = JSON.parse(fs.readFileSync(tradesPath, 'utf8')) as unknown;
        if (!Array.isArray(raw)) return;
        rawEntries = raw as Record<string, unknown>[];
    } catch {
        for (const symbol of targetSymbols) {
            const cell = getOrInitUpRiskCell(t, symbol, routeMode);
            const symbolMode = symbolModes.get(symbol) || 'off';
            cell.checkedAtMs = nowMs;
            if (symbolMode === 'off') {
                cell.tier = 'normal';
                cell.lastAction = 'off';
                cell.reasonCode = 'off_mode';
            } else {
                cell.lastAction = 'parse_error';
                cell.reasonCode = 'parse_error';
            }
            cell.sourceMode = routeMode;
            t.upRiskRuntimeBlockedBySymbol[symbol] = false;
            t.upRiskRuntimeScaleBySymbol[symbol] = 1;
            t.upRiskRuntimeExtraDeltaBySymbol[symbol] = 0;
        }
        return;
    }
    for (const symbol of targetSymbols) {
        const symbolMode = symbolModes.get(symbol) || 'off';
        const isEnforce = symbolMode === 'enforce';
        const isShadow = symbolMode === 'shadow';
        if (symbolMode === 'off') {
            const cell = getOrInitUpRiskCell(t, symbol, routeMode);
            cell.tier = 'normal';
            cell.blocked = false;
            cell.upScale = 1;
            cell.extraDelta = 0;
            cell.checkedAtMs = nowMs;
            cell.lastAction = 'off';
            cell.reasonCode = 'off_mode';
            cell.sourceMode = routeMode;
            t.upRiskRuntimeBlockedBySymbol[symbol] = false;
            t.upRiskRuntimeScaleBySymbol[symbol] = 1;
            t.upRiskRuntimeExtraDeltaBySymbol[symbol] = 0;
            continue;
        }
        const lookbackMainHours = Math.max(1, getUpRiskNumber(t, symbol, 'upRiskLookbackHoursMain', baseLookbackMainHours));
        const lookbackCalibHours = Math.max(lookbackMainHours, getUpRiskNumber(t, symbol, 'upRiskLookbackHoursCalib', baseLookbackCalibHours));
        const minTradesSoft = Math.max(1, getUpRiskNumber(t, symbol, 'upRiskMinTradesSoft', baseMinTradesSoft));
        const wrSoft = Math.max(0, Math.min(1, getUpRiskNumber(t, symbol, 'upRiskWrSoft', baseWrSoft)));
        const pnlSoft = getUpRiskNumber(t, symbol, 'upRiskPnlSoft', basePnlSoft);
        const softExtraDelta = Math.max(0, getUpRiskNumber(t, symbol, 'upRiskSoftExtraDelta', baseSoftExtraDelta));
        const softBetScale = Math.max(0, Math.min(1, getUpRiskNumber(t, symbol, 'upRiskSoftBetScale', baseSoftBetScale)));
        const releaseChecks = Math.max(1, Math.round(getUpRiskNumber(t, symbol, 'upRiskReleaseChecks', baseReleaseChecks)));
        const releaseWr = Math.max(0, Math.min(1, getUpRiskNumber(t, symbol, 'upRiskReleaseWr', baseReleaseWr)));
        const releasePnl = getUpRiskNumber(t, symbol, 'upRiskReleasePnl', baseReleasePnl);

        const cell = getOrInitUpRiskCell(t, symbol, routeMode);
        const mainCutoffMs = nowMs - lookbackMainHours * 3600 * 1000;
        const calibCutoffMs = nowMs - lookbackCalibHours * 3600 * 1000;
        const mainStats = calcUpWindowStats(rawEntries, symbol, routeMode, mainCutoffMs, lookbackMainHours);
        const calibStats = calcUpWindowStats(rawEntries, symbol, routeMode, calibCutoffMs, lookbackCalibHours);

        cell.main = mainStats;
        cell.calib = calibStats;
        cell.checkedAtMs = nowMs;
        cell.sourceMode = routeMode;

        const softTriggered = mainStats.trades >= minTradesSoft
            && (mainStats.wrPost < wrSoft || mainStats.pnlPerTrade < pnlSoft);
        let nextTier: UpRiskTier = softTriggered ? 'soft' : 'normal';
        const calibCanDownshift = calibStats.wrPost >= 0.50 && calibStats.pnlPerTrade >= 0;
        if (calibCanDownshift && nextTier === 'soft') {
            nextTier = 'normal';
        }

        if (cell.tier !== 'soft' && nextTier === 'soft') {
            cell.releasePassStreak = 0;
            cell.lastAction = `enter_soft(main wrPost=${mainStats.wrPost.toFixed(3)} pnl=${mainStats.pnlPerTrade.toFixed(3)})`;
            cell.reasonCode = 'soft_trigger';
        }

        if (cell.tier === 'soft' && nextTier !== 'soft') {
            const releasePass = mainStats.wrPost >= releaseWr && mainStats.pnlPerTrade >= releasePnl;
            if (releasePass) {
                cell.releasePassStreak += 1;
                if (cell.releasePassStreak >= releaseChecks) {
                    nextTier = 'normal';
                    cell.lastAction = `release_soft(streak=${cell.releasePassStreak})`;
                    cell.reasonCode = 'soft_release';
                    cell.releasePassStreak = 0;
                } else {
                    nextTier = 'soft';
                    cell.lastAction = `soft_wait_release(streak=${cell.releasePassStreak}/${releaseChecks})`;
                    cell.reasonCode = 'soft_release_pending';
                }
            } else {
                cell.releasePassStreak = 0;
                nextTier = 'soft';
                cell.lastAction = `soft_extend(release wrPost=${mainStats.wrPost.toFixed(3)} pnl=${mainStats.pnlPerTrade.toFixed(3)})`;
                cell.reasonCode = 'soft_extend';
            }
        }

        cell.tier = nextTier;
        const enforceBlocked = false;
        const enforceScale = isEnforce && nextTier === 'soft' ? softBetScale : 1;
        const enforceDelta = isEnforce && nextTier === 'soft' ? softExtraDelta : 0;
        cell.blocked = enforceBlocked;
        cell.upScale = enforceScale;
        cell.extraDelta = enforceDelta;

        if (isShadow) {
            cell.lastAction = `shadow tier=${nextTier} wrPost6=${mainStats.wrPost.toFixed(3)} pnl6=${mainStats.pnlPerTrade.toFixed(3)} wrPost24=${calibStats.wrPost.toFixed(3)} pnl24=${calibStats.pnlPerTrade.toFixed(3)}`;
            cell.reasonCode = 'shadow_only';
        } else if (!isEnforce) {
            cell.lastAction = `off tier=${nextTier}`;
            cell.reasonCode = 'off_mode';
        } else if (nextTier === 'soft') {
            cell.lastAction = `soft(scale=${softBetScale.toFixed(2)},delta=+${softExtraDelta.toFixed(3)})`;
            cell.reasonCode = 'soft_trigger';
        } else if (nextTier === 'normal' && cell.lastAction === 'init') {
            cell.lastAction = 'normal';
            cell.reasonCode = 'normal';
        } else if (nextTier === 'normal') {
            cell.reasonCode = cell.reasonCode === 'soft_release' ? 'soft_release' : 'normal';
        }

        t.upRiskRuntimeBlockedBySymbol[symbol] = enforceBlocked;
        t.upRiskRuntimeScaleBySymbol[symbol] = enforceScale;
        t.upRiskRuntimeExtraDeltaBySymbol[symbol] = enforceDelta;
    }
}

function maybeUpdateUpRisk(t: TraderInstance, nowMs: number): void {
    const engine = resolveUpRiskEngine(t.config);
    if (engine === 'v2') {
        maybeUpdateUpRiskV2(t, nowMs);
        return;
    }
    const symbols = parseAllowedSymbols(t.config.allowedMarkets || '');
    for (const symbol of symbols) {
        t.upRiskRuntimeBlockedBySymbol[symbol] = false;
        t.upRiskRuntimeScaleBySymbol[symbol] = 1;
        t.upRiskRuntimeExtraDeltaBySymbol[symbol] = 0;
    }
}

function shockRuntimeKey(symbol: string, direction: 'UP' | 'DOWN'): string {
    return `${symbol.toUpperCase()}::${direction}`;
}

function maybeUpdateShockRiskV2(t: TraderInstance, nowMs: number): void {
    const routeMode = resolveShockRiskRouteMode(t);
    const symbols = parseAllowedSymbols(t.config.allowedMarkets || '');
    const targetSymbols = symbols.filter((s) => isShockRiskTargetSymbol(s));
    const symbolModes = new Map<string, ShockRiskMode>();
    let hasActiveMode = false;
    let hasEnforceMode = false;
    for (const symbol of targetSymbols) {
        const mode = resolveShockRiskMode(t.config, symbol);
        symbolModes.set(symbol, mode);
        if (mode !== 'off') hasActiveMode = true;
        if (mode === 'enforce') hasEnforceMode = true;
    }
    for (const symbol of symbols) {
        t.shockRiskRuntimeBlockedBySymbolDir[shockRuntimeKey(symbol, 'UP')] = false;
        t.shockRiskRuntimeBlockedBySymbolDir[shockRuntimeKey(symbol, 'DOWN')] = false;
    }
    if (targetSymbols.length === 0) return;
    if (!hasActiveMode) {
        for (const symbol of targetSymbols) {
            for (const direction of ['UP', 'DOWN'] as const) {
                const cell = getOrInitShockRiskCell(t, symbol, direction, routeMode);
                cell.active = false;
                cell.activeUntilTs = 0;
                cell.releasePassStreak = 0;
                cell.blocked = false;
                cell.checkedAtMs = nowMs;
                cell.lastAction = 'off';
                cell.reasonCode = 'off_mode';
                cell.sourceMode = routeMode;
            }
        }
        return;
    }

    const checkEverySec = Math.max(10, Number(t.config.shockRiskCheckSeconds ?? SHOCK_RISK_CHECK_SECONDS_DEFAULT));
    const lastCheckMs = t.shockRiskLastCheckMsByMode[routeMode] || 0;
    if (nowMs - lastCheckMs < checkEverySec * 1000) return;
    t.shockRiskLastCheckMsByMode[routeMode] = nowMs;

    const baseWindowMinutes = Math.max(30, Number(t.config.shockRiskWindowMinutes ?? SHOCK_RISK_WINDOW_MINUTES_DEFAULT));
    const baseMinTrades = Math.max(1, Number(t.config.shockRiskMinTrades ?? SHOCK_RISK_MIN_TRADES_DEFAULT));
    const baseWrPost = Math.max(0, Math.min(1, Number(t.config.shockRiskWrPost ?? SHOCK_RISK_WR_POST_DEFAULT)));
    const basePnlPerTrade = Number(t.config.shockRiskPnlPerTrade ?? SHOCK_RISK_PNL_PER_TRADE_DEFAULT);
    const baseVolPercentile = Math.max(50, Math.min(99, Number(t.config.shockRiskVolPercentile ?? SHOCK_RISK_VOL_PERCENTILE_DEFAULT)));
    const baseSevereVolRatio = Math.max(1.2, Number(t.config.shockRiskSevereVolRatio ?? SHOCK_RISK_SEVERE_VOL_RATIO_DEFAULT));
    const baseHoldHours = Math.max(1, Number(t.config.shockRiskHoldHours ?? SHOCK_RISK_HOLD_HOURS_DEFAULT));
    const baseSevereHoldHours = Math.max(baseHoldHours, Number(t.config.shockRiskSevereHoldHours ?? SHOCK_RISK_SEVERE_HOLD_HOURS_DEFAULT));
    const baseReleaseChecks = Math.max(1, Number(t.config.shockRiskReleaseChecks ?? SHOCK_RISK_RELEASE_CHECKS_DEFAULT));
    const baseReleaseWr = Math.max(0, Math.min(1, Number(t.config.shockRiskReleaseWr ?? SHOCK_RISK_RELEASE_WR_DEFAULT)));
    const baseReleasePnl = Number(t.config.shockRiskReleasePnl ?? SHOCK_RISK_RELEASE_PNL_DEFAULT);
    const nowSec = Math.floor(nowMs / 1000);

    const tradesPath = path.join(process.cwd(), t.config.logsDir, `prediction_trades.${routeMode}.json`);
    if (!fs.existsSync(tradesPath)) {
        for (const symbol of targetSymbols) {
            const symbolMode = symbolModes.get(symbol) || 'off';
            for (const direction of ['UP', 'DOWN'] as const) {
                const cell = getOrInitShockRiskCell(t, symbol, direction, routeMode);
                cell.checkedAtMs = nowMs;
                if (symbolMode === 'off') {
                    cell.active = false;
                    cell.activeUntilTs = 0;
                    cell.releasePassStreak = 0;
                    cell.lastAction = 'off';
                    cell.reasonCode = 'off_mode';
                } else {
                    cell.lastAction = 'no_data_file';
                    cell.reasonCode = 'no_data';
                }
                cell.blocked = false;
                cell.sourceMode = routeMode;
                t.shockRiskRuntimeBlockedBySymbolDir[shockRuntimeKey(symbol, direction)] = false;
            }
        }
        return;
    }
    const stat = fs.statSync(tradesPath);
    const mtimeMs = stat.mtimeMs;
    if (mtimeMs === t.shockRiskLastFileMtimeMsByMode[routeMode] && !hasEnforceMode) {
        for (const symbol of targetSymbols) {
            const symbolMode = symbolModes.get(symbol) || 'off';
            for (const direction of ['UP', 'DOWN'] as const) {
                const cell = getOrInitShockRiskCell(t, symbol, direction, routeMode);
                cell.checkedAtMs = nowMs;
                if (symbolMode === 'off') {
                    cell.active = false;
                    cell.activeUntilTs = 0;
                    cell.releasePassStreak = 0;
                    cell.lastAction = 'off';
                    cell.reasonCode = 'off_mode';
                } else {
                    cell.lastAction = 'shadow_no_new_data';
                    cell.reasonCode = 'no_new_data';
                }
                cell.blocked = false;
                cell.sourceMode = routeMode;
                t.shockRiskRuntimeBlockedBySymbolDir[shockRuntimeKey(symbol, direction)] = false;
            }
        }
        return;
    }
    t.shockRiskLastFileMtimeMsByMode[routeMode] = mtimeMs;

    let rawEntries: Record<string, unknown>[] = [];
    try {
        const raw = JSON.parse(fs.readFileSync(tradesPath, 'utf8')) as unknown;
        if (!Array.isArray(raw)) return;
        rawEntries = raw as Record<string, unknown>[];
    } catch {
        for (const symbol of targetSymbols) {
            const symbolMode = symbolModes.get(symbol) || 'off';
            for (const direction of ['UP', 'DOWN'] as const) {
                const cell = getOrInitShockRiskCell(t, symbol, direction, routeMode);
                cell.checkedAtMs = nowMs;
                if (symbolMode === 'off') {
                    cell.active = false;
                    cell.activeUntilTs = 0;
                    cell.releasePassStreak = 0;
                    cell.lastAction = 'off';
                    cell.reasonCode = 'off_mode';
                } else {
                    cell.lastAction = 'parse_error';
                    cell.reasonCode = 'parse_error';
                }
                cell.blocked = false;
                cell.sourceMode = routeMode;
                t.shockRiskRuntimeBlockedBySymbolDir[shockRuntimeKey(symbol, direction)] = false;
            }
        }
        return;
    }
    const volBaseCutoffMs = nowMs - 24 * 3600 * 1000;

    for (const symbol of targetSymbols) {
        const symbolMode = symbolModes.get(symbol) || 'off';
        const isEnforce = symbolMode === 'enforce';
        const isShadow = symbolMode === 'shadow';
        if (symbolMode === 'off') {
            for (const direction of ['UP', 'DOWN'] as const) {
                const cell = getOrInitShockRiskCell(t, symbol, direction, routeMode);
                cell.active = false;
                cell.activeUntilTs = 0;
                cell.releasePassStreak = 0;
                cell.blocked = false;
                cell.checkedAtMs = nowMs;
                cell.lastAction = 'off';
                cell.reasonCode = 'off_mode';
                cell.sourceMode = routeMode;
                t.shockRiskRuntimeBlockedBySymbolDir[shockRuntimeKey(symbol, direction)] = false;
            }
            continue;
        }
        const windowMinutes = Math.max(30, Math.round(getShockRiskNumber(t, symbol, 'shockRiskWindowMinutes', baseWindowMinutes)));
        const minTrades = Math.max(1, Math.round(getShockRiskNumber(t, symbol, 'shockRiskMinTrades', baseMinTrades)));
        const wrPostThreshold = Math.max(0, Math.min(1, getShockRiskNumber(t, symbol, 'shockRiskWrPost', baseWrPost)));
        const pnlPerTradeThreshold = getShockRiskNumber(t, symbol, 'shockRiskPnlPerTrade', basePnlPerTrade);
        const volPercentile = Math.max(50, Math.min(99, getShockRiskNumber(t, symbol, 'shockRiskVolPercentile', baseVolPercentile)));
        const severeVolRatio = Math.max(1.2, getShockRiskNumber(t, symbol, 'shockRiskSevereVolRatio', baseSevereVolRatio));
        const holdHours = Math.max(1, getShockRiskNumber(t, symbol, 'shockRiskHoldHours', baseHoldHours));
        const severeHoldHours = Math.max(holdHours, getShockRiskNumber(t, symbol, 'shockRiskSevereHoldHours', baseSevereHoldHours));
        const releaseChecks = Math.max(1, Math.round(getShockRiskNumber(t, symbol, 'shockRiskReleaseChecks', baseReleaseChecks)));
        const releaseWr = Math.max(0, Math.min(1, getShockRiskNumber(t, symbol, 'shockRiskReleaseWr', baseReleaseWr)));
        const releasePnl = getShockRiskNumber(t, symbol, 'shockRiskReleasePnl', baseReleasePnl);
        const holdSecs = Math.max(1, Math.round(holdHours * 3600));
        const severeHoldSecs = Math.max(holdSecs, Math.round(severeHoldHours * 3600));

        for (const direction of ['UP', 'DOWN'] as const) {
            const cell = getOrInitShockRiskCell(t, symbol, direction, routeMode);
            const mainCutoffMs = nowMs - windowMinutes * 60 * 1000;
            const stat1h = calcShockWindowStats(rawEntries, symbol, direction, routeMode, nowMs - 3600 * 1000, 60);
            const stat4h = calcShockWindowStats(rawEntries, symbol, direction, routeMode, nowMs - 4 * 3600 * 1000, 240);
            const stat24h = calcShockWindowStats(rawEntries, symbol, direction, routeMode, nowMs - 24 * 3600 * 1000, 1440);
            const main = calcShockWindowStats(rawEntries, symbol, direction, routeMode, mainCutoffMs, windowMinutes);
            const absSeries = collectShockAbsPnlSeries(rawEntries, symbol, direction, routeMode, volBaseCutoffMs)
                .sort((a, b) => a - b);
            const volPctlValue = quantileFromSorted(absSeries, volPercentile / 100);

            cell.main = main;
            cell.checkedAtMs = nowMs;
            cell.sourceMode = routeMode;
            cell.vol1h = stat1h.avgAbsPnl;
            cell.vol4h = stat4h.avgAbsPnl;
            cell.vol24h = stat24h.avgAbsPnl;
            cell.volRatio = stat24h.avgAbsPnl > 0 ? stat1h.avgAbsPnl / stat24h.avgAbsPnl : 0;
            cell.volPctl = volPctlValue;

            const volRatioThreshold = 1 + Math.max(0, (volPercentile - 50) / 100);
            const volRatioTriggered = stat24h.avgAbsPnl > 0 && cell.volRatio >= volRatioThreshold;
            const volQuantileTriggered = (stat1h.avgAbsPnl >= volPctlValue) || (stat4h.avgAbsPnl >= volPctlValue);
            const volTriggered = volRatioTriggered || volQuantileTriggered;
            const lossTriggered = stat1h.pnl < 0 || stat4h.pnl < 0;
            const severePnlThreshold = Math.min(pnlPerTradeThreshold * 1.8, pnlPerTradeThreshold - 0.01);
            const normalTriggered = main.trades >= minTrades
                && main.wrPost < wrPostThreshold
                && main.pnlPerTrade < pnlPerTradeThreshold
                && volTriggered
                && lossTriggered;
            const severeTriggered = direction === 'DOWN'
                && main.trades >= Math.max(3, Math.floor(minTrades / 2))
                && main.pnlPerTrade < severePnlThreshold
                && cell.volRatio >= severeVolRatio
                && volQuantileTriggered
                && lossTriggered;
            const shockTriggered = normalTriggered || severeTriggered;
            const targetHoldSecs = severeTriggered ? severeHoldSecs : holdSecs;

            if (!cell.active && shockTriggered) {
                cell.active = true;
                cell.activeUntilTs = nowSec + targetHoldSecs;
                cell.releasePassStreak = 0;
                cell.triggerLane = severeTriggered ? 'severe' : 'normal';
                cell.lastTransitionAtMs = nowMs;
                cell.lastAction = `${severeTriggered ? 'enter_shock_severe' : 'enter_shock'}(wr=${main.wrPost.toFixed(3)},pnl=${main.pnlPerTrade.toFixed(3)},vol1h=${stat1h.avgAbsPnl.toFixed(4)},vol24h=${stat24h.avgAbsPnl.toFixed(4)},ratio=${cell.volRatio.toFixed(3)},pctl=${volPctlValue.toFixed(4)})`;
                cell.reasonCode = severeTriggered ? 'shock_trigger_severe' : 'shock_trigger';
            }

            if (cell.active) {
                if (nowSec < cell.activeUntilTs) {
                    // keep active until hold ends
                } else {
                    const releasePass = main.wrPost >= releaseWr && main.pnlPerTrade >= releasePnl;
                    if (releasePass) {
                        cell.releasePassStreak += 1;
                        if (cell.releasePassStreak >= releaseChecks) {
                            cell.active = false;
                            cell.activeUntilTs = 0;
                            cell.releasePassStreak = 0;
                            cell.lastAction = 'release_shock';
                            cell.reasonCode = 'shock_release';
                            cell.triggerLane = 'release';
                            cell.lastTransitionAtMs = nowMs;
                        } else {
                            cell.lastAction = `shock_release_pending(${cell.releasePassStreak}/${releaseChecks})`;
                            cell.reasonCode = 'shock_release_pending';
                        }
                    } else {
                        cell.releasePassStreak = 0;
                        cell.active = true;
                        cell.activeUntilTs = nowSec + (severeTriggered ? severeHoldSecs : holdSecs);
                        cell.triggerLane = severeTriggered ? 'severe' : 'normal';
                        cell.lastTransitionAtMs = nowMs;
                        cell.lastAction = severeTriggered ? 'shock_extend_severe' : 'shock_extend';
                        cell.reasonCode = severeTriggered ? 'shock_extend_severe' : 'shock_extend';
                    }
                }
            }

            if (!cell.active && !shockTriggered) {
                cell.reasonCode = 'normal';
                cell.lastAction = cell.lastAction === 'init' ? 'normal' : cell.lastAction;
            }

            const enforceBlocked = isEnforce && cell.active && nowSec < cell.activeUntilTs;
            cell.blocked = enforceBlocked;
            if (isShadow) {
                cell.blocked = false;
                cell.reasonCode = cell.reasonCode === 'normal' ? 'shadow_only' : cell.reasonCode;
            }
            t.shockRiskRuntimeBlockedBySymbolDir[shockRuntimeKey(symbol, direction)] = enforceBlocked;
        }
    }
}

function maybeUpdateShockRisk(t: TraderInstance, nowMs: number): void {
    const engine = resolveShockRiskEngine(t.config);
    if (engine === 'v2') {
        maybeUpdateShockRiskV2(t, nowMs);
        return;
    }
    const symbols = parseAllowedSymbols(t.config.allowedMarkets || '');
    for (const symbol of symbols) {
        t.shockRiskRuntimeBlockedBySymbolDir[shockRuntimeKey(symbol, 'UP')] = false;
        t.shockRiskRuntimeBlockedBySymbolDir[shockRuntimeKey(symbol, 'DOWN')] = false;
    }
}

function maybeUpdateComboPause(t: TraderInstance, nowMs: number): void {
    const enabled = resolveComboPauseEnabled(t.config);
    const routeMode = resolveComboPauseRouteMode(t);
    const symbols = parseAllowedSymbols(t.config.allowedMarkets || '');
    const targetSymbols = symbols.filter((s) => isComboPauseTargetSymbol(s));
    const symbolDirections = new Map<string, Array<'UP' | 'DOWN' | 'BOTH'>>();
    let hasActiveMode = false;
    let hasEnforceMode = false;
    for (const symbol of targetSymbols) {
        const directions = resolveComboPauseModeCandidates(t.config, symbol);
        symbolDirections.set(symbol, directions);
        for (const direction of directions) {
            const mode = resolveComboPauseModeByCell(t.config, symbol, direction);
            if (mode !== 'off') hasActiveMode = true;
            if (mode === 'enforce') hasEnforceMode = true;
        }
    }
    for (const symbol of symbols) {
        t.comboPauseRuntimeBlockedBySymbol[symbol] = false;
        t.comboPauseRuntimeBlockedBySymbolDir[symbolDirectionRuntimeKey(symbol, 'UP')] = false;
        t.comboPauseRuntimeBlockedBySymbolDir[symbolDirectionRuntimeKey(symbol, 'DOWN')] = false;
    }
    if (!enabled || targetSymbols.length === 0 || !hasActiveMode) {
        for (const symbol of targetSymbols) {
            const directions = symbolDirections.get(symbol) || ['BOTH'];
            for (const direction of directions) {
                const cell = getOrInitComboPauseCell(t, symbol, direction, routeMode);
                cell.active = false;
                cell.blocked = false;
                cell.activeUntilTs = 0;
                cell.checkedAtMs = nowMs;
                cell.reasonCode = enabled ? 'off_mode' : 'disabled';
                cell.lastAction = enabled ? 'off' : 'disabled';
                cell.triggerLane = direction === 'BOTH' ? 'symbol' : 'symbol_direction';
                cell.lastTransitionAtMs = nowMs;
                cell.sourceMode = routeMode;
            }
        }
        return;
    }

    const checkEverySec = Math.max(10, Number(t.config.comboPauseCheckSeconds ?? COMBO_PAUSE_CHECK_SECONDS_DEFAULT));
    const lastCheckMs = t.comboPauseLastCheckMsByMode[routeMode] || 0;
    if (nowMs - lastCheckMs < checkEverySec * 1000) return;
    t.comboPauseLastCheckMsByMode[routeMode] = nowMs;

    const tradesPath = path.join(process.cwd(), t.config.logsDir, `prediction_trades.${routeMode}.json`);
    if (!fs.existsSync(tradesPath)) {
        for (const symbol of targetSymbols) {
            const directions = symbolDirections.get(symbol) || ['BOTH'];
            for (const direction of directions) {
                const cell = getOrInitComboPauseCell(t, symbol, direction, routeMode);
                const symbolMode = resolveComboPauseModeByCell(t.config, symbol, direction);
                cell.checkedAtMs = nowMs;
                if (symbolMode === 'off') {
                    cell.active = false;
                    cell.activeUntilTs = 0;
                    cell.reasonCode = 'off_mode';
                    cell.lastAction = 'off';
                } else {
                    cell.reasonCode = 'no_data';
                    cell.lastAction = 'no_data_file';
                }
                cell.sourceMode = routeMode;
                cell.blocked = false;
            }
        }
        return;
    }
    const stat = fs.statSync(tradesPath);
    const mtimeMs = stat.mtimeMs;
    if (mtimeMs === t.comboPauseLastFileMtimeMsByMode[routeMode] && !hasEnforceMode) {
        for (const symbol of targetSymbols) {
            const directions = symbolDirections.get(symbol) || ['BOTH'];
            for (const direction of directions) {
                const cell = getOrInitComboPauseCell(t, symbol, direction, routeMode);
                const symbolMode = resolveComboPauseModeByCell(t.config, symbol, direction);
                cell.checkedAtMs = nowMs;
                if (symbolMode === 'off') {
                    cell.active = false;
                    cell.activeUntilTs = 0;
                    cell.reasonCode = 'off_mode';
                    cell.lastAction = 'off';
                } else {
                    cell.reasonCode = 'no_new_data';
                    cell.lastAction = 'shadow_no_new_data';
                }
                cell.sourceMode = routeMode;
                cell.blocked = false;
            }
        }
        return;
    }
    t.comboPauseLastFileMtimeMsByMode[routeMode] = mtimeMs;

    let rawEntries: Record<string, unknown>[] = [];
    try {
        const raw = JSON.parse(fs.readFileSync(tradesPath, 'utf8')) as unknown;
        if (!Array.isArray(raw)) return;
        rawEntries = raw as Record<string, unknown>[];
    } catch {
        for (const symbol of targetSymbols) {
            const directions = symbolDirections.get(symbol) || ['BOTH'];
            for (const direction of directions) {
                const cell = getOrInitComboPauseCell(t, symbol, direction, routeMode);
                const symbolMode = resolveComboPauseModeByCell(t.config, symbol, direction);
                cell.checkedAtMs = nowMs;
                if (symbolMode === 'off') {
                    cell.active = false;
                    cell.activeUntilTs = 0;
                    cell.reasonCode = 'off_mode';
                    cell.lastAction = 'off';
                } else {
                    cell.reasonCode = 'parse_error';
                    cell.lastAction = 'parse_error';
                }
                cell.sourceMode = routeMode;
                cell.blocked = false;
            }
        }
        return;
    }

    const baseCapital = resolveComboPauseBaseCapital(t);
    const nowSec = Math.floor(nowMs / 1000);
    const cutoff2h = nowMs - 2 * 3600 * 1000;
    const profile = t.config.profile ?? 'default';

    for (const symbol of targetSymbols) {
        const directions = symbolDirections.get(symbol) || ['BOTH'];
        for (const direction of directions) {
            const symbolMode = resolveComboPauseModeByCell(t.config, symbol, direction);
            const isEnforce = symbolMode === 'enforce';
            const runtimeKey = direction === 'BOTH' ? null : symbolDirectionRuntimeKey(symbol, direction);
            if (symbolMode === 'off') {
                const cell = getOrInitComboPauseCell(t, symbol, direction, routeMode);
                cell.active = false;
                cell.blocked = false;
                cell.activeUntilTs = 0;
                cell.checkedAtMs = nowMs;
                cell.reasonCode = 'off_mode';
                cell.lastAction = 'off';
                cell.sourceMode = routeMode;
                if (runtimeKey) t.comboPauseRuntimeBlockedBySymbolDir[runtimeKey] = false;
                else t.comboPauseRuntimeBlockedBySymbol[symbol] = false;
                continue;
            }
            const drawdown2hThreshold = Math.max(
                0.01,
                Math.min(
                    0.9,
                    getComboPauseNumber(
                        t,
                        symbol,
                        direction,
                        'comboPauseDrawdown2h',
                        Number(t.config.comboPauseDrawdown2h ?? COMBO_PAUSE_DRAWDOWN_2H_DEFAULT),
                    ),
                ),
            );
            const holdMinutes = Math.max(
                15,
                Math.round(
                    getComboPauseNumber(
                        t,
                        symbol,
                        direction,
                        'comboPauseHoldMinutes',
                        Number(t.config.comboPauseHoldMinutes ?? COMBO_PAUSE_HOLD_MINUTES_DEFAULT),
                    ),
                ),
            );
            const holdSecs = holdMinutes * 60;
            const stat2h = calcComboPauseWindow(rawEntries, symbol, routeMode, cutoff2h, direction);
            const drawdown2h = baseCapital > 0 ? Math.max(0, -stat2h.pnl / baseCapital) : 0;
            const cell = getOrInitComboPauseCell(t, symbol, direction, routeMode);
            cell.checkedAtMs = nowMs;
            cell.sourceMode = routeMode;
            cell.pnl2h = stat2h.pnl;
            cell.baseCapital = baseCapital;
            cell.drawdown2h = drawdown2h;
            cell.sampleCount = stat2h.sampleCount;

            const shockCell = direction === 'UP' || direction === 'DOWN'
                ? t.shockRiskCells.get(shockRiskCellKey(profile, t.config.name, symbol, direction, routeMode))
                : null;
            const expectCell = direction === 'DOWN'
                ? t.expectancyGateCells.get(expectancyGateCellKey(profile, t.config.name, symbol, 'DOWN', routeMode))
                : null;
            const crossLayerTriggered = direction === 'DOWN'
                && Boolean(shockCell?.active && nowSec < Number(shockCell?.activeUntilTs || 0))
                && String(expectCell?.status || 'normal').toLowerCase() === 'blocked';
            const triggered = drawdown2h >= drawdown2hThreshold || crossLayerTriggered;
            if (!cell.active && triggered) {
                cell.active = true;
                cell.activeUntilTs = nowSec + holdSecs;
                cell.lastAction = crossLayerTriggered
                    ? `combo_pause_trigger_cross_layer(dir=${direction},drawdown2h=${drawdown2h.toFixed(4)})`
                    : `combo_pause_trigger(drawdown2h=${drawdown2h.toFixed(4)},pnl2h=${stat2h.pnl.toFixed(3)},dir=${direction})`;
                cell.reasonCode = crossLayerTriggered ? 'combo_pause_cross_layer' : 'combo_pause_trigger';
                cell.triggerLane = crossLayerTriggered
                    ? 'symbol_direction'
                    : (direction === 'BOTH' ? 'symbol' : 'symbol_direction');
                cell.lastTransitionAtMs = nowMs;
            }

            if (cell.active && nowSec >= cell.activeUntilTs && !triggered) {
                cell.active = false;
                cell.activeUntilTs = 0;
                cell.lastAction = 'combo_pause_release';
                cell.reasonCode = 'combo_pause_release';
                cell.triggerLane = 'release';
                cell.lastTransitionAtMs = nowMs;
            } else if (cell.active && triggered) {
                cell.activeUntilTs = Math.max(cell.activeUntilTs, nowSec + holdSecs);
                if (cell.reasonCode !== 'combo_pause_trigger') {
                    cell.reasonCode = 'combo_pause_extend';
                    cell.lastAction = `combo_pause_extend(drawdown2h=${drawdown2h.toFixed(4)},dir=${direction})`;
                }
            } else if (!cell.active) {
                cell.reasonCode = 'normal';
                cell.lastAction = 'normal';
            }

            const enforceBlocked = isEnforce && cell.active && nowSec < cell.activeUntilTs;
            cell.blocked = enforceBlocked;
            if (runtimeKey) t.comboPauseRuntimeBlockedBySymbolDir[runtimeKey] = enforceBlocked;
            else t.comboPauseRuntimeBlockedBySymbol[symbol] = enforceBlocked;
        }
    }
}

function maybeUpdatePredictionCalibration(t: TraderInstance, nowMs: number): void {
    const routeMode = resolveCalibrationRouteMode(t);
    const symbols = parseAllowedSymbols(t.config.allowedMarkets || '');
    const targetSymbols = symbols.filter((s) => isCalibrationTargetSymbol(s));
    if (targetSymbols.length === 0) return;

    let hasActiveMode = false;
    for (const symbol of targetSymbols) {
        for (const direction of ['UP', 'DOWN'] as const) {
            if (resolveCalibrationMode(t.config, symbol, direction) !== 'off') {
                hasActiveMode = true;
            }
        }
    }
    if (!hasActiveMode) {
        for (const symbol of targetSymbols) {
            for (const direction of ['UP', 'DOWN'] as const) {
                const cell = getOrInitCalibrationCell(t, symbol, direction, routeMode);
                cell.checkedAtMs = nowMs;
                cell.active = false;
                cell.lastAction = 'off';
                cell.reasonCode = 'off_mode';
                cell.sourceMode = routeMode;
            }
        }
        return;
    }

    const checkEverySec = Math.max(10, Number(t.config.calibrationCheckSeconds ?? CALIBRATION_CHECK_SECONDS_DEFAULT));
    if (nowMs - (t.calibrationLastCheckMsByMode[routeMode] || 0) < checkEverySec * 1000) return;
    t.calibrationLastCheckMsByMode[routeMode] = nowMs;

    const readResult = readTradeEntriesForRoute(t.config.logsDir, routeMode);
    if (!readResult.ok) {
        for (const symbol of targetSymbols) {
            for (const direction of ['UP', 'DOWN'] as const) {
                const cell = getOrInitCalibrationCell(t, symbol, direction, routeMode);
                cell.checkedAtMs = nowMs;
                cell.active = false;
                cell.lastAction = readResult.reason;
                cell.reasonCode = readResult.reason;
                cell.sourceMode = routeMode;
            }
        }
        return;
    }
    t.calibrationLastFileMtimeMsByMode[routeMode] = readResult.mtimeMs;

    for (const symbol of targetSymbols) {
        for (const direction of ['UP', 'DOWN'] as const) {
            const cell = getOrInitCalibrationCell(t, symbol, direction, routeMode);
            const mode = resolveCalibrationMode(t.config, symbol, direction);
            const method = getCalibrationMethod(t, symbol, direction);
            const lookbackHours = Math.max(1, Math.round(getCalibrationNumber(t, symbol, direction, 'calibrationLookbackHours', Number(t.config.calibrationLookbackHours ?? CALIBRATION_LOOKBACK_HOURS_DEFAULT))));
            const minSamples = Math.max(1, Math.round(getCalibrationNumber(t, symbol, direction, 'calibrationMinSamples', Number(t.config.calibrationMinSamples ?? CALIBRATION_MIN_SAMPLES_DEFAULT))));
            const cutoffMs = nowMs - lookbackHours * 3600 * 1000;
            const stats = calcCalibrationWindowStats(readResult.entries, symbol, direction, routeMode, cutoffMs, lookbackHours);
            cell.checkedAtMs = nowMs;
            cell.sourceMode = routeMode;
            cell.method = method;
            cell.minSamples = minSamples;
            cell.stats = stats;
            cell.active = mode !== 'off' && stats.sampleCount >= minSamples;
            if (mode === 'off') {
                cell.lastAction = 'off';
                cell.reasonCode = 'off_mode';
            } else if (stats.sampleCount < minSamples) {
                cell.lastAction = `insufficient_sample(${stats.sampleCount}/${minSamples})`;
                cell.reasonCode = 'insufficient_sample';
            } else if (mode === 'shadow') {
                cell.lastAction = `shadow_${method}(brier=${stats.brier.toFixed(4)},ece=${stats.ece.toFixed(4)})`;
                cell.reasonCode = 'shadow_only';
            } else {
                cell.lastAction = `enforce_${method}(brier=${stats.brier.toFixed(4)},ece=${stats.ece.toFixed(4)})`;
                cell.reasonCode = 'calibration_active';
            }
        }
    }
}

function maybeUpdateExpectancyGate(t: TraderInstance, nowMs: number): void {
    const routeMode = resolveExpectancyGateRouteMode(t);
    const symbols = parseAllowedSymbols(t.config.allowedMarkets || '');
    const targetSymbols = symbols.filter((s) => isExpectancyGateTargetSymbol(s));
    for (const symbol of symbols) {
        t.expectancyGateRuntimeBlockedBySymbolDir[symbolDirectionRuntimeKey(symbol, 'UP')] = false;
        t.expectancyGateRuntimeBlockedBySymbolDir[symbolDirectionRuntimeKey(symbol, 'DOWN')] = false;
        t.expectancyGateRuntimeScaleBySymbolDir[symbolDirectionRuntimeKey(symbol, 'UP')] = 1;
        t.expectancyGateRuntimeScaleBySymbolDir[symbolDirectionRuntimeKey(symbol, 'DOWN')] = 1;
        t.expectancyGateRuntimeExtraDeltaBySymbolDir[symbolDirectionRuntimeKey(symbol, 'UP')] = 0;
        t.expectancyGateRuntimeExtraDeltaBySymbolDir[symbolDirectionRuntimeKey(symbol, 'DOWN')] = 0;
    }
    if (targetSymbols.length === 0) return;

    let hasActiveMode = false;
    for (const symbol of targetSymbols) {
        for (const direction of ['UP', 'DOWN'] as const) {
            if (resolveExpectancyGateMode(t.config, symbol, direction) !== 'off') {
                hasActiveMode = true;
            }
        }
    }
    if (!hasActiveMode) {
        for (const symbol of targetSymbols) {
            for (const direction of ['UP', 'DOWN'] as const) {
                const cell = getOrInitExpectancyGateCell(t, symbol, direction, routeMode);
                cell.checkedAtMs = nowMs;
                cell.status = 'normal';
                cell.blocked = false;
                cell.degradedBetScale = 1;
                cell.degradedExtraDelta = 0;
                cell.activeUntilTs = 0;
                cell.blockedEnterPassStreak = 0;
                cell.releasePassStreak = 0;
                cell.lastAction = 'off';
                cell.reasonCode = 'off_mode';
                cell.sourceMode = routeMode;
            }
        }
        return;
    }

    const checkEverySec = Math.max(10, Number(t.config.expectancyGateCheckSeconds ?? EXPECTANCY_GATE_CHECK_SECONDS_DEFAULT));
    if (nowMs - (t.expectancyGateLastCheckMsByMode[routeMode] || 0) < checkEverySec * 1000) return;
    t.expectancyGateLastCheckMsByMode[routeMode] = nowMs;

    const readResult = readTradeEntriesForRoute(t.config.logsDir, routeMode);
    if (!readResult.ok) {
        for (const symbol of targetSymbols) {
            for (const direction of ['UP', 'DOWN'] as const) {
                const cell = getOrInitExpectancyGateCell(t, symbol, direction, routeMode);
                cell.checkedAtMs = nowMs;
                cell.status = 'normal';
                cell.blocked = false;
                cell.degradedBetScale = 1;
                cell.degradedExtraDelta = 0;
                cell.activeUntilTs = 0;
                cell.blockedEnterPassStreak = 0;
                cell.releasePassStreak = 0;
                cell.lastAction = readResult.reason;
                cell.reasonCode = readResult.reason;
                cell.sourceMode = routeMode;
            }
        }
        return;
    }
    t.expectancyGateLastFileMtimeMsByMode[routeMode] = readResult.mtimeMs;

    const nowSec = Math.floor(nowMs / 1000);
    for (const symbol of targetSymbols) {
        for (const direction of ['UP', 'DOWN'] as const) {
            const cell = getOrInitExpectancyGateCell(t, symbol, direction, routeMode);
            const priorityFastLane = isAcuteExpectancyPriorityCluster((t.config.profile ?? 'default'), symbol, direction);
            const mode = resolveExpectancyGateMode(t.config, symbol, direction);
            const lookbackHours = Math.max(1, Math.round(getExpectancyGateNumber(t, symbol, direction, 'expectancyGateLookbackHours', Number(t.config.expectancyGateLookbackHours ?? EXPECTANCY_GATE_LOOKBACK_HOURS_DEFAULT))));
            const lookbackFastHours = Math.max(1, Math.round(getExpectancyGateNumber(t, symbol, direction, 'expectancyGateLookbackHoursFast', Number(t.config.expectancyGateLookbackHoursFast ?? EXPECTANCY_GATE_LOOKBACK_FAST_HOURS_DEFAULT))));
            const minTrades = Math.max(1, Math.round(getExpectancyGateNumber(t, symbol, direction, 'expectancyGateMinTrades', Number(t.config.expectancyGateMinTrades ?? EXPECTANCY_GATE_MIN_TRADES_DEFAULT))));
            const minTradesBlocked = Math.max(
                minTrades,
                Math.round(
                    getExpectancyGateNumber(
                        t,
                        symbol,
                        direction,
                        'expectancyGateMinTradesBlocked',
                        Number(t.config.expectancyGateMinTradesBlocked ?? EXPECTANCY_GATE_MIN_TRADES_BLOCKED_DEFAULT),
                    ),
                ),
            );
            const minTradesFast = Math.max(
                1,
                Math.round(
                    getExpectancyGateNumber(
                        t,
                        symbol,
                        direction,
                        'expectancyGateMinTradesFast',
                        Number(t.config.expectancyGateMinTradesFast ?? EXPECTANCY_GATE_MIN_TRADES_FAST_DEFAULT),
                    ),
                ),
            );
            const wrDegraded = Math.max(0, Math.min(1, getExpectancyGateNumber(t, symbol, direction, 'expectancyGateWrDegraded', Number(t.config.expectancyGateWrDegraded ?? EXPECTANCY_GATE_WR_DEGRADED_DEFAULT))));
            const pnlDegraded = getExpectancyGateNumber(t, symbol, direction, 'expectancyGatePnlDegraded', Number(t.config.expectancyGatePnlDegraded ?? EXPECTANCY_GATE_PNL_DEGRADED_DEFAULT));
            const degradedBetScale = Math.max(0, Math.min(1, getExpectancyGateNumber(t, symbol, direction, 'expectancyGateDegradedBetScale', Number(t.config.expectancyGateDegradedBetScale ?? EXPECTANCY_GATE_DEGRADED_BET_SCALE_DEFAULT))));
            const degradedExtraDelta = Math.max(0, getExpectancyGateNumber(t, symbol, direction, 'expectancyGateDegradedExtraDelta', Number(t.config.expectancyGateDegradedExtraDelta ?? EXPECTANCY_GATE_DEGRADED_EXTRA_DELTA_DEFAULT)));
            const wrBlocked = Math.max(0, Math.min(1, getExpectancyGateNumber(t, symbol, direction, 'expectancyGateWrBlocked', Number(t.config.expectancyGateWrBlocked ?? EXPECTANCY_GATE_WR_BLOCKED_DEFAULT))));
            const pnlBlocked = getExpectancyGateNumber(t, symbol, direction, 'expectancyGatePnlBlocked', Number(t.config.expectancyGatePnlBlocked ?? EXPECTANCY_GATE_PNL_BLOCKED_DEFAULT));
            const wrFastBlocked = Math.max(0, Math.min(1, getExpectancyGateNumber(t, symbol, direction, 'expectancyGateWrFastBlocked', Number(t.config.expectancyGateWrFastBlocked ?? EXPECTANCY_GATE_WR_FAST_BLOCKED_DEFAULT))));
            const pnlFastBlocked = getExpectancyGateNumber(t, symbol, direction, 'expectancyGatePnlFastBlocked', Number(t.config.expectancyGatePnlFastBlocked ?? EXPECTANCY_GATE_PNL_FAST_BLOCKED_DEFAULT));
            const blockedEnterChecks = Math.max(
                1,
                Math.round(
                    getExpectancyGateNumber(
                        t,
                        symbol,
                        direction,
                        'expectancyGateBlockedEnterChecks',
                        Number(t.config.expectancyGateBlockedEnterChecks ?? EXPECTANCY_GATE_BLOCKED_ENTER_CHECKS_DEFAULT),
                    ),
                ),
            );
            const releaseChecks = Math.max(1, Math.round(getExpectancyGateNumber(t, symbol, direction, 'expectancyGateReleaseChecks', Number(t.config.expectancyGateReleaseChecks ?? EXPECTANCY_GATE_RELEASE_CHECKS_DEFAULT))));
            const releaseWr = Math.max(0, Math.min(1, getExpectancyGateNumber(t, symbol, direction, 'expectancyGateReleaseWr', Number(t.config.expectancyGateReleaseWr ?? EXPECTANCY_GATE_RELEASE_WR_DEFAULT))));
            const releasePnl = getExpectancyGateNumber(t, symbol, direction, 'expectancyGateReleasePnl', Number(t.config.expectancyGateReleasePnl ?? EXPECTANCY_GATE_RELEASE_PNL_DEFAULT));
            const cutoffMs = nowMs - lookbackHours * 3600 * 1000;
            const fastCutoffMs = nowMs - lookbackFastHours * 3600 * 1000;
            const stats = calcExpectancyGateWindowStats(readResult.entries, symbol, direction, routeMode, cutoffMs, lookbackHours);
            const fastStats = calcExpectancyGateWindowStats(readResult.entries, symbol, direction, routeMode, fastCutoffMs, lookbackFastHours);
            cell.checkedAtMs = nowMs;
            cell.sourceMode = routeMode;
            cell.stats = stats;

            if (mode === 'off') {
                cell.status = 'normal';
                cell.blocked = false;
                cell.degradedBetScale = 1;
                cell.degradedExtraDelta = 0;
                cell.activeUntilTs = 0;
                cell.blockedEnterPassStreak = 0;
                cell.releasePassStreak = 0;
                cell.reasonCode = 'off_mode';
                cell.lastAction = 'off';
                t.expectancyGateRuntimeBlockedBySymbolDir[symbolDirectionRuntimeKey(symbol, direction)] = false;
                t.expectancyGateRuntimeScaleBySymbolDir[symbolDirectionRuntimeKey(symbol, direction)] = 1;
                t.expectancyGateRuntimeExtraDeltaBySymbolDir[symbolDirectionRuntimeKey(symbol, direction)] = 0;
                continue;
            }

            const manualAction = getExpectancyGateManualAction(t, symbol, direction);
            if (manualAction !== 'normal') {
                const manualScale = Math.max(
                    0,
                    Math.min(
                        1,
                        getExpectancyGateNumber(
                            t,
                            symbol,
                            direction,
                            'expectancyGateManualDegradedBetScale',
                            Number(t.config.expectancyGateDegradedBetScale ?? EXPECTANCY_GATE_DEGRADED_BET_SCALE_DEFAULT),
                        ),
                    ),
                );
                const manualExtraDelta = Math.max(
                    0,
                    getExpectancyGateNumber(
                        t,
                        symbol,
                        direction,
                        'expectancyGateManualDegradedExtraDelta',
                        Number(t.config.expectancyGateDegradedExtraDelta ?? EXPECTANCY_GATE_DEGRADED_EXTRA_DELTA_DEFAULT),
                    ),
                );
                cell.status = manualAction === 'blocked' ? 'blocked' : 'degraded';
                cell.blocked = mode === 'enforce' && manualAction === 'blocked';
                cell.degradedBetScale = mode === 'enforce' && manualAction === 'degraded' ? manualScale : 1;
                cell.degradedExtraDelta = mode === 'enforce' && manualAction === 'degraded' ? manualExtraDelta : 0;
                cell.activeUntilTs = cell.blocked ? nowSec + checkEverySec : 0;
                cell.blockedEnterPassStreak = 0;
                cell.releasePassStreak = 0;
                cell.triggerLane = 'manual';
                cell.lastTransitionAtMs = nowMs;
                cell.reasonCode = manualAction === 'blocked' ? 'manual_block' : 'manual_degrade';
                cell.lastAction = manualAction === 'blocked'
                    ? 'manual_block'
                    : `manual_degrade(scale=${cell.degradedBetScale.toFixed(3)},delta=${cell.degradedExtraDelta.toFixed(3)})`;
                t.expectancyGateRuntimeBlockedBySymbolDir[symbolDirectionRuntimeKey(symbol, direction)] = cell.blocked;
                t.expectancyGateRuntimeScaleBySymbolDir[symbolDirectionRuntimeKey(symbol, direction)] = cell.degradedBetScale;
                t.expectancyGateRuntimeExtraDeltaBySymbolDir[symbolDirectionRuntimeKey(symbol, direction)] = cell.degradedExtraDelta;
                continue;
            }

            if (stats.trades < minTrades) {
                cell.status = 'normal';
                cell.blocked = false;
                cell.degradedBetScale = 1;
                cell.degradedExtraDelta = 0;
                cell.activeUntilTs = 0;
                cell.blockedEnterPassStreak = 0;
                cell.releasePassStreak = 0;
                cell.reasonCode = 'insufficient_sample';
                cell.lastAction = `insufficient_sample(${stats.trades}/${minTrades})`;
                t.expectancyGateRuntimeBlockedBySymbolDir[symbolDirectionRuntimeKey(symbol, direction)] = false;
                t.expectancyGateRuntimeScaleBySymbolDir[symbolDirectionRuntimeKey(symbol, direction)] = 1;
                t.expectancyGateRuntimeExtraDeltaBySymbolDir[symbolDirectionRuntimeKey(symbol, direction)] = 0;
                continue;
            }

            let targetStatus: ExpectancyGateStatus = (stats.wrPost < wrDegraded || stats.pnlPerTrade < pnlDegraded)
                ? 'degraded'
                : 'normal';
            const blockedEligible = stats.trades >= minTradesBlocked
                && (stats.wrPost < wrBlocked || stats.pnlPerTrade < pnlBlocked);
            const fastBlockedEligible = priorityFastLane
                && fastStats.trades >= minTradesFast
                && (fastStats.wrPost < wrFastBlocked || fastStats.pnlPerTrade < pnlFastBlocked);

            if (fastBlockedEligible) {
                cell.blockedEnterPassStreak = blockedEnterChecks;
                targetStatus = 'blocked';
                cell.triggerLane = 'fast';
                cell.lastTransitionAtMs = nowMs;
            } else if (blockedEligible) {
                cell.blockedEnterPassStreak += 1;
                if (cell.blockedEnterPassStreak >= blockedEnterChecks) {
                    targetStatus = 'blocked';
                    cell.triggerLane = 'main';
                    cell.lastTransitionAtMs = nowMs;
                } else if (targetStatus === 'normal') {
                    targetStatus = 'degraded';
                    cell.reasonCode = 'blocked_pending';
                    cell.lastAction = `blocked_pending(${cell.blockedEnterPassStreak}/${blockedEnterChecks})`;
                }
            } else {
                cell.blockedEnterPassStreak = 0;
            }

            if (cell.status !== 'normal' && targetStatus === 'normal') {
                const releasePass = stats.wrPost >= releaseWr && stats.pnlPerTrade >= releasePnl;
                if (releasePass) {
                    cell.releasePassStreak += 1;
                    if (cell.releasePassStreak < releaseChecks) {
                        targetStatus = cell.status;
                        cell.reasonCode = 'release_pending';
                        cell.lastAction = `release_pending(${cell.releasePassStreak}/${releaseChecks})`;
                    } else {
                        cell.releasePassStreak = 0;
                    }
                } else {
                    cell.releasePassStreak = 0;
                    targetStatus = cell.status;
                    cell.reasonCode = 'release_extend';
                    cell.lastAction = `release_extend(wr=${stats.wrPost.toFixed(3)},pnl=${stats.pnlPerTrade.toFixed(3)})`;
                }
            } else if (targetStatus !== 'normal') {
                cell.releasePassStreak = 0;
            }

            cell.status = targetStatus;
            cell.blocked = mode === 'enforce' && targetStatus === 'blocked';
            cell.degradedBetScale = mode === 'enforce' && targetStatus === 'degraded' ? degradedBetScale : 1;
            cell.degradedExtraDelta = mode === 'enforce' && targetStatus === 'degraded' ? degradedExtraDelta : 0;
            cell.activeUntilTs = cell.blocked ? nowSec + checkEverySec : 0;
            if (mode === 'shadow') {
                const shadowLabel = blockedEligible && cell.blockedEnterPassStreak < blockedEnterChecks
                    ? 'shadow_blocked_pending'
                    : (targetStatus === 'normal' ? 'shadow_ok' : `shadow_${targetStatus}`);
                cell.reasonCode = shadowLabel;
                cell.lastAction = `${shadowLabel}(wr=${stats.wrPost.toFixed(3)},pnl=${stats.pnlPerTrade.toFixed(3)},n=${stats.trades})`;
                cell.blocked = false;
                cell.degradedBetScale = 1;
                cell.degradedExtraDelta = 0;
                cell.activeUntilTs = 0;
            } else if (targetStatus === 'blocked') {
                cell.reasonCode = fastBlockedEligible ? 'blocked_fast' : 'blocked';
                cell.lastAction = `${cell.reasonCode}(${cell.blockedEnterPassStreak}/${blockedEnterChecks},wr=${stats.wrPost.toFixed(3)},pnl=${stats.pnlPerTrade.toFixed(3)},n=${stats.trades},fastWr=${fastStats.wrPost.toFixed(3)},fastPnl=${fastStats.pnlPerTrade.toFixed(3)})`;
                cell.degradedBetScale = 1;
                cell.degradedExtraDelta = 0;
            } else if (targetStatus === 'degraded') {
                cell.reasonCode = blockedEligible && cell.blockedEnterPassStreak < blockedEnterChecks ? 'blocked_pending' : 'degraded';
                cell.lastAction = `${cell.reasonCode}(scale=${degradedBetScale.toFixed(3)},wr=${stats.wrPost.toFixed(3)},pnl=${stats.pnlPerTrade.toFixed(3)},n=${stats.trades},hold=${cell.blockedEnterPassStreak}/${blockedEnterChecks})`;
            } else if (cell.reasonCode !== 'release_pending' && cell.reasonCode !== 'release_extend') {
                cell.reasonCode = 'normal';
                cell.lastAction = 'normal';
                cell.degradedBetScale = 1;
                cell.degradedExtraDelta = 0;
                cell.triggerLane = 'normal';
            }
            t.expectancyGateRuntimeBlockedBySymbolDir[symbolDirectionRuntimeKey(symbol, direction)] = cell.blocked;
            t.expectancyGateRuntimeScaleBySymbolDir[symbolDirectionRuntimeKey(symbol, direction)] = cell.degradedBetScale;
            t.expectancyGateRuntimeExtraDeltaBySymbolDir[symbolDirectionRuntimeKey(symbol, direction)] = cell.degradedExtraDelta;
        }
    }
}

function maybeUpdateThresholdDrift(t: TraderInstance, nowMs: number): void {
    const routeMode = resolveThresholdDriftRouteMode(t);
    const symbols = parseAllowedSymbols(t.config.allowedMarkets || '');
    const targetSymbols = symbols.filter((s) => isThresholdDriftTargetSymbol(s));
    if (targetSymbols.length === 0) return;

    let hasActiveMode = false;
    for (const symbol of targetSymbols) {
        for (const direction of ['UP', 'DOWN'] as const) {
            if (resolveThresholdDriftMode(t.config, symbol, direction) !== 'off') {
                hasActiveMode = true;
            }
        }
    }
    if (!hasActiveMode) {
        for (const symbol of targetSymbols) {
            for (const direction of ['UP', 'DOWN'] as const) {
                const cell = getOrInitThresholdDriftCell(t, symbol, direction, routeMode);
                cell.checkedAtMs = nowMs;
                cell.status = 'normal';
                cell.active = false;
                cell.thresholdDelta = 0;
                cell.releasePassStreak = 0;
                cell.lastAction = 'off';
                cell.reasonCode = 'off_mode';
                cell.sourceMode = routeMode;
                t.thresholdDriftRuntimeExtraDeltaBySymbolDir[symbolDirectionRuntimeKey(symbol, direction)] = 0;
            }
        }
        return;
    }

    const checkEverySec = Math.max(10, Number(t.config.thresholdDriftCheckSeconds ?? THRESHOLD_DRIFT_CHECK_SECONDS_DEFAULT));
    if (nowMs - (t.thresholdDriftLastCheckMsByMode[routeMode] || 0) < checkEverySec * 1000) return;
    t.thresholdDriftLastCheckMsByMode[routeMode] = nowMs;

    const readResult = readTradeEntriesForRoute(t.config.logsDir, routeMode);
    if (!readResult.ok) {
        for (const symbol of targetSymbols) {
            for (const direction of ['UP', 'DOWN'] as const) {
                const cell = getOrInitThresholdDriftCell(t, symbol, direction, routeMode);
                cell.checkedAtMs = nowMs;
                cell.status = 'normal';
                cell.active = false;
                cell.thresholdDelta = 0;
                cell.releasePassStreak = 0;
                cell.lastAction = readResult.reason;
                cell.reasonCode = readResult.reason;
                cell.sourceMode = routeMode;
                t.thresholdDriftRuntimeExtraDeltaBySymbolDir[symbolDirectionRuntimeKey(symbol, direction)] = 0;
            }
        }
        return;
    }
    t.thresholdDriftLastFileMtimeMsByMode[routeMode] = readResult.mtimeMs;

    for (const symbol of targetSymbols) {
        for (const direction of ['UP', 'DOWN'] as const) {
            const cell = getOrInitThresholdDriftCell(t, symbol, direction, routeMode);
            const mode = resolveThresholdDriftMode(t.config, symbol, direction);
            const lookbackHours = Math.max(
                1,
                Math.round(
                    getThresholdDriftNumber(
                        t,
                        symbol,
                        direction,
                        'thresholdDriftLookbackHours',
                        Number(t.config.thresholdDriftLookbackHours ?? THRESHOLD_DRIFT_LOOKBACK_HOURS_DEFAULT),
                    ),
                ),
            );
            const minTrades = Math.max(
                1,
                Math.round(
                    getThresholdDriftNumber(
                        t,
                        symbol,
                        direction,
                        'thresholdDriftMinTrades',
                        Number(t.config.thresholdDriftMinTrades ?? THRESHOLD_DRIFT_MIN_TRADES_DEFAULT),
                    ),
                ),
            );
            const wrDegraded = Math.max(
                0,
                Math.min(
                    1,
                    getThresholdDriftNumber(
                        t,
                        symbol,
                        direction,
                        'thresholdDriftWrDegraded',
                        Number(t.config.thresholdDriftWrDegraded ?? THRESHOLD_DRIFT_WR_DEGRADED_DEFAULT),
                    ),
                ),
            );
            const pnlDegraded = getThresholdDriftNumber(
                t,
                symbol,
                direction,
                'thresholdDriftPnlDegraded',
                Number(t.config.thresholdDriftPnlDegraded ?? THRESHOLD_DRIFT_PNL_DEGRADED_DEFAULT),
            );
            const deltaStep = Math.max(
                0,
                getThresholdDriftNumber(
                    t,
                    symbol,
                    direction,
                    'thresholdDriftDeltaStep',
                    Number(t.config.thresholdDriftDeltaStep ?? THRESHOLD_DRIFT_DELTA_STEP_DEFAULT),
                ),
            );
            const deltaMax = Math.max(
                deltaStep,
                getThresholdDriftNumber(
                    t,
                    symbol,
                    direction,
                    'thresholdDriftDeltaMax',
                    Number(t.config.thresholdDriftDeltaMax ?? THRESHOLD_DRIFT_DELTA_MAX_DEFAULT),
                ),
            );
            const releaseChecks = Math.max(
                1,
                Math.round(
                    getThresholdDriftNumber(
                        t,
                        symbol,
                        direction,
                        'thresholdDriftReleaseChecks',
                        Number(t.config.thresholdDriftReleaseChecks ?? THRESHOLD_DRIFT_RELEASE_CHECKS_DEFAULT),
                    ),
                ),
            );
            const releaseWr = Math.max(
                0,
                Math.min(
                    1,
                    getThresholdDriftNumber(
                        t,
                        symbol,
                        direction,
                        'thresholdDriftReleaseWr',
                        Number(t.config.thresholdDriftReleaseWr ?? THRESHOLD_DRIFT_RELEASE_WR_DEFAULT),
                    ),
                ),
            );
            const releasePnl = getThresholdDriftNumber(
                t,
                symbol,
                direction,
                'thresholdDriftReleasePnl',
                Number(t.config.thresholdDriftReleasePnl ?? THRESHOLD_DRIFT_RELEASE_PNL_DEFAULT),
            );
            const cutoffMs = nowMs - lookbackHours * 3600 * 1000;
            const stats = calcThresholdDriftWindowStats(readResult.entries, symbol, direction, routeMode, cutoffMs, lookbackHours);
            cell.checkedAtMs = nowMs;
            cell.sourceMode = routeMode;
            cell.stats = stats;

            if (mode === 'off') {
                cell.status = 'normal';
                cell.active = false;
                cell.thresholdDelta = 0;
                cell.releasePassStreak = 0;
                cell.lastAction = 'off';
                cell.reasonCode = 'off_mode';
                t.thresholdDriftRuntimeExtraDeltaBySymbolDir[symbolDirectionRuntimeKey(symbol, direction)] = 0;
                continue;
            }

            if (stats.trades < minTrades) {
                cell.status = 'normal';
                cell.active = false;
                cell.thresholdDelta = 0;
                cell.releasePassStreak = 0;
                cell.lastAction = `insufficient_sample(${stats.trades}/${minTrades})`;
                cell.reasonCode = 'insufficient_sample';
                t.thresholdDriftRuntimeExtraDeltaBySymbolDir[symbolDirectionRuntimeKey(symbol, direction)] = 0;
                continue;
            }

            const degraded = stats.wrPost < wrDegraded || stats.pnlPerTrade < pnlDegraded;
            if (degraded) {
                cell.releasePassStreak = 0;
                const wrGap = Math.max(0, wrDegraded - stats.wrPost);
                const pnlGap = Math.max(0, pnlDegraded - stats.pnlPerTrade);
                const scaled = deltaStep * (1 + wrGap * 10 + pnlGap * 5);
                const delta = Math.min(deltaMax, Math.max(deltaStep, scaled));
                cell.status = 'drifted';
                cell.active = mode === 'enforce';
                cell.thresholdDelta = mode === 'enforce' ? delta : 0;
                if (mode === 'shadow') {
                    cell.reasonCode = 'shadow_drifted';
                    cell.lastAction = `shadow_drifted(delta=${delta.toFixed(4)},wr=${stats.wrPost.toFixed(3)},pnl=${stats.pnlPerTrade.toFixed(3)},n=${stats.trades})`;
                } else {
                    cell.reasonCode = 'drifted';
                    cell.lastAction = `drifted(delta=${delta.toFixed(4)},wr=${stats.wrPost.toFixed(3)},pnl=${stats.pnlPerTrade.toFixed(3)},n=${stats.trades})`;
                }
            } else {
                const releasePass = stats.wrPost >= releaseWr && stats.pnlPerTrade >= releasePnl;
                if (cell.status === 'drifted' && releasePass && cell.releasePassStreak + 1 < releaseChecks) {
                    cell.releasePassStreak += 1;
                    cell.status = 'drifted';
                    cell.active = mode === 'enforce';
                    cell.thresholdDelta = mode === 'enforce' ? Math.min(deltaMax, Math.max(deltaStep, cell.thresholdDelta || deltaStep)) : 0;
                    cell.reasonCode = 'release_pending';
                    cell.lastAction = `release_pending(${cell.releasePassStreak}/${releaseChecks})`;
                } else {
                    cell.releasePassStreak = 0;
                    cell.status = 'normal';
                    cell.active = false;
                    cell.thresholdDelta = 0;
                    cell.reasonCode = 'normal';
                    cell.lastAction = 'normal';
                }
            }
            t.thresholdDriftRuntimeExtraDeltaBySymbolDir[symbolDirectionRuntimeKey(symbol, direction)] = cell.thresholdDelta;
        }
    }
}

function maybeUpdateSelectorOverlay(t: TraderInstance, nowMs: number): void {
    const routeMode = resolveSelectorRouteMode(t);
    const symbols = parseAllowedSymbols(t.config.allowedMarkets || '').filter((s) => isSelectorTargetSymbol(s));
    if (symbols.length === 0) return;

    let hasActiveMode = false;
    for (const symbol of symbols) {
        if (resolveSelectorMode(t.config, symbol) !== 'off') hasActiveMode = true;
    }
    if (!hasActiveMode) {
        for (const symbol of symbols) {
            const cell = getOrInitSelectorCell(t, symbol, routeMode);
            cell.checkedAtMs = nowMs;
            cell.sourceMode = routeMode;
            cell.mode = routeMode;
            cell.status = 'normal';
            cell.eligible = true;
            cell.reasonCode = 'off_mode';
            cell.lastAction = 'off';
            t.selectorRuntimeEligibleBySymbol[symbol] = true;
        }
        return;
    }

    const checkEverySec = Math.max(10, Number(t.config.selectorCheckSeconds ?? SELECTOR_CHECK_SECONDS_DEFAULT));
    if (nowMs - (t.selectorLastCheckMsByMode[routeMode] || 0) < checkEverySec * 1000) return;
    t.selectorLastCheckMsByMode[routeMode] = nowMs;

    for (const symbol of symbols) {
        const cell = getOrInitSelectorCell(t, symbol, routeMode);
        const mode = resolveSelectorMode(t.config, symbol);
        const configuredEligible = t.config.selectorEligibleBySymbol?.[symbol];
        const scope = t.config.selectorScopeBySymbol?.[symbol] === 'core'
            ? 'core'
            : t.config.selectorScopeBySymbol?.[symbol] === 'shadow'
                ? 'shadow'
                : 'unknown';
        cell.checkedAtMs = nowMs;
        cell.sourceMode = routeMode;
        cell.mode = routeMode;
        cell.status = 'disabled';
        cell.scope = scope;
        if (mode === 'off') {
            cell.eligible = true;
            cell.reasonCode = 'off_mode';
            cell.lastAction = 'off';
            t.selectorRuntimeEligibleBySymbol[symbol] = true;
            continue;
        }
        if (mode === 'shadow') {
            cell.eligible = true;
            cell.reasonCode = scope === 'core' ? 'selector_shadow_allow' : 'shadow_eligible';
            cell.lastAction = cell.reasonCode;
            cell.status = 'healthy';
            t.selectorRuntimeEligibleBySymbol[symbol] = true;
            continue;
        }
        cell.eligible = typeof configuredEligible === 'boolean' ? configuredEligible : true;
        if (cell.eligible) {
            cell.reasonCode = scope === 'core' ? 'selector_core_allow' : 'selector_ok';
            cell.status = 'healthy';
        } else {
            cell.reasonCode = scope === 'shadow' ? 'selector_core_shadow' : 'selector_blocked';
            cell.status = 'blocked';
        }
        cell.lastAction = cell.reasonCode;
        t.selectorRuntimeEligibleBySymbol[symbol] = cell.eligible;
    }
}

function defaultRegimeGateState(
    profile: 'default' | '70',
    symbol: string,
    mode: RegimeMode,
    routeMode: RegimeRouteMode,
    reasonCode: string,
): RegimeGateSymbolState {
    return {
        symbol,
        profile,
        mode,
        sourceMode: routeMode,
        effectiveMode: mode,
        active: false,
        directionPolicy: 'BOTH',
        policyMode: 'neutral',
        policyConfidence: 0,
        policyReason: reasonCode,
        policyDisagreementFlags: [],
        thresholdDelta: 0,
        regimeId: 'none',
        reasonCode,
        checkedAtMs: Date.now(),
    };
}

function normalizeRegimeDirectionPolicy(raw: unknown): 'BOTH' | 'UP' | 'DOWN' | 'NONE' {
    const v = String(raw || '').trim().toUpperCase();
    if (v === 'UP' || v === 'DOWN' || v === 'NONE') return v;
    return 'BOTH';
}

function normalizeOptionalTradeDirection(raw: unknown): 'UP' | 'DOWN' | 'NONE' {
    const v = String(raw || '').trim().toUpperCase();
    if (v === 'UP' || v === 'DOWN') return v;
    return 'NONE';
}

function normalizeRegimePolicyMode(raw: unknown): 'one_sided' | 'neutral' {
    return String(raw || '').trim().toLowerCase() === 'one_sided' ? 'one_sided' : 'neutral';
}

function maybeUpdateRegimeGate(t: TraderInstance, nowMs: number): void {
    const routeMode = resolveRegimeRouteMode(t);
    const profile = t.config.profile ?? inferProfileFromGroup(t.config.group);
    const symbols = parseAllowedSymbols(t.config.allowedMarkets || '').filter((s) => isRegimeTargetSymbol(s));
    if (symbols.length === 0) return;

    const checkEverySec = Math.max(10, Number(t.config.regimeCheckSeconds ?? REGIME_CHECK_SECONDS_DEFAULT));
    if (nowMs - (t.regimeLastCheckMsByMode[routeMode] || 0) < checkEverySec * 1000) return;
    t.regimeLastCheckMsByMode[routeMode] = nowMs;

    const filePath = path.join(REGIME_RUNTIME_DIR, `regime_gates_v1_${profile}.json`);
    if (!fs.existsSync(filePath)) {
        for (const symbol of symbols) {
            const mode = resolveRegimeMode(t.config, symbol);
            t.regimeRuntimeBySymbol[symbol] = defaultRegimeGateState(profile, symbol, mode, routeMode, 'no_data');
        }
        return;
    }

    try {
        const stat = fs.statSync(filePath);
        t.regimeLastFileMtimeMsByMode[routeMode] = stat.mtimeMs;
    } catch {
        // ignore stat errors
    }

    let symbolsPayload: Record<string, unknown>[] = [];
    try {
        const raw = JSON.parse(fs.readFileSync(filePath, 'utf8')) as Record<string, unknown>;
        const arr = raw.symbols;
        if (Array.isArray(arr)) {
            symbolsPayload = arr.filter((x) => x && typeof x === 'object') as Record<string, unknown>[];
        }
    } catch {
        for (const symbol of symbols) {
            const mode = resolveRegimeMode(t.config, symbol);
            t.regimeRuntimeBySymbol[symbol] = defaultRegimeGateState(profile, symbol, mode, routeMode, 'parse_error');
        }
        return;
    }

    for (const symbol of symbols) {
        const mode = resolveRegimeMode(t.config, symbol);
        if (mode === 'off') {
            t.regimeRuntimeBySymbol[symbol] = defaultRegimeGateState(profile, symbol, mode, routeMode, 'off_mode');
            continue;
        }
        const matched = symbolsPayload.find((row) => {
            const rowSymbol = String(row.symbol || '').trim().toUpperCase();
            if (rowSymbol !== symbol.toUpperCase()) return false;
            const rowSource = String(row.sourceMode || routeMode).toLowerCase();
            if (rowSource !== 'simulation' && rowSource !== 'live') return true;
            return rowSource === routeMode;
        });
        if (!matched) {
            t.regimeRuntimeBySymbol[symbol] = defaultRegimeGateState(profile, symbol, mode, routeMode, 'symbol_missing');
            continue;
        }

        const sourceModeRaw = String(matched.sourceMode || routeMode).toLowerCase();
        const sourceMode: RegimeRouteMode = sourceModeRaw === 'live' ? 'live' : 'simulation';
        const policyMode = normalizeRegimePolicyMode(matched.policyMode);
        const policy = policyMode === 'neutral'
            ? 'BOTH'
            : normalizeRegimeDirectionPolicy(matched.directionPolicy);
        const deltaRaw = Number(matched.thresholdDelta);
        const thresholdDelta = Number.isFinite(deltaRaw)
            ? Math.max(-0.05, Math.min(0.05, deltaRaw))
            : 0;
        const policyConfidenceRaw = Number(matched.policyConfidence);
        const policyConfidence = Number.isFinite(policyConfidenceRaw)
            ? Math.max(0, Math.min(1, policyConfidenceRaw))
            : 0;
        const changePointProbRaw = Number(matched.changePointProb);
        const changePointProb = Number.isFinite(changePointProbRaw)
            ? Math.max(0, Math.min(1, changePointProbRaw))
            : undefined;
        const changePointScoreUpRaw = Number(matched.changePointScoreUp);
        const changePointScoreDownRaw = Number(matched.changePointScoreDown);
        const checkedAtRaw = Date.parse(String(matched.checkedAt || ''));
        const checkedAtMs = Number.isFinite(checkedAtRaw) ? checkedAtRaw : nowMs;
        const rowMode = String(matched.effectiveMode || matched.mode || '').toLowerCase();
        const disagreementFlags = Array.isArray(matched.policyDisagreementFlags)
            ? matched.policyDisagreementFlags.map((x) => String(x || '').trim()).filter(Boolean)
            : [];
        // Config is source of truth for whether regime is enforced.
        // Runtime sidecar mode is only observational and must not downgrade decision mode.
        const effectiveMode: RegimeMode = mode;
        const reasonSuffix = (rowMode && rowMode !== mode) ? `|mode_mismatch:${rowMode}->${mode}` : '';

        t.regimeRuntimeBySymbol[symbol] = {
            symbol,
            profile,
            mode,
            sourceMode,
            effectiveMode,
            active: Boolean(matched.active),
            directionPolicy: policy,
            policyMode,
            policyConfidence,
            policyReason: String(matched.policyReason || matched.reasonCode || 'ok'),
            policyDisagreementFlags: disagreementFlags,
            thresholdDelta,
            regimeId: String(matched.regimeId || 'unknown'),
            reasonCode: `${String(matched.reasonCode || 'ok')}${reasonSuffix}`,
            checkedAtMs,
            changePointProb,
            changePointDirection: normalizeOptionalTradeDirection(matched.changePointDirection),
            changePointScoreUp: Number.isFinite(changePointScoreUpRaw) ? changePointScoreUpRaw : undefined,
            changePointScoreDown: Number.isFinite(changePointScoreDownRaw) ? changePointScoreDownRaw : undefined,
        };
    }
}

interface FilterStats {
    toOrder: PredictionResult[];
    abstainedByMarketSelection: number;
    skippedShouldTrade: number;
    skippedByQuality: number;
    skippedByThresholdDrift: number;
    skippedByUncertainty: number;
    skippedByExpectancyGate: number;
    skippedByMetaLabel: number;
    skippedBySelector: number;
    skippedByRegime: number;
    skippedByShockRisk: number;
    skippedByComboPause: number;
    skippedByDownBreaker: number;
    skippedByUpRisk: number;
    skippedByExtremeMarket: number;
    skippedByDirectionLossCooldown: number;
    skippedByDrawdownAcceleration: number;
}

interface GuardEvidenceStats {
    updatedAtMs: number;
    evaluated: number;
    passed: number;
    abstainedByMarketSelection: number;
    skippedShouldTrade: number;
    skippedByQuality: number;
    skippedByThresholdDrift: number;
    skippedByUncertainty: number;
    skippedByExpectancyGate: number;
    skippedByMetaLabel: number;
    skippedBySelector: number;
    skippedByRegime: number;
    skippedByShockRisk: number;
    skippedByComboPause: number;
    skippedByDownBreaker: number;
    skippedByUpRisk: number;
    skippedByExtremeMarket: number;
    skippedByDirectionLossCooldown: number;
    skippedByDrawdownAcceleration: number;
}

function freshGuardEvidenceStats(): GuardEvidenceStats {
    return {
        updatedAtMs: 0,
        evaluated: 0,
        passed: 0,
        abstainedByMarketSelection: 0,
        skippedShouldTrade: 0,
        skippedByQuality: 0,
        skippedByThresholdDrift: 0,
        skippedByUncertainty: 0,
        skippedByExpectancyGate: 0,
        skippedByMetaLabel: 0,
        skippedBySelector: 0,
        skippedByRegime: 0,
        skippedByShockRisk: 0,
        skippedByComboPause: 0,
        skippedByDownBreaker: 0,
        skippedByUpRisk: 0,
        skippedByExtremeMarket: 0,
        skippedByDirectionLossCooldown: 0,
        skippedByDrawdownAcceleration: 0,
    };
}

function roundRuntimeScale(value: number): number {
    return Math.max(0, Math.min(1, Number(value.toFixed(4))));
}

function accumulateGuardEvidence(t: TraderInstance, parsedCount: number, filtered: FilterStats): void {
    const st = t.guardEvidence;
    st.updatedAtMs = Date.now();
    st.evaluated += Math.max(0, Number(parsedCount) || 0);
    st.passed += Math.max(0, Number(filtered.toOrder.length) || 0);
    st.abstainedByMarketSelection += Math.max(0, Number(filtered.abstainedByMarketSelection) || 0);
    st.skippedShouldTrade += Math.max(0, Number(filtered.skippedShouldTrade) || 0);
    st.skippedByQuality += Math.max(0, Number(filtered.skippedByQuality) || 0);
    st.skippedByThresholdDrift += Math.max(0, Number(filtered.skippedByThresholdDrift) || 0);
    st.skippedByUncertainty += Math.max(0, Number(filtered.skippedByUncertainty) || 0);
    st.skippedByExpectancyGate += Math.max(0, Number(filtered.skippedByExpectancyGate) || 0);
    st.skippedByMetaLabel += Math.max(0, Number(filtered.skippedByMetaLabel) || 0);
    st.skippedBySelector += Math.max(0, Number(filtered.skippedBySelector) || 0);
    st.skippedByRegime += Math.max(0, Number(filtered.skippedByRegime) || 0);
    st.skippedByShockRisk += Math.max(0, Number(filtered.skippedByShockRisk) || 0);
    st.skippedByComboPause += Math.max(0, Number(filtered.skippedByComboPause) || 0);
    st.skippedByDownBreaker += Math.max(0, Number(filtered.skippedByDownBreaker) || 0);
    st.skippedByUpRisk += Math.max(0, Number(filtered.skippedByUpRisk) || 0);
    st.skippedByExtremeMarket += Math.max(0, Number(filtered.skippedByExtremeMarket) || 0);
    st.skippedByDirectionLossCooldown += Math.max(0, Number(filtered.skippedByDirectionLossCooldown) || 0);
    st.skippedByDrawdownAcceleration += Math.max(0, Number(filtered.skippedByDrawdownAcceleration) || 0);
}

function resolveMarketSelectionDecision(
    t: TraderInstance,
    p: PredictionResult,
    symbol: string,
    direction: 'UP' | 'DOWN',
    quality: DataQualityState,
    expectancyCell: ExpectancyGateCellState,
    uncertaintyStatus: 'normal' | 'degraded' | 'blocked',
    uncertaintyDispersion: number | null,
    uncertaintySourceCount: number | null,
    uncertaintyIntervalWidth: number | null,
    regimeState: RegimeGateSymbolState,
    effectiveThreshold: number,
    extremeActive: boolean,
    directionLossState: DirectionLossCooldownState | undefined,
    drawdownState: DrawdownAccelerationState | undefined,
): MarketSelectionDecision {
    const profile = t.config.profile ?? inferProfileFromGroup(t.config.group);
    const confidenceRaw = Number.isFinite(Number(p.calibratedConfidence))
        ? Number(p.calibratedConfidence)
        : Number.isFinite(Number(p.confidence))
            ? Number(p.confidence)
            : null;
    const evidence: MarketSelectionEvidence = {
        qualityMode: quality.mode,
        expectancyStatus: expectancyCell.status,
        expectancySampleCount: Math.max(0, Math.round(Number(expectancyCell.stats.trades) || 0)),
        expectancyPnlPerTrade: Number(expectancyCell.stats.pnlPerTrade || 0),
        uncertaintyStatus,
        uncertaintyDispersion,
        uncertaintyEffectiveSourceCount: uncertaintySourceCount,
        uncertaintyIntervalWidth,
        regimeDirectionPolicy: regimeState.directionPolicy || 'BOTH',
        regimeDisagreementCount: Array.isArray(regimeState.policyDisagreementFlags) ? regimeState.policyDisagreementFlags.length : 0,
        changePointProb: Number.isFinite(Number(regimeState.changePointProb)) ? Number(regimeState.changePointProb) : null,
        confidence: confidenceRaw,
        effectiveThreshold,
        reasons: [],
    };
    if (profile !== 'default') {
        return {
            applicable: false,
            action: 'allow',
            reasonCode: 'profile_not_default',
            marketQualityScore: 1,
            evidence,
        };
    }

    let score = 1.0;
    const reasons = evidence.reasons;

    if (quality.mode !== 'ok' || quality.staleSources.length > 0 || quality.criticalStale.length > 0) {
        score -= 0.18;
        reasons.push('data_not_fresh');
    }

    if (expectancyCell.status === 'blocked' || evidence.expectancySampleCount < 3) {
        score -= 0.24;
        reasons.push('expectancy_insufficient');
    } else if (expectancyCell.status === 'degraded' || evidence.expectancyPnlPerTrade < 0) {
        score -= 0.12;
        reasons.push('expectancy_insufficient');
    }

    if (uncertaintyStatus === 'blocked') {
        score -= 0.22;
        reasons.push('uncertainty_too_high');
    } else if (uncertaintyStatus === 'degraded') {
        score -= 0.12;
        reasons.push('uncertainty_too_high');
    }

    const regimeConflict = (
        (regimeState.directionPolicy || 'BOTH') !== 'BOTH'
        || (Array.isArray(regimeState.policyDisagreementFlags) && regimeState.policyDisagreementFlags.length > 0)
        || (Number.isFinite(Number(regimeState.changePointProb)) && Number(regimeState.changePointProb) >= 0.65)
    );
    if (regimeConflict) {
        score -= 0.14;
        reasons.push('regime_conflict');
    }

    if (
        confidenceRaw !== null
        && Number.isFinite(confidenceRaw)
        && confidenceRaw < Math.min(0.999, effectiveThreshold + 0.02)
    ) {
        score -= 0.12;
        reasons.push('market_quality_low');
    }

    const sameDirectionLossRisk = (
        !extremeActive
        && Boolean(directionLossState?.evidenceFresh)
        && Math.max(0, Number(directionLossState?.streakLosses) || 0) >= 2
        && Math.max(0, Number(directionLossState?.recentHighConfidenceLosses) || 0) >= 2
        && Number(directionLossState?.recentPnl || 0) < 0
    );
    if (sameDirectionLossRisk) {
        score -= 0.18;
        reasons.push('same_direction_loss_risk');
    }

    const drawdownWatchRisk = (
        !extremeActive
        && !Boolean(drawdownState?.blocked)
        && Boolean(drawdownState?.evidenceFresh)
        && Math.max(0, Number(drawdownState?.recentLosses) || 0) >= 2
        && Number(drawdownState?.recentPnl || 0) < 0
    );
    if (drawdownWatchRisk) {
        score -= 0.08;
        reasons.push('drawdown_watch_risk');
    }

    const uniqueReasons = [...new Set(reasons)];
    evidence.reasons = uniqueReasons;
    const marketQualityScore = roundRuntimeScale(Math.max(0, Math.min(1, score)));
    const hasExtraRiskBeyondDrawdownWatch = uniqueReasons.some((reason) =>
        reason !== 'drawdown_watch_risk'
        && reason !== 'market_quality_low'
        && reason !== 'expectancy_insufficient'
        && reason !== 'uncertainty_too_high'
        && reason !== 'same_direction_loss_risk',
    );
    const abstain = (
        uniqueReasons.includes('data_not_fresh')
        || uniqueReasons.includes('same_direction_loss_risk')
        || (uniqueReasons.includes('drawdown_watch_risk') && hasExtraRiskBeyondDrawdownWatch)
    );
    const reasonPriority = [
        'same_direction_loss_risk',
        'drawdown_watch_risk',
        'data_not_fresh',
        'uncertainty_too_high',
        'regime_conflict',
        'expectancy_insufficient',
        'market_quality_low',
    ];
    const selectedReason = abstain
        ? (reasonPriority.find((reason) => uniqueReasons.includes(reason)) || 'market_quality_low')
        : 'market_quality_ok';

    return {
        applicable: true,
        action: abstain ? 'abstain' : 'allow',
        reasonCode: selectedReason,
        marketQualityScore,
        evidence,
    };
}

type RuntimeGuardEvaluationContext = {
    targetTs?: number;
    phase?: number;
    fileTimestamp?: string;
};

function runtimeGuardBlockedOpportunityLogPath(t: TraderInstance): string {
    const logsDir = path.isAbsolute(t.config.logsDir)
        ? t.config.logsDir
        : path.join(process.cwd(), t.config.logsDir);
    return path.join(logsDir, 'direction_loss_cooldown_blocked_opportunities.jsonl');
}

function appendDirectionLossBlockedOpportunity(
    t: TraderInstance,
    p: PredictionResult,
    symbol: string,
    direction: 'UP' | 'DOWN',
    effectiveThreshold: number,
    gatesPassed: string[],
    finalRuntimeScale: number,
    directionLossState: DirectionLossCooldownState,
    ctx: RuntimeGuardEvaluationContext,
): void {
    try {
        const key = [
            t.config.profile ?? inferProfileFromGroup(t.config.group),
            t.config.name,
            symbol,
            direction,
            ctx.targetTs ?? 'no_target_ts',
            ctx.phase ?? 'no_phase',
            ctx.fileTimestamp ?? p.timestamp ?? 'no_file_ts',
        ].join('::');
        if (t.blockedOpportunityLogKeys.has(key)) return;
        t.blockedOpportunityLogKeys.add(key);
        const logPath = runtimeGuardBlockedOpportunityLogPath(t);
        const dir = path.dirname(logPath);
        if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
        const payload = {
            loggedAt: new Date().toISOString(),
            key,
            opportunityType: 'direction_loss_cooldown_blocked',
            counterfactualUse: 'no_cooldown_replay_candidate',
            profile: t.config.profile ?? inferProfileFromGroup(t.config.group),
            group: t.config.group,
            traderName: t.config.name,
            symbol,
            direction,
            targetTs: ctx.targetTs ?? null,
            phase: ctx.phase ?? null,
            predictionFileTimestamp: ctx.fileTimestamp ?? null,
            predictionTimestamp: p.timestamp ?? null,
            confidence: Number.isFinite(Number(p.confidence)) ? Number(p.confidence) : null,
            rawConfidence: Number.isFinite(Number(p.rawConfidence)) ? Number(p.rawConfidence) : null,
            calibratedConfidence: Number.isFinite(Number(p.calibratedConfidence)) ? Number(p.calibratedConfidence) : null,
            effectiveThreshold,
            finalRuntimeScale,
            gatesPassedBeforeCooldown: [...gatesPassed],
            directionLossState: {
                blocked: directionLossState.blocked,
                evidenceFresh: directionLossState.evidenceFresh,
                streakLosses: directionLossState.streakLosses,
                recentTrades: directionLossState.recentTrades,
                recentPnl: directionLossState.recentPnl,
                untilTs: directionLossState.untilTs,
                reasonCode: directionLossState.reasonCode,
                checkedAt: directionLossState.checkedAtMs > 0 ? new Date(directionLossState.checkedAtMs).toISOString() : null,
            },
            noteChinese: '这条记录只说明若去掉方向冷却，此机会已经通过冷却前的运行风控；是否真实成交和最终输赢仍需后续按盘口/结算解析。',
        };
        fs.appendFileSync(logPath, JSON.stringify(payload) + '\n', 'utf8');
    } catch {
        // 机会账本失败不能影响真钱主循环。
    }
}

function applyRuntimeGuards(
    parsed: PredictionResult[],
    t: TraderInstance,
    quality: DataQualityState,
    nowSec: number,
    ctx: RuntimeGuardEvaluationContext = {},
): FilterStats {
    const out: PredictionResult[] = [];
    let abstainedByMarketSelection = 0;
    let skippedShouldTrade = 0;
    let skippedByQuality = 0;
    let skippedByThresholdDrift = 0;
    let skippedByUncertainty = 0;
    let skippedByExpectancyGate = 0;
    let skippedByMetaLabel = 0;
    let skippedBySelector = 0;
    let skippedByRegime = 0;
    let skippedByShockRisk = 0;
    let skippedByComboPause = 0;
    let skippedByDownBreaker = 0;
    let skippedByUpRisk = 0;
    let skippedByExtremeMarket = 0;
    let skippedByDirectionLossCooldown = 0;
    let skippedByDrawdownAcceleration = 0;

    for (const p of parsed) {
        if (p.direction == null) continue;
        const symbol = normalizeSymbol(p.symbol);
        const direction = normalizeDirection(p.direction);
        if (!direction) continue;
        p.rawConfidence = Number.isFinite(Number(p.rawConfidence)) ? Number(p.rawConfidence) : p.confidence;
        const calibrationMode = resolveCalibrationMode(t.config, symbol, direction);
        const calibrationCell = getOrInitCalibrationCell(t, symbol, direction, resolveCalibrationRouteMode(t));
        if (calibrationCell.active) {
            const calibrated = applyCalibrationMethodToConfidence(Number(p.rawConfidence ?? p.confidence), t, symbol, direction);
            calibrationCell.lastConfidence = Number(p.rawConfidence ?? p.confidence);
            calibrationCell.calibratedConfidence = calibrated;
            p.calibratedConfidence = calibrationMode === 'enforce' ? calibrated : undefined;
            p.calibrationMeta = {
                ...(p.calibrationMeta || {}),
                mode: calibrationMode,
                method: calibrationCell.method,
                sampleCount: calibrationCell.stats.sampleCount,
                lookbackHours: calibrationCell.stats.lookbackHours,
                brier: calibrationCell.stats.brier,
                ece: calibrationCell.stats.ece,
                calibratedConfidence: calibrated,
                rawConfidence: Number(p.rawConfidence ?? p.confidence),
            };
        } else {
            p.calibratedConfidence = undefined;
            p.calibrationMeta = {
                ...(p.calibrationMeta || {}),
                mode: calibrationMode,
                method: calibrationCell.method,
                sampleCount: calibrationCell.stats.sampleCount,
                lookbackHours: calibrationCell.stats.lookbackHours,
                reasonCode: calibrationCell.reasonCode,
                rawConfidence: Number(p.rawConfidence ?? p.confidence),
            };
        }
        const expectancyMode = resolveExpectancyGateMode(t.config, symbol, direction);
        const expectancyCell = getOrInitExpectancyGateCell(t, symbol, direction, resolveExpectancyGateRouteMode(t));
        p.expectancyMeta = {
            ...(p.expectancyMeta || {}),
            mode: expectancyMode,
            status: expectancyCell.status,
            sampleCount: expectancyCell.stats.trades,
            lookbackHours: expectancyCell.stats.lookbackHours,
            wrPost: expectancyCell.stats.wrPost,
            pnlPerTrade: expectancyCell.stats.pnlPerTrade,
            degradedBetScale: expectancyCell.degradedBetScale,
            degradedExtraDelta: expectancyCell.degradedExtraDelta,
            reasonCode: expectancyCell.reasonCode,
        };
        const driftMode = resolveThresholdDriftMode(t.config, symbol, direction);
        const driftCell = getOrInitThresholdDriftCell(t, symbol, direction, resolveThresholdDriftRouteMode(t));
        const driftDelta = Math.max(0, Number(t.thresholdDriftRuntimeExtraDeltaBySymbolDir[symbolDirectionRuntimeKey(symbol, direction)] ?? 0));
        const regimeMode = resolveRegimeMode(t.config, symbol);
        const regimeState = t.regimeRuntimeBySymbol[symbol]
            || defaultRegimeGateState(t.config.profile ?? inferProfileFromGroup(t.config.group), symbol, regimeMode, resolveRegimeRouteMode(t), 'init');
        const regimeDelta = regimeMode === 'enforce'
            ? Math.max(-0.05, Math.min(0.05, Number(regimeState.thresholdDelta || 0)))
            : 0;
        p.thresholdDriftMeta = {
            ...(p.thresholdDriftMeta || {}),
            mode: driftMode,
            status: driftCell.status,
            sampleCount: driftCell.stats.trades,
            lookbackHours: driftCell.stats.lookbackHours,
            wrPost: driftCell.stats.wrPost,
            pnlPerTrade: driftCell.stats.pnlPerTrade,
            thresholdDelta: driftDelta,
            reasonCode: driftCell.reasonCode,
        };
        p.regimeMeta = {
            ...(p.regimeMeta || {}),
            mode: regimeMode,
            active: regimeState.active,
            directionPolicy: regimeState.directionPolicy,
            policyMode: regimeState.policyMode,
            policyConfidence: regimeState.policyConfidence,
            policyReason: regimeState.policyReason,
            policyDisagreementFlags: regimeState.policyDisagreementFlags,
            thresholdDelta: regimeDelta,
            regimeId: regimeState.regimeId,
            reasonCode: regimeState.reasonCode,
            sourceMode: regimeState.sourceMode,
            changePointProb: regimeState.changePointProb,
            changePointDirection: regimeState.changePointDirection,
            changePointScoreUp: regimeState.changePointScoreUp,
            changePointScoreDown: regimeState.changePointScoreDown,
        };
        const metaMode = resolveMetaLabelMode(t.config, symbol, direction);
        const metaTakeTrade = p.metaLabelMeta?.takeTrade !== false;
        const metaReasonCode = p.metaLabelMeta?.reasonCode || (metaTakeTrade ? 'allow' : 'meta_blocked');
        const metaConfidenceRaw = Number(p.metaLabelMeta?.confidence);
        const metaCell = getOrInitMetaLabelCell(t, symbol, direction);
        metaCell.mode = metaMode;
        metaCell.checkedAtMs = Date.now();
        metaCell.lastAction = metaTakeTrade ? 'allow' : (metaMode === 'enforce' ? 'blocked' : 'shadow_block_candidate');
        metaCell.reasonCode = metaReasonCode;
        metaCell.takeTrade = metaTakeTrade;
        metaCell.metaConfidence = Number.isFinite(metaConfidenceRaw) ? metaConfidenceRaw : null;
        metaCell.evaluated += 1;
        if (metaTakeTrade) {
            metaCell.passed += 1;
        } else {
            metaCell.blocked += 1;
        }
        p.metaLabelMeta = {
            ...(p.metaLabelMeta || {}),
            mode: metaMode,
            takeTrade: metaTakeTrade,
            reasonCode: metaReasonCode,
        };
        const selectorMode = resolveSelectorMode(t.config, symbol);
        const selectorApplicable = isSelectorTargetSymbol(symbol);
        const selectorCell = selectorApplicable
            ? getOrInitSelectorCell(t, symbol, resolveSelectorRouteMode(t))
            : selectorNotApplicableState(t, symbol, resolveSelectorRouteMode(t));
        const selectorEligible = selectorApplicable
            ? Boolean(t.selectorRuntimeEligibleBySymbol[symbol] ?? true)
            : true;
        p.selectorMeta = {
            ...(p.selectorMeta || {}),
            mode: selectorApplicable ? selectorMode : 'off',
            eligible: selectorEligible,
            status: selectorCell.status,
            reasonCode: selectorCell.reasonCode,
        };
        const uncertaintyMode = resolveUncertaintyGateMode(t.config, symbol, direction);
        const uncertaintyDispersionRaw = Number(p.ensembleMeta?.dispersionScore);
        const uncertaintyDispersion = Number.isFinite(uncertaintyDispersionRaw)
            ? Math.max(0, Math.min(1, uncertaintyDispersionRaw))
            : null;
        const uncertaintySourceCountRaw = Number(p.ensembleMeta?.effectiveSourceCount);
        const uncertaintySourceCount = Number.isFinite(uncertaintySourceCountRaw)
            ? Math.max(0, Math.round(uncertaintySourceCountRaw))
            : null;
        const uncertaintyChangePointRaw = Number(p.regimeMeta?.changePointProb ?? regimeState.changePointProb);
        const uncertaintyChangePoint = Number.isFinite(uncertaintyChangePointRaw)
            ? Math.max(0, Math.min(1, uncertaintyChangePointRaw))
            : null;
        const uncertaintyManualAction = getUncertaintyGateManualAction(t, symbol, direction);
        const uncertaintyDirectionPolicy = regimeState.directionPolicy || 'BOTH';
        const uncertaintyTrendAligned = uncertaintyDirectionPolicy === 'BOTH'
            || (uncertaintyDirectionPolicy === 'UP' && direction === 'UP')
            || (uncertaintyDirectionPolicy === 'DOWN' && direction === 'DOWN');
        const uncertaintyHighVolRegime = ['extreme_up', 'extreme_down', 'rally_up', 'rally_down', 'high_vol_chop'].includes(String(regimeState.regimeId || ''));
        const uncertaintyDispersionMaxBase = Math.max(0.02, Math.min(
            0.5,
            getUncertaintyGateNumber(t, symbol, direction, 'uncertaintyDispersionMax', UNCERTAINTY_GATE_DISPERSION_MAX_DEFAULT),
        ));
        const uncertaintyDispersionMaxHighVol = Math.max(0.02, Math.min(
            uncertaintyDispersionMaxBase,
            getUncertaintyGateNumber(t, symbol, direction, 'uncertaintyHighVolDispersionMax', UNCERTAINTY_GATE_HIGH_VOL_DISPERSION_MAX_DEFAULT),
        ));
        const uncertaintyDispersionMax = uncertaintyHighVolRegime ? uncertaintyDispersionMaxHighVol : uncertaintyDispersionMaxBase;
        const uncertaintyIntervalWidthMaxBase = Math.max(0.05, Math.min(
            0.8,
            getUncertaintyGateNumber(t, symbol, direction, 'uncertaintyIntervalWidthMax', UNCERTAINTY_GATE_INTERVAL_WIDTH_MAX_DEFAULT),
        ));
        const uncertaintyIntervalWidthMaxHighVol = Math.max(0.05, Math.min(
            uncertaintyIntervalWidthMaxBase,
            getUncertaintyGateNumber(t, symbol, direction, 'uncertaintyHighVolIntervalWidthMax', UNCERTAINTY_GATE_HIGH_VOL_INTERVAL_WIDTH_MAX_DEFAULT),
        ));
        const uncertaintyIntervalWidthMax = uncertaintyHighVolRegime ? uncertaintyIntervalWidthMaxHighVol : uncertaintyIntervalWidthMaxBase;
        const uncertaintyIntervalZScore = Math.max(0.2, Math.min(
            4.0,
            getUncertaintyGateNumber(t, symbol, direction, 'uncertaintyIntervalZScore', UNCERTAINTY_GATE_INTERVAL_Z_SCORE_DEFAULT),
        ));
        const uncertaintyIntervalDispersionBlend = Math.max(0, Math.min(
            3.0,
            getUncertaintyGateNumber(
                t,
                symbol,
                direction,
                'uncertaintyIntervalDispersionBlend',
                UNCERTAINTY_GATE_INTERVAL_DISPERSION_BLEND_DEFAULT,
            ),
        ));
        const uncertaintySourceCountMin = Math.max(
            1,
            Math.round(getUncertaintyGateNumber(t, symbol, direction, 'uncertaintyEffectiveSourceCountMin', UNCERTAINTY_GATE_EFFECTIVE_SOURCE_COUNT_MIN_DEFAULT)),
        );
        const uncertaintyChangePointMax = Math.max(0, Math.min(
            1,
            getUncertaintyGateNumber(
                t,
                symbol,
                direction,
                uncertaintyTrendAligned ? 'uncertaintyChangePointProbMax' : 'uncertaintyCountertrendChangePointProbMax',
                uncertaintyTrendAligned ? UNCERTAINTY_GATE_CHANGE_POINT_PROB_MAX_DEFAULT : UNCERTAINTY_GATE_COUNTERTREND_CHANGE_POINT_PROB_MAX_DEFAULT,
            ),
        ));
        const uncertaintyDegradedExtraDelta = Math.max(
            0,
            getUncertaintyGateNumber(t, symbol, direction, 'uncertaintyDegradedExtraDelta', UNCERTAINTY_GATE_DEGRADED_EXTRA_DELTA_DEFAULT),
        );
        const uncertaintyConfidenceRaw = Number.isFinite(Number(p.calibratedConfidence))
            ? Number(p.calibratedConfidence)
            : Number(p.confidence);
        const uncertaintyConfidence = Number.isFinite(uncertaintyConfidenceRaw)
            ? Math.max(0, Math.min(1, uncertaintyConfidenceRaw))
            : null;
        const uncertaintyIntervalBase = (
            uncertaintyConfidence !== null
            && uncertaintySourceCount !== null
            && uncertaintySourceCount > 0
        )
            ? (2 * uncertaintyIntervalZScore * Math.sqrt(
                Math.max(0.000001, uncertaintyConfidence * (1 - uncertaintyConfidence)) / uncertaintySourceCount,
            ))
            : null;
        const uncertaintyIntervalWidth = (
            uncertaintyIntervalBase !== null
            ? Math.max(
                uncertaintyIntervalBase,
                uncertaintyIntervalBase + ((uncertaintyDispersion ?? 0) * uncertaintyIntervalDispersionBlend),
            )
            : (
                uncertaintyDispersion !== null
                ? Math.max(0, uncertaintyDispersion * (1 + uncertaintyIntervalDispersionBlend))
                : null
            )
        );
        let uncertaintyBlocked = false;
        let uncertaintyStatus: 'normal' | 'degraded' | 'blocked' = 'normal';
        let uncertaintyReasonCode = 'ok';
        let uncertaintyDelta = 0;
        if (p.ensembleMeta?.consensusBlocked) {
            uncertaintyBlocked = true;
            uncertaintyStatus = 'blocked';
            uncertaintyReasonCode = 'consensus_blocked';
        } else if (uncertaintySourceCount !== null && uncertaintySourceCount < uncertaintySourceCountMin) {
            uncertaintyBlocked = true;
            uncertaintyStatus = 'blocked';
            uncertaintyReasonCode = 'source_count_low';
        } else if (!uncertaintyTrendAligned && uncertaintyChangePoint !== null && uncertaintyChangePoint > uncertaintyChangePointMax) {
            uncertaintyBlocked = true;
            uncertaintyStatus = 'blocked';
            uncertaintyReasonCode = 'countertrend_change_point_high';
        } else if (uncertaintyIntervalWidth !== null && uncertaintyIntervalWidth > (uncertaintyIntervalWidthMax + 0.05)) {
            uncertaintyBlocked = true;
            uncertaintyStatus = 'blocked';
            uncertaintyReasonCode = 'interval_too_wide';
        } else if (uncertaintyDispersion !== null && uncertaintyDispersion > (uncertaintyDispersionMax + 0.05)) {
            uncertaintyBlocked = true;
            uncertaintyStatus = 'blocked';
            uncertaintyReasonCode = 'dispersion_too_high';
        } else if (
            (uncertaintyIntervalWidth !== null && uncertaintyIntervalWidth > uncertaintyIntervalWidthMax)
            || (
            (uncertaintyDispersion !== null && uncertaintyDispersion > uncertaintyDispersionMax)
            || (uncertaintyChangePoint !== null && uncertaintyChangePoint > uncertaintyChangePointMax)
            )
        ) {
            uncertaintyStatus = uncertaintyManualAction === 'blocked' || (!uncertaintyTrendAligned && uncertaintyHighVolRegime)
                ? 'blocked'
                : 'degraded';
            uncertaintyBlocked = uncertaintyStatus === 'blocked';
            uncertaintyReasonCode = uncertaintyIntervalWidth !== null && uncertaintyIntervalWidth > uncertaintyIntervalWidthMax
                ? 'interval_wide'
                : (
                    uncertaintyDispersion !== null && uncertaintyDispersion > uncertaintyDispersionMax
                        ? 'dispersion_high'
                        : 'change_point_high'
                );
            uncertaintyDelta = uncertaintyBlocked ? 0 : uncertaintyDegradedExtraDelta;
        }
        const extremeState = t.mainlineExtremeRuntimeBySymbol[symbol];
        const extremeExtraDelta = Math.max(0, Number(t.mainlineExtremeRuntimeExtraDeltaBySymbolDir[symbolDirectionRuntimeKey(symbol, direction)] ?? 0));
        const directionLossBlocked = Boolean(t.directionLossCooldownBlockedBySymbolDir[symbolDirectionRuntimeKey(symbol, direction)]);
        const drawdownAccelerationBlocked = Boolean(t.drawdownAccelerationBlockedBySymbol[symbol]);
        const baseThreshold = getDirectionThreshold(p, t.env);
        const directionRiskExtraDelta = p.direction === 'DOWN'
            ? Math.max(0, Number(t.downRiskRuntimeExtraDeltaBySymbol[symbol] ?? 0))
            : p.direction === 'UP'
                ? Math.max(0, Number(t.upRiskRuntimeExtraDeltaBySymbol[symbol] ?? 0))
                : 0;
        const thresholdDeltaStack: RuntimeThresholdDeltaEntry[] = [
            {
                layer: 'data_quality_penalty',
                delta: Math.max(0, Number(quality.confidencePenalty) || 0),
                active: Math.max(0, Number(quality.confidencePenalty) || 0) > 0,
                reasonCode: quality.reason,
            },
            {
                layer: 'threshold_drift',
                delta: Math.max(0, Number(driftDelta) || 0),
                active: Math.max(0, Number(driftDelta) || 0) > 0,
                reasonCode: driftCell.reasonCode,
            },
            {
                layer: 'regime',
                delta: Math.max(0, Number(regimeDelta) || 0),
                active: Math.max(0, Number(regimeDelta) || 0) > 0,
                reasonCode: regimeState.reasonCode,
            },
            {
                layer: 'uncertainty',
                delta: Math.max(0, Number(uncertaintyDelta) || 0),
                active: Math.max(0, Number(uncertaintyDelta) || 0) > 0,
                reasonCode: uncertaintyReasonCode,
            },
            {
                layer: 'extreme_market',
                delta: Math.max(0, Number(extremeExtraDelta) || 0),
                active: Math.max(0, Number(extremeExtraDelta) || 0) > 0,
                reasonCode: extremeState?.reasonCode || 'ok',
            },
            {
                layer: p.direction === 'DOWN' ? 'down_risk' : 'up_risk',
                delta: Math.max(0, Number(directionRiskExtraDelta) || 0),
                active: Math.max(0, Number(directionRiskExtraDelta) || 0) > 0,
                reasonCode: p.direction === 'DOWN' ? 'down_risk_extra_delta' : 'up_risk_extra_delta',
            },
        ];
        const scaleStack: RuntimeScaleEntry[] = [
            {
                layer: 'data_quality',
                scale: roundRuntimeScale(Math.min(1, Math.max(0, Number(quality.betScale) || 1))),
                active: roundRuntimeScale(Math.min(1, Math.max(0, Number(quality.betScale) || 1))) < 1,
                reasonCode: quality.reason,
            },
            {
                layer: 'mainline_extreme',
                scale: roundRuntimeScale(Math.min(1, Math.max(0, Number(t.mainlineExtremeRuntimeScaleBySymbol[symbol] ?? 1)))),
                active: roundRuntimeScale(Math.min(1, Math.max(0, Number(t.mainlineExtremeRuntimeScaleBySymbol[symbol] ?? 1)))) < 1,
                reasonCode: extremeState?.reasonCode || 'ok',
            },
            {
                layer: p.direction === 'DOWN' ? 'down_risk' : 'up_risk',
                scale: roundRuntimeScale(
                    Math.min(
                        1,
                        Math.max(
                            0,
                            Number(
                                p.direction === 'DOWN'
                                    ? (t.downRiskRuntimeScaleBySymbol[symbol] ?? 1)
                                    : (t.upRiskRuntimeScaleBySymbol[symbol] ?? 1),
                            ) || 1,
                        ),
                    ),
                ),
                active: roundRuntimeScale(
                    Math.min(
                        1,
                        Math.max(
                            0,
                            Number(
                                p.direction === 'DOWN'
                                    ? (t.downRiskRuntimeScaleBySymbol[symbol] ?? 1)
                                    : (t.upRiskRuntimeScaleBySymbol[symbol] ?? 1),
                            ) || 1,
                        ),
                    ),
                ) < 1,
                reasonCode: p.direction === 'DOWN' ? 'down_risk_scale' : 'up_risk_scale',
            },
            {
                layer: 'expectancy_gate',
                scale: roundRuntimeScale(Math.min(1, Math.max(0, Number(t.expectancyGateRuntimeScaleBySymbolDir[symbolDirectionRuntimeKey(symbol, direction)] ?? 1)))),
                active: roundRuntimeScale(Math.min(1, Math.max(0, Number(t.expectancyGateRuntimeScaleBySymbolDir[symbolDirectionRuntimeKey(symbol, direction)] ?? 1)))) < 1,
                reasonCode: 'expectancy_scale',
            },
            {
                layer: 'direction_loss_soft_scale',
                scale: roundRuntimeScale(Math.min(1, Math.max(0, Number(t.directionLossCooldownSoftScaleBySymbolDir[symbolDirectionRuntimeKey(symbol, direction)] ?? 1)))),
                active: roundRuntimeScale(Math.min(1, Math.max(0, Number(t.directionLossCooldownSoftScaleBySymbolDir[symbolDirectionRuntimeKey(symbol, direction)] ?? 1)))) < 1,
                reasonCode: String(t.directionLossCooldownBySymbolDir[symbolDirectionRuntimeKey(symbol, direction)]?.softReasonCode || 'direction_loss_soft_scale'),
            },
            {
                layer: 'drawdown_acceleration_watch',
                scale: roundRuntimeScale(Math.min(1, Math.max(0, Number(t.drawdownAccelerationSoftScaleBySymbol[symbol] ?? 1)))),
                active: roundRuntimeScale(Math.min(1, Math.max(0, Number(t.drawdownAccelerationSoftScaleBySymbol[symbol] ?? 1)))) < 1,
                reasonCode: String(t.drawdownAccelerationBySymbol[symbol]?.softReasonCode || 'ok'),
            },
        ];
        const finalRuntimeScale = roundRuntimeScale(
            scaleStack.reduce((acc, item) => acc * Math.min(1, Math.max(0, Number(item.scale) || 1)), 1),
        );
        const effectiveThreshold = roundProb(Math.min(
            0.999,
            Math.max(0, baseThreshold + quality.confidencePenalty + driftDelta + regimeDelta + directionRiskExtraDelta + extremeExtraDelta),
        ));
        const directionLossState = t.directionLossCooldownBySymbolDir[symbolDirectionRuntimeKey(symbol, direction)];
        const drawdownState = t.drawdownAccelerationBySymbol[symbol];
        const selectionDecision = resolveMarketSelectionDecision(
            t,
            p,
            symbol,
            direction,
            quality,
            expectancyCell,
            uncertaintyStatus,
            uncertaintyDispersion,
            uncertaintySourceCount,
            uncertaintyIntervalWidth,
            regimeState,
            effectiveThreshold,
            Boolean(extremeState?.active),
            directionLossState,
            drawdownState,
        );
        p.selectionMeta = {
            ...(p.selectionMeta || {}),
            applicable: selectionDecision.applicable,
            action: selectionDecision.action,
            reasonCode: selectionDecision.reasonCode,
            marketQualityScore: selectionDecision.marketQualityScore,
            evidence: selectionDecision.evidence,
        };
        const decisionKey = symbolDirectionRuntimeKey(symbol, direction);
        const gatesPassed: string[] = [];
        let slot02Aux1hDecisionDebug: ReturnType<typeof evaluateSlot02Aux1hAtDecisionLayer> | null = null;
        const writeDecision = (
            decisionStatus: 'passed' | 'blocked' | 'abstained',
            primaryBlockReason: string | null,
        ): void => {
            const executorCurrentCapital = (() => {
                try {
                    const capital = (t.executor as any)?.getCoinCapital?.(symbol);
                    return Number.isFinite(Number(capital)) ? Number(capital) : null;
                } catch {
                    return null;
                }
            })();
            const runtimeCurrentCapital = readRuntimeCurrentCapitalForDecision(
                t.config.logsDir,
                t.env.TRADING_MODE,
                symbol,
                executorCurrentCapital,
            );
            const cooldownScope = (() => {
                try {
                    const scope = (t.executor as any)?.getCooldownScope?.();
                    if (scope === 'symbol' || scope === 'symbol_direction') {
                        return scope;
                    }
                } catch {
                    // fall through to config fallback
                }
                const cfgScope = String((t.config as any)?.cooldownScope || '').trim().toLowerCase();
                if (cfgScope === 'symbol' || cfgScope === 'symbol_direction') {
                    return cfgScope;
                }
                if (Number((t.config as any)?.cooldownBars || 0) > 0) {
                    return 'symbol_direction';
                }
                return null;
            })();
            const cooldownKey = (() => {
                if (!cooldownScope) {
                    return null;
                }
                try {
                    const key = (t.executor as any)?.buildCooldownKey?.(symbol, direction);
                    return typeof key === 'string' && key.trim() ? key : null;
                } catch {
                    return cooldownScope === 'symbol'
                        ? symbol
                        : `${symbol}_${direction}`;
                }
            })();
            const cooldownScopeLabel = cooldownScope === 'symbol'
                ? 'total_chain_by_symbol'
                : cooldownScope === 'symbol_direction'
                    ? 'symbol_direction'
                    : null;
            const cooldownDecisionDebug = (() => {
                try {
                    const targetPeriodStartTs = parsePeriodStartFromSlug(
                        (p as PredictionResult & { marketSlug?: string }).marketSlug || null,
                    ) ?? undefined;
                    return (t.executor as any)?.getCooldownDecisionDebugSnapshot?.(
                        symbol,
                        direction,
                        targetPeriodStartTs,
                    ) ?? null;
                } catch {
                    return null;
                }
            })();
            const selectionRouteAction = selectionDecision.applicable
                ? selectionDecision.action
                : 'skip';
            const selectionRouteProfileScope = selectionDecision.applicable
                ? 'default_only'
                : 'skipped_non_default';
            const thresholdBlockReason = (
                primaryBlockReason !== null && primaryBlockReason.startsWith('threshold_')
            )
                ? primaryBlockReason
                : null;
            t.executionDecisionBySymbolDir[decisionKey] = {
                key: `${t.config.profile ?? inferProfileFromGroup(t.config.group)}::${t.config.name}::${symbol}::${direction}`,
                profile: t.config.profile ?? inferProfileFromGroup(t.config.group),
                traderName: t.config.name,
                symbol,
                direction,
                decisionStatus,
                primaryBlockReason,
                runtimePrimaryBlockReason: primaryBlockReason,
                thresholdBlockReason,
                selectorMode: selectorApplicable ? selectorMode : 'off',
                selectorEligible,
                selectorScope: selectorCell.scope,
                thresholdDeltaStack,
                scaleStack,
                finalRuntimeScale,
                gatesPassed: [...gatesPassed],
                effectiveThreshold,
                confidence: Number.isFinite(Number(p.confidence)) ? Number(p.confidence) : null,
                rawConfidence: Number.isFinite(Number(p.rawConfidence)) ? Number(p.rawConfidence) : null,
                calibratedConfidence: Number.isFinite(Number(p.calibratedConfidence)) ? Number(p.calibratedConfidence) : null,
                shouldTrade: typeof p.shouldTrade === 'boolean' ? p.shouldTrade : null,
                selectionAction: selectionDecision.action,
                selectionApplicable: selectionDecision.applicable,
                selectionReasonCode: selectionDecision.reasonCode,
                selectionRouteApplied: selectionDecision.applicable,
                selectionRouteProfileScope,
                selectionRouteAction,
                selectionRouteReasonCode: selectionDecision.reasonCode,
                marketQualityScore: selectionDecision.marketQualityScore,
                selectionEvidence: selectionDecision.evidence,
                uncertaintyGateStatus: uncertaintyStatus,
                uncertaintyGateReasonCode: uncertaintyReasonCode,
                finalDecision: decisionStatus,
                noTradeReason: decisionStatus === 'passed' ? null : primaryBlockReason,
                currentCapital: runtimeCurrentCapital,
                cooldownScope,
                cooldownKey,
                cooldownScopeLabel,
                cooldownRemainingBars: Number.isFinite(Number(cooldownDecisionDebug?.cooldownRemainingBars))
                    ? Number(cooldownDecisionDebug.cooldownRemainingBars)
                    : null,
                formalConsecutiveLosses: Number.isFinite(Number(cooldownDecisionDebug?.formalConsecutiveLosses))
                    ? Number(cooldownDecisionDebug.formalConsecutiveLosses)
                    : null,
                provisionalConsecutiveLosses: Number.isFinite(Number(cooldownDecisionDebug?.provisionalConsecutiveLosses))
                    ? Number(cooldownDecisionDebug.provisionalConsecutiveLosses)
                    : null,
                effectiveConsecutiveLosses: Number.isFinite(Number(cooldownDecisionDebug?.effectiveConsecutiveLosses))
                    ? Number(cooldownDecisionDebug.effectiveConsecutiveLosses)
                    : null,
                cooldownPendingResolutionBlocked: Boolean(cooldownDecisionDebug?.pendingResolutionBlocked),
                cooldownPendingResolutionMarketSlug: cooldownDecisionDebug?.pendingResolutionMarketSlug || null,
                cooldownPendingResolutionConsecutiveBeforeUnknown: Number.isFinite(Number(cooldownDecisionDebug?.pendingResolutionConsecutiveBeforeUnknown))
                    ? Number(cooldownDecisionDebug.pendingResolutionConsecutiveBeforeUnknown)
                    : null,
                cooldownPersistedWindowBlocked: Boolean(cooldownDecisionDebug?.persistedWindowBlocked),
                cooldownTriggerMarketSlug: cooldownDecisionDebug?.persistedTriggerMarketSlug || null,
                cooldownTriggerStage: cooldownDecisionDebug?.persistedTriggerStage || null,
                cooldownPersistedRemainingBarsBefore: Number.isFinite(Number(cooldownDecisionDebug?.persistedRemainingBarsBefore))
                    ? Number(cooldownDecisionDebug.persistedRemainingBarsBefore)
                    : null,
                cooldownPersistedRemainingBarsAfter: Number.isFinite(Number(cooldownDecisionDebug?.persistedRemainingBarsAfter))
                    ? Number(cooldownDecisionDebug.persistedRemainingBarsAfter)
                    : null,
                cooldownPersistedError: cooldownDecisionDebug?.persistedError || null,
                slot02Aux1hEnabled: Boolean(slot02Aux1hDecisionDebug?.enabled),
                slot02Aux1hMode: slot02Aux1hDecisionDebug?.mode || null,
                slot02Aux1hDirection: slot02Aux1hDecisionDebug?.auxDirection || null,
                slot02Aux1hConfidence: Number.isFinite(Number(slot02Aux1hDecisionDebug?.confidence))
                    ? Number(slot02Aux1hDecisionDebug?.confidence)
                    : null,
                slot02Aux1hStrength: Number.isFinite(Number(slot02Aux1hDecisionDebug?.strength))
                    ? Number(slot02Aux1hDecisionDebug?.strength)
                    : null,
                slot02Aux1hUpStrengthThreshold: Number.isFinite(Number(slot02Aux1hDecisionDebug?.upStrengthThreshold))
                    ? Number(slot02Aux1hDecisionDebug?.upStrengthThreshold)
                    : null,
                slot02Aux1hDownStrengthThreshold: Number.isFinite(Number(slot02Aux1hDecisionDebug?.downStrengthThreshold))
                    ? Number(slot02Aux1hDecisionDebug?.downStrengthThreshold)
                    : null,
                slot02Aux1hMainConfidenceThreshold: Number.isFinite(Number(slot02Aux1hDecisionDebug?.mainConfidenceThreshold))
                    ? Number(slot02Aux1hDecisionDebug?.mainConfidenceThreshold)
                    : null,
                slot02Aux1hMainConfidenceBelowThreshold: typeof slot02Aux1hDecisionDebug?.mainConfidenceBelowThreshold === 'boolean'
                    ? slot02Aux1hDecisionDebug?.mainConfidenceBelowThreshold
                    : null,
                slot02Aux1hActionCode: Number.isFinite(Number(slot02Aux1hDecisionDebug?.actionCode))
                    ? Number(slot02Aux1hDecisionDebug?.actionCode)
                    : null,
                slot02Aux1hSoftCapPct: Number.isFinite(Number(slot02Aux1hDecisionDebug?.softCapPct))
                    ? Number(slot02Aux1hDecisionDebug?.softCapPct)
                    : null,
                slot02Aux1hWeakSupportActive: typeof slot02Aux1hDecisionDebug?.weakSupportActive === 'boolean'
                    ? slot02Aux1hDecisionDebug?.weakSupportActive
                    : null,
                slot02Aux1hMainConfidenceBucket: Number.isFinite(Number(slot02Aux1hDecisionDebug?.mainConfidenceBucket))
                    ? Number(slot02Aux1hDecisionDebug?.mainConfidenceBucket)
                    : null,
                slot02Aux1hVolValue: Number.isFinite(Number(slot02Aux1hDecisionDebug?.volValue))
                    ? Number(slot02Aux1hDecisionDebug?.volValue)
                    : null,
                slot02Aux1hVolBucket: Number.isFinite(Number(slot02Aux1hDecisionDebug?.volBucket))
                    ? Number(slot02Aux1hDecisionDebug?.volBucket)
                    : null,
                slot02Aux1hFreshnessSec: Number.isFinite(Number(slot02Aux1hDecisionDebug?.freshnessSec))
                    ? Number(slot02Aux1hDecisionDebug?.freshnessSec)
                    : null,
                slot02Aux1hBlocked: Boolean(slot02Aux1hDecisionDebug?.blocked),
                slot02Aux1hReason: slot02Aux1hDecisionDebug?.reason || null,
                slot02Aux1hModelVersion: slot02Aux1hDecisionDebug?.modelVersion || null,
                runtime_primary_block_reason: primaryBlockReason,
                threshold_block_reason: thresholdBlockReason,
                selection_route_applied: selectionDecision.applicable,
                selection_route_profile_scope: selectionRouteProfileScope,
                selection_route_action: selectionRouteAction,
                selection_route_reason_code: selectionDecision.reasonCode,
                uncertainty_gate_status: uncertaintyStatus,
                uncertainty_gate_reason_code: uncertaintyReasonCode,
                checkedAtMs: Date.now(),
                timestamp: p.timestamp,
            };
            appendDecisionFunnelLog(t.config.logsDir, {
                timestamp: p.timestamp,
                checkedAtMs: Date.now(),
                traderName: t.config.name,
                symbol,
                direction,
                marketSlug: (p as PredictionResult & { marketSlug?: string }).marketSlug || null,
                conditionId: (p as PredictionResult & { conditionId?: string }).conditionId || null,
                decisionStatus,
                primaryBlockReason,
                confidence: Number.isFinite(Number(p.confidence)) ? Number(p.confidence) : null,
                rawConfidence: Number.isFinite(Number(p.rawConfidence)) ? Number(p.rawConfidence) : null,
                calibratedConfidence: Number.isFinite(Number(p.calibratedConfidence)) ? Number(p.calibratedConfidence) : null,
                effectiveThreshold,
                gatesPassed: [...gatesPassed],
                selectionAction: selectionDecision.action,
                selectionApplicable: selectionDecision.applicable,
                selectionReasonCode: selectionDecision.reasonCode,
                marketQualityScore: selectionDecision.marketQualityScore,
                runtimePrimaryBlockReason: primaryBlockReason,
                finalDecision: decisionStatus,
                noTradeReason: decisionStatus === 'passed' ? null : primaryBlockReason,
                currentCapital: runtimeCurrentCapital,
                cooldownScope,
                cooldownKey,
                cooldownScopeLabel,
                cooldownRemainingBars: Number.isFinite(Number(cooldownDecisionDebug?.cooldownRemainingBars))
                    ? Number(cooldownDecisionDebug.cooldownRemainingBars)
                    : null,
                formalConsecutiveLosses: Number.isFinite(Number(cooldownDecisionDebug?.formalConsecutiveLosses))
                    ? Number(cooldownDecisionDebug.formalConsecutiveLosses)
                    : null,
                provisionalConsecutiveLosses: Number.isFinite(Number(cooldownDecisionDebug?.provisionalConsecutiveLosses))
                    ? Number(cooldownDecisionDebug.provisionalConsecutiveLosses)
                    : null,
                effectiveConsecutiveLosses: Number.isFinite(Number(cooldownDecisionDebug?.effectiveConsecutiveLosses))
                    ? Number(cooldownDecisionDebug.effectiveConsecutiveLosses)
                    : null,
                cooldownPendingResolutionBlocked: Boolean(cooldownDecisionDebug?.pendingResolutionBlocked),
                cooldownPendingResolutionMarketSlug: cooldownDecisionDebug?.pendingResolutionMarketSlug || null,
                cooldownPendingResolutionConsecutiveBeforeUnknown: Number.isFinite(Number(cooldownDecisionDebug?.pendingResolutionConsecutiveBeforeUnknown))
                    ? Number(cooldownDecisionDebug.pendingResolutionConsecutiveBeforeUnknown)
                    : null,
                cooldownPersistedWindowBlocked: Boolean(cooldownDecisionDebug?.persistedWindowBlocked),
                cooldownTriggerMarketSlug: cooldownDecisionDebug?.persistedTriggerMarketSlug || null,
                cooldownTriggerStage: cooldownDecisionDebug?.persistedTriggerStage || null,
                cooldownPersistedRemainingBarsBefore: Number.isFinite(Number(cooldownDecisionDebug?.persistedRemainingBarsBefore))
                    ? Number(cooldownDecisionDebug.persistedRemainingBarsBefore)
                    : null,
                cooldownPersistedRemainingBarsAfter: Number.isFinite(Number(cooldownDecisionDebug?.persistedRemainingBarsAfter))
                    ? Number(cooldownDecisionDebug.persistedRemainingBarsAfter)
                    : null,
                cooldownPersistedError: cooldownDecisionDebug?.persistedError || null,
                slot02Aux1hEnabled: Boolean(slot02Aux1hDecisionDebug?.enabled),
                slot02Aux1hMode: slot02Aux1hDecisionDebug?.mode || null,
                slot02Aux1hDirection: slot02Aux1hDecisionDebug?.auxDirection || null,
                slot02Aux1hConfidence: Number.isFinite(Number(slot02Aux1hDecisionDebug?.confidence))
                    ? Number(slot02Aux1hDecisionDebug?.confidence)
                    : null,
                slot02Aux1hStrength: Number.isFinite(Number(slot02Aux1hDecisionDebug?.strength))
                    ? Number(slot02Aux1hDecisionDebug?.strength)
                    : null,
                slot02Aux1hUpStrengthThreshold: Number.isFinite(Number(slot02Aux1hDecisionDebug?.upStrengthThreshold))
                    ? Number(slot02Aux1hDecisionDebug?.upStrengthThreshold)
                    : null,
                slot02Aux1hDownStrengthThreshold: Number.isFinite(Number(slot02Aux1hDecisionDebug?.downStrengthThreshold))
                    ? Number(slot02Aux1hDecisionDebug?.downStrengthThreshold)
                    : null,
                slot02Aux1hMainConfidenceThreshold: Number.isFinite(Number(slot02Aux1hDecisionDebug?.mainConfidenceThreshold))
                    ? Number(slot02Aux1hDecisionDebug?.mainConfidenceThreshold)
                    : null,
                slot02Aux1hMainConfidenceBelowThreshold: typeof slot02Aux1hDecisionDebug?.mainConfidenceBelowThreshold === 'boolean'
                    ? slot02Aux1hDecisionDebug?.mainConfidenceBelowThreshold
                    : null,
                slot02Aux1hActionCode: Number.isFinite(Number(slot02Aux1hDecisionDebug?.actionCode))
                    ? Number(slot02Aux1hDecisionDebug?.actionCode)
                    : null,
                slot02Aux1hSoftCapPct: Number.isFinite(Number(slot02Aux1hDecisionDebug?.softCapPct))
                    ? Number(slot02Aux1hDecisionDebug?.softCapPct)
                    : null,
                slot02Aux1hWeakSupportActive: typeof slot02Aux1hDecisionDebug?.weakSupportActive === 'boolean'
                    ? slot02Aux1hDecisionDebug?.weakSupportActive
                    : null,
                slot02Aux1hMainConfidenceBucket: Number.isFinite(Number(slot02Aux1hDecisionDebug?.mainConfidenceBucket))
                    ? Number(slot02Aux1hDecisionDebug?.mainConfidenceBucket)
                    : null,
                slot02Aux1hVolValue: Number.isFinite(Number(slot02Aux1hDecisionDebug?.volValue))
                    ? Number(slot02Aux1hDecisionDebug?.volValue)
                    : null,
                slot02Aux1hVolBucket: Number.isFinite(Number(slot02Aux1hDecisionDebug?.volBucket))
                    ? Number(slot02Aux1hDecisionDebug?.volBucket)
                    : null,
                slot02Aux1hFreshnessSec: Number.isFinite(Number(slot02Aux1hDecisionDebug?.freshnessSec))
                    ? Number(slot02Aux1hDecisionDebug?.freshnessSec)
                    : null,
                slot02Aux1hBlocked: Boolean(slot02Aux1hDecisionDebug?.blocked),
                slot02Aux1hReason: slot02Aux1hDecisionDebug?.reason || null,
                slot02Aux1hModelVersion: slot02Aux1hDecisionDebug?.modelVersion || null,
            });
        };
        p.uncertaintyMeta = {
            ...(p.uncertaintyMeta || {}),
            mode: uncertaintyMode,
            status: uncertaintyStatus,
            dispersionScore: uncertaintyDispersion ?? undefined,
            effectiveSourceCount: uncertaintySourceCount ?? undefined,
            changePointProb: uncertaintyChangePoint ?? undefined,
            intervalWidth: uncertaintyIntervalWidth ?? undefined,
            intervalWidthMax: uncertaintyIntervalWidthMax,
            intervalSource: 'proxy',
            degradedExtraDelta: uncertaintyDelta,
            reasonCode: uncertaintyReasonCode,
        };
        if (quality.mode === 'halt') {
            skippedByQuality += 1;
            writeDecision('blocked', 'data_quality_halt');
            continue;
        }
        gatesPassed.push('data_quality');
        const bypassMarketSelection = t.config.lowPriceBypassMarketSelection === true;
        const bypassThresholdBase = t.config.lowPriceBypassThresholdBase === true;
        if (selectionDecision.applicable && selectionDecision.action === 'abstain') {
            if (!bypassMarketSelection) {
                abstainedByMarketSelection += 1;
                writeDecision('abstained', null);
                continue;
            }
            gatesPassed.push('market_selection_bypassed');
        } else if (selectionDecision.applicable) {
            gatesPassed.push('market_selection');
        }
        const thresholdPassNoExtra = thresholdMetWithQuality(p, t.env, quality.confidencePenalty, 0);
        const thresholdPassWithDrift = thresholdMetWithQuality(p, t.env, quality.confidencePenalty, driftDelta);
        const thresholdPassWithRegime = thresholdMetWithQuality(p, t.env, quality.confidencePenalty, driftDelta + regimeDelta);
        const thresholdPassWithUncertainty = thresholdMetWithQuality(p, t.env, quality.confidencePenalty, driftDelta + regimeDelta + uncertaintyDelta);
        const thresholdPass = thresholdMetWithQuality(p, t.env, quality.confidencePenalty, driftDelta + regimeDelta + uncertaintyDelta + extremeExtraDelta);
        if (!thresholdPass) {
            if (thresholdPassWithUncertainty && extremeExtraDelta > 0) {
                skippedByExtremeMarket += 1;
                writeDecision('blocked', 'threshold_extreme_market');
                continue;
            } else if (thresholdPassWithRegime && uncertaintyDelta > 0) {
                skippedByUncertainty += 1;
                writeDecision('blocked', 'threshold_uncertainty');
                continue;
            } else if (thresholdPassWithDrift && regimeDelta > 0) {
                skippedByRegime += 1;
                writeDecision('blocked', 'threshold_regime');
                continue;
            } else if (thresholdPassNoExtra && driftDelta > 0) {
                skippedByThresholdDrift += 1;
                writeDecision('blocked', 'threshold_drift');
                continue;
            } else {
                if (bypassThresholdBase) {
                    gatesPassed.push('threshold_base_bypassed');
                } else {
                    writeDecision('blocked', 'threshold_base');
                    continue;
                }
            }
        }
        gatesPassed.push('threshold');
        if (regimeMode === 'enforce') {
            const policy = regimeState.directionPolicy;
            if (policy === 'NONE' || (policy === 'UP' && direction !== 'UP') || (policy === 'DOWN' && direction !== 'DOWN')) {
                skippedByRegime += 1;
                writeDecision('blocked', 'regime_direction_policy');
                continue;
            }
        }
        gatesPassed.push('regime');
        if (uncertaintyMode === 'enforce' && uncertaintyBlocked) {
            skippedByUncertainty += 1;
            writeDecision('blocked', 'uncertainty_blocked');
            continue;
        }
        gatesPassed.push('uncertainty');
        if (t.expectancyGateRuntimeBlockedBySymbolDir[symbolDirectionRuntimeKey(symbol, direction)]) {
            skippedByExpectancyGate += 1;
            writeDecision('blocked', 'expectancy_blocked');
            continue;
        }
        gatesPassed.push('expectancy');
        if (metaMode === 'enforce' && metaTakeTrade === false) {
            skippedByMetaLabel += 1;
            writeDecision('blocked', 'meta_label_blocked');
            continue;
        }
        gatesPassed.push('meta_label');
        if (t.shockRiskRuntimeBlockedBySymbolDir[shockRuntimeKey(symbol, direction)]) {
            skippedByShockRisk += 1;
            writeDecision('blocked', 'shock_risk_blocked');
            continue;
        }
        gatesPassed.push('shock_risk');
        if (t.comboPauseRuntimeBlockedBySymbolDir[symbolDirectionRuntimeKey(symbol, direction)]
            || t.comboPauseRuntimeBlockedBySymbol[symbol]) {
            skippedByComboPause += 1;
            writeDecision('blocked', 'combo_pause_blocked');
            continue;
        }
        gatesPassed.push('combo_pause');
        if (drawdownAccelerationBlocked) {
            skippedByDrawdownAcceleration += 1;
            writeDecision('blocked', 'drawdown_acceleration_blocked');
            continue;
        }
        gatesPassed.push('drawdown_acceleration');
        if (directionLossBlocked) {
            skippedByDirectionLossCooldown += 1;
            appendDirectionLossBlockedOpportunity(
                t,
                p,
                symbol,
                direction,
                effectiveThreshold,
                gatesPassed,
                finalRuntimeScale,
                directionLossState,
                ctx,
            );
            writeDecision('blocked', 'direction_loss_cooldown_blocked');
            continue;
        }
        gatesPassed.push('direction_loss_cooldown');
        if (p.direction === 'DOWN' && t.downRiskRuntimeBlockedBySymbol[symbol]) {
            skippedByDownBreaker += 1;
            writeDecision('blocked', 'down_risk_blocked');
            continue;
        }
        if (p.direction === 'UP' && t.upRiskRuntimeBlockedBySymbol[symbol]) {
            skippedByUpRisk += 1;
            writeDecision('blocked', 'up_risk_blocked');
            continue;
        }
        gatesPassed.push(p.direction === 'DOWN' ? 'down_risk' : 'up_risk');
        if (p.direction === 'DOWN') {
            const extraDelta = Math.max(0, Number(t.downRiskRuntimeExtraDeltaBySymbol[symbol] ?? 0));
            if (extraDelta > 0) {
                const base = getDirectionThreshold(p, t.env);
                const required = Math.min(0.999, Math.max(0, base + quality.confidencePenalty + driftDelta + regimeDelta + extraDelta));
                if (roundProb(effectiveConfidence(p)) < roundProb(required)) {
                    skippedByDownBreaker += 1;
                    writeDecision('blocked', 'down_risk_threshold');
                    continue;
                }
            }
        }
        if (p.direction === 'UP') {
            const extraDelta = Math.max(0, Number(t.upRiskRuntimeExtraDeltaBySymbol[symbol] ?? 0));
            if (extraDelta > 0) {
                const base = getDirectionThreshold(p, t.env);
                const required = Math.min(0.999, Math.max(0, base + quality.confidencePenalty + driftDelta + regimeDelta + extraDelta));
                if (roundProb(effectiveConfidence(p)) < roundProb(required)) {
                    skippedByUpRisk += 1;
                    writeDecision('blocked', 'up_risk_threshold');
                    continue;
                }
            }
        }
        if (selectorMode === 'enforce' && !selectorEligible) {
            skippedBySelector += 1;
            writeDecision('blocked', 'selector_ineligible');
            continue;
        }
        gatesPassed.push('selector');
        // v1 熔断退役后不再走该拦截分支，DOWN 拦截统一由 v2 负责。
        if (p.shouldTrade === false) {
            skippedShouldTrade += 1;
            writeDecision('blocked', 'should_trade_false');
            continue;
        }
        gatesPassed.push('should_trade');
        const slot01GentleFilter = evaluateSlot01GentleFilterAtDecisionLayer(
            t.config,
            t.env,
            symbol,
            direction,
            effectiveConfidence(p),
        );
        if (slot01GentleFilter.blocked) {
            const executorCurrentCapital = (() => {
                try {
                    const capital = (t.executor as any)?.getCoinCapital?.(symbol);
                    return Number.isFinite(Number(capital)) ? Number(capital) : null;
                } catch {
                    return null;
                }
            })();
            const gentleFilterCurrentCapital = readRuntimeCurrentCapitalForDecision(
                t.config.logsDir,
                t.env.TRADING_MODE,
                symbol,
                executorCurrentCapital,
            );
            appendSlot01GentleFilterBlockedLog(t.config.logsDir, t.env.TRADING_MODE, {
                timestamp: new Date().toISOString(),
                predictionTimestamp: p.timestamp,
                checkedAtMs: Date.now(),
                traderName: t.config.name,
                symbol,
                direction,
                marketSlug: (p as PredictionResult & { marketSlug?: string }).marketSlug || null,
                conditionId: (p as PredictionResult & { conditionId?: string }).conditionId || null,
                confidence: Number.isFinite(Number(p.confidence)) ? Number(p.confidence) : null,
                roundedConfidence: slot01GentleFilter.roundedConfidence,
                confidenceMin: slot01GentleFilter.confidenceMin,
                confidenceMax: slot01GentleFilter.confidenceMax,
                directionMode: slot01GentleFilter.directionMode,
                currentCapital: gentleFilterCurrentCapital,
                filterName: 'slot01_gentle_confidence_interval',
                filterReason: '温和信心区间过滤',
                executionDecision: 'slot01_gentle_filter_blocked_before_order',
                source: 'multi_prediction_index_decision_layer',
                explanationChinese: '该机会命中 slot1 ETH 温和信心区间过滤门，已在进入下单队列前跳过；模型、阈值、冷却、金额、时点均未改变。',
            });
            writeDecision('blocked', 'slot01_gentle_filter_blocked');
            continue;
        }
        slot02Aux1hDecisionDebug = evaluateSlot02Aux1hAtDecisionLayer(
            t.config,
            t.env,
            symbol,
            direction,
            Number.isFinite(Number(p.confidence)) ? Number(p.confidence) : null,
        );
        if (slot02Aux1hDecisionDebug.blocked) {
            const executorCurrentCapital = (() => {
                try {
                    const capital = (t.executor as any)?.getCoinCapital?.(symbol);
                    return Number.isFinite(Number(capital)) ? Number(capital) : null;
                } catch {
                    return null;
                }
            })();
            const slot02AuxCurrentCapital = readRuntimeCurrentCapitalForDecision(
                t.config.logsDir,
                t.env.TRADING_MODE,
                symbol,
                executorCurrentCapital,
            );
            appendSlot02Aux1hBlockedLog(t.config.logsDir, t.env.TRADING_MODE, {
                timestamp: new Date().toISOString(),
                predictionTimestamp: p.timestamp,
                checkedAtMs: Date.now(),
                traderName: t.config.name,
                symbol,
                direction,
                marketSlug: (p as PredictionResult & { marketSlug?: string }).marketSlug || null,
                conditionId: (p as PredictionResult & { conditionId?: string }).conditionId || null,
                confidence: Number.isFinite(Number(p.confidence)) ? Number(p.confidence) : null,
                currentCapital: slot02AuxCurrentCapital,
                auxMode: slot02Aux1hDecisionDebug.mode,
                auxDirection: slot02Aux1hDecisionDebug.auxDirection,
                auxConfidence: slot02Aux1hDecisionDebug.confidence,
                auxStrength: slot02Aux1hDecisionDebug.strength,
                auxUpStrengthThreshold: slot02Aux1hDecisionDebug.upStrengthThreshold,
                auxDownStrengthThreshold: slot02Aux1hDecisionDebug.downStrengthThreshold,
                auxMainConfidenceThreshold: slot02Aux1hDecisionDebug.mainConfidenceThreshold,
                auxMainConfidenceBelowThreshold: slot02Aux1hDecisionDebug.mainConfidenceBelowThreshold,
                auxFreshnessSec: slot02Aux1hDecisionDebug.freshnessSec,
                auxModelVersion: slot02Aux1hDecisionDebug.modelVersion,
                auxReason: slot02Aux1hDecisionDebug.reason,
                source: 'multi_prediction_index_decision_layer',
                explanationChinese: '该机会命中 slot2 BTC 一小时强逆风辅助门，已在创建订单前跳过；主模型、阈值、冷却、金额、时点均保持不变。',
            });
            writeDecision('blocked', slot02Aux1hDecisionDebug.reason || 'slot02_aux_1h_veto_blocked');
            continue;
        }
        if (Number.isFinite(Number(slot02Aux1hDecisionDebug.softCapPct)) && Number(slot02Aux1hDecisionDebug.softCapPct) > 0) {
            const softCapPct = Number(slot02Aux1hDecisionDebug.softCapPct);
            p.runtimeSizingCapPct = softCapPct;
            p.runtimeSizingCapReason = slot02Aux1hDecisionDebug.reason || 'slot02_aux_1h_soft_scaled';
            p.slot02Aux1hActionCode = Number.isFinite(Number(slot02Aux1hDecisionDebug.actionCode))
                ? Number(slot02Aux1hDecisionDebug.actionCode)
                : null;
            p.slot02Aux1hSoftCapPct = softCapPct;
            p.slot02Aux1hWeakSupportActive = slot02Aux1hDecisionDebug.weakSupportActive;
            p.slot02Aux1hMainConfidenceBucket = slot02Aux1hDecisionDebug.mainConfidenceBucket;
            p.slot02Aux1hVolValue = slot02Aux1hDecisionDebug.volValue;
            p.slot02Aux1hVolBucket = slot02Aux1hDecisionDebug.volBucket;
            const executorCurrentCapital = (() => {
                try {
                    const capital = (t.executor as any)?.getCoinCapital?.(symbol);
                    return Number.isFinite(Number(capital)) ? Number(capital) : null;
                } catch {
                    return null;
                }
            })();
            const slot02AuxCurrentCapital = readRuntimeCurrentCapitalForDecision(
                t.config.logsDir,
                t.env.TRADING_MODE,
                symbol,
                executorCurrentCapital,
            );
            appendSlot02Aux1hHandledLog(t.config.logsDir, t.env.TRADING_MODE, {
                timestamp: new Date().toISOString(),
                predictionTimestamp: p.timestamp,
                checkedAtMs: Date.now(),
                traderName: t.config.name,
                symbol,
                direction,
                marketSlug: (p as PredictionResult & { marketSlug?: string }).marketSlug || null,
                conditionId: (p as PredictionResult & { conditionId?: string }).conditionId || null,
                confidence: Number.isFinite(Number(p.confidence)) ? Number(p.confidence) : null,
                currentCapital: slot02AuxCurrentCapital,
                auxMode: slot02Aux1hDecisionDebug.mode,
                auxDirection: slot02Aux1hDecisionDebug.auxDirection,
                auxConfidence: slot02Aux1hDecisionDebug.confidence,
                auxStrength: slot02Aux1hDecisionDebug.strength,
                auxUpStrengthThreshold: slot02Aux1hDecisionDebug.upStrengthThreshold,
                auxDownStrengthThreshold: slot02Aux1hDecisionDebug.downStrengthThreshold,
                auxMainConfidenceThreshold: slot02Aux1hDecisionDebug.mainConfidenceThreshold,
                auxMainConfidenceBelowThreshold: slot02Aux1hDecisionDebug.mainConfidenceBelowThreshold,
                auxMainConfidenceBucket: slot02Aux1hDecisionDebug.mainConfidenceBucket,
                auxVolValue: slot02Aux1hDecisionDebug.volValue,
                auxVolBucket: slot02Aux1hDecisionDebug.volBucket,
                auxWeakSupportActive: slot02Aux1hDecisionDebug.weakSupportActive,
                auxActionCode: slot02Aux1hDecisionDebug.actionCode,
                auxSoftCapPct: softCapPct,
                auxFreshnessSec: slot02Aux1hDecisionDebug.freshnessSec,
                auxModelVersion: slot02Aux1hDecisionDebug.modelVersion,
                auxReason: slot02Aux1hDecisionDebug.reason,
                source: 'multi_prediction_index_decision_layer',
                explanationChinese: '该机会命中 slot2 BTC 第四版一小时辅助门，未硬拦截，但已在创建订单前把本单金额上限降到更保守档位；主模型、阈值、冷却、时点均保持不变。',
            });
        }
        p.effectiveThreshold = effectiveThreshold;
        writeDecision('passed', null);
        out.push(p);
    }

    return {
        toOrder: out,
        abstainedByMarketSelection,
        skippedShouldTrade,
        skippedByQuality,
        skippedByThresholdDrift,
        skippedByUncertainty,
        skippedByExpectancyGate,
        skippedByMetaLabel,
        skippedBySelector,
        skippedByRegime,
        skippedByShockRisk,
        skippedByComboPause,
        skippedByDownBreaker,
        skippedByUpRisk,
        skippedByExtremeMarket,
        skippedByDirectionLossCooldown,
        skippedByDrawdownAcceleration,
    };
}

function applyDownRiskRuntimeToExecutor(t: TraderInstance): void {
    const symbols = parseAllowedSymbols(t.config.allowedMarkets || '');
    for (const symbol of symbols) {
        const scale = Math.min(1, Math.max(0, Number(t.downRiskRuntimeScaleBySymbol[symbol] ?? 1)));
        const blocked = Boolean(t.downRiskRuntimeBlockedBySymbol[symbol]);
        t.executor.setRuntimeDownRiskScale(symbol, scale);
        t.executor.setRuntimeDownRiskBlocked(symbol, blocked);
    }
}

function applyUpRiskRuntimeToExecutor(t: TraderInstance): void {
    const symbols = parseAllowedSymbols(t.config.allowedMarkets || '');
    for (const symbol of symbols) {
        const scale = Math.min(1, Math.max(0, Number(t.upRiskRuntimeScaleBySymbol[symbol] ?? 1)));
        const blocked = Boolean(t.upRiskRuntimeBlockedBySymbol[symbol]);
        t.executor.setRuntimeUpRiskScale(symbol, scale);
        t.executor.setRuntimeUpRiskBlocked(symbol, blocked);
    }
}

function applyExpectancyGateRuntimeToExecutor(t: TraderInstance): void {
    const symbols = parseAllowedSymbols(t.config.allowedMarkets || '');
    for (const symbol of symbols) {
        for (const direction of ['UP', 'DOWN'] as const) {
            const key = symbolDirectionRuntimeKey(symbol, direction);
            const scale = Math.min(1, Math.max(0, Number(t.expectancyGateRuntimeScaleBySymbolDir[key] ?? 1)));
            const blocked = Boolean(t.expectancyGateRuntimeBlockedBySymbolDir[key]);
            t.executor.setRuntimeExpectancyScale(symbol, direction, scale);
            t.executor.setRuntimeExpectancyBlocked(symbol, direction, blocked);
        }
    }
}

function applyMainlineExtremeRuntimeToExecutor(t: TraderInstance): void {
    const symbols = parseAllowedSymbols(t.config.allowedMarkets || '');
    for (const symbol of symbols) {
        const scale = Math.min(1, Math.max(0, Number(t.mainlineExtremeRuntimeScaleBySymbol[symbol] ?? 1)));
        t.executor.setRuntimeSymbolScale(symbol, scale);
    }
}

function applyDirectionLossRuntimeToExecutor(t: TraderInstance): void {
    const symbols = parseAllowedSymbols(t.config.allowedMarkets || '');
    for (const symbol of symbols) {
        for (const direction of ['UP', 'DOWN'] as const) {
            const key = symbolDirectionRuntimeKey(symbol, direction);
            const scale = Math.min(1, Math.max(0, Number(t.directionLossCooldownSoftScaleBySymbolDir[key] ?? 1)));
            t.executor.setRuntimeDirectionLossScale(symbol, direction, scale);
        }
    }
}

function applyDrawdownAccelerationRuntimeToExecutor(t: TraderInstance): void {
    const symbols = parseAllowedSymbols(t.config.allowedMarkets || '');
    for (const symbol of symbols) {
        const scale = Math.min(1, Math.max(0, Number(t.drawdownAccelerationSoftScaleBySymbol[symbol] ?? 1)));
        t.executor.setRuntimeDrawdownAccelerationScale(symbol, scale);
    }
}

function runtimeGuardStatusFileForGroup(groupName: string): string {
    const safe = groupName.replace(/[^a-zA-Z0-9_-]/g, "_");
    return path.join(RUNTIME_GUARD_STATUS_DIR, `runtime_trade_guards_${safe}.json`);
}

function writeRuntimeGuardStatus(groupName: string, traders: TraderInstance[], quality: DataQualityState): void {
    try {
        const guardEvidenceRows = traders.map((t) => ({
            traderName: t.config.name,
            profile: t.config.profile ?? inferProfileFromGroup(t.config.group),
            group: t.config.group,
            ...t.guardEvidence,
            updatedAtMs: t.guardEvidence.updatedAtMs,
            updatedAt: t.guardEvidence.updatedAtMs > 0 ? new Date(t.guardEvidence.updatedAtMs).toISOString() : null,
        }));
        const guardEvidenceTotals = guardEvidenceRows.reduce((acc, row) => {
            acc.updatedAtMs = Math.max(acc.updatedAtMs, Number(row.updatedAtMs) || 0);
            acc.evaluated += Number(row.evaluated) || 0;
            acc.passed += Number(row.passed) || 0;
            acc.abstainedByMarketSelection += Number(row.abstainedByMarketSelection) || 0;
            acc.skippedShouldTrade += Number(row.skippedShouldTrade) || 0;
            acc.skippedByQuality += Number(row.skippedByQuality) || 0;
            acc.skippedByThresholdDrift += Number(row.skippedByThresholdDrift) || 0;
            acc.skippedByUncertainty += Number(row.skippedByUncertainty) || 0;
            acc.skippedByExpectancyGate += Number(row.skippedByExpectancyGate) || 0;
            acc.skippedByMetaLabel += Number(row.skippedByMetaLabel) || 0;
            acc.skippedBySelector += Number(row.skippedBySelector) || 0;
            acc.skippedByRegime += Number(row.skippedByRegime) || 0;
            acc.skippedByShockRisk += Number(row.skippedByShockRisk) || 0;
            acc.skippedByComboPause += Number(row.skippedByComboPause) || 0;
            acc.skippedByDownBreaker += Number(row.skippedByDownBreaker) || 0;
            acc.skippedByUpRisk += Number(row.skippedByUpRisk) || 0;
            acc.skippedByExtremeMarket += Number(row.skippedByExtremeMarket) || 0;
            acc.skippedByDirectionLossCooldown += Number(row.skippedByDirectionLossCooldown) || 0;
            acc.skippedByDrawdownAcceleration += Number(row.skippedByDrawdownAcceleration) || 0;
            return acc;
        }, freshGuardEvidenceStats());
        const payload = {
            timestamp: new Date().toISOString(),
            group: groupName,
            data_quality: {
                mode: quality.mode,
                severity: quality.severity,
                triggerClass: quality.triggerClass,
                confidencePenalty: quality.confidencePenalty,
                betScale: quality.betScale,
                apiErrorRate: quality.apiErrorRate,
                apiErrorSources: quality.apiErrorSources,
                keySourceErrors: quality.keySourceErrors,
                secondarySourceErrors: quality.secondarySourceErrors,
                nonKeySourceErrors: quality.nonKeySourceErrors,
                observationCounts: quality.observationCounts,
                consecutiveBadWindows: quality.consecutiveBadWindows,
                consecutiveGoodWindows: quality.consecutiveGoodWindows,
                staleSources: quality.staleSources,
                criticalStale: quality.criticalStale,
                reason: quality.reason,
                checkedAt: new Date(quality.checkedAtMs).toISOString(),
                recoveryHoldUntil: _qualityCache.holdUntilSec > 0 ? new Date(_qualityCache.holdUntilSec * 1000).toISOString() : null,
                recoveryStableChecks: _qualityCache.stableOkChecks,
            },
            guard_evidence_v1: {
                totals: {
                    ...guardEvidenceTotals,
                    updatedAt: guardEvidenceTotals.updatedAtMs > 0 ? new Date(guardEvidenceTotals.updatedAtMs).toISOString() : null,
                },
                traders: guardEvidenceRows,
            },
            down_breakers: traders.map((t) => ({
                name: t.config.name,
                group: t.config.group,
                active: Math.floor(Date.now() / 1000) < t.downBreakerUntilTs,
                untilTs: t.downBreakerUntilTs,
                    until: t.downBreakerUntilTs > 0 ? new Date(t.downBreakerUntilTs * 1000).toISOString() : null,
                    stats: t.downBreakerStats,
                })),
            down_breakers_v2: {
                cells: traders.flatMap((t) =>
                    [...t.downRiskCells.values()].map((cell) => ({
                        key: cell.key,
                        profile: cell.profile,
                        traderName: cell.traderName,
                        symbol: cell.symbol,
                        mode: cell.mode,
                        sourceMode: cell.sourceMode,
                        engine: resolveDownRiskEngine(t.config),
                        riskMode: resolveDownRiskMode(t.config, cell.symbol),
                        effectiveRiskMode: resolveDownRiskMode(t.config, cell.symbol),
                        tier: cell.tier,
                        reasonCode: cell.reasonCode,
                        triggerLane: cell.triggerLane,
                        enteredBy: cell.enteredBy,
                        lastTransitionAt: cell.lastTransitionAtMs > 0 ? new Date(cell.lastTransitionAtMs).toISOString() : null,
                        blocked: cell.blocked,
                        downScale: cell.downScale,
                        extraDelta: cell.extraDelta,
                        sampleCount: cell.main.trades,
                        hardUntilTs: cell.hardUntilTs,
                        hardUntil: cell.hardUntilTs > 0 ? new Date(cell.hardUntilTs * 1000).toISOString() : null,
                        releasePassStreak: cell.releasePassStreak,
                        releaseWindowMainPass: cell.releaseWindowMainPass,
                        releaseWindowConfirmPass: cell.releaseWindowConfirmPass,
                        checkedAt: cell.checkedAtMs > 0 ? new Date(cell.checkedAtMs).toISOString() : null,
                        lastAction: cell.lastAction,
                        main: cell.main,
                        fast: cell.fast,
                        calib: cell.calib,
                        releaseConfirm: cell.releaseConfirm,
                    })),
                ),
            },
            up_breakers_v2: {
                cells: traders.flatMap((t) =>
                    [...t.upRiskCells.values()].map((cell) => ({
                        key: cell.key,
                        profile: cell.profile,
                        traderName: cell.traderName,
                        symbol: cell.symbol,
                        mode: cell.mode,
                        sourceMode: cell.sourceMode,
                        engine: resolveUpRiskEngine(t.config),
                        riskMode: resolveUpRiskMode(t.config, cell.symbol),
                        effectiveRiskMode: resolveUpRiskMode(t.config, cell.symbol),
                        tier: cell.tier,
                        reasonCode: cell.reasonCode,
                        blocked: cell.blocked,
                        upScale: cell.upScale,
                        extraDelta: cell.extraDelta,
                        sampleCount: cell.main.trades,
                        releasePassStreak: cell.releasePassStreak,
                        checkedAt: cell.checkedAtMs > 0 ? new Date(cell.checkedAtMs).toISOString() : null,
                        lastAction: cell.lastAction,
                        main: cell.main,
                        calib: cell.calib,
                    })),
                ),
            },
            shock_breakers_v2: {
                cells: traders.flatMap((t) =>
                    [...t.shockRiskCells.values()].map((cell) => ({
                        key: cell.key,
                        profile: cell.profile,
                        traderName: cell.traderName,
                        symbol: cell.symbol,
                        direction: cell.direction,
                        mode: cell.mode,
                        sourceMode: cell.sourceMode,
                        engine: resolveShockRiskEngine(t.config),
                        riskMode: resolveShockRiskMode(t.config, cell.symbol),
                        effectiveRiskMode: resolveShockRiskMode(t.config, cell.symbol),
                        active: cell.active,
                        reasonCode: cell.reasonCode,
                        triggerLane: cell.triggerLane,
                        lastTransitionAt: cell.lastTransitionAtMs > 0 ? new Date(cell.lastTransitionAtMs).toISOString() : null,
                        blocked: cell.blocked,
                        sampleCount: cell.main.trades,
                        activeUntilTs: cell.activeUntilTs,
                        activeUntil: cell.activeUntilTs > 0 ? new Date(cell.activeUntilTs * 1000).toISOString() : null,
                        releasePassStreak: cell.releasePassStreak,
                        checkedAt: cell.checkedAtMs > 0 ? new Date(cell.checkedAtMs).toISOString() : null,
                        lastAction: cell.lastAction,
                        vol1h: cell.vol1h,
                        vol4h: cell.vol4h,
                        vol24h: cell.vol24h,
                        volRatio: cell.volRatio,
                        volPctl: cell.volPctl,
                        main: cell.main,
                    })),
                ),
            },
            combo_pauses_v1: {
                cells: traders.flatMap((t) =>
                    [...t.comboPauseCells.values()].map((cell) => ({
                        key: cell.key,
                        profile: cell.profile,
                        traderName: cell.traderName,
                        symbol: cell.symbol,
                        direction: cell.direction,
                        mode: cell.mode,
                        sourceMode: cell.sourceMode,
                        enabled: resolveComboPauseEnabled(t.config),
                        riskMode: resolveComboPauseModeByCell(t.config, cell.symbol, cell.direction),
                        effectiveRiskMode: resolveComboPauseModeByCell(t.config, cell.symbol, cell.direction),
                        active: cell.active,
                        reasonCode: cell.reasonCode,
                        triggerLane: cell.triggerLane,
                        lastTransitionAt: cell.lastTransitionAtMs > 0 ? new Date(cell.lastTransitionAtMs).toISOString() : null,
                        blocked: cell.blocked,
                        sampleCount: cell.sampleCount,
                        activeUntilTs: cell.activeUntilTs,
                        activeUntil: cell.activeUntilTs > 0 ? new Date(cell.activeUntilTs * 1000).toISOString() : null,
                        checkedAt: cell.checkedAtMs > 0 ? new Date(cell.checkedAtMs).toISOString() : null,
                        lastAction: cell.lastAction,
                        drawdown2h: cell.drawdown2h,
                        pnl2h: cell.pnl2h,
                        baseCapital: cell.baseCapital,
                    })),
                ),
            },
            prediction_calibration_v1: {
                cells: traders.flatMap((t) =>
                    [...t.calibrationCells.values()].map((cell) => ({
                        key: cell.key,
                        profile: cell.profile,
                        traderName: cell.traderName,
                        symbol: cell.symbol,
                        direction: cell.direction,
                        mode: cell.mode,
                        sourceMode: cell.sourceMode,
                        effectiveMode: resolveCalibrationMode(t.config, cell.symbol, cell.direction),
                        method: cell.method,
                        active: cell.active,
                        sampleCount: cell.stats.sampleCount,
                        minSamples: cell.minSamples,
                        lastConfidence: cell.lastConfidence,
                        calibratedConfidence: cell.calibratedConfidence,
                        checkedAt: cell.checkedAtMs > 0 ? new Date(cell.checkedAtMs).toISOString() : null,
                        lastAction: cell.lastAction,
                        reasonCode: cell.reasonCode,
                        stats: cell.stats,
                    })),
                ),
            },
            expectancy_gates_v1: {
                cells: traders.flatMap((t) =>
                    [...t.expectancyGateCells.values()].map((cell) => ({
                        key: cell.key,
                        profile: cell.profile,
                        traderName: cell.traderName,
                        symbol: cell.symbol,
                        direction: cell.direction,
                        mode: cell.mode,
                        sourceMode: cell.sourceMode,
                        effectiveMode: resolveExpectancyGateMode(t.config, cell.symbol, cell.direction),
                        status: cell.status,
                        triggerLane: cell.triggerLane,
                        blocked: cell.blocked,
                        degradedBetScale: cell.degradedBetScale,
                        degradedExtraDelta: cell.degradedExtraDelta,
                        sampleCount: cell.stats.trades,
                        activeUntilTs: cell.activeUntilTs,
                        activeUntil: cell.activeUntilTs > 0 ? new Date(cell.activeUntilTs * 1000).toISOString() : null,
                        releasePassStreak: cell.releasePassStreak,
                        lastTransitionAt: cell.lastTransitionAtMs > 0 ? new Date(cell.lastTransitionAtMs).toISOString() : null,
                        checkedAt: cell.checkedAtMs > 0 ? new Date(cell.checkedAtMs).toISOString() : null,
                        lastAction: cell.lastAction,
                        reasonCode: cell.reasonCode,
                        stats: cell.stats,
                    })),
                ),
            },
            threshold_drifts_v1: {
                cells: traders.flatMap((t) =>
                    [...t.thresholdDriftCells.values()].map((cell) => ({
                        key: cell.key,
                        profile: cell.profile,
                        traderName: cell.traderName,
                        symbol: cell.symbol,
                        direction: cell.direction,
                        mode: cell.mode,
                        sourceMode: cell.sourceMode,
                        effectiveMode: resolveThresholdDriftMode(t.config, cell.symbol, cell.direction),
                        status: cell.status,
                        active: cell.active,
                        thresholdDelta: cell.thresholdDelta,
                        sampleCount: cell.stats.trades,
                        checkedAt: cell.checkedAtMs > 0 ? new Date(cell.checkedAtMs).toISOString() : null,
                        lastAction: cell.lastAction,
                        reasonCode: cell.reasonCode,
                        stats: cell.stats,
                    })),
                ),
            },
            meta_labels_v1: {
                cells: traders.flatMap((t) =>
                    parseAllowedSymbols(t.config.allowedMarkets || '')
                        .filter((symbol) => isThresholdDriftTargetSymbol(symbol))
                        .flatMap((symbol) => (['UP', 'DOWN'] as const).map((direction) => {
                            const profile = t.config.profile ?? inferProfileFromGroup(t.config.group);
                            const key = metaLabelCellKey(profile, t.config.name, symbol, direction);
                            const cell = t.metaLabelCells.get(key);
                            return {
                                key,
                                profile,
                                traderName: t.config.name,
                                symbol,
                                direction,
                                mode: cell?.mode || resolveMetaLabelMode(t.config, symbol, direction),
                                effectiveMode: resolveMetaLabelMode(t.config, symbol, direction),
                                takeTrade: cell?.takeTrade ?? true,
                                reasonCode: cell?.reasonCode || 'no_decision_yet',
                                metaConfidence: cell?.metaConfidence ?? null,
                                evaluated: cell?.evaluated ?? 0,
                                passed: cell?.passed ?? 0,
                                blocked: cell?.blocked ?? 0,
                                checkedAt: (cell?.checkedAtMs ?? 0) > 0 ? new Date(Number(cell?.checkedAtMs || 0)).toISOString() : null,
                                lastAction: cell?.lastAction || 'init',
                            };
                        })),
                ),
            },
            selector_overlays_v1: {
                cells: traders.flatMap((t) =>
                    [...t.selectorCells.values()].map((cell) => {
                        const selectorApplicable = isSelectorTargetSymbol(cell.symbol);
                        return {
                            key: cell.key,
                            profile: cell.profile,
                            traderName: cell.traderName,
                            symbol: cell.symbol,
                            direction: 'BOTH',
                            mode: cell.mode,
                            sourceMode: cell.sourceMode,
                            effectiveMode: selectorApplicable ? resolveSelectorMode(t.config, cell.symbol) : 'off',
                            status: selectorApplicable ? cell.status : 'n_a_by_design',
                            eligible: selectorApplicable ? cell.eligible : true,
                            scope: cell.scope,
                            checkedAt: cell.checkedAtMs > 0 ? new Date(cell.checkedAtMs).toISOString() : null,
                            lastAction: selectorApplicable ? cell.lastAction : 'n_a_by_design',
                            reasonCode: selectorApplicable ? cell.reasonCode : 'symbol_not_target',
                        };
                    }),
                ),
            },
            regime_gates_v1: {
                cells: traders.flatMap((t) =>
                    Object.values(t.regimeRuntimeBySymbol).map((state) => ({
                        key: `${state.profile}::${t.config.name}::${state.symbol}`,
                        profile: state.profile,
                        traderName: t.config.name,
                        symbol: state.symbol,
                        mode: state.mode,
                        sourceMode: state.sourceMode,
                        effectiveMode: state.effectiveMode,
                        active: state.active,
                        directionPolicy: state.directionPolicy,
                        thresholdDelta: state.thresholdDelta,
                        regimeId: state.regimeId,
                        reasonCode: state.reasonCode,
                        checkedAt: state.checkedAtMs > 0 ? new Date(state.checkedAtMs).toISOString() : null,
                    })),
                ),
            },
            cooldown_v2: {
                cells: traders.flatMap((t) =>
                    t.executor.getCooldownSnapshot().map((cd) => ({
                        traderName: t.config.name,
                        profile: t.config.profile || inferProfileFromGroup(t.config.group),
                        group: t.config.group,
                        key: cd.key,
                        symbol: cd.symbol,
                        direction: cd.direction,
                        remainingBars: cd.remainingBars,
                        scope: cd.scope,
                    })),
                ),
            },
            mainline_extreme_v1: {
                cells: traders.flatMap((t) =>
                    Object.values(t.mainlineExtremeRuntimeBySymbol).map((state) => ({
                        key: `${state.profile}::${t.config.name}::${state.symbol}`,
                        traderName: t.config.name,
                        group: t.config.group,
                        profile: state.profile,
                        symbol: state.symbol,
                        active: state.active,
                        laneState: state.laneState,
                        regimeId: state.regimeId,
                        directionPolicy: state.directionPolicy,
                        policyMode: state.policyMode,
                        policyConfidence: state.policyConfidence,
                        thresholdExtraDelta: state.thresholdExtraDelta,
                        betScale: state.betScale,
                        reasonCode: state.reasonCode,
                        checkedAt: state.checkedAtMs > 0 ? new Date(state.checkedAtMs).toISOString() : null,
                    })),
                ),
            },
            direction_loss_cooldown_v1: {
                cells: traders.flatMap((t) =>
                    Object.values(t.directionLossCooldownBySymbolDir).map((state) => ({
                        key: state.key,
                        traderName: t.config.name,
                        group: t.config.group,
                        profile: t.config.profile || inferProfileFromGroup(t.config.group),
                        symbol: state.symbol,
                        direction: state.direction,
                        mode: state.mode || resolveDirectionLossCooldownMode(t.config, state.symbol, state.direction),
                        blocked: state.blocked,
                        evidenceFresh: state.evidenceFresh,
                        softScaled: state.softScaled,
                        softOverlapWithExtreme: state.softOverlapWithExtreme,
                        softScale: state.softScale,
                        softSeverity: state.softSeverity,
                        streakLosses: state.streakLosses,
                        recentTrades: state.recentTrades,
                        recentPnl: state.recentPnl,
                        recentHighConfidenceLosses: state.recentHighConfidenceLosses,
                        recentAvgLossConfidence: state.recentAvgLossConfidence,
                        untilTs: state.untilTs,
                        until: state.untilTs > 0 ? new Date(state.untilTs * 1000).toISOString() : null,
                        reasonCode: state.reasonCode,
                        softReasonCode: state.softReasonCode,
                        checkedAt: state.checkedAtMs > 0 ? new Date(state.checkedAtMs).toISOString() : null,
                    })),
                ),
            },
            drawdown_acceleration_v1: {
                cells: traders.flatMap((t) =>
                    Object.values(t.drawdownAccelerationBySymbol).map((state) => ({
                        key: `${t.config.profile || inferProfileFromGroup(t.config.group)}::${t.config.name}::${state.symbol}`,
                        traderName: t.config.name,
                        group: t.config.group,
                        profile: t.config.profile || inferProfileFromGroup(t.config.group),
                        symbol: state.symbol,
                        mode: state.mode || resolveDrawdownAccelerationMode(t.config, state.symbol),
                        effectiveMode: state.mode || resolveDrawdownAccelerationMode(t.config, state.symbol),
                        blocked: state.blocked,
                        evidenceFresh: state.evidenceFresh,
                        softScaled: state.softScaled,
                        softScale: state.softScale,
                        recentTrades: state.recentTrades,
                        recentLosses: state.recentLosses,
                        recentPnl: state.recentPnl,
                        untilTs: state.untilTs,
                        until: state.untilTs > 0 ? new Date(state.untilTs * 1000).toISOString() : null,
                        reasonCode: state.reasonCode,
                        softReasonCode: state.softReasonCode,
                        checkedAt: state.checkedAtMs > 0 ? new Date(state.checkedAtMs).toISOString() : null,
                    })),
                ),
            },
            execution_decisions_v1: {
                cells: traders.flatMap((t) =>
                    Object.values(t.executionDecisionBySymbolDir).map((state) => ({
                        ...state,
                        checkedAt: state.checkedAtMs > 0 ? new Date(state.checkedAtMs).toISOString() : null,
                    })),
                ),
            },
        };
        const targetFile = runtimeGuardStatusFileForGroup(groupName);
        const dir = path.dirname(targetFile);
        if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
        fs.writeFileSync(targetFile, JSON.stringify(payload, null, 2), 'utf8');
        if (WRITE_LEGACY_RUNTIME_GUARD) {
            fs.writeFileSync(LEGACY_RUNTIME_GUARD_STATUS_FILE, JSON.stringify(payload, null, 2), 'utf8');
        }
    } catch {
        // guard 状态写盘失败不影响主交易循环
    }
}

type RoundResultLike = {
    success?: boolean;
    pendingMonitor?: boolean;
    skippedReason?: string;
    error?: string;
};

function shouldConsumeTargetTs(results: RoundResultLike[] | undefined): boolean {
    if (!Array.isArray(results) || results.length === 0) return false;
    return results.some((r) => Boolean(r.success) || Boolean(r.pendingMonitor) || r.skippedReason === 'already_bought');
}

function summarizeExecutionResults(results: RoundResultLike[] | undefined): {
    consumed: boolean;
    primaryReason: string | null;
    errors: string[];
    successCount: number;
    pendingMonitorCount: number;
} {
    const rows = Array.isArray(results) ? results : [];
    const errors = rows
        .map((r) => String(r.error || r.skippedReason || '').trim())
        .filter((s) => s.length > 0);
    const consumed = shouldConsumeTargetTs(rows);
    const primaryRaw = errors[0] || null;
    let primaryReason: string | null = null;
    if (!consumed) {
        if (primaryRaw && primaryRaw.includes('冷却待确认')) {
            primaryReason = 'execution_pending_cooldown_resolution';
        } else if (primaryRaw && primaryRaw.includes('冷却中:')) {
            primaryReason = 'execution_cooldown_bars_blocked';
        } else if (primaryRaw && primaryRaw.includes('全部未成交')) {
            primaryReason = 'execution_order_created_unfilled';
        } else if (primaryRaw && primaryRaw.includes('全部被交易所拒绝')) {
            primaryReason = 'execution_exchange_rejected';
        } else if (primaryRaw && primaryRaw.includes('GTD 过期窗口不足')) {
            primaryReason = 'execution_market_expired_before_submit';
        } else if (primaryRaw && primaryRaw.includes('official_canceled_zero_fill')) {
            primaryReason = 'execution_official_canceled_zero_fill';
        } else if (primaryRaw && primaryRaw.includes('官网未确认自动订单收单')) {
            primaryReason = 'execution_official_not_accepted';
        } else if (primaryRaw && primaryRaw.includes('金额不足')) {
            primaryReason = 'execution_amount_below_min';
        } else if (primaryRaw && primaryRaw.includes('市场已近乎定价该结果')) {
            primaryReason = 'execution_market_near_priced_skip';
        } else if (primaryRaw && primaryRaw.includes('风险控制')) {
            primaryReason = 'execution_risk_control';
        } else if (primaryRaw && primaryRaw.includes('绝对止损')) {
            primaryReason = 'execution_absolute_stop';
        } else if (primaryRaw === 'already_bought') {
            primaryReason = 'execution_already_bought';
        } else {
            primaryReason = 'execution_no_order_created';
        }
    }
    return {
        consumed,
        primaryReason,
        errors,
        successCount: rows.filter((r) => Boolean(r.success)).length,
        pendingMonitorCount: rows.filter((r) => Boolean(r.pendingMonitor)).length,
    };
}

function appendExecutionOutcomeFunnelLog(
    t: TraderInstance,
    targetTs: number,
    phase: number | null | undefined,
    fileTimestamp: string,
    predictions: PredictionResult[],
    results: RoundResultLike[] | undefined,
): void {
    const summary = summarizeExecutionResults(results);
    const first = predictions[0];
    const symbol = first?.symbol ? normalizeSymbol(first.symbol) : null;
    const direction = first?.direction || null;
    const marketSlug = (first as (PredictionResult & { marketSlug?: string }) | undefined)?.marketSlug || null;
    const cooldownScope = (() => {
        try {
            const scope = (t.executor as any)?.getCooldownScope?.();
            if (scope === 'symbol' || scope === 'symbol_direction') {
                return scope;
            }
        } catch {
            // fall through to config fallback
        }
        const cfgScope = String((t.config as any)?.cooldownScope || '').trim().toLowerCase();
        if (cfgScope === 'symbol' || cfgScope === 'symbol_direction') {
            return cfgScope;
        }
        if (Number((t.config as any)?.cooldownBars || 0) > 0) {
            return 'symbol_direction';
        }
        return null;
    })();
    const cooldownKey = (() => {
        if (!cooldownScope || !symbol || !direction) return null;
        try {
            const key = (t.executor as any)?.buildCooldownKey?.(symbol, direction);
            return typeof key === 'string' && key.trim() ? key : null;
        } catch {
            return cooldownScope === 'symbol' ? symbol : `${symbol}_${direction}`;
        }
    })();
    const cooldownScopeLabel = cooldownScope === 'symbol'
        ? 'total_chain_by_symbol'
        : cooldownScope === 'symbol_direction'
            ? 'symbol_direction'
            : null;
    const cooldownDecisionDebug = (() => {
        try {
            const targetPeriodStartTs = parsePeriodStartFromSlug(marketSlug) ?? undefined;
            return (t.executor as any)?.getCooldownDecisionDebugSnapshot?.(
                symbol,
                direction,
                targetPeriodStartTs,
            ) ?? null;
        } catch {
            return null;
        }
    })();
    const cooldownErrorFallback = (() => {
        const firstError = summary.errors.find((row) => typeof row === 'string' && row.trim().length > 0) || '';
        if (!firstError) return null;
        const triggerSlug = (() => {
            const m = firstError.match(/由\s+([a-z0-9-]+)\s+触发/i);
            return m?.[1] || null;
        })();
        const barsMeta = (() => {
            const m = firstError.match(/第\s*(\d+)\s*\/\s*(\d+)\s*根/);
            if (!m) return null;
            const currentIdx = Number(m[1]);
            const total = Number(m[2]);
            if (!Number.isFinite(currentIdx) || !Number.isFinite(total) || currentIdx <= 0 || total <= 0) {
                return null;
            }
            return {
                remainingBarsBefore: Math.max(0, total - currentIdx + 1),
                remainingBarsAfter: Math.max(0, total - currentIdx),
            };
        })();
        return {
            pendingResolutionBlocked: firstError.includes('冷却待确认'),
            persistedWindowBlocked: firstError.includes('冷却中:'),
            triggerSlug,
            remainingBarsBefore: barsMeta?.remainingBarsBefore ?? null,
            remainingBarsAfter: barsMeta?.remainingBarsAfter ?? null,
        };
    })();
    appendDecisionFunnelLog(t.config.logsDir, {
        timestamp: new Date().toISOString(),
        checkedAtMs: Date.now(),
        traderName: t.config.name,
        symbol,
        direction,
        marketSlug,
        conditionId: (first as (PredictionResult & { conditionId?: string }) | undefined)?.conditionId || null,
        decisionStatus: summary.consumed ? 'passed' : 'blocked',
        primaryBlockReason: summary.primaryReason,
        finalDecision: summary.consumed ? 'passed' : 'blocked',
        noTradeReason: summary.primaryReason,
        executionStage: 'post_execution',
        executionStatus: summary.consumed ? 'order_created_or_monitoring' : 'execution_not_consumed',
        executionPrimaryReason: summary.primaryReason,
        executionErrors: summary.errors,
        executionSuccessCount: summary.successCount,
        executionPendingMonitorCount: summary.pendingMonitorCount,
        executionResultCount: Array.isArray(results) ? results.length : 0,
        targetPeriodEndTs: targetTs,
        phase,
        predictionFileTimestamp: fileTimestamp,
        confidence: Number.isFinite(Number(first?.confidence)) ? Number(first?.confidence) : null,
        rawConfidence: Number.isFinite(Number(first?.rawConfidence)) ? Number(first?.rawConfidence) : null,
        calibratedConfidence: Number.isFinite(Number(first?.calibratedConfidence)) ? Number(first?.calibratedConfidence) : null,
        currentCapital: (() => {
            const executorCurrentCapital = (() => {
                try {
                    const capital = symbol ? (t.executor as any)?.getCoinCapital?.(symbol) : null;
                    return Number.isFinite(Number(capital)) ? Number(capital) : null;
                } catch {
                    return null;
                }
            })();
            return readRuntimeCurrentCapitalForDecision(
                t.config.logsDir,
                t.env.TRADING_MODE,
                symbol,
                executorCurrentCapital,
            );
        })(),
        cooldownScope,
        cooldownKey,
        cooldownScopeLabel,
        cooldownRemainingBars: Number.isFinite(Number(cooldownDecisionDebug?.cooldownRemainingBars))
            ? Number(cooldownDecisionDebug.cooldownRemainingBars)
            : Number.isFinite(Number(cooldownErrorFallback?.remainingBarsAfter))
                ? Number(cooldownErrorFallback?.remainingBarsAfter)
            : null,
        formalConsecutiveLosses: Number.isFinite(Number(cooldownDecisionDebug?.formalConsecutiveLosses))
            ? Number(cooldownDecisionDebug.formalConsecutiveLosses)
            : null,
        provisionalConsecutiveLosses: Number.isFinite(Number(cooldownDecisionDebug?.provisionalConsecutiveLosses))
            ? Number(cooldownDecisionDebug.provisionalConsecutiveLosses)
            : null,
        effectiveConsecutiveLosses: Number.isFinite(Number(cooldownDecisionDebug?.effectiveConsecutiveLosses))
            ? Number(cooldownDecisionDebug.effectiveConsecutiveLosses)
            : null,
        cooldownPendingResolutionBlocked: Boolean(
            cooldownDecisionDebug?.pendingResolutionBlocked
            || cooldownErrorFallback?.pendingResolutionBlocked
        ),
        cooldownPendingResolutionMarketSlug: cooldownDecisionDebug?.pendingResolutionMarketSlug
            || (cooldownErrorFallback?.pendingResolutionBlocked ? cooldownErrorFallback?.triggerSlug : null)
            || null,
        cooldownPendingResolutionConsecutiveBeforeUnknown: Number.isFinite(Number(cooldownDecisionDebug?.pendingResolutionConsecutiveBeforeUnknown))
            ? Number(cooldownDecisionDebug.pendingResolutionConsecutiveBeforeUnknown)
            : null,
        cooldownPersistedWindowBlocked: Boolean(
            cooldownDecisionDebug?.persistedWindowBlocked
            || cooldownErrorFallback?.persistedWindowBlocked
        ),
        cooldownTriggerMarketSlug: cooldownDecisionDebug?.persistedTriggerMarketSlug
            || cooldownErrorFallback?.triggerSlug
            || null,
        cooldownTriggerStage: cooldownDecisionDebug?.persistedTriggerStage || null,
        cooldownPersistedRemainingBarsBefore: Number.isFinite(Number(cooldownDecisionDebug?.persistedRemainingBarsBefore))
            ? Number(cooldownDecisionDebug.persistedRemainingBarsBefore)
            : Number.isFinite(Number(cooldownErrorFallback?.remainingBarsBefore))
                ? Number(cooldownErrorFallback?.remainingBarsBefore)
            : null,
        cooldownPersistedRemainingBarsAfter: Number.isFinite(Number(cooldownDecisionDebug?.persistedRemainingBarsAfter))
            ? Number(cooldownDecisionDebug.persistedRemainingBarsAfter)
            : Number.isFinite(Number(cooldownErrorFallback?.remainingBarsAfter))
                ? Number(cooldownErrorFallback?.remainingBarsAfter)
            : null,
        cooldownPersistedError: cooldownDecisionDebug?.persistedError || null,
    });
}

type ActiveTraderSelection = {
    names: Set<string> | null;
    disabledSymbolsByTrader: Map<string, Set<string>>;
    enabledSymbolsByTrader: Map<string, Set<string>>;
};

function loadActiveTraderSelection(configPath: string): ActiveTraderSelection {
    const explicitPath = process.env.ACTIVE_TRADER_FILE?.trim();
    const activePath = explicitPath && explicitPath.length > 0
        ? explicitPath
        : path.resolve(__dirname, '..', 'active_traders.json');
    if (!fs.existsSync(activePath)) {
        return { names: null, disabledSymbolsByTrader: new Map(), enabledSymbolsByTrader: new Map() };
    }

    try {
        const raw = JSON.parse(fs.readFileSync(activePath, 'utf8')) as unknown;
        let names: string[] = [];
        const disabledSymbolsByTrader = new Map<string, Set<string>>();
        const enabledSymbolsByTrader = new Map<string, Set<string>>();
        if (Array.isArray(raw)) {
            names = raw.filter((x) => typeof x === 'string') as string[];
        } else if (raw && typeof raw === 'object') {
            const obj = raw as Record<string, unknown>;
            // Prefer active_traders (runtime allowlist), fallback to traderNames (legacy field).
            if (Array.isArray(obj.active_traders)) {
                names = obj.active_traders.filter((x) => typeof x === 'string') as string[];
            } else if (Array.isArray(obj.traderNames)) {
                names = obj.traderNames.filter((x) => typeof x === 'string') as string[];
            }
            const disabledRaw = obj.disabledSymbolsByTrader;
            if (disabledRaw && typeof disabledRaw === 'object' && !Array.isArray(disabledRaw)) {
                for (const [trader, symbols] of Object.entries(disabledRaw)) {
                    if (!Array.isArray(symbols)) continue;
                    const normalized = new Set(
                        symbols
                            .filter((x) => typeof x === 'string')
                            .map((x) => String(x).trim().toUpperCase())
                            .filter((x) => x.length > 0),
                    );
                    if (normalized.size > 0) disabledSymbolsByTrader.set(String(trader).trim(), normalized);
                }
            }
            const enabledRaw = obj.enabledSymbolsByTrader;
            if (enabledRaw && typeof enabledRaw === 'object' && !Array.isArray(enabledRaw)) {
                for (const [trader, symbols] of Object.entries(enabledRaw)) {
                    if (!Array.isArray(symbols)) continue;
                    const normalized = new Set(
                        symbols
                            .filter((x) => typeof x === 'string')
                            .map((x) => String(x).trim().toUpperCase())
                            .filter((x) => x.length > 0),
                    );
                    if (normalized.size > 0) enabledSymbolsByTrader.set(String(trader).trim(), normalized);
                }
            }
        }
        const set = new Set(names.map((s) => s.trim()).filter((s) => s.length > 0));
        const selection = {
            names: set.size === 0 ? null : set,
            disabledSymbolsByTrader,
            enabledSymbolsByTrader,
        };
        if (selection.names) {
            console.log(`  🧩 已加载 ACTIVE_TRADER_FILE: ${activePath} (${selection.names.size} names)`);
        }
        return selection;
    } catch (err) {
        console.warn(`  ⚠️  ACTIVE_TRADER_FILE 解析失败，忽略白名单: ${String(err)}`);
        return { names: null, disabledSymbolsByTrader: new Map(), enabledSymbolsByTrader: new Map() };
    }
}

function filterConfigSymbolsByActiveSelection(config: TraderConfig, selection: ActiveTraderSelection): TraderConfig | null {
    const names = selection.names;
    if (names && !names.has(config.name)) return null;
    const allowedSymbols = parseAllowedSymbols(config.allowedMarkets || '');
    if (allowedSymbols.length === 0) return config;
    const explicitEnabled = selection.enabledSymbolsByTrader.get(config.name);
    const explicitDisabled = selection.disabledSymbolsByTrader.get(config.name);
    const filtered = allowedSymbols.filter((symbol) => {
        const normalized = symbol.toUpperCase();
        if (explicitEnabled) return explicitEnabled.has(normalized);
        if (explicitDisabled) return !explicitDisabled.has(normalized);
        return true;
    });
    if (filtered.length === 0) return null;
    if (filtered.length === allowedSymbols.length) return config;
    return {
        ...config,
        allowedMarkets: filtered.join(','),
    };
}

function loadLiveSelectedCapitalByCell(): Map<string, { capital: number; reason: string }> {
    const filePath = path.join(process.cwd(), 'live_selected_cells.json');
    const out = new Map<string, { capital: number; reason: string }>();
    try {
        if (!fs.existsSync(filePath)) return out;
        const payload = JSON.parse(fs.readFileSync(filePath, 'utf8'));
        const rows = Array.isArray(payload?.selected_cells) ? payload.selected_cells : [];
        for (const row of rows) {
            if (!row || typeof row !== 'object' || row.enabled === false) continue;
            const profile = String(row.profile || 'default').trim() || 'default';
            const trader = String(row.trader || '').trim();
            const symbol = String(row.symbol || '').trim().toUpperCase();
            const walletSlot = String(row.walletSlot || '').trim();
            const capital = Number(row.displayReferenceCapitalUsd ?? row.capitalBaselineUsd ?? 0);
            if (!trader || !symbol || !walletSlot || !Number.isFinite(capital) || capital <= 0) continue;
            const reason = String(
                row.capitalBaselineReason
                || row.cutoverReason
                || 'live_selected_display_reference_capital'
            ).trim() || 'live_selected_display_reference_capital';
            out.set(`${profile}::${trader}::${symbol}::${walletSlot}`, {
                capital: Math.round(capital * 10000) / 10000,
                reason,
            });
        }
    } catch {
        return out;
    }
    return out;
}

function _isPlainObject(value: unknown): value is Record<string, unknown> {
    return !!value && typeof value === 'object' && !Array.isArray(value);
}

function _cloneJsonValue<T>(value: T): T {
    return JSON.parse(JSON.stringify(value)) as T;
}

function sanitizeLiveSelectedRuntimeOverrideConfig(raw: unknown): Partial<TraderConfig> {
    if (!_isPlainObject(raw)) return {};
    const out: Partial<TraderConfig> = {};
    const scalarKeys: Array<keyof TraderConfig> = [
        'probThresholdUp',
        'probThresholdDown',
        'consensusThresholdUp',
        'consensusThresholdDown',
        'minEdgeUp',
        'minEdgeDown',
        'capitalRiskCapPct',
        'cooldownBars',
        'cooldownAfterLosses',
    ];
    for (const key of scalarKeys) {
        const value = raw[String(key)];
        if (typeof value === 'number' && Number.isFinite(value)) {
            (out as Record<string, unknown>)[String(key)] = value;
        }
    }
    const modeKeys: Array<keyof TraderConfig> = [
        'downRiskMode',
        'upRiskMode',
        'drawdownAccelerationMode',
        'mainlineExtremeMode',
        'expectancyGateMode',
        'thresholdDriftMode',
        'calibrationMode',
        'runtimeRiskScalingMode',
        'directionLossCooldownMode',
        'forcedDirectionMode',
        'featureVersion',
    ];
    for (const key of modeKeys) {
        const value = raw[String(key)];
        if (typeof value === 'string' && value.trim()) {
            (out as Record<string, unknown>)[String(key)] = value.trim();
        }
    }
    const objectKeys: Array<keyof TraderConfig> = [
        'downRiskModeBySymbol',
        'upRiskModeBySymbol',
        'drawdownAccelerationModeBySymbol',
        'mainlineExtremeModeBySymbol',
        'downRiskBySymbol',
        'expectancyGateModeBySymbolDirection',
        'expectancyGateBySymbolDirection',
        'thresholdDriftModeBySymbolDirection',
        'thresholdDriftBySymbolDirection',
        'calibrationModeBySymbolDirection',
        'calibrationBySymbolDirection',
        'directionLossCooldownModeBySymbolDirection',
    ];
    for (const key of objectKeys) {
        const value = raw[String(key)];
        if (_isPlainObject(value)) {
            (out as Record<string, unknown>)[String(key)] = _cloneJsonValue(value);
        }
    }
    return out;
}

function loadLiveSelectedRuntimeOverridesByCell(): Map<string, { config: Partial<TraderConfig>; source: string; scope: string }> {
    const filePath = path.join(process.cwd(), 'live_selected_cells.json');
    const out = new Map<string, { config: Partial<TraderConfig>; source: string; scope: string }>();
    try {
        if (!fs.existsSync(filePath)) return out;
        const payload = JSON.parse(fs.readFileSync(filePath, 'utf8'));
        const rows = Array.isArray(payload?.selected_cells) ? payload.selected_cells : [];
        for (const row of rows) {
            if (!row || typeof row !== 'object' || row.enabled === false) continue;
            const profile = String(row.profile || 'default').trim() || 'default';
            const trader = String(row.trader || '').trim();
            const symbol = String(row.symbol || '').trim().toUpperCase();
            const walletSlot = String(row.walletSlot || '').trim();
            const overridePayload = _isPlainObject(row.runtimeParamOverrides)
                ? row.runtimeParamOverrides
                : {};
            const overrideConfig = sanitizeLiveSelectedRuntimeOverrideConfig(
                _isPlainObject(overridePayload.config) ? overridePayload.config : overridePayload,
            );
            if (!trader || !symbol || !walletSlot || Object.keys(overrideConfig).length === 0) continue;
            const source = String(
                overridePayload.source
                || row.runtimeParamOverrideSource
                || row.cutoverReason
                || 'live_selected_runtime_override'
            ).trim() || 'live_selected_runtime_override';
            const scope = String(
                overridePayload.scope
                || row.runtimeParamOverrideScope
                || 'cell_specific_runtime_override'
            ).trim() || 'cell_specific_runtime_override';
            out.set(`${profile}::${trader}::${symbol}::${walletSlot}`, {
                config: overrideConfig,
                source,
                scope,
            });
        }
    } catch {
        return out;
    }
    return out;
}

function resolveExpandedGroupConfigs(
    groupName: string,
    configPath: string,
): {
    allConfigs: TraderConfig[];
    groupConfigs: TraderConfig[];
    modeState: ModeRegistryRuntimeState;
} {
    const allConfigs: TraderConfig[] = JSON.parse(fs.readFileSync(configPath, 'utf8'));
    let groupConfigs = allConfigs.filter((c) => c.group === groupName);

    const activeSelection = loadActiveTraderSelection(configPath);
    groupConfigs = groupConfigs
        .map((c) => filterConfigSymbolsByActiveSelection(c, activeSelection))
        .filter((c): c is TraderConfig => c !== null);

    if (groupConfigs.length === 0) {
        throw new Error(`未找到 group="${groupName}" 的配置`);
    }

    const modeState = resolveModeRegistryRuntimeState(groupName, configPath, groupConfigs);
    if (modeState.errors.length > 0 && !modeState.failOpen) {
        throw new Error(`mode registry 校验失败且 failOpen=false: ${modeState.errors.join(' | ')}`);
    }

    const overridesByTrader = new Map<string, Record<string, TradingModeValue>>();
    const slotByTrader = new Map<string, Record<string, string>>();
    const liveSelectedCapitalByCell = loadLiveSelectedCapitalByCell();
    const liveSelectedRuntimeOverridesByCell = loadLiveSelectedRuntimeOverridesByCell();
    for (const entry of modeState.entries) {
        const symbol = entry.symbol.toUpperCase();
        const modeMap = overridesByTrader.get(entry.traderName) || {};
        modeMap[symbol] = entry.appliedMode;
        overridesByTrader.set(entry.traderName, modeMap);
        if (entry.walletSlot) {
            const slotMap = slotByTrader.get(entry.traderName) || {};
            slotMap[symbol] = entry.walletSlot;
            slotByTrader.set(entry.traderName, slotMap);
        }
    }

    const routedConfigs = groupConfigs.map((cfg) => ({
        ...cfg,
        profile: modeState.profile,
        symbolModeOverrides: overridesByTrader.get(cfg.name) || {},
        walletSlotBySymbol: slotByTrader.get(cfg.name) || {},
        liveSelectedCapitalBySymbol: Object.fromEntries(
            Object.entries(slotByTrader.get(cfg.name) || {}).flatMap(([symbol, walletSlot]) => {
                const hit = liveSelectedCapitalByCell.get(`${modeState.profile}::${cfg.name}::${String(symbol).toUpperCase()}::${walletSlot}`);
                return hit ? [[String(symbol).toUpperCase(), hit.capital]] : [];
            }),
        ),
        liveSelectedCapitalReasonBySymbol: Object.fromEntries(
            Object.entries(slotByTrader.get(cfg.name) || {}).flatMap(([symbol, walletSlot]) => {
                const hit = liveSelectedCapitalByCell.get(`${modeState.profile}::${cfg.name}::${String(symbol).toUpperCase()}::${walletSlot}`);
                return hit ? [[String(symbol).toUpperCase(), hit.reason]] : [];
            }),
        ),
        liveSelectedRuntimeOverrideBySymbol: Object.fromEntries(
            Object.entries(slotByTrader.get(cfg.name) || {}).flatMap(([symbol, walletSlot]) => {
                const hit = liveSelectedRuntimeOverridesByCell.get(`${modeState.profile}::${cfg.name}::${String(symbol).toUpperCase()}::${walletSlot}`);
                return hit ? [[String(symbol).toUpperCase(), _cloneJsonValue(hit.config)]] : [];
            }),
        ),
        liveSelectedRuntimeOverrideSourceBySymbol: Object.fromEntries(
            Object.entries(slotByTrader.get(cfg.name) || {}).flatMap(([symbol, walletSlot]) => {
                const hit = liveSelectedRuntimeOverridesByCell.get(`${modeState.profile}::${cfg.name}::${String(symbol).toUpperCase()}::${walletSlot}`);
                return hit ? [[String(symbol).toUpperCase(), hit.source]] : [];
            }),
        ),
        liveSelectedRuntimeOverrideScopeBySymbol: Object.fromEntries(
            Object.entries(slotByTrader.get(cfg.name) || {}).flatMap(([symbol, walletSlot]) => {
                const hit = liveSelectedRuntimeOverridesByCell.get(`${modeState.profile}::${cfg.name}::${String(symbol).toUpperCase()}::${walletSlot}`);
                return hit ? [[String(symbol).toUpperCase(), hit.scope]] : [];
            }),
        ),
    }));

    const expandedConfigs: TraderConfig[] = [];
    for (const cfg of routedConfigs) {
        const expanded = expandRuntimeTraderConfigs(cfg, modeState);
        expandedConfigs.push(...expanded.configs);
    }

    return {
        allConfigs,
        groupConfigs: expandedConfigs,
        modeState,
    };
}

function statFileMtimeMs(filePath: string): number {
    try {
        return fs.statSync(filePath).mtimeMs;
    } catch {
        return 0;
    }
}

function applyReloadedTraderConfig(
    trader: TraderInstance,
    nextConfig: TraderConfig,
): boolean {
    const criticalFieldMismatches: string[] = [];
    if (trader.config.name !== nextConfig.name) criticalFieldMismatches.push('name');
    if (trader.config.logsDir !== nextConfig.logsDir) criticalFieldMismatches.push('logsDir');
    if ((trader.config.allowedMarkets || '') !== (nextConfig.allowedMarkets || '')) criticalFieldMismatches.push('allowedMarkets');
    if ((trader.config.appliedTradingMode || '') !== (nextConfig.appliedTradingMode || '')) criticalFieldMismatches.push('appliedTradingMode');
    if ((trader.config.runtimeSymbol || '') !== (nextConfig.runtimeSymbol || '')) criticalFieldMismatches.push('runtimeSymbol');
    if ((trader.config.runtimeBaseName || '') !== (nextConfig.runtimeBaseName || '')) criticalFieldMismatches.push('runtimeBaseName');
    if ((trader.config.runtimeBaseLogsDir || '') !== (nextConfig.runtimeBaseLogsDir || '')) criticalFieldMismatches.push('runtimeBaseLogsDir');

    if (criticalFieldMismatches.length > 0) {
        console.warn(
            `  ⚠️ [${trader.config.name}] 检测到需重启才能安全应用的配置变化: ${criticalFieldMismatches.join(', ')}，本进程先保留旧关键字段`,
        );
        return false;
    }

    const currentName = trader.config.name;
    const previousPredictionFilePath = trader.predictionFilePath;
    const nextEnv = createPredictionEnv(buildEnvRecord(nextConfig));
    const nextPredictionFilePath = resolveConfiguredPredictionPath(nextConfig);
    const nextPredictionFallbackFilePath = resolveConfiguredPredictionFallbackPath(nextConfig, nextPredictionFilePath);
    trader.config = { ...nextConfig };
    Object.assign(trader.env as Record<string, unknown>, nextEnv as Record<string, unknown>);
    trader.predictionFilePath = nextPredictionFilePath;
    trader.predictionFallbackFilePath = nextPredictionFallbackFilePath;
    trader.lastPredictionPathDecision = undefined;

    if (previousPredictionFilePath !== nextPredictionFilePath) {
        console.log(`  🔁 [${currentName}] prediction suffix 更新 -> ${path.basename(nextPredictionFilePath)}`);
    }
    return true;
}

function maybeReloadRuntimeConfigs(
    groupName: string,
    configPath: string,
    traders: TraderInstance[],
    state: { lastCheckMs: number; lastMtimeMs: number },
    nowMs: number,
): void {
    if (nowMs - state.lastCheckMs < CONFIG_RELOAD_CHECK_INTERVAL_MS) return;
    state.lastCheckMs = nowMs;

    const mtimeMs = statFileMtimeMs(configPath);
    if (mtimeMs <= 0 || mtimeMs <= state.lastMtimeMs) return;

    try {
        const resolved = resolveExpandedGroupConfigs(groupName, configPath);
        writeModeEffectiveSnapshot(groupName, resolved.modeState);
        const nextByName = new Map(resolved.groupConfigs.map((cfg) => [cfg.name, cfg]));
        let updated = 0;

        for (const trader of traders) {
            const nextCfg = nextByName.get(trader.config.name);
            if (!nextCfg) {
                console.warn(`  ⚠️ [${trader.config.name}] 热更新时未找到对应 runtime 配置，保持当前运行态`);
                continue;
            }
            if (applyReloadedTraderConfig(trader, nextCfg)) {
                updated += 1;
            }
        }

        state.lastMtimeMs = mtimeMs;
        console.log(`  🔄 [${groupName}] runtime 配置热更新完成: ${updated}/${traders.length} traders`);
    } catch (err) {
        console.warn(`  ⚠️ [${groupName}] runtime 配置热更新失败: ${String(err)}`);
    }
}

function groupPidFile(groupName: string): string {
    return path.resolve(__dirname, '..', `multi_${groupName}.pid`);
}

function isSameGroupProcessAlive(groupName: string, pid: number): boolean {
    if (!Number.isFinite(pid) || pid <= 0 || pid === process.pid) return false;
    try {
        process.kill(pid, 0);
    } catch {
        return false;
    }
    try {
        const cmd = execSync(`ps -p ${pid} -o args=`, { timeout: 1200, encoding: 'utf8' }).trim();
        if (!cmd) return false;
        const expectedA = `--group ${groupName}`;
        const expectedB = `--group=${groupName}`;
        return cmd.includes('multi_prediction_index') && (cmd.includes(expectedA) || cmd.includes(expectedB));
    } catch {
        return false;
    }
}

function acquireGroupSingletonLock(groupName: string): () => void {
    const pidFile = groupPidFile(groupName);
    const currentPid = process.pid;

    if (fs.existsSync(pidFile)) {
        const raw = fs.readFileSync(pidFile, 'utf8').trim();
        const oldPid = Number(raw);
        if (Number.isFinite(oldPid) && oldPid > 0 && oldPid !== currentPid && isSameGroupProcessAlive(groupName, oldPid)) {
            console.error(`❌ 检测到 group=${groupName} 已有实例在运行 (PID=${oldPid})，当前实例退出以避免重复下单。`);
            process.exit(2);
        }
        try { fs.unlinkSync(pidFile); } catch { /* ignore stale file cleanup */ }
    }

    fs.writeFileSync(pidFile, `${currentPid}\n`, 'utf8');
    const cleanup = () => {
        try {
            if (!fs.existsSync(pidFile)) return;
            const raw = fs.readFileSync(pidFile, 'utf8').trim();
            if (Number(raw) === currentPid) fs.unlinkSync(pidFile);
        } catch {
            // ignore
        }
    };
    process.on('exit', cleanup);
    return cleanup;
}

// ============================================================
// 主函数
// ============================================================

async function main(): Promise<void> {
    const groupArg = process.argv.find((a) => a.startsWith('--group='))?.split('=')[1]
        || process.argv[process.argv.indexOf('--group') + 1];

    if (!groupArg) {
        console.error('用法: npx ts-node src/multi_prediction_index.ts --group <group_name>');
        process.exit(1);
    }
    acquireGroupSingletonLock(groupArg);

    const configPath = process.env.TRADER_CONFIGS_FILE
        ? path.resolve(process.env.TRADER_CONFIGS_FILE)
        : path.resolve(__dirname, '..', 'trader_configs.json');
    if (!fs.existsSync(configPath)) {
        console.error(`❌ 配置文件不存在: ${configPath}`);
        process.exit(1);
    }

    let allConfigs: TraderConfig[] = [];
    let groupConfigs: TraderConfig[] = [];
    let modeState: ModeRegistryRuntimeState;
    try {
        const resolved = resolveExpandedGroupConfigs(groupArg, configPath);
        allConfigs = resolved.allConfigs;
        groupConfigs = resolved.groupConfigs;
        modeState = resolved.modeState;
    } catch (err) {
        console.error(`❌ ${String(err)}`);
        try {
            allConfigs = JSON.parse(fs.readFileSync(configPath, 'utf8'));
            const groups = Array.from(new Set(allConfigs.map((c) => c.group)));
            console.error(`可用的 group: ${groups.join(', ')}`);
        } catch {
            // ignore
        }
        process.exit(1);
    }

    if (modeState.errors.length > 0) {
        const msg = modeState.errors.join(' | ');
        if (modeState.failOpen) {
            console.warn(`  ⚠️ mode registry 校验失败，按 simulation 回落: ${msg}`);
        } else {
            console.error(`  ❌ mode registry 校验失败且 failOpen=false: ${msg}`);
            process.exit(1);
        }
    }
    if (modeState.warnings.length > 0) {
        console.warn(`  ⚠️ mode registry 警告: ${modeState.warnings.join(' | ')}`);
    }
    writeModeEffectiveSnapshot(groupArg, modeState);

    console.log(`\n╔════════════════════════════════════════════════════════════════╗`);
    console.log(`║  多 Trader 编排器 — group: ${groupArg.padEnd(33)}║`);
    console.log(`║  Traders: ${String(groupConfigs.length).padEnd(51)}║`);
    console.log(`╚════════════════════════════════════════════════════════════════╝\n`);

    // ── 创建 trader 实例 ────────────────────────────────────
    const traders: TraderInstance[] = [];

    for (const cfg of groupConfigs) {
        if (cfg.runnerManaged) {
            console.log(
                `  [${cfg.name}] 前向专用链: runnerManaged=true, simulationSource=${String(cfg.simulationSource || 'forward_runner_only')}；`
                + '跳过通用执行器，避免双写模拟账本。',
            );
            continue;
        }
        const envRecord = buildEnvRecord(cfg);
        const env = createPredictionEnv(envRecord);
        const executor = new PredictionExecutor(env, cfg.logsDir);
        const logDirFull = path.join(process.cwd(), cfg.logsDir);
        const predictionFilePath = resolveConfiguredPredictionPath(cfg);
        const predictionFallbackFilePath = resolveConfiguredPredictionFallbackPath(cfg, predictionFilePath);

        // 启动时参数验证打印（风险 C 防护）
        const walletMask = env.PROXY_WALLET
            ? `${env.PROXY_WALLET.slice(0, 6)}...${env.PROXY_WALLET.slice(-4)}`
            : '(模拟模式)';
        const privateKeyPresent = Boolean(String(env.PRIVATE_KEY || '').trim());
        const proxyWalletPresent = Boolean(String(env.PROXY_WALLET || '').trim());
        const polyApiCredsPresent = Boolean(
            String(env.POLY_API_KEY || '').trim()
            && String(env.POLY_API_SECRET || '').trim()
            && String(env.POLY_API_PASSPHRASE || '').trim(),
        );
        const builderCredsPresent = Boolean(
            String(env.BUILDER_API_KEY || '').trim()
            && String(env.BUILDER_API_SECRET || '').trim()
            && String(env.BUILDER_API_PASSPHRASE || '').trim(),
        );
        const upTh = env.PROB_THRESHOLD_UP ?? env.PROB_THRESHOLD;
        const downTh = env.PROB_THRESHOLD_DOWN != null ? (1 - env.PROB_THRESHOLD_DOWN) : env.PROB_THRESHOLD;
        console.log(`  [${cfg.name}] 预测文件: ${predictionFilePath}`);
        if (predictionFallbackFilePath && predictionFallbackFilePath !== predictionFilePath) {
            console.log(`  [${cfg.name}] fallback预测文件: ${predictionFallbackFilePath} | stale阈值=${SYMBOL_OVERRIDE_STALE_SEC}s`);
        }
        if (cfg.runnerManaged) {
            console.log(`  [${cfg.name}] 前向专用链: runnerManaged=true, simulationSource=${String(cfg.simulationSource || 'forward_runner_only')}`);
        }
        if (String(cfg.predictionSourceStatus || '').trim() === 'non_runnable_missing_prediction_source') {
            console.log(`  [${cfg.name}] 预测源状态: non_runnable_missing_prediction_source (${String(cfg.nonRunnableReason || 'missing_prediction_source')})`);
        }
        console.log(
            `  [${cfg.name}] LOGS_DIR: ${cfg.logsDir} | 初始资金: $${cfg.initialCapital} | 币种: ${cfg.allowedMarkets} `
            + `| 阈值: UP>=${upTh.toFixed(3)}, DOWN>=${downTh.toFixed(3)} | 钱包: ${walletMask}`,
        );
        if (cfg.appliedTradingMode === 'live') {
            console.log(
                `  [${cfg.name}] live env wiring: privateKeyEnv=${cfg.privateKeyEnv || '(none)'} `
                + `privateKeyPresent=${privateKeyPresent} proxyWalletEnv=${cfg.proxyWalletEnv || '(none)'} `
                + `proxyWalletPresent=${proxyWalletPresent} polyApiCredsPresent=${polyApiCredsPresent} `
                + `builderCredsPresent=${builderCredsPresent}`,
            );
        }
        if (cfg.usePredictionLimitPrice === true) {
            const envLimitStr = Number.isFinite(env.LIMIT_PRICE) ? `$${env.LIMIT_PRICE.toFixed(2)}` : '—';
            console.log(`  [${cfg.name}] 限价来源: 预测文件优先 | 环境默认限价仅作回退显示 (${envLimitStr})`);
        }

        traders.push({
            config: cfg,
            env,
            executor,
            executedTargetTss: loadExecutedTargetTss(cfg.logsDir),
            executedPhaseKeys: new Map(),
            lastPredictionTimestampByTs: new Map(),
            predictionFilePath,
            predictionFallbackFilePath,
            lastPredictionPathDecision: undefined,
            consecutiveErrors: 0,
            paused: false,
            downBreakerUntilTs: 0,
            downBreakerLastCheckMs: 0,
            downBreakerLastFileMtimeMs: 0,
            downBreakerStats: {
                checkedAtMs: 0,
                lookbackHours: Number(cfg.downBreakerLookbackHours ?? 8),
                trades: 0,
                wins: 0,
                winRate: 1,
                pnl: 0,
                pnlPerTrade: 0,
                triggered: false,
                reason: 'init',
            },
            downRiskCells: new Map(),
            downRiskLastCheckMsByMode: {
                simulation: 0,
                live: 0,
            },
            downRiskLastFileMtimeMsByMode: {
                simulation: 0,
                live: 0,
            },
            downRiskRuntimeScaleBySymbol: {},
            downRiskRuntimeBlockedBySymbol: {},
            downRiskRuntimeExtraDeltaBySymbol: {},
            upRiskCells: new Map(),
            upRiskLastCheckMsByMode: {
                simulation: 0,
                live: 0,
            },
            upRiskLastFileMtimeMsByMode: {
                simulation: 0,
                live: 0,
            },
            upRiskRuntimeScaleBySymbol: {},
            upRiskRuntimeBlockedBySymbol: {},
            upRiskRuntimeExtraDeltaBySymbol: {},
            shockRiskCells: new Map(),
            shockRiskLastCheckMsByMode: {
                simulation: 0,
                live: 0,
            },
            shockRiskLastFileMtimeMsByMode: {
                simulation: 0,
                live: 0,
            },
            shockRiskRuntimeBlockedBySymbolDir: {},
            comboPauseCells: new Map(),
            comboPauseLastCheckMsByMode: {
                simulation: 0,
                live: 0,
            },
            comboPauseLastFileMtimeMsByMode: {
                simulation: 0,
                live: 0,
            },
            comboPauseRuntimeBlockedBySymbol: {},
            comboPauseRuntimeBlockedBySymbolDir: {},
            calibrationCells: new Map(),
            calibrationLastCheckMsByMode: {
                simulation: 0,
                live: 0,
            },
            calibrationLastFileMtimeMsByMode: {
                simulation: 0,
                live: 0,
            },
            expectancyGateCells: new Map(),
            expectancyGateLastCheckMsByMode: {
                simulation: 0,
                live: 0,
            },
            expectancyGateLastFileMtimeMsByMode: {
                simulation: 0,
                live: 0,
            },
            expectancyGateRuntimeBlockedBySymbolDir: {},
            expectancyGateRuntimeScaleBySymbolDir: {},
            expectancyGateRuntimeExtraDeltaBySymbolDir: {},
            thresholdDriftCells: new Map(),
            thresholdDriftLastCheckMsByMode: {
                simulation: 0,
                live: 0,
            },
            thresholdDriftLastFileMtimeMsByMode: {
                simulation: 0,
                live: 0,
            },
            thresholdDriftRuntimeExtraDeltaBySymbolDir: {},
            metaLabelCells: new Map(),
            selectorCells: new Map(),
            selectorLastCheckMsByMode: {
                simulation: 0,
                live: 0,
            },
            selectorRuntimeEligibleBySymbol: {},
            regimeLastCheckMsByMode: {
                simulation: 0,
                live: 0,
            },
            regimeLastFileMtimeMsByMode: {
                simulation: 0,
                live: 0,
            },
            regimeRuntimeBySymbol: {},
            mainlineExtremeRuntimeBySymbol: {},
            mainlineExtremeRuntimeScaleBySymbol: {},
            mainlineExtremeRuntimeExtraDeltaBySymbolDir: {},
            directionLossCooldownBySymbolDir: {},
            directionLossCooldownBlockedBySymbolDir: {},
            directionLossCooldownSoftScaleBySymbolDir: {},
            directionLossCooldownLastTriggerKeyBySymbolDir: {},
            drawdownAccelerationBySymbol: {},
            drawdownAccelerationBlockedBySymbol: {},
            drawdownAccelerationSoftScaleBySymbol: {},
            drawdownAccelerationLastTriggerKeyBySymbol: {},
            guardEvidence: freshGuardEvidenceStats(),
            executionDecisionBySymbolDir: {},
            blockedOpportunityLogKeys: new Set<string>(),
        });
    }

    console.log(`\n✅ 已创建 ${traders.length} 个 trader 实例\n`);

    // ── 初始化所有 executor ─────────────────────────────────
    console.log('🔧 正在初始化所有 executor...\n');
    const initResults = await Promise.all(
        traders.map(async (t) => {
            try {
                const ok = await t.executor.initialize();
                if (!ok) {
                    console.error(`  ❌ [${t.config.name}] 初始化失败`);
                }
                return ok;
            } catch (err) {
                console.error(`  ❌ [${t.config.name}] 初始化异常:`, err);
                return false;
            }
        }),
    );

    const failedCount = initResults.filter((ok) => !ok).length;
    if (failedCount === traders.length) {
        console.error('\n❌ 所有 trader 初始化失败，退出\n');
        process.exit(1);
    }
    if (failedCount > 0) {
        console.warn(`\n⚠️ ${failedCount}/${traders.length} 个 trader 初始化失败，其余继续运行\n`);
    }

    // P2: 将初始化失败的 trader 标记为 paused，避免其参与扫描循环
    for (let i = 0; i < traders.length; i++) {
        if (!initResults[i]) {
            traders[i].paused = true;
            console.warn(`  🚫 [${traders[i].config.name}] 已标记为 paused（初始化失败）`);
        }
    }

    // 资金重算后验证打印（风险 C 防护）
    for (const t of traders) {
        const comboName = t.executor.getComboDisplayName();
        console.log(`  [${comboName}] 已执行周期: ${t.executedTargetTss.size} 个`);
    }

    // ── 共享 WebSocket ──────────────────────────────────────
    const marketOrderbookWS = new MarketOrderbookWS({
        getSubscribedTokenIds: () => {
            const allIds: string[] = [];
            for (const t of traders) {
                allIds.push(...t.executor.getMonitorTokenIds());
            }
            return Array.from(new Set(allIds));
        },
        onBestAsk: (assetId, bestAsk, asksSnapshot, isFullBook) => {
            for (const t of traders) {
                if (t.paused) continue;
                try {
                    t.executor.tryTriggerPriceMonitorByTokenId(assetId, bestAsk, asksSnapshot).catch(() => {});
                    t.executor.tryTriggerLowPriceMonitorByTokenId(assetId, bestAsk, asksSnapshot).catch(() => {});
                    if (asksSnapshot && asksSnapshot.length > 0) {
                        t.executor.tryTriggerLiquidityMonitorByTokenId(assetId, bestAsk, asksSnapshot).catch(() => {});
                    }
                } catch { /* per-trader isolation */ }
            }
        },
        verbose: false,
    });
    marketOrderbookWS.start();
    console.log(`\n📡 共享 WebSocket 已启动\n`);

    // ── User Channel WebSocket（LIVE 模式，按凭证分组避免多钱包串事件）───
    const userChannelWSList: UserChannelWS[] = [];
    {
        const liveByCred = new Map<string, { apiKey: string; apiSecret: string; apiPassphrase: string; traders: TraderInstance[] }>();
        const incompleteLiveTraders: string[] = [];

        for (const t of traders) {
            if (t.env.TRADING_MODE !== TradingMode.LIVE) continue;
            const { POLY_API_KEY, POLY_API_SECRET, POLY_API_PASSPHRASE } = t.env;
            if (!POLY_API_KEY || !POLY_API_SECRET || !POLY_API_PASSPHRASE) {
                incompleteLiveTraders.push(t.config.name);
                continue;
            }
            const credKey = `${POLY_API_KEY}::${POLY_API_SECRET}::${POLY_API_PASSPHRASE}`;
            const group = liveByCred.get(credKey);
            if (group) {
                group.traders.push(t);
            } else {
                liveByCred.set(credKey, {
                    apiKey: POLY_API_KEY,
                    apiSecret: POLY_API_SECRET,
                    apiPassphrase: POLY_API_PASSPHRASE,
                    traders: [t],
                });
            }
        }

        if (incompleteLiveTraders.length > 0) {
            console.log(`⚠️ 以下 LIVE trader 缺少完整 POLY_API_* 凭证，将退回 REST 成交查询: ${incompleteLiveTraders.join(', ')}`);
        }

        for (const group of liveByCred.values()) {
            const ws = new UserChannelWS({
                apiKey: group.apiKey,
                apiSecret: group.apiSecret,
                apiPassphrase: group.apiPassphrase,
                getSubscribedMarkets: () => {
                    const allIds: string[] = [];
                    for (const t of group.traders) {
                        if (t.paused) continue;
                        allIds.push(...t.executor.getLiveOrderConditionIds());
                    }
                    return [...new Set(allIds)];
                },
                onTrade: (event) => {
                    for (const t of group.traders) {
                        if (t.paused) continue;
                        try { t.executor.onWsTrade(event as any); } catch { /* isolation */ }
                    }
                },
                onOrder: (event) => {
                    for (const t of group.traders) {
                        if (t.paused) continue;
                        try { t.executor.onWsOrder(event as any); } catch { /* isolation */ }
                    }
                },
                verbose: false,
            });
            ws.start();
            userChannelWSList.push(ws);
        }

        if (userChannelWSList.length > 0) {
            console.log(`📡 User Channel WebSocket 已启动 (LIVE 模式, ${userChannelWSList.length} 条凭证通道)\n`);
        }
    }

    // ── 信号处理（只在编排器注册 — 风险 H 防护）────────────
    let isShuttingDown = false;

    const shutdown = async () => {
        if (isShuttingDown) return;
        isShuttingDown = true;
        console.log('\n\n🛑 正在优雅关闭所有 trader...\n');

        // 清理所有定时器
        for (const h of timerHandles) clearTimeout(h);
        timerHandles.length = 0;

        marketOrderbookWS.stop();
        for (const ws of userChannelWSList) ws.stop();

        await Promise.all(
            traders.map(async (t) => {
                try {
                    t.executor.stop();
                    await t.executor.generateReports();
                    t.executor.printStatus();
                } catch (err) {
                    console.error(`  [${t.config.name}] 关闭异常:`, err);
                }
            }),
        );

        console.log('\n👋 再见！\n');
        process.exit(0);
    };

    process.on('SIGINT', shutdown);
    process.on('SIGTERM', shutdown);

    // ── 自调度定时器（防重入：上一轮完成后才调度下一轮）────
    const timerHandles: ReturnType<typeof setTimeout>[] = [];

    function scheduleRecurring(label: string, intervalMs: number, fn: () => Promise<void>): void {
        const tick = async () => {
            if (isShuttingDown) return;
            try { await fn(); } catch { /* */ }
            if (!isShuttingDown) {
                timerHandles.push(setTimeout(tick, intervalMs));
            }
        };
        timerHandles.push(setTimeout(tick, intervalMs));
    }

    // 结算检查
    scheduleRecurring('settle', SETTLEMENT_CHECK_INTERVAL, async () => {
        await Promise.all(
            traders.map(async (t) => {
                if (t.paused) return;
                try {
                    await t.executor.settleHistory();
                } catch (e) {
                    console.error(`  [${t.config.name}] 结算检查失败:`, e);
                }
            }),
        );
    });

    // LIVE 模式：每 10 秒同步链上 pUSD 余额，避免多进程里下注比例基于陈旧资金
    scheduleRecurring('liveCapitalSync', 10 * 1000, async () => {
        await Promise.all(
            traders.map(async (t) => {
                if (t.paused || t.env.TRADING_MODE !== TradingMode.LIVE) return;
                try { await t.executor.refreshLiveCapital(); } catch { /* isolation */ }
            }),
        );
    });

    // 价格监控 + WS 订阅同步
    scheduleRecurring('priceMonitor', MONITOR_INTERVAL_MS, async () => {
        marketOrderbookWS.syncSubscribe();
        for (const ws of userChannelWSList) ws.syncSubscribe();
        await Promise.all(
            traders.map(async (t) => {
                if (t.paused) return;
                try { await t.executor.checkPriceMonitorQueue(); } catch { /* */ }
            }),
        );
    });

    // 低价监控
    scheduleRecurring('lowPriceMonitor', MONITOR_INTERVAL_MS, async () => {
        await Promise.all(
            traders.map(async (t) => {
                if (t.paused) return;
                try { await t.executor.checkLowPriceMonitorQueue(); } catch { /* */ }
            }),
        );
    });

    scheduleRecurring('staleLiveOrderPolicy', MONITOR_INTERVAL_MS, async () => {
        await Promise.all(
            traders.map(async (t) => {
                if (t.paused || t.env.TRADING_MODE !== TradingMode.LIVE) return;
                try { await t.executor.checkStaleLiveOrderPolicy(); } catch { /* */ }
            }),
        );
    });

    // 流动性监控
    scheduleRecurring('liquidityMonitor', LIQUIDITY_MONITOR_INTERVAL_MS, async () => {
        await Promise.all(
            traders.map(async (t) => {
                if (t.paused) return;
                try { await t.executor.checkLiquidityMonitorQueue(); } catch { /* */ }
            }),
        );
    });

    // 报告生成（首次延迟后循环）
    scheduleRecurring('reports', REPORT_INTERVAL_MS, async () => {
        await Promise.all(
            traders.map(async (t) => {
                try { await t.executor.generateReports(); } catch { /* */ }
            }),
        );
    });

    // ── 预测文件缓存（风险 I 防护：同 suffix 共享读取）─────
    const predictionCache = new Map<string, { data: any; readAt: number }>();
    const CACHE_TTL_MS = 500;

    function readPredictionCached(filePath: string): any {
        const cached = predictionCache.get(filePath);
        if (cached && Date.now() - cached.readAt < CACHE_TTL_MS) {
            return cached.data;
        }
        const raw = readLocalPredictionsRawFromFile(filePath);
        predictionCache.set(filePath, { data: raw, readAt: Date.now() });
        return raw;
    }

    function canUsePredictionFallbackBySourceStale(filePath?: string | null): boolean {
        if (!filePath) return false;
        const mtimeMs = statFileMtimeMs(filePath);
        if (mtimeMs <= 0) return false;
        const ageSec = (Date.now() - mtimeMs) / 1000;
        return ageSec <= SYMBOL_OVERRIDE_STALE_SEC;
    }

    // ── 主扫描循环 ──────────────────────────────────────────
    console.log(`\n🔄 开始扫描循环 (${traders.length} 个 trader, 每 ${SCAN_INTERVAL_MS / 1000} 秒)\n`);

    let lastQualityModeKey = '';
    let lastGuardWriteMs = 0;
    const runtimeConfigReloadState = {
        lastCheckMs: 0,
        lastMtimeMs: statFileMtimeMs(configPath),
    };

    while (!isShuttingDown) {
        const loopNowMs = Date.now();
        const loopNowSec = Math.floor(loopNowMs / 1000);
        maybeReloadRuntimeConfigs(groupArg, configPath, traders, runtimeConfigReloadState, loopNowMs);
        const qualityState = readDataQualityState(loopNowMs);
        const qualityModeKey = `${qualityState.mode}|${qualityState.reason}|${qualityState.confidencePenalty.toFixed(3)}|${qualityState.betScale.toFixed(2)}|${qualityState.apiErrorRate.toFixed(3)}`;
        if (qualityModeKey !== lastQualityModeKey) {
            lastQualityModeKey = qualityModeKey;
            if (qualityState.mode === 'ok') {
                console.log(`  [质量门控] ✅ 数据源正常 | betScale=${qualityState.betScale.toFixed(2)} | apiErr=${qualityState.apiErrorRate.toFixed(3)}`);
            } else if (qualityState.mode === 'soft_degraded') {
                console.warn(
                    `  [质量门控] ⚠️ soft_degraded, penalty=+${qualityState.confidencePenalty.toFixed(3)}, `
                    + `betScale=${qualityState.betScale.toFixed(2)}, apiErr=${qualityState.apiErrorRate.toFixed(3)}, ${qualityState.reason}`,
                );
            } else if (qualityState.mode === 'hard_degraded') {
                console.warn(
                    `  [质量门控] ⚠️ hard_degraded, penalty=+${qualityState.confidencePenalty.toFixed(3)}, `
                    + `betScale=${qualityState.betScale.toFixed(2)}, apiErr=${qualityState.apiErrorRate.toFixed(3)}, ${qualityState.reason}`,
                );
            } else {
                console.error(
                    `  [质量门控] 🛑 halt, betScale=${qualityState.betScale.toFixed(2)}, `
                    + `apiErr=${qualityState.apiErrorRate.toFixed(3)}, ${qualityState.reason}`,
                );
            }
        }

        // 并行处理所有 trader（风险 D 防护：Promise.all 而非串行）
        await Promise.all(
            traders.map(async (t) => {
                if (t.paused || isShuttingDown) return;

                try {
                    t.executor.setRuntimeBetScale(qualityState.betScale);
                    if (t.config.runnerManaged) {
                        t.lastPredictionPathDecision = 'runner_managed_forward_only';
                        return;
                    }
                    if (String(t.config.predictionSourceStatus || '').trim() === 'non_runnable_missing_prediction_source') {
                        if (t.lastPredictionPathDecision !== 'non_runnable_missing_prediction_source') {
                            t.lastPredictionPathDecision = 'non_runnable_missing_prediction_source';
                            console.warn(
                                `  [${t.config.name}] simulation source disabled: non_runnable_missing_prediction_source `
                                + `(${String(t.config.nonRunnableReason || 'missing_prediction_source')})`,
                            );
                        }
                        return;
                    }
                    maybeUpdatePredictionCalibration(t, loopNowMs);
                    maybeUpdateThresholdDrift(t, loopNowMs);
                    maybeUpdateExpectancyGate(t, loopNowMs);
                    maybeUpdateSelectorOverlay(t, loopNowMs);
                    maybeUpdateRegimeGate(t, loopNowMs);
                    maybeUpdateMainlineExtremeRuntime(t, loopNowMs);
                    maybeUpdateShockRisk(t, loopNowMs);
                    maybeUpdateComboPause(t, loopNowMs);
                    maybeUpdateDirectionLossCooldown(t, loopNowMs);
                    maybeUpdateDrawdownAcceleration(t, loopNowMs);
                    maybeUpdateDownRisk(t, loopNowMs);
                    maybeUpdateUpRisk(t, loopNowMs);
                    applyMainlineExtremeRuntimeToExecutor(t);
                    applyDirectionLossRuntimeToExecutor(t);
                    applyDrawdownAccelerationRuntimeToExecutor(t);
                    applyDownRiskRuntimeToExecutor(t);
                    applyUpRiskRuntimeToExecutor(t);
                    applyExpectancyGateRuntimeToExecutor(t);
                    let predictionTarget = resolvePredictionReadTarget(
                        t.predictionFilePath,
                        t.predictionFallbackFilePath,
                    );
                    if (!predictionTarget.filePath) {
                        if (predictionTarget.decision !== t.lastPredictionPathDecision) {
                            t.lastPredictionPathDecision = predictionTarget.decision;
                            console.warn(
                                `  [${t.config.name}] prediction source unavailable due to stale/missing file `
                                + `(decision=${predictionTarget.decision})`,
                            );
                        }
                        return;
                    }
                    let raw = readPredictionCached(predictionTarget.filePath);
                    if (!raw) return;
                    if (predictionTarget.filePath === t.predictionFilePath) {
                        const aliasBasePath = deriveCore70AliasBasePredictionPath(t.predictionFilePath);
                        if (
                            primaryPathLooksLikeAlias(t.predictionFilePath, t.config.name)
                            && aliasBasePath
                            && canUsePredictionFallbackBySourceStale(aliasBasePath)
                        ) {
                            const aliasBaseRaw = readPredictionCached(aliasBasePath);
                            if (shouldFallbackFromCanonicalAliasToBase(raw, aliasBaseRaw)) {
                                predictionTarget = {
                                    filePath: aliasBasePath,
                                    decision: 'fallback_primary_alias_base',
                                };
                                raw = aliasBaseRaw;
                            }
                        }
                    }
                    if (
                        predictionTarget.filePath === t.predictionFilePath
                        && t.predictionFallbackFilePath
                        && raw?.data?.source_stale === true
                        && canUsePredictionFallbackBySourceStale(t.predictionFallbackFilePath)
                    ) {
                        const fallbackRaw = readPredictionCached(t.predictionFallbackFilePath);
                        if (fallbackRaw?.data && fallbackRaw.data.source_stale !== true) {
                            predictionTarget = {
                                filePath: t.predictionFallbackFilePath,
                                decision: 'fallback_primary_source_stale',
                            };
                            raw = fallbackRaw;
                        }
                    }
                    if (predictionTarget.decision !== t.lastPredictionPathDecision) {
                        t.lastPredictionPathDecision = predictionTarget.decision;
                        const selectedPredictionPath = predictionTarget.filePath!;
                        if (predictionTarget.filePath === t.predictionFilePath) {
                            if (isCanonicalFilterEmptyFresh(raw)) {
                                console.log(
                                    `  [${t.config.name}] prediction source=${path.basename(selectedPredictionPath)} `
                                    + `(filtered_empty_due_to_threshold)`,
                                );
                            } else {
                                console.log(`  [${t.config.name}] prediction source=${path.basename(selectedPredictionPath)} (${predictionTarget.decision})`);
                            }
                        } else if (predictionTarget.decision === 'fallback_primary_alias_base') {
                            console.warn(
                                `  [${t.config.name}] primary prediction alias stale/outdated -> fallback ${path.basename(selectedPredictionPath)} `
                                + `(decision=${predictionTarget.decision})`,
                            );
                        } else if (predictionTarget.decision === 'fallback_primary_source_stale') {
                            console.warn(
                                `  [${t.config.name}] primary prediction payload marked source_stale -> fallback ${path.basename(selectedPredictionPath)} `
                                + `(decision=${predictionTarget.decision})`,
                            );
                        } else {
                            console.warn(
                                `  [${t.config.name}] symbol override stale/missing -> fallback ${path.basename(selectedPredictionPath)} `
                                + `(decision=${predictionTarget.decision})`,
                            );
                        }
                    }

                    const ts = raw.target_period_end_ts;
                    const phase = raw.phase;
                    const fileTimestamp = raw.data?.timestamp;
                    if (!ts || !fileTimestamp) return;

                    // 限价单模式（与 prediction_index.ts 逻辑一致）
                    const directionalLadderMode =
                        Object.keys(t.config.lowPriceSelectedBuyPriceRangeBySymbolDirection || {}).length > 0
                        || Object.keys(t.config.lowPriceRungCountBySymbolDirection || {}).length > 0;
                    const isLadderMode = directionalLadderMode || (t.config.limitPriceLadder || '').length > 0;
                    const ladderPrices = isLadderMode
                        ? String(t.config.limitPriceLadder || '').split(',').map((s) => parseFloat(s.trim())).filter((n) => !isNaN(n))
                        : [];
                    const predictionLimitPrice = Number.isFinite(raw.limit_price) ? Number(raw.limit_price) : null;
                    const usePredictionLimitPrice = t.config.usePredictionLimitPrice === true;
                    const limitPrice = (usePredictionLimitPrice && predictionLimitPrice != null)
                        ? predictionLimitPrice
                        : t.env.LIMIT_PRICE;
                    const limitPriceSource = (usePredictionLimitPrice && predictionLimitPrice != null) ? 'prediction' : 'env';
                    const predictionLimitStr = predictionLimitPrice != null ? `$${predictionLimitPrice.toFixed(2)}` : '—';
                    const envLimitStr = Number.isFinite(t.env.LIMIT_PRICE) ? `$${t.env.LIMIT_PRICE.toFixed(2)}` : '—';

                    if (phase === 1 && !t.env.TWO_PHASE_ENABLED) {
                        const decisionMinute = Math.max(0, Number(t.config.decisionMinute || 0));
                        const decisionReadyTs = Number(ts) + decisionMinute * 60;
                        if (decisionMinute > 0 && loopNowSec < decisionReadyTs) return;
                        if (t.executedTargetTss.has(ts)) return;
                        const limitKey = `${ts}_limit_${fileTimestamp}`;
                        if (t.executedPhaseKeys.has(limitKey)) return;
                        if (fileTimestamp === t.lastPredictionTimestampByTs.get(ts)) return;

                        const parsed = parseLocalPredictions(raw.data, t.env.TIMEFRAME, t.env.ALLOWED_MARKETS);
                        const filtered = applyRuntimeGuards(parsed, t, qualityState, loopNowSec, {
                            targetTs: ts,
                            phase,
                            fileTimestamp,
                        });
                        accumulateGuardEvidence(t, parsed.length, filtered);
                        const toOrder = filtered.toOrder;
                        if (filtered.abstainedByMarketSelection > 0) {
                            console.log(`  [${t.config.name}] 弃权 ${filtered.abstainedByMarketSelection} 条: 默认链前置选盘/弃权路由 active`);
                        }
                        if (filtered.skippedShouldTrade > 0) {
                            console.log(`  [${t.config.name}] 跳过 ${filtered.skippedShouldTrade} 条: trade_decision.should_trade=false`);
                        }
                        if (filtered.skippedByQuality > 0) {
                            console.warn(`  [${t.config.name}] 跳过 ${filtered.skippedByQuality} 条: 数据质量门控(${qualityState.mode})`);
                        }
                        if (filtered.skippedByThresholdDrift > 0) {
                            console.warn(`  [${t.config.name}] 跳过 ${filtered.skippedByThresholdDrift} 条: 方向级阈值漂移 active`);
                        }
                        if (filtered.skippedByExpectancyGate > 0) {
                            console.warn(`  [${t.config.name}] 跳过 ${filtered.skippedByExpectancyGate} 条: 近期OOS方向启停 active`);
                        }
                        if (filtered.skippedByMetaLabel > 0) {
                            console.warn(`  [${t.config.name}] 跳过 ${filtered.skippedByMetaLabel} 条: Meta-label gate active`);
                        }
                        if (filtered.skippedBySelector > 0) {
                            console.warn(`  [${t.config.name}] 跳过 ${filtered.skippedBySelector} 条: 组合选择器 overlay active`);
                        }
                        if (filtered.skippedByRegime > 0) {
                            console.warn(`  [${t.config.name}] 跳过 ${filtered.skippedByRegime} 条: regime 门控 active`);
                        }
                        if (filtered.skippedByShockRisk > 0) {
                            console.warn(`  [${t.config.name}] 跳过 ${filtered.skippedByShockRisk} 条: 极端波动风控 active`);
                        }
                        if (filtered.skippedByComboPause > 0) {
                            console.warn(`  [${t.config.name}] 跳过 ${filtered.skippedByComboPause} 条: 组合级停机 active`);
                        }
                        if (filtered.skippedByDownBreaker > 0) {
                            console.warn(`  [${t.config.name}] 跳过 ${filtered.skippedByDownBreaker} 条: DOWN 熔断 active`);
                        }
                        if (filtered.skippedByUpRisk > 0) {
                            console.warn(`  [${t.config.name}] 跳过 ${filtered.skippedByUpRisk} 条: UP 风控 active`);
                        }

                        t.lastPredictionTimestampByTs.set(ts, fileTimestamp);
                        t.executedPhaseKeys.set(limitKey, Date.now());

                        if (toOrder.length === 0) {
                            console.log(
                                `  [${t.config.name}] 最终执行: 无可下单预测 `
                                + `(trade_decision=${filtered.skippedShouldTrade}, quality=${filtered.skippedByQuality}, `
                                + `threshold_drift=${filtered.skippedByThresholdDrift}, expectancy=${filtered.skippedByExpectancyGate}, `
                                + `meta=${filtered.skippedByMetaLabel}, selector=${filtered.skippedBySelector}, regime=${filtered.skippedByRegime}, `
                                + `shock=${filtered.skippedByShockRisk}, combo_pause=${filtered.skippedByComboPause}, `
                                + `down_breaker=${filtered.skippedByDownBreaker}, up_risk=${filtered.skippedByUpRisk})`,
                            );
                            return;
                        }

                        const range = `${new Date(ts * 1000).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}–${new Date((ts + t.executor.periodSeconds) * 1000).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}`;
                        console.log(
                            `  [${t.config.name}] 最终执行通过 ts=${ts} (${range}): ${toOrder.length} 条, `
                            + `limitPrice=$${limitPrice.toFixed(2)} (source=${limitPriceSource}, prediction=${predictionLimitStr}, env=${envLimitStr})`,
                        );
                        for (const p of toOrder) {
                            console.log(`    - ${p.symbol} limitPrice=$${limitPrice.toFixed(2)} (source=${limitPriceSource}) bestAsk=—`);
                        }

                        if (isLadderMode) {
                            console.log(`  [${t.config.name}] 阶梯限价单 ts=${ts} (${range}): ${toOrder.length} 条, ${directionalLadderMode ? '方向梯子主导' : `${ladderPrices.length} 档`}`);
                            const execResults = await t.executor.executeLadderOrder({
                                targetPeriodEndTs: ts,
                                predictions: toOrder,
                                ladderPrices,
                                sizingReferencePrice: Number.isFinite(Number(t.config.lowPriceSourceSizingReferencePrice))
                                    ? Number(t.config.lowPriceSourceSizingReferencePrice)
                                    : undefined,
                                minTotalAmountUsd: Number.isFinite(Number(t.config.lowPriceDynamicMinTotalAmountUsd))
                                    ? Number(t.config.lowPriceDynamicMinTotalAmountUsd)
                                    : undefined,
                            });
                            appendExecutionOutcomeFunnelLog(t, ts, phase, fileTimestamp, toOrder, execResults);
                            if (shouldConsumeTargetTs(execResults)) {
                                t.executedTargetTss.add(ts);
                                saveExecutedTargetTss(t.config.logsDir, t.executedTargetTss);
                            } else {
                                console.warn(`  [${t.config.name}] ts=${ts} 本轮未成交/未挂单，不标记已执行，等待下一次预测更新时间重试`);
                            }
                        } else {
                            if (!Number.isFinite(limitPrice)) {
                                console.warn(`  [${t.config.name}] 跳过 ts=${ts}: compare-only 预测未提供有效 limit_price`);
                                return;
                            }
                            console.log(
                                `  [${t.config.name}] 单限价单 ts=${ts} (${range}): ${toOrder.length} 条, `
                                + `@$${limitPrice.toFixed(2)} (source=${limitPriceSource}, prediction=${predictionLimitStr}, env=${envLimitStr})`,
                            );
                            const execResults = await t.executor.executeLimitOrder({ targetPeriodEndTs: ts, predictions: toOrder, limitPrice });
                            appendExecutionOutcomeFunnelLog(t, ts, phase, fileTimestamp, toOrder, execResults);
                            if (shouldConsumeTargetTs(execResults)) {
                                t.executedTargetTss.add(ts);
                                saveExecutedTargetTss(t.config.logsDir, t.executedTargetTss);
                            } else {
                                console.warn(`  [${t.config.name}] ts=${ts} 本轮未成交/未挂单，不标记已执行，等待下一次预测更新时间重试`);
                            }
                        }
                    } else if (phase === 0 || phase == null) {
                        const decisionMinute = Math.max(0, Number(t.config.decisionMinute || 0));
                        const decisionReadyTs = Number(ts) + decisionMinute * 60;
                        if (decisionMinute > 0 && loopNowSec < decisionReadyTs) return;
                        if (t.executedTargetTss.has(ts)) return;
                        if (fileTimestamp === t.lastPredictionTimestampByTs.get(ts)) return;

                        t.lastPredictionTimestampByTs.set(ts, fileTimestamp);

                        const parsed = parseLocalPredictions(raw.data, t.env.TIMEFRAME, t.env.ALLOWED_MARKETS);
                        const filtered = applyRuntimeGuards(parsed, t, qualityState, loopNowSec, {
                            targetTs: ts,
                            phase,
                            fileTimestamp,
                        });
                        accumulateGuardEvidence(t, parsed.length, filtered);
                        const toOrder = filtered.toOrder;
                        if (filtered.abstainedByMarketSelection > 0) {
                            console.log(`  [${t.config.name}] 弃权 ${filtered.abstainedByMarketSelection} 条: 默认链前置选盘/弃权路由 active`);
                        }
                        if (filtered.skippedShouldTrade > 0) {
                            console.log(`  [${t.config.name}] 跳过 ${filtered.skippedShouldTrade} 条: trade_decision.should_trade=false`);
                        }
                        if (filtered.skippedByQuality > 0) {
                            console.warn(`  [${t.config.name}] 跳过 ${filtered.skippedByQuality} 条: 数据质量门控(${qualityState.mode})`);
                        }
                        if (filtered.skippedByThresholdDrift > 0) {
                            console.warn(`  [${t.config.name}] 跳过 ${filtered.skippedByThresholdDrift} 条: 方向级阈值漂移 active`);
                        }
                        if (filtered.skippedByExpectancyGate > 0) {
                            console.warn(`  [${t.config.name}] 跳过 ${filtered.skippedByExpectancyGate} 条: 近期OOS方向启停 active`);
                        }
                        if (filtered.skippedByMetaLabel > 0) {
                            console.warn(`  [${t.config.name}] 跳过 ${filtered.skippedByMetaLabel} 条: Meta-label gate active`);
                        }
                        if (filtered.skippedBySelector > 0) {
                            console.warn(`  [${t.config.name}] 跳过 ${filtered.skippedBySelector} 条: 组合选择器 overlay active`);
                        }
                        if (filtered.skippedByRegime > 0) {
                            console.warn(`  [${t.config.name}] 跳过 ${filtered.skippedByRegime} 条: regime 门控 active`);
                        }
                        if (filtered.skippedByShockRisk > 0) {
                            console.warn(`  [${t.config.name}] 跳过 ${filtered.skippedByShockRisk} 条: 极端波动风控 active`);
                        }
                        if (filtered.skippedByComboPause > 0) {
                            console.warn(`  [${t.config.name}] 跳过 ${filtered.skippedByComboPause} 条: 组合级停机 active`);
                        }
                        if (filtered.skippedByDownBreaker > 0) {
                            console.warn(`  [${t.config.name}] 跳过 ${filtered.skippedByDownBreaker} 条: DOWN 熔断 active`);
                        }
                        if (filtered.skippedByUpRisk > 0) {
                            console.warn(`  [${t.config.name}] 跳过 ${filtered.skippedByUpRisk} 条: UP 风控 active`);
                        }

                        if (toOrder.length === 0) {
                            console.log(
                                `  [${t.config.name}] 最终执行: 无可下单预测 `
                                + `(trade_decision=${filtered.skippedShouldTrade}, quality=${filtered.skippedByQuality}, `
                                + `threshold_drift=${filtered.skippedByThresholdDrift}, expectancy=${filtered.skippedByExpectancyGate}, `
                                + `meta=${filtered.skippedByMetaLabel}, selector=${filtered.skippedBySelector}, regime=${filtered.skippedByRegime}, `
                                + `shock=${filtered.skippedByShockRisk}, combo_pause=${filtered.skippedByComboPause}, `
                                + `down_breaker=${filtered.skippedByDownBreaker}, up_risk=${filtered.skippedByUpRisk})`,
                            );
                            return;
                        }

                        const useLimitOrderMode = t.config.limitPrice != null || t.config.usePredictionLimitPrice === true;
                        const range = `${new Date(ts * 1000).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}–${new Date((ts + t.executor.periodSeconds) * 1000).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}`;
                        const limitPriceStr = useLimitOrderMode && Number.isFinite(limitPrice)
                            ? `limitPrice=$${limitPrice.toFixed(2)} (source=${limitPriceSource}, prediction=${predictionLimitStr}, env=${envLimitStr})`
                            : 'limitPrice=—';
                        console.log(`  [${t.config.name}] 最终执行通过 ts=${ts} (${range}): ${toOrder.length} 条, ${limitPriceStr}`);
                        for (const p of toOrder) {
                            console.log(
                                `    - ${p.symbol} limitPrice=$${useLimitOrderMode && Number.isFinite(limitPrice) ? limitPrice.toFixed(2) : '—'} `
                                + `(${useLimitOrderMode && Number.isFinite(limitPrice) ? `source=${limitPriceSource}` : 'source=—'}) bestAsk=—`,
                            );
                        }
                        if (useLimitOrderMode) {
                            if (!Number.isFinite(limitPrice)) {
                                console.warn(`  [${t.config.name}] 跳过 ts=${ts}: compare-only 预测未提供有效 limit_price`);
                                return;
                            }
                            const execResults = await t.executor.executeLimitOrder({ targetPeriodEndTs: ts, predictions: toOrder, limitPrice });
                            appendExecutionOutcomeFunnelLog(t, ts, phase, fileTimestamp, toOrder, execResults);
                            if (shouldConsumeTargetTs(execResults)) {
                                t.executedTargetTss.add(ts);
                                saveExecutedTargetTss(t.config.logsDir, t.executedTargetTss);
                            } else {
                                console.warn(`  [${t.config.name}] ts=${ts} 本轮未成交/未挂单，不标记已执行，等待下一次预测更新时间重试`);
                            }
                        } else {
                            const execResults = await t.executor.executeRound({ targetPeriodEndTs: ts, predictions: toOrder });
                            appendExecutionOutcomeFunnelLog(t, ts, phase, fileTimestamp, toOrder, execResults);
                            if (shouldConsumeTargetTs(execResults)) {
                                t.executedTargetTss.add(ts);
                                saveExecutedTargetTss(t.config.logsDir, t.executedTargetTss);
                            } else {
                                console.warn(`  [${t.config.name}] ts=${ts} 本轮未成交/未挂单，不标记已执行，等待下一次预测更新时间重试`);
                            }
                        }
                    } else if (t.env.TWO_PHASE_ENABLED) {
                        // S6 fix: TWO_PHASE 模式下的 phase 1/2 警告
                        console.warn(`  [${t.config.name}] TWO_PHASE_ENABLED=true 但 multi 模式暂不支持两阶段执行 (phase=${phase}, ts=${ts})`);
                    }

                    t.consecutiveErrors = 0;
                } catch (err) {
                    t.consecutiveErrors++;
                    console.error(`  [${t.config.name}] 扫描异常 (${t.consecutiveErrors}/${MAX_CONSECUTIVE_ERRORS}):`, err);
                    if (t.consecutiveErrors >= MAX_CONSECUTIVE_ERRORS) {
                        console.error(`  ⚠️ [${t.config.name}] 连续 ${MAX_CONSECUTIVE_ERRORS} 次异常，暂停该 trader`);
                        t.paused = true;
                    }
                }
            }),
        );

        // M1/M2: 定期清理过期的 phaseKeys 和 timestamp 缓存（保留近 2 小时）
        const cleanupCutoff = Date.now() - 2 * 3600 * 1000;
        const tsCutoff = Math.floor(Date.now() / 1000) - 2 * 3600;
        for (const t of traders) {
            for (const [key, createdAt] of t.executedPhaseKeys) {
                if (createdAt < cleanupCutoff) t.executedPhaseKeys.delete(key);
            }
            for (const [tsKey] of t.lastPredictionTimestampByTs) {
                if (tsKey < tsCutoff) t.lastPredictionTimestampByTs.delete(tsKey);
            }
        }

        if (loopNowMs - lastGuardWriteMs >= 15000) {
            lastGuardWriteMs = loopNowMs;
            writeRuntimeGuardStatus(groupArg, traders, qualityState);
        }

        await sleep(SCAN_INTERVAL_MS);
    }
}

// ============================================================
// 运行
// ============================================================

main().catch((error) => {
    console.error('\n❌ 致命错误:', error);
    process.exit(1);
});
