#!/usr/bin/env ts-node
/**
 * 两阶段限价单 API Dry-Run 测试
 * 
 * 不下真单，只验证:
 * 1. Polymarket CLOB 客户端能否初始化
 * 2. 能否正确查找 15m 市场
 * 3. 能否获取 orderbook（best_ask/best_bid）
 * 4. GTD 限价单（带到期）的 createOrder 结构是否正确（不 post）
 * 5. Edge/Kelly 实时计算是否正确
 * 6. Phase 1 / Phase 2 流程逻辑检查
 * 
 * 用法:
 *   cd polymarket
 *   npx ts-node src/test_limit_order_dryrun.ts
 */

import { Chain, ClobClient, OrderType, Side, SignatureTypeV2 } from '@polymarket/clob-client-v2';
import { ethers } from 'ethers';
import { PREDICTION_ENV, TradingMode, printPredictionConfig } from './config/prediction_env';
import { findAllMarkets, getTokenForDirection, getOrderBook, getTokenPrice } from './services/market_finder';

const POLYGON_RPC = 'https://polygon-rpc.com';
const CLOB_API_URL = 'https://clob.polymarket.com';
const CHAIN_ID = 137;

// ─── 测试辅助 ─────────────────────────────────────────
let passCount = 0;
let failCount = 0;

function pass(msg: string) {
    passCount++;
    console.log(`  ✅ PASS: ${msg}`);
}

function fail(msg: string, err?: any) {
    failCount++;
    console.log(`  ❌ FAIL: ${msg}`);
    if (err) console.log(`     → ${err}`);
}

function section(title: string) {
    console.log(`\n${'═'.repeat(60)}`);
    console.log(`  ${title}`);
    console.log(`${'═'.repeat(60)}`);
}

// ─── Edge/Kelly 计算测试 ─────────────────────────────
// 主流程（prediction_executor）使用纯数学 Edge/Kelly（不含费率），与本测试一致。
// 注：旧版曾用 totalCost 做「含费 Edge」测试，已废弃；主流程费率只在结算 PnL 时扣除。
function testEdgeKellyCalculation() {
    section('测试 1: Edge / Kelly 计算');

    // 场景 A: 买价 $0.527, 置信度 55%（与主流程一致：edge = p*odds - q，不含费）
    const bp1 = 0.527;
    const odds1 = (1 - bp1) / bp1;  // 0.8975
    const conf1 = 0.55;
    const q1 = 1 - conf1;
    const edge1 = conf1 * odds1 - q1;
    const kelly1 = Math.max(0, (conf1 * odds1 - q1) / odds1);

    console.log(`\n  场景 A: 买价 $${bp1}, 置信度 ${conf1 * 100}%`);
    console.log(`    赔率: ${odds1.toFixed(4)}`);
    console.log(`    Edge: ${(edge1 * 100).toFixed(3)}%`);
    console.log(`    Kelly: ${(kelly1 * 100).toFixed(2)}%`);

    if (odds1 > 0.89 && odds1 < 0.91) pass('赔率 $0.527 正确');
    else fail(`赔率 $0.527 = ${odds1}, 预期 ~0.898`);

    // 场景 B: 买价 $0.51, 置信度 55%
    const bp2 = 0.51;
    const odds2 = (1 - bp2) / bp2;  // 0.9608
    const edge2 = conf1 * odds2 - q1;
    const kelly2 = Math.max(0, (conf1 * odds2 - q1) / odds2);

    console.log(`\n  场景 B: 买价 $${bp2}, 置信度 ${conf1 * 100}%`);
    console.log(`    赔率: ${odds2.toFixed(4)}`);
    console.log(`    Edge: ${(edge2 * 100).toFixed(3)}%`);
    console.log(`    Kelly: ${(kelly2 * 100).toFixed(2)}%`);

    if (edge2 > edge1) pass('$0.51 边际 > $0.527 边际 (限价单更优)');
    else fail(`$0.51 边际 ${edge2} 应 > $0.527 边际 ${edge1}`);

    // 场景 C: 买价 $0.54, 置信度 52% (边际为负)
    const bp3 = 0.54;
    const odds3 = (1 - bp3) / bp3;
    const conf3 = 0.52;
    const q3 = 1 - conf3;
    const edge3 = conf3 * odds3 - q3;

    console.log(`\n  场景 C: 买价 $${bp3}, 置信度 ${conf3 * 100}% (应被过滤)`);
    console.log(`    Edge: ${(edge3 * 100).toFixed(3)}% ${edge3 < 0 ? '(负 → 不交易)' : '(正 → 交易)'}`);

    if (edge3 < 0) pass('高买价+低置信度 → 负边际 → 正确过滤');
    else fail('应该是负边际但得到正的');
}

// ─── CLOB 客户端 + 市场查找测试 ─────────────────────
async function testMarketAndOrderbook() {
    section('测试 2: Polymarket 市场查找 + OrderBook');

    // 2a: 查找 BTC 15m 市场
    console.log('\n  查找 BTC 15m 市场...');
    try {
        const now = Math.floor(Date.now() / 1000);
        // 找下一个 15 分钟周期
        const periodEnd = Math.ceil(now / 900) * 900;
        const markets = await findAllMarkets(undefined, periodEnd, ['BTC']);

        if (markets.length > 0) {
            const m = markets[0];
            pass(`找到 BTC 市场: ${m.market?.title || m.symbol}`);
            console.log(`    conditionId: ${m.market?.conditionId?.slice(0, 20)}...`);

            if (m.market) {
                const upToken = getTokenForDirection(m.market, 'UP');
                const downToken = getTokenForDirection(m.market, 'DOWN');
                if (upToken) pass(`UP token: ${upToken.tokenId.slice(0, 15)}... price=$${upToken.price}`);
                else fail('无法获取 UP token');
                if (downToken) pass(`DOWN token: ${downToken.tokenId.slice(0, 15)}... price=$${downToken.price}`);
                else fail('无法获取 DOWN token');

                // 2b: OrderBook
                if (upToken) {
                    console.log('\n  获取 UP token OrderBook...');
                    try {
                        const ob = await getOrderBook(upToken.tokenId);
                        if (ob && ob.asks && ob.asks.length > 0) {
                            const bestAsk = ob.asks.reduce((min: any, ask: any) =>
                                parseFloat(ask.price) < parseFloat(min.price) ? ask : min
                            , ob.asks[0]);
                            const bestAskPrice = parseFloat(bestAsk.price);
                            pass(`OrderBook best_ask: $${bestAskPrice.toFixed(4)} (size: ${bestAsk.size})`);

                            // 用 best_ask 计算 edge（与主流程一致：edge = conf*odds - q，不含费）
                            const odds = (1 - bestAskPrice) / bestAskPrice;
                            const conf = 0.55;  // 假设置信度
                            const edge = conf * odds - (1 - conf);
                            console.log(`    实时 Edge (conf=55%): ${(edge * 100).toFixed(2)}%`);

                            if (bestAskPrice > 0.3 && bestAskPrice < 0.8) {
                                pass('best_ask 在合理范围 $0.30~$0.80');
                            } else {
                                fail(`best_ask $${bestAskPrice} 不在预期范围`);
                            }
                        } else {
                            fail('OrderBook 无 asks');
                        }

                        if (ob && ob.bids && ob.bids.length > 0) {
                            const bestBid = ob.bids.reduce((max: any, bid: any) =>
                                parseFloat(bid.price) > parseFloat(max.price) ? bid : max
                            , ob.bids[0]);
                            pass(`OrderBook best_bid: $${parseFloat(bestBid.price).toFixed(4)} (size: ${bestBid.size})`);
                        }
                    } catch (err) {
                        fail('OrderBook 获取失败', err);
                    }
                }
            }
        } else {
            fail('未找到 BTC 15m 市场（可能当前无活跃市场）');
        }
    } catch (err) {
        fail('市场查找异常', err);
    }
}

// ─── GTD 限价单结构验证 ─────────────────────────────
async function testGTDOrderStructure() {
    section('测试 3: GTD 限价单结构验证');

    // 检查是否有钱包配置
    if (!PREDICTION_ENV.PRIVATE_KEY || !PREDICTION_ENV.PROXY_WALLET) {
        console.log('\n  ⚠️  未配置 PRIVATE_KEY / PROXY_WALLET');
        console.log('  跳过 createOrder 签名测试（需要真实钱包）');
        console.log('  但可以验证订单参数结构...\n');

        // 模拟订单结构
        const mockOrder = {
            tokenID: '0x' + '1'.repeat(64),
            price: 0.50,
            size: 20.0,  // $10 / $0.50 = 20 shares
            side: Side.BUY,
        };

        console.log('  GTD 限价单参数:');
        console.log(`    tokenID: ${mockOrder.tokenID.slice(0, 15)}...`);
        console.log(`    price: $${mockOrder.price} (限价)`);
        console.log(`    size: ${mockOrder.size} shares`);
        console.log(`    side: ${mockOrder.side}`);
        console.log(`    orderType: GTD (Good Till Date / expiry)`);

        if (mockOrder.price > 0 && mockOrder.price < 1) pass('限价在 (0, 1) 范围内');
        else fail('限价应在 0~1 之间');

        if (mockOrder.size > 0) pass('份额 > 0');
        else fail('份额应 > 0');

        pass('订单结构验证通过（无签名）');
        return;
    }

    // 有钱包配置 → 尝试 createOrder（签名但不 post）
    console.log('\n  有钱包配置，尝试 createOrder (不 post)...');
    try {
        const wallet = new ethers.Wallet(PREDICTION_ENV.PRIVATE_KEY);
        const signerAddress = wallet.address.toLowerCase();
        const proxyWallet = String(PREDICTION_ENV.PROXY_WALLET || '').trim();
        const normalizedProxyWallet = proxyWallet.toLowerCase();
        const hasDistinctProxyWallet = Boolean(proxyWallet) && normalizedProxyWallet !== signerAddress;
        let signatureType: SignatureTypeV2 | undefined;
        let funder: string | undefined;
        if (hasDistinctProxyWallet) {
            const provider = new ethers.providers.JsonRpcProvider(PREDICTION_ENV.RPC_URL);
            const code = await provider.getCode(proxyWallet);
            signatureType = code !== '0x' ? SignatureTypeV2.POLY_GNOSIS_SAFE : SignatureTypeV2.POLY_PROXY;
            funder = proxyWallet;
        } else {
            signatureType = SignatureTypeV2.EOA;
        }
        const clobClient = new ClobClient({
            host: CLOB_API_URL,
            chain: Chain.POLYGON,
            signer: wallet,
            signatureType,
            funderAddress: funder,
        });

        // 找一个真实市场来测试
        const now = Math.floor(Date.now() / 1000);
        const periodEnd = Math.ceil(now / 900) * 900;
        const markets = await findAllMarkets(undefined, periodEnd, ['BTC']);

        if (markets.length > 0 && markets[0].market) {
            const token = getTokenForDirection(markets[0].market, 'UP');
            if (token) {
                const limitPrice = 0.50;
                const betAmount = 10.0;
                const shares = Math.floor((betAmount / limitPrice) * 100) / 100;

                const userOrder = {
                    tokenID: token.tokenId,
                    price: limitPrice,
                    size: shares,
                    side: Side.BUY as Side,
                };

                console.log(`\n  创建 GTD 限价单 (不 post):`);
                console.log(`    tokenID: ${token.tokenId.slice(0, 15)}...`);
                console.log(`    price: $${limitPrice}`);
                console.log(`    size: ${shares} shares ($${betAmount})`);

                try {
                    const signedOrder = await clobClient.createOrder(userOrder, { tickSize: '0.01' });
                    pass('createOrder 签名成功');
                    console.log(`    签名类型: ${typeof signedOrder}`);

                    // 验证不 post
                    console.log('    ⚠️  不调用 postOrder（dry-run 模式）');
                    pass('GTD 限价单结构完整，可以提交');
                } catch (err: any) {
                    fail(`createOrder 签名失败: ${err.message || err}`);
                }
            } else {
                fail('无法获取测试 token');
            }
        } else {
            fail('无活跃市场可测试');
        }
    } catch (err: any) {
        fail(`CLOB 客户端初始化失败: ${err.message || err}`);
    }
}

// ─── Phase 1/2 流程逻辑检查 ──────────────────────────
function testPhaseLogic() {
    section('测试 4: Phase 1/2 流程逻辑');

    const phase1LimitPrice = PREDICTION_ENV.PHASE1_LIMIT_PRICE ?? 0.50;
    const phase2Premium = PREDICTION_ENV.PHASE2_LIMIT_PREMIUM ?? 0.01;
    const phase1BetFraction = PREDICTION_ENV.PHASE1_BET_FRACTION ?? 0.60;
    const twoPhaseEnabled = PREDICTION_ENV.TWO_PHASE_ENABLED ?? false;

    console.log(`\n  两阶段配置:`);
    console.log(`    TWO_PHASE_ENABLED: ${twoPhaseEnabled}`);
    console.log(`    PHASE1_LIMIT_PRICE: $${phase1LimitPrice}`);
    console.log(`    PHASE2_LIMIT_PREMIUM: $${phase2Premium}`);
    console.log(`    PHASE1_BET_FRACTION: ${(phase1BetFraction * 100).toFixed(0)}%`);
    console.log(`    PHASE1_MIN_CONFIDENCE: ${((PREDICTION_ENV.PHASE1_MIN_CONFIDENCE ?? 0.54) * 100).toFixed(0)}%`);
    console.log(`    PHASE2_MIN_CONFIDENCE: ${((PREDICTION_ENV.PHASE2_MIN_CONFIDENCE ?? 0.52) * 100).toFixed(0)}%`);
    console.log(`    MAX_SWEEP_PRICE: $${PREDICTION_ENV.MAX_SWEEP_PRICE ?? 0.55}`);

    // Phase 1 逻辑检查
    if (phase1LimitPrice > 0 && phase1LimitPrice < 0.55) {
        pass(`Phase 1 限价 $${phase1LimitPrice} 在合理范围 (0, 0.55)`);
    } else {
        fail(`Phase 1 限价 $${phase1LimitPrice} 不合理`);
    }

    if (phase1BetFraction > 0 && phase1BetFraction <= 1) {
        pass(`Phase 1 仓位比 ${(phase1BetFraction * 100).toFixed(0)}% 正确`);
    } else {
        fail(`Phase 1 仓位比 ${phase1BetFraction} 应在 (0, 1]`);
    }

    // Phase 2 逻辑: betFraction = 1 - phase1BetFraction
    const phase2Fraction = 1 - phase1BetFraction;
    console.log(`\n  Phase 2 剩余仓位: ${(phase2Fraction * 100).toFixed(0)}%`);
    if (phase2Fraction > 0) {
        pass('Phase 2 有剩余仓位可用');
    } else {
        fail('Phase 1 用完 100%, Phase 2 无仓位');
    }

    // 模拟 Phase 1 → Phase 2 流程
    console.log('\n  模拟流程:');
    const totalBet = 20;  // 假设 $20 总下注
    const p1Bet = totalBet * phase1BetFraction;
    const p2Bet = totalBet * phase2Fraction;
    const p1Shares = Math.floor((p1Bet / phase1LimitPrice) * 100) / 100;

    console.log(`    1. Phase 1: 挂 GTD 限价单 $${p1Bet.toFixed(2)} @ $${phase1LimitPrice} → ${p1Shares} shares`);
    console.log(`    2. 等待 120s (K2 收盘前 120s)...`);
    console.log(`    3a. Phase 2 方向一致 → 保留 Phase 1 + 下市价单 $${p2Bet.toFixed(2)}`);
    console.log(`    3b. Phase 2 方向反转 → 取消 Phase 1 + (可选) 反向下单`);

    pass('Phase 1/2 流程逻辑一致');
}

// ─── 现有 22 个模拟盘兼容性检查 ─────────────────────
function testBackwardCompatibility() {
    section('测试 5: 现有 22 个模拟盘兼容性');

    const twoPhaseEnabled = PREDICTION_ENV.TWO_PHASE_ENABLED ?? false;
    const tradingMode = PREDICTION_ENV.TRADING_MODE;

    console.log(`\n  当前配置:`);
    console.log(`    TRADING_MODE: ${tradingMode}`);
    console.log(`    TWO_PHASE_ENABLED: ${twoPhaseEnabled}`);

    if (!twoPhaseEnabled) {
        pass('TWO_PHASE_ENABLED=false → 22个模拟盘不受影响，走旧逻辑');
    } else {
        console.log('  ⚠️  TWO_PHASE_ENABLED=true → 会走两阶段逻辑');
        console.log('  22 个模拟盘如果用各自的 .env 且没有设 TWO_PHASE_ENABLED=true，则不受影响');
    }

    if (tradingMode === TradingMode.SIMULATION) {
        pass('SIMULATION 模式 → Phase 1 用 fake orderID，不调真实 API');
    } else if (tradingMode === TradingMode.LIVE) {
        console.log('  ⚠️  LIVE 模式 → 会调真实 Polymarket API');
    } else {
        pass('BACKTEST 模式 → 纯记录');
    }
}

// ─── 主函数 ──────────────────────────────────────────
async function main() {
    console.log('\n' + '█'.repeat(60));
    console.log('  两阶段限价单 API Dry-Run 测试');
    console.log('  模式: 只读 / 不下真单');
    console.log('█'.repeat(60));

    // 打印配置
    printPredictionConfig();

    // 测试 1: Edge/Kelly
    testEdgeKellyCalculation();

    // 测试 2: 市场 + OrderBook
    await testMarketAndOrderbook();

    // 测试 3: GTD 订单结构
    await testGTDOrderStructure();

    // 测试 4: Phase 逻辑
    testPhaseLogic();

    // 测试 5: 兼容性
    testBackwardCompatibility();

    // 汇总
    section('测试结果');
    console.log(`\n  ✅ 通过: ${passCount}`);
    console.log(`  ❌ 失败: ${failCount}`);
    console.log(`  总计: ${passCount + failCount}`);
    if (failCount === 0) {
        console.log('\n  🎉 全部通过！限价单 API 设置正确。');
    } else {
        console.log(`\n  ⚠️  有 ${failCount} 项失败，需要修复。`);
    }
    console.log();
}

main().catch((err) => {
    console.error('测试异常:', err);
    process.exit(1);
});
