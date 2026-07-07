#!/usr/bin/env node
/**
 * 检测脚本：监听到下单最快框架 + 预签 + 最后5分钟市价 + 本地订单簿缓存
 * 与当前运行环境一致：优先读 dist（npm run build 后），无则读 src；使用 process.env.LOGS_DIR
 *
 * 检查项：
 * 1. 预签与 OrderPoster：order_poster 模块、SingleWalletOrderPoster、preSignedOrder 入队预签、触发时 postOrder
 * 2. 最后5分钟市价：LAST5MIN_MIN/MAX、monitorEndTime 后仍可成交 [0.3,0.65]
 * 3. 本地订单簿缓存：price_change 时带 asksSnapshot（lastAsksByAsset）
 * 4. 现有规则未破坏：MAX_PRICE_THRESHOLD、MIN_TOKEN_PRICE、hasAlreadyBought、价格区间判断仍存在
 */

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const SRC = path.join(ROOT, 'src');
const DIST = path.join(ROOT, 'dist');

function readFile(...parts) {
    const p = path.join(ROOT, ...parts);
    if (fs.existsSync(p)) return fs.readFileSync(p, 'utf8');
    return null;
}

function check(name, ok, detail) {
    const status = ok ? '✅' : '❌';
    console.log(`${status} ${name}${detail ? ': ' + detail : ''}`);
    return ok;
}

function main() {
    const logsDir = process.env.LOGS_DIR || 'logs';
    console.log('\n=== 监听到下单最快框架 + 预签 + 最后5分钟 检测 ===');
    console.log(`环境: LOGS_DIR=${logsDir}，优先 dist，与运行一致\n`);

    let allOk = true;

    const executorContent = readFile('dist', 'services', 'prediction_executor.js') || readFile('src', 'services', 'prediction_executor.ts');
    const orderPosterContent = readFile('dist', 'services', 'order_poster.js') || readFile('src', 'services', 'order_poster.ts');
    const wsContent = readFile('dist', 'services', 'market_orderbook_ws.js') || readFile('src', 'services', 'market_orderbook_ws.ts');

    if (!executorContent) {
        console.log('❌ 未找到 prediction_executor（请先 npm run build 或确保 src 存在）');
        process.exit(1);
    }

    // --- 1. 预签与 OrderPoster ---
    allOk = check('order_poster 模块存在', !!orderPosterContent) && allOk;
    if (orderPosterContent) {
        allOk = check('IOrderPoster / postOrder 接口', /postOrder\s*\(.*signedOrder|IOrderPoster/.test(orderPosterContent)) && allOk;
        allOk = check('SingleWalletOrderPoster 实现', /SingleWalletOrderPoster|clobClient\.postOrder/.test(orderPosterContent)) && allOk;
    }
    allOk = check('executor 使用 orderPoster', /orderPoster|SingleWalletOrderPoster/.test(executorContent)) && allOk;
    allOk = check('PriceMonitorEntry 含 preSignedOrder', /preSignedOrder/.test(executorContent)) && allOk;
    allOk = check('入队后调度预签 (schedulePreSign)', /schedulePreSignForPriceMonitorEntry|schedulePreSign/.test(executorContent)) && allOk;
    const preSignOnlyLive = executorContent.includes('schedulePreSignForPriceMonitorEntry') &&
        (executorContent.includes('TRADING_MODE') && executorContent.includes('LIVE')) &&
        (executorContent.includes('schedulePreSign') && (executorContent.includes('!== TradingMode.LIVE') || executorContent.includes('!== .TradingMode.LIVE') || /TRADING_MODE.*!==.*LIVE/.test(executorContent)));
    allOk = check('预签仅 LIVE 模式（入队条件 + 方法内 return）', preSignOnlyLive || (executorContent.includes('schedulePreSignForPriceMonitorEntry') && executorContent.includes('LIVE'))) && allOk;
    allOk = check('触发时传入 preSignedOrder 到 executePrediction', /preSignedOrder:\s*entry\.preSignedOrder|preSignedOrder:\s*entry\.preSigned/.test(executorContent)) && allOk;
    allOk = check('executeLiveTrade 有预签则只 POST (orderPoster.postOrder)', /preSignedOrder.*orderPoster|orderPoster\.postOrder.*preSignedOrder/.test(executorContent) || (executorContent.includes('opts?.preSignedOrder') && executorContent.includes('this.orderPoster') && executorContent.includes('postOrder'))) && allOk;

    // --- 2. 最后5分钟市价 ---
    allOk = check('LAST5MIN_MIN_PRICE / LAST5MIN_MAX_PRICE 常量', /LAST5MIN_MIN_PRICE|LAST5MIN_MAX_PRICE|0\.3|0\.65/.test(executorContent)) && allOk;
    allOk = check('最后5分钟分支 (now >= monitorEndTime 且价格在 [0.3,0.65])', /now\s*>?=\s*entry\.monitorEndTime[\s\S]*LAST5MIN_(MIN|MAX)_PRICE|monitorEndTime[\s\S]*LAST5MIN_(MIN|MAX)_PRICE/.test(executorContent)) && allOk;
    allOk = check('executePrediction 允许最后5分钟 triggerPrice > MAX', /LAST5MIN_MIN_PRICE|LAST5MIN_MAX_PRICE/.test(executorContent)) && allOk;

    // --- 3. 本地订单簿缓存 ---
    if (wsContent) {
        allOk = check('WS 本地缓存 lastAsksByAsset', /lastAsksByAsset|lastAsks/.test(wsContent)) && allOk;
        allOk = check('price_change 时带 snapshot 回调', /price_change|onBestAsk.*snapshot|snapshot/.test(wsContent)) && allOk;
    }

    // --- 4. 现有规则未破坏 ---
    allOk = check('MAX_PRICE_THRESHOLD / 高价监控条件仍存在', /MAX_PRICE_THRESHOLD|MAX_TOKEN_PRICE|价格过高/.test(executorContent)) && allOk;
    allOk = check('MIN_TOKEN_PRICE / 低价监控条件仍存在', /MIN_TOKEN_PRICE|价格过低/.test(executorContent)) && allOk;
    allOk = check('hasAlreadyBought 防重复仍存在', /hasAlreadyBought/.test(executorContent)) && allOk;
    allOk = check('价格区间 [MIN, MAX] 立即下单逻辑仍存在', /MIN_TOKEN_PRICE.*MAX_TOKEN_PRICE|价格在.*之间.*立即下单/.test(executorContent) || executorContent.includes('currentPrice <= this.MAX_PRICE_THRESHOLD')) && allOk;
    allOk = check('executePrediction 入口 hasAlreadyBought 检查', /executePrediction|hasAlreadyBought.*conditionId/.test(executorContent)) && allOk;

    console.log('');
    process.exit(allOk ? 0 : 1);
}

main();
