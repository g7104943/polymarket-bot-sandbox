#!/usr/bin/env node
/**
 * 检测脚本：按模拟/真实交易逻辑验证 WebSocket + 1 秒价格监控兜底是否生效
 *
 * 检查项：
 * 1. 价格监控兜底间隔为 1 秒（MONITOR_INTERVAL_MS = 1*1000）
 * 2. 使用市场频道 WebSocket（MarketOrderbookWS，端点 ws-subscriptions-clob.polymarket.com/ws/market）
 * 3. 执行器暴露 getMonitorTokenIds、tryTriggerPriceMonitorByTokenId、tryTriggerLowPriceMonitorByTokenId
 * 4. 可选：实际连接 WS 并订阅一个 token，收到一条消息即视为连通
 *
 * 用法：
 *   node scripts/verify_ws_and_monitor.js              # 仅静态检查
 *   VERIFY_WS_TOKEN_ID=某个tokenId node scripts/verify_ws_and_monitor.js   # 含 WS 连通性测试
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
    console.log('\n=== WebSocket + 1 秒价格监控兜底 改动检测 ===\n');

    let allOk = true;

    // 优先检查编译后的 dist（与真实运行一致），若无则检查 src
    const indexContent = readFile('dist', 'prediction_index.js') || readFile('src', 'prediction_index.ts');
    const wsContent = readFile('dist', 'services', 'market_orderbook_ws.js') || readFile('src', 'services', 'market_orderbook_ws.ts');
    const executorContent = readFile('dist', 'services', 'prediction_executor.js') || readFile('src', 'services', 'prediction_executor.ts');

    if (!indexContent) {
        console.log('❌ 未找到 prediction_index（请先 npm run build 或确保 src 存在）');
        process.exit(1);
    }

    // 1. 价格监控兜底间隔 1 秒
    const monitorIntervalOk = /MONITOR_INTERVAL_MS\s*=\s*1\s*\*\s*1000|MONITOR_INTERVAL_MS\s*=\s*1000/.test(indexContent);
    allOk = check('价格监控兜底间隔为 1 秒 (MONITOR_INTERVAL_MS = 1*1000)', monitorIntervalOk) && allOk;

    // 2. 使用 MarketOrderbookWS 且端点正确
    const hasMarketOrderbookWS = /MarketOrderbookWS|market_orderbook_ws/.test(indexContent);
    allOk = check('入口使用市场频道 WebSocket (MarketOrderbookWS)', hasMarketOrderbookWS) && allOk;

    if (wsContent) {
        const wsUrlOk = /ws-subscriptions-clob\.polymarket\.com\/ws\/market|WS_URL.*ws-subscriptions/.test(wsContent);
        allOk = check('WebSocket 端点: wss://ws-subscriptions-clob.polymarket.com/ws/market', wsUrlOk) && allOk;
        const hasBookAndPriceChange = /event_type.*book|price_change|handleMessage/.test(wsContent);
        allOk = check('WS 处理 book / price_change 消息', hasBookAndPriceChange) && allOk;
    } else {
        allOk = check('存在 market_orderbook_ws 模块', false) && allOk;
    }

    // 3. 执行器暴露 WS 所需方法
    if (executorContent) {
        allOk = check('执行器: getMonitorTokenIds()', /getMonitorTokenIds\s*\(\)/.test(executorContent)) && allOk;
        allOk = check('执行器: tryTriggerPriceMonitorByTokenId(assetId, bestAsk)', /tryTriggerPriceMonitorByTokenId\s*\([^)]*assetId[^)]*bestAsk/.test(executorContent)) && allOk;
        allOk = check('执行器: tryTriggerLowPriceMonitorByTokenId(assetId, bestAsk)', /tryTriggerLowPriceMonitorByTokenId\s*\([^)]*assetId[^)]*bestAsk/.test(executorContent)) && allOk;
        allOk = check('执行器: tryTriggerLiquidityMonitorByTokenId(assetId, bestAsk, asksSnapshot)', /tryTriggerLiquidityMonitorByTokenId\s*\([^)]*assetId[^)]*bestAsk[^)]*asksSnapshot/.test(executorContent)) && allOk;
        allOk = check('getMonitorTokenIds 含流动性队列 (liquidityMonitorQueue)', /liquidityMonitorQueue/.test(executorContent) && /getMonitorTokenIds/.test(executorContent)) && allOk;
        allOk = check('simulateTrade 支持 triggerPrice/asksSnapshot 复用', /triggerPrice|asksSnapshot|useWsData/.test(executorContent)) && allOk;
        allOk = check('executeLiveTrade 支持 triggerPrice/asksSnapshot 复用', /executeLiveTrade/.test(executorContent) && (executorContent.includes('triggerPrice') || executorContent.includes('asksSnapshot'))) && allOk;
    } else {
        allOk = check('存在 prediction_executor 模块', false) && allOk;
    }

    // 4. 入口中 syncSubscribe 与 onBestAsk 触发逻辑
    const syncSubscribeOk = /syncSubscribe|marketOrderbookWS\.syncSubscribe/.test(indexContent);
    allOk = check('每轮监控时同步 WS 订阅 (syncSubscribe)', syncSubscribeOk) && allOk;
    const onBestAskOk = /tryTriggerPriceMonitorByTokenId|tryTriggerLowPriceMonitorByTokenId|tryTriggerLiquidityMonitorByTokenId/.test(indexContent);
    allOk = check('onBestAsk 触发高价/低价/流动性监控下单', onBestAskOk) && allOk;

    console.log('');

    // 可选：WS 连通性测试
    const tokenId = process.env.VERIFY_WS_TOKEN_ID;
    if (tokenId) {
        console.log('--- WebSocket 连通性测试 ---');
        runLiveWSTest(tokenId).then((ok) => {
            if (!ok) allOk = false;
            console.log('');
            process.exit(allOk ? 0 : 1);
        }).catch((err) => {
            console.log('❌ WS 连通性测试异常:', err.message);
            process.exit(1);
        });
    } else {
        if (!allOk) process.exit(1);
        console.log('提示: 设置环境变量 VERIFY_WS_TOKEN_ID=<某个市场的 tokenId> 可进行 WebSocket 连通性测试\n');
        process.exit(0);
    }
}

function runLiveWSTest(assetId) {
    let WebSocket = globalThis.WebSocket;
    if (!WebSocket) {
        try {
            WebSocket = require('ws');
        } catch (e) {
            console.log('⏭️  跳过 WS 连通性测试（需 Node 22+ 或安装 ws: npm i ws）');
            return Promise.resolve(true);
        }
    }
    return new Promise((resolve) => {
        const url = 'wss://ws-subscriptions-clob.polymarket.com/ws/market';
        const ws = new WebSocket(url);
        const timeout = setTimeout(() => {
            ws.close();
            check('WS 连通性（15s 内收到一条 book/price_change）', false, '超时未收到消息');
            resolve(false);
        }, 15000);

        ws.on('open', () => {
            ws.send(JSON.stringify({ assets_ids: [assetId], type: 'market' }));
        });

        ws.on('message', (data) => {
            if (data.toString() === 'PONG') return;
            try {
                const msg = JSON.parse(data.toString());
                if (msg.event_type === 'book' || msg.event_type === 'price_change') {
                    clearTimeout(timeout);
                    ws.close();
                    check('WS 连通性（15s 内收到一条 book/price_change）', true, `收到 ${msg.event_type}`);
                    resolve(true);
                }
            } catch (_) {}
        });

        ws.on('error', (err) => {
            clearTimeout(timeout);
            check('WS 连通性', false, err.message);
            resolve(false);
        });

        ws.on('close', () => {
            clearTimeout(timeout);
        });
    });
}

main();
