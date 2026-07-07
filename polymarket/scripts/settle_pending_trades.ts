/**
 * 手动结算所有待结算交易
 * 遍历各组合的 prediction_trades.json，对 status=executed 且 result 为 pending/未设置的交易
 * 查询 Polymarket 市场结果，若已结算则更新为 win/lose 并写入 pnl，然后重新生成报告。
 *
 * 用法（在 polymarket 目录下）:
 *   npx ts-node scripts/settle_pending_trades.ts
 *   LOGS_DIR=logs_eth npx ts-node scripts/settle_pending_trades.ts   # 只处理指定目录
 */

import * as path from 'path';
import { TradeLogger } from '../src/models/trade_log';
import {
    getMarketResultByConditionId,
    getMarketResult,
} from '../src/services/market_finder';

const ALL_LOG_DIRS = ['logs_eth', 'logs_eth_10_90', 'logs_btc', 'logs_xrp', 'logs_xrp_20_80'];

async function settleOneDir(logDir: string): Promise<{ settled: number; skipped: number }> {
    const cwd = process.cwd();
    const fullPath = path.join(cwd, logDir);
    const logger = new TradeLogger(fullPath);
    const pending = logger.getPendingTrades();

    if (pending.length === 0) {
        return { settled: 0, skipped: 0 };
    }

    console.log(`\n📂 ${logDir}: ${pending.length} 笔待结算`);
    let settled = 0;

    for (const trade of pending) {
        if (!trade.conditionId && !trade.marketSlug) {
            console.log(`  ⏭ 跳过 ${trade.id}: 无 conditionId/marketSlug`);
            continue;
        }

        let marketResult;
        if (trade.conditionId) {
            marketResult = await getMarketResultByConditionId(
                trade.conditionId,
                trade.marketSlug
            );
            if (!marketResult.resolved && trade.marketSlug) {
                marketResult = await getMarketResult(trade.marketSlug);
            }
        } else {
            marketResult = await getMarketResult(trade.marketSlug!);
        }

        if (!marketResult.resolved) {
            console.log(`  ⏳ ${trade.symbol} ${trade.marketSlug || trade.conditionId?.slice(0, 18)}: 市场尚未结算`);
            continue;
        }

        const tokenOutcome = trade.tokenOutcome || (trade.direction === 'UP' ? 'Up' : 'Down');
        const won = marketResult.winner === tokenOutcome;
        const buyPrice = trade.tokenPrice ?? 0.5;
        // Polymarket 15m crypto: 赢局到账为扣费后代币数，与 prediction_executor 一致
        const tokensOwned = trade.amount / buyPrice;
        const feeRate = buyPrice > 0 && buyPrice < 1 ? 0.25 * Math.pow(buyPrice * (1 - buyPrice), 2) : 0;
        const tokensAfterFee = tokensOwned * (1 - feeRate);
        const pnl = won ? tokensAfterFee - trade.amount : -trade.amount;

        logger.settleTrade(trade.id, won, pnl);
        settled++;
        console.log(`  ✅ ${trade.id.slice(-10)}: ${won ? '胜' : '负'} $${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}`);
    }

    return { settled, skipped: pending.length - settled };
}

async function main(): Promise<void> {
    const singleDir = process.env.LOGS_DIR;
    const dirs = singleDir ? [singleDir] : ALL_LOG_DIRS;

    console.log('════════════════════════════════════════════════════════════');
    console.log('  📊 手动结算待结算交易');
    console.log('════════════════════════════════════════════════════════════');
    if (singleDir) {
        console.log(`  仅处理: ${singleDir}`);
    } else {
        console.log(`  处理目录: ${dirs.join(', ')}`);
    }

    let totalSettled = 0;
    for (const dir of dirs) {
        try {
            const { settled } = await settleOneDir(dir);
            totalSettled += settled;
        } catch (e) {
            console.error(`  ❌ ${dir} 处理失败:`, e);
        }
    }

    console.log('\n' + '─'.repeat(60));
    console.log(`  合计结算: ${totalSettled} 笔`);
    console.log('─'.repeat(60) + '\n');

    if (totalSettled > 0) {
        console.log('正在重新生成汇总报告...');
        const { spawnSync } = require('child_process');
        const r = spawnSync('node', ['scripts/generate_summary_reports.js'], {
            cwd: process.cwd(),
            stdio: 'inherit',
        });
        if (r.status !== 0) {
            console.error('生成报告失败，可手动执行: node scripts/generate_summary_reports.js');
        }
    }
}

main().catch((e) => {
    console.error(e);
    process.exit(1);
});
