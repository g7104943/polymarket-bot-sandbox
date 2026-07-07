#!/usr/bin/env node
/**
 * 立即为所有日志目录生成总汇总报告
 * 
 * 用法：
 *   node scripts/generate_summary_reports.js
 * 
 * 会为当前 5 个旧组合 (logs_eth, logs_eth_10_90, logs_btc, logs_xrp, logs_xrp_20_80) 分别生成 report_summary.txt/json
 */

const fs = require('fs');
const path = require('path');

const PROJECT_ROOT = path.resolve(__dirname, '..', '..');
const LOG_DIRS = ['logs_eth', 'logs_eth_10_90', 'logs_btc', 'logs_xrp', 'logs_xrp_20_80'];

// 从 prediction_trades.json 读取所有交易
function loadTrades(logDir) {
    const logFile = path.join(PROJECT_ROOT, 'polymarket', logDir, 'prediction_trades.json');
    if (!fs.existsSync(logFile)) {
        return [];
    }
    try {
        const data = fs.readFileSync(logFile, 'utf-8');
        return JSON.parse(data) || [];
    } catch (e) {
        console.error(`  ❌ 读取 ${logDir}/prediction_trades.json 失败:`, e.message);
        return [];
    }
}

// 计算汇总统计
// 胜率按逻辑订单计算：同一 conditionId 的多笔成交（部分成交/补单）算一笔订单
// 待结算 = 仅「已执行但市场尚未出结果」的笔数（不含下单失败）
function calculateSummary(trades) {
    const logicalKey = (t) => t.conditionId || t.id;
    const groups = new Map();
    for (const t of trades) {
        const key = logicalKey(t);
        if (!groups.has(key)) groups.set(key, []);
        groups.get(key).push(t);
    }
    const logicalOrders = Array.from(groups.values());

    let wins = 0, losses = 0, pending = 0, failed = 0, totalPnL = 0, totalConfidence = 0;
    let upCount = 0, downCount = 0;
    const bySymbol = {};

    for (const trade of trades) {
        totalConfidence += (trade.confidence || 0);
        if (trade.pnl !== undefined) totalPnL += trade.pnl;
        const sym = trade.symbol || 'UNKNOWN';
        if (!bySymbol[sym]) {
            bySymbol[sym] = { trades: 0, wins: 0, losses: 0, pending: 0, failed: 0, pnl: 0, totalConfidence: 0, upCount: 0, downCount: 0 };
        }
        if (trade.status === 'failed') {
            failed++;
            bySymbol[sym].failed++;
        }
    }
    for (const group of logicalOrders) {
        const first = group[0];
        const sym = first.symbol || 'UNKNOWN';
        if (!bySymbol[sym]) {
            bySymbol[sym] = { trades: 0, wins: 0, losses: 0, pending: 0, failed: 0, pnl: 0, totalConfidence: 0, upCount: 0, downCount: 0 };
        }
        const s = bySymbol[sym];
        s.trades++;
        group.forEach(t => {
            s.totalConfidence += (t.confidence || 0);
            if (t.pnl !== undefined) s.pnl += t.pnl;
        });
        const result = group.some(t => t.result === 'win') ? 'win' : group.some(t => t.result === 'lose') ? 'lose' : (group.some(t => t.status === 'executed') ? 'pending' : null);
        if (result === 'win') { wins++; s.wins++; }
        else if (result === 'lose') { losses++; s.losses++; }
        else if (result === 'pending') { pending++; s.pending++; }
        if (first.direction === 'UP') { upCount++; s.upCount++; }
        else if (first.direction === 'DOWN') { downCount++; s.downCount++; }
    }

    for (const k of Object.keys(bySymbol)) {
        const s = bySymbol[k];
        const fillCount = trades.filter(t => (t.symbol || 'UNKNOWN') === k).length;
        s.avgConfidence = fillCount > 0 ? s.totalConfidence / fillCount : 0;
    }

    const completed = wins + losses;
    const winRate = completed > 0 ? wins / completed : 0;
    const avgConf = trades.length > 0 ? totalConfidence / trades.length : 0;

    return {
        totalTrades: logicalOrders.length,
        totalFills: trades.length,
        completedTrades: completed,
        wins, losses, pending, failed,
        upCount, downCount,
        winRate: Math.round(winRate * 1000) / 10,
        totalPnL: Math.round(totalPnL * 100) / 100,
        avgConfidence: Math.round(avgConf * 1000) / 10,
        bySymbol: Object.fromEntries(
            Object.entries(bySymbol).map(([sym, s]) => [
                sym,
                {
                    trades: s.trades,
                    wins: s.wins,
                    losses: s.losses,
                    pending: s.pending,
                    failed: s.failed,
                    upCount: s.upCount,
                    downCount: s.downCount,
                    winRate: (s.wins + s.losses) > 0 ? Math.round(s.wins / (s.wins + s.losses) * 1000) / 10 : 0,
                    pnl: Math.round(s.pnl * 100) / 100,
                    avgConfidence: Math.round(s.avgConfidence * 1000) / 10,
                },
            ])
        ),
    };
}

// 按日期汇总：每日逻辑订单数、胜/负、盈亏（同一 conditionId 在同一天只算一笔，与主报告一致）
function computeDailyStats(trades) {
    const logicalKey = (t) => t.conditionId || t.id;
    const byDay = {};
    const seenPerDay = {};
    for (const t of trades) {
        const ts = t.timestamp ? new Date(t.timestamp) : null;
        const day = ts ? `${ts.getFullYear()}/${String(ts.getMonth() + 1).padStart(2, '0')}/${String(ts.getDate()).padStart(2, '0')}` : 'unknown';
        if (!byDay[day]) { byDay[day] = { trades: 0, wins: 0, losses: 0, pnl: 0 }; seenPerDay[day] = new Set(); }
        const key = logicalKey(t);
        if (!seenPerDay[day].has(key)) {
            seenPerDay[day].add(key);
            byDay[day].trades++;
            if (t.result === 'win') byDay[day].wins++;
            else if (t.result === 'lose') byDay[day].losses++;
        }
        if (t.result === 'win' && t.pnl !== undefined) byDay[day].pnl += t.pnl;
        else if (t.result === 'lose' && t.pnl !== undefined) byDay[day].pnl += t.pnl;
    }
    const days = Object.keys(byDay).filter(d => d !== 'unknown').sort();
    return days.map(d => ({ date: d, ...byDay[d], pnl: Math.round(byDay[d].pnl * 100) / 100 }));
}

// 格式化文本报告
function formatTextReport(report, firstTrade, now, initialCapital, currentCapital, trades = []) {
    const lines = [];
    lines.push('═'.repeat(60));
    lines.push('  POLYMARKET 预测交易 - 总汇总报告（全部历史）');
    lines.push('═'.repeat(60));
    lines.push('');
    lines.push(`报告时间: ${now.toISOString()}`);
    lines.push(`统计周期: ${firstTrade.toISOString()} ~ ${now.toISOString()} (从开始到现在的全部历史)`);
    lines.push('');
    lines.push('─'.repeat(60));
    lines.push('  综合统计');
    lines.push('─'.repeat(60));
    lines.push(`  总交易数:     ${report.totalTrades}`);
    lines.push(`  下单方向:     Up: ${report.upCount ?? 0} 笔, Down: ${report.downCount ?? 0} 笔`);
    lines.push(`  已完成:       ${report.completedTrades}`);
    lines.push(`  胜利:         ${report.wins}`);
    lines.push(`  失败:         ${report.losses}`);
    lines.push(`  待结算:       ${report.pending} (已执行、等待市场出结果)`);
    lines.push(`  下单失败:     ${report.failed ?? 0}`);
    lines.push(`  胜率:         ${report.winRate}%`);
    lines.push(`  总盈亏:       $${report.totalPnL}`);
    lines.push(`  平均置信度:   ${report.avgConfidence}%`);
    lines.push(`  初始资金:     $${initialCapital}`);
    lines.push(`  当前资金:     $${currentCapital}`);
    lines.push(`  资金变化:     $${(currentCapital - initialCapital) >= 0 ? '+' : ''}${(currentCapital - initialCapital).toFixed(2)}`);
    lines.push('');
    // 每日胜率
    const dailyStats = computeDailyStats(trades);
    if (dailyStats.length > 0) {
        lines.push('─'.repeat(60));
        lines.push('  每日胜率统计');
        lines.push('─'.repeat(60));
        for (const d of dailyStats) {
            const completed = d.wins + d.losses;
            const winRate = completed > 0 ? Math.round((d.wins / completed) * 1000) / 10 : 0;
            const pnlStr = (d.pnl >= 0 ? '$+' : '$') + d.pnl.toFixed(2);
            lines.push(`  ${d.date}:`);
            lines.push(`    交易数: ${d.trades}, 胜率: ${winRate}%, 盈亏: ${pnlStr}`);
        }
        lines.push('');
    }
    lines.push('─'.repeat(60));
    lines.push('  各币种统计');
    lines.push('─'.repeat(60));
    for (const [symbol, stats] of Object.entries(report.bySymbol)) {
        lines.push(`  ${symbol}:`);
        lines.push(`    交易数: ${stats.trades}, 胜率: ${stats.winRate}%, 盈亏: $${stats.pnl}`);
        lines.push(`    下单: Up ${stats.upCount ?? 0} 笔, Down ${stats.downCount ?? 0} 笔 | 胜/负/待/失败: ${stats.wins}/${stats.losses}/${stats.pending}/${stats.failed ?? 0}, 平均置信度: ${stats.avgConfidence}%`);
    }
    lines.push('');
    lines.push('═'.repeat(60));
    lines.push('  报告结束');
    lines.push('═'.repeat(60));
    return lines.join('\n');
}

// 计算当前资金（从交易记录推算）
function calculateCurrentCapital(trades, initialCapital = 400) {
    let capital = initialCapital;
    for (const trade of trades) {
        if (trade.pnl !== undefined) {
            capital += trade.pnl;
        }
    }
    return Math.round(capital * 100) / 100;
}

// 为单个日志目录生成汇总报告
function generateReportForLogDir(logDir) {
    const reportsDir = path.join(PROJECT_ROOT, 'polymarket', logDir, 'reports');
    if (!fs.existsSync(reportsDir)) {
        fs.mkdirSync(reportsDir, { recursive: true });
    }

    const liveSummaryPath = path.join(reportsDir, 'report_summary.live.json');
    const liveSummaryTxtPath = path.join(reportsDir, 'report_summary.live.txt');
    if (fs.existsSync(liveSummaryPath)) {
        const jsonPath = path.join(reportsDir, 'report_summary.json');
        const txtPath = path.join(reportsDir, 'report_summary.txt');
        const liveReport = JSON.parse(fs.readFileSync(liveSummaryPath, 'utf-8'));
        fs.writeFileSync(jsonPath, JSON.stringify(liveReport, null, 2), 'utf-8');
        if (fs.existsSync(liveSummaryTxtPath)) {
            fs.copyFileSync(liveSummaryTxtPath, txtPath);
        }
        const summary = liveReport.summary || {};
        console.log(`  ✅ ${logDir}: 复用 live 汇总报告作为兼容汇总`);
        console.log(`     总交易: ${summary.totalTrades ?? 0} 笔 | 胜/负/待/失败: ${summary.wins ?? 0}/${summary.losses ?? 0}/${summary.pending ?? 0}/${summary.failed ?? 0}`);
        console.log(`     胜率: ${summary.winRate ?? 0}% | 总盈亏: $${Number(summary.totalPnL || 0) >= 0 ? '+' : ''}${summary.totalPnL ?? 0} | 资金: $${summary.currentCapital ?? 0}`);
        console.log(`     📁 ${jsonPath}`);
        console.log(`     📁 ${txtPath}`);
        return;
    }
    
    const trades = loadTrades(logDir);
    if (trades.length === 0) {
        console.log(`  ⚠️  ${logDir}: 无交易记录，跳过`);
        return;
    }
    
    const now = new Date();
    const firstTrade = trades.length > 0 ? new Date(trades[0].timestamp) : now;
    const initialCapital = 400; // 默认初始资金
    const currentCapital = calculateCurrentCapital(trades, initialCapital);
    
    const summary = calculateSummary(trades);
    
    const report = {
        reportType: 'summary',
        reportDate: now.toISOString(),
        reportPeriod: {
            start: firstTrade.toISOString(),
            end: now.toISOString(),
        },
        summary: {
            ...summary,
            currentCapital,
            initialCapital,
            capitalChange: Math.round((currentCapital - initialCapital) * 100) / 100,
        },
        bySymbol: summary.bySymbol,
        trades: trades.map((t) => ({
            id: t.id,
            timestamp: t.timestamp,
            symbol: t.symbol,
            direction: t.direction,
            confidence: t.confidence,
            amount: t.amount,
            result: t.result || 'pending',
            pnl: t.pnl,
        })),
    };
    
    const jsonPath = path.join(reportsDir, 'report_summary.json');
    const txtPath = path.join(reportsDir, 'report_summary.txt');
    
    fs.writeFileSync(jsonPath, JSON.stringify(report, null, 2), 'utf-8');
    fs.writeFileSync(txtPath, formatTextReport(summary, firstTrade, now, initialCapital, currentCapital, trades), 'utf-8');
    
    console.log(`  ✅ ${logDir}: 生成汇总报告`);
    console.log(`     总交易: ${summary.totalTrades} 笔 | 胜/负/待/失败: ${summary.wins}/${summary.losses}/${summary.pending}/${summary.failed ?? 0}`);
    console.log(`     胜率: ${summary.winRate}% | 总盈亏: $${summary.totalPnL >= 0 ? '+' : ''}${summary.totalPnL} | 资金: $${currentCapital}`);
    console.log(`     📁 ${jsonPath}`);
    console.log(`     📁 ${txtPath}`);
}

// 主函数
function main() {
    console.log('═'.repeat(60));
    console.log('  📊 立即生成所有日志目录的总汇总报告');
    console.log('═'.repeat(60));
    console.log('');
    
    for (const logDir of LOG_DIRS) {
        const logPath = path.join(PROJECT_ROOT, 'polymarket', logDir);
        if (!fs.existsSync(logPath)) {
            console.log(`  ⚠️  ${logDir}: 目录不存在，跳过`);
            continue;
        }
        generateReportForLogDir(logDir);
        console.log('');
    }
    
    console.log('═'.repeat(60));
    console.log('  ✅ 完成！所有汇总报告已生成');
    console.log('═'.repeat(60));
    console.log('');
    console.log('📁 报告位置：');
    for (const logDir of LOG_DIRS) {
        const txtPath = path.join(PROJECT_ROOT, 'polymarket', logDir, 'reports', 'report_summary.txt');
        if (fs.existsSync(txtPath)) {
            console.log(`  - ${logDir}/reports/report_summary.txt`);
        }
    }
    console.log('');
}

if (require.main === module) {
    main();
}

module.exports = { generateReportForLogDir, loadTrades, calculateSummary };
