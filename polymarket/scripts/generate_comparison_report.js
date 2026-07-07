/**
 * 生成8套模型的每日对比报告
 * 读取所有日志目录的每日报告，生成对比文件
 */

const fs = require('fs');
const path = require('path');

const LOG_DIRS = [
    { name: '1. models (旧特征集)', dir: 'logs', suffix: '' },
    { name: '2. models_A (旧特征集)', dir: 'logs_A', suffix: '_A' },
    { name: '3. models_B (新特征集 v4)', dir: 'logs_B', suffix: '_B' },
    { name: '4. models_C (新特征集 v4)', dir: 'logs_C', suffix: '_C' },
    { name: '5. models (新特征集 v4)', dir: 'logs_v4', suffix: '_v4' },
    { name: '6. models_A (新特征集 v4)', dir: 'logs_A_v4', suffix: '_A_v4' },
    { name: '7. models_B (旧特征集)', dir: 'logs_B_old', suffix: '_B_old' },
    { name: '8. models_C (旧特征集)', dir: 'logs_C_old', suffix: '_C_old' },
];

function readDailyReport(logDir, dateStr) {
    const reportPath = path.join(process.cwd(), logDir, 'reports', `report_daily_${dateStr}.json`);
    if (!fs.existsSync(reportPath)) {
        return null;
    }
    try {
        return JSON.parse(fs.readFileSync(reportPath, 'utf-8'));
    } catch (e) {
        return null;
    }
}

function formatCurrency(value) {
    if (value === null || value === undefined) return 'N/A';
    return `$${value.toFixed(2)}`;
}

function formatPercent(value) {
    if (value === null || value === undefined) return 'N/A';
    return `${(value * 100).toFixed(2)}%`;
}

function generateComparisonReport(dateStr = null) {
    if (!dateStr) {
        const now = new Date();
        dateStr = now.toISOString().split('T')[0].replace(/-/g, '');
    }

    console.log(`\n📊 生成 ${dateStr} 的对比报告...\n`);

    const reports = [];
    for (const config of LOG_DIRS) {
        const report = readDailyReport(config.dir, dateStr);
        if (report) {
            reports.push({
                name: config.name,
                dir: config.dir,
                suffix: config.suffix,
                data: report,
            });
        } else {
            console.log(`⚠️  ${config.name} (${config.dir}) 无报告数据`);
        }
    }

    if (reports.length === 0) {
        console.log('❌ 没有找到任何报告数据');
        return;
    }

    // 生成对比报告
    const comparisonDir = path.join(process.cwd(), 'logs_comparison');
    if (!fs.existsSync(comparisonDir)) {
        fs.mkdirSync(comparisonDir, { recursive: true });
    }

    const txtPath = path.join(comparisonDir, `comparison_${dateStr}.txt`);
    const jsonPath = path.join(comparisonDir, `comparison_${dateStr}.json`);

    let txt = `============================================================\n`;
    txt += `8套模型每日对比报告 - ${dateStr}\n`;
    txt += `============================================================\n\n`;

    // 汇总表格
    txt += `8套模型对比汇总\n`;
    txt += `${'='.repeat(90)}\n`;
    txt += `${'模型套别'.padEnd(35)} ${'总盈亏'.padEnd(12)} ${'收益率'.padEnd(12)} ${'交易数'.padEnd(10)} ${'胜率'.padEnd(10)} ${'当前资金'.padEnd(12)}\n`;
    txt += `${'-'.repeat(90)}\n`;

    const jsonData = {
        date: dateStr,
        models: [],
    };

    for (const item of reports) {
        const r = item.data;
        const summary = r.summary || {};
        const totalPnL = summary.totalPnL || 0;
        const initialCapital = summary.initialCapital || 400;
        const currentCapital = summary.currentCapital || initialCapital;
        const totalReturn = initialCapital > 0 ? (currentCapital - initialCapital) / initialCapital : 0;
        const tradeCount = summary.totalTrades || 0;
        const winRate = summary.winRate ? summary.winRate / 100 : 0; // winRate 是百分比，需要除以100

        txt += `${item.name.padEnd(35)} ${formatCurrency(totalPnL).padEnd(12)} ${formatPercent(totalReturn).padEnd(12)} ${tradeCount.toString().padEnd(10)} ${formatPercent(winRate).padEnd(10)} ${formatCurrency(currentCapital).padEnd(12)}\n`;

        jsonData.models.push({
            name: item.name,
            dir: item.dir,
            suffix: item.suffix,
            summary: {
                totalPnL,
                totalReturn,
                tradeCount,
                winRate,
                currentCapital,
                initialCapital,
            },
        });
    }

    txt += `\n${'='.repeat(90)}\n\n`;

    // 详细对比
    txt += `详细对比\n`;
    txt += `${'='.repeat(80)}\n\n`;

    for (const item of reports) {
        const r = item.data;
        const s = r.summary || {};
        txt += `${item.name}\n`;
        txt += `${'-'.repeat(40)}\n`;
        txt += `  初始资金: ${formatCurrency(s.initialCapital || 400)}\n`;
        txt += `  当前资金: ${formatCurrency(s.currentCapital || s.initialCapital || 400)}\n`;
        txt += `  总盈亏: ${formatCurrency(s.totalPnL || 0)}\n`;
        const initCap = s.initialCapital || 400;
        const currCap = s.currentCapital || initCap;
        const ret = initCap > 0 ? (currCap - initCap) / initCap : 0;
        txt += `  收益率: ${formatPercent(ret)}\n`;
        txt += `  交易数: ${s.totalTrades || 0}\n`;
        txt += `  胜率: ${formatPercent((s.winRate || 0) / 100)}\n`;
        txt += `\n`;
    }

    // 排名
    txt += `\n${'='.repeat(90)}\n`;
    txt += `收益率排名（从高到低）\n`;
    txt += `${'='.repeat(90)}\n`;

    const sorted = [...reports].sort((a, b) => {
        const sA = a.data.summary || {};
        const sB = b.data.summary || {};
        const initA = sA.initialCapital || 400;
        const initB = sB.initialCapital || 400;
        const currA = sA.currentCapital || initA;
        const currB = sB.currentCapital || initB;
        const returnA = initA > 0 ? (currA - initA) / initA : -Infinity;
        const returnB = initB > 0 ? (currB - initB) / initB : -Infinity;
        return returnB - returnA;
    });

    sorted.forEach((item, index) => {
        const s = item.data.summary || {};
        const initCap = s.initialCapital || 400;
        const currCap = s.currentCapital || initCap;
        const returnVal = initCap > 0 ? (currCap - initCap) / initCap : 0;
        const totalPnL = s.totalPnL || 0;
        txt += `${(index + 1).toString().padStart(2)}. ${item.name.padEnd(35)} ${formatPercent(returnVal).padEnd(12)} ${formatCurrency(totalPnL)}\n`;
    });

    // 保存文件
    fs.writeFileSync(txtPath, txt, 'utf-8');
    fs.writeFileSync(jsonPath, JSON.stringify(jsonData, null, 2), 'utf-8');

    console.log(`✅ 对比报告已生成:`);
    console.log(`   ${txtPath}`);
    console.log(`   ${jsonPath}`);
    console.log(`\n📈 收益率排名（Top 3）:`);
    sorted.slice(0, 3).forEach((item, index) => {
        const s = item.data.summary || {};
        const initCap = s.initialCapital || 400;
        const currCap = s.currentCapital || initCap;
        const returnVal = initCap > 0 ? (currCap - initCap) / initCap : 0;
        console.log(`   ${index + 1}. ${item.name}: ${formatPercent(returnVal)} (${formatCurrency(s.totalPnL || 0)})`);
    });
}

// 命令行参数
const args = process.argv.slice(2);
if (args.length > 0 && args[0] === '--date') {
    generateComparisonReport(args[1]);
} else {
    generateComparisonReport();
}
