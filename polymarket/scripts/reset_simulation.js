#!/usr/bin/env node
/**
 * 清空模拟盘历史记录，从 INITIAL_CAPITAL（如 400）重新跑。
 *
 * 说明：当前资金是在 predict-trade 启动时从 logs/prediction_trades.json 重算的，
 * 不会单独存盘。若只删日志但不重启进程，内存里仍是旧资金。
 *
 * - 将 logs/prediction_trades.json 置为 []
 * - 删除 logs/reports/ 下所有日报、小时报
 *
 * ⚠️ 必须先停止 npm run predict-trade (Ctrl+C)，执行本脚本后，再重新启动。
 */

const fs = require('fs');
const path = require('path');

const cwd = process.cwd();
const logsBase = process.env.LOGS_DIR || 'logs';
const logFile = path.join(cwd, logsBase, 'prediction_trades.json');
const reportsDir = path.join(cwd, logsBase, 'reports');

// 1. 清空交易记录
if (fs.existsSync(logFile)) {
  fs.writeFileSync(logFile, '[]', 'utf8');
  console.log(`✓ 已清空 模拟盘交易记录: ${logsBase}/prediction_trades.json`);
} else {
  const logDir = path.dirname(logFile);
  if (!fs.existsSync(logDir)) {
    fs.mkdirSync(logDir, { recursive: true });
  }
  fs.writeFileSync(logFile, '[]', 'utf8');
  console.log(`✓ 已创建空的 ${logsBase}/prediction_trades.json`);
}

// 2. 删除历史报告
let removed = 0;
if (fs.existsSync(reportsDir)) {
  const files = fs.readdirSync(reportsDir);
  for (const f of files) {
    const p = path.join(reportsDir, f);
    if (fs.statSync(p).isFile()) {
      fs.unlinkSync(p);
      removed++;
    }
  }
  if (removed > 0) {
    console.log(`✓ 已删除 ${logsBase}/reports/ 下 ${removed} 个历史报告`);
  }
}

console.log('');
console.log('⚠️  请先停止 npm run predict-trade (Ctrl+C)，再执行本脚本。');
console.log('    完成后重新运行 npm run predict-trade，将从 INITIAL_CAPITAL（如 400）重新开始。');
console.log('');
