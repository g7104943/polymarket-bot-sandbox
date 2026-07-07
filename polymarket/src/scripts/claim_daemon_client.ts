#!/usr/bin/env node

import { createPredictionEnv } from '../config/prediction_env';
import {
    deploySafeWalletGasless,
    fetchTransactionById,
    fetchSafeDeploymentStatus,
    fetchRecentTransactionsForUser,
    inspectFundStatus,
    redeemPositionsGasless,
    resolveClaimAuthModes,
    resolveWalletModel,
    wrapLegacyUsdceToPusdGasless,
} from '../services/claim_relayer';

type Command = 'redeem' | 'recent-transactions' | 'transaction' | 'wallet-model' | 'deployed' | 'deploy' | 'wrap-usdce' | 'fund-status';

function usage(): never {
    console.error(
        [
            '用法:',
                '  node dist/scripts/claim_daemon_client.js redeem --condition-id 0x... [--asset-id 123...] [--collateral-token 0x...]',
            '  node dist/scripts/claim_daemon_client.js recent-transactions --user 0x... [--limit 100]',
            '  node dist/scripts/claim_daemon_client.js transaction --id 019d...',
            '  node dist/scripts/claim_daemon_client.js wallet-model',
            '  node dist/scripts/claim_daemon_client.js deployed',
            '  node dist/scripts/claim_daemon_client.js deploy',
            '  node dist/scripts/claim_daemon_client.js wrap-usdce [--amount-raw 123000000]',
            '  node dist/scripts/claim_daemon_client.js fund-status',
        ].join('\n'),
    );
    process.exit(1);
}

function parseArgs(argv: string[]): { command: Command; options: Record<string, string> } {
    const [commandRaw, ...rest] = argv;
    const command = String(commandRaw || '').trim() as Command;
    if (command !== 'redeem' && command !== 'recent-transactions' && command !== 'transaction' && command !== 'wallet-model' && command !== 'deployed' && command !== 'deploy' && command !== 'wrap-usdce' && command !== 'fund-status') usage();
    const options: Record<string, string> = {};
    for (let i = 0; i < rest.length; i++) {
        const token = rest[i];
        if (!token.startsWith('--')) usage();
        const key = token.slice(2);
        const value = rest[i + 1];
        if (!value || value.startsWith('--')) usage();
        options[key] = value;
        i += 1;
    }
    return { command, options };
}

async function main(): Promise<void> {
    const { command, options } = parseArgs(process.argv.slice(2));
    const env = createPredictionEnv(process.env);
    if (command === 'wallet-model') {
        const walletModel = await resolveWalletModel(env);
        const authModes = await resolveClaimAuthModes(env);
        process.stdout.write(`${JSON.stringify({ command, ...walletModel, ...authModes })}\n`);
        process.exit(walletModel.selectedWalletModel === 'BLOCKED' ? 2 : 0);
    }
    if (command === 'deployed') {
        const result = await fetchSafeDeploymentStatus(env);
        process.stdout.write(`${JSON.stringify({ command, ...result })}\n`);
        process.exit(result.ok ? 0 : 2);
    }
    if (command === 'fund-status') {
        const result = await inspectFundStatus(env);
        process.stdout.write(`${JSON.stringify({ command, ...result })}\n`);
        process.exit(result.ok ? 0 : 2);
    }
    if (command === 'deploy') {
        const result = await deploySafeWalletGasless(env);
        process.stdout.write(`${JSON.stringify({ command, ...result })}\n`);
        process.exit(result.ok ? 0 : 2);
    }
    if (command === 'redeem') {
        const conditionId = String(options['condition-id'] || '').trim();
        if (!conditionId) usage();
        const assetId = String(options['asset-id'] || '').trim() || undefined;
        const collateralToken = String(options['collateral-token'] || '').trim() || undefined;
        const result = await redeemPositionsGasless(conditionId, env, assetId, collateralToken);
        process.stdout.write(`${JSON.stringify({ command, conditionId, ...result })}\n`);
        process.exit(result.ok ? 0 : 2);
    }
    if (command === 'wrap-usdce') {
        const amountRaw = String(options['amount-raw'] || '').trim() || undefined;
        const result = await wrapLegacyUsdceToPusdGasless(env, amountRaw);
        process.stdout.write(`${JSON.stringify({ command, amountRaw: amountRaw || null, ...result })}\n`);
        process.exit(result.ok ? 0 : 2);
    }

    if (command === 'transaction') {
        const transactionId = String(options.id || '').trim();
        if (!transactionId) usage();
        const result = await fetchTransactionById(transactionId, env);
        process.stdout.write(`${JSON.stringify({ command, transactionId, ...result })}\n`);
        process.exit(result.ok ? 0 : 2);
    }

    const user = String(options.user || '').trim();
    const limit = Number(options.limit || '100');
    if (!user) usage();
    const result = await fetchRecentTransactionsForUser(user, env, limit);
    process.stdout.write(`${JSON.stringify({ command, user, limit, ...result })}\n`);
    process.exit(result.ok ? 0 : 2);
}

main().catch((error) => {
    process.stdout.write(`${JSON.stringify({ ok: false, error: String((error as Error)?.message || error) })}\n`);
    process.exit(1);
});
