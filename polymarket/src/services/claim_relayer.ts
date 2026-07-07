import { PredictionEnvConfig } from '../config/prediction_env';

const POLYGON_CHAIN_ID = 137;
const CTF_ADDRESS = '0x4d97dcd97ec945f40cf65f87097ace5ea0476045';
const PUSD_ADDRESS = '0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB';
const USDC_E_ADDRESS = '0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174';
const COLLATERAL_ONRAMP_ADDRESS = '0x93070a847efEf7F70739046A929D47a521F5B8ee';
const ZERO_BYTES32 = '0x' + '0'.repeat(64);
const RELAYER_GET_TRANSACTIONS = '/transactions';
const RELAYER_GET_TRANSACTION = '/transaction';
const RELAYER_GET_DEPLOYED = '/deployed';

const CTF_REDEEM_ABI = [
    {
        constant: false,
        inputs: [
            { name: 'collateralToken', type: 'address' },
            { name: 'parentCollectionId', type: 'bytes32' },
            { name: 'conditionId', type: 'bytes32' },
            { name: 'indexSets', type: 'uint256[]' },
        ],
        name: 'redeemPositions',
        outputs: [],
        payable: false,
        stateMutability: 'nonpayable',
        type: 'function',
    },
] as const;

const ERC20_ABI = [
    {
        constant: true,
        inputs: [{ name: 'account', type: 'address' }],
        name: 'balanceOf',
        outputs: [{ name: '', type: 'uint256' }],
        payable: false,
        stateMutability: 'view',
        type: 'function',
    },
    {
        constant: true,
        inputs: [
            { name: 'owner', type: 'address' },
            { name: 'spender', type: 'address' },
        ],
        name: 'allowance',
        outputs: [{ name: '', type: 'uint256' }],
        payable: false,
        stateMutability: 'view',
        type: 'function',
    },
    {
        constant: false,
        inputs: [
            { name: 'spender', type: 'address' },
            { name: 'amount', type: 'uint256' },
        ],
        name: 'approve',
        outputs: [{ name: '', type: 'bool' }],
        payable: false,
        stateMutability: 'nonpayable',
        type: 'function',
    },
] as const;

const COLLATERAL_ONRAMP_ABI = [
    {
        constant: false,
        inputs: [
            { name: '_asset', type: 'address' },
            { name: '_to', type: 'address' },
            { name: '_amount', type: 'uint256' },
        ],
        name: 'wrap',
        outputs: [],
        payable: false,
        stateMutability: 'nonpayable',
        type: 'function',
    },
] as const;
const ZERO_BIGINT = BigInt(0);

export type ClaimWalletModel = 'EOA' | 'SAFE' | 'PROXY' | 'BLOCKED';
export type ClaimAuthMode = 'builder_auth' | 'relayer_key_auth' | 'unavailable';

export interface WalletModelResolution {
    signerEoa: string;
    configuredProfileWallet: string;
    derivedSafeWallet?: string;
    derivedProxyWallet?: string;
    configuredWalletHasCode: boolean;
    selectedWalletModel: ClaimWalletModel;
    walletModelReason: string;
}

export interface ClaimAuthResolution {
    claimSubmitAuthMode: ClaimAuthMode;
    claimVerifyAuthMode: ClaimAuthMode;
}

export interface GaslessClaimResult {
    ok: boolean;
    txHash?: string;
    transactionId?: string;
    state?: string;
    confirmed?: boolean;
    error?: string;
    warning?: string;
    walletModel?: ClaimWalletModel;
    walletModelReason?: string;
    claimSubmitAuthMode?: ClaimAuthMode;
    claimVerifyAuthMode?: ClaimAuthMode;
    collateralToken?: string;
    collateralTokenSource?: string;
    amountRaw?: string;
    amountFormatted?: string;
    legacyUsdceBalanceRaw?: string;
    legacyUsdceBalanceFormatted?: string;
}

export interface FundStatusResult {
    ok: boolean;
    status: 'empty' | 'legacy_wrap_required' | 'ready' | 'activation_review_required' | 'error';
    warning?: string;
    error?: string;
    walletModel?: ClaimWalletModel;
    walletModelReason?: string;
    claimSubmitAuthMode?: ClaimAuthMode;
    claimVerifyAuthMode?: ClaimAuthMode;
    legacyUsdceBalanceRaw?: string;
    legacyUsdceBalanceFormatted?: string;
    legacyUsdceAllowanceRaw?: string;
    legacyUsdceAllowanceFormatted?: string;
    onchainPusdBalanceRaw?: string;
    onchainPusdBalanceFormatted?: string;
    onchainPusdAllowanceRaw?: string;
    onchainPusdAllowanceFormatted?: string;
    clobBalanceRaw?: string | null;
    clobBalanceFormatted?: string | null;
    clobAllowanceRaw?: string | null;
    clobAllowanceFormatted?: string | null;
    effectiveTradingBalanceRaw?: string | null;
    effectiveTradingBalanceFormatted?: string | null;
    effectiveTradingAllowanceRaw?: string | null;
    effectiveTradingAllowanceFormatted?: string | null;
    balanceSource?: string | null;
    allowanceSource?: string | null;
}

export interface RelayerRecentTransactionsResult {
    ok: boolean;
    transactions?: any[];
    error?: string;
    authMode?: ClaimAuthMode;
}

export interface RelayerTransactionResult {
    ok: boolean;
    transaction?: any;
    error?: string;
    authMode?: ClaimAuthMode;
}

export interface ClaimWalletDeploymentResult {
    ok: boolean;
    deployed: boolean;
    walletModel?: ClaimWalletModel;
    walletModelReason?: string;
    authMode?: ClaimAuthMode;
    transactionId?: string;
    txHash?: string;
    state?: string;
    error?: string;
}

async function resolveRedeemCollateralToken(
    conditionId: string,
    env: PredictionEnvConfig,
    assetId?: string,
    overrideCollateralToken?: string,
): Promise<{ collateralToken: string; source: string }> {
    const override = String(overrideCollateralToken || '').trim();
    if (/^0x[a-fA-F0-9]{40}$/.test(override)) {
        return { collateralToken: override, source: 'override' };
    }

    const targetAsset = String(assetId || '').trim();
    if (!targetAsset) {
        return { collateralToken: PUSD_ADDRESS, source: 'default_pusd_no_asset' };
    }

    const rpcCandidates = Array.from(new Set([
        String(env.RPC_URL || '').trim(),
        'https://polygon-rpc.com',
        'https://polygon.llamarpc.com',
        'https://rpc.ankr.com/polygon',
    ].filter(Boolean)));
    const matchErrors: string[] = [];
    try {
        const { ethers } = require('ethers');
        const cond = normalizeHex(conditionId);
        for (const rpcUrl of rpcCandidates) {
            try {
                // Static provider avoids ethers network auto-detect failures on RPCs that
                // intermittently reject eth_chainId. The chain is Polygon by contract.
                const provider = new ethers.providers.StaticJsonRpcProvider(rpcUrl, POLYGON_CHAIN_ID);
                const ctf = new ethers.Contract(
                    CTF_ADDRESS,
                    [
                        'function getCollectionId(bytes32 parentCollectionId, bytes32 conditionId, uint256 indexSet) view returns (bytes32)',
                        'function getPositionId(address collateralToken, bytes32 collectionId) view returns (uint256)',
                    ],
                    provider,
                );
                for (const [name, collateral] of [
                    ['pUSD', PUSD_ADDRESS],
                    ['USDC.e', USDC_E_ADDRESS],
                ] as const) {
                    for (const indexSet of [1, 2]) {
                        const collectionId = await ctf.getCollectionId(ZERO_BYTES32, cond, indexSet);
                        const positionId = await ctf.getPositionId(collateral, collectionId);
                        if (String(positionId) === targetAsset) {
                            return { collateralToken: collateral, source: `matched_asset_${name}_index_${indexSet}` };
                        }
                    }
                }
            } catch (error) {
                matchErrors.push(`${rpcUrl}:${String((error as Error)?.message || error)}`.slice(0, 240));
            }
        }
    } catch (error) {
        matchErrors.push(`setup:${String((error as Error)?.message || error)}`.slice(0, 240));
    }

    if (targetAsset) {
        // Some still-redeemable CTF positions were minted against legacy USDC.e
        // even though V2 trading collateral is pUSD. Submitting those with pUSD
        // produces a confirmed on-chain transaction with payout=0, leaving the
        // website claimable amount unchanged. When the asset-id match helper is
        // unavailable, prefer the legacy collateral for asset-backed claims and
        // keep pUSD as the no-asset default below.
        return {
            collateralToken: USDC_E_ADDRESS,
            source: `fallback_legacy_usdce_asset_match_unavailable:${matchErrors.join('|').slice(0, 500)}`,
        };
    }
    return { collateralToken: PUSD_ADDRESS, source: 'default_pusd_no_asset_match' };
}

function normalizeHex(value: string): string {
    return value.startsWith('0x') ? value : `0x${value}`;
}

function normalizeAddress(value: string | undefined): string {
    const raw = String(value || '').trim();
    if (!raw) return '';
    return normalizeHex(raw).toLowerCase();
}

function normalizeRelayerBaseUrl(baseUrl: string | undefined): string {
    const raw = String(baseUrl || 'https://relayer-v2.polymarket.com/').trim();
    return raw.endsWith('/') ? raw.slice(0, -1) : raw;
}

function hasBuilderCreds(env: PredictionEnvConfig): boolean {
    return Boolean(env.BUILDER_API_KEY && env.BUILDER_API_SECRET && env.BUILDER_API_PASSPHRASE);
}

function hasRelayerCreds(env: PredictionEnvConfig): boolean {
    return Boolean(env.RELAYER_API_KEY);
}

function createBuilderHeaders(
    method: string,
    pathWithQuery: string,
    env: PredictionEnvConfig,
    body: string = '',
): Record<string, string> {
    const { BuilderSigner }: any = require('@polymarket/builder-signing-sdk');
    const signer = new BuilderSigner({
        key: env.BUILDER_API_KEY,
        secret: env.BUILDER_API_SECRET,
        passphrase: env.BUILDER_API_PASSPHRASE,
    });
    return signer.createBuilderHeaderPayload(method, pathWithQuery, body);
}

function getAccount(env: PredictionEnvConfig): any {
    const { privateKeyToAccount }: any = require('viem/accounts');
    return privateKeyToAccount(normalizeHex(env.PRIVATE_KEY));
}

function getContractConfig(): any {
    const { getContractConfig }: any = require('@polymarket/builder-relayer-client/dist/config');
    return getContractConfig(POLYGON_CHAIN_ID);
}

async function getWalletClient(env: PredictionEnvConfig): Promise<any> {
    const { createWalletClient, http }: any = require('viem');
    const { polygon }: any = require('viem/chains');
    const account = getAccount(env);
    return createWalletClient({
        account,
        chain: polygon,
        transport: http(env.RPC_URL),
    });
}

async function getPublicClient(env: PredictionEnvConfig): Promise<any> {
    const { createPublicClient, http }: any = require('viem');
    const { polygon }: any = require('viem/chains');
    return createPublicClient({
        chain: polygon,
        transport: http(env.RPC_URL),
    });
}

async function readUsdceBalanceAndAllowanceWithFallback(
    walletAddress: string,
    env: PredictionEnvConfig,
): Promise<{ balanceRaw: bigint; allowanceRaw: bigint; rpcUrl: string }> {
    const rpcCandidates = Array.from(new Set([
        String(env.RPC_URL || '').trim(),
        'https://polygon.llamarpc.com',
        'https://polygon-rpc.com',
    ].filter(Boolean)));
    const { ethers } = require('ethers');
    const abi = [
        'function balanceOf(address account) view returns (uint256)',
        'function allowance(address owner, address spender) view returns (uint256)',
    ];
    let lastError = 'usdce balance read failed';
    for (const rpcUrl of rpcCandidates) {
        try {
            const provider = new ethers.providers.StaticJsonRpcProvider(rpcUrl, POLYGON_CHAIN_ID);
            const token = new ethers.Contract(USDC_E_ADDRESS, abi, provider);
            const balanceRaw = BigInt((await token.balanceOf(walletAddress)).toString());
            const allowanceRaw = BigInt((await token.allowance(walletAddress, COLLATERAL_ONRAMP_ADDRESS)).toString());
            return { balanceRaw, allowanceRaw, rpcUrl };
        } catch (error: any) {
            lastError = `${rpcUrl}: ${String(error?.message || error)}`;
        }
    }
    throw new Error(lastError);
}

async function readPusdBalanceAndAllowanceWithFallback(
    walletAddress: string,
    env: PredictionEnvConfig,
): Promise<{ balanceRaw: bigint; allowanceRaw: bigint; rpcUrl: string }> {
    const rpcCandidates = Array.from(new Set([
        String(env.RPC_URL || '').trim(),
        'https://polygon.llamarpc.com',
        'https://polygon-rpc.com',
    ].filter(Boolean)));
    const { ethers } = require('ethers');
    const abi = [
        'function balanceOf(address account) view returns (uint256)',
        'function allowance(address owner, address spender) view returns (uint256)',
    ];
    const exchangeAddress = '0xE111180000d2663C0091e4f400237545B87B996B';
    let lastError = 'pusd balance read failed';
    for (const rpcUrl of rpcCandidates) {
        try {
            const provider = new ethers.providers.StaticJsonRpcProvider(rpcUrl, POLYGON_CHAIN_ID);
            const token = new ethers.Contract(PUSD_ADDRESS, abi, provider);
            const balanceRaw = BigInt((await token.balanceOf(walletAddress)).toString());
            const allowanceRaw = BigInt((await token.allowance(walletAddress, exchangeAddress)).toString());
            return { balanceRaw, allowanceRaw, rpcUrl };
        } catch (error: any) {
            lastError = `${rpcUrl}: ${String(error?.message || error)}`;
        }
    }
    throw new Error(lastError);
}

function format6(raw: bigint | string | null | undefined): string | null {
    if (raw == null) return null;
    try {
        const value = typeof raw === 'bigint' ? raw : BigInt(String(raw));
        const negative = value < ZERO_BIGINT;
        const abs = negative ? value * BigInt(-1) : value;
        const whole = abs / BigInt(1_000_000);
        const frac = String(abs % BigInt(1_000_000)).padStart(6, '0').replace(/0+$/, '');
        const rendered = frac ? `${whole.toString()}.${frac}` : whole.toString();
        return negative ? `-${rendered}` : rendered;
    } catch {
        return null;
    }
}

export async function inspectFundStatus(env: PredictionEnvConfig): Promise<FundStatusResult> {
    const walletModel = await resolveWalletModel(env);
    const authModes = await resolveClaimAuthModes(env);
    if (!env.PRIVATE_KEY || !env.PROXY_WALLET || !env.RPC_URL) {
        return {
            ok: false,
            status: 'error',
            error: 'missing PRIVATE_KEY / PROXY_WALLET / RPC_URL',
            walletModel: walletModel.selectedWalletModel,
            walletModelReason: walletModel.walletModelReason,
            claimSubmitAuthMode: authModes.claimSubmitAuthMode,
            claimVerifyAuthMode: authModes.claimVerifyAuthMode,
        };
    }
    const walletAddress = walletModel.configuredProfileWallet;
    try {
        let legacyReadError: string | null = null;
        let pusdReadError: string | null = null;
        let legacy = { balanceRaw: ZERO_BIGINT, allowanceRaw: ZERO_BIGINT, rpcUrl: '' };
        let pusd = { balanceRaw: ZERO_BIGINT, allowanceRaw: ZERO_BIGINT, rpcUrl: '' };
        try {
            legacy = await readUsdceBalanceAndAllowanceWithFallback(walletAddress, env);
        } catch (error: any) {
            legacyReadError = String(error?.message || error);
        }
        try {
            pusd = await readPusdBalanceAndAllowanceWithFallback(walletAddress, env);
        } catch (error: any) {
            pusdReadError = String(error?.message || error);
        }

        let clobBalanceRaw: string | null = null;
        let clobAllowanceRaw: string | null = null;
        let balanceSource: string | null = null;
        let allowanceSource: string | null = null;

        try {
            const { Wallet, providers } = require('ethers');
            const { AssetType, Chain, ClobClient, SignatureTypeV2 } = require('@polymarket/clob-client-v2');
            const signer = new Wallet(normalizeHex(env.PRIVATE_KEY));
            const proxyWallet = normalizeAddress(env.PROXY_WALLET);
            const provider = new providers.StaticJsonRpcProvider(env.RPC_URL, POLYGON_CHAIN_ID);
            let signatureType = SignatureTypeV2.EOA;
            let funderAddress = undefined;
            if (proxyWallet && proxyWallet.toLowerCase() !== signer.address.toLowerCase()) {
                let code = '0x';
                try {
                    code = await provider.getCode(proxyWallet);
                } catch {}
                signatureType = code && code !== '0x' ? SignatureTypeV2.POLY_GNOSIS_SAFE : SignatureTypeV2.POLY_PROXY;
                funderAddress = proxyWallet;
            }
            const client = new ClobClient({
                host: 'https://clob.polymarket.com',
                chain: Chain.POLYGON,
                signer,
                creds: {
                    key: env.POLY_API_KEY,
                    secret: env.POLY_API_SECRET,
                    passphrase: env.POLY_API_PASSPHRASE,
                },
                signatureType,
                funderAddress,
            });
            const bal = await client.getBalanceAllowance({ asset_type: AssetType.COLLATERAL });
            clobBalanceRaw = bal?.balance != null ? String(bal.balance) : null;
            if (bal?.allowance != null) {
                clobAllowanceRaw = String(bal.allowance);
            } else if (bal?.allowances && typeof bal.allowances === 'object') {
                const exchangeAddress = '0xE111180000d2663C0091e4f400237545B87B996B';
                const bySpender = bal.allowances as Record<string, unknown>;
                const match = bySpender[exchangeAddress] ?? bySpender[exchangeAddress.toLowerCase()];
                if (match != null) clobAllowanceRaw = String(match);
            }
            balanceSource = clobBalanceRaw && BigInt(clobBalanceRaw) > ZERO_BIGINT ? 'clob_getBalanceAllowance' : 'onchain_pusd_fallback';
            allowanceSource = clobAllowanceRaw && BigInt(clobAllowanceRaw) > ZERO_BIGINT ? 'clob_getBalanceAllowance' : 'onchain_pusd_fallback';
        } catch {}

        const effectiveTradingBalanceRaw =
            clobBalanceRaw && BigInt(clobBalanceRaw) > ZERO_BIGINT ? clobBalanceRaw : pusd.balanceRaw.toString();
        const effectiveTradingAllowanceRaw =
            clobAllowanceRaw && BigInt(clobAllowanceRaw) > ZERO_BIGINT ? clobAllowanceRaw : pusd.allowanceRaw.toString();

        if (!clobBalanceRaw && !clobAllowanceRaw && pusd.balanceRaw <= ZERO_BIGINT && pusd.allowanceRaw <= ZERO_BIGINT && pusdReadError) {
            return {
                ok: false,
                status: 'error',
                error: pusdReadError,
                walletModel: walletModel.selectedWalletModel,
                walletModelReason: walletModel.walletModelReason,
                claimSubmitAuthMode: authModes.claimSubmitAuthMode,
                claimVerifyAuthMode: authModes.claimVerifyAuthMode,
            };
        }

        const legacyBalancePositive = legacy.balanceRaw > ZERO_BIGINT;
        const effectiveBalancePositive = BigInt(effectiveTradingBalanceRaw || '0') > ZERO_BIGINT;
        const effectiveAllowancePositive = BigInt(effectiveTradingAllowanceRaw || '0') > ZERO_BIGINT;
        const pusdOnchainPositive = pusd.balanceRaw > ZERO_BIGINT;

        let status: FundStatusResult['status'] = 'empty';
        let warning: string | undefined;
        if (legacyBalancePositive) {
            status = 'legacy_wrap_required';
            warning = 'legacy_usdce_wrap_required';
        } else if (effectiveBalancePositive && effectiveAllowancePositive) {
            status = 'ready';
        } else if (pusdOnchainPositive || effectiveBalancePositive) {
            status = 'activation_review_required';
            warning = effectiveAllowancePositive ? 'funds_present_but_frontend_not_activated' : 'pusd_allowance_zero_or_pending_activation';
        }

        return {
            ok: true,
            status,
            warning: warning || legacyReadError || pusdReadError || undefined,
            walletModel: walletModel.selectedWalletModel,
            walletModelReason: walletModel.walletModelReason,
            claimSubmitAuthMode: authModes.claimSubmitAuthMode,
            claimVerifyAuthMode: authModes.claimVerifyAuthMode,
            legacyUsdceBalanceRaw: legacy.balanceRaw.toString(),
            legacyUsdceBalanceFormatted: format6(legacy.balanceRaw) || undefined,
            legacyUsdceAllowanceRaw: legacy.allowanceRaw.toString(),
            legacyUsdceAllowanceFormatted: format6(legacy.allowanceRaw) || undefined,
            onchainPusdBalanceRaw: pusd.balanceRaw.toString(),
            onchainPusdBalanceFormatted: format6(pusd.balanceRaw) || undefined,
            onchainPusdAllowanceRaw: pusd.allowanceRaw.toString(),
            onchainPusdAllowanceFormatted: format6(pusd.allowanceRaw) || undefined,
            clobBalanceRaw,
            clobBalanceFormatted: format6(clobBalanceRaw),
            clobAllowanceRaw,
            clobAllowanceFormatted: format6(clobAllowanceRaw),
            effectiveTradingBalanceRaw,
            effectiveTradingBalanceFormatted: format6(effectiveTradingBalanceRaw),
            effectiveTradingAllowanceRaw,
            effectiveTradingAllowanceFormatted: format6(effectiveTradingAllowanceRaw),
            balanceSource,
            allowanceSource,
        };
    } catch (error: any) {
        return {
            ok: false,
            status: 'error',
            error: String(error?.message || error),
            walletModel: walletModel.selectedWalletModel,
            walletModelReason: walletModel.walletModelReason,
            claimSubmitAuthMode: authModes.claimSubmitAuthMode,
            claimVerifyAuthMode: authModes.claimVerifyAuthMode,
        };
    }
}

function getBuilderConfig(env: PredictionEnvConfig): any {
    const { BuilderConfig }: any = require('@polymarket/builder-signing-sdk');
    return new BuilderConfig({
        localBuilderCreds: {
            key: env.BUILDER_API_KEY,
            secret: env.BUILDER_API_SECRET,
            passphrase: env.BUILDER_API_PASSPHRASE,
        },
    });
}

async function createRelayClient(env: PredictionEnvConfig, relayTxType: 'SAFE' | 'PROXY'): Promise<any> {
    const { RelayClient, RelayerTxType }: any = require('@polymarket/builder-relayer-client');
    const walletClient = await getWalletClient(env);
    const builderConfig = getBuilderConfig(env);
    return new RelayClient(
        `${normalizeRelayerBaseUrl(env.RELAYER_BASE_URL)}/`,
        POLYGON_CHAIN_ID,
        walletClient,
        builderConfig,
        relayTxType === 'SAFE' ? RelayerTxType.SAFE : RelayerTxType.PROXY,
    );
}

function getRelayerReadHeaders(env: PredictionEnvConfig): { authMode: ClaimAuthMode; headers: Record<string, string> } {
    if (hasRelayerCreds(env)) {
        const signerEoa = env.PRIVATE_KEY ? normalizeAddress(getAccount(env).address) : '';
        const ownerAddress = normalizeAddress(env.RELAYER_API_KEY_ADDRESS) || signerEoa || normalizeAddress(env.PROXY_WALLET);
        return {
            authMode: 'relayer_key_auth',
            headers: {
                RELAYER_API_KEY: String(env.RELAYER_API_KEY || '').trim(),
                RELAYER_API_KEY_ADDRESS: ownerAddress,
            },
        };
    }
    if (hasBuilderCreds(env)) {
        return {
            authMode: 'builder_auth',
            headers: createBuilderHeaders('GET', '/', env, ''),
        };
    }
    return { authMode: 'unavailable', headers: {} };
}

async function relayerGet(
    path: string,
    params: Record<string, string | number | undefined>,
    env: PredictionEnvConfig,
): Promise<{ ok: boolean; data?: any; error?: string; authMode: ClaimAuthMode }> {
    const axios: any = require('axios');
    const baseUrl = normalizeRelayerBaseUrl(env.RELAYER_BASE_URL);
    const queryEntries = Object.entries(params).filter(([, value]) => value !== undefined && value !== null && String(value).trim() !== '');
    const query = new URLSearchParams(queryEntries.map(([key, value]) => [key, String(value)])).toString();
    const pathWithQuery = query ? `${path}?${query}` : path;
    const authAttempts: Array<{ authMode: ClaimAuthMode; headers: Record<string, string> }> = [];
    if (hasRelayerCreds(env)) {
        const signerEoa = env.PRIVATE_KEY ? normalizeAddress(getAccount(env).address) : '';
        const ownerAddress = normalizeAddress(env.RELAYER_API_KEY_ADDRESS) || signerEoa || normalizeAddress(env.PROXY_WALLET);
        authAttempts.push({
            authMode: 'relayer_key_auth',
            headers: {
                RELAYER_API_KEY: String(env.RELAYER_API_KEY || '').trim(),
                RELAYER_API_KEY_ADDRESS: ownerAddress,
            },
        });
    }
    if (hasBuilderCreds(env)) {
        authAttempts.push({
            authMode: 'builder_auth',
            headers: createBuilderHeaders('GET', pathWithQuery, env, ''),
        });
    }
    if (authAttempts.length === 0) {
        return { ok: false, error: 'missing builder/relayer credentials', authMode: 'unavailable' };
    }
    let lastError = 'unknown relayer read error';
    for (const attempt of authAttempts) {
        try {
            const response = await axios.get(`${baseUrl}${path}`, {
                params,
                headers: attempt.headers,
                timeout: 20_000,
            });
            return { ok: true, data: response.data, authMode: attempt.authMode };
        } catch (error: any) {
            lastError = String(error?.response?.data?.error || error?.response?.data?.message || error?.message || error);
        }
    }
    return { ok: false, error: lastError, authMode: authAttempts[0]?.authMode || 'unavailable' };
}

export async function resolveWalletModel(env: PredictionEnvConfig): Promise<WalletModelResolution> {
    const signerEoa = env.PRIVATE_KEY ? normalizeAddress(getAccount(env).address) : '';
    const configuredProfileWallet = normalizeAddress(env.PROXY_WALLET);
    if (!env.PRIVATE_KEY || !env.PROXY_WALLET || !env.RPC_URL) {
        return {
            signerEoa,
            configuredProfileWallet,
            configuredWalletHasCode: false,
            selectedWalletModel: 'BLOCKED',
            walletModelReason: 'missing PRIVATE_KEY / PROXY_WALLET / RPC_URL',
        };
    }

    const { deriveProxyWallet, deriveSafe }: any = require('@polymarket/builder-relayer-client/dist/builder/derive');
    const { createPublicClient, http }: any = require('viem');
    const { polygon }: any = require('viem/chains');
    const contractConfig = getContractConfig();
    const derivedProxyWallet = normalizeAddress(deriveProxyWallet(signerEoa, contractConfig.ProxyContracts.ProxyFactory));
    const derivedSafeWallet = normalizeAddress(deriveSafe(signerEoa, contractConfig.SafeContracts.SafeFactory));
    const publicClient = createPublicClient({
        chain: polygon,
        transport: http(env.RPC_URL),
    });
    let configuredWalletHasCode = false;
    try {
        const code = await publicClient.getCode({ address: configuredProfileWallet as `0x${string}` });
        configuredWalletHasCode = Boolean(code && String(code) !== '0x');
    } catch {
        configuredWalletHasCode = false;
    }

    if (configuredProfileWallet === signerEoa) {
        return {
            signerEoa,
            configuredProfileWallet,
            derivedSafeWallet,
            derivedProxyWallet,
            configuredWalletHasCode,
            selectedWalletModel: 'EOA',
            walletModelReason: 'configured profile wallet equals signer EOA',
        };
    }
    if (configuredProfileWallet === derivedSafeWallet) {
        return {
            signerEoa,
            configuredProfileWallet,
            derivedSafeWallet,
            derivedProxyWallet,
            configuredWalletHasCode,
            selectedWalletModel: 'SAFE',
            walletModelReason: configuredWalletHasCode
                ? 'configured profile wallet matches derived SAFE wallet and has deployed code'
                : 'configured profile wallet matches derived SAFE wallet but code is absent or deployment state is unknown',
        };
    }
    if (configuredProfileWallet === derivedProxyWallet) {
        return {
            signerEoa,
            configuredProfileWallet,
            derivedSafeWallet,
            derivedProxyWallet,
            configuredWalletHasCode,
            selectedWalletModel: 'PROXY',
            walletModelReason: configuredWalletHasCode
                ? 'configured profile wallet matches derived PROXY wallet but unexpectedly has code'
                : 'configured profile wallet matches derived PROXY wallet',
        };
    }
    return {
        signerEoa,
        configuredProfileWallet,
        derivedSafeWallet,
        derivedProxyWallet,
        configuredWalletHasCode,
        selectedWalletModel: 'BLOCKED',
        walletModelReason: `configured profile wallet ${configuredProfileWallet} matches neither derived SAFE ${derivedSafeWallet} nor derived PROXY ${derivedProxyWallet}`,
    };
}

export async function resolveClaimAuthModes(env: PredictionEnvConfig): Promise<ClaimAuthResolution> {
    return {
        claimSubmitAuthMode: hasBuilderCreds(env) ? 'builder_auth' : 'unavailable',
        claimVerifyAuthMode: hasRelayerCreds(env) ? 'relayer_key_auth' : (hasBuilderCreds(env) ? 'builder_auth' : 'unavailable'),
    };
}

export async function fetchRecentTransactionsForUser(
    user: string,
    env: PredictionEnvConfig,
    limit: number = 100,
): Promise<RelayerRecentTransactionsResult> {
    const normalizedUser = normalizeAddress(user);
    const response = await relayerGet(RELAYER_GET_TRANSACTIONS, {}, env);
    if (!response.ok) {
        return { ok: false, error: response.error, authMode: response.authMode };
    }
    const rows = Array.isArray(response.data) ? response.data : [];
    const filtered = rows.filter((row: any) => {
        const from = normalizeAddress(String(row?.from || ''));
        const proxy = normalizeAddress(String(row?.proxyAddress || ''));
        const to = normalizeAddress(String(row?.to || ''));
        return from === normalizedUser || proxy === normalizedUser || to === normalizedUser;
    });
    return {
        ok: true,
        transactions: filtered.slice(0, Math.max(1, Math.min(500, Math.trunc(limit) || 100))),
        authMode: response.authMode,
    };
}

export async function fetchTransactionById(
    transactionId: string,
    env: PredictionEnvConfig,
): Promise<RelayerTransactionResult> {
    const normalizedTransactionId = String(transactionId || '').trim();
    if (!normalizedTransactionId) {
        return { ok: false, error: 'missing transaction id', authMode: 'unavailable' };
    }
    const response = await relayerGet(RELAYER_GET_TRANSACTION, { id: normalizedTransactionId }, env);
    if (!response.ok) {
        return { ok: false, error: response.error, authMode: response.authMode };
    }
    const row = Array.isArray(response.data) ? (response.data[0] || null) : (response.data && typeof response.data === 'object' ? response.data : null);
    return { ok: true, transaction: row, authMode: response.authMode };
}

export async function fetchSafeDeploymentStatus(env: PredictionEnvConfig): Promise<ClaimWalletDeploymentResult> {
    const walletModel = await resolveWalletModel(env);
    const authModes = await resolveClaimAuthModes(env);
    if (walletModel.selectedWalletModel !== 'SAFE') {
        return {
            ok: walletModel.selectedWalletModel !== 'BLOCKED',
            deployed: false,
            walletModel: walletModel.selectedWalletModel,
            walletModelReason: walletModel.walletModelReason,
            authMode: authModes.claimVerifyAuthMode,
            error: walletModel.selectedWalletModel === 'BLOCKED' ? walletModel.walletModelReason : 'wallet model is not SAFE',
        };
    }
    const response = await relayerGet(RELAYER_GET_DEPLOYED, { address: walletModel.configuredProfileWallet }, env);
    if (!response.ok) {
        return {
            ok: false,
            deployed: false,
            walletModel: walletModel.selectedWalletModel,
            walletModelReason: walletModel.walletModelReason,
            authMode: response.authMode,
            error: response.error,
        };
    }
    return {
        ok: true,
        deployed: Boolean(response.data?.deployed),
        walletModel: walletModel.selectedWalletModel,
        walletModelReason: walletModel.walletModelReason,
        authMode: response.authMode,
    };
}

export async function deploySafeWalletGasless(env: PredictionEnvConfig): Promise<GaslessClaimResult> {
    const walletModel = await resolveWalletModel(env);
    const authModes = await resolveClaimAuthModes(env);
    if (walletModel.selectedWalletModel !== 'SAFE') {
        return {
            ok: false,
            error: walletModel.walletModelReason,
            walletModel: walletModel.selectedWalletModel,
            walletModelReason: walletModel.walletModelReason,
            claimSubmitAuthMode: authModes.claimSubmitAuthMode,
            claimVerifyAuthMode: authModes.claimVerifyAuthMode,
        };
    }
    if (authModes.claimSubmitAuthMode !== 'builder_auth') {
        return {
            ok: false,
            error: 'missing BUILDER_API_* credentials for gasless SAFE deploy',
            walletModel: walletModel.selectedWalletModel,
            walletModelReason: walletModel.walletModelReason,
            claimSubmitAuthMode: authModes.claimSubmitAuthMode,
            claimVerifyAuthMode: authModes.claimVerifyAuthMode,
        };
    }
    try {
        const relayerClient = await createRelayClient(env, 'SAFE');
        const deployed = await relayerClient.getDeployed(walletModel.configuredProfileWallet);
        if (deployed) {
            return {
                ok: true,
                confirmed: true,
                state: 'STATE_CONFIRMED',
                walletModel: walletModel.selectedWalletModel,
                walletModelReason: walletModel.walletModelReason,
                claimSubmitAuthMode: authModes.claimSubmitAuthMode,
                claimVerifyAuthMode: authModes.claimVerifyAuthMode,
                warning: 'safe already deployed',
            };
        }
        const response = await relayerClient.deploy();
        const transactionId = String(response?.transactionID || '').trim();
        const txHash = String(response?.transactionHash || response?.hash || '').trim();
        const state = String(response?.state || '').trim() || 'STATE_NEW';
        if (!transactionId && !txHash) {
            return {
                ok: false,
                error: 'safe deploy returned no transaction id/hash',
                walletModel: walletModel.selectedWalletModel,
                walletModelReason: walletModel.walletModelReason,
                claimSubmitAuthMode: authModes.claimSubmitAuthMode,
                claimVerifyAuthMode: authModes.claimVerifyAuthMode,
            };
        }
        return {
            ok: true,
            txHash: txHash || undefined,
            transactionId: transactionId || undefined,
            state,
            confirmed: state === 'STATE_MINED' || state === 'STATE_CONFIRMED',
            walletModel: walletModel.selectedWalletModel,
            walletModelReason: walletModel.walletModelReason,
            claimSubmitAuthMode: authModes.claimSubmitAuthMode,
            claimVerifyAuthMode: authModes.claimVerifyAuthMode,
        };
    } catch (error: any) {
        return {
            ok: false,
            error: String(error?.message || error),
            walletModel: walletModel.selectedWalletModel,
            walletModelReason: walletModel.walletModelReason,
            claimSubmitAuthMode: authModes.claimSubmitAuthMode,
            claimVerifyAuthMode: authModes.claimVerifyAuthMode,
        };
    }
}

export async function redeemPositionsGasless(
    conditionId: string,
    env: PredictionEnvConfig,
    assetId?: string,
    collateralTokenOverride?: string,
): Promise<GaslessClaimResult> {
    const walletModel = await resolveWalletModel(env);
    const authModes = await resolveClaimAuthModes(env);
    if (!env.PRIVATE_KEY || !env.RPC_URL) {
        return {
            ok: false,
            error: 'missing PRIVATE_KEY or RPC_URL',
            walletModel: walletModel.selectedWalletModel,
            walletModelReason: walletModel.walletModelReason,
            claimSubmitAuthMode: authModes.claimSubmitAuthMode,
            claimVerifyAuthMode: authModes.claimVerifyAuthMode,
        };
    }
    if (!env.PROXY_WALLET) {
        return {
            ok: false,
            error: 'missing PROXY_WALLET for gasless redeem',
            walletModel: walletModel.selectedWalletModel,
            walletModelReason: walletModel.walletModelReason,
            claimSubmitAuthMode: authModes.claimSubmitAuthMode,
            claimVerifyAuthMode: authModes.claimVerifyAuthMode,
        };
    }
    if (authModes.claimSubmitAuthMode !== 'builder_auth') {
        return {
            ok: false,
            error: 'missing BUILDER_API_* credentials for gasless redeem submit',
            walletModel: walletModel.selectedWalletModel,
            walletModelReason: walletModel.walletModelReason,
            claimSubmitAuthMode: authModes.claimSubmitAuthMode,
            claimVerifyAuthMode: authModes.claimVerifyAuthMode,
        };
    }
    if (walletModel.selectedWalletModel === 'BLOCKED') {
        return {
            ok: false,
            error: walletModel.walletModelReason,
            walletModel: walletModel.selectedWalletModel,
            walletModelReason: walletModel.walletModelReason,
            claimSubmitAuthMode: authModes.claimSubmitAuthMode,
            claimVerifyAuthMode: authModes.claimVerifyAuthMode,
        };
    }
    if (walletModel.selectedWalletModel === 'EOA') {
        return {
            ok: false,
            error: 'gasless redeem requires a SAFE or PROXY wallet model, not EOA',
            walletModel: walletModel.selectedWalletModel,
            walletModelReason: walletModel.walletModelReason,
            claimSubmitAuthMode: authModes.claimSubmitAuthMode,
            claimVerifyAuthMode: authModes.claimVerifyAuthMode,
        };
    }

    try {
        const relayTxType = walletModel.selectedWalletModel === 'SAFE' ? 'SAFE' : 'PROXY';
        const relayerClient = await createRelayClient(env, relayTxType);
        if (relayTxType === 'SAFE') {
            const deployed = await relayerClient.getDeployed(walletModel.configuredProfileWallet);
            if (!deployed) {
                return {
                    ok: false,
                    error: `safe wallet ${walletModel.configuredProfileWallet} is not deployed`,
                    walletModel: walletModel.selectedWalletModel,
                    walletModelReason: walletModel.walletModelReason,
                    claimSubmitAuthMode: authModes.claimSubmitAuthMode,
                    claimVerifyAuthMode: authModes.claimVerifyAuthMode,
                };
            }
        }

        const { encodeFunctionData }: any = require('viem');
        const collateral = await resolveRedeemCollateralToken(conditionId, env, assetId, collateralTokenOverride);
        const tx = {
            to: CTF_ADDRESS,
            data: encodeFunctionData({
                abi: CTF_REDEEM_ABI,
                functionName: 'redeemPositions',
                args: [collateral.collateralToken, ZERO_BYTES32, normalizeHex(conditionId), [1, 2]],
            }),
            value: '0',
        };

        const response = await relayerClient.execute([tx], `redeem positions ${conditionId.slice(0, 12)}`);
        const transactionId = String(response?.transactionID || '').trim();
        const responseTxHash = String(response?.transactionHash || response?.hash || '').trim();
        const relayerState = String(response?.state || '').trim();
        const txHash = String(responseTxHash || '').trim();
        if (!transactionId && !txHash) {
            return {
                ok: false,
                error: 'gasless redeem submission returned no transaction id/hash',
                walletModel: walletModel.selectedWalletModel,
                walletModelReason: walletModel.walletModelReason,
                claimSubmitAuthMode: authModes.claimSubmitAuthMode,
                claimVerifyAuthMode: authModes.claimVerifyAuthMode,
            };
        }
        const normalizedState = relayerState || 'STATE_NEW';
        return {
            ok: true,
            txHash: txHash || undefined,
            transactionId: transactionId || undefined,
            state: normalizedState,
            confirmed: normalizedState === 'STATE_MINED' || normalizedState === 'STATE_CONFIRMED',
            walletModel: walletModel.selectedWalletModel,
            walletModelReason: walletModel.walletModelReason,
            claimSubmitAuthMode: authModes.claimSubmitAuthMode,
            claimVerifyAuthMode: authModes.claimVerifyAuthMode,
            collateralToken: collateral.collateralToken,
            collateralTokenSource: collateral.source,
        };
    } catch (error: any) {
        return {
            ok: false,
            error: String(error?.message || error),
            walletModel: walletModel.selectedWalletModel,
            walletModelReason: walletModel.walletModelReason,
            claimSubmitAuthMode: authModes.claimSubmitAuthMode,
            claimVerifyAuthMode: authModes.claimVerifyAuthMode,
        };
    }
}

export async function wrapLegacyUsdceToPusdGasless(
    env: PredictionEnvConfig,
    amountOverrideRaw?: string,
): Promise<GaslessClaimResult> {
    const walletModel = await resolveWalletModel(env);
    const authModes = await resolveClaimAuthModes(env);
    if (!env.PRIVATE_KEY || !env.RPC_URL) {
        return {
            ok: false,
            error: 'missing PRIVATE_KEY or RPC_URL',
            walletModel: walletModel.selectedWalletModel,
            walletModelReason: walletModel.walletModelReason,
            claimSubmitAuthMode: authModes.claimSubmitAuthMode,
            claimVerifyAuthMode: authModes.claimVerifyAuthMode,
        };
    }
    if (!env.PROXY_WALLET) {
        return {
            ok: false,
            error: 'missing PROXY_WALLET for gasless wrap',
            walletModel: walletModel.selectedWalletModel,
            walletModelReason: walletModel.walletModelReason,
            claimSubmitAuthMode: authModes.claimSubmitAuthMode,
            claimVerifyAuthMode: authModes.claimVerifyAuthMode,
        };
    }
    if (authModes.claimSubmitAuthMode !== 'builder_auth') {
        return {
            ok: false,
            error: 'missing BUILDER_API_* credentials for gasless wrap',
            walletModel: walletModel.selectedWalletModel,
            walletModelReason: walletModel.walletModelReason,
            claimSubmitAuthMode: authModes.claimSubmitAuthMode,
            claimVerifyAuthMode: authModes.claimVerifyAuthMode,
        };
    }
    if (walletModel.selectedWalletModel === 'BLOCKED') {
        return {
            ok: false,
            error: walletModel.walletModelReason,
            walletModel: walletModel.selectedWalletModel,
            walletModelReason: walletModel.walletModelReason,
            claimSubmitAuthMode: authModes.claimSubmitAuthMode,
            claimVerifyAuthMode: authModes.claimVerifyAuthMode,
        };
    }
    if (walletModel.selectedWalletModel === 'EOA') {
        return {
            ok: false,
            error: 'gasless wrap requires a SAFE or PROXY wallet model, not EOA',
            walletModel: walletModel.selectedWalletModel,
            walletModelReason: walletModel.walletModelReason,
            claimSubmitAuthMode: authModes.claimSubmitAuthMode,
            claimVerifyAuthMode: authModes.claimVerifyAuthMode,
        };
    }

    try {
        const { encodeFunctionData, formatUnits }: any = require('viem');
        const walletAddress = walletModel.configuredProfileWallet as `0x${string}`;
        const onchain = await readUsdceBalanceAndAllowanceWithFallback(walletAddress, env);
        const legacyBalanceRaw = onchain.balanceRaw;
        const legacyBalanceFormatted = String(formatUnits(legacyBalanceRaw, 6));
        if (legacyBalanceRaw <= ZERO_BIGINT) {
            return {
                ok: true,
                confirmed: true,
                state: 'STATE_CONFIRMED',
                warning: 'legacy_usdce_balance_zero',
                walletModel: walletModel.selectedWalletModel,
                walletModelReason: walletModel.walletModelReason,
                claimSubmitAuthMode: authModes.claimSubmitAuthMode,
                claimVerifyAuthMode: authModes.claimVerifyAuthMode,
                amountRaw: '0',
                amountFormatted: '0',
                legacyUsdceBalanceRaw: '0',
                legacyUsdceBalanceFormatted: legacyBalanceFormatted,
            };
        }

        let wrapAmountRaw = legacyBalanceRaw;
        const override = String(amountOverrideRaw || '').trim();
        if (override) {
            try {
                const parsed = BigInt(override);
                if (parsed > ZERO_BIGINT) {
                    wrapAmountRaw = parsed > legacyBalanceRaw ? legacyBalanceRaw : parsed;
                }
            } catch {
                return {
                    ok: false,
                    error: `invalid wrap amount raw: ${override}`,
                    walletModel: walletModel.selectedWalletModel,
                    walletModelReason: walletModel.walletModelReason,
                    claimSubmitAuthMode: authModes.claimSubmitAuthMode,
                    claimVerifyAuthMode: authModes.claimVerifyAuthMode,
                    legacyUsdceBalanceRaw: legacyBalanceRaw.toString(),
                    legacyUsdceBalanceFormatted: legacyBalanceFormatted,
                };
            }
        }
        if (wrapAmountRaw <= ZERO_BIGINT) {
            return {
                ok: true,
                confirmed: true,
                state: 'STATE_CONFIRMED',
                warning: 'wrap_amount_zero',
                walletModel: walletModel.selectedWalletModel,
                walletModelReason: walletModel.walletModelReason,
                claimSubmitAuthMode: authModes.claimSubmitAuthMode,
                claimVerifyAuthMode: authModes.claimVerifyAuthMode,
                amountRaw: '0',
                amountFormatted: '0',
                legacyUsdceBalanceRaw: legacyBalanceRaw.toString(),
                legacyUsdceBalanceFormatted: legacyBalanceFormatted,
            };
        }

        const currentAllowanceRaw = onchain.allowanceRaw;

        const relayTxType = walletModel.selectedWalletModel === 'SAFE' ? 'SAFE' : 'PROXY';
        const relayerClient = await createRelayClient(env, relayTxType);
        if (relayTxType === 'SAFE') {
            const deployed = await relayerClient.getDeployed(walletModel.configuredProfileWallet);
            if (!deployed) {
                return {
                    ok: false,
                    error: `safe wallet ${walletModel.configuredProfileWallet} is not deployed`,
                    walletModel: walletModel.selectedWalletModel,
                    walletModelReason: walletModel.walletModelReason,
                    claimSubmitAuthMode: authModes.claimSubmitAuthMode,
                    claimVerifyAuthMode: authModes.claimVerifyAuthMode,
                    legacyUsdceBalanceRaw: legacyBalanceRaw.toString(),
                    legacyUsdceBalanceFormatted: legacyBalanceFormatted,
                };
            }
        }

        const txs: Array<{ to: string; data: string; value: string }> = [];
        if (currentAllowanceRaw < wrapAmountRaw) {
            txs.push({
                to: USDC_E_ADDRESS,
                data: encodeFunctionData({
                    abi: ERC20_ABI,
                    functionName: 'approve',
                    args: [COLLATERAL_ONRAMP_ADDRESS, wrapAmountRaw],
                }),
                value: '0',
            });
        }
        txs.push({
            to: COLLATERAL_ONRAMP_ADDRESS,
            data: encodeFunctionData({
                abi: COLLATERAL_ONRAMP_ABI,
                functionName: 'wrap',
                args: [USDC_E_ADDRESS, walletAddress, wrapAmountRaw],
            }),
            value: '0',
        });

        const response = await relayerClient.execute(txs, `wrap legacy usdce -> pusd ${String(formatUnits(wrapAmountRaw, 6))}`);
        const transactionId = String(response?.transactionID || '').trim();
        const txHash = String(response?.transactionHash || response?.hash || '').trim();
        const relayerState = String(response?.state || '').trim() || 'STATE_NEW';
        if (!transactionId && !txHash) {
            return {
                ok: false,
                error: 'gasless wrap submission returned no transaction id/hash',
                walletModel: walletModel.selectedWalletModel,
                walletModelReason: walletModel.walletModelReason,
                claimSubmitAuthMode: authModes.claimSubmitAuthMode,
                claimVerifyAuthMode: authModes.claimVerifyAuthMode,
                amountRaw: wrapAmountRaw.toString(),
                amountFormatted: String(formatUnits(wrapAmountRaw, 6)),
                legacyUsdceBalanceRaw: legacyBalanceRaw.toString(),
                legacyUsdceBalanceFormatted: legacyBalanceFormatted,
            };
        }
        return {
            ok: true,
            txHash: txHash || undefined,
            transactionId: transactionId || undefined,
            state: relayerState,
            confirmed: relayerState === 'STATE_MINED' || relayerState === 'STATE_CONFIRMED',
            walletModel: walletModel.selectedWalletModel,
            walletModelReason: walletModel.walletModelReason,
            claimSubmitAuthMode: authModes.claimSubmitAuthMode,
            claimVerifyAuthMode: authModes.claimVerifyAuthMode,
            amountRaw: wrapAmountRaw.toString(),
            amountFormatted: String(formatUnits(wrapAmountRaw, 6)),
            legacyUsdceBalanceRaw: legacyBalanceRaw.toString(),
            legacyUsdceBalanceFormatted: legacyBalanceFormatted,
        };
    } catch (error: any) {
        return {
            ok: false,
            error: String(error?.message || error),
            walletModel: walletModel.selectedWalletModel,
            walletModelReason: walletModel.walletModelReason,
            claimSubmitAuthMode: authModes.claimSubmitAuthMode,
            claimVerifyAuthMode: authModes.claimVerifyAuthMode,
        };
    }
}
