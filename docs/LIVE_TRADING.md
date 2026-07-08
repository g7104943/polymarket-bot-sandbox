# Optional Live Polymarket Trading

This repository can be configured to submit real Polymarket orders, but live mode is intentionally opt-in and unsafe for casual use. Treat this as a self-custody trading template, not a profitable strategy or managed bot.

## Before You Start

You are responsible for all consequences of live trading:

- Real funds can be lost quickly.
- A leaked private key means the wallet should be treated as compromised.
- Polymarket API behavior, market rules, regional restrictions, and legal obligations can change.
- Bugs, stale predictions, rejected orders, partial fills, or duplicate orders can create losses.
- This public repository does not include private live ledgers, raw market data, runtime state, or a production monitoring stack.

Use a dedicated wallet with only the funds you are willing to risk. Do not use your main wallet.

## What Official/Community Projects Show

Polymarket's own CLOB examples require a private key and optional funder/proxy address for trading, and they recommend storing secrets in environment variables. The public Polymarket authentication docs also explain that authenticated CLOB trading signs requests with wallet/API credentials. Several public community bots expose similar `.env` based setup flows. At the same time, malicious Polymarket bot repositories have been used to steal wallet keys, so inspect dependencies and code before entering any secret.

## Local Setup

From the repository root:

```bash
npm install
cp .env.example .env
```

Edit `.env` locally. Never commit it.

Minimum live-mode fields:

```dotenv
TRADING_MODE=live
POLYFUN_ENABLE_LIVE_TRADING=true
POLYFUN_I_UNDERSTAND_LIVE_RISK=YES
POLYFUN_I_WILL_NOT_COMMIT_KEYS=YES

PROXY_WALLET=0xYourPolymarketFunderOrProxyAddress
PRIVATE_KEY=0xyour_private_key_here
RPC_URL=https://your_polygon_rpc_url

# Optional if you already have CLOB API credentials.
POLY_API_KEY=
POLY_API_SECRET=
POLY_API_PASSPHRASE=
```

Recommended safety limits for first tests:

```dotenv
ALLOWED_MARKETS=ETH
TIMEFRAME=15m
BET_SIZE_PERCENT=1
MAX_TRADES_PER_SESSION=1
RISK_CONTROL_ENABLED=true
LOGS_DIR=logs_live_local
```

## Run Simulation First

```bash
npm run start:simulation
```

Only continue if simulation behaves as expected and you understand what file or service is supplying predictions.

## Run Live Mode

```bash
npm run start:trading
```

The process refuses to start unless `TRADING_MODE=live` and all three live confirmation flags are present. If it starts, it may submit real orders when its configured signal path produces a qualifying trade.

## Verify Secrets Are Not Committed

Before every commit:

```bash
git status --short
git ls-files | grep -E '(^|/)\.env$|wallet|ledger|runtime|logs|\.parquet$|\.pkl$|\.db$|\.sqlite$' && echo "STOP" || echo "OK"
rg -n "0x[a-fA-F0-9]{64}|PRIVATE_KEY=.+|MNEMONIC=.+|PASSWORD=.+|SECRET=.+|API_KEY=.+" . --glob '!package-lock.json' --glob '!SANITIZED_EXPORT_MANIFEST.md'
```

Expected result: no real secret values. Example placeholders are acceptable; real keys are not.

## Safer Operating Rules

- Keep live trading disabled by default in public forks.
- Use a dedicated wallet and a tiny balance.
- Start with one trade per session.
- Keep logs local and out of Git.
- Never paste `.env` contents into GitHub issues or screenshots.
- Review all order parameters before trusting automation.
- Stop immediately if orders are rejected, duplicated, or appear in the wrong market.

