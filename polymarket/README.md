# Polymarket Research Components

> Public learning export: automated live Polymarket trading is intentionally disabled in this repository. Do not add private keys, wallet files, real credentials, or funds.

This directory contains experimental TypeScript and Python components that were used to study prediction-market data access, simulation flows, and execution-risk ideas. The original private workspace contained live-operation tooling, credentials, ledgers, and runtime state; those files are not included here.

## What Is Included

- Research and simulation-oriented source code.
- Example environment templates with blank credential fields.
- Utility code for inspecting market data and historical behavior.

## What Is Not Included

- Real wallets or private keys.
- Real order ledgers, fills, claims, or balances.
- Runtime state, logs, model artifacts, or local data caches.
- A supported live trading workflow.

## Public Export Guard

The package scripts that previously implied live trading have been replaced with a hard failure message. The environment parser also rejects `TRADING_MODE=live` in this sanitized export.

Use this directory for code reading, simulation, and educational experiments only.
