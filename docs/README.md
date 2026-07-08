# Documentation Notes

This folder contains a mix of current public-export documentation and older research notes from the private workspace.

## Current Public Export

The current public repository includes only one bundled learning model:

- `ETH/USDT` on the `15m` timeframe
- `data/models/lightgbm_ETH_USDT_15m_20260121_014635/model.joblib`
- `data/models/lightgbm_ETH_USDT_15m_20260121_014635/metadata.json`

Start with:

- [Public Model Pack](PUBLIC_MODEL_PACK.md)
- [Repository README](../README.md)
- [Sanitized Export Manifest](../SANITIZED_EXPORT_MANIFEST.md)

## Legacy Research Notes

Some older notes mention BTC, SOL, XRP, 1h/4h models, GRU experiments, local parquet data, private report paths, or historical simulation workflows. Those notes are preserved for learning context only. The referenced data, private/live models, runtime state, and real trading artifacts are intentionally not included in this public repository.

If a legacy note conflicts with the root README or `PUBLIC_MODEL_PACK.md`, treat the root README and `PUBLIC_MODEL_PACK.md` as the current public-export truth.

## Safety Boundary

This repository is an AI-assisted learning/research export. It is not a production trading system, not financial advice, and must not be used to place automated Polymarket orders.
