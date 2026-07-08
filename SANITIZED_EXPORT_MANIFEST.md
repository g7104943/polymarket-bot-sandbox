# Sanitized Export Manifest

- Export time: 2026-07-06 18:31:09 CST
- Source directory: /Users/mac/polyfun
- Output directory: /Users/mac/polyfun-github-lite
- Visibility target: public GitHub-safe
- Git history: fresh repository only; original history intentionally not copied

## Included

- Root source, scripts, docs, tests, dependency manifests, and safe config files.
- A single audited public ETH 15m LightGBM learning model under `data/models`:
  - `ETH/USDT`
  - `15m`
  - `model.joblib` plus `metadata.json` for the model
- polyfun-next source, scripts, docs, tests, pyproject, and `config/*.example.json` only.
- polymarket source, scripts, docs, dependency manifests, and example environment files only.

## Excluded

- Real `.env` files, wallet files, private keys, credentials, live configs, runtime state, ledgers, official order/trade records, claim state, WebSocket state, PID files.
- Raw data, feature caches, reports, logs, private/live trained model artifacts, notebook-like outputs, archives, virtual environments, and node_modules.
- Any model artifacts outside the allowlisted public learning model pack.
- Original `.git` directory and commit history.

## Required Verification

Run these before pushing:

```bash
cd /Users/mac/polyfun-github-lite
find . -type f -size +5M -print
rg -n --hidden -i '(PRIVATE_KEY|MNEMONIC|SECRET|API_KEY|PASSWORD|CREDENTIAL|WALLET|CLOB|0x[a-f0-9]{64})' .
python scripts/verify_public_model_pack.py
git ls-files | rg '(^|/)(runtime|reports|logs|cache|archive|node_modules|\.venv)(/|$)|\.env$|wallet|ledger|\.jsonl$|\.parquet$|\.pkl$|\.cbm$|\.db$|\.sqlite$'
git ls-files data | rg -v '^data/models/lightgbm_ETH_USDT_15m_20260121_014635/(model\.joblib|metadata\.json)$'
```

Any hit must be reviewed before upload. Placeholder-only documentation is allowed only when it contains no real secret value.

## Final Sanitization Notes

- Final verification time: 2026-07-06 19:02:57 
- Current export size after the ETH-only model update: about 13.2 MiB excluding local `.git` metadata.
- Removed live/keychain/ops helper scripts from the public export.
- Replaced hard-disabled npm live entrypoints with explicit local opt-in live guards.
- Added a live-mode guard in `polymarket/src/config/prediction_env.ts` that rejects `TRADING_MODE=live` unless the user sets all required local confirmation flags.
- Rewrote public-facing README files to state this is an AI-assisted learning/research export with optional self-custody live mode, not financial advice or a production trading system.
- Removed private live setup artifacts, ledgers, runtime state, and real-order records. Added `docs/LIVE_TRADING.md` with local-only secret handling and risk warnings.

## Public Model Pack Update

- Update time: 2026-07-08 CST.
- Reduced the public model pack to one audited ETH/USDT 15m LightGBM learning model under `data/models`.
- Model pack size: about 4.4 MB.
- Individual model files are below 5 MB.
- Model metadata and binary strings were scanned for private-key, mnemonic, password, token, wallet, PEM key, and local-path patterns.
- The pack is documented in `docs/PUBLIC_MODEL_PACK.md` and checked by `scripts/verify_public_model_pack.py`.
- This is a public learning example, not a private live-trading model.

## Verification Performed

```text
polyfun-next tests: 79 passed
root tests: 6 passed
live npm entrypoints: require explicit local opt-in flags before startup
large files >5MB: none
forbidden data/model/runtime artifacts: none; only the allowlisted ETH 15m model is tracked
secret value pattern scan: no private-key/JWT/PEM-style values found
```

TypeScript build was not run because this lightweight export intentionally does not include `node_modules`, and `tsc` is not installed globally in the current shell.
