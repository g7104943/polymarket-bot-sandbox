# Sanitized Export Manifest

- Export time: 2026-07-06 18:31:09 CST
- Source directory: /Users/mac/polyfun
- Output directory: /Users/mac/polyfun-github-lite
- Visibility target: public GitHub-safe
- Git history: fresh repository only; original history intentionally not copied

## Included

- Root source, scripts, docs, tests, dependency manifests, and safe config files.
- polyfun-next source, scripts, docs, tests, pyproject, and `config/*.example.json` only.
- polymarket source, scripts, docs, dependency manifests, and example environment files only.

## Excluded

- Real `.env` files, wallet files, private keys, credentials, live configs, runtime state, ledgers, official order/trade records, claim state, WebSocket state, PID files.
- Raw data, feature caches, reports, logs, trained model artifacts, notebook-like outputs, archives, virtual environments, and node_modules.
- Original `.git` directory and commit history.

## Required Verification

Run these before pushing:

```bash
cd /Users/mac/polyfun-github-lite
find . -type f -size +5M -print
rg -n --hidden -i '(PRIVATE_KEY|MNEMONIC|SECRET|API_KEY|PASSWORD|CREDENTIAL|WALLET|CLOB|0x[a-f0-9]{64})' .
git ls-files | rg '(^|/)(data|runtime|reports|logs|cache|archive|node_modules|\.venv)(/|$)|\.env$|wallet|ledger|\.jsonl$|\.parquet$|\.pkl$|\.joblib$|\.cbm$|\.db$|\.sqlite$'
```

Any hit must be reviewed before upload. Placeholder-only documentation is allowed only when it contains no real secret value.

## Final Sanitization Notes

- Final verification time: 2026-07-06 19:02:57 
- Final export size: about 9.8 MB.
- Removed live/keychain/ops helper scripts from the public export.
- Disabled npm live trading entrypoints with hard-fail messages.
- Added a public-export guard in `polymarket/src/config/prediction_env.ts` that rejects `TRADING_MODE=live` during config creation.
- Rewrote public-facing README files to state this is an AI-assisted learning/research export and must not be used for automated Polymarket ordering.
- Removed live setup docs and real-order/runtime oriented artifacts.

## Verification Performed

```text
polyfun-next tests: 79 passed
root tests: 5 passed
live npm entrypoints: disabled with exit code 1
large files >5MB: none
forbidden data/model/runtime artifacts: none
secret value pattern scan: no private-key/JWT/PEM-style values found
```

TypeScript build was not run because this lightweight export intentionally does not include `node_modules`, and `tsc` is not installed globally in the current shell.
