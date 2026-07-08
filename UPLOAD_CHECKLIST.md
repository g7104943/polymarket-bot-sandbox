# Upload Checklist

Before pushing this repository to GitHub:

- [ ] Confirm this is the sanitized export, not the live project: `/Users/mac/polyfun-github-lite`.
- [ ] Confirm no real `.env`, wallet, private key, mnemonic, ledger, runtime, raw data, logs, reports, or private/live model artifacts are present.
- [ ] Confirm the only uploaded model artifacts are the audited public learning models documented in `docs/PUBLIC_MODEL_PACK.md`.
- [ ] Run `git status --short` and review every file to be uploaded.
- [ ] Run the sensitive keyword scan documented in `SANITIZED_EXPORT_MANIFEST.md`.
- [ ] Confirm example configs keep `live_enabled=false`.
- [ ] Run `python scripts/verify_public_model_pack.py`.
- [ ] Confirm no file over 5MB is present unless manually approved. The current public model pack is expected to stay below that threshold per file.
- [ ] Do not copy old `.git` history from the live project.

This export is intended for public GitHub code sharing only. It includes small public learning models, but it is not a runnable live-trading deployment and must not be connected to real funds.
