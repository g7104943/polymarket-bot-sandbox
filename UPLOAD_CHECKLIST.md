# Upload Checklist

Before pushing this repository to GitHub:

- [ ] Confirm this is the sanitized export, not the live project: `/Users/mac/polyfun-github-lite`.
- [ ] Confirm no real `.env`, wallet, private key, mnemonic, ledger, runtime, data, logs, reports, or model artifacts are present.
- [ ] Run `git status --short` and review every file to be uploaded.
- [ ] Run the sensitive keyword scan documented in `SANITIZED_EXPORT_MANIFEST.md`.
- [ ] Confirm example configs keep `live_enabled=false`.
- [ ] Confirm no file over 5MB is present unless manually approved.
- [ ] Do not copy old `.git` history from the live project.

This export is intended for public GitHub code sharing only. It is not a runnable live-trading deployment without private environment variables, market data, and locally generated models.
