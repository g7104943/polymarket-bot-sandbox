# polyfun-next Research Components

> Public learning export: this directory is provided for study, testing, and code review only. Automated live Polymarket trading is intentionally disabled. Do not add private keys, wallet files, real credentials, or funds.

`polyfun-next` contains Python research code, candidate-generation experiments, risk-control prototypes, and tests extracted from a private local workspace. The private workspace included live runtime state and real-trading artifacts; this public version intentionally excludes them.

## Included

- Python package source under `src/`.
- Research and dry-run scripts under `scripts/`.
- Unit tests under `tests/`.
- Example config files under `config/*.example.json`.

## Excluded

- `runtime/`, live ledgers, official order records, claim state, PID files, WebSocket state, logs, and model caches.
- Real `canary.eth15m.json` and any private live configuration.
- Private keys, wallet files, API credentials, and local data.

## Safe Usage

```bash
python3 -m unittest discover -s tests
python3 scripts/preflight_v2.py --config config/canary.eth15m.example.json
python3 scripts/run_canary_once.py --config config/canary.eth15m.example.json --dry-run
```

Keep all experimentation in dry-run or simulation mode. This export is not a production trading system and is not financial advice.
