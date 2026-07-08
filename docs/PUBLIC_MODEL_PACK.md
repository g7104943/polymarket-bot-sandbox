# Public Model Pack

This repository includes a single audited ETH 15m model so readers can load the predictor without retraining models first.

## Purpose

The model pack is for learning, code review, and offline experimentation. It is not a production trading model pack, it is not financial advice, and it must not be used to place automated Polymarket orders.

## Included Model

The pack contains one LightGBM model:

| Symbol | Timeframe |
| --- | --- |
| `ETH/USDT` | `15m` |

The model directory contains:

- `model.joblib`: LightGBM Booster serialized for local loading.
- `metadata.json`: feature names, symbol, timeframe, training period metadata, and holdout metrics.

Raw market data is intentionally not included. To generate predictions, provide your own OHLCV data and keep all experiments in simulation or offline research mode.

## Security Notes

`joblib` files use Python pickle-style serialization. Only load model files from repositories and commits you trust. Do not load arbitrary `joblib`, `pkl`, or pickle files from unknown sources.

Before upload, this public model pack was checked for:

- private-key, mnemonic, password, token, and wallet keyword patterns
- absolute local paths such as `/Users/mac`
- PEM/private key headers
- large-file surprises
- metadata fields that could expose live runtime state

No raw ledgers, wallet files, live runtime state, private keys, or raw training data are included.

## Verify The Pack

From the repository root:

```bash
python scripts/verify_public_model_pack.py
```

Expected output ends with something like:

```text
Verified 1 public model directory.
```

## Load One Model Manually

```bash
python3 - <<'PY'
from pathlib import Path
from src.python.predictor import load_predictor

model_dir = Path("data/models/lightgbm_ETH_USDT_15m_20260121_014635")
model, features, metadata = load_predictor(model_dir)
print(type(model).__name__)
print(metadata["symbol"], metadata["timeframe"], len(features))
PY
```

The model can be loaded without retraining. Prediction still requires OHLCV features in the expected format.

## What Is Not Included

- private live-trading models
- raw `.parquet` market data
- order ledgers or official trade records
- wallet files, private keys, `.env`, or credentials
- runtime state, websocket state, claim state, PID files, or logs
