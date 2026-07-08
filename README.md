# Polyfun Research Sandbox

> **Public safety notice**: This repository is a sanitized, lightweight research export. It was created with extensive AI assistance and is shared for learning, code review, and experimentation only. It is **not** a production trading system and must **not** be used to place automated Polymarket orders.

## What This Project Is

Polyfun is an experimental research codebase for studying short-horizon crypto prediction, feature engineering, model evaluation, and prediction-market execution ideas. The original private project included live-trading experiments, local market data, model artifacts, and wallet-specific runtime state. This public version intentionally keeps source code, example configuration, documentation, tests, and a single audited public ETH 15m learning model.

## What This Project Is Not

- It is not financial advice.
- It is not a reliable trading bot.
- It is not ready for live Polymarket trading.
- It does not include private keys, wallet files, real credentials, order ledgers, trained production/live models, raw data, or live runtime state.
- The included model is a small public LightGBM example for learning and offline experimentation only.
- It should not be connected to real funds without a complete independent security, legal, and trading-risk review.

## How This Repository Was Created

This GitHub-ready version was generated from a private local research workspace by copying only a safe allowlist of files into a fresh repository. Large raw data, logs, private/live model artifacts, runtime state, and sensitive configuration were excluded. A compact public ETH 15m LightGBM example model was added after separate metadata, binary-string, and secret-pattern checks. The export was also checked for large files and common secret patterns before upload.

## Intended Use

Use this repository to learn from the project structure, inspect model/research scripts, run local simulations, or adapt parts of the code in a private sandbox. Keep all experiments in simulation mode unless you fully understand and independently rebuild the trading, wallet, risk, compliance, and monitoring layers.

## Included Public Model Pack

This repository includes one small LightGBM example model under `data/models`:

- Symbol: `ETH/USDT`
- Timeframe: `15m`
- Files: `model.joblib` and `metadata.json`

This model lets readers inspect and load the predictor without retraining first. It is **not** the private live-trading model and is **not** sufficient to run a production trading system. Raw market data is intentionally not included; use your own data source for offline experiments.

See [docs/PUBLIC_MODEL_PACK.md](docs/PUBLIC_MODEL_PACK.md) for verification and loading examples. See [docs/README.md](docs/README.md) for how to read legacy research notes that mention assets or model families not included in this public export.

## Quick Start (Simulation / Research Only)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
npm install
cp .env.example .env
python scripts/verify_public_model_pack.py
```

Keep `TRADING_MODE=simulation` in `.env`. Do not add real wallet keys to this public repository.

---

## Original Project Notes

The notes below are preserved from the private research project for context. Some paths or workflows may refer to local data/model artifacts that are intentionally not included in this public export. The current bundled model truth is only `ETH/USDT 15m`; references to other assets, timeframes, GRU models, local parquet data, or private report paths are legacy research context unless explicitly documented in [docs/PUBLIC_MODEL_PACK.md](docs/PUBLIC_MODEL_PACK.md).

# LightGBM-FreqAI 加密货币预测与 Polymarket 自动交易系统

基于 LightGBM 二分类的加密货币涨跌预测，结合 FreqAI 式训练流程，通过 TypeScript 在 Polymarket 进行预测市场交易。

## 功能概览

- **代码历史支持的交易对**: BTC/USDT, ETH/USDT, SOL/USDT, XRP/USDT  
- **代码历史支持的周期**: 15m, 1h, 4h  
- **本公开仓库内置模型**: 仅 `ETH/USDT 15m` 学习模型  
- **预测**: 当前 K 线预测下一根 K 线 **UP (1)** / **DOWN (0)**  
- **特征**: KDJ、Range、MACD、OBV、RSI、布林带、成交量等（Kdj.txt、Rang.txt 可调）  
- **模型**: LightGBM 二分类，超参见 `config/lightgbm_params.json`  
- **月度改进**: 每月用近 1 个月数据重训，验证集更优则替换模型  

## 项目结构

```
project_root/
├── README.md
├── requirements.txt
├── package.json
├── .env.example          # 复制为 .env 并填写
├── config/
│   ├── Kdj.txt           # KDJ 超参
│   ├── Rang.txt          # Range 超参
│   ├── lightgbm_params.json
│   └── freqai_config.json
├── data/
│   ├── raw/              # 原始 K 线 (parquet)
│   ├── processed/
│   └── models/           # 训练好的模型
├── src/
│   ├── python/
│   │   ├── data_fetcher.py
│   │   ├── feature_engineering.py
│   │   ├── indicators/   # kdj_range, macd, obv, traditional
│   │   ├── model_trainer.py
│   │   ├── predictor.py
│   │   ├── model_improvement.py
│   │   └── api_server.py
│   └── typescript/
│       ├── main.ts
│       ├── polymarket_client.ts
│       ├── prediction_listener.ts
│       ├── trade_executor.ts
│       ├── database.ts
│       └── utils.ts
├── scripts/
│   ├── setup.sh
│   ├── start_training.sh
│   ├── start_prediction.sh
│   └── start_trading.sh
└── tests/
```

## 运行流程

### 方式一：独立 LightGBM（默认，可不装 freqtrade）

1. 安装：`pip install -r requirements.txt`（若只走此路径，可从 requirements 删掉 `freqtrade`）
2. 下载：`python -m src.python.data_fetcher --download-historical`
3. 训练：`python -m src.python.model_trainer --initial-training`，月度：`python -m src.python.model_improvement --monthly-update`
4. 预测 API：`python -m src.python.api_server`

### 方式二：FreqAI（需 freqtrade）

用 freqtrade 拉数据并回测，由 FreqAI 训练 LightGBMClassifier，模型在 `user_data/models/`。详见 **`FreqAI与独立训练说明.md`**。

---

### 1. 安装

```bash
pip install -r requirements.txt
npm install
cp .env.example .env   # 编辑 .env
```

### 2. 下载历史数据（独立路径）

```bash
python -m src.python.data_fetcher --download-historical
```

### 3. 训练（独立路径）

```bash
# 首次训练
python -m src.python.model_trainer --initial-training

# 月度改进
python -m src.python.model_improvement --monthly-update
```

### 4. 启动预测 API

```bash
python -m src.python.api_server
# 默认 http://0.0.0.0:8080，MODEL_PREDICTION_PORT 可改
```

### 5. 启动模拟/研究脚本（禁止实盘）

```bash
# Simulation/research only. Do not connect real wallets.
npm run build
```

- 轮询 `GET /predict`，用于本地研究和模拟。  
- Public export should remain in `TRADING_MODE=simulation`; live execution code is historical/research context only.  

## 配置说明

- **Kdj.txt**: `rsv_length`, `k_smooth`, `d_smooth`  
- **Rang.txt**: `range_period`, `range_multiplier`  
- **lightgbm_params.json**: LightGBM 超参，支持热更新  
- **.env**: 本地模拟配置、RPC、MongoDB、`TRADING_MODE`、`MODEL_PREDICTION_PORT` 等；不要提交真实钱包或密钥。  

## API

- `GET /health`  
- `GET /predict/{symbol}?timeframe=15m|1h|4h`  
- `GET /predict?timeframes=15m,1h,4h`  

## Docker（Win10）

- 卷挂载：`./config`、`./data` 已配好，在项目根执行 `docker-compose up -d api` 即可。
- Win10 上已去掉 `cpuset`、`gpus` 等易报错项；详细见 **`Docker与本地运行说明.md`**。

## 风险与扩展

- 市场、模型、技术、资金风险需自控。  
- 可选：情绪/链上特征、强化学习、告警与监控。  

## 依赖版本

- Python: freqtrade, lightgbm>=4, pandas>=2, ccxt>=4, fastapi, scikit-learn  
- Node: typescript ^5, ethers ^6, mongodb ^6, axios, dotenv
