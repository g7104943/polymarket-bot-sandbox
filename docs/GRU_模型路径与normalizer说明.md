# GRU 模拟/真实交易：模型路径与 normalizer 说明

本文档确认主程序（`./启动GRU新模型.sh`、`prediction_writer_gru.py`）中各组合使用的**模型目录**与 **normalizer 文件路径**，确保均为训练导出的最佳模型且路径正确。

---

## 一、模型目录来源

- **有 1h/4h 组合（组合 1–5、4b）**  
  不设置环境变量 `GRU_MODELS_BEST` 时，`prediction_writer_gru.py` 使用：
  - **模型根目录**：`experiments/gru_regime_v1/outputs/models_best`
  - 即 `export_best_models.py` 从随机搜索中选出的最佳 run 导出后的目录。

- **无 1h/4h 组合（组合 6–8）**  
  启动脚本设置 `GRU_MODELS_BEST=<项目根>/experiments/gru_regime_v1/outputs/models_best_no1h4h` 与 `GRU_SKIP_MTF=1` 时使用：
  - **模型根目录**：`experiments/gru_regime_v1/outputs/models_best_no1h4h`
  - 即无 1h/4h 特征训练后导出的最佳模型目录。

---

## 二、各组合与模型路径对应

| 组合 | 名称 | 模型根目录 | 资产子目录 | 使用的 normalizer |
|------|------|------------|------------|-------------------|
| 1 | gru_eth_0.55_无动态 | models_best | ETH_USDT | ETH_USDT/ETH_USDT_normalizer.json |
| 2 | gru_xrp_0.55_无动态 | models_best | XRP_USDT | XRP_USDT/XRP_USDT_normalizer.json |
| 3 | gru_btc_0.55_无动态 | models_best | BTC_USDT | BTC_USDT/BTC_USDT_normalizer.json |
| 4 | gru_eth_0.54_有动态 | models_best | ETH_USDT | ETH_USDT/ETH_USDT_normalizer.json |
| 4b | gru_eth_0.55_有动态_有1h4h | models_best | ETH_USDT | ETH_USDT/ETH_USDT_normalizer.json |
| 5 | gru_sol_0.52_无动态 | models_best | SOL_USDT | SOL_USDT/SOL_USDT_normalizer.json |
| 6 | gru_eth_0.55_无动态_无1h4h | models_best_no1h4h | ETH_USDT | ETH_USDT/ETH_USDT_normalizer.json |
| 7 | gru_btc_0.57_无动态_无1h4h | models_best_no1h4h | BTC_USDT | BTC_USDT/BTC_USDT_normalizer.json |
| 8 | gru_xrp_0.53_有动态_无1h4h | models_best_no1h4h | XRP_USDT | XRP_USDT/XRP_USDT_normalizer.json |

组合 1、4、4b 共用同一套 ETH 预测（同一 Python 进程与 `predictions_gru_eth.json`），故模型与 normalizer 路径一致。

---

## 三、代码中的路径逻辑（prediction_writer_gru.py）

- **模型根目录**（`_get_models_best()`）  
  - 若环境变量 `GRU_MODELS_BEST` 已设置 → 使用其指向的目录（无 1h4h 时为 `models_best_no1h4h`）。  
  - 未设置 → 使用 `experiments/gru_regime_v1/outputs/models_best`。

- **资产目录**  
  - `asset_dir = _get_models_best() / asset`  
  - 例如 ETH：`models_best/ETH_USDT` 或 `models_best_no1h4h/ETH_USDT`。

- **normalizer 路径**（优先模型目录，再回退到实验 data）  
  - 首选：`asset_dir / f"{asset}_normalizer.json"`  
    - 例：`ETH_USDT/ETH_USDT_normalizer.json`、`BTC_USDT/BTC_USDT_normalizer.json`。  
  - 若不存在则：`experiments/gru_regime_v1/data/{asset}_normalizer.json`。

- **其他模型文件（同一 asset_dir 下）**  
  - `encoder.pt`、`encoder_config.json`、`lightgbm_with_embedding.joblib`  
  - 均来自对应「模型根目录」下的最佳导出，与上表一致。

---

## 四、目录与文件存在性确认

以下结构已在仓库中确认存在：

**models_best/**  
- ETH_USDT：encoder.pt, encoder_config.json, ETH_USDT_normalizer.json, lightgbm_with_embedding.joblib  
- BTC_USDT：同上（BTC_USDT_normalizer.json）  
- XRP_USDT：同上（XRP_USDT_normalizer.json）  
- SOL_USDT：同上（SOL_USDT_normalizer.json）  

**models_best_no1h4h/**  
- ETH_USDT：encoder.pt, encoder_config.json, ETH_USDT_normalizer.json, lightgbm_with_embedding.joblib  
- BTC_USDT：同上（BTC_USDT_normalizer.json）  
- XRP_USDT：同上（XRP_USDT_normalizer.json）  

结论：  
- 新加的 **gru_eth_0.55_有动态_有1h4h（4b）** 与组合 1、4 相同，使用 **models_best/ETH_USDT** 及 **ETH_USDT/ETH_USDT_normalizer.json**，为同一套训练最佳模型。  
- 新加的三个无 1h4h 组合（6、7、8）使用 **models_best_no1h4h** 下对应资产目录及 **{asset}_normalizer.json**，路径与代码逻辑一致，且均为无 1h/4h 训练导出的最佳模型。
