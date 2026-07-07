"""
GRU Regime 预测写入器：使用 experiments/gru_regime_v1/outputs/models_best 下的
GRU encoder + LightGBM 模型进行预测，定时运行并将预测结果写入本地 JSON 文件。
TypeScript 交易脚本直接读取这个文件。

与 prediction_writer.py 的区别：
- 使用 GRU encoder 提取 embedding
- 使用带 embedding 的 LightGBM 模型
- 模型路径在 experiments/gru_regime_v1/outputs/models_best/{asset}/
"""

import gc
import os
import sys
import json
import time
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import joblib
import torch

# Windows 终端 UTF-8 支持
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "gru_regime_v1"


def _get_models_best() -> Path:
    """模型目录：GRU_MODELS_BEST 指定时用无1h4h 等，否则用默认 models_best"""
    env = os.environ.get("GRU_MODELS_BEST")
    if env:
        return Path(env).resolve()
    return EXPERIMENT_ROOT / "outputs" / "models_best"

# 添加项目路径
sys.path.insert(0, str(PROJECT_ROOT))

from src.python.data_fetcher import load_ohlcv, update_latest, validate_ohlcv_df, run_data_health_check_and_repair, fetch_simulated_15m_candle, update_and_verify_ohlcv_t0, SYMBOLS
from src.python.feature_engineering import build_features, add_multi_timeframe_features

# gru_regime 模块
from experiments.gru_regime_v1.src.data_loader import compute_features, create_sliding_windows
from experiments.gru_regime_v1.src.features import ZScoreNormalizer
from experiments.gru_regime_v1.src.gru_encoder import GRUVolatilityPredictor, create_model
from experiments.gru_regime_v1.src.train_lightgbm import merge_embeddings, get_feature_columns
from experiments.gru_regime_v1.src.utils import load_config, get_device

# 预测结果文件路径
PREDICTION_FILE = PROJECT_ROOT / "polymarket" / "predictions_gru.json"

# 日志目录
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# 预测热路径只使用最近 N 行做特征与 embedding，避免 24 万行导致单次跑 30+ 分钟、阻塞下一轮周期
PREDICTION_TAIL_ROWS = 8000

# GRU 输出文件名 → 模型/组合展示名
GRU_OUTPUT_TO_COMBO = {
    "predictions_gru_eth": "GRU_ETH_55",
    "predictions_gru_btc": "GRU_BTC_55",
    "predictions_gru_xrp": "GRU_XRP_55",
    "predictions_gru_sol": "GRU_SOL_52",
    "predictions_gru_eth_no1h4h": "GRU_ETH_55_无1h4h",
    "predictions_gru_btc_no1h4h": "GRU_BTC_57_无1h4h",
    "predictions_gru_xrp_no1h4h": "GRU_XRP_53_无1h4h",
}

# 支持的交易对（GRU 模型：BTC/ETH/XRP；若存在 SOL_USDT 则含 SOL）
def _gru_symbols():
    models_best = _get_models_best()
    syms = ["BTC/USDT", "ETH/USDT", "XRP/USDT"]
    if (models_best / "SOL_USDT").exists():
        syms = syms + ["SOL/USDT"]
    return syms

def _symbol_to_asset():
    models_best = _get_models_best()
    m = {"BTC/USDT": "BTC_USDT", "ETH/USDT": "ETH_USDT", "XRP/USDT": "XRP_USDT"}
    if (models_best / "SOL_USDT").exists():
        m["SOL/USDT"] = "SOL_USDT"
    return m

GRU_SYMBOLS = ["BTC/USDT", "ETH/USDT", "XRP/USDT"]  # 默认；含 SOL 时由下面动态扩展
SYMBOL_TO_ASSET = {"BTC/USDT": "BTC_USDT", "ETH/USDT": "ETH_USDT", "XRP/USDT": "XRP_USDT"}
TIMEFRAME = "15m"

# T+0 完整 K 线模式（与 V5 一致：收盘后触发，验证刚收盘 bar 再预测，不用模拟 K 线）
T0_TRIGGER_AFTER_CLOSE = 1  # K 线收盘后 1 秒触发

# 全局缓存：避免每次预测都重新加载模型
_model_cache: Dict[str, Dict[str, Any]] = {}
_model_cache_lgb_mtime: Dict[str, float] = {}  # 每资产上次加载时 lightgbm_with_embedding.joblib 的 mtime，用于热重载
_device = None


def setup_logging(log_file: Optional[str] = None, verbose: bool = True):
    """设置日志输出"""
    global _log_file_handler
    
    if log_file is None:
        log_file = LOGS_DIR / "prediction_writer_gru.log"
    else:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
    
    log_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger = logging.getLogger('prediction_writer_gru')
    logger.setLevel(logging.WARNING)
    logger.handlers.clear()
    
    file_handler = RotatingFileHandler(
        log_file,
        encoding='utf-8',
        maxBytes=50 * 1024 * 1024,
        backupCount=5,
        mode='a'
    )
    file_handler.setLevel(logging.WARNING)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    
    if verbose:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    
    return logger


def get_gru_device(use_mps: bool = True) -> torch.device:
    """获取设备（支持 MPS/CUDA/CPU）"""
    global _device
    if _device is not None:
        return _device
    _device = get_device(use_mps=use_mps)
    return _device


def load_gru_model(asset: str, device: torch.device) -> Dict[str, Any]:
    """加载 GRU encoder + LightGBM 模型，带缓存；按 lightgbm_with_embedding.joblib 的 mtime 热重载。"""
    lgb_path = _get_models_best() / asset / "lightgbm_with_embedding.joblib"
    if asset in _model_cache and lgb_path.exists():
        try:
            current_mtime = lgb_path.stat().st_mtime
            cached_mtime = _model_cache_lgb_mtime.get(asset, 0.0)
            if current_mtime > cached_mtime:
                del _model_cache[asset]
                _model_cache_lgb_mtime.pop(asset, None)
                logging.getLogger("prediction_writer_gru").info(
                    f"GRU 热重载: {asset} (lightgbm_with_embedding.joblib mtime 已更新)"
                )
        except OSError:
            pass
    if asset in _model_cache:
        return _model_cache[asset]

    asset_dir = _get_models_best() / asset
    if not asset_dir.exists():
        raise FileNotFoundError(f"models_best 目录不存在: {asset_dir}")
    
    pt_path = asset_dir / "encoder.pt"
    config_path = asset_dir / "encoder_config.json"
    # 优先从模型目录加载 normalizer，否则回退到实验 data 目录
    normalizer_path = asset_dir / f"{asset}_normalizer.json"
    if not normalizer_path.exists():
        normalizer_path = EXPERIMENT_ROOT / "data" / f"{asset}_normalizer.json"
    lgb_path = asset_dir / "lightgbm_with_embedding.joblib"
    
    if not pt_path.exists() or not config_path.exists():
        raise FileNotFoundError(f"缺少 encoder 或 config: {asset_dir}")
    if not normalizer_path.exists():
        raise FileNotFoundError(f"缺少 normalizer: {normalizer_path}")
    if not lgb_path.exists():
        raise FileNotFoundError(f"缺少 LightGBM: {lgb_path}")
    
    # 加载 GRU encoder
    train_config = load_config(str(config_path))
    hyperparams = train_config["hyperparams"]
    input_dim = train_config["input_dim"]
    
    encoder = create_model(
        input_dim=input_dim,
        hidden_size=hyperparams["hidden_size"],
        embedding_dim=hyperparams["embedding_dim"],
        num_layers=hyperparams.get("num_layers", 1),
        dropout=0.0,
        device=device,
    )
    ckpt = torch.load(pt_path, map_location=device)
    encoder.load_state_dict(ckpt["model_state_dict"])
    encoder.eval()
    
    # 加载 normalizer
    normalizer = ZScoreNormalizer.load(str(normalizer_path))
    
    # 加载 LightGBM
    lgb_model = joblib.load(lgb_path)
    
    # 获取特征列
    if hasattr(lgb_model, "feature_names_in_") and lgb_model.feature_names_in_ is not None:
        feature_cols = list(lgb_model.feature_names_in_)
    else:
        feature_cols = list(lgb_model.feature_name_()) if hasattr(lgb_model, "feature_name_") else None
    
    # 加载 calibrator（若存在）—— 步骤 2 增强：GRU 预测集成校准
    calibrator = None
    calibration_method = None
    cal_path = asset_dir / "calibrator.joblib"
    meta_path = asset_dir / "metadata.json"
    if cal_path.exists():
        calibrator = joblib.load(cal_path)
        # 从 metadata 读取校准方法
        if meta_path.exists():
            import json
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                calibration_method = meta.get("calibration")
            except Exception:
                pass
        if calibration_method is None:
            calibration_method = "isotonic"  # 默认回退
        logging.getLogger("prediction_writer_gru").info(
            f"[{asset}] 已加载 calibrator ({calibration_method})"
        )
    
    _model_cache[asset] = {
        "encoder": encoder,
        "normalizer": normalizer,
        "train_config": train_config,
        "lgb_model": lgb_model,
        "feature_cols": feature_cols,
        "calibrator": calibrator,
        "calibration_method": calibration_method,
    }
    try:
        _model_cache_lgb_mtime[asset] = lgb_path.stat().st_mtime
    except OSError:
        pass
    return _model_cache[asset]


def build_embeddings(
    df_raw: pd.DataFrame,
    encoder: GRUVolatilityPredictor,
    normalizer: ZScoreNormalizer,
    train_config: Dict,
    device: torch.device,
) -> pd.DataFrame:
    """构建 GRU embedding"""
    feature_cols = train_config["feature_cols"]
    lookback = train_config["hyperparams"]["lookback"]
    
    df = df_raw.copy()
    df = compute_features(df)
    df["volatility_label"] = 0.0
    X, _, timestamps = create_sliding_windows(df, lookback, feature_cols)
    X = normalizer.transform_array(X)
    
    num_samples = len(X)
    if num_samples == 0:
        return pd.DataFrame()
    
    embedding_dim = encoder.get_embedding_dim()
    embeddings = np.zeros((num_samples, embedding_dim), dtype=np.float32)
    
    with torch.no_grad():
        batch_x = torch.FloatTensor(X).to(device)
        embeddings = encoder.get_embedding(batch_x).cpu().numpy()
    
    ts = timestamps if hasattr(timestamps, "__len__") else np.arange(num_samples)
    if hasattr(ts, "dtype") and np.issubdtype(ts.dtype, np.datetime64):
        ts_ms = pd.to_datetime(ts).astype("int64") // 10**6
    else:
        ts_ms = np.asarray(ts, dtype=np.int64)
    
    out = {"timestamp": ts_ms}
    for j in range(embedding_dim):
        out[f"emb_{j}"] = embeddings[:, j]
    return pd.DataFrame(out)


def _parse_env_allowed_markets(content: str) -> Optional[str]:
    """从 .env 文件内容解析 ALLOWED_MARKETS"""
    found = None
    for line in content.replace("\ufeff", "").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if "=" not in s:
            continue
        key = s.split("=", 1)[0].strip()
        if key.upper().startswith("EXPORT "):
            key = key[7:].strip()
        if key.upper() != "ALLOWED_MARKETS":
            continue
        val = s.split("=", 1)[1]
        if "#" in val:
            val = val.split("#", 1)[0]
        val = val.strip().strip("'\"")
        if val:
            found = val
    return found


def _get_prediction_symbols() -> List[str]:
    """获取允许的交易对（受 ALLOWED_MARKETS 限制，仅限 GRU 支持的币种）"""
    raw = os.environ.get("ALLOWED_MARKETS") or ""
    raw = str(raw).strip() if raw else ""
    
    for env_path in [PROJECT_ROOT / "polymarket" / ".env", PROJECT_ROOT / ".env"]:
        if not raw and env_path.exists():
            try:
                parsed = _parse_env_allowed_markets(env_path.read_text(encoding="utf-8", errors="ignore"))
                if parsed:
                    raw = parsed
                    break
            except Exception:
                pass
    
    allowed = _gru_symbols()
    if not raw:
        return allowed

    tokens = [t.strip().strip("'\"") for t in raw.split(",") if t.strip()]
    out = []
    for t in tokens:
        if "/" in t:
            sym = t
        else:
            sym = f"{t.upper()}/USDT"
        if sym in allowed:
            out.append(sym)
    return out if out else allowed


def predict_gru(symbol: str, device: torch.device, skip_sim_candle: bool = False) -> Tuple[str, float, Dict[str, Any]]:
    """使用 GRU regime 模型进行预测。
    skip_sim_candle=True 时（T+0 模式）：不追加模拟 K 线，只用本地已收盘的完整 15m。"""
    asset = _symbol_to_asset().get(symbol) or SYMBOL_TO_ASSET.get(symbol)
    if not asset:
        raise ValueError(f"不支持的交易对: {symbol}（GRU 支持 BTC/ETH/XRP，若存在 SOL_USDT 则含 SOL）")
    
    details = {
        "model_path": f"gru_regime_best/{asset}",
        "data_source": None,
        "data_rows": 0,
        "last_close": 0,
        "raw_prob": 0,
    }
    
    # 加载模型
    model_data = load_gru_model(asset, device)
    encoder = model_data["encoder"]
    normalizer = model_data["normalizer"]
    train_config = model_data["train_config"]
    lgb_model = model_data["lgb_model"]
    feature_cols = model_data["feature_cols"]
    
    # 加载数据：T+0 模式用调度器已验证并 strip 过的本地 15m，不再拉取（避免把未收盘 bar 拉进来）
    _log = logging.getLogger("prediction_writer_gru")
    if skip_sim_candle:
        df = load_ohlcv(symbol, TIMEFRAME)
        details["data_source"] = "本地(T+0已验证)"
    else:
        _max_retries, _delay = 10, 10
        df = None
        for attempt in range(1, _max_retries + 1):
            try:
                df = update_latest(symbol, TIMEFRAME)
                details["data_source"] = "本地+实时更新"
                break
            except Exception as e:
                _log.warning(f"[{symbol}] 实时更新失败 (尝试 {attempt}/{_max_retries}): {type(e).__name__}: {e}")
                if attempt < _max_retries:
                    time.sleep(_delay)
                else:
                    _log.warning(f"[{symbol}] 重试用尽，改用仅本地")
                    df = load_ohlcv(symbol, TIMEFRAME)
                    details["data_source"] = "仅本地(网络失败)"
    # 热路径只做最小检查，不跑完整校验（完整校验在错峰时单独跑，见 run_scheduler）
    if df.empty or len(df) < 100:
        raise ValueError(f"数据不足: {symbol}, 仅有 {len(df)} 行")
    if len(df) > PREDICTION_TAIL_ROWS:
        df = df.tail(PREDICTION_TAIL_ROWS).reset_index(drop=True)

    # ─── 追加模拟 15m K 线（T+0 模式跳过：已用完整已收盘 bar）───
    if not skip_sim_candle:
        _log = logging.getLogger("prediction_writer_gru")
        try:
            sim_candle = fetch_simulated_15m_candle(symbol)
            if sim_candle is not None and not sim_candle.empty:
                sim_ts = sim_candle["timestamp"].iloc[0]
                last_ts = df["timestamp"].iloc[-1]
                if abs(sim_ts - last_ts) < 1_000:  # 时间戳差 <1 秒，视为同一根
                    for col in ["open", "high", "low", "close", "volume"]:
                        df.iloc[-1, df.columns.get_loc(col)] = sim_candle[col].iloc[0]
                else:
                    new_row = {}
                    for col in df.columns:
                        if col in sim_candle.columns:
                            new_row[col] = sim_candle[col].iloc[0]
                        else:
                            new_row[col] = df[col].iloc[-1]
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                _log.info(f"[{symbol}] 追加模拟 15m K 线 (close={sim_candle['close'].iloc[0]:.2f})")
                details["data_source"] = (details.get("data_source") or "") + "+模拟K线"
        except Exception as e:
            _log.warning(f"[{symbol}] 模拟 K 线获取失败（继续使用历史数据）: {e}")

    _log = logging.getLogger("prediction_writer_gru")

    details["data_rows"] = len(df)
    details["last_close"] = float(df["close"].iloc[-1])
    
    # 构建原始特征；无1h4h 模型时跳过 1h/4h 特征（GRU_SKIP_MTF=1）
    df_orig = build_features(df.copy(), symbol)
    if os.environ.get("GRU_SKIP_MTF") != "1":
        try:
            if skip_sim_candle:
                # T+0：调度器已更新并验证 1h/4h，直接用本地，不再拉取
                pass
            else:
                update_latest(symbol, "1h")
                update_latest(symbol, "4h")
        except Exception as e:
            _log.warning(f"[{symbol}] 1h/4h 实时更新失败（继续用本地）: {e}")
        try:
            df_orig = add_multi_timeframe_features(df_orig, symbol)
        except Exception:
            pass
    
    if "timestamp" not in df_orig.columns and "date" in df_orig.columns:
        df_orig["timestamp"] = pd.to_datetime(df_orig["date"]).astype("int64") // 10**6
    
    # 构建 embedding（需要用原始 OHLCV 数据）
    df_raw = df.copy()
    if "timestamp" not in df_raw.columns:
        if "date" in df_raw.columns:
            df_raw["timestamp"] = pd.to_datetime(df_raw["date"]).astype("int64") // 10**6
    
    emb_df = build_embeddings(df_raw, encoder, normalizer, train_config, device)
    if emb_df.empty:
        raise ValueError(f"无法生成 embedding: {symbol}")
    
    # 合并特征 + embedding
    merged, _ = merge_embeddings(df_orig, emb_df, fill_strategy="zero")
    if len(merged) < 1:
        raise ValueError(f"合并后数据为空: {symbol}")
    
    # 取最后一行进行预测
    last_row = merged.iloc[[-1]]
    
    # 获取特征
    if feature_cols is None:
        feature_cols = get_feature_columns(merged, include_embeddings=True)
    
    missing = [c for c in feature_cols if c not in last_row.columns]
    if missing:
        raise ValueError(f"特征缺失: {missing[:5]}")
    
    X = last_row[feature_cols].fillna(0)
    prob = lgb_model.predict_proba(X)[0, 1]
    
    details["raw_prob"] = round(prob, 4)

    # 应用 calibrator（若已加载）—— 步骤 2 增强：GRU 校准集成
    calibrator = model_data.get("calibrator")
    calibration_method = model_data.get("calibration_method")
    if calibrator is not None and calibration_method:
        if os.environ.get("SKIP_CALIBRATION") != "1":
            from .predictor import apply_calibration
            prob = apply_calibration(calibrator, calibration_method, prob)
            details["calibrated_prob"] = round(float(prob), 4)

    pred = 1 if prob >= 0.5 else 0
    label = "UP" if pred == 1 else "DOWN"
    # confidence = 预测方向的概率（见 docs/概率与阈值逻辑.md）
    conf = prob if pred == 1 else (1 - prob)
    return label, round(conf, 4), details


def write_predictions(
    verbose: bool = True,
    output_file: Optional[str] = None,
    log_file: Optional[str] = None,
    use_mps: bool = True,
    early_seconds: int = 40,
    allowed_symbols: Optional[List[str]] = None,
    use_t0: bool = False,
) -> Dict[str, Any]:
    """获取所有预测并写入 JSON 文件。early_seconds 与调度器一致，用于判断目标周期（下一根 K 线开盘前 N 秒视为「收盘中」）。
    若传入 allowed_symbols，仅对其中且属于 _get_prediction_symbols() 的交易对预测；未在 allowed_symbols 中的将跳过并打 WARNING。
    use_t0=True：T+0 模式，不追加模拟 K 线，用刚收盘的完整 15m（由调用方在调用前完成验证）。"""
    logger = setup_logging(log_file=log_file, verbose=verbose)
    device = get_gru_device(use_mps=use_mps)

    all_symbols = _get_prediction_symbols()
    if allowed_symbols is not None:
        allowed_set = set(allowed_symbols)
        symbols = [s for s in all_symbols if s in allowed_set]
        skipped = [s for s in all_symbols if s not in allowed_set]
        for s in skipped:
            logger.warning("[跳过] %s：15m 数据不可用（健康检查修复失败），本轮回过预测", s)
            if verbose:
                print(f"  [跳过] {s}：15m 数据不可用（健康检查修复失败），本轮回过预测")
    else:
        symbols = all_symbols
    out_path = Path(output_file).resolve() if output_file else PREDICTION_FILE
    
    # 计算目标周期（Unix 秒，对齐 15 分钟边界，与 Polymarket slug 一致；与调度触发点一致：提前 early_seconds 秒视为收盘中）
    now_ts = int(time.time())
    in_closing = (now_ts % 900) >= (900 - early_seconds)
    target_period_end_ts = ((now_ts // 900) + (1 if in_closing else 0)) * 900
    
    write_start = datetime.now()
    result = {
        "timestamp": write_start.isoformat(),
        "target_period_end_ts": target_period_end_ts,
        "predictions": {}
    }
    
    # 显示用本地时间（与机器时区一致）
    period_start = datetime.fromtimestamp(target_period_end_ts)
    period_end = datetime.fromtimestamp(target_period_end_ts + 900)
    period_str = f"{period_start.strftime('%H:%M')}–{period_end.strftime('%H:%M')}"
    
    combo_name = GRU_OUTPUT_TO_COMBO.get(out_path.stem, out_path.stem.upper().replace("_", "_"))
    if verbose:
        print(f"\n{'='*70}")
        print(f"  GRU Regime 预测执行 - {write_start.strftime('%Y-%m-%d %H:%M:%S')} 本地")
        print(f"{'='*70}")
        print(f"  模型/组合: {combo_name}")
        print(f"  交易对: {', '.join(symbols)}")
        print(f"  时间周期: {TIMEFRAME}")
        print(f"  目标周期(slug ts=周期开始): {target_period_end_ts} → {period_str} 本地")
        print(f"  输出文件: {out_path}")
        print(f"  模型: GRU Regime Best ({_get_models_best()})")
        print(f"{'='*70}\n")
    
    success_count = 0
    fail_count = 0
    
    # 预测失败时重试次数与间隔（秒）
    max_predict_retries = 10
    retry_delay_seconds = 1.5

    for symbol in symbols:
        symbol_short = symbol.replace('/USDT', '')
        if verbose:
            print(f"  [{symbol_short}-{TIMEFRAME}] 开始预测...")
        
        last_error = None
        for attempt in range(max_predict_retries + 1):
            try:
                direction, confidence, details = predict_gru(symbol, device, skip_sim_candle=use_t0)
                break
            except Exception as e:
                last_error = e
                if attempt < max_predict_retries:
                    if verbose:
                        print(f"      重试 {attempt + 1}/{max_predict_retries} ({retry_delay_seconds}s 后)...")
                    time.sleep(retry_delay_seconds)
                    continue
                raise

        try:
            key = f"{symbol.replace('/', '_')}_{TIMEFRAME}"
            
            result["predictions"][key] = {
                "symbol": symbol,
                "timeframe": TIMEFRAME,
                "direction": direction,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "details": details
            }
            
            dir_text = "上涨" if direction == "UP" else "下跌"
            conf_pct = f"{confidence * 100:.1f}%"
            raw_prob_pct = f"{details['raw_prob'] * 100:.1f}%"
            
            if verbose:
                print(f"      模型: {details['model_path']}")
                print(f"      数据: {details['data_source']} ({details['data_rows']} 行)")
                print(f"      最新收盘价: ${details['last_close']:.2f}")
                print(f"      原始概率(UP): {raw_prob_pct}")
                print(f"      预测结果: [{direction}] {dir_text}")
                print(f"      置信度: {conf_pct}")
                print(f"      ---> 完成\n")
            
            success_count += 1
            
        except Exception as e:
            if verbose:
                print(f"      错误: {e}")
                print(f"      ---> 失败\n")
            
            result["predictions"][f"{symbol.replace('/', '_')}_{TIMEFRAME}"] = {
                "symbol": symbol,
                "timeframe": TIMEFRAME,
                "direction": None,
                "confidence": 0,
                "error": str(e)
            }
            fail_count += 1
    
    # 写入文件
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_end = datetime.now()
    result["timestamp"] = write_end.isoformat()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    if verbose:
        elapsed = (write_end - write_start).total_seconds()
        elapsed_str = f"{int(elapsed // 60)}分{int(elapsed % 60)}秒" if elapsed >= 60 else f"{int(elapsed)}秒"
        print(f"{'='*70}")
        print(f"  预测完成")
        print(f"{'='*70}")
        print(f"  模型/组合: {combo_name}")
        print(f"  成功: {success_count} 个")
        print(f"  失败: {fail_count} 个")
        print(f"  写入: {out_path}")
        print(f"  完成时间: {write_end.strftime('%Y-%m-%d %H:%M:%S')} 本地")
        print(f"  耗时: {elapsed_str}")
        print(f"{'='*70}\n")
        
        print("  预测汇总:")
        print("  " + "-"*50)
        for key, pred in result["predictions"].items():
            if pred.get("direction"):
                sym = pred["symbol"].replace('/USDT', '')
                conf = f"{pred['confidence'] * 100:.1f}%"
                print(f"    {sym}-{pred['timeframe']}: [{pred['direction']}] {conf}")
            else:
                sym = pred["symbol"].replace('/USDT', '')
                print(f"    {sym}-{pred['timeframe']}: [ERROR] {pred.get('error', '未知错误')}")
        print("  " + "-"*50 + "\n")
    
    return result


def run_scheduler_t0(
    output_file: Optional[str] = None,
    log_file: Optional[str] = None,
    verbose: bool = True,
    use_mps: bool = True,
):
    """T+0 完整 K 线调度器：K 线收盘后触发，先验证本地已有刚收盘的 15m bar，再用真实已收盘 15m 预测，不用模拟 K 线。
    与 V5 T+0 行为一致。"""
    logger = setup_logging(log_file=log_file, verbose=verbose)
    symbols = _get_prediction_symbols()
    out_path = Path(output_file).resolve() if output_file else PREDICTION_FILE
    last_predicted_bar_ts: int = 0

    if verbose:
        print("\n" + "="*70)
        print("  GRU Regime 预测写入器 · T+0 完整 K 线模式")
        print("="*70)
        print(f"  交易对: {', '.join(symbols)}")
        print(f"  输出文件: {out_path}")
        print(f"  触发: K 线收盘后 +{T0_TRIGGER_AFTER_CLOSE}s，验证完整 bar 后再预测，不用模拟 K 线")
        print("  按 Ctrl+C 停止")
        print("="*70 + "\n")

    try:
        while True:
            now = int(time.time())
            bar_start = (now // 900) * 900
            next_trigger_ts = bar_start + T0_TRIGGER_AFTER_CLOSE
            if next_trigger_ts <= now:
                next_trigger_ts += 900
            target_bar_ts = next_trigger_ts - T0_TRIGGER_AFTER_CLOSE - 900
            if target_bar_ts == last_predicted_bar_ts:
                next_trigger_ts += 900
                target_bar_ts += 900
            wait_seconds = next_trigger_ts - now
            next_dt = datetime.fromtimestamp(next_trigger_ts)
            logger.info("  下次预测触发: %s (%ds 后)", next_dt.strftime("%H:%M:%S"), wait_seconds)
            if verbose:
                print(f"\n  ⏰ 下次预测触发: {next_dt.strftime('%H:%M:%S')} ({wait_seconds}s 后)")

            while True:
                remaining = next_trigger_ts - int(time.time())
                if remaining <= 0:
                    break
                if verbose:
                    to_trigger = next_trigger_ts - int(time.time())
                    print(f"\r  等待: {to_trigger // 60}分{to_trigger % 60}秒后执行 | 当前: {datetime.now().strftime('%H:%M:%S')}     ", end="", flush=True)
                time.sleep(min(5, remaining))
            if verbose:
                print()

            # 执行前健康检查与修复（15m+1h+4h）
            syms = _get_prediction_symbols()
            results = run_data_health_check_and_repair(syms, ["15m", "1h", "4h"], min_rows=50)
            failed_15m = {s for s, tf, _msg, repaired in results if tf == "15m" and not repaired}
            allowed_symbols = [s for s in syms if s not in failed_15m]
            for sym, tf, msg, repaired in results:
                if repaired:
                    logger.info("[数据健康检查] %s %s: 已修复", sym, tf)
                elif tf == "15m":
                    logger.warning("[数据健康检查] %s 15m 不通过，本轮回过该 symbol", sym)

            # 更新 15m/1h/4h（与 V5 一致，预测前拉取最新）
            if verbose:
                print("  T+0: 更新本地 K 线 (15m/1h/4h)...")
            for s in allowed_symbols:
                try:
                    update_latest(s, "15m")
                    update_latest(s, "1h")
                    update_latest(s, "4h")
                except Exception as e:
                    logger.warning("  %s: 更新 K 线失败: %s", s, e)

            expected_bar_start_ms = target_bar_ts * 1000  # 刚收盘的 bar 起始（毫秒）
            current_bar_start_ms = (target_bar_ts + 900) * 1000  # 当前正在形成的 bar 起始
            ok = update_and_verify_ohlcv_t0(
                allowed_symbols,
                expected_bar_start_ms=expected_bar_start_ms,
                current_bar_start_ms=current_bar_start_ms,
            )
            if not ok:
                logger.error("  完整 K 线验证失败，跳过本轮预测")
                if verbose:
                    print("  完整 K 线验证失败，跳过本轮预测")
                time.sleep(5)
                continue

            try:
                write_predictions(
                    verbose=verbose,
                    output_file=output_file,
                    log_file=log_file,
                    use_mps=use_mps,
                    early_seconds=40,
                    allowed_symbols=allowed_symbols,
                    use_t0=True,
                )
                last_predicted_bar_ts = target_bar_ts
            except Exception as e:
                logger.error("  T+0 预测失败: %s", e, exc_info=True)
                if verbose:
                    print(f"  T+0 预测失败: {e}")
            time.sleep(5)
    except KeyboardInterrupt:
        if verbose:
            print("\n\n  [停止] GRU T+0 预测写入器已停止\n")


def run_scheduler(
    interval_seconds: int = 3,
    early_seconds: int = 40,
    output_file: Optional[str] = None,
    log_file: Optional[str] = None,
    verbose: bool = True,
    use_mps: bool = True,
):
    """定时运行预测写入器"""
    logger = setup_logging(log_file=log_file, verbose=verbose)
    
    out_path = Path(output_file).resolve() if output_file else PREDICTION_FILE
    symbols = _get_prediction_symbols()
    
    if verbose:
        print("\n" + "="*70)
        print("  GRU Regime 预测写入器启动")
        print("="*70)
        print(f"  交易对: {', '.join(symbols)}")
        print(f"  输出文件: {out_path}")
        print(f"  检查间隔: {interval_seconds} 秒")
        print(f"  提前执行: 下一市场开盘前 {early_seconds} 秒")
        print(f"  执行时机: 每15分钟周期的第14分{60-early_seconds}秒（下一根K线开盘前）")
        print(f"  模型: GRU Regime Best")
        print("  按 Ctrl+C 停止")
        print("="*70 + "\n")
        print("  [等待中] 等待下一个执行时机...\n")
    
    last_execution_minute = -1
    execution_count = 0
    loop_count = 0
    
    try:
        while True:
            now = datetime.now()
            minute = now.minute
            second = now.second
            
            minutes_in_cycle = minute % 15
            target_minute = 14
            target_second = 60 - early_seconds
            
            should_execute = (
                minutes_in_cycle == target_minute and 
                second >= target_second and
                minute != last_execution_minute
            )
            
            if should_execute:
                execution_count += 1
                if verbose:
                    print(f"\n  [触发] 第 {execution_count} 次执行")
                    print(f"  [时间] {now.strftime('%H:%M:%S')}")
                    print(f"  [原因] 当前处于15分钟周期的第{minutes_in_cycle}分{second}秒")
                    print(f"         满足执行条件: 第{target_minute}分 >= {target_second}秒\n")
                
                # 执行前做一次数据健康检查与修复，避免 1h/4h parquet 损坏导致「特征与模型不一致」
                if verbose:
                    print("  [数据健康检查] 执行中...")
                logger.info("  [数据健康检查] 执行中...")
                syms = _get_prediction_symbols()
                results = run_data_health_check_and_repair(syms, ["15m", "1h", "4h"], min_rows=50)
                for sym, tf, msg, repaired in results:
                    if repaired:
                        logger.info(f"[数据健康检查] {sym} {tf}: 不通过({msg})，已拉取更新并修复")
                        if verbose:
                            print(f"  [数据健康检查] {sym} {tf}: 不通过({msg})，已拉取更新并修复")
                    else:
                        logger.warning(f"[数据健康检查] {sym} {tf}: 不通过({msg})，尝试修复后仍不通过")
                        if verbose:
                            print(f"  [数据健康检查] {sym} {tf}: 不通过({msg})，尝试修复后仍不通过")
                if not results and verbose:
                    print("  [数据健康检查] 通过")
                if not results:
                    logger.info("  [数据健康检查] 通过")

                # 仅对 15m 健康（或修复成功）的交易对做预测，避免重复重试
                failed_15m = {s for s, tf, _msg, repaired in results if tf == "15m" and not repaired}
                allowed_symbols = [s for s in syms if s not in failed_15m]

                write_predictions(
                    verbose=verbose,
                    output_file=output_file,
                    log_file=log_file,
                    use_mps=use_mps,
                    early_seconds=early_seconds,
                    allowed_symbols=allowed_symbols,
                )
                last_execution_minute = minute
                # 每轮执行后释放 PyTorch/NumPy 缓存，减缓长时间运行内存增长
                gc.collect()
                try:
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                except Exception:
                    pass
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                if verbose:
                    print(f"  [完成] 下一次执行将在约15分钟后\n")
            
            # 计算到下次执行的等待时间（仅用于显示）
            if minutes_in_cycle < target_minute:
                next_min = target_minute - minutes_in_cycle
                next_sec = target_second
            elif minutes_in_cycle == target_minute and second < target_second:
                next_min = 0
                next_sec = target_second - second
            else:
                # 已过本周期 14:20，下次为下一 15 分钟块的 14:20
                if minutes_in_cycle == target_minute:
                    remainder_sec = (60 - second) + target_second
                    next_min = 14 + remainder_sec // 60
                    next_sec = remainder_sec % 60
                else:
                    next_min = 14 - minutes_in_cycle
                    next_sec = target_second
            
            to_candle_close_min = 15 - minutes_in_cycle - 1
            to_candle_close_sec = 60 - second
            if to_candle_close_sec == 60:
                to_candle_close_sec = 0
                to_candle_close_min += 1
            
            if verbose:
                status = f"  等待: {next_min}分{next_sec}秒后执行 | K线收盘: {to_candle_close_min}分{to_candle_close_sec}秒 | 当前: {now.strftime('%H:%M:%S')}"
                print(f"\r{status}     ", end="", flush=True)
            
            # 错峰数据健康检查+修复：约 10 分钟一次，仅在非执行窗口；不通过则自动 update_latest 拉取并重验
            loop_count += 1
            if (loop_count % 20 == 0) and (minutes_in_cycle <= 12):
                syms = _get_prediction_symbols()
                results = run_data_health_check_and_repair(syms, ["15m", "1h", "4h"], min_rows=50)
                for sym, tf, msg, repaired in results:
                    if repaired:
                        logger.info(f"[数据健康检查] {sym} {tf}: 不通过({msg})，已拉取更新并修复")
                    else:
                        logger.warning(f"[数据健康检查] {sym} {tf}: 不通过({msg})，尝试修复后仍不通过")
            
            time.sleep(interval_seconds)
            
    except KeyboardInterrupt:
        if verbose:
            print(f"\n\n  [停止] GRU 预测写入器已停止")
            print(f"  [统计] 共执行 {execution_count} 次预测\n")


def main():
    """命令行入口。仅 T+0 模式：收盘后触发，验证完整 15m bar 再预测，不用模拟 K 线。"""
    import argparse

    parser = argparse.ArgumentParser(description="GRU Regime 预测写入器（仅 T+0 模式）")
    parser.add_argument("--once", action="store_true", help="只执行一次（仍用 T+0 数据：不追加模拟 K 线）")
    parser.add_argument("--output", type=str, default=None, help="输出 JSON 路径")
    parser.add_argument("--log-file", type=str, default=None, help="日志文件路径")
    parser.add_argument("--no-verbose", action="store_true", help="不输出到控制台")
    parser.add_argument("--no-mps", action="store_true", help="禁用 MPS（Mac GPU）")
    args = parser.parse_args()

    verbose = not args.no_verbose
    use_mps = not args.no_mps

    if args.once:
        write_predictions(
            verbose=verbose,
            output_file=args.output,
            log_file=args.log_file,
            use_mps=use_mps,
            early_seconds=40,
            use_t0=True,
        )
    else:
        run_scheduler_t0(
            output_file=args.output,
            log_file=args.log_file,
            verbose=verbose,
            use_mps=use_mps,
        )


if __name__ == "__main__":
    main()
