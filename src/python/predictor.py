"""
实时预测：加载 data/models 下最新 LightGBM 模型，用最新 K 线计算特征，预测下一根 15m/1h/4h K 线涨跌，输出 UP(1) 或 DOWN(0) 及置信度。
支持加载按 timeframe 分别训练的独立模型。
"""

import json
import os
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union

from .data_fetcher import fetch_latest, SYMBOLS, TIMEFRAMES, load_ohlcv, update_latest
from .feature_engineering import build_features, get_feature_columns, add_multi_timeframe_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "data" / "models"


def _latest_model_dir(timeframe: Optional[str] = None) -> Path:
    """
    获取最新的模型目录。
    如果指定 timeframe，优先查找对应的 timeframe 专用模型。
    """
    dirs = [d for d in MODELS_DIR.iterdir() if d.is_dir() and (d / "model.joblib").exists() and "_bak" not in d.name]
    if not dirs:
        raise FileNotFoundError(f"未找到模型，请先在 {MODELS_DIR} 下训练")
    
    if timeframe:
        # 优先查找 timeframe 专用模型（如 lightgbm_15m_*, lightgbm_1h_*）
        tf_dirs = [d for d in dirs if f"_{timeframe}_" in d.name or d.name.endswith(f"_{timeframe}")]
        if tf_dirs:
            return max(tf_dirs, key=lambda d: d.stat().st_mtime)
    
    # 返回最新的通用模型
    return max(dirs, key=lambda d: d.stat().st_mtime)


def _find_all_timeframe_models(models_root: Optional[Path] = None) -> Dict[str, Path]:
    """
    查找所有按 timeframe 分类的模型。
    models_root: 模型根目录，默认 data/models；用于多套模型对比时指定 data/models_setA 等。
    返回 { timeframe: model_dir_path }
    """
    root = Path(models_root) if models_root is not None else MODELS_DIR
    result = {}
    dirs = [d for d in root.iterdir() if d.is_dir() and (d / "model.joblib").exists() and "_bak" not in d.name]
    
    for tf in TIMEFRAMES:
        tf_dirs = [d for d in dirs if f"_{tf}_" in d.name]
        if tf_dirs:
            result[tf] = max(tf_dirs, key=lambda d: d.stat().st_mtime)
    
    return result


def _find_symbol_timeframe_model(symbol: str, timeframe: str, models_root: Optional[Path] = None) -> Optional[Path]:
    """
    查找特定 symbol+timeframe 的模型。
    models_root: 模型根目录，默认 data/models；用于多套模型对比时指定 data/models_setA 等。
    返回 model_dir_path 或 None
    """
    root = Path(models_root) if models_root is not None else MODELS_DIR
    dirs = [d for d in root.iterdir() if d.is_dir() and (d / "model.joblib").exists() and "_bak" not in d.name]
    
    # 将 symbol 转换为文件名格式（如 BTC/USDT -> BTC_USDT）
    symbol_key = symbol.replace("/", "_")
    
    # 查找匹配的模型（如 lightgbm_BTC_USDT_4h_20260119_123456），排除 _bak 目录，按 mtime 最新选用
    matching = [d for d in dirs if symbol_key in d.name and f"_{timeframe}_" in d.name]
    if matching:
        return max(matching, key=lambda d: d.stat().st_mtime)
    return None


def _find_symbol_timeframe_models(
    symbol: str, timeframe: str, max_models: int = 3, models_root: Optional[Path] = None
) -> List[Path]:
    """
    查找同一 symbol+timeframe 的多个模型，按 mtime 倒序，最多 max_models 个，用于集成。
    models_root: 模型根目录，默认 data/models；用于多套模型对比。
    """
    root = Path(models_root) if models_root is not None else MODELS_DIR
    dirs = [d for d in root.iterdir() if d.is_dir() and (d / "model.joblib").exists() and "_bak" not in d.name]
    symbol_key = symbol.replace("/", "_")
    matching = [d for d in dirs if symbol_key in d.name and f"_{timeframe}_" in d.name]
    if not matching:
        return []
    matching = sorted(matching, key=lambda d: d.stat().st_mtime, reverse=True)
    return matching[:max_models]


def _find_all_symbol_timeframe_models() -> Dict[str, Path]:
    """
    查找所有按 symbol+timeframe 分类的模型。
    返回 { "BTC/USDT_4h": model_dir_path, ... }
    """
    result = {}
    dirs = [d for d in MODELS_DIR.iterdir() if d.is_dir() and (d / "model.joblib").exists() and "_bak" not in d.name]
    
    for symbol in SYMBOLS:
        symbol_key = symbol.replace("/", "_")
        for tf in TIMEFRAMES:
            matching = [d for d in dirs if symbol_key in d.name and f"_{tf}_" in d.name]
            if matching:
                key = f"{symbol}_{tf}"
                result[key] = max(matching, key=lambda d: d.stat().st_mtime)
    
    return result


def load_predictor(
    model_dir: Optional[Path] = None,
    timeframe: Optional[str] = None,
) -> Tuple[object, List[str], dict]:
    """
    加载模型。
    
    参数:
        model_dir: 模型目录（可选）
        timeframe: 时间周期，优先加载对应的专用模型
    
    返回 (model, feature_names, metadata)
    """
    model_dir = model_dir or _latest_model_dir(timeframe)
    model = joblib.load(model_dir / "model.joblib")
    meta_path = model_dir / "metadata.json"
    meta = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    feats = meta.get("feature_names", [])
    if (model_dir / "calibrator.joblib").exists():
        # 抑制 sklearn 版本不匹配警告（影响很小，可忽略）
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
            meta["calibrator"] = joblib.load(model_dir / "calibrator.joblib")
    else:
        meta["calibrator"] = None
    return model, feats, meta


def load_all_timeframe_predictors() -> Dict[str, Tuple[object, List[str], dict]]:
    """
    加载所有按 timeframe 分类的模型。
    返回 { timeframe: (model, feature_names, metadata) }
    """
    tf_models = _find_all_timeframe_models()
    result = {}
    
    for tf, model_dir in tf_models.items():
        try:
            model, feats, meta = load_predictor(model_dir, timeframe=tf)
            result[tf] = (model, feats, meta)
        except Exception as e:
            print(f"警告: 加载 {tf} 模型失败: {e}")
    
    return result


def load_all_symbol_timeframe_predictors() -> Dict[str, Tuple[object, List[str], dict]]:
    """
    加载所有按 symbol+timeframe 分类的模型。
    返回 { "BTC/USDT_15m": (model, feature_names, metadata), ... }
    
    这确保每个交易对使用其专属的模型，避免模型混用。
    """
    st_models = _find_all_symbol_timeframe_models()
    result = {}
    
    for key, model_dir in st_models.items():
        try:
            model, feats, meta = load_predictor(model_dir)
            result[key] = (model, feats, meta)
        except Exception as e:
            print(f"警告: 加载 {key} 模型失败: {e}")
    
    return result


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    for c in ("open", "high", "low", "close", "volume"):
        if c not in df.columns:
            raise ValueError(f"需要列: {c}")
    return df


def apply_calibration(
    calibrator: object,
    method: str,
    probs: Union[float, np.ndarray],
) -> Union[float, np.ndarray]:
    """
    对模型原始 P(UP) 做校准。
    支持 isotonic（等渗回归）、sigmoid（Platt）、temperature_scaling（温度缩放）。
    probs: 标量或数组；返回同形状，且 clip 到 [0,1]。
    """
    p = np.asarray(probs, dtype=float).reshape(-1, 1)
    if method == "isotonic":
        out = calibrator.predict(p)
    elif method == "sigmoid":
        out = calibrator.predict_proba(p)[:, 1]
    elif method == "temperature_scaling":
        # calibrator 是 dict: {"temperature": float}
        T = calibrator.get("temperature", 1.0) if isinstance(calibrator, dict) else 1.0
        eps = 1e-7
        p_flat = np.clip(p.ravel(), eps, 1 - eps)
        logits = np.log(p_flat / (1 - p_flat))
        scaled = logits / max(T, 0.01)
        out = 1 / (1 + np.exp(-scaled))
    else:
        return probs
    out = np.clip(np.asarray(out).ravel(), 0.0, 1.0)
    return float(out[0]) if out.size == 1 else out


def predict_one(
    model,
    feature_names: List[str],
    df: pd.DataFrame,
    calibrator: Optional[object] = None,
    calibration_method: Optional[str] = None,
) -> Tuple[int, float]:
    """
    对最后一行做下一根K线的涨跌预测。
    df: 已包含足够历史的 OHLCV（计算指标会用到最后一行）。
    calibrator/calibration_method: 若均非 None，则对原始 prob 做校准。
    返回 (pred: 0=DOWN/1=UP, prob_up: 模型给出的 UP 概率 P(UP)，经校准时为校准后)。

    概率与阈值逻辑（与训练时一致）见 docs/概率与阈值逻辑.md。
    """
    df = _ensure_ohlcv(df)
    full = build_features(df)
    missing = [f for f in feature_names if f not in full.columns]
    if missing:
        raise ValueError(
            f"特征与模型不一致：缺少 {len(missing)} 个特征（示例: {missing[:5]}{'...' if len(missing) > 5 else ''}）。"
            f"数据 {len([c for c in feature_names if c in full.columns])} 维，模型需 {len(feature_names)} 维。"
            "请用当前特征工程重新训练该交易对的模型（如 SOL 等）。"
        )
    # 若最后一行有 NaN（如 mtf_* 因 1h/4h 未对齐、或边界 K 线），先 ffill+bfill 兜底再取末行
    filled = full[feature_names].ffill().bfill()
    last = filled.iloc[[-1]]
    if last.isna().any().any():
        raise ValueError("最后一行特征含 NaN，历史 K 线可能不足")
    # 双阈值/实盘需真实概率，必须用 predict_proba；predict() 返回 0/1 会导致置信度错误
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(last)[0, 1])
    else:
        prob = float(model.predict(last)[0])
    # 可选：环境变量 SKIP_CALIBRATION=1 时不用校准（用于对比 XRP_20_80 等双阈值组合：校准常把概率压向 0.5，导致很少触发 0.8/0.2）
    if calibrator is not None and calibration_method and (os.environ.get("SKIP_CALIBRATION") != "1"):
        prob = apply_calibration(calibrator, calibration_method, prob)
    pred = 1 if prob >= 0.5 else 0
    return pred, prob


def predict(
    symbol: str,
    timeframe: str,
    model_dir: Optional[Path] = None,
    use_local: bool = True,
    use_symbol_model: bool = True,
) -> Tuple[str, float]:
    """
    对 symbol+timeframe 预测下一根 K 线涨跌。
    
    参数:
        symbol: 交易对（如 "BTC/USDT"）
        timeframe: 时间周期（如 "15m"）
        model_dir: 模型目录（可选，指定则直接使用）
        use_local: 是否优先使用本地数据
        use_symbol_model: 是否使用 symbol+timeframe 专用模型（推荐 True）
    
    返回 ("UP"|"DOWN", confidence)
    
    注意：
        use_symbol_model=True 时，会确保 BTC 使用 BTC 模型，ETH 使用 ETH 模型，
        避免模型混用导致预测不准确。
    """
    if model_dir is not None:
        model, feats, meta = load_predictor(model_dir)
    elif use_symbol_model:
        specific_model_dir = _find_symbol_timeframe_model(symbol, timeframe)
        if specific_model_dir:
            model, feats, meta = load_predictor(specific_model_dir)
        else:
            print(f"警告: 未找到 {symbol} {timeframe} 专用模型，使用 timeframe 通用模型")
            model, feats, meta = load_predictor(timeframe=timeframe)
    else:
        model, feats, meta = load_predictor(timeframe=timeframe)
    
    # 加载数据
    if use_local:
        df = load_ohlcv(symbol, timeframe)
        if df.empty or len(df) < 100:
            df = fetch_latest(symbol, timeframe, limit=500)
    else:
        df = fetch_latest(symbol, timeframe, limit=500)
    
    if df.empty or len(df) < 50:
        raise ValueError(f"数据不足: {symbol} {timeframe}")

    # 若模型含 mtf_*（训练时用了 --add-multi-timeframe），预测前补上 1h/4h 特征
    if any((f or "").startswith("mtf_") for f in (feats or [])):
        df = add_multi_timeframe_features(df, symbol)

    pred, prob = predict_one(
        model, feats, df,
        calibrator=meta.get("calibrator"),
        calibration_method=meta.get("calibration"),
    )
    label = "UP" if pred == 1 else "DOWN"
    conf = prob if pred == 1 else (1 - prob)
    return label, round(conf, 4)


def predict_all(
    symbols: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None,
    model_dir: Optional[Path] = None,
    use_symbol_models: bool = True,
) -> dict:
    """
    批量预测。
    
    参数:
        symbols: 交易对列表
        timeframes: 时间周期列表
        model_dir: 模型目录（可选，指定则所有预测使用同一模型）
        use_symbol_models: 是否使用 symbol+timeframe 专用模型（推荐 True）
    
    返回 { (symbol, tf): {"direction": "UP"|"DOWN", "confidence": float, "prob": float } }
    
    注意：
        use_symbol_models=True 时，会确保每个交易对使用其专属模型，
        例如 BTC 使用 BTC 模型，ETH 使用 ETH 模型，避免混用。
    """
    symbols = symbols or SYMBOLS
    timeframes = timeframes or TIMEFRAMES
    
    # 预加载所有 symbol+timeframe 专用模型
    st_models = {}
    if use_symbol_models and model_dir is None:
        st_models = load_all_symbol_timeframe_predictors()
    
    out = {}
    for s in symbols:
        for tf in timeframes:
            try:
                model_key = f"{s}_{tf}"
                
                # 优先使用 symbol+timeframe 专用模型
                if model_key in st_models:
                    model, feats, meta = st_models[model_key]
                    df = load_ohlcv(s, tf)
                    if df.empty or len(df) < 100:
                        df = fetch_latest(s, tf, limit=500)
                    if df.empty or len(df) < 50:
                        raise ValueError(f"数据不足: {s} {tf}")
                    if any((f or "").startswith("mtf_") for f in (feats or [])):
                        df = add_multi_timeframe_features(df, s)
                    pred, prob = predict_one(
                        model, feats, df,
                        calibrator=meta.get("calibrator"),
                        calibration_method=meta.get("calibration"),
                    )
                    label = "UP" if pred == 1 else "DOWN"
                    conf = prob if pred == 1 else (1 - prob)
                    out[(s, tf)] = {
                        "direction": label, 
                        "confidence": round(conf, 4),
                        "prob": round(prob, 4),
                        "model": model_key,
                    }
                else:
                    # 回退到 predict 函数（会尝试查找专用模型或使用通用模型）
                    label, conf = predict(s, tf, model_dir=model_dir, use_symbol_model=use_symbol_models)
                    out[(s, tf)] = {"direction": label, "confidence": conf, "model": "fallback"}
            except Exception as e:
                out[(s, tf)] = {"direction": None, "confidence": 0, "prob": 0.5, "error": str(e)}
    return out


def predict_all_to_json(
    symbols: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None,
    model_dir: Optional[Path] = None,
    use_symbol_models: bool = True,
) -> List[dict]:
    """
    批量预测，输出为 JSON 友好格式（list of dict）。
    用于和 aggregate_predictions.py 对接。
    
    参数:
        use_symbol_models: 是否使用 symbol+timeframe 专用模型（推荐 True）
    """
    results = predict_all(symbols, timeframes, model_dir, use_symbol_models)
    
    output = []
    for (s, tf), data in results.items():
        output.append({
            "pair": s,
            "timeframe": tf,
            "direction": data.get("direction"),
            "prob": data.get("prob", data.get("confidence", 0.5)),
            "confidence": data.get("confidence", 0),
            "error": data.get("error"),
        })
    
    return output
