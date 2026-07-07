"""
模型训练：集成 LightGBM 二分类，超参数从 config/lightgbm_params.json、config/freqai_config.json 读取，训练后保存到 data/models/。
支持按 timeframe (15m/1h/4h) 分别训练独立模型，使用时间序列交叉验证。

注意：本脚本不使用任何内存自动清理功能（如 gc.collect()），以保持最佳性能。
"""

import json
import joblib
import lightgbm as lgb
import pandas as pd
import numpy as np
# 明确禁用垃圾回收的自动清理，使用系统默认行为
# 不导入 gc 模块，不调用任何内存清理函数
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
import multiprocessing
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .data_fetcher import SYMBOLS, TIMEFRAMES, load_ohlcv
from .feature_engineering import prepare_train_data, get_feature_columns, build_features

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "data" / "models"
FEATURE_ALLOWLIST_PATH = CONFIG_DIR / "feature_allowlist.txt"


def load_feature_allowlist() -> Optional[List[str]]:
    """读取 config/feature_allowlist.txt，每行一个特征名；不存在或空则返回 None。"""
    if not FEATURE_ALLOWLIST_PATH.exists():
        return None
    lines = [s.strip() for s in FEATURE_ALLOWLIST_PATH.read_text(encoding="utf-8").strip().splitlines() if s.strip()]
    return lines if lines else None


def compute_feature_allowlist(model, feature_names: List[str], keep_top_pct: float) -> List[str]:
    """
    按 gain 重要性排序，保留前 keep_top_pct（如 0.7=70%）的特征名。
    model: LightGBM Booster；feature_names: 与训练时 X.columns 顺序一致。
    """
    imp = np.asarray(model.feature_importance(importance_type="gain"))
    names = list(feature_names)
    n = min(len(imp), len(names))
    if n == 0:
        return []
    imp, names = imp[:n], names[:n]
    pairs = sorted(zip(names, imp), key=lambda x: -float(x[1]))
    keep_n = max(1, int(len(pairs) * keep_top_pct))
    return [p[0] for p in pairs[:keep_n]]


def _load_lightgbm_params(params_file: Optional[str] = None, num_threads: Optional[int] = None) -> Dict[str, Any]:
    """
    加载 LightGBM 参数。
    
    Args:
        params_file: 参数文件路径
        num_threads: CPU 线程数，如果提供会覆盖参数文件中的设置
    
    参数:
        params_file: 可选，指定参数文件路径（相对于 CONFIG_DIR 或绝对路径）
                     如果为 None，按优先级：optuna > default
    """
    if params_file:
        # 使用指定的参数文件
        if Path(params_file).is_absolute():
            params_path = Path(params_file)
        else:
            params_path = CONFIG_DIR / params_file
        if params_path.exists():
            print(f"📊 使用指定参数文件: {params_path}")
            params = json.loads(params_path.read_text(encoding="utf-8"))
        else:
            raise FileNotFoundError(f"参数文件不存在: {params_path}")
    else:
        # 默认优先级：Optuna > default
        optuna_path = CONFIG_DIR / "lightgbm_params_optuna.json"
        default_path = CONFIG_DIR / "lightgbm_params.json"
        
        if optuna_path.exists():
            print("📊 使用 Optuna 优化后的参数")
            params = json.loads(optuna_path.read_text(encoding="utf-8"))
        elif default_path.exists():
            params = json.loads(default_path.read_text(encoding="utf-8"))
        else:
            return {"objective": "binary", "metric": "binary_logloss", "verbosity": -1}
    
    # 移除注释字段
    params = {k: v for k, v in params.items() if not k.startswith("_")}
    
    # 检测 GPU 是否可用
    if params.get("device") == "gpu":
        try:
            import lightgbm as lgb
            # 尝试创建一个小数据集测试 GPU
            test_data = lgb.Dataset([[1, 2], [3, 4]], label=[0, 1])
            test_params = {"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0, 
                          "objective": "binary", "num_iterations": 1, "verbosity": -1}
            lgb.train(test_params, test_data, num_boost_round=1)
            print("✅ GPU 加速已启用")
        except Exception as e:
            print(f"⚠️ GPU 不可用，回退到 CPU: {e}")
            params.pop("device", None)
            params.pop("gpu_platform_id", None)
            params.pop("gpu_device_id", None)
    
    # 如果提供了 num_threads，覆盖参数文件中的设置
    if num_threads is not None:
        params["num_threads"] = num_threads
    elif "num_threads" not in params:
        # 如果参数文件中也没有设置，默认使用全部核心（但可以通过命令行覆盖）
        import os
        params["num_threads"] = os.cpu_count() or 4
    
    return params


def _load_freqai_config() -> Dict[str, Any]:
    p = CONFIG_DIR / "freqai_config.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("freqai", {})


def _train_days_cutoff(days: int) -> int:
    """计算 N 天前的时间戳（用于筛选训练数据起始点）"""
    return int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp() * 1000)


def _holdout_cutoff(holdout_days: int) -> int:
    """计算 N 天前的时间戳（用于排除最近数据，留给回测）"""
    return int((datetime.now(timezone.utc) - timedelta(days=holdout_days)).timestamp() * 1000)


def load_training_data(
    symbols: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None,
    train_period_days: Optional[int] = None,
    holdout_days: int = 0,
    start_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    从 data/raw 加载 K 线用于训练。
    
    参数:
        train_period_days: 使用最近多少天的数据（与 start_date 二选一）
        holdout_days: 排除最近多少天的数据（留给回测，防止数据泄露）
        start_date: 训练数据起始日期（ISO 格式，如 "2024-06-01"），优先级高于 train_period_days
    """
    symbols = symbols or SYMBOLS
    timeframes = timeframes or TIMEFRAMES
    dfs = []
    for s in symbols:
        for tf in timeframes:
            path = DATA_RAW / f"{s.replace('/', '_').lower()}_{tf}.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            df["symbol"] = s
            df["timeframe"] = tf
            dfs.append(df)
    if not dfs:
        raise FileNotFoundError("data/raw 下没有 K 线文件，请先运行: python -m src.python.data_fetcher --download-historical")
    out = pd.concat(dfs, ignore_index=True)
    
    # 筛选训练数据的起始点（start_date 优先级更高）
    if start_date:
        try:
            start_dt = pd.to_datetime(start_date, utc=True)
            start_ts = int(start_dt.timestamp() * 1000)
            if "timestamp" in out.columns:
                out = out[out["timestamp"] >= start_ts]
            elif "date" in out.columns:
                out = out[pd.to_datetime(out["date"], utc=True) >= start_dt]
            print(f"📅 训练数据起始日期: {start_date} (UTC)")
        except Exception as e:
            raise ValueError(f"无效的起始日期格式 '{start_date}'，请使用 ISO 格式如 '2024-06-01': {e}")
    elif train_period_days:
        ts = _train_days_cutoff(train_period_days)
        if "timestamp" in out.columns:
            out = out[out["timestamp"] >= ts]
        elif "date" in out.columns:
            out = out[pd.to_datetime(out["date"]).astype("int64") // 10**6 >= ts]
    
    # 排除最近 N 天的数据（留给回测，防止数据泄露）
    if holdout_days > 0:
        holdout_ts = _holdout_cutoff(holdout_days)
        if "timestamp" in out.columns:
            out = out[out["timestamp"] < holdout_ts]
        elif "date" in out.columns:
            out = out[pd.to_datetime(out["date"]).astype("int64") // 10**6 < holdout_ts]
    
    return out.sort_values(["timestamp"] if "timestamp" in out.columns else ["date"]).reset_index(drop=True)


def load_training_data_by_timeframe(
    timeframe: str,
    symbols: Optional[List[str]] = None,
    train_period_days: Optional[int] = None,
    holdout_days: int = 0,
    start_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    加载指定 timeframe 的所有 symbols K 线数据用于训练。
    
    参数:
        holdout_days: 排除最近多少天的数据（留给回测，防止数据泄露）
        start_date: 训练数据起始日期（ISO 格式，如 "2024-06-01"），优先级高于 train_period_days
    """
    symbols = symbols or SYMBOLS
    dfs = []
    for s in symbols:
        path = DATA_RAW / f"{s.replace('/', '_').lower()}_{timeframe}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df["symbol"] = s
        df["timeframe"] = timeframe
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"data/raw 下没有 {timeframe} K 线文件")
    out = pd.concat(dfs, ignore_index=True)
    
    # 筛选训练数据的起始点（start_date 优先级更高）
    if start_date:
        try:
            start_dt = pd.to_datetime(start_date, utc=True)
            start_ts = int(start_dt.timestamp() * 1000)
            if "timestamp" in out.columns:
                out = out[out["timestamp"] >= start_ts]
            elif "date" in out.columns:
                out = out[pd.to_datetime(out["date"], utc=True) >= start_dt]
        except Exception as e:
            raise ValueError(f"无效的起始日期格式 '{start_date}'，请使用 ISO 格式如 '2024-06-01': {e}")
    elif train_period_days:
        ts = _train_days_cutoff(train_period_days)
        if "timestamp" in out.columns:
            out = out[out["timestamp"] >= ts]
        elif "date" in out.columns:
            out = out[pd.to_datetime(out["date"]).astype("int64") // 10**6 >= ts]
    
    # 排除最近 N 天的数据（留给回测，防止数据泄露）
    if holdout_days > 0:
        holdout_ts = _holdout_cutoff(holdout_days)
        if "timestamp" in out.columns:
            out = out[out["timestamp"] < holdout_ts]
        elif "date" in out.columns:
            out = out[pd.to_datetime(out["date"]).astype("int64") // 10**6 < holdout_ts]
    
    return out.sort_values(["timestamp"] if "timestamp" in out.columns else ["date"]).reset_index(drop=True)


def compute_detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """计算详细的评估指标。"""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc"] = 0.5  # 只有一个类别时
    try:
        metrics["log_loss"] = float(log_loss(y_true, y_prob))
    except ValueError:
        metrics["log_loss"] = float("nan")
    
    # UP/DOWN 的比例
    metrics["up_ratio_true"] = float(y_true.mean())
    metrics["up_ratio_pred"] = float(y_pred.mean())
    
    return metrics


def fit_calibrator(
    raw_probs: np.ndarray,
    y_true: np.ndarray,
    method: str = "isotonic",
) -> object:
    """
    在验证集上拟合概率校准器，使模型输出的 P(UP) 更接近真实胜率。
    
    参数:
        raw_probs: 模型原始输出的 P(UP)， shape (n,)
        y_true: 真实标签 0/1，shape (n,)
        method: "isotonic"（等渗回归，非参数）、"sigmoid"（Platt，参数少更稳）
                或 "temperature_scaling"（温度缩放，1 个参数，最简单）
    
    返回:
        拟合好的校准器，用于 apply_calibration(calibrator, method, probs)
    """
    X = np.asarray(raw_probs, dtype=float).reshape(-1, 1)
    y = np.asarray(y_true, dtype=int).ravel()
    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(X.ravel(), y)  # sklearn 的 IsotonicRegression.fit 接受 (n,) 或 (n,1)
        return cal
    if method == "sigmoid":
        cal = LogisticRegression(C=1e10, max_iter=1000)  # C 大≈无正则，拟合 P(UP)
        cal.fit(X, y)
        return cal
    if method == "temperature_scaling":
        cal = _fit_temperature_scaling(X.ravel(), y)
        return cal
    raise ValueError(f'calibration method 须为 "isotonic"、"sigmoid" 或 "temperature_scaling"，当前: {method}')


def _fit_temperature_scaling(raw_probs: np.ndarray, y_true: np.ndarray) -> dict:
    """
    温度缩放：找最优 T 使 scaled_logits = logits / T 的 NLL 最小。
    
    返回 dict: {"temperature": float} ，在 apply_calibration 中用 method="temperature_scaling" 应用。
    """
    from scipy.optimize import minimize_scalar

    # 将概率转为 logit
    eps = 1e-7
    probs_clipped = np.clip(raw_probs, eps, 1 - eps)
    logits = np.log(probs_clipped / (1 - probs_clipped))

    def nll(T):
        scaled = logits / max(T, 0.01)
        p = 1 / (1 + np.exp(-scaled))
        p = np.clip(p, eps, 1 - eps)
        return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return {"temperature": float(result.x)}


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier Score = mean((y_prob - y_true)^2)，越低越好。"""
    return float(np.mean((np.asarray(y_prob) - np.asarray(y_true)) ** 2))


def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error：衡量预测概率与真实频率的偏差。
    
    将预测概率分成 n_bins 个等宽区间，对每个区间计算
    |avg_confidence - avg_accuracy| * (n_in_bin / n_total)，求和。
    
    返回 ECE 值（0-1，越低越好）。
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if not np.any(mask):
            continue
        avg_conf = y_prob[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += abs(avg_conf - avg_acc) * mask.sum() / n
    return float(ece)


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob_before: np.ndarray,
    y_prob_after: Optional[np.ndarray] = None,
    n_bins: int = 10,
    output_path: Optional[Path] = None,
    title: str = "Reliability Diagram",
) -> Optional[Path]:
    """
    绘制校准前后的 reliability diagram（校准曲线）。
    
    参数:
        y_true: 真实标签
        y_prob_before: 校准前的预测概率
        y_prob_after: 校准后的预测概率（可选）
        n_bins: 分箱数
        output_path: 保存路径（默认不保存）
        title: 图标题
    
    返回:
        保存的文件路径（若 output_path 不为 None）
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("警告: matplotlib 未安装，跳过 reliability diagram 绘制")
        return None

    y_true = np.asarray(y_true).ravel()
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    def _plot_one(probs, label, color):
        bin_edges = np.linspace(0, 1, n_bins + 1)
        bin_centers = []
        bin_accs = []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (probs >= lo) & (probs < hi)
            if not np.any(mask):
                continue
            bin_centers.append(probs[mask].mean())
            bin_accs.append(y_true[mask].mean())
        ax.plot(bin_centers, bin_accs, "o-", label=label, color=color, markersize=6)

    _plot_one(np.asarray(y_prob_before).ravel(), "Before calibration", "tab:blue")
    if y_prob_after is not None:
        _plot_one(np.asarray(y_prob_after).ravel(), "After calibration", "tab:orange")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        return output_path
    plt.close(fig)
    return None


def select_best_calibration(
    raw_probs: np.ndarray,
    y_true: np.ndarray,
    methods: Optional[list] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[object, str, Dict[str, Any]]:
    """
    自动选择 Brier Score 最低的校准方法。
    
    参数:
        raw_probs: 模型原始 P(UP)
        y_true: 真实标签
        methods: 要测试的方法列表（默认 3 种全试）
        output_dir: 若提供，保存 reliability diagram
    
    返回:
        (best_calibrator, best_method, metrics_dict)
        metrics_dict: {method: {brier, ece, brier_before, ece_before}}
    """
    from .predictor import apply_calibration

    if methods is None:
        methods = ["isotonic", "sigmoid", "temperature_scaling"]

    y_true = np.asarray(y_true).ravel()
    raw_probs = np.asarray(raw_probs).ravel()

    brier_before = compute_brier_score(y_true, raw_probs)
    ece_before = compute_ece(y_true, raw_probs)

    results = {}
    best_brier = float("inf")
    best_cal = None
    best_method = None

    for method in methods:
        try:
            cal = fit_calibrator(raw_probs, y_true, method=method)
            calibrated = apply_calibration(cal, method, raw_probs)
            brier = compute_brier_score(y_true, calibrated)
            ece = compute_ece(y_true, calibrated)
            results[method] = {
                "brier_before": brier_before,
                "brier_after": brier,
                "brier_improvement": brier_before - brier,
                "ece_before": ece_before,
                "ece_after": ece,
                "ece_improvement": ece_before - ece,
            }
            if brier < best_brier:
                best_brier = brier
                best_cal = cal
                best_method = method

            # 保存 reliability diagram
            if output_dir is not None:
                plot_reliability_diagram(
                    y_true, raw_probs, calibrated,
                    output_path=Path(output_dir) / f"reliability_{method}.png",
                    title=f"Reliability: {method} (Brier {brier_before:.4f} → {brier:.4f})",
                )
        except Exception as e:
            results[method] = {"error": str(e)}

    return best_cal, best_method, results


def optuna_tune_lightgbm(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int = 100,
    n_splits: int = 5,
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """
    使用 Optuna 自动搜索 LightGBM 最优超参数。
    
    参数:
        n_trials: 搜索次数（越多越好，但更耗时）
        n_splits: 时间序列交叉验证折数
        timeout: 超时时间（秒），None 表示不限制
    
    返回:
        最优参数字典
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
        from optuna.pruners import MedianPruner
    except ImportError:
        print("⚠️ 未安装 optuna，请运行: pip install optuna")
        return _load_lightgbm_params()
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            # 核心参数（搜索范围）
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "max_depth": trial.suggest_int("max_depth", 5, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 500, 3000),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            # 正则化
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            # 防止过拟合
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
        }
        # 时间序列交叉验证（每折报告一次，供 MedianPruner 早停）
        tscv = TimeSeriesSplit(n_splits=n_splits)
        auc_scores = []
        
        for split_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]
            
            dtrain = lgb.Dataset(X_tr, label=y_tr)
            dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)
            
            model = lgb.train(
                params,
                dtrain,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
            )
            
            y_prob = model.predict(X_va)
            try:
                auc = roc_auc_score(y_va, y_prob)
                auc_scores.append(auc)
            except ValueError:
                pass
            
            mean_auc = np.mean(auc_scores) if auc_scores else 0.5
            trial.report(mean_auc, step=split_idx)
            if trial.should_prune():
                import optuna
                raise optuna.TrialPruned()
        
        return np.mean(auc_scores) if auc_scores else 0.5
    
    print(f"🔍 Optuna 超参数搜索中... (最多 {n_trials} 次试验{f'，超时 {timeout}s' if timeout else '，无时间限制'})")
    
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=1, interval_steps=1),
    )
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,  # None=不限制
        show_progress_bar=True,
    )
    
    best_params = study.best_params
    best_params["objective"] = "binary"
    best_params["metric"] = "auc"
    best_params["verbosity"] = -1
    
    print(f"✅ 最优 AUC: {study.best_value:.4f}")
    print(f"   最优参数: {best_params}")
    
    return best_params


def train_one(
    X: pd.DataFrame,
    y: pd.Series,
    params: Optional[Dict] = None,
    early_stopping_rounds: int = 20,
    val_ratio: float = 0.2,
    purge_embargo: int = 0,
    use_scale_pos_weight: bool = True,
    calibration_method: Optional[str] = None,
) -> Tuple[object, Dict[str, Any], Optional[object]]:
    """训练一个 LightGBM 二分类模型，返回 (model, metrics_dict, calibrator 或 None)。"""
    params = dict(params or _load_lightgbm_params())
    params.setdefault("objective", "binary")
    params.setdefault("verbosity", -1)
    # 确保 num_threads 已设置（如果 params 中没有，从环境变量读取）
    if "num_threads" not in params:
        import os
        if "OMP_NUM_THREADS" in os.environ:
            params["num_threads"] = int(os.environ["OMP_NUM_THREADS"])

    n = len(X)
    if n < 100:
        raise ValueError("样本过少，至少需要 100 条")
    split = int(n * (1 - val_ratio))
    # Purging/Embargo：train 末尾删 purge 行，val 开头删 embargo 行，减小泄漏
    pe = max(0, int(purge_embargo))
    if pe > 0 and split - pe > 50 and split + pe < n - 50:
        train_end = split - pe
        val_start = split + pe
        X_tr, X_va = X.iloc[:train_end], X.iloc[val_start:]
        y_tr, y_va = y.iloc[:train_end], y.iloc[val_start:]
    else:
        X_tr, X_va = X.iloc[:split], X.iloc[split:]
        y_tr, y_va = y.iloc[:split], y.iloc[split:]

    if use_scale_pos_weight:
        r = float(y_tr.mean())
        if r and (r < 0.45 or r > 0.55):
            params = {**params, "scale_pos_weight": (1 - r) / r}

    dtrain = lgb.Dataset(X_tr, label=y_tr)
    dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

    callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
    model = lgb.train(
        params,
        dtrain,
        valid_sets=[dval],
        callbacks=callbacks,
    )
    
    y_prob = model.predict(X_va)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = compute_detailed_metrics(y_va.values, y_pred, y_prob)

    calibrator = None
    if calibration_method in ("isotonic", "sigmoid") and len(y_va) >= 50:
        calibrator = fit_calibrator(y_prob, y_va.values, method=calibration_method)

    return model, metrics, calibrator


def train_with_cv(
    X: pd.DataFrame,
    y: pd.Series,
    params: Optional[Dict] = None,
    n_splits: int = 5,
    early_stopping_rounds: int = 50,
    verbose: bool = True,
    purge_embargo: int = 0,
    use_scale_pos_weight: bool = True,
    calibration_method: Optional[str] = None,
) -> Tuple[object, Dict[str, Any], Optional[object]]:
    """
    使用时间序列交叉验证训练 LightGBM 模型。
    返回 (最终模型, fold 指标汇总, calibrator 或 None)。
    
    参数:
        verbose: 是否打印进度（并行训练时设为 False）
        purge_embargo: train 末/val 头各删几行，防泄漏，0=关闭
        use_scale_pos_weight: 类别不平衡时自动设 scale_pos_weight
        calibration_method: "isotonic"|"sigmoid" 时在验证集上拟合概率校准器
    """
    params = dict(params or _load_lightgbm_params())
    params.setdefault("objective", "binary")
    params.setdefault("verbosity", -1)
    # 确保 num_threads 已设置（如果 params 中没有，从环境变量读取）
    if "num_threads" not in params:
        import os
        if "OMP_NUM_THREADS" in os.environ:
            params["num_threads"] = int(os.environ["OMP_NUM_THREADS"])

    n = len(X)
    if n < 500:
        return train_one(X, y, params, early_stopping_rounds, purge_embargo=purge_embargo, use_scale_pos_weight=use_scale_pos_weight, calibration_method=calibration_method)

    pe = max(0, int(purge_embargo))

    # 使用 PurgedWalkForward 替代 TimeSeriesSplit（增加 purge/embargo 防泄漏）
    from .validation.purged_walk_forward import PurgedWalkForward
    pwf = PurgedWalkForward(
        n_splits=n_splits,
        purge_bars=pe if pe > 0 else None,
        purge_pct=0.01 if pe == 0 else 0,
        embargo_pct=0.005 if pe == 0 else 0,
        embargo_bars=pe if pe > 0 else None,
    )
    fold_metrics = []

    if verbose:
        purge_desc = f"purge_embargo={pe}" if pe else "purge_pct=1%, embargo_pct=0.5%"
        print(f"开始 {n_splits} 折 PurgedWalkForward 交叉验证 ({purge_desc})...")

    for fold_idx, (train_idx, val_idx) in enumerate(pwf.split(X)):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx], y.iloc[val_idx]

        p = dict(params)
        if use_scale_pos_weight:
            r = float(y_tr.mean())
            if r and (r < 0.45 or r > 0.55):
                p["scale_pos_weight"] = (1 - r) / r

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_va, label=y_va, reference=dtrain)

        callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
        model = lgb.train(
            p,
            dtrain,
            valid_sets=[dval],
            callbacks=callbacks,
        )
        
        y_prob = model.predict(X_va)
        y_pred = (y_prob >= 0.5).astype(int)
        fold_m = compute_detailed_metrics(y_va.values, y_pred, y_prob)
        fold_m["fold"] = fold_idx + 1
        fold_m["train_samples"] = len(train_idx)
        fold_m["val_samples"] = len(val_idx)
        fold_metrics.append(fold_m)
        
        if verbose:
            print(f"  Fold {fold_idx + 1}: accuracy={fold_m['accuracy']:.4f}, auc={fold_m['auc']:.4f}")
    
    # 使用全部数据训练最终模型
    if verbose:
        print("训练最终模型（使用全部数据）...")
    split = int(n * 0.8)
    if pe and split - pe > 100 and split + pe < n - 50:
        X_tr_final = X.iloc[: split - pe]
        X_va_final = X.iloc[split + pe :]
        y_tr_final = y.iloc[: split - pe]
        y_va_final = y.iloc[split + pe :]
    else:
        X_tr_final, X_va_final = X.iloc[:split], X.iloc[split:]
        y_tr_final, y_va_final = y.iloc[:split], y.iloc[split:]

    p_final = dict(params)
    if use_scale_pos_weight:
        r = float(y_tr_final.mean())
        if r and (r < 0.45 or r > 0.55):
            p_final["scale_pos_weight"] = (1 - r) / r

    dtrain_final = lgb.Dataset(X_tr_final, label=y_tr_final)
    dval_final = lgb.Dataset(X_va_final, label=y_va_final, reference=dtrain_final)

    callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
    final_model = lgb.train(
        p_final,
        dtrain_final,
        valid_sets=[dval_final],
        callbacks=callbacks,
    )
    
    # 汇总指标
    cv_summary = {
        "n_splits": n_splits,
        "total_samples": n,
        "fold_metrics": fold_metrics,
        "mean_accuracy": float(np.mean([m["accuracy"] for m in fold_metrics])),
        "std_accuracy": float(np.std([m["accuracy"] for m in fold_metrics])),
        "mean_auc": float(np.mean([m["auc"] for m in fold_metrics])),
        "std_auc": float(np.std([m["auc"] for m in fold_metrics])),
        "mean_f1": float(np.mean([m["f1"] for m in fold_metrics])),
        "std_f1": float(np.std([m["f1"] for m in fold_metrics])),
    }
    
    # 最终模型在验证集上的指标
    y_prob_final = final_model.predict(X_va_final)
    y_pred_final = (y_prob_final >= 0.5).astype(int)
    final_metrics = compute_detailed_metrics(y_va_final.values, y_pred_final, y_prob_final)
    cv_summary["final_metrics"] = final_metrics

    calibrator = None
    if calibration_method in ("isotonic", "sigmoid") and len(y_va_final) >= 50:
        calibrator = fit_calibrator(y_prob_final, y_va_final.values, method=calibration_method)

    if verbose:
        print(f"CV 平均: accuracy={cv_summary['mean_accuracy']:.4f}±{cv_summary['std_accuracy']:.4f}, "
              f"auc={cv_summary['mean_auc']:.4f}±{cv_summary['std_auc']:.4f}")
    
    return final_model, cv_summary, calibrator


def run_initial_training(
    symbols: Optional[List[str]] = None,
    timeframes: Optional[List[str]] = None,
    train_period_days: Optional[int] = None,
    model_suffix: str = "",
    use_cv: bool = False,
    label_threshold_pct: Optional[float] = None,
    primary_only: bool = False,
    calibration_method: Optional[str] = None,
    start_date: Optional[str] = None,
    models_dir: Optional[Path] = None,
    params_file: Optional[str] = None,
    num_threads: Optional[int] = None,
) -> Path:
    """
    首次训练：加载数据 -> 特征+标签 -> 训练 -> 保存到 data/models/。
    返回模型目录路径。
    
    参数:
        use_cv: 是否使用时间序列交叉验证
        label_threshold_pct: 涨跌幅阈值（百分比）
        primary_only: 如果为 True，只使用主要特征（KDJ、Range）
        calibration_method: "isotonic"|"sigmoid" 时在验证集上拟合概率校准并保存
    """
    freqai = _load_freqai_config()
    # 默认使用全部数据（2555天≈7年，覆盖2019年至今）
    train_period_days = train_period_days or freqai.get("train_period_days") or 2555
    early = freqai.get("model_training_parameters", {}).get("early_stopping_rounds", 50)

    df = load_training_data(symbols=symbols, timeframes=timeframes, train_period_days=train_period_days, start_date=start_date)
    X, y = prepare_train_data(df, label_threshold_pct=label_threshold_pct, primary_only=primary_only)
    if X.empty or y.empty:
        raise ValueError("prepare_train_data 结果为空，请检查数据与特征")

    params = _load_lightgbm_params(params_file=params_file, num_threads=num_threads)
    
    if use_cv:
        model, metrics, calibrator = train_with_cv(X, y, params=params, early_stopping_rounds=early, calibration_method=calibration_method)
    else:
        model, metrics, calibrator = train_one(X, y, params=params, early_stopping_rounds=early, calibration_method=calibration_method)

    output_dir = Path(models_dir) if models_dir else MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    name = f"lightgbm_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}{model_suffix}"
    out_dir = output_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, out_dir / "model.joblib")
    if calibrator is not None:
        joblib.dump(calibrator, out_dir / "calibrator.joblib")
    feature_names = list(X.columns)
    meta = {
        "feature_names": feature_names,
        "metrics": metrics,
        "train_period_days": train_period_days,
        "label_threshold_pct": label_threshold_pct,
        "primary_only": primary_only,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "timeframes": timeframes or TIMEFRAMES,
        "symbols": symbols or SYMBOLS,
    }
    if calibrator is not None:
        meta["calibration"] = calibration_method
    (out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Model saved: {out_dir}")
    print(f"Features ({len(feature_names)}): {feature_names[:10]}..." if len(feature_names) > 10 else f"Features: {feature_names}")
    print("Metrics:", metrics)
    return out_dir


def train_per_timeframe(
    symbols: Optional[List[str]] = None,
    train_period_days: Optional[int] = None,
    use_cv: bool = True,
    label_threshold_pct: Optional[float] = None,
    primary_only: bool = False,
    calibration_method: Optional[str] = None,
    start_date: Optional[str] = None,
    models_dir: Optional[Path] = None,
    params_file: Optional[str] = None,
    num_threads: Optional[int] = None,
) -> Dict[str, Path]:
    """
    按 timeframe 分别训练独立模型（15m / 1h / 4h）。
    返回 { timeframe: model_dir_path }
    
    参数:
        primary_only: 如果为 True，只使用主要特征（KDJ、Range）
        calibration_method: "isotonic"|"sigmoid" 时拟合概率校准并保存
    """
    freqai = _load_freqai_config()
    # 默认使用全部数据（2555天≈7年，覆盖2019年至今）
    train_period_days = train_period_days or freqai.get("train_period_days") or 2555
    early = freqai.get("model_training_parameters", {}).get("early_stopping_rounds", 50)
    params = _load_lightgbm_params(params_file=params_file, num_threads=num_threads)
    
    output_dir = Path(models_dir) if models_dir else MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    
    if models_dir:
        print(f"  [模型目录] 输出到: {output_dir}")
    
    results = {}
    all_metrics = {}
    
    for tf in TIMEFRAMES:
        print(f"\n{'='*60}")
        print(f"训练 {tf} 模型...")
        print(f"{'='*60}")
        
        try:
            df = load_training_data_by_timeframe(tf, symbols=symbols, train_period_days=train_period_days, start_date=start_date)
            X, y = prepare_train_data(df, label_threshold_pct=label_threshold_pct, primary_only=primary_only)
            
            if X.empty or y.empty:
                print(f"  警告: {tf} 数据不足，跳过")
                continue
            
            print(f"  样本数: {len(X)}, 特征数: {len(X.columns)}")
            print(f"  特征: {list(X.columns)[:8]}..." if len(X.columns) > 8 else f"  特征: {list(X.columns)}")
            print(f"  UP 比例: {y.mean():.4f}")
            
            if use_cv:
                model, metrics, calibrator = train_with_cv(X, y, params=params, early_stopping_rounds=early, calibration_method=calibration_method)
            else:
                model, metrics, calibrator = train_one(X, y, params=params, early_stopping_rounds=early, calibration_method=calibration_method)
            
            # 保存模型
            out_dir = output_dir / f"lightgbm_{tf}_{timestamp}"
            out_dir.mkdir(parents=True, exist_ok=True)
            
            joblib.dump(model, out_dir / "model.joblib")
            if calibrator is not None:
                joblib.dump(calibrator, out_dir / "calibrator.joblib")
            feature_names = list(X.columns)
            meta = {
                "feature_names": feature_names,
                "metrics": metrics,
                "train_period_days": train_period_days,
                "label_threshold_pct": label_threshold_pct,
                "primary_only": primary_only,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "timeframe": tf,
                "symbols": symbols or SYMBOLS,
                "sample_count": len(X),
            }
            if calibrator is not None:
                meta["calibration"] = calibration_method
            (out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            
            results[tf] = out_dir
            all_metrics[tf] = metrics
            
            print(f"  模型已保存: {out_dir}")
            
        except Exception as e:
            print(f"  错误: {e}")
            continue
    
    # 打印汇总
    print(f"\n{'='*60}")
    print("训练汇总")
    print(f"{'='*60}")
    for tf, m in all_metrics.items():
        if isinstance(m, dict):
            if "mean_accuracy" in m:
                # CV 结果
                print(f"{tf}: accuracy={m['mean_accuracy']:.4f}±{m['std_accuracy']:.4f}, "
                      f"auc={m['mean_auc']:.4f}±{m['std_auc']:.4f}")
            else:
                # 简单拆分结果
                print(f"{tf}: accuracy={m.get('accuracy', 0):.4f}, auc={m.get('auc', 0):.4f}")
    
    return results


def _train_single_model(
    args_tuple: Tuple,
) -> Tuple[str, Optional[Path], Optional[Dict], Optional[str], Optional[List[str]]]:
    """训练单个模型（用于并行）。返回 (key, out_dir, metrics, error, allowlist_or_None)。"""
    (s, tf, train_period_days, holdout_days, use_cv, label_threshold_pct, primary_only,
     lookahead, params, early, timestamp, models_dir, feature_drop_bottom_pct, feature_cols_allowlist,
     purge_embargo, use_scale_pos_weight, add_multi_timeframe, calibration_method, start_date, params_file, num_threads) = args_tuple
    
    # 如果提供了 num_threads，更新 params（确保所有训练调用都使用限制的线程数）
    if num_threads is not None:
        params = dict(params)
        params["num_threads"] = num_threads
        # 强制设置，确保生效
        import os
        os.environ["OMP_NUM_THREADS"] = str(num_threads)
        os.environ["MKL_NUM_THREADS"] = str(num_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(num_threads)

    key = f"{s.replace('/', '_')}_{tf}"

    try:
        path = DATA_RAW / f"{s.replace('/', '_').lower()}_{tf}.parquet"
        if not path.exists():
            return key, None, None, f"文件不存在: {path}", None

        df = pd.read_parquet(path)
        df["symbol"] = s
        df["timeframe"] = tf

        # 筛选训练数据起始点（start_date 优先级更高）
        if start_date:
            try:
                start_dt = pd.to_datetime(start_date, utc=True)
                start_ts = int(start_dt.timestamp() * 1000)
                if "timestamp" in df.columns:
                    df = df[df["timestamp"] >= start_ts]
                elif "date" in df.columns:
                    df = df[pd.to_datetime(df["date"], utc=True) >= start_dt]
            except Exception as e:
                return key, None, None, f"无效的起始日期格式 '{start_date}': {e}", None
        elif train_period_days:
            ts = _train_days_cutoff(train_period_days)
            if "timestamp" in df.columns:
                df = df[df["timestamp"] >= ts]

        # 排除最近 N 天数据（留给回测，防止数据泄露）
        train_cutoff_date = None
        if holdout_days > 0:
            holdout_ts = _holdout_cutoff(holdout_days)
            train_cutoff_date = datetime.fromtimestamp(holdout_ts / 1000, tz=timezone.utc).isoformat()
            if "timestamp" in df.columns:
                df = df[df["timestamp"] < holdout_ts]

        fc = feature_cols_allowlist if (feature_cols_allowlist and len(feature_cols_allowlist) > 0) else None
        X, y = prepare_train_data(
            df,
            label_threshold_pct=label_threshold_pct,
            primary_only=primary_only,
            lookahead=lookahead,
            feature_cols=fc,
            add_multi_timeframe=add_multi_timeframe,
        )

        if len(X) < 100:
            return key, None, None, f"样本不足 ({len(X)})", None

        # 训练日志：UP/DOWN 比例、scale_pos_weight、is_unbalance
        r = float(y.mean())
        up, down = int(y.sum()), len(y) - int(y.sum())
        if use_scale_pos_weight and r and (r < 0.45 or r > 0.55):
            spw = f"{(1 - r) / r:.3f}"
        elif use_scale_pos_weight:
            spw = "未设置(UP%∈[45,55])"
        else:
            spw = "关(--no-scale-pos-weight)"
        print(f"  [{key}] 训练集: UP={up}, DOWN={down}, UP%={100*r:.1f}%; scale_pos_weight={spw}; is_unbalance=无", flush=True)

        if use_cv and len(X) >= 500:
            model, metrics, calibrator = train_with_cv(
                X, y, params=params, early_stopping_rounds=early, n_splits=5, verbose=False,
                purge_embargo=purge_embargo, use_scale_pos_weight=use_scale_pos_weight,
                calibration_method=calibration_method,
            )
        else:
            model, metrics, calibrator = train_one(
                X, y, params=params, early_stopping_rounds=early,
                purge_embargo=purge_embargo, use_scale_pos_weight=use_scale_pos_weight,
                calibration_method=calibration_method,
            )

        # 按重要性剔尾：若设置了 drop_bottom_pct，计算本模型保留的特征名，供主进程合并写入
        allowlist = None
        if feature_drop_bottom_pct and feature_drop_bottom_pct > 0:
            keep = 1.0 - float(feature_drop_bottom_pct)
            allowlist = compute_feature_allowlist(model, list(X.columns), keep)

        # 保存模型
        out_dir = models_dir / f"lightgbm_{key}_{timestamp}"
        out_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, out_dir / "model.joblib")
        if calibrator is not None:
            joblib.dump(calibrator, out_dir / "calibrator.joblib")
        meta = {
            "feature_names": list(X.columns),
            "metrics": metrics,
            "train_period_days": train_period_days,
            "holdout_days": holdout_days,
            "train_cutoff_date": train_cutoff_date,
            "label_threshold_pct": label_threshold_pct,
            "lookahead": lookahead,
            "primary_only": primary_only,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "timeframe": tf,
            "symbol": s,
            "sample_count": len(X),
            "y_up": up,
            "y_down": down,
            "y_up_pct": round(100 * r, 2),
            "scale_pos_weight": float((1 - r) / r) if (use_scale_pos_weight and r and (r < 0.45 or r > 0.55)) else None,
        }
        if calibrator is not None:
            meta["calibration"] = calibration_method
        (out_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

        return key, out_dir, metrics, None, allowlist

    except Exception as e:
        return key, None, None, str(e), None


def train_per_symbol_timeframe(
    train_period_days: Optional[int] = None,
    holdout_days: int = 90,
    use_cv: bool = False,
    label_threshold_pct: Optional[float] = None,
    primary_only: bool = False,
    lookahead: int = 1,
    n_jobs: int = -1,
    feature_drop_bottom_pct: float = 0,
    use_feature_allowlist: bool = False,
    purge_embargo: int = 10,
    use_scale_pos_weight: bool = True,
    add_multi_timeframe: bool = False,
    calibration_method: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    models_dir: Optional[Path] = None,
    params_file: Optional[str] = None,
    num_threads: Optional[int] = None,
) -> Dict[str, Path]:
    """
    按 (symbol, timeframe) 分别训练独立模型。
    总共训练 4 symbols × 3 timeframes = 12 个模型。
    返回 { "symbol_timeframe": model_dir_path }
    
    参数:
        holdout_days: 排除最近多少天数据（留给回测，防止数据泄露），默认 90 天
        symbols: 只训练这些交易对，如 ["BTC/USDT","ETH/USDT","XRP/USDT"]；None 表示全部。与 ALLOWED_MARKETS 对应时可只训 BTC,ETH,XRP
        primary_only: 如果为 True，只使用主要特征（KDJ、Range）
        lookahead: 预测未来多少根 K 线（默认 1，建议 3-5 更稳定）
        n_jobs: 并行数，-1 表示使用全部 CPU 核心
        feature_drop_bottom_pct: 剔除 importance 最低的占比，如 0.3 表示去掉后 30%，保留 top 70% 写入 config/feature_allowlist.txt（供下次 --use-feature-allowlist 使用）
        use_feature_allowlist: 若 True，只使用 config/feature_allowlist.txt 中的特征训练（需先跑一轮带 --feature-drop-bottom-pct 生成该文件）
        purge_embargo: train 末/val 头各删几行防泄漏，默认 10，0=关闭
        use_scale_pos_weight: 类别不平衡时自动设 scale_pos_weight，默认 True
        add_multi_timeframe: 15m 时是否加入 1h/4h 高层特征，默认 False
        calibration_method: "isotonic"|"sigmoid" 时拟合概率校准并保存
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import time

    freqai = _load_freqai_config()
    train_period_days = train_period_days or freqai.get("train_period_days") or 2555
    early = freqai.get("model_training_parameters", {}).get("early_stopping_rounds", 50)
    params = _load_lightgbm_params(params_file=params_file, num_threads=num_threads)

    feature_cols_allowlist = None
    if use_feature_allowlist:
        feature_cols_allowlist = load_feature_allowlist()
        if not feature_cols_allowlist:
            print("  [特征] --use-feature-allowlist 已开启，但 config/feature_allowlist.txt 不存在或为空，本轮使用全部特征。")
            print("  [特征] 请先跑一轮并加 --feature-drop-bottom-pct 0.3 生成 feature_allowlist.txt")
        else:
            print(f"  [特征] 使用 feature_allowlist.txt，共 {len(feature_cols_allowlist)} 个特征")

    # 标准化 symbols：BTC -> BTC/USDT，且只保留 SYMBOLS 中有的
    _sym_raw = symbols or SYMBOLS
    _sym_list = []
    for x in _sym_raw:
        s = f"{x.upper()}/USDT" if "/" not in str(x) else x
        if s in SYMBOLS:
            _sym_list.append(s)
    _sym_list = _sym_list if _sym_list else list(SYMBOLS)
    if symbols:
        print(f"  [交易对] 仅训练: {', '.join(_sym_list)}")

    output_dir = Path(models_dir) if models_dir else MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    
    if models_dir:
        print(f"  [模型目录] 输出到: {output_dir}")

    # 准备所有任务
    tasks = []
    for s in _sym_list:
        for tf in TIMEFRAMES:
            tasks.append((s, tf, train_period_days, holdout_days, use_cv, label_threshold_pct,
                         primary_only, lookahead, params, early, timestamp, output_dir,
                         feature_drop_bottom_pct, feature_cols_allowlist,
                         purge_embargo, use_scale_pos_weight, add_multi_timeframe, calibration_method, start_date, params_file, num_threads))

    total = len(tasks)
    
    # 确定并行数（如果设置了 num_threads，建议减少并行进程数以避免 CPU 过载）
    if n_jobs == -1:
        if num_threads is not None:
            # 如果限制了线程数，也限制并行进程数，避免总 CPU 使用率过高
            # 例如：num_threads=4，8核，最多并行 2 个进程（2*4=8核）
            cpu_count = os.cpu_count() or 8
            max_parallel = max(1, cpu_count // num_threads)
            n_jobs = min(max_parallel, total)
            print(f"  💡 已设置 --num-threads {num_threads}，自动限制并行进程数为 {n_jobs}（避免 CPU 过载）")
            print(f"  💡 总 CPU 使用: {n_jobs} 进程 × {num_threads} 线程 = {n_jobs * num_threads} 线程（共 {cpu_count} 核）")
        else:
            n_jobs = os.cpu_count() or 4
    n_jobs = min(n_jobs, total)
    
    print(f"\n{'='*60}")
    print(f"🚀 开始训练 {total} 个模型")
    print(f"   并行数: {n_jobs} 进程")
    if num_threads:
        print(f"   CPU 线程数: {num_threads} (每个模型，用于控制温度)")
        print(f"   ⚠️  如果 CPU 使用率仍然 100%，请手动设置 --n-jobs 2 来减少并行进程数")
    if start_date:
        print(f"   训练数据起始日期: {start_date} (UTC)")
    else:
        print(f"   训练数据: 最近 {train_period_days} 天（约 {train_period_days//365} 年）")
    print(f"   排除最近: {holdout_days} 天（留给回测，防止数据泄露）")
    print(f"   预测周期: 未来 {lookahead} 根 K 线方向")
    print(f"   交叉验证: {'是' if use_cv else '否'}")
    print(f"{'='*60}\n")
    
    results = {}
    all_metrics = {}
    errors = []
    allowlists: List[List[str]] = []

    start_time = time.time()
    completed = 0

    # 并行训练
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {executor.submit(_train_single_model, task): task for task in tasks}

        for future in as_completed(futures):
            completed += 1
            key, out_dir, metrics, error, allowlist = future.result()

            elapsed = time.time() - start_time
            avg_time = elapsed / completed
            remaining = avg_time * (total - completed)

            # 进度显示
            pct = completed / total * 100
            bar_len = 30
            filled = int(bar_len * completed / total)
            bar = '█' * filled + '░' * (bar_len - filled)

            if error:
                errors.append(f"{key}: {error}")
                status = f"❌ {key}: {error}"
            else:
                results[key] = out_dir
                all_metrics[key] = metrics
                if allowlist:
                    allowlists.append(allowlist)
                acc = metrics.get("mean_accuracy", metrics.get("accuracy", 0))
                auc = metrics.get("mean_auc", metrics.get("auc", 0))
                status = f"✅ {key}: acc={acc:.4f}, auc={auc:.4f}"

            # 打印进度
            print(f"\r[{bar}] {pct:5.1f}% | {completed}/{total} | "
                  f"⏱️ {elapsed:.0f}s 已用 | 剩余 ~{remaining:.0f}s | {status}")

    total_time = time.time() - start_time

    # 若本轮有剔尾，合并各模型 top 特征（取并集）写入 config/feature_allowlist.txt
    if allowlists:
        merged = sorted(set(f for a in allowlists for f in a))
        FEATURE_ALLOWLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        FEATURE_ALLOWLIST_PATH.write_text("\n".join(merged) + "\n", encoding="utf-8")
        print(f"\n  [特征] 已写入 config/feature_allowlist.txt，共 {len(merged)} 个（下一轮可加 --use-feature-allowlist 使用）")

    # 打印汇总
    print(f"\n{'='*60}")
    print(f"✨ 训练完成！")
    print(f"   成功: {len(results)}/{total} 个模型")
    print(f"   总耗时: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
    print(f"{'='*60}")
    
    if all_metrics:
        print(f"\n📊 模型指标汇总（按 AUC 排序）:")
        sorted_metrics = sorted(
            all_metrics.items(), 
            key=lambda x: x[1].get("mean_auc", x[1].get("auc", 0)), 
            reverse=True
        )
        print(f"{'排名':<4} {'模型':<20} {'Accuracy':<12} {'AUC':<12}")
        print("-" * 50)
        for i, (k, m) in enumerate(sorted_metrics, 1):
            acc = m.get("mean_accuracy", m.get("accuracy", 0))
            auc = m.get("mean_auc", m.get("auc", 0))
            print(f"{i:<4} {k:<20} {acc:.4f}       {auc:.4f}")
    
    if errors:
        print(f"\n⚠️ 错误 ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
    
    return results


if __name__ == "__main__":
    import argparse
    import os
    
    # 先解析 num_threads 参数（在导入其他库之前设置环境变量）
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--num-threads", type=int, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    
    # 如果设置了 num_threads，立即设置环境变量（必须在导入 LightGBM 之前）
    if pre_args.num_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(pre_args.num_threads)
        os.environ["MKL_NUM_THREADS"] = str(pre_args.num_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(pre_args.num_threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(pre_args.num_threads)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(pre_args.num_threads)
        print(f"🔧 已设置 CPU 线程数限制: {pre_args.num_threads} (环境变量已更新，影响所有库)")

    ap = argparse.ArgumentParser()
    ap.add_argument("--initial-training", action="store_true", help="首次训练（全局单模型）")
    ap.add_argument("--per-timeframe", action="store_true", help="按 timeframe 分别训练（推荐）")
    ap.add_argument("--per-symbol-timeframe", action="store_true", help="按 symbol+timeframe 分别训练（12 个模型）")
    ap.add_argument("--symbols", nargs="*", default=None, help="只训练这些交易对，如 --symbols BTC ETH XRP；与 ALLOWED_MARKETS 一致时可只训 3 个")
    ap.add_argument("--timeframes", nargs="*", default=None)
    ap.add_argument("--train-period-days", type=int, default=None, help="训练数据天数，默认使用全部数据（2019年至今）")
    ap.add_argument("--holdout-days", type=int, default=90, help="排除最近 N 天数据，留给回测（防止数据泄露），默认 90")
    ap.add_argument("--lookahead", type=int, default=3, help="预测未来多少根K线方向，默认 3（更稳定），1=只看下一根")
    ap.add_argument("--use-cv", action="store_true", help="使用时间序列交叉验证")
    ap.add_argument("--label-threshold", type=float, default=None, help="涨跌幅阈值（百分比，如 0.1）")
    ap.add_argument("--primary-only", action="store_true", help="只使用主要特征（KDJ、Range），不包含其他辅助特征")
    ap.add_argument("--n-jobs", type=int, default=-1, help="并行进程数，-1 表示使用全部 CPU 核心")
    ap.add_argument("--num-threads", type=int, default=None, metavar="N", 
                    help="LightGBM CPU 线程数，用于控制温度（MacBook M4 建议 2-4）。同时会设置环境变量 OMP_NUM_THREADS 等，确保生效。建议配合 --n-jobs 2 使用（减少并行进程数）")
    ap.add_argument("--optuna", action="store_true", help="使用 Optuna 自动搜索最优超参数（耗时但效果好）")
    ap.add_argument("--optuna-trials", type=int, default=100, help="Optuna 搜索次数，默认 100")
    ap.add_argument("--optuna-timeout", type=int, default=None, metavar="SEC", help="Optuna 超时（秒），默认不限制。如 3600=1小时")
    ap.add_argument("--optuna-output", type=str, default=None, metavar="FILE", help="Optuna 优化结果保存文件名（相对于 config/），默认 lightgbm_params_optuna.json")
    ap.add_argument("--params-file", type=str, default=None, metavar="FILE", help="训练时使用的参数文件（相对于 config/ 或绝对路径），默认按优先级：optuna > default")
    ap.add_argument("--feature-drop-bottom-pct", type=float, default=0, metavar="0.3",
                    help="剔除 importance 最低的占比，如 0.3=去掉后30%%，保留 top 70%% 写入 config/feature_allowlist.txt")
    ap.add_argument("--use-feature-allowlist", action="store_true",
                    help="只使用 config/feature_allowlist.txt 中的特征（需先跑一轮 --feature-drop-bottom-pct 0.3 生成）")
    ap.add_argument("--purge-embargo", type=int, default=10, metavar="N",
                    help="train 末/val 头各删 N 行防泄漏，默认 10，0=关闭")
    ap.add_argument("--no-scale-pos-weight", action="store_true",
                    help="关闭类别不平衡时的 scale_pos_weight")
    ap.add_argument("--add-multi-timeframe", action="store_true",
                    help="15m 时加入 1h/4h 高层特征（需有 1h、4h parquet）")
    ap.add_argument("--calibration", type=str, default=None, choices=["isotonic", "sigmoid"], metavar="METHOD",
                    help="概率校准：isotonic（非参数）或 sigmoid（Platt），在验证集上拟合后保存 calibrator.joblib")
    ap.add_argument("--start-date", type=str, default=None, metavar="YYYY-MM-DD",
                    help="训练数据起始日期（ISO 格式，如 2024-06-01），优先级高于 --train-period-days")
    ap.add_argument("--models-dir", type=str, default=None, metavar="PATH",
                    help="模型输出目录，默认 data/models；可指定 data/models_B 等多套并行")
    args = ap.parse_args()

    models_dir_path = Path(args.models_dir).resolve() if args.models_dir else None

    # 如果启用 Optuna，先搜索最优参数
    if args.optuna:
        print("📊 加载数据用于超参数搜索...")
        # 用 BTC 4H 数据作为代表进行搜索
        df = load_training_data(
            symbols=["BTC/USDT"],
            timeframes=["4h"],
            train_period_days=args.train_period_days or 2555,
            holdout_days=args.holdout_days,
            start_date=args.start_date,
        )
        print(f"   数据加载完成，共 {len(df)} 行")
        print("   正在计算特征（新特征集 v4，约 150+ 特征）...")
        X, y = prepare_train_data(df, label_threshold_pct=args.label_threshold, 
                                  primary_only=args.primary_only, lookahead=args.lookahead)
        print(f"   特征计算完成，共 {len(X.columns)} 个特征，{len(X)} 个样本")
        print("   开始 Optuna 超参数优化...")
        
        best_params = optuna_tune_lightgbm(X, y, n_trials=args.optuna_trials, timeout=args.optuna_timeout, num_threads=args.num_threads)
        
        # 保存最优参数（支持指定输出文件）
        if args.optuna_output:
            params_path = CONFIG_DIR / args.optuna_output if not Path(args.optuna_output).is_absolute() else Path(args.optuna_output)
        else:
            params_path = CONFIG_DIR / "lightgbm_params_optuna.json"
        params_path.write_text(json.dumps(best_params, indent=2), encoding="utf-8")
        print(f"💾 最优参数已保存到: {params_path}")
        if not args.optuna_output:
            print("提示：下次训练时会自动使用 lightgbm_params_optuna.json")
        else:
            print(f"提示：训练时使用 --params-file {args.optuna_output} 来使用这些参数")
    
    if args.per_timeframe:
        train_per_timeframe(
            symbols=args.symbols,
            train_period_days=args.train_period_days,
            use_cv=args.use_cv,
            label_threshold_pct=args.label_threshold,
            primary_only=args.primary_only,
            calibration_method=args.calibration,
            start_date=args.start_date,
            models_dir=models_dir_path,
            params_file=args.params_file,
            num_threads=args.num_threads,
        )
    elif args.per_symbol_timeframe:
        train_per_symbol_timeframe(
            train_period_days=args.train_period_days,
            holdout_days=args.holdout_days,
            use_cv=args.use_cv,
            label_threshold_pct=args.label_threshold,
            primary_only=args.primary_only,
            lookahead=args.lookahead,
            n_jobs=args.n_jobs,
            feature_drop_bottom_pct=args.feature_drop_bottom_pct,
            use_feature_allowlist=args.use_feature_allowlist,
            purge_embargo=args.purge_embargo,
            use_scale_pos_weight=not args.no_scale_pos_weight,
            add_multi_timeframe=args.add_multi_timeframe,
            calibration_method=args.calibration,
            symbols=args.symbols if args.symbols else None,
            start_date=args.start_date,
            models_dir=models_dir_path,
            params_file=args.params_file,
            num_threads=args.num_threads,
        )
    elif not args.optuna:  # 如果只是运行 optuna，不训练
        run_initial_training(
            symbols=args.symbols,
            timeframes=args.timeframes,
            train_period_days=args.train_period_days,
            use_cv=args.use_cv,
            label_threshold_pct=args.label_threshold,
            primary_only=args.primary_only,
            calibration_method=args.calibration,
            num_threads=args.num_threads,
            start_date=args.start_date,
            models_dir=models_dir_path,
            params_file=args.params_file,
        )
