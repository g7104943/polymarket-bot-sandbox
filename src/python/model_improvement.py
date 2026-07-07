"""
月度模型改进：
- 方案 A（默认）：滚动 90 天窗口，每月重训，始终匹配当前市场
- 方案 B（可选）：分层采样 2 年数据，保留罕见事件，作为 Ensemble 长期模型

两者在 Ensemble（步骤 8）中互补：A 负责新鲜度，B 负责覆盖面。
"""

import json
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from .model_trainer import (
    run_initial_training,
    load_training_data,
    prepare_train_data,
    train_one,
    _load_lightgbm_params,
    _load_freqai_config,
)
from .predictor import load_predictor, predict_one
from .data.rolling_window import get_rolling_window_data, get_rolling_window_info
from .data.stratified_sampler import stratified_sampling, get_sampling_stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "data" / "models"


def _latest_model_dir(suffix: str = "") -> Optional[Path]:
    """查找最新模型目录。suffix 可过滤如 '_monthly'、'_stratified' 等。"""
    if not MODELS_DIR.exists():
        return None
    dirs = [
        d for d in MODELS_DIR.iterdir()
        if d.is_dir() and (d / "model.joblib").exists() and (not suffix or suffix in d.name)
    ]
    return max(dirs, key=lambda d: d.stat().st_mtime) if dirs else None


def evaluate_on_validation(model, feature_names: list, X_val, y_val) -> float:
    """返回验证集准确率"""
    cols = [c for c in feature_names if c in X_val.columns]
    if not cols:
        return 0.0
    pred = (model.predict(X_val[cols]) >= 0.5).astype(int)
    return (pred == y_val).mean()


def monthly_update(
    symbols: Optional[list] = None,
    timeframes: Optional[list] = None,
    force_replace: bool = False,
    train_days: int = 90,
    val_days: int = 30,
    calibration_method: Optional[str] = None,
) -> Path:
    """
    方案 A：滚动窗口月度更新。

    使用最近 train_days（默认 90）天训练 + val_days（默认 30）天验证，
    与当前模型在验证集上比较；若新模型更优则保存。

    参数:
        train_days: 训练窗口天数（默认 90，原来是 30）
        val_days: 验证窗口天数（默认 30）
        calibration_method: 校准方法（isotonic/sigmoid/temperature_scaling）
    """
    freqai = _load_freqai_config()

    # 加载全量数据（不限天数），再用滚动窗口切分
    df_full = load_training_data(
        symbols=symbols,
        timeframes=timeframes,
        train_period_days=None,  # 加载全部可用数据
    )

    # 滚动窗口切分
    train_df, val_df = get_rolling_window_data(
        df_full, end_date=None, train_days=train_days, val_days=val_days
    )
    window_info = get_rolling_window_info(train_days=train_days, val_days=val_days)
    print(f"[monthly] 方案 A 滚动窗口: 训练 {window_info['train_start']}~{window_info['train_end']} "
          f"({train_days}d), 验证 {window_info['val_start']}~{window_info['val_end']} ({val_days}d)")

    # 分别准备特征+标签
    from .feature_engineering import prepare_train_data as _prep
    X_tr, y_tr = _prep(train_df)
    X_va, y_va = _prep(val_df)

    if len(X_tr) < 100:
        raise ValueError(f"训练集样本不足 100（{len(X_tr)}），窗口可能太窄或数据缺失")
    if len(X_va) < 50:
        raise ValueError(f"验证集样本不足 50（{len(X_va)}），窗口可能太窄或数据缺失")

    print(f"[monthly] 训练集: {len(X_tr)} 样本, 验证集: {len(X_va)} 样本, 特征: {len(X_tr.columns)}")

    # 训练新模型
    params = _load_lightgbm_params()
    early = freqai.get("model_training_parameters", {}).get("early_stopping_rounds", 20)
    new_model, metrics, calibrator = train_one(
        X_tr, y_tr, params=params, early_stopping_rounds=early,
        val_ratio=0.2, calibration_method=calibration_method,
    )
    feature_names = list(X_tr.columns)
    new_acc = evaluate_on_validation(new_model, feature_names, X_va, y_va)

    # 旧模型对比
    old_dir = _latest_model_dir(suffix="_monthly")
    old_acc = 0.0
    if old_dir and not force_replace:
        try:
            old_model, old_feats, _ = load_predictor(old_dir)
            old_acc = evaluate_on_validation(old_model, old_feats, X_va, y_va)
        except Exception:
            old_acc = 0.0

    # 决策
    if force_replace or new_acc >= old_acc:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        name = f"lightgbm_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_monthly"
        out_dir = MODELS_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        import joblib

        joblib.dump(new_model, out_dir / "model.joblib")
        if calibrator is not None:
            joblib.dump(calibrator, out_dir / "calibrator.joblib")
        meta = {
            "feature_names": feature_names,
            "metrics": metrics,
            "val_accuracy": float(new_acc),
            "data_strategy": "rolling_window",
            "train_days": train_days,
            "val_days": val_days,
            "window_info": window_info,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "replaced_old_val_acc": float(old_acc),
        }
        if calibrator is not None:
            meta["calibration"] = calibration_method
        (out_dir / "metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[monthly] 新模型更好 (val_acc={new_acc:.4f} >= {old_acc:.4f})，已保存: {out_dir}")
        return out_dir
    else:
        print(f"[monthly] 保留旧模型 (old_val_acc={old_acc:.4f} > new_val_acc={new_acc:.4f})")
        return old_dir or MODELS_DIR


def train_stratified_long_term(
    symbols: Optional[list] = None,
    timeframes: Optional[list] = None,
    force_replace: bool = False,
    cutoff_days: Optional[list] = None,
    sample_rates: Optional[list] = None,
    calibration_method: Optional[str] = None,
) -> Path:
    """
    方案 B：分层采样 2 年长期模型。

    用分层采样保留罕见事件（黑天鹅、监管冲击），
    训练一个长期模型作为 Ensemble 的成员。

    默认分层:
        0-90d: 100%, 90-180d: 50%, 180-365d: 25%, >365d: 14%
        ~70,000 根 K 线 → ~15,000 根

    参数:
        cutoff_days: 分层边界（默认 [90, 180, 365]）
        sample_rates: 每层采样率（默认 [1.0, 0.5, 0.25, 0.14]）
        calibration_method: 校准方法
    """
    freqai = _load_freqai_config()

    # 加载全量数据
    df_full = load_training_data(
        symbols=symbols,
        timeframes=timeframes,
        train_period_days=None,  # 加载全部
    )

    # 分层采样
    df_sampled = stratified_sampling(
        df_full,
        cutoff_days=cutoff_days,
        sample_rates=sample_rates,
    )
    stats = get_sampling_stats(df_full, df_sampled, cutoff_days=cutoff_days)
    print(f"[stratified] 方案 B 分层采样: {stats['total_original']} → {stats['total_sampled']} "
          f"行 (减少 {stats['reduction_pct']}%)")
    for layer in stats["layers"]:
        print(f"  {layer['range']}: {layer['original']} → {layer['sampled']} (采样率 {layer['rate']})")

    # 准备特征+标签，80/20 时间序列划分
    X, y = prepare_train_data(df_sampled)
    if len(X) < 200:
        raise ValueError(f"分层采样后样本不足 200（{len(X)}），历史数据可能太少")

    n = len(X)
    split = int(n * 0.8)
    X_tr, X_va = X.iloc[:split], X.iloc[split:]
    y_tr, y_va = y.iloc[:split], y.iloc[split:]

    print(f"[stratified] 训练集: {len(X_tr)} 样本, 验证集: {len(X_va)} 样本, 特征: {len(X_tr.columns)}")

    # 训练
    params = _load_lightgbm_params()
    early = freqai.get("model_training_parameters", {}).get("early_stopping_rounds", 50)
    new_model, metrics, calibrator = train_one(
        X_tr, y_tr, params=params, early_stopping_rounds=early,
        val_ratio=0.2, calibration_method=calibration_method,
    )
    feature_names = list(X_tr.columns)
    new_acc = evaluate_on_validation(new_model, feature_names, X_va, y_va)

    # 旧模型对比
    old_dir = _latest_model_dir(suffix="_stratified")
    old_acc = 0.0
    if old_dir and not force_replace:
        try:
            old_model, old_feats, _ = load_predictor(old_dir)
            old_acc = evaluate_on_validation(old_model, old_feats, X_va, y_va)
        except Exception:
            old_acc = 0.0

    # 决策
    if force_replace or new_acc >= old_acc:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        name = f"lightgbm_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_stratified"
        out_dir = MODELS_DIR / name
        out_dir.mkdir(parents=True, exist_ok=True)
        import joblib

        joblib.dump(new_model, out_dir / "model.joblib")
        if calibrator is not None:
            joblib.dump(calibrator, out_dir / "calibrator.joblib")
        meta = {
            "feature_names": feature_names,
            "metrics": metrics,
            "val_accuracy": float(new_acc),
            "data_strategy": "stratified_sampling",
            "sampling_stats": stats,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "replaced_old_val_acc": float(old_acc),
        }
        if calibrator is not None:
            meta["calibration"] = calibration_method
        (out_dir / "metadata.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[stratified] 新模型更好 (val_acc={new_acc:.4f} >= {old_acc:.4f})，已保存: {out_dir}")
        return out_dir
    else:
        print(f"[stratified] 保留旧模型 (old_val_acc={old_acc:.4f} > new_val_acc={new_acc:.4f})")
        return old_dir or MODELS_DIR


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="月度模型改进（方案 A 滚动窗口 / 方案 B 分层采样）")
    ap.add_argument("--monthly-update", action="store_true", help="方案 A：滚动窗口月度更新")
    ap.add_argument("--stratified", action="store_true", help="方案 B：分层采样长期模型")
    ap.add_argument("--force-replace", action="store_true", help="强制用新模型替换")
    ap.add_argument("--train-days", type=int, default=90, help="方案 A 训练窗口天数（默认 90）")
    ap.add_argument("--val-days", type=int, default=30, help="方案 A 验证窗口天数（默认 30）")
    ap.add_argument("--calibration", type=str, default=None,
                    choices=["isotonic", "sigmoid"], help="校准方法")
    args = ap.parse_args()

    if args.stratified:
        train_stratified_long_term(
            force_replace=args.force_replace,
            calibration_method=args.calibration,
        )
    else:
        monthly_update(
            force_replace=args.force_replace,
            train_days=args.train_days,
            val_days=args.val_days,
            calibration_method=args.calibration,
        )
