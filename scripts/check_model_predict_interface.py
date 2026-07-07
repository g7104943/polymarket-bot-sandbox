#!/usr/bin/env python3
"""
检查已保存模型在 predict() 与 predict_proba() 下的返回值，确认训练/推理时应用的是哪套接口。

用法:
  python3 scripts/check_model_predict_interface.py
  python3 scripts/check_model_predict_interface.py --models-dir data/models_C --symbol XRP/USDT --timeframe 15m

会加载指定模型、用一条样本特征调用 model.predict() 和 model.predict_proba()，并打印：
  - 返回值类型、形状、以及若干样本值
便于判断：若 predict() 只返回 0/1，则置信度必须用 predict_proba()。
"""

import argparse
import sys
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.python.predictor import load_predictor, _find_symbol_timeframe_model
from src.python.data_fetcher import load_ohlcv
from src.python.feature_engineering import build_features, add_multi_timeframe_features


def main():
    ap = argparse.ArgumentParser(description="检查模型 predict / predict_proba 返回值")
    ap.add_argument("--models-dir", type=str, default="data/models_C", help="模型根目录，如 data/models 或 data/models_C")
    ap.add_argument("--symbol", type=str, default="XRP/USDT", help="交易对，如 XRP/USDT")
    ap.add_argument("--timeframe", type=str, default="15m", help="时间周期，如 15m")
    args = ap.parse_args()

    root = PROJECT_ROOT / args.models_dir.strip()
    if not root.is_dir():
        print(f"错误: 目录不存在: {root}")
        sys.exit(1)

    model_dir = _find_symbol_timeframe_model(args.symbol, args.timeframe, models_root=root)
    if not model_dir:
        print(f"错误: 在 {root} 下未找到 {args.symbol} {args.timeframe} 的模型目录")
        sys.exit(1)

    print(f"加载模型: {model_dir}")
    model, feature_names, meta = load_predictor(model_dir)
    print(f"模型类型: {type(model).__module__}.{type(model).__name__}")
    print(f"特征数: {len(feature_names)}")

    # 用本地数据构造一条样本（与 predictor 一致）
    df = load_ohlcv(args.symbol, args.timeframe)
    if df.empty or len(df) < 100:
        print("警告: 本地 K 线不足 100 根，尝试用少量数据；若仍缺特征会报错")
    if df.empty or len(df) < 50:
        print("错误: 数据不足，无法构造特征")
        sys.exit(1)

    if any((f or "").startswith("mtf_") for f in (feature_names or [])):
        df = add_multi_timeframe_features(df, args.symbol)
    df = build_features(df)
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        print(f"错误: 缺少特征: {missing[:10]}{'...' if len(missing) > 10 else ''}")
        sys.exit(1)

    filled = df[feature_names].ffill().bfill()
    X = filled.iloc[[-1]]
    if X.isna().any().any():
        print("错误: 最后一行特征含 NaN")
        sys.exit(1)

    print("\n--- model.predict(X) ---")
    try:
        pred = model.predict(X)
        print(f"  类型: {type(pred)}")
        if hasattr(pred, "shape"):
            print(f"  形状: {pred.shape}")
        print(f"  取值（前 10）: {pred.ravel()[:10].tolist()}")
        uniq = sorted(set(pred.ravel().tolist()))
        print(f"  唯一值: {uniq}")
    except Exception as e:
        print(f"  异常: {e}")

    print("\n--- model.predict_proba(X) ---")
    try:
        proba = model.predict_proba(X)
        print(f"  类型: {type(proba)}")
        if hasattr(proba, "shape"):
            print(f"  形状: {proba.shape}")
        print(f"  首行 [P(DOWN), P(UP)]: {proba[0].tolist()}")
        if proba.shape[0] > 1:
            print(f"  前 3 行:\n{proba[:3]}")
    except Exception as e:
        print(f"  异常: {e}")

    print("\n--- 结论 ---")
    if hasattr(model, "predict_proba"):
        print("  模型为 sklearn 风格（如 LGBMClassifier）：predict()=类别 0/1，置信度用 predict_proba(X)[:, 1]。")
    else:
        print("  模型为 Booster/原生接口：无 predict_proba，predict() 即概率/得分；当前 predictor 的 else 分支会使用 predict() 作为 prob。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
