#!/usr/bin/env python3
"""
Walk-Forward 验证 — 替代单次 train/test 划分，检测 regime-specific 过拟合

核心思路:
  把时间序列切成多个连续窗口，每个窗口独立训练+测试：
    Window 1: Train [M1-M4]  Test [M5]
    Window 2: Train [M2-M5]  Test [M6]
    Window 3: Train [M3-M6]  Test [M7]
    ...
  最终分数 = mean(all windows)

  这揭示了模型在不同市场环境下的真实表现，而非单一 15 天的运气。

用法:
  python scripts/walk_forward_validation.py
  python scripts/walk_forward_validation.py --regularized
  python scripts/walk_forward_validation.py --sim-noise --regularized
  python scripts/walk_forward_validation.py --sim-noise --btc-weight-boost 1.5
  python scripts/walk_forward_validation.py --train-days 90 --test-days 15 --step-days 15

输出:
  每个窗口的 AUC / 方向准确率 / per-asset AUC
  汇总统计（mean, std, min, max）
  诊断: 哪些窗口/资产最弱
"""
from __future__ import annotations

# ★ 必须在所有 import 之前设置 — 防止 PyTorch/LightGBM/PyArrow 线程池死锁
import os as _os
for _env_key in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_MAX_THREADS",
    "ARROW_NUM_THREADS",      # PyArrow 读取线程
    "PYARROW_NUM_THREADS",    # PyArrow 备用变量名
):
    _os.environ[_env_key] = "1"

import argparse
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "gru_regime_v1"))

import lightgbm as lgb
from sklearn.metrics import roc_auc_score, brier_score_loss

# ★ 延迟导入 run_grid（包含 import torch）— 只在 __name__=="__main__" 时加载
# 先从 data_prep 导入不含 torch 的工具
from experiments.sentiment_grid_search.data_prep import (
    _ensure_datetime_col,
)

# ─── 从 train_production_model 内联的常量/函数（避免间接导入 torch）────
EXP7_TUNED_PARAMS = {
    "objective": "binary", "metric": "auc", "n_jobs": 4, "verbosity": -1,
    "num_leaves": 212, "max_depth": 9, "learning_rate": 0.00606389,
    "min_child_samples": 25, "feature_fraction": 0.56639651,
    "bagging_fraction": 0.76211669, "bagging_freq": 2,
    "lambda_l1": 0.00087003, "lambda_l2": 3.64924272,
    "min_split_gain": 0.20273105, "path_smooth": 8.31359266,
}
REGULARIZED_PARAMS = {
    "objective": "binary", "metric": "auc", "n_jobs": 4, "verbosity": -1,
    "num_leaves": 63, "max_depth": 6, "learning_rate": 0.02,
    "min_child_samples": 50, "feature_fraction": 0.4,
    "bagging_fraction": 0.6, "bagging_freq": 2,
    "lambda_l1": 0.1, "lambda_l2": 10.0,
    "min_split_gain": 0.5, "path_smooth": 10.0,
}
EXP7_GROUPS = ["fgi_daily", "cfgi", "news", "ob", "funding", "oi", "lsratio", "polymarket_prob"]
EXP8_GROUPS = [
    "fgi_daily", "cfgi", "news", "ob", "funding", "oi", "lsratio",
    "polymarket_prob", "polymarket_prob_target",
]
# Exp13/14: Exp8 基础 + TradingView 宏观特征 (去掉 cfgi/news)
EXP13_GROUPS = [
    "fgi_daily", "ob", "funding", "oi", "lsratio",
    "polymarket_prob", "polymarket_prob_target",
    "tv_macro",
]
# Exp15: 365d 训练 (基于 Exp8，去掉 cfgi/news)
EXP15_GROUPS = [
    "fgi_daily", "ob", "funding", "oi", "lsratio",
    "polymarket_prob", "polymarket_prob_target",
]
# Exp16: 365d 训练 + TV宏观 (与 Exp13 相同特征组)
EXP16_GROUPS = EXP13_GROUPS
SIM_NOISE_STD = 0.0013
SEED = 42

def apply_simulated_candle_noise(tech_df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """在训练数据的 close/high/low/volume 上添加随机噪声，模拟 T-120s 不完整K线。"""
    rng = np.random.RandomState(seed)
    df = tech_df.copy()
    n = len(df)
    log_noise = rng.normal(0, SIM_NOISE_STD, n)
    df["close"] = df["close"] * np.exp(log_noise)
    high_noise = rng.normal(0, SIM_NOISE_STD * 0.5, n)
    df["high"] = np.maximum(df["high"] * np.exp(high_noise), df["close"])
    low_noise = rng.normal(0, SIM_NOISE_STD * 0.5, n)
    df["low"] = np.minimum(df["low"] * np.exp(low_noise), df["close"])
    vol_factor = rng.uniform(0.83, 0.93, n)
    df["volume"] = df["volume"] * vol_factor
    return df

warnings.filterwarnings("ignore", category=UserWarning)

# ─── 从 run_grid 内联的常量（避免 import torch）─────────────────────
ASSETS = ["BTC_USDT", "ETH_USDT", "XRP_USDT"]
WINDOW_DAYS_LIST = [60, 90, 120]
GRU_FEATURE_COLS = [
    "open", "high", "low", "close", "volume", "log_return",
    "upper_shadow_ratio", "lower_shadow_ratio", "body_ratio",
    "rsi_7", "rsi_14", "roc_3", "roc_5", "price_pos_5", "price_pos_50",
    "adx", "minus_di", "di_diff", "volume_delta",
    "return_lag_1", "close_vs_ema_10",
    "vol_ratio_4vs16", "signed_volume_momentum",
    "funding_rate", "trend_alignment",
]

def _lazy_import_run_grid():
    """延迟导入 run_grid 模块（含 torch），仅在需要时调用"""
    # 防止环境里已有第三方 `src` 模块占位，导致 run_grid 的
    # `from src.python...` 导入落到错误模块。
    import importlib
    src_mod = sys.modules.get("src")
    if src_mod is not None:
        src_file = str(getattr(src_mod, "__file__", "") or "")
        if src_file and (not src_file.startswith(str(PROJECT_ROOT))):
            sys.modules.pop("src", None)
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    importlib.import_module("src.python.wallet_cohort_features")
    from experiments.sentiment_grid_search import run_grid
    return run_grid

# ─── 宏观 Regime 特征 ──────────────────────────────────────────────
def add_macro_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    宏观市场状态（Regime）特征 — 帮助模型识别牛熊周期位置。

    与现有 regime 特征的区别：
      - 现有特征: 50 bar (12.5 小时) 窗口 → 日内微观状态
      - 本函数:   200 天级别窗口 → 宏观牛熊周期

    关键洞察：
      特征的 lookback（200 天）和训练 window（90 天）是独立的。
      90 天窗口内，模型能看到 price_vs_sma200d 从 +0.1 变到 -0.15，
      这就是 regime 转换的信号，不需要训练窗口也是 200 天。
    """
    res = df.copy()
    close = df["close"]
    ts = df["timestamp"]

    # ── 先聚合到日线，再计算长周期特征，最后合并回 15 分钟 ──
    daily = (
        df.set_index("timestamp")[["close", "high", "low", "volume"]]
        .resample("1D")
        .agg({"close": "last", "high": "max", "low": "min", "volume": "sum"})
        .dropna(subset=["close"])
    )
    d_close = daily["close"]

    # 1. 价格 vs 200 日均线 — 最经典的牛熊分界线
    #    >0 = 牛市区域, <0 = 熊市区域
    sma_200 = d_close.rolling(200, min_periods=100).mean()
    daily["macro_price_vs_sma200d"] = (d_close / sma_200 - 1).clip(-0.5, 0.5)

    # 2. 价格 vs 50 日均线 — 中期趋势
    sma_50 = d_close.rolling(50, min_periods=25).mean()
    daily["macro_price_vs_sma50d"] = (d_close / sma_50 - 1).clip(-0.3, 0.3)

    # 3. 50 日均线 vs 200 日均线 — 金叉/死叉信号
    #    >0 = 金叉（牛）, <0 = 死叉（熊）
    daily["macro_sma50_vs_sma200"] = (sma_50 / sma_200 - 1).clip(-0.3, 0.3)

    # 4. 距离 365 日最高点的回撤幅度 — 衡量熊市深度
    #    0 = 在高点, -0.5 = 从高点跌了 50%
    rolling_max_365 = d_close.rolling(365, min_periods=100).max()
    daily["macro_drawdown_365d"] = (d_close / rolling_max_365 - 1).clip(-0.8, 0)

    # 5. 距离 365 日最低点的涨幅 — 衡量牛市高度
    rolling_min_365 = d_close.rolling(365, min_periods=100).min()
    daily["macro_rally_365d"] = (d_close / rolling_min_365 - 1).clip(0, 5.0)

    # 6. 短期动量 vs 长期动量 — 趋势加速/减速
    #    >0 = 趋势加速, <0 = 趋势减速/反转中
    mom_30 = d_close.pct_change(30)
    mom_90 = d_close.pct_change(90)
    daily["macro_momentum_accel"] = (mom_30 - mom_90 / 3).clip(-0.5, 0.5)

    # 7. 波动率变化率 — 波动率骤变 = regime 切换信号
    #    >1.5 = 波动率骤升（可能正在转势）
    d_returns = d_close.pct_change()
    vol_30 = d_returns.rolling(30, min_periods=15).std()
    vol_90 = d_returns.rolling(90, min_periods=45).std()
    daily["macro_vol_ratio"] = (vol_30 / vol_90.replace(0, np.nan)).clip(0.3, 3.0)

    # ── 合并回 15 分钟数据 ──
    regime_cols = [c for c in daily.columns if c.startswith("macro_")]
    daily_regime = daily[regime_cols].copy()
    daily_regime.index = pd.to_datetime(daily_regime.index)
    daily_regime["_merge_ts"] = daily_regime.index

    res = res.sort_values("timestamp").reset_index(drop=True)
    res = pd.merge_asof(
        res,
        daily_regime.reset_index(drop=True).sort_values("_merge_ts"),
        left_on="timestamp",
        right_on="_merge_ts",
        direction="backward",
    )
    res.drop("_merge_ts", axis=1, inplace=True, errors="ignore")

    n_valid = res[regime_cols[0]].notna().sum()
    print(f"regime 特征: {len(regime_cols)} 列, {n_valid}/{len(res)} 行有效")

    return res


# ─── 默认配置 ─────────────────────────────────────────────────────
DEFAULT_TRAIN_DAYS = 90     # 训练窗口长度
DEFAULT_VAL_DAYS = 15       # 验证集长度（用于 early stopping）
DEFAULT_TEST_DAYS = 15      # 测试窗口长度
DEFAULT_STEP_DAYS = 15      # 每步前进天数
DEFAULT_WARMUP_DAYS = 30    # 特征预热天数
DEFAULT_DATA_DAYS = 965     # BTC ETF 批准 (2024-01-10) 前 200 天起 ≈ 2023-06-24
PURGE_BARS = 4              # train/val 间隔
WF_ASSETS = ["BTC_USDT", "ETH_USDT", "XRP_USDT"]


def load_all_features(paths: dict, use_sim_noise: bool, data_days: int = DEFAULT_DATA_DAYS,
                      use_regime_features: bool = False) -> Dict[str, tuple]:
    """加载所有资产的特征 + 预缓存的 GRU 嵌入（限制到最近 data_days 天）"""
    import pickle

    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=data_days)
    cache_dir = PROJECT_ROOT / "experiments" / "sentiment_grid_search" / "cache"
    noise_tag = "noise" if use_sim_noise else "clean"

    # ★ 一次性从 pickle 加载所有资产（避免 PyArrow 多次读取死锁）
    pkl_path = cache_dir / f"all_data_{noise_tag}.pkl"
    t0 = time.time()
    with open(pkl_path, "rb") as f:
        raw_data = pickle.load(f)
    print(f"  pickle 加载: {time.time()-t0:.1f}s ({len(raw_data)} assets)")

    asset_data = {}
    for idx, asset in enumerate(WF_ASSETS):
        print(f"  {asset}:", end=" ", flush=True)

        tech_df, emb_df = raw_data[asset]
        tech_df = _ensure_datetime_col(tech_df, "timestamp")
        full_len = len(tech_df)

        # Regime 特征在 cutoff 之前计算（需要 200 天+ 历史算 SMA200）
        if use_regime_features:
            tech_df = add_macro_regime_features(tech_df)

        tech_df = tech_df[tech_df["timestamp"] >= cutoff_date].reset_index(drop=True)

        # sim-noise（仅影响 tech_df，嵌入已按 clean/noise 分别缓存）
        if use_sim_noise:
            tech_df = apply_simulated_candle_noise(tech_df, seed=SEED + idx)

        print(f"{full_len} → {len(tech_df)} rows (cache: {noise_tag})")
        asset_data[asset] = (tech_df, emb_df)

    del raw_data  # 释放原始数据内存
    return asset_data


def pool_assets(asset_data: dict, groups: list, paths: dict, btc_weight_boost: float) -> pd.DataFrame:
    """合池所有资产数据"""
    asset_names = sorted(asset_data.keys())
    frames = []
    for idx, asset in enumerate(asset_names):
        tech_df, emb_df = asset_data[asset]
        rg = _lazy_import_run_grid()
        prepared = rg._prepare_asset_for_pooling(
            tech_df, emb_df, groups, paths, asset, asset_id=idx,
        )
        frames.append(prepared)

    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled.sort_values("timestamp").reset_index(drop=True)

    if btc_weight_boost != 1.0:
        btc_mask = pooled["asset_id"] == 0
        pooled.loc[btc_mask, "sample_weight"] *= btc_weight_boost

    return pooled


def get_feature_cols(pooled: pd.DataFrame) -> list:
    """获取特征列"""
    exclude = {
        "timestamp", "timestamp_ms", "date", "asset", "_asset_name",
        "direction_label", "sample_weight",
        "symbol", "open", "high", "low", "close", "volume", "btc_close",
    }
    feature_cols = []
    for c in pooled.columns:
        if c in exclude:
            continue
        if c == "asset_id":
            feature_cols.append(c)
            continue
        if pooled[c].dtype not in ["float64", "float32", "int64", "int32"]:
            continue
        if pooled[c].isna().all():
            continue
        feature_cols.append(c)

    numeric_cols = [c for c in feature_cols if c != "asset_id"]
    rg = _lazy_import_run_grid()
    numeric_cols = rg._deduplicate_features(pooled, numeric_cols, threshold=0.95)
    feature_cols = numeric_cols + ["asset_id"]
    return feature_cols


def train_and_eval_window(
    pooled: pd.DataFrame,
    feature_cols: list,
    lgb_params: dict,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    n_assets: int,
    window_days_override: list = None,
) -> dict:
    """在一个窗口上训练 + 评估"""
    purge_rows = PURGE_BARS * n_assets

    train_mask = (pooled["timestamp"] >= train_start) & (pooled["timestamp"] < train_end)
    val_mask = (pooled["timestamp"] >= val_start) & (pooled["timestamp"] < val_end)
    test_mask = (pooled["timestamp"] >= test_start) & (pooled["timestamp"] < test_end)

    train_df = pooled[train_mask].copy()
    val_df = pooled[val_mask].copy()
    test_df = pooled[test_mask].copy()

    if len(train_df) < 100 or len(val_df) < 20 or len(test_df) < 20:
        return None

    # Purge gap
    if len(train_df) > purge_rows:
        train_df = train_df.iloc[:-purge_rows]

    X_va = val_df[feature_cols]
    y_va = val_df["direction_label"]
    X_te = test_df[feature_cols]
    y_te = test_df["direction_label"]

    # Expanding window ensemble
    ensemble_probas = []
    _window_list = window_days_override if window_days_override else WINDOW_DAYS_LIST
    for w_days in _window_list:
        sub_start = train_end - pd.Timedelta(days=w_days)
        if sub_start < train_start:
            sub_start = train_start

        sub_mask = (train_df["timestamp"] >= sub_start)
        sub_train = train_df[sub_mask]

        if len(sub_train) < 50:
            continue

        model = lgb.LGBMClassifier(**lgb_params, n_estimators=1000, random_state=SEED)
        model.fit(
            sub_train[feature_cols], sub_train["direction_label"],
            sample_weight=sub_train["sample_weight"],
            eval_set=[(X_va, y_va)],
            categorical_feature=["asset_id"],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

        proba = model.predict_proba(X_te)[:, 1]
        ensemble_probas.append(proba)

    if not ensemble_probas:
        return None

    proba_te = np.mean(ensemble_probas, axis=0)
    y_te_np = y_te.values

    # Overall metrics
    auc = roc_auc_score(y_te_np, proba_te)
    brier = brier_score_loss(y_te_np, proba_te)
    acc = ((proba_te >= 0.5).astype(int) == y_te_np).mean()
    pred_dir = (proba_te >= 0.5).astype(int)
    correct = (pred_dir == y_te_np).astype(float)
    pnl = correct * 2 - 1 - 0.001
    sharpe = pnl.mean() / (pnl.std() + 1e-10) * np.sqrt(252 * 96)

    # Per-asset metrics
    asset_name_np = test_df["_asset_name"].values
    per_asset = {}
    for a in sorted(set(asset_name_np)):
        mask = asset_name_np == a
        if mask.sum() < 10:
            continue
        a_auc = roc_auc_score(y_te_np[mask], proba_te[mask])
        a_acc = ((proba_te[mask] >= 0.5).astype(int) == y_te_np[mask]).mean()
        per_asset[a] = {"auc": round(float(a_auc), 4), "accuracy": round(float(a_acc), 4), "n": int(mask.sum())}

    return {
        "test_start": str(test_start.date()),
        "test_end": str(test_end.date()),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "auc": round(float(auc), 5),
        "accuracy": round(float(acc), 4),
        "brier": round(float(brier), 5),
        "sharpe": round(float(sharpe), 2),
        "per_asset": per_asset,
    }


def main():
    parser = argparse.ArgumentParser(description="Walk-Forward 验证")
    parser.add_argument("--train-days", type=int, default=DEFAULT_TRAIN_DAYS,
                        help=f"训练窗口天数 (默认 {DEFAULT_TRAIN_DAYS})")
    parser.add_argument("--val-days", type=int, default=DEFAULT_VAL_DAYS,
                        help=f"验证集天数 (默认 {DEFAULT_VAL_DAYS})")
    parser.add_argument("--test-days", type=int, default=DEFAULT_TEST_DAYS,
                        help=f"测试窗口天数 (默认 {DEFAULT_TEST_DAYS})")
    parser.add_argument("--step-days", type=int, default=DEFAULT_STEP_DAYS,
                        help=f"每步前进天数 (默认 {DEFAULT_STEP_DAYS})")
    parser.add_argument("--sim-noise", action="store_true",
                        help="启用模拟K线噪声增强")
    parser.add_argument("--regularized", action="store_true",
                        help="使用正则化 LightGBM 参数")
    parser.add_argument("--btc-weight-boost", type=float, default=1.0,
                        help="BTC 样本权重提升倍数")
    parser.add_argument("--exp", type=int, choices=[7, 8, 13, 14, 15, 16], default=8,
                        help="实验版本: 7=Exp7, 8=Exp8, 13=Exp13(Exp8+TV), 14=Exp14(Exp10+TV), "
                             "15=Exp15(Exp8基础,365d), 16=Exp16(Exp13+TV,365d)")
    parser.add_argument("--regime-features", action="store_true",
                        help="添加宏观 Regime 特征 (SMA200/50, 回撤, 动量加速等)")
    args = parser.parse_args()

    lgb_params = REGULARIZED_PARAMS if args.regularized else EXP7_TUNED_PARAMS
    lgb_name = "REGULARIZED" if args.regularized else "EXP7_TUNED"
    if args.exp == 7:
        groups = EXP7_GROUPS
    elif args.exp in (13, 14, 16):
        groups = EXP13_GROUPS
    elif args.exp == 15:
        groups = EXP15_GROUPS
    else:
        groups = EXP8_GROUPS

    # Exp14 隐含 sim-noise（与 Exp10 一致）
    if args.exp == 14 and not args.sim_noise:
        args.sim_noise = True
        print("  ⚡ Exp14 自动启用 sim-noise (与 Exp10 对齐)")

    # 训练窗口较长时自动调整 expanding window 列表
    if args.train_days > 120:
        half = args.train_days // 2
        two_third = args.train_days * 2 // 3
        WINDOW_DAYS_LIST_OVERRIDE = [half, two_third, args.train_days]
        print(f"  ⚡ 训练窗口 > 120d，自动调整 Expanding Window: {WINDOW_DAYS_LIST_OVERRIDE}")
    else:
        WINDOW_DAYS_LIST_OVERRIDE = None  # 用默认 [60, 90, 120]

    print(f"{'=' * 70}")
    print(f"  Walk-Forward 验证")
    print(f"  训练窗口: {args.train_days}d, 验证: {args.val_days}d, "
          f"测试: {args.test_days}d, 步长: {args.step_days}d")
    print(f"  LightGBM: {lgb_name}")
    print(f"  sim-noise: {'启用' if args.sim_noise else '关闭'}")
    print(f"  BTC boost: {args.btc_weight_boost}x")
    print(f"  Regime 特征: {'启用' if args.regime_features else '关闭'}")
    print(f"  特征组: Exp{args.exp}")
    print(f"{'=' * 70}")

    # paths — 内联 get_data_paths()（避免提前触发 torch 加载）
    _sent_dir = str(PROJECT_ROOT / "data" / "sentiment")
    _ob_dir = str(PROJECT_ROOT / "data" / "processed")
    paths = {
        "data_src": str(PROJECT_ROOT / "data" / "raw"),
        "cfgi_path": _sent_dir + "/cfgi_15m_history.parquet",
        "fgi_path": _sent_dir + "/fear_greed_history_daily.parquet",
        "news_path": _sent_dir + "/news_sentiment_history_15m.parquet",
        "ob_path": _ob_dir,
        "funding_path": _sent_dir + "/funding_rate_history.parquet",
        "oi_path": _sent_dir + "/open_interest_15m.parquet",
        "ls_path": _sent_dir + "/long_short_ratio_15m.parquet",
        "taker_path": _sent_dir + "/taker_buy_sell_15m.parquet",
        "polymarket_prob_path": _sent_dir,
        "polymarket_prob_target_path": _sent_dir,
        "tv_dir": _sent_dir,
    }

    # Phase 1: 加载全量数据
    print(f"\n{'─' * 70}")
    print(f"  Phase 1: 加载特征 + GRU 嵌入")
    print(f"{'─' * 70}")

    asset_data = load_all_features(paths, use_sim_noise=args.sim_noise,
                                    use_regime_features=args.regime_features)
    if not asset_data:
        print("❌ 数据加载失败")
        return

    # Phase 2: 合池
    print(f"\n{'─' * 70}")
    print(f"  Phase 2: 合池数据")
    print(f"{'─' * 70}")

    pooled = pool_assets(asset_data, groups, paths, args.btc_weight_boost)
    feature_cols = get_feature_cols(pooled)
    n_assets = len(sorted(asset_data.keys()))

    print(f"  合池: {len(pooled)} rows, {len(feature_cols)} features")
    print(f"  日期范围: {pooled['timestamp'].min()} ~ {pooled['timestamp'].max()}")

    # Phase 3: Walk-Forward
    print(f"\n{'─' * 70}")
    print(f"  Phase 3: Walk-Forward 验证")
    print(f"{'─' * 70}")

    latest_ts = pooled["timestamp"].max()
    earliest_ts = pooled["timestamp"].min()

    # 从最早可以的窗口开始，到最新
    window_size = args.train_days + args.val_days + args.test_days
    first_test_end = earliest_ts + pd.Timedelta(days=window_size + DEFAULT_WARMUP_DAYS)

    results = []
    window_idx = 0
    current_test_end = first_test_end

    while current_test_end <= latest_ts:
        test_end = current_test_end
        test_start = test_end - pd.Timedelta(days=args.test_days)
        val_end = test_start
        val_start = val_end - pd.Timedelta(days=args.val_days)
        train_end = val_start
        train_start = train_end - pd.Timedelta(days=args.train_days)

        window_idx += 1
        print(f"\n  窗口 {window_idx}: "
              f"Train [{train_start.strftime('%m/%d')}~{train_end.strftime('%m/%d')}] "
              f"Val [{val_start.strftime('%m/%d')}~{val_end.strftime('%m/%d')}] "
              f"Test [{test_start.strftime('%m/%d')}~{test_end.strftime('%m/%d')}]", end="")

        result = train_and_eval_window(
            pooled, feature_cols, lgb_params,
            train_start, train_end,
            val_start, val_end,
            test_start, test_end,
            n_assets,
            window_days_override=WINDOW_DAYS_LIST_OVERRIDE,
        )

        if result:
            result["window"] = window_idx
            results.append(result)
            print(f"  → AUC={result['auc']:.4f}, Acc={result['accuracy']:.3f}, "
                  f"Sharpe={result['sharpe']:.1f}")

            # Per-asset breakdown
            for a, m in sorted(result["per_asset"].items()):
                print(f"    {a}: AUC={m['auc']:.4f}, Acc={m['accuracy']:.3f} (n={m['n']})")
        else:
            print(f"  → 跳过（数据不足）")

        current_test_end += pd.Timedelta(days=args.step_days)

    # Phase 4: 汇总
    if not results:
        print("\n❌ 没有有效窗口")
        return

    print(f"\n{'=' * 70}")
    print(f"  Walk-Forward 汇总 ({len(results)} 个窗口)")
    print(f"{'=' * 70}")

    aucs = [r["auc"] for r in results]
    accs = [r["accuracy"] for r in results]
    sharpes = [r["sharpe"] for r in results]

    print(f"\n  Overall AUC:      {np.mean(aucs):.4f} ± {np.std(aucs):.4f} "
          f"(min={np.min(aucs):.4f}, max={np.max(aucs):.4f})")
    print(f"  Overall Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f} "
          f"(min={np.min(accs):.3f}, max={np.max(accs):.3f})")
    print(f"  Overall Sharpe:   {np.mean(sharpes):.1f} ± {np.std(sharpes):.1f} "
          f"(min={np.min(sharpes):.1f}, max={np.max(sharpes):.1f})")

    # Per-asset 汇总
    all_assets = set()
    for r in results:
        all_assets.update(r["per_asset"].keys())

    print(f"\n  Per-Asset AUC/Accuracy:")
    for a in sorted(all_assets):
        a_aucs = [r["per_asset"][a]["auc"] for r in results if a in r["per_asset"]]
        a_accs = [r["per_asset"][a]["accuracy"] for r in results if a in r["per_asset"]]
        if a_aucs:
            print(f"    {a}: AUC={np.mean(a_aucs):.4f}±{np.std(a_aucs):.4f}, "
                  f"Acc={np.mean(a_accs):.3f}±{np.std(a_accs):.3f} ({len(a_aucs)} windows)")

    # 找最差窗口
    worst = min(results, key=lambda r: r["auc"])
    print(f"\n  最差窗口: #{worst['window']} ({worst['test_start']}~{worst['test_end']}) "
          f"AUC={worst['auc']:.4f}")
    for a, m in sorted(worst["per_asset"].items()):
        print(f"    {a}: AUC={m['auc']:.4f}, Acc={m['accuracy']:.3f}")

    # 保存结果
    output_dir = PROJECT_ROOT / "experiments" / "sentiment_grid_search" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{'reg' if args.regularized else 'orig'}_{'noise' if args.sim_noise else 'clean'}"
    if args.exp in (13, 14, 15, 16):
        tag += f"_exp{args.exp}"
    if args.regime_features:
        tag += "_regime"
    if args.btc_weight_boost != 1.0:
        tag += f"_btc{args.btc_weight_boost:.1f}"
    if args.train_days != DEFAULT_TRAIN_DAYS:
        tag += f"_t{args.train_days}d"
    output_file = output_dir / f"walk_forward_{tag}.json"

    output = {
        "config": {
            "train_days": args.train_days,
            "val_days": args.val_days,
            "test_days": args.test_days,
            "step_days": args.step_days,
            "lgb_params": lgb_name,
            "sim_noise": args.sim_noise,
            "regime_features": args.regime_features,
            "btc_weight_boost": args.btc_weight_boost,
            "exp": args.exp,
        },
        "summary": {
            "n_windows": len(results),
            "mean_auc": round(float(np.mean(aucs)), 5),
            "std_auc": round(float(np.std(aucs)), 5),
            "mean_accuracy": round(float(np.mean(accs)), 4),
            "std_accuracy": round(float(np.std(accs)), 4),
            "mean_sharpe": round(float(np.mean(sharpes)), 2),
        },
        "windows": results,
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存: {output_file}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
