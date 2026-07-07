#!/usr/bin/env python3
"""
交易规则超参优化 v2 — Optuna 贝叶斯搜索 + Bootstrap 鲁棒性检验

12 个可调参数（连续空间），用 v5 模型在测试集上的预测进行回测。
比 v1 网格搜索更强：
  1. Optuna TPE（贝叶斯）：10,000+ trials，连续空间不漏最优
  2. Bootstrap 重采样：30 次随机打乱，防止过拟合单一序列
  3. Walk-forward 验证：前半段优化、后半段验证，确保泛化
  4. 12 参数、更大范围（激进模式）

用法:
  python scripts/optimize_trading_rules.py
  python scripts/optimize_trading_rules.py --trials 20000
  python scripts/optimize_trading_rules.py --trials 10000 --bootstrap 50
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# macOS 防死锁：强制 spawn（不用 fork）
if sys.platform == "darwin":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    os.environ.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numba
import numpy as np
import pandas as pd
import optuna

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "gru_regime_v1"))

from experiments.sentiment_grid_search.run_grid import (
    ASSETS, SEED, WINDOW_DAYS_LIST,
    DEFAULT_DAYS, WARMUP_DAYS, TEST_DAYS, VAL_DAYS, PURGE_BARS,
    GRU_HPARAMS, GRU_FEATURE_COLS,
    build_tech_features as _upstream_build_tech_features, extract_embeddings,
    _prepare_asset_for_pooling, _deduplicate_features,
    get_data_paths,
)
from experiments.sentiment_grid_search.data_prep import (
    _ensure_datetime_col, load_ohlcv, merge_funding_rate,
)
from experiments.gru_regime_v1.src.utils import get_device
from experiments.gru_regime_v1.src.utils import get_asset_filename

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

MODEL_DIR = PROJECT_ROOT / "data" / "models" / "v5_production"  # 默认; main() 可覆盖
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "sentiment_grid_search" / "results"
BUILD_FEATURES_TIMEOUT_SEC = int(os.getenv("POLYFUN_BUILD_FEATURES_TIMEOUT_SEC", "1800"))
_BUILD_TECH_FEATURES_CACHE: dict[tuple[str, str, int, str], pd.DataFrame] = {}
_BUILD_TECH_FEATURES_CACHE_LOCK = threading.RLock()
_BUILD_TECH_FEATURES_SINGLEFLIGHT = threading.Lock()


def build_tech_features(
    data_src: str | Path,
    asset: str,
    tail_rows: int = 0,
    recipe: Dict[str, Any] | None = None,
    freq_min: int = 15,
) -> pd.DataFrame:
    """Single-flight + cache wrapper to avoid repeated full feature builds in parallel."""
    recipe_key = json.dumps(recipe or {}, ensure_ascii=False, sort_keys=True)
    key = (str(Path(data_src)), str(asset), int(tail_rows), recipe_key)
    with _BUILD_TECH_FEATURES_CACHE_LOCK:
        cached = _BUILD_TECH_FEATURES_CACHE.get(key)
        if cached is not None:
            return cached.copy()

    with _BUILD_TECH_FEATURES_SINGLEFLIGHT:
        with _BUILD_TECH_FEATURES_CACHE_LOCK:
            cached = _BUILD_TECH_FEATURES_CACHE.get(key)
            if cached is not None:
                return cached.copy()

        data_file = Path(data_src) / get_asset_filename(asset, freq_min)
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_out = tmp.name

        tail_line = f"\ndf = df.tail({tail_rows}).reset_index(drop=True)" if tail_rows > 0 else ""
        script = f'''
import sys, pandas as pd
import json
sys.path.insert(0, "{PROJECT_ROOT}")
from src.python.feature_engineering import build_features
recipe = json.loads({recipe_key!r})
df = pd.read_parquet("{data_file}"){tail_line}
df = build_features(df, recipe=recipe)
df.to_parquet("{tmp_out}", index=False)
print(f"SUCCESS:{{len(df.columns)}}")
'''
        try:
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                timeout=BUILD_FEATURES_TIMEOUT_SEC,
                cwd=str(PROJECT_ROOT),
            )
            if result.returncode != 0 or "SUCCESS:" not in result.stdout:
                raise RuntimeError(f"build_features 失败: {result.stderr[:500]}")
            df = pd.read_parquet(tmp_out)
        finally:
            Path(tmp_out).unlink(missing_ok=True)

        with _BUILD_TECH_FEATURES_CACHE_LOCK:
            _BUILD_TECH_FEATURES_CACHE[key] = df
        return df.copy()

# ─── 模拟 K 线噪声参数（与 train_production_model.py 完全一致）─────
SIM_NOISE_STD = 0.0013  # 对数空间标准差


def apply_simulated_candle_noise(tech_df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """在 close/high/low/volume 上添加随机噪声, 模拟 T-120s 不完整 K 线。"""
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


def _fix_direction_labels(df: pd.DataFrame, clean_closes: dict):
    """将噪声方向标签替换为基于真实 close 的方向标签。

    实盘中 Polymarket 按真实 close 结算，所以超参优化/回测的标签必须用真实方向。
    """
    for asset, clean_df in clean_closes.items():
        mask = df["_asset_name"] == asset
        if mask.sum() == 0:
            continue
        clean_sorted = clean_df.sort_values("timestamp").reset_index(drop=True)
        next_close = clean_sorted["close"].shift(-1)
        clean_label = np.where(
            next_close.isna(), np.nan,
            (next_close > clean_sorted["close"]).astype(np.float32),
        )
        ts_to_label = dict(zip(clean_sorted["timestamp"], clean_label))
        df.loc[mask, "direction_label"] = df.loc[mask, "timestamp"].map(ts_to_label)
    df.dropna(subset=["direction_label"], inplace=True)

# ════════════════════════════════════════════════════════════
#  12 个交易规则超参 — 搜索范围定义
# ════════════════════════════════════════════════════════════
#
# ┌─────────────────────────┬──────────────────┬────────────────────────────────────────────┐
# │ 参数名                  │ 搜索范围          │ 含义                                       │
# ├─────────────────────────┼──────────────────┼────────────────────────────────────────────┤
# │ min_confidence          │ [0.500, 0.580]   │ 最低置信度门槛：低于此值不交易               │
# │                         │                  │ 模型输出 P(UP)=0.54 → confidence=0.54      │
# │                         │                  │ 越高 → 交易越少但胜率越高                   │
# ├─────────────────────────┼──────────────────┼────────────────────────────────────────────┤
# │ min_edge                │ [0.000, 0.080]   │ 最小期望边际：edge = p*odds - (1-p)         │
# │                         │                  │ 类似赌场优势率，确保正期望                   │
# │                         │                  │ 0 = 只要置信度过关就交易                    │
# ├─────────────────────────┼──────────────────┼────────────────────────────────────────────┤
# │ kelly_frac              │ [0.05, 2.00]     │ Kelly 乘数：full Kelly × 此值 = 下注比例     │
# │                         │                  │ <1: 保守，1=标准 Kelly，>1=超级激进          │
# │                         │                  │ 研究表明 0.25~0.5 最优（防尾部风险）         │
# │                         │                  │ 但我们开放到 2.0 让 Optuna 自己找            │
# ├─────────────────────────┼──────────────────┼────────────────────────────────────────────┤
# │ max_capital_pct         │ [0.01, 0.50]     │ 单笔最大仓位（占总资金比例）                 │
# │                         │                  │ 0.10 = 最多用 10% 资金下一单                │
# │                         │                  │ 上限 50% 允许搜索激进策略                   │
# ├─────────────────────────┼──────────────────┼────────────────────────────────────────────┤
# │ conf_tier1_bound        │ [0.510, 0.560]   │ 低信心档上界 = 中信心档下界                  │
# │                         │                  │ [min_conf, tier1_bound) = 低档              │
# ├─────────────────────────┼──────────────────┼────────────────────────────────────────────┤
# │ conf_tier2_bound        │ [0.560, 0.650]   │ 中信心档上界 = 高信心档下界                  │
# │                         │                  │ [tier1_bound, tier2_bound) = 中档            │
# │                         │                  │ [tier2_bound, 1.0) = 高档                   │
# ├─────────────────────────┼──────────────────┼────────────────────────────────────────────┤
# │ tier1_mult              │ [0.05, 1.00]     │ 低档下注乘数（0.2=只下 20% of Kelly）        │
# │                         │                  │ 越低 → 低信心交易越谨慎                     │
# ├─────────────────────────┼──────────────────┼────────────────────────────────────────────┤
# │ tier2_mult              │ [0.20, 1.00]     │ 中档下注乘数                               │
# ├─────────────────────────┼──────────────────┼────────────────────────────────────────────┤
# │ tier3_mult              │ [0.50, 1.50]     │ 高档下注乘数（>1 = 超配高信心交易）           │
# ├─────────────────────────┼──────────────────┼────────────────────────────────────────────┤
# │ cooldown_bars           │ [0, 8]           │ 连续亏损后暂停 N 根 bar                     │
# │                         │                  │ 0 = 不暂停，8 = 亏损后休息 2 小时            │
# │                         │                  │ 防止"倾斜"连亏                              │
# ├─────────────────────────┼──────────────────┼────────────────────────────────────────────┤
# │ drawdown_halt           │ [0.05, 0.50]     │ 回撤熔断线：DD 超此值 → 停止交易              │
# │                         │                  │ 0.15 = 跌 15% 停机                         │
# │                         │                  │ 小 = 保守（常停），大 = 激进（不轻易停）       │
# ├─────────────────────────┼──────────────────┼────────────────────────────────────────────┤
# │ compound                │ {True, False}    │ 复利 vs 固定本金                            │
# │                         │                  │ True = 赢了钱加大下注（指数增长/回撤）        │
# │                         │                  │ False = 固定用初始资金计算下注               │
# └─────────────────────────┴──────────────────┴────────────────────────────────────────────┘
#
# 搜索方法：Optuna TPE（Tree-structured Parzen Estimator）
# - 贝叶斯优化：前 200 trials 随机探索，后续根据历史结果自动聚焦高分区域
# - 比 Grid Search 效率高 10-100 倍（11 维连续空间 Grid 需要 ~10^8 点）
# - 比 Random Search 效率高 3-5 倍（TPE 有记忆，不重复探索差区域）


def generate_predictions(model_dir: Path, apply_noise: bool = False, test_days: int | None = None) -> pd.DataFrame:
    """用 v5 模型对测试集生成逐 bar 预测。

    Args:
        model_dir: 模型目录路径
        apply_noise: 对特征应用 T-120s 模拟噪声，但标签用真实 close 方向。
                     用于 sim_noise 模型的超参优化（匹配实盘条件）。
    """
    print("=" * 70)
    print("  Phase 1: 生成 v5 模型预测")
    if apply_noise:
        print("  ⚠️  噪声模式: close/high/low/volume 加 T-120s 模拟噪声, 标签保持真实方向")
    print("=" * 70)

    with open(model_dir / "config.json") as f:
        config = json.load(f)

    models = []
    for w_days in config["window_days_list"]:
        models.append(joblib.load(model_dir / f"lgb_{w_days}d.joblib"))

    with open(model_dir / "feature_cols.json") as f:
        feature_cols = json.load(f)

    asset_map = config["asset_map"]
    device = get_device(use_mps=True)
    paths = get_data_paths()

    effective_test_days = int(test_days) if test_days is not None else int(TEST_DAYS)
    total_window = DEFAULT_DAYS + WARMUP_DAYS + VAL_DAYS + effective_test_days
    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=total_window)

    gru_model_dir = PROJECT_ROOT / "experiments" / "sentiment_grid_search" / "models"

    print("构建特征 + GRU 嵌入...")
    asset_data = {}
    clean_closes = {}  # 噪声模式: 保存干净 close 用于标签修正
    for idx, asset in enumerate(sorted(asset_map.keys())):
        tech_df = build_tech_features(paths["data_src"], asset)
        tech_df = _ensure_datetime_col(tech_df, "timestamp")
        tech_df = tech_df[tech_df["timestamp"] >= cutoff_date].reset_index(drop=True)

        if apply_noise:
            # 保存干净 close（后续修正 direction_label 用）
            clean_closes[asset] = tech_df[["timestamp", "close"]].copy()
            # 应用噪声（与训练 sim_noise 模型时完全一致）
            tech_df = apply_simulated_candle_noise(tech_df, seed=SEED + idx)
            # 噪声后强制重算 log_return
            tech_df["log_return"] = np.log(tech_df["close"] / tech_df["close"].shift(1))
        elif "log_return" not in tech_df.columns:
            tech_df["log_return"] = np.log(tech_df["close"] / tech_df["close"].shift(1))
        if Path(paths.get("funding_path", "")).exists():
            tech_df = merge_funding_rate(tech_df, paths["funding_path"], asset)

        ohlcv_df = load_ohlcv(paths["data_src"], asset)
        ohlcv_df = ohlcv_df[ohlcv_df["timestamp"] >= cutoff_date].reset_index(drop=True)
        if "timestamp_ms" not in tech_df.columns and "timestamp_ms" in ohlcv_df.columns:
            tech_df = pd.merge_asof(
                tech_df.sort_values("timestamp"),
                ohlcv_df[["timestamp", "timestamp_ms"]].sort_values("timestamp"),
                on="timestamp", direction="nearest",
                tolerance=pd.Timedelta("1min"),
            )

        for c in GRU_FEATURE_COLS:
            if c not in tech_df.columns:
                tech_df[c] = 0.0

        model_path = gru_model_dir / f"{asset}_enhanced.pt"
        norm_path = gru_model_dir / f"{asset}_enhanced_normalizer.json"
        emb_df = extract_embeddings(tech_df, model_path, norm_path, device)

        asset_data[asset] = (tech_df, emb_df)
        print(f"  {asset}: {len(tech_df)} rows")

    # 合池
    print("合池 + 预测...")
    feature_groups = config["feature_groups"]
    asset_names = sorted(asset_data.keys())
    frames = []
    for idx, asset in enumerate(asset_names):
        tech_df, emb_df = asset_data[asset]
        prepared = _prepare_asset_for_pooling(
            tech_df, emb_df, feature_groups, paths, asset, asset_id=idx,
        )
        frames.append(prepared)

    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled.sort_values("timestamp").reset_index(drop=True)

    # 取测试集
    latest_ts = pooled["timestamp"].max()
    test_cutoff = latest_ts - pd.Timedelta(days=effective_test_days)
    test_df = pooled[pooled["timestamp"] >= test_cutoff].copy()

    print(f"测试集: {len(test_df)} rows ({effective_test_days} 天)")

    # 噪声模式: 用真实 close 重算 direction_label（Polymarket 按真实 close 结算）
    if apply_noise and clean_closes:
        _fix_direction_labels(test_df, clean_closes)
        print(f"  ✅ 标签已修正为真实方向（{len(clean_closes)} 币种）")

    for c in feature_cols:
        if c not in test_df.columns:
            test_df[c] = 0

    X_te = test_df[feature_cols]

    # 3 模型集成
    probas_list = [m.predict_proba(X_te)[:, 1] for m in models]
    ensemble_proba = np.mean(probas_list, axis=0)

    result = pd.DataFrame({
        "timestamp": test_df["timestamp"].values,
        "asset": test_df["_asset_name"].values,
        "proba_up": ensemble_proba,
        "actual": test_df["direction_label"].values.astype(int),
        "log_return": test_df["log_return"].values if "log_return" in test_df.columns else 0,
    })

    print(f"\n预测统计:")
    print(f"  总 bars:    {len(result)}")
    print(f"  各币:       {dict(result['asset'].value_counts())}")
    print(f"  P(UP) 分布: mean={ensemble_proba.mean():.4f}, "
          f"std={ensemble_proba.std():.4f}, "
          f"[{ensemble_proba.min():.4f}, {ensemble_proba.max():.4f}]")
    print(f"  实际涨跌:   {result['actual'].value_counts().to_dict()}")

    # 置信度分布（用于参数范围参考）
    conf = np.maximum(ensemble_proba, 1 - ensemble_proba)
    for pct in [50, 75, 90, 95, 99]:
        print(f"  置信度 P{pct}: {np.percentile(conf, pct):.4f}")

    return result


# ════════════════════════════════════════════════════════════
#  Numba JIT 核心模拟循环 — 与原始 Python 逻辑 100% 等价
#  编译成机器码执行，速度提升 30~50 倍
# ════════════════════════════════════════════════════════════

@numba.njit(cache=True)
def _simulate_trading_core(
    probas, actuals, odds_arr,
    initial_capital, settlement_cost,
    min_conf, min_edge, kelly_frac,
    bet_pct_normal, bet_pct_conservative,
    tier1_bound, tier2_bound,
    tier1_mult, tier2_mult, tier3_mult,
    cooldown, dd_halt,
    liquidity_cap, liquidity_bet,
    halt_duration_bars, recovery_target,
):
    """Numba JIT 编译的核心交易模拟循环。

    所有参数均为 numpy 数组或标量（无 Python 对象），
    逻辑与原始 simulate_trading 的 for 循环完全一致。

    费率策略（与 prediction_executor.ts 一致）：
    - Edge/Kelly 使用纯数学公式，不含费率
    - 费率 (settlement_cost) 仅在 PnL 结算时扣除
    """
    n = len(probas)
    pnl_arr = np.zeros(n, dtype=np.float64)

    capital = initial_capital
    peak_capital = initial_capital
    max_drawdown = 0.0
    wins = 0
    losses = 0
    cooldown_remaining = 0
    halted = False
    halt_start_bar = 0
    ever_reached_cap = False

    for i in range(n):
        proba = probas[i]
        actual = actuals[i]

        # 回撤熔断 — 24h circuit breaker（halt_duration_bars 自动解除）
        if peak_capital > 0.0:
            dd = (peak_capital - capital) / peak_capital
        else:
            dd = 0.0

        if halted:
            bars_since_halt = i - halt_start_bar
            capital_recovered = capital >= peak_capital * recovery_target
            time_expired = bars_since_halt >= halt_duration_bars
            if capital_recovered or time_expired:
                halted = False
            else:
                continue

        if dd >= dd_halt:
            halted = True
            halt_start_bar = i
            continue

        # 冷却期
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue

        # 方向和置信度
        if proba >= 0.5:
            direction = 1
            confidence = proba
        else:
            direction = 0
            confidence = 1.0 - proba

        # Layer 0: 置信度门槛
        if confidence < min_conf:
            continue

        odds_this = odds_arr[i]

        # Layer 1: Edge 过滤（纯数学，不含费率）
        p = confidence
        q = 1.0 - p
        edge = p * odds_this - q
        if edge < min_edge:
            continue

        # Layer 2: Kelly（纯数学，不含费率）
        if odds_this > 0.0:
            kelly_f = (p * odds_this - q) / odds_this
        else:
            kelly_f = 0.0
        if kelly_f < 0.0:
            kelly_f = 0.0
        bet_ratio = kelly_f * kelly_frac

        # 置信度分档调整
        if confidence < tier1_bound:
            bet_ratio *= tier1_mult
        elif confidence < tier2_bound:
            bet_ratio *= tier2_mult
        else:
            bet_ratio *= tier3_mult

        # ─── 流动性封顶规则（三阶段仓位）────────────
        if capital >= liquidity_cap:
            # Phase 2: >= $60K → 固定 $3K（模拟真实流动性上限）
            ever_reached_cap = True
            bet_amount = liquidity_bet
        elif ever_reached_cap:
            # Phase 3: 曾到过 $60K 后跌回 → 保守仓位%
            if bet_ratio > bet_pct_conservative:
                bet_ratio = bet_pct_conservative
            bet_amount = capital * bet_ratio
        else:
            # Phase 1: < $60K 且从未到过 → 正常百分比下注
            if bet_ratio > bet_pct_normal:
                bet_ratio = bet_pct_normal
            bet_amount = capital * bet_ratio

        # 确保不超过可用资金，且有最低下注
        cap_limit = capital * 0.95
        if bet_amount > cap_limit:
            bet_amount = cap_limit
        if bet_amount < 1.0:  # 最低 $1
            continue

        # 判断胜负
        correct = (direction == 1 and actual == 1) or (direction == 0 and actual == 0)

        if correct:
            # 赢：获得赔率收益，结算时扣除手续费（Polymarket 手续费在买入时从份额扣除，赢时体现在净利）
            pnl = bet_amount * odds_this - bet_amount * settlement_cost
            wins += 1
            cooldown_remaining = 0
        else:
            # 输：亏掉本金（Polymarket 不另扣费，手续费已在买入时体现在 fewer shares）
            pnl = -bet_amount
            losses += 1
            if cooldown > 0:
                cooldown_remaining = cooldown

        capital += pnl
        pnl_arr[i] = pnl

        if capital > peak_capital:
            peak_capital = capital
        if peak_capital > 0.0:
            dd = (peak_capital - capital) / peak_capital
        else:
            dd = 0.0
        if dd > max_drawdown:
            max_drawdown = dd

        if capital <= 0.0:
            capital = 0.0
            break

    return capital, wins, losses, max_drawdown, ever_reached_cap, pnl_arr


def _warmup_numba():
    """预热 Numba JIT 编译（首次调用需 2~3 秒，后续用磁盘缓存秒启动）。"""
    dummy_p = np.array([0.55, 0.48, 0.52], dtype=np.float64)
    dummy_a = np.array([1, 0, 1], dtype=np.int64)
    dummy_o = np.array([0.9, 0.9, 0.9], dtype=np.float64)
    _simulate_trading_core(
        dummy_p, dummy_a, dummy_o,
        400.0, 0.005,
        0.51, 0.01, 0.5, 0.05, 0.03,
        0.52, 0.56, 0.3, 0.6, 1.0,
        2, 0.2, 60000.0, 3000.0,
        96, 0.85,
    )


def simulate_trading(
    predictions: pd.DataFrame,
    params: Dict[str, Any],
    initial_capital: float = 400.0,
    fee_rate: float = 0.015,        # Polymarket taker 手续费 ~1.5% (2026-01-19 起 15分钟加密市场)
    slippage_rate: float = 0.003,   # 滑点 ~0.3%
    buy_price: float = 0.527,       # 买入价（固定值 or 范围中心）
    buy_price_range: Tuple[float, float] | None = None,  # 动态买价范围 (low, high)
    rng_seed: int | None = None,    # 随机种子（动态买价用）
) -> Dict[str, float]:
    """模拟 Polymarket 二元期权交易 — Numba JIT 加速版。

    逻辑与原始版本 100% 等价，核心 for 循环由 Numba 编译为机器码执行。
    固定买价模式结果 bit-exact 一致；动态买价模式统计等价（随机序列映射不同）。
    """
    min_conf     = params["min_confidence"]
    min_edge     = params["min_edge"]
    kelly_frac   = params["kelly_frac"]
    bet_pct_normal = params["bet_pct_normal"]
    bet_pct_conservative = params["bet_pct_conservative"]
    tier1_bound  = params["conf_tier1_bound"]
    tier2_bound  = params["conf_tier2_bound"]
    tier1_mult   = params["tier1_mult"]
    tier2_mult   = params["tier2_mult"]
    tier3_mult   = params["tier3_mult"]
    # 周期制冷却: cooldown_bars 表示跳过的完整 15 分钟周期数，每周期含 num_assets 个预测机会
    num_assets   = int(predictions["asset"].nunique()) if "asset" in predictions.columns else 3
    cooldown     = int(params["cooldown_bars"]) * num_assets
    dd_halt      = params["drawdown_halt"]

    # 费率策略（与 prediction_executor.ts 一致）：
    # Edge/Kelly 不含费率（纯数学），费率只在 PnL 结算时扣除
    settlement_cost = fee_rate + slippage_rate  # 结算时扣除: taker ~1.5% + 滑点 0.3%
    odds = (1.0 - buy_price) / buy_price

    probas = np.ascontiguousarray(predictions["proba_up"].values, dtype=np.float64)
    actuals = np.ascontiguousarray(predictions["actual"].values, dtype=np.int64)
    n = len(probas)

    # 预生成每 bar 的赔率数组（固定模式=全部相同，动态模式=随机）
    if buy_price_range is not None:
        bp_rng = np.random.RandomState(rng_seed or 42)
        bp_prices = bp_rng.uniform(buy_price_range[0], buy_price_range[1], size=n)
        odds_arr = np.ascontiguousarray((1.0 - bp_prices) / bp_prices, dtype=np.float64)
    else:
        odds_arr = np.full(n, odds, dtype=np.float64)

    # 调用 Numba JIT 核心（edge/Kelly 无费率，settlement_cost 只用于 PnL 结算）
    capital, wins, losses, max_drawdown, ever_reached_cap, pnl_arr = _simulate_trading_core(
        probas, actuals, odds_arr,
        initial_capital, settlement_cost,
        min_conf, min_edge, kelly_frac,
        bet_pct_normal, bet_pct_conservative,
        tier1_bound, tier2_bound,
        tier1_mult, tier2_mult, tier3_mult,
        cooldown, dd_halt,
        60_000.0, 3_000.0,
        96, 0.85,  # 24h circuit breaker: 96 bars × 15min, recover at 85% of peak
    )

    # ─── 后处理（Python 层，只执行一次，不影响性能）───
    n_trades = wins + losses
    win_rate = wins / n_trades if n_trades > 0 else 0

    nonzero = pnl_arr[pnl_arr != 0]
    if len(nonzero) > 2:
        sharpe = float(nonzero.mean() / (nonzero.std() + 1e-10) * np.sqrt(252 * 96))
    else:
        sharpe = 0.0

    if len(predictions) > 1:
        ts_min = pd.to_datetime(predictions["timestamp"]).min()
        ts_max = pd.to_datetime(predictions["timestamp"]).max()
        span_days = max(1.0, float((ts_max - ts_min).total_seconds() / 86400.0))
    else:
        span_days = max(1.0, float(TEST_DAYS))
    trades_per_day = n_trades / span_days
    return_pct = (capital / initial_capital - 1) * 100

    gross_profit = float(nonzero[nonzero > 0].sum()) if len(nonzero) > 0 else 0.0
    gross_loss = abs(float(nonzero[nonzero < 0].sum())) if len(nonzero) > 0 else 1e-10
    profit_factor = gross_profit / (gross_loss + 1e-10)
    avg_win = float(nonzero[nonzero > 0].mean()) if len(nonzero[nonzero > 0]) > 0 else 0.0
    avg_loss = abs(float(nonzero[nonzero < 0].mean())) if len(nonzero[nonzero < 0]) > 0 else 0.0

    return {
        "final_capital": float(capital),
        "return_pct": float(return_pct),
        "n_trades": int(n_trades),
        "wins": int(wins),
        "losses": int(losses),
        "win_rate": float(win_rate),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "trades_per_day": float(trades_per_day),
        "profit_factor": float(profit_factor),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "ever_reached_cap": bool(ever_reached_cap),
    }


def simulate_bootstrap(
    predictions: pd.DataFrame,
    params: Dict[str, Any],
    n_bootstrap: int = 30,
    initial_capital: float = 1000.0,
    seed: int = SEED,
    buy_price: float = 0.527,
    buy_price_range: Tuple[float, float] | None = None,
) -> Dict[str, float]:
    """Bootstrap 重采样回测 — 防止过拟合单一序列。

    做 N 次：
    1. 按天分组（保留日内时序），随机打乱天的顺序
    2. 用打乱后的序列模拟交易
    3. 取各指标的中位数（不用均值，更鲁棒）
    """
    rng = np.random.RandomState(seed)

    # 按天分组
    predictions = predictions.copy()
    predictions["date"] = pd.to_datetime(predictions["timestamp"]).dt.date
    dates = sorted(predictions["date"].unique())

    all_results = []

    # 第 0 次 = 原始顺序（真实回测）
    res_orig = simulate_trading(predictions, params, initial_capital,
                               buy_price=buy_price, buy_price_range=buy_price_range)
    all_results.append(res_orig)

    # 后续 = 打乱天顺序
    for b in range(n_bootstrap - 1):
        shuffled_dates = rng.permutation(dates)
        chunks = [predictions[predictions["date"] == d] for d in shuffled_dates]
        shuffled_pred = pd.concat(chunks, ignore_index=True)
        res = simulate_trading(shuffled_pred, params, initial_capital,
                              buy_price=buy_price, buy_price_range=buy_price_range,
                              rng_seed=seed + b + 1)
        all_results.append(res)

    # 取中位数
    keys = list(all_results[0].keys())
    median_result = {}
    for k in keys:
        vals = [r[k] for r in all_results]
        median_result[k] = float(np.median(vals))

    # 也保留原始序列的结果
    
    median_result["orig_final_capital"] = res_orig["final_capital"]
    median_result["orig_return_pct"] = res_orig["return_pct"]
    median_result["orig_win_rate"] = res_orig["win_rate"]
    median_result["bootstrap_std_capital"] = float(np.std([r["final_capital"] for r in all_results]))

    return median_result


def run_optuna_search(
    predictions: pd.DataFrame,
    n_trials: int = 10000,
    n_bootstrap: int = 30,
    initial_capital: float = 1000.0,
    buy_price: float = 0.527,
    buy_price_range: Tuple[float, float] | None = None,
    min_trades: int = 100,
    optimize_buy_price: bool = False,
    buy_price_search_range: Tuple[float, float] = (0.44, 0.54),
    min_conf_range: Tuple[float, float] = (0.50, 0.56),
    min_edge_range: Tuple[float, float] = (0.00, 0.06),
    tier1_offset_range: Tuple[float, float] = (0.005, 0.04),
    tier2_offset_range: Tuple[float, float] = (0.03, 0.10),
) -> Tuple[optuna.Study, Dict[str, Any]]:
    """Optuna 贝叶斯搜索最优交易规则。

    Args:
        optimize_buy_price: 若为 True，将 buy_price 作为第 13 个可优化参数，
                           与其他 12 个交易规则参数联合优化。
        buy_price_search_range: 买价搜索范围，默认 [0.44, 0.54]。
    """

    odds = (1.0 - buy_price) / buy_price
    be = 1 / (1 + odds) + 0.009  # 盈亏平衡点（含 1.8% 成本）

    print("\n" + "=" * 70)
    print(f"  Phase 2: Optuna 贝叶斯搜索 ({n_trials} trials, {n_bootstrap}x bootstrap)")
    print(f"  最少交易笔数: {min_trades}")
    if optimize_buy_price:
        print(f"  买入价: 🔍 可优化参数 [{buy_price_search_range[0]:.2f}, {buy_price_search_range[1]:.2f}]")
        print(f"  搜索参数: 13 个（12 交易规则 + 买价）")
    elif buy_price_range:
        print(f"  买入价: 动态 ${buy_price_range[0]:.3f}~${buy_price_range[1]:.3f}")
    else:
        print(f"  买入价: ${buy_price:.3f} | 赔率: {odds:.4f} | 盈亏平衡: {be:.2%}")
    print("=" * 70)

    conf_low = max(0.0, min(0.999, float(min_conf_range[0])))
    conf_high = max(conf_low + 1e-6, min(0.999, float(min_conf_range[1])))
    edge_low = max(0.0, float(min_edge_range[0]))
    edge_high = max(edge_low + 1e-6, float(min_edge_range[1]))
    tier1_off_low = max(0.001, float(tier1_offset_range[0]))
    tier1_off_high = max(tier1_off_low + 1e-6, float(tier1_offset_range[1]))
    tier2_off_low = max(tier1_off_low + 1e-6, float(tier2_offset_range[0]))
    tier2_off_high = max(tier2_off_low + 1e-6, float(tier2_offset_range[1]))

    def _sample_tiers(trial: optuna.Trial, min_confidence: float) -> Tuple[float, float] | None:
        t1_low = max(min_confidence + tier1_off_low, 0.0)
        t1_high = min(min_confidence + tier1_off_high, 0.985)
        if t1_low >= t1_high:
            return None
        conf_tier1_bound = trial.suggest_float("conf_tier1_bound", t1_low, t1_high)

        t2_low = max(min_confidence + tier2_off_low, conf_tier1_bound + 0.003)
        t2_high = min(min_confidence + tier2_off_high, 0.995)
        if t2_low >= t2_high:
            return None
        conf_tier2_bound = trial.suggest_float("conf_tier2_bound", t2_low, t2_high)
        return conf_tier1_bound, conf_tier2_bound

    # 先做一次快速搜索（无 bootstrap）找前 50 名，再用 bootstrap 精选
    def objective_fast(trial):
        """快速目标函数 — 基于真实 Polymarket 约束。

        搜索空间（12 参数）:
        ┌──────────────────────┬──────────────┬──────────────────────────────┐
        │ 参数                 │ 范围          │ 说明                         │
        ├──────────────────────┼──────────────┼──────────────────────────────┤
        │ min_confidence       │ [0.500,0.560]│ 最低置信度（模型 P95=0.53）   │
        │ min_edge             │ [0.000,0.060]│ 净边际（扣除 1.8% 成本后）    │
        │ kelly_frac           │ [0.10, 1.00] │ Kelly 乘数（1=全 Kelly）      │
        │ bet_pct_normal       │ [0.02, 0.10] │ Phase1 仓位（资金<$60K）      │
        │ bet_pct_conservative │ [0.02, 0.05] │ Phase3 仓位（跌回<$60K）      │
        │ conf_tier1_bound     │ [0.505,0.540]│ 低→中档分界                  │
        │ conf_tier2_bound     │ [0.540,0.600]│ 中→高档分界                  │
        │ tier1_mult           │ [0.05, 1.00] │ 低档缩仓乘数                 │
        │ tier2_mult           │ [0.20, 1.00] │ 中档缩仓乘数                 │
        │ tier3_mult           │ [0.50, 1.50] │ 高档超配乘数                 │
        │ cooldown_bars        │ [0, 8]       │ 亏损后暂停 bar 数             │
        │ drawdown_halt        │ [0.08, 0.40] │ 回撤熔断线                   │
        └──────────────────────┴──────────────┴──────────────────────────────┘

        固定约束（不可调）:
        - 买入价: 由 --buy-price 决定（或 --optimize-buy-price 时作为第 13 参数搜索）
        - 初始资金: $400
        - 资金≥$60K: 固定 $3000/笔
        - 手续费+滑点: 1.8% (taker ~1.5% + 滑点 0.3%)
        """
        min_confidence = trial.suggest_float("min_confidence", conf_low, conf_high)
        tier_bounds = _sample_tiers(trial, min_confidence)
        if tier_bounds is None:
            return -1e6
        conf_tier1_bound, conf_tier2_bound = tier_bounds
        params = {
            "min_confidence":      min_confidence,
            "min_edge":            trial.suggest_float("min_edge", edge_low, edge_high),
            "kelly_frac":          trial.suggest_float("kelly_frac", 0.10, 1.00),
            "bet_pct_normal":      trial.suggest_float("bet_pct_normal", 0.02, 0.10),
            "bet_pct_conservative":trial.suggest_float("bet_pct_conservative", 0.02, 0.05),
            "conf_tier1_bound":    conf_tier1_bound,
            "conf_tier2_bound":    conf_tier2_bound,
            "tier1_mult":          trial.suggest_float("tier1_mult", 0.05, 1.00),
            "tier2_mult":          trial.suggest_float("tier2_mult", 0.20, 1.00),
            "tier3_mult":          trial.suggest_float("tier3_mult", 0.50, 1.50),
            "cooldown_bars":       trial.suggest_int("cooldown_bars", 0, 8),
            "drawdown_halt":       trial.suggest_float("drawdown_halt", 0.08, 0.40),
        }

        # 确保 tier1 < tier2, conservative <= normal
        if params["conf_tier1_bound"] >= params["conf_tier2_bound"]:
            return -1e6
        if params["bet_pct_conservative"] > params["bet_pct_normal"]:
            return -1e6

        # 买价：固定 or 可优化
        trial_buy_price = buy_price
        trial_buy_price_range = buy_price_range
        if optimize_buy_price:
            trial_buy_price = trial.suggest_float(
                "buy_price", buy_price_search_range[0], buy_price_search_range[1])
            trial_buy_price_range = None  # 优化模式下用固定值，不用范围

        res = simulate_trading(predictions, params, initial_capital,
                              buy_price=trial_buy_price,
                              buy_price_range=trial_buy_price_range,
                              rng_seed=SEED + trial.number)

        # 目标：最大化风险调整后的收益
        if res["n_trades"] < min_trades:
            return -1e6

        score = (
            max(res["return_pct"] + 100, 1)
            * (1 - res["max_drawdown"]) ** 2
            * min(res["profit_factor"], 3.0) ** 0.5
            * min(res["win_rate"], 0.99) ** 0.5
        )
        return score

    # ─── 目标函数 B: 最大资金模式 ─────────────────────────
    def objective_max_capital(trial):
        """最大资金目标 — 鼓励更多交易，用复利增长。

        与 safe 模式的区别:
        - 允许更低置信度（0.500~0.535）
        - 允许 edge=0（只要有正期望就交易）
        - 目标直接最大化 final_capital
        - 要求至少 50 笔交易
        """
        min_confidence = trial.suggest_float("min_confidence", conf_low, conf_high)
        tier_bounds = _sample_tiers(trial, min_confidence)
        if tier_bounds is None:
            return -1e6
        conf_tier1_bound, conf_tier2_bound = tier_bounds
        params = {
            "min_confidence":      min_confidence,
            "min_edge":            trial.suggest_float("min_edge", edge_low, edge_high),
            "kelly_frac":          trial.suggest_float("kelly_frac", 0.10, 1.00),
            "bet_pct_normal":      trial.suggest_float("bet_pct_normal", 0.02, 0.10),
            "bet_pct_conservative":trial.suggest_float("bet_pct_conservative", 0.02, 0.05),
            "conf_tier1_bound":    conf_tier1_bound,
            "conf_tier2_bound":    conf_tier2_bound,
            "tier1_mult":          trial.suggest_float("tier1_mult", 0.05, 1.00),
            "tier2_mult":          trial.suggest_float("tier2_mult", 0.20, 1.00),
            "tier3_mult":          trial.suggest_float("tier3_mult", 0.50, 1.50),
            "cooldown_bars":       trial.suggest_int("cooldown_bars", 0, 5),
            "drawdown_halt":       trial.suggest_float("drawdown_halt", 0.10, 0.40),
        }

        if params["conf_tier1_bound"] >= params["conf_tier2_bound"]:
            return -1e6
        if params["bet_pct_conservative"] > params["bet_pct_normal"]:
            return -1e6

        # 买价：固定 or 可优化
        trial_buy_price = buy_price
        trial_buy_price_range = buy_price_range
        if optimize_buy_price:
            trial_buy_price = trial.suggest_float(
                "buy_price", buy_price_search_range[0], buy_price_search_range[1])
            trial_buy_price_range = None

        res = simulate_trading(predictions, params, initial_capital,
                              buy_price=trial_buy_price,
                              buy_price_range=trial_buy_price_range,
                              rng_seed=SEED + trial.number)

        if res["n_trades"] < min_trades:
            return -1e6  # 最少 min_trades 笔

        # 目标: 直接最大化最终资金 × 交易频率奖励 × 回撤惩罚
        score = (
            max(res["final_capital"], 1)
            * np.log1p(res["n_trades"])  # 鼓励更多交易（对数，防止纯刷单）
            * (1 - res["max_drawdown"]) ** 1.5
        )
        return score

    # Phase 2A-1: Safe 模式搜索
    print(f"\n  Phase 2A-1: Safe 模式（高胜率、少交易）{n_trials} trials...")
    t0 = time.time()

    study_safe = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=300),
    )
    study_safe.optimize(objective_fast, n_trials=n_trials, show_progress_bar=True)

    elapsed_safe = time.time() - t0
    print(f"  完成: {elapsed_safe:.1f}s, best_score={study_safe.best_value:.4f}")

    # Phase 2A-2: Max Capital 模式搜索
    print(f"\n  Phase 2A-2: Max Capital 模式（更多交易、复利增长）{n_trials} trials...")
    t1_mc = time.time()

    study_maxcap = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED + 1, n_startup_trials=300),
    )
    study_maxcap.optimize(objective_max_capital, n_trials=n_trials, show_progress_bar=True)

    elapsed_mc = time.time() - t1_mc
    print(f"  完成: {elapsed_mc:.1f}s, best_score={study_maxcap.best_value:.4f}")

    # ─── 目标函数 C: Robust 模式（bootstrap 嵌入 Optuna）─────
    # 预计算 bootstrap 所需的按天分组索引（避免每个 trial 重复分组）
    # reset index to positional range to avoid iloc out-of-bounds when
    # upstream predictions keep sparse/original indices.
    _bs_predictions = predictions.reset_index(drop=True).copy()
    _bs_predictions["date"] = pd.to_datetime(_bs_predictions["timestamp"]).dt.date
    _bs_dates = sorted(_bs_predictions["date"].unique())
    _bs_date_indices = {}  # date -> row indices
    for d in _bs_dates:
        _bs_date_indices[d] = _bs_predictions[_bs_predictions["date"] == d].index.values

    N_BOOTSTRAP_IN_OPTUNA = 10  # 优化阶段用 10 次（快速），验证阶段用 30 次（精确）

    def objective_robust(trial):
        """Robust 目标函数 — 每个 trial 内部跑 bootstrap，直接优化中位数表现。

        解决 $0.48 类过拟合问题：
        - 不再优化"在特定时序下的最佳表现"
        - 而是优化"在任意时序打乱下的中位表现"
        - 确保找到的参数对时序顺序不敏感
        """
        min_confidence = trial.suggest_float("min_confidence", conf_low, conf_high)
        tier_bounds = _sample_tiers(trial, min_confidence)
        if tier_bounds is None:
            return -1e6
        conf_tier1_bound, conf_tier2_bound = tier_bounds
        params = {
            "min_confidence":      min_confidence,
            "min_edge":            trial.suggest_float("min_edge", edge_low, edge_high),
            "kelly_frac":          trial.suggest_float("kelly_frac", 0.10, 1.00),
            "bet_pct_normal":      trial.suggest_float("bet_pct_normal", 0.02, 0.10),
            "bet_pct_conservative":trial.suggest_float("bet_pct_conservative", 0.02, 0.05),
            "conf_tier1_bound":    conf_tier1_bound,
            "conf_tier2_bound":    conf_tier2_bound,
            "tier1_mult":          trial.suggest_float("tier1_mult", 0.05, 1.00),
            "tier2_mult":          trial.suggest_float("tier2_mult", 0.20, 1.00),
            "tier3_mult":          trial.suggest_float("tier3_mult", 0.50, 1.50),
            "cooldown_bars":       trial.suggest_int("cooldown_bars", 0, 8),
            "drawdown_halt":       trial.suggest_float("drawdown_halt", 0.08, 0.40),
        }

        if params["conf_tier1_bound"] >= params["conf_tier2_bound"]:
            return -1e6
        if params["bet_pct_conservative"] > params["bet_pct_normal"]:
            return -1e6

        # 买价：固定 or 可优化
        trial_buy_price = buy_price
        trial_buy_price_range = buy_price_range
        if optimize_buy_price:
            trial_buy_price = trial.suggest_float(
                "buy_price", buy_price_search_range[0], buy_price_search_range[1])
            trial_buy_price_range = None

        # 先跑原始序列做快速过滤
        res_orig = simulate_trading(predictions, params, initial_capital,
                                   buy_price=trial_buy_price,
                                   buy_price_range=trial_buy_price_range,
                                   rng_seed=SEED + trial.number)

        if res_orig["n_trades"] < min_trades:
            return -1e6

        # 跑 N 次 bootstrap
        rng = np.random.RandomState(SEED + trial.number)
        all_capitals = [res_orig["final_capital"]]
        all_dds = [res_orig["max_drawdown"]]
        all_n_trades = [res_orig["n_trades"]]

        for b in range(N_BOOTSTRAP_IN_OPTUNA - 1):
            shuffled_dates = rng.permutation(_bs_dates)
            idx_list = [_bs_date_indices[d] for d in shuffled_dates]
            shuffled_idx = np.concatenate(idx_list)
            shuffled_pred = _bs_predictions.iloc[shuffled_idx].reset_index(drop=True)
            res_b = simulate_trading(shuffled_pred, params, initial_capital,
                                    buy_price=trial_buy_price,
                                    buy_price_range=trial_buy_price_range,
                                    rng_seed=SEED + trial.number + b + 1)
            all_capitals.append(res_b["final_capital"])
            all_dds.append(res_b["max_drawdown"])
            all_n_trades.append(res_b["n_trades"])

        median_capital = float(np.median(all_capitals))
        median_dd = float(np.median(all_dds))
        median_n_trades = float(np.median(all_n_trades))

        if median_capital <= initial_capital:
            return -1e6  # 中位数都不赚钱

        # 目标：中位资金 × 回撤惩罚 × 交易频率奖励
        score = (
            max(median_capital, 1)
            * (1 - median_dd) ** 2
            * np.log1p(median_n_trades)
        )
        return score

    # Phase 2A-3: Robust 模式搜索
    print(f"\n  Phase 2A-3: Robust 模式（bootstrap-in-Optuna, {N_BOOTSTRAP_IN_OPTUNA}x）{n_trials} trials...")
    t2_rb = time.time()

    study_robust = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED + 2, n_startup_trials=300),
    )
    study_robust.optimize(objective_robust, n_trials=n_trials, show_progress_bar=True)

    elapsed_rb = time.time() - t2_rb
    print(f"  完成: {elapsed_rb:.1f}s, best_score={study_robust.best_value:.4f}")

    # 合并三个 study 的 top trials
    all_trials = []
    for trial in study_safe.trials:
        if trial.value and trial.value > -1e5:
            all_trials.append(trial)
    for trial in study_maxcap.trials:
        if trial.value and trial.value > -1e5:
            all_trials.append(trial)
    for trial in study_robust.trials:
        if trial.value and trial.value > -1e5:
            all_trials.append(trial)

    # 按原始序列 final_capital 重新评估排名
    print(f"\n  重新统一评估 {len(all_trials)} 个有效 trials（min_trades={min_trades}）...")

    # 如果 all_trials 为空，回退到所有 trial 中取最优
    if len(all_trials) == 0:
        print("  ⚠ all_trials 为空（所有 trial 均不满足阈值），回退到全部 trial 中寻找最优...")
        _all_study_trials = list(study_safe.trials) + list(study_maxcap.trials) + list(study_robust.trials)
        # 取所有有值的 trial（即使 value < -1e5）
        for trial in _all_study_trials:
            if trial.value is not None:
                all_trials.append(trial)
        print(f"    回退后共 {len(all_trials)} 个 trial")

    # 辅助函数：从 trial.params 提取买价（optimize_buy_price 模式下每个 trial 有自己的买价）
    def _trial_buy_price(trial_params):
        if optimize_buy_price and "buy_price" in trial_params:
            return trial_params["buy_price"], None
        return buy_price, buy_price_range

    unified = []
    for trial in all_trials:
        params = trial.params.copy()
        if params.get("conf_tier1_bound", 0) >= params.get("conf_tier2_bound", 1):
            continue
        t_bp, t_bpr = _trial_buy_price(params)
        res = simulate_trading(predictions, params, initial_capital,
                              buy_price=t_bp, buy_price_range=t_bpr)
        if res["n_trades"] < min_trades:
            continue
        entry = {**res, **params, "study_score": trial.value}
        unified.append(entry)

    unified_df = pd.DataFrame(unified)
    if unified_df.empty:
        print("  ⚠ 所有 trial 都不满足 min_trades 条件，放宽 min_trades=1")
        # 放宽条件重试 — 只要有任何交易就收
        for trial in all_trials:
            params = trial.params.copy()
            if params.get("conf_tier1_bound", 0) >= params.get("conf_tier2_bound", 1):
                continue
            t_bp, t_bpr = _trial_buy_price(params)
            res = simulate_trading(predictions, params, initial_capital,
                                  buy_price=t_bp, buy_price_range=t_bpr)
            if res["n_trades"] >= 1:
                entry = {**res, **params, "study_score": trial.value}
                unified.append(entry)
        unified_df = pd.DataFrame(unified)
        if unified_df.empty:
            print("  ⚠⚠ 完全无交易，生成保守默认参数（不交易）")
            fallback_conf = round((conf_low + conf_high) / 2.0, 4)
            fallback_t1 = min(0.99, fallback_conf + max(tier1_off_low, 0.01))
            fallback_t2 = min(0.995, max(fallback_t1 + 0.01, fallback_conf + max(tier2_off_low, 0.04)))
            # 创建一个默认不交易的条目
            default_params = {
                "min_confidence": fallback_conf, "min_edge": max(0.02, edge_low),
                "kelly_frac": 0.1, "bet_pct_normal": 0.02,
                "bet_pct_conservative": 0.01, "conf_tier1_bound": 0.54,
                "conf_tier2_bound": 0.56, "tier1_mult": 0.5,
                "tier2_mult": 0.7, "tier3_mult": 1.0,
                "cooldown_bars": 4, "drawdown_halt": 0.15,
                "reward_penalty_ratio": 1.0,
            }
            default_params["conf_tier1_bound"] = fallback_t1
            default_params["conf_tier2_bound"] = fallback_t2
            default_res = simulate_trading(predictions, default_params, initial_capital,
                                          buy_price=buy_price, buy_price_range=buy_price_range)
            entry = {**default_res, **default_params, "study_score": 0}
            unified_df = pd.DataFrame([entry])

    # 去重（参数几乎相同的）
    unified_df = unified_df.drop_duplicates(
        subset=["final_capital", "n_trades", "win_rate"], keep="first"
    )
    unified_df = unified_df.sort_values("final_capital", ascending=False).reset_index(drop=True)

    # 取 Top 100（按 final_capital）做 bootstrap
    top_trials_unified = unified_df.head(100)
    top_trial_dicts = top_trials_unified.to_dict("records")

    print(f"\n  Phase 2B: Bootstrap 验证 Top 100 ({n_bootstrap}x 重采样)...")
    t1 = time.time()

    param_keys = ["min_confidence", "min_edge", "kelly_frac", "bet_pct_normal",
                  "bet_pct_conservative", "conf_tier1_bound", "conf_tier2_bound",
                  "tier1_mult", "tier2_mult", "tier3_mult", "cooldown_bars", "drawdown_halt"]
    if optimize_buy_price:
        param_keys.append("buy_price")

    bootstrap_results = []
    for rank, rec in enumerate(top_trial_dicts):
        params = {k: rec[k] for k in param_keys if k in rec}
        if params.get("conf_tier1_bound", 0) >= params.get("conf_tier2_bound", 1):
            continue

        # 用每个 trial 自己的买价
        t_bp, t_bpr = _trial_buy_price(params)
        bs_res = simulate_bootstrap(predictions, params, n_bootstrap, initial_capital,
                                    buy_price=t_bp, buy_price_range=t_bpr)
        bs_res.update(params)
        bs_res["rank_capital"] = rank + 1
        bootstrap_results.append(bs_res)

        if (rank + 1) % 20 == 0:
            elapsed = time.time() - t1
            print(f"    {rank+1}/100 ({elapsed:.1f}s)...")

    elapsed_bs = time.time() - t1
    print(f"  完成: {elapsed_bs:.1f}s")

    # 用 bootstrap 中位数重新排名
    bs_df = pd.DataFrame(bootstrap_results)

    # 鲁棒得分 = bootstrap 中位收益 × (1-中位DD)^2 × 盈亏比 × 稳定性
    bs_df["robust_score"] = (
        np.maximum(bs_df["return_pct"] + 100, 1)
        * (1 - bs_df["max_drawdown"]) ** 2
        * np.minimum(bs_df["profit_factor"], 3.0) ** 0.5
        * np.minimum(bs_df["win_rate"], 0.99) ** 0.5
        * (1 - bs_df["bootstrap_std_capital"] / (bs_df["orig_final_capital"] + 1e-10)).clip(0.1, 1.0)
    )

    bs_df = bs_df.sort_values("robust_score", ascending=False).reset_index(drop=True)

    return study_safe, bs_df


def print_results(bs_df: pd.DataFrame, initial_capital: float = 400.0, buy_price: float = 0.527):
    """打印结果报表。"""
    odds = (1.0 - buy_price) / buy_price
    print("\n" + "=" * 110)
    print("  交易规则超参搜索 — Top 20 (Bootstrap 鲁棒排名)")
    print(f"  真实约束: 买价${buy_price:.3f} | 赔率{odds:.4f} | 手续费+滑点1.8% | <$60K按比例 | >=$60K固定$3000")
    print("=" * 110)

    # Top 20 表格
    print(f"\n{'Rk':<3} {'Score':>7} {'Final$':>8} {'Ret%':>8} {'WR':>6} {'Shrp':>5} "
          f"{'MaxDD':>6} {'PF':>5} {'#Tr':>5} {'T/d':>5} "
          f"{'Conf':>5} {'Edge':>5} {'Kly':>5} {'Nrm%':>5} {'Csv%':>5} "
          f"{'T1M':>4} {'T2M':>4} {'T3M':>4} "
          f"{'CD':>2} {'DDH':>4}")
    print("─" * 120)

    for idx, row in bs_df.head(20).iterrows():
        print(f"{idx+1:<3} {row['robust_score']:>7.1f} "
              f"${row['orig_final_capital']:>7.0f} {row['orig_return_pct']:>7.1f}% "
              f"{row['win_rate']:>5.1%} {row['sharpe']:>5.1f} "
              f"{row['max_drawdown']:>5.1%} {row['profit_factor']:>5.2f} "
              f"{int(row['n_trades']):>5} {row['trades_per_day']:>5.1f} "
              f"{row['min_confidence']:>5.3f} {row['min_edge']:>5.3f} "
              f"{row['kelly_frac']:>5.2f} "
              f"{row['bet_pct_normal']:>5.2f} {row['bet_pct_conservative']:>5.2f} "
              f"{row['tier1_mult']:>4.2f} {row['tier2_mult']:>4.2f} "
              f"{row['tier3_mult']:>4.2f} "
              f"{int(row['cooldown_bars']):>2} {row['drawdown_halt']:>4.2f}")

    # 最优配置详情
    best = bs_df.iloc[0]
    print(f"\n{'═' * 110}")
    print(f"  ★ 综合最优配置 (Robust Score)")
    print(f"{'═' * 110}")
    print(f"\n  ┌─ 交易门槛 ──────────────────────────────────────────────────────┐")
    print(f"  │  min_confidence:       {best['min_confidence']:.4f}   (最低置信度)                │")
    print(f"  │  min_edge:             {best['min_edge']:.4f}   (扣成本后最小净边际)        │")
    print(f"  └──────────────────────────────────────────────────────────────────┘")
    print(f"\n  ┌─ 仓位管理（Polymarket 真实）──────────────────────────────────────┐")
    print(f"  │  kelly_frac:           {best['kelly_frac']:.4f}   (Kelly 乘数)                │")
    print(f"  │  bet_pct_normal:       {best['bet_pct_normal']:.4f}   (Phase1: 资金<$60K, 仓位%)   │")
    print(f"  │  bet_pct_conservative: {best['bet_pct_conservative']:.4f}   (Phase3: 跌回<$60K, 仓位%)  │")
    print(f"  │  Phase2 固定:          $3,000  (资金≥$60K, 流动性限制)     │")
    print(f"  │  买入价:               ${buy_price:.3f}  (赔率 {odds:.4f})              │")
    print(f"  └──────────────────────────────────────────────────────────────────┘")
    print(f"\n  ┌─ 置信度分档 ────────────────────────────────────────────────────┐")
    print(f"  │  低档: [{best['min_confidence']:.3f}, {best['conf_tier1_bound']:.3f})  × {best['tier1_mult']:.2f}                       │")
    print(f"  │  中档: [{best['conf_tier1_bound']:.3f}, {best['conf_tier2_bound']:.3f})  × {best['tier2_mult']:.2f}                       │")
    print(f"  │  高档: [{best['conf_tier2_bound']:.3f}, 1.000)  × {best['tier3_mult']:.2f}                       │")
    print(f"  └──────────────────────────────────────────────────────────────────┘")
    print(f"\n  ┌─ 风控 ─────────────────────────────────────────────────────────┐")
    print(f"  │  cooldown_bars:        {int(best['cooldown_bars'])}       (亏损后暂停 bar 数)          │")
    print(f"  │  drawdown_halt:        {best['drawdown_halt']:.2f}    (回撤熔断线 → 停止交易)      │")
    print(f"  └──────────────────────────────────────────────────────────────────┘")
    print(f"\n  预期绩效 (15天测试期):")
    print(f"    初始资金:        ${initial_capital:,.0f}")
    print(f"    最终资金 (原始):  ${best['orig_final_capital']:,.2f}")
    print(f"    最终资金 (中位):  ${best['final_capital']:,.2f}")
    print(f"    收益率 (原始):    {best['orig_return_pct']:+.2f}%")
    print(f"    胜率 (中位):      {best['win_rate']:.1%}")
    print(f"    Sharpe (中位):    {best['sharpe']:.1f}")
    print(f"    最大回撤 (中位):  {best['max_drawdown']:.1%}")
    print(f"    利润因子:         {best['profit_factor']:.2f}")
    print(f"    交易笔数:         {int(best['n_trades'])} ({best['trades_per_day']:.1f}/天)")
    print(f"    Bootstrap 波动:   ±${best['bootstrap_std_capital']:.2f}")

    # 不同风格
    print(f"\n{'─' * 110}")
    print(f"  各风格最优:")

    def _print_style(label, row):
        print(f"    conf={row['min_confidence']:.3f} edge={row['min_edge']:.3f} "
              f"kelly={row['kelly_frac']:.2f} nrm={row['bet_pct_normal']:.2f} "
              f"csv={row['bet_pct_conservative']:.2f} "
              f"cd={int(row['cooldown_bars'])} ddh={row['drawdown_halt']:.2f}")

    max_cap_row = bs_df.loc[bs_df["orig_final_capital"].idxmax()]
    print(f"\n  【最高资金】 ${max_cap_row['orig_final_capital']:,.2f} "
          f"(ret {max_cap_row['orig_return_pct']:+.1f}%, DD {max_cap_row['max_drawdown']:.1%}, "
          f"WR {max_cap_row['win_rate']:.1%}, PF {max_cap_row['profit_factor']:.2f})")
    _print_style("最高资金", max_cap_row)

    high_trade = bs_df[bs_df["n_trades"] >= 30]
    if len(high_trade) > 0:
        max_wr_row = high_trade.loc[high_trade["win_rate"].idxmax()]
        print(f"\n  【最高胜率】 WR {max_wr_row['win_rate']:.1%} "
              f"(${max_wr_row['orig_final_capital']:,.2f}, DD {max_wr_row['max_drawdown']:.1%}, "
              f"{int(max_wr_row['n_trades'])} trades)")
        _print_style("最高胜率", max_wr_row)

    safe = bs_df[(bs_df["n_trades"] >= 20) & (bs_df["orig_return_pct"] > 0)]
    if len(safe) > 0:
        min_dd_row = safe.loc[safe["max_drawdown"].idxmin()]
        print(f"\n  【最低回撤】 DD {min_dd_row['max_drawdown']:.1%} "
              f"(${min_dd_row['orig_final_capital']:,.2f}, WR {min_dd_row['win_rate']:.1%})")
        _print_style("最低回撤", min_dd_row)

    sharp_enough = bs_df[bs_df["n_trades"] >= 30]
    if len(sharp_enough) > 0:
        max_sh_row = sharp_enough.loc[sharp_enough["sharpe"].idxmax()]
        print(f"\n  【最高Sharpe】 {max_sh_row['sharpe']:.1f} "
              f"(${max_sh_row['orig_final_capital']:,.2f}, WR {max_sh_row['win_rate']:.1%}, "
              f"DD {max_sh_row['max_drawdown']:.1%})")
        _print_style("最高Sharpe", max_sh_row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10000,
                        help="Optuna 搜索 trial 数（默认 10000）")
    parser.add_argument("--bootstrap", type=int, default=30,
                        help="Bootstrap 重采样次数（默认 30）")
    parser.add_argument("--capital", type=float, default=400.0,
                        help="初始资金（默认 $400）")
    parser.add_argument("--buy-price", type=float, default=0.527,
                        help="买入价（默认 $0.527 市价单；限价单用 0.51 等）")
    parser.add_argument("--buy-price-range", type=str, default=None,
                        help="动态买价范围，如 '0.50,0.54'。每笔交易随机取该范围内的价格，"
                             "模拟真实限价单场景。指定后 --buy-price 作为输出标签使用。")
    parser.add_argument("--min-trades", type=int, default=100,
                        help="最少交易笔数（默认 100）。过滤掉交易太少的脆弱配置。")
    parser.add_argument("--model-dir", type=str, default="v5_production",
                        help="模型目录名（在 data/models/ 下），默认 v5_production")
    parser.add_argument("--sim-noise", action="store_true",
                        help="对测试特征应用 T-120s 模拟噪声（sim_noise 模型用）。"
                             "噪声加在 close/high/low/volume 上, 标签保持真实方向。")
    parser.add_argument("--optimize-buy-price", action="store_true",
                        help="将买入价(buy_price)作为第 13 个 Optuna 可优化参数，"
                             "与其他 12 个交易规则参数联合优化。搜索范围 [0.44, 0.54]。"
                             "启用后 --buy-price 仅作为标签使用。")
    parser.add_argument("--buy-price-range-search", type=str, default="0.44,0.54",
                        help="买价搜索范围（仅 --optimize-buy-price 时生效），"
                             "如 '0.44,0.54'。默认 [0.44, 0.54]。")
    parser.add_argument("--test-days", type=int, default=0,
                        help="预测测试窗口天数（0=沿用默认常量；可设 120/180/365）")
    parser.add_argument("--assets", type=str, default="",
                        help="仅优化这些资产，逗号分隔，如 ETH_USDT 或 BTC_USDT,ETH_USDT")
    parser.add_argument("--min-confidence-range", type=str, default="0.50,0.56",
                        help="min_confidence 搜索范围，如 '0.60,0.70'")
    parser.add_argument("--min-edge-range", type=str, default="0.00,0.06",
                        help="min_edge 搜索范围，如 '0.00,0.08'")
    parser.add_argument("--tier1-offset-range", type=str, default="0.005,0.04",
                        help="conf_tier1_bound 相对 min_confidence 的偏移范围，如 '0.01,0.05'")
    parser.add_argument("--tier2-offset-range", type=str, default="0.03,0.10",
                        help="conf_tier2_bound 相对 min_confidence 的偏移范围，如 '0.04,0.14'")
    args = parser.parse_args()

    # ── 模型目录 + 输出路径 ──
    model_dir = PROJECT_ROOT / "data" / "models" / args.model_dir
    if not model_dir.exists():
        print(f"❌ 模型目录不存在: {model_dir}")
        sys.exit(1)
    model_name = args.model_dir

    # 模型专属输出子目录（v5_production 用默认路径, 其他模型用子目录）
    if model_name == "v5_production":
        model_output_dir = OUTPUT_DIR
    else:
        model_output_dir = OUTPUT_DIR / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    buy_price = args.buy_price
    odds = (1.0 - buy_price) / buy_price
    be = 1 / (1 + odds) + 0.0025

    # 解析动态买价范围
    buy_price_range = None
    if args.buy_price_range:
        parts = args.buy_price_range.split(",")
        buy_price_range = (float(parts[0].strip()), float(parts[1].strip()))
        # 用范围中值作为参考
        mid_price = (buy_price_range[0] + buy_price_range[1]) / 2
        odds_lo = (1.0 - buy_price_range[0]) / buy_price_range[0]
        odds_hi = (1.0 - buy_price_range[1]) / buy_price_range[1]

    # 解析买价搜索范围
    buy_price_search_range = (0.44, 0.54)
    if args.optimize_buy_price and args.buy_price_range_search:
        parts = args.buy_price_range_search.split(",")
        buy_price_search_range = (float(parts[0].strip()), float(parts[1].strip()))

    n_params = 13 if args.optimize_buy_price else 12

    def _parse_range(raw: str) -> Tuple[float, float]:
        parts = [p.strip() for p in str(raw).split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError(f"范围参数格式错误: {raw}")
        lo = float(parts[0])
        hi = float(parts[1])
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi

    min_conf_range = _parse_range(args.min_confidence_range)
    min_edge_range = _parse_range(args.min_edge_range)
    tier1_offset_range = _parse_range(args.tier1_offset_range)
    tier2_offset_range = _parse_range(args.tier2_offset_range)

    print("=" * 70)
    print(f"  交易规则超参优化 v3 — Polymarket 真实约束版")
    print(f"  模型: {model_name}" + (" + T-120s 噪声特征" if args.sim_noise else " (完美 T+0 特征)"))
    print(f"  方法: Optuna TPE 贝叶斯搜索")
    print(f"  Trials: {args.trials}")
    print(f"  Bootstrap: {args.bootstrap}x 重采样")
    print(f"  初始资金: ${args.capital:,.0f}")
    if args.optimize_buy_price:
        print(f"  买入价: 🔍 可优化参数 [{buy_price_search_range[0]:.2f}, {buy_price_search_range[1]:.2f}]")
    elif buy_price_range:
        print(f"  买入价: 动态范围 ${buy_price_range[0]:.3f}~${buy_price_range[1]:.3f}")
        print(f"          赔率范围 {odds_hi:.4f}~{odds_lo:.4f}")
        print(f"          每笔交易随机一个买价（模拟真实限价单）")
    else:
        print(f"  买入价: ${buy_price:.3f} (赔率 {odds:.4f}, 盈亏平衡 {be:.2%})")
    print(f"  手续费+滑点: 1.8% (taker ~1.5% + 滑点 0.3%)")
    print(f"  仓位: <$60K 按比例(2-10%) | ≥$60K 固定$3000")
    print(f"  min_conf 搜索: [{min_conf_range[0]:.3f}, {min_conf_range[1]:.3f}]")
    print(f"  min_edge 搜索: [{min_edge_range[0]:.3f}, {min_edge_range[1]:.3f}]")
    print(f"  搜索参数: {n_params} 个（连续空间" + ("，含买价" if args.optimize_buy_price else "") + "）")
    print("=" * 70)

    effective_test_days = int(args.test_days) if int(args.test_days) > 0 else int(TEST_DAYS)
    selected_assets = [x.strip().upper() for x in str(args.assets).split(",") if x.strip()]

    # 1. 生成预测（如果已有缓存则跳过）
    # 将 test_days 编入缓存名，避免“窗口变了但还复用旧缓存”
    td_suffix = f"td{effective_test_days}d"
    if args.sim_noise:
        pred_path = OUTPUT_DIR / f"{model_name}_noisy_test_predictions_{td_suffix}.parquet"
    else:
        pred_path = OUTPUT_DIR / f"{model_name}_test_predictions_{td_suffix}.parquet"

    # 向后兼容: 仅在默认窗口时允许回退到旧缓存名
    if not pred_path.exists() and effective_test_days == int(TEST_DAYS):
        if args.sim_noise:
            legacy = OUTPUT_DIR / f"{model_name}_noisy_test_predictions.parquet"
            if legacy.exists():
                pred_path = legacy
        else:
            legacy = OUTPUT_DIR / f"{model_name}_test_predictions.parquet"
            if legacy.exists():
                pred_path = legacy
            elif model_name == "v5_production":
                old_path = OUTPUT_DIR / "v5_test_predictions.parquet"
                if old_path.exists():
                    pred_path = old_path

    if pred_path.exists():
        print(f"\n加载已缓存预测: {pred_path}")
        predictions = pd.read_parquet(pred_path)
        print(f"  {len(predictions)} bars, {predictions['asset'].nunique()} 币")
    else:
        predictions = generate_predictions(
            model_dir,
            apply_noise=args.sim_noise,
            test_days=effective_test_days,
        )
        predictions.to_parquet(pred_path, index=False)
        print(f"预测已保存: {pred_path}")

    if selected_assets:
        predictions = predictions[predictions["asset"].astype(str).str.upper().isin(selected_assets)].copy()
        if predictions.empty:
            print(f"❌ 资产过滤后无预测数据: {selected_assets}")
            sys.exit(1)
        print(f"  资产过滤: {selected_assets} -> {len(predictions)} bars, {predictions['asset'].nunique()} 币")

    # 1.5 预热 Numba JIT（首次编译 ~2s，后续用磁盘缓存）
    print("\n预热 Numba JIT...")
    t_jit = time.time()
    _warmup_numba()
    print(f"  JIT 就绪: {time.time() - t_jit:.1f}s")

    # 2. Optuna 搜索 + Bootstrap
    study, bs_df = run_optuna_search(
        predictions, args.trials, args.bootstrap, args.capital,
        buy_price=buy_price, buy_price_range=buy_price_range,
        min_trades=args.min_trades,
        optimize_buy_price=args.optimize_buy_price,
        buy_price_search_range=buy_price_search_range,
        min_conf_range=min_conf_range,
        min_edge_range=min_edge_range,
        tier1_offset_range=tier1_offset_range,
        tier2_offset_range=tier2_offset_range,
    )

    # 3. 打印结果
    print_results(bs_df, args.capital, buy_price=buy_price)

    # 4. 保存
    best = bs_df.iloc[0]
    # 买价标签
    if args.optimize_buy_price and "buy_price" in best.index:
        optimized_bp = round(float(best["buy_price"]), 3)
        price_tag = f"bp_opt_{optimized_bp:.3f}".replace(".", "")
    elif buy_price_range:
        optimized_bp = buy_price
        price_tag = f"bp_dyn_{buy_price_range[0]:.3f}_{buy_price_range[1]:.3f}".replace(".", "")
    else:
        optimized_bp = buy_price
        price_tag = f"bp{buy_price:.3f}".replace(".", "")
    csv_path = model_output_dir / f"trading_rules_v2_{price_tag}_{ts}.csv"
    bs_df.to_csv(csv_path, index=False)
    print(f"\n完整结果: {csv_path}")

    # 保存最优配置
    best_config = {
        "trading_rules": {
            "min_confidence":      round(float(best["min_confidence"]), 5),
            "min_edge":            round(float(best["min_edge"]), 5),
            "kelly_frac":          round(float(best["kelly_frac"]), 5),
            "bet_pct_normal":      round(float(best["bet_pct_normal"]), 5),
            "bet_pct_conservative": round(float(best["bet_pct_conservative"]), 5),
            "confidence_tiers": [
                [round(float(best["min_confidence"]), 4),
                 round(float(best["conf_tier1_bound"]), 4),
                 round(float(best["tier1_mult"]), 4)],
                [round(float(best["conf_tier1_bound"]), 4),
                 round(float(best["conf_tier2_bound"]), 4),
                 round(float(best["tier2_mult"]), 4)],
                [round(float(best["conf_tier2_bound"]), 4),
                 1.0,
                 round(float(best["tier3_mult"]), 4)],
            ],
            "cooldown_bars":       int(best["cooldown_bars"]),
            "drawdown_halt":       round(float(best["drawdown_halt"]), 4),
        },
        "polymarket_constraints": {
            "buy_price": optimized_bp,
            "buy_price_optimized": args.optimize_buy_price,
            "buy_price_search_range": list(buy_price_search_range) if args.optimize_buy_price else None,
            "buy_price_range": list(buy_price_range) if buy_price_range else None,
            "odds": round((1.0 - optimized_bp) / optimized_bp, 4),
            "fee_rate": 0.015,
            "slippage_rate": 0.003,
            "liquidity_cap": 60000,
            "liquidity_bet": 3000,
        },
        "metrics_bootstrap_median": {
            "final_capital":    round(float(best["final_capital"]), 2),
            "return_pct":       round(float(best["return_pct"]), 2),
            "win_rate":         round(float(best["win_rate"]), 4),
            "sharpe":           round(float(best["sharpe"]), 2),
            "max_drawdown":     round(float(best["max_drawdown"]), 4),
            "profit_factor":    round(float(best["profit_factor"]), 3),
            "n_trades":         int(best["n_trades"]),
        },
        "metrics_original_sequence": {
            "final_capital":    round(float(best["orig_final_capital"]), 2),
            "return_pct":       round(float(best["orig_return_pct"]), 2),
            "win_rate":         round(float(best["orig_win_rate"]), 4),
        },
        "search_config": {
            "method": "Optuna TPE (Bayesian)",
            "n_trials": args.trials,
            "n_bootstrap": args.bootstrap,
            "initial_capital": args.capital,
            "test_days": effective_test_days,
            "assets": selected_assets or sorted(predictions["asset"].astype(str).unique().tolist()),
        },
        "optimized_at": ts,
    }
    config_path = model_output_dir / f"optimal_trading_rules_v3_{price_tag}.json"
    with open(config_path, "w") as f:
        json.dump(best_config, f, indent=2, ensure_ascii=False)
    print(f"最优配置: {config_path}")

    print(f"\n{'═' * 70}")
    if args.optimize_buy_price:
        print(f"  完成！模型 {model_name} / 买价联合优化 → 最优买价 ${optimized_bp:.3f}")
    else:
        print(f"  完成！模型 {model_name} / 买价 ${buy_price:.3f} 的最优参数已保存")
    print(f"  文件: {config_path}")
    print(f"  下一步: 用最优参数回测对比")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
