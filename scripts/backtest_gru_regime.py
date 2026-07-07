#!/usr/bin/env python3
"""
GRU Regime 最佳模型回测：使用 outputs/models_best 下的 encoder + lightgbm_with_embedding，
对 BTC/ETH/XRP 15m 做一年回测，支持多组初始资金（如 400、1000）。

在项目根目录执行:
  python scripts/backtest_gru_regime.py
  python scripts/backtest_gru_regime.py --test-days 365 --capitals 400 1000
"""
import sys
import json
import argparse
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import torch

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = PROJECT_ROOT / "experiments" / "gru_regime_v1"
import os
_MODELS_BEST_ENV = os.environ.get("GRU_MODELS_BEST")
MODELS_BEST = Path(_MODELS_BEST_ENV) if _MODELS_BEST_ENV else (EXPERIMENT_ROOT / "outputs" / "models_best")
sys.path.insert(0, str(PROJECT_ROOT))

# 项目侧
from src.python.data_fetcher import load_ohlcv
from src.python.feature_engineering import build_features, add_multi_timeframe_features

# gru_regime 侧（需 experiments/__init__.py 与 experiments/gru_regime_v1/__init__.py）
from experiments.gru_regime_v1.src.data_loader import load_parquet_data, compute_features, create_sliding_windows
from experiments.gru_regime_v1.src.features import ZScoreNormalizer
from experiments.gru_regime_v1.src.gru_encoder import GRUVolatilityPredictor, create_model
from experiments.gru_regime_v1.src.train_lightgbm import merge_embeddings, get_feature_columns
from experiments.gru_regime_v1.src.utils import load_config, get_device

# Polymarket 输赢规则：买 Yes 价格 P（order_price），下注 B 美元 → 买到 B/P 份。
# 赢：每份结算 1 美元，得 B/P 美元 → 净赚 B*(1/P - 1) = B*(1-P)/P（例如 P=0.53 时赚 B*0.47/0.53 ≈ B*0.887）。
# 输：得 0（或止损时按 stop_loss_exit_price 回收部分），capital 已扣 bet+fee。
# 下注额（use_smart_bet）：每笔前看当前 capital；≥6万固定3000，<6万按当前资金5%；输下6万后下一笔即按5%。
# 止损：stop_loss_exit_price 为败局时按该价卖出 UP 的近似（当前为 15m 一根 K，无周期内价格）。
# 反转：前 14 分钟 UP 价 < 止损价（如 0.07）、15m 收盘为 up → 若已止损则算败，需 Polymarket 1m 数据。
# 1m 数据仅用 Polymarket CLOB prices-history：https://docs.polymarket.com/developers/CLOB/timeseries
# 需先运行 scripts/download_polymarket_1m.py 生成 data/polymarket_1m_{asset}.parquet（列 15m_start_ts, t_sec, p）。
MIN_BET = 1.0
CLOB_PRICES_HISTORY = "https://clob.polymarket.com/prices-history"
REVERSAL_STOP_THRESHOLD = 0.07  # 前 14 分钟 UP 价低于此视为触发止损
DEFAULT_FEE_RATE = 0.015  # Polymarket taker 手续费 ~1.5% (2026-01-19 起)
DEFAULT_SLIPPAGE = 0.003  # 滑点 ~0.3%
DEFAULT_MAX_TRADES_PER_DAY = 20
DEFAULT_ORDER_PRICE = 0.503
DEFAULT_INITIAL_CAPITAL = 60000.0
FIXED_BET_AMOUNT = 3000.0  # 固定每笔 3000；有动态时按回撤减为 2250/1500

# 高波动过滤（方式 A：自适应阈值 + 双指标 OR + 保守 A）
HIGH_VOL_ATR_N_DAYS = 30
HIGH_VOL_ATR_QUANTILE = 0.9
HIGH_VOL_VOL_RATIO_THRESHOLD = 1.8
BARS_PER_DAY_15M = 96

ASSETS = ["BTC_USDT", "ETH_USDT", "XRP_USDT", "SOL_USDT"]
TIMEFRAME = "15m"
SYMBOL_MAP = {"BTC_USDT": "BTC/USDT", "ETH_USDT": "ETH/USDT", "XRP_USDT": "XRP/USDT", "SOL_USDT": "SOL/USDT"}


def get_no_leak_start_date(data_src: Path, config_path: Optional[Path] = None, asset: str = "BTC_USDT") -> str:
    """
    按训练时 LightGBM 的时序划分比例，从 parquet 算出「测试集起始日」= 无泄露回测的起始日。
    与 experiments/gru_regime_v1 中 config lightgbm.time_series_cv 一致：train+val 后的第一行即测试集起点。
    返回 YYYY-MM-DD，供 --after-date 使用；回测取该日起 365 天即无泄露一年。
    """
    if config_path is None:
        config_path = EXPERIMENT_ROOT / "configs" / "config.json"
    config = load_config(str(config_path))
    train_pct = config["lightgbm"]["time_series_cv"]["train_window_pct"]
    val_pct = config["lightgbm"]["time_series_cv"]["val_window_pct"]
    # 测试集 = 最后 test_pct，起始行索引 = (train_pct + val_pct) * n
    test_start_ratio = train_pct + val_pct
    freq_min = 15
    filepath = data_src / f"{asset.lower()}_{freq_min}m.parquet"
    if not filepath.exists():
        raise FileNotFoundError(f"用于计算无泄露起始日的 parquet 不存在: {filepath}")
    df = pd.read_parquet(filepath)
    if "timestamp" not in df.columns:
        raise ValueError("parquet 需含 timestamp 列")
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    if n == 0:
        raise ValueError("parquet 为空")
    test_start_idx = int(n * test_start_ratio)
    test_start_idx = min(test_start_idx, n - 1)
    ts = df["timestamp"].iloc[test_start_idx]
    if hasattr(ts, "item"):
        ts = ts.item()
    if isinstance(ts, (int, np.integer)):
        dt = pd.to_datetime(ts, unit="ms", utc=True)
    else:
        dt = pd.to_datetime(ts, utc=True)
    return dt.strftime("%Y-%m-%d")


def add_high_vol_skip_column(
    merged: pd.DataFrame,
    atr_n_days: Optional[int] = None,
    atr_quantile: Optional[float] = None,
    vol_ratio_threshold: Optional[float] = None,
    combine: str = "or",
) -> pd.DataFrame:
    """
    在 merged 上增加 high_vol_skip：自适应阈值 + 双指标 + 组合方式（OR/AND）。
    atr_n_days/atr_quantile/vol_ratio_threshold 为 None 时使用模块常量。
    combine: "or" = 满足其一即跳过, "and" = 两条件都满足才跳过（更保守）。
    high_vol_skip=True 的 bar 在 run_trading_loop(enable_high_vol_filter=True) 时会被跳过。
    """
    if "atr_pct_14" not in merged.columns or "vol_ratio_4vs16" not in merged.columns:
        merged["high_vol_skip"] = False
        return merged
    n_days = atr_n_days if atr_n_days is not None else HIGH_VOL_ATR_N_DAYS
    q = atr_quantile if atr_quantile is not None else HIGH_VOL_ATR_QUANTILE
    vol_th = vol_ratio_threshold if vol_ratio_threshold is not None else HIGH_VOL_VOL_RATIO_THRESHOLD
    roll_bars = n_days * BARS_PER_DAY_15M
    atr = merged["atr_pct_14"].replace([np.inf, -np.inf], np.nan)
    roll_q = atr.rolling(roll_bars, min_periods=BARS_PER_DAY_15M).quantile(q).shift(1)
    atr_high = (merged["atr_pct_14"].fillna(0) >= roll_q.fillna(np.inf))
    vol_ratio = merged["vol_ratio_4vs16"].replace([np.inf, -np.inf], np.nan)
    vol_high = (vol_ratio.fillna(0) >= vol_th)
    if combine == "and":
        merged["high_vol_skip"] = (atr_high & vol_high).fillna(False)
    else:
        merged["high_vol_skip"] = (atr_high | vol_high).fillna(False)
    return merged


def get_parquet_end_date(data_src: Path, asset: str = "BTC_USDT") -> str:
    """取 parquet 最后一行日期 YYYY-MM-DD，用于无泄漏区间「用满到数据末尾」。"""
    freq_min = 15
    filepath = data_src / f"{asset.lower()}_{freq_min}m.parquet"
    if not filepath.exists():
        raise FileNotFoundError(f"parquet 不存在: {filepath}")
    df = pd.read_parquet(filepath)
    if "timestamp" not in df.columns or len(df) == 0:
        raise ValueError("parquet 需含 timestamp 且非空")
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts = df["timestamp"].iloc[-1]
    if hasattr(ts, "item"):
        ts = ts.item()
    if isinstance(ts, (int, np.integer)):
        dt = pd.to_datetime(ts, unit="ms", utc=True)
    else:
        dt = pd.to_datetime(ts, utc=True)
    return dt.strftime("%Y-%m-%d")


def load_encoder_and_normalizer(
    asset: str, device: torch.device, models_best_override: Optional[Path] = None
) -> Tuple[GRUVolatilityPredictor, ZScoreNormalizer, Dict]:
    """加载 encoder.pt、encoder_config.json、normalizer。models_best_override 可指定模型目录（如无1h4h）。"""
    models_root = models_best_override if models_best_override is not None else MODELS_BEST
    asset_dir = models_root / asset
    if not asset_dir.exists():
        raise FileNotFoundError(f"models_best 目录不存在: {asset_dir}")
    pt_path = asset_dir / "encoder.pt"
    config_path = asset_dir / "encoder_config.json"
    normalizer_path = EXPERIMENT_ROOT / "data" / f"{asset}_normalizer.json"
    if not pt_path.exists() or not config_path.exists():
        raise FileNotFoundError(f"缺少 encoder 或 config: {asset_dir}")
    if not normalizer_path.exists():
        raise FileNotFoundError(f"缺少 normalizer: {normalizer_path}")

    train_config = load_config(str(config_path))
    hyperparams = train_config["hyperparams"]
    input_dim = train_config["input_dim"]

    model = create_model(
        input_dim=input_dim,
        hidden_size=hyperparams["hidden_size"],
        embedding_dim=hyperparams["embedding_dim"],
        num_layers=hyperparams.get("num_layers", 1),
        dropout=0.0,
        device=device,
    )
    ckpt = torch.load(pt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    normalizer = ZScoreNormalizer.load(str(normalizer_path))
    return model, normalizer, train_config


def build_embeddings_for_backtest(
    asset: str,
    df_raw: pd.DataFrame,
    encoder: GRUVolatilityPredictor,
    normalizer: ZScoreNormalizer,
    train_config: Dict,
    device: torch.device,
    batch_size: int = 256,
) -> pd.DataFrame:
    """对 df_raw 构建 GRU 输入、滑窗、归一化、取 embedding，返回 (timestamp, emb_0..emb_n) 的 DataFrame。"""
    feature_cols = train_config["feature_cols"]
    lookback = train_config["hyperparams"]["lookback"]

    df = df_raw.copy()
    df = compute_features(df)
    df["volatility_label"] = 0.0
    X, _, timestamps = create_sliding_windows(df, lookback, feature_cols)
    X = normalizer.transform_array(X)

    num_samples = len(X)
    embedding_dim = encoder.get_embedding_dim()
    embeddings = np.zeros((num_samples, embedding_dim), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            end = min(i + batch_size, num_samples)
            batch_x = torch.FloatTensor(X[i:end]).to(device)
            embeddings[i:end] = encoder.get_embedding(batch_x).cpu().numpy()

    # 时间戳：create_sliding_windows 返回的可能是 int64 ms 或 numpy
    ts = timestamps if hasattr(timestamps, "__len__") else np.arange(num_samples)
    if hasattr(ts, "dtype") and np.issubdtype(ts.dtype, np.datetime64):
        ts_ms = pd.to_datetime(ts).astype("int64") // 10**6
    else:
        ts_ms = np.asarray(ts, dtype=np.int64)

    out = {"timestamp": ts_ms}
    for j in range(embedding_dim):
        out[f"emb_{j}"] = embeddings[:, j]
    return pd.DataFrame(out)


def fetch_polymarket_prices_history(
    token_id: str,
    start_ts_sec: int,
    end_ts_sec: int,
    fidelity_min: int = 1,
) -> List[Dict[str, float]]:
    """
    从 Polymarket CLOB 拉取 token 的 1m 价格历史。
    GET https://clob.polymarket.com/prices-history?market=<token_id>&startTs=&endTs=&fidelity=1
    见 https://docs.polymarket.com/developers/CLOB/timeseries
    返回 [{"t": unix_sec, "p": price}, ...]，失败返回 []。
    """
    try:
        import urllib.request
        url = f"{CLOB_PRICES_HISTORY}?market={token_id}&startTs={start_ts_sec}&endTs={end_ts_sec}&fidelity={fidelity_min}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
        return data.get("history") or []
    except Exception:
        return []


def compute_reversal_from_polymarket_1m(
    merged_15m: pd.DataFrame,
    df_1m: pd.DataFrame,
    stop_threshold: float = REVERSAL_STOP_THRESHOLD,
) -> pd.DataFrame:
    """
    用 Polymarket CLOB 1m 价格判断「前 14 分钟 UP 价 < 止损价、15m 收盘为 up」的反转条。
    df_1m 须含列 15m_start_ts, t_sec, p（由 download_polymarket_1m.py 生成）。
    """
    if df_1m.empty or merged_15m.empty:
        out = merged_15m.copy()
        out["reversal"] = False
        return out
    if "15m_start_ts" not in df_1m.columns or "t_sec" not in df_1m.columns or "p" not in df_1m.columns:
        out = merged_15m.copy()
        out["reversal"] = False
        return out
    out = merged_15m.copy()
    rev = []
    ts_15_sec = (out["timestamp"].values // 1000).astype(int)
    actual_up = out["actual_up"].values
    window_end = 14 * 60
    for i in range(len(out)):
        t0 = ts_15_sec[i]
        sub = df_1m[(df_1m["15m_start_ts"] == t0) & (df_1m["t_sec"] >= t0) & (df_1m["t_sec"] < t0 + window_end)]
        if sub.empty:
            rev.append(False)
            continue
        min_p = sub["p"].min()
        rev.append((min_p < stop_threshold) and (actual_up[i] == 1))
    out["reversal"] = rev
    return out


def load_backtest_data_original_features(
    symbol: str,
    test_days: int,
    data_src: Path,
    after_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    加载回测期 15m 数据并构建项目侧原始特征。
    after_date 若指定（YYYY-MM-DD），则仅使用该日期之后的数据（防泄露）。
    end_date 若同时指定，则只取 (after_date, end_date] 区间，用于固定 365 天窗口。
    否则 after_date 时取到 now；无 after_date 时使用最近 test_days 天。
    """
    df = load_ohlcv(symbol, TIMEFRAME)
    if df.empty or len(df) < 100:
        return pd.DataFrame()
    if "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    now = pd.Timestamp.now(tz="UTC")
    if after_date:
        cutoff = pd.Timestamp(after_date, tz="UTC")
        df = df[df["date"] > cutoff].copy().reset_index(drop=True)
        if end_date:
            end_ts = pd.Timestamp(end_date, tz="UTC")
            df = df[df["date"] <= end_ts].copy().reset_index(drop=True)
        else:
            df = df[df["date"] <= now].copy().reset_index(drop=True)
    else:
        cutoff = now - pd.Timedelta(days=test_days)
        df = df[df["date"] >= cutoff].copy().reset_index(drop=True)
    if len(df) < 100:
        return pd.DataFrame()

    df = build_features(df)
    if os.environ.get("GRU_SKIP_MTF") != "1":
        symbol_lower = symbol.replace("/", "_").lower()
        try:
            df = add_multi_timeframe_features(df, symbol_lower)
        except Exception:
            pass
    if "timestamp" not in df.columns and "date" in df.columns:
        df["timestamp"] = df["date"].astype("int64") // 10**6
    return df


def run_trading_loop(
    df: pd.DataFrame,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    bet_ratio: float = 0.05,
    prob_threshold: float = 0.55,
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage: float = DEFAULT_SLIPPAGE,
    max_trades_per_day: int = DEFAULT_MAX_TRADES_PER_DAY,
    enable_dynamic_bet_ratio: bool = False,
    order_price: float = DEFAULT_ORDER_PRICE,
    stop_loss_exit_price: Optional[float] = None,
    use_fixed_bet: bool = True,
    use_smart_bet: bool = False,
    smart_bet_capital_threshold: float = 60000.0,
    enable_high_vol_filter: bool = False,
    high_vol_bet_scale: Optional[float] = None,
) -> Dict[str, Any]:
    """
    下注方式（优先级）：use_smart_bet > use_fixed_bet > 比例。
    - use_smart_bet=True：贴近实盘，资金>threshold 固定 3000，否则 总资金×bet_ratio（如 5%%）。
    - use_fixed_bet=True（默认）：固定每笔 3000；有动态时 >20%% 用 1500，>10%% 用 2250。
    - use_fixed_bet=False：每笔 = 当前总资金 × bet_ratio；有动态时比例按回撤缩小。
    - stop_loss_exit_price：败局时按该价卖出 UP 回收（如 0.07）；不设则败局全损。买价 order_price 下注 B 得 B/order_price 份，败局回收 (B/order_price)*stop_loss_exit_price。
    - high_vol_bet_scale：高波 bar 下注比例，None 或 0=不交易，>0=少交易（bet_amount *= scale）。
    """
    capital = float(initial_capital)
    order_price = float(order_price)
    equity_curve = [capital]
    wins = 0
    losses = 0
    reversal_count = 0
    peak_capital = capital
    use_reversal = False  # 已去掉止损（reversal）功能：不再按前 14 分钟止损判败
    max_drawdown = 0.0
    trade_history: List[Dict[str, Any]] = []
    current_date = None
    trades_today = 0

    for idx, row in df.iterrows():
        if capital < MIN_BET:
            break
        if capital > peak_capital:
            peak_capital = capital

        prob = row["pred_prob"]
        pred = row["pred_up"]
        actual = row["actual_up"]
        effective_prob = prob if pred == 1 else (1.0 - prob)
        if effective_prob < prob_threshold:
            continue
        row_date = row.get("date")
        if row_date is not None:
            trade_date = pd.to_datetime(row_date).date() if hasattr(row_date, "date") else pd.to_datetime(row_date).date()
            if current_date != trade_date:
                current_date = trade_date
                trades_today = 0
        if max_trades_per_day > 0 and trades_today >= max_trades_per_day:
            continue

        # 高波动过滤：scale 为 None 或 0 时该 bar 跳过不下单；否则后续按 scale 缩小下注（少交易）
        if enable_high_vol_filter and row.get("high_vol_skip"):
            if high_vol_bet_scale is None or high_vol_bet_scale <= 0:
                continue

        # 下注金额：use_smart_bet 时按当前 capital 判断（赢到≥6万固定3000，输下<6万按5%，每笔前重算）
        if use_smart_bet:
            if capital > smart_bet_capital_threshold:
                if enable_dynamic_bet_ratio and peak_capital > 0:
                    dd = (peak_capital - capital) / peak_capital
                    if dd > 0.2:
                        bet_amount = 1500.0
                    elif dd > 0.1:
                        bet_amount = 2250.0
                    else:
                        bet_amount = FIXED_BET_AMOUNT
                else:
                    bet_amount = FIXED_BET_AMOUNT
            else:
                effective_ratio = bet_ratio
                if enable_dynamic_bet_ratio and peak_capital > 0:
                    dd = (peak_capital - capital) / peak_capital
                    if dd > 0.2:
                        effective_ratio = bet_ratio * 0.5
                    elif dd > 0.1:
                        effective_ratio = bet_ratio * 0.75
                bet_amount = max(MIN_BET, min(capital * effective_ratio, capital - 1.0))
        elif use_fixed_bet:
            if enable_dynamic_bet_ratio and peak_capital > 0:
                dd = (peak_capital - capital) / peak_capital
                if dd > 0.2:
                    bet_amount = 1500.0
                elif dd > 0.1:
                    bet_amount = 2250.0
                else:
                    bet_amount = FIXED_BET_AMOUNT
            else:
                bet_amount = FIXED_BET_AMOUNT
        else:
            # 比例下注：每笔 = 当前总资金 × 比例；有动态时比例按回撤缩小
            effective_ratio = bet_ratio
            if enable_dynamic_bet_ratio and peak_capital > 0:
                dd = (peak_capital - capital) / peak_capital
                if dd > 0.2:
                    effective_ratio = bet_ratio * 0.5
                elif dd > 0.1:
                    effective_ratio = bet_ratio * 0.75
            bet_amount = max(MIN_BET, min(capital * effective_ratio, capital - 1.0))
        if capital < bet_amount:
            break
        # 高波少交易：高波 bar 下注额乘以 scale，不足 MIN_BET 则跳过
        if enable_high_vol_filter and row.get("high_vol_skip") and high_vol_bet_scale is not None and high_vol_bet_scale > 0:
            bet_amount = bet_amount * high_vol_bet_scale
            if bet_amount < MIN_BET:
                continue
        if capital < bet_amount:
            break
        fee = bet_amount * (fee_rate + slippage)
        capital = capital - bet_amount - fee
        # 有止损且用 1m 反转时：前 14 分钟 down 已止损则算败，即使 15m 收盘 up
        reversal = row.get("reversal", False) if use_reversal else False
        if use_reversal and pred == 1:
            correct = actual == 1 and not reversal
            if actual == 1 and reversal:
                reversal_count += 1
        else:
            correct = pred == actual
        if correct:
            capital = capital + (bet_amount / order_price)
            wins += 1
        else:
            if stop_loss_exit_price is not None and stop_loss_exit_price > 0:
                capital = capital + (bet_amount / order_price) * float(stop_loss_exit_price)
            losses += 1
        trades_today += 1
        equity_curve.append(capital)
        trade_history.append({"correct": correct, "bet": bet_amount, "capital_after": capital})
        if peak_capital > 0:
            dd = (peak_capital - capital) / peak_capital
            if dd > max_drawdown:
                max_drawdown = dd

    total = wins + losses
    win_rate = wins / total if total > 0 else 0.0
    profit_pct = (capital - initial_capital) / initial_capital * 100.0 if initial_capital > 0 else 0.0
    min_cap = min(equity_curve) if equity_curve else initial_capital
    min_over_initial_pct = min_cap / initial_capital * 100.0 if initial_capital > 0 else 100.0

    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "reversal_count": reversal_count,
        "final_capital": capital,
        "profit_pct": profit_pct,
        "max_drawdown": max_drawdown * 100.0,
        "min_capital": min_cap,
        "min_over_initial_pct": min_over_initial_pct,
        "equity_curve": equity_curve,
        "trade_history": trade_history,
    }


def run_backtest_one_asset(
    asset: str,
    test_days: int,
    initial_capital: float,
    device: torch.device,
    data_src: Path,
    bet_ratio: float = 0.05,
    prob_threshold: float = 0.55,
    after_date: Optional[str] = None,
    enable_high_vol_filter: bool = False,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """对单个资产跑 GRU Regime 回测。after_date 为 YYYY-MM-DD 时仅用该日期之后的数据（防泄露）。end_date 可选，指定则只用 (after_date, end_date]。"""
    symbol = SYMBOL_MAP[asset]
    # 1) 原始特征数据（项目侧）
    now = pd.Timestamp.now(tz="UTC")
    after_ts = int(pd.Timestamp(after_date, tz="UTC").value // 10**6) if after_date else None

    df_orig = load_backtest_data_original_features(symbol, test_days, data_src, after_date=after_date, end_date=end_date)
    if df_orig.empty or len(df_orig) < 50:
        return {"asset": asset, "error": "数据不足", "total_trades": 0, "final_capital": initial_capital, "profit_pct": 0.0}

    # 2) 原始数据用于 GRU（同一数据源：data/raw）
    data_src_str = str(data_src)
    freq_min = 15
    filepath = data_src / f"{asset.lower()}_{freq_min}m.parquet"
    if not filepath.exists():
        return {"asset": asset, "error": f"Parquet 不存在: {filepath}", "total_trades": 0, "final_capital": initial_capital, "profit_pct": 0.0}
    df_raw = pd.read_parquet(filepath)
    if "timestamp" in df_raw.columns:
        df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)
    if after_ts is not None:
        cutoff_ts = after_ts
        df_raw = df_raw[df_raw["timestamp"] > cutoff_ts].copy().reset_index(drop=True)
        if end_date:
            end_ts = int(pd.Timestamp(end_date, tz="UTC").value // 10**6)
            df_raw = df_raw[df_raw["timestamp"] <= end_ts].copy().reset_index(drop=True)
    else:
        cutoff_ts = (now - pd.Timedelta(days=test_days)).value // 10**6
        df_raw = df_raw[df_raw["timestamp"] >= cutoff_ts].copy().reset_index(drop=True)
    if len(df_raw) < 50:
        return {"asset": asset, "error": "数据不足", "total_trades": 0, "final_capital": initial_capital, "profit_pct": 0.0}

    encoder, normalizer, train_config = load_encoder_and_normalizer(asset, device)
    emb_df = build_embeddings_for_backtest(asset, df_raw, encoder, normalizer, train_config, device)

    # 3) 合并原始特征 + embedding
    if "timestamp" not in df_orig.columns and "date" in df_orig.columns:
        df_orig["timestamp"] = pd.to_datetime(df_orig["date"]).astype("int64") // 10**6
    merged, _ = merge_embeddings(df_orig, emb_df, fill_strategy="zero")

    # 4) 方向标签（下一根 K 线涨跌）
    merged["direction_label"] = (merged["close"].shift(-1) > merged["close"]).astype(int)
    merged = merged[:-1].reset_index(drop=True)
    merged["actual_up"] = merged["direction_label"]

    # 5) 特征列与预测（必须与训练时一致：用模型保存的 feature 名称与顺序）
    lgb_path = MODELS_BEST / asset / "lightgbm_with_embedding.joblib"
    if not lgb_path.exists():
        return {"asset": asset, "error": f"LightGBM 不存在: {lgb_path}", "total_trades": 0, "final_capital": initial_capital, "profit_pct": 0.0}
    lgb_model = joblib.load(lgb_path)
    # LightGBM sklearn API: feature_names_in_ 或 _Booster.feature_name()
    if hasattr(lgb_model, "feature_names_in_") and lgb_model.feature_names_in_ is not None:
        feature_cols = list(lgb_model.feature_names_in_)
    else:
        feature_cols = list(lgb_model.feature_name_()) if hasattr(lgb_model, "feature_name_") else get_feature_columns(merged, include_embeddings=True)
    missing = [c for c in feature_cols if c not in merged.columns]
    if missing:
        return {"asset": asset, "error": f"特征缺失: {missing[:5]}{'...' if len(missing) > 5 else ''}", "total_trades": 0, "final_capital": initial_capital, "profit_pct": 0.0}
    X = merged[feature_cols].fillna(0)
    probs = lgb_model.predict_proba(X)[:, 1]
    merged["pred_prob"] = probs
    merged["pred_up"] = (probs >= 0.5).astype(int)

    # 高波动过滤列（无泄漏：过去 N 天 90 分位 + vol_ratio 固定阈）
    merged = add_high_vol_skip_column(merged)

    # 6) 交易循环
    result = run_trading_loop(
        merged,
        initial_capital=initial_capital,
        bet_ratio=bet_ratio,
        prob_threshold=prob_threshold,
        enable_high_vol_filter=enable_high_vol_filter,
    )
    result["asset"] = asset
    return result


def get_backtest_df_one_asset(
    asset: str,
    test_days: int,
    device: torch.device,
    data_src: Path,
    after_date: Optional[str] = None,
    end_date: Optional[str] = None,
    models_best_override: Optional[Path] = None,
) -> pd.DataFrame:
    """
    对单个资产加载数据、跑 GRU+LightGBM 预测，返回带 pred_prob/pred_up/actual_up/date 的 DataFrame。
    after_date/end_date 为 YYYY-MM-DD 时仅用 (after_date, end_date] 数据（防泄露、固定 365 天）。
    models_best_override: 指定模型目录（如无1h4h 的 models_best_no1h4h），不传则用默认 MODELS_BEST。
    """
    symbol = SYMBOL_MAP[asset]
    df_orig = load_backtest_data_original_features(
        symbol, test_days, data_src, after_date=after_date, end_date=end_date
    )
    if df_orig.empty or len(df_orig) < 50:
        raise ValueError(f"{asset}: 原始特征数据不足")
    freq_min = 15
    filepath = data_src / f"{asset.lower()}_{freq_min}m.parquet"
    if not filepath.exists():
        raise FileNotFoundError(f"{asset}: Parquet 不存在 {filepath}")
    df_raw = pd.read_parquet(filepath)
    if "timestamp" in df_raw.columns:
        df_raw = df_raw.sort_values("timestamp").reset_index(drop=True)
    if after_date:
        cutoff_ts = int(pd.Timestamp(after_date, tz="UTC").value // 10**6)
        df_raw = df_raw[df_raw["timestamp"] > cutoff_ts].copy().reset_index(drop=True)
        if end_date:
            end_ts = int(pd.Timestamp(end_date, tz="UTC").value // 10**6)
            df_raw = df_raw[df_raw["timestamp"] <= end_ts].copy().reset_index(drop=True)
    else:
        cutoff_ts = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=test_days)).value // 10**6
        df_raw = df_raw[df_raw["timestamp"] >= cutoff_ts].copy().reset_index(drop=True)
    if len(df_raw) < 50:
        raise ValueError(f"{asset}: 回测窗口内 parquet 数据不足")
    models_root = models_best_override if models_best_override is not None else MODELS_BEST
    encoder, normalizer, train_config = load_encoder_and_normalizer(asset, device, models_best_override=models_best_override)
    emb_df = build_embeddings_for_backtest(asset, df_raw, encoder, normalizer, train_config, device)
    if "timestamp" not in df_orig.columns and "date" in df_orig.columns:
        df_orig["timestamp"] = pd.to_datetime(df_orig["date"]).astype("int64") // 10**6
    merged, _ = merge_embeddings(df_orig, emb_df, fill_strategy="zero")
    merged["direction_label"] = (merged["close"].shift(-1) > merged["close"]).astype(int)
    merged = merged[:-1].reset_index(drop=True)
    merged["actual_up"] = merged["direction_label"]
    lgb_path = models_root / asset / "lightgbm_with_embedding.joblib"
    if not lgb_path.exists():
        raise FileNotFoundError(f"{asset}: LightGBM 不存在 {lgb_path}")
    lgb_model = joblib.load(lgb_path)
    if hasattr(lgb_model, "feature_names_in_") and lgb_model.feature_names_in_ is not None:
        feature_cols = list(lgb_model.feature_names_in_)
    else:
        feature_cols = list(lgb_model.feature_name_()) if hasattr(lgb_model, "feature_name_") else get_feature_columns(merged, include_embeddings=True)
    missing = [c for c in feature_cols if c not in merged.columns]
    if missing:
        raise ValueError(f"{asset}: 特征缺失 {missing[:5]}")
    X = merged[feature_cols].fillna(0)
    probs = lgb_model.predict_proba(X)[:, 1]
    merged["pred_prob"] = probs
    merged["pred_up"] = (probs >= 0.5).astype(int)
    if "date" not in merged.columns and "timestamp" in merged.columns:
        merged["date"] = pd.to_datetime(merged["timestamp"], unit="ms", utc=True)
    # 1m 反转：仅用 Polymarket CLOB prices-history 数据，见 https://docs.polymarket.com/developers/CLOB/timeseries
    # 需先运行 scripts/download_polymarket_1m.py 生成 data/polymarket_1m_{asset}.parquet
    poly_1m_path = PROJECT_ROOT / "data" / f"polymarket_1m_{asset.lower()}.parquet"
    if poly_1m_path.exists():
        df_1m = pd.read_parquet(poly_1m_path)
        if after_date:
            cutoff_sec = int(pd.Timestamp(after_date, tz="UTC").value // 10**9)
            df_1m = df_1m[df_1m["15m_start_ts"] > cutoff_sec].copy()
            if end_date:
                end_sec = int(pd.Timestamp(end_date, tz="UTC").value // 10**9)
                df_1m = df_1m[df_1m["15m_start_ts"] <= end_sec].copy()
        merged = compute_reversal_from_polymarket_1m(merged, df_1m)
    else:
        merged["reversal"] = False
    return merged


def main():
    ap = argparse.ArgumentParser(description="GRU Regime 最佳模型回测（一年，多组初始资金）")
    ap.add_argument("--test-days", type=int, default=365, help="回测天数，默认 365")
    ap.add_argument("--capitals", type=float, nargs="+", default=[400.0, 1000.0], help="初始资金列表，默认 400 1000")
    ap.add_argument("--bet-ratio", type=float, default=0.05, help="下注比例，默认 0.05")
    ap.add_argument("--prob-threshold", type=float, default=0.55, help="置信度阈值，默认 0.55")
    ap.add_argument("--data-src", type=str, default=None, help="数据目录，默认 data/raw")
    ap.add_argument("--after-date", type=str, default=None, help="防泄露：仅用该日期之后的数据回测，格式 YYYY-MM-DD（需晚于训练/验证集结束日）")
    ap.add_argument("--auto-after-date", action="store_true", help="自动使用无泄漏起始日（get_no_leak_start_date）作为回测起点")
    ap.add_argument("--no-mps", action="store_true", help="禁用 MPS（Mac GPU）")
    ap.add_argument("--enable-high-vol-filter", action="store_true", help="启用高波动过滤（保守 A：高波动 bar 跳过）")
    ap.add_argument("--compare-high-vol", action="store_true", help="有/无高波动过滤对比：同一无泄漏区间跑两遍，输出对比表（会用 --auto-after-date 并尽量用满到 parquet 末尾）")
    args = ap.parse_args()

    data_src = Path(args.data_src) if args.data_src else PROJECT_ROOT / "data" / "raw"
    if not data_src.is_absolute():
        data_src = PROJECT_ROOT / data_src
    device = get_device(use_mps=not args.no_mps)

    after_date = args.after_date
    if after_date is None and args.auto_after_date:
        after_date = get_no_leak_start_date(data_src, None, "BTC_USDT")
        print(f"  [--auto-after-date] 无泄漏起始日: {after_date}")

    if args.compare_high_vol:
        if after_date is None:
            after_date = get_no_leak_start_date(data_src, None, "BTC_USDT")
            print(f"  [--compare-high-vol] 无泄漏起始日: {after_date}")
        try:
            end_date = get_parquet_end_date(data_src, "BTC_USDT")
        except Exception:
            end_date = None
        start_ts = pd.Timestamp(after_date, tz="UTC")
        end_ts = pd.Timestamp(end_date, tz="UTC") if end_date else pd.Timestamp.now(tz="UTC")
        actual_days = (end_ts - start_ts).days
        cap = args.capitals[0] if args.capitals else 60000.0
        if cap < 3000:
            cap = 60000.0
        print("=" * 60)
        print("  高波动过滤对比（无泄漏区间，同一模型同一数据）")
        print("=" * 60)
        print(f"  区间: {after_date} ~ {end_date or 'now'}（约 {actual_days} 天）")
        print(f"  初始资金: ${cap:.0f}")
        print("=" * 60)
        no_filter: List[Dict[str, Any]] = []
        with_filter: List[Dict[str, Any]] = []
        for asset in ASSETS:
            print(f"  {asset} 无过滤...", end=" ", flush=True)
            try:
                r0 = run_backtest_one_asset(
                    asset, args.test_days, cap, device, data_src,
                    bet_ratio=args.bet_ratio, prob_threshold=args.prob_threshold,
                    after_date=after_date, enable_high_vol_filter=False, end_date=end_date,
                )
                no_filter.append(r0)
                print(f" 笔数 {r0.get('total_trades', 0)} 胜率 {r0.get('win_rate', 0)*100:.1f}% 最终 ${r0.get('final_capital', cap):.2f}")
            except Exception as e:
                no_filter.append({"asset": asset, "error": str(e), "total_trades": 0, "final_capital": cap, "win_rate": 0, "profit_pct": 0})
                print(f" ❌ {e}")
            print(f"  {asset} 有过滤...", end=" ", flush=True)
            try:
                r1 = run_backtest_one_asset(
                    asset, args.test_days, cap, device, data_src,
                    bet_ratio=args.bet_ratio, prob_threshold=args.prob_threshold,
                    after_date=after_date, enable_high_vol_filter=True, end_date=end_date,
                )
                with_filter.append(r1)
                print(f" 笔数 {r1.get('total_trades', 0)} 胜率 {r1.get('win_rate', 0)*100:.1f}% 最终 ${r1.get('final_capital', cap):.2f}")
            except Exception as e:
                with_filter.append({"asset": asset, "error": str(e), "total_trades": 0, "final_capital": cap, "win_rate": 0, "profit_pct": 0})
                print(f" ❌ {e}")
        print("\n" + "=" * 60)
        print("  对比汇总（无过滤 vs 有高波动过滤）")
        print("=" * 60)
        print(f"  {'资产':<12} {'无过滤笔数':>10} {'无过滤胜率':>10} {'无过滤最终$':>12} {'无过滤盈亏%':>10} | {'有过滤笔数':>10} {'有过滤胜率':>10} {'有过滤最终$':>12} {'有过滤盈亏%':>10}")
        print("  " + "-" * 110)
        for i in range(len(ASSETS)):
            asset = ASSETS[i]
            n0 = no_filter[i] if i < len(no_filter) else {}
            n1 = with_filter[i] if i < len(with_filter) else {}
            t0 = n0.get("total_trades", 0) or 0
            w0 = (n0.get("win_rate") or 0) * 100
            c0 = n0.get("final_capital") or cap
            p0 = n0.get("profit_pct") or 0
            t1 = n1.get("total_trades", 0) or 0
            w1 = (n1.get("win_rate") or 0) * 100
            c1 = n1.get("final_capital") or cap
            p1 = n1.get("profit_pct") or 0
            print(f"  {asset:<12} {t0:>10} {w0:>9.1f}% {c0:>12.2f} {p0:>9.1f}% | {t1:>10} {w1:>9.1f}% {c1:>12.2f} {p1:>9.1f}%")
        total_cap_no = sum((r.get("final_capital") or cap) for r in no_filter if not r.get("error"))
        total_cap_yes = sum((r.get("final_capital") or cap) for r in with_filter if not r.get("error"))
        total_trades_no = sum((r.get("total_trades") or 0) for r in no_filter)
        total_trades_yes = sum((r.get("total_trades") or 0) for r in with_filter)
        print("  " + "-" * 110)
        print(f"  {'合计':<12} {total_trades_no:>10} {'—':>10} {total_cap_no:>12.2f} {'—':>10} | {total_trades_yes:>10} {'—':>10} {total_cap_yes:>12.2f} {'—':>10}")
        print("=" * 60)
        return

    print("=" * 60)
    print("  GRU Regime 回测（models_best，15m）")
    print("=" * 60)
    print(f"  回测天数: {args.test_days}")
    print(f"  初始资金: {args.capitals}")
    print(f"  下注比例: {args.bet_ratio * 100}%")
    print(f"  置信度阈值: {args.prob_threshold}")
    if after_date:
        print(f"  防泄露截止日: 仅用 > {after_date} 的数据")
    if args.enable_high_vol_filter:
        print("  高波动过滤: 已启用（保守 A）")
    print("=" * 60)

    all_results: List[Dict[str, Any]] = []
    for initial_capital in args.capitals:
        print(f"\n--- 初始资金 ${initial_capital:.0f} ---")
        for asset in ASSETS:
            print(f"  回测 {asset}...", end=" ", flush=True)
            try:
                res = run_backtest_one_asset(
                    asset,
                    test_days=args.test_days,
                    initial_capital=initial_capital,
                    device=device,
                    data_src=data_src,
                    bet_ratio=args.bet_ratio,
                    prob_threshold=args.prob_threshold,
                    after_date=after_date,
                    enable_high_vol_filter=args.enable_high_vol_filter,
                )
                res["initial_capital"] = initial_capital
                all_results.append(res)
                if res.get("error"):
                    print(f"❌ {res['error']}")
                else:
                    print(f"✅ 交易 {res['total_trades']} 笔, 胜率 {res['win_rate']*100:.1f}%, 最终 ${res['final_capital']:.2f}, 盈亏 {res['profit_pct']:+.1f}%")
            except Exception as e:
                import traceback
                traceback.print_exc()
                all_results.append({
                    "asset": asset,
                    "initial_capital": initial_capital,
                    "error": str(e),
                    "total_trades": 0,
                    "final_capital": initial_capital,
                    "profit_pct": 0.0,
                })
                print(f"❌ {e}")

    # 汇总表
    print("\n" + "=" * 60)
    print("  汇总")
    print("=" * 60)
    for cap in args.capitals:
        rows = [r for r in all_results if r.get("initial_capital") == cap]
        print(f"\n  初始资金 ${cap:.0f}:")
        for r in rows:
            if r.get("error"):
                print(f"    {r['asset']}: {r['error']}")
            else:
                print(f"    {r['asset']}: 交易 {r['total_trades']} 笔, 胜率 {r['win_rate']*100:.1f}%, 最终 ${r['final_capital']:.2f}, 盈亏 {r['profit_pct']:+.1f}%, 最大回撤 {r.get('max_drawdown', 0):.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
