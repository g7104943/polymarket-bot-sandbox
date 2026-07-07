#!/usr/bin/env python3
"""
公平对比脚本 — 所有模型在同一环境下统一回测

两种模式:
  --from-models   金标准：所有模型从 parquet 重新预测 + 统一回测（耗时 ~20 分钟，最公平）
  (默认)          快速模式：旧模型用 prediction_trades.json 交易决策 + 统一买价重算 PnL

核心原则:
  ✅ 同一无泄漏日期范围（与 Exp8 测试集对齐: 2026-01-26 ~ 2026-02-10）
  ✅ 同一买入价格（支持同时跑 $0.527 和 $0.51 等多价格）
  ✅ 同一初始资金（$400）
  ✅ 同一手续费 + 滑点假设（0.5%）
  ✅ v5 新模型（Exp8 含 5 个 target PM 特征）: 用 parquet 预测 + 新超参回测

用法:
  python scripts/compare_fair.py                                  # 快速模式
  python scripts/compare_fair.py --from-models                    # 金标准：全模型从头预测
  python scripts/compare_fair.py --buy-prices 0.527               # 只跑一个价格
  python scripts/compare_fair.py --buy-prices 0.527,0.51,0.54    # 多个价格
  python scripts/compare_fair.py --date-from 2026-02-01           # 自定义起始日
"""
from __future__ import annotations

import json
import sys
import glob
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

POLYMARKET_DIR = PROJECT_ROOT / "polymarket"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "sentiment_grid_search" / "results"
V5_PRED_FILE = RESULTS_DIR / "v5_test_predictions.parquet"

# ─── 常量 ──────────────────────────────────────────────
DEFAULT_INITIAL_CAPITAL = 400.0
DEFAULT_FEE_RATE = 0.015        # Polymarket taker 手续费 ~1.5% (2026-01-19 起 15分钟加密市场)
DEFAULT_SLIPPAGE_RATE = 0.003   # 滑点 ~0.3%
SETTLEMENT_COST = DEFAULT_FEE_RATE + DEFAULT_SLIPPAGE_RATE  # 1.8% — 仅用于 PnL 结算
TOTAL_COST = SETTLEMENT_COST    # 兼容旧引用（仅用于结算和显示，不用于 edge/Kelly）
LIQUIDITY_CAP = 60_000.0
LIQUIDITY_BET = 3_000.0

# ─── 模拟 K 线噪声（与 train_production_model.py 完全一致）─────
SIM_NOISE_STD = 0.0013  # 对数空间标准差
_NOISE_SEED_BASE = 42   # 与 run_grid.py SEED 一致


# ─── 实验标签映射（model_name + rules_version -> Exp 名称）──
_EXP_LABEL_MAP = {
    ("v5_production", "v3"): "Exp8",
    ("v5_production", "v4"): "Exp9",
    ("v5_production_sim_noise", "v3"): "Exp10",
    ("v5_production_tv", "v3"): "Exp13",
    ("v5_production_sim_noise_tv", "v3"): "Exp14",
    ("v5_production_tv_365d", "v3"): "Exp16",
    ("v5_production_no_target_pm", "v3"): "Exp17",
}

def _get_exp_label(model_name: str, rules_version: str) -> str:
    """根据模型名和规则版本返回实验标签（Exp8/Exp9/Exp10/...）。"""
    return _EXP_LABEL_MAP.get((model_name, rules_version), f"{model_name}_{rules_version}")


def _apply_simulated_candle_noise(tech_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
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
    """将噪声方向标签替换为基于真实 close 的方向标签。"""
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

# v5 测试集日期范围（无泄漏期 — 与 Exp8 模型测试集对齐）
DEFAULT_DATE_FROM = "2026-01-26"
DEFAULT_DATE_TO = "2026-02-10"


# ═══════════════════════════════════════════════════════════
# Part 1: 旧模型重新模拟（用统一买价 + 统一日期）
# ═══════════════════════════════════════════════════════════

# 旧模型配置：从启动脚本提取的每个组合的下注规则
OLD_MODEL_CONFIGS = {
    # ── 三个版本（5 组）──
    "logs_eth":         {"bet_pct": 5.0, "coin": "ETH",  "label": "ETH_92无动态"},
    "logs_eth_10_90":   {"bet_pct": 5.0, "coin": "ETH",  "label": "ETH_10-90有动态", "risk_ctrl": True},
    "logs_btc":         {"bet_pct": 5.0, "coin": "BTC",  "label": "BTC_55无动态"},
    "logs_xrp":         {"bet_pct": 5.0, "coin": "XRP",  "label": "XRP_53无动态"},
    "logs_xrp_20_80":   {"bet_pct": 5.0, "coin": "XRP",  "label": "XRP_20-80无动态"},
    # ── GRU 新模型（8 组）──
    "logs_gru_xrp_55":         {"bet_pct": 5.0, "coin": "XRP",  "label": "GRU_XRP_55无动态"},
    "logs_gru_btc_55":         {"bet_pct": 5.0, "coin": "BTC",  "label": "GRU_BTC_55无动态"},
    "logs_gru_eth_54":         {"bet_pct": 5.0, "coin": "ETH",  "label": "GRU_ETH_54有动态", "risk_ctrl": True},
    "logs_gru_eth_55_dyn":     {"bet_pct": 5.0, "coin": "ETH",  "label": "GRU_ETH_55有动态", "risk_ctrl": True},
    "logs_gru_sol_52":         {"bet_pct": 5.0, "coin": "SOL",  "label": "GRU_SOL_52无动态"},
    "logs_gru_eth_55_no1h4h":  {"bet_pct": 5.0, "coin": "ETH",  "label": "GRU_ETH_55无1h4h"},
    "logs_gru_btc_57_no1h4h":  {"bet_pct": 5.0, "coin": "BTC",  "label": "GRU_BTC_57无1h4h"},
    "logs_gru_xrp_53_no1h4h":  {"bet_pct": 5.0, "coin": "XRP",  "label": "GRU_XRP_53无1h4h", "risk_ctrl": True},
    "logs_gru_eth_53_no1h4h":  {"bet_pct": 5.0, "coin": "ETH",  "label": "GRU_ETH_53无1h4h"},
}

# 超参组合配置
HYPERPARAM_CONFIGS = {
    "logs_eth_超参":                {"bet_pct": 5.0, "coin": "ETH",  "label": "ETH_超参", "tiers": [2, 4, 2, 6]},
    "logs_eth_10_90_超参":          {"bet_pct": 5.0, "coin": "ETH",  "label": "ETH_10-90_超参", "tiers": [6, 2, 4, 6], "risk_ctrl": True},
    "logs_btc_超参":                {"bet_pct": 5.0, "coin": "BTC",  "label": "BTC_超参", "tiers": [4, 10, 6, 2]},
    "logs_gru_eth_54_超参":         {"bet_pct": 5.0, "coin": "ETH",  "label": "GRU_ETH_54_超参", "tiers": [4, 10, 10, 2], "risk_ctrl": True},
    "logs_gru_eth_55_dyn_超参":     {"bet_pct": 5.0, "coin": "ETH",  "label": "GRU_ETH_55dyn_超参", "tiers": [8, 10, 4, 2], "risk_ctrl": True},
    "logs_gru_eth_55_no1h4h_超参":  {"bet_pct": 5.0, "coin": "ETH",  "label": "GRU_ETH_55no1h_超参", "tiers": [8, 10, 2, 6]},
    "logs_gru_btc_57_no1h4h_超参":  {"bet_pct": 5.0, "coin": "BTC",  "label": "GRU_BTC_57no1h_超参", "tiers": [8, 2, 10, 10]},
    "logs_gru_xrp_53_no1h4h_超参":  {"bet_pct": 5.0, "coin": "XRP",  "label": "GRU_XRP_53no1h_超参", "tiers": [4, 4, 2, 6], "risk_ctrl": True},
    "logs_gru_sol_52_超参":         {"bet_pct": 5.0, "coin": "SOL",  "label": "GRU_SOL_52_超参", "tiers": [2, 6, 10, 6]},
}


def _load_trades(trades_path: Path) -> List[Dict]:
    """加载交易记录，支持 .json 和 .bak 文件。"""
    # 优先读 .json，没有则读最新的 .bak
    if trades_path.exists():
        with open(trades_path) as f:
            return json.load(f)

    bak_files = sorted(trades_path.parent.glob(f"{trades_path.name}.bak.*"))
    if bak_files:
        with open(bak_files[-1]) as f:
            return json.load(f)

    return []


def resimulate_combo(
    trades: List[Dict],
    buy_price: float,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    bet_pct: float = 5.0,
    tiers: Optional[List[float]] = None,
    risk_ctrl: bool = False,
    date_from: str = DEFAULT_DATE_FROM,
    date_to: str = DEFAULT_DATE_TO,
) -> Dict[str, Any]:
    """
    用统一买价重新模拟一个旧模型的资金曲线。

    核心逻辑:
    - 交易决策（哪些 bar 交易、方向）: 来自旧模型实际运行记录，不做改变
    - PnL 计算: 用统一的 buy_price 重算（赔率 = (1 - buy_price) / buy_price）
    - 仓位: 用模型原始的 bet_pct（固定 5%）或 tiers
    """
    odds = (1.0 - buy_price) / buy_price

    # 过滤: 只保留已执行且已结算的交易，且在日期范围内
    valid = []
    for t in trades:
        if t.get("status") != "executed":
            continue
        if t.get("result") not in ("win", "lose"):
            continue
        ts = t.get("timestamp", "")[:10]
        if ts < date_from or ts > date_to:
            continue
        valid.append(t)

    # 按时间排序
    valid.sort(key=lambda t: t.get("timestamp", ""))

    capital = initial_capital
    peak = capital
    min_capital = capital
    max_dd = 0.0          # 正确追踪：每笔交易后的实时回撤最大值
    n_trades = 0
    wins = 0
    losses = 0
    total_profit = 0.0
    total_loss = 0.0

    for t in valid:
        # 回撤风控（仅 risk_ctrl 模型）
        if risk_ctrl and peak > 0:
            dd = (peak - capital) / peak
            max_dd = max(max_dd, dd)
            if dd > 0.2:
                # 回撤 >20% 减半
                effective_pct = bet_pct * 0.5
            elif dd > 0.1:
                effective_pct = bet_pct * 0.75
            else:
                effective_pct = bet_pct
        else:
            effective_pct = bet_pct

        # 计算下注金额
        # 超参档位（如果有）：这里简化为取中位数档位
        # 实际超参用 confidence + B1/B2/B3 档位，但 re-sim 时我们保持和原模拟盘一致
        if tiers:
            # 简化: 用 tiers 的均值作为 bet_pct
            effective_pct = sum(tiers) / len(tiers)

        # 简单流动性封顶：>= $60K 固定 $3K，< $60K 正常百分比
        if capital >= LIQUIDITY_CAP:
            bet_amount = LIQUIDITY_BET
        else:
            bet_amount = capital * (effective_pct / 100.0)

        bet_amount = min(bet_amount, capital)
        if bet_amount < 1.0:
            continue

        n_trades += 1
        result = t["result"]

        if result == "win":
            pnl = bet_amount * odds - bet_amount * TOTAL_COST
            wins += 1
            total_profit += pnl
        else:
            pnl = -(bet_amount + bet_amount * TOTAL_COST)
            losses += 1
            total_loss += abs(pnl)

        capital += pnl
        peak = max(peak, capital)
        min_capital = min(min_capital, capital)
        # 交易后立即更新回撤
        dd_now = (peak - capital) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd_now)

        if capital <= 0:
            capital = 0
            break
    win_rate = wins / n_trades if n_trades > 0 else 0
    return_pct = (capital - initial_capital) / initial_capital * 100
    pf = total_profit / total_loss if total_loss > 0 else (999 if total_profit > 0 else 0)

    return {
        "final_capital": round(capital, 2),
        "return_pct": round(return_pct, 1),
        "win_rate": round(win_rate * 100, 1),
        "max_drawdown": round(max_dd * 100, 1),
        "n_trades": n_trades,
        "wins": wins,
        "losses": losses,
        "profit_factor": round(pf, 2),
        "min_capital": round(min_capital, 2),
        "min_pct": round(min_capital / initial_capital * 100, 1) if initial_capital > 0 else 0,
    }


def resimulate_all_old_combos(
    buy_price: float,
    date_from: str = DEFAULT_DATE_FROM,
    date_to: str = DEFAULT_DATE_TO,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
) -> List[Dict[str, Any]]:
    """重新模拟所有旧模型组合。"""
    results = []

    all_configs = {}
    all_configs.update(OLD_MODEL_CONFIGS)
    all_configs.update(HYPERPARAM_CONFIGS)

    for dir_name, cfg in all_configs.items():
        trades_path = POLYMARKET_DIR / dir_name / "prediction_trades.json"
        trades = _load_trades(trades_path)

        if not trades:
            continue

        res = resimulate_combo(
            trades,
            buy_price=buy_price,
            initial_capital=initial_capital,
            bet_pct=cfg["bet_pct"],
            tiers=cfg.get("tiers"),
            risk_ctrl=cfg.get("risk_ctrl", False),
            date_from=date_from,
            date_to=date_to,
        )

        if res["n_trades"] == 0:
            continue

        is_hp = "超参" in dir_name
        results.append({
            "排名": 0,
            "组合": cfg["label"],
            "组合ID": dir_name.replace("logs_", "").upper()[:20],
            "币种": cfg["coin"],
            "下注方式": f"档位{cfg['tiers']}" if cfg.get("tiers") else f"固定{cfg['bet_pct']}%",
            "最终资金": res["final_capital"],
            "胜率%": res["win_rate"],
            "盈亏%": res["return_pct"],
            "最大回撤%": res["max_drawdown"],
            "交易次数": res["n_trades"],
            "最低/初始%": res["min_pct"],
            "来源": "超参重模拟" if is_hp else "旧模型重模拟",
        })

    return results


# ═══════════════════════════════════════════════════════════
# Part 2: v5 新模型回测（与 compare_all_combos.py 一致的逻辑）
# ═══════════════════════════════════════════════════════════

def simulate_v5_trading(
    predictions: pd.DataFrame,
    params: Dict[str, Any],
    buy_price: float,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
) -> Dict[str, float]:
    """v5 模型回测模拟交易 — 与 optimize_trading_rules._simulate_trading_core 完全对齐。"""
    odds = (1.0 - buy_price) / buy_price

    min_conf = params.get("min_confidence", 0.50)
    min_edge = params.get("min_edge", 0.02)
    kelly_frac = params.get("kelly_frac", 0.33)
    bet_pct_normal = params.get("bet_pct_normal", 0.05)
    bet_pct_conservative = params.get("bet_pct_conservative", 0.03)
    conf_tier1 = params.get("conf_tier1_bound", 0.55)
    conf_tier2 = params.get("conf_tier2_bound", 0.60)
    tier1_mult = params.get("tier1_mult", 0.30)
    tier2_mult = params.get("tier2_mult", 0.60)
    tier3_mult = params.get("tier3_mult", 1.00)
    # 周期制冷却: cooldown_bars 表示跳过的完整 15 分钟周期数
    assert "asset" in predictions.columns, "predictions DataFrame 缺少 'asset' 列，无法计算 num_assets"
    num_assets = int(predictions["asset"].nunique())
    cooldown_bars = int(params.get("cooldown_bars", 0)) * num_assets
    drawdown_halt = params.get("drawdown_halt", 0.20)

    capital = initial_capital
    peak = capital
    min_capital = capital
    max_dd = 0.0
    n_trades = 0
    wins = 0
    losses = 0
    total_profit = 0.0
    total_loss = 0.0
    cooldown_remaining = 0
    dd_halt_remaining = 0  # 回撤熔断冷却（24h = 96 bars）
    DD_HALT_BARS = 96      # 15 分钟 × 96 = 24 小时
    ever_reached_cap = False  # 三阶段仓位追踪
    trade_returns: List[float] = []  # 每笔交易的收益率（用于算 Sharpe）

    for _, row in predictions.iterrows():
        proba = row["proba_up"]
        actual = int(row["actual"])

        # 回撤熔断：触发后暂停 24 小时，之后重置 peak 恢复交易
        if dd_halt_remaining > 0:
            dd_halt_remaining -= 1
            if dd_halt_remaining == 0:
                # 24h 冷却结束，重置 peak 为当前资金，重新开始追踪回撤
                peak = capital
            continue

        dd_now = (peak - capital) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd_now)
        if dd_now >= drawdown_halt:
            dd_halt_remaining = DD_HALT_BARS
            continue

        # 冷却期（在回撤检查之后）
        if cooldown_remaining > 0:
            cooldown_remaining -= 1
            continue

        direction = 1 if proba >= 0.5 else 0
        p = proba if direction == 1 else (1 - proba)

        if p < min_conf:
            continue

        # Edge 公式（纯数学，不含费率）
        q = 1.0 - p
        edge = p * odds - q
        if edge < min_edge:
            continue

        # Kelly 公式（纯数学，不含费率）
        kelly_f = (p * odds - q) / odds if odds > 0 else 0
        kelly_f = max(0, kelly_f)
        bet_ratio = kelly_f * kelly_frac

        # 置信度分档乘数
        if p < conf_tier1:
            bet_ratio *= tier1_mult
        elif p < conf_tier2:
            bet_ratio *= tier2_mult
        else:
            bet_ratio *= tier3_mult

        # 仓位规则
        if capital >= LIQUIDITY_CAP:
            # >= $60K → 固定 $3K（流动性封顶）
            bet_amount = LIQUIDITY_BET
        else:
            # < $60K → 正常百分比
            bet_ratio = min(bet_ratio, bet_pct_normal)
            bet_amount = capital * bet_ratio

        # 不超过可用资金 95%（与优化器一致）
        cap_limit = capital * 0.95
        if bet_amount > cap_limit:
            bet_amount = cap_limit

        if bet_amount < 1.0:
            continue

        n_trades += 1
        correct = (direction == 1 and actual == 1) or (direction == 0 and actual == 0)
        cap_before = capital

        if correct:
            pnl = bet_amount * odds - bet_amount * TOTAL_COST
            wins += 1
            total_profit += pnl
            cooldown_remaining = 0
        else:
            pnl = -bet_amount  # Polymarket: 输不另扣费
            losses += 1
            total_loss += abs(pnl)
            if cooldown_bars > 0:
                cooldown_remaining = cooldown_bars

        if cap_before > 0:
            trade_returns.append(pnl / cap_before)
        capital += pnl
        peak = max(peak, capital)
        min_capital = min(min_capital, capital)
        dd_now = (peak - capital) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd_now)

        if capital <= 0:
            capital = 0
            break

    win_rate = wins / n_trades if n_trades > 0 else 0
    return_pct = (capital - initial_capital) / initial_capital * 100
    pf = total_profit / total_loss if total_loss > 0 else (999 if total_profit > 0 else 0)

    # 按笔收益序列算 Sharpe（供融合超参/回测用）
    if len(trade_returns) >= 2:
        import numpy as np
        arr = np.array(trade_returns)
        sharpe = float(np.mean(arr) / (np.std(arr) + 1e-10) * np.sqrt(len(arr)))
    else:
        sharpe = 0.0

    return {
        "final_capital": round(capital, 2),
        "return_pct": round(return_pct, 1),
        "win_rate": round(win_rate * 100, 1),
        "max_drawdown": round(max_dd * 100, 1),
        "n_trades": n_trades,
        "wins": wins,
        "losses": losses,
        "profit_factor": round(pf, 2),
        "min_capital": round(min_capital, 2),
        "min_pct": round(min_capital / initial_capital * 100, 1) if initial_capital > 0 else 0,
        "sharpe": round(sharpe, 4),
    }


def _load_v5_params(rules_file: Path) -> Optional[Dict]:
    """加载 v5 参数文件并转换格式。"""
    if not rules_file.exists():
        return None
    with open(rules_file) as f:
        opt = json.load(f)
    params = opt["trading_rules"]
    tiers = params.get("confidence_tiers", [])
    if tiers and len(tiers) >= 2:
        params["conf_tier1_bound"] = tiers[0][1] if len(tiers[0]) > 1 else 0.55
        params["conf_tier2_bound"] = tiers[1][1] if len(tiers[1]) > 1 else 0.60
        params["tier1_mult"] = tiers[0][2] if len(tiers[0]) > 2 else 0.30
        params["tier2_mult"] = tiers[1][2] if len(tiers[1]) > 2 else 0.60
        params["tier3_mult"] = tiers[2][2] if len(tiers[2]) > 2 else 1.00
    bp_info = opt.get("polymarket_constraints", {}).get("buy_price", "?")
    bp_range = opt.get("polymarket_constraints", {}).get("buy_price_range")
    return {"params": params, "buy_price": bp_info, "buy_price_range": bp_range}


def run_v5_backtest_all(
    buy_price: float,
    date_from: str = DEFAULT_DATE_FROM,
    date_to: str = DEFAULT_DATE_TO,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    rules_version: str = "v3",
    predictions: Optional[pd.DataFrame] = None,
    model_label: str = "v5",
    exp_label: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """v5 Grid 回测: 每个模型用自己的优化买价和参数。

    v5 Grid 架构:
      - 固定买价模型，各自在对应买价下交易
        v3: 7 个 ($0.45~$0.51)  |  v4: 9 个 ($0.45~$0.53)
      - 1 个动态阶梯模型，均分资金在各档同时交易
      - buy_price 参数仅用于旧模型对比，v5 各自使用自己的买价
      - rules_version: "v3" 或 "v4"（v4 = 0.3~0.7 bet_pct 搜索范围）
      - predictions: 外部传入的预测 DataFrame（用于多模型对比）
      - model_label: 模型标签（用于结果命名区分）
    """
    if predictions is not None:
        pred_df = predictions.copy()
    elif V5_PRED_FILE.exists():
        pred_df = pd.read_parquet(V5_PRED_FILE)
    else:
        print(f"  ⚠️  v5 预测文件不存在: {V5_PRED_FILE}")
        return []

    # 过滤日期范围
    pred_df["date_str"] = pred_df["timestamp"].astype(str).str[:10]
    pred_df = pred_df[(pred_df["date_str"] >= date_from) & (pred_df["date_str"] <= date_to)]
    pred_df = pred_df.drop(columns=["date_str"])

    if len(pred_df) == 0:
        print(f"  ⚠️  日期范围内无预测数据")
        return []

    n_days = pred_df["timestamp"].dt.date.nunique() if hasattr(pred_df["timestamp"], "dt") else "?"
    print(f"  {model_label} 预测: {len(pred_df)} bars, {n_days} 天")
    print(f"  时间范围: {pred_df['timestamp'].min()} ~ {pred_df['timestamp'].max()}")

    # v5 Grid 活跃模型配置（v3/v4 均为 $0.45~$0.53 共 9 个固定买价）
    V5_ACTIVE_PRICES = [0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53]

    results = []
    all_df = pred_df.sort_values("timestamp").reset_index(drop=True)
    source_label = exp_label or _get_exp_label(model_label, rules_version)

    # ─── 7 个固定买价模型 ─────────────────────────────
    for bp in V5_ACTIVE_PRICES:
        tag = f"bp{bp:.3f}".replace(".", "")
        # 优先加载模型专属超参, 否则回退到共享超参
        model_rules_dir = RESULTS_DIR / model_label
        rules_file = model_rules_dir / f"optimal_trading_rules_{rules_version}_{tag}.json"
        if not rules_file.exists():
            rules_file = RESULTS_DIR / f"optimal_trading_rules_{rules_version}_{tag}.json"
        loaded = _load_v5_params(rules_file)
        if not loaded:
            print(f"  ⚠️ {model_label}_{tag}: JSON 不存在，跳过")
            continue

        params = loaded["params"]
        kelly_str = f"{params.get('kelly_frac', 0):.2f}"

        # 用该模型自己的优化买价回测
        res_all = simulate_v5_trading(all_df, params, bp, initial_capital)

        results.append({
            "排名": 0,
            "组合": f"{model_label}_{tag}_ALL",
            "组合ID": f"{model_label}_{tag}"[:20],
            "币种": "BTC+ETH+XRP",
            "下注方式": f"限价@${bp:.2f} K×{kelly_str}",
            "最终资金": res_all["final_capital"],
            "胜率%": res_all["win_rate"],
            "盈亏%": res_all["return_pct"],
            "最大回撤%": res_all["max_drawdown"],
            "交易次数": res_all["n_trades"],
            "最低/初始%": res_all["min_pct"],
            "来源": source_label,
        })

        # 分币种
        for asset in ["BTC_USDT", "ETH_USDT", "XRP_USDT"]:
            asset_df = pred_df[pred_df["asset"] == asset].sort_values("timestamp").reset_index(drop=True)
            if len(asset_df) == 0:
                continue
            coin = asset.split("_")[0]
            res = simulate_v5_trading(asset_df, params, bp, initial_capital)
            if res["n_trades"] == 0:
                continue
            results.append({
                "排名": 0,
                "组合": f"{model_label}_{tag}_{coin}",
                "组合ID": f"{model_label}_{tag}_{coin}"[:20],
                "币种": coin,
                "下注方式": f"限价@${bp:.2f} K×{kelly_str}",
                "最终资金": res["final_capital"],
                "胜率%": res["win_rate"],
                "盈亏%": res["return_pct"],
                "最大回撤%": res["max_drawdown"],
                "交易次数": res["n_trades"],
                "最低/初始%": res["min_pct"],
                "来源": source_label,
            })

    # ─── 动态阶梯模型 ────────────────────────────────
    dyn_file = model_rules_dir / f"optimal_trading_rules_{rules_version}_bp_dyn_0450_0530.json"
    if not dyn_file.exists():
        dyn_file = RESULTS_DIR / f"optimal_trading_rules_{rules_version}_bp_dyn_0450_0530.json"
    dyn_loaded = _load_v5_params(dyn_file)
    if dyn_loaded:
        dyn_params = dyn_loaded["params"]
        kelly_str = f"{dyn_params.get('kelly_frac', 0):.2f}"

        # 阶梯: 均分资金到每个价位，各自独立回测后汇总
        n_rungs = len(V5_ACTIVE_PRICES)
        ladder_capital = initial_capital / n_rungs
        total_final = 0
        total_trades = 0
        total_wins = 0
        total_losses = 0
        max_dd_all = 0

        for bp in V5_ACTIVE_PRICES:
            res = simulate_v5_trading(all_df, dyn_params, bp, ladder_capital)
            total_final += res["final_capital"]
            total_trades += res["n_trades"]
            total_wins += res["wins"]
            total_losses += res["losses"]
            max_dd_all = max(max_dd_all, res["max_drawdown"])

        wr = total_wins / total_trades * 100 if total_trades > 0 else 0
        ret_pct = (total_final - initial_capital) / initial_capital * 100

        results.append({
            "排名": 0,
            "组合": f"{model_label}_ladder_ALL",
            "组合ID": f"{model_label}_ladder"[:20],
            "币种": "BTC+ETH+XRP",
            "下注方式": f"阶梯${V5_ACTIVE_PRICES[0]:.2f}~${V5_ACTIVE_PRICES[-1]:.2f}",
            "最终资金": round(total_final, 2),
            "胜率%": round(wr, 1),
            "盈亏%": round(ret_pct, 1),
            "最大回撤%": max_dd_all,
            "交易次数": total_trades,
            "最低/初始%": round(total_final / initial_capital * 100, 1) if total_final < initial_capital else 100.0,
            "来源": source_label,
        })

    return results


# ═══════════════════════════════════════════════════════════
# Part 3: 排名输出
# ═══════════════════════════════════════════════════════════

def _load_exp7_backup(bp: float) -> Optional[Dict]:
    """加载 Exp7 备份参数（如果存在），用于对比显示。"""
    backup_dir = RESULTS_DIR / "exp7_backup"
    tag = f"bp{bp:.3f}".replace(".", "")
    backup_file = backup_dir / f"exp7_best_{tag}.json"
    if backup_file.exists():
        with open(backup_file) as f:
            return json.load(f)
    return None


def print_ranking(combos: List[Dict[str, Any]], buy_price: float,
                  date_from: str, date_to: str):
    """按最终资金排名打印表格。"""
    combos = sorted(combos, key=lambda x: -x["最终资金"])
    for i, c in enumerate(combos):
        c["排名"] = i + 1

    odds = (1.0 - buy_price) / buy_price
    print("\n" + "=" * 140)
    print("  📊 公平对比排名 — 所有模型统一环境回测")
    print(f"  ✅ 日期范围: {date_from} ~ {date_to}  |  买价: ${buy_price:.3f}  |  赔率: {odds:.4f}")
    print(f"  ✅ 初始资金: $400  |  手续费+滑点: {TOTAL_COST*100:.1f}%  |  流动性封顶: ${LIQUIDITY_BET:.0f}@${LIQUIDITY_CAP/1000:.0f}K")
    # 检查来源类型决定描述
    sources = set(c.get("来源", "") for c in combos)
    if any("模型回测" in s for s in sources):
        print(f"  ✅ 旧模型+GRU: 从模型文件重新预测（金标准模式）")
    else:
        print(f"  ✅ 旧模型: prediction_trades.json 的交易决策 + 统一买价重算PnL")
    # 列出所有 v5 实验来源
    exp_sources = sorted(set(c.get("来源", "") for c in combos
                             if any(tag in c.get("来源", "") for tag in ["Exp8", "Exp9", "Exp10", "v5"])))
    if exp_sources:
        print(f"  ✅ v5模型 ({', '.join(exp_sources)}): parquet预测 + target PM 特征 + 超参回测")
    print("=" * 140)

    # 表头
    header = (
        f"{'#':>3} {'组合':<28} {'币种':<12} "
        f"{'下注方式':<20} {'最终资金':>8} {'胜率%':>6} {'盈亏%':>7} "
        f"{'回撤%':>6} {'交易':>5} {'最低%':>6} {'来源':<12}"
    )
    print(header)
    print("─" * 140)

    _v5_tags = ("v5", "exp8", "exp9", "exp10")
    for c in combos:
        is_v5 = any(tag in c["来源"].lower() for tag in _v5_tags)
        marker = "★" if is_v5 else " "

        line = (
            f"{c['排名']:>3}{marker}"
            f" {c['组合']:<27} {c['币种']:<12} "
            f"{c['下注方式']:<20} {'$' + str(c['最终资金']):>8} {c['胜率%']:>5.1f}% "
            f"{c['盈亏%']:>+6.1f}% {c['最大回撤%']:>5.1f}% {c['交易次数']:>5} "
            f"{c['最低/初始%']:>5.1f}% {c['来源']:<12}"
        )
        print(line)

    print("─" * 140)

    # 统计
    old_combos = [c for c in combos if not any(tag in c["来源"].lower() for tag in _v5_tags)]
    v5_combos = [c for c in combos if any(tag in c["来源"].lower() for tag in _v5_tags)]

    if old_combos:
        profitable = len([c for c in old_combos if c["最终资金"] > DEFAULT_INITIAL_CAPITAL])
        avg_wr = np.mean([c["胜率%"] for c in old_combos])
        avg_ret = np.mean([c["盈亏%"] for c in old_combos])
        best_old = max(old_combos, key=lambda x: x["最终资金"]) if old_combos else None
        print(f"\n  旧模型 {len(old_combos)} 个: 盈利 {profitable} 个, "
              f"平均胜率 {avg_wr:.1f}%, 平均盈亏 {avg_ret:+.1f}%"
              + (f", 最佳: {best_old['组合']} ${best_old['最终资金']}" if best_old else ""))

    if v5_combos:
        # 按来源分组统计（Exp8/Exp9/Exp10 各自独立）
        source_groups = {}
        for c in v5_combos:
            src = c.get("来源", "v5")
            source_groups.setdefault(src, []).append(c)

        for src_label in sorted(source_groups.keys()):
            grp = source_groups[src_label]
            grp_all = [c for c in grp if "_ALL" in c["组合"]]
            if not grp_all:
                continue
            profitable_v5 = len([c for c in grp_all if c["最终资金"] > DEFAULT_INITIAL_CAPITAL])
            avg_wr_v5 = np.mean([c["胜率%"] for c in grp_all])
            avg_ret_v5 = np.mean([c["盈亏%"] for c in grp_all])
            best_v5 = max(grp_all, key=lambda x: x["最终资金"])
            print(f"\n  {src_label} 模型 (ALL 汇总) {len(grp_all)} 个: 盈利 {profitable_v5} 个, "
                  f"平均胜率 {avg_wr_v5:.1f}%, 平均盈亏 {avg_ret_v5:+.1f}%")
            for c in grp_all:
                print(f"  ★ {c['组合']}: ${c['最终资金']:,.2f} "
                      f"(胜率{c['胜率%']:.1f}%, 盈亏{c['盈亏%']:+.1f}%, "
                      f"回撤{c['最大回撤%']:.1f}%, {c['交易次数']}笔)")

    # ─── Exp7 vs Exp8 超参对比 ────────────────────────
    exp7_backup_dir = RESULTS_DIR / "exp7_backup"
    if exp7_backup_dir.exists() and any(exp7_backup_dir.glob("*.json")):
        print(f"\n{'─' * 80}")
        print("  📊 Exp7 vs Exp8 超参优化对比（bootstrap median，同一预测数据）")
        print(f"{'─' * 80}")
        print(f"  {'买价':<8} {'Exp7 最优资金':>16} {'Exp8 最优资金':>16} {'变化':>10} "
              f"{'Exp7 WR':>8} {'Exp8 WR':>8}")

        for bp_cents in [45, 46, 47, 48, 49, 50, 51]:
            bp = bp_cents / 100
            tag = f"bp{bp:.3f}".replace(".", "")
            exp7 = _load_exp7_backup(bp)
            exp8_file = RESULTS_DIR / f"optimal_trading_rules_v3_{tag}.json"

            if not exp7 or not exp8_file.exists():
                continue

            with open(exp8_file) as f:
                exp8 = json.load(f)

            e7_cap = exp7["best_metrics"]["final_capital"]
            e7_wr = exp7["best_metrics"]["win_rate"]
            e8_met = exp8.get("metrics_bootstrap_median", {})
            e8_cap = e8_met.get("final_capital", 0)
            e8_wr = e8_met.get("win_rate", 0)

            change = ((e8_cap / e7_cap - 1) * 100) if e7_cap > 0 else 0
            marker = "↑" if change > 0 else ("↓" if change < 0 else "→")
            print(f"  ${bp:.2f}   ${e7_cap:>14,.0f} ${e8_cap:>14,.0f} {marker}{abs(change):>7.1f}% "
                  f"{e7_wr:>7.3f} {e8_wr:>7.3f}")

    print("=" * 140)
    return combos


def save_results(combos: List[Dict], buy_price: float, output_dir: Path):
    """保存排名到 CSV。"""
    price_tag = f"bp{buy_price:.3f}".replace(".", "")
    output_csv = output_dir / f"fair_ranking_{price_tag}.csv"
    df = pd.DataFrame(combos)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\n  📁 排名已保存: {output_csv}")
    return output_csv


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def run_from_models_backtest(
    buy_price: float,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    date_from: str = DEFAULT_DATE_FROM,
    date_to: str = DEFAULT_DATE_TO,
) -> Tuple[List[Dict[str, Any]], str, str]:
    """
    金标准模式：从模型文件重新加载、重新预测、统一回测。
    调用 backtest_gru_regime_report 的基础设施。

    返回: (结果列表, 实际起始日, 实际结束日)
    """
    from scripts.backtest_simulation import run_backtest_for_pair
    from src.python.predictor import load_predictor, _find_symbol_timeframe_model
    from scripts.backtest_gru_regime import (
        get_backtest_df_one_asset, run_trading_loop, get_device,
    )

    data_src = PROJECT_ROOT / "data" / "raw"

    # 用 v5 测试集日期作为统一回测窗口（确保所有模型在同一时间段内比较）
    after_date = date_from
    year_days = (pd.Timestamp(date_to, tz="UTC") - pd.Timestamp(date_from, tz="UTC")).days
    end_date = date_to
    device = get_device(use_mps=True)

    print(f"  回测窗口: {after_date} ~ {end_date} ({year_days} 天)")
    print(f"  买入价格: ${buy_price:.3f}")
    print(f"  初始资金: ${initial_capital:.0f}")

    results: List[Dict[str, Any]] = []

    # ─── 1) 5 旧模型 ───────────────────────────────────────
    from scripts.backtest_gru_regime_report import OLD_POLYMARKET_COMBOS
    TIMEFRAME = "15m"

    print(f"\n  📊 加载旧模型 5 组合...")
    for combo in OLD_POLYMARKET_COMBOS:
        md = combo["models_dir"]
        models_dir = (PROJECT_ROOT / md) if not Path(md).is_absolute() else Path(md)
        model_dir = _find_symbol_timeframe_model(combo["symbol"], TIMEFRAME, models_root=models_dir)
        if not model_dir:
            print(f"    ⚠️ 未找到 {combo['symbol']} {combo['name']}，跳过")
            continue
        try:
            model, feats, meta = load_predictor(model_dir)
        except Exception as e:
            print(f"    ❌ {combo['name']}: 加载失败 {e}")
            continue

        sym_short = combo["symbol"].replace("/USDT", "")
        thr_str = (f"{combo['prob_threshold']}" if combo.get("prob_threshold") is not None
                   else f"{int((combo.get('prob_threshold_down') or 0)*100)}-{int((combo.get('prob_threshold_up') or 0)*100)}")
        filter_str = "有动态" if combo["enable_dynamic"] else "无动态"

        try:
            res = run_backtest_for_pair(
                combo["symbol"], TIMEFRAME, model, feats,
                initial_capital=initial_capital, bet_ratio=0.05, test_days=year_days,
                prob_threshold=combo.get("prob_threshold"),
                prob_threshold_up=combo.get("prob_threshold_up"),
                prob_threshold_down=combo.get("prob_threshold_down"),
                train_cutoff_date=after_date, end_date=end_date,
                enable_dynamic_bet_ratio=combo["enable_dynamic"],
                order_price=buy_price, use_fixed_bet=False, use_smart_bet=True,
                calibrator=None, calibration_method=None,
            )
        except Exception as e:
            print(f"    ❌ {combo['name']} {filter_str}: {e}")
            continue

        if res.get("error"):
            continue

        total_trades = res["total_trades"]
        wins = res["wins"]
        results.append({
            "排名": 0,
            "组合": f"old_{combo['name']}",
            "组合ID": combo["name"][:20],
            "币种": sym_short,
            "下注方式": f"智能({thr_str}){filter_str}",
            "最终资金": round(res["final_capital"], 2),
            "胜率%": round(wins / total_trades * 100, 1) if total_trades > 0 else 0,
            "盈亏%": round(res["profit_pct"], 2),
            "最大回撤%": round(max(0, min(100, res.get("max_drawdown", 0))), 2),
            "交易次数": total_trades,
            "最低/初始%": round(max(0, min(1000, res.get("min_over_initial_pct", 100))), 2),
            "来源": "模型回测(旧)",
        })
        print(f"    ✅ {combo['name']}: ${res['final_capital']:.2f} (胜率{wins}/{total_trades}={wins/max(1,total_trades)*100:.0f}%)")

    # ─── 2) 8 GRU 模型组合 ─────────────────────────────────
    from scripts.backtest_gru_regime_report import SIM_13_GRU_COMBOS
    models_best_no1h4h = PROJECT_ROOT / "experiments" / "gru_regime_v1" / "outputs" / "models_best_no1h4h"

    print(f"\n  📊 加载 GRU 模型 {len(SIM_13_GRU_COMBOS)} 组合...")
    for g in SIM_13_GRU_COMBOS:
        asset = g["asset"]
        models_override = models_best_no1h4h if g.get("use_no1h4h") and models_best_no1h4h.exists() else None
        try:
            df = get_backtest_df_one_asset(
                asset, test_days=year_days, device=device, data_src=data_src,
                after_date=after_date, end_date=end_date,
                models_best_override=models_override,
            )
        except Exception as e:
            print(f"    ❌ {g['combo_name']}: {e}")
            continue

        if len(df) < 10:
            print(f"    ⚠️ {g['combo_name']}: 数据不足 ({len(df)} bars)")
            continue

        symbol = asset.replace("_USDT", "")
        thr_str = f"{g['threshold']:.2f}"
        filter_str = "有动态" if g["enable_dynamic"] else "无动态"

        res = run_trading_loop(
            df, initial_capital=initial_capital, bet_ratio=0.05,
            prob_threshold=g["threshold"],
            enable_dynamic_bet_ratio=g["enable_dynamic"],
            order_price=buy_price, stop_loss_exit_price=None,
            use_fixed_bet=False, use_smart_bet=True,
            max_trades_per_day=0,
        )

        total_trades = res["total_trades"]
        wins = res["wins"]
        no1h4h_tag = "(无1h4h)" if g.get("use_no1h4h") else ""
        results.append({
            "排名": 0,
            "组合": f"gru_{g['combo_name']}",
            "组合ID": g["combo_name"][:20],
            "币种": symbol,
            "下注方式": f"GRU({thr_str}){filter_str}{no1h4h_tag}",
            "最终资金": round(res["final_capital"], 2),
            "胜率%": round(wins / total_trades * 100, 1) if total_trades > 0 else 0,
            "盈亏%": round(res["profit_pct"], 2),
            "最大回撤%": round(max(0, min(100, res.get("max_drawdown", 0))), 2),
            "交易次数": total_trades,
            "最低/初始%": round(max(0, min(1000, res.get("min_over_initial_pct", 100))), 2),
            "来源": "模型回测(GRU)",
        })
        print(f"    ✅ {g['combo_name']}: ${res['final_capital']:.2f} (胜率{wins}/{total_trades}={wins/max(1,total_trades)*100:.0f}%)")

    return results, after_date, end_date


def generate_model_predictions(
    model_dirs: List[Path],
    date_from: str = DEFAULT_DATE_FROM,
    date_to: str = DEFAULT_DATE_TO,
    regenerate: bool = False,
    test_days: int | None = None,
) -> Dict[str, pd.DataFrame]:
    """为多个模型生成测试集预测（共享特征构建，仅模型推理不同）。

    核心思路:
      1. 特征构建（GRU 嵌入 + 技术指标 + 情感特征）只做一次
      2. 对每个模型目录，加载该模型的 3 个 LightGBM 集成模型
      3. 用相同的测试特征分别推理，产出各自的预测 DataFrame
      4. 缓存到 RESULTS_DIR/{model_name}_test_predictions.parquet

    返回: {model_name: DataFrame(timestamp, asset, proba_up, actual, log_return)}
    """
    from experiments.sentiment_grid_search.run_grid import (
        DEFAULT_DAYS, WARMUP_DAYS, TEST_DAYS as RG_TEST_DAYS, VAL_DAYS,
        GRU_FEATURE_COLS,
        build_tech_features, extract_embeddings,
        _prepare_asset_for_pooling, get_data_paths,
    )
    from experiments.sentiment_grid_search.data_prep import (
        _ensure_datetime_col, load_ohlcv, merge_funding_rate,
    )
    from experiments.gru_regime_v1.src.utils import get_device

    # ── 检测哪些模型需要噪声特征（必须在缓存检查前，缓存文件名依赖此结果）──
    noise_models = set()
    for md in model_dirs:
        with open(md / "config.json") as f:
            mcfg = json.load(f)
        if mcfg.get("sim_noise_augment", False):
            noise_models.add(md.name)

    # 检查缓存: 如果所有模型都有缓存且不强制刷新，直接返回
    # 噪声模型和干净模型用不同缓存文件名
    all_cached = True
    for md in model_dirs:
        is_noisy = md.name in noise_models
        suffix = "_noisy_test_predictions.parquet" if is_noisy else "_test_predictions.parquet"
        cache_path = RESULTS_DIR / f"{md.name}{suffix}"
        if not cache_path.exists() or regenerate:
            all_cached = False
            break

    if all_cached:
        results: Dict[str, pd.DataFrame] = {}
        for md in model_dirs:
            is_noisy = md.name in noise_models
            suffix = "_noisy_test_predictions.parquet" if is_noisy else "_test_predictions.parquet"
            cache_path = RESULTS_DIR / f"{md.name}{suffix}"
            pred_df = pd.read_parquet(cache_path)
            results[md.name] = pred_df
            feat_tag = "噪声T-120s" if is_noisy else "干净T+0"
            acc = ((pred_df["proba_up"].values >= 0.5).astype(int) == pred_df["actual"].values).mean()
            print(f"  缓存命中: {md.name} ({feat_tag}) → {len(pred_df)} bars, accuracy={acc:.4f}")
        return results

    if noise_models:
        print(f"  噪声模型 (T-120s 特征): {', '.join(noise_models)}")
        print(f"  干净模型 (T+0 特征): {', '.join(md.name for md in model_dirs if md.name not in noise_models)}")

    # 测试集天数：未指定则用 run_grid 的 15 天；指定则用历史 K 线拉长（融合超参建议 60～90 天）
    use_test_days = test_days if test_days is not None else RG_TEST_DAYS

    # ── 内部函数: 构建测试特征 ──
    def _build_test_features(apply_noise: bool = False, label: str = "clean"):
        """构建测试特征集。apply_noise=True 时在 close/high/low/volume 上加 T-120s 噪声。"""
        with open(model_dirs[0] / "config.json") as f:
            config = json.load(f)

        device = get_device(use_mps=True)
        paths = get_data_paths()
        total_window = DEFAULT_DAYS + WARMUP_DAYS + VAL_DAYS + max(RG_TEST_DAYS, use_test_days)
        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=total_window)
        gru_model_dir = PROJECT_ROOT / "experiments" / "sentiment_grid_search" / "models"

        print(f"\n{'─' * 70}")
        noise_tag = " + T-120s 噪声" if apply_noise else ""
        print(f"  构建 {label} 测试特征（GRU + 技术 + 情感{noise_tag}）")
        print(f"{'─' * 70}")

        asset_data = {}
        clean_closes = {}
        for idx, asset in enumerate(sorted(config["asset_map"].keys())):
            print(f"  {asset}:", end=" ", flush=True)
            tech_df = build_tech_features(paths["data_src"], asset)
            tech_df = _ensure_datetime_col(tech_df, "timestamp")
            tech_df = tech_df[tech_df["timestamp"] >= cutoff_date].reset_index(drop=True)

            if apply_noise:
                clean_closes[asset] = tech_df[["timestamp", "close"]].copy()
                tech_df = _apply_simulated_candle_noise(tech_df, seed=_NOISE_SEED_BASE + idx)
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
            print(f"{len(tech_df)} rows")

        # 合池
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

        # 取测试集（use_test_days 由外层传入，可拉长以生成更长 parquet 供融合超参用）
        latest_ts = pooled["timestamp"].max()
        test_cutoff = latest_ts - pd.Timedelta(days=use_test_days)
        tdf = pooled[pooled["timestamp"] >= test_cutoff].copy()

        # 噪声模式: 用真实 close 修正 direction_label
        if apply_noise and clean_closes:
            _fix_direction_labels(tdf, clean_closes)
            print(f"  ✅ 标签已修正为真实方向（{len(clean_closes)} 币种）")

        print(f"  测试集: {len(tdf)} rows, "
              f"{tdf['timestamp'].min()} ~ {tdf['timestamp'].max()}")
        return tdf

    # ── Phase 0: 构建测试特征 ──
    # 干净特征（给 v5_production 等常规模型用）
    clean_models = [md for md in model_dirs if md.name not in noise_models]
    noisy_model_list = [md for md in model_dirs if md.name in noise_models]

    clean_test_df = None
    noisy_test_df = None

    if clean_models:
        clean_test_df = _build_test_features(apply_noise=False, label="clean")
    if noisy_model_list:
        noisy_test_df = _build_test_features(apply_noise=True, label="noisy")

    # ── Phase 1: 每个模型分别推理 ──
    results: Dict[str, pd.DataFrame] = {}
    for model_dir in model_dirs:
        model_name = model_dir.name
        is_noisy = model_name in noise_models
        test_df = noisy_test_df if is_noisy else clean_test_df
        feat_label = "噪声T-120s" if is_noisy else "干净T+0"
        print(f"\n  模型推理: {model_name} ({feat_label} 特征)")

        with open(model_dir / "config.json") as f:
            mcfg = json.load(f)
        with open(model_dir / "feature_cols.json") as f:
            feature_cols = json.load(f)

        models = []
        for w_days in mcfg["window_days_list"]:
            models.append(joblib.load(model_dir / f"lgb_{w_days}d.joblib"))

        # 确保特征列存在
        for c in feature_cols:
            if c not in test_df.columns:
                test_df[c] = 0

        X_te = test_df[feature_cols]

        # 3 模型集成预测
        probas_list = [m.predict_proba(X_te)[:, 1] for m in models]
        ensemble_proba = np.mean(probas_list, axis=0)

        pred_df = pd.DataFrame({
            "timestamp": test_df["timestamp"].values,
            "asset": test_df["_asset_name"].values,
            "proba_up": ensemble_proba,
            "actual": test_df["direction_label"].values.astype(int),
            "log_return": test_df["log_return"].values if "log_return" in test_df.columns else 0,
        })

        # 缓存（噪声 vs 干净用不同文件名）
        suffix = "_noisy_test_predictions.parquet" if is_noisy else "_test_predictions.parquet"
        cache_path = RESULTS_DIR / f"{model_name}{suffix}"
        pred_df.to_parquet(cache_path, index=False)
        results[model_name] = pred_df

        acc = ((ensemble_proba >= 0.5).astype(int) == pred_df["actual"].values).mean()
        print(f"    {len(pred_df)} bars, accuracy={acc:.4f}")
        print(f"    特征类型: {feat_label} | 标签: 真实方向")
        print(f"    已缓存: {cache_path.name}")

    return results


def print_model_comparison(
    all_model_results: Dict[str, List[Dict]],
    buy_price: float,
):
    """打印多模型头对头对比摘要。"""
    if len(all_model_results) < 2:
        return

    model_names = list(all_model_results.keys())
    odds = (1.0 - buy_price) / buy_price

    print(f"\n{'═' * 100}")
    print(f"  📊 模型头对头对比 @ ${buy_price:.3f} (赔率 {odds:.4f})")
    print(f"{'═' * 100}")

    # 汇总每个模型的 ALL 结果
    for mname in model_names:
        combos = all_model_results[mname]
        all_combos = [c for c in combos if "_ALL" in c.get("组合", "") and "ladder" not in c.get("组合", "")]
        if not all_combos:
            continue

        avg_ret = np.mean([c["盈亏%"] for c in all_combos])
        avg_wr = np.mean([c["胜率%"] for c in all_combos])
        avg_dd = np.mean([c["最大回撤%"] for c in all_combos])
        best = max(all_combos, key=lambda x: x["最终资金"])

        print(f"\n  【{mname}】")
        print(f"    固定买价策略数: {len(all_combos)}")
        print(f"    平均盈亏: {avg_ret:+.1f}%  |  平均胜率: {avg_wr:.1f}%  |  平均回撤: {avg_dd:.1f}%")
        print(f"    最佳策略: {best['组合']} → ${best['最终资金']:,.2f} "
              f"(胜率{best['胜率%']:.1f}%, 盈亏{best['盈亏%']:+.1f}%)")

    # 逐买价对比
    import re as _re
    print(f"\n  {'买价':<10}", end="")
    for mname in model_names:
        print(f"  {mname + ' 资金':>18} {mname + ' 胜率':>10}", end="")
    print()
    print(f"  {'─' * (10 + 30 * len(model_names))}")

    # 配对对比 — 用纯买价 tag（bp0450, bp_dyn_0450_0530 等）统一匹配
    def _extract_bp_tag(combo_name: str) -> str | None:
        """从组合名中提取买价 tag，如 bp0450 或 bp_dyn_0450_0530。"""
        m = _re.search(r'(bp_dyn_\d+_\d+|bp\d+)', combo_name)
        return m.group(1) if m else None

    price_results = {}  # {bp_tag: {exp_label: result}}
    for mname in model_names:
        for c in all_model_results[mname]:
            combo = c.get("组合", "")
            if "_ALL" not in combo or "ladder" in combo:
                continue
            bp_tag = _extract_bp_tag(combo)
            if bp_tag:
                price_results.setdefault(bp_tag, {})[mname] = c

    for tag in sorted(price_results.keys()):
        # 格式化买价显示
        if tag.startswith("bp_dyn"):
            bp_str = tag.replace("bp_dyn_", "dyn $0.").replace("_", "~0.")
        elif tag.startswith("bp"):
            cents = tag.replace("bp0", "0.").replace("bp", "")
            bp_str = f"${cents[:4]}" if len(cents) >= 4 else tag
        else:
            bp_str = tag
        print(f"  {bp_str:<10}", end="")
        values = {}
        for mname in model_names:
            c = price_results[tag].get(mname)
            if c:
                values[mname] = c["最终资金"]
                print(f"  ${c['最终资金']:>16,.2f} {c['胜率%']:>8.1f}%", end="")
            else:
                print(f"  {'—':>18} {'—':>10}", end="")
        print()

    print(f"{'═' * 100}")


def main():
    parser = argparse.ArgumentParser(description="公平对比 — 所有模型统一环境回测")
    parser.add_argument("--buy-prices", type=str, default="0.50",
                        help="逗号分隔的买入价格列表（用于旧模型对比, v5 各自用自己的买价）")
    parser.add_argument("--date-from", type=str, default=DEFAULT_DATE_FROM,
                        help=f"回测起始日期（默认 {DEFAULT_DATE_FROM}）")
    parser.add_argument("--date-to", type=str, default=DEFAULT_DATE_TO,
                        help=f"回测结束日期（默认 {DEFAULT_DATE_TO}）")
    parser.add_argument("--capital", type=float, default=DEFAULT_INITIAL_CAPITAL,
                        help=f"初始资金（默认 ${DEFAULT_INITIAL_CAPITAL}）")
    parser.add_argument("--from-models", action="store_true",
                        help="金标准模式：所有旧模型从 parquet 原始数据重新预测（最公平，耗时 ~20 分钟）")
    parser.add_argument("--rules-version", type=str, default="v3",
                        choices=["v3", "v4"],
                        help="v5 Exp8 超参规则版本（v3=默认, v4=0.3~0.7 bet_pct 搜索范围）")
    parser.add_argument("--model-dirs", type=str, nargs="+", default=None,
                        help="模型目录名（在 data/models/ 下），可多个。"
                             "例: --model-dirs v5_production v5_production_sim_noise")
    parser.add_argument("--rules-list", type=str, nargs="+", default=None,
                        help="每个模型目录对应的超参版本，空格分隔，与 --model-dirs 一一对应。"
                             "例: --model-dirs v5_production v5_production v5_production_sim_noise "
                             "--rules-list v3 v4 v3 → Exp8(v3) + Exp9(v4) + Exp10(v3)")
    parser.add_argument("--regenerate", action="store_true",
                        help="强制重新生成预测（忽略缓存）")
    parser.add_argument("--test-days", type=int, default=None,
                        help="生成 parquet 时使用的测试集天数（默认 15）。设 60～90 可得到更长 parquet 供融合超参用")
    args = parser.parse_args()

    buy_prices = [float(p.strip()) for p in args.buy_prices.split(",")]

    # ── 多模型对比模式 ──
    model_predictions: Dict[str, pd.DataFrame] = {}
    if args.model_dirs:
        model_dirs = []
        for d in args.model_dirs:
            p = Path(d)
            if not p.is_absolute():
                p = PROJECT_ROOT / "data" / "models" / d
            if not p.exists():
                print(f"  ❌ 模型目录不存在: {p}")
                sys.exit(1)
            model_dirs.append(p)

        # 构建每个模型的 rules_version（支持 --rules-list 一一对应）
        if args.rules_list:
            if len(args.rules_list) != len(model_dirs):
                print(f"  ❌ --rules-list 数量({len(args.rules_list)})必须与 --model-dirs({len(model_dirs)})一致")
                sys.exit(1)
            rules_per_model = args.rules_list
        else:
            rules_per_model = [args.rules_version] * len(model_dirs)

        # 构建实验配置列表: (model_dir, rules_version, exp_label)
        exp_configs = []
        for md, rv in zip(model_dirs, rules_per_model):
            el = _get_exp_label(md.name, rv)
            exp_configs.append((md, rv, el))

        # 去重模型目录用于预测生成（同一模型只需预测一次）
        unique_model_dirs = list({md.name: md for md in model_dirs}.values())

        print("\n" + "█" * 70)
        print("  公平对比系统 — 多模型统一环境回测")
        print("█" * 70)
        print(f"  模式: 多模型对比（{len(exp_configs)} 组实验, {len(unique_model_dirs)} 个模型）")
        for md, rv, el in exp_configs:
            print(f"    - {el}: {md.name} + {rv} rules")
        print(f"  日期范围: {args.date_from} ~ {args.date_to}")
        print(f"  买入价格: {', '.join(f'${p:.3f}' for p in buy_prices)}")
        print(f"  初始资金: ${args.capital:.0f}")
        print(f"  手续费+滑点: {TOTAL_COST*100:.1f}%")

        # 生成所有模型的预测（共享特征构建，去重避免重复推理）
        model_predictions = generate_model_predictions(
            unique_model_dirs,
            date_from=args.date_from,
            date_to=args.date_to,
            regenerate=args.regenerate,
            test_days=args.test_days,
        )

        all_csvs = []

        for bp in buy_prices:
            odds = (1.0 - bp) / bp
            print(f"\n\n{'▓' * 70}")
            print(f"  买价 ${bp:.3f} | 赔率 {odds:.4f}")
            print(f"{'▓' * 70}")

            all_combos = []
            per_model_results: Dict[str, List[Dict]] = {}

            for md, rv, el in exp_configs:
                model_name = md.name
                pred_df = model_predictions.get(model_name)
                if pred_df is None:
                    print(f"  ⚠️ {model_name} 无预测数据，跳过")
                    continue
                print(f"\n🧪 {el} ({model_name}) 回测 (规则={rv}) @ ${bp:.3f}...")
                v5_results = run_v5_backtest_all(
                    buy_price=bp,
                    date_from=args.date_from,
                    date_to=args.date_to,
                    initial_capital=args.capital,
                    rules_version=rv,
                    predictions=pred_df,
                    model_label=model_name,
                    exp_label=el,
                )
                print(f"  {el} 回测结果: {len(v5_results)} 条")
                all_combos.extend(v5_results)
                per_model_results[el] = v5_results

            if all_combos:
                ranked = print_ranking(all_combos, bp, args.date_from, args.date_to)
                csv_path = save_results(ranked, bp, RESULTS_DIR)
                all_csvs.append(csv_path)

            # 头对头对比
            print_model_comparison(per_model_results, bp)

        # 最终汇总
        print(f"\n\n{'█' * 70}")
        print("  ✅ 多模型公平对比完成！")
        print(f"{'█' * 70}")
        for csv_path in all_csvs:
            print(f"  📁 {csv_path}")

    else:
        # ── 原有模式（单模型 + 旧模型对比）──
        mode_str = "金标准（模型重新预测）" if args.from_models else "快速（模拟交易记录 + 统一买价）"
        print("\n" + "█" * 70)
        print("  公平对比系统 — 所有模型统一环境回测")
        print("█" * 70)
        print(f"  模式: {mode_str}")
        if not args.from_models:
            print(f"  日期范围: {args.date_from} ~ {args.date_to}")
        else:
            print(f"  日期范围: 自动（无泄漏起始日 + 365 天）")
        print(f"  买入价格: {', '.join(f'${p:.3f}' for p in buy_prices)}")
        print(f"  初始资金: ${args.capital:.0f}")
        print(f"  手续费+滑点: {TOTAL_COST*100:.1f}%")
        print(f"  超参版本: {args.rules_version}" + (" (0.3~0.7 bet_pct)" if args.rules_version == "v4" else " (默认)"))

        all_csvs = []

        for bp in buy_prices:
            odds = (1.0 - bp) / bp
            print(f"\n\n{'▓' * 70}")
            print(f"  买价 ${bp:.3f} | 赔率 {odds:.4f}")
            print(f"{'▓' * 70}")

            if args.from_models:
                # ─── 金标准模式：所有旧模型从 parquet 重新预测 ───
                print(f"\n📋 从模型文件重新预测（买价 ${bp:.3f}）...")
                old_results, actual_from, actual_to = run_from_models_backtest(
                    buy_price=bp,
                    initial_capital=args.capital,
                    date_from=args.date_from,
                    date_to=args.date_to,
                )
                print(f"  模型回测结果: {len(old_results)} 条")

                date_label_from = actual_from
                date_label_to = actual_to
            else:
                # ─── 快速模式：用 prediction_trades.json 重模拟 ───
                print(f"\n📋 重模拟旧模型组合（买价 ${bp:.3f}）...")
                old_results = resimulate_all_old_combos(
                    buy_price=bp,
                    date_from=args.date_from,
                    date_to=args.date_to,
                    initial_capital=args.capital,
                )
                old_std = [r for r in old_results if "超参" not in r["来源"]]
                old_hp = [r for r in old_results if "超参" in r["来源"]]
                print(f"  找到 {len(old_std)} 个标准组合 + {len(old_hp)} 个超参组合")
                date_label_from = args.date_from
                date_label_to = args.date_to

            # v5 Exp8 新模型回测（两种模式都做）
            print(f"\n🧪 v5 Exp8 新模型回测 (含 target PM 特征, 规则={args.rules_version}) @ ${bp:.3f}...")
            v5_results = run_v5_backtest_all(
                buy_price=bp,
                date_from=args.date_from,
                date_to=args.date_to,
                initial_capital=args.capital,
                rules_version=args.rules_version,
            )
            print(f"  v5 Exp8 回测结果: {len(v5_results)} 条")

            # 合并 + 排名
            all_combos = old_results + v5_results
            if all_combos:
                ranked = print_ranking(all_combos, bp, str(date_label_from), str(date_label_to))
                csv_path = save_results(ranked, bp, RESULTS_DIR)
                all_csvs.append(csv_path)

        # 最终汇总
        print(f"\n\n{'█' * 70}")
        print("  ✅ 公平对比完成！")
        print(f"{'█' * 70}")
        for csv_path in all_csvs:
            print(f"  📁 {csv_path}")


if __name__ == "__main__":
    main()
