#!/usr/bin/env python3
"""
与 polymarket/aggregate_avg_buy_price.py 的 RANKING_COMBOS 完全一致：92 个组合的回测规格。
用于 optuna_high_vol 等脚本，保证「币种+模型组合」与 aggregate 排名一致。

每个 spec: dict with log_dir, kind ("gru_ek" | "v5_exp" | "ensemble"), model_dir(或 None),
  symbols[], order_price, initial_cap_per_coin, 以及 kind 专用字段。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 与 polymarket/aggregate_avg_buy_price.py 的 RANKING_COMBOS 完全一致（92 个）
try:
    from polymarket.aggregate_avg_buy_price import RANKING_COMBOS
except Exception:
    RANKING_COMBOS = []

MODELS_DIR = PROJECT_ROOT / "data" / "models"


def _bp_to_float(log_dir: str) -> float:
    m = re.search(r"bp0?(\d{3})", log_dir)
    if m:
        return int(m.group(1)) / 1000.0
    if "dyn_0450_0530" in log_dir or "dyn_0480_0510" in log_dir:
        return 0.50
    return 0.50


EXP_MODEL_DIR = {
    10: "v5_production_sim_noise",
    11: "v5_production_sim_noise",
    13: "v5_production_tv",
    14: "v5_production_sim_noise_tv",
    15: "v5_production_365d",
    16: "v5_production_tv_365d",
    17: "v5_production_no_target_pm",
}


def get_ranking_backtest_specs() -> List[Dict[str, Any]]:
    """返回与 aggregate 92 个排名一致的 backtest 规格列表。"""
    specs = []
    for log_dir in RANKING_COMBOS:
        if log_dir == "logs_gru_eth_ek":
            specs.append({
                "log_dir": log_dir, "kind": "gru_ek", "asset": "ETH_USDT",
                "threshold": 0.54, "enable_dynamic": True, "use_no1h4h": False,
                "symbols": ["ETH"], "order_price": 0.46, "initial_cap_per_coin": 400.0,
            })
        elif log_dir == "logs_gru_eth_no1h4h_ek":
            specs.append({
                "log_dir": log_dir, "kind": "gru_ek", "asset": "ETH_USDT",
                "threshold": 0.54, "enable_dynamic": False, "use_no1h4h": True,
                "symbols": ["ETH"], "order_price": 0.46, "initial_cap_per_coin": 400.0,
            })
        elif log_dir == "logs_gru_btc_no1h4h_ek":
            specs.append({
                "log_dir": log_dir, "kind": "gru_ek", "asset": "BTC_USDT",
                "threshold": 0.57, "enable_dynamic": False, "use_no1h4h": True,
                "symbols": ["BTC"], "order_price": 0.46, "initial_cap_per_coin": 400.0,
            })
        elif log_dir == "logs_gru_sol_ek":
            specs.append({
                "log_dir": log_dir, "kind": "gru_ek", "asset": "SOL_USDT",
                "threshold": 0.52, "enable_dynamic": False, "use_no1h4h": False,
                "symbols": ["SOL"], "order_price": 0.46, "initial_cap_per_coin": 400.0,
            })
        elif log_dir.startswith("logs_ensemble_"):
            specs.append({
                "log_dir": log_dir, "kind": "ensemble", "model_dir": None,
                "symbols": ["ETH", "BTC"], "order_price": _bp_to_float(log_dir),
                "initial_cap_per_coin": 400.0,
            })
        elif "logs_v5_exp" in log_dir:
            m = re.search(r"exp(1[0-7])", log_dir)
            exp_num = int(m.group(1)) if m else 13
            model_dir = EXP_MODEL_DIR.get(exp_num, "v5_production_tv")
            specs.append({
                "log_dir": log_dir, "kind": "v5_exp", "model_dir": model_dir,
                "symbols": ["ETH", "BTC"], "order_price": _bp_to_float(log_dir),
                "initial_cap_per_coin": 400.0,
            })
    return specs


def get_backtestable_specs(skip_ensemble: bool = True) -> List[Dict[str, Any]]:
    """可回测规格（默认跳过 ensemble）。"""
    all_specs = get_ranking_backtest_specs()
    if skip_ensemble:
        return [s for s in all_specs if s["kind"] != "ensemble"]
    return all_specs


def get_backtestable_specs_eth_btc_only(skip_ensemble: bool = True) -> List[Dict[str, Any]]:
    """仅 ETH+BTC 可回测规格（排除 SOL、XRP），用于高波超参仅联动 ETH+BTC。"""
    specs = get_backtestable_specs(skip_ensemble=skip_ensemble)
    return [
        s for s in specs
        if s["log_dir"] != "logs_gru_sol_ek"
        and "XRP" not in (s.get("asset") or "")
        and "XRP" not in (s.get("symbols") or [])
    ]
