"""
Orderbook 微结构特征工程（步骤 6.2）

从 L2 深度快照中提取 10 个核心特征：
1. bid_ask_imbalance   - 买卖量不平衡
2. micro_price         - 微观价格
3. depth_ratio         - 深度集中度
4. orderbook_pressure  - 订单簿压力
5. weighted_mid_dev    - 加权中间价偏离
6. spread_ratio        - 买卖价差比
7. bid_ask_skew        - 买卖量偏度差
8. vwap_distance       - VWAP 距离
9. depth_imbalance_ma  - 不平衡移动平均
10. order_flow_toxicity - 订单流毒性

数据源：Bybit L2 深度快照（500 档、10ms）
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any


def extract_orderbook_features(
    bids: List[List[float]],
    asks: List[List[float]],
    last_price: Optional[float] = None,
    max_levels: int = 20,
) -> Dict[str, float]:
    """
    从单个 orderbook 快照提取 10 个特征。

    参数：
        bids: [[price, volume], ...] 买方档位（从高到低）
        asks: [[price, volume], ...] 卖方档位（从低到高）
        last_price: 最新成交价（用于 vwap_distance）
        max_levels: 使用的最大档位数

    返回：
        dict of 10 features
    """
    bids = bids[:max_levels] if bids else [[0, 0]]
    asks = asks[:max_levels] if asks else [[0, 0]]

    bid_prices = np.array([b[0] for b in bids], dtype=float)
    bid_volumes = np.array([b[1] for b in bids], dtype=float)
    ask_prices = np.array([a[0] for a in asks], dtype=float)
    ask_volumes = np.array([a[1] for a in asks], dtype=float)

    best_bid = bid_prices[0] if len(bid_prices) > 0 else 0
    best_ask = ask_prices[0] if len(ask_prices) > 0 else 0
    mid_price = (best_bid + best_ask) / 2 if (best_bid + best_ask) > 0 else 1

    sum_bid_vol = bid_volumes.sum()
    sum_ask_vol = ask_volumes.sum()
    total_vol = sum_bid_vol + sum_ask_vol

    features = {}

    # 1. bid_ask_imbalance (买卖量不平衡)
    features["ob_bid_ask_imbalance"] = (
        (sum_bid_vol - sum_ask_vol) / total_vol if total_vol > 0 else 0
    )

    # 2. micro_price (微观价格)
    best_bid_vol = bid_volumes[0] if len(bid_volumes) > 0 else 0
    best_ask_vol = ask_volumes[0] if len(ask_volumes) > 0 else 0
    denom = best_bid_vol + best_ask_vol
    features["ob_micro_price"] = (
        (best_bid * best_ask_vol + best_ask * best_bid_vol) / denom
        if denom > 0 else mid_price
    )

    # 3. depth_ratio (深度集中度: 前5档占总量比)
    top5_vol = bid_volumes[:5].sum() + ask_volumes[:5].sum()
    features["ob_depth_ratio"] = top5_vol / total_vol if total_vol > 0 else 0

    # 4. orderbook_pressure (订单簿压力)
    w_bid = np.average(bid_prices, weights=bid_volumes) if sum_bid_vol > 0 else best_bid
    w_ask = np.average(ask_prices, weights=ask_volumes) if sum_ask_vol > 0 else best_ask
    features["ob_pressure"] = (w_bid - w_ask) / mid_price if mid_price > 0 else 0

    # 5. weighted_mid_deviation (加权中间价偏离)
    features["ob_weighted_mid_dev"] = (
        abs(features["ob_micro_price"] - mid_price) / mid_price if mid_price > 0 else 0
    )

    # 6. spread_ratio (买卖价差比)
    features["ob_spread_ratio"] = (
        (best_ask - best_bid) / mid_price if mid_price > 0 else 0
    )

    # 7. bid_ask_skew (买卖量偏度差)
    from scipy.stats import skew as _skew
    bid_skew = float(_skew(bid_volumes)) if len(bid_volumes) > 2 else 0
    ask_skew = float(_skew(ask_volumes)) if len(ask_volumes) > 2 else 0
    features["ob_bid_ask_skew"] = bid_skew - ask_skew

    # 8. vwap_distance (VWAP 距离)
    all_prices = np.concatenate([bid_prices, ask_prices])
    all_vols = np.concatenate([bid_volumes, ask_volumes])
    vwap = np.average(all_prices, weights=all_vols) if all_vols.sum() > 0 else mid_price
    lp = last_price if last_price is not None else mid_price
    features["ob_vwap_distance"] = (lp - vwap) / lp if lp > 0 else 0

    # 9. depth_imbalance_ma (不平衡 - 无历史时退化为当前值，聚合时再做 MA)
    features["ob_depth_imbalance"] = features["ob_bid_ask_imbalance"]

    # 10. order_flow_toxicity (需价格变化+签名量，快照级退化为 imbalance 方向)
    features["ob_flow_toxicity"] = features["ob_bid_ask_imbalance"] * features["ob_spread_ratio"]

    return features


def aggregate_ob_features_to_bar(
    snapshots: List[Dict[str, float]],
) -> Dict[str, float]:
    """
    将多个快照的 OB 特征聚合为一根 K 线（如 15m）的统计量。

    对每个特征计算 5 个统计量：mean, std, min, max, last
    10 特征 × 5 统计量 = 50 列

    参数：
        snapshots: 该 K 线内所有快照的特征列表

    返回：
        dict of 50 features
    """
    if not snapshots:
        return {}

    feature_keys = list(snapshots[0].keys())
    result = {}

    for key in feature_keys:
        values = [s.get(key, 0) for s in snapshots]
        arr = np.array(values, dtype=float)
        result[f"{key}_mean"] = float(np.mean(arr))
        result[f"{key}_std"] = float(np.std(arr)) if len(arr) > 1 else 0.0
        result[f"{key}_min"] = float(np.min(arr))
        result[f"{key}_max"] = float(np.max(arr))
        result[f"{key}_last"] = float(arr[-1])

    return result


def get_ob_feature_names(stats: List[str] = None) -> List[str]:
    """返回所有 OB 特征名列表（10 特征 × 5 统计量 = 50 个）。"""
    base_features = [
        "ob_bid_ask_imbalance", "ob_micro_price", "ob_depth_ratio",
        "ob_pressure", "ob_weighted_mid_dev", "ob_spread_ratio",
        "ob_bid_ask_skew", "ob_vwap_distance", "ob_depth_imbalance",
        "ob_flow_toxicity",
    ]
    if stats is None:
        stats = ["mean", "std", "min", "max", "last"]
    return [f"{feat}_{stat}" for feat in base_features for stat in stats]
