"""
交易摩擦模拟器（步骤 3）：在回测中模拟真实交易成本。

功能：
- 滑点模拟 (slippage)
- 订单簿深度冲击 (depth impact)
- 执行延迟 (execution delay)
- 部分成交 (partial fill)

可选使用 Bybit L2 历史快照做真实深度模拟。

用法：
    from src.python.backtesting.trading_friction import TradingFrictionSimulator
    sim = TradingFrictionSimulator()
    result = sim.simulate_execution(price=0.527, volume=100, side="BUY")
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List


def _load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """加载交易摩擦配置。"""
    defaults = {
        "default_slippage_ticks": 1.0,
        "tick_size": 0.001,
        "orderbook_depth_levels": 10,
        "execution_delay_mean": 2.0,
        "execution_delay_std": 1.0,
        "min_order_size_usd": 10.0,
        "fee_rate": 0.001,
    }
    if config_path and Path(config_path).exists():
        try:
            custom = json.loads(Path(config_path).read_text(encoding="utf-8"))
            defaults.update(custom)
        except Exception:
            pass
    return defaults


class TradingFrictionSimulator:
    """
    交易摩擦模拟器。

    参数：
        slippage_ticks: 滑点（单位 tick，默认 1.0）
        tick_size: 每 tick 的价格（默认 0.001）
        depth_levels: 模拟的订单簿档位数（默认 10）
        delay_mean: 执行延迟均值（秒，默认 2.0）
        delay_std: 执行延迟标准差（秒，默认 1.0）
        min_order_size: 最小下单量（USD，默认 10）
        fee_rate: 手续费率（默认 0.001）
        config_path: 配置文件路径
    """

    def __init__(
        self,
        slippage_ticks: Optional[float] = None,
        tick_size: Optional[float] = None,
        depth_levels: Optional[int] = None,
        delay_mean: Optional[float] = None,
        delay_std: Optional[float] = None,
        min_order_size: Optional[float] = None,
        fee_rate: Optional[float] = None,
        config_path: Optional[str] = None,
    ):
        cfg = _load_config(config_path)
        self.slippage_ticks = slippage_ticks if slippage_ticks is not None else cfg["default_slippage_ticks"]
        self.tick_size = tick_size if tick_size is not None else cfg["tick_size"]
        self.depth_levels = depth_levels if depth_levels is not None else cfg["orderbook_depth_levels"]
        self.delay_mean = delay_mean if delay_mean is not None else cfg["execution_delay_mean"]
        self.delay_std = delay_std if delay_std is not None else cfg["execution_delay_std"]
        self.min_order_size = min_order_size if min_order_size is not None else cfg["min_order_size_usd"]
        self.fee_rate = fee_rate if fee_rate is not None else cfg["fee_rate"]

        # 统计
        self._total_trades = 0
        self._total_slippage_cost = 0.0
        self._total_delay_cost = 0.0
        self._unfilled_count = 0
        self._partial_fill_count = 0

    def apply_slippage(self, best_price: float, side: str) -> float:
        """
        计算滑点后的实际成交价。

        BUY: 成交价 = best_price + slippage（买贵了）
        SELL: 成交价 = best_price - slippage（卖便宜了）
        """
        slippage = self.slippage_ticks * self.tick_size
        if side.upper() == "BUY":
            return best_price + slippage
        else:
            return best_price - slippage

    def simulate_depth_impact(
        self,
        volume_usd: float,
        orderbook_levels: Optional[List[List[float]]] = None,
    ) -> Dict[str, float]:
        """
        模拟订单簿深度冲击。

        如果下单量超过最优价位的挂单量，需要吃到更深的档位，
        导致加权平均成交价偏离最优价。

        参数：
            volume_usd: 下单金额（USD）
            orderbook_levels: [[price, volume_usd], ...] 各档位
                若不提供，使用指数衰减模拟

        返回：
            {
                "vwap": 加权平均成交价（相对最优价的偏移比例），
                "levels_used": 使用的档位数,
                "depth_cost_pct": 深度冲击成本占总金额的百分比,
            }
        """
        if orderbook_levels is not None and len(orderbook_levels) > 0:
            return self._depth_from_real_ob(volume_usd, orderbook_levels)
        return self._depth_simulated(volume_usd)

    def _depth_simulated(self, volume_usd: float) -> Dict[str, float]:
        """用指数衰减模拟订单簿深度。"""
        # 假设每档挂单量按指数衰减：level_0 最大，越深越少
        per_level = volume_usd / self.depth_levels * 2  # 假设第一档能承接总量的 20%
        remaining = volume_usd
        weighted_cost = 0.0
        levels_used = 0

        for i in range(self.depth_levels):
            level_volume = per_level * np.exp(-0.1 * i)
            fill = min(remaining, level_volume)
            price_offset = i * self.tick_size
            weighted_cost += fill * price_offset
            remaining -= fill
            levels_used += 1
            if remaining <= 0:
                break

        depth_cost_pct = (weighted_cost / volume_usd * 100) if volume_usd > 0 else 0.0
        return {
            "vwap_offset": weighted_cost / volume_usd if volume_usd > 0 else 0.0,
            "levels_used": levels_used,
            "depth_cost_pct": depth_cost_pct,
            "unfilled_usd": max(0, remaining),
        }

    def _depth_from_real_ob(
        self, volume_usd: float, levels: List[List[float]]
    ) -> Dict[str, float]:
        """用真实订单簿档位计算深度冲击。"""
        remaining = volume_usd
        total_cost = 0.0
        best_price = levels[0][0] if levels else 0
        levels_used = 0

        for price, qty_usd in levels[:self.depth_levels]:
            fill = min(remaining, qty_usd)
            total_cost += fill * price
            remaining -= fill
            levels_used += 1
            if remaining <= 0:
                break

        filled = volume_usd - remaining
        vwap = total_cost / filled if filled > 0 else best_price
        depth_cost_pct = abs(vwap - best_price) / best_price * 100 if best_price > 0 else 0

        return {
            "vwap_offset": vwap - best_price,
            "levels_used": levels_used,
            "depth_cost_pct": depth_cost_pct,
            "unfilled_usd": max(0, remaining),
        }

    def simulate_execution_delay(self) -> float:
        """模拟执行延迟（秒），用正态分布 clip 到 [0, +inf)。"""
        delay = max(0, np.random.normal(self.delay_mean, self.delay_std))
        return delay

    def simulate_execution(
        self,
        price: float,
        volume_usd: float,
        side: str = "BUY",
        orderbook_levels: Optional[List[List[float]]] = None,
    ) -> Dict[str, Any]:
        """
        模拟一笔交易的完整执行过程。

        参数：
            price: 最优价
            volume_usd: 下单金额
            side: "BUY" 或 "SELL"
            orderbook_levels: 真实订单簿（可选）

        返回：
            {
                "actual_price": 实际成交价,
                "filled_volume": 成交金额,
                "unfilled_volume": 未成交金额,
                "partial_fill": 是否部分成交,
                "execution_delay": 延迟秒数,
                "slippage_cost": 滑点成本,
                "depth_cost": 深度冲击成本,
                "fee": 手续费,
                "total_friction_cost": 总摩擦成本,
            }
        """
        self._total_trades += 1

        # 1. 滑点
        slipped_price = self.apply_slippage(price, side)
        slippage_cost = abs(slipped_price - price) * volume_usd / price

        # 2. 深度冲击
        depth = self.simulate_depth_impact(volume_usd, orderbook_levels)
        depth_cost = depth["depth_cost_pct"] / 100 * volume_usd

        # 3. 执行延迟
        delay = self.simulate_execution_delay()

        # 4. 部分成交
        unfilled = depth.get("unfilled_usd", 0)
        filled = volume_usd - unfilled
        partial = unfilled > 0
        if partial:
            self._partial_fill_count += 1
        if filled < self.min_order_size:
            self._unfilled_count += 1
            filled = 0
            unfilled = volume_usd

        # 5. 手续费
        fee = filled * self.fee_rate

        # 6. 实际成交价（含滑点+深度偏移）
        actual_price = slipped_price + depth.get("vwap_offset", 0)

        total_cost = slippage_cost + depth_cost + fee
        self._total_slippage_cost += slippage_cost
        self._total_delay_cost += depth_cost

        return {
            "actual_price": actual_price,
            "filled_volume": filled,
            "unfilled_volume": unfilled,
            "partial_fill": partial,
            "execution_delay": delay,
            "slippage_cost": slippage_cost,
            "depth_cost": depth_cost,
            "fee": fee,
            "total_friction_cost": total_cost,
        }

    def get_stats(self) -> Dict[str, Any]:
        """返回累积统计信息。"""
        return {
            "total_trades": self._total_trades,
            "unfilled_count": self._unfilled_count,
            "partial_fill_count": self._partial_fill_count,
            "avg_slippage_cost": self._total_slippage_cost / max(1, self._total_trades),
            "total_slippage_cost": self._total_slippage_cost,
            "total_depth_cost": self._total_delay_cost,
            "total_friction_cost": self._total_slippage_cost + self._total_delay_cost,
        }

    def reset_stats(self):
        """重置统计。"""
        self._total_trades = 0
        self._total_slippage_cost = 0.0
        self._total_delay_cost = 0.0
        self._unfilled_count = 0
        self._partial_fill_count = 0
