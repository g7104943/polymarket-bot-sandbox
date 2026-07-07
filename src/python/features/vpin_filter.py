"""
VPIN (Volume-Synchronized Probability of Informed Trading) 订单流毒性过滤器。

用于 7 层交易规则的 Layer 6：当订单流毒性过高时跳过交易，
避免 crowded trade 导致实际 fill edge 被吃光。

参考文献：
  Easley, Lopez de Prado & O'Hara (2012)
  "Flow Toxicity and Liquidity in a High-Frequency World"

加密货币永续合约中的 VPIN 典型值（2024-2026）：
  正常水平：0.45–0.48
  危险阈值：0.60+（≈ 均值 + 1.5~2 std）

使用方式：
  filter = CryptoVPINFilter(symbols=["BTCUSDT", "ETHUSDT"])
  filter.on_trade("BTCUSDT", side="Buy", size_usdt=120.5)
  is_toxic, reason = filter.should_trade("BTCUSDT")
"""

from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ───────────────────────────────────────────────────
# 数据结构
# ───────────────────────────────────────────────────

@dataclass
class _Bucket:
    """一桶成交量。"""
    buy_vol: float = 0.0
    sell_vol: float = 0.0
    total_vol: float = 0.0
    timestamp: float = 0.0  # 桶完成时的 epoch 秒

    @property
    def buy_fraction(self) -> float:
        return self.buy_vol / self.total_vol if self.total_vol > 0 else 0.5


@dataclass
class _SymbolState:
    """单个币种的 VPIN 状态。"""
    # 当前正在累积的桶
    current_bucket: _Bucket = field(default_factory=_Bucket)
    # 已完成的桶（滚动窗口）
    completed_buckets: deque = field(default_factory=lambda: deque(maxlen=200))
    # 当前 VPIN 值
    vpin: float = 0.0
    # 相对阈值的滚动历史（用于计算 85th percentile）
    vpin_history: deque = field(default_factory=lambda: deque(maxlen=500))
    # 持久高计数（连续超标桶数）
    persistent_high_count: int = 0
    # 冷却结束时间（epoch 秒）
    cooldown_until: float = 0.0
    # 统计
    total_trades_received: int = 0
    total_buckets_completed: int = 0


# ───────────────────────────────────────────────────
# 核心过滤器
# ───────────────────────────────────────────────────

class CryptoVPINFilter:
    """
    实时 VPIN 订单流毒性过滤器。

    Parameters
    ----------
    symbols : list of str
        监控的交易对（如 ["BTCUSDT", "ETHUSDT"]）。
    bucket_volume : float
        每桶累积的 USDT 成交额（默认 5000）。
        经验值：BTC 15min 平均成交额 ~500 万，5000/桶 ≈ 1000 桶/15min。
    lookback_buckets : int
        滚动窗口长度（默认 80 桶 ≈ 15-20 分钟）。
    threshold_absolute : float
        VPIN 绝对阈值（默认 0.62）。
    threshold_percentile : float
        相对阈值分位数（默认 0.85 = 85th percentile）。
        使用最近 1 小时的 VPIN 历史计算。
    persistent_high_limit : int
        连续超标桶数限制（默认 5）。
        连续超过绝对阈值的桶数 >= 此值时触发冷却。
    cooldown_minutes : float
        触发持久高冷却后暂停交易的分钟数（默认 20）。
    verbose : bool
        是否打印日志。
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        bucket_volume: float = 5000.0,
        lookback_buckets: int = 80,
        threshold_absolute: float = 0.62,
        threshold_percentile: float = 0.85,
        persistent_high_limit: int = 5,
        cooldown_minutes: float = 20.0,
        verbose: bool = True,
    ):
        self.symbols = [s.upper() for s in (symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"])]
        self.bucket_volume = bucket_volume
        self.lookback_buckets = lookback_buckets
        self.threshold_absolute = threshold_absolute
        self.threshold_percentile = threshold_percentile
        self.persistent_high_limit = persistent_high_limit
        self.cooldown_minutes = cooldown_minutes
        self.verbose = verbose

        # 每个币种的独立状态
        self._states: Dict[str, _SymbolState] = {}
        for sym in self.symbols:
            self._states[sym] = _SymbolState(
                completed_buckets=deque(maxlen=lookback_buckets * 3),
            )

    # ───────────────────────────────────────────────
    # 核心计算
    # ───────────────────────────────────────────────

    def on_trade(self, symbol: str, side: str, size_usdt: float) -> None:
        """
        接收一笔真实成交。

        Parameters
        ----------
        symbol : str
            交易对（如 "BTCUSDT"）。
        side : str
            "Buy" 或 "Sell"（Bybit publicTrade 格式）。
        size_usdt : float
            该笔成交的 USDT 金额。
        """
        sym = symbol.upper()
        if sym not in self._states:
            return

        state = self._states[sym]
        state.total_trades_received += 1
        now = time.time()

        # 累积到当前桶
        bucket = state.current_bucket
        is_buy = side.lower() in ("buy", "b")
        if is_buy:
            bucket.buy_vol += size_usdt
        else:
            bucket.sell_vol += size_usdt
        bucket.total_vol += size_usdt

        # 桶已满 → 完成，开始新桶
        while bucket.total_vol >= self.bucket_volume:
            overflow = bucket.total_vol - self.bucket_volume

            # 完成当前桶（截断到 bucket_volume）
            if overflow > 0:
                # 按比例分配溢出
                ratio = self.bucket_volume / (bucket.total_vol) if bucket.total_vol > 0 else 1
                completed = _Bucket(
                    buy_vol=bucket.buy_vol * ratio,
                    sell_vol=bucket.sell_vol * ratio,
                    total_vol=self.bucket_volume,
                    timestamp=now,
                )
                # 溢出部分进入新桶
                remaining_buy = bucket.buy_vol * (1 - ratio)
                remaining_sell = bucket.sell_vol * (1 - ratio)
            else:
                completed = _Bucket(
                    buy_vol=bucket.buy_vol,
                    sell_vol=bucket.sell_vol,
                    total_vol=bucket.total_vol,
                    timestamp=now,
                )
                remaining_buy = 0.0
                remaining_sell = 0.0

            state.completed_buckets.append(completed)
            state.total_buckets_completed += 1

            # 重新计算 VPIN
            self._update_vpin(state)

            # 检查持久高
            self._check_persistent_high(state, now)

            # 新桶
            bucket = _Bucket(
                buy_vol=remaining_buy,
                sell_vol=remaining_sell,
                total_vol=remaining_buy + remaining_sell,
            )
            state.current_bucket = bucket

    def _update_vpin(self, state: _SymbolState) -> None:
        """
        计算滚动 VPIN。

        VPIN = (1/n) × Σ |V_buy_i - V_sell_i| / V_bucket
             = (1/n) × Σ |2 × buy_fraction_i - 1|

        这是 Easley et al. (2012) 的标准公式。
        """
        buckets = list(state.completed_buckets)
        n = min(len(buckets), self.lookback_buckets)
        if n < 5:
            # 样本太少，不计算
            state.vpin = 0.0
            return

        recent = buckets[-n:]
        order_imbalances = [abs(2 * b.buy_fraction - 1) for b in recent]
        state.vpin = sum(order_imbalances) / n

        # 记录到历史（用于相对阈值）
        state.vpin_history.append(state.vpin)

    def _check_persistent_high(self, state: _SymbolState, now: float) -> None:
        """检查是否触发持久高冷却。"""
        if state.vpin > self.threshold_absolute:
            state.persistent_high_count += 1
        else:
            state.persistent_high_count = 0

        if state.persistent_high_count >= self.persistent_high_limit:
            state.cooldown_until = now + self.cooldown_minutes * 60
            if self.verbose:
                print(
                    f"[VPIN] 持久高毒性触发冷却: "
                    f"连续 {state.persistent_high_count} 桶 > {self.threshold_absolute:.2f}, "
                    f"暂停 {self.cooldown_minutes} 分钟"
                )

    def _get_relative_threshold(self, state: _SymbolState) -> float:
        """
        计算相对阈值：最近 VPIN 历史的 85th percentile。
        """
        history = list(state.vpin_history)
        if len(history) < 20:
            # 历史不足，使用绝对阈值
            return self.threshold_absolute

        import numpy as np
        return float(np.percentile(history, self.threshold_percentile * 100))

    # ───────────────────────────────────────────────
    # 交易决策接口
    # ───────────────────────────────────────────────

    def should_trade(self, symbol: str) -> Tuple[bool, str]:
        """
        判断当前是否应该交易。

        Parameters
        ----------
        symbol : str
            交易对（如 "BTCUSDT"）。

        Returns
        -------
        (is_safe, reason) : (bool, str)
            is_safe=True  → 可以交易
            is_safe=False → 不应交易，reason 说明原因
        """
        sym = symbol.upper()

        # 未知币种 → 默认允许
        if sym not in self._states:
            return True, "unknown_symbol"

        state = self._states[sym]
        now = time.time()

        # 检查冷却
        if state.cooldown_until > now:
            remaining = (state.cooldown_until - now) / 60
            reason = (
                f"VPIN 持久高冷却中 (剩余 {remaining:.1f} 分钟, "
                f"连续 {state.persistent_high_count} 桶超标)"
            )
            return False, reason

        # 样本不足 → 允许交易（但标注）
        if state.total_buckets_completed < 10:
            return True, f"warmup (仅 {state.total_buckets_completed} 桶)"

        vpin = state.vpin

        # 绝对阈值检查
        if vpin > self.threshold_absolute:
            reason = (
                f"VPIN={vpin:.4f} > 绝对阈值 {self.threshold_absolute:.2f} "
                f"(连续高 {state.persistent_high_count} 桶)"
            )
            return False, reason

        # 相对阈值检查（85th percentile of recent history）
        relative_th = self._get_relative_threshold(state)
        if vpin > relative_th:
            reason = (
                f"VPIN={vpin:.4f} > 相对阈值 {relative_th:.4f} "
                f"(最近历史 {self.threshold_percentile*100:.0f}th percentile)"
            )
            return False, reason

        return True, f"VPIN={vpin:.4f} 正常"

    # ───────────────────────────────────────────────
    # 状态查询
    # ───────────────────────────────────────────────

    def get_status(self, symbol: str) -> Dict[str, Any]:
        """获取币种的 VPIN 详细状态。"""
        sym = symbol.upper()
        if sym not in self._states:
            return {"symbol": sym, "error": "not_tracked"}

        state = self._states[sym]
        now = time.time()
        relative_th = self._get_relative_threshold(state)

        return {
            "symbol": sym,
            "vpin": round(state.vpin, 6),
            "threshold_absolute": self.threshold_absolute,
            "threshold_relative": round(relative_th, 6),
            "is_toxic": state.vpin > min(self.threshold_absolute, relative_th),
            "persistent_high_count": state.persistent_high_count,
            "in_cooldown": state.cooldown_until > now,
            "cooldown_remaining_min": round(max(0, (state.cooldown_until - now) / 60), 1),
            "total_trades": state.total_trades_received,
            "total_buckets": state.total_buckets_completed,
            "current_bucket_fill_pct": round(
                state.current_bucket.total_vol / self.bucket_volume * 100, 1
            ) if self.bucket_volume > 0 else 0,
        }

    def get_all_status(self) -> Dict[str, Dict[str, Any]]:
        """获取所有币种的状态。"""
        return {sym: self.get_status(sym) for sym in self.symbols}

    def dump_status_json(self, output_path: Path) -> None:
        """将所有币种状态写入 JSON 文件（供 TypeScript 读取）。"""
        status = self.get_all_status()
        status["_meta"] = {
            "updated_at": time.time(),
            "bucket_volume": self.bucket_volume,
            "lookback_buckets": self.lookback_buckets,
            "threshold_absolute": self.threshold_absolute,
            "threshold_percentile": self.threshold_percentile,
            "persistent_high_limit": self.persistent_high_limit,
            "cooldown_minutes": self.cooldown_minutes,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = output_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(status, f, indent=2)
        tmp.rename(output_path)

    # ───────────────────────────────────────────────
    # 离线分析
    # ───────────────────────────────────────────────

    @staticmethod
    def compute_from_trades(
        trades: List[Dict[str, Any]],
        bucket_volume: float = 5000.0,
        lookback_buckets: int = 80,
    ) -> List[Dict[str, Any]]:
        """
        从历史成交数据离线计算 VPIN 序列。

        Parameters
        ----------
        trades : list of dict
            每笔含 {"timestamp", "side" ("Buy"/"Sell"), "size_usdt"}。
        bucket_volume : float
            桶大小。
        lookback_buckets : int
            滚动窗口。

        Returns
        -------
        list of dict
            每个桶完成时的 VPIN 值。
        """
        buckets: List[_Bucket] = []
        current = _Bucket()
        results = []

        for t in trades:
            side = t.get("side", "Buy")
            size = float(t.get("size_usdt", 0))
            ts = float(t.get("timestamp", 0))

            is_buy = side.lower() in ("buy", "b")
            if is_buy:
                current.buy_vol += size
            else:
                current.sell_vol += size
            current.total_vol += size

            while current.total_vol >= bucket_volume:
                overflow = current.total_vol - bucket_volume
                if overflow > 0:
                    ratio = bucket_volume / current.total_vol if current.total_vol > 0 else 1
                    completed = _Bucket(
                        buy_vol=current.buy_vol * ratio,
                        sell_vol=current.sell_vol * ratio,
                        total_vol=bucket_volume,
                        timestamp=ts,
                    )
                    remaining_buy = current.buy_vol * (1 - ratio)
                    remaining_sell = current.sell_vol * (1 - ratio)
                else:
                    completed = _Bucket(
                        buy_vol=current.buy_vol,
                        sell_vol=current.sell_vol,
                        total_vol=current.total_vol,
                        timestamp=ts,
                    )
                    remaining_buy = 0.0
                    remaining_sell = 0.0

                buckets.append(completed)

                # 计算 VPIN
                n = min(len(buckets), lookback_buckets)
                if n >= 5:
                    recent = buckets[-n:]
                    imbalances = [abs(2 * b.buy_fraction - 1) for b in recent]
                    vpin = sum(imbalances) / n
                else:
                    vpin = 0.0

                results.append({
                    "timestamp": ts,
                    "vpin": vpin,
                    "buy_fraction": completed.buy_fraction,
                    "bucket_index": len(buckets) - 1,
                })

                current = _Bucket(
                    buy_vol=remaining_buy,
                    sell_vol=remaining_sell,
                    total_vol=remaining_buy + remaining_sell,
                )

        return results

    @staticmethod
    def compute_statistics(vpin_values: List[float]) -> Dict[str, float]:
        """
        计算 VPIN 序列的统计量。

        Returns
        -------
        dict with keys: mean, std, median, p75, p85, p90, p95, min, max, count
        """
        import numpy as np
        arr = np.array(vpin_values)
        if len(arr) == 0:
            return {"count": 0}
        return {
            "count": len(arr),
            "mean": round(float(np.mean(arr)), 6),
            "std": round(float(np.std(arr)), 6),
            "median": round(float(np.median(arr)), 6),
            "p75": round(float(np.percentile(arr, 75)), 6),
            "p85": round(float(np.percentile(arr, 85)), 6),
            "p90": round(float(np.percentile(arr, 90)), 6),
            "p95": round(float(np.percentile(arr, 95)), 6),
            "min": round(float(np.min(arr)), 6),
            "max": round(float(np.max(arr)), 6),
        }
