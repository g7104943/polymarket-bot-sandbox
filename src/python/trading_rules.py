"""
5 层交易防护系统 — 核心引擎。

Layer 1: 信号过滤（边际优势 Edge 检查）
Layer 2: 动态 Kelly 下注
Layer 3: 连胜/连败调节（滚动胜率）
Layer 4: 回撤熔断（阶梯式降档 / 暂停 / 恢复）
Layer 5: 单笔上限

用于回测 / 超参搜索，不影响模拟盘 Node.js 进程。
"""
from __future__ import annotations

import math
from collections import deque
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# 常量（与 hyperparam_tune_13_combos.py 保持一致）
# ---------------------------------------------------------------------------
INITIAL_CAPITAL = 400.0
MIN_BET = 1.0
CAPITAL_THRESHOLD = 60000.0
MAX_BET_CAP = 3000.0
DEFAULT_FEE_RATE = 0.001
DEFAULT_SLIPPAGE = 0.001
FIXED_ORDER_PRICE_DEFAULT = 0.527


# ===================================================================
# 辅助函数
# ===================================================================

def compute_edge(calibrated_prob: float, price: float) -> float:
    """
    计算二元期权边际优势。
    edge = p * (1/price - 1) - (1 - p)
         = p * payoff_ratio - q
    正值 = 有优势，负值 = 无优势。
    """
    if price <= 0 or price >= 1:
        return -1.0
    payoff = 1.0 / price - 1.0  # 赢时倍率（净）
    return calibrated_prob * payoff - (1.0 - calibrated_prob)


def compute_kelly(calibrated_prob: float, price: float) -> float:
    """
    Kelly Criterion：f* = (p * b - q) / b
    其中 b = 赔率 = 1/price - 1，q = 1 - p。
    返回 [0, 1] 范围的推荐下注比例（满 Kelly）。
    """
    if price <= 0 or price >= 1:
        return 0.0
    b = 1.0 / price - 1.0  # 净赔率
    if b <= 0:
        return 0.0
    q = 1.0 - calibrated_prob
    f = (calibrated_prob * b - q) / b
    return max(0.0, min(1.0, f))


def advanced_score(
    final_cap: float,
    min_cap: float,
    max_dd_pct: float,
    initial: float = INITIAL_CAPITAL,
) -> float:
    """
    目标函数（收益优先，安全兜底）:
      score = final_cap * (min_cap / initial)^0.3 * (1 - dd/100)^0.3

    设计思路：
    - final_cap     : 线性，直接奖励赚钱
    - (ratio)^0.3   : 0.3次方轻惩罚低谷，60% → 0.86, 40% → 0.80
    - (1-dd)^0.3    : 0.3次方轻惩罚回撤，60% → 0.86, 80% → 0.73

    举例（initial=400）:
      最终$10000, 最低$240(60%), 回撤55% → 10000 × 0.86 × 0.87 = 7482
      最终$500,   最低$380(95%), 回撤10% → 500  × 0.98 × 0.97 = 475
      → 高收益策略大幅胜出，允许较大回撤换取增长
    """
    if final_cap <= 0 or min_cap <= 0 or initial <= 0:
        return 0.0
    ratio = min(min_cap / initial, 1.0)  # cap at 1.0
    dd_factor = max(0.0, 1.0 - max_dd_pct / 100.0)
    return final_cap * (ratio ** 0.3) * (dd_factor ** 0.3)


# ===================================================================
# 5 层防护回测引擎
# ===================================================================

class AdvancedEquityCurve:
    """
    5 层交易防护系统。

    Parameters
    ----------
    min_edge : float
        Layer 1 - 最小边际优势，低于此值不入场。
    kelly_frac : float
        Layer 2 - Kelly 安全系数（1/3 Kelly = 0.33）。
    roll_window : int
        Layer 3 - 滚动胜率窗口（笔数）。
    cold_threshold : float
        Layer 3 - 冷却阈值：滚动胜率 < 此值时 streak_multiplier 降低。
    hot_threshold : float
        Layer 3 - 热手阈值：滚动胜率 > 此值时轻微加仓。
    cold_multiplier : float
        Layer 3 - 冷却时的乘数。
    freeze_threshold : float
        Layer 3 - 完全冻结阈值：滚动胜率 < 此值时暂停。
    dd_level1 : float
        Layer 4 - 回撤 Level 1（如 0.10 = 10%），bet × dd_mult1。
    dd_level2 : float
        Layer 4 - 回撤 Level 2（如 0.20 = 20%），bet × dd_mult2。
    dd_halt : float
        Layer 4 - 回撤暂停线（如 0.30 = 30%），完全暂停交易。
    dd_mult1 : float
        Layer 4 - Level 1 回撤乘数。
    dd_mult2 : float
        Layer 4 - Level 2 回撤乘数。
    recovery_target : float
        Layer 4 - 恢复目标：资金恢复到 peak × recovery_target 时重启。
    halt_duration_bars : int
        Layer 4 - 熔断持续时间（交易笔数），默认 96（96 × 15 min = 24 h）。
        资金恢复或时间到期，两者之一满足即解除熔断。
    max_capital_pct : float
        Layer 5 - 单笔下注不超过总资金的此比例。
    initial_capital : float
        初始资金。
    fee_rate : float
        手续费率。
    slippage : float
        滑点。
    """

    def __init__(
        self,
        # Layer 1
        min_edge: float = 0.02,
        # Layer 2
        kelly_frac: float = 0.33,
        # Layer 3
        roll_window: int = 20,
        cold_threshold: float = 0.48,
        hot_threshold: float = 0.60,
        cold_multiplier: float = 0.5,
        freeze_threshold: float = 0.40,
        # Layer 4
        dd_level1: float = 0.10,
        dd_level2: float = 0.20,
        dd_halt: float = 0.30,
        dd_mult1: float = 0.50,
        dd_mult2: float = 0.25,
        recovery_target: float = 0.85,
        halt_duration_bars: int = 96,
        # Layer 5
        max_capital_pct: float = 0.10,
        # 基础
        initial_capital: float = INITIAL_CAPITAL,
        fee_rate: float = DEFAULT_FEE_RATE,
        slippage: float = DEFAULT_SLIPPAGE,
    ):
        self.min_edge = min_edge
        self.kelly_frac = kelly_frac
        self.roll_window = roll_window
        self.cold_threshold = cold_threshold
        self.hot_threshold = hot_threshold
        self.cold_multiplier = cold_multiplier
        self.freeze_threshold = freeze_threshold
        self.dd_level1 = dd_level1
        self.dd_level2 = dd_level2
        self.dd_halt = dd_halt
        self.dd_mult1 = dd_mult1
        self.dd_mult2 = dd_mult2
        self.recovery_target = recovery_target
        self.halt_duration_bars = halt_duration_bars
        self.max_capital_pct = max_capital_pct
        self.initial_capital = initial_capital
        self.fee_rate = fee_rate
        self.slippage = slippage

    def run(
        self,
        trades: List[Dict[str, Any]],
        threshold: float,
        fixed_order_price: float = FIXED_ORDER_PRICE_DEFAULT,
    ) -> Dict[str, Any]:
        """
        逐笔遍历 trades，5 层检查后下注。

        Parameters
        ----------
        trades : list of dict
            每笔含 'confidence' (float), 'result' ('win'/'loss'),
            可选 'date'/'timestamp'。
        threshold : float
            基础置信度阈值（低于此值直接跳过）。
        fixed_order_price : float
            固定买入价。

        Returns
        -------
        dict with keys:
            final_capital, equity_curve, wins, losses, skipped,
            max_drawdown_pct, min_over_initial_pct,
            halted_count, filtered_by_edge, filtered_by_streak,
            n_trades (实际交易笔数), win_rate_pct
        """
        sorted_trades = sorted(
            trades,
            key=lambda t: str(t.get("date") or t.get("timestamp") or ""),
        )

        capital = self.initial_capital
        peak_capital = self.initial_capital
        min_capital = self.initial_capital
        wins = 0
        losses = 0
        skipped = 0
        halted_count = 0
        filtered_by_edge = 0
        filtered_by_streak = 0
        equity_curve = [capital]

        # Layer 3 状态：滚动窗口
        recent_results: deque = deque(maxlen=self.roll_window)

        # Layer 4 状态：是否暂停
        is_halted = False
        halt_start_idx = 0  # 熔断开始的交易索引

        price = fixed_order_price or FIXED_ORDER_PRICE_DEFAULT
        if price <= 0:
            price = FIXED_ORDER_PRICE_DEFAULT

        for trade_idx, t in enumerate(sorted_trades):
            if capital < MIN_BET:
                break

            conf = t.get("confidence")
            if conf is None or conf < threshold:
                skipped += 1
                continue

            # --- Layer 4: 回撤熔断 ---
            if peak_capital > 0:
                current_dd = (peak_capital - capital) / peak_capital
            else:
                current_dd = 0.0

            if is_halted:
                # 检查是否恢复（两个条件之一满足即恢复）
                bars_since_halt = trade_idx - halt_start_idx
                capital_recovered = capital >= peak_capital * self.recovery_target
                time_expired = bars_since_halt >= self.halt_duration_bars
                if capital_recovered or time_expired:
                    is_halted = False
                else:
                    halted_count += 1
                    skipped += 1
                    continue

            if current_dd >= self.dd_halt:
                is_halted = True
                halt_start_idx = trade_idx  # 记录熔断开始位置
                halted_count += 1
                skipped += 1
                continue

            # --- Layer 1: 边际优势检查 ---
            edge = compute_edge(conf, price)
            if edge < self.min_edge:
                filtered_by_edge += 1
                skipped += 1
                continue

            # --- Layer 2: 动态 Kelly 下注 ---
            kelly_f = compute_kelly(conf, price)
            base_bet_ratio = kelly_f * self.kelly_frac

            # --- Layer 3: 连胜/连败调节 ---
            streak_mult = 1.0
            if len(recent_results) >= 5:  # 至少 5 笔才启用
                rolling_wr = sum(recent_results) / len(recent_results)
                if rolling_wr < self.freeze_threshold:
                    streak_mult = 0.0  # 完全暂停
                    filtered_by_streak += 1
                    skipped += 1
                    continue
                elif rolling_wr < self.cold_threshold:
                    streak_mult = self.cold_multiplier
                elif rolling_wr > self.hot_threshold:
                    streak_mult = 1.2  # 轻微加仓

            # --- Layer 4: 回撤阶梯降档 ---
            dd_mult = 1.0
            if current_dd >= self.dd_level2:
                dd_mult = self.dd_mult2
            elif current_dd >= self.dd_level1:
                dd_mult = self.dd_mult1

            # 综合计算下注金额
            bet_ratio = base_bet_ratio * streak_mult * dd_mult

            # --- Layer 5: 单笔上限 ---
            if capital >= CAPITAL_THRESHOLD:
                bet = min(MAX_BET_CAP, capital)
                # 在高资金阶段也应用回撤/连败乘数
                bet = bet * streak_mult * dd_mult
            else:
                bet = capital * bet_ratio

            bet = min(bet, capital * self.max_capital_pct)  # 硬上限
            bet = min(bet, capital)  # 不超过总资金

            if bet < MIN_BET:
                skipped += 1
                continue

            # 执行交易
            fee = bet * (self.fee_rate + self.slippage)
            if t.get("result") == "win":
                pnl = bet * (1.0 / price - 1.0) - fee
                wins += 1
                recent_results.append(1)
            else:
                pnl = -bet - fee
                losses += 1
                recent_results.append(0)

            capital += pnl
            if capital > peak_capital:
                peak_capital = capital
            if capital < min_capital:
                min_capital = capital
            equity_curve.append(capital)

        # 计算最大回撤
        peak = self.initial_capital
        max_dd = 0.0
        for c in equity_curve:
            if c > peak:
                peak = c
            if peak > 0:
                dd = (peak - c) / peak
                if dd > max_dd:
                    max_dd = dd

        n_trades = wins + losses
        min_over_initial = (min_capital / self.initial_capital * 100.0) if self.initial_capital > 0 else 100.0
        win_rate_pct = (wins / n_trades * 100.0) if n_trades > 0 else 0.0

        return {
            "final_capital": round(capital, 2),
            "equity_curve": equity_curve,
            "wins": wins,
            "losses": losses,
            "n_trades": n_trades,
            "skipped": skipped,
            "win_rate_pct": round(win_rate_pct, 1),
            "max_drawdown_pct": round(max_dd * 100.0, 1),
            "min_over_initial_pct": round(min_over_initial, 2),
            "halted_count": halted_count,
            "filtered_by_edge": filtered_by_edge,
            "filtered_by_streak": filtered_by_streak,
        }


# ===================================================================
# 超参搜索空间
# ===================================================================

ADVANCED_PARAM_GRID = {
    "min_edge": [0.0, 0.005, 0.01, 0.02],
    "kelly_frac": [0.50, 0.67, 0.80, 1.0],
    "roll_window": [10, 15, 20],
    "cold_threshold": [0.40, 0.44, 0.48],
    "dd_level1": [0.15, 0.25, 0.35],
    "dd_level2": [0.30, 0.40, 0.50],
    "dd_halt": [0.50, 0.60, 0.70, 0.80],
    "max_capital_pct": [0.10, 0.15, 0.20, 0.30],
}

# 固定不搜索的参数（减少搜索空间）
ADVANCED_FIXED_PARAMS = {
    "hot_threshold": 0.58,
    "cold_multiplier": 0.5,
    "freeze_threshold": 0.35,
    "dd_mult1": 0.60,
    "dd_mult2": 0.35,
    "recovery_target": 0.80,
}


def iter_advanced_grid(
    grid: Optional[Dict[str, list]] = None,
    fixed: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    展开参数网格为参数字典列表。
    全网格 = 4*4*3*3*3*3*3*3 = 11664，使用采样或裁剪后约 2000-4000。
    """
    grid = grid or ADVANCED_PARAM_GRID
    fixed = fixed or ADVANCED_FIXED_PARAMS

    keys = sorted(grid.keys())
    values = [grid[k] for k in keys]
    result = []
    for combo in product(*values):
        params = dict(zip(keys, combo))
        # 约束: dd_level1 < dd_level2 < dd_halt
        if params["dd_level1"] >= params["dd_level2"]:
            continue
        if params["dd_level2"] >= params["dd_halt"]:
            continue
        params.update(fixed)
        result.append(params)
    return result


def run_advanced_backtest(
    trades: List[Dict[str, Any]],
    threshold: float,
    params: Dict[str, Any],
    fixed_order_price: float = FIXED_ORDER_PRICE_DEFAULT,
) -> Dict[str, Any]:
    """
    便捷函数：用给定参数创建 AdvancedEquityCurve 并跑回测。
    """
    engine = AdvancedEquityCurve(**params)
    result = engine.run(trades, threshold, fixed_order_price)
    result["params"] = dict(params)
    result["threshold"] = threshold
    return result
