#!/usr/bin/env python3
"""
生成简化交易规则 JSON — 消除 Stage 2 过拟合

理念:
  世界顶级量化基金的做法：用固定的、理论驱动的规则替代 Optuna 优化出的 12 参数规则。
  
  核心变更:
    - Quarter-Kelly (kelly_frac=0.25): 保守但稳健，不需要优化
    - 无置信度分档: 模型输出的概率直接用于 Kelly 公式
    - 无冷却期: 没有统计基础的情绪化干预
    - 固定 30% 回撤熔断: 行业标准
    - 禁用 min_edge 过滤: Kelly 公式本身已包含 edge 计算
    - 宽松的 min_confidence (0.505): 让 Kelly 决定仓位大小

  从 12 个参数减到实质 3 个:
    min_confidence (0.505), kelly_frac (0.25), drawdown_halt (0.30)

用法:
  python scripts/generate_simple_rules.py
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "sentiment_grid_search" / "results" / "simplified_rules"

# ─── 简化规则（理论驱动，非数据驱动） ─────────────────────────────
SIMPLE_TRADING_RULES = {
    "min_confidence": 0.505,           # 略高于随机（让 Kelly 自行决定仓位）
    "min_edge": 0.0,                   # 禁用（Kelly 公式已包含 edge 计算）
    "kelly_frac": 0.25,                # Quarter-Kelly（保守、抗过拟合）
    "bet_pct_normal": 0.05,            # 固定 5% 上限
    "bet_pct_conservative": 0.05,      # 与 normal 一致（取消保守/激进区分）
    "confidence_tiers": [              # 全部设为 1.0 = 等效禁用分档
        [0.505, 0.55, 1.0],            # Tier 1: 低置信度 → 乘数 1.0
        [0.55, 0.60, 1.0],             # Tier 2: 中置信度 → 乘数 1.0
        [0.60, 1.00, 1.0],             # Tier 3: 高置信度 → 乘数 1.0
    ],
    "cooldown_bars": 0,                # 禁用冷却期
    "drawdown_halt": 0.30,             # 固定 30%（行业标准）
}

# ─── 买价配置 ─────────────────────────────────────────────────────
BUY_PRICES = [0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53]
DYNAMIC_CONFIGS = [
    ("bp_dyn_0450_0530", None, [0.45, 0.53]),
    ("bp_dyn_0480_0510", None, [0.48, 0.51]),
]

FEE_RATE = 0.002
SLIPPAGE_RATE = 0.003
LIQUIDITY_CAP = 60000
LIQUIDITY_BET = 3000


def make_rule_json(buy_price: float | None, buy_price_range: list | None = None) -> dict:
    """生成一个买价对应的完整规则 JSON（兼容现有系统）"""
    bp = buy_price if buy_price else 0.50
    return {
        "trading_rules": dict(SIMPLE_TRADING_RULES),
        "polymarket_constraints": {
            "buy_price": bp,
            "buy_price_range": buy_price_range,
            "odds": round(1.0 / bp - 1.0, 4) if buy_price else 1.0,
            "fee_rate": FEE_RATE,
            "slippage_rate": SLIPPAGE_RATE,
            "liquidity_cap": LIQUIDITY_CAP,
            "liquidity_bet": LIQUIDITY_BET,
        },
        "search_config": {
            "method": "固定规则（Quarter-Kelly + 30% 回撤熔断）",
            "note": "理论驱动，非 Optuna 数据驱动优化",
            "n_parameters": 3,
            "parameters": "min_confidence=0.505, kelly_frac=0.25, drawdown_halt=0.30",
        },
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    generated = []
    
    # 固定买价
    for bp in BUY_PRICES:
        tag = f"bp{bp:.3f}".replace(".", "")
        filename = f"optimal_trading_rules_v3_{tag}.json"
        filepath = OUTPUT_DIR / filename
        
        rule = make_rule_json(bp)
        with open(filepath, "w") as f:
            json.dump(rule, f, indent=2, ensure_ascii=False)
        
        generated.append(f"  ✅ {tag} (${bp}) → {filename}")
    
    # 动态范围
    for tag, _, price_range in DYNAMIC_CONFIGS:
        filename = f"optimal_trading_rules_v3_{tag}.json"
        filepath = OUTPUT_DIR / filename
        
        rule = make_rule_json(buy_price=None, buy_price_range=price_range)
        with open(filepath, "w") as f:
            json.dump(rule, f, indent=2, ensure_ascii=False)
        
        generated.append(f"  ✅ {tag} (动态) → {filename}")
    
    print(f"\n📋 生成简化交易规则 JSON ({len(generated)} 个)")
    print(f"   输出目录: {OUTPUT_DIR}")
    print()
    for line in generated:
        print(line)
    
    print()
    print("简化规则（3 个有效参数）:")
    print(f"  min_confidence = {SIMPLE_TRADING_RULES['min_confidence']}")
    print(f"  kelly_frac     = {SIMPLE_TRADING_RULES['kelly_frac']} (Quarter-Kelly)")
    print(f"  drawdown_halt  = {SIMPLE_TRADING_RULES['drawdown_halt']} (行业标准 30%)")
    print()
    print("禁用的参数:")
    print("  confidence_tiers = 全部 1.0（等效禁用）")
    print("  cooldown_bars    = 0（禁用）")
    print("  min_edge         = 0（禁用，Kelly 已包含 edge）")
    print("  bet_pct_conservative = 与 normal 一致（取消区分）")
    print()


if __name__ == "__main__":
    main()
