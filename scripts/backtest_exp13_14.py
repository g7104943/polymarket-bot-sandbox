#!/usr/bin/env python3
"""
backtest_exp13_14.py — Exp8 / Exp10 / Exp13 / Exp14 四方对比回测

调用 compare_fair.py 的多模型对比基础设施,
在同一无泄漏测试集上统一回测, 对比:
  - Exp8  (v5_production, T+0, 无TV)       vs  Exp13 (v5_production_tv, T+0, 有TV)
  - Exp10 (v5_production_sim_noise, T-120s) vs  Exp14 (v5_production_sim_noise_tv, T-120s+TV)
  - Exp13 vs Exp14 (同TV特征, T+0 vs T-120s)

用法:
  python scripts/backtest_exp13_14.py
  python scripts/backtest_exp13_14.py --regenerate   # 强制重新生成预测
  python scripts/backtest_exp13_14.py --buy-prices 0.50,0.51
"""
from __future__ import annotations

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

MODELS_DIR = PROJECT_ROOT / "data" / "models"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "sentiment_grid_search" / "results"

# ── 四个实验配置 ──
EXPERIMENTS = [
    {
        "model_dir": "v5_production",
        "rules_version": "v3",
        "label": "Exp8",
        "sim_noise": False,
        "description": "T+0 完整K线, 无TV宏观特征",
    },
    {
        "model_dir": "v5_production_sim_noise",
        "rules_version": "v3",
        "label": "Exp10",
        "sim_noise": True,
        "description": "T-120s 模拟噪声, 无TV宏观特征",
    },
    {
        "model_dir": "v5_production_tv",
        "rules_version": "v3",
        "label": "Exp13",
        "sim_noise": False,
        "description": "T+0 完整K线 + TV宏观特征",
    },
    {
        "model_dir": "v5_production_sim_noise_tv",
        "rules_version": "v3",
        "label": "Exp14",
        "sim_noise": True,
        "description": "T-120s 模拟噪声 + TV宏观特征",
    },
]


def print_comparison_matrix(summaries: Dict[str, Dict[str, Any]], buy_price: float):
    """打印四方对比矩阵。"""
    odds = (1.0 - buy_price) / buy_price

    print(f"\n{'═' * 100}")
    print(f"  四方对比总结 @ ${buy_price:.3f} (赔率 {odds:.4f})")
    print(f"{'═' * 100}")

    # 表头
    print(f"  {'实验':<8} {'描述':<35} {'最终资金':>12} {'胜率%':>8} {'盈亏%':>10} {'回撤%':>8} {'交易':>6} {'Sharpe':>8}")
    print(f"  {'─'*8} {'─'*35} {'─'*12} {'─'*8} {'─'*10} {'─'*8} {'─'*6} {'─'*8}")

    for exp in EXPERIMENTS:
        label = exp["label"]
        s = summaries.get(label)
        if not s:
            print(f"  {label:<8} {'(无数据)':<35}")
            continue
        desc = exp["description"][:35]
        sharpe = s.get("sharpe", 0)
        print(f"  {label:<8} {desc:<35} ${s['final_capital']:>10,.2f} "
              f"{s['win_rate']:>7.1f} {s['return_pct']:>+9.1f} "
              f"{s['max_drawdown']:>7.1f} {s['n_trades']:>5d} {sharpe:>7.2f}")

    # 对比分析
    print(f"\n  ── TV 宏观特征增益 ──")
    for base, tv in [("Exp8", "Exp13"), ("Exp10", "Exp14")]:
        sb, st = summaries.get(base), summaries.get(tv)
        if sb and st:
            ret_diff = st["return_pct"] - sb["return_pct"]
            wr_diff = st["win_rate"] - sb["win_rate"]
            print(f"    {base} → {tv}: 盈亏差 {ret_diff:+.1f}pp, 胜率差 {wr_diff:+.1f}pp")

    print(f"\n  ── T+0 vs T-120s ──")
    for t0, t120 in [("Exp8", "Exp10"), ("Exp13", "Exp14")]:
        s0, s1 = summaries.get(t0), summaries.get(t120)
        if s0 and s1:
            ret_diff = s1["return_pct"] - s0["return_pct"]
            wr_diff = s1["win_rate"] - s0["win_rate"]
            print(f"    {t0}(T+0) → {t120}(T-120s): 盈亏差 {ret_diff:+.1f}pp, 胜率差 {wr_diff:+.1f}pp")

    print(f"{'═' * 100}")


def main():
    parser = argparse.ArgumentParser(description="Exp8/10/13/14 四方对比回测")
    parser.add_argument("--buy-prices", type=str, default="0.50",
                        help="逗号分隔的买入价格列表")
    parser.add_argument("--date-from", type=str, default=None,
                        help="回测起始日期（默认用 compare_fair 的 DEFAULT_DATE_FROM）")
    parser.add_argument("--date-to", type=str, default=None,
                        help="回测结束日期（默认用 compare_fair 的 DEFAULT_DATE_TO）")
    parser.add_argument("--capital", type=float, default=400.0,
                        help="初始资金")
    parser.add_argument("--regenerate", action="store_true",
                        help="强制重新生成预测（忽略缓存）")
    args = parser.parse_args()

    buy_prices = [float(p.strip()) for p in args.buy_prices.split(",")]

    # 验证所有模型目录存在
    available_exps = []
    for exp in EXPERIMENTS:
        md = MODELS_DIR / exp["model_dir"]
        if md.exists() and (md / "config.json").exists():
            available_exps.append(exp)
        else:
            print(f"  ⚠️ {exp['label']} 模型目录不存在或缺少 config.json: {md}")
            print(f"      跳过 {exp['label']}，回测将不包含此实验")

    if len(available_exps) < 2:
        print("  ❌ 至少需要 2 个可用实验才能对比")
        sys.exit(1)

    # 构建 compare_fair 的 CLI 参数
    model_dirs_args = []
    rules_list_args = []
    for exp in available_exps:
        model_dirs_args.append(exp["model_dir"])
        rules_list_args.append(exp["rules_version"])

    # 调用 compare_fair 的核心函数
    from scripts.compare_fair import (
        generate_model_predictions,
        run_v5_backtest_all,
        simulate_v5_trading,
        _load_v5_params,
        print_ranking,
        save_results,
        print_model_comparison,
        _get_exp_label,
        DEFAULT_DATE_FROM,
        DEFAULT_DATE_TO,
        DEFAULT_INITIAL_CAPITAL,
        TOTAL_COST,
    )

    date_from = args.date_from or DEFAULT_DATE_FROM
    date_to = args.date_to or DEFAULT_DATE_TO

    print(f"\n{'█' * 80}")
    print(f"  Exp8 / Exp10 / Exp13 / Exp14 四方对比回测")
    print(f"{'█' * 80}")
    print(f"  实验数: {len(available_exps)}")
    for exp in available_exps:
        print(f"    - {exp['label']}: {exp['description']} ({exp['model_dir']})")
    print(f"  日期范围: {date_from} ~ {date_to}")
    print(f"  买入价格: {', '.join(f'${p:.3f}' for p in buy_prices)}")
    print(f"  初始资金: ${args.capital:.0f}")

    # ── Phase 0: 生成所有模型的预测 ──
    unique_dirs = list({exp["model_dir"]: MODELS_DIR / exp["model_dir"] for exp in available_exps}.values())
    model_predictions = generate_model_predictions(
        unique_dirs,
        date_from=date_from,
        date_to=date_to,
        regenerate=args.regenerate,
    )

    # ── Phase 1: AUC 和 Accuracy 对比 ──
    print(f"\n{'─' * 80}")
    print(f"  模型预测性能（测试集）")
    print(f"{'─' * 80}")
    print(f"  {'实验':<8} {'模型目录':<35} {'Bars':>6} {'Accuracy':>10} {'AUC':>10}")
    print(f"  {'─'*8} {'─'*35} {'─'*6} {'─'*10} {'─'*10}")

    from sklearn.metrics import roc_auc_score
    for exp in available_exps:
        pred_df = model_predictions.get(exp["model_dir"])
        if pred_df is None:
            continue
        actual = pred_df["actual"].values
        proba = pred_df["proba_up"].values
        acc = ((proba >= 0.5).astype(int) == actual).mean()
        try:
            auc = roc_auc_score(actual, proba)
        except ValueError:
            auc = float("nan")
        print(f"  {exp['label']:<8} {exp['model_dir']:<35} {len(pred_df):>5d} {acc:>9.4f} {auc:>9.4f}")

    # ── Phase 2: 分币种 AUC ──
    print(f"\n  分币种 AUC:")
    print(f"  {'实验':<8} ", end="")
    assets = ["BTC_USDT", "ETH_USDT", "XRP_USDT"]
    for a in assets:
        print(f" {a.split('_')[0]:>10}", end="")
    print()
    for exp in available_exps:
        pred_df = model_predictions.get(exp["model_dir"])
        if pred_df is None:
            continue
        print(f"  {exp['label']:<8} ", end="")
        for a in assets:
            adf = pred_df[pred_df["asset"] == a]
            if len(adf) < 10:
                print(f" {'N/A':>10}", end="")
                continue
            try:
                auc = roc_auc_score(adf["actual"].values, adf["proba_up"].values)
            except ValueError:
                auc = float("nan")
            print(f" {auc:>10.4f}", end="")
        print()

    # ── Phase 3: 交易回测 ──
    for bp in buy_prices:
        odds = (1.0 - bp) / bp
        print(f"\n\n{'▓' * 80}")
        print(f"  交易回测 @ ${bp:.3f} (赔率 {odds:.4f})")
        print(f"{'▓' * 80}")

        all_combos = []
        per_model_results: Dict[str, List[Dict]] = {}
        summaries: Dict[str, Dict[str, Any]] = {}

        for exp in available_exps:
            model_name = exp["model_dir"]
            label = exp["label"]
            pred_df = model_predictions.get(model_name)
            if pred_df is None:
                continue

            print(f"\n  🧪 {label} ({model_name}) 回测...")
            v5_results = run_v5_backtest_all(
                buy_price=bp,
                date_from=date_from,
                date_to=date_to,
                initial_capital=args.capital,
                rules_version=exp["rules_version"],
                predictions=pred_df,
                model_label=model_name,
                exp_label=label,
            )
            all_combos.extend(v5_results)
            per_model_results[label] = v5_results

            # 提取 ALL 池化结果作为摘要
            all_pool = [r for r in v5_results if "_ALL" in r.get("组合", "") and "ladder" not in r.get("组合", "")]
            if all_pool:
                best = max(all_pool, key=lambda x: x["最终资金"])
                summaries[label] = {
                    "final_capital": best["最终资金"],
                    "return_pct": best["盈亏%"],
                    "win_rate": best["胜率%"],
                    "max_drawdown": best["最大回撤%"],
                    "n_trades": best["交易次数"],
                    "sharpe": 0,  # compare_fair 的回测结果中没有 sharpe
                }

        if all_combos:
            ranked = print_ranking(all_combos, bp, date_from, date_to)
            csv_path = save_results(ranked, bp, RESULTS_DIR)
            print(f"\n  📁 结果已保存: {csv_path}")

        # 打印多模型头对头对比
        print_model_comparison(per_model_results, bp)

        # 打印四方对比矩阵
        print_comparison_matrix(summaries, bp)

    # ── 最终汇总 ──
    print(f"\n\n{'█' * 80}")
    print("  ✅ Exp8/10/13/14 四方对比回测完成！")
    print(f"{'█' * 80}")


if __name__ == "__main__":
    main()
