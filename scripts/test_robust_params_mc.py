#!/usr/bin/env python3
"""
Monte Carlo Bootstrap 稳健性测试（步骤 5.2）

用全量历史数据（方案 B 的 2 年）做 bootstrap 抽样，验证稳健配置的可靠性。
2 年 ≈ 35 个不重叠的 21 天段，bootstrap 1000 次覆盖多种市场状态。

用法:
    python scripts/test_robust_params_mc.py --n-iterations 1000
    python scripts/test_robust_params_mc.py --n-iterations 500 --segment-days 21

输入:
    hyperparam_tune_13_combos_robust.csv（稳健配置表）

输出:
    hyperparam_tune_13_combos_robust_mc_test.json

通过条件: p5 (5% 分位) > 0（即 95% 的 bootstrap 样本盈利）
"""

import argparse
import json
import csv
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_robust_config(csv_path: Path) -> List[Dict[str, Any]]:
    """从 robust CSV 加载稳健配置。"""
    configs = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            configs.append(row)
    return configs


def load_trades_for_combo(
    combo_id: str,
    data_src: Path,
) -> List[Dict[str, Any]]:
    """
    加载某个组合的全量历史交易记录（用于 Monte Carlo 抽样）。
    尝试从 polymarket/logs_*/prediction_trades.json 或回测结果加载。
    """
    # 尝试从 prediction_trades.json 加载
    logs_dir = PROJECT_ROOT / "polymarket" / combo_id
    trades_file = logs_dir / "prediction_trades.json"
    if trades_file.exists():
        try:
            trades = json.loads(trades_file.read_text(encoding="utf-8"))
            if isinstance(trades, list):
                return trades
        except Exception:
            pass
    return []


def bootstrap_pnl(
    trades: List[Dict[str, Any]],
    segment_days: int = 21,
    n_iterations: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    对交易记录做 bootstrap 抽样，统计 PnL 分布。

    步骤：
    1. 将交易按日期分成 segment_days 天的段
    2. 有放回地抽取 N 个段，拼成一个 bootstrap 样本
    3. 计算每个 bootstrap 样本的总 PnL
    4. 统计 PnL 分布的百分位数

    参数：
        trades: 交易记录列表
        segment_days: 每段天数
        n_iterations: bootstrap 次数
        seed: 随机种子

    返回：
        {
            "n_trades": int,
            "n_segments": int,
            "n_iterations": int,
            "pnl_distribution": {p5, p25, p50, p75, p95},
            "mean_pnl": float,
            "pass_p5_gt_0": bool,
        }
    """
    if not trades:
        return {
            "n_trades": 0, "n_segments": 0, "n_iterations": n_iterations,
            "pnl_distribution": {}, "mean_pnl": 0.0, "pass_p5_gt_0": False,
            "error": "no trades"
        }

    # 提取每笔交易的日期和 PnL
    trade_data = []
    for t in trades:
        pnl = t.get("pnl") or t.get("profit") or 0.0
        try:
            pnl = float(pnl)
        except (TypeError, ValueError):
            continue
        ts = t.get("settled_at") or t.get("timestamp") or t.get("created_at") or ""
        if isinstance(ts, (int, float)):
            # 毫秒时间戳
            day = datetime.utcfromtimestamp(ts / 1000).strftime("%Y-%m-%d") if ts > 1e10 else datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
        elif isinstance(ts, str) and len(ts) >= 10:
            day = ts[:10]
        else:
            continue
        trade_data.append({"day": day, "pnl": pnl})

    if not trade_data:
        return {
            "n_trades": 0, "n_segments": 0, "n_iterations": n_iterations,
            "pnl_distribution": {}, "mean_pnl": 0.0, "pass_p5_gt_0": False,
            "error": "no valid trades with pnl"
        }

    # 按日分组 PnL
    daily_pnl: Dict[str, float] = {}
    for td in trade_data:
        daily_pnl[td["day"]] = daily_pnl.get(td["day"], 0.0) + td["pnl"]

    sorted_days = sorted(daily_pnl.keys())
    daily_values = [daily_pnl[d] for d in sorted_days]

    # 将日级别 PnL 分成 segment_days 天的段
    segments = []
    for i in range(0, len(daily_values), segment_days):
        seg = daily_values[i:i + segment_days]
        if len(seg) >= segment_days // 2:  # 至少半段长度
            segments.append(sum(seg))

    if not segments:
        return {
            "n_trades": len(trade_data), "n_segments": 0,
            "n_iterations": n_iterations,
            "pnl_distribution": {}, "mean_pnl": 0.0, "pass_p5_gt_0": False,
            "error": "not enough days to form segments"
        }

    # Bootstrap
    rng = np.random.RandomState(seed)
    n_segs_per_sample = max(1, len(segments))
    bootstrap_pnls = []
    for _ in range(n_iterations):
        indices = rng.randint(0, len(segments), size=n_segs_per_sample)
        sample_pnl = sum(segments[i] for i in indices)
        bootstrap_pnls.append(sample_pnl)

    bootstrap_pnls = np.array(bootstrap_pnls)
    percentiles = {
        "p5": float(np.percentile(bootstrap_pnls, 5)),
        "p25": float(np.percentile(bootstrap_pnls, 25)),
        "p50": float(np.percentile(bootstrap_pnls, 50)),
        "p75": float(np.percentile(bootstrap_pnls, 75)),
        "p95": float(np.percentile(bootstrap_pnls, 95)),
    }

    return {
        "n_trades": len(trade_data),
        "n_segments": len(segments),
        "n_iterations": n_iterations,
        "segment_days": segment_days,
        "pnl_distribution": percentiles,
        "mean_pnl": float(bootstrap_pnls.mean()),
        "std_pnl": float(bootstrap_pnls.std()),
        "pass_p5_gt_0": percentiles["p5"] > 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Monte Carlo Bootstrap 稳健性测试"
    )
    parser.add_argument("--n-iterations", type=int, default=1000,
                        help="bootstrap 迭代次数（默认 1000）")
    parser.add_argument("--segment-days", type=int, default=21,
                        help="每段天数（默认 21 天）")
    parser.add_argument("--robust-csv", type=str, default=None,
                        help="稳健配置 CSV 路径")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="输出目录")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    args = parser.parse_args()

    out_dir = Path(args.out_dir or PROJECT_ROOT)
    robust_csv = Path(args.robust_csv or (PROJECT_ROOT / "hyperparam_tune_13_combos_robust.csv"))

    # 加载稳健配置
    if robust_csv.exists():
        configs = load_robust_config(robust_csv)
        combo_ids = [c.get("组合id") or c.get("combo_id") or c.get("组合") for c in configs]
        print(f"已加载 {len(configs)} 个组合的稳健配置")
    else:
        # 如果没有稳健配置，直接扫描所有 logs 目录
        logs_parent = PROJECT_ROOT / "polymarket"
        combo_ids = [
            d.name for d in logs_parent.iterdir()
            if d.is_dir() and d.name.startswith("logs_") and (d / "prediction_trades.json").exists()
        ]
        print(f"未找到稳健配置 CSV，扫描到 {len(combo_ids)} 个组合")

    # 对每个组合做 bootstrap 测试
    results = {}
    n_pass = 0
    n_total = 0

    for combo_id in combo_ids:
        if not combo_id:
            continue
        trades = load_trades_for_combo(combo_id, PROJECT_ROOT)
        if not trades:
            print(f"  {combo_id}: 无交易记录，跳过")
            continue

        mc = bootstrap_pnl(
            trades,
            segment_days=args.segment_days,
            n_iterations=args.n_iterations,
            seed=args.seed,
        )
        results[combo_id] = mc
        n_total += 1
        status = "PASS" if mc["pass_p5_gt_0"] else "FAIL"
        if mc["pass_p5_gt_0"]:
            n_pass += 1

        dist = mc.get("pnl_distribution", {})
        print(f"  {combo_id}: {status} | p5={dist.get('p5', 0):.2f} p50={dist.get('p50', 0):.2f} "
              f"p95={dist.get('p95', 0):.2f} | {mc['n_trades']} trades, {mc['n_segments']} segments")

    # 汇总
    summary = {
        "test_date": datetime.now(timezone.utc).isoformat(),
        "n_iterations": args.n_iterations,
        "segment_days": args.segment_days,
        "n_combos_tested": n_total,
        "n_pass": n_pass,
        "n_fail": n_total - n_pass,
        "combos": results,
    }

    out_path = out_dir / "hyperparam_tune_13_combos_robust_mc_test.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写出: {out_path}")
    print(f"通过: {n_pass}/{n_total}  (条件: p5 > 0)")


if __name__ == "__main__":
    main()
