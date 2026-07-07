#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polyfun_next.realistic_backtest import (
    compare_realistic_methods,
    load_candidates_with_episode_paths,
    load_episode_candidates,
    metrics_to_markdown,
)


def _direction_win_rate(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    return 100.0 * (df["actual_up"].astype(int) == df["direction_target"].astype(int)).mean()


def _write_blocked_report(report_dir: Path, payload: dict) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "polyfun_next_realistic_backtest_latest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = [
        "# polyfun-next 真实盘口成交约束回测",
        "",
        f"生成时间：{payload['generatedAt']}",
        "",
        "## 结论",
        "当前不能给出 polyfun-next 的有效胜率 / 盈亏 / 回撤表，因为本地可用 episode 文件里的候选方向字段存在后验泄漏。",
        "",
        "## 泄漏审计",
        "|候选源|样本数|方向命中率|结论|",
        "|---|---:|---:|---|",
    ]
    for row in payload["audits"]:
        lines.append(f"|{row['source']}|{row['rows']}|{row['directionWinRatePct']:.4f}%|{row['verdict']}|")
    lines += [
        "",
        "## 为什么不能用",
        "- `best_action_key != ABSTAIN` 的方向命中率达到 100%，这是未来答案，不是预测。",
        "- 全量 `direction_target` 的方向命中率约 95%，仍然明显不是可实盘使用的预测方向。",
        "- 用这些字段回测会得到虚假的超高胜率和零回撤，等于拿未来结果交易。",
        "",
        "## 现在缺什么",
        "要得到真正可用的 polyfun-next 回测，必须先导出一个不含未来信息的候选流：",
        "1. 每个 15分钟市场在下单前生成的 ETH 方向和分数。",
        "2. 当时真实可见的 token 盘口：买价、卖价、价差、深度。",
        "3. 下单价、5分钟取消规则、是否成交、成交比例。",
        "4. 最终结算结果。",
        "",
        "## 临时可用但不是真钱证明的表",
        "可参考 `/Users/mac/polyfun/reports/proxy_180_365_fair_compare_latest.md`：那是原始线/token代理，不解决真实成交选择偏差。",
        "",
        "## 下一步必须做",
        "先写候选导出器，把 ETH 15m 5年训练持有到期模型在每根市场下单前的候选保存成 JSONL；然后再用本回测器重跑。没有这个候选流，就不应该真钱上线。",
    ]
    (report_dir / "polyfun_next_realistic_backtest_latest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eth-episodes", default="/Users/mac/polyfun/data/processed/vnext_entry_exit_episodes_eth_usdt.parquet")
    ap.add_argument("--candidate-stream", default="/Users/mac/polyfun/polyfun-next/runtime/eth15m_5y_candidate_stream.jsonl")
    ap.add_argument("--reports", default="/Users/mac/polyfun/reports")
    ap.add_argument("--force-invalid-demo", action="store_true", help="write the invalid demo metrics anyway for debugging")
    args = ap.parse_args()
    reports = Path(args.reports)
    candidate_stream = Path(args.candidate_stream)

    if candidate_stream.exists():
        df = load_candidates_with_episode_paths(args.eth_episodes, candidate_stream)
        metrics = compare_realistic_methods(df, ["180d", "365d", "all"])
        reports.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "ok_pre_entry_candidate_stream",
            "candidateStream": str(candidate_stream),
            "rows": len(df),
            "metrics": [m.__dict__ for m in metrics],
        }
        (reports / "polyfun_next_realistic_backtest_latest.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        (reports / "polyfun_next_realistic_backtest_latest.md").write_text(metrics_to_markdown(metrics, "polyfun-next 真实盘口成交约束回测"), encoding="utf-8")
        print(reports / "polyfun_next_realistic_backtest_latest.md")
        return 0

    all_df = load_episode_candidates(args.eth_episodes, require_non_abstain=False)
    non_abstain_df = load_episode_candidates(args.eth_episodes, require_non_abstain=True)
    audits = [
        {
            "source": "direction_target 全量",
            "rows": int(len(all_df)),
            "directionWinRatePct": _direction_win_rate(all_df),
            "verdict": "无效：命中率过高，疑似后验标签" if _direction_win_rate(all_df) > 70 else "可继续审计",
        },
        {
            "source": "best_action_key 非 ABSTAIN",
            "rows": int(len(non_abstain_df)),
            "directionWinRatePct": _direction_win_rate(non_abstain_df),
            "verdict": "无效：明显偷看未来" if _direction_win_rate(non_abstain_df) > 70 else "可继续审计",
        },
    ]
    invalid = any(a["directionWinRatePct"] > 70 for a in audits)
    if invalid and not args.force_invalid_demo:
        payload = {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "status": "blocked_no_valid_pre_entry_candidate_stream",
            "audits": audits,
            "requiredNextData": [
                "pre_entry_candidate_jsonl_without_future_labels",
                "official_visible_orderbook_at_decision_time",
                "fill_or_cancel_lifecycle",
                "resolved_market_result",
            ],
        }
        _write_blocked_report(reports, payload)
        print(reports / "polyfun_next_realistic_backtest_latest.md")
        return 0

    metrics = compare_realistic_methods(all_df, ["180d", "365d", "all"])
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "polyfun_next_realistic_backtest_latest.json").write_text(json.dumps([m.__dict__ for m in metrics], indent=2, ensure_ascii=False), encoding="utf-8")
    (reports / "polyfun_next_realistic_backtest_latest.md").write_text(metrics_to_markdown(metrics, "polyfun-next 真实盘口成交约束回测（无效演示）"), encoding="utf-8")
    print(reports / "polyfun_next_realistic_backtest_latest.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
