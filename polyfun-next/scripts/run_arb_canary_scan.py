#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from polyfun_next.arb_canary import load_arb_config, scan_orderbook_arbitrage, to_jsonable


def main() -> int:
    ap = argparse.ArgumentParser(description="Scan Polymarket orderbooks for complete-set arbitrage canary opportunities")
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    cfg = load_arb_config(args.config)
    report = scan_orderbook_arbitrage(cfg, market_limit=args.limit)
    out_json = REPO / "reports" / "polyfun_next_arb_canary_scan_latest.json"
    out_md = REPO / "reports" / "polyfun_next_arb_canary_scan_latest.md"
    contract = REPO / "reports" / "polyfun_next_arb_canary_contract_latest.md"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(to_jsonable(report), indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(_markdown_report(report), encoding="utf-8")
    contract.write_text(_contract_markdown(cfg), encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")
    print(f"wrote {contract}")
    print(f"opportunities={len(report.opportunities)} markets_seen={report.markets_seen}")
    return 0


def _markdown_report(report) -> str:
    lines = [
        "# polyfun-next 套利金丝雀扫描报告",
        "",
        f"- 生成时间：`{report.generated_at}`",
        f"- 市场扫描数：`{report.markets_seen}`",
        f"- 通过流动性/结果数初筛：`{report.markets_eligible}`",
        f"- 成功读取订单簿：`{report.markets_with_books}`",
        f"- 套利候选数：`{len(report.opportunities)}`",
        "",
        "## 候选表",
        "|市场|结果数|总成本|锁定回收|锁定收益|收益率|订单类型|腿明细|",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    if not report.opportunities:
        lines.append("|无|0|0|0|0|0|无|当前扫描未发现满足 5U 金丝雀门槛的完整套利机会|")
    for opp in report.opportunities[:25]:
        legs = "<br>".join(
            f"{leg.outcome}: {leg.shares:.2f} shares @ <= {leg.worst_price:.4f}, cost {leg.cost_usd:.4f}"
            for leg in opp.legs
        )
        lines.append(
            f"|{opp.market.slug}|{len(opp.legs)}|{opp.total_cost_usd:.4f}|{opp.payout_usd:.4f}|"
            f"{opp.edge_usd:.4f}|{opp.edge_pct:.3%}|{opp.order_type}|{legs}|"
        )
    if report.errors:
        lines.extend(["", "## 读取错误样本"])
        for err in report.errors[:20]:
            lines.append(f"- `{err}`")
    lines.extend(
        [
            "",
            "## 结论",
            "- 本报告只做只读扫描，不提交真钱订单。",
            "- 若候选数为 0，正确动作是不交易；套利金丝雀不能为了交易而交易。",
            "- 真钱执行仍需要单独 live ACK，并且默认禁止非原子多腿执行。",
        ]
    )
    return "\n".join(lines) + "\n"


def _contract_markdown(cfg) -> str:
    return f"""# polyfun-next 套利金丝雀合同

- 策略：完整结果组合套利，不做方向预测。
- 单笔最大总成本：`{cfg.max_total_cost_usd}U`
- 目标回收：`{cfg.target_payout_usd}U`
- 最小锁定收益：`{cfg.min_edge_usd}U`
- 最小收益率：`{cfg.min_edge_pct:.3%}`
- 最小流动性：`{cfg.min_liquidity_usd}U`
- 最小成交量：`{cfg.min_volume_usd}U`
- 订单类型：`{cfg.order_type}`
- live 默认：`disabled`
- 非原子多腿真钱执行：`{"enabled" if cfg.allow_non_atomic_live_execution else "disabled"}`

## 上线安全门
- 官网订单状态是最高真相。
- 没有完整多腿成交把握就不交易。
- 任一腿失败必须进入 `unbalanced_exposure` 风险状态。
- 扫描 24 小时没有机会也属于正常结果。
"""


if __name__ == "__main__":
    raise SystemExit(main())
