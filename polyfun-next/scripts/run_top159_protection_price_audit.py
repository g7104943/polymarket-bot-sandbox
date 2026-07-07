#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path("/Users/mac/polyfun")
NEXT = ROOT / "polyfun-next"
REPORTS = ROOT / "reports"
DRYRUN = NEXT / "runtime" / "top159_official_orderbook_dryrun.jsonl"
LOCAL_HYPEROPT = REPORTS / "top159_local_orderbook_hyperopt_latest.json"
OUT_JSON = REPORTS / "top159_protection_price_hyperopt_latest.json"
OUT_MD = REPORTS / "top159_protection_price_hyperopt_latest.md"

FIXED_CAPS = [round(x / 100, 2) for x in range(50, 81)]
EDGES = [0.0, 0.01, 0.02, 0.03, 0.045, 0.06]
MIN_OFFICIAL_RESOLVED = 50


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def fnum(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def observed_values(row: dict[str, Any]) -> dict[str, Any]:
    sig = row.get("signal") or {}
    book = row.get("orderbook") or {}
    policy = row.get("policyEvaluation") or {}
    score = fnum(sig.get("modelScore"))
    ask = fnum(book.get("bestAsk"))
    spread = fnum(book.get("spread"))
    depth = fnum(book.get("askDepthTop3Shares"))
    won = row.get("won")
    if won is None:
        won = row.get("finalWon")
    if isinstance(won, str):
        won = won.lower() in {"true", "1", "yes", "won", "win"}
    return {
        "score": score,
        "ask": ask,
        "spread": spread,
        "depth": depth,
        "entryAllowed": bool((row.get("entryWindow") or {}).get("allowed")),
        "policyAccepted": bool(policy.get("accepted")),
        "won": won if isinstance(won, bool) else None,
    }


def pass_candidate(v: dict[str, Any], *, cap: float | None, edge: float) -> bool:
    score = v["score"]
    ask = v["ask"]
    if score is None or ask is None:
        return False
    if not v["entryAllowed"]:
        return False
    if cap is not None and ask > cap:
        return False
    if score < 0.5 + edge:
        return False
    if score - ask < edge:
        return False
    return True


def official_candidate_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values = [observed_values(r) for r in rows]
    table: list[dict[str, Any]] = []

    configs: list[tuple[str, float | None, float]] = [("current_0.52_edge0.045", 0.52, 0.045)]
    for edge in EDGES:
        configs.append((f"no_cap_edge{edge:g}", None, edge))
    for cap in FIXED_CAPS:
        for edge in EDGES:
            configs.append((f"fixed_cap{cap:.2f}_edge{edge:g}", cap, edge))

    for name, cap, edge in configs:
        selected = [v for v in values if pass_candidate(v, cap=cap, edge=edge)]
        resolved = [v for v in selected if isinstance(v.get("won"), bool)]
        wins = sum(1 for v in resolved if v["won"])
        losses = sum(1 for v in resolved if not v["won"])
        asks = [v["ask"] for v in selected if v["ask"] is not None]
        table.append(
            {
                "candidate": name,
                "maxEntryPrice": cap,
                "minValueEdge": edge,
                "observations": len(values),
                "selected": len(selected),
                "resolved": len(resolved),
                "wins": wins,
                "losses": losses,
                "winRatePct": round(100 * wins / len(resolved), 6) if resolved else None,
                "avgAsk": round(sum(asks) / len(asks), 6) if asks else None,
                "minAsk": round(min(asks), 6) if asks else None,
                "maxAsk": round(max(asks), 6) if asks else None,
            }
        )
    return table


def official_ask_buckets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        v = observed_values(row)
        ask = v["ask"]
        if ask is None:
            bucket = "missing"
        else:
            low = int(ask * 10) / 10
            high = low + 0.1
            bucket = f"{low:.1f}-{high:.1f}"
        buckets[bucket].append(v)
    out = []
    for bucket, vals in sorted(buckets.items()):
        resolved = [v for v in vals if isinstance(v.get("won"), bool)]
        wins = sum(1 for v in resolved if v["won"])
        asks = [v["ask"] for v in vals if v["ask"] is not None]
        out.append(
            {
                "askBucket": bucket,
                "observations": len(vals),
                "resolved": len(resolved),
                "wins": wins,
                "losses": len(resolved) - wins,
                "winRatePct": round(100 * wins / len(resolved), 6) if resolved else None,
                "avgAsk": round(sum(asks) / len(asks), 6) if asks else None,
            }
        )
    return out


def local_summary(local: dict[str, Any] | None) -> dict[str, Any]:
    if not local:
        return {"exists": False, "usableForPriceDecision": False, "reason": "local hyperopt report missing"}
    rows = local.get("rows") or []
    strict_coverages = [fnum(r.get("strictEntryCoveragePct")) or 0.0 for r in rows]
    any_coverages = [fnum(r.get("orderbookCoveragePct")) or 0.0 for r in rows]
    filled = [int(r.get("filledTrades") or 0) for r in rows]
    audit = local.get("bookAudit") or {}
    usable = max(strict_coverages or [0.0]) >= 20.0 and max(filled or [0]) >= 100
    return {
        "exists": True,
        "bookAudit": audit,
        "rows": len(rows),
        "maxAnyCoveragePct": max(any_coverages or [0.0]),
        "maxStrictEntryCoveragePct": max(strict_coverages or [0.0]),
        "maxFilledTrades": max(filled or [0]),
        "usableForPriceDecision": usable,
        "reason": "本地历史订单簿严格入场窗口覆盖足够" if usable else "本地历史订单簿覆盖不足，不能单独决定保护价",
        "uniqueVerdict": local.get("uniqueVerdict"),
    }


def make_verdict(official_rows: list[dict[str, Any]], official_table: list[dict[str, Any]], local: dict[str, Any]) -> dict[str, Any]:
    resolved = max((r.get("resolved") or 0 for r in official_table), default=0)
    official_enough = resolved >= MIN_OFFICIAL_RESOLVED
    local_usable = bool(local.get("usableForPriceDecision"))
    if not official_enough:
        return {
            "status": "keep_current_collect_more_official_samples",
            "selected": {"maxEntryPrice": 0.52, "minValueEdge": 0.045},
            "reason": f"官方 dry-run 已解析样本不足：resolved={resolved}，至少需要 {MIN_OFFICIAL_RESOLVED} 个已结样本才可证明高价单胜率。",
        }
    if not local_usable:
        return {
            "status": "keep_current_local_history_insufficient",
            "selected": {"maxEntryPrice": 0.52, "minValueEdge": 0.045},
            "reason": "官方样本即使足够，也需要本地历史盘口或压力口径不冲突；当前本地历史订单簿覆盖不足。",
        }
    # Conservative placeholder: only select a candidate with resolved win rate above average ask + edge.
    scored = []
    for row in official_table:
        if not row.get("resolved") or not row.get("avgAsk") or row.get("winRatePct") is None:
            continue
        edge = float(row["minValueEdge"])
        if row["resolved"] >= MIN_OFFICIAL_RESOLVED and row["winRatePct"] / 100.0 >= float(row["avgAsk"]) + edge:
            scored.append(row)
    if not scored:
        return {
            "status": "keep_current_no_high_price_proof",
            "selected": {"maxEntryPrice": 0.52, "minValueEdge": 0.045},
            "reason": "没有候选能证明真实胜率高于买价隐含概率加安全边际。",
        }
    best = max(scored, key=lambda r: (r["selected"], r["winRatePct"], -(r["avgAsk"] or 0)))
    return {
        "status": "candidate_for_preflight_only",
        "selected": {"maxEntryPrice": best["maxEntryPrice"], "minValueEdge": best["minValueEdge"]},
        "reason": "官方已结 dry-run 与本地盘口覆盖同时过线；该候选仍需单独 live preflight，不会自动改配置。",
        "candidate": best,
    }


def render_md(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# top159 保护价重超参审计")
    lines.append("")
    lines.append(f"生成时间：`{payload['generatedAt']}`")
    lines.append("")
    lines.append("## 官方数据真相")
    truth = payload["officialDataTruth"]
    lines.append("- 官方当前订单簿：可用，来自 `https://clob.polymarket.com/book?token_id=...`。")
    lines.append("- 官方实时盘口：可用，WebSocket market channel 可流式采集。")
    lines.append("- 官方成交/活动：可用，activity / user channel 可对账。")
    lines.append("- 完整 180/365 天官方历史订单簿：未发现官方可直接回放接口；不能用官方当前盘口伪造历史。")
    lines.append(f"- 本次 dry-run 样本：`{truth['officialDryRunObservations']}`，已结可验样本：`{truth['resolvedOfficialObservations']}`。")
    lines.append("")
    lines.append("## 本地历史订单簿覆盖")
    local = payload["localHistoricalOrderbook"]
    lines.append(f"- 可用作定价结论：`{local.get('usableForPriceDecision')}`")
    lines.append(f"- 最大严格入场覆盖：`{local.get('maxStrictEntryCoveragePct', 0):.6f}%`")
    lines.append(f"- 最大成交数：`{local.get('maxFilledTrades', 0)}`")
    lines.append(f"- 原因：{local.get('reason')}")
    lines.append("")
    lines.append("## 官方 dry-run 候选表（未足样本前只看可行性，不当盈利结论）")
    lines.append("|候选|最高买价|价值边际|样本|选中|已结|胜/负|胜率|平均买价|最低/最高买价|")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    show = []
    for row in payload["officialCandidateTable"]:
        if row["candidate"] == "current_0.52_edge0.045" or row["selected"] > 0:
            show.append(row)
    for row in show[:80]:
        cap = "无" if row["maxEntryPrice"] is None else f"{row['maxEntryPrice']:.2f}"
        wr = "-" if row["winRatePct"] is None else f"{row['winRatePct']:.2f}%"
        avg = "-" if row["avgAsk"] is None else f"{row['avgAsk']:.4f}"
        mm = "-" if row["minAsk"] is None else f"{row['minAsk']}/{row['maxAsk']}"
        lines.append(f"|{row['candidate']}|{cap}|{row['minValueEdge']:.3f}|{row['observations']}|{row['selected']}|{row['resolved']}|{row['wins']}/{row['losses']}|{wr}|{avg}|{mm}|")
    lines.append("")
    lines.append("## 买价分桶")
    lines.append("|买价桶|样本|已结|胜/负|胜率|平均买价|")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in payload["officialAskBuckets"]:
        wr = "-" if row["winRatePct"] is None else f"{row['winRatePct']:.2f}%"
        avg = "-" if row["avgAsk"] is None else f"{row['avgAsk']:.4f}"
        lines.append(f"|{row['askBucket']}|{row['observations']}|{row['resolved']}|{row['wins']}/{row['losses']}|{wr}|{avg}|")
    lines.append("")
    lines.append("## 唯一结论")
    v = payload["uniqueVerdict"]
    lines.append(f"- 状态：`{v['status']}`")
    lines.append(f"- 当前选择：`{v.get('selected')}`")
    lines.append(f"- 原因：{v['reason']}")
    lines.append("- 本轮不会改 live 配置；若要放宽最高买价，必须等官方 dry-run 有足够已结高价样本，并且本地盘口压力口径不冲突。")
    return "\n".join(lines) + "\n"


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    official_rows = read_jsonl(DRYRUN)
    official_table = official_candidate_table(official_rows)
    ask_buckets = official_ask_buckets(official_rows)
    local = local_summary(read_json(LOCAL_HYPEROPT))
    resolved = max((r.get("resolved") or 0 for r in official_table), default=0)
    payload = {
        "generatedAt": now_iso(),
        "scope": "top159 protection price official dry-run + local historical orderbook audit; research only; no live config change",
        "officialDataTruth": {
            "officialCurrentOrderbook": True,
            "officialRealtimeMarketChannel": True,
            "officialActivityAndUserChannel": True,
            "officialFullHistoricalOrderbookReplay": False,
            "officialDryRunPath": str(DRYRUN),
            "officialDryRunObservations": len(official_rows),
            "resolvedOfficialObservations": resolved,
            "minimumResolvedForDecision": MIN_OFFICIAL_RESOLVED,
        },
        "localHistoricalOrderbook": local,
        "candidateGrid": {
            "fixedCaps": FIXED_CAPS,
            "edges": EDGES,
            "includesNoCap": True,
            "currentLiveContract": {"maxEntryPrice": 0.52, "minValueEdge": 0.045},
        },
        "officialCandidateTable": official_table,
        "officialAskBuckets": ask_buckets,
        "uniqueVerdict": make_verdict(official_rows, official_table, local),
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    OUT_MD.write_text(render_md(payload), encoding="utf-8")
    print(json.dumps({"ok": True, "report": str(OUT_MD), "verdict": payload["uniqueVerdict"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
