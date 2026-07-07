#!/usr/bin/env python3
"""
按历史归档+当前交易记录的综合 PnL 选择 active 组合。

输出:
  - reports/top_exp_gru_selection.json
  - reports/top_exp_gru_selection_70.json
  - polymarket/active_traders.json
  - polymarket/active_traders_70.json
  - reports/pause_groups_top_exp_gru.txt
  - reports/pause_groups_top_exp_gru_70.txt
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLYMARKET_DIR = PROJECT_ROOT / "polymarket"
PROFILE_FILES = {
    "default": {
        "config": POLYMARKET_DIR / "trader_configs.json",
        "active": POLYMARKET_DIR / "active_traders.json",
        "report": PROJECT_ROOT / "reports" / "top_exp_gru_selection.json",
        "pause": PROJECT_ROOT / "reports" / "pause_groups_top_exp_gru.txt",
        "scope": "exp+gru",
    },
    "70": {
        "config": POLYMARKET_DIR / "trader_configs_70.json",
        "active": POLYMARKET_DIR / "active_traders_70.json",
        "report": PROJECT_ROOT / "reports" / "top_exp_gru_selection_70.json",
        "pause": PROJECT_ROOT / "reports" / "pause_groups_top_exp_gru_70.txt",
        "scope": "70 exp+gru",
    },
}


@dataclass
class ComboStats:
    name: str
    group: str
    logs_dir: str
    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0
    unique_ids: int = 0
    files_scanned: int = 0

    @property
    def win_rate(self) -> float:
        settled = self.wins + self.losses
        return (self.wins / settled) if settled > 0 else 0.0


def _num(v: Any) -> float | None:
    if isinstance(v, (int, float)):
        fv = float(v)
        if math.isfinite(fv):
            return fv
    return None


def _trade_key(e: dict[str, Any]) -> str:
    tid = e.get("id")
    if isinstance(tid, str) and tid:
        return tid
    ts = str(e.get("timestamp", ""))
    slug = str(e.get("marketSlug", ""))
    sym = str(e.get("symbol", ""))
    amt = str(e.get("amount", ""))
    direction = str(e.get("direction", ""))
    return f"{ts}|{slug}|{sym}|{amt}|{direction}"


def _collect_trade_files(logs_dir: str) -> list[Path]:
    out: list[Path] = []
    for stem in ("prediction_trades.simulation.json", "prediction_trades.json"):
        cur = POLYMARKET_DIR / logs_dir / stem
        if cur.exists():
            out.append(cur)
    for bdir in sorted(POLYMARKET_DIR.glob("backup_*")):
        for stem in ("prediction_trades.simulation.json", "prediction_trades.json"):
            p = bdir / logs_dir / stem
            if p.exists():
                out.append(p)
    return out


def _build_stats(cfg: dict[str, Any]) -> ComboStats:
    stats = ComboStats(
        name=str(cfg["name"]),
        group=str(cfg["group"]),
        logs_dir=str(cfg["logsDir"]),
    )
    seen_ids: set[str] = set()

    files = _collect_trade_files(stats.logs_dir)
    stats.files_scanned = len(files)

    for fp in files:
        try:
            raw = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, list):
            continue
        for e in raw:
            if not isinstance(e, dict):
                continue
            if str(e.get("mode", "simulation")).lower() != "simulation":
                continue
            k = _trade_key(e)
            if k in seen_ids:
                continue
            seen_ids.add(k)

            pnl = _num(e.get("simulatedPnl"))
            if pnl is None:
                pnl = _num(e.get("pnl"))
            if pnl is None:
                continue

            res = str(e.get("result", "")).lower()
            if res not in {"win", "lose"}:
                continue

            stats.trades += 1
            stats.pnl += pnl
            if res == "win":
                stats.wins += 1
            else:
                stats.losses += 1

    stats.unique_ids = len(seen_ids)
    return stats


def _collect_mandatory_trader_names(all_cfg: list[dict[str, Any]]) -> tuple[list[str], set[str]]:
    out: list[str] = []
    groups: set[str] = set()
    seen: set[str] = set()
    for row in all_cfg:
        if not isinstance(row, dict):
            continue
        group = str(row.get("group", "")).strip()
        if not group.startswith("ensemble"):
            continue
        groups.add(group)
        name = str(row.get("name", "")).strip()
        if not name or name in seen:
            continue
        out.append(name)
        seen.add(name)
    return out, groups


def _select_for_profile(
    profile: str,
    all_cfg: list[dict[str, Any]],
    top_ratio: float,
    min_trades: int,
    target_active_total: int,
    include_gru: bool,
) -> dict[str, Any]:
    target_cfg = []
    for c in all_cfg:
        if not isinstance(c, dict):
            continue
        group = str(c.get("group", "")).strip()
        if group.startswith("v5_exp") or (include_gru and group.startswith("gru_all")):
            target_cfg.append(c)
    if not target_cfg:
        raise RuntimeError(f"{profile}: no EXP/GRU configs found")

    rows = [_build_stats(c) for c in target_cfg]

    # 评分: 优先 pnl，其次样本量与胜率；低样本用轻微惩罚避免纯噪声。
    def score(r: ComboStats) -> tuple[float, int, float]:
        trade_penalty = 0.0
        if r.trades < min_trades:
            trade_penalty = (min_trades - r.trades) * 0.5
        return (r.pnl - trade_penalty, r.trades, r.win_rate)

    mandatory_names, mandatory_groups = _collect_mandatory_trader_names(all_cfg)
    rows.sort(key=score, reverse=True)

    if target_active_total > 0:
        keep_n = max(0, target_active_total - len(mandatory_names))
    else:
        keep_n = max(1, math.ceil(len(rows) * max(0.01, min(top_ratio, 1.0))))
    keep_n = min(len(rows), keep_n)

    keep = rows[:keep_n]
    drop = rows[keep_n:]

    keep_names = [r.name for r in keep]
    for name in mandatory_names:
        if name not in keep_names:
            keep_names.append(name)

    all_groups = sorted(
        {
            str(c.get("group", "")).strip()
            for c in all_cfg
            if isinstance(c, dict) and str(c.get("group", "")).strip()
        }
    )
    keep_groups = sorted({r.group for r in keep}.union(mandatory_groups))
    pause_groups = [g for g in all_groups if g not in keep_groups]
    return {
        "rows": rows,
        "keep": keep,
        "drop": drop,
        "keep_names": keep_names,
        "keep_groups": keep_groups,
        "pause_groups": pause_groups,
        "mandatory_groups": sorted(mandatory_groups),
        "selection_basis": "target_active_total" if target_active_total > 0 else "top_ratio",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Select top EXP/GRU combos by aggregated archived PnL")
    parser.add_argument("--top-ratio", type=float, default=0.30, help="keep ratio, default 0.30")
    parser.add_argument("--min-trades", type=int, default=10, help="minimum settled trades to trust score")
    parser.add_argument(
        "--target-active-total",
        type=int,
        default=30,
        help="target active trader count per profile including mandatory ensemble, default 30",
    )
    parser.add_argument(
        "--include-gru",
        action="store_true",
        help="include gru_all candidates in ranking (default off to match 30+30/120-cell universe)",
    )
    parser.add_argument(
        "--profiles",
        choices=("default", "70", "both"),
        default="both",
        help="which active universe(s) to refresh, default both",
    )
    args = parser.parse_args()

    profiles = ["default", "70"] if args.profiles == "both" else [args.profiles]
    generated_at = datetime.now(timezone.utc).isoformat()
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    for profile in profiles:
        pf = PROFILE_FILES[profile]
        cfg_path = pf["config"]
        if not cfg_path.exists():
            raise SystemExit(f"{profile}: missing config: {cfg_path}")
        all_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        selected = _select_for_profile(
            profile=profile,
            all_cfg=all_cfg,
            top_ratio=args.top_ratio,
            min_trades=args.min_trades,
            target_active_total=args.target_active_total,
            include_gru=bool(args.include_gru),
        )

        active_out = {
            "generatedAt": generated_at,
            "source": "select_top_exp_gru_traders.py",
            "topRatio": args.top_ratio,
            "minTrades": args.min_trades,
            "targetActiveTotal": args.target_active_total,
            "includeGru": bool(args.include_gru),
            "scope": pf["scope"],
            "source_window": "all_archives_plus_current_simulation",
            "mandatoryGroups": selected["mandatory_groups"],
            "selected_traders": selected["keep_names"],
            "selected_groups": selected["keep_groups"],
            "traderNames": selected["keep_names"],
            "active_traders": selected["keep_names"],
            "groups": selected["keep_groups"],
        }
        Path(pf["active"]).write_text(
            json.dumps(active_out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        report = {
            "generatedAt": generated_at,
            "profile": profile,
            "summary": {
                "totalCandidates": len(selected["rows"]),
                "keepCount": len(selected["keep"]),
                "dropCount": len(selected["drop"]),
                "activeCount": len(selected["keep_names"]),
                "topRatio": args.top_ratio,
                "minTrades": args.min_trades,
                "targetActiveTotal": args.target_active_total,
                "includeGru": bool(args.include_gru),
                "selectionBasis": selected["selection_basis"],
                "pauseGroups": selected["pause_groups"],
            },
            "selected_traders": selected["keep_names"],
            "selected_groups": selected["keep_groups"],
            "source_window": "all_archives_plus_current_simulation",
            "keep": [
                {
                    **asdict(r),
                    "win_rate": r.win_rate,
                }
                for r in selected["keep"]
            ],
            "drop": [
                {
                    **asdict(r),
                    "win_rate": r.win_rate,
                }
                for r in selected["drop"]
            ],
        }
        Path(pf["report"]).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        Path(pf["pause"]).write_text(
            ",".join(selected["pause_groups"]),
            encoding="utf-8",
        )

        print(f"[{profile}] 候选组合: {len(selected['rows'])}")
        if selected["selection_basis"] == "target_active_total":
            print(
                f"[{profile}] active 目标: {args.target_active_total} "
                f"(候选={len(selected['keep'])}, mandatory={len(selected['mandatory_groups'])}, "
                f"active={len(selected['keep_names'])})"
            )
        else:
            print(f"[{profile}] 保留前30%: {len(selected['keep'])} (active={len(selected['keep_names'])})")
        print(
            f"[{profile}] 暂停组: "
            f"{','.join(selected['pause_groups']) if selected['pause_groups'] else '(none)'}"
        )
        keep = selected["keep"]
        if keep:
            print(
                f"[{profile}] Top1: {keep[0].name} pnl={keep[0].pnl:.2f}, "
                f"trades={keep[0].trades}, wr={keep[0].win_rate:.2%}"
            )
        print(f"[{profile}] 输出: {pf['active']}")
        print(f"[{profile}] 输出: {pf['report']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
