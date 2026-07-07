#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"
OUT_FILE = REPORTS_DIR / "online_learning_group_best_params.json"
TARGET_GROUPS = ("v5_short", "v5_long", "gru_core")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _score_of(g: Dict[str, Any]) -> float:
    return float(g.get("best_summary", {}).get("score", -1e18))


def build_profile(files: List[Path]) -> Dict[str, Any]:
    best: Dict[str, Dict[str, Any]] = {}
    source: Dict[str, str] = {}

    for fp in files:
        try:
            payload = _load_json(fp)
        except Exception:
            continue
        groups = payload.get("groups") or payload.get("group_results") or {}
        if not isinstance(groups, dict):
            continue
        for gname, gdata in groups.items():
            if gname not in TARGET_GROUPS:
                continue
            if not isinstance(gdata, dict) or not gdata.get("best_params"):
                continue
            cand_score = _score_of(gdata)
            if gname not in best or cand_score > _score_of(best[gname]):
                best[gname] = gdata
                source[gname] = str(fp)

    missing = [g for g in TARGET_GROUPS if g not in best]
    if missing:
        raise SystemExit(f"缺少分组结果: {missing}")

    return {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "groups": {g: best[g] for g in TARGET_GROUPS},
        "sources": source,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="汇总在线增量分组超参结果为统一 profile")
    ap.add_argument(
        "--files",
        type=str,
        default="",
        help="逗号分隔 tuning json 路径；空=自动扫描 reports/online_learning_group_tuning_*.json",
    )
    ap.add_argument("--output", type=str, default=str(OUT_FILE), help="输出 profile 路径")
    args = ap.parse_args()

    if args.files.strip():
        files = [Path(x.strip()) for x in args.files.split(",") if x.strip()]
    else:
        files = sorted(
            REPORTS_DIR.glob("online_learning_group_tuning_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    if not files:
        raise SystemExit("没有找到 online_learning_group_tuning_*.json")

    profile = build_profile(files)
    out = Path(args.output)
    if not out.is_absolute():
        out = (PROJECT_ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ 已生成: {out}")
    for g in TARGET_GROUPS:
        s = profile["groups"][g].get("best_summary", {}).get("score")
        print(f"  {g}: score={s}  source={profile['sources'].get(g)}")


if __name__ == "__main__":
    main()
