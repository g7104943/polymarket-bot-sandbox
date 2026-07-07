#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from optimize_shock_risk_v2 import (
    DEFAULT_PANEL,
    build_cells,
    build_folds,
    evaluate_params,
    load_panel,
    params_from_dict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "shock_risk_v2_tuning_report.json"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cluster_key_to_tuple(cluster_key: str) -> Tuple[str, str]:
    if "_" not in cluster_key:
        return ("default", "BTC")
    profile, symbol = cluster_key.split("_", 1)
    return profile, symbol


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="按新 SHOCK 覆盖率口径重评估已有报告")
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--panel", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--min-trigger-fold-coverage", type=float, default=0.70)
    ap.add_argument("--min-trigger-fold-coverage-recent", type=float, default=None)
    ap.add_argument("--trigger-coverage-recent-folds", type=int, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    report_path = args.report
    if not report_path.exists():
        raise SystemExit(f"report not found: {report_path}")

    report = _load_json(report_path)
    options = report.get("options") if isinstance(report, dict) else {}
    constraints = options.get("constraints") if isinstance(options, dict) else {}
    build_meta = report.get("build_meta") if isinstance(report, dict) else {}

    panel_path = args.panel
    if panel_path is None:
        # 优先读报告顶层 panel（optimize_shock_risk_v2.py 的标准输出），
        # 再兼容旧版本 options.panel。
        raw_panel = report.get("panel") if isinstance(report, dict) else None
        if not raw_panel:
            raw_panel = options.get("panel")
        panel_path = Path(str(raw_panel)) if raw_panel else DEFAULT_PANEL
    if not panel_path.is_absolute():
        panel_path = (PROJECT_ROOT / panel_path).resolve()

    assets = options.get("assets") if isinstance(options, dict) else None
    if not isinstance(assets, list) or not assets:
        assets = ["BTC", "ETH"]
    assets = [str(x).upper() for x in assets]

    window_days = int(options.get("window_days", 365))
    half_life_days = int(options.get("half_life_days", 30))
    max_trades_per_cell = int(
        ((build_meta or {}).get("sampling") or {}).get("maxTradesPerCell", 1500)
    )
    cv_protocol = str(options.get("cv_protocol", "walk_forward"))
    train_min_days = int(options.get("train_min_days", 180))
    test_days = int(options.get("test_days", 14))
    step_days = int(options.get("step_days", 14))
    purge_hours = int(options.get("purge_hours", 24))

    net_drop_max = float(constraints.get("net_drop_max", 0.08))
    mdd_improve_min = float(constraints.get("mdd_improve_min", 0.10))
    suppression_max = float(constraints.get("suppression_max", 0.40))
    wr_drop_max = float(constraints.get("wr_drop_max", 0.01))
    min_trigger_count = float(constraints.get("min_trigger_count", 5.0))
    min_trigger_rate = float(constraints.get("min_trigger_rate", 0.0001))
    min_trigger_fold_coverage = float(args.min_trigger_fold_coverage)
    min_trigger_fold_coverage_recent = float(
        constraints.get("min_trigger_fold_coverage_recent", min_trigger_fold_coverage)
    )
    if args.min_trigger_fold_coverage_recent is not None:
        min_trigger_fold_coverage_recent = float(args.min_trigger_fold_coverage_recent)
    trigger_coverage_recent_folds = int(constraints.get("trigger_coverage_recent_folds", 6))
    if args.trigger_coverage_recent_folds is not None:
        trigger_coverage_recent_folds = int(args.trigger_coverage_recent_folds)

    df = load_panel(panel_path, assets)
    max_ts = int(df["ts"].max())
    cutoff_ts = max_ts - int(window_days) * 86400
    buckets, new_build_meta = build_cells(
        df=df,
        cutoff_ts=cutoff_ts,
        half_life_days=half_life_days,
        max_trades_per_cell=max_trades_per_cell,
    )
    folds = build_folds(
        min_ts=int(new_build_meta["min_ts"]),
        max_ts=int(new_build_meta["max_ts"]),
        protocol=cv_protocol,
        train_min_days=train_min_days,
        test_days=test_days,
        step_days=step_days,
    )

    clusters = report.get("clusters") if isinstance(report, dict) else {}
    if not isinstance(clusters, dict):
        raise SystemExit("invalid report: clusters missing")

    out_clusters: Dict[str, Any] = {}
    for cluster_key in ("default_BTC", "default_ETH", "70_BTC", "70_ETH"):
        payload = clusters.get(cluster_key, {})
        best = payload.get("best", {}) if isinstance(payload, dict) else {}
        params_raw = best.get("params") if isinstance(best, dict) else None
        if not isinstance(params_raw, dict):
            out_clusters[cluster_key] = {
                "best": {
                    "final_is_feasible": False,
                    "is_feasible": False,
                    "constraint_violations": ["missing_best_params"],
                },
                "candidates": [],
                "cells": 0,
                "folds": len(folds),
            }
            continue

        params = params_from_dict(params_raw)
        profile, symbol = _cluster_key_to_tuple(cluster_key)
        cells = buckets.get((profile, symbol), [])
        ev = evaluate_params(
            cells=cells,
            params=params,
            folds=folds,
            purge_hours=purge_hours,
            min_trigger_count=min_trigger_count,
            min_trigger_rate=min_trigger_rate,
            min_trigger_fold_coverage=min_trigger_fold_coverage,
            min_trigger_fold_coverage_recent=min_trigger_fold_coverage_recent,
            trigger_coverage_recent_folds=trigger_coverage_recent_folds,
            net_drop_max=net_drop_max,
            mdd_improve_min=mdd_improve_min,
            suppression_max=suppression_max,
            wr_drop_max=wr_drop_max,
        )
        out_clusters[cluster_key] = {
            "cells": len(cells),
            "folds": len(folds),
            "best": {
                "params": params.as_dict(),
                "score": float(ev["score"]),
                "is_feasible": bool(ev["is_feasible"]),
                "final_is_feasible": bool(ev["is_feasible"]),
                "constraint_violations": list(ev["constraint_violations"]),
                "metrics": ev["metrics"],
                "fold_trigger_counts": list(ev["fold_trigger_counts"]),
                "nonzero_fold_count": int(ev["nonzero_fold_count"]),
                "fold_count": int(ev["fold_count"]),
                "trigger_fold_coverage": float(ev["trigger_fold_coverage"]),
                "constraints_snapshot": dict(ev["constraints_snapshot"]),
                "search_pass": best.get("search_pass"),
                "rejudge_source_report": str(report_path),
            },
            "candidates": [],
        }

    generated_at = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = args.output
    if output_path is None:
        output_path = PROJECT_ROOT / "reports" / f"shock_rejudge_{generated_at}.json"

    out = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "source_report": str(report_path),
        "panel": str(panel_path),
        "options": {
            "window_days": window_days,
            "half_life_days": half_life_days,
            "cv_protocol": cv_protocol,
            "train_min_days": train_min_days,
            "test_days": test_days,
            "step_days": step_days,
            "purge_hours": purge_hours,
            "assets": assets,
            "constraints": {
                "net_drop_max": net_drop_max,
                "mdd_improve_min": mdd_improve_min,
                "suppression_max": suppression_max,
                "wr_drop_max": wr_drop_max,
                "min_trigger_count": min_trigger_count,
                "min_trigger_rate": min_trigger_rate,
                "min_trigger_fold_coverage": min_trigger_fold_coverage,
                "min_trigger_fold_coverage_recent": min_trigger_fold_coverage_recent,
                "trigger_coverage_recent_folds": trigger_coverage_recent_folds,
            },
        },
        "build_meta": new_build_meta,
        "folds": [{"test_start": int(a), "test_end": int(b)} for a, b in folds],
        "clusters": out_clusters,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
