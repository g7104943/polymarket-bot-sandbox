#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
UP_PANEL = REPORTS_DIR / "up_risk_v2_panel.parquet"
DOWN_PANEL = REPORTS_DIR / "down_risk_v2_panel.parquet"
DEFAULT_OUTPUT = REPORTS_DIR / "direction_calibration_panel.parquet"
DEFAULT_META = REPORTS_DIR / "direction_calibration_panel_meta.json"


def _read_panel(path: Path, direction: str, threshold_col: str) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"source panel not found: {path}")
    df = pd.read_parquet(path)
    required = {"timestamp", "ts", "profile", "traderName", "symbol", "cell_id", "confidence", "win", "pnl", threshold_col}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"panel missing columns {sorted(missing)}: {path}")
    out = df[list(required)].copy()
    out["direction"] = direction
    out["cluster_id"] = out["profile"].astype(str) + "_" + out["symbol"].astype(str) + "_" + direction
    out["base_conf_threshold"] = pd.to_numeric(out[threshold_col], errors="coerce")
    out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce")
    out["win"] = pd.to_numeric(out["win"], errors="coerce")
    out["pnl"] = pd.to_numeric(out["pnl"], errors="coerce")
    out["ts"] = pd.to_numeric(out["ts"], errors="coerce")
    out = out.dropna(subset=["base_conf_threshold", "confidence", "win", "pnl", "ts"])
    out["data_source"] = path.name
    return out[[
        "timestamp", "ts", "profile", "traderName", "symbol", "direction", "cell_id",
        "cluster_id", "confidence", "win", "pnl", "base_conf_threshold", "data_source",
    ]]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build direction calibration panel from long-history UP/DOWN risk panels")
    ap.add_argument("--up-panel", type=Path, default=UP_PANEL)
    ap.add_argument("--down-panel", type=Path, default=DOWN_PANEL)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--meta-output", type=Path, default=DEFAULT_META)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    up_df = _read_panel(args.up_panel, "UP", "up_conf_threshold")
    down_df = _read_panel(args.down_panel, "DOWN", "down_conf_threshold")
    panel_df = pd.concat([up_df, down_df], ignore_index=True).sort_values(
        ["cluster_id", "cell_id", "ts"]
    ).reset_index(drop=True)
    if panel_df.empty:
        raise RuntimeError("no calibration rows after merging up/down source panels")

    meta: Dict[str, Any] = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "sourcePanels": {
            "up": str(args.up_panel),
            "down": str(args.down_panel),
        },
        "anti_overfit": {
            "cluster_only": True,
            "cluster_count": int(panel_df["cluster_id"].nunique()),
            "cell_specific_params_forbidden": True,
        },
        "rows_total": int(len(panel_df)),
        "cells_total": int(panel_df["cell_id"].nunique()),
        "clusters_total": int(panel_df["cluster_id"].nunique()),
        "time_range": {
            "min": str(panel_df["timestamp"].min()),
            "max": str(panel_df["timestamp"].max()),
        },
        "panel_span_days": round((int(panel_df["ts"].max()) - int(panel_df["ts"].min())) / 86400.0, 3),
        "rows_by_cluster": {
            str(k): int(v) for k, v in panel_df.groupby("cluster_id").size().to_dict().items()
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.meta_output.parent.mkdir(parents=True, exist_ok=True)
    panel_df.to_parquet(args.output, index=False)
    args.meta_output.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote calibration panel: {args.output}")
    print(f"[INFO] rows={len(panel_df)} cells={panel_df['cell_id'].nunique()} clusters={panel_df['cluster_id'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
