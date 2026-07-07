#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
POLY = ROOT / "polymarket"
REPORTS = ROOT / "reports"
CONFIG_PATH = POLY / "trader_configs_monitor_only_entry_exit.json"
OUTPUT_PATH = REPORTS / "vnext_entry_exit_reset_latest.json"

TRADER_NAMES = {"vnext_btc_entry_exit_v1", "vnext_eth_entry_exit_v1"}
LOG_DIRS = [
    "logs_vnext_btc_entry_exit_v1_raw",
    "logs_vnext_eth_entry_exit_v1_raw",
    "logs_vnext_btc_entry_exit_v1",
    "logs_vnext_eth_entry_exit_v1",
]
POLY_FILES = [
    "predictions_vnext_btc_entry_exit_v1.json",
    "predictions_vnext_eth_entry_exit_v1.json",
    "predictions_vnext_btceth_entry_exit_v1.json",
    "predictions_vnext_btceth_entry_exit_v1_ledger.json",
    "predictions_vnext_btc_entry_exit_v1.writer_state.json",
    "predictions_vnext_eth_entry_exit_v1.writer_state.json",
]
REPORT_FILES = [
    "vnext_btc_entry_exit_v1_leaderboard_latest.json",
    "vnext_eth_entry_exit_v1_leaderboard_latest.json",
    "vnext_btceth_entry_exit_v1_compare_latest.json",
    "vnext_btceth_entry_exit_v1_training_latest.json",
]
STOP_COMMANDS = [
    ROOT / "reload_launchctl_overlay_vnext_btceth_entry_exit_v1.sh",
    ROOT / "reload_launchctl_multi_trading_monitor_only_entry_exit_v1.sh",
    ROOT / "reload_launchctl_prediction_writer_vnext_btceth_entry_exit_v1.sh",
]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _archive_path(src: Path, archive_root: Path) -> str | None:
    if not src.exists():
        return None
    archive_root.mkdir(parents=True, exist_ok=True)
    target = archive_root / src.name
    suffix = 1
    while target.exists():
        target = archive_root / f"{src.name}__dup{suffix}"
        suffix += 1
    shutil.move(str(src), str(target))
    return str(target)


def _stop_runtime() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for script in STOP_COMMANDS:
        proc = subprocess.run([str(script), "stop"], cwd=str(ROOT), capture_output=True, text=True)
        rows.append(
            {
                "script": str(script),
                "returncode": int(proc.returncode),
                "stdout": proc.stdout.strip(),
                "stderr": proc.stderr.strip(),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Reset vnext entry-exit compare-only lane to a clean starting capital")
    ap.add_argument("--initial-capital", type=float, default=400.0)
    ap.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_root = POLY / "archived_vnext_entry_exit_resets" / ts

    stop_results = _stop_runtime()

    archived_logs = []
    for name in LOG_DIRS:
        archived = _archive_path(POLY / name, archive_root / "logs")
        if archived:
            archived_logs.append(archived)

    archived_files = []
    for name in POLY_FILES:
        archived = _archive_path(POLY / name, archive_root / "polymarket_files")
        if archived:
            archived_files.append(archived)

    archived_reports = []
    for name in REPORT_FILES:
        archived = _archive_path(REPORTS / name, archive_root / "reports")
        if archived:
            archived_reports.append(archived)

    rows = _load_json(CONFIG_PATH)
    if not isinstance(rows, list):
        raise SystemExit(f"invalid config: {CONFIG_PATH}")
    changed = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("name") or "").strip() not in TRADER_NAMES:
            continue
        row["initialCapital"] = round(float(args.initial_capital), 4)
        row["perCoinCapital"] = False
        row["numTradingAssets"] = 1
        changed.append(
            {
                "name": row["name"],
                "allowedMarkets": row.get("allowedMarkets"),
                "initialCapital": row["initialCapital"],
            }
        )
    _write_json(CONFIG_PATH, rows)

    payload = {
        "generated_at": datetime.now().isoformat(),
        "scope": "vnext_entry_exit_monitor_lane_reset",
        "initial_capital_per_trader": float(args.initial_capital),
        "archive_root": str(archive_root),
        "stopped_runtime": stop_results,
        "archived_logs": archived_logs,
        "archived_polymarket_files": archived_files,
        "archived_reports": archived_reports,
        "updated_traders": changed,
    }
    _write_json(args.output, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
