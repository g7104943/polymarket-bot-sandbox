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
REPORTS = ROOT / "reports"
POLY = ROOT / "polymarket"
MODELS = ROOT / "data" / "models"
RESET_REPORT = REPORTS / "vnext_entry_exit_reset_latest.json"
OUTPUT_PATH = REPORTS / "vnext_mainline_reset_latest.json"

V1_RESET_SCRIPT = ROOT / "scripts" / "reset_vnext_entry_exit_monitor_lane.py"

STOP_COMMANDS = [
    ROOT / "scripts" / "run_vnext_mainline_supervisor.sh",
    ROOT / "reload_launchctl_overlay_vnext_btceth_execution_v2.sh",
    ROOT / "reload_launchctl_multi_trading_monitor_only_execution_v2.sh",
    ROOT / "reload_launchctl_prediction_writer_vnext_btceth_execution_v2.sh",
    ROOT / "reload_launchctl_overlay_vnext_btceth_sequence_v3.sh",
    ROOT / "reload_launchctl_multi_trading_monitor_only_sequence_v3.sh",
    ROOT / "reload_launchctl_prediction_writer_vnext_btceth_sequence_v3.sh",
]

MODEL_DIRS = [
    MODELS / "vnext_btc_entry_exit_v1",
    MODELS / "vnext_eth_entry_exit_v1",
    MODELS / "vnext_btc_execution_v2",
    MODELS / "vnext_eth_execution_v2",
    MODELS / "vnext_btc_profit_alpha_v2",
    MODELS / "vnext_eth_profit_alpha_v2",
    MODELS / "vnext_btc_sequence_v3",
    MODELS / "vnext_eth_sequence_v3",
]

POLY_RUNTIME_DIRS = [
    POLY / "logs_vnext_btc_execution_v2",
    POLY / "logs_vnext_eth_execution_v2",
    POLY / "logs_vnext_btc_execution_v2_raw",
    POLY / "logs_vnext_eth_execution_v2_raw",
    POLY / "logs_vnext_btc_sequence_v3",
    POLY / "logs_vnext_eth_sequence_v3",
    POLY / "logs_vnext_btc_sequence_v3_raw",
    POLY / "logs_vnext_eth_sequence_v3_raw",
]

POLY_FILES = [
    POLY / "monitor_only_traders_entry_exit.json",
    POLY / "monitor_only_traders_execution_v2.json",
    POLY / "active_traders_monitor_only_execution_v2.json",
    POLY / "trader_configs_monitor_only_execution_v2.json",
    POLY / "monitor_only_traders_sequence_v3.json",
    POLY / "active_traders_monitor_only_sequence_v3.json",
    POLY / "trader_configs_monitor_only_sequence_v3.json",
    POLY / "predictions_vnext_btceth_execution_v2.json",
    POLY / "predictions_vnext_btceth_execution_v2_ledger.json",
    POLY / "predictions_vnext_btceth_sequence_v3.json",
    POLY / "predictions_vnext_btceth_sequence_v3_ledger.json",
]

REPORT_FILES = [
    REPORTS / "vnext_entry_exit_v1_alignment_audit_latest.json",
    REPORTS / "vnext_entry_exit_v1_realign_status_latest.json",
    REPORTS / "vnext_execution_v2_readiness_latest.json",
    REPORTS / "vnext_execution_v2_training_latest.json",
    REPORTS / "vnext_execution_v2_audit_latest.json",
    REPORTS / "vnext_execution_v2_historical_bootstrap_latest.json",
    REPORTS / "vnext_profit_relabel_v2_latest.json",
    REPORTS / "vnext_profit_alpha_v2_training_latest.json",
    REPORTS / "vnext_sequence_v3_training_latest.json",
    REPORTS / "vnext_sequence_v3_audit_latest.json",
    REPORTS / "vnext_execution_v2_promotion_latest.json",
    REPORTS / "vnext_sequence_v3_promotion_latest.json",
    REPORTS / "vnext_mainline_supervisor_latest.json",
]

TEMPLATE_DEFAULTS = {
    POLY / "monitor_only_traders_entry_exit.json": {"traders": []},
    POLY / "monitor_only_traders_execution_v2.json": {"traders": []},
    POLY / "active_traders_monitor_only_execution_v2.json": {"groups": []},
    POLY / "trader_configs_monitor_only_execution_v2.json": [],
    POLY / "monitor_only_traders_sequence_v3.json": {"traders": []},
    POLY / "active_traders_monitor_only_sequence_v3.json": {"groups": []},
    POLY / "trader_configs_monitor_only_sequence_v3.json": [],
}

PKILL_PATTERNS = [
    r"run_vnext_btceth_entry_exit_followup.sh",
    r"train_vnext_btceth_entry_exit_v1.py --force-episodes",
    r"watch_vnext_entry_exit_v1_status.py",
    r"run_vnext_execution_observation_recorder.sh",
    r"run_vnext_endpoint_v2_prepare_loop.sh",
    r"scripts/vnext_endpoint_v2.py readiness",
    r"scripts/vnext_endpoint_v2.py assert-manual-ready",
    r"scripts/vnext_endpoint_v2.py (build-execution-dataset|train-execution|build-profit-relabel|train-profit-alpha)",
    r"(train_vnext_sequence_v3.py|audit_vnext_sequence_v3.py|promote_vnext_sequence_v3_monitor.py)",
    r"prediction_writer_vnext_btceth_execution_v2.py",
    r"reconcile_vnext_btceth_execution_v2_overlay.py",
    r"prediction_writer_vnext_btceth_sequence_v3.py",
    r"reconcile_vnext_btceth_sequence_v3_overlay.py",
    r"supervise_vnext_mainline.py",
]

SCREEN_SESSIONS = [
    "vnext_v1_realign",
    "vnext_v1_watch",
    "vnext_exec_obs",
    "vnext_v2_prepare_loop",
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


def _run(cmd: list[str], cwd: Path = ROOT) -> dict[str, Any]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _stop_runtime() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for script in STOP_COMMANDS:
        if script.exists():
            rows.append(_run([str(script), "stop"]))
    for pattern in PKILL_PATTERNS:
        rows.append(_run(["pkill", "-f", pattern]))
    for session in SCREEN_SESSIONS:
        rows.append(_run(["screen", "-S", session, "-X", "quit"]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Reset and archive all compare-only vnext mainline lanes")
    ap.add_argument("--initial-capital", type=float, default=400.0)
    ap.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = ap.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_root = POLY / "archived_vnext_mainline_resets" / ts

    stop_results = _stop_runtime()
    v1_reset = _run(["python3", str(V1_RESET_SCRIPT), "--initial-capital", str(args.initial_capital)])
    v1_reset_payload = _load_json(RESET_REPORT) if RESET_REPORT.exists() else {}

    archived_models = []
    for path in MODEL_DIRS:
        archived = _archive_path(path, archive_root / "models")
        if archived:
            archived_models.append(archived)

    archived_runtime_dirs = []
    for path in POLY_RUNTIME_DIRS:
        archived = _archive_path(path, archive_root / "polymarket_logs")
        if archived:
            archived_runtime_dirs.append(archived)

    archived_poly_files = []
    for path in POLY_FILES:
        archived = _archive_path(path, archive_root / "polymarket_files")
        if archived:
            archived_poly_files.append(archived)

    archived_reports = []
    for path in REPORT_FILES:
        archived = _archive_path(path, archive_root / "reports")
        if archived:
            archived_reports.append(archived)

    for path, payload in TEMPLATE_DEFAULTS.items():
        _write_json(path, payload)

    payload = {
        "generated_at": datetime.now().isoformat(),
        "scope": "vnext_mainline_compare_only_reset",
        "initial_capital_per_trader": float(args.initial_capital),
        "archive_root": str(archive_root),
        "stop_results": stop_results,
        "v1_reset_result": v1_reset,
        "v1_reset_payload": v1_reset_payload,
        "archived_models": archived_models,
        "archived_runtime_dirs": archived_runtime_dirs,
        "archived_polymarket_files": archived_poly_files,
        "archived_reports": archived_reports,
        "emptied_files": [str(path) for path in TEMPLATE_DEFAULTS],
    }
    _write_json(args.output, payload)
    _write_json(RESET_REPORT, payload)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
