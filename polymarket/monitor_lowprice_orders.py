#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from monitor_all_orders import main as monitor_main
from dual_eth_live_truth_common import resolve_live_slot_specs
import os

ROOT = Path("/Users/mac/polyfun")
REPORTS = ROOT / "reports"
POLY = ROOT / "polymarket"
CURRENT_TRUTH = REPORTS / "dual_eth_live_current_truth_snapshot_latest.json"
REFRESH_CMDS = [
    ["python3", str(ROOT / "scripts" / "ops" / "generate_dual_eth_live_current_truth_snapshot_latest.py")],
    ["python3", str(ROOT / "scripts" / "ops" / "generate_dual_eth_live_settled_snapshot_latest.py")],
    ["python3", str(ROOT / "scripts" / "ops" / "generate_dual_eth_live_performance_attribution_latest.py")],
    ["python3", str(ROOT / "scripts" / "ops" / "generate_slot01_slot02_real_trade_chain_deep_audit_latest.py")],
    ["python3", str(ROOT / "scripts" / "ops" / "claim_status_report.py"), "--slot", "active"],
    ["python3", str(ROOT / "scripts" / "ops" / "generate_slot01_slot02_v2_10pass_full_audit_latest.py")],
]


def _current_live_expectations() -> dict[str, Path]:
    specs = resolve_live_slot_specs()
    expected: dict[str, Path] = {}
    selected = json.loads((POLY / "live_selected_cells.json").read_text(encoding="utf-8"))
    rows = selected.get("selected_cells") if isinstance(selected, dict) else []
    logs_by_trader = {
        str(row.get("trader") or "").strip(): str(row.get("logsDir") or "").strip()
        for row in rows or []
        if isinstance(row, dict) and row.get("enabled") is True
    }
    for slot in ("slot01", "slot02"):
        spec = specs.get(slot) or {}
        trader = str(spec.get("trader") or "").strip()
        if not trader:
            continue
        logs_dir = logs_by_trader.get(trader)
        if not logs_dir:
            continue
        expected[trader] = POLY / logs_dir / "reports" / "report_summary.live.json"
    return expected


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _needs_live_truth_refresh() -> bool:
    if not CURRENT_TRUTH.exists():
        return True
    payload = _load_json(CURRENT_TRUTH)
    rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
    row_map = {
        str(row.get("trader") or "").strip(): row
        for row in rows
        if isinstance(row, dict) and str(row.get("trader") or "").strip()
    }
    expected = _current_live_expectations()
    for trader, report_path in expected.items():
        row = row_map.get(trader)
        if not row or not report_path.exists():
            return True
        stored = str(row.get("sourceReportUpdatedAt") or row.get("reportSummaryUpdatedAt") or "").strip()
        if not stored:
            return True
        try:
            stored_dt = datetime.fromisoformat(stored.replace("Z", "+00:00")).timestamp()
        except Exception:
            return True
        if report_path.stat().st_mtime > stored_dt:
            return True
    return False


def _refresh_live_truth_reports_if_needed() -> None:
    if not _needs_live_truth_refresh():
        return
    for cmd in REFRESH_CMDS:
        try:
            subprocess.run(
                cmd,
                cwd=str(ROOT),
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=20,
            )
        except Exception:
            continue


def _run_top159_monitor_if_default() -> int | None:
    """In top159 cutover mode profile 70 shows only the new slot1/top159 truth.

    Use --legacy or POLYFUN_MONITOR_LEGACY=1 to force the old lowprice monitor.
    """
    if "--legacy" in sys.argv:
        sys.argv = [arg for arg in sys.argv if arg != "--legacy"]
        return None
    if os.environ.get("POLYFUN_MONITOR_LEGACY") == "1":
        return None
    # Keep this wrapper compatible with historical invocations. Only profile 70 is redirected.
    args = sys.argv[1:]
    if "--profile" in args:
        try:
            profile = args[args.index("--profile") + 1]
        except Exception:
            profile = ""
        if profile != "70":
            return None
    top_script = ROOT / "polymarket" / "monitor_top159_orders.py"
    if not top_script.exists():
        return None
    result = subprocess.run(
        ["python3", str(top_script)],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(result.stdout, end="")
    return result.returncode


def main() -> int:
    top159_rc = _run_top159_monitor_if_default()
    if top159_rc is not None:
        return top159_rc
    _refresh_live_truth_reports_if_needed()
    extra_args = list(sys.argv[1:])
    if "--mode" not in extra_args:
        extra_args = ["--mode", "both", *extra_args]
    sys.argv = [
        sys.argv[0],
        "--config-set",
        "lowprice",
        "--monitor-set",
        "lowprice",
        "--scope",
        "all",
        *extra_args,
    ]
    return monitor_main()


if __name__ == "__main__":
    raise SystemExit(main())
