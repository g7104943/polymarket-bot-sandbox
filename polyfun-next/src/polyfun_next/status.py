from __future__ import annotations

import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import os

from .config import load_config
from .official import check_v2_sdk


LEGACY_LIVE_PATTERN = "multi_prediction_index.js|lowprice_70_selected|prediction_writer_v5.py"


def legacy_live_processes() -> list[str]:
    out = subprocess.run(
        ["pgrep", "-fl", LEGACY_LIVE_PATTERN],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ).stdout.strip()
    return [line for line in out.splitlines() if _is_legacy_live_process(line)]


def _is_legacy_live_process(line: str) -> bool:
    """Detect old live/slot01 writer processes, not shell/search commands."""
    if not line.strip():
        return False
    command = line.split(" ", 1)[1] if " " in line else line
    if any(token in command for token in ("pgrep", "rg ", "grep", "git ", "status_report.py", "top159_monitor_report.py")):
        return False
    if "multi_prediction_index.js" in command and "--group lowprice_70_selected" in command:
        return True
    if "prediction_writer_v5.py" in command and (
        "slot01_4y_traditional_reduced_mid_v1" in command
        or "predictions_core10_v5_exp10_bp_dyn_0480_0510_eth.json" in command
    ):
        return True
    return False


def ledger_summary(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"exists": False, "events": 0, "truth_counts": {}}
    event_count = 0
    truths: Counter[str] = Counter()
    event_types: Counter[str] = Counter()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            event_count += 1
            row = json.loads(line)
            event_types[str(row.get("event_type", "unknown"))] += 1
            truth = _dig(row, "truth")
            if truth:
                truths[str(truth)] += 1
    return {
        "exists": True,
        "events": event_count,
        "event_type_counts": dict(event_types),
        "truth_counts": dict(truths),
        "official_missing": truths.get("official_missing", 0),
    }


def readiness(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    sdk = check_v2_sdk()
    legacy = legacy_live_processes()
    ledger = ledger_summary(root / "runtime" / "canary_ledger.jsonl")
    cfg = load_config(root / "config" / "canary.eth15m.json")
    required_env = [
        "POLYFUN_NEXT_PRIVATE_KEY",
        "POLYFUN_NEXT_API_KEY",
        "POLYFUN_NEXT_API_SECRET",
        "POLYFUN_NEXT_API_PASSPHRASE",
    ]
    missing_env = [name for name in required_env if not os.environ.get(name)]
    base_ok = sdk.installed and len(legacy) == 0 and ledger.get("official_missing", 0) == 0
    return {
        "sdk_installed": sdk.installed,
        "sdk_message": sdk.message,
        "legacy_live_process_count": len(legacy),
        "legacy_live_processes": legacy,
        "top159_risk_profile": {
            "strategy_id": cfg.strategy_id,
            "live_enabled": cfg.live_enabled,
            "symbol": cfg.symbol,
            "period": cfg.period,
            "preferred_order_type": cfg.preferred_order_type,
            "min_value_edge": cfg.min_value_edge,
            "max_entry_price": cfg.max_entry_price,
            "stake_fraction": cfg.stake_fraction,
            "daily_loss_stop_fraction": cfg.daily_loss_stop_fraction,
            "daily_loss_pause_seconds": cfg.daily_loss_pause_seconds,
            "failure_pause_count": cfg.failure_pause_count,
            "failure_pause_seconds": cfg.failure_pause_seconds,
            "min_depth_multiplier": cfg.min_depth_multiplier,
            "entry_window_seconds": [cfg.entry_window_start_seconds, cfg.entry_window_end_seconds],
        },
        "live_env_missing": missing_env,
        "ledger": ledger,
        "ready_for_dry_run": sdk.installed and len(legacy) == 0,
        "ready_for_min_manual_live_test": base_ok and not missing_env,
        "ready_for_automatic_live_canary": False,
        "automatic_live_canary_reason": "requires successful 20-order manual/minimal live acceptance first",
    }


def _dig(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        if key in value:
            return value[key]
        for child in value.values():
            found = _dig(child, key)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _dig(child, key)
            if found is not None:
                return found
    return None
