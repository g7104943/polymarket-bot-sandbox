#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.exp_lowprice_selected_common import (  # noqa: E402
    PROFILES,
    active_names,
    dump_json,
    load_json,
    parse_symbols,
    resolve_calibration_mode_from_cfg,
    resolve_expectancy_mode_from_cfg,
    resolve_regime_mode_from_cfg,
    resolve_selector_mode_from_cfg,
)

POLY = ROOT / "polymarket"
REPORT = ROOT / "reports" / "risk_stack_audit_latest.json"
EXP_RUNTIME_REPORT = ROOT / "reports" / "exp_runtime_truth_latest.json"
LOWPRICE_PARITY_REPORT = ROOT / "reports" / "exp_lowprice_parity_audit_latest.json"
LOWPRICE_METHOD_REPORT = ROOT / "reports" / "exp_lowprice_method_audit_latest.json"
LOWPRICE_RUNTIME_REPORT = ROOT / "reports" / "exp_lowprice_runtime_audit_latest.json"

PROFILE_CONFIGS = {
    "default": {
        "config": POLY / "trader_configs.json",
        "active": POLY / "active_traders.json",
    },
    "70": {
        "config": POLY / "trader_configs_70.json",
        "active": POLY / "active_traders_70.json",
    },
}

SCALAR_MODE_KEYS = [
    "thresholdDriftMode",
    "metaLabelMode",
    "jointOptimizationMode",
    "jointOptimizationExpectedPayoffMode",
    "runtimeGuardMode",
    "overlayMode",
]


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def resolve_scalar_mode(cfg: dict[str, Any], key: str) -> str:
    raw = str(cfg.get(key) or "off").lower()
    if raw in {"off", "shadow", "enforce"}:
        return raw
    return "off"


def count_mode(bucket: dict[str, dict[str, int]], layer: str, mode: str) -> None:
    inner = bucket.setdefault(layer, {"off": 0, "shadow": 0, "enforce": 0})
    inner[mode] = inner.get(mode, 0) + 1


def build_active_mode_summary() -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for profile, meta in PROFILE_CONFIGS.items():
        rows = load_json(meta["config"])
        active = active_names(meta["active"])
        counts: dict[str, dict[str, int]] = {}
        row_count = 0
        for cfg in rows:
            if not isinstance(cfg, dict):
                continue
            name = str(cfg.get("name") or "")
            if name not in active or not name.startswith("v5_exp"):
                continue
            for symbol in parse_symbols(cfg.get("allowedMarkets")):
                row_count += 1
                count_mode(counts, "selector", resolve_selector_mode_from_cfg(cfg, symbol))
                count_mode(counts, "regime", resolve_regime_mode_from_cfg(cfg, symbol))
                for direction in ("UP", "DOWN"):
                    count_mode(counts, f"calibration_{direction}", resolve_calibration_mode_from_cfg(cfg, symbol, direction))
                    count_mode(counts, f"expectancy_{direction}", resolve_expectancy_mode_from_cfg(cfg, symbol, direction))
                for key in SCALAR_MODE_KEYS:
                    count_mode(counts, key, resolve_scalar_mode(cfg, key))
        payload[profile] = {
            "active_symbol_rows": row_count,
            "mode_counts": counts,
        }
    return payload


def run_prediction_side_audit() -> dict[str, Any]:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "ops" / "audit_prediction_side_controls.py")],
        capture_output=True,
        text=True,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    findings: list[dict[str, str]] = []
    for line in lines:
        if not line.startswith("[") or "] " not in line:
            continue
        prefix, rest = line.split("] ", 1)
        level = prefix.strip("[]")
        if ": " in rest:
            check, detail = rest.split(": ", 1)
        else:
            check, detail = rest, ""
        findings.append({"level": level, "check": check, "detail": detail})
    return {
        "exit_code": proc.returncode,
        "findings": findings,
        "stderr": proc.stderr.strip() or None,
    }


def main() -> None:
    exp_runtime = load_json(EXP_RUNTIME_REPORT)
    lowprice_parity = load_json(LOWPRICE_PARITY_REPORT)
    lowprice_method = load_json(LOWPRICE_METHOD_REPORT)
    lowprice_runtime = load_json(LOWPRICE_RUNTIME_REPORT)
    prediction_side = run_prediction_side_audit()

    payload = {
        "generatedAt": now_iso(),
        "runtime_defaults": exp_runtime.get("runtime_defaults"),
        "prediction_side_audit": prediction_side,
        "active_mode_summary": build_active_mode_summary(),
        "exp_runtime_issue_counts": exp_runtime.get("overall_issue_counts") or {},
        "lowprice_parity_issue_counts": {
            profile: ((lowprice_parity.get("profiles") or {}).get(profile) or {}).get("issue_counts") or {}
            for profile in ("default", "70")
        },
        "lowprice_method_verdict": lowprice_method.get("overall_verdict"),
        "lowprice_method_issue_counts": lowprice_method.get("overall_issue_counts") or {},
        "lowprice_runtime_classification_counts": {
            profile: ((lowprice_runtime.get("profiles") or {}).get(profile) or {}).get("classification_counts") or {}
            for profile in ("default", "70")
        },
        "ok": False,
    }
    payload["ok"] = (
        prediction_side.get("exit_code") == 0
        and not bool(payload["exp_runtime_issue_counts"])
        and all(not bool(v) for v in payload["lowprice_parity_issue_counts"].values())
    )
    dump_json(REPORT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
