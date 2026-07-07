#!/usr/bin/env python3
"""
Live/Sim 共存版订单监控

特性:
1. 动态发现目录（active + 常驻组，或全量）。
2. 支持 runtime 拆分目录（__live_* / __simulation_*）。
3. 支持 mode 过滤，默认 both。
4. 表格增加 mode 与 runtime logsDir。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import re

import live_rollout_evidence_common as live_rollout_common
from live_rollout_evidence_common import (
    build_live_rollout_evidence_map,
    load_selected_live_targets,
    read_mode_activity,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OPS_DIR = SCRIPT_DIR.parent / "scripts" / "ops"
if str(OPS_DIR) not in sys.path:
    sys.path.insert(0, str(OPS_DIR))

from lowprice_retirement_common import is_retired_compare_only_trader
from dual_eth_live_truth_common import build_current_truth_payload, TARGETS as DUAL_ETH_LIVE_TARGETS, resolve_live_slot_specs

MODE_REGISTRY_FILE = SCRIPT_DIR / "trading_mode_registry.json"
LOWPRICE_PROFILE_FILES = {
    "default": {
        "configs": SCRIPT_DIR / "trader_configs_monitor_only_lowprice.json",
        "active": SCRIPT_DIR / "active_traders_monitor_only_lowprice.json",
        "monitor": SCRIPT_DIR / "monitor_only_traders_lowprice.json",
        "logs_prefix": "logs_lowprice_selected_",
        "label_prefix": "",
    },
    "70": {
        "configs": SCRIPT_DIR / "trader_configs_monitor_only_lowprice_70.json",
        "active": SCRIPT_DIR / "active_traders_monitor_only_lowprice_70.json",
        "monitor": SCRIPT_DIR / "monitor_only_traders_lowprice_70.json",
        "logs_prefix": "logs_70_lowprice_selected_",
        "label_prefix": "70_",
    },
    "threshold_pack": {
        "configs": SCRIPT_DIR / "trader_configs_monitor_only_lowprice_threshold_pack.json",
        "active": SCRIPT_DIR / "active_traders_monitor_only_lowprice_threshold_pack.json",
        "monitor": SCRIPT_DIR / "monitor_only_traders_lowprice_threshold_pack.json",
        "logs_prefix": "logs_threshold_pack_lowprice_selected_",
        "label_prefix": "tp_",
    },
}
MONITOR_ONLY_TRADER_FILE_SETS = {
    "main": [
        SCRIPT_DIR / "monitor_only_traders.json",
        SCRIPT_DIR / "monitor_only_traders_v2.json",
        SCRIPT_DIR / "monitor_only_traders_entry_exit.json",
        SCRIPT_DIR / "monitor_only_traders_execution_v2.json",
        SCRIPT_DIR / "monitor_only_traders_sequence_v3.json",
    ],
    "lowprice": [
        LOWPRICE_PROFILE_FILES["default"]["monitor"],
        LOWPRICE_PROFILE_FILES["70"]["monitor"],
        LOWPRICE_PROFILE_FILES["threshold_pack"]["monitor"],
    ],
}
MONITOR_ONLY_TRADER_FILE_SETS["all"] = (
    MONITOR_ONLY_TRADER_FILE_SETS["main"] + MONITOR_ONLY_TRADER_FILE_SETS["lowprice"]
)
MONITOR_ONLY_ACTIVE_FILE_MAP = {
    str(SCRIPT_DIR / "monitor_only_traders.json"): SCRIPT_DIR / "active_traders_monitor_only.json",
    str(SCRIPT_DIR / "monitor_only_traders_v2.json"): SCRIPT_DIR / "active_traders_monitor_only_v2.json",
    str(SCRIPT_DIR / "monitor_only_traders_entry_exit.json"): SCRIPT_DIR / "active_traders_monitor_only_entry_exit.json",
    str(SCRIPT_DIR / "monitor_only_traders_execution_v2.json"): SCRIPT_DIR / "active_traders_monitor_only_execution_v2.json",
    str(SCRIPT_DIR / "monitor_only_traders_sequence_v3.json"): SCRIPT_DIR / "active_traders_monitor_only_sequence_v3.json",
    str(LOWPRICE_PROFILE_FILES["default"]["monitor"]): LOWPRICE_PROFILE_FILES["default"]["active"],
    str(LOWPRICE_PROFILE_FILES["70"]["monitor"]): LOWPRICE_PROFILE_FILES["70"]["active"],
    str(LOWPRICE_PROFILE_FILES["threshold_pack"]["monitor"]): LOWPRICE_PROFILE_FILES["threshold_pack"]["active"],
}
CORE_UNIVERSE_LATEST = SCRIPT_DIR.parent / "reports" / "core_universe_latest.json"
WIDE_OBSERVATION_LATEST = SCRIPT_DIR.parent / "reports" / "wide_universe_observation_latest.json"
ROTATION_WATCHLIST_LATEST = SCRIPT_DIR.parent / "reports" / "core10_rotation_watchlist_latest.json"
RECENT_WINDOW_WATCH_LATEST = SCRIPT_DIR.parent / "reports" / "core10_recent_window_watch_latest.json"
EXTREME_REGIME_STATE_LATEST = SCRIPT_DIR.parent / "reports" / "coremain_extreme_regime_state_latest.json"
EXTREME_OVERLAY_APPLY_LATEST = SCRIPT_DIR.parent / "reports" / "coremain_extreme_overlay_apply_latest.json"
MAINLINE_FINAL_STATUS_LATEST = SCRIPT_DIR.parent / "reports" / "polyfun_mainline_final_status_latest.json"
MAINLINE_SOURCE_AUDIT_LATEST = SCRIPT_DIR.parent / "reports" / "core10_prediction_source_audit_latest.json"
LIVE_TRUTH_CONTRACT_LATEST = SCRIPT_DIR.parent / "reports" / "live_truth_contract_latest.json"
LIVE_SELECTED_CELLS = SCRIPT_DIR / "live_selected_cells.json"
LIVE_ROLLOUT_EVIDENCE_LATEST = SCRIPT_DIR.parent / "reports" / "live_rollout_evidence_latest.json"
LIVE_RUNTIME_FINAL_TRUTH_LATEST = SCRIPT_DIR.parent / "reports" / "live_runtime_final_truth_ledger_latest.json"
LIVE_OPERATIONAL_AUDIT_LATEST = SCRIPT_DIR.parent / "reports" / "live_operational_audit_latest.json"
LIVE_BOOT_RUNTIME_AUDIT_LATEST = SCRIPT_DIR.parent / "reports" / "live_boot_runtime_audit_latest.json"
LIVE_CLAIM_STATUS_ACTIVE_LATEST = SCRIPT_DIR.parent / "reports" / "live_claim_status_active_latest.json"
DUAL_ETH_LIVE_RULE_PARITY_LATEST = SCRIPT_DIR.parent / "reports" / "dual_eth_live_rule_parity_latest.json"
DUAL_ETH_LIVE_CURRENT_TRUTH_SNAPSHOT_LATEST = SCRIPT_DIR.parent / "reports" / "dual_eth_live_current_truth_snapshot_latest.json"
DUAL_ETH_LIVE_SETTLED_SNAPSHOT_LATEST = SCRIPT_DIR.parent / "reports" / "dual_eth_live_settled_snapshot_latest.json"
DUAL_ETH_LIVE_TIMING_PARITY_LATEST = SCRIPT_DIR.parent / "reports" / "dual_eth_live_timing_parity_latest.json"
DUAL_ETH_LIVE_DIRECTION_COOLDOWN_TRUTH_LATEST = SCRIPT_DIR.parent / "reports" / "dual_eth_live_direction_cooldown_truth_latest.json"
DUAL_ETH_LIVE_SIZING_BACKTEST_LATEST = SCRIPT_DIR.parent / "reports" / "dual_eth_live_sizing_backtest_latest.json"
PROFILE70_FOUR_SIM_CHAIN_TRUTH_LATEST = SCRIPT_DIR.parent / "reports" / "profile70_four_sim_chain_truth_latest.json"
BTC_XRP_JOINT_WINNER_MONITOR_OBSERVATION_LATEST = SCRIPT_DIR.parent / "reports" / "btc_xrp_joint_winner_monitor_observation_latest.json"
SLOT01_MAPPED_COOLDOWN_SIMULATIONS_LATEST = SCRIPT_DIR.parent / "reports" / "slot01_mapped_cooldown_simulations_latest.json"
SLOT01_MAPPED_COOLDOWN_SIMULATIONS_GENERATOR = SCRIPT_DIR.parent / "scripts" / "ops" / "generate_slot01_mapped_cooldown_simulations_latest.py"
SLOT01_CANONICAL_TRADE_LEDGER_LATEST = SCRIPT_DIR.parent / "reports" / "slot01_canonical_trade_ledger_latest.json"
SLOT01_CANONICAL_TRADE_LEDGER_GENERATOR = SCRIPT_DIR.parent / "scripts" / "ops" / "generate_slot01_canonical_trade_ledger_latest.py"
SLOT01_MAPPED_EXTERNAL_CASH_ADJUSTMENT = SCRIPT_DIR.parent / "reports" / "slot01_mapped_external_cash_adjustment.json"
SLOT01_MAPPED_CASH_SYNC_BASELINE = SCRIPT_DIR.parent / "reports" / "slot01_mapped_cash_sync_baseline.json"
SLOT01_LIVE_REPORT_SUMMARY = SCRIPT_DIR / "logs_70_lowprice_selected_v5_exp10_bp_dyn_0300_0440_eth" / "reports" / "report_summary.live.json"
SLOT01_RAW_LIVE_TRADES = SCRIPT_DIR / "logs_70_lowprice_selected_v5_exp10_bp_dyn_0300_0440_eth" / "prediction_trades.live.json"
SLOT01_OFFICIAL_RECONCILIATION_LATEST = SCRIPT_DIR.parent / "reports" / "slot01_official_reconciliation_latest.json"
SLOT01_LIVE_WALLET_BALANCE_LATEST = SCRIPT_DIR.parent / "reports" / "live_wallet_balance_slot01_latest.json"
SIMULATION_LIVE_EQUIVALENT_LATEST = SCRIPT_DIR.parent / "reports" / "limit_price_fill_realism_gate_latest.json"
PROFILE70_HIGH_WIN_TRUTH_LATEST = SCRIPT_DIR.parent / "reports" / "profile70_high_win_truth_latest.json"
GUARD_TRUTH_INVENTORY_LATEST = SCRIPT_DIR.parent / "reports" / "guard_truth_inventory_latest.json"
EXP13_OFFICIAL_SIM_CANDIDATE_LATEST = SCRIPT_DIR.parent / "reports" / "core10_exp13_official_sim_candidate_latest.json"
ROUTE_F_EXP10_OFFICIAL_SIM_CANDIDATE_LATEST = SCRIPT_DIR.parent / "reports" / "core10_route_f_exp10_official_sim_candidate_latest.json"
ALWAYS_INCLUDE_GROUPS: set[str] = set()
SUPPORTED_MONITOR_SYMBOLS = {"BTC", "ETH", "SOL", "XRP"}
MODE_SUFFIX_RE = re.compile(r"__(simulation|live|backtest)(?:_[a-z0-9_]+)?$", re.IGNORECASE)
RUNTIME_SPLIT_SUFFIX_RE = re.compile(r"__(simulation|live|backtest)_[a-z0-9_]+$", re.IGNORECASE)
RUNTIME_SPLIT_ASSET_RE = re.compile(r"__(?:simulation|live|backtest)_([a-z0-9_]+)$", re.IGNORECASE)
RUNTIME_SAMPLE_SUFFIX_RE = re.compile(r"__(simulation|live|backtest)_[a-z0-9_]*sample(?:_[a-z0-9_]+)?$", re.IGNORECASE)
ARCHIVED_TRADER_PREFIXES: tuple[str, ...] = ()
ARCHIVED_LOG_DIR_PREFIXES: tuple[str, ...] = ()
_LIVE_TRUTH_ROWS_CACHE: Dict[Tuple[str, str, str, str], Dict[str, Any]] | None = None
_SIMULATION_LIVE_EQUIV_CACHE: Dict[Tuple[str, str, str], Dict[str, Any]] | None = None
_GUARD_TRUTH_ROWS_CACHE: Dict[Tuple[str, str, str, str], Dict[str, Any]] | None = None
_PROFILE70_HIGH_WIN_ROWS_CACHE: Dict[str, Dict[str, Any]] | None = None
_MONITOR_PREVIOUS_RENDER_STATE: Dict[str, Dict[str, Any]] = {}
_EARLY_ADMISSION_OBSERVATION_CACHE: Dict[Tuple[str, str], Dict[str, Any]] | None = None
_LIVE_CLAIM_STATUS_MAP_CACHE: Dict[str, Dict[str, Any]] | None = None
_DUAL_ETH_RULE_PARITY_MAP_CACHE: Dict[str, Dict[str, Any]] | None = None
_DUAL_ETH_CURRENT_TRUTH_MAP_CACHE: Dict[str, Dict[str, Any]] | None = None
_DUAL_ETH_SETTLED_SNAPSHOT_MAP_CACHE: Dict[str, Dict[str, Any]] | None = None
_DUAL_ETH_TRUTH_REPORT_REFRESH_ATTEMPTED = False
_DUAL_ETH_TIMING_PARITY_MAP_CACHE: Dict[str, Dict[str, Any]] | None = None
_DUAL_ETH_DIRECTION_COOLDOWN_MAP_CACHE: Dict[str, Dict[str, Any]] | None = None
_DUAL_ETH_SIZING_MAP_CACHE: Dict[str, Dict[str, Any]] | None = None
_PROFILE70_FOUR_SIM_CHAIN_TRUTH_CACHE: Dict[str, Dict[str, Any]] | None = None

_FRIENDLY_LIVE_LABELS: Dict[str, str] = {
    spec["trader"]: spec["display"]
    for spec in resolve_live_slot_specs().values()
    if spec.get("trader") and spec.get("display")
}


def _reset_monitor_runtime_caches() -> None:
    global _LIVE_TRUTH_ROWS_CACHE, _SIMULATION_LIVE_EQUIV_CACHE, _GUARD_TRUTH_ROWS_CACHE, _MODE_REGISTRY_CACHE, _EARLY_ADMISSION_OBSERVATION_CACHE, _PROFILE70_HIGH_WIN_ROWS_CACHE, _LIVE_CLAIM_STATUS_MAP_CACHE, _DUAL_ETH_CURRENT_TRUTH_MAP_CACHE, _DUAL_ETH_SETTLED_SNAPSHOT_MAP_CACHE, _DUAL_ETH_TIMING_PARITY_MAP_CACHE, _PROFILE70_FOUR_SIM_CHAIN_TRUTH_CACHE
    _LIVE_TRUTH_ROWS_CACHE = None
    _SIMULATION_LIVE_EQUIV_CACHE = None
    _GUARD_TRUTH_ROWS_CACHE = None
    _PROFILE70_HIGH_WIN_ROWS_CACHE = None
    _LIVE_CLAIM_STATUS_MAP_CACHE = None
    _DUAL_ETH_CURRENT_TRUTH_MAP_CACHE = None
    _DUAL_ETH_SETTLED_SNAPSHOT_MAP_CACHE = None
    _DUAL_ETH_TIMING_PARITY_MAP_CACHE = None
    _PROFILE70_FOUR_SIM_CHAIN_TRUTH_CACHE = None
    _MODE_REGISTRY_CACHE = None
    _EARLY_ADMISSION_OBSERVATION_CACHE = None
    _ACTIVE_CONTRACT_CACHE.clear()
    _LOWPRICE_EXPECTED_COMPARE_LABELS_CACHE.clear()
    _LOWPRICE_EXPECTED_COMPARE_TRADERS_CACHE.clear()
    for cache_clear in (
        getattr(live_rollout_common, "_fetch_public_activity_trades", None),
        getattr(live_rollout_common, "_fetch_positions_for_user", None),
        getattr(live_rollout_common, "_load_claim_daemon_slots", None),
    ):
        clear_fn = getattr(cache_clear, "cache_clear", None)
        if callable(clear_fn):
            clear_fn()


def _path_mtime_iso(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat() if path.exists() else None
    except Exception:
        return None


def _monitor_parse_iso(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None


def _row_source_report_updated_at(row: Dict[str, Any]) -> str | None:
    role = str(row.get("role") or "").strip().lower()
    if role == "live_rollout" or str(row.get("mode") or "").strip().lower() == "live":
        candidates = [
            _path_mtime_iso(LIVE_ROLLOUT_EVIDENCE_LATEST),
            _path_mtime_iso(LIVE_RUNTIME_FINAL_TRUTH_LATEST),
            _path_mtime_iso(LIVE_OPERATIONAL_AUDIT_LATEST),
            _path_mtime_iso(LIVE_BOOT_RUNTIME_AUDIT_LATEST),
            _path_mtime_iso(LIVE_TRUTH_CONTRACT_LATEST),
        ]
        parsed = [dt for dt in (_monitor_parse_iso(value) for value in candidates) if dt is not None]
        if not parsed:
            return None
        return max(parsed).isoformat()
    candidates: List[str | None] = [
        _path_mtime_iso(Path(str(row.get("summaryPath")))) if str(row.get("summaryPath") or "").strip() else None,
        _path_mtime_iso(Path(str(row.get("predictionTradesPath")))) if str(row.get("predictionTradesPath") or "").strip() else None,
        _path_mtime_iso(SIMULATION_LIVE_EQUIVALENT_LATEST),
    ]
    parsed = [dt for dt in (_monitor_parse_iso(value) for value in candidates) if dt is not None]
    if not parsed:
        return None
    return max(parsed).isoformat()


def _load_profile70_four_sim_chain_truth_map() -> Dict[str, Dict[str, Any]]:
    global _PROFILE70_FOUR_SIM_CHAIN_TRUTH_CACHE
    if _PROFILE70_FOUR_SIM_CHAIN_TRUTH_CACHE is not None:
        return dict(_PROFILE70_FOUR_SIM_CHAIN_TRUTH_CACHE)
    payload = _load_json(PROFILE70_FOUR_SIM_CHAIN_TRUTH_LATEST)
    rows = payload.get("rows") if isinstance(payload, dict) else []
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        trader = str(row.get("trader") or "").strip()
        if trader:
            out[trader] = row
    _PROFILE70_FOUR_SIM_CHAIN_TRUTH_CACHE = dict(out)
    return out


def _apply_profile70_four_sim_chain_truth(row: Dict[str, Any]) -> Dict[str, Any]:
    if str(row.get("profile") or "").strip() != "70":
        return row
    if str(row.get("mode") or "").strip().lower() != "simulation":
        return row
    trader = str(row.get("trader") or row.get("name") or "").strip()
    if not trader:
        return row
    truth = _load_profile70_four_sim_chain_truth_map().get(trader)
    if not truth:
        return row
    merged = dict(row)
    conclusion = str(truth.get("conclusion") or "").strip()
    merged["staleRawLedger"] = conclusion == "stale_raw_ledger"
    merged["summaryFreshButRawStale"] = bool(truth.get("summaryFreshButRawStale"))
    merged["runnerManaged"] = bool(truth.get("runnerManaged"))
    merged["simulationSource"] = truth.get("simulationSource")
    merged["predictionSourceStatus"] = truth.get("predictionSourceStatus")
    merged["nonRunnableReason"] = truth.get("nonRunnableReason")
    merged["lastDecisionStatus"] = truth.get("lastDecisionStatus")
    merged["lastDecisionReason"] = truth.get("lastDecisionReason")
    if truth.get("lastDecisionAt"):
        merged["lastDecisionAt"] = truth.get("lastDecisionAt")
    if truth.get("lastRawTradeAt"):
        merged["profile70TruthLastRawTradeAt"] = truth.get("lastRawTradeAt")
    merged["profile70FourSimTruthConclusion"] = conclusion or None
    merged["profile70FourSimTruthApplied"] = True
    return merged


def _display_settled_win_rate_cell(row: Dict[str, Any]) -> str:
    settled_sample_count = int(row.get("displaySettledSampleCount") or 0)
    if settled_sample_count <= 0:
        return "-"
    return f"{float(row.get('displaySettledWinRate') or row.get('displayWinRate') or 0.0):.2f}"


def _render_state_key(row: Dict[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("canonicalCellKey") or "").strip() or str(row.get("short") or "").strip(),
            str(row.get("role") or "").strip().lower(),
            str(row.get("mode") or "").strip().lower(),
        ]
    )


def _render_signature(row: Dict[str, Any]) -> str:
    payload = {
        "displayPnlUsd": round(float(row.get("displayPnlUsd") or 0.0), 2),
        "displayTrades": int(row.get("displayTrades") or 0),
        "displayWinRate": round(float(row.get("displayWinRate") or 0.0), 2),
        "latestDisplayedTradeAt": str(row.get("latestDisplayedTradeAt") or row.get("latest_str") or "-"),
        "promotionEligibility": str(row.get("promotionEligibility") or "").strip(),
        "displayLiveWalletPnlUsd": round(float(row.get("display_live_wallet_pnl_usd") or 0.0), 2),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _attach_monitor_freshness(rows: List[Dict[str, Any]], render_at: datetime) -> List[Dict[str, Any]]:
    rendered_at_iso = render_at.isoformat()
    for row in rows:
        source_updated_at = _row_source_report_updated_at(row)
        source_dt = _monitor_parse_iso(source_updated_at)
        freshness_lag = None
        if source_dt is not None:
            freshness_lag = max(0, int((render_at - source_dt).total_seconds()))
        key = _render_state_key(row)
        signature = _render_signature(row)
        previous = _MONITOR_PREVIOUS_RENDER_STATE.get(key) or {}
        stale_render_detected = bool(
            source_updated_at
            and previous.get("sourceReportUpdatedAt")
            and str(previous.get("sourceReportUpdatedAt")) != str(source_updated_at)
            and str(previous.get("rowSignature") or "") == signature
        )
        row["sourceReportUpdatedAt"] = source_updated_at
        row["monitorRenderAt"] = rendered_at_iso
        row["freshnessLagSeconds"] = freshness_lag
        row["staleRenderDetected"] = stale_render_detected
        row["staleRenderReason"] = "source_report_changed_but_render_signature_unchanged" if stale_render_detected else ""
        _MONITOR_PREVIOUS_RENDER_STATE[key] = {
            "sourceReportUpdatedAt": source_updated_at,
            "rowSignature": signature,
        }
    return [_apply_profile70_four_sim_chain_truth(row) for row in rows]


def _lowprice_profile_meta(profile: str) -> Dict[str, Any]:
    return LOWPRICE_PROFILE_FILES.get(profile, LOWPRICE_PROFILE_FILES["default"])


def _config_set_for_runtime_value(profile: str, value: Any, default_set: str = "main") -> str:
    raw = str(value or "").strip()
    if not raw:
        return default_set
    if any(raw.startswith(str(meta.get("logs_prefix") or "")) for meta in LOWPRICE_PROFILE_FILES.values()):
        return "lowprice"
    if "lowprice_selected" in raw:
        return "lowprice"
    if profile == "threshold_pack":
        return "lowprice"
    return default_set


def _profile_logs_prefix(profile: str, config_set: str = "main") -> str:
    if config_set == "lowprice":
        return str(_lowprice_profile_meta(profile).get("logs_prefix") or "logs_")
    if profile == "70":
        return "logs_70_"
    return "logs_"


def _profile_label_prefix(profile: str, config_set: str = "main") -> str:
    if config_set == "lowprice":
        return str(_lowprice_profile_meta(profile).get("label_prefix") or "")
    return "70_" if profile == "70" else ""


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_profile70_high_win_rows() -> Dict[str, Dict[str, Any]]:
    global _PROFILE70_HIGH_WIN_ROWS_CACHE
    if _PROFILE70_HIGH_WIN_ROWS_CACHE is not None:
        return _PROFILE70_HIGH_WIN_ROWS_CACHE
    payload = _load_json(PROFILE70_HIGH_WIN_TRUTH_LATEST)
    out: Dict[str, Dict[str, Any]] = {}
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            display_name = str(row.get("displayName") or "").strip()
            if display_name:
                out[display_name] = row
    _PROFILE70_HIGH_WIN_ROWS_CACHE = out
    return out


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _present(value: Any) -> bool:
    return value is not None and value != ""


def _first_present(*values: Any) -> Any:
    for value in values:
        if _present(value):
            return value
    return None


def _slot01_selected_display_reference() -> Dict[str, Any]:
    payload = _load_json(LIVE_SELECTED_CELLS)
    rows = payload.get("selected_cells") if isinstance(payload, dict) else payload
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            if (
                str(row.get("walletSlot") or "").strip() == "slot01"
                and str(row.get("trader") or "").strip() == "v5_exp10_bp_dyn_0300_0440_eth"
                and str(row.get("symbol") or "").strip().upper() == "ETH"
            ):
                selected_reference = _float_or_none(row.get("displayReferenceCapitalUsd"))
                if selected_reference is not None and selected_reference > 0:
                    return {
                        "displayReferenceCapitalUsd": round(selected_reference, 2),
                        "externalCashAdjustmentUsd": 0.0,
                        "source": "slot01_selected_reference_capital",
                        "capitalBaselineReason": str(row.get("capitalBaselineReason") or "").strip()
                        or "slot01_reference_capital_1600_monitor_only",
                    }
    return {
        "displayReferenceCapitalUsd": 1600.0,
        "externalCashAdjustmentUsd": 0.0,
        "source": "slot01_selected_reference_default_1600",
        "capitalBaselineReason": "slot01_reference_capital_1600_monitor_only",
    }


def _slot01_external_cash_adjustment(wallet_current: Any, strategy_pnl: Any) -> Dict[str, Any]:
    return _slot01_selected_display_reference()


def _path_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime if path.exists() else 0.0
    except Exception:
        return 0.0


def _refresh_slot01_canonical_trade_ledger_if_needed(timeout: int = 30) -> None:
    source_mtime = max(
        _path_mtime(path)
        for path in (
            SLOT01_LIVE_REPORT_SUMMARY,
            SLOT01_RAW_LIVE_TRADES,
            LIVE_ROLLOUT_EVIDENCE_LATEST,
            SLOT01_OFFICIAL_RECONCILIATION_LATEST,
            SLOT01_LIVE_WALLET_BALANCE_LATEST,
        )
    )
    ledger_mtime = _path_mtime(SLOT01_CANONICAL_TRADE_LEDGER_LATEST)
    if (
        SLOT01_CANONICAL_TRADE_LEDGER_GENERATOR.exists()
        and (
            not SLOT01_CANONICAL_TRADE_LEDGER_LATEST.exists()
            or ledger_mtime + 1.0 < source_mtime
        )
    ):
        subprocess.run(
            [sys.executable, str(SLOT01_CANONICAL_TRADE_LEDGER_GENERATOR)],
            cwd=str(SCRIPT_DIR.parent),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )


def _slot01_canonical_summary() -> Dict[str, Any]:
    payload = _load_json(SLOT01_CANONICAL_TRADE_LEDGER_LATEST)
    if not isinstance(payload, dict):
        return {}
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    if not summary:
        return {}
    return {
        "generated_at": payload.get("generated_at"),
        "status": payload.get("status"),
        **summary,
    }


_MODE_REGISTRY_CACHE: dict[str, Any] | None = None
_ACTIVE_CONTRACT_CACHE: dict[str, Any] = {}


def _load_mode_registry() -> dict[str, Any]:
    global _MODE_REGISTRY_CACHE
    if _MODE_REGISTRY_CACHE is None:
        payload = _load_json(MODE_REGISTRY_FILE)
        _MODE_REGISTRY_CACHE = payload if isinstance(payload, dict) else {}
    return _MODE_REGISTRY_CACHE


def _load_active_contract(path: Path) -> Any:
    key = str(path)
    if key not in _ACTIVE_CONTRACT_CACHE:
        _ACTIVE_CONTRACT_CACHE[key] = _load_json(path)
    return _ACTIVE_CONTRACT_CACHE[key]


def _load_guard_truth_index() -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
    global _GUARD_TRUTH_ROWS_CACHE
    if _GUARD_TRUTH_ROWS_CACHE is not None:
        return _GUARD_TRUTH_ROWS_CACHE
    payload = _load_json(GUARD_TRUTH_INVENTORY_LATEST)
    rows = payload.get("rows") if isinstance(payload, dict) else []
    out: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict):
            continue
        key = (
            str(row.get("scope") or "").strip(),
            str(row.get("profile") or "").strip(),
            str(row.get("trader") or "").strip(),
            str(row.get("symbol") or "").strip().upper(),
        )
        if all(key):
            out[key] = row
    _GUARD_TRUTH_ROWS_CACHE = out
    return out


def _scope_for_guard_row(row: Dict[str, Any]) -> str:
    return "lowprice" if str(row.get("_config_set") or "").strip().lower() == "lowprice" else "baseline"


def _source_freshness_contract(row: Dict[str, Any]) -> Tuple[str, str]:
    mode = str(row.get("mode") or "").strip().lower()
    role = str(row.get("role") or "").strip().lower()
    if mode == "simulation" and role == "threshold_simulation":
        decision_status = str(row.get("lastDecisionStatus") or "").strip()
        decision_reason = str(row.get("lastDecisionReason") or "").strip()
        if decision_status or decision_reason:
            return "forward_runtime_only", decision_reason or decision_status or "forward_runtime_only"
    guard_row = _load_guard_truth_index().get(
        (
            _scope_for_guard_row(row),
            str(row.get("profile") or "").strip(),
            str(row.get("trader") or "").strip(),
            str(row.get("symbol") or "").strip().upper(),
        )
    ) or {}
    prediction = guard_row.get("prediction") if isinstance(guard_row.get("prediction"), dict) else {}
    path_decision = str(prediction.get("path_decision") or "").strip()
    if (
        path_decision.startswith("fallback_")
        and bool(prediction.get("exists"))
        and bool(prediction.get("fresh"))
        and bool(prediction.get("payload_present"))
    ):
        if path_decision == "fallback_primary_missing":
            return "fallback_active_primary_missing", "回退源生效，主源缺失"
        return "fallback_active_primary_stale", "回退源生效，主源陈旧"
    classification = str(guard_row.get("finalClassification") or "").strip()
    reason = str(guard_row.get("finalReason") or "").strip()
    if classification == "source_stale":
        return "source_stale", reason or "prediction_file_missing_or_stale"
    if classification == "source_empty":
        return "source_empty", reason or "prediction_payload_empty"
    if classification:
        return "fresh_or_other", reason or classification
    return "unknown", "guard_truth_missing"


def _with_symbol_suffix(label: str, symbol: str) -> str:
    label = str(label or "").strip()
    symbol = str(symbol or "").strip().upper()
    if not label:
        return symbol
    if not symbol:
        return label
    suffix = f"-{symbol}"
    if label.upper().endswith(suffix):
        return label
    return f"{label}{suffix}"


def _ever_executed(row: Dict[str, Any]) -> bool:
    return bool(
        int(row.get("n_executed") or 0) > 0
        or int(row.get("n_logical_orders") or 0) > 0
        or int(row.get("summaryTrades") or 0) > 0
        or str(row.get("latestExecutedAt") or "").strip()
    )


def _activity_state(row: Dict[str, Any], row_kind: str) -> str:
    if row_kind == "live":
        return "live"
    if row_kind == "forced_zero_state":
        return "forced_zero_state"
    if row_kind == "zero_state_hidden":
        return "zero_state_hidden"
    if _ever_executed(row):
        return "runtime_backed_active"
    return "runtime_backed_zero_activity"


def _extract_disabled_symbol_map(raw: Any) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    if not isinstance(raw, dict):
        return out
    payload = raw.get("disabledSymbolsByTrader")
    if not isinstance(payload, dict):
        return out
    for trader, symbols in payload.items():
        if not isinstance(symbols, list):
            continue
        normalized = {
            str(item).strip().upper()
            for item in symbols
            if str(item).strip()
        }
        if normalized:
            out[str(trader).strip()] = normalized
    return out


def _extract_enabled_symbol_map(raw: Any) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {}
    if not isinstance(raw, dict):
        return out
    payload = raw.get("enabledSymbolsByTrader")
    if not isinstance(payload, dict):
        return out
    for trader, symbols in payload.items():
        if not isinstance(symbols, list):
            continue
        normalized = {
            str(item).strip().upper()
            for item in symbols
            if str(item).strip()
        }
        if normalized:
            out[str(trader).strip()] = normalized
    return out


def _symbol_enabled_for_active_path(active_path: Path, trader: str, symbol: str) -> bool:
    payload = _load_active_contract(active_path)
    active_names = _extract_active_names(payload)
    normalized_trader = str(trader or "").strip()
    normalized_symbol = str(symbol or "").strip().upper()
    if active_names and normalized_trader not in active_names:
        return False
    enabled_map = _extract_enabled_symbol_map(payload)
    if normalized_trader in enabled_map:
        return normalized_symbol in enabled_map[normalized_trader]
    disabled_map = _extract_disabled_symbol_map(payload)
    if normalized_trader in disabled_map and normalized_symbol in disabled_map[normalized_trader]:
        return False
    return True if (not active_names or normalized_trader in active_names) else False


def _selected_live_target_set() -> Set[Tuple[str, str, str]]:
    return set(load_selected_live_targets().keys())


def _registry_mode_for_cell(profile: str, trader: str, symbol: str) -> str:
    payload = _load_mode_registry()
    profiles = payload.get("profiles") if isinstance(payload.get("profiles"), dict) else {}
    profile_rows = profiles.get(profile) if isinstance(profiles.get(profile), dict) else {}
    entries = profile_rows.get("entries") if isinstance(profile_rows.get("entries"), list) else []
    target_symbol = str(symbol or "").strip().upper()
    target_trader = str(trader or "").strip()
    for row in entries:
        if not isinstance(row, dict):
            continue
        if str(row.get("traderName") or "").strip() != target_trader:
            continue
        if str(row.get("symbol") or "").strip().upper() != target_symbol:
            continue
        return str(row.get("mode") or "simulation").strip().lower() or "simulation"
    return "simulation"


def _load_scope_cells(scope: str) -> Set[Tuple[str, str, str]]:
    if scope == "all" or not CORE_UNIVERSE_LATEST.exists():
        if scope not in {"watchlist", "alerts", "regime_alerts"}:
            return set()
    payload = _load_json(CORE_UNIVERSE_LATEST)
    if not isinstance(payload, dict):
        payload = {}
    if scope == "watchlist":
        watch_payload = _load_json(ROTATION_WATCHLIST_LATEST)
        rows = []
        if isinstance(watch_payload, dict):
            top = watch_payload.get("top_candidates_by_profile") or {}
            if isinstance(top, dict):
                for vals in top.values():
                    if isinstance(vals, list):
                        rows.extend(vals)
        if not rows:
            watch_payload = _load_json(WIDE_OBSERVATION_LATEST)
            rows = (watch_payload or {}).get("shadow_watchlist") if isinstance(watch_payload, dict) else []
    elif scope == "alerts":
        alert_payload = _load_json(RECENT_WINDOW_WATCH_LATEST)
        rows = (alert_payload or {}).get("alert_cells") if isinstance(alert_payload, dict) else []
    elif scope == "regime_alerts":
        rows = []
        alert_payload = _load_json(RECENT_WINDOW_WATCH_LATEST)
        if isinstance(alert_payload, dict):
            vals = alert_payload.get("alert_cells")
            if isinstance(vals, list):
                rows.extend(vals)
        regime_payload = _load_json(EXTREME_REGIME_STATE_LATEST)
        regime_rows = regime_payload.get("rows") if isinstance(regime_payload, dict) and isinstance(regime_payload.get("rows"), list) else []
        active_keys: Set[Tuple[str, str]] = set()
        for item in regime_rows:
            if not isinstance(item, dict):
                continue
            if not bool(item.get("active")):
                continue
            regime_id = str(item.get("regimeId") or "")
            if not regime_id.startswith("extreme_"):
                continue
            profile = str(item.get("profile") or "")
            symbol = str(item.get("symbol") or "").upper()
            active_keys.add((profile, symbol))
        core_rows = payload.get("core_cells") if isinstance(payload.get("core_cells"), list) else []
        for row in core_rows:
            if not isinstance(row, dict):
                continue
            key = (str(row.get("profile") or ""), str(row.get("symbol") or "").upper())
            if key in active_keys:
                rows.append(row)
    else:
        key = "core_cells" if scope == "core" else "shadow_cells"
        rows = payload.get(key)
    out: Set[Tuple[str, str, str]] = set()
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            out.add(
                (
                    str(row.get("profile") or ""),
                    str(row.get("trader") or ""),
                    str(row.get("symbol") or "").upper(),
                )
            )
    if scope == "core":
        official = _load_json(EXP13_OFFICIAL_SIM_CANDIDATE_LATEST)
        official_summary = official.get("summary") if isinstance(official, dict) and isinstance(official.get("summary"), dict) else {}
        candidate_name = str(official_summary.get("candidate_trader_name") or "").strip()
        if candidate_name and (
            bool(official_summary.get("qualified_for_official_simulation"))
            or bool(official_summary.get("official_simulation_live"))
            or bool(official_summary.get("active_present"))
        ):
            out.add(("default", candidate_name, "ETH"))
        route_f = _load_json(ROUTE_F_EXP10_OFFICIAL_SIM_CANDIDATE_LATEST)
        route_f_summary = route_f.get("summary") if isinstance(route_f, dict) and isinstance(route_f.get("summary"), dict) else {}
        route_f_name = str(route_f_summary.get("candidate_trader_name") or "").strip()
        if route_f_name and (
            bool(route_f_summary.get("qualified_for_official_simulation"))
            or bool(route_f_summary.get("official_simulation_live"))
            or bool(route_f_summary.get("config_present"))
        ):
            out.add(("default", route_f_name, "ETH"))
        threshold_cfg = _load_json(SCRIPT_DIR / "trader_configs_70.json")
        threshold_active = _extract_active_names(_load_active_contract(SCRIPT_DIR / "active_traders_70.json"))
        if isinstance(threshold_cfg, list):
            for row in threshold_cfg:
                if not isinstance(row, dict):
                    continue
                if not bool(row.get("thresholdSimulation")):
                    continue
                name = str(row.get("name") or "").strip()
                if not name or name not in threshold_active:
                    continue
                for symbol in _cfg_allowed_symbols(row):
                    out.add(("70", name, symbol))
    return out


def _extract_active_names(raw: Any) -> Set[str]:
    if isinstance(raw, list):
        return {str(x).strip() for x in raw if isinstance(x, str) and str(x).strip()}
    if isinstance(raw, dict):
        out: Set[str] = set()
        for key in ("traderNames", "active_traders"):
            vals = raw.get(key)
            if isinstance(vals, list):
                out.update(str(x).strip() for x in vals if isinstance(x, str) and str(x).strip())
        return out
    return set()


def _extract_active_groups(raw: Any) -> Set[str]:
    if isinstance(raw, dict):
        vals = raw.get("groups")
        if isinstance(vals, list):
            return {str(x).strip() for x in vals if str(x).strip()}
    return set()


def _cfg_paths(profile: str, config_set: str = "main") -> tuple[Path, Path]:
    if config_set == "lowprice":
        meta = _lowprice_profile_meta(profile)
        return meta["configs"], meta["active"]
    if profile == "70":
        return SCRIPT_DIR / "trader_configs_70.json", SCRIPT_DIR / "active_traders_70.json"
    return SCRIPT_DIR / "trader_configs.json", SCRIPT_DIR / "active_traders.json"


def _load_monitor_only_rows(profile: str, monitor_set: str = "main", active_only: bool = False) -> List[Dict[str, Any]]:
    raw_rows: List[Dict[str, Any]] = []
    for manifest_path in MONITOR_ONLY_TRADER_FILE_SETS.get(monitor_set, []):
        active_names: Set[str] = set()
        if active_only:
            active_path = MONITOR_ONLY_ACTIVE_FILE_MAP.get(str(manifest_path))
            if active_path is not None:
                active_names = _extract_active_names(_load_active_contract(active_path))
        payload = _load_json(manifest_path)
        if isinstance(payload, list):
            source_rows = [row for row in payload if isinstance(row, dict)]
        elif isinstance(payload, dict):
            traders = payload.get("traders")
            if isinstance(traders, list):
                source_rows = [row for row in traders if isinstance(row, dict)]
            else:
                source_rows = []
        else:
            source_rows = []
        if active_only and active_names:
            source_rows = [
                row for row in source_rows
                if str(row.get("name") or "").strip() in active_names
            ]
        raw_rows.extend(source_rows)

    out: List[Dict[str, Any]] = []
    for row in raw_rows:
        row_profile = str(row.get("profile") or "default").strip().lower()
        if profile != "all" and row_profile not in {"all", profile}:
            continue
        trader_name = str(row.get("name") or "").strip()
        symbol = str(row.get("symbol") or "").strip().upper()
        row_config_set = str(row.get("_config_set") or monitor_set).strip().lower()
        if not trader_name:
            continue
        if not str(row.get("logsDir") or "").strip():
            continue
        if row_config_set == "lowprice" and is_retired_compare_only_trader(
            "default" if row_profile in {"", "all"} else row_profile,
            trader_name,
            symbol,
        ):
            continue
        copied = dict(row)
        copied["profile"] = "default" if row_profile in {"", "all"} else row_profile
        copied["role"] = "compare_only"
        out.append(copied)
    return out


def _cfg_allowed_symbols_for_active_row(cfg_row: Dict[str, Any], profile: str, config_set: str = "main") -> List[str]:
    _, active_path = _cfg_paths(profile, config_set=config_set)
    trader = str(cfg_row.get("name") or "").strip()
    return [
        symbol
        for symbol in _cfg_allowed_symbols(cfg_row)
        if _symbol_enabled_for_active_path(active_path, trader, symbol)
    ]


def _mode_of(log_dir: str) -> str:
    m = MODE_SUFFIX_RE.search(log_dir)
    return m.group(1).lower() if m else "simulation"


def _base_of(log_dir: str) -> str:
    return MODE_SUFFIX_RE.sub("", log_dir)


def _mode_match(mode: str, mode_filter: str) -> bool:
    return mode_filter == "both" or mode == mode_filter


def _is_runtime_split_dir(log_dir: str) -> bool:
    return bool(RUNTIME_SPLIT_SUFFIX_RE.search(log_dir))


def _is_sample_runtime_dir(log_dir: str) -> bool:
    return bool(RUNTIME_SAMPLE_SUFFIX_RE.search(log_dir))


def _runtime_split_symbol_hint(log_dir: str) -> str:
    m = RUNTIME_SPLIT_ASSET_RE.search(log_dir)
    if not m:
        return ""
    asset = str(m.group(1) or "").strip().upper()
    if not asset:
        return ""
    return asset.split("_", 1)[0]


def _normalize_lowprice_finalist_key(raw: Any) -> str:
    value = str(raw or "").strip()
    if value.startswith("forced_fixed_"):
        return value[len("forced_") :]
    if value.startswith("forced_dynamic_"):
        return "dynamic_" + value[len("forced_dynamic_") :]
    if value.startswith("fixed_dynamic_"):
        return "dynamic_" + value[len("fixed_dynamic_") :]
    return value


def _is_forced_lowprice_row(row: Dict[str, Any]) -> bool:
    if bool(row.get("forcedLowpriceMaterialization")):
        return True
    family_rule_version = str(row.get("familyRuleVersion") or "").strip()
    return family_rule_version.startswith("forced::lowprice_manual_")


def _is_visible_zero_state_row(row: Dict[str, Any]) -> bool:
    if _is_forced_lowprice_row(row):
        return True
    if bool(row.get("thresholdSimulation")):
        return True
    role = str(row.get("role") or "").strip().lower()
    return role == "threshold_simulation"


def _live_truth_contract_rows() -> Dict[Tuple[str, str, str, str], Dict[str, Any]]:
    global _LIVE_TRUTH_ROWS_CACHE
    if _LIVE_TRUTH_ROWS_CACHE is not None:
        return dict(_LIVE_TRUTH_ROWS_CACHE)
    payload = _load_json(LIVE_TRUTH_CONTRACT_LATEST)
    rows = payload.get("rows") if isinstance(payload, dict) and isinstance(payload.get("rows"), list) else []
    out: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        target = row.get("target_cell") if isinstance(row.get("target_cell"), dict) else {}
        key = (
            str(target.get("profile") or row.get("profile") or "").strip(),
            str(target.get("trader") or row.get("trader") or "").strip(),
            str(target.get("symbol") or row.get("symbol") or "").strip().upper(),
            str(target.get("walletSlot") or row.get("walletSlot") or "").strip(),
        )
        if key[0] and key[1] and key[2]:
            out[key] = dict(row)
    _LIVE_TRUTH_ROWS_CACHE = dict(out)
    return out


def _live_claim_status_map() -> Dict[str, Dict[str, Any]]:
    global _LIVE_CLAIM_STATUS_MAP_CACHE
    if _LIVE_CLAIM_STATUS_MAP_CACHE is not None:
        return dict(_LIVE_CLAIM_STATUS_MAP_CACHE)
    payload = _load_json(LIVE_CLAIM_STATUS_ACTIVE_LATEST)
    rows = payload.get("slots") if isinstance(payload, dict) and isinstance(payload.get("slots"), list) else []
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        slot = str(row.get("slot") or "").strip().lower()
        if slot:
            out[slot] = dict(row)
    _LIVE_CLAIM_STATUS_MAP_CACHE = dict(out)
    return out


def _dual_eth_rule_parity_map() -> Dict[str, Dict[str, Any]]:
    global _DUAL_ETH_RULE_PARITY_MAP_CACHE
    if _DUAL_ETH_RULE_PARITY_MAP_CACHE is not None:
        return dict(_DUAL_ETH_RULE_PARITY_MAP_CACHE)
    payload = _load_json(DUAL_ETH_LIVE_RULE_PARITY_LATEST)
    rows = payload.get("rows") if isinstance(payload, dict) and isinstance(payload.get("rows"), list) else []
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        trader = str(row.get("trader") or "").strip()
        if trader:
            out[trader] = dict(row)
    _DUAL_ETH_RULE_PARITY_MAP_CACHE = dict(out)
    return out


def _dual_eth_current_truth_payload_is_fresh(payload: Dict[str, Any]) -> bool:
    rows = payload.get("rows") if isinstance(payload, dict) and isinstance(payload.get("rows"), list) else []
    if not rows:
        return False
    row_map: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        trader = str(row.get("trader") or "").strip()
        if trader:
            row_map[trader] = row
    for trader in DUAL_ETH_LIVE_TARGETS:
        row = row_map.get(trader)
        if not row:
            return False
        if row.get("consumerAllowed") is False:
            return False
        report_path_raw = str(row.get("sourceReportPath") or row.get("reportSummaryPath") or "").strip()
        if not report_path_raw:
            return False
        report_updated_at = _path_mtime_iso(Path(report_path_raw))
        stored_updated_at = str(row.get("sourceReportUpdatedAt") or row.get("reportSummaryUpdatedAt") or "").strip() or None
        report_dt = _monitor_parse_iso(report_updated_at)
        stored_dt = _monitor_parse_iso(stored_updated_at)
        if report_dt is None or stored_dt is None:
            return False
        if report_dt > stored_dt:
            return False
    return True


def _maybe_refresh_dual_eth_live_truth_reports() -> None:
    global _DUAL_ETH_TRUTH_REPORT_REFRESH_ATTEMPTED
    global _DUAL_ETH_CURRENT_TRUTH_MAP_CACHE
    global _DUAL_ETH_SETTLED_SNAPSHOT_MAP_CACHE
    if _DUAL_ETH_TRUTH_REPORT_REFRESH_ATTEMPTED:
        return
    payload = _load_json(DUAL_ETH_LIVE_CURRENT_TRUTH_SNAPSHOT_LATEST)
    if _dual_eth_current_truth_payload_is_fresh(payload if isinstance(payload, dict) else {}):
        return
    _DUAL_ETH_TRUTH_REPORT_REFRESH_ATTEMPTED = True
    refresh_cmds = [
        ["python3", str(OPS_DIR / "generate_dual_eth_live_current_truth_snapshot_latest.py")],
        ["python3", str(OPS_DIR / "generate_dual_eth_live_settled_snapshot_latest.py")],
        ["python3", str(OPS_DIR / "generate_dual_eth_live_performance_attribution_latest.py")],
        ["python3", str(OPS_DIR / "generate_slot01_slot02_real_trade_chain_deep_audit_latest.py")],
        ["python3", str(OPS_DIR / "claim_status_report.py"), "--slot", "active"],
    ]
    for cmd in refresh_cmds:
        try:
            subprocess.run(
                cmd,
                cwd=str(SCRIPT_DIR.parent),
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=20,
            )
        except Exception:
            continue
    _DUAL_ETH_CURRENT_TRUTH_MAP_CACHE = None
    _DUAL_ETH_SETTLED_SNAPSHOT_MAP_CACHE = None


def _dual_eth_live_current_truth_map() -> Dict[str, Dict[str, Any]]:
    global _DUAL_ETH_CURRENT_TRUTH_MAP_CACHE
    if _DUAL_ETH_CURRENT_TRUTH_MAP_CACHE is not None:
        return dict(_DUAL_ETH_CURRENT_TRUTH_MAP_CACHE)
    _maybe_refresh_dual_eth_live_truth_reports()
    payload = _load_json(DUAL_ETH_LIVE_CURRENT_TRUTH_SNAPSHOT_LATEST)
    if not _dual_eth_current_truth_payload_is_fresh(payload if isinstance(payload, dict) else {}):
        payload = build_current_truth_payload()
    rows = payload.get("rows") if isinstance(payload, dict) and isinstance(payload.get("rows"), list) else []
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        trader = str(row.get("trader") or "").strip()
        if trader:
            out[trader] = dict(row)
    _DUAL_ETH_CURRENT_TRUTH_MAP_CACHE = dict(out)
    return out


def _dual_eth_live_settled_snapshot_map() -> Dict[str, Dict[str, Any]]:
    global _DUAL_ETH_SETTLED_SNAPSHOT_MAP_CACHE
    if _DUAL_ETH_SETTLED_SNAPSHOT_MAP_CACHE is not None:
        return dict(_DUAL_ETH_SETTLED_SNAPSHOT_MAP_CACHE)
    _maybe_refresh_dual_eth_live_truth_reports()
    payload = _load_json(DUAL_ETH_LIVE_SETTLED_SNAPSHOT_LATEST)
    rows = payload.get("rows") if isinstance(payload, dict) and isinstance(payload.get("rows"), list) else []
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        trader = str(row.get("trader") or "").strip()
        if trader:
            out[trader] = dict(row)
    _DUAL_ETH_SETTLED_SNAPSHOT_MAP_CACHE = dict(out)
    return out


def _dual_eth_timing_parity_map() -> Dict[str, Dict[str, Any]]:
    global _DUAL_ETH_TIMING_PARITY_MAP_CACHE
    if _DUAL_ETH_TIMING_PARITY_MAP_CACHE is not None:
        return dict(_DUAL_ETH_TIMING_PARITY_MAP_CACHE)
    payload = _load_json(DUAL_ETH_LIVE_TIMING_PARITY_LATEST)
    rows = payload.get("rows") if isinstance(payload, dict) and isinstance(payload.get("rows"), list) else []
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        trader = str(row.get("trader") or "").strip()
        if trader:
            out[trader] = dict(row)
    _DUAL_ETH_TIMING_PARITY_MAP_CACHE = dict(out)
    return out


def _dual_eth_direction_cooldown_map() -> Dict[str, Dict[str, Any]]:
    global _DUAL_ETH_DIRECTION_COOLDOWN_MAP_CACHE
    if _DUAL_ETH_DIRECTION_COOLDOWN_MAP_CACHE is not None:
        return dict(_DUAL_ETH_DIRECTION_COOLDOWN_MAP_CACHE)
    payload = _load_json(DUAL_ETH_LIVE_DIRECTION_COOLDOWN_TRUTH_LATEST)
    rows = payload.get("rows") if isinstance(payload, dict) and isinstance(payload.get("rows"), list) else []
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        trader = str(row.get("trader") or "").strip()
        if trader:
            out[trader] = dict(row)
    _DUAL_ETH_DIRECTION_COOLDOWN_MAP_CACHE = dict(out)
    return out


def _dual_eth_sizing_map() -> Dict[str, Dict[str, Any]]:
    global _DUAL_ETH_SIZING_MAP_CACHE
    if _DUAL_ETH_SIZING_MAP_CACHE is not None:
        return dict(_DUAL_ETH_SIZING_MAP_CACHE)
    payload = _load_json(DUAL_ETH_LIVE_SIZING_BACKTEST_LATEST)
    slots = payload.get("summary", {}).get("slots") if isinstance(payload, dict) else {}
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(slots, dict):
        for slot_payload in slots.values():
            if not isinstance(slot_payload, dict):
                continue
            trader = str(slot_payload.get("trader") or "").strip()
            if trader:
                out[trader] = dict(slot_payload)
    _DUAL_ETH_SIZING_MAP_CACHE = dict(out)
    return out


def _friendly_live_label(trader: str, fallback: str) -> str:
    return _FRIENDLY_LIVE_LABELS.get(str(trader or "").strip(), fallback)


def _disabled_live_selection_keys() -> set[Tuple[str, str, str]]:
    payload = _load_json(LIVE_SELECTED_CELLS)
    rows = payload.get("selected_cells") if isinstance(payload, dict) else payload
    out: set[Tuple[str, str, str]] = set()
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict) or row.get("enabled", True) is not False:
            continue
        profile = "70" if str(row.get("profile") or "").strip() == "70" else "default"
        trader = str(row.get("trader") or row.get("traderName") or "").strip()
        symbol = str(row.get("symbol") or "").strip().upper()
        if trader and symbol:
            out.add((profile, trader, symbol))
    return out


def _is_disabled_live_selection_display_row(row: Dict[str, Any], disabled_keys: set[Tuple[str, str, str]]) -> bool:
    if not disabled_keys:
        return False
    if str(row.get("role") or "").strip().lower() == "live_rollout":
        return False
    if str(row.get("mode") or "").strip().lower() != "live":
        return False
    profile = "70" if str(row.get("profile") or "").strip() == "70" else "default"
    trader = str(row.get("trader") or "").strip()
    symbol = str(row.get("symbol") or "").strip().upper()
    return bool(trader and symbol and (profile, trader, symbol) in disabled_keys)


def _simulation_live_equivalent_rows() -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    global _SIMULATION_LIVE_EQUIV_CACHE
    if _SIMULATION_LIVE_EQUIV_CACHE is not None:
        return dict(_SIMULATION_LIVE_EQUIV_CACHE)
    payload = _load_json(SIMULATION_LIVE_EQUIVALENT_LATEST)
    rows = payload.get("rows") if isinstance(payload, dict) and isinstance(payload.get("rows"), list) else []
    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = (
            str(row.get("scope") or "").strip(),
            str(row.get("profile") or "").strip(),
            str(row.get("label") or "").strip(),
        )
        if key[0] and key[1] and key[2]:
            out[key] = dict(row)
    _SIMULATION_LIVE_EQUIV_CACHE = dict(out)
    return out


def _row_scope_for_monitor(row: Dict[str, Any]) -> str:
    config_set = str(row.get("_config_set") or "").strip().lower()
    if config_set == "lowprice":
        return "lowprice"
    runtime_dir = str(row.get("runtime_dir") or row.get("base_dir") or "").strip()
    if "lowprice_selected" in runtime_dir:
        return "lowprice"
    return "all"


def _promotion_display(reason: str, status: str, role: str) -> str:
    if role == "live_rollout":
        return "真实交易"
    if status == "ready":
        return "可切真实交易候选"
    mapping = {
        "fill_realism_fail": "成交现实性不足",
        "queue_realism_fail": "排队现实性不足",
        "worst_case_pnl_fail": "最坏成交下收益不成立",
        "prediction_cutover_incomplete": "预测接管未完成",
        "hyperparam_coverage_missing": "超参覆盖缺失",
        "runtime_values_not_latest": "运行参数未接最新批准值",
        "risk_contract_fail": "风控合同未满足",
        "no_trade_evidence": "无成交样本",
    }
    return mapping.get(reason or "", reason or "不可切真实交易")


def _attach_live_equivalent_contract(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    realism_rows = _simulation_live_equivalent_rows()
    high_win_rows = _load_profile70_high_win_rows()
    current_truth_rows = _dual_eth_live_current_truth_map()
    settled_snapshot_rows = _dual_eth_live_settled_snapshot_map()
    out: List[Dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        truth_source = str(copied.get("truthSource") or "").strip()
        role = str(copied.get("role") or "").strip().lower()
        if role in {"mapped_simulation", "reinit_simulation"} and str(copied.get("_mapped_from_slot") or "") == "slot01":
            display_trades = int(copied.get("n_logical_orders") or copied.get("n_markets") or 0)
            display_win_rate = float(copied.get("win_rate") or 0.0)
            explicit_pnl = _float_or_none(copied.get("displayPnlUsd"))
            if explicit_pnl is None:
                explicit_pnl = _float_or_none(copied.get("total_pnl"))
            if explicit_pnl is None:
                explicit_pnl = 0.0
            strategy_pnl = _float_or_none(copied.get("strategyPnlUsd"))
            if strategy_pnl is None:
                strategy_pnl = explicit_pnl
            copied.update(
                {
                    "rawSimTrades": display_trades,
                    "rawSimWinRate": round(display_win_rate, 2),
                    "rawSimPnlUsd": round(strategy_pnl, 2),
                    "realisticSimTrades": display_trades,
                    "realisticSimWinRate": round(display_win_rate, 2),
                    "realisticSimPnlUsd": round(explicit_pnl, 2),
                    "realisticPnlBasis": "slot01_mapped_report_explicit_pnl",
                    "liveEquivalentTrades": display_trades,
                    "liveEquivalentWinRate": round(display_win_rate, 2),
                    "liveEquivalentPnlUsd": round(explicit_pnl, 2),
                    "conservativeStressPnlUsd": round(explicit_pnl, 2),
                    "displayTrades": display_trades,
                    "displayWinRate": round(display_win_rate, 2),
                    "displayAllHitRate": round(display_win_rate, 2),
                    "displaySettledWinRate": round(display_win_rate, 2),
                    "displayNoFillRate": 0.0,
                    "displaySettledConversionRate": 100.0,
                    "allLogicTradeCount": display_trades,
                    "displaySettledSampleCount": display_trades,
                    "displayTrustRating": "映射模拟",
                    "displayTrustReason": "映射模拟不参与真钱评级",
                    "displayPnlUsd": round(explicit_pnl, 2),
                    "total_pnl": round(explicit_pnl, 2),
                    "displayPnlBasis": "slot01_mapped_report_explicit_pnl",
                    "currentCapitalDisplayOnly": copied.get("currentCapital"),
                    "realismGateStatus": "mapped_slot01_bypass",
                    "realismFailReasons": [],
                    "promotionEligibility": "blocked",
                    "promotionEligibilityReason": "mapped_simulation_not_live_candidate",
                    "promotionEligibilityDisplay": "映射模拟，不作为真钱候选",
                    "rawShortDisplay": f"${strategy_pnl:+.2f}",
                }
            )
            out.append(copied)
            continue
        if role == "live_rollout":
            trader = str(copied.get("trader") or "").strip()
            current_truth = current_truth_rows.get(trader, {})
            settled_snapshot = settled_snapshot_rows.get(trader, {})
            # Live table truth must follow the freshest live rollout/truth row.
            # The settled snapshot is only a fallback report; it can lag by a
            # few refreshes and must not overwrite current live win/loss/pnl.
            settled_trades = int(
                current_truth.get("settledTradeCount")
                or copied.get("settledTradeCount")
                or copied.get("n_completed_markets")
                or settled_snapshot.get("settledTradeCount")
                or (int(copied.get("wins") or 0) + int(copied.get("losses") or 0))
                or 0
            )
            settled_wins = int(
                current_truth.get("wins")
                if current_truth.get("wins") is not None
                else copied.get("wins")
                if copied.get("wins") is not None
                else settled_snapshot.get("wins")
                or 0
            )
            settled_losses = int(
                current_truth.get("losses")
                if current_truth.get("losses") is not None
                else copied.get("losses")
                if copied.get("losses") is not None
                else settled_snapshot.get("losses")
                or 0
            )
            pending_count = int(
                current_truth.get("pending")
                if current_truth.get("pending") is not None
                else copied.get("pending")
                if copied.get("pending") is not None
                else settled_snapshot.get("pending")
                or 0
            )
            total_logical = int(
                current_truth.get("totalLogicalTrades")
                or copied.get("n_logical_orders")
                or copied.get("n_markets")
                or (settled_trades + pending_count)
                or 0
            )
            cleared_no_fill = int(copied.get("clearedNoFillCount") or 0)
            display_trades = settled_trades
            display_win_rate = (
                round(settled_wins / (settled_wins + settled_losses) * 100.0, 2)
                if (settled_wins + settled_losses) > 0
                else float(current_truth.get("settledWinRate") or copied.get("settledWinRate") or copied.get("win_rate") or settled_snapshot.get("settledWinRate") or 0.0)
            )
            display_pnl = float(
                current_truth.get("displayLiveWalletPnlUsd")
                if current_truth.get("displayLiveWalletPnlUsd") is not None
                else copied.get("display_live_wallet_pnl_usd")
                if copied.get("display_live_wallet_pnl_usd") is not None
                else copied.get("total_pnl")
                if copied.get("total_pnl") is not None
                else settled_snapshot.get("displayLiveWalletPnlUsd")
                if settled_snapshot.get("displayLiveWalletPnlUsd") is not None
                else settled_snapshot.get("displayPnlUsd")
                if settled_snapshot.get("displayPnlUsd") is not None
                else 0.0
            )
            strategy_pnl = float(
                current_truth.get("strategyLogicalPnlUsd")
                if current_truth.get("strategyLogicalPnlUsd") is not None
                else copied.get("strategy_logical_pnl_usd")
                if copied.get("strategy_logical_pnl_usd") is not None
                else settled_snapshot.get("strategyLogicalPnlUsd")
                if settled_snapshot.get("strategyLogicalPnlUsd") is not None
                else display_pnl
            )
            copied.update(
                {
                    "rawSimTrades": settled_trades,
                    "rawSimWinRate": round(display_win_rate, 2),
                    "rawSimPnlUsd": round(strategy_pnl, 2),
                    "liveEquivalentTrades": total_logical,
                    "liveEquivalentWinRate": round(display_win_rate, 2),
                    "liveEquivalentPnlUsd": round(display_pnl, 2),
                    "displayTrades": display_trades,
                    "displayWinRate": round(display_win_rate, 2),
                    "displayAllHitRate": round(display_win_rate, 2),
                    "displaySettledWinRate": round(display_win_rate, 2) if settled_trades > 0 else None,
                    "displayNoFillRate": round((cleared_no_fill / total_logical * 100.0) if total_logical else 0.0, 2),
                    "displaySettledConversionRate": round((settled_trades / total_logical * 100.0) if total_logical else 0.0, 2),
                    "allLogicTradeCount": total_logical,
                    "displaySettledSampleCount": settled_trades,
                    "displayTrustRating": "真实",
                    "displayTrustReason": "真实交易直接口径",
                    "displayPnlUsd": round(display_pnl, 2),
                    "displayPnlBasis": "official_api_portfolio_minus_live_baseline",
                    "strategyPnlUsd": round(strategy_pnl, 2),
                    "walletDisplayPnlUsd": round(display_pnl, 2),
                    "wins": settled_wins,
                    "losses": settled_losses,
                    "pending": pending_count,
                    # The table column is "latest executed trade", not
                    # "latest settled trade". Prefer the direct live ledger
                    # timestamp so fresh matched orders are visible before
                    # slower settled/portfolio reports catch up.
                    "latestSettledTradeDisplay": str(copied.get("latestDisplayedTradeAt") or copied.get("latest_str") or current_truth.get("latestSettledTradeAt") or settled_snapshot.get("latestSettledTradeAt") or "-"),
                    "realismGateStatus": "live",
                    "realismFailReasons": [],
                    "promotionEligibility": "live",
                    "promotionEligibilityReason": "live",
                    "promotionEligibilityDisplay": "真实交易",
                    "rawShortDisplay": f"${float(copied.get('rawSimPnlUsd') or 0.0):+.2f}",
                }
            )
            out.append(copied)
            continue
        scope = _row_scope_for_monitor(copied)
        profile = str(copied.get("profile") or "").strip()
        label = str(copied.get("short") or "").strip()
        realism = realism_rows.get((scope, profile, label), {})
        raw_trades = int(
            copied.get("settledTradeCount")
            or copied.get("n_completed_markets")
            or (int(copied.get("wins") or 0) + int(copied.get("losses") or 0))
            or 0
        )
        raw_win_rate = float(copied.get("settledWinRate") or copied.get("win_rate") or 0.0)
        raw_pnl = float(copied.get("total_pnl") or 0.0)
        realism_trades = int(realism.get("liveEquivalentTrades") or raw_trades) if realism else raw_trades
        realism_win_rate = float(realism.get("liveEquivalentWinRate") or raw_win_rate) if realism else raw_win_rate
        realism_pnl = float(realism.get("liveEquivalentPnlUsd") or raw_pnl) if realism else raw_pnl
        realistic_pnl = float(realism.get("realisticSimPnlUsd") or raw_pnl) if realism else raw_pnl
        derived_total_logical = int(
            copied.get("n_logical_orders")
            or (int(copied.get("wins") or 0) + int(copied.get("losses") or 0) + int(copied.get("pending") or 0) + int(copied.get("clearedNoFillCount") or 0))
            or raw_trades
        )
        derived_all_hit_rate = (float(copied.get("wins") or 0) / derived_total_logical * 100.0) if derived_total_logical else 0.0
        derived_no_fill_rate = (float(copied.get("clearedNoFillCount") or 0) / derived_total_logical * 100.0) if derived_total_logical else 0.0
        high_win = high_win_rows.get(label, {})
        conservative_stress_pnl = float(
            realism.get("conservativeStressPnlUsd")
            if realism.get("conservativeStressPnlUsd") is not None
            else realism.get("liveEquivalentPnlUsd") if realism else raw_pnl
        )
        promotion_status = str(realism.get("promotionEligibility") or "").strip() or "blocked"
        promotion_reason = str(realism.get("promotionEligibilityReason") or "").strip()
        # For current simulation rows, the main monitor must reflect the raw
        # simulation ledger truth for logical orders / win rate / pnl.
        # Realism rows stay available as metadata, but they must not override
        # the primary display fields or we regress back to stale pending/pnl.
        use_raw_display_truth = truth_source == "simulation_raw_ledger"
        copied.update(
            {
                "rawSimTrades": raw_trades,
                "rawSimWinRate": round(raw_win_rate, 2),
                "rawSimPnlUsd": round(raw_pnl, 2),
                "realisticSimTrades": int(realism.get("realisticSimTrades") or raw_trades) if realism else raw_trades,
                "realisticSimWinRate": round(float(realism.get("realisticSimWinRate") or raw_win_rate), 2) if realism else round(raw_win_rate, 2),
                "realisticSimPnlUsd": round(realistic_pnl, 2),
                "realisticPnlBasis": str(realism.get("realisticPnlBasis") or "raw_simulation_fallback") if realism else "raw_simulation_fallback",
                "liveEquivalentTrades": realism_trades,
                "liveEquivalentWinRate": round(realism_win_rate, 2),
                "liveEquivalentPnlUsd": round(realism_pnl, 2),
                "conservativeStressPnlUsd": round(conservative_stress_pnl, 2),
                "displayTrades": raw_trades if use_raw_display_truth else (int(realism.get("realisticSimTrades") or raw_trades) if realism else raw_trades),
                "displayWinRate": round(raw_win_rate, 2) if use_raw_display_truth else (round(float(realism.get("realisticSimWinRate") or raw_win_rate), 2) if realism else round(raw_win_rate, 2)),
                "displayAllHitRate": round(float(high_win.get("allLogicHitRate") or derived_all_hit_rate), 2),
                "displaySettledWinRate": round(raw_win_rate, 2),
                "displayNoFillRate": round(float(high_win.get("noFillRate") or derived_no_fill_rate), 2),
                "displaySettledConversionRate": round(float(high_win.get("settledConversionRate") or ((raw_trades / derived_total_logical * 100.0) if derived_total_logical else 0.0)), 2),
                "allLogicTradeCount": int(high_win.get("totalLogicalTradeCount") or derived_total_logical),
                "displaySettledSampleCount": int(high_win.get("settledTradeCount") or raw_trades),
                "displayTrustRating": str(high_win.get("trustRatingShort") or high_win.get("trustRating") or "模拟"),
                "displayTrustReason": str(high_win.get("trustReason") or ""),
                "displayPnlUsd": round(raw_pnl, 2) if use_raw_display_truth else round(realistic_pnl, 2),
                "displayPnlBasis": "raw_simulation_ledger" if use_raw_display_truth else ("realistic_simulated_pnl" if realism else "raw_simulation_fallback"),
                "realismGateStatus": str(realism.get("realismGateStatus") or ""),
                "realismFailReasons": list(realism.get("realismFailReasons") or []),
                "promotionEligibility": promotion_status,
                "promotionEligibilityReason": promotion_reason,
                "promotionEligibilityDisplay": _promotion_display(promotion_reason, promotion_status, role),
                "rawShortDisplay": f"${raw_pnl:+.2f}",
                "replacedByLive": bool(copied.get("replacedByLive") or realism.get("replacedByLive")),
            }
        )
        out.append(copied)
    return out


def _row_sort_key(row: Dict[str, Any]) -> Tuple[int, float, float, float, str]:
    if row.get("_pinned_live_rollout"):
        return (0, 0.0, 0.0, 0.0, str(row.get("short") or ""))
    return (
        1,
        -float(row.get("displayPnlUsd") or row.get("total_pnl") or 0.0),
        -float(row.get("displayTrades") or row.get("n_logical_orders") or row.get("n_markets") or 0.0),
        -float(row.get("displayWinRate") or row.get("win_rate") or 0.0),
        str(row.get("short") or ""),
    )


def _independent_live_label(profile: str, trader: str, symbol: str) -> str:
    special = _FRIENDLY_LIVE_LABELS.get(str(trader or "").strip())
    if special:
        return special
    prefix = _profile_label_prefix(profile, config_set="main")
    base = str(trader or "").strip().replace("v5_", "").replace("ensemble_", "ens_")
    symbol_text = str(symbol or "").strip().upper()
    label = f"live_{base}"
    if prefix:
        label = f"{prefix}{label}"
    return f"{label}-{symbol_text}" if symbol_text else label


def _discover_base_dirs(profile: str, active_only: bool, config_set: str = "main") -> List[str]:
    cfg_path, active_path = _cfg_paths(profile, config_set=config_set)
    cfg_raw = _load_json(cfg_path)
    if not isinstance(cfg_raw, list):
        return []

    all_dirs = []
    for row in cfg_raw:
        if not isinstance(row, dict):
            continue
        logs_dir = str(row.get("logsDir", "")).strip()
        if logs_dir:
            all_dirs.append(logs_dir)

    if not active_only:
        return sorted(set(all_dirs))

    active_raw = _load_json(active_path)
    active_names = _extract_active_names(active_raw)
    active_groups = _extract_active_groups(active_raw)
    use_group_fallback = (len(active_names) == 0 and len(active_groups) > 0)

    selected: Set[str] = set()
    for row in cfg_raw:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        group = str(row.get("group", "")).strip()
        logs_dir = str(row.get("logsDir", "")).strip()
        if not logs_dir:
            continue
        if name in active_names:
            selected.add(logs_dir)
            continue
        if group in ALWAYS_INCLUDE_GROUPS:
            selected.add(logs_dir)
            continue
        if use_group_fallback and group and group in active_groups:
            selected.add(logs_dir)

    return sorted(selected if selected else set(all_dirs))


def _discover_config_rows(profile: str, active_only: bool, config_set: str = "main") -> List[Dict[str, Any]]:
    cfg_path, active_path = _cfg_paths(profile, config_set=config_set)
    cfg_raw = _load_json(cfg_path)
    if not isinstance(cfg_raw, list):
        return []

    rows: List[Dict[str, Any]] = []
    for row in cfg_raw:
        if not isinstance(row, dict) or not str(row.get("logsDir", "")).strip():
            continue
        name = str(row.get("name") or "").strip()
        logs_dir = str(row.get("logsDir") or "").strip()
        symbol = str(row.get("symbol") or "").strip().upper()
        if config_set == "lowprice" and is_retired_compare_only_trader(profile, name, symbol):
            continue
        if name.startswith(ARCHIVED_TRADER_PREFIXES) or logs_dir.startswith(ARCHIVED_LOG_DIR_PREFIXES):
            continue
        rows.append(row)
    if not active_only:
        return rows

    active_raw = _load_json(active_path)
    active_names = _extract_active_names(active_raw)
    active_groups = _extract_active_groups(active_raw)
    use_group_fallback = (len(active_names) == 0 and len(active_groups) > 0)

    selected: List[Dict[str, Any]] = []
    for row in rows:
        name = str(row.get("name", "")).strip()
        group = str(row.get("group", "")).strip()
        if name in active_names:
            selected.append(row)
            continue
        if group in ALWAYS_INCLUDE_GROUPS:
            selected.append(row)
            continue
        if use_group_fallback and group and group in active_groups:
            selected.append(row)
    if profile == "default":
        special_names: Set[str] = set()
        for report_path in (EXP13_OFFICIAL_SIM_CANDIDATE_LATEST, ROUTE_F_EXP10_OFFICIAL_SIM_CANDIDATE_LATEST):
            payload = _load_json(report_path)
            summary = payload.get("summary") if isinstance(payload, dict) and isinstance(payload.get("summary"), dict) else {}
            name = str(summary.get("candidate_trader_name") or "").strip()
            if name and (
                bool(summary.get("qualified_for_official_simulation"))
                or bool(summary.get("official_simulation_live"))
                or bool(summary.get("active_present"))
            ):
                special_names.add(name)
        if special_names:
            existing = {str(row.get("name") or "").strip() for row in selected}
            for row in rows:
                name = str(row.get("name") or "").strip()
                if name in special_names and name not in existing:
                    selected.append(row)
                    existing.add(name)
    return selected if selected else rows


def _runtime_dir_index(profile: str, config_set: str = "main") -> Dict[str, List[str]]:
    prefix = _profile_logs_prefix(profile, config_set=config_set)
    fs_dirs = [
        d.name
        for d in SCRIPT_DIR.iterdir()
        if d.is_dir()
        and d.name.startswith(prefix)
        and not _is_sample_runtime_dir(d.name)
    ]
    by_base: Dict[str, List[str]] = {}
    for d in fs_dirs:
        by_base.setdefault(_base_of(d), []).append(d)
    return by_base


def _runtime_dirs(base_dirs: List[str], profile: str, mode_filter: str, config_set: str = "main") -> List[str]:
    by_base = _runtime_dir_index(profile, config_set=config_set)
    out: Set[str] = set()
    for base in base_dirs:
        candidates = by_base.get(base, [])
        if not candidates and (SCRIPT_DIR / base).is_dir():
            candidates = [base]
        for c in candidates:
            if c.startswith(ARCHIVED_LOG_DIR_PREFIXES):
                continue
            if _mode_match(_mode_of(c), mode_filter):
                out.add(c)
    return sorted(out)


def _runtime_rows(
    cfg_rows: List[Dict[str, Any]],
    profile: str,
    mode_filter: str,
    include_runtime_splits: bool,
    prefer_fresh_runtime_dir: bool = False,
    collapse_runtime_splits: bool = False,
    config_set: str = "main",
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in cfg_rows:
        base = str(row.get("logsDir", "")).strip()
        if base.startswith(ARCHIVED_LOG_DIR_PREFIXES):
            continue
        if not base:
            continue
        row_config_set = _config_set_for_runtime_value(profile, base, default_set=config_set)
        by_base = _runtime_dir_index(profile, config_set=row_config_set)
        candidates = by_base.get(base, [])
        if not candidates and (SCRIPT_DIR / base).is_dir():
            candidates = [base]
        if collapse_runtime_splits:
            eligible = [c for c in candidates if _mode_match(_mode_of(c), mode_filter)]
            if eligible:
                split_candidates = [c for c in eligible if _is_runtime_split_dir(c)]
                aggregate_dirs = sorted(eligible)
                if (SCRIPT_DIR / base).is_dir():
                    chosen = base
                elif split_candidates:
                    chosen = sorted(
                        split_candidates,
                        key=lambda c: (
                            _trade_file_activity_ts(c, _mode_of(c)),
                            c,
                        ),
                        reverse=True,
                    )[0]
                else:
                    chosen = max(
                        eligible,
                        key=lambda c: (
                            _trade_file_activity_ts(c, _mode_of(c)),
                            c,
                        ),
                    )
                candidates = [chosen]
            else:
                candidates = []
                aggregate_dirs = []
        elif not include_runtime_splits and prefer_fresh_runtime_dir:
            eligible = [c for c in candidates if _mode_match(_mode_of(c), mode_filter)]
            if eligible:
                split_candidates = [c for c in eligible if _is_runtime_split_dir(c)]
                aggregate_dirs = sorted(split_candidates if split_candidates else eligible)
                if split_candidates:
                    # When a base trader has multiple symbol-specific split dirs, we
                    # aggregate all matching splits for stats and keep the base dir as
                    # the representative runtime_dir. This avoids leaking the freshest
                    # sibling split (for example XRP) into another symbol row's
                    # metadata.
                    if len(split_candidates) > 1:
                        chosen = base
                    else:
                        chosen = sorted(
                            split_candidates,
                            key=lambda c: (
                                _trade_file_activity_ts(c, _mode_of(c)),
                                c,
                            ),
                            reverse=True,
                        )[0]
                else:
                    chosen = max(
                        eligible,
                        key=lambda c: (
                            _trade_file_activity_ts(c, _mode_of(c)),
                            c,
                        ),
                    )
                candidates = [chosen]
            else:
                candidates = []
                aggregate_dirs = []
        elif not include_runtime_splits:
            base_exists = (SCRIPT_DIR / base).is_dir()
            eligible = [c for c in candidates if _mode_match(_mode_of(c), mode_filter)]
            has_split_eligible = any(_is_runtime_split_dir(c) for c in eligible)
            filtered = [c for c in candidates if not _is_runtime_split_dir(c)]
            if filtered:
                candidates = filtered
            elif base_exists:
                candidates = [base]
            aggregate_dirs = sorted(
                [c for c in eligible if _is_runtime_split_dir(c)]
                if has_split_eligible
                else [c for c in candidates if _mode_match(_mode_of(c), mode_filter)]
            )
        else:
            aggregate_dirs = sorted(c for c in candidates if _mode_match(_mode_of(c), mode_filter))
        for c in candidates:
            if not _mode_match(_mode_of(c), mode_filter):
                continue
            copied = dict(row)
            copied["_runtime_dir"] = c
            copied["_base_dir"] = base
            copied["_aggregate_runtime_dirs"] = [c] if include_runtime_splits else list(aggregate_dirs)
            copied["_config_set"] = row_config_set
            out.append(copied)
    return out


def _all_runtime_dirs(profile: str, mode_filter: str, include_runtime_splits: bool, config_set: str = "main") -> List[str]:
    prefix = _profile_logs_prefix(profile, config_set=config_set)
    out = []
    for d in sorted(SCRIPT_DIR.iterdir(), key=lambda p: p.name):
        if not d.is_dir() or not d.name.startswith(prefix):
            continue
        if d.name.startswith(ARCHIVED_LOG_DIR_PREFIXES):
            continue
        if _is_sample_runtime_dir(d.name):
            continue
        if not include_runtime_splits and _is_runtime_split_dir(d.name):
            continue
        if _mode_match(_mode_of(d.name), mode_filter):
            out.append(d.name)
    return out


def _read_mode_file(base: Path, stem: str, mode: str) -> Any:
    mode_path = base / f"{stem}.{mode}.json"
    if mode_path.exists():
        data = _load_json(mode_path)
        if data is not None:
            return data
    # merged compatibility files are simulation-first legacy artifacts; never
    # use them to synthesize live/backtest rows, otherwise live mode will
    # silently inherit simulation history.
    if mode != "simulation":
        return None
    merged_path = base / f"{stem}.json"
    if merged_path.exists():
        return _load_json(merged_path)
    return None


def _read_trades(log_dir: str, mode: str) -> List[Dict[str, Any]]:
    data = _read_mode_file(SCRIPT_DIR / log_dir, "prediction_trades", mode)
    return data if isinstance(data, list) else []


def _parse_summary_txt_metrics(base: Path, mode: str) -> Dict[str, Any]:
    path = base / f"report_summary.{mode}.txt"
    if not path.exists():
        return {"exists": False}
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return {"exists": False}
    out: Dict[str, Any] = {"exists": True}
    patterns = {
        "report_date": (r"^报告时间:\s*(.+)$", str),
        "total_trades": (r"^\s*总逻辑单:\s*([0-9]+)", int),
        "total_fills": (r"^\s*总成交fills:\s*([0-9]+)", int),
        "total_pnl": (r"^\s*总盈亏:\s*\$([+-]?[0-9]+(?:\.[0-9]+)?)", float),
        "current_capital": (r"^\s*当前资金:\s*\$([+-]?[0-9]+(?:\.[0-9]+)?)", float),
    }
    for key, (pattern, cast) in patterns.items():
        match = re.search(pattern, text, re.MULTILINE)
        if not match:
            continue
        raw = match.group(1).strip()
        try:
            out[key] = cast(raw)
        except Exception:
            continue
    return out


def _report_summary_is_valid(base: Path, mode: str, payload: Dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        return False
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return False
    report_date = str(payload.get("reportDate") or "").strip()
    report_period = payload.get("reportPeriod") if isinstance(payload.get("reportPeriod"), dict) else {}
    report_period_end = str(report_period.get("end") or "").strip()
    if not report_date or not report_period_end or report_date != report_period_end:
        return False
    txt_metrics = _parse_summary_txt_metrics(base, mode)
    if not txt_metrics.get("exists"):
        return False
    if txt_metrics.get("report_date") and str(txt_metrics.get("report_date")) != report_date:
        return False
    comparisons = (
        (txt_metrics.get("total_trades"), int(summary.get("totalTrades") or summary.get("totalLogicalTrades") or 0), False),
        (txt_metrics.get("total_fills"), int(summary.get("totalFills") or summary.get("fillCount") or 0), False),
        (txt_metrics.get("total_pnl"), float(summary.get("totalPnL") or 0.0), True),
        (txt_metrics.get("current_capital"), float(summary.get("currentCapital") or 0.0), True),
    )
    for txt_value, payload_value, is_float in comparisons:
        if txt_value is None:
            continue
        if is_float:
            if abs(float(txt_value) - payload_value) > 0.05:
                return False
        elif int(txt_value) != payload_value:
            return False
    return True


def _read_report_summary(log_dir: str, mode: str) -> Dict[str, Any]:
    base = SCRIPT_DIR / _base_of(log_dir) / "reports"
    payload = _read_mode_file(base, "report_summary", mode)
    if isinstance(payload, dict) and _report_summary_is_valid(base, mode, payload):
        summary = payload.get("summary")
        return summary if isinstance(summary, dict) else {}
    return {}


def _report_summary_bundle(log_dir: str, mode: str) -> Dict[str, Any]:
    report_bases: List[Path] = []
    runtime_base = SCRIPT_DIR / log_dir / "reports"
    base_base = SCRIPT_DIR / _base_of(log_dir) / "reports"
    for candidate in (runtime_base, base_base):
        if candidate not in report_bases:
            report_bases.append(candidate)
    for reports_base in report_bases:
        lifecycle_path = reports_base / f"lifecycle_snapshot.{mode}.json"
        lifecycle_payload = _load_json(lifecycle_path) if lifecycle_path.exists() else {}
        lifecycle_summary = lifecycle_payload.get("summary") if isinstance(lifecycle_payload, dict) and isinstance(lifecycle_payload.get("summary"), dict) else {}
        lifecycle_by_symbol = lifecycle_payload.get("bySymbol") if isinstance(lifecycle_payload, dict) and isinstance(lifecycle_payload.get("bySymbol"), dict) else {}
        report_payload = _read_mode_file(reports_base, "report_summary", mode)
        if not isinstance(report_payload, dict) or not _report_summary_is_valid(reports_base, mode, report_payload):
            report_payload = {}
        report_summary = report_payload.get("summary") if isinstance(report_payload.get("summary"), dict) else {}
        report_by_symbol = report_payload.get("bySymbol") if isinstance(report_payload.get("bySymbol"), dict) else {}
        report_summary_path = reports_base / f"report_summary.{mode}.json"
        if not report_summary_path.exists() and mode == "simulation":
            merged = reports_base / "report_summary.json"
            if merged.exists():
                report_summary_path = merged

        def _summary_sig(summary_payload: Dict[str, Any]) -> Tuple[int, int, int, int, float, float]:
            return (
                int(summary_payload.get("tradeCount") or summary_payload.get("totalTrades") or summary_payload.get("totalLogicalTrades") or 0),
                int(summary_payload.get("winCount") or summary_payload.get("wins") or 0),
                int(summary_payload.get("lossCount") or summary_payload.get("losses") or 0),
                int(summary_payload.get("pendingCount") or summary_payload.get("pending") or 0),
                round(float(summary_payload.get("capitalChange") or summary_payload.get("totalPnL") or 0.0), 2),
                round(float(summary_payload.get("currentCapital") or 0.0), 2),
            )

        compat_summary_path = reports_base / "report_summary.json"
        compat_payload = _load_json(compat_summary_path) if mode == "simulation" and compat_summary_path.exists() else {}
        compat_summary = compat_payload.get("summary") if isinstance(compat_payload, dict) and isinstance(compat_payload.get("summary"), dict) else {}
        mode_summary_path = reports_base / f"report_summary.{mode}.json"
        mode_summary_sig = _summary_sig(report_summary) if report_summary else None
        compat_summary_sig = _summary_sig(compat_summary) if compat_summary else None
        compat_summary_stale = bool(
            mode == "simulation"
            and report_summary
            and compat_summary
            and mode_summary_path.exists()
            and compat_summary_sig != mode_summary_sig
        )
        compat_meta = {
            "modeSummaryPath": str(mode_summary_path) if mode_summary_path.exists() else None,
            "compatSummaryPath": str(compat_summary_path) if compat_summary_path.exists() else None,
            "compatSummary": compat_summary,
            "compatSummaryStale": compat_summary_stale,
            "compatSummarySig": compat_summary_sig,
            "modeSummarySig": mode_summary_sig,
        }

        if lifecycle_summary and report_summary:
            lifecycle_sig = _summary_sig(lifecycle_summary)
            report_sig = _summary_sig(report_summary)
            lifecycle_mtime = float(lifecycle_path.stat().st_mtime) if lifecycle_path.exists() else 0.0
            report_mtime = float(report_summary_path.stat().st_mtime) if report_summary_path.exists() else 0.0
            if report_mtime >= lifecycle_mtime or report_sig != lifecycle_sig:
                return {
                    "summaryPath": str(report_summary_path) if report_summary_path.exists() else None,
                    "payload": report_payload,
                    "summary": report_summary,
                    "bySymbol": report_by_symbol,
                    "reportBase": str(reports_base),
                    **compat_meta,
                }
            return {
                "summaryPath": str(lifecycle_path),
                "payload": lifecycle_payload,
                "summary": lifecycle_summary,
                "bySymbol": lifecycle_by_symbol,
                "reportBase": str(reports_base),
                **compat_meta,
            }
        if report_summary:
            return {
                "summaryPath": str(report_summary_path) if report_summary_path.exists() else None,
                "payload": report_payload,
                "summary": report_summary,
                "bySymbol": report_by_symbol,
                "reportBase": str(reports_base),
                **compat_meta,
            }
        if lifecycle_summary:
            return {
                "summaryPath": str(lifecycle_path),
                "payload": lifecycle_payload,
                "summary": lifecycle_summary,
                "bySymbol": lifecycle_by_symbol,
                "reportBase": str(reports_base),
                **compat_meta,
            }
    return {
        "summaryPath": None,
        "payload": {},
        "summary": {},
        "bySymbol": {},
        "reportBase": None,
        "modeSummaryPath": None,
        "compatSummaryPath": None,
        "compatSummary": {},
        "compatSummaryStale": False,
        "compatSummarySig": None,
        "modeSummarySig": None,
    }


def _prediction_trades_path(log_dir: str, mode: str) -> str | None:
    runtime_path = SCRIPT_DIR / log_dir / f"prediction_trades.{mode}.json"
    if runtime_path.exists():
        return str(runtime_path)
    if mode == "simulation":
        merged_runtime = SCRIPT_DIR / log_dir / "prediction_trades.json"
        if merged_runtime.exists():
            return str(merged_runtime)
    base_dir = _base_of(log_dir)
    if base_dir != log_dir:
        base_path = SCRIPT_DIR / base_dir / f"prediction_trades.{mode}.json"
        if base_path.exists():
            return str(base_path)
        if mode == "simulation":
            merged_base = SCRIPT_DIR / base_dir / "prediction_trades.json"
            if merged_base.exists():
                return str(merged_base)
    return None


def _summary_number(summary: Dict[str, Any], key: str, fallback: Any, cast):
    if key not in summary:
        return fallback
    value = summary.get(key)
    if value is None:
        return fallback
    try:
        return cast(value)
    except Exception:
        return fallback


def _trade_stats_for_rows(raw: List[Dict[str, Any]]) -> Dict[str, Any]:
    executed = [t for t in raw if str(t.get("status", "")).lower() == "executed"]
    pending_rows = [t for t in raw if str(t.get("status", "")).lower() == "pending"]
    cleared_rows = [
        t for t in raw
        if str(t.get("result", "")).lower() == "no_fill_cleared"
        or str(t.get("status", "")).lower() == "cleared_no_fill"
    ]

    def _logical_key(t: Dict[str, Any]) -> str:
        condition_id = str(t.get("conditionId") or "").strip()
        direction = str(t.get("direction") or t.get("tokenOutcome") or "").strip().upper()
        suffix = f"::{direction}" if direction in {"UP", "DOWN"} else ""
        if condition_id:
            return condition_id + suffix
        market_slug = str(t.get("marketSlug") or "").strip()
        if market_slug:
            return market_slug + suffix
        trade_id = str(t.get("id") or "").strip()
        return trade_id or f"_no_key_{id(t)}"

    def _row_event_ts(t: Dict[str, Any]) -> float | None:
        for key in ("settledAt", "decisionTs", "timestamp", "arrivalTs"):
            text = str(t.get(key) or "").strip()
            if not text:
                continue
            try:
                return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
            except Exception:
                continue
        return None

    def _row_target_end_ts(t: Dict[str, Any]) -> float | None:
        for key in ("targetPeriodEndTs", "endTs"):
            raw_value = t.get(key)
            if raw_value in (None, "", 0):
                continue
            try:
                return float(raw_value)
            except Exception:
                continue
        return None

    def _settled_result(t: Dict[str, Any]) -> str:
        result = str(t.get("result") or t.get("settlementOutcome") or "").lower()
        return result if result in {"win", "lose"} else ""

    by_order: Dict[str, str | None] = {}
    grouped_rows: Dict[str, List[Dict[str, Any]]] = {}
    latest_ts: float | None = None
    latest_settled_ts: float | None = None
    total_pnl = 0.0
    for t in raw:
        logical_key = _logical_key(t)
        by_order.setdefault(logical_key, None)
        grouped_rows.setdefault(logical_key, []).append(t)
        r = str(t.get("result") or "").lower()
        if str(t.get("status", "")).lower() == "executed" and r in ("win", "lose"):
            by_order[logical_key] = r
        elif r == "no_fill_cleared" or str(t.get("status", "")).lower() == "cleared_no_fill":
            by_order[logical_key] = "no_fill_cleared"
    for t in raw:
        pnl = t.get("pnl")
        if str(t.get("result") or "").lower() in {"win", "lose"} and pnl is not None:
            try:
                total_pnl += float(pnl)
            except Exception:
                pass
        ts_str = t.get("timestamp")
        if isinstance(ts_str, str) and ts_str:
            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
                if latest_ts is None or ts > latest_ts:
                    latest_ts = ts
            except Exception:
                pass
        settled = _settled_result(t)
        if settled:
            settled_ts = _row_event_ts(t)
            if settled_ts is not None and (latest_settled_ts is None or settled_ts > latest_settled_ts):
                latest_settled_ts = settled_ts
    wins = sum(1 for v in by_order.values() if v == "win")
    losses = sum(1 for v in by_order.values() if v == "lose")
    pending = sum(1 for v in by_order.values() if v is None)
    cleared_no_fill = sum(1 for v in by_order.values() if v == "no_fill_cleared")
    completed = wins + losses
    win_rate = (wins / completed * 100) if completed else 0.0
    latest_str = datetime.fromtimestamp(latest_ts).strftime("%m-%d %H:%M") if latest_ts else "-"
    latest_settled_str = datetime.fromtimestamp(latest_settled_ts).strftime("%m-%d %H:%M") if latest_settled_ts else "-"
    latest_pending_row: Dict[str, Any] | None = None
    latest_pending_target_end_ts: float | None = None
    latest_pending_ts: float | None = None
    latest_pending_filled_usd = 0.0
    for logical_key, state in by_order.items():
        if state is not None:
            continue
        rows = grouped_rows.get(logical_key) or []
        target_end_ts = max((_row_target_end_ts(row) or 0.0) for row in rows) if rows else 0.0
        event_ts = max((_row_event_ts(row) or 0.0) for row in rows) if rows else 0.0
        if latest_pending_row is None or (target_end_ts, event_ts) > ((latest_pending_target_end_ts or 0.0), (latest_pending_ts or 0.0)):
            latest_pending_target_end_ts = target_end_ts or None
            latest_pending_ts = event_ts or None
            latest_pending_row = max(rows, key=lambda row: ((_row_target_end_ts(row) or 0.0), (_row_event_ts(row) or 0.0))) if rows else None
            latest_pending_filled_usd = 0.0
            for row in rows:
                try:
                    latest_pending_filled_usd += float(row.get("filledUsd") or row.get("amount") or 0.0)
                except Exception:
                    continue
    result_pnl_mismatch_examples: List[Dict[str, Any]] = []
    for logical_key, rows in grouped_rows.items():
        executed_rows = [row for row in rows if str(row.get("status", "")).lower() == "executed"]
        if not executed_rows:
            continue
        result = None
        for row in executed_rows:
            candidate = str(row.get("result") or "").lower()
            if candidate in {"win", "lose"}:
                result = candidate
        if result not in {"win", "lose"}:
            continue
        pnl = 0.0
        for row in executed_rows:
            try:
                pnl += float(row.get("pnl") or 0.0)
            except Exception:
                continue
        mismatch = (result == "win" and pnl < -0.01) or (result == "lose" and pnl > 0.01)
        if mismatch:
            result_pnl_mismatch_examples.append(
                {
                    "logicalKey": logical_key,
                    "result": result,
                    "pnlUsd": round(pnl, 2),
                    "firstTimestamp": str(executed_rows[0].get("timestamp") or ""),
                    "marketSlug": str(executed_rows[0].get("marketSlug") or ""),
                }
            )
    return {
        "n_executed": len(executed),
        "n_pending_fills": len(pending_rows),
        "n_logical_orders": len(by_order),
        "n_markets": len(by_order),
        "n_completed_markets": completed,
        "settled_trades": completed,
        "settled_win_rate": win_rate,
        "wins": wins,
        "losses": losses,
        "pending": pending,
        "cleared_no_fill": cleared_no_fill,
        "n_cleared_no_fill": len(cleared_rows),
        "win_rate": win_rate,
        "total_pnl": round(total_pnl, 2),
        "latest_str": latest_str,
        "latest_ts": latest_ts,
        "latest_settled_str": latest_settled_str,
        "latest_settled_ts": latest_settled_ts,
        "latest_pending_market_slug": str((latest_pending_row or {}).get("marketSlug") or "").strip() or None,
        "latest_pending_direction": str((latest_pending_row or {}).get("direction") or (latest_pending_row or {}).get("tokenOutcome") or "").strip().upper() or None,
        "latest_pending_target_period_end_ts": latest_pending_target_end_ts,
        "latest_pending_posted_usd": round(float((latest_pending_row or {}).get("postedUsd") or (latest_pending_row or {}).get("requestedAmountUsd") or (latest_pending_row or {}).get("requestedUsd") or (latest_pending_row or {}).get("targetUsd") or 0.0), 2),
        "latest_pending_filled_usd": round(latest_pending_filled_usd, 2),
        "resultPnlMismatchCount": len(result_pnl_mismatch_examples),
        "resultPnlMismatchExamples": result_pnl_mismatch_examples[:5],
    }


def _iso_from_timestamp(value: Any) -> str | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat()
    except Exception:
        return None


def _summary_stats_for_symbol(
    summary_bundle: Dict[str, Any],
    symbol: str,
    *,
    only_symbol: bool,
) -> Dict[str, Any]:
    summary = summary_bundle.get("summary") if isinstance(summary_bundle.get("summary"), dict) else {}
    by_symbol = summary_bundle.get("bySymbol") if isinstance(summary_bundle.get("bySymbol"), dict) else {}
    symbol_text = str(symbol or "").strip().upper()
    symbol_summary = by_symbol.get(symbol_text) if isinstance(by_symbol.get(symbol_text), dict) else {}
    source = symbol_summary if symbol_summary else summary
    if not source:
        return {}
    if source is summary and by_symbol and not only_symbol:
        return {}
    trades = _summary_number(
        source,
        "trades",
        _summary_number(source, "totalTrades", _summary_number(source, "totalLogicalTrades", 0, int), int),
        int,
    )
    fills = _summary_number(
        source,
        "fillCount",
        _summary_number(source, "totalFills", 0, int),
        int,
    )
    wins = _summary_number(source, "wins", 0, int)
    losses = _summary_number(source, "losses", 0, int)
    pending = _summary_number(source, "pending", 0, int)
    completed = wins + losses
    win_rate = _summary_number(source, "winRate", ((wins / completed * 100) if completed else 0.0), float)
    pnl = _summary_number(
        source,
        "pnl",
        _summary_number(source, "logicalTradePnl", _summary_number(source, "totalPnL", 0.0, float), float),
        float,
    )
    initial_capital = None
    current_capital = None
    if isinstance(summary, dict):
        if only_symbol or not by_symbol or len(by_symbol) <= 1:
            initial_capital = _summary_number(summary, "initialCapital", None, float)
            current_capital = _summary_number(summary, "currentCapital", None, float)
    return {
        "n_executed": fills,
        "n_pending_fills": _summary_number(source, "pendingOrderCount", 0, int),
        "n_logical_orders": trades,
        "n_markets": trades,
        "n_completed_markets": completed,
        "wins": wins,
        "losses": losses,
        "pending": pending,
        "win_rate": win_rate,
        "total_pnl": round(pnl, 2),
        "summaryTrades": trades,
        "summaryPnL": round(pnl, 2),
        "initialCapital": initial_capital,
        "currentCapital": current_capital,
        "capitalChange": (
            round(float(current_capital) - float(initial_capital), 2)
            if initial_capital is not None and current_capital is not None
            else None
        ),
    }


def _existing_trade_modes(log_dir: str) -> List[str]:
    base = SCRIPT_DIR / _base_of(log_dir)
    modes: List[str] = []
    for mode in ("simulation", "live", "backtest"):
        path = base / f"prediction_trades.{mode}.json"
        try:
            if path.exists() and bool(read_mode_activity(_base_of(log_dir), mode).get("has_activity")):
                modes.append(mode)
        except Exception:
            continue
    return modes


def _trade_file_activity_ts(log_dir: str, mode: str) -> float:
    base = SCRIPT_DIR / log_dir
    candidates = [base / f"prediction_trades.{mode}.json"]
    if mode == "simulation":
        candidates.append(base / "prediction_trades.json")
    latest = 0.0
    for path in candidates:
        try:
            if path.exists():
                latest = max(latest, float(path.stat().st_mtime))
        except Exception:
            continue
    return latest


def _core_scope_snapshot() -> Dict[str, Any]:
    payload = _load_json(CORE_UNIVERSE_LATEST)
    if not isinstance(payload, dict):
        return {}
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    core_cells = payload.get("core_cells") if isinstance(payload.get("core_cells"), list) else []
    top = sorted(
        [
            r for r in core_cells
            if isinstance(r, dict)
        ],
        key=lambda r: float(((r.get("runtime_stats") or {}).get("total_pnl") or 0.0)),
        reverse=True,
    )[:3]
    top_by_profile: Dict[str, List[Dict[str, Any]]] = {}
    for profile_key in ("default", "70"):
        scoped = [
            r for r in core_cells
            if isinstance(r, dict) and str(r.get("profile") or "") == profile_key
        ]
        top_by_profile[profile_key] = [
            {
                "profile": str(r.get("profile") or ""),
                "trader": str(r.get("trader") or ""),
                "symbol": str(r.get("symbol") or ""),
                "total_pnl": round(float(((r.get("runtime_stats") or {}).get("total_pnl") or 0.0)), 2),
            }
            for r in sorted(
                scoped,
                key=lambda r: float(((r.get("runtime_stats") or {}).get("total_pnl") or 0.0)),
                reverse=True,
            )[:3]
        ]
    return {
        "core_cells": int(summary.get("core_cells") or len(core_cells)),
        "core_traders": int(summary.get("core_traders") or 0),
        "shadow_cells": int(summary.get("shadow_cells") or 0),
        "core_by_profile": summary.get("core_by_profile") if isinstance(summary.get("core_by_profile"), dict) else {},
        "core_traders_by_profile": summary.get("core_traders_by_profile") if isinstance(summary.get("core_traders_by_profile"), dict) else {},
        "ranking_source": str(summary.get("ranking_source") or payload.get("ranking_source") or ""),
        "top_core_by_profile": top_by_profile,
        "top_core": [
            {
                "profile": str(r.get("profile") or ""),
                "trader": str(r.get("trader") or ""),
                "symbol": str(r.get("symbol") or ""),
                "total_pnl": round(float(((r.get("runtime_stats") or {}).get("total_pnl") or 0.0)), 2),
            }
            for r in top
        ],
    }


def _extreme_guard_snapshot() -> Dict[str, Any]:
    state = _load_json(EXTREME_REGIME_STATE_LATEST)
    overlay = _load_json(EXTREME_OVERLAY_APPLY_LATEST)
    state_summary = state.get("summary") if isinstance(state, dict) and isinstance(state.get("summary"), dict) else {}
    overlay_summary = overlay.get("summary") if isinstance(overlay, dict) and isinstance(overlay.get("summary"), dict) else {}
    up_symbols = int(state_summary.get("extreme_up_symbols") or 0)
    down_symbols = int(state_summary.get("extreme_down_symbols") or 0)
    return {
        "extreme_active_symbols": up_symbols + down_symbols,
        "extreme_up_symbols": up_symbols,
        "extreme_down_symbols": down_symbols,
        "countertrend_hard_block_cells": int(overlay_summary.get("countertrend_hard_block_cells") or 0),
        "actions_total": int(overlay_summary.get("actions_total") or 0),
    }


def _fusion_execution_snapshot() -> Dict[str, Any]:
    payload = _load_json(MAINLINE_FINAL_STATUS_LATEST)
    summary = payload.get("summary") if isinstance(payload, dict) and isinstance(payload.get("summary"), dict) else {}
    return {
        "default_cells": int(summary.get("fusion_default_cells") or 0),
        "cells_70": int(summary.get("fusion_70_cells") or 0),
        "prediction_fresh": bool(summary.get("fusion_prediction_fresh")),
        "execution_state": str(summary.get("fusion_execution_state") or ""),
        "simulated_trade_cells": int(summary.get("fusion_simulated_trade_cells") or 0),
        "prediction_only_cells": int(summary.get("fusion_prediction_only_cells") or 0),
        "shadow_only_cells": int(summary.get("fusion_shadow_only_cells") or 0),
        "skipped_cells": int(summary.get("fusion_skipped_by_trade_decision_cells") or 0),
        "execution_not_landed_cells": int(summary.get("fusion_execution_not_landed_cells") or 0),
    }


def _hidden_zero_state_reason_summary(rows: List[Dict[str, Any]], scope: str, active_only: bool, view: str) -> Dict[str, int]:
    if not (scope == "core" and active_only and view == "cell"):
        return {}
    payload = _load_json(MAINLINE_SOURCE_AUDIT_LATEST)
    audit_rows = payload.get("rows") if isinstance(payload, dict) and isinstance(payload.get("rows"), list) else []
    audit_by_cell: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in audit_rows:
        if not isinstance(row, dict):
            continue
        key = (
            str(row.get("profile") or ""),
            str(row.get("trader") or ""),
            str(row.get("symbol") or "").upper(),
        )
        audit_by_cell[key] = row

    guard_by_trader: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for guard_path in (SCRIPT_DIR / "logs").glob("runtime_trade_guards_*.json"):
        guard_payload = _load_json(guard_path)
        traders = (
            ((guard_payload or {}).get("guard_evidence_v1") or {}).get("traders")
            if isinstance(guard_payload, dict)
            else None
        )
        if not isinstance(traders, list):
            continue
        for item in traders:
            if not isinstance(item, dict):
                continue
            guard_key = (
                str(item.get("profile") or "default"),
                str(item.get("traderName") or ""),
            )
            if not guard_key[1]:
                continue
            guard_by_trader[guard_key] = item

    out = {
        "候选未提级": 0,
        "当前无下单信号": 0,
        "选择器拦截": 0,
        "运行态风控拦截": 0,
        "70阈值过滤为空": 0,
        "主线已生效但尚无成交": 0,
        "仅配置零状态": 0,
    }
    for row in rows:
        if not _is_config_only_zero_row(row, scope, active_only, view):
            continue
        key = (
            str(row.get("profile") or ""),
            str(row.get("trader") or ""),
            str(row.get("symbol") or "").upper(),
        )
        audit = audit_by_cell.get(key)
        if not isinstance(audit, dict):
            out["仅配置零状态"] += 1
            continue
        status = str(audit.get("status") or "")
        if status == "candidate_only":
            out["候选未提级"] += 1
            continue
        if status == "mainline_active":
            prediction = audit.get("prediction") if isinstance(audit.get("prediction"), dict) else {}
            sample = audit.get("effective_prediction_sample")
            details = sample.get("details") if isinstance(sample, dict) and isinstance(sample.get("details"), dict) else {}
            trade_decision = details.get("trade_decision") if isinstance(details.get("trade_decision"), dict) else {}
            should_trade = trade_decision.get("should_trade")
            trader = str(audit.get("trader") or row.get("trader") or "")
            runtime_trader_name = f"{trader}__simulation_{str(row.get('symbol') or '').lower()}"
            guard = guard_by_trader.get((key[0], runtime_trader_name), {})
            passed = int(guard.get("passed") or 0) if isinstance(guard, dict) else 0
            skipped_selector = int(guard.get("skippedBySelector") or 0) if isinstance(guard, dict) else 0
            skipped_other_guard = 0
            if isinstance(guard, dict):
                skipped_other_guard = sum(
                    int(guard.get(name) or 0)
                    for name in (
                        "skippedByUncertainty",
                        "skippedByRegime",
                        "skippedByDirectionLossCooldown",
                        "skippedByDrawdownAcceleration",
                    )
                )
            if bool(audit.get("derived_via_filter70")) and not bool(prediction.get("sample_found_for_symbol")):
                out["70阈值过滤为空"] += 1
            elif passed == 0 and skipped_selector > 0:
                out["选择器拦截"] += 1
            elif passed == 0 and skipped_other_guard > 0:
                out["运行态风控拦截"] += 1
            elif should_trade is False:
                out["当前无下单信号"] += 1
            else:
                out["主线已生效但尚无成交"] += 1
            continue
        out["仅配置零状态"] += 1
    return {k: int(v) for k, v in out.items() if int(v) > 0}


def _is_shadow_only_fusion_row(row: Dict[str, Any], fusion: Dict[str, Any], scope: str, active_only: bool, view: str) -> bool:
    if not (scope == "core" and active_only and view in {"cell", "model"}):
        return False
    if int(fusion.get("simulated_trade_cells") or 0) > 0:
        return False
    if int(fusion.get("shadow_only_cells") or 0) <= 0:
        return False
    base_dir = str(row.get("base_dir") or "").lower()
    trader = str(row.get("trader") or "").lower()
    short = str(row.get("short") or "").lower()
    return (
        ("ensemble" in base_dir or "ensemble" in trader or short.startswith("ens_"))
        and int(row.get("n_executed") or 0) <= 0
    )


def _is_config_only_zero_row(row: Dict[str, Any], scope: str, active_only: bool, view: str) -> bool:
    if not (scope == "core" and active_only and view == "cell"):
        return False
    if str(row.get("role") or "").strip().lower() == "compare_only":
        return False
    return str(row.get("mode") or "") == "config_only_zero_state"


def _short_name(log_dir: str) -> str:
    base = _base_of(log_dir)
    for prefix in ("logs_threshold_pack_", "logs_70_", "logs_"):
        if base.startswith(prefix):
            base = base[len(prefix):]
            break
    base = base.replace("v5_", "").replace("ensemble_", "ens_")
    m = re.search(r"slot02_newmethod_(eth|btc|sol|xrp)(?:_sim400)?", base, flags=re.IGNORECASE)
    if m:
        return f"cmp_slot02_newmethod_{m.group(1).lower()}"
    return base


def _aggregate_trade_stats(raw: List[Dict[str, Any]]) -> Dict[str, Any]:
    stats = _trade_stats_for_rows(raw)
    return {
        "n_executed": int(stats.get("n_executed") or 0),
        "n_logical_orders": int(stats.get("n_logical_orders") or 0),
        "n_markets": int(stats.get("n_markets") or 0),
        "n_completed_markets": int(stats.get("n_completed_markets") or 0),
        "wins": int(stats.get("wins") or 0),
        "losses": int(stats.get("losses") or 0),
        "pending": int(stats.get("pending") or 0),
        "win_rate": float(stats.get("win_rate") or 0.0),
        "total_pnl": round(float(stats.get("total_pnl") or 0.0), 2),
        "latest_str": str(stats.get("latest_str") or "-"),
        "latest_ts": stats.get("latest_ts"),
    }


def _load_model_stats(
    log_dir: str,
    profile: str,
    aggregate_dirs: List[str] | None = None,
    config_set: str = "main",
) -> Dict[str, Any]:
    mode = _mode_of(log_dir)
    dirs = [str(x) for x in aggregate_dirs if str(x).strip()] if isinstance(aggregate_dirs, list) else [log_dir]
    raw: List[Dict[str, Any]] = []
    for d in dirs:
        raw.extend(_read_trades(d, _mode_of(d)))
    stats = _aggregate_trade_stats(raw)
    display_dir = _base_of(log_dir) if len(dirs) > 1 else log_dir

    return {
        "profile": profile,
        "runtime_dir": display_dir,
        "base_dir": _base_of(log_dir),
        "short": _short_name(display_dir),
        "mode": mode,
        "_aggregate_runtime_dirs": dirs,
        "_config_set": config_set,
        **stats,
    }


def _normalize_symbols(row: Dict[str, Any], raw: List[Dict[str, Any]], runtime_dir: str) -> List[str]:
    seen = {str(r.get("symbol", "")).upper() for r in raw if isinstance(r, dict)}
    out: List[str] = sorted(s for s in seen if s)
    if out:
        return out
    hint = _runtime_split_symbol_hint(runtime_dir)
    if hint:
        return [hint]
    allowed = row.get("allowedMarkets")
    out = []
    if isinstance(allowed, list):
        out = [str(x).upper() for x in allowed if str(x).strip()]
    elif isinstance(allowed, str):
        out = [x.strip().upper() for x in allowed.split(",") if x.strip()]
    if not out:
        out = ["BTC", "ETH", "XRP"]
    return out


def _label_from_row(row: Dict[str, Any], runtime_dir: str, profile: str, config_set: str = "main") -> str:
    display_alias = str(row.get("displayAlias") or "").strip()
    if display_alias:
        return display_alias
    for raw_value in (
        row.get("name"),
        row.get("trader"),
        row.get("lowPriceSourceTrader"),
        row.get("sourceTrader"),
        runtime_dir,
        row.get("logsDir"),
    ):
        value = str(raw_value or "").strip()
        m = re.search(r"slot02_newmethod_(eth|btc|sol|xrp)(?:_sim400)?", value, flags=re.IGNORECASE)
        if m:
            return f"cmp_slot02_newmethod_{m.group(1).lower()}"
    label_prefix = _profile_label_prefix(profile, config_set=config_set)
    lowprice_source = str(row.get("lowPriceSourceTrader") or row.get("sourceTrader") or "").strip()
    if lowprice_source:
        source = lowprice_source.replace("v5_", "").replace("ensemble_", "ens_")
        selection_mode = str(
            row.get("selectionMode")
            or row.get("selection_mode")
            or row.get("lowPriceSelectionMode")
            or ""
        ).strip().lower()
        buy_price = row.get("lowPriceSelectedBuyPrice")
        if buy_price is None:
            buy_price = row.get("buyPrice") if row.get("buyPrice") is not None else row.get("selected_buy_price")
        buy_price_range = row.get("lowPriceSelectedBuyPriceRange")
        if buy_price_range is None:
            buy_price_range = (
                row.get("buyPriceRange")
                if row.get("buyPriceRange") is not None
                else row.get("selected_buy_price_range")
            )
        finalist_key = str(
            row.get("lowPriceFinalistKey")
            or row.get("finalistKey")
            or row.get("finalist_key")
            or row.get("lowpriceFamilyId")
            or row.get("lowPriceTag")
            or row.get("priceTag")
            or ""
        ).strip()
        finalist_key = _normalize_lowprice_finalist_key(finalist_key)
        tag = ""
        try:
            if selection_mode == "fixed_price" and buy_price is not None:
                tag = f"bp{int(round(float(buy_price) * 1000)):04d}"
            elif isinstance(buy_price_range, list) and len(buy_price_range) >= 2:
                low = int(round(float(buy_price_range[0]) * 1000))
                high = int(round(float(buy_price_range[-1]) * 1000))
                tag = f"bp_dyn_{low:04d}_{high:04d}"
            elif finalist_key.startswith("fixed_bp"):
                tag = finalist_key[len("fixed_"):]
            elif finalist_key.startswith("dynamic_bp"):
                tag = finalist_key[len("dynamic_"):]
        except Exception:
            tag = ""
        base = f"cmp_{source}"
        if tag:
            base = f"{base}_{tag}"
        return f"{label_prefix}{base}" if label_prefix else base
    name = str(row.get("name", "")).strip()
    if name:
        base = name.replace("v5_", "").replace("ensemble_", "ens_")
        if str(row.get("role") or "").strip().lower() == "compare_only" and not base.startswith("cmp_"):
            base = f"cmp_{base}"
        if label_prefix and not base.startswith(label_prefix):
            return f"{label_prefix}{base}"
        return base
    return _short_name(runtime_dir)


def _load_cell_stats(runtime_row: Dict[str, Any], profile: str, config_set: str = "main") -> List[Dict[str, Any]]:
    runtime_dir = str(runtime_row.get("_runtime_dir") or runtime_row.get("logsDir") or "")
    mode = _mode_of(runtime_dir)
    aggregate_dirs = runtime_row.get("_aggregate_runtime_dirs")
    dirs = [str(x) for x in aggregate_dirs if str(x).strip()] if isinstance(aggregate_dirs, list) else [runtime_dir]
    row_config_set = str(runtime_row.get("_config_set") or _config_set_for_runtime_value(profile, runtime_dir, default_set=config_set))
    base_label = _label_from_row(runtime_row, runtime_dir, profile, config_set=row_config_set)
    trader_name = str(runtime_row.get("name") or "").strip()
    role = str(runtime_row.get("role") or "").strip().lower()
    allowed_symbols = _normalize_symbols(runtime_row, [], runtime_dir)
    live_target_symbol = None
    if role == "compare_only" and len(allowed_symbols) == 1:
        candidate_symbol = allowed_symbols[0]
        if _registry_mode_for_cell(profile, trader_name, candidate_symbol) == "live":
            live_target_symbol = candidate_symbol
    rows: List[Dict[str, Any]] = []
    mode_groups: List[tuple[str, List[Dict[str, Any]], str]] = []

    if mode == "simulation":
        existing_modes = _existing_trade_modes(runtime_dir)
        if len(existing_modes) > 1:
            for existing_mode in existing_modes:
                scoped_raw: List[Dict[str, Any]] = []
                for d in dirs:
                    scoped_raw.extend(_read_trades(_base_of(d), existing_mode))
                suffix = "sim" if existing_mode == "simulation" else existing_mode
                mode_groups.append((existing_mode, scoped_raw, f"{base_label}-{suffix}"))
        else:
            raw: List[Dict[str, Any]] = []
            for d in dirs:
                raw.extend(_read_trades(d, _mode_of(d)))
            mode_groups.append((mode, raw, base_label))
        if live_target_symbol and not any(existing_mode == "live" for existing_mode, _, _ in mode_groups):
            live_raw: List[Dict[str, Any]] = []
            for d in dirs:
                live_raw.extend(_read_trades(_base_of(d), "live"))
            mode_groups = [
                (
                    existing_mode,
                    scoped_raw,
                    (f"{base_label}-sim" if existing_mode == "simulation" and short_base == base_label else short_base),
                )
                for existing_mode, scoped_raw, short_base in mode_groups
            ]
            mode_groups.append(("live", live_raw, f"{base_label}-live"))
    else:
        raw: List[Dict[str, Any]] = []
        for d in dirs:
            raw.extend(_read_trades(d, _mode_of(d)))
        mode_groups.append((mode, raw, base_label))

    all_raw: List[Dict[str, Any]] = []
    for _, scoped_raw, _ in mode_groups:
        all_raw.extend(scoped_raw)
    symbols = _normalize_symbols(runtime_row, all_raw, runtime_dir)

    trader_name = str(runtime_row.get("name") or "").strip()
    only_symbol = len(symbols) == 1
    for row_mode, raw, short_base in mode_groups:
        summary_bundle = _report_summary_bundle(runtime_dir, row_mode)
        summary_present = bool(summary_bundle.get("summary"))
        summary_path = summary_bundle.get("summaryPath")
        prediction_trades_path = _prediction_trades_path(runtime_dir, row_mode)
        state_path = Path(runtime_dir) / f"forward_state.{row_mode}.json" if runtime_dir else None
        state_payload = _load_json(state_path) if state_path and state_path.exists() else {}
        for sym in symbols:
            rows_for_sym = [t for t in raw if str(t.get("symbol", "")).upper() == sym]
            trade_stats = _trade_stats_for_rows(rows_for_sym)
            summary_stats = _summary_stats_for_symbol(summary_bundle, sym, only_symbol=only_symbol)
            truth_source = "trades_only"
            stats = dict(trade_stats)
            strategy_logical_pnl_usd = round(float(trade_stats.get("total_pnl") or 0.0), 2)
            display_live_wallet_pnl_usd = round(float(trade_stats.get("total_pnl") or 0.0), 2)
            if row_mode == "live":
                live_summary = summary_bundle.get("summary") if summary_present else {}
                if isinstance(live_summary, dict) and live_summary:
                    initial_capital = _summary_number(live_summary, "initialCapital", 0.0, float)
                    strategy_logical_pnl_usd = _summary_number(
                        live_summary,
                        "strategyLogicalPnlUsd",
                        _summary_number(live_summary, "totalPnL", 0.0, float),
                        float,
                    )
                    display_live_wallet_pnl_usd = _summary_number(
                        live_summary,
                        "displayLiveWalletPnlUsd",
                        (
                            _summary_number(live_summary, "currentCapital", 0.0, float) - initial_capital
                            if initial_capital
                            else strategy_logical_pnl_usd
                        ),
                        float,
                    )
                    display_reference_capital = initial_capital
                    external_cash_adjustment = 0.0
                    external_cash_source = None
                    if trader_name == "v5_exp10_bp_dyn_0300_0440_eth":
                        cash_sync = _slot01_external_cash_adjustment(
                            live_summary.get("walletCurrentCapital", live_summary.get("currentCapital")),
                            strategy_logical_pnl_usd,
                        )
                        external_cash_adjustment = float(cash_sync.get("externalCashAdjustmentUsd") or 0.0)
                        display_reference_capital = float(cash_sync.get("displayReferenceCapitalUsd") or initial_capital or 0.0)
                        external_cash_source = cash_sync.get("source")
                        current_for_display = _summary_number(live_summary, "currentCapital", None, float)
                        if current_for_display is not None and display_reference_capital:
                            display_live_wallet_pnl_usd = round(float(current_for_display) - display_reference_capital, 2)
                    wins = _summary_number(live_summary, "wins", int(trade_stats.get("wins") or 0), int)
                    losses = _summary_number(live_summary, "losses", int(trade_stats.get("losses") or 0), int)
                    pending = _summary_number(live_summary, "pending", int(trade_stats.get("pending") or 0), int)
                    completed = wins + losses
                    stats.update(
                        {
                            "n_logical_orders": _summary_number(
                                live_summary,
                                "totalTrades",
                                _summary_number(live_summary, "totalLogicalTrades", int(trade_stats.get("n_logical_orders") or 0), int),
                                int,
                            ),
                            "n_executed": _summary_number(
                                live_summary,
                                "totalFills",
                                _summary_number(live_summary, "fillCount", int(trade_stats.get("n_executed") or 0), int),
                                int,
                            ),
                            "wins": wins,
                            "losses": losses,
                            "pending": pending,
                            "n_completed_markets": completed,
                            "win_rate": _summary_number(live_summary, "winRate", ((wins / completed * 100) if completed else 0.0), float),
                            "total_pnl": round(display_live_wallet_pnl_usd, 2),
                            "initialCapital": initial_capital,
                            "displayReferenceCapitalUsd": display_reference_capital,
                            "externalCashAdjustmentUsd": external_cash_adjustment,
                            "externalCashAdjustmentSource": external_cash_source,
                            "currentCapital": _summary_number(live_summary, "currentCapital", None, float),
                            "capitalChange": (
                                round(
                                    _summary_number(live_summary, "currentCapital", 0.0, float) - display_reference_capital,
                                    2,
                                )
                                if display_reference_capital is not None
                                else None
                            ),
                        }
                    )
                    truth_source = "live_summary"
            else:
                # Simulation rows must use the raw simulation ledger as the
                # source of truth for pending / win / loss / pnl. Summary and
                # lifecycle snapshots are useful for metadata, but stale
                # derived reports must not keep a finished order stuck at
                # "pending 1" in the main monitor.
                truth_source = "simulation_raw_ledger"
                initial_capital = None
                if summary_stats:
                    initial_capital = _float_or_none(summary_stats.get("initialCapital"))
                if initial_capital is None:
                    initial_capital = _float_or_none(runtime_row.get("initialCapital"))
                if initial_capital is not None:
                    stats["initialCapital"] = initial_capital
                    stats["currentCapital"] = round(initial_capital + float(trade_stats.get("total_pnl") or 0.0), 2)
                    stats["capitalChange"] = round(float(trade_stats.get("total_pnl") or 0.0), 2)
                strategy_logical_pnl_usd = round(float(trade_stats.get("total_pnl") or 0.0), 2)
                display_live_wallet_pnl_usd = strategy_logical_pnl_usd

            raw_ledger_mtime = None
            summary_mtime = None
            state_mtime = None
            try:
                if prediction_trades_path:
                    raw_ledger_mtime = float(Path(prediction_trades_path).stat().st_mtime)
            except Exception:
                raw_ledger_mtime = None
            try:
                if summary_path:
                    summary_mtime = float(Path(str(summary_path)).stat().st_mtime)
            except Exception:
                summary_mtime = None
            try:
                if state_path and state_path.exists():
                    state_mtime = float(state_path.stat().st_mtime)
            except Exception:
                state_mtime = None
            decision_funnel_path = None
            execution_events_path = None
            decision_funnel_mtime = None
            execution_events_mtime = None
            try:
                if runtime_dir:
                    decision_funnel_path = Path(runtime_dir) / "decision_funnel.jsonl"
                    execution_events_path = Path(runtime_dir) / f"execution_v2_events.{row_mode}.jsonl"
                    if decision_funnel_path.exists():
                        decision_funnel_mtime = float(decision_funnel_path.stat().st_mtime)
                    if execution_events_path.exists():
                        execution_events_mtime = float(execution_events_path.stat().st_mtime)
            except Exception:
                decision_funnel_path = None
                execution_events_path = None
                decision_funnel_mtime = None
                execution_events_mtime = None
            stale_due_to_missing_runtime_activity = bool(
                row_mode == "simulation"
                and raw_ledger_mtime is not None
                and raw_ledger_mtime < (time.time() - 6 * 3600)
                and decision_funnel_mtime is None
                and execution_events_mtime is None
            )
            summary_fresh_but_raw_stale = bool(
                row_mode == "simulation"
                and raw_ledger_mtime is not None
                and summary_mtime is not None
                and summary_mtime > raw_ledger_mtime + 300.0
            )
            stale_raw_ledger = bool(
                summary_fresh_but_raw_stale
                or stale_due_to_missing_runtime_activity
            )
            prediction_source_status = str(runtime_row.get("predictionSourceStatus") or "").strip()
            if prediction_source_status == "non_runnable_missing_prediction_source":
                stale_raw_ledger = False
            last_decision_status = str(state_payload.get("lastDecisionStatus") or "").strip()
            last_decision_reason = str(state_payload.get("lastDecisionReason") or "").strip()
            if prediction_source_status == "non_runnable_missing_prediction_source":
                last_decision_status = "non_runnable_missing_prediction_source"
                last_decision_reason = str(runtime_row.get("nonRunnableReason") or "missing_prediction_source").strip()
            elif bool(runtime_row.get("runnerManaged")) and not last_decision_status:
                last_decision_status = "runner_managed_waiting_next_market"
                last_decision_reason = str(runtime_row.get("simulationSource") or "forward_runner_only").strip()

            rows.append(
                {
                    "profile": profile,
                    "trader": trader_name,
                    "role": str(runtime_row.get("role") or "mainline"),
                    "runtime_dir": runtime_dir,
                    "base_dir": _base_of(runtime_dir),
                    "short": _with_symbol_suffix(short_base, sym),
                    "displayKey": _with_symbol_suffix(short_base, sym),
                    "mode": row_mode,
                    "_aggregate_runtime_dirs": dirs,
                    "_config_set": row_config_set,
                    "truthSource": truth_source,
                    "summaryPath": summary_path,
                    "modeSummaryPath": summary_bundle.get("modeSummaryPath"),
                    "compatSummaryPath": summary_bundle.get("compatSummaryPath"),
                    "compatSummaryStale": bool(summary_bundle.get("compatSummaryStale")),
                    "compatSummarySig": summary_bundle.get("compatSummarySig"),
                    "modeSummarySig": summary_bundle.get("modeSummarySig"),
                    "predictionTradesPath": prediction_trades_path,
                    "summaryPresent": summary_present,
                    "tradesPresent": bool(prediction_trades_path),
                    "summaryTrades": int(summary_stats.get("summaryTrades") or 0) if summary_stats else None,
                    "tradesExecuted": int(trade_stats.get("n_executed") or 0),
                    "rawLedgerMtime": raw_ledger_mtime,
                    "summaryMtime": summary_mtime,
                    "stateMtime": state_mtime,
                    "decisionFunnelMtime": decision_funnel_mtime,
                    "executionEventsMtime": execution_events_mtime,
                    "staleRawLedger": stale_raw_ledger,
                    "summaryFreshButRawStale": summary_fresh_but_raw_stale,
                    "runnerManaged": bool(runtime_row.get("runnerManaged")),
                    "simulationSource": runtime_row.get("simulationSource"),
                    "predictionSourceStatus": prediction_source_status or None,
                    "nonRunnableReason": runtime_row.get("nonRunnableReason"),
                    "lastDecisionStatus": last_decision_status or None,
                    "lastDecisionReason": last_decision_reason or None,
                    "lastDecisionAt": str(state_payload.get("lastDecisionIso") or state_payload.get("lastCycleIso") or "").strip() or None,
                    "summaryPnL": round(float(summary_stats.get("summaryPnL") or 0.0), 2) if summary_stats else None,
                    "computedPnL": round(float(trade_stats.get("total_pnl") or 0.0), 2),
                    "resultPnlMismatchCount": int(stats.get("resultPnlMismatchCount") or 0),
                    "resultPnlMismatchExamples": list(stats.get("resultPnlMismatchExamples") or []),
                    "n_executed": int(stats.get("n_executed") or 0),
                    "n_pending_fills": int(stats.get("n_pending_fills") or 0),
                    "n_logical_orders": int(stats.get("n_logical_orders") or 0),
                    "n_markets": int(stats.get("n_markets") or stats.get("n_logical_orders") or 0),
                    "n_completed_markets": int(stats.get("n_completed_markets") or 0),
                    "settledTradeCount": int(stats.get("settled_trades") or stats.get("n_completed_markets") or 0),
                    "settledWinRate": float(stats.get("settled_win_rate") or stats.get("win_rate") or 0.0),
                    "wins": int(stats.get("wins") or 0),
                    "losses": int(stats.get("losses") or 0),
                    "pending": int(stats.get("pending") or 0),
                    "clearedNoFillCount": int(stats.get("cleared_no_fill") or 0),
                    "win_rate": float(stats.get("win_rate") or 0.0),
                    "total_pnl": round(float(stats.get("total_pnl") or 0.0), 2),
                    "strategy_logical_pnl_usd": round(strategy_logical_pnl_usd, 2),
                    "display_live_wallet_pnl_usd": round(display_live_wallet_pnl_usd, 2),
                    "initialCapital": stats.get("initialCapital"),
                    "currentCapital": stats.get("currentCapital"),
                    "capitalChange": stats.get("capitalChange"),
                    "latest_ts": stats.get("latest_ts"),
                    "latestExecutedAt": _iso_from_timestamp(stats.get("latest_ts")),
                    "latestSettledTradeAt": _iso_from_timestamp(stats.get("latest_settled_ts")),
                    "latestSettledTradeDisplay": str(stats.get("latest_settled_str") or "-"),
                    "latestDisplayedTradeAt": str(stats.get("latest_settled_str") or stats.get("latest_str") or "-"),
                    "latestTradeSource": (
                        "live_rollout"
                        if row_mode == "live"
                        else "trade_ledger"
                    ),
                    "latest_str": str(stats.get("latest_str") or "-"),
                    "latest_settled_str": str(stats.get("latest_settled_str") or "-"),
                    "latestPendingMarketSlug": stats.get("latest_pending_market_slug"),
                    "latestPendingDirection": stats.get("latest_pending_direction"),
                    "latestPendingTargetPeriodEndTs": stats.get("latest_pending_target_period_end_ts"),
                    "latestPendingPostedUsd": stats.get("latest_pending_posted_usd"),
                    "latestPendingFilledUsd": stats.get("latest_pending_filled_usd"),
                    "symbol": sym,
                    "familyRuleVersion": runtime_row.get("familyRuleVersion"),
                    "staleOrderPolicy": runtime_row.get("staleOrderPolicy"),
                    "forcedLowpriceMaterialization": runtime_row.get("forcedLowpriceMaterialization"),
                    "forcedLowpriceReason": runtime_row.get("forcedLowpriceReason"),
                    "forcedLowpriceSelectionMethod": runtime_row.get("forcedLowpriceSelectionMethod"),
                    "predictionSurfaceChoice": runtime_row.get("predictionSurfaceChoice"),
                    "scientificEligibilityStatus": runtime_row.get("scientificEligibilityStatus"),
                    "earlyAdmissionClass": runtime_row.get("earlyAdmissionClass"),
                    "earlyAdmissionReason": runtime_row.get("earlyAdmissionReason"),
                    "jointWinnerParamSource": runtime_row.get("jointWinnerParamSource"),
                    "historyResetMode": runtime_row.get("historyResetMode"),
                    "coverage365Pct": runtime_row.get("coverage365Pct"),
                    "coverage540Pct": runtime_row.get("coverage540Pct"),
                    "_bypass_lowprice_orphan_filter": bool(runtime_row.get("_bypass_lowprice_orphan_filter")),
                    "lowPriceSelectionMode": runtime_row.get("lowPriceSelectionMode"),
                    "lowPriceSelectedBuyPrice": runtime_row.get("lowPriceSelectedBuyPrice"),
                    "lowPriceSelectedBuyPriceRange": runtime_row.get("lowPriceSelectedBuyPriceRange"),
                    "lowPriceFinalistKey": runtime_row.get("lowPriceFinalistKey"),
                    "lowpriceFamilyId": runtime_row.get("lowpriceFamilyId"),
                    "lowpriceCutoverStatus": runtime_row.get("lowpriceCutoverStatus"),
                    "liveConsumptionStatus": runtime_row.get("liveConsumptionStatus"),
                }
            )
    return rows


def _zero_cell_stats_from_config(cfg_row: Dict[str, Any], profile: str, config_set: str = "main") -> List[Dict[str, Any]]:
    symbols = _cfg_allowed_symbols_for_active_row(cfg_row, profile, config_set=config_set)
    if not symbols:
        return []
    base_dir = str(cfg_row.get("logsDir") or "").strip()
    row_config_set = _config_set_for_runtime_value(profile, base_dir, default_set=config_set)
    base_label = _label_from_row(cfg_row, base_dir, profile, config_set=row_config_set)
    return [
        {
            "profile": profile,
            "trader": str(cfg_row.get("name") or "").strip(),
            "role": str(cfg_row.get("role") or "mainline"),
            "runtime_dir": base_dir,
            "base_dir": _base_of(base_dir),
            "short": _with_symbol_suffix(base_label, sym),
            "displayKey": _with_symbol_suffix(base_label, sym),
            "mode": "config_only_zero_state",
            "_aggregate_runtime_dirs": [],
            "_config_set": row_config_set,
            "truthSource": "config_only_zero_state",
            "summaryPath": None,
            "predictionTradesPath": None,
            "summaryPresent": False,
            "tradesPresent": False,
            "summaryTrades": None,
            "tradesExecuted": 0,
            "summaryPnL": None,
            "computedPnL": 0.0,
            "n_executed": 0,
            "n_pending_fills": 0,
            "n_logical_orders": 0,
            "n_markets": 0,
            "n_completed_markets": 0,
            "settledTradeCount": 0,
            "settledWinRate": 0.0,
            "wins": 0,
            "losses": 0,
            "pending": 0,
            "clearedNoFillCount": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "strategy_logical_pnl_usd": 0.0,
            "display_live_wallet_pnl_usd": 0.0,
            "initialCapital": cfg_row.get("initialCapital"),
            "currentCapital": cfg_row.get("initialCapital"),
            "capitalChange": 0.0 if cfg_row.get("initialCapital") is not None else None,
            "latest_ts": None,
            "latestExecutedAt": None,
            "latestSettledTradeAt": None,
            "latestSettledTradeDisplay": "-",
            "latestDisplayedTradeAt": "-",
            "latestTradeSource": "none",
            "latest_str": "-",
            "latest_settled_str": "-",
            "latestPendingMarketSlug": None,
            "latestPendingDirection": None,
            "latestPendingTargetPeriodEndTs": None,
            "latestPendingPostedUsd": 0.0,
            "latestPendingFilledUsd": 0.0,
            "symbol": sym,
            "familyRuleVersion": cfg_row.get("familyRuleVersion"),
            "staleOrderPolicy": cfg_row.get("staleOrderPolicy"),
            "forcedLowpriceMaterialization": cfg_row.get("forcedLowpriceMaterialization"),
            "forcedLowpriceReason": cfg_row.get("forcedLowpriceReason"),
            "forcedLowpriceSelectionMethod": cfg_row.get("forcedLowpriceSelectionMethod"),
            "predictionSurfaceChoice": cfg_row.get("predictionSurfaceChoice"),
            "scientificEligibilityStatus": cfg_row.get("scientificEligibilityStatus"),
            "earlyAdmissionClass": cfg_row.get("earlyAdmissionClass"),
            "earlyAdmissionReason": cfg_row.get("earlyAdmissionReason"),
            "jointWinnerParamSource": cfg_row.get("jointWinnerParamSource"),
            "historyResetMode": cfg_row.get("historyResetMode"),
            "coverage365Pct": cfg_row.get("coverage365Pct"),
            "coverage540Pct": cfg_row.get("coverage540Pct"),
            "_bypass_lowprice_orphan_filter": bool(cfg_row.get("_bypass_lowprice_orphan_filter")),
            "thresholdSimulation": cfg_row.get("thresholdSimulation"),
            "thresholdSimulationSourceTrader": cfg_row.get("thresholdSimulationSourceTrader"),
            "lowPriceSelectionMode": cfg_row.get("lowPriceSelectionMode"),
            "lowPriceSelectedBuyPrice": cfg_row.get("lowPriceSelectedBuyPrice"),
            "lowPriceSelectedBuyPriceRange": cfg_row.get("lowPriceSelectedBuyPriceRange"),
            "lowPriceFinalistKey": cfg_row.get("lowPriceFinalistKey"),
            "lowpriceFamilyId": cfg_row.get("lowpriceFamilyId"),
            "lowpriceCutoverStatus": cfg_row.get("lowpriceCutoverStatus"),
            "liveConsumptionStatus": cfg_row.get("liveConsumptionStatus"),
        }
        for sym in symbols
    ]


def _cfg_allowed_symbols(cfg_row: Dict[str, Any]) -> List[str]:
    allowed = cfg_row.get("allowedMarkets")
    if isinstance(allowed, list):
        out = [str(x).strip().upper() for x in allowed if str(x).strip()]
    elif isinstance(allowed, str):
        out = [x.strip().upper() for x in allowed.split(",") if x.strip()]
    else:
        out = []
    out = [sym for sym in out if sym in SUPPORTED_MONITOR_SYMBOLS]
    return out or list(SUPPORTED_MONITOR_SYMBOLS)


_LOWPRICE_EXPECTED_COMPARE_LABELS_CACHE: Dict[str, Set[str]] = {}
_LOWPRICE_EXPECTED_COMPARE_TRADERS_CACHE: Dict[str, Set[str]] = {}
_LAST_FILTERED_LOWPRICE_ORPHAN_ROWS = 0


def _is_early_admission_monitor_candidate(row: Dict[str, Any]) -> bool:
    return (
        bool(row.get("forcedLowpriceMaterialization"))
        and str(row.get("earlyAdmissionClass") or "").strip() == "early_admission_monitor_candidate"
        and str(row.get("predictionSurfaceChoice") or "").strip() == "use_base"
        and str(row.get("scientificEligibilityStatus") or "").strip() == "recommendation_only"
    )


def _early_admission_display_label(row: Dict[str, Any]) -> str | None:
    if not _is_early_admission_monitor_candidate(row):
        return None
    symbol = str(row.get("symbol") or "").strip().upper()
    profile = str(row.get("profile") or "default").strip()
    if symbol == "BTC" and profile == "70":
        return "70_joint_base_BTC"
    if symbol == "XRP" and profile == "default":
        return "joint_base_XRP"
    return None


def _load_early_admission_observation_rows() -> Dict[Tuple[str, str], Dict[str, Any]]:
    global _EARLY_ADMISSION_OBSERVATION_CACHE
    if _EARLY_ADMISSION_OBSERVATION_CACHE is not None:
        return dict(_EARLY_ADMISSION_OBSERVATION_CACHE)
    payload = _load_json(BTC_XRP_JOINT_WINNER_MONITOR_OBSERVATION_LATEST)
    rows = payload.get("rows") if isinstance(payload, dict) and isinstance(payload.get("rows"), list) else []
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = (
            str(row.get("profile") or "").strip(),
            str(row.get("trader") or "").strip(),
        )
        if key[0] and key[1]:
            out[key] = dict(row)
    _EARLY_ADMISSION_OBSERVATION_CACHE = dict(out)
    return out


def _attach_early_admission_observation(row: Dict[str, Any]) -> Dict[str, Any]:
    if not _is_early_admission_monitor_candidate(row):
        return row
    observed = _load_early_admission_observation_rows().get(
        (
            str(row.get("profile") or "").strip(),
            str(row.get("trader") or row.get("name") or "").strip(),
        ),
        {},
    )
    if not observed:
        return row
    merged = dict(row)
    for key in (
        "jointWinnerParamsApplied",
        "initialCapitalUsd",
        "currentStatus",
        "currentBlockingReason",
        "latestNoTradeReason",
        "realBlockingIssue",
        "displaySemanticsConsistent",
    ):
        if observed.get(key) is not None:
            merged[key] = observed.get(key)
    has_runtime_activity = bool(
        int(merged.get("n_logical_orders") or 0) > 0
        or int(merged.get("n_executed") or 0) > 0
        or str(merged.get("latestExecutedAt") or "").strip()
    )
    if has_runtime_activity:
        merged["currentStatus"] = "active_and_executing"
        merged["currentBlockingReason"] = None
        merged["latestNoTradeReason"] = ""
    return merged


def _prepare_all_orders_early_admission_row(row: Dict[str, Any]) -> Dict[str, Any]:
    prepared = _attach_early_admission_observation(dict(row))
    display_label = _early_admission_display_label(prepared)
    if display_label:
        prepared["short"] = display_label
        prepared["displayKey"] = display_label
    prepared["_include_in_all_orders"] = True
    prepared["_bypass_lowprice_orphan_filter"] = True
    return prepared


def _lowprice_compare_label_from_cfg_row(cfg_row: Dict[str, Any], profile: str, symbol: str) -> str:
    runtime_dir = str(cfg_row.get("logsDir") or "").strip()
    return f"{_label_from_row(cfg_row, runtime_dir, profile, config_set='lowprice')}-{symbol}"


def _expected_lowprice_compare_labels(profile: str) -> Set[str]:
    cached = _LOWPRICE_EXPECTED_COMPARE_LABELS_CACHE.get(profile)
    if cached is not None:
        return cached
    rows = _discover_config_rows(profile, active_only=True, config_set="lowprice")
    labels: Set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        for symbol in _cfg_allowed_symbols(row):
            labels.add(_lowprice_compare_label_from_cfg_row(row, profile, symbol))
    _LOWPRICE_EXPECTED_COMPARE_LABELS_CACHE[profile] = labels
    return labels


def _expected_lowprice_compare_traders(profile: str) -> Set[str]:
    cached = _LOWPRICE_EXPECTED_COMPARE_TRADERS_CACHE.get(profile)
    if cached is not None:
        return cached
    cfg_rows = _discover_config_rows(profile, active_only=True, config_set="lowprice")
    trader_names = {
        str(row.get("name") or "").strip()
        for row in cfg_rows
        if isinstance(row, dict) and str(row.get("name") or "").strip()
    }
    _LOWPRICE_EXPECTED_COMPARE_TRADERS_CACHE[profile] = trader_names
    return trader_names


def _is_lowprice_compare_snapshot_orphan(row: Dict[str, Any]) -> bool:
    if bool(row.get("_bypass_lowprice_orphan_filter")):
        return False
    if str(row.get("role") or "").strip().lower() != "compare_only":
        return False
    runtime_dir = str(row.get("runtime_dir") or row.get("base_dir") or "").strip()
    if not (
        runtime_dir.startswith("logs_lowprice_selected_")
        or runtime_dir.startswith("logs_70_lowprice_selected_")
        or runtime_dir.startswith("logs_threshold_pack_lowprice_selected_")
    ):
        return False
    profile = str(row.get("profile") or "default").strip() or "default"
    short = str(row.get("short") or "").strip()
    if not short:
        return False
    trader = str(row.get("trader") or row.get("name") or "").strip()
    if trader and trader not in _expected_lowprice_compare_traders(profile):
        return True
    return short not in _expected_lowprice_compare_labels(profile)


def _cell_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return (
        str(row.get("profile") or "default"),
        str(row.get("trader") or row.get("name") or "").strip(),
        str(row.get("symbol") or "").strip().upper(),
    )


def _canonical_cell_key(row: Dict[str, Any]) -> str:
    role = str(row.get("role") or "").strip().lower() or "mainline"
    profile = str(row.get("profile") or "default").strip()
    symbol = str(row.get("symbol") or "").strip().upper()
    trader = str(row.get("trader") or row.get("name") or "").strip()
    config_set = str(row.get("_config_set") or "main").strip().lower()
    row_kind = str(row.get("rowKind") or "").strip().lower()
    if row_kind == "live":
        return f"live|{profile}|{trader}|{symbol}"
    return f"sim|{profile}|{config_set}|{role}|{trader}|{symbol}"


def _row_kind(row: Dict[str, Any]) -> str:
    mode = str(row.get("mode") or "").strip().lower()
    if mode == "live" or str(row.get("role") or "").strip().lower() == "live_rollout":
        return "live"
    if mode == "config_only_zero_state":
        if _is_visible_zero_state_row(row):
            return "forced_zero_state"
        return "zero_state_hidden"
    return "runtime_backed_simulation"


def _row_priority(row: Dict[str, Any]) -> int:
    row_kind = str(row.get("rowKind") or "").strip().lower()
    if row_kind == "live":
        return 4
    if row_kind == "runtime_backed_simulation":
        return 3
    if row_kind == "forced_zero_state":
        return 2
    return 1


def _canonicalize_cell_rows(
    rows: List[Dict[str, Any]],
    *,
    include_hidden_rows: bool = False,
    show_zero_state: bool = False,
) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for raw_row in rows:
        row = dict(raw_row)
        row_kind = _row_kind(row)
        activity_state = _activity_state(row, row_kind)
        source_freshness_state, source_freshness_reason = _source_freshness_contract(row)
        row["rowKind"] = row_kind
        row["activityState"] = activity_state
        row["sourceFreshnessState"] = source_freshness_state
        row["sourceFreshnessReason"] = source_freshness_reason
        row["everExecuted"] = _ever_executed(row)
        row["zeroActivityReason"] = (
            source_freshness_reason
            if activity_state == "runtime_backed_zero_activity"
            else ""
        )
        row["canonicalCellKey"] = _canonical_cell_key(row)
        row["displayKey"] = str(row.get("displayKey") or row.get("short") or "").strip()
        visible = row_kind != "zero_state_hidden"
        visibility_reason = "live_visible"
        if row_kind == "runtime_backed_simulation" and activity_state == "runtime_backed_zero_activity":
            role = str(row.get("role") or "").strip().lower()
            compare_only = role == "compare_only"
            forced_zero_state_visible = _is_forced_lowprice_row(row)
            visible = bool(show_zero_state or not compare_only or forced_zero_state_visible)
            visibility_reason = (
                "forced_runtime_backed_zero_activity_visible"
                if visible and forced_zero_state_visible and compare_only
                else
                "runtime_backed_zero_activity_visible_by_flag"
                if visible and show_zero_state and compare_only
                else "runtime_backed_zero_activity_visible"
                if visible
                else "compare_only_runtime_backed_zero_activity_hidden"
            )
        elif row_kind == "runtime_backed_simulation":
            visibility_reason = "runtime_backed_simulation_visible"
        elif row_kind == "forced_zero_state":
            visibility_reason = "forced_zero_state_visible"
        elif row_kind == "zero_state_hidden":
            visibility_reason = "non_forced_compare_only_zero_state_hidden"
        row["visibility"] = visible
        if row_kind == "runtime_backed_simulation" and activity_state == "runtime_backed_active":
            row["visibilityReason"] = "runtime_backed_simulation_visible"
        else:
            row["visibilityReason"] = visibility_reason
        row["selectedForDisplay"] = False
        row["selectionReason"] = None
        row["hiddenReason"] = None
        buckets.setdefault(str(row["canonicalCellKey"]), []).append(row)

    selected_rows: List[Dict[str, Any]] = []
    all_rows: List[Dict[str, Any]] = []
    for _, bucket in buckets.items():
        ordered = sorted(
            bucket,
            key=lambda item: (
                _row_priority(item),
                int(item.get("n_executed") or 0),
                int(item.get("n_logical_orders") or 0),
                float(item.get("total_pnl") or 0.0),
                str(item.get("displayKey") or ""),
            ),
            reverse=True,
        )
        winner: Dict[str, Any] | None = None
        for candidate in ordered:
            if bool(candidate.get("visibility")):
                winner = candidate
                break
        for candidate in ordered:
            if winner is not None and candidate is winner:
                candidate["selectedForDisplay"] = True
                candidate["selectionReason"] = f"selected_highest_priority_{candidate['rowKind']}"
                candidate["hiddenReason"] = None
                selected_rows.append(candidate)
            else:
                candidate["selectedForDisplay"] = False
                if not bool(candidate.get("visibility")):
                    candidate["hiddenReason"] = str(candidate.get("visibilityReason") or "hidden")
                elif winner is not None:
                    candidate["hiddenReason"] = f"superseded_by_{winner['rowKind']}"
                else:
                    candidate["hiddenReason"] = "hidden"
            all_rows.append(candidate)
    if include_hidden_rows:
        return sorted(all_rows, key=_row_sort_key)
    return sorted(selected_rows, key=_row_sort_key)


def _row_mode_has_activity(row: Dict[str, Any]) -> bool:
    mode = str(row.get("mode") or "").strip().lower()
    if mode not in {"live", "backtest"}:
        return True
    base_dir = str(row.get("base_dir") or row.get("runtime_dir") or "").strip()
    if not base_dir:
        return False
    return bool(read_mode_activity(base_dir, mode).get("has_activity"))


def _synthesize_selected_live_rows(
    selected_targets: Set[Tuple[str, str, str]],
    existing_rows: List[Dict[str, Any]],
    cfg_lookup: Dict[Tuple[str, str, str], Tuple[Dict[str, Any], str]],
    profile_filter: str,
) -> List[Dict[str, Any]]:
    if not selected_targets:
        return []
    existing_live_keys = {
        _cell_key(row)
        for row in existing_rows
        if str(row.get("mode") or "").strip().lower() == "live"
    }
    synthesized: List[Dict[str, Any]] = []
    for key in sorted(selected_targets):
        if profile_filter != "all" and key[0] != profile_filter:
            continue
        if key in existing_live_keys:
            continue
        cfg_bundle = cfg_lookup.get(key)
        if not cfg_bundle:
            continue
        cfg_row, row_config_set = cfg_bundle
        runtime_row = dict(cfg_row)
        runtime_row["_runtime_dir"] = str(cfg_row.get("logsDir") or "")
        runtime_row["_aggregate_runtime_dirs"] = [str(cfg_row.get("logsDir") or "")]
        runtime_row["_config_set"] = row_config_set
        live_rows = [
            row
            for row in _load_cell_stats(runtime_row, profile=key[0], config_set=row_config_set)
            if str(row.get("mode") or "").strip().lower() == "live"
            and str(row.get("symbol") or "").strip().upper() == key[2]
        ]
        synthesized.extend(live_rows)
    return synthesized


def _filter_visible_rows(
    rows: List[Dict[str, Any]],
    *,
    mode_filter: str,
    view: str,
    selected_targets: Set[Tuple[str, str, str]],
) -> List[Dict[str, Any]]:
    if view != "cell":
        return rows
    filtered: List[Dict[str, Any]] = []
    for row in rows:
        mode = str(row.get("mode") or "").strip().lower()
        role = str(row.get("role") or "").strip().lower()
        row_config_set = str(row.get("_config_set") or "").strip().lower()
        key = _cell_key(row)
        if mode == "simulation" and key in selected_targets:
            continue
        if mode in {"live", "backtest"}:
            if mode == "live" and role == "compare_only":
                continue
            if mode == "live" and key in selected_targets:
                filtered.append(row)
                continue
            if _row_mode_has_activity(row):
                filtered.append(row)
            continue
        if mode_filter == "live" and mode == "config_only_zero_state":
            continue
        filtered.append(row)
    return filtered


def _collect_rows(
    profile: str,
    mode_filter: str,
    active_only: bool,
    view: str,
    scope: str,
    include_runtime_splits: bool,
    config_set: str = "main",
    monitor_set: str = "main",
    include_hidden_rows: bool = False,
    show_zero_state: bool = False,
) -> List[Dict[str, Any]]:
    global _LAST_FILTERED_LOWPRICE_ORPHAN_ROWS
    rows: List[Dict[str, Any]] = []
    cfg_lookup: Dict[Tuple[str, str, str], Tuple[Dict[str, Any], str]] = {}
    selected_targets = _selected_live_target_set()
    scope_cells = _load_scope_cells(scope)
    if profile == "all":
        if config_set == "lowprice" or monitor_set == "lowprice":
            profiles = ["default", "70", "threshold_pack"]
        else:
            profiles = ["default", "70"]
    else:
        profiles = [profile]
    for current_profile in profiles:
        if active_only:
            cfg_rows = _discover_config_rows(profile=current_profile, active_only=True, config_set=config_set)
            compare_cfg_rows = _load_monitor_only_rows(current_profile, monitor_set=monitor_set, active_only=True)
            all_orders_early_cfg_rows: List[Dict[str, Any]] = []
            if config_set == "main" and monitor_set == "main" and view == "cell":
                all_orders_early_cfg_rows = [
                    dict(row)
                    for row in _load_monitor_only_rows(current_profile, monitor_set="lowprice", active_only=True)
                    if isinstance(row, dict) and _is_early_admission_monitor_candidate(row)
                ]
            for cfg_row in cfg_rows:
                trader = str(cfg_row.get("name") or "").strip()
                for symbol in _cfg_allowed_symbols_for_active_row(cfg_row, current_profile, config_set=config_set):
                    cfg_lookup[(current_profile, trader, symbol)] = (cfg_row, config_set)
            for cfg_row in compare_cfg_rows:
                trader = str(cfg_row.get("name") or "").strip()
                row_config_set = str(cfg_row.get("_config_set") or (monitor_set if monitor_set == "lowprice" else config_set))
                for symbol in _cfg_allowed_symbols_for_active_row(cfg_row, current_profile, config_set=row_config_set):
                    cfg_lookup[(current_profile, trader, symbol)] = (cfg_row, row_config_set)
            for cfg_row in all_orders_early_cfg_rows:
                trader = str(cfg_row.get("name") or "").strip()
                for symbol in _cfg_allowed_symbols_for_active_row(cfg_row, current_profile, config_set="lowprice"):
                    cfg_lookup[(current_profile, trader, symbol)] = (cfg_row, "lowprice")
            # Low-price monitor uses dedicated compare-only config and manifest files
            # for the same trader population. Avoid double-counting the same rows by
            # treating the manifest entries as the single source of truth here.
            if config_set == "lowprice" and monitor_set == "lowprice":
                cfg_rows = []
            rt_rows = _runtime_rows(
                cfg_rows,
                profile=current_profile,
                mode_filter=mode_filter,
                include_runtime_splits=include_runtime_splits,
                prefer_fresh_runtime_dir=bool(scope == "core" and view == "cell" and not include_runtime_splits),
                collapse_runtime_splits=bool(view == "cell"),
                config_set=config_set,
            )
            compare_rt_rows = _runtime_rows(
                compare_cfg_rows,
                profile=current_profile,
                mode_filter=mode_filter,
                include_runtime_splits=include_runtime_splits,
                prefer_fresh_runtime_dir=False,
                collapse_runtime_splits=bool(view == "cell"),
                config_set=monitor_set if monitor_set == "lowprice" else config_set,
            )
            all_orders_early_rt_rows = _runtime_rows(
                all_orders_early_cfg_rows,
                profile=current_profile,
                mode_filter=mode_filter,
                include_runtime_splits=include_runtime_splits,
                prefer_fresh_runtime_dir=False,
                collapse_runtime_splits=True,
                config_set="lowprice",
            )
            if view == "cell":
                seen_keys: Set[Tuple[str, str, str]] = set()
                for rr in rt_rows:
                    trader = str(rr.get("name") or "").strip()
                    for cell in _load_cell_stats(rr, profile=current_profile, config_set=config_set):
                        symbol = str(cell.get("symbol") or "").upper()
                        if not _symbol_enabled_for_active_path(_cfg_paths(current_profile, config_set=config_set)[1], trader, symbol):
                            continue
                        if scope_cells and (current_profile, trader, symbol) not in scope_cells:
                            continue
                        seen_keys.add((current_profile, trader, symbol))
                        rows.append(cell)
                for cfg_row in cfg_rows:
                    trader = str(cfg_row.get("name") or "").strip()
                    for cell in _zero_cell_stats_from_config(cfg_row, profile=current_profile, config_set=config_set):
                        symbol = str(cell.get("symbol") or "").upper()
                        key = (current_profile, trader, symbol)
                        if key in seen_keys:
                            continue
                        if scope_cells and key not in scope_cells:
                            continue
                        rows.append(cell)
                compare_seen: Set[Tuple[str, str, str]] = set()
                for rr in compare_rt_rows:
                    trader = str(rr.get("name") or "").strip()
                    row_active_path = _cfg_paths(
                        current_profile,
                        config_set=monitor_set if monitor_set == "lowprice" else config_set,
                    )[1]
                    for cell in _load_cell_stats(
                        rr,
                        profile=current_profile,
                        config_set=monitor_set if monitor_set == "lowprice" else config_set,
                    ):
                        symbol = str(cell.get("symbol") or "").upper()
                        if not _symbol_enabled_for_active_path(row_active_path, trader, symbol):
                            continue
                        compare_seen.add((current_profile, trader, symbol))
                        rows.append(cell)
                for cfg_row in compare_cfg_rows:
                    trader = str(cfg_row.get("name") or "").strip()
                    row_config_set = str(cfg_row.get("_config_set") or (monitor_set if monitor_set == "lowprice" else config_set))
                    for cell in _zero_cell_stats_from_config(
                        cfg_row,
                        profile=current_profile,
                        config_set=row_config_set,
                    ):
                        symbol = str(cell.get("symbol") or "").upper()
                        key = (current_profile, trader, symbol)
                        if key in compare_seen:
                            continue
                        rows.append(cell)
                early_seen: Set[Tuple[str, str, str]] = set()
                for rr in all_orders_early_rt_rows:
                    trader = str(rr.get("name") or "").strip()
                    row_active_path = _cfg_paths(current_profile, config_set="lowprice")[1]
                    for cell in _load_cell_stats(
                        rr,
                        profile=current_profile,
                        config_set="lowprice",
                    ):
                        symbol = str(cell.get("symbol") or "").upper()
                        if not _symbol_enabled_for_active_path(row_active_path, trader, symbol):
                            continue
                        early_seen.add((current_profile, trader, symbol))
                        rows.append(_prepare_all_orders_early_admission_row(cell))
                for cfg_row in all_orders_early_cfg_rows:
                    trader = str(cfg_row.get("name") or "").strip()
                    for cell in _zero_cell_stats_from_config(
                        cfg_row,
                        profile=current_profile,
                        config_set="lowprice",
                    ):
                        symbol = str(cell.get("symbol") or "").upper()
                        key = (current_profile, trader, symbol)
                        if key in early_seen:
                            continue
                        rows.append(_prepare_all_orders_early_admission_row(cell))
            else:
                for rr in rt_rows:
                    runtime_dir = str(rr.get("_runtime_dir") or "").strip()
                    if not runtime_dir:
                        continue
                    aggregate_dirs = rr.get("_aggregate_runtime_dirs")
                    rows.append(
                        _load_model_stats(
                            runtime_dir,
                            profile=current_profile,
                            aggregate_dirs=aggregate_dirs if isinstance(aggregate_dirs, list) else None,
                            config_set=config_set,
                        )
                    )
                for rr in compare_rt_rows:
                    runtime_dir = str(rr.get("_runtime_dir") or "").strip()
                    if not runtime_dir:
                        continue
                    aggregate_dirs = rr.get("_aggregate_runtime_dirs")
                    rows.append(
                        _load_model_stats(
                            runtime_dir,
                            profile=current_profile,
                            aggregate_dirs=aggregate_dirs if isinstance(aggregate_dirs, list) else None,
                            config_set=monitor_set if monitor_set == "lowprice" else config_set,
                        )
                    )
        else:
            dirs = _all_runtime_dirs(
                profile=current_profile,
                mode_filter=mode_filter,
                include_runtime_splits=include_runtime_splits,
                config_set=config_set,
            )
            if view == "cell":
                for d in dirs:
                    rows.extend(
                        _load_cell_stats(
                            {"_runtime_dir": d, "logsDir": _base_of(d), "name": _short_name(d)},
                            profile=current_profile,
                            config_set=config_set,
                        )
                    )
            else:
                rows.extend(_load_model_stats(d, profile=current_profile, config_set=config_set) for d in dirs)
    rows.extend(
        _synthesize_selected_live_rows(
            selected_targets,
            rows,
            cfg_lookup,
            profile_filter=profile,
        )
    )
    rows = _filter_visible_rows(
        rows,
        mode_filter=mode_filter,
        view=view,
        selected_targets=selected_targets,
    )
    if view == "cell":
        orphan_filtered = [
            row for row in rows
            if _is_lowprice_compare_snapshot_orphan(row)
        ]
        _LAST_FILTERED_LOWPRICE_ORPHAN_ROWS = len(orphan_filtered)
        if orphan_filtered:
            rows = [row for row in rows if not _is_lowprice_compare_snapshot_orphan(row)]
        rows = _canonicalize_cell_rows(
            rows,
            include_hidden_rows=include_hidden_rows,
            show_zero_state=show_zero_state,
        )
    else:
        _LAST_FILTERED_LOWPRICE_ORPHAN_ROWS = 0
        rows.sort(key=lambda x: x["total_pnl"], reverse=True)
    return rows


def _visible_live_rollout_rows(profile: str) -> List[Dict[str, Any]]:
    # Monitors should consume the current truth surface, not become a hidden writer.
    evidence_map = build_live_rollout_evidence_map(force_refresh=False)
    truth_rows = _live_truth_contract_rows()
    rows: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str]] = set()
    for (row_profile, _, _), row in sorted(evidence_map.items()):
        if profile != "all" and row_profile != profile:
            continue
        target = row.get("target_cell") if isinstance(row.get("target_cell"), dict) else {}
        trader = str(target.get("trader") or row.get("target_source_trader") or "").strip()
        symbol = str(target.get("symbol") or "").strip().upper()
        if trader and symbol:
            seen.add((row_profile, trader, symbol))
        rows.append(row)
    for (row_profile, trader, symbol, _wallet_slot), row in sorted(truth_rows.items()):
        if profile != "all" and row_profile != profile:
            continue
        compact_key = (row_profile, trader, symbol)
        if compact_key in seen:
            continue
        fallback = dict(row)
        fallback["_liveRolloutFallback"] = "live_truth_contract_missing_evidence_row"
        rows.append(fallback)
    return rows


def _live_rollout_rows_for_table(profile: str, existing_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    existing_labels = {
        str(row.get("short") or "").strip()
        for row in existing_rows
        if str(row.get("short") or "").strip()
        and str(row.get("mode") or "").strip().lower() != "live"
        and str(row.get("role") or "").strip().lower() != "live_rollout"
    }
    truth_rows = _live_truth_contract_rows()
    current_truth_rows = _dual_eth_live_current_truth_map()
    rows: List[Dict[str, Any]] = []
    for rollout in _visible_live_rollout_rows(profile):
        target = rollout.get("target_cell") if isinstance(rollout.get("target_cell"), dict) else {}
        target_profile = str(target.get("profile") or profile).strip()
        target_trader = str(rollout.get("target_source_trader") or target.get("trader") or "").strip()
        target_symbol = str(target.get("symbol") or "").strip().upper()
        wallet_slot = str(target.get("walletSlot") or "").strip()
        runtime_dir = str(rollout.get("logs_dir") or "").strip()
        truth_row = truth_rows.get((target_profile, target_trader, target_symbol, wallet_slot)) or {}
        current_truth = current_truth_rows.get(target_trader, {})
        selection_contract = rollout.get("selection_contract") if isinstance(rollout.get("selection_contract"), dict) else {}
        selection_display_key = str(selection_contract.get("displayKey") or selection_contract.get("target") or "").strip()
        if target_trader in _FRIENDLY_LIVE_LABELS:
            cell_label = _FRIENDLY_LIVE_LABELS[target_trader]
        elif wallet_slot == "slot02" and target_trader == "v5_exp10_bp0480" and target_symbol == "ETH":
            cell_label = f"slot2/slot02 live {selection_display_key or 'exp10_bp0480-ETH'}"
        elif selection_display_key:
            cell_label = f"live_{selection_display_key}"
        else:
            cell_label = _independent_live_label(target_profile, target_trader, target_symbol)
        if not cell_label or cell_label in existing_labels:
            continue
        slot01_canonical: Dict[str, Any] = {}
        if target_trader == "v5_exp10_bp_dyn_0300_0440_eth":
            try:
                _refresh_slot01_canonical_trade_ledger_if_needed(timeout=30)
            except Exception:
                pass
            slot01_canonical = _slot01_canonical_summary()
        canonical_active = target_trader == "v5_exp10_bp_dyn_0300_0440_eth" and bool(slot01_canonical)
        wins = int(
            _first_present(
                slot01_canonical.get("wins") if canonical_active else None,
                current_truth.get("wins"),
                truth_row.get("wins"),
                rollout.get("wins"),
                0,
            )
        )
        losses = int(
            _first_present(
                slot01_canonical.get("losses") if canonical_active else None,
                current_truth.get("losses"),
                truth_row.get("losses"),
                rollout.get("losses"),
                0,
            )
        )
        pending = int(
            _first_present(
                slot01_canonical.get("pending") if canonical_active else None,
                current_truth.get("pending"),
                current_truth.get("pendingLogicalOrders"),
                rollout.get("pending_logical_orders"),
                rollout.get("pending"),
                truth_row.get("pending"),
                0,
            )
        )
        completed = wins + losses
        live_trade_rows_for_latest = _read_trades(runtime_dir, "live") if runtime_dir else []
        if target_symbol:
            live_trade_rows_for_latest = [
                row
                for row in live_trade_rows_for_latest
                if str(row.get("symbol") or "").strip().upper() == target_symbol
            ]
        live_trade_stats_for_latest = _trade_stats_for_rows(live_trade_rows_for_latest) if live_trade_rows_for_latest else {}
        latest_ts = None
        latest_trade_source = "live_rollout_evidence"
        source_watermark = rollout.get("source_watermark") if isinstance(rollout.get("source_watermark"), dict) else {}
        if not source_watermark and isinstance(truth_row.get("source_watermark"), dict):
            source_watermark = truth_row.get("source_watermark")
        latest_candidates: List[Tuple[str, Any]] = [
            ("slot01_canonical", slot01_canonical.get("latestAt")),
            ("current_truth_settled", current_truth.get("latestSettledTradeAt")),
            ("current_truth_executed", current_truth.get("latestExecutedAt")),
            ("public_trade", source_watermark.get("latest_public_trade_at")),
            ("rollout_public_trade", rollout.get("last_live_trade_at")),
            ("rollout_attempt", rollout.get("last_order_attempt_at")),
            ("truth_public_trade", truth_row.get("last_live_trade_at")),
            ("order_attempt", truth_row.get("last_order_attempt_at")),
            ("local_prediction_trades_executed", _iso_from_timestamp(live_trade_stats_for_latest.get("latest_ts"))),
            ("truth_generated", truth_row.get("generated_at")),
            ("rollout_generated", rollout.get("generated_at")),
        ]
        # Prefer the freshest observable activity across the rollout/truth surfaces.
        # Older canonical summaries can lag official/order-ledger reconciliation after
        # V2 fills, so taking the first parseable source makes the live table look dead.
        for candidate_source, candidate in latest_candidates:
            if not isinstance(candidate, str) or not candidate:
                continue
            try:
                candidate_ts = datetime.fromisoformat(candidate.replace("Z", "+00:00")).timestamp()
            except Exception:
                continue
            if latest_ts is None or candidate_ts > latest_ts:
                latest_ts = candidate_ts
                latest_trade_source = candidate_source
        external_trade_count = int(rollout.get("external_trade_count") or 0)
        logical_total = int(
            _first_present(
                slot01_canonical.get("totalLogicalTrades") if canonical_active else None,
                slot01_canonical.get("totalTrades") if canonical_active else None,
                current_truth.get("totalLogicalTrades"),
                truth_row.get("total_logical_trades"),
                rollout.get("total_logical_trades"),
                rollout.get("settled_logical_orders"),
                wins + losses + pending,
                0,
            )
        )
        canonical_fills = int(
            _first_present(
                slot01_canonical.get("totalFills") if canonical_active else None,
                slot01_canonical.get("fillCount") if canonical_active else None,
                0,
            )
        )
        total_fills = (
            canonical_fills
            if canonical_active
            else max(
                int(rollout.get("total_fills") or 0),
                int(rollout.get("filled_rungs") or 0),
                int(truth_row.get("total_fills") or 0),
                external_trade_count,
            )
        )
        display_pnl = float(
            _first_present(
                slot01_canonical.get("displayLiveWalletPnlUsd") if canonical_active else None,
                current_truth.get("displayLiveWalletPnlUsd"),
                truth_row.get("display_live_wallet_pnl_usd"),
                rollout.get("display_live_wallet_pnl_usd"),
                0.0,
            )
        )
        strategy_pnl = float(
            _first_present(
                slot01_canonical.get("strategyLogicalPnlUsd") if canonical_active else None,
                current_truth.get("strategyLogicalPnlUsd"),
                truth_row.get("strategy_logical_pnl_usd"),
                rollout.get("strategy_logical_pnl_usd"),
                display_pnl,
            )
        )
        display_reference_capital = None
        external_cash_adjustment = 0.0
        external_cash_source = None
        external_cash_reason = None
        if target_trader == "v5_exp10_bp_dyn_0300_0440_eth":
            selected_reference_capital = _float_or_none(selection_contract.get("displayReferenceCapitalUsd"))
            current_capital_for_display = (
                slot01_canonical.get("walletCurrentCapital")
                if canonical_active and slot01_canonical.get("walletCurrentCapital") is not None
                else current_truth.get("officialWalletPortfolioCapitalUsd")
                if current_truth.get("officialWalletPortfolioCapitalUsd") is not None
                else truth_row.get("official_wallet_portfolio_capital_usd")
                if truth_row.get("official_wallet_portfolio_capital_usd") is not None
                else rollout.get("official_portfolio_capital_usd")
                if rollout.get("official_portfolio_capital_usd") is not None
                else None
            )
            if selected_reference_capital is not None:
                display_reference_capital = selected_reference_capital
                external_cash_adjustment = 0.0
                external_cash_source = "selection_contract_display_reference_capital"
                external_cash_reason = selection_contract.get("capitalBaselineReason")
            else:
                cash_sync = _slot01_external_cash_adjustment(current_capital_for_display, strategy_pnl)
                external_cash_adjustment = float(cash_sync.get("externalCashAdjustmentUsd") or 0.0)
                display_reference_capital = float(cash_sync.get("displayReferenceCapitalUsd") or 1600.0)
                external_cash_source = cash_sync.get("source")
                external_cash_reason = cash_sync.get("capitalBaselineReason")
            current_capital_float = _float_or_none(current_capital_for_display)
            if current_capital_float is not None:
                display_pnl = round(current_capital_float - display_reference_capital, 2)
        else:
            selected_reference_capital = _float_or_none(selection_contract.get("displayReferenceCapitalUsd"))
            current_capital_for_display = (
                current_truth.get("officialWalletPortfolioCapitalUsd")
                if current_truth.get("officialWalletPortfolioCapitalUsd") is not None
                else truth_row.get("official_wallet_portfolio_capital_usd")
                if truth_row.get("official_wallet_portfolio_capital_usd") is not None
                else rollout.get("official_portfolio_capital_usd")
            )
            current_capital_float = _float_or_none(current_capital_for_display)
            if selected_reference_capital is not None:
                display_reference_capital = selected_reference_capital
                external_cash_source = "selection_contract_display_reference_capital"
                external_cash_reason = selection_contract.get("capitalBaselineReason")
                if current_capital_float is not None:
                    display_pnl = round(current_capital_float - display_reference_capital, 2)
        live_trade_rows_for_result_check = _read_trades(runtime_dir, "live") if runtime_dir else []
        if target_symbol:
            live_trade_rows_for_result_check = [
                row
                for row in live_trade_rows_for_result_check
                if str(row.get("symbol") or "").strip().upper() == target_symbol
            ]
        result_pnl_mismatch = _trade_stats_for_rows(live_trade_rows_for_result_check)
        rows.append(
            {
                "profile": target_profile,
                "trader": target_trader,
                "role": "live_rollout",
                "runtime_dir": runtime_dir,
                "base_dir": _base_of(runtime_dir) if runtime_dir else "",
                "short": cell_label,
                "mode": "live",
                "_pinned_live_rollout": True,
                "n_executed": total_fills,
                "n_pending_fills": int(rollout.get("pending_rungs") or 0),
                "n_logical_orders": logical_total,
                "n_markets": logical_total,
                "n_completed_markets": completed,
                "wins": wins,
                "losses": losses,
                "pending": pending,
                "win_rate": (wins / completed * 100) if completed else 0.0,
                "resultPnlMismatchCount": int(result_pnl_mismatch.get("resultPnlMismatchCount") or 0),
                "resultPnlMismatchExamples": list(result_pnl_mismatch.get("resultPnlMismatchExamples") or []),
                "total_pnl": round(display_pnl, 2),
                "strategy_logical_pnl_usd": round(strategy_pnl, 2),
                "display_live_wallet_pnl_usd": round(display_pnl, 2),
                "displayPnlUsd": round(display_pnl, 2),
                "displayPnlBasis": "official_api_portfolio_minus_live_baseline",
                "displayReferenceCapitalUsd": display_reference_capital,
                "externalCashAdjustmentUsd": external_cash_adjustment,
                "externalCashAdjustmentSource": external_cash_source,
                "capitalBaselineReason": external_cash_reason,
                "latest_ts": latest_ts,
                "latestExecutedAt": _iso_from_timestamp(latest_ts),
                "latestDisplayedTradeAt": datetime.fromtimestamp(latest_ts).strftime("%m-%d %H:%M") if latest_ts else "-",
                "latestTradeSource": latest_trade_source,
                "latest_str": datetime.fromtimestamp(latest_ts).strftime("%m-%d %H:%M") if latest_ts else "-",
                "symbol": target_symbol,
                "slot": wallet_slot,
                "currentWalletSlot": wallet_slot,
                "activeCapUsd": _float_or_none(
                    current_truth.get("activeCapUsd")
                    if current_truth.get("activeCapUsd") is not None
                    else truth_row.get("active_cap_usd")
                    if truth_row.get("active_cap_usd") is not None
                    else rollout.get("active_cap_usd")
                    if rollout.get("active_cap_usd") is not None
                    else selection_contract.get("betTargetCapUsd")
                ),
                "minTotalUsd": _float_or_none(
                    current_truth.get("minTotalUsd")
                    if current_truth.get("minTotalUsd") is not None
                    else truth_row.get("min_total_usd")
                    if truth_row.get("min_total_usd") is not None
                    else rollout.get("min_total_usd")
                    if rollout.get("min_total_usd") is not None
                    else selection_contract.get("minTotalUsd")
                ),
                "runtimePredictionPath": str(
                    rollout.get("runtime_prediction_path")
                    or truth_row.get("runtime_prediction_path")
                    or selection_contract.get("runtimePredictionPath")
                    or selection_contract.get("predictionPath")
                    or ""
                ).strip() or None,
                "runtimeRulesPath": (
                    rollout.get("runtime_rules_path")
                    if rollout.get("runtime_rules_path") is not None
                    else truth_row.get("runtime_rules_path")
                    if truth_row.get("runtime_rules_path") is not None
                    else selection_contract.get("runtimeRulesPath")
                    or selection_contract.get("rulesJsonPath")
                ),
                "runtimeValuesMatchCandidate": rollout.get("runtime_values_match_candidate")
                if rollout.get("runtime_values_match_candidate") is not None
                else truth_row.get("runtime_values_match_candidate"),
                "runtimeTruthLedgerFresh": rollout.get("runtime_truth_ledger_fresh")
                if rollout.get("runtime_truth_ledger_fresh") is not None
                else truth_row.get("runtime_truth_ledger_fresh"),
                "oldModelIsolationOk": rollout.get("old_model_isolation_ok")
                if rollout.get("old_model_isolation_ok") is not None
                else truth_row.get("old_model_isolation_ok"),
                "official_portfolio_capital_usd": current_truth.get("officialWalletPortfolioCapitalUsd") or truth_row.get("official_wallet_portfolio_capital_usd") or rollout.get("official_portfolio_capital_usd"),
                "claim_status": rollout.get("claim_status") or truth_row.get("claim_status"),
                "claimable_positions": rollout.get("official_claimable_positions") or truth_row.get("official_claimable_positions"),
                "official_claimable_usd": rollout.get("official_claimable_usd") or truth_row.get("official_claimable_usd"),
                "claim_daemon_status": rollout.get("claim_daemon_status") or truth_row.get("claim_daemon_status"),
                "claim_last_run": rollout.get("claim_last_run") or truth_row.get("claim_last_run"),
                "claim_last_result": rollout.get("claim_last_result") or truth_row.get("claim_last_result"),
                "claim_failed_count": rollout.get("claim_failed_count") or truth_row.get("claim_failed_count"),
                "claim_blocked_reason": rollout.get("claim_blocked_reason") or truth_row.get("claim_blocked_reason"),
                "last_attempt_result": rollout.get("last_order_attempt_result") or truth_row.get("last_order_attempt_result"),
                "source_classification": rollout.get("classification"),
                "officialActivityRows": slot01_canonical.get("officialActivityRows") if canonical_active else None,
                "officialPositionsRows": slot01_canonical.get("officialPositionsRows") if canonical_active else None,
                "officialMergedLocalRows": slot01_canonical.get("officialMergedLocalRows") if canonical_active else None,
                "officialBackfilledLogicalRows": slot01_canonical.get("officialBackfilledLogicalRows") if canonical_active else None,
            }
        )
    return rows


def _slot01_mapped_cooldown_rows_for_table(profile: str, monitor_set: str, view: str) -> List[Dict[str, Any]]:
    return []


def _print_loading_progress(
    profile: str,
    mode_filter: str,
    view: str,
    scope: str,
    active_only: bool,
    include_runtime_splits: bool,
    config_set: str,
    monitor_set: str,
    phase: str,
) -> None:
    runtime_split_label = "on" if include_runtime_splits else "off"
    active_label = "active + 常驻组" if active_only else "全部目录"
    print(
        f"  正在加载: {phase} | profile={profile} mode={mode_filter} view={view} "
        f"scope={scope} active={active_label} runtime_splits={runtime_split_label} "
        f"config={config_set} monitor={monitor_set}",
        flush=True,
    )


def _print_table(
    rows: List[Dict[str, Any]],
    zero_state_rows: int,
    zero_state_reasons: Dict[str, int],
    profile: str,
    mode_filter: str,
    view: str,
    active_only: bool,
    verbose: bool,
    scope: str,
    include_runtime_splits: bool,
    show_debug_surfaces: bool,
) -> None:
    def _display_pending_eta(ts: Any) -> str:
        if ts in (None, "", 0):
            return "-"
        try:
            return datetime.fromtimestamp(float(ts)).strftime("%m-%d %H:%M")
        except Exception:
            return "-"

    def _display_latest_time(value: Any) -> str:
        text = str(value or "").strip()
        if not text or text == "-":
            return "-"
        try:
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            return dt.astimezone().strftime("%m-%d %H:%M")
        except Exception:
            return text

    def _short_path_name(value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return "-"
        try:
            return Path(text).name
        except Exception:
            return text

    runtime_split_label = (
        "core-auto"
        if scope == "core" and view == "cell" and not include_runtime_splits
        else ("on" if include_runtime_splits else "off")
    )
    print()
    table_width = 170
    print("=" * table_width)
    print("  所有模型最新成交情况（Live/Sim 共存）")
    print(f"  刷新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(
        f"  范围: {'active + 常驻组' if active_only else '全部目录'} | "
        f"profile={profile} | mode={mode_filter} | view={view} | scope={scope} | "
        f"runtime_splits={runtime_split_label}"
    )
    print("=" * table_width)
    unit_label = "模型+币种" if view == "cell" else "模型"
    if verbose:
        fmt = "  {:<44} {:<18} {:<24} {:>9} {:>8} {:>8} {:>6} {:>8} {:>11} {:>12}"
        print(fmt.format("模型", "mode", "来源logsDir", "成交fill", "逻辑单", "胜/负", "待结", "胜率%", "最新成交", "总盈亏"))
    else:
        fmt = "  {:<44} {:>9} {:>8} {:>8} {:>6} {:>8} {:>11} {:>12}"
        print(fmt.format("模型", "成交fill", "逻辑单", "胜/负", "待结", "胜率%", "最新成交", "总盈亏"))
    print("  说明: 成交fill=分笔成交记录数；逻辑单=当前口径主交易单位；胜/负/待结=逻辑单结算口径；最新成交=最后一笔 executed 交易时间")
    print("  口径: live 总盈亏=官网组合资金(现金+未成交挂单占用+未结持仓估值)-初始资金；本地策略逻辑盈亏只作辅助，不进入主表。")
    print("-" * table_width)
    has_stale_raw_rows = False
    for r in rows:
        latest_display = _display_latest_time(r.get("latestSettledTradeDisplay") or r.get("latestDisplayedTradeAt") or r.get("latest_str") or "-")
        short_label = str(r["short"])
        prediction_source_status = str(r.get("predictionSourceStatus") or "").strip()
        if prediction_source_status == "non_runnable_no_1h_market":
            short_label = f"{short_label} [no-1h-market]"
        elif prediction_source_status == "non_runnable_missing_prediction_source":
            short_label = f"{short_label} [no-source]"
        elif prediction_source_status.startswith("research_"):
            short_label = f"{short_label} [1h-research]"
        elif r.get("runnerManaged"):
            short_label = f"{short_label} [runner]"
        elif r.get("staleRawLedger"):
            short_label = f"{short_label} [stale]"
            has_stale_raw_rows = True
        logical_count = int(
            r.get("allLogicTradeCount")
            or r.get("displayTrades")
            or r.get("displaySettledSampleCount")
            or (int(r.get("wins") or 0) + int(r.get("losses") or 0) + int(r.get("pending") or 0))
            or 0
        )
        win_rate_cell = _display_settled_win_rate_cell(r)
        if win_rate_cell == "-":
            win_rate_cell = f"{float(r.get('displayWinRate') or 0.0):.1f}"
        if verbose:
            print(
                fmt.format(
                    short_label,
                    r["mode"][:18],
                    r["runtime_dir"][:24],
                    r["n_executed"],
                    logical_count,
                    f"{r['wins']}/{r['losses']}",
                    r["pending"],
                    win_rate_cell,
                    latest_display,
                    f"${float(r.get('displayPnlUsd') or 0.0):+.2f}",
                )
            )
        else:
            print(
                fmt.format(
                    short_label,
                    r["n_executed"],
                    logical_count,
                    f"{r['wins']}/{r['losses']}",
                    r["pending"],
                    win_rate_cell,
                    latest_display,
                    f"${float(r.get('displayPnlUsd') or 0.0):+.2f}",
                )
            )
    print("-" * table_width)
    if has_stale_raw_rows:
        print("  注: [stale] 表示原始 simulation 账本停更，仅 summary 在重刷。")
    total_display_trades = sum(int(r.get("displaySettledSampleCount") or r.get("displayTrades") or 0) for r in rows)
    sim_rows_for_total = [r for r in rows if str(r.get("mode") or "").strip().lower() != "live" and str(r.get("role") or "").strip().lower() != "live_rollout"]
    live_rows_for_total = [r for r in rows if r not in sim_rows_for_total]
    sim_total_display_pnl = sum(float(r.get("displayPnlUsd") or 0.0) for r in sim_rows_for_total)
    live_total_pnl = sum(float(r.get("displayPnlUsd") or 0.0) for r in live_rows_for_total)
    print(
        f"  合计: {len(rows)} 个{unit_label}, {total_display_trades} 笔已结算逻辑单, "
        f"模拟已结盈亏 ${sim_total_display_pnl:+.2f}, live真钱官网盈亏 ${live_total_pnl:+.2f}"
    )
    probe_path = Path("/Users/mac/polyfun/reports/slot01_slot02_v2_balance_orders_probe_latest.json")
    recent_auto_path = Path("/Users/mac/polyfun/reports/slot01_slot02_latest_auto_order_official_status_latest.json")
    try:
        probe = json.loads(probe_path.read_text(encoding="utf-8")) if probe_path.exists() else {}
        generated_at = str(probe.get("generatedAt") or "").strip()
        recent_auto = json.loads(recent_auto_path.read_text(encoding="utf-8")) if recent_auto_path.exists() else {}
        recent_generated_at = str(recent_auto.get("generatedAt") or "").strip()
        def _parse_report_ts(value: str) -> float:
            raw = str(value or "").strip()
            if not raw:
                return 0.0
            try:
                return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
            except Exception:
                return 0.0
        # The balance probe and the recent-auto-order probe are generated by
        # different scripts. If the open-order probe is older, do not print it
        # above a newer official-order reconciliation line; that made canceled
        # zero-fill orders look like current website open orders.
        if recent_generated_at and _parse_report_ts(generated_at) < _parse_report_ts(recent_generated_at):
            raise RuntimeError("skip_stale_open_order_probe")
        slots = probe.get("slots") if isinstance(probe.get("slots"), dict) else {}
        open_parts: List[str] = []
        for key, label in (("slot1", "slot1"), ("slot2", "slot2")):
            row = slots.get(key) if isinstance(slots.get(key), dict) else {}
            orders = row.get("openOrders") if isinstance(row.get("openOrders"), list) else []
            if not orders:
                continue
            pending_usd = 0.0
            for order in orders:
                try:
                    price = float(order.get("price") or 0.0)
                    original_size = float(order.get("original_size") or 0.0)
                    matched = float(order.get("size_matched") or 0.0)
                    pending_usd += max(0.0, original_size - matched) * price
                except Exception:
                    continue
            open_parts.append(f"{label}={len(orders)}单/${pending_usd:.2f}")
        if open_parts:
            suffix = f" probe={generated_at}" if generated_at else ""
            print("  官网开放挂单: " + " ".join(open_parts) + suffix)
    except Exception:
        pass
    try:
        recent_auto = json.loads(recent_auto_path.read_text(encoding="utf-8")) if recent_auto_path.exists() else {}
        display_line = str(recent_auto.get("displayLine") or "").strip()
        generated_at = str(recent_auto.get("generatedAt") or "").strip()
        if display_line:
            suffix = f" probe={generated_at}" if generated_at else ""
            print("  最近自动单官网状态: " + display_line + "（open=0 表示官网未成交栏为空；本地未被官网确认的 created 不是当前挂单）" + suffix)
    except Exception:
        pass
    # 详细待结、规则、claim 和口径提示只写入报告文件，主表保持旧版简洁表格。
    early_admission_notes: List[str] = []
    for r in rows:
        if not _is_early_admission_monitor_candidate(r):
            continue
        tags: List[str] = []
        surface = str(r.get("predictionSurfaceChoice") or "").strip()
        if surface:
            tags.append(surface.replace("use_", ""))
        if bool(r.get("jointWinnerParamsApplied")):
            tags.append("联合赢家")
        tags.append("观察期")
        initial_capital = _float_or_none(r.get("initialCapitalUsd"))
        if initial_capital is None:
            initial_capital = _float_or_none(r.get("initialCapital"))
        if initial_capital is not None:
            tags.append(f"{int(initial_capital)}模拟")
        eligibility = str(r.get("scientificEligibilityStatus") or "").strip()
        if eligibility:
            tags.append(eligibility)
        coverage365 = _float_or_none(r.get("coverage365Pct"))
        latest_reason = str(r.get("latestNoTradeReason") or r.get("currentBlockingReason") or "").strip()
        status = str(r.get("currentStatus") or "").strip()
        details: List[str] = []
        if coverage365 is not None:
            details.append(f"365覆盖={coverage365:.2f}%")
        if status:
            details.append(f"状态={status}")
        if latest_reason:
            details.append(f"原因={latest_reason}")
        early_admission_notes.append(
            f"{r['short']}: {' / '.join(tags)}"
            + (f" | {'; '.join(details)}" if details else "")
        )
    if early_admission_notes:
        print("  观察期候选提示:")
        for note in early_admission_notes:
            print(f"    - {note}")
    if zero_state_rows > 0:
        print(f"  零状态行: {zero_state_rows}（配置在主线中，但当前尚无模拟成交）")
        if zero_state_reasons:
            reason_str = " ".join(f"{k}={v}" for k, v in zero_state_reasons.items())
            print(f"  零状态原因: {reason_str}")
    if scope == "core" and show_debug_surfaces:
        core_snap = _core_scope_snapshot()
        if core_snap:
            by_profile = core_snap.get("core_by_profile") if isinstance(core_snap.get("core_by_profile"), dict) else {}
            print(
                "  Core10: "
                f"core_cells={int(core_snap.get('core_cells') or 0)} "
                f"core_traders={int(core_snap.get('core_traders') or 0)} "
                f"shadow_cells={int(core_snap.get('shadow_cells') or 0)} "
                f"default={int(by_profile.get('default') or 0)} "
                f"70={int(by_profile.get('70') or 0)} "
                f"ranking_source={core_snap.get('ranking_source') or 'n/a'}"
            )
            trader_by_profile = core_snap.get("core_traders_by_profile") if isinstance(core_snap.get("core_traders_by_profile"), dict) else {}
            if trader_by_profile:
                print(
                    "  Core10 Traders: "
                    f"default={int(trader_by_profile.get('default') or 0)} "
                    f"70={int(trader_by_profile.get('70') or 0)}"
                )
            useful_by_profile: Dict[str, int] = {}
            for row in rows:
                profile_name = str(row.get("profile") or "").strip()
                if not profile_name:
                    continue
                if str(row.get("role") or "").strip().lower() == "compare_only":
                    continue
                if _is_shadow_only_fusion_row(row, fusion if 'fusion' in locals() else {}, scope, active_only, view):
                    continue
                useful_by_profile[profile_name] = useful_by_profile.get(profile_name, 0) + 1
            if useful_by_profile:
                if profile == "all":
                    print(
                        "  Useful Visible: "
                        f"default={int(useful_by_profile.get('default') or 0)} "
                        f"70={int(useful_by_profile.get('70') or 0)}"
                    )
                else:
                    print(f"  Useful Visible({profile})={int(useful_by_profile.get(profile) or 0)}")
            top_by_profile = core_snap.get("top_core_by_profile") if isinstance(core_snap.get("top_core_by_profile"), dict) else {}
            if profile == "all":
                top_rows = core_snap.get("top_core") if isinstance(core_snap.get("top_core"), list) else []
            else:
                scoped = top_by_profile.get(profile)
                top_rows = scoped if isinstance(scoped, list) else []
            if top_rows:
                top_str = ", ".join(
                    f"{str(x.get('profile') or '')}/{str(x.get('trader') or '')}-{str(x.get('symbol') or '')}:{float(x.get('total_pnl') or 0.0):+.2f}"
                    for x in top_rows
                    if isinstance(x, dict)
                )
                if top_str:
                    label = "Core10 Top" if profile == "all" else f"Core10 Top({profile})"
                    print(f"  {label}: {top_str}")
        extreme = _extreme_guard_snapshot()
        if extreme:
            print(
                "  Extreme Guard: "
                f"active_symbols={int(extreme.get('extreme_active_symbols') or 0)} "
                f"up={int(extreme.get('extreme_up_symbols') or 0)} "
                f"down={int(extreme.get('extreme_down_symbols') or 0)} "
                f"countertrend_hard_blocks={int(extreme.get('countertrend_hard_block_cells') or 0)}"
            )
        fusion = _fusion_execution_snapshot()
        if fusion.get("default_cells") or fusion.get("cells_70"):
            print(
                "  Fusion(global shadow, 非当前 profile 局部新鲜度): "
                f"default={int(fusion.get('default_cells') or 0)} "
                f"70={int(fusion.get('cells_70') or 0)} "
                f"fresh={'yes' if fusion.get('prediction_fresh') else 'no'} "
                f"state={fusion.get('execution_state') or '-'} "
                f"trade_cells={int(fusion.get('simulated_trade_cells') or 0)} "
                f"prediction_only={int(fusion.get('prediction_only_cells') or 0)} "
                f"shadow_only={int(fusion.get('shadow_only_cells') or 0)} "
                f"exec_not_landed={int(fusion.get('execution_not_landed_cells') or 0)}"
            )
    rollout_rows = _visible_live_rollout_rows(profile)
    if view == "cell" and rollout_rows and show_debug_surfaces:
        print("  Live Rollout:")
        for rollout in rollout_rows:
            target = rollout.get("target_cell") if isinstance(rollout.get("target_cell"), dict) else {}
            cell_label = str(
                rollout.get("target_display_label")
                or f"{target.get('profile')}/{target.get('trader')}-{target.get('symbol')}"
            )
            source_trader = str(rollout.get("target_source_trader") or target.get("trader") or "-")
            classification = str(rollout.get("classification") or "unknown")
            slot_name = str(target.get("walletSlot") or "-")
            country = str(rollout.get("egress_country") or "-")
            policy = str(rollout.get("egress_policy") or "-")
            last_block = str(rollout.get("last_runtime_block_reason") or "-")
            last_attempt = str(rollout.get("last_order_attempt_result") or "-")
            summary_stale = "yes" if rollout.get("summary_stale_vs_latest_trade") else "no"
            pending_logical = int(rollout.get("pending_logical_orders") or 0)
            pending_rungs = int(rollout.get("pending_rungs") or 0)
            open_order_logical_groups = int(rollout.get("open_order_logical_groups") or 0)
            pending_order_count = int(rollout.get("pending_order_count") or 0)
            settled_logical = int(rollout.get("settled_logical_orders") or 0)
            claim_ready_logical = int(rollout.get("claim_ready_logical_orders") or 0)
            claimed_logical = int(rollout.get("claimed_logical_orders") or 0)
            attempted_rungs = int(rollout.get("attempted_rungs") or 0)
            posted_rungs = int(rollout.get("posted_rungs") or 0)
            filled_rungs = int(rollout.get("filled_rungs") or 0)
            rejected_rungs = int(rollout.get("rejected_rungs") or 0)
            pending_notional = rollout.get("pending_notional_usd")
            last_region = str(rollout.get("last_region_restriction_error") or "-")
            resolution_status = str(rollout.get("resolution_status") or "-")
            settlement_status = str(rollout.get("settlement_status") or "-")
            claim_status = str(rollout.get("claim_status") or "-")
            preferred_claim_status = str(rollout.get("preferred_claim_status") or claim_status or "-")
            truth_priority = str(rollout.get("truth_priority") or "local")
            truth_mismatch = bool(rollout.get("truth_mismatch"))
            truth_mismatch_fields = ",".join(str(x) for x in (rollout.get("truth_mismatch_fields") or []) if str(x)) or "-"
            official_claimable_usd = rollout.get("official_claimable_usd")
            official_claimable_positions = rollout.get("official_claimable_positions")
            official_cash_pnl_usd = rollout.get("official_cash_pnl_usd")
            official_portfolio_capital_usd = rollout.get("official_portfolio_capital_usd")
            display_live_wallet_pnl_usd = rollout.get("display_live_wallet_pnl_usd")
            strategy_logical_pnl_usd = rollout.get("strategy_logical_pnl_usd")
            last_no_trade_reason = str(rollout.get("last_no_trade_reason") or "-")
            recent_guard_skip_counts = rollout.get("recent_guard_skip_counts") if isinstance(rollout.get("recent_guard_skip_counts"), dict) else {}
            recent_no_signal_counts = rollout.get("recent_no_signal_counts") if isinstance(rollout.get("recent_no_signal_counts"), dict) else {}
            live_consumption_status = str(rollout.get("live_consumption_status") or "-")
            family_rule_version = str(rollout.get("family_rule_version") or "-")
            stale_order_policy = str(rollout.get("stale_order_policy") or "-")
            claim_daemon_status = str(rollout.get("claim_daemon_status") or "-")
            claim_last_run = str(rollout.get("claim_last_run") or "-")
            claim_last_result = str(rollout.get("claim_last_result") or "-")
            claim_wallet_model = str(rollout.get("claim_wallet_model") or "-")
            claim_submit_auth_mode = str(rollout.get("claim_submit_auth_mode") or "-")
            claim_verify_auth_mode = str(rollout.get("claim_verify_auth_mode") or "-")
            claim_blocked_reason = str(rollout.get("claim_blocked_reason") or "-")
            latest_rejection_map = rollout.get("last_attempt_rejection_reasons") if isinstance(rollout.get("last_attempt_rejection_reasons"), dict) else {}
            rejection_map = rollout.get("rejection_reasons") if isinstance(rollout.get("rejection_reasons"), dict) else {}
            last_non_execution_reason = str(rollout.get("last_non_execution_reason") or "-")
            execution_decision = str(rollout.get("execution_decision") or "-")
            min_order_size = rollout.get("exchange_min_order_size_shares")
            latest_rejection_summary = ",".join(
                f"{reason}:{int(count)}"
                for reason, count in sorted(latest_rejection_map.items())
                if int(count or 0) > 0
            ) or "-"
            guard_skip_summary = ",".join(
                f"{reason}:{int(count)}"
                for reason, count in sorted(recent_guard_skip_counts.items())
                if int(count or 0) > 0
            ) or "-"
            no_signal_summary = ",".join(
                f"{reason}:{int(count)}"
                for reason, count in sorted(recent_no_signal_counts.items())
                if int(count or 0) > 0
            ) or "-"
            region_history = int(rejection_map.get("region_restricted") or 0)
            region_hint = last_region[:48] if last_region != "-" else "-"
            print(
                f"    {cell_label}: class={classification} source={source_trader} slot={slot_name} "
                f"egress={country}/{policy} last_attempt={last_attempt} "
                f"pending={pending_logical}L "
                f"open_orders={open_order_logical_groups}G/{pending_order_count}O/{pending_rungs}R "
                f"pending_usd={(('$' + format(float(pending_notional), '.2f')) if pending_notional is not None else '-')} "
                f"attempted={attempted_rungs} posted={posted_rungs} filled={filled_rungs} rejected={rejected_rungs} "
                f"settled={settled_logical} claim_ready={claim_ready_logical} claimed={claimed_logical} "
                f"stage={resolution_status}/{settlement_status}/{claim_status} "
                f"preferred_stage={preferred_claim_status} truth={truth_priority} mismatch={'yes' if truth_mismatch else 'no'}[{truth_mismatch_fields}] "
                f"official_claimable={(('$' + format(float(official_claimable_usd), '.2f')) if official_claimable_usd is not None else '-')} "
                f"claimable_positions={official_claimable_positions if official_claimable_positions is not None else '-'} "
                f"official_cash_pnl={(('$' + format(float(official_cash_pnl_usd), '.2f')) if official_cash_pnl_usd is not None else '-')} "
                f"official_capital={(('$' + format(float(official_portfolio_capital_usd), '.2f')) if official_portfolio_capital_usd is not None else '-')} "
                f"display_pnl={(('$' + format(float(display_live_wallet_pnl_usd), '.2f')) if display_live_wallet_pnl_usd is not None else '-')} "
                f"strategy_pnl={(('$' + format(float(strategy_logical_pnl_usd), '.2f')) if strategy_logical_pnl_usd is not None else '-')} "
                f"claim_last_run={claim_last_run} claim_last_result={claim_last_result} claim_daemon={claim_daemon_status} "
                f"claim_model={claim_wallet_model} submit_auth={claim_submit_auth_mode} verify_auth={claim_verify_auth_mode} "
                f"claim_blocked={claim_blocked_reason[:40]} "
                f"exec_decision={execution_decision} "
                f"min_shares={format(float(min_order_size), '.2f') if min_order_size is not None else '-'} "
                f"non_exec={last_non_execution_reason[:48]} "
                f"latest_rejects={latest_rejection_summary} "
                f"region_history={region_history} guard_block={last_block} "
                f"last_no_trade={last_no_trade_reason} guard_skips={guard_skip_summary} no_signal={no_signal_summary} "
                f"live_values={live_consumption_status} family_rule={family_rule_version} stale={stale_order_policy} "
                f"region_hint={region_hint} summary_stale={summary_stale}"
            )
    print()


def _build_payload(
    rows: List[Dict[str, Any]],
    zero_state_rows: int,
    zero_state_reasons: Dict[str, int],
    filtered_lowprice_orphan_rows: int,
    profile: str,
    mode_filter: str,
    active_only: bool,
    view: str,
    scope: str,
    include_runtime_splits: bool,
) -> Dict[str, Any]:
    runtime_split_mode = (
        "core_auto"
        if scope == "core" and view == "cell" and not include_runtime_splits
        else ("explicit_on" if include_runtime_splits else "explicit_off")
    )
    by_profile: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        prof = str(row.get("profile") or "default")
        bucket = by_profile.setdefault(
            prof,
            {
                "rows": 0,
                "executed": 0,
                "display_trades": 0,
                "display_total_pnl": 0.0,
                "raw_total_pnl": 0.0,
                "ready": 0,
                "blocked": 0,
                "live": 0,
            },
        )
        bucket["rows"] += 1
        bucket["executed"] += int(row.get("n_executed") or 0)
        bucket["display_trades"] += int(row.get("displayTrades") or 0)
        bucket["display_total_pnl"] += float(row.get("displayPnlUsd") or 0.0)
        bucket["raw_total_pnl"] += float(row.get("rawSimPnlUsd") or 0.0)
        status = str(row.get("promotionEligibility") or "").strip()
        if status in {"ready", "blocked", "live"}:
            bucket[status] += 1
    compare_only_rows = sum(1 for row in rows if str(row.get("role") or "").strip().lower() == "compare_only")
    payload = {
        "generated_at": datetime.now().isoformat(),
        "profile": profile,
        "mode": mode_filter,
        "view": view,
        "scope": scope,
        "active_only": active_only,
        "include_runtime_splits": include_runtime_splits,
        "runtime_split_mode": runtime_split_mode,
        "summary": {
            "rows": len(rows),
            "compare_only_rows": int(compare_only_rows),
            "filtered_lowprice_orphan_rows": int(filtered_lowprice_orphan_rows),
            "zero_state_rows": int(zero_state_rows),
            "zero_state_reasons": zero_state_reasons,
            "executed": int(sum(int(r.get("n_executed") or 0) for r in rows)),
            "display_trades": int(sum(int(r.get("displayTrades") or 0) for r in rows)),
            "display_total_pnl": round(sum(float(r.get("displayPnlUsd") or 0.0) for r in rows), 2),
            "raw_total_pnl": round(sum(float(r.get("rawSimPnlUsd") or 0.0) for r in rows), 2),
            "total_pnl": round(sum(float(r.get("displayPnlUsd") or 0.0) for r in rows), 2),
            "ready_rows": int(sum(1 for r in rows if str(r.get("promotionEligibility") or "") == "ready")),
            "blocked_rows": int(sum(1 for r in rows if str(r.get("promotionEligibility") or "") == "blocked")),
            "live_rows": int(sum(1 for r in rows if str(r.get("promotionEligibility") or "") == "live")),
            "stale_render_detected_rows": int(sum(1 for r in rows if bool(r.get("staleRenderDetected")))),
            "by_profile": {
                k: {
                    "rows": int(v["rows"]),
                    "executed": int(v["executed"]),
                    "display_trades": int(v["display_trades"]),
                    "display_total_pnl": round(float(v["display_total_pnl"]), 2),
                    "raw_total_pnl": round(float(v["raw_total_pnl"]), 2),
                    "total_pnl": round(float(v["display_total_pnl"]), 2),
                    "ready": int(v["ready"]),
                    "blocked": int(v["blocked"]),
                    "live": int(v["live"]),
                }
                for k, v in sorted(by_profile.items())
            },
            "live_rollout_reason_surface": _visible_live_rollout_rows(profile) if view == "cell" else [],
        },
        "rows": rows,
    }
    if scope == "core":
        payload["core10_composition"] = _core_scope_snapshot()
        payload["extreme_guard"] = _extreme_guard_snapshot()
    return payload
def run_once(
    profile: str,
    mode_filter: str,
    active_only: bool,
    view: str,
    verbose: bool,
    scope: str,
    include_runtime_splits: bool,
    show_debug_surfaces: bool = False,
    show_zero_state: bool = False,
    show_hidden_diagnostics: bool = False,
    config_set: str = "main",
    monitor_set: str = "main",
    emit_progress: bool = False,
) -> Dict[str, Any]:
    _reset_monitor_runtime_caches()
    if emit_progress:
        _print_loading_progress(
            profile=profile,
            mode_filter=mode_filter,
            view=view,
            scope=scope,
            active_only=active_only,
            include_runtime_splits=include_runtime_splits,
            config_set=config_set,
            monitor_set=monitor_set,
            phase="收集运行目录与成交快照",
        )
    rows = _collect_rows(
        profile=profile,
        mode_filter=mode_filter,
        active_only=active_only,
        view=view,
        scope=scope,
        include_runtime_splits=include_runtime_splits,
        config_set=config_set,
        monitor_set=monitor_set,
        include_hidden_rows=bool(show_hidden_diagnostics),
        show_zero_state=bool(show_zero_state),
    )
    if emit_progress:
        _print_loading_progress(
            profile=profile,
            mode_filter=mode_filter,
            view=view,
            scope=scope,
            active_only=active_only,
            include_runtime_splits=include_runtime_splits,
            config_set=config_set,
            monitor_set=monitor_set,
            phase="整理主表与真实交易行",
        )
    filtered_lowprice_orphan_rows = int(_LAST_FILTERED_LOWPRICE_ORPHAN_ROWS)
    fusion = _fusion_execution_snapshot() if scope == "core" else {}
    zero_state_rows = sum(1 for row in rows if _is_config_only_zero_row(row, scope, active_only, view))
    zero_state_reasons = _hidden_zero_state_reason_summary(rows, scope, active_only, view)
    rows = [row for row in rows if not _is_shadow_only_fusion_row(row, fusion, scope, active_only, view)]
    if view == "cell":
        rollout_rows = _live_rollout_rows_for_table(profile, rows)
        if rollout_rows:
            rollout_labels = {
                str(row.get("short") or "").strip()
                for row in rollout_rows
                if str(row.get("short") or "").strip()
            }
            slot2_rollout_present = any(
                str(row.get("slot") or "").strip().lower() == "slot02"
                and str(row.get("trader") or "").strip() == "v5_exp10_bp0480"
                and str(row.get("symbol") or "").strip().upper() == "ETH"
                for row in rollout_rows
            )
            rows = [
                row for row in rows
                if str(row.get("short") or "").strip() not in rollout_labels
                and not (
                    slot2_rollout_present
                    and str(row.get("role") or "").strip().lower() != "live_rollout"
                    and str(row.get("mode") or "").strip().lower() == "live"
                    and str(row.get("trader") or "").strip() == "v5_exp10_bp0480"
                    and str(row.get("symbol") or "").strip().upper() == "ETH"
                )
            ]
            rows.extend(rollout_rows)
        mapped_rows = _slot01_mapped_cooldown_rows_for_table(profile, monitor_set, view)
        if mapped_rows:
            mapped_labels = {
                str(row.get("short") or "").strip()
                for row in mapped_rows
                if str(row.get("short") or "").strip()
            }
            rows = [
                row for row in rows
                if str(row.get("short") or "").strip() not in mapped_labels
            ]
            rows.extend(mapped_rows)
        disabled_live_keys = _disabled_live_selection_keys()
        if disabled_live_keys:
            rows = [
                row for row in rows
                if not _is_disabled_live_selection_display_row(row, disabled_live_keys)
            ]
    rows = _attach_live_equivalent_contract(rows)
    rows = _attach_monitor_freshness(rows, datetime.now(timezone.utc))
    rows.sort(key=_row_sort_key)
    _print_table(
        rows,
        zero_state_rows=zero_state_rows,
        zero_state_reasons=zero_state_reasons,
        profile=profile,
        mode_filter=mode_filter,
        view=view,
        active_only=active_only,
        verbose=verbose,
        scope=scope,
        include_runtime_splits=include_runtime_splits,
        show_debug_surfaces=show_debug_surfaces,
    )
    return _build_payload(
        rows,
        zero_state_rows=zero_state_rows,
        zero_state_reasons=zero_state_reasons,
        filtered_lowprice_orphan_rows=filtered_lowprice_orphan_rows,
        profile=profile,
        mode_filter=mode_filter,
        active_only=active_only,
        view=view,
        scope=scope,
        include_runtime_splits=include_runtime_splits,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="所有模型最新下单情况 - 持续监控（Live/Sim 共存）")
    parser.add_argument("--interval", "-i", type=int, default=60, help="刷新间隔(秒)，默认 60")
    parser.add_argument("--once", action="store_true", help="只跑一次，不持续刷新")
    parser.add_argument("--all-combos", action="store_true", help="显示全量目录（默认仅 active + 常驻组）")
    parser.add_argument("--profile", choices=("default", "70", "threshold_pack", "all"), default="default")
    # 默认给“最稳最可比”的统计口径，避免 mixed mode/cell 造成体感跳变。
    parser.add_argument("--mode", choices=("simulation", "live", "both"), default="simulation")
    parser.add_argument("--view", choices=("cell", "model"), default="cell", help="cell=模型+币种; model=按运行目录汇总")
    parser.add_argument("--scope", choices=("core", "shadow", "watchlist", "alerts", "regime_alerts", "all"), default="core", help="core=主交易, shadow=剩余观察, watchlist=单体替补候选, alerts=短窗口恶化, regime_alerts=极端态+短窗口联合告警, all=全量")
    parser.add_argument("--verbose", action="store_true", help="显示 mode/runtime logsDir 等详细列")
    parser.add_argument("--include-runtime-splits", action="store_true", help="显示全部 __simulation_* / __live_* 运行时拆分目录；默认 core/cell 会自动选最新目录但不重复展示")
    parser.add_argument("--config-set", choices=("main", "lowprice"), default="main", help="读取哪套 trader configs/active")
    parser.add_argument("--monitor-set", choices=("main", "lowprice", "all"), default="main", help="读取哪套 monitor-only manifests")
    parser.add_argument("--json-out", type=Path, default=None, help="可选：将当前排行写入 JSON")
    parser.add_argument("--show-debug-surfaces", action="store_true", help="显示 Core10/Fusion/Live Rollout 等调试大段")
    parser.add_argument("--show-zero-state", action="store_true", help="显示默认隐藏的零活动 compare-only 行")
    parser.add_argument("--show-hidden-diagnostics", action="store_true", help="输出全部 hidden/superseded 诊断行")
    args = parser.parse_args()
    runtime_root = str(SCRIPT_DIR.parent)
    runtime_pm = str(SCRIPT_DIR)

    active_only = not args.all_combos
    if args.once:
        if args.json_out is None:
            print(f"  运行根: {runtime_root}", flush=True)
            print(f"  Polymarket根: {runtime_pm}", flush=True)
            print("  已启动监控，正在准备首屏...", flush=True)
            print(
                f"  正在加载: 初始化监控上下文 | profile={args.profile} mode={args.mode} "
                f"view={args.view} scope={args.scope} config={args.config_set} monitor={args.monitor_set}",
                flush=True,
            )
        payload = run_once(
            profile=args.profile,
            mode_filter=args.mode,
            active_only=active_only,
            view=args.view,
            verbose=args.verbose,
            scope=args.scope,
            include_runtime_splits=bool(args.include_runtime_splits),
            show_debug_surfaces=bool(args.show_debug_surfaces),
            show_zero_state=bool(args.show_zero_state),
            show_hidden_diagnostics=bool(args.show_hidden_diagnostics),
            config_set=args.config_set,
            monitor_set=args.monitor_set,
            emit_progress=bool(args.json_out is None),
        )
        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    print("  按 Ctrl+C 退出")
    print(f"  运行根: {runtime_root}")
    print(f"  Polymarket根: {runtime_pm}")
    print("  已启动监控，正在准备首屏...", flush=True)
    print(
        f"  正在加载: 初始化监控上下文 | profile={args.profile} mode={args.mode} "
        f"view={args.view} scope={args.scope} config={args.config_set} monitor={args.monitor_set}",
        flush=True,
    )
    try:
        while True:
            payload = run_once(
                profile=args.profile,
                mode_filter=args.mode,
                active_only=active_only,
                view=args.view,
                verbose=args.verbose,
                scope=args.scope,
                include_runtime_splits=bool(args.include_runtime_splits),
                show_debug_surfaces=bool(args.show_debug_surfaces),
                show_zero_state=bool(args.show_zero_state),
                show_hidden_diagnostics=bool(args.show_hidden_diagnostics),
                config_set=args.config_set,
                monitor_set=args.monitor_set,
                emit_progress=True,
            )
            if args.json_out is not None:
                args.json_out.parent.mkdir(parents=True, exist_ok=True)
                args.json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n  已停止")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
