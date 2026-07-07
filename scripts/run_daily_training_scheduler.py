#!/usr/bin/env python3
"""
每日定时训练 + 链式监控：每天到点跑 Exp + GRU 增量训练并写盘，等待一段时间后跑 monitor 校验
「热重载是否发生、预测输出是否用上最新模型」。
不依赖 crontab，不会触发 macOS「想要管理你的电脑」提示。

用法（在项目根目录）:
  python3 scripts/run_daily_training_scheduler.py
  python3 scripts/run_daily_training_scheduler.py 8        # 每天 8 点
  python3 scripts/run_daily_training_scheduler.py 0 0     # 每天 0 点，训练后不等待直接跑监控
  python3 scripts/run_daily_training_scheduler.py 8 20    # 每天 8 点，训练后等 20 分钟再跑监控
  python3 scripts/run_daily_training_scheduler.py --interval-minutes 30 --interval-offset-minutes 2
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

from scripts.ops.incremental_param_search_pause_common import load_pause_state, maintenance_window_active
from scripts.ops.night_window_dispatch_common import write_dispatch_plan

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 训练完成后等待多少分钟再跑监控（让预测写入器至少跑完一轮并热重载，便于监控校验）
DEFAULT_WAIT_BEFORE_MONITOR_MINUTES = 18
DEFAULT_INTERVAL_OFFSET_MINUTES = 2
DEFAULT_TRAIN_SCRIPT_TIMEOUT_SECONDS = 2400
DEFAULT_GROUPED_TRAIN_SCRIPT_TIMEOUT_SECONDS = 4200
DEFAULT_PARAM_SEARCH_CADENCE_MINUTES = 120
DEFAULT_TRAINING_CADENCE_MINUTES = 30
DEFAULT_PARAM_SEARCH_MAX_CATCHUP_SLICES_PER_LAUNCH = 16
DEFAULT_PARAM_SEARCH_SAFE_MAX_MEMBER_WORKERS = 1
DEFAULT_PARAM_SEARCH_SAFE_MAX_CANDIDATE_WORKERS = 1
DEFAULT_PARAM_SEARCH_SAFE_NUMERIC_THREADS = 1
DEFAULT_PARAM_SEARCH_SAFE_FAMILY_CONCURRENCY = 1
DEFAULT_NIGHTLY_LAUNCHER_GRACE_MINUTES = 5
DEFAULT_NIGHT_SEARCH_START_HOUR = 0
DEFAULT_NIGHT_SEARCH_END_HOUR = 8
RUN_LOCK_FILE = PROJECT_ROOT / "logs" / "online_learning_scheduler.lock"
SCHEDULER_PID_FILE = PROJECT_ROOT / "logs" / "daily_training_scheduler.pid"
ONLINE_HISTORY_FILE = PROJECT_ROOT / "logs" / "online_learning_history.jsonl"
COVERAGE_REPORT_FILE = PROJECT_ROOT / "logs" / "daily_training_coverage_report.json"
INCREMENTAL_PARAM_SEARCH_RUNTIME_STATUS = PROJECT_ROOT / "reports" / "incremental_param_search_runtime_status_latest.json"
DEFAULT_COVERAGE_LOOKBACK_HOURS = 72
DEFAULT_COVERAGE_GRACE_MINUTES = 40
DEFAULT_COVERAGE_CHECK_INTERVAL_MINUTES = 10
DEFAULT_COVERAGE_HISTORY_MAX_LINES = 8000
DEFAULT_MAX_CATCHUP_RUNS = 3
DEFAULT_PM_BACKFILL_LOOKBACK_HOURS = 72
DEFAULT_PM_BACKFILL_CHECK_INTERVAL_MINUTES = 10
DEFAULT_PM_BACKFILL_MAX_MISSING_BARS = 4
DEFAULT_PM_BACKFILL_STALE_MINUTES = 35
DEFAULT_DISK_HYGIENE_INTERVAL_MINUTES = 240
DEFAULT_ARCHIVE_HYGIENE_INTERVAL_MINUTES = 240
DISK_HYGIENE_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "disk_hygiene_guard.py"
ARCHIVE_TUNING_HISTORY_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "archive_tuning_resume_history.py"
ARCHIVE_RUNTIME_BACKUPS_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "archive_runtime_backups.py"
MONTHLY_TRADE_REGIME_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_monthly_trade_regime_analysis.py"
WIDE_OBSERVATION_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_wide_universe_observation_latest.py"
CORE10_REGIME_REVIEW_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_regime_review_latest.py"
CORE_UNIVERSE_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core_universe_latest.py"
CORE_SHADOW_APPLY_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "apply_core_shadow_selector.py"
CORE10_ROTATION_WATCHLIST_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_rotation_watchlist_latest.py"
CORE10_RECENT_WINDOW_WATCH_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_recent_window_watch_latest.py"
CORE10_FOCUS_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_focus_latest.py"
CORE10_INCREMENTAL_PROFILE_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_incremental_profile.py"
CORE10_MODEL_BOOST_SPEC_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_model_boost_spec.py"
ACTIVE_INCREMENTAL_SEARCH_UNIVERSE_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_active_incremental_search_universe_latest.py"
INCREMENTAL_NIGHT_SEARCH_BACKLOG_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_incremental_night_search_backlog_latest.py"
CORE10_COMBO_BOOTSTRAP_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "bootstrap_core10_combo_symbol_model_dirs.py"
CORE10_SOL_RESEARCH_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_sol_research_latest.py"
CORE10_INCREMENTAL_WRITEBACK_GUARD_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_incremental_writeback_guard_latest.py"
CORE10_INCREMENTAL_PARAM_SEARCH_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_incremental_param_search_latest.py"
INCREMENTAL_PARAM_SEARCH_NONBLOCKING_RUNNER = PROJECT_ROOT / "scripts" / "ops" / "run_incremental_param_search_nonblocking.py"
CORE10_SIMULATION_MONEY_RECONCILIATION_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_simulation_money_reconciliation_latest.py"
CORE10_CODE_AUDIT_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_code_audit_latest.py"
CORE10_LIVE_COEXISTENCE_AUDIT_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_live_coexistence_audit.py"
CORE10_REGIME_BACKTEST_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_regime_backtest_latest.py"
CORE10_DEEP_OPT_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_deep_optimization_latest.py"
CORE10_DIRECTION_BUCKET_WATCH_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_direction_bucket_watch_latest.py"
CORE10_DIRECTIONAL_CALIBRATION_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_directional_calibration_latest.py"
CORE10_SELECTIVE_ABSTAIN_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_selective_abstain_latest.py"
CORE10_SELECTIVE_ABSTAIN_APPLY_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "apply_core10_selective_abstain.py"
CORE10_UNCERTAINTY_GATE_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_uncertainty_gate_latest.py"
CORE10_UNCERTAINTY_GATE_APPLY_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "apply_core10_uncertainty_gate.py"
CORE10_PREDICTION_SOURCE_AUDIT_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_prediction_source_audit_latest.py"
SELECTOR_ZERO_TRADE_AUDIT_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_selector_zero_trade_audit_latest.py"
MAINLINE_RUNTIME_GUARD_REVIEW_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_mainline_runtime_guard_review_latest.py"
MAINLINE_POSTLAUNCH_GUARD_EFFECT_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_mainline_postlaunch_guard_effect_review_latest.py"
MAINLINE_OVERNIGHT_REVIEW_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_mainline_overnight_review_latest.py"
MAINLINE_DEFAULT_WEAKNESS_REVIEW_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_mainline_default_weakness_review_latest.py"
MAINLINE_BUY_PRICE_TRUTH_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_mainline_buy_price_truth_latest.py"
MAINLINE_DEFAULT_BUY_PRICE_BALANCE_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_mainline_default_buy_price_balance_latest.py"
MAINLINE_DEFAULT_SHADOW_VALIDATION_QUEUE_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_mainline_default_shadow_validation_queue_latest.py"
MAINLINE_RECENT_CHANGE_SEMANTIC_AUDIT_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_mainline_recent_change_semantic_audit_latest.py"
MAINLINE_ACCEPTANCE_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_mainline_acceptance_latest.py"
MAINLINE_CHAIN_ALIGNMENT_AUDIT_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_mainline_chain_alignment_audit_latest.py"
ACTIVE_TRUTH_SURFACE_CONSISTENCY_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_active_truth_surface_consistency_latest.py"
POST_RESET_TRADE_STRUCTURE_REVIEW_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_post_reset_trade_structure_review_latest.py"
SIMULATION_ORDER_SEMANTICS_AUDIT_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_simulation_order_semantics_audit_latest.py"
MONITOR_CELL_CAPITAL_CONTRACT_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_monitor_cell_capital_contract_latest.py"
MAINLINE_FINAL_WRAPUP_AUDIT_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_mainline_final_wrapup_audit_latest.py"
MAINLINE_SAFE_CLEANUP_INVENTORY_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_mainline_safe_cleanup_inventory_latest.py"
CORE10_LEGACY_CLEANUP_INVENTORY_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_legacy_cleanup_inventory_latest.py"
CORE10_WRITER_LAUNCHCTL_INVENTORY_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_writer_launchctl_inventory_latest.py"
MAINLINE_AUDIT_LOOP_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_mainline_audit_loop_latest.py"
MAINLINE_REFRESH_CHAIN_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "run_mainline_refresh_chain.py"
MAINLINE_REFRESH_NONBLOCKING_RUNNER = PROJECT_ROOT / "scripts" / "ops" / "run_mainline_refresh_nonblocking.py"
RELOAD_PREDICTION_WRITERS_SCRIPT = PROJECT_ROOT / "reload_launchctl_prediction_writers.sh"
CORE10_GUARD_STACK_AUDIT_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_guard_stack_audit_latest.py"
CORE10_GUARD_EFFECTIVENESS_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_guard_effectiveness_latest.py"
DEFAULT_EXP10_ETH_TRUTH_MANIFEST_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_default_exp10_bp0520_eth_truth_manifest_latest.py"
DEFAULT_EXP10_ETH_INCREMENTAL_BINDING_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_default_exp10_bp0520_eth_incremental_binding_latest.py"
DEFAULT_EXP10_ETH_ACTIVE_GUARD_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_default_exp10_bp0520_eth_active_guard_layers_latest.py"
DEFAULT_EXP10_ETH_EXECUTOR_ENV_AUDIT_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_default_exp10_bp0520_eth_executor_env_audit_latest.py"
DEFAULT_EXP10_ETH_GUARD_EXECUTOR_COMPARE_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_default_exp10_bp0520_eth_guard_executor_compare_latest.py"
CORE10_ONLY_CLOSED_LOOP_AUDIT_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_only_closed_loop_audit_latest.py"
CORE10_FOLLOW_LANE_REPAIR_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_follow_lane_repair_latest.py"
CORE10_FOLLOW_LANE_REPAIR_APPLY_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "apply_core10_follow_lane_repair.py"
CORE10_REGIME_POLICY_AUDIT_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_regime_policy_audit_latest.py"
CORE10_BACKLOG_RECONCILIATION_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_backlog_reconciliation_latest.py"
CORE10_RUNTIME_PARAM_SEARCH_SPEC_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_runtime_param_search_spec_latest.py"
CORE10_RUNTIME_PARAM_SEARCH_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_runtime_param_search_latest.py"
CORE10_RUNTIME_PARAM_APPLY_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "apply_core10_runtime_param_search.py"
CORE10_EXP13_UP_LANE_SEARCH_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_exp13_up_lane_search_latest.py"
CORE10_EXP13_UP_LANE_APPLY_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "apply_core10_exp13_up_lane_search.py"
CORE10_EXP15_UP_LANE_SEARCH_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_exp15_up_lane_search_latest.py"
CORE10_EXP15_UP_LANE_APPLY_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "apply_core10_exp15_up_lane_search.py"
CORE10_DIRECTIONAL_DRAG_AUDIT_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_directional_drag_audit_latest.py"
CORE10_DEFAULT_ETH_DOWN_TUNING_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_default_eth_down_tuning_latest.py"
CORE10_DEFAULT_ETH_DOWN_TUNING_APPLY_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "apply_core10_default_eth_down_tuning.py"
CORE10_PROTECTED_BAND_CAPS_APPLY_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "apply_core10_protected_band_caps.py"
CORE10_PROTECTED_BAND_DRIFT_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_protected_band_drift_latest.py"
CORE10_PROFILE70_ETH_TUNING_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_profile70_eth_direction_tuning_latest.py"
CORE10_PROFILE70_ETH_TUNING_APPLY_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "apply_core10_profile70_eth_direction_tuning.py"
CORE10_REGIME_DIRECTION_TRUTH_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_regime_direction_truth_latest.py"
CORE10_EXP10_DIRECTION_TUNING_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_exp10_direction_tuning_latest.py"
CORE10_EXP10_DIRECTION_TUNING_APPLY_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "apply_core10_exp10_direction_tuning.py"
CORE10_RECENT_SAMPLE_REVIEW_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_recent_sample_review_latest.py"
CORE10_FINAL_LOGIC_AUDIT_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_final_logic_audit_latest.py"
CORE10_STAGE16_SHADOW_GATE_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_stage16_shadow_gate_latest.py"
CORE10_STAGE16_SHADOW_COMPARE_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_stage16_shadow_compare_latest.py"
CORE10_STAGE16_SHADOW_SIM_CANDIDATE_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_stage16_shadow_sim_candidate_latest.py"
CORE10_STAGE16_SHADOW_LIVE_WATCH_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_stage16_shadow_live_watch_latest.py"
CORE10_STAGE16_SHADOW_LIVE_COMPARE_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_stage16_shadow_live_compare_latest.py"
CORE10_STAGE16_SHADOW_STABILITY_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_stage16_shadow_stability_latest.py"
CORE10_EXP13_FULL_STACK_REPLAY_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_exp13_full_stack_replay_latest.py"
CORE10_STAGE16_EVENT_PRECISION_SEARCH_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_stage16_event_precision_search_latest.py"
CORE10_EXP13_OFFICIAL_SIM_CANDIDATE_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_exp13_official_sim_candidate_latest.py"
CORE10_ROUTE_F_EXP10_PILOT_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_route_f_exp10_pilot_latest.py"
CORE10_ROUTE_F_EXP10_OFFICIAL_SIM_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_route_f_exp10_official_sim_candidate_latest.py"
CORE10_LEAKAGE_AUDIT_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_leakage_audit_latest.py"
COREMAIN_RALLY_STATE_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "write_regime_shadow_state.py"
COREMAIN_RALLY_OVERLAY_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "apply_coremain_rally_overlays.py"
COREMAIN_REGIME_ATTRIBUTION_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_coremain_regime_pnl_attribution_latest.py"
ALLORA_FEATURE_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "fetch_allora_features.py"
POLYFUN_MAINLINE_FINAL_STATUS_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_polyfun_mainline_final_status_latest.py"
MAINLINE_70_THRESHOLD_SEARCH_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_mainline_70_btc_eth_threshold_search_latest.py"
DEFAULT_INCREMENTAL_PARAM_SEARCH_STALE_HOURS = 12
MAINLINE_REFRESH_CHAIN_TIMEOUT_SECONDS = int(os.environ.get("MAINLINE_REFRESH_CHAIN_TIMEOUT_SECONDS", "1200") or "1200")
BEIJING_TZ = ZoneInfo("Asia/Shanghai")
NIGHT_BACKLOG_REPORT = PROJECT_ROOT / "reports" / "incremental_night_search_backlog_latest.json"
NIGHT_DISPATCH_PLAN_REPORT = PROJECT_ROOT / "reports" / "night_window_dispatch_plan_latest.json"
NIGHTLY_PARAM_SEARCH_LAUNCHER_REPORT = PROJECT_ROOT / "reports" / "nightly_param_search_launcher_latest.json"
JOINT_DIRECTION_LAUNCHER_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "run_nightly_joint_direction_launcher.py"
JOINT_DIRECTION_LAUNCHER_REPORT = PROJECT_ROOT / "reports" / "joint_direction_nightly_launcher_latest.json"


def next_run_at(hour: int) -> float:
    """下次运行时间戳（本地时间当天或明天该小时 0 分 0 秒）。"""
    from datetime import datetime as dt
    now = dt.now()
    today = now.replace(hour=hour % 24, minute=0, second=0, microsecond=0)
    if now >= today:
        from datetime import timedelta
        today += timedelta(days=1)
    return today.timestamp()


def next_run_every(interval_minutes: int, offset_minutes: int = 0) -> float:
    """按固定间隔分钟调度，且在每个间隔上偏移 offset 分钟后触发。"""
    step_sec = max(1, int(interval_minutes)) * 60
    offset_sec = (max(0, int(offset_minutes)) % max(1, int(interval_minutes))) * 60
    now = time.time()
    ts = ((int(now) - offset_sec) // step_sec + 1) * step_sec + offset_sec
    return float(ts)


def _python_has_ml_deps(python_bin: str) -> bool:
    try:
        p = subprocess.run(
            [python_bin, "-c", "import torch, lightgbm"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=8,
        )
        return p.returncode == 0
    except Exception:
        return False


def resolve_training_python() -> str:
    """
    选择可用训练解释器（优先带 torch/lightgbm）。
    可通过 TRAIN_PYTHON 显式指定。
    """
    env_py = os.environ.get("TRAIN_PYTHON", "").strip()
    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    conda_python_exe = os.environ.get("CONDA_PYTHON_EXE", "").strip()
    candidates: list[str] = []
    if env_py:
        candidates.append(env_py)
    if conda_prefix:
        candidates.extend(
            [
                str(Path(conda_prefix) / "bin" / "python"),
                str(Path(conda_prefix) / "bin" / "python3"),
            ]
        )
    if conda_python_exe:
        candidates.append(conda_python_exe)
    candidates.extend(
        [
            sys.executable,
            "/Users/mac/miniforge3/bin/python3",
            "/opt/homebrew/Caskroom/miniforge/base/bin/python3",
            "python3",
        ]
    )
    seen: set[str] = set()
    for py in candidates:
        if not py or py in seen:
            continue
        seen.add(py)
        if _python_has_ml_deps(py):
            return py
    return sys.executable


def _report_is_stale(path: Path, stale_hours: int) -> bool:
    if not path.exists():
        return True
    try:
        age_hours = (time.time() - path.stat().st_mtime) / 3600.0
    except Exception:
        return True
    return age_hours >= float(max(1, stale_hours))


def _run_mainline_refresh_chain(python_bin: str, context_label: str) -> bool:
    detail = _run_mainline_refresh_chain_with_detail(python_bin, context_label)
    return bool(detail.get("success"))


def _run_mainline_refresh_chain_with_detail(
    python_bin: str,
    context_label: str,
    *,
    skip_labels: Optional[list[str]] = None,
) -> dict[str, Any]:
    started_at = time.time()
    if not MAINLINE_REFRESH_CHAIN_SCRIPT.exists():
        print(f"[{datetime.now().isoformat()}] {context_label} 失败：缺少共用刷新链脚本 {MAINLINE_REFRESH_CHAIN_SCRIPT}", flush=True)
        return {
            "success": False,
            "duration_seconds": round(max(0.0, time.time() - started_at), 3),
            "timeout": False,
            "returncode": None,
            "skip_labels": list(skip_labels or []),
        }
    cmd = [python_bin, "-B", str(MAINLINE_REFRESH_CHAIN_SCRIPT), "--python-bin", python_bin]
    for label in skip_labels or []:
        cmd.extend(["--skip-label", str(label)])
    try:
        ret = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            check=False,
            timeout=MAINLINE_REFRESH_CHAIN_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        print(
            f"[{datetime.now().isoformat()}] {context_label} 超时：timeout={MAINLINE_REFRESH_CHAIN_TIMEOUT_SECONDS}s",
            flush=True,
        )
        return {
            "success": False,
            "duration_seconds": round(max(0.0, time.time() - started_at), 3),
            "timeout": True,
            "returncode": None,
            "skip_labels": list(skip_labels or []),
        }
    if ret.returncode != 0:
        print(f"[{datetime.now().isoformat()}] {context_label} 失败：rc={ret.returncode}", flush=True)
        return {
            "success": False,
            "duration_seconds": round(max(0.0, time.time() - started_at), 3),
            "timeout": False,
            "returncode": int(ret.returncode),
            "skip_labels": list(skip_labels or []),
        }
    return {
        "success": True,
        "duration_seconds": round(max(0.0, time.time() - started_at), 3),
        "timeout": False,
        "returncode": int(ret.returncode),
        "skip_labels": list(skip_labels or []),
    }


def _append_online_history_event(payload: dict[str, Any]) -> None:
    ONLINE_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with ONLINE_HISTORY_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass


def _read_latest_grouped_cycle_summary(profile_path: Path) -> dict[str, Any]:
    if not ONLINE_HISTORY_FILE.exists():
        return {}
    expected = profile_path.name
    try:
        lines = ONLINE_HISTORY_FILE.read_text(encoding="utf-8").splitlines()
    except Exception:
        return {}
    for raw in reversed(lines):
        try:
            row = json.loads(raw)
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        if str(row.get("entry_type") or "") != "grouped_online_learning":
            continue
        profile_json = str(row.get("profile_json") or "")
        if not profile_json.endswith(expected):
            continue
        summary = row.get("summary")
        return summary if isinstance(summary, dict) else {}
    return {}


def _parse_history_ts(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _param_search_history_ts(row: dict[str, Any]) -> datetime | None:
    return (
        _parse_history_ts(row.get("timestamp"))
        or _parse_history_ts(row.get("finished_at"))
        or _parse_history_ts(row.get("generated_at"))
    )


def _beijing_now() -> datetime:
    return datetime.now(timezone.utc).astimezone(BEIJING_TZ)


def _night_search_window_state(now_local: datetime | None = None) -> dict[str, Any]:
    local_now = now_local or _beijing_now()
    start = local_now.replace(
        hour=DEFAULT_NIGHT_SEARCH_START_HOUR,
        minute=0,
        second=0,
        microsecond=0,
    )
    end = local_now.replace(
        hour=DEFAULT_NIGHT_SEARCH_END_HOUR,
        minute=0,
        second=0,
        microsecond=0,
    )
    active = start <= local_now < end
    if active:
        next_window_start = start + timedelta(days=1)
        next_window_end = end
    elif local_now < start:
        next_window_start = start
        next_window_end = end
    else:
        next_window_start = start + timedelta(days=1)
        next_window_end = end + timedelta(days=1)
    return {
        "night_search_window_active": bool(active),
        "night_search_local_now": local_now.isoformat(),
        "night_search_window_start_at": start.astimezone(timezone.utc).isoformat(),
        "night_search_window_end_at": end.astimezone(timezone.utc).isoformat(),
        "night_search_next_window_start_at": next_window_start.astimezone(timezone.utc).isoformat(),
        "night_search_next_window_end_at": next_window_end.astimezone(timezone.utc).isoformat(),
    }


def _load_night_search_backlog() -> dict[str, Any]:
    payload = _read_json_file(NIGHT_BACKLOG_REPORT)
    rows = payload.get("rows")
    queued_rows = [
        row for row in rows
        if isinstance(row, dict) and bool(row.get("queuedForNightSearch"))
    ] if isinstance(rows, list) else []
    queued_keys = [
        str(row.get("familySymbolKey") or "").strip()
        for row in queued_rows
        if str(row.get("familySymbolKey") or "").strip()
    ]
    return {
        "payload": payload,
        "queued_keys": queued_keys,
        "queued_total": len(queued_keys),
    }


def _nightly_launcher_state_for_scheduler(param_state: dict[str, Any]) -> dict[str, Any]:
    payload = _read_json_file(NIGHTLY_PARAM_SEARCH_LAUNCHER_REPORT)
    generated = _parse_history_ts(payload.get("generated_at"))
    window_start = _parse_history_ts(param_state.get("night_search_window_start_at"))
    local_now = _parse_history_ts(param_state.get("night_search_local_now"))
    status = str(payload.get("status") or "").strip()
    runtime_root = str(payload.get("runtimeRoot") or "")
    command = payload.get("command") if isinstance(payload.get("command"), list) else []
    command_text = " ".join(str(item) for item in command)
    old_root_seen = "/Users/mac/Downloads/polyfun" in command_text or "/Users/mac/Downloads/polyfun" in runtime_root
    generated_after_window_start = bool(generated and window_start and generated >= window_start)
    launcher_status_ok = status in {"launched", "already_running"}
    fallback_reason = ""
    if not payload:
        fallback_reason = "missing_nightly_launcher_report"
    elif runtime_root != str(PROJECT_ROOT):
        fallback_reason = "nightly_launcher_wrong_runtime_root"
    elif old_root_seen:
        fallback_reason = "nightly_launcher_command_uses_old_root"
    elif not generated_after_window_start:
        fallback_reason = "nightly_launcher_report_not_from_current_window"
    elif not launcher_status_ok:
        fallback_reason = f"nightly_launcher_status_{status or 'unknown'}"
    if local_now and window_start and not generated_after_window_start:
        minutes_after_window_start = (local_now - window_start).total_seconds() / 60.0
        if minutes_after_window_start < DEFAULT_NIGHTLY_LAUNCHER_GRACE_MINUTES:
            fallback_reason = ""
    return {
        "reportPath": str(NIGHTLY_PARAM_SEARCH_LAUNCHER_REPORT),
        "status": status,
        "runtimeRoot": runtime_root,
        "generatedAt": generated.isoformat() if generated else "",
        "generatedAfterWindowStart": generated_after_window_start,
        "launcherStatusOk": launcher_status_ok,
        "oldRootSeen": old_root_seen,
        "fallbackRequired": bool(fallback_reason),
        "fallbackReason": fallback_reason,
    }


def _param_search_status_clears_backlog(status: str) -> bool:
    return str(status or "").strip() in {
        "completed_recommendation_only",
        "completed_apply_eligible",
        "partial_checkpointed",
    }


def _param_search_status_finished(status: str) -> bool:
    return _param_search_status_clears_backlog(status) or str(status or "").strip() == "failed_non_blocking"


def _param_search_cadence_state(status: str, launch_reason: str, due: bool, backlog_windows: int) -> str:
    status_text = str(status or "").strip()
    launch_reason_text = str(launch_reason or "").strip()
    backlog = max(0, int(backlog_windows))
    if launch_reason_text == "night_backlog_drain":
        if status_text == "paused":
            return "paused"
        if status_text == "failed_non_blocking":
            return "failed_non_blocking"
        if status_text == "already_running":
            return "night_backlog_running"
        if due:
            return "night_backlog_due"
        return "night_window_idle"
    if status_text == "paused":
        return "paused"
    if status_text == "failed_non_blocking":
        return "failed_non_blocking"
    if status_text == "already_running":
        return "catchup_running" if launch_reason_text == "catchup" or backlog >= 2 else "running_due_slice"
    if status_text in {"queued", "running", "partial_checkpointed", "completed_recommendation_only", "completed_apply_eligible"}:
        if due:
            return "catchup_running" if launch_reason_text == "catchup" or backlog >= 2 else "running_due_slice"
    if due:
        return "catchup_due" if backlog >= 2 else "scheduled_due"
    return "not_due"


def _write_param_search_runtime_status_snapshot(
    status: str,
    state: dict[str, Any],
    *,
    pid: int | None = None,
    slices_completed: int | None = None,
) -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    existing = _read_json_file(INCREMENTAL_PARAM_SEARCH_RUNTIME_STATUS)
    param_report = _read_json_file(PROJECT_ROOT / "reports" / "core10_incremental_param_search_latest.json")
    param_apply = _read_json_file(PROJECT_ROOT / "reports" / "core10_incremental_param_apply_latest.json")
    resolved_pid = int(pid or state.get("param_search_runtime_pid") or 0)
    launch_reason = str(state.get("param_search_launch_reason") or "")
    due = bool(state.get("param_search_due"))
    backlog_windows = int(state.get("param_search_backlog_windows") or 0)
    effective_status = str(status or "").strip() or "not_due"
    canonical_backlog = _load_night_search_backlog()
    canonical_backlog_total = int(canonical_backlog.get("queued_total") or 0)
    completed_slices = int(slices_completed if slices_completed is not None else state.get("param_search_slices_completed") or 0)
    canonical_backlog_remaining = max(0, canonical_backlog_total - completed_slices)
    payload = {
        "generated_at": now_iso,
        "status": effective_status,
        "entry_type": "incremental_param_search_nonblocking",
        "pid": resolved_pid if resolved_pid > 0 else None,
        "current_conclusion": f"incremental_param_search_nonblocking_{effective_status}",
        "training_cadence_minutes": DEFAULT_TRAINING_CADENCE_MINUTES,
        "param_search_cadence_minutes": int(state.get("param_search_cadence_minutes") or DEFAULT_PARAM_SEARCH_CADENCE_MINUTES),
        "cadence_contract": "training_every_30m_param_search_nightly_0000_start_until_complete_asia_shanghai",
        "param_search_does_not_gate_training": True,
        "runtime_apply_reads_stable_manifest_only": True,
        "param_search_freshness_tradeoff_vs_30m_search": "night_window_backlog_drain_replaces_daytime_30m_search",
        "param_search_due": due,
        "param_search_next_due_at": str(state.get("param_search_next_due_at") or ""),
        "param_search_backlog_windows": backlog_windows,
        "param_search_launch_reason": launch_reason,
        "param_search_slices_requested": int(state.get("param_search_slices_requested") or 0),
        "param_search_slices_completed": completed_slices,
        "param_search_last_finished_at": str(state.get("param_search_last_finished_at") or ""),
        "param_search_last_progress_at": str(state.get("param_search_last_progress_at") or ""),
        "param_search_latest_finished_status": str(state.get("param_search_latest_finished_status") or ""),
        "param_search_pause_reason": str(state.get("param_search_pause_reason") or ""),
        "param_search_paused_for_lowprice_full_tuning": bool(state.get("param_search_paused_for_lowprice_full_tuning")),
        "param_search_family_symbol_keys": list(state.get("param_search_family_symbol_keys") or []),
        "param_search_cadence_state": _param_search_cadence_state(
            effective_status,
            launch_reason,
            due,
            backlog_windows,
        ),
        "night_search_window_active": bool(state.get("night_search_window_active")),
        "night_search_backlog_total": canonical_backlog_total,
        "night_search_backlog_remaining": canonical_backlog_remaining,
        "night_search_last_family_symbol": str(state.get("night_search_last_family_symbol") or ""),
        "night_search_stop_reason": str(state.get("night_search_stop_reason") or ""),
        "night_search_local_now": str(state.get("night_search_local_now") or ""),
        "night_search_window_start_at": str(state.get("night_search_window_start_at") or ""),
        "night_search_window_end_at": str(state.get("night_search_window_end_at") or ""),
        "night_search_next_window_start_at": str(state.get("night_search_next_window_start_at") or ""),
        "night_search_next_window_end_at": str(state.get("night_search_next_window_end_at") or ""),
        "night_search_canonical_backlog_total": canonical_backlog_total,
        "nightly_launcher_report_path": str(state.get("nightly_launcher_report_path") or ""),
        "nightly_launcher_status": str(state.get("nightly_launcher_status") or ""),
        "nightly_launcher_runtime_root": str(state.get("nightly_launcher_runtime_root") or ""),
        "nightly_launcher_fallback_required": bool(state.get("nightly_launcher_fallback_required")),
        "nightly_launcher_fallback_reason": str(state.get("nightly_launcher_fallback_reason") or ""),
        "schedulerFallbackLaunched": bool(state.get("schedulerFallbackLaunched")),
        "param_search_report_status": str(param_report.get("status") or ""),
        "param_search_report_family_symbol_keys_total": int((param_report.get("summary") or {}).get("families_total") or 0),
        "param_search_report_multi_candidate_ready_total": int(
            (param_report.get("summary") or {}).get("families_with_multi_candidate_comparison") or 0
        ),
        "param_apply_ready_to_apply_total": int((param_apply.get("summary") or {}).get("param_apply_ready_to_apply_total") or 0),
        "param_apply_applied_total": int((param_apply.get("summary") or {}).get("param_apply_applied_total") or 0),
    }
    started_at = existing.get("started_at")
    if effective_status == "already_running" and started_at:
        payload["started_at"] = started_at
    elif effective_status in {"not_due", "skipped_not_training_safe"}:
        payload["finished_at"] = now_iso
    _write_json_file(INCREMENTAL_PARAM_SEARCH_RUNTIME_STATUS, payload)


def _load_latest_param_search_state(
    *,
    cadence_minutes: int,
    max_catchup_slices_per_launch: int,
) -> dict[str, Any]:
    runtime_status = _read_json_file(INCREMENTAL_PARAM_SEARCH_RUNTIME_STATUS)
    running_pid = int(runtime_status.get("pid") or 0)
    runtime_status_name = str(runtime_status.get("status") or "").strip()
    running = runtime_status_name in {"queued", "running"} and _pid_alive(running_pid)

    latest_finished_row: dict[str, Any] = {}
    latest_finished_ts: datetime | None = None
    latest_progress_row: dict[str, Any] = {}
    latest_progress_ts: datetime | None = None
    if ONLINE_HISTORY_FILE.exists():
        try:
            lines = ONLINE_HISTORY_FILE.read_text(encoding="utf-8").splitlines()
        except Exception:
            lines = []
        for raw in reversed(lines):
            try:
                row = json.loads(raw)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            if str(row.get("entry_type") or "") != "incremental_param_search_nonblocking":
                continue
            row_status = str(row.get("status") or "").strip()
            row_ts = _param_search_history_ts(row)
            if row_ts is None:
                continue
            if not latest_finished_row and _param_search_status_finished(row_status):
                latest_finished_row = row
                latest_finished_ts = row_ts
            if not latest_progress_row and _param_search_status_clears_backlog(row_status):
                latest_progress_row = row
                latest_progress_ts = row_ts
            if latest_finished_row and latest_progress_row:
                break

    cadence_minutes = max(1, int(cadence_minutes))
    window_state = _night_search_window_state()
    backlog_state = _load_night_search_backlog()
    dispatch_plan = write_dispatch_plan()
    dispatch_summary = dispatch_plan.get("summary") if isinstance(dispatch_plan.get("summary"), dict) else {}
    night_keys = [
        str(item).strip().lower()
        for item in (dispatch_summary.get("searchFamilySymbolKeys") or [])
        if str(item).strip()
    ]
    night_total = len(night_keys)
    due = bool(window_state.get("night_search_window_active")) and night_total > 0
    launch_reason = "night_backlog_drain" if due else ""
    slices_requested = min(night_total, max(1, int(max_catchup_slices_per_launch))) if due else 0
    next_due_at = (
        str(window_state.get("night_search_next_window_start_at") or "")
        if not due else ""
    )
    night_stop_reason = ""
    if not bool(window_state.get("night_search_window_active")):
        night_stop_reason = "outside_night_window"
    elif night_total <= 0:
        night_stop_reason = "empty_backlog"

    state = {
        "param_search_running": bool(running),
        "param_search_runtime_status": runtime_status_name or ("running" if running else ""),
        "param_search_runtime_pid": running_pid if running else None,
        "param_search_cadence_minutes": cadence_minutes,
        "param_search_due": bool(due),
        "param_search_next_due_at": next_due_at,
        "param_search_backlog_windows": night_total,
        "param_search_launch_reason": launch_reason,
        "param_search_slices_requested": int(slices_requested),
        "param_search_family_symbol_keys": night_keys[:slices_requested] if slices_requested > 0 else [],
        "param_search_last_finished_at": (
            latest_finished_ts.isoformat() if latest_finished_ts is not None else ""
        ),
        "param_search_last_progress_at": (
            latest_progress_ts.isoformat() if latest_progress_ts is not None else ""
        ),
        "param_search_latest_finished_status": str(latest_finished_row.get("status") or ""),
        "night_search_window_active": bool(window_state.get("night_search_window_active")),
        "night_search_backlog_total": night_total,
        "night_search_backlog_remaining": night_total,
        "night_search_last_family_symbol": "",
        "night_search_stop_reason": night_stop_reason,
        "night_search_local_now": str(window_state.get("night_search_local_now") or ""),
        "night_search_window_start_at": str(window_state.get("night_search_window_start_at") or ""),
        "night_search_window_end_at": str(window_state.get("night_search_window_end_at") or ""),
        "night_search_next_window_start_at": str(window_state.get("night_search_next_window_start_at") or ""),
        "night_search_next_window_end_at": str(window_state.get("night_search_next_window_end_at") or ""),
        "night_dispatch_plan_report": str(NIGHT_DISPATCH_PLAN_REPORT),
        "night_dispatch_carry_forward_total": int(dispatch_summary.get("carryForwardUnchangedTotal") or 0),
        "night_dispatch_local_search_total": int(dispatch_summary.get("localSearchOnlyTotal") or 0),
        "night_dispatch_full_search_total": int(dispatch_summary.get("escalatedFullSearchTotal") or 0),
        "night_search_canonical_backlog_total": int(backlog_state.get("queued_total") or 0),
    }
    launcher_state = _nightly_launcher_state_for_scheduler(state)
    state.update(
        {
            "nightly_launcher_report_path": str(launcher_state.get("reportPath") or ""),
            "nightly_launcher_status": str(launcher_state.get("status") or ""),
            "nightly_launcher_runtime_root": str(launcher_state.get("runtimeRoot") or ""),
            "nightly_launcher_generated_at": str(launcher_state.get("generatedAt") or ""),
            "nightly_launcher_fallback_required": bool(launcher_state.get("fallbackRequired")),
            "nightly_launcher_fallback_reason": str(launcher_state.get("fallbackReason") or ""),
        }
    )
    return state


def _launch_nonblocking_param_search(python_bin: str, context_label: str, dispatch: dict[str, Any]) -> dict[str, Any]:
    if not INCREMENTAL_PARAM_SEARCH_NONBLOCKING_RUNNER.exists():
        print(f"[{datetime.now().isoformat()}] {context_label}: 缺少异步推荐 runner {INCREMENTAL_PARAM_SEARCH_NONBLOCKING_RUNNER}", flush=True)
        return {"launched": False, "status": "missing_runner", "pid": None}
    cmd = [
        python_bin,
        "-B",
        str(INCREMENTAL_PARAM_SEARCH_NONBLOCKING_RUNNER),
        "--python-bin",
        python_bin,
        "--cadence-minutes",
        str(dispatch.get("param_search_cadence_minutes") or DEFAULT_PARAM_SEARCH_CADENCE_MINUTES),
        "--due",
        "1" if bool(dispatch.get("param_search_due")) else "0",
        "--next-due-at",
        str(dispatch.get("param_search_next_due_at") or ""),
        "--backlog-windows",
        str(dispatch.get("param_search_backlog_windows") or 0),
        "--launch-reason",
        str(dispatch.get("param_search_launch_reason") or ""),
        "--slices-requested",
        str(dispatch.get("param_search_slices_requested") or 0),
        "--last-finished-at",
        str(dispatch.get("param_search_last_finished_at") or ""),
        "--night-search-window-active",
        "1" if bool(dispatch.get("night_search_window_active")) else "0",
        "--night-search-backlog-total",
        str(dispatch.get("night_search_backlog_total") or 0),
        "--night-search-stop-reason",
        str(dispatch.get("night_search_stop_reason") or ""),
        "--max-member-workers",
        str(DEFAULT_PARAM_SEARCH_SAFE_MAX_MEMBER_WORKERS),
        "--max-candidate-workers",
        str(DEFAULT_PARAM_SEARCH_SAFE_MAX_CANDIDATE_WORKERS),
        "--numeric-threads",
        str(DEFAULT_PARAM_SEARCH_SAFE_NUMERIC_THREADS),
        "--family-concurrency",
        str(DEFAULT_PARAM_SEARCH_SAFE_FAMILY_CONCURRENCY),
    ]
    for family_symbol_key in dispatch.get("param_search_family_symbol_keys") or []:
        if str(family_symbol_key).strip():
            cmd.extend(["--family-symbol-key", str(family_symbol_key).strip()])
    ret = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    status_payload = _read_json_file(INCREMENTAL_PARAM_SEARCH_RUNTIME_STATUS)
    status = str(status_payload.get("status") or "unknown")
    pid = status_payload.get("pid")
    if ret.returncode != 0:
        print(f"[{datetime.now().isoformat()}] {context_label}: 启动失败 rc={ret.returncode}", flush=True)
        return {"launched": False, "status": "failed_non_blocking", "pid": pid}
    print(
        f"[{datetime.now().isoformat()}] {context_label}: status={status} pid={pid or '-'} "
        f"due={bool(status_payload.get('param_search_due'))} "
        f"night_window={bool(status_payload.get('night_search_window_active'))} "
        f"backlog={int(status_payload.get('night_search_backlog_total') or 0)} "
        f"launch_reason={status_payload.get('param_search_launch_reason') or '-'} "
        f"slices={int(status_payload.get('param_search_slices_requested') or 0)} "
        f"last_key={status_payload.get('night_search_last_family_symbol') or '-'}",
        flush=True,
    )
    return {
        "launched": True,
        "status": status,
        "pid": pid,
        "param_search_due": bool(status_payload.get("param_search_due")),
        "param_search_next_due_at": status_payload.get("param_search_next_due_at"),
        "param_search_backlog_windows": int(status_payload.get("param_search_backlog_windows") or 0),
        "param_search_launch_reason": str(status_payload.get("param_search_launch_reason") or ""),
        "param_search_slices_requested": int(status_payload.get("param_search_slices_requested") or 0),
        "param_search_slices_completed": int(status_payload.get("param_search_slices_completed") or 0),
        "param_search_last_finished_at": str(status_payload.get("param_search_last_finished_at") or ""),
        "night_search_window_active": bool(status_payload.get("night_search_window_active")),
        "night_search_backlog_total": int(status_payload.get("night_search_backlog_total") or 0),
        "night_search_backlog_remaining": int(status_payload.get("night_search_backlog_remaining") or 0),
        "night_search_last_family_symbol": str(status_payload.get("night_search_last_family_symbol") or ""),
        "night_search_stop_reason": str(status_payload.get("night_search_stop_reason") or ""),
    }


def _launch_joint_direction_nightly(python_bin: str, context_label: str) -> dict[str, Any]:
    if not JOINT_DIRECTION_LAUNCHER_SCRIPT.exists():
        print(f"[{datetime.now().isoformat()}] {context_label}: 缺少联合方向 launcher {JOINT_DIRECTION_LAUNCHER_SCRIPT}", flush=True)
        return {"launched": False, "status": "missing_runner", "pid": None}
    state = _read_json_file(JOINT_DIRECTION_LAUNCHER_REPORT)
    state_pid = int(state.get("pid") or 0) if isinstance(state, dict) else 0
    state_status = str(state.get("status") or "") if isinstance(state, dict) else ""
    if state_status == "running" and _pid_alive(state_pid):
        return {"launched": False, "status": "already_running", "pid": state_pid}
    env = os.environ.copy()
    env["POLYFUN_REPO_ROOT"] = str(PROJECT_ROOT)
    env["PROJECT_RUNTIME_ROOT"] = str(PROJECT_ROOT)
    proc = subprocess.Popen(
        [python_bin, "-B", str(JOINT_DIRECTION_LAUNCHER_SCRIPT), "--python-bin", python_bin],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    print(f"[{datetime.now().isoformat()}] {context_label}: 已启动联合方向 launcher pid={proc.pid}", flush=True)
    return {"launched": True, "status": "launched", "pid": int(proc.pid), "report": str(JOINT_DIRECTION_LAUNCHER_REPORT)}


def _launch_nonblocking_mainline_refresh(python_bin: str, context_label: str) -> dict[str, Any]:
    if not MAINLINE_REFRESH_NONBLOCKING_RUNNER.exists():
        print(f"[{datetime.now().isoformat()}] {context_label}: 缺少异步 refresh runner {MAINLINE_REFRESH_NONBLOCKING_RUNNER}", flush=True)
        return {"launched": False, "status": "missing_runner", "pid": None}
    ret = subprocess.run(
        [python_bin, "-B", str(MAINLINE_REFRESH_NONBLOCKING_RUNNER), "--python-bin", python_bin],
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    status_payload = _read_json_file(PROJECT_ROOT / "reports" / "mainline_refresh_nonblocking_runtime_status_latest.json")
    status = str(status_payload.get("status") or "unknown")
    pid = status_payload.get("pid")
    if ret.returncode != 0:
        print(f"[{datetime.now().isoformat()}] {context_label}: 启动失败 rc={ret.returncode}", flush=True)
        return {"launched": False, "status": "failed_non_blocking", "pid": pid}
    print(f"[{datetime.now().isoformat()}] {context_label}: status={status} pid={pid or '-'}", flush=True)
    return {"launched": True, "status": status, "pid": pid}


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _pid_command(pid: int) -> str:
    if pid <= 0:
        return ""
    try:
        p = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return ""
    if p.returncode != 0:
        return ""
    return (p.stdout or "").strip()


def _is_scheduler_pid(pid: int) -> bool:
    cmd = _pid_command(pid)
    return "run_daily_training_scheduler.py" in cmd


def _load_profile_assets(profile_path: Path) -> list[str]:
    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    assets: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        asset = str(row.get("asset") or "").strip().upper()
        if asset and asset not in assets:
            assets.append(asset)
    return assets


def acquire_scheduler_pid(pid_file: Path) -> tuple[bool, int | None]:
    """单实例保护：已有活跃调度器时拒绝重复启动。"""
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    current_pid = os.getpid()

    if pid_file.exists():
        try:
            old_pid = int(pid_file.read_text(encoding="utf-8").strip() or "0")
        except (ValueError, OSError):
            old_pid = 0
        if old_pid > 0 and old_pid != current_pid and _pid_alive(old_pid):
            if _is_scheduler_pid(old_pid):
                return False, old_pid
            # PID 复用到其他进程，视为陈旧 PID 文件
            try:
                pid_file.unlink()
            except OSError:
                return False, None

    tmp = pid_file.with_suffix(".pid.tmp")
    try:
        tmp.write_text(str(current_pid), encoding="utf-8")
        os.replace(tmp, pid_file)
    except OSError:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        return False, None
    return True, None


def release_scheduler_pid(pid_file: Path) -> None:
    """仅清理当前实例写入的 PID 文件。"""
    try:
        if not pid_file.exists():
            return
        txt = pid_file.read_text(encoding="utf-8").strip()
        if txt and int(txt) == os.getpid():
            pid_file.unlink()
    except (OSError, ValueError):
        pass


def acquire_run_lock(lock_file: Path) -> int | None:
    """获取互斥锁。若已有活跃锁则返回 None；若为僵尸锁会自动清理。"""
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    if lock_file.exists():
        try:
            old_pid = int(lock_file.read_text(encoding="utf-8").strip() or "0")
        except (ValueError, OSError):
            old_pid = 0
        if _pid_alive(old_pid) and _is_scheduler_pid(old_pid):
            return None
        try:
            lock_file.unlink()
        except OSError:
            return None

    try:
        fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    os.write(fd, str(os.getpid()).encode("utf-8"))
    return fd


def release_run_lock(fd: int | None, lock_file: Path) -> None:
    if fd is not None:
        try:
            os.close(fd)
        except OSError:
            pass
    try:
        if lock_file.exists():
            lock_file.unlink()
    except OSError:
        pass


def cleanup_stale_run_lock(lock_file: Path) -> bool:
    """
    清理陈旧的训练互斥锁：
    - lock PID 不存在
    - lock PID 被复用到非调度器进程
    返回是否发生清理。
    """
    if not lock_file.exists():
        return False
    try:
        txt = lock_file.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    try:
        pid = int(txt) if txt else 0
    except ValueError:
        pid = 0

    stale = False
    if pid <= 0:
        stale = True
    elif not _pid_alive(pid):
        stale = True
    elif not _is_scheduler_pid(pid):
        stale = True

    if not stale:
        return False
    try:
        lock_file.unlink()
        return True
    except OSError:
        return False


def _parse_iso_to_ts(ts_text: str) -> float | None:
    text = str(ts_text or "").strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt.astimezone(timezone.utc).timestamp()


def _history_entry_has_effective_success(obj: dict[str, Any]) -> bool:
    """
    判定该历史条目是否为“有效成功”训练：
    - 优先使用 summary.success > 0
    - slot1 专属正增益门下，训练完成但被科学 veto 也算训练覆盖完成
    - 兼容旧格式：results 中至少一项 success=true
    """
    summary = obj.get("summary")
    if isinstance(summary, dict):
        try:
            return int(summary.get("success", 0) or 0) > 0 or int(summary.get("trained_but_not_written", 0) or 0) > 0
        except (TypeError, ValueError):
            return False

    results = obj.get("results")
    if isinstance(results, list):
        for it in results:
            if isinstance(it, dict) and bool(it.get("success")):
                return True
    return False


def _read_training_history_timestamps(path: Path, max_lines: int = DEFAULT_COVERAGE_HISTORY_MAX_LINES) -> list[float]:
    if not path.exists():
        return []
    lines = deque(maxlen=max(200, int(max_lines)))
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    lines.append(s)
    except OSError:
        return []

    ts_list: list[float] = []
    for line in lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        # 覆盖补训基于主训练链路（Exp）判定，避免仅 GRU 成功掩盖 Exp 断档。
        if str(obj.get("type", "")).strip().lower() == "gru":
            continue
        if not _history_entry_has_effective_success(obj):
            continue
        ts = _parse_iso_to_ts(str(obj.get("timestamp", "")))
        if ts is not None:
            ts_list.append(ts)
    return sorted(ts_list)


def _slot_floor(ts: float, interval_minutes: int, offset_minutes: int) -> int:
    step = max(1, int(interval_minutes)) * 60
    offset = (max(0, int(offset_minutes)) % max(1, int(interval_minutes))) * 60
    return int(((int(ts) - offset) // step) * step + offset)


def analyze_hourly_coverage(
    history_file: Path,
    interval_minutes: int,
    offset_minutes: int,
    lookback_hours: int,
    grace_minutes: int,
    max_history_lines: int = DEFAULT_COVERAGE_HISTORY_MAX_LINES,
    now_ts: float | None = None,
) -> dict[str, Any]:
    now = float(now_ts if now_ts is not None else time.time())
    step = max(1, int(interval_minutes)) * 60
    lookback_sec = max(1, int(lookback_hours)) * 3600
    grace_sec = max(0, int(grace_minutes)) * 60
    due_age_sec = step + grace_sec
    start_ts = now - lookback_sec

    history_ts = _read_training_history_timestamps(history_file, max_lines=max_history_lines)
    covered_slots = {
        _slot_floor(ts, interval_minutes, offset_minutes)
        for ts in history_ts
        if ts >= start_ts - step
    }

    first_slot = _slot_floor(start_ts, interval_minutes, offset_minutes)
    while first_slot < start_ts:
        first_slot += step

    expected_slots: list[int] = []
    missing_slots_total: list[int] = []
    slot = first_slot
    while slot <= int(now):
        if now - slot > due_age_sec:
            expected_slots.append(slot)
            if slot not in covered_slots:
                missing_slots_total.append(slot)
        slot += step

    latest_ts = max(history_ts) if history_ts else None
    latest_age_minutes = ((now - latest_ts) / 60.0) if latest_ts is not None else None
    latest_slot = _slot_floor(latest_ts, interval_minutes, offset_minutes) if latest_ts is not None else None

    # 历史缺口（发生在最新成功训练槽位之前）无法通过“现在补跑”真正回填，自动归档。
    actionable_missing_slots: list[int] = []
    historical_missing_slots: list[int] = []
    for ts in missing_slots_total:
        if latest_slot is not None and ts < latest_slot:
            historical_missing_slots.append(ts)
        else:
            actionable_missing_slots.append(ts)

    needs_catchup = latest_ts is None or (now - latest_ts) > due_age_sec

    return {
        "checked_at": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
        "interval_minutes": int(interval_minutes),
        "offset_minutes": int(offset_minutes),
        "lookback_hours": int(lookback_hours),
        "grace_minutes": int(grace_minutes),
        "due_age_minutes": round(due_age_sec / 60.0, 2),
        "expected_slots": len(expected_slots),
        # 覆盖率口径按“可行动缺口(actionable)”计算，历史缺口单独归档展示。
        "covered_slots": len(expected_slots) - len(actionable_missing_slots),
        "missing_slots": len(actionable_missing_slots),
        "missing_slots_total": len(missing_slots_total),
        "historical_missing_slots": len(historical_missing_slots),
        "missing_slot_timestamps": [
            datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() for ts in actionable_missing_slots
        ],
        "historical_missing_slot_timestamps": [
            datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() for ts in historical_missing_slots
        ],
        "latest_training_at": (
            datetime.fromtimestamp(latest_ts, tz=timezone.utc).isoformat() if latest_ts is not None else None
        ),
        "latest_training_age_minutes": (round(latest_age_minutes, 2) if latest_age_minutes is not None else None),
        "needs_catchup": bool(needs_catchup),
    }


def write_coverage_report(report_file: Path, report: dict[str, Any]) -> None:
    try:
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser(description="每日定时训练调度（不依赖 crontab）")
    ap.add_argument("hour", nargs="?", type=int, default=0, help="每天几点跑，0-23，默认 0")
    ap.add_argument(
        "wait_minutes",
        nargs="?",
        type=int,
        default=DEFAULT_WAIT_BEFORE_MONITOR_MINUTES,
        help=f"训练完成后等待几分钟再跑监控（便于检测热重载与最新预测），默认 {DEFAULT_WAIT_BEFORE_MONITOR_MINUTES}",
    )
    ap.add_argument(
        "--interval-minutes",
        type=int,
        default=30,
        help="按固定间隔分钟执行（如 30 或 45）；>0 时忽略 hour 参数；默认 30",
    )
    ap.add_argument(
        "--interval-offset-minutes",
        type=int,
        default=DEFAULT_INTERVAL_OFFSET_MINUTES,
        help=f"间隔模式下每轮偏移分钟，避免整点抢资源（默认 {DEFAULT_INTERVAL_OFFSET_MINUTES}）",
    )
    ap.add_argument("--hours", type=int, default=48, help="在线增量训练使用最近 N 小时数据（默认 48）")
    ap.add_argument("--base-rounds", type=int, default=12, help="普通波动增量树数（默认 12）")
    ap.add_argument("--calm-rounds", type=int, default=8, help="平稳波动增量树数（默认 8）")
    ap.add_argument("--shock-rounds", type=int, default=20, help="冲击波动增量树数（默认 20）")
    ap.add_argument("--recency-halflife-bars", type=int, default=64, help="近期样本半衰期（15m bars，默认 64）")
    ap.add_argument("--shock-weight-mult", type=float, default=1.8, help="冲击样本加权倍数（默认 1.8）")
    ap.add_argument("--calm-vol-ratio", type=float, default=1.08, help="平稳判定波动比阈值（默认 1.08）")
    ap.add_argument("--shock-vol-ratio", type=float, default=1.45, help="冲击判定波动比阈值（默认 1.45）")
    ap.add_argument("--shock-ret-q", type=float, default=0.85, help="冲击分位阈值Q（默认 0.85）")
    ap.add_argument("--rollback-auc-drop-abs", type=float, default=0.004, help="AUC绝对回滚阈值（默认 0.004）")
    ap.add_argument("--rollback-auc-drop-rel", type=float, default=0.008, help="AUC相对回滚阈值（默认 0.008）")
    ap.add_argument("--rollback-utility-drop-abs", type=float, default=0.0, help="效用pnl代理绝对回滚阈值（严格模式默认 0.0）")
    ap.add_argument("--rollback-down-wr-drop-abs", type=float, default=0.0, help="DOWN胜率绝对回滚阈值（严格模式默认 0.0）")
    ap.add_argument("--utility-min-confidence", type=float, default=0.53, help="效用统计最小置信度（默认 0.53）")
    ap.add_argument("--utility-min-samples", type=int, default=120, help="效用统计最小样本（默认 120）")
    ap.add_argument("--utility-min-down-samples", type=int, default=40, help="DOWN效用统计最小样本（默认 40）")
    ap.add_argument(
        "--train-script-timeout-seconds",
        type=int,
        default=DEFAULT_TRAIN_SCRIPT_TIMEOUT_SECONDS,
        help=f"单个训练脚本最长运行秒数（默认 {DEFAULT_TRAIN_SCRIPT_TIMEOUT_SECONDS}）",
    )
    ap.add_argument(
        "--group-profile-json",
        type=str,
        default="",
        help="按分组最优参数运行训练（传 reports/online_learning_group_best_params.json 路径）",
    )
    ap.add_argument(
        "--coverage-lookback-hours",
        type=int,
        default=DEFAULT_COVERAGE_LOOKBACK_HOURS,
        help=f"覆盖检查回看最近多少小时（默认 {DEFAULT_COVERAGE_LOOKBACK_HOURS}）",
    )
    ap.add_argument(
        "--coverage-grace-minutes",
        type=int,
        default=DEFAULT_COVERAGE_GRACE_MINUTES,
        help=f"覆盖检查容忍延迟分钟（默认 {DEFAULT_COVERAGE_GRACE_MINUTES}）",
    )
    ap.add_argument(
        "--coverage-check-interval-minutes",
        type=int,
        default=DEFAULT_COVERAGE_CHECK_INTERVAL_MINUTES,
        help=f"覆盖检查执行间隔分钟（默认 {DEFAULT_COVERAGE_CHECK_INTERVAL_MINUTES}）",
    )
    ap.add_argument(
        "--coverage-history-max-lines",
        type=int,
        default=DEFAULT_COVERAGE_HISTORY_MAX_LINES,
        help=f"覆盖检查最多读取历史行数（默认 {DEFAULT_COVERAGE_HISTORY_MAX_LINES}）",
    )
    ap.add_argument(
        "--disable-auto-catchup",
        action="store_true",
        help="禁用“检测到长时间未训练时自动补训一次”",
    )
    ap.add_argument(
        "--max-catchup-runs",
        type=int,
        default=DEFAULT_MAX_CATCHUP_RUNS,
        help=f"自动补训最多连续执行轮数（默认 {DEFAULT_MAX_CATCHUP_RUNS}）",
    )
    ap.add_argument(
        "--pm-backfill-lookback-hours",
        type=int,
        default=DEFAULT_PM_BACKFILL_LOOKBACK_HOURS,
        help=f"PM 概率/目标概率断档回看窗口小时数（默认 {DEFAULT_PM_BACKFILL_LOOKBACK_HOURS}）",
    )
    ap.add_argument(
        "--pm-backfill-check-interval-minutes",
        type=int,
        default=DEFAULT_PM_BACKFILL_CHECK_INTERVAL_MINUTES,
        help=f"PM 断档检查间隔分钟（默认 {DEFAULT_PM_BACKFILL_CHECK_INTERVAL_MINUTES}）",
    )
    ap.add_argument(
        "--pm-backfill-max-missing-bars",
        type=int,
        default=DEFAULT_PM_BACKFILL_MAX_MISSING_BARS,
        help=f"PM 断档允许缺失 bar 数（默认 {DEFAULT_PM_BACKFILL_MAX_MISSING_BARS}）",
    )
    ap.add_argument(
        "--pm-backfill-stale-minutes-threshold",
        type=int,
        default=DEFAULT_PM_BACKFILL_STALE_MINUTES,
        help=f"PM 最新点陈旧阈值分钟（默认 {DEFAULT_PM_BACKFILL_STALE_MINUTES}）",
    )
    ap.add_argument(
        "--pm-backfill-workers",
        type=int,
        default=3,
        help="PM 回补 worker 数（默认 3）",
    )
    ap.add_argument(
        "--disable-pm-auto-backfill",
        action="store_true",
        help="禁用 PM 概率/目标概率自动断档回补",
    )
    ap.add_argument(
        "--disk-hygiene-interval-minutes",
        type=int,
        default=DEFAULT_DISK_HYGIENE_INTERVAL_MINUTES,
        help=f"磁盘保洁最小间隔分钟（默认 {DEFAULT_DISK_HYGIENE_INTERVAL_MINUTES}，0=禁用）",
    )
    ap.add_argument(
        "--archive-hygiene-interval-minutes",
        type=int,
        default=DEFAULT_ARCHIVE_HYGIENE_INTERVAL_MINUTES,
        help=f"历史归档最小间隔分钟（默认 {DEFAULT_ARCHIVE_HYGIENE_INTERVAL_MINUTES}，0=禁用）",
    )
    ap.add_argument(
        "--param-search-cadence-minutes",
        type=int,
        default=DEFAULT_PARAM_SEARCH_CADENCE_MINUTES,
        help=f"增量参数搜索兼容元数据 cadence（分钟，夜间主搜模式下仅作状态记录，默认 {DEFAULT_PARAM_SEARCH_CADENCE_MINUTES}）",
    )
    ap.add_argument(
        "--param-search-max-catchup-slices-per-launch",
        type=int,
        default=DEFAULT_PARAM_SEARCH_MAX_CATCHUP_SLICES_PER_LAUNCH,
        help=f"夜间增量参数搜索每次启动最多连续 slice 数（默认 {DEFAULT_PARAM_SEARCH_MAX_CATCHUP_SLICES_PER_LAUNCH}）",
    )
    args = ap.parse_args()
    hour = args.hour % 24
    wait_minutes = max(0, args.wait_minutes)
    interval_minutes = max(0, args.interval_minutes)
    interval_offset_minutes = max(0, args.interval_offset_minutes)
    train_script_timeout_seconds = max(60, int(args.train_script_timeout_seconds))
    coverage_lookback_hours = max(1, int(args.coverage_lookback_hours))
    coverage_grace_minutes = max(0, int(args.coverage_grace_minutes))
    coverage_check_interval_minutes = max(1, int(args.coverage_check_interval_minutes))
    coverage_history_max_lines = max(200, int(args.coverage_history_max_lines))
    auto_catchup_enabled = not bool(args.disable_auto_catchup)
    max_catchup_runs = max(1, int(args.max_catchup_runs))
    pm_auto_backfill_enabled = not bool(args.disable_pm_auto_backfill)
    pm_backfill_lookback_hours = max(24, int(args.pm_backfill_lookback_hours))
    pm_backfill_check_interval_minutes = max(5, int(args.pm_backfill_check_interval_minutes))
    pm_backfill_max_missing_bars = max(1, int(args.pm_backfill_max_missing_bars))
    pm_backfill_stale_minutes = max(10, int(args.pm_backfill_stale_minutes_threshold))
    pm_backfill_workers = max(1, int(args.pm_backfill_workers))
    disk_hygiene_interval_minutes = max(0, int(args.disk_hygiene_interval_minutes))
    archive_hygiene_interval_minutes = max(0, int(args.archive_hygiene_interval_minutes))
    param_search_cadence_minutes = max(1, int(args.param_search_cadence_minutes))
    param_search_max_catchup_slices_per_launch = max(1, int(args.param_search_max_catchup_slices_per_launch))
    last_disk_hygiene_ts = 0.0
    last_archive_hygiene_ts = 0.0
    train_success_events: deque[float] = deque()
    train_timeout_events: deque[float] = deque()
    last_train_duration_sec: Optional[float] = None

    def _prune_recent(buf: deque[float], now_ts: float) -> None:
        horizon = now_ts - 24 * 3600
        while buf and buf[0] < horizon:
            buf.popleft()

    def _append_recent(buf: deque[float], now_ts: float, repeat: int = 1) -> None:
        for _ in range(max(1, int(repeat))):
            buf.append(now_ts)
        _prune_recent(buf, now_ts)

    def _fmt_duration(sec: Optional[float]) -> str:
        if sec is None:
            return "-"
        s = int(max(0, sec))
        if s < 60:
            return f"{s}s"
        if s < 3600:
            return f"{s // 60}m{s % 60:02d}s"
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"

    def _emit_scheduler_status(next_ts: float) -> None:
        now_ts = time.time()
        _prune_recent(train_success_events, now_ts)
        _prune_recent(train_timeout_events, now_ts)
        if interval_minutes > 0:
            sched_mode = os.environ.get("CORE10_INCREMENTAL_TRAINING_MODE", "shared_or_legacy")
            print(
                f"调度状态: interval={interval_minutes}m mode={sched_mode} | "
                f"next_run_at={datetime.fromtimestamp(next_ts).strftime('%Y-%m-%d %H:%M:%S')} | "
                f"last_train_duration={_fmt_duration(last_train_duration_sec)} | "
                f"success_24h={len(train_success_events)} | timeout_count_24h={len(train_timeout_events)}",
                flush=True,
            )

    ok, old_pid = acquire_scheduler_pid(SCHEDULER_PID_FILE)
    if not ok:
        if old_pid:
            print(f"检测到已有调度器运行中 (PID={old_pid})，当前实例退出。", flush=True)
        else:
            print("无法写入 daily_training_scheduler.pid，当前实例退出。", flush=True)
        return

    if cleanup_stale_run_lock(RUN_LOCK_FILE):
        print("检测到并已清理过期训练锁 logs/online_learning_scheduler.lock。", flush=True)

    if interval_minutes > 0 and wait_minutes >= interval_minutes:
        adjusted = max(0, interval_minutes // 3)
        print(
            f"检测到 wait_minutes({wait_minutes}) >= interval({interval_minutes})，自动调整 wait_minutes={adjusted} 以避免重叠。",
            flush=True,
        )
        wait_minutes = adjusted

    if interval_minutes > 0:
        print(
            f"增量训练调度已启动：每 {interval_minutes} 分钟运行（偏移 {interval_offset_minutes} 分钟），训练后等待 {wait_minutes} 分钟再跑监控。",
            flush=True,
        )
        print(
            f"覆盖检查已启用：回看 {coverage_lookback_hours} 小时，容忍延迟 {coverage_grace_minutes} 分钟，报告文件 logs/daily_training_coverage_report.json。",
            flush=True,
        )
        print(
            f"覆盖自检节拍：每 {coverage_check_interval_minutes} 分钟执行一次（含自动补训判定）。",
            flush=True,
        )
        if pm_auto_backfill_enabled:
            print(
                f"PM 断档自检已启用：每 {pm_backfill_check_interval_minutes} 分钟检查，回看 {pm_backfill_lookback_hours} 小时，缺失阈值 {pm_backfill_max_missing_bars} bars。",
                flush=True,
            )
        if archive_hygiene_interval_minutes > 0:
            print(
                f"历史归档已启用：每 {archive_hygiene_interval_minutes} 分钟归档历史 tuning/backup，不触碰 prediction_trades/report_summary/latest。",
                flush=True,
            )
    else:
        print(f"每日训练调度已启动：每天 {hour} 点运行，训练后等待 {wait_minutes} 分钟再跑监控。", flush=True)

    train_python = resolve_training_python()
    print(f"训练 Python: {train_python}", flush=True)
    group_profile_json = (args.group_profile_json or "").strip()
    group_profile_path = None
    if group_profile_json:
        p = Path(group_profile_json)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        group_profile_path = p
        if group_profile_path.exists():
            print(
                f"分组参数: {group_profile_path} | mode={os.environ.get('CORE10_INCREMENTAL_TRAINING_MODE', 'shared_or_legacy')}",
                flush=True,
            )
        else:
            print(f"⚠️ 分组参数文件不存在，回退到常规训练: {group_profile_path}", flush=True)
            group_profile_path = None
    common_train_args = [
        "--hours", str(max(1, args.hours)),
        "--base-rounds", str(max(1, args.base_rounds)),
        "--calm-rounds", str(max(1, args.calm_rounds)),
        "--shock-rounds", str(max(1, args.shock_rounds)),
        "--recency-halflife-bars", str(max(1, args.recency_halflife_bars)),
        "--shock-weight-mult", str(max(1.0, args.shock_weight_mult)),
        "--calm-vol-ratio", str(max(1.0, args.calm_vol_ratio)),
        "--shock-vol-ratio", str(max(1.0, args.shock_vol_ratio)),
        "--shock-ret-q", str(min(0.99, max(0.50, args.shock_ret_q))),
        "--rollback-auc-drop-abs", str(max(0.0, args.rollback_auc_drop_abs)),
        "--rollback-auc-drop-rel", str(max(0.0, args.rollback_auc_drop_rel)),
        "--rollback-utility-drop-abs", str(max(0.0, args.rollback_utility_drop_abs)),
        "--rollback-down-wr-drop-abs", str(max(0.0, args.rollback_down_wr_drop_abs)),
        "--utility-min-confidence", str(min(0.99, max(0.50, args.utility_min_confidence))),
        "--utility-min-samples", str(max(20, args.utility_min_samples)),
        "--utility-min-down-samples", str(max(10, args.utility_min_down_samples)),
    ]

    def report_coverage_and_maybe_log() -> dict[str, Any] | None:
        if interval_minutes <= 0:
            return None
        report = analyze_hourly_coverage(
            history_file=ONLINE_HISTORY_FILE,
            interval_minutes=interval_minutes,
            offset_minutes=interval_offset_minutes,
            lookback_hours=coverage_lookback_hours,
            grace_minutes=coverage_grace_minutes,
            max_history_lines=coverage_history_max_lines,
        )
        write_coverage_report(COVERAGE_REPORT_FILE, report)

        missing = int(report.get("missing_slots", 0) or 0)
        historical_missing = int(report.get("historical_missing_slots", 0) or 0)
        expected = int(report.get("expected_slots", 0) or 0)
        covered = int(report.get("covered_slots", 0) or 0)
        latest_age = report.get("latest_training_age_minutes")
        latest_age_text = "N/A" if latest_age is None else f"{float(latest_age):.1f} 分钟"
        print(
            f"[{datetime.now().isoformat()}] 覆盖检查: 覆盖 {covered}/{expected}, 缺口 {missing}, 最新训练年龄 {latest_age_text}",
            flush=True,
        )
        if missing > 0:
            missing_ts = report.get("missing_slot_timestamps") or []
            if isinstance(missing_ts, list) and missing_ts:
                show = ", ".join(str(x) for x in missing_ts[:4])
                extra = f" ...(+{max(0, len(missing_ts) - 4)}个)" if len(missing_ts) > 4 else ""
                print(f"  缺口样例(UTC): {show}{extra}", flush=True)
        if historical_missing > 0:
            missing_hist_ts = report.get("historical_missing_slot_timestamps") or []
            if isinstance(missing_hist_ts, list) and missing_hist_ts:
                show = ", ".join(str(x) for x in missing_hist_ts[:3])
                extra = f" ...(+{max(0, len(missing_hist_ts) - 3)}个)" if len(missing_hist_ts) > 3 else ""
                print(
                    f"  历史缺口(自动归档,不触发补训): {historical_missing} | 样例(UTC): {show}{extra}",
                    flush=True,
                )
        return report

    def _run_optional_script(path: Path, *extra_args: str) -> None:
        if not path.exists():
            return
        subprocess.run(
            [train_python, str(path), *extra_args],
            cwd=str(PROJECT_ROOT),
            check=False,
        )

    def _log_default_exp10_binding() -> None:
        binding_report = PROJECT_ROOT / "reports" / "default_exp10_bp0520_eth_incremental_binding_latest.json"
        if not binding_report.exists():
            print("[default_exp10_eth] 增量绑定报告缺失", flush=True)
            return
        try:
            payload = json.loads(binding_report.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[default_exp10_eth] 增量绑定报告读取失败: {exc}", flush=True)
            return
        print(
            "[default_exp10_eth] "
            f"runtime_model_dir={payload.get('default_runtime_model_dir') or '-'} | "
            f"incremental_target={payload.get('incremental_target_model_dir') or '-'} | "
            f"matches_runtime_source={payload.get('matches_runtime_source')} | "
            f"runtime_effective_status={payload.get('runtime_effective_status') or '-'}",
            flush=True,
        )

    def _log_core10_mainline_summary(label: str = "snapshot") -> None:
        universe_path = PROJECT_ROOT / "reports" / "core_universe_latest.json"
        profile_path = PROJECT_ROOT / "reports" / "core10_incremental_profile.json"
        source_audit_path = PROJECT_ROOT / "reports" / "core10_prediction_source_audit_latest.json"
        if not universe_path.exists():
            print(f"[core10][{label}] core_universe_latest.json 缺失", flush=True)
            return
        try:
            universe = json.loads(universe_path.read_text(encoding="utf-8"))
            incremental_profile = json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.exists() else {}
            source_audit = json.loads(source_audit_path.read_text(encoding="utf-8")) if source_audit_path.exists() else {}
        except Exception as exc:
            print(f"[core10][{label}] 主线摘要读取失败: {exc}", flush=True)
            return
        core_cells = [row for row in (universe.get("core_cells") or []) if isinstance(row, dict)]
        profile_counts: dict[str, int] = {}
        symbol_counts: dict[str, int] = {}
        trader_by_profile: dict[str, set[str]] = {"default": set(), "70": set()}
        for row in core_cells:
            profile = str(row.get("profile") or "")
            symbol = str(row.get("symbol") or "").upper()
            trader = str(row.get("trader") or "")
            profile_counts[profile] = profile_counts.get(profile, 0) + 1
            if symbol:
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
            if profile in trader_by_profile and trader:
                trader_by_profile[profile].add(trader)
        summary = source_audit.get("summary") if isinstance(source_audit.get("summary"), dict) else {}
        incremental_summary = incremental_profile.get("summary") if isinstance(incremental_profile.get("summary"), dict) else {}
        print(
            f"[core10][{label}] "
            f"active_traders={sum(len(v) for v in trader_by_profile.values())} "
            f"(default={len(trader_by_profile['default'])},70={len(trader_by_profile['70'])}) | "
            f"active_cells={len(core_cells)} "
            f"(default={profile_counts.get('default', 0)},70={profile_counts.get('70', 0)}) | "
            f"cells_by_symbol={dict(sorted(symbol_counts.items()))}",
            flush=True,
        )
        print(
            f"[core10][{label}] "
            f"rules_json_drift_count={summary.get('rules_json_drift_count', 0)} | "
            f"source_drift_count={summary.get('source_drift_count', 0)} | "
            f"prediction_model_version_mismatch_count={summary.get('prediction_model_version_mismatch_count', 0)} | "
            f"incremental_jobs_total={incremental_summary.get('jobs_total', 0)} | "
            f"jobs_by_symbol={incremental_summary.get('jobs_by_symbol', {})} | "
            f"sol_status={incremental_summary.get('sol_status', '-')}",
            flush=True,
        )

    def _log_core10_incremental_outcomes(label: str = "snapshot") -> None:
        report_path = PROJECT_ROOT / "reports" / "core10_incremental_writeback_guard_latest.json"
        if not report_path.exists():
            print(f"[core10_incremental][{label}] 写回守门报告缺失", flush=True)
            return
        try:
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"[core10_incremental][{label}] 写回守门报告读取失败: {exc}", flush=True)
            return
        rows = payload.get("v5_units") if isinstance(payload.get("v5_units"), list) else []
        for row in sorted(rows, key=lambda item: str(item.get("job_id") or "")):
            if not isinstance(row, dict):
                continue
            print(
                f"[core10_incremental][{label}] "
                f"{row.get('job_id') or '-'} | "
                f"target={row.get('primary_target_cell') or '-'} | "
                f"latest_training_at={row.get('latest_training_at') or '-'} | "
                f"writeback={row.get('latest_writeback_allowed')} | "
                f"veto={row.get('latest_writeback_veto_reason') or '-'}",
                flush=True,
            )

    def _grouped_training_in_progress() -> bool:
        if group_profile_path is None:
            return False
        try:
            out = subprocess.check_output(
                ["ps", "-Ao", "pid=,command="],
                text=True,
            )
        except Exception:
            return False
        profile_str = str(group_profile_path)
        current_pid = os.getpid()
        for raw in out.splitlines():
            line = raw.strip()
            if "run_grouped_online_learning.py" not in line:
                continue
            if profile_str not in line:
                continue
            parts = line.split(None, 1)
            if not parts:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            if pid == current_pid:
                continue
            if _pid_alive(pid):
                return True
        return False

    def run_training_once(
        trigger: str,
        wait_override_minutes: int | None = None,
        run_monitor: bool = True,
    ) -> bool:
        nonlocal last_train_duration_sec
        lock_fd = acquire_run_lock(RUN_LOCK_FILE)
        if lock_fd is None:
            print(f"[{datetime.now().isoformat()}] 上一轮训练仍在运行，跳过本轮({trigger})。", flush=True)
            return False

        started_wall_ts = time.time()
        cycle_summary: dict[str, Any] = {
            "trigger": trigger,
            "training_success": False,
            "runtime_apply_success": None,
            "runtime_apply_status": "not_started",
            "core_refresh_success": None,
            "core_refresh_timeout": False,
            "nonblocking_refresh_status": "not_started",
            "param_search_status": "not_started",
            "param_search_nonblocking_timeout": False,
            "param_search_cadence_minutes": param_search_cadence_minutes,
            "param_search_due": False,
            "param_search_next_due_at": "",
            "param_search_backlog_windows": 0,
            "param_search_launch_reason": "",
            "param_search_slices_requested": 0,
            "param_search_slices_completed": 0,
            "param_search_last_finished_at": "",
            "critical_path_seconds": 0.0,
            "core_refresh_seconds": 0.0,
            "nonblocking_refresh_seconds": 0.0,
            "param_search_seconds": 0.0,
            "reload_mode": "",
            "reconcile_status": "not_started",
            "stable_writer_count": 0,
            "stale_writer_count": 0,
            "max_wait_exceeded": False,
            "history_source": "daily_training_scheduler_cycle",
            "result_class": "failed_critical_path",
            "failed_scripts": [],
            "timeout_hits": 0,
        }
        try:
            print(f"\n[{datetime.now().isoformat()}] 开始增量训练... trigger={trigger}", flush=True)
            log_path = PROJECT_ROOT / "logs" / "online_learning.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            had_failed_scripts = False
            timeout_hits = 0
            try:
                failed_scripts: list[tuple[str, int | str]] = []
                with open(log_path, "a", encoding="utf-8") as logf:
                    commands: list[tuple[str, list[str]]] = []
                    if maintenance_window_active():
                        pause_state = load_pause_state()
                        cycle_summary["training_success"] = True
                        cycle_summary["result_class"] = "ok_with_observation"
                        cycle_summary["training_pause_reason"] = str(pause_state.get("reason") or "")
                        print(
                            f"[{datetime.now().isoformat()}] 维护窗生效，跳过本轮训练命令 "
                            f"(reason={cycle_summary['training_pause_reason']})",
                            flush=True,
                        )
                    elif group_profile_path is not None:
                        if group_profile_path.name == "core10_incremental_profile.json" and CORE10_INCREMENTAL_PROFILE_SCRIPT.exists():
                            try:
                                if CORE10_MODEL_BOOST_SPEC_SCRIPT.exists():
                                    subprocess.run(
                                        [train_python, str(CORE10_MODEL_BOOST_SPEC_SCRIPT)],
                                        cwd=str(PROJECT_ROOT),
                                        check=False,
                                    )
                                if CORE10_COMBO_BOOTSTRAP_SCRIPT.exists():
                                    subprocess.run(
                                        [train_python, str(CORE10_COMBO_BOOTSTRAP_SCRIPT)],
                                        cwd=str(PROJECT_ROOT),
                                        check=False,
                                    )
                                subprocess.run(
                                    [train_python, str(CORE10_INCREMENTAL_PROFILE_SCRIPT)],
                                    cwd=str(PROJECT_ROOT),
                                    check=False,
                                )
                                if CORE10_PREDICTION_SOURCE_AUDIT_SCRIPT.exists():
                                    subprocess.run(
                                        [train_python, str(CORE10_PREDICTION_SOURCE_AUDIT_SCRIPT)],
                                        cwd=str(PROJECT_ROOT),
                                        check=False,
                                    )
                                if SELECTOR_ZERO_TRADE_AUDIT_SCRIPT.exists():
                                    subprocess.run(
                                        [train_python, str(SELECTOR_ZERO_TRADE_AUDIT_SCRIPT)],
                                        cwd=str(PROJECT_ROOT),
                                        check=False,
                                    )
                                if MAINLINE_RUNTIME_GUARD_REVIEW_SCRIPT.exists():
                                    subprocess.run(
                                        [train_python, str(MAINLINE_RUNTIME_GUARD_REVIEW_SCRIPT)],
                                        cwd=str(PROJECT_ROOT),
                                        check=False,
                                    )
                                if MAINLINE_POSTLAUNCH_GUARD_EFFECT_SCRIPT.exists():
                                    subprocess.run(
                                        [train_python, str(MAINLINE_POSTLAUNCH_GUARD_EFFECT_SCRIPT)],
                                        cwd=str(PROJECT_ROOT),
                                        check=False,
                                    )
                                if MAINLINE_OVERNIGHT_REVIEW_SCRIPT.exists():
                                    subprocess.run(
                                        [train_python, str(MAINLINE_OVERNIGHT_REVIEW_SCRIPT)],
                                        cwd=str(PROJECT_ROOT),
                                        check=False,
                                    )
                                if MAINLINE_BUY_PRICE_TRUTH_SCRIPT.exists():
                                    subprocess.run(
                                        [train_python, str(MAINLINE_BUY_PRICE_TRUTH_SCRIPT)],
                                        cwd=str(PROJECT_ROOT),
                                        check=False,
                                    )
                                if MAINLINE_DEFAULT_BUY_PRICE_BALANCE_SCRIPT.exists():
                                    subprocess.run(
                                        [train_python, str(MAINLINE_DEFAULT_BUY_PRICE_BALANCE_SCRIPT)],
                                        cwd=str(PROJECT_ROOT),
                                        check=False,
                                    )
                                if POLYFUN_MAINLINE_FINAL_STATUS_SCRIPT.exists():
                                    subprocess.run(
                                        [train_python, str(POLYFUN_MAINLINE_FINAL_STATUS_SCRIPT)],
                                        cwd=str(PROJECT_ROOT),
                                        check=False,
                                    )
                                if MAINLINE_RECENT_CHANGE_SEMANTIC_AUDIT_SCRIPT.exists():
                                    subprocess.run(
                                        [train_python, str(MAINLINE_RECENT_CHANGE_SEMANTIC_AUDIT_SCRIPT)],
                                        cwd=str(PROJECT_ROOT),
                                        check=False,
                                    )
                                if MAINLINE_SAFE_CLEANUP_INVENTORY_SCRIPT.exists():
                                    subprocess.run(
                                        [train_python, str(MAINLINE_SAFE_CLEANUP_INVENTORY_SCRIPT)],
                                        cwd=str(PROJECT_ROOT),
                                        check=False,
                                    )
                                if CORE10_WRITER_LAUNCHCTL_INVENTORY_SCRIPT.exists():
                                    subprocess.run(
                                        [train_python, str(CORE10_WRITER_LAUNCHCTL_INVENTORY_SCRIPT)],
                                        cwd=str(PROJECT_ROOT),
                                        check=False,
                                    )
                                if CORE10_LEGACY_CLEANUP_INVENTORY_SCRIPT.exists():
                                    subprocess.run(
                                        [train_python, str(CORE10_LEGACY_CLEANUP_INVENTORY_SCRIPT)],
                                        cwd=str(PROJECT_ROOT),
                                        check=False,
                                    )
                                if MAINLINE_70_THRESHOLD_SEARCH_SCRIPT.exists():
                                    subprocess.run(
                                        [train_python, str(MAINLINE_70_THRESHOLD_SEARCH_SCRIPT)],
                                        cwd=str(PROJECT_ROOT),
                                        check=False,
                                    )
                                if POLYFUN_MAINLINE_FINAL_STATUS_SCRIPT.exists():
                                    subprocess.run(
                                        [train_python, str(POLYFUN_MAINLINE_FINAL_STATUS_SCRIPT)],
                                        cwd=str(PROJECT_ROOT),
                                        check=False,
                                    )
                                if MAINLINE_ACCEPTANCE_SCRIPT.exists():
                                    subprocess.run(
                                        [train_python, str(MAINLINE_ACCEPTANCE_SCRIPT)],
                                        cwd=str(PROJECT_ROOT),
                                        check=False,
                                    )
                                if MAINLINE_CHAIN_ALIGNMENT_AUDIT_SCRIPT.exists():
                                    subprocess.run(
                                        [train_python, str(MAINLINE_CHAIN_ALIGNMENT_AUDIT_SCRIPT)],
                                        cwd=str(PROJECT_ROOT),
                                        check=False,
                                    )
                                if ACTIVE_TRUTH_SURFACE_CONSISTENCY_SCRIPT.exists():
                                    subprocess.run(
                                        [train_python, str(ACTIVE_TRUTH_SURFACE_CONSISTENCY_SCRIPT)],
                                        cwd=str(PROJECT_ROOT),
                                        check=False,
                                    )
                                if POST_RESET_TRADE_STRUCTURE_REVIEW_SCRIPT.exists():
                                    subprocess.run(
                                        [train_python, str(POST_RESET_TRADE_STRUCTURE_REVIEW_SCRIPT)],
                                        cwd=str(PROJECT_ROOT),
                                        check=False,
                                    )
                                if SIMULATION_ORDER_SEMANTICS_AUDIT_SCRIPT.exists():
                                    subprocess.run(
                                        [train_python, str(SIMULATION_ORDER_SEMANTICS_AUDIT_SCRIPT)],
                                        cwd=str(PROJECT_ROOT),
                                        check=False,
                                    )
                                if MONITOR_CELL_CAPITAL_CONTRACT_SCRIPT.exists():
                                    subprocess.run(
                                        [train_python, str(MONITOR_CELL_CAPITAL_CONTRACT_SCRIPT)],
                                        cwd=str(PROJECT_ROOT),
                                        check=False,
                                    )
                                if MAINLINE_FINAL_WRAPUP_AUDIT_SCRIPT.exists():
                                    subprocess.run(
                                        [train_python, str(MAINLINE_FINAL_WRAPUP_AUDIT_SCRIPT)],
                                        cwd=str(PROJECT_ROOT),
                                        check=False,
                                    )
                                _log_core10_mainline_summary("pre_run_snapshot")
                                if CORE10_INCREMENTAL_WRITEBACK_GUARD_SCRIPT.exists():
                                    subprocess.run(
                                        [train_python, str(CORE10_INCREMENTAL_WRITEBACK_GUARD_SCRIPT)],
                                        cwd=str(PROJECT_ROOT),
                                        check=False,
                                    )
                                if CORE10_SIMULATION_MONEY_RECONCILIATION_SCRIPT.exists():
                                    subprocess.run(
                                        [train_python, str(CORE10_SIMULATION_MONEY_RECONCILIATION_SCRIPT)],
                                        cwd=str(PROJECT_ROOT),
                                        check=False,
                                    )
                            except Exception as e:
                                print(f"core10 incremental profile 刷新异常: {e}", flush=True)
                            cmd = [
                                train_python,
                                str(PROJECT_ROOT / "scripts" / "run_grouped_online_learning.py"),
                                "--profile-json", str(group_profile_path),
                                "--python-bin", train_python,
                            ]
                            commands.append(("run_grouped_online_learning.py", cmd))
                    else:
                        for script in ["online_learning_daily.py", "online_learning_gru.py"]:
                            cmd = [train_python, str(PROJECT_ROOT / "scripts" / script), *common_train_args]
                            commands.append((script, cmd))

                    for script_name, cmd in commands:
                        try:
                            child_env = os.environ.copy()
                            # 子脚本 stdout 已重定向到 online_learning.log，
                            # 再开 FileHandler 会导致同一日志重复写两遍。
                            child_env["ONLINE_LEARNING_NO_FILE_HANDLER"] = "1"
                            proc = subprocess.Popen(
                                cmd,
                                cwd=str(PROJECT_ROOT),
                                stdout=logf,
                                stderr=subprocess.STDOUT,
                                env=child_env,
                                start_new_session=True,
                            )
                            effective_timeout_seconds = int(train_script_timeout_seconds)
                            if script_name == "run_grouped_online_learning.py":
                                effective_timeout_seconds = max(
                                    effective_timeout_seconds,
                                    int(DEFAULT_GROUPED_TRAIN_SCRIPT_TIMEOUT_SECONDS),
                                )
                            try:
                                ret_code = proc.wait(timeout=effective_timeout_seconds)
                            except subprocess.TimeoutExpired:
                                # 只 kill 主进程会遗留 multiprocessing 子进程，这里按进程组回收。
                                try:
                                    os.killpg(proc.pid, signal.SIGTERM)
                                except ProcessLookupError:
                                    pass
                                try:
                                    proc.wait(timeout=10)
                                except subprocess.TimeoutExpired:
                                    try:
                                        os.killpg(proc.pid, signal.SIGKILL)
                                    except ProcessLookupError:
                                        pass
                                    try:
                                        proc.wait(timeout=5)
                                    except Exception:
                                        pass
                                timeout_hits += 1
                                failed_scripts.append((script_name, f"timeout>{effective_timeout_seconds}s"))
                                continue

                            if ret_code != 0:
                                failed_scripts.append((script_name, ret_code))
                        except Exception as e:
                            failed_scripts.append((script_name, f"spawn_error:{e}"))
                if failed_scripts:
                    cycle_summary["failed_scripts"] = [
                        {"script": name, "rc": str(rc)}
                        for name, rc in failed_scripts
                    ]
                    cycle_summary["timeout_hits"] = int(timeout_hits)
                    had_failed_scripts = True
                    print(
                        f"[{datetime.now().isoformat()}] 训练脚本异常: "
                        + ", ".join(f"{name}(rc={rc})" for name, rc in failed_scripts),
                        flush=True,
                    )
            except Exception as e:
                print(f"训练异常: {e}", flush=True)
                had_failed_scripts = True
            if had_failed_scripts:
                last_train_duration_sec = max(0.0, time.time() - started_wall_ts)
                cycle_summary["training_success"] = False
                cycle_summary["runtime_apply_success"] = False
                cycle_summary["critical_path_seconds"] = round(last_train_duration_sec, 3)
                if timeout_hits > 0:
                    _append_recent(train_timeout_events, time.time(), repeat=timeout_hits)
                _append_online_history_event(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "entry_type": "daily_training_scheduler_cycle",
                        "summary": cycle_summary,
                    }
                )
                print(
                    f"[{datetime.now().isoformat()}] 跳过等待与链式监控（本轮训练存在异常）。",
                    flush=True,
                )
                print(
                    f"[{datetime.now().isoformat()}] 本次完成（含训练脚本异常，请检查 online_learning.log）。\n",
                    flush=True,
                )
                return False
            cycle_summary["training_success"] = True
            cycle_summary["critical_path_seconds"] = round(max(0.0, time.time() - started_wall_ts), 3)
            if group_profile_path is not None and group_profile_path.name == "core10_incremental_profile.json":
                grouped_summary = _read_latest_grouped_cycle_summary(group_profile_path)
                if grouped_summary:
                    cycle_summary["training_success"] = grouped_summary.get("training_success") is not False
                    cycle_summary["runtime_apply_success"] = grouped_summary.get("runtime_apply_success")
                    cycle_summary["runtime_apply_status"] = str(grouped_summary.get("runtime_apply_status") or "not_started")
                    cycle_summary["core_refresh_success"] = grouped_summary.get("core_refresh_success")
                    cycle_summary["critical_path_seconds"] = float(
                        grouped_summary.get("critical_path_seconds") or cycle_summary["critical_path_seconds"] or 0.0
                    )
                    cycle_summary["reload_mode"] = str(grouped_summary.get("reload_mode") or "")
                    cycle_summary["reconcile_status"] = str(grouped_summary.get("reconcile_status") or "not_started")
                    cycle_summary["stable_writer_count"] = int(grouped_summary.get("stable_writer_count") or 0)
                    cycle_summary["stale_writer_count"] = int(grouped_summary.get("stale_writer_count") or 0)
                    cycle_summary["max_wait_exceeded"] = bool(grouped_summary.get("max_wait_exceeded"))
                    cycle_summary["result_class"] = str(grouped_summary.get("result_class") or "ok")
                else:
                    cycle_summary["runtime_apply_success"] = True
                    cycle_summary["runtime_apply_status"] = "not_applicable"
                    cycle_summary["core_refresh_success"] = True
                    cycle_summary["result_class"] = "ok"

                if (
                    cycle_summary["training_success"] is False
                    or cycle_summary["runtime_apply_success"] is False
                    or cycle_summary["core_refresh_success"] is False
                ):
                    had_failed_scripts = True

                refresh_launch = _launch_nonblocking_mainline_refresh(train_python, "训练后主线刷新链异步续跑")
                cycle_summary["nonblocking_refresh_status"] = str(refresh_launch.get("status") or "unknown")
                if (
                    cycle_summary["training_success"] is True
                    and cycle_summary["runtime_apply_success"] is True
                    and cycle_summary["core_refresh_success"] is True
                ):
                    _run_optional_script(ACTIVE_INCREMENTAL_SEARCH_UNIVERSE_SCRIPT)
                    _run_optional_script(CORE10_MODEL_BOOST_SPEC_SCRIPT)
                    _run_optional_script(INCREMENTAL_NIGHT_SEARCH_BACKLOG_SCRIPT)
                    pause_state = load_pause_state()
                    param_search_state = _load_latest_param_search_state(
                        cadence_minutes=param_search_cadence_minutes,
                        max_catchup_slices_per_launch=param_search_max_catchup_slices_per_launch,
                    )
                    param_search_state["param_search_pause_reason"] = str(pause_state.get("reason") or "")
                    param_search_state["param_search_paused_for_lowprice_full_tuning"] = bool(
                        pause_state.get("active")
                    ) and (
                        "lowprice" in str(pause_state.get("reason") or "").lower()
                        or "stale" in str(pause_state.get("reason") or "").lower()
                    )
                    cycle_summary["param_search_cadence_minutes"] = int(
                        param_search_state.get("param_search_cadence_minutes") or param_search_cadence_minutes
                    )
                    cycle_summary["param_search_due"] = bool(param_search_state.get("param_search_due"))
                    cycle_summary["param_search_next_due_at"] = str(
                        param_search_state.get("param_search_next_due_at") or ""
                    )
                    cycle_summary["param_search_backlog_windows"] = int(
                        param_search_state.get("param_search_backlog_windows") or 0
                    )
                    cycle_summary["param_search_launch_reason"] = str(
                        param_search_state.get("param_search_launch_reason") or ""
                    )
                    cycle_summary["param_search_slices_requested"] = int(
                        param_search_state.get("param_search_slices_requested") or 0
                    )
                    cycle_summary["param_search_family_symbol_keys"] = list(
                        param_search_state.get("param_search_family_symbol_keys") or []
                    )
                    cycle_summary["param_search_last_finished_at"] = str(
                        param_search_state.get("param_search_last_finished_at") or ""
                    )
                    cycle_summary["night_search_window_active"] = bool(
                        param_search_state.get("night_search_window_active")
                    )
                    cycle_summary["night_search_backlog_total"] = int(
                        param_search_state.get("night_search_backlog_total") or 0
                    )
                    cycle_summary["night_search_backlog_remaining"] = int(
                        param_search_state.get("night_search_backlog_remaining") or 0
                    )
                    cycle_summary["night_search_stop_reason"] = str(
                        param_search_state.get("night_search_stop_reason") or ""
                    )
                    cycle_summary["nightly_launcher_status"] = str(
                        param_search_state.get("nightly_launcher_status") or ""
                    )
                    cycle_summary["nightly_launcher_runtime_root"] = str(
                        param_search_state.get("nightly_launcher_runtime_root") or ""
                    )
                    cycle_summary["nightly_launcher_fallback_required"] = bool(
                        param_search_state.get("nightly_launcher_fallback_required")
                    )
                    cycle_summary["nightly_launcher_fallback_reason"] = str(
                        param_search_state.get("nightly_launcher_fallback_reason") or ""
                    )
                    cycle_summary["param_search_delegated_to_nightly_launcher"] = True
                    if bool(pause_state.get("active")):
                        cycle_summary["param_search_status"] = "paused"
                        _write_param_search_runtime_status_snapshot("paused", param_search_state)
                    elif bool(param_search_state.get("param_search_running")):
                        cycle_summary["param_search_status"] = "already_running"
                        _write_param_search_runtime_status_snapshot(
                            "already_running",
                            param_search_state,
                            pid=int(param_search_state.get("param_search_runtime_pid") or 0),
                        )
                    elif not bool(param_search_state.get("param_search_due")):
                        cycle_summary["param_search_status"] = "not_due"
                        _write_param_search_runtime_status_snapshot("not_due", param_search_state)
                    elif bool(param_search_state.get("nightly_launcher_fallback_required")):
                        param_search_state["param_search_launch_reason"] = "night_backlog_drain_scheduler_fallback"
                        param_search_state["night_search_stop_reason"] = str(
                            param_search_state.get("nightly_launcher_fallback_reason") or "scheduler_fallback"
                        )
                        fallback_launch = _launch_nonblocking_param_search(
                            train_python,
                            "夜窗超參 scheduler 兜底启动",
                            param_search_state,
                        )
                        param_search_state["schedulerFallbackLaunched"] = bool(fallback_launch.get("launched"))
                        cycle_summary["param_search_delegated_to_nightly_launcher"] = False
                        cycle_summary["param_search_scheduler_fallback"] = True
                        cycle_summary["param_search_scheduler_fallback_reason"] = str(
                            param_search_state.get("night_search_stop_reason") or ""
                        )
                        cycle_summary["param_search_status"] = str(fallback_launch.get("status") or "unknown")
                    else:
                        cycle_summary["param_search_status"] = "delegated_to_nightly_launcher"
                        param_search_state["param_search_launch_reason"] = (
                            str(param_search_state.get("param_search_launch_reason") or "")
                            or "nightly_param_search_launcher_owns_0000_start"
                        )
                        param_search_state["night_search_stop_reason"] = "delegated_to_nightly_launcher"
                        _write_param_search_runtime_status_snapshot("delegated_to_nightly_launcher", param_search_state)
                    if bool(param_search_state.get("night_search_window_active")) and not bool(pause_state.get("active")):
                        joint_direction = _launch_joint_direction_nightly(train_python, "夜窗联合方向搜索")
                        cycle_summary["joint_direction_status"] = str(joint_direction.get("status") or "")
                        cycle_summary["joint_direction_pid"] = int(joint_direction.get("pid") or 0)
                        cycle_summary["joint_direction_report"] = str(joint_direction.get("report") or "")
                    else:
                        cycle_summary["joint_direction_status"] = "skipped_outside_night_window_or_paused"
                        cycle_summary["joint_direction_pid"] = 0
                        cycle_summary["joint_direction_report"] = str(JOINT_DIRECTION_LAUNCHER_REPORT)
                else:
                    cycle_summary["param_search_status"] = "skipped_not_training_safe"
                    _write_param_search_runtime_status_snapshot(
                        "skipped_not_training_safe",
                        {
                            "param_search_cadence_minutes": param_search_cadence_minutes,
                            "param_search_due": False,
                            "param_search_next_due_at": "",
                            "param_search_backlog_windows": 0,
                            "param_search_launch_reason": "",
                            "param_search_slices_requested": 0,
                            "param_search_slices_completed": 0,
                            "param_search_family_symbol_keys": [],
                            "param_search_last_finished_at": "",
                            "param_search_last_progress_at": "",
                            "param_search_latest_finished_status": "",
                            "night_search_window_active": False,
                            "night_search_backlog_total": 0,
                            "night_search_backlog_remaining": 0,
                            "night_search_last_family_symbol": "",
                            "night_search_stop_reason": "skipped_not_training_safe",
                            "night_search_local_now": _beijing_now().isoformat(),
                            "night_search_window_start_at": "",
                            "night_search_window_end_at": "",
                            "night_search_next_window_start_at": "",
                            "night_search_next_window_end_at": "",
                        },
                    )
                    cycle_summary["joint_direction_status"] = "skipped_not_training_safe"
                    cycle_summary["joint_direction_pid"] = 0
                    cycle_summary["joint_direction_report"] = str(JOINT_DIRECTION_LAUNCHER_REPORT)
                if cycle_summary["result_class"] == "ok" and cycle_summary["runtime_apply_status"] == "ok_with_observation":
                    cycle_summary["result_class"] = "ok_with_observation"
                if cycle_summary["result_class"] == "ok" and cycle_summary["nonblocking_refresh_status"] in {"failed_non_blocking", "nonblocking_slow"}:
                    cycle_summary["result_class"] = "nonblocking_slow"
                if cycle_summary["result_class"] == "ok" and cycle_summary["param_search_status"] in {"failed_non_blocking", "partial_checkpointed"}:
                    cycle_summary["result_class"] = "ok_with_observation"
                _log_core10_mainline_summary("post_run_snapshot")
                _log_core10_incremental_outcomes("post_run_snapshot")
            if had_failed_scripts:
                last_train_duration_sec = max(0.0, time.time() - started_wall_ts)
                cycle_summary["critical_path_seconds"] = round(
                    float(cycle_summary.get("critical_path_seconds") or last_train_duration_sec),
                    3,
                )
                _append_online_history_event(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "entry_type": "daily_training_scheduler_cycle",
                        "summary": cycle_summary,
                    }
                )
                print(
                    f"[{datetime.now().isoformat()}] 跳过等待与链式监控（本轮 runtime apply / core refresh 未通过）。",
                    flush=True,
                )
                return False
            effective_wait_minutes = wait_minutes if wait_override_minutes is None else max(0, int(wait_override_minutes))
            if effective_wait_minutes > 0:
                print(
                    f"[{datetime.now().isoformat()}] 训练完成，等待 {effective_wait_minutes} 分钟（让预测写入器热重载并写出最新预测）后再跑监控...",
                    flush=True,
                )
                time.sleep(effective_wait_minutes * 60)
            if run_monitor:
                print(f"[{datetime.now().isoformat()}] 运行链式监控（校验热重载是否发生、预测是否用上最新模型）...", flush=True)
                try:
                    subprocess.run(
                        [train_python, str(PROJECT_ROOT / "scripts" / "monitor_daily_training_chain.py")],
                        cwd=str(PROJECT_ROOT),
                        check=False,
                    )
                except Exception as e:
                    print(f"监控异常: {e}", flush=True)
                print(f"[{datetime.now().isoformat()}] 刷新主线运行池必要报告...", flush=True)
                monitor_refresh_detail = _run_mainline_refresh_chain_with_detail(
                    train_python,
                    "监控后主线刷新链",
                    skip_labels=["主线增量参数搜索推荐"],
                )
                if not monitor_refresh_detail.get("success"):
                    had_failed_scripts = True
            last_train_duration_sec = max(0.0, time.time() - started_wall_ts)
            _append_recent(train_success_events, time.time())
            if timeout_hits > 0:
                _append_recent(train_timeout_events, time.time(), repeat=timeout_hits)
            _append_online_history_event(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "entry_type": "daily_training_scheduler_cycle",
                    "summary": cycle_summary,
                }
            )
            print(f"[{datetime.now().isoformat()}] 本次完成。\n", flush=True)
            return True
        finally:
            release_run_lock(lock_fd, RUN_LOCK_FILE)

    def maybe_run_coverage_and_catchup() -> bool:
        """执行覆盖检查；若发现断档则自动补训。返回本轮是否尝试过补训。"""
        coverage = report_coverage_and_maybe_log()
        ran_any_catchup = False
        if (
            interval_minutes <= 0
            or (not auto_catchup_enabled)
            or coverage is None
            or (not bool(coverage.get("needs_catchup")))
        ):
            return ran_any_catchup

        latest_age = coverage.get("latest_training_age_minutes")
        latest_age_text = "N/A" if latest_age is None else f"{float(latest_age):.1f} 分钟"
        missing_slots = int(coverage.get("missing_slots", 0) or 0)
        if _grouped_training_in_progress():
            print(
                f"[{datetime.now().isoformat()}] 覆盖检查命中缺口，但当前主线训练仍在运行，跳过自动补训。 reason=training_in_progress_skip_catchup",
                flush=True,
            )
            return ran_any_catchup
        # 断档后按缺口轮数连续补训（受 max_catchup_runs 限制），
        # 避免首轮后因“最新训练已刷新”而过早停止，确保补训有实际意义。
        rounds = min(max_catchup_runs, max(1, missing_slots))
        print(
            f"[{datetime.now().isoformat()}] 检测到训练链路中断（最新训练年龄 {latest_age_text}），执行自动补训 {rounds} 轮（上限={max_catchup_runs}）。",
            flush=True,
        )
        for i in range(rounds):
            is_last = i == (rounds - 1)
            ran_any_catchup = True
            ok = run_training_once(
                f"auto_catchup_gap_{i+1}/{rounds}",
                wait_override_minutes=(wait_minutes if is_last else 0),
                run_monitor=is_last,
            )
            coverage = report_coverage_and_maybe_log()
            if not ok or coverage is None:
                break
        return ran_any_catchup

    def maybe_run_pm_backfill_guard() -> bool:
        if not pm_auto_backfill_enabled:
            return False
        cmd = [
            train_python,
            str(PROJECT_ROOT / "scripts" / "pm_recent_backfill_guard.py"),
            "--lookback-hours",
            str(pm_backfill_lookback_hours),
            "--max-missing-bars",
            str(pm_backfill_max_missing_bars),
            "--stale-minutes-threshold",
            str(pm_backfill_stale_minutes),
            "--workers",
            str(pm_backfill_workers),
            "--python-bin",
            train_python,
            "--fix",
        ]
        try:
            started = time.time()
            ret = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=2400,
                check=False,
            )
            cost = time.time() - started
            tail = "\n".join((ret.stdout or "").splitlines()[-6:])
            print(
                f"[{datetime.now().isoformat()}] PM 断档自检完成 rc={ret.returncode}, 耗时 {cost/60:.1f} 分钟。",
                flush=True,
            )
            if tail:
                print(tail, flush=True)
            return ret.returncode == 0
        except subprocess.TimeoutExpired:
            print(f"[{datetime.now().isoformat()}] PM 断档自检超时，已跳过本轮。", flush=True)
            return False
        except Exception as e:
            print(f"[{datetime.now().isoformat()}] PM 断档自检异常: {e}", flush=True)
            return False

    def maybe_run_disk_hygiene() -> bool:
        nonlocal last_disk_hygiene_ts
        if disk_hygiene_interval_minutes <= 0:
            return False
        if not DISK_HYGIENE_SCRIPT.exists():
            return False
        now_ts = time.time()
        if last_disk_hygiene_ts > 0 and (now_ts - last_disk_hygiene_ts) < disk_hygiene_interval_minutes * 60:
            return False
        cmd = [train_python, str(DISK_HYGIENE_SCRIPT), "--apply"]
        try:
            ret = subprocess.run(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=300,
                check=False,
            )
            print(f"[{datetime.now().isoformat()}] 磁盘保洁 rc={ret.returncode}", flush=True)
            if ret.stdout:
                tail = "\n".join(ret.stdout.splitlines()[-6:])
                if tail:
                    print(tail, flush=True)
            last_disk_hygiene_ts = now_ts
            return ret.returncode == 0
        except Exception as e:
            print(f"[{datetime.now().isoformat()}] 磁盘保洁异常: {e}", flush=True)
            return False

    def maybe_run_archive_hygiene() -> bool:
        nonlocal last_archive_hygiene_ts
        if archive_hygiene_interval_minutes <= 0:
            return False
        if not ARCHIVE_TUNING_HISTORY_SCRIPT.exists() or not ARCHIVE_RUNTIME_BACKUPS_SCRIPT.exists():
            return False
        now_ts = time.time()
        if last_archive_hygiene_ts > 0 and (now_ts - last_archive_hygiene_ts) < archive_hygiene_interval_minutes * 60:
            return False

        ok = True
        for script in (ARCHIVE_TUNING_HISTORY_SCRIPT, ARCHIVE_RUNTIME_BACKUPS_SCRIPT):
            try:
                ret = subprocess.run(
                    [train_python, str(script), "--apply"],
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=1800,
                    check=False,
                )
                print(f"[{datetime.now().isoformat()}] 历史归档 {script.name} rc={ret.returncode}", flush=True)
                if ret.stdout:
                    tail = "\n".join(ret.stdout.splitlines()[-6:])
                    if tail:
                        print(tail, flush=True)
                ok = ok and (ret.returncode == 0)
            except Exception as e:
                print(f"[{datetime.now().isoformat()}] 历史归档异常 {script.name}: {e}", flush=True)
                ok = False

        last_archive_hygiene_ts = now_ts
        return ok

    next_coverage_check_ts = time.time()
    next_pm_backfill_check_ts = time.time()
    try:
        while True:
            ts = (
                next_run_every(interval_minutes, interval_offset_minutes)
                if interval_minutes > 0
                else next_run_at(hour)
            )
            now = time.time()
            wake_ts = ts
            if interval_minutes > 0:
                wake_ts = min(ts, next_coverage_check_ts)
                if pm_auto_backfill_enabled:
                    wake_ts = min(wake_ts, next_pm_backfill_check_ts)
            wait = max(0, wake_ts - now)

            if interval_minutes > 0:
                wait_cov = max(0.0, next_coverage_check_ts - now)
                wait_pm = max(0.0, next_pm_backfill_check_ts - now) if pm_auto_backfill_enabled else 0.0
                print(
                    f"下次训练: {datetime.fromtimestamp(ts)} (约 {max(0.0, ts - now)/3600:.2f} 小时后) | "
                    f"下次覆盖检查: {datetime.fromtimestamp(next_coverage_check_ts)} (约 {wait_cov/60:.1f} 分钟后)"
                    + (
                        f" | 下次 PM 回补检查: {datetime.fromtimestamp(next_pm_backfill_check_ts)} (约 {wait_pm/60:.1f} 分钟后)"
                        if pm_auto_backfill_enabled
                        else ""
                    ),
                    flush=True,
                )
                _emit_scheduler_status(ts)
            else:
                print(f"下次运行: {datetime.fromtimestamp(ts)} (约 {wait/3600:.1f} 小时后)", flush=True)

            time.sleep(wait)

            ran_catchup = False
            if interval_minutes > 0 and time.time() >= (next_coverage_check_ts - 0.5):
                ran_catchup = maybe_run_coverage_and_catchup()
                next_coverage_check_ts = time.time() + coverage_check_interval_minutes * 60

            if pm_auto_backfill_enabled and time.time() >= (next_pm_backfill_check_ts - 0.5):
                maybe_run_pm_backfill_guard()
                next_pm_backfill_check_ts = time.time() + pm_backfill_check_interval_minutes * 60

            maybe_run_disk_hygiene()
            maybe_run_archive_hygiene()

            if time.time() >= (ts - 0.5):
                if ran_catchup:
                    print(
                        f"[{datetime.now().isoformat()}] 本轮已执行自动补训（无论成功与否），跳过重复 scheduled 训练。",
                        flush=True,
                    )
                else:
                    run_training_once("scheduled")
    finally:
        release_scheduler_pid(SCHEDULER_PID_FILE)


if __name__ == "__main__":
    main()
