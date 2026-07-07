#!/usr/bin/env python3
"""
系统健康监控 v5 — 适配当前合并进程架构（按 active 配置动态计算）

检查项:
  1.  系统代理 (Clash等) 是否在线
  2.  外部API连通性 (Binance/Polymarket)
  3.  ccxt 库能否通过代理连接 Binance
  4.  Market Data Daemon 进程和缓存新鲜度
  5.  数据采集器进程检查 (collect_derivatives_realtime)
  5b. 数据源新鲜度 (采集器文件 + data_ready 门控源)
  5c. 本地K线新鲜度 (data/raw 下 15m/1h/4h BTC/ETH/SOL/XRP 是否更新到最新)
  6.  所有预测写入器进程 (v5/GRU/ensemble) 是否存活
  7.  所有预测文件新鲜度（按类型阈值：V5=35m, GRU=25m, T-120s=50m）
  8.  合并交易进程是否存活 (按 active_traders / active_traders_70 动态计算)
  9.  重复/残留进程检测 (写入器/合并进程/旧式独立进程)
  10. 全量 WebSocket + REST 超时扫描 (合并日志 + 旧日志)
  11. 代理连接数统计 (lsof)
  12. 日志异常模式检测 (crash/OOM/unhandled/Error)
  13. 融合模型输出验证 (方向/置信度/源数量合理性)
  14. 交易执行统计 (近期是否有模拟/真实成交)

原则: 纯只读监控，绝不影响任何模型的模拟交易。

用法:
  python scripts/api_health_check.py           # 持续监控(每3分钟)
  python scripts/api_health_check.py --once    # 单次检查
"""
from __future__ import annotations

import argparse
from collections import deque
import fcntl
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLYMARKET_DIR = PROJECT_ROOT / "polymarket"
SLOT01_SLOT02_TARGET_STACK_CONTRACT = PROJECT_ROOT / "reports" / "slot01_slot02_target_stack_contract_latest.json"

# 尽早加载 .env 中的代理，使 getproxies() 和 urlopen 能用到
_env_file = PROJECT_ROOT / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(_env_file)
    except ImportError:
        pass

CHECK_INTERVAL = 180  # 3分钟
MAINLINE_REPORT_FRESH_MAX_AGE_SEC = 2 * 60 * 60

# ─── 预期的进程和文件 ────────────────────────────────────

LEGACY_V5_GROUP_LABELS = [
    ("v5_exp10", "Exp10†"),  # † = T-120s 模型, 收盘前写入
    ("v5_exp11", "Exp11"),
    ("v5_exp13", "Exp13"),
    ("v5_exp14", "Exp14†"),  # † = T-120s 模型, 收盘前写入
    ("v5_exp15", "Exp15"),
    ("v5_exp16", "Exp16"),
    ("v5_exp17", "Exp17"),
]

LEGACY_V5_GROUP_LABELS_70 = [
    ("v5_exp10_70", "Exp10† 70"),
    ("v5_exp11_70", "Exp11 70"),
    ("v5_exp13_70", "Exp13 70"),
    ("v5_exp14_70", "Exp14† 70"),
    ("v5_exp15_70", "Exp15 70"),
    ("v5_exp16_70", "Exp16 70"),
]

EXPECTED_V5_WRITERS = [
    (label, f"predictions_{group}.json")
    for group, label in LEGACY_V5_GROUP_LABELS
]

EXPECTED_V5_WRITERS_70 = [
    (label, f"predictions_{group}.json")
    for group, label in LEGACY_V5_GROUP_LABELS_70
]

WALLET_V5_FILE_TO_GROUP = {
    "predictions_v5_exp10_wallet.json": "v5_exp10",
    "predictions_v5_exp10_wallet_70.json": "v5_exp10_70",
}

# T-120s 模型写入周期更长（收盘前2min写入，重启后最多跳过1bar静默~45min）
T120S_STALE_THRESHOLD_MIN = 50

EXPECTED_GRU_WRITERS = [
    ("GRU BTC",         "predictions_gru_btc.json"),       # 预测器保留（为融合供数据），交易已停
    ("GRU ETH",         "predictions_gru_eth.json"),
    ("GRU BTC no1h4h",  "predictions_gru_btc_no1h4h.json"),  # 预测器保留，交易已停
    ("GRU ETH no1h4h",  "predictions_gru_eth_no1h4h.json"),  # 预测器保留，交易已停
    ("GRU XRP",         "predictions_gru_xrp.json"),
    # GRU XRP no1h4h 已停止（不在融合中）
    ("GRU SOL",         "predictions_gru_sol.json"),
]

# ensemble/ensemble_70 仅保留参考产物，不再作为在线模拟交易链健康前提。
REFERENCE_ONLY_ENSEMBLE = [
    ("Ensemble",        "predictions_ensemble.json"),
]

REFERENCE_ONLY_ENSEMBLE_70 = [
    ("Ensemble 70",     "predictions_ensemble_70.json"),
]

API_ENDPOINTS = [
    ("Binance ping",     "https://api.binance.com/api/v3/ping"),
    ("Binance klines",   "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=15m&limit=1"),
    ("Polymarket time",  "https://clob.polymarket.com/time"),
    ("Polymarket API",   "https://gamma-api.polymarket.com/events?limit=1"),
]

# 预测文件新鲜度阈值（与 monitor_daily_training_chain.py 对齐口径）
STALE_THRESHOLD_MIN_V5 = 35
STALE_THRESHOLD_MIN_GRU = 25
DEFAULT_EXP10_ETH_TRUTH_MANIFEST = PROJECT_ROOT / "reports" / "default_exp10_bp0520_eth_truth_manifest_latest.json"
DEFAULT_EXP10_ETH_INCREMENTAL_BINDING = PROJECT_ROOT / "reports" / "default_exp10_bp0520_eth_incremental_binding_latest.json"
DEFAULT_EXP10_ETH_ACTIVE_GUARD = PROJECT_ROOT / "reports" / "default_exp10_bp0520_eth_active_guard_layers_latest.json"
DEFAULT_EXP10_ETH_EXECUTOR_ENV_AUDIT = PROJECT_ROOT / "reports" / "default_exp10_bp0520_eth_executor_env_audit_latest.json"
DEFAULT_EXP10_ETH_GUARD_EXECUTOR_COMPARE = PROJECT_ROOT / "reports" / "default_exp10_bp0520_eth_guard_executor_compare_latest.json"

# ─── 期望的合并交易进程组（9 个 multi_prediction_index 进程）────
EXPECTED_MERGED_GROUPS = {
    "v5_exp10": 11,   # 9 fixed bp + 2 dynamic
    "v5_exp11": 11,
    "v5_exp13": 11,
    "v5_exp14": 11,
    "v5_exp15": 11,
    "v5_exp16": 11,
    "v5_exp17": 11,
    "gru_all":  4,    # 4 GRU traders（gru_eth, gru_eth_no1h4h, gru_btc_no1h4h, gru_sol）
}
EXPECTED_TOTAL_TRADERS = sum(EXPECTED_MERGED_GROUPS.values())  # fallback only; active files override
ACTIVE_TRADER_FILE = POLYMARKET_DIR / "active_traders.json"
TRADER_CONFIG_FILE = POLYMARKET_DIR / "trader_configs.json"
ACTIVE_TRADER_FILE_70 = POLYMARKET_DIR / "active_traders_70.json"
TRADER_CONFIG_FILE_70 = POLYMARKET_DIR / "trader_configs_70.json"
CORE10_TRADE_TUNING_TARGETS_LATEST = PROJECT_ROOT / "reports" / "core10_promoted_trade_tuning_targets_latest.json"

EXPECTED_MERGED_GROUPS_70 = {
    "v5_exp10_70": 11,
    "v5_exp11_70": 11,
    "v5_exp13_70": 11,
    "v5_exp14_70": 11,
    "v5_exp15_70": 11,
    "v5_exp16_70": 11,
}
ALL_EXPECTED_MERGED_GROUPS = {**EXPECTED_MERGED_GROUPS, **EXPECTED_MERGED_GROUPS_70}

V5_FILE_TO_GROUP = {
    **{f"predictions_{group}.json": group for group, _label in LEGACY_V5_GROUP_LABELS},
    **{f"predictions_{group}.json": group for group, _label in LEGACY_V5_GROUP_LABELS_70},
    **WALLET_V5_FILE_TO_GROUP,
}

PREDICTION_FILE_TO_GROUP = {
    **V5_FILE_TO_GROUP,
    "predictions_ensemble.json": "ensemble",
    "predictions_ensemble_70.json": "ensemble_70",
    "predictions_gru_btc.json": "gru_all",
    "predictions_gru_eth.json": "gru_all",
    "predictions_gru_btc_no1h4h.json": "gru_all",
    "predictions_gru_eth_no1h4h.json": "gru_all",
    "predictions_gru_xrp.json": "gru_all",
    "predictions_gru_sol.json": "gru_all",
}
EXPECTED_INCREMENTAL_INTERVAL_MINUTES = 30
EXPECTED_BASE_ROUNDS = 12
EXPECTED_CALM_ROUNDS = 8
EXPECTED_SHOCK_ROUNDS = 20
EXPECTED_SHOCK_VOL_RATIO = 1.45
EXPECTED_SHOCK_RET_Q = 0.85
EXPECTED_ROLLBACK_UTILITY_DROP_ABS = 0.0
EXPECTED_ROLLBACK_DOWN_WR_DROP_ABS = 0.0
SCHEDULER_LAUNCHD_LABEL = "polyfun.daily_training_scheduler_abs"
RUNTIME_GUARD_STATUS_FILE = POLYMARKET_DIR / "logs" / "runtime_trade_guards.json"
RUNTIME_GUARD_STATUS_GLOB = "runtime_trade_guards*.json"
DATA_READY_FILE = PROJECT_ROOT / "data" / "data_ready.json"
RISK_TUNING_REPORTS = {
    "down": PROJECT_ROOT / "reports" / "down_risk_v2_tuning_report.json",
    "up": PROJECT_ROOT / "reports" / "up_risk_v2_tuning_report.json",
    "shock": PROJECT_ROOT / "reports" / "shock_risk_v2_tuning_report.json",
}
PREDICTION_V12_EVAL_LATEST = PROJECT_ROOT / "reports" / "prediction_v12_base_eval_latest.json"
RECENT_2DAY_DRAWDOWN_ROOTCAUSE_JSON = PROJECT_ROOT / "reports" / "recent_2day_drawdown_rootcause_latest.json"
MANUAL_ACTION_OVERLAY_LATEST = PROJECT_ROOT / "reports" / "manual_action_overlay_latest.json"
CORE_UNIVERSE_LATEST = PROJECT_ROOT / "reports" / "core_universe_latest.json"
CORE10_FOCUS_LATEST = PROJECT_ROOT / "reports" / "core10_focus_latest.json"
CORE10_MODEL_BOOST_SPEC_LATEST = PROJECT_ROOT / "reports" / "core10_model_boost_spec_latest.json"
CORE10_MODEL_BOOST_RUN_LATEST = PROJECT_ROOT / "reports" / "core10_model_boost_run_latest.json"
CORE10_SHADOW_COMPARE_LATEST = PROJECT_ROOT / "reports" / "core10_shadow_compare_latest.json"
CORE10_PROMOTION_STATE = PROJECT_ROOT / "reports" / "core10_promotion_state.json"
CORE10_INCREMENTAL_PROFILE = PROJECT_ROOT / "reports" / "core10_incremental_profile.json"
CORE10_INCREMENTAL_WRITEBACK_GUARD = PROJECT_ROOT / "reports" / "core10_incremental_writeback_guard_latest.json"
CORE10_INCREMENTAL_PARAM_APPLY = PROJECT_ROOT / "reports" / "core10_incremental_param_apply_latest.json"
CORE10_SOL_RESEARCH_REPORT = PROJECT_ROOT / "reports" / "core10_sol_research_latest.json"
DAILY_TRAINING_COVERAGE_REPORT = PROJECT_ROOT / "logs" / "daily_training_coverage_report.json"
ONLINE_LEARNING_HISTORY = PROJECT_ROOT / "logs" / "online_learning_history.jsonl"
INCREMENTAL_PARAM_SEARCH_RUNTIME_STATUS = PROJECT_ROOT / "reports" / "incremental_param_search_runtime_status_latest.json"
CORE10_SIMULATION_MONEY_RECONCILIATION = PROJECT_ROOT / "reports" / "core10_simulation_money_reconciliation_latest.json"
CORE10_GUARD_EFFECTIVENESS = PROJECT_ROOT / "reports" / "core10_guard_effectiveness_latest.json"
CORE10_LEAKAGE_AUDIT = PROJECT_ROOT / "reports" / "core10_leakage_audit_latest.json"
CORE10_PREDICTION_SOURCE_AUDIT = PROJECT_ROOT / "reports" / "core10_prediction_source_audit_latest.json"
CORE10_CLOSED_LOOP_AUDIT = PROJECT_ROOT / "reports" / "core10_only_closed_loop_audit_latest.json"
CORE10_LEGACY_CLEANUP_INVENTORY = PROJECT_ROOT / "reports" / "core10_legacy_cleanup_inventory_latest.json"
CORE10_FEATURE_COMPARE_LATEST = PROJECT_ROOT / "reports" / "core10_feature_compare_latest.json"
CORE10_HYPERPARAM_SPEC_LATEST = PROJECT_ROOT / "reports" / "core10_hyperparam_boost_spec_latest.json"
CORE10_HYPERPARAM_COMPARE_LATEST = PROJECT_ROOT / "reports" / "core10_hyperparam_compare_latest.json"
POLYFUN_MAINLINE_FINAL_STATUS = PROJECT_ROOT / "reports" / "polyfun_mainline_final_status_latest.json"
MAINLINE_RECENT_CHANGE_SEMANTIC_AUDIT = PROJECT_ROOT / "reports" / "mainline_recent_change_semantic_audit_latest.json"
MAINLINE_OPERATIONAL_BUG_AUDIT = PROJECT_ROOT / "reports" / "mainline_operational_bug_audit_latest.json"
MAINLINE_RUNTIME_GUARD_REVIEW = PROJECT_ROOT / "reports" / "mainline_runtime_guard_review_latest.json"
MAINLINE_POSTLAUNCH_GUARD_EFFECT = PROJECT_ROOT / "reports" / "mainline_postlaunch_guard_effect_review_latest.json"
MAINLINE_DEFAULT_WEAKNESS_REVIEW = PROJECT_ROOT / "reports" / "mainline_default_weakness_review_latest.json"
CORE10_WRITER_LAUNCHCTL_INVENTORY = PROJECT_ROOT / "reports" / "core10_writer_launchctl_inventory_latest.json"
SELECTOR_ZERO_TRADE_AUDIT = PROJECT_ROOT / "reports" / "selector_zero_trade_audit_latest.json"
MAINLINE_70_THRESHOLD_SEARCH = PROJECT_ROOT / "reports" / "mainline_70_btc_eth_threshold_search_latest.json"
RUNTIME_DATA_QUALITY_GATE_AUDIT = PROJECT_ROOT / "reports" / "runtime_data_quality_gate_audit_latest.json"
API_HEALTH_LOCK_FILE = PROJECT_ROOT / "logs" / "api_health_check.lock"
CORE10_TRADE_TUNING_LATEST = PROJECT_ROOT / "reports" / "core10_promoted_trade_tuning_latest.json"
CORE10_TRADE_RETUNE_READINESS_LATEST = PROJECT_ROOT / "reports" / "core10_promoted_trade_retune_readiness_latest.json"
CORE10_CONFIDENCE_BAND_WARMSTART_LATEST = PROJECT_ROOT / "reports" / "core10_confidence_band_warmstart_latest.json"
CORE10_CODE_AUDIT_LATEST = PROJECT_ROOT / "reports" / "core10_code_audit_latest.json"
CHALLENGER_INCREMENTAL_PROFILE = PROJECT_ROOT / "reports" / "challenger_incremental_profile.json"
COREMAIN_EXTREME_STATE_LATEST = PROJECT_ROOT / "reports" / "coremain_extreme_regime_state_latest.json"
COREMAIN_EXTREME_OVERLAY_APPLY_LATEST = PROJECT_ROOT / "reports" / "coremain_extreme_overlay_apply_latest.json"
CONFIDENCE_BAND_OVERLAP_AUDIT_LATEST = PROJECT_ROOT / "reports" / "confidence_band_overlap_audit_latest.json"
ALLORA_FEATURE_HEALTH_LATEST = PROJECT_ROOT / "reports" / "allora_feature_health_latest.json"
SHOCK_REJUDGE_GLOB = "shock_rejudge_*.json"
TOP20_CYCLE_STATE_GLOB = "top20_cycle_state_*.json"
TOP20_ROLLOUT_GLOB = "top20_rollout_plan_cycle*.json"
TOP20_AUDIT_GLOB = "ninelayer_conflict_audit_cycle*.json"
RISK_JOB_LOG_GLOBS = {
    "down": [PROJECT_ROOT / "reports" / "down_optimize*.log"],
    "up": [PROJECT_ROOT / "reports" / "up_optimize*.log"],
    "combo": [PROJECT_ROOT / "reports" / "combo_pause*.log"],
    "shock": [PROJECT_ROOT / "reports" / "shock_optimize*.log", PROJECT_ROOT / "reports" / "risk_pipeline*.log"],
    "calibration": [PROJECT_ROOT / "reports" / "*direction_calibration*.log"],
    "expectancy": [PROJECT_ROOT / "reports" / "*expectancy_gate*.log"],
    "joint": [PROJECT_ROOT / "reports" / "*joint_execution*.log"],
    "regime": [PROJECT_ROOT / "logs" / "regime_shadow*.log"],
    "watch": [PROJECT_ROOT / "logs" / "watch_up_down_risk*.log"],
}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return int(default)
    try:
        return int(float(raw))
    except ValueError:
        return int(default)


API_RETRY_ATTEMPTS = max(1, _env_int("HEALTH_API_RETRY_ATTEMPTS", 3))
API_RETRY_BASE_DELAY_SEC = max(0.05, float(os.environ.get("HEALTH_API_RETRY_BASE_DELAY_SEC", "0.35")))
CCXT_RETRY_ATTEMPTS = max(1, _env_int("HEALTH_CCXT_RETRY_ATTEMPTS", 3))
DAEMON_CACHE_WARMUP_SEC = max(10, _env_int("HEALTH_DAEMON_CACHE_WARMUP_SEC", 120))


DATA_QUALITY_MAX_AGE_SEC_EXPECTED = max(60, _env_int("DATA_QUALITY_MAX_AGE_SEC", 1200))
DATA_QUALITY_MAX_AGE_BY_SOURCE_SEC = {
    "open_interest": max(60, _env_int("DATA_QUALITY_MAX_AGE_OPEN_INTEREST_SEC", 600)),
    "long_short_ratio": max(60, _env_int("DATA_QUALITY_MAX_AGE_LONG_SHORT_RATIO_SEC", 600)),
    "polymarket_prob": max(60, _env_int("DATA_QUALITY_MAX_AGE_PM_PROB_SEC", 1800)),
    "polymarket_prob_target": max(60, _env_int("DATA_QUALITY_MAX_AGE_PM_TARGET_SEC", 1800)),
    "funding_rate": max(60, _env_int("DATA_QUALITY_MAX_AGE_FUNDING_RATE_SEC", 900)),
    "ob_realtime": max(60, _env_int("DATA_QUALITY_MAX_AGE_OB_REALTIME_SEC", 900)),
}
DATA_QUALITY_MTIME_FALLBACK_FILES = {
    "polymarket_prob": [
        PROJECT_ROOT / "data" / "sentiment" / "polymarket_prob_btc_usdt.parquet",
        PROJECT_ROOT / "data" / "sentiment" / "polymarket_prob_eth_usdt.parquet",
        PROJECT_ROOT / "data" / "sentiment" / "polymarket_prob_xrp_usdt.parquet",
    ],
    "polymarket_prob_target": [
        PROJECT_ROOT / "data" / "sentiment" / "polymarket_prob_target_btc_usdt.parquet",
        PROJECT_ROOT / "data" / "sentiment" / "polymarket_prob_target_eth_usdt.parquet",
        PROJECT_ROOT / "data" / "sentiment" / "polymarket_prob_target_xrp_usdt.parquet",
    ],
}
RUNTIME_GUARD_MAX_AGE_SEC = max(30, _env_int("RUNTIME_GUARD_MAX_AGE_SEC", 90))
RUNTIME_GUARD_PAYLOAD_MAX_AGE_SEC = max(60, _env_int("RUNTIME_GUARD_PAYLOAD_MAX_AGE_SEC", 180))
RUNTIME_GUARD_EVIDENCE_MIN_EVAL = max(50, _env_int("RUNTIME_GUARD_EVIDENCE_MIN_EVAL", 500))
DOWN_BREAKER_ALERT_MIN_ACTIVE = max(1, _env_int("HEALTH_DOWN_BREAKER_ALERT_MIN_ACTIVE", 3))
DOWN_BREAKER_ALERT_MIN_RATIO = max(
    0.0, min(1.0, float(os.environ.get("HEALTH_DOWN_BREAKER_ALERT_MIN_RATIO", "0.15")))
)


def _load_active_groups_from_file(
    active_file: Path,
    config_file: Path,
) -> Optional[set[str]]:
    if not active_file.exists():
        return None
    try:
        raw = json.loads(active_file.read_text(encoding="utf-8"))
    except Exception:
        return None
    names: set[str] = set()
    if isinstance(raw, dict):
        groups = raw.get("groups")
        if isinstance(groups, list):
            cleaned = {str(g).strip() for g in groups if str(g).strip()}
            if cleaned:
                return cleaned
        for key in ("active_traders", "traderNames"):
            vals = raw.get(key)
            if isinstance(vals, list):
                names.update(str(x).strip() for x in vals if str(x).strip())
    elif isinstance(raw, list):
        names.update(str(x).strip() for x in raw if str(x).strip())
    else:
        return None

    if not names:
        return None
    if not config_file.exists():
        return None
    try:
        cfg = json.loads(config_file.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(cfg, list):
        return None
    derived = {
        str(row.get("group")).strip()
        for row in cfg
        if isinstance(row, dict)
        and str(row.get("name", "")).strip() in names
        and str(row.get("group", "")).strip()
    }
    return derived or None


def _load_active_group_counts_from_file(
    active_file: Path,
    config_file: Path,
) -> Dict[str, int]:
    if not active_file.exists() or not config_file.exists():
        return {}
    try:
        raw = json.loads(active_file.read_text(encoding="utf-8"))
    except Exception:
        return {}

    names: set[str] = set()
    if isinstance(raw, dict):
        for key in ("active_traders", "traderNames"):
            vals = raw.get(key)
            if isinstance(vals, list):
                names.update(str(x).strip() for x in vals if str(x).strip())
    elif isinstance(raw, list):
        names.update(str(x).strip() for x in raw if str(x).strip())

    if not names:
        return {}

    try:
        cfg = json.loads(config_file.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(cfg, list):
        return {}

    counts: Dict[str, int] = {}
    for row in cfg:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        group = str(row.get("group", "")).strip()
        if not name or not group:
            continue
        if name not in names:
            continue
        counts[group] = counts.get(group, 0) + 1
    return counts


def _load_active_trader_names(active_file: Path) -> set[str]:
    if not active_file.exists():
        return set()
    try:
        raw = json.loads(active_file.read_text(encoding="utf-8"))
    except Exception:
        return set()
    names: set[str] = set()
    if isinstance(raw, dict):
        for key in ("active_traders", "traderNames"):
            vals = raw.get(key)
            if isinstance(vals, list):
                names.update(str(x).strip() for x in vals if str(x).strip())
    elif isinstance(raw, list):
        names.update(str(x).strip() for x in raw if str(x).strip())
    return names


def _iter_active_trader_rows() -> List[dict]:
    """返回当前 active(default+70) 对应的 trader 配置行。"""
    rows: List[dict] = []
    for active_file, config_file in (
        (ACTIVE_TRADER_FILE, TRADER_CONFIG_FILE),
        (ACTIVE_TRADER_FILE_70, TRADER_CONFIG_FILE_70),
    ):
        active_names = _load_active_trader_names(active_file)
        if not active_names or not config_file.exists():
            continue
        try:
            cfg = json.loads(config_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(cfg, list):
            continue
        for row in cfg:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", "")).strip()
            if name and name in active_names:
                rows.append(row)
    return rows


def _normalize_prediction_suffix_to_file(pred_suffix: str) -> str:
    s = str(pred_suffix or "").strip()
    if not s:
        return ""
    if s.startswith("predictions_") and s.endswith(".json"):
        return s
    if not s.startswith("_") and not s.startswith("predictions_"):
        s = f"_{s}"
    return f"predictions{s}.json"


def _load_active_prediction_files_from_configs() -> set[str]:
    """从 active traders 推导当前实际使用的预测文件名集合。"""
    out: set[str] = set()
    for row in _iter_active_trader_rows():
        fname = _normalize_prediction_suffix_to_file(str(row.get("predictionSuffix", "")))
        if fname:
            out.add(fname)
    return out


def _load_legacy_cleanup_rows() -> Dict[str, Dict[str, Any]]:
    path = CORE10_LEGACY_CLEANUP_INVENTORY
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    rows = payload.get("rows") if isinstance(payload, dict) else []
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = Path(str(row.get("prediction_file") or "")).name
        if name:
            out[name] = row
    return out


def _legacy_prediction_should_be_expected(prediction_file_name: str) -> bool:
    if not str(prediction_file_name or "").startswith("predictions_v5_"):
        return True
    row = _load_legacy_cleanup_rows().get(prediction_file_name)
    if not row:
        return True
    consumer_cells = int(row.get("consumer_cells") or 0)
    if consumer_cells > 0:
        return True
    action = str(row.get("action") or "")
    return action not in {"migrate_then_stop", "stop_now"}


def _static_v5_prediction_files() -> set[str]:
    return {
        *[fname for _, fname in EXPECTED_V5_WRITERS if _legacy_prediction_should_be_expected(fname)],
        *[fname for _, fname in EXPECTED_V5_WRITERS_70 if _legacy_prediction_should_be_expected(fname)],
        *WALLET_V5_FILE_TO_GROUP.keys(),
    }


def _dynamic_active_v5_writers() -> List[Tuple[str, str]]:
    entries: Dict[str, Tuple[str, str]] = {}
    static_files = _static_v5_prediction_files()
    for row in _iter_active_trader_rows():
        fname = _normalize_prediction_suffix_to_file(str(row.get("predictionSuffix", "")))
        if not fname or not fname.startswith("predictions_v5_"):
            continue
        if fname in static_files:
            continue
        name = str(row.get("name", "")).strip() or fname.replace(".json", "")
        group = str(row.get("group", "")).strip()
        entries[fname] = (name, group)
    return [(label, fname) for fname, (label, _group) in sorted(entries.items())]


def _dynamic_active_v5_file_to_group() -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    static_files = _static_v5_prediction_files()
    for row in _iter_active_trader_rows():
        fname = _normalize_prediction_suffix_to_file(str(row.get("predictionSuffix", "")))
        if not fname or not fname.startswith("predictions_v5_"):
            continue
        if fname in static_files:
            continue
        group = str(row.get("group", "")).strip()
        if group:
            mapping[fname] = group
    return mapping


def _load_active_symbols_from_configs() -> set[str]:
    """从 active traders 推导当前资产范围（BTC/ETH/XRP/SOL）。"""
    symbols: set[str] = set()
    for row in _iter_active_trader_rows():
        allowed = row.get("allowedMarkets")
        vals: List[str] = []
        if isinstance(allowed, list):
            vals = [str(x).strip().upper() for x in allowed if str(x).strip()]
        elif isinstance(allowed, str):
            vals = [x.strip().upper() for x in allowed.split(",") if x.strip()]
        for v in vals:
            if v in {"BTC", "ETH", "XRP", "SOL"}:
                symbols.add(v)
    return symbols


def _ordered_active_symbols() -> List[str]:
    symbols = _load_active_symbols_from_configs()
    ordered = [s for s in ("BTC", "ETH", "XRP", "SOL") if s in symbols]
    return ordered or ["BTC", "ETH"]


def _slot_scoped_live_health_only_enabled() -> bool:
    mode = os.environ.get("HEALTH_SCOPE_MODE", "").strip().lower()
    if mode in {"slot_live_eth_only", "slot_scoped_live_only", "slot_only"}:
        return True
    if mode in {"legacy", "all"}:
        return False
    profile = _read_report_json(CORE10_INCREMENTAL_PROFILE)
    summary = profile.get("summary") if isinstance(profile.get("summary"), dict) else {}
    return bool(summary.get("slot_scoped_live_only_mode"))


def _load_slot_scoped_live_targets() -> List[dict[str, Any]]:
    payload = _read_report_json(SLOT01_SLOT02_TARGET_STACK_CONTRACT)
    out: List[dict[str, Any]] = []
    for slot_key in ("slot01", "slot02"):
        row = payload.get(slot_key)
        if not isinstance(row, dict):
            continue
        operator_status = str(
            row.get("operatorStatus")
            or row.get("scientificEligibilityStatus")
            or row.get("liveConsumptionStatus")
            or ""
        ).strip().lower()
        if row.get("enabled") is False or operator_status == "disabled_by_operator":
            continue
        pred_output = str(row.get("predictionOutput") or "").strip()
        pred_suffix = str(row.get("predictionSuffix") or "").strip()
        pred_file = Path(pred_output).name if pred_output else _normalize_prediction_suffix_to_file(pred_suffix)
        if not pred_file:
            continue
        out.append(
            {
                "slot": slot_key,
                "label": slot_key.upper(),
                "prediction_file": pred_file,
                "group": str(row.get("group") or "").strip() or "lowprice_70_selected",
                "symbol": str(row.get("symbol") or "ETH").strip().upper(),
                "wallet_slot": str(row.get("walletSlot") or slot_key).strip(),
            }
        )
    return out


def _slot_scoped_health_prediction_files() -> List[Tuple[str, str]]:
    files: List[Tuple[str, str]] = []
    for row in _load_slot_scoped_live_targets():
        files.append((f"{row['label']} Live", str(row["prediction_file"])))
    return files


def _runtime_cluster_keys() -> tuple[str, ...]:
    keys: List[str] = []
    for profile in ("default", "70"):
        for symbol in _ordered_active_symbols():
            keys.append(f"{profile}_{symbol}")
    return tuple(keys)


def _load_active_scope_summary() -> Dict[str, object]:
    def _profile_summary(active_file: Path, config_file: Path) -> Dict[str, object]:
        names = _load_active_trader_names(active_file)
        groups = _load_active_groups_from_file(active_file, config_file) or set()
        cells = 0
        symbols: set[str] = set()
        mtime = active_file.stat().st_mtime if active_file.exists() else 0.0
        if names and config_file.exists():
            try:
                cfg = json.loads(config_file.read_text(encoding="utf-8"))
            except Exception:
                cfg = []
            if isinstance(cfg, list):
                for row in cfg:
                    if not isinstance(row, dict):
                        continue
                    name = str(row.get("name", "")).strip()
                    if name not in names:
                        continue
                    allowed = row.get("allowedMarkets")
                    vals: List[str] = []
                    if isinstance(allowed, list):
                        vals = [str(x).strip().upper() for x in allowed if str(x).strip()]
                    elif isinstance(allowed, str):
                        vals = [x.strip().upper() for x in allowed.split(",") if x.strip()]
                    cleaned = [v for v in vals if v in {"BTC", "ETH", "XRP", "SOL"}]
                    cells += len(cleaned) if cleaned else 1
                    symbols.update(cleaned)
        return {
            "traders": len(names),
            "cells": cells,
            "groups": sorted(groups),
            "symbols": sorted(symbols),
            "active_file": str(active_file),
            "mtime": mtime,
        }

    default_summary = _profile_summary(ACTIVE_TRADER_FILE, TRADER_CONFIG_FILE)
    prof70_summary = _profile_summary(ACTIVE_TRADER_FILE_70, TRADER_CONFIG_FILE_70)
    return {
        "default": default_summary,
        "70": prof70_summary,
        "total_traders": int(default_summary["traders"]) + int(prof70_summary["traders"]),
        "total_cells": int(default_summary["cells"]) + int(prof70_summary["cells"]),
    }


def _first_present(*values):
    for value in values:
        if value is not None:
            return value
    return None


def _report_age_hours(path: Path) -> float | None:
    if not path.exists():
        return None
    return round((time.time() - path.stat().st_mtime) / 3600, 2)


def _is_report_fresh(path: Path, max_age_sec: int = MAINLINE_REPORT_FRESH_MAX_AGE_SEC) -> bool:
    if not path.exists():
        return False
    return (time.time() - path.stat().st_mtime) <= max_age_sec


def _load_core_universe_summary() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "present": False,
        "core_cells": 0,
        "shadow_cells": 0,
        "core_traders": 0,
        "shadow_traders": 0,
        "core_recent_positive": {},
        "core_top_drag": None,
        "core_by_profile": {},
        "shadow_by_profile": {},
    }
    if not CORE_UNIVERSE_LATEST.exists():
        return out
    try:
        payload = json.loads(CORE_UNIVERSE_LATEST.read_text(encoding="utf-8"))
    except Exception:
        return out
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    out["present"] = True
    for key in ("core_cells", "shadow_cells", "core_traders", "shadow_traders", "core_recent_positive", "core_top_drag", "core_by_profile", "shadow_by_profile"):
        if key in summary:
            out[key] = summary.get(key)
    return out


def _load_core10_focus_summary() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "present": False,
        "core_cells": 0,
        "current_core_size": 0,
        "shadow_cells": 0,
        "core_target_min": 0,
        "core_target_max": 0,
        "core_size_mode": "",
        "layer_summary": {},
        "optimized_core_rollout_summary": {},
        "top_core_cells": [],
        "rotation_watchlist_summary": {},
        "top_challenger": None,
        "top_rotation_recommendation": None,
        "top_rotation_ready_recommendation": None,
        "recent_window_watch_summary": {},
        "top_recent_drag_24h": None,
        "top_recent_drag_6h": None,
        "challenger_profile_summary": {},
        "rally_state_summary": {},
        "extreme_state_summary": {},
        "extreme_overlay_summary": {},
        "confidence_band_overlap_summary": {},
        "allora_feature_health_summary": {},
        "regime_pnl_attribution_summary": {},
        "backlog_reconciliation_summary": {},
        "runtime_param_search_summary": {},
        "final_logic_audit_summary": {},
        "retrain_manifest_summary": {},
        "retrain_research_summary": {},
        "tree_candidate_summary": {},
        "sequence_candidate_summary": {},
        "cell_promotion_gate_summary": {},
        "retrain_failure_analysis_summary": {},
        "phase2_score_calibration_summary": {},
        "phase2_tree_hparam_summary": {},
        "phase2_sequence_hparam_summary": {},
        "phase2_promotion_gate_summary": {},
        "stage3_specialist_summary": {},
        "recent_strict_probe_summary": {},
        "stage4_action_probe_summary": {},
        "stage5_market_action_search_summary": {},
        "stage5_market_action_probe_summary": {},
        "stage6_market_final_action_search_summary": {},
        "stage6_market_final_action_probe_summary": {},
        "stage8_regime_cluster_action_search_summary": {},
        "stage8_regime_cluster_action_probe_summary": {},
        "stage9_precision_action_search_summary": {},
        "stage9_precision_action_probe_summary": {},
        "stage10_state_expert_search_summary": {},
        "stage10_state_expert_probe_summary": {},
        "stage11_state_expert_policy_summary": {},
    }
    if not CORE10_FOCUS_LATEST.exists():
        return out
    try:
        payload = json.loads(CORE10_FOCUS_LATEST.read_text(encoding="utf-8"))
    except Exception:
        return out
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    out["present"] = True
    for key in (
        "core_cells",
        "current_core_size",
        "shadow_cells",
        "core_target_min",
        "core_target_max",
        "core_size_mode",
        "layer_summary",
        "optimized_core_rollout_summary",
        "top_core_cells",
        "rotation_watchlist_summary",
        "top_challenger",
        "top_rotation_recommendation",
        "top_rotation_ready_recommendation",
        "recent_window_watch_summary",
        "top_recent_drag_24h",
        "top_recent_drag_6h",
        "challenger_profile_summary",
        "rally_state_summary",
        "extreme_state_summary",
        "extreme_overlay_summary",
        "confidence_band_overlap_summary",
        "allora_feature_health_summary",
        "regime_pnl_attribution_summary",
        "backlog_reconciliation_summary",
        "runtime_param_search_summary",
        "final_logic_audit_summary",
        "retrain_manifest_summary",
        "retrain_research_summary",
        "tree_candidate_summary",
        "sequence_candidate_summary",
        "cell_promotion_gate_summary",
        "retrain_failure_analysis_summary",
        "phase2_score_calibration_summary",
        "phase2_tree_hparam_summary",
        "phase2_sequence_hparam_summary",
        "phase2_promotion_gate_summary",
        "stage3_specialist_summary",
        "recent_strict_probe_summary",
        "stage4_action_probe_summary",
        "stage5_market_action_search_summary",
        "stage5_market_action_probe_summary",
        "stage6_market_final_action_search_summary",
        "stage6_market_final_action_probe_summary",
        "stage8_regime_cluster_action_search_summary",
        "stage8_regime_cluster_action_probe_summary",
        "stage9_precision_action_search_summary",
        "stage9_precision_action_probe_summary",
        "stage10_state_expert_search_summary",
        "stage10_state_expert_probe_summary",
        "stage11_state_expert_policy_summary",
    ):
        if key in summary:
            out[key] = summary.get(key)
    return out


def _load_core10_model_boost_summary() -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "present": False,
        "jobs_total": 0,
        "jobs_trained": 0,
        "jobs_shadow_compared": 0,
        "jobs_promoted": 0,
        "feature_variants_total": 0,
        "feature_variants_trained": 0,
        "feature_variants_promising_vs_base": 0,
        "feature_variants_promising_vs_shared": 0,
        "hyperparam_candidates_total": 0,
        "hyperparam_candidates_trained": 0,
        "hyperparam_promising_vs_source": 0,
        "hyperparam_promising_vs_shared": 0,
        "trade_tuning_target_cells": 0,
        "trade_tuning_optimize_success": 0,
        "trade_tuning_apply_success": 0,
        "trade_tuning_status": "",
        "trade_retune_ready": False,
        "trade_retune_ready_jobs": 0,
        "trade_retune_fresh_trades": 0,
        "trade_retune_fresh_settled": 0,
        "confidence_band_applied": 0,
        "incremental_training_mode": "shared_or_legacy",
        "incremental_writeback_policy": "",
        "writeback_guard_status": "",
        "writeback_guard_units_total": 0,
        "writeback_guard_allowed_units": 0,
        "writeback_guard_trained_not_written_units": 0,
        "writeback_guard_stale_warning_units": 0,
        "writeback_guard_stale_critical_units": 0,
        "writeback_guard_baseline_insufficient_units": 0,
        "writeback_guard_observed_but_baseline_insufficient_units": 0,
        "writeback_guard_real_money_weakened_units": 0,
        "writeback_guard_effective_weakened_units": 0,
        "writeback_guard_effective_weakened_with_baseline_insufficient_units": 0,
        "writeback_guard_challenger_trigger_units": 0,
        "writeback_guard_effective_challenger_trigger_units": 0,
        "writeback_guard_effective_challenger_trigger_with_baseline_insufficient_units": 0,
        "writeback_guard_oldest_successful_writeback_age_hours": None,
        "incremental_param_search_status": "",
        "incremental_param_search_families": 0,
        "incremental_param_multi_candidate_ready_families": 0,
        "incremental_param_apply_status": "",
        "incremental_param_apply_families": 0,
        "incremental_param_ready_to_apply_families": 0,
        "incremental_param_applied_families": 0,
        "simulation_money_reconciliation_status": "",
        "simulation_money_reconciliation_families": 0,
        "simulation_money_reconciliation_warning_families": 0,
        "guard_effectiveness_status": "",
        "guard_effectiveness_groups_total": 0,
        "guard_effectiveness_groups_fresh": 0,
        "guard_effectiveness_groups_stale": 0,
        "guard_effectiveness_data_quality_ok_groups": 0,
        "prediction_source_audit_status": "",
        "prediction_source_target_traders": 0,
        "prediction_source_core10_family_writers": 0,
        "prediction_source_legacy_writers": 0,
        "prediction_source_main_v5_use_core10_family": False,
        "prediction_source_main_v5_use_legacy_dirs": False,
        "prediction_source_conclusion": "",
        "leakage_audit_status": "",
        "leakage_audit_windows_pass": 0,
        "leakage_audit_windows_total": 0,
        "leakage_audit_future_like_column_count": 0,
        "challenger_status": "",
        "challenger_route": "",
        "challenger_target_cell": "",
        "extreme_active_symbols": 0,
        "countertrend_hard_block_cells": 0,
        "confidence_band_overlap_count_core": 0,
        "challenger_bootstrap_success": 0,
        "allora_feature_freshness_sec": None,
        "legacy_optimizer_running": False,
        "core_fast_lane_total": 0,
        "core_fast_lane_covered": 0,
        "core_fast_lane_deferred": 0,
        "allora_auth_ok": False,
        "runtime_sidecar_marker_failures": 0,
        "selective_abstain_targets": 0,
        "selective_abstain_applied": 0,
        "uncertainty_gate_targets": 0,
        "uncertainty_gate_applied": 0,
        "uncertainty_interval_targets": 0,
        "uncertainty_interval_blocked_manual_actions": 0,
        "guard_stack_hard_blocked_directions": 0,
        "guard_stack_overcompressed_follow_directions": 0,
        "guard_stack_logic_conflict_directions": 0,
        "guard_stack_dead_core_cells": 0,
        "follow_lane_repair_targets": 0,
        "follow_lane_repair_applied": 0,
        "follow_lane_repair_drift_count": 0,
        "regime_one_sided_symbols": 0,
        "regime_neutral_symbols": 0,
        "regime_flip_risk_symbols": 0,
        "feature_trade_objective_weighted_variants": 0,
        "core10_hparam_phase": "none",
        "core10_extreme_guard_status": "idle",
        "backlog_open_items": 0,
        "backlog_partial_items": 0,
        "backlog_reconciliation_status": "",
        "runtime_param_search_status": "",
        "runtime_param_search_groups": 0,
        "runtime_param_search_recommended_groups": 0,
        "runtime_param_search_applied_groups": 0,
        "exp13_up_lane_search_status": "",
        "exp13_up_lane_search_candidates": 0,
        "exp13_up_lane_search_applied": 0,
        "exp15_up_lane_search_status": "",
        "exp15_up_lane_search_candidates": 0,
        "exp15_up_lane_search_applied": 0,
        "directional_drag_bad_lanes": 0,
        "default_eth_down_tuning_status": "",
        "default_eth_down_tuning_candidates": 0,
        "default_eth_down_tuning_recommended": 0,
        "default_eth_down_tuning_applied": 0,
        "regime_direction_structural_weakness": 0,
        "regime_direction_regime_conditioned": 0,
        "exp10_direction_tuning_status": "",
        "exp10_direction_tuning_candidates": 0,
        "exp10_direction_tuning_recommended": 0,
        "exp10_direction_tuning_applied": 0,
        "protected_band_caps_status": "",
        "protected_band_caps_applied": 0,
        "protected_band_drift_count": 0,
        "recent_sample_review_status": "",
        "recent_sample_last_3h_pnl": 0.0,
        "recent_sample_prev_3h_pnl": 0.0,
        "recent_sample_last_6h_pnl": 0.0,
        "recent_sample_prev_6h_pnl": 0.0,
        "recent_sample_last_6h_wr": None,
        "recent_sample_prev_6h_wr": None,
        "final_logic_audit_status": "",
        "final_logic_audit_blockers": [],
        "core10_retrain_pipeline_status": "",
        "tree_candidates_trained": 0,
        "sequence_candidates_trained": 0,
        "cells_promotion_ready": 0,
        "core10_phase2_status": "",
        "exp13_phase2_ready": False,
        "phase2_candidates_promotion_ready": 0,
        "no_trade_collapse_cells": 0,
        "stage3_specialist_status": "",
        "stage3_specialist_promotion_ready": 0,
        "recent_strict_probe_status": "",
        "recent_strict_probe_exp13_ready": False,
        "recent_strict_probe_expansion_allowed": False,
        "stage4_action_probe_status": "",
        "stage4_action_probe_exp13_ready": False,
        "stage4_action_probe_expansion_allowed": False,
        "stage5_market_action_search_status": "",
        "stage5_market_action_probe_status": "",
        "stage5_market_action_exp13_ready": False,
        "stage5_market_action_expansion_allowed": False,
        "stage6_market_final_action_search_status": "",
        "stage6_market_final_action_probe_status": "",
        "stage6_market_final_action_exp13_ready": False,
        "stage6_market_final_action_expansion_allowed": False,
        "stage8_regime_cluster_action_search_status": "",
        "stage8_regime_cluster_action_probe_status": "",
        "stage8_regime_cluster_action_exp13_ready": False,
        "stage8_regime_cluster_action_expansion_allowed": False,
        "stage9_precision_action_search_status": "",
        "stage9_precision_action_probe_status": "",
        "stage9_precision_action_exp13_ready": False,
        "stage9_precision_action_expansion_allowed": False,
        "stage10_state_expert_search_status": "",
        "stage10_state_expert_probe_status": "",
        "stage10_state_expert_exp13_ready": False,
        "stage10_state_expert_expansion_allowed": False,
        "stage11_state_expert_policy_status": "",
        "stage11_state_expert_policy_exp13_ready": False,
        "stage11_state_expert_policy_expansion_allowed": False,
        "stage12_lane_split_status": "",
        "stage12_lane_split_exp13_ready": False,
        "stage12_lane_split_rally_up_ready": False,
        "stage12_lane_split_rally_down_ready": False,
        "stage13_rally_up_precision_status": "",
        "stage13_rally_up_precision_exp13_ready": False,
        "stage13_rally_up_precision_promotion_ready": 0,
        "stage14_rally_up_teacher_residual_status": "",
        "stage14_rally_up_teacher_residual_exp13_ready": False,
        "stage14_rally_up_teacher_residual_promotion_ready": 0,
        "stage15_event_ranker_status": "",
        "stage15_event_ranker_exp13_ready": False,
        "stage15_event_ranker_promotion_ready": 0,
        "stage16_event_precision_status": "",
        "stage16_event_precision_exp13_ready": False,
        "stage16_event_precision_promotion_ready": 0,
        "stage16_shadow_gate_status": "",
        "stage16_shadow_gate_ready": 0,
        "stage16_shadow_gate_best_ready": False,
        "stage16_shadow_compare_status": "",
        "stage16_shadow_compare_ready": 0,
        "stage16_shadow_compare_shadow_simulation_ready": 0,
        "stage16_shadow_compare_best_ready": False,
        "stage16_shadow_compare_best_shadow_simulation_ready": False,
        "stage16_shadow_sim_candidate_status": "",
        "stage16_shadow_sim_candidate_ready": False,
        "stage16_shadow_sim_candidate_count": 0,
        "stage16_shadow_live_watch_status": "",
        "stage16_shadow_live_watch_last_24h_pnl": 0.0,
        "stage16_shadow_live_watch_last_72h_pnl": 0.0,
        "stage16_shadow_live_watch_latest_would_trade": False,
        "stage16_shadow_live_watch_caution": "",
        "stage16_shadow_live_compare_status": "",
        "stage16_shadow_live_compare_last_24h_vs_active_pnl_delta": 0.0,
        "stage16_shadow_live_compare_last_72h_vs_active_pnl_delta": 0.0,
        "stage16_shadow_live_compare_tail_96_vs_active_pnl_delta": 0.0,
        "stage16_shadow_stability_status": "",
        "stage16_shadow_stability_label": "",
        "stage16_shadow_stable_candidate": False,
        "stage16_shadow_cautious_candidate": False,
        "exp13_full_stack_replay_status": "",
        "exp13_full_stack_replay_candidate_ready": False,
        "exp13_full_stack_replay_verdict": "",
        "exp13_full_stack_replay_candidate_recent14_pnl": 0.0,
        "exp13_full_stack_replay_active_recent14_pnl": 0.0,
        "exp13_full_stack_replay_candidate_recent14_score": 0.0,
        "exp13_full_stack_replay_active_recent14_score": 0.0,
        "exp13_full_stack_replay_candidate_strict_score": 0.0,
        "exp13_full_stack_replay_active_strict_score": 0.0,
        "exp13_official_sim_candidate_status": "",
        "exp13_official_sim_candidate_qualified": False,
        "exp13_official_sim_candidate_live": False,
        "exp13_official_sim_candidate_prediction_file_exists": False,
        "exp13_official_sim_candidate_prediction_file_age_sec": None,
        "exp13_official_sim_candidate_recent14_pnl": 0.0,
        "exp13_official_sim_candidate_strict_score": 0.0,
        "exp13_official_sim_candidate_verdict": "",
        "route_tournament_status": "",
        "route_tournament_current_leader": "",
        "route_tournament_final_verdict": "",
        "route_rollout_status": "",
        "route_rollout_winning_route": "",
        "route_e_status": "",
        "route_e_target_cell": "",
        "route_e_pilot_ready": False,
        "route_e_pilot_status": "",
        "route_e_pilot_guard_depth_best": 0,
        "route_e_pilot_guard_name_best": "",
        "route_e_pilot_recent14_pnl": 0.0,
        "route_e_pilot_strict_pnl": 0.0,
        "route_e_pilot_full_pnl": 0.0,
        "route_e_pilot_recent14_score": 0.0,
        "route_e_pilot_strict_score": 0.0,
        "route_e_pilot_full_score": 0.0,
        "route_e_pilot_drawdown_proxy": 0.0,
        "route_e_pilot_coverage_pass": False,
        "route_e_pilot_low_support_benchmark": False,
        "route_e_pilot_benchmark_recent14_pnl": 0.0,
        "route_e_pilot_benchmark_strict_pnl": 0.0,
        "route_e_pilot_benchmark_full_drawdown_proxy": 0.0,
        "route_e_official_sim_status": "",
        "route_e_official_sim_qualified": False,
        "route_e_official_sim_live": False,
        "route_e_official_sim_prediction_file_exists": False,
        "route_e_official_sim_prediction_file_age_sec": None,
        "route_e_official_sim_recent14_pnl": 0.0,
        "route_e_official_sim_strict_pnl": 0.0,
        "route_e_official_sim_full_pnl": 0.0,
        "route_e_official_sim_guard_depth_best": 0,
        "route_e_official_sim_verdict": "",
        "route_e_core10_sweep_status": "",
        "route_e_core10_sweep_ready_cells": 0,
        "route_e_core10_sweep_total_cells": 0,
        "route_e_core10_sweep_verdict": "",
        "route_e_rollout_status": "",
        "route_e_rollout_live": False,
        "route_e_rollout_verdict": "",
        "route_f_status": "",
        "route_f_target_cell": "",
        "route_f_pilot_ready": False,
        "route_f_pilot_status": "",
        "route_f_pilot_guard_depth_best": 0,
        "route_f_pilot_guard_name_best": "",
        "route_f_pilot_recent14_capital_change": 0.0,
        "route_f_pilot_strict_capital_change": 0.0,
        "route_f_pilot_full_capital_change": 0.0,
        "route_f_pilot_recent14_win_rate": 0.0,
        "route_f_pilot_strict_win_rate": 0.0,
        "route_f_pilot_full_win_rate": 0.0,
        "route_f_pilot_full_drawdown_capital": 0.0,
        "route_f_pilot_coverage_pass": False,
        "route_f_pilot_low_support_benchmark": False,
        "route_f_pilot_benchmark_recent14_capital_change": 0.0,
        "route_f_pilot_benchmark_strict_capital_change": 0.0,
        "route_f_pilot_benchmark_full_capital_change": 0.0,
        "route_f_pilot_benchmark_recent14_win_rate": 0.0,
        "route_f_pilot_benchmark_strict_win_rate": 0.0,
        "route_f_pilot_benchmark_full_win_rate": 0.0,
        "route_f_pilot_benchmark_full_drawdown_capital": 0.0,
        "route_f_official_sim_status": "",
        "route_f_official_sim_qualified": False,
        "route_f_official_sim_live": False,
        "route_f_official_sim_prediction_file_exists": False,
        "route_f_official_sim_prediction_file_age_sec": None,
        "route_f_official_sim_recent14_capital_change": 0.0,
        "route_f_official_sim_strict_capital_change": 0.0,
        "route_f_official_sim_full_capital_change": 0.0,
        "route_f_official_sim_guard_depth_best": 0,
        "route_f_official_sim_verdict": "",
        "promotion_gate_fairness_status": "",
        "promotion_gate_sample_bias_rows": 0,
        "promotion_gate_active_small_sample_rows": 0,
    }
    try:
        if CORE10_MODEL_BOOST_SPEC_LATEST.exists():
            spec = json.loads(CORE10_MODEL_BOOST_SPEC_LATEST.read_text(encoding="utf-8"))
            out["jobs_total"] = int(((spec.get("summary") or {}).get("trainable_jobs") or 0))
            out["present"] = True
        if CORE10_SHADOW_COMPARE_LATEST.exists():
            shadow = json.loads(CORE10_SHADOW_COMPARE_LATEST.read_text(encoding="utf-8"))
            summary = shadow.get("summary") if isinstance(shadow.get("summary"), dict) else {}
            out["jobs_trained"] = int(summary.get("jobs_trained") or 0)
            out["jobs_shadow_compared"] = int(summary.get("jobs_shadow_compared") or 0)
            out["jobs_promoted"] = int(summary.get("jobs_promoted") or 0)
            out["present"] = True
        if CORE10_INCREMENTAL_PROFILE.exists():
            profile = json.loads(CORE10_INCREMENTAL_PROFILE.read_text(encoding="utf-8"))
            raw_mode = str(profile.get("mode") or "mainline_runtime_pool")
            out["incremental_training_mode"] = "mainline_runtime_pool" if raw_mode == "core10_only" else raw_mode
            out["incremental_writeback_policy"] = str(profile.get("writeback_policy") or "")
            out["core_fast_lane_total"] = int(((profile.get("summary") or {}).get("jobs_total") or 0))
            out["core_fast_lane_covered"] = int(((profile.get("summary") or {}).get("covered_jobs") or 0))
            out["core_fast_lane_deferred"] = int(((profile.get("summary") or {}).get("deferred_core_jobs") or 0))
            out["present"] = True
        if CORE10_INCREMENTAL_WRITEBACK_GUARD.exists():
            writeback_guard = json.loads(CORE10_INCREMENTAL_WRITEBACK_GUARD.read_text(encoding="utf-8"))
            wb_summary = writeback_guard.get("summary") if isinstance(writeback_guard.get("summary"), dict) else {}
            out["writeback_guard_status"] = str(writeback_guard.get("status") or "")
            out["writeback_guard_units_total"] = int(wb_summary.get("units_total") or 0)
            out["writeback_guard_allowed_units"] = int(wb_summary.get("writeback_allowed_latest_units") or 0)
            out["writeback_guard_trained_not_written_units"] = int(wb_summary.get("trained_but_not_written_latest_units") or 0)
            out["writeback_guard_stale_warning_units"] = int(wb_summary.get("stale_warning_units") or 0)
            out["writeback_guard_stale_critical_units"] = int(wb_summary.get("stale_critical_units") or 0)
            out["writeback_guard_baseline_insufficient_units"] = int(wb_summary.get("baseline_insufficient_units") or 0)
            out["writeback_guard_observed_but_baseline_insufficient_units"] = int(wb_summary.get("observed_but_baseline_insufficient_units") or 0)
            out["writeback_guard_real_money_weakened_units"] = int(wb_summary.get("real_money_weakened_units") or 0)
            out["writeback_guard_effective_weakened_units"] = int(_first_present(wb_summary.get("effective_weakened_units"), wb_summary.get("real_money_weakened_units"), 0))
            out["writeback_guard_effective_weakened_with_baseline_insufficient_units"] = int(wb_summary.get("effective_weakened_with_baseline_insufficient_units") or 0)
            out["writeback_guard_challenger_trigger_units"] = int(wb_summary.get("challenger_trigger_units") or 0)
            out["writeback_guard_effective_challenger_trigger_units"] = int(_first_present(wb_summary.get("effective_challenger_trigger_units"), wb_summary.get("challenger_trigger_units"), 0))
            out["writeback_guard_effective_challenger_trigger_with_baseline_insufficient_units"] = int(wb_summary.get("effective_challenger_trigger_with_baseline_insufficient_units") or 0)
            out["writeback_guard_oldest_successful_writeback_age_hours"] = wb_summary.get("oldest_successful_writeback_age_hours")
            out["present"] = True
        if CORE10_INCREMENTAL_PARAM_APPLY.exists():
            param_apply = json.loads(CORE10_INCREMENTAL_PARAM_APPLY.read_text(encoding="utf-8"))
            pa_summary = param_apply.get("summary") if isinstance(param_apply.get("summary"), dict) else {}
            out["incremental_param_search_status"] = str(pa_summary.get("param_search_report_status") or pa_summary.get("param_search_status") or "")
            out["incremental_param_search_families"] = int(
                pa_summary.get("param_search_report_family_symbol_keys_total")
                or pa_summary.get("families_total")
                or 0
            )
            out["incremental_param_multi_candidate_ready_families"] = int(
                pa_summary.get("param_search_report_multi_candidate_ready_total") or 0
            )
            out["incremental_param_apply_status"] = str(pa_summary.get("param_apply_status") or param_apply.get("status") or "")
            out["incremental_param_apply_families"] = int(
                pa_summary.get("param_apply_family_symbol_keys_total")
                or pa_summary.get("families_total")
                or 0
            )
            out["incremental_param_ready_to_apply_families"] = int(
                pa_summary.get("param_apply_ready_to_apply_total")
                or pa_summary.get("applied_family_symbol_keys_total")
                or 0
            )
            out["incremental_param_applied_families"] = int(
                pa_summary.get("param_apply_applied_total")
                or pa_summary.get("applied_family_symbol_keys_total")
                or 0
            )
            out["present"] = True
        if CORE10_SIMULATION_MONEY_RECONCILIATION.exists():
            money = json.loads(CORE10_SIMULATION_MONEY_RECONCILIATION.read_text(encoding="utf-8"))
            money_summary = money.get("summary") if isinstance(money.get("summary"), dict) else {}
            out["simulation_money_reconciliation_status"] = str(money.get("status") or "")
            out["simulation_money_reconciliation_families"] = int(money_summary.get("families_total") or 0)
            out["simulation_money_reconciliation_warning_families"] = int(money_summary.get("families_warning") or 0)
            out["present"] = True
        if CORE10_GUARD_EFFECTIVENESS.exists():
            guard_eff = json.loads(CORE10_GUARD_EFFECTIVENESS.read_text(encoding="utf-8"))
            ge_summary = guard_eff.get("summary") if isinstance(guard_eff.get("summary"), dict) else {}
            out["guard_effectiveness_status"] = str(guard_eff.get("status") or "")
            out["guard_effectiveness_groups_total"] = int(ge_summary.get("groups_total") or 0)
            out["guard_effectiveness_groups_fresh"] = int(ge_summary.get("groups_fresh") or 0)
            out["guard_effectiveness_groups_stale"] = int(ge_summary.get("groups_stale") or 0)
            out["guard_effectiveness_data_quality_ok_groups"] = int(ge_summary.get("data_quality_ok_groups") or 0)
            out["present"] = True
        if CORE10_PREDICTION_SOURCE_AUDIT.exists():
            source_audit = json.loads(CORE10_PREDICTION_SOURCE_AUDIT.read_text(encoding="utf-8"))
            ps_summary = source_audit.get("summary") if isinstance(source_audit.get("summary"), dict) else {}
            out["prediction_source_audit_status"] = str(source_audit.get("status") or "")
            out["prediction_source_target_traders"] = int(ps_summary.get("target_trader_count") or 0)
            out["prediction_source_core10_family_writers"] = int(ps_summary.get("writers_using_expected_core10_family") or 0)
            out["prediction_source_legacy_writers"] = int(ps_summary.get("writers_using_legacy_dir") or 0)
            out["prediction_source_main_v5_use_core10_family"] = bool(ps_summary.get("main_v5_traders_use_core10_family"))
            out["prediction_source_main_v5_use_legacy_dirs"] = bool(ps_summary.get("main_v5_traders_use_legacy_dirs"))
            out["prediction_source_conclusion"] = str(ps_summary.get("current_conclusion") or "")
            out["present"] = True
        if CORE10_LEAKAGE_AUDIT.exists():
            leakage = json.loads(CORE10_LEAKAGE_AUDIT.read_text(encoding="utf-8"))
            leakage_summary = leakage.get("summary") if isinstance(leakage.get("summary"), dict) else {}
            out["leakage_audit_status"] = "pass" if bool(leakage_summary.get("pass")) else ("fail" if leakage_summary else "")
            out["leakage_audit_windows_pass"] = int(leakage_summary.get("windows_pass") or 0)
            out["leakage_audit_windows_total"] = int(leakage_summary.get("windows_total") or 0)
            out["leakage_audit_future_like_column_count"] = int(leakage_summary.get("future_like_column_count") or 0)
            out["present"] = True
        if CORE10_FEATURE_COMPARE_LATEST.exists():
            feature = json.loads(CORE10_FEATURE_COMPARE_LATEST.read_text(encoding="utf-8"))
            summary = feature.get("summary") if isinstance(feature.get("summary"), dict) else {}
            out["feature_variants_total"] = int(summary.get("variants_total") or 0)
            out["feature_variants_trained"] = int(summary.get("trained") or 0)
            out["feature_variants_promising_vs_base"] = int(summary.get("promising") or 0)
            out["feature_variants_promising_vs_shared"] = int(summary.get("promising_vs_shared") or 0)
            out["present"] = True
        if CORE10_HYPERPARAM_COMPARE_LATEST.exists():
            hparam = json.loads(CORE10_HYPERPARAM_COMPARE_LATEST.read_text(encoding="utf-8"))
            summary = hparam.get("summary") if isinstance(hparam.get("summary"), dict) else {}
            out["hyperparam_candidates_total"] = int(summary.get("candidates_total") or 0)
            out["hyperparam_candidates_trained"] = int(summary.get("trained") or 0)
            out["hyperparam_promising_vs_source"] = int(summary.get("promising_vs_source") or 0)
            out["hyperparam_promising_vs_shared"] = int(summary.get("promising_vs_shared") or 0)
            if out["hyperparam_candidates_total"] <= 0:
                out["core10_hparam_phase"] = "none"
            elif out["hyperparam_promising_vs_shared"] > 0:
                out["core10_hparam_phase"] = "promising_ready"
            elif out["hyperparam_candidates_trained"] > 0:
                out["core10_hparam_phase"] = "trained_no_promotion"
            else:
                out["core10_hparam_phase"] = "running"
            out["present"] = True
        elif CORE10_PROMOTION_STATE.exists():
            promotion = json.loads(CORE10_PROMOTION_STATE.read_text(encoding="utf-8"))
            out["jobs_promoted"] = max(
                int(out.get("jobs_promoted") or 0),
                int(((promotion.get("summary") or {}).get("promoted") or 0)),
            )
            out["present"] = True
        if CORE10_TRADE_TUNING_LATEST.exists():
            tuning = json.loads(CORE10_TRADE_TUNING_LATEST.read_text(encoding="utf-8"))
            summary = tuning.get("summary") if isinstance(tuning.get("summary"), dict) else {}
            out["trade_tuning_target_cells"] = int(summary.get("target_cells") or 0)
            out["trade_tuning_optimize_success"] = int(summary.get("optimize_success") or 0)
            out["trade_tuning_apply_success"] = int(summary.get("apply_success") or 0)
            trade_status = str(tuning.get("status") or "").strip()
            if not trade_status:
                optimize_success = int(summary.get("optimize_success") or 0)
                optimize_failed = int(summary.get("optimize_failed") or 0)
                apply_success = int(summary.get("apply_success") or 0)
                apply_failed = int(summary.get("apply_failed") or 0)
                if apply_success > 0 and optimize_failed == 0 and apply_failed == 0:
                    trade_status = "applied"
                elif apply_success > 0:
                    trade_status = "partial_applied"
                elif optimize_success > 0:
                    trade_status = "optimized_only"
                elif optimize_failed > 0 or apply_failed > 0:
                    trade_status = "failed"
            out["trade_tuning_status"] = trade_status
            out["present"] = True
        if CORE10_TRADE_RETUNE_READINESS_LATEST.exists():
            readiness = json.loads(CORE10_TRADE_RETUNE_READINESS_LATEST.read_text(encoding="utf-8"))
            summary = readiness.get("summary") if isinstance(readiness.get("summary"), dict) else {}
            out["trade_retune_ready"] = bool(summary.get("full_retune_ready"))
            out["trade_retune_ready_jobs"] = int(summary.get("ready_jobs") or 0)
            out["trade_retune_fresh_trades"] = int(summary.get("fresh_trades_total") or 0)
            out["trade_retune_fresh_settled"] = int(summary.get("fresh_settled_total") or 0)
            out["present"] = True
        if CORE10_CONFIDENCE_BAND_WARMSTART_LATEST.exists():
            band = json.loads(CORE10_CONFIDENCE_BAND_WARMSTART_LATEST.read_text(encoding="utf-8"))
            applied = band.get("applied") if isinstance(band.get("applied"), list) else []
            out["confidence_band_applied"] = int(len(applied))
            out["present"] = True
            if out["confidence_band_applied"] <= 0 and CORE10_TRADE_TUNING_TARGETS_LATEST.exists():
                try:
                    targets = json.loads(CORE10_TRADE_TUNING_TARGETS_LATEST.read_text(encoding="utf-8"))
                    target_rows = targets.get("target_cells") if isinstance(targets.get("target_cells"), list) else []
                    cfg_default = json.loads(TRADER_CONFIG_FILE.read_text(encoding="utf-8")) if TRADER_CONFIG_FILE.exists() else []
                    cfg_70 = json.loads(TRADER_CONFIG_FILE_70.read_text(encoding="utf-8")) if TRADER_CONFIG_FILE_70.exists() else []
                    cfg_maps = {
                        "default": {str(row.get("name") or ""): row for row in cfg_default if isinstance(row, dict)},
                        "70": {str(row.get("name") or ""): row for row in cfg_70 if isinstance(row, dict)},
                    }
                    active_count = 0
                    for row in target_rows:
                        if not isinstance(row, dict):
                            continue
                        profile = "70" if str(row.get("profile") or "") == "70" else "default"
                        trader = str(row.get("trader") or "")
                        symbol = str(row.get("symbol") or "").upper().strip()
                        cfg = cfg_maps.get(profile, {}).get(trader)
                        if not isinstance(cfg, dict):
                            continue
                        band_map = cfg.get("confidenceScaleBandsBySymbolDirection")
                        if not isinstance(band_map, dict):
                            continue
                        if any(
                            isinstance(bands, list) and bands
                            for key, bands in band_map.items()
                            if str(key).upper().startswith(f"{symbol}_")
                        ):
                            active_count += 1
                    out["confidence_band_applied"] = int(active_count)
                except Exception:
                    pass
        if COREMAIN_EXTREME_STATE_LATEST.exists():
            ext_state = json.loads(COREMAIN_EXTREME_STATE_LATEST.read_text(encoding="utf-8"))
            ext_summary = ext_state.get("summary") if isinstance(ext_state.get("summary"), dict) else {}
            up_symbols = int(ext_summary.get("extreme_up_symbols") or 0)
            down_symbols = int(ext_summary.get("extreme_down_symbols") or 0)
            out["extreme_active_symbols"] = up_symbols + down_symbols
            out["present"] = True
        if COREMAIN_EXTREME_OVERLAY_APPLY_LATEST.exists():
            ext_apply = json.loads(COREMAIN_EXTREME_OVERLAY_APPLY_LATEST.read_text(encoding="utf-8"))
            out["countertrend_hard_block_cells"] = int(((ext_apply.get("summary") or {}).get("countertrend_hard_block_cells") or 0))
            out["present"] = True
        if CONFIDENCE_BAND_OVERLAP_AUDIT_LATEST.exists():
            band_audit = json.loads(CONFIDENCE_BAND_OVERLAP_AUDIT_LATEST.read_text(encoding="utf-8"))
            out["confidence_band_overlap_count_core"] = int(((band_audit.get("summary") or {}).get("overlap_count") or 0))
            out["present"] = True
        if CHALLENGER_INCREMENTAL_PROFILE.exists():
            challenger = json.loads(CHALLENGER_INCREMENTAL_PROFILE.read_text(encoding="utf-8"))
            out["challenger_bootstrap_success"] = int(((challenger.get("summary") or {}).get("bootstrap_success") or 0))
            out["present"] = True
        if ALLORA_FEATURE_HEALTH_LATEST.exists():
            allora = json.loads(ALLORA_FEATURE_HEALTH_LATEST.read_text(encoding="utf-8"))
            out["allora_feature_freshness_sec"] = ((allora.get("summary") or {}).get("allora_feature_freshness_sec"))
            out["allora_auth_ok"] = bool(((allora.get("summary") or {}).get("auth_enabled"))) and int(((allora.get("summary") or {}).get("topics_ok") or 0)) > 0
            out["present"] = True
        if CORE10_CODE_AUDIT_LATEST.exists():
            audit = json.loads(CORE10_CODE_AUDIT_LATEST.read_text(encoding="utf-8"))
            audit_summary = audit.get("summary") if isinstance(audit.get("summary"), dict) else {}
            out["legacy_optimizer_running"] = bool(audit_summary.get("legacy_optimizer_running"))
            out["core_fast_lane_total"] = int(audit_summary.get("core_fast_lane_total") or out.get("core_fast_lane_total") or 0)
            out["core_fast_lane_covered"] = int(audit_summary.get("core_fast_lane_covered") or out.get("core_fast_lane_covered") or 0)
            out["core_fast_lane_deferred"] = int(audit_summary.get("core_fast_lane_deferred") or out.get("core_fast_lane_deferred") or 0)
            out["allora_auth_ok"] = bool(audit_summary.get("allora_auth_ok") or out.get("allora_auth_ok"))
            out["runtime_sidecar_marker_failures"] = int(audit_summary.get("runtime_sidecar_marker_failures") or 0)
            out["present"] = True
        if CORE10_FOCUS_LATEST.exists():
            focus = json.loads(CORE10_FOCUS_LATEST.read_text(encoding="utf-8"))
            model_status = ((focus.get("summary") or {}).get("model_branch_status") or {}) if isinstance((focus.get("summary") or {}).get("model_branch_status"), dict) else {}
            out["selective_abstain_targets"] = int(model_status.get("selective_abstain_targets") or 0)
            out["selective_abstain_applied"] = int(model_status.get("selective_abstain_applied") or 0)
            out["uncertainty_gate_targets"] = int(model_status.get("uncertainty_gate_targets") or 0)
            out["uncertainty_gate_applied"] = int(model_status.get("uncertainty_gate_applied") or 0)
            out["uncertainty_interval_targets"] = int(model_status.get("uncertainty_interval_targets") or 0)
            out["uncertainty_interval_blocked_manual_actions"] = int(model_status.get("uncertainty_interval_blocked_manual_actions") or 0)
            out["guard_stack_hard_blocked_directions"] = int(model_status.get("guard_stack_hard_blocked_directions") or 0)
            out["guard_stack_overcompressed_follow_directions"] = int(model_status.get("guard_stack_overcompressed_follow_directions") or 0)
            out["guard_stack_logic_conflict_directions"] = int(model_status.get("guard_stack_logic_conflict_directions") or 0)
            out["guard_stack_dead_core_cells"] = int(model_status.get("guard_stack_dead_core_cells") or 0)
            out["writeback_guard_status"] = str(model_status.get("incremental_writeback_guard_status") or out.get("writeback_guard_status") or "")
            out["writeback_guard_units_total"] = int(_first_present(model_status.get("writeback_guard_units_total"), out.get("writeback_guard_units_total"), 0))
            out["writeback_guard_allowed_units"] = int(_first_present(model_status.get("writeback_guard_allowed_units"), out.get("writeback_guard_allowed_units"), 0))
            out["writeback_guard_trained_not_written_units"] = int(_first_present(model_status.get("writeback_guard_trained_not_written_units"), out.get("writeback_guard_trained_not_written_units"), 0))
            out["writeback_guard_stale_warning_units"] = int(_first_present(model_status.get("writeback_guard_stale_warning_units"), out.get("writeback_guard_stale_warning_units"), 0))
            out["writeback_guard_stale_critical_units"] = int(_first_present(model_status.get("writeback_guard_stale_critical_units"), out.get("writeback_guard_stale_critical_units"), 0))
            out["writeback_guard_baseline_insufficient_units"] = int(_first_present(model_status.get("writeback_guard_baseline_insufficient_units"), out.get("writeback_guard_baseline_insufficient_units"), 0))
            out["writeback_guard_observed_but_baseline_insufficient_units"] = int(_first_present(model_status.get("writeback_guard_observed_but_baseline_insufficient_units"), out.get("writeback_guard_observed_but_baseline_insufficient_units"), 0))
            out["writeback_guard_real_money_weakened_units"] = int(_first_present(model_status.get("writeback_guard_real_money_weakened_units"), out.get("writeback_guard_real_money_weakened_units"), 0))
            out["writeback_guard_effective_weakened_units"] = int(_first_present(model_status.get("writeback_guard_effective_weakened_units"), out.get("writeback_guard_effective_weakened_units"), out.get("writeback_guard_real_money_weakened_units"), 0))
            out["writeback_guard_effective_weakened_with_baseline_insufficient_units"] = int(_first_present(model_status.get("writeback_guard_effective_weakened_with_baseline_insufficient_units"), out.get("writeback_guard_effective_weakened_with_baseline_insufficient_units"), 0))
            out["writeback_guard_challenger_trigger_units"] = int(_first_present(model_status.get("writeback_guard_challenger_trigger_units"), out.get("writeback_guard_challenger_trigger_units"), 0))
            out["writeback_guard_effective_challenger_trigger_units"] = int(_first_present(model_status.get("writeback_guard_effective_challenger_trigger_units"), out.get("writeback_guard_effective_challenger_trigger_units"), out.get("writeback_guard_challenger_trigger_units"), 0))
            out["writeback_guard_effective_challenger_trigger_with_baseline_insufficient_units"] = int(_first_present(model_status.get("writeback_guard_effective_challenger_trigger_with_baseline_insufficient_units"), out.get("writeback_guard_effective_challenger_trigger_with_baseline_insufficient_units"), 0))
            out["writeback_guard_oldest_successful_writeback_age_hours"] = model_status.get("writeback_guard_oldest_successful_writeback_age_hours") if model_status.get("writeback_guard_oldest_successful_writeback_age_hours") is not None else out.get("writeback_guard_oldest_successful_writeback_age_hours")
            out["incremental_param_search_status"] = str(model_status.get("incremental_param_search_status") or out.get("incremental_param_search_status") or "")
            out["incremental_param_search_families"] = int(model_status.get("incremental_param_search_families") or out.get("incremental_param_search_families") or 0)
            out["incremental_param_multi_candidate_ready_families"] = int(
                model_status.get("incremental_param_multi_candidate_ready_families")
                or out.get("incremental_param_multi_candidate_ready_families")
                or 0
            )
            out["incremental_param_apply_status"] = str(model_status.get("incremental_param_apply_status") or out.get("incremental_param_apply_status") or "")
            out["incremental_param_apply_families"] = int(model_status.get("incremental_param_apply_families") or out.get("incremental_param_apply_families") or 0)
            out["incremental_param_ready_to_apply_families"] = int(
                model_status.get("incremental_param_ready_to_apply_families")
                or out.get("incremental_param_ready_to_apply_families")
                or 0
            )
            out["incremental_param_applied_families"] = int(model_status.get("incremental_param_applied_families") or out.get("incremental_param_applied_families") or 0)
            out["simulation_money_reconciliation_status"] = str(model_status.get("simulation_money_reconciliation_status") or out.get("simulation_money_reconciliation_status") or "")
            out["simulation_money_reconciliation_families"] = int(model_status.get("simulation_money_reconciliation_families") or out.get("simulation_money_reconciliation_families") or 0)
            out["simulation_money_reconciliation_warning_families"] = int(model_status.get("simulation_money_reconciliation_warning_families") or out.get("simulation_money_reconciliation_warning_families") or 0)
            out["guard_effectiveness_status"] = str(model_status.get("guard_effectiveness_status") or out.get("guard_effectiveness_status") or "")
            out["guard_effectiveness_groups_total"] = int(model_status.get("guard_effectiveness_groups_total") or out.get("guard_effectiveness_groups_total") or 0)
            out["guard_effectiveness_groups_fresh"] = int(model_status.get("guard_effectiveness_groups_fresh") or out.get("guard_effectiveness_groups_fresh") or 0)
            out["guard_effectiveness_groups_stale"] = int(model_status.get("guard_effectiveness_groups_stale") or out.get("guard_effectiveness_groups_stale") or 0)
            out["guard_effectiveness_data_quality_ok_groups"] = int(model_status.get("guard_effectiveness_data_quality_ok_groups") or out.get("guard_effectiveness_data_quality_ok_groups") or 0)
            out["prediction_source_audit_status"] = str(model_status.get("prediction_source_audit_status") or out.get("prediction_source_audit_status") or "")
            out["prediction_source_target_traders"] = int(model_status.get("prediction_source_target_traders") or out.get("prediction_source_target_traders") or 0)
            out["prediction_source_core10_family_writers"] = int(model_status.get("prediction_source_core10_family_writers") or out.get("prediction_source_core10_family_writers") or 0)
            out["prediction_source_legacy_writers"] = int(model_status.get("prediction_source_legacy_writers") or out.get("prediction_source_legacy_writers") or 0)
            out["prediction_source_main_v5_use_core10_family"] = bool(model_status.get("prediction_source_main_v5_use_core10_family") or out.get("prediction_source_main_v5_use_core10_family") or False)
            out["prediction_source_main_v5_use_legacy_dirs"] = bool(model_status.get("prediction_source_main_v5_use_legacy_dirs") or out.get("prediction_source_main_v5_use_legacy_dirs") or False)
            out["prediction_source_conclusion"] = str(model_status.get("prediction_source_conclusion") or out.get("prediction_source_conclusion") or "")
            out["leakage_audit_status"] = str(model_status.get("leakage_audit_status") or out.get("leakage_audit_status") or "")
            out["leakage_audit_windows_pass"] = int(model_status.get("leakage_audit_windows_pass") or out.get("leakage_audit_windows_pass") or 0)
            out["leakage_audit_windows_total"] = int(model_status.get("leakage_audit_windows_total") or out.get("leakage_audit_windows_total") or 0)
            out["leakage_audit_future_like_column_count"] = int(model_status.get("leakage_audit_future_like_column_count") or out.get("leakage_audit_future_like_column_count") or 0)
            out["challenger_status"] = str(model_status.get("challenger_status") or out.get("challenger_status") or "")
            out["challenger_route"] = str(model_status.get("challenger_route") or out.get("challenger_route") or "")
            out["challenger_target_cell"] = str(model_status.get("challenger_target_cell") or out.get("challenger_target_cell") or "")
            out["follow_lane_repair_targets"] = int(model_status.get("follow_lane_repair_targets") or 0)
            out["follow_lane_repair_applied"] = int(model_status.get("follow_lane_repair_applied") or 0)
            out["follow_lane_repair_drift_count"] = int(model_status.get("follow_lane_repair_drift_count") or 0)
            out["regime_one_sided_symbols"] = int(model_status.get("regime_one_sided_symbols") or 0)
            out["regime_neutral_symbols"] = int(model_status.get("regime_neutral_symbols") or 0)
            out["regime_flip_risk_symbols"] = int(model_status.get("regime_flip_risk_symbols") or 0)
            out["feature_trade_objective_weighted_variants"] = int(model_status.get("feature_trade_objective_weighted_variants") or 0)
            out["live_coexistence_status"] = str(model_status.get("live_coexistence_status") or "")
            out["live_ready_now"] = int(model_status.get("live_ready_now") or 0)
            out["live_needs_fix"] = int(model_status.get("live_needs_fix") or 0)
            out["live_blocked"] = int(model_status.get("live_blocked") or 0)
            out["live_slot_conflicts"] = int(model_status.get("live_slot_conflicts") or 0)
            out["live_unique_slot_ready"] = bool(model_status.get("live_unique_slot_ready"))
            out["backlog_open_items"] = int(model_status.get("backlog_open_items") or 0)
            out["backlog_partial_items"] = int(model_status.get("backlog_partial_items") or 0)
            out["backlog_reconciliation_status"] = str(model_status.get("backlog_reconciliation_status") or "")
            out["runtime_param_search_status"] = str(model_status.get("runtime_param_search_status") or "")
            out["runtime_param_search_groups"] = int(model_status.get("runtime_param_search_groups") or 0)
            out["runtime_param_search_recommended_groups"] = int(model_status.get("runtime_param_search_recommended_groups") or 0)
            out["runtime_param_search_applied_groups"] = int(model_status.get("runtime_param_search_applied_groups") or 0)
            out["exp13_up_lane_search_status"] = str(model_status.get("exp13_up_lane_search_status") or "")
            out["exp13_up_lane_search_candidates"] = int(model_status.get("exp13_up_lane_search_candidates") or 0)
            out["exp13_up_lane_search_applied"] = int(model_status.get("exp13_up_lane_search_applied") or 0)
            out["exp15_up_lane_search_status"] = str(model_status.get("exp15_up_lane_search_status") or "")
            out["exp15_up_lane_search_candidates"] = int(model_status.get("exp15_up_lane_search_candidates") or 0)
            out["exp15_up_lane_search_applied"] = int(model_status.get("exp15_up_lane_search_applied") or 0)
            out["directional_drag_bad_lanes"] = int(model_status.get("directional_drag_bad_lanes") or 0)
            out["default_eth_down_tuning_status"] = str(model_status.get("default_eth_down_tuning_status") or "")
            out["default_eth_down_tuning_candidates"] = int(model_status.get("default_eth_down_tuning_candidates") or 0)
            out["default_eth_down_tuning_recommended"] = int(model_status.get("default_eth_down_tuning_recommended") or 0)
            out["default_eth_down_tuning_applied"] = int(model_status.get("default_eth_down_tuning_applied") or 0)
            out["profile70_eth_tuning_status"] = str(model_status.get("profile70_eth_tuning_status") or "")
            out["profile70_eth_tuning_candidates"] = int(model_status.get("profile70_eth_tuning_candidates") or 0)
            out["profile70_eth_tuning_recommended"] = int(model_status.get("profile70_eth_tuning_recommended") or 0)
            out["profile70_eth_tuning_applied"] = int(model_status.get("profile70_eth_tuning_applied") or 0)
            out["regime_direction_structural_weakness"] = int(model_status.get("regime_direction_structural_weakness") or 0)
            out["regime_direction_regime_conditioned"] = int(model_status.get("regime_direction_regime_conditioned") or 0)
            out["exp10_direction_tuning_status"] = str(model_status.get("exp10_direction_tuning_status") or "")
            out["exp10_direction_tuning_candidates"] = int(model_status.get("exp10_direction_tuning_candidates") or 0)
            out["exp10_direction_tuning_recommended"] = int(model_status.get("exp10_direction_tuning_recommended") or 0)
            out["exp10_direction_tuning_applied"] = int(model_status.get("exp10_direction_tuning_applied") or 0)
            out["protected_band_caps_status"] = str(model_status.get("protected_band_caps_status") or "")
            out["protected_band_caps_applied"] = int(model_status.get("protected_band_caps_applied") or 0)
            out["protected_band_drift_count"] = int(model_status.get("protected_band_drift_count") or 0)
            out["recent_sample_review_status"] = str(model_status.get("recent_sample_review_status") or "")
            out["recent_sample_last_3h_pnl"] = float(model_status.get("recent_sample_last_3h_pnl") or 0.0)
            out["recent_sample_prev_3h_pnl"] = float(model_status.get("recent_sample_prev_3h_pnl") or 0.0)
            out["recent_sample_last_6h_pnl"] = float(model_status.get("recent_sample_last_6h_pnl") or 0.0)
            out["recent_sample_prev_6h_pnl"] = float(model_status.get("recent_sample_prev_6h_pnl") or 0.0)
            out["recent_sample_last_6h_wr"] = model_status.get("recent_sample_last_6h_wr")
            out["recent_sample_prev_6h_wr"] = model_status.get("recent_sample_prev_6h_wr")
            out["final_logic_audit_status"] = str(model_status.get("final_logic_audit_status") or "")
            out["final_logic_audit_blockers"] = list(model_status.get("final_logic_audit_blockers") or []) if isinstance(model_status.get("final_logic_audit_blockers"), list) else []
            out["core10_retrain_pipeline_status"] = str(model_status.get("core10_retrain_pipeline_status") or "")
            out["tree_candidates_trained"] = int(model_status.get("tree_candidates_trained") or 0)
            out["sequence_candidates_trained"] = int(model_status.get("sequence_candidates_trained") or 0)
            out["cells_promotion_ready"] = int(model_status.get("cells_promotion_ready") or 0)
            out["core10_phase2_status"] = str(model_status.get("core10_phase2_status") or "")
            out["exp13_phase2_ready"] = bool(model_status.get("exp13_phase2_ready"))
            out["phase2_candidates_promotion_ready"] = int(model_status.get("phase2_candidates_promotion_ready") or 0)
            out["no_trade_collapse_cells"] = int(model_status.get("no_trade_collapse_cells") or 0)
            out["stage3_specialist_status"] = str(model_status.get("stage3_specialist_status") or "")
            out["stage3_specialist_promotion_ready"] = int(model_status.get("stage3_specialist_promotion_ready") or 0)
            out["recent_strict_probe_status"] = str(model_status.get("recent_strict_probe_status") or "")
            out["recent_strict_probe_exp13_ready"] = bool(model_status.get("recent_strict_probe_exp13_ready"))
            out["recent_strict_probe_expansion_allowed"] = bool(model_status.get("recent_strict_probe_expansion_allowed"))
            out["stage4_action_probe_status"] = str(model_status.get("stage4_action_probe_status") or "")
            out["stage4_action_probe_exp13_ready"] = bool(model_status.get("stage4_action_probe_exp13_ready"))
            out["stage4_action_probe_expansion_allowed"] = bool(model_status.get("stage4_action_probe_expansion_allowed"))
            out["stage5_market_action_search_status"] = str(model_status.get("stage5_market_action_search_status") or "")
            out["stage5_market_action_probe_status"] = str(model_status.get("stage5_market_action_probe_status") or "")
            out["stage5_market_action_exp13_ready"] = bool(model_status.get("stage5_market_action_exp13_ready"))
            out["stage5_market_action_expansion_allowed"] = bool(model_status.get("stage5_market_action_expansion_allowed"))
            out["stage6_market_final_action_search_status"] = str(model_status.get("stage6_market_final_action_search_status") or "")
            out["stage6_market_final_action_probe_status"] = str(model_status.get("stage6_market_final_action_probe_status") or "")
            out["stage6_market_final_action_exp13_ready"] = bool(model_status.get("stage6_market_final_action_exp13_ready"))
            out["stage6_market_final_action_expansion_allowed"] = bool(model_status.get("stage6_market_final_action_expansion_allowed"))
            out["stage8_regime_cluster_action_search_status"] = str(model_status.get("stage8_regime_cluster_action_search_status") or "")
            out["stage8_regime_cluster_action_probe_status"] = str(model_status.get("stage8_regime_cluster_action_probe_status") or "")
            out["stage8_regime_cluster_action_exp13_ready"] = bool(model_status.get("stage8_regime_cluster_action_exp13_ready"))
            out["stage8_regime_cluster_action_expansion_allowed"] = bool(model_status.get("stage8_regime_cluster_action_expansion_allowed"))
            out["stage9_precision_action_search_status"] = str(model_status.get("stage9_precision_action_search_status") or "")
            out["stage9_precision_action_probe_status"] = str(model_status.get("stage9_precision_action_probe_status") or "")
            out["stage9_precision_action_exp13_ready"] = bool(model_status.get("stage9_precision_action_exp13_ready"))
            out["stage9_precision_action_expansion_allowed"] = bool(model_status.get("stage9_precision_action_expansion_allowed"))
            out["stage10_state_expert_search_status"] = str(model_status.get("stage10_state_expert_search_status") or "")
            out["stage10_state_expert_probe_status"] = str(model_status.get("stage10_state_expert_probe_status") or "")
            out["stage10_state_expert_exp13_ready"] = bool(model_status.get("stage10_state_expert_exp13_ready"))
            out["stage10_state_expert_expansion_allowed"] = bool(model_status.get("stage10_state_expert_expansion_allowed"))
            out["stage11_state_expert_policy_status"] = str(model_status.get("stage11_state_expert_policy_status") or "")
            out["stage11_state_expert_policy_exp13_ready"] = bool(model_status.get("stage11_state_expert_policy_exp13_ready"))
            out["stage11_state_expert_policy_expansion_allowed"] = bool(model_status.get("stage11_state_expert_policy_expansion_allowed"))
            out["stage12_lane_split_status"] = str(model_status.get("stage12_lane_split_status") or "")
            out["stage12_lane_split_exp13_ready"] = bool(model_status.get("stage12_lane_split_exp13_ready"))
            out["stage12_lane_split_rally_up_ready"] = bool(model_status.get("stage12_lane_split_rally_up_ready"))
            out["stage12_lane_split_rally_down_ready"] = bool(model_status.get("stage12_lane_split_rally_down_ready"))
            out["stage13_rally_up_precision_status"] = str(model_status.get("stage13_rally_up_precision_status") or "")
            out["stage13_rally_up_precision_exp13_ready"] = bool(model_status.get("stage13_rally_up_precision_exp13_ready"))
            out["stage13_rally_up_precision_promotion_ready"] = int(model_status.get("stage13_rally_up_precision_promotion_ready") or 0)
            out["stage14_rally_up_teacher_residual_status"] = str(model_status.get("stage14_rally_up_teacher_residual_status") or "")
            out["stage14_rally_up_teacher_residual_exp13_ready"] = bool(model_status.get("stage14_rally_up_teacher_residual_exp13_ready"))
            out["stage14_rally_up_teacher_residual_promotion_ready"] = int(model_status.get("stage14_rally_up_teacher_residual_promotion_ready") or 0)
            out["stage15_event_ranker_status"] = str(model_status.get("stage15_event_ranker_status") or "")
            out["stage15_event_ranker_exp13_ready"] = bool(model_status.get("stage15_event_ranker_exp13_ready"))
            out["stage15_event_ranker_promotion_ready"] = int(model_status.get("stage15_event_ranker_promotion_ready") or 0)
            out["stage16_event_precision_status"] = str(model_status.get("stage16_event_precision_status") or "")
            out["stage16_event_precision_exp13_ready"] = bool(model_status.get("stage16_event_precision_exp13_ready"))
            out["stage16_event_precision_promotion_ready"] = int(model_status.get("stage16_event_precision_promotion_ready") or 0)
            out["stage16_shadow_gate_status"] = str(model_status.get("stage16_shadow_gate_status") or "")
            out["stage16_shadow_gate_ready"] = int(model_status.get("stage16_shadow_gate_ready") or 0)
            out["stage16_shadow_gate_best_ready"] = bool(model_status.get("stage16_shadow_gate_best_ready"))
            out["stage16_shadow_compare_status"] = str(model_status.get("stage16_shadow_compare_status") or "")
            out["stage16_shadow_compare_ready"] = int(model_status.get("stage16_shadow_compare_ready") or 0)
            out["stage16_shadow_compare_shadow_simulation_ready"] = int(model_status.get("stage16_shadow_compare_shadow_simulation_ready") or 0)
            out["stage16_shadow_compare_best_ready"] = bool(model_status.get("stage16_shadow_compare_best_ready"))
            out["stage16_shadow_compare_best_shadow_simulation_ready"] = bool(model_status.get("stage16_shadow_compare_best_shadow_simulation_ready"))
            out["stage16_shadow_sim_candidate_status"] = str(model_status.get("stage16_shadow_sim_candidate_status") or "")
            out["stage16_shadow_sim_candidate_ready"] = bool(model_status.get("stage16_shadow_sim_candidate_ready"))
            out["stage16_shadow_sim_candidate_count"] = int(model_status.get("stage16_shadow_sim_candidate_count") or 0)
            out["stage16_shadow_live_watch_status"] = str(model_status.get("stage16_shadow_live_watch_status") or "")
            out["stage16_shadow_live_watch_last_24h_pnl"] = float(model_status.get("stage16_shadow_live_watch_last_24h_pnl") or 0.0)
            out["stage16_shadow_live_watch_last_72h_pnl"] = float(model_status.get("stage16_shadow_live_watch_last_72h_pnl") or 0.0)
            out["stage16_shadow_live_watch_latest_would_trade"] = bool(model_status.get("stage16_shadow_live_watch_latest_would_trade"))
            out["stage16_shadow_live_watch_caution"] = str(model_status.get("stage16_shadow_live_watch_caution") or "")
            out["stage16_shadow_live_compare_status"] = str(model_status.get("stage16_shadow_live_compare_status") or "")
            out["stage16_shadow_live_compare_last_24h_vs_active_pnl_delta"] = float(model_status.get("stage16_shadow_live_compare_last_24h_vs_active_pnl_delta") or 0.0)
            out["stage16_shadow_live_compare_last_72h_vs_active_pnl_delta"] = float(model_status.get("stage16_shadow_live_compare_last_72h_vs_active_pnl_delta") or 0.0)
            out["stage16_shadow_live_compare_tail_96_vs_active_pnl_delta"] = float(model_status.get("stage16_shadow_live_compare_tail_96_vs_active_pnl_delta") or 0.0)
            out["stage16_shadow_stability_status"] = str(model_status.get("stage16_shadow_stability_status") or "")
            out["stage16_shadow_stability_label"] = str(model_status.get("stage16_shadow_stability_label") or "")
            out["stage16_shadow_stable_candidate"] = bool(model_status.get("stage16_shadow_stable_candidate"))
            out["stage16_shadow_cautious_candidate"] = bool(model_status.get("stage16_shadow_cautious_candidate"))
            out["exp13_full_stack_replay_status"] = str(model_status.get("exp13_full_stack_replay_status") or "")
            out["exp13_full_stack_replay_candidate_ready"] = bool(model_status.get("exp13_full_stack_replay_candidate_ready"))
            out["exp13_full_stack_replay_verdict"] = str(model_status.get("exp13_full_stack_replay_verdict") or "")
            out["exp13_full_stack_replay_candidate_recent14_pnl"] = float(model_status.get("exp13_full_stack_replay_candidate_recent14_pnl") or 0.0)
            out["exp13_full_stack_replay_active_recent14_pnl"] = float(model_status.get("exp13_full_stack_replay_active_recent14_pnl") or 0.0)
            out["exp13_full_stack_replay_candidate_recent14_score"] = float(model_status.get("exp13_full_stack_replay_candidate_recent14_score") or 0.0)
            out["exp13_full_stack_replay_active_recent14_score"] = float(model_status.get("exp13_full_stack_replay_active_recent14_score") or 0.0)
            out["exp13_full_stack_replay_candidate_strict_score"] = float(model_status.get("exp13_full_stack_replay_candidate_strict_score") or 0.0)
            out["exp13_full_stack_replay_active_strict_score"] = float(model_status.get("exp13_full_stack_replay_active_strict_score") or 0.0)
            out["exp13_official_sim_candidate_status"] = str(model_status.get("exp13_official_sim_candidate_status") or "")
            out["exp13_official_sim_candidate_qualified"] = bool(model_status.get("exp13_official_sim_candidate_qualified"))
            out["exp13_official_sim_candidate_live"] = bool(model_status.get("exp13_official_sim_candidate_live"))
            out["exp13_official_sim_candidate_prediction_file_exists"] = bool(model_status.get("exp13_official_sim_candidate_prediction_file_exists"))
            out["exp13_official_sim_candidate_prediction_file_age_sec"] = (
                None
                if model_status.get("exp13_official_sim_candidate_prediction_file_age_sec") is None
                else float(model_status.get("exp13_official_sim_candidate_prediction_file_age_sec") or 0.0)
            )
            out["exp13_official_sim_candidate_recent14_pnl"] = float(model_status.get("exp13_official_sim_candidate_recent14_pnl") or 0.0)
            out["exp13_official_sim_candidate_strict_score"] = float(model_status.get("exp13_official_sim_candidate_strict_score") or 0.0)
            out["exp13_official_sim_candidate_verdict"] = str(model_status.get("exp13_official_sim_candidate_verdict") or "")
            out["route_tournament_status"] = str(model_status.get("route_tournament_status") or "")
            out["route_tournament_current_leader"] = str(model_status.get("route_tournament_current_leader") or "")
            out["route_tournament_final_verdict"] = str(model_status.get("route_tournament_final_verdict") or "")
            out["route_rollout_status"] = str(model_status.get("route_rollout_status") or "")
            out["route_rollout_winning_route"] = str(model_status.get("route_rollout_winning_route") or "")
            out["route_e_status"] = str(model_status.get("route_e_status") or "")
            out["route_e_target_cell"] = str(model_status.get("route_e_target_cell") or "")
            out["route_e_pilot_ready"] = bool(model_status.get("route_e_pilot_ready"))
            out["route_e_pilot_status"] = str(model_status.get("route_e_pilot_status") or "")
            out["route_e_pilot_guard_depth_best"] = int(model_status.get("route_e_pilot_guard_depth_best") or 0)
            out["route_e_pilot_guard_name_best"] = str(model_status.get("route_e_pilot_guard_name_best") or "")
            out["route_e_pilot_recent14_pnl"] = float(model_status.get("route_e_pilot_recent14_pnl") or 0.0)
            out["route_e_pilot_strict_pnl"] = float(model_status.get("route_e_pilot_strict_pnl") or 0.0)
            out["route_e_pilot_full_pnl"] = float(model_status.get("route_e_pilot_full_pnl") or 0.0)
            out["route_e_pilot_recent14_score"] = float(model_status.get("route_e_pilot_recent14_score") or 0.0)
            out["route_e_pilot_strict_score"] = float(model_status.get("route_e_pilot_strict_score") or 0.0)
            out["route_e_pilot_full_score"] = float(model_status.get("route_e_pilot_full_score") or 0.0)
            out["route_e_pilot_drawdown_proxy"] = float(model_status.get("route_e_pilot_drawdown_proxy") or 0.0)
            out["route_e_pilot_coverage_pass"] = bool(model_status.get("route_e_pilot_coverage_pass"))
            out["route_e_pilot_low_support_benchmark"] = bool(model_status.get("route_e_pilot_low_support_benchmark"))
            out["route_e_pilot_benchmark_recent14_pnl"] = float(model_status.get("route_e_pilot_benchmark_recent14_pnl") or 0.0)
            out["route_e_pilot_benchmark_strict_pnl"] = float(model_status.get("route_e_pilot_benchmark_strict_pnl") or 0.0)
            out["route_e_pilot_benchmark_full_drawdown_proxy"] = float(model_status.get("route_e_pilot_benchmark_full_drawdown_proxy") or 0.0)
            out["route_e_official_sim_status"] = str(model_status.get("route_e_official_sim_status") or "")
            out["route_e_official_sim_qualified"] = bool(model_status.get("route_e_official_sim_qualified"))
            out["route_e_official_sim_live"] = bool(model_status.get("route_e_official_sim_live"))
            out["route_e_official_sim_prediction_file_exists"] = bool(model_status.get("route_e_official_sim_prediction_file_exists"))
            out["route_e_official_sim_prediction_file_age_sec"] = (
                None
                if model_status.get("route_e_official_sim_prediction_file_age_sec") is None
                else float(model_status.get("route_e_official_sim_prediction_file_age_sec") or 0.0)
            )
            out["route_e_official_sim_recent14_pnl"] = float(model_status.get("route_e_official_sim_recent14_pnl") or 0.0)
            out["route_e_official_sim_strict_pnl"] = float(model_status.get("route_e_official_sim_strict_pnl") or 0.0)
            out["route_e_official_sim_full_pnl"] = float(model_status.get("route_e_official_sim_full_pnl") or 0.0)
            out["route_e_official_sim_guard_depth_best"] = int(model_status.get("route_e_official_sim_guard_depth_best") or 0)
            out["route_e_official_sim_verdict"] = str(model_status.get("route_e_official_sim_verdict") or "")
            out["route_e_core10_sweep_status"] = str(model_status.get("route_e_core10_sweep_status") or "")
            out["route_e_core10_sweep_ready_cells"] = int(model_status.get("route_e_core10_sweep_ready_cells") or 0)
            out["route_e_core10_sweep_total_cells"] = int(model_status.get("route_e_core10_sweep_total_cells") or 0)
            out["route_e_core10_sweep_verdict"] = str(model_status.get("route_e_core10_sweep_verdict") or "")
            out["route_e_rollout_status"] = str(model_status.get("route_e_rollout_status") or "")
            out["route_e_rollout_live"] = bool(model_status.get("route_e_rollout_live"))
            out["route_e_rollout_verdict"] = str(model_status.get("route_e_rollout_verdict") or "")
            out["route_f_status"] = str(model_status.get("route_f_status") or "")
            out["route_f_target_cell"] = str(model_status.get("route_f_target_cell") or "")
            out["route_f_pilot_ready"] = bool(model_status.get("route_f_pilot_ready"))
            out["route_f_pilot_status"] = str(model_status.get("route_f_pilot_status") or "")
            out["route_f_pilot_guard_depth_best"] = int(model_status.get("route_f_pilot_guard_depth_best") or 0)
            out["route_f_pilot_guard_name_best"] = str(model_status.get("route_f_pilot_guard_name_best") or "")
            out["route_f_pilot_recent14_capital_change"] = float(model_status.get("route_f_pilot_recent14_capital_change") or 0.0)
            out["route_f_pilot_strict_capital_change"] = float(model_status.get("route_f_pilot_strict_capital_change") or 0.0)
            out["route_f_pilot_full_capital_change"] = float(model_status.get("route_f_pilot_full_capital_change") or 0.0)
            out["route_f_pilot_recent14_win_rate"] = float(model_status.get("route_f_pilot_recent14_win_rate") or 0.0)
            out["route_f_pilot_strict_win_rate"] = float(model_status.get("route_f_pilot_strict_win_rate") or 0.0)
            out["route_f_pilot_full_win_rate"] = float(model_status.get("route_f_pilot_full_win_rate") or 0.0)
            out["route_f_pilot_full_drawdown_capital"] = float(model_status.get("route_f_pilot_full_drawdown_capital") or 0.0)
            out["route_f_pilot_coverage_pass"] = bool(model_status.get("route_f_pilot_coverage_pass"))
            out["route_f_pilot_low_support_benchmark"] = bool(model_status.get("route_f_pilot_low_support_benchmark"))
            out["route_f_pilot_benchmark_recent14_capital_change"] = float(model_status.get("route_f_pilot_benchmark_recent14_capital_change") or 0.0)
            out["route_f_pilot_benchmark_strict_capital_change"] = float(model_status.get("route_f_pilot_benchmark_strict_capital_change") or 0.0)
            out["route_f_pilot_benchmark_full_capital_change"] = float(model_status.get("route_f_pilot_benchmark_full_capital_change") or 0.0)
            out["route_f_pilot_benchmark_recent14_win_rate"] = float(model_status.get("route_f_pilot_benchmark_recent14_win_rate") or 0.0)
            out["route_f_pilot_benchmark_strict_win_rate"] = float(model_status.get("route_f_pilot_benchmark_strict_win_rate") or 0.0)
            out["route_f_pilot_benchmark_full_win_rate"] = float(model_status.get("route_f_pilot_benchmark_full_win_rate") or 0.0)
            out["route_f_pilot_benchmark_full_drawdown_capital"] = float(model_status.get("route_f_pilot_benchmark_full_drawdown_capital") or 0.0)
            out["route_f_official_sim_status"] = str(model_status.get("route_f_official_sim_status") or "")
            out["route_f_official_sim_qualified"] = bool(model_status.get("route_f_official_sim_qualified"))
            out["route_f_official_sim_live"] = bool(model_status.get("route_f_official_sim_live"))
            out["route_f_official_sim_prediction_file_exists"] = bool(model_status.get("route_f_official_sim_prediction_file_exists"))
            out["route_f_official_sim_prediction_file_age_sec"] = (
                None
                if model_status.get("route_f_official_sim_prediction_file_age_sec") is None
                else float(model_status.get("route_f_official_sim_prediction_file_age_sec") or 0.0)
            )
            out["route_f_official_sim_recent14_capital_change"] = float(model_status.get("route_f_official_sim_recent14_capital_change") or 0.0)
            out["route_f_official_sim_strict_capital_change"] = float(model_status.get("route_f_official_sim_strict_capital_change") or 0.0)
            out["route_f_official_sim_full_capital_change"] = float(model_status.get("route_f_official_sim_full_capital_change") or 0.0)
            out["route_f_official_sim_guard_depth_best"] = int(model_status.get("route_f_official_sim_guard_depth_best") or 0)
            out["route_f_official_sim_verdict"] = str(model_status.get("route_f_official_sim_verdict") or "")
            out["promotion_gate_fairness_status"] = str(model_status.get("promotion_gate_fairness_status") or "")
            out["promotion_gate_sample_bias_rows"] = int(model_status.get("promotion_gate_sample_bias_rows") or 0)
            out["promotion_gate_active_small_sample_rows"] = int(model_status.get("promotion_gate_active_small_sample_rows") or 0)
            out["present"] = True
        if int(out.get("extreme_active_symbols") or 0) > 0:
            if int(out.get("countertrend_hard_block_cells") or 0) > 0 and int(out.get("confidence_band_overlap_count_core") or 0) == 0:
                out["core10_extreme_guard_status"] = "active_ok"
            elif int(out.get("countertrend_hard_block_cells") or 0) > 0:
                out["core10_extreme_guard_status"] = "active_with_overlap_risk"
            else:
                out["core10_extreme_guard_status"] = "active_missing_blocks"
        else:
            out["core10_extreme_guard_status"] = "idle"
    except Exception:
        return out
    return out


def _dynamic_wallet_v5_writers() -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    for active_file, config_file, profile in (
        (ACTIVE_TRADER_FILE, TRADER_CONFIG_FILE, "default"),
        (ACTIVE_TRADER_FILE_70, TRADER_CONFIG_FILE_70, "70"),
    ):
        active_names = _load_active_trader_names(active_file)
        if not active_names or not config_file.exists():
            continue
        try:
            cfg = json.loads(config_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(cfg, list):
            continue
        suffixes: set[str] = set()
        for row in cfg:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name", "")).strip()
            if name not in active_names or "_wallet" not in name:
                continue
            suffix = str(row.get("predictionSuffix", "")).strip()
            if not suffix:
                continue
            if suffix.startswith("_"):
                suffix = suffix[1:]
            if not suffix.endswith(".json"):
                suffix = f"{suffix}.json"
            suffixes.add(suffix)
        for suffix in sorted(suffixes):
            label = "Exp10 Wallet" if profile == "default" else "Exp10 Wallet 70"
            entries.append((label, suffix))
    return entries


def get_expected_prediction_files() -> List[Tuple[str, str]]:
    if _slot_scoped_live_health_only_enabled():
        return _slot_scoped_health_prediction_files()
    files = (
        EXPECTED_V5_WRITERS
        + EXPECTED_V5_WRITERS_70
        + _dynamic_wallet_v5_writers()
        + _dynamic_active_v5_writers()
        + EXPECTED_GRU_WRITERS
    )
    # GRU 仅检查 active trader 实际引用到的预测文件，避免“已停用源”误报。
    active_pred_files = _load_active_prediction_files_from_configs()
    if not active_pred_files:
        return files
    filtered: List[Tuple[str, str]] = []
    for label, fname in files:
        if fname.startswith("predictions_v5_") and not _legacy_prediction_should_be_expected(fname):
            continue
        if fname.startswith("predictions_gru_") and fname not in active_pred_files:
            continue
        filtered.append((label, fname))
    return filtered


def _has_running_70_groups() -> bool:
    try:
        ret = subprocess.run(
            ["ps", "-eo", "args", "-ww"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return False
    return any(
        "multi_prediction_index" in l and "_70" in l and "--group" in l
        for l in (ret.stdout or "").splitlines()
    )


def _run_shell(cmd: str, timeout_sec: int = 180) -> Tuple[bool, str]:
    try:
        ret = subprocess.run(
            ["/bin/bash", "-lc", cmd],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=max(10, int(timeout_sec)),
            check=False,
        )
        tail = "\n".join((ret.stdout or "").splitlines()[-10:])
        return ret.returncode == 0, tail
    except Exception as e:
        return False, str(e)


def auto_heal_market_data_daemon() -> Tuple[bool, str]:
    cmd = (
        "if [ -f polymarket/market_data_daemon.pid ]; then "
        "kill $(cat polymarket/market_data_daemon.pid) 2>/dev/null || true; "
        "fi; "
        "rm -f polymarket/market_data_daemon.pid; "
        "./启动市场数据服务.sh"
    )
    return _run_shell(cmd, timeout_sec=150)


def auto_heal_derivatives_collector() -> Tuple[bool, str]:
    script = PROJECT_ROOT / "reload_launchctl_derivatives_collector.sh"
    if not script.exists():
        return False, f"脚本不存在: {script}"
    cmd = f"bash {shlex.quote(str(script))} restart"
    return _run_shell(cmd, timeout_sec=180)


def auto_heal_data_source_backfill() -> Tuple[bool, str]:
    """仅补历史（Funding/OI/LS），不重启采集器。VPN 断线导致数据过期时可调用。"""
    script = PROJECT_ROOT / "reload_launchctl_derivatives_collector.sh"
    if not script.exists():
        return False, f"脚本不存在: {script}"
    cmd = f"bash {shlex.quote(str(script))} backfill"
    return _run_shell(cmd, timeout_sec=120)


def auto_heal_ensemble_writer(profile: str = "default") -> Tuple[bool, str]:
    script = PROJECT_ROOT / "reload_launchctl_ensemble_writer.sh"
    if not script.exists():
        return False, f"脚本不存在: {script}"
    prof = "70" if str(profile).strip() == "70" else "default"
    cmd = f"bash {shlex.quote(str(script))} restart --profile {prof}"
    ok, out = _run_shell(cmd, timeout_sec=120)
    if ok:
        return True, out

    # 默认不回退 nohup，避免和 launchd 双轨并存导致重复进程。
    # 仅当显式允许时才回退（临时救火模式）。
    allow_fallback = os.environ.get("ALLOW_WRITER_NOHUP_FALLBACK", "0").strip() == "1"
    if not allow_fallback:
        msg = (out or "").strip()
        msg = msg + ("\n" if msg else "") + "已禁止 nohup 回退（避免与 launchd 双轨冲突）"
        return False, msg

    # launchd 失败时，按显式开关回退到 legacy nohup，避免写入器长时间缺失
    if prof == "70":
        fallback = (
            "mkdir -p polymarket/logs_ensemble; "
            "if [ -f polymarket/logs_ensemble/ensemble_writer70.pid ]; then "
            "kill $(cat polymarket/logs_ensemble/ensemble_writer70.pid) 2>/dev/null || true; fi; "
            "nohup /Users/mac/miniforge3/bin/python scripts/ensemble_prediction_writer.py --output-70 "
            "> polymarket/logs_ensemble/ensemble_writer70_stdout.log 2>&1 < /dev/null & "
            "echo $! > polymarket/logs_ensemble/ensemble_writer70.pid; "
            "sleep 2; pgrep -f 'ensemble_prediction_writer.py.*--output-70' >/dev/null"
        )
    else:
        fallback = (
            "mkdir -p polymarket/logs_ensemble; "
            "if [ -f polymarket/logs_ensemble/ensemble_writer.pid ]; then "
            "kill $(cat polymarket/logs_ensemble/ensemble_writer.pid) 2>/dev/null || true; fi; "
            "nohup /Users/mac/miniforge3/bin/python scripts/ensemble_prediction_writer.py "
            "--output polymarket/predictions_ensemble.json "
            ">> polymarket/logs_ensemble/ensemble_writer_stdout.log 2>&1 < /dev/null & "
            "echo $! > polymarket/logs_ensemble/ensemble_writer.pid; "
            "sleep 2; pgrep -f 'ensemble_prediction_writer.py.*predictions_ensemble.json' >/dev/null"
        )
    ok_fb, out_fb = _run_shell(fallback, timeout_sec=120)
    merged = (out or "").strip() + ("\n" if out and out_fb else "") + (out_fb or "").strip()
    return ok_fb, merged


def get_active_groups() -> Optional[set[str]]:
    # default=auto: 检测到 70 组在跑时，自动把 active_traders_70 也纳入过滤
    include_70_mode = os.environ.get("HEALTH_INCLUDE_70_GROUPS", "auto").strip().lower()
    include_70 = False
    if include_70_mode in ("1", "true", "yes", "on", "all"):
        include_70 = True
    elif include_70_mode in ("0", "false", "no", "off", "non70"):
        include_70 = False
    else:
        include_70 = _has_running_70_groups()

    base_groups = _load_active_groups_from_file(ACTIVE_TRADER_FILE, TRADER_CONFIG_FILE) or set()
    if not include_70:
        return base_groups or None

    groups70 = _load_active_groups_from_file(ACTIVE_TRADER_FILE_70, TRADER_CONFIG_FILE_70) or set()
    merged = set(base_groups) | set(groups70)
    return merged or None


def get_expected_merged_groups() -> Dict[str, int]:
    if _slot_scoped_live_health_only_enabled():
        slot_targets = _load_slot_scoped_live_targets()
        if not slot_targets:
            return {"lowprice_70_selected": 2}
        counts: Dict[str, int] = {}
        for row in slot_targets:
            group = str(row.get("group") or "").strip() or "lowprice_70_selected"
            counts[group] = counts.get(group, 0) + 1
        return counts
    active = get_active_groups()
    runtime_counts: Dict[str, int] = {}
    for src in (
        _load_active_group_counts_from_file(ACTIVE_TRADER_FILE, TRADER_CONFIG_FILE),
        _load_active_group_counts_from_file(ACTIVE_TRADER_FILE_70, TRADER_CONFIG_FILE_70),
    ):
        for k, v in src.items():
            runtime_counts[k] = runtime_counts.get(k, 0) + int(v)

    if active:
        filtered_runtime = {
            k: v for k, v in runtime_counts.items()
            if k in active and int(v) > 0
        }
        if filtered_runtime:
            return filtered_runtime
        filtered_legacy = {k: v for k, v in ALL_EXPECTED_MERGED_GROUPS.items() if k in active}
        return filtered_legacy if filtered_legacy else dict(EXPECTED_MERGED_GROUPS)

    return runtime_counts if runtime_counts else dict(EXPECTED_MERGED_GROUPS)


# ─── 检查函数 ─────────────────────────────────────────────

# 无环境代理时依次尝试的本地代理端口 (Clash/V2Ray 常见)
_FALLBACK_PROXY_PORTS = [7890, 7891, 8080, 7892, 1087, 1080]


def _build_opener_with_proxy(proxy: str):
    proxy_handler = urllib.request.ProxyHandler({"https": proxy, "http": proxy})
    urllib.request.install_opener(urllib.request.build_opener(proxy_handler))


def _urlopen_with_retries(
    url: str,
    timeout: int,
    attempts: Optional[int] = None,
) -> Tuple[bool, int, str, int]:
    """
    统一网络探测重试:
      返回 (ok, total_ms, last_error, used_attempts)
    """
    max_attempts = max(1, int(attempts or API_RETRY_ATTEMPTS))
    start = time.time()
    last_err = ""
    used = 0
    for i in range(max_attempts):
        used = i + 1
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            urllib.request.urlopen(req, timeout=timeout)
            ms = round((time.time() - start) * 1000)
            return True, ms, "", used
        except Exception as e:
            last_err = str(e)
            if i < max_attempts - 1:
                time.sleep(API_RETRY_BASE_DELAY_SEC * (2 ** i))
    ms = round((time.time() - start) * 1000)
    return False, ms, last_err, used


def check_system_proxy() -> Tuple[bool, str]:
    proxies = urllib.request.getproxies()
    proxy = proxies.get("https") or proxies.get("http")
    if proxy:
        try:
            _build_opener_with_proxy(proxy)
            ok, ms, err, used = _urlopen_with_retries("https://api.binance.com/api/v3/ping", timeout=10)
            if ok:
                return True, f"{proxy} (通过代理访问Binance正常, {ms}ms, {used}次)"
            return False, f"{proxy} 无法访问Binance: {err[:80]} ({ms}ms, {used}次)"
        except Exception as e:
            return False, f"{proxy} 无法访问Binance: {str(e)[:80]}"
    for port in _FALLBACK_PROXY_PORTS:
        proxy = f"http://127.0.0.1:{port}"
        try:
            _build_opener_with_proxy(proxy)
            ok, ms, err, used = _urlopen_with_retries("https://api.binance.com/api/v3/ping", timeout=5)
            if ok:
                return True, f"{proxy} (本地代理可用, {ms}ms, {used}次)"
        except Exception:
            continue
    return False, "未检测到可用代理 (已尝试常见端口)"


def check_api_endpoints() -> List[dict]:
    results = []
    for name, url in API_ENDPOINTS:
        ok, ms, err, used = _urlopen_with_retries(url, timeout=15)
        item = {"name": name, "ok": ok, "ms": ms, "attempts": used}
        if not ok:
            item["error"] = err[:120]
        results.append(item)
    return results


def _ccxt_python_candidates() -> List[str]:
    """用于探测 ccxt 的候选 Python 解释器。"""
    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    conda_python_exe = os.environ.get("CONDA_PYTHON_EXE", "").strip()
    train_python = os.environ.get("TRAIN_PYTHON", "").strip()
    candidates: List[str] = []
    if train_python:
        candidates.append(train_python)
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
            "/Users/mac/miniforge3/bin/python",
            "/opt/homebrew/Caskroom/miniforge/base/bin/python3",
            "python3",
        ]
    )
    seen = set()
    out: List[str] = []
    for py in candidates:
        if py and py not in seen:
            seen.add(py)
            out.append(py)
    return out


def _probe_ccxt_with_python(py_bin: str) -> Tuple[bool, str]:
    """在指定解释器中探测 ccxt + Binance 拉取链路。"""
    script = (
        "import sys,time\n"
        f"sys.path.insert(0, {str(PROJECT_ROOT)!r})\n"
        "from src.python.data_fetcher import get_exchange\n"
        "t=time.time()\n"
        "ex=get_exchange(); ex.load_markets(); ex.fetch_ohlcv('BTC/USDT','15m',limit=1)\n"
        "proxy=getattr(ex,'httpsProxy',None) or '无'\n"
        "print('OK|%d|%s|%d' % (len(ex.markets), proxy, int((time.time()-t)*1000)))\n"
    )
    try:
        p = subprocess.run(
            [py_bin, "-c", script],
            capture_output=True,
            text=True,
            timeout=35,
        )
    except Exception as e:
        return False, f"{py_bin}: 调用失败({str(e)[:80]})"

    stdout = (p.stdout or "").strip().splitlines()
    last = stdout[-1] if stdout else ""
    if p.returncode == 0 and last.startswith("OK|"):
        parts = last.split("|", 3)
        if len(parts) == 4:
            _, n_markets, proxy, ms = parts
            return True, f"{py_bin} ({n_markets}交易对, 代理:{proxy}, {ms}ms)"
        return True, f"{py_bin} (探测通过)"

    err = ((p.stderr or "").strip() or (last or "unknown"))[:160]
    return False, f"{py_bin}: {err}"


def check_ccxt() -> Tuple[bool, str]:
    sys.path.insert(0, str(PROJECT_ROOT))
    start = time.time()
    first_error = ""
    try:
        import src.python.data_fetcher as data_fetcher  # type: ignore

        for i in range(CCXT_RETRY_ATTEMPTS):
            try:
                # 避免单次失败后复用坏连接对象
                if hasattr(data_fetcher, "_exchange_cache"):
                    data_fetcher._exchange_cache = None
                ex = data_fetcher.get_exchange()
                ex.load_markets()
                ex.fetch_ohlcv("BTC/USDT", "15m", limit=1)
                ms = round((time.time() - start) * 1000)
                proxy = getattr(ex, "httpsProxy", None) or "无"
                return True, f"正常 ({len(ex.markets)}交易对, 代理:{proxy}, {ms}ms, {i+1}次)"
            except Exception as e:
                if not first_error:
                    first_error = str(e)[:140]
                if i < CCXT_RETRY_ATTEMPTS - 1:
                    time.sleep(0.4 * (2 ** i))
    except Exception as e:
        if not first_error:
            first_error = str(e)[:140]

    fallback_errors: List[str] = []
    for py_bin in _ccxt_python_candidates():
        if py_bin == sys.executable:
            continue
        ok, msg = _probe_ccxt_with_python(py_bin)
        if ok:
            return True, f"正常 (fallback={msg})"
        fallback_errors.append(msg)

    ms = round((time.time() - start) * 1000)
    if fallback_errors:
        return False, f"失败 ({ms}ms): {first_error or 'unknown'} | fallback: {fallback_errors[0]}"
    return False, f"失败 ({ms}ms): {first_error or 'unknown'}"


def check_prediction_files() -> List[dict]:
    def _parse_etime_to_seconds(etime_text: str) -> Optional[int]:
        """将 ps etime 文本解析为秒。支持 MM:SS / HH:MM:SS / DD-HH:MM:SS。"""
        s = (etime_text or "").strip()
        if not s:
            return None
        days = 0
        if "-" in s:
            day_part, s = s.split("-", 1)
            try:
                days = int(day_part)
            except ValueError:
                return None
        parts = s.split(":")
        try:
            nums = [int(x) for x in parts]
        except ValueError:
            return None
        if len(nums) == 2:
            h, m, sec = 0, nums[0], nums[1]
        elif len(nums) == 3:
            h, m, sec = nums[0], nums[1], nums[2]
        else:
            return None
        return days * 86400 + h * 3600 + m * 60 + sec

    def _writer_uptime_sec_by_prediction_file() -> Dict[str, int]:
        """返回 {prediction_file_name: writer_uptime_seconds}，用于判定“重启后等待下一周期”的过期豁免。"""
        out: Dict[str, int] = {}
        try:
            ps = subprocess.run(
                ["ps", "-eo", "etime,args", "-ww"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            lines = (ps.stdout or "").splitlines()
        except Exception:
            return out

        for line in lines:
            raw = line.strip()
            if not raw:
                continue
            parts = raw.split(None, 1)
            if len(parts) < 2:
                continue
            etimes = _parse_etime_to_seconds(parts[0])
            if etimes is None:
                continue
            args = parts[1]
            if "prediction_writer_v5" in args or "prediction_writer_gru" in args:
                m = re.search(r"(predictions_(?:v5|gru)_[\w]+\.json)", args)
                if m:
                    key = m.group(1)
                    prev = out.get(key)
                    if prev is None or etimes < prev:
                        out[key] = etimes
        return out

    now = time.time()
    results = []
    active_groups = get_active_groups()
    file_to_group = {**PREDICTION_FILE_TO_GROUP, **_dynamic_active_v5_file_to_group()}
    writer_uptime = _writer_uptime_sec_by_prediction_file()
    for label, fname in get_expected_prediction_files():
        if active_groups:
            if fname in file_to_group and file_to_group[fname] not in active_groups:
                continue
        fpath = POLYMARKET_DIR / fname
        writer_age_sec = writer_uptime.get(fname)
        if not fpath.exists():
            if writer_age_sec is not None and writer_age_sec <= 35 * 60:
                results.append({
                    "name": label,
                    "file": fname,
                    "ok": True,
                    "status": f"写入器刚重启(运行{writer_age_sec//60}m)，等待首次写入",
                })
            else:
                results.append({"name": label, "file": fname, "ok": False,
                                "status": "文件不存在"})
            continue
        if "†" in label:
            threshold = T120S_STALE_THRESHOLD_MIN
        elif fname.startswith("predictions_gru_"):
            threshold = STALE_THRESHOLD_MIN_GRU
        else:
            threshold = STALE_THRESHOLD_MIN_V5
        mtime = os.path.getmtime(fpath)
        age_min = (now - mtime) / 60
        update_time = datetime.fromtimestamp(mtime).strftime("%H:%M:%S")
        age_sec = now - mtime
        if age_min <= threshold:
            results.append({"name": label, "file": fname, "ok": True,
                            "status": f"更新:{update_time} ({age_min:.0f}分钟前)"})
        elif (
            writer_age_sec is not None
            and writer_age_sec < age_sec
            and writer_age_sec <= 35 * 60
        ):
            # 写入器刚重启，可能尚未到下一轮写入窗口（特别是 T+0 写入器在 15m 周期边界）
            results.append({
                "name": label,
                "file": fname,
                "ok": True,
                "status": f"写入器刚重启(运行{writer_age_sec//60}m)，等待下一周期；当前文件:{update_time} ({age_min:.0f}分钟前)",
            })
        else:
            hours = age_min / 60
            results.append({"name": label, "file": fname, "ok": False,
                            "status": f"过期! 更新:{update_time} ({hours:.1f}小时前)"})
    return results


def check_writer_processes() -> List[dict]:
    """检查每个预测写入器进程是否存活。"""
    result = subprocess.run(["ps", "-eo", "pid,lstart,args", "-ww"],
                            capture_output=True, text=True)
    ps_lines = result.stdout
    ps_rows = [l for l in ps_lines.split("\n") if l.strip()]

    checks = []
    if _slot_scoped_live_health_only_enabled():
        for row in _load_slot_scoped_live_targets():
            output_key = str(row.get("prediction_file") or "")
            count = sum(
                1
                for l in ps_rows
                if "prediction_writer_v5" in l
                and output_key in l
                and "grep" not in l
                and "tail" not in l
            )
            checks.append(
                {
                    "name": f"{row['label']} Writer",
                    "expected": 1,
                    "actual": count,
                    "ok": count >= 1,
                }
            )
        return checks
    active_groups = get_active_groups()
    active_pred_files = _load_active_prediction_files_from_configs()
    running_70_groups = any("--group " in l and "_70" in l and "multi_prediction_index" in l for l in ps_rows)
    expect_70_chain = running_70_groups or bool(active_groups and any(g.endswith("_70") for g in active_groups))

    def _prediction_file_fresh(name: str, max_age_sec: int) -> bool:
        path = POLYMARKET_DIR / name
        if not path.exists():
            return False
        try:
            age = time.time() - path.stat().st_mtime
        except Exception:
            return False
        return age <= max_age_sec

    # 70 writer + filter（仅在 70 交易组运行时要求）
    if expect_70_chain:
        filter70_count = sum(1 for l in ps_rows
                             if "filter_predictions_70.py" in l
                             and "python" in l
                             and "screen -s " not in l and "SCREEN -S " not in l and "SCREEN -dmS " not in l
                             and " login -pflq " not in l
                             and "grep" not in l and "tail" not in l)
        checks.append({
            "name": "Filter70",
            "expected": 1, "actual": filter70_count,
            "ok": filter70_count >= 1,
        })

    # V5 writers
    file_to_group = {**V5_FILE_TO_GROUP, **_dynamic_active_v5_file_to_group()}
    for label, pred_file in EXPECTED_V5_WRITERS + _dynamic_wallet_v5_writers() + _dynamic_active_v5_writers():
        if pred_file.startswith("predictions_v5_") and not _legacy_prediction_should_be_expected(pred_file):
            continue
        group_name = file_to_group.get(pred_file)
        if active_groups and group_name not in active_groups:
            continue
        output_key = pred_file
        count = sum(1 for l in ps_rows
                    if "prediction_writer_v5" in l
                    and output_key in l
                    and "grep" not in l and "tail" not in l)
        checks.append({
            "name": f"V5 {label}",
            "expected": 1, "actual": count,
            "ok": count >= 1,
        })

    # GRU writers
    if not active_groups or "gru_all" in active_groups:
        for label, pred_file in EXPECTED_GRU_WRITERS:
            if active_pred_files and pred_file not in active_pred_files:
                continue
            output_key = pred_file
            count = sum(1 for l in ps_rows
                        if "prediction_writer_gru" in l
                        and output_key in l
                        and "grep" not in l and "tail" not in l)
            checks.append({
                "name": label,  # label 已是 "GRU XRP" 等，无需再加 "GRU "
                "expected": 1, "actual": count,
                "ok": count >= 1,
            })

    return checks


def _check_pid_alive(pid_str: str) -> bool:
    """检查 PID 是否存活。"""
    try:
        os.kill(int(pid_str), 0)
        return True
    except (OSError, ValueError):
        return False


def check_ts_trading_processes() -> dict:
    """检查合并交易进程（multi_prediction_index）是否存活。

    新架构: 9 个 multi_prediction_index 进程，每个进程内含多个 trader 实例。
    通过 PID 文件 (multi_{group}.pid) 和 ps 进程表双重校验。

    返回:
        {
            "total_groups_alive": int,
            "total_traders": int,
            "groups": [{"name", "traders", "alive", "pid", "log_lines"}],
            "old_processes": int,  # 不应存在的旧式独立进程数
        }
    """
    groups: List[dict] = []

    # 获取进程表用于双重校验
    result = subprocess.run(["ps", "-eo", "pid,args", "-ww"],
                            capture_output=True, text=True)
    ps_lines = result.stdout

    expected_groups = get_expected_merged_groups()
    for group_name, expected_traders in expected_groups.items():
        pid_file = POLYMARKET_DIR / f"multi_{group_name}.pid"
        log_file = POLYMARKET_DIR / f"multi_{group_name}_stdout.log"
        alive = False
        pid_str = ""

        if pid_file.exists():
            pid_str = pid_file.read_text().strip()
            if pid_str and _check_pid_alive(pid_str):
                alive = True

        # 双重校验: ps 中是否有对应 --group 的进程
        ps_alive = any(f"multi_prediction_index" in l and (f"--group {group_name}" in l or f"--group={group_name}" in l)
                       for l in ps_lines.split("\n") if "grep" not in l)
        if ps_alive and not alive:
            alive = True  # PID 文件可能过时，但进程确实在运行

        log_lines = 0
        if log_file.exists():
            try:
                with open(log_file, "rb") as f:
                    log_lines = sum(1 for _ in f)
            except OSError:
                pass

        groups.append({
            "name": group_name,
            "traders": expected_traders,
            "alive": alive,
            "pid": pid_str if alive else "",
            "log_lines": log_lines,
        })

    # 检测旧式独立交易进程（不应存在，无论 ts-node 还是 node dist/）
    old_count = sum(1 for l in ps_lines.split("\n")
                    if "prediction_index" in l
                    and "multi_prediction_index" not in l
                    and "grep" not in l and "tail" not in l
                    and "api_health" not in l
                    and ("ts-node" in l or "node " in l))

    total_alive = sum(1 for g in groups if g["alive"])
    total_traders = sum(g["traders"] for g in groups if g["alive"])

    return {
        "total_groups_alive": total_alive,
        "total_traders": total_traders,
        "groups": groups,
        "old_processes": old_count,
        "expected_groups": expected_groups,
        "expected_total_traders": sum(expected_groups.values()),
    }


# ─── v2 新增检查 ─────────────────────────────────────────

def check_websocket_health_all() -> dict:
    """扫描合并进程日志 + 旧日志的 WebSocket 状态和 REST timeout 统计。"""
    ws_connected = 0
    ws_disconnected = 0
    ws_unknown = 0
    rest_timeouts = 0
    rest_network_errors = 0
    disconnected_dirs: List[str] = []
    timeout_dirs: List[Tuple[str, int]] = []

    log_files_to_scan: List[Tuple[str, Path]] = []

    # 合并进程日志（优先）
    for log_file in sorted(POLYMARKET_DIR.glob("multi_*_stdout.log")):
        log_files_to_scan.append((log_file.stem, log_file))

    # 旧式日志（兜底，可能有历史数据）
    for log_dir in sorted(POLYMARKET_DIR.glob("logs_*")):
        for log_file in log_dir.glob("trading*.log"):
            log_files_to_scan.append((log_dir.name, log_file))

    for dir_label, log_file in log_files_to_scan:
        try:
            with open(log_file, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                read_from = max(0, size - 8000)
                f.seek(read_from)
                tail = f.read().decode("utf-8", errors="replace")
        except OSError:
            continue

        ws_lines = [l for l in tail.split("\n") if "WebSocket" in l]
        if ws_lines:
            last_ws = ws_lines[-1]
            if "已连接" in last_ws or "活跃" in last_ws:
                ws_connected += 1
            elif "已断开" in last_ws or "重连" in last_ws:
                ws_disconnected += 1
                disconnected_dirs.append(dir_label)
            else:
                ws_unknown += 1
        else:
            ws_unknown += 1

        to_count = tail.count("timeout of 10000ms") + tail.count("timeout of 15000ms")
        ne_count = tail.count("网络错误")
        rest_timeouts += to_count
        rest_network_errors += ne_count
        if to_count > 2:
            timeout_dirs.append((dir_label, to_count))

    return {
        "ws_connected": ws_connected,
        "ws_disconnected": ws_disconnected,
        "ws_unknown": ws_unknown,
        "rest_timeouts": rest_timeouts,
        "rest_network_errors": rest_network_errors,
        "disconnected_dirs": disconnected_dirs[:10],
        "timeout_dirs": sorted(timeout_dirs, key=lambda x: -x[1])[:10],
    }


def check_duplicate_processes() -> List[dict]:
    """检测重复的 Python 写入器和 TS 交易进程。"""
    result = subprocess.run(["ps", "-eo", "pid,args", "-ww"],
                            capture_output=True, text=True)
    lines = result.stdout.split("\n")
    dupes = []

    # V5 写入器: 检查同一 predictions 文件是否有多个写入器
    v5_pids: Dict[str, List[str]] = {}
    for l in lines:
        m = re.search(r"prediction_writer_v5.*?(predictions_v5_\w+\.json)", l)
        if m and "grep" not in l and "tail" not in l:
            key = m.group(1)
            pid = l.strip().split()[0]
            v5_pids.setdefault(key, []).append(pid)
    for key, pids in v5_pids.items():
        if len(pids) > 1:
            dupes.append({"type": "V5 Writer", "key": key,
                          "count": len(pids), "pids": pids})

    # Ensemble 写入器
    ens_pids = [l.strip().split()[0] for l in lines
                if "ensemble_prediction_writer" in l
                and "python" in l
                and "--output-70" not in l
                and "screen -s " not in l and "SCREEN -S " not in l and "SCREEN -dmS " not in l
                and " login -pflq " not in l
                and "grep" not in l and "tail" not in l
                and "api_health" not in l]
    if len(ens_pids) > 1:
        dupes.append({"type": "Ensemble Writer", "key": "ensemble",
                      "count": len(ens_pids), "pids": ens_pids})

    # GRU 写入器
    gru_pids: Dict[str, List[str]] = {}
    for l in lines:
        m = re.search(r"prediction_writer_gru.*?(predictions_gru_\w+\.json)", l)
        if m and "grep" not in l and "tail" not in l:
            key = m.group(1)
            pid = l.strip().split()[0]
            gru_pids.setdefault(key, []).append(pid)
    for key, pids in gru_pids.items():
        if len(pids) > 1:
            dupes.append({"type": "GRU Writer", "key": key,
                          "count": len(pids), "pids": pids})

    # 合并进程: 检查同一 --group 是否有多个 multi_prediction_index 进程
    group_pids: Dict[str, List[str]] = {}
    for l in lines:
        m = re.search(r"multi_prediction_index.*--group[=\s]+(\S+)", l)
        if m and "grep" not in l and "tail" not in l:
            key = m.group(1)
            pid = l.strip().split()[0]
            group_pids.setdefault(key, []).append(pid)
    for key, pids in group_pids.items():
        node_pids = [p for p in pids if any(
            f"node" in ll and p in ll for ll in lines
            if "npm exec" not in ll)]
        if len(node_pids) > 1:
            dupes.append({"type": "合并进程", "key": f"group={key}",
                          "count": len(node_pids), "pids": node_pids})

    # 旧式独立交易进程: 不应再存在（无论 ts-node 还是 node dist/）
    old_ts_pids = []
    for l in lines:
        if ("prediction_index" in l
                and "multi_prediction_index" not in l
                and "grep" not in l and "tail" not in l
                and "api_health" not in l
                and ("ts-node" in l or "node " in l)):
            pid = l.strip().split()[0]
            old_ts_pids.append(pid)
    if old_ts_pids:
        dupes.append({"type": "旧式独立TS(应停止)", "key": "prediction_index",
                      "count": len(old_ts_pids), "pids": old_ts_pids})

    # 旧模型进程检测: 不应存在的 PREDICTION_SUFFIX=_A/_B/_C/_E/_F 进程
    old_model_pids: Dict[str, List[str]] = {}
    for l in lines:
        m = re.search(r"PREDICTION_SUFFIX=_([ABCEF])\b", l)
        if m and "prediction_index" in l and "grep" not in l:
            key = f"旧模型 SUFFIX=_{m.group(1)}"
            pid = l.strip().split()[0]
            old_model_pids.setdefault(key, []).append(pid)
    for key, pids in old_model_pids.items():
        dupes.append({"type": "旧模型(应停止)", "key": key,
                      "count": len(pids), "pids": pids})

    return dupes


def check_zombie_processes() -> List[dict]:
    """检测僵尸进程 (state Z)。僵尸无法被 kill，需杀父进程 PPID 才能清理。"""
    result = subprocess.run(
        ["ps", "-eo", "state,ppid,pid,args", "-ww"],
        capture_output=True, text=True,
    )
    zombies: List[dict] = []
    for line in result.stdout.split("\n"):
        if not line.strip():
            continue
        parts = line.split(None, 3)  # state ppid pid args
        if len(parts) < 4:
            continue
        state, ppid, pid, args = parts[0], parts[1], parts[2], (parts[3] if len(parts) > 3 else "")
        if state == "Z":
            zombies.append({"pid": pid, "ppid": ppid, "args": args[:80]})
    return zombies


CORE_ZOMBIE_PARENT_KEYWORDS = (
    "prediction_writer",
    "multi_prediction_index",
    "run_daily_training_scheduler.py",
    "collect_derivatives_realtime.py",
    "market_data_daemon",
    "run_top20_cycle.py",
    "run_top20_tuning_resume.py",
    "run_top20_profit_first_pipeline.py",
    "ensemble_prediction_writer.py",
)


def classify_zombie_processes(zombies: List[dict]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "core_blocking": [],
        "external_orphan": [],
        "other": [],
    }
    if not zombies:
        return out
    by_ppid: Dict[str, List[dict]] = {}
    for z in zombies:
        by_ppid.setdefault(str(z.get("ppid")), []).append(z)
    for ppid, rows in by_ppid.items():
        parent_cmd = get_process_command(ppid)
        info = {
            "ppid": ppid,
            "parent_cmd": parent_cmd,
            "count": len(rows),
            "zombies": rows,
        }
        low = str(parent_cmd).lower()
        if any(k.lower() in low for k in CORE_ZOMBIE_PARENT_KEYWORDS):
            out["core_blocking"].append(info)
        elif low.strip().startswith("python3 -") or low.strip().startswith("python -"):
            out["external_orphan"].append(info)
        else:
            out["other"].append(info)
    return out


def get_process_command(pid: str) -> str:
    """获取进程命令行，用于显示父进程是什么。"""
    try:
        r = subprocess.run(
            ["ps", "-p", pid, "-o", "args=", "-ww"],
            capture_output=True, text=True, timeout=2,
        )
        if r.returncode == 0 and r.stdout and r.stdout.strip():
            return r.stdout.strip()[:100]
    except Exception:
        pass
    return "(无法读取)"


def check_data_collector() -> Tuple[bool, str]:
    """检查实时数据采集器进程。"""
    result = subprocess.run(["ps", "-eo", "pid,args", "-ww"],
                            capture_output=True, text=True)
    count = 0
    for line in result.stdout.split("\n"):
        l = (line or "").strip()
        if not l or "collect_derivatives_realtime.py" not in l:
            continue
        ll = l.lower()
        # 仅统计真正的 python 采集器进程，过滤 screen/login/shell 包装层与查询命令自身
        if ("python" not in ll) or ("screen -s " in ll) or (" login -pflq " in ll):
            continue
        if ("api_health_check.py" in ll) or ("grep " in ll) or ("rg " in ll) or (" tail " in ll):
            continue
        count += 1
    if count == 1:
        return True, f"正常 ({count}个进程)"
    elif count == 0:
        return False, "未运行! V5/GRU 模型将无法获取最新数据"
    else:
        return False, f"重复! {count}个进程 (应为1个)"


def _parse_int_arg(command: str, name: str, default: int) -> int:
    m = re.search(rf"{re.escape(name)}(?:=|\s+)(\d+)", command)
    if not m:
        return int(default)
    try:
        return int(m.group(1))
    except ValueError:
        return int(default)


def _param_search_cadence_state(status: str, launch_reason: str, due: Optional[bool], backlog_windows: int) -> str:
    status = str(status or "").strip()
    launch_reason = str(launch_reason or "").strip()
    if status in {"queued", "running", "already_running"}:
        if launch_reason == "catchup" or int(backlog_windows or 0) >= 2:
            return "catchup_running"
        return "running_due_slice"
    if status == "failed_non_blocking":
        return "failed_non_blocking"
    if status == "not_due":
        return "not_due"
    if due is False and status in {"completed_recommendation_only", "completed_apply_eligible", "partial_checkpointed"}:
        return "not_due"
    return status or ("not_due" if due is False else "")


def _load_current_param_search_state(
    *,
    cadence_minutes: int = 240,
    max_catchup_slices_per_launch: int = 2,
) -> dict:
    runtime_status = _read_report_json(INCREMENTAL_PARAM_SEARCH_RUNTIME_STATUS)
    runtime_pid = int(runtime_status.get("pid") or 0)
    runtime_name = str(runtime_status.get("status") or "").strip()
    running = runtime_name in {"queued", "running"} and _check_pid_alive(str(runtime_pid))

    latest_finished_ts: Optional[float] = None
    latest_finished_status = ""
    latest_progress_ts: Optional[float] = None
    if ONLINE_LEARNING_HISTORY.exists():
        try:
            rows = ONLINE_LEARNING_HISTORY.read_text(encoding="utf-8").splitlines()
        except Exception:
            rows = []
        for raw in reversed(rows):
            try:
                row = json.loads(raw)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            if str(row.get("entry_type") or "") != "incremental_param_search_nonblocking":
                continue
            row_status = str(row.get("status") or "").strip()
            row_ts = (
                _parse_iso_ts(row.get("timestamp"))
                or _parse_iso_ts(row.get("finished_at"))
                or _parse_iso_ts(row.get("generated_at"))
            )
            if row_ts is None:
                continue
            if latest_finished_ts is None and row_status in {"completed_recommendation_only", "completed_apply_eligible", "partial_checkpointed", "failed_non_blocking"}:
                latest_finished_ts = row_ts
                latest_finished_status = row_status
            if latest_progress_ts is None and row_status in {"completed_recommendation_only", "completed_apply_eligible", "partial_checkpointed"}:
                latest_progress_ts = row_ts
            if latest_finished_ts is not None and latest_progress_ts is not None:
                break

    cadence_minutes = max(1, int(cadence_minutes))
    cadence_seconds = cadence_minutes * 60
    now_ts = time.time()
    if latest_progress_ts is None:
        due = True
        backlog_windows = 1
        next_due_at = None
    else:
        elapsed_seconds = max(0.0, now_ts - latest_progress_ts)
        backlog_windows = int(elapsed_seconds // cadence_seconds)
        due = backlog_windows >= 1
        next_due_at = datetime.fromtimestamp(latest_progress_ts + cadence_seconds, tz=timezone.utc).isoformat()

    launch_reason = ""
    slices_requested = 0
    if due:
        launch_reason = "catchup" if backlog_windows >= 2 else "scheduled"
        slices_requested = 1 if backlog_windows < 2 else min(backlog_windows, max(1, int(max_catchup_slices_per_launch)))

    return {
        "param_search_running": bool(running),
        "param_search_runtime_status": runtime_name or ("running" if running else ""),
        "param_search_cadence_minutes": cadence_minutes,
        "param_search_due": bool(due),
        "param_search_next_due_at": next_due_at,
        "param_search_backlog_windows": int(backlog_windows),
        "param_search_launch_reason": launch_reason,
        "param_search_slices_requested": int(slices_requested),
        "param_search_last_finished_at": (
            datetime.fromtimestamp(latest_finished_ts, tz=timezone.utc).isoformat() if latest_finished_ts is not None else None
        ),
        "param_search_latest_finished_status": latest_finished_status,
    }


def _parse_scheduler_log_stats_24h() -> dict:
    """从调度日志提取最近24小时成功/超时次数、最近训练时长、下次训练时间。"""
    out = {
        "success_24h": None,
        "timeout_24h": None,
        "training_timeout_24h": None,
        "runtime_apply_failure_24h": None,
        "legacy_timeout_24h": None,
        "legacy_cycle_timeout_24h": None,
        "post_training_refresh_timeout_24h": None,
        "param_search_nonblocking_timeout_24h": None,
        "latest_param_search_status": None,
        "latest_param_search_cadence_state": None,
        "param_search_cadence_minutes": None,
        "latest_param_search_due": None,
        "latest_param_search_next_due_at": None,
        "latest_param_search_backlog_windows": None,
        "latest_param_search_launch_reason": None,
        "latest_param_search_slices_requested": None,
        "latest_param_search_slices_completed": None,
        "latest_param_search_last_finished_at": None,
        "latest_nonblocking_refresh_status": None,
        "latest_runtime_apply_success": None,
        "latest_result_class": None,
        "has_cycle_history_24h": False,
        "history_source": None,
        "current_runtime_apply_repaired": False,
        "runtime_apply_repair_pending_verification": False,
        "last_train_duration_sec": None,
        "next_run_at": None,
    }
    # canonical 调度日志只认 launchd stdout/stderr；旧 daily_training_scheduler.log 保留历史证据，不再作为 live 主入口
    log_file = Path.home() / "Library" / "Logs" / "polyfun" / "daily_training_scheduler.launchd.log"
    if not log_file.exists():
        return out

    try:
        # 取尾部避免超大日志拖慢健康检查
        lines = deque(maxlen=5000)
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s = line.strip()
                if s:
                    lines.append(s)
    except OSError:
        return out

    now_ts = time.time()
    horizon_ts = now_ts - 24 * 3600
    ts_pat = re.compile(r"^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)]")
    next_run_pat = re.compile(r"下次训练:\s*([0-9]{4}-[0-9]{2}-[0-9]{2}\s+[0-9]{2}:[0-9]{2}:[0-9]{2})")

    success_24h = 0
    timeout_24h = 0
    last_start_ts: Optional[float] = None
    last_duration_sec: Optional[float] = None
    next_run_at: Optional[str] = None

    for line in lines:
        m_next = next_run_pat.search(line)
        if m_next:
            # 只做展示，不参与告警
            next_run_at = m_next.group(1)

        m = ts_pat.match(line)
        if not m:
            continue
        ts_text = m.group(1)
        try:
            line_ts = datetime.fromisoformat(ts_text).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue

        if "开始增量训练... trigger=" in line:
            last_start_ts = line_ts

        if last_start_ts is not None and (
            "训练完成，等待" in line
            or "跳过等待与链式监控" in line
            or "本次完成（含训练脚本异常" in line
        ):
            last_duration_sec = max(0.0, line_ts - last_start_ts)
            last_start_ts = None

        if line_ts < horizon_ts:
            continue
        if "本次完成。" in line and "含训练脚本异常" not in line:
            success_24h += 1
        if "timeout>" in line or ("训练脚本异常" in line and "timeout" in line.lower()):
            timeout_24h += 1

    out["success_24h"] = int(success_24h)
    out["timeout_24h"] = int(timeout_24h)
    out["legacy_timeout_24h"] = int(timeout_24h)
    out["last_train_duration_sec"] = int(last_duration_sec) if last_duration_sec is not None else None
    out["next_run_at"] = next_run_at

    if ONLINE_LEARNING_HISTORY.exists():
        def _history_row_ts(row: dict) -> Optional[float]:
            return (
                _parse_iso_ts(row.get("timestamp"))
                or _parse_iso_ts(row.get("finished_at"))
                or _parse_iso_ts(row.get("generated_at"))
            )

        def _infer_grouped_runtime_apply_success(summary: dict) -> bool | None:
            explicit = summary.get("runtime_apply_success")
            if isinstance(explicit, bool):
                return explicit
            writer_reload_errors = summary.get("writer_reload_errors") or []
            if writer_reload_errors:
                return False
            writers_to_reload = {
                str(item) for item in (summary.get("writers_to_reload") or []) if str(item).strip()
            }
            writers_reloaded = {
                str(item) for item in (summary.get("writers_reloaded") or []) if str(item).strip()
            }
            training_success = summary.get("training_success")
            core_refresh_success = summary.get("core_refresh_success")
            if not writers_to_reload and training_success is True and core_refresh_success is True:
                return True
            if writers_to_reload and writers_to_reload.issubset(writers_reloaded) and core_refresh_success is not False:
                return True
            return None

        def _infer_grouped_nonblocking_refresh_status(summary: dict) -> str:
            status = str(summary.get("nonblocking_refresh_status") or "").strip()
            return status or "not_started"

        def _classify_grouped_result(
            *,
            training_success: bool,
            runtime_apply_success: bool | None,
            core_refresh_success: bool | None,
            nonblocking_refresh_status: str,
        ) -> str:
            if not training_success:
                return "failed_critical_path"
            if runtime_apply_success is False:
                return "failed_runtime_apply"
            if core_refresh_success is False:
                return "failed_core_refresh"
            if nonblocking_refresh_status in {"failed_non_blocking", "nonblocking_slow"}:
                return "nonblocking_slow"
            if runtime_apply_success is True:
                return "ok"
            return ""

        def _extract_failed_writer_labels(summary: dict) -> list[str]:
            labels = set()
            for item in summary.get("writer_reload_errors") or []:
                text = str(item or "")
                for match in re.findall(r"polyfun\.writer\.core10\.[A-Za-z0-9_]+", text):
                    labels.add(match)
            return sorted(labels)

        def _launchctl_label_running(label: str) -> bool:
            if not label:
                return False
            try:
                proc = subprocess.run(
                    ["launchctl", "print", f"gui/{os.getuid()}/{label}"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
            except Exception:
                return False
            if proc.returncode != 0:
                return False
            stdout = proc.stdout or ""
            return "state = running" in stdout or "active count = 1" in stdout

        cycle_success = 0
        cycle_training_timeout = 0
        cycle_refresh_timeout = 0
        grouped_success = 0
        grouped_training_timeout = 0
        grouped_refresh_timeout = 0
        grouped_runtime_apply_failure = 0
        cycle_runtime_apply_failure = 0
        param_search_timeout = 0
        nonblocking_refresh_timeout = 0
        latest_param_search_status = None
        latest_nonblocking_refresh_status = None
        latest_runtime_apply_success = None
        latest_result_class = None
        cycle_row_count = 0
        grouped_row_count = 0
        latest_cycle_ts = None
        latest_grouped_ts = None
        latest_grouped_runtime_apply_failure = None
        try:
            for raw in ONLINE_LEARNING_HISTORY.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(raw)
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue
                ts = _history_row_ts(row)
                if ts is None or ts < horizon_ts:
                    continue
                entry_type = str(row.get("entry_type") or "")
                if entry_type == "daily_training_scheduler_cycle":
                    cycle_row_count += 1
                    latest_cycle_ts = ts if latest_cycle_ts is None or ts > latest_cycle_ts else latest_cycle_ts
                    summary = row.get("summary") if isinstance(row.get("summary"), dict) else {}
                    if bool(summary.get("training_success")):
                        cycle_success += 1
                    timeout_hits = int(summary.get("timeout_hits") or 0)
                    if timeout_hits > 0:
                        cycle_training_timeout += timeout_hits
                    if summary.get("runtime_apply_success") is False:
                        cycle_runtime_apply_failure += 1
                    if bool(summary.get("core_refresh_timeout")):
                        cycle_refresh_timeout += 1
                    if summary.get("param_search_status"):
                        latest_param_search_status = str(summary.get("param_search_status"))
                    if summary.get("nonblocking_refresh_status"):
                        latest_nonblocking_refresh_status = str(summary.get("nonblocking_refresh_status"))
                    latest_runtime_apply_success = summary.get("runtime_apply_success")
                    latest_result_class = str(summary.get("result_class") or latest_result_class or "")
                elif (
                    entry_type == "grouped_online_learning"
                    and str(row.get("mode") or "") in {"mainline_runtime_pool", "core10_only"}
                    and str(row.get("profile_json") or "").endswith("core10_incremental_profile.json")
                ):
                    grouped_row_count += 1
                    latest_grouped_ts = ts if latest_grouped_ts is None or ts > latest_grouped_ts else latest_grouped_ts
                    summary = row.get("summary") if isinstance(row.get("summary"), dict) else {}
                    training_success = summary.get("training_success")
                    runtime_apply_success = _infer_grouped_runtime_apply_success(summary)
                    if training_success is True:
                        grouped_success += 1
                    elif training_success is False:
                        grouped_training_timeout += 1
                    if runtime_apply_success is False:
                        grouped_runtime_apply_failure += 1
                        if latest_grouped_runtime_apply_failure is None or ts > latest_grouped_runtime_apply_failure[0]:
                            latest_grouped_runtime_apply_failure = (ts, summary)
                    if summary.get("core_refresh_success") is False:
                        grouped_refresh_timeout += 1
                    latest_runtime_apply_success = runtime_apply_success
                    latest_nonblocking_refresh_status = (
                        latest_nonblocking_refresh_status
                        or _infer_grouped_nonblocking_refresh_status(summary)
                    )
                    latest_result_class = str(
                        summary.get("result_class")
                        or _classify_grouped_result(
                            training_success=bool(training_success),
                            runtime_apply_success=runtime_apply_success,
                            core_refresh_success=(
                                summary.get("core_refresh_success")
                                if isinstance(summary.get("core_refresh_success"), bool)
                                else None
                            ),
                            nonblocking_refresh_status=_infer_grouped_nonblocking_refresh_status(summary),
                        )
                        or latest_result_class
                        or ""
                    )
                elif entry_type == "incremental_param_search_nonblocking":
                    if bool(row.get("param_search_nonblocking_timeout")):
                        param_search_timeout += 1
                    latest_param_search_status = str(row.get("status") or latest_param_search_status or "")
                elif entry_type == "mainline_refresh_nonblocking":
                    if bool(row.get("nonblocking_refresh_timeout")):
                        nonblocking_refresh_timeout += 1
                    latest_nonblocking_refresh_status = str(row.get("status") or latest_nonblocking_refresh_status or "")
        except Exception:
            pass
        use_grouped_history = bool(
            grouped_row_count > 0 and (
                cycle_row_count == 0
                or (latest_grouped_ts is not None and latest_cycle_ts is not None and latest_grouped_ts >= latest_cycle_ts)
            )
        )
        if cycle_row_count > 0 or grouped_row_count > 0:
            out["has_cycle_history_24h"] = True
        if use_grouped_history:
            out["success_24h"] = int(grouped_success)
            out["timeout_24h"] = int(grouped_training_timeout)
            out["training_timeout_24h"] = int(grouped_training_timeout)
            out["runtime_apply_failure_24h"] = int(grouped_runtime_apply_failure)
            out["history_source"] = "grouped_online_learning"
            out["legacy_cycle_timeout_24h"] = int(cycle_training_timeout)
        elif cycle_row_count > 0:
            out["has_cycle_history_24h"] = True
            out["success_24h"] = int(cycle_success)
            out["timeout_24h"] = int(cycle_training_timeout)
            out["training_timeout_24h"] = int(cycle_training_timeout)
            out["runtime_apply_failure_24h"] = int(cycle_runtime_apply_failure)
            out["history_source"] = "daily_training_scheduler_cycle"
        else:
            out["success_24h"] = int(success_24h)
            out["timeout_24h"] = int(timeout_24h)
            out["training_timeout_24h"] = None
            out["runtime_apply_failure_24h"] = None
            out["history_source"] = "legacy_log_only"
        failed_writer_labels = _extract_failed_writer_labels(latest_grouped_runtime_apply_failure[1]) if latest_grouped_runtime_apply_failure else []
        current_runtime_apply_repaired = bool(failed_writer_labels) and all(
            _launchctl_label_running(label) for label in failed_writer_labels
        )
        out["current_runtime_apply_repaired"] = bool(current_runtime_apply_repaired)
        out["runtime_apply_repair_pending_verification"] = bool(
            current_runtime_apply_repaired and int(out.get("runtime_apply_failure_24h") or 0) > 0
        )
        out["post_training_refresh_timeout_24h"] = int(nonblocking_refresh_timeout)
        out["param_search_nonblocking_timeout_24h"] = int(param_search_timeout)
        out["latest_param_search_status"] = latest_param_search_status or (
            "not_started" if str(out.get("history_source") or "") == "grouped_online_learning" and out.get("has_cycle_history_24h") else None
        )
        out["latest_nonblocking_refresh_status"] = latest_nonblocking_refresh_status or (
            "not_started" if str(out.get("history_source") or "") == "grouped_online_learning" and out.get("has_cycle_history_24h") else None
        )
        out["latest_runtime_apply_success"] = latest_runtime_apply_success
        out["latest_result_class"] = latest_result_class
        param_runtime = _read_report_json(INCREMENTAL_PARAM_SEARCH_RUNTIME_STATUS)
        current_param_state = _load_current_param_search_state(cadence_minutes=240, max_catchup_slices_per_launch=2)
        out["param_search_cadence_minutes"] = int(
            param_runtime.get("param_search_cadence_minutes")
            or current_param_state.get("param_search_cadence_minutes")
            or 0
        ) or None
        out["latest_param_search_due"] = (
            param_runtime.get("param_search_due")
            if param_runtime.get("param_search_due") is not None
            else current_param_state.get("param_search_due")
        )
        out["latest_param_search_next_due_at"] = (
            param_runtime.get("night_search_next_window_start_at")
            or param_runtime.get("param_search_next_due_at")
            or current_param_state.get("night_search_next_window_start_at")
            or current_param_state.get("param_search_next_due_at")
            or param_runtime.get("param_search_next_due_at")
            or None
        )
        out["latest_param_search_backlog_windows"] = int(
            param_runtime.get("night_search_canonical_backlog_total")
            or param_runtime.get("night_search_backlog_total")
            or current_param_state.get("night_search_backlog_total")
            or current_param_state.get("param_search_backlog_windows")
            or param_runtime.get("param_search_backlog_windows")
            or 0
        )
        out["latest_night_search_backlog_total"] = int(
            param_runtime.get("night_search_canonical_backlog_total")
            or param_runtime.get("night_search_backlog_total")
            or current_param_state.get("night_search_backlog_total")
            or out.get("latest_param_search_backlog_windows")
            or 0
        )
        out["latest_night_search_backlog_remaining"] = int(
            param_runtime.get("night_search_backlog_remaining")
            or out.get("latest_night_search_backlog_total")
            or 0
        )
        out["latest_param_search_launch_reason"] = str(
            current_param_state.get("param_search_launch_reason")
            or param_runtime.get("param_search_launch_reason")
            or ""
        )
        out["latest_param_search_slices_requested"] = int(
            current_param_state.get("param_search_slices_requested")
            or param_runtime.get("param_search_slices_requested")
            or 0
        )
        out["latest_param_search_slices_completed"] = int(param_runtime.get("param_search_slices_completed") or 0)
        out["latest_param_search_last_finished_at"] = (
            current_param_state.get("param_search_last_finished_at")
            or param_runtime.get("param_search_last_finished_at")
            or None
        )
        out["night_search_completion_status"] = (
            "completed"
            if int(out.get("latest_night_search_backlog_remaining") or 0) == 0
            and str(param_runtime.get("param_search_latest_finished_status") or "").strip() in {"completed", "success", "ok"}
            else "incomplete"
        )
        out["apply_effective_status"] = (
            "applied"
            if int(param_runtime.get("param_apply_applied_total") or 0) > 0
            else "not_applied"
        )
        if bool(current_param_state.get("param_search_running")):
            out["latest_param_search_status"] = str(
                current_param_state.get("param_search_runtime_status")
                or out.get("latest_param_search_status")
                or "running"
            )
        elif current_param_state.get("param_search_due") is False:
            out["latest_param_search_status"] = "not_due"
        out["latest_param_search_cadence_state"] = _param_search_cadence_state(
            str(out.get("latest_param_search_status") or ""),
            str(out.get("latest_param_search_launch_reason") or ""),
            out.get("latest_param_search_due"),
            int(out.get("latest_param_search_backlog_windows") or 0),
        )
    return out


def check_incremental_training_scheduler(expected_interval_minutes: Optional[int] = None) -> dict:
    """检查增量训练调度器是否在跑、参数是否符合预期。"""
    def _pid_matches_scheduler(pid_str: str) -> bool:
        try:
            pid_i = int(pid_str)
        except (TypeError, ValueError):
            return False
        try:
            p = subprocess.run(
                ["ps", "-p", str(pid_i), "-o", "command="],
                capture_output=True,
                text=True,
                timeout=3,
            )
        except Exception:
            return False
        if p.returncode != 0:
            return False
        cmd = (p.stdout or "").strip()
        return (
            (
                "run_daily_training_scheduler.py" in cmd
                or "scripts.run_daily_training_scheduler" in cmd
            )
            and "api_health_check.py" not in cmd
        )

    def _launchd_scheduler_loaded() -> bool:
        uid = str(os.getuid())
        try:
            proc = subprocess.run(
                ["launchctl", "print", f"gui/{uid}/{SCHEDULER_LAUNCHD_LABEL}"],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception:
            return False
        return proc.returncode == 0

    result = {
        "running": False,
        "launchd_loaded": False,
        "scheduler_process_running": False,
        "scheduler_pid_file_consistent": None,
        "pid": "",
        "command": "",
        "interval_minutes": None,
        "base_rounds": None,
        "calm_rounds": None,
        "shock_rounds": None,
        "shock_vol_ratio": None,
        "shock_ret_q": None,
        "rollback_utility_drop_abs": None,
        "rollback_down_wr_drop_abs": None,
        "success_24h": None,
        "timeout_24h": None,
        "training_timeout_24h": None,
        "runtime_apply_failure_24h": None,
        "legacy_timeout_24h": None,
        "legacy_cycle_timeout_24h": None,
        "post_training_refresh_timeout_24h": None,
        "param_search_nonblocking_timeout_24h": None,
        "latest_param_search_status": "",
        "latest_param_search_cadence_state": "",
        "param_search_cadence_minutes": None,
        "latest_param_search_due": None,
        "latest_param_search_next_due_at": None,
        "latest_param_search_backlog_windows": 0,
        "latest_param_search_launch_reason": "",
        "latest_param_search_slices_requested": 0,
        "latest_param_search_slices_completed": 0,
        "latest_param_search_last_finished_at": None,
        "latest_nonblocking_refresh_status": "",
        "latest_runtime_apply_success": None,
        "latest_result_class": "",
        "has_cycle_history_24h": False,
        "history_source": "",
        "current_runtime_apply_repaired": False,
        "runtime_apply_repair_pending_verification": False,
        "last_train_duration_sec": None,
        "next_run_at": None,
        "issues": [],
        "warnings": [],
    }
    expected_interval = (
        int(expected_interval_minutes)
        if isinstance(expected_interval_minutes, int) and int(expected_interval_minutes) > 0
        else int(EXPECTED_INCREMENTAL_INTERVAL_MINUTES)
    )

    try:
        ps = subprocess.run(
            ["ps", "-Ao", "pid,command"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        lines = []
        for raw in ps.stdout.splitlines():
            line = raw.strip()
            if "api_health_check.py" in line:
                continue
            if "run_daily_training_scheduler.py" in line or "scripts.run_daily_training_scheduler" in line:
                lines.append(line)
    except Exception as e:
        result["issues"].append(f"无法读取进程表: {str(e)[:60]}")
        return result

    result["launchd_loaded"] = _launchd_scheduler_loaded()
    if not lines:
        result["issues"].append("未发现 scripts.run_daily_training_scheduler 调度进程在运行")
    else:
        if len(lines) > 1:
            result["issues"].append(f"发现 {len(lines)} 个调度器进程（应为 1 个）")

        parts = lines[0].split(None, 1)
        if len(parts) == 2:
            pid, command = parts[0], parts[1]
            result["running"] = True
            result["scheduler_process_running"] = True
            result["pid"] = pid
            result["command"] = command
            result["interval_minutes"] = _parse_int_arg(
                command, "--interval-minutes", expected_interval
            )
            result["base_rounds"] = _parse_int_arg(command, "--base-rounds", EXPECTED_BASE_ROUNDS)
            result["calm_rounds"] = _parse_int_arg(command, "--calm-rounds", EXPECTED_CALM_ROUNDS)
            result["shock_rounds"] = _parse_int_arg(command, "--shock-rounds", EXPECTED_SHOCK_ROUNDS)
            m_sv = re.search(r"--shock-vol-ratio(?:=|\s+)([\d.]+)", command)
            m_sq = re.search(r"--shock-ret-q(?:=|\s+)([\d.]+)", command)
            m_ru = re.search(r"--rollback-utility-drop-abs(?:=|\s+)([\d.]+)", command)
            m_rd = re.search(r"--rollback-down-wr-drop-abs(?:=|\s+)([\d.]+)", command)
            result["shock_vol_ratio"] = float(m_sv.group(1)) if m_sv else EXPECTED_SHOCK_VOL_RATIO
            result["shock_ret_q"] = float(m_sq.group(1)) if m_sq else EXPECTED_SHOCK_RET_Q
            result["rollback_utility_drop_abs"] = (
                float(m_ru.group(1)) if m_ru else EXPECTED_ROLLBACK_UTILITY_DROP_ABS
            )
            result["rollback_down_wr_drop_abs"] = (
                float(m_rd.group(1)) if m_rd else EXPECTED_ROLLBACK_DOWN_WR_DROP_ABS
            )

            if result["interval_minutes"] != expected_interval:
                result["issues"].append(
                    f"interval={result['interval_minutes']} 分钟（期望 {expected_interval}）"
                )
            if result["base_rounds"] != EXPECTED_BASE_ROUNDS:
                result["issues"].append(
                    f"base_rounds={result['base_rounds']}（期望 {EXPECTED_BASE_ROUNDS}）"
                )
            if result["calm_rounds"] != EXPECTED_CALM_ROUNDS:
                result["issues"].append(
                    f"calm_rounds={result['calm_rounds']}（期望 {EXPECTED_CALM_ROUNDS}）"
                )
            if result["shock_rounds"] != EXPECTED_SHOCK_ROUNDS:
                result["issues"].append(
                    f"shock_rounds={result['shock_rounds']}（期望 {EXPECTED_SHOCK_ROUNDS}）"
                )
            if abs(float(result["shock_vol_ratio"]) - EXPECTED_SHOCK_VOL_RATIO) > 1e-9:
                result["issues"].append(
                    f"shock_vol_ratio={result['shock_vol_ratio']}（期望 {EXPECTED_SHOCK_VOL_RATIO}）"
                )
            if abs(float(result["shock_ret_q"]) - EXPECTED_SHOCK_RET_Q) > 1e-9:
                result["issues"].append(
                    f"shock_ret_q={result['shock_ret_q']}（期望 {EXPECTED_SHOCK_RET_Q}）"
                )
    if not result["launchd_loaded"]:
        result["issues"].append(f"launchd 未加载 {SCHEDULER_LAUNCHD_LABEL}")

    pid_file = PROJECT_ROOT / "logs" / "daily_training_scheduler.pid"
    if pid_file.exists():
        try:
            pid_from_file = pid_file.read_text(encoding="utf-8").strip()
        except OSError:
            pid_from_file = ""
        if pid_from_file:
            if not _check_pid_alive(pid_from_file):
                result["scheduler_pid_file_consistent"] = False
                result["warnings"].append(f"PID 文件残留: {pid_from_file} 已不存在")
            elif not _pid_matches_scheduler(pid_from_file):
                result["scheduler_pid_file_consistent"] = False
                result["warnings"].append(
                    f"PID 文件疑似陈旧/复用: {pid_from_file} 存活但不是调度器进程"
                )
            elif result["running"] and result["pid"] and pid_from_file != result["pid"]:
                result["scheduler_pid_file_consistent"] = False
                result["warnings"].append(
                    f"PID 文件({pid_from_file})与实际调度器 PID({result['pid']})不一致"
                )
            else:
                result["scheduler_pid_file_consistent"] = True
    else:
        result["scheduler_pid_file_consistent"] = False
        result["warnings"].append("缺少 logs/daily_training_scheduler.pid（仅属辅助元数据）")

    lock_file = PROJECT_ROOT / "logs" / "online_learning_scheduler.lock"
    if lock_file.exists():
        try:
            lock_pid = lock_file.read_text(encoding="utf-8").strip()
        except OSError:
            lock_pid = ""
        if lock_pid:
            stale_lock = (not _check_pid_alive(lock_pid)) or (not _pid_matches_scheduler(lock_pid))
            if stale_lock:
                removed = False
                try:
                    lock_file.unlink()
                    removed = True
                except OSError:
                    removed = False
                if not removed:
                    if not _check_pid_alive(lock_pid):
                        result["warnings"].append(
                            f"发现过期锁文件 online_learning_scheduler.lock (PID={lock_pid})"
                        )
                    else:
                        result["warnings"].append(
                            f"发现疑似陈旧锁文件 online_learning_scheduler.lock (PID={lock_pid} 已复用到其他进程)"
                        )

    log_stats = _parse_scheduler_log_stats_24h()
    result["success_24h"] = log_stats.get("success_24h")
    result["timeout_24h"] = log_stats.get("timeout_24h")
    result["training_timeout_24h"] = log_stats.get("training_timeout_24h")
    result["runtime_apply_failure_24h"] = log_stats.get("runtime_apply_failure_24h")
    result["legacy_timeout_24h"] = log_stats.get("legacy_timeout_24h")
    result["legacy_cycle_timeout_24h"] = log_stats.get("legacy_cycle_timeout_24h")
    result["post_training_refresh_timeout_24h"] = log_stats.get("post_training_refresh_timeout_24h")
    result["param_search_nonblocking_timeout_24h"] = log_stats.get("param_search_nonblocking_timeout_24h")
    result["latest_param_search_status"] = log_stats.get("latest_param_search_status")
    result["latest_param_search_cadence_state"] = log_stats.get("latest_param_search_cadence_state")
    result["param_search_cadence_minutes"] = log_stats.get("param_search_cadence_minutes")
    result["latest_param_search_due"] = log_stats.get("latest_param_search_due")
    result["latest_param_search_next_due_at"] = log_stats.get("latest_param_search_next_due_at")
    result["latest_param_search_backlog_windows"] = log_stats.get("latest_param_search_backlog_windows")
    result["latest_night_search_backlog_total"] = log_stats.get("latest_night_search_backlog_total")
    result["latest_night_search_backlog_remaining"] = log_stats.get("latest_night_search_backlog_remaining")
    result["latest_param_search_launch_reason"] = log_stats.get("latest_param_search_launch_reason")
    result["latest_param_search_slices_requested"] = log_stats.get("latest_param_search_slices_requested")
    result["latest_param_search_slices_completed"] = log_stats.get("latest_param_search_slices_completed")
    result["latest_param_search_last_finished_at"] = log_stats.get("latest_param_search_last_finished_at")
    result["night_search_completion_status"] = log_stats.get("night_search_completion_status")
    result["apply_effective_status"] = log_stats.get("apply_effective_status")
    result["latest_nonblocking_refresh_status"] = log_stats.get("latest_nonblocking_refresh_status")
    result["latest_runtime_apply_success"] = log_stats.get("latest_runtime_apply_success")
    result["latest_result_class"] = log_stats.get("latest_result_class")
    result["has_cycle_history_24h"] = bool(log_stats.get("has_cycle_history_24h"))
    result["history_source"] = log_stats.get("history_source")
    result["current_runtime_apply_repaired"] = bool(log_stats.get("current_runtime_apply_repaired"))
    result["runtime_apply_repair_pending_verification"] = bool(log_stats.get("runtime_apply_repair_pending_verification"))
    result["last_train_duration_sec"] = log_stats.get("last_train_duration_sec")
    result["next_run_at"] = log_stats.get("next_run_at")

    return result


def _read_report_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _acquire_api_health_lock() -> tuple[Optional[Any], bool]:
    API_HEALTH_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    handle = open(API_HEALTH_LOCK_FILE, "a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        return handle, True
    except BlockingIOError:
        return handle, False


def check_mainline_runtime_pool_status() -> dict:
    result = {
        "available": False,
        "issues": [],
        "active_traders_total": None,
        "active_cells_total": None,
        "active_symbols": [],
        "rules_json_drift_count": None,
        "source_drift_count": None,
        "prediction_model_version_mismatch_count": None,
        "runtime_cells_missing_process": None,
        "writer_process_missing_count": None,
        "runtime_cells_with_mainline_target": None,
        "runtime_cells_using_expected_mainline_family": None,
        "mainline_latest_revision_target_cells": None,
        "mainline_latest_revision_loaded_cells": None,
        "candidate_latest_revision_loaded_cells": None,
        "selector_zero_rows_default": None,
        "selector_zero_rows_70": None,
        "selector_overstrict_rows_total": None,
        "selector_active_shadow_scope_rows_total": None,
        "selector_xrp_runtime_unknown_cells_total": None,
        "filter70_btceth_threshold_selected": None,
        "filter70_xrp_threshold_selected": None,
        "filter70_constraints_passed": None,
        "prediction_uses_latest_revision": None,
        "prediction_uses_latest_revision_ratio": None,
        "prediction_latest_revision_sync_state": "unknown",
        "latest_revision_sync_state": "unknown",
        "latest_revision_lag_cells": None,
        "latest_revision_runtime_sync_gap_cells": None,
        "prediction_header_lag_cells": None,
        "latest_revision_lag_by_reason": {},
        "latest_revision_sync_issue_visible": None,
        "latest_revision_observation_lag_visible": None,
        "runtime_guard_review_age_hours": None,
        "postlaunch_guard_effect_age_hours": None,
        "runtime_guard_review_fresh": None,
        "postlaunch_guard_effect_fresh": None,
        "incremental_jobs_total": None,
        "incremental_jobs_by_symbol": {},
        "successful_writeback_jobs": None,
        "veto_only_jobs": None,
        "failed_jobs": None,
        "skipped_jobs": None,
        "incremental_stalled_jobs": None,
        "invalid_bootstrap_jobs": None,
        "incremental_needs_catchup": None,
        "incremental_missing_slots": None,
        "incremental_historical_missing_slots": None,
        "mainline_active_cells": None,
        "candidate_only_cells": None,
        "n_a_by_design_cells": None,
        "xrp_guard_coverage": "unknown",
        "fusion_prediction_fresh": False,
        "fusion_execution_state": "unknown",
        "fusion_default_cells": None,
        "fusion_70_cells": None,
        "fusion_simulated_trade_cells": None,
        "fusion_prediction_only_cells": None,
        "fusion_shadow_only_cells": None,
        "fusion_skipped_by_trade_decision_cells": None,
        "fusion_execution_not_landed_cells": None,
        "postlaunch_guarded_negative_cells_total": None,
        "postlaunch_unguarded_negative_cells_total": None,
        "postlaunch_drawdown_acceleration_soft_scaled_cells_total": None,
        "postlaunch_drawdown_acceleration_watch_only_negative_cells_total": None,
        "postlaunch_drawdown_acceleration_watch_only_worsening_negative_cells_total": None,
        "postlaunch_high_conf_same_direction_non_extreme_cells_total": None,
        "postlaunch_non_extreme_high_conf_same_direction_unmitigated_cells_total": None,
        "drawdown_halt_executor_consumed": None,
        "executor_env_audit_status": "missing",
        "guard_executor_compare_status": "missing",
        "buy_price_truth_status": "unknown",
        "buy_price_runtime_price_counts": {},
        "buy_price_runtime_source_counts": {},
        "buy_price_bp_tag_runtime_mismatch_count": None,
        "buy_price_rules_runtime_mismatch_count": None,
        "buy_price_rules_path_tag_mismatch_count": None,
        "buy_price_prediction_runtime_mismatch_count": None,
        "buy_price_prediction_observation_mismatch_count": None,
        "buy_price_rules_runtime_semantic_mismatch_count": None,
        "buy_price_rules_path_semantic_mismatch_count": None,
        "buy_price_current_runtime_truth_inconsistent_count": None,
        "buy_price_runtime_truth_current_conclusion": "unknown",
        "buy_price_prediction_observation_current_conclusion": "unknown",
        "buy_price_rules_path_semantics_current_conclusion": "unknown",
        "buy_price_current_conclusion": "unknown",
        "buy_price_shadow_validation_rows_total": None,
        "buy_price_shadow_validation_priority_counts": {},
        "buy_price_shadow_validation_results_status": "unknown",
        "buy_price_shadow_validation_results_rows_total": None,
        "buy_price_shadow_validation_results_candidate_for_micro_adjust_rows_total": None,
        "buy_price_focus_exp10_bp0520_eth_runtime_price": None,
        "buy_price_focus_exp10_bp0520_eth_runtime_source_of_truth": "unknown",
        "buy_price_focus_exp10_bp0520_eth_prediction_limit_price": None,
        "buy_price_focus_exp10_bp0520_eth_rules_json_buy_price": None,
        "buy_price_focus_exp10_bp0520_eth_best_balance_price": None,
        "buy_price_focus_exp10_bp0520_eth_recommendation": "unknown",
        "buy_price_focus_exp10_bp0520_eth_pre_init_045_clearly_better": None,
        "route_effectiveness_status": "unknown",
        "route_effectiveness_postlaunch_conclusion": "unknown",
        "route_effectiveness_postlaunch_abstain_effect_positive": None,
        "route_effectiveness_postlaunch_abstain_effect_neutral": None,
        "route_effectiveness_postlaunch_abstain_effect_negative": None,
        "route_effectiveness_postlaunch_route_reduction_without_quality_gain": None,
        "recent_change_semantic_audit_status": "unknown",
        "recent_change_semantic_confirmed_bug_count": None,
        "recent_change_semantic_semantic_misleading_count": None,
        "operational_bug_audit_status": "unknown",
        "operational_bug_audit_confirmed_bug_count": None,
        "operational_bug_audit_semantic_misleading_count": None,
        "operational_bug_audit_operational_risk_count": None,
        "runtime_parquet_writer_map_status": "unknown",
        "runtime_parquet_multiple_writers_detected_count": None,
        "runtime_parquet_collector_process_count": None,
        "runtime_parquet_sentiment_process_count": None,
        "runtime_sentiment_supervision_state": "unknown",
        "cfgi_runtime_state": "unknown",
        "safe_cleanup_status": "unknown",
        "safe_cleanup_delete_now_count": None,
        "safe_cleanup_deleted_now_count": None,
        "safe_cleanup_archive_candidate_count": None,
        "safe_cleanup_keep_count": None,
        "sol_status": "research_only_not_in_mainline",
        "current_conclusion": "",
    }

    if not POLYFUN_MAINLINE_FINAL_STATUS.exists():
        result["issues"].append("主线运行池审计报告缺失")
        return result

    mainline_report = _read_report_json(POLYFUN_MAINLINE_FINAL_STATUS)
    source_audit = _read_report_json(CORE10_PREDICTION_SOURCE_AUDIT)
    selector_zero_trade_audit = _read_report_json(SELECTOR_ZERO_TRADE_AUDIT)
    threshold70_search = _read_report_json(MAINLINE_70_THRESHOLD_SEARCH)
    recent_change_semantic_audit = _read_report_json(MAINLINE_RECENT_CHANGE_SEMANTIC_AUDIT)
    operational_bug_audit = _read_report_json(MAINLINE_OPERATIONAL_BUG_AUDIT)
    coverage_report = _read_report_json(DAILY_TRAINING_COVERAGE_REPORT)
    sol_report = _read_report_json(CORE10_SOL_RESEARCH_REPORT)
    executor = _read_report_json(DEFAULT_EXP10_ETH_EXECUTOR_ENV_AUDIT)
    compare = _read_report_json(DEFAULT_EXP10_ETH_GUARD_EXECUTOR_COMPARE)
    mainline_summary = mainline_report.get("summary") if isinstance(mainline_report.get("summary"), dict) else {}
    result["available"] = True
    source_summary = source_audit.get("summary") if isinstance(source_audit.get("summary"), dict) else {}
    selector_summary = (
        selector_zero_trade_audit.get("summary")
        if isinstance(selector_zero_trade_audit.get("summary"), dict)
        else {}
    )
    recent_change_semantic_audit_summary = (
        recent_change_semantic_audit.get("summary")
        if isinstance(recent_change_semantic_audit.get("summary"), dict)
        else {}
    )
    operational_bug_audit_summary = (
        operational_bug_audit.get("summary")
        if isinstance(operational_bug_audit.get("summary"), dict)
        else {}
    )
    threshold70_summary = (
        threshold70_search.get("summary")
        if isinstance(threshold70_search.get("summary"), dict)
        else {}
    )
    result["active_traders_total"] = int(mainline_summary.get("active_traders_total") or 0)
    result["active_cells_total"] = int(mainline_summary.get("active_cells_total") or 0)
    result["active_symbols"] = sorted(mainline_summary.get("active_symbols") or [])
    # Prefer the consolidated mainline final status when both reports exist.
    # The source audit can lag behind after cutovers/restarts and otherwise
    # pollute the health surface with stale loaded/writer counts.
    result["rules_json_drift_count"] = int(_first_present(mainline_summary.get("rules_json_drift_count"), source_summary.get("rules_json_drift_count"), 0))
    result["source_drift_count"] = int(_first_present(mainline_summary.get("source_drift_count"), source_summary.get("source_drift_count"), 0))
    result["prediction_model_version_mismatch_count"] = int(_first_present(mainline_summary.get("prediction_model_version_mismatch_count"), source_summary.get("prediction_model_version_mismatch_count"), 0))
    result["runtime_cells_missing_process"] = int(_first_present(source_summary.get("runtime_cells_missing_process"), 0))
    result["writer_process_missing_count"] = int(_first_present(mainline_summary.get("writer_process_missing_count"), source_summary.get("writer_process_missing_count"), 0))
    result["runtime_cells_with_mainline_target"] = int(_first_present(mainline_summary.get("runtime_cells_with_mainline_target"), source_summary.get("runtime_cells_with_core10_target"), 0))
    result["runtime_cells_using_expected_mainline_family"] = int(_first_present(mainline_summary.get("runtime_cells_using_expected_mainline_family"), source_summary.get("runtime_cells_using_expected_core10_family"), 0))
    mainline_latest_revision_target_cells = int(
        _first_present(
            mainline_summary.get("mainline_latest_revision_target_cells"),
            source_summary.get("mainline_latest_revision_target_cells"),
            0,
        )
    )
    mainline_latest_revision_loaded_cells = int(
        _first_present(
            mainline_summary.get("mainline_latest_revision_loaded_cells"),
            source_summary.get("mainline_latest_revision_loaded_cells"),
            0,
        )
    )
    candidate_latest_revision_loaded_cells = int(
        _first_present(
            mainline_summary.get("candidate_latest_revision_loaded_cells"),
            source_summary.get("candidate_latest_revision_loaded_cells"),
            0,
        )
    )
    result["mainline_latest_revision_target_cells"] = mainline_latest_revision_target_cells
    result["mainline_latest_revision_loaded_cells"] = mainline_latest_revision_loaded_cells
    result["candidate_latest_revision_loaded_cells"] = candidate_latest_revision_loaded_cells
    result["latest_revision_lag_cells"] = int(
        _first_present(
            mainline_summary.get("latest_revision_lag_cells"),
            source_summary.get("latest_revision_lag_cells"),
            0,
        )
    )
    result["prediction_header_lag_cells"] = int(
        _first_present(
            mainline_summary.get("prediction_header_lag_cells"),
            source_summary.get("prediction_header_lag_cells"),
            result["latest_revision_lag_cells"],
        )
    )
    result["prediction_header_pending_refresh_cells"] = int(
        _first_present(
            mainline_summary.get("prediction_header_pending_refresh_cells"),
            source_summary.get("prediction_header_pending_refresh_cells"),
            0,
        )
    )
    result["latest_revision_lag_by_reason"] = dict(
        (
            mainline_summary.get("latest_revision_lag_by_reason")
            if isinstance(mainline_summary.get("latest_revision_lag_by_reason"), dict)
            else source_summary.get("latest_revision_lag_by_reason")
        )
        or {}
    )
    result["selector_zero_rows_default"] = int(
        _first_present(
            selector_summary.get("default_zero_rows"),
            mainline_summary.get("selector_zero_rows_default"),
            0,
        )
    )
    result["selector_zero_rows_70"] = int(
        _first_present(
            selector_summary.get("profile70_zero_rows"),
            mainline_summary.get("selector_zero_rows_70"),
            0,
        )
    )
    result["selector_overstrict_rows_total"] = int(
        _first_present(
            selector_summary.get("overstrict_rows_total"),
            mainline_summary.get("selector_overstrict_rows_total"),
            0,
        )
    )
    result["selector_active_shadow_scope_rows_total"] = int(
        _first_present(
            selector_summary.get("active_shadow_scope_rows_total"),
            mainline_summary.get("selector_active_shadow_scope_rows_total"),
            0,
        )
    )
    result["selector_xrp_runtime_unknown_cells_total"] = int(
        _first_present(
            selector_summary.get("xrp_selector_runtime_unknown_cells_total"),
            mainline_summary.get("selector_xrp_runtime_unknown_cells_total"),
            0,
        )
    )
    result["filter70_btceth_threshold_selected"] = threshold70_summary.get(
        "selected_threshold_btceth",
        mainline_summary.get("filter70_btceth_threshold_selected"),
    )
    result["filter70_xrp_threshold_selected"] = threshold70_summary.get(
        "selected_threshold_xrp",
        mainline_summary.get("filter70_xrp_threshold_selected"),
    )
    result["filter70_constraints_passed"] = (
        threshold70_summary.get("selected_threshold_constraints_passed")
        if "selected_threshold_constraints_passed" in threshold70_summary
        else mainline_summary.get("filter70_constraints_passed")
    )
    if mainline_latest_revision_target_cells > 0:
        result["prediction_uses_latest_revision_ratio"] = round(mainline_latest_revision_loaded_cells / mainline_latest_revision_target_cells, 4)
        if mainline_latest_revision_loaded_cells <= 0:
            result["prediction_latest_revision_sync_state"] = "none"
            result["prediction_uses_latest_revision"] = False
        elif mainline_latest_revision_loaded_cells < mainline_latest_revision_target_cells:
            result["prediction_latest_revision_sync_state"] = "partial"
            result["prediction_uses_latest_revision"] = True
        else:
            result["prediction_latest_revision_sync_state"] = "full"
            result["prediction_uses_latest_revision"] = True
    else:
        result["prediction_latest_revision_sync_state"] = "n_a"
        result["prediction_uses_latest_revision"] = False
    result["latest_revision_sync_state"] = result["prediction_latest_revision_sync_state"]
    result["latest_revision_runtime_sync_gap_cells"] = max(0, mainline_latest_revision_target_cells - mainline_latest_revision_loaded_cells)
    result["latest_revision_sync_issue_visible"] = bool(mainline_latest_revision_target_cells > 0 and mainline_latest_revision_loaded_cells < mainline_latest_revision_target_cells)
    result["latest_revision_observation_lag_visible"] = bool(
        result["prediction_header_lag_cells"] or result["prediction_header_pending_refresh_cells"]
    )
    result["prediction_header_lag_streak_runs"] = int(mainline_summary.get("prediction_header_lag_streak_runs") or 0)
    result["prediction_header_lag_warning_visible"] = bool(mainline_summary.get("prediction_header_lag_warning_visible"))
    result["runtime_guard_review_age_hours"] = _report_age_hours(MAINLINE_RUNTIME_GUARD_REVIEW)
    result["postlaunch_guard_effect_age_hours"] = _report_age_hours(MAINLINE_POSTLAUNCH_GUARD_EFFECT)
    result["default_weakness_review_age_hours"] = _report_age_hours(MAINLINE_DEFAULT_WEAKNESS_REVIEW)
    result["runtime_guard_review_fresh"] = _is_report_fresh(MAINLINE_RUNTIME_GUARD_REVIEW)
    result["postlaunch_guard_effect_fresh"] = _is_report_fresh(MAINLINE_POSTLAUNCH_GUARD_EFFECT)
    result["default_weakness_review_fresh"] = _is_report_fresh(MAINLINE_DEFAULT_WEAKNESS_REVIEW)
    result["overnight_window_source"] = str(mainline_summary.get("overnight_window_source") or "unknown")
    result["incremental_jobs_total"] = int(mainline_summary.get("incremental_jobs_total") or 0)
    result["incremental_jobs_by_symbol"] = dict(mainline_summary.get("incremental_jobs_by_symbol") or {})
    result["successful_writeback_jobs"] = int(mainline_summary.get("successful_writeback_jobs") or 0)
    result["veto_only_jobs"] = int(mainline_summary.get("veto_only_jobs") or 0)
    result["failed_jobs"] = int(mainline_summary.get("failed_jobs") or 0)
    result["skipped_jobs"] = int(mainline_summary.get("skipped_jobs") or 0)
    result["invalid_bootstrap_jobs"] = int(mainline_summary.get("invalid_bootstrap_jobs_total") or 0)
    result["mainline_active_cells"] = int(mainline_summary.get("mainline_active_cells") or 0)
    result["candidate_only_cells"] = int(mainline_summary.get("candidate_only_cells") or 0)
    result["n_a_by_design_cells"] = int(mainline_summary.get("n_a_by_design_cells") or 0)
    result["incremental_stalled_jobs"] = 0
    result["incremental_needs_catchup"] = coverage_report.get("needs_catchup") if isinstance(coverage_report, dict) else None
    result["incremental_missing_slots"] = int((coverage_report.get("missing_slots") or 0)) if isinstance(coverage_report, dict) else None
    result["incremental_historical_missing_slots"] = int((coverage_report.get("historical_missing_slots") or 0)) if isinstance(coverage_report, dict) else None
    result["xrp_guard_coverage"] = "runtime_present" if "XRP" in set(result["active_symbols"] or []) else "n/a_by_design"
    result["fusion_prediction_fresh"] = bool(mainline_summary.get("fusion_prediction_fresh"))
    result["fusion_execution_state"] = str(mainline_summary.get("fusion_execution_state") or "unknown")
    result["fusion_default_cells"] = int(mainline_summary.get("fusion_default_cells") or 0)
    result["fusion_70_cells"] = int(mainline_summary.get("fusion_70_cells") or 0)
    result["fusion_simulated_trade_cells"] = int(mainline_summary.get("fusion_simulated_trade_cells") or 0)
    result["fusion_prediction_only_cells"] = int(mainline_summary.get("fusion_prediction_only_cells") or 0)
    result["fusion_shadow_only_cells"] = int(mainline_summary.get("fusion_shadow_only_cells") or 0)
    result["fusion_skipped_by_trade_decision_cells"] = int(mainline_summary.get("fusion_skipped_by_trade_decision_cells") or 0)
    result["fusion_execution_not_landed_cells"] = int(mainline_summary.get("fusion_execution_not_landed_cells") or 0)
    result["postlaunch_guarded_negative_cells_total"] = int(mainline_summary.get("postlaunch_guarded_negative_cells_total") or 0)
    result["postlaunch_unguarded_negative_cells_total"] = int(mainline_summary.get("postlaunch_unguarded_negative_cells_total") or 0)
    result["postlaunch_drawdown_acceleration_soft_scaled_cells_total"] = int(mainline_summary.get("postlaunch_drawdown_acceleration_soft_scaled_cells_total") or 0)
    result["postlaunch_drawdown_acceleration_watch_only_negative_cells_total"] = int(mainline_summary.get("postlaunch_drawdown_acceleration_watch_only_negative_cells_total") or 0)
    result["postlaunch_drawdown_acceleration_watch_only_worsening_negative_cells_total"] = int(
        mainline_summary.get("postlaunch_drawdown_acceleration_watch_only_worsening_negative_cells_total") or 0
    )
    result["postlaunch_high_conf_same_direction_non_extreme_cells_total"] = int(mainline_summary.get("postlaunch_high_conf_same_direction_non_extreme_cells_total") or 0)
    result["postlaunch_non_extreme_high_conf_same_direction_unmitigated_cells_total"] = int(mainline_summary.get("postlaunch_non_extreme_high_conf_same_direction_unmitigated_cells_total") or 0)
    result["postlaunch_comparison_evidence"] = str(mainline_summary.get("postlaunch_comparison_evidence") or "unknown")
    result["postlaunch_comparison_evidence_reasons"] = list(mainline_summary.get("postlaunch_comparison_evidence_reasons") or [])
    result["postlaunch_weaker_profile_by_total_pnl"] = str(mainline_summary.get("postlaunch_weaker_profile_by_total_pnl") or "parity")
    result["postlaunch_likely_primary_gap"] = str(mainline_summary.get("postlaunch_likely_primary_gap") or "balanced")
    result["postlaunch_default_market_selection_allowed_cells_total"] = int(mainline_summary.get("postlaunch_default_market_selection_allowed_cells_total") or 0)
    result["postlaunch_default_market_selection_abstained_cells_total"] = int(mainline_summary.get("postlaunch_default_market_selection_abstained_cells_total") or 0)
    result["postlaunch_default_market_selection_allow_reason_counts"] = dict(mainline_summary.get("postlaunch_default_market_selection_allow_reason_counts") or {})
    result["postlaunch_default_market_selection_abstain_reason_counts"] = dict(mainline_summary.get("postlaunch_default_market_selection_abstain_reason_counts") or {})
    result["postlaunch_default_market_quality_avg"] = round(float(mainline_summary.get("postlaunch_default_market_quality_avg") or 0.0), 4)
    result["default_weakness_primary_loss_symbol"] = str(mainline_summary.get("default_weakness_primary_loss_symbol") or "unknown")
    result["default_weakness_primary_loss_family"] = str(mainline_summary.get("default_weakness_primary_loss_family") or "unknown")
    result["default_weakness_primary_loss_trader"] = str(mainline_summary.get("default_weakness_primary_loss_trader") or "unknown")
    result["default_weakness_primary_current_selection_bucket"] = str(mainline_summary.get("default_weakness_primary_current_selection_bucket") or "unknown")
    result["default_weakness_selection_known_cells_total"] = int(mainline_summary.get("default_weakness_selection_known_cells_total") or 0)
    result["default_weakness_selection_unknown_cells_total"] = int(mainline_summary.get("default_weakness_selection_unknown_cells_total") or 0)
    result["default_weakness_selection_unknown_negative_cells_total"] = int(mainline_summary.get("default_weakness_selection_unknown_negative_cells_total") or 0)
    result["default_weakness_selection_unknown_total_pnl"] = round(float(mainline_summary.get("default_weakness_selection_unknown_total_pnl") or 0.0), 2)
    result["default_weakness_current_conclusion"] = str(mainline_summary.get("default_weakness_current_conclusion") or "unknown")
    result["loss_attribution_status"] = str(mainline_summary.get("loss_attribution_status") or "unknown")
    result["loss_attribution_total_pnl"] = round(float(mainline_summary.get("loss_attribution_total_pnl") or 0.0), 2)
    result["loss_attribution_primary_drag_profile"] = str(mainline_summary.get("loss_attribution_primary_drag_profile") or "unknown")
    result["loss_attribution_primary_drag_symbol"] = str(mainline_summary.get("loss_attribution_primary_drag_symbol") or "unknown")
    result["loss_attribution_primary_drag_family"] = str(mainline_summary.get("loss_attribution_primary_drag_family") or "unknown")
    result["loss_attribution_primary_drag_trader"] = str(mainline_summary.get("loss_attribution_primary_drag_trader") or "unknown")
    result["loss_attribution_primary_drag_guard_bucket"] = str(mainline_summary.get("loss_attribution_primary_drag_guard_bucket") or "unknown")
    result["loss_attribution_weaker_profile_by_total_pnl"] = str(mainline_summary.get("loss_attribution_weaker_profile_by_total_pnl") or "parity")
    result["loss_attribution_likely_primary_gap"] = str(mainline_summary.get("loss_attribution_likely_primary_gap") or "unknown")
    result["loss_attribution_current_conclusion"] = str(mainline_summary.get("loss_attribution_current_conclusion") or "unknown")
    result["buy_price_truth_status"] = str(mainline_summary.get("buy_price_truth_status") or "unknown")
    result["buy_price_runtime_price_counts"] = dict(mainline_summary.get("buy_price_runtime_price_counts") or {})
    result["buy_price_runtime_source_counts"] = dict(mainline_summary.get("buy_price_runtime_source_counts") or {})
    result["buy_price_bp_tag_runtime_mismatch_count"] = int(mainline_summary.get("buy_price_bp_tag_runtime_mismatch_count") or 0)
    result["buy_price_rules_runtime_mismatch_count"] = int(mainline_summary.get("buy_price_rules_runtime_mismatch_count") or 0)
    result["buy_price_rules_path_tag_mismatch_count"] = int(mainline_summary.get("buy_price_rules_path_tag_mismatch_count") or 0)
    result["buy_price_prediction_runtime_mismatch_count"] = int(mainline_summary.get("buy_price_prediction_runtime_mismatch_count") or 0)
    result["buy_price_prediction_observation_mismatch_count"] = int(mainline_summary.get("buy_price_prediction_observation_mismatch_count") or 0)
    result["buy_price_rules_runtime_semantic_mismatch_count"] = int(mainline_summary.get("buy_price_rules_runtime_semantic_mismatch_count") or 0)
    result["buy_price_rules_path_semantic_mismatch_count"] = int(mainline_summary.get("buy_price_rules_path_semantic_mismatch_count") or 0)
    result["buy_price_current_runtime_truth_inconsistent_count"] = int(mainline_summary.get("buy_price_current_runtime_truth_inconsistent_count") or 0)
    result["buy_price_runtime_truth_current_conclusion"] = str(mainline_summary.get("buy_price_runtime_truth_current_conclusion") or "unknown")
    result["buy_price_prediction_observation_current_conclusion"] = str(mainline_summary.get("buy_price_prediction_observation_current_conclusion") or "unknown")
    result["buy_price_rules_path_semantics_current_conclusion"] = str(mainline_summary.get("buy_price_rules_path_semantics_current_conclusion") or "unknown")
    result["buy_price_current_conclusion"] = str(mainline_summary.get("buy_price_current_conclusion") or "unknown")
    result["buy_price_shadow_validation_rows_total"] = int(mainline_summary.get("buy_price_shadow_validation_rows_total") or 0)
    result["buy_price_shadow_validation_priority_counts"] = dict(mainline_summary.get("buy_price_shadow_validation_priority_counts") or {})
    result["buy_price_shadow_validation_results_status"] = str(mainline_summary.get("buy_price_shadow_validation_results_status") or "unknown")
    result["buy_price_shadow_validation_results_rows_total"] = int(mainline_summary.get("buy_price_shadow_validation_results_rows_total") or 0)
    result["buy_price_shadow_validation_results_candidate_for_micro_adjust_rows_total"] = int(
        mainline_summary.get("buy_price_shadow_validation_results_candidate_for_micro_adjust_rows_total") or 0
    )
    result["buy_price_shadow_micro_adjust_review_status"] = str(mainline_summary.get("buy_price_shadow_micro_adjust_review_status") or "unknown")
    result["buy_price_shadow_micro_adjust_review_ready_rows_total"] = int(
        mainline_summary.get("buy_price_shadow_micro_adjust_review_ready_rows_total") or 0
    )
    result["buy_price_shadow_micro_adjust_review_hold_rows_total"] = int(
        mainline_summary.get("buy_price_shadow_micro_adjust_review_hold_rows_total") or 0
    )
    result["buy_price_shadow_micro_adjust_review_current_conclusion"] = str(
        mainline_summary.get("buy_price_shadow_micro_adjust_review_current_conclusion") or "unknown"
    )
    result["buy_price_focus_exp10_bp0520_eth_runtime_price"] = mainline_summary.get("buy_price_focus_exp10_bp0520_eth_runtime_price")
    result["buy_price_focus_exp10_bp0520_eth_runtime_source_of_truth"] = str(mainline_summary.get("buy_price_focus_exp10_bp0520_eth_runtime_source_of_truth") or "unknown")
    result["buy_price_focus_exp10_bp0520_eth_prediction_limit_price"] = mainline_summary.get("buy_price_focus_exp10_bp0520_eth_prediction_limit_price")
    result["buy_price_focus_exp10_bp0520_eth_rules_json_buy_price"] = mainline_summary.get("buy_price_focus_exp10_bp0520_eth_rules_json_buy_price")
    result["buy_price_focus_exp10_bp0520_eth_best_balance_price"] = mainline_summary.get("buy_price_focus_exp10_bp0520_eth_best_balance_price")
    result["buy_price_focus_exp10_bp0520_eth_recommendation"] = str(mainline_summary.get("buy_price_focus_exp10_bp0520_eth_recommendation") or "unknown")
    result["buy_price_focus_exp10_bp0520_eth_pre_init_045_clearly_better"] = mainline_summary.get("buy_price_focus_exp10_bp0520_eth_pre_init_045_clearly_better")
    result["route_effectiveness_status"] = str(mainline_summary.get("route_effectiveness_status") or "unknown")
    result["route_effectiveness_postlaunch_conclusion"] = str(mainline_summary.get("route_effectiveness_postlaunch_conclusion") or "unknown")
    result["route_effectiveness_postlaunch_abstain_effect_positive"] = int(mainline_summary.get("route_effectiveness_postlaunch_abstain_effect_positive") or 0)
    result["route_effectiveness_postlaunch_abstain_effect_neutral"] = int(mainline_summary.get("route_effectiveness_postlaunch_abstain_effect_neutral") or 0)
    result["route_effectiveness_postlaunch_abstain_effect_negative"] = int(mainline_summary.get("route_effectiveness_postlaunch_abstain_effect_negative") or 0)
    result["route_effectiveness_postlaunch_route_reduction_without_quality_gain"] = bool(
        mainline_summary.get("route_effectiveness_postlaunch_route_reduction_without_quality_gain")
    )
    result["recent_change_semantic_audit_status"] = str(
        _first_present(recent_change_semantic_audit.get("status"), mainline_summary.get("recent_change_semantic_audit_status"), "unknown")
    )
    result["recent_change_semantic_confirmed_bug_count"] = int(
        _first_present(
            (recent_change_semantic_audit_summary.get("classification_counts") or {}).get("confirmed_bug"),
            mainline_summary.get("recent_change_semantic_confirmed_bug_count"),
            0,
        )
    )
    result["recent_change_semantic_semantic_misleading_count"] = int(
        _first_present(
            (recent_change_semantic_audit_summary.get("classification_counts") or {}).get("semantic_misleading_but_runtime_correct"),
            mainline_summary.get("recent_change_semantic_semantic_misleading_count"),
            0,
        )
    )
    result["operational_bug_audit_status"] = str(
        _first_present(operational_bug_audit.get("status"), mainline_summary.get("operational_bug_audit_status"), "unknown")
    )
    result["operational_bug_audit_confirmed_bug_count"] = int(
        _first_present(
            (operational_bug_audit_summary.get("classification_counts") or {}).get("confirmed_bug"),
            mainline_summary.get("operational_bug_audit_confirmed_bug_count"),
            0,
        )
    )
    result["operational_bug_audit_semantic_misleading_count"] = int(
        _first_present(
            (operational_bug_audit_summary.get("classification_counts") or {}).get("semantic_misleading"),
            mainline_summary.get("operational_bug_audit_semantic_misleading_count"),
            0,
        )
    )
    result["operational_bug_audit_operational_risk_count"] = int(
        _first_present(
            (operational_bug_audit_summary.get("classification_counts") or {}).get("operational_risk"),
            mainline_summary.get("operational_bug_audit_operational_risk_count"),
            0,
        )
    )
    result["runtime_parquet_writer_map_status"] = str(mainline_summary.get("runtime_parquet_writer_map_status") or "unknown")
    result["runtime_parquet_multiple_writers_detected_count"] = int(mainline_summary.get("runtime_parquet_multiple_writers_detected_count") or 0)
    result["runtime_parquet_collector_process_count"] = int(mainline_summary.get("runtime_parquet_collector_process_count") or 0)
    result["runtime_parquet_sentiment_process_count"] = int(mainline_summary.get("runtime_parquet_sentiment_process_count") or 0)
    result["runtime_sentiment_supervision_state"] = str(mainline_summary.get("runtime_sentiment_supervision_state") or "unknown")
    result["cfgi_runtime_state"] = str(mainline_summary.get("cfgi_runtime_state") or "unknown")
    result["safe_cleanup_status"] = str(mainline_summary.get("safe_cleanup_status") or "unknown")
    result["safe_cleanup_delete_now_count"] = int(mainline_summary.get("safe_cleanup_delete_now_count") or 0)
    result["safe_cleanup_deleted_now_count"] = int(mainline_summary.get("safe_cleanup_deleted_now_count") or 0)
    result["safe_cleanup_archive_candidate_count"] = int(mainline_summary.get("safe_cleanup_archive_candidate_count") or 0)
    result["safe_cleanup_keep_count"] = int(mainline_summary.get("safe_cleanup_keep_count") or 0)
    result["overnight_comparison_evidence"] = str(mainline_summary.get("overnight_comparison_evidence") or "unknown")
    result["overnight_comparison_evidence_reasons"] = list(mainline_summary.get("overnight_comparison_evidence_reasons") or [])
    result["overnight_weaker_profile_by_total_pnl"] = str(mainline_summary.get("overnight_weaker_profile_by_total_pnl") or "parity")
    result["overnight_likely_primary_gap"] = str(mainline_summary.get("overnight_likely_primary_gap") or "balanced")
    result["overnight_default_market_selection_allowed_cells_total"] = int(mainline_summary.get("overnight_default_market_selection_allowed_cells_total") or 0)
    result["overnight_default_market_selection_abstained_cells_total"] = int(mainline_summary.get("overnight_default_market_selection_abstained_cells_total") or 0)
    result["overnight_default_market_selection_allow_reason_counts"] = dict(mainline_summary.get("overnight_default_market_selection_allow_reason_counts") or {})
    result["overnight_default_market_selection_abstain_reason_counts"] = dict(mainline_summary.get("overnight_default_market_selection_abstain_reason_counts") or {})
    result["overnight_default_market_quality_avg"] = round(float(mainline_summary.get("overnight_default_market_quality_avg") or 0.0), 4)
    result["writer_launchctl_status"] = str(mainline_summary.get("writer_launchctl_status") or "unknown")
    result["writer_launchctl_duplicate_count"] = int(mainline_summary.get("writer_launchctl_duplicate_count") or 0)
    result["writer_launchctl_manual_count"] = int(mainline_summary.get("writer_launchctl_manual_count") or 0)
    result["writer_launchctl_stopped_count"] = int(mainline_summary.get("writer_launchctl_stopped_count") or 0)
    result["writer_launchctl_critical_stopped_count"] = int(mainline_summary.get("writer_launchctl_critical_stopped_count") or 0)
    result["writer_launchctl_stopped_observation_count"] = int(mainline_summary.get("writer_launchctl_stopped_observation_count") or 0)
    result["writer_launchctl_not_launchd_managed_count"] = int(mainline_summary.get("writer_launchctl_not_launchd_managed_count") or 0)
    result["writer_launchctl_launchd_count"] = int(mainline_summary.get("writer_launchctl_launchd_count") or 0)
    result["writeback_guard_baseline_insufficient_units"] = int(_first_present(mainline_summary.get("writeback_guard_baseline_insufficient_units"), result.get("writeback_guard_baseline_insufficient_units"), 0))
    result["writeback_guard_observed_but_baseline_insufficient_units"] = int(_first_present(mainline_summary.get("writeback_guard_observed_but_baseline_insufficient_units"), result.get("writeback_guard_observed_but_baseline_insufficient_units"), 0))
    result["writeback_guard_effective_weakened_units"] = int(_first_present(mainline_summary.get("writeback_guard_effective_weakened_units"), result.get("writeback_guard_effective_weakened_units"), 0))
    result["writeback_guard_effective_weakened_with_baseline_insufficient_units"] = int(_first_present(mainline_summary.get("writeback_guard_effective_weakened_with_baseline_insufficient_units"), result.get("writeback_guard_effective_weakened_with_baseline_insufficient_units"), 0))
    result["writeback_guard_effective_challenger_trigger_units"] = int(_first_present(mainline_summary.get("writeback_guard_effective_challenger_trigger_units"), result.get("writeback_guard_effective_challenger_trigger_units"), 0))
    result["writeback_guard_effective_challenger_trigger_with_baseline_insufficient_units"] = int(_first_present(mainline_summary.get("writeback_guard_effective_challenger_trigger_with_baseline_insufficient_units"), result.get("writeback_guard_effective_challenger_trigger_with_baseline_insufficient_units"), 0))
    result["executor_env_audit_status"] = str(executor.get("status") or ("ok" if executor else "missing"))
    result["guard_executor_compare_status"] = str(compare.get("status") or ("ok" if compare else "missing"))
    result["drawdown_halt_executor_consumed"] = compare.get("drawdown_halt_executor_consumed")
    result["sol_status"] = str(
        (sol_report.get("status") if isinstance(sol_report, dict) else "")
        or mainline_summary.get("sol_status")
        or "research_only_not_in_mainline"
    )
    result["current_conclusion"] = str(_first_present(mainline_report.get("current_conclusion"), source_summary.get("current_conclusion"), ""))

    if result["active_traders_total"] <= 0:
        result["issues"].append("主线运行池交易器数量为 0")
    if result["active_cells_total"] <= 0:
        result["issues"].append("主线运行池组合单元数量为 0")
    if not result["active_symbols"]:
        result["issues"].append("主线运行池 active symbol 为空")
    if result["sol_status"] != "research_only_not_in_mainline":
        result["issues"].append(f"SOL 主线状态异常: {result['sol_status']}")
    if result["invalid_bootstrap_jobs"] > 0:
        result["issues"].append(f"增量训练无效引导数量异常: {result['invalid_bootstrap_jobs']}")
    if result["fusion_execution_not_landed_cells"] > 0:
        result["issues"].append(f"融合预测已生成但执行未落账单元: {result['fusion_execution_not_landed_cells']}")
    if result["rules_json_drift_count"] > 0:
        result["issues"].append(f"规则文件漂移数量异常: {result['rules_json_drift_count']}")
    if result["source_drift_count"] > 0:
        result["issues"].append(f"运行态来源切换未收口: {result['source_drift_count']}")
    if result["selector_overstrict_rows_total"] > 0:
        result["issues"].append(f"选择器过严样本仍存在: {result['selector_overstrict_rows_total']}")
    if result["selector_active_shadow_scope_rows_total"] > 0:
        result["issues"].append(f"active 主线仍存在 simulation-only 选择器范围: {result['selector_active_shadow_scope_rows_total']}")
    if result["selector_xrp_runtime_unknown_cells_total"] > 0:
        result["issues"].append(f"XRP 选择器运行态未知单元: {result['selector_xrp_runtime_unknown_cells_total']}")
    if result["writeback_guard_effective_weakened_with_baseline_insufficient_units"] > 0:
        result["issues"].append("增量写回守门仍把零基线样本计入 effective_weakened")
    if result["writeback_guard_effective_challenger_trigger_with_baseline_insufficient_units"] > 0:
        result["issues"].append("增量写回守门仍把零基线样本计入 effective_challenger_trigger")
    if result["drawdown_halt_executor_consumed"] is False:
        result["issues"].append("回撤暂停尚未被执行器真正消费")
    if result["buy_price_current_runtime_truth_inconsistent_count"] > 0:
        result["issues"].append(f"主线买价真相存在 live conflict: {result['buy_price_current_runtime_truth_inconsistent_count']}")
    if result["recent_change_semantic_confirmed_bug_count"] > 0:
        result["issues"].append(f"最近改动附近仍有 confirmed bug: {result['recent_change_semantic_confirmed_bug_count']}")
    if result["operational_bug_audit_confirmed_bug_count"] > 0:
        result["issues"].append(f"主线运维口径仍有 confirmed bug: {result['operational_bug_audit_confirmed_bug_count']}")
    if result["runtime_parquet_multiple_writers_detected_count"] > 0:
        result["issues"].append(f"活跃 parquet 数据集仍存在多写者: {result['runtime_parquet_multiple_writers_detected_count']}")
    if result["runtime_sentiment_supervision_state"] not in {"independent_launchd", "not_running"}:
        result["issues"].append(f"情绪采集运行态托管未独立: {result['runtime_sentiment_supervision_state']}")
    if result["cfgi_runtime_state"] not in {"retired_cleanly", "active_and_fresh"}:
        result["issues"].append(f"CFGI 运行态仍处于半残状态: {result['cfgi_runtime_state']}")
    return result


def check_stale_dry_run_processes(max_age_min: int = 60) -> List[dict]:
    """检查超过阈值仍在运行的 online_learning --dry-run 进程。"""
    findings: List[dict] = []
    try:
        r = subprocess.run(
            ["ps", "-eo", "pid,lstart,args", "-ww"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return findings
    if r.returncode != 0:
        return findings

    now_local = datetime.now()
    for line in r.stdout.splitlines():
        if "online_learning_" not in line or "--dry-run" not in line:
            continue
        parts = line.strip().split(None, 7)
        if len(parts) < 7:
            continue
        pid = parts[0]
        start_txt = " ".join(parts[1:6])  # Thu Feb 26 08:19:53 2026
        cmd = " ".join(parts[6:])
        try:
            started = datetime.strptime(start_txt, "%a %b %d %H:%M:%S %Y")
        except Exception:
            continue
        age_s = max(0.0, (now_local - started).total_seconds())
        if age_s >= max_age_min * 60:
            findings.append(
                {
                    "pid": pid,
                    "age_min": round(age_s / 60.0, 1),
                    "cmd": cmd[:180],
                }
            )
    return findings


def _parse_iso_ts(value: Optional[str]) -> Optional[float]:
    if not value or not isinstance(value, str):
        return None
    txt = value.strip()
    if not txt:
        return None
    try:
        return datetime.fromisoformat(txt.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _scan_recent_transport_errors(hours: int = 24) -> Dict[str, int]:
    """扫描 /tmp/polyfun_multi*.log 最近 N 小时常见传输异常计数。"""
    out = {
        "json_parse": 0,
        "json_unexpected_end": 0,
        "clob_400": 0,
        "clob_502": 0,
    }
    now_ts = time.time()
    horizon = max(1, int(hours)) * 3600
    ts_pattern = re.compile(r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")
    for p in sorted(Path("/tmp").glob("polyfun_multi*.log")):
        try:
            lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            continue
        for line in lines:
            m = ts_pattern.search(line)
            if not m:
                continue
            try:
                ts = datetime.fromisoformat(m.group(1)).replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                continue
            if now_ts - ts > horizon:
                continue
            lo = line.lower()
            if "json parse error" in lo:
                out["json_parse"] += 1
            if "unexpected end of json input" in lo:
                out["json_unexpected_end"] += 1
            if "400" in lo and ("clob" in lo or "api key" in lo):
                out["clob_400"] += 1
            if "502" in lo or "bad gateway" in lo:
                out["clob_502"] += 1
    return out


def check_runtime_trade_guards() -> dict:
    """检查多组合运行时风控状态（数据质量门控 + DOWN熔断）。"""
    result = {
        "exists": False,
        "mode": "unknown",
        "severity": "unknown",
        "absolute_stop_policy": "disabled_global",
        "runtime_protective_state": "none",
        "live_rollout_target": "70_cmp_exp10_bp_dyn_0300_0440_eth-ETH",
        "rollback_armed": True,
        "trigger_class": "",
        "penalty": 0.0,
        "bet_scale": 1.0,
        "api_error_rate": 0.0,
        "api_error_sources": [],
        "key_source_errors": [],
        "secondary_source_errors": [],
        "non_key_source_errors": [],
        "observation_counts": {},
        "consecutive_bad_windows": {},
        "consecutive_good_windows": {},
        "reason": "",
        "stale_sources": [],
        "active_breakers": 0,
        "total_breakers": 0,
        "active_names": [],
        "v2_total_cells": 0,
        "v2_enforce_cells": 0,
        "v2_shadow_cells": 0,
        "v2_blocked_cells": 0,
        "v2_hard_cells": 0,
        "v2_soft_cells": 0,
        "down_fast_hard_count": 0,
        "cross_layer_escalation_count": 0,
        "stale_last_action_count": 0,
        "up_v2_total_cells": 0,
        "up_v2_enforce_cells": 0,
        "up_v2_shadow_cells": 0,
        "up_v2_blocked_cells": 0,
        "up_v2_soft_cells": 0,
        "calibration_total_cells": 0,
        "calibration_enforce_cells": 0,
        "calibration_shadow_cells": 0,
        "calibration_active_cells": 0,
        "expectancy_total_cells": 0,
        "expectancy_enforce_cells": 0,
        "expectancy_shadow_cells": 0,
        "expectancy_blocked_cells": 0,
        "expectancy_degraded_cells": 0,
        "expect_fast_block_count": 0,
        "expect_manual_block_count": 0,
        "expect_manual_degrade_count": 0,
        "manual_block_count_by_symbol_direction": {},
        "manual_degrade_count_by_symbol_direction": {},
        "manual_action_runtime_hits": [],
        "drift_total_cells": 0,
        "drift_enforce_cells": 0,
        "drift_shadow_cells": 0,
        "drift_active_cells": 0,
        "drift_skip_cells": 0,
        "drift_nonzero_delta_cells": 0,
        "meta_total_cells": 0,
        "meta_enforce_cells": 0,
        "meta_shadow_cells": 0,
        "selector_total_cells": 0,
        "selector_enforce_cells": 0,
        "selector_shadow_cells": 0,
        "selector_ineligible_cells": 0,
        "ensemble_consensus_total_symbols": 0,
        "ensemble_consensus_enforce_symbols": 0,
        "ensemble_consensus_shadow_symbols": 0,
        "ensemble_consensus_blocked_symbols": 0,
        "ensemble_consensus_blocked_effective_symbols": 0,
        "ensemble_consensus_mode_map": {},
        "regime_gate_total_symbols": 0,
        "regime_gate_enforce_symbols": 0,
        "regime_gate_shadow_symbols": 0,
        "regime_gate_active_symbols": 0,
        "model_review_total_cells": 0,
        "model_review_attention_cells": 0,
        "model_review_demote_cells": 0,
        "shock_v2_total_cells": 0,
        "shock_v2_enforce_cells": 0,
        "shock_v2_shadow_cells": 0,
        "shock_v2_active_cells": 0,
        "shock_v2_blocked_cells": 0,
        "shock_v2_blocked_with_eta": 0,
        "shock_v2_blocked_without_eta": 0,
        "shock_v2_blocked_eta_by_cluster": {},
        "combo_pause_total_cells": 0,
        "combo_pause_enforce_cells": 0,
        "combo_pause_shadow_cells": 0,
        "combo_pause_active_cells": 0,
        "combo_pause_blocked_cells": 0,
        "combo_directional_total": 0,
        "combo_directional_active": 0,
        "combo_directional_blocked": 0,
        "cluster_breakdown": {},
        "cooldown_total_active": 0,
        "cooldown_symbol_active": 0,
        "cooldown_symbol_direction_active": 0,
        "cooldown_scope": "",
        "mainline_extreme_total_symbols": 0,
        "mainline_extreme_active_symbols": 0,
        "direction_loss_cooldown_total": 0,
        "direction_loss_cooldown_blocked": 0,
        "drawdown_acceleration_total": 0,
        "drawdown_acceleration_blocked": 0,
        "evidence_evaluated": 0,
        "evidence_passed": 0,
        "evidence_skipped_should_trade": 0,
        "evidence_skipped_quality": 0,
        "evidence_skipped_drift": 0,
        "evidence_skipped_expectancy": 0,
        "evidence_skipped_meta": 0,
        "evidence_skipped_selector": 0,
        "evidence_skipped_regime": 0,
        "evidence_skipped_shock": 0,
        "evidence_skipped_combo": 0,
        "evidence_skipped_down": 0,
        "evidence_skipped_up": 0,
        "evidence_skipped_extreme": 0,
        "evidence_skipped_direction_loss": 0,
        "evidence_skipped_drawdown_acceleration": 0,
        "evidence_dummy_eval_ready": False,
        "layer_evidence_counts": {},
        "recent_json_parse_24h": 0,
        "recent_json_unexpected_end_24h": 0,
        "recent_clob_400_24h": 0,
        "recent_clob_502_24h": 0,
        "guard_files": [],
        "covered_groups": [],
        "running_groups": [],
        "newest_guard_age_s": None,
        "newest_payload_age_s": None,
        "hold_until": "",
        "hold_remaining_min": 0.0,
        "worst_group": "",
        "runtime_data_quality_audit_status": "unknown",
        "runtime_data_quality_audit_open_p1": None,
        "runtime_data_quality_false_positive_risk": None,
        "observation_only_data_quality": False,
        "issues": [],
    }
    running_groups: set[str] = set()
    try:
        ps_out = subprocess.run(
            ["ps", "axww", "-o", "pid=,command="],
            capture_output=True,
            text=True,
            timeout=10,
        )
        for line in ps_out.stdout.splitlines():
            m = re.search(r"multi_prediction_index.*--group[=\s]+(\S+)", line)
            if m:
                running_groups.add(m.group(1))
    except Exception:
        running_groups = set()
    result["running_groups"] = sorted(running_groups)

    guard_dir = POLYMARKET_DIR / "logs"
    all_guard_files = sorted(guard_dir.glob(RUNTIME_GUARD_STATUS_GLOB))
    group_guard_files = [
        p for p in all_guard_files
        if re.match(r"runtime_trade_guards_[A-Za-z0-9_-]+\.json$", p.name)
    ]
    # 若已存在按 group 拆分的文件，则忽略 legacy 单文件，避免历史残留干扰判定。
    guard_files = group_guard_files if group_guard_files else all_guard_files
    if not guard_files and RUNTIME_GUARD_STATUS_FILE.exists():
        guard_files = [RUNTIME_GUARD_STATUS_FILE]
    if not guard_files:
        result["issues"].append("缺少 runtime_trade_guards.json（无法判定数据质量门控/熔断状态）")
        return result

    def _guard_group_name(path: Path) -> Optional[str]:
        m = re.match(r"runtime_trade_guards_([A-Za-z0-9_-]+)\.json$", path.name)
        return m.group(1) if m else None

    # 仅对“当前运行中的 group”做时效和状态判定，避免停用组的旧 guard 文件造成误报。
    if running_groups:
        filtered: List[Path] = []
        for gf in guard_files:
            g = _guard_group_name(gf)
            if g and g not in running_groups:
                continue
            filtered.append(gf)
        if filtered:
            guard_files = filtered

    result["exists"] = True
    result["guard_files"] = [p.name for p in guard_files]

    def _to_float(v, default=0.0) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)

    mode_rank = {"ok": 0, "soft_degraded": 1, "hard_degraded": 2, "halt": 3, "unknown": -1}
    worst_mode = "ok"
    worst_reason = ""
    worst_group = ""
    max_penalty = 0.0
    min_bet_scale = 1.0
    max_api_error_rate = 0.0
    api_source_union: set[str] = set()
    stale_union: set[str] = set()
    all_breaker_keys: set[str] = set()
    active_breaker_keys: set[str] = set()
    v2_cells: Dict[str, dict] = {}
    up_v2_cells: Dict[str, dict] = {}
    calibration_cells: Dict[str, dict] = {}
    expectancy_cells: Dict[str, dict] = {}
    drift_cells: Dict[str, dict] = {}
    meta_cells: Dict[str, dict] = {}
    selector_cells: Dict[str, dict] = {}
    shock_v2_cells: Dict[str, dict] = {}
    combo_pause_cells: Dict[str, dict] = {}
    cooldown_cells: Dict[str, dict] = {}
    mainline_extreme_cells: Dict[str, dict] = {}
    direction_loss_cooldown_cells: Dict[str, dict] = {}
    drawdown_acceleration_cells: Dict[str, dict] = {}
    ensemble_consensus_symbols: Dict[str, dict] = {}
    regime_gate_symbols: Dict[str, dict] = {}
    evidence_totals = {
        "evaluated": 0,
        "passed": 0,
        "skipped_should_trade": 0,
        "skipped_quality": 0,
        "skipped_drift": 0,
        "skipped_expectancy": 0,
        "skipped_meta": 0,
        "skipped_selector": 0,
        "skipped_regime": 0,
        "skipped_shock": 0,
        "skipped_combo": 0,
        "skipped_down": 0,
        "skipped_up": 0,
        "skipped_extreme": 0,
        "skipped_direction_loss": 0,
        "skipped_drawdown_acceleration": 0,
    }
    covered_groups: set[str] = set()
    now_ts = time.time()
    newest_guard_age_s: Optional[float] = None
    newest_payload_age_s: Optional[float] = None
    stale_guard_files: List[str] = []
    stale_payload_files: List[str] = []
    hold_until_ts_max: Optional[float] = None

    for gf in guard_files:
        file_group = _guard_group_name(gf)
        if file_group:
            covered_groups.add(file_group)
        try:
            file_age_s = max(0.0, now_ts - gf.stat().st_mtime)
            if newest_guard_age_s is None or file_age_s < newest_guard_age_s:
                newest_guard_age_s = file_age_s
            if file_age_s > RUNTIME_GUARD_MAX_AGE_SEC:
                stale_guard_files.append(f"{gf.name}({file_age_s:.0f}s)")
        except OSError:
            stale_guard_files.append(f"{gf.name}(mtime读取失败)")

        try:
            payload = json.loads(gf.read_text(encoding="utf-8"))
        except Exception as e:
            result["issues"].append(f"{gf.name} 读取失败: {str(e)[:80]}")
            continue
        if not isinstance(payload, dict):
            result["issues"].append(f"{gf.name} 格式异常（非 JSON 对象）")
            continue

        dq = payload.get("data_quality", {})
        if not isinstance(dq, dict):
            dq = {}
        payload_ts = _parse_iso_ts(payload.get("timestamp"))
        checked_at_ts = _parse_iso_ts(dq.get("checkedAt"))
        hold_until_ts = _parse_iso_ts(dq.get("recoveryHoldUntil"))
        if hold_until_ts is not None:
            if hold_until_ts_max is None or hold_until_ts > hold_until_ts_max:
                hold_until_ts_max = hold_until_ts
        ts_for_age = checked_at_ts if checked_at_ts is not None else payload_ts
        if ts_for_age is not None:
            payload_age_s = max(0.0, now_ts - ts_for_age)
            if newest_payload_age_s is None or payload_age_s < newest_payload_age_s:
                newest_payload_age_s = payload_age_s
            if payload_age_s > RUNTIME_GUARD_PAYLOAD_MAX_AGE_SEC:
                stale_payload_files.append(f"{gf.name}({payload_age_s:.0f}s)")

        mode = str(dq.get("mode", "unknown"))
        severity = str(dq.get("severity", mode or "unknown"))
        trigger_class = str(dq.get("triggerClass", ""))
        penalty = _to_float(dq.get("confidencePenalty"), 0.0)
        bet_scale = _to_float(dq.get("betScale"), 1.0)
        api_error_rate = _to_float(dq.get("apiErrorRate"), 0.0)
        api_sources = dq.get("apiErrorSources") or []
        key_sources = dq.get("keySourceErrors") or []
        secondary_sources = dq.get("secondarySourceErrors") or []
        non_key_sources = dq.get("nonKeySourceErrors") or []
        observation_counts = dq.get("observationCounts") if isinstance(dq.get("observationCounts"), dict) else {}
        consecutive_bad = dq.get("consecutiveBadWindows") if isinstance(dq.get("consecutiveBadWindows"), dict) else {}
        consecutive_good = dq.get("consecutiveGoodWindows") if isinstance(dq.get("consecutiveGoodWindows"), dict) else {}
        reason = str(dq.get("reason", ""))
        stale = dq.get("staleSources") or []
        if isinstance(stale, list):
            stale_union.update(str(x) for x in stale if x is not None)
        if isinstance(api_sources, list):
            api_source_union.update(str(x) for x in api_sources if x is not None)
        if isinstance(key_sources, list):
            result["key_source_errors"] = sorted(set(result["key_source_errors"]) | {str(x) for x in key_sources if x is not None})
        if isinstance(secondary_sources, list):
            result["secondary_source_errors"] = sorted(set(result["secondary_source_errors"]) | {str(x) for x in secondary_sources if x is not None})
        if isinstance(non_key_sources, list):
            result["non_key_source_errors"] = sorted(set(result["non_key_source_errors"]) | {str(x) for x in non_key_sources if x is not None})
        if observation_counts:
            result["observation_counts"].update({str(k): int(v or 0) for k, v in observation_counts.items()})
        if consecutive_bad:
            result["consecutive_bad_windows"].update({str(k): int(v or 0) for k, v in consecutive_bad.items()})
        if consecutive_good:
            result["consecutive_good_windows"].update({str(k): int(v or 0) for k, v in consecutive_good.items()})

        guard_evidence = payload.get("guard_evidence_v1", {})
        guard_evidence_totals = guard_evidence.get("totals") if isinstance(guard_evidence, dict) else None
        if isinstance(guard_evidence_totals, dict):
            evidence_totals["evaluated"] += int(guard_evidence_totals.get("evaluated", 0) or 0)
            evidence_totals["passed"] += int(guard_evidence_totals.get("passed", 0) or 0)
            evidence_totals["skipped_should_trade"] += int(guard_evidence_totals.get("skippedShouldTrade", 0) or 0)
            evidence_totals["skipped_quality"] += int(guard_evidence_totals.get("skippedByQuality", 0) or 0)
            evidence_totals["skipped_drift"] += int(guard_evidence_totals.get("skippedByThresholdDrift", 0) or 0)
            evidence_totals["skipped_expectancy"] += int(guard_evidence_totals.get("skippedByExpectancyGate", 0) or 0)
            evidence_totals["skipped_meta"] += int(guard_evidence_totals.get("skippedByMetaLabel", 0) or 0)
            evidence_totals["skipped_selector"] += int(guard_evidence_totals.get("skippedBySelector", 0) or 0)
            evidence_totals["skipped_regime"] += int(guard_evidence_totals.get("skippedByRegime", 0) or 0)
            evidence_totals["skipped_shock"] += int(guard_evidence_totals.get("skippedByShockRisk", 0) or 0)
            evidence_totals["skipped_combo"] += int(guard_evidence_totals.get("skippedByComboPause", 0) or 0)
            evidence_totals["skipped_down"] += int(guard_evidence_totals.get("skippedByDownBreaker", 0) or 0)
            evidence_totals["skipped_up"] += int(guard_evidence_totals.get("skippedByUpRisk", 0) or 0)
            evidence_totals["skipped_extreme"] += int(guard_evidence_totals.get("skippedByExtremeMarket", 0) or 0)
            evidence_totals["skipped_direction_loss"] += int(guard_evidence_totals.get("skippedByDirectionLossCooldown", 0) or 0)
            evidence_totals["skipped_drawdown_acceleration"] += int(guard_evidence_totals.get("skippedByDrawdownAcceleration", 0) or 0)

        if (
            mode_rank.get(mode, -1) > mode_rank.get(worst_mode, -1)
            or (
                mode_rank.get(mode, -1) == mode_rank.get(worst_mode, -1)
                and bet_scale < min_bet_scale
            )
        ):
            worst_mode = mode
            result["severity"] = severity
            result["trigger_class"] = trigger_class
            worst_reason = reason
            worst_group = file_group or ""
        if penalty > max_penalty:
            max_penalty = penalty
        if bet_scale < min_bet_scale:
            min_bet_scale = max(0.0, min(1.0, bet_scale))
        if api_error_rate > max_api_error_rate:
            max_api_error_rate = api_error_rate

        breakers = payload.get("down_breakers", [])
        if not isinstance(breakers, list):
            breakers = []
        for b in breakers:
            if not isinstance(b, dict):
                continue
            name = str(b.get("name", "?"))
            group = str(b.get("group", "")).strip()
            if group:
                covered_groups.add(group)
            key = f"{group}/{name}" if group else name
            all_breaker_keys.add(key)
            if bool(b.get("active")):
                active_breaker_keys.add(key)

        breakers_v2 = payload.get("down_breakers_v2", {})
        cells = breakers_v2.get("cells") if isinstance(breakers_v2, dict) else []
        if not isinstance(cells, list):
            cells = []
        for c in cells:
            if not isinstance(c, dict):
                continue
            key = str(c.get("key") or "")
            if not key:
                # 兼容旧字段缺失，退化拼接
                profile = str(c.get("profile", ""))
                trader_name = str(c.get("traderName", ""))
                symbol = str(c.get("symbol", ""))
                mode = str(c.get("mode", ""))
                key = f"{profile}:{trader_name}:{symbol}:{mode}"
            v2_cells[key] = c

        up_breakers_v2 = payload.get("up_breakers_v2", {})
        up_cells = up_breakers_v2.get("cells") if isinstance(up_breakers_v2, dict) else []
        if not isinstance(up_cells, list):
            up_cells = []
        for c in up_cells:
            if not isinstance(c, dict):
                continue
            key = str(c.get("key") or "")
            if not key:
                profile = str(c.get("profile", ""))
                trader_name = str(c.get("traderName", ""))
                symbol = str(c.get("symbol", ""))
                mode = str(c.get("mode", ""))
                key = f"{profile}:{trader_name}:{symbol}:{mode}"
            up_v2_cells[key] = c

        calibration_v1 = payload.get("prediction_calibration_v1", {})
        cal_cells = calibration_v1.get("cells") if isinstance(calibration_v1, dict) else []
        if not isinstance(cal_cells, list):
            cal_cells = []
        for c in cal_cells:
            if not isinstance(c, dict):
                continue
            key = str(c.get("key") or "")
            if not key:
                profile = str(c.get("profile", ""))
                trader_name = str(c.get("traderName", ""))
                symbol = str(c.get("symbol", ""))
                direction = str(c.get("direction", ""))
                mode = str(c.get("mode", ""))
                key = f"{profile}:{trader_name}:{symbol}:{direction}:{mode}"
            calibration_cells[key] = c

        expectancy_v1 = payload.get("expectancy_gates_v1", {})
        exp_cells = expectancy_v1.get("cells") if isinstance(expectancy_v1, dict) else []
        if not isinstance(exp_cells, list):
            exp_cells = []
        for c in exp_cells:
            if not isinstance(c, dict):
                continue
            key = str(c.get("key") or "")
            if not key:
                profile = str(c.get("profile", ""))
                trader_name = str(c.get("traderName", ""))
                symbol = str(c.get("symbol", ""))
                direction = str(c.get("direction", ""))
                mode = str(c.get("mode", ""))
                key = f"{profile}:{trader_name}:{symbol}:{direction}:{mode}"
            expectancy_cells[key] = c

        drift_v1 = payload.get("threshold_drifts_v1", {})
        drift_raw = drift_v1.get("cells") if isinstance(drift_v1, dict) else []
        if not isinstance(drift_raw, list):
            drift_raw = []
        for c in drift_raw:
            if not isinstance(c, dict):
                continue
            key = str(c.get("key") or "")
            if not key:
                profile = str(c.get("profile", ""))
                trader_name = str(c.get("traderName", ""))
                symbol = str(c.get("symbol", ""))
                direction = str(c.get("direction", ""))
                mode = str(c.get("mode", ""))
                key = f"{profile}:{trader_name}:{symbol}:{direction}:{mode}"
            drift_cells[key] = c

        meta_v1 = payload.get("meta_labels_v1", {})
        meta_raw = meta_v1.get("cells") if isinstance(meta_v1, dict) else []
        if not isinstance(meta_raw, list):
            meta_raw = []
        for c in meta_raw:
            if not isinstance(c, dict):
                continue
            key = str(c.get("key") or "")
            if not key:
                profile = str(c.get("profile", ""))
                trader_name = str(c.get("traderName", ""))
                symbol = str(c.get("symbol", ""))
                direction = str(c.get("direction", ""))
                mode = str(c.get("mode", ""))
                key = f"{profile}:{trader_name}:{symbol}:{direction}:{mode}"
            meta_cells[key] = c

        selector_v1 = payload.get("selector_overlays_v1", {})
        selector_raw = selector_v1.get("cells") if isinstance(selector_v1, dict) else []
        if not isinstance(selector_raw, list):
            selector_raw = []
        for c in selector_raw:
            if not isinstance(c, dict):
                continue
            key = str(c.get("key") or "")
            if not key:
                profile = str(c.get("profile", ""))
                trader_name = str(c.get("traderName", ""))
                symbol = str(c.get("symbol", ""))
                mode = str(c.get("mode", ""))
                key = f"{profile}:{trader_name}:{symbol}:{mode}"
            selector_cells[key] = c

        shock_breakers_v2 = payload.get("shock_breakers_v2", {})
        shock_cells = shock_breakers_v2.get("cells") if isinstance(shock_breakers_v2, dict) else []
        if not isinstance(shock_cells, list):
            shock_cells = []
        for c in shock_cells:
            if not isinstance(c, dict):
                continue
            key = str(c.get("key") or "")
            if not key:
                profile = str(c.get("profile", ""))
                trader_name = str(c.get("traderName", ""))
                symbol = str(c.get("symbol", ""))
                direction = str(c.get("direction", ""))
                mode = str(c.get("mode", ""))
                key = f"{profile}:{trader_name}:{symbol}:{direction}:{mode}"
            shock_v2_cells[key] = c

        combo_pauses_v1 = payload.get("combo_pauses_v1", {})
        combo_cells = combo_pauses_v1.get("cells") if isinstance(combo_pauses_v1, dict) else []
        if not isinstance(combo_cells, list):
            combo_cells = []
        for c in combo_cells:
            if not isinstance(c, dict):
                continue
            key = str(c.get("key") or "")
            if not key:
                profile = str(c.get("profile", ""))
                trader_name = str(c.get("traderName", ""))
                symbol = str(c.get("symbol", ""))
                mode = str(c.get("mode", ""))
                key = f"{profile}:{trader_name}:{symbol}:{mode}"
            combo_pause_cells[key] = c

        cooldown_v2 = payload.get("cooldown_v2", {})
        cd_cells = cooldown_v2.get("cells") if isinstance(cooldown_v2, dict) else []
        if not isinstance(cd_cells, list):
            cd_cells = []
        for c in cd_cells:
            if not isinstance(c, dict):
                continue
            trader_name = str(c.get("traderName", ""))
            key = str(c.get("key") or "")
            if not key and trader_name:
                symbol = str(c.get("symbol", ""))
                direction = str(c.get("direction", ""))
                key = f"{trader_name}:{symbol}:{direction}"
            if not key:
                continue
            cooldown_cells[key] = c

        mainline_extreme_v1 = payload.get("mainline_extreme_v1", {})
        extreme_cells = mainline_extreme_v1.get("cells") if isinstance(mainline_extreme_v1, dict) else []
        if not isinstance(extreme_cells, list):
            extreme_cells = []
        for c in extreme_cells:
            if not isinstance(c, dict):
                continue
            key = str(c.get("key") or "")
            if not key:
                profile = str(c.get("profile", ""))
                trader_name = str(c.get("traderName", ""))
                symbol = str(c.get("symbol", ""))
                key = f"{profile}:{trader_name}:{symbol}"
            if not key:
                continue
            mainline_extreme_cells[key] = c

        direction_loss_v1 = payload.get("direction_loss_cooldown_v1", {})
        direction_loss_cells_raw = direction_loss_v1.get("cells") if isinstance(direction_loss_v1, dict) else []
        if not isinstance(direction_loss_cells_raw, list):
            direction_loss_cells_raw = []
        for c in direction_loss_cells_raw:
            if not isinstance(c, dict):
                continue
            key = str(c.get("key") or "")
            if not key:
                profile = str(c.get("profile", ""))
                trader_name = str(c.get("traderName", ""))
                symbol = str(c.get("symbol", ""))
                direction = str(c.get("direction", ""))
                key = f"{profile}:{trader_name}:{symbol}:{direction}"
            if not key:
                continue
            direction_loss_cooldown_cells[key] = c

        drawdown_acceleration_v1 = payload.get("drawdown_acceleration_v1", {})
        drawdown_acc_cells = drawdown_acceleration_v1.get("cells") if isinstance(drawdown_acceleration_v1, dict) else []
        if not isinstance(drawdown_acc_cells, list):
            drawdown_acc_cells = []
        for c in drawdown_acc_cells:
            if not isinstance(c, dict):
                continue
            key = str(c.get("key") or "")
            if not key:
                profile = str(c.get("profile", ""))
                trader_name = str(c.get("traderName", ""))
                symbol = str(c.get("symbol", ""))
                key = f"{profile}:{trader_name}:{symbol}"
            if not key:
                continue
            drawdown_acceleration_cells[key] = c

    if not calibration_cells:
        calibration_state_files = sorted((POLYMARKET_DIR / "logs" / "runtime").glob("prediction_calibration_v1*.json"))
        for path in calibration_state_files:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            cells = payload.get("cells")
            if not isinstance(cells, list):
                continue
            for item in cells:
                if not isinstance(item, dict):
                    continue
                profile = str(item.get("profile") or "").strip()
                trader_name = str(item.get("traderName") or "").strip()
                symbol = str(item.get("symbol") or "").strip().upper()
                direction = str(item.get("direction") or "").strip().upper()
                if symbol not in set(_ordered_active_symbols()) or direction not in {"UP", "DOWN"} or not trader_name:
                    continue
                key = str(item.get("key") or f"{profile}:{trader_name}:{symbol}:{direction}:simulation")
                calibration_cells[key] = item

    if not expectancy_cells:
        expectancy_state_files = sorted((POLYMARKET_DIR / "logs" / "runtime").glob("expectancy_gates_v1*.json"))
        for path in expectancy_state_files:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            cells = payload.get("cells")
            if not isinstance(cells, list):
                continue
            for item in cells:
                if not isinstance(item, dict):
                    continue
                profile = str(item.get("profile") or "").strip()
                trader_name = str(item.get("traderName") or "").strip()
                symbol = str(item.get("symbol") or "").strip().upper()
                direction = str(item.get("direction") or "").strip().upper()
                if symbol not in set(_ordered_active_symbols()) or direction not in {"UP", "DOWN"} or not trader_name:
                    continue
                key = str(item.get("key") or f"{profile}:{trader_name}:{symbol}:{direction}:simulation")
                expectancy_cells[key] = item

    result["mode"] = worst_mode
    result["penalty"] = max_penalty
    result["bet_scale"] = min_bet_scale
    result["api_error_rate"] = max_api_error_rate
    result["api_error_sources"] = sorted(api_source_union)
    result["reason"] = worst_reason
    result["stale_sources"] = sorted(stale_union)
    result["total_breakers"] = len(all_breaker_keys)
    result["active_breakers"] = len(active_breaker_keys)
    result["active_names"] = sorted(active_breaker_keys)[:8]
    v2_values = list(v2_cells.values())
    result["v2_total_cells"] = len(v2_values)
    result["v2_enforce_cells"] = sum(
        1 for c in v2_values if str(c.get("effectiveRiskMode") or c.get("riskMode")) == "enforce"
    )
    result["v2_shadow_cells"] = sum(
        1 for c in v2_values if str(c.get("effectiveRiskMode") or c.get("riskMode")) == "shadow"
    )
    result["v2_blocked_cells"] = sum(1 for c in v2_values if bool(c.get("blocked")))
    result["v2_hard_cells"] = sum(1 for c in v2_values if str(c.get("tier")) == "hard")
    result["v2_soft_cells"] = sum(1 for c in v2_values if str(c.get("tier")) == "soft")
    result["down_fast_hard_count"] = sum(
        1
        for c in v2_values
        if str(c.get("tier") or "").strip().lower() == "hard"
        and str(c.get("triggerLane") or "").strip().lower() == "fast"
    )
    result["cross_layer_escalation_count"] = sum(
        1
        for c in v2_values
        if str(c.get("triggerLane") or "").strip().lower() == "cross_layer"
    )
    result["stale_last_action_count"] = sum(
        1
        for c in v2_values
        if str(c.get("reasonCode") or "").strip().lower() == "normal"
        and str(c.get("lastAction") or "").strip().lower() not in {"", "normal", "off", "init"}
    )
    up_v2_values = list(up_v2_cells.values())
    result["up_v2_total_cells"] = len(up_v2_values)
    result["up_v2_enforce_cells"] = sum(
        1 for c in up_v2_values if str(c.get("effectiveRiskMode") or c.get("riskMode")) == "enforce"
    )
    result["up_v2_shadow_cells"] = sum(
        1 for c in up_v2_values if str(c.get("effectiveRiskMode") or c.get("riskMode")) == "shadow"
    )
    result["up_v2_blocked_cells"] = sum(1 for c in up_v2_values if bool(c.get("blocked")))
    result["up_v2_soft_cells"] = sum(1 for c in up_v2_values if str(c.get("tier")) == "soft")
    calibration_values = list(calibration_cells.values())
    result["calibration_total_cells"] = len(calibration_values)
    result["calibration_enforce_cells"] = sum(
        1 for c in calibration_values if str(c.get("effectiveMode") or c.get("mode")) == "enforce"
    )
    result["calibration_shadow_cells"] = sum(
        1 for c in calibration_values if str(c.get("effectiveMode") or c.get("mode")) == "shadow"
    )
    result["calibration_active_cells"] = sum(1 for c in calibration_values if bool(c.get("active")))
    expectancy_values = list(expectancy_cells.values())
    result["expectancy_total_cells"] = len(expectancy_values)
    result["expectancy_enforce_cells"] = sum(
        1 for c in expectancy_values if str(c.get("effectiveMode") or c.get("mode")) == "enforce"
    )
    result["expectancy_shadow_cells"] = sum(
        1 for c in expectancy_values if str(c.get("effectiveMode") or c.get("mode")) == "shadow"
    )
    result["expectancy_blocked_cells"] = sum(1 for c in expectancy_values if bool(c.get("blocked")))
    result["expectancy_degraded_cells"] = sum(1 for c in expectancy_values if str(c.get("status", "")).lower() == "degraded")
    result["expect_fast_block_count"] = sum(
        1
        for c in expectancy_values
        if bool(c.get("blocked")) and str(c.get("triggerLane") or "").strip().lower() == "fast"
    )
    result["expect_manual_block_count"] = sum(
        1
        for c in expectancy_values
        if str(c.get("reasonCode") or "").strip().lower() == "manual_block"
    )
    result["expect_manual_degrade_count"] = sum(
        1
        for c in expectancy_values
        if str(c.get("reasonCode") or "").strip().lower() == "manual_degrade"
    )
    manual_block_by_symdir: Dict[str, int] = {}
    manual_degrade_by_symdir: Dict[str, int] = {}
    manual_hits: List[Dict[str, Any]] = []
    for c in expectancy_values:
        reason = str(c.get("reasonCode") or "").strip().lower()
        if reason not in {"manual_block", "manual_degrade"}:
            continue
        profile = str(c.get("profile") or "").strip() or "unknown"
        symbol = str(c.get("symbol") or "").strip().upper() or "UNKNOWN"
        direction = str(c.get("direction") or "").strip().upper() or "UNKNOWN"
        symdir_key = f"{profile}/{symbol}_{direction}"
        if reason == "manual_block":
            manual_block_by_symdir[symdir_key] = int(manual_block_by_symdir.get(symdir_key, 0)) + 1
        else:
            manual_degrade_by_symdir[symdir_key] = int(manual_degrade_by_symdir.get(symdir_key, 0)) + 1
        manual_hits.append(
            {
                "profile": profile,
                "trader": str(c.get("traderName") or ""),
                "symbol": symbol,
                "direction": direction,
                "action": "BLOCK" if reason == "manual_block" else "DEGRADE",
                "layer": "expectancy",
            }
        )
    result["manual_block_count_by_symbol_direction"] = dict(sorted(manual_block_by_symdir.items()))
    result["manual_degrade_count_by_symbol_direction"] = dict(sorted(manual_degrade_by_symdir.items()))
    result["manual_action_runtime_hits"] = manual_hits[:30]
    drift_values = list(drift_cells.values())
    result["drift_total_cells"] = len(drift_values)
    result["drift_enforce_cells"] = sum(
        1 for c in drift_values if str(c.get("effectiveMode") or c.get("mode")) == "enforce"
    )
    result["drift_shadow_cells"] = sum(
        1 for c in drift_values if str(c.get("effectiveMode") or c.get("mode")) == "shadow"
    )
    result["drift_active_cells"] = sum(1 for c in drift_values if bool(c.get("active")))
    result["drift_skip_cells"] = sum(1 for c in drift_values if str(c.get("status", "")).lower() == "drifted")
    result["drift_nonzero_delta_cells"] = sum(
        1
        for c in drift_values
        if abs(_to_float(c.get("effectiveThresholdDelta", c.get("thresholdDelta", 0.0)), 0.0)) > 1e-9
    )
    meta_values = list(meta_cells.values())
    result["meta_total_cells"] = len(meta_values)
    result["meta_enforce_cells"] = sum(1 for c in meta_values if str(c.get("mode", "")).strip().lower() == "enforce")
    result["meta_shadow_cells"] = sum(1 for c in meta_values if str(c.get("mode", "")).strip().lower() == "shadow")
    result["meta_take_false_cells"] = sum(
        1
        for c in meta_values
        if isinstance(c, dict) and ("takeTrade" in c) and (not bool(c.get("takeTrade", True)))
    )
    result["meta_runtime_observable"] = any(
        isinstance(c, dict) and ("takeTrade" in c or "metaConfidence" in c or "reasonCode" in c)
        for c in meta_values
    )
    selector_values = list(selector_cells.values())
    result["selector_total_cells"] = len(selector_values)
    result["selector_enforce_cells"] = sum(
        1 for c in selector_values if str(c.get("effectiveMode") or c.get("mode")) == "enforce"
    )
    result["selector_shadow_cells"] = sum(
        1 for c in selector_values if str(c.get("effectiveMode") or c.get("mode")) == "shadow"
    )
    result["selector_ineligible_cells"] = sum(1 for c in selector_values if not bool(c.get("eligible", True)))
    shock_v2_values = list(shock_v2_cells.values())
    result["shock_v2_total_cells"] = len(shock_v2_values)
    result["shock_v2_enforce_cells"] = sum(
        1 for c in shock_v2_values if str(c.get("effectiveRiskMode") or c.get("riskMode")) == "enforce"
    )
    result["shock_v2_shadow_cells"] = sum(
        1 for c in shock_v2_values if str(c.get("effectiveRiskMode") or c.get("riskMode")) == "shadow"
    )
    result["shock_v2_active_cells"] = sum(1 for c in shock_v2_values if bool(c.get("active")))
    result["shock_v2_blocked_cells"] = sum(1 for c in shock_v2_values if bool(c.get("blocked")))
    combo_values = list(combo_pause_cells.values())
    result["combo_pause_total_cells"] = len(combo_values)
    result["combo_pause_enforce_cells"] = sum(
        1 for c in combo_values if str(c.get("effectiveRiskMode") or c.get("riskMode")) == "enforce"
    )
    result["combo_pause_shadow_cells"] = sum(
        1 for c in combo_values if str(c.get("effectiveRiskMode") or c.get("riskMode")) == "shadow"
    )
    result["combo_pause_active_cells"] = sum(1 for c in combo_values if bool(c.get("active")))
    result["combo_pause_blocked_cells"] = sum(1 for c in combo_values if bool(c.get("blocked")))
    combo_directional_values = [
        c for c in combo_values
        if str(c.get("direction") or "BOTH").strip().upper() != "BOTH"
    ]
    result["combo_directional_total"] = len(combo_directional_values)
    result["combo_directional_active"] = sum(1 for c in combo_directional_values if bool(c.get("active")))
    result["combo_directional_blocked"] = sum(1 for c in combo_directional_values if bool(c.get("blocked")))

    cluster_keys = _runtime_cluster_keys()
    down_cluster = {k: {"cells": 0, "enforce": 0, "shadow": 0, "blocked": 0, "soft": 0, "hard": 0} for k in cluster_keys}
    up_cluster = {k: {"cells": 0, "enforce": 0, "shadow": 0, "blocked": 0, "soft": 0} for k in cluster_keys}
    calibration_cluster = {k: {"cells": 0, "enforce": 0, "shadow": 0, "active": 0} for k in cluster_keys}
    expectancy_cluster = {k: {"cells": 0, "enforce": 0, "shadow": 0, "blocked": 0, "degraded": 0} for k in cluster_keys}
    drift_cluster = {k: {"cells": 0, "enforce": 0, "shadow": 0, "active": 0, "drifted": 0} for k in cluster_keys}
    meta_cluster = {k: {"cells": 0, "enforce": 0, "shadow": 0} for k in cluster_keys}
    selector_cluster = {k: {"cells": 0, "enforce": 0, "shadow": 0, "ineligible": 0} for k in cluster_keys}
    shock_cluster = {k: {"cells": 0, "enforce": 0, "shadow": 0, "blocked": 0, "active": 0} for k in cluster_keys}
    combo_cluster = {k: {"cells": 0, "enforce": 0, "shadow": 0, "blocked": 0, "active": 0} for k in cluster_keys}
    shock_blocked_eta_by_cluster: Dict[str, List[float]] = {k: [] for k in cluster_keys}
    shock_blocked_with_eta = 0
    shock_blocked_without_eta = 0
    now_ts = time.time()

    def _cluster_key_of(cell: dict) -> Optional[str]:
        profile = str(cell.get("profile") or "").strip()
        symbol = str(cell.get("symbol") or "").strip().upper()
        if symbol not in set(_ordered_active_symbols()):
            return None
        profile_norm = "70" if profile == "70" else "default"
        return f"{profile_norm}_{symbol}"

    def _effective_mode(cell: dict) -> str:
        return str(cell.get("effectiveRiskMode") or cell.get("riskMode") or "").strip().lower()

    for c in v2_values:
        ck = _cluster_key_of(c)
        if not ck:
            continue
        bucket = down_cluster[ck]
        bucket["cells"] += 1
        mode = _effective_mode(c)
        if mode == "enforce":
            bucket["enforce"] += 1
        elif mode == "shadow":
            bucket["shadow"] += 1
        if bool(c.get("blocked")):
            bucket["blocked"] += 1
        tier = str(c.get("tier") or "").lower()
        if tier == "soft":
            bucket["soft"] += 1
        elif tier == "hard":
            bucket["hard"] += 1

    for c in up_v2_values:
        ck = _cluster_key_of(c)
        if not ck:
            continue
        bucket = up_cluster[ck]
        bucket["cells"] += 1
        mode = _effective_mode(c)
        if mode == "enforce":
            bucket["enforce"] += 1
        elif mode == "shadow":
            bucket["shadow"] += 1
        if bool(c.get("blocked")):
            bucket["blocked"] += 1
        if str(c.get("tier") or "").lower() == "soft":
            bucket["soft"] += 1

    for c in calibration_values:
        ck = _cluster_key_of(c)
        if not ck:
            continue
        bucket = calibration_cluster[ck]
        bucket["cells"] += 1
        mode = str(c.get("effectiveMode") or c.get("mode") or "").strip().lower()
        if mode == "enforce":
            bucket["enforce"] += 1
        elif mode == "shadow":
            bucket["shadow"] += 1
        if bool(c.get("active")):
            bucket["active"] += 1

    for c in expectancy_values:
        ck = _cluster_key_of(c)
        if not ck:
            continue
        bucket = expectancy_cluster[ck]
        bucket["cells"] += 1
        mode = str(c.get("effectiveMode") or c.get("mode") or "").strip().lower()
        if mode == "enforce":
            bucket["enforce"] += 1
        elif mode == "shadow":
            bucket["shadow"] += 1
        if bool(c.get("blocked")):
            bucket["blocked"] += 1
        if str(c.get("status", "")).lower() == "degraded":
            bucket["degraded"] += 1

    for c in drift_values:
        ck = _cluster_key_of(c)
        if not ck:
            continue
        bucket = drift_cluster[ck]
        bucket["cells"] += 1
        mode = str(c.get("effectiveMode") or c.get("mode") or "").strip().lower()
        if mode == "enforce":
            bucket["enforce"] += 1
        elif mode == "shadow":
            bucket["shadow"] += 1
        if bool(c.get("active")):
            bucket["active"] += 1
        if str(c.get("status", "")).lower() == "drifted":
            bucket["drifted"] += 1

    for c in meta_values:
        ck = _cluster_key_of(c)
        if not ck:
            continue
        bucket = meta_cluster[ck]
        bucket["cells"] += 1
        mode = str(c.get("mode") or "").strip().lower()
        if mode == "enforce":
            bucket["enforce"] += 1
        elif mode == "shadow":
            bucket["shadow"] += 1

    for c in selector_values:
        ck = _cluster_key_of(c)
        if not ck:
            continue
        bucket = selector_cluster[ck]
        bucket["cells"] += 1
        mode = str(c.get("effectiveMode") or c.get("mode") or "").strip().lower()
        if mode == "enforce":
            bucket["enforce"] += 1
        elif mode == "shadow":
            bucket["shadow"] += 1
        if not bool(c.get("eligible", True)):
            bucket["ineligible"] += 1

    for c in shock_v2_values:
        ck = _cluster_key_of(c)
        if not ck:
            continue
        bucket = shock_cluster[ck]
        bucket["cells"] += 1
        mode = _effective_mode(c)
        if mode == "enforce":
            bucket["enforce"] += 1
        elif mode == "shadow":
            bucket["shadow"] += 1
        if bool(c.get("blocked")):
            bucket["blocked"] += 1
            eta_ts = _parse_iso_ts(str(c.get("activeUntil") or ""))
            if eta_ts is None:
                raw_eta = c.get("activeUntilTs")
                try:
                    eta_val = float(raw_eta)
                except (TypeError, ValueError):
                    eta_val = 0.0
                if eta_val > 0:
                    if eta_val > 1_000_000_000_000:
                        eta_val /= 1000.0
                    eta_ts = eta_val
            if eta_ts and eta_ts > 0:
                shock_blocked_with_eta += 1
                remain_min = max(0.0, (float(eta_ts) - now_ts) / 60.0)
                shock_blocked_eta_by_cluster[ck].append(remain_min)
            else:
                shock_blocked_without_eta += 1
        if bool(c.get("active")):
            bucket["active"] += 1

    for c in combo_values:
        ck = _cluster_key_of(c)
        if not ck:
            continue
        bucket = combo_cluster[ck]
        bucket["cells"] += 1
        mode = _effective_mode(c)
        if mode == "enforce":
            bucket["enforce"] += 1
        elif mode == "shadow":
            bucket["shadow"] += 1
        if bool(c.get("blocked")):
            bucket["blocked"] += 1
        if bool(c.get("active")):
            bucket["active"] += 1

    result["cluster_breakdown"] = {
        "down": down_cluster,
        "up": up_cluster,
        "calibration": calibration_cluster,
        "expectancy": expectancy_cluster,
        "drift": drift_cluster,
        "meta": meta_cluster,
        "selector": selector_cluster,
        "shock": shock_cluster,
        "combo": combo_cluster,
    }
    result["shock_v2_blocked_with_eta"] = int(shock_blocked_with_eta)
    result["shock_v2_blocked_without_eta"] = int(shock_blocked_without_eta)
    result["shock_v2_blocked_eta_by_cluster"] = {
        ck: {
            "count": int(len(vals)),
            "min_min": round(min(vals), 1),
            "max_min": round(max(vals), 1),
        }
        for ck, vals in shock_blocked_eta_by_cluster.items()
        if vals
    }
    cd_values = list(cooldown_cells.values())
    result["cooldown_total_active"] = len(cd_values)
    result["cooldown_symbol_active"] = sum(1 for c in cd_values if str(c.get("scope", "")).strip() == "symbol")
    result["cooldown_symbol_direction_active"] = sum(
        1 for c in cd_values if str(c.get("scope", "")).strip() == "symbol_direction"
    )
    if cd_values:
        if result["cooldown_symbol_direction_active"] > 0 and result["cooldown_symbol_active"] == 0:
            result["cooldown_scope"] = "symbol_direction"
        elif result["cooldown_symbol_active"] > 0 and result["cooldown_symbol_direction_active"] == 0:
            result["cooldown_scope"] = "symbol"
        else:
            result["cooldown_scope"] = "mixed"

    extreme_values = list(mainline_extreme_cells.values())
    result["mainline_extreme_total_symbols"] = len(extreme_values)
    result["mainline_extreme_active_symbols"] = sum(1 for c in extreme_values if bool(c.get("active")))
    direction_loss_values = list(direction_loss_cooldown_cells.values())
    result["direction_loss_cooldown_total"] = len(direction_loss_values)
    result["direction_loss_cooldown_blocked"] = sum(1 for c in direction_loss_values if bool(c.get("blocked")))
    drawdown_acceleration_values = list(drawdown_acceleration_cells.values())
    result["drawdown_acceleration_total"] = len(drawdown_acceleration_values)
    result["drawdown_acceleration_blocked"] = sum(1 for c in drawdown_acceleration_values if bool(c.get("blocked")))

    result["evidence_evaluated"] = int(evidence_totals["evaluated"])
    result["evidence_passed"] = int(evidence_totals["passed"])
    result["evidence_skipped_should_trade"] = int(evidence_totals["skipped_should_trade"])
    result["evidence_skipped_quality"] = int(evidence_totals["skipped_quality"])
    result["evidence_skipped_drift"] = int(evidence_totals["skipped_drift"])
    result["evidence_skipped_expectancy"] = int(evidence_totals["skipped_expectancy"])
    result["evidence_skipped_meta"] = int(evidence_totals["skipped_meta"])
    result["evidence_skipped_selector"] = int(evidence_totals["skipped_selector"])
    result["evidence_skipped_regime"] = int(evidence_totals["skipped_regime"])
    result["evidence_skipped_shock"] = int(evidence_totals["skipped_shock"])
    result["evidence_skipped_combo"] = int(evidence_totals["skipped_combo"])
    result["evidence_skipped_down"] = int(evidence_totals["skipped_down"])
    result["evidence_skipped_up"] = int(evidence_totals["skipped_up"])
    result["evidence_skipped_extreme"] = int(evidence_totals["skipped_extreme"])
    result["evidence_skipped_direction_loss"] = int(evidence_totals["skipped_direction_loss"])
    result["evidence_skipped_drawdown_acceleration"] = int(evidence_totals["skipped_drawdown_acceleration"])
    result["evidence_dummy_eval_ready"] = result["evidence_evaluated"] >= RUNTIME_GUARD_EVIDENCE_MIN_EVAL

    calibration_effective = sum(
        1
        for c in calibration_values
        if str(c.get("effectiveMode") or c.get("mode") or "").strip().lower() == "enforce"
        and bool(c.get("active"))
        and isinstance(c.get("calibratedConfidence"), (int, float))
        and isinstance(c.get("lastConfidence"), (int, float))
        and abs(float(c.get("calibratedConfidence")) - float(c.get("lastConfidence"))) > 1e-9
    )
    meta_eval_effective = sum(
        1
        for c in meta_values
        if str(c.get("effectiveMode") or c.get("mode") or "").strip().lower() == "enforce"
        and int(c.get("evaluated", 0) or 0) > 0
    )
    result["layer_evidence_counts"] = {
        "calibration": int(calibration_effective),
        "threshold_drift": int(max(result["evidence_skipped_drift"], result["drift_skip_cells"], result["drift_active_cells"])),
        "expectancy": int(max(
            result["evidence_skipped_expectancy"],
            result["expectancy_blocked_cells"] + result["expectancy_degraded_cells"],
        )),
        "meta_label": int(max(result["evidence_skipped_meta"], result["meta_take_false_cells"], meta_eval_effective)),
        "shock": int(max(
            result["evidence_skipped_shock"],
            result["shock_v2_active_cells"] + result["shock_v2_blocked_cells"],
        )),
        "combo_pause": int(max(
            result["evidence_skipped_combo"],
            result["combo_pause_active_cells"] + result["combo_pause_blocked_cells"],
        )),
        "up_down_risk": int(result["evidence_skipped_down"] + result["evidence_skipped_up"]),
        "selector": int(max(result["evidence_skipped_selector"], result["selector_ineligible_cells"])),
        "regime": int(max(result["evidence_skipped_regime"], result["regime_gate_active_symbols"])),
        "cooldown": int(result["cooldown_total_active"]),
        "extreme_market": int(max(result["evidence_skipped_extreme"], result["mainline_extreme_active_symbols"])),
        "direction_loss_cooldown": int(max(result["evidence_skipped_direction_loss"], result["direction_loss_cooldown_blocked"])),
        "drawdown_acceleration": int(max(result["evidence_skipped_drawdown_acceleration"], result["drawdown_acceleration_blocked"])),
    }

    recent_err = _scan_recent_transport_errors(hours=24)
    result["recent_json_parse_24h"] = int(recent_err.get("json_parse", 0))
    result["recent_json_unexpected_end_24h"] = int(recent_err.get("json_unexpected_end", 0))
    result["recent_clob_400_24h"] = int(recent_err.get("clob_400", 0))
    result["recent_clob_502_24h"] = int(recent_err.get("clob_502", 0))
    result["covered_groups"] = sorted(covered_groups)
    result["worst_group"] = worst_group
    result["newest_guard_age_s"] = round(newest_guard_age_s, 1) if newest_guard_age_s is not None else None
    result["newest_payload_age_s"] = round(newest_payload_age_s, 1) if newest_payload_age_s is not None else None
    if hold_until_ts_max is not None:
        remaining_min = max(0.0, (hold_until_ts_max - now_ts) / 60.0)
        result["hold_until"] = datetime.fromtimestamp(hold_until_ts_max, tz=timezone.utc).isoformat()
        result["hold_remaining_min"] = round(remaining_min, 1)

    dq_audit = _read_report_json(RUNTIME_DATA_QUALITY_GATE_AUDIT)
    dq_summary = dq_audit.get("summary") if isinstance(dq_audit.get("summary"), dict) else {}
    dq_rows = dq_audit.get("rows") if isinstance(dq_audit.get("rows"), list) else []
    dq_row = next(
        (
            row for row in dq_rows
            if isinstance(row, dict) and str(row.get("group") or "").strip() == worst_group
        ),
        {},
    )
    false_positive_risk = bool(dq_row.get("false_positive_risk")) if isinstance(dq_row, dict) else False
    result["runtime_data_quality_audit_status"] = str(dq_audit.get("status") or "unknown")
    result["runtime_data_quality_audit_open_p1"] = int(dq_summary.get("open_p1") or 0)
    result["runtime_data_quality_false_positive_risk"] = false_positive_risk
    result["observation_only_data_quality"] = bool(
        result["mode"] == "hard_degraded"
        and result["runtime_data_quality_audit_status"] == "ok"
        and result["runtime_data_quality_audit_open_p1"] == 0
        and not false_positive_risk
        and not result["key_source_errors"]
        and not result["non_key_source_errors"]
        and not result["stale_sources"]
        and result["api_error_rate"] < 0.20
    )

    ensemble_state_files = sorted((POLYMARKET_DIR / "logs" / "runtime").glob("ensemble_consensus_v1*.json"))
    for path in ensemble_state_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        symbols = payload.get("symbols")
        if not isinstance(symbols, list):
            continue
        for item in symbols:
            if not isinstance(item, dict):
                continue
            profile = str(item.get("profile") or "default").strip()
            symbol = str(item.get("symbol") or "").strip().upper()
            if symbol not in set(_ordered_active_symbols()):
                continue
            key = f"{profile}_{symbol}"
            ensemble_consensus_symbols[key] = item
    ensemble_values = list(ensemble_consensus_symbols.values())
    result["ensemble_consensus_mode_map"] = {
        key: str(item.get("mode", "")).strip().lower()
        for key, item in ensemble_consensus_symbols.items()
    }
    result["ensemble_consensus_total_symbols"] = len(ensemble_values)
    result["ensemble_consensus_enforce_symbols"] = sum(
        1 for c in ensemble_values if str(c.get("mode", "")).strip().lower() == "enforce"
    )
    result["ensemble_consensus_shadow_symbols"] = sum(
        1 for c in ensemble_values if str(c.get("mode", "")).strip().lower() == "shadow"
    )
    result["ensemble_consensus_blocked_symbols"] = sum(1 for c in ensemble_values if bool(c.get("consensusBlocked")))
    result["ensemble_consensus_blocked_effective_symbols"] = sum(
        1
        for c in ensemble_values
        if bool(c.get("consensusBlocked"))
        and str(c.get("mode", "")).strip().lower() == "enforce"
    )

    regime_state_files = sorted((POLYMARKET_DIR / "logs" / "runtime").glob("regime_gates_v1*.json"))
    for path in regime_state_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        symbols = payload.get("symbols")
        if not isinstance(symbols, list):
            continue
        for item in symbols:
            if not isinstance(item, dict):
                continue
            profile = str(item.get("profile") or "default").strip()
            symbol = str(item.get("symbol") or "").strip().upper()
            if symbol not in set(_ordered_active_symbols()):
                continue
            key = f"{profile}_{symbol}"
            regime_gate_symbols[key] = item
    regime_values = list(regime_gate_symbols.values())
    result["regime_gate_total_symbols"] = len(regime_values)
    result["regime_gate_enforce_symbols"] = sum(
        1 for c in regime_values if str(c.get("effectiveMode") or c.get("mode") or "").strip().lower() == "enforce"
    )
    result["regime_gate_shadow_symbols"] = sum(
        1 for c in regime_values if str(c.get("effectiveMode") or c.get("mode") or "").strip().lower() == "shadow"
    )
    result["regime_gate_active_symbols"] = sum(
        1
        for c in regime_values
        if bool(c.get("active")) or str(c.get("directionPolicy") or "").strip().upper() not in {"", "BOTH", "NONE"}
    )

    # model_review 已停用；保留字段但固定为0，避免旧runtime文件干扰健康判断
    result["model_review_total_cells"] = 0
    result["model_review_attention_cells"] = 0
    result["model_review_demote_cells"] = 0

    if len(running_groups) > 1 and len(guard_files) <= 1:
        result["issues"].append(
            f"仅检测到 {len(guard_files)} 个 runtime_guard 文件，但有 {len(running_groups)} 个合并进程在跑（状态可能被覆盖）"
        )
    # 运行中的合并进程与 runtime_guard 文件的粒度并不完全一致。
    # 这里保留统计，不再把它直接升级成异常，避免把“按设计汇总”误报成“未接线”。
    if stale_guard_files:
        show = ", ".join(stale_guard_files[:4])
        suffix = "..." if len(stale_guard_files) > 4 else ""
        result["issues"].append(f"runtime_guard 文件过期(>{RUNTIME_GUARD_MAX_AGE_SEC}s): {show}{suffix}")
    if stale_payload_files:
        show = ", ".join(stale_payload_files[:4])
        suffix = "..." if len(stale_payload_files) > 4 else ""
        result["issues"].append(
            f"runtime_guard payload 过期(>{RUNTIME_GUARD_PAYLOAD_MAX_AGE_SEC}s): {show}{suffix}"
        )

    if result["mode"] == "halt":
        if result["hold_remaining_min"] > 0:
            result["issues"].append(
                f"数据质量门控=halt(恢复期剩余{result['hold_remaining_min']:.1f}m): {result['reason']}"
            )
        else:
            result["issues"].append(f"数据质量门控=halt: {result['reason']}")
    elif result["mode"] == "hard_degraded" and not result["observation_only_data_quality"]:
        result["issues"].append(
            f"数据质量门控=hard_degraded: penalty={result['penalty']:.3f}, bet_scale={result['bet_scale']:.2f}, "
            f"trigger={result.get('trigger_class') or '-'}, key={','.join(result['key_source_errors'][:4]) or '-'}"
        )
    elif result["mode"] == "soft_degraded":
        result["issues"].append(
            f"数据质量门控=soft_degraded(保护性缩仓/观察): penalty={result['penalty']:.3f}, "
            f"bet_scale={result['bet_scale']:.2f}, trigger={result.get('trigger_class') or '-'}, "
            f"key={','.join(result['key_source_errors'][:3]) or '-'}, non_key={','.join(result['non_key_source_errors'][:3]) or '-'}"
        )
    if result["api_error_rate"] >= 0.20:
        result["issues"].append(
            f"外部API错误率偏高: {result['api_error_rate']:.1%} (sources={','.join(result['api_error_sources'][:4]) or '-'})"
        )
    if result["bet_scale"] < 0.999:
        result["runtime_protective_state"] = (
            "protective_observation" if result["mode"] == "soft_degraded" else "protective_scaling"
        )
    active_ratio = result["active_breakers"] / max(1, result["total_breakers"])
    severe_breakers = max(DOWN_BREAKER_ALERT_MIN_ACTIVE * 2, 8)
    if (
        (result["active_breakers"] >= DOWN_BREAKER_ALERT_MIN_ACTIVE and active_ratio >= DOWN_BREAKER_ALERT_MIN_RATIO)
        or result["active_breakers"] >= severe_breakers
    ):
        result["issues"].append(
            f"DOWN熔断激活 {result['active_breakers']}/{max(1, result['total_breakers'])}"
        )
    # V2 风控：按 cell 维度告警（兼容 V1，不替代）
    if result["v2_total_cells"] > 0:
        v2_block_ratio = result["v2_blocked_cells"] / max(1, result["v2_total_cells"])
        if (
            result["v2_blocked_cells"] >= DOWN_BREAKER_ALERT_MIN_ACTIVE
            and v2_block_ratio >= DOWN_BREAKER_ALERT_MIN_RATIO
        ) or result["v2_blocked_cells"] >= severe_breakers:
            result["issues"].append(
                f"DOWN风控V2封禁 {result['v2_blocked_cells']}/{result['v2_total_cells']}"
            )
    if result["up_v2_total_cells"] > 0:
        up_v2_block_ratio = result["up_v2_blocked_cells"] / max(1, result["up_v2_total_cells"])
        if (
            result["up_v2_blocked_cells"] >= DOWN_BREAKER_ALERT_MIN_ACTIVE
            and up_v2_block_ratio >= DOWN_BREAKER_ALERT_MIN_RATIO
        ) or result["up_v2_blocked_cells"] >= severe_breakers:
            result["issues"].append(
                f"UP风控V2封禁 {result['up_v2_blocked_cells']}/{result['up_v2_total_cells']}"
            )
    if result["shock_v2_total_cells"] > 0:
        shock_block_ratio = result["shock_v2_blocked_cells"] / max(1, result["shock_v2_total_cells"])
        if (
            result["shock_v2_blocked_cells"] >= DOWN_BREAKER_ALERT_MIN_ACTIVE
            and shock_block_ratio >= DOWN_BREAKER_ALERT_MIN_RATIO
        ) or result["shock_v2_blocked_cells"] >= severe_breakers:
            result["issues"].append(
                f"极端波动风控V2封禁 {result['shock_v2_blocked_cells']}/{result['shock_v2_total_cells']}"
            )
    if result["combo_pause_total_cells"] > 0:
        combo_block_ratio = result["combo_pause_blocked_cells"] / max(1, result["combo_pause_total_cells"])
        if (
            result["combo_pause_blocked_cells"] >= DOWN_BREAKER_ALERT_MIN_ACTIVE
            and combo_block_ratio >= DOWN_BREAKER_ALERT_MIN_RATIO
        ) or result["combo_pause_blocked_cells"] >= severe_breakers:
            result["issues"].append(
                f"组合级停机封禁 {result['combo_pause_blocked_cells']}/{result['combo_pause_total_cells']}"
            )

    # enforce 已开启但无生效证据，标记为疑似摆设层（样本不足时不判定）。
    if result["evidence_dummy_eval_ready"]:
        if (
            result["drift_enforce_cells"] > 0
            and result["drift_active_cells"] > 0
            and result["evidence_skipped_drift"] > 0
            and result["layer_evidence_counts"].get("threshold_drift", 0) <= 0
        ):
            result["issues"].append("方向阈值漂移 enforce 但无生效证据（疑似摆设层）")
        if result["expectancy_enforce_cells"] > 0 and result["layer_evidence_counts"].get("expectancy", 0) <= 0:
            result["issues"].append("近期OOS启停 enforce 但无生效证据（疑似摆设层）")
        if result["meta_enforce_cells"] > 0:
            if not bool(result.get("meta_runtime_observable")):
                result["issues"].append("Meta-Label runtime缺少可观测字段（无法判定是否真实生效）")
            elif result["layer_evidence_counts"].get("meta_label", 0) <= 0:
                result["issues"].append("Meta-Label enforce 但无生效证据（疑似摆设层）")
        # selector 当前由 Core/Shadow overlay 共同接管运行态；仅凭“无 ineligible/skip 证据”
        # 不能再判成健康异常，否则会把“本周期没有被 selector 拦截”误报成摆设层。
        if (
            result["shock_v2_enforce_cells"] > 0
            and (result["shock_v2_active_cells"] + result["shock_v2_blocked_cells"]) > 0
            and result["layer_evidence_counts"].get("shock", 0) <= 0
        ):
            result["issues"].append("极端波动风控 enforce 但无生效证据（疑似摆设层）")
        if (
            result["regime_gate_enforce_symbols"] > 0
            and result["regime_gate_active_symbols"] > 0
            and result["evidence_skipped_regime"] > 0
            and result["layer_evidence_counts"].get("regime", 0) <= 0
        ):
            result["issues"].append("regime门控 enforce 但无生效证据（疑似摆设层）")
    return result


def check_risk_tuning_reports() -> dict:
    """读取 UP/DOWN/SHOCK 超参报告，输出四簇可行性与违例摘要。"""
    out = {
        "exists": True,
        "families": {},
        "issues": [],
    }
    cluster_keys = ("default_BTC", "default_ETH", "70_BTC", "70_ETH")

    for family, path in RISK_TUNING_REPORTS.items():
        report_path = path
        if family == "shock":
            candidates = sorted((PROJECT_ROOT / "reports").glob(SHOCK_REJUDGE_GLOB), key=lambda p: p.stat().st_mtime)
            if candidates:
                report_path = candidates[-1]
        info = {
            "path": str(report_path),
            "present": report_path.exists(),
            "mtime": None,
            "clusters": {},
            "feasible_clusters": 0,
            "total_clusters": 0,
        }
        if not report_path.exists():
            out["exists"] = False
            out["issues"].append(f"{family.upper()} 报告缺失: {report_path.name}")
            out["families"][family] = info
            continue
        try:
            info["mtime"] = datetime.fromtimestamp(report_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            payload = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception as e:
            out["exists"] = False
            out["issues"].append(f"{family.upper()} 报告读取失败: {str(e)[:80]}")
            out["families"][family] = info
            continue

        clusters = payload.get("clusters", {}) if isinstance(payload, dict) else {}
        if not isinstance(clusters, dict):
            out["exists"] = False
            out["issues"].append(f"{family.upper()} 报告格式异常: clusters 缺失")
            out["families"][family] = info
            continue

        for ck in cluster_keys:
            if ck not in clusters or not isinstance(clusters.get(ck), dict):
                info["clusters"][ck] = {
                    "feasible": False,
                    "search_pass": None,
                    "violations": ["cluster_missing"],
                    "fold_trigger_counts": [],
                    "nonzero_fold_count": None,
                    "fold_count": None,
                    "trigger_fold_coverage": None,
                    "constraints_snapshot": None,
                }
                info["total_clusters"] += 1
                continue
            best = clusters.get(ck, {}).get("best", {}) if isinstance(clusters.get(ck), dict) else {}
            violations = best.get("constraint_violations") if isinstance(best, dict) else []
            if not isinstance(violations, list):
                violations = []
            feasible = bool(best.get("final_is_feasible")) if isinstance(best, dict) and ("final_is_feasible" in best) else (len(violations) == 0)
            info["clusters"][ck] = {
                "feasible": feasible,
                "search_pass": best.get("search_pass") if isinstance(best, dict) else None,
                "violations": [str(v) for v in violations],
                "fold_trigger_counts": best.get("fold_trigger_counts") if isinstance(best, dict) else [],
                "nonzero_fold_count": best.get("nonzero_fold_count") if isinstance(best, dict) else None,
                "fold_count": best.get("fold_count") if isinstance(best, dict) else None,
                "trigger_fold_coverage": best.get("trigger_fold_coverage") if isinstance(best, dict) else None,
                "constraints_snapshot": best.get("constraints_snapshot") if isinstance(best, dict) else None,
            }
            info["total_clusters"] += 1
            if feasible:
                info["feasible_clusters"] += 1
        out["families"][family] = info

    return out


def check_prediction_v13_admission() -> dict:
    out = {
        "present": False,
        "path": str(PREDICTION_V12_EVAL_LATEST),
        "mtime": None,
        "layers": {},
        "current_not_ready": [],
        "issues": [],
    }
    p = PREDICTION_V12_EVAL_LATEST
    if not p.exists():
        out["issues"].append("prediction_v12_base_eval_latest.json 缺失")
        return out
    try:
        out["mtime"] = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        out["issues"].append(f"读取准入评估失败: {str(e)[:80]}")
        return out

    adm = payload.get("admission_table") if isinstance(payload, dict) else None
    rows = []
    if isinstance(adm, dict):
        rows = adm.get("strict_cluster_rows") or adm.get("cluster_rows") or []
    if not isinstance(rows, list):
        out["issues"].append("准入评估格式异常: strict_cluster_rows 缺失")
        return out

    for row in rows:
        if not isinstance(row, dict):
            continue
        layer = str(row.get("layer") or "").strip().lower()
        if layer not in {
            "calibration",
            "expectancy",
            "shock",
            "ensemble_consensus",
            "threshold_drift",
            "meta_label",
            "selector",
            "regime",
            "joint_optimization",
        }:
            continue
        cluster = str(row.get("cluster") or "-")
        tier = str(row.get("admission_tier") or "C").upper()
        # recommended_mode 必须由评估层显式产出；若缺失，记录为数据契约异常。
        _raw_mode = row.get("recommended_mode")
        if _raw_mode is None or str(_raw_mode).strip() == "":
            out["issues"].append(
                f"准入评估缺少recommended_mode: layer={layer} cluster={cluster} tier={tier}"
            )
            # 缺字段时采用保守映射，避免误放行；同时通过 issues 暴露问题。
            mode = "off" if tier == "C" else "shadow"
        else:
            mode = str(_raw_mode).lower()
        ready = bool(row.get("ready_for_simulation"))
        violations = row.get("violations") if isinstance(row.get("violations"), list) else []
        root_cause = str(row.get("root_cause_type") or "unknown")
        li = out["layers"].setdefault(layer, {
            "A": 0, "B": 0, "C": 0, "ready": 0, "total": 0, "clusters": {}
        })
        if tier not in {"A", "B", "C"}:
            tier = "C"
        li[tier] += 1
        li["total"] += 1
        if ready:
            li["ready"] += 1
        li["clusters"][cluster] = {
            "tier": tier,
            "ready": ready,
            "mode": mode,
            "violations": [str(v) for v in violations],
        }
        if not ready:
            n_a_by_design = (
                layer == "selector"
                and mode == "off"
                and any(str(v) == "missing_cluster" for v in violations)
            )
            # C-tier+off in non-selector layers still means this cluster is disabled by admission.
            is_blocking_now = (not n_a_by_design) and (
                mode != "off" or tier == "C"
            )
            out["current_not_ready"].append(
                {
                    "layer": layer,
                    "cluster": cluster,
                    "tier": tier,
                    "mode": mode,
                    "n_a_by_design": n_a_by_design,
                    "is_blocking_now": is_blocking_now,
                    "root_cause_type": root_cause,
                    "violations": [str(v) for v in violations],
                }
            )

    out["present"] = True
    return out


def check_recent_drawdown_rootcause() -> dict:
    out = {
        "present": False,
        "path": str(RECENT_2DAY_DRAWDOWN_ROOTCAUSE_JSON),
        "generated_at": None,
        "overall_delta_pnl": None,
        "default_delta_pnl": None,
        "profile70_delta_pnl": None,
        "top_bucket": None,
        "top_worst_cells": [],
        "issues": [],
    }
    path = RECENT_2DAY_DRAWDOWN_ROOTCAUSE_JSON
    if not path.exists():
        out["issues"].append("recent_2day_drawdown_rootcause 缺失")
        return out
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        out["issues"].append(f"recent_2day_drawdown_rootcause 读取失败: {str(e)[:80]}")
        return out

    out["present"] = True
    out["generated_at"] = payload.get("generated_at")
    overall = payload.get("overall", {}) if isinstance(payload.get("overall"), dict) else {}
    profiles = payload.get("profiles", {}) if isinstance(payload.get("profiles"), dict) else {}
    out["overall_delta_pnl"] = ((overall.get("delta") or {}).get("pnl") if isinstance(overall.get("delta"), dict) else None)
    out["default_delta_pnl"] = (((profiles.get("default") or {}).get("delta") or {}).get("pnl") if isinstance((profiles.get("default") or {}).get("delta"), dict) else None)
    out["profile70_delta_pnl"] = (((profiles.get("70") or {}).get("delta") or {}).get("pnl") if isinstance((profiles.get("70") or {}).get("delta"), dict) else None)

    buckets = payload.get("direction_bucket_delta_rank")
    if isinstance(buckets, list) and buckets:
        b0 = buckets[0] if isinstance(buckets[0], dict) else {}
        out["top_bucket"] = {
            "profile": b0.get("profile"),
            "symbol": b0.get("symbol"),
            "direction": b0.get("direction"),
            "delta_pnl_sum": b0.get("delta_pnl_sum"),
        }
    cells = payload.get("worst_cells_by_delta_pnl")
    if isinstance(cells, list):
        out["top_worst_cells"] = [x for x in cells[:5] if isinstance(x, dict)]
    return out


def check_risk_jobs() -> dict:
    """检查风控优化与盯盘任务是否在运行，并给出最近日志。"""
    out = {"jobs": {}, "issues": []}
    patterns = {
        "down": "optimize_down_risk_v2.py",
        "up": "optimize_up_risk_v2.py",
        "combo": "optimize_combo_pause_v1.py",
        "shock": "optimize_shock_risk_v2.py",
        "calibration": "optimize_direction_calibration.py",
        "expectancy": "optimize_expectancy_gate.py",
        "joint": "optimize_joint_execution_layer.py",
        "regime": "write_regime_shadow_state.py",
        "watch": "watch_up_down_risk_4h.py",
    }
    ps_lines: List[str] = []
    try:
        ret = subprocess.run(
            ["ps", "axww", "-o", "pid=,etime=,command="],
            capture_output=True,
            text=True,
            timeout=10,
        )
        ps_lines = ret.stdout.splitlines()
    except Exception as e:
        out["issues"].append(f"风控任务进程扫描失败: {str(e)[:80]}")

    for name, marker in patterns.items():
        matched = []
        for line in ps_lines:
            if marker not in line:
                continue
            parts = line.strip().split(None, 2)
            if len(parts) < 3:
                continue
            pid, etime, cmd = parts
            cmd_head = os.path.basename(cmd.strip().split()[0]).lower() if cmd.strip() else ""
            if "python" not in cmd_head:
                continue
            matched.append({
                "pid": pid,
                "etime": etime,
                "cmd": cmd[:220],
            })
        latest_log = None
        latest_mtime = None
        for pattern in RISK_JOB_LOG_GLOBS.get(name, []):
            for p in pattern.parent.glob(pattern.name):
                try:
                    mtime = p.stat().st_mtime
                except OSError:
                    continue
                if latest_mtime is None or mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_log = p
        out["jobs"][name] = {
            "running": bool(matched),
            "processes": matched,
            "latest_log": str(latest_log) if latest_log else None,
            "latest_log_mtime": (
                datetime.fromtimestamp(latest_mtime).strftime("%Y-%m-%d %H:%M:%S")
                if latest_mtime is not None
                else None
            ),
        }
    return out


def check_proxy_connections() -> Tuple[int, str]:
    """统计通过代理的连接数。"""
    try:
        result = subprocess.run(
            ["lsof", "-nP", "-i", "TCP:7897"],
            capture_output=True, text=True, timeout=10)
        established = sum(1 for l in result.stdout.split("\n")
                         if "ESTABLISHED" in l)
        return established, f"{established}个活跃连接"
    except Exception as e:
        return -1, f"检查失败: {str(e)[:40]}"


def check_log_anomalies() -> List[dict]:
    """扫描合并进程日志和旧日志中的异常模式：crash, OOM, unhandled, 严重错误。"""
    anomaly_patterns = [
        (r"FATAL|fatal error", "FATAL", "error", 1),
        (r"out of memory|heap out of memory|OOM", "OOM", "error", 1),
        (r"unhandled.*rejection|uncaught.*exception", "未捕获异常", "error", 1),
        (r"ENOSPC|disk full|no space", "磁盘满", "error", 1),
        (r"EMFILE|too many open files", "文件描述符耗尽", "error", 1),
        (r"segfault|Segmentation fault|SIGSEGV", "段错误", "error", 1),
        (r"Cannot compare tz-naive and tz-aware timestamps", "时区比较错误", "error", 1),
        (r"FGI 历史文件写入失败", "FGI写盘失败", "error", 1),
        (r"完整K线验证失败，跳过本轮预测", "K线验证失败", "warn", 2),
        (r"update_latest.*最终失败", "update_latest最终失败", "warn", 3),
        (r"PM目标\s*0/3OK", "PM目标全失败", "warn", 2),
        (r"上一轮训练仍在运行，跳过本轮", "训练重叠跳过", "warn", 2),
    ]
    anomalies: List[dict] = []

    log_files_to_scan: List[Tuple[str, Path]] = []

    # 合并进程日志（优先扫描）
    for log_file in sorted(POLYMARKET_DIR.glob("multi_*_stdout.log")):
        log_files_to_scan.append((log_file.name, log_file))

    # 旧式日志目录
    for log_dir in POLYMARKET_DIR.glob("logs_*"):
        for log_file in log_dir.glob("*.log"):
            log_files_to_scan.append((f"{log_dir.name}/{log_file.name}", log_file))
    for extra in [
        Path.home() / "Library" / "Logs" / "polyfun" / "daily_training_scheduler.launchd.log",
        PROJECT_ROOT / "logs" / "online_learning.log",
        PROJECT_ROOT / "logs" / "sentiment_collector.log",
    ]:
        if extra.exists():
            log_files_to_scan.append((f"logs/{extra.name}", extra))

    for label, log_file in log_files_to_scan:
        try:
            with open(log_file, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                read_from = max(0, size - 10000)
                f.seek(read_from)
                tail = f.read().decode("utf-8", errors="replace")
        except OSError:
            continue
        for pattern, plabel, severity, min_count in anomaly_patterns:
            matches = re.findall(pattern, tail, re.IGNORECASE)
            if len(matches) >= min_count:
                anomalies.append({
                    "file": label,
                    "type": plabel,
                    "count": len(matches),
                    "severity": severity,
                })

    return anomalies


def check_ensemble_output() -> dict:
    """验证融合参考产物的合理性（信息级，不参与在线交易健康故障判定）。"""
    def _parse_etime_to_seconds(etime_text: str) -> Optional[int]:
        s = (etime_text or "").strip()
        if not s:
            return None
        days = 0
        if "-" in s:
            day_part, s = s.split("-", 1)
            try:
                days = int(day_part)
            except ValueError:
                return None
        parts = s.split(":")
        try:
            nums = [int(x) for x in parts]
        except ValueError:
            return None
        if len(nums) == 2:
            h, m, sec = 0, nums[0], nums[1]
        elif len(nums) == 3:
            h, m, sec = nums[0], nums[1], nums[2]
        else:
            return None
        return days * 86400 + h * 3600 + m * 60 + sec

    def _ensemble_writer_uptime_sec() -> Optional[int]:
        try:
            ps = subprocess.run(
                ["ps", "-eo", "etime,args", "-ww"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            best: Optional[int] = None
            for line in (ps.stdout or "").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                etime = _parse_etime_to_seconds(parts[0])
                args = parts[1]
                if (
                    etime is not None
                    and "ensemble_prediction_writer.py" in args
                    and "--output-70" not in args
                    and "api_health_check.py" not in args
                ):
                    if best is None or etime < best:
                        best = etime
            return best
        except Exception:
            return None

    fpath = POLYMARKET_DIR / "predictions_ensemble.json"
    if not fpath.exists():
        return {"exists": False, "ok": True, "issues": [], "timestamp": "", "target_ts": 0}

    try:
        file_age = time.time() - fpath.stat().st_mtime
    except Exception:
        file_age = 99999

    try:
        with open(fpath) as f:
            data = json.load(f)
    except Exception as e:
        return {"exists": True, "ok": False, "issues": [f"JSON解析失败: {str(e)[:50]}"], "timestamp": "", "target_ts": 0}

    issues = []
    preds = data.get("predictions", {})

    for coin_key in ("BTC_USDT_15m", "ETH_USDT_15m"):
        entry = preds.get(coin_key, {})
        if not entry:
            issues.append(f"{coin_key}: 缺失")
            continue
        if entry.get("error"):
            issues.append(f"{coin_key}: {entry['error']}")
            continue

        conf = entry.get("confidence", 0)
        direction = entry.get("direction")
        details = entry.get("details", {})
        n_sources = details.get("n_sources", 0)
        consensus = details.get("consensus_ratio", 0)
        proba_up = details.get("proba_up", 0)
        proba_raw = details.get("proba_up_raw", 0)
        trade_decision = details.get("trade_decision", {}) if isinstance(details, dict) else {}
        should_trade = bool(trade_decision.get("should_trade", False))
        skip_reason = str(trade_decision.get("skip_reason", ""))
        skipped_for_sources = "sources=" in skip_reason and "<" in skip_reason

        if direction not in ("UP", "DOWN"):
            issues.append(f"{coin_key}: direction={direction} 异常")
        if conf < 0.001 or conf > 0.999:
            issues.append(f"{coin_key}: confidence={conf} 越界")
        # 若融合已明确按“源数不足”跳过交易，则不视为健康异常（属于策略保护逻辑）。
        if n_sources < 2 and not skipped_for_sources:
            issues.append(f"{coin_key}: 仅{n_sources}个源(不足)")
        # 低共识仅在“准备交易”时才报错；已跳过交易时属于预期。
        if consensus < 0.4 and should_trade:
            issues.append(f"{coin_key}: 共识={consensus:.0%} 极低")

        # 方向与概率一致性
        if direction == "UP" and proba_up < 0.5:
            issues.append(f"{coin_key}: direction=UP但proba_up={proba_up:.4f}<0.5")
        elif direction == "DOWN" and proba_up > 0.5:
            issues.append(f"{coin_key}: direction=DOWN但proba_up={proba_up:.4f}>0.5")

        # 缩放合理性: raw 和 scaled 方向应一致
        if proba_raw and proba_up:
            if (proba_raw - 0.5) * (proba_up - 0.5) < 0:
                issues.append(f"{coin_key}: 缩放翻转方向! raw={proba_raw:.4f} scaled={proba_up:.4f}")

    target_ts = data.get("target_period_end_ts", 0)
    age = time.time() - target_ts if target_ts else 99999
    # 仅在 target_period 与文件mtime同时过期时才告警，避免写入器刚重启时误报。
    if age > 1200 and file_age > 1200:
        writer_age = _ensemble_writer_uptime_sec()
        if not (
            writer_age is not None
            and writer_age <= 35 * 60
            and writer_age < file_age
        ):
            issues.append(f"target_period 过期 ({age:.0f}s, file_age={file_age:.0f}s)")

    return {
        "exists": True,
        "ok": len(issues) == 0,
        "issues": issues,
        "target_ts": target_ts,
        "timestamp": data.get("timestamp", ""),
    }


def check_market_data_daemon() -> dict:
    """检查 Market Data Daemon 进程和缓存文件的健康状态。"""
    result: dict = {"daemon_running": False, "cache_fresh": False, "cache_age_s": -1,
                    "market_count": 0, "token_count": 0, "issues": []}
    pid_file = POLYMARKET_DIR / "market_data_daemon.pid"
    cache_file = POLYMARKET_DIR / "market_data_cache.json"

    # 检查 daemon 进程
    daemon_uptime_s: Optional[int] = None
    if pid_file.exists():
        try:
            pid = pid_file.read_text().strip()
            ret = subprocess.run(["kill", "-0", pid], capture_output=True)
            result["daemon_running"] = ret.returncode == 0
            if result["daemon_running"] and pid:
                et = subprocess.run(
                    ["ps", "-p", pid, "-o", "etime="],
                    capture_output=True,
                    text=True,
                    timeout=3,
                )
                etxt = (et.stdout or "").strip()
                # 解析 MM:SS / HH:MM:SS / DD-HH:MM:SS
                if etxt:
                    days = 0
                    if "-" in etxt:
                        d, etxt = etxt.split("-", 1)
                        days = int(d)
                    parts = [int(x) for x in etxt.split(":")]
                    if len(parts) == 2:
                        h, m, s = 0, parts[0], parts[1]
                    else:
                        h, m, s = parts[0], parts[1], parts[2]
                    daemon_uptime_s = days * 86400 + h * 3600 + m * 60 + s
        except Exception:
            pass

    # 兜底：PID 文件缺失/失效时，从进程表识别 Daemon，避免误报。
    if not result["daemon_running"]:
        try:
            ps_ret = subprocess.run(
                ["ps", "-Ao", "pid,command", "-ww"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if ps_ret.returncode == 0:
                for line in (ps_ret.stdout or "").splitlines():
                    l = line.strip()
                    if not l:
                        continue
                    if ("node dist/market_data_daemon.js" in l) or ("ts-node src/market_data_daemon.ts" in l):
                        pid_guess = l.split(None, 1)[0]
                        result["daemon_running"] = True
                        try:
                            pid_file.write_text(pid_guess, encoding="utf-8")
                        except Exception:
                            pass
                        break
        except Exception:
            pass

    if not result["daemon_running"]:
        result["issues"].append("Daemon 进程未运行")

    # 检查缓存文件
    if cache_file.exists():
        try:
            age_s = time.time() - cache_file.stat().st_mtime
            result["cache_age_s"] = round(age_s, 1)
            result["cache_fresh"] = age_s < 30
            if age_s >= 30:
                result["issues"].append(f"缓存过期: {result['cache_age_s']}s (>30s)")
            with open(cache_file) as f:
                data = json.load(f)
            result["market_count"] = len(data.get("markets", {}))
            result["token_count"] = len(data.get("prices", {}))
            if result["market_count"] == 0:
                result["issues"].append("缓存中无市场数据")
            ob = data.get("orderbooks", {})
            stale_ob = sum(1 for v in ob.values() if time.time() - v.get("updated_at", 0) > 60)
            result["stale_orderbooks"] = stale_ob
            result["total_orderbooks"] = len(ob)
            if stale_ob > len(ob) * 0.5 and len(ob) > 0:
                result["issues"].append(f"过期订单簿偏多: {stale_ob}/{len(ob)} (>60s)")
        except Exception as e:
            result["issues"].append(f"缓存读取失败: {e}")
    else:
        if result["daemon_running"] and daemon_uptime_s is not None and daemon_uptime_s <= DAEMON_CACHE_WARMUP_SEC:
            # 刚启动预热，允许缓存文件短暂未生成
            result["issues"].append(f"缓存预热中: daemon_uptime={daemon_uptime_s}s")
        else:
            result["issues"].append("缓存文件不存在")

    return result


def check_data_sources() -> List[dict]:
    """检查数据采集器各数据源的新鲜度。"""
    now = time.time()
    results = []

    DATA_DIR = PROJECT_ROOT / "data" / "sentiment"
    sources = [
        ("Funding Rate",     "funding_rate_history.parquet",         5,  "WS"),
        ("Open Interest",    "open_interest_15m.parquet",            5,  "REST"),
        ("Long/Short Ratio", "long_short_ratio_15m.parquet",        5,  "REST"),
        ("PM OB Snapshots",  "polymarket_ob_snapshots.parquet",     20, "REST"),
        ("PM Target BTC",    "polymarket_prob_target_btc_usdt.parquet", 20, "REST"),
        ("PM Target ETH",    "polymarket_prob_target_eth_usdt.parquet", 20, "REST"),
        ("PM Target XRP",    "polymarket_prob_target_xrp_usdt.parquet", 20, "REST"),
    ]

    for label, fname, max_age_min, src_type in sources:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            results.append({"name": label, "ok": False, "src": src_type,
                            "status": "文件不存在"})
            continue
        mtime = os.path.getmtime(fpath)
        age_min = (now - mtime) / 60
        update_time = datetime.fromtimestamp(mtime).strftime("%H:%M:%S")
        if age_min <= max_age_min:
            results.append({"name": label, "ok": True, "src": src_type,
                            "status": f"更新:{update_time} ({age_min:.0f}分钟前)"})
        else:
            results.append({"name": label, "ok": False, "src": src_type,
                            "status": f"过期! 更新:{update_time} ({age_min:.0f}分钟前, >{max_age_min}min)"})
    return results


def check_data_quality_sources() -> List[dict]:
    """检查 data_ready.json 中门控关键源的新鲜度（与运行时门控同口径）。"""
    now = time.time()
    max_age_min = max(1, int(round(DATA_QUALITY_MAX_AGE_SEC_EXPECTED / 60)))
    key_specs = [
        ("DQ Open Interest", "open_interest"),
        ("DQ Long/Short Ratio", "long_short_ratio"),
        ("DQ PM Prob", "polymarket_prob"),
        ("DQ PM Prob Target", "polymarket_prob_target"),
        ("DQ Funding Rate", "funding_rate"),
        ("DQ OB Realtime", "ob_realtime"),
    ]
    if not DATA_READY_FILE.exists():
        return [{
            "name": "DQ data_ready.json",
            "key": "-",
            "ok": False,
            "status": f"文件不存在 ({DATA_READY_FILE})",
            "max_age_min": max_age_min,
        }]

    try:
        payload = json.loads(DATA_READY_FILE.read_text(encoding="utf-8"))
    except Exception as e:
        return [{
            "name": "DQ data_ready.json",
            "key": "-",
            "ok": False,
            "status": f"JSON解析失败: {str(e)[:80]}",
            "max_age_min": max_age_min,
        }]
    if not isinstance(payload, dict):
        return [{
            "name": "DQ data_ready.json",
            "key": "-",
            "ok": False,
            "status": "JSON格式异常（非对象）",
            "max_age_min": max_age_min,
        }]

    results: List[dict] = []
    for label, key in key_specs:
        max_age_sec = DATA_QUALITY_MAX_AGE_BY_SOURCE_SEC.get(key, DATA_QUALITY_MAX_AGE_SEC_EXPECTED)
        max_age_min_key = max(1, int(round(max_age_sec / 60)))

        ts_raw = payload.get(key)
        try:
            ts_val = float(ts_raw)
        except (TypeError, ValueError):
            ts_val = float("nan")

        chosen_ts = ts_val if math.isfinite(ts_val) else None
        chosen_src = "data_ready"

        if chosen_ts is None or (now - chosen_ts) > max_age_sec:
            fallback_files = DATA_QUALITY_MTIME_FALLBACK_FILES.get(key, [])
            fallback_ts = None
            for fp in fallback_files:
                if not fp.exists():
                    continue
                mt = fp.stat().st_mtime
                if fallback_ts is None or mt > fallback_ts:
                    fallback_ts = mt
            if fallback_ts is not None and (now - fallback_ts) <= max_age_sec:
                chosen_ts = fallback_ts
                chosen_src = "file_mtime_fallback"

        if chosen_ts is None:
            ok = False
            age_min = float("inf")
            status = f"过期! 缺少或非数值时间戳 (> {max_age_min_key}min)"
        else:
            age_min = max(0.0, (now - chosen_ts) / 60.0)
            update_time = datetime.fromtimestamp(chosen_ts, tz=timezone.utc).strftime("%H:%M:%S")
            if age_min <= max_age_min_key:
                suffix = "" if chosen_src == "data_ready" else " (fallback)"
                status = f"更新:{update_time} UTC ({age_min:.0f}分钟前){suffix}"
                ok = True
            else:
                status = f"过期! 更新:{update_time} UTC ({age_min:.0f}分钟前, >{max_age_min_key}min)"
                ok = False

        results.append({
            "name": label,
            "key": key,
            "ok": ok,
            "status": status,
            "max_age_min": max_age_min_key,
            "age_min": age_min,
        })
    return results


# 本地 K 线 (15m/1h/4h) 过期阈值（分钟）：超过则视为「未更新到最新」
RAW_OHLCV_STALE_MIN = {"15m": 30, "1h": 120, "4h": 300}


def check_raw_ohlcv_freshness() -> List[dict]:
    """检查 data/raw 下 BTC/ETH/SOL/XRP 的 15m/1h/4h parquet 是否更新到最新（最后一根K线距今）。"""
    sys.path.insert(0, str(PROJECT_ROOT))
    try:
        import pandas as pd
    except ImportError:
        return [{"symbol": "?", "tf": "?", "ok": False, "status": "pandas 不可用"}]
    raw_dir = PROJECT_ROOT / "data" / "raw"
    if _slot_scoped_live_health_only_enabled():
        symbols = ["ETH/USDT"]
    else:
        active_symbols = _load_active_symbols_from_configs()
        if active_symbols:
            symbols = [f"{s}/USDT" for s in ("BTC", "ETH", "XRP", "SOL") if s in active_symbols]
        else:
            # 兜底: 配置读取失败时沿用历史检查集合
            symbols = ["BTC/USDT", "ETH/USDT", "XRP/USDT"]
            active_groups = get_active_groups()
            if (not active_groups) or ("gru_all" in active_groups):
                symbols.append("SOL/USDT")
    results = []
    now_ms = int(time.time() * 1000)
    for sym in symbols:
        safe = sym.replace("/", "_").lower()
        for tf in ("15m", "1h", "4h"):
            path = raw_dir / f"{safe}_{tf}.parquet"
            max_age = RAW_OHLCV_STALE_MIN[tf]
            if not path.exists():
                results.append({
                    "symbol": sym, "tf": tf, "ok": False,
                    "status": "文件不存在(下载未成功或未跑过)",
                })
                continue
            try:
                df = pd.read_parquet(path)
                if df.empty or "timestamp" not in df.columns:
                    results.append({"symbol": sym, "tf": tf, "ok": False, "status": "无有效数据"})
                    continue
                last_ts = int(df["timestamp"].iloc[-1])
                age_min = (now_ms - last_ts) / (60 * 1000)
                last_utc = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).strftime("%m-%d %H:%M")
                if age_min <= max_age:
                    results.append({
                        "symbol": sym, "tf": tf, "ok": True,
                        "status": f"最后K线 {last_utc} UTC, 距今 {age_min:.0f}min",
                    })
                else:
                    results.append({
                        "symbol": sym, "tf": tf, "ok": False,
                        "status": f"过期 最后K线 {last_utc} UTC, 距今 {age_min:.0f}min (>{max_age}min)",
                    })
            except Exception as e:
                results.append({"symbol": sym, "tf": tf, "ok": False, "status": f"读失败: {str(e)[:40]}"})
    return results


def check_recent_trades() -> dict:
    """统计各日志目录中最近的交易活动。"""
    trade_stats = {"with_trades": 0, "no_trades": 0, "total_recent": 0}

    def _trade_event_ts(trade: dict) -> Optional[float]:
        # 优先使用结算时间，再回退到通用时间字段；兼容 epoch 秒/毫秒与 ISO 文本。
        for key in ("settledAt", "timestamp", "createdAt", "updatedAt"):
            v = trade.get(key)
            if v is None:
                continue
            if isinstance(v, (int, float)):
                ts = float(v)
                if ts > 1e12:  # 毫秒
                    ts = ts / 1000.0
                return ts if ts > 0 else None
            if isinstance(v, str):
                s = v.strip()
                if not s:
                    continue
                # 纯数字字符串
                if s.isdigit():
                    ts = float(s)
                    if ts > 1e12:
                        ts = ts / 1000.0
                    return ts if ts > 0 else None
                t = _parse_iso_ts(s)
                if t is not None:
                    return float(t)
        return None

    for log_dir in sorted(POLYMARKET_DIR.glob("logs_*")):
        # 仅读取分模式账本，避免监控口径混入合并账本。
        mode_files = [
            log_dir / "prediction_trades.simulation.json",
            log_dir / "prediction_trades.live.json",
            log_dir / "prediction_trades.backtest.json",
        ]
        existing_mode_files = [p for p in mode_files if p.exists()]
        files_to_read = existing_mode_files
        files_to_read = [p for p in files_to_read if p.exists()]
        if not files_to_read:
            trade_stats["no_trades"] += 1
            continue
        try:
            trades: List[dict] = []
            for trades_file in files_to_read:
                with open(trades_file, encoding="utf-8") as f:
                    data = json.load(f)
                part = data if isinstance(data, list) else data.get("trades", [])
                if isinstance(part, list):
                    trades.extend([x for x in part if isinstance(x, dict)])
            now = time.time()
            recent = 0
            for t in trades:
                ts = _trade_event_ts(t)
                if ts is not None and now - ts < 3600:
                    recent += 1
            if len(trades) > 0:
                trade_stats["with_trades"] += 1
            else:
                trade_stats["no_trades"] += 1
            trade_stats["total_recent"] += recent
        except Exception:
            trade_stats["no_trades"] += 1

    return trade_stats


def classify_root_causes(alerts: List[str]) -> Dict[str, List[str]]:
    """把告警按三类根因分组，便于快速判断是 VPN、逻辑还是重启问题。"""
    buckets: Dict[str, List[str]] = {
        "P0 网络/VPN链路": [],
        "P1 逻辑/数据链路": [],
        "P2 进程/重启治理": [],
    }

    network_keys = (
        "系统代理不可用", "API不通", "ccxt无法连接", "WebSocket大面积断开",
        "REST大量超时", "代理连接数过高", "PM目标全失败",
        "update_latest最终失败", "外部API错误率偏高",
    )
    logic_keys = (
        "时区比较错误", "FGI写盘失败", "K线验证失败", "融合输出异常",
        "target_period", "本地K线过期", "监控环境缺少依赖",
    )
    process_keys = (
        "进程缺失", "合并进程未运行", "旧式独立交易进程仍在运行", "重复进程",
        "僵尸进程", "调度器异常", "训练重叠跳过", "数据采集器异常",
    )

    def add_unique(bucket: str, msg: str) -> None:
        if msg not in buckets[bucket]:
            buckets[bucket].append(msg)

    for alert in alerts:
        hit = False
        if any(k in alert for k in network_keys):
            add_unique("P0 网络/VPN链路", alert)
            hit = True
        if any(k in alert for k in logic_keys):
            add_unique("P1 逻辑/数据链路", alert)
            hit = True
        if any(k in alert for k in process_keys):
            add_unique("P2 进程/重启治理", alert)
            hit = True

        # 数据源过期默认按网络链路优先处理
        if not hit and "数据源过期" in alert:
            add_unique("P0 网络/VPN链路", alert)
            hit = True

        if not hit and "日志异常" in alert:
            add_unique("P1 逻辑/数据链路", alert)
            hit = True

        if not hit and ("运行时风控" in alert or "DOWN熔断" in alert):
            add_unique("P1 逻辑/数据链路", alert)
            hit = True

        if not hit:
            add_unique("P1 逻辑/数据链路", alert)

    return buckets


# ─── 报告输出 ─────────────────────────────────────────────

def run_check(auto_heal: bool = False, expected_interval_minutes: Optional[int] = None) -> bool:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alerts: List[str] = []
    all_ok = True
    expected_groups_runtime = get_expected_merged_groups()
    expected_total_traders_runtime = sum(expected_groups_runtime.values())
    active_groups_cfg = get_active_groups()
    active_scope = _load_active_scope_summary()
    slot_scoped_health = _slot_scoped_live_health_only_enabled()
    slot_scoped_targets = _load_slot_scoped_live_targets() if slot_scoped_health else []
    slot_scoped_target_symbols = sorted({
        str((row or {}).get("symbol") or "").strip().upper()
        for row in slot_scoped_targets
        if str((row or {}).get("symbol") or "").strip()
    })

    print(f"\n{'═' * 70}")
    print(f"  系统健康监控 v5 (合并架构)  {now_str}")
    print(f"{'═' * 70}")

    # 1. 系统代理
    proxy_ok, proxy_msg = check_system_proxy()
    icon = "✅" if proxy_ok else "❌"
    print(f"\n🔌 系统代理: {icon} {proxy_msg}")
    if not proxy_ok:
        all_ok = False
        alerts.append("系统代理不可用 — Binance/API可能无法连接")
    if slot_scoped_health and slot_scoped_targets:
        print("   运行组过滤: lowprice_70_selected (slot1/slot2 + ETH)")
        print(
            "   Active宇宙: "
            f"slot_live={len(slot_scoped_targets)} traders/{len(slot_scoped_targets)} cells, "
            f"symbols={','.join(slot_scoped_target_symbols) or '-'}"
        )
        prof70_scope = active_scope.get("70", {}) if isinstance(active_scope, dict) else {}
        prof70_mtime = prof70_scope.get("mtime", 0.0)
        prof70_mtime_s = datetime.fromtimestamp(prof70_mtime, tz=timezone.utc).strftime("%m-%d %H:%M:%S UTC") if prof70_mtime else "-"
        print(f"   Active来源: slot_scoped_live_contract@{prof70_mtime_s} | BTC/XRP 已 parked")
    else:
        if active_groups_cfg:
            print(f"   运行组过滤: {', '.join(sorted(active_groups_cfg))}")
        default_scope = active_scope.get("default", {}) if isinstance(active_scope, dict) else {}
        prof70_scope = active_scope.get("70", {}) if isinstance(active_scope, dict) else {}
        print(
            "   Active宇宙: "
            f"default={int(default_scope.get('traders', 0))} traders/{int(default_scope.get('cells', 0))} cells, "
            f"70={int(prof70_scope.get('traders', 0))} traders/{int(prof70_scope.get('cells', 0))} cells, "
            f"total={int(active_scope.get('total_traders', 0))} traders/{int(active_scope.get('total_cells', 0))} cells"
        )
        def_mtime = default_scope.get("mtime", 0.0)
        prof70_mtime = prof70_scope.get("mtime", 0.0)
        def_mtime_s = datetime.fromtimestamp(def_mtime, tz=timezone.utc).strftime("%m-%d %H:%M:%S UTC") if def_mtime else "-"
        prof70_mtime_s = datetime.fromtimestamp(prof70_mtime, tz=timezone.utc).strftime("%m-%d %H:%M:%S UTC") if prof70_mtime else "-"
        print(f"   Active来源: default@{def_mtime_s} | 70@{prof70_mtime_s}")

    # 2. API连通性
    print(f"\n📡 外部API:")
    api_results = check_api_endpoints()
    pending_api_failures: List[str] = []
    for r in api_results:
        if r["ok"]:
            print(f"  ✅ {r['name']:<20} {r['ms']}ms")
        else:
            print(f"  ❌ {r['name']:<20} {r['ms']}ms  {r['error']}")
            pending_api_failures.append(str(r["name"]))

    # 3. ccxt
    print(f"\n🐍 ccxt Binance:")
    ccxt_ok, ccxt_msg = check_ccxt()
    icon = "✅" if ccxt_ok else "❌"
    print(f"  {icon} {ccxt_msg}")
    ccxt_soft_fail = False
    if not ccxt_ok:
        if "ccxt 库未安装" in ccxt_msg:
            all_ok = False
            alerts.append("监控环境缺少依赖: ccxt（请在当前 Python 安装）")
        else:
            # 先标记，待后续结合 Binance API / 本地K线 / 预测文件再判断是否致命告警。
            ccxt_soft_fail = True

    daemon_check = check_market_data_daemon()
    dc_ok, dc_msg = check_data_collector()
    heal_actions: List[str] = []
    if auto_heal:
        need_heal_daemon = (not daemon_check["daemon_running"]) or (not daemon_check["cache_fresh"])
        if need_heal_daemon:
            ok, out = auto_heal_market_data_daemon()
            heal_actions.append(f"{'✅' if ok else '❌'} Daemon 自愈: {'已重启' if ok else '失败'}")
            if out:
                heal_actions.append(f"   ↳ {out.splitlines()[-1][:120]}")
        if not dc_ok:
            ok, out = auto_heal_derivatives_collector()
            heal_actions.append(f"{'✅' if ok else '❌'} 采集器自愈: {'已重载' if ok else '失败'}")
            if out:
                heal_actions.append(f"   ↳ {out.splitlines()[-1][:120]}")
        if heal_actions:
            time.sleep(2.0)
            daemon_check = check_market_data_daemon()
            dc_ok, dc_msg = check_data_collector()

    # 4. Market Data Daemon
    print(f"\n🏪 Market Data Daemon:")
    daemon_icon = "✅" if daemon_check["daemon_running"] else "❌"
    cache_icon = "✅" if daemon_check["cache_fresh"] else "❌"
    print(f"  进程: {daemon_icon} {'运行中' if daemon_check['daemon_running'] else '未运行'}")
    stale_ob = daemon_check.get("stale_orderbooks", 0)
    total_ob = daemon_check.get("total_orderbooks", 0)
    print(f"  缓存: {cache_icon} age={daemon_check['cache_age_s']}s | "
          f"{daemon_check['market_count']}市场 | {daemon_check['token_count']}token | "
          f"订单簿过期: {stale_ob}/{total_ob}")
    for issue in daemon_check["issues"]:
        all_ok = False
        alerts.append(f"Market Data Daemon: {issue}")

    # 5. 数据采集器
    print(f"\n📊 数据采集器:")
    icon = "✅" if dc_ok else "❌"
    print(f"  {icon} {dc_msg}")
    if not dc_ok:
        all_ok = False
        alerts.append(f"数据采集器异常: {dc_msg}")

    # 5b. 数据源新鲜度
    print(f"\n📈 数据源新鲜度 (采集器实时数据):")
    ds_checks = check_data_sources()
    for d in ds_checks:
        if d["ok"]:
            print(f"  ✅ {d['name']:<20} [{d['src']}] {d['status']}")
        else:
            all_ok = False
            print(f"  ❌ {d['name']:<20} [{d['src']}] {d['status']}")
            alerts.append(f"数据源过期: {d['name']} — {d['status']}")

    # 数据源过期时 --auto-heal：仅补历史（不重启采集器），VPN 恢复后下一轮检查会变绿
    if auto_heal and any(not d["ok"] for d in ds_checks):
        ok_backfill, out_backfill = auto_heal_data_source_backfill()
        print(f"\n🛠️ 数据源补历史 (--auto-heal，未重启采集器):")
        print(f"  {'✅' if ok_backfill else '❌'} 补拉 Funding/OI/LS 最近2天: {'成功' if ok_backfill else '失败或跳过'}")
        if out_backfill and out_backfill.strip():
            for line in out_backfill.strip().splitlines()[-3:]:
                print(f"   ↳ {line[:100]}")

    # 5b-2. data_ready 门控源（与运行时风控同口径）
    print(f"\n📉 数据质量门控源 (data_ready.json):")
    dq_checks = check_data_quality_sources()
    for d in dq_checks:
        if d["ok"]:
            print(f"  ✅ {d['name']:<20} {d['status']}")
        else:
            all_ok = False
            print(f"  ❌ {d['name']:<20} {d['status']}")
            alerts.append(f"数据源过期(data_ready): {d['name']} — {d['status']}")

    # 5c. 本地 K 线 (15m/1h/4h) 新鲜度 — 是否更新到最新、下载成功与否
    local_kline_scope = "ETH (slot1/slot2 + 必要共享链)" if _slot_scoped_live_health_only_enabled() else "BTC/ETH/SOL/XRP"
    print(f"\n📊 本地K线 (15m/1h/4h) {local_kline_scope}:")
    ohlcv_checks = check_raw_ohlcv_freshness()
    for r in ohlcv_checks:
        label = f"{r['symbol']} {r['tf']}"
        if r["ok"]:
            print(f"  ✅ {label:<16} {r['status']}")
        else:
            all_ok = False
            print(f"  ❌ {label:<16} {r['status']}")
            alerts.append(f"本地K线过期: {label} — {r['status']}")

    # 6. 写入器进程
    print(f"\n⚙️ 预测写入器进程:")
    writer_checks = check_writer_processes()
    for w in writer_checks:
        if w["ok"]:
            print(f"  ✅ {w['name']:<25} ({w['actual']}个进程)")
        else:
            all_ok = False
            status = f"期望{w['expected']}个, 实际{w['actual']}个"
            print(f"  ❌ {w['name']:<25} {status}")
            alerts.append(f"进程缺失: {w['name']} ({status})")

    # 7. 预测文件新鲜度
    print(
        "\n📄 预测文件 (V5>35分钟、GRU>25分钟、T-120s>50分钟=过期):"
    )
    file_checks = check_prediction_files()
    for f in file_checks:
        if f["ok"]:
            print(f"  ✅ {f['name']:<18} {f['status']}")
        else:
            all_ok = False
            print(f"  ❌ {f['name']:<18} {f['status']}")
            alerts.append(f"预测过期: {f['name']} — {f['status']}")

    if ccxt_soft_fail:
        has_binance_api_fail = any((not r["ok"]) and ("Binance" in str(r.get("name", ""))) for r in api_results)
        has_raw_kline_fail = any(not r["ok"] for r in ohlcv_checks)
        has_v5_gru_file_stale = any(
            (not f["ok"]) and (str(f.get("name", "")).startswith("Exp") or str(f.get("name", "")).startswith("GRU"))
            for f in file_checks
        )
        if has_binance_api_fail or has_raw_kline_fail or has_v5_gru_file_stale:
            all_ok = False
            alerts.append("ccxt无法连接Binance — v5/GRU链路受影响")
        else:
            print("  ⚠️ ccxt探测失败但链路新鲜（API/K线/预测文件正常），判定为瞬时网络抖动")

    if pending_api_failures:
        has_data_source_stale = any(not d["ok"] for d in ds_checks) or any(not d["ok"] for d in dq_checks)
        has_kline_stale = any(not r["ok"] for r in ohlcv_checks)
        has_prediction_stale = any(not f["ok"] for f in file_checks)
        daemon_or_collector_bad = (not daemon_check.get("daemon_running", False)) or (not daemon_check.get("cache_fresh", False)) or (not dc_ok)
        if has_data_source_stale or has_kline_stale or has_prediction_stale or daemon_or_collector_bad:
            all_ok = False
            for name in pending_api_failures:
                alerts.append(f"API不通: {name}")
        else:
            print(
                "  ⚠️ API探测出现瞬时失败，但 Daemon/采集器/数据源/预测文件均新鲜，"
                "判定为网络抖动（暂不作为故障）"
            )

    # 8. 合并交易进程 (9 个 multi_prediction_index)
    print(f"\n🔄 合并交易进程 (期望 {len(expected_groups_runtime)} 个进程, {expected_total_traders_runtime} 个 trader):")
    ts_result = check_ts_trading_processes()
    for g in ts_result["groups"]:
        if g["alive"]:
            print(f"  ✅ {g['name']:<22} PID={g['pid']:<8} {g['traders']} traders | 日志 {g['log_lines']} 行")
        else:
            all_ok = False
            print(f"  ❌ {g['name']:<22} 未运行! (期望 {g['traders']} traders)")
            alerts.append(f"合并进程未运行: {g['name']} ({g['traders']} traders)")
    print(f"  总计: {ts_result['total_groups_alive']}/{len(expected_groups_runtime)} 进程, "
          f"{ts_result['total_traders']}/{expected_total_traders_runtime} traders")
    if ts_result["old_processes"] > 0:
        all_ok = False
        print(f"  ⚠️  检测到 {ts_result['old_processes']} 个旧式独立交易进程(应已停止)!")
        alerts.append(f"旧式独立交易进程仍在运行: {ts_result['old_processes']} 个")

    # 8b. 运行时风控状态（数据质量门控 + DOWN熔断）
    print(f"\n🛡️ 运行时风控状态:")
    guards = check_runtime_trade_guards()
    if not guards["exists"]:
        all_ok = False
        for issue in guards["issues"]:
            print(f"  ❌ {issue}")
            alerts.append(f"运行时风控: {issue}")
    else:
        mode_icon = (
            "✅" if guards["mode"] == "ok"
            else ("⚠️" if guards["mode"] in {"soft_degraded", "hard_degraded"} else "❌")
        )
        if guards.get("guard_files"):
            print(f"  文件: {len(guards['guard_files'])} 个 ({', '.join(guards['guard_files'][:4])}{'...' if len(guards['guard_files']) > 4 else ''})")
        if guards.get("newest_guard_age_s") is not None:
            print(f"  文件时效: 最新文件 age={guards['newest_guard_age_s']:.1f}s")
        if guards.get("newest_payload_age_s") is not None:
            print(f"  payload时效: 最新 checkedAt age={guards['newest_payload_age_s']:.1f}s")
        print(
            f"  {mode_icon} 数据质量门控: {guards['mode']} | penalty={guards['penalty']:.3f} "
            f"| bet_scale={guards['bet_scale']:.2f} | api_err={guards['api_error_rate']:.3f} "
            f"| trigger={guards.get('trigger_class') or '-'} | reason={guards['reason'] or '-'}"
        )
        if guards.get("hold_until"):
            print(
                f"  恢复保持: until={guards['hold_until']} "
                f"(remaining={guards['hold_remaining_min']:.1f}m)"
            )
        if guards.get("key_source_errors"):
            print(f"  关键异常源: {', '.join(guards['key_source_errors'])}")
        if guards.get("non_key_source_errors"):
            print(f"  非关键异常源: {', '.join(guards['non_key_source_errors'][:6])}")
        if guards.get("covered_groups"):
            print(f"  覆盖 group: {len(guards['covered_groups'])} 个")
        if guards["stale_sources"]:
            print(f"  过期源: {', '.join(guards['stale_sources'])}")
        if guards.get("api_error_sources"):
            print(f"  API错误源: {', '.join(guards['api_error_sources'])}")
        if guards.get("v2_total_cells", 0) > 0:
            blocked = int(guards.get("v2_blocked_cells", 0))
            total_cells = int(guards.get("v2_total_cells", 0))
            enforce_cells = int(guards.get("v2_enforce_cells", 0))
            soft_cells = int(guards.get("v2_soft_cells", 0))
            hard_cells = int(guards.get("v2_hard_cells", 0))
            icon = "⚠️" if blocked > 0 else "✅"
            print(
                f"  {icon} DOWN风控V2: 封禁 {blocked}/{total_cells} cells "
                f"(enforce={enforce_cells}, soft={soft_cells}, hard={hard_cells})"
            )
            fast_hard = int(guards.get("down_fast_hard_count", 0))
            cross_layer = int(guards.get("cross_layer_escalation_count", 0))
            stale_actions = int(guards.get("stale_last_action_count", 0))
            print(
                f"  ℹ️ DOWN快触发证据: fast_hard={fast_hard}, cross_layer={cross_layer}, stale_last_action={stale_actions}"
            )
        elif guards["active_breakers"] > 0:
            print(
                f"  ⚠️ DOWN熔断: {guards['active_breakers']}/{guards['total_breakers']} 激活 "
                f"({', '.join(guards['active_names'])})"
            )
        else:
            print(f"  ✅ DOWN熔断: 0/{guards['total_breakers']} 激活")

        if guards.get("up_v2_total_cells", 0) > 0:
            blocked = int(guards.get("up_v2_blocked_cells", 0))
            total_cells = int(guards.get("up_v2_total_cells", 0))
            enforce_cells = int(guards.get("up_v2_enforce_cells", 0))
            soft_cells = int(guards.get("up_v2_soft_cells", 0))
            icon = "⚠️" if blocked > 0 else "✅"
            print(
                f"  {icon} UP风控V2: 封禁 {blocked}/{total_cells} cells "
                f"(enforce={enforce_cells}, soft={soft_cells})"
            )
        else:
            print("  ℹ️ UP风控V2: 未启用或无cell")
        if guards.get("calibration_total_cells", 0) > 0:
            total_cells = int(guards.get("calibration_total_cells", 0))
            enforce_cells = int(guards.get("calibration_enforce_cells", 0))
            shadow_cells = int(guards.get("calibration_shadow_cells", 0))
            active_cells = int(guards.get("calibration_active_cells", 0))
            icon = "⚠️" if enforce_cells > 0 else "ℹ️"
            print(
                f"  {icon} 校准状态V1: 激活 {active_cells}/{total_cells} cells "
                f"(enforce={enforce_cells}, shadow={shadow_cells})"
            )
        else:
            print("  ℹ️ 校准状态V1: 未启用或无cell")
        if guards.get("expectancy_total_cells", 0) > 0:
            total_cells = int(guards.get("expectancy_total_cells", 0))
            enforce_cells = int(guards.get("expectancy_enforce_cells", 0))
            shadow_cells = int(guards.get("expectancy_shadow_cells", 0))
            blocked = int(guards.get("expectancy_blocked_cells", 0))
            degraded = int(guards.get("expectancy_degraded_cells", 0))
            fast_blocked = int(guards.get("expect_fast_block_count", 0))
            manual_blocked = int(guards.get("expect_manual_block_count", 0))
            manual_degraded = int(guards.get("expect_manual_degrade_count", 0))
            icon = "⚠️" if blocked > 0 else "✅"
            print(
                f"  {icon} 近期OOS启停V1: blocked {blocked}/{total_cells} cells "
                f"(enforce={enforce_cells}, shadow={shadow_cells}, degraded={degraded})"
            )
            print(
                "  ℹ️ EXPECT证据: "
                f"fast_blocked={fast_blocked}, "
                f"manual_block={manual_blocked}, "
                f"manual_degrade={manual_degraded}"
            )
            mb = guards.get("manual_block_count_by_symbol_direction", {})
            md = guards.get("manual_degrade_count_by_symbol_direction", {})
            parts = []
            for key in sorted(set(list(mb.keys()) + list(md.keys()))):
                parts.append(f"{key}(B={int(mb.get(key, 0))},D={int(md.get(key, 0))})")
            if parts:
                print("  ℹ️ EXPECT manual分布: " + " | ".join(parts[:6]))
        else:
            print("  ℹ️ 近期OOS启停V1: 未启用或无cell")
        if guards.get("drift_total_cells", 0) > 0:
            total_cells = int(guards.get("drift_total_cells", 0))
            enforce_cells = int(guards.get("drift_enforce_cells", 0))
            shadow_cells = int(guards.get("drift_shadow_cells", 0))
            active_cells = int(guards.get("drift_active_cells", 0))
            drifted_cells = int(guards.get("drift_skip_cells", 0))
            icon = "⚠️" if drifted_cells > 0 else "✅"
            print(
                f"  {icon} 方向阈值漂移V1: 激活 {active_cells}/{total_cells} cells "
                f"(enforce={enforce_cells}, shadow={shadow_cells}, drifted={drifted_cells})"
            )
        else:
            print("  ℹ️ 方向阈值漂移V1: 未启用或无cell")
        if guards.get("meta_total_cells", 0) > 0:
            total_cells = int(guards.get("meta_total_cells", 0))
            enforce_cells = int(guards.get("meta_enforce_cells", 0))
            shadow_cells = int(guards.get("meta_shadow_cells", 0))
            icon = "⚠️" if enforce_cells > 0 else "ℹ️"
            print(
                f"  {icon} Meta-Label V1: cells {total_cells} "
                f"(enforce={enforce_cells}, shadow={shadow_cells})"
            )
        else:
            print("  ℹ️ Meta-Label V1: 未启用或无cell")
        if guards.get("selector_total_cells", 0) > 0:
            total_cells = int(guards.get("selector_total_cells", 0))
            enforce_cells = int(guards.get("selector_enforce_cells", 0))
            shadow_cells = int(guards.get("selector_shadow_cells", 0))
            ineligible = int(guards.get("selector_ineligible_cells", 0))
            icon = "⚠️" if ineligible > 0 else "✅"
            print(
                f"  {icon} 组合选择器V1: ineligible {ineligible}/{total_cells} cells "
                f"(enforce={enforce_cells}, shadow={shadow_cells})"
            )
        else:
            print("  ℹ️ 组合选择器V1: 未启用或无cell")
        combo_dir_total = int(guards.get("combo_directional_total", 0))
        if combo_dir_total > 0:
            print(
                "  ℹ️ COMBO方向化runtime: "
                f"total={combo_dir_total}, "
                f"active={int(guards.get('combo_directional_active', 0))}, "
                f"blocked={int(guards.get('combo_directional_blocked', 0))}"
            )
        if guards.get("ensemble_consensus_total_symbols", 0) > 0:
            total_symbols = int(guards.get("ensemble_consensus_total_symbols", 0))
            enforce_symbols = int(guards.get("ensemble_consensus_enforce_symbols", 0))
            shadow_symbols = int(guards.get("ensemble_consensus_shadow_symbols", 0))
            blocked_symbols = int(guards.get("ensemble_consensus_blocked_symbols", 0))
            blocked_effective = int(guards.get("ensemble_consensus_blocked_effective_symbols", 0))
            icon = "⚠️" if blocked_effective > 0 else "✅"
            print(
                f"  {icon} ensemble共识V1: blocked {blocked_effective}/{total_symbols} symbols "
                f"(enforce={enforce_symbols}, shadow={shadow_symbols}, raw_blocked={blocked_symbols})"
            )
        else:
            print("  ℹ️ ensemble共识V1: 未启用或无symbol")
        if guards.get("regime_gate_total_symbols", 0) > 0:
            total_symbols = int(guards.get("regime_gate_total_symbols", 0))
            enforce_symbols = int(guards.get("regime_gate_enforce_symbols", 0))
            shadow_symbols = int(guards.get("regime_gate_shadow_symbols", 0))
            active_symbols = int(guards.get("regime_gate_active_symbols", 0))
            icon = "⚠️" if active_symbols > 0 else "ℹ️"
            print(
                f"  {icon} regime门控V1: 激活 {active_symbols}/{total_symbols} symbols "
                f"(enforce={enforce_symbols}, shadow={shadow_symbols})"
            )
        else:
            print("  ℹ️ regime门控V1: 未启用或无symbol")
        print("  ℹ️ 模型老化复核V1: 已停用")
        if guards.get("shock_v2_total_cells", 0) > 0:
            blocked = int(guards.get("shock_v2_blocked_cells", 0))
            total_cells = int(guards.get("shock_v2_total_cells", 0))
            enforce_cells = int(guards.get("shock_v2_enforce_cells", 0))
            active_cells = int(guards.get("shock_v2_active_cells", 0))
            icon = "⚠️" if blocked > 0 else "✅"
            print(
                f"  {icon} 极端波动风控V2: 激活 {active_cells}/{total_cells} cells "
                f"(enforce={enforce_cells}, blocked={blocked})"
            )
            if blocked > 0:
                eta_by_cluster = (
                    guards.get("shock_v2_blocked_eta_by_cluster")
                    if isinstance(guards.get("shock_v2_blocked_eta_by_cluster"), dict)
                    else {}
                )
                eta_parts: List[str] = []
                for ck in _runtime_cluster_keys():
                    info = eta_by_cluster.get(ck) if isinstance(eta_by_cluster, dict) else None
                    if not isinstance(info, dict):
                        continue
                    count = int(info.get("count", 0))
                    if count <= 0:
                        continue
                    min_min = info.get("min_min")
                    max_min = info.get("max_min")
                    try:
                        min_v = float(min_min)
                        max_v = float(max_min)
                        if abs(max_v - min_v) < 0.1:
                            eta_parts.append(f"{ck}:{count}个(约{min_v:.0f}m)")
                        else:
                            eta_parts.append(f"{ck}:{count}个({min_v:.0f}-{max_v:.0f}m)")
                    except (TypeError, ValueError):
                        eta_parts.append(f"{ck}:{count}个(剩余未知)")
                if eta_parts:
                    print(f"  ℹ️ SHOCK封禁剩余(簇): {' | '.join(eta_parts)}")
                blocked_without_eta = int(guards.get("shock_v2_blocked_without_eta", 0))
                if blocked_without_eta > 0:
                    print(f"  ℹ️ SHOCK无剩余时间字段: {blocked_without_eta} cells")
        else:
            print("  ℹ️ 极端波动风控V2: 未启用或无cell")
        if guards.get("combo_pause_total_cells", 0) > 0:
            blocked = int(guards.get("combo_pause_blocked_cells", 0))
            total_cells = int(guards.get("combo_pause_total_cells", 0))
            enforce_cells = int(guards.get("combo_pause_enforce_cells", 0))
            active_cells = int(guards.get("combo_pause_active_cells", 0))
            icon = "⚠️" if blocked > 0 else "✅"
            print(
                f"  {icon} 组合级停机: 激活 {active_cells}/{total_cells} cells "
                f"(enforce={enforce_cells}, blocked={blocked})"
            )
        else:
            print("  ℹ️ 组合级停机: 未启用或无cell")
        cluster_breakdown = guards.get("cluster_breakdown") if isinstance(guards, dict) else None
        if isinstance(cluster_breakdown, dict):
            print("  簇级分解:")
            down_cb = cluster_breakdown.get("down") if isinstance(cluster_breakdown.get("down"), dict) else {}
            up_cb = cluster_breakdown.get("up") if isinstance(cluster_breakdown.get("up"), dict) else {}
            cal_cb = cluster_breakdown.get("calibration") if isinstance(cluster_breakdown.get("calibration"), dict) else {}
            exp_cb = cluster_breakdown.get("expectancy") if isinstance(cluster_breakdown.get("expectancy"), dict) else {}
            drift_cb = cluster_breakdown.get("drift") if isinstance(cluster_breakdown.get("drift"), dict) else {}
            meta_cb = cluster_breakdown.get("meta") if isinstance(cluster_breakdown.get("meta"), dict) else {}
            selector_cb = cluster_breakdown.get("selector") if isinstance(cluster_breakdown.get("selector"), dict) else {}
            shock_cb = cluster_breakdown.get("shock") if isinstance(cluster_breakdown.get("shock"), dict) else {}
            combo_cb = cluster_breakdown.get("combo") if isinstance(cluster_breakdown.get("combo"), dict) else {}
            for ck in _runtime_cluster_keys():
                d = down_cb.get(ck, {}) if isinstance(down_cb, dict) else {}
                u = up_cb.get(ck, {}) if isinstance(up_cb, dict) else {}
                cal = cal_cb.get(ck, {}) if isinstance(cal_cb, dict) else {}
                exp = exp_cb.get(ck, {}) if isinstance(exp_cb, dict) else {}
                drift = drift_cb.get(ck, {}) if isinstance(drift_cb, dict) else {}
                meta = meta_cb.get(ck, {}) if isinstance(meta_cb, dict) else {}
                sel = selector_cb.get(ck, {}) if isinstance(selector_cb, dict) else {}
                s = shock_cb.get(ck, {}) if isinstance(shock_cb, dict) else {}
                c = combo_cb.get(ck, {}) if isinstance(combo_cb, dict) else {}
                print(
                    f"    - {ck}: "
                    f"DOWN(e={int(d.get('enforce', 0))},s={int(d.get('shadow', 0))},soft={int(d.get('soft', 0))},hard={int(d.get('hard', 0))},blk={int(d.get('blocked', 0))}) "
                    f"UP(e={int(u.get('enforce', 0))},s={int(u.get('shadow', 0))},soft={int(u.get('soft', 0))},blk={int(u.get('blocked', 0))}) "
                    f"CAL(e={int(cal.get('enforce', 0))},s={int(cal.get('shadow', 0))},act={int(cal.get('active', 0))}) "
                    f"EXP(e={int(exp.get('enforce', 0))},s={int(exp.get('shadow', 0))},deg={int(exp.get('degraded', 0))},blk={int(exp.get('blocked', 0))}) "
                    f"DRIFT(e={int(drift.get('enforce', 0))},s={int(drift.get('shadow', 0))},act={int(drift.get('active', 0))},drift={int(drift.get('drifted', 0))}) "
                    f"META(e={int(meta.get('enforce', 0))},s={int(meta.get('shadow', 0))}) "
                    f"SEL(e={int(sel.get('enforce', 0))},s={int(sel.get('shadow', 0))},inelig={int(sel.get('ineligible', 0))}) "
                    f"SHOCK(e={int(s.get('enforce', 0))},s={int(s.get('shadow', 0))},act={int(s.get('active', 0))},blk={int(s.get('blocked', 0))}) "
                    f"COMBO(e={int(c.get('enforce', 0))},s={int(c.get('shadow', 0))},act={int(c.get('active', 0))},blk={int(c.get('blocked', 0))})"
                )
        if guards.get("cooldown_total_active", 0) > 0:
            scope = guards.get("cooldown_scope") or "unknown"
            print(
                f"  ℹ️ 冷却状态: active={int(guards.get('cooldown_total_active', 0))} "
                f"(symbol={int(guards.get('cooldown_symbol_active', 0))}, "
                f"symbol_direction={int(guards.get('cooldown_symbol_direction_active', 0))}, scope={scope})"
            )
        else:
            print("  ℹ️ 冷却状态: active=0")
        print(
            f"  ℹ️ 极端行情运行态: active_symbols={int(guards.get('mainline_extreme_active_symbols', 0))}/"
            f"{int(guards.get('mainline_extreme_total_symbols', 0))} "
            f"| 同方向连续亏损冷却={int(guards.get('direction_loss_cooldown_blocked', 0))}/"
            f"{int(guards.get('direction_loss_cooldown_total', 0))} "
            f"| 回撤加速度暂停={int(guards.get('drawdown_acceleration_blocked', 0))}/"
            f"{int(guards.get('drawdown_acceleration_total', 0))}"
        )
        ev = guards.get("layer_evidence_counts") if isinstance(guards.get("layer_evidence_counts"), dict) else {}
        print(
            "  生效证据计数: "
            f"CAL={int(ev.get('calibration', 0))}, "
            f"DRIFT={int(ev.get('threshold_drift', 0))}, "
            f"EXPECT={int(ev.get('expectancy', 0))}, "
            f"META={int(ev.get('meta_label', 0))}, "
            f"SHOCK={int(ev.get('shock', 0))}, "
            f"COMBO={int(ev.get('combo_pause', 0))}, "
            f"UP/DOWN={int(ev.get('up_down_risk', 0))}, "
            f"SELECTOR={int(ev.get('selector', 0))}, "
            f"REGIME={int(ev.get('regime', 0))}, "
            f"COOLDOWN={int(ev.get('cooldown', 0))}, "
            f"EXTREME={int(ev.get('extreme_market', 0))}, "
            f"DIR_LOSS={int(ev.get('direction_loss_cooldown', 0))}, "
            f"DD_ACCEL={int(ev.get('drawdown_acceleration', 0))}"
        )
        print(
            "  累计过滤计数: "
            f"evaluated={int(guards.get('evidence_evaluated', 0))}, "
            f"passed={int(guards.get('evidence_passed', 0))}, "
            f"quality={int(guards.get('evidence_skipped_quality', 0))}, "
            f"should_trade={int(guards.get('evidence_skipped_should_trade', 0))}, "
            f"drift={int(guards.get('evidence_skipped_drift', 0))}, "
            f"expect={int(guards.get('evidence_skipped_expectancy', 0))}, "
            f"meta={int(guards.get('evidence_skipped_meta', 0))}, "
            f"selector={int(guards.get('evidence_skipped_selector', 0))}, "
            f"regime={int(guards.get('evidence_skipped_regime', 0))}, "
            f"shock={int(guards.get('evidence_skipped_shock', 0))}, "
            f"combo={int(guards.get('evidence_skipped_combo', 0))}, "
            f"down={int(guards.get('evidence_skipped_down', 0))}, "
            f"up={int(guards.get('evidence_skipped_up', 0))}, "
            f"extreme={int(guards.get('evidence_skipped_extreme', 0))}, "
            f"dir_loss={int(guards.get('evidence_skipped_direction_loss', 0))}, "
            f"dd_accel={int(guards.get('evidence_skipped_drawdown_acceleration', 0))}"
        )
        if not bool(guards.get("evidence_dummy_eval_ready", False)):
            print(
                f"  ℹ️ 摆设层判定暂缓: evaluated<{int(RUNTIME_GUARD_EVIDENCE_MIN_EVAL)} "
                f"(当前 {int(guards.get('evidence_evaluated', 0))})"
            )
        print(
            "  近24h异常: "
            f"json_parse={int(guards.get('recent_json_parse_24h', 0))}, "
            f"unexpected_end={int(guards.get('recent_json_unexpected_end_24h', 0))}, "
            f"clob_400={int(guards.get('recent_clob_400_24h', 0))}, "
            f"clob_502={int(guards.get('recent_clob_502_24h', 0))}"
        )
        for issue in guards["issues"]:
            if "degraded" in issue:
                print(f"  ⚠️ {issue}")
                alerts.append(f"运行时风控: {issue}")
            elif "DOWN熔断激活" in issue:
                print(f"  ⚠️ {issue}")
                alerts.append(f"运行时风控: {issue}")
            elif "DOWN风控V2封禁" in issue:
                print(f"  ⚠️ {issue}")
                alerts.append(f"运行时风控: {issue}")
            elif "UP风控V2封禁" in issue:
                print(f"  ⚠️ {issue}")
                alerts.append(f"运行时风控: {issue}")
            elif "极端波动风控V2封禁" in issue:
                print(f"  ⚠️ {issue}")
                alerts.append(f"运行时风控: {issue}")
            elif "组合级停机封禁" in issue:
                print(f"  ⚠️ {issue}")
                alerts.append(f"运行时风控: {issue}")
            elif "恢复期剩余" in issue:
                print(f"  ⚠️ {issue}")
                alerts.append(f"运行时风控: {issue}")
            elif "自动降仓" in issue or "外部API错误率偏高" in issue:
                print(f"  ⚠️ {issue}")
                alerts.append(f"运行时风控: {issue}")
            else:
                all_ok = False
                print(f"  ❌ {issue}")
                alerts.append(f"运行时风控: {issue}")

    # 9. 重复进程检测
    print(f"\n🧪 风控超参报告（历史优化 + 当前任务状态）:")
    risk_jobs = check_risk_jobs()
    jobs = risk_jobs.get("jobs", {}) if isinstance(risk_jobs, dict) else {}
    if jobs:
        print("  说明: 这里的“任务状态”只表示优化任务是否在跑，不代表运行态是否已接入。")
        print("  说明: 下面的 DOWN/UP/SHOCK 簇结果来自历史优化报告，只用于回溯；当前 simulation 是否已接入，以后面的“V1.3 准入评估”和 runtime 状态为准。")
        print("  任务状态:")
        for key, label in (
            ("down", "DOWN"),
            ("up", "UP"),
            ("combo", "COMBO"),
            ("shock", "SHOCK"),
            ("calibration", "CAL"),
            ("expectancy", "EXPECT"),
            ("regime", "REGIME"),
            ("watch", "WATCH"),
        ):
            job = jobs.get(key, {}) if isinstance(jobs, dict) else {}
            running = bool(job.get("running"))
            icon = "⏳" if running else "ℹ️"
            latest_log = job.get("latest_log")
            latest_mtime = job.get("latest_log_mtime") or "-"
            if running:
                proc = (job.get("processes") or [{}])[0]
                print(
                    f"    - {label}: {icon} 运行中 pid={proc.get('pid')} "
                    f"etime={proc.get('etime')} | 最新日志 {latest_mtime}"
                )
            else:
                print(f"    - {label}: {icon} 未运行 | 最新日志 {latest_mtime}")
            if latest_log:
                print(f"      log: {latest_log}")
    tuning = check_risk_tuning_reports()
    fams = tuning.get("families", {}) if isinstance(tuning, dict) else {}
    if not fams:
        print("  ❌ 未读取到任何超参报告")
        all_ok = False
        alerts.append("超参报告缺失")
    else:
        for fam in ("down", "up", "shock"):
            f = fams.get(fam) if isinstance(fams, dict) else None
            if not isinstance(f, dict):
                continue
            if not f.get("present"):
                all_ok = False
                print(f"  ❌ {fam.upper()}: 报告缺失")
                alerts.append(f"{fam.upper()} 报告缺失")
                continue
            feasible = int(f.get("feasible_clusters", 0))
            total = int(f.get("total_clusters", 0))
            mtime = str(f.get("mtime") or "-")
            fam_job = jobs.get(fam, {}) if isinstance(jobs, dict) else {}
            fam_running = bool(fam_job.get("running")) if isinstance(fam_job, dict) else False
            historical_label = f"历史{fam.upper()}优化报告"
            if fam_running and feasible < total:
                icon = "⏳"
                print(
                    f"  {icon} {historical_label}: 达标簇 {feasible}/{total} | "
                    f"报告时间 {mtime} | 当前重跑中"
                )
            else:
                icon = "ℹ️" if feasible < total else "✅"
                print(f"  {icon} {historical_label}: 达标簇 {feasible}/{total} | 报告时间 {mtime}")
            clusters = f.get("clusters", {}) if isinstance(f.get("clusters"), dict) else {}
            for ck in ("default_BTC", "default_ETH", "70_BTC", "70_ETH"):
                c = clusters.get(ck, {}) if isinstance(clusters, dict) else {}
                if not isinstance(c, dict):
                    continue
                c_icon = "✅" if bool(c.get("feasible")) else "⚠️"
                c_pass = c.get("search_pass") or "-"
                viol = c.get("violations") if isinstance(c.get("violations"), list) else []
                viol_text = ",".join(str(v) for v in viol[:2]) if viol else "-"
                extra = ""
                if fam == "shock":
                    cov_raw = c.get("trigger_fold_coverage")
                    nz = c.get("nonzero_fold_count")
                    fc = c.get("fold_count")
                    try:
                        cov = float(cov_raw)
                        extra = f" cov={cov:.3f} ({int(nz or 0)}/{int(fc or 0)})"
                    except Exception:
                        extra = ""
                print(
                    f"    - {ck}: {c_icon} feasible={bool(c.get('feasible'))} "
                    f"pass={c_pass} viol={viol_text}{extra}"
                )
            # 旧超参报告仅用于历史对照，不再作为接入层面的硬告警；
            # 接入是否达标以 V1.3 准入评估（A/B/C）为准。
        for issue in tuning.get("issues", []):
            all_ok = False
            print(f"  ❌ {issue}")
            alerts.append(f"超参报告: {issue}")

    admission = check_prediction_v13_admission()
    if admission.get("present"):
        print(
            f"  ✅ V1.3 准入评估: {admission.get('mtime') or '-'} "
            f"| 文件: {admission.get('path')}"
        )
        layers = admission.get("layers", {}) if isinstance(admission.get("layers"), dict) else {}
        layer_labels = (
            ("calibration", "CAL"),
            ("expectancy", "EXPECT"),
            ("shock", "SHOCK"),
            ("ensemble_consensus", "ENS"),
            ("threshold_drift", "DRIFT"),
            ("meta_label", "META"),
            ("selector", "SELECTOR"),
            ("regime", "REGIME"),
            ("joint_optimization", "JOINT"),
        )
        for lk, label in layer_labels:
            li = layers.get(lk, {}) if isinstance(layers, dict) else {}
            if not isinstance(li, dict) or int(li.get("total", 0)) == 0:
                continue
            clusters = li.get("clusters", {}) if isinstance(li.get("clusters"), dict) else {}
            a = int(li.get("A", 0)); b = int(li.get("B", 0)); c = int(li.get("C", 0))
            ready = int(li.get("ready", 0)); total = int(li.get("total", 0))
            c_clusters: List[dict] = []
            if c > 0:
                for _, cinfo in clusters.items():
                    if not isinstance(cinfo, dict):
                        continue
                    if str(cinfo.get("tier", "")).upper() == "C":
                        c_clusters.append(cinfo)
            c_clusters_all_off = bool(c_clusters) and all(
                str(ci.get("mode", "")).strip().lower() == "off" for ci in c_clusters
            )
            selector_off_by_design = (
                lk == "selector"
                and total > 0
                and c == total
                and c_clusters_all_off
                and int((guards or {}).get("selector_enforce_cells", 0)) == 0
            )
            if selector_off_by_design:
                icon = "ℹ️"
            else:
                icon = "✅" if c == 0 else ("⚠️" if (a + b) > 0 else "❌")
            print(f"    - {label}: {icon} ready={ready}/{total} | A={a} B={b} C={c}")
            for ck in (
                "default", "70",
                "default_BTC", "default_ETH", "70_BTC", "70_ETH",
                "default_BTC_UP", "default_BTC_DOWN", "default_ETH_UP", "default_ETH_DOWN",
                "70_BTC_UP", "70_BTC_DOWN", "70_ETH_UP", "70_ETH_DOWN",
            ):
                cinfo = clusters.get(ck)
                if not isinstance(cinfo, dict):
                    continue
                tier = cinfo.get("tier", "C")
                mode = cinfo.get("mode", "-")
                ready_flag = bool(cinfo.get("ready"))
                viol = cinfo.get("violations") if isinstance(cinfo.get("violations"), list) else []
                viol_text = ",".join(str(v) for v in viol[:2]) if viol else "-"
                print(f"      · {ck}: tier={tier} ready={ready_flag} mode={mode} viol={viol_text}")
            if c > 0:
                if lk == "ensemble_consensus":
                    mode_map = guards.get("ensemble_consensus_mode_map", {}) if isinstance(guards, dict) else {}
                    mismatched = 0
                    disabled = 0
                    for ck, cinfo in clusters.items():
                        if not isinstance(cinfo, dict):
                            continue
                        if str(cinfo.get("tier") or "").upper() != "C":
                            continue
                        parts = str(ck).split("_")
                        if len(parts) < 2:
                            continue
                        profile = parts[0]
                        symbol = parts[1]
                        if profile not in {"default", "70"} or symbol not in {"BTC", "ETH"}:
                            continue
                        disabled += 1
                        runtime_mode = str(mode_map.get(f"{profile}_{symbol}", "")).strip().lower()
                        if runtime_mode != "off":
                            mismatched += 1
                    if mismatched > 0:
                        alerts.append(
                            f"{label} C档簇未按off落地: {mismatched}/{max(disabled, 1)}"
                        )
                    else:
                        print(f"      · {label} C档簇运行态已按 off 落地")
                elif c_clusters_all_off:
                    if selector_off_by_design:
                        print("      · SELECTOR 当前为 off（按设计未启用），不视为运行故障")
                    elif lk == "selector" and int((guards or {}).get("selector_enforce_cells", 0)) > 0:
                        print(
                            "      · SELECTOR 已由 Core/Shadow overlay 接管运行态，"
                            f"当前 enforce={int((guards or {}).get('selector_enforce_cells', 0))} "
                            f"ineligible={int((guards or {}).get('selector_ineligible_cells', 0))}"
                        )
                    else:
                        print(f"      · {label} C档簇按 mode=off 禁用（符合准入表）")
                else:
                    alerts.append(f"{label} 存在 C 档禁用簇: {c}/{total}")
        not_ready_rows = admission.get("current_not_ready") if isinstance(admission, dict) else []
        if isinstance(not_ready_rows, list):
            blocking_now = [x for x in not_ready_rows if isinstance(x, dict) and bool(x.get("is_blocking_now"))]
            n_a_rows = [x for x in not_ready_rows if isinstance(x, dict) and bool(x.get("n_a_by_design"))]
            print(
                f"  当前阻断清单: blocking={len(blocking_now)} "
                f"| n_a_by_design={len(n_a_rows)}"
            )
            for row in blocking_now[:8]:
                layer = row.get("layer")
                cluster = row.get("cluster")
                tier = row.get("tier")
                mode = row.get("mode")
                root = row.get("root_cause_type")
                viol = row.get("violations") if isinstance(row.get("violations"), list) else []
                viol_text = ",".join(str(v) for v in viol[:2]) if viol else "-"
                print(
                    f"    - {layer}/{cluster}: tier={tier} mode={mode} "
                    f"root={root} viol={viol_text}"
                )
            if len(blocking_now) > 8:
                print(f"    - ... 其余 {len(blocking_now)-8} 条见 v13 admission json")
    else:
        issues = admission.get("issues") if isinstance(admission, dict) else []
        if issues:
            print("  ⚠️ V1.3 准入评估缺失/异常:")
            for issue in issues[:3]:
                print(f"    - {issue}")

    # 旧慢周期接入状态已降级为归档链，主健康检查不再混入显示。

    drawdown_rootcause = check_recent_drawdown_rootcause()
    if drawdown_rootcause.get("present"):
        gen_at = drawdown_rootcause.get("generated_at") or "-"
        o_delta = drawdown_rootcause.get("overall_delta_pnl")
        d_delta = drawdown_rootcause.get("default_delta_pnl")
        p70_delta = drawdown_rootcause.get("profile70_delta_pnl")

        def _fmt_p(v: Any) -> str:
            try:
                return f"{float(v):+.2f}"
            except Exception:
                return "-"

        print(f"  📉 近2天回撤归因: {gen_at}")
        print(
            f"    - delta pnl: overall={_fmt_p(o_delta)} "
            f"default={_fmt_p(d_delta)} 70={_fmt_p(p70_delta)}"
        )
        b = drawdown_rootcause.get("top_bucket") if isinstance(drawdown_rootcause.get("top_bucket"), dict) else None
        if b:
            print(
                "    - 主拖累方向簇: "
                f"{b.get('profile')}/{b.get('symbol')}_{b.get('direction')} "
                f"{_fmt_p(b.get('delta_pnl_sum'))}"
            )
        worst = drawdown_rootcause.get("top_worst_cells") if isinstance(drawdown_rootcause.get("top_worst_cells"), list) else []
        for row in worst[:3]:
            try:
                print(
                    "      · "
                    f"{row.get('profile')}/{row.get('trader')}-{row.get('symbol')} "
                    f"delta={_fmt_p(row.get('delta_pnl'))}"
                )
            except Exception:
                continue
    else:
        for issue in (drawdown_rootcause.get("issues") or [])[:1]:
            print(f"  ⚠️ 近2天回撤归因: {issue}")

    # 9. 重复进程检测
    print(f"\n🔍 重复进程检测:")
    dupes = check_duplicate_processes()
    if not dupes:
        print(f"  ✅ 无重复进程")
    else:
        for d in dupes:
            all_ok = False
            print(f"  ❌ {d['type']} [{d['key']}]: {d['count']}个重复 PIDs={d['pids']}")
            alerts.append(f"重复进程: {d['type']} {d['key']} ({d['count']}个)")

    # 9b. 僵尸进程检测（state Z = 已退出但父进程未 wait 的子进程）
    print(f"\n🧟 僵尸进程检测:")
    zombies = check_zombie_processes()
    if not zombies:
        print(f"  ✅ 无僵尸进程")
    else:
        zc = classify_zombie_processes(zombies)
        core = zc.get("core_blocking") or []
        orphan = zc.get("external_orphan") or []
        other = zc.get("other") or []
        print(
            f"  分类: core_blocking={sum(int(x.get('count', 0)) for x in core)} "
            f"external_orphan={sum(int(x.get('count', 0)) for x in orphan)} "
            f"other={sum(int(x.get('count', 0)) for x in other)}"
        )
        for row in core:
            print(f"  ❌ {row.get('count')} 个僵尸 — 父进程 PPID={row.get('ppid')} (核心链路): {str(row.get('parent_cmd') or '')[:65]}…")
        for row in orphan:
            print(f"  ⓪ {row.get('count')} 个僵尸 — 父进程 PPID={row.get('ppid')} (外部孤儿): {str(row.get('parent_cmd') or '')[:65]}")
        for row in other:
            print(f"  ⓪ {row.get('count')} 个僵尸 — 父进程 PPID={row.get('ppid')}: {str(row.get('parent_cmd') or '')[:65]}")
        if core:
            all_ok = False
            alerts.append(f"僵尸进程(core): {sum(int(x.get('count', 0)) for x in core)} 个")
        print(f"  💡 清理: python3 scripts/kill_zombie_parents.py --apply")

    # 10. 全量 WebSocket + REST 超时
    daemon_active = daemon_check["daemon_running"] and daemon_check["cache_fresh"]
    print(f"\n🌐 WebSocket & REST 全量扫描{' (Daemon模式: per-process WS 已禁用)' if daemon_active else ''}:")
    ws_stats = check_websocket_health_all()
    print(f"  WebSocket: ✅已连接 {ws_stats['ws_connected']} | "
          f"❌断开 {ws_stats['ws_disconnected']} | "
          f"⓪未知 {ws_stats['ws_unknown']}")
    print(f"  REST超时: {ws_stats['rest_timeouts']}条 | "
          f"网络错误: {ws_stats['rest_network_errors']}条")
    if not daemon_active and ws_stats["ws_disconnected"] > 5:
        all_ok = False
        alerts.append(f"WebSocket大面积断开: {ws_stats['ws_disconnected']}个")
        print(f"  ⚠️  断开目录: {', '.join(ws_stats['disconnected_dirs'][:5])}...")
    if ws_stats["rest_timeouts"] > 50:
        all_ok = False
        alerts.append(f"REST大量超时: {ws_stats['rest_timeouts']}条")
    if ws_stats["timeout_dirs"]:
        top3 = ws_stats["timeout_dirs"][:3]
        print(f"  超时TOP: {', '.join(f'{d}({c}条)' for d, c in top3)}")

    # 11. 代理连接数
    print(f"\n🔗 代理连接:")
    conn_count, conn_msg = check_proxy_connections()
    print(f"  {conn_msg}")
    if conn_count > 500:
        alerts.append(f"代理连接数过高: {conn_count} (可能过载)")

    # 12. 日志异常模式
    print(f"\n🐛 日志异常检测:")
    anomalies = check_log_anomalies()
    if not anomalies:
        print(f"  ✅ 无严重异常")
    else:
        for a in anomalies[:10]:
            sev = str(a.get("severity", "error")).lower()
            if sev == "warn":
                print(f"  ⚠️ [{a['type']}] {a['file']} ({a['count']}次)")
            else:
                all_ok = False
                print(f"  ❌ [{a['type']}] {a['file']} ({a['count']}次)")
                alerts.append(f"日志异常: {a['file']} — {a['type']}({a['count']}次)")

    # 13. 融合参考产物（信息级；不再视为在线模拟交易链）
    print(f"\n🧬 融合参考产物:")
    ens_check = check_ensemble_output()
    if not ens_check.get("exists"):
        print("  ℹ️ 未生成参考产物（已不参与在线交易健康判定）")
    elif ens_check["ok"]:
        print(f"  ℹ️ 参考输出存在，结构正常")
        print(f"     更新: {ens_check['timestamp'][:19]}")
    else:
        print("  ℹ️ 参考产物存在，但已降级为信息项：")
        for issue in ens_check["issues"]:
            print(f"     - {issue}")

    # 14. 交易执行统计
    print(f"\n💰 交易执行:")
    trade_stats = check_recent_trades()
    print(f"  有交易记录: {trade_stats['with_trades']}个目录 | "
          f"无记录: {trade_stats['no_trades']}个 | "
          f"近1小时成交: {trade_stats['total_recent']}笔")

    # 15. 增量训练调度器（当前默认 30m + 12/8/20）状态
    print(f"\n🕒 增量训练调度器:")
    sched = check_incremental_training_scheduler(expected_interval_minutes=expected_interval_minutes)
    if sched["running"]:
        if sched.get("has_cycle_history_24h"):
            training_timeout_display = str(int(sched.get("training_timeout_24h") or 0))
            if str(sched.get("history_source") or "") == "grouped_online_learning" and int(sched.get("legacy_cycle_timeout_24h") or 0) > 0:
                training_timeout_display += f" (legacy_cycle={int(sched.get('legacy_cycle_timeout_24h') or 0)})"
        else:
            training_timeout_display = f"- (legacy={int(sched.get('legacy_timeout_24h') or sched.get('timeout_24h') or 0)})"
        history_source = str(sched.get("history_source") or "-")
        print(
            f"  ✅ 运行中 PID={sched['pid']} | interval={sched['interval_minutes']}m | "
            f"树数={sched['base_rounds']}/{sched['calm_rounds']}/{sched['shock_rounds']} | "
            f"shock={sched['shock_vol_ratio']}/{sched['shock_ret_q']}"
        )
        print(
            f"  合同: launchd_loaded={'yes' if sched.get('launchd_loaded') else 'no'} | "
            f"scheduler_process_running={'yes' if sched.get('scheduler_process_running') else 'no'} | "
            f"scheduler_pid_file_consistent="
            f"{'yes' if sched.get('scheduler_pid_file_consistent') else 'no' if sched.get('scheduler_pid_file_consistent') is not None else '-'}"
        )
        if sched["command"]:
            cmd_show = sched["command"][:140]
            suffix = "..." if len(sched["command"]) > 140 else ""
            print(f"  命令: {cmd_show}{suffix}")
        dur = sched.get("last_train_duration_sec")
        dur_txt = "-" if not isinstance(dur, int) else (f"{dur}s" if dur < 60 else f"{dur//60}m{dur%60:02d}s")
        repaired_pending = bool(sched.get("runtime_apply_repair_pending_verification"))
        result_class = "ok_with_observation" if repaired_pending else str(sched.get("latest_result_class") or "-")
        runtime_apply_txt = (
            "repaired_pending_verification"
            if repaired_pending
            else "-"
            if sched.get("latest_runtime_apply_success") is None
            else "ok"
            if bool(sched.get("latest_runtime_apply_success"))
            else "failed"
        )
        runtime_apply_failure_display = str(int(sched.get("runtime_apply_failure_24h") or 0))
        if repaired_pending:
            runtime_apply_failure_display += " (historical_repaired)"
        param_search_state = str(sched.get("latest_param_search_cadence_state") or sched.get("latest_param_search_status") or "-")
        param_search_cadence = sched.get("param_search_cadence_minutes")
        param_search_backlog = int(
            sched.get("latest_night_search_backlog_total")
            or sched.get("latest_param_search_backlog_windows")
            or 0
        )
        param_search_next_due = sched.get("latest_param_search_next_due_at") or "-"
        print(
            f"  近24h: success={int(sched.get('success_24h') or 0)} | "
            f"training_timeout={training_timeout_display} | "
            f"runtime_apply_failure={runtime_apply_failure_display} | "
            f"refresh_timeout={int(sched.get('post_training_refresh_timeout_24h') or 0)} | "
            f"nonblocking_refresh_status={sched.get('latest_nonblocking_refresh_status') or '-'} | "
            f"param_search_timeout={int(sched.get('param_search_nonblocking_timeout_24h') or 0)} | "
            f"param_search={param_search_state} "
            f"(status={sched.get('latest_param_search_status') or '-'} "
            f"cadence={param_search_cadence or '-'}m "
            f"backlog={param_search_backlog} "
            f"next_due={param_search_next_due}) | "
            f"night_search_completion={sched.get('night_search_completion_status') or '-'} | "
            f"apply_effective={sched.get('apply_effective_status') or '-'} | "
            f"result_class={result_class} | "
            f"latest_runtime_apply={runtime_apply_txt} | "
            f"history_source={history_source} | "
            f"last_train_duration={dur_txt}"
            + (f" | next_run_at={sched.get('next_run_at')}" if sched.get("next_run_at") else "")
        )
    else:
        print("  ❌ 未运行")
    if sched["issues"]:
        for issue in sched["issues"]:
            all_ok = False
            print(f"  ❌ {issue}")
            alerts.append(f"调度器异常: {issue}")
    for warning in sched.get("warnings", []):
        print(f"  ⚠️ {warning}")
    stale_dry_runs = check_stale_dry_run_processes(max_age_min=60)
    if stale_dry_runs:
        all_ok = False
        for p in stale_dry_runs[:5]:
            msg = f"发现长驻 dry-run 进程 PID={p['pid']} 运行 {p['age_min']}m"
            print(f"  ❌ {msg}")
            print(f"     {p['cmd']}")
            alerts.append(f"调度器异常: {msg}")
    else:
        print("  ✅ 无长驻 dry-run 训练进程")

    print(f"\n🎯 主线运行池:")
    mainline = check_mainline_runtime_pool_status()
    slot_scoped_health = _slot_scoped_live_health_only_enabled()
    slot_scoped_targets = _load_slot_scoped_live_targets() if slot_scoped_health else []
    slot_scoped_target_symbols = sorted({
        str((row or {}).get("symbol") or "").strip().upper()
        for row in slot_scoped_targets
        if str((row or {}).get("symbol") or "").strip()
    })
    if mainline["available"]:
        if slot_scoped_health and slot_scoped_targets:
            print(
                f"  规模: traders={len(slot_scoped_targets)} "
                f"cells={len(slot_scoped_targets)} "
                f"symbols={','.join(slot_scoped_target_symbols) or '-'}"
            )
            print("  范围: 仅 slot1/slot2 真钱链 + 必要 ETH 共享链 (BTC/XRP 已停用并移出健康范围)")
            print(
                f"  运行组: lowprice_70_selected "
                f"target_cells={mainline.get('runtime_cells_with_mainline_target')} "
                f"using_expected_family={mainline.get('runtime_cells_using_expected_mainline_family')}"
            )
            print(
                f"  审计: rules_drift_count={mainline.get('rules_json_drift_count')} "
                f"source_drift_count={mainline.get('source_drift_count')} "
                f"model_version_mismatch_count={mainline.get('prediction_model_version_mismatch_count')} "
                f"runtime_cells_missing_process={mainline.get('runtime_cells_missing_process')} "
                f"writer_process_missing_count={mainline.get('writer_process_missing_count')}"
            )
            print(
                f"  增量训练: jobs_total={mainline.get('incremental_jobs_total')} "
                f"jobs_by_symbol={mainline.get('incremental_jobs_by_symbol') or {}} "
                f"writeback={mainline.get('successful_writeback_jobs')} "
                f"veto_only={mainline.get('veto_only_jobs')} "
                f"failed={mainline.get('failed_jobs')} "
                f"skipped={mainline.get('skipped_jobs')} "
                f"stalled_jobs={mainline.get('incremental_stalled_jobs')} "
                f"invalid_bootstrap_jobs={mainline.get('invalid_bootstrap_jobs')}"
            )
            print(
                f"  最新修订: mainline_target={mainline.get('mainline_latest_revision_target_cells')} "
                f"mainline_loaded={mainline.get('mainline_latest_revision_loaded_cells')} "
                f"candidate_loaded={mainline.get('candidate_latest_revision_loaded_cells')} "
                f"sync={mainline.get('latest_revision_sync_state')} "
                f"runtime_gap={mainline.get('latest_revision_runtime_sync_gap_cells')} "
                f"header_lag={mainline.get('prediction_header_lag_cells')} "
                f"prediction_uses_latest_revision={mainline.get('prediction_uses_latest_revision')}"
            )
            print(
                f"  补训状态: needs_catchup={mainline.get('incremental_needs_catchup')} "
                f"missing_slots={mainline.get('incremental_missing_slots')} "
                f"historical_missing_slots={mainline.get('incremental_historical_missing_slots')} "
                f"conclusion={mainline.get('current_conclusion') or '-'}"
            )
        else:
            print(
                f"  规模: active_traders={mainline.get('active_traders_total')} "
                f"active_cells={mainline.get('active_cells_total')} "
                f"active_symbols={','.join(mainline.get('active_symbols') or []) or '-'}"
            )
            print(
                f"  来源/审计: rules_drift_count={mainline.get('rules_json_drift_count')} "
                f"source_drift_count={mainline.get('source_drift_count')} "
                f"model_version_mismatch_count={mainline.get('prediction_model_version_mismatch_count')} "
                f"runtime_cells_missing_process={mainline.get('runtime_cells_missing_process')} "
                f"writer_process_missing_count={mainline.get('writer_process_missing_count')} "
                f"target_cells={mainline.get('runtime_cells_with_mainline_target')} "
                f"using_expected_family={mainline.get('runtime_cells_using_expected_mainline_family')}"
            )
            print(
                f"  增量训练: jobs_total={mainline.get('incremental_jobs_total')} "
                f"jobs_by_symbol={mainline.get('incremental_jobs_by_symbol') or {}} "
                f"writeback={mainline.get('successful_writeback_jobs')} "
                f"veto_only={mainline.get('veto_only_jobs')} "
                f"failed={mainline.get('failed_jobs')} "
                f"skipped={mainline.get('skipped_jobs')} "
                f"stalled_jobs={mainline.get('incremental_stalled_jobs')} "
                f"invalid_bootstrap_jobs={mainline.get('invalid_bootstrap_jobs')} "
                    f"mainline_latest_revision_target_cells={mainline.get('mainline_latest_revision_target_cells')} "
                    f"mainline_latest_revision_loaded_cells={mainline.get('mainline_latest_revision_loaded_cells')} "
                    f"candidate_latest_revision_loaded_cells={mainline.get('candidate_latest_revision_loaded_cells')} "
                    f"latest_revision_sync_state={mainline.get('latest_revision_sync_state')} "
                    f"latest_revision_runtime_sync_gap_cells={mainline.get('latest_revision_runtime_sync_gap_cells')} "
                    f"latest_revision_lag_cells={mainline.get('latest_revision_lag_cells')} "
                    f"prediction_header_lag_cells={mainline.get('prediction_header_lag_cells')} "
                    f"prediction_uses_latest_revision={mainline.get('prediction_uses_latest_revision')}"
            )
            print(
                f"  选择器: zero_rows_default={mainline.get('selector_zero_rows_default')} "
                f"zero_rows_70={mainline.get('selector_zero_rows_70')} "
                f"overstrict_rows={mainline.get('selector_overstrict_rows_total')} "
                f"active_simulation_scope_rows={mainline.get('selector_active_simulation_scope_rows_total') or mainline.get('selector_active_shadow_scope_rows_total')} "
                f"xrp_runtime_unknown={mainline.get('selector_xrp_runtime_unknown_cells_total')}"
            )
            print(
                f"  70阈值: btceth_selected={mainline.get('filter70_btceth_threshold_selected')} "
                f"xrp_selected={mainline.get('filter70_xrp_threshold_selected')} "
                f"constraints_passed={mainline.get('filter70_constraints_passed')}"
            )
            print(
                f"  状态分布: mainline_active={mainline.get('mainline_active_cells')} "
                f"candidate_only={mainline.get('candidate_only_cells')} "
                f"n_a_by_design={mainline.get('n_a_by_design_cells')}"
            )
            print(
                f"  覆盖/执行器: xrp_guard_coverage={mainline.get('xrp_guard_coverage')} "
                f"executor_env={mainline.get('executor_env_audit_status')} "
                f"guard_compare={mainline.get('guard_executor_compare_status')} "
                f"drawdown_halt_consumed={mainline.get('drawdown_halt_executor_consumed')} "
                f"sol_status={mainline.get('sol_status')}"
            )
            print(
                f"  补训状态: needs_catchup={mainline.get('incremental_needs_catchup')} "
                f"missing_slots={mainline.get('incremental_missing_slots')} "
                f"historical_missing_slots={mainline.get('incremental_historical_missing_slots')} "
                f"conclusion={mainline.get('current_conclusion') or '-'}"
            )
    else:
        print("  ⚠️  尚未生成主线运行池审计")
    for issue in mainline.get("issues", []):
        all_ok = False
        print(f"  ❌ {issue}")
        alerts.append(f"主线运行池: {issue}")

    if heal_actions:
        print(f"\n🛠️ 自动自愈(--auto-heal):")
        for line in heal_actions:
            print(f"  {line}")

    # ─── 醒目告警 ───
    print(f"\n{'═' * 70}")
    if all_ok and not alerts:
        print(f"  ✅ 全部正常 (核心检查通过)")
    else:
        print(f"  🚨🚨🚨  发现 {len(alerts)} 个异常  🚨🚨🚨")
        print(f"{'─' * 70}")
        for i, alert in enumerate(alerts, 1):
            print(f"  [{i}] ❌ {alert}")
        print(f"{'─' * 70}")
        print(f"  影响评估:")
        api_down = any("API不通" in a or "ccxt无法连接Binance" in a or "代理不可用" in a for a in alerts)
        writer_down = any("进程缺失" in a for a in alerts)
        file_stale = any("预测过期" in a for a in alerts)
        ws_issue = any("WebSocket" in a for a in alerts)
        dupe_issue = any("重复" in a for a in alerts)
        if api_down:
            print(f"    → API/代理故障: v5和GRU预测器无法获取K线数据，预测文件将停止更新")
        if writer_down:
            print(f"    → 写入器宕机: 对应模型的预测文件不会更新，融合模型将缺少数据源")
        if file_stale and not api_down and not writer_down:
            print(f"    → 文件过期: 可能是临时延迟，持续过期则表示写入器已挂")
        if ws_issue:
            print(f"    → WebSocket断连: 挂单监控失效，交易可能无法在最优价格成交")
        if dupe_issue:
            print(f"    → 重复进程: 可能导致预测文件互相覆盖、交易重复下单")
        data_stale = any("数据源过期" in a for a in alerts)
        if data_stale:
            print(f"    → 数据源过期: Funding/OI/LS/PM 特征陈旧，预测准确性可能下降")
        raw_stale = any("本地K线过期" in a for a in alerts)
        if raw_stale:
            print(f"    → 本地K线过期: 15m/1h/4h 未更新到最新，预测器可能用旧K线预测下一根")
        if api_down or writer_down:
            print(f"    → 融合模型影响: 活跃源减少，但仍会使用可用数据正常融合")
        root = classify_root_causes(alerts)
        print(f"{'─' * 70}")
        print("  三项优先级根因归类:")
        for title in ("P0 网络/VPN链路", "P1 逻辑/数据链路", "P2 进程/重启治理"):
            items = root.get(title, [])
            if not items:
                continue
            print(f"    {title}:")
            for msg in items[:6]:
                print(f"      - {msg}")
            if len(items) > 6:
                print(f"      - ...其余 {len(items) - 6} 条")
        print("  排查顺序: 先 P0，再 P1，最后 P2。")
    print(f"{'═' * 70}")

    # ─── 在线学习状态 ──────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("  📚 在线增量学习状态")
    print(f"{'─' * 70}")
    ol_log = PROJECT_ROOT / "logs" / "online_learning_history.jsonl"
    if ol_log.exists():
        import json as _json
        lines = ol_log.read_text().strip().splitlines()
        if lines:
            parsed = []
            for raw in reversed(lines):
                try:
                    parsed.append(_json.loads(raw))
                except Exception:
                    continue
            grouped = next(
                (
                    row
                    for row in parsed
                    if row.get("entry_type") == "grouped_online_learning"
                    and str(row.get("mode") or "") == "mainline_runtime_pool"
                ),
                None,
            )
            if not isinstance(grouped, dict):
                grouped = next(
                    (
                        row
                        for row in parsed
                        if row.get("entry_type") == "grouped_online_learning"
                        and str(row.get("mode") or "") == "core10_only"
                    ),
                    None,
            )
            if isinstance(grouped, dict):
                ts = grouped.get("timestamp", "?")
                grouped_summary = grouped.get("summary", {}) if isinstance(grouped.get("summary"), dict) else {}
                summary = {
                    "jobs_total": int(mainline.get("incremental_jobs_total") or grouped_summary.get("jobs_total", 0) or 0),
                    "successful_writeback_jobs": int(
                        mainline.get("incremental_writeback_success_24h")
                        or grouped_summary.get("successful_writeback_jobs", grouped_summary.get("success", 0))
                        or 0
                    ),
                    "veto_only_jobs": int(grouped_summary.get("veto_only_jobs", 0) or 0),
                    "invalid_bootstrap_jobs": int(mainline.get("core10_invalid_bootstrap_jobs") or grouped_summary.get("invalid_bootstrap_jobs", 0) or 0),
                    "failed": int(grouped_summary.get("failed", 0) or 0),
                    "skipped": int(grouped_summary.get("skipped", 0) or 0),
                }
                raw_mode = mainline.get("incremental_training_mode") or grouped.get("mode") or "mainline_runtime_pool"
                mode = "mainline_runtime_pool" if raw_mode == "core10_only" else raw_mode
                print(f"    最近运行: {ts} ({mode})")
                print(
                    f"    结果: jobs={summary.get('jobs_total', 0)} "
                    f"writeback={summary.get('successful_writeback_jobs', 0)} "
                    f"veto_only={summary.get('veto_only_jobs', 0)} "
                    f"invalid_bootstrap={summary.get('invalid_bootstrap_jobs', 0)} "
                    f"failed={summary.get('failed', 0)} "
                    f"skipped={summary.get('skipped', 0)}"
                )
                jobs = grouped.get("jobs") if isinstance(grouped.get("jobs"), list) else []
                if jobs:
                    show = []
                    for row in jobs:
                        rc = row.get("returncode", 1)
                        try:
                            rc_int = int(1 if rc is None else rc)
                        except Exception:
                            rc_int = 1
                        child_summary = row.get("child_summary") if isinstance(row.get("child_summary"), dict) else {}
                        child_success = int(child_summary.get("success", 0) or 0)
                        child_veto = int(child_summary.get("trained_but_not_written", 0) or 0)
                        child_skipped = int(child_summary.get("skipped", 0) or 0)
                        skipped_reason = str(row.get("skipped_reason") or "")
                        if skipped_reason == "SKIP_INVALID_BOOTSTRAP":
                            status = "SKIP_INVALID_BOOTSTRAP"
                        elif bool(row.get("skipped")) and not child_summary:
                            status = "CADENCE_SKIP"
                        elif rc_int != 0:
                            status = "FAIL"
                        elif child_success > 0 and child_veto > 0:
                            status = "MIXED"
                        elif child_success > 0:
                            status = "OK"
                        elif child_veto > 0:
                            status = "VETO_ONLY"
                        elif child_skipped > 0:
                            status = "CHILD_SKIP"
                        else:
                            status = "OK"
                        show.append(
                            f"{row.get('job_id')}="
                            f"{status}"
                        )
                    print(f"    {len(jobs)}个Core10 job: " + ", ".join(show))
            else:
                print("    最近运行: 未找到主线分组训练历史")
                print(
                    f"    当前口径: jobs={int(mainline.get('incremental_jobs_total') or 0)} "
                    f"writeback={int(mainline.get('incremental_writeback_success_24h') or 0)} "
                    f"stalled_jobs={int(mainline.get('incremental_stalled_jobs') or 0)} "
                    f"invalid_bootstrap={int(mainline.get('core10_invalid_bootstrap_jobs') or 0)}"
                )
            if len(lines) >= 2:
                print(f"    历史记录: {len(lines)} 条")
        else:
            print("    ⚠️  日志文件为空")
    else:
        print("    ⚠️  尚未运行过在线学习 (logs/online_learning_history.jsonl 不存在)")
    print(f"{'═' * 70}")

    return all_ok


def run_mainline_snapshot_check(expected_interval_minutes: Optional[int] = None, busy_note: str = "") -> bool:
    sched = check_incremental_training_scheduler(expected_interval_minutes=expected_interval_minutes)
    mainline = check_mainline_runtime_pool_status()
    slot_scoped_health = _slot_scoped_live_health_only_enabled()
    slot_scoped_targets = _load_slot_scoped_live_targets() if slot_scoped_health else {}
    slot_scoped_target_symbols = sorted({
        str((row or {}).get("symbol") or "").strip().upper()
        for row in slot_scoped_targets
        if str((row or {}).get("symbol") or "").strip()
    })
    runtime_guard_review = _read_report_json(MAINLINE_RUNTIME_GUARD_REVIEW)
    postlaunch_guard_effect = _read_report_json(MAINLINE_POSTLAUNCH_GUARD_EFFECT)
    runtime_summary = runtime_guard_review.get("summary") if isinstance(runtime_guard_review.get("summary"), dict) else {}
    postlaunch_summary = postlaunch_guard_effect.get("summary") if isinstance(postlaunch_guard_effect.get("summary"), dict) else {}
    postlaunch_default_summary = postlaunch_guard_effect.get("default_summary") if isinstance(postlaunch_guard_effect.get("default_summary"), dict) else {}
    postlaunch_profile70_summary = postlaunch_guard_effect.get("profile70_summary") if isinstance(postlaunch_guard_effect.get("profile70_summary"), dict) else {}
    postlaunch_profile_compare = postlaunch_guard_effect.get("profile_compare") if isinstance(postlaunch_guard_effect.get("profile_compare"), dict) else {}

    print(f"\n{'═' * 70}")
    print("  Polyfun 主线快照健康检查")
    print(f"{'═' * 70}")
    if busy_note:
        print(f"  ⚠️ {busy_note}")

    print("\n🕒 主调度:")
    if sched.get("running"):
        if sched.get("has_cycle_history_24h"):
            training_timeout_display = str(int(sched.get("training_timeout_24h") or 0))
            if str(sched.get("history_source") or "") == "grouped_online_learning" and int(sched.get("legacy_cycle_timeout_24h") or 0) > 0:
                training_timeout_display += f" (legacy_cycle={int(sched.get('legacy_cycle_timeout_24h') or 0)})"
        else:
            training_timeout_display = f"- (legacy={int(sched.get('legacy_timeout_24h') or sched.get('timeout_24h') or 0)})"
        history_source = str(sched.get("history_source") or "-")
        repaired_pending = bool(sched.get("runtime_apply_repair_pending_verification"))
        result_class = "ok_with_observation" if repaired_pending else str(sched.get("latest_result_class") or "-")
        runtime_apply_txt = (
            "repaired_pending_verification"
            if repaired_pending
            else "-"
            if sched.get("latest_runtime_apply_success") is None
            else "ok"
            if bool(sched.get("latest_runtime_apply_success"))
            else "failed"
        )
        runtime_apply_failure_display = str(int(sched.get("runtime_apply_failure_24h") or 0))
        if repaired_pending:
            runtime_apply_failure_display += " (historical_repaired)"
        param_search_state = str(sched.get("latest_param_search_cadence_state") or sched.get("latest_param_search_status") or "-")
        param_search_cadence = sched.get("param_search_cadence_minutes")
        param_search_backlog = int(
            sched.get("latest_night_search_backlog_total")
            or sched.get("latest_param_search_backlog_windows")
            or 0
        )
        param_search_next_due = sched.get("latest_param_search_next_due_at") or "-"
        print(
            f"  模式: 主线运行池 | interval={sched.get('interval_minutes')}m "
            f"success_24h={int(sched.get('success_24h') or 0)} "
            f"training_timeout_24h={training_timeout_display} "
            f"runtime_apply_failure_24h={runtime_apply_failure_display} "
            f"refresh_timeout_24h={int(sched.get('post_training_refresh_timeout_24h') or 0)} "
            f"nonblocking_refresh_status={sched.get('latest_nonblocking_refresh_status') or '-'} "
            f"param_search_timeout_24h={int(sched.get('param_search_nonblocking_timeout_24h') or 0)} "
            f"param_search={param_search_state} "
            f"(status={sched.get('latest_param_search_status') or '-'} "
            f"cadence={param_search_cadence or '-'}m "
            f"backlog={param_search_backlog} "
            f"next_due={param_search_next_due}) "
            f"night_search_completion={sched.get('night_search_completion_status') or '-'} "
            f"apply_effective={sched.get('apply_effective_status') or '-'} "
            f"result_class={result_class} "
            f"latest_runtime_apply={runtime_apply_txt} "
            f"history_source={history_source} "
            f"next_run_at={sched.get('next_run_at') or '-'}"
        )
    else:
        print("  ❌ 主调度未运行")
    for issue in sched.get("issues", []):
        print(f"  ⚠️ {issue}")
    for warning in sched.get("warnings", []):
        print(f"  ℹ️ {warning}")

    print("\n🎯 主线运行池:")
    if mainline.get("available"):
        if slot_scoped_health and slot_scoped_targets:
            print(
                f"  规模: traders={len(slot_scoped_targets)} "
                f"cells={len(slot_scoped_targets)} "
                f"symbols={','.join(slot_scoped_target_symbols) or '-'}"
            )
            print("  范围: 仅 slot1/slot2 真钱链 + 必要 ETH 共享链 (BTC/XRP 已停用并移出健康范围)")
        else:
            print(
                f"  规模: traders={mainline.get('active_traders_total')} "
                f"cells={mainline.get('active_cells_total')} "
                f"symbols={','.join(mainline.get('active_symbols') or []) or '-'}"
            )
            print(
                f"  状态: mainline_active={mainline.get('mainline_active_cells')} "
                f"candidate_only={mainline.get('candidate_only_cells')} "
                f"n_a_by_design={mainline.get('n_a_by_design_cells')}"
            )
        print(
            f"  增量训练: jobs={mainline.get('incremental_jobs_total')} "
            f"writeback={mainline.get('successful_writeback_jobs')} "
            f"veto_only={mainline.get('veto_only_jobs')} "
            f"failed={mainline.get('failed_jobs')} "
            f"skipped={mainline.get('skipped_jobs')}"
        )
        print(
            f"  写回守门: baseline_insufficient={mainline.get('writeback_guard_baseline_insufficient_units')} "
            f"observed_only={mainline.get('writeback_guard_observed_but_baseline_insufficient_units')} "
            f"effective_weakened={mainline.get('writeback_guard_effective_weakened_units')} "
            f"effective_challenger={mainline.get('writeback_guard_effective_challenger_trigger_units')}"
        )
        print(
            f"  最新修订: mainline_target={mainline.get('mainline_latest_revision_target_cells')} "
            f"mainline_loaded={mainline.get('mainline_latest_revision_loaded_cells')} "
            f"candidate_loaded={mainline.get('candidate_latest_revision_loaded_cells')} "
            f"sync={mainline.get('latest_revision_sync_state')} "
            f"runtime_gap={mainline.get('latest_revision_runtime_sync_gap_cells')} "
            f"header_lag={mainline.get('prediction_header_lag_cells')} "
            f"header_lag_streak={mainline.get('prediction_header_lag_streak_runs')}"
        )
        if int(mainline.get("prediction_header_pending_refresh_cells") or 0) > 0:
            print(
                f"        header_pending_refresh={int(mainline.get('prediction_header_pending_refresh_cells') or 0)} "
                f"(writer 刚重启/热重载后的短暂待刷新)"
            )
        if bool(mainline.get("prediction_header_lag_warning_visible")):
            print("        header_lag 已连续超过 2 个主调度周期，需继续跟踪")
        print(
            f"  writer托管: status={mainline.get('writer_launchctl_status') or 'unknown'} "
            f"launchd={int(mainline.get('writer_launchctl_launchd_count') or 0)} "
            f"manual={int(mainline.get('writer_launchctl_manual_count') or 0)} "
            f"duplicate={int(mainline.get('writer_launchctl_duplicate_count') or 0)} "
            f"stopped_obs={int(mainline.get('writer_launchctl_stopped_observation_count') or 0)} "
            f"critical_stopped={int(mainline.get('writer_launchctl_critical_stopped_count') or 0)}"
        )
        print(
            f"  选择器: zero_default={mainline.get('selector_zero_rows_default')} "
            f"zero_70={mainline.get('selector_zero_rows_70')} "
            f"overstrict={mainline.get('selector_overstrict_rows_total')} "
            f"simulation_scope={mainline.get('selector_active_simulation_scope_rows_total') or mainline.get('selector_active_shadow_scope_rows_total')} "
            f"xrp_unknown={mainline.get('selector_xrp_runtime_unknown_cells_total')}"
        )
        print(
            f"  70阈值: btceth={mainline.get('filter70_btceth_threshold_selected')} "
            f"xrp={mainline.get('filter70_xrp_threshold_selected')} "
            f"constraints_passed={mainline.get('filter70_constraints_passed')}"
        )
        print(
            f"  审计: rules_drift={mainline.get('rules_json_drift_count')} "
            f"source_drift={mainline.get('source_drift_count')} "
            f"model_version_mismatch={mainline.get('prediction_model_version_mismatch_count')} "
            f"writer_missing={mainline.get('writer_process_missing_count')}"
        )
        if bool(mainline.get('runtime_guard_review_fresh')):
            print(
                f"  风控: extreme_active={int(runtime_summary.get('extreme_active_total') or 0)} "
                f"extreme_scaled={int(runtime_summary.get('extreme_scaled_bet_total') or 0)} "
                f"direction_loss_blocked={int(runtime_summary.get('direction_loss_blocked_total') or 0)} "
                f"direction_loss_soft_scaled={int(runtime_summary.get('direction_loss_soft_scaled_total') or 0)} "
                f"(mid={int(runtime_summary.get('direction_loss_soft_scaled_mid_total') or 0)} "
                f"strong={int(runtime_summary.get('direction_loss_soft_scaled_strong_total') or 0)} "
                f"extreme_overlap={int(runtime_summary.get('direction_loss_soft_scaled_with_extreme_overlap_total') or 0)}) "
                f"drawdown_acc_blocked={int(runtime_summary.get('drawdown_acceleration_blocked_total') or 0)} "
                f"drawdown_acc_soft_scaled={int(runtime_summary.get('drawdown_acceleration_soft_scaled_total') or 0)}"
            )
        else:
            print(
                f"  风控: 过期快照 age={mainline.get('runtime_guard_review_age_hours')}h"
            )
        if bool(mainline.get('postlaunch_guard_effect_fresh')):
            print(
                f"  上线后: trade_cells={int(postlaunch_summary.get('trade_cells_total') or 0)} "
                f"negative_cells={int(postlaunch_summary.get('negative_pnl_cells_total') or 0)} "
                f"high_conf_loss_cells={int(postlaunch_summary.get('high_conf_loss_cells_total') or 0)} "
                f"total_pnl={round(float(postlaunch_summary.get('postlaunch_total_pnl') or 0.0), 2)}"
            )
            print(
                f"  对照: default trade_cells={int(postlaunch_default_summary.get('trade_cells_total') or 0)} "
                f"negative_cells={int(postlaunch_default_summary.get('negative_pnl_cells_total') or 0)} "
                f"pnl={round(float(postlaunch_default_summary.get('postlaunch_total_pnl') or 0.0), 2)}"
            )
            print(
                f"        70      trade_cells={int(postlaunch_profile70_summary.get('trade_cells_total') or 0)} "
                f"negative_cells={int(postlaunch_profile70_summary.get('negative_pnl_cells_total') or 0)} "
                f"pnl={round(float(postlaunch_profile70_summary.get('postlaunch_total_pnl') or 0.0), 2)} "
                f"weaker={postlaunch_profile_compare.get('weaker_profile_by_total_pnl') or 'parity'} "
                f"evidence={postlaunch_profile_compare.get('comparison_evidence') or 'unknown'} "
                f"likely_gap={postlaunch_profile_compare.get('likely_primary_gap') or 'balanced'}"
            )
            if str(postlaunch_profile_compare.get('comparison_evidence') or '') != 'sufficient':
                reasons = ",".join(postlaunch_profile_compare.get("comparison_evidence_reasons") or []) or "-"
                print(f"        对照证据不足: reasons={reasons}")
            print(
                f"  默认链前置路由: allowed={int(mainline.get('postlaunch_default_market_selection_allowed_cells_total') or 0)} "
                f"abstained={int(mainline.get('postlaunch_default_market_selection_abstained_cells_total') or 0)} "
                f"quality_avg={round(float(mainline.get('postlaunch_default_market_quality_avg') or 0.0), 4)}"
            )
            allow_reason_counts = mainline.get("postlaunch_default_market_selection_allow_reason_counts") or {}
            abstain_reason_counts = mainline.get("postlaunch_default_market_selection_abstain_reason_counts") or {}
            if isinstance(allow_reason_counts, dict) and allow_reason_counts:
                top_items = sorted(allow_reason_counts.items(), key=lambda item: (-int(item[1] or 0), item[0]))[:3]
                print(f"        allow主因: {','.join(f'{k}={v}' for k, v in top_items)}")
            if isinstance(abstain_reason_counts, dict) and abstain_reason_counts:
                top_items = sorted(abstain_reason_counts.items(), key=lambda item: (-int(item[1] or 0), item[0]))[:3]
                print(f"        abstain主因: {','.join(f'{k}={v}' for k, v in top_items)}")
            if bool(mainline.get('default_weakness_review_fresh')):
                print(
                    f"  默认链弱盘: symbol={mainline.get('default_weakness_primary_loss_symbol') or 'unknown'} "
                    f"family={mainline.get('default_weakness_primary_loss_family') or 'unknown'} "
                    f"trader={mainline.get('default_weakness_primary_loss_trader') or 'unknown'} "
                    f"当前选盘桶位={mainline.get('default_weakness_primary_current_selection_bucket') or 'unknown'}"
                )
                print(
                    f"        selection_known={int(mainline.get('default_weakness_selection_known_cells_total') or 0)} "
                    f"selection_unknown={int(mainline.get('default_weakness_selection_unknown_cells_total') or 0)} "
                    f"unknown_negative={int(mainline.get('default_weakness_selection_unknown_negative_cells_total') or 0)} "
                    f"unknown_pnl={round(float(mainline.get('default_weakness_selection_unknown_total_pnl') or 0.0), 2)}"
                )
            else:
                print(
                    f"  默认链弱盘: 过期快照 age={mainline.get('default_weakness_review_age_hours')}h"
                )
            print(
                f"  总亏损归因: status={mainline.get('loss_attribution_status') or 'unknown'} "
                f"total_pnl={round(float(mainline.get('loss_attribution_total_pnl') or 0.0), 2)} "
                f"drag_profile={mainline.get('loss_attribution_primary_drag_profile') or 'unknown'} "
                f"drag_symbol={mainline.get('loss_attribution_primary_drag_symbol') or 'unknown'} "
                f"drag_family={mainline.get('loss_attribution_primary_drag_family') or 'unknown'} "
                f"drag_trader={mainline.get('loss_attribution_primary_drag_trader') or 'unknown'} "
                f"guard_bucket={mainline.get('loss_attribution_primary_drag_guard_bucket') or 'unknown'} "
                f"weaker_profile={mainline.get('loss_attribution_weaker_profile_by_total_pnl') or 'parity'} "
                f"likely_gap={mainline.get('loss_attribution_likely_primary_gap') or 'unknown'}"
            )
            print(
                f"  买价真相: status={mainline.get('buy_price_truth_status') or 'unknown'} "
                f"runtime_conflict={int(mainline.get('buy_price_current_runtime_truth_inconsistent_count') or 0)} "
                f"bp_tag_mismatch={int(mainline.get('buy_price_bp_tag_runtime_mismatch_count') or 0)} "
                f"prediction_observation_mismatch={int(mainline.get('buy_price_prediction_observation_mismatch_count') or 0)} "
                f"rules_runtime_semantic_mismatch={int(mainline.get('buy_price_rules_runtime_semantic_mismatch_count') or 0)} "
                f"rules_path_semantic_mismatch={int(mainline.get('buy_price_rules_path_semantic_mismatch_count') or 0)}"
            )
            print(
                f"        exp10_bp0520_eth runtime={mainline.get('buy_price_focus_exp10_bp0520_eth_runtime_price')} "
                f"prediction={mainline.get('buy_price_focus_exp10_bp0520_eth_prediction_limit_price')} "
                f"rules={mainline.get('buy_price_focus_exp10_bp0520_eth_rules_json_buy_price')} "
                f"source={mainline.get('buy_price_focus_exp10_bp0520_eth_runtime_source_of_truth') or 'unknown'} "
                f"best_balance={mainline.get('buy_price_focus_exp10_bp0520_eth_best_balance_price')} "
                f"pre_045_better={mainline.get('buy_price_focus_exp10_bp0520_eth_pre_init_045_clearly_better')} "
                f"建议={mainline.get('buy_price_focus_exp10_bp0520_eth_recommendation') or 'unknown'}"
            )
            print(
                f"        runtime_truth={mainline.get('buy_price_runtime_truth_current_conclusion') or 'unknown'} "
                f"prediction_observation={mainline.get('buy_price_prediction_observation_current_conclusion') or 'unknown'} "
                f"rules_path_semantics={mainline.get('buy_price_rules_path_semantics_current_conclusion') or 'unknown'}"
            )
            simulation_counts = mainline.get('buy_price_simulation_validation_priority_counts') or mainline.get('buy_price_shadow_validation_priority_counts') or {}
            print(
                f"        模拟验证队列 rows={int(mainline.get('buy_price_simulation_validation_rows_total') or mainline.get('buy_price_shadow_validation_rows_total') or 0)} "
                f"tier1={int(simulation_counts.get('tier1_post_window_first') or 0)} "
                f"tier2={int(simulation_counts.get('tier2_pre_month_strong') or 0)} "
                f"tier3={int(simulation_counts.get('tier3_observe_only') or 0)}"
            )
            print(
                f"        第一批模拟结果 status={mainline.get('buy_price_simulation_validation_results_status') or mainline.get('buy_price_shadow_validation_results_status') or 'unknown'} "
                f"rows={int(mainline.get('buy_price_simulation_validation_results_rows_total') or mainline.get('buy_price_shadow_validation_results_rows_total') or 0)} "
                f"candidate_for_micro_adjust={int(mainline.get('buy_price_simulation_validation_results_candidate_for_micro_adjust_rows_total') or mainline.get('buy_price_shadow_validation_results_candidate_for_micro_adjust_rows_total') or 0)}"
            )
            print(
                f"        第一批微调评审 status={mainline.get('buy_price_simulation_micro_adjust_review_status') or mainline.get('buy_price_shadow_micro_adjust_review_status') or 'unknown'} "
                f"ready={int(mainline.get('buy_price_simulation_micro_adjust_review_ready_rows_total') or mainline.get('buy_price_shadow_micro_adjust_review_ready_rows_total') or 0)} "
                f"hold={int(mainline.get('buy_price_simulation_micro_adjust_review_hold_rows_total') or mainline.get('buy_price_shadow_micro_adjust_review_hold_rows_total') or 0)} "
                f"结论={str(mainline.get('buy_price_simulation_micro_adjust_review_current_conclusion') or mainline.get('buy_price_shadow_micro_adjust_review_current_conclusion') or 'unknown').replace('shadow', 'simulation').replace('影子', '模拟')}"
            )
            print(
                f"  前置路由效果: status={mainline.get('route_effectiveness_status') or 'unknown'} "
                f"postlaunch={mainline.get('route_effectiveness_postlaunch_conclusion') or 'unknown'} "
                f"positive={int(mainline.get('route_effectiveness_postlaunch_abstain_effect_positive') or 0)} "
                f"neutral={int(mainline.get('route_effectiveness_postlaunch_abstain_effect_neutral') or 0)} "
                f"negative={int(mainline.get('route_effectiveness_postlaunch_abstain_effect_negative') or 0)} "
                f"reduction_without_quality_gain={bool(mainline.get('route_effectiveness_postlaunch_route_reduction_without_quality_gain'))}"
            )
            print(
                f"  语义审计: status={mainline.get('recent_change_semantic_audit_status') or 'unknown'} "
                f"confirmed_bug={int(mainline.get('recent_change_semantic_confirmed_bug_count') or 0)} "
                f"semantic_misleading={int(mainline.get('recent_change_semantic_semantic_misleading_count') or 0)}"
            )
            print(
                f"  运维口径审计: status={mainline.get('operational_bug_audit_status') or 'unknown'} "
                f"confirmed_bug={int(mainline.get('operational_bug_audit_confirmed_bug_count') or 0)} "
                f"semantic_misleading={int(mainline.get('operational_bug_audit_semantic_misleading_count') or 0)} "
                f"operational_risk={int(mainline.get('operational_bug_audit_operational_risk_count') or 0)}"
            )
            print(
                f"  parquet写链: status={mainline.get('runtime_parquet_writer_map_status') or 'unknown'} "
                f"multiple_writers={int(mainline.get('runtime_parquet_multiple_writers_detected_count') or 0)} "
                f"collector={int(mainline.get('runtime_parquet_collector_process_count') or 0)} "
                f"sentiment={int(mainline.get('runtime_parquet_sentiment_process_count') or 0)} "
                f"sentiment_supervision={mainline.get('runtime_sentiment_supervision_state') or 'unknown'} "
                f"cfgi={mainline.get('cfgi_runtime_state') or 'unknown'}"
            )
            print(
                f"  安全清理: status={mainline.get('safe_cleanup_status') or 'unknown'} "
                f"delete_now={int(mainline.get('safe_cleanup_delete_now_count') or 0)} "
                f"deleted_now={int(mainline.get('safe_cleanup_deleted_now_count') or 0)} "
                f"archive_candidate={int(mainline.get('safe_cleanup_archive_candidate_count') or 0)} "
                f"keep={int(mainline.get('safe_cleanup_keep_count') or 0)}"
            )
        else:
            print(
                f"  上线后: 过期快照 age={mainline.get('postlaunch_guard_effect_age_hours')}h"
            )
        if bool(mainline.get('postlaunch_guard_effect_fresh')):
            print(
                f"  回撤压力: guarded_negative={int(mainline.get('postlaunch_guarded_negative_cells_total') or 0)} "
                f"unguarded_negative(粗)={int(mainline.get('postlaunch_unguarded_negative_cells_total') or 0)} "
                f"watch_only_negative={int(mainline.get('postlaunch_drawdown_acceleration_watch_only_negative_cells_total') or 0)} "
                f"watch_only_worsening={int(mainline.get('postlaunch_drawdown_acceleration_watch_only_worsening_negative_cells_total') or 0)} "
                f"non_extreme_same_dir={int(mainline.get('postlaunch_high_conf_same_direction_non_extreme_cells_total') or 0)} "
                f"unmitigated_same_dir={int(mainline.get('postlaunch_non_extreme_high_conf_same_direction_unmitigated_cells_total') or 0)}"
            )
            print(
                f"  夜间对照: default trade_cells={int(mainline.get('overnight_default_trade_cells_total') or 0)} "
                f"negative_cells={int(mainline.get('overnight_default_negative_pnl_cells_total') or 0)} "
                f"pnl={round(float(mainline.get('overnight_default_total_pnl') or 0.0), 2)}"
            )
            print(
                f"        70      trade_cells={int(mainline.get('overnight_70_trade_cells_total') or 0)} "
                f"negative_cells={int(mainline.get('overnight_70_negative_pnl_cells_total') or 0)} "
                f"pnl={round(float(mainline.get('overnight_70_total_pnl') or 0.0), 2)} "
                f"weaker={mainline.get('overnight_weaker_profile_by_total_pnl') or 'parity'} "
                f"evidence={mainline.get('overnight_comparison_evidence') or 'unknown'} "
                f"likely_gap={mainline.get('overnight_likely_primary_gap') or 'balanced'}"
            )
            if str(mainline.get('overnight_comparison_evidence') or '') != 'sufficient':
                reasons = ",".join(mainline.get("overnight_comparison_evidence_reasons") or []) or "-"
                print(f"        夜间对照证据不足: reasons={reasons}")
            if not (
                str(mainline.get("overnight_window_source") or "") == "fallback_last_12_hours"
                and int(mainline.get("overnight_default_trade_cells_total") or 0) == 0
            ):
                print(
                    f"        默认链前置路由 allowed={int(mainline.get('overnight_default_market_selection_allowed_cells_total') or 0)} "
                    f"abstained={int(mainline.get('overnight_default_market_selection_abstained_cells_total') or 0)} "
                    f"quality_avg={round(float(mainline.get('overnight_default_market_quality_avg') or 0.0), 4)}"
                )
                overnight_allow_reason_counts = mainline.get("overnight_default_market_selection_allow_reason_counts") or {}
                overnight_abstain_reason_counts = mainline.get("overnight_default_market_selection_abstain_reason_counts") or {}
                if isinstance(overnight_allow_reason_counts, dict) and overnight_allow_reason_counts:
                    top_items = sorted(overnight_allow_reason_counts.items(), key=lambda item: (-int(item[1] or 0), item[0]))[:3]
                    print(f"        allow主因: {','.join(f'{k}={v}' for k, v in top_items)}")
                if isinstance(overnight_abstain_reason_counts, dict) and overnight_abstain_reason_counts:
                    top_items = sorted(overnight_abstain_reason_counts.items(), key=lambda item: (-int(item[1] or 0), item[0]))[:3]
                    print(f"        abstain主因: {','.join(f'{k}={v}' for k, v in top_items)}")
            else:
                print(
                    f"        夜间前置路由摘要省略: window_source={mainline.get('overnight_window_source')}"
                )
        else:
            print(
                f"  回撤压力: 过期快照 age={mainline.get('postlaunch_guard_effect_age_hours')}h"
            )
        print(
            f"  融合: default={mainline.get('fusion_default_cells')} "
            f"70={mainline.get('fusion_70_cells')} "
            f"state={str(mainline.get('fusion_execution_state') or 'unknown').replace('影子', '模拟')} "
            f"trade_cells={mainline.get('fusion_simulated_trade_cells')} "
            f"prediction_only={mainline.get('fusion_prediction_only_cells')} "
            f"simulation_only={mainline.get('fusion_simulation_only_cells') or mainline.get('fusion_shadow_only_cells')} "
            f"exec_not_landed={mainline.get('fusion_execution_not_landed_cells')}"
        )
        print(f"  结论: {mainline.get('current_conclusion') or '-'}")
    else:
        print("  ❌ 主线报告缺失")

    issues = list(sched.get("issues") or []) + list(mainline.get("issues") or [])
    print(f"\n{'═' * 70}")
    if issues:
        print(f"  ⚠️ 快照检查发现 {len(issues)} 个异常/注意项")
        for idx, issue in enumerate(issues, 1):
            print(f"  [{idx}] {issue}")
        return False
    print("  ✅ 主线快照正常")
    return True


def main():
    parser = argparse.ArgumentParser(description="系统健康监控 v5")
    parser.add_argument("--once", action="store_true", help="单次检查后退出")
    parser.add_argument(
        "--expected-interval-minutes",
        type=int,
        default=None,
        help=f"调度器期望间隔（分钟），默认使用脚本内置值 {EXPECTED_INCREMENTAL_INTERVAL_MINUTES}",
    )
    parser.add_argument(
        "--auto-heal",
        action="store_true",
        help="检测到 Daemon/采集器异常时自动尝试修复（重启 Daemon、重载采集器）",
    )
    args = parser.parse_args()
    runtime_root = os.environ.get("PROJECT_RUNTIME_ROOT") or str(PROJECT_ROOT)
    runtime_pm = os.environ.get("PROJECT_RUNTIME_PM") or str(POLYMARKET_DIR)

    if args.once:
        print(f"🧭 运行根: {runtime_root}")
        print(f"🧭 Polymarket根: {runtime_pm}")
        lock_handle, lock_ok = _acquire_api_health_lock()
        try:
            if not lock_ok:
                run_mainline_snapshot_check(
                    expected_interval_minutes=args.expected_interval_minutes,
                    busy_note="已有健康检查在运行，本次直接返回最近主线快照",
                )
                return
            if args.auto_heal:
                run_check(auto_heal=args.auto_heal, expected_interval_minutes=args.expected_interval_minutes)
            else:
                run_mainline_snapshot_check(expected_interval_minutes=args.expected_interval_minutes)
        finally:
            if lock_handle is not None:
                try:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
                try:
                    lock_handle.close()
                except Exception:
                    pass
    else:
        expected_groups_runtime = get_expected_merged_groups()
        expected_total_traders_runtime = sum(expected_groups_runtime.values())
        print(f"🧭 运行根: {runtime_root}")
        print(f"🧭 Polymarket根: {runtime_pm}")
        print(f"🔄 系统健康监控 v5 已启动 (每 {CHECK_INTERVAL // 60} 分钟全面检查)")
        print(
            f"   架构: {len(expected_groups_runtime)} 个合并进程 (multi_prediction_index), "
            f"共 {expected_total_traders_runtime} traders"
        )
        print(f"   检查项: 代理/API/ccxt/Daemon/采集器/数据源/写入器/预测文件/"
              f"合并进程/重复检测/WS+REST/代理连接/日志异常/融合验证/交易统计/增量训练调度器")
        if args.auto_heal:
            print("   自动修复: 已开启（Daemon/采集器异常时自动自愈）")
        print(f"   纯只读监控，不影响任何模拟交易\n")
        while True:
            try:
                run_check(auto_heal=args.auto_heal, expected_interval_minutes=args.expected_interval_minutes)
            except Exception as e:
                print(f"\n❌ 监控出错: {e}")
            next_time = datetime.fromtimestamp(time.time() + CHECK_INTERVAL)
            print(f"\n⏰ 下次检查: {next_time.strftime('%H:%M:%S')} ({CHECK_INTERVAL // 60}分钟后)...")
            time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
