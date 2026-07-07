#!/usr/bin/env python3
"""
按分组最优参数执行增量训练（v5_short / v5_long / gru_core）。

默认读取:
  reports/online_learning_group_best_params.json

用法:
  /Users/mac/miniforge3/bin/python scripts/run_grouped_online_learning.py
  /Users/mac/miniforge3/bin/python scripts/run_grouped_online_learning.py --dry-run
  /Users/mac/miniforge3/bin/python scripts/run_grouped_online_learning.py --groups v5_short,gru_core
"""
from __future__ import annotations

import argparse
import json
import os
import plistlib
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

OPS = Path(__file__).resolve().parent / "ops"
if str(OPS) not in sys.path:
    sys.path.insert(0, str(OPS))

from pipeline_resume_common import (
    atomic_write_checkpoint,
    checkpoint_contract,
    checkpoint_path,
    matching_checkpoint,
    stable_hash,
)
from incremental_param_search_pause_common import load_pause_state, maintenance_window_active

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROFILE = PROJECT_ROOT / "reports" / "online_learning_group_best_params.json"
ONLINE_HISTORY_FILE = PROJECT_ROOT / "logs" / "online_learning_history.jsonl"
LAUNCH_AGENTS_DIR = Path.home() / "Library" / "LaunchAgents"
CORE10_PROMOTION_STATE_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_promotion_state.py"
CORE10_APPLY_PROMOTIONS_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "apply_core10_incremental_promotions.py"
CORE10_RUNTIME_CUTOVER_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "apply_core10_incremental_runtime_cutover.py"
RELOAD_PREDICTION_WRITERS_SCRIPT = PROJECT_ROOT / "reload_launchctl_prediction_writers.sh"
MAINLINE_REFRESH_CHAIN_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "run_mainline_refresh_chain.py"
CORE10_PARAM_APPLY_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_incremental_param_apply_latest.py"
CORE10_INCREMENTAL_PROFILE_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_core10_incremental_profile.py"
INCREMENTAL_PARAM_SEARCH_COMPLETION_AUDIT_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_incremental_param_search_completion_audit_latest.py"
INCREMENTAL_CHAIN_TRUTH_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_incremental_training_chain_truth_inventory_latest.py"
INCREMENTAL_PUBLICATION_STATE_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_incremental_training_publication_state_latest.py"
INCREMENTAL_PREDICTION_USAGE_AUDIT_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_incremental_training_prediction_usage_audit_latest.py"
INCREMENTAL_RUNTIME_VS_RECOMMENDATION_AUDIT_SCRIPT = PROJECT_ROOT / "scripts" / "ops" / "generate_incremental_training_runtime_vs_recommendation_audit_latest.py"

CORE10_MINIMAL_REFRESH_SCRIPTS: tuple[Path, ...] = (
    CORE10_PARAM_APPLY_SCRIPT,
    CORE10_INCREMENTAL_PROFILE_SCRIPT,
    INCREMENTAL_PARAM_SEARCH_COMPLETION_AUDIT_SCRIPT,
    INCREMENTAL_CHAIN_TRUTH_SCRIPT,
    INCREMENTAL_PUBLICATION_STATE_SCRIPT,
    INCREMENTAL_PREDICTION_USAGE_AUDIT_SCRIPT,
    INCREMENTAL_RUNTIME_VS_RECOMMENDATION_AUDIT_SCRIPT,
)


def _classify_result_class(
    *,
    training_success: bool,
    runtime_apply_success: bool,
    core_refresh_success: bool,
    runtime_apply_status: str,
    nonblocking_refresh_status: str,
) -> str:
    if not training_success:
        return "failed_critical_path"
    if not runtime_apply_success:
        return "failed_runtime_apply"
    if not core_refresh_success:
        return "failed_core_refresh"
    if nonblocking_refresh_status in {"failed_non_blocking", "nonblocking_slow"}:
        return "nonblocking_slow"
    if runtime_apply_status in {"ok_with_observation"}:
        return "ok_with_observation"
    return "ok"

GROUP_TARGETS = {
    "v5_short": {"kind": "v5", "exp_ids": [10, 11, 13, 14]},
    "v5_long": {"kind": "v5", "exp_ids": [15, 16, 17]},
    "gru_core": {"kind": "gru", "assets": ["BTC_USDT", "ETH_USDT", "SOL_USDT"]},
}


def _checkpoint_job_signature(profile_path: Path, payload: Dict[str, object], groups: List[str], jobs: List[Dict[str, object]] | None) -> tuple[str, str, Path]:
    mode = str(payload.get("mode") or "shared_or_legacy").strip()
    combo_key = "jobs" if jobs is not None else ",".join(groups)
    job_key = f"{profile_path.stem}__{mode}"
    config_payload = payload
    data_window_payload = {
        "groups": groups,
        "job_ids": [str(job.get("job_id") or "") for job in (jobs or []) if isinstance(job, dict)],
    }
    config_hash = stable_hash(config_payload)
    data_window_hash = stable_hash(data_window_payload)
    return config_hash, data_window_hash, checkpoint_path("grouped_online_learning", job_key)


def _build_common_args(p: Dict[str, object]) -> List[str]:
    def g(name: str, default: object) -> object:
        return p.get(name, default)

    args = [
        "--hours", str(int(g("hours", 48))),
        "--base-rounds", str(int(g("base_rounds", 12))),
        "--calm-rounds", str(int(g("calm_rounds", 8))),
        "--shock-rounds", str(int(g("shock_rounds", 20))),
        "--recency-halflife-bars", str(int(g("recency_halflife_bars", 64))),
        "--shock-weight-mult", str(float(g("shock_weight_mult", 1.8))),
        "--calm-vol-ratio", str(float(g("calm_vol_ratio", 1.08))),
        "--shock-vol-ratio", str(float(g("shock_vol_ratio", 1.45))),
        "--shock-ret-q", str(float(g("shock_ret_q", 0.85))),
        "--rollback-auc-drop-abs", str(float(g("rollback_auc_drop_abs", 0.004))),
        "--rollback-auc-drop-rel", str(float(g("rollback_auc_drop_rel", 0.008))),
        "--rollback-utility-drop-abs", str(float(g("rollback_utility_drop_abs", 0.0))),
        "--rollback-down-wr-drop-abs", str(float(g("rollback_down_wr_drop_abs", 0.0))),
        "--utility-min-confidence", str(float(g("utility_min_confidence", 0.53))),
        "--utility-min-samples", str(int(g("utility_min_samples", 120))),
        "--utility-min-down-samples", str(int(g("utility_min_down_samples", 40))),
    ]
    if bool(g("trade_objective_priority", False)):
        args.append("--trade-objective-priority")
    if "trade_objective_auc_catastrophic_mult" in p:
        args.extend([
            "--trade-objective-auc-catastrophic-mult",
            str(float(g("trade_objective_auc_catastrophic_mult", 3.0))),
        ])
    return args


def _build_cmd(group: str, params: Dict[str, object], python_bin: str, dry_run: bool) -> List[str]:
    target = GROUP_TARGETS[group]
    common = _build_common_args(params)
    if target["kind"] == "v5":
        cmd = [
            python_bin,
            str(PROJECT_ROOT / "scripts" / "online_learning_daily.py"),
            "--exp",
            *[str(x) for x in target["exp_ids"]],
            *common,
        ]
    else:
        cmd = [
            python_bin,
            str(PROJECT_ROOT / "scripts" / "online_learning_gru.py"),
            "--assets",
            *target["assets"],
            *common,
        ]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def _build_job_cmd(
    job: Dict[str, object],
    python_bin: str,
    dry_run: bool,
    as_of_timestamp: str = "",
    history_timestamp_override: str = "",
) -> Tuple[List[str], List[str]]:
    kind = str(job.get("kind") or "v5").strip().lower()
    params = job.get("best_params") if isinstance(job.get("best_params"), dict) else {}
    common = _build_common_args(params)
    temp_json_paths: List[str] = []
    target_slots = {
        str(item).strip()
        for item in (job.get("targetSlots") or [])
        if str(item).strip()
    }
    slot01_only_incremental = bool(job.get("slot01OnlyIncrementalTarget")) or target_slots == {"slot01"}
    if kind == "v5":
        exp_ids = [int(x) for x in (job.get("exp_ids") or [])]
        if not exp_ids:
            raise SystemExit(f"job 缺少 exp_ids: {job}")
        tier_name = str(job.get("tier") or "").strip().lower()
        history_mode = "mainline_runtime_pool" if tier_name.startswith("tier1") or tier_name.startswith("tier2") else ""
        cmd = [
            python_bin,
            str(PROJECT_ROOT / "scripts" / "online_learning_daily.py"),
            "--exp",
            *[str(x) for x in exp_ids],
            *common,
            "--history-job-id",
            str(job.get("job_id") or ""),
            "--history-mode",
            history_mode,
            "--history-family",
            str(job.get("family") or ""),
            "--history-asset",
            str(job.get("asset") or ""),
        ]
        if history_mode in {"core10_only", "mainline_runtime_pool"}:
            cmd.append("--trade-objective-priority")
            if slot01_only_incremental:
                cmd.append("--slot01-positive-writeback")
            else:
                cmd.append("--family-atomic-writeback")
        elif bool(job.get("family_atomic_writeback", False)) and not slot01_only_incremental:
            cmd.append("--family-atomic-writeback")
        elif slot01_only_incremental:
            cmd.append("--slot01-positive-writeback")
        if str(as_of_timestamp or "").strip():
            cmd.extend(["--as-of-timestamp", str(as_of_timestamp).strip()])
        if str(history_timestamp_override or "").strip():
            cmd.extend(["--history-timestamp", str(history_timestamp_override).strip()])
        model_dir_map = job.get("model_dir_map")
        if isinstance(model_dir_map, dict) and model_dir_map:
            with tempfile.NamedTemporaryFile("w", suffix=".json", prefix="core10_model_dir_map_", delete=False) as fh:
                json.dump(model_dir_map, fh, ensure_ascii=False, indent=2)
                temp_json_paths.append(fh.name)
            cmd.extend(["--model-dir-map-json", temp_json_paths[-1]])
        bootstrap_src = str(job.get("bootstrap_from_model_dir") or job.get("source_model_dir") or "").strip()
        if bootstrap_src and isinstance(model_dir_map, dict) and model_dir_map:
            bootstrap_map = {str(exp_id): bootstrap_src for exp_id in exp_ids}
            with tempfile.NamedTemporaryFile("w", suffix=".json", prefix="core10_bootstrap_model_dir_map_", delete=False) as fh:
                json.dump(bootstrap_map, fh, ensure_ascii=False, indent=2)
                temp_json_paths.append(fh.name)
            cmd.extend(["--bootstrap-model-dir-map-json", temp_json_paths[-1]])
    elif kind == "gru":
        assets = [str(x) for x in (job.get("assets") or []) if str(x).strip()]
        if not assets:
            raise SystemExit(f"job 缺少 assets: {job}")
        cmd = [
            python_bin,
            str(PROJECT_ROOT / "scripts" / "online_learning_gru.py"),
            "--assets",
            *assets,
            *common,
        ]
    else:
        raise SystemExit(f"未知 job kind: {kind}")
    if dry_run:
        cmd.append("--dry-run")
    return cmd, temp_json_paths


def _append_group_history_entry(
    profile_path: Path,
    mode: str,
    dry_run: bool,
    job_results: List[Dict[str, object]],
    writers_to_reload: List[str] | None = None,
    writers_reloaded: List[str] | None = None,
    writer_reload_errors: List[str] | None = None,
    summary_updates: Dict[str, object] | None = None,
    timestamp_override: str | None = None,
) -> None:
    ONLINE_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    def _rc(row: Dict[str, object]) -> int:
        value = row.get("returncode", 1)
        try:
            return int(1 if value is None else value)
        except Exception:
            return 1

    success = sum(1 for row in job_results if _rc(row) == 0 and not bool(row.get("skipped", False)))
    failed = sum(1 for row in job_results if _rc(row) != 0)
    skipped = sum(1 for row in job_results if bool(row.get("skipped", False)))
    successful_writeback_jobs = 0
    veto_only_jobs = 0
    mixed_outcome_jobs = 0
    invalid_bootstrap_jobs = 0
    cadence_skip_jobs = 0
    for row in job_results:
        child_summary = row.get("child_summary") if isinstance(row.get("child_summary"), dict) else {}
        child_success = int(child_summary.get("success", 0) or 0)
        child_veto = int(child_summary.get("trained_but_not_written", 0) or 0)
        if child_success > 0:
            successful_writeback_jobs += 1
        if child_success == 0 and child_veto > 0:
            veto_only_jobs += 1
        if child_success > 0 and child_veto > 0:
            mixed_outcome_jobs += 1
        if str(row.get("skipped_reason") or "") == "SKIP_INVALID_BOOTSTRAP":
            invalid_bootstrap_jobs += 1
        if str(row.get("skipped_reason") or "") == "cadence_not_due":
            cadence_skip_jobs += 1
    entry = {
        "timestamp": str(timestamp_override).strip() if str(timestamp_override or "").strip() else datetime.now(timezone.utc).isoformat(),
        "entry_type": "grouped_online_learning",
        "mode": mode,
        "profile_json": str(profile_path),
        "dry_run": bool(dry_run),
        "jobs": job_results,
        "summary": {
            "jobs_total": len(job_results),
            "success": success,
            "failed": failed,
            "skipped": skipped,
            "successful_writeback_jobs": successful_writeback_jobs,
            "veto_only_jobs": veto_only_jobs,
            "mixed_outcome_jobs": mixed_outcome_jobs,
            "invalid_bootstrap_jobs": invalid_bootstrap_jobs,
            "cadence_skip_jobs": cadence_skip_jobs,
            "writers_to_reload": sorted(set(writers_to_reload or [])),
            "writers_reloaded": sorted(set(writers_reloaded or [])),
            "writer_reload_errors": list(writer_reload_errors or []),
        },
    }
    if isinstance(summary_updates, dict) and summary_updates:
        entry["summary"].update(summary_updates)
    with ONLINE_HISTORY_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _read_history_lines() -> List[str]:
    if not ONLINE_HISTORY_FILE.exists():
        return []
    try:
        return ONLINE_HISTORY_FILE.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []


def _parse_history_line(raw: str) -> Dict[str, Any] | None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _parse_history_ts(value: object) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _latest_job_success(job_id: str) -> Dict[str, Any]:
    for raw in reversed(_read_history_lines()):
        payload = _parse_history_line(raw)
        if not payload or payload.get("entry_type") == "grouped_online_learning":
            continue
        if str(payload.get("job_id") or "").strip() != job_id:
            continue
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        try:
            success = int(summary.get("success", 0) or 0)
        except Exception:
            success = 0
        if success <= 0:
            continue
        timestamp = _parse_history_ts(payload.get("timestamp"))
        if timestamp is None:
            continue
        return {
            "timestamp": timestamp,
            "summary": summary,
        }
    return {}


def _writer_launch_agents() -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for plist_path in sorted(LAUNCH_AGENTS_DIR.glob("polyfun*.plist")):
        try:
            payload = plistlib.loads(plist_path.read_bytes())
        except Exception:
            continue
        label = str(payload.get("Label") or "").strip()
        args = payload.get("ProgramArguments")
        if not label or not isinstance(args, list):
            continue
        joined = " ".join(str(x) for x in args)
        if "prediction_writer_v5.py" not in joined:
            continue
        model_dir = ""
        output_path = ""
        for idx, token in enumerate(args):
            if token == "--model-dir" and idx + 1 < len(args):
                model_dir = str(args[idx + 1])
            elif token == "--output" and idx + 1 < len(args):
                output_path = str(args[idx + 1])
        if not model_dir:
            model_match = re.search(r'--model-dir\s+"?([^"\s]+)"?', joined)
            if model_match:
                model_dir = str(model_match.group(1))
        if not output_path:
            output_match = re.search(r'--output\s+"?([^"\s]+)"?', joined)
            if output_match:
                output_path = str(output_match.group(1))
        if model_dir:
            rows.append(
                {
                    "label": label,
                    "model_dir": model_dir,
                    "output": output_path,
                }
            )
    return rows


def _reload_matching_writers(success_job_results: List[Dict[str, object]]) -> Tuple[List[str], List[str], List[str]]:
    target_model_dirs = {
        str(model_dir).strip()
        for row in success_job_results
        for model_dir in (dict(row.get("model_dir_map") or {}).values())
        if str(model_dir).strip()
    }
    if not target_model_dirs:
        return [], [], []
    writer_rows = _writer_launch_agents()
    writers_to_reload = sorted(
        {
            row["label"]
            for row in writer_rows
            if str(row.get("model_dir") or "").strip() in target_model_dirs
        }
    )
    if not writers_to_reload:
        return [], [], []
    uid = str(os.getuid())
    reloaded: List[str] = []
    errors: List[str] = []
    for label in writers_to_reload:
        try:
            ret = subprocess.run(
                ["launchctl", "kickstart", "-k", f"gui/{uid}/{label}"],
                cwd=str(PROJECT_ROOT),
                check=False,
                capture_output=True,
                text=True,
            )
        except Exception as exc:
            errors.append(f"{label}:spawn_error:{exc}")
            continue
        if ret.returncode == 0:
            reloaded.append(label)
        else:
            stderr = (ret.stderr or ret.stdout or "").strip()
            errors.append(f"{label}:rc={ret.returncode}:{stderr[:160]}")
    return writers_to_reload, reloaded, errors


def _reconcile_core10_writer_revisions() -> Dict[str, Any]:
    if not RELOAD_PREDICTION_WRITERS_SCRIPT.exists():
        return {
            "success": False,
            "status": "failed",
            "details": f"missing_reload_script:{RELOAD_PREDICTION_WRITERS_SCRIPT}",
            "stale_writer_count": 0,
            "max_wait_exceeded": False,
        }
    ret = subprocess.run(
        ["bash", str(RELOAD_PREDICTION_WRITERS_SCRIPT), "reconcile_only"],
        cwd=str(PROJECT_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    details = ((ret.stdout or "") + "\n" + (ret.stderr or "")).strip()
    stale_writer_count = 0
    match = re.search(r"仍有\s+(\d+)\s+个任务未追平", details)
    if match:
        try:
            stale_writer_count = int(match.group(1))
        except Exception:
            stale_writer_count = 0
    status = "ok" if ret.returncode == 0 else "partial" if ret.returncode == 2 else "failed"
    return {
        "success": status != "failed",
        "status": status,
        "details": details,
        "stale_writer_count": stale_writer_count,
        "max_wait_exceeded": status == "partial",
    }


def _refresh_core10_reports(python_bin: str, profile_path: Path, mode: str) -> bool:
    if profile_path.name != "core10_incremental_profile.json" or mode not in {"core10_only", "mainline_runtime_pool"}:
        return True
    for script_path in CORE10_MINIMAL_REFRESH_SCRIPTS:
        if not script_path.exists():
            continue
        ret = subprocess.run(
            [python_bin, "-B", str(script_path)],
            cwd=str(PROJECT_ROOT),
            check=False,
        )
        if ret.returncode != 0:
            print(f"[core10_minimal_refresh] {script_path.name} 失败: rc={ret.returncode}", file=sys.stderr)
            return False
    return True


def _apply_core10_runtime_promotions(
    python_bin: str,
    profile_path: Path,
    mode: str,
    success_job_results: List[Dict[str, object]],
) -> None:
    if profile_path.name != "core10_incremental_profile.json" or mode not in {"core10_only", "mainline_runtime_pool"}:
        return
    if not success_job_results:
        return
    for script_path in (
        CORE10_PROMOTION_STATE_SCRIPT,
        CORE10_APPLY_PROMOTIONS_SCRIPT,
        CORE10_RUNTIME_CUTOVER_SCRIPT,
    ):
        if script_path.exists():
            subprocess.run([python_bin, str(script_path)], cwd=str(PROJECT_ROOT), check=False)
    # 训练成功 job 对应的 writer 已经在 _reload_matching_writers() 里做过定向 kickstart。
    # 这里不再做一次全量 restart，避免把训练关键路径拖进整批 writer 重载。
    # 后续仍会走 reconcile 与最小 refresh，保证 runtime 真相收口。


def main() -> None:
    ap = argparse.ArgumentParser(description="按分组最优参数执行增量训练")
    ap.add_argument("--profile-json", type=str, default=str(DEFAULT_PROFILE), help="best params JSON 路径")
    ap.add_argument("--groups", type=str, default="v5_short,v5_long,gru_core", help="逗号分隔分组")
    ap.add_argument("--python-bin", type=str, default=sys.executable, help="用于运行训练脚本的 Python")
    ap.add_argument("--dry-run", action="store_true", help="只干跑，不写模型")
    ap.add_argument("--continue-on-error", action="store_true", help="某组失败后继续下一组")
    ap.add_argument("--as-of-timestamp", type=str, default="", help="按指定 UTC/ISO 时刻构建子训练数据窗口，用于历史缺口回放")
    ap.add_argument("--history-timestamp-override", type=str, default="", help="覆盖 grouped/child 历史记录时间戳，用于 coverage 补齐")
    ap.add_argument("--ignore-cadence", action="store_true", help="历史补齐模式下忽略 job cadence 限制")
    ap.add_argument("--disable-checkpoint", action="store_true", help="历史补齐模式下禁用 grouped checkpoint，避免不同历史槽位互相跳过")
    args = ap.parse_args()

    if maintenance_window_active():
        pause_state = load_pause_state()
        reason = str(pause_state.get("reason") or "").strip() or "maintenance_window_active"
        print(f"维护窗生效，grouped 训练暂停（reason={reason}）", flush=True)
        return

    profile_path = Path(args.profile_json)
    if not profile_path.exists():
        raise SystemExit(f"找不到 profile: {profile_path}")

    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    jobs = payload.get("jobs") if isinstance(payload.get("jobs"), list) else None
    mode = str(payload.get("mode") or "shared_or_legacy")
    groups_data = payload.get("groups", {})
    groups: List[str] = []
    if jobs is None:
        groups = [g.strip() for g in args.groups.split(",") if g.strip()]
        for g in groups:
            if g not in GROUP_TARGETS:
                raise SystemExit(f"未知 group: {g}, 可选: {list(GROUP_TARGETS.keys())}")
            if g not in groups_data or not groups_data[g].get("best_params"):
                raise SystemExit(f"profile 缺少 {g}.best_params")
    config_hash, data_window_hash, checkpoint_file = _checkpoint_job_signature(profile_path, payload, groups, jobs)
    matched_checkpoint = {} if args.disable_checkpoint else matching_checkpoint(
        path=checkpoint_file,
        config_hash=config_hash,
        data_window_hash=data_window_hash,
    )
    completed_units = {
        str(item).strip()
        for item in (matched_checkpoint.get("completedUnits") or [])
        if str(item).strip()
    }
    failed_units = list(matched_checkpoint.get("failedUnits") or []) if isinstance(matched_checkpoint.get("failedUnits"), list) else []
    resumed_from_checkpoint = bool(matched_checkpoint and str(matched_checkpoint.get("updatedAt") or "").strip())
    resume_source = str(checkpoint_file) if resumed_from_checkpoint else ""

    print("=== 分组增量训练开始 ===")
    print(f"profile: {profile_path}")
    print(f"python:  {args.python_bin}")
    print(f"mode:    {mode}")
    if jobs is not None:
        print(f"jobs:    {len(jobs)}")
    else:
        print(f"groups:  {groups}")
    print(f"dry_run: {args.dry_run}")
    started_at = time.time()

    if jobs is not None:
        job_results: List[Dict[str, object]] = []
        all_units = [f"job:{str(job.get('job_id') or '')}" for job in jobs if isinstance(job, dict) and str(job.get("job_id") or "").strip()]
        all_units.extend(["stage:runtime_apply", "stage:core_refresh"])

        def _write_training_checkpoint(stage: str, stage_cursor: str, terminal_status: str = "running") -> None:
            if args.disable_checkpoint:
                return
            atomic_write_checkpoint(
                checkpoint_file,
                checkpoint_contract(
                    pipeline_type="grouped_online_learning",
                    job_key=checkpoint_file.stem,
                    model_family="core10_grouped_training",
                    symbol="",
                    combo_key=str(profile_path.stem),
                    config_hash=config_hash,
                    data_window_hash=data_window_hash,
                    stage=stage,
                    stage_cursor=stage_cursor,
                    completed_units=sorted(completed_units),
                    pending_units=[unit for unit in all_units if unit not in completed_units],
                    failed_units=failed_units,
                    last_good_artifacts={
                        "profileJson": str(profile_path),
                        "historyFile": str(ONLINE_HISTORY_FILE),
                    },
                    terminal_status=terminal_status,
                    extra={
                        "resumedFromCheckpoint": resumed_from_checkpoint,
                        "resumeSource": resume_source,
                        "resumeSkippedCompletedUnits": sorted([unit for unit in all_units if unit in completed_units]),
                        "resumeRemainingUnits": [unit for unit in all_units if unit not in completed_units],
                    },
                ),
            )

        _write_training_checkpoint("training_jobs", "", "running")
        for raw_job in jobs:
            if not isinstance(raw_job, dict):
                continue
            job_name = str(raw_job.get("job_id") or "unknown_job")
            unit_name = f"job:{job_name}"
            if unit_name in completed_units:
                job_results.append(
                    {
                        "job_id": job_name,
                        "family": str(raw_job.get("family") or ""),
                        "asset": str(raw_job.get("asset") or ""),
                        "tier": str(raw_job.get("tier") or ""),
                        "lane": str(raw_job.get("lane") or ""),
                        "coverage_status": str(raw_job.get("coverage_status") or ""),
                        "cadence_minutes": int(raw_job.get("cadence_minutes") or 0),
                        "returncode": 0,
                        "skipped": True,
                        "skipped_reason": "checkpoint_completed",
                        "command": [],
                    }
                )
                print(f"\n[{job_name}] 跳过: checkpoint_completed", flush=True)
                continue
            coverage_status = str(raw_job.get("coverage_status") or "").strip()
            if coverage_status == "invalid_bootstrap":
                job_results.append(
                    {
                        "job_id": job_name,
                        "family": str(raw_job.get("family") or ""),
                        "asset": str(raw_job.get("asset") or ""),
                        "tier": str(raw_job.get("tier") or ""),
                        "lane": str(raw_job.get("lane") or ""),
                        "coverage_status": coverage_status,
                        "cadence_minutes": int(raw_job.get("cadence_minutes") or 0),
                        "returncode": 0,
                        "skipped": True,
                        "skipped_reason": "SKIP_INVALID_BOOTSTRAP",
                        "invalid_bootstrap": True,
                        "command": [],
                    }
                )
                print(f"\n[{job_name}] 跳过: invalid_bootstrap", flush=True)
                completed_units.add(unit_name)
                _write_training_checkpoint("training_jobs", job_name, "running")
                continue
            if coverage_status.startswith("deferred"):
                job_results.append(
                    {
                        "job_id": job_name,
                        "family": str(raw_job.get("family") or ""),
                        "asset": str(raw_job.get("asset") or ""),
                        "tier": str(raw_job.get("tier") or ""),
                        "lane": str(raw_job.get("lane") or ""),
                        "coverage_status": coverage_status,
                        "cadence_minutes": int(raw_job.get("cadence_minutes") or 0),
                        "returncode": 0,
                        "skipped": True,
                        "skipped_reason": str(raw_job.get("defer_reason") or "deferred"),
                        "command": [],
                    }
                )
                print(f"\n[{job_name}] 跳过: deferred={coverage_status}")
                completed_units.add(unit_name)
                _write_training_checkpoint("training_jobs", job_name, "running")
                continue
            cadence_minutes = max(0, int(raw_job.get("cadence_minutes") or 0))
            latest_success = _latest_job_success(job_name) if cadence_minutes > 0 else {}
            last_success_ts = latest_success.get("timestamp")
            if isinstance(last_success_ts, datetime) and not args.ignore_cadence:
                elapsed_minutes = (datetime.now(timezone.utc) - last_success_ts).total_seconds() / 60.0
                if elapsed_minutes < float(cadence_minutes):
                    job_results.append(
                        {
                            "job_id": job_name,
                            "family": str(raw_job.get("family") or ""),
                            "asset": str(raw_job.get("asset") or ""),
                            "tier": str(raw_job.get("tier") or ""),
                            "lane": str(raw_job.get("lane") or ""),
                            "coverage_status": str(raw_job.get("coverage_status") or ""),
                            "cadence_minutes": cadence_minutes,
                            "returncode": 0,
                            "skipped": True,
                            "skipped_reason": "cadence_not_due",
                            "last_success_at": last_success_ts.isoformat(),
                            "minutes_until_due": round(float(cadence_minutes) - elapsed_minutes, 2),
                            "command": [],
                        }
                    )
                    print(
                        f"\n[{job_name}] 跳过: cadence={cadence_minutes}m "
                        f"last_success={last_success_ts.isoformat()} elapsed={elapsed_minutes:.1f}m"
                    )
                    completed_units.add(unit_name)
                    _write_training_checkpoint("training_jobs", job_name, "running")
                    continue
            cmd, temp_json_paths = _build_job_cmd(
                raw_job,
                args.python_bin,
                args.dry_run,
                args.as_of_timestamp,
                args.history_timestamp_override,
            )
            print(f"\n[{job_name}] 执行:")
            print("  " + " ".join(cmd))
            history_before = _read_history_lines()
            try:
                ret = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
            finally:
                for temp_json_path in temp_json_paths:
                    try:
                        os.unlink(temp_json_path)
                    except OSError:
                        pass
            child_summary: Dict[str, object] = {}
            child_results_count = 0
            history_after = _read_history_lines()
            if len(history_after) > len(history_before):
                for raw in reversed(history_after[len(history_before):]):
                    payload = _parse_history_line(raw)
                    if not payload:
                        continue
                    if payload.get("entry_type") == "grouped_online_learning":
                        continue
                    if str(payload.get("job_id") or "").strip() != job_name:
                        continue
                    summary = payload.get("summary")
                    results = payload.get("results")
                    if isinstance(summary, dict):
                        child_summary = {
                            "success": int(summary.get("success", 0) or 0),
                            "trained_but_not_written": int(summary.get("trained_but_not_written", 0) or 0),
                            "skipped": int(summary.get("skipped", 0) or 0),
                        }
                    if isinstance(results, list):
                        child_results_count = len(results)
                    break
            job_results.append(
                {
                    "job_id": job_name,
                    "family": str(raw_job.get("family") or ""),
                    "asset": str(raw_job.get("asset") or ""),
                    "tier": str(raw_job.get("tier") or ""),
                    "lane": str(raw_job.get("lane") or ""),
                    "coverage_status": str(raw_job.get("coverage_status") or ""),
                    "cadence_minutes": cadence_minutes,
                    "exp_ids": [int(x) for x in (raw_job.get("exp_ids") or [])],
                    "model_dir_map": dict(raw_job.get("model_dir_map") or {}),
                    "returncode": int(ret.returncode),
                    "child_summary": child_summary,
                    "child_results_count": child_results_count,
                    "command": cmd,
                }
            )
            if int(ret.returncode) == 0:
                completed_units.add(unit_name)
            else:
                failed_units.append({"unit": unit_name, "detail": f"rc={int(ret.returncode)}"})
            _write_training_checkpoint("training_jobs", job_name, "running")
            if ret.returncode != 0:
                msg = f"[{job_name}] 失败: rc={ret.returncode}"
                if not args.continue_on_error:
                    _append_group_history_entry(
                        profile_path,
                        mode,
                        args.dry_run,
                        job_results,
                        summary_updates={
                            "training_success": False,
                            "runtime_apply_success": False,
                            "runtime_apply_status": "not_started",
                            "core_refresh_success": False,
                            "nonblocking_refresh_status": "not_started",
                            "history_source": "grouped_online_learning",
                            "result_class": "failed_critical_path",
                            "reload_mode": "targeted_reload",
                            "reconcile_status": "not_started",
                            "stable_writer_count": 0,
                            "stale_writer_count": 0,
                            "max_wait_exceeded": False,
                            "refresh_profile": "core10_minimal_only",
                            "critical_path_seconds": round(max(0.0, time.time() - started_at), 3),
                        },
                        timestamp_override=args.history_timestamp_override,
                    )
                    raise SystemExit(msg)
                print(msg)
        success_job_results = []
        for row in job_results:
            child_summary = row.get("child_summary") if isinstance(row.get("child_summary"), dict) else {}
            if int(child_summary.get("success", 0) or 0) > 0 and int(row.get("returncode") or 0) == 0:
                success_job_results.append(row)
        writers_to_reload, writers_reloaded, writer_reload_errors = ([], [], [])
        runtime_apply_success = True
        runtime_apply_status = "not_applicable" if args.dry_run else "ok"
        reconcile_status = "not_started"
        stale_writer_count = 0
        max_wait_exceeded = False
        if not args.dry_run:
            if "stage:runtime_apply" in completed_units:
                runtime_apply_status = "resumed_completed"
            else:
                writers_to_reload, writers_reloaded, writer_reload_errors = _reload_matching_writers(success_job_results)
                _write_training_checkpoint("runtime_apply", "reload_writers", "running")
                _apply_core10_runtime_promotions(args.python_bin, profile_path, mode, success_job_results)
                if writers_to_reload:
                    if writer_reload_errors or len(writers_reloaded) != len(writers_to_reload):
                        runtime_apply_success = False
                        runtime_apply_status = "failed_runtime_apply"
                    elif writers_reloaded:
                        reconcile = _reconcile_core10_writer_revisions()
                        reconcile_status = str(reconcile.get("status") or "failed")
                        stale_writer_count = int(reconcile.get("stale_writer_count") or 0)
                        max_wait_exceeded = bool(reconcile.get("max_wait_exceeded"))
                        if reconcile_status == "failed":
                            runtime_apply_success = False
                            runtime_apply_status = "failed_runtime_apply"
                            writer_reload_errors.append(f"reconcile_failed:{str(reconcile.get('details') or '')[:300]}")
                        elif reconcile_status == "partial":
                            runtime_apply_success = True
                            runtime_apply_status = "ok_with_observation"
                            writer_reload_errors.append(f"reconcile_partial:{str(reconcile.get('details') or '')[:300]}")
                        else:
                            runtime_apply_success = True
                            runtime_apply_status = "ok"
                else:
                    runtime_apply_status = "not_applicable"
        refresh_ok = bool(args.dry_run)
        if not args.dry_run:
            if not runtime_apply_success:
                _write_training_checkpoint("runtime_apply", "failed", "failed_runtime_apply")
                _append_group_history_entry(
                    profile_path,
                    mode,
                    args.dry_run,
                    job_results,
                    writers_to_reload=writers_to_reload,
                    writers_reloaded=writers_reloaded,
                    writer_reload_errors=writer_reload_errors,
                    summary_updates={
                        "training_success": True,
                        "runtime_apply_success": False,
                        "runtime_apply_status": runtime_apply_status,
                        "core_refresh_success": False,
                        "nonblocking_refresh_status": "not_started",
                        "history_source": "grouped_online_learning",
                        "result_class": _classify_result_class(
                            training_success=True,
                            runtime_apply_success=False,
                            core_refresh_success=False,
                            runtime_apply_status=runtime_apply_status,
                            nonblocking_refresh_status="not_started",
                        ),
                        "reload_mode": "targeted_reload",
                        "reconcile_status": reconcile_status,
                        "stable_writer_count": len(writers_reloaded),
                        "stale_writer_count": stale_writer_count,
                        "max_wait_exceeded": max_wait_exceeded,
                        "refresh_profile": "core10_minimal_only",
                        "critical_path_seconds": round(max(0.0, time.time() - started_at), 3),
                    },
                    timestamp_override=args.history_timestamp_override,
                )
                raise SystemExit("core10 writer runtime apply failed")
            completed_units.add("stage:runtime_apply")
            if "stage:core_refresh" in completed_units:
                refresh_ok = True
            else:
                _write_training_checkpoint("core_refresh", "minimal_refresh", "running")
                refresh_ok = _refresh_core10_reports(args.python_bin, profile_path, mode)
            if not refresh_ok:
                _write_training_checkpoint("core_refresh", "failed", "failed_core_refresh")
                _append_group_history_entry(
                    profile_path,
                    mode,
                    args.dry_run,
                    job_results,
                    writers_to_reload=writers_to_reload,
                    writers_reloaded=writers_reloaded,
                    writer_reload_errors=writer_reload_errors,
                    summary_updates={
                        "training_success": True,
                        "runtime_apply_success": True,
                        "runtime_apply_status": runtime_apply_status,
                        "core_refresh_success": False,
                        "nonblocking_refresh_status": "not_started",
                        "history_source": "grouped_online_learning",
                        "result_class": _classify_result_class(
                            training_success=True,
                            runtime_apply_success=True,
                            core_refresh_success=False,
                            runtime_apply_status=runtime_apply_status,
                            nonblocking_refresh_status="not_started",
                        ),
                        "reload_mode": "targeted_reload",
                        "reconcile_status": reconcile_status,
                        "stable_writer_count": len(writers_reloaded),
                        "stale_writer_count": stale_writer_count,
                        "max_wait_exceeded": max_wait_exceeded,
                        "refresh_profile": "core10_minimal_only",
                        "critical_path_seconds": round(max(0.0, time.time() - started_at), 3),
                    },
                    timestamp_override=args.history_timestamp_override,
                )
                raise SystemExit("core10 grouped refresh chain failed")
            completed_units.add("stage:core_refresh")
        _append_group_history_entry(
            profile_path,
            mode,
            args.dry_run,
            job_results,
            writers_to_reload=writers_to_reload,
            writers_reloaded=writers_reloaded,
            writer_reload_errors=writer_reload_errors,
            summary_updates={
                "training_success": True,
                "runtime_apply_success": True,
                "runtime_apply_status": runtime_apply_status,
                "core_refresh_success": refresh_ok,
                "nonblocking_refresh_status": "not_started",
                "history_source": "grouped_online_learning",
                "result_class": _classify_result_class(
                    training_success=True,
                    runtime_apply_success=True,
                    core_refresh_success=refresh_ok,
                    runtime_apply_status=runtime_apply_status,
                    nonblocking_refresh_status="not_started",
                ),
                "reload_mode": "targeted_reload",
                "reconcile_status": reconcile_status,
                "stable_writer_count": len(writers_reloaded),
                "stale_writer_count": stale_writer_count,
                "max_wait_exceeded": max_wait_exceeded,
                "refresh_profile": "core10_minimal_only",
                "critical_path_seconds": round(max(0.0, time.time() - started_at), 3),
                "resumed_from_checkpoint": resumed_from_checkpoint,
                "resume_source": resume_source,
            },
            timestamp_override=args.history_timestamp_override,
        )
        _write_training_checkpoint("completed", "", "completed")
    else:
        for g in groups:
            params = groups_data[g]["best_params"]
            cmd = _build_cmd(g, params, args.python_bin, args.dry_run)
            print(f"\n[{g}] 执行:")
            print("  " + " ".join(cmd))
            ret = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
            if ret.returncode != 0:
                msg = f"[{g}] 失败: rc={ret.returncode}"
                if not args.continue_on_error:
                    raise SystemExit(msg)
                print(msg)

    print("\n=== 分组增量训练完成 ===")


if __name__ == "__main__":
    main()
