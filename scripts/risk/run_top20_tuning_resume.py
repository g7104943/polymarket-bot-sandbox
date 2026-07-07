#!/usr/bin/env python3
from __future__ import annotations

import argparse
import threading
import json
import os
import shutil
import subprocess
import tempfile
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
POLYMARKET_DIR = PROJECT_ROOT / "polymarket"
PYTHON_BIN = sys.executable or "python3"

ACTIVE_DEFAULT = POLYMARKET_DIR / "active_traders.json"
ACTIVE_70 = POLYMARKET_DIR / "active_traders_70.json"
CFG_DEFAULT = POLYMARKET_DIR / "trader_configs.json"
CFG_70 = POLYMARKET_DIR / "trader_configs_70.json"

LAYER_TO_SCRIPT = {
    "expectancy": PROJECT_ROOT / "scripts" / "risk" / "optimize_expectancy_gate.py",
    "calibration": PROJECT_ROOT / "scripts" / "risk" / "optimize_direction_calibration.py",
    "up": PROJECT_ROOT / "scripts" / "risk" / "optimize_up_risk_v2.py",
    "down": PROJECT_ROOT / "scripts" / "risk" / "optimize_down_risk_v2.py",
    "combo": PROJECT_ROOT / "scripts" / "risk" / "optimize_combo_pause_v1.py",
    "shock": PROJECT_ROOT / "scripts" / "risk" / "optimize_shock_risk_v2.py",
}
LAYER_TO_PANEL = {
    "expectancy": REPORTS_DIR / "expectancy_gate_panel.parquet",
    "calibration": REPORTS_DIR / "direction_calibration_panel.parquet",
    "up": REPORTS_DIR / "up_risk_v2_panel.parquet",
    "down": REPORTS_DIR / "down_risk_v2_panel.parquet",
    "combo": REPORTS_DIR / "combo_pause_v1_panel.parquet",
    "shock": REPORTS_DIR / "shock_risk_v2_panel.parquet",
}
LAYER_TO_PANEL_BUILDER = {
    "combo": PROJECT_ROOT / "scripts" / "risk" / "build_combo_pause_panel.py",
}


@dataclass
class TraderRank:
    profile: str
    trader_name: str
    logs_dir: str
    pnl_total: float
    trades: int


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _active_names(path: Path) -> List[str]:
    raw = _load_json(path)
    if isinstance(raw, dict):
        for k in ("active_traders", "traderNames"):
            v = raw.get(k)
            if isinstance(v, list):
                return [str(x) for x in v]
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return []


def _config_map(path: Path) -> Dict[str, Dict[str, Any]]:
    raw = _load_json(path)
    if not isinstance(raw, list):
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for row in raw:
        if isinstance(row, dict) and row.get("name"):
            out[str(row["name"])] = row
    return out


def _trader_pnl(logs_dir: str, mode: str) -> tuple[float, int]:
    trade_file = POLYMARKET_DIR / logs_dir / f"prediction_trades.{mode}.json"
    if not trade_file.exists():
        return 0.0, 0
    try:
        rows = _load_json(trade_file)
    except Exception:
        return 0.0, 0
    if not isinstance(rows, list):
        return 0.0, 0
    pnl = 0.0
    n = 0
    for r in rows:
        if not isinstance(r, dict):
            continue
        try:
            v = float(r.get("pnl"))
        except Exception:
            continue
        pnl += v
        n += 1
    return float(pnl), int(n)


def rank_top_traders(
    top_k: int,
    mode: str,
    include_ensemble: bool = True,
    exclude_gru: bool = False,
) -> List[TraderRank]:
    cfg_default = _config_map(CFG_DEFAULT)
    cfg_70 = _config_map(CFG_70)
    active_default = _active_names(ACTIVE_DEFAULT)
    active_70 = _active_names(ACTIVE_70)

    ranked: List[TraderRank] = []
    for profile, names, cfg_map in (
        ("default", active_default, cfg_default),
        ("70", active_70, cfg_70),
    ):
        rows: List[TraderRank] = []
        for name in names:
            lname = str(name).lower()
            if (not include_ensemble) and ("ensemble" in lname or lname.startswith("ens_") or lname.startswith("v5_ensemble")):
                continue
            if exclude_gru and ("gru" in lname):
                continue
            cfg = cfg_map.get(name)
            if not cfg:
                continue
            logs_dir = str(cfg.get("logsDir") or "").strip()
            if not logs_dir:
                continue
            pnl, n = _trader_pnl(logs_dir, mode=mode)
            rows.append(
                TraderRank(
                    profile=profile,
                    trader_name=name,
                    logs_dir=logs_dir,
                    pnl_total=pnl,
                    trades=n,
                )
            )
        rows.sort(key=lambda x: x.pnl_total, reverse=True)
        ranked.extend(rows[:top_k])
    return ranked


def _cluster_args_from_panel(panel_path: Path, profile: str, layer: str) -> str:
    df = pd.read_parquet(panel_path)
    if df.empty:
        return ""
    if layer in {"expectancy", "calibration", "combo"}:
        vals = sorted({str(x) for x in df.get("cluster_id", pd.Series(dtype=str)).dropna().astype(str).tolist() if x})
    else:
        symbols = sorted({str(x).upper() for x in df.get("symbol", pd.Series(dtype=str)).dropna().astype(str).tolist() if x})
        vals = [f"{profile}_{s}" for s in symbols if s in {"BTC", "ETH"}]
    return ",".join(vals)


def _build_cmd(
    layer: str,
    panel_path: Path,
    output_path: Path,
    checkpoint_dir: Path,
    profile: str,
    workers: int,
    window_days: int,
    train_min_days: int,
    test_days: int,
    step_days: int,
    purge_hours: int,
    half_life_days: int,
    recent_days: int,
) -> List[str]:
    script = LAYER_TO_SCRIPT[layer]
    clusters = _cluster_args_from_panel(panel_path, profile, layer)
    # Use the same interpreter as the orchestrator so workers don't fall back
    # to the system Python when launched under launchd.
    cmd = [PYTHON_BIN, str(script), "--panel", str(panel_path), "--output", str(output_path)]
    if clusters:
        cmd += ["--clusters", clusters]
    cmd += [
        "--window-days",
        str(int(window_days)),
        "--train-min-days",
        str(int(train_min_days)),
        "--test-days",
        str(int(test_days)),
        "--step-days",
        str(int(step_days)),
        "--purge-hours",
        str(int(purge_hours)),
    ]
    if layer in {"up", "down", "shock", "combo"}:
        cmd += [
            "--half-life-days",
            str(int(half_life_days)),
            "--recent-days",
            str(int(recent_days)),
        ]
    if layer in {"up", "down", "shock", "combo"}:
        cmd += ["--checkpoint-dir", str(checkpoint_dir)]
    if layer == "shock":
        cmd += ["--workers", str(max(1, workers))]
    return cmd


def _tail_text(path: Path, max_chars: int = 4000) -> str:
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    return txt[-max_chars:]


def _terminate_process(proc: subprocess.Popen[Any]) -> None:
    try:
        proc.terminate()
    except Exception:
        return
    try:
        proc.wait(timeout=5)
        return
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        return
    try:
        proc.wait(timeout=3)
    except Exception:
        pass


def _task_signal_ts(task_row: Dict[str, Any]) -> float:
    latest = 0.0
    try:
        output = task_row.get("output")
        if isinstance(output, str) and output:
            op = Path(output)
            if op.exists():
                latest = max(latest, float(op.stat().st_mtime))
    except Exception:
        pass
    try:
        checkpoint_dir = task_row.get("checkpoint_dir")
        if isinstance(checkpoint_dir, str) and checkpoint_dir:
            cp = Path(checkpoint_dir)
            if cp.exists() and cp.is_dir():
                for pat in ("**/*.json", "**/*.parquet", "**/*.csv"):
                    for fp in cp.glob(pat):
                        if not fp.is_file():
                            continue
                        try:
                            ts = float(fp.stat().st_mtime)
                        except Exception:
                            continue
                        if ts > latest:
                            latest = ts
    except Exception:
        pass
    return latest


def _run_cmd_once_watchdog(
    *,
    cmd: List[str],
    child_env: Dict[str, str],
    task_row: Dict[str, Any],
    timeout_sec: int,
    no_progress_sec: int,
    poll_sec: int = 10,
) -> Dict[str, Any]:
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", prefix="top20_task_stdout_", suffix=".log", delete=False) as so, \
         tempfile.NamedTemporaryFile("w+", encoding="utf-8", prefix="top20_task_stderr_", suffix=".log", delete=False) as se:
        so_path = Path(so.name)
        se_path = Path(se.name)
        proc = subprocess.Popen(
            cmd,
            stdout=so,
            stderr=se,
            text=True,
            env=child_env,
        )
    started_ts = time.time()
    last_signal_seen = _task_signal_ts(task_row)
    last_progress_ts = started_ts
    if last_signal_seen > 0:
        last_progress_ts = max(last_progress_ts, last_signal_seen)
    stall_reason = ""
    try:
        while True:
            rc = proc.poll()
            now_ts = time.time()
            sig_ts = _task_signal_ts(task_row)
            if sig_ts > last_signal_seen:
                last_signal_seen = sig_ts
                last_progress_ts = now_ts

            elapsed = now_ts - started_ts
            no_progress_now = now_ts - last_progress_ts
            if rc is None and elapsed > timeout_sec:
                stall_reason = "task_timeout"
                _terminate_process(proc)
                break
            if rc is None and no_progress_sec > 0 and no_progress_now > no_progress_sec:
                stall_reason = "no_progress"
                _terminate_process(proc)
                break
            if rc is not None:
                break
            time.sleep(max(1, int(poll_sec)))
    finally:
        rc = proc.poll()
        if rc is None:
            _terminate_process(proc)
            rc = proc.poll()
        rc = int(rc if rc is not None else 1)
        elapsed_sec = int(max(0, time.time() - started_ts))
        stdout_tail = _tail_text(so_path)
        stderr_tail = _tail_text(se_path)
        if stall_reason == "task_timeout":
            rc = 124
            stderr_tail = (f"task_timeout>{timeout_sec}s\n{stderr_tail}")[-4000:]
        elif stall_reason == "no_progress":
            rc = 125
            stderr_tail = (f"task_no_progress>{no_progress_sec}s\n{stderr_tail}")[-4000:]
        try:
            so_path.unlink(missing_ok=True)
            se_path.unlink(missing_ok=True)
        except Exception:
            pass
    last_progress_at = None
    if last_signal_seen > 0:
        last_progress_at = datetime.fromtimestamp(last_signal_seen, tz=timezone.utc).isoformat()
    else:
        last_progress_at = datetime.fromtimestamp(started_ts, tz=timezone.utc).isoformat()
    return {
        "rc": int(rc),
        "elapsed_sec": int(elapsed_sec),
        "stdout_tail": stdout_tail[-4000:],
        "stderr_tail": stderr_tail[-4000:],
        "stall_reason": stall_reason,
        "last_progress_at": last_progress_at,
        "no_progress_sec": int(max(0, time.time() - (last_signal_seen if last_signal_seen > 0 else started_ts))),
    }


def _run_cmd_with_restarts(
    *,
    cmd: List[str],
    child_env: Dict[str, str],
    task_row: Dict[str, Any],
    timeout_sec: int,
    no_progress_sec: int,
    max_restarts: int,
) -> Dict[str, Any]:
    restart_count = 0
    last_res: Dict[str, Any] = {}
    while True:
        last_res = _run_cmd_once_watchdog(
            cmd=cmd,
            child_env=child_env,
            task_row=task_row,
            timeout_sec=timeout_sec,
            no_progress_sec=no_progress_sec,
        )
        if int(last_res.get("rc", 1)) == 0:
            break
        if str(last_res.get("stall_reason") or "") == "no_progress" and restart_count < max(0, int(max_restarts)):
            restart_count += 1
            continue
        break
    last_res["restart_count"] = int(restart_count)
    return last_res


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _manifest_matches_args(manifest: Dict[str, Any], args: argparse.Namespace, selected_layers: Sequence[str]) -> bool:
    try:
        m_mode = str(manifest.get("mode") or "")
        m_layers = [str(x) for x in (manifest.get("layers") or [])]
        if m_mode != str(args.mode):
            return False
        if set(m_layers) != set(selected_layers):
            return False
        return True
    except Exception:
        return False


def _prepare_resume_tasks(
    manifest: Dict[str, Any],
    *,
    selected_layers: Sequence[str],
    force_layers: set[str],
    dry_run: bool,
) -> List[Dict[str, Any]]:
    def _parse_ts(raw: Any) -> float | None:
        try:
            if not raw:
                return None
            return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()
        except Exception:
            return None

    selected = set(selected_layers)
    runnable: List[Dict[str, Any]] = []
    terminal = {"done", "failed", "timeout", "missing_panel", "no_rows", "planned"}
    for tr in manifest.get("traders", []):
        if not isinstance(tr, dict):
            continue
        tasks = tr.get("tasks")
        if not isinstance(tasks, list):
            continue
        for row in tasks:
            if not isinstance(row, dict):
                continue
            layer = str(row.get("layer") or "")
            if layer not in selected:
                continue
            output_path = Path(str(row.get("output") or "")) if row.get("output") else None
            status = str(row.get("status") or "").strip().lower()
            # 上次中断时 running 任务，恢复后应重新入队。
            if status == "running":
                status = "queued"
            # force layer: 无条件重跑该层（删除旧输出）。
            if layer in force_layers:
                if output_path and output_path.exists():
                    output_path.unlink(missing_ok=True)
                checkpoint_dir = Path(str(row.get("checkpoint_dir") or "")) if row.get("checkpoint_dir") else None
                if checkpoint_dir and checkpoint_dir.exists():
                    shutil.rmtree(checkpoint_dir, ignore_errors=True)
                status = "queued"
                row["forced_rerun"] = True
            # 若任务曾失败/超时，但之后已生成更新过的输出文件，说明结果已被外部续跑补齐；
            # 恢复时应认定为 done，避免 manifest 与实际产物脱节。
            finished_ts = _parse_ts(row.get("finished_at")) or _parse_ts(row.get("started_at"))
            output_mtime = output_path.stat().st_mtime if output_path and output_path.exists() else None
            if status in {"failed", "timeout"} and output_mtime and (finished_ts is None or output_mtime >= finished_ts):
                status = "done"
                row["returncode"] = 0
                row["stall_reason"] = ""
            # 非强制模式下，输出存在即视为 done。
            if (layer not in force_layers) and output_path and output_path.exists():
                status = "done"
            # 标记为 done 但文件丢失，回退为 queued。
            if status == "done" and (not output_path or not output_path.exists()):
                status = "queued"
            if dry_run and status not in terminal:
                status = "planned"
            row["status"] = status
            if (not dry_run) and status == "queued":
                runnable.append(row)
    return runnable


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Top20 分批超参（可续跑）")
    ap.add_argument("--top-k-per-profile", type=int, default=10)
    ap.add_argument("--mode", choices=["simulation", "live"], default="simulation")
    ap.add_argument("--layers", type=str, default="expectancy,calibration,up,down,combo,shock")
    ap.add_argument("--workers", type=int, default=4, help="仅用于 SHOCK 内并行")
    ap.add_argument("--task-workers", type=int, default=1, help="跨 trader+layer 的任务并行数")
    ap.add_argument("--task-timeout-sec", type=int, default=4500, help="单任务超时秒数（默认 4500 = 75 分钟）")
    ap.add_argument("--task-no-progress-sec", type=int, default=1500, help="单任务无进展超时秒数（默认 1500 = 25 分钟）")
    ap.add_argument("--task-max-restarts", type=int, default=1, help="无进展自动重启次数（仅该任务）")
    ap.add_argument("--output-dir", type=Path, default=REPORTS_DIR / "top20_tuning_resume")
    ap.add_argument("--include-ensemble", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--exclude-gru", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--exclude-ensemble", action="store_true")
    ap.add_argument("--window-days", type=int, default=365)
    ap.add_argument("--train-min-days", type=int, default=180)
    ap.add_argument("--test-days", type=int, default=14)
    ap.add_argument("--step-days", type=int, default=14)
    ap.add_argument("--purge-hours", type=int, default=24)
    ap.add_argument("--half-life-days", type=int, default=30)
    ap.add_argument("--recent-days", type=int, default=21)
    ap.add_argument("--force-layers", type=str, default="", help="comma list, e.g. up,down")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    include_ensemble = bool(args.include_ensemble) and (not bool(args.exclude_ensemble))
    selected_layers = [x.strip() for x in str(args.layers).split(",") if x.strip() in LAYER_TO_SCRIPT]
    if not selected_layers:
        raise RuntimeError("layers empty after filter")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    force_layers = {x.strip() for x in str(args.force_layers).split(",") if x.strip()}
    task_workers = max(1, int(args.task_workers))
    task_timeout_sec = max(60, int(args.task_timeout_sec))
    task_no_progress_sec = max(60, int(args.task_no_progress_sec))
    task_max_restarts = max(0, int(args.task_max_restarts))

    # 优先从已有 manifest 续跑，避免“重启后重新选 top 集合”导致任务漂移和重复计算。
    if manifest_path.exists():
        try:
            existing = _load_json(manifest_path)
            if isinstance(existing, dict) and _manifest_matches_args(existing, args, selected_layers):
                existing["resumed_at"] = datetime.now(timezone.utc).isoformat()
                existing["manifest_complete"] = False
                existing["task_workers"] = int(task_workers)
                runnable_tasks = _prepare_resume_tasks(
                    existing,
                    selected_layers=selected_layers,
                    force_layers=force_layers,
                    dry_run=bool(args.dry_run),
                )
                _write_json(manifest_path, existing)
                if (not args.dry_run) and runnable_tasks:
                    write_lock = threading.Lock()
                    child_env = os.environ.copy()
                    # 多任务并发下，限制 BLAS/OpenMP 线程，避免线程过度嵌套拖慢整体吞吐。
                    child_env.setdefault("OMP_NUM_THREADS", "1")
                    child_env.setdefault("MKL_NUM_THREADS", "1")
                    child_env.setdefault("NUMEXPR_NUM_THREADS", "1")

                    def _run_one(task_row: Dict[str, Any]) -> int:
                        cmd = task_row.get("cmd") or []
                        with write_lock:
                            task_row["status"] = "running"
                            task_row["started_at"] = datetime.now(timezone.utc).isoformat()
                            task_row["task_timeout_sec"] = int(task_timeout_sec)
                            task_row["task_no_progress_sec"] = int(task_no_progress_sec)
                            task_row["restart_count"] = int(task_row.get("restart_count") or 0)
                            _write_json(manifest_path, existing)
                        run_res = _run_cmd_with_restarts(
                            cmd=cmd,
                            child_env=child_env,
                            task_row=task_row,
                            timeout_sec=task_timeout_sec,
                            no_progress_sec=task_no_progress_sec,
                            max_restarts=task_max_restarts,
                        )
                        rc = int(run_res.get("rc", 1))
                        stall_reason = str(run_res.get("stall_reason") or "")
                        with write_lock:
                            task_row["status"] = "timeout" if stall_reason == "task_timeout" else ("done" if rc == 0 else "failed")
                            task_row["stall_reason"] = stall_reason
                            task_row["returncode"] = int(rc)
                            task_row["stdout_tail"] = str(run_res.get("stdout_tail") or "")
                            task_row["stderr_tail"] = str(run_res.get("stderr_tail") or "")
                            task_row["task_elapsed_sec"] = int(run_res.get("elapsed_sec") or 0)
                            task_row["last_progress_at"] = run_res.get("last_progress_at")
                            task_row["no_progress_sec"] = int(run_res.get("no_progress_sec") or 0)
                            task_row["restart_count"] = int(run_res.get("restart_count") or 0)
                            task_row["finished_at"] = datetime.now(timezone.utc).isoformat()
                            _write_json(manifest_path, existing)
                        return int(rc)

                    if task_workers <= 1:
                        for row in runnable_tasks:
                            _run_one(row)
                    else:
                        with ThreadPoolExecutor(max_workers=task_workers) as pool:
                            futures = [pool.submit(_run_one, row) for row in runnable_tasks]
                            for _ in as_completed(futures):
                                pass

                existing["manifest_complete"] = True
                existing["completed_at"] = datetime.now(timezone.utc).isoformat()
                _write_json(manifest_path, existing)
                print(f"[OK] resumed manifest: {manifest_path}")
                done = 0
                failed = 0
                timeouted = 0
                planned = 0
                for tr in existing.get("traders", []):
                    for task in tr.get("tasks", []):
                        st = str(task.get("status"))
                        if st == "done":
                            done += 1
                        elif st == "failed":
                            failed += 1
                        elif st == "timeout":
                            timeouted += 1
                        elif st == "planned":
                            planned += 1
                print(f"[SUMMARY] done={done} failed={failed} timeout={timeouted} planned={planned}")
                return 0
        except Exception:
            # manifest 读取/校验异常时，回退到全量构建逻辑
            pass

    top_traders = rank_top_traders(
        top_k=int(args.top_k_per_profile),
        mode=str(args.mode),
        include_ensemble=include_ensemble,
        exclude_gru=bool(args.exclude_gru),
    )
    if not top_traders:
        raise RuntimeError("no top traders found from active configs")

    expected_traders_by_profile = {
        "default": int(sum(1 for t in top_traders if t.profile == "default")),
        "70": int(sum(1 for t in top_traders if t.profile == "70")),
    }
    expected_total_traders = int(len(top_traders))
    expected_total_tasks = int(expected_total_traders * len(selected_layers))

    manifest: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": str(args.mode),
        "top_k_per_profile": int(args.top_k_per_profile),
        "include_ensemble": include_ensemble,
        "exclude_gru": bool(args.exclude_gru),
        "layers": selected_layers,
        "expected_traders_by_profile": expected_traders_by_profile,
        "expected_total_traders": expected_total_traders,
        "expected_total_tasks": expected_total_tasks,
        "task_workers": int(task_workers),
        "manifest_complete": False,
        "cv": {
            "window_days": int(args.window_days),
            "train_min_days": int(args.train_min_days),
            "test_days": int(args.test_days),
            "step_days": int(args.step_days),
            "purge_hours": int(args.purge_hours),
            "half_life_days": int(args.half_life_days),
            "recent_days": int(args.recent_days),
            "task_timeout_sec": int(task_timeout_sec),
            "task_no_progress_sec": int(task_no_progress_sec),
            "task_max_restarts": int(task_max_restarts),
        },
        "traders": [],
    }
    runnable_tasks: List[Dict[str, Any]] = []
    panel_cache: Dict[str, pd.DataFrame] = {}
    panel_indices: Dict[str, Dict[tuple[str, str], Any]] = {}

    # 只读一次大 panel，后续按 profile+traderName 索引取切片，减少重复 I/O 与全表扫描。
    for layer in selected_layers:
        base_panel = LAYER_TO_PANEL[layer]
        if not base_panel.exists():
            builder = LAYER_TO_PANEL_BUILDER.get(layer)
            if builder is not None and builder.exists() and not args.dry_run:
                subprocess.run([PYTHON_BIN, str(builder), "--output", str(base_panel)], check=False)
        if not base_panel.exists():
            continue
        df0 = pd.read_parquet(base_panel)
        panel_cache[layer] = df0
        if {"profile", "traderName"}.issubset(df0.columns):
            key_df = pd.DataFrame(
                {
                    "_profile": df0["profile"].astype(str),
                    "_trader": df0["traderName"].astype(str),
                }
            )
            panel_indices[layer] = {
                (str(k[0]), str(k[1])): v
                for k, v in key_df.groupby(["_profile", "_trader"], sort=False).indices.items()
            }
        else:
            panel_indices[layer] = {}

    for tr in top_traders:
        trader_root = output_dir / tr.profile / tr.trader_name
        trader_root.mkdir(parents=True, exist_ok=True)
        task_rows: List[Dict[str, Any]] = []
        trader_entry = {
            "profile": tr.profile,
            "trader_name": tr.trader_name,
            "logs_dir": tr.logs_dir,
            "pnl_total": tr.pnl_total,
            "trades": tr.trades,
            "tasks": task_rows,
        }
        manifest["traders"].append(trader_entry)
        # Persist trader registration early so callers can observe progress while running.
        _write_json(manifest_path, manifest)
        for layer in selected_layers:
            base_panel = LAYER_TO_PANEL[layer]
            if not base_panel.exists():
                task_rows.append({"layer": layer, "status": "missing_panel", "panel": str(base_panel)})
                continue
            df0 = panel_cache.get(layer)
            idx = panel_indices.get(layer, {}).get((tr.profile, tr.trader_name))
            if df0 is None or idx is None or len(idx) == 0:
                task_rows.append({"layer": layer, "status": "no_rows"})
                continue
            df = df0.iloc[idx]
            panel_path = trader_root / f"{layer}_panel.parquet"
            output_path = trader_root / f"{layer}_report.json"
            checkpoint_dir = trader_root / f"{layer}_checkpoint"
            if not panel_path.exists():
                df.to_parquet(panel_path, index=False)
            cmd = _build_cmd(
                layer=layer,
                panel_path=panel_path,
                output_path=output_path,
                checkpoint_dir=checkpoint_dir,
                profile=tr.profile,
                workers=int(args.workers),
                window_days=int(args.window_days),
                train_min_days=int(args.train_min_days),
                test_days=int(args.test_days),
                step_days=int(args.step_days),
                purge_hours=int(args.purge_hours),
                half_life_days=int(args.half_life_days),
                recent_days=int(args.recent_days),
            )
            row = {
                "layer": layer,
                "panel": str(panel_path),
                "output": str(output_path),
                "checkpoint_dir": str(checkpoint_dir),
                "cmd": cmd,
            }
            task_rows.append(row)
            if output_path.exists() and layer not in force_layers:
                row["status"] = "done"
            elif args.dry_run:
                row["status"] = "planned"
            else:
                if output_path.exists() and layer in force_layers:
                    output_path.unlink(missing_ok=True)
                    row["forced_rerun"] = True
                row["status"] = "queued"
                runnable_tasks.append(row)
        _write_json(manifest_path, manifest)

    if (not args.dry_run) and runnable_tasks:
        write_lock = threading.Lock()
        child_env = os.environ.copy()
        child_env.setdefault("OMP_NUM_THREADS", "1")
        child_env.setdefault("MKL_NUM_THREADS", "1")
        child_env.setdefault("NUMEXPR_NUM_THREADS", "1")

        def _run_one(task_row: Dict[str, Any]) -> int:
            cmd = task_row.get("cmd") or []
            with write_lock:
                task_row["status"] = "running"
                task_row["started_at"] = datetime.now(timezone.utc).isoformat()
                task_row["task_timeout_sec"] = int(task_timeout_sec)
                task_row["task_no_progress_sec"] = int(task_no_progress_sec)
                task_row["restart_count"] = int(task_row.get("restart_count") or 0)
                _write_json(manifest_path, manifest)
            run_res = _run_cmd_with_restarts(
                cmd=cmd,
                child_env=child_env,
                task_row=task_row,
                timeout_sec=task_timeout_sec,
                no_progress_sec=task_no_progress_sec,
                max_restarts=task_max_restarts,
            )
            rc = int(run_res.get("rc", 1))
            stall_reason = str(run_res.get("stall_reason") or "")
            with write_lock:
                task_row["status"] = "timeout" if stall_reason == "task_timeout" else ("done" if rc == 0 else "failed")
                task_row["stall_reason"] = stall_reason
                task_row["returncode"] = int(rc)
                task_row["stdout_tail"] = str(run_res.get("stdout_tail") or "")
                task_row["stderr_tail"] = str(run_res.get("stderr_tail") or "")
                task_row["task_elapsed_sec"] = int(run_res.get("elapsed_sec") or 0)
                task_row["last_progress_at"] = run_res.get("last_progress_at")
                task_row["no_progress_sec"] = int(run_res.get("no_progress_sec") or 0)
                task_row["restart_count"] = int(run_res.get("restart_count") or 0)
                task_row["finished_at"] = datetime.now(timezone.utc).isoformat()
                _write_json(manifest_path, manifest)
            return int(rc)

        if task_workers <= 1:
            for row in runnable_tasks:
                _run_one(row)
        else:
            with ThreadPoolExecutor(max_workers=task_workers) as pool:
                futures = [pool.submit(_run_one, row) for row in runnable_tasks]
                for _ in as_completed(futures):
                    pass

    _write_json(manifest_path, manifest)
    manifest["manifest_complete"] = True
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(manifest_path, manifest)
    print(f"[OK] wrote manifest: {manifest_path}")
    done = 0
    failed = 0
    timeouted = 0
    planned = 0
    for tr in manifest["traders"]:
        for task in tr.get("tasks", []):
            st = str(task.get("status"))
            if st == "done":
                done += 1
            elif st == "failed":
                failed += 1
            elif st == "timeout":
                timeouted += 1
            elif st == "planned":
                planned += 1
    print(f"[SUMMARY] done={done} failed={failed} timeout={timeouted} planned={planned}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
