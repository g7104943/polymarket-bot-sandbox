#!/usr/bin/env python3
"""
70阈值独立版：完整版超参总控（可断点续跑）

覆盖四大块：
1) 在线增量训练分组超参（optimize_online_learning_groups.py）
2) 运行时风控超参（optimize_runtime_guards.py --profile 70）
3) 融合层多种子两阶段超参（optimize_ensemble_params_multiseed.py）
4) 交易规则超参（Exp v5 + GRU）

完成后自动汇总 min_confidence（中位数）并生成 production_best_70.env。
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "sentiment_grid_search" / "results"
GRU_OUT_DIR = PROJECT_ROOT / "experiments" / "gru_edge_kelly_optuna"
PROGRESS_PATH_DEFAULT = REPORTS_DIR / "full_hyperopt_70_progress.json"
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_15M_FILES = [
    DATA_RAW_DIR / "btc_usdt_15m.parquet",
    DATA_RAW_DIR / "eth_usdt_15m.parquet",
    DATA_RAW_DIR / "sol_usdt_15m.parquet",
    DATA_RAW_DIR / "xrp_usdt_15m.parquet",
]

V5_MODELS = [
    "v5_production_sim_noise",
    "v5_production_tv",
    "v5_production_365d",
    "v5_production_tv_365d",
    "v5_production_no_target_pm",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_range(raw: str) -> Tuple[float, float]:
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"范围参数格式错误: {raw}")
    lo = float(parts[0])
    hi = float(parts[1])
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def _parse_int_csv(raw: str) -> List[int]:
    out: List[int] = []
    for part in str(raw).split(","):
        s = part.strip()
        if not s:
            continue
        out.append(int(float(s)))
    return out


def _detect_available_days_15m() -> float:
    spans: List[float] = []
    for fp in RAW_15M_FILES:
        if not fp.exists():
            continue
        try:
            df = pd.read_parquet(fp, columns=["date", "timestamp"])
            ts = None
            if "date" in df.columns:
                ts = pd.to_datetime(df["date"], utc=True, errors="coerce").dropna()
            if (ts is None or ts.empty) and "timestamp" in df.columns:
                raw = pd.to_numeric(df["timestamp"], errors="coerce").dropna()
                if not raw.empty:
                    vmax = float(raw.max())
                    if vmax >= 1e15:
                        ts = pd.to_datetime(raw, unit="ns", utc=True, errors="coerce").dropna()
                    elif vmax >= 1e12:
                        ts = pd.to_datetime(raw, unit="ms", utc=True, errors="coerce").dropna()
                    elif vmax >= 1e9:
                        ts = pd.to_datetime(raw, unit="s", utc=True, errors="coerce").dropna()
                    else:
                        ts = pd.to_datetime(raw, utc=True, errors="coerce").dropna()
            if ts is None:
                continue
            if ts.empty:
                continue
            span = max(0.0, float((ts.max() - ts.min()).total_seconds() / 86400.0))
            if span > 1:
                spans.append(span)
        except Exception:
            continue
    if not spans:
        return 0.0
    return float(min(spans))


def _select_target_window_days(prefer_days: int, fallback_days: List[int], min_days: int, available_days: float) -> int:
    cands: List[int] = []
    for d in [int(prefer_days), *[int(x) for x in fallback_days]]:
        if d > 0 and d not in cands:
            cands.append(d)
    for d in cands:
        if available_days >= d * 0.98:
            return int(d)
    if available_days >= min_days:
        return int(min_days)
    # 极端不足时至少保留 30 天
    return int(max(30, min(int(available_days), min_days)))


def _run(cmd: List[str], cwd: Path) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd))
    if proc.returncode != 0:
        raise RuntimeError(f"命令失败(exit={proc.returncode}): {' '.join(cmd)}")


def _load_progress(path: Path, resume: bool) -> Dict[str, Any]:
    if not resume or not path.exists():
        return {"generated_at": _now_iso(), "steps": {}}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"generated_at": _now_iso(), "steps": {}}


def _save_progress(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["updated_at"] = _now_iso()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_completed(progress: Dict[str, Any], key: str) -> bool:
    st = ((progress.get("steps") or {}).get(key) or {}).get("status")
    return st == "completed"


def _mark_step(progress: Dict[str, Any], key: str, *, status: str, cmd: List[str] | None = None, error: str | None = None) -> None:
    steps = progress.setdefault("steps", {})
    item = steps.setdefault(key, {})
    item["status"] = status
    if status in ("running", "completed"):
        item.pop("error", None)
    if status == "running":
        item["started_at"] = _now_iso()
    if status in ("completed", "failed"):
        item["finished_at"] = _now_iso()
    if cmd is not None:
        item["cmd"] = cmd
    if error:
        item["error"] = error


def _extract_min_conf_from_v5() -> List[float]:
    vals: List[float] = []
    for model in V5_MODELS:
        path = OUTPUT_DIR / model / "optimal_trading_rules_v3_bp0500.json"
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            v = float(payload.get("trading_rules", {}).get("min_confidence"))
            vals.append(v)
        except Exception:
            continue
    return vals


def _extract_min_conf_from_gru() -> List[float]:
    vals: List[float] = []
    for path in sorted(GRU_OUT_DIR.glob("optimal_*_bp0460.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            v = float(payload.get("best_params", {}).get("min_confidence"))
            vals.append(v)
        except Exception:
            continue
    return vals


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="70阈值完整版超参总控（可断点续跑）")
    ap.add_argument("--python", default="/Users/mac/miniforge3/bin/python", help="用于运行子脚本的 Python")
    ap.add_argument("--resume", action="store_true", help="启用断点续跑")
    ap.add_argument("--progress-file", default=str(PROGRESS_PATH_DEFAULT), help="进度文件路径")

    ap.add_argument("--min-confidence-range", default="0.60,0.70", help="统一 min_confidence 搜索范围")
    ap.add_argument("--min-edge-range", default="0.00,0.08", help="统一 min_edge 搜索范围")
    ap.add_argument("--tier1-offset-range", default="0.01,0.05", help="tier1 偏移范围")
    ap.add_argument("--tier2-offset-range", default="0.04,0.14", help="tier2 偏移范围")

    ap.add_argument("--online-max-trials", type=int, default=240)
    ap.add_argument("--online-hours-grid", default="", help="留空则按窗口策略自动生成（推荐）")
    ap.add_argument("--runtime-window-days", type=int, default=0, help="0=按窗口策略自动")
    ap.add_argument("--runtime-groups", default="", help="留空=profile70里全部group")
    ap.add_argument("--prefer-window-days", type=int, default=365, help="优先窗口天数（默认1年）")
    ap.add_argument("--fallback-window-days", default="180,120", help="回退窗口天数列表（默认180,120）")
    ap.add_argument("--min-window-days", type=int, default=120, help="最低目标窗口天数（默认120）")

    ap.add_argument("--ensemble-seeds", default="42,2026,3407")
    ap.add_argument("--ensemble-stage1-trials", type=int, default=240)
    ap.add_argument("--ensemble-stage2-trials", type=int, default=240)
    ap.add_argument("--ensemble-n-jobs", type=int, default=6)
    ap.add_argument(
        "--ensemble-objective",
        choices=("balanced", "sharpe", "return", "win_rate"),
        default="balanced",
        help="融合层优化目标（默认 balanced）",
    )

    ap.add_argument("--v5-trials", type=int, default=2000)
    ap.add_argument("--v5-bootstrap", type=int, default=40)
    ap.add_argument("--gru-trials", type=int, default=1200)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    min_conf_range = _parse_range(args.min_confidence_range)
    min_edge_range = _parse_range(args.min_edge_range)
    tier1_offset_range = _parse_range(args.tier1_offset_range)
    tier2_offset_range = _parse_range(args.tier2_offset_range)
    min_conf_median_seed = round((min_conf_range[0] + min_conf_range[1]) / 2.0, 4)

    available_days = _detect_available_days_15m()
    fallback_days = _parse_int_csv(args.fallback_window_days)
    target_days = _select_target_window_days(
        prefer_days=int(args.prefer_window_days),
        fallback_days=fallback_days,
        min_days=int(args.min_window_days),
        available_days=available_days,
    )

    # 多窗口稳健：优先 [120,180,target] 的子集（<=target），若用户手动指定则按用户值
    if str(args.online_hours_grid).strip():
        online_hours_grid = str(args.online_hours_grid).strip()
        online_days_grid = sorted({max(1, int(float(x) / 24.0)) for x in _parse_int_csv(online_hours_grid)})
    else:
        base_days = [120, 180, int(target_days)]
        online_days_grid = sorted({int(d) for d in base_days if int(d) >= 1 and int(d) <= int(target_days)})
        if not online_days_grid:
            online_days_grid = [max(30, int(target_days))]
        online_hours_grid = ",".join(str(int(d) * 24) for d in online_days_grid)

    # 分组窗口：短组偏近期，长组偏长期，GRU取全网格
    short_days = [d for d in online_days_grid if d <= 180] or list(online_days_grid)
    long_days = [d for d in online_days_grid if d >= 180] or list(online_days_grid)
    gru_days = list(online_days_grid)
    group_hours = (
        f"v5_short={','.join(str(int(d)*24) for d in short_days)};"
        f"v5_long={','.join(str(int(d)*24) for d in long_days)};"
        f"gru_core={','.join(str(int(d)*24) for d in gru_days)}"
    )

    runtime_window_days = int(args.runtime_window_days) if int(args.runtime_window_days) > 0 else int(target_days)
    gru_data_years = round(max(30, int(target_days)) / 365.0, 4)

    print(
        f"[窗口策略] 可用15m数据约 {available_days:.1f} 天 | 目标窗口={target_days} 天 "
        f"(prefer={args.prefer_window_days}, fallback={fallback_days})"
    )
    print(f"[窗口策略] 在线hours_grid={online_hours_grid} | group_hours={group_hours}")
    print(f"[窗口策略] runtime_window_days={runtime_window_days} | v5_test_days={target_days} | gru_data_years={gru_data_years}")

    progress_path = Path(args.progress_file).expanduser()
    if not progress_path.is_absolute():
        progress_path = (PROJECT_ROOT / progress_path).resolve()
    progress = _load_progress(progress_path, resume=bool(args.resume))

    try:
        # 1) 在线增量分组超参
        step = "online_learning_groups"
        if not _is_completed(progress, step):
            cmd = [
                args.python,
                str(PROJECT_ROOT / "scripts" / "optimize_online_learning_groups.py"),
                "--groups", "v5_short,v5_long,gru_core",
                "--hours-grid", online_hours_grid,
                "--group-hours", group_hours,
                "--max-trials-per-group", str(args.online_max_trials),
                "--seed", "20260228",
                "--output-dir", str(REPORTS_DIR),
            ]
            _mark_step(progress, step, status="running", cmd=cmd)
            _save_progress(progress_path, progress)
            _run(cmd, PROJECT_ROOT)
            _mark_step(progress, step, status="completed")
            _save_progress(progress_path, progress)

        # 2) 运行时风控超参（profile 70）
        step = "runtime_guards_70"
        if not _is_completed(progress, step):
            cmd = [
                args.python,
                str(PROJECT_ROOT / "scripts" / "optimize_runtime_guards.py"),
                "--profile", "70",
                "--window-days", str(runtime_window_days),
                "--delta-grid", "0.02,0.04,0.06,0.08,0.10",
                "--lookback-grid", "6,8,12,16",
                "--min-trades-grid", "20,30,40,50",
                "--min-winrate-grid", "0.35,0.40,0.45",
                "--min-pnl-grid=-0.30,-0.20,-0.10",
                "--hold-bars-grid", "6,8,12",
                "--top-k", "8",
            ]
            if str(args.runtime_groups).strip():
                cmd.extend(["--groups", str(args.runtime_groups).strip()])
            _mark_step(progress, step, status="running", cmd=cmd)
            _save_progress(progress_path, progress)
            _run(cmd, PROJECT_ROOT)
            _mark_step(progress, step, status="completed")
            _save_progress(progress_path, progress)

        # 3) 融合多种子两阶段超参（min_conf 中位值覆盖）
        step = "ensemble_multiseed"
        if not _is_completed(progress, step):
            cmd = [
                args.python,
                str(PROJECT_ROOT / "scripts" / "optimize_ensemble_params_multiseed.py"),
                "--python", args.python,
                "--seeds", args.ensemble_seeds,
                "--stage1-trials", str(args.ensemble_stage1_trials),
                "--stage2-trials", str(args.ensemble_stage2_trials),
                "--n-jobs", str(args.ensemble_n_jobs),
                "--objective", str(args.ensemble_objective),
                "--buy-price", "0.50",
                "--capital", "10000",
                "--eval-split", "0.4",
                "--min-confidence", str(min_conf_median_seed),
                "--promote-best",
            ]
            if args.resume:
                cmd.append("--resume")
            _mark_step(progress, step, status="running", cmd=cmd)
            _save_progress(progress_path, progress)
            _run(cmd, PROJECT_ROOT)
            _mark_step(progress, step, status="completed")
            _save_progress(progress_path, progress)

        # 4) v5 交易规则超参（逐模型）
        for model in V5_MODELS:
            step = f"v5_rules::{model}"
            if _is_completed(progress, step):
                continue
            cmd = [
                args.python,
                str(PROJECT_ROOT / "scripts" / "optimize_trading_rules.py"),
                "--model-dir", model,
                "--buy-price", "0.50",
                "--trials", str(args.v5_trials),
                "--bootstrap", str(args.v5_bootstrap),
                "--min-trades", "100",
                "--test-days", str(target_days),
                "--min-confidence-range", args.min_confidence_range,
                "--min-edge-range", args.min_edge_range,
                "--tier1-offset-range", args.tier1_offset_range,
                "--tier2-offset-range", args.tier2_offset_range,
            ]
            if model == "v5_production_sim_noise":
                cmd.append("--sim-noise")
            _mark_step(progress, step, status="running", cmd=cmd)
            _save_progress(progress_path, progress)
            _run(cmd, PROJECT_ROOT)
            _mark_step(progress, step, status="completed")
            _save_progress(progress_path, progress)

        # 5) GRU 交易规则超参
        step = "gru_rules"
        if not _is_completed(progress, step):
            cmd = [
                args.python,
                str(PROJECT_ROOT / "scripts" / "optuna_gru_edge_kelly.py"),
                "--trials", str(args.gru_trials),
                "--data-years", str(gru_data_years),
                "--min-confidence-range", args.min_confidence_range,
                "--min-edge-range", "0.01,0.10",
                "--tier1-offset-range", args.tier1_offset_range,
                "--tier2-offset-range", args.tier2_offset_range,
            ]
            _mark_step(progress, step, status="running", cmd=cmd)
            _save_progress(progress_path, progress)
            _run(cmd, PROJECT_ROOT)
            _mark_step(progress, step, status="completed")
            _save_progress(progress_path, progress)

        # 6) 汇总 min_conf 并生成 production_best_70.env
        step = "build_env_70"
        if not _is_completed(progress, step):
            vals = _extract_min_conf_from_v5() + _extract_min_conf_from_gru()
            if vals:
                selected_min_conf = round(float(statistics.median(vals)), 4)
            else:
                selected_min_conf = min_conf_median_seed

            summary = {
                "generated_at": _now_iso(),
                "source_count": len(vals),
                "min_confidence_values": vals,
                "selected_min_confidence_70": selected_min_conf,
                "configured_range": [min_conf_range[0], min_conf_range[1]],
            }
            summary_path = REPORTS_DIR / "min_confidence_70_best.json"
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

            cmd = [
                str(PROJECT_ROOT / "scripts" / "build_production_best_env_70.sh"),
                str(selected_min_conf),
            ]
            _mark_step(progress, step, status="running", cmd=cmd)
            _save_progress(progress_path, progress)
            _run(cmd, PROJECT_ROOT)
            _mark_step(progress, step, status="completed")
            _save_progress(progress_path, progress)

    except Exception as e:
        key = "unknown"
        # 尝试标记最后一个 running 的步骤失败
        for k, v in (progress.get("steps") or {}).items():
            if isinstance(v, dict) and v.get("status") == "running":
                key = k
        _mark_step(progress, key, status="failed", error=str(e))
        _save_progress(progress_path, progress)
        print(f"\n❌ 失败: {e}", file=sys.stderr)
        return 1

    print("\n✅ 全部完成")
    print(f"进度文件: {progress_path}")
    print(f"70生产参数: {REPORTS_DIR / 'production_best_70.env'}")
    print(f"70最小置信度汇总: {REPORTS_DIR / 'min_confidence_70_best.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
