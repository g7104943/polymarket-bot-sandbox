#!/usr/bin/env python3
"""
融合层多随机种子稳健超参：
- 调用 optimize_ensemble_params_full.py 多次（不同 seed）
- 每次可用 two-stage 240+240
- 结果按参数中位数/多数票聚合，输出稳定版 env/json
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
FULL_SCRIPT = PROJECT_ROOT / "scripts" / "optimize_ensemble_params_full.py"
PROGRESS_JSON = REPORTS_DIR / "ensemble_params_full_multiseed_progress.json"


def _parse_seeds(raw: str) -> List[int]:
    vals: List[int] = []
    for part in str(raw).split(","):
        s = part.strip()
        if not s:
            continue
        vals.append(int(s))
    if not vals:
        raise ValueError("seeds 不能为空")
    return vals


def _is_bool(v: Any) -> bool:
    return isinstance(v, bool)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _majority_bool(values: List[bool]) -> bool:
    true_cnt = sum(1 for v in values if v)
    return true_cnt >= (len(values) - true_cnt)


def _aggregate_params(params_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    keys = sorted(set().union(*[set(p.keys()) for p in params_list]))
    for k in keys:
        vals = [p[k] for p in params_list if k in p]
        if not vals:
            continue
        if all(_is_bool(v) for v in vals):
            out[k] = _majority_bool([bool(v) for v in vals])
            continue
        if all(_is_number(v) for v in vals):
            out[k] = float(statistics.median([float(v) for v in vals]))
            continue
        # 非数值参数保留首个，避免丢字段
        out[k] = vals[0]

    # 约束修正
    if "consensus_threshold_up" in out and "consensus_threshold_down" in out:
        out["consensus_threshold_down"] = max(
            float(out["consensus_threshold_down"]),
            float(out["consensus_threshold_up"]),
        )
    if "min_edge_up" in out and "min_edge_down" in out:
        out["min_edge_down"] = max(
            float(out["min_edge_down"]),
            float(out["min_edge_up"]),
        )
    return out


def _render_env(best_params: Dict[str, Any]) -> str:
    lines = [
        f"CONSENSUS_THRESHOLD_UP={best_params['consensus_threshold_up']}",
        f"CONSENSUS_THRESHOLD_DOWN={best_params['consensus_threshold_down']}",
        f"MIN_EDGE_UP={best_params['min_edge_up']}",
        f"MIN_EDGE_DOWN={best_params['min_edge_down']}",
        f"CONFIDENCE_SCALE={best_params['confidence_scale']}",
        f"CONSENSUS_BY_WEIGHT={'true' if best_params['consensus_by_weight'] else 'false'}",
    ]
    for key in (
        "decay_halflife_hours",
        "recent_window_hours",
        "wr_blend_recent",
        "weight_softplus_temp",
        "weight_softplus_scale",
        "weight_min",
        "weight_small_sample",
    ):
        if key in best_params:
            lines.append(f"{key.upper()}={best_params[key]}")
    lines.append("")
    return "\n".join(lines)


def _latest_report_after(start_ts: float) -> Path:
    cands = sorted(
        REPORTS_DIR.glob("ensemble_params_full_tuning_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    newer = [p for p in cands if p.stat().st_mtime >= start_ts]
    if newer:
        return newer[-1]
    if cands:
        return cands[-1]
    raise RuntimeError("未找到 ensemble_params_full_tuning_*.json")


def _report_has_target_trials(report_path: Path, stage1_trials: int, stage2_trials: int) -> bool:
    if not report_path.exists():
        return False
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    s1 = int(payload.get("finished_trials_stage1") or payload.get("stage1_trials") or 0)
    s2 = int(payload.get("finished_trials_stage2") or payload.get("stage2_trials") or 0)
    best_params = payload.get("best_params")
    return (
        isinstance(best_params, dict)
        and len(best_params) > 0
        and s1 >= int(stage1_trials)
        and s2 >= int(stage2_trials)
    )


def _config_hash(args: argparse.Namespace, seeds: List[int]) -> str:
    payload = {
        "seeds": list(seeds),
        "stage1_trials": int(args.stage1_trials),
        "stage2_trials": int(args.stage2_trials),
        "objective": str(args.objective),
        "buy_price": float(args.buy_price),
        "capital": float(args.capital),
        "min_confidence": (None if args.min_confidence is None else float(args.min_confidence)),
        "eval_split": 0.0 if bool(args.no_eval_split) else float(args.eval_split),
        "no_eval_split": bool(args.no_eval_split),
        "n_jobs": int(args.n_jobs),
        "storage": str(args.storage),
        "study_prefix": str(args.study_prefix),
    }
    return hashlib.sha1(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _load_progress(config_hash: str) -> Dict[int, Dict[str, Any]]:
    if not PROGRESS_JSON.exists():
        return {}
    try:
        payload = json.loads(PROGRESS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if str(payload.get("config_hash", "")) != str(config_hash):
        return {}
    runs = payload.get("runs")
    if not isinstance(runs, list):
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    for r in runs:
        if not isinstance(r, dict):
            continue
        seed = r.get("seed")
        if isinstance(seed, int):
            out[seed] = r
    return out


def _save_progress(config_hash: str, args: argparse.Namespace, seeds: List[int], runs: List[Dict[str, Any]]) -> None:
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash,
        "seeds": list(seeds),
        "stage1_trials": int(args.stage1_trials),
        "stage2_trials": int(args.stage2_trials),
        "objective": str(args.objective),
        "buy_price": float(args.buy_price),
        "capital": float(args.capital),
        "min_confidence": (None if args.min_confidence is None else float(args.min_confidence)),
        "eval_split": 0.0 if args.no_eval_split else float(args.eval_split),
        "storage": str(args.storage),
        "study_prefix": str(args.study_prefix),
        "runs": runs,
    }
    PROGRESS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run_one(seed: int, args: argparse.Namespace) -> Dict[str, Any]:
    cmd = [
        args.python,
        str(FULL_SCRIPT),
        "--two-stage",
        "--stage1-trials",
        str(args.stage1_trials),
        "--stage2-trials",
        str(args.stage2_trials),
        "--objective",
        args.objective,
        "--buy-price",
        str(args.buy_price),
        "--capital",
        str(args.capital),
        "--seed",
        str(seed),
        "--n-jobs",
        str(args.n_jobs),
        "--storage",
        str(args.storage),
        "--study-prefix",
        str(args.study_prefix),
    ]
    if args.min_confidence is not None:
        cmd.extend(["--min-confidence", str(args.min_confidence)])
    if args.resume:
        cmd.append("--resume")
    if args.no_eval_split:
        cmd.append("--no-eval-split")
    else:
        cmd.extend(["--eval-split", str(args.eval_split)])

    print(f"\n=== Seed {seed} 开始 ===")
    print(" ".join(cmd))
    started = datetime.now(timezone.utc).isoformat()
    start_ts = datetime.now().timestamp()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"seed={seed} 运行失败，退出码={proc.returncode}")
    report_path = _latest_report_after(start_ts)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    best_params = payload.get("best_params", {})
    if not isinstance(best_params, dict) or not best_params:
        raise RuntimeError(f"seed={seed} 报告缺少 best_params: {report_path}")
    print(f"=== Seed {seed} 完成: {report_path.name} ===")
    return {
        "seed": seed,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "report": str(report_path.relative_to(PROJECT_ROOT)),
        "best_value": payload.get("best_value"),
        "best_params": best_params,
        "eval_metrics": payload.get("eval_metrics"),
        "window_days": payload.get("window_days"),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="融合层多随机种子稳健超参")
    ap.add_argument("--python", default=sys.executable, help="用于调用单次超参脚本的 Python")
    ap.add_argument("--seeds", default="42,202,777", help="随机种子列表，逗号分隔")
    ap.add_argument("--stage1-trials", type=int, default=240)
    ap.add_argument("--stage2-trials", type=int, default=240)
    ap.add_argument("--n-jobs", type=int, default=6, help="单次 full 优化并行 worker 数")
    ap.add_argument(
        "--storage",
        default=str(PROJECT_ROOT / "reports" / "optuna" / "ensemble_full_trials.db"),
        help="传给 full 脚本的 Optuna storage",
    )
    ap.add_argument("--study-prefix", default="ensemble_full", help="传给 full 脚本的 study 前缀")
    ap.add_argument("--resume", action="store_true", help="启用断点续跑（按 trial + seed 进度）")
    ap.add_argument("--objective", choices=["sharpe", "return", "win_rate", "balanced"], default="balanced")
    ap.add_argument("--buy-price", type=float, default=0.50)
    ap.add_argument("--capital", type=float, default=10000)
    ap.add_argument("--min-confidence", type=float, default=None, help="透传 full 脚本的 min_confidence 覆盖值（如 0.60~0.70）")
    ap.add_argument("--eval-split", type=float, default=0.4)
    ap.add_argument("--no-eval-split", action="store_true")
    ap.add_argument(
        "--promote-best",
        action="store_true",
        help="聚合完成后覆盖 ensemble_params_full_best.{json,env}",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    seeds = _parse_seeds(args.seeds)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    conf_hash = _config_hash(args, seeds)
    existing = _load_progress(conf_hash) if args.resume else {}
    if args.resume:
        print(f"resume=on  config={conf_hash}  progress={PROGRESS_JSON}")

    runs: List[Dict[str, Any]] = []
    for seed in seeds:
        reused = None
        if args.resume and seed in existing:
            cand = existing[seed]
            report_rel = cand.get("report")
            report_path = (PROJECT_ROOT / str(report_rel)).resolve() if report_rel else None
            if report_path and _report_has_target_trials(report_path, args.stage1_trials, args.stage2_trials):
                print(f"\n=== Seed {seed} 已完成，复用进度: {report_path.name} ===")
                reused = cand
        if reused is not None:
            runs.append(reused)
        else:
            run = _run_one(seed, args)
            runs.append(run)
            if args.resume:
                _save_progress(conf_hash, args, seeds, runs)

    agg_params = _aggregate_params([r["best_params"] for r in runs])
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "median_across_seeds",
        "seeds": seeds,
        "objective": args.objective,
        "buy_price": float(args.buy_price),
        "capital": float(args.capital),
        "min_confidence": (None if args.min_confidence is None else float(args.min_confidence)),
        "eval_split": 0.0 if args.no_eval_split else float(args.eval_split),
        "stage1_trials": int(args.stage1_trials),
        "stage2_trials": int(args.stage2_trials),
        "best_params": agg_params,
        "runs": runs,
    }

    out_json = REPORTS_DIR / "ensemble_params_full_multiseed_best.json"
    out_env = REPORTS_DIR / "ensemble_params_full_multiseed_best.env"
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out_env.write_text(_render_env(agg_params), encoding="utf-8")
    if args.resume:
        _save_progress(conf_hash, args, seeds, runs)

    print("\n=== 多 seed 聚合完成 ===")
    print(f"JSON: {out_json}")
    print(f"ENV : {out_env}")

    if args.promote_best:
        shutil.copy2(out_json, REPORTS_DIR / "ensemble_params_full_best.json")
        shutil.copy2(out_env, REPORTS_DIR / "ensemble_params_full_best.env")
        print("已覆盖 canonical: ensemble_params_full_best.{json,env}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
