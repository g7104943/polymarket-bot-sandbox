#!/usr/bin/env python3
"""
模型健康检查 — 一键查看全部模型/在线学习/Writer 对齐状态。

用法:
    python3 scripts/check_model_health.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLYMARKET_DIR = PROJECT_ROOT / "polymarket"

EXP_MODEL_DIRS: Dict[int, str] = {
    10: "v5_production_sim_noise",
    11: "v5_production_sim_noise",
    13: "v5_production_tv",
    14: "v5_production_sim_noise_tv",
    15: "v5_production_365d",
    16: "v5_production_tv_365d",
    17: "v5_production_no_target_pm",
}

GRU_ASSETS = ["BTC_USDT", "ETH_USDT", "SOL_USDT"]
GRU_MODELS_BEST = PROJECT_ROOT / "experiments" / "gru_regime_v1" / "outputs" / "models_best"
ONLINE_LOG = PROJECT_ROOT / "logs" / "online_learning_history.jsonl"

STALE_DAYS = 7
WARN = "\u26a0\ufe0f"
OK = "\u2705"
ERR = "\u274c"


def _fmt_time(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%m/%d %H:%M")


def _days_ago(ts: float) -> float:
    return (time.time() - ts) / 86400


def _load_tree_count(path: Path) -> Optional[int]:
    try:
        m = joblib.load(path)
        return m.booster_.num_trees()
    except Exception:
        return None


# ──────────────────────────────────────────────
# Section 1: Exp model files
# ──────────────────────────────────────────────
def check_exp_models() -> List[str]:
    lines = []
    lines.append("")
    lines.append("[1] Exp 模型文件状态")
    lines.append("-" * 60)
    warnings = []
    seen_dirs = set()
    for exp_id, dir_name in sorted(EXP_MODEL_DIRS.items()):
        if dir_name in seen_dirs:
            continue
        seen_dirs.add(dir_name)
        model_dir = PROJECT_ROOT / "data" / "models" / dir_name
        if not model_dir.exists():
            lines.append(f"  {ERR} {dir_name}: 目录不存在")
            warnings.append(f"{dir_name} 目录不存在")
            continue

        fc_path = model_dir / "feature_cols.json"
        fc_mtime = fc_path.stat().st_mtime if fc_path.exists() else 0

        model_files = sorted(model_dir.glob("lgb_*.joblib"))
        if not model_files:
            lines.append(f"  {ERR} {dir_name}: 无 joblib 模型文件")
            warnings.append(f"{dir_name} 无模型文件")
            continue

        mtimes = [f.stat().st_mtime for f in model_files]
        latest = max(mtimes)
        trees = []
        for f in model_files:
            tc = _load_tree_count(f)
            trees.append(str(tc) if tc else "?")

        exps_using = [str(k) for k, v in EXP_MODEL_DIRS.items() if v == dir_name]
        fc_count = 0
        mtf_count = 0
        if fc_path.exists():
            with open(fc_path) as f:
                cols = json.load(f)
            fc_count = len(cols)
            mtf_count = sum(1 for c in cols if c.startswith("mtf_"))

        stale = _days_ago(latest) > STALE_DAYS
        marker = WARN if stale else OK
        lines.append(f"  {marker} {dir_name:35s} (Exp{','.join(exps_using)})")
        lines.append(f"      {len(model_files)} 个模型, 最后修改 {_fmt_time(latest)}, "
                     f"树数: {'/'.join(trees)}, 特征: {fc_count} ({mtf_count} mtf)")
        if stale:
            warnings.append(f"{dir_name} 超过 {STALE_DAYS} 天未更新")

    return lines, warnings


# ──────────────────────────────────────────────
# Section 2: GRU model files
# ──────────────────────────────────────────────
def check_gru_models() -> List[str]:
    lines = []
    lines.append("")
    lines.append("[2] GRU 模型文件状态")
    lines.append("-" * 60)
    warnings = []

    for asset in GRU_ASSETS:
        model_path = GRU_MODELS_BEST / asset / "lightgbm_with_embedding.joblib"
        if not model_path.exists():
            lines.append(f"  {ERR} {asset}: 模型不存在")
            warnings.append(f"GRU {asset} 模型不存在")
            continue
        mtime = model_path.stat().st_mtime
        tc = _load_tree_count(model_path)

        fc_path = GRU_MODELS_BEST / asset / "feature_cols.json"
        fc_count = 0
        if fc_path.exists():
            with open(fc_path) as f:
                fc_count = len(json.load(f))

        stale = _days_ago(mtime) > STALE_DAYS
        marker = WARN if stale else OK
        lines.append(f"  {marker} {asset:12s}: 最后修改 {_fmt_time(mtime)}, 树数: {tc}, 特征: {fc_count}")
        if stale:
            warnings.append(f"GRU {asset} 超过 {STALE_DAYS} 天未更新")

    return lines, warnings


# ──────────────────────────────────────────────
# Section 3: Feature alignment
# ──────────────────────────────────────────────
def check_feature_alignment() -> List[str]:
    lines = []
    lines.append("")
    lines.append("[3] 特征对齐检查")
    lines.append("-" * 60)
    warnings = []

    writer_pids = _find_writer_pids()
    writer_start_time = None
    if writer_pids:
        try:
            result = subprocess.run(
                ["ps", "-o", "lstart=", "-p", str(writer_pids[0])],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                writer_start_time = result.stdout.strip()
        except Exception:
            pass

    lines.append(f"  Writer 进程: {'PID ' + ','.join(map(str, writer_pids)) if writer_pids else '未找到'}"
                 f"{' (启动于 ' + writer_start_time + ')' if writer_start_time else ''}")

    seen_dirs = set()
    for exp_id, dir_name in sorted(EXP_MODEL_DIRS.items()):
        if dir_name in seen_dirs:
            continue
        seen_dirs.add(dir_name)
        model_dir = PROJECT_ROOT / "data" / "models" / dir_name
        fc_path = model_dir / "feature_cols.json"
        if not fc_path.exists():
            continue
        fc_mtime = fc_path.stat().st_mtime

        model_files = sorted(model_dir.glob("lgb_*.joblib"))
        model_mtime = max(f.stat().st_mtime for f in model_files) if model_files else 0

        aligned = abs(fc_mtime - model_mtime) < 60
        marker = OK if aligned else WARN
        lines.append(f"  {marker} {dir_name}: fc={_fmt_time(fc_mtime)}, model={_fmt_time(model_mtime)}"
                     f"{'' if aligned else ' (不一致!)'}")
        if not aligned:
            warnings.append(f"{dir_name}: feature_cols 和模型文件时间不一致")

    return lines, warnings


def _find_writer_pids() -> List[int]:
    try:
        result = subprocess.run(
            ["pgrep", "-f", "prediction_writer_v5"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return [int(p) for p in result.stdout.strip().split("\n") if p.strip()]
    except Exception:
        pass
    return []


# ──────────────────────────────────────────────
# Section 4: Online learning status
# ──────────────────────────────────────────────
def check_online_learning() -> List[str]:
    lines = []
    lines.append("")
    lines.append("[4] 在线学习状态")
    lines.append("-" * 60)
    warnings = []

    if not ONLINE_LOG.exists():
        lines.append(f"  {ERR} 日志文件不存在: {ONLINE_LOG}")
        warnings.append("在线学习日志不存在")
        return lines, warnings

    last_exp = None
    last_gru = None
    with open(ONLINE_LOG) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("type") == "gru":
                last_gru = entry
            else:
                last_exp = entry

    for label, entry in [("Exp 增量", last_exp), ("GRU 增量", last_gru)]:
        if entry is None:
            lines.append(f"  {WARN} {label}: 无记录")
            warnings.append(f"{label} 无记录")
            continue
        ts = entry.get("timestamp", "?")[:19]
        summary = entry.get("summary", {})
        s = summary.get("success", 0)
        sk = summary.get("skipped", 0)
        results = entry.get("results", [])
        is_dry = any((r.get("skipped_reason") or "").startswith("dry") for r in results if r.get("success"))
        dry_tag = " (dry-run)" if is_dry else ""
        marker = WARN if is_dry else OK
        lines.append(f"  {marker} {label}: {ts}{dry_tag}, {s} 成功, {sk} 跳过")
        if is_dry:
            warnings.append(f"{label} 最后一次为 dry-run，未实际更新模型")

    return lines, warnings


# ──────────────────────────────────────────────
# Section 5: Prediction distribution
# ──────────────────────────────────────────────
def check_prediction_distribution() -> List[str]:
    lines = []
    lines.append("")
    lines.append("[5] 最新预测分布")
    lines.append("-" * 60)
    warnings = []

    pred_dir = POLYMARKET_DIR
    pred_files = sorted(pred_dir.glob("predictions_*.json"))
    if not pred_files:
        lines.append(f"  {WARN} 无预测文件")
        return lines, warnings

    for pf in pred_files:
        try:
            mtime = pf.stat().st_mtime
            age_min = (time.time() - mtime) / 60
            if age_min > 30:
                continue
            with open(pf) as f:
                data = json.load(f)
            name = pf.stem.replace("predictions_", "")
            preds = data.get("predictions", data)
            parts = []
            for key, info in preds.items():
                if isinstance(info, dict) and "confidence" in info:
                    conf = info["confidence"]
                    d = (info.get("direction") or "?")[0]
                    sym = key.split("_")[0]
                    parts.append(f"{sym}:{d}{conf*100:.1f}%")
            if parts:
                lines.append(f"  {name:25s}: {' | '.join(parts)}  ({age_min:.0f}分钟前)")
        except Exception:
            continue

    return lines, warnings


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    now_str = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print("=" * 64)
    print(f"  模型健康检查 - {now_str}")
    print("=" * 64)

    all_warnings = []

    for check_fn in [
        check_exp_models,
        check_gru_models,
        check_feature_alignment,
        check_online_learning,
        check_prediction_distribution,
    ]:
        lines, warnings = check_fn()
        for line in lines:
            print(line)
        all_warnings.extend(warnings)

    print()
    print("=" * 64)
    if all_warnings:
        print(f"  {WARN} {len(all_warnings)} 个警告:")
        for w in all_warnings:
            print(f"    - {w}")
    else:
        print(f"  {OK} 全部正常")
    print("=" * 64)


if __name__ == "__main__":
    main()
