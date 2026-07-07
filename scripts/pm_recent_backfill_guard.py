#!/usr/bin/env python3
"""
PM 概率/目标概率近窗断档检测与自动回补（默认 72 小时）。

说明：
- `polymarket_prob_*` 与 `polymarket_prob_target_*` 支持自动回补（调用现有 pull 脚本）。
- `polymarket_ob_snapshots` 仅做断档检测与告警（无可靠历史 API，无法做真实历史回补）。
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SENT_DIR = PROJECT_ROOT / "data" / "sentiment"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORT_FILE = LOGS_DIR / "pm_recent_backfill_report.json"
OB_RESTART_STAMP_FILE = LOGS_DIR / "pm_ob_daemon_restart.last_ts"
STEP_SEC = 900  # 15m


@dataclass
class DatasetSpec:
    key: str
    path: Path
    columns: tuple[str, ...]
    backfillable: bool
    asset: str | None = None
    kind: str | None = None  # prob | target | ob


def _to_unix_seconds(values: pd.Series) -> np.ndarray:
    if values.empty:
        return np.array([], dtype=np.int64)
    s = values.dropna()
    if s.empty:
        return np.array([], dtype=np.int64)
    if np.issubdtype(s.dtype, np.datetime64):
        ts = (s.astype("int64") // 10**9).astype(np.int64)
    else:
        arr = pd.to_numeric(s, errors="coerce").dropna().astype(np.int64)
        if arr.empty:
            return np.array([], dtype=np.int64)
        # ms -> s
        if arr.iloc[0] > 10**12:
            arr = arr // 1000
        ts = arr.to_numpy(dtype=np.int64)
    return np.sort(np.unique(ts))


def _load_timestamps(path: Path, columns: Iterable[str]) -> np.ndarray:
    if not path.exists():
        return np.array([], dtype=np.int64)
    for col in columns:
        try:
            df = pd.read_parquet(path, columns=[col])
        except Exception:
            continue
        if col not in df.columns:
            continue
        ts = _to_unix_seconds(df[col])
        if ts.size > 0:
            return ts
    return np.array([], dtype=np.int64)


def _analyze_recent(ts: np.ndarray, now_sec: int, lookback_hours: int, step_sec: int) -> dict[str, object]:
    window_start = now_sec - lookback_hours * 3600
    window_start = (window_start // step_sec) * step_sec
    recent = ts[ts >= window_start]
    latest = int(recent[-1]) if recent.size else None
    latest_age_sec = None if latest is None else max(0, now_sec - latest)

    expected = np.arange(window_start, now_sec + 1, step_sec, dtype=np.int64)
    expected_set = set(int(x) for x in expected.tolist())
    got_set = set(int(x) for x in recent.tolist())
    missing = sorted(expected_set - got_set)
    missing_count = len(missing)

    max_gap_bars = 0
    if recent.size >= 2:
        diffs = np.diff(recent)
        max_gap = int(diffs.max()) if diffs.size else step_sec
        max_gap_bars = max(0, (max_gap // step_sec) - 1)
    elif recent.size == 1:
        max_gap_bars = max(0, (now_sec - int(recent[0])) // step_sec - 1)
    else:
        max_gap_bars = len(expected)

    return {
        "window_start": window_start,
        "window_end": now_sec,
        "expected_points": int(len(expected)),
        "actual_points": int(len(recent)),
        "latest_ts": latest,
        "latest_age_sec": latest_age_sec,
        "missing_count": missing_count,
        "max_gap_bars": int(max_gap_bars),
        "missing_sample": missing[:10],
    }


def _run_backfill(train_python: str, script: str, asset: str, lookback_hours: int, workers: int) -> tuple[bool, str]:
    cmd = [
        train_python,
        str(PROJECT_ROOT / "scripts" / script),
        "--asset",
        asset,
        "--recent-hours",
        str(int(lookback_hours)),
        "--workers",
        str(int(max(1, workers))),
    ]
    try:
        ret = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=1800,
            check=False,
        )
        ok = ret.returncode == 0
        tail = "\n".join((ret.stdout or "").splitlines()[-8:])
        return ok, tail
    except Exception as e:  # pragma: no cover - defensive
        return False, str(e)


def _detect_daemon_pid() -> int | None:
    pid_file = PROJECT_ROOT / "polymarket" / "market_data_daemon.pid"
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip() or "0")
        except Exception:
            pid = 0
        if pid > 0:
            try:
                p = subprocess.run(["kill", "-0", str(pid)], capture_output=True, check=False)
                if p.returncode == 0:
                    return pid
            except Exception:
                pass
    try:
        p = subprocess.run(
            ["ps", "-Ao", "pid,command", "-ww"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if p.returncode == 0:
            for line in (p.stdout or "").splitlines():
                l = line.strip()
                if not l:
                    continue
                if ("node dist/market_data_daemon.js" in l) or ("ts-node src/market_data_daemon.ts" in l):
                    try:
                        return int(l.split(None, 1)[0])
                    except Exception:
                        continue
    except Exception:
        pass
    return None


def _run_ob_daemon_restart(cooldown_minutes: int) -> tuple[bool, str]:
    now = int(time.time())
    cooldown_sec = max(300, int(cooldown_minutes) * 60)
    try:
        if OB_RESTART_STAMP_FILE.exists():
            last = int(OB_RESTART_STAMP_FILE.read_text(encoding="utf-8").strip() or "0")
            if last > 0 and now - last < cooldown_sec:
                wait = cooldown_sec - (now - last)
                return True, f"跳过重启(冷却中): {wait}s 后可再次重启"
    except Exception:
        pass

    # Daemon 存活时不做硬重启，避免 VPN 抖动阶段频繁 kill 造成“启动后又退出”。
    # 仅当 Daemon 真实未运行时才拉起。
    live_pid = _detect_daemon_pid()
    if live_pid is not None:
        return True, f"跳过重启: Daemon 运行中 (PID={live_pid})"

    cmd = (
        "rm -f polymarket/market_data_daemon.pid; "
        "./启动市场数据服务.sh"
    )
    try:
        ret = subprocess.run(
            ["/bin/bash", "-lc", cmd],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=120,
            check=False,
        )
        tail = "\n".join((ret.stdout or "").splitlines()[-12:])
        if ret.returncode == 0:
            OB_RESTART_STAMP_FILE.parent.mkdir(parents=True, exist_ok=True)
            OB_RESTART_STAMP_FILE.write_text(str(now), encoding="utf-8")
            return True, tail
        return False, tail
    except Exception as e:  # pragma: no cover
        return False, str(e)


def _build_dataset_summary(
    spec: DatasetSpec,
    *,
    now: int,
    lookback_hours: int,
    max_missing_bars: int,
    stale_minutes_threshold: int,
) -> dict[str, object]:
    ts = _load_timestamps(spec.path, spec.columns)
    summary = _analyze_recent(ts, now_sec=now, lookback_hours=lookback_hours, step_sec=STEP_SEC)
    latest_age = summary.get("latest_age_sec")
    missing_count = int(summary.get("missing_count", 0) or 0)
    max_gap_bars = int(summary.get("max_gap_bars", 0) or 0)
    stale_limit_sec = stale_minutes_threshold * 60
    too_stale = latest_age is None or int(latest_age) > stale_limit_sec
    has_gap = missing_count > max_missing_bars or max_gap_bars > max_missing_bars
    if spec.backfillable:
        needs_fix = bool(too_stale or has_gap)
    else:
        needs_fix = bool(too_stale)
    summary["needs_fix"] = needs_fix
    summary["backfillable"] = spec.backfillable
    summary["gap_detect_only"] = bool((not spec.backfillable) and has_gap and (not too_stale))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="PM 近窗断档检测与自动回补")
    ap.add_argument("--lookback-hours", type=int, default=72, help="回看窗口小时数，默认 72")
    ap.add_argument("--max-missing-bars", type=int, default=4, help="允许缺失 bar 数阈值")
    ap.add_argument("--stale-minutes-threshold", type=int, default=35, help="最新点陈旧阈值（分钟）")
    ap.add_argument("--workers", type=int, default=3, help="回补并发 worker")
    ap.add_argument("--python-bin", type=str, default=sys.executable, help="用于回补脚本的 Python")
    ap.add_argument(
        "--ob-daemon-restart-cooldown-minutes",
        type=int,
        default=20,
        help="订单簿快照陈旧时，自动重启 Daemon 的最小冷却分钟数（默认 20）",
    )
    ap.add_argument("--fix", action="store_true", help="发现断档时自动回补")
    args = ap.parse_args()

    now = int(time.time())
    specs = [
        DatasetSpec(
            key="polymarket_prob_btc",
            path=SENT_DIR / "polymarket_prob_btc_usdt.parquet",
            columns=("timestamp_s", "timestamp"),
            backfillable=True,
            asset="BTC_USDT",
            kind="prob",
        ),
        DatasetSpec(
            key="polymarket_prob_eth",
            path=SENT_DIR / "polymarket_prob_eth_usdt.parquet",
            columns=("timestamp_s", "timestamp"),
            backfillable=True,
            asset="ETH_USDT",
            kind="prob",
        ),
        DatasetSpec(
            key="polymarket_prob_xrp",
            path=SENT_DIR / "polymarket_prob_xrp_usdt.parquet",
            columns=("timestamp_s", "timestamp"),
            backfillable=True,
            asset="XRP_USDT",
            kind="prob",
        ),
        DatasetSpec(
            key="polymarket_prob_target_btc",
            path=SENT_DIR / "polymarket_prob_target_btc_usdt.parquet",
            columns=("timestamp_s", "timestamp"),
            backfillable=True,
            asset="BTC_USDT",
            kind="target",
        ),
        DatasetSpec(
            key="polymarket_prob_target_eth",
            path=SENT_DIR / "polymarket_prob_target_eth_usdt.parquet",
            columns=("timestamp_s", "timestamp"),
            backfillable=True,
            asset="ETH_USDT",
            kind="target",
        ),
        DatasetSpec(
            key="polymarket_prob_target_xrp",
            path=SENT_DIR / "polymarket_prob_target_xrp_usdt.parquet",
            columns=("timestamp_s", "timestamp"),
            backfillable=True,
            asset="XRP_USDT",
            kind="target",
        ),
        DatasetSpec(
            key="polymarket_ob_snapshots",
            path=SENT_DIR / "polymarket_ob_snapshots.parquet",
            columns=("bar_ts", "timestamp_s", "timestamp"),
            backfillable=False,
            asset=None,
            kind="ob",
        ),
    ]

    report: dict[str, object] = {
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "lookback_hours": int(args.lookback_hours),
        "max_missing_bars": int(args.max_missing_bars),
        "stale_minutes_threshold": int(args.stale_minutes_threshold),
        "fix_enabled": bool(args.fix),
        "datasets": {},
        "actions": [],
    }

    need_prob: set[str] = set()
    need_target: set[str] = set()

    stale_limit_sec = int(args.stale_minutes_threshold) * 60
    for spec in specs:
        summary = _build_dataset_summary(
            spec,
            now=now,
            lookback_hours=int(args.lookback_hours),
            max_missing_bars=int(args.max_missing_bars),
            stale_minutes_threshold=int(args.stale_minutes_threshold),
        )
        needs_fix = bool(summary.get("needs_fix"))
        report["datasets"][spec.key] = summary
        if needs_fix and spec.backfillable and spec.asset and spec.kind == "prob":
            need_prob.add(spec.asset)
        if needs_fix and spec.backfillable and spec.asset and spec.kind == "target":
            need_target.add(spec.asset)

    if args.fix:
        for asset in sorted(need_prob):
            ok, out = _run_backfill(
                train_python=args.python_bin,
                script="pull_polymarket_prob.py",
                asset=asset,
                lookback_hours=int(args.lookback_hours),
                workers=int(args.workers),
            )
            report["actions"].append({"type": "backfill_prob", "asset": asset, "ok": ok, "tail": out})
        for asset in sorted(need_target):
            ok, out = _run_backfill(
                train_python=args.python_bin,
                script="pull_polymarket_prob_target.py",
                asset=asset,
                lookback_hours=int(args.lookback_hours),
                workers=int(args.workers),
            )
            report["actions"].append({"type": "backfill_target", "asset": asset, "ok": ok, "tail": out})
        ob_summary = report["datasets"].get("polymarket_ob_snapshots")
        ob_needs_fix = bool(isinstance(ob_summary, dict) and ob_summary.get("needs_fix"))
        if ob_needs_fix:
            ok, out = _run_ob_daemon_restart(int(args.ob_daemon_restart_cooldown_minutes))
            report["actions"].append({"type": "restart_ob_daemon", "asset": "N/A", "ok": ok, "tail": out})

        # fix 执行后重新刷新 datasets，避免报告仍停留在修前快照
        refreshed_now = int(time.time())
        for spec in specs:
            report["datasets"][spec.key] = _build_dataset_summary(
                spec,
                now=refreshed_now,
                lookback_hours=int(args.lookback_hours),
                max_missing_bars=int(args.max_missing_bars),
                stale_minutes_threshold=int(args.stale_minutes_threshold),
            )

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 返回码：0 正常，2 有需要修复但未开启 fix，3 有回补失败
    any_needs_fix = any(bool(v.get("needs_fix")) for v in report["datasets"].values())  # type: ignore[union-attr]
    if any_needs_fix and not args.fix:
        print("⚠️ 检测到 PM 断档/陈旧，但未开启 --fix")
        return 2
    if args.fix and any(a.get("ok") is False for a in report["actions"]):  # type: ignore[union-attr]
        print("❌ PM 自动回补部分失败，请查看 logs/pm_recent_backfill_report.json")
        return 3
    print("✅ PM 近窗断档检查完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
