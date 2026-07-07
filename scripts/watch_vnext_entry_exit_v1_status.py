#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
REPORTS_DIR = ROOT / "reports"
PM_DIR = ROOT / "polymarket"

REALIGN_LOG_GLOB = "vnext_btceth_entry_exit_realign_*.log"
STATUS_REPORT = REPORTS_DIR / "vnext_entry_exit_v1_realign_status_latest.json"
ALIGNMENT_AUDIT = REPORTS_DIR / "vnext_entry_exit_v1_alignment_audit_latest.json"

BTC_OVERLAY = PM_DIR / "logs_vnext_btc_entry_exit_v1" / "prediction_trades.simulation.json"
ETH_OVERLAY = PM_DIR / "logs_vnext_eth_entry_exit_v1" / "prediction_trades.simulation.json"

PROC_PATTERNS = {
    "pull": "pull_polymarket_1m_path_vnext.py",
    "train": "train_vnext_btceth_entry_exit_v1.py",
    "writer": "prediction_writer_vnext_btceth_entry_exit_v1.py",
    "multi": "multi_prediction_index.js --group vnext_btceth_entry_exit_compare",
    "overlay": "reconcile_vnext_btceth_entry_exit_overlay.py",
}


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _count_trades(path: Path) -> Dict[str, Any]:
    rows = _load_json(path)
    if not isinstance(rows, list):
        return {"rows": 0, "wins": 0, "losses": 0, "pending": 0, "pnl": 0.0}
    wins = 0
    losses = 0
    pending = 0
    pnl = 0.0
    for row in rows:
        if not isinstance(row, dict):
            continue
        result = str(row.get("result") or "").lower()
        if result == "win":
            wins += 1
        elif result == "lose":
            losses += 1
        else:
            pending += 1
        try:
            pnl += float(row.get("pnl") or 0.0)
        except Exception:
            pass
    return {
        "rows": len(rows),
        "wins": wins,
        "losses": losses,
        "pending": pending,
        "pnl": round(pnl, 4),
    }


def _latest_realign_log() -> Path | None:
    logs = sorted(LOGS_DIR.glob(REALIGN_LOG_GLOB))
    return logs[-1] if logs else None


def _tail_lines(path: Path | None, n: int = 20) -> List[str]:
    if path is None or not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="ignore").splitlines()[-n:]
    except Exception:
        return []


def _ps_rows() -> List[str]:
    try:
        out = subprocess.run(
            ["ps", "ax", "-o", "pid=,etime=,pcpu=,command="],
            check=False,
            capture_output=True,
            text=True,
        )
        return [line.strip() for line in out.stdout.splitlines() if line.strip()]
    except Exception:
        return []


def _matching_processes() -> Dict[str, List[Dict[str, str]]]:
    rows = _ps_rows()
    out: Dict[str, List[Dict[str, str]]] = {key: [] for key in PROC_PATTERNS}
    for line in rows:
        for key, pattern in PROC_PATTERNS.items():
            if pattern not in line:
                continue
            parts = line.split(None, 3)
            if len(parts) < 4:
                continue
            pid, etime, pcpu, command = parts
            out[key].append(
                {
                    "pid": pid,
                    "etime": etime,
                    "pcpu": pcpu,
                    "command": command,
                }
            )
    return out


def _infer_stage(processes: Dict[str, List[Dict[str, str]]], tail: List[str]) -> str:
    if processes.get("overlay") or processes.get("multi") or processes.get("writer"):
        return "runtime"
    if processes.get("train"):
        return "train"
    if processes.get("pull"):
        return "pull"
    tail_text = "\n".join(tail)
    if '"status": "trained"' in tail_text or "start writer" in tail_text:
        return "post_train"
    if "[1m-path]" in tail_text:
        return "pull"
    return "idle_or_finished"


def _run_alignment_audit(sample_size: int) -> Dict[str, Any]:
    cmd = [
        "python",
        str(ROOT / "scripts" / "audit_vnext_entry_exit_v1_alignment.py"),
        "--sample-size",
        str(sample_size),
    ]
    out = subprocess.run(cmd, check=False, capture_output=True, text=True)
    return {
        "returncode": out.returncode,
        "stdout": out.stdout.strip()[-2000:],
        "stderr": out.stderr.strip()[-2000:],
    }


def _write_waiting_alignment_audit(reason: str, sample_size: int) -> None:
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scope": "vnext_entry_exit_v1_alignment_audit",
        "sample_size": int(sample_size),
        "waiting": True,
        "reason": str(reason),
        "assets": {},
    }
    ALIGNMENT_AUDIT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_status(run_audit: bool, sample_size: int) -> Dict[str, Any]:
    log_path = _latest_realign_log()
    tail = _tail_lines(log_path)
    processes = _matching_processes()
    btc = _count_trades(BTC_OVERLAY)
    eth = _count_trades(ETH_OVERLAY)
    stage = _infer_stage(processes, tail)
    status: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "stage": stage,
        "realign_log": str(log_path) if log_path else None,
        "tail": tail,
        "processes": processes,
        "v1_overlay_counts": {"BTC": btc, "ETH": eth},
        "alignment_audit_path": str(ALIGNMENT_AUDIT),
    }
    if run_audit and (btc["rows"] > 0 or eth["rows"] > 0):
        status["alignment_audit_run"] = _run_alignment_audit(sample_size)
    else:
        _write_waiting_alignment_audit("waiting_for_runtime_rows", sample_size)
        status["alignment_audit_run"] = {"skipped": True}
    return status


def main() -> int:
    ap = argparse.ArgumentParser(description="Watch aligned v1 lifecycle and refresh status/audit reports.")
    ap.add_argument("--loop", type=int, default=0, help="Loop interval seconds; 0 means run once")
    ap.add_argument("--sample-size", type=int, default=20)
    ap.add_argument("--run-audit", action="store_true")
    args = ap.parse_args()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        status = build_status(run_audit=bool(args.run_audit), sample_size=args.sample_size)
        STATUS_REPORT.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.loop <= 0:
            break
        time.sleep(args.loop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
