#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

CORE_KEYWORDS = (
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


def _ps_rows() -> List[Dict[str, str]]:
    proc = subprocess.run(
        ["ps", "-eo", "ppid=,pid=,state=,args=", "-ww"],
        capture_output=True,
        text=True,
        check=False,
    )
    rows: List[Dict[str, str]] = []
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        ppid, pid, state, args = parts
        rows.append({"ppid": ppid, "pid": pid, "state": state, "args": args})
    return rows


def _group_zombies(rows: List[Dict[str, str]]) -> Dict[str, Any]:
    by_ppid: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        ppid = str(r["ppid"])
        g = by_ppid.setdefault(ppid, {"ppid": ppid, "parent_cmd": "", "zombies": [], "non_zombie_children": []})
        if r["state"] == "Z":
            g["zombies"].append({"pid": r["pid"], "args": r["args"][:120]})
        else:
            g["non_zombie_children"].append({"pid": r["pid"], "state": r["state"], "args": r["args"][:120]})
        if not g["parent_cmd"]:
            # best effort: child rows do not include parent cmd; query separately later
            pass
    # hydrate parent cmd
    for ppid, g in by_ppid.items():
        try:
            rp = subprocess.run(["ps", "-p", ppid, "-o", "args=", "-ww"], capture_output=True, text=True, timeout=2)
            cmd = (rp.stdout or "").strip()
        except Exception:
            cmd = ""
        g["parent_cmd"] = cmd
    return by_ppid


def _is_core_cmd(cmd: str) -> bool:
    low = (cmd or "").lower()
    return any(k.lower() in low for k in CORE_KEYWORDS)


def _is_safe_orphan(cmd: str) -> bool:
    low = (cmd or "").strip().lower()
    return low.startswith("python3 -") or low.startswith("python -")


def _kill_parent(ppid: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"ppid": ppid, "term_sent": False, "kill_sent": False, "gone": False}
    try:
        os.kill(int(ppid), signal.SIGTERM)
        out["term_sent"] = True
    except ProcessLookupError:
        out["gone"] = True
        return out
    except Exception as e:
        out["error"] = str(e)
        return out
    time.sleep(1.0)
    try:
        os.kill(int(ppid), 0)
    except ProcessLookupError:
        out["gone"] = True
        return out
    except Exception:
        pass
    try:
        os.kill(int(ppid), signal.SIGKILL)
        out["kill_sent"] = True
    except ProcessLookupError:
        out["gone"] = True
        return out
    except Exception as e:
        out["error"] = str(e)
        return out
    time.sleep(0.5)
    try:
        os.kill(int(ppid), 0)
        out["gone"] = False
    except ProcessLookupError:
        out["gone"] = True
    except Exception:
        out["gone"] = False
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="安全清理僵尸父进程（仅外部孤儿 python3 -）")
    ap.add_argument("--apply", action="store_true", help="实际执行 kill；默认仅预览")
    ap.add_argument("--ppid", type=str, default="", help="只处理指定 PPID")
    args = ap.parse_args()

    rows = _ps_rows()
    grouped = _group_zombies(rows)
    report: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "APPLY" if args.apply else "DRY_RUN",
        "core_blocking": [],
        "external_orphan": [],
        "other": [],
        "actions": [],
    }

    for ppid, g in sorted(grouped.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 10**9):
        zombies = g.get("zombies") or []
        if not zombies:
            continue
        if args.ppid and str(args.ppid) != str(ppid):
            continue
        cmd = str(g.get("parent_cmd") or "")
        item = {
            "ppid": ppid,
            "parent_cmd": cmd,
            "zombie_count": len(zombies),
            "non_zombie_children": len(g.get("non_zombie_children") or []),
        }
        if _is_core_cmd(cmd):
            report["core_blocking"].append(item)
            continue
        if _is_safe_orphan(cmd):
            report["external_orphan"].append(item)
            if args.apply and int(item["non_zombie_children"]) == 0:
                act = _kill_parent(ppid)
                report["actions"].append(act)
            continue
        report["other"].append(item)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
