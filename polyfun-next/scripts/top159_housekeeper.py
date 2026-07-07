#!/usr/bin/env python3
"""Safe research-artifact housekeeper for top159.

This script deliberately avoids live runtime/config/state paths.  It only
targets large, reproducible research intermediates under /Users/mac/polyfun/reports.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List


ROOT = Path("/Users/mac/polyfun")
REPORTS = ROOT / "reports"
MANIFEST_DIR = REPORTS / "_housekeeper_manifests"

LIVE_PROTECTED_PARTS = (
    "/polyfun-next/runtime/",
    "/polyfun-next/config/",
    "/polyfun-next/.venv/",
    "/config/",
    "/runtime/",
    "/user_data/",
    "/polymarket/",
)

KEEP_NAME_PARTS = (
    "_compare_latest",
    "_unique_verdict_latest",
    "_bug_audit_latest",
    "top159_formal_live_loop_latest",
    "top159_live_candidate_generation_latest",
    "top159_v2_acceptance_latest",
    "top159_claim_status_latest",
    "top159_ws_accelerator_status_latest",
    "monitor_top159",
)

RESEARCH_INTERMEDIATE_PARTS = (
    "_candidates_latest",
    "_results_latest",
    "_candidate_summary_latest",
    "_all_candidate_summary_latest",
    "_cache_",
    "_checkpoint",
    "_marathon_state",
    "fullfield_hyperopt_results",
    "wide_fast_candidates",
    "extreme_candidates",
    "cluster_candidates",
    "pure_prediction_curve_trade_table",
)


@dataclass
class Candidate:
    path: str
    size: int
    mtime: float
    risk: str
    reason: str
    action: str
    digest: str | None = None


def short_digest(path: Path, size: int) -> str:
    """Fast content marker for manifest use without reading multi-GB files."""
    h = hashlib.sha256()
    h.update(str(path).encode())
    h.update(str(size).encode())
    try:
        with path.open("rb") as f:
            head = f.read(1024 * 1024)
            h.update(head)
            if size > 2 * 1024 * 1024:
                f.seek(max(0, size - 1024 * 1024))
                h.update(f.read(1024 * 1024))
    except OSError:
        return "unreadable"
    return h.hexdigest()[:16]


def is_live_protected(path: Path) -> bool:
    s = str(path)
    return any(part in s for part in LIVE_PROTECTED_PARTS)


def should_keep(path: Path) -> bool:
    name = path.name
    return any(part in name for part in KEEP_NAME_PARTS)


def is_research_intermediate(path: Path) -> tuple[bool, str]:
    name = path.name
    if any(part in name for part in RESEARCH_INTERMEDIATE_PARTS):
        return True, "research_intermediate_name"
    if path.suffix in {".tmp", ".log"} and path.name.startswith("_tmp"):
        return True, "tmp_research_file"
    return False, ""


def collect(min_mb: int) -> List[Candidate]:
    min_bytes = min_mb * 1024 * 1024
    out: List[Candidate] = []
    for path in REPORTS.rglob("*"):
        if not path.is_file():
            continue
        if is_live_protected(path) or should_keep(path):
            continue
        try:
            st = path.stat()
        except OSError:
            continue
        if st.st_size < min_bytes:
            continue
        ok, reason = is_research_intermediate(path)
        if not ok:
            continue
        out.append(
            Candidate(
                path=str(path),
                size=st.st_size,
                mtime=st.st_mtime,
                risk="low_research_only",
                reason=reason,
                action="delete",
                digest=short_digest(path, st.st_size),
            )
        )
    return sorted(out, key=lambda c: c.size, reverse=True)


def write_manifest(candidates: Iterable[Candidate], mode: str) -> Path:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = MANIFEST_DIR / f"top159_housekeeper_{mode}_{ts}.json"
    payload = {
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": mode,
        "root": str(ROOT),
        "reports": str(REPORTS),
        "candidates": [asdict(c) for c in candidates],
        "totalBytes": sum(c.size for c in candidates),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    latest = MANIFEST_DIR / f"top159_housekeeper_{mode}_latest.json"
    latest.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def apply(candidates: Iterable[Candidate]) -> dict:
    removed = []
    errors = []
    for c in candidates:
        path = Path(c.path)
        try:
            if path.exists() and path.is_file():
                path.unlink()
                removed.append(c)
        except OSError as exc:
            errors.append({"path": c.path, "error": str(exc)})
    return {
        "removed": [asdict(c) for c in removed],
        "errors": errors,
        "removedBytes": sum(c.size for c in removed),
    }


def human(n: int) -> str:
    for unit in ["B", "K", "M", "G", "T"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}P"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="Delete low-risk research intermediates.")
    ap.add_argument("--min-mb", type=int, default=100)
    ap.add_argument("--daemon", action="store_true", help="Run periodic low-disk checks.")
    ap.add_argument("--interval-sec", type=int, default=21600)
    ap.add_argument("--free-threshold-gb", type=float, default=20.0)
    args = ap.parse_args()

    def once(force_apply: bool) -> None:
        free = shutil.disk_usage(ROOT).free
        below = free < args.free_threshold_gb * 1024 * 1024 * 1024
        do_apply = force_apply and (below or not args.daemon)
        candidates = collect(args.min_mb)
        mode = "apply" if do_apply else "dry_run"
        manifest = write_manifest(candidates, mode)
        print(f"mode={mode}")
        print(f"free={human(free)} threshold={args.free_threshold_gb:.1f}G below_threshold={below}")
        print(f"manifest={manifest}")
        print(f"candidates={len(candidates)} total={human(sum(c.size for c in candidates))}")
        for c in candidates[:30]:
            print(f"{human(c.size):>8} {c.reason:<28} {c.path}")
        if do_apply:
            result = apply(candidates)
            write_manifest([Candidate(**r) for r in result["removed"]], "applied_removed")
            print(f"removed={len(result['removed'])} removedBytes={human(result['removedBytes'])}")
            if result["errors"]:
                print(f"errors={len(result['errors'])}")
                for e in result["errors"][:10]:
                    print(f"ERROR {e['path']}: {e['error']}")

    if args.daemon:
        while True:
            once(args.apply)
            time.sleep(max(60, args.interval_sec))
    else:
        once(args.apply)


if __name__ == "__main__":
    main()
