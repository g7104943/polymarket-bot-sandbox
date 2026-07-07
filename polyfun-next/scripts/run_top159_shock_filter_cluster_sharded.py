#!/usr/bin/env python3
from __future__ import annotations

"""Robust 4-core launcher for targeted top159 shock cluster search.

Each shard is an independent Python process that evaluates a disjoint slice of
the candidate list and writes separate files. A final merge step ranks all
rows and writes the normal `*_latest` reports. This gives multi-core speed
without shared-file contention.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path("/Users/mac/polyfun")
REPORTS = ROOT / "reports"
BJ = timezone(timedelta(hours=8))

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_top159_shock_filter_cluster_targeted_search as target  # type: ignore  # noqa: E402


def bj_now() -> datetime:
    return datetime.now(BJ)


def iso(dt: datetime) -> str:
    return dt.astimezone(BJ).isoformat(timespec="seconds")


def parse_deadline(s: str | None, hours: float) -> datetime:
    if s:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=BJ)
        return dt.astimezone(BJ)
    return bj_now() + timedelta(hours=hours)


def remove_default_outputs() -> None:
    if os.environ.get("TOP159_CLUSTER_PRESERVE_MERGED", "0").strip() == "1":
        # Segmented long runs refresh the merged latest reports after every
        # segment. A new segment still needs fresh shard files, but it should
        # not briefly erase the merged table/verdict while the next segment is
        # running.
        return
    for p in [
        target.OUT_AUDIT,
        target.OUT_CLUSTERS,
        target.OUT_CANDIDATES,
        target.OUT_CHECKPOINT,
        target.OUT_COMPARE,
        target.OUT_VERDICT,
        target.OUT_LOG,
        REPORTS / "top159_shock_filter_cluster_stdout.log",
        REPORTS / "top159_shock_filter_cluster.pid",
    ]:
        if p.exists():
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()


def read_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def read_completed_count(path: Path, fallback: int) -> int:
    if not path.exists():
        return fallback
    try:
        obj = json.loads(path.read_text())
        return int(obj.get("completed_candidates") or fallback)
    except Exception:
        return fallback


def trim_top(rows: list[dict], limit: int) -> list[dict]:
    if limit <= 0 or len(rows) <= limit:
        return rows
    rows.sort(key=target.row_score, reverse=True)
    return rows[:limit]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline-bj", default=os.environ.get("TOP159_CLUSTER_DEADLINE_BJ"))
    ap.add_argument("--hours", type=float, default=float(os.environ.get("TOP159_CLUSTER_HOURS", "5")))
    ap.add_argument("--shards", type=int, default=int(os.environ.get("TOP159_CLUSTER_SHARDS", "4")))
    ap.add_argument("--max-candidates", type=int, default=int(os.environ.get("TOP159_CLUSTER_MAX_CANDIDATES", "30000")))
    ap.add_argument("--cluster-limit", type=int, default=int(os.environ.get("TOP159_CLUSTER_LIMIT", "80")))
    ap.add_argument("--chunk-size", type=int, default=int(os.environ.get("TOP159_CLUSTER_CHUNK_SIZE", "500")))
    ap.add_argument("--keep-top", type=int, default=int(os.environ.get("TOP159_CLUSTER_KEEP_TOP", "5000")))
    ap.add_argument("--start-stagger-sec", type=float, default=float(os.environ.get("TOP159_CLUSTER_START_STAGGER_SEC", "0")))
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()
    if args.shards < 1:
        raise SystemExit("--shards must be >= 1")

    started = bj_now()
    deadline = parse_deadline(args.deadline_bj, args.hours)
    if args.fresh:
        remove_default_outputs()
        for i in range(args.shards):
            suffix = f"shard{i}"
            for base in [
                target.OUT_AUDIT,
                target.OUT_CLUSTERS,
                target.OUT_CANDIDATES,
                target.OUT_CHECKPOINT,
                target.OUT_COMPARE,
                target.OUT_TABLE,
                target.OUT_VERDICT,
                target.OUT_LOG,
            ]:
                p = base.with_name(f"{base.stem}_{suffix}{base.suffix}")
                if p.exists():
                    p.unlink()
            out = REPORTS / f"top159_shock_filter_cluster_stdout_{suffix}.log"
            if out.exists():
                out.unlink()

    target.log(f"cluster sharded launcher started; deadline={iso(deadline)} shards={args.shards}")
    procs: list[tuple[int, subprocess.Popen, object]] = []
    for i in range(args.shards):
        suffix = f"shard{i}"
        stdout_path = REPORTS / f"top159_shock_filter_cluster_stdout_{suffix}.log"
        fh = stdout_path.open("w", encoding="utf-8")
        cmd = [
            sys.executable,
            "-u",
            str(SCRIPT_DIR / "run_top159_shock_filter_cluster_targeted_search.py"),
            "--fresh",
            "--deadline-bj",
            iso(deadline),
            "--workers",
            "1",
            "--chunk-size",
            str(args.chunk_size),
            "--max-candidates",
            str(args.max_candidates),
            "--cluster-limit",
            str(args.cluster_limit),
            "--shard-count",
            str(args.shards),
            "--shard-index",
            str(i),
            "--output-suffix",
            suffix,
        ]
        env = os.environ.copy()
        env["TOP159_CLUSTER_EXECUTOR"] = "thread"
        env["TOP159_CLUSTER_WORKERS"] = "1"
        env["OMP_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["VECLIB_MAXIMUM_THREADS"] = "1"
        env["TOP159_CLUSTER_PERSIST_MODE"] = os.environ.get("TOP159_CLUSTER_PERSIST_MODE", "top")
        env["TOP159_CLUSTER_KEEP_TOP"] = str(args.keep_top)
        proc = subprocess.Popen(cmd, cwd=str(ROOT / "polyfun-next"), env=env, stdout=fh, stderr=subprocess.STDOUT)
        procs.append((i, proc, fh))
        if args.start_stagger_sec > 0 and i < args.shards - 1:
            target.log(f"started shard={i}; staggering next start by {args.start_stagger_sec:.1f}s")
            import time
            time.sleep(args.start_stagger_sec)
    (REPORTS / "top159_shock_filter_cluster.pid").write_text(",".join(str(p.pid) for _i, p, _fh in procs), encoding="utf-8")

    failures: list[dict] = []
    for i, proc, fh in procs:
        rc = proc.wait()
        fh.close()
        if rc != 0:
            failures.append({"shard": i, "returncode": rc})

    all_rows: list[dict] = []
    retained_rows = 0
    completed_candidates = 0
    for i in range(args.shards):
        suffix = f"shard{i}"
        p = target.OUT_CANDIDATES.with_name(f"{target.OUT_CANDIDATES.stem}_{suffix}{target.OUT_CANDIDATES.suffix}")
        rows = read_rows(p)
        retained_rows += len(rows)
        cp = target.OUT_CHECKPOINT.with_name(f"{target.OUT_CHECKPOINT.stem}_{suffix}{target.OUT_CHECKPOINT.suffix}")
        completed_candidates += read_completed_count(cp, len(rows))
        all_rows.extend(rows)
        all_rows = trim_top(all_rows, args.keep_top * args.shards)

    enriched, _truth = target.ext.load_or_build_enriched()
    atoms = target.build_atom_store(enriched)
    compare = target.summarize_final(enriched, atoms, all_rows)
    bug_audit = target.ext.bug_audit(enriched)
    target.write_json(target.OUT_AUDIT, {
        "created_at_bj": target.iso(target.bj_now()),
        "feature_alignment_version": _truth.get("shockFeatureAlignmentVersion"),
        "research_only": True,
        "live_configs_touched": False,
        "search_type": "bad_cluster_targeted_sharded",
        "bugAudit": bug_audit,
    })
    compare["shardedRun"] = {
        "started_at_bj": iso(started),
        "updated_at_bj": iso(bj_now()),
        "deadline_bj": iso(deadline),
        "shards": args.shards,
        "completedCandidates": completed_candidates,
        "retainedRowsForMerge": retained_rows,
        "failures": failures,
        "candidateStartOffset": int(os.environ.get("TOP159_CLUSTER_START_OFFSET", "0")),
        "candidateTopN": int(os.environ.get("TOP159_CLUSTER_TOP", "80")),
        "dedupEnabled": os.environ.get("TOP159_CLUSTER_DEDUP", "1").strip() != "0",
        "live_config_mutated": False,
    }
    compare["audit"] = {
        "bugAudit": bug_audit,
        "featureAlignmentVersion": _truth.get("shockFeatureAlignmentVersion"),
        "researchOnly": True,
        "liveConfigMutated": False,
    }
    target.write_json(target.OUT_COMPARE, compare)
    target.OUT_TABLE.write_text(target.render_profit_table(compare), encoding="utf-8")
    target.OUT_VERDICT.write_text(target.render_verdict(compare), encoding="utf-8")
    with target.OUT_CANDIDATES.open("w", encoding="utf-8") as fh:
        for r in trim_top(all_rows, args.keep_top):
            fh.write(json.dumps(target.compact(r), ensure_ascii=False, sort_keys=True, default=str) + "\n")
    target.write_checkpoint(started, deadline, completed_candidates, completed_candidates, all_rows, "complete" if not failures else "complete_with_failures")
    target.log(f"cluster sharded launcher complete candidates={completed_candidates} retained={retained_rows} failures={failures} compare={target.OUT_COMPARE}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
