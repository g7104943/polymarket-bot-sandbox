#!/usr/bin/env python3
from __future__ import annotations

"""Segmented long-run launcher for top159 shock cluster search.

The targeted search is fast after vectorization, but holding tens of millions
of candidate dicts in memory is unsafe on a MacBook. This wrapper evaluates the
candidate space in deterministic offset segments, merges only the top rows, and
refreshes the normal latest reports after every segment.
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path("/Users/mac/polyfun")
REPORTS = ROOT / "reports"
BJ = timezone(timedelta(hours=8))

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_top159_shock_filter_cluster_targeted_search as target  # type: ignore  # noqa: E402


OUT_SEGMENT_CHECKPOINT = REPORTS / "top159_shock_filter_cluster_segmented_checkpoint_latest.json"
OUT_SEGMENT_AGG = REPORTS / "top159_shock_filter_cluster_segmented_aggregate_latest.jsonl"


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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def write_aggregate(rows: list[dict[str, Any]], keep_top: int) -> None:
    rows = target.leaderboard(rows, keep_top)
    OUT_SEGMENT_AGG.parent.mkdir(parents=True, exist_ok=True)
    with OUT_SEGMENT_AGG.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(target.compact(r), ensure_ascii=False, sort_keys=True, default=str) + "\n")
    with target.OUT_CANDIDATES.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(target.compact(r), ensure_ascii=False, sort_keys=True, default=str) + "\n")


def refresh_reports(rows: list[dict[str, Any]], meta: dict[str, Any]) -> None:
    enriched, truth = target.ext.load_or_build_enriched()
    atoms = target.build_atom_store(enriched)
    compare = target.summarize_final(enriched, atoms, rows)
    bug_audit = target.ext.bug_audit(enriched)
    compare["segmentedRun"] = meta
    compare["audit"] = {
        "bugAudit": bug_audit,
        "featureAlignmentVersion": truth.get("shockFeatureAlignmentVersion"),
        "researchOnly": True,
        "liveConfigMutated": False,
    }
    target.write_json(target.OUT_AUDIT, {
        "created_at_bj": iso(bj_now()),
        "feature_alignment_version": truth.get("shockFeatureAlignmentVersion"),
        "research_only": True,
        "live_configs_touched": False,
        "search_type": "bad_cluster_targeted_segmented",
        "bugAudit": bug_audit,
    })
    target.write_json(target.OUT_COMPARE, compare)
    target.OUT_TABLE.write_text(target.render_profit_table(compare), encoding="utf-8")
    target.OUT_VERDICT.write_text(target.render_verdict(compare), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline-bj", default=os.environ.get("TOP159_CLUSTER_DEADLINE_BJ"))
    ap.add_argument("--hours", type=float, default=float(os.environ.get("TOP159_CLUSTER_HOURS", "5")))
    ap.add_argument("--segment-size", type=int, default=int(os.environ.get("TOP159_CLUSTER_SEGMENT_SIZE", "5000000")))
    ap.add_argument("--start-offset", type=int, default=int(os.environ.get("TOP159_CLUSTER_START_OFFSET", "0")))
    ap.add_argument("--max-segments", type=int, default=int(os.environ.get("TOP159_CLUSTER_MAX_SEGMENTS", "20")))
    ap.add_argument("--shards", type=int, default=int(os.environ.get("TOP159_CLUSTER_SHARDS", "4")))
    ap.add_argument("--cluster-limit", type=int, default=int(os.environ.get("TOP159_CLUSTER_LIMIT", "120")))
    ap.add_argument("--top-n", type=int, default=int(os.environ.get("TOP159_CLUSTER_TOP", "120")))
    ap.add_argument("--chunk-size", type=int, default=int(os.environ.get("TOP159_CLUSTER_CHUNK_SIZE", "1000")))
    ap.add_argument("--keep-top", type=int, default=int(os.environ.get("TOP159_CLUSTER_KEEP_TOP", "40000")))
    ap.add_argument("--start-stagger-sec", type=float, default=float(os.environ.get("TOP159_CLUSTER_START_STAGGER_SEC", "20")))
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    deadline = parse_deadline(args.deadline_bj, args.hours)
    started = bj_now()
    aggregate: list[dict[str, Any]] = [] if args.fresh else read_jsonl(OUT_SEGMENT_AGG)
    completed_segments = 0
    completed_candidates = 0

    target.log(
        f"segmented cluster search started; deadline={iso(deadline)} "
        f"segment_size={args.segment_size} start_offset={args.start_offset}"
    )
    for seg in range(args.max_segments):
        if bj_now() >= deadline - timedelta(minutes=20):
            break
        offset = args.start_offset + seg * args.segment_size
        segment_log = REPORTS / f"top159_shock_filter_cluster_segment_{seg:03d}.log"
        env = os.environ.copy()
        env.update({
            "TOP159_CLUSTER_EXPANDED": "1",
            "TOP159_CLUSTER_DEEP": "1",
            "TOP159_CLUSTER_PAIR_TOP": "80",
            "TOP159_CLUSTER_TRIPLE_TOP": "50",
            "TOP159_CLUSTER_QUAD_TOP": "48",
            "TOP159_CLUSTER_TOP": str(args.top_n),
            "TOP159_CLUSTER_START_OFFSET": str(offset),
            "TOP159_CLUSTER_DEDUP": "0",
            "TOP159_CLUSTER_PERSIST_MODE": "top",
            "TOP159_CLUSTER_KEEP_TOP": str(min(args.keep_top, 10000)),
            "TOP159_CLUSTER_PRESERVE_MERGED": "1",
        })
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "run_top159_shock_filter_cluster_sharded.py"),
            "--fresh",
            "--deadline-bj",
            iso(deadline),
            "--shards",
            str(args.shards),
            "--max-candidates",
            str(args.segment_size),
            "--cluster-limit",
            str(args.cluster_limit),
            "--chunk-size",
            str(args.chunk_size),
            "--keep-top",
            str(min(args.keep_top, 10000)),
            "--start-stagger-sec",
            str(args.start_stagger_sec),
        ]
        target.log(f"segment={seg} offset={offset} running cmd={' '.join(cmd)}")
        with segment_log.open("w", encoding="utf-8") as fh:
            rc = subprocess.call(cmd, cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT)
        if rc != 0:
            target.log(f"segment={seg} failed rc={rc}; stopping segmented search")
            break
        rows = read_jsonl(target.OUT_CANDIDATES)
        aggregate.extend(rows)
        aggregate = target.leaderboard(aggregate, args.keep_top)
        completed_segments += 1
        completed_candidates += args.segment_size
        meta = {
            "started_at_bj": iso(started),
            "updated_at_bj": iso(bj_now()),
            "deadline_bj": iso(deadline),
            "completedSegments": completed_segments,
            "completedCandidates": completed_candidates,
            "lastSegmentOffset": offset,
            "segmentSize": args.segment_size,
            "aggregateRows": len(aggregate),
            "liveConfigMutated": False,
        }
        write_aggregate(aggregate, args.keep_top)
        refresh_reports(aggregate, meta)
        write_json(OUT_SEGMENT_CHECKPOINT, meta)
        target.log(f"segment={seg} complete aggregate={len(aggregate)} candidates={completed_candidates}")

    meta = {
        "started_at_bj": iso(started),
        "updated_at_bj": iso(bj_now()),
        "deadline_bj": iso(deadline),
        "completedSegments": completed_segments,
        "completedCandidates": completed_candidates,
        "aggregateRows": len(aggregate),
        "status": "complete",
        "liveConfigMutated": False,
    }
    write_aggregate(aggregate, args.keep_top)
    refresh_reports(aggregate, meta)
    write_json(OUT_SEGMENT_CHECKPOINT, meta)
    target.log(f"segmented cluster search complete segments={completed_segments} candidates={completed_candidates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
