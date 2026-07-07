#!/usr/bin/env python3
from __future__ import annotations

"""Research-only optimizer for the current 06173 cluster gate.

This wrapper reuses the corrected closed-candle cluster search engine, but
writes the exact report names requested for the 061 follow-up plan. It never
touches live top159 configs, caches, ledgers, or process state.
"""

import os
from pathlib import Path

import run_top159_shock_filter_cluster_targeted_search as search

ROOT = Path("/Users/mac/polyfun")
REPORTS = ROOT / "reports"

search.OUT_AUDIT = REPORTS / "top159_061_optimization_bug_audit_latest.json"
search.OUT_CLUSTERS = REPORTS / "top159_061_optimization_bad_clusters_latest.json"
search.OUT_CANDIDATES = REPORTS / "top159_061_optimization_candidates_latest.jsonl"
search.OUT_CHECKPOINT = REPORTS / "top159_061_optimization_checkpoint_latest.json"
search.OUT_LEADERBOARD = REPORTS / "top159_061_optimization_leaderboard_latest.json"
search.OUT_COMPARE = REPORTS / "top159_061_optimization_180_365_archive_compare_latest.json"
search.OUT_TABLE = REPORTS / "top159_061_optimization_180_365_archive_compare_latest.md"
search.OUT_VERDICT = REPORTS / "top159_061_optimization_unique_verdict_latest.md"
search.OUT_LOG = REPORTS / "top159_061_optimization_run.log"

# Wide enough to be meaningful overnight, still bounded for a 4-worker M4 run.
os.environ.setdefault("TOP159_CLUSTER_EXECUTOR", "thread")
os.environ.setdefault("TOP159_CLUSTER_EXPANDED", "1")
os.environ.setdefault("TOP159_CLUSTER_DEEP", "1")
os.environ.setdefault("TOP159_CLUSTER_PERSIST_MODE", "top")
os.environ.setdefault("TOP159_CLUSTER_KEEP_TOP", "12000")
os.environ.setdefault("TOP159_CLUSTER_TOP", "130")
os.environ.setdefault("TOP159_CLUSTER_PAIR_TOP", "120")
os.environ.setdefault("TOP159_CLUSTER_TRIPLE_TOP", "95")
os.environ.setdefault("TOP159_CLUSTER_QUAD_TOP", "60")
os.environ.setdefault("TOP159_CLUSTER_MAX_CANDIDATES", "900000")
os.environ.setdefault("TOP159_CLUSTER_LIMIT", "150")
os.environ.setdefault("TOP159_CLUSTER_CHUNK_SIZE", "512")


if __name__ == "__main__":
    raise SystemExit(search.main())
