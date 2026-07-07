#!/usr/bin/env python3
"""Overnight hyperopt for the top159 shock filter.

Research-only: reads cached shock-filter datasets and writes reports. It never
modifies live configs, model profiles, ledgers, or order state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Keep each worker modest; the parent runs 4 workers by default.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOP159_SHOCK_EXTREME_THREADS", "1")

ROOT = Path("/Users/mac/polyfun")
NEXT = ROOT / "polyfun-next"
REPORTS = ROOT / "reports"
SCRIPT_DIR = NEXT / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_top159_shock_filter_extreme_search as ext  # type: ignore  # noqa: E402

BJ = timezone(timedelta(hours=8))

BUG_AUDIT = REPORTS / "top159_shock_filter_overnight_bug_audit_latest.json"
CHECKPOINT = REPORTS / "top159_shock_filter_overnight_checkpoint_latest.json"
LEADERBOARD = REPORTS / "top159_shock_filter_overnight_leaderboard_latest.json"
COMPARE = REPORTS / "top159_shock_filter_overnight_compare_latest.json"
VERDICT = REPORTS / "top159_shock_filter_overnight_unique_verdict_latest.md"
CANDIDATES = REPORTS / "top159_shock_filter_overnight_candidates_latest.jsonl"
LOG = REPORTS / "top159_shock_filter_overnight_run.log"

CURRENT_LIVE_SHOCK = {
    "family": "model_gate",
    "rule_mode": "any_all",
    "body_min": 0.45,
    "range_q_min": 0.60,
    "volume_mult_min": 0.0,
    "action": "model_gate",
    "min_gate_win_prob": 0.54,
    "model_hyper": {
        "engine": "lightgbm",
        "n_estimators": 90,
        "learning_rate": 0.045,
        "num_leaves": 16,
        "min_child_samples": 50,
        "reg_lambda": 1.0,
    },
}

G_ENRICHED = None


def bj_now() -> datetime:
    return datetime.now(BJ)


def iso(dt: datetime) -> str:
    return dt.astimezone(BJ).isoformat(timespec="seconds")


def log(msg: str) -> None:
    line = f"[{iso(bj_now())}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def stable_key(params: Dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def row_score(row: Dict[str, Any], baseline_lookup: Optional[Dict[str, Dict[str, Any]]] = None) -> Tuple:
    """Risk-first sortable score. Higher tuple is better."""
    w180 = row.get("w180", {}) or {}
    w365 = row.get("w365", {}) or {}
    p180 = row.get("pressure180", {}) or {}
    p365 = row.get("pressure365", {}) or {}
    strict = int(bool(row.get("strict_pass")))
    # Need both windows to block more losers than winners; emphasize margin.
    edge180 = int(w180.get("blocked_losers", 0)) - int(w180.get("blocked_winners", 0))
    edge365 = int(w365.get("blocked_losers", 0)) - int(w365.get("blocked_winners", 0))
    min_edge = min(edge180, edge365)
    p5 = min(safe_float(p180.get("mc_p05", 0)), safe_float(p365.get("mc_p05", 0)))
    p50 = min(safe_float(p180.get("mc_p50", 0)), safe_float(p365.get("mc_p50", 0)))
    fak2 = min(safe_float(row.get("fak2_180", {}).get("ending_funds", 0)), safe_float(row.get("fak2_365", {}).get("ending_funds", 0)))
    retention = min(safe_float(w180.get("retention_rate", 0)), safe_float(w365.get("retention_rate", 0)))
    max_dd = max(safe_float(w180.get("max_drawdown", 999999)), safe_float(w365.get("max_drawdown", 999999)))
    trade_floor = min(int(w180.get("trades", 0)), int(w365.get("trades", 0)))
    # Sparse shock definitions can look excellent by touching only a handful of
    # trades. Rank them behind broader candidates unless they pass hard gates.
    sample_floor = min(int(w180.get("condition_base_trades", 0)), int(w365.get("condition_base_trades", 0)))
    sample_ok = int(sample_floor >= 30)
    return (strict, sample_ok, min_edge, p5, p50, fak2, retention, sample_floor, -max_dd, trade_floor)


def _metric_row_to_window(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize extreme-search fast metrics to the overnight candidate schema."""
    return {
        "trades": int(row.get("trades", 0)),
        "wins": int(row.get("wins", 0)),
        "losses": int(row.get("losses", 0)),
        "win_rate_pct": safe_float(row.get("winRatePct", 0)),
        "base_trades": int(row.get("baseTrades", 0)),
        "base_win_rate_pct": safe_float(row.get("baseWinRatePct", 0)),
        "retention_rate": safe_float(row.get("retentionPct", 0)),
        "blocked_trades": int(row.get("blockedTrades", 0)),
        "blocked_winners": int(row.get("blockedWinners", 0)),
        "blocked_losers": int(row.get("blockedLosers", 0)),
        "blocked_loser_minus_winner": int(row.get("blockedLoserMinusWinner", 0)),
        "condition_base_trades": int(row.get("conditionBaseTrades", 0)),
        "condition_base_win_rate_pct": safe_float(row.get("conditionBaseWinRatePct", 0)),
        "condition_kept_trades": int(row.get("conditionKeptTrades", 0)),
        "condition_kept_win_rate_pct": safe_float(row.get("conditionKeptWinRatePct", 0)),
    }


def grouped_fast_candidate_to_row(group: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a grouped 180d+365d candidate into one sortable candidate row."""
    by_window = {r.get("window"): r for r in group.get("rows", []) if isinstance(r, dict)}
    params = group.get("params", {}) or {}
    return {
        "candidate_id": stable_key(params),
        "params": params,
        "strict_pass": bool(group.get("passedFast")),
        "fast_score": safe_float(group.get("score", 0)),
        "reasons": group.get("reasons", []),
        "w180": _metric_row_to_window(by_window.get("180d", {})),
        "w365": _metric_row_to_window(by_window.get("365d", {})),
    }


def compact_row(row: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "candidate_id", "strict_pass", "params", "w180", "w365", "pressure180", "pressure365",
        "fak2_180", "fak2_365", "archived_real", "error",
    ]
    return {k: row.get(k) for k in keys if k in row}


def leaderboard(rows: Sequence[Dict[str, Any]], limit: int = 100) -> List[Dict[str, Any]]:
    best_by_id: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if row.get("error"):
            continue
        cid = row.get("candidate_id") or stable_key(row.get("params", {}) or {})
        if cid not in best_by_id or row_score(row) > row_score(best_by_id[cid]):
            best_by_id[cid] = row
    good = list(best_by_id.values())
    good.sort(key=row_score, reverse=True)
    return [compact_row(r) for r in good[:limit]]


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def append_candidates(rows: Sequence[Dict[str, Any]]) -> None:
    CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
    with CANDIDATES.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(compact_row(row), ensure_ascii=False, default=str) + "\n")


def load_done_ids() -> set[str]:
    done: set[str] = set()
    if CANDIDATES.exists():
        with CANDIDATES.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                    cid = obj.get("candidate_id")
                    if cid:
                        done.add(cid)
                except Exception:
                    continue
    return done


def init_worker() -> None:
    global G_ENRICHED
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("TOP159_SHOCK_EXTREME_THREADS", "1")
    G_ENRICHED, _truth = ext.load_or_build_enriched()


def evaluate_chunk(params_chunk: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    global G_ENRICHED
    if G_ENRICHED is None:
        init_worker()
    out = []
    try:
        metric_rows = ext.evaluate_param_list(G_ENRICHED, params_chunk)
        grouped = ext.group_rows(metric_rows)
        for group in grouped:
            out.append(grouped_fast_candidate_to_row(group))
    except Exception as exc:
        for p in params_chunk:
            out.append({"candidate_id": stable_key(p), "params": p, "error": repr(exc)})
    return out


def frange(start: float, stop: float, step: float) -> List[float]:
    vals = []
    x = start
    # float-safe inclusive range
    while x <= stop + 1e-9:
        vals.append(round(x, 4))
        x += step
    return vals


def condition_grid(coarse: bool = True) -> Iterable[Tuple[str, float, float, float]]:
    modes = list(dict.fromkeys(list(ext.RULE_MODES) + ["any_all", "one_h_all", "four_h_all", "any_terminal", "exhaustion"] ))
    if coarse:
        body_vals = frange(0.30, 0.75, 0.05)
        range_vals = frange(0.35, 0.90, 0.05)
    else:
        modes = ["any_all", "one_h_all", "four_h_all", "any_terminal", "exhaustion", "no_confirm"]
        body_vals = frange(0.325, 0.70, 0.025)
        range_vals = frange(0.40, 0.825, 0.025)
    volume_vals = [0.0, 1.1, 1.3, 1.6, 2.0]
    return product(modes, body_vals, range_vals, volume_vals)


def build_params(max_params: int) -> List[Dict[str, Any]]:
    params: List[Dict[str, Any]] = []

    def add(p: Dict[str, Any]) -> None:
        p = json.loads(json.dumps(p, sort_keys=True))
        params.append(p)

    add(CURRENT_LIVE_SHOCK)

    # Rule layer: broad but bounded. Hard-block finds structure; raise-score tests softer action.
    raise_thresholds = [0.54, 0.56, 0.58, 0.60, 0.62, 0.64]
    for mode, body, rq, vol in condition_grid(coarse=True):
        add({"family": "rule", "rule_mode": mode, "body_min": body, "range_q_min": rq, "volume_mult_min": vol, "action": "hard_block"})
        for score_min in raise_thresholds:
            add({"family": "rule", "rule_mode": mode, "body_min": body, "range_q_min": rq, "volume_mult_min": vol, "action": "raise_score", "shock_score_min": score_min})
        if len(params) >= max_params * 0.55:
            break

    # Dense refinement around plausible shock zones.
    for mode, body, rq, vol in condition_grid(coarse=False):
        add({"family": "rule", "rule_mode": mode, "body_min": body, "range_q_min": rq, "volume_mult_min": vol, "action": "hard_block"})
        add({"family": "rule", "rule_mode": mode, "body_min": body, "range_q_min": rq, "volume_mult_min": vol, "action": "raise_score", "shock_score_min": 0.58})
        if len(params) >= max_params * 0.72:
            break

    # Model layer: use logistic and lightweight LightGBM gates. Keep count controlled.
    model_hypers = [
        {"engine": "logistic", "C": 0.08},
        {"engine": "logistic", "C": 0.2},
        {"engine": "logistic", "C": 0.5},
        {"engine": "lightgbm", "n_estimators": 60, "learning_rate": 0.055, "num_leaves": 8, "min_child_samples": 80, "reg_lambda": 2.0},
        {"engine": "lightgbm", "n_estimators": 90, "learning_rate": 0.045, "num_leaves": 16, "min_child_samples": 50, "reg_lambda": 1.0},
        {"engine": "lightgbm", "n_estimators": 120, "learning_rate": 0.035, "num_leaves": 24, "min_child_samples": 70, "reg_lambda": 1.5},
        {"engine": "lightgbm", "n_estimators": 160, "learning_rate": 0.025, "num_leaves": 12, "min_child_samples": 40, "reg_lambda": 0.8},
    ]
    model_modes = ["any_all", "one_h_all", "four_h_all", "any_terminal", "exhaustion", "no_confirm", "both_all"]
    body_vals = frange(0.30, 0.70, 0.05)
    range_vals = frange(0.35, 0.80, 0.05)
    min_probs = frange(0.48, 0.64, 0.01)
    for hp in model_hypers:
        for mode, body, rq, vol, prob in product(model_modes, body_vals, range_vals, [0.0, 1.1, 1.3], min_probs):
            add({"family": "model_gate", "rule_mode": mode, "body_min": body, "range_q_min": rq, "volume_mult_min": vol, "action": "model_gate", "min_gate_win_prob": prob, "model_hyper": hp})
            if len(params) >= max_params:
                break
        if len(params) >= max_params:
            break

    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique: List[Dict[str, Any]] = []
    for p in params:
        sid = stable_key(p)
        if sid in seen:
            continue
        seen.add(sid)
        unique.append(p)
    return unique[:max_params]


def chunked(seq: Sequence[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(seq), size):
        yield list(seq[i:i + size])


def parse_deadline(s: Optional[str]) -> datetime:
    if s:
        # Accept naive Beijing time like 2026-05-04T08:00:00
        if s.endswith("Z"):
            return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(BJ)
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=BJ)
        return dt.astimezone(BJ)
    now = bj_now()
    target = now.replace(hour=8, minute=0, second=0, microsecond=0)
    if now >= target:
        target += timedelta(days=1)
    return target


def write_checkpoint(started: datetime, deadline: datetime, planned: int, done: int, rows: Sequence[Dict[str, Any]], status: str) -> None:
    lb = leaderboard(rows, limit=50)
    obj = {
        "status": status,
        "started_at_bj": iso(started),
        "updated_at_bj": iso(bj_now()),
        "deadline_bj": iso(deadline),
        "planned_candidates": planned,
        "completed_candidates": done,
        "candidate_file": str(CANDIDATES),
        "top": lb[:20],
        "live_config_mutated": False,
        "notes": "Research-only overnight shock-filter hyperopt; live top159 process/config untouched.",
    }
    write_json(CHECKPOINT, obj)
    write_json(LEADERBOARD, {"updated_at_bj": iso(bj_now()), "rows": lb})


def final_compare(rows: Sequence[Dict[str, Any]], top_n: int = 30) -> Dict[str, Any]:
    best_by_id: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if row.get("error"):
            continue
        cid = row.get("candidate_id") or stable_key(row.get("params", {}) or {})
        if cid not in best_by_id or row_score(row) > row_score(best_by_id[cid]):
            best_by_id[cid] = row
    good = list(best_by_id.values())
    good.sort(key=row_score, reverse=True)
    selected_best = good[0] if good else None
    named: List[Tuple[str, Dict[str, Any]]] = [
        ("current_new_archive_top159", {"family": "baseline"}),
        ("current_live_shock_filter", CURRENT_LIVE_SHOCK),
    ]
    if selected_best and selected_best.get("params"):
        named.append(("overnight_best", selected_best["params"]))
    # Add a few runner-ups for traceability, but keep final validation bounded.
    seen = {stable_key(p) for _, p in named}
    for r in good[:top_n]:
        p = r.get("params")
        if not p:
            continue
        sid = stable_key(p)
        if sid in seen:
            continue
        seen.add(sid)
        named.append((f"runner_up_{len(named)-2:02d}_{sid}", p))
        if len(named) >= min(top_n, 12):
            break
    enriched, _truth = ext.load_or_build_enriched()
    validation_rows, _val_keep_map = ext.final_validation_rows(enriched, named)
    try:
        archived_rows, archived_audit = ext.archived_real_rows(enriched, named[:6])
        archived = {"rows": archived_rows, "audit": archived_audit}
    except Exception as exc:
        archived = {"rows": [], "audit": {"error": repr(exc)}}
    return {
        "updated_at_bj": iso(bj_now()),
        "current_live_params": CURRENT_LIVE_SHOCK,
        "recommended_candidate": compact_row(selected_best) if selected_best else None,
        "fast_leaderboard": [compact_row(r) for r in good[:50]],
        "validationRows": validation_rows,
        "archivedRealRows": archived.get("rows", []),
        "audit": {
            "archiveAudit": archived.get("audit", {}),
            "methods": ["full_fill_buy_0.50", "fak_pressure_buy_0.52", "slot1_toxic_monte_carlo"],
            "researchOnlyNoLiveChange": True,
        },
    }


def write_verdict(compare: Dict[str, Any]) -> None:
    validation_rows = compare.get("validationRows", []) or []
    current = {"params": CURRENT_LIVE_SHOCK, "candidate_id": stable_key(CURRENT_LIVE_SHOCK), "validationRows": [r for r in validation_rows if r.get("name") == "current_live_shock_filter"]}
    rec = compare.get("recommended_candidate")
    if not rec:
        text = "# top159 冲击过滤隔夜搜索结论\n\n没有生成有效候选。live 不改。\n"
        VERDICT.write_text(text, encoding="utf-8")
        return
    # Determine whether to suggest shadow validation only.
    strict = bool(rec.get("strict_pass"))
    rec_id = rec.get("candidate_id") or stable_key(rec.get("params", {}))
    current_id = stable_key(CURRENT_LIVE_SHOCK)
    if rec_id == current_id:
        decision = "保留当前 live 冲击过滤参数。"
    elif strict:
        decision = "建议只进入 24 小时影子验证，不直接改真钱。"
    else:
        decision = "没有满足硬门槛的替换候选，保留当前 live 参数。"
    text = [
        "# top159 冲击过滤隔夜搜索结论",
        "",
        f"更新时间（北京时间）：{iso(bj_now())}",
        "",
        f"结论：{decision}",
        "",
        "## 推荐候选",
        "",
        "```json",
        json.dumps(rec, ensure_ascii=False, indent=2, default=str),
        "```",
        "",
        "## 当前 live 候选",
        "",
        "```json",
        json.dumps(current, ensure_ascii=False, indent=2, default=str),
        "```",
        "",
        "说明：本轮只写研究报告，没有修改 top159 live 配置、正式循环或真钱订单。",
    ]
    VERDICT.write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deadline-bj", default=os.environ.get("TOP159_SHOCK_OVERNIGHT_DEADLINE_BJ"))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("TOP159_SHOCK_OVERNIGHT_WORKERS", "4")))
    ap.add_argument("--chunk-size", type=int, default=int(os.environ.get("TOP159_SHOCK_OVERNIGHT_CHUNK_SIZE", "320")))
    ap.add_argument("--max-candidates", type=int, default=int(os.environ.get("TOP159_SHOCK_OVERNIGHT_MAX_CANDIDATES", "90000")))
    ap.add_argument("--status-only", action="store_true")
    args = ap.parse_args()

    REPORTS.mkdir(parents=True, exist_ok=True)
    if args.status_only:
        if CHECKPOINT.exists():
            print(CHECKPOINT.read_text(encoding="utf-8"))
        else:
            print("no checkpoint yet")
        return

    started = bj_now()
    deadline = parse_deadline(args.deadline_bj)
    if deadline <= started + timedelta(minutes=5):
        raise SystemExit(f"deadline too close: {iso(deadline)}")

    log(f"overnight shock hyperopt started; deadline={iso(deadline)} workers={args.workers}")
    log("preflight: research-only; no live config mutation; formal loop is not touched")

    enriched, _truth = ext.load_or_build_enriched()
    audit = ext.bug_audit(enriched)
    audit.update({
        "created_at_bj": iso(bj_now()),
        "research_only": True,
        "live_configs_touched": False,
        "deadline_bj": iso(deadline),
        "workers": args.workers,
        "current_live_shock_params": CURRENT_LIVE_SHOCK,
    })
    write_json(BUG_AUDIT, audit)
    log(f"bug audit written: {BUG_AUDIT}")

    all_params = build_params(args.max_candidates)
    done_ids = load_done_ids()
    params = [p for p in all_params if stable_key(p) not in done_ids]
    planned = len(all_params)
    log(f"candidate plan total={planned} already_done={len(done_ids)} remaining={len(params)}")

    # Load previous rows for leaderboard if resuming.
    all_rows: List[Dict[str, Any]] = []
    if CANDIDATES.exists():
        with CANDIDATES.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    all_rows.append(json.loads(line))
                except Exception:
                    continue

    write_checkpoint(started, deadline, planned, len(done_ids), all_rows, "running")

    chunks = list(chunked(params, max(1, args.chunk_size)))
    submitted = 0
    completed_ids = set(done_ids)
    last_checkpoint = time.time()

    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as pool:
        futures = []
        for chunk in chunks:
            if bj_now() >= deadline - timedelta(minutes=20):
                log("stopping submissions: near deadline")
                break
            futures.append(pool.submit(evaluate_chunk, chunk))
            submitted += len(chunk)
            # Keep the queue bounded to avoid memory blowup.
            if len(futures) >= args.workers * 3:
                for fut in as_completed(futures[: args.workers]):
                    rows = fut.result()
                    all_rows.extend(rows)
                    append_candidates(rows)
                    for r in rows:
                        if r.get("candidate_id"):
                            completed_ids.add(r["candidate_id"])
                futures = futures[args.workers:]
                if time.time() - last_checkpoint >= 15 * 60:
                    write_checkpoint(started, deadline, planned, len(completed_ids), all_rows, "running")
                    log(f"checkpoint: completed={len(completed_ids)} submitted={submitted} best={leaderboard(all_rows,1)[:1]}")
                    last_checkpoint = time.time()
        # Drain pending until deadline buffer.
        for fut in as_completed(futures):
            rows = fut.result()
            all_rows.extend(rows)
            append_candidates(rows)
            for r in rows:
                if r.get("candidate_id"):
                    completed_ids.add(r["candidate_id"])
            if time.time() - last_checkpoint >= 15 * 60:
                write_checkpoint(started, deadline, planned, len(completed_ids), all_rows, "running")
                log(f"checkpoint: completed={len(completed_ids)} submitted={submitted} best={leaderboard(all_rows,1)[:1]}")
                last_checkpoint = time.time()
            if bj_now() >= deadline - timedelta(minutes=12):
                log("deadline buffer reached; finalizing with completed candidates")
                break

    write_checkpoint(started, deadline, planned, len(completed_ids), all_rows, "finalizing")
    log("running final validation for top candidates")
    compare = final_compare(all_rows, top_n=30)
    write_json(COMPARE, compare)
    write_verdict(compare)
    write_checkpoint(started, deadline, planned, len(completed_ids), all_rows, "complete")
    log(f"complete; compare={COMPARE}; verdict={VERDICT}")


if __name__ == "__main__":
    main()
