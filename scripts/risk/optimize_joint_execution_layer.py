#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import itertools
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLYMARKET_DIR = PROJECT_ROOT / "polymarket"
REPORTS_DIR = PROJECT_ROOT / "reports"

ACTIVE_DEFAULT = POLYMARKET_DIR / "active_traders.json"
ACTIVE_70 = POLYMARKET_DIR / "active_traders_70.json"
CFG_DEFAULT = POLYMARKET_DIR / "trader_configs.json"
CFG_70 = POLYMARKET_DIR / "trader_configs_70.json"


@dataclass
class Constraints:
    max_net_drop: float
    min_mdd_improve: float
    max_wr_drop: float
    max_suppression: float


@dataclass
class TradeRow:
    ts: float
    pnl: float
    is_win: bool
    confidence: float
    token_price: Optional[float]
    limit_price: float
    base_prob: float
    base_kelly: float
    base_bet: float


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Joint optimize threshold+limit+kelly+bet for current simulation logs")
    ap.add_argument("--output", type=Path, default=REPORTS_DIR / "joint_execution_optimization_report.json")
    ap.add_argument("--profile", choices=["default", "70", "all"], default="all")
    ap.add_argument("--strict-max-net-drop", type=float, default=0.0)
    ap.add_argument("--strict-min-mdd-improve", type=float, default=0.10)
    ap.add_argument("--strict-max-wr-drop", type=float, default=0.01)
    ap.add_argument("--strict-max-suppression", type=float, default=0.30)
    ap.add_argument("--tier-b-max-net-drop", type=float, default=0.0)
    ap.add_argument("--tier-b-min-mdd-improve", type=float, default=0.0)
    ap.add_argument("--tier-b-max-wr-drop", type=float, default=0.015)
    ap.add_argument("--tier-b-max-suppression", type=float, default=0.40)
    ap.add_argument("--threshold-deltas", type=str, default="-0.01,-0.005,0,0.005,0.01")
    ap.add_argument("--limit-shifts", type=str, default="-0.02,-0.01,0,0.01,0.02")
    ap.add_argument("--kelly-scales", type=str, default="0.85,0.95,1.0,1.05,1.15")
    ap.add_argument("--bet-scales", type=str, default="0.85,0.95,1.0,1.05,1.15")
    return ap.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_float(v: Any) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _parse_grid(raw: str) -> List[float]:
    out: List[float] = []
    for part in str(raw).split(","):
        v = _to_float(part)
        if v is None:
            continue
        out.append(float(v))
    if not out:
        return [0.0]
    uniq = sorted(set(out))
    return uniq


def _parse_ts(v: Any) -> float:
    if not isinstance(v, str) or not v:
        return 0.0
    s = v.strip().replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return 0.0


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _ladder_base_limit(v: Any) -> Optional[float]:
    if not isinstance(v, str) or not v.strip():
        return None
    vals = [_to_float(p) for p in v.split(",")]
    nums = [float(n) for n in vals if n is not None]
    if not nums:
        return None
    return max(nums)


def _active_names(path: Path) -> set[str]:
    if not path.exists():
        return set()
    data = _load_json(path)
    names = data.get("active_traders")
    if not isinstance(names, list):
        names = data.get("traderNames")
    if not isinstance(names, list):
        return set()
    return {str(x).strip() for x in names if str(x).strip()}


def _config_by_name(path: Path) -> Dict[str, Dict[str, Any]]:
    data = _load_json(path)
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(data, list):
        return out
    for row in data:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        out[name] = row
    return out


def _load_trades_for_profile(profile: str) -> Tuple[List[TradeRow], Dict[str, Any]]:
    if profile == "70":
        names = _active_names(ACTIVE_70)
        cfg_map = _config_by_name(CFG_70)
    else:
        names = _active_names(ACTIVE_DEFAULT)
        cfg_map = _config_by_name(CFG_DEFAULT)

    rows: List[TradeRow] = []
    meta = {
        "active_names": len(names),
        "traders_with_logs": 0,
        "loaded_trades": 0,
    }
    for name in sorted(names):
        cfg = cfg_map.get(name)
        if not isinstance(cfg, dict):
            continue
        logs_dir = str(cfg.get("logsDir") or "").strip()
        if not logs_dir:
            continue
        trades_path = POLYMARKET_DIR / logs_dir / "prediction_trades.simulation.json"
        if not trades_path.exists():
            continue
        try:
            entries = _load_json(trades_path)
        except Exception:
            continue
        if not isinstance(entries, list):
            continue
        meta["traders_with_logs"] += 1

        base_prob = float(_to_float(cfg.get("probThreshold")) or 0.55)
        base_limit = _to_float(cfg.get("limitPrice"))
        if base_limit is None:
            base_limit = _ladder_base_limit(cfg.get("limitPriceLadder"))
        if base_limit is None:
            base_limit = 0.50
        base_kelly = float(_to_float(cfg.get("kellyFrac")) or 1.0)
        base_bet = float(_to_float(cfg.get("betPctNormal")) or 1.0)

        for e in entries:
            if not isinstance(e, dict):
                continue
            if str(e.get("mode") or "simulation").lower() != "simulation":
                continue
            if str(e.get("status") or "").lower() != "executed":
                continue
            pnl = _to_float(e.get("pnl"))
            conf = _to_float(e.get("confidence"))
            if pnl is None or conf is None:
                continue
            result = str(e.get("result") or "").lower()
            if result not in {"win", "lose"}:
                continue
            token_price = _to_float(e.get("tokenPrice"))
            limit_cfg = _to_float(e.get("limitPriceConfigured"))
            if limit_cfg is None:
                limit_cfg = base_limit
            ts = _parse_ts(e.get("settledAt")) or _parse_ts(e.get("timestamp"))
            rows.append(
                TradeRow(
                    ts=float(ts),
                    pnl=float(pnl),
                    is_win=(result == "win"),
                    confidence=float(conf),
                    token_price=float(token_price) if token_price is not None else None,
                    limit_price=float(limit_cfg),
                    base_prob=base_prob,
                    base_kelly=base_kelly if base_kelly > 1e-9 else 1.0,
                    base_bet=base_bet if base_bet > 1e-9 else 1.0,
                )
            )
    meta["loaded_trades"] = len(rows)
    return rows, meta


def _max_drawdown_abs(pnls: Iterable[Tuple[float, float]]) -> float:
    seq = sorted(pnls, key=lambda x: x[0])
    if not seq:
        return 0.0
    eq = 0.0
    peak = 0.0
    mdd = 0.0
    for _, pnl in seq:
        eq += float(pnl)
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > mdd:
            mdd = dd
    return max(0.0, float(mdd))


def _baseline_metrics(rows: List[TradeRow]) -> Dict[str, float]:
    if not rows:
        return {"net_pnl": 0.0, "mdd": 0.0, "wr": 0.0, "trades": 0.0}
    wins = sum(1 for r in rows if r.is_win)
    total = len(rows)
    net = float(sum(r.pnl for r in rows))
    mdd = _max_drawdown_abs((r.ts, r.pnl) for r in rows)
    wr = float(wins / total) if total > 0 else 0.0
    return {"net_pnl": net, "mdd": mdd, "wr": wr, "trades": float(total)}


def _candidate_metrics(
    rows: List[TradeRow],
    base: Dict[str, float],
    threshold_delta: float,
    limit_shift: float,
    kelly_scale: float,
    bet_scale: float,
) -> Dict[str, float]:
    taken: List[Tuple[float, float]] = []
    wins = 0
    total = 0
    size_scale = float(max(0.1, kelly_scale) * max(0.1, bet_scale))
    for r in rows:
        th = _clamp(r.base_prob + threshold_delta, 0.45, 0.90)
        lim = _clamp(r.limit_price + limit_shift, 0.30, 0.99)
        if r.confidence < th:
            continue
        if r.token_price is not None and r.token_price > lim:
            continue
        total += 1
        if r.is_win:
            wins += 1
        taken.append((r.ts, r.pnl * size_scale))

    cand_net = float(sum(p for _, p in taken))
    cand_mdd = _max_drawdown_abs(taken)
    cand_wr = float(wins / total) if total > 0 else 0.0
    base_net = float(base.get("net_pnl") or 0.0)
    base_mdd = float(base.get("mdd") or 0.0)
    base_wr = float(base.get("wr") or 0.0)
    base_trades = int(base.get("trades") or 0)

    if abs(base_net) < 1e-9:
        pnl_delta_ratio = 0.0 if abs(cand_net) < 1e-9 else -1.0
    else:
        pnl_delta_ratio = (cand_net - base_net) / abs(base_net)
    if base_mdd <= 1e-9:
        mdd_improve = 0.0
    else:
        mdd_improve = (base_mdd - cand_mdd) / base_mdd
    wr_drop = base_wr - cand_wr
    suppression = 1.0 - (float(total) / float(base_trades)) if base_trades > 0 else 0.0
    return {
        "net_pnl": cand_net,
        "mdd": cand_mdd,
        "wr": cand_wr,
        "trades": float(total),
        "pnl_delta_ratio": float(pnl_delta_ratio),
        "mdd_improve": float(mdd_improve),
        "wr_drop": float(wr_drop),
        "suppression": float(max(0.0, min(1.0, suppression))),
    }


def _check_constraints(m: Dict[str, float], c: Constraints) -> Dict[str, bool]:
    return {
        "net_drop_ok": float(m.get("pnl_delta_ratio", 0.0)) >= -float(c.max_net_drop),
        "mdd_improve_ok": float(m.get("mdd_improve", 0.0)) >= float(c.min_mdd_improve),
        "wr_drop_ok": float(m.get("wr_drop", 0.0)) <= float(c.max_wr_drop),
        "suppression_ok": float(m.get("suppression", 0.0)) <= float(c.max_suppression),
    }


def _violations_from_checks(checks: Dict[str, bool]) -> List[str]:
    out: List[str] = []
    if not checks.get("net_drop_ok", True):
        out.append("net_drop_gt_limit")
    if not checks.get("mdd_improve_ok", True):
        out.append("mdd_improve_lt_limit")
    if not checks.get("wr_drop_ok", True):
        out.append("wr_drop_gt_limit")
    if not checks.get("suppression_ok", True):
        out.append("suppression_gt_limit")
    return out


def _score(m: Dict[str, float]) -> float:
    net_keep = max(0.0, 1.0 + float(m.get("pnl_delta_ratio", 0.0)))
    mdd_improve = max(0.0, float(m.get("mdd_improve", 0.0)))
    suppression = float(m.get("suppression", 0.0))
    return 0.6 * net_keep + 0.25 * mdd_improve + 0.15 * (1.0 - min(max(suppression, 0.0), 1.0))


def _optimize_profile(
    rows: List[TradeRow],
    strict: Constraints,
    tier_b: Constraints,
    threshold_deltas: List[float],
    limit_shifts: List[float],
    kelly_scales: List[float],
    bet_scales: List[float],
) -> Dict[str, Any]:
    baseline = _baseline_metrics(rows)
    if not rows:
        return {
            "best": {
                "search_pass": "pass1",
                "candidate": {},
                "metrics": {
                    "net_pnl": 0.0,
                    "mdd": 0.0,
                    "wr": 0.0,
                    "trades": 0.0,
                    "pnl_delta_ratio": -1.0,
                    "mdd_improve": 0.0,
                    "wr_drop": 0.0,
                    "suppression": 1.0,
                },
                "checks": {},
                "checks_b": {},
                "constraint_violations": ["missing_cluster"],
                "admission_tier": "C",
                "final_is_feasible": False,
                "fallback_score": 0.0,
                "sample_count": 0,
                "baseline_metrics": baseline,
            },
            "candidates_tested": 0,
            "baseline": baseline,
        }

    best_pack: Optional[Dict[str, Any]] = None
    tested = 0
    for td, ls, ks, bs in itertools.product(threshold_deltas, limit_shifts, kelly_scales, bet_scales):
        tested += 1
        cm = _candidate_metrics(rows, baseline, td, ls, ks, bs)
        checks = _check_constraints(cm, strict)
        checks_b = _check_constraints(cm, tier_b)
        tier = "C"
        if all(checks.values()):
            tier = "A"
        elif all(checks_b.values()):
            tier = "B"
        violations = _violations_from_checks(checks)
        pack = {
            "candidate": {
                "thresholdDelta": float(td),
                "limitShift": float(ls),
                "kellyScale": float(ks),
                "betScale": float(bs),
            },
            "metrics": cm,
            "checks": checks,
            "checks_b": checks_b,
            "admission_tier": tier,
            "final_is_feasible": tier in {"A", "B"},
            "constraint_violations": violations,
            "fallback_score": _score(cm),
        }
        if best_pack is None:
            best_pack = pack
            continue
        pri = {"A": 3, "B": 2, "C": 1}
        old_tier = str(best_pack.get("admission_tier") or "C")
        if pri[tier] > pri[old_tier]:
            best_pack = pack
            continue
        if pri[tier] == pri[old_tier]:
            if float(pack["fallback_score"]) > float(best_pack.get("fallback_score") or 0.0):
                best_pack = pack

    assert best_pack is not None
    out_best = {
        "search_pass": "pass1",
        "candidate": best_pack["candidate"],
        "metrics": best_pack["metrics"],
        "checks": best_pack["checks"],
        "checks_b": best_pack["checks_b"],
        "constraint_violations": best_pack["constraint_violations"],
        "admission_tier": best_pack["admission_tier"],
        "final_is_feasible": bool(best_pack["final_is_feasible"]),
        "fallback_score": float(best_pack["fallback_score"]),
        "sample_count": len(rows),
        "baseline_metrics": baseline,
    }
    return {
        "best": out_best,
        "candidates_tested": tested,
        "baseline": baseline,
    }


def main() -> int:
    args = parse_args()
    strict = Constraints(
        max_net_drop=float(args.strict_max_net_drop),
        min_mdd_improve=float(args.strict_min_mdd_improve),
        max_wr_drop=float(args.strict_max_wr_drop),
        max_suppression=float(args.strict_max_suppression),
    )
    tier_b = Constraints(
        max_net_drop=float(args.tier_b_max_net_drop),
        min_mdd_improve=float(args.tier_b_min_mdd_improve),
        max_wr_drop=float(args.tier_b_max_wr_drop),
        max_suppression=float(args.tier_b_max_suppression),
    )
    threshold_deltas = _parse_grid(args.threshold_deltas)
    limit_shifts = _parse_grid(args.limit_shifts)
    kelly_scales = _parse_grid(args.kelly_scales)
    bet_scales = _parse_grid(args.bet_scales)

    profiles = ["default", "70"] if args.profile == "all" else [args.profile]
    clusters: Dict[str, Any] = {}
    profile_meta: Dict[str, Any] = {}
    for profile in profiles:
        rows, meta = _load_trades_for_profile(profile)
        profile_meta[profile] = meta
        clusters[profile] = _optimize_profile(
            rows=rows,
            strict=strict,
            tier_b=tier_b,
            threshold_deltas=threshold_deltas,
            limit_shifts=limit_shifts,
            kelly_scales=kelly_scales,
            bet_scales=bet_scales,
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": "joint_execution_v1",
        "anti_overfit": {
            "cluster_only": True,
            "cell_specific_params_forbidden": True,
            "cluster_count": len(clusters),
        },
        "constraints_snapshot": {
            "strict": strict.__dict__,
            "tier_b": tier_b.__dict__,
        },
        "search_space": {
            "threshold_deltas": threshold_deltas,
            "limit_shifts": limit_shifts,
            "kelly_scales": kelly_scales,
            "bet_scales": bet_scales,
        },
        "profile_meta": profile_meta,
        "clusters": clusters,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] wrote {args.output}")
    for profile, payload in clusters.items():
        best = payload.get("best", {})
        tier = best.get("admission_tier")
        m = best.get("metrics", {})
        print(
            f"[{profile}] tier={tier} net={float(m.get('net_pnl', 0.0)):.2f} "
            f"base={float(best.get('baseline_metrics', {}).get('net_pnl', 0.0)):.2f} "
            f"mddImprove={float(m.get('mdd_improve', 0.0)):.3f} "
            f"supp={float(m.get('suppression', 0.0)):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

