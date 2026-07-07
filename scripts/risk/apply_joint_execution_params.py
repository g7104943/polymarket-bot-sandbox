#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLYMARKET_DIR = PROJECT_ROOT / "polymarket"
CFG_DEFAULT = POLYMARKET_DIR / "trader_configs.json"
CFG_70 = POLYMARKET_DIR / "trader_configs_70.json"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "joint_execution_optimization_report.json"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Apply joint execution optimization params into trader configs")
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--mode", choices=["auto", "enforce", "shadow", "off"], default="auto")
    ap.add_argument("--target-profile", choices=["default", "70", "all"], default="all")
    return ap.parse_args()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


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


def _map_mode(auto_mode: str, forced_mode: str) -> str:
    if forced_mode != "auto":
        return forced_mode
    if auto_mode in {"A", "B"}:
        return "enforce"
    return "off"


def _base_snapshot(cfg: Dict[str, Any]) -> Dict[str, Any]:
    base = cfg.get("jointOptimizationBaseParams")
    if isinstance(base, dict):
        return dict(base)
    return {
        "probThreshold": cfg.get("probThreshold"),
        "limitPrice": cfg.get("limitPrice"),
        "limitPriceLadder": cfg.get("limitPriceLadder"),
        "kellyFrac": cfg.get("kellyFrac"),
        "betPctNormal": cfg.get("betPctNormal"),
        "betPctConservative": cfg.get("betPctConservative"),
    }


def _shift_ladder(raw: Any, shift: float) -> Any:
    if not isinstance(raw, str) or not raw.strip():
        return raw
    out = []
    for part in raw.split(","):
        v = _to_float(part)
        if v is None:
            continue
        out.append(f"{_clamp(v + shift, 0.30, 0.99):.2f}")
    return ",".join(out) if out else raw


def _apply_one(cfg: Dict[str, Any], mode: str, candidate: Dict[str, Any], tier: str) -> None:
    base = _base_snapshot(cfg)
    cfg["jointOptimizationBaseParams"] = base
    cfg["jointOptimizationMode"] = mode
    cfg["jointOptimizationAdmissionTier"] = tier
    cfg["jointOptimizationAppliedAt"] = datetime.now(timezone.utc).isoformat()

    if mode != "enforce":
        for k in ("probThreshold", "limitPrice", "limitPriceLadder", "kellyFrac", "betPctNormal", "betPctConservative"):
            if k in base:
                cfg[k] = base.get(k)
        cfg["jointOptimizationParams"] = {
            "mode": mode,
            "admission_tier": tier,
        }
        return

    td = float(_to_float(candidate.get("thresholdDelta")) or 0.0)
    ls = float(_to_float(candidate.get("limitShift")) or 0.0)
    ks = float(_to_float(candidate.get("kellyScale")) or 1.0)
    bs = float(_to_float(candidate.get("betScale")) or 1.0)

    b_prob = float(_to_float(base.get("probThreshold")) or _to_float(cfg.get("probThreshold")) or 0.55)
    cfg["probThreshold"] = round(_clamp(b_prob + td, 0.45, 0.90), 5)

    b_limit = _to_float(base.get("limitPrice"))
    if b_limit is not None:
        cfg["limitPrice"] = round(_clamp(float(b_limit) + ls, 0.30, 0.99), 2)
    else:
        cfg["limitPriceLadder"] = _shift_ladder(base.get("limitPriceLadder"), ls)

    b_kelly = float(_to_float(base.get("kellyFrac")) or _to_float(cfg.get("kellyFrac")) or 1.0)
    b_bet_n = float(_to_float(base.get("betPctNormal")) or _to_float(cfg.get("betPctNormal")) or 0.05)
    b_bet_c = float(_to_float(base.get("betPctConservative")) or _to_float(cfg.get("betPctConservative")) or 0.03)
    cfg["kellyFrac"] = round(_clamp(b_kelly * ks, 0.01, 3.0), 5)
    cfg["betPctNormal"] = round(_clamp(b_bet_n * bs, 0.001, 1.0), 5)
    cfg["betPctConservative"] = round(_clamp(b_bet_c * bs, 0.001, 1.0), 5)
    cfg["jointOptimizationParams"] = {
        "mode": mode,
        "admission_tier": tier,
        "candidate": {
            "thresholdDelta": td,
            "limitShift": ls,
            "kellyScale": ks,
            "betScale": bs,
        },
    }


def _load_profile_plan(report: Dict[str, Any], profile: str, forced_mode: str) -> Dict[str, Any]:
    clusters = report.get("clusters")
    if not isinstance(clusters, dict):
        return {"mode": "off", "tier": "C", "candidate": {}}
    cluster = clusters.get(profile)
    if not isinstance(cluster, dict):
        return {"mode": "off", "tier": "C", "candidate": {}}
    best = cluster.get("best")
    if not isinstance(best, dict):
        return {"mode": "off", "tier": "C", "candidate": {}}
    tier = str(best.get("admission_tier") or "C").upper()
    mode = _map_mode(tier, forced_mode)
    cand = best.get("candidate") if isinstance(best.get("candidate"), dict) else {}
    return {"mode": mode, "tier": tier, "candidate": cand}


def _apply_profile(config_path: Path, profile: str, profile_plan: Dict[str, Any]) -> Dict[str, Any]:
    data = _load_json(config_path)
    if not isinstance(data, list):
        raise RuntimeError(f"invalid config list: {config_path}")
    changed = 0
    for cfg in data:
        if not isinstance(cfg, dict):
            continue
        before = json.dumps(cfg, ensure_ascii=False, sort_keys=True)
        _apply_one(
            cfg=cfg,
            mode=str(profile_plan.get("mode") or "off"),
            candidate=dict(profile_plan.get("candidate") or {}),
            tier=str(profile_plan.get("tier") or "C"),
        )
        after = json.dumps(cfg, ensure_ascii=False, sort_keys=True)
        if after != before:
            changed += 1
    config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "profile": profile,
        "mode": profile_plan.get("mode"),
        "tier": profile_plan.get("tier"),
        "changed_traders": changed,
        "config": str(config_path),
    }


def main() -> int:
    args = parse_args()
    if not args.report.exists():
        raise SystemExit(f"missing report: {args.report}")
    report = _load_json(args.report)
    if not isinstance(report, dict):
        raise SystemExit("invalid report format")

    profiles = ["default", "70"] if args.target_profile == "all" else [args.target_profile]
    out = {
        "report": str(args.report),
        "mode": args.mode,
        "target_profile": args.target_profile,
        "results": [],
    }
    for profile in profiles:
        plan = _load_profile_plan(report, profile, args.mode)
        cfg_path = CFG_70 if profile == "70" else CFG_DEFAULT
        res = _apply_profile(cfg_path, profile, plan)
        out["results"].append(res)
        print(
            f"[{profile}] mode={res['mode']} tier={res['tier']} "
            f"changed={res['changed_traders']} config={cfg_path}"
        )

    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
