#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLYMARKET_DIR = PROJECT_ROOT / "polymarket"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "shock_risk_v2_tuning_report.json"
CFG_DEFAULT = POLYMARKET_DIR / "trader_configs.json"
CFG_70 = POLYMARKET_DIR / "trader_configs_70.json"
ACTIVE_DEFAULT = POLYMARKET_DIR / "active_traders.json"
ACTIVE_70 = POLYMARKET_DIR / "active_traders_70.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_active(path: Path) -> List[str]:
    obj = _load_json(path)
    if isinstance(obj, list):
        return [str(x) for x in obj]
    if isinstance(obj, dict):
        for key in ("active_traders", "traderNames"):
            v = obj.get(key)
            if isinstance(v, list):
                return [str(x) for x in v]
    return []


def _parse_target_traders(raw: str) -> Tuple[Set[str], Dict[str, Set[str]]]:
    plain: Set[str] = set()
    scoped: Dict[str, Set[str]] = {"default": set(), "70": set()}
    for tok in str(raw or "").split(","):
        token = tok.strip()
        if not token:
            continue
        if "/" in token:
            prof, name = token.split("/", 1)
            profile = prof.strip().lower()
            trader = name.strip()
            if profile in scoped and trader:
                scoped[profile].add(trader)
                continue
        plain.add(token)
    return plain, scoped


def _targets_from_decision(path: Path, profiles: Set[str]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {"default": set(), "70": set()}
    payload = _load_json(path)
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        profile = str(item.get("profile") or "").lower()
        trader = str(item.get("trader") or "")
        if profile not in profiles or not trader:
            continue
        layers = item.get("layers")
        if not isinstance(layers, dict):
            continue
        state = layers.get("shock")
        if isinstance(state, dict) and bool(state.get("ready")):
            out[profile].add(trader)
    return out


def _targets_from_recent(path: Path, profiles: Set[str]) -> Dict[str, Set[str]]:
    out: Dict[str, Set[str]] = {"default": set(), "70": set()}
    payload = _load_json(path)
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        profile = str(item.get("profile") or "").lower()
        trader = str(item.get("trader") or "")
        if profile not in profiles or not trader:
            continue
        layers = item.get("layers")
        if not isinstance(layers, dict):
            continue
        layer_payload = layers.get("shock")
        if not isinstance(layer_payload, dict):
            continue
        clusters = layer_payload.get("clusters")
        if not isinstance(clusters, dict):
            continue
        ok_cells = 0
        all_positive = True
        for cval in clusters.values():
            if not isinstance(cval, dict) or str(cval.get("status")) != "ok":
                continue
            ok_cells += 1
            recent = cval.get("recent")
            if not isinstance(recent, dict) or not bool(recent.get("recent_positive_gain")):
                all_positive = False
                break
        if ok_cells > 0 and all_positive:
            out[profile].add(trader)
    return out


def _filter_active_names(
    active_names: List[str],
    profile: str,
    plain_targets: Set[str],
    scoped_targets: Dict[str, Set[str]],
    decision_targets: Dict[str, Set[str]] | None,
    recent_targets: Dict[str, Set[str]] | None,
) -> List[str]:
    selected: List[str] = []
    for name in active_names:
        if plain_targets or scoped_targets.get(profile):
            if name not in plain_targets and name not in scoped_targets.get(profile, set()):
                continue
        if decision_targets is not None and name not in decision_targets.get(profile, set()):
            continue
        if recent_targets is not None and name not in recent_targets.get(profile, set()):
            continue
        selected.append(name)
    return selected


def _extract_cluster_params(
    report: Dict[str, Any],
    profiles: Set[str],
) -> Tuple[Dict[str, Dict[str, float]], bool, Dict[str, List[str]]]:
    clusters = report.get("clusters") if isinstance(report, dict) else None
    if not isinstance(clusters, dict):
        raise RuntimeError("report.clusters missing")
    needed: List[str] = []
    if "default" in profiles:
        needed += ["default_BTC", "default_ETH"]
    if "70" in profiles:
        needed += ["70_BTC", "70_ETH"]
    out: Dict[str, Dict[str, float]] = {}
    violations: Dict[str, List[str]] = {}
    has_violation = False
    for k in needed:
        node = clusters.get(k)
        if not isinstance(node, dict):
            raise RuntimeError(f"cluster missing: {k}")
        best = node.get("best")
        if not isinstance(best, dict):
            raise RuntimeError(f"cluster best missing: {k}")
        params = best.get("params")
        if not isinstance(params, dict):
            raise RuntimeError(f"cluster params missing: {k}")
        out[k] = {str(pk): float(pv) for pk, pv in params.items()}
        viol = best.get("constraint_violations")
        vlist = [str(x) for x in viol] if isinstance(viol, list) else []
        violations[k] = vlist
        if vlist:
            has_violation = True
    return out, has_violation, violations


def _resolve_mode(requested_mode: str, cluster_violations: List[str]) -> str:
    if requested_mode == "enforce":
        return "enforce"
    if requested_mode == "shadow":
        return "shadow"
    return "shadow" if cluster_violations else "enforce"


def _apply_to_config(
    cfg_path: Path,
    active_names: List[str],
    params_btc: Dict[str, float],
    params_eth: Dict[str, float],
    shock_mode_by_symbol: Dict[str, str],
    combo_mode_by_symbol: Dict[str, str],
    combo_drawdown2h: float,
    combo_hold_minutes: int,
    dry_run: bool,
) -> Dict[str, Any]:
    original_text = cfg_path.read_text(encoding="utf-8")
    arr = json.loads(original_text)
    if not isinstance(arr, list):
        raise RuntimeError(f"config must be array: {cfg_path}")

    by_name = {str(x.get("name")): x for x in arr if isinstance(x, dict)}
    missing = [n for n in active_names if n not in by_name]

    touched = 0
    cells = 0
    for name in active_names:
        cfg = by_name.get(name)
        if not cfg:
            continue
        allowed = [str(x).strip().upper() for x in str(cfg.get("allowedMarkets") or "").split(",") if str(x).strip()]
        has_btc = "BTC" in allowed
        has_eth = "ETH" in allowed

        cfg["shockRiskEngine"] = "v2"
        cfg["shockRiskStatsMode"] = "route"
        cfg["shockRiskCheckSeconds"] = int(cfg.get("shockRiskCheckSeconds") or 60)
        shock_modes: List[str] = []

        d = cfg.get("shockRiskBySymbol")
        if not isinstance(d, dict):
            d = {}
        shock_mode_map = cfg.get("shockRiskModeBySymbol")
        if not isinstance(shock_mode_map, dict):
            shock_mode_map = {}
        if has_btc:
            d["BTC"] = dict(params_btc)
            m = str(shock_mode_by_symbol.get("BTC", "shadow"))
            shock_mode_map["BTC"] = m
            shock_modes.append(m)
            cells += 1
        if has_eth:
            d["ETH"] = dict(params_eth)
            m = str(shock_mode_by_symbol.get("ETH", "shadow"))
            shock_mode_map["ETH"] = m
            shock_modes.append(m)
            cells += 1
        cfg["shockRiskBySymbol"] = d
        cfg["shockRiskModeBySymbol"] = shock_mode_map
        cfg["shockRiskMode"] = shock_modes[0] if shock_modes and len(set(shock_modes)) == 1 else "shadow"

        cfg["comboPauseEnabled"] = True
        cfg["comboPauseStatsMode"] = "route"
        cfg["comboPauseDrawdown2h"] = float(combo_drawdown2h)
        cfg["comboPauseHoldMinutes"] = int(combo_hold_minutes)
        cfg["comboPauseCheckSeconds"] = int(cfg.get("comboPauseCheckSeconds") or 60)
        combo_modes: List[str] = []
        combo_mode_map = cfg.get("comboPauseModeBySymbol")
        if not isinstance(combo_mode_map, dict):
            combo_mode_map = {}
        if has_btc:
            m = str(combo_mode_by_symbol.get("BTC", "shadow"))
            combo_mode_map["BTC"] = m
            combo_modes.append(m)
        if has_eth:
            m = str(combo_mode_by_symbol.get("ETH", "shadow"))
            combo_mode_map["ETH"] = m
            combo_modes.append(m)
        cfg["comboPauseModeBySymbol"] = combo_mode_map
        cfg["comboPauseMode"] = combo_modes[0] if combo_modes and len(set(combo_modes)) == 1 else "shadow"
        touched += 1

    backup = cfg_path.with_suffix(cfg_path.suffix + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    if not dry_run:
        backup.write_text(original_text, encoding="utf-8")
        cfg_path.write_text(json.dumps(arr, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "config": str(cfg_path),
        "backup": str(backup),
        "active": len(active_names),
        "touched_traders": touched,
        "cells": cells,
        "missing_traders": missing,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Apply SHOCK risk V2 params to active trader configs")
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--shock-mode", choices=["auto", "enforce", "shadow"], default="enforce")
    ap.add_argument("--combo-mode", choices=["auto", "enforce", "shadow"], default="enforce")
    ap.add_argument("--target-profile", choices=["default", "70", "all"], default="all")
    ap.add_argument("--target-traders", type=str, default="", help="comma list: trader or profile/trader")
    ap.add_argument("--decision-json", type=Path, default=None, help="top20 decision json; only apply ready traders")
    ap.add_argument("--recent-gain-json", type=Path, default=None, help="recent gain json; only apply all-positive traders")
    ap.add_argument("--combo-drawdown2h", type=float, default=0.12)
    ap.add_argument("--combo-hold-minutes", type=int, default=120)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    report = _load_json(args.report)
    profiles = {"default", "70"} if args.target_profile == "all" else {args.target_profile}
    params, has_violation, violations = _extract_cluster_params(report, profiles)
    default_shock_mode_by_symbol = {
        "BTC": _resolve_mode(args.shock_mode, violations.get("default_BTC", [])),
        "ETH": _resolve_mode(args.shock_mode, violations.get("default_ETH", [])),
    }
    profile70_shock_mode_by_symbol = {
        "BTC": _resolve_mode(args.shock_mode, violations.get("70_BTC", [])),
        "ETH": _resolve_mode(args.shock_mode, violations.get("70_ETH", [])),
    }
    if args.combo_mode == "auto":
        default_combo_mode_by_symbol = dict(default_shock_mode_by_symbol)
        profile70_combo_mode_by_symbol = dict(profile70_shock_mode_by_symbol)
    else:
        default_combo_mode_by_symbol = {"BTC": args.combo_mode, "ETH": args.combo_mode}
        profile70_combo_mode_by_symbol = {"BTC": args.combo_mode, "ETH": args.combo_mode}

    active_default_all = _parse_active(ACTIVE_DEFAULT)
    active_70_all = _parse_active(ACTIVE_70)
    plain_targets, scoped_targets = _parse_target_traders(args.target_traders)
    decision_targets = _targets_from_decision(args.decision_json, profiles) if args.decision_json else None
    recent_targets = _targets_from_recent(args.recent_gain_json, profiles) if args.recent_gain_json else None
    active_default = _filter_active_names(
        active_names=active_default_all,
        profile="default",
        plain_targets=plain_targets,
        scoped_targets=scoped_targets,
        decision_targets=decision_targets,
        recent_targets=recent_targets,
    )
    active_70 = _filter_active_names(
        active_names=active_70_all,
        profile="70",
        plain_targets=plain_targets,
        scoped_targets=scoped_targets,
        decision_targets=decision_targets,
        recent_targets=recent_targets,
    )

    r1: Dict[str, Any] = {"touched_traders": 0, "cells": 0, "missing_traders": [], "backup": ""}
    r2: Dict[str, Any] = {"touched_traders": 0, "cells": 0, "missing_traders": [], "backup": ""}

    if args.target_profile in ("default", "all"):
        r1 = _apply_to_config(
            cfg_path=CFG_DEFAULT,
            active_names=active_default,
            params_btc=params["default_BTC"],
            params_eth=params["default_ETH"],
            shock_mode_by_symbol=default_shock_mode_by_symbol,
            combo_mode_by_symbol=default_combo_mode_by_symbol,
            combo_drawdown2h=float(args.combo_drawdown2h),
            combo_hold_minutes=int(args.combo_hold_minutes),
            dry_run=args.dry_run,
        )
    if args.target_profile in ("70", "all"):
        r2 = _apply_to_config(
            cfg_path=CFG_70,
            active_names=active_70,
            params_btc=params["70_BTC"],
            params_eth=params["70_ETH"],
            shock_mode_by_symbol=profile70_shock_mode_by_symbol,
            combo_mode_by_symbol=profile70_combo_mode_by_symbol,
            combo_drawdown2h=float(args.combo_drawdown2h),
            combo_hold_minutes=int(args.combo_hold_minutes),
            dry_run=args.dry_run,
        )

    total_cells = int(r1["cells"] + r2["cells"])
    print("[OK] applied shock risk v2 params")
    print(
        "[INFO] mode(auto by cluster) "
        f"default_shock={default_shock_mode_by_symbol} profile70_shock={profile70_shock_mode_by_symbol} "
        f"default_combo={default_combo_mode_by_symbol} profile70_combo={profile70_combo_mode_by_symbol} "
        f"(any_violation={has_violation})"
    )
    if args.target_traders:
        print(f"[INFO] target_traders_filter={args.target_traders}")
    if args.decision_json:
        print(f"[INFO] decision_filter={args.decision_json}")
    if args.recent_gain_json:
        print(f"[INFO] recent_gain_filter={args.recent_gain_json}")
    if args.target_profile in ("default", "all"):
        print(f"[INFO] default: touched={r1['touched_traders']} cells={r1['cells']} missing={len(r1['missing_traders'])}")
    else:
        print("[INFO] default: skipped")
    if args.target_profile in ("70", "all"):
        print(f"[INFO] 70:      touched={r2['touched_traders']} cells={r2['cells']} missing={len(r2['missing_traders'])}")
    else:
        print("[INFO] 70:      skipped")
    print(f"[INFO] total BTC/ETH cells={total_cells}")
    if r1["missing_traders"] or r2["missing_traders"]:
        print("[WARN] missing active traders found in configs")
    if has_violation:
        print("[WARN] constraint violations detected in report:")
        for k, v in violations.items():
            if v:
                print(f"  - {k}: {v}")
    if not args.dry_run:
        backups = [b for b in (str(r1.get("backup") or ""), str(r2.get("backup") or "")) if b]
        if backups:
            print("[INFO] backups:")
            for b in backups:
                print(f"  - {b}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
