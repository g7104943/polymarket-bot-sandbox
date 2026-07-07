#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLYMARKET_DIR = PROJECT_ROOT / "polymarket"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "combo_pause_v1_tuning_report.json"
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
            vals = obj.get(key)
            if isinstance(vals, list):
                return [str(x) for x in vals]
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


def _targets_from_decision(path: Path, layer: str, profiles: Set[str]) -> Dict[str, Set[str]]:
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
        state = layers.get(layer)
        if isinstance(state, dict) and bool(state.get("ready")):
            out[profile].add(trader)
    return out


def _targets_from_recent(path: Path, layer: str, profiles: Set[str]) -> Dict[str, Set[str]]:
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
        layer_payload = layers.get(layer)
        if not isinstance(layer_payload, dict):
            continue
        clusters = layer_payload.get("clusters")
        if not isinstance(clusters, dict):
            continue
        if layer == "combo":
            for cval in clusters.values():
                if not isinstance(cval, dict) or str(cval.get("status")) != "ok":
                    continue
                if not bool(cval.get("ready_in_top20")):
                    continue
                recent = cval.get("recent")
                if isinstance(recent, dict) and bool(recent.get("recent_positive_gain")):
                    out[profile].add(trader)
                    break
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


def _cluster_keys_from_recent(path: Path, layer: str, profiles: Set[str]) -> Dict[str, Dict[str, Set[str]]]:
    out: Dict[str, Dict[str, Set[str]]] = {"default": {}, "70": {}}
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
        layer_payload = layers.get(layer)
        if not isinstance(layer_payload, dict):
            continue
        clusters = layer_payload.get("clusters")
        if not isinstance(clusters, dict):
            continue
        selected: Set[str] = set()
        for cluster_key, cval in clusters.items():
            if not isinstance(cval, dict) or str(cval.get("status")) != "ok":
                continue
            if not bool(cval.get("ready_in_top20")):
                continue
            recent = cval.get("recent")
            if not isinstance(recent, dict) or not bool(recent.get("recent_positive_gain")):
                continue
            parts = str(cluster_key).split("_")
            if len(parts) != 3:
                continue
            _, symbol, direction = parts
            selected.add(f"{symbol.upper()}_{direction.upper()}")
        if selected:
            out[profile][trader] = selected
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


def _resolve_mode(requested_mode: str, violations: List[str]) -> str:
    if requested_mode == "enforce":
        return "enforce"
    if requested_mode == "shadow":
        return "shadow"
    return "shadow" if violations else "enforce"


def _extract_cluster_params(
    report: Dict[str, Any],
    profiles: Set[str],
) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], bool]:
    clusters = report.get("clusters") if isinstance(report, dict) else None
    if not isinstance(clusters, dict):
        raise RuntimeError("report.clusters missing")
    out: Dict[str, Dict[str, Dict[str, Any]]] = {"default": {}, "70": {}}
    has_violation = False
    for cluster_key, cluster_obj in clusters.items():
        if not isinstance(cluster_obj, dict):
            continue
        best = cluster_obj.get("best")
        if not isinstance(best, dict):
            continue
        params = best.get("params")
        if not isinstance(params, dict):
            continue
        parts = str(cluster_key).split("_")
        if len(parts) != 3:
            continue
        profile, symbol, direction = parts
        if profile not in profiles:
            continue
        direction_key = f"{symbol.upper()}_{direction.upper()}"
        violations = [str(x) for x in (best.get("constraint_violations") or []) if str(x)]
        if violations:
            has_violation = True
        out[profile][direction_key] = {
            "params": {
                "comboPauseDrawdown2h": float(params["comboPauseDrawdown2h"]),
                "comboPauseHoldMinutes": int(float(params["comboPauseHoldMinutes"])),
            },
            "mode": _resolve_mode("auto", violations),
            "violations": violations,
            "admission_tier": str(best.get("admission_tier") or "C").upper(),
            "ready": bool(best.get("final_is_feasible")),
        }
    return out, has_violation


def _apply_to_config(
    cfg_path: Path,
    active_names: List[str],
    scoped_params: Dict[str, Dict[str, Any]],
    allowed_direction_keys_by_trader: Dict[str, Set[str]],
    forced_mode: str,
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
        cfg["comboPauseEnabled"] = True
        cfg["comboPauseStatsMode"] = "route"
        cfg["comboPauseCheckSeconds"] = int(cfg.get("comboPauseCheckSeconds") or 60)
        mode_by_dir = cfg.get("comboPauseModeBySymbolDirection")
        if not isinstance(mode_by_dir, dict):
            mode_by_dir = {}
        params_by_dir = cfg.get("comboPauseBySymbolDirection")
        if not isinstance(params_by_dir, dict):
            params_by_dir = {}
        resolved_modes: List[str] = []
        allowed_keys = set(allowed_direction_keys_by_trader.get(name) or set())
        for stale_key in list(mode_by_dir.keys()):
            if stale_key in {"BTC_UP", "BTC_DOWN", "ETH_UP", "ETH_DOWN"} and stale_key not in allowed_keys:
                mode_by_dir.pop(stale_key, None)
                params_by_dir.pop(stale_key, None)
        for dir_key, payload in scoped_params.items():
            params = payload.get("params") if isinstance(payload, dict) else None
            if not isinstance(params, dict):
                continue
            if not bool(payload.get("ready")):
                continue
            if allowed_keys and dir_key not in allowed_keys:
                continue
            mode = forced_mode if forced_mode != "auto" else str(payload.get("mode") or "shadow")
            mode_by_dir[dir_key] = mode
            params_by_dir[dir_key] = dict(params)
            resolved_modes.append(mode)
            cells += 1
        if resolved_modes:
            cfg["comboPauseModeBySymbolDirection"] = mode_by_dir
            cfg["comboPauseBySymbolDirection"] = params_by_dir
            cfg["comboPauseMode"] = resolved_modes[0] if len(set(resolved_modes)) == 1 else "shadow"
            touched += 1

    backup = cfg_path.with_suffix(cfg_path.suffix + f".bak_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
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
    ap = argparse.ArgumentParser(description="Apply directional combo pause params to active trader configs")
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    ap.add_argument("--mode", choices=["auto", "enforce", "shadow"], default="enforce")
    ap.add_argument("--target-profile", choices=["default", "70", "all"], default="all")
    ap.add_argument("--target-traders", type=str, default="", help="comma list: trader or profile/trader")
    ap.add_argument("--decision-json", type=Path, default=None)
    ap.add_argument("--recent-gain-json", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    profiles = {"default", "70"} if args.target_profile == "all" else {args.target_profile}
    report = _load_json(args.report)
    params_by_profile, has_violation = _extract_cluster_params(report, profiles)

    active_default_all = _parse_active(ACTIVE_DEFAULT)
    active_70_all = _parse_active(ACTIVE_70)
    plain_targets, scoped_targets = _parse_target_traders(args.target_traders)
    decision_targets = _targets_from_decision(args.decision_json, "combo", profiles) if args.decision_json else None
    recent_targets = _targets_from_recent(args.recent_gain_json, "combo", profiles) if args.recent_gain_json else None
    recent_cluster_keys = _cluster_keys_from_recent(args.recent_gain_json, "combo", profiles) if args.recent_gain_json else {"default": {}, "70": {}}

    active_default = _filter_active_names(active_default_all, "default", plain_targets, scoped_targets, decision_targets, recent_targets)
    active_70 = _filter_active_names(active_70_all, "70", plain_targets, scoped_targets, decision_targets, recent_targets)

    r1: Dict[str, Any] = {"touched_traders": 0, "cells": 0, "missing_traders": [], "backup": ""}
    r2: Dict[str, Any] = {"touched_traders": 0, "cells": 0, "missing_traders": [], "backup": ""}
    if args.target_profile in ("default", "all"):
        r1 = _apply_to_config(
            CFG_DEFAULT,
            active_default,
            params_by_profile["default"],
            recent_cluster_keys["default"],
            args.mode,
            bool(args.dry_run),
        )
    if args.target_profile in ("70", "all"):
        r2 = _apply_to_config(
            CFG_70,
            active_70,
            params_by_profile["70"],
            recent_cluster_keys["70"],
            args.mode,
            bool(args.dry_run),
        )

    print("[OK] applied combo directional params")
    if args.target_traders:
        print(f"[INFO] target_traders_filter={args.target_traders}")
    if args.decision_json:
        print(f"[INFO] decision_filter={args.decision_json}")
    if args.recent_gain_json:
        print(f"[INFO] recent_gain_filter={args.recent_gain_json}")
    print(f"[INFO] default: touched={r1['touched_traders']} cells={r1['cells']} missing={len(r1['missing_traders'])}")
    print(f"[INFO] 70:      touched={r2['touched_traders']} cells={r2['cells']} missing={len(r2['missing_traders'])}")
    print(f"[INFO] report_has_violations={has_violation}")
    if not args.dry_run:
        for backup in (str(r1.get("backup") or ""), str(r2.get("backup") or "")):
            if backup:
                print(f"[INFO] backup={backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
