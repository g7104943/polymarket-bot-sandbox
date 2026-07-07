#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLYMARKET_DIR = PROJECT_ROOT / "polymarket"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "direction_calibration_tuning_report.json"
CFG_DEFAULT = POLYMARKET_DIR / "trader_configs.json"
CFG_70 = POLYMARKET_DIR / "trader_configs_70.json"
ACTIVE_DEFAULT = POLYMARKET_DIR / "active_traders.json"
ACTIVE_70 = POLYMARKET_DIR / "active_traders_70.json"
TARGET_SYMBOLS = ("BTC", "ETH")
TARGET_DIRECTIONS = ("UP", "DOWN")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_active(path: Path) -> List[str]:
    obj = _load_json(path)
    if isinstance(obj, list):
        return [str(x) for x in obj]
    if isinstance(obj, dict):
        for key in ("active_traders", "traderNames"):
            value = obj.get(key)
            if isinstance(value, list):
                return [str(x) for x in value]
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


def _load_target_cells(path: Path | None) -> Dict[str, Dict[str, Set[str]]]:
    out: Dict[str, Dict[str, Set[str]]] = {"default": {}, "70": {}}
    if path is None or not path.exists():
        return out
    payload = _load_json(path)
    rows = payload.get("target_cells") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        profile = str(row.get("profile") or "").lower()
        trader = str(row.get("trader") or "")
        symbol = str(row.get("symbol") or "").upper()
        if profile not in out or not trader or symbol not in TARGET_SYMBOLS:
            continue
        out[profile].setdefault(trader, set()).add(symbol)
    return out


def _mark_core10_experimental_symbol(
    cfg: Dict[str, Any],
    symbol: str,
    source: str,
) -> None:
    tuning_map = cfg.get("core10ExperimentalTuningBySymbol")
    if not isinstance(tuning_map, dict):
        tuning_map = {}
        cfg["core10ExperimentalTuningBySymbol"] = tuning_map
    approval_map = cfg.get("core10ExperimentalLiveApprovalBySymbol")
    if not isinstance(approval_map, dict):
        approval_map = {}
        cfg["core10ExperimentalLiveApprovalBySymbol"] = approval_map

    current = tuning_map.get(symbol)
    node = dict(current) if isinstance(current, dict) else {}
    sources = [str(x) for x in node.get("sources", []) if str(x).strip()] if isinstance(node.get("sources"), list) else []
    if source not in sources:
        sources.append(source)
    tuning_map[symbol] = {
        "active": True,
        "sources": sorted(set(sources)),
        "updatedAt": datetime.now(timezone.utc).isoformat(),
    }
    approval_map.setdefault(symbol, False)


def _parse_allowed_symbols(raw: Any) -> List[str]:
    out: List[str] = []
    for token in str(raw or "").split(","):
        sym = token.strip().upper().replace("/USDT", "").replace("/USD", "")
        if sym in TARGET_SYMBOLS:
            out.append(sym)
    return sorted(set(out))


def _cluster_key(profile: str, symbol: str, direction: str) -> str:
    return f"{profile}_{symbol}_{direction}"


def _symbol_direction_key(symbol: str, direction: str) -> str:
    return f"{symbol}_{direction}"


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
        state = layers.get("calibration")
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
        layer_payload = layers.get("calibration")
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
) -> Tuple[Dict[str, Dict[str, Any]], bool, Dict[str, List[str]], Set[str]]:
    clusters = report.get("clusters") if isinstance(report, dict) else None
    if not isinstance(clusters, dict):
        raise RuntimeError("report.clusters missing")

    params_by_cluster: Dict[str, Dict[str, Any]] = {}
    violations: Dict[str, List[str]] = {}
    available_clusters: Set[str] = set()
    has_violation = False
    for profile in ("default", "70"):
        if profile not in profiles:
            continue
        for symbol in TARGET_SYMBOLS:
            for direction in TARGET_DIRECTIONS:
                cluster_key = _cluster_key(profile, symbol, direction)
                node = clusters.get(cluster_key)
                if not isinstance(node, dict):
                    continue
                available_clusters.add(cluster_key)
                best = node.get("best")
                if not isinstance(best, dict):
                    raise RuntimeError(f"cluster.best missing: {cluster_key}")
                params = best.get("params")
                if isinstance(params, dict):
                    params_by_cluster[cluster_key] = dict(params)
                else:
                    params_by_cluster[cluster_key] = {}
                v = best.get("constraint_violations")
                vlist = [str(x) for x in v] if isinstance(v, list) else []
                if not isinstance(params, dict):
                    vlist = list(dict.fromkeys(vlist + ["missing_params"]))
                violations[cluster_key] = vlist
                if vlist:
                    has_violation = True
    return params_by_cluster, has_violation, violations, available_clusters


def _resolve_mode(requested_mode: str, cluster_violations: List[str]) -> str:
    if requested_mode == "enforce":
        return "enforce"
    if requested_mode == "shadow":
        return "shadow"
    return "shadow" if cluster_violations else "enforce"


def _apply_to_config(
    cfg_path: Path,
    active_names: List[str],
    profile: str,
    params_by_cluster: Dict[str, Dict[str, Any]],
    mode_by_symbol_direction: Dict[str, str],
    target_symbols_by_trader: Dict[str, Set[str]] | None,
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
        allowed_symbols = _parse_allowed_symbols(cfg.get("allowedMarkets"))
        if not allowed_symbols:
            continue
        target_symbols = target_symbols_by_trader.get(name, set()) if target_symbols_by_trader else set()

        cfg["calibrationStatsMode"] = "route"
        cfg["calibrationCheckSeconds"] = int(cfg.get("calibrationCheckSeconds") or 60)

        mode_map = cfg.get("calibrationModeBySymbolDirection")
        if not isinstance(mode_map, dict):
            mode_map = {}
        param_map = cfg.get("calibrationBySymbolDirection")
        if not isinstance(param_map, dict):
            param_map = {}

        resolved_modes: List[str] = []
        for symbol in allowed_symbols:
            if target_symbols and symbol not in target_symbols:
                continue
            for direction in TARGET_DIRECTIONS:
                sd_key = _symbol_direction_key(symbol, direction)
                cluster_key = _cluster_key(profile, symbol, direction)
                mode = str(mode_by_symbol_direction.get(sd_key, "shadow"))
                mode_map[sd_key] = mode
                resolved_modes.append(mode)
                current_params = params_by_cluster.get(cluster_key) or {}
                if current_params:
                    param_map[sd_key] = dict(current_params)
                cells += 1
            _mark_core10_experimental_symbol(cfg, symbol, "calibration")

        cfg["calibrationModeBySymbolDirection"] = mode_map
        cfg["calibrationBySymbolDirection"] = param_map
        cfg["calibrationMode"] = resolved_modes[0] if resolved_modes and len(set(resolved_modes)) == 1 else "shadow"
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
    ap = argparse.ArgumentParser(description="Apply direction calibration params to active trader configs")
    ap.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    # 默认 auto：有违例的方向簇保持 shadow，其余 enforce
    ap.add_argument("--mode", choices=["auto", "enforce", "shadow"], default="auto")
    ap.add_argument("--target-profile", choices=["default", "70", "all"], default="all")
    ap.add_argument("--target-traders", type=str, default="", help="comma list: trader or profile/trader")
    ap.add_argument("--target-cells-json", type=Path, default=None, help="exact target cells json; limit apply to profile/trader/symbol")
    ap.add_argument("--decision-json", type=Path, default=None, help="top20 decision json; only apply ready traders")
    ap.add_argument("--recent-gain-json", type=Path, default=None, help="recent gain json; only apply all-positive traders")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    report = _load_json(args.report)
    profiles = {"default", "70"} if args.target_profile == "all" else {args.target_profile}
    params_by_cluster, has_violation, violations, available_clusters = _extract_cluster_params(report, profiles)

    default_mode_by_symbol_direction = {
        _symbol_direction_key(symbol, direction): _resolve_mode(args.mode, violations.get(_cluster_key("default", symbol, direction), []))
        for symbol in TARGET_SYMBOLS
        for direction in TARGET_DIRECTIONS
        if _cluster_key("default", symbol, direction) in available_clusters
    }
    profile70_mode_by_symbol_direction = {
        _symbol_direction_key(symbol, direction): _resolve_mode(args.mode, violations.get(_cluster_key("70", symbol, direction), []))
        for symbol in TARGET_SYMBOLS
        for direction in TARGET_DIRECTIONS
        if _cluster_key("70", symbol, direction) in available_clusters
    }

    active_default_all = _parse_active(ACTIVE_DEFAULT)
    active_70_all = _parse_active(ACTIVE_70)
    plain_targets, scoped_targets = _parse_target_traders(args.target_traders)
    target_cells = _load_target_cells(args.target_cells_json)
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
            profile="default",
            params_by_cluster=params_by_cluster,
            mode_by_symbol_direction=default_mode_by_symbol_direction,
            target_symbols_by_trader=target_cells.get("default"),
            dry_run=args.dry_run,
        )
    if args.target_profile in ("70", "all"):
        r2 = _apply_to_config(
            cfg_path=CFG_70,
            active_names=active_70,
            profile="70",
            params_by_cluster=params_by_cluster,
            mode_by_symbol_direction=profile70_mode_by_symbol_direction,
            target_symbols_by_trader=target_cells.get("70"),
            dry_run=args.dry_run,
        )

    print("[OK] applied direction calibration params")
    print(f"[INFO] default_modes={default_mode_by_symbol_direction}")
    print(f"[INFO] profile70_modes={profile70_mode_by_symbol_direction}")
    if args.target_traders:
        print(f"[INFO] target_traders_filter={args.target_traders}")
    if args.decision_json:
        print(f"[INFO] decision_filter={args.decision_json}")
    if args.recent_gain_json:
        print(f"[INFO] recent_gain_filter={args.recent_gain_json}")
    if args.target_profile in ("default", "all"):
        print(f"[INFO] default: touched={r1['touched_traders']} cells={r1['cells']} missing={len(r1['missing_traders'])}")
    if args.target_profile in ("70", "all"):
        print(f"[INFO] 70:      touched={r2['touched_traders']} cells={r2['cells']} missing={len(r2['missing_traders'])}")
    if has_violation:
        print("[WARN] constraint violations detected in report:")
        for cluster_key, vlist in violations.items():
            if vlist:
                print(f"  - {cluster_key}: {vlist}")
    if not args.dry_run:
        backups = [b for b in (str(r1.get("backup") or ""), str(r2.get("backup") or "")) if b]
        if backups:
            print("[INFO] backups:")
            for backup in backups:
                print(f"  - {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
