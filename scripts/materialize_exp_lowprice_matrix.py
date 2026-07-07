#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "ops"))

from scripts.exp_lowprice_selected_common import (
    PRICE_BOUNDS,
    PROFILES,
    dump_json,
    expected_clone_initial_capital,
    fixed_dynamic_runtime_row_override,
    format_ladder_from_range,
    is_retired_lowprice_baseline,
    iter_fixed_dynamic_compare_rows,
    lowprice_selection_allowed,
    now_iso,
    resolve_baseline,
    resolve_prediction_suffix,
)
import lowprice_candidate_cutover_common as lowprice_common
from lowprice_retirement_common import is_retired_compare_only_trader


REPORT = ROOT / "reports" / "exp_lowprice_matrix_materialized_latest.json"
EXPANSION_REVIEW_REPORT = ROOT / "reports" / "exp_lowprice_expansion_candidate_review_latest.json"
MODE_REGISTRY = ROOT / "polymarket" / "trading_mode_registry.json"
FIXED_BP_RE = re.compile(r"^(?P<prefix>.+?)_bp(?P<bp>\d{4})$")


def _load_config_rows(path: Path) -> list[dict[str, Any]]:
    payload = lowprice_common.load_json(path, [])
    if not isinstance(payload, list):
        raise SystemExit(f"invalid lowprice config payload: {path}")
    return [dict(row) for row in payload if isinstance(row, dict)]


def _load_monitor_payload(path: Path) -> dict[str, Any]:
    payload = lowprice_common.load_json(path, {})
    if not isinstance(payload, dict):
        raise SystemExit(f"invalid lowprice monitor payload: {path}")
    traders = payload.get("traders")
    if not isinstance(traders, list):
        raise SystemExit(f"invalid lowprice monitor traders payload: {path}")
    fixed_rows = payload.get("fixedDynamicCompareRows")
    if fixed_rows is not None and not isinstance(fixed_rows, list):
        raise SystemExit(f"invalid lowprice fixedDynamicCompareRows payload: {path}")
    return dict(payload)


def _row_is_retired(profile: str, row: dict[str, Any]) -> bool:
    row_name = str(row.get("name") or "").strip()
    row_symbol = str(
        lowprice_common._symbol_from_row(row)
        or row.get("symbol")
        or ""
    ).strip().upper()
    if row_name and row_symbol and is_retired_compare_only_trader(profile, row_name, row_symbol):
        return True
    source_trader = str(
        lowprice_common._source_trader_from_row(row)
        or row.get("source_trader")
        or row.get("sourceTrader")
        or ""
    ).strip()
    symbol = str(
        lowprice_common._symbol_from_row(row)
        or row.get("symbol")
        or ""
    ).strip().upper()
    if not source_trader or not symbol:
        return False
    if is_retired_lowprice_baseline(profile, source_trader, symbol):
        return True
    return not lowprice_selection_allowed(
        profile,
        source_trader,
        symbol,
        lowprice_common._selection_mode_from_row(row),
        lowprice_common._selected_buy_price_from_row(row),
        lowprice_common._selected_buy_price_range_from_row(row),
    )


def _prune_retired_registry_entries(profile: str, retired_names: set[str]) -> int:
    if not retired_names or not MODE_REGISTRY.exists():
        return 0
    payload = lowprice_common.load_json(MODE_REGISTRY, {})
    if not isinstance(payload, dict):
        return 0
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        return 0
    profile_payload = profiles.get(profile)
    if not isinstance(profile_payload, dict):
        return 0
    entries = profile_payload.get("entries")
    if not isinstance(entries, list):
        return 0
    kept: list[dict[str, Any]] = []
    removed = 0
    for entry in entries:
        if not isinstance(entry, dict):
            kept.append(entry)
            continue
        trader_name = str(entry.get("traderName") or "").strip()
        if trader_name and trader_name in retired_names:
            removed += 1
            continue
        kept.append(entry)
    if removed:
        profile_payload = dict(profile_payload)
        profile_payload["entries"] = kept
        profiles = dict(profiles)
        profiles[profile] = profile_payload
        payload = dict(payload)
        payload["profiles"] = profiles
        dump_json(MODE_REGISTRY, payload)
    return removed


def _assert_no_retired_rows(profile: str, label: str, rows: list[dict[str, Any]]) -> None:
    retired = []
    for row in rows:
        if isinstance(row, dict) and _row_is_retired(profile, row):
            retired.append(
                {
                    "name": str(row.get("name") or "").strip(),
                    "sourceTrader": str(
                        lowprice_common._source_trader_from_row(row)
                        or row.get("source_trader")
                        or row.get("sourceTrader")
                        or ""
                    ).strip(),
                    "symbol": str(
                        lowprice_common._symbol_from_row(row)
                        or row.get("symbol")
                        or ""
                    ).strip().upper(),
                }
            )
    if retired:
        raise RuntimeError(f"retired lowprice rows leaked into {label}[{profile}]: {retired}")


def _apply_status(
    optimization: dict[str, Any],
    profile: str,
    row: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    updated = dict(row)
    candidate = lowprice_common.find_candidate_for_row(optimization, profile, updated)
    if isinstance(candidate, dict):
        lowprice_common.apply_candidate_fields(updated, candidate)
        return updated, candidate
    lowprice_common.apply_missing_candidate_status(updated)
    return updated, None


def _load_expansion_review_map() -> dict[tuple[str, str, str], dict[str, Any]]:
    payload = lowprice_common.load_json(EXPANSION_REVIEW_REPORT, {})
    rows = payload.get("rows") if isinstance(payload, dict) else []
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = (
            lowprice_common.normalize_profile(row.get("profile")),
            str(row.get("source_trader") or "").strip(),
            str(row.get("symbol") or "").strip().upper(),
        )
        if key[1] and key[2]:
            out[key] = dict(row)
    return out


def _selection_identity(
    source_trader: str,
    symbol: str,
    selection_mode: str,
    selected_buy_price: Any,
    selected_buy_price_range: Any,
) -> tuple[str, str, str, float | None, tuple[float, ...]]:
    price = None
    if selected_buy_price is not None:
        try:
            price = round(float(selected_buy_price), 2)
        except Exception:
            price = None
    price_range: tuple[float, ...] = ()
    if isinstance(selected_buy_price_range, list):
        try:
            price_range = tuple(round(float(x), 2) for x in selected_buy_price_range)
        except Exception:
            price_range = ()
    return (
        str(source_trader or "").strip(),
        str(symbol or "").strip().upper(),
        str(selection_mode or "").strip(),
        price,
        price_range,
    )


def _selection_identity_from_row(row: dict[str, Any]) -> tuple[str, str, str, float | None, tuple[float, ...]]:
    return _selection_identity(
        lowprice_common._source_trader_from_row(row),
        lowprice_common._symbol_from_row(row),
        lowprice_common._selection_mode_from_row(row),
        lowprice_common._selected_buy_price_from_row(row),
        lowprice_common._selected_buy_price_range_from_row(row),
    )


def _selection_identity_from_candidate(row: dict[str, Any]) -> tuple[str, str, str, float | None, tuple[float, ...]]:
    return _selection_identity(
        str(row.get("source_trader") or "").strip(),
        str(row.get("symbol") or "").strip().upper(),
        str(row.get("selection_mode") or "").strip(),
        row.get("selected_buy_price"),
        row.get("selected_buy_price_range"),
    )


def _fixed_clone_name(source_trader: str, symbol: str, selected_buy_price: float | None) -> str:
    symbol_lower = str(symbol or "").strip().lower()
    if selected_buy_price is None:
        return f"{source_trader}_{symbol_lower}"
    selected_bp = f"{int(round(float(selected_buy_price) * 1000)):04d}"
    match = FIXED_BP_RE.match(str(source_trader or "").strip())
    if not match:
        return f"{source_trader}_bp{selected_bp}_{symbol_lower}"
    prefix = str(match.group("prefix") or "").strip()
    source_bp = str(match.group("bp") or "").strip()
    if source_bp and source_bp != selected_bp:
        return f"{prefix}_bp{selected_bp}_{symbol_lower}_src{source_bp}"
    return f"{prefix}_bp{selected_bp}_{symbol_lower}"


def _dynamic_clone_name(source_trader: str, symbol: str, selected_buy_price_range: list[float] | None) -> str:
    symbol_lower = str(symbol or "").strip().lower()
    if not isinstance(selected_buy_price_range, list) or len(selected_buy_price_range) != 2:
        return f"{source_trader}_{symbol_lower}"
    lo = f"{int(round(float(selected_buy_price_range[0]) * 1000)):04d}"
    hi = f"{int(round(float(selected_buy_price_range[1]) * 1000)):04d}"
    match = FIXED_BP_RE.match(str(source_trader or "").strip())
    if match:
        prefix = str(match.group("prefix") or "").strip()
        source_bp = str(match.group("bp") or "").strip()
        return f"{prefix}_bp_dyn_{lo}_{hi}_{symbol_lower}_src{source_bp}"
    return f"{source_trader}_bp_dyn_{lo}_{hi}_{symbol_lower}"


def _clone_name(source_trader: str, symbol: str, selection_mode: str, selected_buy_price: float | None, selected_buy_price_range: list[float] | None) -> str:
    if str(selection_mode or "").strip() == "dynamic_range":
        return _dynamic_clone_name(source_trader, symbol, selected_buy_price_range)
    return _fixed_clone_name(source_trader, symbol, selected_buy_price)


def _build_clone(baseline: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    source_row = dict(baseline.get("source_row") or {})
    profile = str(baseline.get("profile") or "default")
    symbol = str(result.get("symbol") or baseline.get("symbol") or "").strip().upper()
    source_trader = str(result.get("source_trader") or baseline.get("source_trader") or "").strip()
    selection_mode = str(result.get("selection_mode") or "").strip()
    selected_buy_price = result.get("selected_buy_price")
    selected_buy_price_range = result.get("selected_buy_price_range")
    clone = dict(source_row)
    clone["name"] = _clone_name(
        source_trader,
        symbol,
        selection_mode,
        selected_buy_price if selected_buy_price is None else float(selected_buy_price),
        [float(x) for x in (selected_buy_price_range or [])] if isinstance(selected_buy_price_range, list) else None,
    )
    clone["group"] = str(PROFILES[profile]["group"])
    clone["profile"] = profile
    clone["role"] = "compare_only"
    clone["logsDir"] = f"{PROFILES[profile]['logs_prefix']}{clone['name']}"
    generic_suffix = str(source_row.get("predictionSuffix") or baseline.get("source_prediction_suffix") or "").strip()
    primary_suffix = resolve_prediction_suffix(source_row, symbol)
    clone["predictionSuffix"] = generic_suffix or primary_suffix
    clone["predictionSuffixBySymbol"] = {symbol: primary_suffix}
    fallback_suffix = generic_suffix if generic_suffix and generic_suffix != primary_suffix else None
    if fallback_suffix:
        clone["predictionFallbackSuffix"] = fallback_suffix
    else:
        clone.pop("predictionFallbackSuffix", None)
    clone["allowedMarkets"] = symbol
    clone["initialCapital"] = expected_clone_initial_capital(source_row)
    clone["perCoinCapital"] = False
    clone["numTradingAssets"] = 1
    clone["runtimeSplitBySymbol"] = False
    clone["runtimeSeedCapitalFromReports"] = False
    clone["runtimeIndependentCellCapital"] = True
    clone["usePredictionLimitPrice"] = False
    clone["lowPriceSourceTrader"] = source_trader
    clone["lowPriceSourceProfile"] = profile
    clone["lowPriceSymbol"] = symbol
    clone["lowPriceSourceAllowedMarkets"] = baseline.get("source_allowed_markets") or [symbol]
    clone["lowPriceSourceInitialCapital"] = float(baseline.get("source_initial_capital") or clone["initialCapital"] or 400.0)
    clone["lowPriceExpectedCloneInitialCapital"] = float(baseline.get("expected_clone_initial_capital") or clone["initialCapital"] or 400.0)
    clone["lowPriceSelectionMode"] = selection_mode
    clone["lowPriceSelectedBuyPrice"] = selected_buy_price
    clone["lowPriceSelectedBuyPriceRange"] = selected_buy_price_range
    clone["lowPriceBounds"] = list(PRICE_BOUNDS)
    clone["lowPriceExperiment"] = "finalists_compare_only_1m_180d"
    clone["lowPriceBypassMarketSelection"] = True
    clone["lowPriceBypassThresholdBase"] = True
    clone["lowPriceSourceResolvedBuyPrice"] = baseline.get("resolved_source_buy_price")
    clone["lowPriceSourceResolvedBuyPriceRange"] = baseline.get("resolved_source_buy_price_range")
    clone["lowPriceSourceSizingReferencePrice"] = baseline.get("resolved_source_sizing_reference_price")
    clone["lowPriceFinalistRank"] = result.get("finalist_rank")
    clone["lowPriceFinalistKey"] = result.get("finalist_key")
    clone["lowpriceFamilyId"] = result.get("lowpriceFamilyId") or result.get("finalist_key")
    if selection_mode == "dynamic_range":
        clone["limitPrice"] = None
        clone["limitPriceLadder"] = format_ladder_from_range([float(x) for x in (selected_buy_price_range or [])])
        clone["lowPriceDynamicMinTotalAmountUsd"] = 15.0
    else:
        clone["limitPrice"] = float(selected_buy_price) if selected_buy_price is not None else None
        clone["limitPriceLadder"] = None
        clone["lowPriceDynamicMinTotalAmountUsd"] = None
    lowprice_common.apply_candidate_fields(clone, result)
    return clone


def _monitor_row_from_config(row: dict[str, Any]) -> dict[str, Any]:
    symbol = lowprice_common._symbol_from_row(row)
    return {
        "name": str(row.get("name") or "").strip(),
        "group": row.get("group"),
        "profile": row.get("profile"),
        "logsDir": row.get("logsDir"),
        "allowedMarkets": row.get("allowedMarkets"),
        "initialCapital": row.get("initialCapital"),
        "role": row.get("role") or "compare_only",
        "symbol": symbol,
        "sourceTrader": lowprice_common._source_trader_from_row(row),
        "selectionMode": lowprice_common._selection_mode_from_row(row),
        "buyPrice": lowprice_common._selected_buy_price_from_row(row),
        "buyPriceRange": lowprice_common._selected_buy_price_range_from_row(row),
        "finalistRank": row.get("lowPriceFinalistRank") or row.get("finalistRank"),
        "finalistKey": row.get("lowPriceFinalistKey") or row.get("finalistKey"),
        "selectionOrigin": row.get("selectionOrigin") or "optimizer_finalist",
        "lowpriceFamilyId": row.get("lowpriceFamilyId"),
        "familyRuleVersion": row.get("familyRuleVersion"),
        "staleOrderPolicy": row.get("staleOrderPolicy"),
        "staleCancelMinAgeSec": row.get("staleCancelMinAgeSec"),
        "staleCancelMinDriftTicks": row.get("staleCancelMinDriftTicks"),
        "staleCancelMinTimeToExpirySec": row.get("staleCancelMinTimeToExpirySec"),
        "rulesJsonPath": row.get("rulesJsonPath"),
        "lowpriceCutoverStatus": row.get("lowpriceCutoverStatus"),
        "lowpriceCutoverReason": row.get("lowpriceCutoverReason"),
        "lowpriceLatestApprovedRulesJsonPath": row.get("lowpriceLatestApprovedRulesJsonPath"),
        "lowpriceLatestApprovedFamilyRuleVersion": row.get("lowpriceLatestApprovedFamilyRuleVersion"),
        "liveConsumptionStatus": row.get("liveConsumptionStatus"),
        "liveConsumptionReason": row.get("liveConsumptionReason"),
        "forcedLowpriceMaterialization": row.get("forcedLowpriceMaterialization"),
        "forcedLowpriceReason": row.get("forcedLowpriceReason"),
        "forcedLowpriceSelectionMethod": row.get("forcedLowpriceSelectionMethod"),
        "predictionSurfaceChoice": row.get("predictionSurfaceChoice"),
        "scientificEligibilityStatus": row.get("scientificEligibilityStatus"),
        "earlyAdmissionClass": row.get("earlyAdmissionClass"),
        "earlyAdmissionReason": row.get("earlyAdmissionReason"),
        "jointWinnerParamSource": row.get("jointWinnerParamSource"),
        "historyResetMode": row.get("historyResetMode"),
        "coverage365Pct": row.get("coverage365Pct"),
        "coverage540Pct": row.get("coverage540Pct"),
    }


def _resolve_collisions(rows: list[dict[str, Any]]) -> None:
    seen_names: set[str] = set()
    seen_logs: set[str] = set()
    for row in rows:
        base_name = str(row.get("name") or "").strip()
        name = base_name
        idx = 2
        while name in seen_names:
            name = f"{base_name}_alt{idx}"
            idx += 1
        if name != base_name:
            row["name"] = name
        seen_names.add(name)
        base_logs = str(row.get("logsDir") or "").strip() or name
        logs_dir = base_logs
        idx = 2
        while logs_dir in seen_logs:
            logs_dir = f"{base_logs}_alt{idx}"
            idx += 1
        row["logsDir"] = logs_dir
        seen_logs.add(logs_dir)


def _mirror_row_into_reports(row: dict[str, Any]) -> None:
    logs_dir = str(row.get("logsDir") or "").strip()
    if not logs_dir:
        return
    report_dir = ROOT / "polymarket" / logs_dir / "reports"
    for filename in ("report_summary.json", "report_summary.simulation.json", "report_summary.live.json"):
        path = report_dir / filename
        payload = lowprice_common.load_json(path, None)
        if not isinstance(payload, dict):
            continue
        summary = payload.get("summary")
        if not isinstance(summary, dict):
            continue
        for field in [
            "rulesJsonPath",
            "lowpriceFamilyId",
            "familyRuleVersion",
            "staleOrderPolicy",
            "staleCancelMinAgeSec",
            "staleCancelMinDriftTicks",
            "staleCancelMinTimeToExpirySec",
            "lowpriceCutoverStatus",
            "lowpriceCutoverReason",
            "lowpriceLatestApprovedRulesJsonPath",
            "lowpriceLatestApprovedFamilyRuleVersion",
            "liveConsumptionStatus",
            "liveConsumptionReason",
            "forcedLowpriceMaterialization",
            "forcedLowpriceReason",
            "forcedLowpriceSelectionMethod",
            "predictionSurfaceChoice",
            "scientificEligibilityStatus",
            "earlyAdmissionClass",
            "earlyAdmissionReason",
            "jointWinnerParamSource",
            "historyResetMode",
            "coverage365Pct",
            "coverage540Pct",
        ]:
            summary[field] = row.get(field)
        lowprice_common.write_json(path, payload)


def _candidate_selected_cells(optimization: dict[str, Any], profile: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for source_row in ((optimization.get("profiles") or {}).get(profile) or []):
        if not isinstance(source_row, dict):
            continue
        source_trader = str(source_row.get("source_trader") or "").strip()
        symbol = str(source_row.get("symbol") or "").strip().upper()
        if is_retired_lowprice_baseline(profile, source_trader, symbol):
            continue
        for finalist in source_row.get("runtime_finalists") or []:
            if not isinstance(finalist, dict):
                continue
            if not lowprice_selection_allowed(
                profile,
                source_trader,
                symbol,
                str(finalist.get("selection_mode") or "").strip(),
                finalist.get("selected_buy_price"),
                finalist.get("selected_buy_price_range"),
            ):
                continue
            payload = {
                "source_trader": source_trader,
                "symbol": symbol,
                "selection_mode": finalist.get("selection_mode"),
                "selected_buy_price": finalist.get("selected_buy_price"),
                "selected_buy_price_range": finalist.get("selected_buy_price_range"),
                "finalist_rank": finalist.get("finalist_rank"),
                "finalist_key": finalist.get("finalist_key"),
                "selection_origin": finalist.get("selection_origin") or "optimizer_finalist",
            }
            lowprice_common.apply_candidate_fields(payload, finalist)
            out.append(payload)
    return out


def _expansion_allows_materialization(
    expansion_review: dict[tuple[str, str, str], dict[str, Any]],
    profile: str,
    source_trader: str,
    symbol: str,
) -> bool:
    row = expansion_review.get((profile, source_trader, symbol))
    return str((row or {}).get("decision") or "").strip() == "add"


def _materialize_profile(profile: str, optimization: dict[str, Any]) -> dict[str, Any]:
    spec = PROFILES[profile]
    expansion_review = _load_expansion_review_map()
    raw_config_rows = _load_config_rows(spec["out_config"])
    config_rows = [row for row in raw_config_rows if not _row_is_retired(profile, row)]
    monitor_payload = _load_monitor_payload(spec["out_monitor"])
    retired_names = {
        str(row.get("name") or "").strip()
        for row in raw_config_rows
        if _row_is_retired(profile, row) and str(row.get("name") or "").strip()
    }
    monitor_rows = [
        dict(row)
        for row in (monitor_payload.get("traders") or [])
        if isinstance(row, dict) and not _row_is_retired(profile, row)
    ]
    for row in (monitor_payload.get("traders") or []):
        if isinstance(row, dict) and _row_is_retired(profile, row) and str(row.get("name") or "").strip():
            retired_names.add(str(row.get("name") or "").strip())
    fixed_dynamic_rows = [
        dict(row)
        for row in (monitor_payload.get("fixedDynamicCompareRows") or [])
        if isinstance(row, dict) and not _row_is_retired(profile, row)
    ]

    updated_configs: list[dict[str, Any]] = []
    config_by_name: dict[str, dict[str, Any]] = {}
    consuming_latest = 0
    missing_candidate = 0
    synthesized_candidate_rows = 0
    forced_rows = 0

    for row in config_rows:
        updated, candidate = _apply_status(optimization, profile, row)
        updated_configs.append(updated)
        name = str(updated.get("name") or "").strip()
        if name:
            config_by_name[name] = updated
        if isinstance(candidate, dict):
            consuming_latest += 1
            if candidate.get("forcedLowpriceMaterialization"):
                forced_rows += 1
        else:
            missing_candidate += 1

    existing_identities = {_selection_identity_from_row(row) for row in updated_configs}
    fixed_dynamic_identities = {_selection_identity_from_candidate(row) for row in fixed_dynamic_rows}
    synthesized_rows: list[dict[str, Any]] = []

    # Preserve explicitly protected live fixed-dynamic compare rows in runtime config.
    # Without this, a refresh can silently drop the slot01 alias row from out_config,
    # leaving the source trader alive while the alias itself stops attempting orders.
    for fixed_row in iter_fixed_dynamic_compare_rows(profile):
        if not isinstance(fixed_row, dict):
            continue
        source_trader = str(fixed_row.get("source_trader") or "").strip()
        symbol = str(fixed_row.get("symbol") or "").strip().upper()
        if not source_trader or not symbol:
            continue
        override = fixed_dynamic_runtime_row_override(profile, source_trader, symbol)
        if not isinstance(override, dict):
            continue
        updated_fixed_row, _ = _apply_status(optimization, profile, fixed_row)
        identity = _selection_identity_from_candidate(updated_fixed_row)
        if identity not in fixed_dynamic_identities:
            fixed_dynamic_rows.append(updated_fixed_row)
            fixed_dynamic_identities.add(identity)
        if identity in existing_identities:
            continue
        baseline = resolve_baseline(profile, source_trader, symbol)
        clone = _build_clone(baseline, updated_fixed_row)
        if str(override.get("name") or "").strip():
            clone["name"] = str(override["name"]).strip()
        if str(override.get("logsDir") or "").strip():
            clone["logsDir"] = str(override["logsDir"]).strip()
        if str(override.get("role") or "").strip():
            clone["role"] = str(override["role"]).strip()
        if is_retired_compare_only_trader(profile, str(clone.get("name") or "").strip(), symbol):
            retired_names.add(str(clone.get("name") or "").strip())
            continue
        synthesized_rows.append(clone)
        updated_configs.append(clone)
        config_by_name[str(clone.get("name") or "").strip()] = clone
        existing_identities.add(identity)
        consuming_latest += 1

    for source_row in ((optimization.get("profiles") or {}).get(profile) or []):
        if not isinstance(source_row, dict):
            continue
        source_trader = str(source_row.get("source_trader") or "").strip()
        symbol = str(source_row.get("symbol") or "").strip().upper()
        if not source_trader or not symbol:
            continue
        if is_retired_lowprice_baseline(profile, source_trader, symbol):
            continue
        runtime_finalists = [finalist for finalist in (source_row.get("runtime_finalists") or []) if isinstance(finalist, dict)]
        runtime_finalists = [
            finalist
            for finalist in runtime_finalists
            if lowprice_selection_allowed(
                profile,
                source_trader,
                symbol,
                str(finalist.get("selection_mode") or "").strip(),
                finalist.get("selected_buy_price"),
                finalist.get("selected_buy_price_range"),
            )
        ]
        if not runtime_finalists:
            continue
        force_materialize = any(bool(finalist.get("forcedLowpriceMaterialization")) for finalist in runtime_finalists)
        if not force_materialize and not _expansion_allows_materialization(expansion_review, profile, source_trader, symbol):
            continue
        for finalist in runtime_finalists:
            if not isinstance(finalist, dict):
                continue
            candidate = {
                "source_trader": source_trader,
                "symbol": symbol,
                "selection_mode": finalist.get("selection_mode"),
                "selected_buy_price": finalist.get("selected_buy_price"),
                "selected_buy_price_range": finalist.get("selected_buy_price_range"),
                "finalist_rank": finalist.get("finalist_rank"),
                "finalist_key": finalist.get("finalist_key"),
                "lowpriceFamilyId": finalist.get("lowpriceFamilyId") or finalist.get("finalist_key"),
                "familyRuleVersion": finalist.get("familyRuleVersion"),
                "rulesJsonPath": finalist.get("rulesJsonPath"),
                "selection_origin": finalist.get("selection_origin") or "optimizer_finalist",
                "staleOrderPolicy": finalist.get("staleOrderPolicy"),
                "staleCancelMinAgeSec": finalist.get("staleCancelMinAgeSec"),
                "staleCancelMinDriftTicks": finalist.get("staleCancelMinDriftTicks"),
                "staleCancelMinTimeToExpirySec": finalist.get("staleCancelMinTimeToExpirySec"),
                "forcedLowpriceMaterialization": finalist.get("forcedLowpriceMaterialization"),
                "forcedLowpriceReason": finalist.get("forcedLowpriceReason"),
                "forcedLowpriceSelectionMethod": finalist.get("forcedLowpriceSelectionMethod"),
                "predictionSurfaceChoice": finalist.get("predictionSurfaceChoice"),
                "scientificEligibilityStatus": finalist.get("scientificEligibilityStatus"),
                "earlyAdmissionClass": finalist.get("earlyAdmissionClass"),
                "earlyAdmissionReason": finalist.get("earlyAdmissionReason"),
                "jointWinnerParamSource": finalist.get("jointWinnerParamSource"),
                "historyResetMode": finalist.get("historyResetMode"),
                "coverage365Pct": finalist.get("coverage365Pct"),
                "coverage540Pct": finalist.get("coverage540Pct"),
                "runtimeFieldOverrides": finalist.get("runtimeFieldOverrides"),
            }
            identity = _selection_identity_from_candidate(candidate)
            if identity in existing_identities:
                continue
            baseline = resolve_baseline(profile, source_trader, symbol)
            clone = _build_clone(baseline, candidate)
            if is_retired_compare_only_trader(profile, str(clone.get("name") or "").strip(), symbol):
                retired_names.add(str(clone.get("name") or "").strip())
                continue
            synthesized_rows.append(clone)
            updated_configs.append(clone)
            config_by_name[str(clone.get("name") or "").strip()] = clone
            existing_identities.add(identity)
            consuming_latest += 1
            synthesized_candidate_rows += 1
            if candidate.get("forcedLowpriceMaterialization"):
                forced_rows += 1

    if synthesized_rows:
        _resolve_collisions(updated_configs)
        config_by_name = {
            str(row.get("name") or "").strip(): row
            for row in updated_configs
            if str(row.get("name") or "").strip()
        }

    updated_monitor_rows: list[dict[str, Any]] = []
    for row in monitor_rows:
        name = str(row.get("name") or "").strip()
        if name and name in config_by_name:
            updated = dict(row)
            lowprice_common.copy_cutover_fields(updated, config_by_name[name])
        else:
            updated, _ = _apply_status(optimization, profile, row)
        updated_monitor_rows.append(updated)
    updated_monitor_rows.extend(_monitor_row_from_config(row) for row in synthesized_rows)
    updated_monitor_rows.sort(key=lambda row: str(row.get("name") or ""))

    updated_fixed_dynamic_rows: list[dict[str, Any]] = []
    for row in fixed_dynamic_rows:
        updated, _ = _apply_status(optimization, profile, row)
        updated_fixed_dynamic_rows.append(updated)

    selected_cells = _candidate_selected_cells(optimization, profile)
    updated_monitor = dict(monitor_payload)
    updated_monitor["generatedAt"] = now_iso()
    updated_monitor["scope"] = "exp_lowprice_compare_only_finalists"
    updated_monitor["profile"] = profile
    updated_monitor["source"] = "materialize_exp_lowprice_matrix.py"
    updated_monitor["traders"] = updated_monitor_rows
    updated_monitor["selectedCells"] = selected_cells
    updated_monitor["fixedDynamicCompareRows"] = updated_fixed_dynamic_rows

    active_payload = {
        "generatedAt": now_iso(),
        "scope": "exp_lowprice_compare_only_finalists",
        "source": "materialize_exp_lowprice_matrix.py",
        "profile": profile,
        "traderNames": [str(row.get("name") or "") for row in updated_configs if str(row.get("name") or "").strip()],
    }

    _assert_no_retired_rows(profile, "out_config", updated_configs)
    _assert_no_retired_rows(profile, "out_monitor.traders", updated_monitor_rows)

    dump_json(spec["out_config"], updated_configs)
    dump_json(spec["out_active"], active_payload)
    dump_json(spec["out_monitor"], updated_monitor)
    registry_removed = _prune_retired_registry_entries(profile, retired_names)
    for row in updated_configs:
        _mirror_row_into_reports(row)

    return {
        "profile": profile,
        "summary": {
            "total_rows": len(updated_configs),
            "consuming_latest_rows": consuming_latest,
            "missing_candidate_rows": missing_candidate,
            "selected_cell_rows": len(selected_cells),
            "fixed_dynamic_compare_rows_total": len(updated_fixed_dynamic_rows),
            "synthesized_candidate_rows": synthesized_candidate_rows,
            "forced_rows": forced_rows,
            "retired_registry_entries_removed": registry_removed,
        },
        "rows": [
            {
                "name": str(row.get("name") or ""),
                "sourceTrader": lowprice_common._source_trader_from_row(row),
                "symbol": lowprice_common._symbol_from_row(row),
                "rulesJsonPath": lowprice_common.relative_rules_path(row.get("rulesJsonPath")),
                "familyRuleVersion": row.get("familyRuleVersion"),
                "staleOrderPolicy": row.get("staleOrderPolicy"),
                "lowpriceCutoverStatus": row.get("lowpriceCutoverStatus"),
                "liveConsumptionStatus": row.get("liveConsumptionStatus"),
                "forcedLowpriceMaterialization": row.get("forcedLowpriceMaterialization"),
                "forcedLowpriceSelectionMethod": row.get("forcedLowpriceSelectionMethod"),
                "predictionSurfaceChoice": row.get("predictionSurfaceChoice"),
                "scientificEligibilityStatus": row.get("scientificEligibilityStatus"),
                "earlyAdmissionClass": row.get("earlyAdmissionClass"),
                "earlyAdmissionReason": row.get("earlyAdmissionReason"),
                "jointWinnerParamSource": row.get("jointWinnerParamSource"),
                "historyResetMode": row.get("historyResetMode"),
                "coverage365Pct": row.get("coverage365Pct"),
                "coverage540Pct": row.get("coverage540Pct"),
            }
            for row in updated_configs
        ],
    }


def main() -> None:
    optimization = lowprice_common.load_optimization_payload()
    if not isinstance(optimization, dict):
        raise SystemExit(f"invalid optimization payload: {lowprice_common.OPTIMIZATION_REPORT}")

    profiles = {
        profile: _materialize_profile(profile, optimization)
        for profile in ("default", "70")
    }
    payload = {
        "generatedAt": now_iso(),
        "source": "materialize_exp_lowprice_matrix.py",
        "optimizationReport": str(lowprice_common.OPTIMIZATION_REPORT),
        "fallbackOptimizationReport": str(lowprice_common.FALLBACK_OPTIMIZATION_REPORT),
        "expansionReviewReport": str(EXPANSION_REVIEW_REPORT),
        "profiles": profiles,
    }
    dump_json(REPORT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
