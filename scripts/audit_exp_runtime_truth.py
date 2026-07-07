#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POLY = ROOT / "polymarket"
REPORT = ROOT / "reports" / "exp_runtime_truth_latest.json"

FIXED_RE = re.compile(r"_bp(\d{4})$", re.IGNORECASE)
DYN_RE = re.compile(r"_bp_dyn_(\d{4})_(\d{4})$", re.IGNORECASE)

PROFILE_MAP = {
    "default": {
        "config": POLY / "trader_configs.json",
        "active": POLY / "active_traders.json",
    },
    "70": {
        "config": POLY / "trader_configs_70.json",
        "active": POLY / "active_traders_70.json",
    },
}

RUNTIME_DEFAULTS = {
    "min_token_price": 0.05,
    "max_token_price": 0.54,
    "effective_sim_bet_pct_caps": {"normal": 0.05, "conservative": 0.03},
    "monitor_window_minutes_by_timeframe": {"15m": 5, "5m": 2},
    "last5min_price_band": [0.30, 0.65],
    "sources": {
        "prediction_env": str(POLY / "src" / "config" / "prediction_env.ts"),
        "prediction_executor": str(POLY / "src" / "services" / "prediction_executor.ts"),
    },
}

MODE_KEYS = [
    "selectorMode",
    "selectorModeBySymbol",
    "selectorEligibleBySymbol",
    "selectorScopeBySymbol",
    "regimeMode",
    "regimeModeBySymbol",
    "calibrationMode",
    "calibrationModeBySymbolDirection",
    "calibrationStatsMode",
    "calibrationCheckSeconds",
    "expectancyGateMode",
    "thresholdDriftMode",
    "metaLabelMode",
    "jointOptimizationMode",
    "jointOptimizationExpectedPayoffMode",
    "runtimeGuardMode",
    "overlayMode",
]

RISK_KEYS = [
    "probThreshold",
    "minEdge",
    "kellyFrac",
    "betPctNormal",
    "betPctConservative",
    "confTier1Bound",
    "confTier2Bound",
    "tier1Mult",
    "tier2Mult",
    "tier3Mult",
    "cooldownBars",
    "drawdownHalt",
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def active_names(path: Path) -> set[str]:
    payload = load_json(path)
    out = payload.get("traderNames") or payload.get("active_traders") or []
    return {str(x).strip() for x in out if str(x).strip()}


def parse_allowed_markets(raw: Any) -> list[str]:
    if isinstance(raw, list):
        vals = [str(x).strip().upper() for x in raw if str(x).strip()]
    else:
        vals = [part.strip().upper() for part in str(raw or "").split(",") if part.strip()]
    return sorted(set(vals))


def parse_name_price(name: str) -> tuple[str | None, float | None, list[float] | None]:
    dyn = DYN_RE.search(name)
    if dyn:
        lo = round(int(dyn.group(1)) / 1000.0, 2)
        hi = round(int(dyn.group(2)) / 1000.0, 2)
        return "dynamic", None, [lo, hi]
    fixed = FIXED_RE.search(name)
    if fixed:
        price = round(int(fixed.group(1)) / 1000.0, 2)
        return "fixed", price, None
    return None, None, None


def parse_rules_price(path: Path | None) -> dict[str, Any]:
    out = {
        "path_exists": False,
        "rules_buy_price": None,
        "rules_buy_price_range": None,
        "rules_path_price_mode": None,
        "rules_path_fixed_price": None,
        "rules_path_range": None,
    }
    if not path:
        return out
    out["path_exists"] = path.exists()
    path_mode, path_fixed, path_range = parse_name_price(path.stem)
    out["rules_path_price_mode"] = path_mode
    out["rules_path_fixed_price"] = path_fixed
    out["rules_path_range"] = path_range
    if not path.exists():
        return out
    try:
        payload = load_json(path)
    except Exception:
        return out
    poly = payload.get("polymarket_constraints") if isinstance(payload.get("polymarket_constraints"), dict) else {}
    if isinstance(poly, dict):
        if poly.get("buy_price") is not None:
            out["rules_buy_price"] = round(float(poly["buy_price"]), 2)
        if isinstance(poly.get("buy_price_range"), list) and len(poly["buy_price_range"]) == 2:
            out["rules_buy_price_range"] = [round(float(poly["buy_price_range"][0]), 2), round(float(poly["buy_price_range"][1]), 2)]
    return out


def inspect_prediction_file(path: Path | None, symbol: str) -> dict[str, Any]:
    asset_key = f"{symbol.upper()}_USDT_15m"
    out = {
        "prediction_file_exists": False,
        "prediction_file_age_sec": None,
        "prediction_file_fresh": False,
        "prediction_source_file": None,
        "prediction_source_age_sec": None,
        "prediction_source_stale": None,
        "prediction_payload_present": False,
        "prediction_payload_symbol_key": asset_key,
    }
    if not path or not path.exists():
        return out
    out["prediction_file_exists"] = True
    out["prediction_file_age_sec"] = round(time.time() - path.stat().st_mtime, 3)
    out["prediction_file_fresh"] = bool(out["prediction_file_age_sec"] <= 1800.0)
    try:
        payload = load_json(path)
    except Exception:
        return out
    if not isinstance(payload, dict):
        return out
    out["prediction_source_file"] = payload.get("source_file")
    out["prediction_source_age_sec"] = payload.get("source_age_sec")
    if "source_stale" in payload:
        out["prediction_source_stale"] = bool(payload.get("source_stale"))
    predictions = payload.get("predictions")
    if isinstance(predictions, dict):
        cell = predictions.get(asset_key)
        out["prediction_payload_present"] = isinstance(cell, dict) and bool(cell)
    return out


def resolve_prediction_suffix(row: dict[str, Any], symbol: str) -> tuple[str | None, bool]:
    by_symbol = row.get("predictionSuffixBySymbol") if isinstance(row.get("predictionSuffixBySymbol"), dict) else {}
    if symbol in by_symbol and str(by_symbol.get(symbol) or "").strip():
        return str(by_symbol[symbol]).strip(), False
    suffix = str(row.get("predictionSuffix") or "").strip()
    return (suffix or None), True


def resolve_symbols(row: dict[str, Any]) -> list[str]:
    out = set(parse_allowed_markets(row.get("allowedMarkets")))
    by_symbol = row.get("predictionSuffixBySymbol") if isinstance(row.get("predictionSuffixBySymbol"), dict) else {}
    out.update(str(k).strip().upper() for k in by_symbol.keys() if str(k).strip())
    return sorted(x for x in out if x)


def price_mode_from_config(row: dict[str, Any]) -> str:
    if row.get("limitPriceLadder"):
        return "dynamic_ladder"
    if row.get("limitPrice") is not None:
        return "fixed_limit"
    if row.get("usePredictionLimitPrice") is True:
        return "prediction_limit"
    return "unknown"


def price_range_from_ladder(ladder: str | None) -> list[float] | None:
    if not ladder:
        return None
    vals = [round(float(x.strip()), 2) for x in str(ladder).split(",") if str(x).strip()]
    if not vals:
        return None
    return [min(vals), max(vals)]


def build_row_truth(profile: str, row: dict[str, Any], symbol: str) -> dict[str, Any]:
    name = str(row.get("name") or "")
    suffix, used_fallback = resolve_prediction_suffix(row, symbol)
    pred_path = POLY / f"predictions{suffix}.json" if suffix else None
    pred_truth = inspect_prediction_file(pred_path, symbol)
    pred_exists = bool(pred_truth["prediction_file_exists"])
    pred_age = pred_truth["prediction_file_age_sec"]
    pred_fresh = bool(pred_truth["prediction_file_fresh"])
    rules_path_raw = str(row.get("rulesJsonPath") or "").strip()
    rules_path = (ROOT / rules_path_raw) if rules_path_raw and not Path(rules_path_raw).is_absolute() else (Path(rules_path_raw) if rules_path_raw else None)
    rules_truth = parse_rules_price(rules_path)

    name_mode, name_fixed, name_range = parse_name_price(name)
    config_mode = price_mode_from_config(row)
    config_fixed = round(float(row["limitPrice"]), 2) if row.get("limitPrice") is not None else None
    config_range = price_range_from_ladder(row.get("limitPriceLadder"))
    issues: list[str] = []

    if name_mode == "fixed" and config_fixed is not None and name_fixed is not None and config_fixed != name_fixed:
        issues.append(f"name_config_fixed_mismatch:{name_fixed}!={config_fixed}")
    if name_mode == "dynamic" and config_range is not None and name_range is not None and config_range != name_range:
        issues.append(f"name_config_range_mismatch:{name_range}!={config_range}")
    if config_fixed is not None and rules_truth["rules_buy_price"] is not None and config_fixed != rules_truth["rules_buy_price"]:
        issues.append(f"config_rules_buy_mismatch:{config_fixed}!={rules_truth['rules_buy_price']}")
    if config_range is not None and rules_truth["rules_buy_price_range"] is not None and config_range != rules_truth["rules_buy_price_range"]:
        issues.append(f"config_rules_range_mismatch:{config_range}!={rules_truth['rules_buy_price_range']}")
    if name_mode and rules_truth["rules_path_price_mode"] and name_mode != rules_truth["rules_path_price_mode"]:
        issues.append(f"name_rules_path_mode_mismatch:{name_mode}!={rules_truth['rules_path_price_mode']}")
    prediction_runtime_state = "ok"
    if not pred_exists:
        issues.append("prediction_file_missing")
        prediction_runtime_state = "missing"
    elif not pred_fresh:
        issues.append("prediction_file_stale")
        prediction_runtime_state = "stale"
    elif pred_truth["prediction_source_stale"] is True:
        issues.append("prediction_source_stale")
        prediction_runtime_state = "source_stale"
    elif not pred_truth["prediction_payload_present"]:
        # Fresh-but-empty predictions are a runtime state, not necessarily a bug:
        # 70-side filtering and per-bar abstention can legitimately produce no
        # payload for a given symbol while files remain healthy.
        prediction_runtime_state = "empty_but_fresh"
    if used_fallback and isinstance(row.get("predictionSuffixBySymbol"), dict) and row.get("predictionSuffixBySymbol"):
        prediction_runtime_state = "base_suffix_fallback"
    if rules_path and not rules_path.exists():
        issues.append("rules_path_missing")

    raw_bet_pct_normal = row.get("betPctNormal")
    raw_bet_pct_conservative = row.get("betPctConservative")
    try:
        effective_bet_pct_normal = min(float(raw_bet_pct_normal), 0.05) if raw_bet_pct_normal is not None else 0.05
    except Exception:
        effective_bet_pct_normal = 0.05
    try:
        effective_bet_pct_conservative = min(float(raw_bet_pct_conservative), 0.03) if raw_bet_pct_conservative is not None else 0.03
    except Exception:
        effective_bet_pct_conservative = 0.03

    return {
        "profile": profile,
        "name": name,
        "group": row.get("group"),
        "symbol": symbol,
        "prediction_suffix": suffix,
        "prediction_suffix_fallback_used": used_fallback,
        "prediction_runtime_state": prediction_runtime_state,
        "prediction_file": str(pred_path) if pred_path else None,
        "prediction_file_exists": pred_exists,
        "prediction_file_age_sec": pred_age,
        "prediction_file_fresh": pred_fresh,
        "prediction_source_file": pred_truth["prediction_source_file"],
        "prediction_source_age_sec": pred_truth["prediction_source_age_sec"],
        "prediction_source_stale": pred_truth["prediction_source_stale"],
        "prediction_payload_present": pred_truth["prediction_payload_present"],
        "prediction_payload_symbol_key": pred_truth["prediction_payload_symbol_key"],
        "resolved_runtime_price_mode": config_mode,
        "config_limit_price": config_fixed,
        "config_limit_price_ladder": row.get("limitPriceLadder"),
        "config_ladder_range": config_range,
        "config_usePredictionLimitPrice": row.get("usePredictionLimitPrice"),
        "name_price_mode": name_mode,
        "name_fixed_price": name_fixed,
        "name_price_range": name_range,
        "rules_path": str(rules_path) if rules_path else None,
        "rules_path_exists": rules_truth["path_exists"],
        "rules_buy_price": rules_truth["rules_buy_price"],
        "rules_buy_price_range": rules_truth["rules_buy_price_range"],
        "rules_path_price_mode": rules_truth["rules_path_price_mode"],
        "rules_path_fixed_price": rules_truth["rules_path_fixed_price"],
        "rules_path_range": rules_truth["rules_path_range"],
        "risk_controls": {k: row.get(k) for k in RISK_KEYS if k in row},
        "effective_online_sim_risk_caps": {
            "betPctNormal": round(float(effective_bet_pct_normal), 6),
            "betPctConservative": round(float(effective_bet_pct_conservative), 6),
        },
        "mode_controls": {k: row.get(k) for k in MODE_KEYS if k in row},
        "issues": issues,
    }


def main() -> None:
    payload: dict[str, Any] = {
        "generatedAt": now_iso(),
        "runtime_defaults": RUNTIME_DEFAULTS,
        "profiles": {},
    }
    overall_issue_counts: dict[str, int] = {}

    for profile, meta in PROFILE_MAP.items():
        rows = load_json(meta["config"])
        active = active_names(meta["active"])
        profile_rows = []
        issue_counts: dict[str, int] = {}
        exp_rows = [r for r in rows if isinstance(r, dict) and str(r.get("name") or "").startswith("v5_exp") and str(r.get("name") or "") in active]
        for row in exp_rows:
            for symbol in resolve_symbols(row):
                truth = build_row_truth(profile, row, symbol)
                profile_rows.append(truth)
                for issue in truth["issues"]:
                    issue_counts[issue] = issue_counts.get(issue, 0) + 1
                    overall_issue_counts[issue] = overall_issue_counts.get(issue, 0) + 1
        payload["profiles"][profile] = {
            "active_exp_trader_count": len(exp_rows),
            "active_exp_cell_count": len(profile_rows),
            "issue_counts": dict(sorted(issue_counts.items())),
            "rows": profile_rows,
        }

    payload["overall_issue_counts"] = dict(sorted(overall_issue_counts.items()))
    payload["ok"] = not bool(overall_issue_counts)
    dump_json(REPORT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
