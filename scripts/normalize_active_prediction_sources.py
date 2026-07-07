#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POLY = ROOT / "polymarket"
REPORT = ROOT / "reports" / "normalize_active_prediction_sources_latest.json"

CONFIGS = {
    "default": {
        "config": POLY / "trader_configs.json",
        "active": POLY / "active_traders.json",
    },
    "70": {
        "config": POLY / "trader_configs_70.json",
        "active": POLY / "active_traders_70.json",
    },
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def active_names(path: Path) -> set[str]:
    payload = load_json(path)
    out = payload.get("traderNames") or payload.get("active_traders") or []
    return {str(x).strip() for x in out if str(x).strip()}


def parse_allowed_markets(raw: Any) -> list[str]:
    if isinstance(raw, list):
        vals = [str(x).strip().upper() for x in raw if str(x).strip()]
    else:
        vals = [part.strip().upper() for part in str(raw or "").split(",") if part.strip()]
    return sorted(set(x for x in vals if x))


def is_target_row(row: dict[str, Any], active: set[str]) -> bool:
    name = str(row.get("name") or "")
    if name not in active:
        return False
    if not name.startswith("v5_exp"):
        return False
    if name.startswith("cmp_") or "_cmp_" in name:
        return False
    return True


def inspect_prediction_file(path: Path | None, symbol: str) -> dict[str, Any]:
    asset_key = f"{symbol.upper()}_USDT_15m"
    out = {
        "exists": False,
        "age_sec": None,
        "fresh": False,
        "source_stale": None,
        "payload_present": False,
    }
    if not path or not path.exists():
        return out
    out["exists"] = True
    out["age_sec"] = round(time.time() - path.stat().st_mtime, 3)
    out["fresh"] = bool(out["age_sec"] <= 1800.0)
    try:
        payload = load_json(path)
    except Exception:
        return out
    if not isinstance(payload, dict):
        return out
    if "source_stale" in payload:
        out["source_stale"] = bool(payload.get("source_stale"))
    predictions = payload.get("predictions")
    if isinstance(predictions, dict):
        cell = predictions.get(asset_key)
        out["payload_present"] = isinstance(cell, dict) and bool(cell)
    return out


def materialize_symbol_prediction_from_base(base_path: Path, target_path: Path, symbol: str) -> None:
    asset_key = f"{symbol.upper()}_USDT_15m"
    payload = load_json(base_path)
    if not isinstance(payload, dict):
        return
    predictions = payload.get("predictions")
    if not isinstance(predictions, dict):
        return
    cell = predictions.get(asset_key)
    if not isinstance(cell, dict) or not cell:
        return
    out = json.loads(json.dumps(payload))
    out["predictions"] = {asset_key: cell}
    dump_json(target_path, out)


def resolve_symbols(row: dict[str, Any]) -> list[str]:
    out = set(parse_allowed_markets(row.get("allowedMarkets")))
    by_symbol = row.get("predictionSuffixBySymbol") if isinstance(row.get("predictionSuffixBySymbol"), dict) else {}
    out.update(str(k).strip().upper() for k in by_symbol.keys() if str(k).strip())
    return sorted(x for x in out if x)


def main() -> None:
    payload: dict[str, Any] = {"profiles": {}, "ok": True}
    for profile, meta in CONFIGS.items():
        rows = load_json(meta["config"])
        active = active_names(meta["active"])
        changed_rows: list[dict[str, Any]] = []
        rows_changed = 0
        symbols_changed = 0
        for row in rows:
            if not isinstance(row, dict) or not is_target_row(row, active):
                continue
            base_suffix = str(row.get("predictionSuffix") or "").strip()
            if not base_suffix:
                continue
            base_path = POLY / f"predictions{base_suffix}.json"
            by_symbol = row.get("predictionSuffixBySymbol") if isinstance(row.get("predictionSuffixBySymbol"), dict) else {}
            row_symbol_changes: list[dict[str, Any]] = []
            for symbol in resolve_symbols(row):
                current_suffix = str(by_symbol.get(symbol) or "").strip() or base_suffix
                current_path = POLY / f"predictions{current_suffix}.json" if current_suffix else None
                current_truth = inspect_prediction_file(current_path, symbol)
                base_truth = inspect_prediction_file(base_path, symbol)
                current_bad = (
                    (not current_truth["exists"])
                    or (not current_truth["fresh"])
                    or (current_truth["source_stale"] is True)
                    or (not current_truth["payload_present"])
                )
                base_good = (
                    base_truth["exists"]
                    and base_truth["fresh"]
                    and base_truth["source_stale"] is not True
                    and base_truth["payload_present"]
                )
                if current_suffix == base_suffix:
                    continue
                if current_bad and base_good:
                    current_target_path = POLY / f"predictions{current_suffix}.json"
                    materialize_symbol_prediction_from_base(base_path, current_target_path, symbol)
                    by_symbol[symbol] = base_suffix
                    symbols_changed += 1
                    row_symbol_changes.append(
                        {
                            "symbol": symbol,
                            "old_suffix": current_suffix,
                            "new_suffix": base_suffix,
                            "refreshed_legacy_target": str(current_target_path),
                            "reason": {
                                "current_exists": current_truth["exists"],
                                "current_fresh": current_truth["fresh"],
                                "current_source_stale": current_truth["source_stale"],
                                "current_payload_present": current_truth["payload_present"],
                                "base_exists": base_truth["exists"],
                                "base_fresh": base_truth["fresh"],
                                "base_source_stale": base_truth["source_stale"],
                                "base_payload_present": base_truth["payload_present"],
                            },
                        }
                    )
            if row_symbol_changes:
                row["predictionSuffixBySymbol"] = by_symbol
                rows_changed += 1
                changed_rows.append(
                    {
                        "name": row.get("name"),
                        "changes": row_symbol_changes,
                    }
                )
        if rows_changed:
            dump_json(meta["config"], rows)
        payload["profiles"][profile] = {
            "config": str(meta["config"]),
            "rows_changed": rows_changed,
            "symbols_changed": symbols_changed,
            "changed_rows": changed_rows,
        }
    payload["ok"] = True
    dump_json(REPORT, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
