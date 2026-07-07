#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POLY = ROOT / "polymarket"

CONFIGS = {
    "default": {
        "config": POLY / "trader_configs.json",
        "active": POLY / "active_traders.json",
        "profile_suffix": "",
    },
    "70": {
        "config": POLY / "trader_configs_70.json",
        "active": POLY / "active_traders_70.json",
        "profile_suffix": "_70",
    },
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def active_names(path: Path) -> set[str]:
    payload = load_json(path)
    names = payload.get("traderNames") if isinstance(payload, dict) else []
    return {str(x).strip() for x in names if str(x).strip()}


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


def main() -> None:
    summary: dict[str, Any] = {"profiles": {}, "ok": True}
    for profile, meta in CONFIGS.items():
        rows = load_json(meta["config"])
        active = active_names(meta["active"])
        changed_rows: list[dict[str, Any]] = []
        changed = 0
        filled = 0
        for row in rows:
            if not isinstance(row, dict) or not is_target_row(row, active):
                continue
            base_suffix = str(row.get("predictionSuffix") or "").strip()
            if not base_suffix:
                continue
            allowed = parse_allowed_markets(row.get("allowedMarkets"))
            if not allowed:
                continue
            by_symbol = row.get("predictionSuffixBySymbol")
            if not isinstance(by_symbol, dict):
                by_symbol = {}
            row_changed = False
            added_symbols: list[str] = []
            for symbol in allowed:
                current = str(by_symbol.get(symbol) or "").strip()
                if current:
                    continue
                by_symbol[symbol] = base_suffix
                row_changed = True
                filled += 1
                added_symbols.append(symbol)
            if row_changed:
                row["predictionSuffixBySymbol"] = by_symbol
                changed += 1
                changed_rows.append(
                    {
                        "name": row.get("name"),
                        "allowedMarkets": allowed,
                        "predictionSuffix": base_suffix,
                        "addedSymbols": added_symbols,
                    }
                )
        if changed:
            dump_json(meta["config"], rows)
        summary["profiles"][profile] = {
            "config": str(meta["config"]),
            "rows_changed": changed,
            "symbols_filled": filled,
            "changed_rows": changed_rows,
        }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
