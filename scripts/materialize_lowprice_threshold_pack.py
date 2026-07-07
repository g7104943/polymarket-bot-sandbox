#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POLY = ROOT / "polymarket"
REPORTS = ROOT / "reports"

SOURCE_CONFIGS = {
    "default": POLY / "trader_configs_monitor_only_lowprice.json",
    "70": POLY / "trader_configs_monitor_only_lowprice_70.json",
}
SOURCE_MONITORS = {
    "default": POLY / "monitor_only_traders_lowprice.json",
    "70": POLY / "monitor_only_traders_lowprice_70.json",
}
OUT_CONFIG = POLY / "trader_configs_monitor_only_lowprice_threshold_pack.json"
OUT_ACTIVE = POLY / "active_traders_monitor_only_lowprice_threshold_pack.json"
OUT_MONITOR = POLY / "monitor_only_traders_lowprice_threshold_pack.json"
OUT_RULES_DIR = POLY / "lowprice_rules_selected" / "threshold_pack"
OUT_REPORT = REPORTS / "lowprice_threshold_pack_materialized_latest.json"

OUT_PROFILE = "threshold_pack"
OUT_GROUP = "lowprice_threshold_pack_selected"
OUT_LOGS_PREFIX = "logs_threshold_pack_lowprice_selected_"

SOURCE_SPECS = [
    {"source_profile": "70", "name": "v5_exp10_bp_dyn_0300_0440_eth"},
    {"source_profile": "default", "name": "v5_exp10_bp0310_eth_src0480"},
    {"source_profile": "default", "name": "v5_exp10_bp0300_eth_src0480"},
]
VARIANTS: list[tuple[str, float]] = [
    ("rel075", 0.50504),
    ("rel080", 0.51004),
    ("abs070", 0.70),
    ("abs075", 0.75),
    ("abs080", 0.80),
]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def deep_copy(payload: Any) -> Any:
    return json.loads(json.dumps(payload))


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def _load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path)
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    traders = payload.get("traders") if isinstance(payload, dict) else None
    if isinstance(traders, list):
        return [row for row in traders if isinstance(row, dict)]
    return []


def _load_source_lookup(paths: dict[str, Path]) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for profile, path in paths.items():
        for row in _load_manifest_rows(path):
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            lookup[(profile, name)] = row
    return lookup


def _source_rules_path(config_row: dict[str, Any]) -> Path:
    raw = str(config_row.get("rulesJsonPath") or "").strip()
    if not raw:
        raise RuntimeError(f"missing rulesJsonPath for {config_row.get('name')}")
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _variant_name(base_name: str, variant_key: str) -> str:
    return f"{base_name}_{variant_key}"


def _variant_rules_path(base_name: str, variant_key: str) -> Path:
    return OUT_RULES_DIR / f"{base_name}_{variant_key}.json"


def _materialize_rules(
    source_row: dict[str, Any],
    source_profile: str,
    variant_name: str,
    variant_key: str,
    threshold: float,
) -> str:
    source_rules = load_json(_source_rules_path(source_row))
    payload = deep_copy(source_rules if isinstance(source_rules, dict) else {})
    trading_rules = payload.get("trading_rules") if isinstance(payload.get("trading_rules"), dict) else {}
    trading_rules = dict(trading_rules)
    trading_rules["min_confidence"] = round(float(threshold), 5)
    payload["trading_rules"] = trading_rules

    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    metadata = dict(metadata)
    metadata.update(
        {
            "profile": OUT_PROFILE,
            "generated_at": now_iso(),
            "threshold_pack_source_profile": source_profile,
            "threshold_pack_source_name": str(source_row.get("name") or ""),
            "threshold_pack_variant": variant_key,
            "threshold_pack_prob_threshold": round(float(threshold), 5),
            "threshold_pack_trader_name": variant_name,
        }
    )
    payload["metadata"] = metadata

    out_path = _variant_rules_path(str(source_row.get("name") or ""), variant_key)
    dump_json(out_path, payload)
    return _relative(out_path)


def _materialize_config_row(
    source_row: dict[str, Any],
    source_profile: str,
    variant_key: str,
    threshold: float,
) -> dict[str, Any]:
    row = deep_copy(source_row)
    name = _variant_name(str(source_row.get("name") or ""), variant_key)
    row["name"] = name
    row["profile"] = OUT_PROFILE
    row["group"] = OUT_GROUP
    row["logsDir"] = f"{OUT_LOGS_PREFIX}{name}"
    row["probThreshold"] = round(float(threshold), 5)
    row["initialCapital"] = 400.0
    if isinstance(row.get("jointOptimizationBaseParams"), dict):
        row["jointOptimizationBaseParams"] = deep_copy(row["jointOptimizationBaseParams"])
        row["jointOptimizationBaseParams"]["probThreshold"] = round(float(threshold), 5)
    row["rulesJsonPath"] = _materialize_rules(
        source_row=source_row,
        source_profile=source_profile,
        variant_name=name,
        variant_key=variant_key,
        threshold=threshold,
    )
    return row


def _materialize_monitor_row(
    source_row: dict[str, Any],
    variant_key: str,
) -> dict[str, Any]:
    row = deep_copy(source_row)
    name = _variant_name(str(source_row.get("name") or ""), variant_key)
    row["name"] = name
    row["profile"] = OUT_PROFILE
    row["group"] = OUT_GROUP
    row["logsDir"] = f"{OUT_LOGS_PREFIX}{name}"
    row["initialCapital"] = 400.0
    return row


def main() -> None:
    config_lookup = _load_source_lookup(SOURCE_CONFIGS)
    monitor_lookup = _load_source_lookup(SOURCE_MONITORS)

    config_rows: list[dict[str, Any]] = []
    monitor_rows: list[dict[str, Any]] = []
    source_rows_summary: list[dict[str, Any]] = []
    for spec in SOURCE_SPECS:
        source_profile = str(spec["source_profile"])
        source_name = str(spec["name"])
        source_cfg = config_lookup.get((source_profile, source_name))
        source_monitor = monitor_lookup.get((source_profile, source_name))
        if not isinstance(source_cfg, dict):
            raise RuntimeError(f"missing source config row: {source_profile}:{source_name}")
        if not isinstance(source_monitor, dict):
            raise RuntimeError(f"missing source monitor row: {source_profile}:{source_name}")
        variants_summary: list[dict[str, Any]] = []
        for variant_key, threshold in VARIANTS:
            config_rows.append(
                _materialize_config_row(
                    source_row=source_cfg,
                    source_profile=source_profile,
                    variant_key=variant_key,
                    threshold=threshold,
                )
            )
            monitor_rows.append(
                _materialize_monitor_row(
                    source_row=source_monitor,
                    variant_key=variant_key,
                )
            )
            variants_summary.append(
                {
                    "variant": variant_key,
                    "probThreshold": round(float(threshold), 5),
                    "traderName": _variant_name(source_name, variant_key),
                }
            )
        source_rows_summary.append(
            {
                "source_profile": source_profile,
                "source_name": source_name,
                "variants": variants_summary,
            }
        )

    config_rows.sort(key=lambda row: str(row.get("name") or ""))
    monitor_rows.sort(key=lambda row: str(row.get("name") or ""))
    active_names = [str(row.get("name") or "") for row in config_rows]

    dump_json(OUT_CONFIG, config_rows)
    dump_json(
        OUT_ACTIVE,
        {
            "generatedAt": now_iso(),
            "scope": "lowprice_threshold_pack_experiment",
            "source": "materialize_lowprice_threshold_pack.py",
            "profile": OUT_PROFILE,
            "traderNames": active_names,
        },
    )
    dump_json(
        OUT_MONITOR,
        {
            "generatedAt": now_iso(),
            "scope": "lowprice_threshold_pack_experiment",
            "profile": OUT_PROFILE,
            "group": OUT_GROUP,
            "source": "materialize_lowprice_threshold_pack.py",
            "traders": monitor_rows,
        },
    )
    dump_json(
        OUT_REPORT,
        {
            "generatedAt": now_iso(),
            "status": "ok",
            "profile": OUT_PROFILE,
            "group": OUT_GROUP,
            "trader_count": len(config_rows),
            "monitor_count": len(monitor_rows),
            "active_count": len(active_names),
            "source_rows": source_rows_summary,
            "out_config": _relative(OUT_CONFIG),
            "out_active": _relative(OUT_ACTIVE),
            "out_monitor": _relative(OUT_MONITOR),
        },
    )


if __name__ == "__main__":
    main()
