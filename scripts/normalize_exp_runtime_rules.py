#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
POLY = ROOT / "polymarket"
OUT_DIR = POLY / "normalized_runtime_rules"
CONFIGS = [
    POLY / "trader_configs.json",
    POLY / "trader_configs_70.json",
]

TARGETS = {
    "v5_exp10_bp0460": {
        "source": ROOT / "experiments/sentiment_grid_search/results/v5_production_sim_noise/optimal_trading_rules_v3_bp0460.json",
        "fixed_price": 0.46,
    },
    "v5_exp15_bp0510": {
        "source": ROOT / "experiments/sentiment_grid_search/results/v5_production_365d/optimal_trading_rules_v3_bp0510.json",
        "fixed_price": 0.51,
    },
    "v5_exp15_bp0520": {
        "source": ROOT / "experiments/sentiment_grid_search/results/v5_production_365d/optimal_trading_rules_v3_bp0520.json",
        "fixed_price": 0.52,
    },
    "v5_exp16_bp0500": {
        "source": ROOT / "experiments/sentiment_grid_search/results/v5_production_tv_365d/optimal_trading_rules_v3_bp0500.json",
        "fixed_price": 0.50,
    },
    "v5_exp10_bp_dyn_0480_0510": {
        "source": ROOT / "experiments/sentiment_grid_search/results/v5_production_sim_noise/optimal_trading_rules_v3_bp_dyn_0480_0510.json",
        "dynamic_range": [0.48, 0.51],
    },
    "v5_exp11_bp_dyn_0480_0510": {
        "source": ROOT / "experiments/sentiment_grid_search/results/v5_production_sim_noise/optimal_trading_rules_v3_bp_dyn_0480_0510.json",
        "dynamic_range": [0.48, 0.51],
    },
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def update_price_fields(payload: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(payload))
    poly = out.get("polymarket_constraints")
    if not isinstance(poly, dict):
        poly = {}
    if "fixed_price" in spec:
        price = round(float(spec["fixed_price"]), 2)
        poly["buy_price"] = price
        poly["buy_price_range"] = None
        poly["odds"] = round((1.0 - price) / price, 4)
    else:
        lo, hi = [round(float(x), 2) for x in spec["dynamic_range"]]
        mid = round((lo + hi) / 2.0, 4)
        poly["buy_price"] = mid
        poly["buy_price_range"] = [lo, hi]
        poly["odds"] = round((1.0 - mid) / mid, 4)
    out["polymarket_constraints"] = poly
    meta = out.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
    meta["runtime_truth_normalized"] = True
    meta["runtime_truth_source_file"] = str(spec["source"])
    out["metadata"] = meta
    return out


def rewrite_configs(normalized_map: dict[str, Path]) -> None:
    for cfg_path in CONFIGS:
        rows = load_json(cfg_path)
        changed = 0
        for row in rows:
            name = str(row.get("name") or "")
            out_path = normalized_map.get(name)
            if out_path is None:
                continue
            rel = out_path.relative_to(ROOT)
            if str(row.get("rulesJsonPath") or "") != str(rel):
                row["rulesJsonPath"] = str(rel)
                changed += 1
        if changed:
            dump_json(cfg_path, rows)


def main() -> None:
    normalized_map: dict[str, Path] = {}
    summary: dict[str, Any] = {"normalized": []}
    for name, spec in TARGETS.items():
        source = Path(spec["source"])
        if not source.exists():
            raise FileNotFoundError(source)
        payload = load_json(source)
        normalized = update_price_fields(payload, spec)
        out_path = OUT_DIR / f"{name}.json"
        dump_json(out_path, normalized)
        normalized_map[name] = out_path
        summary["normalized"].append(
            {
                "name": name,
                "source": str(source),
                "out_path": str(out_path),
                "fixed_price": spec.get("fixed_price"),
                "dynamic_range": spec.get("dynamic_range"),
            }
        )
    rewrite_configs(normalized_map)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
