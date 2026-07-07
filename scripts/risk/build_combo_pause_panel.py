#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLYMARKET_DIR = PROJECT_ROOT / "polymarket"
REPORTS_DIR = PROJECT_ROOT / "reports"

ACTIVE_DEFAULT = POLYMARKET_DIR / "active_traders.json"
ACTIVE_70 = POLYMARKET_DIR / "active_traders_70.json"
CFG_DEFAULT = POLYMARKET_DIR / "trader_configs.json"
CFG_70 = POLYMARKET_DIR / "trader_configs_70.json"
DEFAULT_OUTPUT = REPORTS_DIR / "combo_pause_v1_panel.parquet"
DEFAULT_META = REPORTS_DIR / "combo_pause_v1_panel_meta.json"

TARGET_SYMBOLS = {"BTC", "ETH"}
TARGET_DIRECTIONS = {"UP", "DOWN"}


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


def _config_map(path: Path) -> Dict[str, Dict[str, Any]]:
    obj = _load_json(path)
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(obj, list):
        return out
    for row in obj:
        if isinstance(row, dict) and row.get("name"):
            out[str(row["name"])] = row
    return out


def _parse_ts(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _allowed_symbols(cfg: Dict[str, Any]) -> List[str]:
    raw = str(cfg.get("allowedMarkets") or "")
    out = [str(x).strip().upper() for x in raw.split(",") if str(x).strip()]
    return [x for x in out if x in TARGET_SYMBOLS]


def _resolve_base_capital(cfg: Dict[str, Any]) -> float:
    total = float(_to_float(cfg.get("initialCapital")) or 0.0)
    if total <= 0:
        return 0.0
    if bool(cfg.get("perCoinCapital")):
        n = int(_to_float(cfg.get("numTradingAssets")) or len(_allowed_symbols(cfg)) or 1)
        return total / max(1, n)
    return total


def _iter_rows(profile: str, active_names: List[str], cfg_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for trader_name in active_names:
        cfg = cfg_map.get(trader_name)
        if not isinstance(cfg, dict):
            continue
        logs_dir = str(cfg.get("logsDir") or "").strip()
        if not logs_dir:
            continue
        trade_file = POLYMARKET_DIR / logs_dir / "prediction_trades.simulation.json"
        if not trade_file.exists():
            continue
        try:
            trades = _load_json(trade_file)
        except Exception:
            continue
        if not isinstance(trades, list):
            continue
        allowed = set(_allowed_symbols(cfg))
        if not allowed:
            continue
        base_capital = _resolve_base_capital(cfg)
        for row in trades:
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("symbol") or "").replace("/USDT", "").replace("/USD", "").strip().upper()
            direction = str(row.get("direction") or "").strip().upper()
            if symbol not in TARGET_SYMBOLS or symbol not in allowed:
                continue
            if direction not in TARGET_DIRECTIONS:
                continue
            result = str(row.get("result") or "").strip().lower()
            if result not in {"win", "lose"}:
                continue
            pnl = _to_float(row.get("pnl"))
            if pnl is None:
                continue
            ts = _parse_ts(row.get("settledAt")) or _parse_ts(row.get("timestamp"))
            if ts is None:
                continue
            rows.append(
                {
                    "timestamp": ts.isoformat(),
                    "ts": int(ts.timestamp()),
                    "profile": profile,
                    "traderName": trader_name,
                    "symbol": symbol,
                    "direction": direction,
                    "cluster_id": f"{profile}_{symbol}_{direction}",
                    "cell_id": f"{profile}::{trader_name}::{symbol}::{direction}",
                    "pnl": float(pnl),
                    "win": 1 if result == "win" else 0,
                    "base_capital": float(base_capital),
                    "logsDir": logs_dir,
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build combo directional pause panel from settled simulation trades")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--meta-output", type=Path, default=DEFAULT_META)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    active_default = _parse_active(ACTIVE_DEFAULT)
    active_70 = _parse_active(ACTIVE_70)
    cfg_default = _config_map(CFG_DEFAULT)
    cfg_70 = _config_map(CFG_70)

    rows = []
    rows.extend(_iter_rows("default", active_default, cfg_default))
    rows.extend(_iter_rows("70", active_70, cfg_70))
    if not rows:
        raise SystemExit("combo panel rows empty")

    df = pd.DataFrame(rows).sort_values(["profile", "traderName", "symbol", "direction", "ts"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)

    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output": str(args.output),
        "rows": int(len(df)),
        "traders": int(df[["profile", "traderName"]].drop_duplicates().shape[0]),
        "cells": int(df["cell_id"].nunique()),
        "clusters": sorted(str(x) for x in df["cluster_id"].dropna().astype(str).unique().tolist()),
        "ts_min": datetime.fromtimestamp(int(df["ts"].min()), tz=timezone.utc).isoformat(),
        "ts_max": datetime.fromtimestamp(int(df["ts"].max()), tz=timezone.utc).isoformat(),
    }
    args.meta_output.parent.mkdir(parents=True, exist_ok=True)
    args.meta_output.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)
    print(args.meta_output)
    print(json.dumps({"rows": meta["rows"], "cells": meta["cells"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
