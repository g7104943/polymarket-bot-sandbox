#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
POLYMARKET_DIR = PROJECT_ROOT / "polymarket"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "sentiment_grid_search" / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"

ACTIVE_DEFAULT = POLYMARKET_DIR / "active_traders.json"
ACTIVE_70 = POLYMARKET_DIR / "active_traders_70.json"
CFG_DEFAULT = POLYMARKET_DIR / "trader_configs.json"
CFG_70 = POLYMARKET_DIR / "trader_configs_70.json"

DEFAULT_DOWN_DELTA = 0.04
GROUP_DOWN_DELTA_MAP = {
    "v5_exp10": 0.09,
    "v5_exp11": 0.09,
    "v5_exp13": 0.06,
    "v5_exp14": 0.04,
    "v5_exp15": 0.06,
    "v5_exp16": 0.02,
    "v5_exp17": 0.05,
    "ensemble": 0.08,
    "gru_all": 0.09,
}

DIRECT_FAMILY_FILES = {
    "v5_production_tv_365d": "v5_production_tv_365d_test_predictions_td365d.parquet",
    "v5_production_365d": "v5_production_365d_test_predictions_td365d.parquet",
    "v5_production_no_target_pm": "v5_production_no_target_pm_test_predictions_td365d.parquet",
    "v5_production_sim_noise": "v5_production_sim_noise_noisy_test_predictions_td365d.parquet",
    "v5_production_tv": "v5_production_tv_test_predictions_td365d.parquet",
}

PROXY_FAMILY_MAP = {
    "v5_production_sim_noise_tv": "v5_production_sim_noise",
    "ensemble": "v5_production_tv_365d",
}

TARGET_SYMBOLS = {"BTC", "ETH"}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_symbol(value: Any) -> str:
    return str(value or "").replace("/USDT", "").replace("/USD", "").strip().upper()


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


def _family_from_rules_path(cfg: Dict[str, Any]) -> str:
    p = str(cfg.get("rulesJsonPath") or "")
    if "results/" in p:
        try:
            return p.split("results/")[1].split("/")[0]
        except Exception:
            return ""
    return ""


def _load_rules_min_conf(cfg: Dict[str, Any]) -> Optional[float]:
    rules_path = cfg.get("rulesJsonPath")
    if not rules_path:
        return None
    p = PROJECT_ROOT / str(rules_path)
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        tr = obj.get("trading_rules") if isinstance(obj, dict) else None
        if isinstance(tr, dict) and tr.get("min_confidence") is not None:
            return float(tr.get("min_confidence"))
    except Exception:
        return None
    return None


def _resolve_up_threshold(cfg: Dict[str, Any]) -> float:
    rules_min_conf = _load_rules_min_conf(cfg)
    base_prob = cfg.get("probThreshold")
    if base_prob is None:
        base_prob = rules_min_conf if rules_min_conf is not None else 0.55
    base_prob = float(base_prob)
    prob_up = cfg.get("probThresholdUp")
    if prob_up is None:
        prob_up = base_prob
    prob_up = float(prob_up)
    return max(0.01, min(0.999, prob_up))


def _resolve_down_threshold(cfg: Dict[str, Any]) -> float:
    rules_min_conf = _load_rules_min_conf(cfg)
    base_prob = cfg.get("probThreshold")
    if base_prob is None:
        base_prob = rules_min_conf if rules_min_conf is not None else 0.55
    base_prob = float(base_prob)
    down_delta = cfg.get("downThresholdDelta")
    if down_delta is None:
        group = str(cfg.get("group") or "")
        base_group = group[:-3] if group.endswith("_70") else group
        down_delta = GROUP_DOWN_DELTA_MAP.get(group)
        if down_delta is None:
            down_delta = GROUP_DOWN_DELTA_MAP.get(base_group, DEFAULT_DOWN_DELTA)
    down_delta = float(down_delta)
    prob_down = cfg.get("probThresholdDown")
    if prob_down is None:
        prob_down = max(0.01, min(0.50, 1 - base_prob - max(0.0, down_delta)))
    prob_down = float(prob_down)
    return max(0.01, min(0.999, 1.0 - prob_down))


def _compute_up_pnl_proxy(d: pd.DataFrame) -> pd.Series:
    up_conf = d["up_conf"].clip(lower=0.05, upper=0.95)
    pnl_base = np.where(d["up_win"].astype(int).values == 1, (1.0 / up_conf.values) - 1.0, -1.0)
    med_abs = float(d["log_return"].abs().median())
    if not np.isfinite(med_abs) or med_abs <= 1e-8:
        med_abs = 0.001
    mag = (d["log_return"].abs() / med_abs).clip(lower=0.5, upper=3.0).values
    return pd.Series(pnl_base * mag, index=d.index)


def _compute_down_pnl_proxy(d: pd.DataFrame) -> pd.Series:
    down_conf = d["down_conf"].clip(lower=0.05, upper=0.95)
    pnl_base = np.where(d["down_win"].astype(int).values == 1, (1.0 / down_conf.values) - 1.0, -1.0)
    med_abs = float(d["log_return"].abs().median())
    if not np.isfinite(med_abs) or med_abs <= 1e-8:
        med_abs = 0.001
    mag = (d["log_return"].abs() / med_abs).clip(lower=0.5, upper=3.0).values
    return pd.Series(pnl_base * mag, index=d.index)


def _resolve_source_family(model_family: str) -> Tuple[Optional[str], str]:
    if model_family in DIRECT_FAMILY_FILES:
        return model_family, "direct"
    proxy = PROXY_FAMILY_MAP.get(model_family)
    if proxy:
        return proxy, "proxy"
    return None, "missing"


def _load_family_df_cache() -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for fam, file_name in DIRECT_FAMILY_FILES.items():
        p = RESULTS_DIR / file_name
        if not p.exists():
            continue
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        expected = {"timestamp", "asset", "proba_up", "actual", "log_return"}
        if not expected.issubset(set(df.columns)):
            continue
        df = df[["timestamp", "asset", "proba_up", "actual", "log_return"]].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp", "proba_up", "log_return"])
        df["asset"] = df["asset"].astype(str).str.upper()
        out[fam] = df.sort_values("timestamp")
    return out


def _append_direction_rows(
    base_df: pd.DataFrame,
    profile: str,
    trader: str,
    symbol: str,
    model_family: str,
    source_family: str,
    source_kind: str,
    direction: str,
    threshold: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if direction == "UP":
        d = base_df[base_df["trigger_up"]].copy()
        d["confidence"] = d["up_conf"]
        d["win"] = d["up_win"]
        d["pnl"] = d["up_pnl_proxy"]
    else:
        d = base_df[base_df["trigger_down"]].copy()
        d["confidence"] = d["down_conf"]
        d["win"] = d["down_win"]
        d["pnl"] = d["down_pnl_proxy"]
    if d.empty:
        return rows

    d["profile"] = profile
    d["traderName"] = trader
    d["symbol"] = symbol
    d["direction"] = direction
    d["model_family"] = model_family
    d["source_family"] = source_family
    d["source_kind"] = source_kind
    d["conf_threshold"] = threshold
    d["cell_id"] = d["profile"] + "::" + d["traderName"] + "::" + d["symbol"] + "::" + d["direction"]

    for r in d[
        [
            "timestamp",
            "profile",
            "traderName",
            "symbol",
            "direction",
            "cell_id",
            "model_family",
            "source_family",
            "source_kind",
            "confidence",
            "conf_threshold",
            "win",
            "pnl",
            "log_return",
        ]
    ].to_dict("records"):
        ts = pd.Timestamp(r["timestamp"])
        rows.append(
            {
                "timestamp": ts.to_pydatetime().astimezone(timezone.utc).isoformat(),
                "ts": int(ts.timestamp()),
                "profile": r["profile"],
                "traderName": r["traderName"],
                "symbol": r["symbol"],
                "direction": r["direction"],
                "cell_id": r["cell_id"],
                "model_family": r["model_family"],
                "source_family": r["source_family"],
                "source_kind": r["source_kind"],
                "confidence": float(r["confidence"]),
                "conf_threshold": float(r["conf_threshold"]),
                "win": int(r["win"]),
                "pnl": float(r["pnl"]),
                "log_return": float(r["log_return"]),
            }
        )
    return rows


def _build_rows_for_cfg(
    profile: str,
    trader: str,
    cfg: Dict[str, Any],
    family_dfs: Dict[str, pd.DataFrame],
) -> Tuple[List[Dict[str, Any]], Dict[str, int], List[str]]:
    rows: List[Dict[str, Any]] = []
    warn: List[str] = []
    stats = {
        "raw": 0,
        "triggered": 0,
        "missing_source": 0,
        "actual_missing": 0,
        "triggered_proxy": 0,
    }

    model_family = _family_from_rules_path(cfg)
    source_family, source_kind = _resolve_source_family(model_family)
    if not source_family or source_family not in family_dfs:
        stats["missing_source"] += 1
        warn.append(f"{profile}:{trader} family={model_family or '<none>'} missing source")
        return rows, stats, warn

    allowed = [_normalize_symbol(x) for x in str(cfg.get("allowedMarkets") or "").split(",")]
    allowed = [x for x in allowed if x in TARGET_SYMBOLS]
    if not allowed:
        return rows, stats, warn

    df_source = family_dfs[source_family]
    up_thr = _resolve_up_threshold(cfg)
    down_thr = _resolve_down_threshold(cfg)

    for symbol in allowed:
        asset = f"{symbol}_USDT"
        d = df_source[df_source["asset"] == asset].copy()
        if d.empty:
            warn.append(f"{profile}:{trader}:{symbol} source={source_family} no rows")
            continue

        d["proba_up"] = pd.to_numeric(d["proba_up"], errors="coerce")
        d["log_return"] = pd.to_numeric(d["log_return"], errors="coerce")
        d = d.dropna(subset=["proba_up", "log_return", "timestamp"])
        if d.empty:
            continue

        d["up_conf"] = d["proba_up"]
        d["down_conf"] = 1.0 - d["proba_up"]
        d["pred_up"] = d["proba_up"] >= 0.5
        d["pred_down"] = d["proba_up"] < 0.5
        d["trigger_up"] = d["pred_up"] & (d["up_conf"] >= up_thr)
        d["trigger_down"] = d["pred_down"] & (d["down_conf"] >= down_thr)

        actual_num = pd.to_numeric(d["actual"], errors="coerce")
        d["actual_up"] = (actual_num == 1)
        d["actual_down"] = (actual_num == 0)
        miss_actual = actual_num.isna()
        stats["actual_missing"] += int(miss_actual.sum())
        d.loc[miss_actual, "actual_up"] = d.loc[miss_actual, "log_return"] > 0
        d.loc[miss_actual, "actual_down"] = d.loc[miss_actual, "log_return"] < 0

        d["up_win"] = d["actual_up"].astype(int)
        d["down_win"] = d["actual_down"].astype(int)
        d["up_pnl_proxy"] = _compute_up_pnl_proxy(d)
        d["down_pnl_proxy"] = _compute_down_pnl_proxy(d)

        stats["raw"] += int(len(d))
        trig_cnt = int(d["trigger_up"].sum()) + int(d["trigger_down"].sum())
        stats["triggered"] += trig_cnt
        if source_kind == "proxy":
            stats["triggered_proxy"] += trig_cnt

        rows.extend(_append_direction_rows(d, profile, trader, symbol, model_family, source_family, source_kind, "UP", up_thr))
        rows.extend(_append_direction_rows(d, profile, trader, symbol, model_family, source_family, source_kind, "DOWN", down_thr))

    return rows, stats, warn


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build SHOCK risk replay panel from td365d prediction files")
    ap.add_argument("--output", type=Path, default=REPORTS_DIR / "shock_risk_v2_panel.parquet")
    ap.add_argument("--meta-output", type=Path, default=REPORTS_DIR / "shock_risk_v2_panel_meta.json")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    family_dfs = _load_family_df_cache()
    if not family_dfs:
        raise RuntimeError("No usable td365d panel parquet found")

    cfg_default = _load_json(CFG_DEFAULT)
    cfg_70 = _load_json(CFG_70)
    if not isinstance(cfg_default, list) or not isinstance(cfg_70, list):
        raise RuntimeError("trader_configs format invalid")

    by_name_default = {str(x.get("name")): x for x in cfg_default if isinstance(x, dict)}
    by_name_70 = {str(x.get("name")): x for x in cfg_70 if isinstance(x, dict)}
    active_default = _parse_active(ACTIVE_DEFAULT)
    active_70 = _parse_active(ACTIVE_70)

    all_rows: List[Dict[str, Any]] = []
    warns: List[str] = []
    summary: Dict[str, Any] = {
        "generatedAt": datetime.now(tz=timezone.utc).isoformat(),
        "assets": ["BTC", "ETH"],
        "profiles": {
            "default": {"active_traders": len(active_default), "rows": 0, "raw": 0, "triggered": 0, "missing_source": 0, "actual_missing": 0, "triggered_proxy": 0},
            "70": {"active_traders": len(active_70), "rows": 0, "raw": 0, "triggered": 0, "missing_source": 0, "actual_missing": 0, "triggered_proxy": 0},
        },
        "sourceFamiliesLoaded": sorted(family_dfs.keys()),
        "warnings": [],
    }

    for profile, active_names, by_name in [("default", active_default, by_name_default), ("70", active_70, by_name_70)]:
        for trader in active_names:
            cfg = by_name.get(trader)
            if not cfg:
                warns.append(f"{profile}:{trader} not found in trader_configs")
                continue
            rows, st, w = _build_rows_for_cfg(profile, trader, cfg, family_dfs)
            all_rows.extend(rows)
            summary["profiles"][profile]["raw"] += st["raw"]
            summary["profiles"][profile]["triggered"] += st["triggered"]
            summary["profiles"][profile]["missing_source"] += st["missing_source"]
            summary["profiles"][profile]["actual_missing"] += st["actual_missing"]
            summary["profiles"][profile]["triggered_proxy"] += st["triggered_proxy"]
            warns.extend(w)

    if not all_rows:
        raise RuntimeError("No shock replay rows built")

    panel_df = pd.DataFrame(all_rows).sort_values(
        ["ts", "profile", "traderName", "symbol", "direction"]
    ).reset_index(drop=True)
    summary["profiles"]["default"]["rows"] = int((panel_df["profile"] == "default").sum())
    summary["profiles"]["70"]["rows"] = int((panel_df["profile"] == "70").sum())
    summary["rows_total"] = int(len(panel_df))
    summary["cells_total"] = int(panel_df["cell_id"].nunique())
    summary["cells_by_profile_symbol_direction"] = (
        panel_df.groupby(["profile", "symbol", "direction"])["cell_id"]
        .nunique()
        .rename("cells")
        .reset_index()
        .to_dict("records")
    )
    summary["time_range"] = {"min": str(panel_df["timestamp"].min()), "max": str(panel_df["timestamp"].max())}
    min_ts = int(panel_df["ts"].min())
    max_ts = int(panel_df["ts"].max())
    summary["panel_span_days"] = round((max_ts - min_ts) / 86400.0, 3)
    for profile in ("default", "70"):
        raw_n = int(summary["profiles"][profile]["raw"] or 0)
        trig_n = int(summary["profiles"][profile]["triggered"] or 0)
        miss_n = int(summary["profiles"][profile]["actual_missing"] or 0)
        proxy_n = int(summary["profiles"][profile]["triggered_proxy"] or 0)
        summary["profiles"][profile]["missing_actual_ratio"] = (miss_n / raw_n) if raw_n > 0 else 0.0
        summary["profiles"][profile]["proxy_source_ratio"] = (proxy_n / trig_n) if trig_n > 0 else 0.0
    summary["warnings"] = warns[:500]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.meta_output.parent.mkdir(parents=True, exist_ok=True)
    panel_df.to_parquet(args.output, index=False)
    args.meta_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] panel => {args.output} rows={len(panel_df)} cells={summary['cells_total']}")
    print(f"[OK] meta  => {args.meta_output}")
    print(f"[INFO] warnings={len(warns)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
