#!/usr/bin/env python3
"""
70+ 置信度过滤：只读现有 predictions_*.json，只写 *_70.json（confidence >= 阈值）。

与 ensemble_prediction_writer 的 PREDICTION_SOURCES 对齐，不修改任何现有文件。
与现有预测写入器并行运行，可定时或 cron 调用。

用法:
  python scripts/filter_predictions_70.py           # 单次执行
  python scripts/filter_predictions_70.py --loop 60   # 每 60 秒轮询
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

POLYMARKET_DIR = PROJECT_ROOT / "polymarket"
REPORTS_DIR = PROJECT_ROOT / "reports"
PROMOTION_STATE = PROJECT_ROOT / "reports" / "core10_promotion_state.json"
ACTIVE_TRADERS_70 = POLYMARKET_DIR / "active_traders_70.json"
TRADER_CONFIGS_70 = POLYMARKET_DIR / "trader_configs_70.json"
ACTIVE_LOWPRICE_70 = POLYMARKET_DIR / "active_traders_monitor_only_lowprice_70.json"
TRADER_CONFIGS_LOWPRICE_70 = POLYMARKET_DIR / "trader_configs_monitor_only_lowprice_70.json"
LIVE_SELECTION = POLYMARKET_DIR / "live_selected_cells.json"
LOWPRICE_PRELAUNCH_AUDIT = REPORTS_DIR / "exp_lowprice_prelaunch_audit_latest.json"
DEFAULT_MIN_CONFIDENCE = 0.70
DEFAULT_MIN_CONFIDENCE_BTCETH = None
DEFAULT_MIN_CONFIDENCE_XRP = None
DEFAULT_SOURCE_MAX_AGE_SEC = 1800

# 与 ensemble_prediction_writer.PREDICTION_SOURCES 对齐（Exp10~17 + GRU btc/eth；v5 主线扩到 XRP）
PREDICTION_SOURCES = []
for n in (10, 11, 13, 14, 15, 16, 17):
    PREDICTION_SOURCES.append({
        "name": f"exp{n}",
        "file": f"predictions_v5_exp{n}.json",
        "coins": ["BTC", "ETH", "XRP"],
    })
for variant in ("", "_no1h4h"):
    for coin in ("btc", "eth"):
        PREDICTION_SOURCES.append({
            "name": f"gru_{coin}{variant}",
            "file": f"predictions_gru_{coin}{variant}.json",
            "coins": [coin.upper()],
        })

COIN_KEYS = {
    "BTC": "BTC_USDT_15m",
    "ETH": "ETH_USDT_15m",
    "XRP": "XRP_USDT_15m",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [Filter70] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("filter70")


def _writer_state_path(prediction_path: Path) -> Path:
    return prediction_path.with_name(f"{prediction_path.stem}.writer_state.json")


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(path)


def _parse_iso_ms(value: object) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp() * 1000.0
    except Exception:
        return 0.0


def _prediction_meta(payload: dict | None) -> dict[str, object]:
    payload = payload if isinstance(payload, dict) else {}
    predictions = payload.get("predictions") if isinstance(payload.get("predictions"), dict) else {}
    target_period_end_ts = payload.get("target_period_end_ts")
    try:
        target_period_end_ts = int(target_period_end_ts) if target_period_end_ts is not None else 0
    except Exception:
        target_period_end_ts = 0
    return {
        "timestamp_ms": _parse_iso_ms(payload.get("timestamp")),
        "target_period_end_ts": target_period_end_ts,
        "prediction_count": len(predictions),
        "source_stale": bool(payload.get("source_stale")),
    }


def _core10_upstream_prediction_path(canonical_base_path: Path) -> Path | None:
    name = canonical_base_path.name
    if not name.startswith("predictions_core10_") or not name.endswith(".json"):
        return None
    return canonical_base_path.with_name(f"predictions_{name[len('predictions_core10_'):]}")


def _needs_canonical_base_sync(canonical_payload: dict | None, upstream_payload: dict | None) -> bool:
    upstream_meta = _prediction_meta(upstream_payload)
    canonical_meta = _prediction_meta(canonical_payload)
    if int(upstream_meta["target_period_end_ts"] or 0) > int(canonical_meta["target_period_end_ts"] or 0):
        return True
    if float(upstream_meta["timestamp_ms"] or 0.0) > float(canonical_meta["timestamp_ms"] or 0.0) + 1000.0:
        return True
    if bool(canonical_meta["source_stale"]):
        return True
    if int(canonical_meta["prediction_count"] or 0) == 0 and int(upstream_meta["prediction_count"] or 0) > 0:
        return True
    return False


def _sync_core10_canonical_base_from_upstream(canonical_base_path: Path) -> Path:
    upstream_path = _core10_upstream_prediction_path(canonical_base_path)
    if upstream_path is None or not upstream_path.exists():
        return canonical_base_path
    try:
        upstream_payload = json.loads(upstream_path.read_text(encoding="utf-8"))
    except Exception:
        return canonical_base_path
    canonical_payload = None
    if canonical_base_path.exists():
        try:
            canonical_payload = json.loads(canonical_base_path.read_text(encoding="utf-8"))
        except Exception:
            canonical_payload = None
    if _needs_canonical_base_sync(canonical_payload, upstream_payload):
        _atomic_write_json(canonical_base_path, upstream_payload)
        logger.info("同步 canonical base %s <- %s", canonical_base_path.name, upstream_path.name)
    upstream_writer_state_path = _writer_state_path(upstream_path)
    canonical_writer_state_path = _writer_state_path(canonical_base_path)
    if upstream_writer_state_path.exists():
        try:
            upstream_state = json.loads(upstream_writer_state_path.read_text(encoding="utf-8"))
            canonical_state = (
                json.loads(canonical_writer_state_path.read_text(encoding="utf-8"))
                if canonical_writer_state_path.exists()
                else None
            )
        except Exception:
            upstream_state = None
            canonical_state = None
        if isinstance(upstream_state, dict) and _needs_canonical_base_sync(canonical_state, upstream_state):
            _atomic_write_json(canonical_writer_state_path, upstream_state)
            logger.info("同步 canonical writer_state %s <- %s", canonical_writer_state_path.name, upstream_writer_state_path.name)
    return canonical_base_path


def load_core10_promoted_sources() -> list[dict]:
    if not PROMOTION_STATE.exists():
        return []
    try:
        payload = json.loads(PROMOTION_STATE.read_text(encoding="utf-8"))
    except Exception:
        return []

    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for row in payload.get("jobs", []):
        if not isinstance(row, dict) or not bool(row.get("promoted")):
            continue
        job_id = str(row.get("job_id") or "").strip()
        symbol = str(row.get("symbol") or "").upper().strip()
        if not job_id or symbol not in COIN_KEYS:
            continue
        key = (job_id, symbol)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "name": job_id,
                "file": f"predictions_{job_id}.json",
                "coins": [symbol],
            }
        )
    return out


def _load_active_core10_sources(active_path: Path, config_path: Path) -> list[dict]:
    if not active_path.exists() or not config_path.exists():
        return []
    try:
        active_payload = json.loads(active_path.read_text(encoding="utf-8"))
        cfg_rows = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(active_payload, dict) or not isinstance(cfg_rows, list):
        return []

    active_names = {
        str(x).strip()
        for x in ((active_payload.get("active_traders") or active_payload.get("traderNames") or []))
        if str(x).strip()
    }
    disabled_by_trader = active_payload.get("disabledSymbolsByTrader") if isinstance(active_payload.get("disabledSymbolsByTrader"), dict) else {}

    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for row in cfg_rows:
        if not isinstance(row, dict):
            continue
        trader = str(row.get("name") or "").strip()
        if not trader or trader not in active_names:
            continue
        suffix_map = row.get("predictionSuffixBySymbol")
        if not isinstance(suffix_map, dict):
            continue
        disabled_symbols = {str(x).strip().upper() for x in (disabled_by_trader.get(trader) or []) if str(x).strip()}
        for symbol, suffix in suffix_map.items():
            normalized_symbol = str(symbol).strip().upper()
            raw_suffix = str(suffix or "").strip()
            if normalized_symbol not in COIN_KEYS:
                continue
            if normalized_symbol in disabled_symbols:
                continue
            if not raw_suffix.startswith("_core10_") or not raw_suffix.endswith("_70"):
                continue
            base_file = f"predictions{raw_suffix[:-3]}.json"
            key = (base_file, normalized_symbol)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "name": raw_suffix[1:-3],
                    "file": base_file,
                    "coins": [normalized_symbol],
                }
            )
    return out


def load_active_profile70_core10_sources() -> list[dict]:
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for row in [
        *_load_active_core10_sources(ACTIVE_TRADERS_70, TRADER_CONFIGS_70),
        *_load_active_core10_sources(ACTIVE_LOWPRICE_70, TRADER_CONFIGS_LOWPRICE_70),
        *_load_live_selected_lowprice_core10_sources(),
    ]:
        key = (str(row.get("file") or "").strip(), str((row.get("coins") or [""])[0] or "").strip())
        if not all(key) or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _load_live_selected_rows() -> list[dict]:
    try:
        payload = json.loads(LIVE_SELECTION.read_text(encoding="utf-8")) if LIVE_SELECTION.exists() else {}
    except Exception:
        return []
    rows = payload.get("selected_cells") if isinstance(payload, dict) else []
    return [row for row in rows if isinstance(row, dict) and bool(row.get("enabled", True))]


def _lowprice_materialized_source_map() -> dict[tuple[str, str, str], str]:
    try:
        payload = json.loads(LOWPRICE_PRELAUNCH_AUDIT.read_text(encoding="utf-8")) if LOWPRICE_PRELAUNCH_AUDIT.exists() else {}
    except Exception:
        return {}
    profiles = payload.get("profiles") if isinstance(payload.get("profiles"), dict) else {}
    out: dict[tuple[str, str, str], str] = {}
    for raw_profile, profile_payload in profiles.items():
        if not isinstance(profile_payload, dict):
            continue
        materialized_summary = profile_payload.get("materialized_summary") if isinstance(profile_payload.get("materialized_summary"), dict) else {}
        rows = materialized_summary.get("rows") if isinstance(materialized_summary.get("rows"), list) else []
        profile = "70" if str(raw_profile or "").strip() == "70" else "default"
        for row in rows:
            if not isinstance(row, dict):
                continue
            trader = str(row.get("name") or "").strip()
            symbol = str(row.get("symbol") or "").strip().upper()
            source_trader = str(row.get("sourceTrader") or "").strip()
            if trader and symbol and source_trader:
                out[(profile, trader, symbol)] = source_trader
    return out


def _load_live_selected_lowprice_core10_sources() -> list[dict]:
    source_map = _lowprice_materialized_source_map()
    try:
        cfg_rows = json.loads(TRADER_CONFIGS_70.read_text(encoding="utf-8")) if TRADER_CONFIGS_70.exists() else []
    except Exception:
        cfg_rows = []
    cfg_by_name = {
        str(row.get("name") or "").strip(): row
        for row in cfg_rows
        if isinstance(row, dict) and str(row.get("name") or "").strip()
    }
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    for row in _load_live_selected_rows():
        profile = "70" if str(row.get("profile") or "").strip() == "70" else "default"
        if profile != "70":
            continue
        if str(row.get("scope") or "").strip() != "lowprice":
            continue
        trader = str(row.get("trader") or "").strip()
        symbol = str(row.get("symbol") or "").strip().upper()
        if not trader or symbol not in COIN_KEYS:
            continue
        source_trader = source_map.get((profile, trader, symbol), trader)
        cfg_row = cfg_by_name.get(source_trader) or {}
        suffix_map = cfg_row.get("predictionSuffixBySymbol") if isinstance(cfg_row.get("predictionSuffixBySymbol"), dict) else {}
        raw_suffix = str(suffix_map.get(symbol) or cfg_row.get("predictionSuffix") or "").strip()
        if not raw_suffix.startswith("_core10_") or not raw_suffix.endswith("_70"):
            continue
        base_file = f"predictions{raw_suffix[:-3]}.json"
        key = (base_file, symbol)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "name": raw_suffix[1:-3],
                "file": base_file,
                "coins": [symbol],
            }
        )
    return out


def iter_prediction_sources() -> list[dict]:
    return [*PREDICTION_SOURCES, *load_core10_promoted_sources(), *load_active_profile70_core10_sources()]


def get_confidence(entry: dict) -> float | None:
    """从单币种条目得到置信度；无则用 max(proba_up, 1-proba_up)。"""
    if not entry:
        return None
    c = entry.get("confidence")
    if c is not None:
        try:
            return float(c)
        except (TypeError, ValueError):
            pass
    details = entry.get("details") or {}
    p = details.get("proba_up")
    if p is not None:
        try:
            p = float(p)
            return max(p, 1.0 - p)
        except (TypeError, ValueError):
            pass
    return None


def _resolve_symbol_thresholds(
    min_confidence: float,
    min_confidence_btceth: float | None,
    min_confidence_xrp: float | None,
) -> dict[str, float]:
    base = max(0.0, min(0.999, float(min_confidence)))
    env_btceth = os.environ.get("MIN_CONFIDENCE_70_BTCETH")
    env_xrp = os.environ.get("MIN_CONFIDENCE_70_XRP")
    btceth = min_confidence_btceth
    xrp = min_confidence_xrp
    if btceth is None and env_btceth:
        try:
            btceth = float(env_btceth)
        except (TypeError, ValueError):
            btceth = None
    if xrp is None and env_xrp:
        try:
            xrp = float(env_xrp)
        except (TypeError, ValueError):
            xrp = None
    resolved = {
        "BTC": max(0.0, min(0.999, float(btceth if btceth is not None else base))),
        "ETH": max(0.0, min(0.999, float(btceth if btceth is not None else base))),
        "XRP": max(0.0, min(0.999, float(xrp if xrp is not None else base))),
    }
    return resolved


def _resolve_source_max_age_sec() -> int:
    raw = os.environ.get("FILTER70_SOURCE_MAX_AGE_SEC")
    try:
        if raw is not None:
            value = int(float(raw))
        else:
            value = DEFAULT_SOURCE_MAX_AGE_SEC
    except (TypeError, ValueError):
        value = DEFAULT_SOURCE_MAX_AGE_SEC
    return max(60, value)


def run_once(min_confidence: float, min_confidence_btceth: float | None = None, min_confidence_xrp: float | None = None) -> None:
    symbol_thresholds = _resolve_symbol_thresholds(
        min_confidence=min_confidence,
        min_confidence_btceth=min_confidence_btceth,
        min_confidence_xrp=min_confidence_xrp,
    )
    source_max_age_sec = _resolve_source_max_age_sec()
    for src in iter_prediction_sources():
        in_path = POLYMARKET_DIR / src["file"]
        out_name = src["file"].replace(".json", "_70.json")
        out_path = POLYMARKET_DIR / out_name

        if not in_path.exists():
            logger.debug("跳过（不存在）: %s", in_path.name)
            continue

        in_path = _sync_core10_canonical_base_from_upstream(in_path)

        try:
            raw = json.loads(in_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("读取失败 %s: %s", in_path.name, e)
            continue

        source_age_sec = max(0.0, time.time() - in_path.stat().st_mtime)
        source_stale = source_age_sec > source_max_age_sec
        predictions_in = raw.get("predictions") or {}
        predictions_out = {}
        filtered_out_symbols: list[dict[str, object]] = []

        if not source_stale:
            for coin in src["coins"]:
                key = COIN_KEYS.get(coin)
                if not key:
                    continue
                entry = predictions_in.get(key)
                min_confidence_for_coin = float(symbol_thresholds.get(coin, min_confidence))
                conf = get_confidence(entry) if entry else None
                if conf is not None and conf >= min_confidence_for_coin:
                    predictions_out[key] = entry
                elif entry:
                    filtered_out_symbols.append(
                        {
                            "symbol": coin,
                            "confidence": conf,
                            "threshold": min_confidence_for_coin,
                        }
                    )
                # confidence < 当前币种阈值 的币对不入 70+ 输出（该源在该 bar 不贡献该币）

        source_prediction_count = len(predictions_in) if isinstance(predictions_in, dict) else 0
        filtered_empty_due_to_threshold = bool(
            not source_stale
            and source_prediction_count > 0
            and len(predictions_out) == 0
            and len(filtered_out_symbols) > 0
        )
        effective_threshold = max(float(symbol_thresholds.get(coin, min_confidence)) for coin in src["coins"]) if src["coins"] else float(min_confidence)

        out_payload = {
            "timestamp": raw.get("timestamp", ""),
            "target_period_end_ts": raw.get("target_period_end_ts"),
            "model_version": raw.get("model_version", ""),
            "loaded_model_revision": raw.get("loaded_model_revision", ""),
            "loaded_model_mtime": raw.get("loaded_model_mtime", ""),
            "phase": raw.get("phase", 0),
            "limit_price": raw.get("limit_price", 0.5),
            "bet_fraction_this_phase": raw.get("bet_fraction_this_phase", 1.0),
            "max_sweep_price": raw.get("max_sweep_price", 0.54),
            "min_confidence_by_symbol": symbol_thresholds,
            "filter_threshold": effective_threshold,
            "filter_threshold_by_symbol": {coin: float(symbol_thresholds.get(coin, min_confidence)) for coin in src["coins"]},
            "source_file": in_path.name,
            "source_age_sec": round(source_age_sec, 3),
            "source_stale": source_stale,
            "source_prediction_count": source_prediction_count,
            "predictionCount": len(predictions_out),
            "filtered_empty_due_to_threshold": filtered_empty_due_to_threshold,
            "filtered_out_symbols": filtered_out_symbols,
            "predictions": predictions_out,
        }

        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(out_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.rename(out_path)
        n = len(predictions_out)
        if source_stale:
            logger.warning(
                "写入 %s: 源文件过旧 %s (age=%.0fs > %ss) -> 输出清空 predictions",
                out_path.name,
                in_path.name,
                source_age_sec,
                source_max_age_sec,
            )
        else:
            logger.info(
                "写入 %s: %d 币对 (BTC/ETH>=%.1f%% XRP>=%.1f%%)",
                out_path.name,
                n,
                symbol_thresholds["BTC"] * 100.0,
                symbol_thresholds["XRP"] * 100.0,
            )
            if filtered_empty_due_to_threshold:
                logger.info(
                    "  ↳ %s 当前 bar 为新鲜空文件：源预测存在，但全部被 %.1f%% 过滤线挡掉",
                    out_path.name,
                    effective_threshold * 100.0,
                )


def main():
    ap = argparse.ArgumentParser(description="70+ 置信度过滤：只写 *_70.json")
    ap.add_argument("--loop", type=float, default=0, help="轮询间隔(秒)，0=只跑一次")
    ap.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help="最小置信度阈值（默认 0.70，可设 0.60~0.70）",
    )
    ap.add_argument(
        "--min-confidence-btceth",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE_BTCETH,
        help="BTC/ETH 最小置信度阈值（默认跟随 --min-confidence 或环境变量 MIN_CONFIDENCE_70_BTCETH）",
    )
    ap.add_argument(
        "--min-confidence-xrp",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE_XRP,
        help="XRP 最小置信度阈值（默认跟随 --min-confidence 或环境变量 MIN_CONFIDENCE_70_XRP）",
    )
    args = ap.parse_args()
    min_confidence = max(0.0, min(0.999, float(args.min_confidence)))
    min_confidence_btceth = None if args.min_confidence_btceth is None else max(0.0, min(0.999, float(args.min_confidence_btceth)))
    min_confidence_xrp = None if args.min_confidence_xrp is None else max(0.0, min(0.999, float(args.min_confidence_xrp)))
    symbol_thresholds = _resolve_symbol_thresholds(
        min_confidence=min_confidence,
        min_confidence_btceth=min_confidence_btceth,
        min_confidence_xrp=min_confidence_xrp,
    )

    logger.info(
        "70+ 过滤启动 (BTC/ETH >= %.1f%%, XRP >= %.1f%%)",
        symbol_thresholds["BTC"] * 100.0,
        symbol_thresholds["XRP"] * 100.0,
    )
    if args.loop > 0:
        while True:
            run_once(
                min_confidence=min_confidence,
                min_confidence_btceth=min_confidence_btceth,
                min_confidence_xrp=min_confidence_xrp,
            )
            time.sleep(args.loop)
    else:
        run_once(
            min_confidence=min_confidence,
            min_confidence_btceth=min_confidence_btceth,
            min_confidence_xrp=min_confidence_xrp,
        )


if __name__ == "__main__":
    main()
