#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLY = PROJECT_ROOT / "polymarket"
REPORTS = PROJECT_ROOT / "reports"
LOGS = PROJECT_ROOT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

import sys

sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ops.core10_retrain_common import (  # noqa: E402
    build_cell_dataset,
    load_core_cell_map,
    normalize_policy_config,
    policy_actions_from_scores,
)
from scripts.ops.generate_core10_stage16_event_precision_search_latest import (  # noqa: E402
    _predict_candidate_scores,
)

FULL_STACK_REPLAY = REPORTS / "core10_exp13_full_stack_replay_latest.json"
TARGET_CELL = "default/v5_exp13_bp0530/ETH"
DEFAULT_OUTPUT = POLY / "predictions_v5_exp13_official_candidate.json"
DEFAULT_SLEEP_SEC = 30
DEFAULT_FILE_MAX_AGE_SEC = 5 * 60


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _compute_target_period(now_ts: int) -> int:
    return (now_ts // 900) * 900


def _confidence_value(up_score: float, down_score: float) -> float:
    hi = max(up_score, down_score)
    lo = min(up_score, down_score)
    if lo >= -0.5 and hi <= 0.5:
        return float(np.clip(0.5 + hi, 0.0, 1.0))
    return float(np.clip(hi, 0.0, 1.0))


def _load_candidate_spec() -> Dict[str, Any]:
    payload = _load_json(FULL_STACK_REPLAY)
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    compare_context = payload.get("compare_context") if isinstance(payload.get("compare_context"), dict) else {}
    candidate_ready = bool(summary.get("candidate_ready"))
    verdict = str(summary.get("verdict") or "")
    candidate_dir = Path(str(compare_context.get("candidate_dir") or "")).expanduser()
    route_states = [str(x) for x in (compare_context.get("route_states") or []) if str(x).strip()]
    policy_config = normalize_policy_config(compare_context.get("policy_config") if isinstance(compare_context.get("policy_config"), dict) else None)
    output_tag = str(compare_context.get("candidate_output_tag") or "")
    cfg = _load_json(candidate_dir / "config.json") if candidate_dir.exists() else {}
    label_config = cfg.get("label_config") if isinstance(cfg.get("label_config"), dict) else {}
    window_days = [int(x) for x in (cfg.get("window_days_list") or [365]) if str(x).strip()]
    return {
        "candidate_ready": candidate_ready and verdict == "candidate_ready" and candidate_dir.exists(),
        "candidate_dir": candidate_dir,
        "route_states": route_states or ["rally_up"],
        "policy_config": policy_config,
        "candidate_output_tag": output_tag,
        "label_config": label_config,
        "window_days": window_days or [365],
        "full_stack_summary": summary,
    }


def _build_prediction_payload(output_path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    cell = load_core_cell_map()[TARGET_CELL]
    spec = _load_candidate_spec()
    df = build_cell_dataset(
        cell,
        max_window_days=max(spec["window_days"]),
        force_rebuild=True,
        label_config=spec["label_config"],
    )
    infer_df = df.tail(2048).copy().reset_index(drop=True)
    scores = _predict_candidate_scores(spec["candidate_dir"], infer_df, spec["route_states"]) if spec["candidate_ready"] else {
        "up_score": np.zeros(len(infer_df), dtype=float),
        "down_score": np.zeros(len(infer_df), dtype=float),
        "abstain_score": np.ones(len(infer_df), dtype=float),
    }
    actions = policy_actions_from_scores(
        scores["up_score"],
        scores["down_score"],
        scores["abstain_score"],
        policy_config=spec["policy_config"],
    )

    idx = len(infer_df) - 1
    last = infer_df.iloc[idx]
    up_score = float(scores["up_score"][idx])
    down_score = float(scores["down_score"][idx])
    abstain_score = float(scores["abstain_score"][idx])
    action = str(actions[idx]) if idx >= 0 else "ABSTAIN"
    state_label = str(last.get("state_label") or "unknown")

    should_trade = bool(spec["candidate_ready"] and action != "ABSTAIN")
    skip_reason = None
    if not spec["candidate_ready"]:
        skip_reason = "full_stack_candidate_not_ready"
    elif state_label not in set(spec["route_states"]):
        should_trade = False
        skip_reason = f"route_state_not_eligible:{state_label}"
    elif action == "ABSTAIN":
        skip_reason = "policy_abstain"

    confidence = _confidence_value(up_score, down_score)
    proba_up = confidence
    now = datetime.now(timezone.utc)
    now_ts = int(now.timestamp())
    target_period_end_ts = _compute_target_period(now_ts)
    prediction = {
        "symbol": "ETH/USDT",
        "timeframe": "15m",
        "direction": "UP",
        "confidence": round(confidence, 6),
        "timestamp": now.isoformat(),
        "details": {
            "proba_up": round(proba_up, 6),
            "candidate_output_tag": spec["candidate_output_tag"],
            "candidate_dir": str(spec["candidate_dir"]),
            "route_state": state_label,
            "route_states": spec["route_states"],
            "up_score": round(up_score, 6),
            "down_score": round(down_score, 6),
            "abstain_score": round(abstain_score, 6),
            "trade_decision": {
                "should_trade": should_trade,
                "skip_reason": skip_reason,
            },
            "policy_config": spec["policy_config"],
            "full_stack_summary": spec["full_stack_summary"],
            "model_version": f"exp13_official_candidate::{spec['candidate_output_tag'] or 'unknown'}",
        },
    }
    payload = {
        "timestamp": now.isoformat(),
        "target_period_end_ts": target_period_end_ts,
        "model_version": f"exp13_official_candidate::{spec['candidate_output_tag'] or 'unknown'}",
        "phase": 1,
        "limit_price": 0.53,
        "bet_fraction_this_phase": 1.0,
        "max_sweep_price": 0.54,
        "predictions": {
            "ETH_USDT_15m": prediction,
        },
    }
    meta = {
        "target_period_end_ts": target_period_end_ts,
        "candidate_ready": spec["candidate_ready"],
        "candidate_output_tag": spec["candidate_output_tag"],
        "route_state": state_label,
        "should_trade": should_trade,
        "skip_reason": skip_reason,
        "output_file": str(output_path),
    }
    return payload, meta


def _write_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _should_refresh(output_path: Path, target_period_end_ts: int, last_slot: int | None, last_sig: str | None, next_sig: str, max_age_sec: int) -> bool:
    if last_slot is None or target_period_end_ts != last_slot:
        return True
    if last_sig != next_sig:
        return True
    if not output_path.exists():
        return True
    try:
        age = time.time() - output_path.stat().st_mtime
    except OSError:
        return True
    return age >= max_age_sec


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--sleep-sec", type=int, default=DEFAULT_SLEEP_SEC)
    parser.add_argument("--max-age-sec", type=int, default=DEFAULT_FILE_MAX_AGE_SEC)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    output_path = Path(args.output)
    last_slot: int | None = None
    last_sig: str | None = None

    while True:
        spec = _load_candidate_spec()
        next_sig = "|".join(
            [
                str(spec["candidate_dir"]),
                str(spec["candidate_output_tag"]),
                json.dumps(spec["policy_config"], sort_keys=True, ensure_ascii=False),
                ",".join(spec["route_states"]),
                str(spec["candidate_ready"]),
            ]
        )
        target_period_end_ts = _compute_target_period(int(time.time()))
        if _should_refresh(output_path, target_period_end_ts, last_slot, last_sig, next_sig, max(30, int(args.max_age_sec))):
            try:
                payload, meta = _build_prediction_payload(output_path)
                _write_atomic(output_path, payload)
                last_slot = int(meta["target_period_end_ts"])
                last_sig = next_sig
                logging.info(
                    "official candidate write ok candidate=%s state=%s should_trade=%s skip=%s output=%s",
                    meta["candidate_output_tag"],
                    meta["route_state"],
                    meta["should_trade"],
                    meta["skip_reason"],
                    output_path,
                )
            except Exception as exc:
                logging.exception("official candidate write failed: %s", exc)
        if args.once:
            return 0
        time.sleep(max(10, int(args.sleep_sec)))


if __name__ == "__main__":
    raise SystemExit(main())
