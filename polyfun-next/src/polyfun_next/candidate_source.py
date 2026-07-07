from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .types import CandidateSignal


class CandidateSourceError(RuntimeError):
    pass


class JsonlCandidateSource:
    """Read ETH 15m candidates from a JSONL file.

    The new live system deliberately refuses to synthesize trades from old slot state. A candidate
    must be explicitly written by research/export tooling with the minimal official-market fields.
    """

    def __init__(self, path: str | Path, max_age_seconds: int = 120):
        self.path = Path(path)
        self.max_age_seconds = max_age_seconds

    def latest(self) -> Optional[CandidateSignal]:
        if not self.path.exists():
            return None
        last = None
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    last = json.loads(line)
        if last is None:
            return None
        signal = _signal_from_row(last)
        age = (datetime.now(timezone.utc) - signal.generated_at).total_seconds()
        if age > self.max_age_seconds:
            raise CandidateSourceError(f"candidate stale: {age:.1f}s > {self.max_age_seconds}s")
        if signal.symbol != "ETH" or signal.period != "15m":
            raise CandidateSourceError("v1 canary only accepts ETH 15m candidates")
        return signal


def _signal_from_row(row: dict) -> CandidateSignal:
    required = ["symbol", "period", "market_slug", "condition_id", "token_id", "side", "model_score"]
    missing = [k for k in required if row.get(k) in (None, "")]
    if missing:
        raise CandidateSourceError(f"candidate missing fields: {missing}")
    generated_at = row.get("generated_at")
    if generated_at:
        dt = datetime.fromisoformat(str(generated_at).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = datetime.now(timezone.utc)
    return CandidateSignal(
        symbol=str(row["symbol"]),
        period=str(row["period"]),
        market_slug=str(row["market_slug"]),
        condition_id=str(row["condition_id"]),
        token_id=str(row["token_id"]),
        side=str(row["side"]),
        model_score=float(row["model_score"]),
        generated_at=dt,
        source=_optional_str(row.get("source")),
        live_model_profile=_optional_str(row.get("live_model_profile") or row.get("model_profile") or row.get("source")),
        selected_candidate=_optional_str(row.get("selected_candidate") or row.get("selectedCandidate")),
        train_window=_optional_str(row.get("train_window")),
        feature_mode=_optional_str(row.get("feature_mode")),
        edge=_optional_float(row.get("edge")),
        strategy_profile=_optional_str(row.get("strategy_profile")),
        base_model_profile=_optional_str(row.get("base_model_profile")),
        base_selected_candidate=_optional_str(row.get("base_selected_candidate")),
        shock_filter_enabled=bool(row.get("shock_filter_enabled")),
        shock_condition=bool(row.get("shock_condition")),
        shock_gate_probability=_optional_float(row.get("shock_gate_probability")),
        shock_gate_threshold=_optional_float(row.get("shock_gate_threshold")),
        shock_action=_optional_str(row.get("shock_action")),
        shock_reason=_optional_str(row.get("shock_reason")),
        shock_profile=_optional_str(row.get("shock_profile")),
        shock_candidate_id=_optional_str(row.get("shock_candidate_id") or row.get("shock_candidate")),
        shock_model_engine=_optional_str(row.get("shock_model_engine")),
        shock_model_hyper=row.get("shock_model_hyper") if isinstance(row.get("shock_model_hyper"), dict) else None,
        calibration_router_enabled=bool(row.get("calibration_router_enabled")),
        calibration_router_profile=_optional_str(row.get("calibration_router_profile")),
        calibration_router_candidate_id=_optional_str(row.get("calibration_router_candidate_id")),
        calibration_router_shadow_only=bool(row.get("calibration_router_shadow_only")),
        router_probability=_optional_float(row.get("router_probability")),
        router_required_probability=_optional_float(row.get("router_required_probability")),
        router_combo_score=_optional_float(row.get("router_combo_score")),
        router_combo_threshold=_optional_float(row.get("router_combo_threshold")),
        router_policy_mode=_optional_str(row.get("router_policy_mode")),
        router_daily_cap_mode=_optional_str(row.get("router_daily_cap_mode")),
        router_action=_optional_str(row.get("router_action")),
        router_reason=_optional_str(row.get("router_reason")),
        router_model_key=_optional_str(row.get("router_model_key")),
        router_feature_mode=_optional_str(row.get("router_feature_mode")),
    )


def _optional_str(value: object) -> str | None:
    if value in (None, ""):
        return None
    return str(value)


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    return float(value)
