from __future__ import annotations

import json
from datetime import datetime, timezone

from polyfun_next.calibration_router import router_candidate_fields
from polyfun_next.candidate_source import JsonlCandidateSource


def test_router_candidate_fields_preserve_shadow_decision():
    router = {
        "enabled": True,
        "profile": "top159_live_order_calibration_router_v1",
        "candidate_id": "d3e3e0b3a618dfa3",
        "shadow_only": True,
        "router_probability": 0.52,
        "router_required_probability": 0.535,
        "router_combo_score": 0.541,
        "router_combo_threshold": 0.545,
        "router_policy_mode": "combo",
        "router_daily_cap_mode": "chronological",
        "router_action": "shadow_block",
        "router_reason": "router_blocked",
        "model_key": "9262f79d09e87654",
        "feature_mode": "atoms_core",
    }
    fields = router_candidate_fields(router)
    assert fields["calibration_router_enabled"] is True
    assert fields["calibration_router_shadow_only"] is True
    assert fields["calibration_router_candidate_id"] == "d3e3e0b3a618dfa3"
    assert fields["router_action"] == "shadow_block"
    assert fields["router_daily_cap_mode"] == "chronological"


def test_candidate_source_loads_router_evidence(tmp_path):
    path = tmp_path / "candidate.jsonl"
    row = {
        "symbol": "ETH",
        "period": "15m",
        "market_slug": "eth-updown-15m-1",
        "condition_id": "0xabc",
        "token_id": "123",
        "side": "UP",
        "model_score": 0.56,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "calibration_router_enabled": True,
        "calibration_router_profile": "top159_live_order_calibration_router_v1",
        "calibration_router_candidate_id": "d3e3e0b3a618dfa3",
        "calibration_router_shadow_only": True,
        "router_probability": 0.52,
        "router_required_probability": 0.535,
        "router_combo_score": 0.541,
        "router_combo_threshold": 0.545,
        "router_policy_mode": "combo",
        "router_daily_cap_mode": "chronological",
        "router_action": "shadow_block",
        "router_reason": "router_blocked",
        "router_model_key": "9262f79d09e87654",
        "router_feature_mode": "atoms_core",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    signal = JsonlCandidateSource(path, max_age_seconds=300).latest()
    assert signal is not None
    assert signal.calibration_router_enabled is True
    assert signal.calibration_router_shadow_only is True
    assert signal.calibration_router_candidate_id == "d3e3e0b3a618dfa3"
    assert signal.router_action == "shadow_block"
    assert signal.router_combo_score == 0.541
