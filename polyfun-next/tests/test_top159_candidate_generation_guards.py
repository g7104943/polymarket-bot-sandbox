from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_top159_live_candidate.py"
SHOCK_SCRIPT = ROOT / "scripts" / "run_top159_shock_candle_filter_research.py"


def _load_candidate_module():
    spec = importlib.util.spec_from_file_location("generate_top159_live_candidate_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_shock_module():
    spec = importlib.util.spec_from_file_location("run_top159_shock_candle_filter_research_test", SHOCK_SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_market_match_rejects_adjacent_15m_market(monkeypatch):
    module = _load_candidate_module()
    target_start = "2026-05-04T05:00:00+00:00"

    def fake_gamma_get(_url: str):
        return [
            {
                "slug": "eth-updown-15m-1777871700",
                "question": "Ethereum Up or Down - May 4, 1:15AM-1:30AM ET",
                "endDate": "2026-05-04T05:30:00Z",
                "outcomes": '["Up","Down"]',
                "clobTokenIds": '["up-token","down-token"]',
                "conditionId": "adjacent-condition",
            }
        ]

    monkeypatch.setattr(module, "gamma_get", fake_gamma_get)

    assert module.find_eth15m_market(target_start, "UP") is None


def test_market_match_accepts_precise_market(monkeypatch):
    module = _load_candidate_module()
    target_start = "2026-05-04T05:00:00+00:00"

    def fake_gamma_get(_url: str):
        return [
            {
                "slug": "eth-updown-15m-1777870800",
                "question": "Ethereum Up or Down - May 4, 1:00AM-1:15AM ET",
                "endDate": "2026-05-04T05:15:00Z",
                "outcomes": '["Up","Down"]',
                "clobTokenIds": '["up-token","down-token"]',
                "conditionId": "exact-condition",
            }
        ]

    monkeypatch.setattr(module, "gamma_get", fake_gamma_get)

    market = module.find_eth15m_market(target_start, "UP")
    assert market is not None
    assert market["condition_id"] == "exact-condition"
    assert market["token_id"] == "up-token"
    assert market["best_score_seconds"] == 0


def test_shock_candle_features_match_by_closed_available_time():
    module = _load_shock_module()
    raw4h = pd.DataFrame(
        [
            {"dt": pd.Timestamp("2026-05-04T08:00:00Z"), "open": 100, "high": 120, "low": 90, "close": 110, "volume": 10},
            {"dt": pd.Timestamp("2026-05-04T12:00:00Z"), "open": 110, "high": 130, "low": 100, "close": 125, "volume": 10},
        ]
    )
    feats = module.candle_features(raw4h, "4h").sort_values("ts_ns")
    candidate = pd.DataFrame({"ts_ns": [pd.Timestamp("2026-05-04T14:00:00Z").value]})

    matched = pd.merge_asof(candidate, feats.drop(columns=["dt"]), on="ts_ns", direction="backward", allow_exact_matches=True)

    assert str(pd.Timestamp(matched.loc[0, "4h_open_time"])) == "2026-05-04 08:00:00+00:00"
    assert str(pd.Timestamp(matched.loc[0, "4h_available_at"])) == "2026-05-04 12:00:00+00:00"


def test_cluster_gate_raises_required_score_when_bad_cluster_hits():
    module = _load_candidate_module()
    live_row = pd.DataFrame([{
        "direction": "UP",
        "score15": 0.57,
        "15m_same_as_top159": True,
        "1h_volume_mult": 1.7,
        "4h_volume_mult": 1.2,
        "1h_pos20": 0.80,
        "1h_trend_state": "down",
    }])
    profile = {
        "enabled": True,
        "profile": "new_shock_filter_top159_cluster_06173_v1",
        "strategy_profile": "new_shock_filter_top159_cluster_06173",
        "candidate_id": "06173b68b0d86431",
        "base_model_profile": "integrated_main_lightgbm_full_base15_edge0.05_archive_rerank_v1",
        "base_selected_candidate": "integrated_lightgbm_full_base15_edge0.05_6c54c8a7b1d2593a",
    }
    params = {
        "family": "cluster_gate",
        "action": "raise_score",
        "min_cluster_hits": 2,
        "shock_score_min": 0.58,
        "any_clusters": [
            ["15m_same_as_top159", "1h_vol_ge_1.6", "4h_vol_ge_1.1"],
            ["15m_same_as_top159", "4h_vol_ge_1.1", "1h_pos_high"],
        ],
    }

    out = module.evaluate_cluster_shock_filter(
        profile,
        params,
        live_row,
        {"model_score": 0.57},
        {"edge": 0.05, "live_model_profile": "integrated_main_lightgbm_full_base15_edge0.05_archive_rerank_v1"},
    )

    assert out["cluster_hit_count"] == 2
    assert out["shock_condition"] is True
    assert out["shock_required_score"] == 0.58
    assert out["shock_action"] == "block"


def test_cluster_gate_keeps_base_threshold_when_no_bad_cluster_hits():
    module = _load_candidate_module()
    live_row = pd.DataFrame([{
        "direction": "UP",
        "score15": 0.551,
        "15m_same_as_top159": False,
        "1h_volume_mult": 1.0,
        "4h_volume_mult": 1.0,
        "1h_pos20": 0.50,
        "1h_trend_state": "mixed",
    }])
    profile = {"enabled": True, "profile": "new_shock_filter_top159_cluster_06173_v1", "strategy_profile": "new_shock_filter_top159_cluster_06173", "candidate_id": "06173b68b0d86431"}
    params = {
        "family": "cluster_gate",
        "action": "raise_score",
        "min_cluster_hits": 2,
        "shock_score_min": 0.58,
        "any_clusters": [["15m_same_as_top159", "1h_vol_ge_1.6", "4h_vol_ge_1.1"]],
    }

    out = module.evaluate_cluster_shock_filter(profile, params, live_row, {"model_score": 0.551}, {"edge": 0.05})

    assert out["cluster_hit_count"] == 0
    assert out["shock_condition"] is False
    assert out["shock_required_score"] == 0.55
    assert out["shock_action"] == "pass"


def test_cluster_gate_respects_zero_edge_override():
    module = _load_candidate_module()
    live_row = pd.DataFrame([{
        "direction": "UP",
        "score15": 0.516,
        "15m_same_as_top159": False,
        "1h_volume_mult": 1.0,
        "4h_volume_mult": 1.0,
        "1h_pos20": 0.50,
        "1h_trend_state": "mixed",
    }])
    profile = {"enabled": True, "profile": "new_shock_filter_top159_cluster_06173_v1", "strategy_profile": "new_shock_filter_top159_cluster_06173", "candidate_id": "06173b68b0d86431"}
    params = {
        "family": "cluster_gate",
        "action": "raise_score",
        "min_cluster_hits": 2,
        "shock_score_min": 0.58,
        "any_clusters": [["15m_same_as_top159", "1h_vol_ge_1.6", "4h_vol_ge_1.1"]],
    }

    out = module.evaluate_cluster_shock_filter(profile, params, live_row, {"model_score": 0.516}, {"edge": 0.0})

    assert out["shock_base_threshold"] == 0.5
    assert out["shock_required_score"] == 0.5
    assert out["shock_action"] == "pass"
