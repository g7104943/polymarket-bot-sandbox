from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_MODEL_PROFILE = "integrated_main_lightgbm_full_base15_edge0.05_archive_rerank_v1"
EXPECTED_SELECTED_CANDIDATE = "integrated_lightgbm_full_base15_edge0.05_6c54c8a7b1d2593a"
EXPECTED_STRATEGY_PROFILE = "new_shock_filter_top159_cluster_06173"
EXPECTED_SHOCK_PROFILE = "new_shock_filter_top159_cluster_06173_v1"
EXPECTED_SHOCK_CANDIDATE_ID = "06173b68b0d86431"
EXPECTED_MODEL_THRESHOLD = 0.55
EXPECTED_SHOCK_THRESHOLD = 0.58
EXPECTED_SHOCK_FAMILY = "cluster_gate"
EXPECTED_CLUSTER_ACTION = "raise_score"
EXPECTED_MIN_CLUSTER_HITS = 2
EXPECTED_FEATURE_ALIGNMENT = "closed_candle_available_at_v2"
EXPECTED_ROUTER_PROFILE = "top159_live_order_calibration_router_v1"
EXPECTED_ROUTER_CANDIDATE_ID = "d3e3e0b3a618dfa3"
EXPECTED_ROUTER_BASE_STRATEGY = EXPECTED_STRATEGY_PROFILE
EXPECTED_ROUTER_FEATURE_MODE = "atoms_core"
EXPECTED_ROUTER_MODEL_KEY = "9262f79d09e87654"
EXPECTED_ROUTER_MODEL = "LightGBM"
EXPECTED_ROUTER_MODE = "combo"
EXPECTED_ROUTER_PROB_MIN = 0.535
EXPECTED_ROUTER_SCORE_MIN = 0.55
EXPECTED_ROUTER_COMBO_MIN = 0.545
EXPECTED_ROUTER_COMBO_ALPHA = 1.5
EXPECTED_ROUTER_DAILY_CAP_MODE = "chronological"
EXPECTED_ROUTER_SHADOW_ONLY = True
EXPECTED_WS_CACHE_PATH = "/Users/mac/polyfun/polyfun-next/runtime/top159_ws_market_cache.json"
EXPECTED_ANY_CLUSTERS = [
    ["15m_same_as_top159", "1h_vol_ge_1.6", "4h_vol_ge_1.1"],
    ["15m_same_as_top159", "4h_vol_ge_1.1", "1h_pos_high"],
    ["15m_same_as_top159", "1h_pos_low"],
    ["15m_same_as_top159", "1h_same_as_top159", "1h_trend_down"],
]


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def audit_top159_contract(root: Path) -> dict[str, Any]:
    """Validate the live top159 contract that must be used for real trading.

    This is intentionally narrow: it checks the current approved live contract,
    not every historical research profile. Formal trading should block on any
    mismatch instead of silently falling back to an old model/config.
    """
    cfg = load_json(root / "config" / "canary.eth15m.json", {}) or {}
    live = load_json(root / "runtime" / "top159_live_model_profile.json", {}) or {}
    shock = load_json(root / "runtime" / "top159_shock_filter_profile.json", {}) or {}
    router = load_json(root / "runtime" / "top159_calibration_router_profile.json", {}) or {}
    live_params = live.get("params") if isinstance(live.get("params"), dict) else {}
    shock_params = shock.get("params") if isinstance(shock.get("params"), dict) else {}
    router_policy = router.get("policy") if isinstance(router.get("policy"), dict) else {}
    router_model_hyper = router.get("model_hyper") if isinstance(router.get("model_hyper"), dict) else {}
    checks: list[dict[str, Any]] = []

    def check(name: str, actual: Any, expected: Any, ok: bool | None = None) -> None:
        checks.append({
            "name": name,
            "actual": actual,
            "expected": expected,
            "ok": bool(actual == expected if ok is None else ok),
        })

    check("config.symbol", cfg.get("symbol"), "ETH")
    check("config.period", cfg.get("period"), "15m")
    check("config.preferred_order_type", str(cfg.get("preferred_order_type") or "").upper(), "FAK")
    check("config.stake_fraction", float(cfg.get("stake_fraction") or 0.0), 0.01, abs(float(cfg.get("stake_fraction") or 0.0) - 0.01) < 1e-12)
    check("config.enforce_max_entry_price", bool(cfg.get("enforce_max_entry_price")), False)
    check("config.enforce_value_edge_vs_price", bool(cfg.get("enforce_value_edge_vs_price")), False)
    check("config.websocket_quote_enabled", bool(cfg.get("websocket_quote_enabled")), True)
    check("config.websocket_quote_max_age_seconds", float(cfg.get("websocket_quote_max_age_seconds") or 0.0), 3.0, abs(float(cfg.get("websocket_quote_max_age_seconds") or 0.0) - 3.0) < 1e-12)
    check("config.websocket_market_cache_path", cfg.get("websocket_market_cache_path"), EXPECTED_WS_CACHE_PATH)
    check("config.min_value_edge", float(cfg.get("min_value_edge") or 0.0), 0.05, abs(float(cfg.get("min_value_edge") or 0.0) - 0.05) < 1e-12)
    check("live.profile", live.get("profile"), EXPECTED_MODEL_PROFILE)
    check("live.selectedCandidate", live.get("selectedCandidate") or live.get("selected_candidate"), EXPECTED_SELECTED_CANDIDATE)
    check("live.params.engine", live_params.get("engine"), "lightgbm")
    check("live.params.train_window", live_params.get("train_window"), "full")
    check("live.params.feature_mode", live_params.get("feature_mode"), "base15")
    check("live.params.edge", float(live_params.get("edge") or 0.0), 0.05, abs(float(live_params.get("edge") or 0.0) - 0.05) < 1e-12)
    check("live.threshold", 0.5 + float(live_params.get("edge") or 0.0), EXPECTED_MODEL_THRESHOLD, abs((0.5 + float(live_params.get("edge") or 0.0)) - EXPECTED_MODEL_THRESHOLD) < 1e-12)
    check("shock.enabled", bool(shock.get("enabled")), True)
    check("shock.profile", shock.get("profile"), EXPECTED_SHOCK_PROFILE)
    check("shock.strategy_profile", shock.get("strategy_profile"), EXPECTED_STRATEGY_PROFILE)
    check("shock.base_model_profile", shock.get("base_model_profile"), EXPECTED_MODEL_PROFILE)
    check("shock.base_selected_candidate", shock.get("base_selected_candidate"), EXPECTED_SELECTED_CANDIDATE)
    check("shock.candidate_id", shock.get("candidate_id"), EXPECTED_SHOCK_CANDIDATE_ID)
    check("shock.params.family", shock_params.get("family"), EXPECTED_SHOCK_FAMILY)
    check("shock.params.action", shock_params.get("action"), EXPECTED_CLUSTER_ACTION)
    check("shock.params.min_cluster_hits", int(shock_params.get("min_cluster_hits") or 0), EXPECTED_MIN_CLUSTER_HITS)
    check("shock.params.shock_score_min", float(shock_params.get("shock_score_min") or 0.0), EXPECTED_SHOCK_THRESHOLD, abs(float(shock_params.get("shock_score_min") or 0.0) - EXPECTED_SHOCK_THRESHOLD) < 1e-12)
    check("shock.params.feature_alignment", shock_params.get("feature_alignment"), EXPECTED_FEATURE_ALIGNMENT)
    check("shock.params.any_clusters", shock_params.get("any_clusters"), EXPECTED_ANY_CLUSTERS)
    router_enabled = bool(router.get("enabled"))
    check("router.enabled", router_enabled, True)
    if router_enabled:
        check("router.profile", router.get("profile"), EXPECTED_ROUTER_PROFILE)
        check("router.candidate_id", router.get("candidate_id"), EXPECTED_ROUTER_CANDIDATE_ID)
        check("router.base_strategy", router.get("base_strategy"), EXPECTED_ROUTER_BASE_STRATEGY)
        check("router.base_shock_candidate_id", router.get("base_shock_candidate_id"), EXPECTED_SHOCK_CANDIDATE_ID)
        check("router.shadow_only", bool(router.get("shadow_only")), EXPECTED_ROUTER_SHADOW_ONLY)
        check("router.model", router.get("model"), EXPECTED_ROUTER_MODEL)
        check("router.model_key", router.get("model_key"), EXPECTED_ROUTER_MODEL_KEY)
        check("router.feature_mode", router.get("feature_mode"), EXPECTED_ROUTER_FEATURE_MODE)
        check("router.model_hyper.kind", router_model_hyper.get("kind"), "lgbm")
        check("router.model_hyper.n_estimators", int(router_model_hyper.get("n_estimators") or 0), 180)
        check("router.model_hyper.learning_rate", float(router_model_hyper.get("learning_rate") or 0.0), 0.028, abs(float(router_model_hyper.get("learning_rate") or 0.0) - 0.028) < 1e-12)
        check("router.model_hyper.num_leaves", int(router_model_hyper.get("num_leaves") or 0), 24)
        check("router.model_hyper.min_child_samples", int(router_model_hyper.get("min_child_samples") or 0), 80)
        check("router.model_hyper.reg_lambda", float(router_model_hyper.get("reg_lambda") or 0.0), 2.0, abs(float(router_model_hyper.get("reg_lambda") or 0.0) - 2.0) < 1e-12)
        check("router.calibC", float(router.get("calibC") or 0.0), 0.75, abs(float(router.get("calibC") or 0.0) - 0.75) < 1e-12)
        check("router.policy.mode", router_policy.get("mode"), EXPECTED_ROUTER_MODE)
        check("router.policy.prob_min", float(router_policy.get("prob_min") or 0.0), EXPECTED_ROUTER_PROB_MIN, abs(float(router_policy.get("prob_min") or 0.0) - EXPECTED_ROUTER_PROB_MIN) < 1e-12)
        check("router.policy.score_min", float(router_policy.get("score_min") or 0.0), EXPECTED_ROUTER_SCORE_MIN, abs(float(router_policy.get("score_min") or 0.0) - EXPECTED_ROUTER_SCORE_MIN) < 1e-12)
        check("router.policy.combo_min", float(router_policy.get("combo_min") or 0.0), EXPECTED_ROUTER_COMBO_MIN, abs(float(router_policy.get("combo_min") or 0.0) - EXPECTED_ROUTER_COMBO_MIN) < 1e-12)
        check("router.policy.combo_alpha", float(router_policy.get("combo_alpha") or 0.0), EXPECTED_ROUTER_COMBO_ALPHA, abs(float(router_policy.get("combo_alpha") or 0.0) - EXPECTED_ROUTER_COMBO_ALPHA) < 1e-12)
        check("router.policy.require_061", bool(router_policy.get("require_061")), True)
        check("router.policy.daily_cap", router_policy.get("daily_cap"), None)
        check("router.policy.daily_cap_mode", router_policy.get("daily_cap_mode"), EXPECTED_ROUTER_DAILY_CAP_MODE)
        check("router.time_policy", router.get("time_policy"), EXPECTED_FEATURE_ALIGNMENT)

    failed = [c for c in checks if not c["ok"]]
    return {
        "ok": not failed,
        "checkedAt": datetime.now(timezone.utc).isoformat(),
        "expectedModelProfile": EXPECTED_MODEL_PROFILE,
        "expectedSelectedCandidate": EXPECTED_SELECTED_CANDIDATE,
        "expectedStrategyProfile": EXPECTED_STRATEGY_PROFILE,
        "expectedShockCandidateId": EXPECTED_SHOCK_CANDIDATE_ID,
        "expectedModelThreshold": EXPECTED_MODEL_THRESHOLD,
        "expectedShockThreshold": EXPECTED_SHOCK_THRESHOLD,
        "expectedShockFamily": EXPECTED_SHOCK_FAMILY,
        "expectedRouterProfile": EXPECTED_ROUTER_PROFILE if router_enabled else None,
        "expectedRouterCandidateId": EXPECTED_ROUTER_CANDIDATE_ID if router_enabled else None,
        "routerEnabled": router_enabled,
        "checks": checks,
        "failed": failed,
    }


def report_cache_freshness(report: dict[str, Any], *, now: datetime | None = None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    signal = report.get("signal") if isinstance(report.get("signal"), dict) else {}
    candidate = report.get("candidate") if isinstance(report.get("candidate"), dict) else {}
    start_raw = signal.get("candidate_start") or candidate.get("candidate_start")
    if not start_raw:
        return {"ok": False, "reason": "missing_candidate_start"}
    try:
        start = datetime.fromisoformat(str(start_raw).replace("Z", "+00:00"))
    except ValueError:
        return {"ok": False, "reason": "invalid_candidate_start", "candidate_start": start_raw}
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    elapsed = (now - start).total_seconds()
    if elapsed < 0:
        return {"ok": False, "reason": "candidate_start_in_future", "elapsedSeconds": elapsed, "candidate_start": start.isoformat()}
    if elapsed > 15 * 60:
        return {"ok": False, "reason": "market_stale", "elapsedSeconds": elapsed, "candidate_start": start.isoformat()}
    return {"ok": True, "reason": "fresh_current_15m_market", "elapsedSeconds": elapsed, "candidate_start": start.isoformat()}


def report_matches_top159_contract(report: dict[str, Any], audit: dict[str, Any] | None = None) -> tuple[bool, list[str]]:
    audit = audit or {}
    reasons: list[str] = []
    if audit and not audit.get("ok"):
        reasons.append("live_contract_audit_failed")
    signal = report.get("signal") if isinstance(report.get("signal"), dict) else {}
    params = signal.get("params") if isinstance(signal.get("params"), dict) else {}
    shock = report.get("shockFilter") if isinstance(report.get("shockFilter"), dict) else {}
    router = report.get("calibrationRouter") if isinstance(report.get("calibrationRouter"), dict) else {}
    candidate = report.get("candidate") if isinstance(report.get("candidate"), dict) else {}
    router_enabled = bool(audit.get("routerEnabled"))

    if params.get("live_model_profile") != EXPECTED_MODEL_PROFILE:
        reasons.append("signal_live_model_profile_mismatch")
    if params.get("selected_candidate") != EXPECTED_SELECTED_CANDIDATE:
        reasons.append("signal_selected_candidate_mismatch")
    if params.get("train_window") != "full":
        reasons.append("signal_train_window_mismatch")
    if params.get("feature_mode") != "base15":
        reasons.append("signal_feature_mode_mismatch")
    try:
        if abs(float(params.get("edge")) - 0.05) > 1e-12:
            reasons.append("signal_edge_mismatch")
    except Exception:
        reasons.append("signal_edge_missing")

    strategy = report.get("strategy_profile") or shock.get("strategy_profile") or candidate.get("strategy_profile")
    if strategy != EXPECTED_STRATEGY_PROFILE:
        reasons.append("strategy_profile_mismatch")
    if shock.get("enabled") is not True:
        reasons.append("shock_not_enabled_in_report")
    if shock.get("candidate_id") != EXPECTED_SHOCK_CANDIDATE_ID:
        reasons.append("shock_candidate_id_mismatch")
    if router_enabled:
        if not router:
            reasons.append("calibration_router_missing_in_report")
        else:
            if router.get("profile") != EXPECTED_ROUTER_PROFILE:
                reasons.append("calibration_router_profile_mismatch")
            if router.get("candidate_id") != EXPECTED_ROUTER_CANDIDATE_ID:
                reasons.append("calibration_router_candidate_id_mismatch")
            if router.get("base_shock_candidate_id") != EXPECTED_SHOCK_CANDIDATE_ID:
                reasons.append("calibration_router_base_shock_mismatch")
            if router.get("shadow_only") is not True:
                reasons.append("calibration_router_shadow_only_mismatch")
            if router.get("router_policy_mode") not in {None, EXPECTED_ROUTER_MODE}:
                reasons.append("calibration_router_policy_mode_mismatch")
            if router.get("router_daily_cap_mode") not in {None, EXPECTED_ROUTER_DAILY_CAP_MODE}:
                reasons.append("calibration_router_daily_cap_mode_mismatch")

    try:
        score = float(signal.get("model_score"))
    except Exception:
        score = None
    action = shock.get("shock_action")
    if score is not None and score >= EXPECTED_MODEL_THRESHOLD and action == "not_evaluated":
        reasons.append("shock_not_evaluated_for_passing_score")
    if action in {"pass", "block"}:
        if shock.get("base_model_profile") != EXPECTED_MODEL_PROFILE:
            reasons.append("shock_base_model_profile_mismatch")
        if shock.get("base_selected_candidate") != EXPECTED_SELECTED_CANDIDATE:
            reasons.append("shock_base_selected_candidate_mismatch")
        try:
            threshold = float(shock.get("shock_gate_threshold"))
            condition = bool(shock.get("shock_condition"))
            expected = EXPECTED_SHOCK_THRESHOLD if condition else EXPECTED_MODEL_THRESHOLD
            if abs(threshold - expected) > 1e-12:
                reasons.append("shock_gate_threshold_mismatch")
        except Exception:
            reasons.append("shock_gate_threshold_missing")
        if shock.get("cluster_action") != EXPECTED_CLUSTER_ACTION:
            reasons.append("cluster_action_mismatch")
        if shock.get("feature_alignment_version") != EXPECTED_FEATURE_ALIGNMENT:
            reasons.append("feature_alignment_mismatch")

    if candidate:
        if candidate.get("live_model_profile") != EXPECTED_MODEL_PROFILE and candidate.get("source") != EXPECTED_MODEL_PROFILE:
            reasons.append("candidate_live_model_profile_mismatch")
        if candidate.get("selected_candidate") != EXPECTED_SELECTED_CANDIDATE:
            reasons.append("candidate_selected_candidate_mismatch")
        if candidate.get("strategy_profile") != EXPECTED_STRATEGY_PROFILE:
            reasons.append("candidate_strategy_profile_mismatch")
        if candidate.get("shock_candidate_id") != EXPECTED_SHOCK_CANDIDATE_ID:
            reasons.append("candidate_shock_candidate_id_mismatch")
        if router_enabled:
            if candidate.get("calibration_router_enabled") is not True:
                reasons.append("candidate_calibration_router_not_enabled")
            if candidate.get("calibration_router_profile") != EXPECTED_ROUTER_PROFILE:
                reasons.append("candidate_calibration_router_profile_mismatch")
            if candidate.get("calibration_router_candidate_id") != EXPECTED_ROUTER_CANDIDATE_ID:
                reasons.append("candidate_calibration_router_candidate_id_mismatch")
            if candidate.get("calibration_router_shadow_only") is not True:
                reasons.append("candidate_calibration_router_shadow_only_mismatch")
    return not reasons, reasons
