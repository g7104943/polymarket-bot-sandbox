from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


ROOT = Path("/Users/mac/polyfun")
NEXT = ROOT / "polyfun-next"
ROUTER_PROFILE = NEXT / "runtime" / "top159_calibration_router_profile.json"
ROUTER_CACHE = NEXT / "runtime" / "top159_calibration_router_cache.pkl"
BASE_ROUTER_SCRIPT = NEXT / "scripts" / "run_top159_061_score_calibration_router.py"
CLUSTER_SCRIPT = NEXT / "scripts" / "run_top159_shock_filter_cluster_targeted_search.py"

EXPECTED_PROFILE = "top159_live_order_calibration_router_v1"
EXPECTED_CANDIDATE_ID = "d3e3e0b3a618dfa3"
EXPECTED_FEATURE_MODE = "atoms_core"
EXPECTED_MODEL_KEY = "9262f79d09e87654"
EXPECTED_TIME_POLICY = "closed_candle_available_at_v2"


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def load_calibration_router_profile(root: Path = NEXT) -> dict[str, Any]:
    path = root / "runtime" / "top159_calibration_router_profile.json"
    payload = load_json(path, {}) or {}
    if not isinstance(payload, dict) or not payload.get("enabled"):
        return {"enabled": False}
    policy = payload.get("policy") if isinstance(payload.get("policy"), dict) else {}
    model_hyper = payload.get("model_hyper") if isinstance(payload.get("model_hyper"), dict) else {}
    return {
        **payload,
        "shadow_only": bool(payload.get("shadow_only", True)),
        "feature_mode": str(payload.get("feature_mode") or EXPECTED_FEATURE_MODE),
        "model_hyper": model_hyper,
        "policy": policy,
    }


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def router_cache_key(profile: dict[str, Any]) -> str:
    payload = {
        "profile": profile.get("profile"),
        "candidate_id": profile.get("candidate_id"),
        "feature_mode": profile.get("feature_mode"),
        "model_key": profile.get("model_key"),
        "model_hyper": profile.get("model_hyper"),
        "calibC": profile.get("calibC"),
        "policy": profile.get("policy"),
        "time_policy": profile.get("time_policy") or EXPECTED_TIME_POLICY,
        "code_version": "live_order_router_shadow_v1",
    }
    return stable_hash(payload)


def make_model(model_hyper: dict[str, Any]):
    kind = str(model_hyper.get("kind") or model_hyper.get("engine") or "lgbm").lower()
    if kind in {"lgbm", "lightgbm"}:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=int(model_hyper.get("n_estimators", 180)),
            learning_rate=float(model_hyper.get("learning_rate", 0.028)),
            num_leaves=int(model_hyper.get("num_leaves", 24)),
            min_child_samples=int(model_hyper.get("min_child_samples", 80)),
            reg_lambda=float(model_hyper.get("reg_lambda", 2.0)),
            random_state=20260511,
            n_jobs=1,
            verbose=-1,
        )
    if kind in {"logit", "logistic"}:
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        return make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(C=float(model_hyper.get("C", 1.0)), penalty="l2", max_iter=1000, solver="lbfgs"),
        )
    raise ValueError(f"unsupported calibration router model kind: {kind}")


def load_or_fit_router(profile: dict[str, Any]) -> tuple[Any, Any, list[str], dict[str, Any]]:
    key = router_cache_key(profile)
    if ROUTER_CACHE.exists():
        try:
            with ROUTER_CACHE.open("rb") as fh:
                cached = pickle.load(fh)
            if isinstance(cached, dict) and cached.get("cache_key") == key:
                meta = dict(cached.get("meta") or {})
                meta["cache"] = "hit"
                return cached["model"], cached["calibrator"], list(cached["feature_cols"]), meta
        except Exception:
            pass

    base = _load_module("top159_calibration_router_base_live", BASE_ROUTER_SCRIPT)
    _enriched, atom_store, period_vals = base.build_truth()
    train_period = "gate_train_for_365d"
    train_df = period_vals[train_period].reset_index(drop=True)
    train_idx, calib_idx = base.split_train_calib(train_df)
    train_x_all = base.make_feature_frame(train_df, str(profile.get("feature_mode") or EXPECTED_FEATURE_MODE), atom_store[train_period])
    core_x = train_x_all.loc[train_idx]
    calib_x = train_x_all.loc[calib_idx].reindex(columns=core_x.columns, fill_value=0.0)
    y_core = train_df.loc[train_idx, "won"].astype(bool).astype(int).to_numpy()
    y_calib = train_df.loc[calib_idx, "won"].astype(bool).astype(int).to_numpy()

    model = make_model(profile.get("model_hyper") if isinstance(profile.get("model_hyper"), dict) else {})
    model.fit(core_x, y_core)
    raw_calib = model.predict_proba(calib_x)[:, 1]
    calib_c = float(profile.get("calibC") or 0.75)
    calibrator = LogisticRegression(C=calib_c, penalty="l2", max_iter=1000, solver="lbfgs")
    calibrator.fit(raw_calib.reshape(-1, 1), y_calib)
    meta = {
        "cache": "miss",
        "cache_key": key,
        "trainPeriod": train_period,
        "trainRows": int(len(core_x)),
        "calibRows": int(len(calib_x)),
        "trainStart": str(pd.to_datetime(train_df["dt"], utc=True, errors="coerce").min()),
        "trainEnd": str(pd.to_datetime(train_df["dt"], utc=True, errors="coerce").max()),
        "featureCount": int(len(core_x.columns)),
        "featureMode": profile.get("feature_mode"),
        "modelHyper": profile.get("model_hyper"),
        "calibC": calib_c,
    }
    ROUTER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = ROUTER_CACHE.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(
            {
                "cache_key": key,
                "model": model,
                "calibrator": calibrator,
                "feature_cols": list(core_x.columns),
                "meta": meta,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            },
            fh,
        )
    tmp.replace(ROUTER_CACHE)
    return model, calibrator, list(core_x.columns), meta


def live_feature_frame(live_row: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    base = _load_module("top159_calibration_router_base_feature_live", BASE_ROUTER_SCRIPT)
    cluster = _load_module("top159_calibration_router_cluster_live", CLUSTER_SCRIPT)
    x = base.make_feature_frame(live_row.reset_index(drop=True), "base_num", None)
    atom_values = cluster.atom_masks(live_row.reset_index(drop=True))
    row = {}
    for col in feature_cols:
        if col in x.columns:
            row[col] = float(pd.to_numeric(x[col], errors="coerce").fillna(0.0).iloc[0])
        elif col.startswith("atom__"):
            atom = col[len("atom__") :]
            arr = atom_values.get(atom)
            row[col] = float(bool(arr[0])) if arr is not None and len(arr) else 0.0
        else:
            row[col] = 0.0
    return pd.DataFrame([row], columns=feature_cols).replace([np.inf, -np.inf], 0.0).fillna(0.0)


def evaluate_calibration_router(
    *,
    root: Path = NEXT,
    live_row: pd.DataFrame | None,
    signal: dict[str, Any],
    base_061_allows: bool,
) -> dict[str, Any]:
    profile = load_calibration_router_profile(root)
    if not profile.get("enabled"):
        return {
            "enabled": False,
            "router_action": "not_enabled",
            "router_allows_live": True,
        }
    policy = profile.get("policy") if isinstance(profile.get("policy"), dict) else {}
    shadow_only = bool(profile.get("shadow_only", True))
    require_061 = bool(policy.get("require_061", True))
    if require_061 and not base_061_allows:
        return _result(profile, shadow_only, "not_evaluated", "base_061_not_allowed", None, None, None, policy, True)
    if live_row is None or live_row.empty:
        return _result(profile, shadow_only, "not_evaluated", "missing_live_feature_row", None, None, None, policy, True)

    model, calibrator, cols, meta = load_or_fit_router(profile)
    x = live_feature_frame(live_row, cols)
    raw = float(model.predict_proba(x)[:, 1][0])
    prob = float(calibrator.predict_proba(np.array([[raw]], dtype=float))[:, 1][0])
    score = float(signal.get("model_score") or live_row.get("score15", pd.Series([0.0])).iloc[0] or 0.0)
    mode = str(policy.get("mode") or "combo")
    score_min = float(policy.get("score_min") or 0.55)
    prob_min = float(policy.get("prob_min") or 0.535)
    combo_alpha = float(policy.get("combo_alpha") or 0.0)
    combo_threshold = float(policy.get("combo_min") or prob_min)
    combo_score = prob + combo_alpha * (score - 0.55)
    if mode == "combo":
        passed = bool(score >= score_min and combo_score >= combo_threshold)
        required_probability = prob_min
    elif mode == "plain":
        passed = bool(score >= score_min and prob >= prob_min)
        required_probability = prob_min
    else:
        # The approved live-order candidate is combo. Unsupported future modes
        # fail closed in真钱 mode, but shadow mode only records the issue.
        passed = False
        required_probability = prob_min
    if shadow_only:
        action = "shadow_pass" if passed else "shadow_block"
        allows_live = True
    else:
        action = "pass" if passed else "block"
        allows_live = passed
    reason = "router_passed" if passed else "router_blocked"
    out = _result(profile, shadow_only, action, reason, prob, required_probability, combo_score, policy, allows_live)
    out.update({
        "router_combo_threshold": combo_threshold,
        "router_score_min": score_min,
        "router_model_raw_probability": raw,
        "router_cache": meta,
        "router_feature_count": int(len(cols)),
    })
    return out


def _result(
    profile: dict[str, Any],
    shadow_only: bool,
    action: str,
    reason: str,
    probability: float | None,
    required_probability: float | None,
    combo_score: float | None,
    policy: dict[str, Any],
    allows_live: bool,
) -> dict[str, Any]:
    return {
        "enabled": True,
        "profile": profile.get("profile"),
        "strategy_profile": profile.get("strategy_profile") or profile.get("profile"),
        "candidate_id": profile.get("candidate_id"),
        "base_strategy": profile.get("base_strategy"),
        "base_shock_candidate_id": profile.get("base_shock_candidate_id"),
        "shadow_only": shadow_only,
        "model_key": profile.get("model_key"),
        "feature_mode": profile.get("feature_mode"),
        "model_hyper": profile.get("model_hyper") if isinstance(profile.get("model_hyper"), dict) else {},
        "calibC": profile.get("calibC"),
        "policy": policy,
        "router_probability": probability,
        "router_required_probability": required_probability,
        "router_combo_score": combo_score,
        "router_combo_threshold": policy.get("combo_min"),
        "router_policy_mode": policy.get("mode"),
        "router_daily_cap_mode": policy.get("daily_cap_mode") or "chronological",
        "router_daily_cap": policy.get("daily_cap"),
        "router_action": action,
        "router_reason": reason,
        "router_allows_live": allows_live,
        "time_policy": profile.get("time_policy") or EXPECTED_TIME_POLICY,
    }


def router_candidate_fields(router: dict[str, Any]) -> dict[str, Any]:
    if not router.get("enabled"):
        return {
            "calibration_router_enabled": False,
        }
    return {
        "calibration_router_enabled": True,
        "calibration_router_profile": router.get("profile"),
        "calibration_router_candidate_id": router.get("candidate_id"),
        "calibration_router_shadow_only": bool(router.get("shadow_only")),
        "router_probability": router.get("router_probability"),
        "router_required_probability": router.get("router_required_probability"),
        "router_combo_score": router.get("router_combo_score"),
        "router_combo_threshold": router.get("router_combo_threshold"),
        "router_policy_mode": router.get("router_policy_mode"),
        "router_daily_cap_mode": router.get("router_daily_cap_mode"),
        "router_action": router.get("router_action"),
        "router_reason": router.get("router_reason"),
        "router_model_key": router.get("model_key"),
        "router_feature_mode": router.get("feature_mode"),
    }

