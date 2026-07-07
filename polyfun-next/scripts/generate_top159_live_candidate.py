#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pickle
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path('/Users/mac/polyfun')
NEXT = ROOT / 'polyfun-next'
REPORTS = ROOT / 'reports'
BASE_SCRIPT = ROOT / 'scripts' / 'ops' / 'run_crypto15m_1h_multasset_pressure_search_latest.py'
FILL_SCRIPT = ROOT / 'scripts' / 'ops' / 'run_newslot1_fill_rate_toxicity_search_latest.py'
INTEGRATED_SCRIPT = NEXT / 'scripts' / 'run_top159_integrated_main_model_24h_search.py'
SHOCK_SCRIPT = NEXT / 'scripts' / 'run_top159_shock_candle_filter_research.py'
MODEL_CACHE = NEXT / 'runtime' / 'top159_model_cache.pkl'
LIVE_MODEL_PROFILE = NEXT / 'runtime' / 'top159_live_model_profile.json'
SHOCK_PROFILE = NEXT / 'runtime' / 'top159_shock_filter_profile.json'
SHOCK_GATE_CACHE = NEXT / 'runtime' / 'top159_shock_gate_cache.pkl'
SHOCK_FEATURE_ALIGNMENT_VERSION = 'closed_candle_available_at_v2'
sys.path.insert(0, str(NEXT / 'src'))

from polyfun_next.top159_contract import audit_top159_contract
from polyfun_next.calibration_router import (
    evaluate_calibration_router,
    load_calibration_router_profile,
    router_candidate_fields,
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def top159_params() -> dict[str, Any]:
    profile = load_live_model_profile()
    if profile:
        return profile
    report = ROOT / 'reports' / 'newslot1_fak_execution_loop_latest.json'
    if report.exists():
        data = json.loads(report.read_text())
        params = data.get('uniqueVerdict', {}).get('selectedParams')
        if isinstance(params, dict):
            return params
    return {
        'engine': 'lightgbm', 'train_window': '5y', 'feature_mode': 'trend', 'edge': 0.045,
        'vol_q': 0.999, 'trend_mode': 'none', 'bb_abs_max': 2.0, 'loss_n': 0, 'skip_k': 4,
        'n_estimators': 200, 'learning_rate': 0.02193585345721919, 'reg_lambda': 0.0907758387860903,
        'subsample': 0.926502583070262, 'colsample_bytree': 0.8764270068535669,
        'num_leaves': 36, 'min_child_samples': 80, 'depth': 3,
    }


def load_live_model_profile() -> dict[str, Any] | None:
    """Optional live model override.

    This lets live switch from the original top159 model to a research-vetted
    integrated model without changing execution, claim, FAK, or risk code.
    """
    if not LIVE_MODEL_PROFILE.exists():
        return None
    try:
        payload = json.loads(LIVE_MODEL_PROFILE.read_text(encoding='utf-8'))
    except Exception as exc:
        raise RuntimeError(f'live model profile exists but cannot be read: {LIVE_MODEL_PROFILE}') from exc
    if not payload.get('enabled'):
        return None
    params = payload.get('params')
    if not isinstance(params, dict):
        raise RuntimeError(f'live model profile is enabled but missing params: {LIVE_MODEL_PROFILE}')
    required = ('edge', 'engine', 'train_window', 'feature_mode')
    missing = [key for key in required if key not in params]
    if missing:
        raise RuntimeError(f'live model profile is enabled but missing params {missing}: {LIVE_MODEL_PROFILE}')
    out = dict(params)
    out['live_model_profile'] = payload.get('profile') or payload.get('name') or 'custom_live_profile'
    out['live_model_source'] = str(LIVE_MODEL_PROFILE)
    out['selected_candidate'] = payload.get('selectedCandidate') or payload.get('selected_candidate')
    return out


def load_shock_filter_profile() -> dict[str, Any]:
    if not SHOCK_PROFILE.exists():
        return {'enabled': False}
    try:
        payload = json.loads(SHOCK_PROFILE.read_text(encoding='utf-8'))
    except Exception as exc:
        raise RuntimeError(f'shock filter profile exists but cannot be read: {SHOCK_PROFILE}') from exc
    if not payload.get('enabled'):
        return {'enabled': False, **payload}
    params = payload.get('params')
    if not isinstance(params, dict):
        raise RuntimeError(f'shock filter enabled but missing params: {SHOCK_PROFILE}')
    family = str(params.get('family') or 'model_gate')
    if family == 'cluster_gate':
        required = ('action', 'min_cluster_hits', 'shock_score_min')
        if not (isinstance(params.get('any_clusters'), list) or isinstance(params.get('atoms'), list)):
            raise RuntimeError(f'cluster shock filter enabled but missing any_clusters/atoms: {SHOCK_PROFILE}')
    else:
        required = ('rule_mode', 'body_min', 'range_q_min', 'volume_mult_min', 'model_engine', 'min_gate_win_prob', 'model_hyper')
    missing = [key for key in required if key not in params]
    if missing:
        raise RuntimeError(f'shock filter enabled but missing params {missing}: {SHOCK_PROFILE}')
    return payload


def shock_cache_key(profile: dict[str, Any]) -> str:
    payload = {
        'feature_alignment_version': SHOCK_FEATURE_ALIGNMENT_VERSION,
        'profile': profile.get('profile'),
        'strategy_profile': profile.get('strategy_profile'),
        'base_model_profile': profile.get('base_model_profile'),
        'base_selected_candidate': profile.get('base_selected_candidate'),
        'params': profile.get('params'),
        'sourceReport': profile.get('sourceReport'),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode('utf-8')).hexdigest()


def load_or_fit_shock_gate(shock_mod, profile: dict[str, Any]) -> tuple[Any, list[str], dict[str, Any]]:
    cache_key = shock_cache_key(profile)
    if SHOCK_GATE_CACHE.exists():
        try:
            with SHOCK_GATE_CACHE.open('rb') as fh:
                cached = pickle.load(fh)
            if isinstance(cached, dict) and cached.get('cache_key') == cache_key:
                return cached.get('model'), list(cached.get('cols') or []), {**(cached.get('meta') or {}), 'cache': 'hit'}
        except Exception:
            pass
    selected, truth = shock_mod.build_selected_sets()
    enriched = shock_mod.enrich_shock_features(selected)
    train = enriched[enriched['period_name'] == 'gate_train_for_365d'].copy()
    shock_params = profile.get('params') if isinstance(profile.get('params'), dict) else {}
    model, cols, fit_meta = fit_shock_gate_from_profile(shock_mod, train, shock_params)
    if model is None:
        raise RuntimeError('shock gate model failed to fit from gate_train_for_365d')
    meta = {
        'cache': 'miss',
        'feature_alignment_version': SHOCK_FEATURE_ALIGNMENT_VERSION,
        **fit_meta,
        'train_rows': int(len(train)),
        'train_start': str(pd.to_datetime(train['dt'], utc=True, errors='coerce').min()),
        'train_end': str(pd.to_datetime(train['dt'], utc=True, errors='coerce').max()),
        'feature_count': int(len(cols)),
        'integrated_data_truth': truth,
    }
    SHOCK_GATE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = SHOCK_GATE_CACHE.with_suffix('.tmp')
    with tmp.open('wb') as fh:
        pickle.dump({'cache_key': cache_key, 'model': model, 'cols': cols, 'meta': meta, 'saved_at': datetime.now(timezone.utc).isoformat()}, fh)
    tmp.replace(SHOCK_GATE_CACHE)
    return model, list(cols), meta


def fit_shock_gate_from_profile(shock_mod, train: pd.DataFrame, shock_params: dict[str, Any]) -> tuple[Any, list[str], dict[str, Any]]:
    """Fit the live shock gate using the exact model family in the profile.

    The overnight winner uses logistic regression, while the previous live gate
    used LightGBM. Keeping this here avoids a dangerous situation where the JSON
    says "logistic" but live silently trains the old LightGBM model.
    """
    model_hyper = shock_params.get('model_hyper') if isinstance(shock_params.get('model_hyper'), dict) else {}
    engine = str(model_hyper.get('engine') or shock_params.get('model_engine') or 'lightgbm').lower()
    x, cols = shock_mod.model_features(train)
    y = train['won'].astype(int).to_numpy()
    if len(np.unique(y)) < 2 or len(train) < 200:
        return None, cols, {'model_engine': engine, 'model_hyper': model_hyper, 'fit_error': 'insufficient_classes_or_rows'}
    if engine == 'logistic':
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        c_val = float(model_hyper.get('C', 1.0))
        model = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(max_iter=700, C=c_val, random_state=getattr(shock_mod, 'RNG_SEED', 20260503)),
        )
    elif engine == 'lightgbm':
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=int(model_hyper.get('n_estimators', 90)),
            learning_rate=float(model_hyper.get('learning_rate', 0.045)),
            num_leaves=int(model_hyper.get('num_leaves', 16)),
            min_child_samples=int(model_hyper.get('min_child_samples', 50)),
            subsample=float(model_hyper.get('subsample', 0.9)),
            colsample_bytree=float(model_hyper.get('colsample_bytree', 0.9)),
            reg_lambda=float(model_hyper.get('reg_lambda', 1.0)),
            random_state=getattr(shock_mod, 'RNG_SEED', 20260503),
            n_jobs=2,
            verbose=-1,
        )
    else:
        raise ValueError(f'unsupported shock gate model engine: {engine}')
    model.fit(x, y)
    return model, cols, {'model_engine': engine, 'model_hyper': model_hyper}


def enrich_live_shock_features(shock_mod, base, signal: dict[str, Any]) -> pd.DataFrame:
    row = pd.DataFrame([{
        'dt': pd.Timestamp(signal['candidate_start']),
        'label_up': np.nan,
        'pred_up15': bool(signal['side'] == 'UP'),
        'direction': str(signal['side']),
        'score15': float(signal['model_score']),
        'won': False,
        'period_name': 'live',
    }])
    raw15 = merge_recent_closed_binance(base.load_raw('ETH', '15m').sort_values('dt').copy(), '15m')
    raw1h = merge_recent_closed_binance(base.load_raw('ETH', '1h').sort_values('dt').copy(), '1h')
    raw4h = merge_recent_closed_binance(base.load_raw('ETH', '4h').sort_values('dt').copy(), '4h')
    feat15 = shock_mod.candle_features(raw15, '15m')
    feat1h = shock_mod.candle_features(raw1h, '1h')
    feat4h = shock_mod.candle_features(raw4h, '4h')
    out = row.sort_values('dt').copy()
    out['ts_ns'] = pd.to_datetime(out['dt'], utc=True).map(lambda x: pd.Timestamp(x).value).astype('int64')
    merge_base = out[['ts_ns']].sort_values('ts_ns')
    for tf, feats in [('15m', feat15), ('1h', feat1h), ('4h', feat4h)]:
        use = feats.sort_values('ts_ns').drop(columns=['dt'])
        merged = pd.merge_asof(merge_base, use, on='ts_ns', direction='backward', allow_exact_matches=True).sort_index()
        for c in use.columns:
            if c != 'ts_ns':
                out[c] = merged[c].to_numpy()
    out = out.drop(columns=['ts_ns'])
    for tf in ['15m', '1h', '4h']:
        cdir = out[f'{tf}_candle_dir']
        out[f'{tf}_same_as_top159'] = ((out['direction'] == 'UP') & (cdir == 'up')) | ((out['direction'] == 'DOWN') & (cdir == 'down'))
        out[f'{tf}_opposes_top159'] = ((out['direction'] == 'UP') & (cdir == 'down')) | ((out['direction'] == 'DOWN') & (cdir == 'up'))
        out[f'{tf}_terminal_chase'] = ((out['direction'] == 'UP') & (out[f'{tf}_pos20'] >= 0.75)) | ((out['direction'] == 'DOWN') & (out[f'{tf}_pos20'] <= 0.25))
        out[f'{tf}_exhaustion_wick'] = ((out['direction'] == 'UP') & (out[f'{tf}_upper_wick_ratio'] >= 0.35)) | ((out['direction'] == 'DOWN') & (out[f'{tf}_lower_wick_ratio'] >= 0.35))
    return out


def evaluate_shock_filter(base, signal: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    profile = load_shock_filter_profile()
    if not profile.get('enabled'):
        return {'enabled': False, 'strategy_profile': params.get('live_model_profile') or 'top159'}
    shock_mod = _load_module('shock_filter_live', SHOCK_SCRIPT)
    shock_params = dict(profile['params'])
    live_row = enrich_live_shock_features(shock_mod, base, signal)
    if str(shock_params.get('family') or 'model_gate') == 'cluster_gate':
        return evaluate_cluster_shock_filter(profile, shock_params, live_row, signal, params)
    condition = bool(shock_mod.condition_mask(live_row, shock_params).iloc[0])
    threshold = float(shock_params.get('min_gate_win_prob', 0.50))
    probability = None
    cache_meta = {}
    if condition:
        model, cols, cache_meta = load_or_fit_shock_gate(shock_mod, profile)
        probability = float(shock_mod.predict_gate(model, cols, live_row)[0])
        action = 'pass' if probability >= threshold else 'block'
        reason = 'shock_gate_probability_passed' if action == 'pass' else 'shock_filter_blocked'
    else:
        action = 'pass'
        reason = 'no_shock_condition'
    return {
        'enabled': True,
        'strategy_profile': profile.get('strategy_profile') or profile.get('profile') or 'new_shock_filter_top159',
        'profile': profile.get('profile') or 'new_shock_filter_top159',
        'candidate_id': profile.get('candidate_id'),
        'shock_candidate_id': profile.get('candidate_id'),
        'model_engine': (shock_params.get('model_hyper') or {}).get('engine') or shock_params.get('model_engine'),
        'model_hyper': shock_params.get('model_hyper') if isinstance(shock_params.get('model_hyper'), dict) else {},
        'base_model_profile': profile.get('base_model_profile') or params.get('live_model_profile'),
        'base_selected_candidate': profile.get('base_selected_candidate') or params.get('selected_candidate'),
        'shock_condition': condition,
        'shock_gate_probability': probability,
        'shock_gate_threshold': threshold,
        'shock_action': action,
        'shock_reason': reason,
        'shock_params': shock_params,
        'cache': cache_meta,
        'features': {
            '1h_body_ratio': _safe_float(live_row.get('1h_body_ratio', pd.Series([None])).iloc[0]),
            '1h_range_q': _safe_float(live_row.get('1h_range_q', pd.Series([None])).iloc[0]),
            '4h_body_ratio': _safe_float(live_row.get('4h_body_ratio', pd.Series([None])).iloc[0]),
            '4h_range_q': _safe_float(live_row.get('4h_range_q', pd.Series([None])).iloc[0]),
            '1h_candle_dir': str(live_row.get('1h_candle_dir', pd.Series([''])).iloc[0]),
            '4h_candle_dir': str(live_row.get('4h_candle_dir', pd.Series([''])).iloc[0]),
        },
    }


def parse_cluster_numeric_suffix(raw: str) -> float:
    text = str(raw).strip()
    if text.isdigit() and len(text) == 3:
        return int(text) / 100.0
    return float(text.replace('p', '.'))


def cluster_atom_value(row: pd.Series, atom: str) -> bool:
    atom = str(atom)
    if atom in {'dir_UP', 'dir_DOWN'}:
        return str(row.get('direction') or '').upper() == atom.split('_', 1)[1]
    if atom.startswith('score_lt_'):
        try:
            return float(row.get('score15') or 0.0) < parse_cluster_numeric_suffix(atom.rsplit('_', 1)[1])
        except Exception:
            return False
    if atom.startswith('score_ge_'):
        try:
            return float(row.get('score15') or 0.0) >= parse_cluster_numeric_suffix(atom.rsplit('_', 1)[1])
        except Exception:
            return False
    for tf in ['15m', '1h', '4h']:
        prefix = f'{tf}_'
        if not atom.startswith(prefix):
            continue
        rest = atom[len(prefix):]
        if rest in {'same_as_top159', 'opposes_top159', 'terminal_chase', 'exhaustion_wick'}:
            return bool(row.get(atom))
        if rest == 'candle_up':
            return str(row.get(f'{tf}_candle_dir') or '') == 'up'
        if rest == 'candle_down':
            return str(row.get(f'{tf}_candle_dir') or '') == 'down'
        if rest in {'trend_up', 'trend_down', 'trend_mixed'}:
            return str(row.get(f'{tf}_trend_state') or '') == rest.replace('trend_', '')
        if rest == 'pos_high':
            return _safe_float(row.get(f'{tf}_pos20')) is not None and float(row.get(f'{tf}_pos20')) >= 0.75
        if rest == 'pos_low':
            return _safe_float(row.get(f'{tf}_pos20')) is not None and float(row.get(f'{tf}_pos20')) <= 0.25
        if rest.startswith('vol_ge_'):
            try:
                return float(row.get(f'{tf}_volume_mult') or 0.0) >= float(rest.replace('vol_ge_', ''))
            except Exception:
                return False
        if rest.startswith('body_ge_'):
            try:
                return float(row.get(f'{tf}_body_ratio') or 0.0) >= float(rest.replace('body_ge_', ''))
            except Exception:
                return False
        if rest.startswith('rangeq_ge_'):
            try:
                return float(row.get(f'{tf}_range_q') or 0.0) >= float(rest.replace('rangeq_ge_', ''))
            except Exception:
                return False
    return False


def cluster_hits_for_params(live_row: pd.DataFrame, shock_params: dict[str, Any]) -> tuple[list[dict[str, Any]], int, bool]:
    row = live_row.iloc[0]
    global_atoms = list(shock_params.get('global_atoms') or [])
    global_ok = all(cluster_atom_value(row, atom) for atom in global_atoms)
    clusters: list[list[str]]
    if isinstance(shock_params.get('any_clusters'), list):
        clusters = [list(x) for x in shock_params.get('any_clusters') or []]
    else:
        clusters = [list(shock_params.get('atoms') or [])]
    hits: list[dict[str, Any]] = []
    for atoms in clusters:
        atom_results = {atom: cluster_atom_value(row, atom) for atom in atoms}
        hit = bool(atom_results) and all(atom_results.values()) and global_ok
        if hit:
            hits.append({'atoms': atoms, 'atom_results': atom_results})
    min_hits = int(shock_params.get('min_cluster_hits') or 1)
    condition = len(hits) >= min_hits
    return hits, min_hits, condition


def evaluate_cluster_shock_filter(profile: dict[str, Any], shock_params: dict[str, Any], live_row: pd.DataFrame, signal: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    hits, min_hits, condition = cluster_hits_for_params(live_row, shock_params)
    edge_raw = params.get('edge')
    base_threshold = 0.5 + float(0.05 if edge_raw is None else edge_raw)
    raised_threshold = float(shock_params.get('shock_score_min') or base_threshold)
    required_score = raised_threshold if condition and shock_params.get('action') == 'raise_score' else base_threshold
    score = float(signal.get('model_score') or 0.0)
    if condition and shock_params.get('action') == 'hard_block':
        action = 'block'
        reason = 'cluster_gate_hard_block'
    elif score >= required_score:
        action = 'pass'
        reason = 'cluster_gate_score_passed' if condition else 'no_cluster_condition'
    else:
        action = 'block'
        reason = 'cluster_gate_score_blocked'
    return {
        'enabled': True,
        'strategy_profile': profile.get('strategy_profile') or profile.get('profile') or 'new_shock_filter_top159_cluster',
        'profile': profile.get('profile') or 'new_shock_filter_top159_cluster',
        'candidate_id': profile.get('candidate_id'),
        'shock_candidate_id': profile.get('candidate_id'),
        'model_engine': 'cluster_gate',
        'model_hyper': {},
        'base_model_profile': profile.get('base_model_profile') or params.get('live_model_profile'),
        'base_selected_candidate': profile.get('base_selected_candidate') or params.get('selected_candidate'),
        'shock_condition': condition,
        'shock_gate_probability': None,
        'shock_gate_threshold': required_score,
        'shock_required_score': required_score,
        'shock_base_threshold': base_threshold,
        'shock_action': action,
        'shock_reason': reason,
        'shock_params': shock_params,
        'cluster_action': shock_params.get('action'),
        'cluster_min_hits': min_hits,
        'cluster_hit_count': len(hits),
        'cluster_hits': hits,
        'feature_alignment_version': SHOCK_FEATURE_ALIGNMENT_VERSION,
        'features': {
            '15m_candle_dir': str(live_row.get('15m_candle_dir', pd.Series([''])).iloc[0]),
            '1h_candle_dir': str(live_row.get('1h_candle_dir', pd.Series([''])).iloc[0]),
            '4h_candle_dir': str(live_row.get('4h_candle_dir', pd.Series([''])).iloc[0]),
            '1h_volume_mult': _safe_float(live_row.get('1h_volume_mult', pd.Series([None])).iloc[0]),
            '4h_volume_mult': _safe_float(live_row.get('4h_volume_mult', pd.Series([None])).iloc[0]),
            '1h_pos20': _safe_float(live_row.get('1h_pos20', pd.Series([None])).iloc[0]),
            '1h_trend_state': str(live_row.get('1h_trend_state', pd.Series([''])).iloc[0]),
        },
        'cache': {'cache': 'not_required', 'feature_alignment_version': SHOCK_FEATURE_ALIGNMENT_VERSION},
    }


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def router_error_result(exc: Exception) -> dict[str, Any]:
    profile = load_calibration_router_profile(NEXT)
    if not profile.get('enabled'):
        return {'enabled': False, 'router_action': 'not_enabled', 'router_allows_live': True}
    shadow_only = bool(profile.get('shadow_only', True))
    return {
        'enabled': True,
        'profile': profile.get('profile'),
        'strategy_profile': profile.get('strategy_profile') or profile.get('profile'),
        'candidate_id': profile.get('candidate_id'),
        'base_strategy': profile.get('base_strategy'),
        'base_shock_candidate_id': profile.get('base_shock_candidate_id'),
        'shadow_only': shadow_only,
        'model_key': profile.get('model_key'),
        'feature_mode': profile.get('feature_mode'),
        'model_hyper': profile.get('model_hyper') if isinstance(profile.get('model_hyper'), dict) else {},
        'calibC': profile.get('calibC'),
        'policy': profile.get('policy') if isinstance(profile.get('policy'), dict) else {},
        'router_probability': None,
        'router_required_probability': None,
        'router_combo_score': None,
        'router_combo_threshold': (profile.get('policy') or {}).get('combo_min') if isinstance(profile.get('policy'), dict) else None,
        'router_policy_mode': (profile.get('policy') or {}).get('mode') if isinstance(profile.get('policy'), dict) else None,
        'router_daily_cap_mode': (profile.get('policy') or {}).get('daily_cap_mode') if isinstance(profile.get('policy'), dict) else 'chronological',
        'router_action': 'shadow_error' if shadow_only else 'error',
        'router_reason': f'router_exception:{exc!r}',
        'router_allows_live': True if shadow_only else False,
        'time_policy': profile.get('time_policy'),
    }


def router_not_evaluated_result(reason: str) -> dict[str, Any]:
    profile = load_calibration_router_profile(NEXT)
    if not profile.get('enabled'):
        return {'enabled': False, 'router_action': 'not_enabled', 'router_reason': reason, 'router_allows_live': True}
    policy = profile.get('policy') if isinstance(profile.get('policy'), dict) else {}
    return {
        'enabled': True,
        'profile': profile.get('profile'),
        'strategy_profile': profile.get('strategy_profile') or profile.get('profile'),
        'candidate_id': profile.get('candidate_id'),
        'base_strategy': profile.get('base_strategy'),
        'base_shock_candidate_id': profile.get('base_shock_candidate_id'),
        'shadow_only': bool(profile.get('shadow_only', True)),
        'model_key': profile.get('model_key'),
        'feature_mode': profile.get('feature_mode'),
        'model_hyper': profile.get('model_hyper') if isinstance(profile.get('model_hyper'), dict) else {},
        'calibC': profile.get('calibC'),
        'policy': policy,
        'router_probability': None,
        'router_required_probability': policy.get('prob_min'),
        'router_combo_score': None,
        'router_combo_threshold': policy.get('combo_min'),
        'router_policy_mode': policy.get('mode'),
        'router_daily_cap_mode': policy.get('daily_cap_mode') or 'chronological',
        'router_action': 'not_evaluated',
        'router_reason': reason,
        'router_allows_live': True,
        'time_policy': profile.get('time_policy'),
    }


def current_top159_signal(base, fill, params: dict[str, Any]) -> dict[str, Any]:
    if str(params.get('live_model_profile') or '').startswith('integrated_main'):
        integrated = _load_module('integrated_main_live', INTEGRATED_SCRIPT)
        return current_integrated_main_signal(base, integrated, params)
    raw = base.load_raw('ETH', '15m')
    raw = raw.sort_values('dt').copy()
    raw = merge_recent_closed_binance_15m(raw)
    last = raw.iloc[-1].copy()
    next_dt = pd.Timestamp(last['dt']) + pd.Timedelta(minutes=15)
    future = last.copy()
    future['dt'] = next_dt
    future['date'] = next_dt
    # Use last known close as a placeholder. Feature builder shifts all price features, so this row's own close is not used as a future feature.
    raw2 = pd.concat([raw, pd.DataFrame([future])], ignore_index=True)
    df, features = base.build_features(raw2, '15m')
    row = df[df['dt'] == next_dt]
    if row.empty:
        raise RuntimeError('failed to build live feature row')
    feats = fill.feature_subset(features, params['feature_mode'])
    train_days = 1825 if params.get('train_window') == '5y' else fill.TRAIN_DAYS.get(params.get('train_window'), 1825)
    train = df[(df['dt'] < next_dt) & (df['dt'] >= next_dt - pd.Timedelta(days=train_days))].copy()
    if len(train) > fill.MAX_TRAIN_ROWS:
        train = train.sort_values('dt').iloc[-fill.MAX_TRAIN_ROWS:].copy()
    cache_key = model_cache_key(params, feats, train)
    model = None
    cache_status = 'miss'
    try:
        cached = load_model_cache(cache_key)
        if cached is not None:
            model = cached
            cache_status = 'hit'
    except Exception:
        cache_status = 'load_failed'
    if model is None:
        model = fill.fit_model(params['engine'], train, feats, params)
        if model is not None:
            try:
                save_model_cache(cache_key, model)
            except Exception:
                cache_status = 'save_failed'
    if model is None:
        raise RuntimeError('top159 live model failed to fit')
    prob_up = float(fill.predict(model, row, feats)[0])
    side = 'UP' if prob_up >= 0.5 else 'DOWN'
    model_score = max(prob_up, 1.0 - prob_up)
    return {
        'candidate_start': next_dt.to_pydatetime().replace(tzinfo=timezone.utc).isoformat(),
        'prob_up': prob_up,
        'side': side,
        'model_score': model_score,
        'params': params,
        'feature_count': len(feats),
        'train_rows': int(len(train)),
        'raw_last_dt': pd.Timestamp(last['dt']).isoformat(),
        'model_cache': cache_status,
    }


def current_integrated_main_signal(base, integrated, params: dict[str, Any]) -> dict[str, Any]:
    """Live version of the integrated-main research candidate.

    It rebuilds the exact integrated feature family from currently available
    closed candles, then fits only on rows before the target 15m market.
    """
    raw15 = base.load_raw('ETH', '15m').sort_values('dt').copy()
    raw15 = merge_recent_closed_binance(raw15, '15m')
    raw15 = raw15.sort_values('dt').copy()
    last = raw15.iloc[-1].copy()
    next_dt = pd.Timestamp(last['dt']) + pd.Timedelta(minutes=15)
    future = last.copy()
    future['dt'] = next_dt
    future['date'] = next_dt
    raw15_live = pd.concat([raw15, pd.DataFrame([future])], ignore_index=True)
    df, feats15 = base.build_features(raw15_live, '15m')
    feature_cols = list(feats15)

    raw1h = merge_recent_closed_binance(base.load_raw('ETH', '1h').sort_values('dt').copy(), '1h')
    raw4h = merge_recent_closed_binance(base.load_raw('ETH', '4h').sort_values('dt').copy(), '4h')
    for tf, raw_tf in [('1h', raw1h), ('4h', raw4h)]:
        fdf, fcols = base.build_features(raw_tf, tf)
        df, added = integrated.prefix_merge_asof(df, fdf, fcols, tf)
        feature_cols += added

    ddf, dcols = daily_feature_frame_from_1h(raw1h)
    df, added = integrated.prefix_merge_asof(df, ddf, dcols, 'daily')
    feature_cols += added

    clean = []
    forbidden = set(base.FORBIDDEN_FEATURES)
    forbidden_prefix = tuple(base.FORBIDDEN_PREFIXES)
    for c in feature_cols:
        if c in forbidden or c.startswith(forbidden_prefix):
            continue
        s = pd.to_numeric(df[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
        if int(s.notna().sum()) >= max(500, int(len(df) * 0.55)):
            df[c] = s
            clean.append(c)
    df = df.dropna(subset=['dt']).sort_values('dt').reset_index(drop=True)
    row = df[df['dt'] == next_dt]
    if row.empty:
        raise RuntimeError('failed to build integrated live feature row')
    feats = integrated.feature_subset(clean, params['feature_mode'])
    train_days = {'1y': 365, '2y': 730, '3y': 1095, '4y': 1460, '5y': 1825, '7y': 2555}.get(params.get('train_window'))
    if train_days is None and params.get('train_window') != 'full':
        train_days = 1095
    train = df[(df['dt'] < next_dt) & df.get('label_up').notna()].copy()
    if train_days is not None:
        train = train[train['dt'] >= next_dt - pd.Timedelta(days=train_days)].copy()
    max_train_rows = int(getattr(integrated, 'MAX_TRAIN_ROWS', 180000) or 0)
    if max_train_rows > 0 and len(train) > max_train_rows:
        train = train.sort_values('dt').iloc[-max_train_rows:].copy()
    cache_key = model_cache_key(params, feats, train)
    model = None
    cache_status = 'miss'
    try:
        cached = load_model_cache(cache_key)
        if cached is not None:
            model = cached
            cache_status = 'hit'
    except Exception:
        cache_status = 'load_failed'
    if model is None:
        model = integrated.fit_model(params['engine'], train, feats, params)
        if model is not None:
            try:
                save_model_cache(cache_key, model)
            except Exception:
                cache_status = 'save_failed'
    if model is None:
        raise RuntimeError('integrated top159 live model failed to fit')
    prob_up = float(integrated.predict(model, row, feats)[0])
    side = 'UP' if prob_up >= 0.5 else 'DOWN'
    model_score = max(prob_up, 1.0 - prob_up)
    return {
        'candidate_start': next_dt.to_pydatetime().replace(tzinfo=timezone.utc).isoformat(),
        'prob_up': prob_up,
        'side': side,
        'model_score': model_score,
        'params': params,
        'feature_count': len(feats),
        'train_rows': int(len(train)),
        'raw_last_dt': pd.Timestamp(last['dt']).isoformat(),
        'model_cache': cache_status,
        'live_model_profile': params.get('live_model_profile'),
    }


def model_cache_key(params: dict[str, Any], feats: list[str], train: pd.DataFrame) -> str:
    train_dt = pd.to_datetime(train['dt'], utc=True, errors='coerce')
    payload = {
        'params': params,
        'features': list(feats),
        'train_rows': int(len(train)),
        'train_start': str(train_dt.min()),
        'train_end': str(train_dt.max()),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode('utf-8')).hexdigest()


def load_model_cache(cache_key: str) -> Any | None:
    if not MODEL_CACHE.exists():
        return None
    with MODEL_CACHE.open('rb') as fh:
        payload = pickle.load(fh)
    if isinstance(payload, dict) and payload.get('cache_key') == cache_key:
        return payload.get('model')
    return None


def save_model_cache(cache_key: str, model: Any) -> None:
    MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = MODEL_CACHE.with_suffix('.tmp')
    with tmp.open('wb') as fh:
        pickle.dump({'cache_key': cache_key, 'model': model, 'saved_at': datetime.now(timezone.utc).isoformat()}, fh)
    tmp.replace(MODEL_CACHE)


def merge_recent_closed_binance_15m(raw: pd.DataFrame) -> pd.DataFrame:
    """Merge recent fully-closed Binance ETHUSDT 15m candles in-memory.

    The legacy prediction writers used to refresh parquet files. top159 must not
    depend on those old slot1 processes, so live candidate generation fetches a
    small recent window itself. The currently forming candle is discarded.
    """
    try:
        recent = fetch_recent_binance_15m()
    except Exception:
        return raw
    if recent.empty:
        return raw
    merged = pd.concat([raw, recent], ignore_index=True)
    merged['dt'] = pd.to_datetime(merged['dt'], utc=True, errors='coerce')
    merged = merged.dropna(subset=['dt', 'open', 'high', 'low', 'close'])
    merged = merged.sort_values('dt').drop_duplicates(subset=['dt'], keep='last').reset_index(drop=True)
    return merged


def merge_recent_closed_binance(raw: pd.DataFrame, interval: str) -> pd.DataFrame:
    try:
        recent = fetch_recent_binance(interval)
    except Exception:
        return raw
    if recent.empty:
        return raw
    merged = pd.concat([raw, recent], ignore_index=True)
    merged['dt'] = pd.to_datetime(merged['dt'], utc=True, errors='coerce')
    merged = merged.dropna(subset=['dt', 'open', 'high', 'low', 'close'])
    merged = merged.sort_values('dt').drop_duplicates(subset=['dt'], keep='last').reset_index(drop=True)
    return merged


def fetch_recent_binance_15m() -> pd.DataFrame:
    return fetch_recent_binance('15m')


def fetch_recent_binance(interval: str) -> pd.DataFrame:
    now = pd.Timestamp(datetime.now(timezone.utc))
    floor_map = {'15m': '15min', '1h': '1h', '4h': '4h'}
    if interval not in floor_map:
        raise ValueError(f'unsupported interval {interval}')
    current_bar_start = now.floor(floor_map[interval])
    urls = [
        f'https://api.binance.com/api/v3/klines?symbol=ETHUSDT&interval={interval}&limit=128',
        f'https://fapi.binance.com/fapi/v1/klines?symbol=ETHUSDT&interval={interval}&limit=128',
    ]
    last_err = None
    for url in urls:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'polyfun-next-top159/0.1'})
            rows = json.loads(urllib.request.urlopen(req, timeout=8).read().decode('utf-8'))
            out = []
            for r in rows:
                dt = pd.to_datetime(int(r[0]), unit='ms', utc=True)
                if dt >= current_bar_start:
                    continue
                out.append({
                    'date': dt,
                    'dt': dt,
                    'open': float(r[1]),
                    'high': float(r[2]),
                    'low': float(r[3]),
                    'close': float(r[4]),
                    'volume': float(r[5]),
                })
            return pd.DataFrame(out)
        except Exception as exc:
            last_err = exc
            continue
    if last_err:
        raise last_err
    return pd.DataFrame()


def daily_feature_frame_from_1h(raw_1h: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    raw = raw_1h.copy()
    raw['dt'] = pd.to_datetime(raw['dt'], utc=True, errors='coerce')
    d = raw.dropna(subset=['dt']).set_index('dt').resample('1D', label='left', closed='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna().reset_index()
    d['dt'] = d['dt'] + pd.Timedelta(days=1)
    now_floor = pd.Timestamp(datetime.now(timezone.utc)).floor('1D')
    d = d[d['dt'] <= now_floor].sort_values('dt').reset_index(drop=True)
    close = d['close'].astype(float)
    high = d['high'].astype(float)
    low = d['low'].astype(float)
    ret = close.pct_change()
    feats: list[str] = []
    for n in list(range(1, 31)) + [45, 60, 90, 120, 180]:
        d[f'd_ret_{n}'] = close.pct_change(n)
        d[f'd_vol_{n}'] = ret.rolling(n).std()
        d[f'd_range_{n}'] = (high.rolling(n).max() - low.rolling(n).min()) / close
        feats += [f'd_ret_{n}', f'd_vol_{n}', f'd_range_{n}']
    for span in [3, 5, 8, 13, 21, 34, 55, 89, 144]:
        ema = close.ewm(span=span, adjust=False).mean()
        d[f'd_ema_dist_{span}'] = close / ema - 1.0
        feats.append(f'd_ema_dist_{span}')
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    d['d_rsi_14'] = (100 - 100 / (1 + gain / (loss + 1e-12))) / 100.0
    d['d_bb_pos_20'] = (close - close.rolling(20).mean()) / (2 * close.rolling(20).std() + 1e-12)
    feats += ['d_rsi_14', 'd_bb_pos_20']
    for c in feats:
        d[c] = pd.to_numeric(d[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
    return d[['dt'] + feats].dropna(subset=['dt']).sort_values('dt'), feats


def gamma_get(url: str) -> Any:
    req = urllib.request.Request(url, headers={'User-Agent': 'polyfun-next-top159/0.1'})
    return json.loads(urllib.request.urlopen(req, timeout=10).read().decode('utf-8'))


def find_eth15m_market(start_iso: str, side: str) -> dict[str, Any] | None:
    start = pd.Timestamp(start_iso)
    slug = f"eth-updown-15m-{int(start.timestamp())}"
    urls = [
        f"https://gamma-api.polymarket.com/markets?{urllib.parse.urlencode({'slug': slug})}",
        f"https://gamma-api.polymarket.com/markets?{urllib.parse.urlencode({'active':'true','closed':'false','limit':'200','order':'endDate','ascending':'true'})}",
    ]
    candidates: list[dict[str, Any]] = []
    for url in urls:
        try:
            raw = gamma_get(url)
        except Exception:
            continue
        if isinstance(raw, dict):
            raw = [raw]
        if isinstance(raw, list):
            candidates.extend([x for x in raw if isinstance(x, dict)])
    best = None
    best_score = 999999999
    for m in candidates:
        q = str(m.get('question') or '')
        s = str(m.get('slug') or '')
        if 'Ethereum Up or Down' not in q and not s.startswith('eth-updown-15m-'):
            continue
        if '15m' not in s and '15:' not in q and 'Up or Down' not in q:
            continue
        end = pd.to_datetime(m.get('endDate'), utc=True, errors='coerce')
        if pd.isna(end):
            continue
        # ETH 15m market ends 15 minutes after start.
        score = abs((end - (start + pd.Timedelta(minutes=15))).total_seconds())
        if score < best_score:
            best, best_score = m, score
    # The slug encodes the intended market start. If the exact-slug lookup fails
    # and we fall back to the active-market list, never accept an adjacent 15m
    # market. A two-minute tolerance covers minor API timestamp drift while
    # rejecting the previous/next market, which would be 15 minutes away.
    if best is None or best_score > 2 * 60:
        return None
    outcomes = _jsonish(best.get('outcomes'))
    token_ids = _jsonish(best.get('clobTokenIds'))
    side_title = side.title()
    try:
        idx = [str(x).title() for x in outcomes].index(side_title)
        token_id = str(token_ids[idx])
    except Exception:
        return None
    return {
        'market_slug': str(best.get('slug') or ''),
        'condition_id': str(best.get('conditionId') or best.get('condition_id') or ''),
        'token_id': token_id,
        'question': best.get('question'),
        'endDate': best.get('endDate'),
        'outcomes': outcomes,
        'best_score_seconds': best_score,
    }


def _jsonish(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            out = json.loads(value)
            return out if isinstance(out, list) else []
        except Exception:
            return []
    return []


def main() -> int:
    ap = argparse.ArgumentParser(description='Generate a fresh top159 ETH15m candidate only when a real Polymarket market can be matched')
    ap.add_argument('--out', default=str(NEXT / 'runtime' / 'eth15m_live_candidate.jsonl'))
    ap.add_argument('--report', default=str(REPORTS / 'top159_live_candidate_generation_latest.json'))
    ap.add_argument('--edge-override', type=float, default=None, help='One-off test override for candidate edge; formal top159 archive model remains 0.05.')
    args = ap.parse_args()
    contract_audit = audit_top159_contract(NEXT)
    if not contract_audit.get('ok'):
        report = {
            'generatedAt': datetime.now(timezone.utc).isoformat(),
            'wroteCandidate': False,
            'candidate': None,
            'candidateAllowed': False,
            'reason': 'live_contract_mismatch',
            'contractAudit': contract_audit,
        }
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + '\n', encoding='utf-8')
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        return 2
    base = _load_module('crypto_search_live', BASE_SCRIPT)
    fill = _load_module('fill_search_live', FILL_SCRIPT)
    params = top159_params()
    if args.edge_override is not None:
        params = dict(params)
        params['edge'] = float(args.edge_override)
        params['edge_override_reason'] = 'one_off_1u_acceptance_test'
    signal = current_top159_signal(base, fill, params)
    market = find_eth15m_market(signal['candidate_start'], signal['side'])
    row = None
    now = pd.Timestamp(datetime.now(timezone.utc))
    start_ts = pd.Timestamp(signal['candidate_start'])
    elapsed_seconds = (now - start_ts).total_seconds()
    in_entry_window = 30 <= elapsed_seconds <= 180
    threshold = 0.5 + params['edge']
    shock_filter = {
        'enabled': bool(load_shock_filter_profile().get('enabled')),
        'strategy_profile': (load_shock_filter_profile().get('strategy_profile') or load_shock_filter_profile().get('profile') or 'new_shock_filter_top159') if load_shock_filter_profile().get('enabled') else (params.get('live_model_profile') or 'top159'),
        'profile': load_shock_filter_profile().get('profile') if load_shock_filter_profile().get('enabled') else None,
        'candidate_id': load_shock_filter_profile().get('candidate_id') if load_shock_filter_profile().get('enabled') else None,
        'shock_candidate_id': load_shock_filter_profile().get('candidate_id') if load_shock_filter_profile().get('enabled') else None,
        'shock_action': 'not_evaluated',
        'shock_reason': 'model_or_market_not_ready',
    }
    calibration_router = router_not_evaluated_result('model_or_market_not_ready')
    reject_reason = None
    if not market:
        reject_reason = 'real_eth15m_market_not_matched'
    elif signal['model_score'] < threshold:
        reject_reason = 'model_score_below_edge'
    elif not in_entry_window:
        reject_reason = 'outside_entry_window'
    candidate_has_signal = bool(market and signal['model_score'] >= threshold)
    if candidate_has_signal:
        shock_filter = evaluate_shock_filter(base, signal, params)
        if shock_filter.get('enabled') and shock_filter.get('shock_action') == 'block':
            reject_reason = 'shock_filter_blocked'
    shock_allows = not (shock_filter.get('enabled') and shock_filter.get('shock_action') == 'block')
    if candidate_has_signal:
        try:
            shock_mod_for_router = _load_module('shock_filter_live_router_features', SHOCK_SCRIPT)
            router_live_row = enrich_live_shock_features(shock_mod_for_router, base, signal)
            calibration_router = evaluate_calibration_router(
                root=NEXT,
                live_row=router_live_row,
                signal=signal,
                base_061_allows=bool(candidate_has_signal and shock_allows),
            )
        except Exception as exc:
            calibration_router = router_error_result(exc)
        if calibration_router.get('enabled') and not calibration_router.get('router_allows_live', True):
            reject_reason = 'calibration_router_blocked'
    router_allows = not (calibration_router.get('enabled') and not calibration_router.get('router_allows_live', True))
    should_cache_candidate = bool(candidate_has_signal and shock_allows and router_allows and elapsed_seconds <= 180)
    if should_cache_candidate:
        source = str(params.get('live_model_profile') or 'top159_live_model_5y')
        strategy_profile = shock_filter.get('strategy_profile') if shock_filter.get('enabled') else source
        row = {
            'symbol': 'ETH',
            'period': '15m',
            'market_slug': market['market_slug'],
            'condition_id': market['condition_id'],
            'token_id': market['token_id'],
            'side': signal['side'],
            'model_score': signal['model_score'],
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'source': source,
            'live_model_profile': source,
            'selected_candidate': params.get('selected_candidate'),
            'train_window': params.get('train_window'),
            'feature_mode': params.get('feature_mode'),
            'edge': params.get('edge'),
            'prob_up': signal['prob_up'],
            'candidate_start': signal['candidate_start'],
            'strategy_profile': strategy_profile,
            'base_model_profile': shock_filter.get('base_model_profile') or source,
            'base_selected_candidate': shock_filter.get('base_selected_candidate') or params.get('selected_candidate'),
            'shock_filter_enabled': bool(shock_filter.get('enabled')),
            'shock_condition': bool(shock_filter.get('shock_condition')),
            'shock_gate_probability': shock_filter.get('shock_gate_probability'),
            'shock_gate_threshold': shock_filter.get('shock_gate_threshold'),
            'shock_action': shock_filter.get('shock_action'),
            'shock_reason': shock_filter.get('shock_reason'),
            'shock_profile': shock_filter.get('profile'),
            'shock_candidate_id': shock_filter.get('candidate_id') or shock_filter.get('shock_candidate_id'),
            'shock_model_engine': shock_filter.get('model_engine'),
            'shock_model_hyper': shock_filter.get('model_hyper'),
            'shock_required_score': shock_filter.get('shock_required_score'),
            'shock_base_threshold': shock_filter.get('shock_base_threshold'),
            'cluster_action': shock_filter.get('cluster_action'),
            'cluster_min_hits': shock_filter.get('cluster_min_hits'),
            'cluster_hit_count': shock_filter.get('cluster_hit_count'),
            'cluster_hits': shock_filter.get('cluster_hits'),
            'feature_alignment_version': shock_filter.get('feature_alignment_version'),
        }
        row.update(router_candidate_fields(calibration_router))
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('a', encoding='utf-8') as f:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    reason = 'candidate_written' if (row and in_entry_window) else reject_reason
    if row and not in_entry_window:
        reason = 'candidate_cached_before_entry_window'
    report = {
        'generatedAt': datetime.now(timezone.utc).isoformat(),
        'signal': signal,
        'market': market,
        'wroteCandidate': row is not None,
        'candidate': row,
        'entryWindow': {'elapsedSeconds': elapsed_seconds, 'allowed': in_entry_window, 'start': 30, 'end': 180},
        'threshold': threshold,
        'strategy_profile': shock_filter.get('strategy_profile') if shock_filter.get('enabled') else (params.get('live_model_profile') or 'top159'),
        'shockFilter': shock_filter,
        'calibrationRouter': calibration_router,
        'candidateAllowed': bool(row and in_entry_window),
        'reason': reason,
        'contractAudit': contract_audit,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    return 0 if row else 2


if __name__ == '__main__':
    raise SystemExit(main())
