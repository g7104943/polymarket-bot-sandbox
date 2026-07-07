#!/usr/bin/env python3
from __future__ import annotations

# Research-only extreme hyperopt for top159 shock candle filter.
# Does not mutate live config, does not submit orders, does not restart processes.

import hashlib
import importlib.util
import json
import os
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault('OMP_NUM_THREADS', os.environ.get('TOP159_SHOCK_EXTREME_THREADS', '2'))
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', os.environ.get('TOP159_SHOCK_EXTREME_THREADS', '2'))
os.environ.setdefault('MKL_NUM_THREADS', os.environ.get('TOP159_SHOCK_EXTREME_THREADS', '2'))
os.environ.setdefault('OPENBLAS_NUM_THREADS', os.environ.get('TOP159_SHOCK_EXTREME_THREADS', '2'))

import numpy as np
import pandas as pd

ROOT = Path('/Users/mac/polyfun')
REPORTS = ROOT / 'reports'
SHOCK_SCRIPT = ROOT / 'polyfun-next' / 'scripts' / 'run_top159_shock_candle_filter_research.py'
PNL_SCRIPT = ROOT / 'polyfun-next' / 'scripts' / 'run_top159_shock_filter_pnl_compare.py'
BASE_SCRIPT = ROOT / 'scripts' / 'ops' / 'run_crypto15m_1h_multasset_pressure_search_latest.py'
ARCHIVE_SCRIPT = ROOT / 'scripts' / 'ops' / 'run_top159_all_archived_real_eth_compare.py'
CACHE = REPORTS / 'top159_shock_filter_extreme_cache.pkl'
OUT_AUDIT = REPORTS / 'top159_shock_filter_extreme_bug_audit_latest.json'
OUT_SEARCH = REPORTS / 'top159_shock_filter_extreme_search_latest.json'
OUT_COMPARE = REPORTS / 'top159_shock_filter_extreme_compare_latest.json'
OUT_VERDICT = REPORTS / 'top159_shock_filter_extreme_unique_verdict_latest.md'
OUT_CANDIDATES = REPORTS / 'top159_shock_filter_extreme_candidates_latest.jsonl'

RNG_SEED = 20260503
START_BANKROLL = 850.0
DEFAULT_CURRENT_MODEL_HYPER = {'engine': 'lightgbm', 'n_estimators': 90, 'learning_rate': 0.045, 'num_leaves': 16, 'min_child_samples': 50, 'reg_lambda': 1.0}

CURRENT_SHOCK_PARAMS = {
    'family': 'model_gate',
    'action': 'hard_block',
    'rule_mode': 'any_all',
    'body_min': 0.45,
    'range_q_min': 0.60,
    'volume_mult_min': 0.0,
    'model_engine': 'lightgbm',
    'min_gate_win_prob': 0.54,
}

RULE_MODES = [
    '1h_all', '4h_all', 'any_all',
    '1h_terminal', '4h_terminal', 'any_terminal',
    '1h_exhaustion', '4h_exhaustion', 'any_exhaustion',
    '4h_no_confirm', '1h_push_terminal', '4h_push_terminal', 'both_1h_4h',
]
BODY_COARSE = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
RQ_COARSE = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
VOL_COARSE = [0.0, 1.2, 1.5, 2.0]
RAISE_SCORE = [0.55, 0.57, 0.59, 0.61, 0.63]
MODEL_MIN_PROBS = [round(x, 3) for x in np.arange(0.50, 0.6201, 0.01)]


def bj_now() -> str:
    return pd.Timestamp.now(tz='Asia/Shanghai').strftime('%Y-%m-%d %H:%M:%S CST')


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()[:16]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


shock = load_module('shock_filter_extreme_base', SHOCK_SCRIPT)
base = load_module('shock_filter_extreme_pressure', BASE_SCRIPT)
archive = load_module('shock_filter_extreme_archive', ARCHIVE_SCRIPT)
base.TRIALS = int(os.environ.get('TOP159_SHOCK_EXTREME_MC_TRIALS', '1000'))
EXPECTED_SHOCK_FEATURE_ALIGNMENT_VERSION = getattr(shock, 'SHOCK_FEATURE_ALIGNMENT_VERSION', 'closed_candle_available_at_v2')


def load_or_build_enriched() -> tuple[pd.DataFrame, dict[str, Any]]:
    refresh = os.environ.get('TOP159_SHOCK_EXTREME_REFRESH_CACHE') == '1'
    if CACHE.exists() and not refresh:
        with CACHE.open('rb') as fh:
            payload = pickle.load(fh)
        if payload.get('feature_alignment_version') == EXPECTED_SHOCK_FEATURE_ALIGNMENT_VERSION:
            return payload['enriched'], payload['truth']
    selected, truth = shock.build_selected_sets()
    enriched = shock.enrich_shock_features(selected)
    truth = {
        **truth,
        'shockFeatureAlignmentVersion': EXPECTED_SHOCK_FEATURE_ALIGNMENT_VERSION,
        'shockFeatureAlignment': 'candle_available_at=open_time+interval; merge_asof backward; only closed candles visible',
    }
    with CACHE.open('wb') as fh:
        pickle.dump({'enriched': enriched, 'truth': truth, 'feature_alignment_version': EXPECTED_SHOCK_FEATURE_ALIGNMENT_VERSION}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return enriched, truth


@dataclass(frozen=True)
class CandidateKey:
    name: str
    params_hash: str


def is_current_shock_params(params: dict[str, Any]) -> bool:
    core_ok = params.get('family') == 'model_gate' and all(params.get(k) == CURRENT_SHOCK_PARAMS.get(k) for k in ['action','rule_mode','body_min','range_q_min','volume_mult_min','model_engine','min_gate_win_prob'])
    if not core_ok:
        return False
    return params.get('model_hyper', DEFAULT_CURRENT_MODEL_HYPER) == DEFAULT_CURRENT_MODEL_HYPER


def candidate_name(params: dict[str, Any]) -> str:
    if params.get('family') == 'baseline':
        return 'current_new_archive_top159'
    if is_current_shock_params(params):
        return 'current_shock_filter_gate'
    if params.get('family') == 'model_gate':
        mh = params.get('model_hyper', {})
        return f"extreme_model_{params.get('model_engine')}_{params['rule_mode']}_b{params['body_min']}_rq{params['range_q_min']}_v{params['volume_mult_min']}_p{params['min_gate_win_prob']}_{stable_hash({'p': params, 'h': mh})}"
    return f"extreme_rule_{params['action']}_{params['rule_mode']}_b{params['body_min']}_rq{params['range_q_min']}_v{params['volume_mult_min']}_s{params.get('shock_score_min','-')}_{stable_hash(params)}"


def condition_grid() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for mode in RULE_MODES:
        for body in BODY_COARSE:
            for rq in RQ_COARSE:
                for vol in VOL_COARSE:
                    out.append({'rule_mode': mode, 'body_min': body, 'range_q_min': rq, 'volume_mult_min': vol})
    return out


def rule_grid() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for c in condition_grid():
        out.append({'family': 'rule', 'action': 'hard_block', **c})
        for ss in RAISE_SCORE:
            out.append({'family': 'rule', 'action': 'raise_score', 'shock_score_min': ss, **c})
    return out


def model_hyper_grid() -> list[dict[str, Any]]:
    return [
        {'engine': 'logistic', 'C': 0.35},
        {'engine': 'logistic', 'C': 0.75},
        {'engine': 'logistic', 'C': 1.5},
        {'engine': 'lightgbm', 'n_estimators': 70, 'learning_rate': 0.035, 'num_leaves': 12, 'min_child_samples': 35, 'reg_lambda': 0.7},
        {'engine': 'lightgbm', 'n_estimators': 90, 'learning_rate': 0.045, 'num_leaves': 16, 'min_child_samples': 50, 'reg_lambda': 1.0},
        {'engine': 'lightgbm', 'n_estimators': 120, 'learning_rate': 0.035, 'num_leaves': 20, 'min_child_samples': 70, 'reg_lambda': 1.5},
        {'engine': 'lightgbm', 'n_estimators': 150, 'learning_rate': 0.03, 'num_leaves': 24, 'min_child_samples': 90, 'reg_lambda': 2.0},
    ]


def fit_gate_model_ext(train: pd.DataFrame, hyper: dict[str, Any]):
    x, cols = shock.model_features(train)
    y = train['won'].astype(int).to_numpy()
    if len(np.unique(y)) < 2 or len(train) < 200:
        return None, cols
    if hyper['engine'] == 'logistic':
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        model = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=700, C=float(hyper['C']), random_state=RNG_SEED))
    elif hyper['engine'] == 'lightgbm':
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=int(hyper['n_estimators']),
            learning_rate=float(hyper['learning_rate']),
            num_leaves=int(hyper['num_leaves']),
            min_child_samples=int(hyper['min_child_samples']),
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=float(hyper['reg_lambda']),
            random_state=RNG_SEED,
            n_jobs=2,
            verbose=-1,
        )
    else:
        raise ValueError(hyper['engine'])
    model.fit(x, y)
    return model, cols


def fast_metric(val: pd.DataFrame, keep: np.ndarray, cond: np.ndarray, name: str, params: dict[str, Any], window: str) -> dict[str, Any]:
    won = val['won'].astype(bool).to_numpy()
    keep = keep.astype(bool)
    cond = cond.astype(bool)
    trades = int(keep.sum())
    wins = int((won & keep).sum())
    losses = int(trades - wins)
    blocked = ~keep
    blocked_winners = int((won & blocked).sum())
    blocked_losers = int((~won & blocked).sum())
    cond_count = int(cond.sum())
    cond_kept = keep & cond
    return {
        'name': name,
        'window': window,
        'params': params,
        'trades': trades,
        'baseTrades': int(len(val)),
        'wins': wins,
        'losses': losses,
        'winRatePct': round(100.0 * wins / trades, 6) if trades else 0.0,
        'baseWinRatePct': round(100.0 * float(won.mean()), 6) if len(won) else 0.0,
        'retentionPct': round(100.0 * trades / max(1, len(val)), 6),
        'blockedTrades': int(blocked.sum()),
        'blockedWinners': blocked_winners,
        'blockedLosers': blocked_losers,
        'blockedLoserMinusWinner': blocked_losers - blocked_winners,
        'conditionBaseTrades': cond_count,
        'conditionBaseWinRatePct': round(100.0 * float(won[cond].mean()), 6) if cond_count else 0.0,
        'conditionKeptTrades': int(cond_kept.sum()),
        'conditionKeptWinRatePct': round(100.0 * float(won[cond_kept].mean()), 6) if int(cond_kept.sum()) else 0.0,
    }


def evaluate_baseline(enriched: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for window in ['180d', '365d']:
        val = enriched[enriched['period_name'] == f'validation_{window}'].copy()
        keep = np.ones(len(val), dtype=bool)
        cond = np.zeros(len(val), dtype=bool)
        out.append(fast_metric(val, keep, cond, 'current_new_archive_top159', {'family': 'baseline'}, window))
    return out


def evaluate_one_params(enriched: pd.DataFrame, params: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for window in ['180d', '365d']:
        val = enriched[enriched['period_name'] == f'validation_{window}'].copy()
        if params.get('family') == 'model_gate':
            train = enriched[enriched['period_name'] == f'gate_train_for_{window}'].copy()
            model, cols = fit_gate_model_ext(train, params.get('model_hyper', DEFAULT_CURRENT_MODEL_HYPER))
            if model is None:
                continue
            prob = shock.predict_gate(model, cols, val)
            cond_params = {k: params[k] for k in ['rule_mode', 'body_min', 'range_q_min', 'volume_mult_min']}
            cond = shock.condition_mask(val, cond_params).to_numpy(dtype=bool)
            keep = (~cond) | (prob >= float(params['min_gate_win_prob']))
        else:
            cond = shock.condition_mask(val, params).to_numpy(dtype=bool)
            if params['action'] == 'hard_block':
                keep = ~cond
            else:
                keep = (~cond) | (val['score15'].to_numpy(dtype=float) >= float(params['shock_score_min']))
        out.append(fast_metric(val, keep, cond, candidate_name(params), params, window))
    return out


def group_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault(r['name'], []).append(r)
    base = {r['window']: r for r in rows if r['name'] == 'current_new_archive_top159'}
    out: list[dict[str, Any]] = []
    for name, rs in grouped.items():
        by = {r['window']: r for r in rs}
        if '180d' not in by or '365d' not in by:
            continue
        reasons: list[str] = []
        if name == 'current_new_archive_top159':
            reasons.append('baseline')
        for w, min_trades in [('180d', 100), ('365d', 200)]:
            r = by[w]
            if r['trades'] < min_trades:
                reasons.append(f'{w}_too_few_trades')
            if w == '365d' and r['retentionPct'] < 45:
                reasons.append('365d_retention_under_45pct')
            if name != 'current_new_archive_top159' and r['blockedLosers'] <= r['blockedWinners']:
                reasons.append(f'{w}_blocked_losers_not_gt_winners')
        score = 0.0
        for w in ['180d', '365d']:
            r = by[w]
            b = base.get(w, r)
            score += r['blockedLoserMinusWinner'] * 25.0
            score += (r['winRatePct'] - b['winRatePct']) * 70.0
            score += (r['conditionKeptWinRatePct'] - r['conditionBaseWinRatePct']) * 25.0
            score += min(r['retentionPct'], 100) * 0.25
        out.append({'name': name, 'params': by['365d']['params'], 'rows': [by['180d'], by['365d']], 'passedFast': (not reasons or reasons == ['baseline']), 'reasons': reasons, 'score': round(score, 6)})
    out.sort(key=lambda x: (x['passedFast'], x['score']), reverse=True)
    return out


def first_stage_search(enriched: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    rows.extend(evaluate_baseline(enriched))
    total = 0
    t0 = time.time()
    # Current shock first, to keep as fixed compare row.
    rows.extend(evaluate_one_params(enriched, CURRENT_SHOCK_PARAMS))
    params_seen = {stable_hash(CURRENT_SHOCK_PARAMS)}
    for p in rule_grid():
        h = stable_hash(p)
        if h in params_seen:
            continue
        params_seen.add(h)
        rows.extend(evaluate_one_params(enriched, p))
        total += 1
        if total % 2000 == 0:
            print(f'[extreme] rule searched={total} elapsed={time.time()-t0:.1f}s', flush=True)
    # Model gate coarse search. Train each model once per window, evaluate all conditions/prob thresholds.
    conds = condition_grid()
    for hyper in model_hyper_grid():
        print(f'[extreme] model coarse {hyper}', flush=True)
        for window in ['180d', '365d']:
            train = enriched[enriched['period_name'] == f'gate_train_for_{window}'].copy()
            val = enriched[enriched['period_name'] == f'validation_{window}'].copy()
            model, cols = fit_gate_model_ext(train, hyper)
            if model is None:
                continue
            prob = shock.predict_gate(model, cols, val)
            for c in conds:
                cond = shock.condition_mask(val, c).to_numpy(dtype=bool)
                if int(cond.sum()) < 35:
                    continue
                for mp in MODEL_MIN_PROBS:
                    p = {'family': 'model_gate', 'action': 'hard_block', **c, 'model_engine': hyper['engine'], 'model_hyper': hyper, 'min_gate_win_prob': mp}
                    keep = (~cond) | (prob >= mp)
                    rows.append(fast_metric(val, keep, cond, candidate_name(p), p, window))
    leaderboard = group_rows(rows)
    return rows, leaderboard


def refine_params_from_top(top: list[dict[str, Any]], limit: int = 80) -> list[dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for cand in top[:limit]:
        p = cand['params']
        if p.get('family') == 'baseline':
            continue
        bodies = sorted(set(round(x, 3) for x in [p.get('body_min', 0.45)-0.05, p.get('body_min', 0.45)-0.025, p.get('body_min', 0.45), p.get('body_min', 0.45)+0.025, p.get('body_min', 0.45)+0.05] if 0.30 <= x <= 0.75))
        rqs = sorted(set(round(x, 3) for x in [p.get('range_q_min', 0.60)-0.05, p.get('range_q_min', 0.60)-0.025, p.get('range_q_min', 0.60), p.get('range_q_min', 0.60)+0.025, p.get('range_q_min', 0.60)+0.05] if 0.45 <= x <= 0.95))
        vols = sorted(set([p.get('volume_mult_min', 0.0), 0.0, 1.1, 1.2, 1.4, 1.5, 2.0]))
        modes = [p.get('rule_mode', 'any_all')]
        if p.get('rule_mode') not in ['any_all', 'both_1h_4h']:
            modes.append('any_all')
        for mode in modes:
            for b in bodies:
                for rq in rqs:
                    for vol in vols:
                        if p.get('family') == 'model_gate':
                            for mp in sorted(set(round(x, 3) for x in [p['min_gate_win_prob']-0.015, p['min_gate_win_prob']-0.01, p['min_gate_win_prob']-0.005, p['min_gate_win_prob'], p['min_gate_win_prob']+0.005, p['min_gate_win_prob']+0.01, p['min_gate_win_prob']+0.015] if 0.48 <= x <= 0.65)):
                                np_ = {**p, 'rule_mode': mode, 'body_min': b, 'range_q_min': rq, 'volume_mult_min': vol, 'min_gate_win_prob': mp}
                                out[stable_hash(np_)] = np_
                        else:
                            np_ = {**p, 'rule_mode': mode, 'body_min': b, 'range_q_min': rq, 'volume_mult_min': vol}
                            out[stable_hash(np_)] = np_
                            if p.get('action') == 'raise_score':
                                for ss in sorted(set(round(x, 3) for x in [p.get('shock_score_min',0.59)-0.01, p.get('shock_score_min',0.59), p.get('shock_score_min',0.59)+0.01] if 0.53 <= x <= 0.66)):
                                    np2 = {**np_, 'shock_score_min': ss}
                                    out[stable_hash(np2)] = np2
    return list(out.values())



def evaluate_param_list(enriched: pd.DataFrame, params_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rule_params = [p for p in params_list if p.get('family') != 'model_gate']
    model_params = [p for p in params_list if p.get('family') == 'model_gate']
    for i, p in enumerate(rule_params, 1):
        rows.extend(evaluate_one_params(enriched, p))
        if i % 500 == 0:
            print(f'[extreme] refine rules={i}/{len(rule_params)}', flush=True)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for p in model_params:
        grouped.setdefault(stable_hash(p.get('model_hyper', DEFAULT_CURRENT_MODEL_HYPER)), []).append(p)
    done = 0
    for _hk, ps in grouped.items():
        hyper = ps[0].get('model_hyper', DEFAULT_CURRENT_MODEL_HYPER)
        print(f'[extreme] refine model group size={len(ps)} hyper={hyper}', flush=True)
        for window in ['180d', '365d']:
            train = enriched[enriched['period_name'] == f'gate_train_for_{window}'].copy()
            val = enriched[enriched['period_name'] == f'validation_{window}'].copy()
            model, cols = fit_gate_model_ext(train, hyper)
            if model is None:
                continue
            prob = shock.predict_gate(model, cols, val)
            for p in ps:
                c = {k: p[k] for k in ['rule_mode', 'body_min', 'range_q_min', 'volume_mult_min']}
                cond = shock.condition_mask(val, c).to_numpy(dtype=bool)
                if int(cond.sum()) < 20:
                    continue
                keep = (~cond) | (prob >= float(p['min_gate_win_prob']))
                rows.append(fast_metric(val, keep, cond, candidate_name(p), p, window))
        done += len(ps)
        print(f'[extreme] refine models done={done}/{len(model_params)}', flush=True)
    return rows

def refine_search(enriched: pd.DataFrame, leaderboard: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    params = refine_params_from_top(leaderboard, limit=int(os.environ.get('TOP159_SHOCK_EXTREME_REFINE_TOP', '60')))
    max_params = int(os.environ.get('TOP159_SHOCK_EXTREME_REFINE_MAX', '16000'))
    if len(params) > max_params:
        # Keep deterministic order by parameter hash so repeated runs are identical.
        params = sorted(params, key=lambda x: stable_hash(x))[:max_params]
    print(f'[extreme] refine params={len(params)}', flush=True)
    rows = evaluate_param_list(enriched, params)
    grouped = group_rows(evaluate_baseline(enriched) + rows + evaluate_one_params(enriched, CURRENT_SHOCK_PARAMS))
    return rows, grouped


def keep_mask_for_params(enriched: pd.DataFrame, params: dict[str, Any], window: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray | None]:
    val = enriched[enriched['period_name'] == f'validation_{window}'].copy().sort_values('dt').reset_index(drop=True)
    prob = None
    if params.get('family') == 'baseline':
        cond = np.zeros(len(val), dtype=bool)
        keep = np.ones(len(val), dtype=bool)
    elif params.get('family') == 'model_gate':
        train = enriched[enriched['period_name'] == f'gate_train_for_{window}'].copy()
        model, cols = fit_gate_model_ext(train, params.get('model_hyper', DEFAULT_CURRENT_MODEL_HYPER))
        if model is None:
            raise RuntimeError('model fit failed for final params')
        prob = shock.predict_gate(model, cols, val)
        c = {k: params[k] for k in ['rule_mode', 'body_min', 'range_q_min', 'volume_mult_min']}
        cond = shock.condition_mask(val, c).to_numpy(dtype=bool)
        keep = (~cond) | (prob >= float(params['min_gate_win_prob']))
    else:
        cond = shock.condition_mask(val, params).to_numpy(dtype=bool)
        if params['action'] == 'hard_block':
            keep = ~cond
        else:
            keep = (~cond) | (val['score15'].to_numpy(dtype=float) >= float(params['shock_score_min']))
    return val, keep.astype(bool), cond.astype(bool), prob


def summarize_compound(df: pd.DataFrame, name: str, window: str, method: str, entry_price: float) -> dict[str, Any]:
    d = df.sort_values('dt').reset_index(drop=True)
    won = d['won'].astype(bool).to_numpy()
    returns = base.returns_from_wins(won, entry_price)
    fill = np.ones(len(won), dtype=float)
    row = base.summarize_curve(pd.to_datetime(d['dt'], utc=True), won, returns, fill, method, 'ETH', '15m', window, name, {'entryPrice': entry_price})
    return {
        'name': name,
        'window': window,
        'method': method,
        'trades': int(len(d)),
        'wins': int(won.sum()),
        'losses': int((~won).sum()),
        'winRatePct': row['signalWinRatePct'],
        'endingBankroll': row['endingBankroll'],
        'compoundPnl': row['compoundPnl'],
        'maxDrawdownUsd': row['maxDrawdownUsd'],
        'maxDrawdownPct': row['maxDrawdownPct'],
        'returnDrawdownRatio': row['returnDrawdownRatio'],
        'monthlyPositiveRatio': row['monthlyPositiveRatio'],
        'drawdownPeakTime': row['drawdownPeakTime'],
        'drawdownTroughTime': row['drawdownTroughTime'],
        'setHash': row['setHash'],
    }


def summarize_toxic_mc(df: pd.DataFrame, name: str, window: str) -> dict[str, Any]:
    d = df.sort_values('dt').reset_index(drop=True)
    won = d['won'].astype(bool).to_numpy()
    returns = base.returns_from_wins(won, 0.50)
    mc = base.monte_carlo_slot1(pd.to_datetime(d['dt'], utc=True), won, returns, RNG_SEED + (180 if window == '180d' else 365) + len(d) + stable_hash(name).__hash__() % 10000)
    return {
        'name': name,
        'window': window,
        'method': 'slot1_toxic_monte_carlo',
        'trades': int(len(d)),
        'wins': int(won.sum()),
        'losses': int((~won).sum()),
        'winRatePct': round(100.0 * float(won.mean()), 6) if len(won) else 0.0,
        **mc,
        'winnerFillRatePct': round(base.SLOT1_WIN_FILL * 100, 6),
        'loserFillRatePct': round(base.SLOT1_LOSS_FILL * 100, 6),
    }


def final_validation_rows(enriched: pd.DataFrame, named_params: list[tuple[str, dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    val_keep_map: dict[str, pd.DataFrame] = {}
    for label, params in named_params:
        for window in ['180d', '365d']:
            val, keep, cond, prob = keep_mask_for_params(enriched, params, window)
            selected = val[keep].copy().sort_values('dt').reset_index(drop=True)
            selected['extreme_keep'] = True
            selected['shock_condition'] = cond[keep] if len(selected) else []
            if prob is not None:
                selected['shock_prob'] = prob[keep]
            val_full = val.copy()
            val_full[f'{label}_keep'] = keep
            val_full[f'{label}_condition'] = cond
            if prob is not None:
                val_full[f'{label}_prob'] = prob
            val_keep_map[f'{label}_{window}'] = val_full
            rows.append(summarize_compound(selected, label, window, 'full_fill_buy_0.50', 0.50))
            rows.append(summarize_compound(selected, label, window, 'fak_pressure_buy_0.52', 0.52))
            rows.append(summarize_toxic_mc(selected, label, window))
    return rows, val_keep_map


def max_drawdown_from_pnl(pnls: list[float]) -> float:
    if not pnls:
        return 0.0
    curve = np.concatenate([[0.0], np.cumsum(np.asarray(pnls, dtype=float))])
    peak = np.maximum.accumulate(curve)
    return round(float(np.max(peak - curve)), 6)


def archived_real_rows(enriched: pd.DataFrame, named_params: list[tuple[str, dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fill_rows, scan_audit = archive.load_live_fill_rows()
    all_eth = archive.aggregate_logical(fill_rows, '全部归档ETH15m live去重')
    strict_mask = fill_rows['sourcePath'].str.contains('slot01|forcedtrade|base_forcedtrade|__live_eth|logs_v5_exp10_bp0480__live_eth', case=False, regex=True)
    strict = archive.aggregate_logical(fill_rows[strict_mask].copy(), '严格旧slot1/ETH相关live去重') if strict_mask.any() else all_eth.iloc[0:0]
    rows: list[dict[str, Any]] = []
    # Archive history only overlaps recent 180d validation window.
    for label, params in named_params:
        val, keep, cond, prob = keep_mask_for_params(enriched, params, '180d')
        pred = val.assign(keep=keep, cond=cond)
        for scope, old in [('严格旧slot1/ETH相关live去重', strict), ('全部归档ETH15m live去重', all_eth)]:
            if old.empty:
                continue
            merged = old.merge(pred[['dt','pred_up15','score15','won','keep','cond']], left_on='marketStart', right_on='dt', how='left')
            s = merged[merged['keep'].fillna(False).astype(bool)].copy().sort_values('marketStart')
            if s.empty:
                rows.append({'scope': scope, 'name': label, 'oldRealMarkets': int(len(old)), 'selectedTrades': 0})
                continue
            model_won = s['pred_up15'].astype(bool).to_numpy() == s['actualUp'].astype(bool).to_numpy()
            one = np.where(model_won, 1.0, -1.0)
            fak52 = np.where(model_won, (1.0 / 0.52) - 1.0, -1.0)
            s['sameDirectionAsOld'] = s['pred_up15'].astype(bool).to_numpy() == (s['direction'].astype(str).str.upper() == 'UP').to_numpy()
            same = s[s['sameDirectionAsOld']].copy()
            same_pnl = same['pnl'].astype(float).tolist()
            skipped = old[~old['marketStart'].isin(s['marketStart'])]
            rows.append({
                'scope': scope,
                'name': label,
                'oldRealMarkets': int(len(old)),
                'selectedTrades': int(len(s)),
                'skippedTrades': int(len(old) - len(s)),
                'skippedOldWinners': int(skipped['won'].sum()) if len(skipped) else 0,
                'skippedOldLosers': int((~skipped['won']).sum()) if len(skipped) else 0,
                'wins': int(model_won.sum()),
                'losses': int(len(model_won) - int(model_won.sum())),
                'winRatePct': round(100.0 * float(model_won.mean()), 6),
                'oneUnitPnl': round(float(one.sum()), 6),
                'oneUnitMaxDrawdown': max_drawdown_from_pnl(one.tolist()),
                'fak52Pnl': round(float(fak52.sum()), 6),
                'fak52MaxDrawdown': max_drawdown_from_pnl(fak52.tolist()),
                'sameDirectionExecutableTrades': int(len(same)),
                'sameDirectionActualPnlUsd': round(float(sum(same_pnl)), 6) if same_pnl else 0.0,
                'sameDirectionActualMaxDrawdownUsd': max_drawdown_from_pnl(same_pnl),
                'shockConditionTrades': int(s['cond'].fillna(False).astype(bool).sum()),
                'avgScore': round(float(s['score15'].mean()), 6),
                'setHash': stable_hash(s[['marketSlug','pred_up15','actualUp']].to_dict('records')),
            })
    audit = {
        'scanAudit': scan_audit,
        'strictTrades': int(len(strict)),
        'allArchivedTrades': int(len(all_eth)),
        'archivedRangeUtc': [str(all_eth['marketStart'].min()), str(all_eth['marketStart'].max())] if len(all_eth) else [None, None],
        'truthLimit': 'new/shock/extreme models were not actually executed in old archived windows; sameDirectionActualPnlUsd only reuses old fills when model direction equals old executed direction',
    }
    return rows, audit


def bug_audit(enriched: pd.DataFrame) -> dict[str, Any]:
    audit = shock.bug_audit(enriched)
    # Additional index/repeat checks for this extreme search.
    period_dups = enriched.groupby('period_name')['dt'].apply(lambda s: int(s.duplicated().sum())).to_dict()
    audit['periodDuplicateDtCounts'] = period_dups
    audit['interceptIndexIntegrity'] = 'keep/block masks are built directly on each validation dataframe, not merged by external index'
    # Re-evaluate current shock twice and compare hashes.
    try:
        a = evaluate_one_params(enriched, CURRENT_SHOCK_PARAMS)
        b = evaluate_one_params(enriched, CURRENT_SHOCK_PARAMS)
        audit['repeatCurrentShockEqual'] = stable_hash(a) == stable_hash(b)
    except Exception as exc:
        audit['repeatCurrentShockEqual'] = False
        audit['repeatCurrentShockError'] = str(exc)[:300]
    audit['passed'] = bool(audit.get('passed')) and all(v == 0 for v in period_dups.values()) and bool(audit.get('repeatCurrentShockEqual'))
    return audit


def render_verdict(payload: dict[str, Any]) -> str:
    sel = payload['selectedExtreme']
    current = payload['currentShock']
    lines = [
        '# top159 冲击过滤极限超参结论',
        '',
        f"- 北京时间：`{payload['beijingTime']}`",
        f"- live动作：`{payload['liveAction']}`",
        f"- 审计通过：`{payload['audit']['bugAudit']['passed']}`",
        '',
        '## 唯一结论',
    ]
    if sel:
        better = payload['selectionDecision']['decision']
        lines.append(f"- 极限候选：`{sel['name']}`。")
        lines.append(f"- 决策：`{better}`。")
        lines.append(f"- 原因：{payload['selectionDecision']['reason']}")
        lines.append('- 本轮只生成研究结论，不改真实交易；若要用，下一步先接入影子监控。')
    else:
        lines.append('- 没有极限候选，保持当前冲击过滤研究候选。')
    lines += ['', '## 当前冲击过滤参数', '```json', json.dumps(current, ensure_ascii=False, indent=2, default=str), '```']
    if sel:
        lines += ['', '## 极限候选参数', '```json', json.dumps(sel['params'], ensure_ascii=False, indent=2, default=str), '```']
    lines += ['', '## 输出文件', f"- 搜索：`{OUT_SEARCH}`", f"- 对比：`{OUT_COMPARE}`", f"- 审计：`{OUT_AUDIT}`"]
    return '\n'.join(lines) + '\n'


def decide(current_metrics: dict[str, Any], extreme_metrics: dict[str, Any]) -> dict[str, Any]:
    # Compare by 365d toxic P5/P50, FAK 0.52, and drawdown. Require meaningful margin to replace current shock.
    def pick(rows, name, method, window):
        return next(r for r in rows if r['name'] == name and r['method'] == method and r['window'] == window)
    rows = extreme_metrics['validationRows']
    cur = 'current_shock_filter_gate'
    ext = 'extreme_shock_filter_best'
    c_tox = pick(rows, cur, 'slot1_toxic_monte_carlo', '365d')
    e_tox = pick(rows, ext, 'slot1_toxic_monte_carlo', '365d')
    c_fak = pick(rows, cur, 'fak_pressure_buy_0.52', '365d')
    e_fak = pick(rows, ext, 'fak_pressure_buy_0.52', '365d')
    p50_gain = e_tox['endingBankrollP50'] - c_tox['endingBankrollP50']
    p5_gain = e_tox['endingBankrollP5'] - c_tox['endingBankrollP5']
    fak_gain = e_fak['endingBankroll'] - c_fak['endingBankroll']
    dd_worse = e_fak['maxDrawdownUsd'] > c_fak['maxDrawdownUsd'] * 1.10
    if p50_gain > c_tox['endingBankrollP50'] * 0.03 and p5_gain >= 0 and fak_gain >= 0 and not dd_worse:
        return {'decision': 'extreme_candidate_stronger_enter_shadow_validation', 'reason': f'365天毒性P50提升{p50_gain:.2f}、P5不差、FAK压力不差且回撤未明显变坏。'}
    if p50_gain > 0 or p5_gain > 0:
        return {'decision': 'extreme_candidate_observation_only_keep_current_shock', 'reason': f'极限候选有局部提升但不够稳：P50差{p50_gain:.2f}，P5差{p5_gain:.2f}，FAK差{fak_gain:.2f}。'}
    return {'decision': 'keep_current_shock_filter', 'reason': f'极限候选没有稳定超过当前冲击过滤：P50差{p50_gain:.2f}，P5差{p5_gain:.2f}，FAK差{fak_gain:.2f}。'}


def main() -> int:
    start = time.time()
    print(f'[extreme] start {bj_now()}', flush=True)
    enriched, truth = load_or_build_enriched()
    audit = bug_audit(enriched)
    write_json(OUT_AUDIT, audit)
    print(f'[extreme] rows={len(enriched)} auditPassed={audit["passed"]}', flush=True)
    rows1, board1 = first_stage_search(enriched)
    print(f'[extreme] stage1 candidates={len(board1)} top={board1[0]["name"] if board1 else None}', flush=True)
    rows2, board2 = refine_search(enriched, board1)
    merged_rows = rows1 + rows2
    with OUT_CANDIDATES.open('w', encoding='utf-8') as fh:
        for r in merged_rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True, default=str) + '\n')
    board_all = group_rows(merged_rows)
    strict = [c for c in board_all if c['passedFast'] and c['name'] != 'current_new_archive_top159']
    selected = strict[0] if strict else (board_all[0] if board_all else None)
    current_shock_group = next((c for c in board_all if c['name'] == 'current_shock_filter_gate'), None)
    search_payload = {
        'generatedAt': now_iso(),
        'beijingTime': bj_now(),
        'liveAction': 'research_only_no_live_change',
        'runtimeSeconds': round(time.time() - start, 3),
        'dataTruth': truth,
        'bugAudit': audit,
        'candidateRows': len(merged_rows),
        'candidateGroups': len(board_all),
        'strictPassCount': len(strict),
        'selectedFast': selected,
        'currentShock': current_shock_group,
        'leaderboard': board_all[:300],
    }
    write_json(OUT_SEARCH, search_payload)
    # Final comparison for current archive, current shock, selected extreme.
    if selected is None:
        named = [
            ('current_new_archive_top159', {'family': 'baseline'}),
            ('current_shock_filter_gate', CURRENT_SHOCK_PARAMS),
        ]
    else:
        named = [
            ('current_new_archive_top159', {'family': 'baseline'}),
            ('current_shock_filter_gate', CURRENT_SHOCK_PARAMS),
            ('extreme_shock_filter_best', selected['params']),
        ]
    validation, _maps = final_validation_rows(enriched, named)
    archived_rows, archive_audit = archived_real_rows(enriched, named)
    compare_payload = {
        'generatedAt': now_iso(),
        'beijingTime': bj_now(),
        'liveAction': 'research_only_no_live_change',
        'selectedExtreme': selected,
        'currentShock': current_shock_group,
        'validationRows': validation,
        'archivedRealRows': archived_rows,
        'audit': {
            'bugAudit': audit,
            'archiveAudit': archive_audit,
            'bankroll': {'start': START_BANKROLL, 'stakePct': 0.01},
            'methods': ['full_fill_buy_0.50', 'fak_pressure_buy_0.52', 'slot1_toxic_monte_carlo'],
        },
    }
    compare_payload['selectionDecision'] = decide({}, compare_payload) if selected is not None else {'decision': 'no_candidate', 'reason': 'no selected candidate'}
    write_json(OUT_COMPARE, compare_payload)
    OUT_VERDICT.write_text(render_verdict(compare_payload), encoding='utf-8')
    print(json.dumps({
        'ok': True,
        'auditPassed': audit['passed'],
        'strictPassCount': len(strict),
        'selected': selected['name'] if selected else None,
        'decision': compare_payload['selectionDecision'],
        'runtimeSeconds': round(time.time() - start, 3),
        'search': str(OUT_SEARCH),
        'compare': str(OUT_COMPARE),
        'verdict': str(OUT_VERDICT),
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
