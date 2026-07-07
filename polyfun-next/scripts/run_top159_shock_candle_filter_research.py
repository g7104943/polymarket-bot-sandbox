#!/usr/bin/env python3
from __future__ import annotations

# Research-only top159 shock-candle filter.
# Does not mutate live config, does not submit orders, does not restart processes.

import hashlib
import importlib.util
import itertools
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault('OMP_NUM_THREADS', os.environ.get('TOP159_SHOCK_THREADS', '2'))
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', os.environ.get('TOP159_SHOCK_THREADS', '2'))
os.environ.setdefault('MKL_NUM_THREADS', os.environ.get('TOP159_SHOCK_THREADS', '2'))
os.environ.setdefault('OPENBLAS_NUM_THREADS', os.environ.get('TOP159_SHOCK_THREADS', '2'))

import numpy as np
import pandas as pd

ROOT = Path('/Users/mac/polyfun')
REPORTS = ROOT / 'reports'
EXTREME = ROOT / 'polyfun-next' / 'scripts' / 'run_top159_integrated_main_extreme_search.py'
PROFILE = ROOT / 'polyfun-next' / 'runtime' / 'top159_live_model_profile.json'
OUT_JSON = REPORTS / 'top159_shock_candle_filter_research_latest.json'
OUT_MD = REPORTS / 'top159_shock_candle_filter_research_latest.md'
OUT_AUDIT = REPORTS / 'top159_shock_candle_filter_bug_audit_latest.json'
OUT_CANDIDATES = REPORTS / 'top159_shock_candle_filter_candidates_latest.jsonl'
RNG_SEED = 20260503
SHOCK_FEATURE_ALIGNMENT_VERSION = 'closed_candle_available_at_v2'


def bj_now() -> str:
    return pd.Timestamp.now(tz='Asia/Shanghai').strftime('%Y-%m-%d %H:%M:%S %Z')


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
        raise RuntimeError(f'cannot import {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M = load_module('top159_integrated_for_shock_filter', EXTREME)


def profile_params() -> dict[str, Any]:
    return json.loads(PROFILE.read_text())['params']


def selected_rows_for_period(df: pd.DataFrame, features: list[str], params: dict[str, Any], start: pd.Timestamp, end: pd.Timestamp, name: str) -> pd.DataFrame:
    train = df[df['dt'] < start].copy()
    if int(os.environ.get('TOP159_SHOCK_MAX_TRAIN_ROWS', '180000')) > 0 and len(train) > int(os.environ.get('TOP159_SHOCK_MAX_TRAIN_ROWS', '180000')):
        train = train.sort_values('dt').iloc[-int(os.environ.get('TOP159_SHOCK_MAX_TRAIN_ROWS', '180000')):].copy()
    val = df[(df['dt'] >= start) & (df['dt'] < end)].copy()
    feats = M.feature_subset(features, params['feature_mode'])[: int(params.get('max_features') or len(features))]
    model = M.fit_model(params['engine'], train, feats, params)
    if model is None:
        raise RuntimeError(f'integrated model fit failed for {name}')
    prob = M.predict(model, val, feats)
    selected = M.select_rows(val, prob, params)
    selected['period_name'] = name
    selected['direction'] = np.where(selected['pred_up15'].astype(bool), 'UP', 'DOWN')
    selected['score15'] = pd.to_numeric(selected['score15'], errors='coerce')
    selected['won'] = selected['won'].astype(bool)
    return selected[['dt', 'label_up', 'pred_up15', 'direction', 'score15', 'won', 'period_name']].copy()


def build_selected_sets() -> tuple[pd.DataFrame, dict[str, Any]]:
    params = profile_params()
    df, features, truth = M.build_integrated_frame()
    end = df['dt'].max()
    val_starts = {'180d': end - pd.Timedelta(days=180), '365d': end - pd.Timedelta(days=365)}
    rows = []
    # validation sets: current live profile, trained only before each validation window.
    for w, start in val_starts.items():
        rows.append(selected_rows_for_period(df, features, params, start, end + pd.Timedelta(seconds=1), f'validation_{w}'))
    # gate training sets: generated fully before validation start; used only by model/filter training.
    for w, val_start in val_starts.items():
        train_start = val_start - pd.Timedelta(days=365)
        rows.append(selected_rows_for_period(df, features, params, train_start, val_start, f'gate_train_for_{w}'))
    all_sel = pd.concat(rows, ignore_index=True).drop_duplicates(['dt', 'period_name']).sort_values('dt').reset_index(drop=True)
    return all_sel, {'integratedDataTruth': truth, 'profileParams': params, 'featureCount': len(features), 'end': str(end)}


def enrich_shock_features(sel: pd.DataFrame) -> pd.DataFrame:
    raw15 = M.load_raw('ETH', '15m')
    raw1h = M.load_raw('ETH', '1h')
    raw4h = M.load_raw('ETH', '4h')
    feat15 = candle_features(raw15, '15m')
    feat1h = candle_features(raw1h, '1h')
    feat4h = candle_features(raw4h, '4h')
    out = sel.sort_values('dt').copy()
    out['ts_ns'] = pd.to_datetime(out['dt'], utc=True).map(lambda x: pd.Timestamp(x).value).astype('int64')
    base = out[['ts_ns']].sort_values('ts_ns')
    for tf, feats in [('15m', feat15), ('1h', feat1h), ('4h', feat4h)]:
        use = feats.sort_values('ts_ns').drop(columns=['dt'])
        m = pd.merge_asof(base, use, on='ts_ns', direction='backward', allow_exact_matches=True).sort_index()
        for c in use.columns:
            if c != 'ts_ns':
                out[c] = m[c].to_numpy()
    out = out.drop(columns=['ts_ns'])
    for tf in ['15m', '1h', '4h']:
        cdir = out[f'{tf}_candle_dir']
        out[f'{tf}_same_as_top159'] = ((out['direction'] == 'UP') & (cdir == 'up')) | ((out['direction'] == 'DOWN') & (cdir == 'down'))
        out[f'{tf}_opposes_top159'] = ((out['direction'] == 'UP') & (cdir == 'down')) | ((out['direction'] == 'DOWN') & (cdir == 'up'))
        out[f'{tf}_terminal_chase'] = ((out['direction'] == 'UP') & (out[f'{tf}_pos20'] >= 0.75)) | ((out['direction'] == 'DOWN') & (out[f'{tf}_pos20'] <= 0.25))
        out[f'{tf}_exhaustion_wick'] = ((out['direction'] == 'UP') & (out[f'{tf}_upper_wick_ratio'] >= 0.35)) | ((out['direction'] == 'DOWN') & (out[f'{tf}_lower_wick_ratio'] >= 0.35))
    out['shock_base_any'] = False
    return out


def candle_features(raw: pd.DataFrame, prefix: str) -> pd.DataFrame:
    d = raw[['dt', 'open', 'high', 'low', 'close', 'volume']].copy().sort_values('dt').reset_index(drop=True)
    close = d['close'].astype(float); open_ = d['open'].astype(float); high = d['high'].astype(float); low = d['low'].astype(float); vol = d['volume'].astype(float)
    rng = (high - low).replace(0, np.nan)
    body = (close - open_).abs()
    d[f'{prefix}_body_ratio'] = (body / rng).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d[f'{prefix}_range_pct'] = (rng / close).replace([np.inf, -np.inf], np.nan)
    d[f'{prefix}_range_q'] = d[f'{prefix}_range_pct'].rolling(1000, min_periods=100).rank(pct=True)
    d[f'{prefix}_volume_mult'] = vol / (vol.rolling(50, min_periods=10).median() + 1e-12)
    d[f'{prefix}_close_pos'] = ((close - low) / (rng + 1e-12)).clip(0, 1)
    d[f'{prefix}_upper_wick_ratio'] = ((high - np.maximum(close, open_)) / (rng + 1e-12)).clip(0, 1)
    d[f'{prefix}_lower_wick_ratio'] = ((np.minimum(close, open_) - low) / (rng + 1e-12)).clip(0, 1)
    hi20 = high.rolling(20, min_periods=5).max(); lo20 = low.rolling(20, min_periods=5).min()
    d[f'{prefix}_pos20'] = ((close - lo20) / (hi20 - lo20 + 1e-12)).clip(0, 1)
    ema20 = close.ewm(span=20, adjust=False).mean(); ema50 = close.ewm(span=50, adjust=False).mean(); slope = ema20.pct_change(4)
    trend = np.full(len(d), 'mixed', dtype=object)
    trend[(close > ema20) & (ema20 > ema50) & (slope > 0)] = 'up'
    trend[(close < ema20) & (ema20 < ema50) & (slope < 0)] = 'down'
    d[f'{prefix}_trend_state'] = trend
    d[f'{prefix}_candle_dir'] = np.where(close >= open_, 'up', 'down')
    d[f'{prefix}_is_big_body_default'] = (d[f'{prefix}_body_ratio'] >= 0.55) & (d[f'{prefix}_range_q'] >= 0.70)
    interval = {'15m': pd.Timedelta(minutes=15), '1h': pd.Timedelta(hours=1), '4h': pd.Timedelta(hours=4)}.get(prefix)
    if interval is None:
        raise ValueError(f'unsupported candle feature prefix {prefix}')
    d[f'{prefix}_open_time'] = pd.to_datetime(d['dt'], utc=True, errors='coerce')
    d[f'{prefix}_available_at'] = d[f'{prefix}_open_time'] + interval
    keep = ['dt', f'{prefix}_open_time', f'{prefix}_available_at'] + [c for c in d.columns if c.startswith(prefix + '_') and c not in {f'{prefix}_open_time', f'{prefix}_available_at'}]
    out = d[keep].copy()
    # Kline rows are timestamped by open time. Directly matching on open time
    # leaks unfinished 4h candles into 15m decisions. Match by the time the
    # candle is fully visible instead.
    out['ts_ns'] = pd.to_datetime(out[f'{prefix}_available_at'], utc=True).map(lambda x: pd.Timestamp(x).value).astype('int64')
    return out


def condition_mask(df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    body = float(params.get('body_min', 0.55))
    rq = float(params.get('range_q_min', 0.70))
    vol = float(params.get('volume_mult_min', 0.0))
    shock1h = (df['1h_body_ratio'] >= body) & (df['1h_range_q'] >= rq) & (df['1h_volume_mult'].fillna(0) >= vol)
    shock4h = (df['4h_body_ratio'] >= body) & (df['4h_range_q'] >= rq) & (df['4h_volume_mult'].fillna(0) >= vol)
    conds: list[pd.Series] = []
    mode = params['rule_mode']
    if mode in ('1h_all', 'any_all'):
        conds.append(shock1h)
    if mode in ('4h_all', 'any_all'):
        conds.append(shock4h)
    if mode in ('1h_terminal', 'any_terminal'):
        conds.append(shock1h & df['1h_terminal_chase'])
    if mode in ('4h_terminal', 'any_terminal'):
        conds.append(shock4h & df['4h_terminal_chase'])
    if mode in ('1h_exhaustion', 'any_exhaustion'):
        conds.append(shock1h & df['1h_exhaustion_wick'])
    if mode in ('4h_exhaustion', 'any_exhaustion'):
        conds.append(shock4h & df['4h_exhaustion_wick'])
    if mode == '4h_no_confirm':
        conds.append(shock4h & ~(df['1h_same_as_top159'] | df['15m_same_as_top159']))
    if mode == '1h_push_terminal':
        conds.append(shock1h & df['1h_same_as_top159'] & df['1h_terminal_chase'])
    if mode == '4h_push_terminal':
        conds.append(shock4h & df['4h_same_as_top159'] & df['4h_terminal_chase'])
    if mode == 'both_1h_4h':
        conds.append(shock1h & shock4h)
    if not conds:
        return pd.Series(False, index=df.index)
    out = conds[0].copy()
    for c in conds[1:]:
        out |= c
    return out.fillna(False)


def metric_rows(base_df: pd.DataFrame, keep_mask: pd.Series, name: str, params: dict[str, Any], condition: pd.Series | None = None) -> dict[str, Any]:
    kept = base_df[keep_mask].copy()
    blocked = base_df[~keep_mask].copy()
    def wr(x): return round(float(x['won'].mean() * 100.0), 6) if len(x) else 0.0
    cond = condition if condition is not None else pd.Series(False, index=base_df.index)
    cond_kept = base_df[keep_mask & cond]
    cond_base = base_df[cond]
    wins = int(kept['won'].sum()); losses = int(len(kept) - wins)
    blocked_winners = int(blocked['won'].sum()); blocked_losers = int(len(blocked) - blocked_winners)
    return {
        'name': name,
        'params': params,
        'trades': int(len(kept)),
        'baseTrades': int(len(base_df)),
        'retentionPct': round(100.0 * len(kept) / max(1, len(base_df)), 6),
        'wins': wins,
        'losses': losses,
        'winRatePct': wr(kept),
        'baseWinRatePct': wr(base_df),
        'blockedTrades': int(len(blocked)),
        'blockedWinners': blocked_winners,
        'blockedLosers': blocked_losers,
        'blockedLoserMinusWinner': blocked_losers - blocked_winners,
        'conditionBaseTrades': int(len(cond_base)),
        'conditionBaseWinRatePct': wr(cond_base),
        'conditionKeptTrades': int(len(cond_kept)),
        'conditionKeptWinRatePct': wr(cond_kept),
        'setHash': stable_hash([name, params, kept['dt'].astype(str).tolist()[:5000], int(len(kept)), wins, losses]),
    }


def evaluate_rule_candidate(df: pd.DataFrame, params: dict[str, Any], window: str) -> dict[str, Any]:
    cond = condition_mask(df, params)
    if params['action'] == 'hard_block':
        keep = ~cond
    elif params['action'] == 'raise_score':
        keep = (~cond) | (df['score15'] >= float(params['shock_score_min']))
    else:
        raise ValueError(params['action'])
    row = metric_rows(df, keep, candidate_name(params), params, cond)
    row['window'] = window
    return row


def candidate_name(params: dict[str, Any]) -> str:
    if params.get('family') == 'model_gate':
        return f"shock_model_gate_{params.get('model_engine')}_{params['rule_mode']}_body{params['body_min']}_rq{params['range_q_min']}_vol{params['volume_mult_min']}_minprob{params.get('min_gate_win_prob')}_{stable_hash(params)}"
    return f"shock_{params['action']}_{params['rule_mode']}_body{params['body_min']}_rq{params['range_q_min']}_vol{params['volume_mult_min']}_score{params.get('shock_score_min','-')}_{stable_hash(params)}"


def rule_param_grid() -> list[dict[str, Any]]:
    modes = ['1h_all', '4h_all', 'any_all', '1h_terminal', '4h_terminal', 'any_terminal', '1h_exhaustion', '4h_exhaustion', 'any_exhaustion', '4h_no_confirm', '1h_push_terminal', '4h_push_terminal', 'both_1h_4h']
    rows = []
    for mode, body, rq, vol in itertools.product(modes, [0.45, 0.55, 0.65], [0.60, 0.70, 0.80], [0.0, 1.1, 1.4]):
        rows.append({'family': 'rule', 'action': 'hard_block', 'rule_mode': mode, 'body_min': body, 'range_q_min': rq, 'volume_mult_min': vol})
        for ss in [0.57, 0.59, 0.61, 0.63]:
            rows.append({'family': 'rule', 'action': 'raise_score', 'rule_mode': mode, 'body_min': body, 'range_q_min': rq, 'volume_mult_min': vol, 'shock_score_min': ss})
    return rows


def model_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    base_cols = [
        'score15', '1h_body_ratio', '1h_range_q', '1h_volume_mult', '1h_close_pos', '1h_upper_wick_ratio', '1h_lower_wick_ratio', '1h_pos20',
        '4h_body_ratio', '4h_range_q', '4h_volume_mult', '4h_close_pos', '4h_upper_wick_ratio', '4h_lower_wick_ratio', '4h_pos20',
        '15m_body_ratio', '15m_range_q', '15m_volume_mult', '15m_close_pos', '15m_upper_wick_ratio', '15m_lower_wick_ratio', '15m_pos20',
        '1h_same_as_top159', '1h_opposes_top159', '1h_terminal_chase', '1h_exhaustion_wick',
        '4h_same_as_top159', '4h_opposes_top159', '4h_terminal_chase', '4h_exhaustion_wick',
        '15m_same_as_top159', '15m_opposes_top159', '15m_terminal_chase', '15m_exhaustion_wick',
    ]
    cat_cols = ['direction', '1h_trend_state', '4h_trend_state', '15m_trend_state', '1h_candle_dir', '4h_candle_dir', '15m_candle_dir']
    x = df[base_cols + cat_cols].copy()
    for c in base_cols:
        x[c] = pd.to_numeric(x[c], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x = pd.get_dummies(x, columns=cat_cols, dummy_na=True)
    return x, list(x.columns)


def fit_gate_model(train: pd.DataFrame, engine: str):
    x, cols = model_features(train)
    y = train['won'].astype(int).to_numpy()
    if len(np.unique(y)) < 2 or len(train) < 200:
        return None, cols
    if engine == 'lightgbm':
        import lightgbm as lgb
        model = lgb.LGBMClassifier(n_estimators=90, learning_rate=0.045, num_leaves=16, min_child_samples=50, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, random_state=RNG_SEED, n_jobs=2, verbose=-1)
    elif engine == 'logistic':
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        model = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(max_iter=500, C=1.0, random_state=RNG_SEED))
    else:
        raise ValueError(engine)
    model.fit(x, y)
    return model, cols


def predict_gate(model: Any, cols: list[str], val: pd.DataFrame) -> np.ndarray:
    x, _ = model_features(val)
    for c in cols:
        if c not in x.columns:
            x[c] = 0
    x = x[cols]
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def evaluate_model_candidates(enriched: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for window in ['180d', '365d']:
        train = enriched[enriched['period_name'] == f'gate_train_for_{window}'].copy()
        val = enriched[enriched['period_name'] == f'validation_{window}'].copy()
        for engine in ['logistic', 'lightgbm']:
            model, cols = fit_gate_model(train, engine)
            if model is None:
                continue
            prob = predict_gate(model, cols, val)
            model_grid = []
            for mode in ['1h_all', '1h_terminal', '1h_exhaustion', '1h_push_terminal', '4h_all', '4h_terminal', '4h_no_confirm', '4h_push_terminal', 'any_all', 'any_terminal', 'both_1h_4h']:
                for body in [0.45, 0.55, 0.65]:
                    for rq in [0.60, 0.70, 0.80]:
                        model_grid.append({'family': 'rule', 'action': 'hard_block', 'rule_mode': mode, 'body_min': body, 'range_q_min': rq, 'volume_mult_min': 0.0})
            for base_params in model_grid:
                # Model only operates inside hard shock/risk conditions. This keeps the model search fast and avoids generic overfit gates.
                cond = condition_mask(val, base_params)
                if int(cond.sum()) < 50:
                    continue
                for min_prob in [0.46, 0.48, 0.50, 0.52, 0.54]:
                    keep = (~cond) | (prob >= min_prob)
                    params = {**base_params, 'family': 'model_gate', 'model_engine': engine, 'min_gate_win_prob': min_prob}
                    row = metric_rows(val, keep, candidate_name(params), params, cond)
                    row['window'] = window
                    out.append(row)
    return out


def baseline_rows(enriched: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for window in ['180d', '365d']:
        val = enriched[enriched['period_name'] == f'validation_{window}'].copy()
        params = {'family': 'baseline'}
        keep = pd.Series(True, index=val.index)
        row = metric_rows(val, keep, 'current_archive_integrated_top159', params)
        row['window'] = window
        rows.append(row)
    return rows


def evaluate_rules(enriched: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    params_list = rule_param_grid()
    for window in ['180d', '365d']:
        val = enriched[enriched['period_name'] == f'validation_{window}'].copy()
        for p in params_list:
            out.append(evaluate_rule_candidate(val, p, window))
    return out


def candidate_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault(r['name'], []).append(r)
    out = []
    base_by_window = {r['window']: r for r in rows if r['name'] == 'current_archive_integrated_top159'}
    for name, rs in grouped.items():
        by = {r['window']: r for r in rs}
        if '180d' not in by or '365d' not in by:
            continue
        passed_reasons = []
        if name == 'current_archive_integrated_top159':
            passed_reasons.append('baseline_not_a_filter_candidate')
        for w, min_trades in [('180d', 100), ('365d', 200)]:
            r = by[w]
            if r['trades'] < min_trades:
                passed_reasons.append(f'{w}_too_few_trades')
            if w == '365d' and r['retentionPct'] < 45:
                passed_reasons.append('365d_retention_under_45pct')
            if r['blockedLosers'] <= r['blockedWinners'] and name != 'current_archive_integrated_top159':
                passed_reasons.append(f'{w}_blocked_losers_not_gt_winners')
        # score: prioritize fixing shock-specific win rate and blocking more losers than winners while preserving trades.
        score = 0.0
        for w in ['180d', '365d']:
            r = by[w]
            base = base_by_window.get(w, {})
            score += (r['blockedLoserMinusWinner'] * 20)
            score += (r['winRatePct'] - base.get('winRatePct', r['winRatePct'])) * 50
            score += (r['conditionKeptWinRatePct'] - r['conditionBaseWinRatePct']) * 20
            score += min(r['retentionPct'], 100) * 0.2
        out.append({'name': name, 'params': by['365d'].get('params'), 'rows': [by['180d'], by['365d']], 'passed': not passed_reasons, 'reasons': passed_reasons, 'score': round(score, 6)})
    out.sort(key=lambda x: (x['passed'], x['score']), reverse=True)
    return out


def bug_audit(enriched: pd.DataFrame) -> dict[str, Any]:
    audit = {
        'generatedAt': now_iso(),
        'beijingTime': bj_now(),
        'scope': 'research only; no live config change; no order submission',
        'periodCounts': enriched['period_name'].value_counts().to_dict(),
        'futureLeakageControls': [
            '1h/4h/15m shock features are matched by candle available_at=open_time+interval, so only fully closed candles are visible.',
            'final market direction is used only as label won, never as shock feature.',
            'gate_train_for_180d and gate_train_for_365d end before their validation windows start.',
        ],
        'requiredColumnsPresent': {},
    }
    required = ['1h_body_ratio', '4h_body_ratio', '1h_range_q', '4h_range_q', '1h_terminal_chase', '4h_terminal_chase', 'won', 'score15']
    audit['requiredColumnsPresent'] = {c: bool(c in enriched.columns and enriched[c].notna().sum() > 0) for c in required}
    # Random label smoke audit for gate model.
    rnd = enriched[enriched['period_name'] == 'gate_train_for_365d'].copy()
    random_ok = True
    random_wr = None
    try:
        if len(rnd) >= 200:
            shuffled = rnd.copy()
            rng = np.random.default_rng(RNG_SEED)
            shuffled['won'] = rng.permutation(shuffled['won'].to_numpy())
            model, cols = fit_gate_model(shuffled, 'logistic')
            val = enriched[enriched['period_name'] == 'validation_365d'].copy()
            if model is not None:
                prob = predict_gate(model, cols, val)
                pred = prob >= 0.5
                # This is not strategy win rate; it checks the randomized classifier has no magical label signal.
                random_wr = float((pred == val['won'].to_numpy()).mean() * 100.0)
                random_ok = 42.0 <= random_wr <= 58.0
    except Exception as exc:
        random_ok = False
        audit['randomLabelError'] = str(exc)[:300]
    audit['randomLabelGateAccuracyPct'] = round(random_wr, 6) if random_wr is not None else None
    audit['randomLabelPassed'] = random_ok
    audit['passed'] = all(audit['requiredColumnsPresent'].values()) and random_ok
    return audit


def render_md(payload: dict[str, Any]) -> str:
    lines = ['# top159 冲击K线过滤研究', '', f'- 北京时间：`{payload["beijingTime"]}`', f'- live动作：`{payload["liveAction"]}`', f'- 审计通过：`{payload["bugAudit"]["passed"]}`', '']
    lines += ['## 核心对比', '', '|候选|窗口|交易数|胜/负|胜率|拦截赢家|拦截输单|保留率|冲击原胜率|冲击保留后胜率|哈希|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for c in payload['leaderboard'][:20]:
        for r in c['rows']:
            lines.append(f"|{c['name']}|{r['window']}|{r['trades']}|{r['wins']}/{r['losses']}|{r['winRatePct']}|{r['blockedWinners']}|{r['blockedLosers']}|{r['retentionPct']}|{r['conditionBaseWinRatePct']}|{r['conditionKeptWinRatePct']}|`{r['setHash']}`|")
    lines += ['', '## 唯一结论', '']
    sel = payload.get('selected')
    if sel and sel.get('passed'):
        lines.append(f"- 找到过门候选：`{sel['name']}`。")
        lines.append('- 该候选只进入研究结论，不自动改 live。')
    elif sel:
        lines.append(f"- 暂无严格过门候选；当前最强观察候选是 `{sel['name']}`，失败原因：`{', '.join(sel.get('reasons', []))}`。")
        lines.append('- 如果要继续，应扩大冲击定义或转向专门重训主模型，不建议直接上线。')
    else:
        lines.append('- 没有候选。')
    lines += ['', '## 文件', f"- JSON：`{OUT_JSON}`", f"- 候选明细：`{OUT_CANDIDATES}`", f"- 审计：`{OUT_AUDIT}`"]
    return '\n'.join(lines) + '\n'


def main() -> int:
    print(f'[shock-filter] start {bj_now()}', flush=True)
    selected, truth = build_selected_sets()
    enriched = enrich_shock_features(selected)
    audit = bug_audit(enriched)
    write_json(OUT_AUDIT, audit)
    print(f'[shock-filter] selectedRows={len(enriched)} auditPassed={audit["passed"]}', flush=True)
    base = baseline_rows(enriched)
    rule_rows = evaluate_rules(enriched)
    model_rows = evaluate_model_candidates(enriched)
    all_rows = base + rule_rows + model_rows
    OUT_CANDIDATES.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CANDIDATES.open('w', encoding='utf-8') as fh:
        for r in all_rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True, default=str) + '\n')
    leaderboard = candidate_groups(all_rows)
    selected_candidate = next((c for c in leaderboard if c['passed']), leaderboard[0] if leaderboard else None)
    payload = {
        'generatedAt': now_iso(),
        'beijingTime': bj_now(),
        'liveAction': 'research_only_no_live_change',
        'dataTruth': truth,
        'bugAudit': audit,
        'baselineRows': base,
        'candidateCount': len(leaderboard),
        'strictPassCount': sum(1 for c in leaderboard if c['passed']),
        'selected': selected_candidate,
        'leaderboard': leaderboard[:200],
    }
    write_json(OUT_JSON, payload)
    OUT_MD.write_text(render_md(payload), encoding='utf-8')
    print(json.dumps({'ok': True, 'strictPass': payload['strictPassCount'], 'selected': selected_candidate['name'] if selected_candidate else None, 'report': str(OUT_MD), 'json': str(OUT_JSON)}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
