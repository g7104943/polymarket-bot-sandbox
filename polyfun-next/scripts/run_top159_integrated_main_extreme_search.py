#!/usr/bin/env python3
from __future__ import annotations

# Research-only extreme integrated top159 main-model retraining.
# Does not read/write live config. Does not start trading.

import concurrent.futures as cf
import hashlib
import importlib.util
import itertools
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

THREADS = os.environ.get('TOP159_MAIN_EXTREME_WORKER_THREADS', '2')
for k in ['OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ.setdefault(k, THREADS)

import numpy as np
import pandas as pd

ROOT = Path('/Users/mac/polyfun')
REPORTS = ROOT / 'reports'
RAW = ROOT / 'data' / 'raw'
BASE_SCRIPT = ROOT / 'scripts' / 'ops' / 'run_crypto15m_1h_multasset_pressure_search_latest.py'
FILL_SCRIPT = ROOT / 'scripts' / 'ops' / 'run_newslot1_fill_rate_toxicity_search_latest.py'
AUX_SCRIPT = ROOT / 'polyfun-next' / 'scripts' / 'run_top159_multiday_aux_research.py'

WORKERS = int(os.environ.get('TOP159_MAIN_EXTREME_WORKERS', '4'))
MAX_SECONDS = int(os.environ.get('TOP159_MAIN_EXTREME_MAX_SECONDS', str(24 * 3600)))
CHECKPOINT_SECONDS = int(os.environ.get('TOP159_MAIN_EXTREME_CHECKPOINT_SECONDS', '900'))
MC_TRIALS = int(os.environ.get('TOP159_MAIN_EXTREME_MC_TRIALS', '800'))
MAX_TRAIN_ROWS = int(os.environ.get('TOP159_MAIN_EXTREME_MAX_TRAIN_ROWS', '180000'))
ENGINES = [x.strip() for x in os.environ.get('TOP159_MAIN_EXTREME_ENGINES', 'lightgbm,xgboost,logistic').split(',') if x.strip()]
PARAM_LIMIT = int(os.environ.get('TOP159_MAIN_EXTREME_PARAM_LIMIT', '0'))
RNG_SEED = 20260502

OUT_RESULTS = REPORTS / 'top159_integrated_main_extreme_results_latest.jsonl'
OUT_CHECKPOINT = REPORTS / 'top159_integrated_main_extreme_checkpoint_latest.json'
OUT_LEADERBOARD_JSON = REPORTS / 'top159_integrated_main_extreme_leaderboard_latest.json'
OUT_LEADERBOARD_MD = REPORTS / 'top159_integrated_main_extreme_leaderboard_latest.md'
OUT_AUDIT_JSON = REPORTS / 'top159_integrated_main_extreme_bug_audit_latest.json'
OUT_VERDICT_JSON = REPORTS / 'top159_integrated_main_extreme_unique_verdict_latest.json'
OUT_VERDICT_MD = REPORTS / 'top159_integrated_main_extreme_unique_verdict_latest.md'


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot load {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

base = load_module('crypto_pressure_for_integrated', BASE_SCRIPT)
fill = load_module('fill_search_for_integrated', FILL_SCRIPT)
aux = load_module('aux_research_for_integrated', AUX_SCRIPT)
base.TRIALS = MC_TRIALS
aux.MC_TRIALS = MC_TRIALS


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def bj_now() -> str:
    return pd.Timestamp.now(tz='Asia/Shanghai').strftime('%Y-%m-%d %H:%M:%S %Z')


def stable_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def load_raw(asset: str, tf: str) -> pd.DataFrame:
    return base.load_raw(asset, tf)


def prefix_merge_asof(left: pd.DataFrame, right: pd.DataFrame, cols: list[str], prefix: str) -> tuple[pd.DataFrame, list[str]]:
    r = right[['dt'] + cols].copy()
    rename = {c: f'{prefix}_{c}' for c in cols}
    r = r.rename(columns=rename)
    l = left.sort_values('dt').copy()
    r = r.sort_values('dt').copy()
    l['dt'] = pd.to_datetime(l['dt'], utc=True).astype('datetime64[ns, UTC]')
    r['dt'] = pd.to_datetime(r['dt'], utc=True).astype('datetime64[ns, UTC]')
    out = pd.merge_asof(l, r, on='dt', direction='backward')
    return out, list(rename.values())


def daily_feature_frame() -> tuple[pd.DataFrame, list[str]]:
    d = aux.daily_closed_bars().copy()
    close = d['close'].astype(float)
    high = d['high'].astype(float)
    low = d['low'].astype(float)
    vol = d['volume'].astype(float)
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


def build_integrated_frame() -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    raw15 = load_raw('ETH', '15m')
    df, feats15 = base.build_features(raw15, '15m')
    feature_cols = list(feats15)
    data_truth = {
        'raw15Rows': int(len(raw15)),
        'base15Rows': int(len(df)),
        'featureGroups': {'15m': len(feats15)},
        'top159ScoreFeature': 'not_used_in_v1_to_avoid_leakage; old top159 base features are included instead',
    }
    for tf in ['1h', '4h']:
        raw = load_raw('ETH', tf)
        fdf, fcols = base.build_features(raw, tf)
        df, added = prefix_merge_asof(df, fdf, fcols, tf)
        feature_cols += added
        data_truth['featureGroups'][tf] = len(added)
        data_truth[f'raw{tf}Rows'] = int(len(raw))
    ddf, dcols = daily_feature_frame()
    df, added = prefix_merge_asof(df, ddf, dcols, 'daily')
    feature_cols += added
    data_truth['featureGroups']['daily_1d_30d'] = len(added)
    # Clean feature list. Remove features with too little coverage or future-looking names.
    clean: list[str] = []
    forbidden = set(base.FORBIDDEN_FEATURES)
    forbidden_prefix = tuple(base.FORBIDDEN_PREFIXES)
    for c in feature_cols:
        if c in forbidden or c.startswith(forbidden_prefix):
            continue
        s = pd.to_numeric(df[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
        if int(s.notna().sum()) >= max(500, int(len(df) * 0.55)):
            df[c] = s
            clean.append(c)
    df = df.dropna(subset=['dt', 'label_up']).sort_values('dt').reset_index(drop=True)
    data_truth['finalRows'] = int(len(df))
    data_truth['featureCountBeforeModes'] = int(len(clean))
    return df, clean, data_truth


def feature_subset(features: list[str], mode: str) -> list[str]:
    if mode == 'base15':
        out = [c for c in features if not c.startswith(('1h_', '4h_', 'daily_'))]
    elif mode == 'base_plus_1h4h':
        out = [c for c in features if not c.startswith('daily_')]
    elif mode == 'base_plus_daily':
        out = [c for c in features if not c.startswith(('1h_', '4h_'))]
    elif mode == 'base_plus_4h_daily':
        out = [c for c in features if not c.startswith('1h_')]
    elif mode == 'base_plus_1h_daily':
        out = [c for c in features if not c.startswith('4h_')]
    elif mode == 'htf_only':
        out = [c for c in features if c.startswith(('1h_', '4h_', 'daily_'))]
    elif mode == 'daily_only':
        out = [c for c in features if c.startswith('daily_')]
    elif mode == 'trend_multi':
        keys = ('ret_', 'ema_', 'ema_dist_', 'rsi', 'bb_pos', 'range_', 'vol_')
        out = [c for c in features if any(k in c for k in keys)]
    elif mode == 'daily_heavy':
        out = [c for c in features if c.startswith('daily_') or c.startswith(('ret_', 'ema_', 'ema_dist_', 'vol_', 'range_', 'rsi_', 'bb_'))]
    else:
        out = list(features)
    # Candidate-specific caps are applied inside evaluate_params.
    return out


def split_train_val(df: pd.DataFrame, window: str, train_window: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    end = df['dt'].max()
    days = 180 if window == '180d' else 365
    start = end - pd.Timedelta(days=days)
    val = df[df['dt'] >= start].copy()
    if train_window == 'full':
        train = df[df['dt'] < start].copy()
    else:
        train_days = {'1y': 365, '2y': 730, '3y': 1095, '4y': 1460, '5y': 1825, '7y': 2555}[train_window]
        train = df[(df['dt'] < start) & (df['dt'] >= start - pd.Timedelta(days=train_days))].copy()
    if MAX_TRAIN_ROWS > 0 and len(train) > MAX_TRAIN_ROWS:
        train = train.sort_values('dt').iloc[-MAX_TRAIN_ROWS:].copy()
    return train, val


def fit_model(engine: str, train: pd.DataFrame, feats: list[str], params: dict[str, Any], random_labels: bool = False):
    x = train[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = train['label_up'].astype(int).to_numpy()
    if random_labels:
        rng = np.random.default_rng(RNG_SEED + len(train) + len(feats))
        y = rng.permutation(y)
    if len(train) < 1000 or len(np.unique(y)) < 2:
        return None
    if engine == 'lightgbm':
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=int(params['n_estimators']), learning_rate=float(params['learning_rate']),
            num_leaves=int(params['num_leaves']), min_child_samples=int(params['min_child_samples']),
            subsample=float(params['subsample']), colsample_bytree=float(params['colsample_bytree']),
            reg_lambda=float(params['reg_lambda']), random_state=RNG_SEED, n_jobs=int(THREADS), verbose=-1,
        )
    elif engine == 'xgboost':
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=int(params['n_estimators']), max_depth=int(params['depth']), learning_rate=float(params['learning_rate']),
            subsample=float(params['subsample']), colsample_bytree=float(params['colsample_bytree']), reg_lambda=float(params['reg_lambda']),
            random_state=RNG_SEED, n_jobs=int(THREADS), eval_metric='logloss', verbosity=0,
        )
    elif engine == 'catboost':
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            iterations=int(params['n_estimators']), depth=int(params['depth']), learning_rate=float(params['learning_rate']),
            l2_leaf_reg=float(params['reg_lambda']), loss_function='Logloss', eval_metric='Logloss',
            random_seed=RNG_SEED, verbose=False, thread_count=int(THREADS),
        )
    elif engine == 'logistic':
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, C=float(params.get('C', 1.0)), random_state=RNG_SEED))
    else:
        raise ValueError(engine)
    model.fit(x, y)
    return model


def predict(model: Any, val: pd.DataFrame, feats: list[str]) -> np.ndarray:
    x = val[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def select_rows(val: pd.DataFrame, prob: np.ndarray, params: dict[str, Any]) -> pd.DataFrame:
    pred_up = prob >= 0.5
    score = np.maximum(prob, 1.0 - prob)
    mask = score >= 0.5 + float(params['edge'])
    if params.get('score_band') == 'mid_only':
        mask &= score <= float(params.get('max_score', 0.72))
    if params.get('vol_q', 0.999) < 0.999 and 'vol_16' in val.columns:
        v = pd.to_numeric(val['vol_16'], errors='coerce')
        mask &= v <= float(v.quantile(float(params['vol_q'])))
    out = val.loc[mask, ['dt', 'label_up']].copy().reset_index(drop=True)
    pred_sel = pred_up[mask]
    score_sel = score[mask]
    out['pred_up15'] = pred_sel.astype(bool)
    out['score15'] = score_sel
    out['won'] = out['pred_up15'].to_numpy() == out['label_up'].astype(bool).to_numpy()
    return out


def evaluate_params(df: pd.DataFrame, features: list[str], params: dict[str, Any], name: str) -> dict[str, Any] | None:
    feats = feature_subset(features, params['feature_mode'])
    max_features = int(params.get('max_features', os.environ.get('TOP159_MAIN_EXTREME_MAX_FEATURES', '120')))
    if max_features > 0:
        feats = feats[:max_features]
    if len(feats) < 10:
        return None
    rows = []
    for i, window in enumerate(['180d', '365d']):
        train, val = split_train_val(df, window, params['train_window'])
        model = fit_model(params['engine'], train, feats, params)
        if model is None:
            return None
        prob = predict(model, val, feats)
        selected = select_rows(val, prob, params)
        if len(selected) == 0:
            return None
        row = aux.curve_metrics(selected, name, window, 'integrated_main_model')
        row.update({
            'featureCount': len(feats),
            'engine': params['engine'],
            'trainWindow': params['train_window'],
            'featureMode': params['feature_mode'],
            'edge': params['edge'],
            'baseTrades': None,
            'retentionPct': None,
        })
        rows.append(row)
    return {'name': name, 'params': params, 'rows': rows, 'featureCount': len(feats)}


def base_rows() -> dict[str, dict[str, Any]]:
    params = aux.top159_params()
    out = {}
    for window in ['180d', '365d']:
        c = aux.top159_candidates(window, params)
        row = aux.curve_metrics(c, 'current_top159', window, 'baseline')
        row.update({'baseTrades': len(c), 'retentionPct': 100.0, 'blockedTrades': 0, 'blockedWinners': 0, 'blockedLosers': 0})
        out[window] = row
    return out


def pass_gate(candidate: dict[str, Any], base_by_window: dict[str, dict[str, Any]]) -> tuple[bool, list[str]]:
    by = {r['window']: r for r in candidate['rows']}
    reasons = []
    for w in ['180d', '365d']:
        r = by[w]; b = base_by_window[w]
        if r['trades'] < max(100 if w == '180d' else 200, int(b['trades'] * 0.45)):
            reasons.append(f'{w}_trades_too_low')
        if r['fak2EndingBankroll'] <= 850:
            reasons.append(f'{w}_fak2_not_profitable')
        if r['endingBankrollP50'] < b['endingBankrollP50'] * 0.995 and r['endingBankrollP5'] <= b['endingBankrollP5']:
            reasons.append(f'{w}_toxicity_not_better')
        if r['maxDrawdownP50'] > b['maxDrawdownP50'] * 1.10:
            reasons.append(f'{w}_drawdown_worse')
        if r['monthlyPositiveRatioP50'] < b['monthlyPositiveRatioP50']:
            reasons.append(f'{w}_monthly_positive_worse')
    return len(reasons) == 0, reasons


def score_candidate(candidate: dict[str, Any], base_by_window: dict[str, dict[str, Any]]) -> float:
    by = {r['window']: r for r in candidate['rows']}
    gain_p50 = sum(by[w]['endingBankrollP50'] - base_by_window[w]['endingBankrollP50'] for w in ['180d', '365d'])
    gain_p5 = sum(by[w]['endingBankrollP5'] - base_by_window[w]['endingBankrollP5'] for w in ['180d', '365d'])
    fak = sum(by[w]['fak2EndingBankroll'] for w in ['180d', '365d']) / 1000.0
    dd = by['365d']['maxDrawdownP50']
    mp = by['180d']['monthlyPositiveRatioP50'] + by['365d']['monthlyPositiveRatioP50']
    return gain_p50 * 2.0 + gain_p5 + fak + mp * 200.0 - dd * 0.5


def param_grid() -> list[dict[str, Any]]:
    engines = [e for e in ENGINES if e in {'lightgbm', 'xgboost', 'catboost', 'logistic'}]
    rows = []
    for engine in engines:
        train_windows = ['1y', '2y', '3y', '4y', '5y', '7y', 'full']
        feature_modes = [
            'base15', 'base_plus_1h4h', 'base_plus_daily', 'base_plus_1h_daily',
            'base_plus_4h_daily', 'trend_multi', 'daily_heavy', 'htf_only',
            'daily_only', 'wide',
        ]
        edges = [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.09, 0.10]
        vol_qs = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.999]
        max_features_grid = [60, 80, 120, 180, 240]
        score_bands = [('all', 1.0), ('mid_only', 0.68), ('mid_only', 0.72), ('mid_only', 0.76)]
        for tw, fm, edge, vol_q, max_features, band in itertools.product(train_windows, feature_modes, edges, vol_qs, max_features_grid, score_bands):
            score_band, max_score = band
            if engine == 'logistic':
                for C in [0.25, 0.5, 1.0, 2.0, 4.0]:
                    rows.append({
                        'engine': engine, 'train_window': tw, 'feature_mode': fm, 'edge': edge,
                        'vol_q': vol_q, 'max_features': max_features, 'score_band': score_band,
                        'max_score': max_score, 'n_estimators': 1, 'learning_rate': 0.04,
                        'num_leaves': 16, 'min_child_samples': 60, 'subsample': 0.9,
                        'colsample_bytree': 0.9, 'reg_lambda': 1.0, 'depth': 3, 'C': C,
                    })
                continue
            for ne, lr, leaves, mcs, reg, depth in [
                (80, 0.025, 16, 80, 0.5, 3),
                (100, 0.035, 24, 80, 0.5, 4),
                (160, 0.025, 32, 60, 1.0, 4),
                (220, 0.020, 40, 100, 2.0, 5),
                (120, 0.055, 20, 120, 0.8, 3),
                (260, 0.018, 48, 140, 3.0, 5),
            ]:
                rows.append({
                    'engine': engine, 'train_window': tw, 'feature_mode': fm, 'edge': edge,
                    'vol_q': vol_q, 'max_features': max_features, 'score_band': score_band,
                    'max_score': max_score, 'n_estimators': ne, 'learning_rate': lr,
                    'num_leaves': leaves, 'min_child_samples': mcs, 'subsample': 0.88,
                    'colsample_bytree': 0.88, 'reg_lambda': reg, 'depth': depth,
                })
    if PARAM_LIMIT > 0:
        rng = np.random.default_rng(RNG_SEED)
        order = rng.permutation(len(rows))
        rows = [rows[int(i)] for i in order]
        rows = rows[:PARAM_LIMIT]
    return rows

_DF: pd.DataFrame | None = None
_FEATURES: list[str] | None = None


def _init_worker(df: pd.DataFrame, features: list[str]):
    global _DF, _FEATURES
    _DF = df
    _FEATURES = features
    base.TRIALS = MC_TRIALS
    aux.MC_TRIALS = MC_TRIALS


def _eval_worker(item: tuple[int, dict[str, Any]]) -> dict[str, Any]:
    if _DF is None or _FEATURES is None:
        raise RuntimeError('worker not initialized')
    i, params = item
    name = f"integrated_{params['engine']}_{params['train_window']}_{params['feature_mode']}_edge{params['edge']}_{stable_hash(params)}"
    try:
        out = evaluate_params(_DF, _FEATURES, params, name)
        if out is None:
            return {'name': name, 'params': params, 'error': 'empty_or_fit_failed'}
        return out
    except Exception as exc:
        return {'name': name, 'params': params, 'error': str(exc)[:500]}


def write_progress(results: list[dict[str, Any]], base_by_window: dict[str, dict[str, Any]], total: int, started: float, data_truth: dict[str, Any], finished: bool = False):
    valid = [r for r in results if r.get('rows')]
    verdicts = []
    for c in valid:
        ok, reasons = pass_gate(c, base_by_window)
        verdicts.append({**c, 'passed': ok, 'reasons': reasons, 'score': score_candidate(c, base_by_window)})
    verdicts.sort(key=lambda x: x['score'], reverse=True)
    strict = [v for v in verdicts if v['passed']]
    payload = {
        'generatedAt': now_iso(),
        'beijingTime': bj_now(),
        'finished': finished,
        'elapsedSeconds': round(time.time() - started, 3),
        'workers': WORKERS,
        'workerThreads': THREADS,
        'totalCandidates': total,
        'doneCount': len(results),
        'validCount': len(valid),
        'strictPassCount': len(strict),
        'dataTruth': data_truth,
        'baseRows': base_by_window,
        'topCandidates': verdicts[:300],
        'strictPass': strict[:100],
    }
    write_json(OUT_CHECKPOINT, payload)
    write_json(OUT_LEADERBOARD_JSON, {'generatedAt': payload['generatedAt'], 'baseRows': base_by_window, 'rows': verdicts[:500]})
    selected = strict[0] if strict else None
    verdict = {'generatedAt': payload['generatedAt'], 'status': 'candidate_passed_research_gate' if selected else 'running_or_no_integrated_candidate_yet', 'selected': selected, 'baseRows': base_by_window, 'liveAction': 'research_only_no_live_change'}
    write_json(OUT_VERDICT_JSON, verdict)
    lines = ['# top159 主模型重训搜索', '', f'- 北京时间：`{payload["beijingTime"]}`', f'- 完成：`{len(results)}/{total}`', f'- 严格通过：`{len(strict)}`', '', '|候选|窗口|交易数|胜/负|胜率|P5|P50|P95|最大回撤|月正收益|FAK+2资金|模型|训练窗|特征|edge|哈希|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|']
    def add_row(r):
        lines.append(f"|{r['name']}|{r['window']}|{r['trades']}|{r['wins']}/{r['losses']}|{r['winRatePct']}|{r['endingBankrollP5']}|{r['endingBankrollP50']}|{r['endingBankrollP95']}|{r['maxDrawdownP50']}|{r['monthlyPositiveRatioP50']}|{r['fak2EndingBankroll']}|{r.get('engine','baseline')}|{r.get('trainWindow','-')}|{r.get('featureCount','-')}|{r.get('edge','-')}|`{r['setHash']}`|")
    for r in [base_by_window['180d'], base_by_window['365d']]:
        add_row(r)
    for c in verdicts[:25]:
        for r in c['rows']:
            add_row(r)
    write_text(OUT_LEADERBOARD_MD, '\n'.join(lines) + '\n')
    vlines = ['# top159 主模型重训唯一结论', '', f'- 北京时间：`{payload["beijingTime"]}`', f'- 状态：`{verdict["status"]}`', '- live动作：`research_only_no_live_change`']
    if selected:
        vlines += [f'- 选中候选：`{selected["name"]}`', f'- 参数：`{json.dumps(selected["params"], ensure_ascii=False)}`']
    else:
        vlines += ['- 暂无过门候选；搜索继续或需要进入下一轮特征/模型扩展。']
    write_text(OUT_VERDICT_MD, '\n'.join(vlines) + '\n')


def bug_audit(df: pd.DataFrame, features: list[str], base_by_window: dict[str, dict[str, Any]]) -> dict[str, Any]:
    # Deterministic baseline repeatability and shifted feature sanity.
    repeat = []
    params = {
        'engine': 'logistic', 'train_window': '5y', 'feature_mode': 'base_plus_1h4h',
        'edge': 0.045, 'vol_q': 0.999, 'max_features': 120, 'score_band': 'all',
        'max_score': 1.0, 'n_estimators': 1, 'learning_rate': 0.04, 'num_leaves': 16,
        'min_child_samples': 60, 'subsample': 0.9, 'colsample_bytree': 0.9,
        'reg_lambda': 1.0, 'depth': 3, 'C': 1.0,
    }
    c1 = evaluate_params(df, features, params, 'audit_repeat')
    c2 = evaluate_params(df, features, params, 'audit_repeat')
    if c1 and c2:
        for a, b in zip(c1['rows'], c2['rows']):
            repeat.append({'window': a['window'], 'hash1': a['setHash'], 'hash2': b['setHash'], 'passed': a['setHash'] == b['setHash'] and a['endingBankrollP50'] == b['endingBankrollP50']})
    forbidden_hits = [c for c in features if c in set(base.FORBIDDEN_FEATURES) or c.startswith(tuple(base.FORBIDDEN_PREFIXES))]
    random_label_audit: dict[str, Any] = {'status': 'not_run'}
    try:
        feats = feature_subset(features, params['feature_mode'])[: int(params['max_features'])]
        train, val = split_train_val(df, '365d', params['train_window'])
        model = fit_model(params['engine'], train, feats, params, random_labels=True)
        if model is not None:
            prob = predict(model, val, feats)
            selected = select_rows(val, prob, params)
            trades = int(len(selected))
            wr = float(selected['won'].mean() * 100.0) if trades else 0.0
            random_label_audit = {
                'status': 'ok',
                'window': '365d',
                'selectedTrades': trades,
                'winRatePct': round(wr, 6),
                'passed': trades < 50 or (43.0 <= wr <= 57.0),
            }
        else:
            random_label_audit = {'status': 'fit_failed', 'passed': False}
    except Exception as exc:
        random_label_audit = {'status': 'error', 'error': str(exc)[:300], 'passed': False}
    audit = {
        'generatedAt': now_iso(),
        'beijingTime': bj_now(),
        'futureFieldBlacklist': sorted(list(base.FORBIDDEN_FEATURES)) + list(base.FORBIDDEN_PREFIXES),
        'forbiddenFeatureHits': forbidden_hits,
        'repeatability': repeat,
        'repeatabilityPassed': all(x['passed'] for x in repeat) if repeat else False,
        'randomLabelAudit': random_label_audit,
        'windowIsolation': '180d and 365d train/validation splits and curve metrics are computed independently inside evaluate_params',
        'highTimeframeTiming': '1h/4h base features are shifted by one candle; daily features are timestamped after daily candle close and merged backward',
        'top159ScoreFeature': 'not used in v1 because safe OOF stacking is a separate follow-up; avoids leakage',
        'baseRows': base_by_window,
    }
    audit['passed'] = not forbidden_hits and audit['repeatabilityPassed'] and bool(random_label_audit.get('passed'))
    write_json(OUT_AUDIT_JSON, audit)
    return audit


def run() -> int:
    started = time.time()
    print(f'[integrated-main-extreme] start {bj_now()} workers={WORKERS} maxSeconds={MAX_SECONDS}', flush=True)
    df, features, data_truth = build_integrated_frame()
    base_by_window = base_rows()
    audit = bug_audit(df, features, base_by_window)
    print(f'[integrated-main-extreme] features={len(features)} rows={len(df)} auditPassed={audit["passed"]}', flush=True)
    params = param_grid()
    total = len(params)
    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    # Resume existing rows.
    done_names = set()
    results: list[dict[str, Any]] = []
    if OUT_RESULTS.exists():
        for line in OUT_RESULTS.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                done_names.add(row.get('name'))
                results.append(row)
            except Exception:
                pass
    pending = []
    for i, p in enumerate(params):
        name = f"integrated_{p['engine']}_{p['train_window']}_{p['feature_mode']}_edge{p['edge']}_{stable_hash(p)}"
        if name not in done_names:
            pending.append((i, p))
    print(f'[integrated-main-extreme] total={total} done={len(results)} pending={len(pending)}', flush=True)
    write_progress(results, base_by_window, total, started, {**data_truth, 'bugAudit': audit}, finished=False)
    last = time.time()
    with OUT_RESULTS.open('a', encoding='utf-8') as fh:
        with cf.ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_worker, initargs=(df, features)) as ex:
            futures = {}
            it = iter(pending)
            for _ in range(max(WORKERS * 2, 4)):
                try:
                    item = next(it)
                except StopIteration:
                    break
                futures[ex.submit(_eval_worker, item)] = item
            while futures:
                if time.time() - started > MAX_SECONDS:
                    break
                done, _ = cf.wait(futures, timeout=5, return_when=cf.FIRST_COMPLETED)
                for fut in done:
                    item = futures.pop(fut)
                    try:
                        row = fut.result()
                    except Exception as exc:
                        row = {'name': f'candidate_{item[0]}', 'params': item[1], 'error': str(exc)[:500]}
                    fh.write(json.dumps(row, ensure_ascii=False, default=str, separators=(',', ':')) + '\n')
                    fh.flush()
                    results.append(row)
                    try:
                        nxt = next(it)
                        futures[ex.submit(_eval_worker, nxt)] = nxt
                    except StopIteration:
                        pass
                if time.time() - last >= CHECKPOINT_SECONDS:
                    write_progress(results, base_by_window, total, started, {**data_truth, 'bugAudit': audit}, finished=False)
                    print(f'[integrated-main-extreme] checkpoint {len(results)}/{total}', flush=True)
                    last = time.time()
            for fut in futures:
                fut.cancel()
    finished = len([r for r in results if r.get('rows') or r.get('error')]) >= total
    write_progress(results, base_by_window, total, started, {**data_truth, 'bugAudit': audit}, finished=finished)
    print(json.dumps({'status': 'finished' if finished else 'checkpointed', 'done': len(results), 'total': total, 'leaderboard': str(OUT_LEADERBOARD_MD), 'verdict': str(OUT_VERDICT_MD)}, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(run())
