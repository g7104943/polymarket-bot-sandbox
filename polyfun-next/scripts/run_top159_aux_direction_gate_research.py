#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path('/Users/mac/polyfun')
NEXT = ROOT / 'polyfun-next'
REPORTS = ROOT / 'reports'
RAW = ROOT / 'data' / 'raw'
BASE_SCRIPT = ROOT / 'scripts' / 'ops' / 'run_crypto15m_1h_multasset_pressure_search_latest.py'
FILL_SCRIPT = ROOT / 'scripts' / 'ops' / 'run_newslot1_fill_rate_toxicity_search_latest.py'

RNG_SEED = 20260501
START_BANKROLL = 850.0
STAKE_PCT = 0.01
WINDOWS = {'180d': 180, '365d': 365}
TRAIN_DAYS = {'3y': 1095, '5y': 1825, 'full': None}
AUX_ENGINES = [x.strip() for x in os.environ.get('TOP159_AUX_ENGINES', 'lightgbm,catboost,xgboost').split(',') if x.strip()]
AUX_OPTUNA_TRIALS = int(os.environ.get('TOP159_AUX_OPTUNA_TRIALS', '80'))
MC_TRIALS = int(os.environ.get('TOP159_AUX_MC_TRIALS', '800'))
MAX_TRAIN_ROWS = int(os.environ.get('TOP159_AUX_MAX_TRAIN_ROWS', '180000'))


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'cannot import {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

base = _load_module('crypto_search_aux_gate', BASE_SCRIPT)
fill = _load_module('fill_search_aux_gate', FILL_SCRIPT)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(parts: list[Any]) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode('utf-8'))
        h.update(b'\n')
    return h.hexdigest()[:16]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def top159_params() -> dict[str, Any]:
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


def feature_subset(features: list[str], mode: str) -> list[str]:
    if mode == 'core':
        keep = ['ret_1','ret_2','ret_3','ret_4','ret_8','ret_16','vol_8','vol_16','vol_32','range_8','range_16','ema_8_32','ema_16_64','rsi_14','bb_pos','hour_sin','hour_cos','dow_sin','dow_cos']
    elif mode == 'trend':
        keep = ['ret_1','ret_2','ret_4','ret_8','ret_16','ret_32','ema_8_32','ema_16_64','ema_dist_8','ema_dist_16','ema_dist_32','ema_dist_64','vol_16','range_16','rsi_14','bb_pos']
    else:
        keep = features
    return [c for c in keep if c in features]


def fit_model(engine: str, train: pd.DataFrame, features: list[str], params: dict[str, Any]):
    if MAX_TRAIN_ROWS > 0 and len(train) > MAX_TRAIN_ROWS:
        train = train.sort_values('dt').iloc[-MAX_TRAIN_ROWS:].copy()
    x = train[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = train['label_up'].astype(int)
    if y.nunique() < 2 or len(train) < 200:
        return None
    if engine == 'lightgbm':
        import lightgbm as lgb
        model = lgb.LGBMClassifier(
            n_estimators=int(params.get('n_estimators', 140)), learning_rate=float(params.get('learning_rate', 0.035)),
            num_leaves=int(params.get('num_leaves', 24)), min_child_samples=int(params.get('min_child_samples', 50)),
            subsample=float(params.get('subsample', 0.9)), colsample_bytree=float(params.get('colsample_bytree', 0.9)),
            reg_lambda=float(params.get('reg_lambda', 0.5)), random_state=RNG_SEED, n_jobs=-1, verbose=-1,
        )
    elif engine == 'catboost':
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(
            iterations=int(params.get('n_estimators', 140)), depth=int(params.get('depth', 4)),
            learning_rate=float(params.get('learning_rate', 0.035)), l2_leaf_reg=float(params.get('reg_lambda', 1.0)),
            loss_function='Logloss', eval_metric='Logloss', random_seed=RNG_SEED, verbose=False, thread_count=-1,
        )
    elif engine == 'xgboost':
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=int(params.get('n_estimators', 140)), max_depth=int(params.get('depth', 4)),
            learning_rate=float(params.get('learning_rate', 0.035)), subsample=float(params.get('subsample', 0.9)),
            colsample_bytree=float(params.get('colsample_bytree', 0.9)), reg_lambda=float(params.get('reg_lambda', 1.0)),
            random_state=RNG_SEED, n_jobs=-1, eval_metric='logloss', verbosity=0,
        )
    else:
        raise ValueError(engine)
    model.fit(x, y)
    return model


def predict_prob(model: Any, df: pd.DataFrame, features: list[str]) -> np.ndarray:
    x = df[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


def ece_score(prob: np.ndarray, label: np.ndarray, bins: int = 10) -> float:
    if len(prob) == 0:
        return 0.0
    pred = prob >= 0.5
    conf = np.maximum(prob, 1.0 - prob)
    ok = pred == label.astype(bool)
    ece = 0.0
    for lo, hi in zip(np.linspace(0.5, 1.0, bins + 1)[:-1], np.linspace(0.5, 1.0, bins + 1)[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1.0 else conf <= hi)
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(float(np.mean(ok[mask])) - float(np.mean(conf[mask])))
    return round(ece, 6)


def load_raw(asset: str, timeframe: str) -> pd.DataFrame:
    return base.load_raw(asset, timeframe)


def resample_1d_from_1h() -> pd.DataFrame:
    raw = load_raw('ETH', '1h').copy()
    raw['dt'] = pd.to_datetime(raw['dt'], utc=True, errors='coerce')
    raw = raw.dropna(subset=['dt']).sort_values('dt')
    d = raw.set_index('dt').resample('1D', label='left', closed='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna().reset_index()
    d['date'] = d['dt']
    # Drop the possibly incomplete current UTC day.
    today = pd.Timestamp(datetime.now(timezone.utc)).floor('1D')
    d = d[d['dt'] < today].reset_index(drop=True)
    return d


def build_aux_frame(horizon: str) -> tuple[pd.DataFrame, list[str], pd.Timedelta]:
    if horizon == '4h':
        raw = load_raw('ETH', '4h')
        df, features = base.build_features(raw, '4h')
        tol = pd.Timedelta(hours=4)
    elif horizon == '1d':
        raw = resample_1d_from_1h()
        df, features = base.build_features(raw, '1d')
        tol = pd.Timedelta(days=1)
    elif horizon == '7d':
        raw = resample_1d_from_1h()
        df, features = base.build_features(raw, '1d')
        df = df.sort_values('dt').copy()
        df['label_up'] = (df['close'].shift(-6) > df['open']).astype(float)
        df = df.dropna(subset=['label_up']).copy()
        df['label_up'] = df['label_up'].astype(int)
        tol = pd.Timedelta(days=1)
    else:
        raise ValueError(horizon)
    forbidden = [c for c in features if c in base.FORBIDDEN_FEATURES or any(c.startswith(p) for p in base.FORBIDDEN_PREFIXES)]
    if forbidden:
        raise RuntimeError(f'forbidden future features in aux frame {horizon}: {forbidden}')
    return df.sort_values('dt').reset_index(drop=True), features, tol


def split_train_val(df: pd.DataFrame, window: str, train_window: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    end = df['dt'].max()
    start = end - pd.Timedelta(days=WINDOWS[window])
    val = df[df['dt'] >= start].copy()
    if train_window == 'full':
        train = df[df['dt'] < start].copy()
    else:
        train = df[(df['dt'] < start) & (df['dt'] >= start - pd.Timedelta(days=TRAIN_DAYS[train_window]))].copy()
    return train, val


def aux_param_grid(horizon: str) -> list[dict[str, Any]]:
    engines = [e for e in AUX_ENGINES if e in {'lightgbm','catboost','xgboost'}]
    out = []
    for engine in engines:
        for train_window in (['3y','5y','full'] if horizon != '7d' else ['5y','full']):
            for feature_mode in ['core','trend','wide']:
                out.append({
                    'engine': engine, 'train_window': train_window, 'feature_mode': feature_mode,
                    'n_estimators': 140, 'learning_rate': 0.035, 'reg_lambda': 0.8,
                    'subsample': 0.9, 'colsample_bytree': 0.9, 'num_leaves': 24,
                    'min_child_samples': 50, 'depth': 4,
                })
    return out


def train_best_aux(horizon: str, window: str, frame: pd.DataFrame, features: list[str]) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    best = None
    best_score = -1e9
    best_pred = None
    audits = []
    for params in aux_param_grid(horizon):
        feats = feature_subset(features, params['feature_mode'])
        train, val = split_train_val(frame, window, params['train_window'])
        if len(feats) < 6 or len(train) < 300 or len(val) < 40:
            continue
        try:
            model = fit_model(params['engine'], train, feats, params)
            if model is None:
                continue
            prob = predict_prob(model, val, feats)
        except Exception as exc:
            audits.append({'horizon': horizon, 'window': window, 'params': params, 'error': str(exc)[:200]})
            continue
        label = val['label_up'].astype(bool).to_numpy()
        pred = prob >= 0.5
        acc = float(np.mean(pred == label))
        conf = float(np.mean(np.maximum(prob, 1.0 - prob)))
        ece = ece_score(prob, label)
        # Prefer true accuracy, lightly penalize poor calibration and overconfidence.
        score = acc - 0.25 * ece - 0.05 * max(0.0, conf - 0.62)
        audit = {
            'horizon': horizon, 'window': window, 'params': params, 'trainRows': int(len(train)), 'validationRows': int(len(val)),
            'featureCount': int(len(feats)), 'accuracyPct': round(acc * 100, 6), 'avgConfidence': round(conf, 6), 'ece': ece,
        }
        audits.append(audit)
        if score > best_score:
            best_score = score
            best = dict(audit)
            best_pred = val[['dt', 'label_up']].copy()
            best_pred['prob_up'] = prob
            best_pred['pred_up'] = pred
    if best is None or best_pred is None:
        raise RuntimeError(f'no aux model for {horizon} {window}')
    best['searchCandidates'] = len(audits)
    best['candidateAuditsSample'] = [
        dict(x) for x in sorted(audits, key=lambda x: x.get('accuracyPct', 0), reverse=True)[:8]
    ]
    return best, best_pred.sort_values('dt').reset_index(drop=True), {'audits': audits}


def top159_candidates(window: str, params: dict[str, Any], *, start_override: pd.Timestamp | None = None, end_override: pd.Timestamp | None = None) -> pd.DataFrame:
    raw = load_raw('ETH', '15m')
    df, features = base.build_features(raw, '15m')
    end = df['dt'].max() if end_override is None else pd.Timestamp(end_override)
    start = end - pd.Timedelta(days=WINDOWS[window]) if start_override is None else pd.Timestamp(start_override)
    val = df[(df['dt'] >= start) & (df['dt'] <= end)].copy()
    train_start = start - pd.Timedelta(days=1825 if params.get('train_window') == '5y' else fill.TRAIN_DAYS.get(params.get('train_window'), 1825) or 3650)
    train = df[(df['dt'] < start) & (df['dt'] >= train_start)].copy()
    if len(train) > fill.MAX_TRAIN_ROWS:
        train = train.sort_values('dt').iloc[-fill.MAX_TRAIN_ROWS:].copy()
    feats = fill.feature_subset(features, params['feature_mode'])
    model = fill.fit_model(params['engine'], train, feats, params)
    if model is None:
        raise RuntimeError('top159 model failed')
    prob = fill.predict(model, val, feats)
    conf = np.maximum(prob, 1.0 - prob)
    pred_up = prob >= 0.5
    mask = conf >= 0.5 + params['edge']
    if params.get('vol_q', 0.999) < 0.999 and 'vol_16' in val.columns:
        v = pd.to_numeric(val['vol_16'], errors='coerce')
        mask &= v <= float(v.quantile(params['vol_q']))
    if params.get('bb_abs_max', 9.0) < 9 and 'bb_pos' in val.columns:
        mask &= pd.to_numeric(val['bb_pos'], errors='coerce').abs() <= params['bb_abs_max']
    if params.get('trend_mode') != 'none' and 'ema_8_32' in val.columns:
        trend_up = pd.to_numeric(val['ema_8_32'], errors='coerce').fillna(0.0).to_numpy() >= 0
        if params['trend_mode'] == 'agree':
            mask &= pred_up == trend_up
    label_up = val['label_up'].astype(bool).to_numpy()
    won_pre = (pred_up[mask] == label_up[mask]).astype(bool)
    keep = fill.state_filter(won_pre, int(params.get('loss_n', 0)), int(params.get('skip_k', 0)))
    selected = val.loc[mask].reset_index(drop=True).loc[keep].reset_index(drop=True)
    out = pd.DataFrame({
        'dt': pd.to_datetime(selected['dt'], utc=True),
        'prob_up15': prob[mask][keep],
        'score15': conf[mask][keep],
        'pred_up15': pred_up[mask][keep].astype(bool),
        'label_up': selected['label_up'].astype(bool).to_numpy(),
    })
    out['won'] = out['pred_up15'].to_numpy() == out['label_up'].to_numpy()
    out['side'] = np.where(out['pred_up15'], 'UP', 'DOWN')
    out['market_slug'] = out['dt'].map(lambda x: f"eth-updown-15m-{int(pd.Timestamp(x).timestamp())}")
    return out.sort_values('dt').reset_index(drop=True)


def attach_aux(cands: pd.DataFrame, pred: pd.DataFrame, horizon: str, tolerance: pd.Timedelta) -> pd.DataFrame:
    left = cands.sort_values('dt').copy()
    right = pred[['dt','prob_up']].sort_values('dt').copy().rename(columns={'prob_up': f'p_{horizon}_up'})
    merged = pd.merge_asof(left, right, on='dt', direction='backward', tolerance=tolerance)
    p = pd.to_numeric(merged[f'p_{horizon}_up'], errors='coerce')
    merged[f'{horizon}_available'] = p.notna().astype(int)
    merged[f'p_{horizon}_same'] = np.where(merged['pred_up15'], p, 1.0 - p)
    merged[f'p_{horizon}_same'] = merged[f'p_{horizon}_same'].fillna(0.5)
    return merged


def add_all_aux(cands: pd.DataFrame, preds: dict[str, tuple[pd.DataFrame, pd.Timedelta]]) -> pd.DataFrame:
    out = cands.copy()
    for h, (pred, tol) in preds.items():
        out = attach_aux(out, pred, h, tol)
    return out


def curve_metrics(df: pd.DataFrame, name: str, window: str, method: str) -> dict[str, Any]:
    won = df['won'].astype(bool).to_numpy()
    dt = pd.to_datetime(df['dt'], utc=True)
    returns50 = base.returns_from_wins(won, 0.50)
    toxic_fill = np.where(won, base.SLOT1_WIN_FILL, base.SLOT1_LOSS_FILL)
    toxic = base.summarize_curve(dt.reset_index(drop=True), won, returns50, toxic_fill, method + '_slot1_toxic', 'ETH', '15m', window, name, {})
    old_trials = base.TRIALS
    base.TRIALS = MC_TRIALS
    try:
        mc = base.monte_carlo_slot1(dt.reset_index(drop=True), won, returns50, RNG_SEED + (180 if window == '180d' else 365))
    finally:
        base.TRIALS = old_trials
    fak2 = base.summarize_curve(dt.reset_index(drop=True), won, base.returns_from_wins(won, 0.52), np.ones(len(won)), method + '_fak_ask_plus_2c', 'ETH', '15m', window, name, {})
    return {
        'name': name, 'window': window, 'method': method,
        'trades': int(len(df)), 'wins': int(won.sum()), 'losses': int((~won).sum()),
        'winRatePct': round(100.0 * float(won.mean()), 6) if len(won) else 0.0,
        'endingBankrollToxicDeterministic': toxic['endingBankroll'],
        'endingBankrollP5': mc['endingBankrollP5'],
        'endingBankrollP50': mc['endingBankrollP50'],
        'endingBankrollP95': mc['endingBankrollP95'],
        'maxDrawdownP50': mc['maxDrawdownP50'],
        'maxDrawdownP95': mc['maxDrawdownP95'],
        'monthlyPositiveRatioP50': mc['monthlyPositiveRatioP50'],
        'fak2EndingBankroll': fak2['endingBankroll'],
        'fak2MaxDrawdown': fak2['maxDrawdownUsd'],
        'winnerFillRatePct': round(base.SLOT1_WIN_FILL * 100, 6),
        'loserFillRatePct': round(base.SLOT1_LOSS_FILL * 100, 6),
        'setHash': stable_hash([name, window, method, len(df), int(won.sum()), round(float(mc['endingBankrollP50']), 6)]),
    }


def weighted_filter(df: pd.DataFrame, weights: dict[str, float], amp: float, neg_clip: float, pos_clip: float, threshold: float) -> pd.DataFrame:
    denom = 0.0
    support = np.zeros(len(df), dtype=float)
    for h, w in weights.items():
        if w <= 0:
            continue
        avail = df.get(f'{h}_available', pd.Series(np.ones(len(df)))).to_numpy(dtype=float)
        same = df[f'p_{h}_same'].to_numpy(dtype=float)
        support += w * avail * (2.0 * same - 1.0)
        denom += w
    if denom <= 0:
        final = df['score15'].to_numpy(dtype=float)
    else:
        support = support / max(denom, 1e-12)
        adj = np.clip(amp * support, -neg_clip, pos_clip)
        final = df['score15'].to_numpy(dtype=float) + adj
    out = df.copy()
    out['aux_support'] = support if denom > 0 else 0.0
    out['final_score'] = final
    return out[out['final_score'] >= threshold].copy()


def hard_veto_filter(df: pd.DataFrame, oppose_threshold: float = 0.58, min_oppose: int = 2) -> pd.DataFrame:
    oppose = np.zeros(len(df), dtype=int)
    for h in ['4h','1d','7d']:
        same = df[f'p_{h}_same'].to_numpy(dtype=float)
        avail = df.get(f'{h}_available', pd.Series(np.ones(len(df)))).to_numpy(dtype=bool)
        oppose += ((1.0 - same) >= oppose_threshold) & avail
    out = df.copy()
    out['oppose_count'] = oppose
    return out[out['oppose_count'] < min_oppose].copy()


def evaluate_variant(base_df: pd.DataFrame, kept: pd.DataFrame, name: str, window: str, method: str) -> dict[str, Any]:
    m = curve_metrics(kept, name, window, method)
    base_idx = set(base_df.index)
    keep_idx = set(kept.index)
    blocked = base_df.loc[list(base_idx - keep_idx)] if base_idx - keep_idx else base_df.iloc[0:0]
    m.update({
        'baseTrades': int(len(base_df)),
        'blockedTrades': int(len(blocked)),
        'blockedWinners': int(blocked['won'].sum()) if len(blocked) else 0,
        'blockedLosers': int((~blocked['won']).sum()) if len(blocked) else 0,
        'retentionPct': round(100.0 * len(kept) / len(base_df), 6) if len(base_df) else 0.0,
        'blockedSetHash': stable_hash(list(blocked['dt'].astype(str).head(500)) + [len(blocked), int(blocked['won'].sum())]) if len(blocked) else 'empty',
    })
    return m


def optimize_weighted(windows_df: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    weight_sets = [
        {'4h':0.45,'1d':0.35,'7d':0.20}, {'4h':0.60,'1d':0.30,'7d':0.10},
        {'4h':0.35,'1d':0.50,'7d':0.15}, {'4h':0.50,'1d':0.00,'7d':0.50},
        {'4h':1.00,'1d':0.00,'7d':0.00}, {'4h':0.00,'1d':1.00,'7d':0.00},
    ]
    grid = []
    for weights in weight_sets:
        for amp in [0.02,0.04,0.06,0.08]:
            for neg in [0.03,0.05,0.08]:
                for pos in [0.00,0.02,0.03,0.05]:
                    for th in [0.545,0.55,0.56,0.57]:
                        grid.append((weights,amp,neg,pos,th))
    for i, (weights, amp, neg, pos, th) in enumerate(grid):
        rows = []
        valid = True
        for window, df in windows_df.items():
            kept = weighted_filter(df, weights, amp, neg, pos, th)
            if len(kept) == 0:
                valid = False; break
            rows.append(evaluate_variant(df, kept, f'weighted_{i}', window, 'weighted_aux_gate'))
        if not valid:
            continue
        score = rows[0]['endingBankrollP50'] + rows[1]['endingBankrollP50'] - rows[1]['maxDrawdownP50'] * 1.5
        candidates.append({'name': f'weighted_{i}', 'kind': 'weighted', 'params': {'weights':weights,'amp':amp,'negClip':neg,'posClip':pos,'threshold':th}, 'rows': rows, 'score': score})
    try:
        import optuna

        def objective(trial):
            raw_weights = {
                '4h': trial.suggest_float('w4h', 0.0, 1.0),
                '1d': trial.suggest_float('w1d', 0.0, 1.0),
                '7d': trial.suggest_float('w7d', 0.0, 1.0),
            }
            s = sum(raw_weights.values()) or 1.0
            weights = {k: v / s for k, v in raw_weights.items()}
            amp = trial.suggest_float('amp', 0.005, 0.10)
            neg = trial.suggest_float('neg_clip', 0.015, 0.10)
            pos = trial.suggest_float('pos_clip', 0.0, 0.06)
            th = trial.suggest_float('threshold', 0.545, 0.585)
            rows = []
            for window, df in windows_df.items():
                kept = weighted_filter(df, weights, amp, neg, pos, th)
                if len(kept) == 0:
                    return -1e9, -1e9, 1e9, -1e9
                rows.append(evaluate_variant(df, kept, f'weighted_optuna_trial_{trial.number}', window, 'weighted_aux_gate_optuna'))
            r180, r365 = rows
            toxicity_hit = (r180['blockedLosers'] - r180['blockedWinners']) + (r365['blockedLosers'] - r365['blockedWinners'])
            return (
                float(r365['endingBankrollP50']),
                float(r180['endingBankrollP50']),
                float(r365['maxDrawdownP50']),
                float(toxicity_hit),
            )

        sampler = optuna.samplers.NSGAIISampler(seed=RNG_SEED)
        study = optuna.create_study(directions=['maximize', 'maximize', 'minimize', 'maximize'], sampler=sampler)
        study.optimize(objective, n_trials=AUX_OPTUNA_TRIALS, show_progress_bar=False, gc_after_trial=True)
        for t in study.best_trials[:20]:
            raw_weights = {'4h': t.params.get('w4h', 0), '1d': t.params.get('w1d', 0), '7d': t.params.get('w7d', 0)}
            s = sum(raw_weights.values()) or 1.0
            weights = {k: v / s for k, v in raw_weights.items()}
            params = {
                'weights': weights,
                'amp': t.params.get('amp'),
                'negClip': t.params.get('neg_clip'),
                'posClip': t.params.get('pos_clip'),
                'threshold': t.params.get('threshold'),
                'optunaTrial': t.number,
                'optunaValues': t.values,
            }
            rows = []
            for window, df in windows_df.items():
                kept = weighted_filter(df, weights, params['amp'], params['negClip'], params['posClip'], params['threshold'])
                if len(kept) == 0:
                    rows = []
                    break
                rows.append(evaluate_variant(df, kept, f'weighted_optuna_{t.number}', window, 'weighted_aux_gate_optuna'))
            if rows:
                score = rows[0]['endingBankrollP50'] + rows[1]['endingBankrollP50'] - rows[1]['maxDrawdownP50'] * 1.5
                candidates.append({'name': f'weighted_optuna_{t.number}', 'kind': 'weighted_optuna', 'params': params, 'rows': rows, 'score': score})
    except Exception as exc:
        candidates.append({'name': 'weighted_optuna_unavailable', 'kind': 'diagnostic', 'params': {'error': str(exc)[:240]}, 'rows': [], 'score': -1e18})
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return [c for c in candidates if c.get('rows')][:20]


def meta_filter(train_df: pd.DataFrame, val_df: pd.DataFrame, threshold: float, engine: str = 'lightgbm') -> tuple[pd.DataFrame, dict[str, Any]]:
    feats = ['score15','p_4h_same','p_1d_same','p_7d_same','4h_available','1d_available','7d_available']
    train = train_df.dropna(subset=feats + ['won']).copy()
    if len(train) < 200 or train['won'].nunique() < 2:
        return val_df.iloc[0:0].copy(), {'status':'insufficient_meta_train', 'trainRows': len(train)}
    params = {'engine': engine, 'n_estimators': 120, 'learning_rate': 0.04, 'num_leaves': 16, 'min_child_samples': 40, 'depth': 3, 'reg_lambda': 1.0, 'subsample': 0.9, 'colsample_bytree': 0.9}
    model = fit_model(engine, train.rename(columns={'won':'label_up'}), feats, params)
    if model is None:
        return val_df.iloc[0:0].copy(), {'status':'meta_fit_failed'}
    prob = predict_prob(model, val_df, feats)
    out = val_df.copy(); out['meta_prob'] = prob
    return out[out['meta_prob'] >= threshold].copy(), {'status':'ok','trainRows':len(train),'threshold':threshold,'engine':engine}


def build_meta_train(val_start: pd.Timestamp, top_params: dict[str, Any], aux_models_by_horizon: dict[str, tuple[dict[str, Any], pd.DataFrame, pd.Timedelta]]) -> pd.DataFrame:
    start = val_start - pd.Timedelta(days=365)
    end = val_start - pd.Timedelta(minutes=15)
    cands = top159_candidates('365d', top_params, start_override=start, end_override=end)
    preds = {h: (pred, tol) for h, (_, pred, tol) in aux_models_by_horizon.items()}
    return add_all_aux(cands, preds)


def status_pass(row180: dict[str, Any], row365: dict[str, Any], base180: dict[str, Any], base365: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons = []
    if row180['trades'] < 100 or row365['trades'] < 200:
        reasons.append('trade_count_too_low')
    if row365['retentionPct'] < 45:
        reasons.append('retention_below_45pct')
    if row180['fak2EndingBankroll'] <= 850 or row365['fak2EndingBankroll'] <= 850:
        reasons.append('fak2_not_profitable')
    if row180['endingBankrollP50'] < base180['endingBankrollP50'] * 0.995:
        reasons.append('180d_toxic_p50_worse')
    if row365['endingBankrollP50'] < base365['endingBankrollP50'] * 0.995:
        reasons.append('365d_toxic_p50_worse')
    if row180['blockedLosers'] <= row180['blockedWinners'] and row365['blockedLosers'] <= row365['blockedWinners']:
        reasons.append('blocked_losers_not_more_than_winners')
    return len(reasons) == 0, reasons


def main() -> int:
    generated_at = now_iso()
    top_params = top159_params()
    frames: dict[str, tuple[pd.DataFrame, list[str], pd.Timedelta]] = {h: build_aux_frame(h) for h in ['4h','1d','7d']}

    aux_truth: dict[str, Any] = {'generatedAt': generated_at, 'models': {}, 'bugcheck': {}}
    aux_predictions: dict[str, dict[str, tuple[pd.DataFrame, pd.Timedelta]]] = {'180d': {}, '365d': {}}
    aux_weight_allowed: dict[str, bool] = {}
    for h, (frame, features, tol) in frames.items():
        aux_truth['models'][h] = {}
        pass_all = True
        for window in ['180d','365d']:
            best, pred, audit = train_best_aux(h, window, frame, features)
            aux_truth['models'][h][window] = best
            aux_predictions[window][h] = (pred, tol)
            if best['accuracyPct'] <= 50.0:
                pass_all = False
        aux_weight_allowed[h] = pass_all
    aux_truth['allowedForLiveWeight'] = aux_weight_allowed

    windows_df: dict[str, pd.DataFrame] = {}
    base_rows: dict[str, dict[str, Any]] = {}
    for window in ['180d','365d']:
        cands = top159_candidates(window, top_params)
        cands = add_all_aux(cands, aux_predictions[window])
        # Zero out failed aux periods by replacing same-prob with neutral and availability=0.
        for h, allowed in aux_weight_allowed.items():
            if not allowed:
                cands[f'{h}_available'] = 0
                cands[f'p_{h}_same'] = 0.5
        windows_df[window] = cands
        base_rows[window] = curve_metrics(cands, 'current_top159', window, 'baseline')
        base_rows[window].update({'baseTrades': len(cands), 'blockedTrades': 0, 'blockedWinners': 0, 'blockedLosers': 0, 'retentionPct': 100.0})

    weighted = optimize_weighted(windows_df)

    hard_rows = []
    for window, df in windows_df.items():
        kept = hard_veto_filter(df)
        hard_rows.append(evaluate_variant(df, kept, 'hard_veto_two_aux_oppose_058', window, 'hard_veto_aux_gate'))
    hard = {'name':'hard_veto_two_aux_oppose_058','kind':'hard_veto','params':{'opposeThreshold':0.58,'minOppose':2},'rows':hard_rows,'score':hard_rows[0]['endingBankrollP50']+hard_rows[1]['endingBankrollP50']}

    # Meta model: train on the year before each validation window using pre-window labels only.
    meta_candidates = []
    for engine in ['lightgbm','catboost','xgboost']:
        for th in [0.50,0.52,0.54,0.56,0.58]:
            rows=[]; meta_audits=[]; ok=True
            for window, df in windows_df.items():
                val_start = pd.to_datetime(df['dt'], utc=True).min()
                aux_bundle = {h: (aux_truth['models'][h][window], aux_predictions[window][h][0], aux_predictions[window][h][1]) for h in ['4h','1d','7d']}
                try:
                    train_df = build_meta_train(val_start, top_params, aux_bundle)
                    for h, allowed in aux_weight_allowed.items():
                        if not allowed:
                            train_df[f'{h}_available']=0; train_df[f'p_{h}_same']=0.5
                    kept, ma = meta_filter(train_df, df, th, engine)
                except Exception as exc:
                    ok=False; meta_audits.append({'window':window,'error':str(exc)[:240]}); break
                meta_audits.append({'window':window, **ma})
                if len(kept)==0:
                    ok=False; break
                rows.append(evaluate_variant(df, kept, f'meta_{engine}_{th}', window, 'meta_aux_gate'))
            if ok:
                meta_candidates.append({'name':f'meta_{engine}_{th}','kind':'meta','params':{'engine':engine,'threshold':th},'rows':rows,'metaAudits':meta_audits,'score':rows[0]['endingBankrollP50']+rows[1]['endingBankrollP50']-rows[1]['maxDrawdownP50']})
    meta_candidates.sort(key=lambda x: x['score'], reverse=True)

    all_candidates = weighted + [hard] + meta_candidates[:10]
    all_candidates.sort(key=lambda x: x['score'], reverse=True)

    compare_rows = [base_rows['180d'], base_rows['365d']]
    for cand in all_candidates[:12]:
        compare_rows.extend(cand['rows'])

    verdicts = []
    for cand in all_candidates:
        by_window = {r['window']: r for r in cand['rows']}
        passed, reasons = status_pass(by_window['180d'], by_window['365d'], base_rows['180d'], base_rows['365d'])
        verdicts.append({'name':cand['name'],'kind':cand['kind'],'params':cand['params'],'passed':passed,'reasons':reasons,'rows':cand['rows']})
    passed = [v for v in verdicts if v['passed']]
    if passed:
        selected = sorted(passed, key=lambda v: (v['rows'][1]['endingBankrollP50'], -v['rows'][1]['maxDrawdownP50'], v['rows'][1]['blockedLosers']-v['rows'][1]['blockedWinners']), reverse=True)[0]
        status = 'candidate_passed_research_gate'
        live_action = 'prepare_disabled_live_config_for_preflight_only'
    else:
        selected = None
        status = 'no_aux_gate_passed_keep_current_top159'
        live_action = 'do_not_enable_live_aux_gate'

    bugcheck = {
        'futureFieldBlacklist': sorted(list(base.FORBIDDEN_FEATURES)) + list(base.FORBIDDEN_PREFIXES),
        'featuresShifted': 'base build_features shifts price-derived features before each bar',
        'windowIsolation': '180d and 365d trained/evaluated independently',
        'auxAlignment': 'merge_asof backward only; 4h tolerance=4h, 1d/7d tolerance=1d',
        'randomLabelTest': 'not run in this implementation pass; must be added before live enable',
        'liveMutation': 'none; this script writes reports only',
        'optuna': f'weighted gate uses deterministic grid plus Optuna NSGA-II multi-objective search; trials={AUX_OPTUNA_TRIALS}',
    }

    payload = {
        'generatedAt': generated_at,
        'status': status,
        'liveAction': live_action,
        'top159Params': top_params,
        'auxModelTruth': aux_truth,
        'compareRows': compare_rows,
        'candidateVerdicts': verdicts[:20],
        'selected': selected,
        'bugcheck': bugcheck,
    }

    write_json(REPORTS / 'top159_aux_direction_gate_model_truth_latest.json', aux_truth)
    write_json(REPORTS / 'top159_aux_direction_gate_180_365_compare_latest.json', {'generatedAt': generated_at, 'rows': compare_rows})
    write_json(REPORTS / 'top159_aux_direction_gate_unique_verdict_latest.json', {'generatedAt': generated_at, 'status': status, 'selected': selected, 'liveAction': live_action})
    write_json(REPORTS / 'top159_aux_direction_gate_bugcheck_latest.json', bugcheck)
    write_json(REPORTS / 'top159_aux_direction_gate_research_latest.json', payload)

    lines = ['# top159 多周期方向判断门研究', '', f"生成时间：`{generated_at}`", '', '## 结论', f"- 状态：`{status}`", f"- live动作：`{live_action}`", '']
    lines += ['## 辅助模型单独准确率', '|周期|窗口|引擎|训练窗|特征|样本|准确率|校准误差|是否允许入组合|', '|---|---|---|---|---:|---:|---:|---:|---|']
    for h in ['4h','1d','7d']:
        for w in ['180d','365d']:
            m = aux_truth['models'][h][w]
            lines.append(f"|{h}|{w}|{m['params']['engine']}|{m['params']['train_window']}|{m['featureCount']}|{m['validationRows']}|{m['accuracyPct']}|{m['ece']}|{aux_weight_allowed[h]}|")
    lines += ['', '## 绝对对比表', '|方法|窗口|交易数|胜/负|胜率|P5资金|P50资金|P95资金|最大回撤P50|月正收益|FAK+2分资金|拦截单|拦截赢家|拦截输家|保留率|哈希|', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in compare_rows:
        lines.append(f"|{r['name']} / {r['method']}|{r['window']}|{r['trades']}|{r['wins']}/{r['losses']}|{r['winRatePct']}|{r['endingBankrollP5']}|{r['endingBankrollP50']}|{r['endingBankrollP95']}|{r['maxDrawdownP50']}|{r['monthlyPositiveRatioP50']}|{r['fak2EndingBankroll']}|{r.get('blockedTrades',0)}|{r.get('blockedWinners',0)}|{r.get('blockedLosers',0)}|{r.get('retentionPct',100)}|`{r['setHash']}`|")
    if selected:
        lines += ['', '## 选中候选', f"- `{selected['name']}`", f"- 参数：`{json.dumps(selected['params'], ensure_ascii=False)}`"]
    else:
        lines += ['', '## 唯一结论', '- 没有候选同时通过 180天、365天、FAK+2分、slot1毒性和交易数门槛。', '- 不接入 live；当前 top159 继续原样运行。']
    write_text(REPORTS / 'top159_aux_direction_gate_research_latest.md', '\n'.join(lines) + '\n')
    print(json.dumps({'status': status, 'selected': selected['name'] if selected else None, 'report': str(REPORTS / 'top159_aux_direction_gate_research_latest.md')}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
